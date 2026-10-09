"""One residual distribution prices both a reading event and its response."""

from math import isclose

import pytest

from hegel_machine.cognition.evaluation import predict
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_experiments import (
    _confirmation_cdf, _sampling_question_context,
    expected_confirmation_contrast, plan_observed_condition_probe,
)
from test_cognition_revision_confirmation import _rows, paid_cases
from test_cognition_observed_condition_uncertainty import response


UNIFORM_RESIDUALS = [-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.]


def _contrast(new, old, intervals, *, residuals=UNIFORM_RESIDUALS, length=24):
    hypotheses = [{"program": new, "fallback": 0.}, {"program": old, "fallback": 0.}]
    budget = SearchBudget(50000)
    result = expected_confirmation_contrast(hypotheses,
        [{"x": 0., "meter": 0.} for _ in range(length)], "meter",
        [{"condition_field": "meter", "threshold_interval": list(interval)} for interval in intervals],
        list(residuals), 0., budget)
    assert result is not None and 0 < budget.spent <= budget.limit
    return result


@pytest.mark.parametrize("half_width", [.02, .04])
def test_narrow_strip_response_expectation_uses_its_uniform_probability(half_width):
    # A linear empirical CDF on [-1, 1] gives this strip probability h.
    # The paid reading at zero is one observed hit, not one certain future hit.
    new = response(half_width, width=1, difference=3.)
    old = response(-half_width, width=1, difference=3.)
    result = _contrast(new, old, [(-half_width, half_width)])
    exposure, = result["interval_exposure"]
    assert exposure["estimated_probability"] == pytest.approx(half_width)
    assert result["expected_disagreement"] == pytest.approx(9. * half_width)
    reversed_result = _contrast(new, old, [(-half_width, half_width)],
        residuals=list(reversed(UNIFORM_RESIDUALS)))
    assert reversed_result["expected_disagreement"] == pytest.approx(result["expected_disagreement"])


def test_no_residual_mass_in_the_changed_region_means_no_expected_contrast():
    result = _contrast(response(3., width=1, difference=3.),
        response(2., width=1, difference=3.), [(2., 3.)])
    assert result["interval_exposure"][0]["estimated_probability"] == 0.
    assert result["expected_disagreement"] == 0.


def test_repeated_paid_readings_preserve_their_atom_in_the_common_distribution():
    # Three equal-weight CDF pieces: uniform[-1,.05], atom(.05), uniform[.05,1].
    # The atom lies strictly inside the changed strip, avoiding endpoint rules.
    probability = (1. + .05 / 1.05 + .05 / .95) / 3.
    result = _contrast(response(.1, width=1, difference=3.),
        response(0., width=1, difference=3.), [(0., .1)], residuals=[-1., .05, .05, 1.])
    assert result["interval_exposure"][0]["estimated_probability"] == pytest.approx(probability)
    assert result["expected_disagreement"] == pytest.approx(9. * probability)


@pytest.mark.parametrize("interval,probability", [((0., .1), 1.), ((-.1, 0.), 0.)],
                         ids=["at_lower_ge_cut", "at_upper_ge_cut"])
def test_deterministic_instrument_atom_obeys_the_frozen_laws_half_open_cut(interval, probability):
    # A noiseless instrument at fixed controls has a point mass. ge changes
    # its branch at the lower cut, while both laws agree at the upper cut.
    left, right = interval
    new, old = response(right, width=1, difference=3.), response(left, width=1, difference=3.)
    result = _contrast(new, old, [interval], residuals=[0.] * 8)
    observed = Observation({"x": 0., "meter": 0.}, 0.)
    actual_gap = (new.predict(observed) - old.predict(observed)) ** 2
    assert actual_gap == 9. * probability
    assert result["interval_exposure"][0]["estimated_probability"] == pytest.approx(probability)
    assert result["expected_disagreement"] == pytest.approx(actual_gap)


def _two_strips(second_sign):
    meter = Program("feature", name="meter")
    def at_least(value):
        return Program("ge", children=(meter, Program("const", value=value)))
    zero = Program("const", value=0.)
    return Program("if", children=(at_least(-.5),
        Program("if", children=(at_least(0.),
            Program("if", children=(at_least(.5), zero, Program("const", value=second_sign))),
            Program("const", value=1.))), zero))


@pytest.mark.parametrize("second_sign,expected", [(-1., 1.4375), (1., 17. / 6.)],
                         ids=["opposite_signed_changes", "same_signed_changes"])
def test_window_moments_keep_same_reading_events_exclusive_and_cross_time_terms_signed(
        second_sign, expected):
    # Uniform reading: P(left strip)=P(right strip)=1/4 and they are exclusive.
    # Opposite signs give E[f]=0, E[f²]=1/2; equal signs give both moments=1/2.
    # Sum over real startup widths [1,2,3,...,3]: E[S²]=w Var(f)+w² E[f]².
    program = Program("window_sum", lag=3, children=(_two_strips(second_sign),))
    result = _contrast(program, Program("const", value=0.), [(-.5, 0.), (0., .5)])
    assert [item["estimated_probability"] for item in result["interval_exposure"]] == pytest.approx([.25, .25])
    assert result["expected_disagreement"] == pytest.approx(expected)


def test_saved_pair_sampling_context_matches_exposure_at_a_shared_atomic_boundary():
    residuals, intervals = [-1., 0., 0., 1.], [(-.5, 0.), (0., .5)]
    result = _contrast(_two_strips(-1.), Program("const", value=0.),
                       intervals, residuals=residuals)
    context = _sampling_question_context(
        {"basis": "paid_local_control_residuals", "normalized_background": [.5],
         "neighbourhood_radius": 1., "applicability_radius": 1.},
        {"reading_scale": 1., "paid_rows": 4, "control_sensitivity": [1.]},
        residuals, 0., intervals, SearchBudget(1000), cdf=_confirmation_cdf)
    cdf = context["reading_forecast"]["question_cdf"]
    # The atom at zero belongs to [0,.5), never to [-.5,0). The saved scope
    # must describe exactly the same probabilities as the priced experiment.
    assert [right - left for left, right in cdf] == pytest.approx([1. / 6., .5])
    assert [right - left for left, right in cdf] == pytest.approx(
        [entry["estimated_probability"] for entry in result["interval_exposure"]])


@pytest.mark.parametrize("case_index", [0, 1], ids=["paid_p2", "paid_p7"])
def test_real_confirmation_uses_the_local_cdf_without_promising_a_replayed_counterexample(
        paid_cases, case_index):
    case = paid_cases[case_index]
    task, paid = TaskSpec(**case["spec"]), _rows(case["paid_batches_before_confirmation"])
    selected = {"candidate": Candidate("revised", Program.from_dict(case["program"]), "research_program"),
                "fallback": case["fallback"]}
    baseline = {"candidate": Candidate("retained", Program.from_dict(case["baseline_program"]), "retained"),
                "fallback": case["baseline_fallback"]}
    budget = SearchBudget(15000)
    plan = plan_observed_condition_probe(task, paid, selected, budget,
        confirmation_baseline=baseline, length=24)
    assert plan is not None and plan["expected_disagreement"] > 0.
    assert 0 < plan["compute_cost"] == budget.spent <= 15000
    assert all(set(point) == set(task.feature_bounds) for point in plan["probe"].sequence)
    # Independently integrate the piecewise-uniform residual CDF on the chosen
    # actual neighbourhood. This checks the public plan's probability contract.
    mean = Program.from_dict(plan["reading_mean_program"])
    point = plan["fixed_control_background"]
    distances = [sum(((row.features[name] - point[name]) / (upper - lower)) ** 2
        for name, (lower, upper) in task.feature_bounds.items()) for row in paid]
    radius = plan["sampling_context"]["neighbourhood_radius"] ** 2
    local = [row for row, distance in zip(paid, distances)
             if distance <= radius or isclose(distance, radius, rel_tol=1e-12, abs_tol=1e-15)]
    assert len(local) == plan["local_reading_rows"]
    offset = mean.predict(Observation(point, 0.))
    readings = sorted(row.features[plan["condition_field"]]
        - mean.predict(Observation({name: row.features[name] for name in task.feature_bounds}, 0.))
        + offset for row in local)
    for exposure in plan["interval_exposure"]:
        left, right = exposure["threshold_interval"]
        mass = sum((float(left < low < right) if low == high else
            max(0., min(high, right) - max(low, left)) / (high - low))
            for low, high in zip(readings, readings[1:])) / (len(readings) - 1)
        assert exposure["estimated_probability"] == pytest.approx(mass)
    # Existing fresh misses remain outcome evidence, never planning rows.
    fresh = _rows([case["recorded_fresh_batch"]])
    assert all(predict(selected["candidate"].program, row, selected["fallback"])
        == predict(baseline["candidate"].program, row, baseline["fallback"]) for row in fresh)

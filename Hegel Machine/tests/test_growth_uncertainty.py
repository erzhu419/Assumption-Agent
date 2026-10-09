from random import Random

from hegel_machine.cognition.growth_uncertainty import estimate_uncertainty, calibration
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget


def test_repetitions_separate_deterministic_misspecification_from_noise():
    rows = [Observation({"x": x}, x * x) for x in (-1., -.5, 0., .5, 1.) for _ in range(6)]
    wrong = Program("const", value=0.)
    result = estimate_uncertainty(rows, wrong, ("x",), SearchBudget(10000))
    assert result["kind"] == "deterministic"
    assert result["variance"] == 0.
    assert result["repeat_degrees_of_freedom"] == 25


def test_variance_model_is_selected_across_groups_and_calibrates_fresh_outcomes():
    rng = Random(331)
    program = Program("feature", name="x")
    rows = [Observation({"x": x}, x + rng.gauss(0., .04 + .30 * abs(x)))
            for x in (-1., -.8, -.6, -.4, -.2, 0., .2, .4, .6, .8, 1.) for _ in range(30)]
    budget = SearchBudget(10000)
    result = estimate_uncertainty(rows, program, ("x",), budget)
    assert result["kind"] == "stochastic" and result["variance_program"]
    assert result["repeat_groups"] == 11
    assert len(result["variance_selection"]) == 3
    fresh = [Observation({"x": x}, x + rng.gauss(0., .04 + .30 * abs(x)))
             for x in (-.95, -.7, -.3, .1, .35, .65, .95) for _ in range(100)]
    report = calibration(result, program, fresh)
    assert .65 < report["standardized_mse"] < 1.5
    assert .86 < report["normal_95_coverage"] < .995
    assert budget.spent > len(rows)

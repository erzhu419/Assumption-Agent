"""Risk revisions preserve the task loss and do not pardon structural errors."""

from dataclasses import replace
from random import Random

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Definition, Program
from hegel_machine.cognition.models import Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.unified_uncertainty import assess_noisy_adequacy, residual_risk_candidates


SPEC = TaskSpec("uncertain_response", "Predict the externally fixed risk target", ("u",), (), (),
                risk_quantile=.9, feature_bounds={"u": (-1., 1.)})
BASE = Program("add", children=(Program("const", value=.15), Program("mul", children=(
    Program("const", value=.55), Program("feature", name="u")))))


def trajectories(seed, count=4, length=64, *, center=BASE, scale=None):
    rng, rows = Random(seed), []
    scale = scale or (lambda features: .08 + .2 * features["u"] ** 2)
    for _ in range(count):
        history = []
        for _ in range(length):
            features = {"u": rng.uniform(-1., 1.)}
            observation = Observation(features, 0., tuple(history))
            target = center.predict(observation) + scale(features) * rng.uniform(-1., 1.)
            rows.append(Observation(features, target, tuple(history)))
            history.append({**features, "__target__": target})
    return rows


def test_fixed_quantile_conditional_square_scale_improves_future_risk_loss():
    training, validation = trajectories(55), trajectories(151, length=128)
    budget = SearchBudget(60000)
    candidates = residual_risk_candidates(SPEC, BASE, training, budget)
    global_fit = next(item for item in candidates if item["kind"] == "residual_global_quantile")
    conditional = next(item for item in candidates if item["kind"] == "residual_quadratic_scale_quantile")
    assert all(item["fixed_quantile"] == .9 for item in candidates)
    assert loss(conditional["candidate"].program, validation, 0., .9) < .85 * loss(
        global_fit["candidate"].program, validation, 0., .9)
    assert budget.spent > len(training) and budget.spent <= budget.limit
    assessment = assess_noisy_adequacy(SPEC, conditional["candidate"].program, training,
                                      validation, SearchBudget(100000))
    assert assessment["adequate"]
    assert assessment["reason"] == "noise_compatible"
    assert assessment["validation_loss"] > 1e-7


def test_median_quantile_argument_does_not_replace_existing_squared_mean_objective():
    rows = [Observation({"u": float(index)}, value) for index, value in enumerate((0., 0., 0., 8.))]
    candidates = residual_risk_candidates(replace(SPEC, risk_quantile=.5),
        Program("const", value=0.), rows, SearchBudget(1000))
    assert candidates[0]["kind"] == "residual_mean"
    assert candidates[0]["candidate"].program.predict(rows[0]) == 2.


def test_arbitrary_callable_continuous_state_survives_residual_revision():
    state = Program("accumulate", value=.7, children=(Program("arg", name="drive"),
                                                     Program("const", value=0.)))
    definition = Definition("learned_state", ("drive",), state, "concept_0010")
    program = Program("call", name=definition.definition_id, definition=definition,
                      children=(Program("feature", name="u"),))
    rows = trajectories(85, count=2, length=48, center=program)
    candidates = residual_risk_candidates(SPEC, program, rows, SearchBudget(30000))
    assert candidates
    payload = candidates[-1]["candidate"].program.to_dict()
    assert "learned_state" in payload["definitions"]
    restored = Program.from_dict(payload)
    assert restored.history_depth >= 1
    assert restored.predict(rows[-1]) == candidates[-1]["candidate"].program.predict(rows[-1])


def test_predictable_current_residual_cannot_be_relabelled_as_noise():
    training, validation = trajectories(71, scale=lambda _: .05), trajectories(271, scale=lambda _: .05)
    assessment = assess_noisy_adequacy(replace(SPEC, risk_quantile=.5),
        Program("const", value=.15), training, validation, SearchBudget(100000))
    assert not assessment["adequate"]
    assert assessment["reason"] == "predictable_residual"
    assert assessment["structural_counterexample"]["gain"] > assessment["structural_counterexample"]["margin"]


def test_predictable_historical_residual_cannot_be_hidden_by_wide_noise_tolerance():
    remembered = Program("if", children=(Program("available", lag=1),
        Program("lag", name="u", lag=1), Program("const", value=0.)))
    true_program = Program("add", children=(BASE, remembered))
    training = trajectories(74, center=true_program, scale=lambda _: .03)
    validation = trajectories(274, center=true_program, scale=lambda _: .03)
    assessment = assess_noisy_adequacy(replace(SPEC, risk_quantile=.5), BASE,
                                      training, validation, SearchBudget(100000))
    assert not assessment["adequate"]
    assert assessment["reason"] == "predictable_residual"
    assert assessment["structural_counterexample"]["kind"] == "finite_history"


def test_mean_prediction_is_not_accepted_for_fixed_upper_tail_target():
    training, validation = trajectories(77), trajectories(177)
    assessment = assess_noisy_adequacy(SPEC, BASE, training, validation, SearchBudget(100000))
    assert not assessment["adequate"]
    assert assessment["empirical_quantile_coverage"] < .65
    assert assessment["reason"] in {"quantile_miscalibration", "predictable_residual"}


def test_tiny_sample_or_unfinished_residual_search_cannot_support_noise_adoption():
    training, validation = trajectories(59, count=2, length=24), trajectories(159, count=1, length=24)
    small = assess_noisy_adequacy(SPEC, BASE, training[:8], validation[:8], SearchBudget(10000))
    assert not small["adequate"] and small["reason"] == "insufficient_calibration_rows"
    limited = assess_noisy_adequacy(SPEC, BASE, training, validation, SearchBudget(100))
    assert not limited["adequate"] and limited["reason"] == "insufficient_budget"

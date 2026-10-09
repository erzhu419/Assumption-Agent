"""Additional reading experiments need actual information and an affordable bill."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.condition_research_returns import (
    consider_condition_research, record_condition_research)
from hegel_machine.cognition.models import Probe


def state(interval=(-.2, .2), *, status="unresolved", loss=0., revision=False):
    return {"status": status, "program": {"op": "const", "value": 0.},
        "original_target_loss": loss, "expected_prediction_ambiguity": .2,
        "conditions": [{"condition_field": "meter", "threshold_intervals": [list(interval)],
                        "needs_threshold_revision": revision}]}


def plan(probability=.1, interval=(-.2, .2), *, length=24, difference=2.):
    return {"probe": Probe("actual_meter_experiment", tuple({"x": 0.} for _ in range(length))),
        "expected_disagreement": probability * difference,
        "interval_exposure": [{"condition_field": "meter", "threshold_interval": list(interval),
                              "estimated_probability": probability, "response_disagreement": difference}]}


def decide(question, experiment, history=None, **kwargs):
    return consider_condition_research(question, experiment, history,
        observation_remaining=kwargs.pop("observation_remaining", 288),
        measurement_remaining=kwargs.pop("measurement_remaining", 288),
        compute_remaining=kwargs.pop("compute_remaining", 240000),
        measurement_rate=1, compute_quote=2000,
        observation_reserve=24, measurement_reserve=24, compute_reserve=1000, **kwargs)


def paid(history, before, after, experiment, **kwargs):
    return record_condition_research(history, before, after, experiment,
        observation_cost=24, measurement_cost=24, compute_cost=2100, **kwargs)


def test_first_useful_real_probe_is_bought_but_tiny_gap_is_retained():
    worthwhile = decide(state(), plan(.1))
    assert worthwhile["buy"]
    assert worthwhile["reason"] == "first_experiment_has_information_value"
    tiny = decide(state(interval=(-1e-7, 1e-7)), plan(1e-7, interval=(-1e-7, 1e-7)))
    assert not tiny["buy"]
    # A still nonzero prediction question remains pending, rather than being
    # pronounced complete because no more targeted observations were bought.
    assert tiny["reason"] == "retain_question_for_more_valuable_future_evidence"


def test_high_value_random_misses_get_further_trials_without_fixed_attempt_limit():
    question, experiment, history = state(), plan(.2), None
    for index in range(4):
        assert decide(question, experiment, history,
                      observation_remaining=288 - 24 * index,
                      measurement_remaining=288 - 24 * index)["buy"]
        history = paid(history, question, question, experiment)
    # Stop emerges from forecast failure plus the next bill, not an attempt cap.
    result = decide(question, experiment, history, observation_remaining=96, measurement_remaining=96)
    assert not result["buy"]
    assert history["information_batches"] == 0
    assert history["observation_cost"] == history["measurement_cost"] == 96
    assert history["compute_cost"] == 8400


def test_real_narrowing_preserves_value_after_several_low_yield_batches():
    question, experiment, history = state(), plan(.1), None
    for _ in range(5):
        history = paid(history, question, question, experiment)
    assert not decide(question, experiment, history, observation_remaining=144,
                      measurement_remaining=144)["buy"]
    # Additional representative paid evidence, not a fresh guessed midpoint,
    # determines a tighter supported cut with newly valuable exposure.
    narrow = state(interval=(-.1, .1))
    result = decide(narrow, plan(.05, interval=(-.1, .1)), history,
                    observation_remaining=144, measurement_remaining=144)
    assert result["buy"] and not result["reopened"]
    for radius in (.1, .05, .025):
        changed = state(interval=(-radius, radius))
        history = paid(history, question, changed, experiment)
        assert history["last_outcome"]["informative"]
        question = changed
    assert history["information_batches"] == 3


def test_new_original_target_counterexample_reopens_stopped_research():
    before, experiment, history = state(), plan(.03), None
    for _ in range(6):
        history = paid(history, before, before, experiment)
    assert not decide(before, experiment, history, observation_remaining=100,
                      measurement_remaining=100)["buy"]
    contradicted = state(status="contradicted", loss=.4)
    opened = decide(contradicted, experiment, history, observation_remaining=100,
                    measurement_remaining=100)
    assert opened["buy"] and opened["reopened"]
    history = paid(history, before, contradicted, experiment)
    # The same counterexample is not repeatedly presented as new information.
    assert not decide(contradicted, experiment, history, observation_remaining=100,
                      measurement_remaining=100)["buy"]
    explicit = decide(contradicted, experiment, history, observation_remaining=100,
                      measurement_remaining=100, new_counterexample=True)
    assert explicit["buy"]


def test_geometric_real_gap_narrowing_continues_while_next_batch_has_value():
    question, history = state(), None
    for index, probability in enumerate((.3, .15, .075, .0375, .01875)):
        radius = .2 / 2 ** index
        experiment = plan(probability, interval=(-radius, radius))
        result = decide(question, experiment, history,
                        observation_remaining=288 - 24 * index,
                        measurement_remaining=288 - 24 * index)
        assert result["buy"]
        narrowed = state(interval=(-radius / 2., radius / 2.))
        history = paid(history, question, narrowed, experiment)
        question = narrowed
    assert history["information_batches"] == 5
    assert history["last_outcome"]["relative_interval_reduction"] == .5


def test_actual_program_revision_is_retained_without_claiming_accuracy_gain():
    before, after, experiment = state(), state(), plan()
    after["program"] = {"op": "const", "value": 1.}
    history = paid(None, before, after, experiment, changed_program=True)
    assert history["information_batches"] == 1
    assert history["last_outcome"]["changed_program"]
    assert "gain" not in history["last_outcome"]
    assert json.loads(json.dumps(history)) == history


def test_meter_origin_and_unit_change_preserve_decisions_and_measured_contraction():
    before, after, experiment = state(), state(interval=(-.1, .1)), plan()
    transformed_before, transformed_after, transformed_plan = deepcopy(before), deepcopy(after), deepcopy(experiment)
    for question in (transformed_before, transformed_after):
        question["conditions"][0]["threshold_intervals"] = [
            [23. + 1000. * value for value in interval]
            for interval in question["conditions"][0]["threshold_intervals"]]
    transformed_plan["interval_exposure"][0]["threshold_interval"] = [-177., 223.]
    original = paid(None, before, after, experiment)
    changed = paid(None, transformed_before, transformed_after, transformed_plan)
    assert changed["last_outcome"]["relative_interval_reduction"] == pytest.approx(
        original["last_outcome"]["relative_interval_reduction"])
    assert decide(after, experiment, original)["buy"] == decide(transformed_after, transformed_plan, changed)["buy"]
    assert changed["predicted_information_batches"] == original["predicted_information_batches"]


def test_competing_duplicates_do_not_create_independent_reading_chances():
    experiment = plan(.001)
    experiment["interval_exposure"] *= 10
    result = decide(state(), experiment)
    assert result["probability_of_new_information"] < .025
    assert not result["buy"]
    # Forecast trajectory disagreement cannot turn the same rare gap into a
    # certain temporal observation merely by relabelling the rival programs.
    experiment["duration_disagreement"] = 100.
    assert not decide(state(), experiment)["buy"]


def test_effectless_readings_and_fresh_confirmation_reserve_are_respected():
    assert not decide(state(), plan(.5, difference=0.))["buy"]
    assert not decide(state(), plan(.5), observation_remaining=47)["buy"]
    assert not decide(state(), plan(.5), measurement_remaining=47)["buy"]
    assert not decide(state(), plan(.5), compute_remaining=2500)["buy"]
    assert not decide(state(status="supported"), plan(.5))["buy"]


def test_history_alternative_without_reading_bracket_has_information_value():
    experiment = plan()
    experiment["interval_exposure"] = []
    experiment["expected_disagreement"] = 2.
    result = decide(state(status="contradicted", loss=1.), experiment)
    assert result["buy"]
    assert result["probability_of_new_information"] == 1.

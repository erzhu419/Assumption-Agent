"""Unseen paid reading gaps have consequences for original predictions."""

import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Candidate, Observation, SearchBudget, TaskSpec
from hegel_machine.cognition.observed_condition_uncertainty import (
    assess_observed_condition_uncertainty, update_observed_condition_uncertainty)


def response(threshold=0., *, width=3, placement="inside_window", difference=2.,
             field="meter"):
    condition = Program("ge", children=(Program("feature", name=field),
                                        Program("const", value=threshold)))
    high, low = Program("const", value=difference), Program("const", value=0.)
    if placement == "inside_window":
        return Program("window_sum", lag=width,
            children=(Program("if", children=(condition, high, low)),))
    return Program("if", children=(condition,
        Program("window_sum", lag=width, children=(high,)),
        Program("window_sum", lag=width, children=(low,))))


def observations(actual, readings, *, field="meter"):
    rows, history = [], []
    for index, reading in enumerate(readings):
        features = {"x": (index % 5 - 2) / 2., field: reading}
        row = Observation(features, 0., tuple(history))
        target = actual.predict(row)
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return rows


def entry(program, *, interval=(-.2, .2), field="meter", nested=True,
          placement="inside_window"):
    inner = {"operation": "discover_conditional_coordinates",
             "threshold_interval": list(interval),
             "gate_coordinate": Program("feature", name=field).to_dict(),
             "width": 1, "placement": placement}
    steps = [{"operation": "discover_decoded_conditional_coordinates", "width": 3,
              "inner_selection_steps": [inner]}] if nested else [inner]
    return {"candidate": Candidate("fit", program, "research_program"),
            "research_program": {"selection_steps": steps}}


def spec(field="meter"):
    return TaskSpec("unlabelled", "", ("x", field), (), (),
                    feature_bounds={"x": (-1., 1.)}, max_probe_length=32)


READINGS = [-1., -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1.] * 3


@pytest.mark.parametrize("placement", ["inside_window", "outside_window"])
def test_zero_observed_loss_retains_real_future_threshold_question(placement):
    selected = response(placement=placement)
    rows = observations(response(.1, placement=placement), READINGS)
    budget = SearchBudget(20000)
    state = assess_observed_condition_uncertainty(spec(), rows,
        entry(selected, placement=placement), budget)
    assert state["original_target_loss"] == 0.
    assert state["status"] == "unresolved"
    assert state["expected_prediction_ambiguity"] > 1e-7
    condition = state["conditions"][0]
    assert condition["width"] == 3
    assert condition["placement"] == placement
    assert condition["threshold_intervals"] == [[-.2, .2]]
    first, second = (Program.from_dict(item) for item in condition["endpoint_programs"][:2])
    # These are complete executable window responses, not decoded branch loss.
    probe = Observation({"x": 0., "meter": 0.}, 0.,
                        ({"x": 0., "meter": 0.}, {"x": 0., "meter": 0.}))
    assert abs(first.predict(probe) - second.predict(probe)) == 6.
    assert state["compute_cost"] == budget.spent < 20000


def test_new_real_target_evidence_narrows_interval_after_json_restore():
    selected = response()
    rows = observations(response(.1), READINGS)
    state = assess_observed_condition_uncertainty(spec(), rows, entry(selected), SearchBudget(20000))
    new = observations(response(.1), [.05, .15])
    restored = json.loads(json.dumps(state))
    changed = update_observed_condition_uncertainty(restored, [*rows, *new], SearchBudget(20000))
    condition = changed["conditions"][0]
    assert condition["threshold_intervals"] == [[.05, .15]]
    assert condition["needs_threshold_revision"]
    assert condition["gain"] > 0.
    assert condition["conditional_loss"] == 0.
    assert condition["expected_prediction_ambiguity"] < state["expected_prediction_ambiguity"]
    assert changed["status"] == "unresolved"
    # The saved old midpoint lies outside its newly narrowed support. A second
    # continuation must retain that revision question until it actually refits.
    again = update_observed_condition_uncertainty(json.loads(json.dumps(changed)),
        [*rows, *new, *observations(response(.1), [.09, .11])], SearchBudget(20000))
    assert again["conditions"][0]["threshold_intervals"] == [[.09, .11]]
    assert again["status"] == "unresolved"


def test_counterevidence_withdraws_the_fixed_response_instead_of_fake_narrowing():
    rows = observations(response(.1), READINGS)
    selected = response()
    state = assess_observed_condition_uncertainty(spec(), rows, entry(selected), SearchBudget(20000))
    incompatible = Observation({"x": 0., "meter": 0.}, 19.)
    changed = update_observed_condition_uncertainty(state, [*rows, incompatible], SearchBudget(20000))
    assert changed["status"] == "contradicted"
    assert changed["conditions"][0]["threshold_intervals"] == []
    assert update_observed_condition_uncertainty(changed, rows, SearchBudget(20000))["status"] == "contradicted"


def test_smooth_response_and_prediction_irrelevant_gate_can_finish():
    smooth = Program("window_sum", lag=3, children=(Program("feature", name="meter"),))
    rows = observations(smooth, READINGS)
    assert assess_observed_condition_uncertainty(spec(), rows, entry(smooth), SearchBudget(20000)) is None
    small = response(difference=1e-5)
    rows = observations(response(.1, difference=1e-5), READINGS)
    state = assess_observed_condition_uncertainty(spec(), rows, entry(small), SearchBudget(20000))
    assert state["status"] == "supported"
    assert state["expected_prediction_ambiguity"] <= 1e-7


def test_changed_meter_units_preserve_empirical_prediction_risk():
    first = response()
    original = observations(response(.1), READINGS)
    state = assess_observed_condition_uncertainty(spec(), original, entry(first), SearchBudget(20000))
    scale, offset, field = 2.3, -.4, "instrument"
    shifted = observations(response(scale * .1 + offset, field=field),
                           [scale * reading + offset for reading in READINGS], field=field)
    changed = assess_observed_condition_uncertainty(spec(field), shifted,
        entry(response(offset, field=field), field=field,
              interval=(scale * -.2 + offset, scale * .2 + offset)), SearchBudget(20000))
    assert changed["expected_prediction_ambiguity"] == pytest.approx(state["expected_prediction_ambiguity"])


def test_exhausted_prediction_budget_preserves_an_unanswered_question():
    rows = observations(response(.1), READINGS)
    state = assess_observed_condition_uncertainty(spec(), rows, entry(response()), SearchBudget(1))
    assert state["status"] == "unresolved"
    assert state["expected_prediction_ambiguity"] is None
    assert state["compute_cost"] == 1

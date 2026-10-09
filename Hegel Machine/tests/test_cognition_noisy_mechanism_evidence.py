"""A fresh repeat separates noisy mechanism predictions from structural error."""

from dataclasses import replace
from random import Random

import pytest

from hegel_machine.cognition.competing_object_experiments import assess_competing_object_probe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation, SearchBudget


def _trajectory(seed):
    rng, history, rows = Random(seed), [], []
    for index in range(32):
        features = {"x": (index % 7 - 3) / 3.}
        target = 2. * features["x"] + rng.gauss(0., .03)
        rows.append(Observation(features, target, tuple(history)))
        history.append({**features, "__target__": target})
    return rows


def _plan(*programs):
    return {"predicted_fields": {}, "hypotheses": [
        {"hypothesis_id": f"hypothesis_{index}", "candidate_id": f"candidate_{index}",
         "program": program.to_dict(), "predictions": None}
        for index, program in enumerate(programs)]}


def _response():
    return Program("mul", children=(Program("const", value=2.), Program("feature", name="x")))


def test_independent_noisy_repeat_supports_frozen_response_and_refutes_wrong_window():
    original, repeated = _trajectory(71), _trajectory(171)
    response = _response()
    wrong_window = Program("window_sum", lag=3, children=(response,))
    plan, budget = _plan(response, wrong_window), SearchBudget(1000)
    result = assess_competing_object_probe(plan, list(zip(original, original)), budget,
                                          repeated_rows=repeated)
    correct, wrong = result["hypotheses"]
    assert result["status"] == "distinguished"
    assert correct["measured_loss"] > 1e-7
    assert correct["deterministic_status"] == "counterexample"
    assert correct["deterministic_counterexamples"] and correct["counterexamples"] == []
    assert correct["status"] == "supported_on_probe"
    assert correct["noise_replication"]["noise_supported"]
    assert correct["noise_replication"]["matched_rows"] == len(original)
    assert wrong["status"] == "counterexample" and wrong["counterexamples"]
    assert wrong["noise_replication"]["repeatable_error"]
    assert result["compute_cost"] == budget.spent == 9 * len(original)


@pytest.mark.parametrize("mismatch", ["current_input", "input_history", "target_history"])
def test_incompatible_repeat_cannot_supply_support_or_refutation(mismatch):
    original, repeated, program = _trajectory(71), _trajectory(171), _response()
    if mismatch == "current_input":
        repeated[16] = replace(repeated[16], features={"x": repeated[16].features["x"] + 1.})
    elif mismatch == "input_history":
        repeated[16] = replace(repeated[16], history=tuple(
            {**past, "x": past["x"] + 1.} for past in repeated[16].history))
    else:
        program = Program("lag", name="__target__", lag=1)
    result = assess_competing_object_probe(_plan(program), list(zip(original, original)),
        SearchBudget(1000), repeated_rows=repeated)
    item = result["hypotheses"][0]
    assert result["status"] == item["status"] == "unresolved"
    assert item["counterexamples"] == []
    assert not item["noise_replication"]["noise_supported"]
    assert not item["noise_replication"]["repeatable_error"]
    assert item["noise_replication"]["reason"] == (
        "insufficient_matched_rows" if mismatch == "target_history" else "unmatched_trajectory_inputs")


def test_repeat_assessment_without_computation_remains_unresolved_and_keeps_measured_loss():
    original, repeated = _trajectory(71), _trajectory(171)
    budget = SearchBudget(len(original))
    result = assess_competing_object_probe(_plan(_response()), list(zip(original, original)),
                                          budget, repeated_rows=repeated)
    item = result["hypotheses"][0]
    assert item["measured_loss"] > 1e-7 and item["deterministic_status"] == "counterexample"
    assert item["status"] == "unresolved" and item["counterexamples"] == []
    assert item["noise_replication"]["reason"] == "insufficient_budget"
    assert result["compute_cost"] == budget.spent == len(original)


def test_exact_response_remains_supported_when_the_new_repeat_has_no_material_error():
    original = [Observation({"x": float(index)}, 2. * index) for index in range(4)]
    repeated = [Observation({"x": float(index)}, 2. * index) for index in range(4)]
    budget = SearchBudget(1000)
    result = assess_competing_object_probe(_plan(_response(), Program("const", value=0.)),
        list(zip(original, original)), budget, repeated_rows=repeated)
    correct, wrong = result["hypotheses"]
    assert result["status"] == "distinguished"
    assert correct["deterministic_status"] == correct["status"] == "supported_on_probe"
    assert correct["measured_loss"] == 0. and correct["counterexamples"] == []
    assert correct["noise_replication"]["reason"] == "no_material_repeatable_error"
    assert not correct["noise_replication"]["noise_supported"]
    assert wrong["status"] == "counterexample" and wrong["noise_replication"]["repeatable_error"]
    assert result["compute_cost"] == budget.spent == 8 * len(original)


def test_unrequested_repeat_keeps_deterministic_assessment_and_calibration_behavior():
    rows = [Observation({"x": float(index)}, 2. * index) for index in range(4)]
    plan = _plan(_response(), Program("const", value=0.))
    plan["predicted_fields"] = {"x": [row.features["x"] + 1. for row in rows]}
    budget = SearchBudget(1000)
    result = assess_competing_object_probe(plan, list(zip(rows, rows)), budget)
    correct, wrong = result["hypotheses"]
    assert result["status"] == "distinguished"
    assert result["calibration_status"] == "counterexample" and result["field_losses"] == {"x": 1.}
    assert correct["status"] == "supported_on_probe" and correct["measured_loss"] == 0.
    assert wrong["status"] == "counterexample" and wrong["measured_loss"] == 14.
    assert all("noise_replication" not in item and "deterministic_status" not in item
               for item in result["hypotheses"])
    assert result["compute_cost"] == budget.spent == 3 * len(rows)

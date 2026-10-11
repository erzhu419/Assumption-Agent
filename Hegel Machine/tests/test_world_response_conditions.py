"""Paid Boolean outcomes produce falsifiable rules, not equality labels."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.world_response_conditions import (
    _normalize_condition,
    predict_response_condition,
    synthesize_response_conditions,
)


def band_packet():
    """The latent centre is known; the activation condition is not supplied."""
    centre = Program("mul", children=(Program("const", value=10.),
                                     Program("feature", name="density")))
    packet = {
        "measurements": [], "bound_rows": [], "responses": [],
        "control_grids": {}, "controller_path": "shared/controller.py",
        "paid_family_checks": {"linear": {
            "status": "SAT_WITNESS", "witness": {"program": centre.to_dict()}}},
    }
    for number in (1, 2):
        packet["measurements"].append({"crystal": number, "measurements": {
            "density": {"value": float(number), "evidence": [number]}}})
        packet["bound_rows"].append({"crystal": number,
            "features": {"density": float(number)}, "evidence": [number]})
        packet["control_grids"][str(number)] = {
            "lower": 0., "upper": 30., "step": .5, "origin": 0.}
        for offset, activated in ((-2., False), (0., True), (2., False)):
            packet["responses"].append({"crystal": number,
                "frequency": 10. * number + offset, "activated": activated,
                "evidence": [10 + len(packet["responses"])]})
    return packet


def forecast(candidate, packet, number, frequency):
    measured = next(row["measurements"] for row in packet["measurements"]
                    if row["crystal"] == number)
    bound = next(row["features"] for row in packet["bound_rows"]
                 if row["crystal"] == number)
    return predict_response_condition(candidate, measured, bound, frequency)


def compatible(result):
    return [row for row in result["candidates"] if row["status"] == "compatible"]


def test_narrow_positive_band_is_learned_from_boolean_responses_not_setting_labels():
    packet = band_packet()
    original = deepcopy(packet)

    result = synthesize_response_conditions(packet)

    rules = compatible(result)
    assert rules, "the old condition generator should compose both sides of a band"
    for candidate in rules:
        for response in packet["responses"]:
            prediction = forecast(candidate, packet, response["crystal"], response["frequency"])
            assert type(prediction) is bool
            assert prediction == response["activated"]
        assert candidate["status"] != "supported"
    assert result["compute_cost"] > 0
    assert packet == original


def test_selected_trial_contains_actual_rule_disagreement_on_a_legal_unpaid_setting():
    packet = band_packet()

    result = synthesize_response_conditions(packet)

    trial = result["selected_experiment"]
    assert trial is not None
    number, frequency = trial["number"], trial["prediction"]
    grid = packet["control_grids"][str(number)]
    assert grid["lower"] <= frequency <= grid["upper"]
    assert (frequency - grid["origin"]) / grid["step"] == pytest.approx(
        round((frequency - grid["origin"]) / grid["step"]))
    assert (number, frequency) not in {
        (row["crystal"], row["frequency"]) for row in packet["responses"]}
    predictions = {rule["id"]: forecast(rule, packet, number, frequency)
                   for rule in compatible(result)}
    assert trial["predictions"] == predictions
    assert set(predictions.values()) == {False, True}
    action = result["experiment_action"]
    assert action["action"] == "run"
    assert action["arguments"]["operator"] == "reference_response"
    assert action["arguments"]["path"] == packet["controller_path"]
    assert action["arguments"]["experiment"] == {"number": number, "setting": frequency}


def test_fresh_contrary_response_refutes_frozen_rule_without_confirming_survivors():
    packet = band_packet()
    before = synthesize_response_conditions(packet)
    trial = before["selected_experiment"]
    first = compatible(before)[0]
    actual = not trial["predictions"][first["id"]]
    contradicted = {identifier for identifier, prediction in trial["predictions"].items()
                   if prediction != actual}
    survivors = {identifier for identifier, prediction in trial["predictions"].items()
                 if prediction == actual}
    packet["responses"].append({"crystal": trial["number"],
        "frequency": trial["prediction"], "activated": actual, "evidence": [99]})

    after = synthesize_response_conditions(packet, previous=before)

    indexed = {row["id"]: row for row in after["candidates"]}
    assert all(indexed[identifier]["status"] == "refuted" for identifier in contradicted)
    assert all(indexed[identifier]["conflicting_responses"] for identifier in contradicted)
    assert survivors <= {row["id"] for row in compatible(after)}
    assert all(row["status"] != "supported" for row in after["candidates"])
    for candidate in compatible(after):
        assert candidate["program"] == next(row["program"] for row in before["candidates"]
                                             if row["id"] == candidate["id"])
        assert forecast(candidate, packet, trial["number"], trial["prediction"]) == actual
    assert after["compute_cost"] > 0


def test_identical_paid_inputs_after_json_resume_reuse_rules_without_charging_search():
    packet = band_packet()
    before = synthesize_response_conditions(packet)
    saved = json.loads(json.dumps(before))

    cached = synthesize_response_conditions(deepcopy(packet), previous=saved)

    assert cached["cached"] is True
    assert cached["compute_cost"] == 0
    assert cached["candidates"] == before["candidates"]
    assert cached["selected_experiment"] == before["selected_experiment"]


def test_mutually_conflicting_paid_boolean_labels_cannot_yield_a_compatible_rule():
    packet = band_packet()
    contrary = deepcopy(packet["responses"][1])
    contrary.update(activated=False, evidence=[99])
    packet["responses"].append(contrary)

    result = synthesize_response_conditions(packet)

    assert not compatible(result)
    assert result["selected_experiment"] is None
    assert result["experiment_action"] is None


def test_small_compute_allowance_is_finite_search_exhaustion_not_proof_of_no_law():
    result = synthesize_response_conditions(band_packet(), compute_limit=1)

    assert 0 <= result["compute_cost"] <= 1
    assert result["status"] == "search_budget_exhausted"
    assert result["scope_is_exhaustive"] is False
    assert "finite" in result["search_scope"]
    assert not compatible(result)
    assert result["selected_experiment"] is None
    assert result["experiment_action"] is None


def test_unchecked_old_rules_are_not_compatible_after_new_evidence_exhausts_screening():
    packet = band_packet()
    before = synthesize_response_conditions(packet)
    contrary = deepcopy(packet["responses"][0])
    contrary.update(activated=not contrary["activated"], evidence=[99])
    packet["responses"].append(contrary)

    result = synthesize_response_conditions(packet, previous=before, compute_limit=1)

    assert result["status"] == "search_budget_exhausted"
    assert not compatible(result)
    assert result["selected_experiment"] is None
    assert result["experiment_action"] is None


def test_boolean_wrapper_normalization_preserves_distinct_unmeasured_boundaries():
    coordinate = Program("feature", name="residual_0")

    def band(upper):
        return Program("and", children=(
            Program("ge", children=(coordinate, Program("const", value=-1.))),
            Program("lt", children=(coordinate, Program("const", value=upper)))))

    narrow, broad = band(1.), band(1.5)
    wrapped = Program("ge", children=(narrow, Program("const", value=.5)))
    twice_wrapped = Program("ge", children=(wrapped, Program("const", value=.5)))

    assert _normalize_condition(twice_wrapped) == narrow
    assert _normalize_condition(broad) != narrow
    # Their paid forecasts coincide, but that is not equivalence of the rules.
    for value in (-2., 0., 2.):
        row = Observation({"residual_0": value}, 0.)
        assert narrow.predict(row) == broad.predict(row)
    unpaid = Observation({"residual_0": 1.25}, 0.)
    assert not _normalize_condition(twice_wrapped).predict(unpaid)
    assert _normalize_condition(broad).predict(unpaid)

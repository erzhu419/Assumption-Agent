"""Incidental physical success constrains later research without proving its law.

These are synthetic laboratory integration cases. They exercise the real
controller, public observations and historical fitter, not model discovery.
"""

from copy import deepcopy
import json
from math import sqrt

import pytest

from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.models import Observation
from hegel_machine.cognition.synthesis import _fit_linear
from test_world_reference_hypothesis import PATH, ReferenceLab, prepared, square, submit


def _incidental_success(tmp_path):
    agent, lab = prepared(tmp_path)
    # A gap-squared law fits all three original references, but predicts 250
    # for C4. The physical controller passes through the successful setting
    # 260 and then correctly refutes that law at its final prediction, 250.
    lab.gap[4] = sqrt(15)
    # The successful control setting is an observed operational label. The
    # instrument does not reveal the exact latent resonance, 260.3 here.
    lab.required[4] = 260.3
    result = submit(agent, lab, square("gap", "gap_square"))
    return agent, lab, result


def _questions(agent):
    return {question["id"]: question for question in agent.questions}


def test_incidental_activation_restores_target_without_supporting_refuted_formula(tmp_path):
    agent, lab, result = _incidental_success(tmp_path)
    question = _questions(agent)["reference_test_4_gap_square"]
    assert question["status"] == "refuted"
    assert question["answer"]["prediction"] == pytest.approx(250.)
    assert question["answer"]["actual_frequency"] == 250
    assert question["answer"]["activated"] is False

    state = json.loads((tmp_path / recipe.STATE).read_text())
    assert lab.active(4) and lab.frequencies[4] == 260
    assert state["completed"] == [4]
    assert state["reference_numbers"] == [1, 2, 3]
    assert state["target_numbers"] == [4, 5]
    assert state["tests"][0]["status"] == "refuted"
    assert result["setting_recoveries"]
    # Restoring C4 is followed by actual research on the remaining target.
    assert "reference_fit_gap_square_5" in agent.relation_research
    assert agent.costs["model_calls"] == 0

    row = next(row for row in result["research_table"]["targets"] if row["crystal"] == 4)
    successful = [item for item in row["response_settings"] if item["activated"]]
    assert any(item["frequency"] == 260 for item in successful)
    assert any(item["frequency"] == 250 and not item["activated"]
               for item in row["response_settings"])
    assert "response_constraints" in result


def test_observed_target_adds_calibration_row_and_changes_next_target_prediction(tmp_path):
    agent, lab, first = _incidental_success(tmp_path)
    term = {"op": "feature", "name": "density"}
    squared = square("density", "unused")["basis"][0]
    cubed = {"op": "mul", "children": [deepcopy(squared), deepcopy(term)]}
    hypothesis = {"id": "density_polynomial", "basis": [term, squared, cubed],
                  "rationale": "Resolve the remaining polynomial freedom with the new C4 observation."}
    result = submit(agent, lab, hypothesis)

    assert lab.success and result["status"] == "completed"
    assert lab.frequencies[4] == 260 and lab.frequencies[5] == 350
    assert _questions(agent)["reference_test_4_gap_square"]["status"] == "refuted"
    assert _questions(agent)["reference_test_5_density_polynomial"]["status"] == "supported"
    fit = next(item for item in result["fitted_predictions"]
               if item["question_id"] == "reference_fit_density_polynomial_5")
    observed = [row for row in fit["fit_rows"]
                if row.get("target_kind") == "observed_successful_setting"]
    assert len(fit["fit_rows"]) == 4 and len(observed) == 1
    assert observed[0]["crystal"] == 4
    assert observed[0]["target"] == 260
    assert observed[0]["target"] != lab.required[4]
    table_row = next(row for row in first["research_table"]["targets"] if row["crystal"] == 4)
    positive_evidence = {index for item in table_row["response_settings"]
                         if item["activated"] and item["frequency"] == 260
                         for index in item["evidence"]}
    assert positive_evidence.intersection(observed[0]["evidence"])

    # This is not merely recording another citation: with three references
    # this same basis has an unconstrained degree of freedom and predicts a
    # different physical setting. C4 evidence changes C5's executable test.
    terms = [Program.from_dict(value) for value in hypothesis["basis"]]
    original_rows = [Observation(row["features"], row["target"])
                     for row in fit["fit_rows"]
                     if row.get("target_kind") != "observed_successful_setting"]
    without_c4 = _fit_linear(terms, original_rows).predict(Observation({"density": 5.}, 0.))
    prediction = fit["selected_probe"]["predictions"][0]
    assert prediction == pytest.approx(350., abs=.01)
    assert abs(without_c4 - prediction) > 20
    # Inherited laws preserve their original calibration data and roles.
    for family in ("density", "gap"):
        assert len(agent.relation_research["reference_fit_" + family + "_5"]["fit_rows"]) == 3
    state = json.loads((tmp_path / recipe.STATE).read_text())
    assert state["reference_numbers"] == [1, 2, 3]
    assert state["target_numbers"] == [4, 5]
    assert agent.costs["model_calls"] == 0


def test_existing_negative_response_rejects_repeated_prediction_without_retuning(tmp_path):
    agent, lab = prepared(tmp_path)
    first = submit(agent, lab, square("gap", "gap_square"))
    rejected = first["tests"][0]
    assert rejected["number"] == 4 and rejected["status"] == "refuted"
    assert rejected["prediction"] == pytest.approx(350.)
    count = len(lab.actions)
    revised = submit(agent, lab, square("gap", "same_prediction_new_name"))
    assert len(lab.actions) == count
    assert revised["tests"] == []
    constraint = revised["response_constraints"][0]
    assert constraint["family"] == "same_prediction_new_name"
    assert constraint["status"] == "refuted" and constraint["setting"] == 350
    original_evidence = _questions(agent)["reference_test_4_gap_square"]["evidence"]
    assert constraint["evidence"] and max(constraint["evidence"]) <= max(original_evidence)
    cited = next(record for record in agent.observations
                 if record["index"] == constraint["evidence"][-1])
    assert "350" in cited["observation"]["dialog_box"]["dialogIn"]
    assert any(obj["uuid"] == 104 and "(uncalibrated)" in obj["name"]
               for obj in cited["observation"]["accessibleEnvironmentObjects"])
    assert not lab.success


def test_response_setting_requires_actual_installed_crystal_and_observed_reactor_state():
    lab = ReferenceLab()
    lab.dialog_number = 4
    lab.in_dialog = True
    lab.frequencies[4] = 260
    records = []

    def record():
        item = {"index": len(records), "action": {"action": "TALK", "arg1": 104},
                "feedback": {"success": True}, "observation": deepcopy(lab.public_observation())}
        records.append(item)
        return item

    # A dialog showing the right number is not evidence that a loose crystal
    # responds at that setting.
    record()
    lab.installed.add(4)
    positive = record()
    missing_reactor = record()
    missing_reactor["observation"]["accessibleEnvironmentObjects"] = [
        obj for obj in missing_reactor["observation"]["accessibleEnvironmentObjects"]
        if obj["uuid"] != 104]
    other_reactor = record()
    next(obj for obj in other_reactor["observation"]["accessibleEnvironmentObjects"]
         if obj["uuid"] == 204)["description"] = (
             "quantum crystal 4 (in crystal reactor (activated) [uuid: 105])")
    lab.frequencies[4] = 250
    negative = record()

    settings = recipe._response_settings(records, {"4": 204}, {"4": 104})
    assert {index for item in settings for index in item["evidence"]} == {
        positive["index"], negative["index"]}
    assert next(item for item in settings if item["activated"])["frequency"] == 260
    assert next(item for item in settings if not item["activated"])["frequency"] == 250

"""Reference evidence must change shared consequences before target confirmation."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_response_family_pilot.py"
spec = importlib.util.spec_from_file_location("response_family_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_acquired_reference_response_changes_two_unobserved_target_predictions():
    lab = pilot.ResponseLab("parameter_sufficient")
    state = lab.initial_state()
    before = pilot.consequence_snapshot(state)
    assert before["possible_signatures"] == [[False, False], [True, False], [True, True]]

    state["response_settings"].append(lab.measure(3, 8.))
    after = pilot.consequence_snapshot(state)
    assert after["possible_signatures"] == [[True, True]]
    forecasts = pilot.changed_predictions(before, after, state["response_settings"])
    assert [row["crystal"] for row in forecasts] == [4, 5]
    assert not any(row["crystal"] >= 4 for row in lab.records)
    for forecast in forecasts:
        response = lab.measure(forecast["crystal"], forecast["frequency"])
        assert response["activated"] == forecast["predicted_activated"]


def test_new_shared_feature_refutes_old_family_and_implies_unmeasured_predictions():
    lab = pilot.ResponseLab("shared_feature_needed")
    state = lab.initial_state()
    state["response_settings"].append(lab.measure(1, 4.))
    before = pilot.consequence_snapshot(state)
    assert before["families"] == {"shared_x": "UNSAT_VERIFIED", "shared_x_z": "SAT_WITNESS"}
    state["response_settings"].append(lab.measure(3, 8.))
    after = pilot.consequence_snapshot(state)
    assert after["signatures"][-1] == {"outcomes": [True, True], "status": "SAT_WITNESS"}
    forecasts = pilot.changed_predictions(before, after, state["response_settings"])
    assert after["possible_signatures"] == [[True, True]]
    assert all(row["status"] == "UNSAT_VERIFIED" for row in after["signatures"][:-1])
    assert [row["crystal"] for row in forecasts] == [4, 5]
    assert not any(row["crystal"] >= 4 for row in lab.records)


def test_false_shared_assumption_has_no_target_confirmation():
    lab = pilot.ResponseLab("shared_assumption_false")
    state = lab.initial_state()
    before = pilot.consequence_snapshot(state)
    state["response_settings"].append(lab.measure(1, 3.))
    after = pilot.consequence_snapshot(state)
    assert set(after["families"].values()) == {"UNSAT_VERIFIED"}
    assert after["possible_signatures"] == []
    assert pilot.changed_predictions(before, after, state["response_settings"]) == []


def test_public_selector_buys_reference_then_parent_adopts_its_new_predictions():
    result = pilot.run_case("parameter_sufficient")
    assert result["new_target_predictions_confirmed"]
    research, = result["trace"]
    assert research["response"]["crystal"] <= 3
    assert research["decision"]["selected_role"] == "conditional_family_test"
    assert research["before"]["possible_signatures"] != research["after"]["possible_signatures"]
    assert len(result["confirmations"]) == 2
    for confirmation in result["confirmations"]:
        assert confirmation["parent_selection_role"] == "conditional_family_task_action"
        assert confirmation["correct"]
        assert min(confirmation["response"]["evidence"]) > max(research["response"]["evidence"])
    assert result["costs"]["model_calls"] == 0


def test_parent_revises_shared_feature_then_confirms_new_predictions_within_original_budget():
    result = pilot.run_case("shared_feature_needed")
    assert result["refuted_family_ids"] == ["shared_x"]
    assert result["final"]["families"]["shared_x_z"] == "SAT_WITNESS"
    assert result["new_target_predictions_confirmed"]
    assert result["prediction_confirmation"]["confirmed_count"] == 2
    assert any(row["decision"]["selected_role"] == "shared_premise_test"
               for row in result["trace"])
    assert result["costs"]["research_response_actions"] <= 6
    for confirmation in result["confirmations"]:
        assert min(confirmation["response"]["evidence"]) > max(confirmation["source_reference_evidence"])
        assert confirmation["correct"]
    assert result["costs"]["model_calls"] == 0


def test_parent_finds_same_measurement_conflict_without_counting_it_as_prediction_success():
    result = pilot.run_case("shared_assumption_false")
    assert set(result["refuted_family_ids"]) == {"shared_x", "shared_x_z"}
    assert result["scientific_diagnosis"]["status"] == "declared_families_refuted"
    obligation = result["scientific_diagnosis"]["research_obligation"]
    assert obligation["reason"] == "declared_shared_response_families_refuted"
    assert obligation["same_measurement_conflict"]
    assert any(row["decision"]["selected_role"] == "shared_premise_test"
               for row in result["trace"])
    assert result["prediction_confirmation"]["status"] == "unconfirmed"
    assert not result["new_target_predictions_confirmed"]
    assert result["confirmations"] == []
    assert result["costs"]["research_response_actions"] <= 6
    assert result["costs"]["model_calls"] == 0


def test_evaluator_retains_forecasts_across_an_intervening_parent_action(monkeypatch):
    # This controls only a harness test's parent schedule; the automatic pilot
    # above always uses the production selector without replacement.
    original = pilot.propose_response_research

    def defer_confirmation(state, estimators, path):
        proposal = original(state, estimators, path)
        if len(state["response_settings"]) == 4:
            action = next(row for row in proposal["experiments"]
                          if row["number"] == 3 and row["prediction"] == 9.)
            proposal.update(selected_number=3, selected_setting=9.)
            proposal["selection"]["selected_experiment"] = action
        return proposal

    monkeypatch.setattr(pilot, "propose_response_research", defer_confirmation)
    result = pilot.run_case("parameter_sufficient")
    assert result["new_target_predictions_confirmed"]
    assert len(result["trace"]) == 2
    assert len(result["confirmations"]) == 2
    source = result["trace"][0]["response"]["evidence"]
    assert all(row["source_reference_evidence"] == source for row in result["confirmations"])
    assert not result["pending_predictions"]

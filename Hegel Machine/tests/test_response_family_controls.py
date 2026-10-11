"""The matched study must use live child returns and charge every response."""

import pytest

from scripts import run_response_family_controls as study


def test_child_exists_before_acquisition_and_parent_cannot_bypass_its_return(tmp_path):
    lab, agent, _ = study.prepare("parameter_sufficient", "recursive_family", tmp_path)
    agent._question({"id": "reference", "parent_id": "task", "question": "Test reference support"})
    with pytest.raises(RuntimeError, match="before the actual child answer"):
        study.consume_return(agent, "reference")
    assert lab.environment_steps == 3
    child = next(q for q in agent.questions if q["id"] == "reference")
    assert child["opened_before_observation"] == len(agent.observations)


def test_live_child_return_changes_parent_prediction_with_no_extra_budget(tmp_path):
    result = study.run_case("parameter_sufficient", "recursive_family", tmp_path)
    first = result["trace"][0]
    assert first["child_call"]
    assert first["before_possible_signatures"] != first["returned_possible_signatures"]
    assert first["returned_possible_signatures"] == [[True, True]]
    child = next(q for q in result["questions"] if q["id"] == first["question_id"])
    assert child["opened_before_observation"] <= first["response"]["evidence"][0]
    assert child["acquired_observations"] == first["response"]["evidence"]
    assert child["reviewed_through_observation"] == first["response"]["evidence"][0]
    assert result["trace"][1]["response_count_seen_by_parent"] == 5
    assert result["child_answers_returned"] == 1
    assert result["costs"]["world_actions"] == 3 + len(result["trace"]) == 6
    assert result["prediction_task_completed"]
    assert len(result["confirmations"]) == 2
    assert all(row["correct"] for row in result["confirmations"])
    assert result["common_spend"][3]["prediction_task_completed"]
    assert result["common_spend"][8]["state_carried_after_stop"]
    assert result["common_spend"][8]["actual_new_responses"] == 3
    assert result["observed_endpoints"]["first_both_targets_new_response_count"] == 3
    assert result["common_spend"][1]["frozen_forecast_evaluation"]
    assert all(p["correct"] for p in result["common_spend"][1]["frozen_forecast_evaluation"])


def test_confirmation_has_no_exemption_from_the_shared_response_limit(tmp_path):
    row = study.run_case("parameter_sufficient", "flat_family", tmp_path, max_new_responses=2)
    assert row["status"] == "response_budget_exhausted"
    assert not row["prediction_task_completed"]
    assert row["costs"]["new_response_actions"] == 2
    assert row["costs"]["world_actions"] == 5
    assert len(row["confirmations"]) == 1


def test_same_log_family_updaters_cannot_access_the_source_lab_or_buy_data():
    state = study.ResponseLab("parameter_sufficient").initial_state()
    source = {"trace": [{"response": {"crystal": 3, "frequency": 8.,
                                       "activated": True, "evidence": [8]}}]}
    replay = study.replay_log("parameter_sufficient", source)
    assert replay["arms"]["flat_family"] == replay["arms"]["recursive_family"]
    assert replay["arms"]["flat_family"][0]["snapshot"]["possible_signatures"] != [[True, True]]
    assert replay["arms"]["flat_family"][1]["snapshot"]["possible_signatures"] == [[True, True]]
    assert len(state["response_settings"]) == 3


def test_empty_member_set_never_produces_a_confident_prediction():
    state = study.ResponseLab("shared_assumption_false").initial_state()
    snapshot = study.member_snapshot(state, {"hypotheses": []})
    assert study.definite_predictions(snapshot, state["response_settings"]) == []


def test_activation_is_counted_even_without_a_prospective_certain_forecast():
    state = study.ResponseLab("parameter_sufficient").initial_state()
    responses = state["response_settings"] + [
        {"crystal": 4, "frequency": 4., "activated": True, "evidence": [8]},
        {"crystal": 5, "frequency": 7., "activated": True, "evidence": [9]}]
    row = {"case": "parameter_sufficient", "final_responses": responses,
           "final_snapshot": {"panel": state["consequence_panel"]},
           "common_spend": [{"actual_new_responses": n, "unmeasured_predictions": []}
                            for n in range(3)]}
    study.add_observed_endpoints(row)
    assert row["observed_endpoints"]["both_targets_ever_activated"]
    assert row["observed_endpoints"]["first_both_targets_new_response_count"] == 2
    assert row["observed_endpoints"]["reserved_panel_observed_positive"] == 1

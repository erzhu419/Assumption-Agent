"""A failed joint revision must not silently restart member-only trials."""

from copy import deepcopy
import json
from pathlib import Path

from hegel_machine.cognition import world_reference_recalibration as controller
from hegel_machine.cognition.world_response_hypotheses import propose_response_research
from test_world_response_joint_revision import PATH, X, X2, X3, stalled_state


def joint_revision(*, executed, exhausted=False):
    state, estimators = stalled_state(X, X2, X3)
    trial = propose_response_research(state, estimators, PATH)
    assert trial["status"] == "conditional_joint_response_trial"
    state["response_research"] = deepcopy(trial)
    result = trial
    count = len(trial["conditional_simplification"]["joint_revision"]["models"]) if exhausted else 1
    for index in range(count if executed else 0):
        state["response_settings"].append({
            "crystal": result["selected_number"], "frequency": result["selected_setting"],
            "activated": False, "evidence": [102 + index]})
        result = propose_response_research(state, estimators, PATH)
        state["response_research"] = deepcopy(result)
        if result["status"] != "conditional_joint_response_trial":
            break
    result = propose_response_research(state, estimators, PATH)
    result["question_id"] = "reference_response_4_post_revision"
    state["response_research"] = result
    return state, estimators, trial


def controller_context(state, estimators, tmp_path, monkeypatch, *, live=False, number=4):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(controller, "ESTIMATORS", estimators)
    monkeypatch.setattr(controller, "STATE", tmp_path / "shared/state.json")
    Path(PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(PATH).write_text("# Existing research program retained during inquiry.\n")
    rows = state["research_table"]["references"] + state["research_table"]["targets"]
    state["expanded_features"] = sorted({key for row in rows for key in row["measurements"]})
    state["rejected"] = {}
    memory = {"program_arguments": {}, "model_available": live,
              "limits": {"max_model_calls": 4 if live else 0},
              "costs": {"model_calls": 0}}
    request = controller._extend_hypotheses(state, memory, number, PATH)
    assert request["arguments"]["operator"] == "reference_response"
    state["response_context"] = deepcopy(request["arguments"]["context"])
    return memory


def test_exhausted_joint_revision_does_not_resume_broad_member_elimination(tmp_path, monkeypatch):
    state, estimators, _ = joint_revision(executed=True, exhausted=True)
    response = state["response_research"]
    assert response["status"] == "family_research_needed"
    assert response["selection"] is None
    fallback = response["operational_fallback"]
    assert fallback["selected_experiment"]["family"] == "full"
    assert fallback.get("guaranteed_nonlocal_eliminations", 0) == 0
    memory = controller_context(state, estimators, tmp_path, monkeypatch)
    before = deepcopy(state)

    decision = controller._extend_hypotheses(state, memory, 4, PATH)

    # Retain the unresolved full explanation. Any further operational search
    # must respect the explicit conditional support, rather than resuming this
    # broad sample merely because no model is available.
    selected = decision.get("response_trial")
    if selected is not None:
        broad = fallback["selected_experiment"]
        assert (selected.get("number", 4), selected["prediction"]) != (
            broad["number"], broad["prediction"])
    else:
        assert "joint prediction was refuted" in decision["stop"]
        assert "exhausted" not in decision["stop"]
    assert state["response_settings"] == before["response_settings"]
    assert response["hypotheses"] == before["response_research"]["hypotheses"]
    conditional = response["family_research"]["research_obligation"]["conditional_operational_search"]
    assert conditional["remaining_settings"] > 0
    assert conditional["assumption"] == response["conditional_simplification"]["assumption"]


def test_unexecuted_joint_revision_remains_an_actual_controller_trial(tmp_path, monkeypatch):
    state, estimators, trial = joint_revision(executed=False)
    memory = controller_context(state, estimators, tmp_path, monkeypatch)
    responses = deepcopy(state["response_settings"])

    decision = controller._extend_hypotheses(state, memory, 4, PATH)

    selected = decision["response_trial"]
    assert (selected["number"], selected["prediction"]) == (
        trial["selected_number"], trial["selected_setting"])
    assert state["response_settings"] == responses
    assert state["response_research"]["family_research"]["selected_test"] == {
        "crystal": selected["number"], "frequency": selected["prediction"]}


def test_surviving_joint_batch_member_remains_an_actual_controller_trial(tmp_path, monkeypatch):
    state, estimators, first_trial = joint_revision(executed=True)
    response = state["response_research"]
    assert response["status"] == "conditional_joint_response_trial"
    expected = response["selection"]["selected_experiment"]
    assert expected["prediction"] != first_trial["selected_setting"]
    memory = controller_context(state, estimators, tmp_path, monkeypatch)
    responses = deepcopy(state["response_settings"])

    decision = controller._extend_hypotheses(state, memory, 4, PATH)

    selected = decision["response_trial"]
    assert (selected["number"], selected["prediction"]) == (
        expected["number"], expected["prediction"])
    assert state["response_settings"] == responses
    assert state["response_research"]["family_research"]["selected_test"] == {
        "crystal": selected["number"], "frequency": selected["prediction"]}


def test_exhausted_joint_revision_returns_existing_obligation_to_live_research(tmp_path, monkeypatch):
    state, estimators, _ = joint_revision(executed=True, exhausted=True)
    memory = controller_context(state, estimators, tmp_path, monkeypatch, live=True)
    before = deepcopy(state)

    decision = controller._extend_hypotheses(state, memory, 4, PATH)

    assert decision["arguments"]["operator"] == "inquire"
    assert decision["arguments"]["max_model_calls"] == 4
    snapshot = json.loads(Path("shared/reference_extension_4.json").read_text())
    assert snapshot["family_research"] == before["response_research"]["family_research"]
    assert snapshot["operational_fallback"] == before["response_research"]["operational_fallback"]
    assert state == before


def test_branch_proven_nonlocal_experiment_remains_executable(tmp_path, monkeypatch):
    from test_world_response_consequences import LINEAR, assess, experiment, fixture

    state = fixture(panel_crystal=4, panel_frequency=11.)
    experiments = [{**item, "family": LINEAR["id"]} for item in (
        experiment("target", 3, 14.), experiment("reference", 1, 11.))]
    result = assess(state, experiments=experiments)
    result["question_id"] = "reference_response_nonlocal"
    state["response_research"] = result
    value = next(row for row in result["family_research"]["experiment_values"]
                 if row["experiment_id"] == "reference")
    assert value["by_hypothetical_outcome"] == [1, 1]
    memory = controller_context(state, [LINEAR], tmp_path, monkeypatch, number=3)

    decision = controller._extend_hypotheses(state, memory, 3, PATH)

    selected = decision["response_trial"]
    assert (selected["number"], selected["prediction"]) == (1, 11.)


def test_stalled_support_compares_and_executes_existing_reference_boundary(tmp_path, monkeypatch):
    from hegel_machine.cognition.world_response_consequences import (
        assess_response_consequences, select_response_grid_trial,
    )
    from hegel_machine.cognition.world_response_hypotheses import _propose_response_members
    from test_world_response_projection import ESTIMATOR, GRID, reference_history

    rows = [{"crystal": number, "measurements": {
        "x": {"value": value, "evidence": number}}}
        for number, value in ((1, 0.), (2, 1.), (3, 3.))]
    for row, setting in zip(rows, (0., 2.)):
        row["frequency"] = {"value": setting, "evidence": row["crystal"]}
    state = {"research_table": {"references": rows[:2], "targets": rows[2:]},
        "response_settings": reference_history(), "target_numbers": [3], "completed": [],
        "control_grid": GRID, "control_grids": {"1": GRID, "2": GRID}}
    members = _propose_response_members(state, [ESTIMATOR], PATH)
    assert any((item["number"], item["prediction"]) == (1, .5)
               for item in members["experiments"])
    assessed = assess_response_consequences(state, [ESTIMATOR], members)
    # Start at the controller's documented stalled decision. The candidates,
    # acquired observations and both-outcome evaluations remain real.
    assessed.update(status="family_research_needed", selection=None,
                    selected_number=None, selected_setting=None)
    assessed["family_research"].update(no_progress_cycles=2,
        research_obligation={"reason": "completed_tests_changed_members_without_task_or_nonlocal_progress"})
    before = deepcopy(state)

    result = select_response_grid_trial(state, [ESTIMATOR], assessed)

    assessment = result["family_research"]["operational_search"]
    reference = next(item for item in assessment["experiments"]
                     if (item["crystal"], item["frequency"]) == (1, .5))
    assert reference["guaranteed_search_reduction"] == 2
    assert min(reference["reductions_by_outcome"]) == 2
    target = next(item for item in assessment["experiments"] if item["crystal"] == 3)
    assert target["guaranteed_search_reduction"] == 1
    assert state == before
    result["question_id"] = "reference_response_boundary"
    state["response_research"] = result
    memory = controller_context(state, [ESTIMATOR], tmp_path, monkeypatch, number=3)
    decision = controller._extend_hypotheses(state, memory, 3, PATH)
    selected = decision["response_trial"]
    assert (selected["number"], selected["prediction"]) == (1, .5)


def test_completed_target_boundary_restores_paid_success_without_confirming_failure(tmp_path, monkeypatch):
    from hegel_machine.cognition.world_response_hypotheses import _select_response_experiment
    from test_world_multi_object_response import _pool
    from test_world_reference_hypothesis import ReferenceLab

    state, members = _pool()
    state.update(completed=[5], rejected={})
    state["control_grids"]["4"]["upper"] = 10.
    state["control_grids"]["5"]["step"] = 1.
    state["response_settings"] = [{"crystal": 5, "frequency": 2.25,
                                   "activated": True, "evidence": [0]}]
    # Both explanations retain the acquired positive; only the second says
    # that the completed object's band also extends to the adjacent setting.
    members["hypotheses"][1].update(upper=1.)
    members["hypotheses"][1]["predictions_by_crystal"]["5"] = 2.25
    result = _select_response_experiment(members, state, PATH)
    assert result["target_number"] == 4
    experiment = result["selection"]["selected_experiment"]
    assert (experiment["number"], experiment["prediction"]) == (5, 3.25)
    assert all(item["restore_frequency"] == 2.25 for item in result["experiments"]
               if item["number"] == 5)

    monkeypatch.setattr(controller, "STATE", tmp_path / "controller_state.json")
    state["response_research"] = result
    lab = ReferenceLab()
    lab.installed.add(5)
    lab.frequencies[5] = lab.required[5] = 2.25
    records, questions = [], []
    selected = {**experiment, "response_question": "completed_target_boundary",
                "question_id": "completed_target_boundary_physical"}
    action = controller._trial_action(state, selected, 4, {"5": 205}, {"5": 105},
                                     lab.public_observation(), records, None)
    memory = {"program_path": PATH, "questions": questions}
    for _ in range(12):
        for item in action["actions"]:
            records.append({"index": len(records), **lab.act(item)})
        saved = json.loads(controller.STATE.read_text())
        if saved.get("test"):
            answer = controller.answer({"current_observation": lab.public_observation()}, records, {})
            if answer["status"] == "refuted":
                questions[:] = [{"id": selected["question_id"], **answer}]
        action = controller.select_actions({"current_observation": lab.public_observation()}, records, memory)
        saved = json.loads(controller.STATE.read_text())
        if saved.get("setting_recoveries"):
            break

    assert saved["setting_recoveries"][-1]["restored"]
    assert lab.active(5) and lab.frequencies[5] == 2.25
    assert saved["completed"] == [5] and saved["rejected"] == {}
    assert saved["tests"][-1]["status"] == questions[0]["status"] == "refuted"
    assert questions[0]["answer"]["actual_frequency"] == 3.25
    assert questions[0]["answer"]["response_feedback"]["remaining_hypotheses"] == ["a"]

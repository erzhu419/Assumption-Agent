"""Returned physical counterexamples revise experiments, not whole scientific families."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_child_return import CHILD, CONTRAST, acquire_child, return_answer, reviewed_parent
from test_discoveryworld_parent_research import question
from test_discoveryworld_prerequisite import ControlLab
from test_discoveryworld_reference_history import record


def rejected_comparison(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    agent.limits = {**agent.limits, "max_model_calls": 1}
    source = CONTRAST.replace("'chosen_dialog_option_int': 2", "'chosen_dialog_option_int': 1")
    agent.provider = ScriptedProvider([("run", {"operator": "contrast", "path": "shared/first_test.py",
                                                "code": source, "question": CHILD})])
    result = agent.run(lab)
    assert result["status"] == "model_call_budget_exhausted"
    assert question(agent, "response")["status"] == "refuted"
    return agent, lab


def test_changed_physical_counterexample_returns_to_parent_with_its_actual_trial(tmp_path):
    agent, lab = rejected_comparison(tmp_path)
    before = deepcopy(agent.observations)
    child = deepcopy(question(agent, "response"))
    parent = question(agent)
    parent_source = (tmp_path / parent["parent_program_path"]).read_text()

    focus = agent._request()["research_focus"]

    assert focus["stage"] == "child_return" and focus["comparison_status"] == "refuted"
    assert focus["path"] == parent["parent_program_path"]
    assert focus["comparison_question"] == child
    assert focus["changed_child_ids"] == ["response"]
    assert focus["prior_trials"] == agent._state()["experiment_trials"]
    assert len(focus["prior_trials"]) == 1
    assert focus["prior_trials"][0]["observation_indices"] == child["acquired_observations"]
    assert parent["status"] == "reopened" and "answer" not in parent
    assert question(agent, "instrument")["status"] == "supported"
    assert question(agent, "response") == child
    assert agent.observations == before and lab.environment_steps == 3
    assert (tmp_path / parent["parent_program_path"]).read_text() == parent_source
    assert agent._request()["research_focus"] == focus


def test_acknowledging_unchanged_counterexample_does_not_force_another_test(tmp_path):
    agent, lab = rejected_comparison(tmp_path)
    assert agent._request()["research_focus"]["stage"] == "child_return"
    agent._execute("run", {"code": "print('The failed candidate remains refuted; more work is optional')"}, lab)
    assert not agent._request().get("research_focus")
    assert not agent._request().get("research_focus")
    assert question(agent, "response")["status"] == "refuted"
    assert lab.environment_steps == 3


@pytest.mark.parametrize("unselected_choice", [False, True])
def test_refuted_operation_without_selected_physical_comparison_keeps_ordinary_return(tmp_path, unselected_choice):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab, "refuted", "This operation is unavailable")
    if unselected_choice:
        question(agent, "response")["experiment_choice"] = {"selected_experiment": None}
    focus = agent._request()["research_focus"]
    assert focus["stage"] == "child_return"
    assert focus["changed_child_ids"] == ["response"]
    assert "comparison_question" not in focus
    assert "answer" not in question(agent)


REVISION = """def design(spec, observations, memory):
    q = next(q for q in memory['questions'] if q['id'] == QUESTION_ID)
    assert q['question'] == 'Does the alternative setting restore the physical response?'
    assert len(memory['experiment_trials']) == 1
    assert memory['answers']['instrument']['status'] == 'supported'
    return {'hypotheses': [{'id': 'alternative', 'claim': 'The alternative setting activates the device'},
                          {'id': 'condition', 'claim': 'A further condition prevents activation'}],
            'experiments': [
                {'id': 'menu_only', 'outcome': 'The menu opens',
                 'actions': [{'action': 'TALK', 'arg1': 7}],
                 'predictions': {'alternative': True, 'condition': True}},
                {'id': 'alternative_setting', 'outcome': 'The device becomes active',
                 'actions': [{'action': 'TALK', 'arg1': 7}, {'chosen_dialog_option_int': 2}],
                 'predictions': {'alternative': True, 'condition': False}}]}

def answer(spec, observations, memory):
    q = next(q for q in memory['questions'] if q['id'] == QUESTION_ID)
    chosen = q['experiment_choice']['selected_experiment']
    assert chosen['id'] == 'alternative_setting'
    assert observations[-1]['action'] == chosen['actions'][-1]
    active = observations[-1]['public_objects'][0]['name'] == 'device (active)'
    return {'status': 'supported' if active else 'refuted',
            'answer': {'observed_active': active, 'scope': 'This setting on this device'},
            'evidence': [observations[-1]['index']]}
"""


@pytest.mark.parametrize("field, revised_id", [("question", "new_response"), ("research_question", "response")])
def test_authored_comparison_question_survives_explicit_revision_and_real_experiment(tmp_path, field, revised_id):
    agent, lab = rejected_comparison(tmp_path)
    old_trials = deepcopy(agent._state()["experiment_trials"])
    parent = question(agent)
    parent_path = parent["parent_program_path"]
    old_source = (tmp_path / parent_path).read_text()
    focus = agent._request()["research_focus"]
    assert focus["stage"] == "child_return" and focus["path"] == parent_path
    revised = {"id": revised_id, "question": "Does the alternative setting restore the physical response?",
               "parent_id": "task", "depends_on": ["instrument"]}
    source = f"QUESTION_ID = {revised_id!r}\n" + REVISION
    revised_path = "shared/revised_comparison.py"
    saved = agent._execute("write", {"path": revised_path, "code": source, field: revised}, lab)
    assert saved["written"] == revised_path and "operator" not in saved
    assert lab.environment_steps == 3
    result = agent._execute("run", {"operator": "contrast", "path": revised_path,
                                    "question": {"id": revised_id}}, lab)

    assert result["operator"] == "contrast" and result["question_id"] == revised_id
    assert result["diagnosis"] == "discriminating_experiment_available"
    assert result["selected_experiment"]["id"] == "alternative_setting"
    changed = question(agent, revised_id)
    assert changed["question"] == revised["question"]
    assert changed["parent_id"] == "task" and changed["depends_on"] == ["instrument"]
    assert changed["status"] == "supported" and changed["answer"]["observed_active"] is True
    assert changed["evidence"] == [4] and lab.environment_steps == 5
    assert result["experiment"]["after_observation_results"][0]["analysis_is_agent_authored"]
    assert parent["parent_program_path"] == parent_path and (tmp_path / parent_path).read_text() == old_source
    assert "answer" not in parent and question(agent, "instrument")["status"] == "supported"
    assert agent._state()["experiment_trials"][:len(old_trials)] == old_trials
    if revised_id != "response":
        assert question(agent, "response")["status"] == "refuted"


def test_all_paid_measurement_records_reach_both_arms_including_old_and_failed_reads(tmp_path):
    source = [
        record(0, action={"action": "USE", "arg1": 12, "arg2": 9}, objects=[
            {"uuid": 12, "name": "microscope", "description": "Public instrument"},
            {"uuid": 9, "name": "crystal", "description": "Public sample"}]),
        record(1, action={"action": "READ", "arg1": 21}),
        record(2, action={"action": "USE", "arg1": 12, "arg2": 9}),
    ]
    source[0]["observation"]["lastActionMessage"] = "The crystal has a gap of 47.52 nm."
    source[0]["observation"]["extended_action_message"] = "Instrument report: the crystal has a gap of 47.52 nm."
    source[1]["observation"]["lastActionMessage"] = "The instrument manual describes its measurement procedure."
    source[2]["feedback"] = {"success": False, "message": "Cannot reach that sample"}
    source[2]["observation"]["lastActionMessage"] = "Cannot reach that sample"
    source.extend(record(index, objects=[], action={"action": "MOVE_DIRECTION", "arg1": "north"})
                  for index in range(3, 20))
    original = deepcopy(source)
    views = []
    for arm in ("hegel", "code_archive"):
        workspace = tmp_path / arm
        agent = DiscoveryWorldAgent(ScriptedProvider([]), workspace, arm=arm, limits=LIMITS)
        agent.lab = ControlLab()
        agent._question({"id": "task", "question": "Explain the sample response"})
        agent.observations = deepcopy(source)
        agent._prerequisite_checked_through = len(source) - 1
        context = prepare_code_context(workspace, agent._request(), schema={}, instructions="")
        view = context["evidence"]["measurement_history"]
        assert [row["index"] for row in view] == [0, 1, 2]
        assert not {0, 1, 2} & {row["index"] for row in context["evidence"]["recent_action_results"]}
        assert view == [agent._action_observation(row) for row in source[:3]]
        assert view[0]["lastActionMessage"] == source[0]["observation"]["lastActionMessage"]
        assert view[0]["extended_action_message"] == source[0]["observation"]["extended_action_message"]
        assert view[2]["feedback"]["success"] is False
        assert all("status" not in row and "target" not in row for row in view)
        assert agent.observations == original
        views.append(view)
    assert views[0] == views[1] and source == original

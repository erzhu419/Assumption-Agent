"""A rejected authored draft returns to experiment choice before it has a frozen trial."""

from copy import deepcopy

import pytest

from test_discoveryworld_agent import ScriptedProvider
from test_discoveryworld_child_return import CHILD, CONTRAST, acquire_child, return_answer, reviewed_parent
from test_discoveryworld_parent_research import question


DRAFT = "shared/saved_comparison.py"
ANALYSIS = "shared/paid_response.py"
DIRECT_ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    active = record['public_objects'][0]['name'] == 'device (active)'
    return {'status': 'supported' if active else 'refuted',
            'answer': {'observed_active': active}, 'evidence': [record['index']]}
"""
MARKER = "from pathlib import Path\nPath('draft_executed.txt').write_text('executed')\n"


def saved_and_refuted(tmp_path, route="ordinary"):
    agent, lab = reviewed_parent(tmp_path)
    source = MARKER + CONTRAST.replace("'chosen_dialog_option_int': 2", "'chosen_dialog_option_int': 1")
    write = {"action": "write", "arguments": {"path": DRAFT, "code": source, "question": CHILD}}
    first = (write["action"], write["arguments"])
    if route == "parent_program":
        writer = ("def select_actions(spec, observations, memory):\n"
                  "    if memory['program_results']: return {'stop': 'The comparison was saved'}\n"
                  f"    return {write!r}\n")
        agent._execute("write", {"path": "shared/write_comparison.py", "code": writer}, lab)
        first = ("run", {"operator": "procedure", "path": "shared/write_comparison.py", "max_steps": 3})
    agent.limits = {**agent.limits, "max_model_calls": 3}
    agent.provider = ScriptedProvider([
        first,
        ("query", {"question": {"id": "response"},
                   "actions": [{"action": "TALK", "arg1": 7}, {"chosen_dialog_option_int": 1}]}),
        ("evaluate", {"path": ANALYSIS, "code": DIRECT_ANALYSIS, "question_id": "response"}),
    ])
    result = agent.run(lab)
    assert result["status"] == "model_call_budget_exhausted"
    assert question(agent, "response")["status"] == "refuted"
    assert not (tmp_path / "draft_executed.txt").exists()
    return agent, lab


@pytest.mark.parametrize("route", ["ordinary", "parent_program"])
def test_saved_draft_counterexample_returns_once_without_fabricating_an_executed_trial(tmp_path, route):
    agent, lab = saved_and_refuted(tmp_path, route)
    child = deepcopy(question(agent, "response"))
    parent = question(agent)
    parent_source = (tmp_path / parent["parent_program_path"]).read_bytes()
    analysis_source = (tmp_path / ANALYSIS).read_bytes()
    observations = deepcopy(agent.observations)
    executions = agent.program_executions

    focus = agent._request()["research_focus"]

    assert focus["kind"] == "parent_research" and focus["stage"] == "contrast"
    assert focus["comparison_status"] == "refuted" and focus["comparison_question"] == child
    assert focus["changed_child_ids"] == ["response"]
    assert focus["available_designs"] == [{"action": "run", "arguments": {
        "operator": "contrast", "path": DRAFT, "question": {"id": "response"}}}]
    assert focus["answer_observations"] == [agent._action_observation(observations[2])]
    assert focus["prior_trials"] == [] and agent._state()["experiment_trials"] == []
    assert "experiment_choice" not in child and "design_path" not in child
    assert question(agent, "response") == child and "answer" not in parent
    assert question(agent, "instrument")["status"] == "supported"
    assert (tmp_path / parent["parent_program_path"]).read_bytes() == parent_source
    assert (tmp_path / ANALYSIS).read_bytes() == analysis_source
    assert agent._request()["research_focus"] == focus
    assert agent.program_executions == executions and agent.observations == observations
    assert lab.environment_steps == 3 and not (tmp_path / "draft_executed.txt").exists()

    agent._execute("run", {"code": "print('Keep the refutation and defer another experiment')"}, lab)
    assert not agent._request().get("research_focus")
    assert not agent._request().get("research_focus")
    assert lab.environment_steps == 3 and question(agent, "response")["status"] == "refuted"


@pytest.mark.parametrize("destination", ["generated", "retained"])
def test_writing_revised_comparison_at_generated_or_retained_path_executes_actual_research(tmp_path, destination):
    agent, lab = saved_and_refuted(tmp_path)
    parent = question(agent)
    parent_path = parent["parent_program_path"]
    parent_source = (tmp_path / parent_path).read_bytes()
    analysis_source = (tmp_path / ANALYSIS).read_bytes()
    focus = agent._request()["research_focus"]
    path = focus["path"] if destination == "generated" else focus["available_designs"][0]["arguments"]["path"]
    revised_question = {"id": "response", "question": "Does revised setting 8 restore the active response?",
                        "parent_id": "task", "depends_on": ["instrument"]}
    result = agent._execute("write", {"path": path, "code": MARKER + CONTRAST,
                                      "question": revised_question}, lab)

    assert result["operator"] == "contrast" and result["path"] == path
    assert result["diagnosis"] == "discriminating_experiment_available"
    assert result["selected_experiment"]["id"] == "new_setting"
    assert result["experiment"]["observation_indices"] == [3, 4]
    child = question(agent, "response")
    assert child["status"] == "supported" and child["evidence"] == [4]
    assert child["question"] == revised_question["question"] and child["depends_on"] == ["instrument"]
    assert child["analysis_path"] == path and child["answer"] == {"active": True}
    assert (tmp_path / ANALYSIS).read_bytes() == analysis_source
    assert parent["parent_program_path"] == parent_path
    assert (tmp_path / parent_path).read_bytes() == parent_source
    assert (tmp_path / "draft_executed.txt").read_text() == "executed"
    assert lab.environment_steps == 5 and "answer" not in parent


def test_writing_an_unrelated_path_only_saves_without_executing_the_comparison(tmp_path):
    agent, lab = saved_and_refuted(tmp_path)
    agent._request()
    executions = agent.program_executions
    child = deepcopy(question(agent, "response"))
    saved = agent._execute("write", {"path": "shared/another_draft.py", "code": MARKER + CONTRAST,
                                      "question": {"id": "response"}}, lab)
    assert saved["written"] == "shared/another_draft.py" and "operator" not in saved
    assert saved["contrast_action"]["arguments"]["path"] == "shared/another_draft.py"
    assert agent.program_executions == executions and lab.environment_steps == 3
    assert question(agent, "response") == child
    assert not (tmp_path / "draft_executed.txt").exists()


def test_refuted_operation_without_saved_comparison_keeps_generic_parent_return(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab, "refuted", "The requested operation is unsupported")
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "parent_research" and focus["stage"] == "child_return"
    assert "comparison_question" not in focus
    assert agent._question_design_actions()["response"] == []
    assert lab.environment_steps == 2 and "answer" not in question(agent)

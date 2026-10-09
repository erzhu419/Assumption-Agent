"""An analyzed experiment returns direction choice to its actual parent."""

from copy import deepcopy
import json

import pytest

from test_discoveryworld_agent import ScriptedProvider
from test_discoveryworld_child_return import CHILD, CONTRAST, reviewed_parent
from test_discoveryworld_parent_research import question


TRIAL_PATH = "shared/negative_trial.py"


def executed_child(tmp_path, verdict):
    agent, lab = reviewed_parent(tmp_path)
    source = CONTRAST.replace("'chosen_dialog_option_int': 2", "'chosen_dialog_option_int': 1")
    if verdict == "inconclusive":
        source = source.replace("'supported' if active else 'refuted'", "'inconclusive'")
    agent.limits["max_model_calls"] = 1
    agent.provider = ScriptedProvider([("run", {"operator": "contrast", "path": TRIAL_PATH,
                                               "code": source, "question": CHILD})])
    agent.run(lab)
    assert question(agent, CHILD["id"])["status"] == verdict
    agent.status = "running"
    agent.limits["max_model_calls"] = 30
    return agent, lab


@pytest.mark.parametrize("verdict", ["refuted", "inconclusive"])
@pytest.mark.parametrize("unreviewed_stop", [False, True])
def test_executed_child_preserves_parent_review_evidence_and_callable_comparisons(
        tmp_path, verdict, unreviewed_stop):
    agent, lab = executed_child(tmp_path, verdict)
    parent = question(agent)
    path = parent["parent_program_path"]
    if unreviewed_stop:
        # A path-only program invocation records a new stop without consuming
        # the explicit parent-question review of the child's new answer.
        agent._query({"path": path}, lab)
    previous = agent._latest_program_result(path)
    child = deepcopy(question(agent, CHILD["id"]))
    observations = deepcopy(agent.observations)
    executions = agent.program_executions
    source = (tmp_path / path).read_text()

    request = agent._request()
    focus = request["research_focus"]

    assert focus["stage"] == ("review" if unreviewed_stop else "child_return")
    assert focus["origin"] == ("host_parent_program_review" if unreviewed_stop
                                else "host_parent_research_continuation")
    assert focus["path"] == path and focus["question"]["id"] == parent["id"]
    assert focus["comparison_status"] == verdict and focus["comparison_question"] == child
    assert focus["changed_child_ids"] == [CHILD["id"]]
    assert {q["id"] for q in focus["returned_children"]} == {"instrument", CHILD["id"]}
    assert next(q for q in focus["returned_children"] if q["id"] == "instrument")["status"] == "supported"
    assert focus["answer_observations"] == [agent._action_observation(observations[index])
                                             for index in child["evidence"]]
    assert focus["prior_trials"] == agent._state()["experiment_trials"]
    design = {"action": "run", "arguments": {"operator": "contrast", "path": TRIAL_PATH,
                                               "question": {"id": CHILD["id"]}}}
    assert focus["available_designs"] == [design]
    decision = json.loads(request["files"]["method_decision.json"])
    by_id = {q["id"]: q for q in decision["questions"]}
    assert by_id[parent["id"]]["last_program_result"] == previous
    assert by_id[parent["id"]]["procedure_action"]["arguments"]["path"] == path
    assert by_id[CHILD["id"]]["contrast_actions"] == [design]
    if unreviewed_stop:
        assert focus["previous_execution"] == previous
    assert agent._request()["research_focus"] == focus
    assert agent.program_executions == executions and agent.observations == observations
    assert (tmp_path / path).read_text() == source
    assert lab.environment_steps == 3 and "answer" not in parent


@pytest.mark.parametrize("verdict", ["refuted", "inconclusive"])
def test_independent_repair_of_real_trial_returns_to_parent_without_requiring_outer_trial_event(tmp_path, verdict):
    agent, lab = reviewed_parent(tmp_path)
    source = CONTRAST.replace("    record = observations[-1]", "    raise ValueError('Interpret the actual result')\n    record = observations[-1]")
    result = agent._execute("run", {"operator": "contrast", "path": TRIAL_PATH,
                                    "code": source, "question": CHILD}, lab)
    assert "after_observation_error" in result["experiment"]
    assert not agent._state()["experiment_trials"]  # Direct execution has no enclosing model event.
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis" and focus["evidence"] == [1, 2]
    analysis = ("def answer(spec, observations, memory):\n"
                f"    return {{'status': {verdict!r}, 'answer': 'The observed result narrows this comparison', "
                "'evidence': [observations[-1]['index']]}\n")

    agent._execute("write", {"path": focus["path"], "code": analysis}, lab)
    returned = agent._request()["research_focus"]

    assert returned["stage"] == "child_return" and returned["comparison_status"] == verdict
    assert returned["path"] == question(agent)["parent_program_path"]
    assert returned["prior_trials"] == []
    assert returned["comparison_question"]["evidence"] == [2]
    assert lab.environment_steps == 3
    assert (tmp_path / TRIAL_PATH).read_text() == source


def test_local_parent_can_choose_computation_after_return_without_repeating_child_experiment(tmp_path):
    agent, lab = executed_child(tmp_path, "refuted")
    path = question(agent)["parent_program_path"]
    source = (tmp_path / path).read_text()
    executions = agent.program_executions
    provider = ScriptedProvider([
        ("run", {"code": "print('Compare the recorded explanations before choosing another experiment')"}),
        ("finish", {"lesson": "The parent remains unresolved; no new experiment was selected."}),
    ])
    agent.provider = provider

    result = agent._execute("run", {"operator": "inquire", "question": {"id": "task"},
                                   "max_model_calls": 2}, lab)

    first, second = provider.requests
    assert first["llm_inquiry"]["question_id"] == "task"
    assert first["research_focus"]["stage"] == "child_return"
    assert first["research_focus"]["path"] == path
    assert not second.get("research_focus")
    assert result["status"] == "inquiry_finished" and result["model_calls"] == 2
    assert question(agent)["reviewed_child_answers"][CHILD["id"]]["status"] == "refuted"
    assert lab.environment_steps == 3 and agent.program_executions == executions
    assert (tmp_path / path).read_text() == source


def test_old_evidence_does_not_mark_current_selected_experiment_as_analyzed(tmp_path):
    agent, lab = executed_child(tmp_path, "refuted")
    child = question(agent, CHILD["id"])
    assert child["experiment_choice"]["started_before_observation"] == 1
    agent._record_answer(child, {"status": "inconclusive", "answer": "The earlier instrument result is insufficient",
                                 "evidence": [0]})

    focus = agent._request()["research_focus"]

    assert focus["stage"] == "contrast"
    assert focus["answer_observations"] == [agent._action_observation(agent.observations[0])]
    assert lab.environment_steps == 3

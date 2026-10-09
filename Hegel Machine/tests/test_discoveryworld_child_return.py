"""Fresh authored child answers return to an already reviewed parent program."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_parent_research import question
from test_discoveryworld_parent_review import stopped


CHILD = {"id": "response", "question": "Does the setting produce the expected response?",
         "parent_id": "task"}
PATH = "shared/analysis_response.py"


def reviewed_parent(tmp_path):
    agent, lab = stopped(tmp_path)
    agent._request()
    agent._execute("run", {"code": "print('Inspect before continuing')"}, lab)
    assert not agent._request().get("research_focus")
    return agent, lab


def return_answer(agent, lab, status="inconclusive", interpretation="Condition or law remains unresolved"):
    source = ("def answer(spec, observations, memory):\n"
              f"    return {{'status': {status!r}, 'answer': {interpretation!r}, 'evidence': [1]}}\n")
    agent._execute("write", {"path": PATH, "code": source}, lab)


def acquire_child(agent, lab):
    agent._execute("query", {"question": CHILD, "actions": [{"action": "MEASURE"}]}, lab)
    assert agent._request()["research_focus"]["kind"] == "analysis"


@pytest.mark.parametrize("status", ["supported", "refuted", "inconclusive"])
def test_new_answer_returns_after_old_program_review_and_only_once(tmp_path, status):
    agent, lab = reviewed_parent(tmp_path)
    old_step = question(agent)["reviewed_program_step"]
    acquire_child(agent, lab)
    return_answer(agent, lab, status)

    first = agent._request()["research_focus"]
    assert first == agent._request()["research_focus"]
    assert first["kind"] == "parent_research" and first["stage"] == ("contrast" if status == "inconclusive" else "child_return")
    assert first["changed_child_ids"] == ["response"]
    assert "previous_execution" not in first
    assert question(agent)["status"] == "reopened"
    assert question(agent)["reviewed_program_step"] == old_step

    # The parent may choose a direct new experiment; no program rewrite is forced.
    agent._execute("query", {"question": {"id": "task"},
                             "actions": [{"action": "TALK", "arg1": 7}]}, lab)
    assert question(agent)["reviewed_child_answers"]["response"]["status"] == status
    assert not agent._request().get("research_focus")
    assert not agent._request().get("research_focus")
    assert lab.environment_steps == 3
    assert "answer" not in question(agent)


def test_changed_interpretation_of_same_paid_data_returns_again(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab)
    agent._request()
    agent._execute("run", {"code": "print('Assess the returned uncertainty')"}, lab)
    assert not agent._request().get("research_focus")
    first = deepcopy(question(agent, "response"))

    return_answer(agent, lab, "refuted", "The measured response contradicts this candidate")
    agent._execute("evaluate", {"path": PATH, "question_id": "response"}, lab)
    revised = question(agent, "response")
    assert revised["evidence"] == first["evidence"]
    assert revised["reviewed_through_observation"] == first["reviewed_through_observation"]
    assert agent._request()["research_focus"]["changed_child_ids"] == ["response"]
    agent._execute("run", {"code": "print('Use revised verdict')"}, lab)
    agent._execute("evaluate", {"path": PATH, "question_id": "response"}, lab)
    assert not agent._request().get("research_focus")
    assert lab.environment_steps == 2


def test_new_acquisition_precedes_outstanding_parent_answer_review(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab)
    assert agent._request()["research_focus"]["stage"] == "contrast"
    agent._execute("query", {"question": {"id": "response"},
                             "actions": [{"action": "MEASURE"}]}, lab)
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis" and focus["evidence"] == [2]
    assert question(agent, "response")["reviewed_through_observation"] == 1


@pytest.mark.parametrize("block", ["explicit_dependency", "binding", "code"])
def test_child_return_keeps_other_route_constraints(tmp_path, block):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab, "refuted")
    if block == "explicit_dependency":
        agent._question({"id": "task", "depends_on": ["response"]})
    elif block == "binding":
        question(agent)["binding_path"] = "binding.py"
    else:
        agent.agent_arm = "code_archive"
    assert not agent._request().get("research_focus")


def test_decode_retry_does_not_consume_returned_answer(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab)

    class Provider:
        def __init__(self):
            self.requests = []

        def next_action(self, request):
            self.requests.append(deepcopy(request))
            if len(self.requests) == 1:
                return {"action": "provider_error", "error": {"decode_error": "Invalid JSON"}}
            return {"action": "finish", "arguments": {"lesson": "Unresolved response"}}

    agent.provider = Provider()
    result = agent.run(lab)
    assert len(agent.provider.requests) == 2
    first, second = agent.provider.requests
    assert first["research_focus"] == second["research_focus"]
    assert first["research_focus"]["changed_child_ids"] == ["response"]
    assert "response" not in first["research_focus"]["question"]["reviewed_child_answers"]
    assert question(agent)["reviewed_child_answers"]["response"]["status"] == "inconclusive"
    assert result["costs"]["model_calls"] == 2 and lab.environment_steps == 2


def test_acknowledged_child_snapshots_survive_restore(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab)
    agent._request()
    agent._execute("run", {"code": "print('Keep uncertainty')"}, lab)
    assert not agent._request().get("research_focus")
    state = json.loads((tmp_path / "research_state.json").read_text())
    restored = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    restored.lab = lab
    restored.questions = state["questions"]
    restored.method_results = state["method_results"]
    restored.observations = deepcopy(agent.observations)
    assert not restored._request().get("research_focus")
    assert question(restored)["reviewed_child_answers"] == question(agent)["reviewed_child_answers"]


CONTRAST = """def design(spec, observations, memory):
    return {'hypotheses': [{'id': 'setting', 'claim': 'The new setting produces the response'},
                          {'id': 'condition', 'claim': 'An additional condition is required'}],
            'experiments': [{'id': 'new_setting', 'outcome': 'The device becomes active',
                'actions': [{'action': 'TALK', 'arg1': 7}, {'chosen_dialog_option_int': 2}],
                'predictions': {'setting': True, 'condition': False}}]}

def answer(spec, observations, memory):
    record = observations[-1]
    choice = next(q for q in memory['questions'] if q['id'] == 'response')['experiment_choice']
    assert choice['selected_experiment']['predictions'] == {'setting': True, 'condition': False}
    active = record['observation']['accessibleEnvironmentObjects'][0]['name'] == 'device (active)'
    return {'status': 'supported' if active else 'refuted',
            'answer': {'active': active}, 'evidence': [record['index']]}
"""


def contrast_return(tmp_path):
    agent, lab = reviewed_parent(tmp_path)
    acquire_child(agent, lab)
    return_answer(agent, lab)
    focus = agent._request()["research_focus"]
    assert focus["stage"] == "contrast"
    return agent, lab, focus


def test_inconclusive_return_executes_frozen_contrast_preserving_parent_program(tmp_path):
    agent, lab, focus = contrast_return(tmp_path)
    parent_path = question(agent)["parent_program_path"]
    old_source = (tmp_path / parent_path).read_text()
    observed_before_action = []
    act = lab.act

    def inspect(action):
        observed_before_action.append(deepcopy(question(agent, "response")))
        return act(action)

    lab.act = inspect
    result = agent._execute("write", {"path": focus["path"], "code": CONTRAST}, lab)
    assert result["operator"] == "contrast"
    assert result["selected_experiment"]["id"] == "new_setting"
    assert observed_before_action[0]["experiment_choice"]["selected_experiment"]["predictions"] == {
        "setting": True, "condition": False}
    assert observed_before_action[0]["evidence"] == [1]
    assert question(agent, "response")["status"] == "supported"
    assert question(agent, "response")["evidence"] == [3]
    assert question(agent)["parent_program_path"] == parent_path
    assert (tmp_path / parent_path).read_text() == old_source
    assert agent._request()["research_focus"]["stage"] == "child_return"
    assert lab.environment_steps == 4


def test_uninformative_design_consumes_return_without_buying_or_supporting(tmp_path):
    agent, lab, focus = contrast_return(tmp_path)
    source = CONTRAST.replace("'condition': False", "'condition': True")
    result = agent._execute("write", {"path": focus["path"], "code": source}, lab)
    assert result["diagnosis"] == "observational_ambiguity"
    assert result["selected_experiment"] is None
    assert question(agent, "response")["status"] == "inconclusive"
    assert question(agent, "response")["evidence"] == [1]
    assert not agent._request().get("research_focus")
    assert not agent._request().get("research_focus")
    assert lab.environment_steps == 2


@pytest.mark.parametrize("bad_source", ["def design(:\n", "def design(spec, observations, memory):\n return {}\n"])
def test_invalid_contrast_keeps_unconsumed_return_for_repair(tmp_path, bad_source):
    agent, lab, focus = contrast_return(tmp_path)
    with pytest.raises(ValueError):
        agent._execute("write", {"path": focus["path"], "code": bad_source,
                                 "notes": "preserved reasoning"}, lab)
    assert (tmp_path / "research_notes.md").read_text() == "preserved reasoning"
    assert "response" not in question(agent)["reviewed_child_answers"]
    assert "experiment_choice" not in question(agent, "response")
    assert agent._request()["research_focus"] == focus
    assert lab.environment_steps == 2
    repaired = agent._execute("write", {"path": focus["path"], "code": CONTRAST}, lab)
    assert repaired["experiment"]["question"]["status"] == "supported"


def test_valid_contrast_with_failed_answer_uses_existing_analysis_repair(tmp_path):
    agent, lab, focus = contrast_return(tmp_path)
    bad_analysis = CONTRAST.replace("    record = observations[-1]", "    raise ValueError('Interpret actual response')\n    record = observations[-1]")
    result = agent._execute("write", {"path": focus["path"], "code": bad_analysis}, lab)
    assert "after_observation_error" in result["experiment"]
    continuation = agent._request()["research_focus"]
    assert continuation["kind"] == "analysis" and continuation["evidence"] == [2, 3]
    assert continuation["path"] == "shared/analysis_response.py"
    assert (tmp_path / focus["path"]).read_text() == bad_analysis
    agent._execute("write", {"path": continuation["path"], "code": CONTRAST}, lab)
    assert question(agent, "response")["status"] == "supported"
    assert agent._request()["research_focus"]["stage"] == "child_return"
    assert lab.environment_steps == 4

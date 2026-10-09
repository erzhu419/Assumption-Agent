"""Authored forecasts select experiments; paid outcomes determine their answers."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider


PATH = "shared/physical_contrast.py"
QUESTION = {"id": "physical", "question": "Does the extra operation produce the predicted response?",
            "parent_id": "task", "depends_on": ["operation"]}
SOURCE = """def design(spec, observations, memory):
    assert memory['answers']['operation']['status'] == 'supported'
    return {'hypotheses': [{'id': 'extra', 'claim': 'The extra operation is necessary'},
                          {'id': 'law', 'claim': 'The response law is wrong'}],
            'experiments': [
                {'id': 'menu', 'outcome': 'The menu opens',
                 'actions': [{'action': 'TALK', 'arg1': 7}],
                 'predictions': {'extra': True, 'law': True}},
                {'id': 'physical', 'outcome': 'The device is active',
                 'actions': [{'action': 'USE', 'arg1': 8, 'arg2': 7}],
                 'predictions': {'extra': True, 'law': False}}]}

def answer(spec, observations, memory):
    question = next(item for item in memory['questions'] if item['id'] == 'physical')
    experiment = question['experiment_choice']['selected_experiment']
    assert experiment['predictions'] == {'extra': True, 'law': False}
    record = observations[-1]
    assert record['action'] == experiment['actions'][-1]
    outcome = record['observation'].get('physical_outcome')
    status = 'inconclusive' if outcome is None else ('supported' if outcome else 'refuted')
    return {'status': status, 'answer': {'observed_outcome': outcome,
            'tested_prediction': experiment['predictions']['extra']}, 'evidence': [record['index']]}
"""


class PhysicalLab:
    def __init__(self, outcome=False, failed=False):
        self.environment_steps = 0
        self.actions = []
        self.outcome, self.failed = outcome, failed
        self.before_experiment = None

    def public_observation(self):
        view = {"taskProgress": [{"description": "Explain the response", "completed": False}]}
        if self.actions and self.actions[-1].get("action") == "USE" and self.outcome is not None:
            view["physical_outcome"] = self.outcome
        return view

    def capabilities(self):
        return {"actions": ["TALK", "USE"]}

    def act(self, action):
        if action["action"] == "USE":
            self.before_experiment = deepcopy(self.agent.questions)
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        success = not (self.failed and action["action"] == "USE")
        return {"action": deepcopy(action), "feedback": {"success": success},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


def prepared(tmp_path, *, arm="hegel", outcome=False, failed=False):
    lab = PhysicalLab(outcome, failed)
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    agent.lab, lab.agent = lab, agent
    agent._question({"id": "task", "question": "Explain the response"})
    agent._execute("query", {"actions": [{"action": "TALK", "arg1": 7}], "question": {
        "id": "operation", "question": "Can the control menu open?", "parent_id": "task"}}, lab)
    agent._execute("write", {"path": "operation.py", "code": """def answer(spec, observations, memory):
    return {'status': 'supported', 'answer': {'menu_opens': observations[-1]['feedback']['success']},
            'evidence': [observations[-1]['index']]}
"""}, lab)
    agent._execute("evaluate", {"path": "operation.py", "question_id": "operation"}, lab)
    return agent, lab


def question(agent, qid):
    return next(item for item in agent.questions if item["id"] == qid)


def contrast(agent, lab, source=SOURCE):
    return agent._execute("run", {"operator": "contrast", "path": PATH,
                                  "code": source, "question": QUESTION}, lab)


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_shared_choice_executes_physical_contrast_and_preserves_supported_operation(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm=arm)
    retained = deepcopy(question(agent, "operation"))
    result = contrast(agent, lab)
    assert result["operator"] == "contrast" and result["question_id"] == "physical"
    assert result["diagnosis"] == "discriminating_experiment_available"
    assert result["selected_experiment"]["id"] == "physical"
    assert lab.actions == [{"action": "TALK", "arg1": 7}, {"action": "USE", "arg1": 8, "arg2": 7}]
    before = next(item for item in lab.before_experiment if item["id"] == "physical")
    assert "answer" not in before and before["status"] != "supported"
    assert before["experiment_choice"]["selected_experiment"]["predictions"] == {"extra": True, "law": False}
    child = question(agent, "physical")
    assert child["status"] == "refuted" and child["evidence"] == [1]
    assert child["answer"] == {"observed_outcome": False, "tested_prediction": True}
    assert question(agent, "operation") == retained
    assert not agent.relation_research  # Forecast columns never become fitted measurements.


@pytest.mark.parametrize("explicit", [False, True])
def test_negative_child_returns_parent_but_does_not_satisfy_an_explicit_premise(tmp_path, explicit):
    agent, lab = prepared(tmp_path)
    if explicit:
        question(agent, "task")["depends_on"] = ["physical"]
    contrast(agent, lab)
    parent = question(agent, "task")
    assert parent["status"] == "reopened" and "answer" not in parent
    assert agent._pending_dependencies(parent) == (["physical"] if explicit else [])


def test_agreement_acquires_nothing_and_grants_no_answer(tmp_path):
    agent, lab = prepared(tmp_path)
    source = SOURCE.replace("'law': False", "'law': True")
    result = contrast(agent, lab, source)
    assert result["diagnosis"] == "observational_ambiguity"
    assert result["selected_experiment"] is None
    assert lab.environment_steps == 1 and agent.program_executions == 2
    assert "answer" not in question(agent, "physical")


def test_required_operation_must_be_supported_before_design_or_acquisition(tmp_path):
    agent, lab = prepared(tmp_path)
    question(agent, "operation")["status"] = "refuted"
    executions = agent.program_executions
    result = contrast(agent, lab)
    assert result["diagnosis"] == "unresolved_child" and result["blocked_by"] == ["operation"]
    assert lab.environment_steps == 1 and agent.program_executions == executions
    assert "experiment_choice" not in question(agent, "physical")


def test_missing_physical_outcome_is_inconclusive_despite_successful_action(tmp_path):
    agent, lab = prepared(tmp_path, outcome=None)
    contrast(agent, lab)
    assert agent.observations[-1]["feedback"]["success"] is True
    child = question(agent, "physical")
    assert child["status"] == "inconclusive" and child["evidence"] == [1]
    assert child["answer"]["observed_outcome"] is None
    assert not agent._pending_dependencies(question(agent, "task"))


def test_failed_action_is_saved_for_analysis_without_automatic_support(tmp_path):
    agent, lab = prepared(tmp_path, failed=True)
    result = contrast(agent, lab)
    assert result["experiment"]["error"]
    assert len(agent.observations) == 2 and agent.observations[-1]["feedback"]["success"] is False
    assert "answer" not in question(agent, "physical")
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis" and focus["question"]["id"] == "physical"


@pytest.mark.parametrize("ending", ["ambiguity", "failed_acquisition"])
def test_revised_comparison_retains_old_evidence_but_withdraws_dependent_method(tmp_path, ending):
    agent, lab = prepared(tmp_path, outcome=True)
    contrast(agent, lab)
    child = question(agent, "physical")
    old_answer, old_evidence = deepcopy(child["answer"]), deepcopy(child["evidence"])
    question(agent, "task")["depends_on"] = [child["id"]]
    agent._execute("run", {"operator": "method", "id": "use_supported_result", "path": PATH,
                           "requires": [child["id"]]}, lab)
    assert agent.methods[0]["available"]
    source = SOURCE.replace("The extra operation is necessary", "A revised condition is necessary")
    if ending == "ambiguity":
        source = source.replace("'law': False", "'law': True")
    else:
        lab.failed = True
    result = contrast(agent, lab, source)

    assert child["status"] == "reopened"
    assert child["answer"] == old_answer and child["evidence"] == old_evidence
    assert child["reviewed_through_observation"] == 1
    assert agent._pending_dependencies(question(agent, "task")) == [child["id"]]
    assert not agent.methods[0]["available"]
    if ending == "ambiguity":
        assert result["diagnosis"] == "observational_ambiguity"
        assert lab.environment_steps == 2
    else:
        assert result["experiment"]["error"]
        before = next(item for item in lab.before_experiment if item["id"] == child["id"])
        assert before["status"] == "reopened"  # Revised premises are pending even before acquisition.
        assert lab.environment_steps == 3
    blocked = agent._execute("query", {"method_id": "use_supported_result"}, lab)
    assert blocked["error"] == "Method requirements need renewed evidence"
    assert lab.environment_steps == (2 if ending == "ambiguity" else 3)


def test_identical_repeat_with_failed_action_preserves_prior_scientific_answer(tmp_path):
    agent, lab = prepared(tmp_path, outcome=True)
    contrast(agent, lab)
    child = question(agent, "physical")
    old_answer, old_evidence = deepcopy(child["answer"]), deepcopy(child["evidence"])
    agent._execute("run", {"operator": "method", "id": "use_supported_result", "path": PATH,
                           "requires": [child["id"]]}, lab)
    lab.failed = True
    result = contrast(agent, lab)
    assert result["experiment"]["error"]
    assert child["status"] == "supported"
    assert child["answer"] == old_answer and child["evidence"] == old_evidence
    assert agent.methods[0]["available"]
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis" and focus["evidence"] == [2]

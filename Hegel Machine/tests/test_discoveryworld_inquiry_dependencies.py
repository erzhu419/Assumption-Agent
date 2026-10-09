"""A child's negative answer can change research without becoming a true premise."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider


INQUIRY = {"id": "preparation", "question": "Is preparation needed for the known response?",
           "parent_id": "task", "depends_on": []}

ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    active = record['observation']['active']
    return {'status': 'refuted' if active else 'supported',
            'answer': {'necessary': not active, 'observed_active': active},
            'evidence': [record['index']]}
"""

PARENT = """def select_actions(spec, observations, memory):
    condition = memory['answers']['preparation']
    assert condition['evidence'] == [0]
    if any(record['action']['action'] == 'TEST_NEW_SETTING' for record in observations):
        return {'stop': 'The new setting has been physically tested'}
    if condition['status'] == 'supported' and not spec['current_observation']['prepared']:
        return {'actions': [{'action': 'PREPARE'}]}
    assert condition['status'] in ('supported', 'refuted')
    return {'question': {'id': 'new_response', 'question': 'Does the new setting activate?',
                         'parent_id': 'task', 'depends_on': []},
            'actions': [{'action': 'TEST_NEW_SETTING', 'arg1': 3}],
            'after_observation': [{'action': 'evaluate', 'arguments': {
                'path': 'shared/research_task.py', 'question_id': 'new_response'}}]}

def answer(spec, observations, memory):
    record = observations[-1]
    active = record['observation']['active']
    return {'status': 'supported' if active else 'refuted',
            'answer': {'observed_active': active}, 'evidence': [record['index']]}
"""


class PreparationLab:
    def __init__(self, necessary=False):
        self.necessary = necessary
        self.prepared = False
        self.active = False
        self.environment_steps = 0
        self.actions = []

    def public_observation(self):
        return {"taskProgress": [{"description": "Test a new response setting", "completed": False}],
                "prepared": self.prepared, "active": self.active}

    def capabilities(self):
        return {"actions": ["TEST_KNOWN_SETTING", "PREPARE", "TEST_NEW_SETTING"]}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        if action["action"] == "PREPARE":
            self.prepared = True
        else:
            correct_setting = action["action"] == "TEST_KNOWN_SETTING" or action["arg1"] == 3
            self.active = correct_setting and (self.prepared or not self.necessary)
        return {"action": deepcopy(action), "feedback": {"success": True},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


def prepared_agent(tmp_path, *, necessary=False):
    lab = PreparationLab(necessary)
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    agent.lab = lab
    root = agent._question({"id": "task", "question": "Test a new response setting"})
    agent._execute("query", {"actions": [{"action": "TEST_KNOWN_SETTING"}],
                             "question": INQUIRY}, lab)
    return agent, lab, root


@pytest.mark.parametrize("necessary", [False, True])
def test_actual_child_answer_selects_parent_experiment_instead_of_requiring_its_truth(
        tmp_path, necessary):
    agent, lab, parent = prepared_agent(tmp_path, necessary=necessary)
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis"
    agent._execute("write", {"path": focus["path"], "code": ANALYSIS}, lab)
    child = next(q for q in agent.questions if q["id"] == "preparation")
    assert child["status"] == ("supported" if necessary else "refuted")
    assert child["answer"]["observed_active"] is not necessary
    assert not agent._pending_dependencies(parent)
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "parent_research"
    assert focus["returned_children"][0]["answer"] == child["answer"]
    agent._execute("write", {"path": focus["path"], "code": PARENT}, lab)

    expected = [{"action": "TEST_KNOWN_SETTING"}]
    if necessary:
        expected.append({"action": "PREPARE"})
    expected.append({"action": "TEST_NEW_SETTING", "arg1": 3})
    assert lab.actions == expected
    new_response = next(q for q in agent.questions if q["id"] == "new_response")
    assert new_response["status"] == "supported"
    assert new_response["answer"] == {"observed_active": True}
    assert new_response["evidence"] == [len(expected) - 1]
    assert child["evidence"] == [0]
    assert parent["status"] == "reopened" and "answer" not in parent


@pytest.mark.parametrize("status", ["refuted", "inconclusive"])
@pytest.mark.parametrize("origin", [None, "model_authored", "host_execution_failure",
                                    "host_parent_research_followup"])
def test_reviewed_answer_returns_control_independent_of_who_opened_child(tmp_path, status, origin):
    agent, _, parent = prepared_agent(tmp_path)
    child = agent.questions[-1]
    if origin:
        child["origin"] = origin
    agent._record_answer(child, {"status": status, "answer": "An actual reviewed outcome",
                                 "evidence": [0]})
    assert not agent._pending_dependencies(parent)
    assert agent._request()["research_focus"]["kind"] == "parent_research"


@pytest.mark.parametrize("status", ["refuted", "inconclusive"])
def test_explicit_premises_and_method_requirements_still_need_positive_support(tmp_path, status):
    agent, lab, parent = prepared_agent(tmp_path)
    child = agent.questions[-1]
    agent._record_answer(child, {"status": status, "answer": "An actual reviewed outcome",
                                 "evidence": [0]})
    agent._question({"id": "task", "depends_on": [child["id"]]})
    assert agent._pending_dependencies(parent) == [child["id"]]
    assert not agent._request().get("research_focus")
    (tmp_path / "method.py").write_text(
        "def select_actions(spec, observations, memory):\n    return [{'action': 'PREPARE'}]\n")
    agent._execute("run", {"operator": "method", "id": "needs_preparation",
                           "path": "method.py", "requires": [child["id"]]}, lab)
    feedback = agent._execute("query", {"method_id": "needs_preparation"}, lab)
    assert feedback["error"] == "Method requirements need renewed evidence"
    assert lab.actions == [{"action": "TEST_KNOWN_SETTING"}]
    agent._record_answer(child, {"status": "supported", "answer": "Revised condition holds",
                                 "evidence": [0]})
    assert not agent._pending_dependencies(parent)
    assert agent.methods[0]["available"]
    agent._execute("query", {"method_id": "needs_preparation"}, lab)
    assert lab.actions[-1] == {"action": "PREPARE"}


@pytest.mark.parametrize("status", ["refuted", "inconclusive"])
@pytest.mark.parametrize("missing", ["answer", "reviewed_through_observation"])
def test_a_status_without_returned_analysis_is_still_pending(tmp_path, status, missing):
    agent, _, parent = prepared_agent(tmp_path)
    child = agent.questions[-1]
    child.update(status=status, answer="Unreturned claim", reviewed_through_observation=0)
    child.pop(missing)
    assert agent._pending_dependencies(parent) == [child["id"]]


def test_reopening_a_child_blocks_again_without_erasing_its_old_answer(tmp_path):
    agent, _, parent = prepared_agent(tmp_path)
    child = agent.questions[-1]
    agent._record_answer(child, {"status": "refuted", "answer": "Preparation unnecessary here",
                                 "evidence": [0]})
    assert not agent._pending_dependencies(parent)
    agent._question({"id": child["id"], "question": "Is preparation needed after changing the conditions?"})
    assert child["status"] == "reopened"
    assert child["answer"] == "Preparation unnecessary here" and child["evidence"] == [0]
    assert agent._pending_dependencies(parent) == [child["id"]]


def test_supported_existing_condition_keeps_its_original_dependency_meaning(tmp_path):
    agent, _, parent = prepared_agent(tmp_path)
    child = agent.questions[-1]
    child["status"] = "supported"
    assert not agent._pending_dependencies(parent)
    parent["depends_on"] = [child["id"]]
    assert not agent._pending_dependencies(parent)


def test_successful_action_loop_returns_actual_procedure_trajectory_for_one_review(tmp_path):
    agent, lab, parent = prepared_agent(tmp_path)
    loop = "def select_actions(spec, observations, memory):\n    return [{'action': 'PREPARE'}]\n"
    provider = ScriptedProvider([
        ("write", {"path": "shared/analysis_preparation.py", "code": ANALYSIS}),
        ("write", {"path": "shared/research_task.py", "code": loop}),
        ("run", {"code": "print('The loop prepared the device but never tested a new setting')"}),
        ("finish", {"lesson": "The new response remains unresolved"}),
    ])
    agent.provider = provider
    result = agent.run(lab)

    procedure = result["events"][1]["feedback"]["procedure"]
    assert procedure["status"] == "method_step_limit" and len(procedure["steps"]) == 12
    assert all(row["action"] == "PREPARE" for row in lab.actions[1:])
    assert len(lab.actions) == 13
    assert not agent.method_results[-1]["feedback"].get("error")
    focus = provider.requests[2]["research_focus"]
    assert focus["stage"] == "review"
    previous = focus["previous_execution"]
    assert previous["procedure_status"] == "method_step_limit"
    assert [step["observation_indices"] for step in previous["procedure_steps"]] == [
        [index] for index in range(1, 13)]
    assert all(step["proposal"] == [{"action": "PREPARE"}] for step in previous["procedure_steps"])
    assert "research_focus" not in provider.requests[3]
    assert parent["reviewed_program_step"] == previous["step"]
    assert parent["status"] == "reopened" and "answer" not in parent
    assert result["research"]["program_executions"] == 13  # one analysis and twelve selectors

"""Preserve known controls while reopening an unresolved application hypothesis."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_prerequisite_research import next_prerequisite
from test_discoveryworld_agent import LIMITS
from test_discoveryworld_prerequisite import ControlLab, Provider


def unsupported(index, target):
    return {"index": index, "action": {"action": "ACTIVATE", "arg1": target},
            "feedback": {"success": False, "errors": []},
            "observation": {"lastActionMessage":
                            "That object (controller) is not activatable/deactivatable."}}


def known_operation(target, reviewed=None):
    question = {"id": f"operation_{target}", "target": target, "parent_id": "task",
                "status": "supported", "answer": {"operation": "TALK"},
                "evidence": [reviewed] if reviewed is not None else []}
    if reviewed is not None:
        question["reviewed_through_observation"] = reviewed
    return question


def test_new_failures_after_supported_interface_reopen_parent_despite_scan_cursor():
    operation = known_operation(10581, 104)
    before = deepcopy(operation)
    records = [unsupported(151, 10581), unsupported(164, 10581)]
    focus = next_prerequisite(records, [operation], {}, after_index=185)

    assert focus["kind"] == "intervention"
    assert focus["question"]["id"] == "intervention_10581"
    assert focus["question"]["parent_id"] == "task"
    assert focus["question"]["depends_on"] == ["operation_10581"]
    assert focus["origin"] == "host_parent_research_followup"
    assert focus["evidence"] == [151, 164]
    assert focus["known_operation"]["answer"] == {"operation": "TALK"}
    assert focus["known_operation"]["evidence"] == [104]
    assert operation == before


def test_only_failures_after_operation_answer_count_toward_followup():
    records = [unsupported(177, 34810), unsupported(186, 34810)]
    assert next_prerequisite(records, [known_operation(34810, 178)], {}, after_index=185) is None


def test_answer_without_review_boundary_does_not_invent_post_answer_failures():
    records = [unsupported(151, 10581), unsupported(164, 10581)]
    assert next_prerequisite(records, [known_operation(10581)], {}) is None


@pytest.mark.parametrize("status", ["supported", "refuted", "inconclusive"])
def test_reviewed_application_failures_are_not_reopened_until_two_new_failures(status):
    operation = known_operation(10581, 104)
    reviewed = {"id": "intervention_10581", "target": 10581, "status": status,
                "reviewed_through_observation": 164, "answer": "Candidate reviewed", "evidence": [164]}
    records = [unsupported(151, 10581), unsupported(164, 10581)]
    assert next_prerequisite(records, [operation, reviewed], {}, after_index=200) is None
    records.append(unsupported(170, 10581))
    assert next_prerequisite(records, [operation, reviewed], {}, after_index=200) is None
    records.append(unsupported(180, 10581))
    focus = next_prerequisite(records, [operation, reviewed], {}, after_index=200)
    assert focus["question"]["id"] == reviewed["id"]
    assert focus["evidence"] == [170, 180]
    assert focus["known_operation"]["answer"] == {"operation": "TALK"}


def test_application_support_is_not_itself_a_known_operation():
    reviewed = {"id": "intervention_10581", "target": 10581, "status": "supported",
                "reviewed_through_observation": 104}
    focus = next_prerequisite([unsupported(151, 10581), unsupported(164, 10581)], [reviewed], {})
    assert focus["question"]["id"] == "operation_10581"


FOLLOWUP_PATH = "shared/intervention_7.py"
FOLLOWUP = """def select_actions(spec, observations, memory):
    return {'actions': [],
            'hypothesis': {'target': 7, 'candidate_setting': 5000},
            'predicted_outcome': {'calibrated': True}}

def answer(spec, observations, memory):
    for record in reversed(observations):
        objects = record['observation'].get('accessibleEnvironmentObjects', [])
        for obj in objects:
            if obj['uuid'] != 7:
                continue
            actual = obj.get('calibrated')
            status = 'inconclusive' if actual is None else ('supported' if actual else 'refuted')
            return {'status': status,
                    'answer': {'candidate_setting': 5000, 'actual_calibrated': actual},
                    'evidence': [record['index']]}
"""
PARENT = """def select_actions(spec, observations, memory):
    child = memory['answers']['intervention_7']
    action = {'refuted': 'CHANGE_CANDIDATE', 'supported': 'USE_CANDIDATE',
              'inconclusive': 'MEASURE'}[child['status']]
    return [{'action': action, 'arg1': 7, 'evidence': child['evidence']}]
"""


class ApplicationLab(ControlLab):
    def __init__(self, actual):
        super().__init__()
        self.actual = actual

    def public_observation(self):
        result = super().public_observation()
        item = result["accessibleEnvironmentObjects"][0]
        item["candidate_setting"] = 5000
        if self.actual is not None:
            item["calibrated"] = self.actual
        return result

    def capabilities(self):
        result = super().capabilities()
        result["actions"].update({name: {"args": ["arg1"]}
                                  for name in ("CHANGE_CANDIDATE", "USE_CANDIDATE", "MEASURE")})
        return result

    def act(self, action):
        if action["action"] not in {"CHANGE_CANDIDATE", "USE_CANDIDATE", "MEASURE"}:
            return super().act(action)
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        return {"action": deepcopy(action), "feedback": {"success": True, "errors": []},
                "observation": self.public_observation()}


def seeded_agent(tmp_path, steps, *, actual=False, arm="hegel"):
    provider, lab = Provider(steps), ApplicationLab(actual)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm,
                               limits={**LIMITS, "max_model_calls": len(steps)})
    agent.lab = lab
    agent._question({"id": "task", "question": "Find an effective candidate setting"})
    for operation in ("READ", "ACTIVATE", "TALK"):
        record = lab.act({"action": operation, "arg1": 7})
        agent.observations.append({"index": len(agent.observations), **record})
    question = agent._question({"id": "operation_7", "question": "Which interaction works?",
                                "parent_id": "task"})
    question["target"] = 7
    agent._record_answer(question, {"status": "supported", "answer": {"operation": "TALK"},
                                   "evidence": [2]})
    for _ in range(2):
        record = lab.act({"action": "ACTIVATE", "arg1": 7})
        agent.observations.append({"index": len(agent.observations), **record})
    agent.costs["environment_steps"] = lab.environment_steps
    agent._prerequisite_checked_through = 4
    return agent, provider, lab, deepcopy(question)


@pytest.mark.parametrize("actual,status,parent_action", [
    (False, "refuted", "CHANGE_CANDIDATE"),
    (True, "supported", "USE_CANDIDATE"),
    (None, "inconclusive", "MEASURE"),
])
def test_existing_paid_outcome_changes_parent_without_erasing_supported_control(
        tmp_path, actual, status, parent_action):
    steps = [("write", {"path": FOLLOWUP_PATH, "code": FOLLOWUP}),
             ("query", {"path": "shared/parent.py", "question": {"id": "task"}}), ("finish", {})]
    agent, provider, lab, operation = seeded_agent(tmp_path, steps, actual=actual)
    (tmp_path / "shared").mkdir(exist_ok=True)
    (tmp_path / "shared/parent.py").write_text(PARENT)
    result = agent.run(lab)

    focus = provider.requests[0]["research_focus"]
    assert focus["kind"] == "intervention" and focus["evidence"] == [3, 4]
    assert focus["question"]["depends_on"] == ["operation_7"]
    assert focus["question"]["origin"] == "host_parent_research_followup"
    answer = provider.requests[1]["research"]["answers"]["intervention_7"]
    assert answer["status"] == status and answer["evidence"] == [4]
    assert answer["answer"]["actual_calibrated"] is actual
    assert provider.requests[1]["research_focus"]["kind"] == "parent_research"
    assert not provider.requests[2].get("research_focus")
    # The API rejected ACTIVATE in all three cases. Only the observed physical
    # state distinguishes a supported/refuted/inconclusive candidate setting.
    assert len(lab.actions) == 6
    assert lab.actions[-1] == {"action": parent_action, "arg1": 7, "evidence": [4]}
    assert next(q for q in agent.questions if q["id"] == "operation_7") == operation
    assert result["research"]["program_executions"] == 3
    assert not result["score"]["completedSuccessfully"]
    root = next(q for q in agent.questions if q["id"] == "task")
    assert "intervention_7" not in agent._pending_dependencies(root)
    decision = json.loads((tmp_path / "method_decision.json").read_text())
    displayed_root = next(q for q in decision["questions"] if q["id"] == "task")
    assert "intervention_7" not in displayed_root["pending_dependencies"]
    dependent = agent._question({"id": "use_supported_candidate", "question": "Use only a supported candidate",
                                 "depends_on": ["intervention_7"]})
    assert agent._pending_dependencies(dependent) == ([] if status == "supported" else ["intervention_7"])
    proposal = next(item["proposal"] for item in agent.method_results if item["path"] == FOLLOWUP_PATH)
    assert proposal["actions"] == []
    assert proposal["hypothesis"] == {"target": 7, "candidate_setting": 5000}
    assert proposal["predicted_outcome"] == {"calibrated": True}


@pytest.mark.parametrize("broken", [
    FOLLOWUP.replace("return {'actions': [],", "raise ValueError('revise selector')\n    return {'actions': [],"),
    FOLLOWUP.replace("'evidence': [record['index']]", "'evidence': []"),
])
def test_failed_intervention_program_remains_repairable_without_new_measurements(tmp_path, broken):
    steps = [("write", {"path": FOLLOWUP_PATH, "code": broken}),
             ("write", {"path": FOLLOWUP_PATH, "code": FOLLOWUP}), ("finish", {})]
    agent, provider, lab, operation = seeded_agent(tmp_path, steps)
    result = agent.run(lab)

    assert provider.requests[0]["research_focus"]["kind"] == "intervention"
    assert provider.requests[1]["research_focus"]["kind"] == "intervention"
    assert "reviewed_through_observation" not in next(
        q for q in provider.requests[1]["research"]["questions"] if q["id"] == "intervention_7")
    assert provider.requests[2]["research_focus"]["kind"] == "parent_research"
    assert provider.requests[2]["research"]["answers"]["intervention_7"]["status"] == "refuted"
    assert next(q for q in agent.questions if q["id"] == "operation_7") == operation
    assert result["costs"]["environment_steps"] == len(lab.actions) == 5


def test_code_arm_has_same_analysis_tools_without_host_followup(tmp_path):
    agent, _, _, _ = seeded_agent(tmp_path, [("finish", {})], arm="code_archive")
    assert not agent._request().get("research_focus")

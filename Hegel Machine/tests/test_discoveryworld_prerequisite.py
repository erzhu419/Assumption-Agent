"""A real failed-operation prerequisite, an authored probe, and a returned answer."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS
from test_discoveryworld_recovery import DECODE_ERROR, USAGE


QUESTION_ID = "operation_7"
PROGRAM_PATH = "shared/operation_7.py"
PROBE = """def select_actions(spec, observations, memory):
    return [{'action': 'TALK', 'arg1': 7}]

def answer(spec, observations, memory):
    record = observations[-1]
    observed = record['observation']['dialog_box']['is_in_dialog']
    return {'status': 'supported' if observed else 'inconclusive',
            'answer': {'operation': 'TALK'}, 'evidence': [record['index']]}
"""


class ControlLab:
    def __init__(self):
        self.environment_steps = 0
        self.actions = []
        self.dialog = self.completed = self.success = False
        self.message = ""

    def public_observation(self):
        return {
            "taskProgress": [{"description": "Find the control interface, then apply the intervention",
                              "completed": self.completed}],
            "accessibleEnvironmentObjects": [{"uuid": 7, "name": "controller"}],
            "nearbyAgents": {"list_of_agents": {"controller uuid 7": []}},
            "dialog_box": {"is_in_dialog": self.dialog},
            "lastActionMessage": self.message,
        }

    def capabilities(self):
        return {"actions": {name: {"args": ["arg1"]}
                            for name in ("READ", "ACTIVATE", "TALK", "TUNE")}}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        kind = action["action"]
        if kind in ("READ", "ACTIVATE"):
            self.message = ("That object (controller) is not readable." if kind == "READ"
                            else "That object (controller) is not activatable/deactivatable.")
            successful = False
        elif kind == "TALK":
            self.dialog = True
            self.message = "Control dialog opened."
            successful = True
        else:
            assert kind == "TUNE"
            self.completed = True
            self.success = self.dialog and action["arg1"] == 7
            successful = self.success
            self.message = "Intervention accepted." if successful else "Control interface was not established."
        return {"action": deepcopy(action),
                "feedback": {"success": successful, "errors": []},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": self.completed, "completedSuccessfully": self.success,
                "scoreNormalized": float(self.success)}


class Provider:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def next_action(self, request):
        self.requests.append(deepcopy(request))
        action, arguments = next(self.steps)
        if action == "parent_intervention":
            child = request["research"]["answers"][QUESTION_ID]
            assert child["status"] == "supported"
            action = "query"
            arguments = {"actions": [{"action": "TUNE", "arg1":
                          7 if child["answer"]["operation"] == "TALK" else -1}]}
        result = {"action": action, "arguments": deepcopy(arguments), "usage": USAGE}
        if action == "provider_error":
            result["error"] = DECODE_ERROR
        return result


def failed_operations():
    return [("query", {"actions": [{"action": operation, "arg1": 7}]})
            for operation in ("READ", "ACTIVATE")]


def run(tmp_path, steps, *, arm="hegel", call_limit=None):
    provider, lab = Provider(steps), ControlLab()
    limits = {**LIMITS, "max_model_calls": call_limit or len(steps)}
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm, limits=limits)
    return agent.run(lab), agent, provider, lab


def assert_pending_focus(request):
    focus = request["research_focus"]
    assert focus["question"]["id"] == QUESTION_ID
    assert focus["path"] == PROGRAM_PATH
    assert focus["failures"]
    question = next(q for q in request["research"]["questions"] if q["id"] == QUESTION_ID)
    assert question["parent_id"] == "task"
    assert question["status"] != "supported"
    assert question["origin"] == focus["question"]["origin"] == "host_execution_failure"
    records = request["evidence"]["recent_action_results"]
    assert [r["index"] for r in records[:2]] == [0, 1]
    assert all(r["feedback"]["success"] is False for r in records[:2])
    assert focus["failures"][:2] == [{"index": r["index"], "action": r["action"],
        "reason": r["lastActionMessage"], "kind": "unsupported_action"} for r in records[:2]]
    return focus


def test_authored_prerequisite_probe_returns_answer_before_parent_intervention(tmp_path):
    steps = [*failed_operations(),
             ("write", {"path": PROGRAM_PATH, "code": PROBE}),
             ("parent_intervention", {})]
    result, agent, provider, lab = run(tmp_path, steps)

    assert not provider.requests[1].get("research_focus")
    assert_pending_focus(provider.requests[2])
    # Merely proposing/writing a test must not constitute empirical support.
    assert provider.requests[2]["research"]["answers"][QUESTION_ID]["status"] != "supported"
    returned = provider.requests[3]["research"]["answers"][QUESTION_ID]
    assert returned == {"status": "supported", "answer": {"operation": "TALK"}, "evidence": [2]}
    assert provider.requests[3]["research_focus"]["kind"] == "parent_research"
    assert provider.requests[3]["research_focus"]["question"]["id"] == "task"
    assert [a["action"] for a in lab.actions] == ["READ", "ACTIVATE", "TALK", "TUNE"]
    assert result["status"] == "completed" and result["score"]["completedSuccessfully"]
    assert result["research"]["child_answers_returned"] == 1
    assert result["research"]["program_executions"] >= 2
    question = next(q for q in agent.questions if q["id"] == QUESTION_ID)
    assert question["analysis_path"] == PROGRAM_PATH
    assert question["acquired_observations"] == [2]
    assert any(step["path"] == PROGRAM_PATH for step in agent.method_results)
    assert result["costs"]["environment_steps"] == 4
    assert result["costs"]["model_calls"] == 4


def test_code_has_same_program_tools_without_host_imposed_subproblem(tmp_path):
    steps = [*failed_operations(),
             ("write", {"path": PROGRAM_PATH, "code": PROBE}),
             ("finish", {"lesson": "No forced research program"})]
    result, _, provider, lab = run(tmp_path, steps, arm="code_archive")

    assert all(not request.get("research_focus") for request in provider.requests)
    assert all("select_actions" in request["task_interface"] and "evaluate" in request["task_interface"]
               for request in provider.requests)
    assert (tmp_path / PROGRAM_PATH).read_text() == PROBE
    assert [a["action"] for a in lab.actions] == ["READ", "ACTIVATE"]
    assert result["research"]["questions"] == []
    assert result["research"]["program_executions"] == 0


def test_return_to_parent_does_not_reopen_older_unrelated_failures(tmp_path):
    # A restored exploration prefix can contain several failed interfaces.
    # Resolving the latest blocker returns to the parent rather than draining
    # every historical failure as a new mandatory research task.
    provider, lab = Provider([
        ("write", {"path": PROGRAM_PATH, "code": PROBE}),
        ("parent_intervention", {})]), ControlLab()
    agent = DiscoveryWorldAgent(provider, tmp_path, arm="hegel", limits=LIMITS)
    for target in (8, 7):
        for operation in ("READ", "ACTIVATE"):
            record = lab.act({"action": operation, "arg1": target})
            agent.observations.append({"index": len(agent.observations), **record})
    agent.costs["environment_steps"] = lab.environment_steps
    result = agent.run(lab)
    assert provider.requests[0]["research_focus"]["target"] == 7
    assert provider.requests[1]["research_focus"]["kind"] == "parent_research"
    assert provider.requests[1]["research_focus"]["question"]["id"] == "task"
    assert result["score"]["completedSuccessfully"]


@pytest.mark.parametrize("bad_program,extra_actions", [
    (PROBE.replace("return [{'action': 'TALK', 'arg1': 7}]", "raise ValueError('revise the probe')"), 0),
    (PROBE.replace("'action': 'TALK'", "'action': 'READ'"), 1),
    (PROBE.replace("'evidence': [record['index']]", "'evidence': []"), 1),
])
def test_failed_probe_or_empty_answer_keeps_focus_and_repairs_within_original_budget(
        tmp_path, bad_program, extra_actions):
    steps = [*failed_operations(),
             ("write", {"path": PROGRAM_PATH, "code": bad_program}),
             ("provider_error", {}),
             ("write", {"path": PROGRAM_PATH, "code": PROBE}),
             ("parent_intervention", {})]
    result, _, provider, lab = run(tmp_path, steps, call_limit=6)

    assert_pending_focus(provider.requests[2])
    assert_pending_focus(provider.requests[3])
    assert_pending_focus(provider.requests[4])
    assert provider.requests[4]["costs"]["model_calls"] == 4
    assert provider.requests[4]["costs"]["environment_steps"] == 2 + extra_actions
    assert provider.requests[4]["limits"]["max_model_calls"] == 6
    child = provider.requests[5]["research"]["answers"][QUESTION_ID]
    assert child["status"] == "supported"
    assert child["evidence"] == [2 + extra_actions]
    assert result["status"] == "completed" and result["score"]["completedSuccessfully"]
    assert result["costs"]["model_calls"] == len(provider.requests) == 6
    assert result["costs"]["reported_tokens"] == 6 * USAGE["total_tokens"]
    assert result["costs"]["cached_input_tokens"] == 6 * USAGE["cached_input_tokens"]
    assert result["costs"]["environment_steps"] == len(lab.actions) == 4 + extra_actions
    assert result["research"]["child_answers_returned"] == 1

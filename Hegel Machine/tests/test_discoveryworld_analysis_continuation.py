"""Close acquired child evidence through the existing executable answer path."""

from copy import deepcopy

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS
from test_discoveryworld_recovery import DECODE_ERROR, USAGE


CHILD = {"id": "activation", "question": "Does the candidate setting activate the device?",
         "parent_id": "task", "depends_on": []}
ANALYSIS_PATH = "shared/analysis_activation.py"
ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    actual = record['observation'].get('active')
    status = 'inconclusive' if actual is None else ('supported' if actual else 'refuted')
    return {'status': status, 'answer': {'candidate_setting': 5000},
            'evidence': [record['index']]}
"""
PARENT = """def select_actions(spec, observations, memory):
    child = memory['answers']['activation']
    action = {'inconclusive': 'TEST_ACTIVATION', 'supported': 'USE_CANDIDATE',
              'refuted': 'CHANGE_CANDIDATE'}[child['status']]
    return [{'action': action, 'evidence': child['evidence']}]
"""


class ActivationLab:
    def __init__(self, actual=None):
        self.actual = actual
        self.environment_steps = 0
        self.actions = []

    def public_observation(self):
        result = {"taskProgress": [{"description": "Find a setting that activates the device",
                                    "completed": False}]}
        if self.actions:
            result["setting"] = 5000
            if self.actions[-1]["action"] == "OBSERVE":
                result["active"] = self.actual
        return result

    def capabilities(self):
        return {"actions": ["SET", "OBSERVE", "TEST_ACTIVATION", "USE_CANDIDATE", "CHANGE_CANDIDATE"]}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        return {"action": deepcopy(action), "feedback": {"success": True},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


class Provider:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def next_action(self, request):
        self.requests.append(deepcopy(request))
        action, arguments = next(self.steps)
        return {"action": action, "arguments": deepcopy(arguments), "usage": USAGE,
                **({"error": DECODE_ERROR} if action == "provider_error" else {})}


def make_agent(tmp_path, steps=(), *, arm="hegel", actual=None):
    provider, lab = Provider(steps), ActivationLab(actual)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm,
                               limits={**LIMITS, "max_model_calls": max(1, len(steps))})
    agent.lab = lab
    agent._question({"id": "task", "question": "Find a valid activation setting"})
    return agent, provider, lab


@pytest.mark.parametrize("operation,actual,status,parent_action", [
    ("SET", None, "inconclusive", "TEST_ACTIVATION"),
    ("OBSERVE", True, "supported", "USE_CANDIDATE"),
    ("OBSERVE", False, "refuted", "CHANGE_CANDIDATE"),
])
def test_only_actual_outcome_changes_child_support_and_parent_experiment(
        tmp_path, operation, actual, status, parent_action):
    steps = [("query", {"actions": [{"action": operation}], "question": CHILD}),
             ("write", {"path": ANALYSIS_PATH, "code": ANALYSIS}),
             ("query", {"path": "shared/parent.py", "question": {"id": "task"}}), ("finish", {})]
    agent, provider, lab = make_agent(tmp_path, steps, actual=actual)
    (tmp_path / "shared").mkdir(exist_ok=True)
    (tmp_path / "shared/parent.py").write_text(PARENT)
    result = agent.run(lab)

    focus = provider.requests[1]["research_focus"]
    assert focus["kind"] == "analysis"
    assert focus["question"]["id"] == CHILD["id"]
    assert focus["path"] == ANALYSIS_PATH
    question = next(q for q in result["research"]["questions"] if q["id"] == CHILD["id"])
    assert question["question"] == CHILD["question"]
    assert question.get("origin") != "host_execution_failure"
    assert question["status"] == status and question["evidence"] == [0]
    assert question["reviewed_through_observation"] == 0
    assert provider.requests[2]["research_focus"]["kind"] == "parent_research"
    assert not provider.requests[3].get("research_focus")
    assert lab.actions == [{"action": operation}, {"action": parent_action, "evidence": [0]}]
    # The newly written file deliberately has no selector: analysis must not
    # rerun a measurement or require a second program just to interpret it.
    assert result["research"]["program_executions"] == 2
    assert result["costs"]["environment_steps"] == 2
    assert result["costs"]["model_calls"] == 4
    assert not result["score"]["completedSuccessfully"]


def test_reuse_analysis_path_and_rereview_only_after_new_child_acquisition(tmp_path):
    agent, _, lab = make_agent(tmp_path, actual=True)
    child = agent._question({**CHILD, "analysis_path": "existing_analysis.py"})
    child["origin"] = "model_authored"
    agent._execute("query", {"question": {"id": CHILD["id"]},
                            "actions": [{"action": "SET"}]}, lab)
    first = agent._request()["research_focus"]
    assert first["path"] == "existing_analysis.py"
    agent._execute("write", {"path": first["path"], "code": ANALYSIS}, lab)
    assert child["status"] == "inconclusive"
    assert child["origin"] == "model_authored"
    for _ in range(2):
        assert agent._request()["research_focus"]["kind"] == "parent_research"
    assert agent.program_executions == 1 and lab.environment_steps == 1

    # New child evidence takes priority over the parent authoring request.
    agent._execute("query", {"question": {"id": CHILD["id"]},
                            "actions": [{"action": "OBSERVE"}]}, lab)
    assert agent._request()["research_focus"]["question"]["id"] == CHILD["id"]
    agent._execute("write", {"path": first["path"], "code": ANALYSIS}, lab)
    assert child["status"] == "supported" and child["evidence"] == [1]
    assert child["reviewed_through_observation"] == 1
    assert agent._request()["research_focus"]["kind"] == "parent_research"
    assert agent.program_executions == 2 and lab.environment_steps == 2


@pytest.mark.parametrize("bad_analysis", [
    "def answer(spec, observations, memory):\n    raise ValueError('revise analysis')\n",
    ANALYSIS.replace("'evidence': [record['index']]", "'evidence': []"),
])
def test_failed_analysis_remains_repairable_without_buying_new_observations(tmp_path, bad_analysis):
    steps = [("query", {"actions": [{"action": "OBSERVE"}], "question": CHILD}),
             ("write", {"path": ANALYSIS_PATH, "code": bad_analysis}),
             ("provider_error", {}),
             ("write", {"path": ANALYSIS_PATH, "code": ANALYSIS}), ("finish", {})]
    agent, provider, lab = make_agent(tmp_path, steps, actual=False)
    result = agent.run(lab)
    for request in provider.requests[1:4]:
        assert request["research_focus"]["kind"] == "analysis"
        child = next(q for q in request["research"]["questions"] if q["id"] == CHILD["id"])
        assert "reviewed_through_observation" not in child
    assert "error" in result["events"][1]["feedback"]
    assert provider.requests[4]["research_focus"]["kind"] == "parent_research"
    assert provider.requests[4]["research"]["answers"][CHILD["id"]]["status"] == "refuted"
    assert result["costs"]["model_calls"] == 5
    assert result["costs"]["reported_tokens"] == 5 * USAGE["total_tokens"]
    assert result["costs"]["cached_input_tokens"] == 5 * USAGE["cached_input_tokens"]
    assert lab.actions == [{"action": "OBSERVE"}]


@pytest.mark.parametrize("condition", ["code", "root", "binding", "child_dependency", "explicit_dependency"])
def test_host_analysis_does_not_override_other_research_routes(tmp_path, condition):
    agent, _, lab = make_agent(tmp_path, arm="code_archive" if condition == "code" else "hegel")
    question = {"id": "task"} if condition == "root" else CHILD
    agent._execute("query", {"question": question, "actions": [{"action": "SET"}]}, lab)
    if condition == "binding":
        agent.questions[-1]["binding_path"] = "binding.py"
    if condition in ("child_dependency", "explicit_dependency"):
        agent._question({"id": "instrument", "question": "Is the instrument usable?",
                         "parent_id": CHILD["id"] if condition == "child_dependency" else "task"})
        if condition == "explicit_dependency":
            agent._question({"id": CHILD["id"], "depends_on": ["instrument"]})
    assert not agent._request().get("research_focus")
    assert agent.program_executions == 0

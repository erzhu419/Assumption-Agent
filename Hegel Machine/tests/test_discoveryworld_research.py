"""Actual world acquisition and parent action through the reused research loop."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider


class RelationLab:
    def __init__(self, square=False):
        self.square = square
        self.environment_steps = 0
        self.actions = []
        self.completed = self.success = False

    def public_observation(self):
        return {"taskProgress": [{"description": "Measure the response, then tune to its value at x=4",
                                  "completed": self.completed}]}

    def capabilities(self):
        return {"actions": ["READ", "TUNE"]}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        if action["action"] == "READ":
            x = action["arg1"]
            feedback = {"x": x, "reading": x * x if self.square else x}
        else:
            self.completed = True
            self.success = action["arg1"] == (16 if self.square else 4)
            feedback = {"message": "Intervention performed"}
        return {"action": deepcopy(action), "feedback": feedback,
                "observation": self.public_observation()}

    def score(self):
        return {"completed": self.completed, "completedSuccessfully": self.success,
                "scoreNormalized": float(self.success)}


BINDING = """def bind(spec, observations, memory):
    rows = [{'features': {'x': r['feedback']['x']},
             'target': r['feedback']['reading'], 'evidence': [r['index']]}
            for r in observations if r['action']['action'] == 'READ']
    x = 2 if len(rows) < 3 else 3
    feature = {'op': 'feature', 'name': 'x'}
    return {'rows': rows,
            'hypotheses': [feature, {'op': 'mul', 'children': [feature, feature]}],
            'probes': [{'id': 'probe_' + str(x), 'features': {'x': x},
                        'actions': [{'action': 'READ', 'arg1': x}]}],
            'tolerance': 1e-10, 'parent_path': 'parent.py'}
"""

PARENT = """def select_actions(spec, observations, memory):
    child = memory['answers']['relation']
    assert child['status'] == 'supported'
    assert set(child['evidence']) >= {2, 3}
    def predict(program, x):
        if program['op'] == 'feature':
            return x
        assert program['op'] == 'mul'
        return predict(program['children'][0], x) * predict(program['children'][1], x)
    return [{'action': 'TUNE', 'arg1': predict(child['answer']['program'], 4)}]
"""


def run_research(tmp_path, *, arm="hegel", square=False, binding=BINDING,
                 limits=None, parent_dependencies=()):
    (tmp_path / "binding.py").write_text(binding)
    (tmp_path / "parent.py").write_text(PARENT)
    initial = {"actions": [{"action": "READ", "arg1": 0}, {"action": "READ", "arg1": 1}],
               "research_question": {"id": "task", "question": "Which intervention solves the task?",
                                     "parent_id": None, "depends_on": list(parent_dependencies)}}
    if parent_dependencies:
        initial["question"] = {"id": "calibration", "question": "Is the actuator calibrated?",
                               "parent_id": "task", "depends_on": []}
    provider = ScriptedProvider([
        ("query", initial),
        ("run", {"operator": "research", "path": "binding.py", "question": {
            "id": "relation", "parent_id": "task", "question": "Is the response linear or square?"}}),
        ("finish", {"lesson": "Return unresolved research to the model"}),
    ])
    lab = RelationLab(square)
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm, limits=limits or LIMITS)
    return agent.run(lab), agent, provider, lab


@pytest.mark.parametrize("arm", ["code_archive", "hegel"])
@pytest.mark.parametrize("square", [False, True])
def test_one_research_call_discriminates_confirms_and_changes_parent_action(tmp_path, arm, square):
    result, agent, provider, lab = run_research(tmp_path, arm=arm, square=square)

    assert result["status"] == "completed", result["events"][-1]
    assert result["score"]["completedSuccessfully"]
    assert lab.actions == [
        {"action": "READ", "arg1": 0}, {"action": "READ", "arg1": 1},
        {"action": "READ", "arg1": 2}, {"action": "READ", "arg1": 3},
        {"action": "TUNE", "arg1": 16 if square else 4}]
    assert result["costs"]["model_calls"] == len(provider.requests) == 2
    assert result["costs"]["environment_steps"] == 5
    assert result["costs"]["research_compute_units"] > 0
    child = next(q for q in result["research"]["questions"] if q["id"] == "relation")
    assert child["opened_before_observation"] == 2
    assert child["status"] == "supported"
    assert {2, 3}.issubset(child["evidence"])
    assert child["answer"]["program"]["op"] == ("mul" if square else "feature")
    assert result["research"]["child_answers_returned"] > 0
    feedback = result["events"][1]["feedback"]
    assert feedback["status"] == "supported"
    assert feedback["initial_research"]["status"] == "inconclusive"
    assert agent.method_results  # The existing procedure executor ran real steps.
    saved = json.loads((tmp_path / "research_state.json").read_text())
    assert saved["answers"]["relation"]["answer"]["program"] == child["answer"]["program"]


def test_observationally_equivalent_available_probes_return_without_more_acquisition(tmp_path):
    binding = BINDING.replace("x = 2 if len(rows) < 3 else 3", "x = 1")
    result, _, provider, lab = run_research(tmp_path, binding=binding)

    assert result["status"] == "agent_finished"
    assert len(provider.requests) == 3
    assert lab.actions == [{"action": "READ", "arg1": 0}, {"action": "READ", "arg1": 1}]
    child = next(q for q in result["research"]["questions"] if q["id"] == "relation")
    assert child["status"] != "supported"
    assert result["research"]["child_answers_returned"] == 0
    assert not result["events"][1]["feedback"].get("error")


def test_no_confirmation_budget_prevents_supported_answer_and_parent_intervention(tmp_path):
    result, _, _, lab = run_research(tmp_path, limits={**LIMITS, "environment_steps": 3})

    assert result["status"] == "environment_budget_exhausted"
    assert result["costs"]["environment_steps"] == len(lab.actions) == 3
    assert all(action["action"] == "READ" for action in lab.actions)
    child = next(q for q in result["research"]["questions"] if q["id"] == "relation")
    assert child["status"] != "supported"
    assert result["research"]["child_answers_returned"] == 0


def test_supported_relation_does_not_override_unresolved_parent_dependency(tmp_path):
    result, _, _, lab = run_research(tmp_path, parent_dependencies=["calibration"])

    assert result["status"] == "agent_finished"
    questions = {q["id"]: q for q in result["research"]["questions"]}
    assert questions["relation"]["status"] == "supported"
    assert questions["calibration"]["status"] != "supported"
    assert "answer" not in questions["calibration"]
    assert questions["task"]["status"] != "supported"
    assert len(lab.actions) == 4
    assert all(action["action"] == "READ" for action in lab.actions)


def test_revising_child_withdraws_its_previously_scheduled_parent_step(tmp_path):
    _, agent, _, lab = run_research(tmp_path)
    agent._execute("run", {"operator": "research_update", "path": "binding.py",
                           "question": {"id": "relation"}}, lab)
    assert agent.research_advice["next_action"]["arguments"]["path"] == "parent.py"
    agent._execute("run", {"operator": "question", "id": "relation",
                           "question": "Does this response still hold under the changed condition?"}, lab)
    assert agent.research_advice is None
    proposal = agent._program("shared/world_research_driver.py", "select_actions")
    assert "stop" in proposal

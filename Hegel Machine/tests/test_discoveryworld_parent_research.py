"""Parent-program adoption contracts; toy executions are not LLM evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition import code_agent
from test_discoveryworld_agent import LIMITS, ScriptedProvider


PATH = "shared/research_task.py"


class DialogLab:
    """A public dialog exposes controls and the resulting physical state."""

    def __init__(self):
        self.environment_steps = 0
        self.actions = []
        self.in_dialog = False
        self.setting = None

    def public_observation(self):
        return {
            "taskProgress": [{"description": "Find a supported activation setting", "completed": False}],
            "dialog_box": {"is_in_dialog": self.in_dialog,
                           "dialog_options": {"1": "Set 6", "2": "Set 8"} if self.in_dialog else {}},
            "accessibleEnvironmentObjects": [{"uuid": 7, "name":
                "device (active)" if self.setting == 8 else "device (inactive)",
                "description": "Public control device"}],
            "lastActionMessage": f"Setting {self.setting}" if self.setting is not None else "Ready",
        }

    def capabilities(self):
        return {"actions": ["MEASURE", "TALK"]}

    def act(self, action):
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        feedback = {"success": True}
        if action.get("action") == "MEASURE":
            feedback["reading"] = 3
        elif action.get("action") == "TALK":
            self.in_dialog = True
        elif "chosen_dialog_option_int" in action:
            assert self.in_dialog
            self.setting = {1: 6, 2: 8}[action["chosen_dialog_option_int"]]
        return {"action": deepcopy(action), "feedback": feedback,
                "observation": self.public_observation()}

    def score(self):
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


ANALYSIS = """def answer(spec, observations, memory):
    record = observations[-1]
    return {'status': 'supported', 'answer': {'reading': record['feedback']['reading']},
            'evidence': [record['index']]}
"""

PROGRAM = """def select_actions(spec, observations, memory):
    reading = memory['answers']['instrument']['answer']['reading']
    assert reading == 3
    assert 'parent_research' not in str(memory.get('research_focus'))
    answers = memory['answers']
    if answers.get('candidate_b', {}).get('status') == 'supported':
        return {'stop': 'The second candidate has an observed activating outcome'}
    if not spec['current_observation']['dialog_box']['is_in_dialog']:
        return {'actions': [{'action': 'TALK', 'arg1': 7}]}
    rejected = answers.get('candidate_a', {}).get('status') == 'refuted'
    qid = 'candidate_b' if rejected else 'candidate_a'
    setting = reading + 5 if rejected else reading * 2
    return {'question': {'id': qid, 'question': 'Does this candidate activate the device?',
                         'parent_id': 'task', 'depends_on': ['instrument']},
            'hypothesis': {'setting': setting}, 'predicted_outcome': 'device (active)',
            'actions': [{'chosen_dialog_option_int': 2 if rejected else 1}],
            'after_observation': [{'action': 'evaluate', 'arguments': {
                'path': 'shared/research_task.py', 'question_id': qid}}]}

def answer(spec, observations, memory):
    record = observations[-1]
    state = next(item['name'] for item in record['observation']['accessibleEnvironmentObjects']
                 if item['uuid'] == 7)
    return {'status': 'supported' if state == 'device (active)' else 'refuted',
            'answer': {'observed_state': state}, 'evidence': [record['index']]}
"""


def prepared(tmp_path, *, arm="hegel", answer=True):
    lab = DialogLab()
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm, limits=LIMITS)
    agent.lab = lab
    agent._question({"id": "task", "question": "Find a supported activation setting"})
    agent._execute("query", {"actions": [{"action": "MEASURE"}], "question": {
        "id": "instrument", "question": "What does this instrument report?", "parent_id": "task"}}, lab)
    if answer:
        agent._execute("write", {"path": "instrument.py", "code": ANALYSIS}, lab)
        agent._execute("evaluate", {"path": "instrument.py", "question_id": "instrument"}, lab)
    return agent, lab


def question(agent, qid="task"):
    return next(item for item in agent.questions if item["id"] == qid)


def test_returned_child_drives_live_parent_discrimination_and_analysis(tmp_path):
    agent, lab = prepared(tmp_path)
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "parent_research"
    assert focus["origin"] == "host_parent_research_continuation"
    assert focus["path"] == PATH and focus["question"]["id"] == "task"
    assert [item["id"] for item in focus["returned_children"]] == ["instrument"]

    agent._execute("write", {"path": PATH, "code": PROGRAM}, lab)

    assert lab.actions == [{"action": "MEASURE"}, {"action": "TALK", "arg1": 7},
                           {"chosen_dialog_option_int": 1}, {"chosen_dialog_option_int": 2}]
    assert question(agent, "candidate_a")["status"] == "refuted"
    assert question(agent, "candidate_a")["evidence"] == [2]
    assert question(agent, "candidate_b")["status"] == "supported"
    assert question(agent, "candidate_b")["evidence"] == [3]
    proposals = [row["proposal"] for row in agent.method_results if row["proposal"].get("hypothesis")]
    assert [p["hypothesis"]["setting"] for p in proposals] == [6, 8]
    assert question(agent)["parent_program_path"] == PATH
    assert "answer" not in question(agent) and question(agent)["status"] == "reopened"
    assert not lab.score()["completedSuccessfully"]
    review = agent._request()["research_focus"]
    assert review["stage"] == "review"
    assert review["previous_execution"]["proposal"] == {
        "stop": "The second candidate has an observed activating outcome"}
    assert agent._pending_dependencies(question(agent)) == []


def test_fit_stdout_neither_registers_support_nor_selects_a_candidate(tmp_path):
    agent, lab = prepared(tmp_path)
    agent._execute("run", {"code": "print('candidate_a=6, candidate_b=8')"}, lab)
    assert len(agent.questions) == 2
    assert agent.relation_research == {} and agent.method_results == []
    assert "answer" not in question(agent)
    assert agent._request()["research_focus"]["kind"] == "parent_research"
    assert lab.actions == [{"action": "MEASURE"}]


@pytest.mark.parametrize("ending", ["stop", "step_limit"])
def test_existing_program_survives_and_is_offered_for_continuation(tmp_path, ending):
    agent, lab = prepared(tmp_path)
    agent._request()
    if ending == "stop":
        source = "def select_actions(spec, observations, memory):\n    return {'stop': 'Need a revised experiment'}\n"
    else:
        source = "def select_actions(spec, observations, memory):\n    return {'action': 'run', 'arguments': {'code': 'print(1)'}}\n"
    agent._execute("write", {"path": PATH, "code": source}, lab)
    assert question(agent)["parent_program_path"] == PATH
    request = agent._request()
    if ending == "stop":
        assert request["research_focus"]["stage"] == "review"
        assert request["research_focus"]["previous_execution"]["proposal"] == {
            "stop": "Need a revised experiment"}
    else:
        assert not request.get("research_focus")
    summary = json.loads(request["files"]["method_decision.json"])
    parent = next(item for item in summary["questions"] if item["id"] == "task")
    assert parent["procedure_action"] == {
        "action": "run", "arguments": {"operator": "procedure", "path": PATH}}
    assert (tmp_path / PATH).read_text() == source
    saved = json.loads((tmp_path / "research_state.json").read_text())
    assert next(item for item in saved["questions"] if item["id"] == "task")["parent_program_path"] == PATH
    assert "answer" not in question(agent)
    if ending == "step_limit":
        assert len(agent.method_results) == 12


def test_failed_parent_program_keeps_same_authoring_focus_until_repaired(tmp_path):
    agent, lab = prepared(tmp_path)
    # Exercise the retained manual review route. A live provider now delegates
    # execution repair, covered by the separate continuous-inquiry tests.
    agent.provider = None
    agent._request()
    source = "def select_actions(spec, observations, memory):\n    raise ValueError('Need to revise the plan')\n"
    agent._execute("write", {"path": PATH, "code": source}, lab)
    request = agent._request()
    assert request["research_focus"]["kind"] == "parent_research"
    assert request["research_focus"]["path"] == PATH
    assert agent.method_results[-1]["feedback"].get("error")
    assert lab.environment_steps == 1
    agent._execute("write", {"path": PATH, "code": PROGRAM}, lab)
    assert agent._request()["research_focus"]["previous_execution"]["feedback"]["status"] == "method_stopped"
    assert question(agent, "candidate_a")["status"] == "refuted"


@pytest.mark.parametrize("condition", ["code", "pending_child", "pending_dependency", "binding",
                                       "no_actual_answer", "unreviewed_answer", "no_evidence"])
def test_parent_focus_does_not_override_an_unready_or_existing_route(tmp_path, condition):
    if condition == "no_evidence":
        # An untouched world has neither observations nor answers citing them.
        agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
        agent.lab = DialogLab()
        agent._question({"id": "task", "question": "Find a supported activation setting"})
    else:
        agent, _ = prepared(tmp_path, arm="code_archive" if condition == "code" else "hegel")
    if condition == "pending_child":
        agent._question({"id": "other", "question": "Another unresolved condition", "parent_id": "task"})
    elif condition == "pending_dependency":
        question(agent)["depends_on"] = ["missing"]
    elif condition == "binding":
        question(agent)["binding_path"] = "binding.py"
    elif condition == "no_actual_answer":
        question(agent, "instrument").pop("answer")
    elif condition == "unreviewed_answer":
        question(agent, "instrument").pop("reviewed_through_observation")
        question(agent, "instrument")["acquired_observations"] = []
    assert not agent._request().get("research_focus")


def test_unreviewed_child_analysis_precedes_parent_program_authoring(tmp_path):
    agent, _ = prepared(tmp_path, answer=False)
    assert agent._request()["research_focus"]["kind"] == "analysis"


def test_new_operation_failure_precedes_parent_program_authoring(tmp_path):
    agent, lab = prepared(tmp_path)
    for index in (1, 2):
        agent.observations.append({"index": index, "action": {"action": "ACTIVATE", "arg1": 7},
            "feedback": {"success": False}, "observation": {
                **lab.public_observation(),
                "lastActionMessage": "That object (device) is not activatable/deactivatable."}})
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "operation" and focus["question"]["id"] == "operation_7"


def test_explicit_parent_query_registers_existing_program_without_reauthoring(tmp_path):
    agent, lab = prepared(tmp_path)
    source = "def select_actions(spec, observations, memory):\n    return {'stop': 'Already authored parent'}\n"
    agent._execute("write", {"path": "shared/existing.py", "code": source}, lab)
    agent._execute("query", {"path": "shared/existing.py", "question": {"id": "task"}}, lab)
    assert question(agent)["parent_program_path"] == "shared/existing.py"
    focus = agent._request()["research_focus"]
    assert focus["stage"] == "review" and focus["path"] == "shared/existing.py"
    assert focus["previous_execution"]["proposal"] == {"stop": "Already authored parent"}


def test_question_program_registration_survives_json_restore(tmp_path):
    agent, lab = prepared(tmp_path)
    source = "def select_actions(spec, observations, memory):\n    return {'stop': 'Already authored parent'}\n"
    agent._execute("write", {"path": PATH, "code": source}, lab)
    agent._execute("query", {"path": PATH, "question": {"id": "task"}}, lab)
    restored = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    restored.lab = lab
    state = json.loads((tmp_path / "research_state.json").read_text())
    restored.questions = state["questions"]
    restored.observations = deepcopy(agent.observations)
    assert not restored._request().get("research_focus")
    summary = json.loads(restored.files["method_decision.json"])
    parent = next(item for item in summary["questions"] if item["id"] == "task")
    assert parent["procedure_action"]["arguments"]["path"] == PATH


@pytest.mark.parametrize("route", ["analysis", "tuning"])
def test_parent_focus_preserves_an_authored_custom_analysis_or_tuning_path(tmp_path, route):
    agent, lab = prepared(tmp_path)
    assert agent._request()["research_focus"]["kind"] == "parent_research"
    if route == "analysis":
        path = "parent_analysis.py"
        source = """def answer(spec, observations, memory):
    child = memory['answers']['instrument']
    return {'status': 'supported', 'answer': {'interpreted_reading': child['answer']['reading']},
            'evidence': child['evidence']}
"""
    else:
        path = "shared/tune.py"
        source = """def select_actions(spec, observations, memory):
    return {'actions': [{'action': 'TALK', 'arg1': 7}]}
"""
    written = agent._execute("write", {"path": path, "code": source}, lab)
    assert "error" not in written and (tmp_path / path).read_text() == source
    if route == "analysis":
        agent._execute("evaluate", {"path": path, "question_id": "task"}, lab)
        assert question(agent)["answer"] == {"interpreted_reading": 3}
        assert question(agent)["analysis_path"] == path
        assert lab.environment_steps == 1
    else:
        agent._execute("query", {"path": path, "question": {"id": "task"}}, lab)
        assert question(agent)["parent_program_path"] == path
        assert lab.actions[-1] == {"action": "TALK", "arg1": 7}
    assert not agent._request().get("research_focus")


@pytest.mark.parametrize("action,arguments", [
    ("query", {"path": "shared/tune.py", "question": {"id": "task"}}),
    ("evaluate", {"path": "parent_analysis.py", "question_id": "task"}),
    ("run", {"operator": "procedure", "path": "shared/tune.py"}),
])
def test_parent_prompt_allows_existing_routes_while_requesting_live_continuation(
        tmp_path, monkeypatch, action, arguments):
    agent, _ = prepared(tmp_path)
    request = agent._request()

    def run(command, prompt, timeout, cwd):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert schema["properties"]["action"]["enum"] == code_agent._ACTION_SCHEMA["properties"]["action"]["enum"]
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        assert "existing procedure executor" in instructions
        assert "fresh public state" in instructions
        assert "actual response before choosing the next step" in instructions
        assert "Existing programs and analyses remain usable" in instructions
        payload = {"action": action, "arguments_json": json.dumps(arguments),
                   "research_question": None, "research_notes": "Continue from the instrument answer."}
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(payload))
        return 0, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 31, "output_tokens": 7}}), "", False

    monkeypatch.setattr(code_agent, "_run", run)
    reply = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action(request)
    assert reply["action"] == action
    assert {key: reply["arguments"][key] for key in arguments} == arguments

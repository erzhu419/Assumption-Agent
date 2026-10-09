"""Registered public methods execute on fresh feedback, not inherited conclusions."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import public_reference_method
from hegel_machine.cognition import code_agent
from hegel_machine.cognition.code_provider_context import prepare_code_context
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider


METHOD_PATH = "shared/reference_control.py"
SOURCE = Path(public_reference_method.__file__).read_text()
PARAMETERS = {"target": 7, "perturb_label": "Increase by 1", "restore_label": "Decrease by 1"}
PRIOR_EXECUTION = {"origin": "operator-authored public diagnostic", "observations": [
    {"index": 10, "action": {"action": "TALK", "arg1": 7}, "feedback": {"success": True},
     "observation": {"setting": 5.5, "active": True}}]}


class ReferenceLab:
    """A public menu changes numbering after each executed control."""

    def __init__(self, *, fail_adjustment=False, missing_restore=False, baseline_active=True):
        self.environment_steps = 0
        self.actions = []
        self.settings = {7: 5.5, 9: 12.25}
        self.baselines = dict(self.settings)
        self.target = 7
        self.in_dialog = False
        self.fail_adjustment = fail_adjustment
        self.missing_restore = missing_restore
        self.baseline_active = baseline_active

    def public_observation(self):
        changed = self.settings[self.target] != self.baselines[self.target]
        options = {"2": "Decrease by 1", "6": "Increase by 1", "5": "Exit"} if changed else {
            "4": "Increase by 1", "8": "Decrease by 1", "3": "Exit"}
        if self.missing_restore:
            options = {key: value for key, value in options.items() if value != "Decrease by 1"}
        return {"taskProgress": [{"description": "Explain the control response", "completed": False}],
                "target": self.target, "setting": self.settings[self.target],
                "active": self.baseline_active and not changed,
                "dialog_box": {"is_in_dialog": self.in_dialog, "dialogOptions": options}}

    def capabilities(self):
        return {"actions": ["TELEPORT_TO_OBJECT", "TALK"]}

    def act(self, action):
        success = True
        if "chosen_dialog_option_int" in action:
            label = self.public_observation()["dialog_box"]["dialogOptions"].get(
                str(action["chosen_dialog_option_int"]))
            success = self.in_dialog and label is not None
            if label == "Exit":
                self.in_dialog = False
            else:
                success = success and not self.fail_adjustment
                if success:
                    self.settings[self.target] += 1 if label == "Increase by 1" else -1
        else:
            self.target = action["arg1"]
            self.in_dialog = action["action"] == "TALK"
        self.actions.append(deepcopy(action))
        self.environment_steps += 1
        return {"action": deepcopy(action), "feedback": {"success": success},
                "observation": self.public_observation()}

    def score(self):
        return {"completed": False, "completedSuccessfully": False, "scoreNormalized": 0.}


def prepared(tmp_path, arm="hegel", requires=(), **options):
    agent = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm=arm,
                                limits={**LIMITS, "environment_steps": 60})
    agent.lab = lab = ReferenceLab(**options)
    agent._question({"id": "task", "question": "Explain the control response"})
    agent._execute("write", {"path": METHOD_PATH, "code": SOURCE}, lab)
    agent._execute("write", {"path": "shared/prior_execution.json",
                             "code": json.dumps(PRIOR_EXECUTION)}, lab)
    agent._execute("run", {"operator": "method", "id": "reference", "path": METHOD_PATH,
        "parameters": PARAMETERS, "max_steps": 8, "requires": list(requires),
        "applicability": "A caller-selected object offers the declared controls",
        "provenance": "operator-authored reusable procedure",
        "evidence_file": "shared/prior_execution.json"}, lab)
    return agent, lab


def invoke(agent, lab, question_id="physical", **arguments):
    return agent._execute("run", {"operator": "procedure", "method_id": "reference",
        "research_question": {"id": question_id, "question": "Does the response return?",
                              "parent_id": "task", "depends_on": []}, **arguments}, lab)


def test_both_arms_receive_identical_registered_source_support_and_executable_action(tmp_path):
    views = []
    for arm in ("hegel", "code_archive"):
        workspace = tmp_path / arm
        agent, _ = prepared(workspace, arm)
        agent._save_state()
        context = prepare_code_context(workspace / "provider", agent._request(), schema={}, instructions="")
        method = context["method_decision"]["methods"][0]
        assert context["files"][METHOD_PATH] == SOURCE
        assert method["prior_execution"] == PRIOR_EXECUTION
        assert method["provenance"] == "operator-authored reusable procedure"
        assert method["execute_action"] == {"action": "run", "arguments": {
            "operator": "procedure", "method_id": "reference"}}
        assert all("answer" not in q for q in agent.questions)
        views.append(method)
    assert views[0] == views[1]


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_registered_method_reads_live_menu_and_restarts_after_inherited_execution(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm)
    first = invoke(agent, lab)
    assert first["status"] == "method_stopped"
    assert lab.actions == [{"action": "TELEPORT_TO_OBJECT", "arg1": 7},
                           {"action": "TALK", "arg1": 7},
                           {"chosen_dialog_option_int": 4}, {"chosen_dialog_option_int": 2}]
    assert first["steps"][-1]["proposal"]["phase_observations"] == {
        "approach": 0, "baseline": 1, "perturb": 2, "restore": 3}
    assert all(step["arguments"]["procedure_start"] == 0 for step in first["steps"])

    second = invoke(agent, lab, question_id="fresh")
    assert second["status"] == "method_stopped"
    assert lab.actions[4:] == [{"chosen_dialog_option_int": 3},
        {"action": "TELEPORT_TO_OBJECT", "arg1": 7}, {"action": "TALK", "arg1": 7},
        {"chosen_dialog_option_int": 4}, {"chosen_dialog_option_int": 2}]
    assert second["steps"][-1]["proposal"]["phase_observations"] == {
        "exit": 4, "approach": 5, "baseline": 6, "perturb": 7, "restore": 8}
    assert all(step["arguments"]["procedure_start"] == 4 for step in second["steps"])
    fresh = next(q for q in agent.questions if q["id"] == "fresh")
    assert fresh["acquired_observations"] == [4, 5, 6, 7, 8]
    assert fresh["status"] == "open" and "answer" not in fresh
    assert lab.settings[7] == 5.5


@pytest.mark.parametrize("reverse", [False, True])
def test_call_parameters_rebind_reference_and_controls_without_mutating_saved_method(tmp_path, reverse):
    agent, lab = prepared(tmp_path)
    parameters = {"target": 9}
    if reverse:
        parameters.update(perturb_label="Decrease by 1", restore_label="Increase by 1")
    result = invoke(agent, lab, parameters=parameters)
    assert result["status"] == "method_stopped"
    assert lab.actions == [{"action": "TELEPORT_TO_OBJECT", "arg1": 9},
                           {"action": "TALK", "arg1": 9},
                           {"chosen_dialog_option_int": 8 if reverse else 4},
                           {"chosen_dialog_option_int": 6 if reverse else 2}]
    assert [r["observation"]["setting"] for r in agent.observations] == [
        12.25, 12.25, 11.25 if reverse else 13.25, 12.25]
    assert agent.methods[0]["parameters"] == PARAMETERS
    assert lab.settings[7] == 5.5


def test_pending_registered_requirement_prevents_the_whole_procedure(tmp_path):
    agent, lab = prepared(tmp_path, requires=["control_meaning"])
    result = invoke(agent, lab)
    assert result["error"] == "Method requirements need renewed evidence"
    assert not lab.actions and not agent.observations and agent.program_executions == 0
    agent._save_state()
    method = json.loads(agent.files["method_decision.json"])["methods"][0]
    assert method["blocked_by"] == ["control_meaning"]


@pytest.mark.parametrize("case", ["failed_adjustment", "missing_restore", "inactive_baseline"])
def test_incomplete_or_physically_unsuccessful_method_never_invents_support(tmp_path, case):
    agent, lab = prepared(tmp_path, fail_adjustment=case == "failed_adjustment",
                          missing_restore=case == "missing_restore", baseline_active=case != "inactive_baseline")
    if case == "failed_adjustment":
        agent.provider = ScriptedProvider([("finish", {"lesson": "Failed control remains unresolved."})])
    result = invoke(agent, lab)
    assert result["status"] == ("method_error" if case == "failed_adjustment" else "method_stopped")
    question = next(q for q in agent.questions if q["id"] == "physical")
    assert question["status"] == ("reopened" if case == "failed_adjustment" else "open")
    assert "answer" not in question
    assert question["acquired_observations"] == list(range(len(lab.actions)))
    assert len(lab.actions) == (4 if case == "inactive_baseline" else 3)
    if case == "missing_restore":
        assert "No unique live menu option" in result["steps"][-1]["proposal"]["stop"]
    if case == "inactive_baseline":
        assert not any(row["observation"]["active"] for row in agent.observations)


def test_parameterized_contrast_returns_only_fresh_authored_physical_analysis(tmp_path):
    agent, lab = prepared(tmp_path)
    invoke(agent, lab, question_id="old_trial")
    source = f'''def design(spec, observations, memory):
    return {{'hypotheses': [{{'id': 'restores', 'claim': 'The same control restores the response'}},
                           {{'id': 'extra', 'claim': 'An extra operation is necessary'}}],
        'experiments': [{{'id': 'reference', 'outcome': 'Active, inactive, then active on one reference',
            'procedure': {{'path': {METHOD_PATH!r}, 'max_steps': 8, 'parameters': {PARAMETERS!r}}},
            'predictions': {{'restores': True, 'extra': False}}}}]}}

def answer(spec, observations, memory):
    q = next(q for q in memory['questions'] if q['id'] == 'comparison')
    start = q['experiment_choice']['started_before_observation']
    rows = [r for r in observations if r['index'] >= start]
    actual = rows[-3:]
    restored = (len(rows) == 5 and all(r['feedback']['success'] for r in rows)
                and [r['observation']['active'] for r in actual] == [True, False, True]
                and actual[0]['observation']['setting'] == actual[-1]['observation']['setting'])
    return {{'status': 'supported' if restored else 'inconclusive',
            'answer': {{'reference_restored_without_extra_operation': restored}},
            'evidence': [r['index'] for r in rows]}}
'''
    result = agent._execute("run", {"operator": "contrast", "path": "shared/comparison.py", "code": source,
        "question": {"id": "comparison", "parent_id": "task", "question": "Is an extra operation necessary?",
                     "depends_on": []}}, lab)
    assert result["procedure"]["status"] == "method_stopped"
    child = next(q for q in agent.questions if q["id"] == "comparison")
    assert child["status"] == "supported"
    assert child["answer"] == {"reference_restored_without_extra_operation": True}
    assert child["evidence"] == child["acquired_observations"] == [4, 5, 6, 7, 8]
    assert agent.child_answers_returned == 1
    parent = next(q for q in agent.questions if q["id"] == "task")
    assert parent["status"] == "reopened" and parent["reopened_by"] == "comparison"


def test_parent_contrast_schema_allows_selecting_existing_method_and_retains_new_evidence(tmp_path, monkeypatch):
    agent, lab = prepared(tmp_path)
    agent._execute("query", {"question": {"id": "unresolved", "parent_id": "task",
        "question": "Which condition remains unresolved?"},
        "actions": [{"action": "TELEPORT_TO_OBJECT", "arg1": 7}]}, lab)
    parent_path = "shared/existing_parent.py"
    parent_source = "def select_actions(spec, observations, memory):\n    return {'stop': 'Need reference evidence'}\n"
    agent._execute("write", {"path": parent_path, "code": parent_source}, lab)
    agent._execute("query", {"path": parent_path, "question": {"id": "task"}}, lab)
    agent._execute("write", {"path": "shared/unresolved_answer.py", "code":
        "def answer(spec, observations, memory):\n"
        "    return {'status': 'inconclusive', 'answer': 'Physical reference remains untested', 'evidence': [0]}\n"}, lab)
    agent._execute("evaluate", {"path": "shared/unresolved_answer.py", "question_id": "unresolved"}, lab)
    request = agent._request()
    assert request["research_focus"]["stage"] == "contrast"
    new_question = {"id": "reference_replication", "parent_id": "task", "depends_on": [],
                    "question": "Does the chosen reference return after restoring its control?"}

    def run(command, prompt, timeout, cwd):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert schema["properties"]["action"]["enum"] == code_agent._ACTION_SCHEMA["properties"]["action"]["enum"]
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        assert "method's execute_action" in instructions
        payload = {"action": "run", "arguments_json": json.dumps({
            "operator": "procedure", "method_id": "reference"}),
            "research_question": new_question, "research_notes": "Replicate the observed control, then interpret it."}
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(payload))
        return 0, json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 31, "output_tokens": 7}}), "", False

    monkeypatch.setattr(code_agent, "_run", run)
    reply = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action(request)
    assert reply["action"] == "run" and reply["arguments"]["method_id"] == "reference"
    result = agent._execute(reply["action"], reply["arguments"], lab)
    assert result["status"] == "method_stopped" and agent.research_focus is None
    child = next(q for q in agent.questions if q["id"] == "reference_replication")
    assert child["acquired_observations"] == [1, 2, 3, 4] and "answer" not in child
    parent = next(q for q in agent.questions if q["id"] == "task")
    assert parent["reviewed_child_answers"]["unresolved"]["status"] == "inconclusive"
    assert (tmp_path / parent_path).read_text() == parent_source
    focus = agent._request()["research_focus"]
    assert focus["kind"] == "analysis" and focus["question"]["id"] == "reference_replication"
    assert focus["evidence"] == [1, 2, 3, 4]

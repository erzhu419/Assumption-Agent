"""Stopped research remains reviewable without making stopping an error."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.cognition import code_agent
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import LIMITS, ScriptedProvider
from test_discoveryworld_parent_research import PATH, prepared, question


STOP = """def select_actions(spec, observations, memory):
    return {'stop': 'The available intervention does not distinguish my candidates'}
"""


def stopped(tmp_path, *, arm="hegel"):
    agent, lab = prepared(tmp_path, arm=arm)
    agent._execute("write", {"path": PATH, "code": STOP}, lab)
    agent._execute("query", {"path": PATH, "question": {"id": "task"}}, lab)
    return agent, lab


def test_stop_is_reviewed_once_without_rerun_or_scientific_verdict(tmp_path):
    agent, lab = stopped(tmp_path)
    executions = agent.program_executions
    first = agent._request()["research_focus"]
    again = agent._request()["research_focus"]
    assert first == again
    assert first["stage"] == "review" and first["kind"] == "parent_research"
    assert first["origin"] == "host_parent_program_review"
    assert first["path"] == PATH
    assert first["previous_execution"] == agent.method_results[-1]
    assert [row["id"] for row in first["returned_children"]] == ["instrument"]
    assert "reviewed_program_step" not in question(agent)
    assert agent.program_executions == executions and lab.environment_steps == 1
    assert "answer" not in question(agent)
    assert question(agent)["status"] == "reopened"

    agent._execute("run", {"code": "print('Inspect paid data before designing another experiment')"}, lab)
    assert question(agent)["reviewed_program_step"] == first["previous_execution"]["step"]
    assert not agent._request().get("research_focus")
    assert not agent._request().get("research_focus")
    assert agent.program_executions == executions and lab.environment_steps == 1


def test_explicit_paid_analysis_is_a_valid_review_without_new_acquisition(tmp_path):
    agent, lab = stopped(tmp_path)
    analysis = """def answer(spec, observations, memory):
    return {'status': 'inconclusive', 'answer': 'No discriminating intervention yet',
            'evidence': memory['answers']['instrument']['evidence']}
"""
    agent._execute("write", {"path": "parent_analysis.py", "code": analysis}, lab)
    focus = agent._request()["research_focus"]
    agent._execute("evaluate", {"path": "parent_analysis.py", "question_id": "task"}, lab)
    assert question(agent)["reviewed_program_step"] == focus["previous_execution"]["step"]
    assert question(agent)["status"] == "inconclusive"
    assert question(agent)["evidence"] == [0]
    assert lab.environment_steps == 1
    assert not agent._request().get("research_focus")


def test_revised_program_runs_and_only_its_new_stop_reopens_review(tmp_path):
    agent, lab = stopped(tmp_path)
    first = agent._request()["research_focus"]["previous_execution"]
    revised = """def select_actions(spec, observations, memory):
    if not spec['current_observation']['dialog_box']['is_in_dialog']:
        return [{'action': 'TALK', 'arg1': 7}]
    return {'stop': 'Controls are visible; select the discriminating setting next'}
"""
    result = agent._execute("write", {"path": PATH, "code": revised}, lab)
    assert result["procedure"]["status"] == "method_stopped"
    assert lab.actions == [{"action": "MEASURE"}, {"action": "TALK", "arg1": 7}]
    second = agent._request()["research_focus"]
    assert question(agent)["reviewed_program_step"] == first["step"]
    assert second["previous_execution"]["step"] > first["step"]
    assert second["previous_execution"]["feedback"]["reason"].startswith("Controls are visible")
    assert "answer" not in question(agent)


def test_review_marker_and_latest_execution_survive_json_restore(tmp_path):
    agent, lab = stopped(tmp_path)
    agent._request()
    agent._execute("run", {"code": "print('Retain the unresolved stop')"}, lab)
    agent._request()
    state = json.loads((tmp_path / "research_state.json").read_text())
    restored = DiscoveryWorldAgent(ScriptedProvider([]), tmp_path, arm="hegel", limits=LIMITS)
    restored.lab = lab
    restored.questions = state["questions"]
    restored.method_results = state["method_results"]
    restored.observations = deepcopy(agent.observations)
    request = restored._request()
    assert not request.get("research_focus")
    summary = json.loads(request["files"]["method_decision.json"])
    parent = next(row for row in summary["questions"] if row["id"] == "task")
    assert parent["last_program_result"] == restored.method_results[-1]
    assert parent["procedure_action"]["arguments"]["path"] == PATH


@pytest.mark.parametrize("feedback", [
    {"status": "method_stopped", "reason": "Need another observable"},
    {"error": "ValueError: missing observed response"},
    {"after_observation_error": "KeyError: outcome"},
])
def test_review_uses_latest_matching_program_result_not_an_unrelated_program(tmp_path, feedback):
    agent, _ = stopped(tmp_path)
    agent.method_results[-1]["feedback"] = feedback
    own = deepcopy(agent.method_results[-1])
    agent.method_results.append({"step": len(agent.method_results), "path": "unrelated.py",
        "proposal": {"stop": "Other question"},
        "feedback": {"status": "method_stopped", "reason": "Other question"}})
    request = agent._request()
    assert request["research_focus"]["previous_execution"] == own
    summary = json.loads(request["files"]["method_decision.json"])
    assert next(row for row in summary["questions"] if row["id"] == "task")["last_program_result"] == own


@pytest.mark.parametrize("condition", ["code", "dependency", "binding", "later_success"])
def test_review_respects_existing_research_routes(tmp_path, condition):
    agent, _ = stopped(tmp_path, arm="code_archive" if condition == "code" else "hegel")
    if condition == "dependency":
        question(agent)["depends_on"] = ["unresolved"]
    elif condition == "binding":
        question(agent)["binding_path"] = "binding.py"
    elif condition == "later_success":
        agent.method_results.append({"step": len(agent.method_results), "path": PATH,
            "proposal": {"actions": [{"action": "TALK", "arg1": 7}]},
            "feedback": {"observation_indices": [0]}})
    assert not agent._request().get("research_focus")


@pytest.mark.parametrize("arm", ["hegel", "code_archive"])
def test_program_scope_preserves_global_history_and_excludes_other_paths(tmp_path, arm):
    agent, lab = prepared(tmp_path, arm=arm)
    agent.method_results = [{"step": i, "path": "other.py", "proposal": {"stop": "Other"},
        "feedback": {"status": "method_stopped"}} for i in range(4)]
    source = """def select_actions(spec, observations, memory):
    assert memory['program_path'] == 'shared/research_task.py'
    own = memory['program_results']
    assert all(row['path'] == memory['program_path'] for row in own)
    assert len(memory['method_results']) == 4 + len(own)
    if not own:
        return [{'action': 'TALK', 'arg1': 7}]
    assert own[-1]['feedback']['observation_indices'] == [1]
    return {'stop': 'Own experiment completed once'}
"""
    agent._execute("write", {"path": PATH, "code": source}, lab)
    result = agent._execute("run", {"operator": "procedure", "path": PATH}, lab)
    assert result["status"] == "method_stopped"
    assert len(result["steps"]) == 2
    assert lab.actions == [{"action": "MEASURE"}, {"action": "TALK", "arg1": 7}]
    assert len(agent.method_results) == 6


def test_provider_decode_failure_does_not_acknowledge_review(tmp_path):
    agent, lab = stopped(tmp_path)

    class Provider:
        def __init__(self):
            self.requests = []

        def next_action(self, request):
            self.requests.append(deepcopy(request))
            if len(self.requests) == 1:
                return {"action": "provider_error", "error": {"decode_error": "Invalid JSON"}}
            return {"action": "finish", "arguments": {"lesson": "Insufficient discrimination"}}

    agent.provider = Provider()
    agent.run(lab)
    first, second = agent.provider.requests
    assert first["research_focus"] == second["research_focus"]
    assert "reviewed_program_step" not in next(
        row for row in second["research"]["questions"] if row["id"] == "task")
    assert question(agent)["reviewed_program_step"] == first["research_focus"]["previous_execution"]["step"]
    assert lab.environment_steps == 1


def test_review_prompt_preserves_full_actions_and_acknowledges_existing_program(tmp_path, monkeypatch):
    agent, _ = stopped(tmp_path)
    request = agent._request()

    def run(command, prompt, timeout, cwd):
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert schema["properties"]["action"]["enum"] == code_agent._ACTION_SCHEMA["properties"]["action"]["enum"]
        instructions = (Path(cwd) / "code_agent_instructions.txt").read_text()
        assert "no executable research program yet" not in instructions
        assert "review" in instructions.lower()
        assert json.loads(prompt)["research_focus"]["previous_execution"] == agent.method_results[-1]
        payload = {"action": "evaluate", "arguments_json": json.dumps({"path": "analysis.py", "question_id": "task"}),
                   "research_question": None, "research_notes": "The existing program stopped without a new test."}
        Path(command[command.index("--output-last-message") + 1]).write_text(json.dumps(payload))
        return 0, "", "", False

    monkeypatch.setattr(code_agent, "_run", run)
    reply = code_agent.CodexCodeProvider(tmp_path, executable="protocol-test").next_action(request)
    assert reply["action"] == "evaluate"

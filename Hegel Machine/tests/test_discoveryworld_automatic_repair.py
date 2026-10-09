"""Actual failed execution can open a local researcher and resume its caller."""

from copy import deepcopy
import json

import pytest

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from test_discoveryworld_agent import ANALYZE, InstrumentLab, LIMITS, ScriptedProvider
from test_discoveryworld_method_reuse import ReferenceLab


PATH = "shared/experiment.py"
BROKEN = "def select_actions(spec, observations, memory):\n    return [}\n"


class RepairProvider:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def next_action(self, request):
        self.requests.append(deepcopy(request))
        step = next(self.steps)
        action, arguments = step(request) if callable(step) else step
        return {"action": action, "arguments": deepcopy(arguments),
                "usage": {"total_tokens": 7}}


def prepared(tmp_path, provider, *, lab=None, arm="hegel", limits=None, source=BROKEN):
    agent = DiscoveryWorldAgent(provider, tmp_path, arm=arm,
        limits={**LIMITS, "environment_steps": 60, **(limits or {})})
    agent.lab = lab or InstrumentLab()
    agent._question({"id": "task", "question": "Read the instrument and tune its response"})
    agent._execute("write", {"path": PATH, "code": source}, agent.lab)
    return agent, agent.lab


def run_parent(agent, *, max_steps=8):
    return agent._execute("run", {"operator": "procedure", "path": PATH,
        "max_steps": max_steps, "parameters": {"gain": 2},
        "question": {"id": "task"}}, agent.lab)


def rewrite_from_failure(request):
    question_id = request["llm_inquiry"]["question_id"]
    source = (
        "def select_actions(spec, observations, memory):\n"
        f"    repaired = memory['answers'].get({question_id!r}, {{}})\n"
        "    if repaired.get('status') == 'supported':\n"
        "        gain = memory['program_arguments']['parameters']['gain']\n"
        "        return [{'action': 'TUNE', 'arg1': gain * repaired['answer']['gain']}]\n"
        "    if observations:\n"
        "        return {'stop': 'Measured; interpret the instrument before resuming.'}\n"
        "    return [{'action': 'MEASURE'}]\n"
    )
    return "write", {"path": PATH, "code": source}


def analyze_repair(request):
    return "evaluate", {"question_id": request["llm_inquiry"]["question_id"],
        "path": "shared/repair_answer.py", "code": ANALYZE}


REPAIR_STEPS = [rewrite_from_failure,
    ("run", {"operator": "procedure", "path": PATH, "max_steps": 4}), analyze_repair]


def test_syntax_failure_automatically_researches_repairs_and_resumes_real_parent(tmp_path):
    provider = RepairProvider(REPAIR_STEPS)
    agent, lab = prepared(tmp_path, provider)
    result = run_parent(agent)

    assert lab.actions == [{"action": "MEASURE"}, {"action": "TUNE", "arg1": 6}]
    assert result["status"] == "completed" and lab.success
    question_id = provider.requests[0]["llm_inquiry"]["question_id"]
    local = provider.requests[0]
    assert local["research_root"]["id"] == question_id
    assert local["enclosing_research_root"]["id"] == "task"
    assert local["procedure_repair"]["path"] == PATH
    assert "Solve the task in current_observation.taskProgress" not in local["task_interface"]
    assert "local deliverable is a repaired procedure" in local["task_interface"]
    question = next(q for q in agent.questions if q["id"] == question_id)
    failure = json.loads((tmp_path / question["failure_evidence_file"]).read_text())
    assert "SyntaxError" in failure["last_steps"][-1]["feedback"]["error"]
    assert (tmp_path / result["repair"]["original_source_path"]).read_text() == BROKEN
    assert question["parent_id"] == "task" and question["status"] == "supported"
    assert question["evidence"] == [0] and question["answer"] == {"gain": 3}
    assert all(r["llm_inquiry"]["question_id"] == question_id for r in provider.requests)
    assert agent.child_answers_returned == 1
    assert agent.costs["model_calls"] == len(provider.requests) == 3
    assert agent.costs["reported_tokens"] == 21
    # The continuation retained the caller's parameters; the child invoked the
    # same program without them and did not itself perform the intervention.
    assert agent.events[-1]["action"] == "evaluate"


LOOP = """def select_actions(spec, observations, memory):
    target = 9 if spec['current_observation']['target'] == 7 else 7
    return [{'action': 'TELEPORT_TO_OBJECT', 'arg1': target}]
"""


def test_real_navigation_loop_triggers_a_researcher_at_the_procedure_limit(tmp_path):
    provider = RepairProvider([("finish", {"lesson": "The repeated moves need a new plan."})])
    agent, lab = prepared(tmp_path, provider, lab=ReferenceLab(), source=LOOP)
    result = run_parent(agent, max_steps=8)

    assert len(lab.actions) == 8
    assert [a["arg1"] for a in lab.actions] == [9, 7] * 4
    assert len(provider.requests) == 1 and provider.requests[0]["llm_inquiry"]["depth"] == 1
    assert result["status"] == "method_step_limit"
    assert result["repair"]["trigger"]["kind"] == "control_cycle"
    assert agent.status == "running" and agent.child_answers_returned == 0


@pytest.mark.parametrize("case", ["finite_plan", "fresh_measurements"])
def test_finished_plan_or_genuinely_new_evidence_does_not_trigger_repair(tmp_path, case):
    provider = ScriptedProvider([])
    if case == "finite_plan":
        source = LOOP.replace("    target =", "    if len(observations) >= 4:\n"
            "        return {'stop': 'Four planned reference moves are complete.'}\n    target =")
        lab, expected_status, actions = ReferenceLab(), "method_stopped", 4
    else:
        source = "def select_actions(spec, observations, memory):\n    return [{'action': 'MEASURE'}]\n"
        lab, expected_status, actions = InstrumentLab(readings=range(8)), "method_step_limit", 8
    agent, lab = prepared(tmp_path, provider, lab=lab, source=source)
    result = run_parent(agent, max_steps=8)

    assert result["status"] == expected_status and len(lab.actions) == actions
    assert not provider.requests and not agent.llm_inquiries


def test_error_within_repair_returns_to_that_child_without_forking_it_again(tmp_path):
    provider = RepairProvider([
        ("run", {"operator": "procedure", "path": PATH, "max_steps": 3}),
        ("finish", {"lesson": "Syntax still needs a correction."}),
    ])
    agent, lab = prepared(tmp_path, provider)
    result = run_parent(agent)

    assert result["status"] == "method_error" and not lab.actions
    assert len(agent.llm_inquiries) == 1 and len(provider.requests) == 2
    assert [r["llm_inquiry"]["depth"] for r in provider.requests] == [1, 1]
    assert provider.requests[1]["events"][-1]["feedback"]["status"] == "method_error"
    assert agent.child_answers_returned == 0 and agent.status == "running"


def test_edited_code_can_resume_after_local_finish_without_scientific_support(tmp_path):
    source = """def select_actions(spec, observations, memory):
    if observations:
        return {'stop': 'A measurement is available for subsequent interpretation.'}
    return [{'action': 'MEASURE'}]
"""
    provider = RepairProvider([("write", {"path": PATH, "code": source}),
        ("finish", {"lesson": "Syntax repaired; the caller can try this code."})])
    agent, lab = prepared(tmp_path, provider)
    result = run_parent(agent)

    assert result["status"] == "method_stopped"
    assert lab.actions == [{"action": "MEASURE"}]
    assert agent.status == "running" and agent.child_answers_returned == 0
    assert all(q["status"] != "supported" for q in agent.questions)


def test_automatic_repair_spends_only_remaining_global_calls_and_never_invents_support(tmp_path):
    provider = RepairProvider(REPAIR_STEPS)
    agent, lab = prepared(tmp_path, provider, limits={"max_model_calls": 2})
    run_parent(agent)

    assert agent.costs["model_calls"] == len(provider.requests) == 2
    assert agent.costs["reported_tokens"] == 14
    assert lab.actions == [{"action": "MEASURE"}] and not lab.completed
    assert agent.child_answers_returned == 0
    assert all(q["status"] != "supported" for q in agent.questions)
    assert next(iter(agent.llm_inquiries.values()))["status"] == "model_call_budget_exhausted"


def test_new_failure_in_resumed_parent_opens_another_repair_on_the_same_account(tmp_path):
    stopped = "def select_actions(spec, observations, memory):\n    return {'stop': 'Return for the next research decision'}\n"
    provider = RepairProvider([
        ("write", {"path": PATH, "code": LOOP}), ("finish", {"lesson": "Syntax repaired."}),
        ("write", {"path": PATH, "code": stopped}), ("finish", {"lesson": "Removed the navigation loop."}),
    ])
    agent, lab = prepared(tmp_path, provider, lab=ReferenceLab())
    result = run_parent(agent)
    assert result["status"] == "method_stopped"
    assert result["repair"]["trigger"]["kind"] == "execution_error"
    assert result["continuation"]["repair"]["trigger"]["kind"] == "control_cycle"
    assert len(agent.llm_inquiries) == 2
    assert len(provider.requests) == agent.costs["model_calls"] == 4
    assert len(lab.actions) == 8 and agent.child_answers_returned == 0


@pytest.mark.parametrize("mode", ["code", "frozen"])
def test_code_control_and_frozen_replay_do_not_silently_add_a_repair_model(tmp_path, mode):
    provider = ScriptedProvider([]) if mode == "code" else None
    agent, lab = prepared(tmp_path, provider, arm="code_archive" if mode == "code" else "hegel")
    result = run_parent(agent)

    assert result["status"] == "method_error" and not lab.actions
    assert not agent.llm_inquiries and agent.costs["model_calls"] == 0

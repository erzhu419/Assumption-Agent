"""Saved local research retains its original paid account when resumed."""

from copy import deepcopy
import json
import py_compile
from pathlib import Path

from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_method_repair import restore_repair
from test_discoveryworld_agent import InstrumentLab, LIMITS, ScriptedProvider


def test_repair_resume_retains_supported_child_source_and_eight_paid_calls(tmp_path):
    original = DiscoveryWorldAgent(None, tmp_path / "old", arm="hegel",
        limits={**LIMITS, "max_model_calls": 32, "environment_steps": 100})
    original.lab = InstrumentLab(readings=(3,) * 65)
    original._question({"id": "task", "question": "Continue the existing investigation"})
    child = original._question({"id": "operation", "question": "Does the interface work?",
        "parent_id": "task"})
    for _ in range(65):
        record = original.lab.act({"action": "MEASURE"})
        original.observations.append({"index": len(original.observations), **record})
    original._record_answer(child, {"status": "supported", "answer": {"reading": 3},
                                    "evidence": [64]})
    source = """def select_actions(spec, observations, memory):
    assert memory['program_arguments']['procedure_start'] == 64
    assert memory['answers']['operation']['status'] == 'supported'
    return {'stop': 'The learned operation and invocation boundary survived'}
"""
    original._execute("write", {"path": METHOD_PATH, "code": source}, original.lab)
    # Importing an authored binding during paid-evidence comparison generates
    # bytecode in shared/. It is a runtime cache, not a text research artifact.
    cached = Path(py_compile.compile(str(original.workspace / METHOD_PATH), doraise=True))
    original.llm_inquiries = {"operation": {"status": "answered", "model_calls": 8,
        "notes": "Retain this operation", "events": [], "invocations": 1}}
    original.events = [{"action": "evaluate", "arguments": {"question_id": "operation"}}]
    original.method_results = [{"step": 0, "path": METHOD_PATH, "proposal": None,
                                "feedback": {"status": "method_stopped"}}]
    original.costs.update(model_calls=8, reported_tokens=80, environment_steps=65)
    (original.workspace / "research_notes.md").write_text("Parent notebook")
    original._save_state()
    previous = {**original.result(), "episode_status": "running"}

    provider = ScriptedProvider([("provider_error", {})] * 24)
    agent = DiscoveryWorldAgent(provider, tmp_path / "resumed", arm="hegel",
        limits=deepcopy(original.limits))
    agent.lab = InstrumentLab(readings=(3,) * 65)
    replay = restore_repair(agent, agent.lab, previous, original.workspace)
    assert replay["environment_steps"] == 65
    assert agent.costs["model_calls"] == 8 and agent.limits["max_model_calls"] == 32
    assert agent.costs["reported_tokens"] == 80
    assert agent.questions == previous["research"]["questions"]
    assert agent.llm_inquiries == previous["research"]["llm_inquiries"]
    assert agent.events == previous["events"] and agent.method_results == previous["method_results"]
    assert (agent.workspace / METHOD_PATH).read_text() == source
    assert not (agent.workspace / cached.relative_to(original.workspace)).exists()
    assert (agent.workspace / "research_notes.md").read_text() == "Parent notebook"
    result = agent._run_procedure(agent.lab, METHOD_PATH, query_arguments={
        "procedure_start": 64, "parameters": {}, "research_question": {"id": "task"}})
    assert result["status"] == "method_stopped" and not provider.requests
    # Account checks use the same inquiry entry, without live model calls.
    class DecodeProvider:
        def next_action(self, request):
            provider.requests.append(deepcopy(request))
            return {"action": "provider_error", "error": {"decode_error": "retry"},
                    "usage": {"total_tokens": 1}}
    agent.provider = DecodeProvider()
    outcome = agent._execute("run", {"operator": "inquire", "max_model_calls": 100,
        "question": {"id": "next", "question": "Continue the next prerequisite", "parent_id": "task"}},
        agent.lab)
    assert outcome["status"] == "model_call_budget_exhausted"
    assert len(provider.requests) == 24 and agent.costs["model_calls"] == 32
    assert agent.costs["reported_tokens"] == 104
    assert next(q for q in agent.questions if q["id"] == "operation")["status"] == "supported"

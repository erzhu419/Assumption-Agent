"""Reconstruct the saved model's state immediately after its seventh decision.

No provider is instantiated. Replayed world actions are already observed history,
not new experimental evidence or a new autonomous decision.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from scripts.run_discoveryworld_prerequisite_check import freeze_prefix, replay_prefix, write_json

SOURCE = ROOT / "artifacts/discoveryworld_counterexample_revision_20261007/dw_04_normal_s0/hegel"
DECISIONS = ROOT / "artifacts/discoveryworld_cited_answer_adoption_20261008/dw_04_normal_s0/hegel"
OUTPUT = ROOT / "artifacts/discoveryworld_draft_counterexample_seed_20261008"
QUESTION = "test_crystal3_1296"


def main():
    OUTPUT.mkdir(exist_ok=False)
    plan, prefix = freeze_prefix(SOURCE, OUTPUT)
    saved = json.loads((DECISIONS / "result.json").read_text())
    decisions = saved["events"][saved["prefix_event_count"]:][:7]
    expected = json.loads((DECISIONS / "agent/laboratory_observations.json").read_text())
    assert len(prefix["records"]) == 438 and len(decisions) == 7
    assert decisions[-1]["action"] == "evaluate"
    assert decisions[-1]["arguments"]["question_id"] == QUESTION
    provenance = {
        "origin": "saved_model_decisions_replayed_before_parent_consumes_withdrawal",
        "prefix_source": str(SOURCE), "decisions_source": str(DECISIONS),
        "saved_decisions": 7, "new_llm_decision": False,
        "scope": "Reconstruct a continuation checkpoint with current file retention and routing; no new discovery claim.",
    }
    plan.update(study="discoveryworld_draft_counterexample_seed", replay_provenance=provenance)
    write_json(OUTPUT / "protocol.json", plan)
    workspace = OUTPUT / prefix["task"]["id"] / "hegel"
    agent_workspace = workspace / "agent"
    shutil.copytree(OUTPUT / "prefix_files", agent_workspace)
    prior_costs = deepcopy(prefix["prior_costs"])
    prior_costs["model_calls"] = prior_costs.get("model_calls", 0) + len(decisions)
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "wall_time_seconds"):
        prior_costs[key] = prior_costs.get(key, 0) + sum(e.get("usage", {}).get(key, 0) for e in decisions)
    prior_costs["reported_tokens"] = prior_costs.get("reported_tokens", 0) + sum(
        e.get("usage", {}).get("total_tokens", 0) for e in decisions)
    prior_costs["usage_missing_calls"] = prior_costs.get("usage_missing_calls", 0) + sum(
        not isinstance(e.get("usage", {}).get("total_tokens"), int) for e in decisions)
    # The omitted eighth decision was a direct world query with no Python or
    # after-observation analysis. All saved Python cost belongs to these seven.
    last = saved["events"][saved["prefix_event_count"] + 7]
    assert last["action"] == "query" and "actions" in last["arguments"]
    assert not last["arguments"].get("after_observation")
    prior_costs["python_execution_seconds"] = prior_costs.get("python_execution_seconds", 0) + saved["costs"]["python_execution_seconds"]

    with DiscoveryWorldLab(prefix["task"], plan["runtime"], agent_workspace) as lab:
        replay = replay_prefix(lab, prefix)
        agent = DiscoveryWorldAgent(None, agent_workspace, arm="hegel", limits=plan["limits"])
        agent.lab = lab
        agent.observations = deepcopy(prefix["records"])
        for name in ("questions", "methods", "relation_research", "research_advice", "research_focus",
                     "program_executions", "child_answers_returned"):
            setattr(agent, name, deepcopy(prefix["research"][name]))
        agent._prerequisite_checked_through = prefix["research"]["prerequisite_checked_through"]
        agent.method_results = deepcopy(prefix["method_results"])
        agent.events = deepcopy(prefix["events"])
        agent.files.update({name: (agent_workspace / name).read_text()
                            for name in prefix["registered_files"]})
        (agent_workspace / "research_notes.md").write_text(prefix["notes"])
        agent.costs["environment_steps"] = lab.environment_steps
        original_path = decisions[-1]["arguments"]["path"]
        original_design = (agent_workspace / original_path).read_text()
        for number, event in enumerate(decisions, 1):
            agent._request()
            arguments = deepcopy(event["arguments"])
            feedback = agent._execute(event["action"], arguments, lab)
            if feedback.get("error"):
                raise ValueError(f"Saved decision {number} failed: {feedback['error']}")
            agent.events.append({"action": event["action"], "arguments": arguments,
                "feedback": deepcopy(feedback), "phase": "saved_decision_replay",
                "source_decision": number, "original_usage": deepcopy(event.get("usage", {}))})
            for actual in agent.observations[len(prefix["records"]):]:
                old = expected[actual["index"]]
                assert actual["action"] == old["action"]
                assert actual["feedback"].get("success") == old["feedback"].get("success")
                assert actual["observation"].get("lastActionMessage", "") == old["observation"].get("lastActionMessage", "")
        assert len(agent.observations) == 449
        answer = deepcopy(next(q for q in agent.questions if q["id"] == QUESTION))
        assert answer["status"] == "refuted" and answer["evidence"] == [428]
        assert answer["analysis_path"] != original_path
        assert (agent_workspace / original_path).read_text() == original_design
        assert (agent_workspace / answer["analysis_path"]).read_text() == decisions[-1]["arguments"]["code"]
        assert (agent_workspace / "research_notes.md").read_text() == decisions[-1]["arguments"]["notes"]
        request = agent._request()  # Prepare the decision without consuming it.
        focus = request.get("research_focus")
        parent = next(q for q in agent.questions if q["id"] == "task")
        reviewed = deepcopy(parent.get("reviewed_child_answers", {}).get(QUESTION))
        assert reviewed != {key: answer[key] for key in ("status", "answer", "evidence")}
        agent._save_state()
        agent.status = "saved_decision_prefix_replay_complete"
        replay.update(saved_decisions=7, replayed_environment_steps=11,
                      full_records_matched=449, full_environment_steps=lab.environment_steps)
        result = {**agent.result(), "task_id": prefix["task"]["id"], "arm": "hegel",
                  "prior_costs": prior_costs, "prefix_event_count": len(agent.events),
                  "prefix_environment_steps": lab.environment_steps, "new_environment_steps": 0,
                  "replayed_environment_steps": 11, "replay": replay, "replay_provenance": provenance}
        write_json(workspace / "active.json", result)
        write_json(workspace / "result.json", result)
        write_json(OUTPUT / "brief.json", {"status": agent.status, "provenance": provenance,
            "answer": answer, "design_preserved": True, "new_model_calls": 0,
            "new_observations": 0, "replayed_environment_steps": 11,
            "replayed_program_executions": agent.program_executions - prefix["research"]["program_executions"],
            "research_focus": focus, "parent_reviewed_answer": reviewed,
            "future_source": str(workspace), "replay": replay,
            "cost_scope": "Original seven provider calls and Python cost retained in prior_costs; host timing is a known subtotal. Current replay execution cost remains separate; no new provider tokens or scientific observations."})
    print(json.dumps({"status": agent.status, "records": 449, "new_model_calls": 0,
        "replayed_environment_steps": 11, "focus_stage": (focus or {}).get("stage"),
        "comparison_question": (focus or {}).get("comparison_question", {}).get("id"),
        "future_source": str(workspace)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

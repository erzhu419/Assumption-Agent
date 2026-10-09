"""Execute one previously authored draft through the repaired native dispatcher.

The model decision is unchanged historical source. Any selected public-world
actions are actual new execution, not a new autonomous model choice.
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

SOURCE = ROOT / "artifacts/discoveryworld_draft_counterexample_seed_20261008/dw_04_normal_s0/hegel"
DECISIONS = ROOT / "artifacts/discoveryworld_draft_counterexample_choice_20261008/dw_04_normal_s0/hegel"
OUTPUT = ROOT / "artifacts/discoveryworld_draft_revision_execution_20261008"


def main():
    OUTPUT.mkdir(exist_ok=False)
    plan, prefix = freeze_prefix(SOURCE, OUTPUT)
    saved = json.loads((DECISIONS / "result.json").read_text())
    decision = deepcopy(saved["events"][saved["prefix_event_count"]])
    assert decision["action"] == "write"
    assert decision["arguments"]["path"] == "shared/test_crystal3_1296.py"
    provenance = {"origin": "saved_model_draft_through_repaired_dispatcher",
        "prefix_source": str(SOURCE), "decision_source": str(DECISIONS),
        "source_decision": 1, "new_llm_decision": False,
        "scope": "Exact existing author source and question; actual execution under the repaired dispatcher, not a new autonomous adoption."}
    plan.update(study="discoveryworld_draft_revision_execution", replay_provenance=provenance)
    write_json(OUTPUT / "protocol.json", plan)
    workspace = OUTPUT / prefix["task"]["id"] / "hegel"
    agent_workspace = workspace / "agent"
    shutil.copytree(OUTPUT / "prefix_files", agent_workspace)
    prior_costs = deepcopy(prefix["prior_costs"])
    prior_costs["model_calls"] = prior_costs.get("model_calls", 0) + 1
    usage = decision.get("usage", {})
    for field in ("input_tokens", "output_tokens", "cached_input_tokens", "wall_time_seconds"):
        prior_costs[field] = prior_costs.get(field, 0) + usage.get(field, 0)
    prior_costs["reported_tokens"] = prior_costs.get("reported_tokens", 0) + usage.get("total_tokens", 0)
    prior_costs["usage_missing_calls"] = prior_costs.get("usage_missing_calls", 0) + int(
        not isinstance(usage.get("total_tokens"), int))

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
        request = agent._request()
        start = len(agent.observations)
        arguments = deepcopy(decision["arguments"])
        try:
            feedback = agent._execute(decision["action"], arguments, lab)
        except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
            feedback = {"error": f"{type(error).__name__}: {error}"}
        agent.events.append({"action": decision["action"], "arguments": arguments,
            "feedback": deepcopy(feedback), "phase": "saved_decision_new_execution",
            "source_decision": 1, "original_usage": deepcopy(usage)})
        assert agent.costs["model_calls"] == 0
        assert (agent_workspace / arguments["path"]).read_text() == arguments["code"]
        assert (agent_workspace / "research_notes.md").read_text() == arguments["notes"]
        agent._save_state()
        agent.status = "saved_decision_execution_failed" if feedback.get("error") else "saved_decision_execution_complete"
        new_records = agent.observations[start:]
        question_id = arguments["question"]["id"]
        answer = deepcopy(next(q for q in agent.questions if q["id"] == question_id))
        result = {**agent.result(), "task_id": prefix["task"]["id"], "arm": "hegel",
            "prior_costs": prior_costs, "prefix_event_count": len(prefix["events"]),
            "prefix_environment_steps": prefix["environment_steps"],
            "prefix_research_counts": {name: prefix["research"][name]
                for name in ("program_executions", "child_answers_returned")},
            "new_environment_steps": len(new_records), "replay": replay,
            "replay_provenance": provenance}
        write_json(workspace / "active.json", result)
        write_json(workspace / "result.json", result)
        brief = {"status": agent.status, "provenance": provenance, "prefix_records": start,
            "new_model_calls": 0, "new_environment_steps": len(new_records),
            "new_program_executions": agent.program_executions - prefix["research"]["program_executions"],
            "focus_before": request.get("research_focus"),
            "operator": feedback.get("operator"), "diagnosis": feedback.get("diagnosis"),
            "selected_experiment": feedback.get("selected_experiment"),
            "hypotheses": feedback.get("hypotheses"), "answer": answer,
            "procedure_status": feedback.get("procedure", {}).get("status"),
            "error": feedback.get("error"), "analysis_error": feedback.get("analysis_error"),
            "actual_observations": [agent._action_observation(r) for r in new_records],
            "score_normalized": result["score"]["scoreNormalized"],
            "future_source": str(workspace), "replay": replay}
        write_json(OUTPUT / "brief.json", brief)
    print(json.dumps({key: brief[key] for key in ("status", "new_model_calls",
        "new_environment_steps", "new_program_executions", "operator", "diagnosis",
        "procedure_status", "error", "analysis_error", "score_normalized")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

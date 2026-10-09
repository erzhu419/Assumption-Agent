#!/usr/bin/env python3
"""Withdraw one false answer using a retained analysis and existing paid evidence."""

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
OUTPUT = ROOT / "artifacts/discoveryworld_answer_withdrawal_20261008"
QUESTION = "test_crystal3_1296"
ORIGINAL_ANALYSIS = "shared/analysis_test_crystal3_1297.py"
REPAIRED_ANALYSIS = "shared/analysis_test_crystal3_1296_repair.py"


def main():
    source_active = json.loads((SOURCE / "active.json").read_text())
    if source_active["status"] != "model_call_budget_exhausted":
        raise ValueError("The source continuation must be completed before offline withdrawal")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plan, prefix = freeze_prefix(SOURCE, OUTPUT)
    old_answer = next(q for q in prefix["research"]["questions"] if q["id"] == QUESTION)
    if old_answer["status"] != "supported" or old_answer["evidence"] != [428]:
        raise ValueError("Expected the recorded false supported answer at observation 428")
    original = (OUTPUT / "prefix_files" / ORIGINAL_ANALYSIS).read_text()
    repaired = original.replace("429", "428").replace("1297", "1296")
    (OUTPUT / "retained_analysis_1297.py").write_text(original, encoding="utf-8")
    (OUTPUT / "adapted_analysis_1296.py").write_text(repaired, encoding="utf-8")
    provenance = {
        "origin": "human_selected_adaptation_of_existing_model_authored_analysis",
        "source": str(SOURCE), "original_analysis": ORIGINAL_ANALYSIS,
        "changes": ["observation index 429 -> 428", "tested frequency 1297 -> 1296"],
        "scope": "Offline continuation-state correction; no autonomous finding or benchmark improvement.",
        "original_result_preserved": True,
    }
    plan.update(study="discoveryworld_offline_answer_withdrawal", correction=provenance,
                purpose="Withdraw an erroneous supported answer on retained paid observations.")
    write_json(OUTPUT / "protocol.json", plan)
    workspace = OUTPUT / prefix["task"]["id"] / "hegel"
    agent_workspace = workspace / "agent"
    shutil.copytree(OUTPUT / "prefix_files", agent_workspace)
    with DiscoveryWorldLab(prefix["task"], plan["runtime"], agent_workspace) as lab:
        replay = replay_prefix(lab, prefix)
        write_json(workspace / "replay.json", replay)
        agent = DiscoveryWorldAgent(None, agent_workspace, arm="hegel", limits=plan["limits"])
        agent.lab = lab
        agent.observations = deepcopy(prefix["records"])
        agent.questions = deepcopy(prefix["research"]["questions"])
        agent.methods = deepcopy(prefix["research"].get("methods", []))
        agent.relation_research = deepcopy(prefix["research"].get("relation_research", {}))
        agent.research_advice = deepcopy(prefix["research"].get("research_advice"))
        agent.research_focus = deepcopy(prefix["research"].get("research_focus"))
        agent._prerequisite_checked_through = prefix["research"].get("prerequisite_checked_through", -1)
        for name in ("program_executions", "child_answers_returned"):
            setattr(agent, name, prefix["research"].get(name, 0))
        agent.method_results = deepcopy(prefix["method_results"])
        agent.files.update({name: (agent_workspace / name).read_text(encoding="utf-8")
                            for name in prefix["registered_files"]})
        agent.events = deepcopy(prefix["events"])
        agent.costs["environment_steps"] = lab.environment_steps
        (agent_workspace / "research_notes.md").write_text(prefix["notes"], encoding="utf-8")
        arguments = {"path": REPAIRED_ANALYSIS, "question_id": QUESTION, "code": repaired}
        agent._request()
        feedback = agent._execute("evaluate", arguments, lab)
        agent.events.append({"action": "evaluate", "arguments": arguments, "feedback": feedback,
                             "phase": "offline_manual_correction", "provenance": provenance})
        corrected = next(q for q in agent.questions if q["id"] == QUESTION)
        parent = next(q for q in agent.questions if q["id"] == corrected["parent_id"])
        if corrected["status"] != "refuted" or corrected["evidence"] != [428]:
            raise ValueError("The retained analysis did not withdraw the false answer")
        if parent["status"] != "reopened" or parent["reopened_by"] != QUESTION:
            raise ValueError("The corrected answer did not reopen its parent")
        if (len(agent.observations) != len(prefix["records"])
                or lab.environment_steps != prefix["environment_steps"]):
            raise ValueError("Offline answer correction unexpectedly acquired observations")
        agent._save_state()
        agent.status = "offline_correction_complete"
        result = agent.result()
        result.update(task_id=prefix["task"]["id"], arm="hegel", prior_costs=prefix["prior_costs"],
                      replay=replay, correction=provenance, prefix_event_count=len(prefix["events"]),
                      new_environment_steps=0)
        write_json(workspace / "active.json", result)
        write_json(workspace / "result.json", result)
        brief = {"status": result["status"], "provenance": provenance,
                 "before": old_answer, "after": corrected, "parent_status": parent["status"],
                 "parent_reopened_by": parent["reopened_by"],
                 "parent_pending_dependencies": agent._pending_dependencies(parent),
                 "new_model_calls": result["costs"]["model_calls"],
                 "new_reported_tokens": result["costs"]["reported_tokens"],
                 "new_observations": 0, "replayed_observations": len(prefix["records"]),
                 "new_program_executions": agent.program_executions
                                           - prefix["research"].get("program_executions", 0),
                 "python_seconds": result["costs"]["python_execution_seconds"],
                 "score": result["score"], "future_source": str(workspace)}
        write_json(OUTPUT / "brief.json", brief)
    print(json.dumps({key: brief[key] for key in ("status", "new_model_calls", "new_reported_tokens",
        "new_observations", "replayed_observations", "new_program_executions",
        "parent_status", "parent_pending_dependencies", "future_source")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

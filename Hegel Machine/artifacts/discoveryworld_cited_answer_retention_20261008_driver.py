"""Replay the saved model analysis while retaining its original design."""
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

SOURCE = ROOT / "artifacts/discoveryworld_cited_answer_adoption_20261008/dw_04_normal_s0/hegel"
OUTPUT = ROOT / "artifacts/discoveryworld_cited_answer_retention_20261008"
QUESTION = "test_crystal3_1296"


def main():
    OUTPUT.mkdir(exist_ok=False)
    plan, prefix = freeze_prefix(SOURCE, OUTPUT)
    decision = next(e for e in reversed(prefix["events"]) if e["action"] == "evaluate"
                    and e["arguments"].get("question_id") == QUESTION)
    arguments = {k: decision["arguments"][k] for k in ("path", "question_id", "code")}
    original = next(e["arguments"]["code"] for e in prefix["events"] if e["action"] == "write"
                    and e["arguments"].get("path") == arguments["path"])
    provenance = {"origin": "saved_model_analysis_replay_with_archived_design_restored",
                  "source": str(SOURCE), "new_llm_decision": False,
                  "scope": "File-preservation repair only; autonomous withdrawal is in the original run."}
    plan.update(study="discoveryworld_analysis_design_retention", correction=provenance)
    write_json(OUTPUT / "protocol.json", plan)
    workspace = OUTPUT / prefix["task"]["id"] / "hegel"
    agent_workspace = workspace / "agent"
    shutil.copytree(OUTPUT / "prefix_files", agent_workspace)
    (agent_workspace / arguments["path"]).write_text(original)
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
        before = deepcopy(next(q for q in agent.questions if q["id"] == QUESTION))
        feedback = agent._execute("evaluate", arguments, lab)
        agent.events.append({"action": "evaluate", "arguments": arguments, "feedback": feedback,
                             "phase": "saved_decision_replay", "provenance": provenance})
        after = feedback["question"]
        assert all(after[k] == before[k] for k in ("status", "answer", "evidence"))
        assert after["analysis_path"] != arguments["path"]
        assert (agent_workspace / arguments["path"]).read_text() == original
        assert (agent_workspace / after["analysis_path"]).read_text() == arguments["code"]
        assert len(agent.observations) == len(prefix["records"])
        agent._save_state()
        agent.status = "saved_analysis_replay_complete"
        result = {**agent.result(), "task_id": prefix["task"]["id"], "arm": "hegel",
                  "prior_costs": prefix["prior_costs"], "prefix_event_count": len(prefix["events"]),
                  "new_environment_steps": 0, "replay": replay, "correction": provenance}
        write_json(workspace / "active.json", result)
        write_json(workspace / "result.json", result)
        write_json(OUTPUT / "brief.json", {"status": agent.status, "provenance": provenance,
            "answer": after, "design_preserved": True, "new_model_calls": 0,
            "new_observations": 0, "new_program_executions": 1,
            "future_source": str(workspace), "replay": replay})
    print("Saved analysis replay passed; original design retained.")


if __name__ == "__main__":
    main()

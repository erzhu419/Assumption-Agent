"""Explicitly reopen the unwritten premise draft, then return to the parent."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]
from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.world_reference_response import response_premise_design_action
from scripts.run_response_ambiguity_continuation import restore_saved
from scripts.run_response_family_recovery import account
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json

def run(source, output, *, python_seconds=None, additional_calls=0, additional_wall_seconds=0):
    agent, lab, context = restore_saved(source, output, python_seconds=python_seconds,
                                           additional_calls=additional_calls, additional_wall_seconds=additional_wall_seconds)
    protocol = json.loads((output / "protocol.json").read_text())
    protocol.update(study="computed_feedback_to_existing_premise_author",
        auth_mode=verify_codex_subscription(),
        proposal_origin="operator_reopens_previously_unwritten_proposal",
        scope="One existing development condition. The operator reopens the previous proposal; "
              "the model chooses every hypothesis, prediction and experiment. No scientific source "
              "is edited by the runner. Existing costs and all 451 observations remain charged. "
              "At most eight proposal calls followed by the remaining parent account; not autonomous adoption "
              "or a matched performance comparison.")
    write_json(output / "protocol.json", protocol)
    agent.provider = CodexCodeProvider(agent.workspace, model="gpt-6-luna",
                                       reasoning_effort="medium", timeout_seconds=180)

    def checkpoint(data):
        latest = data["events"][-1] if len(data["events"]) > context["prefix_events"] else {}
        brief = {"status": "running", "costs": data["costs"], "score": data["score"],
            "new_model_calls": data["costs"]["model_calls"] - context["costs_before"]["model_calls"],
            "new_observations": len(agent.observations) - context["prefix_observations"],
            "latest_action": latest.get("action"),
            "latest_operator": latest.get("arguments", {}).get("operator"),
            "latest_feedback": agent._compact_feedback(latest.get("feedback", {}))}
        write_json(output / "active.json", brief)
        print(json.dumps({key: brief[key] for key in
                         ("status", "new_model_calls", "new_observations", "latest_action")}), flush=True)

    agent._on_checkpoint = checkpoint
    checkpoint(account(agent))
    comparison = agent.llm_inquiries["proposal:shared/response_representation_proposal.py"]["proposal"]["comparison_result"]
    action = response_premise_design_action(comparison)
    proposal = agent._execute(action["action"], {**action["arguments"], "max_model_calls": 8}, lab)
    write_json(output / "proposal.json", proposal)
    # Record the actual host call so the parent sees the same return a caller would.
    agent.events.append({"action": action["action"], "arguments": action["arguments"],
        "trigger_origin": "operator_reopens_previously_unwritten_proposal",
        "feedback": deepcopy(proposal)})
    agent._save_state()
    task = agent.run(lab, on_checkpoint=checkpoint)
    result = account(agent)
    result.update(study_status="finished", episode_status=result["status"],
        execution={"operator": "task_continuation", "status": task["status"], "proposal": proposal},
        costs_before=context["costs_before"],
        new_costs={key: value - context["costs_before"][key] for key, value in result["costs"].items()},
        observations=deepcopy(agent.observations), replay=context["replay"])
    write_json(output / "result.json", result)
    write_json(output / "active.json", {"status": "finished", "episode_status": result["status"],
        "score": result["score"], "new_costs": result["new_costs"],
        "deliverable_status": proposal["proposal"]["deliverable_status"]})
    print(json.dumps({"status": result["status"], "score": result["score"],
                      "new_costs": result["new_costs"]}), flush=True)


if __name__ == "__main__":
    run(ROOT / "artifacts/response_premise_dispatch_20261009/live/result.json",
        Path(__file__).parent / "live", python_seconds=900,
        additional_calls=12, additional_wall_seconds=1800)

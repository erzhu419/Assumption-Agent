"""One saved-state diagnostic, with no production or physical-policy change."""

from copy import deepcopy
import json
from pathlib import Path
from time import monotonic

from hegel_machine.cognition import world_response_hypotheses as response


ROOT = Path(__file__).parent
BASE = ROOT / "baseline"
state = json.loads((BASE / "current_broad_input.json").read_text())
protocol = json.loads((BASE / "protocol.json").read_text())
baseline = json.loads((BASE / "current_broad_comparison.json").read_text())
estimators = protocol["routes"]["current_broad"]["estimators"]
supports = state["response_research"]["conditional_simplification"]["assumption"]["supports"]
old = response._append_joint_responses
added = {}


def targeted_joint(members, current, declarations, path):
    before_cost = members["compute_cost"]
    before_hypotheses = len(members["hypotheses"])
    support_ids = {item["id"] for item in supports}
    existing_ids = {item["id"] for item in declarations}
    combined = [*declarations, *(item for item in supports if item["id"] not in existing_ids)]
    # One invocation retains the original joint member IDs and ordering. A
    # second append would reuse joint_response_0 and contaminate comparison.
    revised = old(members, current, combined, path)
    added.update(all_joint_compute_cost=revised["compute_cost"] - before_cost,
                 all_new_joint_hypotheses=len(revised["hypotheses"]) - before_hypotheses,
                 targeted_hypotheses=sum(h["family"] in support_ids for h in revised["hypotheses"]
                                         if h["id"].startswith("joint_response_")),
                 search=deepcopy([entry for entry in revised.get("joint_parameter_search", [])
                                 if entry["family"] in support_ids]))
    return revised


response._append_joint_responses = targeted_joint
started = monotonic()
try:
    result = response.propose_response_research(deepcopy(state), estimators, "shared/consolidated_method.py")
finally:
    response._append_joint_responses = old
wall = monotonic() - started
child = result.get("conditional_simplification", {})
support_ids = {item["id"] for item in supports}
joint_members = [h for h in result["hypotheses"]
                 if h["family"] in support_ids and h["id"].startswith("joint_response_")]
summary = {
    "scope": "Same saved broad input and scientific stagnation. Jointly refit only the already selected conditional supports, then retain the original family selector and stop rule. No LLM or physical actions.",
    "response_count": result["response_count"],
    "targeted_supports": supports,
    "added_joint_search": added,
    "status": result["status"],
    "selected_number": result.get("selected_number"),
    "selected_setting": result.get("selected_setting"),
    "selected_role": result.get("family_research", {}).get("selected_role"),
    "no_progress_cycles": result.get("family_research", {}).get("no_progress_cycles"),
    "conditional_status": child.get("status"),
    "conditional_adopted": child.get("adopted"),
    "conditional_no_progress_cycles": child.get("family_research", {}).get("no_progress_cycles"),
    "compute_cost": result["compute_cost"],
    "total_compute_delta_from_baseline": result["compute_cost"] - baseline["compute_cost"],
    "targeted_joint_members": [{"id": h["id"], "prediction": h["target_prediction"],
                                "lower": h["lower"], "upper": h["upper"]} for h in joint_members],
    "target_experiment_settings": [e["prediction"] for e in result["experiments"]
                                   if e["number"] == result["target_number"]],
    "wall_seconds": wall,
    "new_observations": 0,
    "model_calls": 0,
}
(ROOT / "joint_counterfactual_result.json").write_text(json.dumps(result, indent=2))
(ROOT / "joint_counterfactual_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

"""Read completed fixed-anchor outcomes; no learning or terminal rescoring."""
import json
from collections import Counter
from math import isclose
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
output = BASE / "old_capability_anchor"
summary = json.loads((output / "summary.json").read_text())
original = json.loads((ROOT / "artifacts/research_reuse_cost_20260930/fresh_replication/summary.json").read_text())

def paired_result(a, b):
    # Display-only numerical equivalence; neither learning nor scoring changes.
    return "tie" if isclose(a, b, rel_tol=1e-9, abs_tol=0.0) else "win" if b<a else "loss"

by_arm = {}
for row in summary["rows"]:
    by_arm.setdefault(row["arm"], {})[(row["root"], row["position"])] = row
pairs = []
for key, candidate in by_arm["new_cumulative"].items():
    baseline = by_arm["old_cumulative"][key]
    a, b = baseline["mean_response_mse"], candidate["mean_response_mse"]
    pairs.append({"root": key[0], "position": key[1], "stage": candidate["stage"],
                  "old_mean_response_mse": a, "new_mean_response_mse": b,
                  "change": b-a, "raw_strict_result": "win" if b<a else "loss" if b>a else "tie",
                  "result": paired_result(a, b),
                  "old_status": baseline["status"], "new_status": candidate["status"]})
chains = []
for root in sorted({row["root"] for row in pairs}):
    selected = [row for row in pairs if row["root"] == root]
    a = sum(row["old_mean_response_mse"] for row in selected)/len(selected)
    b = sum(row["new_mean_response_mse"] for row in selected)/len(selected)
    chains.append({"root": root, "old_mean_response_mse": a, "new_mean_response_mse": b,
                   "raw_strict_result": "win" if b<a else "loss" if b>a else "tie",
                   "result": paired_result(a, b)})
old, new = (summary["totals"][arm] for arm in ("old_cumulative", "new_cumulative"))
for arm in summary["totals"].values():
    arm["all_cpu_seconds"] = sum(arm[k] for k in ("source_cpu_seconds", "learning_cpu_seconds", "terminal_cpu_seconds"))
historic = {(row["root"], row["position"]):row for row in original["rows"] if row["arm"] == "new_cumulative"}
old_reproduction = {
    "previous_source": "artifacts/research_reuse_cost_20260930/fresh_replication/summary.json:new_cumulative",
    "max_mean_response_mse_difference": max(abs(historic[key]["mean_response_mse"]-row["mean_response_mse"]) for key,row in by_arm["old_cumulative"].items()),
    "learning_costs_match": old["learning_costs"] == original["totals"]["new_cumulative"]["learning_costs"],
    "completed_match": old["completed"] == original["totals"]["new_cumulative"]["completed"],
}
report = {
    "scope": "Original seen0.131 cohort capability retention, not fresh generalization or authoring-model causal comparison",
    "protocol": str(output.relative_to(ROOT)/"protocol.json"),
    "arms": summary["totals"],
    "paired_tasks": {name:sum(row["result"]==name for row in pairs) for name in ("win","tie","loss")},
    "paired_chains": {name:sum(row["result"]==name for row in chains) for name in ("win","tie","loss")},
    "raw_strict_paired_tasks": {name:sum(row["raw_strict_result"]==name for row in pairs) for name in ("win","tie","loss")},
    "raw_strict_paired_chains": {name:sum(row["raw_strict_result"]==name for row in chains) for name in ("win","tie","loss")},
    "numeric_equivalence": {"method": "math.isclose", "rel_tol": 1e-9, "abs_tol": 0.0,
        "scope": "Display-only roundoff equivalence. Original runner uses raw strict counts without a tie tolerance. No learning, scoring, success threshold, or average is modified."},
    "relative_changes": {k: new[k]/old[k]-1 for k in ("mean_response_mse","mean_mse","learning_cpu_seconds","all_cpu_seconds")},
    "all_cost_relative_changes": {k:new["all_costs"][k]/old["all_costs"][k]-1 for k in ("observation_cost","compute_cost")},
    "old_runtime_reproduction": old_reproduction,
    "worst_task_increase": max(pairs,key=lambda r:r["change"]),
    "largest_task_improvement": min(pairs,key=lambda r:r["change"]),
    "chains": chains, "tasks": pairs,
}
counts = {}
for arm in summary["totals"]:
    count = Counter()
    for row in summary["rows"]:
        if row["arm"] == arm:
            count.update(row["actions"])
    counts[arm] = dict(count)
report["actual_final_report_action_counts"] = counts
report["stage_means"] = []
for stage in dict.fromkeys(row["stage"] for row in pairs):
    selected = [row for row in pairs if row["stage"] == stage]
    a = sum(row["old_mean_response_mse"] for row in selected)/len(selected)
    b = sum(row["new_mean_response_mse"] for row in selected)/len(selected)
    report["stage_means"].append({"stage": stage, "old_mean_response_mse": a,
        "new_mean_response_mse": b, "relative_change": b/a-1})
(output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:report[key] for key in ("paired_tasks","paired_chains","relative_changes","all_cost_relative_changes","old_runtime_reproduction")},indent=2))

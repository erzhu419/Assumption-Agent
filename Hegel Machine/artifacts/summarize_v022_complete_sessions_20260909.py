"""Summarize frozen three-seed paired studies and the preselected n11 task."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATE = "20260909"
SEEDS = (5501, 4507, 6781)
ARMS = ("methods", "ordinary")


def outcome(gain, threshold):
    return "improved" if gain > threshold else "worsened" if gain < -threshold else "tied"


def summarize_loss(pairs, arm, *, decision=False):
    old_losses = [(old["decision"] if decision else old)[f"{arm}_loss"] for old, _ in pairs]
    new_losses = [(new["decision"] if decision else new)[f"{arm}_loss"] for _, new in pairs]
    gains = [old - new for old, new in zip(old_losses, new_losses)]
    return {"tasks": len(pairs), "old_mean_loss": mean(old_losses), "new_mean_loss": mean(new_losses),
            "old_minus_new": mean(gains),
            "numerical_outcomes": dict(Counter(outcome(gain, 1e-9) for gain in gains)),
            "material_outcomes": dict(Counter(outcome(gain, .002) for gain in gains))}


def compact_action(action):
    details = action["details"]
    keep = ("candidate_id", "adopted", "selected", "selected_candidate", "policies_compared",
            "reason", "status", "gain", "before_loss", "after_loss", "rows", "trajectories",
            "reference_rows", "standard_error", "support_margin", "program_changed",
            "execution_complete", "evidence", "probe_ids", "probe_id", "sequence", "role",
            "observation_cost", "compute_cost", "comparison_scope", "trajectory_comparisons",
            "targeted_probe_ids", "reference_probe_ids", "targeted_rows", "representative_rows",
            "probe_lengths", "needs_evidence", "payable", "evidence_status", "last_compute_cost",
            "policy", "condition", "target_control", "reference_observation_cost", "candidate_disagreement",
            "action", "name", "method_id", "context", "last_before_loss", "last_after_loss", "last_gain",
            "mean_gain", "mean_observation_cost", "mean_compute_cost", "last_normalizer", "deferred")
    return {"round": action["round"], "action": action["action"], "reason": action["reason"],
            "observation_cost": action["observation_cost"], "compute_cost": action["compute_cost"],
            "details": {key: details[key] for key in keep if key in details}}


def workspace_frontier(workspace):
    frontier = workspace.get("decision_frontier", {})
    return {"question": frontier.get("question"), "observation_cost": frontier.get("observation_cost"),
            "comparison_compute_cost": frontier.get("comparison_compute_cost"),
            "policies": len(frontier.get("policies", [])),
            "evidence": [{"status": item.get("status"), "comparison": item.get("comparison", {})}
                         for item in frontier.get("evidence_states", [])]}


def checkpoint_behavior(path):
    checkpoint = json.loads(Path(path).read_text())
    actions = [action for episode in checkpoint["episodes"] for action in episode["actions"]]
    counts = Counter(action["action"] for action in actions)
    decision_visits = []
    n11_visits = []
    for episode in checkpoint["episodes"]:
        decision_actions = [compact_action(action) for action in episode["actions"]
                            if "decision" in action["action"] or action["action"] in {
                                "compare_calibration", "continuation_feedback"}]
        entry = {"task_index": episode["task_index"], "task_id": episode["task_id"],
                 "visit": episode["visit"], "stop_reason": episode["stop_reason"],
                 "scheduler": episode["scheduler"], "observation_cost": episode["observation_cost"],
                 "compute_cost": episode["compute_cost"], "expression": episode["expression"],
                 "decision_actions": decision_actions}
        if any(action["action"] == "execute_decision_question" for action in episode["actions"]):
            decision_visits.append(entry)
        if episode["task_id"] == "n11":
            entry["observed_probes"] = [compact_action(action) for action in episode["actions"] if action["action"] == "observe"]
            entry["research_actions"] = [compact_action(action) for action in episode["actions"]
                if (action["action"] == "plan_experiment" and action["details"].get("policy") == "decision_history_contrast")
                or action["action"] in {"action_feedback", "credit_action", "defer_method_step", "execute_method_step"}]
            n11_visits.append(entry)
    pending = [{"task_index": int(index), **workspace_frontier(workspace)}
               for index, workspace in checkpoint["workspaces"].items() if workspace.get("decision_frontier")]
    n11 = {"visits": n11_visits,
           "episode_observation_cost": sum(item["observation_cost"] for item in n11_visits),
           "episode_compute_cost": sum(item["compute_cost"] for item in n11_visits)}
    controller_history = sum(action["action"] == "plan_experiment" and action["details"].get("policy") == "decision_history_contrast"
                             for action in actions)
    short_history = sum(action["action"] == "plan_decision_probe_package" and bool(action["details"].get("selected"))
                        for action in actions)
    return {"source": str(path), "action_counts": dict(counts),
            "history_experiments": {"controller": controller_history, "short": short_history,
                                    "total": controller_history + short_history},
            "learn_experiment_total": counts["learn_experiment"],
            "additional_pair_executions": {"controller": counts["extend_decision_comparison"],
                                            "short": counts["extend_decision_question"]},
            "short_decision_visits": decision_visits, "pending_decision_frontiers": pending,
            "measurement_cost": checkpoint.get("total_measurement_cost", 0),
            "experiment_cost": checkpoint.get("total_experiment_cost", 0), "n11": n11}


def compare_seed(seed, label):
    baseline_label = "v022_baseline" if seed == 6781 else "v021_final"
    paths = {"old": ROOT / f"artifacts/hegel_machine_{baseline_label}_seed{seed}_paired_{DATE}.json",
             "new": ROOT / f"artifacts/hegel_machine_{label}_seed{seed}_paired_{DATE}.json"}
    reports = {version: json.loads(path.read_text()) for version, path in paths.items()}
    old, new = reports["old"], reports["new"]
    for key in ("seed", "config", "partitions", "limits_per_arm"):
        if old[key] != new[key]:
            raise ValueError(f"Unmatched {key} in seed {seed}")
    if old["summary"]["task_count"] != 65 or new["summary"]["task_count"] != 65:
        raise ValueError("Both versions must contain all 65 preselected tasks")
    pairs = list(zip(old["summary"]["tasks"], new["summary"]["tasks"]))
    for left, right in pairs:
        if any(left[key] != right[key] for key in ("task_index", "task_id", "partition", "loss_kind")):
            raise ValueError("Task roster or loss kind changed across versions")
    arms = {}
    for arm in ARMS:
        prior, current = old["summary"]["arms"][arm], new["summary"]["arms"][arm]
        decisions = [(left, right) for left, right in pairs if "decision" in left and "decision" in right]
        arms[arm] = {"old": prior, "new": current,
            "new_minus_old_cost": {key: current[key] - prior[key]
                                   for key in ("observation_cost", "compute_cost", "completed_visits")},
            "by_prediction_loss": {kind: summarize_loss([(left, right) for left, right in pairs if left["loss_kind"] == kind], arm)
                                   for kind in sorted({left["loss_kind"] for left, _ in pairs})},
            "decision_loss": summarize_loss(decisions, arm, decision=True) if decisions else None}
    behaviors = {version: {arm: checkpoint_behavior(report["checkpoints"][arm]) for arm in ARMS}
                 for version, report in reports.items()}
    n11_pair = next((pair for pair in pairs if pair[0]["task_id"] == "n11"), None)
    n11 = None
    if n11_pair:
        n11 = {"task_index": n11_pair[0]["task_index"], "task_id": "n11",
               "old": n11_pair[0], "new": n11_pair[1],
               "old_minus_new": {arm: {"prediction_loss": n11_pair[0][f"{arm}_loss"] - n11_pair[1][f"{arm}_loss"],
                   "decision_loss": n11_pair[0]["decision"][f"{arm}_loss"] - n11_pair[1]["decision"][f"{arm}_loss"]}
                   for arm in ARMS},
               "actual_visits": {version: {arm: behaviors[version][arm].pop("n11") for arm in ARMS} for version in reports}}
    return {"seed": seed, "sources": {key: str(path) for key, path in paths.items()},
            "config": new["config"], "limits_per_arm": new["limits_per_arm"], "arms": arms,
            "within_version_paired": {version: {key: report["summary"][key] for key in ("by_prediction_loss", "decision_loss")}
                                      for version, report in reports.items()},
            "behavior": behaviors, "n11": n11}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v022_release")
    args = parser.parse_args()
    output = ROOT / f"artifacts/hegel_machine_{args.label}_cross_version_{DATE}.json"
    diagnostic_path = output.with_name(output.stem + "_decision_diagnostic.json")
    if output.exists() or diagnostic_path.exists():
        raise FileExistsError("Preserve existing comparison summaries")
    rows = [compare_seed(seed, args.label) for seed in SEEDS]
    report = {"experiment": "complete_cross_version_fixed_three_seed_comparison", "seeds": list(SEEDS),
        "interpretation": "Positive old_minus_new means reduced independent loss. Costs are full session costs, including initial research and revisits. Method-arm differences and whole-version changes are distinct comparisons. Pending questions and unchanged decisions are retained.",
        "results": rows,
        "aggregate_costs": {arm: {key: {version: sum(row["arms"][arm][version][key] for row in rows)
                            for version in ("old", "new")}
                            for key in ("observation_cost", "compute_cost", "completed_visits")} for arm in ARMS}}
    compact = {key: value for key, value in report.items() if key != "results"}
    compact["decision_and_n11_diagnostic"] = str(diagnostic_path)
    compact["results"] = [{key: value for key, value in row.items() if key not in {"behavior", "n11"}}
                          for row in rows]
    for full, small in zip(rows, compact["results"]):
        small["n11"] = {key: value for key, value in full["n11"].items() if key != "actual_visits"}
        small["n11"]["costs"] = {version: {arm: {key: value for key, value in full["n11"]["actual_visits"][version][arm].items()
                                              if key != "visits"} for arm in ARMS} for version in ("old", "new")}
        small["behavior"] = {version: {arm: {
            "history_experiments": full["behavior"][version][arm]["history_experiments"],
            "learn_experiment_total": full["behavior"][version][arm]["learn_experiment_total"],
            "additional_pair_executions": full["behavior"][version][arm]["additional_pair_executions"],
            "decision_action_counts": {key: value for key, value in full["behavior"][version][arm]["action_counts"].items()
                                       if "decision" in key},
            "short_decision_visits": len(full["behavior"][version][arm]["short_decision_visits"]),
            "pending_decision_frontiers": len(full["behavior"][version][arm]["pending_decision_frontiers"])}
            for arm in ARMS} for version in ("old", "new")}
    with output.open("x") as stream:
        json.dump(compact, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with diagnostic_path.open("x") as stream:
        json.dump({"summary": str(output), "results": [{"seed": row["seed"], "sources": row["sources"],
                  "behavior": row["behavior"], "n11": row["n11"]} for row in rows]}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": str(output), "diagnostic": str(diagnostic_path),
                      "aggregate_costs": report["aggregate_costs"]}), flush=True)


if __name__ == "__main__":
    main()

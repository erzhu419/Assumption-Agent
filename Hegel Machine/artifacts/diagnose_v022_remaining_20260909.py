"""Extract the already-paid n11 and continuation-cost evidence requested for 0.22."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts"
cache = {}


def read(seed, arm, label):
    key = seed, arm, label
    if key not in cache:
        path = BASE / f"hegel_machine_{label}_seed{seed}_{arm}_20260909.json"
        cache[key] = json.loads(path.read_text())
    return cache[key]


def selected_action(action):
    details = action["details"]
    result = {"round": action["round"], "action": action["action"], "reason": action["reason"],
              "observation_cost": action["observation_cost"], "compute_cost": action["compute_cost"],
              "details": {key: value for key, value in details.items() if not isinstance(value, (dict, list))}}
    condition = details.get("condition", {})
    if condition:
        result["condition"] = {"history_depth": condition.get("history_depth"),
            "preparation": condition.get("preparation"), "alternatives": [
                {"kind": item.get("representation", {}).get("kind"),
                 "history_depth": item.get("representation", {}).get("history_depth"),
                 "action_rule": item.get("decision_rule", {}).get("action_rule")}
                for item in condition.get("alternatives", [])]}
    for key in ("target_control", "reference_probe_ids", "probe_lengths"):
        if key in details:
            result[key] = details[key]
    return result


def visit(episode):
    relevant = [action for action in episode["actions"] if (
        action["action"] == "plan_experiment" and action["details"].get("policy") == "decision_history_contrast")
        or action["action"] in {"compare_decision_plan", "compare_calibration", "retain_decision_plan",
                                "plan_decision_followup", "continuation_feedback", "defer_method_step"}
        or (action["action"] == "action_feedback" and action["details"].get("action") == "task_value")]
    return {"visit": episode["visit"], "scheduler_reason": episode["scheduler"]["reason"],
            "stop_reason": episode["stop_reason"], "observation_cost": episode["observation_cost"],
            "compute_cost": episode["compute_cost"], "expression": episode["expression"],
            "uncertainty_model": {key: value for key, value in episode["uncertainty_model"].items()
                                  if not isinstance(value, (dict, list))},
            "actions": [selected_action(action) for action in relevant]}


def cost_episode(episode):
    return {"visit": episode["visit"], "reason": episode["scheduler"]["reason"],
            "observation_cost": episode["observation_cost"], "compute_cost": episode["compute_cost"],
            "history_packages": sum(action["action"] == "plan_experiment"
                and action["details"].get("policy") == "decision_history_contrast" for action in episode["actions"]),
            "short_extra_pairs": sum(action["action"] == "extend_decision_question" for action in episode["actions"])}


def main():
    result = {"sources": [], "n11_methods": {}, "controller_additional_pair_quotes": []}
    for seed in (5501, 4507):
        result["n11_methods"][seed] = {version: [visit(episode)
            for episode in read(seed, "methods", label)["episodes"] if episode["task_id"] == "n11"]
            for version, label in (("old", "v021_final"), ("new", "v022_release"))}
    versions = {"old": read(6781, "methods", "v022_baseline"),
                "new": read(6781, "methods", "v022_release")}
    decomposition = {"totals": {}, "changed_tasks": []}
    for version, checkpoint in versions.items():
        groups = {}
        for group in ("first", "repeat"):
            episodes = [episode for episode in checkpoint["episodes"] if (episode["visit"] == 1) == (group == "first")]
            groups[group] = {"visits": len(episodes), "observation_cost": sum(e["observation_cost"] for e in episodes),
                             "compute_cost": sum(e["compute_cost"] for e in episodes)}
        groups["global_compute_outside_episodes"] = checkpoint["total_compute_cost"] - sum(e["compute_cost"] for e in checkpoint["episodes"])
        groups["global"] = {"observation_cost": checkpoint["total_observation_cost"], "compute_cost": checkpoint["total_compute_cost"]}
        decomposition["totals"][version] = groups
    for index in range(65):
        episodes = {version: [episode for episode in checkpoint["episodes"] if episode["task_index"] == index]
                    for version, checkpoint in versions.items()}
        delta = {key: sum(e[key] for e in episodes["new"]) - sum(e[key] for e in episodes["old"])
                 for key in ("observation_cost", "compute_cost")}
        if any(delta.values()):
            decomposition["changed_tasks"].append({"task_index": index, "task_id": episodes["old"][0]["task_id"],
                "new_minus_old": delta, **{version: [cost_episode(episode) for episode in values]
                                         for version, values in episodes.items()}})
    result["cost_6781_methods"] = decomposition
    for seed in (5501, 4507, 6781):
        for arm in ("methods", "ordinary"):
            checkpoint = read(seed, arm, "v022_release")
            for episode in checkpoint["episodes"]:
                observed = 0
                for action in episode["actions"]:
                    if action["action"] == "plan_decision_followup":
                        result["controller_additional_pair_quotes"].append({
                            "seed": seed, "arm": arm, "task_id": episode["task_id"], "visit": episode["visit"],
                            "already_observed_this_visit": observed,
                            "observation_remaining": checkpoint["config"]["observation_budget"] - observed,
                            "final_visit_observation_cost": episode["observation_cost"],
                            "final_visit_compute_cost": episode["compute_cost"], **selected_action(action)})
                    observed += action["observation_cost"]
    result["sources"] = [str(BASE / f"hegel_machine_{label}_seed{seed}_{arm}_20260909.json")
                         for seed, arm, label in cache]
    result["interpretation"] = [
        "These are already-paid traces, not new experiments or independent-label-driven retuning.",
        "For 6781, the net cost increase is primarily more or larger full continuation visits rather than extra reference pairs.",
        "4507 n11 changes its predictor and calibration during the first visit. Its second visit retains both and reports no complete-trajectory calibration comparison.",
        "5501 n11 replaces a feedback/history predictor with a static pooled-scale risk predictor during the new first visit; its second full visit leaves the deployed plan unchanged.",
        "An observation shortfall directly shown at a recorded Controller quote is sufficient to explain why that pair was not bought; unlogged intermediate compute is not reconstructed."]
    output = BASE / "v022_release_remaining_diagnosis_20260909.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"diagnosis": str(output), "bytes": output.stat().st_size,
                      "controller_quotes": result["controller_additional_pair_quotes"]}), flush=True)


if __name__ == "__main__":
    main()

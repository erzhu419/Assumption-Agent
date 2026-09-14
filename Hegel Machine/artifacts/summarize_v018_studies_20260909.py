"""Compare frozen transfer repair sessions and the two specified interventions."""

from collections import Counter
import json

from summarize_v016_studies_20260909 import ROOT, compact, compare, run


def mechanisms(session):
    repairs, deferrals = [], []
    for episode in session["episodes"]:
        for action in episode["actions"]:
            identity = {"task_index": episode["task_index"], "task_id": episode["task_id"],
                        "visit": episode["visit"], "round": action["round"]}
            if action["action"] == "refit_transfer":
                repairs.append({**identity, **action["details"], "compute_cost": action["compute_cost"]})
            if action["action"] == "revise_research_step":
                deferrals.append({**identity, **action["details"]})
    return {"source_refits": repairs, "compute_cost": sum(row["compute_cost"] for row in repairs),
            "residuals_resolved": sum(row["residual_resolved"] for row in repairs),
            "research_changes": deferrals,
            "deferred_actions": dict(Counter(item["action"] for row in deferrals for item in row["deferred"]))}


def main():
    summary = {"experiment": "v018_concrete_transfer_repair", "seeds": {}}
    for seed in (1237, 1613, 2027, 2539):
        baseline = "v018_baseline" if seed == 2539 else "v017_mainline"
        arms = ("methods", "ordinary") if seed in (1237, 2539) else ("methods",)
        summary["seeds"][str(seed)] = {}
        for arm in arms:
            before = run(seed, baseline, "20260909", arm)
            after = run(seed, "v018_mainline", "20260909", arm)
            summary["seeds"][str(seed)][arm] = {
                **compare(before, after), "mechanisms": mechanisms(after[0])}
    mainline = run(1237, "v018_mainline", "20260909", "methods")
    for intervention in ("transfer_refit", "method_knowledge_return"):
        without = run(1237, f"v018_without_{intervention}", "20260909", "methods")
        summary[f"{intervention}_ablation_seed1237"] = {
            **compare(without, mainline), "without_mechanisms": mechanisms(without[0]),
            "interpretation": "before is the specified ablation; after is the full 0.18 methods arm"}
    path = ROOT / "hegel_machine_v018_transfer_repair_summary_20260909.json"
    with path.open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(path)
    for seed, arms in summary["seeds"].items():
        for arm, comparison in arms.items():
            print(json.dumps({"seed": seed, "arm": arm,
                "before": {key: comparison["before"][key] for key in
                    ("mean_loss_by_objective", "observation_cost", "compute_cost")},
                "after": {key: comparison["after"][key] for key in
                    ("mean_loss_by_objective", "observation_cost", "compute_cost")},
                "material_outcomes": comparison["material_outcomes_all_65"]}))


if __name__ == "__main__":
    main()

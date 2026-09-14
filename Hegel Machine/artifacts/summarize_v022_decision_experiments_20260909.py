"""Write small numeric evidence from all five actual development experiment arms."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v022_release_decision_experiments")
    args = parser.parse_args()
    source = ROOT / f"artifacts/{args.label}_20260909.json"
    report = json.loads(source.read_text())
    result = {"source": str(source), "version": report["version"], "seed": report["seed"],
              "status": report["status"], "arms": report["arms"], "tasks": []}
    for task in report["tasks"]:
        item = {"task_id": task["task_id"], "task_index": task["task_index"],
                "initial_observation_cost": task["initial"]["observation_cost"],
                "initial_compute_cost": task["initial"]["compute_cost"],
                "saved_question": bool(task["saved_frontier"]),
                "evaluation_costs": task["evaluation_costs"], "arms": {}}
        for name, arm in task["arms"].items():
            probes = [action for action in arm["actions"] if action["action"] == "observe"]
            decisions = [action for action in arm["actions"] if action["action"] in {
                "compare_decision_question", "compare_decision_plan", "compare_calibration"}]
            keep = ("candidate_id", "adopted", "gain", "support_margin", "standard_error", "trajectories",
                    "reason", "evidence_status", "execution_complete", "last_compute_cost")
            item["arms"][name] = {
                **{key: arm[key] for key in ("stop_reason", "observation_cost", "measurement_cost", "compute_cost",
                    "total_observation_cost", "total_compute_cost", "program_changed", "model_changed",
                    "decision_improvement_from_stopping")},
                "prediction_loss": arm["evaluation"]["prediction_loss"],
                "decision_loss": arm["evaluation"]["objective_loss"],
                "targeted_packages_selected": sum(action["action"] == "plan_decision_probe_package"
                                                   and bool(action["details"].get("selected")) for action in arm["actions"]),
                "targeted_observations": sum(action["observation_cost"] for action in probes
                                              if action["details"].get("role") == "targeted"),
                "representative_observations": sum(action["observation_cost"] for action in probes
                                                    if action["details"].get("role") == "representative"),
                "extra_pair_executions": sum(action["action"] == "extend_decision_question" for action in arm["actions"]),
                "comparisons": [{**{key: action["details"][key] for key in keep if key in action["details"]},
                                 "trajectory_gains": [trial["gain"] for trial in action["details"].get("trajectory_comparisons", [])]}
                                for action in decisions],
                "independent_trajectory_improvements": arm["trajectory_decision_improvements"],
                "remaining_candidates": len(arm["research_state"].get("decision_frontier", {}).get("policies", [])) - 1
                                        if arm["research_state"].get("decision_frontier") else 0}
        result["tasks"].append(item)
    arms = report["arms"]
    result["mean_decision_loss"] = {name: sum(task["arms"][name]["decision_loss"] for task in result["tasks"])
                                           / len(result["tasks"]) for name in arms}
    result["additional_total_costs"] = {name: {key: sum(task["arms"][name][key] for task in result["tasks"])
                                              for key in ("observation_cost", "measurement_cost", "compute_cost")} for name in arms}
    result["interpretation"] = [
        "All public risk tasks from development seed1237 were included; independent evaluation ran after every arm stopped.",
        "When zero targeted packages are selected, fixed_representative and targeted_package are empirically the same intervention on that task.",
        "Additional trials can eliminate a challenger or preserve a pending question without improving deployment loss.",
        "Paid representative support and independent decision improvement are different reported outcomes."]
    output = source.with_name(source.stem + "_compact.json")
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"compact": str(output), "bytes": output.stat().st_size,
                      "mean_decision_loss": result["mean_decision_loss"]}))


if __name__ == "__main__":
    main()

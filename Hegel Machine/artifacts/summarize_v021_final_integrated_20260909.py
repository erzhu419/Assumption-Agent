"""Compact facts from the final frozen 0.21 three-process lifecycle run."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "hegel_machine_v021_final_integrated_20260909.json"
SUMMARY = ROOT / "hegel_machine_v021_final_integrated_20260909_summary.json"
PAUSED = ROOT / "hegel_machine_v021_final_integrated_paused_summary_20260909.json"
WAITING = ROOT / "hegel_machine_v021_final_integrated_waiting_20260909.json"
OUTPUT = ROOT / "hegel_machine_v021_final_integrated_runtime_evidence_20260909.json"


def main():
    report = json.loads(CHECKPOINT.read_text())
    summary = json.loads(SUMMARY.read_text())
    paused = json.loads(PAUSED.read_text())
    assert summary["status"] == "completed" and summary["completed_stages"] == 13
    assert not summary["pending_stages"]
    assert summary["sequence"] == paused["sequence"]
    assert summary["adaptive_measurements"] == paused["adaptive_measurements"]
    costs, structure, decisions = {}, {}, []
    for name, stage in report["stages"].items():
        if name in ("continued_tasks", "adaptive_measurements", "delayed_experiment"):
            costs[name] = {kind + "_cost": stage.get("total_" + kind + "_cost", 0)
                           for kind in ("observation", "compute", "measurement", "experiment")}
        elif name == "experiment_transfer":
            costs[name] = {key: stage["summary"][key] for key in ("observation_cost", "compute_cost")}
        elif name == "method_allocation":
            costs[name] = {kind + "_cost": sum(stage["summary"][prefix + "_" + kind + "_cost"]
                for prefix in ("source", "session")) for kind in ("observation", "compute")}
        elif name == "local_mechanisms":
            local = stage["summary"]
            costs[name] = {"observation_cost": sum(local[key] for key in
                ("source_observations", "preparation_observations", "controller_observations")),
                "compute_cost": local["source_fitting_compute"] + local["controller_compute"],
                "separate_reporting_compute": local["report_comparison_compute"]}
        else:
            costs[name] = {key: stage["costs"][key]
                           for key in ("observation_cost", "compute_cost", "measurement_cost")}
        if "discovery" in stage:
            structure[name] = {"selected_scale": stage["selected_scale"],
                "selected_objects": stage["selected_objects"],
                "measurement_reconfiguration": stage["measurement_reconfiguration"],
                "evaluation": stage["evaluation"],
                "prior_assessments": [{key: row.get(key) for key in ("record_id", "status", "loss")}
                                      for row in stage["prior_assessments"]],
                "hypotheses": [{key: row.get(key) for key in ("kind", "status", "prospective_error", "radius")}
                    | {"counterexamples": len(row.get("counterexamples", []))}
                    for row in stage["discovery"]["hypotheses"]]}
    for episode in report["stages"]["continued_tasks"]["episodes"]:
        if any(action["action"] == "execute_decision_question" for action in episode["actions"]):
            decisions.append({"task_id": episode["task_id"],
                "observation_cost": episode["observation_cost"], "compute_cost": episode["compute_cost"],
                "execution": [a["details"] for a in episode["actions"] if a["action"] == "execute_decision_question"],
                "feedback": [a["details"] for a in episode["actions"] if a["action"] == "continuation_feedback"]})
    lifecycle = [{"process": 1, "status": paused["status"], "completed_stages": paused["completed_stages"]}]
    if WAITING.exists():
        waiting = json.loads(WAITING.read_text())
        lifecycle.append({"process": 2, "status": waiting["status"], "completed_stages": waiting["completed_stages"]})
    lifecycle.append({"process": len(lifecycle) + 1, "status": summary["status"], "completed_stages": 13})
    delayed = report["stages"]["delayed_experiment"]
    compact = {"experiment": report["experiment"], "implementation": "v021_final",
        "seed": report["seed"], "status": summary["status"], "completed_stages": 13,
        "pending_stages": [], "lifecycle": lifecycle,
        "first_two_stage_reports_unchanged_after_resume": True,
        "costs_by_stage": costs,
        "total_reported_research_costs": {key: sum(row.get(key, 0) for row in costs.values())
            for key in ("observation_cost", "measurement_cost", "compute_cost", "experiment_cost")},
        "cost_scope": "Paid stage preparation and controller costs; instrument and experiment fees remain separate. Independent evaluation and reporting are excluded.",
        "concepts": summary["concepts"], "sequence": summary["sequence"],
        "decision_research": summary["decision_research"], "actual_decision_episodes": decisions,
        "measurements": summary["adaptive_measurements"], "local_mechanisms": summary["local_mechanisms"],
        "experiment_transfer": summary["experiment_transfer"],
        "method_allocation": {key: value for key, value in summary["method_allocation"].items()
                              if not isinstance(value, (dict, list))},
        "structure": structure,
        "delayed_experiment": {**summary["delayed_experiment"], "pending_experiments": delayed["pending_experiments"],
            "delivery_events": [event for event in delayed["scheduler_events"] if event["reason"] == "experiment_delivered"],
            "task_validation": [{key: task.get(key) for key in ("task_id", "visits", "validation_loss", "status")}
                                for task in delayed["tasks"]]},
        "ordinary_code_comparison": report["ordinary_code_comparison"],
        "source_checkpoint": str(CHECKPOINT), "source_summary": str(SUMMARY)}
    with OUTPUT.open("x") as stream:
        json.dump(compact, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "status": compact["status"],
        "lifecycle": lifecycle, "total_reported_research_costs": compact["total_reported_research_costs"]}))


if __name__ == "__main__":
    main()

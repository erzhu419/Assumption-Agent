#!/usr/bin/env python3
"""Summarize purchased joint experiments and actual learned method transfer."""

import argparse
from collections import Counter
import json
from pathlib import Path


def compact_plan(plan):
    return {key: value for key, value in plan.items()
            if value is None or isinstance(value, (str, int, float, bool))}


def summarize(output):
    rows, method_states = [], []
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        seed, arm = path.stem.split("_", 1)
        method_states.append({"seed": int(seed), "arm": arm, "methods": [{
            "method_id": method["method_id"], "status": method["status"],
            "source_task_id": method["source_task_id"], "solver_strategy": method["solver_strategy"],
            "target_strategy": method["target_strategy"], "support_count": len(method["support"]),
            "supported_tasks": sorted({event["task_id"] for event in method["support"]}),
            "counterexamples": len(method["counterexamples"])}
            for method in report.get("knowledge", {}).get("program_experiment_methods", [])]})
        scores = {row["task_position"]: row for row in report["evaluation"]}
        for entry in report["results"]:
            position, result = entry["task_position"], entry["research"]
            actions = result["actions"]
            purposes = Counter(action["purpose"] for action in actions if action["action"] == "acquire")
            tests = [action for action in actions if action["action"] == "test_competing_object_predictions"]
            revisions = {action["probe_id"]: action for action in actions
                         if action["action"] == "revise_competing_object_programs"}
            experiments = []
            for action in tests:
                plan, evidence = action["plan"], action["evidence"]
                revision = revisions.get(action["probe_id"], {})
                experiments.append({"probe_id": action["probe_id"], "plan": compact_plan(plan),
                    "directed_boundary": plan.get("proposal", {}).get("boundary"),
                    "proposal": plan.get("proposal", {}),
                    "actual_status": evidence["status"],
                    "calibration_status": evidence["calibration_status"],
                    "observation_rows": evidence["observation_rows"],
                    "assess_compute_cost": evidence["compute_cost"],
                    "hypotheses": [{key: hypothesis.get(key) for key in
                        ("hypothesis_id", "status", "measured_loss", "forecast_loss", "rows")}
                        for hypothesis in evidence["hypotheses"]],
                    "refit_after_measurement": bool(revision),
                    "refit_compute_cost": revision.get("compute_cost", 0),
                    "revised_candidate_count": revision.get("after")})
            rows.append({"seed": int(seed), "arm": arm, "task_position": position,
                "terminal_loss": scores.get(position, {}).get("loss"),
                "kind": result["selected_kind"], "channels": result["selected_channels"],
                "learning_costs": result["costs"], "used_sources": result.get("used_sources", []),
                "method_actions": [action for action in actions if "experiment_method" in action["action"]],
                "experiments": experiments,
                "planning_actions": [compact_plan(action) for action in actions
                    if action["action"] == "plan_competing_object_experiment"],
                "boundary_experiments": purposes["resolve_structure_boundary"],
                "competition_counterexamples": sum(len(state.get("counterexamples", []))
                    for state in result.get("program_competitions", [])),
                "object_partition_still_unresolved": not (result.get("object_research") or {}).get("complete", False)})
    result = {"tasks": rows, "method_states": method_states, "arms": {}}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        experiments = [experiment for row in selected for experiment in row["experiments"]]
        plans = [plan for row in selected for plan in row["planning_actions"]]
        result["arms"][arm] = {"tasks": len(selected),
            "evaluated": sum(row["terminal_loss"] is not None for row in selected),
            "competition_experiments": len(experiments),
            "method_actions": Counter(action["action"] for row in selected for action in row["method_actions"]),
            "actual_cross_task_method_uses": sum(action["action"] == "use_program_experiment_method"
                and bool(action.get("cross_task")) for row in selected for action in row["method_actions"]),
            "cross_task_method_tasks": sum(any(action["action"] == "use_program_experiment_method"
                and action.get("cross_task") for action in row["method_actions"]) for row in selected),
            "purchased_proposal_sources": Counter(item["plan"].get("proposal_source", "generic")
                for item in experiments),
            "recorded_planning_attempts": len(plans),
            "planning_outcomes": Counter(item.get("forecast_status", "unknown") for item in plans),
            "directed_candidates": sum(item.get("directed_candidates", 0) for item in plans),
            "generic_candidates": sum(item.get("generic_candidates", 0) for item in plans),
            "all_planning_compute_cost": sum(item.get("compute_cost", 0) for item in plans),
            "actual_statuses": Counter(item["actual_status"] for item in experiments),
            "calibration_statuses": Counter(item["calibration_status"] for item in experiments),
            "refits_after_measurement": sum(item["refit_after_measurement"] for item in experiments),
            "planning_compute_cost": sum(item["plan"].get("compute_cost", 0) for item in experiments),
            "competition_observation_rows": sum(item["observation_rows"] for item in experiments),
            "refit_compute_cost": sum(item["refit_compute_cost"] for item in experiments),
            "boundary_experiments": sum(row["boundary_experiments"] for row in selected),
            "competition_counterexamples": sum(row["competition_counterexamples"] for row in selected),
            "tasks_calling_prior_knowledge": sum(bool(row["used_sources"]) for row in selected)}
    (output / "behavior_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["arms"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)

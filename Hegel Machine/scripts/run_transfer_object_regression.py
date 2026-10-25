#!/usr/bin/env python3
"""Compare 0.72 with completed 0.71 object streams; run only the new arm."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
PROTOCOL = ROOT / "artifacts/joint_experiment_methods_20260913/comparison_v1/protocol.json"
BASELINE = ROOT / "artifacts/compound_experiment_methods_20260913/object_regression_v1"
OUTPUT = ROOT / "artifacts/transfer_condition_research_20260913/object_regression_v1"
COSTS = ("observation_cost", "measurement_cost", "compute_cost")


def write(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=True, allow_nan=True)


def summarize(protocol):
    reports = {arm: [json.loads((folder / f"{seed}_{arm}.json").read_text())
                     for seed in protocol["seeds"]]
               for arm, folder in (("v071", BASELINE), ("v072", OUTPUT))}
    threshold = protocol["deterministic_solved_threshold"]
    summary = {"protocol": str(PROTOCOL), "baseline": str(BASELINE),
               "seeds": protocol["seeds"], "arms": {}, "paired_tasks": []}
    for arm, runs in reports.items():
        results = [item["research"] for run in runs for item in run["results"]]
        scores = [score for run in runs for score in run["evaluation"]]
        actions = [item for result in results for item in result["actions"]]
        condition_acquisitions = [item for item in actions if item["action"] == "acquire"
                                  and item.get("purpose") == "distinguish_transfer_conditions"]
        condition_plans = [item for item in actions if item["action"] == "plan_transfer_condition_experiment"]
        condition_tests = [item for item in actions if item["action"] == "test_transfer_condition_predictions"]
        condition_refits = [item for item in actions if item["action"] == "refit_transfer_after_experiment"]
        short_trials = [item for item in actions if item["action"] == "test_learned_acquisition_method"]
        summary["arms"][arm] = {"tasks": len(results), "evaluated": len(scores),
            "solved": sum(score["loss"] <= threshold for score in scores),
            "max_loss": max(score["loss"] for score in scores),
            "learning_costs": {key: sum(result["costs"][key] for result in results) for key in COSTS},
            "terminal_measurement_cost": sum(score["measurement_cost"] for score in scores),
            "selected_kinds": Counter(result["selected_kind"] for result in results),
            "method_actions": Counter(item["action"] for item in actions
                                      if "program_experiment_method" in item["action"]),
            "transfer_condition_actions": Counter(item["action"] for item in actions
                                                   if "transfer_condition" in item["action"]
                                                   or item["action"] == "refit_transfer_after_experiment"),
            "condition_experiment_costs": {
                "observation_cost": sum(item["observation_cost"] for item in condition_acquisitions),
                "measurement_cost": sum(item["measurement_cost"] for item in condition_acquisitions),
                "planning_compute": sum(item.get("compute_cost", 0) for item in condition_plans),
                "assessment_compute": sum(item["evidence"].get("compute_cost", 0) for item in condition_tests),
                "refit_and_validation_compute": sum(item.get("compute_cost", 0) for item in condition_refits)},
            "condition_plan_outcomes": Counter(item["forecast_status"] for item in condition_plans),
            "condition_measurement_outcomes": Counter(item["evidence"]["status"] for item in condition_tests),
            "short_transfer_trials": len(short_trials),
            "short_trials_with_supported_random_validation": sum(item["validation_loss"] <=
                protocol["research_config"]["minimum_improvement"] for item in short_trials),
            "cross_task_method_uses": sum(item["action"] == "use_program_experiment_method"
                                          and item.get("cross_task", False) for item in actions)}
    for seed, old, new in zip(protocol["seeds"], reports["v071"], reports["v072"]):
        for position, task in enumerate(protocol["tasks_by_seed"][str(seed)]):
            left, right = old["results"][position]["research"], new["results"][position]["research"]
            summary["paired_tasks"].append({"seed": seed, "position": position, "readout": task["readout"],
                "old_loss": old["evaluation"][position]["loss"], "new_loss": new["evaluation"][position]["loss"],
                "old_kind": left["selected_kind"], "new_kind": right["selected_kind"],
                "cost_delta": {key: right["costs"][key] - left["costs"][key] for key in COSTS}})
    summary["learning_cost_delta"] = {key: summary["arms"]["v072"]["learning_costs"][key] -
        summary["arms"]["v071"]["learning_costs"][key] for key in COSTS}
    write(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.summary_only:
        summarize(protocol)
        return
    if args.seed not in protocol["seeds"] or args.source_root is None:
        parser.error("provide a frozen runtime and one existing protocol seed")
    sys.path.insert(0, str(args.source_root.resolve()))
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    path = OUTPUT / f"{args.seed}_v072.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved and len(saved.get("evaluation", [])) == len(protocol["tasks_by_seed"][str(args.seed)]):
        return
    session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"], "seed": args.seed}),
        tasks=protocol["tasks_by_seed"][str(args.seed)], state=saved,
        measurement_budget=protocol["measurement_budget"])
    session.run(on_checkpoint=lambda report: write(path, report))
    write(path, session.evaluate(**protocol["evaluation"]))
    print(f"{args.seed}: {len(protocol['tasks_by_seed'][str(args.seed)])} tasks finished", flush=True)


if __name__ == "__main__":
    main()

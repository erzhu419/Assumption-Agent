#!/usr/bin/env python3
"""Compare inherited relation intervals on twelve fresh apparatus sequences."""

import argparse
from collections import Counter
import json
from pathlib import Path

import run_relation_interval_confirmation as study
import summarize_dynamic_research_replication as summary


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/inherited_relation_intervals_20260914"
PAIR = ("hegel_v081", "hegel_v0802")


def add_work_comparison(result, reports, protocol, position=None):
    group = result if position is None else result["stages"][protocol["stages"][position]]
    paired = group["paired"][f"{PAIR[0]}_minus_{PAIR[1]}"]
    values = {arm: {} for arm in PAIR}
    for arm in PAIR:
        for seed in protocol["paired_seeds"]:
            report = reports.get(seed, {}).get(arm, {})
            entries = [entry for entry in report.get("session", {}).get("results", [])
                       if position is None or entry["task_position"] == position]
            complete = len(entries) == (len(protocol["stages"]) if position is None else 1)
            values[arm][seed] = {
                "compute_cost": sum(entry["research"]["costs"]["compute_cost"] for entry in entries)
                    if complete else None,
                "learning_wall_seconds": (report.get("learning_wall_seconds") if position is None
                    else entries[0].get("learning_wall_seconds")) if complete else None}
        known_wall = [row["learning_wall_seconds"] for row in values[arm].values()
                      if row["learning_wall_seconds"] is not None]
        group["arms"][arm]["wall_counted_sequences"] = len(known_wall)
        group["arms"][arm]["recorded_learning_wall_seconds"] = sum(known_wall) if known_wall else None
    for row in paired["sequences"]:
        for key in ("compute_cost", "learning_wall_seconds"):
            left, right = (values[arm][row["seed"]][key] for arm in PAIR)
            row[key] = left - right if left is not None and right is not None else None
    for key in ("compute_cost", "learning_wall_seconds"):
        paired[key] = summary._deltas(row[key] for row in paired["sequences"])


def summarize(output, protocol):
    summary.PAIRS = (PAIR,)
    reports = {}
    for seed in protocol["paired_seeds"]:
        paths = {arm: output / "runs" / arm / str(seed) / "evaluation_report.json"
                 for arm in protocol["arms"]}
        reports[seed] = {}
        for arm, terminal_path in paths.items():
            path = terminal_path if terminal_path.exists() else terminal_path.with_name("hegel_checkpoint.json")
            if path.exists():
                reports[seed][arm] = json.loads(path.read_text())
        if all(path.exists() for path in paths.values()):
            study.save(output / "comparisons" / f"{seed}.json", {
                "seed": seed, "status": "evaluated",
                "arm_report_paths": {arm: str(path.relative_to(output)) for arm, path in paths.items()}})
    result = summary.summarize(protocol, reports)
    mechanisms = {arm: [] for arm in protocol["arms"]}
    for seed, arms in reports.items():
        for arm, report in arms.items():
            for entry in report["session"]["results"]:
                research = entry["research"]
                actions = research["actions"]
                steps = (research.get("research_program") or {}).get("selection_steps", [])
                mechanisms[arm].append({
                    "seed": seed, "task_position": entry["task_position"],
                    "stage": protocol["stages"][entry["task_position"]],
                    "research_status": research["status"],
                    "action_counts": dict(Counter(action["action"] for action in actions)),
                    "condition_acquisitions": sum(action["action"] == "acquire" and
                        action.get("purpose") == "distinguish_program_conditions" for action in actions),
                    "directed_program_experiments": sum(action["action"] == "test_competing_program_predictions"
                        for action in actions),
                    "interval_refinement_actions": [action for action in actions
                        if action["action"] == "refine_relation_parameter_space"],
                    "inherited_interval_actions": [action for action in actions if action["action"] in
                        ("inherit_relation_intervals", "condition_inherited_relation_intervals")],
                    "inherited_probe_acquisitions": sum(action["action"] == "acquire" and
                        action.get("purpose") == "distinguish_inherited_relation" for action in actions),
                    "inherited_probe_plans": [{key: action[key] for key in
                        ("probe_id", "forecast_status", "proposal_source", "compute_cost") if key in action}
                        for action in actions if action["action"] == "plan_inherited_relation_experiment"],
                    "inherited_probe_results": [{"probe_id": action["probe_id"],
                        **{key: action["evidence"][key] for key in
                           ("status", "observation_rows", "compute_cost", "calibration_status")},
                        "hypotheses": [{key: item[key] for key in
                            ("hypothesis_id", "status", "measured_loss", "forecast_loss", "rows")}
                            for item in action["evidence"]["hypotheses"]]}
                        for action in actions if action["action"] == "test_inherited_relation_predictions"],
                    "relation_selection_steps": [{**{key: step[key] for key in
                        ("operation", "radius", "radius_interval", "parameters", "parameter_index") if key in step},
                        "has_parameter_template": "parameter_template" in step}
                        for step in steps if step.get("radius_interval") is not None or
                            step["operation"] == "resolve_relation_intervals"]})
    result["stage_mechanisms"] = mechanisms
    add_work_comparison(result, reports, protocol)
    for position in range(len(protocol["stages"])):
        add_work_comparison(result, reports, protocol, position)
    study.save(output / "summary.json", result)
    print(json.dumps({arm: {key: values[key] for key in ("solved", "failed", "pending_terminal",
        "fully_solved_sequences", "recorded_session_costs", "recorded_queries")}
        for arm, values in result["arms"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learn", "evaluate", "summarize"))
    parser.add_argument("--arm", choices=PAIR)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = json.loads((args.output / "protocol.json").read_text())
    if args.mode == "summarize":
        summarize(args.output, protocol)
        return
    if args.arm is None or not args.seeds or any(seed not in protocol["paired_seeds"] for seed in args.seeds):
        parser.error("learning/evaluation needs an arm and seeds declared in the fixed protocol")
    module, source = study.load_runtime(protocol, args.arm)
    for seed in args.seeds:
        if args.mode == "learn":
            study.learn(args.output, protocol, args.arm, seed, module, source)
        else:
            study.evaluate(args.output, protocol, args.arm, seed, module)


if __name__ == "__main__":
    main()

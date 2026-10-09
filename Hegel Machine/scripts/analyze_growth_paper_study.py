#!/usr/bin/env python3
"""Derive confirmation endpoints and prefix-inclusive bills from saved cases.

This post-run analysis does not import the researcher or change the frozen
protocol, source snapshot, case files, or original summary.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ("observation_cost", "measurement_cost", "compute_cost")
ARMS = ("shared", "task_local")
PREFIX_LENGTHS = {"expression": 1, "units": 1, "methods": 4, "measurement": 3}


def _percentile(values, fraction):
    position = fraction * (len(values) - 1)
    left = int(position)
    return values[left] + (position - left) * (
        values[min(left + 1, len(values) - 1)] - values[left])


def paired_interval(differences, cases, protocol):
    settings = protocol["uncertainty"]
    rng = Random(settings["seed"])
    groups = [[i for i, case in enumerate(cases) if case["family"] == family]
              for family in protocol["families"]]
    values = []
    for _ in range(settings["replicates"]):
        indices = [index for group in groups if group
                   for index in rng.choices(group, k=len(group))]
        values.append(mean(differences[index] for index in indices))
    values.sort()
    return {"mean_shared_minus_task_local_tasks_per_seed": mean(differences),
            "paired_95_percent_interval": [_percentile(values, .025),
                                           _percentile(values, .975)],
            "shared_better": sum(value > 0 for value in differences),
            "equal": sum(value == 0 for value in differences),
            "shared_worse": sum(value < 0 for value in differences)}


def endpoints(tasks):
    return {"tasks": len(tasks),
            "terminal_solved": sum(task["solved"] for task in tasks),
            "internally_confirmed": sum(task["confirmed"] for task in tasks),
            "confirmed_and_terminal_solved": sum(
                task["confirmed"] and task["solved"] for task in tasks)}


def bill_sum(bills):
    return {"costs": {key: sum(bill["costs"][key] for bill in bills)
                      for key in RESOURCES},
            "training_cpu_seconds": sum(bill["training_cpu_seconds"] for bill in bills)}


def analyze(cases, protocol):
    per_case = [{"seed": case["seed"], "family": case["family"],
                 **{arm: endpoints(case["results"][arm]["tasks"]) for arm in ARMS}}
                for case in cases]
    differences = [row["shared"]["confirmed_and_terminal_solved"]
                   - row["task_local"]["confirmed_and_terminal_solved"] for row in per_case]
    result = {
        "analysis": "Post-run descriptive endpoints; frozen research and acceptance rules unchanged.",
        "planned_cases": len(protocol["seeds"]), "completed_cases": len(cases),
        "joint_endpoint_definition": "The same task is internally confirmed AND terminal-solved.",
        "bootstrap": protocol["uncertainty"],
        "joint_endpoint": {
            "totals": {arm: endpoints([task for case in cases
                       for task in case["results"][arm]["tasks"]]) for arm in ARMS},
            "paired_interval": paired_interval(differences, cases, protocol),
            "per_seed": per_case,
            "per_family": {family: {arm: endpoints([task for case in cases
                if case["family"] == family for task in case["results"][arm]["tasks"]])
                for arm in ARMS} for family in protocol["families"]}},
        "shared_event_activation": [], "uncertainty": {},
        "confirmation_mismatch_cases": [], "prefix_inclusive_bills": {},
    }
    for stage, name in enumerate(protocol["stages"]):
        activations = Counter(action for case in cases for action in {
            event["action"] for event in case["results"]["shared"]["tasks"][stage]["research"]["events"]})
        result["shared_event_activation"].append({"stage": stage, "stage_name": name,
            "denominator": len(cases), "seeds_with_action": dict(sorted(activations.items()))})
    noise_stage = protocol["stages"].index("stochastic_response")
    result["uncertainty"] = {arm: {"denominator": len(cases),
        "coordinate_dependent_variance_program": sum(
            case["results"][arm]["tasks"][noise_stage]["uncertainty"].get("variance_program")
            is not None for case in cases)} for arm in ARMS}
    result["persistent_method_revision"] = {"denominator": len(cases),
        "seeds_with_parent_linked_method": sum(any(method.get("parent_method_id")
            for method in case["persistent_knowledge"]["methods"]["methods"]) for case in cases)}
    for case in cases:
        for arm in ARMS:
            for task in case["results"][arm]["tasks"]:
                if task["confirmed"] == task["solved"]:
                    continue
                result["confirmation_mismatch_cases"].append({"seed": case["seed"],
                    "family": case["family"], "arm": arm, "stage": task["stage"],
                    "stage_name": task["stage_name"], "terminal_solved": task["solved"],
                    "internally_confirmed": task["confirmed"],
                    "status": task["research"]["status"], "terminal_mean_mse": task["mean_mse"],
                    "noise_variance_ratio": task["noise_variance_ratio"],
                    "costs": task["research"]["costs"]})
    result["billing_note"] = (
        "Each whole-sequence bill includes its actually acquired common prefix once plus its "
        "successor. Suffix differences exclude the identical prefix; whole-sequence differences "
        "are therefore equal. Prefixes overlap across mechanism studies and must not be summed "
        "as independent acquisitions. These bills exclude independent terminal evaluation and "
        "prediction-only diagnostics. Main-arm CPU order is randomized; mechanism suffix CPU "
        "comparisons reuse the earlier shared run and are not order-balanced.")
    for name, length in PREFIX_LENGTHS.items():
        rows = [case["contrasts"][name] for case in cases]
        result["prefix_inclusive_bills"][name] = {
            "cases": len(cases), "common_prefix_tasks_per_case": length,
            "common_prefix": bill_sum([row["common_prefix"] for row in rows]),
            "suffix": {arm: bill_sum([row[arm] for row in rows])
                       for arm in ("retained", "control")},
            "whole_sequence": {arm: {
                "tasks_including_prefix": sum(row["whole_sequence"][arm]["tasks"] for row in rows),
                **bill_sum([row["whole_sequence"][arm] for row in rows])}
                for arm in ("retained", "control")}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "artifacts/growth_paper_20261006_main_30")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.input / "protocol.json").read_text())
    paths = [args.input / f"case_{seed}.json" for seed in sorted(protocol["seeds"])]
    cases = [json.loads(path.read_text()) for path in paths]
    result = analyze(cases, protocol)
    result["inputs"] = {"protocol": str(args.input / "protocol.json"),
                        "cases": [str(path) for path in paths]}
    output = args.output or args.input / "derived_analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cases": len(cases),
                     "joint_endpoint": result["joint_endpoint"]["totals"]}))


if __name__ == "__main__":
    main()

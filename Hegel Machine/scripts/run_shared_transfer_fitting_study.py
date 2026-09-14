#!/usr/bin/env python3
"""Measure shared transfer fitting with the unchanged full-library task streams."""

import argparse
import json
from pathlib import Path

import run_shared_condition_research_study as previous

transfer = previous.transfer
ROOT = previous.ROOT
ARTIFACTS = ROOT / "artifacts/shared_transfer_fitting_20260913"
PREVIOUS = ROOT / "artifacts/shared_condition_research_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
ARMS = ("v073", "shared_fitting")
FITTING_ACTIONS = ("shared_transfer_fit", "shared_transfer_validation")


def work_metrics(events):
    sharing_keys = {key for event in events for key, value in event.get("sharing", {}).items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)}
    return {"events": len(events),
        **{key: sum(event.get(key, 0) for event in events)
           for key in ("compute_cost", "observation_rows", "candidate_count")},
        "sharing": {key: sum(event.get("sharing", {}).get(key, 0) for event in events)
                    for key in sorted(sharing_keys)}}


def task_metrics(task):
    result = previous.task_metrics(task)
    events = [a for a in task["research"]["actions"] if a["action"] in FITTING_ACTIONS]
    result["fitting"] = {action: work_metrics([e for e in events if e["action"] == action])
                         for action in FITTING_ACTIONS}
    result["fitting_details"] = events
    return result


def aggregate(tasks):
    events = [event for task in tasks for event in task["fitting_details"]]
    return {**previous.aggregate(tasks),
        "fitting": {action: work_metrics([e for e in events if e["action"] == action])
                    for action in FITTING_ACTIONS}}


def report_path(output, arm, seed, cohort):
    if arm == "v073" and cohort != "development":
        return PREVIOUS / "comparison_v1/runs" / f"{seed}_shared_conditions.json"
    return output / "runs" / f"{seed}_{arm}.json"


def summarize(output, protocol):
    summaries, missing = {}, []
    for cohort, seeds in protocol["cohorts"].items():
        summaries[cohort] = {}
        for arm in ARMS:
            rows = []
            for seed in seeds:
                path = report_path(output, arm, seed, cohort)
                if cohort == "development" and not path.exists():
                    path = ARTIFACTS / "development_v1/runs" / f"{seed}_{arm}.json"
                if not path.exists():
                    missing.append({"cohort": cohort, "arm": arm, "seed": seed})
                    continue
                report = json.loads(path.read_text())
                rows.extend({"seed": seed, "case": case["case"], "stage": task["stage"],
                    "source": str(path.resolve().relative_to(ROOT)), **task_metrics(task)}
                    for case in report["cases"] for task in case["tasks"])
            summaries[cohort][arm] = {"all": aggregate(rows),
                "cases": {case: aggregate([t for t in rows if t["case"] == case])
                          for case in protocol["cases"]},
                "stages": {case: {name: aggregate([t for t in rows if t["case"] == case and t["stage"] == stage])
                                  for stage, name in enumerate(protocol["stages"])}
                           for case in protocol["cases"]}, "tasks": rows}
    result = {"cohorts": summaries, "missing_runs": missing,
        "baseline_source": "artifacts/shared_transfer_fitting_20260913/v073_runtime/src",
        "baseline_development": "Final 0.73 development was run once because the prior study's development source preceded two final cognition edits; old formal60 are reused unchanged."}
    transfer.write(output / "summary.json", result)
    print(json.dumps({cohort: {arm: data["all"] for arm, data in arms.items()}
                      for cohort, arms in summaries.items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison_v1")
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--arm", choices=ARMS, default="shared_fitting")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.summary_only:
        summarize(args.output, protocol)
        return
    cohort = next((name for name, seeds in protocol["cohorts"].items() if args.seed in seeds), None)
    if cohort is None:
        parser.error("seed must be declared in the frozen protocol")
    destination = report_path(args.output, args.arm, args.seed, cohort)
    if destination.exists():
        parser.error(f"completed run already exists: {destination}")
    if args.arm == "v073" and cohort != "development":
        parser.error("reuse the completed final 0.73 formal cohorts")
    source = args.source or ARTIFACTS / ("v073_runtime" if args.arm == "v073" else "v074_runtime") / "src"
    if not source.is_dir():
        parser.error(f"freeze the runtime source first: {source}")
    transfer.load_runtime(source)
    transfer.run(args.output, protocol, args.arm, args.seed)


if __name__ == "__main__":
    main()

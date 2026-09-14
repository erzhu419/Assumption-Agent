#!/usr/bin/env python3
"""Learn independent apparatus sequences with the frozen ordinary-Code loop."""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/dynamic_research_replication_20260913"
SPEC = importlib.util.spec_from_file_location("dynamic_replication_base", ARTIFACTS / "base_dynamic_code_comparison.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
base.ROOT = ROOT
original_tasks_for = base.tasks_for


def tasks_for(seed, protocol):
    return [{**item, "profile": protocol["task_profile"]} for item in original_tasks_for(seed, protocol)]


base.tasks_for = tasks_for


def evaluate_arm(output, protocol, seed, selected):
    reports = {}
    for name, arm in protocol["arms"].items():
        path = output / "runs" / name / str(seed) / f"{arm['session']}_checkpoint.json"
        reports[name] = json.loads(path.read_text())
    if any(len(report["session"]["results"]) != len(report["session"]["tasks"])
           for report in reports.values()):
        raise ValueError("all three learning sequences must stop before independent evaluation")
    directory = output / "runs" / selected / str(seed)
    destination = directory / "evaluation_report.json"
    if destination.exists():
        return json.loads(destination.read_text())
    arm = protocol["arms"][selected]
    base.load_runtime(ROOT / arm["runtime_source"])
    report = reports[selected]
    session = base.new_session(arm["session"], directory, protocol, seed, state=report["session"])
    session.evaluate(**{key: protocol["terminal"][key] for key in ("length", "trajectories", "seed_offset")})
    report["session"] = session.to_dict()
    base.save(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("code", "hegel_v078", "hegel_revised"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ARTIFACTS)
    parser.add_argument("--config", type=Path, default=ARTIFACTS / "protocol.json")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--evaluate-arm", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.seed not in protocol["paired_seeds"]:
        parser.error("seed is not declared in the fixed protocol")
    if args.evaluate_arm:
        evaluate_arm(args.output, protocol, args.seed, args.arm)
        print(f"{args.arm}: independent evaluation retained for seed {args.seed}")
        return
    arm = protocol["arms"][args.arm]
    source = ROOT / arm["runtime_source"]
    base.load_runtime(source)
    base.learn(arm["session"], args.output / "runs" / args.arm / str(args.seed),
               protocol, args.seed, source, args.max_tasks)


if __name__ == "__main__":
    main()

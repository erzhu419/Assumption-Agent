#!/usr/bin/env python3
"""Run an independent anchored apparatus from empty object knowledge."""

import argparse
import json
from pathlib import Path
import shutil

import run_partial_object_study as study


def protocol_for(base):
    return {**base, "experiment": "anchored_objects_from_empty_v066",
        "tasks_by_seed": {str(seed): [{**study.tasks_for(seed)[6], "seed": seed + 900007}]
                          for seed in base["seeds"]},
        "contract": "Independent empty-library anchored focus task. This does not share knowledge "
                    "or prediction coverage with the eight-task partial-object stream.",
        "planned_tasks": len(base["seeds"]) * len(base["arms"])}


def summarize(output, protocol):
    arms = {arm: {"tasks": 0, "solved": 0, "exact_objects": 0, "commands": 0,
        "learning_costs": dict.fromkeys(study.COST_KEYS, 0),
        "terminal_costs": dict.fromkeys(study.COST_KEYS, 0)} for arm in protocol["arms"]}
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        arm = path.stem.split("_", 1)[1]
        result = arms[arm]
        result["tasks"] += len(report["evaluation"])
        result["solved"] += sum(row["loss"] <= 1e-6 for row in report["evaluation"])
        result["exact_objects"] += sum(row["exact"] for row in report.get("object_evaluation", []))
        result["commands"] += sum(not row["research"]["selected_channels"] for row in report["results"])
        for key in study.COST_KEYS:
            result["learning_costs"][key] += report["costs"][key]
            result["terminal_costs"][key] += sum(row.get(key, 0) for row in report["evaluation"])
    summary = {"experiment": protocol["experiment"], "arms": arms}
    study.write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stream", required=True, type=Path, help="Existing eight-task protocol directory")
    parser.add_argument("--arm", choices=study.ARMS, default="partial_objects")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    source = study.BASELINE if args.arm == "v065" else study.ROOT / "src"
    runtime = study.load_runtime(source)
    print(json.dumps({"arm": args.arm, "runtime": runtime.__file__}), flush=True)
    protocol = protocol_for(json.loads((args.stream / "protocol.json").read_text()))
    path = args.output / "tasks.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("anchored task settings changed; preserve this output")
    study.write(path, protocol)
    if args.summary_only:
        summarize(args.output, protocol)
    else:
        study.research(args.output, protocol, args.arm, source, seed=args.seed)
        shutil.copyfile(Path(__file__), args.output / (args.arm + "_source") / Path(__file__).name)


if __name__ == "__main__":
    main()

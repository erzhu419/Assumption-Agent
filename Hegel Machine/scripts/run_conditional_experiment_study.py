#!/usr/bin/env python3
"""Compare paid conditional experiments on the unchanged private branch apparatus."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from run_conditional_discovery_study import evaluate, learn, run_dir, summarize as summarize_predictions
from run_knowledge_lineage_study import ROOT, read, save


def summarize(output, protocol):
    summarize_predictions(output, protocol)
    summary = read(output / "summary.json")
    for arm, total in summary["totals"].items():
        acquisitions, paid_rows = Counter(), Counter()
        for root in protocol["roots"]:
            state = read(run_dir(output, arm, root) / "checkpoint.json.gz")
            for result in state["results"]:
                for action in result["research"]["actions"]:
                    if action["action"] == "acquire":
                        purpose = action["purpose"]
                        acquisitions[purpose] += 1
                        paid_rows[purpose] += action["observation_cost"]
        total["paid_acquisitions_by_purpose"] = dict(acquisitions)
        total["paid_rows_by_purpose"] = dict(paid_rows)
        total["conditional_experiment_rows"] = sum(value for purpose, value in
            paid_rows.items() if purpose.startswith("conditional_"))
    summary["mechanism_groups"] = {}
    for name, positions in (("conditional_branches", (1, 2, 3, 4, 5)),
                            ("unconditional_controls", (0, 6))):
        group = summary["mechanism_groups"][name] = {}
        for arm in protocol["arms"]:
            rows = [row for row in summary["rows"] if row["arm"] == arm
                    and row["position"] in positions]
            group[arm] = {"tasks": len(rows),
                "completed": sum(row["status"] == "completed" for row in rows),
                "solved": sum(row["mean_response_mse"] <= protocol["pass_mse"] for row in rows),
                "false_completed": sum(row["status"] == "completed" and
                    row["mean_response_mse"] > protocol["pass_mse"] for row in rows),
                "mean_response_mse": sum(row["mean_response_mse"] for row in rows) / len(rows)}
    save(output / "summary.json", summary)


def dispatch(output, protocol, phase, arms, workers):
    jobs = [(arm, root) for arm in arms or protocol["arms"] for root in protocol["roots"]]
    def run(job):
        arm, root = job
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
            "--phase", phase, "--arm", arm, "--root", root], cwd=ROOT, check=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(run, jobs))
    print({"phase": phase, "finished_jobs": len(jobs)}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("learn", "evaluate", "summarize"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--arm")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    if args.phase == "summarize":
        summarize(output, protocol)
    elif args.root:
        {"learn": learn, "evaluate": evaluate}[args.phase](output, protocol, args.arm, args.root)
    else:
        dispatch(output, protocol, args.phase, args.arms, args.workers)


if __name__ == "__main__":
    main()

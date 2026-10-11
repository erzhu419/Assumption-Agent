#!/usr/bin/env python3
"""Run this fixed cohort as individual arm/apparatus jobs, at most two at once."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
DRIVER = ROOT / "scripts/run_relation_condition_composition.py"


def completed(mode, arm, seed, task_count):
    directory = OUTPUT / "runs" / arm / str(seed)
    if mode == "evaluate":
        return (directory / "evaluation_report.json").exists()
    path = directory / "hegel_checkpoint.json"
    return path.exists() and len(json.loads(path.read_text())["session"]["results"]) == task_count


def run_job(job):
    mode, arm, seed = job
    log = OUTPUT / "logs" / f"{arm}_{seed}_{mode}.log"
    command = [sys.executable, str(DRIVER), mode, "--arm", arm, "--seeds", str(seed)]
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    report = {"mode": mode, "arm": arm, "seed": seed, "exit_code": result.returncode,
              "command": command, "log": str(log.relative_to(OUTPUT))}
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("learn", "evaluate", "all"))
    args = parser.parse_args()
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    task_count = len(protocol["stages"])
    modes = ("learn", "evaluate") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if mode == "evaluate" and any(not completed("learn", arm, seed, task_count)
            for seed in protocol["paired_seeds"] for arm in protocol["arms"]):
            raise RuntimeError("finish both arms' full declared learning cohort before terminal evaluation")
        jobs = [(mode, arm, seed) for seed in protocol["paired_seeds"] for arm in protocol["arms"]
                if not completed(mode, arm, seed, task_count)]
        print(json.dumps({"mode": mode, "pending_jobs": len(jobs), "max_processes": 2}), flush=True)
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(run_job, jobs))
        (OUTPUT / f"{mode}_queue_results.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(row["exit_code"] != 0 for row in results):
            raise SystemExit("Some processes failed; retained their logs and checkpoints without automatic retries.")
    if args.mode in ("evaluate", "all"):
        subprocess.run([sys.executable, str(DRIVER), "summarize"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

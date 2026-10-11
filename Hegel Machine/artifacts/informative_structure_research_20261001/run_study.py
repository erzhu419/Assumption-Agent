#!/usr/bin/env python3
"""Run one frozen study family, or record the shared learning completion boundary."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def learning_gate():
    chains = []
    for family in ("numeric", "risk"):
        for cohort in ("development", "fresh_replication"):
            directory = BASE / cohort if family == "numeric" else BASE / "risk" / cohort
            protocol = read(directory / "protocol.json" if family == "numeric" else BASE / "risk/protocol.json")
            roots = protocol["roots"] if family == "numeric" else protocol["cohorts"][cohort]
            for arm in protocol["arms"]:
                for root in roots:
                    checkpoint = (directory / "runs" / arm / str(root) / "checkpoint.json.gz"
                        if family == "numeric" else directory / "runs/risk" / arm / str(root) / "checkpoint.json")
                    state = read(checkpoint)
                    complete = (state["study"]["learning_complete"] if family == "numeric"
                                else state["study_learning_complete"])
                    terminal_rows = len(state.get("evaluation", []))
                    if family == "numeric" and (checkpoint.parent / "evaluation.json").exists():
                        terminal_rows += len(read(checkpoint.parent / "evaluation.json"))
                    if not complete or state["pending_revisits"] or terminal_rows:
                        raise ValueError(f"Learning/natural revisits must finish before terminal: {checkpoint}")
                    chains.append({"family": family, "cohort": cohort,
                        "checkpoint": str(checkpoint.relative_to(ROOT)), "complete": True,
                        "natural_revisits_finished": True, "terminal_rows_at_gate": 0})
    if len(chains) != 78:
        raise ValueError(f"Expected 78 learning chains, found {len(chains)}")
    result = {"all_learning_chains_complete": True, "numeric_chains": 39, "risk_chains": 39,
        "total_chains": 78, "terminal_started_at_gate": False, "external_model_calls": 0,
        "chains": chains}
    (BASE / "learning_completion_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print({key: value for key, value in result.items() if key != "chains"}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("learn", "gate", "evaluate", "summarize"))
    parser.add_argument("--family", choices=("numeric", "risk"))
    args = parser.parse_args()
    if args.phase == "gate":
        learning_gate()
        return
    if args.family is None:
        parser.error("--family is required outside gate")
    if args.phase == "evaluate" and not read(BASE / "learning_completion_gate.json")["all_learning_chains_complete"]:
        raise ValueError("Finish all 78 learning chains and natural revisits before terminal")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts")}
    for cohort in ("development", "fresh_replication"):
        if args.family == "numeric":
            command = [sys.executable, str(BASE / "fresh_replication/runtime/run_observed_condition_decoding_study.py"),
                "--output", str(BASE / cohort), "--phase", args.phase, "--workers", "4"]
        else:
            command = [sys.executable, str(BASE / "risk_runner.py"), "--output", str(BASE / "risk"),
                "--phase", args.phase, "--cohort", cohort, "--workers", "4"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        print({"family": args.family, "cohort": cohort, "phase": args.phase, "complete": True}, flush=True)


if __name__ == "__main__":
    main()

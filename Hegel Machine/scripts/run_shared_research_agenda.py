"""Run the declared shared-prerequisite example and its separate-source control."""

import argparse
import json
from pathlib import Path

from hegel_machine.cognition.agenda_dependency_demo import run_dependency_demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the checkpoint and compact result")
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    report = run_dependency_demo(args.output, seed=args.seed)
    print(json.dumps({"summary": str((args.output / "summary.json").resolve()),
        "shared_costs": report["shared"]["spent"], "separate_costs": report["separate_total_costs"],
        "shared_minus_separate_costs": report["shared_minus_separate_costs"],
        "successor_evaluation": report["shared"]["evaluation"]}, indent=2))

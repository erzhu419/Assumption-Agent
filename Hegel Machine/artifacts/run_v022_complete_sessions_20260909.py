"""Run the preselected complete 0.22 paired sessions after production freezes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hegel_machine import __version__
from hegel_machine.cognition.models import ResearchConfig
from run_method_session_study import run_study

DATE = "20260909"
SEEDS = (5501, 4507, 6781)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    parser.add_argument("--label", default="v022_release")
    args = parser.parse_args()
    for seed in args.seed:
        baseline_label = "v022_baseline" if seed == 6781 else "v021_final"
        baseline_path = ROOT / f"artifacts/hegel_machine_{baseline_label}_seed{seed}_paired_{DATE}.json"
        baseline = json.loads(baseline_path.read_text())
        config = ResearchConfig(seed=seed)
        assert baseline["seed"] == seed and baseline["summary"]["task_count"] == 65
        assert (config.observation_budget, config.compute_budget, config.max_rounds) == (96, 24000, 5)
        assert all(baseline["config"][field] == getattr(config, field)
                   for field in ("seed", "observation_budget", "compute_budget", "max_rounds"))
        assert baseline["limits_per_arm"] == {
            "observation_budget": 65 * 96 * 2, "compute_budget": 65 * 24000 * 2, "max_visits": 2}
        report = run_study(seed=seed, label=args.label, date=DATE)
        compact = {"seed": seed, "version": __version__, "baseline_paired": str(baseline_path),
            "config": report["config"], "limits_per_arm": report["limits_per_arm"],
            "partitions": report["partitions"], "evaluation": report["evaluation"],
            "summary": {key: value for key, value in report["summary"].items() if key != "tasks"},
            "checkpoints": report["checkpoints"], "evaluations": report["evaluations"],
            "wall_time_seconds": report["wall_time_seconds"]}
        output = ROOT / f"artifacts/hegel_machine_{args.label}_seed{seed}_summary_{DATE}.json"
        with output.open("x") as stream:
            json.dump(compact, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"completed_seed": seed, "summary": str(output)}), flush=True)


if __name__ == "__main__":
    main()

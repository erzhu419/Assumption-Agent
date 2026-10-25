#!/usr/bin/env python3
"""Run 0.71 against the completed 0.70 object streams, without rerunning them."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/hegel_machine/cognition"))
from checkpoint_storage import write_json_checkpoint
BASELINE = ROOT / "artifacts/joint_experiment_methods_20260913/comparison_v1"
OUTPUT = ROOT / "artifacts/compound_experiment_methods_20260913/object_regression_v1"


def write(path, payload):
    write_json_checkpoint(path, payload, indent=2, ensure_ascii=True, allow_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((BASELINE / "protocol.json").read_text())
    if args.summary_only:
        reports = {}
        for arm, folder, suffix in (("v070", BASELINE / "runs", "joint_methods"),
                                    ("v071", OUTPUT, "v071")):
            reports[arm] = [json.loads((folder / f"{seed}_{suffix}.json").read_text())
                            for seed in protocol["seeds"]]
        summary = {"baseline": str(BASELINE), "seeds": protocol["seeds"], "arms": {}}
        for arm, runs in reports.items():
            results = [item["research"] for run in runs for item in run["results"]]
            scores = [score for run in runs for score in run["evaluation"]]
            actions = [item for result in results for item in result["actions"]]
            summary["arms"][arm] = {"tasks": len(results), "evaluated": len(scores),
                "solved": sum(score["loss"] <= 1e-6 for score in scores),
                "max_loss": max(score["loss"] for score in scores),
                "learning_costs": {key: sum(result["costs"][key] for result in results)
                    for key in ("observation_cost", "measurement_cost", "compute_cost")},
                "terminal_measurement_cost": sum(score["measurement_cost"] for score in scores),
                "method_actions": Counter(item["action"] for item in actions
                                          if "program_experiment_method" in item["action"]),
                "cross_task_method_uses": sum(item["action"] == "use_program_experiment_method"
                                              and item.get("cross_task", False) for item in actions)}
        write(OUTPUT / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return
    if args.seed not in protocol["seeds"] or args.source_root is None:
        parser.error("provide a frozen runtime and one existing protocol seed")
    sys.path.insert(0, str(args.source_root.resolve()))
    from hegel_machine.cognition.models import ResearchConfig
    from hegel_machine.cognition.unified_session import UnifiedResearchSession
    path = OUTPUT / f"{args.seed}_v071.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved and len(saved.get("evaluation", [])) == 5:
        return
    session = UnifiedResearchSession(ResearchConfig(**{**protocol["research_config"], "seed": args.seed}),
        tasks=protocol["tasks_by_seed"][str(args.seed)], state=saved,
        measurement_budget=protocol["measurement_budget"])
    session.run(on_checkpoint=lambda report: write(path, report))
    write(path, session.evaluate(**protocol["evaluation"]))
    print(f"{args.seed}: 5 tasks finished", flush=True)


if __name__ == "__main__":
    main()

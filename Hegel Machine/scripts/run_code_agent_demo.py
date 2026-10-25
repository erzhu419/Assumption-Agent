#!/usr/bin/env python3
"""Run the configured Codex model as an ordinary Python experimental agent."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.checkpoint_storage import write_text_incrementally

from hegel_machine.cognition.code_agent import CodeAgent, CodeAgentConfig, CodexCodeProvider, evaluate_python_model
from hegel_machine.cognition.measurement_lab import make_lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--tasks", type=int, nargs="+", choices=(0, 1, 2), default=[0])
    parser.add_argument("--backend", choices=("process", "local"), default="process")
    parser.add_argument("--seed", type=int, default=313)
    parser.add_argument("--max-calls", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=60000)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--observation-budget", type=int, default=128)
    parser.add_argument("--measurement-budget", type=int, default=512)
    parser.add_argument("--compute-budget", type=int, default=100000)
    parser.add_argument("--execution-seconds", type=float, default=60.)
    parser.add_argument("--model")
    args = parser.parse_args()
    workspace = args.workspace or args.output.with_suffix("").with_name(args.output.stem + "_workspace")
    config = CodeAgentConfig(observation_budget=args.observation_budget, measurement_budget=args.measurement_budget,
        compute_budget=args.compute_budget, execution_seconds=args.execution_seconds,
        max_model_calls=args.max_calls, token_budget=args.token_budget)
    report = {"experiment": "configured_codex_ordinary_python_agent", "config": asdict(config),
              "seed": args.seed, "backend": args.backend, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task_index in args.tasks:
        task_workspace = workspace / f"task_{task_index}"
        provider = CodexCodeProvider(task_workspace, timeout_seconds=args.timeout, model=args.model)
        with make_lab(task_index, args.seed + task_index * 10007, backend=args.backend) as laboratory:
            result = CodeAgent(provider, task_workspace, config).run(laboratory)
            result["task_index"] = task_index
            result["seed"] = args.seed + task_index * 10007
            if result["model"]:
                rows = laboratory.evaluation_rows(result["selected_channels"], seed=args.seed + task_index * 10007 + 1000003,
                                                  length=64, trajectories=4)
                evaluation = evaluate_python_model(task_workspace, result["model"], rows)
                evaluation.pop("predictions", None)
                result["evaluation"] = {**evaluation, "rows": len(rows), "compute_cost": len(rows)}
            report["results"].append(result)
            write_text_incrementally(args.output, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"task_index": task_index, "status": result["status"], "costs": result["costs"],
                              "evaluation": result.get("evaluation")}), flush=True)
            if result["status"] == "provider_failed":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score already-authored, undelivered Python models after frozen learning ends."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.code_agent import evaluate_python_model
from hegel_machine.cognition.unknown_task_lab import UnknownTaskLab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    target = directory / "secondary_authored_models.json"
    if target.exists():
        print(target)
        return
    protocol = json.loads((directory / "protocol.json").read_text())
    report = json.loads((directory / "runs" / f"{protocol['code_seed']}_ordinary_code.json").read_text())
    if len(report["results"]) != len(report["tasks"]) or report["evaluation"] is None:
        raise ValueError("complete all learning and primary evaluation first")
    scores = []
    for position, (task, result) in enumerate(zip(report["tasks"], report["results"])):
        if result["model"]:
            continue
        workspace = Path(result["research_workspace"])
        remaining_seconds = max(0., result["config"]["execution_seconds"]
                                   - result["costs"]["python_execution_seconds"])
        row = {"task_position": position, "primary_status": result["status"],
            "model_present": (workspace / "model.py").is_file(), "model": "model.py",
            "extra_python_seconds": 0., "training_scored_rows": 0,
            "evaluation_rows": 0, "evaluation_measurement_cost": 0, "mse": None}
        if not row["model_present"] or remaining_seconds <= 0:
            row["status"] = "missing_model" if not row["model_present"] else "python_budget_exhausted"
            scores.append(row)
            continue
        training_rows = json.loads((workspace / "observations.json").read_text())
        started = monotonic()
        training = evaluate_python_model(workspace, "model.py", training_rows,
                                         timeout_seconds=remaining_seconds)
        row["extra_python_seconds"] = monotonic()-started
        row["training_scored_rows"] = len(training_rows)
        training.pop("predictions", None)
        row["training_execution"] = training
        remaining_seconds -= row["extra_python_seconds"]
        if training["returncode"] != 0 or remaining_seconds <= 0:
            row["status"] = "training_execution_failed" if training["returncode"] != 0 else "python_budget_exhausted"
            scores.append(row)
            continue
        settings = protocol["evaluation"]
        with UnknownTaskLab(**task) as laboratory:
            observations = laboratory.evaluation_rows(result["selected_channels"],
                seed=task["seed"]+settings["seed_offset"], length=settings["length"],
                trajectories=settings["trajectories"])
            rate = sum(channel.cost_per_row for channel in laboratory.channels
                       if channel.name in result["selected_channels"])
        started = monotonic()
        evaluated = evaluate_python_model(workspace, "model.py", observations,
                                          timeout_seconds=remaining_seconds)
        row["extra_python_seconds"] += monotonic()-started
        evaluated.pop("predictions", None)
        row.update(status="completed" if evaluated["returncode"] == 0 else "evaluation_execution_failed",
            evaluation_execution=evaluated, mse=evaluated.get("mse"), evaluation_rows=len(observations),
            evaluation_measurement_cost=len(observations)*rate,
            selected_channels=result["selected_channels"])
        scores.append(row)
    secondary = {"analysis": "posthoc_execution_of_already_authored_canonical_model",
        "selection": "Primary missing-model tasks only; canonical model.py, without code changes, new model calls or new learning observations.",
        "information": "Training uses existing observations.json; final targets stay outside Python inputs. No final feedback enters session knowledge.",
        "channels": "The last channels selected by the ordinary agent, as retained in its primary result.",
        "primary_results_unchanged": True, "tasks": scores,
        "extra_python_seconds": sum(row["extra_python_seconds"] for row in scores),
        "extra_training_scored_rows": sum(row["training_scored_rows"] for row in scores),
        "extra_evaluation_observations": sum(row["evaluation_rows"] for row in scores),
        "extra_evaluation_measurement_cost": sum(row["evaluation_measurement_cost"] for row in scores)}
    target.write_text(json.dumps(secondary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(secondary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

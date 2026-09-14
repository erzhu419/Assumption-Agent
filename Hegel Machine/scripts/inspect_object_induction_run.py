#!/usr/bin/env python3
"""Print compact per-task evidence from completed or paused object studies."""

import argparse
import json
from pathlib import Path


COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")


def inspect(path):
    report = json.loads(path.read_text())
    evaluations = {row["task_position"]: row for row in report["evaluation"]}
    objects = {row["task_position"]: row for row in report.get("object_evaluation", [])}
    print(json.dumps({"run": str(path), "learned": len(report["results"]),
        "planned": len(report["tasks"]), "evaluated": len(evaluations),
        "solved_at_1e_6": sum(row["loss"] <= 1e-6 for row in evaluations.values()),
        "exact_objects": sum(row["exact"] for row in objects.values()),
        "learning_costs": report["costs"],
        "terminal_costs": {key: sum(row.get(key, 0) for row in evaluations.values())
                           for key in COST_KEYS}}))
    for entry in report["results"]:
        position = entry["task_position"]
        result = entry["research"]
        noise = result.get("noise_assessment") or {}
        object_research = result.get("object_research") or {}
        reopens = [event.get("object_model_id") for event in result["actions"]
                   if event["action"] == "reopen_object_dependencies"]
        print(json.dumps({"position": position, "status": result["status"],
            "loss": evaluations.get(position, {}).get("loss"),
            "schema_exact": objects.get(position, {}).get("exact"),
            "kind": result["selected_kind"], "channels": result["selected_channels"],
            "noise_reason": noise.get("reason"), "noise_adequate": noise.get("adequate"),
            "costs": {key: result["costs"][key] for key in COST_KEYS},
            "object_model_id": result.get("object_model_id"),
            "reopened_object_models": reopens,
            "object_unresolved": object_research.get("unresolved", [])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    for path in parser.parse_args().runs:
        inspect(path)


if __name__ == "__main__":
    main()

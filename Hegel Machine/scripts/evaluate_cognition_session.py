#!/usr/bin/env python3
"""Measure a saved learning session after it has stopped, including revisit value."""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.environments import make_suite
from hegel_machine.cognition.evaluation import decision_metrics, loss, predict
from hegel_machine.cognition.language import Program


def evaluate_session(session):
    config = session["config"]
    sources = [(partition, environment)
               for partition in session.get("partitions", [config["partition"]])
               for environment in make_suite(config["seed"], partition=partition)]
    if [environment.spec.task_id for _, environment in sources] != session["task_ids"]:
        raise ValueError("saved session does not match these simulator tasks")
    measured = []
    for episode in session["episodes"]:
        partition, environment = sources[episode["task_index"]]
        rows = environment.evaluation_rows()
        program = Program.from_dict(episode["program"])
        item = {"episode_index": episode["episode_index"], "task_index": episode["task_index"],
                "task_id": episode["task_id"], "partition": partition, "visit": episode["visit"],
                "loss_kind": "squared_error" if environment.spec.risk_quantile == .5 else "pinball",
                "loss": loss(program, rows, episode["fallback"], environment.spec.risk_quantile),
                "rows": len(rows), "observation_cost": episode["observation_cost"],
                "compute_cost": episode["compute_cost"], "expression": episode["expression"]}
        if environment.spec.decision_costs is not None and episode["uncertainty_model"]:
            item["decision"] = decision_metrics(program, rows, episode["fallback"],
                                               episode["uncertainty_model"], environment.spec.decision_costs)
        if environment.evaluation_metadata.get("prediction_kind") == "binary_state" or environment.evaluation_metadata.get("mechanism") == "hysteresis":
            item["state_errors"] = sum((predict(program, row, episode["fallback"]) >= .5) != (row.target >= .5)
                                       for row in rows)
        measured.append(item)
    tasks = []
    for index, (_, environment) in enumerate(sources):
        visits = [row for row in measured if row["task_index"] == index]
        if not visits:
            tasks.append({"task_index": index, "task_id": environment.spec.task_id, "visits": 0})
            continue
        first, latest = visits[0], visits[-1]
        tasks.append({"task_index": index, "task_id": environment.spec.task_id, "visits": len(visits),
                      "first": first, "latest": latest, "loss_change": latest["loss"] - first["loss"],
                      "revisit_observation_cost": sum(row["observation_cost"] for row in visits[1:]),
                      "revisit_compute_cost": sum(row["compute_cost"] for row in visits[1:])})
    completed = [task for task in tasks if task["visits"]]
    revisited = [task for task in completed if task["visits"] > 1]
    groups = {}
    for kind in ("squared_error", "pinball"):
        selected = [task for task in completed if task["latest"]["loss_kind"] == kind]
        groups[kind] = {"tasks": len(selected),
                        "first_loss": mean(task["first"]["loss"] for task in selected) if selected else None,
                        "latest_loss": mean(task["latest"]["loss"] for task in selected) if selected else None}
    return {"experiment": "post_session_prediction_and_revisit_value", "task_count": len(tasks),
            "visited_tasks": len(completed), "evaluated_visits": len(measured),
            "metrics_by_loss": groups,
            "revisits": {"tasks": len(revisited),
                         "improved": sum(task["loss_change"] < -config["minimum_improvement"] for task in revisited),
                         "worsened": sum(task["loss_change"] > config["minimum_improvement"] for task in revisited),
                         "observation_cost": sum(task["revisit_observation_cost"] for task in revisited),
                         "compute_cost": sum(task["revisit_compute_cost"] for task in revisited)},
            "tasks": tasks, "evaluation_rows": sum(row["rows"] for row in measured),
            "interpretation": "Final environment outcomes are read only after the saved session stops; they never update its scheduler, programs or knowledge."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_session(json.loads(args.checkpoint.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "tasks"}
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize actual interventions and successor calls from completed streams."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(output):
    rows = []
    for path in sorted((output / "runs").glob("*.json")):
        report = json.loads(path.read_text())
        seed, arm = path.stem.split("_", 1)
        scores = {row["task_position"]: row for row in report["evaluation"]}
        for entry in report["results"]:
            position, result = entry["task_position"], entry["research"]
            actions = result["actions"]
            purposes = Counter(action["purpose"] for action in actions if action["action"] == "acquire")
            direct = [action for action in actions if action["action"] == "test_competing_object_predictions"]
            competitions = result.get("program_competitions", [])
            rows.append({"seed": int(seed), "arm": arm, "task_position": position,
                "terminal_loss": scores.get(position, {}).get("loss"),
                "kind": result["selected_kind"], "channels": result["selected_channels"],
                "learning_costs": result["costs"], "used_sources": result.get("used_sources", []),
                "direct_competition_experiments": len(direct),
                "direct_experiment_results": Counter(action["evidence"]["status"] for action in direct),
                "boundary_experiments": purposes["resolve_structure_boundary"],
                "competition_counterexamples": sum(len(state.get("counterexamples", [])) for state in competitions),
                "object_partition_still_unresolved": not (result.get("object_research") or {}).get("complete", False)})
    result = {"tasks": rows, "arms": {}}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        result["arms"][arm] = {"tasks": len(selected),
            "evaluated": sum(row["terminal_loss"] is not None for row in selected),
            "direct_competition_experiments": sum(row["direct_competition_experiments"] for row in selected),
            "boundary_experiments": sum(row["boundary_experiments"] for row in selected),
            "competition_counterexamples": sum(row["competition_counterexamples"] for row in selected),
            "tasks_calling_prior_knowledge": sum(bool(row["used_sources"]) for row in selected)}
    (output / "behavior_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["arms"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)

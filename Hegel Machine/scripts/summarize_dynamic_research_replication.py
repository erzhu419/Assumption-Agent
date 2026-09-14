#!/usr/bin/env python3
"""Summarize declared apparatus sequences without running learners or models."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/dynamic_research_replication_20260913"
COSTS = ("observation_cost", "measurement_cost", "compute_cost")
PAIRS = (("hegel_revised", "hegel_v078"), ("hegel_revised", "code"), ("hegel_v078", "code"))


def read_reports(directory, protocol):
    reports = {}
    for seed in protocol["paired_seeds"]:
        comparison = directory / "comparisons" / f"{seed}.json"
        reports[seed] = json.loads(comparison.read_text())["arms"] if comparison.exists() else {}
        for arm, settings in protocol["arms"].items():
            path = directory / "runs" / arm / str(seed) / f"{settings['session']}_checkpoint.json"
            if arm not in reports[seed] and path.exists():
                reports[seed][arm] = json.loads(path.read_text())
    return reports


def _task(seed, arm, position, stage, entry, score, threshold):
    research = (entry.get("research", entry) if entry is not None else {})
    value = score.get("loss")
    finite_loss = value is not None and isfinite(value)
    terminal = score.get("status", "completed" if value is not None else "pending")
    evaluated = value is not None or terminal in {"missing_model", "model_failed", "not_visited"}
    solved = finite_loss and value <= threshold
    return {"seed": seed, "arm": arm, "task_position": position, "stage": stage,
        "visited": entry is not None, "terminal_status": terminal, "evaluated": evaluated,
        "solved": solved, "failed": evaluated and not solved,
        "loss": value if finite_loss else None, "model_origin": research.get("model_origin"),
        "selected_channels": research.get("selected_channels"),
        "costs": research.get("costs"),
        "queries": (entry or {}).get("actual_costs", {}).get("queries"),
        "terminal_costs": {key: score.get(key) for key in COSTS}}


def _sum_known(values):
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _aggregate(rows):
    losses = [row["loss"] for row in rows if row["loss"] is not None]
    return {"expected_tasks": len(rows), "visited_tasks": sum(row["visited"] for row in rows),
        "evaluated_tasks": sum(row["evaluated"] for row in rows),
        "solved": sum(row["solved"] for row in rows), "failed": sum(row["failed"] for row in rows),
        "measurement_free_solved": sum(row["solved"] and row["selected_channels"] == [] for row in rows),
        "selected_channels": [list(view) for view in sorted({tuple(row["selected_channels"])
                              for row in rows if row["selected_channels"] is not None})],
        "pending_terminal": sum(not row["evaluated"] for row in rows),
        "mean_loss": sum(losses) / len(rows) if rows and len(losses) == len(rows) else None,
        "max_observed_loss": max(losses) if losses else None,
        "model_origins": {name: sum(row["model_origin"] == name for row in rows)
                          for name in ("finish", "last_successful_evaluation")},
        "costed_tasks": sum(row["costs"] is not None for row in rows),
        "recorded_learning_costs": {key: _sum_known((row["costs"] or {}).get(key) for row in rows)
                                    for key in COSTS},
        "query_counted_tasks": sum(row["queries"] is not None for row in rows),
        "recorded_queries": _sum_known(row["queries"] for row in rows),
        "recorded_terminal_costs": {key: _sum_known(row["terminal_costs"][key] for row in rows)
                                    for key in COSTS}}


def _deltas(values):
    values = [value for value in values if value is not None]
    return {"pairs": len(values), "sum": sum(values) if values else None,
        "median": median(values) if values else None,
        "range": [min(values), max(values)] if values else None,
        "lower": sum(value < 0 for value in values), "equal": sum(value == 0 for value in values),
        "higher": sum(value > 0 for value in values)}


def _paired(left, right, seeds):
    pairs = []
    for seed in seeds:
        a, b = left[seed], right[seed]
        cost_complete = all(row["costed_tasks"] == row["expected_tasks"] for row in (a, b))
        evaluated = all(row["evaluated_tasks"] == row["expected_tasks"] for row in (a, b))
        delta = {key: a["recorded_learning_costs"][key] - b["recorded_learning_costs"][key]
                 if cost_complete else None for key in COSTS[:2]}
        queries = all(row["query_counted_tasks"] == row["expected_tasks"] for row in (a, b))
        pairs.append({"seed": seed, "evaluated_pair": evaluated,
            "solved_delta": a["solved"] - b["solved"] if evaluated else None,
            **delta, "query_delta": a["recorded_queries"] - b["recorded_queries"] if queries else None})
    return {"sequences": pairs, "evaluated_pairs": sum(row["evaluated_pair"] for row in pairs),
        "solved_delta": _deltas(row["solved_delta"] for row in pairs),
        **{key: _deltas(row[key] for row in pairs) for key in (*COSTS[:2], "query_delta")}}


def summarize(protocol, reports):
    seeds, stages, arms = protocol["paired_seeds"], protocol["stages"], protocol["arms"]
    tasks, sequences = [], {arm: {} for arm in arms}
    for seed in seeds:
        for arm in arms:
            report = reports.get(seed, {}).get(arm)
            state = report["session"] if report else {}
            results = {entry["task_position"]: entry for entry in state.get("results", [])}
            scores = {score["task_position"]: score for score in state.get("evaluation") or []}
            rows = [_task(seed, arm, position, stage, results.get(position), scores.get(position, {}),
                          protocol["terminal"]["solved_threshold"]) for position, stage in enumerate(stages)]
            tasks.extend(rows)
            sequences[arm][seed] = {**_aggregate(rows), "learning_status": state.get("status", "not_started"),
                "recorded_session_costs": state.get("costs"),
                "learning_wall_seconds": (report or {}).get("learning_wall_seconds")}
    totals = {}
    for arm in arms:
        totals[arm] = {**_aggregate([row for row in tasks if row["arm"] == arm]),
            "fully_solved_sequences": sum(row["solved"] == len(stages) for row in sequences[arm].values()),
            "expected_sequences": len(seeds), "sequences": sequences[arm]}
        # Session totals also retain expenditure in an interrupted active task.
        names = set().union(*(set(row["recorded_session_costs"] or {}) for row in sequences[arm].values()))
        totals[arm]["recorded_session_costs"] = {key: _sum_known((row["recorded_session_costs"] or {}).get(key)
            for row in sequences[arm].values()) for key in sorted(names)}
    by_stage = {}
    for stage in stages:
        selected = [row for row in tasks if row["stage"] == stage]
        per_seed = {arm: {seed: _aggregate([row for row in selected if row["arm"] == arm and row["seed"] == seed])
                          for seed in seeds} for arm in arms}
        by_stage[stage] = {"arms": {arm: _aggregate([row for row in selected if row["arm"] == arm]) for arm in arms},
            "paired": {f"{a}_minus_{b}": _paired(per_seed[a], per_seed[b], seeds) for a, b in PAIRS}}
    return {"statistical_unit": protocol["statistical_unit"], "development_evidence": protocol["development_evidence"],
        "paired_seeds": seeds, "arms": totals, "stages": by_stage,
        "paired": {f"{a}_minus_{b}": _paired(sequences[a], sequences[b], seeds) for a, b in PAIRS},
        "compute_note": protocol["compute_note"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    protocol = json.loads((args.output / "protocol.json").read_text())
    result = summarize(protocol, read_reports(args.output, protocol))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({arm: {key: values[key] for key in ("solved", "failed", "pending_terminal",
        "fully_solved_sequences", "recorded_session_costs", "recorded_queries")} for arm, values in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()

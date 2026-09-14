#!/usr/bin/env python3
"""Score completed measurement studies after the same paid apparatus operation.

The original fresh-apparatus scores remain unchanged. This supplement evaluates
every saved model after 64 controls-only operating rows, so a shorter research
visit cannot receive an earlier physical regime than its comparator.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.measurement_lab import make_lab
from hegel_machine.cognition.models import Probe


PROTOCOL = {
    "experiment": "matched_operation_measurement_evaluation",
    "warmup_rows": 64,
    "warmup_channels": [],
    "warmup_control": 0.0,
    "evaluation": "Use each original study's fixed independent seed, length and trajectories after the same 64 operating rows.",
    "timing": "Evaluate saved models only after both studies have completed every declared learning task. No rows or scores return to learning.",
    "accounting": "Warmup observations, scoring observations, scoring measurements and scored program rows are additional evaluation costs, separate from research costs.",
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def score_model(result, spec):
    episode = result.get("episode")
    costs = {key: 0 for key in ("warmup_observations", "scoring_observations",
                                "scoring_measurements", "scoring_compute")}
    score = {"task_position": result["task_position"], "task_index": result["task_index"],
             "seed": result["seed"], "mse": None, "costs": costs}
    if episode is None:
        return {**score, "error": "no_completed_model"}
    with make_lab(result["task_index"], result["seed"], backend=spec["backend"]) as lab:
        warmup = lab.acquire(Probe("evaluation_operational_warmup",
            tuple({"u": PROTOCOL["warmup_control"]} for _ in range(PROTOCOL["warmup_rows"]))),
            PROTOCOL["warmup_channels"])
        options = spec["evaluation"]
        rows = lab.evaluation_rows(result["selected_channels"],
            seed=result["seed"] + options["seed_offset"], length=options["length"],
            trajectories=options["trajectories"])
        measurement_unit = sum(channel.cost_per_row for channel in lab.channels
                               if channel.name in result["selected_channels"])
    costs.update(warmup_observations=warmup.observation_cost, scoring_observations=len(rows),
                 scoring_measurements=len(rows) * measurement_unit, scoring_compute=len(rows))
    score["mse"] = loss(Program.from_dict(episode["program"]), rows, episode["fallback"], .5)
    return score


def summarize(rows):
    scores = [row["mse"] for row in rows if row["mse"] is not None]
    return {"tasks": len(rows), "completed_models": len(scores),
            "mean_mse": mean(scores) if scores else None,
            "maximum_mse": max(scores) if scores else None,
            "evaluation_costs": {key: sum(row["costs"][key] for row in rows)
                                 for key in rows[0]["costs"]} if rows else {}}


def evaluate_study(source):
    spec, cohorts = source["protocol"], {}
    aggregate = {arm: [] for arm in spec["arms"]}
    for cohort, expected in spec["cohorts"].items():
        arms = {}
        for arm in spec["arms"]:
            completed = source["cohorts"][cohort]["arms"][arm]["results"]
            if len(completed) != len(expected):
                raise ValueError("Finish every declared learning task before evaluation")
            rows = [score_model(result, spec) for result in completed]
            arms[arm] = {**summarize(rows), "results": rows}
            aggregate[arm].extend(rows)
        cohorts[cohort] = {"arms": arms,
            "prefixes": [{"tasks": length, **{arm: summarize(arms[arm]["results"][:length])
                         for arm in spec["arms"]}} for length in spec["prefixes"]]}
    return {"implementation_version": source["implementation_version"], "cohorts": cohorts,
            "aggregate": {arm: summarize(rows) for arm, rows in aggregate.items()},
            "research_costs": {arm: deepcopy(source["aggregate"][arm]["costs"]) for arm in spec["arms"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = {name: json.loads(path.read_text()) for name, path in
               (("current", args.current), ("baseline", args.baseline))}
    if any(source["status"] != "completed" for source in sources.values()):
        raise ValueError("Finish both studies before evaluation")
    if sources["current"]["protocol"] != sources["baseline"]["protocol"]:
        raise ValueError("Evaluate implementations under the same learning protocol")
    result = {"status": "completed", "protocol": PROTOCOL,
              "sources": {"current": str(args.current.resolve()), "baseline": str(args.baseline.resolve())},
              **{name: evaluate_study(source) for name, source in sources.items()}}
    result["current_minus_baseline"] = {arm: {
        "mean_mse": result["current"]["aggregate"][arm]["mean_mse"] - result["baseline"]["aggregate"][arm]["mean_mse"],
        "research_costs": {key: result["current"]["research_costs"][arm][key]
                           - result["baseline"]["research_costs"][arm][key]
                           for key in result["current"]["research_costs"][arm]}}
        for arm in sources["current"]["protocol"]["arms"]}
    write_json(args.output, result)
    print(json.dumps({"current": result["current"]["aggregate"],
                     "baseline": result["baseline"]["aggregate"],
                     "current_minus_baseline": result["current_minus_baseline"]}, indent=2))


if __name__ == "__main__":
    main()

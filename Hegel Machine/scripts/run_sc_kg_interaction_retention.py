#!/usr/bin/env python3
"""Replay the thirteen old re-entry/control tasks with frozen interaction research."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path

import run_sc_kg_reentry_value_study as replay


study = replay.study
BASELINE, METHOD = "retention_baseline_v052", "retention_interaction_candidate"


def prepare(output):
    original_path = study.ROOT / "artifacts/sc_kg_reentry_value_v053_20260911/protocol.json"
    original = study.read(original_path)
    selected = original["development_cases"]
    if (len(selected), len(original["affected_cases"]), len(original["unchanged_controls"])) != (13, 8, 5):
        raise ValueError("this retention comparison requires the original eight entries and five controls")
    if set(selected) != set(replay.DEVELOPMENT):
        raise ValueError("the original thirteen-case roster differs")
    cases = [deepcopy(next(case for case in original["cases"] if case["case_id"] == key))
             for key in selected]
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    spec.update(experiment="interaction_research_retention_on_old_real_trajectories", cases=cases,
        arms=[BASELINE, METHOD], development_cases=selected,
        affected_cases=original["affected_cases"], unchanged_controls=original["unchanged_controls"],
        frozen_baseline_source=original["frozen_baseline_source"],
        frozen_method_source=str((study.ROOT / "artifacts/sc_kg_unknown_revision_20260911/interaction_source").resolve()),
        source_reentry_study=original["source_reentry_study"], source_roster=str(original_path.resolve()),
        reused_cases=[], fresh_seeds=[], fresh_cases=[],
        expected_new_optimizer_runs=13, expected_new_online_calls=416,
        expected_new_initial_calls=0, expected_new_evaluation_calls=0, expected_new_physical_calls=416,
        method="Use the already frozen interaction package unchanged: control-group-held-out objective "
            "cross terms, callable objective programs, and charged joint-optimum proposals. Reuse actual "
            "v0.52 comparison results, all initial32 vectors, and all final evaluation vectors. Only "
            "the new method executes13 optimization runs of32 online observations.",
        selection="The original v0.53 development roster: all eight v0.52 re-entry trajectories and "
            "their five predeclared controls. Do not select or remove cases based on interaction outcomes.",
        limitations="A targeted thirteen-case retention comparison, not complete coverage of the old544 "
            "tasks. Historical wall times were measured in different runs; controller computation and "
            "native KG timing remain separate. Missing historical measured-objective labels stay missing.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved retention protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def run(spec, output):
    replay.BASELINE, replay.METHOD = BASELINE, METHOD
    baseline, old_refs = replay.references(Path(spec["source_reentry_study"]), spec["development_cases"])
    modules = dict(zip((BASELINE, METHOD), replay.validation.arm_modules(spec).values()))
    baseline_module = modules[BASELINE][0]
    study.Program = importlib.import_module(f"{baseline_module.__package__}.language").Program
    study.SCKGEnvironment = baseline_module.SCKGEnvironment
    study.controlled_problem = replay.validation.active.framework.goal.controlled_problem
    research = replay.retention.measurements(spec, output, spec["cases"])
    study.ARMS = (METHOD,)
    actual = study.optimize(spec, output, spec["cases"], research, arm_modules=modules)
    study.evaluate(spec, output, spec["cases"], actual,
        environment_factory=replay.validation.active.framework.goal.evaluation_environment)
    current, refs = {}, []
    for row in actual:
        key = row["case_id"]
        path = output / "runs" / f"{key}_{METHOD}.json"
        study.write(path, row)
        current[key] = row
        refs.extend(({**old_refs[key], "arm": BASELINE, "run_origin": "reused_actual_v052_reference"},
                     {"case_id": key, "arm": METHOD, "case_record": str(path.resolve()),
                      "run_origin": "actual_interaction_retention"}))
    replay.summarize(spec, output, spec["cases"], baseline, current, refs, actual, "replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    spec = prepare(args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "cases": len(spec["cases"]),
                          "new_online_calls": 416, "new_initial_calls": 0, "new_evaluation_calls": 0}))
    else:
        run(spec, args.output)


if __name__ == "__main__":
    main()

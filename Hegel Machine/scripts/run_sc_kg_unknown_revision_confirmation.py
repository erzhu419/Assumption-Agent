#!/usr/bin/env python3
"""Confirm the frozen interaction method and retain the separate old pilot."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import run_sc_kg_unknown_revision_study as revision
import run_sc_kg_unknown_optimum_study as pilot


SEEDS = (2043012523, 2043112549, 2043212573)
study = revision.study
DEVELOPMENT = study.ROOT / "artifacts/sc_kg_unknown_revision_20260911"
PILOT = study.ROOT / "artifacts/sc_kg_unknown_optimum_pilot_20260911"


def save_protocol(output, spec):
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved confirmation protocol differs")
    if not path.exists():
        study.write(path, spec)


def prepare_confirmation(output):
    original = study.read(DEVELOPMENT / "protocol.json")
    spec = deepcopy(original)
    spec["cases"] = []
    for index, (old_seed, seed) in enumerate(zip(original["seeds"], SEEDS), 1):
        for old_case in original["cases"]:
            if old_case["task"]["seed"] != old_seed:
                continue
            case = deepcopy(old_case)
            case["task"]["seed"] = seed
            case.update(case_id=f"unknown_confirmation_{index}_{case['profile']}",
                research_id=f"unknown_confirmation_{index}_{case['task']['problem']}",
                split="confirmation", cohort="fresh9")
            spec["cases"].append(case)
    for key in ("additional_arm", "additional_interaction_optimizer_runs", "additional_interaction_online_calls"):
        spec.pop(key)
    spec.update(experiment="independent_unknown_objective_revision_confirmation", seeds=list(SEEDS),
        arms=[revision.NATIVE, revision.HEGEL, revision.INTERACTION],
        development_reference=str((DEVELOPMENT / "result_brief.json").resolve()),
        expected_all_physical_calls=2208,
        comparison="Native SC-KG, frozen0.52 and the fixed interaction method share each actually paid32 "
            "initial vectors, followed by32 new calls per arm. Initialization research uses frozen0.52. "
            "The initial-goal-frozen ablation is retained in development and is not rerun here.",
        additional_method="The interaction source is exactly the completed development snapshot. No "
            "model, scoring, rank, adoption or budget changes are made after observing these new seeds.",
        accounting="Six unique32-vector initial prefixes cost192; nine tasks times three arms cost864 "
            "online calls; nine final128-vector evaluations cost1152. Total2208 new physical calls. "
            "No development controls are rerun. Research compute and native KG work remain separate.",
        limitations="This is independent confirmation on three new seeds and the same three synthetic "
            "profiles, using conditional shared initialization. All seeds were fixed before acquisition. "
            "The task family was constructed during development; no broader agent claim follows.")
    save_protocol(output, spec)
    return spec


def confirm(output):
    spec = prepare_confirmation(output)
    modules = revision.load_hegel(spec, (revision.HEGEL, revision.INTERACTION))
    research = study.acquire_research(spec, output, spec["cases"])
    rows = study.optimize(spec, output, spec["cases"], research, arm_modules=modules)
    rows.extend(revision.run_native(spec, output, case, research[case["research_id"]]) for case in spec["cases"])
    revision.summarize(spec, output, research, rows, "confirmation")


def retain_pilot(output):
    original = study.read(PILOT / "protocol.json")
    development = study.read(DEVELOPMENT / "protocol.json")
    spec = deepcopy(original)
    spec.update(experiment="interaction_method_retains_old_separable_goal_pilot",
        arms=[pilot.NATIVE, pilot.HEGEL, revision.INTERACTION],
        frozen_hegel_source=development["frozen_interaction_source"],
        source_pilot=str(PILOT.resolve()), expected_optimizer_runs=3,
        expected_initial_calls=0, expected_online_calls=96, expected_evaluation_calls=0,
        expected_physical_calls=96,
        accounting="Append only the interaction method to the three completed separable-goal pilot "
            "tasks: three32-call online runs cost96 physical calls. Reuse every existing initialization, "
            "evaluator vector and native/0.52 result. These cases are retention checks, not new seeds.")
    save_protocol(output, spec)
    for source in (PILOT / "evaluation").glob("*.json"):
        target = output / "evaluation" / source.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    old_hegel = pilot.HEGEL
    pilot.HEGEL = revision.INTERACTION
    modules = pilot.load_hegel(spec)
    research = {case["research_id"]: study.read(PILOT / "research" / f"{case['research_id']}.json")
                for case in spec["cases"]}
    rows = study.optimize(spec, output, spec["cases"], research, arm_modules=modules)
    study.evaluate(spec, output, spec["cases"], rows, environment_factory=pilot.task.evaluation_environment)
    comparisons = []
    for row in rows:
        old = {arm: study.read(PILOT / "runs" / f"{row['case_id']}_{arm}.json")
               for arm in (pilot.NATIVE, old_hegel)}
        all_rows = {**old, revision.INTERACTION: row}
        comparisons.append({"case_id": row["case_id"], "arms": {arm: {
            "objective": result["final"]["true_objective"],
            "regret": result["final"]["true_objective"] - .35,
            "feasible": result["final"]["true_feasible"],
            "recommendation": result["final"]["x_recommended"],
            "objective_mse": result["generalization"]["objective_gp_true_objective_means_mse"],
            "constraint_mse": result["generalization"]["active_gp_true_constraint_means_mse"]}
            for arm, result in all_rows.items()}})
        study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    brief = {"protocol": str((output / "protocol.json").resolve()), "cases": 3,
        "comparisons": comparisons, "research_behavior": [revision.behavior(row) for row in rows],
        "costs": {"initial_physical_calls": 0, "online_physical_calls": sum(
            row["costs"]["optimizer_new_simulator_calls"] for row in rows),
            "evaluation_physical_calls": 0, "total_physical_calls": 96,
            "objective_compute": sum((row.get("objective_research") or {}).get("compute_cost", 0) for row in rows),
            "native_kg_seconds": sum(item.get("t_kg_compute", 0.) for row in rows
                for item in row["costs"]["exact_kg_stages"])}}
    study.write(output / "result_brief.json", brief)
    print(json.dumps(brief), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("confirmation", "pilot-retention"), required=True)
    args = parser.parse_args()
    (confirm if args.mode == "confirmation" else retain_pilot)(args.output)


if __name__ == "__main__":
    main()

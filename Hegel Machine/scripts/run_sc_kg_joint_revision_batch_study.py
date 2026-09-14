#!/usr/bin/env python3
"""Frozen batch-evidence follow-up with reused development data and fresh seeds."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import run_sc_kg_joint_revision_study as study


CONFIRMATION_SEEDS = (1243010117, 1243110131, 1243210153)


def protocol(source, output):
    previous = study.read(source / "protocol.json")
    spec = deepcopy(previous)
    spec["experiment"] = "batch_common_misfit_revision_confirmation"
    spec["development_source"] = str(source.resolve())
    spec["first_controller_source"] = str((output / "first_controller_source").resolve())
    for filename in ("joint_model_revision.py", "sc_kg_environment.py", "prior_adoption.py"):
        if not (Path(spec["first_controller_source"]) / filename).exists():
            raise FileNotFoundError("the first controller source must be saved before the batch change")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
        case["previous_case_id"] = case["case_id"]
    for index, seed in enumerate(CONFIRMATION_SEEDS, 1):
        for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                 ("stable_state_policy", "state_policy_rzdt1"),
                                 ("stable_rzdt2", "rzdt2")):
            spec["cases"].append({"case_id": f"confirmation_{index}_{profile}_noise_0.03",
                "split": "confirmation", "profile": profile,
                "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": 0.03},
                "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                "research_id": f"confirmation_{index}_{problem}_noise_0.03"})
    spec["revision"]["evidence"] = (
        "Keep the original four-observation batch size and nine-times-frozen-reference threshold. "
        "Nonoverlapping actual measurement batches reopen research when both models' batch MSE "
        "exceed the threshold and at least two individual rows exceed it for both models. "
        "The new scope starts at the earliest such counterexample and retains every subsequently "
        "paid row. Eight coverage calls and four fresh validation calls remain inside N64. "
        "No change labels, times, or truth values reach the controller.")
    spec["scope"] = (
        "All 19 previously evaluated cases are development evidence. Reuse their actual initial "
        "research vectors, supplemental initial data, frozen-0.31 baseline results, and independent "
        "evaluation measurements. Rerun only the new batch-revision arm. Three fresh seeds at "
        "noise0.03 each supply mechanism change, stable StatePolicy, and stable RZDT2: nine "
        "confirmation cases with both arms. Run the three old noisy changes first, then the "
        "remaining development roster and frozen confirmation cases without threshold tuning.")
    spec["reuse_accounting"] = (
        "Historical initial and baseline/evaluation simulator calls are retained as paid evidence "
        "but reported separately from this follow-up's actual new calls. The first controller's "
        "original results remain an explicit third historical development reference; they are "
        "never relabeled as results of the new controller.")
    return spec


def prepare_reuse(spec, output):
    source = Path(spec["development_source"])
    copied_research = set()
    for case in spec["cases"]:
        if case["split"] != "development":
            continue
        key = case["research_id"]
        if key not in copied_research:
            path = output / "research" / f"{key}.json"
            if not path.exists():
                saved = study.read(source / "research" / f"{key}.json")
                saved["original_research_origin"] = saved["research_origin"]
                saved["research_origin"] = str((source / "research" / f"{key}.json").resolve())
                saved["actual_new_research_calls"] = 0
                saved["actual_new_initial_calls"] = 0
                study.write(path, saved)
            copied_research.add(key)
        path = output / "runs" / f"{case['case_id']}_frozen_v031_auto.json"
        if not path.exists():
            original_path = source / "runs" / path.name
            row = study.read(original_path)
            row.update(split="development", run_origin="historical_frozen_v031",
                       original_result=str(original_path.resolve()))
            study.write(path, row)
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for path in (source / "evaluation").glob("*.json"):
        destination = evaluation_dir / path.name
        if not destination.exists():
            shutil.copyfile(path, destination)
    amendment_path = output / "initial_data_completion_amendment.json"
    if not amendment_path.exists():
        shutil.copyfile(source / amendment_path.name, amendment_path)


def summarize(spec, output, cases, research, results, evaluations, phase):
    study.summarize(spec, output, cases, research, results, evaluations)
    path = output / ("summary.json" if len(cases) == len(spec["cases"]) else "development_summary.json")
    summary = study.read(path)
    source = Path(spec["development_source"])
    previous = study.read(source / "summary.json")
    old_results = {row["case_id"]: row for row in previous["results"] if row["arm"] == "joint_revision"}
    references = []
    for row in results:
        if row["arm"] != "joint_revision" or row["split"] != "development":
            continue
        first = old_results[row["case_id"]]
        references.append({"case_id": row["case_id"],
            "first_controller_objective": first["final"]["true_objective"],
            "batch_controller_objective": row["final"]["true_objective"],
            "first_controller_program_mse": first["generalization"]["program_true_constraint_means_mse"],
            "batch_controller_program_mse": row["generalization"]["program_true_constraint_means_mse"],
            "first_controller_gp_mse": first["generalization"]["active_gp_true_constraint_means_mse"],
            "batch_controller_gp_mse": row["generalization"]["active_gp_true_constraint_means_mse"],
            "first_controller_replacements": len(first["scope_revisions"]),
            "batch_controller_replacements": len(row["scope_revisions"])})
    summary["first_controller_development_reference"] = references
    summary["actual_new_optimizer_calls"] = sum(row["costs"]["optimizer_new_simulator_calls"]
        for row in results if row.get("run_origin") != "historical_frozen_v031")
    summary["historically_reused_optimizer_calls"] = sum(row["costs"]["optimizer_new_simulator_calls"]
        for row in results if row.get("run_origin") == "historical_frozen_v031")
    summary["actual_supplemental_initial_calls"] = sum(row.get("supplemental_initial_calls", 0)
        for row in research.values() if row["research_origin"] == "new_public_measurements")
    summary["historically_reused_initial_calls"] = sum(row["result"]["costs"]["observation_cost"]
        + row.get("supplemental_initial_calls", 0) for row in research.values()
        if row["research_origin"] != "new_public_measurements")
    summary["independent_evaluation_calls"] = sum(row["simulator_calls"] for key, row in evaluations.items()
        if not (source / "evaluation" / f"{key}.json").exists())
    summary["historically_reused_evaluation_calls"] = sum(row["simulator_calls"] for key, row in evaluations.items()
        if (source / "evaluation" / f"{key}.json").exists())
    destination = output / ("summary.json" if phase == "all" else f"{phase}_summary.json")
    study.write(destination, summary)
    if destination != path:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "confirmation", "all"), default="plan")
    args = parser.parse_args()
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen batch-evidence protocol differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 19,
            "confirmation_cases": 9, "new_optimizer_runs": 19 + 9 * 2}))
        return
    cases = spec["cases"]
    if args.phase == "replay":
        cases = [case for case in cases if case["split"] == "development"
            and case["profile"] == "mechanism_change" and case["task"]["noise"] == 0.03]
    elif args.phase == "confirmation":
        cases = [case for case in cases if case["split"] == "confirmation"]
    prepare_reuse(spec, args.output)
    research = study.acquire_research(spec, args.output, cases)
    results = study.optimize(spec, args.output, cases, research)
    evaluations = study.evaluate(spec, args.output, cases, results)
    summarize(spec, args.output, cases, research, results, evaluations, args.phase)


if __name__ == "__main__":
    main()

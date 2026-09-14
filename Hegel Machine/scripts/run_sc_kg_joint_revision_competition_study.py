#!/usr/bin/env python3
"""Prospective candidate-competition comparison against the frozen 0.32 system."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys
import types

import run_sc_kg_joint_revision_study as study


CONFIRMATION_SEEDS = (1343010161, 1343110177, 1343210189)
ARMS = ("frozen_v032_batch", "joint_revision")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="joint_revision_candidate_competition",
        development_source=str(source.resolve()),
        frozen_baseline_source=str((output / "baseline_source_v032").resolve()),
        arms=list(ARMS))
    spec["baseline_revision_evidence"] = spec["revision"].pop("evidence")
    for name in ("joint_model_revision.py", "sc_kg_environment.py", "prior_adoption.py", "meta_prior_claims.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError("the actual 0.32 source must be frozen before changing the method")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    for index, seed in enumerate(CONFIRMATION_SEEDS, 1):
        for noise in (0.0, 0.03):
            for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                     ("stable_state_policy", "state_policy_rzdt1"),
                                     ("stable_rzdt2", "rzdt2")):
                spec["cases"].append({"case_id": f"new_confirmation_{index}_{profile}_noise_{noise:.2f}",
                    "split": "confirmation", "profile": profile,
                    "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": noise},
                    "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                    "research_id": f"new_confirmation_{index}_{problem}_noise_{noise:.2f}"})
    spec["baseline_scope"] = (
        "For the 28 development cases, the actual completed 0.32 batch-controller arm is the "
        "historical baseline. For each fresh confirmation case, load and execute the saved "
        "0.32 controller, SC adapter, prior-adoption policy, and claim compiler. This baseline "
        "includes actual hypothesis reopening and scope rebasing, not only prior adoption.")
    spec["scope"] = (
        "The previous 28 evaluated cases are development. Three new seeds, each at noise0 and "
        "noise0.03 with mechanism change, stable StatePolicy and stable RZDT2, make 18 frozen "
        "confirmation cases. Every arm keeps 32 actual initial records and 32 online calls. "
        "Both arms use identical initial data and independent final evaluation controls. "
        "First replay the six noisy development changes, then execute the remaining roster. "
        "Keep failed competition, late reopening, false replacement, and infeasible outputs.")
    spec["method_frozen_before_execution"] = True
    spec["new_method"] = {
        "detection": {"window": "sliding_four_actual_rows", "batch_size": 4,
            "reference_multiplier": 9, "minimum_individual_joint_outliers": 2},
        "research_budget": "Coverage is the minimum of the original dimension-based coverage count and remaining_calls minus four. Retain four future validation calls. Previously paid new-scope rows also train candidates; unresolved if this evidence cannot fit an executable candidate.",
        "candidates": "Add a quadratic candidate for each individual public coordinate, traversing all coordinates without a target-axis or family hint. These compete with existing affine, additive, coupled, projection, and boundary candidates.",
        "ranking": "Paid leave-one-out training predictions rank the candidates; retain at most three for future competition.",
        "validation_controls": "Score disagreement among the retained candidates over 64 public control candidates plus boundary controls. Select four distinct points. Freeze every candidate's predictions before observing the same four actual outputs.",
        "acceptance": "Choose by future prediction error. When future errors are indistinguishable within paired uncertainty, choose by training leave-one-out evidence; program length only breaks a leave-one-out tie. Existing training MSE and future MSE must both satisfy the unchanged nine-times-frozen-reference threshold, and future error must improve on the previous models.",
        "total_calls_per_arm": 64, "initial_calls_per_arm": 32,
        "compute": "Charge all candidate fitting, leave-one-out fitting/evaluation, and control-pool prediction work. No extra simulator calls beyond N64."}
    spec["research_initialization"] = "Fresh initial data use MetaPriorResearch with the frozen0.32 claim compiler; all initial vectors and the resulting candidate are shared by both optimizer arms. Historical initial data are reused unchanged."
    spec["reuse_accounting"] = (
        "Reuse all existing actual research/vector records and independent evaluations. "
        "Only rerun the new arm on development cases. Historical baseline calls, reused "
        "initial data and evaluation calls are distinct from all new measured calls. "
        "The public stratified completion rule still supplies 32 initial records when "
        "MetaPriorResearch legitimately finishes earlier; its original result is unchanged.")
    return spec


def prepare_reuse(spec, output):
    source = Path(spec["development_source"])
    for case in spec["cases"]:
        if case["split"] != "development":
            continue
        key = case["research_id"]
        path = output / "research" / f"{key}.json"
        if not path.exists():
            saved = study.read(source / "research" / f"{key}.json")
            saved.update(research_origin=str((source / "research" / f"{key}.json").resolve()),
                         actual_new_research_calls=0, actual_new_initial_calls=0)
            study.write(path, saved)
        path = output / "runs" / f"{case['case_id']}_{ARMS[0]}.json"
        if not path.exists():
            original_path = source / "runs" / f"{case['case_id']}_joint_revision.json"
            row = study.read(original_path)
            row.update(split="development", arm=ARMS[0], run_origin="historical_v032_batch",
                       original_result=str(original_path.resolve()))
            study.write(path, row)
    folder = output / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    for path in (source / "evaluation").glob("*.json"):
        if not (folder / path.name).exists():
            shutil.copyfile(path, folder / path.name)
    path = output / "initial_data_completion_amendment.json"
    if not path.exists():
        shutil.copyfile(source / path.name, path)


def arm_modules(spec):
    # Relative imports first resolve the four frozen modules; unchanged common
    # language/data modules remain available from the installed source package.
    package_name = "hegel_machine.cognition._competition_baseline_v032"
    package = types.ModuleType(package_name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[package_name] = package
    baseline = {name: importlib.import_module(f"{package_name}.{name}") for name in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
                      baseline["joint_model_revision"].JointModelRevision),
            ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}


def summarize(spec, output, cases, research, results, evaluations, phase):
    study.summarize(spec, output, cases, research, results, evaluations)
    path = output / ("summary.json" if len(cases) == len(spec["cases"]) else "development_summary.json")
    summary = study.read(path)
    source = Path(spec["development_source"])
    summary["actual_new_optimizer_calls"] = sum(row["costs"]["optimizer_new_simulator_calls"]
        for row in results if row.get("run_origin") != "historical_v032_batch")
    summary["historically_reused_optimizer_calls"] = sum(row["costs"]["optimizer_new_simulator_calls"]
        for row in results if row.get("run_origin") == "historical_v032_batch")
    summary["actual_supplemental_initial_calls"] = sum(row.get("supplemental_initial_calls", 0)
        for row in research.values() if row["research_origin"] == "new_public_measurements")
    summary["historically_reused_initial_calls"] = sum(row["result"]["costs"]["observation_cost"]
        + row.get("supplemental_initial_calls", 0) for row in research.values()
        if row["research_origin"] != "new_public_measurements")
    summary["independent_evaluation_calls"] = sum(row["simulator_calls"] for key, row in evaluations.items()
        if not (source / "evaluation" / f"{key}.json").exists())
    summary["historically_reused_evaluation_calls"] = sum(row["simulator_calls"] for key, row in evaluations.items()
        if (source / "evaluation" / f"{key}.json").exists())
    summary["revision_sequences"] = [{"case_id": row["case_id"], "arm": row["arm"],
        "stages": [{key: event.get(key) for key in ("action", "stage", "family_id", "kind", "supported", "reason")}
            for event in (row.get("recovery") or {}).get("events", [])
            if event.get("action") != "observe_revision"],
        "after_adoption_actual_forecast": row["phase_prediction_error"].get("after_first_accepted_replacement")}
        for row in summary["results"]]
    destination = output / ("summary.json" if phase == "all" else f"{phase}_summary.json")
    study.write(destination, summary)
    if destination != path:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "all"), default="plan")
    args = parser.parse_args()
    study.ARMS = ARMS
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen candidate-competition roster differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 28,
            "confirmation_cases": 18, "new_optimizer_runs": 28 + 18 * 2}))
        return
    cases = spec["cases"]
    if args.phase == "replay":
        cases = [case for case in cases if case["split"] == "development"
            and case["profile"] == "mechanism_change" and case["task"]["noise"] == 0.03]
    elif args.phase == "development":
        cases = [case for case in cases if case["split"] == "development"]
    elif args.phase == "confirmation":
        cases = [case for case in cases if case["split"] == "confirmation"]
    prepare_reuse(spec, args.output)
    modules = arm_modules(spec)
    study.MetaPriorResearch = importlib.import_module(
        "hegel_machine.cognition._competition_baseline_v032.meta_prior_research").MetaPriorResearch
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results)
    summarize(spec, args.output, cases, research, results, evaluations, args.phase)


if __name__ == "__main__":
    main()

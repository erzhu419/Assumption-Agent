#!/usr/bin/env python3
"""Paid goal exploration and noisy recommendation, independent of world revision."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import run_sc_kg_observation_repair_study as repair

study, accounting = repair.study, repair.accounting
SEEDS = (1743010403, 1743110417, 1743210439)
ARMS = ("frozen_v036_observation_repair", "objective_exploration")
OBJECTIVE_CHANGE_PROBLEM = "objective_rzdt2_constraint_state_policy"
REPLAY_CASES = ("single_outlier_1_state_policy_noise_0.03",
                "decision_single_outlier_2_state_policy_noise_0.03",
                "repair_single_outlier_3_state_policy_noise_0.03")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="independent_goal_exploration_and_noisy_recommendation",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v036").resolve()),
        historical_baseline_arm="observation_repair",
        historical_baseline_origin="historical_v036_observation_repair")
    spec["baseline_method"] = spec.pop("new_method")
    for name in repair.BASELINE_MODULES:
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError(f"the pre-edit 0.36 baseline is required: {name}")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    for index, seed in enumerate(SEEDS, 1):
        for noise in (0., .03):
            for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                     ("stable_state_policy", "state_policy_rzdt1"),
                                     ("stable_rzdt2", "rzdt2")):
                spec["cases"].append({"case_id": f"objective_exploration_{index}_{profile}_noise_{noise:.2f}",
                    "split": "confirmation", "profile": profile,
                    "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": noise},
                    "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                    "research_id": f"objective_exploration_{index}_{problem}_noise_{noise:.2f}"})
        common = {"split": "stress",
            "task": {"problem": "state_policy_rzdt1", "dimension": 3, "seed": seed, "noise": .03},
            "research_id": f"objective_exploration_{index}_state_policy_rzdt1_noise_0.03"}
        spec["cases"].append({**deepcopy(common),
            "case_id": f"objective_single_outlier_{index}_state_policy_noise_0.03",
            "profile": "single_observation_outlier", "next_problem": "state_policy_rzdt1",
            "observation_outlier": {"global_stage": 45, "online_call": 13, "constraint_delta": 1.0}})
        spec["cases"].append({**deepcopy(common),
            "case_id": f"objective_outlier_then_change_{index}_noise_0.03",
            "profile": "outlier_then_change", "next_problem": "rzdt2",
            "change_after_new_calls": 12,
            "observation_outlier": {"global_stage": 41, "online_call": 9, "constraint_delta": 1.0}})
        spec["cases"].append({**deepcopy(common),
            "case_id": f"objective_only_change_{index}_noise_0.03",
            "profile": "objective_only_change", "next_problem": OBJECTIVE_CHANGE_PROBLEM,
            "change_after_new_calls": 12})
    spec["new_method"] = study.read(output / "method.json")
    spec["baseline_scope"] = (
        "The 112 development baselines are the actual completed 0.36 observation-repair runs, "
        "including every failure and regression. Fresh baseline runs load the six saved0.36 "
        "runtime modules, preserving paid confirmation, constraint-only observation repair, "
        "objective-aware revision experiments, and original terminal recommendation.")
    spec["scope"] = (
        "All previous112 cases are development. Three previously unused seeds produce18 "
        "ordinary confirmation cases (two noise levels, change and both stable mechanisms) "
        "and nine separate stress cases (one observation outlier, outlier followed by change, "
        "and objective-only mechanism change with an unchanged constraint). "
        "Keep all cases and costs, regardless of whether their outcomes improve.")
    spec["research_initialization"] = (
        "Fresh initialization uses the saved0.36 claim compiler and unchanged public stratified "
        "completion to32 paid full vectors. Both arms share identical initial observations and "
        "initial programs. The stress cases share their ordinary noisy StatePolicy initialization.")
    spec["reuse_accounting"] = (
        "Reuse112 actual baseline runs and all previously paid initial/evaluation observations. "
        "Run the new arm on112 development cases and both arms on27 fresh cases:166 new "
        "optimizer runs or5312 online calls. Every arm remains32 initial plus32 online calls; "
        "goal experiments and recommendation confirmation consume this original budget.")
    spec["objective_only_control"] = (
        "Starting at global stage45, objectives f1/f2 use the upstream RZDT2 mean mechanism, "
        "while f3 retains StatePolicyRZDT1. A script-owned StatePolicyRZDT1 subclass composes "
        "the upstream deterministic mean methods and inherits the same TestProblem.simulate: "
        "one three-output Gaussian draw is one actual observation, with no uncharged second "
        "simulator query. Final independent evaluation uses this same hybrid mechanism. "
        "The controller receives ordinary controls and measured outputs, without change labels.")
    return spec


def objective_changed_problem(environment):
    from problems.rzdt import RZDT2, StatePolicyRZDT1

    class ObjectiveRZDT2ConstraintStatePolicy(StatePolicyRZDT1):
        problem_name = OBJECTIVE_CHANGE_PROBLEM

        def true_objectives(self, point):
            objectives = RZDT2.true_objectives(self, point)
            constraint = StatePolicyRZDT1.true_objectives(self, point)[2]
            return objectives[0], objectives[1], constraint

    return ObjectiveRZDT2ConstraintStatePolicy(d=environment._problem.d, L=100,
        sigma=environment._settings["noise"], heteroscedastic=False)


def controlled_problem(base_class, environment, program, case):
    if case["profile"] != "objective_only_change":
        return repair.controlled_problem(base_class, environment, program, case)
    scheduled = {**case, "profile": "mechanism_change"}
    problem = repair.controlled_problem(base_class, environment, program, scheduled)
    problem.next_problem = objective_changed_problem(environment)
    return problem


def evaluation_environment(project_root, **task):
    hybrid = task["problem"] == OBJECTIVE_CHANGE_PROBLEM
    settings = {**task, "problem": "state_policy_rzdt1"} if hybrid else task
    environment = study.SCKGEnvironment(project_root, **settings)
    if hybrid:
        environment._problem = objective_changed_problem(environment)
    return environment


def arm_modules(spec):
    package_name = "hegel_machine.cognition._objective_exploration_baseline_v036"
    package = types.ModuleType(package_name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[package_name] = package
    baseline = {name: importlib.import_module(f"{package_name}.{name}") for name in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    study.MetaPriorResearch = importlib.import_module(f"{package_name}.meta_prior_research").MetaPriorResearch
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
                      baseline["joint_model_revision"].JointModelRevision),
            ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}


def enrich_objective_summary(spec, output, phase):
    stem = "" if phase == "all" else f"{phase}_"
    summary = study.read(output / f"{stem}summary.json")
    compact = study.read(output / f"{stem}compact_summary.json")

    def metrics(rows):
        outcomes = [outcome for row in rows for outcome in (row.get("objective_research") or {}).get("outcomes", [])]
        improvements = [item["measured_improvement"] for item in outcomes if "measured_improvement" in item]
        fields = ("objective_exploration_requests", "objective_confirmation_requests",
                  "mean_prediction_rows", "epistemic_prediction_rows", "noise_prediction_rows")
        return {"objective_search_compute": sum((row.get("objective_research") or {}).get("compute_cost", 0) for row in rows),
            **{key: sum(row["costs"].get("objective_diagnostics", {}).get(key, 0) for row in rows) for key in fields},
            "actual_goal_outcomes": len(outcomes),
            "outcomes_with_measured_improvement": len(improvements),
            "positive_measured_improvements": sum(value > 0 for value in improvements),
            "summed_measured_improvement": sum(improvements),
            "positive_measured_improvements_with_measured_feasibility": sum(item.get("measured_improvement", 0) > 0
                and item["constraint"] <= 0 for item in outcomes)}

    for group in compact["groups"]:
        for arm in ARMS:
            rows = [row for row in summary["results"]
                if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], arm)]
            group[arm].update(metrics(rows))
            group[arm]["combined_objective_and_revision_search_compute"] = (
                group[arm]["objective_search_compute"] + group[arm]["recovery_compute"])
    fresh = [row for row in summary["results"] if row.get("run_origin") != spec["historical_baseline_origin"]]
    compact["actual_new_objective_research_costs"] = metrics(fresh)
    compact["actual_new_combined_search_compute"] = (
        compact["actual_new_objective_research_costs"]["objective_search_compute"]
        + compact["costs"]["actual_new_recovery_compute"])
    compact["matched_total_search_compute"] = {arm: sum(
        (row.get("objective_research") or {}).get("compute_cost", 0)
        + row["costs"]["recovery_compute"].get("compute_cost", 0)
        for row in summary["results"] if row["arm"] == arm) for arm in ARMS}
    compact["objective_only_change_cases"] = []
    for case in spec["cases"]:
        if case["profile"] != "objective_only_change":
            continue
        for row in summary["results"]:
            if row["case_id"] != case["case_id"]:
                continue
            requests = row["costs"].get("objective_diagnostics", {}).get("requests", [])
            stable = next((item for item in summary["results"] if item["profile"] == "stable_state_policy"
                and item["task"] == row["task"] and item["arm"] == row["arm"]), None)
            baseline = next(item for item in summary["results"] if item["case_id"] == row["case_id"] and item["arm"] == ARMS[0])
            compact["objective_only_change_cases"].append({"case_id": case["case_id"], "arm": row["arm"],
                "actual_first_changed_stage": 45,
                "same_actual_prefix_before_change": row["simulations"][:12] == stable["simulations"][:12] if stable else None,
                "goal_request_stages": [{key: request[key] for key in ("stage", "purpose", "reason")}
                    for request in requests],
                "goal_requests_after_objective_change": [{key: request[key] for key in ("stage", "purpose", "reason")}
                    for request in requests if request["stage"] >= 45],
                "constraint_research_reopening_stages": [event["stage"] for event in row["recovery"]["events"]
                    if event["action"] == "reopen_hypothesis_research"],
                "constraint_scope_revisions": len(row["scope_revisions"]),
                "objective": row["final"]["true_objective"], "feasible": row["final"]["true_feasible"],
                "objective_delta_from_baseline": row["final"]["true_objective"] - baseline["final"]["true_objective"],
                "objective_gp_true_mse": row["generalization"].get("objective_gp_true_objective_means_mse")})
    note = "Goal search has its own20000 compute cap, separate from constraint research. Recorded measured improvements compare noisy observations and are not estimates of true gain."
    if note not in compact["notes"]:
        compact["notes"].append(note)
    study.write(output / f"{stem}compact_summary.json", compact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    parser.add_argument("--case", action="append", help="Run selected cases from the frozen roster.")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = repair.ARMS = ARMS
    study.controlled_problem = controlled_problem
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved objective-exploration method or roster differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 112,
            "confirmation_cases": 18, "stress_cases": 9, "new_optimizer_runs": 166}))
        return
    cases = spec["cases"]
    if args.phase == "replay" and not args.case:
        cases = [case for case in cases if case["case_id"] in REPLAY_CASES]
    elif args.phase in ("development", "confirmation", "stress"):
        cases = [case for case in cases if case["split"] == args.phase]
    if args.case:
        cases = [case for case in cases if case["case_id"] in args.case]
        if not cases or set(args.case) - {case["case_id"] for case in cases}:
            raise ValueError("case selection must name frozen cases in the requested phase")
    accounting.prepare_reuse(spec, args.output)
    modules = arm_modules(spec)
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results,
                                 environment_factory=evaluation_environment)
    accounting.summarize(spec, args.output, cases, research, results, evaluations, args.phase)
    repair.enrich_repair_summary(spec, args.output, args.phase)
    enrich_objective_summary(spec, args.output, args.phase)


if __name__ == "__main__":
    main()

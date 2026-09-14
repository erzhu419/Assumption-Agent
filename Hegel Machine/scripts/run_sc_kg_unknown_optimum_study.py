#!/usr/bin/env python3
"""Pilot frozen Hegel0.52 and native SC-KG on three unknown interior optima."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
from statistics import mean
import sys
import types

import run_sc_kg_joint_revision_study as study
import run_sc_kg_vanilla_comparison as vanilla
import sc_kg_unknown_optimum as task


SEEDS = (1843012269, 1843112287, 1843212301)
HEGEL, NATIVE = "hegel_v052", "native_sc_kg_shared_paid32"


def prepare(output):
    original = study.read(study.ROOT / "artifacts/sc_kg_objective_reentry_v052_20260911/protocol.json")
    frozen = output / "frozen_hegel_v052"
    if not frozen.exists():
        frozen.mkdir(parents=True)
        source = study.ROOT / "artifacts/sc_kg_reentry_value_v053_20260911/baseline_source_v052"
        for path in source.glob("*.py"):
            shutil.copyfile(path, frozen / path.name)
    if len(list(frozen.glob("*.py"))) != 12:
        raise ValueError("the frozen twelve-module v0.52 package is required")
    cases = [{"case_id": f"unknown_interior_{index}_noise_0.03", "split": "pilot",
        "profile": "stable_unknown_interior_goal", "cohort": "pilot3",
        "task": {"problem": "state_policy_rzdt1", "dimension": 3, "seed": seed, "noise": .03},
        "next_problem": task.PROBLEM, "research_id": f"unknown_interior_{index}"}
        for index, seed in enumerate(SEEDS, 1)]
    spec = {key: deepcopy(original[key]) for key in ("sc_kg_project_root", "research_config",
        "total_calls_per_arm", "change_after_new_calls", "optimizer_seed_offset",
        "optimizer_settings", "policy", "revision", "evaluation")}
    spec.update(experiment="unknown_interior_goal_pilot", cases=cases, seeds=list(SEEDS),
        arms=[NATIVE, HEGEL], frozen_hegel_source=str(frozen.resolve()),
        source_completed_study=str((study.ROOT / "artifacts/sc_kg_objective_reentry_v052_20260911").resolve()),
        task_formula={"normalized_control": "t=x/100, x in {0,...,100}^3",
            "center_generation": "default_rng(task_seed+91117); draw each integer independently, inclusive bounds",
            "center_integer_ranges": [list(bounds) for bounds in task.CENTER_INTEGER_RANGES],
            "objective_means": "f1=f2=0.35+2.2*(t1-c1)^2+2.5*(t2-c2)^2+2.5*(t3-c3)^2; c=center_integer/100",
            "constraint_mean": "StatePolicyRZDT1: 0.12*(((t1-.25)/.28)^2+mean(((t_tail-.70)/.28)^2)-1.5)+.02*std(t_tail)",
            "noise": "One iid Gaussian vector draw with standard deviation.03 on each of three outputs; scalar objective=.5*f1+.5*f2",
            "true_minimum": .35, "mechanism": "Stable throughout all64 observations; no change or outlier injection"},
        task_information="Only the script-owned simulator stores its hidden center. The public TaskSpec, "
            "context, probes, initial completion and optimizer candidates are unchanged functions of "
            "public settings and paid observations. Center coordinates never initialize or nominate "
            "controls. Evaluator noise uses task_seed+6000017 while its true objective uses the original "
            "task_seed; task parameters and analytic minimum enter only postrun evaluation.",
        comparison="Ordinary native SC-KG is conditioned on the same32 actually purchased full vectors "
            "as frozen Hegel0.52. Its only modifications load those records through the two standard "
            "initial-data hooks. Both arms then use the same original32-online-call budget and optimizer "
            "settings. Initial research computation belongs to Hegel and is reported separately.",
        expected_optimizer_runs=6, expected_initial_calls=96, expected_online_calls=192,
        expected_evaluation_calls=384, expected_physical_calls=672,
        accounting="Three shared32-vector research initializations cost96 simulator calls; six "
            "optimizations cost32 new calls each,192 total. After all final predictions are frozen, "
            "three independent128-vector evaluations cost384. Total672 physical calls. Evaluation "
            "is shared across arms, never fed back. Preserve all failures, infeasible recommendations "
            "and model costs. No other optimizer variant or CodeAgent is executed.",
        limitations="A three-task pilot tests optimization room beyond the earlier shared known minima. "
            "It is a conditional shared-initialization comparison and cannot establish broad superiority. "
            "The separable quadratic objective is representable by the existing native basis; report "
            "initial best feasible objective and final regret to assess remaining saturation.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen pilot protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def load_hegel(spec):
    name = "hegel_machine.cognition._unknown_optimum_hegel_v052"
    package = types.ModuleType(name)
    package.__path__ = [spec["frozen_hegel_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[name] = package
    module = importlib.import_module(f"{name}.sc_kg_environment")
    study.Program = importlib.import_module(f"{name}.language").Program
    study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    study.SCKGEnvironment = task.configure(module.SCKGEnvironment)
    study.controlled_problem = task.controlled_problem
    study.ARMS = (HEGEL,)
    return {HEGEL: (module, importlib.import_module(f"{name}.prior_adoption").PriorAdoption,
                   importlib.import_module(f"{name}.joint_model_revision").JointModelRevision)}


def run_native(spec, output, case, saved):
    path = output / "runs" / f"{case['case_id']}_{NATIVE}.json"
    if path.exists():
        return study.read(path)
    state = saved["optimizer_initial_environment_state"]
    records = [{"point": point, "output": [.5 * vector[0] + .5 * vector[1], vector[2]]}
        for acquisition in state["acquisitions"]
        for point, vector in zip(acquisition["integer_points"], acquisition["vector_outputs"])]
    algorithm, problem = vanilla.build_vanilla_optimizer(spec, case, records)
    from problems.single_objective import ScalarizedProblem
    environment = task.Environment(spec["sc_kg_project_root"], **case["task"])
    problem.base = ScalarizedProblem(environment._problem)
    final = algorithm.run()
    if problem.actual_simulator_calls != 32 or len(algorithm.history) != 64:
        raise RuntimeError("native SC-KG must consume the shared32 rows and exactly32 new calls")
    initial_outputs = [[float(value) for value in vector] for _, vector in algorithm.history[:32]]
    if initial_outputs != [record["output"] for record in records]:
        raise RuntimeError("native SC-KG did not consume the same paid prefix")
    points = study.evaluation_controls(spec, case)
    row = {"case_id": case["case_id"], "arm": NATIVE, "task": case["task"],
        "initial_points": algorithm.pre_sampling_log["samples"], "initial_outputs": initial_outputs,
        "simulations": problem.simulation_log,
        "actions": [{key: action.get(key) for key in ("stage", "x_selected", "Y_observed", "mu_before",
            "action_kind", "n_candidates", "t_kg_compute")} for action in algorithm.iteration_log],
        "costs": {"total_initial_simulator_calls": 32, "optimizer_new_simulator_calls": 32,
            "total_simulator_calls": 64, "hegel_research_compute": 0,
            "native_kg_seconds": sum(action.get("t_kg_compute", 0.) for action in algorithm.iteration_log),
            "native_kg_scored_actions": sum(action.get("n_candidates", 0) for action in algorithm.iteration_log),
            "evaluation_prediction_rows": 2 * len(points),
            "initial_belief_timing": getattr(algorithm, "_last_initial_belief_timing", {})},
        "frozen_evaluation_predictions": {"integer_points": points,
            "objective_gp": [float(algorithm.gpr[0].posterior_mean(point)) for point in points],
            "active_gp": [float(algorithm.gpr[1].posterior_mean(point)) for point in points]},
        "final": {key: final.get(key) for key in ("x_recommended", "true_objective",
            "true_constraint_mean", "true_feasible", "n_simulations", "total_time_sec")}}
    study.write(path, row)
    print(json.dumps({"case": case["case_id"], "arm": NATIVE, "final": row["final"]}), flush=True)
    return row


def complete(spec, output, research, hegel, native):
    evaluations = study.evaluate(spec, output, spec["cases"], hegel,
        environment_factory=task.evaluation_environment)
    pairs = []
    for case in spec["cases"]:
        key = case["case_id"]
        learned = next(row for row in hegel if row["case_id"] == key)
        ordinary = next(row for row in native if row["case_id"] == key)
        evaluation = evaluations[f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}"]
        prediction = ordinary["frozen_evaluation_predictions"]
        ordinary["generalization"] = {
            "objective_gp_true_objective_means_mse": mean((a - b) ** 2 for a, b in
                zip(prediction["objective_gp"], evaluation["true_objective_means"])),
            "active_gp_true_constraint_means_mse": mean((a - b) ** 2 for a, b in
                zip(prediction["active_gp"], evaluation["true_constraint_means"]))}
        environment = task.Environment(spec["sc_kg_project_root"], **case["task"])
        scalar = importlib.import_module("problems.single_objective").ScalarizedProblem(environment._problem)
        initial = [scalar.true_objective(point) for point in ordinary["initial_points"]
                   if scalar.is_truly_feasible(point)]
        pairs.append({"case_id": key, "initial_best_true_feasible_objective": min(initial) if initial else None,
            "evaluator_only_center_integer": list(environment._problem._center_integer),
            "true_optimum": .35, "comparison_order": [NATIVE, HEGEL],
            "final_objective": [row["final"]["true_objective"] for row in (ordinary, learned)],
            "final_regret": [row["final"]["true_objective"] - .35 for row in (ordinary, learned)],
            "feasible": [row["final"]["true_feasible"] for row in (ordinary, learned)],
            "recommendation": [row["final"]["x_recommended"] for row in (ordinary, learned)],
            "objective_gp_true_mse": [row["generalization"]["objective_gp_true_objective_means_mse"] for row in (ordinary, learned)],
            "constraint_gp_true_mse": [row["generalization"]["active_gp_true_constraint_means_mse"] for row in (ordinary, learned)],
            "native_kg_seconds": [ordinary["costs"]["native_kg_seconds"], sum(
                action.get("t_kg_compute", 0.) for action in learned["costs"]["exact_kg_stages"])]})
        for row in (ordinary, learned):
            study.write(output / "runs" / f"{key}_{row['arm']}.json", row)
    initial_calls = sum(row["actual_new_initial_calls"] for row in research.values())
    online_calls = sum(row["costs"]["optimizer_new_simulator_calls"] for row in [*hegel, *native])
    evaluator_calls = sum(row["simulator_calls"] for row in evaluations.values())
    brief = {"protocol": str((output / "protocol.json").resolve()), "cases": len(pairs), "pairs": pairs,
        "costs": {"initial_physical_calls": initial_calls, "online_physical_calls": online_calls,
            "evaluation_physical_calls": evaluator_calls, "total_physical_calls": initial_calls + online_calls + evaluator_calls,
            "initial_hegel_research_compute": sum(row["result"]["costs"]["compute_cost"] for row in research.values()),
            "online_hegel_objective_compute": sum((row.get("objective_research") or {}).get("compute_cost", 0) for row in hegel),
            "online_hegel_constraint_compute": sum(row["costs"]["recovery_compute"].get("compute_cost", 0) for row in hegel),
            "native_kg_seconds": {arm: sum(pair["native_kg_seconds"][i] for pair in pairs)
                                  for i, arm in enumerate((NATIVE, HEGEL))}},
        "limitations": spec["limitations"]}
    study.write(output / "result_brief.json", brief)
    print(json.dumps(brief), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    spec = prepare(args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "tasks": 3, "optimizers": 6,
                          "physical_calls": spec["expected_physical_calls"]}), flush=True)
        return
    modules = load_hegel(spec)
    research = study.acquire_research(spec, args.output, spec["cases"])
    hegel = study.optimize(spec, args.output, spec["cases"], research, arm_modules=modules)
    native = [run_native(spec, args.output, case, research[case["research_id"]]) for case in spec["cases"]]
    complete(spec, args.output, research, hegel, native)


if __name__ == "__main__":
    main()

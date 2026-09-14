#!/usr/bin/env python3
"""Unknown objective interactions and changes: native, frozen0.52, and research ablation."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
from statistics import mean
import sys
import types

import run_sc_kg_joint_revision_study as study
import run_sc_kg_vanilla_comparison as vanilla
import sc_kg_unknown_revision as task

SEEDS = (1943012413, 1943112439, 1943212463)
NATIVE = "native_sc_kg_shared_paid32"
HEGEL = "hegel_v052"
FROZEN_GOAL = "hegel_initial_goal_frozen"
INTERACTION = "hegel_interaction_candidate"


def prepare(output):
    original = study.read(study.ROOT / "artifacts/sc_kg_objective_reentry_v052_20260911/protocol.json")
    cases = []
    for index, seed in enumerate(SEEDS, 1):
        for profile in task.PROFILES:
            initial = "stable_interaction" if profile == "stable_interaction" else "initial_separable"
            cases.append({"case_id": f"unknown_revision_{index}_{profile}", "split": "development",
                "profile": profile, "cohort": "new9", "change_after_new_calls": 8,
                "task": {"problem": initial, "dimension": 3, "seed": seed, "noise": .03},
                "next_problem": profile, "research_id": f"unknown_revision_{index}_{initial}"})
    spec = {key: deepcopy(original[key]) for key in ("sc_kg_project_root", "research_config",
        "total_calls_per_arm", "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    spec.update(experiment="unknown_interior_objective_revision", cases=cases, seeds=list(SEEDS),
        arms=[NATIVE, HEGEL, FROZEN_GOAL], additional_arm=INTERACTION,
        frozen_hegel_source=str((output / "baseline_source_v052").resolve()),
        frozen_interaction_source=str((output / "interaction_source").resolve()),
        change_after_new_calls=8,
        task_formula={"domain": "integer x in {0,...,100}^3, t=x/100",
            "center_initial": "default_rng(task_seed+91117), independent inclusive integer ranges[13,38],[58,82],[58,82]",
            "center_final": "Within each initial integer interval, cyclically shift the center by[13,12,12] modulo interval size",
            "goal": "f1=f2=.35+(t-c)'Q(t-c)",
            "quadratics": {key: [list(row) for row in matrix] for key, matrix in task.QUADRATICS.items()},
            "profiles": {"stable_interaction": "Initial center and interaction Q1 throughout64 calls",
                "center_change": "Initial center/diagonal Q0; new center and same Q0 from online call9",
                "center_interaction_change": "Initial center/diagonal Q0; new center and interaction Q2 from online call9"},
            "constraint": "Original StatePolicyRZDT1 mean and chance constraint remain unchanged",
            "noise": "One iid sigma=.03 three-output vector; scalar objective=.5*f1+.5*f2",
            "true_optimum": .35},
        task_information="Centers, quadratic matrices and the change clock belong to the script-owned "
            "simulator. They never enter public TaskSpec/context, initial controls or candidates. Online "
            "calls1..8 use the initial function; call9/global41 is the first changed vector. Evaluation "
            "uses each final function with the original task seed and a separately offset noise stream.",
        comparison="Three fixed arms share each task's32 actually purchased initialization vectors "
            "and use32 new calls each. Native SC-KG uses its original optimizer with only the paid-data "
            "initialization hooks. Frozen0.52 is loaded exclusively from the old immutable package. "
            "The no-online-goal-research ablation permits the first objective-model fit, then freezes "
            "that fitted hypothesis and disables objective monitoring/re-entry; it retains goal "
            "exploration, GP/HVD updating, prior adoption and identical constraint research.",
        expected_control_optimizer_runs=27, expected_control_online_calls=864,
        expected_initial_calls=192, expected_evaluation_calls=1152, expected_control_physical_calls=2208,
        additional_interaction_optimizer_runs=9, additional_interaction_online_calls=288,
        expected_all_physical_calls=2496,
        additional_method="After the three-arm control experiment, append the independently frozen "
            "interaction candidate method on the same nine tasks. It adds control-group-held-out "
            "cross-term candidates and callable programs plus joint-optimum nomination, with fitting "
            "and solve work charged. Its source copies frozen0.52 and changes only objective_research.py, "
            "decision_experiments.py and the budget argument in objective_reentry.py. Reuse all initial "
            "and evaluator vectors; do not rerun controls.",
        accounting="Each seed has two distinct initial functions, so six shared32-vector prefixes cost192. "
            "Nine tasks times three control arms times32 online calls cost864. Nine final-function "
            "evaluations of128 vectors cost1152, totaling2208. The interaction arm adds288 online calls, "
            "for2496 across all four arms. Initialization research computation, online Hegel model work, "
            "native KG timings and evaluation calls are separate. Preserve every regression and failure.",
        limitations="Nine fixed synthetic cases diagnose objective change and representation gaps. "
            "These are shared-initialization conditional comparisons, not broad agent benchmarks. "
            "The three profiles and all centers were fixed before observing outcomes. Interaction "
            "development uses these tasks and must not be described as independent confirmation.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen unknown-revision protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def freeze_initial_goal(algorithm):
    research = algorithm.objective_research
    original_fit = research._fit_if_needed
    attempted = False

    def initial_fit(self, *args, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            return original_fit(*args, **kwargs)

    research._fit_if_needed = types.MethodType(initial_fit, research)
    research._monitor = types.MethodType(lambda self, rows: None, research)
    research._resume_rejected = types.MethodType(lambda self, rows, *, stage, remaining_calls: None, research)
    return algorithm


def load_hegel(spec, arms):
    result = {}
    for arm in arms:
        source = spec["frozen_interaction_source" if arm == INTERACTION else "frozen_hegel_source"]
        name = f"hegel_machine.cognition._unknown_revision_{arm}"
        package = types.ModuleType(name)
        package.__path__ = [source, str(study.ROOT / "src/hegel_machine/cognition")]
        sys.modules[name] = package
        module = importlib.import_module(f"{name}.sc_kg_environment")
        module.SCKGEnvironment = task.configure(module.SCKGEnvironment)
        if arm == FROZEN_GOAL:
            initializer = module.initialize_sc_kg_optimizer
            def initialize(*args, _original=initializer, **kwargs):
                return freeze_initial_goal(_original(*args, **kwargs))
            module.initialize_sc_kg_optimizer = initialize
        result[arm] = (module, importlib.import_module(f"{name}.prior_adoption").PriorAdoption,
                       importlib.import_module(f"{name}.joint_model_revision").JointModelRevision)
        if arm == arms[0]:
            initial_environment = module.SCKGEnvironment
            study.Program = importlib.import_module(f"{name}.language").Program
            study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    task.Environment = study.SCKGEnvironment = initial_environment
    study.controlled_problem = task.controlled_problem
    study.ARMS = tuple(arms)
    return result


def build_native_optimizer(spec, case, records):
    scheduled = {**case, "profile": "stable_interaction" if case["profile"] == "stable_interaction" else "mechanism_change",
                 "change_after_new_calls": task.CHANGE_AFTER_NEW_CALLS}
    algorithm, problem = vanilla.build_vanilla_optimizer(spec, scheduled, records)
    from problems.single_objective import ScalarizedProblem
    environment = task.Environment(spec["sc_kg_project_root"], **case["task"])
    problem.base = ScalarizedProblem(environment._problem)
    problem.next_base = ScalarizedProblem(task.make_vector_problem(environment, case["profile"]))
    return algorithm, problem


def run_native(spec, output, case, saved):
    path = output / "runs" / f"{case['case_id']}_{NATIVE}.json"
    if path.exists():
        return study.read(path)
    state = saved["optimizer_initial_environment_state"]
    records = [{"point": point, "output": [.5 * vector[0] + .5 * vector[1], vector[2]]}
        for acquisition in state["acquisitions"]
        for point, vector in zip(acquisition["integer_points"], acquisition["vector_outputs"])]
    algorithm, problem = build_native_optimizer(spec, case, records)
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


def behavior(row):
    state = row.get("objective_research") or {}
    scopes = list(state.get("scope_attempts", []))
    if state.get("scope_research"):
        scopes.append(state["scope_research"])
    return {"case_id": row["case_id"], "arm": row["arm"],
        "events": [{key: event[key] for key in ("action", "stage", "scope_start", "reason") if key in event}
                   for event in state.get("events", [])],
        "fits": state.get("fits", []),
        "candidate_scopes": [{"scope_start": scope["scope_start"], "status": scope.get("status"),
            "candidate": scope.get("candidate"), "training_indices": scope.get("training_indices"),
            "validation_rounds": scope.get("validation_rounds", []),
            "validation_indices": [entry["global_index"] for entry in scope.get("validation", [])]}
            for scope in scopes],
        "adoptions": row.get("objective_scope_revisions", []),
        "requests": dict(Counter(request["reason"] for request in state.get("requests", []))),
        "objective_compute": state.get("compute_cost", 0),
        "constraint_compute": row["costs"].get("recovery_compute", {}).get("compute_cost", 0)}


def summarize(spec, output, research, rows, phase):
    learned = [row for row in rows if row["arm"] != NATIVE]
    evaluations = study.evaluate(spec, output, spec["cases"], learned,
        environment_factory=task.evaluation_environment)
    arms = [arm for arm in (NATIVE, HEGEL, FROZEN_GOAL, INTERACTION)
            if any(row["arm"] == arm for row in rows)]
    comparisons = []
    for case in spec["cases"]:
        case_rows = {row["arm"]: row for row in rows if row["case_id"] == case["case_id"]}
        evaluation = evaluations[f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}"]
        native = case_rows[NATIVE]
        prediction = native["frozen_evaluation_predictions"]
        native["generalization"] = {"objective_gp_true_objective_means_mse": mean((a-b)**2 for a,b in
            zip(prediction["objective_gp"], evaluation["true_objective_means"])),
            "active_gp_true_constraint_means_mse": mean((a-b)**2 for a,b in
            zip(prediction["active_gp"], evaluation["true_constraint_means"]))}
        environment = task.Environment(spec["sc_kg_project_root"], **case["task"])
        final_problem = task.make_vector_problem(environment, case["profile"])
        initial_points = native["initial_points"]
        feasible_values = [final_problem.true_objectives(point)[0] for point in initial_points
                           if final_problem.is_truly_feasible(point)]
        comparisons.append({"case_id": case["case_id"], "profile": case["profile"],
            "initial_points_best_final_true_feasible": min(feasible_values) if feasible_values else None,
            "evaluator_only_final_center": list(final_problem._center_integer), "true_optimum": .35,
            "arms": {arm: {"objective": case_rows[arm]["final"]["true_objective"],
                "regret": case_rows[arm]["final"]["true_objective"]-.35,
                "feasible": case_rows[arm]["final"]["true_feasible"],
                "recommendation": case_rows[arm]["final"]["x_recommended"],
                "objective_mse": case_rows[arm]["generalization"]["objective_gp_true_objective_means_mse"],
                "constraint_mse": case_rows[arm]["generalization"]["active_gp_true_constraint_means_mse"]}
                for arm in arms}})
        for row in case_rows.values():
            study.write(output/"runs"/f"{row['case_id']}_{row['arm']}.json",row)
    initial = sum(row["actual_new_initial_calls"] for row in research.values())
    online = sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows)
    evaluator = sum(row["simulator_calls"] for row in evaluations.values())
    metrics = {}
    for profile in ("overall", *task.PROFILES):
        chosen = [item for item in comparisons if profile == "overall" or item["profile"] == profile]
        metrics[profile] = {arm: {"feasible": sum(item["arms"][arm]["feasible"] for item in chosen),
            **{key: mean(item["arms"][arm][key] for item in chosen) for key in
               ("objective", "regret", "objective_mse", "constraint_mse")}} for arm in arms}
    brief = {"protocol": str((output/"protocol.json").resolve()), "phase": phase,
        "cases": 9, "optimizer_runs": len(rows), "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [behavior(row) for row in learned],
        "costs": {"initial_physical_calls": initial, "online_physical_calls": online,
            "evaluation_physical_calls": evaluator, "total_physical_calls": initial+online+evaluator,
            "initial_research_compute": sum(row["result"]["costs"]["compute_cost"] for row in research.values()),
            "online_by_arm": {arm: {"objective_compute": sum((row.get("objective_research") or {}).get("compute_cost",0)
                for row in rows if row["arm"]==arm),
                "constraint_compute": sum(row["costs"].get("recovery_compute",{}).get("compute_cost",0)
                for row in rows if row["arm"]==arm),
                "native_kg_seconds": sum(row["costs"]["native_kg_seconds"] if arm==NATIVE else sum(
                    entry.get("t_kg_compute",0.) for entry in row["costs"]["exact_kg_stages"])
                    for row in rows if row["arm"]==arm)} for arm in arms}},
        "limitations": spec["limitations"]}
    filename = "controls_result_brief.json" if phase == "controls" else "result_brief.json"
    study.write(output/filename,brief)
    print(json.dumps({"phase":phase,"costs":brief["costs"],"metrics":metrics}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("plan","controls","interaction"),default="plan")
    args=parser.parse_args()
    spec=prepare(args.output)
    if args.phase=="plan":
        print(json.dumps({"status":"prepared_not_run","control_physical_calls":2208,"additional_interaction_calls":288}),flush=True)
        return
    if args.phase=="controls":
        modules=load_hegel(spec,(HEGEL,FROZEN_GOAL))
        research=study.acquire_research(spec,args.output,spec["cases"])
        rows=study.optimize(spec,args.output,spec["cases"],research,arm_modules=modules)
        rows.extend(run_native(spec,args.output,case,research[case["research_id"]]) for case in spec["cases"])
        summarize(spec,args.output,research,rows,"controls")
    else:
        modules=load_hegel(spec,(INTERACTION,))
        research={case["research_id"]:study.read(args.output/"research"/f"{case['research_id']}.json") for case in spec["cases"]}
        rows=study.optimize(spec,args.output,spec["cases"],research,arm_modules=modules)
        rows.extend(study.read(args.output/"runs"/f"{case['case_id']}_{arm}.json")
                    for case in spec["cases"] for arm in (NATIVE,HEGEL,FROZEN_GOAL))
        summarize(spec,args.output,research,rows,"all")


if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Complete goal research and deploy learned explanations at unchanged budgets."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
from statistics import mean
import sys
import types

import run_sc_kg_unknown_revision_study as unknown
import run_sc_kg_unknown_optimum_study as pilot
import run_sc_kg_interaction_retention as historical


study = unknown.study
METHOD = "hegel_goal_completion"
SEEDS = (2243012671, 2243112689, 2243212717)
SMOKE = (
    "unknown_confirmation_1_stable_interaction",
    "unknown_confirmation_1_center_change",
    "research_scope_3_objective_only_change_noise_0.03",
)
ARTIFACTS = study.ROOT / "artifacts"
SOURCES = {
    "unknown_development": ARTIFACTS / "sc_kg_unknown_revision_20260911",
    "unknown_previous_confirmation": ARTIFACTS / "sc_kg_unknown_revision_confirmation_20260911",
    "historical_retention": ARTIFACTS / "sc_kg_interaction_retention_20260911",
    "separable_pilot": ARTIFACTS / "sc_kg_unknown_optimum_pilot_20260911",
}


def evaluation_name(case):
    return f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"


def prepare(output):
    groups = {}
    for name, source in SOURCES.items():
        original = study.read(source / "protocol.json")
        cases = deepcopy(original["cases"])
        for case in cases:
            case["split"] = "development"
            if name != "historical_retention":
                case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
                case["evaluation_origin"] = str((source / "evaluation" / evaluation_name(case)).resolve())
            prior_source = (SOURCES["unknown_previous_confirmation"] / "pilot_retention"
                            if name == "separable_pilot" else source)
            prior_arm = historical.METHOD if name == "historical_retention" else unknown.INTERACTION
            case["previous_interaction_record"] = str((prior_source / "runs" /
                f"{case['case_id']}_{prior_arm}.json").resolve())
            if name != "historical_retention":
                case["v052_record"] = str((source / "runs" / f"{case['case_id']}_{unknown.HEGEL}.json").resolve())
                case["native_record"] = str((source / "runs" / f"{case['case_id']}_{unknown.NATIVE}.json").resolve())
        groups[name] = {"source_protocol": str((source / "protocol.json").resolve()), "cases": cases}
    original = study.read(SOURCES["unknown_development"] / "protocol.json")
    fresh = []
    for index, (old_seed, seed) in enumerate(zip(original["seeds"], SEEDS), 1):
        for old in original["cases"]:
            if old["task"]["seed"] != old_seed:
                continue
            case = deepcopy(old)
            case["task"]["seed"] = seed
            case.update(case_id=f"goal_completion_{index}_{case['profile']}",
                research_id=f"goal_completion_{index}_{case['task']['problem']}",
                cohort="fresh9", split="confirmation")
            fresh.append(case)
    spec = {"experiment": "goal_research_completion_and_supported_model_deployment",
        "groups": groups, "smoke_cases": list(SMOKE), "method_arm": METHOD,
        "method_source": str((output / "method_source").resolve()),
        "source_interaction": original["frozen_interaction_source"],
        "source_v052": original["frozen_hegel_source"],
        "fresh_seeds": list(SEEDS), "fresh_cases": fresh,
        "development": {"cases": 34, "new_optimizer_runs": 34, "initial_calls": 0,
            "online_calls": 1088, "evaluation_calls": 0, "physical_calls": 1088,
            "smoke_calls_included_once": 96},
        "confirmation": {"cases": 9, "arms": [unknown.NATIVE, unknown.HEGEL, METHOD],
            "new_optimizer_runs": 27, "initial_calls": 192, "online_calls": 864,
            "evaluation_calls": 1152, "physical_calls": 2208},
        "method": "Build the new method from the previously frozen interaction package. Improve "
            "research completion and compare an already learned goal explanation with the deployed "
            "objective posterior. Keep the existing per-task observation and computation limits. "
            "Freeze the final method before the first smoke observation.",
        "selection": "All18 previously run unknown-revision cases are now development evidence, "
            "including the former confirmation failures; add all13 historical retention cases and "
            "all3 separable pilot cases. The smoke subset is the specified two unknown failures and "
            "the objective-only-change regression. Reuse completed smoke runs in the full34.",
        "accounting": "Development reuses all initial vectors, evaluator vectors and controls; "
            "only34 times32 online observations are new. Confirmation uses three unseen fixed seeds "
            "with the same three profiles and two distinct32-row initial datasets per seed. Each "
            "arm receives32 initial plus32 online observations. Confirmation initial research uses "
            "frozen0.52. Report physical calls, research compute and native KG work separately.",
        "limitations": "The34 previous cases support development and retention only. Fresh9 is "
            "independent only if the method remains frozen after development. Native comparisons "
            "condition on shared paid initialization. The13 old cases have existing0.52/interaction "
            "references, while all21 unknown cases also have native controls. Historical timing is "
            "not a simultaneous throughput benchmark."}
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved goal-completion protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def load_method(spec, group):
    """Load the three actual simulator families through their existing adapters."""
    source = Path(spec["method_source"])
    if not (source / "sc_kg_environment.py").exists():
        raise ValueError("the completed method must be frozen before running")
    name = f"hegel_machine.cognition._goal_completion_{group}"
    package = types.ModuleType(name)
    package.__path__ = [str(source), str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[name] = package
    module = importlib.import_module(f"{name}.sc_kg_environment")
    study.Program = importlib.import_module(f"{name}.language").Program
    study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    if group.startswith("unknown"):
        module.SCKGEnvironment = unknown.task.configure(module.SCKGEnvironment)
        study.controlled_problem = unknown.task.controlled_problem
        evaluator = unknown.task.evaluation_environment
    elif group == "separable_pilot":
        module.SCKGEnvironment = pilot.task.configure(module.SCKGEnvironment)
        study.controlled_problem = pilot.task.controlled_problem
        evaluator = pilot.task.evaluation_environment
    else:
        goal = historical.replay.validation.active.framework.goal
        study.controlled_problem = goal.controlled_problem
        evaluator = goal.evaluation_environment
    study.SCKGEnvironment = module.SCKGEnvironment
    study.ARMS = (METHOD,)
    modules = {METHOD: (module, importlib.import_module(f"{name}.prior_adoption").PriorAdoption,
        importlib.import_module(f"{name}.joint_model_revision").JointModelRevision)}
    return modules, evaluator


def reused_measurements(output, cases):
    research = {}
    for case in cases:
        key = case["research_id"]
        if key not in research:
            research[key] = study.read(case["initial_research_origin"])
        source = Path(case["evaluation_origin"])
        target = output / "evaluation" / evaluation_name(case)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return research


def run_development(spec, output, smoke=False):
    for group, settings in spec["groups"].items():
        cases = [case for case in settings["cases"] if not smoke or case["case_id"] in SMOKE]
        if not cases:
            continue
        original = study.read(settings["source_protocol"])
        modules, evaluator = load_method(spec, group)
        research = reused_measurements(output, cases)
        rows = study.optimize(original, output, cases, research, arm_modules=modules)
        study.evaluate(original, output, cases, rows, environment_factory=evaluator)
        for row in rows:
            study.write(output / "runs" / f"{row['case_id']}_{METHOD}.json", row)
    summarize_development(spec, output)


def values(row):
    return {"objective": row["final"]["true_objective"], "feasible": row["final"]["true_feasible"],
        "recommendation": row["final"]["x_recommended"],
        "objective_mse": row["generalization"].get("objective_gp_true_objective_means_mse"),
        "constraint_mse": row["generalization"]["active_gp_true_constraint_means_mse"]}


def work(rows):
    return {"online_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows),
        "objective_compute": sum((row.get("objective_research") or {}).get("compute_cost", 0) for row in rows),
        "constraint_compute": sum(row["costs"].get("recovery_compute", {}).get("compute_cost", 0) for row in rows),
        "optimizer_seconds": sum(row["final"]["total_time_sec"] for row in rows),
        "native_kg_seconds": sum(row["costs"].get("native_kg_seconds", sum(
            item.get("t_kg_compute", 0.) for item in row["costs"].get("exact_kg_stages", []))) for row in rows)}


def compact_behavior(row):
    result = unknown.behavior(row)
    state = row.get("objective_research") or {}
    scopes = list(state.get("scope_attempts", []))
    if state.get("scope_research"):
        scopes.append(state["scope_research"])
    result["adoptions"] = [{"stage": item["stage"], "scope_start": item["scope_start"],
        "deployment": item.get("deployment", False),
        "interactions": item["model"].get("interactions", []), "remaining_calls": 64-item["stage"],
        "future_validation_rows": len(item.get("validation_indices", []))}
        for item in result["adoptions"]]
    result["candidate_scopes"] = [{"scope_start": item["scope_start"], "status": item.get("status"),
        "deployment": item.get("deployment", False),
        "degree": (item["candidate"] or {}).get("degree"),
        "interactions": (item["candidate"] or {}).get("interactions", []),
        "training_end": (item["candidate"] or {}).get("training_end"),
        "future_validation_rows": len(item.get("validation", [])),
        "validation_rounds": item.get("validation_rounds", []),
        "completion_plan": item.get("completion_plan"),
        "completion_fits": (item.get("completion") or {}).get("fits", [])} for item in scopes]
    result["shared_future_comparisons"] = len(state.get("shared_outcomes", []))
    result["compute_budget"] = state.get("compute_budget")
    result["compute_remaining"] = state.get("compute_budget", 0) - state.get("compute_cost", 0)
    return result


def summarize_development(spec, output):
    comparisons, metrics, all_rows = [], {}, []
    for group, settings in spec["groups"].items():
        cases = [case for case in settings["cases"]
                 if (output / "runs" / f"{case['case_id']}_{METHOD}.json").exists()]
        if not cases:
            continue
        old_map = {}
        if group == "historical_retention":
            original = study.read(settings["source_protocol"])
            old_map, _ = historical.replay.references(Path(original["source_reentry_study"]),
                [case["case_id"] for case in cases])
        rows_by_arm = {"v052": [], "previous_interaction": [], METHOD: []}
        if group != "historical_retention":
            rows_by_arm[unknown.NATIVE] = []
        for case in cases:
            key = case["case_id"]
            rows = {"v052": old_map[key] if old_map else study.read(case["v052_record"]),
                "previous_interaction": study.read(case["previous_interaction_record"]),
                METHOD: study.read(output / "runs" / f"{key}_{METHOD}.json")}
            if "native_record" in case:
                rows[unknown.NATIVE] = study.read(case["native_record"])
            comparisons.append({"case_id": key, "group": group, "arms": {a: values(r) for a,r in rows.items()}})
            for arm,row in rows.items():
                rows_by_arm[arm].append(row)
            all_rows.append(rows[METHOD])
        selected = [item for item in comparisons if item["group"] == group]
        paired = {}
        for arm in rows_by_arm:
            if arm == METHOD:
                continue
            decisions = []
            for item in selected:
                old, new = item["arms"][arm], item["arms"][METHOD]
                if old["feasible"] != new["feasible"]:
                    decisions.append("improved" if new["feasible"] else "worsened")
                elif not new["feasible"]:
                    decisions.append("both_infeasible")
                else:
                    delta = new["objective"] - old["objective"]
                    decisions.append("improved" if delta < -1e-9 else "worsened" if delta > 1e-9 else "same")
            paired[arm] = dict(Counter(decisions))
        metrics[group] = {"cases": len(cases), "paired_feasibility_first": paired,
            "arms": {arm: {"feasible": sum(row["final"]["true_feasible"] for row in rows),
                **{key: mean(values(row)[key] for row in rows if values(row)[key] is not None)
                   for key in ("objective", "objective_mse", "constraint_mse")},
                "work": work(rows)} for arm,rows in rows_by_arm.items()}}
    brief = {"protocol": str((output / "protocol.json").resolve()), "completed_cases": len(all_rows),
        "expected_cases": 34, "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [compact_behavior(row) for row in all_rows],
        "new_costs": {"initial_calls": 0, "evaluation_calls": 0,
            "physical_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in all_rows),
            **work(all_rows)}, "limitations": spec["limitations"]}
    study.write(output / "development_result_brief.json", brief)
    print(json.dumps({"completed_cases": len(all_rows), "new_costs": brief["new_costs"],
        "paired": {group: item["paired_feasibility_first"] for group,item in metrics.items()}}), flush=True)


def run_confirmation(spec, output):
    target = output / "fresh_confirmation"
    original = study.read(SOURCES["unknown_development"] / "protocol.json")
    fresh = {key: deepcopy(original[key]) for key in ("sc_kg_project_root", "research_config",
        "total_calls_per_arm", "optimizer_seed_offset", "optimizer_settings", "policy", "revision",
        "evaluation", "change_after_new_calls", "task_formula", "task_information")}
    fresh.update(cases=spec["fresh_cases"], seeds=spec["fresh_seeds"],
        arms=spec["confirmation"]["arms"], frozen_hegel_source=spec["source_v052"],
        frozen_method_source=spec["method_source"],
        source_protocol=str((output / "protocol.json").resolve()),
        accounting=spec["confirmation"], method=spec["method"], limitations=spec["limitations"])
    path = target / "protocol.json"
    if path.exists() and study.read(path) != fresh:
        raise ValueError("the saved fresh confirmation differs")
    if not path.exists():
        study.write(path, fresh)
    baseline = unknown.load_hegel(fresh, (unknown.HEGEL,))
    initial = study.Program, study.MetaPriorResearch, study.SCKGEnvironment, unknown.task.Environment
    current, evaluator = load_method(spec, "unknown_fresh_confirmation")
    study.Program, study.MetaPriorResearch, study.SCKGEnvironment, unknown.task.Environment = initial
    study.ARMS = (unknown.HEGEL, METHOD)
    research = study.acquire_research(fresh, target, fresh["cases"])
    rows = study.optimize(fresh, target, fresh["cases"], research, arm_modules={**baseline, **current})
    natives = [unknown.run_native(fresh, target, case, research[case["research_id"]]) for case in fresh["cases"]]
    evaluations = study.evaluate(fresh, target, fresh["cases"], rows, environment_factory=evaluator)
    for row in natives:
        case = next(case for case in fresh["cases"] if case["case_id"] == row["case_id"])
        saved = evaluations[evaluation_name(case)[:-5]]
        prediction = row["frozen_evaluation_predictions"]
        row["generalization"] = {"objective_gp_true_objective_means_mse": mean((a-b)**2 for a,b in
            zip(prediction["objective_gp"], saved["true_objective_means"])),
            "active_gp_true_constraint_means_mse": mean((a-b)**2 for a,b in
            zip(prediction["active_gp"], saved["true_constraint_means"]))}
    rows.extend(natives)
    comparisons = [{"case_id": case["case_id"], "profile": case["profile"],
        "arms": {row["arm"]: {**values(row), "regret": row["final"]["true_objective"]-.35}
            for row in rows if row["case_id"] == case["case_id"]}} for case in fresh["cases"]]
    metrics = {}
    for profile in ("overall", *unknown.task.PROFILES):
        selected = [case for case in comparisons if profile == "overall" or case["profile"] == profile]
        metrics[profile] = {arm: {"feasible": sum(case["arms"][arm]["feasible"] for case in selected),
            **{key: mean(case["arms"][arm][key] for case in selected) for key in
               ("objective", "regret", "objective_mse", "constraint_mse")}} for arm in fresh["arms"]}
    for row in rows:
        study.write(target / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    initial_calls = sum(row["actual_new_initial_calls"] for row in research.values())
    evaluator_calls = sum(row["simulator_calls"] for row in evaluations.values())
    online_calls = sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows)
    brief = {"protocol": str(path.resolve()), "completed_cases": 9, "optimizer_runs": len(rows),
        "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [compact_behavior(row) for row in rows if row["arm"] != unknown.NATIVE],
        "costs": {"initial_calls": initial_calls, "online_calls": online_calls,
            "evaluation_calls": evaluator_calls, "physical_calls": initial_calls+online_calls+evaluator_calls,
            "initial_research_compute": sum(row["result"]["costs"]["compute_cost"] for row in research.values()),
            "online_by_arm": {arm: work([row for row in rows if row["arm"] == arm]) for arm in fresh["arms"]}},
        "limitations": spec["limitations"]}
    study.write(target / "result_brief.json", brief)
    print(json.dumps({"costs": brief["costs"], "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "smoke", "development", "confirmation", "summary"), default="plan")
    args = parser.parse_args()
    spec = prepare(args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "development": spec["development"],
            "confirmation": spec["confirmation"], "fresh_seeds": spec["fresh_seeds"]}), flush=True)
    elif args.phase == "summary":
        summarize_development(spec, args.output)
    elif args.phase == "confirmation":
        run_confirmation(spec, args.output)
    else:
        run_development(spec, args.output, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()

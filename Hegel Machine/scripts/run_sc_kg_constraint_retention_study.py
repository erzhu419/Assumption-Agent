#!/usr/bin/env python3
"""Retain supported constraint explanations while competing for terminal measurements."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import run_sc_kg_objective_replacement_study as replacement


study, validation, passive = replacement.study, replacement.validation, replacement.passive
BASELINE, METHOD = "constraint_retention_baseline_v050", "constraint_retention_v051"
FRESH_SEEDS = (2043011879, 2043111901, 2043211921)
CONTROLS = (
    "research_value_3_stable_state_policy_noise_0.00",
    "research_value_3_stable_rzdt2_noise_0.00",
    "decision_single_outlier_2_state_policy_noise_0.03",
    "research_scope_3_objective_only_change_noise_0.03",
)
DEVELOPMENT = (
    "objective_replacement_3_mechanism_change_noise_0.03",
    "objective_replacement_1_mechanism_change_noise_0.03",
    "research_scope_1_mechanism_change_noise_0.03",
    "decision_experiments_3_mechanism_change_noise_0.03",
    "active_confirmation_1_mechanism_change_noise_0.03",
    "objective_replacement_2_outlier_then_change_noise_0.03",
    "active_confirmation_2_mechanism_change_noise_0.03",
    "value_precision_3_single_observation_outlier_noise_0.03",
    "forward_2_mechanism_change_noise_0.00",
    "objective_incumbent_3_outlier_then_change_noise_0.03",
    "completion_value_2_mechanism_change_noise_0.00",
    *CONTROLS,
)
METHOD_DESCRIPTION = (
    "After an actual accepted constraint-scope revision, install an equal-weight native sequential "
    "mixture of the supported frozen program and its adaptive affine GP readout. Update weights using "
    "future predictive likelihoods; propagate component uncertainty and disagreement through native "
    "fantasy KG. Nominate one differentiating control from public available_probes alongside terminal "
    "objective offers, preserving the ordinary native candidate pool. Disagreement only nominates the "
    "candidate and is not added to the KG score. Initial learned priors without a constraint-scope "
    "revision remain outside this branch. This is probabilistic predictive competition, not a paired-loss "
    "hard adoption rule. Nomination compute is included in recovery compute and must not be added twice."
)


def references(source):
    original = study.read(source / "protocol.json")
    _, old, _ = replacement.previous.references(Path(original["source_incumbent_study"]))
    baseline, refs = {}, {}
    for ref in study.read(source / "summary.json")["result_references"]:
        if ref["arm"] != original["arms"][1]:
            continue
        key = ref["case_id"]
        baseline[key] = study.read(ref["case_record"]) if ref.get("case_record") else old[replacement.BASELINE][key]
        refs[key] = {**ref, "arm": BASELINE, "source_arm": ref["arm"]}
    if len(baseline) != 490:
        raise ValueError("the completed490 v0.50 references are required")
    return original, baseline, refs


def entry(row):
    revisions = row.get("scope_revisions", [])
    objective = row["costs"].get("objective_diagnostics", {})
    return {"constraint_revision_stages": [revision["stage"] for revision in revisions],
        "terminal_rounds": objective.get("terminal_rounds", 0),
        "terminal_repeat_actions": objective.get("terminal_repeat_actions", 0)}


def prepare(source, output, variant=None):
    original, baseline, refs = references(source)
    cases = deepcopy(original["cases"])
    affected = [case["case_id"] for case in cases if baseline[case["case_id"]].get("scope_revisions")]
    if len(affected) != 163 or set(CONTROLS) & set(affected):
        raise ValueError("the recorded163 constraint-scope entries and four controls differ")
    for case in cases:
        if case["cohort"] == "fresh27":
            case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
            case["cohort"] = "historical_v050_27"
        name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        roots = (source, Path(original["source_incumbent_study"]), Path(original["prior_measurement_source"]),
                 Path(case["initial_research_origin"]).parent.parent)
        case["evaluation_origin"] = str(next(root / "evaluation" / name for root in roots
                                             if (root / "evaluation" / name).exists()).resolve())
    templates = [case for case in cases if case["task"]["seed"] == original["fresh_seeds"][0]]
    previous_seeds = ({case["task"]["seed"] for case in cases}
                      | set(replacement.previous.FRESH_SEEDS))
    if len(templates) != 9 or set(FRESH_SEEDS) & previous_seeds:
        raise ValueError("the next confirmation requires three unused seeds and nine existing profiles")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case.pop("initial_research_origin")
            case.pop("evaluation_origin")
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"constraint_retention_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"constraint_retention_{index}_{problem}_noise_{noise:.2f}",
                cohort="fresh27", split="confirmation")
            fresh.append(case)
    reused = [case["case_id"] for case in cases if case["case_id"] not in set(affected) | set(CONTROLS)]
    cases.extend(fresh)
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    runs = len(affected) + len(CONTROLS) + 2 * len(fresh)
    spec.update(experiment="supported_constraint_retention_and_joint_terminal_measurement_value", cases=cases,
        arms=[BASELINE, METHOD], source_replacement_study=str(source.resolve()),
        frozen_baseline_source=str((output / "baseline_source_v050").resolve()),
        frozen_method_source=str((output / "method_source_v051").resolve()),
        affected_cases=affected, unchanged_controls=list(CONTROLS), reused_cases=reused,
        development_cases=list(DEVELOPMENT), fresh_cases=[case["case_id"] for case in fresh],
        fresh_seeds=list(FRESH_SEEDS), expected_new_optimizer_runs=runs,
        expected_new_online_calls=32 * runs, expected_new_initial_calls=384,
        expected_new_evaluation_calls=1920, expected_new_physical_calls=32 * runs + 2304,
        method=METHOD_DESCRIPTION,
        selection="All163 existing paths with an actual constraint-scope revision, including119 with "
            "terminal objective actions and103 with objective repetitions. All163 require execution, "
            "because retaining the program can affect paths without repetitions. Run four no-entry "
            "controls and reuse323 other no-entry paths. Fifteen development cases include the diagnosed "
            "previous fresh failure, cases where the GP remains better than the program, late revisions, "
            "zero-noise cases, and all four no-entry controls.",
        reuse_rule="The new policy begins only at an actual accepted constraint-scope revision. Reuse "
            "a complete v0.50 path when no scope revision occurred and it is not one of the four actual "
            "controls. No prior run is described as a newly executed trajectory.",
        accounting=f"163 historical method runs plus4 controls and54 fresh paired runs cost{32 * runs} "
            "online calls. Fresh shared acquisition costs384 initial and1920 evaluator calls. Each arm "
            "uses32 initial and32 online calls; fifteen development runs cost480 online calls and are "
            "included once. Historical initial/evaluator data and v0.50 outcomes are reused. Report "
            "program/GP comparisons, experiment scoring and GP update costs independently. No new "
            "v0.48, strict50, vanilla or CodeAgent runs.",
        source_loading="Freeze the adopted ten-module v0.50 source and the complete intervention modules "
            "including constraint_retention.py before execution; use each arm's frozen module package.",
        limitations="Existing490 are development evidence, including the prior27 newly inspected cases. "
            "Execute and report the independently predeclared fresh27 separately after development is "
            "promising. Missing historical measured objective evaluator labels remain missing; saved "
            "true means are used only for explicitly labelled true MSE. Reused historical wall time "
            "does not establish simultaneous runtime improvement.")
    if variant is not None:
        spec.update(variant)
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved constraint-retention protocol differs")
    if not path.exists():
        study.write(path, spec)
    study.write(output / "entry_inventory.json", {"source_cases": len(baseline), "affected": [
        {"case_id": key, **entry(baseline[key])} for key in affected],
        "controls": list(CONTROLS), "reused": reused})
    return spec, baseline, refs


def measurements(spec, output, cases):
    research, refs = {}, {"research": {}, "evaluation": {}}
    for case in cases:
        if case["cohort"] == "fresh27":
            continue
        key = case["research_id"]
        if key not in research:
            row = study.read(case["initial_research_origin"])
            if not row.get("optimizer_initial_environment_state") or (
                row["result"]["costs"]["observation_cost"] + row.get("supplemental_initial_calls", 0) != 32):
                raise ValueError(f"missing already purchased32-row initialization: {key}")
            research[key], refs["research"][key] = row, case["initial_research_origin"]
        source = Path(case["evaluation_origin"])
        evaluation = study.read(source)
        if evaluation["integer_points"] != study.evaluation_controls(spec, case):
            raise ValueError("shared evaluation controls differ")
        target = output / "evaluation" / source.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        refs["evaluation"][source.name] = str(source)
    study.write(output / "measurement_references.json", refs)
    return research


def retention_behavior(row):
    state = row["costs"].get("recovery_diagnostics", {}).get("constraint_retention") or {}
    observations, scopes = state.get("observations", []), state.get("scopes", [])
    last_scope = scopes[-1] if scopes else None
    current = [item for item in observations if last_scope and item["stage"] > last_scope["stage"]]
    weights = current[-1]["weights_after"] if current else last_scope["initial_weights"] if last_scope else None
    return {"case_id": row["case_id"], "retained_scopes": len(scopes),
        "scope_stages": [item["stage"] for item in scopes], "future_observations": len(observations),
        "observations_by_purpose": dict(Counter(item["purpose"] for item in observations)),
        "nominated_offers": len(state.get("offers", [])), "selected_discrimination_requests": len(state.get("requests", [])),
        "final_program_weight": weights[1] if weights else None,
        "final_active_constraint_basis": row.get("active_constraint_basis"),
        "work": {key: state.get(key, 0) for key in ("prediction_rows", "component_updates", "comparison_rows",
                    "nomination_compute", "nomination_seconds", "setup_seconds")}}


def summarize(spec, output, cases, baseline, current, refs, actual, phase):
    comparisons, metrics, changed = {}, {}, []
    groups = {"overall": cases, **{cohort: [case for case in cases if case["cohort"] == cohort]
              for cohort in dict.fromkeys(case["cohort"] for case in cases)}}
    for group, selected in groups.items():
        comparison = validation.reuse.comparison(selected, baseline, current)
        comparisons[group] = {key: comparison[key] for key in (
            "feasibility_first", "prediction_loss", "paid_trajectories_changed", "matched_complete_kg")}
        comparisons[group]["common_feasible_objective"] = validation.active.common_feasible_objectives(comparison["cases"])
        metrics[group] = {}
        for arm, mapping in ((BASELINE, baseline), (METHOD, current)):
            rows = [mapping[case["case_id"]] for case in selected]
            metric = passive.metrics(rows)
            behavior = [retention_behavior(row) for row in rows]
            work, purposes = Counter(), Counter()
            for item in behavior:
                work.update(item["work"])
                purposes.update(item["observations_by_purpose"])
            metric["constraint_retention_work"] = dict(work)
            metric["constraint_retention_behavior"] = {key: sum(item[key] for item in behavior) for key in (
                "retained_scopes", "future_observations", "nominated_offers", "selected_discrimination_requests")}
            metric["constraint_retention_behavior"]["observations_by_purpose"] = dict(purposes)
            metric["objective_replacement_comparison_rows"] = sum(
                (row.get("objective_research") or {}).get("replacement_research", {}).get("comparison_rows", 0)
                for row in rows)
            metric["terminal_repeat_actions"] = sum(
                row["costs"].get("objective_diagnostics", {}).get("terminal_repeat_actions", 0) for row in rows)
            metric["native_value_precision_evaluations"] = sum(
                action.get("value_precision_evaluations", 0) for row in rows for action in row["actions"])
            metric["native_value_precision_seconds"] = sum(
                action.get("value_precision_seconds", 0.) for row in rows for action in row["actions"])
            metric["compute_including_comparisons"] = (metric["constraint_recovery_compute"]
                + metric["objective_research_compute"] + metric["objective_replacement_comparison_rows"]
                + work["comparison_rows"])
            metrics[group][arm] = metric
        if group == "overall":
            changed = [row for row in comparison["cases"] if row["paid_trajectory_changed"] or any(
                abs(row[field][1] - row[field][0]) > 1e-12 for field in
                ("objective", "objective_gp_true_mse", "constraint_gp_true_mse"))]
    research = [study.read(path) for path in (output / "research").glob("*.json")]
    initial = sum(row.get("actual_new_research_calls", 0) + row.get("supplemental_initial_calls", 0) for row in research)
    fresh_seeds = set(spec["fresh_seeds"])
    evaluator = sum(row["simulator_calls"] for row in (study.read(path) for path in (output / "evaluation").glob("*.json"))
                    if row["task"]["seed"] - spec["evaluation"]["seed_offset"] in fresh_seeds)
    online = sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual)
    brief = {"protocol": str((output / "protocol.json").resolve()), "phase": phase, "cases": len(cases),
        "actual_new_optimizer_runs": len(actual), "actual_new_online_calls": online,
        "actual_new_initial_calls": initial, "actual_new_evaluation_calls": evaluator,
        "actual_new_physical_calls": initial + online + evaluator, "comparisons": comparisons,
        "metrics_by_cohort": metrics, "changed_cases": changed,
        "retention_cases": [retention_behavior(current[case["case_id"]]) for case in cases],
        "limitations": spec["limitations"]}
    prefix = "replay_" if phase == "replay" else ""
    study.write(output / f"{prefix}result_brief.json", brief)
    study.write(output / f"{prefix}summary.json", {"protocol": spec, "result_references": refs})
    print(json.dumps({"phase": phase, "cases": len(cases), "new_physical_calls": initial + online + evaluator,
        "comparison": comparisons["overall"]["feasibility_first"],
        "prediction_loss": comparisons["overall"]["prediction_loss"]}), flush=True)


def run(spec, output, baseline, old_refs, phase):
    cases = spec["cases"] if phase == "all" else [next(case for case in spec["cases"]
            if case["case_id"] == key) for key in spec["development_cases"]]
    new_cases = [case for case in cases if case["case_id"] not in spec["reused_cases"]]
    modules = dict(zip((BASELINE, METHOD), validation.arm_modules(spec).values()))
    study.controlled_problem = validation.active.framework.goal.controlled_problem
    module = modules[BASELINE][0]
    study.Program = importlib.import_module(f"{module.__package__}.language").Program
    study.SCKGEnvironment = module.SCKGEnvironment
    research = measurements(spec, output, new_cases)
    fresh = [case for case in cases if case["cohort"] == "fresh27"]
    if fresh:
        research.update(study.acquire_research(spec, output, fresh))
    current, actual, refs = {}, [], []
    for case in cases:
        key = case["case_id"]
        for arm in (BASELINE, METHOD):
            if case["cohort"] != "fresh27" and (arm == BASELINE or key in spec["reused_cases"]):
                row = baseline[key]
                ref = {**old_refs[key], "arm": arm, "run_origin": "reused_actual_v050_reference"}
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                ref = {"case_id": key, "arm": arm, "case_record": str((output / "runs" / f"{key}_{arm}.json").resolve()),
                       "run_origin": "actual_v051_study"}
                actual.append(row)
            (baseline if arm == BASELINE else current)[key] = row
            refs.append(ref)
    study.evaluate(spec, output, new_cases, actual, environment_factory=validation.active.framework.goal.evaluation_environment)
    for row in actual:
        study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    summarize(spec, output, cases, baseline, current, refs, actual, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    spec, baseline, refs = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": len(spec["affected_cases"]),
            "controls": len(CONTROLS), "reuse": len(spec["reused_cases"]),
            "development": len(DEVELOPMENT), "fresh": 27,
            "new_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()

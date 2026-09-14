#!/usr/bin/env python3
"""Compare causal re-entry of rejected objective candidates against frozen v0.51."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path

import run_sc_kg_constraint_retention_study as retention


study, validation, passive = retention.study, retention.validation, retention.passive
BASELINE, METHOD = "objective_reentry_baseline_v051", "objective_reentry_v052"
FRESH_SEEDS = (2143011949, 2143111973, 2143211999)
METHOD_DESCRIPTION = (
    "After a frozen objective candidate has been rejected and a future paid control remains, retain "
    "its hypothesis and original negative evidence for renewed predictive comparison. If at least "
    "four future labels remain, first try adding the already supported constraint program as a learned "
    "objective coordinate; accept this representation proposal only when grouped cross-validation "
    "improves both identifiable base representations and the constant reference. Fit only then on "
    "already paid labels and begin a new future comparison. Without a supported repair, resume the "
    "original candidate only when its paired-evidence upper margin still crosses zero. Share forecasts "
    "on later ordinary paid controls and actively scan the public control pool for the greatest "
    "noise-normalized feasible untested disagreement. Objective head installation still requires "
    "the original future-support rule. Rejecting at the final observation cannot change the installed "
    "objective or final recommendation. Helper fitting, scanning and forecast work are included in "
    "objective_research.compute_cost; report that controller work separately from native KG timing."
)
LATE_CONTROL = "active_noise_2_mechanism_change_noise_0.03"
CONTROLS = (LATE_CONTROL, *retention.CONTROLS)
DEVELOPMENT = (
    "independent_goal_1_mechanism_change_noise_0.03",
    "completion_value_1_mechanism_change_noise_0.03",
    "confirmation_3_mechanism_change_noise_0.03",
    "joint_evidence_3_outlier_then_change_noise_0.03",
    "objective_validation_3_outlier_then_change_noise_0.03",
    "objective_exploration_1_stable_state_policy_noise_0.00",
    "research_scope_3_stable_rzdt2_noise_0.00",
    "noise_revision_3_stable_rzdt2_noise_0.00",
    "objective_validation_1_stable_rzdt2_noise_0.00",
    "objective_incumbent_1_stable_rzdt2_noise_0.00",
    *CONTROLS,
)


def references(source):
    original = study.read(source / "protocol.json")
    _, old, _ = retention.references(Path(original["source_replacement_study"]))
    baseline, refs = {}, {}
    for ref in study.read(source / "summary.json")["result_references"]:
        if ref["arm"] != original["arms"][1]:
            continue
        key = ref["case_id"]
        baseline[key] = study.read(ref["case_record"]) if ref.get("case_record") else old[key]
        refs[key] = {**ref, "arm": BASELINE, "source_arm": ref["arm"]}
    if len(baseline) != 517:
        raise ValueError("the completed517 v0.51 references are required")
    return original, baseline, refs


def rejected_entries(row):
    result = []
    for attempt in (row.get("objective_research") or {}).get("scope_attempts", []):
        if attempt.get("status") != "future_predictions_do_not_improve" or not attempt.get("candidate"):
            continue
        stage = max((item["global_index"] + 1 for item in attempt.get("validation", [])),
                    default=row["final"]["n_simulations"])
        result.append({"scope_start": attempt["scope_start"], "rejected_stage": stage,
            "remaining_observations": row["final"]["n_simulations"] - stage,
            "candidate_degree": attempt["candidate"]["degree"],
            "candidate_training_end": attempt["candidate"]["training_end"]})
    return result


def prepare(source, output):
    original, baseline, refs = references(source)
    cases = deepcopy(original["cases"])
    affected = [case["case_id"] for case in cases if any(
        entry["remaining_observations"] > 0 for entry in rejected_entries(baseline[case["case_id"]]))]
    rejected = [case["case_id"] for case in cases if rejected_entries(baseline[case["case_id"]])]
    if (len(affected), len(rejected)) != (10, 12) or set(CONTROLS) & set(affected):
        raise ValueError("the recorded10 future re-entry paths and five controls differ")
    if set(DEVELOPMENT) != set(affected) | set(CONTROLS):
        raise ValueError("development must cover every old re-entry path plus the five controls")
    for case in cases:
        if case["cohort"] == "fresh27":
            case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
            name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
            case["evaluation_origin"] = str((source / "evaluation" / name).resolve())
            case["cohort"] = "historical_v051_27"
    templates = [case for case in cases if case["task"]["seed"] == original["fresh_seeds"][0]]
    previous_seeds = ({case["task"]["seed"] for case in cases}
                      | set(retention.replacement.previous.FRESH_SEEDS))
    if len(templates) != 9 or set(FRESH_SEEDS) & previous_seeds:
        raise ValueError("confirmation requires three unused seeds and nine existing profiles")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case.pop("initial_research_origin")
            case.pop("evaluation_origin")
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"objective_reentry_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"objective_reentry_{index}_{problem}_noise_{noise:.2f}",
                cohort="fresh27", split="confirmation")
            fresh.append(case)
    reused = [case["case_id"] for case in cases if case["case_id"] not in set(affected) | set(CONTROLS)]
    cases.extend(fresh)
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    runs = len(affected) + len(CONTROLS) + 2 * len(fresh)
    spec.update(experiment="reentry_of_rejected_objective_candidates", cases=cases,
        arms=[BASELINE, METHOD], source_retention_study=str(source.resolve()),
        frozen_baseline_source=str((output / "baseline_source_v051").resolve()),
        frozen_method_source=str((output / "method_source_v052").resolve()),
        affected_cases=affected, unchanged_controls=list(CONTROLS), reused_cases=reused,
        rejected_candidate_cases=rejected, development_cases=list(DEVELOPMENT),
        fresh_cases=[case["case_id"] for case in fresh], fresh_seeds=list(FRESH_SEEDS),
        expected_new_optimizer_runs=runs, expected_new_online_calls=32 * runs,
        expected_new_initial_calls=384, expected_new_evaluation_calls=1920,
        expected_new_physical_calls=32 * runs + 2304,
        method=METHOD_DESCRIPTION,
        selection="All10 old paths with rejected frozen objective candidates and later real observations. "
            "Two of517 reject only at stage64; rerun one as a last-observation control and reuse the other. "
            "Four additional no-entry controls complete the fifteen-case development roster. This covers "
            "both diagnosed useful candidates, zero-noise stable tasks, strongly unfavorable candidates, "
            "and a rejection with only one remaining observation.",
        reuse_rule="The new predictive and experimental branch requires an already rejected frozen "
            "candidate before a later paid control. Reuse502 old paths without that entry, excluding "
            "five executed controls. Original shared-observation entry is included because all10 paths "
            "retain paid observations after rejection regardless of the later action purpose. Reused "
            "records remain previously executed v0.51 evidence, not new v0.52 trajectories.",
        accounting=f"Ten historical new-method runs plusfive controls and54 fresh paired runs cost{32 * runs} "
            "online calls. Fresh shared acquisition costs384 initial and1920 evaluator calls. Each arm "
            "uses32 initial and32 online calls. Development15 costs480 online calls and is included once; "
            "there are no further unexecuted affected historical cases after development. Reuse all old "
            "initialization, evaluator and baseline outcomes. Report added comparisons, predictions and "
            "native KG work separately; no new older-method, vanilla or CodeAgent runs.",
        source_loading="Freeze the current cleaned live v0.51 eleven modules directly, rather than restore "
            "the earlier passive experiment's unreachable nomination branches. Freeze the complete new "
            "intervention module package, including objective_reentry.py and decision_experiments.py, "
            "before development and all fresh acquisition.",
        limitations="The existing517 are development evidence, including the newly inspected prior27. "
            "The predeclared fresh27 remain separate and run only after development is promising. "
            "The two diagnosed candidates are accurate on existing evaluator controls but their observed "
            "local comparison failures remain real. Missing historical measured objective evaluator labels "
            "remain missing, and historical timing does not establish simultaneous runtime improvement.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved objective re-entry protocol differs")
    if not path.exists():
        study.write(path, spec)
    study.write(output / "entry_inventory.json", {"source_cases": len(baseline),
        "affected": [{"case_id": key, "entries": rejected_entries(baseline[key])} for key in affected],
        "last_observation_rejections": [key for key in rejected if key not in affected],
        "controls": list(CONTROLS), "reused": reused})
    historical = [case for case in cases if case["cohort"] != "fresh27" and case["case_id"] not in reused]
    research = retention.measurements(spec, output, historical)
    return spec, baseline, refs, research


def reentry_behavior(row):
    state = row.get("objective_research") or {}
    scopes = [*state.get("scope_attempts", [])]
    active = state.get("scope_research")
    if active:
        scopes.append(active)
    reentered = [scope for scope in scopes if scope.get("reentry")]
    active_outcomes = [item for item in state.get("outcomes", []) if item["request"].get("reentry")]
    shared_outcomes = [item for item in state.get("shared_outcomes", []) if item["request"].get("reentry")]
    return {"case_id": row["case_id"], "attempts": state.get("reentry_attempts", []),
        "constraint_concept": state.get("constraint_concept"),
        "active_requests": sum(bool(item.get("reentry")) for item in state.get("requests", [])),
        "active_observations": len(active_outcomes), "shared_observations": len(shared_outcomes),
        "shared_observations_by_purpose": dict(Counter(item["request"]["purpose"] for item in shared_outcomes)),
        "accepted_revisions": [{"scope_start": revision["scope_start"], "reentry": revision["reentry"],
            "stage": revision["validation_indices"][-1] + 1,
            "validation_rows": len(revision["validation_indices"])}
            for revision in state.get("scope_revisions", []) if revision.get("reentry")],
        "scopes": [{"scope_start": scope["scope_start"], "reentry": scope["reentry"],
            "status": scope.get("status"), "phase": scope.get("phase"),
            "active": scope is active, "validation_rows": len(scope.get("validation", [])),
            "comparison_rows": len(scope.get("incumbent_validation", []))
                + len(scope.get("validation", [])) - scope.get("comparison_start", 0),
            "candidate_degree": (scope.get("candidate") or {}).get("degree"),
            "candidate_training_end": (scope.get("candidate") or {}).get("training_end"),
            "mean_gain": (scope.get("validation_rounds") or [{}])[-1].get("mean_gain")}
            for scope in reentered],
        "objective_revision_stages": [revision["stage"] for revision in row.get("objective_scope_revisions", [])]}


def metrics(rows):
    result = passive.metrics(rows)
    retention_work = Counter()
    for row in rows:
        retention_work.update(retention.retention_behavior(row)["work"])
    result["constraint_retention_work"] = dict(retention_work)
    result["objective_replacement_comparison_rows"] = sum(
        (row.get("objective_research") or {}).get("replacement_research", {}).get("comparison_rows", 0)
        for row in rows)
    behavior = [reentry_behavior(row) for row in rows]
    scopes = [scope for item in behavior for scope in item["scopes"]]
    accepted = [revision for item in behavior for revision in item["accepted_revisions"]]
    result["objective_reentry"] = {"cases": sum(bool(item["attempts"]) for item in behavior),
        "attempts": sum(len(item["attempts"]) for item in behavior),
        "retained_scopes": len(scopes), "head_adoptions": len(accepted),
        "representation_revised": sum(bool(attempt.get("representation_revised"))
            for item in behavior for attempt in item["attempts"]),
        "active_requests": sum(item["active_requests"] for item in behavior),
        "active_observations": sum(item["active_observations"] for item in behavior),
        "shared_observations": sum(item["shared_observations"] for item in behavior),
        "validation_rows": sum(scope["validation_rows"] for scope in scopes),
        "comparison_rows": sum(scope["comparison_rows"] for scope in scopes),
        "statuses": dict(Counter(scope["status"] for scope in scopes))}
    result["terminal_repeat_actions"] = sum(
        row["costs"].get("objective_diagnostics", {}).get("terminal_repeat_actions", 0) for row in rows)
    result["native_value_precision_evaluations"] = sum(
        action.get("value_precision_evaluations", 0) for row in rows for action in row["actions"])
    result["native_value_precision_seconds"] = sum(
        action.get("value_precision_seconds", 0.) for row in rows for action in row["actions"])
    result["compute_including_comparisons"] = (result["constraint_recovery_compute"]
        + result["objective_research_compute"] + result["objective_replacement_comparison_rows"]
        + retention_work["comparison_rows"])
    return result


def summarize(spec, output, cases, baseline, current, refs, actual, phase):
    comparisons, grouped_metrics, changed = {}, {}, []
    groups = {"overall": cases, **{cohort: [case for case in cases if case["cohort"] == cohort]
              for cohort in dict.fromkeys(case["cohort"] for case in cases)}}
    for group, selected in groups.items():
        comparison = validation.reuse.comparison(selected, baseline, current)
        comparisons[group] = {key: comparison[key] for key in (
            "feasibility_first", "prediction_loss", "paid_trajectories_changed", "matched_complete_kg")}
        comparisons[group]["common_feasible_objective"] = validation.active.common_feasible_objectives(comparison["cases"])
        grouped_metrics[group] = {arm: metrics([mapping[case["case_id"]] for case in selected])
                                 for arm, mapping in ((BASELINE, baseline), (METHOD, current))}
        if group == "overall":
            changed = [row for row in comparison["cases"] if row["paid_trajectory_changed"] or any(
                abs(row[field][1] - row[field][0]) > 1e-12 for field in
                ("objective", "objective_gp_true_mse", "constraint_gp_true_mse"))]
            by_id = {case["case_id"]: case for case in selected}
            for row in changed:
                case = by_id[row["case_id"]]
                row.update(split=case["split"], cohort=case["cohort"])
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
        "metrics_by_cohort": grouped_metrics, "changed_cases": changed,
        "reentry_cases": [reentry_behavior(current[case["case_id"]]) for case in cases
                          if (current[case["case_id"]].get("objective_research") or {}).get("reentry_attempts")],
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
    if not (Path(spec["frozen_method_source"]) / "objective_reentry.py").exists():
        raise ValueError("freeze objective_reentry.py with the method before executing either arm")
    modules = dict(zip((BASELINE, METHOD), validation.arm_modules(spec).values()))
    study.controlled_problem = validation.active.framework.goal.controlled_problem
    module = modules[BASELINE][0]
    study.Program = importlib.import_module(f"{module.__package__}.language").Program
    study.SCKGEnvironment = module.SCKGEnvironment
    research = retention.measurements(spec, output, new_cases)
    fresh = [case for case in cases if case["cohort"] == "fresh27"]
    if fresh:
        research.update(study.acquire_research(spec, output, fresh))
    current, actual, refs = {}, [], []
    for case in cases:
        key = case["case_id"]
        for arm in (BASELINE, METHOD):
            if case["cohort"] != "fresh27" and (arm == BASELINE or key in spec["reused_cases"]):
                row = baseline[key]
                ref = {**old_refs[key], "arm": arm, "run_origin": "reused_actual_v051_reference"}
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                ref = {"case_id": key, "arm": arm, "case_record": str((output / "runs" / f"{key}_{arm}.json").resolve()),
                       "run_origin": "actual_v052_study"}
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
    spec, baseline, refs, research = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": len(spec["affected_cases"]),
            "controls": len(CONTROLS), "reused": len(spec["reused_cases"]), "development": len(DEVELOPMENT),
            "fresh": 27, "shared_historical_initial_records": len(research),
            "expected_optimizer_runs": spec["expected_new_optimizer_runs"],
            "expected_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()

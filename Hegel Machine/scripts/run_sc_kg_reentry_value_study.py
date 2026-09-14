#!/usr/bin/env python3
"""Compete rejected-hypothesis experiments with native actions at equal call budget."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path

import run_sc_kg_objective_reentry_study as previous

study, validation, passive = previous.study, previous.validation, previous.passive
retention = previous.retention
reentry_behavior = previous.reentry_behavior
BASELINE, METHOD = "reentry_value_baseline_v052", "reentry_value_v053"
FRESH_SEEDS = (1743012149, 1743112183, 1743212207)
CONTROLS = previous.CONTROLS
DEVELOPMENT = (
    "independent_goal_1_mechanism_change_noise_0.03",
    "completion_value_1_mechanism_change_noise_0.03",
    "confirmation_3_mechanism_change_noise_0.03",
    "joint_evidence_3_outlier_then_change_noise_0.03",
    "objective_validation_3_outlier_then_change_noise_0.03",
    "objective_exploration_1_stable_state_policy_noise_0.00",
    "research_scope_3_stable_rzdt2_noise_0.00",
    "objective_reentry_1_mechanism_change_noise_0.03",
    *CONTROLS,
)


def references(source, selected=None):
    saved = study.read(source / "summary.json")
    refs = {ref["case_id"]: {**ref, "arm": BASELINE, "source_arm": ref["arm"]}
            for ref in saved["result_references"] if ref["arm"] == saved["protocol"]["arms"][1]}
    if len(refs) != 544:
        raise ValueError("the completed544 v0.52 references are required")
    wanted = set(refs) if selected is None else set(selected)
    baseline = {}
    origins = {ref["original_result"] for key, ref in refs.items()
               if key in wanted and not ref.get("case_record")}
    # Resolve the recorded direct source once; do not reload every intervening study.
    for origin in origins:
        needed = {key for key, ref in refs.items() if key in wanted and not ref.get("case_record")
                  and ref["original_result"] == origin}
        rows = study.read(origin)["results"]
        baseline.update({row["case_id"]: row for row in rows
                         if row["case_id"] in needed and row["arm"] == validation.active.ARMS[0]})
        del rows
    for key in wanted:
        if refs[key].get("case_record"):
            baseline[key] = study.read(refs[key]["case_record"])
    if set(baseline) != wanted:
        raise ValueError("recorded baseline references did not resolve the selected cases")
    return baseline, refs


def prepare(source, output, variant=None):
    original = study.read(source / "protocol.json")
    brief = study.read(source / "result_brief.json")
    entries = [row for row in brief["reentry_cases"] if row["scopes"] or row["accepted_revisions"]]
    affected = [row["case_id"] for row in entries]
    refuted = [row["case_id"] for row in brief["reentry_cases"] if row["case_id"] not in affected]
    if (len(original["cases"]), len(affected), len(refuted)) != (544, 8, 3):
        raise ValueError("the recorded544 cases, eight planning entries and three refuted cases differ")
    if set(DEVELOPMENT) != set(affected) | set(CONTROLS) or set(affected) & set(CONTROLS):
        raise ValueError("development must contain the eight actual entries and five unchanged controls")
    cases = deepcopy(original["cases"])
    for case in cases:
        if case["cohort"] == "fresh27":
            case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
            name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
            case["evaluation_origin"] = str((source / "evaluation" / name).resolve())
            case["cohort"] = "historical_v052_27"
    templates = [case for case in cases if case["task"]["seed"] == original["fresh_seeds"][0]]
    used = {case["task"]["seed"] for case in cases} | set(retention.replacement.previous.FRESH_SEEDS)
    if len(templates) != 9 or set(FRESH_SEEDS) & used:
        raise ValueError("confirmation requires three unused seeds and nine existing profiles")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case.pop("initial_research_origin")
            case.pop("evaluation_origin")
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"reentry_value_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"reentry_value_{index}_{problem}_noise_{noise:.2f}",
                cohort="fresh27", split="confirmation")
            fresh.append(case)
    reused = [case["case_id"] for case in cases if case["case_id"] not in DEVELOPMENT]
    cases.extend(fresh)
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    spec.update(experiment="native_value_competition_for_reentered_objective_experiments", cases=cases,
        arms=[BASELINE, METHOD], source_reentry_study=str(source.resolve()),
        frozen_baseline_source=str((output / "baseline_source_v052").resolve()),
        frozen_method_source=str((output / "method_source_v053").resolve()),
        affected_cases=affected, unchanged_controls=list(CONTROLS), reused_cases=reused,
        refuted_unchanged_cases=refuted, development_cases=list(DEVELOPMENT),
        fresh_cases=[case["case_id"] for case in fresh], fresh_seeds=list(FRESH_SEEDS),
        expected_new_optimizer_runs=67, expected_new_online_calls=2144,
        expected_new_initial_calls=384, expected_new_evaluation_calls=1920,
        expected_new_physical_calls=4448,
        method="After an existing objective re-entry planning step, propose its distinguishing control "
            "without registering a pending physical request. Preserve the ordinary native candidate pool "
            "and append that control only if it is absent. Compare all actions using the native objective "
            "and constraint fantasy calculation; register the objective request only if its point is "
            "actually selected. Otherwise share the later native action's paid label with the hypothesis. "
            "In the final four calls, reuse the existing bounded512-fantasy precision procedure when "
            "single-fantasy scores all clip to zero. No artificial disagreement reward is added. "
            "Retain the v0.52 candidate, representation, negative-evidence and head-adoption rules.",
        selection="Run all eight actual re-entry planning paths, including both diagnosed useful "
            "hypotheses, both representation revisions, the previous fresh entry and the two regression "
            "cases. Add the same five no-entry controls, including rejection at the final label. "
            "The three firmly refuted candidates do not enter planning and remain reusable.",
        reuse_rule="The intervention starts only when an actual re-entry scope requests experiment "
            "planning. Reuse531 prior paths lacking that entry, excluding the five executed controls. "
            "Prior outcomes are previously executed v0.52 evidence, not new v0.53 trajectories.",
        accounting="Thirteen historical method runs and54 fresh paired runs cost2144 online calls. "
            "Fresh shared acquisition costs384 initial and1920 evaluator calls, totaling4448 physical. "
            "Development13 costs416 online calls and is included once. Each arm receives32 initial and32 "
            "online calls. Reuse old initialization, evaluator and baseline outcomes. Report controller "
            "compute, prediction/comparison rows and native KG timing separately; controller compute "
            "does not represent total computation. No additional vanilla or CodeAgent runs.",
        source_loading="Freeze the current live v0.52 twelve modules directly; freeze the complete v0.53 "
            "package before development. The intervention modifies objective_research.py and "
            "sc_kg_environment.py. Frozen packages determine both historical and fresh arms.",
        limitations="The existing544 cases are development evidence. The independently predeclared "
            "fresh27 run only after development is promising. Prior v0.52 fresh gain came from one "
            "actual re-entry path, and no re-entered hypothesis was installed. Preserve terminal "
            "regressions and either output's prediction regressions. Historical wall times are not "
            "simultaneous benchmarks. Existing missing measured objective evaluator labels remain missing.")
    if variant is not None:
        spec.update(variant)
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved re-entry value protocol differs")
    if not path.exists():
        study.write(path, spec)
    study.write(output / "entry_inventory.json", {"source_cases": 544,
        "affected": [{key: row[key] for key in ("case_id", "attempts", "active_observations",
            "shared_observations", "scopes", "accepted_revisions")} for row in entries],
        "refuted_unchanged": refuted, "controls": list(CONTROLS), "reused": reused})
    selected = [case for case in cases if case["case_id"] in DEVELOPMENT]
    research = retention.measurements(spec, output, selected)
    return spec, research


def metrics(rows):
    result = previous.metrics(rows)
    diagnostics = [row["costs"].get("objective_diagnostics", {}) for row in rows]
    offers = [item for state in diagnostics for item in state.get("reentry_competitions", [])]
    comparisons = [item for state in diagnostics for item in state.get("reentry_value_comparisons", [])]
    result["reentry_value_competition"] = {
        "offers": len(offers), "offers_already_native": sum(item["already_native"] for item in offers),
        "native_candidates": sum(len(item["native_points"]) for item in offers),
        "value_comparisons": len(comparisons),
        "candidate_score_rows": sum(len(item["candidate_points"]) for item in comparisons),
        "single_fantasy_all_clipped_zero": sum(not any(item["clipped_scores"]) for item in comparisons)}
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
        "actual_new_physical_calls": initial + online + evaluator,
        "batch_physical_calls": spec.get("prior_variant_physical_calls", 0) + initial + online + evaluator,
        "comparisons": comparisons,
        "metrics_by_cohort": grouped_metrics, "changed_cases": changed,
        "reentry_cases": [reentry_behavior(current[case["case_id"]]) for case in cases
                          if (current[case["case_id"]].get("objective_research") or {}).get("reentry_attempts")],
        "competition_cases": [{"case_id": current[case["case_id"]]["case_id"],
            "competitions": current[case["case_id"]]["costs"]["objective_diagnostics"]["reentry_competitions"],
            "value_comparisons": current[case["case_id"]]["costs"]["objective_diagnostics"].get("reentry_value_comparisons", [])}
            for case in cases if current[case["case_id"]]["costs"].get("objective_diagnostics", {}).get("reentry_competitions")],
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
            controls = spec.get("existing_control_records", {})
            if arm == METHOD and key in controls:
                row = study.read(controls[key])
                ref = {"case_id": key, "arm": arm, "case_record": controls[key],
                       "source_arm": row["arm"], "run_origin": "reused_executed_v053_control"}
            elif case["cohort"] != "fresh27" and (arm == BASELINE or key in spec["reused_cases"]):
                row = baseline[key]
                ref = {**old_refs[key], "arm": arm, "run_origin": "reused_actual_v052_reference"}
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                ref = {"case_id": key, "arm": arm, "case_record": str((output / "runs" / f"{key}_{arm}.json").resolve()),
                       "run_origin": "actual_v053_study"}
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
    spec, research = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": len(spec["affected_cases"]),
            "controls": len(CONTROLS), "reused": len(spec["reused_cases"]), "development": len(DEVELOPMENT),
            "fresh": 27, "shared_historical_initial_records": len(research),
            "expected_optimizer_runs": spec["expected_new_optimizer_runs"],
            "expected_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        selected = None if args.phase == "all" else spec["development_cases"]
        baseline, refs = references(args.source_study, selected)
        run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()

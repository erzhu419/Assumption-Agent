#!/usr/bin/env python3
"""Compare frozen candidates with the incumbent installed after a scope rebuild."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import run_sc_kg_objective_validation_passive_completion as passive


validation, study = passive.validation, passive.study
BASELINE, METHOD = "objective_passive_v048", "objective_incumbent_v049"
FRESH_SEEDS = (1743011693, 1743111709, 1743211727)
CONTROLS = (
    "value_precision_3_mechanism_change_noise_0.03",
    "value_precision_3_objective_only_change_noise_0.03",
    "research_value_3_stable_state_policy_noise_0.00",
    "research_value_3_stable_rzdt2_noise_0.00",
    "decision_single_outlier_2_state_policy_noise_0.03",
    "research_scope_3_objective_only_change_noise_0.03",
)
DEVELOPMENT = (
    "completion_value_2_mechanism_change_noise_0.00",
    "active_confirmation_3_mechanism_change_noise_0.03",
    "terminal_value_2_mechanism_change_noise_0.03",
    "forward_1_mechanism_change_noise_0.00",
    "forward_2_mechanism_change_noise_0.03",
    "objective_validation_1_mechanism_change_noise_0.03",
    *CONTROLS,
)


def references(source):
    original, rows, refs = passive.references(source)
    baseline, baseline_refs = {}, {}
    for ref in study.read(source / "passive_completion/summary.json")["result_references"]:
        if ref["arm"] != passive.PASSIVE:
            continue
        key = ref["case_id"]
        baseline[key] = study.read(ref["case_record"]) if ref["case_record"] else rows[passive.FULL][key]
        baseline_refs[key] = ref
    if len(baseline) != len(original["cases"]):
        raise ValueError("the completed actual/reused v0.48 default references are required")
    return original, baseline, baseline_refs


def rebuild_entries(row):
    state = row.get("objective_research") or {}
    spans = []
    for attempt in state.get("scope_attempts", []):
        if attempt.get("candidate"):
            end = max((item["global_index"] + 1 for item in attempt.get("validation", [])), default=64)
            spans.append(("completed", attempt["candidate"]["training_end"] + 1, end))
    current = state.get("scope_research") or {}
    if current.get("candidate"):
        spans.append(("validating", current["candidate"]["training_end"] + 1, 64))
    for accepted in row.get("objective_scope_revisions", []):
        spans.append(("adopted", accepted["model"]["training_end"] + 1, accepted["stage"]))
    return [{"stage": revision["stage"], "candidate_status": status,
             "frozen_stage": start, "end_stage": end}
            for revision in row.get("scope_revisions", [])
            if revision.get("preserved_objective_head") is False
            for status, start, end in spans if start <= revision["stage"] <= end]


def prepare(source, output):
    original, baseline, refs = references(source)
    cases = deepcopy(original["cases"])
    affected = [case["case_id"] for case in cases if rebuild_entries(baseline[case["case_id"]])]
    if len(affected) != 111 or set(CONTROLS) & set(affected):
        raise ValueError("the predeclared 111 rebuild entries and six no-entry controls differ")
    for case in cases:
        if case["cohort"] == "fresh27":
            case["cohort"] = "historical_v048_27"
    templates = [case for case in cases if case["task"]["seed"] == validation.FRESH_SEEDS[0]]
    if len(templates) != 9 or set(FRESH_SEEDS) & {case["task"]["seed"] for case in cases}:
        raise ValueError("fresh confirmation requires nine profiles and three unused seeds")
    fresh = []
    for index, seed in enumerate(FRESH_SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case.update(case_id=f"objective_incumbent_{index}_{case['profile']}_noise_{noise:.2f}",
                research_id=f"objective_incumbent_{index}_{problem}_noise_{noise:.2f}", cohort="fresh27")
            fresh.append(case)
    cases.extend(fresh)
    reused = [case["case_id"] for case in cases if case["cohort"] != "fresh27"
              and case["case_id"] not in set(affected) | set(CONTROLS)]
    spec = {key: deepcopy(original[key]) for key in (
        "sc_kg_project_root", "research_config", "total_calls_per_arm", "change_after_new_calls",
        "optimizer_seed_offset", "optimizer_settings", "policy", "revision", "evaluation")}
    spec.update(experiment="current_incumbent_objective_comparison", cases=cases,
        arms=[BASELINE, METHOD], source_validation_study=str(source.resolve()),
        historical_measurement_source=original["source_study"],
        frozen_baseline_source=str((output / "baseline_source_v048").resolve()),
        frozen_method_source=str((output / "method_source_v049").resolve()),
        affected_cases=affected, unchanged_controls=list(CONTROLS), reused_cases=reused,
        development_cases=list(DEVELOPMENT), fresh_cases=[case["case_id"] for case in fresh],
        fresh_seeds=list(FRESH_SEEDS), expected_new_optimizer_runs=171,
        expected_new_online_calls=5472, expected_new_initial_calls=384,
        expected_new_evaluation_calls=1920, expected_new_physical_calls=7776,
        method="After an actual constraint-driven objective GP rebuild, preserve the frozen candidate "
            "and past evidence but begin its comparison with the new incumbent using four later labels. "
            "Withdraw same-label support before installation; leave ordinary updates and preserved heads unchanged. "
            "Later validation remains passive and the original empirical adoption threshold remains unchanged.",
        selection="All111 existing paths with a frozen objective candidate at an actual objective GP rebuild, "
            "including same-stage first-four adoption, plus six bounded unchanged controls. Selection uses "
            "recorded branch entry, not outcome direction. Initial twelve-case development includes old "
            "cross-rebuild adoptions, a same-stage adoption, retained candidates, and all six controls.",
        reuse_rule="Reuse319 v0.48 paths whose recorded candidate lifetime contains no non-preserved objective "
            "head rebuild, and which are not the six actual controls. The method is otherwise unchanged. "
            "Completed-at-the-same-stage candidates are included conservatively. Saved v0.48 references "
            "include the previously validated passive/full no-entry reuse; they are not new executions.",
        accounting="117 historical v0.49 runs plus54 fresh paired runs cost5472 online calls. "
            "Historical initial/evaluator data and baseline outcomes are reused. Fresh shared acquisition "
            "cost384 initial and1920 evaluation calls:7776 new physical calls. Every arm has32 initial "
            "and32 online calls. No new vanilla or CodeAgent runs.",
        limitations="Historical436 are development evidence. The27 newly declared seed/profile cases are "
            "fresh for this method. Historical timing is not a simultaneous runtime comparison. "
            "Missing historical noisy objective evaluator observations remain missing; no truth substitution.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the prepared incumbent-comparison protocol differs")
    if not path.exists():
        study.write(path, spec)
    study.write(output / "entry_inventory.json", {"cases": 436, "affected": [
        {"case_id": key, "entries": rebuild_entries(baseline[key])} for key in affected],
        "controls": list(CONTROLS), "reused": reused})
    return spec, baseline, refs


def historical_measurements(spec, output, cases):
    source = Path(spec["source_validation_study"])
    research, refs = {}, {"research": {}, "evaluation": {}}
    for case in cases:
        if case["cohort"] == "fresh27":
            continue
        key = case["research_id"]
        if key not in research:
            root = source if case["cohort"] == "historical_v048_27" else Path(spec["historical_measurement_source"])
            path = root / "research" / f"{key}.json"
            row = study.read(path)
            if not row.get("optimizer_initial_environment_state") or (
                row["result"]["costs"]["observation_cost"] + row.get("supplemental_initial_calls", 0) != 32):
                raise ValueError(f"missing paid32-row initialization: {key}")
            research[key] = row
            refs["research"][key] = str(path.resolve())
        name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        if name not in refs["evaluation"]:
            path = source / "evaluation" / name
            evaluation = study.read(path)
            if evaluation["integer_points"] != study.evaluation_controls(spec, case):
                raise ValueError("saved evaluation controls differ")
            destination = output / "evaluation" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
            refs["evaluation"][name] = str(path.resolve())
    study.write(output / "measurement_references.json", refs)
    return research


def summarize(spec, output, cases, baseline, current, result_refs, phase):
    groups = {"overall": cases, **{cohort: [case for case in cases if case["cohort"] == cohort]
              for cohort in dict.fromkeys(case["cohort"] for case in cases)}}
    comparisons, metrics, changed = {}, {}, []
    for group, selected in groups.items():
        result = validation.reuse.comparison(selected, baseline, current)
        comparisons[group] = {key: result[key] for key in (
            "feasibility_first", "prediction_loss", "paid_trajectories_changed", "matched_complete_kg")}
        comparisons[group]["common_feasible_objective"] = validation.active.common_feasible_objectives(result["cases"])
        metrics[group] = {arm: passive.metrics([mapping[case["case_id"]] for case in selected])
                         for arm, mapping in ((BASELINE, baseline), (METHOD, current))}
        if group == "overall":
            changed = [row for row in result["cases"] if row["paid_trajectory_changed"] or any(
                abs(row[field][1] - row[field][0]) > 1e-12 for field in
                ("objective", "objective_gp_true_mse", "constraint_gp_true_mse"))]
    paths = {ref["case_record"] for ref in result_refs if ref.get("run_origin") == "actual_v049_study"}
    actual = [study.read(path) for path in paths]
    fresh_research = [study.read(path) for path in (output / "research").glob("*.json")]
    initial = sum(row.get("actual_new_research_calls", 0) + row.get("supplemental_initial_calls", 0)
                  for row in fresh_research)
    fresh_seeds = set(spec["fresh_seeds"])
    evaluator = sum(study.read(path)["simulator_calls"] for path in (output / "evaluation").glob("*.json")
                    if study.read(path)["task"]["seed"] - spec["evaluation"]["seed_offset"] in fresh_seeds)
    online = sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual)
    resolutions, pending = Counter(), []
    for row in current.values():
        state = row.get("objective_research") or {}
        for attempt in state.get("scope_attempts", []):
            if attempt.get("comparison_rebases"):
                resolutions[attempt["status"]] += 1
        for accepted in state.get("scope_revisions", []):
            if accepted.get("comparison_rebases"):
                resolutions["adopted_after_current_comparison"] += 1
        scope = state.get("scope_research") or {}
        if scope.get("comparison_rebases"):
            resolutions[scope["status"]] += 1
            pending.append({"case_id": row["case_id"], "status": scope["status"],
                "rebuild_stage": scope["comparison_rebases"][-1]["stage"],
                "current_labels": len(scope["validation"]) - scope["comparison_start"],
                "current_target": scope["validation_target"] - scope["comparison_start"]})
    brief = {"protocol": str((output / "protocol.json").resolve()), "phase": phase,
        "cases": len(cases), "actual_new_optimizer_runs": len(actual), "actual_new_online_calls": online,
        "actual_new_initial_calls": initial, "actual_new_evaluation_calls": evaluator,
        "actual_new_physical_calls": initial + evaluator + online, "comparisons": comparisons,
        "metrics_by_cohort": metrics, "changed_cases": changed,
        "current_comparison_resolutions": dict(resolutions), "pending_comparisons": pending,
        "rebuild_events": dict(Counter(event.get("action") for row in current.values()
            for event in (row.get("objective_research") or {}).get("events", [])
            if event.get("action") == "restart_objective_comparison")), "limitations": spec["limitations"]}
    prefix = "replay_" if phase == "replay" else ""
    study.write(output / f"{prefix}result_brief.json", brief)
    study.write(output / f"{prefix}summary.json", {"protocol": spec, "result_references": result_refs})
    print(json.dumps({"phase": phase, "cases": len(cases), "new_physical_calls": brief["actual_new_physical_calls"],
        "comparison": comparisons["overall"]["feasibility_first"],
        "prediction_loss": comparisons["overall"]["prediction_loss"]}), flush=True)


def run(spec, output, baseline, baseline_refs, phase):
    cases = spec["cases"] if phase == "all" else [next(case for case in spec["cases"]
            if case["case_id"] == key) for key in spec["development_cases"]]
    new_cases = [case for case in cases if case["case_id"] not in spec["reused_cases"]]
    modules = dict(zip((BASELINE, METHOD), validation.arm_modules(spec).values()))
    study.controlled_problem = validation.active.framework.goal.controlled_problem
    module = modules[BASELINE][0]
    study.Program = importlib.import_module(f"{module.__package__}.language").Program
    study.SCKGEnvironment = module.SCKGEnvironment
    research = historical_measurements(spec, output, new_cases)
    fresh = [case for case in cases if case["cohort"] == "fresh27"]
    if fresh:
        research.update(study.acquire_research(spec, output, fresh))
    current, actual, result_refs = {}, [], []
    for case in cases:
        key = case["case_id"]
        for arm in (BASELINE, METHOD):
            if case["cohort"] != "fresh27" and (arm == BASELINE or key in spec["reused_cases"]):
                row = baseline[key]
                ref = {**baseline_refs[key], "arm": arm, "run_origin": "reused_actual_v048_reference"}
            else:
                study.ARMS = (arm,)
                row = study.optimize(spec, output, [case], research, arm_modules=modules)[0]
                path = output / "runs" / f"{key}_{arm}.json"
                ref = {"case_id": key, "arm": arm, "case_record": str(path.resolve()),
                       "run_origin": "actual_v049_study"}
                actual.append(row)
            (baseline if arm == BASELINE else current)[key] = row
            result_refs.append(ref)
    study.evaluate(spec, output, new_cases, actual,
                   environment_factory=validation.active.framework.goal.evaluation_environment)
    for row in actual:
        row["run_origin"] = "actual_v049_study"
        study.write(output / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    summarize(spec, output, cases, baseline, current, result_refs, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    spec, baseline, refs = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "affected": 111, "controls": 6,
            "reused": 319, "fresh": 27, "development_cases": 12,
            "new_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        run(spec, args.output, baseline, refs, args.phase)


if __name__ == "__main__":
    main()

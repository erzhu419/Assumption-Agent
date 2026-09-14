#!/usr/bin/env python3
"""Research objective decisions jointly with current constraint knowledge."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

import run_sc_kg_goal_recompetition_study as prior


previous, study, unknown = prior.previous, prior.study, prior.unknown
BASELINE, METHOD = "hegel_v056", "hegel_constraint_knowledge"
SOURCE = study.ROOT / "artifacts/sc_kg_dual_value_20260912/revision3"
DEFAULT_SELECTION = SOURCE.parent / "default_selection.json"
SEEDS = (2545015009, 2545115033, 2545215059)
SMOKE = ("active_noise_2_mechanism_change_noise_0.03",
         "goal_completion_3_center_change",
         "goal_completion_2_center_change",
         "joint_identification_2_center_change",
         "dual_value_2_center_change")


def prepare(output):
    selection = study.read(DEFAULT_SELECTION)
    original = study.read(SOURCE / "protocol.json")
    confirmed = study.read(SOURCE / "fresh_confirmation/protocol.json")
    groups = deepcopy(original["groups"])
    groups["unknown_dual_value_confirmation"] = {
        "source_protocol": str((SOURCE / "fresh_confirmation/protocol.json").resolve()),
        "cases": deepcopy(confirmed["cases"])}
    for group, settings in groups.items():
        for case in settings["cases"]:
            case["split"] = "development"
            prior_output = SOURCE
            if group == "unknown_dual_value_confirmation":
                prior_output = SOURCE / "fresh_confirmation"
                case["initial_research_origin"] = str((prior_output / "research" / f"{case['research_id']}.json").resolve())
                case["evaluation_origin"] = str((prior_output / "evaluation" / previous.evaluation_name(case)).resolve())
                case["native_record"] = str((prior_output / "runs" / f"{case['case_id']}_{unknown.NATIVE}.json").resolve())
            case["baseline_record"] = selection["default_records"][case["case_id"]]
            case["previous_method_record"] = case["baseline_record"]
            case["baseline_execution"] = "actual_selected_v056_record"
    fresh = []
    for index, (old_seed, seed) in enumerate(zip(confirmed["seeds"], SEEDS), 1):
        for old in confirmed["cases"]:
            if old["task"]["seed"] != old_seed:
                continue
            case = deepcopy(old)
            case["task"]["seed"] = seed
            case.update(case_id=f"constraint_knowledge_{index}_{case['profile']}",
                research_id=f"constraint_knowledge_{index}_{case['task']['problem']}",
                cohort="fresh9", split="confirmation")
            fresh.append(case)
    spec = {"experiment": "joint_objective_decision_and_constraint_knowledge",
        "groups": groups, "smoke_cases": list(SMOKE), "arms": [BASELINE, METHOD],
        "source_v056": selection["default_method_source"],
        "baseline_selection": str(DEFAULT_SELECTION.resolve()),
        "baseline_reuse_evidence": "All79 baseline trajectories are actual executions of selected0.56.0 "
            "listed in default_records. This study uses their actual observations, decisions and costs; "
            "no behavior-equivalence substitution or new baseline optimizer execution is claimed.",
        "method_source": str((output / "method_source").resolve()),
        "fresh_seeds": list(SEEDS), "fresh_cases": fresh,
        "development": {"cases": 79, "new_optimizer_runs": 79, "initial_calls": 0,
            "online_calls": 2528, "evaluation_calls": 0, "physical_calls": 2528,
            "smoke_calls_included_once": 160},
        "confirmation": {"cases": 9, "arms": [unknown.NATIVE, BASELINE, METHOD],
            "new_optimizer_runs": 27, "initial_calls": 192, "online_calls": 864,
            "evaluation_calls": 1152, "physical_calls": 2208},
        "method": "Start from selected default0.56.0. Jointly consider objective decisions and "
            "current constraint knowledge using paid observations and current model predictions. "
            "Preserve prospective candidate validation,32 paid initialization plus32 online calls, "
            "the20000 objective-research operation limit and all existing optimizer and constraint "
            "research limits. The completed method description and source will be frozen before "
            "new observations are acquired.",
        "selection": "The70 preceding development cases and9 completed dual-value confirmation cases "
            "form79 development and retention cases. The five named smoke cases cover the largest "
            "constraint-prediction regression, another constraint-prediction regression, delayed "
            "objective adoption and two actual goal improvements. All66 unknown-goal cases retain "
            "native controls. All79 baseline records are actual selected0.56.0 trajectories. Count "
            "the five smoke cases once when extending to full79. The preceding107 optimizations and "
            "4768 physical calls are historical and excluded from this study's new-call ledger.",
        "accounting": "Development purchases79 times32 online vectors:2528. Independent9-case "
            "confirmation uses six shared32-vector initializations,27 optimizers times32 online calls, "
            "and nine128-vector final evaluations:2208. Initialization research uses frozen default0.56.0. "
            "Total if both phases complete:106 optimizer runs and4736 physical calls. Report controller "
            "operations, native KG and optimizer time separately; each paid vector produces both outputs.",
        "limitations": "The79 existing cases are development evidence. New9 cases remain unrun unless "
            "development supports continuing, and use the unchanged frozen method and seeds. Native "
            "SC-KG comparisons condition on shared paid initialization. Both-infeasible results are "
            "separate from improvements. Historical walltime is not a simultaneous throughput test; "
            "previous backend clock anomalies preclude inferring speed from those timings."}
    method_description = output / "method_description.txt"
    if method_description.exists():
        spec["method"] = method_description.read_text().strip()
    smoke_path = output / "smoke_cases.json"
    if smoke_path.exists():
        spec["smoke_cases"] = study.read(smoke_path)
        spec["development"]["smoke_calls_included_once"] = 32 * len(spec["smoke_cases"])
        spec["selection"] += " This revision additionally includes three observed V2 losses in its eight-case smoke subset."
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved constraint-knowledge protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def configure_arms():
    # Reuse the preceding study's result accounting with this study's fixed labels.
    prior.BASELINE, prior.METHOD = BASELINE, METHOD
    previous.METHOD = METHOD


def execute_development(spec, output, smoke=False):
    configure_arms()
    for group, settings in spec["groups"].items():
        cases = [case for case in settings["cases"]
                 if not smoke or case["case_id"] in spec["smoke_cases"]]
        if not cases:
            continue
        original = study.read(settings["source_protocol"])
        modules, evaluator = previous.load_method(spec, group)
        research = previous.reused_measurements(output, cases)
        rows = study.optimize(original, output, cases, research, arm_modules=modules)
        study.evaluate(original, output, cases, rows, environment_factory=evaluator)
        for row in rows:
            study.write(output / "runs" / f"{row['case_id']}_{METHOD}.json", row)
    summarize_development(spec, output)


def report_support(brief, output):
    unknown_cases = [case for case in brief["comparisons"] if case["group"] != "historical_retention"]
    paired = Counter()
    for group in brief["metrics"].values():
        paired.update(group["paired_feasibility_first"][BASELINE])
    summary = {"development_cases": brief["completed_cases"], "expected_development_cases": 79,
        "development_vs_v056": dict(paired),
        "unknown_cases": len(unknown_cases),
        "unknown_regret": {arm: mean(case["arms"][arm]["objective"]-.35 for case in unknown_cases)
            for arm in (BASELINE, METHOD, unknown.NATIVE)} if unknown_cases else {},
        "development_new_costs": brief["new_costs"],
        "source": str((output / "development_result_brief.json").resolve())}
    matched_work = {}
    for values in brief["metrics"].values():
        for arm, result in values["arms"].items():
            totals = matched_work.setdefault(arm, {"cases": 0, "objective_compute": 0,
                "constraint_compute": 0, "optimizer_seconds": 0., "native_kg_seconds": 0.})
            totals["cases"] += values["cases"]
            for key in ("objective_compute", "constraint_compute", "optimizer_seconds", "native_kg_seconds"):
                totals[key] += result["work"][key]
    summary["matched_work_and_timing"] = matched_work
    summary["matched_generalization"] = {
        arm: {metric: sum(group["cases"] * group["arms"][arm][metric]
            for group in brief["metrics"].values() if arm in group["arms"]) / work["cases"]
            for metric in ("objective", "objective_mse", "constraint_mse")}
        for arm, work in matched_work.items()}
    summary["timing_interpretation"] = "Historical timings are not simultaneous throughput comparisons. Research operation counts do not measure total optimizer work."
    promoted = [{"case_id": row["case_id"], **event} for row in brief["research_behavior"]
        for event in row["events"] if event["action"] == "select_interaction_from_shared_validation"]
    summary["future_supported_interaction_promotions"] = promoted
    summary["promoted_interaction_installations"] = [
        {"case_id": row["case_id"], **item} for row in brief["research_behavior"]
        for item in row["adoptions"] if item["interactions"] and any(
            event["case_id"] == row["case_id"] and event["scope_start"] == item["scope_start"]
            and event["stage"] <= item["stage"] for event in promoted)]
    fresh_path = output / "fresh_confirmation/result_brief.json"
    fresh = study.read(fresh_path) if fresh_path.exists() else None
    summary["actual_optimizer_runs"] = brief["optimizer_runs"] + (fresh["optimizer_runs"] if fresh else 0)
    summary["actual_physical_calls"] = brief["new_costs"]["physical_calls"] + (fresh["costs"]["physical_calls"] if fresh else 0)
    prior_path = output / "prior_attempt_accounting.json"
    attempted = study.read(prior_path) if prior_path.exists() else {}
    summary["cumulative_actual_optimizer_runs"] = summary["actual_optimizer_runs"] + attempted.get("completed_previous_optimizer_runs", 0)
    summary["cumulative_actual_physical_calls"] = summary["actual_physical_calls"] + attempted.get("completed_previous_physical_calls", 0)
    mechanism_path = output / "constraint_confirmation/result_brief.json"
    if mechanism_path.exists():
        mechanism = study.read(mechanism_path)
        summary["constraint_confirmation_metrics"] = mechanism["metrics"]
        summary["constraint_confirmation_costs"] = mechanism["costs"]
        summary["constraint_confirmation_source"] = str(mechanism_path.resolve())
        summary["actual_optimizer_runs"] += mechanism["optimizer_runs"]
        summary["actual_physical_calls"] += mechanism["costs"]["physical_calls"]
        summary["cumulative_actual_optimizer_runs"] += mechanism["optimizer_runs"]
        summary["cumulative_actual_physical_calls"] += mechanism["costs"]["physical_calls"]
    accounting_path = output / "final_accounting.json"
    if accounting_path.exists():
        summary["final_accounting"] = str(accounting_path.resolve())
    if fresh:
        summary["confirmation_metrics"] = fresh["metrics"]
        summary["confirmation_costs"] = fresh["costs"]
        summary["confirmation_source"] = str(fresh_path.resolve())
    study.write(output / "report_brief.json", summary)
    mechanism_rows = []
    for comparison in brief["comparisons"]:
        row = study.read(comparison["method_record"])
        objective = row.get("objective_research") or {}
        tail = [item for item in objective.get("requests", [])
                if item.get("reason", "").startswith("resolve_current_goal_")
                or item.get("reason") == "calibrate_current_goal_value"]
        shared = [item for item in objective.get("shared_outcomes", [])
                  if item["request"].get("purpose", "").startswith("objective_")]
        previous_path = Path(comparison["previous_method_record"])
        old = study.read(previous_path)
        baseline = study.read(comparison["baseline_record"])
        retention = row["costs"]["recovery_diagnostics"].get("constraint_retention", {})
        validations = [*objective.get("scope_attempts", []),
                       objective.get("scope_research") or {}]
        mechanism_rows.append({"case_id": row["case_id"], "arms": comparison["arms"],
            "constraint_knowledge": {**{key: retention.get(key, 0) for key in (
                "coefficient_fits", "coefficient_fit_records", "paid_design_fits", "paid_design_fit_records",
                "paid_design_unidentified", "paid_design_seconds", "prediction_rows", "component_updates", "comparison_rows")},
                "scopes": [{key: item.get(key) for key in ("stage", "scope_start", "component_names",
                    "initial_weights", "coefficient_dimension")} for item in retention.get("scopes", [])],
                "future_observations": retention.get("observations", [])},
            "previous_selected_objective": old["final"]["true_objective"],
            "baseline_candidate_adoptions": [{"stage": item["stage"], "scope_start": item["scope_start"]}
                for item in baseline.get("objective_scope_revisions", [])],
            "previous_selected_candidate_adoptions": [{"stage": item["stage"], "scope_start": item["scope_start"]}
                for item in old.get("objective_scope_revisions", [])],
            "terminal_comparisons": [{"stage": item["stage"], "features": item["features"],
                "reason": item["reason"],
                "expected_decision_gain": item.get("expected_decision_gain"),
                "validation_calls_needed": item.get("validation_calls_needed"),
                "value_offers": item.get("value_offers", [])}
                                     for item in tail],
            "own_objective_shared_validation": [{"global_index": item["context"]["global_index"],
                "purpose": item["request"]["purpose"],
                "scope_start": item["request"]["objective_scope_start"],
                "candidate_mean": item["request"]["candidate_mean"],
                "previous_mean": item["request"]["previous_mean"]} for item in shared],
            "validation_value_choices": [{"stage": item["stage"],
                "selected_reason": item["reason"], "features": item["features"],
                "expected_gain_per_call": item["validation_plan"]["expected_gain_per_call"]}
                for item in objective.get("requests", []) if item.get("validation_plan")],
            "final_affordable_rounds": [item for item in objective.get("events", [])
                if item.get("action") == "complete_available_objective_validation"],
            "candidate_adoptions": [{"stage": item["stage"], "scope_start": item["scope_start"]}
                                    for item in row.get("objective_scope_revisions", [])],
            "unresolved_validation": [{"scope_start": item.get("scope_start"),
                "phase": item.get("phase"), "validation_count": len(item.get("validation", []))}
                for item in validations if item.get("phase") in ("collecting", "validating")]})
    study.write(output / "development_mechanisms.json", {
        "cases": mechanism_rows,
        "terminal_comparison_calls": sum(len(row["terminal_comparisons"]) for row in mechanism_rows),
        "terminal_comparison_cases": sum(bool(row["terminal_comparisons"]) for row in mechanism_rows),
        "own_objective_shared_validation_rows": sum(len(row["own_objective_shared_validation"]) for row in mechanism_rows),
        "own_objective_shared_validation_cases": sum(bool(row["own_objective_shared_validation"]) for row in mechanism_rows),
        "unique_own_objective_shared_validation_rows": len({(row["case_id"], item["global_index"])
            for row in mechanism_rows for item in row["own_objective_shared_validation"]}),
        "shared_objective_action_counts": dict(Counter(item["purpose"] for row in mechanism_rows
            for item in row["own_objective_shared_validation"])),
        "candidate_adoptions": sum(len(row["candidate_adoptions"]) for row in mechanism_rows),
        "calibration_calls": sum(item["reason"] == "calibrate_current_goal_value"
            for row in mechanism_rows for item in row["terminal_comparisons"]),
        "repeat_value_calls": sum(item["reason"] == "resolve_current_goal_value"
            for row in mechanism_rows for item in row["terminal_comparisons"]),
        "validation_value_plans": sum(len(row["validation_value_choices"]) for row in mechanism_rows),
        "selected_validation_value_calls": sum(item["selected_reason"] == "validate_objective_scope"
            for row in mechanism_rows for item in row["validation_value_choices"]),
        "final_affordable_rounds": sum(len(row["final_affordable_rounds"]) for row in mechanism_rows),
        "constraint_coefficient_fits": sum(row["constraint_knowledge"]["coefficient_fits"] for row in mechanism_rows),
        "constraint_coefficient_fit_records": sum(row["constraint_knowledge"]["coefficient_fit_records"] for row in mechanism_rows),
        "constraint_paid_design_fits": sum(row["constraint_knowledge"]["paid_design_fits"] for row in mechanism_rows),
        "constraint_paid_design_fit_records": sum(row["constraint_knowledge"]["paid_design_fit_records"] for row in mechanism_rows),
        "constraint_paid_design_unidentified": sum(row["constraint_knowledge"]["paid_design_unidentified"] for row in mechanism_rows),
        "constraint_component_comparison_rows": sum(row["constraint_knowledge"]["comparison_rows"] for row in mechanism_rows),
        "constraint_future_observations": sum(len(row["constraint_knowledge"]["future_observations"]) for row in mechanism_rows),
        "actual_optimizer_runs": summary["actual_optimizer_runs"],
        "actual_physical_calls": summary["actual_physical_calls"]})


def summarize_development(spec, output):
    configure_arms()
    comparisons, metrics, actual, all_rows = [], {}, [], []
    for group, settings in spec["groups"].items():
        selected, matched = [], {BASELINE: [], METHOD: []}
        if group != "historical_retention":
            matched[unknown.NATIVE] = []
        for case in settings["cases"]:
            path = output / "runs" / f"{case['case_id']}_{METHOD}.json"
            if not path.exists():
                continue
            rows = {BASELINE: study.read(case["baseline_record"]), METHOD: study.read(path)}
            if "native_record" in case:
                rows[unknown.NATIVE] = study.read(case["native_record"])
            selected.append({"case_id": case["case_id"], "group": group, "profile": case["profile"],
                "method_execution": "current_revision",
                "baseline_execution": case["baseline_execution"],
                "baseline_record": case["baseline_record"],
                "previous_method_record": case["previous_method_record"],
                "method_record": str(path.resolve()),
                "arms": {arm: previous.values(row) for arm, row in rows.items()}})
            for arm, row in rows.items():
                matched[arm].append(row)
            all_rows.append(rows[METHOD])
            actual.append(rows[METHOD])
        if selected:
            comparisons.extend(selected)
            metrics[group] = prior.comparison_metrics(selected, matched)
    brief = {"protocol": str((output / "protocol.json").resolve()), "completed_cases": len(all_rows),
        "expected_cases": 79, "optimizer_runs": len(actual), "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [prior.behavior(row) for row in all_rows],
        "new_costs": {"initial_calls": 0, "evaluation_calls": 0,
            "physical_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual),
            **prior.work(actual)}, "limitations": spec["limitations"]}
    study.write(output / "development_result_brief.json", brief)
    report_support(brief, output)
    print(json.dumps({"completed_cases": len(all_rows), "optimizer_runs": len(actual),
        "new_costs": brief["new_costs"],
        "paired": {group: value["paired_feasibility_first"] for group, value in metrics.items()}}), flush=True)


def execute_confirmation(spec, output):
    configure_arms()
    target = output / "fresh_confirmation"
    original = study.read(spec["groups"]["unknown_dual_value_confirmation"]["source_protocol"])
    fresh = {key: deepcopy(original[key]) for key in ("sc_kg_project_root", "research_config",
        "total_calls_per_arm", "optimizer_seed_offset", "optimizer_settings", "policy", "revision",
        "evaluation", "change_after_new_calls", "task_formula", "task_information")}
    fresh.update(cases=spec["fresh_cases"], seeds=spec["fresh_seeds"], arms=spec["confirmation"]["arms"],
        frozen_baseline_source=spec["source_v056"], frozen_method_source=spec["method_source"],
        source_protocol=str((output / "protocol.json").resolve()), accounting=spec["confirmation"],
        method=spec["method"], limitations=spec["limitations"])
    path = target / "protocol.json"
    if path.exists() and study.read(path) != fresh:
        raise ValueError("the saved fresh constraint-knowledge confirmation differs")
    if not path.exists():
        study.write(path, fresh)
    previous.METHOD = BASELINE
    baseline, _ = previous.load_method({**spec, "method_source": spec["source_v056"]}, "unknown_constraint_knowledge_fresh_baseline")
    initial = study.Program, study.MetaPriorResearch, study.SCKGEnvironment, unknown.task.Environment
    previous.METHOD = METHOD
    current, evaluator = previous.load_method(spec, "unknown_constraint_knowledge_fresh_method")
    study.Program, study.MetaPriorResearch, study.SCKGEnvironment, unknown.task.Environment = initial
    study.ARMS = (BASELINE, METHOD)
    research = study.acquire_research(fresh, target, fresh["cases"])
    rows = study.optimize(fresh, target, fresh["cases"], research, arm_modules={**baseline, **current})
    natives = [unknown.run_native(fresh, target, case, research[case["research_id"]]) for case in fresh["cases"]]
    evaluations = study.evaluate(fresh, target, fresh["cases"], rows, environment_factory=evaluator)
    for row in natives:
        case = next(case for case in fresh["cases"] if case["case_id"] == row["case_id"])
        saved = evaluations[previous.evaluation_name(case)[:-5]]
        prediction = row["frozen_evaluation_predictions"]
        row["generalization"] = {"objective_gp_true_objective_means_mse": mean((a-b)**2 for a,b in
            zip(prediction["objective_gp"], saved["true_objective_means"])),
            "active_gp_true_constraint_means_mse": mean((a-b)**2 for a,b in
            zip(prediction["active_gp"], saved["true_constraint_means"]))}
    rows.extend(natives)
    comparisons = [{"case_id": case["case_id"], "profile": case["profile"],
        "arms": {row["arm"]: {**previous.values(row), "regret": row["final"]["true_objective"]-.35}
            for row in rows if row["case_id"] == case["case_id"]}} for case in fresh["cases"]]
    metrics = {}
    for profile in ("overall", *unknown.task.PROFILES):
        selected = [case for case in comparisons if profile == "overall" or case["profile"] == profile]
        keys = {case["case_id"] for case in selected}
        matched = {arm: [row for row in rows if row["arm"] == arm and row["case_id"] in keys] for arm in fresh["arms"]}
        metrics[profile] = prior.comparison_metrics(selected, matched)
        for arm in fresh["arms"]:
            metrics[profile]["arms"][arm]["regret"] = mean(case["arms"][arm]["regret"] for case in selected)
    for row in rows:
        study.write(target / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    initial_calls = sum(row["actual_new_initial_calls"] for row in research.values())
    evaluator_calls = sum(row["simulator_calls"] for row in evaluations.values())
    online_calls = sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows)
    brief = {"protocol": str(path.resolve()), "completed_cases": len(comparisons), "optimizer_runs": len(rows),
        "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [prior.behavior(row) for row in rows if row["arm"] != unknown.NATIVE],
        "costs": {"initial_calls": initial_calls, "online_calls": online_calls,
            "evaluation_calls": evaluator_calls, "physical_calls": initial_calls+online_calls+evaluator_calls,
            "initial_research_compute": sum(row["result"]["costs"]["compute_cost"] for row in research.values()),
            "online_by_arm": {arm: prior.work([row for row in rows if row["arm"] == arm]) for arm in fresh["arms"]}},
        "limitations": spec["limitations"]}
    study.write(target / "result_brief.json", brief)
    report_support(study.read(output / "development_result_brief.json"), output)
    print(json.dumps({"costs": brief["costs"], "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "smoke", "development", "confirmation", "summary", "report"), default="plan")
    args = parser.parse_args()
    spec = prepare(args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "development": spec["development"],
            "confirmation": spec["confirmation"], "fresh_seeds": spec["fresh_seeds"]}), flush=True)
    elif args.phase == "summary":
        summarize_development(spec, args.output)
    elif args.phase == "report":
        report_support(study.read(args.output / "development_result_brief.json"), args.output)
    elif args.phase == "confirmation":
        execute_confirmation(spec, args.output)
    else:
        execute_development(spec, args.output, smoke=args.phase == "smoke")


if __name__ == "__main__":
    main()

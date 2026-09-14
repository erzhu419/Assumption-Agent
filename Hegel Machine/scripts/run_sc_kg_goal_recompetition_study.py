#!/usr/bin/env python3
"""Reopen installed goal explanations, complete research, and remeasure boundaries."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

import run_sc_kg_goal_completion_study as previous


study, unknown = previous.study, previous.unknown
BASELINE, METHOD = "hegel_v053", "hegel_goal_recompetition"
SOURCE = study.ROOT / "artifacts/sc_kg_goal_completion_20260911"
SEEDS = (2343012791, 2343112817, 2343212843)
SMOKE = ("objective_reentry_1_mechanism_change_noise_0.03",
         "research_scope_3_objective_only_change_noise_0.03",
         "goal_completion_2_center_interaction_change")


def prepare(output, smoke_cases=None):
    path = output / "protocol.json"
    if smoke_cases is None:
        smoke_cases = study.read(path)["smoke_cases"] if path.exists() else SMOKE
    original = study.read(SOURCE / "protocol.json")
    confirmed = study.read(SOURCE / "fresh_confirmation/protocol.json")
    groups = deepcopy(original["groups"])
    groups["unknown_goal_confirmation"] = {
        "source_protocol": str((SOURCE / "fresh_confirmation/protocol.json").resolve()),
        "cases": deepcopy(confirmed["cases"])}
    for group, settings in groups.items():
        source = SOURCE / "fresh_confirmation" if group == "unknown_goal_confirmation" else SOURCE
        for case in settings["cases"]:
            case["split"] = "development"
            if group == "unknown_goal_confirmation":
                case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
                case["evaluation_origin"] = str((source / "evaluation" / previous.evaluation_name(case)).resolve())
                case["native_record"] = str((source / "runs" / f"{case['case_id']}_{unknown.NATIVE}.json").resolve())
            case["baseline_record"] = str((source / "runs" / f"{case['case_id']}_hegel_goal_completion.json").resolve())
            for field in ("previous_interaction_record", "v052_record"):
                case.pop(field, None)
    fresh = []
    for index, (old_seed, seed) in enumerate(zip(confirmed["seeds"], SEEDS), 1):
        for old in confirmed["cases"]:
            if old["task"]["seed"] != old_seed:
                continue
            case = deepcopy(old)
            case["task"]["seed"] = seed
            case.update(case_id=f"goal_recompetition_{index}_{case['profile']}",
                research_id=f"goal_recompetition_{index}_{case['task']['problem']}",
                cohort="fresh9", split="confirmation")
            fresh.append(case)
    spec = {"experiment": "installed_goal_recompetition_research_completion_and_boundary_remeasurement",
        "groups": groups, "smoke_cases": list(smoke_cases), "arms": [BASELINE, METHOD],
        "source_v053": str((SOURCE / "method_source").resolve()),
        "method_source": str((output / "method_source").resolve()),
        "fresh_seeds": list(SEEDS), "fresh_cases": fresh,
        "development": {"cases": 43, "new_optimizer_runs": 43, "initial_calls": 0,
            "online_calls": 1376, "evaluation_calls": 0, "physical_calls": 1376,
            "smoke_calls_included_once": 32 * len(smoke_cases)},
        "confirmation": {"cases": 9, "arms": [unknown.NATIVE, BASELINE, METHOD],
            "new_optimizer_runs": 27, "initial_calls": 192, "online_calls": 864,
            "evaluation_calls": 1152, "physical_calls": 2208},
        "method": "Build from the completed frozen0.53 thirteen-module package. Reopen a deployed "
            "goal representation when new evidence requires competition, prevent locally tiny CV "
            "error from blocking further research, and remeasure uncertain feasibility boundaries. "
            "Keep the original32-initial/32-online and research computation limits. Freeze the final "
            "implementation before any smoke observation.",
        "selection": "Reuse all34 development plus9 confirmation cases from goal completion. All43 "
            "are development evidence now. Run the named smoke cases first, then reuse those runs "
            "once in the complete43 if development continues. All30 unknown-goal cases also retain "
            "their existing native controls; the historical13 have actual0.53 references.",
        "accounting": "Only43 times32 online observations are new in development:1376. Independent "
            "confirmation uses six shared32-vector initializations and nine128-vector final "
            "evaluations, with native, frozen0.53 and the fixed new method each using32 online calls. "
            "Confirmation initialization research uses frozen0.53. Total if both phases complete3584. "
            "Controller operations, optimizer timing and native KG work are reported separately.",
        "limitations": "The43 existing cases are development and retention evidence. Fresh9 may run "
            "only after development is useful, without changing the frozen method. Comparisons with "
            "native SC-KG condition on the same paid initialization. Both-infeasible recommendations "
            "are reported separately. Historical wall time is not a simultaneous throughput benchmark."}
    if path.exists():
        # Earlier frozen protocols used the original three-case wording.
        spec["selection"] = study.read(path)["selection"]
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved goal-recompetition protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def execute_development(spec, output, smoke=False):
    previous.METHOD = METHOD
    for group, settings in spec["groups"].items():
        cases = [case for case in settings["cases"] if not smoke or case["case_id"] in spec["smoke_cases"]]
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


def work(rows):
    result = previous.work(rows)
    diagnostics = [row["costs"].get("objective_diagnostics", {}) for row in rows]
    rounds = [event for value in diagnostics for event in value.get("value_precision", {}).get("rounds", [])]
    noisy = [event for event in rounds if event.get("noise_evidence")]
    result.update(raw_gp_replacement_fits=sum(value.get("replacement_fits", 0) for value in diagnostics),
        raw_gp_replacement_prediction_rows=sum(value.get("replacement_prediction_rows", 0) for value in diagnostics),
        raw_gp_replacement_comparison_rows=sum(((row.get("objective_research") or {}).get(
            "replacement_research") or {}).get("comparison_rows", 0) for row in rows),
        value_precision_evaluations=sum(event.get("evaluations", 0) for event in rounds),
        boundary_noise_rounds=len(noisy),
        boundary_innovation_evaluations=sum(event.get("innovation_evaluations", 0) for event in noisy),
        boundary_scale_evaluations=sum(event.get("evaluations", 0) for event in noisy))
    for key in ("certification_calls", "certification_rows", "weight_updates", "component_evaluations", "seconds"):
        result[f"boundary_{key}"] = sum(value.get("boundary_noise", {}).get(key, 0) for value in diagnostics)
    return result


def comparison_metrics(comparisons, rows_by_arm):
    paired = {}
    for arm in rows_by_arm:
        if arm == METHOD:
            continue
        decisions = []
        for case in comparisons:
            old, new = case["arms"][arm], case["arms"][METHOD]
            if old["feasible"] != new["feasible"]:
                decisions.append("improved" if new["feasible"] else "worsened")
            elif not new["feasible"]:
                decisions.append("both_infeasible")
            else:
                delta = new["objective"] - old["objective"]
                decisions.append("improved" if delta < -1e-9 else "worsened" if delta > 1e-9 else "same")
        paired[arm] = dict(Counter(decisions))
    return {"cases": len(comparisons), "paired_feasibility_first": paired,
        "arms": {arm: {"feasible": sum(row["final"]["true_feasible"] for row in rows),
            **{key: mean(previous.values(row)[key] for row in rows if previous.values(row)[key] is not None)
               for key in ("objective", "objective_mse", "constraint_mse")},
            "work": work(rows)} for arm,rows in rows_by_arm.items()}}


def behavior(row):
    result = previous.compact_behavior(row)
    replacement = (row.get("objective_research") or {}).get("replacement_research") or {}
    attempts = replacement.get("attempts", [])
    gp_installs = [item for item in attempts if item.get("reason") == "supported_gp_replacement"]
    result["installed_goal_reopens"] = [dict(event) for event in replacement.get("events", [])
        if event.get("action") == "reopen_installed_objective_model"]
    result["raw_gp_installations"] = [{"stage": item["stage"], "scope_start": item["scope_start"],
        "future_validation_rows": len(item.get("validation", [])), "reason": item["reason"]}
        for item in gp_installs]
    result["gp_replacement_comparison_rows"] = replacement.get("comparison_rows", 0)
    result["pending_gp_replacement"] = {"scope_start": replacement.get("scope_start"),
        "confirmed": replacement.get("confirmed", False),
        "future_validation_rows": len(replacement.get("validation", []))}
    # The adapter applies a supported raw GP before any program installation at
    # the same stage. Historical program revisions alone do not identify the
    # deployed representation after the new competition path runs.
    installations = [(item["stage"], 0, "raw_gp") for item in gp_installs]
    installations.extend((item["stage"], 1, "learned_program")
                         for item in row.get("objective_scope_revisions", []))
    result["final_objective_representation_from_install_events"] = (
        max(installations)[2] if installations else "raw_gp")
    diagnostics = row["costs"].get("objective_diagnostics", {})
    result["raw_gp_replacement_work"] = {key: diagnostics.get(key, 0) for key in (
        "replacement_fits", "replacement_fit_records", "replacement_prediction_rows",
        "replacement_seconds", "replacement_install_fit_records", "replacement_install_seconds")}
    choices = {item["stage"]: item for item in diagnostics.get("terminal_choices", [])}
    result["boundary_noise_rounds"] = [{"stage": event["stage"],
        "degrees_of_freedom": event["noise_evidence"]["degrees_of_freedom"],
        "scales": event["noise_evidence"]["scales"],
        "sensitive_repeats": event["noise_evidence"]["sensitive_repeats"],
        "innovation_evaluations": event.get("innovation_evaluations", 0),
        "actual_scale_evaluations": event.get("evaluations", 0),
        "selected_action": choices.get(event["stage"])}
        for event in diagnostics.get("value_precision", {}).get("rounds", []) if event.get("noise_evidence")]
    boundary = diagnostics.get("boundary_noise", {})
    result["boundary_noise_posterior"] = {
        "models": [{"stage": model["stage"], "scope_start": model["scope_start"],
                    "scales": model["evidence"]["scales"], "decision": model["decision"]}
                   for model in boundary.get("models", [])],
        "last_posterior": boundary.get("posteriors", [None])[-1] if boundary.get("posteriors") else None,
        "last_recommendation": boundary.get("last_recommendation"),
        "work": {key: boundary.get(key, 0) for key in ("certification_calls", "certification_rows",
            "weight_updates", "component_evaluations", "seconds")}}
    return result


def report_support(brief, output):
    """A small result packet for a short report, using the existing summary."""
    selected = [case for case in brief["comparisons"] if case["group"] != "historical_retention"]
    regrets = {arm: mean(case["arms"][arm]["objective"]-.35 for case in selected)
               for arm in (BASELINE, METHOD, unknown.NATIVE)} if selected else {}
    groups = {name: {"cases": values["cases"],
        "paired_feasibility_first": values["paired_feasibility_first"],
        "feasible": {arm: result["feasible"] for arm,result in values["arms"].items()}}
        for name,values in brief["metrics"].items()}
    costs = {}
    for values in brief["metrics"].values():
        for arm, result in values["arms"].items():
            cost = costs.setdefault(arm, {"matched_cases": 0, "objective_research_compute": 0,
                "constraint_research_compute": 0, "optimizer_seconds": 0., "native_kg_seconds": 0.})
            cost["matched_cases"] += values["cases"]
            for target, source in (("objective_research_compute", "objective_compute"),
                    ("constraint_research_compute", "constraint_compute"),
                    ("optimizer_seconds", "optimizer_seconds"), ("native_kg_seconds", "native_kg_seconds")):
                cost[target] += result["work"][source]
    behaviors = brief["research_behavior"]
    swapped = [item["case_id"] for item in behaviors if any(
        install["stage"] >= event["stage"] and install["scope_start"] == event["scope_start"]
        for install in item["raw_gp_installations"] for event in item["installed_goal_reopens"])]
    budgeted = [item["case_id"] for item in behaviors if any(
        event["action"] == "begin_budgeted_goal_validation" for event in item["events"])]
    account_path = output / "prior_attempt_accounting.json"
    prior = study.read(account_path).get("completed_previous_physical_calls") if account_path.exists() else None
    confirmation_path = output / "fresh_confirmation/result_brief.json"
    fresh_calls = study.read(confirmation_path)["costs"]["physical_calls"] if confirmation_path.exists() else 0
    new_calls = brief["new_costs"]["physical_calls"] + fresh_calls
    summary = {"completed_cases": brief["completed_cases"], "expected_development_cases": 43,
        "grouped_outcomes": groups, "unknown_goal_cases": len(selected), "unknown_goal_regret": regrets,
        "program_to_raw_gp_cases": swapped, "budgeted_candidate_validation_cases": budgeted,
        "matched_controller_work_and_timing": costs,
        "physical_calls": {"previous_attempts": prior, "this_revision_development": brief["new_costs"]["physical_calls"],
            "this_revision_confirmation": fresh_calls, "this_revision_total": new_calls,
            "cumulative_actual": prior + new_calls if prior is not None else None},
        "source": str((output / "development_result_brief.json").resolve()),
        "interpretation": "Both-infeasible recommendations are separate from improvements. Program-to-GP "
            "counts require an installed-goal reopen followed by a supported GP installation on that scope. "
            "Budgeted-validation counts record entry into future validation, not successful adoption. "
            "Research operations, native KG and optimizer time measure distinct or overlapping work; "
            "historical timings are not simultaneous throughput comparisons."}
    if selected:
        paired = Counter()
        for group in groups.values():
            paired.update(group["paired_feasibility_first"][BASELINE])
        summary["short_report_draft"] = (
            f"本版已完成{brief['completed_cases']}/43例开发。相对0.53，可行性优先比较为"
            f"{paired['improved']}改善、{paired['same']}持平、{paired['worsened']}退化，"
            f"双方均不可行{paired['both_infeasible']}例。未知目标{len(selected)}例的平均regret为"
            f"{regrets[BASELINE]:.6f}→{regrets[METHOD]:.6f}，普通SC-KG为{regrets[unknown.NATIVE]:.6f}。"
            f"{len(swapped)}例完成程序到原GP的替换，{len(budgeted)}例预算受限候选进入后续验证；"
            "这两项分别表示部署变化和研究开始，不代表全部解释已通过验证。"
            f"本版新增{new_calls}次物理调用，既有{prior if prior is not None else '未在本目录登记'}次另列。"
            "逐组可行性、研究计算及优化器耗时见同一JSON；耗时来自分批本地运行。")
    study.write(output / "report_brief.json", summary)
    return summary


def summarize_development(spec, output):
    comparisons, metrics, actual = [], {}, []
    for group, settings in spec["groups"].items():
        group_comparisons, rows_by_arm = [], {BASELINE: [], METHOD: []}
        if group != "historical_retention":
            rows_by_arm[unknown.NATIVE] = []
        for case in settings["cases"]:
            path = output / "runs" / f"{case['case_id']}_{METHOD}.json"
            if not path.exists():
                continue
            rows = {BASELINE: study.read(case["baseline_record"]), METHOD: study.read(path)}
            if "native_record" in case:
                rows[unknown.NATIVE] = study.read(case["native_record"])
            group_comparisons.append({"case_id": case["case_id"], "group": group,
                "profile": case["profile"], "arms": {arm: previous.values(row) for arm,row in rows.items()}})
            for arm,row in rows.items():
                rows_by_arm[arm].append(row)
            actual.append(rows[METHOD])
        if group_comparisons:
            comparisons.extend(group_comparisons)
            metrics[group] = comparison_metrics(group_comparisons, rows_by_arm)
    brief = {"protocol": str((output / "protocol.json").resolve()), "completed_cases": len(actual),
        "expected_cases": 43, "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [behavior(row) for row in actual],
        "new_costs": {"initial_calls": 0, "evaluation_calls": 0,
            "physical_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual),
            **work(actual)}, "limitations": spec["limitations"]}
    study.write(output / "development_result_brief.json", brief)
    report_support(brief, output)
    print(json.dumps({"completed_cases": len(actual), "new_costs": brief["new_costs"],
        "paired": {group: value["paired_feasibility_first"] for group,value in metrics.items()}}), flush=True)


def execute_confirmation(spec, output):
    target = output / "fresh_confirmation"
    original = study.read(SOURCE / "fresh_confirmation/protocol.json")
    fresh = {key: deepcopy(original[key]) for key in ("sc_kg_project_root", "research_config",
        "total_calls_per_arm", "optimizer_seed_offset", "optimizer_settings", "policy", "revision",
        "evaluation", "change_after_new_calls", "task_formula", "task_information")}
    fresh.update(cases=spec["fresh_cases"], seeds=spec["fresh_seeds"],
        arms=spec["confirmation"]["arms"], frozen_baseline_source=spec["source_v053"],
        frozen_method_source=spec["method_source"],
        source_protocol=str((output / "protocol.json").resolve()),
        accounting=spec["confirmation"], method=spec["method"], limitations=spec["limitations"])
    path = target / "protocol.json"
    if path.exists() and study.read(path) != fresh:
        raise ValueError("the saved fresh goal-recompetition confirmation differs")
    if not path.exists():
        study.write(path, fresh)
    previous.METHOD = BASELINE
    baseline, _ = previous.load_method({**spec, "method_source": spec["source_v053"]},
        "unknown_recompetition_fresh_baseline")
    initial = study.Program, study.MetaPriorResearch, study.SCKGEnvironment, unknown.task.Environment
    previous.METHOD = METHOD
    current, evaluator = previous.load_method(spec, "unknown_recompetition_fresh_method")
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
        matched = {arm: [row for row in rows if row["arm"] == arm and row["case_id"] in keys]
                   for arm in fresh["arms"]}
        metrics[profile] = comparison_metrics(selected, matched)
        for arm in fresh["arms"]:
            metrics[profile]["arms"][arm]["regret"] = mean(case["arms"][arm]["regret"] for case in selected)
    for row in rows:
        study.write(target / "runs" / f"{row['case_id']}_{row['arm']}.json", row)
    initial_calls = sum(row["actual_new_initial_calls"] for row in research.values())
    evaluator_calls = sum(row["simulator_calls"] for row in evaluations.values())
    online_calls = sum(row["costs"]["optimizer_new_simulator_calls"] for row in rows)
    brief = {"protocol": str(path.resolve()), "completed_cases": 9, "optimizer_runs": len(rows),
        "comparisons": comparisons, "metrics": metrics,
        "research_behavior": [behavior(row) for row in rows if row["arm"] != unknown.NATIVE],
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
    parser.add_argument("--phase", choices=("plan", "smoke", "development", "confirmation", "summary", "report"), default="plan")
    parser.add_argument("--smoke-cases", nargs="+", help="Case ids for a new protocol's initial smoke subset")
    args = parser.parse_args()
    spec = prepare(args.output, args.smoke_cases)
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

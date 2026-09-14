#!/usr/bin/env python3
"""Identify objective interactions using paid two-coordinate experiments."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean

import run_sc_kg_goal_recompetition_study as prior


previous, study, unknown = prior.previous, prior.study, prior.unknown
BASELINE, METHOD = "hegel_v054", "hegel_joint_identification"
SOURCE = study.ROOT / "artifacts/sc_kg_goal_recompetition_20260911/revision4"
SEEDS = (2443012909, 2443112933, 2443212959)
SMOKE = ("goal_recompetition_3_center_interaction_change",
         "goal_completion_2_center_interaction_change",
         "unknown_confirmation_1_center_change")


def prepare(output):
    original = study.read(SOURCE / "protocol.json")
    confirmed = study.read(SOURCE / "fresh_confirmation/protocol.json")
    groups = deepcopy(original["groups"])
    groups["unknown_recompetition_confirmation"] = {
        "source_protocol": str((SOURCE / "fresh_confirmation/protocol.json").resolve()),
        "cases": deepcopy(confirmed["cases"])}
    for group, settings in groups.items():
        source = SOURCE / "fresh_confirmation" if group == "unknown_recompetition_confirmation" else SOURCE
        for case in settings["cases"]:
            case["split"] = "development"
            if group == "unknown_recompetition_confirmation":
                case["initial_research_origin"] = str((source / "research" / f"{case['research_id']}.json").resolve())
                case["evaluation_origin"] = str((source / "evaluation" / previous.evaluation_name(case)).resolve())
                case["native_record"] = str((source / "runs" / f"{case['case_id']}_{unknown.NATIVE}.json").resolve())
            case["baseline_record"] = str((source / "runs" / f"{case['case_id']}_hegel_goal_recompetition.json").resolve())
    fresh = []
    for index, (old_seed, seed) in enumerate(zip(confirmed["seeds"], SEEDS), 1):
        for old in confirmed["cases"]:
            if old["task"]["seed"] != old_seed:
                continue
            case = deepcopy(old)
            case["task"]["seed"] = seed
            case.update(case_id=f"joint_identification_{index}_{case['profile']}",
                research_id=f"joint_identification_{index}_{case['task']['problem']}",
                cohort="fresh9", split="confirmation")
            fresh.append(case)
    spec = {"experiment": "targeted_joint_coordinate_objective_identification",
        "groups": groups, "smoke_cases": list(SMOKE), "arms": [BASELINE, METHOD],
        "source_v054": str((SOURCE / "method_source").resolve()),
        "method_source": str((output / "method_source").resolve()),
        "fresh_seeds": list(SEEDS), "fresh_cases": fresh,
        "development": {"cases": 52, "new_optimizer_runs": 52, "initial_calls": 0,
            "online_calls": 1664, "evaluation_calls": 0, "physical_calls": 1664,
            "smoke_calls_included_once": 96},
        "confirmation": {"cases": 9, "arms": [unknown.NATIVE, BASELINE, METHOD],
            "new_optimizer_runs": 27, "initial_calls": 192, "online_calls": 864,
            "evaluation_calls": 1152, "physical_calls": 2208},
        "method": "Start from the completed frozen0.54 fifteen-module package. Identify unresolved "
            "objective interactions with targeted two-coordinate controls selected using paid evidence. "
            "Keep32 paid initialization plus32 online calls, the20000 objective-research operation "
            "limit and all existing optimizer, constraint-research and value-comparison budgets. "
            "Freeze the finalized implementation before the first new observation.",
        "selection": "All43 prior development and9 prior confirmation cases now form52 development "
            "and retention cases. The three named smoke cases cover failed interaction identification "
            "and existing gains to preserve. Reuse their completed runs once in full development. "
            "All39 unknown-goal cases retain native controls; all52 retain actual0.54 controls.",
        "accounting": "Development purchases only52 times32 online vectors:1664. Independent9-case "
            "confirmation uses six shared32-vector initializations,27 optimizers times32 online calls, "
            "and nine128-vector final evaluations:2208. Initialization research uses frozen0.54. "
            "Total if both phases complete:79 optimizer runs and3872 physical calls. Report controller "
            "operations, native KG and optimizer time separately; paid vectors produce both outputs.",
        "limitations": "The52 reused cases are development evidence. The new9 cases remain unrun "
            "unless development supports continuing, and use the unchanged frozen method and seeds. "
            "Native SC-KG comparisons condition on shared paid initialization. Both-infeasible results "
            "are separate from improvements. Historical timings are not simultaneous throughput tests."}
    equivalence = output / "source_equivalence.json"
    if equivalence.exists():
        reused = study.read(equivalence)["reused_records"]
        new_runs = 52-len(reused)
        spec["development"].update(new_optimizer_runs=new_runs, online_calls=32*new_runs,
            physical_calls=32*new_runs, reused_trajectory_cases=len(reused),
            smoke_calls_included_once=32*sum(case not in reused for case in SMOKE),
            source_equivalence=str(equivalence.resolve()))
        spec["accounting"] = (f"Development references{len(reused)} unchanged prior trajectories and "
            f"executes{new_runs} affected cases at32 online calls each:{32*new_runs} physical calls. "
            "The referenced runs and timings retain their actual source. Confirmation still executes "
            "all27 new optimizer runs with192 initial,864 online and1152 evaluator calls:2208. "
            "Previously executed attempts remain separate in prior_attempt_accounting.json.")
    path = output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved joint-identification protocol differs")
    if not path.exists():
        study.write(path, spec)
    return spec


def configure_arms():
    # Reuse the preceding study's result accounting with this study's fixed labels.
    prior.BASELINE, prior.METHOD = BASELINE, METHOD
    previous.METHOD = METHOD


def execute_development(spec, output, smoke=False):
    configure_arms()
    equivalence = output / "source_equivalence.json"
    reused = study.read(equivalence)["reused_records"] if equivalence.exists() else {}
    for group, settings in spec["groups"].items():
        cases = [case for case in settings["cases"] if case["case_id"] not in reused
                 and (not smoke or case["case_id"] in spec["smoke_cases"])]
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
    summary = {"development_cases": brief["completed_cases"], "expected_development_cases": 52,
        "development_vs_v054": dict(paired),
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
    if fresh:
        summary["confirmation_metrics"] = fresh["metrics"]
        summary["confirmation_costs"] = fresh["costs"]
        summary["confirmation_source"] = str(fresh_path.resolve())
    study.write(output / "report_brief.json", summary)


def summarize_development(spec, output):
    configure_arms()
    comparisons, metrics, actual, all_rows = [], {}, [], []
    equivalence = output / "source_equivalence.json"
    reused = study.read(equivalence)["reused_records"] if equivalence.exists() else {}
    for group, settings in spec["groups"].items():
        selected, matched = [], {BASELINE: [], METHOD: []}
        if group != "historical_retention":
            matched[unknown.NATIVE] = []
        for case in settings["cases"]:
            path = output / "runs" / f"{case['case_id']}_{METHOD}.json"
            executed = path.exists()
            if not executed and case["case_id"] in reused:
                path = Path(reused[case["case_id"]])
            if not path.exists():
                continue
            rows = {BASELINE: study.read(case["baseline_record"]), METHOD: study.read(path)}
            if "native_record" in case:
                rows[unknown.NATIVE] = study.read(case["native_record"])
            selected.append({"case_id": case["case_id"], "group": group, "profile": case["profile"],
                "method_execution": "current_revision" if executed else "source_equivalent_prior_trajectory",
                "method_record": str(path.resolve()),
                "arms": {arm: previous.values(row) for arm, row in rows.items()}})
            for arm, row in rows.items():
                matched[arm].append(row)
            all_rows.append(rows[METHOD])
            if executed:
                actual.append(rows[METHOD])
        if selected:
            comparisons.extend(selected)
            metrics[group] = prior.comparison_metrics(selected, matched)
    brief = {"protocol": str((output / "protocol.json").resolve()), "completed_cases": len(all_rows),
        "expected_cases": 52, "optimizer_runs": len(actual), "comparisons": comparisons, "metrics": metrics,
        "reused_trajectory_cases": len(all_rows)-len(actual),
        "research_behavior": [prior.behavior(row) for row in all_rows],
        "new_costs": {"initial_calls": 0, "evaluation_calls": 0,
            "physical_calls": sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual),
            **prior.work(actual)}, "limitations": spec["limitations"]}
    study.write(output / "development_result_brief.json", brief)
    report_support(brief, output)
    print(json.dumps({"completed_cases": len(all_rows), "optimizer_runs": len(actual),
        "reused_trajectory_cases": len(all_rows)-len(actual), "new_costs": brief["new_costs"],
        "paired": {group: value["paired_feasibility_first"] for group, value in metrics.items()}}), flush=True)


def execute_confirmation(spec, output):
    configure_arms()
    target = output / "fresh_confirmation"
    original = study.read(SOURCE / "fresh_confirmation/protocol.json")
    fresh = {key: deepcopy(original[key]) for key in ("sc_kg_project_root", "research_config",
        "total_calls_per_arm", "optimizer_seed_offset", "optimizer_settings", "policy", "revision",
        "evaluation", "change_after_new_calls", "task_formula", "task_information")}
    fresh.update(cases=spec["fresh_cases"], seeds=spec["fresh_seeds"], arms=spec["confirmation"]["arms"],
        frozen_baseline_source=spec["source_v054"], frozen_method_source=spec["method_source"],
        source_protocol=str((output / "protocol.json").resolve()), accounting=spec["confirmation"],
        method=spec["method"], limitations=spec["limitations"])
    path = target / "protocol.json"
    if path.exists() and study.read(path) != fresh:
        raise ValueError("the saved fresh joint-identification confirmation differs")
    if not path.exists():
        study.write(path, fresh)
    previous.METHOD = BASELINE
    baseline, _ = previous.load_method({**spec, "method_source": spec["source_v054"]}, "unknown_joint_fresh_baseline")
    initial = study.Program, study.MetaPriorResearch, study.SCKGEnvironment, unknown.task.Environment
    previous.METHOD = METHOD
    current, evaluator = previous.load_method(spec, "unknown_joint_fresh_method")
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

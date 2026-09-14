#!/usr/bin/env python3
"""Compare terminal new-control exploration and repeated observations after each outcome."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import run_sc_kg_joint_evidence_study as joint

goal, scope = joint.goal, joint.scope

study, accounting, repair = goal.study, goal.accounting, goal.repair
SEEDS = (1043010803, 1043110817, 1043210831)
ARMS = ("frozen_v039_joint_evidence", "terminal_value")
REPLAY_CASES = ("objective_only_change_1_noise_0.03", "objective_only_change_3_noise_0.03",
    "research_scope_3_stable_rzdt2_noise_0.00", "active_confirmation_2_mechanism_change_noise_0.03",
    "objective_only_change_2_noise_0.03", "independent_goal_2_objective_only_change_noise_0.03",
    "objective_exploration_1_mechanism_change_noise_0.03",
    "independent_goal_1_outlier_then_change_noise_0.03")


def protocol(source, output, method=None):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="terminal_exploration_and_remeasurement_value_competition",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v039").resolve()),
        historical_baseline_arm="joint_evidence",
        historical_baseline_origin="historical_v039_joint_evidence")
    spec["baseline_method"] = spec.pop("new_method")
    for name in (*repair.BASELINE_MODULES, "objective_research.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError(f"the actual pre-edit0.39 baseline is required: {name}")
    template_seed = next(case["task"]["seed"] for case in spec["cases"] if case["split"] == "confirmation")
    templates = [deepcopy(case) for case in spec["cases"]
                 if case["task"]["seed"] == template_seed and case["split"] in ("confirmation", "stress")]
    if len(templates) != 9:
        raise ValueError("the source study must contain six ordinary and three stress profiles per new seed")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    for index, seed in enumerate(SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case["case_id"] = f"terminal_value_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"terminal_value_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    spec["new_method"] = method if method is not None else study.read(output / "method.json")
    spec["baseline_scope"] = (
        "Reuse all220 actual completed0.39 optimizer runs, with every failure and regression. "
        "Fresh baselines load all seven saved0.39 runtime modules, including shared goal/constraint "
        "evidence, independently preserved objective heads and scope/HVD0, and the original "
        "terminal confirmation queue policy.")
    spec["scope"] = (
        "All previous220 cases are development. Three unused seeds each provide six ordinary "
        "confirmation cases (two noise levels, true change, and two stable mechanisms) plus "
        "three separate noisy stress cases: one constraint outlier, outlier followed by true "
        "change, and objective-only change with unchanged constraint. Preserve all outcomes.")
    spec["research_initialization"] = (
        "Use the saved0.39 claim compiler for fresh initial research; retain the unchanged "
        "public completion rule to32 actual full vectors. Both arms share the same initial "
        "measurements and executable prior; same-task stress profiles reuse that initialization.")
    no_head = [case["case_id"] for case in spec["cases"] if case["split"] == "development"
        and not study.read(source / "runs" / f"{case['case_id']}_joint_evidence.json").get("objective_scope_revisions")]
    control = "research_scope_3_stable_rzdt2_noise_0.00"
    spec["unchanged_no_head_reuse"] = [key for key in no_head if key != control]
    spec["actual_unchanged_control"] = control
    spec["expected_new_optimizer_runs"] = len(spec["cases"]) * 2 - 220 - len(spec["unchanged_no_head_reuse"])
    spec["reuse_accounting"] = (
        "Reuse220 actual baseline runs, all existing initial measurements, and independent "
        "evaluation measurements. Reuse162 new-arm runs whose unentered head-only branch cannot change behavior; actually run "
        "57 headed development arms, one no-head unchanged control, and54 fresh arms: "
        "112 actual new optimizer runs or3584 online calls. Every arm remains32 initial plus32 "
        "online calls. Report objective and constraint search together and separately, all GP "
        "forecast counts, coverage/confirmation/validation actions, and paid rejected evidence.")
    previous = output / "attempt_1_singleton_action_pool"
    if previous.exists():
        inventory = study.read(previous / "singleton_entry_inventory.json")
        spec["previous_attempt_path"] = str(previous.resolve())
        spec["unchanged_attempt_reuse"] = inventory["unchanged_case_ids"]
        spec["affected_development_cases"] = [item["case_id"] for item in inventory["affected"]]
        spec["expected_new_optimizer_runs"] = inventory["affected_cases"] + 54
        spec["expected_cumulative_optimizer_runs"] = 58 + spec["expected_new_optimizer_runs"]
        spec["reuse_accounting"] = (
            "The completed first implementation retains220 development pairs and1856 actual online calls. "
            "Only42 actual single-repeat-entry new arms are rerun;178 unaffected development arms "
            "and220 original0.39 baselines are reused. Run both arms on27 still-unused fresh cases. "
            "This final implementation adds96 optimizer runs/3072 online calls; cumulative across "
            "both implementations154 actual runs/4928 online calls. Initial/evaluation acquisitions "
            "for the new seeds are counted once separately. The unchanged control was bought only "
            "in the first implementation and is now reused.")
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for key in spec["unchanged_no_head_reuse"]:
        target = output / "runs" / f"{key}_{ARMS[1]}.json"
        if target.exists():
            continue
        original = source / "runs" / f"{key}_joint_evidence.json"
        row = study.read(original)
        row.update(arm=ARMS[1], split="development",
            run_origin=spec["historical_baseline_origin"], original_result=str(original.resolve()),
            reuse_reason="No objective head installed; terminal_offers entry never reached and original selection unchanged.")
        study.write(target, row)


def prepare_previous_attempt_reuse(spec, output):
    if "previous_attempt_path" not in spec:
        return
    source = Path(spec["previous_attempt_path"])
    for key in spec["unchanged_attempt_reuse"]:
        target = output / "runs" / f"{key}_{ARMS[1]}.json"
        if target.exists():
            continue
        original = source / "runs" / f"{key}_{ARMS[1]}.json"
        row = study.read(original)
        row.update(run_origin=spec["historical_baseline_origin"], original_result=str(original.resolve()),
            reuse_reason="No actual single-repeat terminal pool; the changed fallback branch is never entered.")
        study.write(target, row)


def arm_modules(spec):
    package_name = "hegel_machine.cognition._terminal_value_baseline_v039"
    package = types.ModuleType(package_name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[package_name] = package
    baseline = {name: importlib.import_module(f"{package_name}.{name}") for name in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    study.MetaPriorResearch = importlib.import_module(f"{package_name}.meta_prior_research").MetaPriorResearch
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
                      baseline["joint_model_revision"].JointModelRevision),
            ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}



def complete_cost_brief(brief):
    costs = brief["actual_new_costs"]
    initial = costs["actual_new_research_calls"] + costs["actual_supplemental_initial_calls"]
    brief["actual_new_initial_calls"] = initial
    brief["actual_new_physical_calls"] = costs["actual_new_optimizer_calls"] + initial + costs["independent_evaluation_calls"]
    previous = brief.get("previous_completed_attempt", {}).get("actual_new_costs", {})
    brief["actual_cumulative_physical_calls"] = brief["actual_new_physical_calls"] + sum(previous.get(key, 0) for key in
        ("actual_new_optimizer_calls", "actual_new_research_calls", "actual_supplemental_initial_calls", "independent_evaluation_calls"))
    brief["fresh_matched_search"] = [sum(row["search"][index] for row in brief["cases"] if row["split"] != "development") for index in (0, 1)]
    brief["prediction_loss_counts"] = {key: {
        "improved": sum(row[key][1] < row[key][0] - 1e-9 for row in brief["cases"]),
        "unchanged": sum(abs(row[key][1] - row[key][0]) <= 1e-9 for row in brief["cases"]),
        "worse": sum(row[key][1] > row[key][0] + 1e-9 for row in brief["cases"])}
        for key in ("objective_gp_true_mse", "constraint_gp_true_mse")}
    previous_kg = brief.get("previous_completed_attempt", {}).get("native_kg", {})
    brief["actual_cumulative_native_kg"] = {key: brief["native_kg"][key] + previous_kg.get(key, 0)
        for key in ("native_terminal_kg_scored_actions", "native_terminal_kg_seconds",
            "native_all_kg_scored_actions", "native_all_kg_seconds")}
    return brief


def enrich_terminal_summary(spec, output, phase):
    joint.enrich_joint_evidence_summary(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    compact = study.read(output / f"{stem}compact_summary.json")
    compact.pop("previous_completed_attempt_costs", None)
    for key in list(compact):
        if key.startswith("actual_cumulative_"):
            del compact[key]
    compact["current_study_accounting"] = (
        "This study compares the final0.39 method with terminal value competition. Prior "
        "versions and their historical development costs are not new0.40 acquisitions. "
        "Every actual new optimizer run remains32 online observations; reused baselines, "
        "initial measurements, and evaluation vectors are not rebought.")
    summary = study.read(output / f"{stem}summary.json")
    fields = ("terminal_rounds", "terminal_candidates", "terminal_selected_offers",
        "terminal_new_actions", "terminal_repeat_actions", "terminal_outside_offer_actions",
        "terminal_native_pool_rounds", "terminal_native_pool_candidates")
    def terminal_metrics(rows):
        recorded = [row for row in rows if "exact_kg_stages" in row["costs"]]
        terminal = []
        for row in recorded:
            # SC logs stage before the draw; actual terminal choices log stage after it.
            chosen = {item["stage"] - 1 for item in row["costs"].get(
                "objective_diagnostics", {}).get("terminal_choices", [])}
            terminal.extend(item for item in row["costs"]["exact_kg_stages"] if item["stage"] in chosen)
        all_stages = [item for row in recorded for item in row["costs"]["exact_kg_stages"]]
        return {**{key: sum(row["costs"].get("objective_diagnostics", {}).get(key, 0)
                for row in rows) for key in fields},
            "runs_with_native_kg_costs": len(recorded), "runs_without_native_kg_costs": len(rows) - len(recorded),
            "native_terminal_kg_scored_actions": sum(item.get("exact_kg_active_action_count") or 0 for item in terminal),
            "native_terminal_kg_seconds": sum(item.get("t_kg_compute") or 0 for item in terminal),
            "native_all_kg_scored_actions": sum(item.get("exact_kg_active_action_count") or 0 for item in all_stages),
            "native_all_kg_seconds": sum(item.get("t_kg_compute") or 0 for item in all_stages),
            "recorded_mc_sample_settings": sorted({item["exact_kg_mc_samples"] for item in all_stages
                if item.get("exact_kg_mc_samples") is not None})}
    for group in compact["groups"]:
        for arm in ARMS:
            rows = [row for row in summary["results"] if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], arm)]
            group[arm].update(terminal_metrics(rows))
    fresh = [row for row in summary["results"] if row.get("run_origin") != spec["historical_baseline_origin"]]
    compact["actual_new_terminal_and_kg_costs"] = terminal_metrics(fresh)
    compact["native_kg_comparable_fresh_runs"] = {arm: terminal_metrics([row for row in fresh if row["arm"] == arm
        and row["split"] != "development"]) for arm in ARMS}
    compact["notes"].append("Original0.39 runs did not export native KG scoring counts/times; these are unavailable, not zero. New27 both arms use the same native counters. Active-action counts and actual recorded MC settings are reported separately; no inferred fantasy workload is claimed. Native subcomponent times recorded as zero reflect upstream unavailable timing instrumentation.")
    paired = []
    by_key = {(row["case_id"], row["arm"]): row for row in summary["results"]}
    for case in spec["cases"]:
        if (case["case_id"], ARMS[1]) not in by_key:
            continue
        old, new = [by_key[(case["case_id"], arm)] for arm in ARMS]
        paired.append({"case_id": case["case_id"], "split": case["split"], "profile": case["profile"],
            "noise": case["task"]["noise"], "objective": [row["final"]["true_objective"] for row in (old, new)],
            "feasible": [row["final"]["true_feasible"] for row in (old, new)],
            "objective_gp_true_mse": [row["generalization"].get("objective_gp_true_objective_means_mse") for row in (old, new)],
            "constraint_gp_true_mse": [row["generalization"]["active_gp_true_constraint_means_mse"] for row in (old, new)],
            "search": [row.get("objective_research", {}).get("compute_cost", 0)
                + row["costs"]["recovery_compute"].get("compute_cost", 0) for row in (old, new)],
            "terminal": terminal_metrics([new]), "total_calls": new["costs"]["total_simulator_calls"]})
    counts = lambda rows: {"cases": len(rows),
        "improved": sum(row["objective"][1] < row["objective"][0] - 1e-9 for row in rows),
        "unchanged": sum(abs(row["objective"][1] - row["objective"][0]) <= 1e-9 for row in rows),
        "worse": sum(row["objective"][1] > row["objective"][0] + 1e-9 for row in rows),
        "feasible": [sum(row["feasible"][i] for row in rows) for i in (0, 1)]}
    brief = {"overall": counts(paired), "development": counts([row for row in paired if row["split"] == "development"]),
        "fresh": counts([row for row in paired if row["split"] != "development"]), "cases": paired,
        "actual_new_costs": compact["costs"], "matched_search": compact["matched_total_search_compute"],
        "actual_new_search": compact["actual_new_combined_search_compute"],
        "native_kg": compact["actual_new_terminal_and_kg_costs"],
        "fresh_native_kg": compact["native_kg_comparable_fresh_runs"]}
    control_id = spec["actual_unchanged_control"]
    if (control_id, ARMS[1]) in by_key:
        old, new = [by_key[(control_id, arm)] for arm in ARMS]
        brief["unchanged_control_same_behavior"] = all(old[key] == new[key] for key in
            ("simulations", "actions", "frozen_evaluation_predictions", "objective_scope_revisions", "scope_revisions"))
    if "previous_attempt_path" in spec:
        previous = study.read(Path(spec["previous_attempt_path"]) / "development_result_brief.json")
        compact["previous_completed_attempt_costs"] = previous["actual_new_costs"]
        compact["previous_completed_attempt_search_compute"] = previous["actual_new_search"]
        compact["previous_completed_attempt_native_kg"] = previous["native_kg"]
        compact["actual_cumulative_online_calls"] = (compact["costs"]["actual_new_optimizer_calls"]
            + previous["actual_new_costs"]["actual_new_optimizer_calls"])
        compact["actual_cumulative_search_compute"] = (compact["actual_new_combined_search_compute"]
            + previous["actual_new_search"])
        brief["previous_completed_attempt"] = {key: previous[key] for key in
            ("overall", "matched_search", "actual_new_costs", "actual_new_search", "native_kg")}
        brief["actual_cumulative_online_calls"] = compact["actual_cumulative_online_calls"]
        brief["actual_cumulative_search_compute"] = compact["actual_cumulative_search_compute"]
    study.write(output / f"{stem}result_brief.json", complete_cost_brief(brief))
    study.write(output / f"{stem}compact_summary.json", compact)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    parser.add_argument("--case", action="append", help="Select cases from the frozen roster.")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = repair.ARMS = goal.ARMS = scope.ARMS = joint.ARMS = ARMS
    study.controlled_problem = goal.controlled_problem
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen terminal-value roster or method differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 220,
            "confirmation_cases": 18, "stress_cases": 9, "new_optimizer_runs": spec["expected_new_optimizer_runs"]}))
        return
    cases = spec["cases"]
    if args.phase == "replay" and not args.case:
        cases = [case for case in cases if case["case_id"] in REPLAY_CASES]
    elif args.phase in ("development", "confirmation", "stress"):
        cases = [case for case in cases if case["split"] == args.phase]
    if args.case:
        cases = [case for case in cases if case["case_id"] in args.case]
        if not cases or set(args.case) - {case["case_id"] for case in cases}:
            raise ValueError("case selection must name frozen cases in this phase")
    accounting.prepare_reuse(spec, args.output)
    prepare_unaffected_reuse(spec, args.output)
    prepare_previous_attempt_reuse(spec, args.output)
    modules = arm_modules(spec)
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results,
                                 environment_factory=goal.evaluation_environment)
    accounting.summarize(spec, args.output, cases, research, results, evaluations, args.phase)
    repair.enrich_repair_summary(spec, args.output, args.phase)
    goal.enrich_objective_summary(spec, args.output, args.phase)
    enrich_terminal_summary(spec, args.output, args.phase)


if __name__ == "__main__":
    main()

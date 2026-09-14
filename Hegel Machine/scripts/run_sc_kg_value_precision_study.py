#!/usr/bin/env python3
"""Refine disputed terminal acquisition estimates using independent simulation randomness."""

import argparse
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import types

import run_sc_kg_terminal_value_study as previous

goal, scope, joint = previous.goal, previous.scope, previous.joint
study, accounting, repair = previous.study, previous.accounting, previous.repair
ARMS = ("frozen_v040_terminal_value", "value_precision")
SEEDS = (1143010907, 1143110921, 1143210939)
CONTROL_CASES = ("forward_1_mechanism_change_noise_0.00", "research_scope_3_stable_rzdt2_noise_0.00")
REPLAY_CASES = (CONTROL_CASES[0], "research_scope_3_objective_only_change_noise_0.03",
    "objective_only_change_1_noise_0.03", "objective_only_change_2_noise_0.03",
    "objective_only_change_3_noise_0.03", "active_confirmation_2_mechanism_change_noise_0.03",
    "terminal_value_1_objective_only_change_noise_0.03", CONTROL_CASES[1])


def protocol(source, output, method=None):
    spec = deepcopy(study.read(source / "protocol.json"))
    for key in ("previous_attempt_path", "unchanged_attempt_reuse", "unchanged_no_head_reuse",
            "expected_cumulative_optimizer_runs", "actual_unchanged_control"):
        spec.pop(key, None)
    spec.update(experiment="terminal_acquisition_value_precision", development_source=str(source.resolve()),
        arms=list(ARMS), historical_baseline_arm="terminal_value",
        historical_baseline_origin="historical_v040_terminal_value",
        frozen_baseline_source=str((output / "baseline_source_v040").resolve()))
    spec["baseline_method"] = spec.pop("new_method")
    for name in (*repair.BASELINE_MODULES, "objective_research.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError(f"pre-edit0.40 baseline missing: {name}")
    template_seed = next(case["task"]["seed"] for case in spec["cases"] if case["split"] == "confirmation")
    templates = [deepcopy(case) for case in spec["cases"]
        if case["task"]["seed"] == template_seed and case["split"] in ("confirmation", "stress")]
    if len(templates) != 9:
        raise ValueError("six ordinary and three stress templates per seed are required")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    old_cases = list(spec["cases"])
    for index, seed in enumerate(SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case["case_id"] = f"value_precision_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"value_precision_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    inventory = study.read(output / "entry_inventory.json")
    affected = {entry["case_id"] for entry in inventory["entries"]}
    spec["affected_development_cases"] = sorted(affected)
    spec["actual_control_cases"] = [CONTROL_CASES[1]]
    spec["actual_unchanged_control"] = CONTROL_CASES[1]
    spec["previous_precision_attempt_path"] = str((output / "attempt_1_precision_only").resolve())
    spec["unchanged_entry_reuse"] = [case["case_id"] for case in old_cases if case["case_id"] not in affected]
    spec["expected_new_optimizer_runs"] = len(affected) + len(SEEDS) * 18
    spec["expected_cumulative_optimizer_runs"] = 8 + spec["expected_new_optimizer_runs"]
    spec["new_method"] = method if method is not None else study.read(output / "method.json")
    spec["baseline_scope"] = (
        "The comparator is the final0.40 terminal action competition with native MC1, preserved "
        "SC actions beside a singleton measured offer, independent objective heads, and shared "
        "constraint/goal future evidence. Reuse all247 actual original runs as development comparators.")
    spec["scope"] = (
        "All247 old cases are development. Three prespecified unused seeds each provide six "
        "ordinary cases (two noise levels and true change/two stable mechanisms) plus noisy "
        "single-outlier, outlier-followed-by-change, and objective-only-change stress. "
        "All outcomes, regressions, and extra simulation compute are retained.")
    spec["research_initialization"] = (
        "Fresh initial research uses the saved0.40 compiler and the unchanged public completion "
        "rule to32 actual full vectors, shared across arms and same-task profiles. Initial and "
        "independent evaluation vectors are acquired once and separately charged.")
    spec["reuse_accounting"] = (
        "The first eight-case precision-only attempt is retained separately:256 online calls "
        "and760 actual additional candidate-draw evaluations. The second implementation changes "
        "terminal nomination as well as numerical value precision, so rerun all64 actually headed "
        "development trajectories and reuse183 no-head trajectories. Reuse the first attempt's "
        "actually measured no-head control once; do not charge it again. Add54 fresh arms. "
        "Final phase118 actual runs/3776 online calls; cumulative126 runs/4032 online calls. "
        "New initial and independent evaluation vectors are charged separately. Additional "
        "acquisition simulations buy no physical observations and remain separately measured.")
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for key in spec["unchanged_entry_reuse"]:
        target = output / "runs" / f"{key}_{ARMS[1]}.json"
        if target.exists():
            continue
        original = source / "runs" / f"{key}_terminal_value.json"
        if key == CONTROL_CASES[1]:
            original = Path(spec["previous_precision_attempt_path"]) / "runs" / f"{key}_{ARMS[1]}.json"
        row = study.read(original)
        row.update(arm=ARMS[1], split="development", run_origin=spec["historical_baseline_origin"],
            original_result=str(original.resolve()),
            reuse_reason="No objective head was installed, so neither terminal nomination nor value precision can enter; original decisions and RNG are unchanged.")
        study.write(target, row)


def arm_modules(spec):
    package_name = "hegel_machine.cognition._value_precision_baseline_v040"
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


def enrich_precision_summary(spec, output, phase):
    previous.enrich_terminal_summary(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    compact = study.read(output / f"{stem}compact_summary.json")
    compact["current_study_accounting"] = (
        "This study compares final0.40 with independently sampled terminal value refinement. "
        "Past0.40 development costs are historical; only current actual acquisitions and "
        "extra numerical integration are charged here. Native MC1 scoring and additional "
        "value refinement are separate measured costs.")
    brief = study.read(output / f"{stem}result_brief.json")
    summary = study.read(output / f"{stem}summary.json")
    fields = ("evaluations", "tie_forecast_rows", "score_one_calls", "terminal_value_calls", "terminal_pool_rows", "seconds")
    def metrics(rows):
        diagnostics = [row["costs"].get("objective_diagnostics", {}).get("value_precision", {}) for row in rows]
        rounds = [event for diag in diagnostics for event in diag.get("rounds", [])]
        comparable = [event for event in rounds if "candidate_points" in event]
        reason = "research_promising_constraint_certificate"
        nominations = [offer for row in rows for event in row["costs"].get("objective_diagnostics", {}).get("terminal_candidate_sets", [])
            for offer in event["offers"] if offer.get("reason") == reason]
        requests = [request for row in rows for request in row["costs"].get("objective_diagnostics", {}).get("requests", [])
            if request.get("reason") == reason]
        return {**{key: sum(diag.get(key, 0) for diag in diagnostics) for key in fields},
            "rounds": len(rounds),
            "numerically_flat_stops": sum(event["stop_reason"] == "numerically_flat_paired_values" for event in rounds),
            "constraint_certificate_nominations": len(nominations),
            "constraint_certificate_paid_repeats": len(requests),
            "final_recommendation_equals_certificate_repeat": sum(any(request.get("reason") == reason
                and request["point"] == row["final"]["x_recommended"] for request in
                row["costs"].get("objective_diagnostics", {}).get("requests", [])) for row in rows),
            "original_mc1_seconds_on_refined_rounds": sum(event.get("original_mc1_seconds", 0) for event in rounds),
            "empirically_separated_rounds": sum(event["status"] == "empirically_separated" for event in rounds),
            "unresolved_rounds": sum(event["status"] == "unresolved" for event in rounds),
            "initial_action_comparison_available": len(comparable),
            "changed_initial_action_rounds": sum(event["selected_point"] != event["candidate_points"][max(range(len(event["initial_clipped"])), key=event["initial_clipped"].__getitem__)] for event in comparable),
            "maximum_evaluations_per_run": max((diag.get("evaluations", 0) for diag in diagnostics), default=0)}
    for group in compact["groups"]:
        for arm in ARMS:
            rows = [row for row in summary["results"] if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], arm)]
            group[arm]["value_precision"] = metrics(rows)
    actual = [row for row in summary["results"] if row.get("run_origin") != spec["historical_baseline_origin"]]
    compact["actual_new_value_precision"] = metrics(actual)
    compact["fresh_value_precision"] = {arm: metrics([row for row in actual if row["arm"] == arm and row["split"] != "development"]) for arm in ARMS}
    compact["value_precision_accounting"] = (
        "Native t_kg_compute is the complete acquisition scoring time and includes precision "
        "refinement on entered rounds. Additional seconds are a measured subset and must not "
        "be added again. Native MC1 active-action counts and actual additional candidate-draw "
        "evaluations are separate counters. Policy indicators0/1 only pass a selected action "
        "through the original argmax and are not KG values, improvements, or zero-score rates. "
        "The initial MC1 raw values and additional action mean_raw remain numerical value records.")
    brief["actual_new_value_precision"] = compact["actual_new_value_precision"]
    brief["fresh_value_precision"] = compact["fresh_value_precision"]
    brief["complete_kg_scoring_includes_precision_seconds"] = True
    by_case = {row["case_id"]: row for row in summary["results"] if row["arm"] == ARMS[1]}
    for case in brief["cases"]:
        case["value_precision"] = metrics([by_case[case["case_id"]]])
    by_key = {}
    for key in CONTROL_CASES:
        for arm in ARMS:
            path = output / "runs" / f"{key}_{arm}.json"
            if path.exists():
                by_key[key, arm] = study.read(path)
    brief["actual_unchanged_controls"] = {}
    brief["earlier_headed_control_behavior"] = {}
    for key in CONTROL_CASES:
        if all((key, arm) in by_key for arm in ARMS):
            old, new = [by_key[key, arm] for arm in ARMS]
            control_group = "actual_unchanged_controls" if key == CONTROL_CASES[1] else "earlier_headed_control_behavior"
            brief[control_group][key] = all(old[field] == new[field] for field in
                ("simulations", "actions", "frozen_evaluation_predictions", "objective_scope_revisions", "scope_revisions"))
    if "previous_precision_attempt_path" in spec:
        first = study.read(Path(spec["previous_precision_attempt_path"]) / "replay_result_brief.json")
        brief["previous_completed_attempt"] = {key: first[key] for key in
            ("overall", "matched_search", "actual_new_costs", "actual_new_search", "native_kg", "actual_new_value_precision")}
        brief["actual_cumulative_online_calls"] = first["actual_new_costs"]["actual_new_optimizer_calls"] + brief["actual_new_costs"]["actual_new_optimizer_calls"]
        brief["actual_cumulative_search_compute"] = first["actual_new_search"] + brief["actual_new_search"]
        brief["actual_cumulative_value_precision"] = {key: (max(first["actual_new_value_precision"].get(key, 0), value)
            if key == "maximum_evaluations_per_run" else first["actual_new_value_precision"].get(key, 0) + value)
            for key, value in brief["actual_new_value_precision"].items()}
        compact["previous_completed_attempt_costs"] = first["actual_new_costs"]
        compact["previous_completed_attempt_value_precision"] = first["actual_new_value_precision"]
        compact["actual_cumulative_online_calls"] = brief["actual_cumulative_online_calls"]
        compact["actual_cumulative_search_compute"] = brief["actual_cumulative_search_compute"]
        compact["actual_cumulative_value_precision"] = brief["actual_cumulative_value_precision"]
    study.write(output / f"{stem}compact_summary.json", compact)
    study.write(output / f"{stem}result_brief.json", previous.complete_cost_brief(brief))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = repair.ARMS = goal.ARMS = scope.ARMS = joint.ARMS = previous.ARMS = ARMS
    study.controlled_problem = goal.controlled_problem
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the saved value precision roster or method differs")
    study.write(path, spec)
    if args.phase == "plan":
        print({"cases": len(spec["cases"]), "new_optimizer_runs": spec["expected_new_optimizer_runs"]})
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
    modules = arm_modules(spec)
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results, environment_factory=goal.evaluation_environment)
    accounting.summarize(spec, args.output, cases, research, results, evaluations, args.phase)
    repair.enrich_repair_summary(spec, args.output, args.phase)
    goal.enrich_objective_summary(spec, args.output, args.phase)
    enrich_precision_summary(spec, args.output, args.phase)


if __name__ == "__main__":
    main()

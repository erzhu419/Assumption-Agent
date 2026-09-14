#!/usr/bin/env python3
"""Revise an ongoing research scope and preserve goal-search compute for later evidence."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import run_sc_kg_objective_exploration_study as goal

study, accounting, repair = goal.study, goal.accounting, goal.repair
SEEDS = (1843010503, 1843110517, 1843210531)
ARMS = ("frozen_v037_objective_exploration", "research_scope")
REPLAY_CASES = ("new_confirmation_2_mechanism_change_noise_0.03",
                "objective_only_change_1_noise_0.03", "objective_only_change_2_noise_0.03",
                "objective_only_change_3_noise_0.03",
                "objective_exploration_1_mechanism_change_noise_0.03",
                "single_outlier_1_state_policy_noise_0.03")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="ongoing_research_scope_revision_and_goal_search_pacing",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v037").resolve()),
        historical_baseline_arm="objective_exploration",
        historical_baseline_origin="historical_v037_objective_exploration")
    spec["baseline_method"] = spec.pop("new_method")
    for name in (*repair.BASELINE_MODULES, "objective_research.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError(f"the actual pre-edit0.37 baseline is required: {name}")
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
            case["case_id"] = f"research_scope_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"research_scope_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    spec["new_method"] = study.read(output / "method.json")
    spec["baseline_scope"] = (
        "Reuse all139 actual completed0.37 optimizer runs, with every failure and regression. "
        "Fresh baselines load all seven saved0.37 runtime modules, including the original "
        "independent objective search and noisy terminal recommendation policy.")
    spec["scope"] = (
        "All previous139 cases are development. Three unused seeds each provide six ordinary "
        "confirmation cases (two noise levels, true change, and two stable mechanisms) plus "
        "three separate noisy stress cases: one constraint outlier, outlier followed by true "
        "change, and objective-only change with unchanged constraint. Preserve all outcomes.")
    spec["research_initialization"] = (
        "Use the saved0.37 claim compiler for fresh initial research; retain the unchanged "
        "public completion rule to32 actual full vectors. Both arms share the same initial "
        "measurements and executable prior; same-task stress profiles reuse that initialization.")
    spec["reuse_accounting"] = (
        "Reuse139 actual baseline runs, all existing initial measurements, and independent "
        "evaluation measurements. Run139 new development arms and both arms on27 fresh cases: "
        "193 actual new optimizer runs or6176 online calls. Every arm remains32 initial plus32 "
        "online calls. Report objective and constraint search together and separately, all GP "
        "forecast counts, coverage/confirmation/validation actions, and paid rejected evidence.")
    return spec


def arm_modules(spec):
    package_name = "hegel_machine.cognition._research_scope_baseline_v037"
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


def enrich_scope_summary(spec, output, phase):
    stem = "" if phase == "all" else f"{phase}_"
    summary = study.read(output / f"{stem}summary.json")
    compact = study.read(output / f"{stem}compact_summary.json")

    def metrics(rows):
        events = [event for row in rows for event in row["recovery"]["events"]]
        goals = [row.get("objective_research") or {} for row in rows]
        outcomes = [item for state in goals for item in state.get("outcomes", [])]
        available = [state for state in goals if "planning_passes" in state]
        keys = ("head_refits", "head_fit_records", "head_hvd_initializations",
                "head_refit_seconds", "scope_filtered_updates", "scope_filtered_fantasy_updates")
        return {"research_scope_revisions": sum(event["action"] == "revise_research_scope" for event in events),
            "resumed_research_plans": sum(event["action"] == "resume_research" for event in events),
            "objective_scope_revisions": sum(len(row.get("objective_scope_revisions", [])) for row in rows),
            "failed_objective_scope_attempts": sum(len(state.get("scope_attempts", [])) for state in goals),
            "unfinished_objective_scope_research": sum(state.get("scope_research") is not None for state in goals),
            "objective_scope_coverage_calls": sum(item["request"]["reason"] == "identify_objective_scope" for item in outcomes),
            "objective_scope_validation_calls": sum(item["request"]["reason"] == "validate_objective_scope" for item in outcomes),
            "pacing_counters_available_runs": len(available),
            **{key: sum(state.get(key, 0) for state in available) for key in
                ("planning_passes", "skipped_unchanged_evidence", "screened_controls", "screened_gp_controls")},
            **{key: sum(row["costs"].get("objective_diagnostics", {}).get(key, 0) for row in rows) for key in keys}}

    for group in compact["groups"]:
        paired = [item for item in summary["comparisons"]
            if (item["split"], item["profile"], item["noise"])
            == (group["split"], group["profile"], group["noise"])]
        tolerance = spec["revision"]["minimum_error"]
        group["paired_objective_change"] = {
            "improved": sum(item["objective_delta"] < -tolerance for item in paired),
            "equal": sum(abs(item["objective_delta"]) <= tolerance for item in paired),
            "worse": sum(item["objective_delta"] > tolerance for item in paired)}
        for arm in ARMS:
            rows = [row for row in summary["results"]
                if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], arm)]
            group[arm].update(metrics(rows))
            group[arm]["feasible_rate"] = group[arm]["feasible"] / len(rows) if rows else None
    compact["research_scope_sequences"] = [{"case_id": row["case_id"], "arm": row["arm"],
        "events": [{key: event[key] for key in ("action", "stage", "previous_scope_start", "scope_start") if key in event}
            for event in row["recovery"]["events"] if event["action"] in ("revise_research_scope", "resume_research")],
        "episodes": [{key: episode[key] for key in ("scope_start", "status", "subsequent_scope_start") if key in episode}
            for episode in row["recovery"]["episodes"]]}
        for row in summary["results"] if any(event["action"] in ("revise_research_scope", "resume_research")
            for event in row["recovery"]["events"])]
    def future_predictions(row, revision):
        return [{"stage": outcome["context"]["stage"],
            "global_index": outcome["context"]["global_index"],
            "features": outcome["request"]["features"],
            "candidate_error": (outcome["objective"] - outcome["request"]["candidate_mean"]) ** 2,
            "previous_error": (outcome["objective"] - outcome["request"]["previous_mean"]) ** 2}
            for outcome in (row.get("objective_research") or {}).get("outcomes", [])
            if outcome["request"]["reason"] == "validate_objective_scope"
            and outcome["context"]["global_index"] in revision["validation_indices"]]

    compact["objective_scope_sequences"] = [{"case_id": row["case_id"], "arm": row["arm"],
        "events": (row.get("objective_research") or {}).get("events", []),
        "search_compute": (row.get("objective_research") or {}).get("compute_cost", 0),
        "remaining_search_compute": (row.get("objective_research") or {}).get("compute_budget", 0)
            - (row.get("objective_research") or {}).get("compute_cost", 0),
        "failed_attempts": [{**{key: attempt[key] for key in
            ("scope_start", "phase", "status", "training_indices", "validation") if key in attempt},
            "candidate_training_rows": (attempt.get("candidate") or {}).get("training_rows")}
            for attempt in (row.get("objective_research") or {}).get("scope_attempts", [])],
        "unfinished": {**{key: (row.get("objective_research") or {}).get("scope_research", {})[key]
            for key in ("scope_start", "phase", "status", "training_indices", "validation", "coverage_calls")
            if key in ((row.get("objective_research") or {}).get("scope_research") or {})},
            "candidate_training_rows": (((row.get("objective_research") or {}).get("scope_research") or {})
                .get("candidate") or {}).get("training_rows")},
        "actual_query_stages": [{key: request[key] for key in ("stage", "purpose", "reason")}
            for request in row["costs"].get("objective_diagnostics", {}).get("requests", [])
            if request["reason"] in ("identify_objective_scope", "validate_objective_scope")],
        "applied": [{**{key: revision[key] for key in ("stage", "scope_start", "fit_records", "total_paid_records",
            "training_indices", "validation_indices", "validation_mse", "previous_mse") if key in revision},
            "candidate_training_rows": revision.get("model", {}).get("training_rows"),
            "future_predictions": future_predictions(row, revision)}
            for revision in row.get("objective_scope_revisions", [])]}
        for row in summary["results"] if (row.get("objective_research") or {}).get("events")]
    fresh = [row for row in summary["results"] if row.get("run_origin") != spec["historical_baseline_origin"]]
    compact["actual_new_scope_and_pacing_costs"] = metrics(fresh)
    compact["constraint_rebuild_after_objective_head"] = [
        {"case_id": row["case_id"], "arm": row["arm"],
         "objective_stage": objective["stage"], "objective_scope_start": objective["scope_start"],
         "constraint_stage": constraint["stage"], "constraint_scope_start": constraint["scope_start"],
         "same_stage": objective["stage"] == constraint["stage"]}
        for row in summary["results"]
        for objective in row.get("objective_scope_revisions", [])
        for constraint in row.get("scope_revisions", [])
        if constraint["stage"] >= objective["stage"]]
    compact["separate_scope_only_development_probe"] = {
        "summary": str((output / "scope_only_development/summary.json").resolve()),
        "additional_online_calls": 32, "included_in_main_method_totals": False,
        "method": "saved0.37 adapter and goal policy with new JointModelRevision only"}
    probe_path = output / "scope_only_development/compact_summary.json"
    if probe_path.exists():
        probe = study.read(probe_path)
        compact["separate_scope_only_development_probe"].update(
            combined_search_compute=probe["actual_new_combined_search_compute"],
            objective_search_compute=probe["actual_new_objective_research_costs"]["objective_search_compute"],
            constraint_revision_search_compute=probe["costs"]["actual_new_recovery_compute"],
            objective_forecast_rows=probe["actual_new_objective_research_costs"]["mean_prediction_rows"])
    attempt_path = output / "development_attempt_1_fixed_center/accounting.json"
    if attempt_path.exists():
        compact["separate_failed_goal_design_development"] = {
            **study.read(attempt_path),
            "summary": str((attempt_path.parent / "replay_compact_summary.json").resolve()),
            "included_in_main_method_totals": False}
        compact["online_calls_including_separate_development"] = (
            compact["costs"]["actual_new_optimizer_calls"] + 32
            + compact["separate_failed_goal_design_development"]["actual_new_online_calls"])
        compact["actual_new_search_including_separate_development"] = (
            compact["actual_new_combined_search_compute"]
            + compact["separate_scope_only_development_probe"]["combined_search_compute"]
            + compact["separate_failed_goal_design_development"]["combined_objective_and_constraint_search_compute"])
    study.write(output / f"{stem}compact_summary.json", compact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    parser.add_argument("--case", action="append", help="Select cases from the frozen roster.")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = repair.ARMS = goal.ARMS = ARMS
    study.controlled_problem = goal.controlled_problem
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen research-scope roster or method differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 139,
            "confirmation_cases": 18, "stress_cases": 9, "new_optimizer_runs": 193}))
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
    enrich_scope_summary(spec, args.output, args.phase)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare paid-replicate noise revision with the executed default SC-KG adapter."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import types

import run_sc_kg_supported_exploration_study as previous

framework, study = previous.framework, previous.study
ARMS = ("frozen_v042_supported_exploration", "noise_revision")
SEEDS = (1443011409, 1443111423, 1443211437)
REPLAY_CASES = (
    "research_scope_3_objective_only_change_noise_0.03",
    "value_precision_3_mechanism_change_noise_0.03",
    "supported_exploration_1_objective_only_change_noise_0.03",
    "research_value_1_objective_only_change_noise_0.03",
    "forward_1_mechanism_change_noise_0.00",
    "joint_evidence_1_objective_only_change_noise_0.03",
    "decision_single_outlier_2_state_policy_noise_0.03",
    *(f"research_value_3_{profile}_noise_{noise:.2f}" for profile, noise in (
        ("mechanism_change", 0.), ("stable_state_policy", 0.), ("stable_rzdt2", 0.),
        ("mechanism_change", .03), ("stable_state_policy", .03), ("stable_rzdt2", .03),
        ("single_observation_outlier", .03), ("outlier_then_change", .03),
        ("objective_only_change", .03))),
)
CONTROL_CASES = ("research_value_3_stable_state_policy_noise_0.00",
                 "research_value_3_stable_rzdt2_noise_0.00")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    for key in ("diagnostic_replay_path", "previous_precision_attempt_path",
                "previous_attempt_path", "expected_cumulative_optimizer_runs"):
        spec.pop(key, None)
    spec.update(experiment="paid_replicate_noise_model_revision",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_default_v043").resolve()),
        frozen_method_source=str((output / "method_source_v045").resolve()),
        historical_baseline_arm=ARMS[0], historical_baseline_origin="historical_v042_supported_exploration")
    seed = next(case["task"]["seed"] for case in spec["cases"] if case["split"] == "confirmation")
    templates = [deepcopy(case) for case in spec["cases"]
        if case["task"]["seed"] == seed and case["split"] in ("confirmation", "stress")]
    if len(templates) != 9 or set(SEEDS) & {case["task"]["seed"] for case in spec["cases"]}:
        raise ValueError("Require nine existing profile templates and three unused seeds")
    for case in spec["cases"]:
        case["previous_split"], case["split"] = case["split"], "development"
    old_cases = list(spec["cases"])
    for index, seed in enumerate(SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case["case_id"] = f"noise_revision_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"noise_revision_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    inventory = study.read(output / "entry_inventory.json")
    spec["unchanged_entry_reuse"] = inventory["reused_no_repeat_cases"]
    spec["affected_development_cases"] = [case["case_id"] for case in old_cases
        if case["case_id"] not in spec["unchanged_entry_reuse"]]
    spec["actual_control_cases"] = inventory["actually_executed_no_repeat_controls"]
    spec["reused_control_cases"] = []
    spec["actual_unchanged_control"] = CONTROL_CASES[0]
    spec["expected_new_optimizer_runs"] = inventory["actual_total_optimizer_runs"]
    spec["new_method"] = study.read(output / "method.json")
    spec["baseline_scope"] = "Reuse all355 executed default v0.42 outcomes, including four infeasible recommendations; package0.43 includes no adopted v0.43 or v0.44 policy changes."
    spec["scope"] = "Sixteen old cases diagnose all four infeasible defaults and stable, change, outlier and mixed profiles. Freeze the final method before acquiring three unused seeds, each with six ordinary and three stress profiles. Keep every outcome."
    spec["reuse_accounting"] = "Reuse old baselines, paid initial vectors and evaluation. Of156 old paths without any repeated control, three were actually executed as development controls; reuse the other153 unchanged method paths. Run202 old new-arm trajectories and both arms on27 fresh cases:256 optimizer runs/8192 online calls, including the16 development runs without repeating them. Fresh initial384 and evaluation1920 bring this implementation to10496 physical calls; the retained first16-case attempt adds512, cumulative11008. All355 old plus27 new cases remain in comparison. No extra vanilla or CodeAgent runs."
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for key in spec["unchanged_entry_reuse"]:
        destination = output / "runs" / f"{key}_{ARMS[1]}.json"
        if destination.exists():
            continue
        original = source / "runs" / f"{key}_{ARMS[0]}.json"
        row = study.read(original)
        row.update(arm=ARMS[1], split="development", run_origin=spec["historical_baseline_origin"],
            original_result=str(original.resolve()),
            reuse_reason="No paid same-point repeats in the full64-row trajectory. No noise candidate can exist, so noise comparison and adopted-overlay paths never execute and the original decisions/RNG remain unchanged.")
        study.write(destination, row)


def arm_modules(spec):
    result = {}
    for arm, directory, suffix in ((ARMS[0], spec["frozen_baseline_source"], "baseline"),
                                  (ARMS[1], spec["frozen_method_source"], "method")):
        name = f"hegel_machine.cognition._noise_revision_v045_{suffix}"
        package = types.ModuleType(name)
        package.__path__ = [directory, str(study.ROOT / "src/hegel_machine/cognition")]
        sys.modules[name] = package
        modules = {item: importlib.import_module(f"{name}.{item}") for item in
                   ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
        result[arm] = (modules["sc_kg_environment"], modules["prior_adoption"].PriorAdoption,
                       modules["joint_model_revision"].JointModelRevision)
        if arm == ARMS[0]:
            study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    return result


def noise_metrics(rows):
    diagnostics = [row["costs"].get("objective_diagnostics", {}).get("noise_revision", {}) for row in rows]
    counters = sorted({key for diag in diagnostics for key, value in diag.items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)})
    events = [event for diag in diagnostics for event in diag.get("events", [])]
    comparisons = [event["comparison"] for event in events if event.get("kind") == "scope_reset"]
    comparisons += [state["comparison"] for diag in diagnostics for state in diag.get("current", {}).values()]
    distribution = {}
    first_test_stages = {}
    categories = {"fewer_than_four_future_repeats": 0, "four_or_more_unresolved": 0,
                  "baseline_supported_without_candidate_support": 0, "candidate_supported": 0}
    for comparison in comparisons:
        count = len(comparison["observations"])
        distribution[str(count)] = distribution.get(str(count), 0) + 1
        if count:
            stage = str(comparison["observations"][0]["context"]["stage"])
            first_test_stages[stage] = first_test_stages.get(stage, 0) + 1
        reasons = {decision["reason"] for decision in comparison["decisions"]}
        if "candidate_predicts_better" in reasons:
            category = "candidate_supported"
        elif "baseline_predicts_better" in reasons:
            category = "baseline_supported_without_candidate_support"
        else:
            category = "four_or_more_unresolved" if count >= 4 else "fewer_than_four_future_repeats"
        categories[category] += 1
    return {"runs_with_diagnostics": sum(bool(diag) for diag in diagnostics),
            "runs_with_adoption": sum(diag.get("adoptions", 0) > 0 for diag in diagnostics),
            **{key: sum(diag.get(key, 0) for diag in diagnostics) for key in counters},
            "event_count": len(events),
            "candidate_comparisons": {"count": len(comparisons),
                "future_repeat_count_distribution": distribution,
                "first_future_repeat_stage_distribution": first_test_stages,
                "first_test_too_late_for_four_within_N64": sum(count for stage, count
                    in first_test_stages.items() if int(stage) > 61),
                "timing_interpretation": "Stage is the first actual future same-point noise comparison, not inferred candidate creation. A first test after61 leaves fewer than three calls and cannot reach four tests within N64. Candidates with no future comparison have no recorded first-test stage.",
                **categories},
            "event_kinds": {kind: sum(event.get("kind") == kind for event in events)
                            for kind in sorted({event.get("kind", "unspecified") for event in events})}}


def enrich_summary(spec, output, phase):
    previous._enrich_precision(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    brief = study.read(output / f"{stem}result_brief.json")
    previous.add_outcome_groups(brief)
    compact = study.read(output / f"{stem}compact_summary.json")
    summary = study.read(output / f"{stem}summary.json")
    rows = summary["results"]
    actual = [row for row in rows if row.get("run_origin") != spec["historical_baseline_origin"]]
    brief["actual_noise_revision"] = noise_metrics(actual)
    brief["fresh_noise_revision"] = {arm: noise_metrics([row for row in actual if row["arm"] == arm
        and row["split"] != "development"]) for arm in ARMS}
    by_case = {row["case_id"]: row for row in rows if row["arm"] == ARMS[1]}
    baselines = {row["case_id"]: row for row in rows if row["arm"] == ARMS[0]}
    for case in brief["cases"]:
        old, new = baselines[case["case_id"]], by_case[case["case_id"]]
        case["noise_revision"] = noise_metrics([new])
        case["paid_trajectory_changed"] = old["simulations"] != new["simulations"]
        case["recommendation_changed"] = old["final"]["x_recommended"] != new["final"]["x_recommended"]
        case["final_recommendation_certificate"] = new.get("final_recommendation_certificate")
    brief["noise_revision_behavior"] = {group: {
        "cases": len(selected),
        "paid_trajectories_changed": sum(case["paid_trajectory_changed"] for case in selected),
        "recommendations_changed": sum(case["recommendation_changed"] for case in selected),
        "adopted_runs": sum(case["noise_revision"]["runs_with_adoption"] for case in selected),
        "adopted_runs_with_changed_trajectory": sum(case["paid_trajectory_changed"]
            and case["noise_revision"]["runs_with_adoption"] > 0 for case in selected),
    } for group, selected in (("overall", brief["cases"]), ("fresh", [case for case in brief["cases"]
        if case["split"] != "development"]))}
    comparable = [key for key, row in by_case.items()
        if "exact_kg_stages" in row["costs"] and "exact_kg_stages" in baselines[key]["costs"]]
    brief["matched_complete_kg"] = {"paired_cases": len(comparable),
        "unavailable_pairs": len(by_case) - len(comparable),
        "seconds": [sum(event.get("t_kg_compute") or 0 for key in comparable
            for event in mapping[key]["costs"]["exact_kg_stages"]) for mapping in (baselines, by_case)]}
    brief["noise_revision_accounting"] = "The method uses already paid observations; all actual N64 simulation calls, noise-model fitting and predictive comparisons are retained. Noise timing inside a parent timing total must not be added twice."
    compact["current_study_accounting"] = spec["reuse_accounting"]
    compact["actual_noise_revision"] = brief["actual_noise_revision"]
    compact["fresh_noise_revision"] = brief["fresh_noise_revision"]
    compact["noise_revision_behavior"] = brief["noise_revision_behavior"]
    compact["matched_complete_kg"] = brief["matched_complete_kg"]
    first_attempt = output / "development_attempt_v1/replay_result_brief.json"
    if first_attempt.exists():
        first = study.read(first_attempt)
        brief["previous_development_attempt"] = {"path": str(first_attempt.resolve()),
            "feasibility_first": first["feasibility_first"],
            "actual_new_physical_calls": first["actual_new_physical_calls"],
            "actual_noise_revision": first["actual_noise_revision"]}
        brief["actual_cumulative_physical_calls"] = (brief["actual_new_physical_calls"]
            + first["actual_new_physical_calls"])
        compact["previous_development_attempt"] = brief["previous_development_attempt"]
        compact["actual_cumulative_physical_calls"] = brief["actual_cumulative_physical_calls"]
    study.write(output / f"{stem}result_brief.json", brief)
    study.write(output / f"{stem}compact_summary.json", compact)


def main():
    framework.ARMS, framework.REPLAY_CASES, framework.CONTROL_CASES = ARMS, REPLAY_CASES, CONTROL_CASES
    framework.protocol, framework.arm_modules = protocol, arm_modules
    framework.prepare_unaffected_reuse = prepare_unaffected_reuse
    framework.enrich_precision_summary = enrich_summary
    framework.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare active noise discrimination with the executed default SC-KG adapter."""
from copy import deepcopy
import importlib
from pathlib import Path
from statistics import mean, median
import sys
import types

import run_sc_kg_noise_revision_study as previous

framework, study = previous.framework, previous.study
ARMS = ("frozen_v042_supported_exploration", "active_noise_research")
SEEDS = (1543011541, 1543111559, 1543211573)
REPLAY_CASES = previous.REPLAY_CASES
CONTROL_CASES = previous.CONTROL_CASES


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    for key in ("diagnostic_replay_path", "previous_precision_attempt_path",
                "previous_attempt_path", "expected_cumulative_optimizer_runs"):
        spec.pop(key, None)
    spec.update(experiment="active_noise_discrimination_and_small_sample_uncertainty",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_default_v043").resolve()),
        frozen_method_source=str((output / "method_source_v046").resolve()),
        historical_baseline_arm=ARMS[0],
        historical_baseline_origin="historical_v042_supported_exploration")
    seed = next(case["task"]["seed"] for case in spec["cases"] if case["split"] == "confirmation")
    templates = [deepcopy(case) for case in spec["cases"]
        if case["task"]["seed"] == seed and case["split"] in ("confirmation", "stress")]
    if len(templates) != 9 or set(SEEDS) & {case["task"]["seed"] for case in spec["cases"]}:
        raise ValueError("Require nine profile templates and three unused seeds")
    for case in spec["cases"]:
        case["previous_split"], case["split"] = case["split"], "development"
    old_cases = list(spec["cases"])
    for index, seed in enumerate(SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case["case_id"] = f"active_noise_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"active_noise_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    spec["unchanged_entry_reuse"] = []
    spec["affected_development_cases"] = [case["case_id"] for case in old_cases]
    spec["actual_control_cases"] = list(CONTROL_CASES)
    spec["reused_control_cases"] = []
    spec["actual_unchanged_control"] = CONTROL_CASES[0]
    spec["expected_new_optimizer_runs"] = len(old_cases) + 2 * len(SEEDS) * len(templates)
    spec["new_method"] = study.read(output / "method.json")
    spec["baseline_scope"] = "Reuse all382 executed default v0.42 outcomes, including four infeasible recommendations. Package0.43 contains no adopted v0.43-v0.45 policy changes."
    spec["scope"] = "Run the sixteen prior development cases first. Freeze final method before acquiring three unused seeds, each with six ordinary and three stress profiles. Retain all382 old and27 new cases, including regressions."
    spec["reuse_accounting"] = "Reuse all382 old default results, paid initial vectors and evaluation. Actively buying repeats may alter every old trajectory, so rerun all382 new-arm trajectories with no no-repeat reuse. Both arms run on27 fresh cases. Total436 optimizer runs/13952 online calls, including the sixteen development runs once; fresh initial384 and evaluation1920 give16256 physical calls. No vanilla or CodeAgent executions."
    return spec


def arm_modules(spec):
    result = {}
    for arm, directory, suffix in ((ARMS[0], spec["frozen_baseline_source"], "baseline"),
                                  (ARMS[1], spec["frozen_method_source"], "method")):
        name = f"hegel_machine.cognition._active_noise_v046_{suffix}"
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
    updates = [event for event in events if event.get("kind") == "future_repeat"]
    first_updates = [min(event["stage"] for event in diag.get("events", [])
        if event.get("kind") == "future_repeat") for diag in diagnostics
        if any(event.get("kind") == "future_repeat" for event in diag.get("events", []))]
    return {"runs_with_diagnostics": sum(bool(diag) for diag in diagnostics),
            "runs_with_future_repeat_update": len(first_updates),
            "runs_with_update_before_last_observation": sum(stage < 64 for stage in first_updates),
            "first_update_stage_distribution": {str(stage): first_updates.count(stage)
                for stage in sorted(set(first_updates))},
            "future_updates_favoring_candidate": sum(event["weight_after"] > event["weight_before"]
                for event in updates),
            "future_updates_favoring_baseline": sum(event["weight_after"] < event["weight_before"]
                for event in updates),
            **{key: sum(diag.get(key, 0) for diag in diagnostics) for key in counters},
            "event_count": len(events),
            "event_kinds": {kind: sum(event.get("kind") == kind for event in events)
                            for kind in sorted({event.get("kind", "unspecified") for event in events})}}


def common_feasible_objectives(cases):
    paired = [{"case_id": case["case_id"], "objective": case["objective"],
        "relative_reduction": (case["objective"][0] - case["objective"][1]) / abs(case["objective"][0])}
        for case in cases if case["feasible"] == [True, True]]
    gains = [case["relative_reduction"] for case in paired]
    improvements = [case for case in paired if case["relative_reduction"] > 0]
    degradations = [case for case in paired if case["relative_reduction"] < 0]
    return {"cases": len(paired), "mean_relative_reduction": mean(gains) if gains else None,
        "median_relative_reduction": median(gains) if gains else None,
        "greatest_improvement": max(improvements, key=lambda case: case["relative_reduction"]) if improvements else None,
        "greatest_degradation": min(degradations, key=lambda case: case["relative_reduction"]) if degradations else None,
        "interpretation": "Positive relative reduction favors the new method. Include only recommendations feasible in both arms; feasibility changes remain separate in feasibility_first."}


def paid_repeat_allocation(rows):
    result = {"overall": dict.fromkeys(ARMS, 0), "fresh": dict.fromkeys(ARMS, 0)}
    for row in rows:
        seen = {tuple(point) for point in row["initial_points"]}
        repeated = 0
        for observation in row["simulations"]:
            point = tuple(observation["point"])
            repeated += point in seen
            seen.add(point)
        result["overall"][row["arm"]] += repeated
        if row["split"] != "development":
            result["fresh"][row["arm"]] += repeated
    return result


def enrich_summary(spec, output, phase):
    previous.previous._enrich_precision(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    brief = study.read(output / f"{stem}result_brief.json")
    previous.previous.add_outcome_groups(brief)
    compact = study.read(output / f"{stem}compact_summary.json")
    rows = study.read(output / f"{stem}summary.json")["results"]
    brief["paid_repeat_allocation"] = paid_repeat_allocation(rows)
    actual = [row for row in rows if row.get("run_origin") != spec["historical_baseline_origin"]]
    by_case = {row["case_id"]: row for row in rows if row["arm"] == ARMS[1]}
    baselines = {row["case_id"]: row for row in rows if row["arm"] == ARMS[0]}
    for case in brief["cases"]:
        old, new = baselines[case["case_id"]], by_case[case["case_id"]]
        case["noise_revision"] = noise_metrics([new])
        case["paid_trajectory_changed"] = old["simulations"] != new["simulations"]
        case["recommendation_changed"] = old["final"]["x_recommended"] != new["final"]["x_recommended"]
        case["final_recommendation_certificate"] = new.get("final_recommendation_certificate")
    brief["actual_noise_revision"] = noise_metrics(actual)
    brief["fresh_noise_revision"] = {arm: noise_metrics([row for row in actual if row["arm"] == arm
        and row["split"] != "development"]) for arm in ARMS}
    brief["noise_revision_behavior"] = {group: {
        "cases": len(selected),
        "paid_trajectories_changed": sum(case["paid_trajectory_changed"] for case in selected),
        "recommendations_changed": sum(case["recommendation_changed"] for case in selected),
        "runs_with_future_repeat_update": sum(case["noise_revision"]["runs_with_future_repeat_update"]
            for case in selected),
    } for group, selected in (("overall", brief["cases"]), ("fresh", [case for case in brief["cases"]
        if case["split"] != "development"]))}
    comparable = [key for key, row in by_case.items()
        if "exact_kg_stages" in row["costs"] and "exact_kg_stages" in baselines[key]["costs"]]
    brief["matched_complete_kg"] = {"paired_cases": len(comparable),
        "unavailable_pairs": len(by_case) - len(comparable),
        "seconds": [sum(event.get("t_kg_compute") or 0 for key in comparable
            for event in mapping[key]["costs"]["exact_kg_stages"]) for mapping in (baselines, by_case)]}
    brief["common_feasible_objective"] = {group: common_feasible_objectives(selected)
        for group, selected in (("overall", brief["cases"]),
            ("development", [case for case in brief["cases"] if case["split"] == "development"]),
            ("fresh", [case for case in brief["cases"] if case["split"] != "development"]))}
    brief["mean_prediction_mse"] = {group: {key: [mean(case[key][arm] for case in selected)
        for arm in (0, 1)] for key in ("objective_gp_true_mse", "constraint_gp_true_mse")}
        for group, selected in (("overall", brief["cases"]),
            ("fresh", [case for case in brief["cases"] if case["split"] != "development"])) if selected}
    brief["noise_revision_accounting"] = "All actual N64 simulations are charged; active repeats consume the existing online budget. Noise fitting, comparison and fantasy times nested in complete KG timing are subsets and must not be added twice."
    compact["current_study_accounting"] = spec["reuse_accounting"]
    for key in ("actual_noise_revision", "fresh_noise_revision", "noise_revision_behavior", "matched_complete_kg",
                "common_feasible_objective", "mean_prediction_mse", "paid_repeat_allocation"):
        compact[key] = brief[key]
    study.write(output / f"{stem}result_brief.json", brief)
    study.write(output / f"{stem}compact_summary.json", compact)


def main():
    framework.ARMS, framework.REPLAY_CASES, framework.CONTROL_CASES = ARMS, REPLAY_CASES, CONTROL_CASES
    framework.protocol, framework.arm_modules = protocol, arm_modules
    framework.prepare_unaffected_reuse = lambda spec, output: None
    framework.enrich_precision_summary = enrich_summary
    framework.main()


if __name__ == "__main__":
    main()

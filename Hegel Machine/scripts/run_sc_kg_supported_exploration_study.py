#!/usr/bin/env python3
"""Condition accepted objective heads on paid labels and retain supported research offers."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys
import types

import run_sc_kg_value_precision_study as framework

study = framework.study
ARMS = ("frozen_v041_value_precision", "supported_exploration")
SEEDS = (1243011013, 1243111027, 1243211041)
CONTROL_CASES = ("value_precision_3_mechanism_change_noise_0.03",
    "research_scope_3_stable_rzdt2_noise_0.00", "forward_1_mechanism_change_noise_0.03")
REPLAY_CASES = ("research_scope_2_objective_only_change_noise_0.03",
    "forward_1_mechanism_change_noise_0.00", "objective_only_change_3_noise_0.03",
    "value_precision_1_objective_only_change_noise_0.03",
    "value_precision_2_objective_only_change_noise_0.03", *CONTROL_CASES)
_enrich_precision = framework.enrich_precision_summary


def protocol(source, output, method=None):
    spec = deepcopy(study.read(source / "protocol.json"))
    for key in ("previous_precision_attempt_path", "expected_cumulative_optimizer_runs"):
        spec.pop(key, None)
    spec.update(experiment="paid_objective_posterior_and_supported_exploration",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v041").resolve()),
        historical_baseline_arm="value_precision", historical_baseline_origin="historical_v041_value_precision")
    spec["baseline_method"] = spec.pop("new_method")
    seed = next(case["task"]["seed"] for case in spec["cases"] if case["split"] == "confirmation")
    templates = [deepcopy(case) for case in spec["cases"]
        if case["task"]["seed"] == seed and case["split"] in ("confirmation", "stress")]
    if len(templates) != 9:
        raise ValueError("six ordinary plus three stress profiles are required per seed")
    for case in spec["cases"]:
        case["previous_split"], case["split"] = case["split"], "development"
    for index, seed in enumerate(SEEDS, 1):
        for template in templates:
            case = deepcopy(template)
            case["task"]["seed"] = seed
            noise, problem = case["task"]["noise"], case["task"]["problem"]
            case["case_id"] = f"supported_exploration_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"supported_exploration_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    inventory = study.read(output / "entry_inventory.json")
    spec["affected_development_cases"] = [row["case_id"] for row in inventory["entries"] if row["head_stages"]]
    spec["unchanged_entry_reuse"] = [row["case_id"] for row in inventory["entries"] if not row["head_stages"]]
    spec["actual_control_cases"] = []
    spec["reused_control_cases"] = list(CONTROL_CASES)
    spec["actual_unchanged_control"] = CONTROL_CASES[1]
    spec["expected_new_optimizer_runs"] = len(spec["affected_development_cases"]) + 54
    spec["new_method"] = method if method is not None else study.read(output / "method.json")
    spec["diagnostic_replay_path"] = str((output / "instrumented_baseline_replay.json").resolve())
    spec["baseline_scope"] = "All274 final0.41 actual runs are reused as development comparators; the baseline is the executed eight-module0.41 snapshot, directly compared to current production before editing."
    spec["scope"] = "Three unused seeds each provide six ordinary cases and three noisy stress profiles: single constraint outlier, outlier followed by real change, and objective-only change. No outcomes are dropped."
    spec["reuse_accounting"] = (
        "Both changes only enter after an accepted objective head. Rerun74 headed development "
        "trajectories and reuse200 no-head trajectories, including36 with constraint-only scope "
        "reconstruction. Add54 fresh arms:128 actual optimizer runs/4096 online calls, all N64. "
        "The independent frozen0.41 instrumented replay adds32 online calls, no new initial/evaluation "
        "vectors, and is separately charged; it is never reused as a0.42 result. Fresh initial "
        "and evaluation data are measured once. All search, posterior updates and full acquisition "
        "costs are recorded without adding nested timing subsets twice.")
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for key in spec["unchanged_entry_reuse"]:
        path = output / "runs" / f"{key}_{ARMS[1]}.json"
        if path.exists():
            continue
        original = source / "runs" / f"{key}_value_precision.json"
        row = study.read(original)
        row.update(arm=ARMS[1], split="development", run_origin=spec["historical_baseline_origin"],
            original_result=str(original.resolve()), reuse_reason="No installed objective head; neither objective-head conditioning nor the revised terminal nomination is reached. Constraint-only scope rebuilding remains unchanged.")
        study.write(path, row)


def arm_modules(spec):
    name = "hegel_machine.cognition._supported_exploration_baseline_v041"
    package = types.ModuleType(name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[name] = package
    baseline = {item: importlib.import_module(f"{name}.{item}") for item in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    study.MetaPriorResearch = importlib.import_module(f"{name}.meta_prior_research").MetaPriorResearch
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
        baseline["joint_model_revision"].JointModelRevision),
        ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}


def add_outcome_groups(brief):
    def counts(rows, key):
        return {
            "improved": sum(row[key][1] < row[key][0] - 1e-9 for row in rows),
            "unchanged": sum(abs(row[key][1] - row[key][0]) <= 1e-9 for row in rows),
            "worse": sum(row[key][1] > row[key][0] + 1e-9 for row in rows),
        }
    groups = {"overall": brief["cases"],
        "development": [row for row in brief["cases"] if row["split"] == "development"],
        "fresh": [row for row in brief["cases"] if row["split"] != "development"]}
    brief["feasibility_first"] = {}
    brief["prediction_loss_by_split"] = {}
    for name, rows in groups.items():
        equal_feasibility = [row for row in rows if row["feasible"][0] == row["feasible"][1]]
        outcome = counts(equal_feasibility, "objective")
        outcome["improved"] += sum(row["feasible"] == [False, True] for row in rows)
        outcome["worse"] += sum(row["feasible"] == [True, False] for row in rows)
        brief["feasibility_first"][name] = {"cases": len(rows), **outcome,
            "feasible": [sum(row["feasible"][i] for row in rows) for i in (0, 1)]}
        brief["prediction_loss_by_split"][name] = {
            key: counts(rows, key) for key in ("objective_gp_true_mse", "constraint_gp_true_mse")}
    brief["outcome_interpretation"] = (
        "feasibility_first counts losing true feasibility as worse and gaining it as improved; "
        "with equal feasibility status it compares objective. Original overall/development/fresh "
        "fields retain raw objective comparisons, including an objective decrease that lost feasibility.")


def enrich_summary(spec, output, phase):
    _enrich_precision(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    brief = study.read(output / f"{stem}result_brief.json")
    add_outcome_groups(brief)
    brief["reused_control_behavior"] = {**brief.pop("actual_unchanged_controls"),
        **brief.pop("earlier_headed_control_behavior")}
    diagnostic = study.read(Path(spec["diagnostic_replay_path"]))
    brief["independent_frozen_diagnostic"] = {key: diagnostic[key] for key in
        ("case_id", "same_paid_trajectory", "initial_paid_vectors_reused", "new_physical_calls",
         "extra_diagnostic_prediction_calls", "replay_seconds", "objective_compute", "recovery_costs")}
    brief["independent_frozen_diagnostic"]["value_precision"] = {key: value for key, value in diagnostic["value_precision"].items() if key != "rounds"}
    brief["actual_cumulative_online_calls"] = brief["actual_new_costs"]["actual_new_optimizer_calls"] + diagnostic["new_physical_calls"]
    brief["actual_cumulative_physical_calls"] = brief["actual_new_physical_calls"] + diagnostic["new_physical_calls"]
    brief["actual_cumulative_search_compute"] = brief["actual_new_search"] + diagnostic["objective_compute"] + diagnostic["recovery_costs"]["compute_cost"]
    compact = study.read(output / f"{stem}compact_summary.json")
    summary = study.read(output / f"{stem}summary.json")
    fields = ("head_posterior_updates", "head_posterior_noise_predictions", "head_posterior_seconds")
    def metrics(rows):
        reason = "supported_goal_with_uncertain_constraint"
        return {**{key: sum(row["costs"].get("objective_diagnostics", {}).get(key, 0) for row in rows) for key in fields},
            "supported_boundary_nominations": sum(offer.get("reason") == reason for row in rows
                for event in row["costs"].get("objective_diagnostics", {}).get("terminal_candidate_sets", []) for offer in event["offers"]),
            "supported_boundary_paid_probes": sum(request.get("reason") == reason for row in rows
                for request in row["costs"].get("objective_diagnostics", {}).get("requests", [])),
            "final_recommendation_equals_supported_probe": sum(any(request.get("reason") == reason
                and request["point"] == row["final"]["x_recommended"] for request in
                row["costs"].get("objective_diagnostics", {}).get("requests", [])) for row in rows)}
    for group in compact["groups"]:
        for arm in ARMS:
            rows = [row for row in summary["results"] if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], arm)]
            group[arm]["supported_exploration"] = metrics(rows)
    actual = [row for row in summary["results"] if row.get("run_origin") != spec["historical_baseline_origin"]]
    brief["actual_new_supported_exploration"] = metrics(actual)
    brief["fresh_supported_exploration"] = {arm: metrics([row for row in actual if row["arm"] == arm and row["split"] != "development"]) for arm in ARMS}
    by_case = {row["case_id"]: row for row in summary["results"] if row["arm"] == ARMS[1]}
    for case in brief["cases"]:
        case["supported_exploration"] = metrics([by_case[case["case_id"]]])
    brief["actual_cumulative_numerical_refinement_costs"] = {key: brief["actual_new_value_precision"][key] + diagnostic["value_precision"][key]
        for key in ("evaluations", "tie_forecast_rows", "score_one_calls", "terminal_value_calls", "terminal_pool_rows", "seconds")}
    brief.pop("actual_cumulative_native_kg", None)
    brief["additional_diagnostic_timing"] = "The frozen diagnostic records full replay time but no separate native KG total. Main-run KG times exclude it; its measured precision subset is separately included in cumulative numerical-refinement costs."
    brief["offline_saved_history_diagnostic"] = {"path": str((output / "saved_objective_posterior_comparison.json").resolve()),
        "new_simulator_calls": 0, "interpretation": "Already paid future controls, not new policy trajectories or additional policy gains."}
    compact["posterior_cost_interpretation"] = "head_posterior_seconds is included in head_refit_seconds, not added to it. Offline fixed-history comparison acquires no new simulator observations and is separate from paired policy outcomes."
    compact["actual_new_supported_exploration"] = brief["actual_new_supported_exploration"]
    compact["fresh_supported_exploration"] = brief["fresh_supported_exploration"]
    compact["current_study_accounting"] = spec["reuse_accounting"]
    for key in ("independent_frozen_diagnostic", "actual_cumulative_online_calls", "actual_cumulative_physical_calls", "actual_cumulative_search_compute"):
        compact[key] = brief[key]
    study.write(output / f"{stem}result_brief.json", brief)
    study.write(output / f"{stem}compact_summary.json", compact)


def main():
    framework.ARMS, framework.REPLAY_CASES, framework.CONTROL_CASES = ARMS, REPLAY_CASES, CONTROL_CASES
    framework.protocol, framework.arm_modules = protocol, arm_modules
    framework.prepare_unaffected_reuse, framework.enrich_precision_summary = prepare_unaffected_reuse, enrich_summary
    framework.main()


if __name__ == "__main__":
    main()

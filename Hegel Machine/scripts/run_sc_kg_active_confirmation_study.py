#!/usr/bin/env python3
"""Frozen active-confirmation study with a true single-outlier control."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import sys
import types

import run_sc_kg_joint_revision_study as study


SEEDS = (1443010201, 1443110227, 1443210241)
ARMS = ("frozen_v033_competition", "active_confirmation")


def protocol(source, output):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="first_counterexample_active_confirmation",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v033").resolve()))
    spec["baseline_method"] = spec.pop("new_method")
    spec.pop("method_frozen_before_execution", None)
    for name in ("joint_model_revision.py", "revision_candidates.py", "sc_kg_environment.py",
                 "prior_adoption.py", "meta_prior_claims.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError("the actual 0.33 implementation must be saved before active-confirmation edits")
    for case in spec["cases"]:
        case["previous_split"] = case["split"]
        case["split"] = "development"
    for index, seed in enumerate(SEEDS, 1):
        for noise in (0.0, 0.03):
            for profile, problem in (("mechanism_change", "state_policy_rzdt1"),
                                     ("stable_state_policy", "state_policy_rzdt1"),
                                     ("stable_rzdt2", "rzdt2")):
                spec["cases"].append({"case_id": f"active_confirmation_{index}_{profile}_noise_{noise:.2f}",
                    "split": "confirmation", "profile": profile,
                    "task": {"problem": problem, "dimension": 3, "seed": seed, "noise": noise},
                    "next_problem": "rzdt2" if profile == "mechanism_change" else problem,
                    "research_id": f"active_confirmation_{index}_{problem}_noise_{noise:.2f}"})
        spec["cases"].append({"case_id": f"single_outlier_{index}_state_policy_noise_0.03",
            "split": "stress", "profile": "single_observation_outlier",
            "task": {"problem": "state_policy_rzdt1", "dimension": 3, "seed": seed, "noise": 0.03},
            "next_problem": "state_policy_rzdt1",
            "research_id": f"active_confirmation_{index}_state_policy_rzdt1_noise_0.03",
            "observation_outlier": {"global_stage": 45, "online_call": 13, "constraint_delta": 1.0}})
    spec["baseline_scope"] = (
        "Development baselines are the actual completed 0.33 candidate-competition runs. "
        "Fresh baselines execute saved 0.33 joint_model_revision, revision_candidates, "
        "sc_kg_environment, prior_adoption, and meta_prior_claims. They retain the actual "
        "sliding-window detector, candidate competition, paid validation, and scope rebasing.")
    spec["new_method"] = {"status": "method_and_roster_frozen_before_optimization",
        "trigger": "A current actual observation exceeds the unchanged nine-times-frozen-reference error threshold for both predictions made before that observation, while the ordinary sliding window does not yet support reopening.",
        "required_remaining_calls": 8,
        "reservation": "One same-point confirmation, at least three new research controls, and four future validation calls must fit inside the original remaining budget.",
        "confirmation": "Repeat the same actual quantized control point. Freeze the anchor's two original pre-measurement means, target and threshold. The repeated target must exceed that frozen threshold for both original means and preserve each original residual's sign. Updated GP predictions, updated noise scales, change labels and evaluator truth do not decide confirmation.",
        "confirmed": "Begin the new evidence scope at the original anchor. Retain the previously frozen candidate competition; eight coverage controls may shrink with available budget, while four future validation calls remain reserved.",
        "refuted": "Keep every paid observation in the complete ledger, remove the original anomaly from the monitoring window, and resume ordinary optimization.",
        "accounting": "Record the actual same-point confirmation call with its purpose and cost. It replaces an ordinary optimizer query inside N64 and never adds an external simulator call.",
        "reference_multiplier": 9, "total_calls_per_arm": 64, "initial_calls_per_arm": 32}
    spec["scope"] = (
        "The previous 46 evaluated cases are development. Three frozen fresh seeds at "
        "noise0 and noise0.03 with mechanism change, stable StatePolicy, and stable RZDT2 "
        "form 18 confirmation cases. Separately, three noisy StatePolicy cases inject one "
        "constraint-measurement disturbance of +1.0 at global stage45, while the underlying "
        "mechanism remains unchanged. These stress cases are reported separately and never "
        "pooled into ordinary stable controls or mechanism-change effectiveness.")
    spec["outlier_control"] = (
        "The actual simulator runs normally and its constraint output receives the frozen "
        "+1.0 observation disturbance on the thirteenth online call only. Objectives, future "
        "measurements, and evaluator truth remain unchanged. A report-only log records the "
        "disturbance; the runtime receives only ordinary measured outputs and public controls. "
        "The corresponding ordinary stable case receives no disturbance. Initial data and "
        "final evaluation measurements are shared with that stable case.")
    spec["research_initialization"] = (
        "Fresh initial research uses the frozen0.33 claim compiler. Both optimizer arms "
        "share every resulting full-vector observation and the same initial program. "
        "The existing public stratified completion rule supplies 32 initial rows if research "
        "finishes sooner, preserving its original result and costs.")
    spec["reuse_accounting"] = (
        "Reuse all previously paid initial vectors, actual0.33 baseline runs, and independent "
        "evaluation measurements. Only the new arm reruns development cases. For confirmation "
        "and outlier stress cases, run both arms; no extra simulator calls beyond each N64. "
        "Separately count new and historical initialization, online observations, candidate "
        "compute, confirmation queries, and independent evaluation.")
    return spec


def prepare_reuse(spec, output):
    source = Path(spec["development_source"])
    historical_arm = spec.get("historical_baseline_arm", "joint_revision")
    historical_origin = spec.get("historical_baseline_origin", "historical_v033_competition")
    for case in spec["cases"]:
        if case["split"] != "development":
            continue
        key = case["research_id"]
        path = output / "research" / f"{key}.json"
        if not path.exists():
            saved = study.read(source / "research" / f"{key}.json")
            saved.update(research_origin=str((source / "research" / f"{key}.json").resolve()),
                         actual_new_research_calls=0, actual_new_initial_calls=0)
            study.write(path, saved)
        path = output / "runs" / f"{case['case_id']}_{ARMS[0]}.json"
        if not path.exists():
            original_path = source / "runs" / f"{case['case_id']}_{historical_arm}.json"
            row = study.read(original_path)
            row.update(split="development", arm=ARMS[0], run_origin=historical_origin,
                       original_result=str(original_path.resolve()))
            study.write(path, row)
    folder = output / "evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    for path in (source / "evaluation").glob("*.json"):
        if not (folder / path.name).exists():
            shutil.copyfile(path, folder / path.name)
    path = output / "initial_data_completion_amendment.json"
    if not path.exists():
        shutil.copyfile(source / path.name, path)


def arm_modules(spec):
    package_name = "hegel_machine.cognition._active_confirmation_baseline_v033"
    package = types.ModuleType(package_name)
    package.__path__ = [spec["frozen_baseline_source"], str(study.ROOT / "src/hegel_machine/cognition")]
    sys.modules[package_name] = package
    baseline = {name: importlib.import_module(f"{package_name}.{name}") for name in
        ("sc_kg_environment", "prior_adoption", "joint_model_revision")}
    from hegel_machine.cognition import sc_kg_environment
    from hegel_machine.cognition.joint_model_revision import JointModelRevision
    from hegel_machine.cognition.prior_adoption import PriorAdoption
    return {ARMS[0]: (baseline["sc_kg_environment"], baseline["prior_adoption"].PriorAdoption,
                      baseline["joint_model_revision"].JointModelRevision),
            ARMS[1]: (sc_kg_environment, PriorAdoption, JointModelRevision)}


def enrich_aggregate(summary):
    for row in summary["results"]:
        recovery = row.get("recovery") or {}
        objective_calls = improvements = 0
        for episode in recovery.get("episodes", []):
            for proposal in episode["proposals"]:
                plan = proposal.get("experiment_plan") or {}
                objective = plan.get("objective_model") or {}
                indices = proposal.get("validation", {}).get("observation_indices", [])
                for choice, index in zip(plan.get("selected", []), indices):
                    if choice["purpose"] not in ("objective_improvement", "objective_coverage"):
                        continue
                    objective_calls += 1
                    measured = recovery["observations"][index]
                    if (objective.get("incumbent") is not None
                            and measured["context"]["objective"] < objective["incumbent"]
                            and measured["observation"]["target"] <= objective["constraint_upper_bound"]):
                        improvements += 1
        row["experiment_decision_metrics"] = {"objective_guided_validation_calls": objective_calls,
            "objective_guided_actual_improvements_over_frozen_incumbent": improvements}
    for group in summary["aggregate"]:
        rows = [row for row in summary["results"] if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], group["arm"])]
        events = [event for row in rows for event in (row.get("recovery") or {}).get("events", [])]
        confirmations = [event for event in events if event["action"] == "measure_counterexample_confirmation"]
        group["confirmation_calls"] = len(confirmations)
        group["confirmed_anchors"] = sum(event["confirmed"] for event in confirmations)
        group["refuted_anchors"] = sum(not event["confirmed"] for event in confirmations)
        group["research_reopenings"] = sum(event["action"] == "reopen_hypothesis_research" for event in events)
        group["reopenings_on_unchanged_mechanism"] = (group["research_reopenings"]
            if group["profile"] != "mechanism_change" else 0)
        group["recovery_compute"] = sum(row["costs"]["recovery_compute"].get("compute_cost", 0) for row in rows)
        for purpose in ("coverage_calls", "validation_calls"):
            group[purpose] = sum(row["costs"]["recovery_compute"].get(purpose, 0) for row in rows)
        post = [row["phase_prediction_error"].get("after_first_accepted_replacement", {"rows": 0}) for row in rows]
        group["actual_prediction_rows_after_adoption"] = sum(item["rows"] for item in post)
        group["actual_active_mse_after_adoption"] = (sum(item["rows"] * item.get("active_mse", 0.) for item in post)
            / group["actual_prediction_rows_after_adoption"] if group["actual_prediction_rows_after_adoption"] else None)
        objective_rows = sum(item.get("objective_rows", 0) for item in post)
        group["actual_objective_rows_after_adoption"] = objective_rows
        group["actual_objective_mse_after_adoption"] = (sum(item.get("objective_rows", 0) * item.get("objective_mse", 0.)
            for item in post) / objective_rows if objective_rows else None)
        grid = [row["generalization"]["objective_gp_true_objective_means_mse"] for row in rows
                if "objective_gp_true_objective_means_mse" in row["generalization"]]
        group["objective_gp_grid_cases"] = len(grid)
        group["mean_final_objective_gp_true_mse"] = sum(grid) / len(grid) if grid else None
        group["objective_guided_validation_calls"] = sum(row["experiment_decision_metrics"]["objective_guided_validation_calls"] for row in rows)
        group["objective_guided_actual_improvements"] = sum(row["experiment_decision_metrics"]["objective_guided_actual_improvements_over_frozen_incumbent"] for row in rows)
    return summary


def compact_summary(summary):
    groups = []
    for entry in summary["aggregate"]:
        key = (entry["split"], entry["profile"], entry["noise"])
        group = next((item for item in groups if (item["split"], item["profile"], item["noise"]) == key), None)
        if group is None:
            group = {"split": key[0], "profile": key[1], "noise": key[2], "cases": entry["runs"]}
            groups.append(group)
        calls = entry["confirmation_calls"]
        group[entry["arm"]] = {"revisions": entry["accepted_replacements"],
            "reopenings": entry["research_reopenings"], "confirmations": calls,
            "confirmed": entry["confirmed_anchors"],
            "confirmation_support_rate": entry["confirmed_anchors"] / calls if calls else None,
            "program_mse": entry["mean_final_program_true_mse"], "gp_mse": entry["mean_final_gp_true_mse"],
            "objective": entry["mean_objective"], "feasible": entry["feasible"],
            "path_last20_mse": entry["mean_active_last20_forecast_mse"],
            "after_adoption_rows": entry["actual_prediction_rows_after_adoption"],
            "after_adoption_mse": entry["actual_active_mse_after_adoption"],
            "after_adoption_objective_rows": entry["actual_objective_rows_after_adoption"],
            "after_adoption_objective_mse": entry["actual_objective_mse_after_adoption"],
            "objective_gp_grid_cases": entry["objective_gp_grid_cases"],
            "objective_gp_grid_mse": entry["mean_final_objective_gp_true_mse"],
            "objective_guided_validation_calls": entry["objective_guided_validation_calls"],
            "objective_guided_actual_improvements": entry["objective_guided_actual_improvements"],
            "recovery_compute": entry["recovery_compute"],
            "coverage_calls": entry["coverage_calls"], "validation_calls": entry["validation_calls"]}
    regressions = []
    tolerance = summary["protocol"]["revision"]["minimum_error"]
    for comparison in summary["comparisons"]:
        old, new = [next(row for row in summary["results"] if row["case_id"] == comparison["case_id"] and row["arm"] == arm)
                    for arm in ARMS]
        gp_delta = new["generalization"]["active_gp_true_constraint_means_mse"] - old["generalization"]["active_gp_true_constraint_means_mse"]
        if comparison["objective_delta"] > tolerance or gp_delta > tolerance or (old["final"]["true_feasible"] and not new["final"]["true_feasible"]):
            regressions.append({"case_id": comparison["case_id"], "split": comparison["split"],
                "objective_old": old["final"]["true_objective"], "objective_new": new["final"]["true_objective"],
                "gp_mse_old": old["generalization"]["active_gp_true_constraint_means_mse"],
                "gp_mse_new": new["generalization"]["active_gp_true_constraint_means_mse"],
                "feasible_old": old["final"]["true_feasible"], "feasible_new": new["final"]["true_feasible"]})
    historical_origin = summary["protocol"].get("historical_baseline_origin", "historical_v033_competition")
    fresh = [row for row in summary["results"] if row.get("run_origin") != historical_origin]
    cost_keys = ("actual_new_research_calls", "actual_supplemental_initial_calls", "actual_new_initial_research_compute",
        "actual_new_optimizer_calls", "independent_evaluation_calls", "historically_reused_initial_calls",
        "historically_reused_optimizer_calls", "historically_reused_evaluation_calls")
    costs = {key: summary.get(key) for key in cost_keys}
    costs["actual_new_recovery_compute"] = sum(row["costs"]["recovery_compute"].get("compute_cost", 0) for row in fresh)
    costs["actual_new_confirmation_calls_within_N64"] = sum(row["costs"]["recovery_compute"].get("confirmation_calls", 0) for row in fresh)
    costs["actual_new_rebase_gp_fits"] = sum(row["costs"]["recovery_diagnostics"].get("scope_model_fits", 0) for row in fresh)
    costs["actual_new_rebase_fit_records"] = sum(row["costs"]["recovery_diagnostics"].get("scope_fit_records", 0) for row in fresh)
    costs["actual_new_paired_adoption_gp_fits"] = sum(row["costs"]["adoption_compute"].get("additional_model_fits", 0) for row in fresh)
    costs["actual_new_shadow_gp_updates"] = sum(row["costs"]["adoption_compute"].get("shadow_gp_updates", 0) for row in fresh)
    costs["actual_new_objective_evaluation_prediction_rows"] = sum(row["costs"].get("objective_evaluation_prediction_rows", 0) for row in fresh)
    unchanged_research = [{"case_id": row["case_id"], "arm": row["arm"],
        "accepted_replacements": len(row["scope_revisions"]),
        "sequence": [{key: event[key] for key in ("action", "stage", "confirmed", "scope_start", "supported") if key in event}
            for event in row["recovery"]["events"] if event["action"] in
                ("measure_counterexample_confirmation", "reopen_hypothesis_research", "validate_replacement")]}
        for row in summary["results"] if row["profile"] != "mechanism_change"
        and any(event["action"] == "reopen_hypothesis_research" for event in (row.get("recovery") or {}).get("events", []))]
    notes = ["Fixed-grid Program/GP errors use the same independent true means for both arms; noisy observation errors and all raw events remain in summary.json.",
        "Path prediction errors include different paid research controls and do not alone measure matched-location accuracy."]
    if any("objective_gp_true_objective_means_mse" not in row["generalization"] for row in summary["results"]):
        notes.append("Some historical baselines have no saved final objective-GP grid predictions. Their objective_gp_grid_cases is zero and MSE is unavailable, never replaced by zero or a retrospective refit.")
    if any(row["final"].get("total_time_sec", 0.) < 0 for row in summary["results"]):
        notes.append("An upstream time.time clock produced a negative duration; wall-clock total_time_sec is excluded from latency comparisons.")
    return {"groups": groups, "costs": costs, "regressions_above_research_tolerance": regressions,
        "unchanged_mechanism_research": unchanged_research,
        "all_records_N64": all(row["costs"]["total_simulator_calls"] == row["optimizer_history_count"] == 64 for row in summary["results"]),
        "all_paired_initial_data_equal": all(row["same_initial_points"] and row["same_initial_outputs"] for row in summary["comparisons"]),
        "outlier_prefixes_equal": all(row.get("same_measured_prefix_before_outlier", True) for row in summary["outlier_control_comparisons"]),
        "notes": notes}


def summarize(spec, output, cases, research, results, evaluations, phase):
    study.summarize(spec, output, cases, research, results, evaluations)
    path = output / ("summary.json" if len(cases) == len(spec["cases"]) else "development_summary.json")
    summary = enrich_aggregate(study.read(path))
    source = Path(spec["development_source"])
    historical_origin = spec.get("historical_baseline_origin", "historical_v033_competition")
    summary["actual_new_optimizer_calls"] = sum(row["costs"]["optimizer_new_simulator_calls"]
        for row in results if row.get("run_origin") != historical_origin)
    summary["historically_reused_optimizer_calls"] = sum(row["costs"]["optimizer_new_simulator_calls"]
        for row in results if row.get("run_origin") == historical_origin)
    summary["actual_supplemental_initial_calls"] = sum(row.get("supplemental_initial_calls", 0)
        for row in research.values() if row["research_origin"] == "new_public_measurements")
    summary["actual_new_initial_research_compute"] = sum(row["result"]["costs"]["compute_cost"]
        for row in research.values() if row["research_origin"] == "new_public_measurements")
    summary["historically_reused_initial_calls"] = sum(row["result"]["costs"]["observation_cost"]
        + row.get("supplemental_initial_calls", 0) for row in research.values()
        if row["research_origin"] != "new_public_measurements")
    summary["independent_evaluation_calls"] = sum(row["simulator_calls"] for key, row in evaluations.items()
        if not (source / "evaluation" / f"{key}.json").exists())
    summary["historically_reused_evaluation_calls"] = sum(row["simulator_calls"] for key, row in evaluations.items()
        if (source / "evaluation" / f"{key}.json").exists())
    summary["revision_sequences"] = [{"case_id": row["case_id"], "arm": row["arm"],
        "stages": [{key: event.get(key) for key in ("action", "stage", "family_id", "kind", "supported", "reason")}
            for event in (row.get("recovery") or {}).get("events", [])
            if event.get("action") != "observe_revision"],
        "after_adoption_actual_forecast": row["phase_prediction_error"].get("after_first_accepted_replacement")}
        for row in summary["results"]]
    summary["outlier_control_comparisons"] = []
    for case in cases:
        if case["profile"] != "single_observation_outlier":
            continue
        control_split = "confirmation" if case["split"] == "stress" else case["split"]
        stable = next(item for item in spec["cases"] if item["profile"] == "stable_state_policy"
            and item["split"] == control_split and item["task"] == case["task"])
        for arm in ARMS:
            disturbed = next(row for row in summary["results"] if row["case_id"] == case["case_id"] and row["arm"] == arm)
            ordinary = next((row for row in summary["results"] if row["case_id"] == stable["case_id"] and row["arm"] == arm), None)
            comparison = {"case_id": case["case_id"], "arm": arm,
                "recorded_disturbances": [row for row in disturbed["simulations"] if row.get("measurement_constraint_disturbance")],
                "accepted_replacements": len(disturbed["scope_revisions"]),
                "ordinary_control_available": ordinary is not None}
            if ordinary is not None:
                comparison.update(same_measured_prefix_before_outlier=disturbed["simulations"][:12] == ordinary["simulations"][:12],
                    objective_delta=disturbed["final"]["true_objective"] - ordinary["final"]["true_objective"],
                    program_true_mse_delta=disturbed["generalization"]["program_true_constraint_means_mse"] - ordinary["generalization"]["program_true_constraint_means_mse"],
                    gp_true_mse_delta=disturbed["generalization"]["active_gp_true_constraint_means_mse"] - ordinary["generalization"]["active_gp_true_constraint_means_mse"])
            summary["outlier_control_comparisons"].append(comparison)
    destination = output / ("summary.json" if phase == "all" else f"{phase}_summary.json")
    study.write(destination, summary)
    study.write(output / ("compact_summary.json" if phase == "all" else f"{phase}_compact_summary.json"), compact_summary(summary))
    if destination != path:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    args = parser.parse_args()
    study.ARMS = ARMS
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen active-confirmation roster differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 46,
            "confirmation_cases": 18, "single_outlier_stress_cases": 3, "new_optimizer_runs": 46 + 21 * 2}))
        return
    cases = spec["cases"]
    if args.phase == "replay":
        cases = [case for case in cases if case["case_id"] == "new_confirmation_1_mechanism_change_noise_0.03"]
    elif args.phase in ("development", "confirmation", "stress"):
        cases = [case for case in cases if case["split"] == args.phase]
    prepare_reuse(spec, args.output)
    modules = arm_modules(spec)
    study.MetaPriorResearch = importlib.import_module(
        "hegel_machine.cognition._active_confirmation_baseline_v033.meta_prior_research").MetaPriorResearch
    research = study.acquire_research(spec, args.output, cases)
    if args.phase == "research":
        return
    results = study.optimize(spec, args.output, cases, research, arm_modules=modules)
    evaluations = study.evaluate(spec, args.output, cases, results)
    summarize(spec, args.output, cases, research, results, evaluations, args.phase)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Use shared goal and constraint evidence under the original paid observation budget."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import types

import run_sc_kg_research_scope_study as scope

goal = scope.goal

study, accounting, repair = goal.study, goal.accounting, goal.repair
SEEDS = (2043010709, 2043110723, 2043210737)
ARMS = ("frozen_v038_research_scope", "joint_evidence")
REPLAY_CASES = ("active_confirmation_1_mechanism_change_noise_0.03",
    "active_confirmation_2_mechanism_change_noise_0.03",
    "confirmation_2_mechanism_change_noise_0.03",
    "new_confirmation_1_mechanism_change_noise_0.03",
    "new_confirmation_3_mechanism_change_noise_0.03",
    "objective_exploration_1_mechanism_change_noise_0.03",
    "objective_outlier_then_change_1_noise_0.03",
    "observation_repair_1_mechanism_change_noise_0.03",
    "repair_outlier_then_change_1_noise_0.03",
    "repair_outlier_then_change_2_noise_0.03")


def protocol(source, output, method=None):
    spec = deepcopy(study.read(source / "protocol.json"))
    spec.update(experiment="joint_goal_constraint_evidence_and_shared_observation_budget",
        development_source=str(source.resolve()), arms=list(ARMS),
        frozen_baseline_source=str((output / "baseline_source_v038").resolve()),
        historical_baseline_arm=ARMS[0],
        historical_baseline_origin="historically_reused_pre_independent_head_run")
    for name in (*repair.BASELINE_MODULES, "objective_research.py"):
        if not (Path(spec["frozen_baseline_source"]) / name).exists():
            raise FileNotFoundError(f"the actual pre-edit0.38 baseline is required: {name}")
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
            case["case_id"] = f"independent_goal_{index}_{case['profile']}_noise_{noise:.2f}"
            case["research_id"] = f"independent_goal_{index}_{problem}_noise_{noise:.2f}"
            spec["cases"].append(case)
    spec["new_method"] = method if method is not None else study.read(output / "method.json")
    spec["baseline_scope"] = (
        "Reuse all193 actual completed0.38 comparator runs from the first completed0.39 study, "
        "with every failure and regression. "
        "Fresh baselines load all seven saved0.38 runtime modules, including objective scope "
        "research, paced goal search, and the objective-basis preservation after a later constraint "
        "rebuild. All193 original comparators remain valid and unchanged.")
    spec["scope"] = (
        "All previous193 cases are development. Three unused seeds each provide six ordinary "
        "confirmation cases (two noise levels, true change, and two stable mechanisms) plus "
        "three separate noisy stress cases: one constraint outlier, outlier followed by true "
        "change, and objective-only change with unchanged constraint. Preserve all outcomes.")
    spec["research_initialization"] = (
        "Use the saved0.38 claim compiler for fresh initial research; retain the unchanged "
        "public completion rule to32 actual full vectors. Both arms share the same initial "
        "measurements and executable prior; same-task stress profiles reuse that initialization.")
    spec["reuse_accounting"] = (
        "The first method completed193 pairs using220 actual new optimizer runs/7040 online "
        "calls; keep that attempt and all its costs. Reuse193 actual0.38 comparator runs and "
        "183 actual0.39 new-arm runs whose trajectories never enter a changed independence "
        "branch. Rerun the10 affected development new arms and both arms on27 fresh cases: "
        "64 additional optimizer runs/2048 online calls, for9088 cumulative online calls. "
        "All arms retain32 shared initial plus32 online calls. Report current additional and "
        "cumulative costs separately; no unchanged trajectory or initial/evaluation vector is rebought.")
    spec["affected_development_cases"] = list(REPLAY_CASES)
    return spec


def prepare_unaffected_reuse(spec, output):
    source = Path(spec["development_source"])
    for case in study.read(source / "protocol.json")["cases"]:
        if case["case_id"] in spec["affected_development_cases"]:
            continue
        path = output / "runs" / f"{case['case_id']}_{ARMS[1]}.json"
        if path.exists():
            continue
        source_path = source / "runs" / path.name
        row = study.read(source_path)
        row.update(split="development", run_origin=spec["historical_baseline_origin"],
            reuse_source=str(source_path.resolve()),
            reuse_reason="This actual trajectory never entered a modified objective-head independence branch.")
        study.write(path, row)


def arm_modules(spec):
    package_name = "hegel_machine.cognition._joint_evidence_baseline_v038"
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



def enrich_joint_evidence_summary(spec, output, phase):
    scope.enrich_scope_summary(spec, output, phase)
    stem = "" if phase == "all" else f"{phase}_"
    summary = study.read(output / f"{stem}summary.json")
    compact = study.read(output / f"{stem}compact_summary.json")
    # The preceding study's extra scope-only probe is not an acquisition in this study.
    compact.pop("separate_scope_only_development_probe", None)
    compact["current_study_accounting"] = (
        "The first193-pair attempt and all its costs remain preserved. The10 affected new-arm "
        "replays use the final method and are reused within the final220-pair cohort. The183 "
        "unaffected new-arm results and193 original0.38 comparators are historical reuse.")

    def metrics(rows):
        goals = [(row.get("objective_research") or {}) for row in rows]
        probes = [probe for row in rows for probe in row["recovery"].get("applicability_probes", [])]
        shared = [outcome for state in goals for outcome in state.get("shared_outcomes", [])]
        return {
            "joint_evidence_recording_available_runs": sum(row.get("joint_evidence") is not None for row in rows),
            "applicability_probe_calls": sum(row["costs"]["recovery_compute"].get("applicability_probe_calls", 0) for row in rows),
            "applicability_confirmation_calls": sum(row["costs"]["recovery_compute"].get("applicability_confirmation_calls", 0) for row in rows),
            "applicability_confirmed": sum(probe["confirmed"] for probe in probes),
            "applicability_refuted": sum(not probe["confirmed"] for probe in probes),
            "applicability_control_prediction_rows": sum((row.get("joint_evidence") or {}).get("prediction_rows", 0) for row in rows),
            "shared_goal_prediction_records": sum(len(state.get("shared_predictions", [])) for state in goals),
            "shared_goal_validation_observations": len(shared),
            "objective_scope_validation_observations": len(shared) + sum(
                item["request"]["reason"] == "validate_objective_scope"
                for state in goals for item in state.get("outcomes", [])),
            "goal_head_installations_using_shared_validation": sum(
                any(outcome["context"]["global_index"] in revision["validation_indices"]
                    for outcome in (row.get("objective_research") or {}).get("shared_outcomes", []))
                for row in rows for revision in row.get("objective_scope_revisions", [])),
            "constraint_revisions_preserving_objective_head": sum(
                bool(revision.get("preserved_objective_head"))
                for row in rows for revision in row.get("scope_revisions", []))}

    for group in compact["groups"]:
        for arm in ARMS:
            rows = [row for row in summary["results"]
                if (row["split"], row["profile"], row["task"]["noise"], row["arm"])
                == (group["split"], group["profile"], group["noise"], arm)]
            group[arm].update(metrics(rows))
    for sequence in compact["objective_scope_sequences"]:
        row = next(item for item in summary["results"]
            if item["case_id"] == sequence["case_id"] and item["arm"] == sequence["arm"])
        state = row.get("objective_research") or {}
        for revision in sequence["applied"]:
            revision["future_predictions"] = [
                {"source": source, "stage": outcome["context"]["stage"],
                 "global_index": outcome["context"]["global_index"],
                 "features": outcome["request"]["features"],
                 "purpose": outcome["request"]["purpose"],
                 "candidate_error": (outcome["objective"] - outcome["request"]["candidate_mean"]) ** 2,
                 "previous_error": (outcome["objective"] - outcome["request"]["previous_mean"]) ** 2}
                for source, outcomes in (("dedicated", state.get("outcomes", [])),
                                         ("shared_constraint_query", state.get("shared_outcomes", [])))
                for outcome in outcomes if outcome["request"]["reason"] == "validate_objective_scope"
                and outcome["context"]["global_index"] in revision["validation_indices"]]
    compact["joint_evidence_sequences"] = [
        {"case_id": row["case_id"], "arm": row["arm"],
         "adapter": row.get("joint_evidence"),
         "applicability_probes": row["recovery"].get("applicability_probes", []),
         "pending_applicability_probe": row["recovery"].get("applicability_probe"),
         "shared_predictions": (row.get("objective_research") or {}).get("shared_predictions", []),
         "shared_outcomes": (row.get("objective_research") or {}).get("shared_outcomes", [])}
        for row in summary["results"] if row.get("joint_evidence") is not None]
    fresh = [row for row in summary["results"] if row.get("run_origin") != spec["historical_baseline_origin"]]
    compact["actual_new_joint_evidence_costs"] = metrics(fresh)
    prior = study.read(Path(spec["development_source"]) / "compact_summary.json")
    compact["previous_completed_attempt_costs"] = prior["costs"]
    compact["actual_cumulative_optimizer_calls"] = (
        prior["costs"]["actual_new_optimizer_calls"] + compact["costs"]["actual_new_optimizer_calls"])
    compact["actual_cumulative_initial_calls"] = sum(
        entry["actual_new_research_calls"] + entry["actual_supplemental_initial_calls"]
        for entry in (prior["costs"], compact["costs"]))
    compact["actual_cumulative_evaluation_calls"] = (
        prior["costs"]["independent_evaluation_calls"] + compact["costs"]["independent_evaluation_calls"])
    compact["actual_cumulative_search_compute"] = (
        prior["actual_new_combined_search_compute"] + compact["actual_new_combined_search_compute"])
    compact["constraint_research_in_unchanged_mechanisms"] = [
        {"case_id": row["case_id"], "arm": row["arm"], "profile": row["profile"],
         "split": row["split"],
         "reopening_stages": [event["stage"] for event in row["recovery"]["events"]
            if event["action"] == "reopen_hypothesis_research"],
         "applied_constraint_stages": [revision["stage"] for revision in row["scope_revisions"]]}
        for row in summary["results"]
        if row["profile"] in {"stable_state_policy", "stable_rzdt2",
                              "single_observation_outlier", "objective_only_change"}
        and any(event["action"] == "reopen_hypothesis_research" for event in row["recovery"]["events"])]
    compact["joint_evidence_accounting"] = (
        "Shared predictions are made before the actual constraint experiment. Its full paid "
        "vector then supplies constraint evidence and prospective goal validation within N64. "
        "All applicability selection predictions are included in the "
        "existing constraint search cap. Missing baseline counters are reported as unavailable.")
    study.write(output / f"{stem}compact_summary.json", compact)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "research", "replay", "development", "confirmation", "stress", "all"), default="plan")
    parser.add_argument("--case", action="append", help="Select cases from the frozen roster.")
    args = parser.parse_args()
    study.ARMS = accounting.ARMS = repair.ARMS = goal.ARMS = scope.ARMS = ARMS
    study.controlled_problem = goal.controlled_problem
    spec = protocol(args.source_study, args.output)
    path = args.output / "protocol.json"
    if path.exists() and study.read(path) != spec:
        raise ValueError("the frozen joint-evidence roster or method differs")
    study.write(path, spec)
    if args.phase == "plan":
        print(json.dumps({"protocol": str(path.resolve()), "development_cases": 193,
            "confirmation_cases": 18, "stress_cases": 9, "additional_optimizer_runs": 64}))
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
    evaluations = study.evaluate(spec, args.output, cases, results,
                                 environment_factory=goal.evaluation_environment)
    accounting.summarize(spec, args.output, cases, research, results, evaluations, args.phase)
    repair.enrich_repair_summary(spec, args.output, args.phase)
    goal.enrich_objective_summary(spec, args.output, args.phase)
    enrich_joint_evidence_summary(spec, args.output, args.phase)


if __name__ == "__main__":
    main()

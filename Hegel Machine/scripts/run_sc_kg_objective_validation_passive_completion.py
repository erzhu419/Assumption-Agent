#!/usr/bin/env python3
"""Complete the frozen passive-continuation comparison with 52 paid replays."""

import argparse
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
from statistics import mean

import run_sc_kg_objective_validation_ablation as ablation


validation, study = ablation.validation, ablation.study
BASELINE, FULL = validation.ARMS
PASSIVE = ablation.ARM
COHORTS = ("historical382", "historical27", "fresh27")
REUSE_RULE = (
    "The passive source differs only after a retained objective candidate has four validation "
    "labels: it skips the complete active continuation pool scan/quote and returns no active "
    "continuation plan. Retention, ordinary paid-label sharing and four/six/eight rounds remain. "
    "Every full trajectory that enters this retained state records continue_objective_validation. "
    "Reuse the full path only when that event is absent and no earlier actual passive record exists. "
    "The 13 existing passive cases without this event remain old executed controls, not new runs."
)


def has_continuation(row):
    return any(event.get("action") == "continue_objective_validation"
               for event in (row.get("objective_research") or {}).get("events", []))


def references(source):
    saved = study.read(source / "summary.json")
    spec = saved["protocol"]
    old, existing = validation.source_results(Path(spec["source_study"]))
    rows = {BASELINE: {}, FULL: {}}
    refs = {BASELINE: {}, FULL: {}}
    for ref in saved["result_references"]:
        key, arm = ref["case_id"], ref["arm"]
        if ref["case_record"]:
            row = study.read(ref["case_record"])
        else:
            row = (existing if ref["run_origin"] == "executed_v047_development" else old)[key]
        rows[arm][key] = row
        source_arm = row["arm"]
        refs[arm][key] = {**ref, "source_arm": source_arm}
    previous = study.read(source / "passive_ablation/summary.json")["result_references"]
    rows[PASSIVE] = {ref["case_id"]: study.read(ref["case_record"]) for ref in previous}
    refs[PASSIVE] = {ref["case_id"]: {**ref, "source_arm": PASSIVE,
                                  "run_origin": "existing_passive_ablation"} for ref in previous}
    # A missing or misrouted reference would invalidate the bounded case selection.
    expected = {case["case_id"] for case in spec["cases"]}
    if any(set(rows[arm]) != expected for arm in (BASELINE, FULL)):
        raise ValueError("the complete 436-case baseline/full references are required")
    return spec, rows, refs


def prepare(source, output):
    original, rows, refs = references(source)
    cases = original["cases"]
    entered = {key for key, row in rows[FULL].items() if has_continuation(row)}
    existing = set(rows[PASSIVE])
    new = [case["case_id"] for case in cases if case["case_id"] in entered - existing]
    reused = [case["case_id"] for case in cases if case["case_id"] not in entered | existing]
    # Changed coverage would require a new budget, not silently more simulations.
    counts = (len(cases), len(entered), len(existing), len(new), len(reused))
    if counts != (436, 55, 16, 52, 368):
        raise ValueError(f"the authorized completion inventory differs: {counts}")
    spec = {key: deepcopy(original[key]) for key in (
        "cases", "sc_kg_project_root", "research_config", "total_calls_per_arm",
        "change_after_new_calls", "optimizer_seed_offset", "optimizer_settings",
        "policy", "revision", "evaluation", "frozen_baseline_source")}
    spec.update(experiment="passive_objective_validation_bounded_completion", arms=[PASSIVE],
        source_validation_study=str(source.resolve()),
        frozen_method_source=str((source / "passive_ablation/method_source_passive").resolve()),
        frozen_method_origin=str((source / "passive_ablation/protocol.json").resolve()),
        method=ablation.ABLATION, new_cases=new, existing_passive_cases=sorted(existing),
        full_path_reuse_cases=reused, continued_full_cases=sorted(entered), reuse_rule=REUSE_RULE,
        expected_new_optimizer_runs=52, expected_new_online_calls=1664,
        expected_new_initial_calls=0, expected_new_evaluation_calls=0,
        existing_passive_optimizer_runs=16, existing_passive_online_calls=512,
        existing_passive_controls_without_continuation=len(existing - entered),
        research_initialization="Restore existing v0.48/research for fresh27; restore original "
            "v0.46/research for historical cases. Never invoke acquire_research or initial completion.",
        evaluation_reuse="Copy the existing v0.48 evaluation JSONs for the 52 new runs; evaluate "
            "only frozen predictions against these paid observations and saved truth means.",
        source_loading="Use validation.arm_modules with frozen_method_source set to the existing "
            "passive snapshot; use its Program class for initialization. The nine snapshot runtime "
            "modules override the unchanged common cognition support modules, as in the original "
            "v0.48 and sixteen-case ablation. The live optimizer initializer is never selected.",
        accounting="52 new passive runs cost1664 online calls. The existing16 cost512 in the "
            "earlier ablation. Together68 actual passive runs cover all55 continued full cases "
            "plus13 old noncontinuation controls. The remaining368 full paths are reused. "
            "All436 baseline/full references are already paid; new initial/evaluator calls are zero.",
        scope="Complete the existing 436 cases, preserving historical382/historical27/fresh27. "
            "Case selection uses entry into the altered branch, not performance outcomes. "
            "No additional fresh seeds, initial acquisition, evaluation acquisition or other arms.")
    protocol = output / "protocol.json"
    if protocol.exists() and study.read(protocol) != spec:
        raise ValueError("the saved passive-completion protocol differs")
    # Freeze the budget before preparing or running any new optimization.
    if not protocol.exists():
        study.write(protocol, spec)
    research, measurement_refs = {}, {"research": {}, "evaluation": {}}
    for case in cases:
        if case["case_id"] not in new:
            continue
        key = case["research_id"]
        if key not in research:
            root = source if case["cohort"] == "fresh27" else Path(original["source_study"])
            path = root / "research" / f"{key}.json"
            row = study.read(path)
            if (not row.get("optimizer_initial_environment_state") or
                    row["result"]["costs"]["observation_cost"] + row.get("supplemental_initial_calls", 0) != 32):
                raise ValueError(f"missing already paid 32-row initialization: {key}")
            research[key] = row
            measurement_refs["research"][key] = str(path.resolve())
        name = f"{case['next_problem']}_{case['task']['seed']}_{case['task']['noise']:.2f}.json"
        if name not in measurement_refs["evaluation"]:
            path = source / "evaluation" / name
            evaluation = study.read(path)
            # Missing targets/coordinates must stop before execution; never acquire replacements.
            if (evaluation["integer_points"] != study.evaluation_controls(spec, case)
                    or any(field not in evaluation for field in
                           ("measured_targets", "true_constraint_means", "true_objective_means"))):
                raise ValueError(f"the complete shared evaluator record is required: {name}")
            destination = output / "evaluation" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(path, destination)
            elif study.read(destination) != evaluation:
                raise ValueError(f"copied evaluation differs from the paid source: {name}")
            measurement_refs["evaluation"][name] = str(path.resolve())
    study.write(output / "measurement_references.json", measurement_refs)
    return spec, research, rows, refs


def metrics(rows):
    states = [row.get("objective_research") or {} for row in rows]
    requests = [request for state in states for request in state.get("requests", [])
                if request.get("reason") == "validate_objective_scope"]
    shared = [item for state in states for item in state.get("shared_outcomes", [])
              if item["request"].get("reason") == "validate_objective_scope"]
    diagnostics = [row["costs"].get("objective_diagnostics", {}) for row in rows]
    return {"cases": len(rows), "feasible": sum(row["final"]["true_feasible"] for row in rows),
        "mean_objective_gp_mse": mean(row["generalization"]["objective_gp_true_objective_means_mse"] for row in rows),
        "mean_constraint_gp_mse": mean(row["generalization"]["active_gp_true_constraint_means_mse"] for row in rows),
        "objective_head_adoptions": sum(len(row.get("objective_scope_revisions", [])) for row in rows),
        "cases_with_objective_head_adoption": sum(bool(row.get("objective_scope_revisions")) for row in rows),
        "objective_research_compute": sum(state.get("compute_cost", 0) for state in states),
        "initial_research_compute": sum(row["costs"].get("research_compute", 0) for row in rows),
        "constraint_recovery_compute": sum(row["costs"].get("recovery_compute", {}).get("compute_cost", 0) for row in rows),
        "active_validation_requests": len(requests),
        "active_continuation_requests": sum("continuation_plan" in request for request in requests),
        "shared_validation_observations": len(shared),
        "shared_validation_by_purpose": dict(Counter(item["request"].get("purpose") for item in shared)),
        "events": dict(Counter(event.get("action") for state in states for event in state.get("events", []))),
        "objective_diagnostic_totals": {field: sum(row.get(field, 0) for row in diagnostics) for field in
            ("controller_seconds", "forecast_seconds", "head_refit_seconds", "head_posterior_seconds",
             "forecast_rows", "mean_prediction_rows", "epistemic_prediction_rows", "noise_prediction_rows")}}


def semantic_equality(full, passive):
    def without_elapsed_time(value):
        if isinstance(value, dict):
            return {key: without_elapsed_time(item) for key, item in value.items()
                    if key not in ("value_precision_seconds", "posterior_seconds")}
        if isinstance(value, list):
            return [without_elapsed_time(item) for item in value]
        return value

    return {"paid_trajectory": full["simulations"] == passive["simulations"],
        "actions": without_elapsed_time(full["actions"]) == without_elapsed_time(passive["actions"]),
        "final_outcome": all(full["final"][key] == passive["final"][key] for key in
            ("x_recommended", "true_objective", "true_constraint_mean", "true_feasible", "n_simulations")),
        "frozen_predictions": full["frozen_evaluation_predictions"] == passive["frozen_evaluation_predictions"],
        "generalization": full["generalization"] == passive["generalization"],
        "objective_heads": without_elapsed_time(full.get("objective_scope_revisions"))
            == without_elapsed_time(passive.get("objective_scope_revisions")),
        "constraint_heads": full.get("scope_revisions") == passive.get("scope_revisions")}


def summarize(spec, output, rows, refs):
    groups = {"overall": spec["cases"], **{cohort: [case for case in spec["cases"]
              if case["cohort"] == cohort] for cohort in COHORTS}}
    comparisons, by_group = {}, {}
    for name, cases in groups.items():
        comparisons[name] = {}
        for label, left in (("baseline_to_passive", BASELINE), ("full_to_passive", FULL)):
            comparison = validation.reuse.comparison(cases, rows[left], rows[PASSIVE])
            metadata = {case["case_id"]: case for case in cases}
            for row in comparison["cases"]:
                row.update(split=metadata[row["case_id"]]["split"], cohort=metadata[row["case_id"]]["cohort"])
            comparison["common_feasible_objective"] = validation.active.common_feasible_objectives(comparison["cases"])
            comparisons[name][label] = comparison
        by_group[name] = {arm: metrics([rows[arm][case["case_id"]] for case in cases])
                          for arm in (BASELINE, FULL, PASSIVE)}
    details = []
    for case in spec["cases"]:
        key = case["case_id"]
        details.append({"case_id": key, "cohort": case["cohort"], "profile": case["profile"],
            "passive_origin": refs[PASSIVE][key]["run_origin"],
            "full_entered_continuation": has_continuation(rows[FULL][key]),
            "full_to_passive_equal": semantic_equality(rows[FULL][key], rows[PASSIVE][key]),
            "arms": {arm: {"final": rows[arm][key]["final"],
                "generalization": rows[arm][key]["generalization"],
                "objective_head_adoptions": len(rows[arm][key].get("objective_scope_revisions", [])),
                "objective_research_compute": (rows[arm][key].get("objective_research") or {}).get("compute_cost", 0)}
                for arm in (BASELINE, FULL, PASSIVE)}})
    actual = [rows[PASSIVE][key] for key in spec["new_cases"]]
    online = sum(row["costs"]["optimizer_new_simulator_calls"] for row in actual)
    if len(actual) != 52 or online != 1664:
        raise ValueError("the terminal new-run budget differs from the authorized 52/1664")
    equality = {field: sum(row["full_to_passive_equal"][field] for row in details)
                for field in details[0]["full_to_passive_equal"]}
    brief = {"protocol": str((output / "protocol.json").resolve()), "cases": len(details),
        "actual_new_optimizer_runs": len(actual), "actual_new_online_calls": online,
        "actual_new_initial_calls": 0, "actual_new_evaluation_calls": 0, "actual_new_physical_calls": online,
        "existing_passive_optimizer_runs": 16, "existing_passive_online_calls": 512,
        "total_actual_passive_optimizer_runs": 68, "total_actual_passive_online_calls": 2176,
        "reused_full_paths": 368, "existing_passive_controls_without_continuation": 13,
        "comparisons": comparisons, "metrics_by_cohort": by_group,
        "full_to_passive_equal_counts": equality,
        "semantic_comparison": "Compare recorded decisions, outcomes and prediction values exactly; "
            "exclude only value_precision_seconds in actions and posterior_seconds in objective heads.",
        "full_to_passive_equal_counts_by_origin": {origin: {
            field: sum(row["full_to_passive_equal"][field] for row in details if row["passive_origin"] == origin)
            for field in equality} for origin in sorted({row["passive_origin"] for row in details})},
        "full_to_passive_changed_cases": [row["case_id"] for row in details
            if not all(row["full_to_passive_equal"].values())],
        "default_decision": "prefer_passive_on_completed_suite" if all(value == 436 for value in equality.values())
            and by_group["overall"][PASSIVE]["objective_research_compute"] < by_group["overall"][FULL]["objective_research_compute"]
            else "inspect_observed_differences_before_default_change",
        "limitations": "The 52 completion cases are selected by full-path continuation entry; fresh27 remains "
            "the original v0.48 fresh cohort, not a new post-ablation confirmation sample. Reused368 "
            "paths depend on the documented unchanged-branch condition. Complete KG and component timing "
            "mix historical runs and current runs; no end-to-end speed claim follows from these sums. "
            "Objective-head evidence that spans a constraint-head rebuild retains the original v0.48 interpretation."}
    study.write(output / "case_metrics.json", {"protocol": brief["protocol"], "cases": details})
    study.write(output / "summary.json", {"protocol": spec, "result_references": [
        refs[arm][case["case_id"]] for case in spec["cases"] for arm in (BASELINE, FULL, PASSIVE)]})
    study.write(output / "result_brief.json", brief)
    compact = {key: brief[key] for key in (
        "protocol", "cases", "actual_new_optimizer_runs", "actual_new_online_calls",
        "actual_new_initial_calls", "actual_new_evaluation_calls", "actual_new_physical_calls",
        "existing_passive_optimizer_runs", "existing_passive_online_calls",
        "total_actual_passive_optimizer_runs", "total_actual_passive_online_calls",
        "reused_full_paths", "existing_passive_controls_without_continuation",
        "full_to_passive_equal_counts", "full_to_passive_equal_counts_by_origin",
        "semantic_comparison", "default_decision", "limitations")}
    compact["outcomes_by_cohort"] = {cohort: {label: {key: comparison[key]
        for key in ("feasibility_first", "prediction_loss")}
        for label, comparison in group.items()} for cohort, group in comparisons.items()}
    compact["metrics_by_cohort"] = {cohort: {arm: {key: metric[key] for key in (
        "cases", "feasible", "mean_objective_gp_mse", "mean_constraint_gp_mse",
        "objective_head_adoptions", "objective_research_compute", "active_continuation_requests")}
        for arm, metric in group.items()} for cohort, group in by_group.items()}
    study.write(output / "adoption_summary.json", compact)
    print(json.dumps({"cases": 436, "new_online_calls": online,
        "full_to_passive_equal_counts": equality, "default_decision": brief["default_decision"],
        "objective_research_compute": {arm: by_group["overall"][arm]["objective_research_compute"]
                                        for arm in (BASELINE, FULL, PASSIVE)}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run", "summarize"), default="plan")
    args = parser.parse_args()
    spec, research, rows, refs = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "cases": 436, "new_runs": 52,
            "new_online_calls": 1664, "new_initial_calls": 0, "new_evaluation_calls": 0,
            "existing_passive_runs": 16, "reused_full_paths": 368,
            "protocol": str((args.output / "protocol.json").resolve())}), flush=True)
        return
    new_cases = [case for case in spec["cases"] if case["case_id"] in spec["new_cases"]]
    if args.phase == "summarize":
        results = [study.read(args.output / "runs" / f"{case['case_id']}_{PASSIVE}.json") for case in new_cases]
    else:
        module, policy, recovery = validation.arm_modules(spec)[FULL]
        study.Program = importlib.import_module(f"{module.__package__}.language").Program
        study.SCKGEnvironment = module.SCKGEnvironment
        study.ARMS = (PASSIVE,)
        study.controlled_problem = validation.active.framework.goal.controlled_problem
        results = study.optimize(spec, args.output, new_cases, research,
            arm_modules={PASSIVE: (module, policy, recovery)})
        # prepare already required every evaluator file, so evaluate only reads paid observations.
        study.evaluate(spec, args.output, new_cases, results,
            environment_factory=validation.active.framework.goal.evaluation_environment)
    for row in results:
        key = row["case_id"]
        row.update(run_origin="actual_passive_completion", actual_new_initial_calls=0,
                   actual_new_evaluation_calls=0, frozen_method_source=spec["frozen_method_source"])
        path = args.output / "runs" / f"{key}_{PASSIVE}.json"
        study.write(path, row)
        rows[PASSIVE][key] = row
        refs[PASSIVE][key] = {"case_id": key, "arm": PASSIVE, "source_arm": PASSIVE,
            "run_origin": "actual_passive_completion", "case_record": str(path.resolve())}
    for key in spec["full_path_reuse_cases"]:
        rows[PASSIVE][key] = rows[FULL][key]
        refs[PASSIVE][key] = {**refs[FULL][key], "arm": PASSIVE,
            "run_origin": "reused_full_without_continuation", "full_reference_origin": refs[FULL][key]["run_origin"]}
    summarize(spec, args.output, rows, refs)


if __name__ == "__main__":
    main()

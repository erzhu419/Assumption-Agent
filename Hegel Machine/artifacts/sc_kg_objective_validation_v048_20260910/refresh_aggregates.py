#!/usr/bin/env python3
"""Refresh v0.48 reports from saved records; no optimization or evaluation."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_sc_kg_objective_validation_study as validation


def outcome_detail(paired):
    def group(feasible):
        selected = [row for row in paired if row["feasible"] == feasible]
        changed = [{"case_id": row["case_id"], "objective": row["objective"],
                    "absolute_gain": row["objective"][0] - row["objective"][1],
                    "relative_reduction": (row["objective"][0] - row["objective"][1])
                        / abs(row["objective"][0])}
                   for row in selected if abs(row["objective"][0] - row["objective"][1]) > 1e-9]
        return {"cases": len(selected), "improved": sum(row["absolute_gain"] > 0 for row in changed),
                "worse": sum(row["absolute_gain"] < 0 for row in changed),
                "unchanged": len(selected) - len(changed),
                "mean_absolute_gain": mean(row["objective"][0] - row["objective"][1]
                                           for row in selected) if selected else None,
                "changed_cases": changed}
    return {"common_feasible": group([True, True]), "both_infeasible": group([False, False])}


def main():
    study = validation.study
    saved = study.read(OUTPUT / "summary.json")
    prior_brief = study.read(OUTPUT / "result_brief.json")
    spec = saved["protocol"]
    old, existing = validation.source_results(Path(spec["source_study"]))
    results = []
    for ref in saved["result_references"]:
        if ref["case_record"]:
            row = study.read(ref["case_record"])
        else:
            source = existing if ref["run_origin"] == "executed_v047_development" else old
            row = dict(source[ref["case_id"]])
        row.update(arm=ref["arm"], run_origin=ref["run_origin"], original_result=ref["original_result"])
        results.append(row)
    expected = {(case["case_id"], arm) for case in spec["cases"] for arm in validation.ARMS}
    assert len(results) == len(expected) and {(row["case_id"], row["arm"]) for row in results} == expected
    # A source-resolution mismatch would change an already settled paired result.
    by_arm = {arm: {row["case_id"]: row for row in results if row["arm"] == arm}
              for arm in validation.ARMS}
    recalculated = validation.reuse.comparison(spec["cases"], *[by_arm[arm] for arm in validation.ARMS])
    assert recalculated == prior_brief["comparisons"]["overall"] or all(
        recalculated[key] == prior_brief["comparisons"]["overall"][key] for key in recalculated)
    validation.summarize(spec, OUTPUT, spec["cases"], results, "all")
    brief = study.read(OUTPUT / "result_brief.json")
    for comparison in brief["comparisons"].values():
        comparison["objective_outcome_detail"] = outcome_detail(comparison["cases"])
    timing_fields = ("controller_seconds", "forecast_seconds", "head_refit_seconds",
                     "head_posterior_seconds", "forecast_rows", "mean_prediction_rows",
                     "epistemic_prediction_rows", "noise_prediction_rows")
    for arm, mapping in by_arm.items():
        states = [row.get("objective_research") or {} for row in mapping.values()]
        requested = [request for state in states for request in state.get("requests", [])
                     if request.get("reason") == "validate_objective_scope"]
        purchased = [item for state in states for item in state.get("outcomes", [])
                     if item["request"].get("reason") == "validate_objective_scope"]
        shared = [item for state in states for item in state.get("shared_outcomes", [])
                  if item["request"].get("reason") == "validate_objective_scope"]
        stats = brief["objective_behavior"][arm]
        stats.update(active_validation_requests=len(requested),
                     active_validation_observations=len(purchased),
                     active_continuation_requests=sum("continuation_plan" in request for request in requested),
                     active_continuation_observations=sum("continuation_plan" in item["request"] for item in purchased),
                     shared_validation_observations=len(shared),
                     shared_validation_by_purpose=dict(Counter(item["request"].get("purpose") for item in shared)),
                     cases_with_objective_head_adoption=sum(bool(row.get("objective_scope_revisions"))
                                                            for row in mapping.values()),
                     objective_diagnostic_totals={field: sum(row["costs"].get("objective_diagnostics", {}).get(field, 0)
                                                              for row in mapping.values()) for field in timing_fields})
    brief["actual_new_physical_calls"] = sum(brief[key] for key in
        ("actual_new_online_calls", "actual_new_initial_calls", "actual_new_evaluation_calls"))
    brief["aggregation_refresh"] = {
        "method": "Read existing actual run JSONs and the original historical result sources only.",
        "new_optimizer_runs": 0, "new_simulator_calls": 0,
        "saved_result_references": len(saved["result_references"]),
        "run_origins": dict(Counter(row["run_origin"] for row in results)),
        "timing_note": "Objective diagnostic fields are reported individually, not added to complete KG timing.",
        "script": str(Path(__file__).resolve())}
    study.write(OUTPUT / "result_brief.json", brief)
    print(json.dumps({
        "objective_outcome_detail": brief["comparisons"]["overall"]["objective_outcome_detail"],
        "objective_behavior": brief["objective_behavior"],
        "comparator_rebuild_diagnostic": brief["comparator_rebuild_diagnostic"],
        "matched_complete_kg": brief["comparisons"]["overall"]["matched_complete_kg"],
        "actual_new_physical_calls": brief["actual_new_physical_calls"]}, indent=2))


if __name__ == "__main__":
    main()

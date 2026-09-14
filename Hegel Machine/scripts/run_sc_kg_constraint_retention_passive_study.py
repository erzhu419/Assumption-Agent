#!/usr/bin/env python3
"""Ablate terminal nomination while retaining native program/GP competition."""

import argparse
import json
from pathlib import Path

import run_sc_kg_constraint_retention_study as retention


METHOD = "constraint_retention_passive_v051"


def compare_development(spec, output, full_output):
    cases = [case for case in spec["cases"] if case["case_id"] in spec["development_cases"]]
    full, current = {}, {}
    for case in cases:
        key = case["case_id"]
        full[key] = retention.study.read(full_output / "runs" / f"{key}_constraint_retention_v051.json")
        current[key] = retention.study.read(output / "runs" / f"{key}_{METHOD}.json")
    comparison = retention.validation.reuse.comparison(cases, full, current)
    for row in comparison["cases"]:
        key = row["case_id"]
        row["semantic_equality"] = retention.passive.semantic_equality(full[key], current[key])
        row["nomination_compute"] = [retention.retention_behavior(item)["work"]["nomination_compute"]
                                      for item in (full[key], current[key])]
        row["recovery_compute"] = [item["costs"]["recovery_compute"]["compute_cost"]
                                   for item in (full[key], current[key])]
        row["native_value_precision_evaluations"] = [sum(
            action.get("value_precision_evaluations", 0) for action in item["actions"])
            for item in (full[key], current[key])]
    retention.study.write(output / "full_nomination_comparison.json", comparison)
    print(json.dumps({"full_to_passive": comparison["feasibility_first"],
        "identical_trajectories": sum(row["semantic_equality"]["paid_trajectory"] for row in comparison["cases"]),
        "identical_predictions": sum(row["semantic_equality"]["frozen_predictions"] for row in comparison["cases"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--full-development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "replay", "all"), default="plan")
    args = parser.parse_args()
    retention.METHOD = METHOD
    spec, baseline, refs = retention.prepare(args.source_study, args.output, {
        "experiment": "native_constraint_retention_without_terminal_nomination",
        "frozen_baseline_source": str((args.full_development / "baseline_source_v050").resolve()),
        "frozen_method_source": str((args.output / "method_source_passive").resolve()),
        "full_development_reference": str((args.full_development / "replay_result_brief.json").resolve()),
        "method": "Retain the accepted constraint program and adaptive affine GP as the same equal-weight "
            "native sequential mixture, updating weights on future predictive likelihoods and using native "
            "fantasy KG. The sole source change against full v051 replaces the terminal call to "
            "_constraint_retention_offer with None, so no additional constraint candidate is nominated "
            "or scored. Preserve all native and ordinary objective offers; no production feature flag.",
        "method_selection": "The full fifteen-case development run nominated23 controls, selected none, "
            "and charged8640 nomination compute. Run the same fifteen cases independently to distinguish "
            "mixture benefits from candidate-set effects. Only after this comparison may the predeclared "
            "full517-case/27-fresh protocol proceed. No fresh measurements exist at variant selection.",
        "budget_scope": "This variant's221 runs/9376 physical calls include its development15 once. "
            "The prior full-nomination development15 cost480 physical calls separately. Reusing its "
            "outcomes as this variant requires absence of every terminal nomination call, not simply "
            "absence of a purchased nominated control; this protocol instead reruns all163 affected cases."})
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "development": len(spec["development_cases"]),
            "full_runs": spec["expected_new_optimizer_runs"], "full_physical_calls": spec["expected_new_physical_calls"]}), flush=True)
    else:
        retention.run(spec, args.output, baseline, refs, args.phase)
        if args.phase == "replay":
            compare_development(spec, args.output, args.full_development)


if __name__ == "__main__":
    main()

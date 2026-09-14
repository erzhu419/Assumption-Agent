#!/usr/bin/env python3
"""Replay the default plus objective continuation and paid-label sharing fixes."""

import argparse
import json
from math import sqrt
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess

import run_sc_kg_active_noise_ablation as reuse


active, study = reuse.active, reuse.study
ARM = "objective_continuation"
PATCHES = ("01_resume_frozen_objective_validation.patch", "02_share_ordinary_paid_observations.patch")


def prepare(source, output):
    method_source = output / "method_source_v042_with_objective_continuation"
    patches = source / "objective_continuation"
    if not (output / "protocol.json").exists():
        method_source.mkdir(parents=True)
        for path in (source / "baseline_source_default_v043").glob("*.py"):
            shutil.copyfile(path, method_source / path.name)
        for name in PATCHES:
            subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p4",
                "-d", str(method_source.resolve()), "-i", str((patches / name).resolve())], check=True)

    def transform(spec):
        spec.update(experiment="objective_continuation_and_ordinary_paid_validation", arms=[ARM],
            frozen_method_source=str(method_source.resolve()),
            frozen_method_origin=str((source / "baseline_source_default_v043").resolve()),
            source_patches=[str((patches / name).resolve()) for name in PATCHES],
            ablation="Start with the executed v0.42 default. Resume an affordable frozen objective "
                "validation after constraint research becomes unresolved; share ordinary already-selected "
                "controls with that frozen objective candidate before observing their paid vectors. "
                "Retain all original four-validation and adoption criteria. No v0.46 noise posterior, "
                "noise model averaging or active noise candidate nominations.",
            reuse_accounting="Sixteen original v0.46 development cases, each with32 new online calls. "
                "Reuse the existing default comparator, paid initial32 records and independent evaluation. "
                "No fresh seeds or new initial/evaluation measurements;512 additional physical calls.")
        spec["new_method"] = {"name": "v042_plus_objective_continuation_and_paid_label_sharing",
                              "patches": list(PATCHES), "new_noise_method": False}

    return reuse.prepare(source, output, transform_spec=transform)


def objective_behavior(row):
    state = row.get("objective_research") or {}
    research = state.get("scope_research") or {}
    diagnostics = row["costs"].get("objective_diagnostics", {})
    events = state.get("events", [])
    validation = research.get("validation", [])
    shared = [item for item in state.get("shared_outcomes", [])
              if item["request"].get("reason") == "validate_objective_scope"]
    all_validation = [*shared, *[item for item in state.get("outcomes", [])
        if item["request"].get("reason") == "validate_objective_scope"]]
    completed = []
    for attempt in state.get("scope_attempts", []):
        validation_rows = attempt.get("validation", [])
        gains = [item["previous_error"] - item["candidate_error"] for item in validation_rows]
        completed.append({"scope_start": attempt["scope_start"], "phase": attempt["phase"],
            "status": attempt.get("status"), "validation_count": len(validation_rows),
            "validation_sources": [item.get("source") for item in validation_rows],
            "candidate_validation_mse": mean(item["candidate_error"] for item in validation_rows)
                if validation_rows else None,
            "previous_validation_mse": mean(item["previous_error"] for item in validation_rows)
                if validation_rows else None,
            "mean_gain": mean(gains) if gains else None,
            "twice_standard_error": 2 * stdev(gains) / sqrt(len(gains)) if len(gains) > 1 else None})
    return {"final_scope_research_phase": research.get("phase"),
        "final_validation_count": len(validation),
        "final_validation_sources": [item.get("source") for item in validation],
        "objective_scope_revisions": row.get("objective_scope_revisions", []),
        "completed_scope_attempts": completed,
        "objective_research_events": events,
        "validation_outcomes": [{"context": item["context"], "target": item["objective"],
            "features": item["request"]["features"],
            "candidate_mean_before_measurement": item["request"].get("candidate_mean"),
            "previous_mean_before_measurement": item["request"].get("previous_mean")}
            for item in all_validation],
        "shared_validation_rows": len(shared),
        "shared_ordinary_validation_rows": sum(item["request"]["purpose"] == "optimization" for item in shared),
        "objective_head_refits": diagnostics.get("head_refits", 0),
        "objective_requests": [{key: request.get(key) for key in
            ("stage", "purpose", "reason", "point")} for request in diagnostics.get("requests", [])],
        "constraint_final_phase": (row.get("recovery") or {}).get("phase"),
        "final_recommendation": row["final"]["x_recommended"],
        "final_objective": row["final"]["true_objective"],
        "final_feasible": row["final"]["true_feasible"]}


def summarize(spec, output, results):
    originals = study.read(spec["comparison_references"])["results"]
    baseline = {row["case_id"]: row for row in originals if row["arm"] == active.ARMS[0]}
    revised = {row["case_id"]: row for row in results}
    comparison = reuse.comparison(spec["cases"], baseline, revised)
    brief = {"protocol": str((output / "protocol.json").resolve()),
        "cases": len(spec["cases"]), "actual_new_optimizer_calls": sum(
            row["costs"]["optimizer_new_simulator_calls"] for row in results),
        "actual_new_initial_calls": 0, "actual_new_evaluation_calls": 0,
        "baseline_to_objective_continuation": comparison,
        "behavior": [{"case_id": case["case_id"],
            "baseline": objective_behavior(baseline[case["case_id"]]),
            "objective_continuation": objective_behavior(revised[case["case_id"]])}
            for case in spec["cases"]],
        "interpretation": "Development replay of two objective scheduling/paid-label fixes on the "
            "default method, without v0.46 noise changes. Compare final feasibility first, then objective; "
            "validation/adoption behavior is reported separately from optimization benefit."}
    study.write(output / "summary.json", {"protocol": spec, "results": results})
    study.write(output / "result_brief.json", brief)
    print(json.dumps({"cases": len(spec["cases"]), "new_online_calls": brief["actual_new_optimizer_calls"],
        "comparison": comparison["feasibility_first"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    spec, research = prepare(args.source_study, args.output)
    if args.phase == "plan":
        print(json.dumps({"status": "prepared_not_run", "cases": len(spec["cases"]),
            "new_online_calls": 512, "new_initial_calls": 0, "new_evaluation_calls": 0,
            "protocol": str((args.output / "protocol.json").resolve())}))
        return
    module, policy, recovery = active.arm_modules(spec)[active.ARMS[1]]
    study.ARMS = (ARM,)
    study.controlled_problem = active.framework.goal.controlled_problem
    results = study.optimize(spec, args.output, spec["cases"], research,
        arm_modules={ARM: (module, policy, recovery)})
    study.evaluate(spec, args.output, spec["cases"], results,
        environment_factory=active.framework.goal.evaluation_environment)
    summarize(spec, args.output, results)


if __name__ == "__main__":
    main()

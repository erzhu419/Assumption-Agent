#!/usr/bin/env python3
"""Measured preparation revision and later reuse, with the original paid loop."""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/temporal_preparation_revision_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
SPEC = importlib.util.spec_from_file_location("preparation_study_base", ARTIFACTS / "base_temporal_method_study.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
base.ROOT, base.ARTIFACTS = ROOT, ARTIFACTS
ARMS = ("v076", "preparation_revision")
COUNTS = ("identified", "solved", "planning_compute", "calibration_compute", "assessment_compute",
          "knowledge_compute", "actual_experiments", "calibration_failures", "recalibrations",
          "method_uses", "original_method_uses", "source_method_uses", "variant_uses", "method_restrictions",
          "methods_learned", "revision_proposals", "supported_revision_proposals", "failed_revision_proposals",
          "fresh_synthesis_skipped", "terminal_rows", "terminal_measurement_cost")


def load_runtime(source):
    base.load_runtime(source)
    laboratory_class = base.TemporalMethodLab

    def laboratory(case="window", stage=0, seed=151, **kwargs):
        # The extra return visit starts with the previously changed apparatus.
        # Both arms still pay the same two independent calibration trajectories.
        if stage == 5:
            kwargs["calibration_rows"] = 0
            stage = 3
        return laboratory_class(case, stage, seed, **kwargs)

    base.TemporalMethodLab = laboratory


def task_rows(payload, path, protocol):
    rows = []
    for stream in payload["streams"]:
        records = {record["method_id"]: record for record in stream["knowledge"]["program_experiment_methods"]}
        for task in stream["tasks"]:
            experiments = [a for a in task["actions"] if a["action"] == "paid_temporal_experiment"]
            calibrations = [a for a in task["actions"] if a["action"] == "paid_instrument_calibration"]
            feedback = [a["method_feedback"] for a in experiments if a.get("method_feedback")]
            used = [a for a in feedback if a["action"] == "use_program_experiment_method"]
            revised = [a for a in experiments if a["plan"].get("proposal", {}).get("parent_method_id")]
            supported = [a for a in revised if (a.get("method_feedback") or {}).get("action") ==
                         "learn_program_experiment_method"]
            row = {"seed": payload["seed"], "case": stream["case"], "stage": task["stage"],
                "stage_name": protocol["stages"][task["stage"]], "source": str(path),
                "runtime_source": payload["runtime_source"], "costs": task["costs"],
                "identified": task["identified"], "solved": task["terminal"]["loss"] <= protocol["terminal"]["solved_threshold"],
                "terminal_loss": task["terminal"]["loss"], "terminal_rows": task["terminal"]["rows"],
                "terminal_measurement_cost": task["terminal"]["measurement_cost"],
                "planning_compute": sum(a["plan"]["compute_cost"] for a in task["actions"]
                    if a["action"] in {"paid_temporal_experiment", "unresolved_temporal_plan"}),
                "calibration_compute": sum(a["compute_cost"] for a in calibrations),
                "assessment_compute": sum(a["assessment"]["compute_cost"] for a in experiments),
                "knowledge_compute": sum(a["compute_cost"] for a in task["actions"] if a["action"] == "learn_supported_hypothesis"),
                "actual_experiments": len(experiments),
                "calibration_failures": sum(a["assessment"]["calibration_status"] == "counterexample" for a in experiments),
                "recalibrations": max(0, len(calibrations) - 1), "method_uses": len(used),
                "original_method_uses": sum(not records[a["method_id"]].get("parent_method_id") for a in used),
                "source_method_uses": sum(records[a["method_id"]]["source_task_id"].endswith(":0") for a in used),
                "variant_uses": sum(bool(records[a["method_id"]].get("parent_method_id")) for a in used),
                "method_restrictions": sum(a["action"] == "revise_program_experiment_method" for a in feedback),
                "methods_learned": sum(a["action"] == "learn_program_experiment_method" for a in feedback),
                "revision_proposals": len(revised), "supported_revision_proposals": len(supported),
                "failed_revision_proposals": sum((a.get("method_feedback") or {}).get("action") ==
                    "revise_program_experiment_method" for a in revised),
                "fresh_synthesis_skipped": sum(a["plan"].get("fresh_synthesis_skipped", False) for a in experiments),
                "used_method_ids": [a["method_id"] for a in used],
                "revision_parent_ids": [a["plan"]["proposal"]["parent_method_id"] for a in revised]}
            rows.append(row)
    return rows


def aggregate(rows):
    return {"tasks": len(rows), "max_loss": max((r["terminal_loss"] for r in rows), default=None),
            "costs": {key: sum(r["costs"][key] for r in rows)
                      for key in ("observation_cost", "measurement_cost", "compute_cost", "queries")},
            **{key: sum(row[key] for row in rows) for key in COUNTS}}


def run_path(output, cohort, arm, seed, protocol):
    if cohort == "development" and arm == "v076":
        return ROOT / protocol["baseline_development_result"]
    local = output / "runs" / f"{seed}_{arm}.json"
    if cohort == "development" and not local.exists():
        return ARTIFACTS / "development_v1/runs" / f"{seed}_{arm}.json"
    return local


def summarize(output, protocol):
    summary = {"cohorts": {}, "missing_runs": [], "baseline_development_note": protocol["baseline_development_note"]}
    for cohort in ("development", "formal"):
        summary["cohorts"][cohort] = {}
        for arm in ARMS:
            rows, methods = [], []
            for seed in protocol[f"{cohort}_seeds"]:
                path = run_path(output, cohort, arm, seed, protocol)
                if not path.exists():
                    summary["missing_runs"].append({"cohort": cohort, "arm": arm, "seed": seed})
                    continue
                payload = json.loads(path.read_text())
                rows.extend(task_rows(payload, path, protocol))
                for stream in payload["streams"]:
                    methods.extend({"seed": seed, "case": stream["case"],
                        **{key: record.get(key) for key in ("method_id", "parent_method_id", "source_task_id",
                                                          "status", "preparation_revision", "failure_scopes")},
                        "support_count": len(record["support"]), "counterexample_count": len(record["counterexamples"])}
                        for record in stream["knowledge"]["program_experiment_methods"])
            summary["cohorts"][cohort][arm] = {"all": aggregate(rows),
                "cases": {case: aggregate([r for r in rows if r["case"] == case]) for case in protocol["cases"]},
                "seeds": {str(seed): aggregate([r for r in rows if r["seed"] == seed]) for seed in protocol[f"{cohort}_seeds"]},
                "stages": {name: aggregate([r for r in rows if r["stage_name"] == name]) for name in protocol["stages"]},
                "tasks": rows, "methods": methods}
    base.write(output / "summary.json", summary)
    print(json.dumps({cohort: {arm: data["all"] for arm, data in arms.items()}
                      for cohort, arms in summary["cohorts"].items()}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison_v1")
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--arm", choices=ARMS, default="preparation_revision")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.summary_only:
        summarize(args.output, protocol)
        return
    if args.seed not in protocol["development_seeds"] + protocol["formal_seeds"]:
        parser.error("seed is not declared in the protocol")
    if args.arm == "v076" and args.seed in protocol["development_seeds"]:
        parser.error("the completed 0.76 development result is reused")
    if (args.output / "runs" / f"{args.seed}_{args.arm}.json").exists():
        parser.error("completed result already exists")
    source = args.source or ROOT / protocol["runtime_sources"][args.arm]
    if not source.is_dir():
        parser.error("freeze runtime source before running")
    if args.seed in protocol["development_seeds"]:
        protocol = {**protocol, **{key: protocol[key][:5] for key in ("stages", "centers", "true_gains")}}
    load_runtime(source)
    base.run(args.output, protocol, args.arm, args.seed, source)


if __name__ == "__main__":
    main()

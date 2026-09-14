#!/usr/bin/env python3
"""Compare shared condition research with completed 0.72 runs and fresh seeds."""

import argparse
import json
from pathlib import Path

import run_transfer_condition_research_study as transfer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/shared_condition_research_20260913"
PROTOCOL = ARTIFACTS / "protocol.json"
PREVIOUS = ROOT / "artifacts/transfer_condition_research_20260913"
ARMS = ("v072", "shared_conditions")
COST_KEYS = ("observation_cost", "measurement_cost", "compute_cost")
SHARING_KEYS = ("condition_questions", "callable_hypotheses", "new_prediction_evaluations",
                "reused_prediction_evaluations", "new_phase_evaluations", "reused_phase_evaluations")


def task_metrics(task):
    report = task["research"]
    actions = report["actions"]
    plans = [a for a in actions if a["action"] == "plan_transfer_condition_experiment"]
    experiments = [a for a in actions if a["action"] == "test_transfer_condition_predictions"]
    accepted = any(a["action"] == "validate_frozen_program" and a.get("short_trial")
                   and a.get("accepted") for a in actions)
    return {"solved": task["evaluation"]["loss"] <= 1e-6,
        "loss": task["evaluation"]["loss"],
        "costs": {key: report["costs"][key] for key in COST_KEYS},
        "queries": task["actual_queries"], "short_transfer_accepted": accepted,
        "false_short_transfer": accepted and task["evaluation"]["loss"] > 1e-6,
        "condition_experiments": len(experiments),
        "methods_learned": sum(a["action"] == "learn_program_experiment_method" for a in actions),
        "methods_used": sum(a["action"] == "use_program_experiment_method" for a in actions),
        "condition_planning_compute": sum(a.get("compute_cost", 0) for a in plans),
        "condition_assessment_compute": sum(a["evidence"].get("compute_cost", 0) for a in experiments),
        "refit_compute": sum(a.get("compute_cost", 0) for a in actions
                             if a["action"] == "refit_transfer_after_experiment"),
        "complete_hypothesis_evaluations": sum(len(a["plan"].get("hypotheses", [])) for a in experiments),
        "condition_references": sum(len(c.get("conditions", [])) for a in plans
                                    for c in a.get("transfer_scope", {}).get("calls", [])),
        "history_condition_references": sum(any(p != "active" for p in condition.get("required_phases", []))
            for a in plans for call in a.get("transfer_scope", {}).get("calls", [])
            for condition in call.get("conditions", [])),
        "sharing_instrumented_plans": sum("sharing" in a.get("transfer_scope", {}) for a in plans),
        "sharing": {key: sum(a.get("transfer_scope", {}).get("sharing", {}).get(key, 0) for a in plans)
                    for key in SHARING_KEYS},
        "planning": [{key: value for key, value in a.items()
                      if key not in ("action", "proposal", "transfer_scope")}
                     for a in plans]}


def aggregate(tasks):
    return {"tasks": len(tasks), "solved": sum(t["solved"] for t in tasks),
        "max_loss": max((t["loss"] for t in tasks), default=None),
        "costs": {key: sum(t["costs"][key] for t in tasks) for key in COST_KEYS},
        "sharing": {key: sum(t["sharing"][key] for t in tasks) for key in SHARING_KEYS},
        **{key: sum(t[key] for t in tasks) for key in (
            "queries", "short_transfer_accepted", "false_short_transfer", "condition_experiments", "sharing_instrumented_plans",
            "methods_learned", "methods_used",
            "condition_planning_compute", "condition_assessment_compute", "refit_compute",
            "complete_hypothesis_evaluations", "condition_references", "history_condition_references")}}


def report_path(output, arm, seed, cohort):
    if arm == "v072" and cohort in ("development", "replication"):
        prior_output = "development_v2" if cohort == "development" else "comparison_v1"
        return PREVIOUS / prior_output / "runs" / f"{seed}_condition_research.json"
    return output / "runs" / f"{seed}_{arm}.json"


def summarize(output, protocol):
    summaries, missing = {}, []
    for cohort, seeds in protocol["cohorts"].items():
        summaries[cohort] = {}
        for arm in ARMS:
            rows = []
            for seed in seeds:
                path = report_path(output, arm, seed, cohort)
                if cohort == "development" and arm == "shared_conditions" and not path.exists():
                    path = ARTIFACTS / "development_v1/runs" / f"{seed}_{arm}.json"
                if not path.exists():
                    missing.append({"cohort": cohort, "arm": arm, "seed": seed})
                    continue
                report = json.loads(path.read_text())
                rows.extend({"seed": seed, "case": case["case"], "stage": task["stage"],
                    "source": str(path.resolve().relative_to(ROOT)), **task_metrics(task)}
                    for case in report["cases"] for task in case["tasks"])
            summaries[cohort][arm] = {"all": aggregate(rows),
                "cases": {case: aggregate([t for t in rows if t["case"] == case])
                          for case in protocol["cases"]},
                "stages": {case: {name: aggregate([t for t in rows if t["case"] == case and t["stage"] == stage])
                                  for stage, name in enumerate(protocol["stages"])}
                           for case in protocol["cases"]}, "tasks": rows}
    result = {"cohorts": summaries, "missing_runs": missing}
    transfer.write(output / "summary.json", result)
    print(json.dumps({cohort: {arm: data["all"] for arm, data in arms.items()}
                      for cohort, arms in summaries.items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "comparison_v1")
    parser.add_argument("--config", type=Path, default=PROTOCOL)
    parser.add_argument("--arm", choices=ARMS, default="shared_conditions")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    if args.summary_only:
        summarize(args.output, protocol)
        return
    cohort = next((name for name, seeds in protocol["cohorts"].items() if args.seed in seeds), None)
    if cohort is None:
        parser.error("seed must be declared in the frozen protocol")
    destination = report_path(args.output, args.arm, args.seed, cohort)
    if destination.exists():
        parser.error(f"completed run already exists: {destination}")
    if args.arm == "v072" and cohort != "generalization":
        parser.error("reuse the completed 0.72 development and replication runs")
    source = args.source or ARTIFACTS / ("v072_runtime" if args.arm == "v072" else "v073_runtime") / "src"
    if not source.is_dir():
        parser.error(f"freeze the runtime source first: {source}")
    transfer.load_runtime(source)
    transfer.run(args.output, protocol, args.arm, args.seed)


if __name__ == "__main__":
    main()

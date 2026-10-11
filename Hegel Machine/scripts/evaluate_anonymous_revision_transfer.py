#!/usr/bin/env python3
"""Evaluate frozen authored predictors on new apparatuses without model calls.

This tests predictor structure with either source fitting data or a common paid
target refit. It does not run or attribute autonomous research-method transfer.
Run only after every registered source arm has written its final training file.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
from statistics import mean
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.anonymous_history_lab import AnonymousHistoryLab
from hegel_machine.cognition.code_agent import (
    _copy_model_context, evaluate_python_model, python_model_context,
)
from scripts.run_online_inquiry_llm import write_json


CONDITIONS = ("source_context_frozen", "local_refit_same_code")


def protocol(source_dir, source_plan):
    return {
        "study": "anonymous_revision_new_apparatus_transfer",
        "source_dir": str(source_dir), "source_protocol": str(source_dir / "protocol.json"),
        "sources": [
            {"case": case["id"], "kind": case["kind"], "arm": arm,
             "source_seed": case["seed"], "target_seed": case["seed"] + 700001,
             "prefix_seed": case["seed"] + 800011,
             "evaluation_seed": case["seed"] + 900001,
             "training": str(source_dir / case["id"] / arm / "training.json")}
            for case in source_plan["cases"] for arm in source_plan["arms"]
        ],
        "conditions": list(CONDITIONS), "target_prefix_rows": 72,
        "evaluation": {"rows": 512, "trajectory_length": 48, "mse_threshold": 1e-10},
        "execution_timeout_seconds": source_plan["limits"]["execution_seconds"],
        "model_calls": 0,
        "ordering": "Write this protocol before reading any source training outcome. "
                    "All registered source training files must exist before target evaluation.",
        "eligibility": "Require an explicitly submitted, freshly confirmed source model. "
                       "Keep every registered source failure or absence as unavailable in both conditions.",
        "frozen": "Copy the submitted model's original fitting workspace and dependencies. "
                  "Keep source code and source fitting data. The existing predictor executor "
                  "calls its original fit function, if present, on that same source fitting view.",
        "refit": "Copy the same original model context independently, replace the paid fitting "
                 "view with 72 target rows, and execute its existing fit function. No fit function "
                 "means unavailable, with the attempted target acquisition retained in costs. "
                 "No author, fitting algorithm, candidate choice or code revision is supplied.",
        "cost_scope": "Source learning costs appear once per registered source. Target training "
                      "and terminal observations are separate. Charged computation quotes terminal "
                      "candidate-row scoring and target refit training rows before execution, retaining "
                      "the quote on execution failure; an explicit no-fit result releases that quote. "
                      "Completed scored_rows and target_fit_rows are reported separately. Arbitrary Python fit "
                      "internals, source refitting and context copying are represented by measured "
                      "time, not a claimed FLOP total. Failed executions retain their elapsed time.",
        "scope": "Finite-control deterministic development transfer to new physical parameters. "
                 "This isolates frozen predictor structure with or without local refitting. It "
                 "does not establish autonomous method reuse, learned meta-priors or an advantage "
                 "over an ordinary Code agent given the same executable archive.",
    }


def _empty_costs():
    return {"observation_cost": 0, "measurement_cost": 0, "queries": 0,
            "terminal_observations": 0, "compute_cost": 0,
            "target_fit_rows": 0, "scored_rows": 0,
            "context_copy_seconds": 0.0, "acquisition_seconds": 0.0,
            "python_execution_seconds": 0.0, "elapsed_seconds": 0.0,
            "model_calls": 0, "reported_tokens": 0}


def _unavailable(condition, reason):
    return {"condition": condition, "status": "unavailable", "reason": reason,
            "passed": False, "mse": None, "costs": _empty_costs()}


def evaluate_condition(source, registration, condition, output, plan):
    """Keep all model execution and modified fitting views inside a new clone."""
    started, costs = monotonic(), _empty_costs()
    outcome = {"condition": condition, "status": "execution_failed", "passed": False,
               "mse": None, "costs": costs}
    try:
        original_workspace, original_model = python_model_context(source["workspace"], source["model"])
        original_code = (original_workspace / original_model).read_bytes()
        fitting_file = original_workspace / "model_observations.json"
        if not fitting_file.exists():
            fitting_file = original_workspace / "observations.json"
        outcome["source_fitting_rows"] = len(json.loads(fitting_file.read_text()))
        copy_started = monotonic()
        inner_model = _copy_model_context(source["workspace"], source["model"], output)
        costs["context_copy_seconds"] = monotonic() - copy_started
        outcome.update(workspace=str(output), model=inner_model)
        lab = AnonymousHistoryLab(kind=registration["kind"], seed=registration["target_seed"])
        refit = condition == "local_refit_same_code"
        if refit:
            acquisition_started = monotonic()
            records, training_rows = [], []
            for action in lab.confirmation_actions(plan["target_prefix_rows"], Random(registration["prefix_seed"])):
                record = lab.execute(action)
                records.append(record)
                training_rows.extend(lab.observation_rows(record))
            costs["acquisition_seconds"] = monotonic() - acquisition_started
            costs.update(lab.costs)
            # inner_model is relative to the resolved source fitting workspace,
            # even when the submitted model came from an _evaluated_model copy.
            payload = [asdict(row) for row in training_rows]
            write_json(output / "model_observations.json", payload)
            write_json(output / "observations.json", payload)
            write_json(output / "laboratory_observations.json", records)
        rows = lab.evaluation_rows(length=plan["evaluation"]["rows"],
                                   seed=registration["evaluation_seed"])
        costs["terminal_observations"] = len(rows)
        costs["compute_cost"] = len(rows) + (plan["target_prefix_rows"] if refit else 0)
        execution_started = monotonic()
        try:
            result = evaluate_python_model(output, inner_model, rows,
                timeout_seconds=plan["execution_timeout_seconds"], require_fit=refit)
        finally:
            costs["python_execution_seconds"] = monotonic() - execution_started
        result.pop("predictions", None)
        outcome["evaluation"] = result
        outcome["source_code_unchanged"] = (output / inner_model).read_bytes() == original_code
        if refit and result.get("fit_available") is False:
            costs["compute_cost"] = 0
            outcome.update(status="unavailable", reason="source_predictor_has_no_fit")
        elif result.get("returncode") == 0 and result.get("mse") is not None:
            costs["scored_rows"] = len(rows)
            costs["target_fit_rows"] = plan["target_prefix_rows"] if refit else 0
            passed = (outcome["source_code_unchanged"] and
                      result["mse"] <= plan["evaluation"]["mse_threshold"])
            outcome.update(status="evaluated", mse=result["mse"], passed=passed)
            if not outcome["source_code_unchanged"]:
                outcome.update(status="execution_failed", reason="source_code_changed_during_execution")
        else:
            outcome["reason"] = "predictor_execution_failed"
    except (OSError, ValueError, KeyError, TypeError) as error:
        outcome["reason"] = f"{type(error).__name__}: {error}"
    finally:
        costs["elapsed_seconds"] = monotonic() - started
    return outcome


def run_transfer(source_dir, output):
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    source_plan = json.loads((source_dir / "protocol.json").read_text())
    plan = protocol(source_dir, source_plan)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    missing = [item["training"] for item in plan["sources"] if not Path(item["training"]).is_file()]
    if missing:
        write_json(output / "summary.json", {"status": "source_training_incomplete", "missing": missing})
        raise ValueError("every registered source arm must finish training before transfer evaluation")

    outcomes = []
    for registration in plan["sources"]:
        source = json.loads(Path(registration["training"]).read_text())
        item = {**registration, "source_status": source["status"],
                "source_model": source.get("model"), "source_learning_costs": source.get("costs", {}),
                "source_fresh_confirmation_passed": bool((source.get("fresh_confirmation") or {}).get("passed")),
                "conditions": []}
        for condition in CONDITIONS:
            if not source.get("model"):
                result = _unavailable(condition, "no_submitted_source_model")
            elif not item["source_fresh_confirmation_passed"]:
                result = _unavailable(condition, "source_model_not_freshly_confirmed")
            else:
                result = evaluate_condition(source, registration, condition,
                    output / registration["case"] / registration["arm"] / condition, plan)
            item["conditions"].append(result)
        outcomes.append(item)
        write_json(output / "outcomes.json", outcomes)

    groups = []
    for arm in source_plan["arms"]:
        sources = [item for item in outcomes if item["arm"] == arm]
        for condition in CONDITIONS:
            rows = [result for item in sources for result in item["conditions"]
                    if result["condition"] == condition]
            losses = [row["mse"] for row in rows if row["mse"] is not None]
            groups.append({"arm": arm, "condition": condition, "registered_sources": len(rows),
                "evaluated": sum(row["status"] == "evaluated" for row in rows),
                "unavailable": sum(row["status"] == "unavailable" for row in rows),
                "execution_failed": sum(row["status"] == "execution_failed" for row in rows),
                "passed": sum(row["passed"] for row in rows),
                "mean_mse_among_evaluated": mean(losses) if losses else None,
                "costs": {key: sum(row["costs"][key] for row in rows) for key in _empty_costs()}})
    source_cost_keys = sorted({key for item in outcomes for key in item["source_learning_costs"]})
    source_costs = {}
    for key in source_cost_keys:
        values = [item["source_learning_costs"].get(key, 0) for item in outcomes]
        source_costs[key] = None if any(value is None for value in values) else sum(values)
    summary = {"study": plan["study"], "status": "completed", "registered_sources": len(outcomes),
               "source_learning_costs_once": source_costs,
               "groups": groups, "model_calls": 0, "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_transfer(args.source_dir, args.output), indent=2))


if __name__ == "__main__":
    main()

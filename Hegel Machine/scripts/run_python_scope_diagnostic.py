#!/usr/bin/env python3
"""Researcher-directed paid comparison of current-record refits after fresh rejection."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.temporal_operator_agent import TemporalOperatorAgent
from scripts.run_candidate_retention_cycle_llm import prepare_episode
from scripts.run_python_method_learning_study import laboratory, total_costs, write_json

PREVIOUS = ROOT / "artifacts/python_scoped_method_completion_luna_20261006"


class RecordedDiagnosticAgent(TemporalOperatorAgent):
    """Keep the actual operator feedback, including the preparation prefix."""

    def __init__(self, *args, **kwargs):
        self.diagnostic_trace = []
        super().__init__(*args, **kwargs)

    def _execute(self, action, arguments, laboratory):
        before = deepcopy(self.costs)
        feedback = super()._execute(action, arguments, laboratory)
        self.diagnostic_trace.append({"action": action, "arguments": deepcopy(arguments),
            "feedback": deepcopy(feedback), "costs_before": before,
            "costs_after": deepcopy(self.costs)})
        return feedback


def paid_summary(label, agent, status):
    """Expose paid evidence without hidden task settings, terminal data or policy source."""
    steps = []
    for index, event in enumerate(agent.diagnostic_trace):
        args, feedback = event["arguments"], event["feedback"]
        arguments = {key: deepcopy(args[key]) for key in (
            "operator", "model", "path", "records", "validation_records") if key in args}
        if "actions" in args:
            arguments["trajectories"] = [{"rows": len(action["sequence"]),
                "channels": action.get("channels", [])} for action in args["actions"]]
        shown = {key: deepcopy(feedback[key]) for key in (
            "operator", "status", "error", "path", "source_model", "split", "mse",
            "record_indices", "final_fitting_rows", "paid_rows_at_refit") if key in feedback}
        validation = feedback.get("validation")
        if isinstance(validation, dict):
            shown["validation"] = {key: deepcopy(validation[key]) for key in (
                "returncode", "fit_available", "mse", "stderr", "timeout") if key in validation}
        fresh = feedback.get("fresh_confirmation")
        if isinstance(fresh, dict):
            shown["fresh_confirmation"] = {key: deepcopy(fresh[key]) for key in (
                "passed", "mse", "rows", "mse_threshold") if key in fresh}
            indices = fresh.get("observation_indices", [])
            if indices:
                shown["fresh_confirmation"]["row_interval"] = [min(indices), max(indices)]
        steps.append({"step": index, "action": event["action"], "arguments": arguments,
            "feedback": shown, "costs_after": event["costs_after"]})
    return {"practice_case": label, "status": status,
        "paid_records": [{"record": index,
            "row_interval": [min(record["row_indices"]), max(record["row_indices"])],
            "rows": len(record["row_indices"]), "meter_rows": len(record.get("measurements", []))}
            for index, record in enumerate(agent.observations)],
        "steps": steps, "costs": deepcopy(agent.costs)}


def run_diagnostic(output, *, previous=PREVIOUS):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    plan = json.loads((previous / "protocol.json").read_text())
    sources = json.loads((previous / "existing_source_outcomes.json").read_text())["sources"]
    source = sources[plan["author_source_index"]]
    labels = ("case_3", "case_4")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", {"study": "python_scope_comparison",
        "controller": "researcher_directed_diagnostic", "previous": str(previous),
        "source": source, "cases": {label: plan["practice_cases"][label] for label in labels},
        "prefix": plan["prefix"], "limits": plan["limits"],
        "additional_trajectory_seed_offset": 104729,
        "terminal_evaluation": False})
    cases = []
    for label in labels:
        spec = plan["practice_cases"][label]
        payload = prepare_episode(spec["case"], 1, output / label,
            [deepcopy(source["candidate"])], plan, laboratory=laboratory(spec),
            agent_class=RecordedDiagnosticAgent)
        agent, lab = payload["agent"], payload["laboratory"]
        agent.confirmation_rng.seed(spec["confirmation_seed"])
        prior = source["candidate"]["model"]
        first = agent._execute("finish", {"model": prior}, lab)
        if first.get("fresh_confirmation", {}).get("passed") is not False:
            raise ValueError("the registered initial predictor did not produce the expected paid rejection")
        general = agent._execute("run", {"operator": "refit", "model": prior,
            "path": "diagnostic_refit/general"}, lab)
        rejected = agent._execute("finish", {"model": general["path"]}, lab)
        if rejected.get("fresh_confirmation", {}).get("passed") is not False:
            raise ValueError("the registered post-validation trajectory did not reject the refit")
        rng = Random(spec["confirmation_seed"] + 104729)
        agent._execute("query", {"actions": [{"kind": "trajectory", "channels": [],
            "sequence": [{"u": rng.uniform(-1., 1.)} for _ in range(24)]}]}, lab)
        scoped = agent._execute("run", {"operator": "refit", "model": prior,
            "path": "diagnostic_refit/current_records", "records": [4, 5],
            "validation_records": [5]}, lab)
        status = scoped["status"]
        if status == "refit_supported_on_validation":
            fresh = agent._execute("finish", {"model": scoped["path"]}, lab)
            status = "fresh_confirmed" if fresh.get("fresh_confirmation", {}).get("passed") else "fresh_rejected"
        training = agent.result(status)
        training["controller"] = "researcher_directed_diagnostic"
        training["controller_trace"] = deepcopy(agent.diagnostic_trace)
        write_json(payload["path"] / "training.json", training)
        write_json(payload["path"] / "paid_observations.json", agent.observations)
        cases.append(paid_summary(label, agent, status))
    compact = {"study": "paid_scope_comparison", "controller": "researcher_directed_diagnostic",
        "evidence_scope": "Two registered anonymous practice cases, using the same archived predictor "
            "and the same prescribed sequence of paid operations. The displayed losses come from "
            "actual separate-trajectory validation and fresh confirmation. No terminal evaluation "
            "or learned-program selection was performed.",
        "cases": cases, "costs": total_costs([case["costs"] for case in cases])}
    write_json(output / "scope_comparison.json", compact)
    write_json(output / "summary.json", {"study": "python_scope_comparison",
        "controller": compact["controller"], "new_model_calls": 0, "new_reported_tokens": 0,
        "costs": compact["costs"], "author_evidence_file": "scope_comparison.json",
        "cases": [{"practice_case": case["practice_case"], "status": case["status"],
            "costs": case["costs"], "trace": f"{case['practice_case']}/training.json"} for case in cases]})
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    args = parser.parse_args()
    result = run_diagnostic(args.output, previous=args.previous)
    print(json.dumps({"costs": result["costs"], "cases": [{"practice_case": case["practice_case"],
        "status": case["status"], "costs": case["costs"]} for case in result["cases"]]}, indent=2))

#!/usr/bin/env python3
"""Matched model-selected adoption of the same supplied response procedure."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_method_transfer import install_method_bundle
from hegel_machine.cognition import world_reference_recalibration as recipe
from hegel_machine.cognition import world_response_hypotheses, world_response_consequences
from scripts import consolidate_response_measurement as study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json

SOURCE = ROOT / "artifacts/response_completed_target_routing_20261010"
OUTPUT = ROOT / "artifacts/response_matched_adoption_20261010"
CASES = ("cubic_original_mid_high", "cubic_original_mid_low", "aliased_shifted_mid")
ARMS = ("hegel", "code_archive")
MODEL = {"requested_alias": "gpt-6-luna", "reasoning_effort": "medium"}


def point_confirmations(result):
    """Keep recorded verdicts separate from their fresh, installed point evidence."""
    state = result.get("reference_state", {})
    trials = {item["question"]: item for item in state.get("tests", [])}
    if state.get("test"):
        trials[state["test"]["question"]] = state["test"]
    questions = {item["id"]: item for item in result.get("research", {}).get("questions", [])}
    observations = {item["index"]: item for item in result.get("observations", [])}
    rows = []
    for qid, trial in trials.items():
        if trial["number"] not in (4, 5) or "restore_frequency" in trial:
            continue
        question = questions.get(qid, {})
        answer = question.get("answer", {})
        answer = answer if isinstance(answer, dict) else {}
        tolerance = 1e-6 if trial.get("authored_experiment") else .51
        matches = []
        for index in question.get("evidence", []):
            record = observations.get(index)
            if record is None or index < trial["started"]:
                continue
            view = recipe._view(record)
            number, frequency = recipe._dialog(view)
            objects = recipe._objects(view)
            if (number == trial["number"] and abs(frequency - trial["prediction"]) <= tolerance
                    and recipe._installed(objects, trial["crystal"], trial["reactor"])
                    and recipe._activated(objects, trial["reactor"])):
                matches.append({"index": index, "frequency": frequency})
        system_supported = question.get("status") == "supported"
        actual = answer.get("actual_frequency")
        confirmed = (system_supported and answer.get("activated") is True
            and isinstance(actual, (int, float))
            and any(abs(actual - item["frequency"]) <= 1e-6 for item in matches))
        rows.append({"question_id": qid, "number": trial["number"],
            "family": trial.get("family"), "prediction": trial["prediction"],
            "question_status": question.get("status"), "trial_status": trial.get("status"),
            "system_supported": system_supported, "physically_confirmed": bool(confirmed),
            "actual_positive_evidence": matches, "controller_tolerance": tolerance})
    return rows


def supplied_method_calls(result, bundle):
    """Actual model events only; nested model turns already occur in events once."""
    methods = {item["id"]: item["path"] for item in bundle["methods"]}
    paths = set(methods.values())
    calls = []
    for index, event in enumerate(result.get("events", [])):
        if "usage" not in event:
            continue
        args, feedback = event.get("arguments", {}), event.get("feedback", {})
        path = methods.get(args.get("method_id"), args.get("path"))
        if path not in paths or not (event.get("action") == "query" or
                event.get("action") == "run" and args.get("operator") == "procedure"):
            continue
        executed = (bool(feedback.get("steps")) if args.get("operator") == "procedure"
                    else feedback.get("executed_path") == path)
        calls.append({"event_index": index, "model_call": event.get("model_call"),
            "action": event["action"], "path": path, "method_id": args.get("method_id"),
            "inquiry_question_id": event.get("inquiry_question_id"),
            "status": feedback.get("status"), "executed": executed})
    return calls


def brief(result, bundle):
    calls = supplied_method_calls(result, bundle)
    # Child turns share this flat event ledger and the same agent account.
    usages = [event["usage"] for event in result.get("events", []) if "usage" in event]
    reported = [usage["cached_input_tokens"] for usage in usages
                if isinstance(usage.get("cached_input_tokens"), int)]
    costs = result.get("costs", {})
    return {key: result[key] for key in ("case", "repetition", "arm", "status")} | {
        "success": result.get("score", {}).get("completedSuccessfully") is True,
        "score": result.get("score"), "supplied_method_calls": calls,
        "supplied_method_selected": bool(calls),
        "supplied_method_executed": any(call["executed"] for call in calls),
        "point_confirmations": point_confirmations(result),
        "source_changed": result.get("source_changed"), "costs": costs,
        "cache": {"reported_calls": len(reported),
                  "missing_calls": costs.get("model_calls", 0) - len(reported),
                  "hit_calls": sum(value > 0 for value in reported),
                  "zero_hit_calls": sum(value == 0 for value in reported)},
        **({"runner_error": result["runner_error"]} if "runner_error" in result else {})}


def run_case(case, config, repetition, arm, bundle, output, limits, *, provider_factory=None):
    directory = Path(output) / case / f"repeat_{repetition:02d}" / arm
    directory.mkdir(parents=True, exist_ok=False)
    identity = {"case": case, "repetition": repetition, "arm": arm}
    agent = None

    def checkpoint(data):
        latest = data.get("events", [])[-1] if data.get("events") else {}
        write_json(directory / "active.json", {**identity, "status": data["status"],
            "costs": data["costs"], "score": data["score"],
            "latest": {"action": latest.get("action"),
                "operator": latest.get("arguments", {}).get("operator"),
                "inquiry_question_id": latest.get("inquiry_question_id")}})

    write_json(directory / "active.json", {**identity, "status": "starting"})
    try:
        study.CASES = {case: config}
        lab = study.TransferSignalLab(case)
        workspace = directory / "agent"
        provider = (provider_factory(workspace) if provider_factory else
            CodexCodeProvider(workspace, model=MODEL["requested_alias"],
                reasoning_effort=MODEL["reasoning_effort"],
                timeout_seconds=limits["provider_request_timeout_seconds"]))
        agent = DiscoveryWorldAgent(provider, workspace, arm=arm, limits=limits)
        agent.lab = lab
        installation = install_method_bundle(agent, bundle, lab)
        # Install files and registrations only: the model makes the first choice.
        result = agent.run(lab, on_checkpoint=checkpoint)
        result["method_bundle_installation"] = installation
    except Exception as error:
        result = agent.result() if agent is not None and agent.lab is not None else {}
        result.update(status="runner_failed", runner_error={
            "type": type(error).__name__, "message": str(error)})
    result.update(identity)
    if agent is not None:
        result["observations"] = agent.observations
        method_paths = {item["path"] for item in bundle["methods"]}
        result["final_method_sources"] = {path: (agent.workspace / path).read_text()
            if (agent.workspace / path).exists() else None for path in method_paths}
        result["source_changed"] = any(source != bundle["files"][path]
            for path, source in result["final_method_sources"].items())
        state_path = agent.workspace / recipe.STATE
        if state_path.exists():
            state = json.loads(state_path.read_text())
            result["reference_state"] = {key: state.get(key) for key in ("tests", "test")
                                          if key in state}
    write_json(directory / "result.json", result)
    row = brief(result, bundle)
    write_json(directory / "brief.json", row)
    write_json(directory / "active.json", {**row, "run_finished": True})
    return row


def summarize(rows, planned_runs=18):
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        trials = [trial for row in selected for trial in row["point_confirmations"]]
        arms[arm] = {"finished_runs": len(selected), "successful_runs": sum(row["success"] for row in selected),
            "statuses": dict(Counter(row["status"] for row in selected)),
            "method_selected_runs": sum(row["supplied_method_selected"] for row in selected),
            "method_executed_runs": sum(row["supplied_method_executed"] for row in selected),
            "source_changed_runs": sum(row["source_changed"] is True for row in selected),
            "target_point_confirmations": sum(trial["physically_confirmed"] for trial in trials),
            "unsupported_positive_verdicts": sum(trial["system_supported"] and not trial["physically_confirmed"]
                                                  for trial in trials),
            "costs": {key: sum(row["costs"].get(key, 0) for row in selected) for key in (
                "model_calls", "environment_steps", "research_compute_units", "python_execution_seconds",
                "wall_time_seconds", "input_tokens", "output_tokens", "cached_input_tokens",
                "reported_tokens", "usage_missing_calls")},
            "cache": {key: sum(row["cache"][key] for row in selected)
                      for key in ("reported_calls", "missing_calls", "hit_calls", "zero_hit_calls")}}
    return {"planned_runs": planned_runs, "finished_runs": len(rows),
        "status": "finished" if len(rows) == planned_runs else "running", "arms": arms,
        "conditions": {case: {arm: {"finished_runs": len(chosen),
            "successful_runs": sum(row["success"] for row in chosen)}
            for arm in ARMS for chosen in [[row for row in rows if row["case"] == case and row["arm"] == arm]]}
            for case in CASES}}


def run(output=OUTPUT):
    prior = json.loads((SOURCE / "protocol.json").read_text())
    bundle = json.loads((SOURCE / "completed_target_routing_bundle.json").read_text())
    limits = {**prior["limits"], "max_model_calls": 24, "environment_steps": 512,
        "task_wall_seconds": 5400, "provider_request_timeout_seconds": 300,
        "python_seconds_per_call": 5, "python_seconds_per_task": 600}
    jobs = [{"case": case, "repetition": repetition, "arm": arm}
        for repetition in range(1, 4) for index, case in enumerate(CASES)
        for arm in (ARMS if (repetition + index) % 2 else tuple(reversed(ARMS)))]
    auth = verify_codex_subscription()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_bundle.json", bundle)
    for module in (recipe, world_response_hypotheses, world_response_consequences):
        (output / Path(module.__file__).name).write_text(Path(module.__file__).read_text())
    protocol = {"study": "response_matched_model_adoption", "source": str(SOURCE),
        "cases": {case: prior["cases"][case] for case in CASES}, "jobs": jobs,
        "arms": ARMS, "repetitions_per_condition": 3, "workers": 3,
        "model": MODEL, "auth_mode": auth, "limits": limits,
        "scope": "Three consumed development conditions, each with three independent model runs per arm. "
            "Both arms receive identical source files and registered methods, zero paid observations, "
            "and equal tools/budgets. The first choice and later revisions belong to the model. "
            "The runner never pre-executes the supplied procedure or injects settings, outcome-based repairs, "
            "or automatic reruns. Existing agent controllers remain available after model decisions. "
            "This is a strong executable-archive comparator, not an untooled vanilla agent or a pure "
            "recursive-organization ablation. Source changes may be authored or runtime-generated. "
            "Physical task success, method selection/execution, and requested-point confirmation are separate. "
            "The larger common live budgets are not paired with prior offline costs. Cache usage is reported, "
            "not guaranteed; missing usage is distinct from measured zero cache hits."}
    write_json(output / "protocol.json", protocol)
    rows = []
    write_json(output / "summary.json", summarize(rows))
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_case, job["case"], prior["cases"][job["case"]],
            job["repetition"], job["arm"], bundle, output, limits): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                row = future.result()
            except Exception as error:
                directory = output / job["case"] / f"repeat_{job['repetition']:02d}" / job["arm"]
                directory.mkdir(parents=True, exist_ok=True)
                result = {**job, "status": "runner_failed", "runner_error": {
                    "type": type(error).__name__, "message": str(error)}}
                write_json(directory / "runner_error.json", result)
                row = brief(result, bundle)
            rows.append(row)
            rows.sort(key=lambda row: jobs.index({key: row[key] for key in ("case", "repetition", "arm")}))
            write_json(output / "brief.json", rows)
            write_json(output / "summary.json", summarize(rows))
            print(json.dumps({key: row[key] for key in ("case", "repetition", "arm", "status", "success")}), flush=True)
    return summarize(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(run(parser.parse_args().output), indent=2))

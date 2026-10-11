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
from scripts.run_response_post_revision_transfer import include_reference_research_question

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


def scientific_trace(result):
    """Join model research decisions to paid evidence; continuation is not causal gain."""
    research = result.get("research", {})
    questions = {q["id"]: q for q in research.get("questions", [])}
    observations = {row["index"] for row in result.get("observations", [])}
    state = result.get("reference_state", {})
    trials = {t["question"]: t for t in state.get("tests", [])}
    if state.get("test"):
        trials[state["test"]["question"]] = state["test"]
    confirmed = {row["question_id"] for row in point_confirmations(result)
                 if row["physically_confirmed"]}
    decisions, inquiries = [], []
    for index, event in enumerate(result.get("events", [])):
        args, feedback = event.get("arguments", {}), event.get("feedback", {})
        operator = args.get("operator")
        if "usage" not in event or event.get("action") != "run" or not (
                operator == "reference_hypothesis" or
                operator == "reference_response" and isinstance(args.get("experiment"), dict)):
            continue
        inquiry_id = event.get("inquiry_question_id")
        fresh = sorted(observations.intersection(feedback.get("new_observation_indices", [])))
        returned = [t["question"] for t in feedback.get("tests", [])]
        if feedback.get("pending_test"):
            returned.append(feedback["pending_test"]["question"])
        if feedback.get("question_id"):
            returned.append(feedback["question_id"])
        physical = []
        for qid in dict.fromkeys(returned):
            trial, question = trials.get(qid, {}), questions.get(qid, {})
            evidence = sorted(set(question.get("evidence", [])).intersection(fresh))
            evidence = [i for i in evidence if i >= trial.get("started", float("inf"))]
            reviewed = [q["id"] for q in questions.values()
                if set(q.get("reviewed_child_answers", {}).get(qid, {}).get("evidence", []))
                .intersection(evidence)]
            physical.append({"question_id": qid, "parent_id": trial.get("parent_id", question.get("parent_id")),
                "number": trial.get("number"), "family": trial.get("family"),
                "prediction": trial.get("prediction"), "status": question.get("status"),
                "fresh_evidence": evidence, "reviewed_by": reviewed,
                "physically_confirmed": qid in confirmed and bool(evidence)})
        decisions.append({"event_index": index, "model_call": event.get("model_call"),
            "inquiry_question_id": inquiry_id, "operator": operator,
            "hypothesis_id": args.get("hypothesis", {}).get("id"),
            "declared_hypothesis": deepcopy(feedback.get("hypothesis")),
            "experiment": deepcopy(args.get("experiment")),
            "fresh_observations": fresh, "physical_trials": physical})
    methods = result.get("method_results", [])
    for qid, session in research.get("llm_inquiries", {}).items():
        child_events = [e for e in result.get("events", [])
                        if "usage" in e and e.get("inquiry_question_id") == qid]
        returns = []
        for step in methods:
            args = (step.get("proposal") or {}).get("arguments", {})
            if args.get("operator") != "inquire" or args.get("question", {}).get("id") != qid:
                continue
            following = next((later for later in methods
                if later["step"] > step["step"] and later.get("path") == step.get("path")
                and later.get("proposal") is not None), None)
            proposal = following["proposal"] if following else {}
            returns.append({"returned_at_program_step": step["step"],
                "parent_next_step": following["step"] if following else None,
                "parent_next_action": proposal.get("action"),
                "parent_next_operator": proposal.get("arguments", {}).get("operator"),
                "parent_next_experiment": deepcopy(proposal.get("arguments", {}).get("experiment")),
                "parent_next_world_actions": deepcopy(proposal.get("actions", [])),
                "parent_stop": proposal.get("stop")})
        inquiries.append({"question_id": qid,
            "origin": questions.get(qid, {}).get("origin"),
            "parent_id": questions.get(qid, {}).get("parent_id"),
            "model_events": len(child_events), "status": session.get("status"),
            "scientific_decision_indices": [i for i, row in enumerate(decisions)
                if row["inquiry_question_id"] == qid], "parent_returns": returns})
    return {"inquiries": inquiries, "decisions": decisions,
        "confirmed_decision_trials": sum(t["physically_confirmed"]
            for row in decisions for t in row["physical_trials"]),
        "scope": "Recorded model declarations, experiments, fresh evidence and parent continuation; "
            "these links alone do not establish a new law or a causal advantage over flat research."}


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
        "scientific_trace": scientific_trace(result),
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


def summarize(rows, planned_runs=18, *, cases=CASES):
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
            for case in cases}}


def prepare_bundle(bundle):
    """Complete a new source copy; never alter an archived or running package."""
    prepared = deepcopy(bundle)
    path = study.METHOD_PATH
    prepared["files"][path] = include_reference_research_question(prepared["files"][path])
    return prepared


def run(output=OUTPUT, *, cases=None, bundle=None, knowledge_scope=None):
    prior = json.loads((SOURCE / "protocol.json").read_text())
    cases = deepcopy(cases if cases is not None else {case: prior["cases"][case] for case in CASES})
    bundle = prepare_bundle(bundle if bundle is not None else
        json.loads((SOURCE / "completed_target_routing_bundle.json").read_text()))
    # The first live cohort exceeded 5 s while executing a growing research state.
    # Give both arms the same usable call window; retain the total Python budget.
    limits = {**prior["limits"], "max_model_calls": 24, "environment_steps": 512,
        "task_wall_seconds": 5400, "provider_request_timeout_seconds": 300,
        "python_seconds_per_call": 30, "python_seconds_per_task": 600}
    jobs = [{"case": case, "repetition": repetition, "arm": arm}
        for repetition in range(1, 4) for index, case in enumerate(cases)
        for arm in (ARMS if (repetition + index) % 2 else tuple(reversed(ARMS)))]
    auth = verify_codex_subscription()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_bundle.json", bundle)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    for module in (recipe, world_response_hypotheses, world_response_consequences):
        (output / Path(module.__file__).name).write_text(Path(module.__file__).read_text())
    protocol = {"study": "response_matched_model_adoption", "source": str(SOURCE),
        "bundle_preparation": "Include the existing reference_research_question definition only when missing. "
            "Retain the other source AST, supplied declarations and registered methods. "
            "The historical source package is unchanged; this output freezes the prepared copy and runner.",
        "cases": cases, "jobs": jobs, "initial_knowledge": knowledge_scope,
        "arms": ARMS, "repetitions_per_condition": 3, "workers": 3,
        "model": MODEL, "auth_mode": auth, "limits": limits,
        "scope": "Existing development conditions, each with three independent model runs per arm. "
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
    summarize_cohort = lambda: summarize(rows, len(jobs), cases=cases)
    write_json(output / "summary.json", summarize_cohort())
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_case, job["case"], cases[job["case"]],
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
            write_json(output / "summary.json", summarize_cohort())
            print(json.dumps({key: row[key] for key in ("case", "repetition", "arm", "status", "success")}), flush=True)
    return summarize_cohort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(run(parser.parse_args().output), indent=2))

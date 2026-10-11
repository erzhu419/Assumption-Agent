#!/usr/bin/env python3
"""Compare two existing research assignments at the same paid failure state."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_llm_inquiry import inquiry_request
from scripts import run_response_matched_adoption as matched
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_response_family_recovery import declarations

SOURCE = ROOT / "artifacts/response_revision_replay_20261010_fixed/changed_mechanism/repeat_01/hegel"
HISTORY = ROOT / "artifacts/response_family_recovery_20261009/changed_mechanism/agent/research_context.json"
OUTPUT = ROOT / "artifacts/response_assignment_comparison_20261010"
TREATMENTS = ("current_assignment", "historical_assignment")
QUESTION_ID = "reference_assignment_comparison"
ALLOWANCE = {"model_calls": 24, "environment_steps": 512,
             "python_execution_seconds": 600, "wall_time_seconds": 5400}
MODEL = matched.MODEL


def assignments(history=HISTORY):
    original = json.loads(Path(history).read_text())["llm_inquiry"]["question"]["question"]
    historical = original.split(" Current measured table: ", 1)[0]
    common = " Current measurements and responses are in shared/reference_recalibration.json and the paid observations."
    return {"current_assignment": matched.recipe.reference_research_question(
        5, "shared/reference_extension_5.json") + common,
        "historical_assignment": historical + common}, original


def restore(source, directory, text, *, config, base_limits):
    """Replay the complete saved state, retaining its account and scientific files."""
    source, directory = Path(source), Path(directory)
    previous = json.loads((source / "result.json").read_text())
    prior_costs = deepcopy(previous["costs"])
    limits = {**base_limits,
        "max_model_calls": prior_costs["model_calls"] + ALLOWANCE["model_calls"],
        "environment_steps": prior_costs["environment_steps"] + ALLOWANCE["environment_steps"],
        "python_seconds_per_task": prior_costs["python_execution_seconds"] + ALLOWANCE["python_execution_seconds"],
        "task_wall_seconds": prior_costs["wall_time_seconds"] + ALLOWANCE["wall_time_seconds"],
        "python_seconds_per_call": 30, "provider_request_timeout_seconds": 300}
    directory.mkdir(parents=True, exist_ok=False)
    matched.study.CASES = {"changed_mechanism": deepcopy(config)}
    lab = matched.study.TransferSignalLab("changed_mechanism")
    agent = DiscoveryWorldAgent(None, directory / "agent", arm="hegel", limits=limits)
    agent.lab = lab
    previous["episode_status"] = "running"
    replay = restore_repair(agent, lab, previous, source / "agent")
    method = agent.workspace / matched.study.METHOD_PATH
    original = method.read_text()
    prepared = matched.prepare_bundle({"files": {matched.study.METHOD_PATH: original}})["files"][matched.study.METHOD_PATH]
    method.write_text(prepared)
    (directory / "source_original.py").write_text(original)
    (directory / "source_before.py").write_text(prepared)
    question = {"id": QUESTION_ID, "parent_id": "task", "question": text,
                "depends_on": [], "parent_program_path": matched.study.METHOD_PATH}
    agent._question(question)
    agent._save_state()
    context = {"source": str(source), "prior_status": previous["status"],
        "prefix_observations": len(agent.observations), "prefix_events": len(agent.events),
        "costs_before": prior_costs, "question": question, "replay": replay,
        "declarations_before": declarations(prepared)}
    write_json(directory / "restored.json", {**context, "limits": limits,
        "restored_score": lab.score(), "new_model_calls": 0,
        "shared_method_preparation": "Existing helper inclusion and declared-trial routing guard only."})
    # Reconstruction is not new research; start the continuation wall account now.
    agent.started = monotonic() - prior_costs["wall_time_seconds"]
    return agent, lab, context


def prepared_request(agent, context):
    """Build the actual first child request without a provider or model call."""
    question = next(q for q in agent.questions if q["id"] == QUESTION_ID)
    session = {"notes": "", "events": [], "invocations": 1}
    frame = {"question_id": QUESTION_ID, "question": question,
             "assignment": deepcopy(question), "call_limit": agent.costs["model_calls"] + 20}
    focus = agent.research_focus
    agent.research_focus = None
    agent._inquiry_stack.append(frame)
    try:
        return inquiry_request(agent, session, frame)
    finally:
        agent._inquiry_stack.pop()
        agent.research_focus = focus


def physical_responses(result, confirmations):
    """Report actual new positive settings even if task completion precedes review."""
    records = result.get("observations", [])
    if not records:
        return []
    _, crystals, reactors, _, _ = matched.recipe._scan(records, matched.recipe._view(records[-1]))
    responses = matched.recipe._response_settings(records, crystals, reactors)
    prefix = result.get("prefix_observations", 0)
    previously_solved = {r["crystal"] for r in responses if r["activated"]
                         and any(i < prefix for i in r["evidence"])}
    fresh = []
    for response in responses:
        indices = [i for i in response["evidence"] if i >= prefix]
        if response["crystal"] not in (4, 5) or not response["activated"] or not indices:
            continue
        confirmed = any(p["number"] == response["crystal"] and p["physically_confirmed"]
            and any(e["index"] in indices for e in p["actual_positive_evidence"]) for p in confirmations)
        fresh.append({**response, "evidence": indices,
            "new_target_completion": response["crystal"] not in previously_solved,
            "formal_confirmation_recorded": confirmed})
    return fresh


def brief(result):
    prefix = result.get("prefix_observations", 0)
    trace = matched.scientific_trace({**result, "events": result.get("new_events", [])})
    new_ids = {e.get("inquiry_question_id") for e in result.get("new_events", [])}
    trace["inquiries"] = [q for q in trace["inquiries"] if q["question_id"] in new_ids]
    points = [p for p in matched.point_confirmations(result)
              if any(e["index"] >= prefix for e in p["actual_positive_evidence"])]
    before = result.get("declarations_before", [])
    added = [d for d in result.get("declarations_after", []) if d not in before]
    usages = [e["usage"] for e in result.get("new_events", []) if "usage" in e]
    cached = [u["cached_input_tokens"] for u in usages if isinstance(u.get("cached_input_tokens"), int)]
    return {key: result[key] for key in ("treatment", "repetition", "status")} | {
        "success": result.get("score", {}).get("completedSuccessfully") is True,
        "score": result.get("score"), "costs": result.get("new_costs", {}),
        "prior_costs": result.get("costs_before", {}), "prefix_observations": prefix,
        "added_or_revised_declarations": added, "source_changed": result.get("source_changed"),
        "new_observations": len(result.get("observations", [])) - prefix,
        "point_confirmations": points, "scientific_trace": trace,
        "new_positive_responses": physical_responses(result, points),
        "cache": {"reported_calls": len(cached), "hit_calls": sum(n > 0 for n in cached),
                  "missing_calls": result.get("new_costs", {}).get("model_calls", 0) - len(cached),
                  "cached_input_tokens": sum(cached)},
        **({"runner_error": result["runner_error"]} if "runner_error" in result else {})}


def run_case(source, output, treatment, repetition, text, config, base_limits):
    directory = Path(output) / f"repeat_{repetition:02d}" / treatment
    identity = {"treatment": treatment, "repetition": repetition}
    agent = context = None
    execution = None
    try:
        agent, lab, context = restore(source, directory, text, config=config, base_limits=base_limits)

        def checkpoint(data):
            new_events = data["events"][context["prefix_events"]:]
            latest = new_events[-1] if new_events else {}
            write_json(directory / "active.json", {**identity, "pid": os.getpid(),
                "status": data["status"], "score": data["score"],
                "costs": {k: value - context["costs_before"][k] for k, value in data["costs"].items()},
                "latest": {"action": latest.get("action"),
                    "operator": latest.get("arguments", {}).get("operator"),
                    "inquiry_question_id": latest.get("inquiry_question_id")}})

        agent.provider = CodexCodeProvider(agent.workspace, model=MODEL["requested_alias"],
            reasoning_effort=MODEL["reasoning_effort"], timeout_seconds=300)
        agent._on_checkpoint = checkpoint
        checkpoint(agent.result())
        execution = agent._execute("run", {"operator": "inquire",
            "question": context["question"], "max_model_calls": 20}, lab)
        if agent.status == "running":
            agent.run(lab, on_checkpoint=checkpoint)
        agent._save_state()
        result = agent.result()
    except Exception as error:
        result = agent.result() if agent is not None else {}
        result.update(status="runner_failed", runner_error={
            "type": type(error).__name__, "message": str(error)})
    result.update(identity, execution=execution)
    if agent is not None and context is not None:
        after = (agent.workspace / matched.study.METHOD_PATH).read_text()
        (directory / "source_after.py").write_text(after)
        result.update(observations=deepcopy(agent.observations),
            prefix_observations=context["prefix_observations"],
            costs_before=context["costs_before"], replay=context["replay"],
            new_costs={k: value - context["costs_before"][k] for k, value in result["costs"].items()},
            new_events=deepcopy(agent.events[context["prefix_events"]:]),
            declarations_before=context["declarations_before"], declarations_after=declarations(after),
            source_changed=after != (directory / "source_before.py").read_text())
        state = json.loads((agent.workspace / matched.recipe.STATE).read_text())
        result["reference_state"] = {k: state[k] for k in ("tests", "test") if k in state}
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "result.json", result)
    row = brief(result)
    write_json(directory / "brief.json", row)
    write_json(directory / "active.json", {**row, "run_finished": True, "pid": os.getpid()})
    return row


def summarize(rows):
    return {"planned_runs": 6, "finished_runs": len(rows),
        "status": "finished" if len(rows) == 6 else "running",
        "treatments": {t: {"finished_runs": len(selected),
            "successful_runs": sum(r["success"] for r in selected),
            "statuses": dict(Counter(r["status"] for r in selected)),
            "declaration_revision_runs": sum(bool(r["added_or_revised_declarations"]) for r in selected),
            "confirmed_points": sum(p["physically_confirmed"] for r in selected for p in r["point_confirmations"]),
            "newly_solved_targets": sum(len({p["crystal"] for p in r["new_positive_responses"]
                if p["new_target_completion"]}) for r in selected),
            "new_costs": {k: sum(r["costs"].get(k, 0) for r in selected) for k in (
                "model_calls", "environment_steps", "python_execution_seconds", "wall_time_seconds",
                "research_compute_units", "input_tokens", "output_tokens", "cached_input_tokens")}}
            for t in TREATMENTS for selected in [[r for r in rows if r["treatment"] == t]]}}


def run(output=OUTPUT, *, dry_run=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "runner.pid").write_text(str(os.getpid()) + "\n")
    texts, original = assignments()
    source_protocol = json.loads((SOURCE.parents[2] / "protocol.json").read_text())
    config, limits = source_protocol["cases"]["changed_mechanism"], source_protocol["limits"]
    jobs = [{"treatment": t, "repetition": n} for n in range(1, 4)
            for t in (TREATMENTS if n % 2 else tuple(reversed(TREATMENTS)))]
    write_json(output / "protocol.json", {"study": "same_state_assignment_comparison",
        "source": str(SOURCE), "history": str(HISTORY), "case": config,
        "treatments": texts, "jobs": jobs, "agent_arm": "hegel", "model": MODEL,
        "additional_allowance": ALLOWANCE, "inquiry_max_model_calls": 20,
        "python_seconds_per_call": 30, "provider_request_timeout_seconds": 300,
        "workers": 3, "auth_mode": None if dry_run else verify_codex_subscription(),
        "scope": "Six continuations of the same complete 184-observation development failure. "
            "Only assignment text differs. The historical instruction is retained before its old "
            "inline measurement table; all current evidence, files, notes, questions and prior costs "
            "are restored equally. Both use the existing declaration-trial repair. No target setting "
            "or new hypothesis is supplied. New inquiry has at most 20 calls; the original task loop "
            "receives the remaining common allowance. Previous failure is retained. Replay is not "
            "charged as new research. This is an operator-requested matched development continuation, "
            "not autonomous reopening, unseen discovery or a recursion ablation."})
    (output / "historical_assignment_original.txt").write_text(original)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())
    # Freeze the imported production modules and fixture that execute this cohort.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and str(filename).endswith(".py"):
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.parts[len(ROOT.parts)] in {"src", "scripts", "tests"}:
                destination = output / "frozen_runtime" / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(path.read_text())
    if dry_run:
        for treatment, text in texts.items():
            directory = output / treatment
            agent, _, context = restore(SOURCE, directory, text, config=config, base_limits=limits)
            write_json(directory / "prepared_request.json", prepared_request(agent, context))
        return {"status": "dry_run", "new_model_calls": 0, "output": str(output)}
    rows = []
    write_json(output / "summary.json", summarize(rows))
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_case, SOURCE, output, job["treatment"], job["repetition"],
            texts[job["treatment"]], config, limits): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                row = future.result()
            except Exception as error:
                row = brief({**job, "status": "runner_failed", "runner_error": {
                    "type": type(error).__name__, "message": str(error)}})
                directory = output / f"repeat_{job['repetition']:02d}" / job["treatment"]
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "runner_error.json", row)
                write_json(directory / "active.json", {**row, "run_finished": True})
            rows.append(row)
            rows.sort(key=lambda r: jobs.index({key: r[key] for key in ("treatment", "repetition")}))
            write_json(output / "brief.json", rows)
            write_json(output / "summary.json", summarize(rows))
            print(json.dumps({k: row[k] for k in ("treatment", "repetition", "status", "success")}), flush=True)
    return summarize(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, dry_run=args.dry_run), indent=2))

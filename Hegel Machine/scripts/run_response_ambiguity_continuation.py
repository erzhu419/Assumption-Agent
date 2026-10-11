#!/usr/bin/env python3
"""Continue the paid development branch with its original research and account."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.world_reference_recalibration import reference_research_question
from scripts import consolidate_response_measurement as study
from scripts.run_discoveryworld_method_repair import restore_repair
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json
from scripts.run_response_family_recovery import account, declarations

SOURCE = ROOT / "artifacts/response_boundary_progress_20261009"
ORIGINAL = ROOT / "artifacts/response_controller_adoption_20261009"


def restore(output, calls=32):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_workspace = SOURCE / "boundary_progress/agent"
    saved = json.loads((source_workspace / "research_state.json").read_text())
    previous_run = json.loads((ORIGINAL / "task_01/live/result.json").read_text())
    original_protocol = json.loads((ORIGINAL / "protocol.json").read_text())
    records = deepcopy(previous_run["observations"][:160])
    records += json.loads((SOURCE / "boundary_progress/new_observations.json").read_text())
    if records != json.loads((source_workspace / "laboratory_observations.json").read_text()):
        raise ValueError("Saved branch does not equal the paid original prefix plus recorded continuation")
    if len(records) != 192:
        raise ValueError("Expected the selected 192-observation checkpoint")
    branch = next(row for row in json.loads((SOURCE / "results.json").read_text())
                  if row["arm"] == "boundary_progress")
    events = deepcopy(previous_run["events"][:6])
    for event in events[:5]:
        if any(index >= 160 for index in event["feedback"].get("new_observation_indices", [])):
            raise ValueError("Prefix event includes later world evidence")
    events[-1]["feedback"] = json.loads((SOURCE / "boundary_progress/feedback.json").read_text())
    events[-1]["phase"] = "paid_proposal_replayed_in_selected_branch"
    costs = deepcopy(saved["costs"])
    costs["wall_time_seconds"] = branch["new_costs"]["wall_time_seconds"]
    usage_keys = {"input_tokens": "input_tokens", "output_tokens": "output_tokens",
                  "cached_input_tokens": "cached_input_tokens", "reported_tokens": "total_tokens"}
    for event in events:
        usage = event.get("usage", {})
        costs["model_calls"] += usage.get("calls", 1)
        costs["wall_time_seconds"] += usage.get("wall_time_seconds", 0)
        for cost_key, usage_key in usage_keys.items():
            costs[cost_key] += usage.get(usage_key, 0)
    costs.update(observation_cost=len(records), compute_cost=costs["research_compute_units"])
    research = {key: deepcopy(saved[key]) for key in (
        "questions", "methods", "relation_research", "research_advice", "research_focus",
        "prerequisite_checked_through")}
    research["llm_inquiries"] = json.loads((source_workspace / "llm_inquiries.json").read_text())
    previous = {"episode_status": "running", "research": research,
                "method_results": deepcopy(saved["method_results"]), "events": events,
                "costs": costs}
    limits = deepcopy(original_protocol["limits"])
    limits.update(max_model_calls=costs["model_calls"] + calls,
                  task_wall_seconds=costs["wall_time_seconds"] + 3600)
    study.CASES = {"task_01": original_protocol["cases"]["task_01"]["config"]}
    agent = DiscoveryWorldAgent(None, output / "agent", arm="hegel", limits=limits)
    lab = study.TransferSignalLab("task_01")
    agent.lab = lab
    replay = restore_repair(agent, lab, previous, source_workspace)
    source = (agent.workspace / study.METHOD_PATH).read_text()
    (output / "source_before.py").write_text(source)
    context = {"prefix_observations": len(records), "prefix_events": len(events),
               "costs_before": account(agent)["costs"], "replay": replay,
               "declarations_before": declarations(source)}
    write_json(output / "protocol.json", {
        "study": "live_adoption_of_returned_target_ambiguity", "source": str(SOURCE),
        "source_branch": "boundary_progress", "case": "task_01", "prefix_observations": 192,
        "model": {"requested_alias": "gpt-6-luna", "reasoning_effort": "medium"},
        "auth_mode": None, "additional_model_calls": calls, "additional_wall_seconds": 3600,
        "procedure_max_steps": 512, "limits": limits, "costs_before": context["costs_before"],
        "accounting": "All 192 paid observations, selected branch execution costs, and six original "
                      "proposal-prefix LLM usages are retained. Historical pre-160 offline compute, "
                      "Python execution and non-provider wall time lack a prefix checkpoint and are "
                      "not reconstructed. New costs are measured as continuation deltas. The original "
                      "later three model calls and their evidence are excluded.",
        "scope": "Resume the saved parent source, declarations, paid evidence and research memory. "
                 "The unchanged parent naturally starts its existing inquiry. No hypothesis, target "
                 "setting, new instrument or scientific hint is added by this runner. One previously "
                 "opened development condition; no matched performance or generalization claim.",
        "saved_inquiry_count": len(research["llm_inquiries"]),
        "historical_result": str(ORIGINAL / "task_01/live/result.json")})
    write_json(output / "replay.json", replay)
    write_json(output / "restored.json", {"status": "restored", "new_model_calls": 0,
        "source_unchanged": source == (source_workspace / study.METHOD_PATH).read_text(),
        "paid_records_equal": agent.observations == records, "costs": context["costs_before"],
        "score": lab.score(), "declarations": context["declarations_before"]})
    return agent, lab, context


def restore_saved(result_path, output, python_seconds=None, additional_calls=0, additional_wall_seconds=0):
    """Continue the saved child with its original account and invocation memory."""
    result_path, output = Path(result_path).resolve(), Path(output).resolve()
    previous = json.loads(result_path.read_text())
    protocol = json.loads((result_path.parent / "protocol.json").read_text())
    interrupted = previous["episode_status"] == "provider_failed"
    stopped = (previous["episode_status"] == "running"
               and previous["execution"]["status"] == "method_stopped")
    compute_stopped = (previous["episode_status"] == "running"
                       and (previous["execution"]["status"] == "python_budget_exhausted"
                            or previous["costs"]["python_execution_seconds"]
                            >= protocol["limits"]["python_seconds_per_task"]))
    calls_stopped = (additional_calls > 0 and (
        previous["episode_status"] == "model_call_budget_exhausted" or (
            previous["episode_status"] == "running"
            and previous["execution"]["status"] == "model_call_budget_exhausted")))
    inquiry_returned = (previous["episode_status"] == "running"
                        and previous["execution"]["status"] in {"inquiry_finished", "answered"})
    task_stopped = previous["episode_status"] == "agent_finished" and not previous["score"]["completed"]
    if not (interrupted or stopped or compute_stopped or calls_stopped or inquiry_returned or task_stopped):
        raise ValueError("Select an interrupted or unfinished stopped continuation")
    if python_seconds is not None:
        old = protocol["limits"]["python_seconds_per_task"]
        if python_seconds <= old:
            raise ValueError("Additional Python allowance must exceed the original total ceiling")
        protocol["python_budget_extension"] = {
            "from_seconds": old, "to_seconds": python_seconds,
            "origin": "operator_requested_continuation", "source": str(result_path),
            "scope": "Additional development allowance; not an unchanged-budget comparison. "
                     "All previous work remains charged."}
        protocol["limits"]["python_seconds_per_task"] = python_seconds
    if additional_calls:
        if additional_calls < 0:
            raise ValueError("Additional calls must be positive")
        old_calls = protocol["limits"]["max_model_calls"]
        protocol["limits"]["max_model_calls"] += additional_calls
        protocol["model_budget_extension"] = {
            "from_calls": old_calls, "to_calls": protocol["limits"]["max_model_calls"],
            "source": str(result_path), "origin": "operator_requested_continuation"}
    if additional_wall_seconds:
        if additional_wall_seconds < 0:
            raise ValueError("Additional wall allowance must be positive")
        old_wall = protocol["limits"]["task_wall_seconds"]
        protocol["limits"]["task_wall_seconds"] += additional_wall_seconds
        protocol["wall_budget_extension"] = {
            "from_seconds": old_wall, "to_seconds": protocol["limits"]["task_wall_seconds"],
            "source": str(result_path), "origin": "operator_requested_continuation"}
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((ORIGINAL / "protocol.json").read_text())
    study.CASES = {"task_01": original["cases"]["task_01"]["config"]}
    agent = DiscoveryWorldAgent(None, output / "agent", arm="hegel", limits=protocol["limits"])
    lab = study.TransferSignalLab("task_01")
    agent.lab = lab
    previous["episode_status"] = "running"
    replay = restore_repair(agent, lab, previous, result_path.parent / "agent")
    source = (agent.workspace / study.METHOD_PATH).read_text()
    (output / "source_before.py").write_text(source)
    request = json.loads((result_path.parent / "agent/research_context.json").read_text())
    assignment = deepcopy(request.get("llm_inquiry", {}).get("assignment"))
    # An interrupted proposal leaves its child assignment as the latest request.
    # Resume its recorded caller, not the child as if it were a parent program.
    if assignment and protocol.get("response_proposal") and not assignment.get("parent_program_path"):
        assignment = deepcopy(next(question for question in agent.questions
            if question["id"] == assignment["parent_id"]))
    write_json(output / "assignment_before.json", assignment)
    if assignment:
        state = json.loads((agent.workspace / "shared/reference_recalibration.json").read_text())
        number = next(n for n in state['target_numbers'] if n not in state.get('completed', []))
        assignment['question'] = reference_research_question(number, 'shared/' + assignment['id'] + '.json')
    context = {"prefix_observations": len(agent.observations), "prefix_events": len(agent.events),
        "costs_before": account(agent)["costs"], "replay": replay,
        "declarations_before": declarations(source), "resume_assignment": assignment,
        "resume_task": inquiry_returned or assignment is None}
    protocol["resume"] = {"result": str(result_path), "costs_before": context["costs_before"],
        "reason": ("provider_interruption" if interrupted else "python_budget_continuation"
                   if compute_stopped else "model_budget_continuation" if calls_stopped
                   else "returned_inquiry_task_continuation" if inquiry_returned
                   else "operator_requested_task_reopening" if task_stopped
                   else "operator_requested_interface_continuation"),
        "assignment_revision": "Preserve original assignment in assignment_before.json; use the current "
                               "production scientific assignment, with opening snapshot labeled historical "
                               "and current methods/evidence referenced rather than frozen inline.",
        "remaining_model_calls": protocol["limits"]["max_model_calls"] - agent.costs["model_calls"],
        "scope": "Resume an interrupted child or return a completed inquiry to the existing task loop, "
                 "retaining its notebook, events and world. The world-step ceiling is unchanged; "
                 "any additional model/Python/wall allowance is recorded separately in this protocol; "
                 "no new scientific message is inserted. The current tool-interface instructions "
                 "are supplied by the production agent. A normal prior stop is retained in its "
                 "source result; this explicit continuation is not autonomous reopening."}
    write_json(output / "protocol.json", protocol)
    write_json(output / "replay.json", replay)
    return agent, lab, context


def run(output, calls=32, dry_run=False, resume=None, proposal_path=None, python_seconds=None, additional_calls=0, additional_wall_seconds=0):
    agent, lab, context = restore_saved(resume, output, python_seconds, additional_calls, additional_wall_seconds) if resume else restore(output, calls)
    output = Path(output).resolve()
    if dry_run:
        return {"status": "restored", "new_model_calls": 0,
                "prefix_observations": context["prefix_observations"]}
    auth = verify_codex_subscription()
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["auth_mode"] = auth
    write_json(output / "protocol.json", protocol)
    agent.provider = CodexCodeProvider(agent.workspace, model="gpt-6-luna",
        reasoning_effort="medium", timeout_seconds=180)

    def checkpoint(data):
        latest = data["events"][-1] if len(data["events"]) > context["prefix_events"] else {}
        brief = {"status": "running", "costs": data["costs"], "score": data["score"],
            "new_model_calls": data["costs"]["model_calls"] - context["costs_before"]["model_calls"],
            "new_actions": lab.environment_steps - context["prefix_observations"],
            "latest_event": {"action": latest.get("action"),
                "operator": latest.get("arguments", {}).get("operator"),
                "status": latest.get("feedback", {}).get("status"),
                "inquiry_question_id": latest.get("inquiry_question_id")}}
        write_json(output / "active.json", brief)
        print(json.dumps({key: brief[key] for key in ("status", "new_model_calls", "new_actions")}), flush=True)

    agent._on_checkpoint = checkpoint
    checkpoint(account(agent))
    if context.get("resume_task"):
        task = agent.run(lab, on_checkpoint=checkpoint)
        execution = {"operator": "task_continuation", "status": task["status"]}
    elif resume:
        proposal = None
        if proposal_path:
            assignment = context["resume_assignment"]
            protocol["response_proposal"] = {
                "path": proposal_path, "response_path": assignment["parent_program_path"],
                "origin": "operator_requested_executable_proposal_continuation",
                "scope": "Host requests an executable proposal; the model chooses all scientific "
                         "content. Draft writes use the production paid-comparison path. "
                         "No offline binding, excluded family or target setting is supplied. "
                         "Budget changes, if requested, are recorded in the extension entries."}
            write_json(output / "protocol.json", protocol)
            proposal = agent._execute("run", {"operator": "propose_explanations",
                "path": proposal_path, "response_path": assignment["parent_program_path"],
                "question": {"id": assignment["id"] + "_representation",
                    "parent_id": assignment["id"], "question": assignment["question"], "depends_on": []},
                "max_model_calls": max(1, min(calls,
                    agent.limits["max_model_calls"] - agent.costs["model_calls"] - 2))}, lab)
            write_json(output / "proposal.json", proposal)
        execution = agent._execute("run", {"operator": "inquire",
            "question": context["resume_assignment"],
            "max_model_calls": agent.limits["max_model_calls"] - agent.costs["model_calls"]}, lab)
        if agent.status == "running" and execution["status"] in {"inquiry_finished", "answered"}:
            task = agent.run(lab, on_checkpoint=checkpoint)
            execution = {"operator": "task_continuation", "status": task["status"],
                         "resumed_inquiry": execution}
        elif agent.status == "running" and not protocol.get("response_proposal"):
            parent = agent._run_procedure(lab, study.METHOD_PATH, max_steps=512)
            execution = {"status": parent["status"], "resumed_inquiry": execution,
                         "parent_continuation": parent}
        if proposal is not None:
            execution["response_proposal"] = proposal
    else:
        execution = agent._run_procedure(lab, study.METHOD_PATH, max_steps=512)
    agent._save_state()
    result = account(agent)
    result.update(study_status="finished", episode_status=result["status"], execution=execution,
        costs_before=context["costs_before"],
        new_costs={key: value - context["costs_before"][key] for key, value in result["costs"].items()},
        observations=deepcopy(agent.observations), replay=context["replay"])
    write_json(output / "result.json", result)
    source = (agent.workspace / study.METHOD_PATH).read_text()
    (output / "source_after.py").write_text(source)
    brief = {"status": "finished", "execution_status": execution["status"],
        "success": result["score"]["completedSuccessfully"], "score": result["score"],
        "new_costs": result["new_costs"], "total_actions": lab.environment_steps,
        "declarations_before": context["declarations_before"], "declarations_after": declarations(source),
        "new_events": [{"action": event["action"], "operator": event.get("arguments", {}).get("operator"),
            "hypothesis": event.get("arguments", {}).get("hypothesis"),
            "status": event.get("feedback", {}).get("status"),
            "paid_family_status": event.get("feedback", {}).get("paid_family_check", {}).get("status"),
            "tests": [{key: test.get(key) for key in ("number", "family", "prediction", "status", "question")}
                      for test in event.get("feedback", {}).get("tests", [])],
            "observed_responses": event.get("feedback", {}).get("observed_responses"),
            "error": event.get("feedback", {}).get("error"),
            "inquiry_question_id": event.get("inquiry_question_id")}
            for event in agent.events[context["prefix_events"]:]],
        "inquiries": [{"id": key, "status": value.get("status"), "model_calls": value.get("model_calls")}
            for key, value in agent.llm_inquiries.items()]}
    write_json(output / "brief.json", brief)
    write_json(output / "active.json", {key: brief[key] for key in (
        "status", "execution_status", "success", "score", "new_costs", "total_actions")})
    return brief


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--proposal-path", help="Request a computed response draft before resuming the caller")
    parser.add_argument("--python-seconds", type=float, help="Explicitly extend the cumulative Python allowance")
    parser.add_argument("--additional-calls", type=int, default=0, help="Record additional continuation model calls")
    parser.add_argument("--additional-wall-seconds", type=float, default=0,
                        help="Record extra cumulative wall allowance for this continuation")
    args = parser.parse_args()
    result = run(args.output, args.calls, args.dry_run, args.resume, args.proposal_path,
                 args.python_seconds, args.additional_calls, args.additional_wall_seconds)
    print(json.dumps({key: result[key] for key in ("status", "success", "new_model_calls") if key in result}), flush=True)

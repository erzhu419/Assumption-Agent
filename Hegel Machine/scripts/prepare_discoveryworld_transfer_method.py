#!/usr/bin/env python3
"""Let the source model consolidate its public trajectory before fresh transfer."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent, INTERFACE, INQUIRY_INTERFACE
from hegel_machine.cognition.world_method_learning import WorldMethodLearner, compact_author_events
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.world_prerequisite_research import public_reference_history
from scripts.run_discoveryworld_pilot import SELECTION, write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription


AUTHOR_TASK = """
SOURCE METHOD LEARNING: turn the public trajectory in shared/source_episode.json
into a reusable self-contained research program at shared/consolidated_method.py.
The archive is earlier source experience, never a new practice's observations.
source_experiences exposes each source author's existing notebook and archive
sections. These notebooks may precede the episode's final actions; use the
linked public records to check their hypotheses and outcomes. Read selected
messages or sections, rather than printing entire archives into truncated stdout.
Rebind current public objects and measurements, calculate candidate predictions,
execute and interpret actual tests. Old IDs, record indices, coefficients and
frequencies are source facts, not target bindings. Use current-invocation phases.

Authoring actions are write at shared/consolidated_method.py, ordinary Python run,
run {operator:'practice',case:<a key from practice_cases>},
run {operator:'restore_method',trial:'trials/trial_NNN'}, and
finish {path:'shared/consolidated_method.py',lesson:'...'}.
Registering with run {operator:'method',path:'shared/consolidated_method.py',...}
requests the same finish procedure; registration alone never publishes.
Changing the entry source automatically executes one still-missing source practice
and returns its actual failures. finish completes missing practices, compares a
previous genuinely supported source if this revision fails, and publishes only a
source that solves every listed practice case. A source must retain its exact entry
code throughout each passing practice. When a revised entry completes a world,
practice reruns that candidate without an LLM in a fresh instance of the same
case, retaining it only if that unchanged-source execution succeeds. Otherwise
the previous draft remains selected; both trials and sources stay available.
Inspect method_decision.json for actual
paid support and explicit restore actions; no support is supplied for an old draft.

source_initial starts the source development world before any action;
source_continuation replays its paid public actions to the unfinished checkpoint.
Both use only the source development task. An already completed world is not a
practice. Practices reset local questions, observations and authored scratch files;
the continuation supplies its replayed observations, never old supported answers.
A program may call run operator:'inquire' using the child-loop contract above.
Those are real model calls and share the total author budget, not free reasoning.
Actual procedure errors and repeated public navigation/dialogue cycles also open
that researcher during practice. Inspect repaired_candidate_confirmation and
current_source_coverage before editing again; retain phases already working.
Automatic rescue is research feedback, not support for the original program,
even if the researcher completes the world without editing. Incorporate the
successful steps and rerun. Explicit inquire calls authored in your program
remain part of that program. When shared/source_repair.json is supplied, it is
additional public source experience: consolidate its successful interactions
with the original source method before the next practice. The saved repaired
draft may only handle the late phase where it was edited. Preserve earlier
working phases and reconstruct each next action from current public feedback.
Additional source cases start previously opened tasks from zero current actions.
Their archived experience is also development data. Use practice_cases to choose
them. Additional practice tasks need no successful archive: their actual failed
executions are counterexamples returned for revision, and they become development
tasks. Their conditions may differ, so one source's control constants are not a
reusable estimator. Retain the working navigation/control program; restore the
source's measurement, calculation and physical confirmation steps, fitting from
the current world. A reference readback labels a correct setting only when the
reference is actually activated. Failed activation requires revising the current
prediction or its assumed observable, not claiming success from a setpoint.
When consolidating a newly successful source, keep the working program and add
the measurement and estimation steps justified by that source's actual records.
The intended knowledge product executes with no model calls on the practiced
sources; delegation remains a fallback for unresolved future tasks. After
publication, unchanged zero-model execution is checked separately. Repeating a
child's live discovery on each source is not consolidation of its successful steps.
Use public current evidence; do not inspect benchmark internals or other workspaces.
Scientific laws and action schedules remain your decisions. The host does not fix
your program. Source-practice completion is not held-out transfer confirmation.
"""


def source_archive(source):
    source = Path(source)
    result = json.loads((source / "result.json").read_text())
    if not result["score"]["completedSuccessfully"]:
        raise ValueError("This consolidation uses an actually successful source episode")
    observations = json.loads((source / "agent/laboratory_observations.json").read_text())
    context = json.loads((source / "agent/research_context.json").read_text())
    recent = result["events"][result.get("prefix_event_count", 0):]
    start = result.get("prefix_environment_steps", result.get("replayed_environment_steps", 0))
    return {
        "origin": "Public source episode; no target observations or answers",
        "record_kind": "source_experience_not_consolidated_program_execution",
        "method_validation": "The exported program was authored after these events; it has no world execution yet.",
        "source": str(source.resolve()),
        "public_spec": {"capabilities": context["capabilities"],
                        "current_observation": deepcopy(observations[-1]["observation"])},
        "events": [{"action": event["action"], "arguments": deepcopy(event.get("arguments", {})),
            "feedback": {key: deepcopy(value) for key, value in event.get("feedback", {}).items()
                         if key in {"stdout", "stderr", "error", "returncode", "analysis", "question", "status"}}}
            for event in recent],
        "notes": context.get("research_notes", ""),
        "measurement_history": [DiscoveryWorldAgent._action_observation(row)
            for row in observations if row["action"].get("action") in {"USE", "READ"}],
        "reference_history": public_reference_history(observations),
        "continuation_observations": [DiscoveryWorldAgent._action_observation(row)
            for row in observations[start:]],
    }


def repair_experience(repair_result, source):
    """Return successful local repair to the existing source method author."""
    path = Path(repair_result).resolve()
    result = json.loads(path.read_text())
    protocol = json.loads((path.parent / "protocol.json").read_text())
    learning = Path(protocol["trial"]).parents[2]
    original = json.loads((learning / "protocol.json").read_text())
    if Path(original["source"]).resolve() != Path(source).resolve():
        raise ValueError("Repair must belong to the same development source")
    if not result["score"]["completedSuccessfully"]:
        raise ValueError("Consolidation needs a successful source repair")
    workspace = path.parent / "agent"
    records = json.loads((workspace / "laboratory_observations.json").read_text())
    draft_path = "shared/source_repaired_method.py"
    archive = {
        "origin": "Public source repair experience, separate from each new practice",
        "record_kind": "source_experience_not_consolidated_program_execution",
        "method_validation": "Interactive source recovery only; unchanged program not confirmed",
        "revised_source_path": draft_path,
        "observations": [{**DiscoveryWorldAgent._action_observation(row),
            "agentLocation": deepcopy(row["observation"].get("agentLocation", {}))}
            for row in records],
        "questions": deepcopy(result["research"]["questions"]),
        "notes": (workspace / "research_notes.md").read_text()
            if (workspace / "research_notes.md").exists() else "",
        "inquiry_notes": {key: item["notes"] for key, item in
                          result["research"].get("llm_inquiries", {}).items()},
    }
    previous = json.loads((learning / "result.json").read_text())
    return {"shared/source_repair.json": json.dumps(archive, ensure_ascii=False),
            draft_path: (workspace / METHOD_PATH).read_text()}, {
        "result": str(path), "costs": result["costs"],
        "source_learning_result": str(learning / "result.json"),
        "source_learning_costs": previous["costs"],
        "accounting": "Earlier learning and repair costs are retained as prior work, "
                      "not added to the new author phase or copied as practice support."}


def additional_source(source):
    source = Path(source).resolve()
    archive = source_archive(source)
    result = json.loads((source / "result.json").read_text())
    protocol_path = source / "protocol.json"
    if not protocol_path.exists():
        protocol_path = source.parents[1] / "protocol.json"
    plan = json.loads(protocol_path.read_text())
    tasks = [plan["task"]] if "task" in plan else plan["tasks"]
    task = next((task for task in tasks if task["id"] == result["task_id"]), None)
    if task is None:
        raise ValueError("Source protocol must identify the result's actual task")
    task = deepcopy(task)
    return task, archive, {"source": str(source), "task": task, "costs": result["costs"],
        "scope": "Previously opened source experience reused for development; excluded from new held-out evidence."}


def prepare(source, output, calls=None, initial_bundle=None, resume_author=None, repair_result=None,
            additional_sources=(), frozen_practice=False, practice_task_ids=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    previous = previous_protocol = None
    if resume_author:
        if initial_bundle or repair_result or additional_sources or practice_task_ids:
            raise ValueError("Resume keeps the existing draft and source experience")
        resume_author = Path(resume_author).resolve()
        previous = json.loads(resume_author.read_text())
        previous_protocol = json.loads((resume_author.parent / "protocol.json").read_text())
        frozen_practice = previous_protocol.get("frozen_practice", False)
        if Path(previous["source"]).resolve() != source:
            raise ValueError("Author recovery must use the same development source")
        if calls is not None and calls != previous_protocol["max_model_calls"]:
            raise ValueError("Author recovery retains the original total model-call allowance")
        calls = previous_protocol["max_model_calls"]
    elif calls is None:
        calls = 32
    practice_tasks = deepcopy(previous_protocol.get("additional_practice_tasks", [])) if previous else []
    if practice_task_ids:
        registered = {task["id"]: task for task in json.loads(SELECTION.read_text())["tasks"]}
        practice_tasks = [deepcopy(registered[task_id]) for task_id in dict.fromkeys(practice_task_ids)]
    auth_mode = verify_codex_subscription()
    plan = json.loads((source.parents[1] / "protocol.json").read_text())
    source_result = json.loads((source / "result.json").read_text())
    task = next(task for task in plan["tasks"] if task["id"] == source_result["task_id"])
    if (task["split"], task["difficulty"], task["random_seed"]) != ("development", "Normal", 0):
        raise ValueError("Method learning uses the Normal seed-0 development source only")
    archive = (json.loads(previous["outgoing_knowledge"]["files"]["shared/source_episode.json"])
               if previous else source_archive(source))
    records = json.loads((source / "agent/laboratory_observations.json").read_text())
    prefix_steps = source_result["prefix_environment_steps"]
    seed = "def select_actions(spec, observations, memory):\n    return {'stop': 'Author a research procedure from the public source experience'}\n"
    if initial_bundle:
        initial = json.loads(Path(initial_bundle).read_text())
        seed = initial["files"][initial["methods"][0]["path"]]
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "author"

    def provider_factory(path):
        return CodexCodeProvider(path, model=plan["model"]["requested_alias"],
            reasoning_effort=plan["model"]["reasoning_effort"],
            timeout_seconds=plan["limits"]["provider_request_timeout_seconds"])

    knowledge = {"files": {METHOD_PATH: seed,
        "shared/source_episode.json": json.dumps(archive, ensure_ascii=False)}}
    repair = None
    if repair_result:
        repair_files, repair = repair_experience(repair_result, source)
        knowledge["files"].update(repair_files)
    if previous:
        knowledge = deepcopy(previous["outgoing_knowledge"])
    additions = deepcopy(previous_protocol.get("additional_sources", [])) if previous else []
    for extra in additional_sources:
        extra_task, extra_archive, provenance = additional_source(extra)
        knowledge["files"][f"shared/source_{extra_task['id']}.json"] = json.dumps(extra_archive, ensure_ascii=False)
        additions.append(provenance)
    agent = WorldMethodLearner(provider_factory(workspace), workspace, knowledge=knowledge,
        task=task, records=records, prefix_steps=prefix_steps, public_spec=archive["public_spec"],
        limits=plan["limits"], calls=calls, runtime=plan["runtime"],
        provider_factory=(lambda _: None) if frozen_practice else provider_factory,
        max_steps=previous_protocol.get("practice_step_limit", 64) if previous else 256,
        additional_tasks=[*[item["task"] for item in additions], *practice_tasks])
    if previous:
        restored = agent.restore_practice(previous)
        agent.costs = deepcopy(previous["costs"])
        # Keep the original wall allowance and report elapsed work across phases;
        # time between the stopped phase and this resume is not research time.
        agent.started -= agent.costs["wall_time_seconds"]
        agent.events = deepcopy(previous["events"])
        agent.lesson = previous["lesson"]
        agent.manifest = deepcopy(previous["outgoing_knowledge"].get("manifest", {}))
        agent.questions = deepcopy(previous.get("online_questions", []))
        notes = Path(previous["workspace"]) / "research_notes.md"
        if notes.exists():
            (workspace / notes.name).write_text(notes.read_text())
        else:
            previous_notes = next((event["arguments"]["notes"] for event in reversed(agent.events)
                if event["action"] != "provider_error" and event.get("arguments", {}).get("notes")), None)
            if previous_notes is not None:
                (workspace / "research_notes.md").write_text(previous_notes)
        restored["cost_scope"] = ("All earlier paid costs and model calls remain in the current cumulative "
            "account under the original total limits. Restored trials are copied, not re-executed.")
        agent.files["resume_context.json"] = json.dumps(restored, indent=2)
        write_json(workspace / "resume_context.json", restored)
        agent._refresh_method_candidates()
        agent._prepare_program_state()
    protocol = deepcopy(previous_protocol) if previous else {
        "source": str(source), "model": plan["model"], "auth_mode": auth_mode,
        "max_model_calls": calls, "practice_cases": agent.cases, "practice_step_limit": agent.max_steps,
        "frozen_practice": frozen_practice,
        "initial_bundle": str(Path(initial_bundle).resolve()) if initial_bundle else None,
        "initial_consolidation_costs": deepcopy(initial["consolidation_costs"]) if initial_bundle else None,
        "initial_accounting": str(Path(initial_bundle).resolve().parent / "accounting.json")
            if initial_bundle else None,
        "source_repair": repair,
        "additional_sources": additions,
        "additional_practice_tasks": practice_tasks,
        "source_prior_costs": source_result["prior_costs"], "source_last_costs": source_result["costs"],
        "validation": "Publish only after real source-world practice of unchanged entry code on every listed case",
        "accounting": "Author and child model calls share one account; replay and new world actions are reported separately. "
                      "wall_time_seconds is elapsed research time across author phases, including author requests and practices."}
    if previous:
        protocol.update(auth_mode=auth_mode, resume_author={"result": str(resume_author),
            "previous_costs": deepcopy(previous["costs"]),
            "previous_event_count": len(previous["events"]),
            "remaining_model_calls": max(0, calls - agent.costs["model_calls"]),
            "restoration": "Existing restore_practice retains exact sources, paid trials and support; "
                           "earlier costs remain charged to the original total allowance."},
            accounting="Author and child costs are cumulative across phases; earlier trials are copied, not replayed. "
                       "wall_time_seconds adds elapsed research phases, excluding time between them.")
    write_json(output / "protocol.json", protocol)
    source_experiences = []
    for name in ["shared/source_episode.json",
                 *[f"shared/source_{item['task']['id']}.json" for item in additions]]:
        experience = json.loads(knowledge["files"][name])
        source_experiences.append({"file": name, "notes": experience.get("notes", ""),
            "sections": {key: len(experience[key]) for key in
                ("measurement_history", "reference_history", "continuation_observations", "events")}})
    # The inherited draft gets measured feedback before the first revision.
    if previous is None and repair is None:
        for case in ["source_initial", *[f"{task['id']}_initial" for task in practice_tasks]]:
            first = agent._practice(case)
            agent.events.append({"action": "run", "arguments": {"operator": "practice", "case": case},
                                 "feedback": first, "origin": "initial_source_practice"})
    status = "authoring"

    def save():
        agent.costs["wall_time_seconds"] = monotonic() - agent.started
        state = agent.result(status)
        state.update(source=str(source), model=plan["model"], auth_mode=auth_mode,
                     validation=protocol["validation"])
        write_json(output / "result.json", state)

    save()
    while agent.costs["model_calls"] < calls and agent.published_source is None:
        agent._prepare_program_state()
        request = {"task_interface": INTERFACE + INQUIRY_INTERFACE + AUTHOR_TASK,
            "files": {path: (workspace / path).read_text() for path in agent.files},
            "source_experience_file": "shared/source_episode.json",
            "source_experiences": source_experiences,
            "practice_cases": deepcopy(agent.cases),
            "research_notes": (workspace / "research_notes.md").read_text()
                if (workspace / "research_notes.md").exists() else "",
            "events": compact_author_events(agent.events), "costs": deepcopy(agent.costs),
            "limits": {**plan["limits"], "max_model_calls": calls}}
        if frozen_practice:
            request["task_interface"] += ("\nIn this consolidation study every practice executes without an LLM. "
                "You are the method author: use actual source experience to revise the measurement, "
                "estimation and test program. A failure returns here for you to repair; a practice "
                "cannot delegate its missing estimator to another model. Preserve working phases.")
        decision = agent._model_decision(request)
        action, arguments = decision["action"], decision.get("arguments", {})
        started = monotonic()
        recorded_wall = agent.costs["wall_time_seconds"]
        if action == "provider_error":
            feedback = {"error": decision.get("error")}
        else:
            try:
                feedback = agent._execute(action, arguments, None)
            except (ValueError, KeyError, TypeError, OSError) as error:
                feedback = {"error": f"{type(error).__name__}: {error}"}
        agent.events.append({"action": action, "arguments": deepcopy(arguments),
                             "feedback": feedback, "usage": decision.get("usage", {})})
        # Practices already charged their duration, including child calls.
        # Add only the outer author request and remaining dispatch/inspection time.
        charged_practice_wall = agent.costs["wall_time_seconds"] - recorded_wall
        agent.costs["wall_time_seconds"] += (decision.get("usage", {}).get("wall_time_seconds", 0)
            + max(0., monotonic() - started - charged_practice_wall))
        status = "published_source_practiced" if agent.published_source else "authoring"
        save()
        print(json.dumps({"calls": agent.costs["model_calls"], "action": action,
                          "status": status, "error": feedback.get("error")}), flush=True)
        if action == "provider_error":
            error = decision.get("error")
            if not isinstance(error, dict) or not error.get("decode_error"):
                status = "provider_failed"
                break
    if agent.published_source:
        practice = agent.public_practice_archive()
        write_json(output / "source_practice.json", practice)
        files = {METHOD_PATH: agent.published_source,
                 "shared/source_episode.json": knowledge["files"]["shared/source_episode.json"],
                 "shared/source_practice.json": json.dumps(practice, ensure_ascii=False)}
        files.update({path: content for path, content in knowledge["files"].items()
                      if any(path == f"shared/source_{item['task']['id']}.json" for item in additions)})
        entry = {"id": "practiced_world_research", "path": METHOD_PATH, "parameters": {},
                 "max_steps": agent.max_steps, "requires": [], "provenance": "model_source_practiced",
                 "evidence_file": "shared/source_practice.json", "applicability": agent.lesson}
        write_json(output / "method_bundle.json", {"files": files, "methods": [entry],
            "source": str(source), "consolidation_costs": deepcopy(agent.costs),
            "prior_consolidation_costs": protocol["initial_consolidation_costs"],
            "development_task_ids": list(dict.fromkeys([task["id"],
                *[item["task"]["id"] for item in additions], *[item["id"] for item in practice_tasks]])),
            "validation": "All listed source-world cases completed with unchanged entry source; held-out transfer untested"})
        status = "published_source_practiced"
    elif status == "authoring":
        status = "model_call_budget_exhausted"
    save()
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calls", type=int, help="Total author calls (default 32, or the original limit on resume)")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--initial-bundle", type=Path, help="Earlier unvalidated draft; no support is inherited")
    start.add_argument("--resume-author", type=Path, help="Interrupted author result; retain paid trials and original total allowance")
    parser.add_argument("--repair-result", type=Path,
                        help="Successful same-source repair to consolidate before new practice")
    parser.add_argument("--additional-source", action="append", default=[], type=Path,
                        help="Previously opened successful episode to reuse as another development source")
    parser.add_argument("--frozen-practice", action="store_true",
                        help="Author with the LLM, but execute source practices without a model")
    parser.add_argument("--practice-task-ids", nargs="+", default=[],
                        help="Registered tasks to open as development practices, including failed transfers")
    args = parser.parse_args()
    prepare(args.source, args.output, args.calls, args.initial_bundle, args.resume_author,
            args.repair_result, args.additional_source, args.frozen_practice, args.practice_task_ids)

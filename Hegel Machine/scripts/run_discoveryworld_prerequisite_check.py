#!/usr/bin/env python3
"""Check research adoption from one frozen public DiscoveryWorld prefix.

This is a short integration continuation, not a whole-task performance study.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from scripts.run_discoveryworld_pilot import failed_result, summarize, write_json
from scripts.run_online_inquiry_llm import verify_codex_subscription


def freeze_prefix(source, destination, *, prefix_calls=None):
    """Read the live checkpoint once; later work reads only these frozen files."""
    source, destination = Path(source).resolve(), Path(destination)
    source_plan = json.loads((source.parents[1] / "protocol.json").read_text())
    active = json.loads((source / "active.json").read_text())
    task = next(task for task in source_plan["tasks"] if task["id"] == source.parent.name)
    steps = active["costs"]["environment_steps"]
    records = json.loads((source / "agent/laboratory_observations.json").read_text())
    records = [record for record in records if record["environment_steps"] <= steps]
    if (records[-1]["environment_steps"] if records else 0) != steps:
        raise ValueError("The public records do not reach the frozen checkpoint's environment step")
    if [record["index"] for record in records] != list(range(len(records))):
        raise ValueError("The frozen public record prefix is not contiguous")
    if len(records) != active["score"]["agent_actions"]:
        raise ValueError("Checkpoint action count and frozen public records disagree")

    events = deepcopy(active["events"])
    research = deepcopy(active["research"])
    method_results = deepcopy(active.get("method_results", []))
    prior_costs = deepcopy(active.get("prior_costs", {}))
    for name, value in active["costs"].items():
        # Environment steps already include the prefix; model/Python usage is
        # charged only to the immediately preceding continuation.
        prior_costs[name] = value if name == "environment_steps" else prior_costs.get(name, 0) + value
    if prefix_calls is None:
        state_path = source / "agent/research_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        research.setdefault("research_focus", deepcopy(state.get("research_focus")))
        # Older completed continuations did not serialize this cursor. Their
        # last request checked the observations preceding its resulting actions.
        last_indices = events[-1].get("feedback", {}).get("observation_indices", []) if events else []
        research.setdefault("prerequisite_checked_through",
                            state.get("prerequisite_checked_through",
                                (min(last_indices) - 1 if last_indices else len(records) - 1)
                                if "prefix_event_count" in active else -1))
    if prefix_calls is not None:
        if not 1 <= prefix_calls <= len(events) or len(events) != active["costs"]["model_calls"]:
            raise ValueError("--prefix-calls requires one recorded event per call and an available positive cutoff")
        events = events[:prefix_calls]
        # This cutoff resumes the observed pre-research failure state. Do not
        # reconstruct a past program/answer state from a later checkpoint.
        for event in events:
            arguments = event.get("arguments", {})
            questions = [arguments.get(name) for name in ("question", "research_question")]
            if (event["action"] == "evaluate" or arguments.get("operator")
                    or (event["action"] == "query" and (arguments.get("path") or arguments.get("method_id")))
                    or any(question and (question.get("id") != "task" or question.get("parent_id"))
                           for question in questions)):
                raise ValueError("Historical call cutoff only supports a root-only prefix before research programs/answers")
        indices = [index for event in events for index in event.get("feedback", {}).get("observation_indices", [])]
        records = records[:max(indices, default=-1) + 1]
        steps = records[-1]["environment_steps"] if records else 0
        root = next((deepcopy(event["arguments"]["research_question"]) for event in reversed(events)
                     if event.get("arguments", {}).get("research_question")), None)
        if root is None or root.get("depends_on"):
            raise ValueError("Historical call cutoff requires the recorded independent public task root")
        root.update(status="open", evidence=[], opened_before_observation=0, origin="public_task",
                    acquired_observations=[index for event in events
                        if (event.get("arguments", {}).get("question") or
                            event.get("arguments", {}).get("research_question") or {}).get("id") == "task"
                        for index in event.get("feedback", {}).get("observation_indices", [])])
        research = {"questions": [root], "methods": [], "relation_research": {}, "research_advice": None,
                    "child_answers_returned": 0, "program_executions": 0}
        method_results = []
        usage = [event.get("usage", {}) for event in events]
        prior_costs = {"model_calls": prefix_calls, "environment_steps": steps,
                       "usage_missing_calls": sum(not isinstance(item.get("total_tokens"), int) for item in usage)}
        for field in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens"):
            prior_costs["reported_tokens" if field == "total_tokens" else field] = sum(
                item[field] for item in usage if isinstance(item.get(field), int))
    notes = next((event["arguments"]["notes"] for event in reversed(events)
                  if event.get("arguments", {}).get("notes")), "")
    prefix = {"source": str(source), "task": task, "environment_steps": steps,
              "prefix_calls": prefix_calls, "prior_costs": prior_costs, "records": records,
              "events": events, "notes": notes,
              "research": research, "method_results": method_results,
              "source_checkpoint_costs": deepcopy(active["costs"]),
              "source_prior_costs": deepcopy(active.get("prior_costs", {})),
              "cost_scope": ("Call-cut prefix costs reconstruct calls and reported token usage from its events; "
                             "Python/wall costs are not reconstructed. Source checkpoint totals are listed separately."
                             if prefix_calls is not None else
                             "Prefix costs combine prior continuations and the source checkpoint; environment steps "
                             "are cumulative, not added twice. Python/wall costs are known subtotals if an ancestral "
                             "call-cut prefix lacked those costs.")}
    files = destination / "prefix_files"
    files.mkdir()
    references, authored = set(), {}

    def find_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"executed_path", "stdout_file", "stderr_file"} and item:
                    references.add(item)
                else:
                    find_paths(item)
        elif isinstance(value, list):
            for item in value:
                find_paths(item)

    for event in events:
        arguments = event.get("arguments", {})
        find_paths(event.get("feedback", {}))
        if arguments.get("path"):
            references.add(arguments["path"])
        if event["action"] == "write" and "code" in arguments:
            authored[arguments["path"]] = arguments["code"]
        if event["action"] == "run" and "code" in arguments:
            path = event.get("feedback", {}).get("executed_path") or arguments.get("path")
            if path:
                authored[path] = arguments["code"]
    source_files = (source / "agent").resolve()
    registered = set(authored)
    if prefix_calls is None:
        # Programs generated by research operators need not have a write event.
        # Preserve their current contents, including edits made by run.
        shared = {str(path.relative_to(source_files)) for path in (source_files / "shared").rglob("*")
                  if path.is_file()}
        references.update(shared)
        registered.update(shared)
    for name in sorted(references | set(authored)):
        path = (source_files / name).resolve()
        relative = path.relative_to(source_files)
        target = files / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if prefix_calls is not None and name in authored:
            target.write_text(authored[name], encoding="utf-8")
        else:
            shutil.copyfile(path, target)
    prefix["copied_files"] = sorted(str(path.relative_to(files)) for path in files.rglob("*") if path.is_file())
    prefix["registered_files"] = sorted(registered)
    write_json(destination / "prefix.json", prefix)
    return source_plan, prefix


def replay_prefix(lab, prefix):
    for record in prefix["records"]:
        replayed = lab.act(record["action"])
        expected = (record["feedback"].get("success"), record["observation"].get("lastActionMessage", ""))
        actual = (replayed["feedback"].get("success"), replayed["observation"].get("lastActionMessage", ""))
        if actual != expected:
            raise ValueError(f"Public replay differs at record {record['index']}: {actual!r} != {expected!r}")
    if lab.environment_steps != prefix["environment_steps"]:
        raise ValueError("Replay environment step does not match the frozen checkpoint")
    return {"status": "matched", "records": len(prefix["records"]),
            "environment_steps": lab.environment_steps,
            "compared": ["feedback.success", "observation.lastActionMessage"]}


def continue_arm(plan, arm, output):
    output = Path(output)
    prefix = json.loads((output / "prefix.json").read_text())
    task = prefix["task"]
    workspace = output / task["id"] / arm
    workspace.mkdir(parents=True)
    agent_workspace = workspace / "agent"
    shutil.copytree(output / "prefix_files", agent_workspace)
    write_json(workspace / "active.json", {"task_id": task["id"], "arm": arm, "status": "replaying_prefix"})
    replay_matched = False
    try:
        with DiscoveryWorldLab(task, plan["runtime"], agent_workspace) as lab:
            replay = replay_prefix(lab, prefix)
            write_json(workspace / "replay.json", replay)
            replay_matched = True
            provider = CodexCodeProvider(agent_workspace, model=plan["model"]["requested_alias"],
                reasoning_effort=plan["model"]["reasoning_effort"],
                timeout_seconds=plan["limits"]["provider_request_timeout_seconds"])
            agent = DiscoveryWorldAgent(provider, agent_workspace, arm=arm, limits=plan["limits"])
            agent.observations = deepcopy(prefix["records"])
            agent.questions = deepcopy(prefix["research"]["questions"])
            agent.methods = deepcopy(prefix["research"].get("methods", []))
            agent.relation_research = deepcopy(prefix["research"].get("relation_research", {}))
            agent.research_advice = deepcopy(prefix["research"].get("research_advice"))
            # Share learned state, not the other arm's pending host decision.
            agent.research_focus = (deepcopy(prefix["research"].get("research_focus"))
                                    if arm == "hegel" else None)
            agent._prerequisite_checked_through = prefix["research"].get("prerequisite_checked_through", -1)
            for name in ("program_executions", "child_answers_returned"):
                setattr(agent, name, prefix["research"].get(name, 0))
            agent.method_results = deepcopy(prefix["method_results"])
            agent.files.update({name: (agent_workspace / name).read_text(encoding="utf-8")
                                for name in prefix["registered_files"]})
            # Keep all past decisions inspectable without counting old usage as
            # newly consumed model tokens in the pilot's shared summarizer.
            agent.events = [{**{k: deepcopy(v) for k, v in event.items() if k != "usage"},
                             "phase": "frozen_prefix"} for event in prefix["events"]]
            history = json.dumps({"events": agent.events, "prior_costs": prefix["prior_costs"]},
                                 ensure_ascii=False)
            agent.files["prefix_history.json"] = history
            (agent_workspace / "prefix_history.json").write_text(history, encoding="utf-8")
            (agent_workspace / "research_notes.md").write_text(prefix["notes"], encoding="utf-8")
            agent.costs["environment_steps"] = lab.environment_steps

            def checkpoint(result):
                result.update(task_id=task["id"], arm=arm, prefix_environment_steps=prefix["environment_steps"],
                              prefix_event_count=len(prefix["events"]), prior_costs=prefix["prior_costs"],
                              prefix_research_counts={name: prefix["research"].get(name, 0)
                                  for name in ("program_executions", "child_answers_returned")},
                              replay=deepcopy(replay))
                result["new_environment_steps"] = result["costs"]["environment_steps"] - prefix["environment_steps"]
                write_json(workspace / "active.json", result)

            result = agent.run(lab, on_checkpoint=checkpoint)
            checkpoint(result)
        write_json(workspace / "result.json", result)
        return result
    except Exception as exc:
        result = failed_result(task, arm, workspace, exc)
        if not replay_matched:
            result["status"] = "replay_failed_no_model_calls"
            write_json(workspace / "result.json", result)
        return result


def check_summary(plan, results):
    result = summarize(plan, results)
    result.update(disposition="development_integration_check_not_full_task_performance",
                  prefix_source=plan["prefix_source"], prefix_prior_costs=plan["prefix_prior_costs"],
                  paired_continuation_available=len(results) == 2 and all(
                      run.get("replay", {}).get("status") == "matched" for run in results))
    for arm, entry in result["arms"].items():
        runs = [run for run in results if run["arm"] == arm]
        events = [event for run in runs for event in run.get("events", [])[run.get("prefix_event_count", 0):]]
        entry["new_environment_steps"] = sum(run.get("new_environment_steps", 0) for run in runs)
        entry["new_research_calls"] = sum(event.get("arguments", {}).get("operator") == "research" for event in events)
        entry["new_evaluate_calls"] = sum(event["action"] == "evaluate" for event in events)
        for name in ("program_executions", "child_answers_returned"):
            entry["new_" + name] = sum(run.get("research", {}).get(name, 0)
                - run.get("prefix_research_counts", {}).get(name, 0) for run in runs)
        entry["new_child_questions"] = sum(question.get("parent_id") is not None
            and question["id"] not in plan["prefix_question_ids"] for run in runs
            for question in run.get("research", {}).get("questions", []))
        entry["numerical_programs"] = [question["answer"]["program"] for run in runs
            for question in run.get("research", {}).get("questions", [])
            if isinstance(question.get("answer"), dict) and question["answer"].get("program")]
    return result


def run_check(source, output, *, calls=12, workers=2, prefix_calls=None, method_bundle=None):
    output = Path(output).resolve()
    auth_mode = verify_codex_subscription()
    output.mkdir(parents=True, exist_ok=False)
    source_plan, prefix = freeze_prefix(source, output, prefix_calls=prefix_calls)
    if method_bundle:
        supplied = json.loads(Path(method_bundle).read_text())
        for name, content in supplied["files"].items():
            target = output / "prefix_files" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        prefix["registered_files"] = sorted(set(prefix["registered_files"]) | set(supplied["files"]))
        prefix["copied_files"] = sorted(set(prefix["copied_files"]) | set(supplied["files"]))
        # Both arms receive the same supplied programs and separate practice
        # archive. The archive never becomes a local observation or answer.
        supported = {q["id"] for q in prefix["research"]["questions"] if q["status"] == "supported"}
        for definition in supplied["methods"]:
            method = deepcopy(definition)
            method.setdefault("requires", [])
            method["available"] = set(method["requires"]) <= supported
            existing = prefix["research"].setdefault("methods", [])
            prefix["research"]["methods"] = [m for m in existing if m["id"] != method["id"]] + [method]
        prefix["supplied_method_bundle"] = str(Path(method_bundle).resolve())
        write_json(output / "prefix.json", prefix)
    plan = {**source_plan, "study": "discoveryworld_prerequisite_integration_check",
            "tasks": [prefix["task"]], "arms": ["code_archive", "hegel"],
            "jobs": [{"task_id": prefix["task"]["id"], "arm": arm} for arm in ["code_archive", "hegel"]],
            "limits": {**source_plan["limits"], "max_model_calls": calls}, "workers": workers,
            "auth_mode": auth_mode, "prefix_source": prefix["source"],
            "prefix_calls": prefix_calls, "source_checkpoint_costs": prefix["source_checkpoint_costs"],
            "prefix_cost_scope": prefix["cost_scope"],
            "prefix_environment_steps": prefix["environment_steps"], "prefix_prior_costs": prefix["prior_costs"],
            "prefix_question_ids": [question["id"] for question in prefix["research"]["questions"]],
            "supplied_methods": {"bundle": prefix.get("supplied_method_bundle"),
                "definitions": deepcopy(supplied["methods"]) if method_bundle else [],
                "scope": "Same operator-supplied programs and separate public practice for both arms; no local answers supplied."},
            "purpose": "Test host-scheduled prerequisite or child-analysis continuation from the same frozen "
                       "public state; hypotheses and analyses remain model-authored. "
                       "The call ceiling is for this development continuation, not full-task performance.",
            "replay_policy": "Each arm replays the frozen official actions and compares public success/messages "
                             "before its model calls. Replay failure stops that attempt; its paired comparison is unavailable.",
            "focus_policy": "Both arms inherit the same questions, answers, evidence and programs. Only Hegel "
                            "inherits its pending host research focus; Code chooses its own next action.",
            "accounting": "New model calls and tokens start at zero. Original environment ceiling includes "
                          "the prefix; prior costs and new environment steps are reported separately."}
    if plan["model"]["backend"] != "codex_cli":
        raise ValueError("The source must use the authorized subscription CLI provider")
    write_json(output / "protocol.json", plan)
    results = []
    write_json(output / "summary.json", check_summary(plan, results))
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(continue_arm, plan, arm, str(output)): arm for arm in plan["arms"]}
        for future in as_completed(pending):
            arm = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                workspace = output / prefix["task"]["id"] / arm
                workspace.mkdir(parents=True, exist_ok=True)
                result = failed_result(prefix["task"], arm, workspace, exc)
            results.append(result)
            write_json(output / "summary.json", check_summary(plan, results))
            print(json.dumps({"arm": arm, "status": result["status"], "finished": len(results)}), flush=True)
    return check_summary(plan, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calls", type=int, default=12)
    parser.add_argument("--prefix-calls", type=int, help="cut the source at this previously observed model call")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--method-bundle", type=Path, help="Shared authored method files and their separate public practice archive")
    args = parser.parse_args()
    if args.calls < 1 or args.workers < 1:
        parser.error("--calls and --workers must be positive")
    if args.output.exists():
        parser.error("--output already exists; use a new directory")
    print(json.dumps(run_check(args.source, args.output, calls=args.calls, workers=args.workers,
                               prefix_calls=args.prefix_calls, method_bundle=args.method_bundle), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the registered DiscoveryWorld development tasks with two independent agents."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from scripts.run_online_inquiry_llm import verify_codex_subscription

SELECTION = ROOT / "config/discoveryworld_hard_selection_20261007.json"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_plan(task_ids, arms, runtime, workers):
    selection = json.loads(SELECTION.read_text())
    development = [task for task in selection["tasks"] if task["split"] == "development"]
    if task_ids is not None:
        unknown = set(task_ids) - {task["id"] for task in development}
        if unknown or len(task_ids) != len(set(task_ids)):
            raise ValueError(f"select unique development task IDs only: {sorted(unknown)}")
        development = [task for task in development if task["id"] in task_ids]
    if len(arms) != len(set(arms)):
        raise ValueError("each arm can be selected only once")
    jobs = [{"task_id": task["id"], "arm": arm}
            for index, task in enumerate(development)
            for arm in arms[index % len(arms):] + arms[:index % len(arms)]]
    return {"study": "discoveryworld_development_pilot", "selection_file": str(SELECTION),
            "split": "development", "tasks": development, "arms": arms, "jobs": jobs,
            "model": selection["model"], "limits": selection["development_limits"],
            "runtime": str(Path(runtime).resolve()), "workers": workers,
            "failure_policy": "Keep every initial attempt. Completed replies with malformed arguments return "
                              "their error to the model within the original call budget; service failures stop.",
            "resume_policy": "New output directory required; interrupted attempts remain preserved.",
            "outcomes": selection["outcomes"], "fairness": selection["fairness"]}


def failed_result(task, arm, workspace, exc):
    checkpoint = workspace / "active.json"
    result = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    result.update(task_id=task["id"], arm=arm, status="runner_failed",
                  runner_error={"type": type(exc).__name__, "message": str(exc)})
    write_json(workspace / "result.json", result)
    return result


def run_task(plan, task, arm, output):
    workspace = Path(output) / task["id"] / arm
    workspace.mkdir(parents=True, exist_ok=True)
    write_json(workspace / "active.json", {"task_id": task["id"], "arm": arm, "status": "starting"})
    try:
        from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
        from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab

        # Evaluation checkpoints live outside the agent's working directory.
        # Public evidence and authored programs remain together under agent/.
        agent_workspace = workspace / "agent"
        provider = CodexCodeProvider(agent_workspace,
            model=plan["model"]["requested_alias"],
            reasoning_effort=plan["model"]["reasoning_effort"],
            timeout_seconds=plan["limits"]["provider_request_timeout_seconds"])
        agent = DiscoveryWorldAgent(provider, agent_workspace, arm=arm, limits=plan["limits"])
        installation = None
        with DiscoveryWorldLab(task, plan["runtime"], agent_workspace) as lab:
            if plan.get("method_bundle"):
                from hegel_machine.cognition.world_method_transfer import install_method_bundle

                bundle = json.loads(Path(plan["method_bundle"]).read_text())
                installation = install_method_bundle(agent, bundle, lab)
            result = agent.run(lab, on_checkpoint=lambda data: write_json(workspace / "active.json", data))
        result.update(task_id=task["id"], arm=arm)
        if installation is not None:
            result["method_bundle_installation"] = installation
        write_json(workspace / "result.json", result)
        return result
    except Exception as exc:
        return failed_result(task, arm, workspace, exc)


def summarize(plan, results):
    summary = {"study": plan["study"], "model": plan["model"],
               "registered_runs": len(plan["jobs"]), "terminal_runs": len(results),
               "status": "finished" if len(results) == len(plan["jobs"]) else "running",
               "arms": {}}
    for arm in plan["arms"]:
        runs = [result for result in results if result["arm"] == arm]
        scores = [result["score"] for result in runs if result.get("score") is not None]
        usage = [event["usage"] for result in runs for event in result.get("events", []) if "usage" in event]
        research = [result.get("research", {}) for result in runs]
        aggregate = {"registered_tasks": len(plan["tasks"]), "terminal_tasks": len(runs),
                     "pending_tasks": len(plan["tasks"]) - len(runs),
                     "statuses": dict(Counter(result["status"] for result in runs)),
                     "scored_tasks": len(scores),
                     "successful_tasks": sum(score["completedSuccessfully"] for score in scores),
                     "observed_score_mean": sum(score["scoreNormalized"] for score in scores) / len(scores) if scores else None,
                     "model_calls": sum(result.get("costs", {}).get("model_calls", 0) for result in runs),
                     "questions": sum(len(item.get("questions", [])) for item in research),
                     "methods": sum(len(item.get("methods", [])) for item in research),
                     "child_answers_returned": sum(item.get("child_answers_returned", 0) for item in research),
                     "program_executions": sum(item.get("program_executions", 0) for item in research),
                     "numerical_questions": sum(len(item.get("relation_research", {})) for item in research),
                     "confirmed_numerical_questions": sum(
                         relation.get("status") == "supported" for item in research
                         for relation in item.get("relation_research", {}).values()),
                     "calls_without_token_usage": sum(not isinstance(item.get("total_tokens"), int) for item in usage),
                     "calls_without_cache_usage": sum(not isinstance(item.get("cached_input_tokens"), int) for item in usage),
                     "cache_hit_calls": sum(isinstance(item.get("cached_input_tokens"), int) and item["cached_input_tokens"] > 0 for item in usage)}
        aggregate["costs"] = {
            field: sum(result.get("costs", {}).get(field, 0) for result in runs)
            if all(isinstance(result.get("costs", {}).get(field), (int, float)) for result in runs) else None
            for field in ("environment_steps", "python_execution_seconds", "wall_time_seconds", "research_compute_units")}
        for field in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            aggregate[field + "_known_lower_bound"] = sum(item[field] for item in usage if isinstance(item.get(field), int))
        summary["arms"][arm] = aggregate
    return summary


def run_study(plan, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if plan.get("method_bundle"):
        bundle = json.loads(Path(plan["method_bundle"]).read_text())
        bundle_path = output / "method_bundle.json"
        write_json(bundle_path, bundle)
        plan = {**plan, "method_bundle": str(bundle_path)}
    write_json(output / "protocol.json", plan)
    results = []
    write_json(output / "summary.json", summarize(plan, results))
    tasks = {task["id"]: task for task in plan["tasks"]}
    with ProcessPoolExecutor(max_workers=plan["workers"],
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(run_task, plan, tasks[job["task_id"]], job["arm"], str(output)): job
                   for job in plan["jobs"]}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                workspace = output / job["task_id"] / job["arm"]
                workspace.mkdir(parents=True, exist_ok=True)
                result = failed_result(tasks[job["task_id"]], job["arm"], workspace, exc)
            results.append(result)
            write_json(output / "summary.json", summarize(plan, results))
            print(json.dumps({"task_id": job["task_id"], "arm": job["arm"],
                              "status": result["status"], "finished": len(results),
                              "registered": len(plan["jobs"])}), flush=True)
    return summarize(plan, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory; existing attempts are never overwritten")
    parser.add_argument("--tasks", nargs="+", help="development task IDs; default all 16")
    parser.add_argument("--arms", nargs="+", choices=("code_archive", "hegel"), default=["code_archive", "hegel"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--runtime", type=Path, default=Path("/home/erzhu419/.cache/hegel-machine/discoveryworld"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory to preserve previous attempts")
    try:
        plan = make_plan(args.tasks, args.arms, args.runtime, args.workers)
        if plan["model"]["backend"] != "codex_cli":
            raise ValueError("this pilot uses the registered Codex CLI provider")
        plan["auth_mode"] = verify_codex_subscription()
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(run_study(plan, args.output), indent=2))


if __name__ == "__main__":
    main()

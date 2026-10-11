#!/usr/bin/env python3
"""Exercise automatic local research after a real source-program failure.

The host selects the saved source trial; the ordinary procedure executor selects
whether its new execution needs repair. This is a source integration diagnostic,
not evidence of a learned delegation policy or held-out transfer.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.discoveryworld_agent import DiscoveryWorldAgent
from hegel_machine.cognition.discoveryworld_lab import DiscoveryWorldLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from scripts.run_discoveryworld_pilot import write_json
from scripts.run_discoveryworld_prerequisite_check import replay_prefix
from scripts.run_online_inquiry_llm import verify_codex_subscription


def restore_repair(agent, laboratory, previous, previous_workspace):
    """Resume the saved world, research and original paid account together."""
    previous_workspace = Path(previous_workspace)
    records = json.loads((previous_workspace / "laboratory_observations.json").read_text())
    replay = replay_prefix(laboratory, {"records": records,
        "environment_steps": previous["costs"]["environment_steps"]})
    agent.observations = deepcopy(records)
    for path in (previous_workspace / "shared").rglob("*"):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            agent._execute("write", {"path": str(path.relative_to(previous_workspace)),
                "code": path.read_text()}, laboratory)
    notes = previous_workspace / "research_notes.md"
    if notes.exists():
        (agent.workspace / notes.name).write_text(notes.read_text())
    research = previous["research"]
    for name, default in (("questions", []), ("methods", []), ("relation_research", {}),
                          ("research_advice", None), ("research_focus", None),
                          ("llm_inquiries", {}), ("program_executions", 0),
                          ("child_answers_returned", 0)):
        setattr(agent, name, deepcopy(research.get(name, default)))
    agent._prerequisite_checked_through = research.get("prerequisite_checked_through", -1)
    agent.method_results = deepcopy(previous["method_results"])
    agent.events = deepcopy(previous["events"])
    agent.score_checkpoints = deepcopy(previous.get("score_checkpoints", []))
    agent.lesson = previous.get("lesson", "")
    agent.costs = deepcopy(previous["costs"])
    agent.started -= agent.costs["wall_time_seconds"]
    agent.status = previous["episode_status"]
    agent._save_state()
    return replay


def run(trial, output, calls=None, resume=None):
    previous = previous_protocol = None
    if resume:
        resume = Path(resume).resolve()
        previous = json.loads(resume.read_text())
        previous_protocol = json.loads((resume.parent / "protocol.json").read_text())
        trial = previous_protocol["trial"]
        if calls is not None and calls != previous_protocol["limits"]["max_model_calls"]:
            raise ValueError("Resume retains the original total model-call allowance")
    trial, output = Path(trial).resolve(), Path(output).resolve()
    learning = json.loads((trial.parents[2] / "protocol.json").read_text())
    source = Path(learning["source"])
    source_plan = json.loads((source.parents[1] / "protocol.json").read_text())
    task = next(item for item in source_plan["tasks"] if item["id"] == source.parent.name)
    failed = json.loads((trial / "result.json").read_text())
    if failed["score"]["completed"]:
        raise ValueError("Select an unfinished real source practice")
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "agent"
    limits = (deepcopy(previous_protocol["limits"]) if previous_protocol else
              {**source_plan["limits"], "max_model_calls": 32 if calls is None else calls})
    procedure_arguments = (deepcopy(previous_protocol.get("procedure_arguments") or
        previous["procedure"]["repair"]["trigger"]["query_arguments"]) if previous else
        {"parameters": {}, "procedure_start": len(failed["paid_observations"]),
         "research_question": {"id": "task"}})
    protocol = {"trial": str(trial), "task": task, "model": source_plan["model"],
                "limits": limits, "auth_mode": verify_codex_subscription(),
                "procedure_arguments": procedure_arguments,
                "scope": "Replay an operator-selected source failure and run its unchanged procedure. "
                         "The production executor triggers local research from an actual execution "
                         "error or repeated public control cycle. This source diagnostic does not "
                         "establish a learned delegation policy or held-out transfer."}
    if resume:
        protocol["resume"] = {"result": str(resume), "costs": previous["costs"],
            "scope": "Restore the supported child, edited source and original cumulative account; "
                     "continue the same parent invocation without repeating source learning."}
    write_json(output / "protocol.json", protocol)
    provider = CodexCodeProvider(workspace, model=source_plan["model"]["requested_alias"],
        reasoning_effort=source_plan["model"]["reasoning_effort"],
        timeout_seconds=limits["provider_request_timeout_seconds"])
    agent = DiscoveryWorldAgent(provider, workspace, arm="hegel", limits=limits)
    with DiscoveryWorldLab(task, source_plan["runtime"], workspace) as lab:
        agent.lab = lab
        if previous:
            replay = restore_repair(agent, lab, previous, resume.parent / "agent")
            write_json(output / "replay.json", replay)
        else:
            for record in failed["paid_observations"]:
                observed = lab.act(record["action"])
                if (observed["feedback"].get("success"), observed["observation"].get("lastActionMessage")) != (
                        record["feedback"].get("success"), record["observation"].get("lastActionMessage")):
                    raise ValueError(f"Public replay differs at record {record['index']}")
                agent.observations.append({"index": len(agent.observations), **observed})
            agent._execute("write", {"path": METHOD_PATH,
                "code": (trial / METHOD_PATH).read_text()}, lab)
            agent.method_results = deepcopy(failed["method_results"])
            root = agent._question({"id": "task", "question": "\n".join(
                row.get("description", row.get("taskName", ""))
                for row in lab.public_observation().get("taskProgress", []))})
            root["origin"] = "public_task"
        prefix_steps = lab.environment_steps
        agent.costs["environment_steps"] = prefix_steps
        costs_before = agent.result()["costs"]
        agent._on_checkpoint = lambda data: write_json(output / "active.json", {
            **data, "study_status": "running", "costs_before": costs_before,
            "replayed_environment_steps": prefix_steps,
            "new_environment_steps": lab.environment_steps - prefix_steps})
        agent._on_checkpoint(agent.result())
        result = {"operator": "procedure", **agent._run_procedure(lab, METHOD_PATH,
            max_steps=12, query_arguments=procedure_arguments)}
        final = agent.result()
        write_json(output / "result.json", {**final, "study_status": "finished",
            "episode_status": final["status"], "procedure": result,
            "costs_before": costs_before,
            "new_costs": {key: value - costs_before[key]
                          for key, value in final["costs"].items()},
            "replayed_environment_steps": prefix_steps,
            "new_environment_steps": lab.environment_steps - prefix_steps})
        repair = result.get("repair", {})
        inquiry = repair.get("outcome", {})
        continuation = result.get("continuation")
        repair_chain, execution = [], result
        while execution.get("repair"):
            item = execution["repair"]
            repair_chain.append({"question_id": item["question_id"],
                "trigger": item["trigger"]["kind"],
                "model_calls": item["outcome"]["model_calls"],
                "question_status": item["outcome"]["question"]["status"],
                "continuation_status": execution.get("continuation", {}).get("status")})
            execution = execution.get("continuation", {})
        brief = {"study_status": "finished", "episode_status": final["status"],
            "procedure_status": result["status"],
            "trigger": repair.get("trigger", {}).get("kind"),
            "child_model_calls": inquiry.get("model_calls", 0),
            "child_question": inquiry.get("question"),
            "parent_continuation": None if continuation is None else {
                "status": continuation["status"], "steps": len(continuation["steps"])},
            "model_calls": agent.costs["model_calls"],
            "new_model_calls": agent.costs["model_calls"] - costs_before["model_calls"],
            "repair_chain": repair_chain,
            "replayed_environment_steps": prefix_steps,
            "new_environment_steps": lab.environment_steps - prefix_steps}
        write_json(output / "brief.json", brief)
    return brief


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--trial", type=Path)
    start.add_argument("--resume", type=Path, help="Saved result; retain its research and total call allowance")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calls", type=int, help="Total call allowance; defaults to 32 or the original limit on resume")
    args = parser.parse_args()
    print(json.dumps(run(args.trial, args.output, args.calls, args.resume)))

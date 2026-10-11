#!/usr/bin/env python3
"""Matched development rescue from one unchanged, failed online investigation."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from scripts.run_anonymous_online_research import (
    RESEARCH_TASK, prepare_research_continuation, representation_laboratory,
)
from scripts.run_anonymous_revision_inquiry import run_study
from scripts.run_online_inquiry_llm import verify_codex_subscription, write_json


def rescue_protocol(source_workspace):
    source = Path(source_workspace).resolve()
    previous = json.loads((source / "training.json").read_text())
    original = json.loads((source.parent.parent / "protocol.json").read_text())
    if previous["model"] is not None:
        raise ValueError("this rescue starts from an unresolved investigation")
    case = deepcopy(next(case for case in original["cases"] if case["id"] == source.parent.name))
    case["arms"] = ["code_react", "hegel"]
    costs = previous["costs"]
    plan = deepcopy(original)
    plan.update(study="observable_dependency_matched_rescue", model="gpt-6-luna",
        arms=case["arms"], cases=[case], source_workspace=str(source),
        source_actor=previous["agent_arm"], prior_research_cost=deepcopy(costs),
        limits={"max_model_calls": costs["model_calls"] + 24,
                "token_budget": costs["reported_tokens"] + 1_600_000,
                "execution_seconds": costs["python_execution_seconds"] + 120.,
                "compute_budget": costs["compute_cost"] + 25_000,
                "observation_budget": 576, "confirmation_rows": 96},
        provider={"backend": "codex_cli", "reasoning_effort": "medium", "timeout_seconds": 300.},
        shared_research_task=RESEARCH_TASK,
        controls="Both branches restore the same original actor's code, paid observations, "
            "events, notebook and research history. Both have identical observable-target "
            "evaluation, retained dependencies, Python and experiment tools. Hegel additionally "
            "organizes an actor-selected observable target as a child research result returned "
            "to its parent. Code retains readable history without a mandatory question focus. "
            "Continuation does not require inventing an experimental program for submission.",
        prefix_protocol="Restore the complete saved paid account; buy no prefix and advance "
            "the original confirmation random stream past its actual paid confirmations.",
        archive_protocol="Restore original session_knowledge and all actual source files. "
            "Neither branch receives the other rescue's work or host analysis of the failure.",
        continuation_protocol="Two rescue branches from one already examined failed development "
            "trajectory. Each receives 24 additional model calls, 1600000 reported tokens, "
            "120 Python seconds and 25000 scoring units; total observation allowance stays 576. "
            "No fitted solution, successful source step, scientific formula or new research "
            "direction is supplied by the host.",
        cost_scope="Each costs record includes the same prior investigation. Count that origin "
            "once; continuation_cost measures newly incurred work separately for each branch.",
        outcome_protocol="Record actor-selected target evaluation, supported dependency retention, "
            "parent return, actual parent predictor use, fresh confirmation and terminal error. "
            "Automatically organized questions alone are not autonomous decomposition evidence.",
        scope="One failed development trajectory reused in two matched rescue branches; not two "
            "independent tasks, a new autonomous discovery cohort or an estimate of broad superiority.")
    for key in ("resume_study", "original_limits", "independent_parameter_pairs"):
        plan.pop(key, None)
    plan["runtime_snapshot"] = sorted(set(original["runtime_snapshot"]) | {
        str(path.relative_to(ROOT)) for path in (ROOT / "src/hegel_machine").rglob("*.py")
    } | {"src/hegel_machine/cognition/observable_research.py", str(Path(__file__).relative_to(ROOT)),
         "scripts/run_online_inquiry_llm.py"})
    return plan


def prepare_rescue(plan):
    source = Path(plan["source_workspace"])
    previous = json.loads((source / "training.json").read_text())

    def prepare(agent, laboratory, case):
        agent.common_interface += plan["shared_research_task"]
        agent.restore_research(previous, source, laboratory,
                               confirmation_seed=case["confirmation_seed"])
        agent.source_task = False
        if agent.arm == "code_react":
            agent._active_question_id = None
        return {"prior_research_cost": deepcopy(previous["costs"]),
                "resume_origin": {"training": str(source / "training.json"),
                    "actor": previous["agent_arm"], "status": previous["status"],
                    "shared_by_both_rescue_branches": True}}

    return prepare


def rescue_continuation_protocol(source):
    """Finish the already allocated calls in every token-truncated rescue arm."""
    source = Path(source).resolve()
    plan = json.loads((source / "protocol.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if plan["study"] != "observable_dependency_matched_rescue":
        raise ValueError("resume-study requires an observable-dependency rescue")
    expected = {(case["id"], arm) for case in plan["cases"]
                for arm in case.get("arms", plan["arms"])}
    actual = {(trial["case"], trial["arm"]) for trial in summary["trials"]}
    if summary["status"] != "completed" or actual != expected:
        raise ValueError("wait for all registered rescue branches and terminal evaluations to finish")
    selected = set()
    for trial in summary["trials"]:
        previous = json.loads((source / trial["case"] / trial["arm"] / "training.json").read_text())
        if previous["status"] != trial["status"]:
            raise ValueError("saved training and the completed rescue summary disagree")
        if previous["status"] == "token_budget_exhausted":
            selected.add((trial["case"], trial["arm"]))
    if not selected:
        raise ValueError("the completed rescue has no token-truncated branches")
    if plan["limits"]["token_budget"] >= 10_000_000:
        raise ValueError("the ten-million cumulative token allowance is already allocated")
    cases = []
    for case in plan["cases"]:
        arms = [arm for arm in case.get("arms", plan["arms"])
                if (case["id"], arm) in selected]
        if arms:
            cases.append({**deepcopy(case), "arms": arms})
    plan.update(cases=cases, resume_study=str(source), original_limits=deepcopy(plan["limits"]),
        previous_runtime_snapshot=str(source / "source_at_run"),
        continuation_protocol="After all original rescue branches stopped, continue every and only "
            "token-truncated branch from its own saved research. Raise cumulative reported tokens "
            "to 10000000; preserve every other original total allowance, including model calls. "
            "Keep each branch's actual code, paid evidence, notebook and confirmation stream. "
            "No terminal labels, opposite-arm work or new scientific instructions enter research.",
        cost_scope="costs retains cumulative research from the one common historical origin. "
            "prior_research_cost is this branch's own previous cumulative account; continuation_cost "
            "and continuation cache counters include only this added phase. Earlier outcomes and "
            "their runtime remain in resume_study; count the common historical origin once.")
    plan["limits"]["token_budget"] = 10_000_000
    plan["runtime_snapshot"] = sorted(set(plan["runtime_snapshot"]) | {
        str(path.relative_to(ROOT)) for path in (ROOT / "src/hegel_machine").rglob("*.py")
    })
    return plan


def run_rescue(plan, output, *, provider_factory):
    output = Path(output).resolve()
    if (output / "protocol.json").exists():
        raise ValueError("use a fresh output directory to preserve earlier results")
    output.mkdir(parents=True, exist_ok=True)
    if "resume_study" in plan:
        source = Path(plan["resume_study"])
        knowledge = json.loads((source / "initial_archive.json").read_text())
        prepare = prepare_research_continuation(plan)
        prior_event_counts = {(case["id"], arm): len(json.loads(
            (source / case["id"] / arm / "training.json").read_text())["events"])
            for case in plan["cases"] for arm in case.get("arms", plan["arms"])}
        write_json(output / "previous_summary.json", json.loads((source / "summary.json").read_text()))
    else:
        source = Path(plan["source_workspace"])
        knowledge = json.loads((source / "research_context.json").read_text())["session_knowledge"]
        prepare = prepare_rescue(plan)
        count = len(json.loads((source / "training.json").read_text())["events"])
        prior_event_counts = {(case["id"], arm): count for case in plan["cases"]
                              for arm in case.get("arms", plan["arms"])}
    for name in plan["runtime_snapshot"]:
        destination = output / "source_at_run" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    summary = run_study(plan, output, provider_factory=provider_factory,
        knowledge=knowledge, prepare_prefix=prepare,
        laboratory_factory=representation_laboratory)
    summary.update(shared_origin=plan["source_workspace"], independent_source_count=1,
                   cost_scope=plan["cost_scope"], outcome_protocol=plan["outcome_protocol"])
    for trial in summary["trials"]:
        result = json.loads((output / trial["case"] / trial["arm"] / "training.json").read_text())
        added = result["events"][prior_event_counts[trial["case"], trial["arm"]]:]
        usage = [event.get("usage", {}) for event in added]
        trial["continuation_cached_input_tokens"] = sum(u.get("cached_input_tokens") or 0 for u in usage)
        trial["continuation_calls_with_cache_hits"] = sum(bool(u.get("cached_input_tokens")) for u in usage)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--source-workspace", type=Path)
    origin.add_argument("--resume-study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = (rescue_continuation_protocol(args.resume_study) if args.resume_study
            else rescue_protocol(args.source_workspace))
    plan["provider"]["auth_mode"] = verify_codex_subscription()
    summary = run_rescue(plan, args.output, provider_factory=lambda workspace:
        CodexCodeProvider(workspace, model=plan["model"], reasoning_effort="medium", timeout_seconds=300.))
    print(json.dumps({"status": summary["status"], "trials": len(summary["trials"])}), flush=True)


if __name__ == "__main__":
    main()

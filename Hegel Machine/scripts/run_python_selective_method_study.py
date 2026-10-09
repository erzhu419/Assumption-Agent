#!/usr/bin/env python3
"""Preserve a learned refit branch while learning recovery from fresh rejection."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.python_selective_method_learning import SelectivePythonMethodLearner
from scripts.python_selective_method_cases import (
    OLD_AUTHOR_RESULT, PUBLISHED_AUTHOR_RESULT, SOURCE_OUTCOMES, KINDS, protocol,
)
from scripts.run_python_method_learning_study import (
    DEFAULT_RESEARCH_PROCEDURE, InquiryLLMConfig, METHOD_PATH, archive_memory,
    LIMIT_COSTS,
    execution_evidence, laboratory, score_episode, selected_candidate, total_costs,
    trial_preparer, usage_summary, verify_codex_subscription, write_json,
)
from scripts.run_temporal_counterexample_learning import supply_counterexample

CONDITIONS = ("seed_selective", "prior_published", "revised", "fixed")


def study_protocol(sources, diagnostic):
    plan = protocol(sources)
    plan.update(conditions=list(CONDITIONS), model="gpt-6-luna",
        provider={"backend": "codex_cli", "reasoning_effort": "high", "timeout_seconds": 180},
        author_limits=asdict(InquiryLLMConfig(max_model_calls=12, token_budget=500000,
            observation_budget=3000, compute_budget=3000000,
            execution_seconds=240., confirmation_rows=0)),
        seed_method=str(OLD_AUTHOR_RESULT), previous_published_method=str(PUBLISHED_AUTHOR_RESULT),
        diagnostic=str(Path(diagnostic).resolve()),
        learning="Start from the exact earlier selective source, retaining its working branches. "
            "Supply its actual paid case_3 failure using the existing counterexample interface. "
            "The author chooses all source edits and practice. No repaired branch is supplied.",
        publication="All four current-source cases must confirm, with unchanged-prior reuse and "
            "revision. Also demonstrate zero-measurement refit confirmation, and recovery after "
            "a validation-supported refit fails fresh confirmation, in different cases.",
        transfer="Freeze the published revision and run all eight registered targets against the "
            "exact earlier selective source, previous published source and developer controller. "
            "All four arms share priors, prefixes, actual laboratories and per-episode budgets. "
            "Transfer uses zero model calls. Without publication, keep revised slots skipped.",
        failure_policy="Preserve the first author phase and every paid practice. If publication "
            "fails, skip eight revised slots and run all 24 registered reference episodes. "
            "No unpublished source is substituted for the revised arm.",
        accounting="Earlier source acquisition, author phases and replay costs remain separate. "
            "The supplied diagnostic is charged once. New authoring and all four target arms "
            "are charged at their actual costs; unknown token usage remains unknown.")
    return plan


def target_jobs(sources, output, plan, learned, methods):
    jobs, skipped = [], []
    published = learned["status"] == "completed" and learned.get("method_source") is not None
    conditions = tuple(plan["conditions"])
    for index, spec in enumerate(plan["transfer_cases"]):
        order = conditions[index % len(conditions):] + conditions[:index % len(conditions)]
        for condition in order:
            if condition == "revised" and not published:
                skipped.append({"target_id": spec["id"], "condition": condition,
                    "reason": "method_not_published"})
                continue
            jobs.append({"spec": spec, "condition": condition,
                "source": sources[spec["source_index"]],
                "path": output / f"{spec['id']}_{condition}",
                "method_source": learned["method_source"] if condition == "revised" else methods[condition]})
    return jobs, skipped


def aggregate(results):
    return {"attempted_targets": len(results),
        "confirmed_and_correct": sum(item["confirmed_and_correct"] for item in results),
        "false_confirmations": sum(item["false_confirmation"] for item in results),
        "unchanged_prior_selected": sum(item["unchanged_prior_selected"] for item in results),
        "statuses": dict(Counter(item["status"] for item in results)),
        "costs": total_costs([item["costs"] for item in results])}


def interrupted_state(path, *, additional_development=False):
    """Restore paid work after service interruption or declared extra development."""
    path = Path(path).resolve()
    previous = json.loads(path.read_text())
    allowed = {"model_call_budget_exhausted", "token_budget_exhausted"} if additional_development else {"provider_failed"}
    if previous["status"] not in allowed:
        if additional_development:
            raise ValueError("additional development requires a budget-exhausted author")
        raise ValueError("resume requires an actual provider-failed author")
    prior_root = path.parent.parent
    plan = json.loads((prior_root / "protocol.json").read_text())
    summary = json.loads((prior_root / "summary.json").read_text())
    original_limits = deepcopy(plan["author_limits"])
    for limit, cost in LIMIT_COSTS:
        plan["author_limits"][limit] = max(0, original_limits[limit] - previous["costs"][cost])
    plan["author_limits"]["token_budget"] = max(0,
        original_limits["token_budget"] - usage_summary(previous)["known_tokens"])
    if additional_development:
        plan["author_limits"]["max_model_calls"] += 6
        plan["author_limits"]["token_budget"] += 200000
    phase = "continue_author" if additional_development else "resume_author"
    plan.pop("resume_author", None)
    plan.pop("continue_author", None)
    plan[phase] = {"result": str(path), "previous_status": previous["status"],
        "previous_costs": previous["costs"], "remaining_limits": deepcopy(plan["author_limits"]),
        "added_allowance": {"model_calls": 6, "tokens": 200000} if additional_development else None,
        "scope": "Preserve actual source and paid trials. Unknown usage remains unknown; remaining "
            "token allowance subtracts reported usage only. Completed reference episodes are reused "
            "by the evaluator and excluded from author context; no reference acquisition is repeated."}
    if additional_development:
        plan[phase]["author_guidance"] = ("Your current source was revised after the last failed "
            "practice and has not yet been tested. Complete its paid validation and publication; "
            "use actual practice feedback to decide whether further source revision is needed.")
    return plan, previous, summary


def run_study(output, diagnostic, *, provider_factory=None, auth_mode=None, resume_author=None, continue_author=None):
    output, diagnostic = Path(output).resolve(), Path(diagnostic).resolve()
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    plan = study_protocol(sources, diagnostic)
    previous = prior_phase = None
    if resume_author is not None and continue_author is not None:
        raise ValueError("choose service resume or extra development")
    prior_result = resume_author or continue_author
    if prior_result is not None:
        plan, previous, prior_phase = interrupted_state(prior_result,
            additional_development=continue_author is not None)
    diagnostic_plan = json.loads((diagnostic / "protocol.json").read_text())
    if diagnostic_plan["diagnostic_case"] != plan["practice_cases"]["case_3"]:
        raise ValueError("the supplied paid failure must belong to this exact practice case_3")
    diagnostic_summary = json.loads((diagnostic / "summary.json").read_text())
    failed = json.loads((diagnostic / "selective/training.json").read_text())
    if failed["status"] == "completed":
        raise ValueError("the proposed counterexample did not reproduce a research failure")
    old = json.loads(OLD_AUTHOR_RESULT.read_text())
    published = json.loads(PUBLISHED_AUTHOR_RESULT.read_text())
    methods = {"seed_selective": old["outgoing_knowledge"]["files"][METHOD_PATH],
        "prior_published": published["method_source"], "fixed": DEFAULT_RESEARCH_PROCEDURE}
    output.mkdir(parents=True, exist_ok=False)
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "scripts", output / "source_at_run/scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = sources[plan["author_source_index"]]
    memory = archive_memory(source)
    memory["files"][METHOD_PATH] = methods["seed_selective"]
    author_path = output / "learning"
    provider = provider_factory(author_path) if provider_factory else CodexCodeProvider(author_path,
        model=plan["model"], reasoning_effort=plan["provider"]["reasoning_effort"], timeout_seconds=180)
    learner = SelectivePythonMethodLearner(provider, author_path, knowledge=memory,
        cases=plan["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]), prepare_trial=trial_preparer(source, plan))
    if previous is not None:
        write_json(output / "restored_practice.json", learner.restore_practice(previous))
    if continue_author is not None:
        learner.common_interface += "\n" + plan["continue_author"]["author_guidance"] + "\n"
    failure_path = output / "paid_counterexample.json"
    write_json(failure_path, {"tasks": [failed]})
    supply_counterexample(learner, failure_path, case_label="case_3")

    def checkpoint(result):
        write_json(author_path / "active.json", result)
        event = result["events"][-1] if result["events"] else {}
        feedback = event.get("feedback", {})
        print(json.dumps({"phase": "author", "calls": result["costs"]["model_calls"],
            "action": event.get("action"), "case": event.get("arguments", {}).get("case"),
            "status": feedback.get("status", result["status"]),
            "coverage": feedback.get("current_source_coverage")}), flush=True)

    try:
        learned = learner.run(on_checkpoint=checkpoint)
    except Exception:
        learned = learner.result("execution_failed")
        learned["execution_error"] = traceback.format_exc()
    write_json(author_path / "result.json", learned)
    jobs, skipped = target_jobs(sources, output, plan, learned, methods)
    reused = []
    if prior_phase is not None:
        reused = [item for item in prior_phase["results"] if item["condition"] != "revised"]
        completed_references = {(item["target_id"], item["condition"]) for item in reused}
        jobs = [job for job in jobs if (job["spec"]["id"], job["condition"]) not in completed_references]

    def train(job):
        payload = trial_preparer(job["source"], plan)(job["spec"], job["path"],
            job["method_source"], InquiryLLMConfig(**plan["limits"]))
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"])
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        write_json(job["path"] / "training.json", payload["training"])
        return job, payload

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    results = list(reused)
    executed = []
    for job, payload in trained:
        result = score_episode(payload, plan)
        result.update(target_id=job["spec"]["id"], kind=job["spec"]["kind"], condition=job["condition"],
            workspace=str(payload["path"]), unchanged_prior_selected=selected_candidate(payload) is not None,
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(job["path"] / "evaluation.json", result)
        results.append(result)
        executed.append(result)
    conditions = {condition: {"registered_targets": len(plan["transfer_cases"]),
        **aggregate([item for item in results if item["condition"] == condition]),
        "by_kind": {kind: aggregate([item for item in results
            if item["condition"] == condition and item["kind"] == kind]) for kind in KINDS}}
        for condition in CONDITIONS}
    usage = usage_summary(learned)
    prior = json.loads((PUBLISHED_AUTHOR_RESULT.parent.parent / "analysis.json").read_text())
    accounting_parent = prior_phase or prior
    previous_development = (prior_phase["all_method_development_costs"] if prior_phase is not None
        else prior["accounting"]["all_method_learning_development_including_replay"])
    new_diagnostic = total_costs([]) if prior_phase is not None else diagnostic_summary["costs"]
    new_costs = total_costs([learned["costs"], *[item["costs"] for item in executed]])
    cumulative_usage = {key: value + accounting_parent["cumulative_author_usage"][key] for key, value in usage.items()}
    summary = {"study": plan["study"], "author_status": learned["status"],
        "method_published": learned["status"] == "completed" and learned.get("method_source") is not None,
        "author_costs": learned["costs"], "author_usage": usage,
        "cumulative_author_costs": total_costs([accounting_parent["cumulative_author_costs"], learned["costs"]]),
        "cumulative_author_usage": cumulative_usage, "diagnostic_costs": diagnostic_summary["costs"],
        "resumed_from": str(Path(prior_result).resolve()) if prior_result else None,
        "additional_development": continue_author is not None,
        "reused_reference_episodes": len(reused), "new_diagnostic_costs": new_diagnostic,
        "new_author_and_transfer_costs": new_costs,
        "new_work_including_diagnostic": total_costs([new_diagnostic, new_costs]),
        "prior_method_development_costs": previous_development,
        "all_method_development_costs": total_costs([previous_development, new_diagnostic, new_costs]),
        "source_acquisition_costs": prior["accounting"]["separate_existing_source_acquisition"],
        "conditions": conditions, "results": results, "skipped": skipped, "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume-author", type=Path)
    continuation.add_argument("--continue-author", type=Path)
    args = parser.parse_args()
    summary = run_study(args.output, args.diagnostic, auth_mode=verify_codex_subscription(),
        resume_author=args.resume_author, continue_author=args.continue_author)
    print(json.dumps({key: summary[key] for key in ("author_status", "method_published",
        "author_costs", "conditions", "new_work_including_diagnostic")}, indent=2))

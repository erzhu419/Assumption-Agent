#!/usr/bin/env python3
"""Learn when to refit current evidence and when to replace an instrument model."""

import argparse
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
from hegel_machine.cognition.python_scoped_method_learning import ScopedPythonMethodLearner
from scripts.python_scoped_method_cases import protocol
from scripts.python_selective_method_cases import SOURCE_OUTCOMES
from scripts.run_python_method_learning_study import (
    DEFAULT_RESEARCH_PROCEDURE, InquiryLLMConfig, METHOD_PATH, archive_memory,
    execution_evidence, laboratory, score_episode, selected_candidate, total_costs,
    trial_preparer, usage_summary, verify_codex_subscription, write_json,
)
from scripts.run_python_selective_method_study import aggregate, interrupted_state, target_jobs
from scripts.run_temporal_counterexample_learning import supply_counterexample

PREVIOUS = ROOT / "artifacts/python_selective_method_completion_luna_20261006"
CONDITIONS = ("prior_published", "revised", "fixed")


def study_protocol(sources):
    plan = protocol(sources)
    plan.update(study="python_scoped_evidence_method_development", conditions=list(CONDITIONS),
        model="gpt-6-luna", seed_method=str(PREVIOUS / "learning/result.json"),
        provider={"backend": "codex_cli", "reasoning_effort": "high", "timeout_seconds": 180},
        author_limits=asdict(InquiryLLMConfig(max_model_calls=16, token_budget=650000,
            observation_budget=4800, compute_budget=3000000,
            execution_seconds=300., confirmation_rows=0)),
        learning="Start from the exact last published method and its actual paid case_3 trace. "
            "Preserve its working branches. The author chooses edits, evidence scope and practice. "
            "Only research source and lesson transfer, not practice observations or fitted models.",
        publication="All five current-source cases must confirm with prior reuse and revision. "
            "After a validated refit fails fresh confirmation, demonstrate zero-meter recovery "
            "using a separately validated refit of current records; in a different case demonstrate "
            "measured instrument/response repair and fresh confirmation.",
        author_guidance="Inspect the inherited source and paid trace. Improve its choice between "
            "refitting current evidence and repairing the representation after fresh rejection. "
            "Preserve working branches. After a substantive edit, use finish to request paid "
            "validation of the current source; revise from actual feedback before further edits.",
        transfer="Freeze the published source for ten registered cross-source targets. Compare "
            "the previous published source and developer controller at identical episode limits. "
            "Reuse their sixteen unchanged target outcomes; acquire only the new structural-change "
            "reference episodes. No reference outcomes enter author context. Transfer uses zero LLM calls.",
        failure_policy="Retain all author attempts. Without actual publication skip all ten revised "
            "slots and keep reference outcomes. Do not substitute unpublished source.",
        accounting="Charge new authoring and executed targets once. Prior authoring, supplied paid "
            "trace and reused references retain their historical costs without another charge.")
    return plan


def reusable_references(plan, previous_plan, previous_summary):
    """Original tasks are unchanged; the old revised arm becomes this study's prior."""
    old_cases = {spec["id"]: spec for spec in previous_plan["transfer_cases"]}
    new_cases = {spec["id"]: spec for spec in plan["transfer_cases"]}
    results = []
    for item in previous_summary["results"]:
        if item["condition"] not in ("revised", "fixed"):
            continue
        if old_cases[item["target_id"]] != new_cases.get(item["target_id"]):
            raise ValueError("reference reuse requires the same task and confirmation stream")
        reused = deepcopy(item)
        reused["condition"] = "prior_published" if item["condition"] == "revised" else "fixed"
        results.append(reused)
    return results


def paid_seed_trace(seed):
    trials = [trial for trial in seed["practice_trials"]
              if trial["case"] == "case_3" and trial["method_source"] == seed["method_source"]]
    return PREVIOUS / "learning" / trials[-1]["directory"] / "result.json"


def run_study(output, *, provider_factory=None, auth_mode=None, resume_author=None, continue_author=None,
              scope_comparison=None):
    output = Path(output).resolve()
    sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
    seed = json.loads((PREVIOUS / "learning/result.json").read_text())
    previous_plan = json.loads((PREVIOUS / "protocol.json").read_text())
    previous_summary = json.loads((PREVIOUS / "summary.json").read_text())
    plan, previous = study_protocol(sources), None
    if resume_author and continue_author:
        raise ValueError("choose service resume or additional development")
    prior_result = resume_author or continue_author
    if prior_result:
        plan, previous, accounting_parent = interrupted_state(prior_result,
            additional_development=continue_author is not None)
        if continue_author:
            plan["author_limits"]["token_budget"] += 200000
            plan["continue_author"].update(added_allowance={"model_calls": 6, "tokens": 400000},
                remaining_limits=deepcopy(plan["author_limits"]),
                author_guidance="Continue from the restored source and actual paid practice failures. "
                    "The author interface now distinguishes compact trace fields from runtime fields. "
                    "Inspect that contract and the recorded feedback, revise, then request paid "
                    "practice and publication of the current source.")
            if previous["events"] and previous["events"][-1]["action"] == "write":
                plan["continue_author"]["author_guidance"] = (
                    "The last action rewrote the research source; this version has not been practiced. "
                    "Submit the current shared/consolidated_method.py using finish now to obtain paid "
                    "practice feedback. If publication fails, revise from that feedback and submit "
                    "the revised source for practice. The current source and previous paid trials "
                    "are already restored.")
            plan["author_guidance"] = plan["continue_author"]["author_guidance"]
        reused = [item for item in accounting_parent["results"] if item["condition"] != "revised"]
    else:
        accounting_parent = previous_summary
        reused = reusable_references(plan, previous_plan, previous_summary)
    methods = {"prior_published": seed["method_source"], "fixed": DEFAULT_RESEARCH_PROCEDURE}
    new_diagnostic_costs = total_costs([])
    if scope_comparison is not None:
        comparison_path = Path(scope_comparison).resolve()
        plan["scope_comparison"] = str(comparison_path)
        new_diagnostic_costs = json.loads((comparison_path.parent / "summary.json").read_text())["costs"]
        plan["author_guidance"] += (" A new paid numerical comparison is supplied in scope_comparison.json. "
            "Use its actual validation and confirmation evidence to revise the repair choice, then "
            "submit the changed source for paid practice.")
    output.mkdir(parents=True, exist_ok=False)
    plan["provider"]["auth_mode"] = auth_mode
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    for folder in ("src/hegel_machine/cognition", "scripts"):
        shutil.copytree(ROOT / folder, output / "source_at_run" / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = sources[plan["author_source_index"]]
    memory = archive_memory(source)
    memory["files"][METHOD_PATH] = methods["prior_published"]
    author_path = output / "learning"
    provider = provider_factory(author_path) if provider_factory else CodexCodeProvider(author_path,
        model=plan["model"], reasoning_effort=plan["provider"]["reasoning_effort"], timeout_seconds=180)
    learner = ScopedPythonMethodLearner(provider, author_path, knowledge=memory,
        cases=plan["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]), prepare_trial=trial_preparer(source, plan))
    if previous is not None:
        write_json(output / "restored_practice.json", learner.restore_practice(previous))
    learner.common_interface += "\n" + plan["author_guidance"] + "\n"
    trace_path, trace_case = paid_seed_trace(seed), "case_3"
    if previous is not None:
        current_source = previous["outgoing_knowledge"]["files"][METHOD_PATH]
        failures = [trial for trial in previous["practice_trials"] if trial["status"] != "completed"]
        if failures:
            current_failures = [trial for trial in failures if trial["method_source"] == current_source]
            failed_trial = (current_failures or failures)[-1]
            trace_path = Path(prior_result).resolve().parent / failed_trial["directory"] / "result.json"
            trace_case = failed_trial["case"]
    paid = json.loads(trace_path.read_text())
    evidence = output / "paid_prior_execution.json"
    write_json(evidence, {"tasks": [paid]})
    supply_counterexample(learner, evidence, case_label=trace_case)
    if plan.get("scope_comparison"):
        text = json.dumps(json.loads(Path(plan["scope_comparison"]).read_text()), separators=(",", ":"))
        (learner.workspace / "scope_comparison.json").write_text(text)
        learner.files["scope_comparison.json"] = text
        learner.common_interface += (
            "\nThe additional scope_comparison.json reports researcher-arranged experiments on two "
            "listed practice cases, using the same public tools and paid observations. They were "
            "not chosen by your program and do not confer current-source practice coverage. "
            "The file contains observed outcomes, not a repaired procedure.\n")

    def checkpoint(result):
        write_json(author_path / "active.json", result)
        event = result["events"][-1] if result["events"] else {}
        feedback = event.get("feedback", {})
        print(json.dumps({"phase": "author", "calls": result["costs"]["model_calls"],
            "action": event.get("action"), "status": feedback.get("status", result["status"]),
            "coverage": feedback.get("current_source_coverage")}), flush=True)

    try:
        learned = learner.run(on_checkpoint=checkpoint)
    except Exception:
        learned = learner.result("execution_failed")
        learned["execution_error"] = traceback.format_exc()
    write_json(author_path / "result.json", learned)
    jobs, skipped = target_jobs(sources, output, plan, learned, methods)
    existing = {(item["target_id"], item["condition"]) for item in reused}
    jobs = [job for job in jobs if (job["spec"]["id"], job["condition"]) not in existing]

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
    executed = []
    for job, payload in trained:
        result = score_episode(payload, plan)
        result.update(target_id=job["spec"]["id"], kind=job["spec"]["kind"], condition=job["condition"],
            workspace=str(payload["path"]), unchanged_prior_selected=selected_candidate(payload) is not None,
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(job["path"] / "evaluation.json", result)
        executed.append(result)
    results = [*reused, *executed]
    conditions = {condition: {"registered_targets": len(plan["transfer_cases"]),
        **aggregate([item for item in results if item["condition"] == condition]),
        "by_kind": {kind: aggregate([item for item in results
            if item["condition"] == condition and item["kind"] == kind]) for kind in plan["target_kinds"]}}
        for condition in CONDITIONS}
    usage = usage_summary(learned)
    new_costs = total_costs([learned["costs"], *[item["costs"] for item in executed]])
    summary = {"study": plan["study"], "author_status": learned["status"],
        "method_published": learned["status"] == "completed" and learned.get("method_source") is not None,
        "author_costs": learned["costs"], "author_usage": usage,
        "cumulative_author_costs": total_costs([accounting_parent["cumulative_author_costs"], learned["costs"]]),
        "cumulative_author_usage": {key: value + accounting_parent["cumulative_author_usage"][key]
                                    for key, value in usage.items()},
        "reused_reference_episodes": len(reused), "reused_paid_trace": str(trace_path),
        "resumed_from": str(Path(prior_result).resolve()) if prior_result else None,
        "additional_development": continue_author is not None,
        "new_author_and_transfer_costs": new_costs,
        "new_scope_comparison_costs": new_diagnostic_costs,
        "new_work_including_diagnostic": total_costs([new_costs, new_diagnostic_costs]),
        "prior_method_development_costs": accounting_parent["all_method_development_costs"],
        "all_method_development_costs": total_costs([
            accounting_parent["all_method_development_costs"], new_costs, new_diagnostic_costs]),
        "source_acquisition_costs": previous_summary["source_acquisition_costs"],
        "conditions": conditions, "results": results, "skipped": skipped, "scope": plan["scope"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume-author", type=Path)
    continuation.add_argument("--continue-author", type=Path)
    parser.add_argument("--scope-comparison", type=Path)
    args = parser.parse_args()
    summary = run_study(args.output, auth_mode=verify_codex_subscription(),
        resume_author=args.resume_author, continue_author=args.continue_author,
        scope_comparison=args.scope_comparison)
    print(json.dumps({key: summary[key] for key in ("author_status", "method_published",
        "author_costs", "conditions", "new_author_and_transfer_costs")}, indent=2))

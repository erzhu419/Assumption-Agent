#!/usr/bin/env python3
"""Learn a Python research procedure, then compare frozen cross-source execution."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hegel_machine.cognition.code_agent import CodexCodeProvider
from hegel_machine.cognition.online_inquiry_llm import InquiryLLMConfig
from hegel_machine.cognition.python_method_learning import (
    PreparedPythonMethodAgent, PythonArchiveMethodLearner, SEED_PYTHON_METHOD,
)
from hegel_machine.cognition.temporal_inquiry import TemporalInquiryLab
from hegel_machine.cognition.temporal_method_learning import METHOD_PATH
from hegel_machine.cognition.temporal_paired_method import DEFAULT_RESEARCH_PROCEDURE
from scripts.run_candidate_retention_cycle import task_for
from scripts.run_candidate_retention_cycle_llm import (
    BANK_PATH, CATALOG_PATH, prepare_episode, public_candidate, score_episode, selected_candidate, write_json,
)
from scripts.run_legacy_session_llm import total_costs
from scripts.run_online_inquiry_llm import verify_codex_subscription
from scripts.run_python_procedure_study import (
    SOURCE_OUTCOMES, USAGE_KEYS, execution_evidence, protocol as procedure_protocol,
)

KINDS = ("unchanged", "response_parameter_change", "recurrent_instrument_change")
CONTINUATION_ALLOWANCE = asdict(InquiryLLMConfig(max_model_calls=8, token_budget=300000,
    observation_budget=1800, compute_budget=3000000, execution_seconds=180., confirmation_rows=0))
LIMIT_COSTS = (("max_model_calls", "model_calls"), ("observation_budget", "observation_cost"),
               ("compute_budget", "compute_cost"), ("execution_seconds", "python_execution_seconds"))


def case_spec(source, kind, *, seed_offset, response_offset):
    case = deepcopy(source["case"])
    case["seed"] += seed_offset
    case["changed_response_seed"] += response_offset
    task = task_for(case, 1)
    if kind != "response_parameter_change":
        task["response_seed"] = case["apparatus_seed"]
    if kind == "recurrent_instrument_change":
        task["instrument_mode"] = "recurrent"
    return {"case": case, "kind": kind, "task": task, "confirmation_seed": task["seed"] + 7919}


def protocol(sources):
    admitted = [(index, item) for index, item in enumerate(sources) if item["candidate"] is not None]
    author_index, author = admitted[0]
    plan = procedure_protocol("offline", sources)
    plan.update(study="python_authored_research_method_cross_source_development", mode="method_learning",
        model="gpt-6-luna", conditions=["learned", "fixed"], target_kinds=list(KINDS),
        author_source_index=author_index,
        author_limits=asdict(InquiryLLMConfig(max_model_calls=24, token_budget=900000,
            observation_budget=2400, compute_budget=3600000, execution_seconds=360., confirmation_rows=0)),
        practice_cases={f"case_{index}": case_spec(author, kind, seed_offset=300001, response_offset=400003)
                        for index, kind in enumerate(KINDS)},
        transfer_cases=[{"id": f"target_{position}_{kind_index}", "source_index": source_index,
            **case_spec(source, kind, seed_offset=500003, response_offset=600011)}
            for position, (source_index, source) in enumerate(admitted[1:])
            for kind_index, kind in enumerate(KINDS)],
        source="The first actual published archive supplies author practice. The remaining two "
            "published archives supply transfer; their files are loaded only after authoring "
            "stops. Preserve the earlier unsuccessful acquisition and its costs separately.",
        intervention="Luna writes, practices and publishes select_actions on three anonymous "
            "cases. The seed only checks the old predictor; the fixed DEFAULT procedure is "
            "not supplied to the author. Transfer compares its frozen published source with "
            "the existing developer-authored DEFAULT_RESEARCH_PROCEDURE on identical evidence.",
        controller="The author selects program edits, paid practice and publication. Each "
            "practice and transfer begins with 72 paid controls-only rows, then executes at "
            "most 12 declared program steps with 24-row fresh confirmation. All fits and "
            "observations share each episode's actual budgets. Transfer uses zero LLM calls; "
            "this is LLM-authored method execution, not online LLM selection of each action.",
        failure_policy="Keep the first author attempt and all paid practice. Do not replace "
            "a failed author/source or retry publication. If no method is published, preserve "
            "all six learned-arm slots as skipped; execute the six preregistered fixed cases "
            "as resource references only, without a paired-benefit claim.",
        accounting="Report old source acquisition, new author/practice and zero-LLM transfer "
            "costs separately. Charge each paid prefix in its own practice/transfer episode. "
            "The actual run total adds the author once and all executed targets; unknown "
            "provider usage remains null. Program steps are not provider calls.",
        terminal_order="Publish using paid practice and fresh confirmation only. Freeze the "
            "method before loading transfer archives. Complete all transfer research before "
            "terminal scoring; no terminal outcome affects publication or target admission.",
        scope="Two previously learned apparatus realizations, each tested under unchanged, "
            "response-parameter and recurrent-instrument conditions: six paired targets. "
            "New preregistered trajectory and changed-response seeds avoid replaying the old "
            "six targets. This is cross-source transfer within existing apparatus families, "
            "not new-apparatus or open-world discovery.")
    plan.pop("cases", None)
    return plan


def laboratory(spec):
    return TemporalInquiryLab(**spec["task"])


def archive_memory(source):
    """Read the author's real archive without acquiring an unused extra prefix."""
    candidate = source["candidate"]
    archive = Path(candidate["snapshot_workspace"])
    destination = Path(BANK_PATH) / candidate["candidate_id"] / "_evaluated_model"
    catalog = [public_candidate(candidate)]
    files = {CATALOG_PATH: json.dumps(catalog, indent=2), METHOD_PATH: SEED_PYTHON_METHOD}
    for path in archive.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
            files[str(destination / path.relative_to(archive))] = path.read_text(encoding="utf-8")
    return {"files": files, "candidates": catalog,
            "research_roles": {"predictor": candidate["model"]}}


def trial_preparer(source, plan):
    def prepare_trial(spec, path, method_source, remaining_limits):
        local = {**plan, "limits": asdict(remaining_limits)}
        payload = prepare_episode(spec["case"], 1, path, [deepcopy(source["candidate"])], local,
            laboratory=laboratory(spec), agent_class=PreparedPythonMethodAgent)
        payload["agent"].knowledge["research_roles"] = {"predictor": source["candidate"]["model"]}
        payload["agent"].bind_method(method_source)
        return payload
    return prepare_trial


def target_jobs(sources, output, plan, learned):
    published = learned["status"] == "completed" and learned.get("method_source") is not None
    jobs, skipped = [], []
    for index, spec in enumerate(plan["transfer_cases"]):
        order = plan["conditions"] if index % 2 == 0 else plan["conditions"][::-1]
        for condition in order:
            entry = {"spec": spec, "condition": condition,
                     "path": Path(output) / f"trial_{2 * index + order.index(condition):03d}"}
            if condition == "learned" and not published:
                skipped.append({"target_id": spec["id"], "kind": spec["kind"], "condition": condition,
                                "reason": "method_not_published", "author_status": learned["status"]})
                continue
            entry.update(source=sources[spec["source_index"]],
                method_source=learned["method_source"] if condition == "learned" else DEFAULT_RESEARCH_PROCEDURE)
            jobs.append(entry)
    return jobs, skipped


def usage_summary(training):
    usage = [event.get("usage", {}) for event in training["events"]]
    return {"known_tokens": sum(item["total_tokens"] for item in usage if isinstance(item.get("total_tokens"), int)),
        "missing_usage_calls": sum(not isinstance(item.get("total_tokens"), int) for item in usage),
        "input_tokens": sum(item.get("input_tokens") or 0 for item in usage),
        "cached_input_tokens": sum(item.get("cached_input_tokens") or 0 for item in usage)}


def _previous_author_state(result_path, allowed_statuses):
    """Keep the actual source, trials and costs from the completed previous phase."""
    result_path = Path(result_path).resolve()
    previous = json.loads(result_path.read_text())
    if previous["status"] not in allowed_statuses or previous.get("method_source") is not None:
        raise ValueError("author restoration requires an unpublished result with status: "
                         + ", ".join(sorted(allowed_statuses)))
    prior_root = result_path.parent.parent
    plan = json.loads((prior_root / "protocol.json").read_text())
    sources = json.loads((prior_root / "existing_source_outcomes.json").read_text())["sources"]
    prior_summary = json.loads((prior_root / "summary.json").read_text())
    fixed = prior_summary["conditions"]["fixed"]
    plan["prior_author_phases"] = [*plan.get("prior_author_phases", []), {"result": str(result_path),
        "status": previous["status"], "costs": deepcopy(previous["costs"]), "usage": usage_summary(previous)}]
    plan["prior_fixed_reference_runs"] = [*plan.get("prior_fixed_reference_runs", []), {
        "summary": str(prior_root / "summary.json"), "attempted_targets": fixed["attempted_targets"],
        "costs": deepcopy(fixed["costs"]), "usage": {key: fixed[key] for key in USAGE_KEYS}}]
    return plan, sources, previous


def resume_author_state(result_path):
    """Resume an actual provider interruption under its existing total allowance."""
    plan, sources, previous = _previous_author_state(result_path, {"provider_failed"})
    total_limits = deepcopy(plan.get("author_total_limits", plan["author_limits"]))
    phases = plan["prior_author_phases"]
    spent = total_costs([phase["costs"] for phase in phases])
    known_tokens = sum(phase["usage"]["known_tokens"] for phase in phases)
    remaining = deepcopy(total_limits)
    for limit, cost in LIMIT_COSTS:
        remaining[limit] = max(0, total_limits[limit] - spent[cost])
    remaining["token_budget"] = max(0, total_limits["token_budget"] - known_tokens)
    plan.update(author_total_limits=total_limits, author_limits=remaining,
        resume_author={"result": str(Path(result_path).resolve()),
            "restoration": "Use the existing restore_practice path to retain exact current source and "
                "paid practice coverage. Cases, archive identities, seeds, target conditions and thresholds "
                "are inherited unchanged. Earlier practice is copied, not replayed; fixed-reference "
                "outcomes are excluded from author context. No automatic retry is scheduled.",
            "budget": "Subtract all prior actual author costs from the original total allowance. "
                "Subtract known tokens only; unknown usage remains disclosed rather than estimated.",
            "execution_repair": "Restore the existing PairedMethodLearner rule that advances each "
                "practice confirmation seed by the retained practice-trial count. Case base seeds, "
                "paid prefixes, response parameters and transfer thresholds are unchanged. The first "
                "interrupted phase had one practice with no fresh confirmation, so no completed "
                "confirmation coverage is invalidated.",
            "prior_unknown_usage_calls": sum(phase["usage"]["missing_usage_calls"] for phase in phases),
            "transfer": "After this author phase ends, execute registered transfer targets afresh. "
                "Earlier fixed-reference executions remain separate additional costs."})
    return plan, sources, previous


def continue_author_state(result_path):
    """Begin an explicit additional development phase with all paid evidence retained."""
    plan, sources, previous = _previous_author_state(result_path,
        {"token_budget_exhausted", "model_call_budget_exhausted"})
    phases = plan["prior_author_phases"]
    spent = total_costs([phase["costs"] for phase in phases])
    known_tokens = sum(phase["usage"]["known_tokens"] for phase in phases)
    original = deepcopy(plan.get("author_original_limits", plan.get("author_total_limits", plan["author_limits"])))
    cumulative_limit = deepcopy(CONTINUATION_ALLOWANCE)
    for limit, cost in LIMIT_COSTS:
        cumulative_limit[limit] += spent[cost]
    cumulative_limit["token_budget"] += known_tokens
    plan.pop("resume_author", None)
    plan["provider"]["reasoning_effort"] = "high"
    for spec in plan["practice_cases"].values():
        spec["confirmation_seed"] += 700001
    plan["transfer_cases"] = [{**spec, **case_spec({"case": spec["case"]}, spec["kind"],
        seed_offset=700003, response_offset=700009)} for spec in plan["transfer_cases"]]
    plan.update(author_phase="additional_development", author_original_limits=original,
        author_limits=deepcopy(CONTINUATION_ALLOWANCE), author_total_limits=cumulative_limit,
        continue_author={"result": str(Path(result_path).resolve()),
            "additional_allowance": deepcopy(CONTINUATION_ALLOWANCE),
            "author_interface": "The author's explicit finish requests missing paid practice cases "
                "in declaration order and publication under the existing reuse-and-revision condition. "
                "The first failed practice returns its actual trace for author revision. The host "
                "does not edit the source or choose the experiments within a practice. This development "
                "phase uses the same Luna model with high reasoning.",
            "prior_unknown_usage_calls": sum(phase["usage"]["missing_usage_calls"] for phase in phases),
            "restoration": "Keep the exact current source, paid trial files and source-specific coverage. "
                "Practice physics, paid prefixes, archives and thresholds are unchanged. "
                "Previously paid trials are not rerun during restoration. Source edits require "
                "fresh confirmation of every case with that edited source before publication.",
            "new_streams": "Before this run, add 700001 to each practice confirmation seed, so "
                "restoring the author's trial count does not repeat the intervening offline replay. "
                "For all six transfer cases, advance trajectory seeds by 700003 and changed-response "
                "seeds by 700009; keep the two archives and three condition types. No case filtering.",
            "budget": "This is an explicitly added development allowance after budget exhaustion, "
                "not success within the original author budget. Old actual costs remain cumulative; "
                "they are not subtracted again from this new allowance. Cumulative limits equal "
                "prior actual costs plus this allowance, using known tokens only and disclosing unknown usage.",
            "evidence": "Only prior author practice and public interfaces enter the author context. "
                "Previous fixed-reference and offline-replay outcomes remain excluded. Continued "
                "authoring and new-trajectory transfer are development evidence, not independent authoring.",
            "transfer": "After authoring stops, execute the registered targets afresh. Retain all "
                "earlier fixed-reference costs separately."},
        failure_policy="This explicitly requested additional development phase retains all prior failures. "
            "No automatic retry or unpublished-source transfer follows. If publication fails, preserve "
            "the six learned slots as skipped and run fixed cases only as resource references.")
    return plan, sources, previous


def summarize(sources, learned, results, skipped, plan):
    def aggregate(items):
        return {"attempted_targets": len(items),
            "confirmed_and_correct": sum(item["confirmed_and_correct"] for item in items),
            "false_confirmations": sum(item["false_confirmation"] for item in items),
            "unchanged_prior_selected": sum(item["unchanged_prior_selected"] for item in items),
            "statuses": dict(Counter(item["status"] for item in items)),
            "costs": total_costs([item["costs"] for item in items]),
            **{key: sum(item[key] for item in items) for key in USAGE_KEYS},
            "method_steps": sum(item["method_steps"] for item in items)}

    conditions = {}
    for condition in plan["conditions"]:
        selected = [item for item in results if item["condition"] == condition]
        conditions[condition] = {"registered_targets": len(plan["transfer_cases"]), **aggregate(selected),
            "author_blocked_targets": sum(item["condition"] == condition for item in skipped),
            "by_kind": {kind: aggregate([item for item in selected if item["kind"] == kind]) for kind in KINDS}}
    author_usage = usage_summary(learned)
    prior_phases = plan.get("prior_author_phases", [])
    prior_fixed = plan.get("prior_fixed_reference_runs", [])
    author_costs = [*[phase["costs"] for phase in prior_phases], learned["costs"]]
    return {"study": plan["study"], "author_status": learned["status"],
        "author_phase": plan.get("author_phase", "original_budget"),
        "additional_author_allowance": plan.get("continue_author", {}).get("additional_allowance"),
        "method_published": learned["status"] == "completed" and learned.get("method_source") is not None,
        "paired_comparison_available": not skipped, "registered_target_episodes": 2 * len(plan["transfer_cases"]),
        "author_costs": learned["costs"], "author_usage": author_usage,
        "prior_author_phases": prior_phases, "prior_fixed_reference_runs": prior_fixed,
        "cumulative_author_costs": total_costs(author_costs),
        "cumulative_author_usage": {key: author_usage[key] + sum(phase["usage"][key] for phase in prior_phases)
                                    for key in USAGE_KEYS},
        "existing_sources": sources,
        "existing_all_source_acquisition_costs": total_costs([item["costs"] for item in sources]),
        "new_transfer_costs": total_costs([item["costs"] for item in results]),
        "actual_new_author_plus_transfer_costs": total_costs([learned["costs"], *[item["costs"] for item in results]]),
        "prior_fixed_reference_costs": total_costs([item["costs"] for item in prior_fixed]),
        "actual_all_author_phases_plus_all_transfer_costs": total_costs(
            [*author_costs, *[item["costs"] for item in prior_fixed], *[item["costs"] for item in results]]),
        "all_phases_usage": {key: author_usage[key] + sum(phase["usage"][key] for phase in prior_phases)
            + sum(item["usage"][key] for item in prior_fixed) + sum(item[key] for item in results)
            for key in USAGE_KEYS},
        **{key: author_usage[key] + sum(item[key] for item in results) for key in USAGE_KEYS},
        "conditions": conditions, "results": results, "skipped": skipped, "scope": plan["scope"]}


def run_study(output, *, provider_factory=None, auth_mode=None, resume_author=None, continue_author=None):
    previous = None
    if resume_author is not None and continue_author is not None:
        raise ValueError("choose either provider resume or additional development continuation")
    if continue_author is not None:
        plan, sources, previous = continue_author_state(continue_author)
    elif resume_author is None:
        sources = json.loads(SOURCE_OUTCOMES.read_text())["sources"]
        plan = protocol(sources)
    else:
        plan, sources, previous = resume_author_state(resume_author)
    plan["provider"]["auth_mode"] = auth_mode
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "existing_source_outcomes.json", {"sources": sources})
    for index, phase in enumerate(plan.get("prior_author_phases", [])):
        directory = output / "prior_author_phases" / f"phase_{index:03d}"
        write_json(directory / "costs.json", phase)
        shutil.copy2(phase["result"], directory / "result.json")
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (Path(__file__).name, "run_python_procedure_study.py", "run_python_pending_advice_study.py",
                 "run_python_advice_ablation.py", "run_python_research_routing.py", "run_python_refit_diagnosis.py",
                 "run_candidate_retention_cycle_llm.py", "run_candidate_retention_cycle.py",
                 "run_legacy_session_llm.py", "run_online_inquiry_llm.py"):
        target = output / "source_at_run/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    source = sources[plan["author_source_index"]]
    author_path = output / "learning"
    provider = provider_factory(author_path) if provider_factory else CodexCodeProvider(
        author_path, model=plan["model"], reasoning_effort=plan["provider"]["reasoning_effort"],
        timeout_seconds=180)
    learner = PythonArchiveMethodLearner(provider, author_path,
        knowledge=archive_memory(source), cases=plan["practice_cases"], lab_factory=laboratory,
        config=InquiryLLMConfig(**plan["author_limits"]), prepare_trial=trial_preparer(source, plan))
    if previous is not None:
        write_json(output / "restored_practice.json", learner.restore_practice(previous))

    def author_checkpoint(result):
        write_json(author_path / "active.json", result)
        event = result["events"][-1] if result["events"] else {}
        arguments, feedback = event.get("arguments", {}), event.get("feedback", {})
        print(json.dumps({"phase": "author", "calls": result["costs"]["model_calls"],
            "action": event.get("action"), "operator": arguments.get("operator"),
            "practice": arguments.get("case"), "status": feedback.get("status", result["status"])}), flush=True)

    try:
        learned = learner.run(on_checkpoint=author_checkpoint)
    except Exception:
        learned = learner.result("execution_failed")
        learned["execution_error"] = traceback.format_exc()
    write_json(author_path / "result.json", learned)
    jobs, skipped = target_jobs(sources, output, plan, learned)

    def train(job):
        payload = trial_preparer(job["source"], plan)(job["spec"], job["path"], job["method_source"],
                                                     InquiryLLMConfig(**plan["limits"]))
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"],
                on_checkpoint=lambda result: write_json(job["path"] / "active.json", result))
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        write_json(job["path"] / "training.json", payload["training"])
        return job, payload

    # Only now are transfer archive contents loaded. No target changes authoring.
    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    results = []
    for job, payload in trained:
        result = score_episode(payload, plan)
        result.update(target_id=job["spec"]["id"], case=job["spec"]["case"], kind=job["spec"]["kind"],
            condition=job["condition"], workspace=str(payload["path"]),
            unchanged_prior_selected=selected_candidate(payload) is not None,
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(output / "evaluations" / (job["path"].name + ".json"), result)
        results.append(result)
    summary = summarize(sources, learned, results, skipped, plan)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume-author", type=Path,
        help="Restore a provider-failed learning/result.json under its remaining existing budget")
    continuation.add_argument("--continue-author", type=Path,
        help="Add the registered development allowance after token/model-call budget exhaustion")
    args = parser.parse_args()
    summary = run_study(args.output, resume_author=args.resume_author, continue_author=args.continue_author,
        auth_mode=verify_codex_subscription())
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"existing_sources", "results"}}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay an unchanged, unpublished method candidate without further model calls."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_python_method_learning_study import (
    DEFAULT_RESEARCH_PROCEDURE, InquiryLLMConfig, METHOD_PATH, PythonArchiveMethodLearner,
    archive_memory, execution_evidence, laboratory, score_episode, selected_candidate,
    total_costs, trial_preparer, write_json,
)


def run_replay(author_result, output):
    author_result, output = Path(author_result).resolve(), Path(output).resolve()
    previous = json.loads(author_result.read_text())
    if previous["status"] not in {"token_budget_exhausted", "model_call_budget_exhausted"} or previous.get("method_source"):
        raise ValueError("candidate replay requires an unpublished budget-exhausted author")
    prior_root = author_result.parent.parent
    plan = json.loads((prior_root / "protocol.json").read_text())
    prior_summary = json.loads((prior_root / "summary.json").read_text())
    sources = json.loads((prior_root / "existing_source_outcomes.json").read_text())["sources"]
    candidate = previous["outgoing_knowledge"]["files"][METHOD_PATH]
    practice_limits = replace(InquiryLLMConfig(**plan["author_limits"]), max_model_calls=0, token_budget=0)
    plan.update(study="python_frozen_unpublished_method_candidate_replay", mode="offline_candidate_replay",
        conditions=["candidate", "fixed"], author_result=str(author_result),
        prior_summary=str(prior_root / "summary.json"), replay_practice_limits=vars(practice_limits),
        replay_scope="Freeze the exact last outgoing source. Restore paid trials without replaying them, "
            "then execute every declared practice case anew. Require all new practices to confirm, with "
            "unchanged-prior reuse and revision, before candidate transfer. This host-selected offline "
            "development replay is not autonomous author publication or independent authoring.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    (output / "candidate.py").write_text(candidate)
    shutil.copytree(ROOT / "src/hegel_machine/cognition", output / "source_at_run/src/hegel_machine/cognition",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "scripts", output / "source_at_run/scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = sources[plan["author_source_index"]]
    learner = PythonArchiveMethodLearner(None, output / "practice", knowledge=archive_memory(source),
        cases=plan["practice_cases"], lab_factory=laboratory, config=practice_limits,
        prepare_trial=trial_preparer(source, plan))
    write_json(output / "restored_practice.json", learner.restore_practice(previous))
    practices = []
    for case in plan["practice_cases"]:
        outcome = learner._practice(case)
        practices.append({"case": case, **outcome})
        write_json(output / "practice_outcomes.json", practices)
    eligible = (all(item.get("status") == "completed" and
                    (item.get("fresh_confirmation") or {}).get("passed") is True for item in practices)
                and any(item.get("selected_prior_predictor") is True for item in practices)
                and any(item.get("selected_prior_predictor") is False for item in practices))
    write_json(output / "practice/result.json", learner.result("offline_candidate_replay_finished"))
    trained, skipped = [], []
    for index, spec in enumerate(plan["transfer_cases"]):
        for condition in plan["conditions"] if index % 2 == 0 else plan["conditions"][::-1]:
            if condition == "candidate" and not eligible:
                skipped.append({"target_id": spec["id"], "kind": spec["kind"], "condition": condition,
                                "reason": "candidate_replay_practice_gate_failed"})
                continue
            path = output / f"target_{index:03d}_{condition}"
            method = candidate if condition == "candidate" else DEFAULT_RESEARCH_PROCEDURE
            payload = trial_preparer(sources[spec["source_index"]], plan)(
                spec, path, method, InquiryLLMConfig(**plan["limits"]))
            started = monotonic()
            try:
                payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                    max_steps=plan["procedure_step_limit"])
            except Exception:
                payload["training"] = payload["agent"].result("execution_failed")
                payload["training"]["execution_error"] = traceback.format_exc()
            payload["research_wall_seconds"] = monotonic() - started
            write_json(path / "training.json", payload["training"])
            trained.append((spec, condition, payload))
    results = []
    for spec, condition, payload in trained:  # Every practice and target stops before terminal scoring.
        result = score_episode(payload, plan)
        result.update(target_id=spec["id"], kind=spec["kind"], condition=condition,
            workspace=str(payload["path"]), unchanged_prior_selected=selected_candidate(payload) is not None,
            **execution_evidence(payload["agent"], payload["training"]))
        write_json(payload["path"] / "evaluation.json", result)
        results.append(result)
    practice_costs = deepcopy(learner.costs)
    transfer_costs = total_costs([item["costs"] for item in results])
    new_costs = total_costs([practice_costs, transfer_costs])
    conditions = {}
    for condition in plan["conditions"]:
        items = [item for item in results if item["condition"] == condition]
        conditions[condition] = {"registered_targets": len(plan["transfer_cases"]),
            "attempted_targets": len(items), "confirmed_and_correct": sum(item["confirmed_and_correct"] for item in items),
            "false_confirmations": sum(item["false_confirmation"] for item in items),
            "unchanged_prior_selected": sum(item["unchanged_prior_selected"] for item in items),
            "statuses": dict(Counter(item["status"] for item in items)),
            "costs": total_costs([item["costs"] for item in items])}
    summary = {"study": plan["study"], "author_status": previous["status"], "method_published": False,
        "candidate_replay_eligible": eligible, "paired_comparison_available": eligible,
        "candidate_source": "candidate.py", "prior_summary": plan["prior_summary"],
        "cumulative_author_costs": prior_summary["cumulative_author_costs"],
        "cumulative_author_usage": prior_summary["cumulative_author_usage"],
        "prior_fixed_reference_costs": total_costs([prior_summary["prior_fixed_reference_costs"],
                                                     prior_summary["conditions"]["fixed"]["costs"]]),
        "prior_all_author_and_transfer_costs": prior_summary["actual_all_author_phases_plus_all_transfer_costs"],
        "new_practice_costs": practice_costs, "new_transfer_costs": transfer_costs, "new_replay_costs": new_costs,
        "actual_cumulative_costs": total_costs([prior_summary["actual_all_author_phases_plus_all_transfer_costs"], new_costs]),
        "new_model_calls": 0, "new_reported_tokens": 0, "new_cached_input_tokens": 0,
        "practices": practices, "conditions": conditions, "results": results, "skipped": skipped,
        "scope": plan["replay_scope"]}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_replay(args.author_result, args.output)
    print(json.dumps({key: result[key] for key in ("author_status", "method_published",
        "candidate_replay_eligible", "conditions", "new_replay_costs")}, indent=2))

#!/usr/bin/env python3
"""Compare a submitted revision with a retained research program after failure."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts import run_python_method_reopening_study as previous

PRIOR = ROOT / "artifacts/python_method_edit_practice_luna_20261006"
CONTINUATION = """
Continue the unfinished authoring phase recorded in resume_context.json. Its paid
practice trials and source snapshots remain available. Choose how to continue
from their actual evidence and publish a program satisfying all existing cases.
The current phase has its own limited allowance. No additional source repair,
selected historical draft, or target outcomes are supplied.
"""


def run_study(output, *, provider_factory=None, auth_mode=None):
    output = Path(output).resolve()
    plan = json.loads((PRIOR / "protocol.json").read_text())
    old = json.loads((PRIOR / "author_2/learning/result.json").read_text())
    sources = json.loads((PRIOR / "existing_source_outcomes.json").read_text())["sources"]
    plan.update(study="python_retained_method_comparison_development",
        mode="retained_parent_comparison",
        conditions=["previous_compact_recall", "retained_parent_comparison"],
        continuation_from=str(PRIOR / "author_2/learning/result.json"),
        registered_attempts=3,
        scope="Three model attempts on one existing regression state, not three independent tasks. "
              "The host compares a retained program after an author submits a failed revision. "
              "Selecting the retained source is host-supported retention, not autonomous LLM recall. "
              "The prior compact-context three-author cohort is a sequential development control, "
              "not independent task confirmation or a randomized estimate of reliability.",
        intervention="Expose recorded source candidates and their actual paid outcomes; allow "
              "restore_method. Start each author with the last failed source, not the earlier "
              "successful source. A failed finish now tests an eligible retained source, beginning "
              "with the failed case if it lacks confirmation, then its own other missing cases. "
              "Both sources share the same budget and unchanged publication requirements. "
              "The parent replaces the draft only if the parent's own evidence passes all gates.",
        controller="The author chooses edits, ordinary inspection and submission. On a failed "
              "submission the host may restore a retained supported source and execute its own "
              "missing practices through the same tools, author account and publication gates. "
              "This controller decision is reported separately from explicit model restore actions.",
        accounting="Prior authoring is already paid and reported separately. Each new author "
              "is charged once. Execute each distinct published source on the twelve old target "
              "tasks once; identical sources share that cohort without adding replications.",
        failure_policy="Three fixed attempts, up to four calls each, without replacement.",
        terminal_order="Complete all author attempts and all target research before target scoring.")
    plan["author_limits"].update(max_model_calls=4, token_budget=300000,
        observation_budget=2400, compute_budget=1500000, execution_seconds=180.)
    control = ROOT / "artifacts/python_method_recall_compact_luna_20261007"
    control_plan = json.loads((control / "protocol.json").read_text())
    for key in ("continuation_from", "author_limits", "author_replicas", "transfer_cases", "limits",
                "procedure_step_limit", "evaluation", "model"):
        if plan[key] != control_plan[key]:
            raise ValueError(f"the previous recall control has a different {key}")
    plan["control_reference"] = str(control / "summary.json")
    plan["provider"]["auth_mode"] = auth_mode
    output.mkdir(parents=True, exist_ok=False)
    previous.write_json(output / "protocol.json", plan)
    shutil.copy2(__file__, output / "runner_at_start.py")
    shutil.copy2(ROOT / "src/hegel_machine/cognition/temporal_paired_practice.py",
                 output / "candidate_interface_at_start.py")
    shutil.copy2(ROOT / "src/hegel_machine/cognition/python_method_learning.py",
                 output / "publication_comparison_at_start.py")
    seed = (PRIOR / "unchanged_method.py").read_text()

    def author(index):
        name, path = f"author_{index}", output / f"author_{index}/learning"
        provider = provider_factory(path) if provider_factory else previous.CodexCodeProvider(
            path, model=plan["model"], reasoning_effort="high", timeout_seconds=180)
        learner = previous.prepare_author(provider, path, plan, plan["author_replicas"][2],
            sources, seed, PRIOR / "paid_source_counterexample.json")
        learner.restore_practice(old)
        learner.common_interface += CONTINUATION

        def checkpoint(result):
            previous.write_json(path / "active.json", result)
            event = result["events"][-1] if result["events"] else {}
            print(json.dumps({"author": name, "calls": result["costs"]["model_calls"],
                "action": event.get("action"),
                "restored": event.get("feedback", {}).get("restored_method"),
                "coverage": event.get("feedback", {}).get("current_source_coverage")}), flush=True)

        try:
            result = learner.run(on_checkpoint=checkpoint)
        except Exception:
            result = learner.result("execution_failed")
            result["execution_error"] = traceback.format_exc()
        result["current_source_coverage"] = learner._current_source_coverage()
        previous.write_json(path / "result.json", result)
        return name, result

    with ThreadPoolExecutor(max_workers=3) as pool:
        learned = list(pool.map(author, range(plan["registered_attempts"])))
    cohorts, authors = {}, []
    for name, result in learned:
        source = result.get("method_source") if result["status"] == "completed" else None
        if source is not None and source not in cohorts:
            cohorts[source] = f"cohort_{len(cohorts)}"
        authors.append({"id": name, "status": result["status"], "published": source is not None,
            "restored_trials": [event["feedback"]["restored_method"] for event in result["events"]
                                if event.get("feedback", {}).get("restored_method")],
            "published_prior_trials": [trial["directory"] for trial in old["practice_trials"]
                                       if source is not None and trial["method_source"] == source],
            "incumbent_comparisons": [{key: value for key, value in event["feedback"]["incumbent_comparison"].items()
                                        if key != "practices"}
                                       for event in result["events"]
                                       if event.get("feedback", {}).get("incumbent_comparison")],
            "costs": result["costs"], "usage": previous.usage_summary(result),
            "new_practices": len(result["practice_trials"]) - len(old["practice_trials"]),
            "coverage": result["current_source_coverage"], "target_cohort": cohorts.get(source)})
    previous.write_json(output / "authors.json", authors)

    def train(job):
        source, cohort, spec = job
        path = output / cohort / spec["id"]
        payload = previous.trial_preparer(sources[spec["source_index"]], plan)(
            spec, path, source, previous.InquiryLLMConfig(**plan["limits"]))
        started = monotonic()
        try:
            payload["training"] = payload["agent"].execute_method(payload["laboratory"],
                max_steps=plan["procedure_step_limit"])
        except Exception:
            payload["training"] = payload["agent"].result("execution_failed")
            payload["training"]["execution_error"] = traceback.format_exc()
        payload["research_wall_seconds"] = monotonic() - started
        previous.write_json(path / "training.json", payload["training"])
        return cohort, spec, payload

    jobs = [(source, cohort, spec) for source, cohort in cohorts.items()
            for spec in plan["transfer_cases"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train, jobs))
    outcomes = {cohort: [] for cohort in cohorts.values()}
    for cohort, spec, payload in trained:
        row = previous.score_episode(payload, plan)
        row.update(target_id=spec["id"], kind=spec["kind"],
            unchanged_prior_selected=previous.selected_candidate(payload) is not None,
            scoped_revision=previous.scoped_revision_evidence(payload["training"]))
        previous.write_json(payload["path"] / "evaluation.json", row)
        outcomes[cohort].append(row)
    summary = {"study": plan["study"], "authors": authors,
        "published_authors": sum(item["published"] for item in authors),
        "target_cohorts": {name: {**previous.aggregate(rows), "results": rows}
                           for name, rows in outcomes.items()},
        "unique_published_sources": len(cohorts), "new_target_executions": len(trained),
        "prior_author_costs": old["costs"],
        "new_author_costs": previous.total_costs([item["costs"] for item in authors]),
        "new_target_costs": previous.total_costs([row["costs"] for rows in outcomes.values() for row in rows]),
        "author_usage": {key: sum(item["usage"][key] for item in authors) for key in authors[0]["usage"]},
        "prior_control": {"reference": plan["control_reference"],
            "published_authors": json.loads((control / "summary.json").read_text())["published_authors"],
            "reused_author_attempts": 3},
        "scope": plan["scope"]}
    previous.write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output, auth_mode=previous.verify_codex_subscription())
    print(json.dumps({key: value for key, value in result.items() if key != "target_cohorts"}, indent=2))

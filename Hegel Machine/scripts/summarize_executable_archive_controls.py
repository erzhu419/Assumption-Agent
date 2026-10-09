#!/usr/bin/env python3
"""Read existing controls without treating a relabelled executor as a new arm."""

import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def summarize():
    retention_path = "artifacts/paper_scoped_retention_20261005/summary.json"
    flat_path = "artifacts/temporal_matched_repertoire_20261006/summary.json"
    methods_path = "artifacts/python_method_edit_practice_luna_20261006/summary.json"
    sources_path = "artifacts/python_method_learning_contract_luna_20261006/existing_source_outcomes.json"
    retention, flat = read(retention_path), read(flat_path)
    methods, sources = read(methods_path), read(sources_path)["sources"]

    # Compare the recorded per-task outcomes, excluding elapsed CPU time.
    same_retention = []
    for row in retention["rows"]:
        arms = row.get("arms", {})
        if "conditional_retention" not in arms or "unmarked_retention" not in arms:
            continue
        left, right = arms["conditional_retention"], arms["unmarked_retention"]
        same_retention.append(all(left[key] == right[key] for key in
            ("solved", "fresh_confirmed", "mean_mse", "costs", "successors")))

    original_trials, method_summaries = [], []

    def collect_runtime(owner, results):
        for result in results:
            training_path = Path(result["workspace"]) / "training.json"
            training = json.loads(training_path.read_text())
            original_trials.append({
                "owner": owner, "target": result["target_id"],
                "training": str(training_path.relative_to(ROOT)),
                "agent_arm": training["agent_arm"],
                "online_question_count": len(training["online_questions"]),
                "child_question_count": sum(bool(q.get("parent_id"))
                    for q in training["online_questions"]),
                "published_manifest": training["outgoing_knowledge"].get("manifest"),
                "model_calls": training["costs"]["model_calls"],
            })

    # The shared control executions are counted once, not once per author.
    collect_runtime("unchanged_control", methods["unchanged_control"]["results"])
    for author in methods["authors"]:
        method_summaries.append({
            key: author[key] for key in
            ("replica_id", "author_status", "method_published", "source_changed")
        } | {"executed_targets": len(author["results"]),
             "successful_targets": sum(r["confirmed_and_correct"] for r in author["results"]),
             "skipped_targets": len(author["skipped"])})
        collect_runtime(author["replica_id"], author["results"])

    return {
        "study": "existing_executable_archive_control_attribution",
        "evidence_kind": "read_only_reaggregation_of_existing_runs",
        "new_experiments": 0, "new_model_calls": 0,
        "retention_marker_control": {
            "source": retention_path,
            "registered_cases": retention["planned_cases"],
            "recorded_cases": len(retention["rows"]),
            "execution_failures": retention["execution_failures"],
            "complete_retention_pairs": len(same_retention),
            "pairs_with_identical_quality_and_charged_costs": sum(same_retention),
            "arms": {arm: {
                "full_sequence_solved": result["solved"],
                "full_sequence_tasks": result["tasks"],
                "successor_solved": sum(r["solved"] for r in result["successors"]),
                "successor_tasks": 3 * retention["planned_cases"],
                "costs": result["costs"],
            } for arm, result in retention["totals"].items()},
            "implementation": "scripts/run_scoped_retention_comparison.py:run_benchmark",
            "contrast": "Only the observed-failure conditional-prior record changes between retention arms.",
            "organization_retained": "Both arms clone KnowledgeLibrary and execute UnifiedResearchSession.",
            "attribution": "No independent measured benefit from the extra conditional marker; not a complete removal of hypothesis organization.",
        },
        "matched_repertoire_flat_control": {
            "source": flat_path,
            "apparatus_pairs": flat["apparatus_pairs"],
            "totals": flat["totals"], "limits": flat["limits"],
            "shared_repertoire": flat["shared_repertoire"],
            "contrast": flat["contrast"], "compute_scope": flat["compute_scope"],
            "implementation": "src/hegel_machine/cognition/temporal_flat_research.py:run_flat_temporal_research",
            "attribution": "Shared candidates and ordinary held-out response selection match recursive success without a question graph or separate child acceptance.",
        },
        "frozen_python_archive_execution": {
            "source": methods_path, "source_acquisition": sources_path,
            "registered_source_attempts": len(sources),
            "published_source_archives": sum(s["candidate"] is not None for s in sources),
            "failed_sources": [{"source_index": i, "status": s["status"], "costs": s["costs"]}
                for i, s in enumerate(sources) if s["candidate"] is None],
            "authors": method_summaries,
            "unchanged_control": {key: methods["unchanged_control"][key] for key in
                ("attempted_targets", "confirmed_and_correct", "false_confirmations", "costs")},
            "reused_control_count": methods["reused_control_count"],
            "actual_runtime_arm_counts": dict(Counter(r["agent_arm"] for r in original_trials)),
            "child_question_count": sum(r["child_question_count"] for r in original_trials),
            "trials": original_trials,
            "entry_points": ["scripts/run_python_method_learning_study.py:trial_preparer",
                "scripts/run_candidate_retention_cycle_llm.py:prepare_episode",
                "src/hegel_machine/cognition/python_method_learning.py:PreparedPythonMethodAgent"],
            "attribution": "The published source and fixed repair control already execute as code_react on the same archived code/data with shared refit, response fitting and fresh confirmation. Differences identify procedure source behavior, not a Hegel organization advantage.",
        },
        "remaining_comparison": "A single same-history archive comparison removing support-obligation, applicability and recursive control together while retaining all ordinary repair permissions has not been executed. The existing controls answer narrower, distinct questions.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "new_experiments": 0,
        "retention_identical_pairs": result["retention_marker_control"]["pairs_with_identical_quality_and_charged_costs"],
        "python_runtime_arms": result["frozen_python_archive_execution"]["actual_runtime_arm_counts"]}))


if __name__ == "__main__":
    main()

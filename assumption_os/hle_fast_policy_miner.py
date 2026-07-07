"""Mine shadow fast-policy hypotheses from redacted HLE transition rows.

The miner is intentionally conservative: it proposes typed fast policies from
recurring transition failures, but marks them as candidates until a separate
fixed-regression and unseen-cohort promotion gate passes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .autonomy_journal import stable_hash
from .fast_policy_memory import FAST_POLICY_MEMORY_VERSION, FastPolicyHypothesis
from .hle_transition_dataset import TRANSITION_DATASET_VERSION, build_transition_dataset_from_paths


POLICY_MINER_VERSION = "hle_fast_policy_miner_v1"


def mine_fast_policy_hypotheses(
    transition_payload: dict[str, Any] | Iterable[dict[str, Any]],
    *,
    min_support: int = 2,
    max_evidence_rows: int = 5,
) -> dict[str, Any]:
    records = _records(transition_payload)
    summary = _summarize(records)
    hypotheses: list[FastPolicyHypothesis] = []

    candidate_gap = _records_with_failure(records, "candidate_generation_missed_gold")
    if len(candidate_gap) >= min_support:
        hypotheses.append(
            _make_policy(
                kind="source_binding",
                action="deterministic_option_coverage_and_required_term_source_bundle",
                trigger_terms=[
                    "multiple_choice",
                    "source_bearing",
                    "option_sweep_gap",
                    "missing_gold_candidate",
                ],
                support_rows=candidate_gap,
                all_records=records,
                max_evidence_rows=max_evidence_rows,
                notes=[
                    "Candidate generation missed the after-run gold hash often enough to justify a shadow source-binding policy.",
                    "Keep this out of live selection until a fresh unseen cohort shows improved candidate coverage without regression.",
                ],
            )
        )

    source_directness = [
        row
        for row in records
        if _contains_any(
            str(row.get("failure_bucket") or ""),
            (
                "source",
                "direct_source_insufficient",
                "generic",
                "directness",
                "SourceEvidenceMissing",
            ),
        )
    ]
    if len(source_directness) >= min_support:
        hypotheses.append(
            _make_policy(
                kind="source_binding",
                action="candidate_specific_direct_relation_span_bundle",
                trigger_terms=[
                    "source_verifier_generic",
                    "direct_relation_span",
                    "required_term_completion",
                    "candidate_specific_witness",
                ],
                support_rows=source_directness,
                all_records=records,
                max_evidence_rows=max_evidence_rows,
                notes=[
                    "Recurring source/directness failures point to answer-bearing span coverage rather than looser gates.",
                    "Require same-row or same-span value/relation evidence before any future promotion.",
                ],
            )
        )

    no_fallback_rows = [
        row
        for row in records
        if "no_fallback" in str(row.get("failure_bucket") or "")
        or (row.get("path_hashes") or {}).get("verified_or_abstain_gate_status") == "no_fallback"
    ]
    preserve_original_no_direct_rows = [
        row
        for row in records
        if "preserve_original_no_direct_fallback" in str(row.get("failure_bucket") or "")
        or (
            (row.get("path_hashes") or {}).get("verified_or_abstain_gate_fallback_policy")
            == "preserve_original_selection_no_direct_fallback"
        )
    ]
    fallback_gap_rows = no_fallback_rows + [
        row for row in preserve_original_no_direct_rows if row not in no_fallback_rows
    ]
    if len(fallback_gap_rows) >= min_support:
        hypotheses.append(
            _make_policy(
                kind="fallback_policy",
                action="preserve_slow_baseline_when_verified_gate_has_no_direct_candidate",
                trigger_terms=[
                    "verified_or_abstain",
                    "no_fallback",
                    "no_direct_candidate",
                    "preserve_original",
                    "slow_baseline",
                ],
                support_rows=fallback_gap_rows,
                all_records=records,
                max_evidence_rows=max_evidence_rows,
                expected_harm=0.1,
                notes=[
                    "No-direct-candidate states must not force a weak source path into final selection.",
                    "This is a safety policy; it still needs budget-matched unseen validation.",
                ],
            )
        )

    if summary["latency_mean_seconds"] and summary["latency_mean_seconds"] > 300:
        hypotheses.append(
            _make_policy(
                kind="solver_lane",
                action="batch_or_cap_source_directness_calls_before_slow_baseline_fallback",
                trigger_terms=[
                    "source_directness",
                    "model_call_budget",
                    "latency_tail",
                    "batch_verifier",
                ],
                support_rows=records,
                all_records=records,
                max_evidence_rows=max_evidence_rows,
                expected_harm=0.05,
                notes=[
                    "Mean latency is high enough that source/directness stages need a budgeted lane.",
                    "Promotion should require equal or better accuracy plus lower latency or call count.",
                ],
            )
        )

    hypothesis_payloads = [hyp.to_dict() for hyp in hypotheses]
    return {
        "miner_version": POLICY_MINER_VERSION,
        "transition_dataset_version": TRANSITION_DATASET_VERSION,
        "fast_policy_memory_version": FAST_POLICY_MEMORY_VERSION,
        "summary": summary,
        "hypotheses": hypothesis_payloads,
        "promotion_gate_blockers": _promotion_gate_blockers(records, summary),
        "raw_content_persisted": False,
    }


def mine_fast_policy_hypotheses_from_paths(
    paths: Sequence[str | Path],
    *,
    min_support: int = 2,
    max_evidence_rows: int = 5,
) -> dict[str, Any]:
    dataset = build_transition_dataset_from_paths(paths)
    return mine_fast_policy_hypotheses(
        dataset,
        min_support=min_support,
        max_evidence_rows=max_evidence_rows,
    )


def _make_policy(
    *,
    kind: str,
    action: str,
    trigger_terms: list[str],
    support_rows: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    max_evidence_rows: int,
    expected_harm: float | None = None,
    notes: list[str] | None = None,
) -> FastPolicyHypothesis:
    support_count = len(support_rows)
    total = max(1, len(all_records))
    wrong_count = max(1, sum(1 for row in all_records if row.get("correct") is False))
    utility = min(0.85, max(0.05, support_count / wrong_count))
    harm = expected_harm
    if harm is None:
        harm = 0.25 if any(row.get("correct") is True for row in all_records) else 0.15
    policy_id = f"hle_{kind}_{stable_hash({'action': action, 'support': support_count, 'total': total})[:12]}"
    return FastPolicyHypothesis(
        id=policy_id,
        kind=kind,
        action=action,
        trigger_terms=trigger_terms,
        anti_trigger_terms=["debug_seed_only", "seen_cohort_only", "gold_label_known"],
        expected_utility=round(utility, 4),
        expected_harm=round(harm, 4),
        evidence_rows=[_redacted_evidence_row(row) for row in support_rows[:max_evidence_rows]],
        failure_rows=[_redacted_evidence_row(row) for row in _failure_rows_for_policy(support_rows)[:max_evidence_rows]],
        promotion_status="candidate",
        fallback_behavior="preserve_slow_baseline",
        source_refs=["GPT_advice.md", "self_evo_continual_reference_bundle_20260707"],
        notes=notes or [],
    )


def _failure_rows_for_policy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("correct") is False]


def _redacted_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_ref_hash": stable_hash({
            "question_id": row.get("question_id"),
            "question_hash": row.get("question_hash"),
            "action": row.get("action"),
        }),
        "question_id": row.get("question_id"),
        "question_hash": row.get("question_hash"),
        "action": row.get("action"),
        "category": row.get("category"),
        "domain": row.get("domain"),
        "correct": row.get("correct"),
        "failure_bucket": row.get("failure_bucket"),
        "selected_label_hash": row.get("selected_label_hash"),
        "gold_after_run_label_hash": row.get("gold_after_run_label_hash"),
        "verified_or_abstain_gate_status": (row.get("path_hashes") or {}).get("verified_or_abstain_gate_status"),
        "latency_seconds": row.get("latency_seconds"),
        "cost": row.get("cost"),
        "raw_content_persisted": False,
    }


def _records(payload: dict[str, Any] | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("records")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
        return []
    return [dict(row) for row in payload if isinstance(row, dict)]


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct_count = sum(1 for row in records if row.get("correct") is True)
    known_correct = sum(1 for row in records if row.get("correct") is not None)
    failure_buckets = Counter(str(row.get("failure_bucket") or "none") for row in records)
    actions = Counter(str(row.get("action") or "unknown") for row in records)
    gate_statuses = Counter(
        str((row.get("path_hashes") or {}).get("verified_or_abstain_gate_status") or "unknown")
        for row in records
    )
    latencies = [float(row["latency_seconds"]) for row in records if isinstance(row.get("latency_seconds"), (int, float))]
    costs = [float(row["cost"]) for row in records if isinstance(row.get("cost"), (int, float))]
    return {
        "record_count": total,
        "known_correct_count": known_correct,
        "correct_count": correct_count,
        "accuracy": round(correct_count / known_correct, 4) if known_correct else None,
        "failure_buckets": dict(failure_buckets),
        "action_counts": dict(actions),
        "verified_or_abstain_gate_status_counts": dict(gate_statuses),
        "no_fallback_count": sum(
            1
            for row in records
            if "no_fallback" in str(row.get("failure_bucket") or "")
            or (row.get("path_hashes") or {}).get("verified_or_abstain_gate_status") == "no_fallback"
        ),
        "latency_mean_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "latency_sum_seconds": round(sum(latencies), 4) if latencies else 0.0,
        "cost_sum": round(sum(costs), 4) if costs else 0.0,
        "raw_content_persisted": False,
    }


def _promotion_gate_blockers(records: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if summary["record_count"] < 24:
        blockers.append("insufficient_unseen_transition_rows_min_24")
    if summary["no_fallback_count"] > 0:
        blockers.append("no_fallback_present")
    if not _has_control_label(records):
        blockers.append("missing_fair_control_or_split_metadata")
    if summary["accuracy"] is None:
        blockers.append("missing_after_run_correctness")
    return blockers


def _has_control_label(records: list[dict[str, Any]]) -> bool:
    actions = {str(row.get("action") or "") for row in records}
    return any(action in {"raw", "HippoRAG", "raw_fallback", "hipporag"} for action in actions)


def _records_with_failure(records: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    return [row for row in records if needle in str(row.get("failure_bucket") or "")]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Transition dataset or HLE artifact JSON paths")
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-evidence-rows", type=int, default=5)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    payloads = []
    hle_artifact_paths = []
    for path in args.paths:
        artifact = Path(path).expanduser()
        loaded = json.loads(artifact.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("dataset_version") == TRANSITION_DATASET_VERSION:
            payloads.extend(_records(loaded))
        else:
            hle_artifact_paths.append(artifact)
    if hle_artifact_paths:
        converted = build_transition_dataset_from_paths(hle_artifact_paths)
        payloads.extend(_records(converted))
    report = mine_fast_policy_hypotheses(
        payloads,
        min_support=args.min_support,
        max_evidence_rows=args.max_evidence_rows,
    )
    text = json.dumps(report, ensure_ascii=True, indent=2 if args.pretty else None, sort_keys=args.pretty)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sanitized HLE medical-order/source diagnostic scan.

This scanner is intentionally diagnostic-only.  It reads local HLE question
text and metadata, but never reads the gold answer field and never persists raw
question, option, or source text.  Use it to find fresh patient-order/guideline
cohorts and to audit whether local source rows support strict comparator lanes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .autonomy_journal import stable_hash
from .hle_guideline_pair_comparators import (
    fe_hyperfine_pair_binding_detail,
    medical_guideline_permutation_ordering_detail,
)
from .hle_smoke_eval import (
    _has_image_payload,
    _load_hle_test_dataset,
    _local_evidence_corpus_search,
    _split_multiple_choice_question,
    apply_hle_offline_defaults_to_environ,
)


DEFAULT_CORPUS_PATH = (
    Path("phase four")
    / "assumption_graph"
    / "local_evidence_corpus"
    / "hle_seed70_678_guideline_fulltext_20260706.jsonl"
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_int_csv(raw: str) -> set[int]:
    out: set[int] = set()
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.add(int(chunk))
        except ValueError:
            continue
    return out


def _safe_hash(value: Any) -> str:
    return stable_hash(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload.get("summary", {})
    lines = [
        "# HLE Medical-Order/Guideline Source Scan",
        "",
        f"- eval_id: `{payload.get('eval_id', '')}`",
        f"- timestamp_utc: `{payload.get('timestamp_utc', '')}`",
        "- raw_content_persisted: `false`",
        "- gold_answer_accessed: `false`",
        f"- scanned_count: `{summary.get('scanned_count', 0)}`",
        f"- broad_medical_order_candidate_count: `{summary.get('broad_medical_order_candidate_count', 0)}`",
        f"- medical_candidate_count: `{summary.get('medical_candidate_count', 0)}`",
        f"- medical_unique_exact_count: `{summary.get('medical_unique_exact_count', 0)}`",
        f"- fe_candidate_count: `{summary.get('fe_candidate_count', 0)}`",
        f"- fe_direct_candidate_count: `{summary.get('fe_direct_candidate_count', 0)}`",
        f"- fe_partial_candidate_count: `{summary.get('fe_partial_candidate_count', 0)}`",
        "",
        "## Status Counts",
        "",
        "```json",
        json.dumps(
            {
                "medical_status_counts": summary.get("medical_status_counts", {}),
                "medical_reason_counts": summary.get("medical_reason_counts", {}),
                "fe_status_counts": summary.get("fe_status_counts", {}),
                "fe_reason_counts": summary.get("fe_reason_counts", {}),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        str(payload.get("interpretation") or ""),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _problem_from_hle_row_without_answer(
    row: dict[str, Any],
    *,
    scanned_index: int,
    skipped_before: int,
) -> dict[str, Any]:
    question = str(row.get("question") or "")
    return {
        "id_hash": _safe_hash({"hle_id": row.get("id")}),
        "question_hash": _safe_hash({"question": question}),
        "category": str(row.get("category") or ""),
        "raw_subject": str(row.get("raw_subject") or ""),
        "answer_type": str(row.get("answer_type") or ""),
        "scanned_index": scanned_index,
        "seed_offset_compat": scanned_index - 1,
        "skipped_before": skipped_before,
        "_question": question,
    }


def _patient_order(option_text: str) -> list[str]:
    return list(dict.fromkeys(
        match.group(1)
        for match in re.finditer(
            r"\bPatient\s+(\d{1,2})\b",
            str(option_text or ""),
            flags=re.IGNORECASE,
        )
    ))


def _looks_like_medical_order_problem(
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
) -> tuple[bool, str]:
    domain = f"{problem.get('category', '')} {problem.get('raw_subject', '')}".lower()
    if not any(token in domain for token in ("medicine", "clinical", "biology/medicine")):
        return False, "non_medical_domain"
    stem_lower = str(stem or "").lower()
    if len(re.findall(r"\bPatient\s+\d{1,2}\s*:", stem, flags=re.IGNORECASE)) < 2:
        return False, "missing_patient_descriptions"
    if not re.search(
        r"\b(prioriti[sz]e|rank|order|sequence|surgical|operative|indication|guideline|classification|treatment|management)\b",
        stem_lower,
        flags=re.IGNORECASE,
    ):
        return False, "missing_order_or_guideline_cue"
    permutation_options = sum(1 for option in options.values() if len(_patient_order(option)) >= 2)
    if permutation_options < 2:
        return False, "missing_patient_permutation_options"
    return True, "medical_patient_order_guideline_candidate"


def _looks_like_broad_medical_order_problem(
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
) -> tuple[bool, str]:
    domain = f"{problem.get('category', '')} {problem.get('raw_subject', '')}".lower()
    if not any(token in domain for token in ("medicine", "clinical", "biology/medicine")):
        return False, "non_medical_domain"
    text = " ".join([str(stem or ""), *list(options.values())]).lower()
    if "patient" not in text:
        return False, "missing_patient_context"
    if not re.search(
        r"\b(prioriti[sz]e|rank|order|sequence|surgical|operative|indication|guideline|classification|treatment|management|reasonable)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return False, "missing_order_or_guideline_cue"
    return True, "broad_medical_order_guideline_candidate"


def _looks_like_fe_hyperfine_problem(stem: str, options: dict[str, str]) -> tuple[bool, str]:
    stem_lower = str(stem or "").lower()
    if not ("hyperfine" in stem_lower and ("mossbauer" in stem_lower or "57fe" in stem_lower)):
        return False, "not_fe_hyperfine_stem"
    option_text = " ".join(options.values())
    if not re.search(r"\bFe\s*\(\s*[IVX]+\s*\)", option_text, flags=re.IGNORECASE):
        return False, "missing_fe_oxidation_options"
    return True, "fe_hyperfine_pair_binding_candidate"


def _medical_evidence_query(stem: str, option_text: str) -> str:
    return " ".join([
        str(stem or "")[:1200],
        str(option_text or "")[:500],
        "thoracolumbar spine trauma surgical indication TLICS morphology neurologic posterior ligamentous complex guideline operative",
    ])


def _fe_evidence_query(stem: str, option_text: str) -> str:
    return " ".join([
        str(stem or "")[:800],
        str(option_text or "")[:500],
        "57Fe Mossbauer hyperfine field oxidation spin geometry largest highest comparison",
    ])


def _sanitize_medical_detail(detail: dict[str, Any], *, option_label: str, option_text: str, row_count: int) -> dict[str, Any]:
    return {
        "option_label": option_label,
        "option_hash": _safe_hash({"option_label": option_label, "option_text": option_text}),
        "option_text_hash": _safe_hash({"option_text": option_text}),
        "row_count": row_count,
        "status": detail.get("status"),
        "reason": detail.get("reason"),
        "candidate_exact_expected_order": bool(detail.get("candidate_exact_expected_order")),
        "candidate_guideline_order_score": detail.get("candidate_guideline_order_score"),
        "candidate_rank_penalty": detail.get("candidate_rank_penalty"),
        "ambiguous_patient_scores": bool(detail.get("ambiguous_patient_scores")),
        "patient_count": detail.get("patient_count"),
        "candidate_order_hash": detail.get("candidate_order_hash"),
        "expected_order_hash": detail.get("expected_order_hash"),
        "source_hashes": list(detail.get("source_hashes") or [])[:5],
        "score_rows": list(detail.get("score_rows") or [])[:8],
    }


def _sanitize_fe_detail(detail: dict[str, Any], *, option_label: str, option_text: str, row_count: int) -> dict[str, Any]:
    return {
        "option_label": option_label,
        "option_hash": _safe_hash({"option_label": option_label, "option_text": option_text}),
        "option_text_hash": _safe_hash({"option_text": option_text}),
        "row_count": row_count,
        "status": detail.get("status"),
        "reason": detail.get("reason"),
        "option_feature_hash": detail.get("option_feature_hash"),
        "oxidation": detail.get("oxidation"),
        "spin_hash": detail.get("spin_hash"),
        "geometry_hash": detail.get("geometry_hash"),
        "source_hashes": list(detail.get("source_hashes") or [])[:5],
        "partial_pair_binding_row_count": int(detail.get("partial_pair_binding_row_count") or 0),
        "direct_pair_binding_row_count": int(detail.get("direct_pair_binding_row_count") or 0),
        "missing_geometry_row_count": int(detail.get("missing_geometry_row_count") or 0),
        "missing_relation_row_count": int(detail.get("missing_relation_row_count") or 0),
        "best_pair_binding_score": detail.get("best_pair_binding_score"),
        "row_details": list(detail.get("row_details") or [])[:5],
    }


def _scan_medical_candidate(problem: dict[str, Any], stem: str, options: dict[str, str], *, row_limit: int) -> dict[str, Any]:
    option_diagnostics: list[dict[str, Any]] = []
    exact_hashes: list[str] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for label, option_text in sorted(options.items()):
        rows = _local_evidence_corpus_search(
            _medical_evidence_query(stem, option_text),
            problem=problem,
            limit=row_limit,
        )
        detail = medical_guideline_permutation_ordering_detail(
            stem=stem,
            option_text=option_text,
            rows=rows,
        )
        sanitized = _sanitize_medical_detail(
            detail,
            option_label=label,
            option_text=option_text,
            row_count=len(rows),
        )
        option_diagnostics.append(sanitized)
        status_counts[str(sanitized.get("status") or "")] += 1
        reason_counts[str(sanitized.get("reason") or "")] += 1
        if sanitized["candidate_exact_expected_order"]:
            exact_hashes.append(str(sanitized["option_hash"]))
    return {
        "family": "medical_guideline_permutation_ordering",
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "category_hash": _safe_hash({"category": problem.get("category", "")}),
        "raw_subject_hash": _safe_hash({"raw_subject": problem.get("raw_subject", "")}),
        "answer_type": problem.get("answer_type", ""),
        "scanned_index": problem["scanned_index"],
        "seed_offset_compat": problem["seed_offset_compat"],
        "option_count": len(options),
        "unique_exact_option_hash": exact_hashes[0] if len(set(exact_hashes)) == 1 else "",
        "exact_option_hashes": sorted(set(exact_hashes)),
        "option_diagnostics": option_diagnostics,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "raw_content_persisted": False,
        "gold_answer_accessed": False,
    }


def _scan_broad_medical_order_candidate(
    problem: dict[str, Any],
    stem: str,
    options: dict[str, str],
    *,
    strict_applicable: bool,
    strict_reason: str,
) -> dict[str, Any]:
    text = " ".join([str(stem or ""), *list(options.values())])
    option_texts = list(options.values())
    cue_matches = sorted(set(re.findall(
        r"prioriti[sz]e|rank|order|sequence|surgical|operative|indication|guideline|classification|treatment|management|reasonable",
        text.lower(),
    )))
    return {
        "family": "broad_medical_order_guideline",
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "category_hash": _safe_hash({"category": problem.get("category", "")}),
        "raw_subject_hash": _safe_hash({"raw_subject": problem.get("raw_subject", "")}),
        "answer_type": problem.get("answer_type", ""),
        "scanned_index": problem["scanned_index"],
        "seed_offset_compat": problem["seed_offset_compat"],
        "option_count": len(options),
        "question_text_hash": _safe_hash({"question": problem.get("_question", "")}),
        "cue_hashes": [_safe_hash({"cue": cue}) for cue in cue_matches],
        "cue_count": len(cue_matches),
        "stem_patient_number_colon_count": len(re.findall(r"\bPatient\s+\d{1,2}\s*:", stem, re.IGNORECASE)),
        "stem_patient_word_count": len(re.findall(r"\bpatient\b", stem, re.IGNORECASE)),
        "option_patient_word_counts": [
            len(re.findall(r"\bpatient\b", option, re.IGNORECASE))
            for option in option_texts[:24]
        ],
        "option_numbered_patient_counts": [
            len(re.findall(r"\bPatient\s+\d{1,2}\b", option, re.IGNORECASE))
            for option in option_texts[:24]
        ],
        "option_order_symbol_flags": [
            bool(re.search(r"(?:>|<|then|before|after|first|second|third|,)", option, re.IGNORECASE))
            for option in option_texts[:24]
        ],
        "strict_patient_permutation_comparator_applicable": strict_applicable,
        "strict_patient_permutation_detector_reason": strict_reason,
        "raw_content_persisted": False,
        "gold_answer_accessed": False,
    }


def _scan_fe_candidate(problem: dict[str, Any], stem: str, options: dict[str, str], *, row_limit: int) -> dict[str, Any]:
    option_diagnostics: list[dict[str, Any]] = []
    direct_hashes: list[str] = []
    partial_hashes: list[str] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for label, option_text in sorted(options.items()):
        rows = _local_evidence_corpus_search(
            _fe_evidence_query(stem, option_text),
            problem=problem,
            limit=row_limit,
        )
        detail = fe_hyperfine_pair_binding_detail(
            stem=stem,
            option_text=option_text,
            rows=rows,
        )
        sanitized = _sanitize_fe_detail(
            detail,
            option_label=label,
            option_text=option_text,
            row_count=len(rows),
        )
        option_diagnostics.append(sanitized)
        status_counts[str(sanitized.get("status") or "")] += 1
        reason_counts[str(sanitized.get("reason") or "")] += 1
        if int(sanitized["direct_pair_binding_row_count"]) > 0:
            direct_hashes.append(str(sanitized["option_hash"]))
        if int(sanitized["partial_pair_binding_row_count"]) > 0:
            partial_hashes.append(str(sanitized["option_hash"]))
    return {
        "family": "fe_hyperfine_pair_binding",
        "problem_id_hash": problem["id_hash"],
        "question_hash": problem["question_hash"],
        "category_hash": _safe_hash({"category": problem.get("category", "")}),
        "raw_subject_hash": _safe_hash({"raw_subject": problem.get("raw_subject", "")}),
        "answer_type": problem.get("answer_type", ""),
        "scanned_index": problem["scanned_index"],
        "seed_offset_compat": problem["seed_offset_compat"],
        "option_count": len(options),
        "unique_direct_option_hash": direct_hashes[0] if len(set(direct_hashes)) == 1 else "",
        "direct_option_hashes": sorted(set(direct_hashes)),
        "partial_option_hashes": sorted(set(partial_hashes)),
        "option_diagnostics": option_diagnostics,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "raw_content_persisted": False,
        "gold_answer_accessed": False,
    }


def _configure_offline_source_env(root: Path, corpus_paths: str) -> list[Path]:
    apply_hle_offline_defaults_to_environ(os.environ)
    resolved_paths: list[Path] = []
    if corpus_paths.strip():
        for chunk in corpus_paths.split(os.pathsep):
            for item in chunk.split(","):
                text = item.strip()
                if not text:
                    continue
                path = Path(text)
                if not path.is_absolute():
                    path = root / path
                resolved_paths.append(path)
    elif (root / DEFAULT_CORPUS_PATH).exists():
        resolved_paths.append(root / DEFAULT_CORPUS_PATH)
    if resolved_paths:
        os.environ["HLE_EVIDENCE_SOURCE_CORPUS_PATHS"] = os.pathsep.join(str(path) for path in resolved_paths)
    os.environ["HLE_DISABLE_EVIDENCE_CACHE_CORPUS"] = "1"
    os.environ["HLE_EVIDENCE_SOURCE_CACHE_ONLY"] = "1"
    os.environ["HLE_SOURCE_SEARCH_CACHE_ONLY"] = "1"
    os.environ["HLE_DISABLE_LIVE_SOURCE_SEARCH"] = "1"
    os.environ["HLE_ALLOW_LIVE_SOURCE_SEARCH"] = "0"
    return resolved_paths


def scan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    corpus_paths = _configure_offline_source_env(root, str(args.corpus_paths or ""))
    exclude_offsets = _parse_int_csv(args.exclude_seed_offsets)
    force_offsets = _parse_int_csv(args.force_seed_offsets)
    dataset = _load_hle_test_dataset()
    skipped_before = 0
    scanned_count = 0
    skipped_counts: Counter[str] = Counter()
    medical_records: list[dict[str, Any]] = []
    broad_medical_records: list[dict[str, Any]] = []
    fe_records: list[dict[str, Any]] = []
    medical_detector_counts: Counter[str] = Counter()
    broad_medical_detector_counts: Counter[str] = Counter()
    fe_detector_counts: Counter[str] = Counter()

    for scanned_index, row in enumerate(dataset, start=1):
        seed_offset_compat = scanned_index - 1
        if force_offsets:
            if seed_offset_compat not in force_offsets:
                continue
        else:
            if scanned_index <= args.start_offset:
                continue
            if scanned_index > args.max_scan:
                break
            if seed_offset_compat in exclude_offsets:
                skipped_counts["excluded_seed_offset"] += 1
                continue
        scanned_count += 1
        if _has_image_payload(row):
            skipped_before += 1
            skipped_counts["image_payload"] += 1
            continue
        if str(row.get("answer_type") or "") != "multipleChoice":
            skipped_counts["non_multiple_choice"] += 1
            continue
        if not str(row.get("question") or "").strip():
            skipped_counts["missing_question"] += 1
            continue
        problem = _problem_from_hle_row_without_answer(
            row,
            scanned_index=scanned_index,
            skipped_before=skipped_before,
        )
        stem, options = _split_multiple_choice_question(problem)
        if len(options) < 2:
            skipped_counts["unparsed_options"] += 1
            continue
        medical_ok, medical_reason = _looks_like_medical_order_problem(problem, stem, options)
        medical_detector_counts[medical_reason] += 1
        broad_medical_ok, broad_medical_reason = _looks_like_broad_medical_order_problem(problem, stem, options)
        broad_medical_detector_counts[broad_medical_reason] += 1
        if (
            args.scan_medical
            and broad_medical_ok
            and len(broad_medical_records) < args.broad_limit
        ):
            broad_medical_records.append(_scan_broad_medical_order_candidate(
                problem,
                stem,
                options,
                strict_applicable=medical_ok,
                strict_reason=medical_reason,
            ))
        if args.scan_medical and medical_ok and len(medical_records) < args.limit:
            medical_records.append(_scan_medical_candidate(problem, stem, options, row_limit=args.row_limit))
        fe_ok, fe_reason = _looks_like_fe_hyperfine_problem(stem, options)
        fe_detector_counts[fe_reason] += 1
        if args.scan_fe and fe_ok and len(fe_records) < args.fe_limit:
            fe_records.append(_scan_fe_candidate(problem, stem, options, row_limit=args.row_limit))
        if (
            not force_offsets
            and len(medical_records) >= args.limit
            and len(broad_medical_records) >= args.broad_limit
            and (not args.scan_fe or len(fe_records) >= args.fe_limit)
        ):
            break

    medical_status_counts: Counter[str] = Counter()
    medical_reason_counts: Counter[str] = Counter()
    for record in medical_records:
        medical_status_counts.update(record.get("status_counts") or {})
        medical_reason_counts.update(record.get("reason_counts") or {})
    fe_status_counts: Counter[str] = Counter()
    fe_reason_counts: Counter[str] = Counter()
    for record in fe_records:
        fe_status_counts.update(record.get("status_counts") or {})
        fe_reason_counts.update(record.get("reason_counts") or {})

    summary = {
        "scanned_count": scanned_count,
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "broad_medical_detector_counts": dict(sorted(broad_medical_detector_counts.items())),
        "broad_medical_order_candidate_count": len(broad_medical_records),
        "broad_medical_strict_applicable_count": sum(
            1
            for record in broad_medical_records
            if record.get("strict_patient_permutation_comparator_applicable")
        ),
        "medical_detector_counts": dict(sorted(medical_detector_counts.items())),
        "fe_detector_counts": dict(sorted(fe_detector_counts.items())),
        "medical_candidate_count": len(medical_records),
        "medical_unique_exact_count": sum(1 for record in medical_records if record.get("unique_exact_option_hash")),
        "medical_status_counts": dict(sorted(medical_status_counts.items())),
        "medical_reason_counts": dict(sorted(medical_reason_counts.items())),
        "fe_candidate_count": len(fe_records),
        "fe_direct_candidate_count": sum(1 for record in fe_records if record.get("unique_direct_option_hash")),
        "fe_partial_candidate_count": sum(1 for record in fe_records if record.get("partial_option_hashes")),
        "fe_status_counts": dict(sorted(fe_status_counts.items())),
        "fe_reason_counts": dict(sorted(fe_reason_counts.items())),
    }
    interpretation = (
        "Diagnostic-only scan. Medical unique-exact rows indicate a source-backed "
        "ordering signal candidate, not an accuracy claim. Fe partial rows are "
        "reported but non-promotable; only unique_direct_option_hash may be "
        "considered for a future strict selector."
    )
    return {
        "eval_id": args.eval_id,
        "timestamp_utc": _utc_timestamp(),
        "raw_content_persisted": False,
        "gold_answer_accessed": False,
        "decision_path_uses_gold": False,
        "offline_source_only": True,
        "dataset_access_mode": "local_hle_no_answer_field_read",
        "source_corpus_path_hashes": [_safe_hash({"path": str(path)}) for path in corpus_paths],
        "config": {
            "start_offset": args.start_offset,
            "max_scan": args.max_scan,
            "limit": args.limit,
            "scan_medical": bool(args.scan_medical),
            "scan_fe": bool(args.scan_fe),
            "fe_limit": args.fe_limit,
            "row_limit": args.row_limit,
            "exclude_seed_offsets": sorted(exclude_offsets),
            "force_seed_offsets": sorted(force_offsets),
        },
        "summary": summary,
        "broad_medical_order_records": broad_medical_records,
        "medical_records": medical_records,
        "fe_records": fe_records,
        "interpretation": interpretation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan local HLE for sanitized medical-order/source diagnostics.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hle_medical_order_unseen_scan_20260706")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-scan", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--broad-limit", type=int, default=24)
    parser.add_argument("--row-limit", type=int, default=5)
    parser.add_argument("--exclude-seed-offsets", default="")
    parser.add_argument("--force-seed-offsets", default="")
    parser.add_argument("--scan-medical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan-fe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fe-limit", type=int, default=4)
    parser.add_argument("--corpus-paths", default="")
    parser.add_argument("--out", default="phase four/assumption_graph/hle_medical_order_unseen_scan_20260706.json")
    parser.add_argument("--md-out", default="reconstruction/md/hle_medical_order_unseen_scan_20260706.md")
    args = parser.parse_args()

    payload = scan(args)
    root = Path(args.root).resolve()
    _write_json(root / args.out, payload)
    _write_md(root / args.md_out, payload)
    print(json.dumps(payload["summary"], ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

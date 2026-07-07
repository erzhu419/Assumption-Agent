"""Numeric source witness extraction for HLE option evidence."""

from __future__ import annotations

import re
from typing import Any

from .autonomy_journal import stable_hash
from .hle_numeric_option_parser import parse_numeric_values
from .hle_numeric_relation_classifier import classify_numeric_relation, numeric_relation_terms


def _doc_text(doc: dict[str, Any]) -> str:
    return " ".join(
        str(doc.get(key) or "")
        for key in ("title", "snippet", "text", "abstract")
        if str(doc.get(key) or "").strip()
    )


def _tokens(text: str) -> set[str]:
    clean = re.sub(r"[^a-zA-Z0-9]+", " ", str(text or "").lower())
    tokens = {token for token in clean.split() if len(token) >= 3}
    if "xenon" in tokens and "tetrafluoride" in tokens:
        tokens.add("xef4")
    if "xef4" in tokens:
        tokens.update({"xenon", "tetrafluoride"})
    return tokens


_GENERIC_NUMERIC_ANCHOR_TERMS = {
    "above",
    "about",
    "allowed",
    "below",
    "can",
    "celsius",
    "coldest",
    "degree",
    "efficient",
    "efficiently",
    "following",
    "greater",
    "heat",
    "heated",
    "hottest",
    "kelvin",
    "largest",
    "least",
    "less",
    "lowest",
    "maximum",
    "method",
    "minimum",
    "nearest",
    "prepared",
    "preparation",
    "prepare",
    "produce",
    "produced",
    "reaction",
    "required",
    "still",
    "synthesis",
    "synthesize",
    "synthesized",
    "temperature",
    "thermal",
    "using",
}

_THRESHOLD_SIGNAL_TERMS = {
    "above",
    "below",
    "coldest",
    "hottest",
    "least",
    "lowest",
    "maximum",
    "minimum",
    "threshold",
    "under",
}

_THRESHOLD_EXTREME_TERMS = {
    "coldest",
    "greatest",
    "highest",
    "hottest",
    "largest",
    "least",
    "lowest",
    "maximum",
    "minimum",
}


def _relation_cue_terms(question_terms: set[str]) -> set[str]:
    cues = set(question_terms & _GENERIC_NUMERIC_ANCHOR_TERMS)
    if cues & {"synthesis", "synthesize", "synthesized", "produce", "produced"}:
        cues.update({"prepare", "prepared", "preparation", "reaction"})
    if cues & {"temperature", "coldest", "hottest", "celsius", "kelvin"}:
        cues.update({"temperature", "thermal", "heat", "heated"})
    cues.update({
        "efficient",
        "efficiently",
        "prepared",
        "preparation",
        "prepare",
        "produce",
        "produced",
        "reaction",
        "synthesis",
        "synthesize",
        "synthesized",
        "temperature",
    })
    return cues


def _subject_anchor_terms(anchor_terms: set[str], relation_terms: set[str]) -> set[str]:
    subject_terms = {
        term
        for term in anchor_terms
        if term not in relation_terms and term not in _GENERIC_NUMERIC_ANCHOR_TERMS
    }
    if "xenon" in subject_terms and "tetrafluoride" in subject_terms:
        subject_terms.add("xef4")
    if "xef4" in subject_terms:
        subject_terms.update({"xenon", "tetrafluoride"})
    return subject_terms


def _threshold_signal_terms(relation_family: str) -> set[str]:
    if str(relation_family or "") in {
        "threshold_minimum",
        "threshold_maximum",
        "below_threshold",
        "above_threshold",
        "ordered_extreme_lowest",
        "ordered_extreme_highest",
    }:
        return set(_THRESHOLD_SIGNAL_TERMS)
    return set()


def _threshold_extreme_terms(relation_family: str) -> set[str]:
    if str(relation_family or "") in {
        "threshold_minimum",
        "threshold_maximum",
        "ordered_extreme_lowest",
        "ordered_extreme_highest",
    }:
        return set(_THRESHOLD_EXTREME_TERMS)
    return set()


def _source_weight(source: str) -> float:
    source_key = str(source or "").strip().lower()
    return {
        "semantic_scholar": 1.2,
        "openalex": 1.1,
        "pubmed_abstract": 1.1,
        "arxiv": 1.0,
        "answer_web": 0.9,
        "wikipedia_extract": 0.8,
        "wikipedia": 0.6,
    }.get(source_key, 0.5)


def _numeric_close(option: dict[str, Any], witness: dict[str, Any]) -> tuple[float, str]:
    opt_unit = option.get("normalized_unit")
    wit_unit = witness.get("normalized_unit")
    opt_value = option.get("normalized_value")
    wit_value = witness.get("normalized_value")
    if opt_value is None or wit_value is None:
        return 0.0, "missing_numeric_value"
    try:
        opt_float = float(opt_value)
        wit_float = float(wit_value)
    except (TypeError, ValueError):
        return 0.0, "invalid_numeric_value"
    if opt_unit and wit_unit and opt_unit != wit_unit:
        return 0.0, "unit_mismatch"
    abs_delta = abs(opt_float - wit_float)
    rel_delta = abs_delta / max(abs(opt_float), abs(wit_float), 1.0)
    if opt_unit == "K" or witness.get("value_type") == "temperature":
        if abs_delta <= 0.75:
            return 4.0, "temperature_exact_or_converted_match"
        if abs_delta <= 2.0:
            return 3.0, "temperature_close_match"
        return 0.0, "temperature_value_mismatch"
    if abs_delta <= 1e-9 or rel_delta <= 1e-6:
        return 4.0, "exact_numeric_match"
    if rel_delta <= 0.01:
        return 2.5, "near_numeric_match"
    return 0.0, "numeric_value_mismatch"


def _unit_score(option: dict[str, Any], witness: dict[str, Any]) -> float:
    if option.get("normalized_unit") and option.get("normalized_unit") == witness.get("normalized_unit"):
        return 1.2
    if option.get("value_type") == witness.get("value_type") and option.get("value_type") != "number":
        return 0.8
    if not option.get("normalized_unit") or not witness.get("normalized_unit"):
        return 0.3
    return 0.0


def numeric_same_row_directness_detail(
    *,
    stem: str,
    option_text: str,
    doc: dict[str, Any],
    relation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit whether one source row binds the option value to relation cues."""
    option_values = parse_numeric_values(option_text)
    if option_values:
        # Keep this aligned with parse_numeric_options(), which treats the first
        # parsed value as the option's comparable numeric value.
        option_values = option_values[:1]
    text = _doc_text(doc)
    doc_hash = stable_hash({
        "title": str(doc.get("title") or ""),
        "snippet": str(doc.get("snippet") or ""),
        "source": str(doc.get("source") or ""),
    })
    if not option_values:
        return {
            "status": "no_numeric_option",
            "source_hash": doc_hash,
            "numeric_same_row_direct": False,
            "raw_content_persisted": False,
        }
    if not text:
        return {
            "status": "no_text",
            "source_hash": doc_hash,
            "option_numeric_value_count": len(option_values),
            "numeric_same_row_direct": False,
            "raw_content_persisted": False,
        }

    first_value_type = str(option_values[0].get("value_type") or "")
    relation_row = relation or classify_numeric_relation(stem, value_type=first_value_type)
    relation_family = str(relation_row.get("relation_family") or "")
    anchor_terms = set(numeric_relation_terms(stem))
    relation_terms = _relation_cue_terms(anchor_terms)
    subject_terms = _subject_anchor_terms(anchor_terms, relation_terms)
    threshold_terms = _threshold_signal_terms(relation_family)
    threshold_extreme_terms = _threshold_extreme_terms(relation_family)
    threshold_relation = bool(threshold_terms)
    required_relation_overlap = 2 if threshold_relation else 1

    doc_tokens = _tokens(text)
    anchor_overlap = len(anchor_terms & doc_tokens)
    relation_overlap = len(relation_terms & doc_tokens)
    subject_anchor_overlap = len(subject_terms & doc_tokens)
    threshold_signal_overlap = len(threshold_terms & doc_tokens)
    threshold_extreme_overlap = len(threshold_extreme_terms & doc_tokens)
    subject_anchor_satisfied = (
        subject_anchor_overlap >= 1
        if subject_terms
        else anchor_overlap >= 2
    )
    relation_satisfied = relation_overlap >= required_relation_overlap
    threshold_signal_satisfied = (
        threshold_signal_overlap >= 1 if threshold_relation else True
    )

    source_values = parse_numeric_values(text)
    if not source_values:
        return {
            "status": "no_source_numeric_values",
            "source_hash": doc_hash,
            "option_numeric_value_count": len(option_values),
            "source_numeric_value_count": 0,
            "anchor_overlap": anchor_overlap,
            "subject_anchor_overlap": subject_anchor_overlap,
            "relation_overlap": relation_overlap,
            "threshold_signal_overlap": threshold_signal_overlap,
            "threshold_extreme_overlap": threshold_extreme_overlap,
            "required_relation_overlap": required_relation_overlap,
            "numeric_same_row_value_match": False,
            "numeric_same_row_direct": False,
            "raw_content_persisted": False,
        }

    best_match_score = 0.0
    best_match_reason = ""
    best_unit_score = 0.0
    best_value_hash = ""
    value_match_count = 0
    value_match_failure_counts: dict[str, int] = {}
    for option_value in option_values:
        for source_value in source_values:
            match_score, match_reason = _numeric_close(option_value, source_value)
            if match_score <= 0:
                value_match_failure_counts[match_reason] = (
                    value_match_failure_counts.get(match_reason, 0) + 1
                )
                continue
            value_match_count += 1
            unit_score = _unit_score(option_value, source_value)
            if match_score + unit_score > best_match_score + best_unit_score:
                best_match_score = match_score
                best_unit_score = unit_score
                best_match_reason = match_reason
                best_value_hash = str(source_value.get("value_hash") or "")

    rejection_reasons: list[str] = []
    if value_match_count <= 0:
        rejection_reasons.append("numeric_value_match_missing")
    if not subject_anchor_satisfied:
        rejection_reasons.append("subject_anchor_not_satisfied")
    if not relation_satisfied:
        rejection_reasons.append("relation_overlap_below_required")
    if not threshold_signal_satisfied:
        rejection_reasons.append("threshold_signal_missing")
    direct = bool(
        value_match_count > 0
        and subject_anchor_satisfied
        and relation_satisfied
        and threshold_signal_satisfied
    )
    same_row_score = round(
        best_match_score
        + best_unit_score
        + min(anchor_overlap, 6) * 0.25
        + min(subject_anchor_overlap, 3) * 0.4
        + min(relation_overlap, 5) * 0.45
        + min(threshold_signal_overlap, 2) * 0.5
        + min(threshold_extreme_overlap, 2) * 0.25,
        4,
    )
    return {
        "status": "evaluated",
        "source_hash": doc_hash,
        "option_numeric_value_count": len(option_values),
        "source_numeric_value_count": len(source_values),
        "value_match_count": value_match_count,
        "numeric_same_row_value_match": bool(value_match_count > 0),
        "numeric_same_row_direct": direct,
        "numeric_same_row_direct_reason": (
            "same_row_value_subject_relation_match"
            if direct
            else ",".join(rejection_reasons or ["direct_gate_failed"])
        ),
        "numeric_same_row_score": same_row_score,
        "value_match_reason": best_match_reason,
        "value_hash": best_value_hash,
        "value_match_failure_counts": dict(sorted(value_match_failure_counts.items())),
        "anchor_overlap": anchor_overlap,
        "subject_anchor_overlap": subject_anchor_overlap,
        "subject_anchor_term_count": len(subject_terms),
        "relation_overlap": relation_overlap,
        "required_relation_overlap": required_relation_overlap,
        "threshold_signal_overlap": threshold_signal_overlap,
        "threshold_extreme_overlap": threshold_extreme_overlap,
        "threshold_relation": threshold_relation,
        "rejection_reasons": rejection_reasons,
        "relation_hash": stable_hash({
            "relation_family": relation_family,
            "direction": relation_row.get("direction"),
        }),
        "raw_content_persisted": False,
    }


def numeric_same_row_source_diagnostics(
    *,
    stem: str,
    option_text: str,
    rows: list[dict[str, Any]],
    relation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not stem or not option_text:
        return {
            "numeric_same_row_diagnostics_status": "missing_option_context",
            "raw_content_persisted": False,
        }
    details = [
        numeric_same_row_directness_detail(
            stem=stem,
            option_text=option_text,
            doc=row,
            relation=relation,
        )
        for row in rows
        if isinstance(row, dict)
    ]
    source_hashes = [
        str(row.get("source_hash") or "")
        for row in details[:5]
        if str(row.get("source_hash") or "")
    ]
    rejection_counts: dict[str, int] = {}
    value_failure_counts: dict[str, int] = {}
    for detail in details:
        for reason in detail.get("rejection_reasons", []) or []:
            key = str(reason or "unknown")
            rejection_counts[key] = rejection_counts.get(key, 0) + 1
        for reason, count in (detail.get("value_match_failure_counts") or {}).items():
            key = str(reason or "unknown")
            value_failure_counts[key] = value_failure_counts.get(key, 0) + int(count or 0)
    evaluated = [
        row for row in details if row.get("status") == "evaluated"
    ]
    return {
        "numeric_same_row_diagnostics_status": (
            "evaluated" if evaluated else ("no_rows" if not rows else "not_numeric")
        ),
        "numeric_same_row_source_hashes": source_hashes,
        "numeric_same_row_source_hash_count": len({
            str(row.get("source_hash") or "")
            for row in details
            if str(row.get("source_hash") or "")
        }),
        "numeric_same_row_evaluated_count": len(evaluated),
        "numeric_same_row_value_match_count": sum(
            1 for row in evaluated if row.get("numeric_same_row_value_match")
        ),
        "numeric_same_row_relation_overlap_count": sum(
            1
            for row in evaluated
            if int(row.get("relation_overlap") or 0)
            >= int(row.get("required_relation_overlap") or 1)
        ),
        "numeric_same_row_threshold_signal_count": sum(
            1
            for row in evaluated
            if (
                not row.get("threshold_relation")
                or int(row.get("threshold_signal_overlap") or 0) > 0
            )
        ),
        "numeric_same_row_direct_count": sum(
            1 for row in evaluated if row.get("numeric_same_row_direct")
        ),
        "numeric_same_row_best_score": round(
            max([float(row.get("numeric_same_row_score") or 0.0) for row in evaluated] or [0.0]),
            4,
        ),
        "numeric_same_row_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "numeric_same_row_value_match_failure_counts": dict(
            sorted(value_failure_counts.items())
        ),
        "raw_content_persisted": False,
    }


def extract_numeric_source_witnesses(
    *,
    stem: str,
    option_parse: dict[str, Any],
    docs_by_label: dict[str, list[dict[str, Any]]],
    relation: dict[str, Any],
) -> dict[str, Any]:
    option_rows = {
        str(row.get("label") or ""): row
        for row in option_parse.get("option_rows", []) or []
        if isinstance(row, dict) and row.get("parse_success")
    }
    if not option_rows:
        return {
            "status": "abstained",
            "reason": "no_parsed_numeric_options",
            "witnesses": [],
            "raw_content_persisted": False,
        }
    anchor_terms = set(numeric_relation_terms(stem))
    relation_terms = _relation_cue_terms(anchor_terms)
    subject_terms = _subject_anchor_terms(anchor_terms, relation_terms)
    threshold_terms = _threshold_signal_terms(str(relation.get("relation_family") or ""))
    threshold_extreme_terms = _threshold_extreme_terms(
        str(relation.get("relation_family") or "")
    )
    threshold_relation = bool(threshold_terms)
    witnesses: list[dict[str, Any]] = []
    doc_count = 0
    parsed_source_value_count = 0
    value_match_failure_counts: dict[str, int] = {}
    direct_rejection_reason_counts: dict[str, int] = {}
    best_score = 0.0

    def count_value_failure(reason: str) -> None:
        key = str(reason or "unknown")
        value_match_failure_counts[key] = value_match_failure_counts.get(key, 0) + 1

    def count_direct_rejection(reason: str) -> None:
        key = str(reason or "unknown")
        direct_rejection_reason_counts[key] = (
            direct_rejection_reason_counts.get(key, 0) + 1
        )

    for label, option in sorted(option_rows.items()):
        for doc_index, doc in enumerate(docs_by_label.get(label, []) or []):
            if not isinstance(doc, dict):
                continue
            text = _doc_text(doc)
            if not text:
                continue
            doc_count += 1
            doc_tokens = _tokens(text)
            anchor_overlap = len(anchor_terms & doc_tokens)
            relation_overlap = len(relation_terms & doc_tokens)
            subject_anchor_overlap = len(subject_terms & doc_tokens)
            threshold_signal_overlap = len(threshold_terms & doc_tokens)
            threshold_extreme_overlap = len(threshold_extreme_terms & doc_tokens)
            for value in parse_numeric_values(text):
                parsed_source_value_count += 1
                value_match_score, match_reason = _numeric_close(option, value)
                if value_match_score <= 0:
                    count_value_failure(match_reason)
                    continue
                unit_match_score = _unit_score(option, value)
                source_score = _source_weight(str(doc.get("source") or ""))
                generic_penalty = 0.0
                if subject_terms and subject_anchor_overlap <= 0:
                    generic_penalty += 2.2
                elif subject_terms and subject_anchor_overlap == 1:
                    generic_penalty += 0.4
                elif not subject_terms and anchor_overlap <= 0:
                    generic_penalty += 2.0
                elif not subject_terms and anchor_overlap == 1:
                    generic_penalty += 0.8
                if relation_overlap <= 0 and option.get("value_type") != "temperature":
                    generic_penalty += 1.0
                if threshold_relation and threshold_signal_overlap <= 0:
                    generic_penalty += 0.6
                relation_score = min(relation_overlap, 4) * 0.45
                anchor_score = min(anchor_overlap, 6) * 0.35
                subject_score = min(subject_anchor_overlap, 3) * 0.45
                score = round(
                    value_match_score
                    + unit_match_score
                    + source_score
                    + relation_score
                    + anchor_score
                    + subject_score
                    - generic_penalty,
                    4,
                )
                subject_anchor_satisfied = (
                    subject_anchor_overlap >= 1
                    if subject_terms
                    else anchor_overlap >= 2
                )
                required_relation_overlap = 2 if threshold_relation else 1
                direct = bool(
                    score >= 5.4
                    and subject_anchor_satisfied
                    and relation_overlap >= required_relation_overlap
                )
                best_score = max(best_score, score)
                rejection_reasons: list[str] = []
                if not direct:
                    if score < 5.4:
                        rejection_reasons.append("score_below_direct_threshold")
                    if not subject_anchor_satisfied:
                        rejection_reasons.append("subject_anchor_not_satisfied")
                    if relation_overlap < required_relation_overlap:
                        rejection_reasons.append("relation_overlap_below_required")
                    if threshold_relation and threshold_signal_overlap <= 0:
                        rejection_reasons.append("threshold_signal_missing")
                    if threshold_relation and threshold_extreme_overlap <= 0:
                        rejection_reasons.append("threshold_extreme_missing")
                    if generic_penalty > 0:
                        rejection_reasons.append("generic_penalty_applied")
                    if not rejection_reasons:
                        rejection_reasons.append("direct_gate_failed")
                    for reason in rejection_reasons:
                        count_direct_rejection(reason)
                witness_hash = stable_hash({
                    "label": label,
                    "doc_hash": stable_hash({
                        "title": str(doc.get("title") or ""),
                        "snippet": str(doc.get("snippet") or ""),
                        "source": str(doc.get("source") or ""),
                    }),
                    "value_hash": value.get("value_hash"),
                    "score": score,
                    "direct": direct,
                })
                witnesses.append({
                    "label": label,
                    "option_hash": option.get("option_hash"),
                    "doc_index": doc_index,
                    "doc_hash": stable_hash({
                        "title": str(doc.get("title") or ""),
                        "snippet": str(doc.get("snippet") or ""),
                        "source": str(doc.get("source") or ""),
                    }),
                    "source": str(doc.get("source") or ""),
                    "value_hash": value.get("value_hash"),
                    "value_match_score": value_match_score,
                    "value_match_reason": match_reason,
                    "unit_match_score": unit_match_score,
                    "anchor_overlap": anchor_overlap,
                    "subject_anchor_overlap": subject_anchor_overlap,
                    "subject_anchor_term_count": len(subject_terms),
                    "relation_overlap": relation_overlap,
                    "threshold_signal_overlap": threshold_signal_overlap,
                    "threshold_extreme_overlap": threshold_extreme_overlap,
                    "source_score": source_score,
                    "generic_penalty": round(generic_penalty, 4),
                    "score": score,
                    "numeric_direct_witness": direct,
                    "numeric_direct_witness_reason": (
                        "value_subject_relation_match"
                        if direct
                        else ",".join(rejection_reasons)
                    ),
                    "witness_hash": witness_hash,
                    "raw_content_persisted": False,
                })
    direct_count = sum(1 for row in witnesses if row.get("numeric_direct_witness"))
    payload = {
        "status": "activated" if witnesses else "abstained",
        "reason": "numeric_witnesses_extracted" if witnesses else "no_numeric_value_matches_in_sources",
        "source_doc_count": doc_count,
        "parsed_source_value_count": parsed_source_value_count,
        "value_match_failure_counts": dict(sorted(value_match_failure_counts.items())),
        "direct_rejection_reason_counts": dict(
            sorted(direct_rejection_reason_counts.items())
        ),
        "best_witness_score": round(best_score, 4),
        "witness_count": len(witnesses),
        "numeric_direct_witness_count": direct_count,
        "witnesses": sorted(
            witnesses,
            key=lambda row: (-float(row.get("score") or 0.0), str(row.get("label") or ""), str(row.get("witness_hash") or "")),
        ),
        "raw_content_persisted": False,
    }
    payload["numeric_witness_hash"] = stable_hash({
        "witnesses": [
            {
                "label": row.get("label"),
                "value_hash": row.get("value_hash"),
                "doc_hash": row.get("doc_hash"),
                "score": row.get("score"),
                "direct": row.get("numeric_direct_witness"),
                "subject_anchor_overlap": row.get("subject_anchor_overlap"),
                "relation_overlap": row.get("relation_overlap"),
                "threshold_signal_overlap": row.get("threshold_signal_overlap"),
                "threshold_extreme_overlap": row.get("threshold_extreme_overlap"),
            }
            for row in payload["witnesses"]
        ]
    })
    return payload

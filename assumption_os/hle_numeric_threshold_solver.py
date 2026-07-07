"""Conservative numeric-threshold lane for HLE multiple-choice answers."""

from __future__ import annotations

from typing import Any

from .autonomy_journal import stable_hash
from .hle_numeric_option_parser import parse_numeric_options
from .hle_numeric_relation_classifier import classify_numeric_relation
from .hle_numeric_source_witness import extract_numeric_source_witnesses


def _option_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("normalized_value"))
    except (TypeError, ValueError):
        return 0.0


def _score_rows(
    *,
    option_parse: dict[str, Any],
    witnesses: dict[str, Any],
    relation: dict[str, Any],
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for witness in witnesses.get("witnesses", []) or []:
        if not isinstance(witness, dict):
            continue
        by_label.setdefault(str(witness.get("label") or ""), []).append(witness)
    rows: list[dict[str, Any]] = []
    direction = str(relation.get("direction") or "")
    parsed_rows = [
        row for row in option_parse.get("option_rows", []) or []
        if isinstance(row, dict) and row.get("parse_success")
    ]
    values = [_option_value(row) for row in parsed_rows]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 0.0
    for option in parsed_rows:
        label = str(option.get("label") or "")
        label_witnesses = sorted(
            by_label.get(label, []),
            key=lambda row: -float(row.get("score") or 0.0),
        )
        best = label_witnesses[0] if label_witnesses else {}
        direct_count = sum(1 for row in label_witnesses if row.get("numeric_direct_witness"))
        base_score = float(best.get("score") or 0.0)
        direction_bonus = 0.0
        value = _option_value(option)
        direction_extreme = bool(
            (direction == "lower_is_correct" and value == min_value)
            or (direction == "higher_is_correct" and value == max_value)
        )
        if direct_count > 0 and direction == "lower_is_correct" and value == min_value:
            direction_bonus = 0.7
        elif direct_count > 0 and direction == "higher_is_correct" and value == max_value:
            direction_bonus = 0.7
        final_score = round(base_score + direction_bonus, 4)
        rows.append({
            "label": label,
            "option_hash": option.get("option_hash"),
            "value_hash": option.get("value_hash"),
            "normalized_value": option.get("normalized_value"),
            "normalized_unit": option.get("normalized_unit"),
            "value_type": option.get("value_type"),
            "best_witness_hash": best.get("witness_hash"),
            "best_witness_doc_hash": best.get("doc_hash"),
            "best_witness_score": round(base_score, 4),
            "best_witness_anchor_overlap": int(best.get("anchor_overlap") or 0),
            "best_witness_subject_anchor_overlap": int(
                best.get("subject_anchor_overlap") or 0
            ),
            "best_witness_relation_overlap": int(best.get("relation_overlap") or 0),
            "best_witness_threshold_signal_overlap": int(
                best.get("threshold_signal_overlap") or 0
            ),
            "best_witness_threshold_extreme_overlap": int(
                best.get("threshold_extreme_overlap") or 0
            ),
            "numeric_direct_witness_count": direct_count,
            "direction_extreme": direction_extreme,
            "direction_bonus": direction_bonus,
            "score": final_score,
            "raw_content_persisted": False,
        })
    return rows


def _select_conservative(
    rows: list[dict[str, Any]],
    *,
    relation: dict[str, Any],
    min_score: float,
    min_margin: float,
) -> dict[str, Any]:
    direct_rows = [
        row for row in rows
        if int(row.get("numeric_direct_witness_count") or 0) > 0
        and float(row.get("score") or 0.0) >= min_score
    ]
    if not direct_rows:
        return {
            "status": "abstained",
            "reason": "no_high_confidence_numeric_direct_witness",
            "selected_label": None,
        }
    direction = str(relation.get("direction") or "")
    sorted_by_score = sorted(
        direct_rows,
        key=lambda row: (-float(row.get("score") or 0.0), str(row.get("label") or "")),
    )
    top_score = float(sorted_by_score[0].get("score") or 0.0)
    near_top = [
        row for row in sorted_by_score
        if top_score - float(row.get("score") or 0.0) <= 1.0
    ]
    if direction == "lower_is_correct" and len(near_top) > 1:
        selected = sorted(near_top, key=lambda row: (_option_value(row), -float(row.get("score") or 0.0)))[0]
    elif direction == "higher_is_correct" and len(near_top) > 1:
        selected = sorted(near_top, key=lambda row: (-_option_value(row), -float(row.get("score") or 0.0)))[0]
    else:
        selected = sorted_by_score[0]

    relation_family = str(relation.get("relation_family") or "")
    if relation_family in {
        "threshold_minimum",
        "threshold_maximum",
        "below_threshold",
        "above_threshold",
        "ordered_extreme_lowest",
        "ordered_extreme_highest",
    }:
        if (
            not bool(selected.get("direction_extreme"))
            and int(selected.get("best_witness_threshold_extreme_overlap") or 0) <= 0
        ):
            return {
                "status": "abstained",
                "reason": "threshold_numeric_witness_missing_direction_extreme_evidence",
                "selected_label": None,
                "top_score": round(float(selected.get("score") or 0.0), 4),
                "runner_up_score": round(
                    float(sorted_by_score[1].get("score") or 0.0)
                    if len(sorted_by_score) > 1
                    else 0.0,
                    4,
                ),
                "margin": round(
                    float(selected.get("score") or 0.0)
                    - (
                        float(sorted_by_score[1].get("score") or 0.0)
                        if len(sorted_by_score) > 1
                        else 0.0
                    ),
                    4,
                ),
                "direct_candidate_count": len(direct_rows),
            }

    selected_score = float(selected.get("score") or 0.0)
    runner_up_scores = [
        float(row.get("score") or 0.0)
        for row in direct_rows
        if str(row.get("label") or "") != str(selected.get("label") or "")
    ]
    runner_up_score = max(runner_up_scores or [0.0])
    margin = round(selected_score - runner_up_score, 4)
    if len(direct_rows) > 1 and margin < min_margin:
        return {
            "status": "abstained",
            "reason": "numeric_direct_witness_margin_too_small",
            "selected_label": None,
            "top_score": round(selected_score, 4),
            "runner_up_score": round(runner_up_score, 4),
            "margin": margin,
        }
    return {
        "status": "activated",
        "reason": "unique_high_confidence_numeric_direct_witness",
        "selected_label": selected.get("label"),
        "selected_option_hash": selected.get("option_hash"),
        "top_score": round(selected_score, 4),
        "runner_up_score": round(runner_up_score, 4),
        "margin": margin,
        "selected_row": selected,
        "direct_candidate_count": len(direct_rows),
    }


def solve_numeric_threshold_lane(
    *,
    stem: str,
    options: dict[str, str],
    docs_by_label: dict[str, list[dict[str, Any]]],
    category: str = "",
    raw_subject: str = "",
    min_score: float = 5.8,
    min_margin: float = 1.25,
) -> dict[str, Any]:
    option_parse = parse_numeric_options(options)
    if option_parse.get("status") != "activated":
        payload = {
            "status": "abstained",
            "reason": option_parse.get("reason") or "numeric_options_not_detected",
            "policy": "numeric_threshold_lane_v1",
            "option_parse": option_parse,
            "raw_content_persisted": False,
        }
        payload["router_payload_hash"] = stable_hash({
            "status": payload["status"],
            "reason": payload["reason"],
            "parse_hash": option_parse.get("parse_hash"),
        })
        return payload

    relation = classify_numeric_relation(
        stem,
        value_type=str(option_parse.get("dominant_value_type") or ""),
    )
    witnesses = extract_numeric_source_witnesses(
        stem=stem,
        option_parse=option_parse,
        docs_by_label=docs_by_label,
        relation=relation,
    )
    rows = _score_rows(option_parse=option_parse, witnesses=witnesses, relation=relation)
    selection = _select_conservative(
        rows,
        relation=relation,
        min_score=min_score,
        min_margin=min_margin,
    )
    status = str(selection.get("status") or "abstained")
    payload = {
        "status": status,
        "reason": selection.get("reason"),
        "policy": "numeric_threshold_lane_v1",
        "selected_label": selection.get("selected_label"),
        "selected_option_hash": selection.get("selected_option_hash"),
        "direct_high_confidence": status == "activated",
        "relation_satisfied": status == "activated",
        "supports_answer": status == "activated",
        "confidence": "numeric_direct_witness" if status == "activated" else "abstained",
        "option_count": len(options or {}),
        "numeric_option_parse_rate": option_parse.get("numeric_option_parse_rate"),
        "numeric_option_count": option_parse.get("numeric_option_count"),
        "dominant_value_type": option_parse.get("dominant_value_type"),
        "relation_family": relation.get("relation_family"),
        "relation_direction": relation.get("direction"),
        "numeric_source_span_found_rate": round(
            len({row.get("label") for row in witnesses.get("witnesses", []) or []})
            / max(1, int(option_parse.get("numeric_option_count") or 0)),
            4,
        ),
        "numeric_direct_witness_accept_rate": round(
            len({
                row.get("label")
                for row in witnesses.get("witnesses", []) or []
                if row.get("numeric_direct_witness")
            })
            / max(1, int(option_parse.get("numeric_option_count") or 0)),
            4,
        ),
        "numeric_direct_witness_count": int(witnesses.get("numeric_direct_witness_count") or 0),
        "witness_count": int(witnesses.get("witness_count") or 0),
        "witness_source_doc_count": int(witnesses.get("source_doc_count") or 0),
        "witness_parsed_source_value_count": int(
            witnesses.get("parsed_source_value_count") or 0
        ),
        "witness_value_match_failure_counts": dict(
            witnesses.get("value_match_failure_counts") or {}
        ),
        "witness_direct_rejection_reason_counts": dict(
            witnesses.get("direct_rejection_reason_counts") or {}
        ),
        "best_witness_score": witnesses.get("best_witness_score"),
        "direct_candidate_count": int(selection.get("direct_candidate_count") or 0),
        "top_score": selection.get("top_score"),
        "runner_up_score": selection.get("runner_up_score"),
        "margin": selection.get("margin"),
        "option_rows": sorted(
            rows,
            key=lambda row: (-float(row.get("score") or 0.0), str(row.get("label") or "")),
        ),
        "parse_hash": option_parse.get("parse_hash"),
        "relation_hash": relation.get("relation_hash"),
        "numeric_witness_hash": witnesses.get("numeric_witness_hash"),
        "category_hash": stable_hash({"category": category or ""}) if category else None,
        "raw_subject_hash": stable_hash({"raw_subject": raw_subject or ""}) if raw_subject else None,
        "raw_content_persisted": False,
    }
    payload["router_payload_hash"] = stable_hash({
        "policy": payload["policy"],
        "status": payload["status"],
        "reason": payload["reason"],
        "selected_option_hash": payload.get("selected_option_hash"),
        "parse_hash": payload.get("parse_hash"),
        "relation_hash": payload.get("relation_hash"),
        "numeric_witness_hash": payload.get("numeric_witness_hash"),
        "top_score": payload.get("top_score"),
        "margin": payload.get("margin"),
    })
    return payload

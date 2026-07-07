"""Option-centered matrix assembly for HLE multiple-choice routing."""

from __future__ import annotations

from typing import Any

from .autonomy_journal import stable_hash
from .hle_option_span_bundle import build_option_span_bundles
from .hle_pair_binding_comparator import adjudicate_pair_binding
from .hle_self_contained_operator_matrix import build_self_contained_operator_matrix


def _solver_row_by_label(solver_matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in solver_matrix.get("option_rows", []) if isinstance(solver_matrix, dict) else []:
        if isinstance(row, dict) and row.get("label"):
            rows[str(row["label"])] = row
    return rows


def _baseline_features(label: str, baseline_predictions: dict[str, str] | None) -> dict[str, Any]:
    baseline_predictions = baseline_predictions or {}
    supporting = sorted(
        variant
        for variant, predicted_label in baseline_predictions.items()
        if str(predicted_label or "").strip() == label
    )
    return {
        "supporting_variants": supporting,
        "support_count": len(supporting),
        "raw_content_persisted": False,
    }


def build_option_matrix(
    *,
    question: str,
    options: dict[str, str],
    category: str = "",
    raw_subject: str = "",
    source_records_by_option: dict[str, list[Any]] | None = None,
    relation_terms: list[str] | None = None,
    required_terms: list[str] | None = None,
    baseline_predictions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble source, solver, and baseline features per option."""

    span_bundles = build_option_span_bundles(
        question=question,
        options=options,
        source_records_by_option=source_records_by_option,
        relation_terms=relation_terms,
        required_terms=required_terms,
    )
    source_lane = adjudicate_pair_binding(span_bundles_by_option=span_bundles)
    solver_matrix = build_self_contained_operator_matrix(
        stem=question,
        options=options,
        category=category,
        raw_subject=raw_subject,
    )
    solver_rows = _solver_row_by_label(solver_matrix)
    option_rows: list[dict[str, Any]] = []
    for label, option_text in sorted(options.items()):
        bundles = span_bundles.get(label, [])
        best_bundle = bundles[0] if bundles else {}
        solver_row = solver_rows.get(label, {})
        option_rows.append({
            "label": label,
            "option_hash": stable_hash({"option_label": label}),
            "option_text_hash": stable_hash({"option_text": option_text}),
            "source_lane": {
                "top_span_bundle_hashes": [
                    str(bundle.get("bundle_hash") or "")
                    for bundle in bundles
                    if isinstance(bundle, dict)
                ],
                "best_directness_score": best_bundle.get("directness_score", 0.0),
                "best_bundle_type": best_bundle.get("bundle_type"),
                "best_bundle_hash": best_bundle.get("bundle_hash"),
            },
            "self_contained_lane": {
                "operator_family": solver_row.get("operator_family"),
                "solver_score": solver_row.get("solver_score", 0),
                "is_alkyne_like": solver_row.get("is_alkyne_like", False),
                "is_reagent_not_probe_handle": solver_row.get("is_reagent_not_probe_handle", False),
            },
            "baseline_lane": _baseline_features(label, baseline_predictions),
        })
    matrix = {
        "question_hash": stable_hash({"question": question}),
        "option_rows": option_rows,
        "source_lane": source_lane,
        "self_contained_lane": solver_matrix,
        "baseline_predictions_hash": stable_hash(baseline_predictions or {}),
        "raw_content_persisted": False,
    }
    matrix["option_matrix_hash"] = stable_hash({
        "question_hash": matrix["question_hash"],
        "option_rows": option_rows,
        "source_lane_hash": source_lane.get("pair_binding_payload_hash"),
        "solver_hash": solver_matrix.get("matrix_hash"),
        "baseline_predictions_hash": matrix["baseline_predictions_hash"],
    })
    return matrix

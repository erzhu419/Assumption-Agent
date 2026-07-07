"""Programmatic pair-binding comparator over fixed option span bundles."""

from __future__ import annotations

from typing import Any

from .autonomy_journal import stable_hash


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _best_bundle(bundles: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not bundles:
        return None
    return max(
        (bundle for bundle in bundles if isinstance(bundle, dict)),
        key=lambda bundle: (
            float(bundle.get("directness_score") or 0.0),
            1 if bundle.get("bundle_type") == "direct_relation" else 0,
            str(bundle.get("bundle_hash") or ""),
        ),
        default=None,
    )


def _bundle_relation_established(bundle: dict[str, Any] | None) -> bool:
    if not isinstance(bundle, dict):
        return False
    if str(bundle.get("bundle_type") or "") not in {
        "direct_relation",
        "numeric_direct",
        "formula",
        "table_row",
    }:
        return False
    relation_terms = (
        bundle.get("relation_signature_terms")
        or bundle.get("relation_overlap_terms")
        or []
    )
    if not bundle.get("option_overlap_terms") or not relation_terms:
        return False
    if bundle.get("required_terms_missing"):
        return False
    return float(bundle.get("directness_score") or 0.0) >= 0.7


def _bundle_missing_elements(bundle: dict[str, Any] | None) -> list[str]:
    if not isinstance(bundle, dict):
        return ["bundle_missing"]
    missing: list[str] = []
    if not bundle.get("option_overlap_terms"):
        missing.append("option_overlap")
    relation_terms = (
        bundle.get("relation_signature_terms")
        or bundle.get("relation_overlap_terms")
        or []
    )
    if not relation_terms:
        missing.append("relation_signature")
    if bundle.get("required_terms_missing"):
        missing.append("required_terms")
    if float(bundle.get("directness_score") or 0.0) < 0.7:
        missing.append("directness_score")
    return missing


def adjudicate_pair_binding(
    *,
    span_bundles_by_option: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Select a source-bound option only when a direct bundle has positive margin."""

    scored: list[tuple[float, str, dict[str, Any]]] = []
    option_summaries: list[dict[str, Any]] = []
    for label, bundles in sorted((span_bundles_by_option or {}).items()):
        bundle = _best_bundle(bundles)
        score = float((bundle or {}).get("directness_score") or 0.0)
        established = _bundle_relation_established(bundle)
        option_summaries.append({
            "label": label,
            "option_hash": stable_hash({"option_label": label}),
            "best_bundle_hash": (bundle or {}).get("bundle_hash"),
            "best_directness_score": round(score, 3),
            "relation_established": established,
            "bundle_type": (bundle or {}).get("bundle_type"),
            "binding_strength": (
                "direct" if established else str((bundle or {}).get("bundle_type") or "generic")
            ),
            "missing_elements": _bundle_missing_elements(bundle),
            "required_terms_missing_count": len((bundle or {}).get("required_terms_missing") or []),
        })
        if established:
            scored.append((score, label, bundle or {}))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    if not ranked:
        out = {
            "status": "no_candidate",
            "pair_binding_accept": False,
            "pair_binding_verdict": "reject_generic",
            "reason": "no_direct_pair_bound_span",
            "option_summaries": option_summaries,
            "raw_content_persisted": False,
        }
    else:
        top_score, top_label, top_bundle = ranked[0]
        runner_up_score = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = round(top_score - runner_up_score, 3)
        top_audit_features = (
            top_bundle.get("audit_features")
            if isinstance(top_bundle.get("audit_features"), dict)
            else {}
        )
        top_bundle_audit = {
            "required_overlap": _to_int(top_audit_features.get("required_overlap")),
            "required_missing": _to_int(top_audit_features.get("required_missing")),
            "option_overlap": _to_int(top_audit_features.get("option_overlap")),
            "relation_overlap": _to_int(top_audit_features.get("relation_overlap")),
            "slot_coverage": _to_int(top_audit_features.get("slot_coverage")),
            "relation_proximity": _to_bool(
                top_audit_features.get("relation_proximity")
            ),
            "candidate_specific": _to_bool(
                top_audit_features.get("candidate_specific")
            ),
            "strict_answer_bearing": _to_bool(
                top_audit_features.get("strict_answer_bearing")
            ),
            "source_cache_answer_bearing": _to_bool(
                top_audit_features.get("source_cache_answer_bearing")
            ),
        }
        if margin < 0.15:
            out = {
                "status": "ambiguous",
                "selected_label": top_label,
                "pair_binding_accept": False,
                "pair_binding_verdict": "reject_ambiguous",
                "relation_established": True,
                "binding_strength": "direct",
                "beats_runner_up": False,
                "missing_elements": [],
                "reason": "direct_pair_bound_margin_too_small",
                "best_direct_margin": margin,
                "option_summaries": option_summaries,
                "best_bundle_hash": top_bundle.get("bundle_hash"),
                "best_bundle_audit_features": top_bundle_audit,
                "best_required_overlap": top_bundle_audit["required_overlap"],
                "best_required_missing": top_bundle_audit["required_missing"],
                "best_option_overlap": top_bundle_audit["option_overlap"],
                "best_relation_overlap": top_bundle_audit["relation_overlap"],
                "best_slot_coverage": top_bundle_audit["slot_coverage"],
                "raw_content_persisted": False,
            }
        else:
            out = {
                "status": "activated",
                "selected_label": top_label,
                "selected_option_hash": stable_hash({"option_label": top_label}),
                "pair_binding_accept": True,
                "pair_binding_verdict": "accept_direct",
                "relation_established": True,
                "evidence_relation": "direct",
                "binding_strength": "direct",
                "beats_runner_up": True,
                "missing_elements": [],
                "confidence": "high",
                "confidence_score": min(0.95, max(0.7, top_score)),
                "reason": "direct_pair_bound_span_beats_runner_up",
                "best_direct_margin": margin,
                "best_bundle_hash": top_bundle.get("bundle_hash"),
                "best_bundle_audit_features": top_bundle_audit,
                "best_required_overlap": top_bundle_audit["required_overlap"],
                "best_required_missing": top_bundle_audit["required_missing"],
                "best_option_overlap": top_bundle_audit["option_overlap"],
                "best_relation_overlap": top_bundle_audit["relation_overlap"],
                "best_slot_coverage": top_bundle_audit["slot_coverage"],
                "option_summaries": option_summaries,
                "raw_content_persisted": False,
            }
    out["pair_binding_payload_hash"] = stable_hash(out)
    out["fixed_pair_binding_payload_hash"] = stable_hash({
        "option_summaries": option_summaries,
        "selected_option_hash": out.get("selected_option_hash"),
        "pair_binding_verdict": out.get("pair_binding_verdict"),
        "best_direct_margin": out.get("best_direct_margin"),
    })
    return out


def _source_audit_row_to_bundle(row: dict[str, Any], label: str) -> dict[str, Any]:
    option_hash = stable_hash({"option_label": label})
    direct_span_count = _to_int(row.get("candidate_direct_relation_span_count"))
    required_overlap = _to_int(
        row.get("candidate_direct_relation_span_top_relation_signature_required_overlap")
    )
    raw_missing_terms = row.get(
        "candidate_direct_relation_span_top_relation_signature_missing_term_count"
    )
    missing_terms = _to_int(raw_missing_terms, 999 if raw_missing_terms is None else 0)
    relation_proximity = bool(row.get("candidate_direct_relation_span_top_relation_proximity"))
    signature_proximity = bool(
        row.get("candidate_direct_relation_span_top_relation_signature_proximity")
    )
    shared_doc = bool(
        row.get("candidate_direct_relation_span_top_shared_doc")
        or row.get("candidate_direct_relation_span_top_source_doc_shared")
    )
    strict_direct = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_strict_direct_support_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_strict_direct_support_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_strict_direct_support_doc_count")),
    )
    candidate_specific = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_candidate_specific_span_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_candidate_specific_span_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_candidate_specific_span_doc_count")),
    )
    directish = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_directish_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_directish_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_directish_doc_count")),
    )
    witness_required = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_required_overlap_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_required_overlap_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_required_overlap_doc_count")),
    )
    witness_proximity = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_relation_proximity_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_relation_proximity_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_relation_proximity_doc_count")),
    )
    source_quality_score = _to_float(row.get("source_quality_score"))
    source_quality_doc_count = _to_int(row.get("source_quality_doc_count"))
    support_doc_count = _to_int(row.get("support_doc_count"))
    refute_doc_count = _to_int(row.get("refute_doc_count"))
    ambiguous_doc_count = _to_int(row.get("ambiguous_doc_count"))
    source_refutation_doc_count = _to_int(row.get("source_quality_statement_fact_refutation_doc_count"))
    source_refutation_high_count = _to_int(
        row.get("source_quality_statement_fact_refutation_high_confidence_doc_count")
    )
    zero_slot_rejected = _to_int(row.get("source_quality_statement_fact_zero_slot_rejected_doc_count"))
    no_refutation = bool(
        refute_doc_count <= 0
        and source_refutation_doc_count <= 0
        and source_refutation_high_count <= 0
        and zero_slot_rejected <= 0
    )
    span_complete = bool(
        direct_span_count >= 2
        and required_overlap >= 2
        and missing_terms <= 0
        and relation_proximity
        and signature_proximity
        and not shared_doc
    )
    witness_complete = bool(
        strict_direct >= 2
        and candidate_specific >= 4
        and directish >= 4
        and witness_required >= 5
        and witness_proximity >= 5
        and required_overlap >= 2
        and missing_terms <= 1
        and relation_proximity
        and signature_proximity
        and not shared_doc
    )
    source_supported = bool(
        source_quality_doc_count > 0
        or support_doc_count > 0
        or source_quality_score >= 8.0
    )
    direct = bool(
        no_refutation
        and ambiguous_doc_count <= 1
        and source_supported
        and (span_complete or witness_complete)
    )
    if direct:
        bundle_type = "direct_relation"
        directness_score = min(
            0.97,
            0.72
            + 0.03 * min(direct_span_count, 4)
            + 0.02 * min(strict_direct, 4)
            + 0.015 * min(witness_required, 6)
            + 0.01 * min(source_quality_doc_count + support_doc_count, 4)
            + min(max(source_quality_score, 0.0), 16.0) / 400.0,
        )
    elif relation_proximity or signature_proximity or direct_span_count > 0:
        bundle_type = "indirect"
        directness_score = min(0.68, 0.30 + 0.05 * min(direct_span_count, 4))
    else:
        bundle_type = "generic"
        directness_score = 0.0
    bundle = {
        "option_label": label,
        "option_hash": option_hash,
        "source_id_hash": stable_hash({"source_audit_option_hash": option_hash}),
        "span_hash": row.get("candidate_direct_relation_span_top_span_hash")
        or row.get("candidate_direct_relation_span_candidate_specific_span_hash")
        or stable_hash({
            "source_audit_bundle": option_hash,
            "direct_span_count": direct_span_count,
            "required_overlap": required_overlap,
            "missing_terms": missing_terms,
        }),
        "option_overlap_terms": ["candidate_specific"] if candidate_specific or direct_span_count else [],
        "anchor_overlap_terms": ["source_witness"] if candidate_specific or witness_proximity else [],
        "relation_overlap_terms": ["relation"] if relation_proximity or signature_proximity or required_overlap else [],
        "relation_signature_terms": ["relation"] if relation_proximity or signature_proximity or required_overlap else [],
        "required_terms_present": ["required"] * min(required_overlap, 4),
        "required_terms_missing": [] if missing_terms <= 0 or direct else ["missing"],
        "shared_doc_option_count": 2 if shared_doc else 1,
        "shared_doc_penalty": 0.2 if shared_doc else 0.0,
        "generic_penalty": 0.0 if direct_span_count or candidate_specific else 0.2,
        "bundle_type": bundle_type,
        "directness_score": round(max(0.0, min(1.0, directness_score)), 3),
        "audit_features": {
            "direct_span_count": direct_span_count,
            "required_overlap": required_overlap,
            "missing_terms": missing_terms,
            "relation_proximity": relation_proximity,
            "signature_proximity": signature_proximity,
            "strict_direct": strict_direct,
            "candidate_specific": candidate_specific,
            "directish": directish,
            "witness_required": witness_required,
            "witness_proximity": witness_proximity,
            "source_quality_score": round(source_quality_score, 4),
            "source_quality_doc_count": source_quality_doc_count,
            "support_doc_count": support_doc_count,
            "refute_doc_count": refute_doc_count,
            "ambiguous_doc_count": ambiguous_doc_count,
            "span_complete": span_complete,
            "witness_complete": witness_complete,
        },
        "raw_content_persisted": False,
    }
    bundle["bundle_hash"] = stable_hash({
        "option_hash": option_hash,
        "span_hash": bundle["span_hash"],
        "bundle_type": bundle_type,
        "directness_score": bundle["directness_score"],
        "audit_features": bundle["audit_features"],
    })
    return bundle


def source_verifier_audit_source_lane(
    *,
    candidate_summaries: list[dict[str, Any]],
    label_by_hash: dict[str, str],
) -> dict[str, Any]:
    """Build a pair-binding source lane from source verifier audit summaries."""

    bundles_by_label: dict[str, list[dict[str, Any]]] = {}
    audit_row_count = 0
    for row in candidate_summaries or []:
        if not isinstance(row, dict):
            continue
        option_hash = str(row.get("option_hash") or "")
        label = str(row.get("label") or label_by_hash.get(option_hash) or "").strip()
        if not label:
            continue
        audit_row_count += 1
        bundles_by_label.setdefault(label, []).append(_source_audit_row_to_bundle(row, label))
    for label, bundles in list(bundles_by_label.items()):
        bundles_by_label[label] = sorted(
            bundles,
            key=lambda bundle: (
                -float(bundle.get("directness_score") or 0.0),
                str(bundle.get("bundle_hash") or ""),
            ),
        )[:3]
    lane = adjudicate_pair_binding(span_bundles_by_option=bundles_by_label)
    lane["policy"] = "source_verifier_audit_option_matrix_source_lane_v1"
    lane["audit_row_count"] = audit_row_count
    lane["span_bundle_candidate_count"] = len(bundles_by_label)
    lane["span_bundle_hashes_by_option_hash"] = {
        stable_hash({"option_label": label}): [
            str(bundle.get("bundle_hash") or "")
            for bundle in bundles
            if str(bundle.get("bundle_hash") or "")
        ]
        for label, bundles in sorted(bundles_by_label.items())
    }
    lane["source_lane_payload_hash"] = stable_hash({
        "policy": lane["policy"],
        "audit_row_count": lane["audit_row_count"],
        "span_bundle_hashes_by_option_hash": lane["span_bundle_hashes_by_option_hash"],
        "selected_option_hash": lane.get("selected_option_hash"),
        "pair_binding_verdict": lane.get("pair_binding_verdict"),
        "best_direct_margin": lane.get("best_direct_margin"),
    })
    return lane


def _source_quality_identity_signal(row: dict[str, Any]) -> bool:
    selection_reason = str(row.get("selection_reason") or "").strip().lower()
    if selection_reason in {
        "best_sweep_only_source_quality",
        "high_confidence_source_quality_challenger",
        "best_sweep_only_ranked",
        "source_cache_answer_bearing_challenger",
    }:
        return True
    if selection_reason.startswith("finite_option_coverage_"):
        return True
    return any(
        _to_bool(row.get(key))
        for key in (
            "sweep_only_candidate",
            "source_quality_challenger",
            "source_cache_answer_bearing_challenger",
            "ranked_sweep_only_challenger",
            "finite_option_coverage_candidate",
        )
    )


def _source_quality_pair_binding_bundle(
    row: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    option_hash = str(row.get("option_hash") or stable_hash({"option_label": label}))
    direct_span_count = _to_int(row.get("candidate_direct_relation_span_count"))
    required_overlap = max(
        _to_int(
            row.get(
                "candidate_direct_relation_span_top_relation_signature_required_overlap"
            )
        ),
        _to_int(row.get("answer_web_cache_sweep_relation_signature_required_overlap_count")),
        _to_int(row.get("source_cache_corpus_backfill_required_overlap_count")),
        _to_int(row.get("planned_query_required_overlap_doc_count")),
    )
    raw_missing_terms = row.get(
        "candidate_direct_relation_span_top_relation_signature_missing_term_count"
    )
    missing_terms = _to_int(raw_missing_terms, 999 if raw_missing_terms is None else 0)
    relation_proximity = bool(
        _to_bool(row.get("candidate_direct_relation_span_top_relation_proximity"))
        or _to_int(row.get("answer_web_cache_sweep_relation_proximity_count")) > 0
        or _to_int(row.get("source_cache_corpus_backfill_relation_proximity_count")) > 0
        or _to_int(row.get("planned_query_relation_proximity_doc_count")) > 0
    )
    signature_proximity = bool(
        _to_bool(row.get("candidate_direct_relation_span_top_relation_signature_proximity"))
        or required_overlap > 0
    )
    shared_doc = bool(
        _to_bool(row.get("candidate_direct_relation_span_top_shared_doc"))
        or _to_bool(row.get("candidate_direct_relation_span_top_source_doc_shared"))
    )
    strict_direct = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_strict_direct_support_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_strict_direct_support_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_strict_direct_support_doc_count")),
    )
    candidate_specific = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_candidate_specific_span_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_candidate_specific_span_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_candidate_specific_span_doc_count")),
    )
    directish = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_directish_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_directish_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_directish_doc_count")),
        _to_int(row.get("answer_web_cache_sweep_general_relation_directish_count")),
        _to_int(row.get("planned_query_answer_bearing_direct_doc_count")),
    )
    witness_required = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_required_overlap_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_required_overlap_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_required_overlap_doc_count")),
        _to_int(row.get("answer_web_cache_sweep_relation_signature_required_overlap_count")),
        _to_int(row.get("planned_query_required_overlap_doc_count")),
    )
    witness_proximity = max(
        _to_int(row.get("source_cache_answer_bearing_focused_retry_relation_proximity_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_relation_proximity_doc_count")),
        _to_int(row.get("source_verifier_structured_candidate_relation_witness_audit_relation_proximity_doc_count")),
        _to_int(row.get("answer_web_cache_sweep_relation_proximity_count")),
        _to_int(row.get("planned_query_relation_proximity_doc_count")),
    )
    source_quality_score = _to_float(row.get("source_quality_score"))
    source_quality_doc_count = _to_int(row.get("source_quality_doc_count"))
    support_doc_count = _to_int(row.get("support_doc_count"))
    refute_doc_count = _to_int(row.get("refute_doc_count"))
    ambiguous_doc_count = _to_int(row.get("ambiguous_doc_count"))
    source_refutation_doc_count = _to_int(row.get("source_quality_statement_fact_refutation_doc_count"))
    source_refutation_high_count = _to_int(
        row.get("source_quality_statement_fact_refutation_high_confidence_doc_count")
    )
    source_refutation_strength = _to_float(
        row.get("source_quality_max_statement_fact_refutation_strength")
    )
    zero_slot_rejected = _to_int(row.get("source_quality_statement_fact_zero_slot_rejected_doc_count"))
    local_relation_doc_count = (
        _to_int(row.get("local_relation_corpus_doc_count"))
        + _to_int(row.get("local_relation_query_expansion_doc_count"))
        + _to_int(row.get("sweep_gap_local_relation_backfill_doc_count"))
        + _to_int(row.get("source_cache_corpus_backfill_doc_count"))
    )
    answer_web_relation = bool(
        _to_int(row.get("answer_web_cache_sweep_general_relation_directish_count")) > 0
        and _to_int(row.get("answer_web_cache_sweep_relation_proximity_count")) > 0
        and (
            _to_int(row.get("answer_web_cache_sweep_relation_slot_covered_count")) > 0
            or _to_int(row.get("answer_web_cache_sweep_relation_signature_required_overlap_count")) > 0
        )
    )
    planned_direct = bool(
        _to_int(row.get("planned_query_answer_bearing_direct_doc_count")) > 0
        and _to_int(row.get("planned_query_relation_proximity_doc_count")) > 0
        and (
            _to_int(row.get("planned_query_slot_covered_doc_count")) > 0
            or _to_int(row.get("planned_query_required_overlap_doc_count")) > 0
        )
    )
    source_rejection = str(row.get("source_verifier_rejection_reason") or "").strip()
    comparator_relation = str(
        row.get("relation_span_comparator_evidence_relation") or ""
    ).strip().lower()
    span_generic_rejected = bool(
        _to_bool(row.get("span_directness_lexical_unique_but_relation_generic"))
        or comparator_relation == "indirect"
    )
    identity_signal = _source_quality_identity_signal(row)
    no_refutation = bool(
        refute_doc_count <= 0
        and source_refutation_doc_count <= 0
        and source_refutation_high_count <= 0
        and source_refutation_strength <= 0.0
        and zero_slot_rejected <= 0
    )
    has_source_support = bool(
        source_quality_doc_count > 0
        or support_doc_count > 0
        or local_relation_doc_count >= 2
        or source_quality_score >= 8.0
    )
    span_signal = bool(
        direct_span_count >= 1
        and required_overlap >= 1
        and missing_terms <= 1
        and relation_proximity
        and signature_proximity
        and not shared_doc
    )
    witness_signal = bool(
        strict_direct >= 1
        and candidate_specific >= 2
        and directish >= 2
        and witness_required >= 2
        and witness_proximity >= 2
        and required_overlap >= 1
        and missing_terms <= 1
        and relation_proximity
        and signature_proximity
        and not shared_doc
    )
    source_relation_signal = bool(
        identity_signal
        and (answer_web_relation or planned_direct)
        and required_overlap >= 1
        and missing_terms <= 1
        and relation_proximity
        and not shared_doc
    )
    source_verifier_compatible = bool(
        not span_generic_rejected
        and (
            source_rejection
            not in {"no_selected_label_generic", "no_selected_label_indirect"}
            or witness_signal
            or planned_direct
        )
    )
    eligible = bool(
        identity_signal
        and no_refutation
        and ambiguous_doc_count <= 1
        and has_source_support
        and source_verifier_compatible
        and (span_signal or witness_signal or source_relation_signal)
    )
    row_summary = {
        "label": label,
        "option_hash": option_hash,
        "selection_reason": row.get("selection_reason"),
        "identity_signal": identity_signal,
        "source_quality_score": round(source_quality_score, 4),
        "source_quality_doc_count": source_quality_doc_count,
        "support_doc_count": support_doc_count,
        "refute_doc_count": refute_doc_count,
        "ambiguous_doc_count": ambiguous_doc_count,
        "source_refutation_doc_count": source_refutation_doc_count,
        "source_refutation_high_count": source_refutation_high_count,
        "source_refutation_strength": round(source_refutation_strength, 4),
        "zero_slot_rejected": zero_slot_rejected,
        "local_relation_doc_count": local_relation_doc_count,
        "direct_span_count": direct_span_count,
        "required_overlap": required_overlap,
        "missing_terms": missing_terms,
        "relation_proximity": relation_proximity,
        "signature_proximity": signature_proximity,
        "shared_doc": shared_doc,
        "strict_direct": strict_direct,
        "candidate_specific": candidate_specific,
        "directish": directish,
        "witness_required": witness_required,
        "witness_proximity": witness_proximity,
        "answer_web_relation": answer_web_relation,
        "planned_direct": planned_direct,
        "source_verifier_rejection_reason": source_rejection or None,
        "relation_span_comparator_evidence_relation": comparator_relation or None,
        "span_generic_rejected": span_generic_rejected,
        "source_verifier_compatible": source_verifier_compatible,
        "span_signal": span_signal,
        "witness_signal": witness_signal,
        "source_relation_signal": source_relation_signal,
        "eligible": eligible,
        "raw_content_persisted": False,
    }
    if not eligible:
        if not identity_signal:
            row_summary["rejection_reason"] = "not_source_quality_ranking_candidate"
        elif not no_refutation:
            row_summary["rejection_reason"] = "hard_refutation_or_zero_slot_rejection"
        elif ambiguous_doc_count > 1:
            row_summary["rejection_reason"] = "too_many_ambiguous_docs"
        elif not has_source_support:
            row_summary["rejection_reason"] = "missing_source_support"
        elif span_generic_rejected:
            row_summary["rejection_reason"] = "span_or_comparator_generic_indirect"
        elif not source_verifier_compatible:
            row_summary["rejection_reason"] = "source_verifier_generic_without_direct_witness"
        elif shared_doc:
            row_summary["rejection_reason"] = "shared_relation_doc"
        elif required_overlap <= 0:
            row_summary["rejection_reason"] = "missing_required_overlap"
        elif missing_terms > 1:
            row_summary["rejection_reason"] = "missing_required_terms"
        elif not relation_proximity or not signature_proximity:
            row_summary["rejection_reason"] = "missing_relation_proximity"
        else:
            row_summary["rejection_reason"] = "missing_direct_relation_signal"
        return None, row_summary

    evidence_score = (
        (2.0 * min(direct_span_count, 3))
        + (1.5 * min(required_overlap, 3))
        + (1.25 * min(source_quality_doc_count + support_doc_count, 4))
        + (1.0 * min(local_relation_doc_count, 4))
        + (0.75 * min(strict_direct, 4))
        + (0.50 * min(candidate_specific, 6))
        + (0.45 * min(directish, 6))
        + (0.35 * min(witness_required, 6))
        + (0.25 * min(witness_proximity, 6))
        + (1.0 if answer_web_relation else 0.0)
        + (1.0 if planned_direct else 0.0)
        + min(max(source_quality_score, 0.0), 16.0) / 4.0
    )
    row_summary["evidence_score"] = round(evidence_score, 4)
    directness_score = min(
        0.97,
        0.70
        + 0.015 * min(evidence_score, 10.0)
        + (0.03 if span_signal else 0.0)
        + (0.02 if witness_signal else 0.0)
        + (0.015 if source_relation_signal else 0.0),
    )
    span_hash = str(
        row.get("candidate_direct_relation_span_top_span_hash")
        or row.get("candidate_direct_relation_span_candidate_specific_span_hash")
        or stable_hash({
            "source_quality_pair_binding_bundle": option_hash,
            "direct_span_count": direct_span_count,
            "required_overlap": required_overlap,
            "missing_terms": missing_terms,
            "selection_reason": row.get("selection_reason"),
        })
    )
    bundle = {
        "option_label": label,
        "option_hash": option_hash,
        "source_id_hash": stable_hash({
            "source_quality_pair_binding_source": option_hash,
            "selection_reason": row.get("selection_reason"),
        }),
        "span_hash": span_hash,
        "option_overlap_terms": ["candidate_specific"],
        "anchor_overlap_terms": ["source_quality_ranking"],
        "relation_overlap_terms": ["relation"],
        "relation_signature_terms": ["relation"],
        "required_terms_present": ["required"] * min(required_overlap, 4),
        "required_terms_missing": [] if missing_terms <= 0 else ["missing"],
        "shared_doc_option_count": 1,
        "shared_doc_penalty": 0.0,
        "generic_penalty": 0.0,
        "bundle_type": "direct_relation",
        "directness_score": round(max(0.0, min(1.0, directness_score)), 3),
        "audit_features": row_summary,
        "raw_content_persisted": False,
    }
    bundle["bundle_hash"] = stable_hash({
        "option_hash": option_hash,
        "span_hash": span_hash,
        "bundle_type": bundle["bundle_type"],
        "directness_score": bundle["directness_score"],
        "audit_features": row_summary,
    })
    return bundle, row_summary


def source_quality_pair_binding_source_lane(
    *,
    candidate_summaries: list[dict[str, Any]],
    label_by_hash: dict[str, str],
) -> dict[str, Any]:
    """Rank source-quality/sweep candidates only when relation evidence is direct."""

    bundles_by_label: dict[str, list[dict[str, Any]]] = {}
    candidate_rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    candidate_row_count = 0
    eligible_candidate_count = 0
    for row in candidate_summaries or []:
        if not isinstance(row, dict):
            rejection_counts["non_dict_candidate"] = rejection_counts.get("non_dict_candidate", 0) + 1
            continue
        option_hash = str(row.get("option_hash") or "")
        label = str(row.get("label") or label_by_hash.get(option_hash) or "").strip()
        if not label:
            rejection_counts["option_hash_not_in_candidate_labels"] = (
                rejection_counts.get("option_hash_not_in_candidate_labels", 0) + 1
            )
            continue
        candidate_row_count += 1
        bundle, row_summary = _source_quality_pair_binding_bundle(row, label=label)
        candidate_rows.append(row_summary)
        if bundle is None:
            reason = str(row_summary.get("rejection_reason") or "rejected")
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        eligible_candidate_count += 1
        bundles_by_label.setdefault(label, []).append(bundle)

    for label, bundles in list(bundles_by_label.items()):
        bundles_by_label[label] = sorted(
            bundles,
            key=lambda bundle: (
                -float(bundle.get("directness_score") or 0.0),
                str(bundle.get("bundle_hash") or ""),
            ),
        )[:2]

    lane = adjudicate_pair_binding(span_bundles_by_option=bundles_by_label)
    if lane.get("status") == "activated":
        margin = _to_float(lane.get("best_direct_margin"))
        selected_hash = str(lane.get("selected_option_hash") or "")
        selected_row = next(
            (
                row
                for row in candidate_rows
                if str(row.get("option_hash") or "") == selected_hash
            ),
            {},
        )
        selected_evidence_score = _to_float(selected_row.get("evidence_score"))
        if margin < 0.20 or selected_evidence_score < 8.0:
            lane.update({
                "status": "ambiguous",
                "pair_binding_accept": False,
                "pair_binding_verdict": "reject_source_quality_ranking_margin",
                "reason": "source_quality_ranking_requires_stronger_margin",
                "selected_source_quality_evidence_score": round(
                    selected_evidence_score,
                    4,
                ),
            })
    lane["policy"] = "source_quality_pair_binding_option_matrix_source_lane_v1"
    lane["candidate_row_count"] = candidate_row_count
    lane["eligible_candidate_count"] = eligible_candidate_count
    lane["candidate_rows"] = candidate_rows[:8]
    lane["candidate_rows_hash"] = stable_hash({
        "source_quality_pair_binding_candidate_rows": candidate_rows
    })
    lane["rejection_counts"] = dict(sorted(rejection_counts.items()))
    lane["span_bundle_candidate_count"] = len(bundles_by_label)
    lane["span_bundle_hashes_by_option_hash"] = {
        stable_hash({"option_label": label}): [
            str(bundle.get("bundle_hash") or "")
            for bundle in bundles
            if str(bundle.get("bundle_hash") or "")
        ]
        for label, bundles in sorted(bundles_by_label.items())
    }
    lane["source_lane_payload_hash"] = stable_hash({
        "policy": lane["policy"],
        "candidate_rows_hash": lane["candidate_rows_hash"],
        "span_bundle_hashes_by_option_hash": lane["span_bundle_hashes_by_option_hash"],
        "selected_option_hash": lane.get("selected_option_hash"),
        "pair_binding_verdict": lane.get("pair_binding_verdict"),
        "best_direct_margin": lane.get("best_direct_margin"),
        "rejection_counts": lane["rejection_counts"],
    })
    return lane


def _candidate_span_witness_to_bundle(
    witness: dict[str, Any],
    *,
    label: str,
    option_hash: str,
) -> dict[str, Any]:
    witness_type = str(witness.get("witness_type") or "")
    required_count = _to_int(witness.get("required_count"))
    required_overlap = _to_int(witness.get("required_overlap"))
    required_missing = _to_int(witness.get("required_missing_count"), 999)
    option_overlap = _to_int(witness.get("option_overlap"))
    relation_overlap = _to_int(witness.get("relation_overlap"))
    slot_coverage = _to_int(witness.get("slot_coverage"))
    source_quality_score = _to_float(witness.get("source_quality_score"))
    relation_proximity = bool(witness.get("relation_proximity"))
    candidate_specific = bool(witness.get("candidate_specific"))
    strict_answer_bearing = bool(witness.get("strict_answer_bearing"))
    source_cache_answer_bearing = bool(witness.get("source_cache_answer_bearing"))
    multi_witness_required_completion = _to_bool(
        witness.get("multi_witness_required_completion")
    )
    required_complete_inferred = bool(
        witness.get("required_completion_inferred_from_source_verifier")
    )
    option_bound_inferred = bool(
        witness.get("option_bound_inferred_from_source_verifier")
        or witness.get("option_bound_inferred_from_candidate_specific_span")
    )
    shared_or_other = bool(witness.get("shared_or_other"))
    refutation = bool(witness.get("refutation"))
    option_bound = bool(option_overlap > 0 or option_bound_inferred)
    relation_bound = bool(
        relation_overlap > 0 or slot_coverage > 0 or relation_proximity
    )
    strict_source = bool(
        strict_answer_bearing
        or source_cache_answer_bearing
        or required_complete_inferred
    )
    direct_required_floor_met = bool(
        _to_bool(witness.get("direct_required_floor_met"))
        or required_overlap >= 2
        or (
            required_count > 0
            and required_count <= 1
            and required_overlap > 0
            and required_missing <= 0
            and not required_complete_inferred
        )
        or bool(witness.get("multi_witness_required_completion"))
    )
    direct = bool(
        witness_type == "direct_relation"
        and option_bound
        and relation_bound
        and required_overlap > 0
        and required_missing <= 0
        and direct_required_floor_met
        and candidate_specific
        and strict_source
        and not shared_or_other
        and not refutation
    )
    if direct:
        bundle_type = "direct_relation"
        directness_score = min(
            0.97,
            0.70
            + 0.04 * min(required_overlap, 4)
            + 0.02 * min(relation_overlap + slot_coverage, 4)
            + (0.025 if source_cache_answer_bearing else 0.0)
            + (0.02 if multi_witness_required_completion else 0.0)
            + (0.01 if option_bound_inferred else 0.0)
            + min(max(source_quality_score, 0.0), 16.0) / 320.0,
        )
    elif witness_type in {"indirect_relation", "relation_only"} or relation_bound:
        bundle_type = "indirect"
        directness_score = min(
            0.68,
            0.28
            + min(max(source_quality_score, 0.0), 12.0) / 120.0
            + 0.03 * min(required_overlap, 3)
            + 0.02 * min(relation_overlap + slot_coverage, 3),
        )
    else:
        bundle_type = "generic"
        directness_score = 0.0
    span_hash = str(
        witness.get("witness_id")
        or witness.get("span_hash")
        or stable_hash({
            "candidate_span_witness": option_hash,
            "source_doc_hash": str(witness.get("source_doc_hash") or ""),
            "witness_type": witness_type,
            "required_count": required_count,
            "required_overlap": required_overlap,
            "required_missing": required_missing,
        })
    )
    bundle = {
        "option_label": label,
        "option_hash": option_hash,
        "source_id_hash": stable_hash({
            "candidate_span_bundle_source": option_hash,
            "source_doc_hash": str(witness.get("source_doc_hash") or ""),
        }),
        "span_hash": span_hash,
        "option_overlap_terms": ["candidate_specific"] if option_bound else [],
        "anchor_overlap_terms": ["source_witness"] if candidate_specific or slot_coverage else [],
        "relation_overlap_terms": ["relation"] if relation_bound else [],
        "relation_signature_terms": ["relation"] if relation_bound else [],
        "required_terms_present": ["required"] * min(required_overlap, 4),
        "required_terms_missing": (
            [] if required_missing <= 0 else ["missing"] * min(required_missing, 4)
        ),
        "shared_doc_option_count": 2 if shared_or_other else 1,
        "shared_doc_penalty": 0.2 if shared_or_other else 0.0,
        "generic_penalty": 0.0 if direct else 0.2,
        "bundle_type": bundle_type,
        "directness_score": round(max(0.0, min(1.0, directness_score)), 3),
        "audit_features": {
            "witness_type": witness_type,
            "required_overlap": required_overlap,
            "required_missing": required_missing,
            "option_overlap": option_overlap,
            "relation_overlap": relation_overlap,
            "slot_coverage": slot_coverage,
            "relation_proximity": relation_proximity,
            "candidate_specific": candidate_specific,
            "strict_answer_bearing": strict_answer_bearing,
            "source_cache_answer_bearing": source_cache_answer_bearing,
            "multi_witness_required_completion": multi_witness_required_completion,
            "required_complete_inferred": required_complete_inferred,
            "direct_required_floor_met": direct_required_floor_met,
            "option_bound_inferred": option_bound_inferred,
            "option_bound_inferred_from_candidate_specific_span": _to_bool(
                witness.get("option_bound_inferred_from_candidate_specific_span")
            ),
            "shared_or_other": shared_or_other,
            "refutation": refutation,
            "source_quality_score": round(source_quality_score, 4),
            "direct": direct,
        },
        "raw_content_persisted": False,
    }
    bundle["bundle_hash"] = stable_hash({
        "option_hash": option_hash,
        "span_hash": span_hash,
        "bundle_type": bundle_type,
        "directness_score": bundle["directness_score"],
        "audit_features": bundle["audit_features"],
    })
    return bundle


def candidate_span_bundle_source_lane(
    *,
    candidate_span_bundle_detail: dict[str, Any] | None,
    label_by_hash: dict[str, str],
    candidate_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a source lane from deterministic candidate-specific span bundles."""

    detail = (
        candidate_span_bundle_detail
        if isinstance(candidate_span_bundle_detail, dict)
        else {}
    )
    bundles_by_label: dict[str, list[dict[str, Any]]] = {}
    rejection_counts: dict[str, int] = {}
    witness_row_count = 0
    direct_witness_row_count = 0
    summary_by_hash = {
        str(row.get("option_hash") or ""): row
        for row in (candidate_summaries or [])
        if isinstance(row, dict) and str(row.get("option_hash") or "").strip()
    }

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    def source_rejection_guard(lane_detail: dict[str, Any]) -> dict[str, Any]:
        selected_hash = str(lane_detail.get("selected_option_hash") or "")
        selected = summary_by_hash.get(selected_hash) or {}
        base = {
            "policy": "candidate_span_bundle_source_verifier_rejection_guard_v1",
            "status": "not_required",
            "reason": "not_evaluated",
            "blocked": False,
            "selected_option_hash": selected_hash or None,
            "selected_source_verifier_rejection_reason": "",
            "selected_required_overlap": 0,
            "selected_required_missing_terms": None,
            "selected_refute_doc_count": 0,
            "selected_ambiguous_doc_count": 0,
            "selected_source_quality_score": 0.0,
            "competing_complete_candidate_count": 0,
            "competing_complete_option_hashes": [],
            "raw_content_persisted": False,
        }
        if not selected_hash or lane_detail.get("status") != "activated":
            return {**base, "reason": "source_lane_not_activated"}
        if not summary_by_hash:
            return {**base, "reason": "no_candidate_summaries"}
        if not selected:
            return {**base, "status": "blocked", "reason": "selected_summary_missing"}
        selected_rejection = str(
            selected.get("source_verifier_rejection_reason") or ""
        ).strip()
        selected_required_overlap = max(
            _to_int(
                selected.get(
                    "candidate_direct_relation_span_top_relation_signature_required_overlap"
                )
            ),
            _to_int(selected.get("source_cache_corpus_backfill_required_overlap_count")),
            _to_int(selected.get("planned_query_required_overlap_doc_count")),
            _to_int(
                selected.get(
                    "source_verifier_structured_candidate_relation_witness_audit_required_overlap_doc_count"
                )
            ),
        )
        raw_missing = selected.get(
            "candidate_direct_relation_span_top_relation_signature_missing_term_count"
        )
        selected_required_missing = _to_int(
            raw_missing,
            999 if raw_missing is None else 0,
        )
        selected_refute_count = max(
            _to_int(selected.get("refute_doc_count")),
            _to_int(selected.get("source_quality_statement_fact_refutation_doc_count")),
            _to_int(
                selected.get(
                    "source_quality_statement_fact_refutation_high_confidence_doc_count"
                )
            ),
            _to_int(selected.get("source_quality_statement_fact_zero_slot_rejected_doc_count")),
        )
        selected_ambiguous_count = _to_int(selected.get("ambiguous_doc_count"))
        selected_source_quality_score = _to_float(
            selected.get("source_quality_score")
        )
        selected_accepted_direct = bool(
            _to_bool(selected.get("source_verifier_any_accepted_direct_high_confidence"))
            or _to_bool(selected.get("source_verifier_any_direct_high_confidence"))
        )
        competing_complete_hashes: list[str] = []
        for option_hash, row in summary_by_hash.items():
            if option_hash == selected_hash:
                continue
            competitor_overlap = max(
                _to_int(
                    row.get(
                        "candidate_direct_relation_span_top_relation_signature_required_overlap"
                    )
                ),
                _to_int(row.get("source_cache_corpus_backfill_required_overlap_count")),
                _to_int(row.get("planned_query_required_overlap_doc_count")),
            )
            competitor_raw_missing = row.get(
                "candidate_direct_relation_span_top_relation_signature_missing_term_count"
            )
            competitor_missing = _to_int(
                competitor_raw_missing,
                999 if competitor_raw_missing is None else 0,
            )
            competitor_refute = max(
                _to_int(row.get("refute_doc_count")),
                _to_int(row.get("source_quality_statement_fact_refutation_doc_count")),
                _to_int(
                    row.get(
                        "source_quality_statement_fact_refutation_high_confidence_doc_count"
                    )
                ),
            )
            if (
                competitor_overlap > 0
                and competitor_missing <= 0
                and competitor_refute <= 1
            ):
                competing_complete_hashes.append(option_hash)
        bad_rejection = selected_rejection in {
            "no_selected_label_generic",
            "no_selected_label_indirect",
        }
        required_gap = bool(
            selected_required_missing > 0 or selected_required_overlap <= 0
        )
        hard_refuted = selected_refute_count > 0
        blocked = bool(
            not selected_accepted_direct
            and (
                hard_refuted
                or (bad_rejection and required_gap)
            )
        )
        reason = "source_verifier_guard_passed"
        if blocked and hard_refuted:
            reason = "selected_option_source_verifier_refuted"
        elif blocked:
            reason = "selected_option_source_verifier_rejected_required_gap"
        return {
            **base,
            "status": "blocked" if blocked else "allowed",
            "reason": reason,
            "blocked": blocked,
            "selected_source_verifier_rejection_reason": selected_rejection,
            "selected_required_overlap": selected_required_overlap,
            "selected_required_missing_terms": selected_required_missing,
            "selected_refute_doc_count": selected_refute_count,
            "selected_ambiguous_doc_count": selected_ambiguous_count,
            "selected_source_quality_score": round(selected_source_quality_score, 4),
            "selected_source_verifier_accepted_direct": selected_accepted_direct,
            "competing_complete_candidate_count": len(competing_complete_hashes),
            "competing_complete_option_hashes": competing_complete_hashes[:8],
        }

    for option_bundle in detail.get("option_bundles", []) or []:
        if not isinstance(option_bundle, dict):
            reject("non_dict_option_bundle")
            continue
        option_hash = str(option_bundle.get("option_hash") or "")
        label = str(label_by_hash.get(option_hash) or "").strip()
        if not option_hash or not label:
            reject("option_hash_not_in_candidate_labels")
            continue
        for witness in option_bundle.get("top_witnesses", []) or []:
            if not isinstance(witness, dict):
                reject("non_dict_witness")
                continue
            witness_row_count += 1
            bundle = _candidate_span_witness_to_bundle(
                witness,
                label=label,
                option_hash=option_hash,
            )
            if bundle.get("bundle_type") == "direct_relation":
                direct_witness_row_count += 1
            else:
                witness_type = str(witness.get("witness_type") or "unknown")
                reject(f"non_direct_{witness_type}")
            bundles_by_label.setdefault(label, []).append(bundle)

    for label, bundles in list(bundles_by_label.items()):
        bundles_by_label[label] = sorted(
            bundles,
            key=lambda bundle: (
                -float(bundle.get("directness_score") or 0.0),
                str(bundle.get("bundle_hash") or ""),
            ),
        )[:3]

    lane = adjudicate_pair_binding(span_bundles_by_option=bundles_by_label)
    recommended_hash = str(detail.get("selected_option_hash") or "")
    selected_hash = str(lane.get("selected_option_hash") or "")
    if (
        lane.get("status") == "activated"
        and recommended_hash
        and selected_hash
        and selected_hash != recommended_hash
    ):
        margin = _to_float(lane.get("best_direct_margin"))
        if margin >= 0.30:
            lane.update({
                "candidate_span_bundle_conflict": True,
                "candidate_span_bundle_recommended_option_hash": recommended_hash,
                "independent_lattice_override": True,
                "reason": "independent_lattice_override_recommendation_conflict_high_margin",
            })
        else:
            lane.update({
                "status": "ambiguous",
                "pair_binding_accept": False,
                "pair_binding_verdict": "reject_candidate_span_bundle_mismatch",
                "reason": "candidate_span_bundle_recommendation_conflict_requires_semantic_comparator",
                "candidate_span_bundle_recommended_option_hash": recommended_hash,
            })
    elif (
        lane.get("status") == "activated"
        and not recommended_hash
    ):
        margin = _to_float(lane.get("best_direct_margin"))
        if margin >= 0.25:
            lane.update({
                "independent_lattice_override": True,
                "reason": "independent_lattice_override_without_prior_recommendation",
            })
        else:
            lane.update({
                "status": "ambiguous",
                "pair_binding_accept": False,
                "pair_binding_verdict": "reject_candidate_span_bundle_no_strong_margin",
                "reason": "candidate_span_bundle_no_recommendation_requires_semantic_comparator",
            })

    source_guard = source_rejection_guard(lane)
    if source_guard.get("blocked"):
        lane.update({
            "status": "ambiguous",
            "pair_binding_accept": False,
            "pair_binding_verdict": "reject_candidate_span_bundle_source_verifier_guard",
            "reason": source_guard.get("reason"),
        })

    top_audit_features = (
        lane.get("best_bundle_audit_features")
        if isinstance(lane.get("best_bundle_audit_features"), dict)
        else {}
    )
    contested_single_required_direct = bool(
        lane.get("status") == "activated"
        and _to_int(detail.get("option_with_witness_count")) > 1
        and _to_int(top_audit_features.get("required_overlap")) < 2
        and not _to_bool(top_audit_features.get("source_cache_answer_bearing"))
        and not _to_bool(
            top_audit_features.get("multi_witness_required_completion")
        )
    )
    if contested_single_required_direct:
        lane.update({
            "status": "ambiguous",
            "pair_binding_accept": False,
            "pair_binding_verdict": (
                "reject_candidate_span_bundle_contested_single_required_direct"
            ),
            "reason": "contested_single_required_direct_requires_stronger_source",
        })

    lane["policy"] = "candidate_span_bundle_option_matrix_source_lane_v1"
    lane["candidate_span_bundle_status"] = detail.get("status")
    lane["candidate_span_bundle_reason"] = detail.get("reason")
    lane["candidate_span_bundle_hash"] = detail.get("bundle_hash")
    lane["candidate_span_bundle_rows_hash"] = detail.get("rows_hash")
    lane["candidate_span_bundle_selected_option_hash"] = recommended_hash or None
    lane["candidate_span_bundle_direct_source_margin"] = detail.get(
        "direct_source_margin"
    )
    lane["candidate_span_bundle_recommendation_reason"] = detail.get(
        "recommendation_reason"
    )
    lane["candidate_span_bundle_option_with_direct_witness_count"] = _to_int(
        detail.get("option_with_direct_witness_count")
    )
    lane["candidate_span_bundle_direct_witness_count"] = _to_int(
        detail.get("direct_witness_count")
    )
    lane["witness_row_count"] = witness_row_count
    lane["direct_witness_row_count"] = direct_witness_row_count
    lane["source_verifier_rejection_guard"] = source_guard
    lane["source_verifier_rejection_guard_status"] = source_guard.get("status")
    lane["source_verifier_rejection_guard_reason"] = source_guard.get("reason")
    lane["source_verifier_rejection_guard_blocked"] = bool(
        source_guard.get("blocked")
    )
    lane["contested_single_required_direct"] = bool(
        contested_single_required_direct
    )
    lane["span_bundle_candidate_count"] = len(bundles_by_label)
    lane["rejection_counts"] = dict(sorted(rejection_counts.items()))
    lane["span_bundle_hashes_by_option_hash"] = {
        str(option_bundle.get("option_hash") or ""): [
            str(bundle.get("bundle_hash") or "")
            for bundle in bundles_by_label.get(
                str(label_by_hash.get(str(option_bundle.get("option_hash") or "")) or ""),
                [],
            )
            if str(bundle.get("bundle_hash") or "")
        ]
        for option_bundle in detail.get("option_bundles", []) or []
        if isinstance(option_bundle, dict)
        and str(option_bundle.get("option_hash") or "") in label_by_hash
    }
    lane["source_lane_payload_hash"] = stable_hash({
        "policy": lane["policy"],
        "candidate_span_bundle_hash": lane["candidate_span_bundle_hash"],
        "candidate_span_bundle_rows_hash": lane["candidate_span_bundle_rows_hash"],
        "span_bundle_hashes_by_option_hash": lane["span_bundle_hashes_by_option_hash"],
        "selected_option_hash": lane.get("selected_option_hash"),
        "pair_binding_verdict": lane.get("pair_binding_verdict"),
        "best_direct_margin": lane.get("best_direct_margin"),
        "source_verifier_rejection_guard": {
            "status": source_guard.get("status"),
            "reason": source_guard.get("reason"),
            "blocked": bool(source_guard.get("blocked")),
            "selected_option_hash": source_guard.get("selected_option_hash"),
        },
        "rejection_counts": lane["rejection_counts"],
    })
    return lane

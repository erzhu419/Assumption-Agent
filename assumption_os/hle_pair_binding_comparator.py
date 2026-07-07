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
    if str(bundle.get("bundle_type") or "") != "direct_relation":
        return False
    if not bundle.get("option_overlap_terms") or not bundle.get("relation_overlap_terms"):
        return False
    if bundle.get("required_terms_missing"):
        return False
    return float(bundle.get("directness_score") or 0.0) >= 0.7


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
        if margin < 0.15:
            out = {
                "status": "ambiguous",
                "selected_label": top_label,
                "pair_binding_accept": False,
                "pair_binding_verdict": "reject_ambiguous",
                "reason": "direct_pair_bound_margin_too_small",
                "best_direct_margin": margin,
                "option_summaries": option_summaries,
                "raw_content_persisted": False,
            }
        else:
            out = {
                "status": "activated",
                "selected_label": top_label,
                "selected_option_hash": stable_hash({"option_label": top_label}),
                "pair_binding_accept": True,
                "pair_binding_verdict": "accept_direct",
                "evidence_relation": "direct",
                "confidence": "high",
                "confidence_score": min(0.95, max(0.7, top_score)),
                "reason": "direct_pair_bound_span_beats_runner_up",
                "best_direct_margin": margin,
                "best_bundle_hash": top_bundle.get("bundle_hash"),
                "option_summaries": option_summaries,
                "raw_content_persisted": False,
            }
    out["pair_binding_payload_hash"] = stable_hash(out)
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
        "anchor_overlap_terms": [],
        "relation_overlap_terms": ["relation"] if relation_proximity or signature_proximity or required_overlap else [],
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


def _candidate_span_witness_to_bundle(
    witness: dict[str, Any],
    *,
    label: str,
    option_hash: str,
) -> dict[str, Any]:
    witness_type = str(witness.get("witness_type") or "")
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
    required_complete_inferred = bool(
        witness.get("required_completion_inferred_from_source_verifier")
    )
    option_bound_inferred = bool(
        witness.get("option_bound_inferred_from_source_verifier")
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
    direct = bool(
        witness_type == "direct_relation"
        and option_bound
        and relation_bound
        and required_overlap > 0
        and required_missing <= 0
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
            + min(max(source_quality_score, 0.0), 16.0) / 80.0
            + 0.025 * min(required_overlap, 4)
            + 0.015 * min(relation_overlap + slot_coverage, 4)
            + (0.025 if source_cache_answer_bearing else 0.0)
            + (0.015 if required_complete_inferred else 0.0),
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
        "anchor_overlap_terms": [],
        "relation_overlap_terms": ["relation"] if relation_bound else [],
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
            "required_complete_inferred": required_complete_inferred,
            "option_bound_inferred": option_bound_inferred,
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

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

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
        lane.update({
            "status": "blocked",
            "pair_binding_accept": False,
            "pair_binding_verdict": "reject_candidate_span_bundle_mismatch",
            "reason": "candidate_span_bundle_recommendation_mismatch",
            "candidate_span_bundle_recommended_option_hash": recommended_hash,
        })
    elif (
        lane.get("status") == "activated"
        and not recommended_hash
    ):
        lane.update({
            "status": "blocked",
            "pair_binding_accept": False,
            "pair_binding_verdict": "reject_candidate_span_bundle_no_strong_margin",
            "reason": "candidate_span_bundle_no_strong_direct_margin",
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
        "rejection_counts": lane["rejection_counts"],
    })
    return lane

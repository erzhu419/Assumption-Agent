"""Conservative option-lane router for HLE multiple-choice selection."""

from __future__ import annotations

from typing import Any

from .autonomy_journal import stable_hash


def _confidence_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    normalized = str(value or "").strip().lower()
    if normalized in {"verified", "high"}:
        return 0.9
    if normalized in {"mechanistic_domain_rule", "strong"}:
        return 0.86
    if normalized == "medium":
        return 0.6
    if normalized == "low":
        return 0.35
    return 0.0


def _label_from_lane(lane: dict[str, Any] | None, keys: tuple[str, ...]) -> str:
    if not isinstance(lane, dict):
        return ""
    for key in keys:
        label = str(lane.get(key) or "").strip()
        if label:
            return label
    return ""


def _baseline_label(baseline_lane: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(baseline_lane, dict):
        return "", "baseline_missing"
    ordered = (
        ("selected_label", "baseline_selected"),
        ("baseline_label", "baseline_selected"),
        ("hippo_label", "hipporag_fallback"),
        ("hipporag_label", "hipporag_fallback"),
        ("raw_label", "raw_fallback"),
    )
    for key, source in ordered:
        label = str(baseline_lane.get(key) or "").strip()
        if label:
            return label, source
    return "", "baseline_missing"


def _source_accepts_direct(source_lane: dict[str, Any] | None) -> bool:
    if not isinstance(source_lane, dict):
        return False
    verdict = str(
        source_lane.get("pair_binding_verdict")
        or source_lane.get("verdict")
        or source_lane.get("evidence_relation")
        or ""
    ).strip().lower()
    margin = float(source_lane.get("best_direct_margin") or source_lane.get("direct_margin") or 0.0)
    confidence = _confidence_score(source_lane.get("confidence_score", source_lane.get("confidence")))
    accepts = bool(source_lane.get("pair_binding_accept")) or verdict in {
        "accept_direct",
        "direct",
        "answer_bearing",
        "support_refute",
    }
    return accepts and margin >= 0.15 and confidence >= 0.7


def _source_is_weak(source_lane: dict[str, Any] | None) -> bool:
    if not isinstance(source_lane, dict):
        return True
    status = str(source_lane.get("status") or "").strip().lower()
    verdict = str(
        source_lane.get("pair_binding_verdict")
        or source_lane.get("verdict")
        or source_lane.get("evidence_relation")
        or ""
    ).strip().lower()
    reason = str(source_lane.get("reason") or "").strip().lower()
    return (
        status in {"", "none", "missing", "no_candidate", "generic", "abstained"}
        or verdict in {"", "none", "generic", "reject_generic", "reject_indirect", "reject_ambiguous"}
        or "no_candidate" in reason
        or "generic" in reason
    )


def _solver_is_strong(solver_lane: dict[str, Any] | None) -> bool:
    if not isinstance(solver_lane, dict):
        return False
    status = str(solver_lane.get("status") or "").strip().lower()
    if status != "activated":
        return False
    confidence = _confidence_score(solver_lane.get("confidence_score", solver_lane.get("confidence")))
    try:
        margin = float(solver_lane.get("unique_margin") or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    return bool(_label_from_lane(solver_lane, ("selected_label", "label"))) and confidence >= 0.75 and margin >= 1.0


def route_option_lanes(
    *,
    source_lane: dict[str, Any] | None = None,
    solver_lane: dict[str, Any] | None = None,
    baseline_lane: dict[str, Any] | None = None,
    fast_policy_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route source, solver, and baseline lanes without returning no_fallback."""

    source_label = _label_from_lane(source_lane, ("selected_label", "label"))
    solver_label = _label_from_lane(solver_lane, ("selected_label", "label"))
    baseline_label, baseline_source = _baseline_label(baseline_lane)
    source_confidence = _confidence_score((source_lane or {}).get("confidence_score", (source_lane or {}).get("confidence")))
    solver_confidence = _confidence_score((solver_lane or {}).get("confidence_score", (solver_lane or {}).get("confidence")))

    if source_label and _source_accepts_direct(source_lane):
        method = "source_direct_override"
        selected = source_label
        reason = "source_pair_binding_direct_positive_margin"
    elif source_label and solver_label and source_label == solver_label and min(source_confidence, solver_confidence) >= 0.55:
        method = "source_solver_agreement"
        selected = source_label
        reason = "source_and_solver_agree_with_moderate_confidence"
    elif solver_label and _solver_is_strong(solver_lane):
        method = "self_contained_solver_override"
        selected = solver_label
        reason = "self_contained_solver_unique_high_confidence"
    else:
        selected = baseline_label
        if _source_is_weak(source_lane):
            reason = "fallback_weak_source"
        else:
            reason = "fallback_conflict_or_low_margin"
        method = baseline_source

    policy_summary = _fast_policy_summary(fast_policy_decision)
    route = {
        "selected_label": selected,
        "selection_method": method,
        "reason": reason,
        "source_label_hash": stable_hash({"option_label": source_label}) if source_label else None,
        "solver_label_hash": stable_hash({"option_label": solver_label}) if solver_label else None,
        "baseline_label_hash": stable_hash({"option_label": baseline_label}) if baseline_label else None,
        "selected_label_hash": stable_hash({"option_label": selected}) if selected else None,
        "source_confidence": source_confidence,
        "solver_confidence": solver_confidence,
        "fast_policy_memory": policy_summary,
        "slow_baseline_required": policy_summary["slow_baseline_required"],
        "raw_content_persisted": False,
    }
    route["router_payload_hash"] = stable_hash(route)
    return route


def _fast_policy_summary(decision: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {
            "policy_version": "",
            "selected_policy_ids": [],
            "selected_policy_kinds": [],
            "selected_actions": [],
            "slow_baseline_required": True,
            "fast_policy_payload_hash": None,
            "raw_content_persisted": False,
        }
    return {
        "policy_version": str(decision.get("policy_version") or ""),
        "selected_policy_ids": list(decision.get("selected_policy_ids") or []),
        "selected_policy_kinds": list(decision.get("selected_policy_kinds") or []),
        "selected_actions": list(decision.get("selected_actions") or []),
        "slow_baseline_required": bool(decision.get("slow_baseline_required", True)),
        "fast_policy_payload_hash": decision.get("fast_policy_payload_hash"),
        "raw_content_persisted": False,
    }

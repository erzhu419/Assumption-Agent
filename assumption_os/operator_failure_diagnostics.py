"""Post-hoc failure taxonomy for assumption operator experiments.

This module is intentionally post-decision.  It may consume correctness labels
from evaluation rows, but it should never be used inside answer selection.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


TAXONOMY_VERSION = "operator_failure_taxonomy_v1"

FAILURE_CATEGORIES = {
    "NotOperatorFailure",
    "TriggerFalsePositive",
    "WrongOperator",
    "OverStructured",
    "VerifierFalsePositive",
    "SourceEvidenceMissing",
    "BaseGenerationNoise",
    "DomainOperatorMissing",
    "OperatorDeferred",
    "Unknown",
}


def classify_operator_failure(row_or_efficacy: dict[str, Any]) -> dict[str, Any]:
    """Classify an evaluated row into an operator failure bucket.

    The input can be a full run row with ``component_efficacy`` or the
    ``component_efficacy`` object itself.  The output stores only metadata.
    """

    efficacy = _efficacy(row_or_efficacy)
    flags = efficacy.get("flags") if isinstance(efficacy.get("flags"), dict) else {}
    final_correct = bool(efficacy.get("final_correct") or flags.get("final_correct"))
    if final_correct:
        return _result("NotOperatorFailure", "final_answer_correct")

    operator_specs = efficacy.get("operator_specs") if isinstance(efficacy.get("operator_specs"), dict) else {}
    operator_application = (
        efficacy.get("operator_application_verifier")
        if isinstance(efficacy.get("operator_application_verifier"), dict)
        else {}
    )
    policy = operator_specs.get("operator_policy") if isinstance(operator_specs.get("operator_policy"), dict) else {}
    kind = str(efficacy.get("kind") or efficacy.get("variant") or "")

    if not kind.startswith("assumption_agent"):
        return _result("BaseGenerationNoise", "non_agent_row_wrong")

    if _source_evidence_missing(efficacy):
        return _result("SourceEvidenceMissing", "source_verifier_attempted_without_direct_accepted_evidence")
    specs_requested = bool(flags.get("operator_specs_requested"))
    specs_activated = bool(flags.get("operator_specs_activated") or operator_specs.get("status") == "activated")
    specs_blocked = bool(flags.get("operator_specs_blocked"))
    application_applied = bool(
        flags.get("operator_application_applied")
        or operator_application.get("operator_application_applied")
    )
    application_passed = bool(
        flags.get("operator_application_passed")
        or operator_application.get("pass")
    )
    decorative = bool(
        flags.get("operator_decorative_use")
        or operator_application.get("decorative_use")
    )
    changed_candidate = bool(
        flags.get("operator_changed_candidate")
        or operator_application.get("operator_changed_candidate")
    )
    semantic = (
        operator_application.get("semantic_fidelity")
        if isinstance(operator_application.get("semantic_fidelity"), dict)
        else {}
    )
    semantic_pass = semantic.get("semantic_pass")
    slot_rate = _float(operator_application.get("slot_completion_rate"))
    p_trigger = _float(policy.get("p_trigger"))
    p_harm = _float(policy.get("p_harm"))
    selected_families = list(policy.get("selected_operator_family_ids", []) or [])
    abstain_reason = str(policy.get("abstain_reason") or "")

    if not specs_requested and _looks_like_operator_domain_missing(efficacy):
        return _result("DomainOperatorMissing", "agent_wrong_without_operator_request_in_operator_bearing_problem")
    if specs_blocked and abstain_reason:
        return _result("TriggerFalsePositive", f"operator_policy_abstained:{abstain_reason}")
    if specs_activated and p_harm >= p_trigger and p_harm > 0.0:
        return _result("TriggerFalsePositive", "policy_harm_not_lower_than_trigger")
    if specs_activated and not application_applied:
        return _result("OperatorDeferred", "operator_compiled_but_not_selected_after_noharm_gate")
    if specs_activated and decorative:
        return _result("OverStructured", "operator_branch_decorative_or_slot_surface_only")
    if application_passed and semantic_pass is False:
        return _result("VerifierFalsePositive", "slot_verifier_passed_but_semantic_fidelity_failed")
    if application_passed and not changed_candidate:
        return _result("VerifierFalsePositive", "operator_passed_without_changing_selection")
    if application_passed:
        return _result("WrongOperator", "operator_applied_cleanly_but_final_answer_wrong")
    if specs_activated and slot_rate < 0.75:
        return _result("OverStructured", "required_slots_not_substantively_filled")
    if specs_requested and not specs_activated:
        return _result("DomainOperatorMissing", "operator_requested_but_no_specific_operator_activated")
    if selected_families:
        return _result("WrongOperator", "selected_operator_family_did_not_resolve_item")
    return _result("BaseGenerationNoise", "no_operator_specific_signal_explains_wrong_answer")


def summarize_operator_failure_taxonomy(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = [classify_operator_failure(row) for row in rows]
    counts = Counter(str(item.get("category") or "Unknown") for item in diagnostics)
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "category_counts": dict(sorted(counts.items())),
        "row_count": len(diagnostics),
        "raw_content_persisted": False,
    }


def _result(category: str, reason: str) -> dict[str, Any]:
    if category not in FAILURE_CATEGORIES:
        category = "Unknown"
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "category": category,
        "reason": reason,
        "raw_content_persisted": False,
    }


def _efficacy(row_or_efficacy: dict[str, Any]) -> dict[str, Any]:
    row = row_or_efficacy if isinstance(row_or_efficacy, dict) else {}
    nested = row.get("component_efficacy")
    return nested if isinstance(nested, dict) else row


def _looks_like_operator_domain_missing(efficacy: dict[str, Any]) -> bool:
    operator_families = (
        efficacy.get("operator_family_tags")
        if isinstance(efficacy.get("operator_family_tags"), list)
        else []
    )
    domain = str(efficacy.get("domain") or "")
    flags = efficacy.get("flags") if isinstance(efficacy.get("flags"), dict) else {}
    return bool(
        operator_families
        or domain in {"daily_life", "hle_general", "science", "software_engineering"}
        or flags.get("morphism_hit")
        or flags.get("evidence_bridge_activated")
    )


def _source_evidence_missing(efficacy: dict[str, Any]) -> bool:
    flags = efficacy.get("flags") if isinstance(efficacy.get("flags"), dict) else {}
    verifier = (
        efficacy.get("mc_option_claim_evidence_verifier")
        if isinstance(efficacy.get("mc_option_claim_evidence_verifier"), dict)
        else {}
    )
    attempt_count = int(_float(verifier.get("source_verifier_attempt_count")))
    accepted_count = int(_float(verifier.get("source_verifier_accepted_attempt_count")))
    direct_count = int(_float(verifier.get("source_verifier_direct_high_confidence_count")))
    status = str(verifier.get("status") or "")
    span_status = str(verifier.get("span_directness_verifier_status") or "")
    direct_insufficient = bool(flags.get("gold_option_source_verifier_direct_source_insufficient"))
    indirect_or_generic = bool(flags.get("gold_option_source_verifier_indirect_or_generic"))
    source_supported = bool(flags.get("source_supported_evidence_candidate"))
    return bool(
        attempt_count > 0
        and accepted_count == 0
        and direct_count == 0
        and not source_supported
        and (
            status.startswith("blocked")
            or span_status.startswith("blocked")
            or direct_insufficient
            or indirect_or_generic
        )
    )


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0

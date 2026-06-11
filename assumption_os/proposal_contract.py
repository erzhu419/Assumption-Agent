"""Production proposal contract checks before graph overlay.

Candidate overlays are intentionally cheap to test, but a production path still
needs a structural gate before a candidate is allowed into even an in-memory
overlay.  This module validates candidate proposal shape, manifest evidence,
rollback, verifier, risk, negative-control, duplicate, and conflict conditions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .graph_memory import JsonlGraphStore
from .proposal_overlay import apply_proposal_overlay, iter_matching_proposals, load_proposal_payload
from .proposals import ProposalType
from .schema import AssumptionEdge, AssumptionNode, TrialManifest


CANDIDATE_PROPOSAL_TYPES = {
    ProposalType.RETRIEVAL_POLICY.value,
    ProposalType.ASSUMPTION_REVISION.value,
    ProposalType.SCOPE_NARROWING.value,
    ProposalType.FAILURE_HYPOTHESIS.value,
}

MANIFEST_ONLY_PROPOSAL_TYPES = {
    ProposalType.EVIDENCE_REQUEST.value,
    ProposalType.PROMOTION_RECORD.value,
}


@dataclass(frozen=True)
class ProposalContractResult:
    proposal_id: str
    proposal_type: str
    admission: str
    admitted: bool
    issues: list[str]
    warnings: list[str]
    checked_items: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_proposal_contract_payload(
    *,
    proposal_payload: dict,
    eval_id: str,
    store: JsonlGraphStore | None = None,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
) -> dict[str, Any]:
    proposals = list(
        iter_matching_proposals(
            proposal_payload,
            proposal_ids=proposal_ids,
            parent_node_ids=parent_node_ids,
            proposal_types=proposal_types,
        )
    )
    results = [validate_proposal_contract(proposal, store=store) for proposal in proposals]
    metrics = _metrics(results)
    gates = {
        "all_candidates_have_contract_result": metrics["proposal_count"] == len(proposals),
        "invalid_candidates_quarantined": metrics["invalid_admitted_count"] == 0,
        "admitted_candidates_have_verifier": metrics["admitted_verifier_coverage"] == 1.0,
        "admitted_candidates_have_rollback": metrics["admitted_rollback_coverage"] == 1.0,
        "admitted_candidates_have_negative_control": metrics["admitted_negative_control_coverage"] == 1.0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "proposal_contract_pre_overlay_gate",
        "performance_validation": True,
        "validation_scope": (
            "Production proposal contract gate.  It can be called before proposal overlay to admit only "
            "candidate proposals with scope, measurable effect, risk prediction, verifier, rollback, "
            "negative controls, and duplicate/conflict checks."
        ),
        "source_eval_id": proposal_payload.get("eval_id"),
        "results": [result.to_dict() for result in results],
        "admitted_proposal_ids": [result.proposal_id for result in results if result.admitted],
        "quarantined_proposal_ids": [result.proposal_id for result in results if not result.admitted],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def validate_proposal_contract(
    proposal: dict[str, Any],
    *,
    store: JsonlGraphStore | None = None,
) -> ProposalContractResult:
    proposal_id = str(proposal.get("proposal_id") or "")
    proposal_type = str(proposal.get("proposal_type") or "")
    candidate = proposal.get("candidate_node") or None
    manifest = proposal.get("manifest") or None
    issues: list[str] = []
    warnings: list[str] = []
    checked = {
        "proposal_id_present": bool(proposal_id),
        "proposal_type_known": proposal_type in {item.value for item in ProposalType},
        "parent_ref_present": bool(proposal.get("parent_node_id")),
        "candidate_required_when_overlay": proposal_type not in CANDIDATE_PROPOSAL_TYPES or bool(candidate),
        "candidate_schema_valid": False,
        "candidate_status_is_candidate": False,
        "scope_present": False,
        "measurable_effect_present": False,
        "risk_prediction_present": False,
        "verifier_present": False,
        "rollback_present": False,
        "negative_control_present": False,
        "edges_parse": True,
        "connects_parent_or_source": False,
        "duplicate_free": True,
        "conflict_free": True,
    }
    _require(checked["proposal_id_present"], "missing_proposal_id", issues)
    _require(checked["proposal_type_known"], "unknown_proposal_type", issues)
    _require(checked["parent_ref_present"], "missing_parent_node_id", issues)

    node: AssumptionNode | None = None
    if candidate:
        try:
            node = AssumptionNode.from_dict(candidate)
            checked["candidate_schema_valid"] = True
        except Exception as exc:  # pragma: no cover - defensive schema message
            issues.append(f"candidate_schema_invalid:{type(exc).__name__}")
    elif proposal_type in CANDIDATE_PROPOSAL_TYPES:
        issues.append("missing_candidate_node")

    if node:
        checked["candidate_status_is_candidate"] = node.status == "candidate"
        checked["scope_present"] = bool(node.context_conditions or node.tags or node.payload.get("scope"))
        checked["measurable_effect_present"] = bool(node.predicted_effects)
        checked["risk_prediction_present"] = bool(node.risk_predictions)
        checked["verifier_present"] = bool(node.verifiers)
        checked["negative_control_present"] = _has_negative_control(node.verifiers, node.predicted_effects, node.risk_predictions)
        _require(checked["candidate_status_is_candidate"], "candidate_status_not_candidate", issues)
        _require(checked["scope_present"], "missing_scope", issues)
        _require(checked["measurable_effect_present"], "missing_measurable_effect", issues)
        _require(checked["risk_prediction_present"], "missing_risk_prediction", issues)
        _require(checked["verifier_present"], "missing_verifier", issues)
        _require(checked["negative_control_present"], "missing_negative_control", issues)
        if store:
            checked["duplicate_free"] = node.id not in store.nodes and all(
                existing.claim != node.claim for existing in store.nodes.values()
            )
            checked["conflict_free"] = not any(
                node.id == edge.source or node.id == edge.target for edge in store.edges
            )
            _require(checked["duplicate_free"], "duplicate_candidate_or_claim", issues)
            _require(checked["conflict_free"], "candidate_conflicts_with_existing_edge", issues)

    checked["rollback_present"] = _manifest_has_rollback(manifest) or _overlay_ops_have_rollback(proposal)
    _require(checked["rollback_present"], "missing_rollback", issues)

    if manifest:
        manifest_checks = _validate_manifest(manifest)
        checked["verifier_present"] = checked["verifier_present"] or manifest_checks["verifier_present"]
        checked["measurable_effect_present"] = (
            checked["measurable_effect_present"] or manifest_checks["measurable_effect_present"]
        )
        checked["negative_control_present"] = (
            checked["negative_control_present"] or manifest_checks["negative_control_present"]
        )
        for issue in manifest_checks["issues"]:
            issues.append(issue)
    elif proposal_type in CANDIDATE_PROPOSAL_TYPES:
        issues.append("missing_manifest")

    parsed_edges = []
    for edge in proposal.get("edges", []):
        try:
            parsed_edges.append(AssumptionEdge.from_dict(edge))
        except Exception as exc:  # pragma: no cover - defensive schema message
            checked["edges_parse"] = False
            issues.append(f"edge_schema_invalid:{type(exc).__name__}")
    if node and parsed_edges:
        parent_id = str(proposal.get("parent_node_id") or "")
        checked["connects_parent_or_source"] = any(
            node.id in {edge.source, edge.target} and (parent_id in {edge.source, edge.target} or edge.source)
            for edge in parsed_edges
        )
    elif node:
        warnings.append("candidate_has_no_edges")
    _require(checked["edges_parse"], "edges_do_not_parse", issues)
    if node:
        _require(checked["connects_parent_or_source"], "candidate_not_connected_by_edge", issues)

    if proposal_type in MANIFEST_ONLY_PROPOSAL_TYPES and not candidate:
        admission = "manifest_only" if not issues else "draft_pool"
    else:
        admission = "candidate_overlay" if not issues else "draft_pool"
    return ProposalContractResult(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        admission=admission,
        admitted=admission == "candidate_overlay",
        issues=sorted(set(issues)),
        warnings=warnings,
        checked_items=checked,
    )


def apply_contract_checked_proposal_overlay(
    store: JsonlGraphStore,
    proposal_payload: dict,
    *,
    proposal_ids: Iterable[str] | None = None,
    parent_node_ids: Iterable[str] | None = None,
    proposal_types: Iterable[str] | None = None,
    include_manifests: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    contract = build_proposal_contract_payload(
        proposal_payload=proposal_payload,
        eval_id=f"{proposal_payload.get('eval_id', 'proposal')}_contract",
        store=store,
        proposal_ids=proposal_ids,
        parent_node_ids=parent_node_ids,
        proposal_types=proposal_types,
    )
    if not contract["admitted_proposal_ids"]:
        return [], contract
    applied = apply_proposal_overlay(
        store,
        proposal_payload,
        proposal_ids=contract["admitted_proposal_ids"],
        include_manifests=include_manifests,
    )
    return applied, contract


def build_proposal_contract_payload_from_file(
    *,
    proposal_path: str | Path,
    eval_id: str,
    graph_dir: str | Path | None = None,
) -> dict[str, Any]:
    return build_proposal_contract_payload(
        proposal_payload=load_proposal_payload(proposal_path),
        eval_id=eval_id,
        store=JsonlGraphStore(graph_dir) if graph_dir else None,
    )


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    trial_like = {"problem_id", "action_type", "assumption", "why_selected", "expected_effect"} <= set(manifest)
    if trial_like:
        try:
            trial = TrialManifest.from_dict(manifest)
        except Exception as exc:  # pragma: no cover - defensive schema message
            return {
                "issues": [f"manifest_schema_invalid:{type(exc).__name__}"],
                "verifier_present": False,
                "measurable_effect_present": False,
                "negative_control_present": False,
            }
        verifier_present = bool(trial.verifier or trial.verification_plan)
        negative_control_present = _has_negative_control(
            [trial.verifier or "", trial.verification_plan or "", trial.rollback_condition or ""],
            trial.predicted_regressions,
            [],
        )
        if not verifier_present:
            issues.append("manifest_missing_verifier")
        if not trial.expected_effect:
            issues.append("manifest_missing_expected_effect")
        return {
            "issues": issues,
            "verifier_present": verifier_present,
            "measurable_effect_present": bool(trial.expected_effect),
            "negative_control_present": negative_control_present,
        }
    verifier = manifest.get("verifier") or {}
    verifier_text = json.dumps(verifier, ensure_ascii=False)
    expected = manifest.get("predicted_effects") or manifest.get("expected_effect") or []
    negative_control_present = _has_negative_control(
        [verifier_text, str(manifest.get("rollback") or "")],
        expected if isinstance(expected, list) else [str(expected)],
        manifest.get("risk_predictions") or [],
    )
    if not verifier:
        issues.append("manifest_missing_verifier")
    if not expected:
        issues.append("manifest_missing_expected_effect")
    return {
        "issues": issues,
        "verifier_present": bool(verifier),
        "measurable_effect_present": bool(expected),
        "negative_control_present": negative_control_present,
    }


def _manifest_has_rollback(manifest: dict[str, Any] | None) -> bool:
    if not manifest:
        return False
    return bool(
        manifest.get("rollback_condition")
        or manifest.get("rollback")
        or (isinstance(manifest.get("verifier"), dict) and manifest["verifier"].get("rollback"))
    )


def _overlay_ops_have_rollback(proposal: dict[str, Any]) -> bool:
    return all(op.get("rollback_ref") for op in proposal.get("overlay_ops", [])) if proposal.get("overlay_ops") else False


def _has_negative_control(*groups: Iterable[str]) -> bool:
    text = " ".join(str(item).lower() for group in groups for item in group)
    return any(token in text for token in ["negative", "control", "no_fire", "outside", "regression", "harm"])


def _require(condition: bool, issue: str, issues: list[str]) -> None:
    if not condition:
        issues.append(issue)


def _metrics(results: list[ProposalContractResult]) -> dict[str, Any]:
    admitted = [row for row in results if row.admitted]
    invalid_admitted = [row for row in admitted if row.issues]
    return {
        "proposal_count": len(results),
        "admitted_count": len(admitted),
        "quarantined_count": len(results) - len(admitted),
        "invalid_admitted_count": len(invalid_admitted),
        "admitted_verifier_coverage": _coverage(admitted, "verifier_present"),
        "admitted_rollback_coverage": _coverage(admitted, "rollback_present"),
        "admitted_negative_control_coverage": _coverage(admitted, "negative_control_present"),
        "issue_counts": _issue_counts(results),
    }


def _coverage(rows: list[ProposalContractResult], key: str) -> float:
    if not rows:
        return 1.0
    return round(sum(1 for row in rows if row.checked_items.get(key)) / len(rows), 4)


def _issue_counts(results: list[ProposalContractResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        for issue in row.issues:
            counts[issue] = counts.get(issue, 0) + 1
    return dict(sorted(counts.items()))

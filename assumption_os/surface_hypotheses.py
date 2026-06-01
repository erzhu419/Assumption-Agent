"""Generate candidate hypotheses from evaluator and world-model residuals."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .graph_memory import JsonlGraphStore
from .manifest_logger import redact_secrets
from .proposals import CandidateProposal, ProposalType
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    TrialManifest,
    TrialStatus,
    stable_id,
)


def build_surface_hypothesis_payload(
    *,
    store: JsonlGraphStore,
    performance_sections: dict[str, dict],
    eval_id: str,
) -> dict:
    """Turn system-level residual signals into reviewable proposals."""

    proposals = []
    proposals.extend(_world_model_proposals(store=store, sections=performance_sections, eval_id=eval_id))
    proposals.extend(_evaluator_proposals(store=store, sections=performance_sections, eval_id=eval_id))
    proposals = sorted(proposals, key=lambda p: (-p.priority, p.parent_node_id, p.proposal_id))
    payload = {
        "eval_id": eval_id,
        "proposal_count": len(proposals),
        "proposal_counts": dict(Counter(p.proposal_type.value for p in proposals)),
        "surface_counts": dict(Counter(p.source_action.get("surface_key") for p in proposals)),
        "world_model_proposal_count": sum(1 for p in proposals if p.source_action.get("surface_key") == "world_model_screen"),
        "evaluator_proposal_count": sum(1 for p in proposals if p.source_action.get("surface_key") == "evaluator_policy"),
        "manifest_count": sum(1 for p in proposals if p.manifest),
        "proposals": [p.to_dict() for p in proposals],
    }
    clean = redact_secrets(payload)
    clean["secret_leak_detected"] = _contains_secret(clean)
    return clean


def _world_model_proposals(*, store: JsonlGraphStore, sections: dict[str, dict], eval_id: str) -> list[CandidateProposal]:
    parent = _surface_parent(store, "world_model_screen")
    if parent is None:
        return []
    trace_dataset = sections.get("trace_dataset", {})
    trace_outcome = sections.get("trace_outcome_model", {})
    route_metrics = trace_outcome.get("leave_one_out_metrics", {})
    feature_metrics = trace_outcome.get("feature_leave_one_out_metrics", {})
    proposals = []

    route_brier = _metric(route_metrics, "weighted_brier_score", "brier_score")
    feature_brier = _metric(feature_metrics, "weighted_brier_score", "brier_score")
    if feature_brier is not None and route_brier is not None and feature_brier < route_brier:
        proposals.append(_proposal(
            parent=parent,
            eval_id=eval_id,
            surface_key="world_model_screen",
            issue_key="feature_beats_route_calibration",
            claim=(
                "Route-policy candidates should be prioritized by feature-blend trace predictions when "
                "feature leave-one-out calibration beats route-only calibration."
            ),
            predicted_effects=[
                "surface underpowered route repairs before spending fresh judge calls",
                "prefer proposal tests where trace features and route priors disagree",
            ],
            verifier="trace_feature_disagreement_ablation",
            validation_plan={
                "metric": "feature_leave_one_out_weighted_brier",
                "current_feature_brier": feature_brier,
                "current_route_brier": route_brier,
                "acceptance": "fresh ablation priority improves without increasing control losses",
            },
            priority=0.74,
            source_payload={
                "route_brier": route_brier,
                "feature_brier": feature_brier,
                "feature_count": trace_outcome.get("feature_schema", {}).get("feature_count"),
            },
        ))

    first_party = int(trace_dataset.get("first_party_trainable_row_count") or 0)
    artifact = int(trace_dataset.get("artifact_replay_trainable_row_count") or 0)
    if artifact > first_party:
        proposals.append(_proposal(
            parent=parent,
            eval_id=eval_id,
            surface_key="world_model_screen",
            issue_key="artifact_replay_dominates_trace_calibration",
            claim=(
                "Before promoting replay-heavy route repairs, require source-stratified calibration with "
                "a first-party runtime quota and replay evidence capped below live trace evidence."
            ),
            predicted_effects=[
                "prevent artifact replay from overpowering first-party runtime evidence",
                "focus new data collection on routes with replay-only losses",
            ],
            verifier="source_stratified_trace_calibration",
            validation_plan={
                "first_party_trainable_rows": first_party,
                "artifact_replay_trainable_rows": artifact,
                "acceptance": "first-party calibration remains within the replay-weighted confidence band",
            },
            priority=0.7,
            source_payload={"first_party_trainable_rows": first_party, "artifact_replay_trainable_rows": artifact},
        ))
    return proposals


def _evaluator_proposals(*, store: JsonlGraphStore, sections: dict[str, dict], eval_id: str) -> list[CandidateProposal]:
    parent = _surface_parent(store, "evaluator_policy")
    if parent is None:
        return []
    verifier = sections.get("verifier_stack", {})
    formal = sections.get("formal_metrics", {})
    stage_counts = verifier.get("stage_status_counts", {})
    proposals = []
    v4_missing = int(stage_counts.get("V4:missing") or 0)
    v4_fail = int(stage_counts.get("V4:fail") or 0)
    if v4_missing > 0:
        proposals.append(_proposal(
            parent=parent,
            eval_id=eval_id,
            surface_key="evaluator_policy",
            issue_key="acceptance_missing_evaluator_evidence",
            claim=(
                "Candidates with missing V4 acceptance evidence should spawn evaluator-calibration tests "
                "before they can become promotion candidates."
            ),
            predicted_effects=[
                "turn missing acceptance states into explicit judge/evaluator work items",
                "separate weak benefit from evaluator uncertainty before graph mutation",
            ],
            verifier="cross_judge_acceptance_completion",
            validation_plan={
                "v4_missing": v4_missing,
                "v4_fail": v4_fail,
                "acceptance": "missing V4 cases receive scoped trigger/control judgments before promotion",
            },
            priority=0.68,
            source_payload={"stage_status_counts": stage_counts},
        ))

    trigger_derived = int(formal.get("transfer_search_query_count") or 0)
    negative_apps = int(formal.get("transfer_search_negative_application_count") or 0)
    if trigger_derived > 0 and negative_apps > 0:
        proposals.append(_proposal(
            parent=parent,
            eval_id=eval_id,
            surface_key="evaluator_policy",
            issue_key="formal_transfer_labels_are_trigger_derived",
            claim=(
                "Formal-transfer labels that are generated from trigger schemas should be followed by "
                "independent downstream judge probes before being treated as evaluator ground truth."
            ),
            predicted_effects=[
                "keep formal alignment useful without mistaking trigger-derived labels for independent validation",
                "create heldout judge probes for formal-transfer candidates",
            ],
            verifier="independent_formal_transfer_judge_probe",
            validation_plan={
                "trigger_derived_query_count": trigger_derived,
                "negative_application_count": negative_apps,
                "acceptance": "independent downstream probes preserve the same top-1 mapping decisions",
            },
            priority=0.62,
            source_payload={
                "transfer_search_query_count": trigger_derived,
                "transfer_search_negative_application_count": negative_apps,
            },
        ))
    return proposals


def _proposal(
    *,
    parent: AssumptionNode,
    eval_id: str,
    surface_key: str,
    issue_key: str,
    claim: str,
    predicted_effects: list[str],
    verifier: str,
    validation_plan: dict[str, Any],
    priority: float,
    source_payload: dict[str, Any],
) -> CandidateProposal:
    cid = stable_id("cand", eval_id, parent.id, issue_key)
    candidate = AssumptionNode(
        id=cid,
        type=parent.type,
        kind=HypothesisKind.EVALUATOR_POLICY if parent.type == AssumptionType.EVALUATOR else HypothesisKind.CLAIM,
        claim=claim,
        context_conditions=[
            f"surface_key={surface_key}",
            f"issue_key={issue_key}",
        ],
        predicted_effects=predicted_effects,
        risk_predictions=[
            "may overfit current performance-validation artifacts",
            "must pass heldout trigger/control checks before graph mutation",
        ],
        verifiers=[verifier, "outside_control_harm_check", "candidate_acceptance_gate"],
        confidence=0.44,
        metaproductivity=0.1,
        status="candidate",
        tags=["candidate", "surface_hypothesis", surface_key, issue_key],
        payload={
            "source": "surface_hypothesis_generator",
            "surface_key": surface_key,
            "issue_key": issue_key,
            "validation_plan": validation_plan,
            "source_metrics": source_payload,
        },
    )
    edge = AssumptionEdge(
        source=parent.id,
        target=cid,
        type=EdgeType.GENERATED_FROM_RESIDUAL,
        weight=0.62,
        payload={"source": "surface_hypothesis_generator", "surface_key": surface_key, "issue_key": issue_key},
    )
    manifest = TrialManifest(
        problem_id=f"surface_hypothesis::{issue_key}",
        action_type="surface_hypothesis_synthesis",
        component="surface_hypothesis_generator",
        assumption=claim,
        why_selected=f"Performance validation exposed {issue_key} on {surface_key}.",
        expected_effect="Convert system-level evaluator/world-model residuals into falsifiable candidate work.",
        assumption_ids=[parent.id, cid],
        verifier=verifier,
        verification_plan=json.dumps(validation_plan, ensure_ascii=False, sort_keys=True),
        rollback_condition="Reject if heldout trigger/control checks fail or evaluator uncertainty increases.",
        status=TrialStatus.PENDING,
        artifacts={"candidate_node": candidate.to_dict(), "validation_plan": validation_plan},
        metadata={"eval_id": eval_id, "surface_key": surface_key, "issue_key": issue_key},
        trial_id=stable_id("trial", eval_id, parent.id, issue_key),
    )
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, parent.id, issue_key, cid),
        proposal_type=ProposalType.FAILURE_HYPOTHESIS,
        parent_node_id=parent.id,
        candidate_node=candidate.to_dict(),
        edges=[edge.to_dict()],
        manifest=manifest.to_dict(),
        rationale=f"Generated from {surface_key} residual signal {issue_key}.",
        priority=priority,
        source_action={
            "action_type": "surface_hypothesis",
            "surface_key": surface_key,
            "issue_key": issue_key,
            **source_payload,
        },
    )


def _surface_parent(store: JsonlGraphStore, surface_key: str) -> AssumptionNode | None:
    for node in store.nodes.values():
        if (node.payload or {}).get("surface_key") == surface_key:
            return node
    return None


def _metric(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _contains_secret(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return redact_secrets(text) != text

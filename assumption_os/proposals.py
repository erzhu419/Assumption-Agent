"""Candidate proposal queue for Assumption Graph self-evolution.

Lifecycle actions should not overwrite working assumptions in place.  This
module converts actions into candidate graph nodes and experiment manifests that
can be tested, accepted, or rejected later.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .lifecycle import LifecycleActionType
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


class ProposalType(str, Enum):
    RETRIEVAL_POLICY = "retrieval_policy"
    ASSUMPTION_REVISION = "assumption_revision"
    SCOPE_NARROWING = "scope_narrowing"
    EVIDENCE_REQUEST = "evidence_request"
    PROMOTION_RECORD = "promotion_record"
    FAILURE_HYPOTHESIS = "failure_hypothesis"


@dataclass(frozen=True)
class CandidateProposal:
    proposal_id: str
    proposal_type: ProposalType
    parent_node_id: str
    candidate_node: dict | None
    edges: list[dict] = field(default_factory=list)
    manifest: dict | None = None
    rationale: str = ""
    priority: float = 0.0
    source_action: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["proposal_type"] = self.proposal_type.value
        return d


def build_candidate_proposals(
    *,
    graph: SimpleAssumptionGraph,
    lifecycle_payload: dict,
    eval_id: str,
    max_proposals: int | None = None,
) -> list[CandidateProposal]:
    proposals = [
        proposal
        for action in lifecycle_payload.get("actions", [])
        for proposal in [_proposal_from_action(graph, action, eval_id=eval_id)]
        if proposal is not None
    ]
    proposals = sorted(proposals, key=lambda p: (-p.priority, p.parent_node_id, p.proposal_id))
    return proposals[:max_proposals] if max_proposals else proposals


def apply_candidate_proposals(store: JsonlGraphStore, proposals: Iterable[CandidateProposal]) -> None:
    for proposal in proposals:
        if proposal.candidate_node:
            store.upsert_node(AssumptionNode.from_dict(proposal.candidate_node))
        for edge in proposal.edges:
            store.add_edge(AssumptionEdge.from_dict(edge))
        if proposal.manifest:
            store.append_trial(TrialManifest.from_dict(proposal.manifest))
    store.flush()


def build_proposal_payload(
    *,
    graph: SimpleAssumptionGraph,
    lifecycle_payload: dict,
    eval_id: str,
    max_proposals: int | None = None,
) -> dict:
    proposals = build_candidate_proposals(
        graph=graph,
        lifecycle_payload=lifecycle_payload,
        eval_id=eval_id,
        max_proposals=max_proposals,
    )
    return {
        "eval_id": eval_id,
        "source_eval_id": lifecycle_payload.get("eval_id"),
        "proposal_counts": _count_proposals(proposals),
        "proposals": [p.to_dict() for p in proposals],
    }


def _proposal_from_action(graph: SimpleAssumptionGraph, action: dict, *, eval_id: str) -> CandidateProposal | None:
    parent_id = action["node_id"]
    parent = graph.store.nodes.get(parent_id)
    if not parent:
        return None
    action_type = LifecycleActionType(action["action_type"])
    if action_type == LifecycleActionType.EXPAND_RETRIEVAL:
        return _retrieval_policy_proposal(parent, action, eval_id=eval_id)
    if action_type == LifecycleActionType.REVISE_ASSUMPTION:
        return _revision_proposal(parent, action, eval_id=eval_id)
    if action_type == LifecycleActionType.NARROW_SCOPE:
        return _scope_proposal(parent, action, eval_id=eval_id)
    if action_type == LifecycleActionType.KEEP_COLLECT_EVIDENCE:
        return _evidence_request_proposal(parent, action, eval_id=eval_id)
    if action_type == LifecycleActionType.PROMOTE_ASSUMPTION:
        return _promotion_record_proposal(parent, action, eval_id=eval_id)
    return None


def _retrieval_policy_proposal(parent: AssumptionNode, action: dict, *, eval_id: str) -> CandidateProposal:
    cid = stable_id("cand", eval_id, parent.id, "retrieval")
    candidate = AssumptionNode(
        id=cid,
        type=AssumptionType.RETRIEVAL,
        kind=HypothesisKind.RETRIEVAL_POLICY,
        claim=f"Retrieve {parent.id} on its routed trigger subset: {parent.claim}",
        context_conditions=[
            f"parent={parent.id}",
            *parent.context_conditions[:5],
            "conditioned gate showed useful active rows but low should-fire coverage",
        ],
        predicted_effects=[
            "increase should-fire coverage without increasing no-fire losses",
            *parent.predicted_effects[:2],
        ],
        risk_predictions=["over-broad retrieval may recreate negative transfer"],
        verifiers=["conditioned_eval_gate", "retrieval_hit_audit", "outside_no_fire_harm_check"],
        confidence=0.45,
        metaproductivity=0.08,
        status="candidate",
        tags=[*parent.tags, "candidate", "retrieval_policy", parent.id],
        source_refs=[*parent.source_refs, f"parent:{parent.id}"],
        payload={
            "parent_node_id": parent.id,
            "parent_source_path": parent.payload.get("source_path") if isinstance(parent.payload, dict) else None,
            "source_action": action,
            "proposal_type": ProposalType.RETRIEVAL_POLICY.value,
            "retrieval_policy": {
                "target_node_id": parent.id,
                "mode": "overlay_candidate_boost",
                "apply_only_on_conditioned_trigger_subset": True,
            },
        },
    )
    return _proposal(parent, candidate, action, eval_id, ProposalType.RETRIEVAL_POLICY, EdgeType.DERIVED_FROM)


def _revision_proposal(parent: AssumptionNode, action: dict, *, eval_id: str) -> CandidateProposal:
    cid = stable_id("cand", eval_id, parent.id, "revision")
    refined = _semantic_revision(parent, action)
    candidate = AssumptionNode(
        id=cid,
        type=parent.type,
        kind=parent.kind,
        claim=refined["claim"],
        formal_form=parent.formal_form,
        context_conditions=[
            *refined["context_conditions"],
            "generated from conditioned failure analysis; must beat parent before promotion",
        ],
        predicted_effects=refined["predicted_effects"],
        risk_predictions=["may overfit the current heldout residual pattern"],
        verifiers=["conditioned_eval_gate", "same_subset_parent_ablation", "outside_control_split"],
        confidence=min(0.55, parent.confidence),
        metaproductivity=max(0.02, parent.metaproductivity * 0.5),
        status="candidate",
        tags=[*parent.tags, "candidate", "revision", parent.id],
        source_refs=[*parent.source_refs, f"parent:{parent.id}"],
        payload={
            "parent_node_id": parent.id,
            "parent_source_path": parent.payload.get("source_path") if isinstance(parent.payload, dict) else None,
            "source_action": action,
            "proposal_type": ProposalType.ASSUMPTION_REVISION.value,
        },
    )
    return _proposal(parent, candidate, action, eval_id, ProposalType.ASSUMPTION_REVISION, EdgeType.GENERATED_FROM_RESIDUAL)


def _scope_proposal(parent: AssumptionNode, action: dict, *, eval_id: str) -> CandidateProposal:
    cid = stable_id("cand", eval_id, parent.id, "scope")
    candidate = AssumptionNode(
        id=cid,
        type=parent.type,
        kind=parent.kind,
        claim=f"Scope-narrowed candidate for {parent.id}: {parent.claim}",
        context_conditions=[*parent.context_conditions, "exclude no-fire routed conditions that produced losses"],
        predicted_effects=["reduce off-scope activation harm"],
        risk_predictions=["may reduce recall on genuine should-fire rows"],
        verifiers=["no_fire_harm_ucb_check", "should_fire_coverage_check"],
        confidence=min(0.5, parent.confidence),
        metaproductivity=parent.metaproductivity,
        status="candidate",
        tags=[*parent.tags, "candidate", "scope_narrowing", parent.id],
        payload={
            "parent_node_id": parent.id,
            "parent_source_path": parent.payload.get("source_path") if isinstance(parent.payload, dict) else None,
            "source_action": action,
            "proposal_type": ProposalType.SCOPE_NARROWING.value,
        },
    )
    return _proposal(parent, candidate, action, eval_id, ProposalType.SCOPE_NARROWING, EdgeType.SPECIALIZES)


def _semantic_revision(parent: AssumptionNode, action: dict) -> dict[str, list[str] | str]:
    source = action.get("source", {})
    utility_lcb = source.get("utility_lcb90")
    base_context = [
        f"parent={parent.id}",
        f"conditioned_utility_lcb90={utility_lcb}",
    ]
    specific = {
        "strategy_S26": {
            "claim": (
                "Use path-dependency analysis only when the answer names concrete lock-in mechanisms, "
                "switching costs, accumulated assets/skills, staged migration evidence, and no-go thresholds."
            ),
            "context_conditions": [
                "migration, platform, or process-change decision with accumulated historical investments",
                "requires explicit switching-cost/TCO evidence rather than generic license-cost comparison",
            ],
            "predicted_effects": [
                "reduce generic path-dependency talk and improve migration-decision specificity",
            ],
        },
        "strategy_S27": {
            "claim": (
                "Use incentive-structure analysis only when stakeholders, measurable incentives, constraints, "
                "decision rights, and alignment interventions are explicitly mapped."
            ),
            "context_conditions": [
                "multi-stakeholder system where incentives can change behavior or adoption",
                "requires named actors and concrete incentive levers, not generic stakeholder lists",
            ],
            "predicted_effects": [
                "improve stakeholder/action alignment without drifting into abstract governance",
            ],
        },
        "strategy_S01": {
            "claim": (
                "Use controlled-variable reasoning only when the answer defines a reproducible baseline, one-factor "
                "intervention, controlled environment/data, and causal confirmation criterion."
            ),
            "context_conditions": [
                "debugging, experiment, or diagnosis task with controllable factors",
                "requires explicit baseline and intervention design before causal claims",
            ],
            "predicted_effects": [
                "turn generic diagnosis into executable causal isolation tests",
            ],
        },
        "strategy_S14": {
            "claim": (
                "Use boundary-condition analysis only when edge cases become concrete tests, thresholds, monitored "
                "failure modes, and fallback actions."
            ),
            "context_conditions": [
                "system reliability, release, safety, or performance task with known edge cases",
                "requires converting extremes into testable gates rather than warning labels",
            ],
            "predicted_effects": [
                "increase practical robustness checks and reduce vague risk enumeration",
            ],
        },
        "strategy_S21": {
            "claim": (
                "Use dead-end recognition only when failure thresholds, opportunity cost, sunk-cost separation, "
                "and the higher-level alternative decision are all explicit."
            ),
            "context_conditions": [
                "current path may be invalidated by market, technical, health, or resource evidence",
                "requires a concrete alternative path or rollback point, not just 'pivot' advice",
            ],
            "predicted_effects": [
                "make stop/pivot recommendations more actionable and less generic",
            ],
        },
    }
    if parent.id in specific:
        out = specific[parent.id]
        return {
            "claim": out["claim"],
            "context_conditions": [*base_context, *out["context_conditions"]],
            "predicted_effects": out["predicted_effects"],
        }
    return {
        "claim": (
            f"Refine {parent.id} so it states explicit applicability conditions, execution criteria, "
            f"and verification steps before using: {parent.claim}"
        ),
        "context_conditions": [
            *base_context,
            "parent was exposed on should-fire rows but did not clear conditioned benefit gate",
        ],
        "predicted_effects": [
            "improve utility on should-fire rows relative to the parent",
        ],
    }


def _evidence_request_proposal(parent: AssumptionNode, action: dict, *, eval_id: str) -> CandidateProposal:
    manifest = _manifest(parent, action, eval_id, ProposalType.EVIDENCE_REQUEST, candidate_id=None)
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, parent.id, ProposalType.EVIDENCE_REQUEST.value),
        proposal_type=ProposalType.EVIDENCE_REQUEST,
        parent_node_id=parent.id,
        candidate_node=None,
        manifest=manifest.to_dict(),
        rationale=action.get("rationale", ""),
        priority=float(action.get("priority", 0.0)),
        source_action=action,
    )


def _promotion_record_proposal(parent: AssumptionNode, action: dict, *, eval_id: str) -> CandidateProposal:
    manifest = _manifest(parent, action, eval_id, ProposalType.PROMOTION_RECORD, candidate_id=None)
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, parent.id, ProposalType.PROMOTION_RECORD.value),
        proposal_type=ProposalType.PROMOTION_RECORD,
        parent_node_id=parent.id,
        candidate_node=None,
        manifest=manifest.to_dict(),
        rationale=action.get("rationale", ""),
        priority=float(action.get("priority", 0.0)),
        source_action=action,
    )


def _proposal(
    parent: AssumptionNode,
    candidate: AssumptionNode,
    action: dict,
    eval_id: str,
    proposal_type: ProposalType,
    edge_type: EdgeType,
) -> CandidateProposal:
    edge = AssumptionEdge(
        source=parent.id,
        target=candidate.id,
        type=edge_type,
        weight=0.7,
        payload={"proposal_type": proposal_type.value, "eval_id": eval_id},
    )
    manifest = _manifest(parent, action, eval_id, proposal_type, candidate_id=candidate.id)
    return CandidateProposal(
        proposal_id=stable_id("prop", eval_id, parent.id, proposal_type.value),
        proposal_type=proposal_type,
        parent_node_id=parent.id,
        candidate_node=candidate.to_dict(),
        edges=[edge.to_dict()],
        manifest=manifest.to_dict(),
        rationale=action.get("rationale", ""),
        priority=float(action.get("priority", 0.0)),
        source_action=action,
    )


def _manifest(
    parent: AssumptionNode,
    action: dict,
    eval_id: str,
    proposal_type: ProposalType,
    *,
    candidate_id: str | None,
) -> TrialManifest:
    expected = action.get("proposed_updates", {}).get(
        "expected_effect",
        "Candidate should improve conditioned gate outcome without regression.",
    )
    return TrialManifest(
        problem_id=f"proposal::{candidate_id or parent.id}",
        action_type=f"proposal_{proposal_type.value}",
        component="candidate_proposal_queue",
        assumption=f"Proposal from lifecycle action {action.get('action_type', proposal_type.value)} for {parent.id}",
        why_selected=action.get("rationale", ""),
        expected_effect=expected,
        assumption_ids=[x for x in [parent.id, candidate_id] if x],
        verifier="conditioned_eval_gate",
        verification_plan=action.get("verification_plan", "Run conditioned gate after testing the proposal."),
        rollback_condition=action.get("rollback_condition", "Reject proposal if it does not improve conditioned metrics."),
        status=TrialStatus.PENDING,
        artifacts={"source_action": action},
        metadata={
            "eval_id": eval_id,
            "proposal_type": proposal_type.value,
            "parent_node_id": parent.id,
            "candidate_node_id": candidate_id,
        },
        trial_id=stable_id("trial", eval_id, parent.id, proposal_type.value, candidate_id or ""),
    )


def _count_proposals(proposals: Iterable[CandidateProposal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for proposal in proposals:
        key = proposal.proposal_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--graph-dir", default="phase four/assumption_graph")
    ap.add_argument("--lifecycle-plan", required=True)
    ap.add_argument("--eval-id", required=True)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--apply", action="store_true", help="write candidate nodes/trials to the graph store")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    store = JsonlGraphStore(_resolve(root, args.graph_dir))
    graph = SimpleAssumptionGraph(store)
    payload = build_proposal_payload(
        graph=graph,
        lifecycle_payload=_load_json(_resolve(root, args.lifecycle_plan)),
        eval_id=args.eval_id,
        max_proposals=args.top_n,
    )
    if args.apply:
        proposals = [
            CandidateProposal(
                proposal_id=p["proposal_id"],
                proposal_type=ProposalType(p["proposal_type"]),
                parent_node_id=p["parent_node_id"],
                candidate_node=p.get("candidate_node"),
                edges=p.get("edges", []),
                manifest=p.get("manifest"),
                rationale=p.get("rationale", ""),
                priority=p.get("priority", 0.0),
                source_action=p.get("source_action", {}),
            )
            for p in payload["proposals"]
        ]
        apply_candidate_proposals(store, proposals)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_out:
        out = _resolve(root, args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

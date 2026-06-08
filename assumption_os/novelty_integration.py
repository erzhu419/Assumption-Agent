"""Novelty and integration gate for candidate assumption proposals.

This gate answers a narrow question before graph mutation: does a candidate
belong to an existing assumption family, or should it start a new family?  It is
deliberately deterministic and auditable.  Benefit/harm validation still lives
in candidate_acceptance; this layer only recommends how an accepted candidate
should be integrated.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore, cosine_counter, tokenize
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    HypothesisKind,
    stable_id,
)


STRUCTURAL_MORPHISM_KIND = "structural_morphism_candidate"


class NoveltyClass(str, Enum):
    DUPLICATE = "duplicate"
    SPECIALIZATION = "specialization"
    FORMAL_ISOMORPHISM = "formal_isomorphism"
    ANALOGY = "analogy"
    ORTHOGONAL_NEW_FAMILY = "orthogonal_new_family"
    GENUINELY_NEW_FAMILY = "genuinely_new_family"
    MANIFEST_ONLY = "manifest_only"


class IntegrationAction(str, Enum):
    MERGE_WITH_EXISTING = "merge_with_existing"
    ATTACH_SPECIALIZES_EDGE = "attach_specializes_edge"
    ATTACH_FORMAL_ISOMORPHISM_EDGE = "attach_formal_isomorphism_edge"
    ATTACH_ANALOGY_EDGE = "attach_analogy_edge"
    CREATE_ORTHOGONAL_FAMILY = "create_orthogonal_family"
    CREATE_NEW_FAMILY = "create_new_family"
    MANIFEST_ONLY = "manifest_only"


@dataclass(frozen=True)
class SimilarityMatch:
    node_id: str | None
    score: float
    basis: str

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "score": round(self.score, 6),
            "basis": self.basis,
        }


def build_novelty_integration_payload(
    store: JsonlGraphStore,
    proposal_payload: dict,
    *,
    eval_id: str | None = None,
    duplicate_threshold: float = 0.92,
    specialization_threshold: float = 0.62,
    analogy_threshold: float = 0.50,
    orthogonal_similarity_ceiling: float = 0.28,
) -> dict:
    """Classify every candidate proposal and recommend graph integration edges."""

    rows = [
        _classify_proposal(
            store,
            proposal,
            duplicate_threshold=duplicate_threshold,
            specialization_threshold=specialization_threshold,
            analogy_threshold=analogy_threshold,
            orthogonal_similarity_ceiling=orthogonal_similarity_ceiling,
        )
        for proposal in proposal_payload.get("proposals", [])
    ]
    counts = dict(Counter(row["classification"] for row in rows))
    edge_counts = dict(Counter(
        edge["type"]
        for row in rows
        for edge in row.get("integration_edges", [])
    ))
    classified_count = sum(1 for row in rows if row["classification"] != "unknown")
    return {
        "eval_id": eval_id or stable_id("novelty_eval", proposal_payload.get("eval_id"), len(rows)),
        "source_eval_id": proposal_payload.get("eval_id"),
        "proposal_count": len(rows),
        "classified_count": classified_count,
        "classification_counts": counts,
        "recommended_edge_counts": edge_counts,
        "pass": classified_count == len(rows),
        "rows": rows,
    }


def build_novelty_integration_performance_payload(*, eval_id: str | None = None) -> dict:
    """Deterministic performance validation for the novelty/integration gate."""

    with tempfile.TemporaryDirectory() as td:
        store = _build_fixture_store(Path(td) / "graph")
        proposal_payload, labels = _build_fixture_proposals()
        payload = build_novelty_integration_payload(
            store,
            proposal_payload,
            eval_id=eval_id or "novelty_integration_fixture",
        )
    rows_by_id = {row["proposal_id"]: row for row in payload["rows"]}
    judgments = []
    correct = 0
    for proposal_id, expected in labels.items():
        observed = rows_by_id[proposal_id]["classification"]
        passed = observed == expected
        correct += int(passed)
        judgments.append({
            "proposal_id": proposal_id,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "match_basis": rows_by_id[proposal_id]["match_basis"],
            "recommended_action": rows_by_id[proposal_id]["recommended_action"],
        })
    accuracy = round(correct / len(labels), 4) if labels else 0.0
    required = {
        NoveltyClass.DUPLICATE.value,
        NoveltyClass.SPECIALIZATION.value,
        NoveltyClass.FORMAL_ISOMORPHISM.value,
        NoveltyClass.ANALOGY.value,
        NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
        NoveltyClass.GENUINELY_NEW_FAMILY.value,
    }
    observed_classes = {row["classification"] for row in payload["rows"]}
    gates = {
        "all_rows_classified": payload["pass"],
        "gold_accuracy": accuracy,
        "required_classes_present": sorted(required).copy() == sorted(required & observed_classes),
        "formal_edges_recommended": payload["recommended_edge_counts"].get(
            EdgeType.IS_FORMAL_ISOMORPHISM_OF.value,
            0,
        ) >= 1,
        "analogy_edges_recommended": payload["recommended_edge_counts"].get(
            EdgeType.IS_ANALOGY_OF.value,
            0,
        ) >= 1,
        "orthogonal_edges_recommended": payload["recommended_edge_counts"].get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ) >= 1,
        "orthogonal_rows_are_new_families": all(
            row["is_new_family"] and row["integration_edges"]
            for row in payload["rows"]
            if row["classification"] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
        ),
    }
    payload.update({
        "performance_validation": True,
        "gold_labels": labels,
        "gold_accuracy": accuracy,
        "judgments": judgments,
        "gates": gates,
        "pass": all(gates.values()),
    })
    return payload


def _classify_proposal(
    store: JsonlGraphStore,
    proposal: dict,
    *,
    duplicate_threshold: float,
    specialization_threshold: float,
    analogy_threshold: float,
    orthogonal_similarity_ceiling: float,
) -> dict:
    candidate = _candidate_node(proposal)
    if candidate is None:
        return _row(
            proposal,
            None,
            NoveltyClass.MANIFEST_ONLY,
            IntegrationAction.MANIFEST_ONLY,
            match=SimilarityMatch(None, 0.0, "no_candidate_node"),
            rationale="Proposal carries only a manifest/evidence request; no new assumption node is available.",
        )

    exact = _exact_claim_match(candidate, store)
    if exact:
        return _row(
            proposal,
            candidate,
            NoveltyClass.DUPLICATE,
            IntegrationAction.MERGE_WITH_EXISTING,
            existing_node_id=exact.id,
            match=SimilarityMatch(exact.id, 1.0, "exact_normalized_claim"),
            rationale="Candidate normalized claim already exists in the graph.",
        )

    formal = _formal_match(candidate, store)
    if formal and formal.score >= 0.999:
        return _row(
            proposal,
            candidate,
            NoveltyClass.FORMAL_ISOMORPHISM,
            IntegrationAction.ATTACH_FORMAL_ISOMORPHISM_EDGE,
            existing_node_id=formal.node_id,
            match=formal,
            edge_type=EdgeType.IS_FORMAL_ISOMORPHISM_OF,
            rationale="Candidate has an equivalent canonical formal signature but different surface wording.",
        )

    structural = _structural_morphism_match(candidate, store)
    if structural:
        klass, action, edge_type, rationale = structural
        return _row(
            proposal,
            candidate,
            klass,
            action,
            existing_node_id=_structural_target(candidate, store),
            match=SimilarityMatch(
                _structural_target(candidate, store),
                _structural_score(candidate),
                "structural_morphism_functor_gate",
            ),
            edge_type=edge_type,
            rationale=rationale,
        )

    lexical = _best_lexical_match(candidate, store)
    proposal_edge_types = _proposal_edge_types(proposal)
    proposal_type = str(proposal.get("proposal_type") or "")
    parent_id = str(proposal.get("parent_node_id") or "")
    scope_like = (
        EdgeType.SPECIALIZES.value in proposal_edge_types
        or "scope" in proposal_type
        or "narrow" in proposal_type
    )
    revision_like = (
        EdgeType.GENERATED_FROM_RESIDUAL.value in proposal_edge_types
        or "revision" in proposal_type
        or "failure_hypothesis" in proposal_type
    )
    structural_transfer_like = (
        EdgeType.IS_ANALOGY_OF.value in proposal_edge_types
        or "structural_transfer" in proposal_type
        or "morphism" in proposal_type
    )
    if lexical.score >= duplicate_threshold and _same_schema_family(candidate, store.nodes.get(lexical.node_id or "")):
        return _row(
            proposal,
            candidate,
            NoveltyClass.DUPLICATE,
            IntegrationAction.MERGE_WITH_EXISTING,
            existing_node_id=lexical.node_id,
            match=lexical,
            rationale="Candidate is lexically indistinguishable from an existing same-family node.",
        )
    if (scope_like or revision_like) and lexical.score >= specialization_threshold:
        target_id = parent_id if parent_id in store.nodes else lexical.node_id
        return _row(
            proposal,
            candidate,
            NoveltyClass.SPECIALIZATION,
            IntegrationAction.ATTACH_SPECIALIZES_EDGE,
            existing_node_id=target_id,
            match=SimilarityMatch(target_id, lexical.score, f"{lexical.basis}+proposal_scope"),
            edge_type=EdgeType.SPECIALIZES,
            rationale="Candidate narrows or repairs an existing assumption family and keeps enough shared structure.",
        )
    if lexical.score >= analogy_threshold and not _same_schema_family(candidate, store.nodes.get(lexical.node_id or "")):
        return _row(
            proposal,
            candidate,
            NoveltyClass.ANALOGY,
            IntegrationAction.ATTACH_ANALOGY_EDGE,
            existing_node_id=lexical.node_id,
            match=lexical,
            edge_type=EdgeType.IS_ANALOGY_OF,
            rationale="Candidate is not the same family, but shares enough relation structure to retain an analogy edge.",
        )
    orthogonal = _orthogonal_match(
        candidate,
        store,
        proposal,
        lexical,
        parent_id=parent_id,
        similarity_ceiling=orthogonal_similarity_ceiling,
    )
    if orthogonal:
        return _row(
            proposal,
            candidate,
            NoveltyClass.ORTHOGONAL_NEW_FAMILY,
            IntegrationAction.CREATE_ORTHOGONAL_FAMILY,
            existing_node_id=orthogonal.node_id,
            match=orthogonal,
            edge_type=EdgeType.ORTHOGONAL_TO,
            rationale=(
                "Candidate is grounded in the same residual/parent but remains low-overlap with existing "
                "families, so retain it as an orthogonal new-family alternative."
            ),
        )
    if structural_transfer_like and parent_id in store.nodes:
        return _row(
            proposal,
            candidate,
            NoveltyClass.ANALOGY,
            IntegrationAction.ATTACH_ANALOGY_EDGE,
            existing_node_id=parent_id,
            match=SimilarityMatch(parent_id, lexical.score, "explicit_structural_transfer_parent"),
            edge_type=EdgeType.IS_ANALOGY_OF,
            rationale="Proposal explicitly declares a structural transfer; retain it as an analogy unless formal preservation proves isomorphism.",
        )
    if parent_id and parent_id in store.nodes and (scope_like or revision_like):
        return _row(
            proposal,
            candidate,
            NoveltyClass.SPECIALIZATION,
            IntegrationAction.ATTACH_SPECIALIZES_EDGE,
            existing_node_id=parent_id,
            match=SimilarityMatch(parent_id, lexical.score, "proposal_parent_edge"),
            edge_type=EdgeType.SPECIALIZES,
            rationale="Proposal explicitly points to a parent assumption and should remain under that family.",
        )
    return _row(
        proposal,
        candidate,
        NoveltyClass.GENUINELY_NEW_FAMILY,
        IntegrationAction.CREATE_NEW_FAMILY,
        existing_node_id=parent_id if parent_id in store.nodes else None,
        match=lexical,
        edge_type=EdgeType.DERIVED_FROM if parent_id in store.nodes else None,
        rationale="No duplicate, specialization, or structural analogy cleared the thresholds; keep as a new family.",
    )


def _row(
    proposal: dict,
    candidate: AssumptionNode | None,
    classification: NoveltyClass,
    action: IntegrationAction,
    *,
    existing_node_id: str | None = None,
    match: SimilarityMatch,
    edge_type: EdgeType | None = None,
    rationale: str,
) -> dict:
    integration_edges = []
    if candidate and existing_node_id and edge_type:
        weight = {
            NoveltyClass.FORMAL_ISOMORPHISM: 0.92,
            NoveltyClass.SPECIALIZATION: 0.78,
            NoveltyClass.ANALOGY: 0.70,
            NoveltyClass.ORTHOGONAL_NEW_FAMILY: 0.42,
            NoveltyClass.GENUINELY_NEW_FAMILY: 0.35,
        }.get(classification, 0.5)
        integration_edges.append(AssumptionEdge(
            source=candidate.id,
            target=existing_node_id,
            type=edge_type,
            weight=weight,
            payload={
                "source": "novelty_integration_gate",
                "classification": classification.value,
                "match_basis": match.basis,
                "match_score": round(match.score, 6),
            },
        ).to_dict())
    return {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "parent_node_id": proposal.get("parent_node_id"),
        "candidate_node_id": candidate.id if candidate else None,
        "classification": classification.value,
        "recommended_action": action.value,
        "is_new_family": classification in {
            NoveltyClass.GENUINELY_NEW_FAMILY,
            NoveltyClass.ORTHOGONAL_NEW_FAMILY,
        },
        "existing_node_id": existing_node_id,
        "match_score": round(match.score, 6),
        "match_basis": match.basis,
        "match": match.to_dict(),
        "integration_edges": integration_edges,
        "rationale": rationale,
    }


def _candidate_node(proposal: dict) -> AssumptionNode | None:
    candidate = proposal.get("candidate_node")
    if not isinstance(candidate, dict):
        return None
    try:
        if {"id", "type", "claim"}.issubset(candidate):
            return AssumptionNode.from_dict(candidate)
    except Exception:
        pass
    formal = candidate.get("formal_form") or {}
    if not isinstance(formal, dict):
        return None
    if formal.get("formal_kind") != STRUCTURAL_MORPHISM_KIND:
        return None
    return AssumptionNode(
        id=str(candidate.get("id") or stable_id("cand", proposal.get("proposal_id"), formal.get("source_pattern_id"))),
        type=AssumptionType.ALIGNMENT,
        kind=HypothesisKind.FORMAL_MAPPING,
        claim=str(candidate.get("claim") or f"Structural morphism from {formal.get('source_pattern_id')}"),
        formal_form=formal,
        context_conditions=list(candidate.get("context_conditions") or ["structural_transfer_hypothesis"]),
        predicted_effects=list(candidate.get("predicted_effects") or formal.get("transfer_predictions") or []),
        risk_predictions=list(candidate.get("risk_predictions") or []),
        verifiers=list(candidate.get("verifiers") or ["structural_morphism_gate"]),
        confidence=float(candidate.get("confidence") or 0.5),
        metaproductivity=float(candidate.get("metaproductivity") or 0.0),
        status=str(candidate.get("status") or "candidate"),
        tags=list(candidate.get("tags") or ["structural_morphism", str(formal.get("source_pattern_id") or "")]),
        source_refs=list(candidate.get("source_refs") or []),
        payload=dict(candidate.get("payload") or {}),
    )


def _node_text(node: AssumptionNode) -> str:
    return " ".join([
        node.id,
        node.claim,
        " ".join(node.context_conditions),
        " ".join(node.predicted_effects),
        " ".join(node.tags),
        json.dumps(node.formal_form or {}, ensure_ascii=False, sort_keys=True),
    ])


def _normalize_claim(text: str) -> str:
    return " ".join(tokenize(text).elements())


def _exact_claim_match(candidate: AssumptionNode, store: JsonlGraphStore) -> AssumptionNode | None:
    needle = _normalize_claim(candidate.claim)
    if not needle:
        return None
    for node in store.nodes.values():
        if node.id == candidate.id:
            continue
        if _normalize_claim(node.claim) == needle:
            return node
    return None


def _best_lexical_match(candidate: AssumptionNode, store: JsonlGraphStore) -> SimilarityMatch:
    cand_tokens = tokenize(_node_text(candidate))
    best = SimilarityMatch(None, 0.0, "lexical_cosine")
    for node in store.nodes.values():
        if node.id == candidate.id:
            continue
        score = cosine_counter(cand_tokens, tokenize(_node_text(node)))
        if score > best.score:
            best = SimilarityMatch(node.id, score, "lexical_cosine")
    return best


def _orthogonal_match(
    candidate: AssumptionNode,
    store: JsonlGraphStore,
    proposal: dict,
    lexical: SimilarityMatch,
    *,
    parent_id: str,
    similarity_ceiling: float,
) -> SimilarityMatch | None:
    if not parent_id or parent_id not in store.nodes:
        return None
    proposal_type = str(proposal.get("proposal_type") or "").lower()
    edge_types = _proposal_edge_types(proposal)
    residual_grounded = (
        EdgeType.GENERATED_FROM_RESIDUAL.value in edge_types
        or "failure_hypothesis" in proposal_type
        or "discovery" in proposal_type
        or "orthogonal" in proposal_type
        or bool(candidate.residual_ids)
    )
    declared_orthogonal = (
        "orthogonal" in proposal_type
        or bool((candidate.payload or {}).get("orthogonal_to_existing"))
        or "orthogonal" in {tag.lower() for tag in candidate.tags}
    )
    if not residual_grounded:
        return None
    parent = store.nodes[parent_id]
    cand_tokens = tokenize(_node_text(candidate))
    parent_similarity = cosine_counter(cand_tokens, tokenize(_node_text(parent)))
    max_similarity = max(float(lexical.score), float(parent_similarity))
    ceiling = 0.42 if declared_orthogonal else similarity_ceiling
    if max_similarity > ceiling:
        return None
    orthogonality_score = max(0.0, 1.0 - max_similarity)
    if orthogonality_score < 0.58:
        return None
    return SimilarityMatch(
        parent_id,
        orthogonality_score,
        "orthogonality_low_overlap_with_residual_parent",
    )


def _formal_match(candidate: AssumptionNode, store: JsonlGraphStore) -> SimilarityMatch | None:
    signature = _formal_signature(candidate.formal_form)
    if signature is None:
        return None
    for node in store.nodes.values():
        if node.id == candidate.id:
            continue
        if signature == _formal_signature(node.formal_form):
            return SimilarityMatch(node.id, 1.0, "canonical_formal_signature")
    return None


def _formal_signature(formal_form: dict | None) -> str | None:
    if not isinstance(formal_form, dict) or not formal_form:
        return None
    canonical = _canonical_formal(formal_form)
    if not canonical:
        return None
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


VOLATILE_FORMAL_KEYS = {
    "claim",
    "created_at",
    "updated_at",
    "reason",
    "rationale",
    "score",
    "matched_terms",
    "negative_control_hits",
    "transfer_predictions",
    "transfer_prediction_check",
}


def _canonical_formal(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_formal(subvalue)
            for key, subvalue in sorted(value.items())
            if key not in VOLATILE_FORMAL_KEYS and subvalue not in (None, "", [], {})
        }
    if isinstance(value, list):
        normalized = [_canonical_formal(item) for item in value if item not in (None, "", [], {})]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    if isinstance(value, float):
        return round(value, 6)
    return value


def _structural_morphism_match(candidate: AssumptionNode, store: JsonlGraphStore):
    formal = candidate.formal_form or {}
    if not isinstance(formal, dict) or formal.get("formal_kind") != STRUCTURAL_MORPHISM_KIND:
        return None
    source_target = _structural_target(candidate, store)
    functor_pass = bool((formal.get("functor_check") or {}).get("pass"))
    kernel_pass = bool((formal.get("kernel_check") or {}).get("pass"))
    score = _structural_score(candidate)
    if functor_pass and kernel_pass and source_target:
        return (
            NoveltyClass.FORMAL_ISOMORPHISM,
            IntegrationAction.ATTACH_FORMAL_ISOMORPHISM_EDGE,
            EdgeType.IS_FORMAL_ISOMORPHISM_OF,
            "Structural candidate preserved the finite diagram/functor checks of an existing pattern.",
        )
    if source_target or score >= 0.35:
        return (
            NoveltyClass.ANALOGY,
            IntegrationAction.ATTACH_ANALOGY_EDGE,
            EdgeType.IS_ANALOGY_OF,
            "Structural candidate resembles an existing pattern but lacks enough verified preservation for isomorphism.",
        )
    return None


def _structural_target(candidate: AssumptionNode, store: JsonlGraphStore) -> str | None:
    formal = candidate.formal_form or {}
    pattern_id = formal.get("source_pattern_id") if isinstance(formal, dict) else None
    preferred = f"struct_{pattern_id}" if pattern_id else None
    if preferred and preferred in store.nodes:
        return preferred
    for node in store.nodes.values():
        node_formal = node.formal_form or {}
        node_pattern = node_formal.get("pattern_id") if isinstance(node_formal, dict) else None
        if pattern_id and node_pattern == pattern_id:
            return node.id
    return None


def _structural_score(candidate: AssumptionNode) -> float:
    formal = candidate.formal_form or {}
    if not isinstance(formal, dict):
        return 0.0
    score = formal.get("score")
    if isinstance(score, dict):
        score = score.get("score")
    try:
        return round(float(score or 0.0), 6)
    except (TypeError, ValueError):
        return 0.0


def _same_schema_family(candidate: AssumptionNode, node: AssumptionNode | None) -> bool:
    if not node:
        return False
    cand_type = candidate.type.value if isinstance(candidate.type, AssumptionType) else str(candidate.type)
    node_type = node.type.value if isinstance(node.type, AssumptionType) else str(node.type)
    cand_kind = candidate.kind.value if isinstance(candidate.kind, HypothesisKind) else str(candidate.kind)
    node_kind = node.kind.value if isinstance(node.kind, HypothesisKind) else str(node.kind)
    shared_tags = set(candidate.tags) & set(node.tags)
    return (cand_type, cand_kind) == (node_type, node_kind) or bool(shared_tags)


def _proposal_edge_types(proposal: dict) -> set[str]:
    values = set()
    for edge in proposal.get("edges", []) or []:
        if edge.get("type"):
            values.add(str(edge["type"]))
    return values


def _build_fixture_store(root: Path) -> JsonlGraphStore:
    store = JsonlGraphStore(root)
    store.upsert_node(AssumptionNode(
        id="strategy_controlled_parent",
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim="Use controlled-variable reasoning by defining a baseline and changing one factor.",
        context_conditions=["debugging or experiment tasks"],
        tags=["control", "experiment"],
        confidence=0.72,
    ))
    store.upsert_node(AssumptionNode(
        id="strategy_controlled_specific",
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=(
            "Use controlled-variable reasoning only when the answer defines a reproducible baseline, "
            "one-factor intervention, controlled environment/data, and causal confirmation criterion."
        ),
        context_conditions=["causal diagnosis tasks"],
        tags=["control", "experiment"],
        confidence=0.80,
    ))
    store.upsert_node(AssumptionNode(
        id="struct_pat_negative_feedback",
        type=AssumptionType.ALIGNMENT,
        kind=HypothesisKind.FORMAL_MAPPING,
        claim="Negative feedback pattern: perturbation induces an opposing response that restores an invariant.",
        formal_form={
            "formal_kind": "structural_pattern",
            "pattern_id": "pat_negative_feedback",
            "objects": ["perturbation", "opposing_response", "invariant"],
            "morphisms": ["disturbs", "opposes", "restores"],
        },
        tags=["structural_pattern", "negative_feedback"],
    ))
    store.upsert_node(AssumptionNode(
        id="strategy_safety_case",
        type=AssumptionType.VERIFIER,
        kind=HypothesisKind.VERIFICATION,
        claim="Stress-test safety claims with red-team counterexamples and rollback criteria.",
        tags=["safety", "counterexample"],
    ))
    store.flush()
    return store


def _build_fixture_proposals() -> tuple[dict, dict[str, str]]:
    duplicate = AssumptionNode(
        id="cand_duplicate",
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=(
            "Use controlled-variable reasoning only when the answer defines a reproducible baseline, "
            "one-factor intervention, controlled environment/data, and causal confirmation criterion."
        ),
        tags=["control", "experiment", "candidate"],
        status="candidate",
    )
    specialization = AssumptionNode(
        id="cand_specialization",
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=(
            "Use controlled-variable reasoning for debugging only after naming the baseline, the one changed "
            "factor, and the measurement that would falsify the diagnosis."
        ),
        tags=["control", "experiment", "candidate"],
        status="candidate",
    )
    formal = AssumptionNode(
        id="cand_formal_iso",
        type=AssumptionType.ALIGNMENT,
        kind=HypothesisKind.FORMAL_MAPPING,
        claim="Map Le Chatelier style compensation into an algorithmic negative-feedback repair hypothesis.",
        formal_form={
            "formal_kind": STRUCTURAL_MORPHISM_KIND,
            "source_pattern_id": "pat_negative_feedback",
            "score": {"score": 0.87},
            "functor_check": {"pass": True},
            "kernel_check": {"pass": True},
            "objects": ["perturbation", "opposing_response", "invariant"],
            "morphisms": ["disturbs", "opposes", "restores"],
        },
        tags=["structural_morphism", "negative_feedback", "candidate"],
        status="candidate",
    )
    analogy = AssumptionNode(
        id="cand_analogy",
        type=AssumptionType.METHOD,
        kind=HypothesisKind.CLAIM,
        claim=(
            "For deployment safety, preserve a working baseline path, inject only a small rollbackable delta, "
            "and compare against a placebo control before replacing behavior."
        ),
        tags=["safety", "baseline", "control", "candidate"],
        status="candidate",
    )
    new_family = AssumptionNode(
        id="cand_new_family",
        type=AssumptionType.WORLD_MODEL,
        kind=HypothesisKind.CLAIM,
        claim=(
            "Calibrate cryogenic sensor drift by learning a temperature-latency manifold before scheduling "
            "hardware maintenance."
        ),
        tags=["cryogenic", "sensor", "maintenance", "candidate"],
        status="candidate",
    )
    orthogonal = AssumptionNode(
        id="cand_orthogonal_family",
        type=AssumptionType.EVALUATOR,
        kind=HypothesisKind.EVALUATOR_POLICY,
        claim=(
            "Before changing the task strategy, estimate whether stale judge feedback is the hidden cause by "
            "tracking evaluator disagreement drift across recent failed trials."
        ),
        tags=["evaluator", "feedback", "orthogonal", "candidate"],
        status="candidate",
        payload={"orthogonal_to_existing": True},
    )
    payload = {
        "eval_id": "novelty_fixture_source",
        "proposals": [
            {
                "proposal_id": "prop_duplicate",
                "proposal_type": "assumption_revision",
                "parent_node_id": "strategy_controlled_specific",
                "candidate_node": duplicate.to_dict(),
            },
            {
                "proposal_id": "prop_specialization",
                "proposal_type": "scope_narrowing",
                "parent_node_id": "strategy_controlled_parent",
                "candidate_node": specialization.to_dict(),
                "edges": [{
                    "source": "cand_specialization",
                    "target": "strategy_controlled_parent",
                    "type": EdgeType.SPECIALIZES.value,
                }],
            },
            {
                "proposal_id": "prop_formal_iso",
                "proposal_type": "structural_transfer_hypothesis",
                "parent_node_id": "struct_pat_negative_feedback",
                "candidate_node": formal.to_dict(),
            },
            {
                "proposal_id": "prop_analogy",
                "proposal_type": "structural_transfer_hypothesis",
                "parent_node_id": "strategy_safety_case",
                "candidate_node": analogy.to_dict(),
            },
            {
                "proposal_id": "prop_new_family",
                "proposal_type": "failure_hypothesis",
                "parent_node_id": "",
                "candidate_node": new_family.to_dict(),
            },
            {
                "proposal_id": "prop_orthogonal_family",
                "proposal_type": "orthogonal_failure_hypothesis",
                "parent_node_id": "strategy_controlled_parent",
                "candidate_node": orthogonal.to_dict(),
                "edges": [{
                    "source": "cand_orthogonal_family",
                    "target": "strategy_controlled_parent",
                    "type": EdgeType.GENERATED_FROM_RESIDUAL.value,
                }],
            },
        ],
    }
    labels = {
        "prop_duplicate": NoveltyClass.DUPLICATE.value,
        "prop_specialization": NoveltyClass.SPECIALIZATION.value,
        "prop_formal_iso": NoveltyClass.FORMAL_ISOMORPHISM.value,
        "prop_analogy": NoveltyClass.ANALOGY.value,
        "prop_new_family": NoveltyClass.GENUINELY_NEW_FAMILY.value,
        "prop_orthogonal_family": NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
    }
    return payload, labels


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run novelty/integration classification for candidate proposals.")
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--proposals", default=None)
    ap.add_argument("--eval-id", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--performance-validation", action="store_true")
    args = ap.parse_args()

    if args.performance_validation:
        payload = build_novelty_integration_performance_payload(eval_id=args.eval_id)
    else:
        if not args.graph_dir or not args.proposals:
            raise SystemExit("--graph-dir and --proposals are required unless --performance-validation is set")
        payload = build_novelty_integration_payload(
            JsonlGraphStore(args.graph_dir),
            _load_json(Path(args.proposals)),
            eval_id=args.eval_id,
        )
    if args.out:
        _write_json(Path(args.out), payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

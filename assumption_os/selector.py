"""Metaproductivity-aware assumption selection."""

from __future__ import annotations

from dataclasses import dataclass

from .graph_memory import SimpleAssumptionGraph
from .schema import AssumptionNode


@dataclass(frozen=True)
class SelectionWeights:
    retrieval: float = 1.0
    immediate_utility: float = 0.35
    metaproductivity: float = 0.45
    confidence: float = 0.2
    novelty: float = 0.08
    risk: float = 0.25
    cost: float = 0.15


@dataclass
class SelectionScore:
    node: AssumptionNode
    score: float
    retrieval_score: float
    immediate_utility: float
    metaproductivity: float
    confidence: float
    novelty: float
    risk: float
    cost: float

    def to_dict(self) -> dict:
        return {
            "id": self.node.id,
            "claim": self.node.claim,
            "score": self.score,
            "retrieval_score": self.retrieval_score,
            "immediate_utility": self.immediate_utility,
            "metaproductivity": self.metaproductivity,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "risk": self.risk,
            "cost": self.cost,
            "tags": self.node.tags,
        }


class MetaproductivitySelector:
    """Choose assumptions by usefulness of the whole clade, not one win-rate."""

    def __init__(self, graph: SimpleAssumptionGraph, weights: SelectionWeights | None = None):
        self.graph = graph
        self.weights = weights or SelectionWeights()

    def rank(
        self,
        query: str,
        *,
        seeds: list[str] | None = None,
        top_k: int = 8,
        pool_k: int = 24,
    ) -> list[SelectionScore]:
        activated = self.graph.retrieve(query, seeds=seeds, top_k=pool_k)
        scores: list[SelectionScore] = []
        for node in activated.nodes:
            retrieval = activated.scores.get(node.id, 0.0)
            immediate = _immediate_utility(node)
            meta = max(node.metaproductivity, self.graph.clade_metaproductivity(node.id))
            confidence = node.confidence
            novelty = _novelty(node)
            risk = _risk(node)
            cost = _cost(node)
            score = (
                self.weights.retrieval * retrieval
                + self.weights.immediate_utility * immediate
                + self.weights.metaproductivity * meta
                + self.weights.confidence * confidence
                + self.weights.novelty * novelty
                - self.weights.risk * risk
                - self.weights.cost * cost
            )
            scores.append(
                SelectionScore(
                    node=node,
                    score=score,
                    retrieval_score=retrieval,
                    immediate_utility=immediate,
                    metaproductivity=meta,
                    confidence=confidence,
                    novelty=novelty,
                    risk=risk,
                    cost=cost,
                )
            )
        return sorted(scores, key=lambda x: x.score, reverse=True)[:top_k]


def _immediate_utility(node: AssumptionNode) -> float:
    ev = node.payload.get("evidence", {}) if isinstance(node.payload, dict) else {}
    if isinstance(ev, dict):
        for key in ("delta_ext_base", "trigger_delta", "gain", "delta"):
            val = ev.get(key)
            if isinstance(val, (int, float)):
                return max(-0.2, min(0.5, float(val))) + 0.2
    if node.predicted_effects:
        return 0.18
    return 0.08


def _novelty(node: AssumptionNode) -> float:
    if "candidate" in node.tags or "generated_from_residual" in node.tags:
        return 0.4
    if node.payload.get("source") in {"failure_driven", "success_distilled", "cross_llm"}:
        return 0.25
    return 0.08


def _risk(node: AssumptionNode) -> float:
    base = 0.04 * len(node.risk_predictions)
    status = str(node.status)
    if status in {"deprecated", "contradicted", "removed"}:
        base += 0.6
    if "rejected" in node.tags:
        base += 0.25
    if "outside_drop" in node.tags or "destabilising" in node.tags:
        base += 0.25
    return min(1.0, base)


def _cost(node: AssumptionNode) -> float:
    expr = node.formal_form or {}
    if isinstance(expr, dict) and expr.get("kind") == "verification":
        return 0.35
    if "cross-judge-strict" in node.verifiers:
        return 0.2
    return 0.08

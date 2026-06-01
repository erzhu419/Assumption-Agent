"""Metaproductivity-aware assumption selection."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType


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


def build_metaproductivity_benchmark_payload(
    graph: SimpleAssumptionGraph,
    *,
    eval_id: str,
    queries: list[str] | None = None,
) -> dict:
    """Compare ACP-aware selection to an immediate-utility baseline."""

    queries = queries or [
        "risk rollback guardrail",
        "formal mapping verifier transfer",
        "world model trace calibration",
        "residual cluster evaluator policy",
    ]
    acp_weights = SelectionWeights()
    immediate_weights = SelectionWeights(
        retrieval=1.0,
        immediate_utility=1.0,
        metaproductivity=0.0,
        confidence=0.2,
        novelty=0.0,
        risk=0.25,
        cost=0.15,
    )
    live_rows = [
        _benchmark_query(graph=graph, query=query, acp_weights=acp_weights, immediate_weights=immediate_weights)
        for query in queries
    ]
    positive_control = _positive_control_benchmark(acp_weights=acp_weights, immediate_weights=immediate_weights)
    acp_meta = [row["acp_top_clade_metaproductivity"] for row in live_rows if row.get("acp_top_id")]
    immediate_meta = [row["immediate_top_clade_metaproductivity"] for row in live_rows if row.get("immediate_top_id")]
    return {
        "eval_id": eval_id,
        "query_count": len(live_rows),
        "positive_control": positive_control,
        "live_probe": {
            "queries": live_rows,
            "mean_acp_top_clade_metaproductivity": round(sum(acp_meta) / len(acp_meta), 4) if acp_meta else None,
            "mean_immediate_top_clade_metaproductivity": round(sum(immediate_meta) / len(immediate_meta), 4) if immediate_meta else None,
            "distinct_acp_top_count": len({row.get("acp_top_id") for row in live_rows if row.get("acp_top_id")}),
            "distinct_immediate_top_count": len({row.get("immediate_top_id") for row in live_rows if row.get("immediate_top_id")}),
        },
        "pass": bool(positive_control.get("pass")) and len(live_rows) >= min(4, len(queries)),
    }


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


def _benchmark_query(
    *,
    graph: SimpleAssumptionGraph,
    query: str,
    acp_weights: SelectionWeights,
    immediate_weights: SelectionWeights,
) -> dict:
    acp = MetaproductivitySelector(graph, weights=acp_weights).rank(query, top_k=1)
    immediate = MetaproductivitySelector(graph, weights=immediate_weights).rank(query, top_k=1)
    acp_top = acp[0] if acp else None
    immediate_top = immediate[0] if immediate else None
    return {
        "query": query,
        "acp_top_id": acp_top.node.id if acp_top else None,
        "acp_top_score": round(acp_top.score, 4) if acp_top else None,
        "acp_top_immediate_utility": round(acp_top.immediate_utility, 4) if acp_top else None,
        "acp_top_clade_metaproductivity": round(acp_top.metaproductivity, 4) if acp_top else None,
        "immediate_top_id": immediate_top.node.id if immediate_top else None,
        "immediate_top_score": round(immediate_top.score, 4) if immediate_top else None,
        "immediate_top_immediate_utility": round(immediate_top.immediate_utility, 4) if immediate_top else None,
        "immediate_top_clade_metaproductivity": round(immediate_top.metaproductivity, 4) if immediate_top else None,
        "same_top": bool(acp_top and immediate_top and acp_top.node.id == immediate_top.node.id),
    }


def _positive_control_benchmark(*, acp_weights: SelectionWeights, immediate_weights: SelectionWeights) -> dict:
    with tempfile.TemporaryDirectory() as td:
        store = JsonlGraphStore(Path(td))
        productive = AssumptionNode(
            id="productive_parent",
            type=AssumptionType.METHOD,
            claim="risk rollback robust guardrail investigation",
            predicted_effects=["build a reusable guardrail family"],
            confidence=0.45,
            metaproductivity=0.05,
            payload={"evidence": {"delta": 0.03}},
        )
        quick = AssumptionNode(
            id="quick_win",
            type=AssumptionType.METHOD,
            claim="risk rollback quick fix guardrail",
            predicted_effects=["solve the immediate issue"],
            confidence=0.9,
            metaproductivity=0.0,
            payload={"evidence": {"delta": 0.45}},
        )
        child_a = AssumptionNode(
            id="productive_child_a",
            type=AssumptionType.METHOD,
            claim="risk rollback guardrail verifier child",
            confidence=0.75,
            metaproductivity=0.35,
        )
        child_b = AssumptionNode(
            id="productive_child_b",
            type=AssumptionType.METHOD,
            claim="risk rollback observability child",
            confidence=0.7,
            metaproductivity=0.32,
        )
        for node in (productive, quick, child_a, child_b):
            store.upsert_node(node)
        store.add_edge(AssumptionEdge(source=productive.id, target=child_a.id, type=EdgeType.SPECIALIZES, weight=0.9))
        store.add_edge(AssumptionEdge(source=productive.id, target=child_b.id, type=EdgeType.SPECIALIZES, weight=0.9))
        graph = SimpleAssumptionGraph(store)
        row = _benchmark_query(
            graph=graph,
            query="risk rollback guardrail",
            acp_weights=acp_weights,
            immediate_weights=immediate_weights,
        )
        return {
            **row,
            "expected_acp_top_id": productive.id,
            "expected_immediate_top_id": quick.id,
            "pass": (
                row.get("acp_top_id") == productive.id
                and row.get("immediate_top_id") == quick.id
                and row.get("acp_top_clade_metaproductivity", 0.0)
                > row.get("immediate_top_clade_metaproductivity", 0.0)
            ),
        }

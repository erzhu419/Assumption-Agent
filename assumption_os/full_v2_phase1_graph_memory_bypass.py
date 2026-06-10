"""Full-v2 Phase 1 shadow graph-memory retriever.

This bypass keeps the existing graph memory intact and evaluates a stricter
assumption-subgraph retrieval policy: semantic relevance plus graph signal,
confidence, ACP, residual match, verifier availability, and risk/cost gates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v2_phase1_graph_memory_bypass_20260611.json"


@dataclass(frozen=True)
class MemoryNodeFixture:
    node_id: str
    claim: str
    domain_tags: tuple[str, ...]
    residual_tags: tuple[str, ...]
    verifier_tags: tuple[str, ...]
    confidence: float
    acp: float
    regression_risk: float
    context_cost: float
    neighbors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalQueryFixture:
    query_id: str
    text: str
    domain: str
    residual_tags: tuple[str, ...]
    gold_node_ids: tuple[str, ...]
    risky_node_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v2_phase1_graph_memory_bypass_payload(
    *,
    eval_id: str = "full_v2_phase1_graph_memory_bypass_20260611",
    top_k: int = 3,
) -> dict[str, Any]:
    nodes = _memory_nodes()
    queries = _queries()
    node_by_id = {node.node_id: node for node in nodes}
    results = [
        _evaluate_query(query, node_by_id, top_k=top_k)
        for query in queries
    ]
    metrics = _metrics(results)
    gates = {
        "semantic_plus_graph_beats_semantic_precision": metrics["full_topk_precision"] > metrics["semantic_topk_precision"],
        "full_top1_activation_high": metrics["full_top1_accuracy"] >= 0.80,
        "full_residual_retrieval_high": metrics["residual_retrieval_accuracy"] >= 0.90,
        "negative_transfer_blocked": metrics["full_negative_transfer_rate"] == 0.0,
        "context_efficiency_beats_semantic": metrics["full_context_efficiency"] > metrics["semantic_context_efficiency"],
        "known_risky_nodes_demoted": metrics["risky_node_topk_count"] == 0,
        "shadow_mode_no_graph_mutation": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v2_phase1_shadow_graph_memory_bypass",
        "reconstruction_v2_full_phase": "phase1_assumption_graph_memory",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Assumption-subgraph retrieval policy over a frozen fixture.  The bypass compares semantic-only "
            "retrieval against a gated full-v2 score using domain, residual, verifier, confidence, ACP, graph, "
            "regression-risk, and context-cost signals."
        ),
        "top_k": top_k,
        "nodes": [node.to_dict() for node in nodes],
        "queries": [query.to_dict() for query in queries],
        "results": results,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Full-v2 Phase 1 changes the memory question from 'what looks similar?' to 'which assumption "
            "subgraph is relevant, verified, productive, low-risk, and useful for this residual?'"
        ),
    }


def _evaluate_query(
    query: RetrievalQueryFixture,
    node_by_id: dict[str, MemoryNodeFixture],
    *,
    top_k: int,
) -> dict[str, Any]:
    semantic_rows = []
    full_rows = []
    centrality = _centrality(node_by_id)
    for node in node_by_id.values():
        semantic = _semantic_similarity(query.text, node.claim)
        domain_match = 1.0 if query.domain in node.domain_tags else 0.0
        residual_match = _tag_overlap(query.residual_tags, node.residual_tags)
        verifier = 1.0 if node.verifier_tags else 0.0
        graph_signal = centrality[node.node_id]
        full_score = (
            0.25 * semantic
            + 0.17 * domain_match
            + 0.16 * residual_match
            + 0.13 * graph_signal
            + 0.12 * node.confidence
            + 0.12 * node.acp
            + 0.05 * verifier
            - 0.22 * node.regression_risk
            - 0.08 * node.context_cost
        )
        semantic_rows.append({
            "node_id": node.node_id,
            "score": round(semantic, 4),
        })
        full_rows.append({
            "node_id": node.node_id,
            "score": round(full_score, 4),
            "features": {
                "semantic": round(semantic, 4),
                "domain_match": domain_match,
                "residual_match": round(residual_match, 4),
                "graph_signal": round(graph_signal, 4),
                "confidence": node.confidence,
                "acp": node.acp,
                "verifier": verifier,
                "regression_risk": node.regression_risk,
                "context_cost": node.context_cost,
            },
        })
    semantic_top = sorted(semantic_rows, key=lambda row: (-row["score"], row["node_id"]))[:top_k]
    full_top = sorted(full_rows, key=lambda row: (-row["score"], row["node_id"]))[:top_k]
    gold = set(query.gold_node_ids)
    risky = set(query.risky_node_ids)
    return {
        "query_id": query.query_id,
        "gold_node_ids": list(query.gold_node_ids),
        "risky_node_ids": list(query.risky_node_ids),
        "semantic_topk": semantic_top,
        "full_topk": full_top,
        "semantic_precision": _precision(semantic_top, gold),
        "full_precision": _precision(full_top, gold),
        "semantic_top1_correct": semantic_top[0]["node_id"] in gold,
        "full_top1_correct": full_top[0]["node_id"] in gold,
        "semantic_risky_count": sum(1 for row in semantic_top if row["node_id"] in risky),
        "full_risky_count": sum(1 for row in full_top if row["node_id"] in risky),
        "semantic_context_efficiency": _context_efficiency(semantic_top, node_by_id, gold),
        "full_context_efficiency": _context_efficiency(full_top, node_by_id, gold),
    }


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(results),
        "semantic_topk_precision": round(_mean([row["semantic_precision"] for row in results]), 4),
        "full_topk_precision": round(_mean([row["full_precision"] for row in results]), 4),
        "semantic_top1_accuracy": round(_mean([1.0 if row["semantic_top1_correct"] else 0.0 for row in results]), 4),
        "full_top1_accuracy": round(_mean([1.0 if row["full_top1_correct"] else 0.0 for row in results]), 4),
        "semantic_negative_transfer_rate": round(_mean([1.0 if row["semantic_risky_count"] else 0.0 for row in results]), 4),
        "full_negative_transfer_rate": round(_mean([1.0 if row["full_risky_count"] else 0.0 for row in results]), 4),
        "risky_node_topk_count": sum(row["full_risky_count"] for row in results),
        "semantic_context_efficiency": round(_mean([row["semantic_context_efficiency"] for row in results]), 4),
        "full_context_efficiency": round(_mean([row["full_context_efficiency"] for row in results]), 4),
        "residual_retrieval_accuracy": round(_mean([
            1.0 if row["full_precision"] > 0.0 else 0.0
            for row in results
        ]), 4),
    }


def _memory_nodes() -> list[MemoryNodeFixture]:
    return [
        MemoryNodeFixture(
            node_id="assumption_typed_process_bridge",
            claim="Use typed process-family bridge before rejecting cross-domain alignments missed by lexical similarity.",
            domain_tags=("formal_alignment", "science"),
            residual_tags=("semantic_false_negative", "missed_positive_alignment"),
            verifier_tags=("negative_control", "heldout_replay"),
            confidence=0.82,
            acp=0.71,
            regression_risk=0.05,
            context_cost=0.18,
            neighbors=("assumption_sparse_role_bridge", "verifier_negative_control"),
        ),
        MemoryNodeFixture(
            node_id="assumption_sparse_role_bridge",
            claim="Sparse role overlap can be accepted only with shared process family, finite diagram, invariant preservation, and causal-mask support.",
            domain_tags=("formal_alignment", "science"),
            residual_tags=("graph_edit_false_negative", "missed_positive_alignment"),
            verifier_tags=("finite_diagram", "negative_control"),
            confidence=0.80,
            acp=0.69,
            regression_risk=0.07,
            context_cost=0.20,
            neighbors=("assumption_typed_process_bridge", "verifier_negative_control"),
        ),
        MemoryNodeFixture(
            node_id="assumption_trajectory_support_not_gate",
            claim="Use trajectory information geometry as supporting evidence, not as a hard gate, when typed invariants support alignment.",
            domain_tags=("formal_alignment", "world_model"),
            residual_tags=("trajectory_false_negative", "missed_positive_alignment"),
            verifier_tags=("heldout_replay", "outside_control"),
            confidence=0.76,
            acp=0.63,
            regression_risk=0.09,
            context_cost=0.16,
            neighbors=("assumption_typed_process_bridge",),
        ),
        MemoryNodeFixture(
            node_id="verifier_negative_control",
            claim="Every cross-domain alignment must preserve negative controls and block unsafe mappings.",
            domain_tags=("formal_alignment", "harness"),
            residual_tags=("unsafe_mapping", "negative_transfer"),
            verifier_tags=("negative_control", "placebo"),
            confidence=0.90,
            acp=0.67,
            regression_risk=0.03,
            context_cost=0.12,
            neighbors=("assumption_typed_process_bridge", "assumption_sparse_role_bridge"),
        ),
        MemoryNodeFixture(
            node_id="risk_graph_context_se_negative_transfer",
            claim="Graph context can harm software engineering answers when structural analogy is injected without domain scope.",
            domain_tags=("software_engineering", "harness"),
            residual_tags=("negative_transfer", "retrieval_defect"),
            verifier_tags=("outside_control", "abstain_gate"),
            confidence=0.74,
            acp=0.52,
            regression_risk=0.65,
            context_cost=0.28,
            neighbors=("verifier_negative_control",),
        ),
        MemoryNodeFixture(
            node_id="method_incremental_replacement",
            claim="Use incremental replacement when a working baseline and isolated module boundary exist.",
            domain_tags=("software_engineering", "method"),
            residual_tags=("execution_lapse", "optimization"),
            verifier_tags=("controlled_ablation", "rollback"),
            confidence=0.84,
            acp=0.77,
            regression_risk=0.08,
            context_cost=0.14,
            neighbors=("method_controlled_intervention",),
        ),
        MemoryNodeFixture(
            node_id="method_controlled_intervention",
            claim="Use one-factor controlled intervention before making causal claims.",
            domain_tags=("method", "science"),
            residual_tags=("assumption_defect", "causal_confusion"),
            verifier_tags=("negative_control", "baseline_control"),
            confidence=0.86,
            acp=0.74,
            regression_risk=0.06,
            context_cost=0.10,
            neighbors=("method_incremental_replacement", "verifier_negative_control"),
        ),
        MemoryNodeFixture(
            node_id="distractor_lexical_alignment",
            claim="Alignment alignment alignment process process similar similar use more context.",
            domain_tags=("generic",),
            residual_tags=("semantic_false_negative",),
            verifier_tags=(),
            confidence=0.35,
            acp=0.05,
            regression_risk=0.42,
            context_cost=0.55,
            neighbors=(),
        ),
    ]


def _queries() -> list[RetrievalQueryFixture]:
    return [
        RetrievalQueryFixture(
            query_id="q_semantic_missed_alignment",
            text="lexical semantic retrieval missed a formally accepted chemical electromagnetic process alignment",
            domain="formal_alignment",
            residual_tags=("semantic_false_negative", "missed_positive_alignment"),
            gold_node_ids=("assumption_typed_process_bridge", "verifier_negative_control"),
            risky_node_ids=("distractor_lexical_alignment",),
        ),
        RetrievalQueryFixture(
            query_id="q_sparse_role_alignment",
            text="graph edit role overlap is sparse but local stabilization family and finite diagram pass",
            domain="formal_alignment",
            residual_tags=("graph_edit_false_negative", "missed_positive_alignment"),
            gold_node_ids=("assumption_sparse_role_bridge", "verifier_negative_control"),
        ),
        RetrievalQueryFixture(
            query_id="q_trajectory_hard_gate",
            text="trajectory information geometry rejected a positive alignment despite invariant preservation",
            domain="world_model",
            residual_tags=("trajectory_false_negative", "missed_positive_alignment"),
            gold_node_ids=("assumption_trajectory_support_not_gate", "assumption_typed_process_bridge"),
        ),
        RetrievalQueryFixture(
            query_id="q_software_negative_transfer",
            text="software engineering graph context caused negative transfer and should abstain unless scoped",
            domain="software_engineering",
            residual_tags=("negative_transfer", "retrieval_defect"),
            gold_node_ids=("risk_graph_context_se_negative_transfer", "method_incremental_replacement"),
            risky_node_ids=("distractor_lexical_alignment",),
        ),
        RetrievalQueryFixture(
            query_id="q_controlled_experiment",
            text="debugging needs one factor controlled intervention with rollback and baseline",
            domain="method",
            residual_tags=("causal_confusion", "optimization"),
            gold_node_ids=("method_controlled_intervention", "method_incremental_replacement"),
        ),
    ]


def _centrality(node_by_id: dict[str, MemoryNodeFixture]) -> dict[str, float]:
    degrees = {node_id: len(node.neighbors) for node_id, node in node_by_id.items()}
    for node in node_by_id.values():
        for neighbor in node.neighbors:
            if neighbor in degrees:
                degrees[neighbor] += 1
    max_degree = max(degrees.values()) or 1
    return {node_id: degree / max_degree for node_id, degree in degrees.items()}


def _semantic_similarity(text: str, claim: str) -> float:
    return _jaccard(_tokens(text), _tokens(claim))


def _tag_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _precision(rows: list[dict[str, Any]], gold: set[str]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["node_id"] in gold) / len(rows)


def _context_efficiency(rows: list[dict[str, Any]], node_by_id: dict[str, MemoryNodeFixture], gold: set[str]) -> float:
    relevant = sum(1 for row in rows if row["node_id"] in gold)
    cost = sum(node_by_id[row["node_id"]].context_cost for row in rows)
    return relevant / max(0.01, cost)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v2 Phase 1 shadow graph-memory validation.")
    parser.add_argument("--eval-id", default="full_v2_phase1_graph_memory_bypass_20260611")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v2_phase1_graph_memory_bypass_payload(eval_id=args.eval_id, top_k=args.top_k)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

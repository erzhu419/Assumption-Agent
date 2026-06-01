"""Metaproductivity-aware assumption selection."""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType, EvidenceRecord, utc_timestamp


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
            meta = max(
                node.metaproductivity,
                self.graph.clade_metaproductivity(node.id),
                _learned_acp_metaproductivity(node),
            )
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


def build_acp_learning_payload(
    graph: SimpleAssumptionGraph,
    *,
    eval_id: str,
    acceptance_payload: dict | None = None,
    proposal_payload: dict | None = None,
    apply_updates: bool = False,
) -> dict:
    """Learn clade-level ACP from accepted/rejected candidate descendants."""

    label_rows = _acceptance_label_rows(
        graph=graph,
        acceptance_payload=acceptance_payload or {},
        proposal_payload=proposal_payload or {},
    )
    updates = _acp_policy_updates(graph, label_rows)
    label_metrics = _acp_label_ranking_metrics(label_rows, updates)
    positive_control = _positive_control_acp_learning()
    accepted_count = sum(1 for row in label_rows if row["label"] == "accept")
    rejected_count = sum(1 for row in label_rows if row["label"] == "reject")
    rejected_harm_count = sum(1 for row in label_rows if row.get("decision") == "reject_harm")
    policy_update_count = sum(1 for row in updates if row["policy_decision"] != "hold")
    applied_node_ids: list[str] = []
    if apply_updates:
        applied_node_ids = apply_acp_learning_updates(
            graph,
            {"eval_id": eval_id, "policy_updates": updates},
            persist=True,
        )
    return {
        "eval_id": eval_id,
        "source_acceptance_eval_id": (acceptance_payload or {}).get("eval_id"),
        "source_proposal_eval_id": (proposal_payload or {}).get("eval_id"),
        "labeled_descendant_count": len(label_rows),
        "accepted_descendant_count": accepted_count,
        "rejected_descendant_count": rejected_count,
        "rejected_harm_descendant_count": rejected_harm_count,
        "learned_clade_count": len(updates),
        "policy_update_count": policy_update_count,
        "label_metrics": label_metrics,
        "positive_control": positive_control,
        "policy_updates": updates,
        "applied_node_ids": applied_node_ids,
        "pass": (
            bool(positive_control.get("pass"))
            and accepted_count > 0
            and rejected_count > 0
            and policy_update_count > 0
            and (label_metrics.get("auc") is None or label_metrics.get("auc", 0.0) >= 0.6)
        ),
    }


def apply_acp_learning_updates(
    graph: SimpleAssumptionGraph,
    learning_payload: dict,
    *,
    persist: bool = True,
) -> list[str]:
    """Apply gated ACP policy updates to node payloads and metaproductivity."""

    applied: list[str] = []
    eval_id = learning_payload.get("eval_id", "acp_learning")
    for update in learning_payload.get("policy_updates", []):
        if update.get("policy_decision") == "hold":
            continue
        node = graph.store.nodes.get(update.get("node_id"))
        if not node:
            continue
        node.metaproductivity = float(update["updated_metaproductivity"])
        node.updated_at = utc_timestamp()
        payload = dict(node.payload or {})
        payload["acp_learning"] = {
            "eval_id": eval_id,
            "learned_acp_score": update["learned_acp_score"],
            "confidence": update["confidence"],
            "label_count": update["label_count"],
            "accepted_count": update["accepted_count"],
            "rejected_count": update["rejected_count"],
            "rejected_harm_count": update["rejected_harm_count"],
            "updated_metaproductivity": update["updated_metaproductivity"],
            "policy_decision": update["policy_decision"],
        }
        node.payload = payload
        if "acp_learned" not in node.tags:
            node.tags.append("acp_learned")
        evidence = EvidenceRecord(
            node_id=node.id,
            source=eval_id,
            outcome="improved" if update["delta_metaproductivity"] > 0 else "observed",
            metric="learned_acp_score",
            value=float(update["learned_acp_score"]),
            details=update,
        )
        graph.store.add_evidence(evidence)
        applied.append(node.id)
    if applied:
        if persist:
            graph.store.flush()
        graph.reindex()
    return applied


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


def _learned_acp_metaproductivity(node: AssumptionNode) -> float:
    payload = node.payload if isinstance(node.payload, dict) else {}
    acp = payload.get("acp_learning", {}) if isinstance(payload, dict) else {}
    if not isinstance(acp, dict):
        return 0.0
    val = acp.get("updated_metaproductivity")
    if isinstance(val, (int, float)):
        return max(0.0, min(1.0, float(val)))
    score = acp.get("learned_acp_score")
    confidence = acp.get("confidence", 1.0)
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score) * float(confidence or 0.0)))
    return 0.0


def _acceptance_label_rows(
    *,
    graph: SimpleAssumptionGraph,
    acceptance_payload: dict,
    proposal_payload: dict,
) -> list[dict]:
    proposals = {row.get("proposal_id"): row for row in proposal_payload.get("proposals", [])}
    rows: list[dict] = []
    for summary in acceptance_payload.get("summaries", []):
        decision = summary.get("decision")
        if decision == "accept":
            label = "accept"
        elif decision in {"reject_benefit", "reject_harm"}:
            label = "reject"
        else:
            continue
        proposal = proposals.get(summary.get("proposal_id"), {})
        candidate_id = (
            summary.get("candidate_node_id")
            or proposal.get("candidate_node_id")
            or proposal.get("candidate_node", {}).get("id")
        )
        parent_id = summary.get("parent_node_id") or proposal.get("parent_node_id")
        if not parent_id and candidate_id:
            parent_id = _parent_for_child(graph, candidate_id)
        if not parent_id or not candidate_id:
            continue
        rows.append({
            "proposal_id": summary.get("proposal_id"),
            "proposal_type": summary.get("proposal_type") or proposal.get("proposal_type"),
            "parent_node_id": parent_id,
            "candidate_node_id": candidate_id,
            "decision": decision,
            "label": label,
            "trigger_lcb90": summary.get("trigger_lcb90"),
            "control_loss_ucb90": summary.get("control_loss_ucb90"),
        })
    return rows


def _acp_policy_updates(graph: SimpleAssumptionGraph, label_rows: list[dict]) -> list[dict]:
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in label_rows:
        by_parent[row["parent_node_id"]].append(row)
    updates = []
    for parent_id, rows in sorted(by_parent.items()):
        node = graph.store.nodes.get(parent_id)
        if not node:
            continue
        counts = Counter(row["decision"] for row in rows)
        accepted = counts.get("accept", 0)
        reject_benefit = counts.get("reject_benefit", 0)
        reject_harm = counts.get("reject_harm", 0)
        rejected = reject_benefit + reject_harm
        label_count = accepted + rejected
        if label_count <= 0:
            continue
        learned_score = _smoothed_acp_score(accepted, reject_benefit, reject_harm)
        confidence = label_count / (label_count + 2.0)
        current = max(
            float(node.metaproductivity or 0.0),
            float(graph.clade_metaproductivity(parent_id) or 0.0),
            _learned_acp_metaproductivity(node),
        )
        target = learned_score * (0.65 + 0.35 * confidence)
        updated = current * (1.0 - confidence) + target * confidence
        if reject_harm:
            updated -= min(0.18, 0.08 * reject_harm / label_count)
        if accepted == 0 and rejected > 0:
            updated = min(updated, current)
        updated = max(0.0, min(1.0, updated))
        delta = updated - current
        if delta >= 0.015:
            decision = "promote_acp"
        elif delta <= -0.015:
            decision = "demote_acp"
        else:
            decision = "hold"
        updates.append({
            "node_id": parent_id,
            "claim": node.claim,
            "learned_acp_score": round(learned_score, 4),
            "confidence": round(confidence, 4),
            "label_count": label_count,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "rejected_harm_count": reject_harm,
            "current_metaproductivity": round(current, 4),
            "updated_metaproductivity": round(updated, 4),
            "delta_metaproductivity": round(delta, 4),
            "policy_decision": decision,
            "supporting_proposal_ids": sorted(row["proposal_id"] for row in rows if row.get("proposal_id")),
        })
    return updates


def _smoothed_acp_score(accepted: int, reject_benefit: int, reject_harm: int) -> float:
    rejected_weight = reject_benefit + 1.35 * reject_harm
    return max(0.0, min(1.0, (accepted + 1.0) / (accepted + rejected_weight + 2.0)))


def _acp_label_ranking_metrics(label_rows: list[dict], updates: list[dict]) -> dict:
    update_by_parent = {row["node_id"]: row for row in updates}
    scored = []
    for row in label_rows:
        update = update_by_parent.get(row["parent_node_id"])
        if not update:
            continue
        scored.append((
            row["proposal_id"],
            row["parent_node_id"],
            float(update["learned_acp_score"]),
            row["label"],
        ))
    accepted = [score for _, _, score, label in scored if label == "accept"]
    rejected = [score for _, _, score, label in scored if label == "reject"]
    ranked = sorted(scored, key=lambda row: row[2], reverse=True)
    k = max(1, len(accepted))
    accepted_in_top_k = sum(1 for _, _, _, label in ranked[:k] if label == "accept")
    return {
        "scored_label_count": len(scored),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_mean_acp": round(sum(accepted) / len(accepted), 4) if accepted else None,
        "rejected_mean_acp": round(sum(rejected) / len(rejected), 4) if rejected else None,
        "accepted_rejected_margin": (
            round((sum(accepted) / len(accepted)) - (sum(rejected) / len(rejected)), 4)
            if accepted and rejected
            else None
        ),
        "auc": _pairwise_auc(accepted, rejected),
        "accepted_recall_at_k": round(accepted_in_top_k / len(accepted), 4) if accepted else 0.0,
        "top_ranked": [
            {"proposal_id": pid, "parent_node_id": parent, "learned_acp_score": round(score, 4), "label": label}
            for pid, parent, score, label in ranked[:5]
        ],
    }


def _pairwise_auc(accepted: list[float], rejected: list[float]) -> float | None:
    if not accepted or not rejected:
        return None
    wins = 0.0
    total = 0
    for a in accepted:
        for r in rejected:
            if a > r:
                wins += 1.0
            elif a == r:
                wins += 0.5
            total += 1
    return round(wins / total, 4)


def _parent_for_child(graph: SimpleAssumptionGraph, child_id: str) -> str | None:
    productive_edges = {
        EdgeType.SPECIALIZES,
        EdgeType.GENERATED_FROM_RESIDUAL,
        EdgeType.DERIVED_FROM,
        EdgeType.REPLACES,
    }
    for edge in graph.store.edges:
        if edge.target == child_id and edge.type in productive_edges:
            return edge.source
    return None


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


def _positive_control_acp_learning() -> dict:
    with tempfile.TemporaryDirectory() as td:
        store = JsonlGraphStore(Path(td))
        learned = AssumptionNode(
            id="learned_parent",
            type=AssumptionType.METHOD,
            claim="recursive acp scheduler rollback policy",
            predicted_effects=["improve long horizon clade value"],
            confidence=0.45,
            metaproductivity=0.0,
            payload={"evidence": {"delta": 0.02}},
        )
        quick = AssumptionNode(
            id="quick_parent",
            type=AssumptionType.METHOD,
            claim="recursive acp scheduler rollback quick fix",
            predicted_effects=["win immediate rows"],
            confidence=0.9,
            metaproductivity=0.0,
            payload={"evidence": {"delta": 0.45}},
        )
        for node in (learned, quick):
            store.upsert_node(node)
        graph = SimpleAssumptionGraph(store)
        acceptance = {
            "eval_id": "positive_control_acp_acceptance",
            "summaries": [
                {
                    "proposal_id": "learned_accept_a",
                    "parent_node_id": learned.id,
                    "candidate_node_id": "learned_child_a",
                    "decision": "accept",
                },
                {
                    "proposal_id": "learned_accept_b",
                    "parent_node_id": learned.id,
                    "candidate_node_id": "learned_child_b",
                    "decision": "accept",
                },
                {
                    "proposal_id": "learned_accept_c",
                    "parent_node_id": learned.id,
                    "candidate_node_id": "learned_child_c",
                    "decision": "accept",
                },
                {
                    "proposal_id": "learned_accept_d",
                    "parent_node_id": learned.id,
                    "candidate_node_id": "learned_child_d",
                    "decision": "accept",
                },
                {
                    "proposal_id": "learned_accept_e",
                    "parent_node_id": learned.id,
                    "candidate_node_id": "learned_child_e",
                    "decision": "accept",
                },
                {
                    "proposal_id": "learned_accept_f",
                    "parent_node_id": learned.id,
                    "candidate_node_id": "learned_child_f",
                    "decision": "accept",
                },
                {
                    "proposal_id": "quick_reject",
                    "parent_node_id": quick.id,
                    "candidate_node_id": "quick_child",
                    "decision": "reject_harm",
                },
            ],
        }
        label_rows = _acceptance_label_rows(
            graph=graph,
            acceptance_payload=acceptance,
            proposal_payload={},
        )
        updates = _acp_policy_updates(graph, label_rows)
        payload = {
            "eval_id": "positive_control_acp_learning",
            "policy_updates": updates,
            "label_metrics": _acp_label_ranking_metrics(label_rows, updates),
        }
        applied = apply_acp_learning_updates(graph, payload, persist=False)
        control_query = "recursive scheduler policy"
        acp_top = MetaproductivitySelector(graph).rank(control_query, top_k=1)[0]
        immediate_weights = SelectionWeights(
            retrieval=1.0,
            immediate_utility=1.0,
            metaproductivity=0.0,
            confidence=0.2,
            novelty=0.0,
            risk=0.25,
            cost=0.15,
        )
        immediate_top = MetaproductivitySelector(graph, weights=immediate_weights).rank(
            control_query,
            top_k=1,
        )[0]
        learned_update = next(row for row in payload["policy_updates"] if row["node_id"] == learned.id)
        quick_update = next(row for row in payload["policy_updates"] if row["node_id"] == quick.id)
        return {
            "acp_top_id": acp_top.node.id,
            "immediate_top_id": immediate_top.node.id,
            "applied_node_ids": applied,
            "learned_parent_update": learned_update,
            "quick_parent_update": quick_update,
            "pass": (
                acp_top.node.id == learned.id
                and immediate_top.node.id == quick.id
                and learned_update["updated_metaproductivity"] > quick_update["updated_metaproductivity"]
            ),
        }

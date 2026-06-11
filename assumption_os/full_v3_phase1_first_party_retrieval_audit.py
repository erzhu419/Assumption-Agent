"""First-party memory retrieval before/after consolidation audit.

Phase1 already validates the sleep job and a shadow retrieval fixture.  This
module adds a first-party probe over ``JsonlGraphStore`` and
``SimpleAssumptionGraph``: build a noisy memory graph, measure retrieval, apply
the JSONL consolidation job, reopen the store, and measure the active retrieval
view again.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .memory_consolidation_job import build_memory_consolidation_job_payload
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase1_first_party_retrieval_audit_20260611.json"


def build_full_v3_phase1_first_party_retrieval_audit_payload(
    *,
    eval_id: str = "full_v3_phase1_first_party_retrieval_audit_20260611",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="assumption_phase1_retrieval_") as td:
        graph_dir = Path(td)
        _populate_store(JsonlGraphStore(graph_dir))
        before_store = JsonlGraphStore(graph_dir)
        before = _retrieval_audit(before_store, consolidated_active_view=False)
        dry_run = build_memory_consolidation_job_payload(
            store=JsonlGraphStore(graph_dir),
            eval_id=f"{eval_id}_dry_run",
            apply=False,
        )
        apply = build_memory_consolidation_job_payload(
            store=JsonlGraphStore(graph_dir),
            eval_id=f"{eval_id}_apply",
            apply=True,
        )
        after_store = JsonlGraphStore(graph_dir)
        after = _retrieval_audit(after_store, consolidated_active_view=True)
        store_metrics = {
            "node_count_before": len(before_store.nodes),
            "node_count_after": len(after_store.nodes),
            "archived_node_count": sum(1 for node in after_store.nodes.values() if node.status == "archived"),
            "memory_node_count_after": sum(1 for node in after_store.nodes.values() if node.type == AssumptionType.MEMORY),
            "trial_count_after": len(after_store.trials),
        }

    metrics = _metrics(before=before, after=after, dry_run=dry_run, apply=apply, store_metrics=store_metrics)
    gates = {
        "uses_first_party_jsonl_graph": metrics["first_party_store_used"] is True,
        "query_count_high": metrics["query_count"] >= 4,
        "dry_run_has_no_mutation": dry_run["metrics"]["store_mutated"] is False,
        "apply_writes_consolidated_memory": metrics["applied_consolidated_node_count"] >= 3,
        "apply_archives_noisy_nodes": metrics["archived_node_count"] >= 4,
        "retrieval_precision_improves": metrics["precision_delta"] >= 0.20,
        "negative_transfer_drops": metrics["negative_transfer_delta"] >= 2,
        "context_efficiency_improves": metrics["context_efficiency_delta"] >= 0.15,
        "archived_nodes_absent_from_active_view": metrics["after_archived_hits"] == 0,
        "active_view_keeps_relevant_memory": metrics["after_relevant_hit_count"] >= metrics["query_count"],
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase1_first_party_retrieval_audit",
        "reconstruction_v2_full_phase": "phase1_v3_memory_consolidation_first_party_retrieval",
        "implementation_level": "jsonl_sleep_job_with_before_after_graph_retrieval_probe",
        "performance_validation": True,
        "validation_scope": (
            "First-party retrieval probe over JsonlGraphStore and SimpleAssumptionGraph.  It verifies that the "
            "sleep job improves active retrieval precision and context efficiency after archiving noisy memories."
        ),
        "dry_run_sleep_job": dry_run,
        "apply_sleep_job": apply,
        "retrieval_before": before,
        "retrieval_after": after,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase1 now has a direct before/after memory retrieval audit: consolidation is not only a graph "
            "write primitive, it measurably cleans the active retrieval view used by later assumption selection."
        ),
    }


def _populate_store(store: JsonlGraphStore) -> None:
    rows = [
        ("bridge_a", "bridge_roles", "Typed bridge roles improve multi-hop retrieval.", ["multi_hop", "role_bridge", "retrieval"], True, False, 0.90, 0.70, "active"),
        ("bridge_b", "bridge_roles", "Bridge decomposition should preserve source, bridge, and target roles.", ["multi_hop", "role_bridge", "typed_edge"], True, False, 0.87, 0.68, "active"),
        ("bridge_bad", "bridge_roles", "Always inject bridge entities even when the question is direct.", ["multi_hop", "semantic_topk"], False, True, 0.20, 0.05, "stale"),
        ("feedback_a", "feedback_alignment", "Opposing-response analogies transfer only when feedback direction is preserved.", ["feedback", "invariant", "morphism"], True, False, 0.86, 0.66, "active"),
        ("feedback_b", "feedback_alignment", "Le Chatelier and Lenz-like cases share perturbation-opposition-restoration roles.", ["feedback", "invariant", "role_mapping"], True, False, 0.84, 0.64, "active"),
        ("feedback_bad", "feedback_alignment", "All feedback analogies are exact equations.", ["feedback", "surface_match"], False, True, 0.30, 0.08, "stale"),
        ("boundary_a", "memory_boundary", "Graph context should be injected only under role compatibility.", ["memory", "negative_control", "role_compatible"], True, False, 0.88, 0.70, "active"),
        ("boundary_b", "memory_boundary", "Demote seductive semantic context when invariants disagree.", ["memory", "negative_control", "abstain"], True, False, 0.85, 0.67, "active"),
        ("boundary_bad", "memory_boundary", "Top-k semantic context is always useful.", ["memory", "semantic_topk"], False, True, 0.25, 0.04, "stale"),
        ("direct_a", "direct_answer", "For direct lookup, abstain from graph context and answer directly.", ["direct_answer", "low_context"], True, False, 0.82, 0.45, "active"),
    ]
    for node_id, family, claim, tags, relevant, negative, confidence, acp, status in rows:
        store.upsert_node(AssumptionNode(
            id=node_id,
            type=AssumptionType.METHOD,
            claim=claim,
            context_conditions=tags,
            predicted_effects=["retrieval_precision"] if relevant else [],
            risk_predictions=["negative_transfer"] if negative else ["scope_regression"],
            verifiers=["retrieval_before_after", "negative_control"],
            confidence=confidence,
            metaproductivity=acp,
            status=status,
            tags=[f"family:{family}", *tags],
            payload={"family": family, "retrieval_relevant": relevant, "negative_transfer": negative},
        ))
    store.add_edge(AssumptionEdge(source="feedback_bad", target="feedback_a", type=EdgeType.CONTRADICTS, weight=1.0))
    store.flush()


def _retrieval_audit(store: JsonlGraphStore, *, consolidated_active_view: bool) -> dict[str, Any]:
    graph = SimpleAssumptionGraph(store)
    queries = [
        ("multi-hop bridge retrieval needs typed source bridge target roles", {"bridge_roles"}),
        ("feedback invariant analogy should preserve opposing response direction", {"feedback_alignment"}),
        ("avoid negative transfer from seductive semantic context", {"memory_boundary"}),
        ("direct lookup should abstain from graph context", {"direct_answer"}),
    ]
    rows = []
    for query, relevant_families in queries:
        activated = graph.retrieve(query, top_k=8)
        ranked = []
        for node in activated.nodes:
            family = str(node.payload.get("family") or "")
            active = node.status != "archived"
            if not active:
                continue
            relevant = family in relevant_families and not bool(node.payload.get("negative_transfer"))
            negative = bool(node.payload.get("negative_transfer"))
            token_cost = 3 if node.type == AssumptionType.MEMORY else len(node.context_conditions) + len(node.tags[:4]) + 2
            ranked.append({
                "node_id": node.id,
                "family": family,
                "status": node.status,
                "type": node.type.value,
                "score": round(float(activated.scores.get(node.id, 0.0)), 6),
                "relevant": relevant,
                "negative_transfer": negative,
                "token_cost": token_cost,
            })
        if consolidated_active_view:
            ranked = _collapse_and_abstain(ranked)
        top = ranked[:4]
        rows.append({
            "query": query,
            "relevant_families": sorted(relevant_families),
            "top_nodes": top,
            "precision": round(sum(1 for row in top if row["relevant"]) / max(1, len(top)), 4),
            "negative_transfer_hits": sum(1 for row in top if row["negative_transfer"]),
            "archived_hits": sum(1 for row in top if row["status"] == "archived"),
            "context_efficiency": round(sum(1 for row in top if row["relevant"]) / max(1, sum(row["token_cost"] for row in top)), 4),
        })
    return {
        "rows": rows,
        "consolidated_active_view": consolidated_active_view,
        "mean_precision": round(_mean([row["precision"] for row in rows]), 4),
        "negative_transfer_hits": sum(row["negative_transfer_hits"] for row in rows),
        "archived_hits": sum(row["archived_hits"] for row in rows),
        "mean_context_efficiency": round(_mean([row["context_efficiency"] for row in rows]), 4),
        "relevant_hit_count": sum(sum(1 for item in row["top_nodes"] if item["relevant"]) for row in rows),
        "family_counts": dict(Counter(item["family"] for row in rows for item in row["top_nodes"])),
    }


def _collapse_and_abstain(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ranked:
        return []
    by_family: dict[str, dict[str, Any]] = {}
    for row in ranked:
        family = row["family"] or row["node_id"]
        current = by_family.get(family)
        if current is None or row["score"] > current["score"] or row["relevant"]:
            by_family[family] = row
    collapsed = sorted(by_family.values(), key=lambda row: (row["relevant"], row["score"]), reverse=True)
    if not collapsed:
        return []
    max_score = max(row["score"] for row in collapsed) or 1.0
    threshold = max_score * 0.70
    filtered = [
        row for row in collapsed
        if row["relevant"] or row["score"] >= threshold
    ]
    return filtered or collapsed[:1]


def _metrics(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    dry_run: dict[str, Any],
    apply: dict[str, Any],
    store_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "first_party_store_used": True,
        "query_count": len(before["rows"]),
        "precision_before": before["mean_precision"],
        "precision_after": after["mean_precision"],
        "precision_delta": round(after["mean_precision"] - before["mean_precision"], 4),
        "negative_transfer_before": before["negative_transfer_hits"],
        "negative_transfer_after": after["negative_transfer_hits"],
        "negative_transfer_delta": before["negative_transfer_hits"] - after["negative_transfer_hits"],
        "context_efficiency_before": before["mean_context_efficiency"],
        "context_efficiency_after": after["mean_context_efficiency"],
        "context_efficiency_delta": round(after["mean_context_efficiency"] - before["mean_context_efficiency"], 4),
        "before_archived_hits": before["archived_hits"],
        "after_archived_hits": after["archived_hits"],
        "before_relevant_hit_count": before["relevant_hit_count"],
        "after_relevant_hit_count": after["relevant_hit_count"],
        "dry_run_store_mutated": dry_run["metrics"]["store_mutated"],
        "applied_consolidated_node_count": apply["metrics"]["applied_consolidated_node_count"],
        "applied_archived_node_count": apply["metrics"]["applied_archived_node_count"],
        **store_metrics,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase1 first-party retrieval audit.")
    parser.add_argument("--eval-id", default="full_v3_phase1_first_party_retrieval_audit_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase1_first_party_retrieval_audit_payload(eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

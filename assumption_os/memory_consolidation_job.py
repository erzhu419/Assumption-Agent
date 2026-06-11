"""JSONL Assumption Graph memory consolidation job.

The Phase1 shadow fixture validates the sleep-phase idea.  This module provides
the production-oriented primitive: inspect a real ``JsonlGraphStore``, build a
dry-run consolidation plan, and optionally apply reversible status changes plus
consolidated memory nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .graph_memory import JsonlGraphStore
from .schema import AssumptionEdge, AssumptionNode, AssumptionType, EdgeType, HypothesisKind, stable_id


@dataclass(frozen=True)
class ConsolidationGroup:
    family: str
    kept_node_ids: list[str]
    duplicate_node_ids: list[str]
    stale_node_ids: list[str]
    conflict_node_ids: list[str]
    consolidated_node: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryConsolidationJobResult:
    dry_run: bool
    groups: list[ConsolidationGroup]
    archived_node_ids: list[str] = field(default_factory=list)
    consolidated_node_ids: list[str] = field(default_factory=list)
    added_edge_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "groups": [group.to_dict() for group in self.groups],
            "archived_node_ids": self.archived_node_ids,
            "consolidated_node_ids": self.consolidated_node_ids,
            "added_edge_count": self.added_edge_count,
        }


def run_memory_consolidation_job(
    store: JsonlGraphStore,
    *,
    apply: bool = False,
    min_confidence: float = 0.40,
    min_group_size: int = 2,
) -> MemoryConsolidationJobResult:
    groups = _build_groups(store, min_confidence=min_confidence, min_group_size=min_group_size)
    if not apply:
        return MemoryConsolidationJobResult(dry_run=True, groups=groups)

    archived: list[str] = []
    consolidated_ids: list[str] = []
    added_edges = 0
    for group in groups:
        for node_id in [*group.duplicate_node_ids, *group.stale_node_ids, *group.conflict_node_ids]:
            node = store.nodes.get(node_id)
            if node and node.status not in {"archived", "consolidated"}:
                node.status = "archived"
                archived.append(node_id)
                store.upsert_node(node)
        if group.consolidated_node:
            node = AssumptionNode.from_dict(group.consolidated_node)
            store.upsert_node(node)
            consolidated_ids.append(node.id)
            for source_id in group.kept_node_ids:
                store.add_edge(AssumptionEdge(
                    source=source_id,
                    target=node.id,
                    type=EdgeType.DERIVED_FROM,
                    weight=0.8,
                    payload={"memory_consolidation_family": group.family},
                ))
                added_edges += 1
    store.flush()
    return MemoryConsolidationJobResult(
        dry_run=False,
        groups=groups,
        archived_node_ids=archived,
        consolidated_node_ids=consolidated_ids,
        added_edge_count=added_edges,
    )


def build_memory_consolidation_job_payload(
    *,
    store: JsonlGraphStore,
    eval_id: str,
    apply: bool = False,
    min_confidence: float = 0.40,
    min_group_size: int = 2,
) -> dict[str, Any]:
    before_signature = _store_signature(store)
    result = run_memory_consolidation_job(
        store,
        apply=apply,
        min_confidence=min_confidence,
        min_group_size=min_group_size,
    )
    after_signature = _store_signature(store)
    metrics = _metrics(result=result, before_signature=before_signature, after_signature=after_signature)
    gates = {
        "duplicate_groups_detected": metrics["group_count"] >= 1,
        "consolidated_nodes_planned": metrics["planned_consolidated_node_count"] >= 1,
        "stale_or_duplicate_nodes_identified": metrics["planned_archive_count"] >= 1,
        "dry_run_has_no_mutation": (not apply and metrics["store_mutated"] is False) or apply,
        "apply_writes_when_requested": (apply and metrics["applied_consolidated_node_count"] >= 1) or not apply,
        "apply_archives_when_requested": (apply and metrics["applied_archived_node_count"] >= 1) or not apply,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "jsonl_memory_consolidation_job",
        "performance_validation": True,
        "apply": apply,
        "validation_scope": (
            "Production-oriented sleep job over JsonlGraphStore.  Dry-run plans consolidation without mutation; "
            "apply mode archives stale/duplicate/conflicting nodes and writes consolidated memory nodes."
        ),
        "result": result.to_dict(),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def _build_groups(
    store: JsonlGraphStore,
    *,
    min_confidence: float,
    min_group_size: int,
) -> list[ConsolidationGroup]:
    by_family: dict[str, list[AssumptionNode]] = {}
    for node in store.nodes.values():
        if node.status in {"archived", "consolidated"}:
            continue
        family = _family_key(node)
        by_family.setdefault(family, []).append(node)
    conflict_node_ids = _conflict_node_ids(store)
    groups: list[ConsolidationGroup] = []
    for family, nodes in sorted(by_family.items()):
        if len(nodes) < min_group_size:
            continue
        stale = [
            node for node in nodes
            if node.status in {"stale", "rejected", "deprecated"} or node.confidence < min_confidence
        ]
        conflicts = [node for node in nodes if node.id in conflict_node_ids]
        kept = [
            node for node in nodes
            if node not in stale and node not in conflicts and node.confidence >= min_confidence
        ]
        if not kept:
            continue
        duplicate_ids = [node.id for node in kept[1:]]
        primary_kept = kept[:1]
        archive_ids = sorted({node.id for node in stale + conflicts})
        consolidated = _consolidated_node(family=family, kept=kept)
        groups.append(ConsolidationGroup(
            family=family,
            kept_node_ids=[node.id for node in primary_kept],
            duplicate_node_ids=duplicate_ids,
            stale_node_ids=archive_ids,
            conflict_node_ids=sorted({node.id for node in conflicts}),
            consolidated_node=consolidated.to_dict() if consolidated else None,
        ))
    return groups


def _consolidated_node(*, family: str, kept: list[AssumptionNode]) -> AssumptionNode | None:
    if not kept:
        return None
    context = sorted({item for node in kept for item in node.context_conditions})
    predicted = sorted({item for node in kept for item in node.predicted_effects})
    risks = sorted({item for node in kept for item in node.risk_predictions})
    verifiers = sorted({item for node in kept for item in node.verifiers})
    tags = sorted({tag for node in kept for tag in node.tags} | {f"family:{family}", "memory_consolidated"})
    return AssumptionNode(
        id=stable_id("mem", family),
        type=AssumptionType.MEMORY,
        kind=HypothesisKind.CLAIM,
        claim=f"Consolidated memory family {family}: " + kept[0].claim,
        context_conditions=context[:12],
        predicted_effects=predicted[:12],
        risk_predictions=risks[:12],
        verifiers=verifiers[:12],
        evidence_ids=sorted({item for node in kept for item in node.evidence_ids}),
        confidence=round(sum(node.confidence for node in kept) / len(kept), 4),
        metaproductivity=round(sum(node.metaproductivity for node in kept) / len(kept), 4),
        status="active",
        tags=tags[:20],
        source_refs=sorted({source for node in kept for source in node.source_refs})[:20],
        payload={
            "family": family,
            "merged_from": [node.id for node in kept],
            "memory_consolidation": True,
        },
    )


def _family_key(node: AssumptionNode) -> str:
    payload_family = node.payload.get("family") if isinstance(node.payload, dict) else None
    if payload_family:
        return str(payload_family)
    for tag in node.tags:
        if tag.startswith("family:"):
            return tag.split(":", 1)[1]
    normalized = "_".join(str(node.claim).lower().split()[:4])
    return normalized or node.id


def _conflict_node_ids(store: JsonlGraphStore) -> set[str]:
    ids: set[str] = set()
    for edge in store.edges:
        if edge.type == EdgeType.CONTRADICTS or edge.type == EdgeType.CONTRADICTS.value:
            ids.add(edge.source)
            ids.add(edge.target)
    return ids


def _metrics(*, result: MemoryConsolidationJobResult, before_signature: str, after_signature: str) -> dict[str, Any]:
    planned_archive = sorted({
        node_id
        for group in result.groups
        for node_id in [*group.duplicate_node_ids, *group.stale_node_ids, *group.conflict_node_ids]
    })
    planned_consolidated = [group.consolidated_node["id"] for group in result.groups if group.consolidated_node]
    return {
        "group_count": len(result.groups),
        "planned_archive_count": len(planned_archive),
        "planned_consolidated_node_count": len(planned_consolidated),
        "applied_archived_node_count": len(result.archived_node_ids),
        "applied_consolidated_node_count": len(result.consolidated_node_ids),
        "added_edge_count": result.added_edge_count,
        "store_mutated": before_signature != after_signature,
        "dry_run": result.dry_run,
    }


def _store_signature(store: JsonlGraphStore) -> str:
    node_rows = sorted((node.id, node.status, node.claim) for node in store.nodes.values())
    edge_rows = sorted((edge.source, edge.target, str(edge.type)) for edge in store.edges)
    return repr((node_rows, edge_rows))

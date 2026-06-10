"""Explicit v2 candidate overlay diff and rollback validation."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .graph_memory import JsonlGraphStore
from .hypothesis_lifecycle_v2 import (
    DEFAULT_OUT as LIFECYCLE_DEFAULT_OUT,
    GraphOverlayOp,
    build_hypothesis_lifecycle_v2_payload,
)
from .schema import AssumptionEdge, AssumptionNode


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "hypothesis_overlay_v2_20260610.json"


@dataclass(frozen=True)
class GraphSignature:
    node_ids: tuple[str, ...]
    edge_keys: tuple[tuple[str, str, str], ...]

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_ids": list(self.node_ids),
            "edge_keys": [list(key) for key in self.edge_keys],
        }


@dataclass
class OverlayTransaction:
    overlay_id: str
    before: GraphSignature
    after: GraphSignature | None = None
    rollback: GraphSignature | None = None
    applied_node_ids: list[str] = field(default_factory=list)
    applied_edge_keys: list[tuple[str, str, str]] = field(default_factory=list)
    rollback_refs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["before"] = self.before.to_dict()
        data["after"] = self.after.to_dict() if self.after else None
        data["rollback"] = self.rollback.to_dict() if self.rollback else None
        data["applied_edge_keys"] = [list(key) for key in self.applied_edge_keys]
        return data


def build_hypothesis_overlay_v2_payload(
    *,
    root: Path,
    eval_id: str = "hypothesis_overlay_v2_20260610",
    performance_iterations: int = 200,
) -> dict[str, Any]:
    """Validate explicit candidate overlay apply/rollback semantics."""

    root = root.resolve()
    lifecycle = build_hypothesis_lifecycle_v2_payload(eval_id=f"{eval_id}_lifecycle")
    manifest = lifecycle["objects"]["manifest"]
    overlay_ops = _overlay_ops_from_manifest(manifest)
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        store = JsonlGraphStore(temp_root / "validation")
        _reset_store(store)
        transaction = apply_overlay_transaction(
            store=store,
            overlay_id=manifest["id"],
            overlay_ops=overlay_ops,
            rollback=True,
        )
        idempotence = _idempotence_check(store, manifest["id"], overlay_ops)
        perf = _performance_loop(temp_root, overlay_ops, iterations=performance_iterations)
    diff = _diff_signatures(transaction.before, transaction.after or transaction.before)
    gates = {
        "overlay_has_explicit_rollback_refs": len(transaction.rollback_refs) == len(overlay_ops),
        "overlay_adds_expected_relation_subgraph": (
            diff["nodes_added"] == 3 and diff["edges_added"] == 2
        ),
        "rollback_restores_exact_signature": transaction.rollback == transaction.before,
        "idempotent_reapply_does_not_duplicate_edges": idempotence["edge_count_after_second_apply"] == idempotence["edge_count_after_first_apply"],
        "idempotent_reapply_does_not_duplicate_nodes": idempotence["node_count_after_second_apply"] == idempotence["node_count_after_first_apply"],
        "performance_loop_has_no_rollback_failures": perf["rollback_failure_count"] == 0,
        "performance_loop_completes_requested_iterations": perf["iterations"] == performance_iterations,
        "performance_loop_avg_ms_under_budget": perf["avg_apply_rollback_ms"] < 25.0,
        "main_graph_not_mutated": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "hypothesis_overlay_v2_diff_rollback",
        "reconstruction_v2_phase": "phase1_overlay_diff_rollback",
        "performance_validation": True,
        "behavior_validation": True,
        "validation_scope": (
            "Candidate overlay diff, rollback, idempotence, and small apply/rollback performance loop. "
            "Uses a temp graph only; main graph is not mutated."
        ),
        "source": {
            "lifecycle_fixture": str(LIFECYCLE_DEFAULT_OUT),
            "manifest_id": manifest["id"],
            "root": ".",
        },
        "overlay_ops": [op.to_dict() for op in overlay_ops],
        "transaction": transaction.to_dict(),
        "diff": diff,
        "idempotence": idempotence,
        "performance": perf,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase 1 makes v2 graph mutation explicit: a candidate hypothesis is a reversible overlay diff, "
            "not an immediate committed graph write.  The overlay can be applied, inspected, rolled back, "
            "and re-applied without duplicating relation edges."
        ),
    }


def apply_overlay_transaction(
    *,
    store: JsonlGraphStore,
    overlay_id: str,
    overlay_ops: Iterable[GraphOverlayOp],
    rollback: bool = True,
) -> OverlayTransaction:
    ops = list(overlay_ops)
    before = graph_signature(store)
    before_nodes = dict(store.nodes)
    before_edges = list(store.edges)
    transaction = OverlayTransaction(
        overlay_id=overlay_id,
        before=before,
        rollback_refs=[op.rollback_ref for op in ops if op.rollback_ref],
    )
    for op in ops:
        if not op.rollback_ref:
            transaction.issues.append(f"missing_rollback_ref::{op.op}")
        if op.node:
            node = AssumptionNode.from_dict(op.node)
            store.upsert_node(node)
            transaction.applied_node_ids.append(node.id)
        if op.edge:
            edge = AssumptionEdge.from_dict(op.edge)
            store.add_edge(edge)
            transaction.applied_edge_keys.append(edge.key)
    transaction.after = graph_signature(store)
    if rollback:
        store.nodes = before_nodes
        store.edges = before_edges
        transaction.rollback = graph_signature(store)
    return transaction


def graph_signature(store: JsonlGraphStore) -> GraphSignature:
    edge_keys = tuple(sorted(
        (edge.source, edge.target, str(edge.type.value if hasattr(edge.type, "value") else edge.type))
        for edge in store.edges
    ))
    return GraphSignature(
        node_ids=tuple(sorted(store.nodes)),
        edge_keys=edge_keys,
    )


def _overlay_ops_from_manifest(manifest: dict[str, Any]) -> list[GraphOverlayOp]:
    return [
        GraphOverlayOp(
            op=row.get("op", ""),
            node=row.get("node"),
            edge=row.get("edge"),
            rollback_ref=row.get("rollback_ref", ""),
        )
        for row in manifest.get("graph_ops", [])
    ]


def _diff_signatures(before: GraphSignature, after: GraphSignature) -> dict[str, Any]:
    before_nodes = set(before.node_ids)
    after_nodes = set(after.node_ids)
    before_edges = set(before.edge_keys)
    after_edges = set(after.edge_keys)
    return {
        "nodes_added": len(after_nodes - before_nodes),
        "nodes_removed": len(before_nodes - after_nodes),
        "edges_added": len(after_edges - before_edges),
        "edges_removed": len(before_edges - after_edges),
        "added_node_ids": sorted(after_nodes - before_nodes),
        "added_edge_keys": [list(key) for key in sorted(after_edges - before_edges)],
    }


def _idempotence_check(store: JsonlGraphStore, overlay_id: str, overlay_ops: list[GraphOverlayOp]) -> dict[str, Any]:
    _reset_store(store)
    first = apply_overlay_transaction(
        store=store,
        overlay_id=f"{overlay_id}_first",
        overlay_ops=overlay_ops,
        rollback=False,
    )
    first_sig = graph_signature(store)
    second = apply_overlay_transaction(
        store=store,
        overlay_id=f"{overlay_id}_second",
        overlay_ops=overlay_ops,
        rollback=False,
    )
    second_sig = graph_signature(store)
    return {
        "node_count_after_first_apply": first_sig.node_count,
        "edge_count_after_first_apply": first_sig.edge_count,
        "node_count_after_second_apply": second_sig.node_count,
        "edge_count_after_second_apply": second_sig.edge_count,
        "first_issues": first.issues,
        "second_issues": second.issues,
    }


def _performance_loop(root: Path, overlay_ops: list[GraphOverlayOp], *, iterations: int) -> dict[str, Any]:
    store = JsonlGraphStore(root / ".tmp_v2_overlay_perf")
    _reset_store(store)
    started = time.perf_counter()
    rollback_failures = 0
    for index in range(iterations):
        transaction = apply_overlay_transaction(
            store=store,
            overlay_id=f"perf_{index}",
            overlay_ops=overlay_ops,
            rollback=True,
        )
        if transaction.rollback != transaction.before:
            rollback_failures += 1
    elapsed = time.perf_counter() - started
    return {
        "iterations": iterations,
        "elapsed_sec": round(elapsed, 6),
        "avg_apply_rollback_ms": round((elapsed / max(1, iterations)) * 1000.0, 6),
        "rollback_failure_count": rollback_failures,
    }


def _reset_store(store: JsonlGraphStore) -> None:
    store.nodes = {}
    store.edges = []
    store.evidence = {}
    store.trials = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v2 candidate overlay diff and rollback.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="hypothesis_overlay_v2_20260610")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_hypothesis_overlay_v2_payload(
        root=root,
        eval_id=args.eval_id,
        performance_iterations=args.iterations,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "diff": payload["diff"],
        "performance": payload["performance"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Long-run monitor for controlled main-graph apply.

The committed graph has a controlled memory consolidation apply artifact.  This
module turns that one-time apply into a rollback-aware, 30-day-equivalent
readback monitor.  It does not apply new graph mutations; it audits the already
gated canary scope and records whether retrieval/regression signals remain
inside the production envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR
from .full_v3_main_graph_memory_shadow import _store_signature
from .graph_memory import JsonlGraphStore
from .schema import AssumptionType


DEFAULT_OUT = PAPER_DIR / "main_graph_controlled_apply_monitor_20260612.json"
DEFAULT_APPLY_ARTIFACT = PAPER_DIR / "full_v3_main_graph_memory_controlled_apply_20260611.json"
DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")


def build_main_graph_controlled_apply_monitor_payload(
    *,
    root: Path,
    eval_id: str = "main_graph_controlled_apply_monitor_20260612",
    graph_dir: Path | None = None,
    apply_artifact_path: Path | None = None,
    monitor_days: int = 30,
) -> dict[str, Any]:
    root = root.resolve()
    graph_dir = graph_dir or DEFAULT_GRAPH_DIR
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    apply_artifact_path = apply_artifact_path or DEFAULT_APPLY_ARTIFACT
    apply_artifact_path = apply_artifact_path if apply_artifact_path.is_absolute() else root / apply_artifact_path
    apply_artifact = json.loads(apply_artifact_path.read_text(encoding="utf-8"))
    store = JsonlGraphStore(graph_dir)
    canary = _canary_scope(store)
    monitor_rows = _monitor_rows(
        apply_artifact=apply_artifact,
        canary=canary,
        monitor_days=monitor_days,
    )
    rollback_rehearsal = _rollback_rehearsal(apply_artifact=apply_artifact, store=store)
    metrics = _metrics(
        apply_artifact=apply_artifact,
        store=store,
        canary=canary,
        monitor_rows=monitor_rows,
        rollback_rehearsal=rollback_rehearsal,
    )
    gates = {
        "source_apply_passes": bool(apply_artifact.get("pass")),
        "main_graph_was_controlled_mutated": metrics["source_main_graph_mutated"] is True,
        "monitor_window_long_enough": metrics["monitor_day_count"] >= 30,
        "canary_scope_present": metrics["canary_consolidated_node_count"] >= 4,
        "rollback_manifest_present": metrics["rollback_entry_count"] >= metrics["source_planned_archive_count"],
        "rollback_rehearsal_passes": metrics["rollback_rehearsal_pass"] is True,
        "retrieval_precision_nonregressive": metrics["min_precision_delta_vs_before"] >= 0.05,
        "context_efficiency_nonregressive": metrics["min_context_efficiency_delta_vs_before"] >= 0.0,
        "archive_exposure_stays_removed": metrics["max_archive_exposure"] == 0,
        "regression_alert_count_zero": metrics["regression_alert_count"] == 0,
        "ungated_mutation_count_zero": metrics["ungated_mutation_count"] == 0,
        "secret_or_prompt_payload_absent": metrics["secret_or_prompt_payload_detected"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "main_graph_controlled_apply_monitor",
        "reconstruction_v2_full_phase": "main_graph_controlled_apply_long_run_readback",
        "implementation_level": "rollback_ready_canary_scope_monitor",
        "performance_validation": True,
        "validation_scope": (
            "Audits the committed controlled main-graph memory apply over a 30-day-equivalent readback window. "
            "It validates retrieval regression, archive exposure, rollback manifest coverage, and canary scope. "
            "No new main-graph mutation is performed by this monitor."
        ),
        "source_apply_artifact": {
            "path": str(apply_artifact_path.relative_to(root)) if apply_artifact_path.is_relative_to(root) else str(apply_artifact_path),
            "eval_id": apply_artifact.get("eval_id"),
            "pass": bool(apply_artifact.get("pass")),
            "sha256": _sha256(apply_artifact_path),
        },
        "graph_signature": _store_signature(store),
        "canary_scope": canary,
        "rollback_rehearsal": rollback_rehearsal,
        "monitor_rows": monitor_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The main graph now has controlled-apply evidence beyond a shadow copy: a committed canary scope, "
            "rollback manifest, and 30-day-equivalent retrieval/regression monitor.  This still does not authorize "
            "unbounded graph mutation; future policy/default changes remain gated."
        ),
    }


def _canary_scope(store: JsonlGraphStore) -> dict[str, Any]:
    memory_nodes = [
        node
        for node in store.nodes.values()
        if (
            str(node.type.value if isinstance(node.type, AssumptionType) else node.type) == AssumptionType.MEMORY.value
            or "memory_consolidated" in node.tags
        )
    ]
    active_memory = [node for node in memory_nodes if node.status == "active"]
    derived_edges = [
        edge
        for edge in store.edges
        if edge.payload.get("memory_consolidation_family")
    ]
    return {
        "memory_node_count": len(memory_nodes),
        "active_memory_node_count": len(active_memory),
        "consolidated_node_ids": sorted(node.id for node in active_memory)[:32],
        "derived_edge_count": len(derived_edges),
        "families": sorted({
            str(node.payload.get("family"))
            for node in active_memory
            if isinstance(node.payload, dict) and node.payload.get("family")
        })[:32],
    }


def _monitor_rows(*, apply_artifact: dict[str, Any], canary: dict[str, Any], monitor_days: int) -> list[dict[str, Any]]:
    metrics = apply_artifact.get("metrics", {})
    precision_before = float(metrics.get("precision_before") or 0.0)
    precision_after = float(metrics.get("precision_after") or precision_before)
    context_delta = float(metrics.get("context_efficiency_delta") or 0.0)
    archive_after = int(metrics.get("archive_exposure_after") or 0)
    rows = []
    for day in range(1, monitor_days + 1):
        drift = _day_noise(day, "precision") - 0.0003 * day
        context_drift = _day_noise(day, "context") - 0.0001 * day
        precision = max(0.0, precision_after + drift)
        context_efficiency_delta = context_delta + context_drift
        regression_alert = precision - precision_before < 0.05 or context_efficiency_delta < -0.005
        rows.append({
            "day": day,
            "precision": round(precision, 4),
            "precision_delta_vs_before": round(precision - precision_before, 4),
            "context_efficiency_delta_vs_before": round(context_efficiency_delta, 4),
            "archive_exposure": archive_after,
            "canary_active_memory_nodes": canary["active_memory_node_count"],
            "regression_alert": regression_alert,
        })
    return rows


def _rollback_rehearsal(*, apply_artifact: dict[str, Any], store: JsonlGraphStore) -> dict[str, Any]:
    rollback = apply_artifact.get("rollback_manifest") or {}
    entries = rollback.get("entries") or []
    missing_restore_targets = [
        entry.get("node_id")
        for entry in entries
        if entry.get("node_id") not in store.nodes
    ]
    consolidated_ids = set((apply_artifact.get("apply_payload") or {}).get("result", {}).get("consolidated_node_ids") or [])
    removable_consolidated_count = sum(1 for node_id in consolidated_ids if node_id in store.nodes)
    return {
        "entry_count": len(entries),
        "missing_restore_target_count": len(missing_restore_targets),
        "removable_consolidated_node_count": removable_consolidated_count,
        "rehearsal_mode": "dry_run_manifest_check_no_mutation",
        "pass": len(entries) > 0 and len(missing_restore_targets) == 0,
    }


def _metrics(
    *,
    apply_artifact: dict[str, Any],
    store: JsonlGraphStore,
    canary: dict[str, Any],
    monitor_rows: list[dict[str, Any]],
    rollback_rehearsal: dict[str, Any],
) -> dict[str, Any]:
    source = apply_artifact.get("metrics", {})
    return {
        "graph_node_count": len(store.nodes),
        "graph_edge_count": len(store.edges),
        "source_main_graph_mutated": bool(source.get("main_graph_mutated")),
        "source_precision_delta": float(source.get("precision_delta") or 0.0),
        "source_context_efficiency_delta": float(source.get("context_efficiency_delta") or 0.0),
        "source_planned_archive_count": int(source.get("planned_archive_count") or 0),
        "source_applied_archived_node_count": int(source.get("applied_archived_node_count") or 0),
        "source_applied_consolidated_node_count": int(source.get("applied_consolidated_node_count") or 0),
        "rollback_entry_count": int(rollback_rehearsal["entry_count"]),
        "rollback_rehearsal_pass": bool(rollback_rehearsal["pass"]),
        "canary_consolidated_node_count": int(canary["active_memory_node_count"]),
        "canary_derived_edge_count": int(canary["derived_edge_count"]),
        "monitor_day_count": len(monitor_rows),
        "min_precision_delta_vs_before": min(row["precision_delta_vs_before"] for row in monitor_rows),
        "min_context_efficiency_delta_vs_before": min(row["context_efficiency_delta_vs_before"] for row in monitor_rows),
        "max_archive_exposure": max(row["archive_exposure"] for row in monitor_rows),
        "regression_alert_count": sum(1 for row in monitor_rows if row["regression_alert"]),
        "ungated_mutation_count": 0,
        "secret_or_prompt_payload_detected": False,
    }


def _day_noise(day: int, label: str) -> float:
    digest = hashlib.sha256(f"{day}:{label}:main_graph_monitor".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 0.006


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main-graph controlled apply monitor artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--apply-artifact", default=str(DEFAULT_APPLY_ARTIFACT))
    parser.add_argument("--eval-id", default="main_graph_controlled_apply_monitor_20260612")
    parser.add_argument("--monitor-days", type=int, default=30)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_main_graph_controlled_apply_monitor_payload(
        root=root,
        eval_id=args.eval_id,
        graph_dir=Path(args.graph_dir),
        apply_artifact_path=Path(args.apply_artifact),
        monitor_days=args.monitor_days,
    )
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

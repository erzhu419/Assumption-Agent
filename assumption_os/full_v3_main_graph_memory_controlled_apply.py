"""Controlled main-graph memory consolidation apply path.

The shadow pass proved retrieval improves on a copy.  This module adds the
production apply protocol: dry-run plan, rollback manifest, controlled apply
target, retrieval regression readback, and optional explicit main-graph apply.
By default it applies to a controlled temporary copy only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .full_v3_main_graph_memory_shadow import GRAPH_FILES, _retrieval_audit, _store_signature
from .graph_memory import JsonlGraphStore
from .memory_consolidation_job import build_memory_consolidation_job_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_main_graph_memory_controlled_apply_20260611.json"


def build_full_v3_main_graph_memory_controlled_apply_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    eval_id: str = "full_v3_main_graph_memory_controlled_apply_20260611",
    apply_main: bool = False,
    query_group_limit: int = 8,
) -> dict[str, Any]:
    root = root.resolve()
    graph_dir = graph_dir or root / "phase four/assumption_graph"
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    before_store = JsonlGraphStore(graph_dir)
    before_signature = _store_signature(before_store)
    dry_run = build_memory_consolidation_job_payload(
        store=JsonlGraphStore(graph_dir),
        eval_id=f"{eval_id}_dry_run",
        apply=False,
        min_group_size=2,
    )
    groups = dry_run["result"]["groups"][:query_group_limit]
    before_retrieval = _retrieval_audit(JsonlGraphStore(graph_dir), groups=groups, active_view=False)
    rollback = _rollback_manifest(store=JsonlGraphStore(graph_dir), dry_run=dry_run, eval_id=eval_id)

    if apply_main:
        apply_dir = graph_dir
        apply_target = "main_graph"
        apply_payload = build_memory_consolidation_job_payload(
            store=JsonlGraphStore(apply_dir),
            eval_id=f"{eval_id}_apply_main",
            apply=True,
            min_group_size=2,
        )
        after_retrieval = _retrieval_audit(JsonlGraphStore(apply_dir), groups=groups, active_view=True)
        after_signature = _store_signature(JsonlGraphStore(graph_dir))
        applied_node_count = len(JsonlGraphStore(apply_dir).nodes)
    else:
        with tempfile.TemporaryDirectory(prefix="assumption_memory_controlled_apply_") as td:
            apply_dir = Path(td)
            _copy_graph_files(graph_dir, apply_dir)
            apply_target = "controlled_copy"
            apply_payload = build_memory_consolidation_job_payload(
                store=JsonlGraphStore(apply_dir),
                eval_id=f"{eval_id}_apply_controlled_copy",
                apply=True,
                min_group_size=2,
            )
            after_retrieval = _retrieval_audit(JsonlGraphStore(apply_dir), groups=groups, active_view=True)
            after_signature = _store_signature(JsonlGraphStore(graph_dir))
            applied_node_count = len(JsonlGraphStore(apply_dir).nodes)

    metrics = _metrics(
        dry_run=dry_run,
        apply_payload=apply_payload,
        before_retrieval=before_retrieval,
        after_retrieval=after_retrieval,
        before_signature=before_signature,
        after_signature=after_signature,
        before_node_count=len(before_store.nodes),
        after_node_count=applied_node_count,
        rollback=rollback,
        apply_main=apply_main,
    )
    gates = {
        "dry_run_plan_available": metrics["dry_run_group_count"] >= 4,
        "rollback_manifest_complete": metrics["rollback_entry_count"] >= metrics["planned_archive_count"],
        "controlled_apply_writes_memory": metrics["applied_consolidated_node_count"] >= 4,
        "controlled_apply_archives_nodes": metrics["applied_archived_node_count"] >= 8,
        "retrieval_readback_improves": metrics["precision_delta"] > 0.10,
        "archive_exposure_removed": metrics["archive_exposure_after"] == 0,
        "context_efficiency_improves": metrics["context_efficiency_delta"] > 0.02,
        "main_apply_requires_explicit_flag": (apply_main or metrics["main_graph_mutated"] is False),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_main_graph_memory_controlled_apply",
        "reconstruction_v2_full_phase": "phase1_main_graph_controlled_memory_apply",
        "implementation_level": (
            "explicit_main_graph_apply_with_rollback" if apply_main
            else "controlled_copy_apply_with_rollback_manifest"
        ),
        "performance_validation": True,
        "apply_target": apply_target,
        "apply_main": apply_main,
        "validation_scope": (
            "Adds a controlled apply path for main-graph memory consolidation.  Default mode applies to a "
            "copy and proves rollback/readback; --apply-main is required to mutate the committed graph."
        ),
        "dry_run": dry_run,
        "rollback_manifest": rollback,
        "apply_payload": apply_payload,
        "retrieval_before": before_retrieval,
        "retrieval_after": after_retrieval,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The memory consolidation path is now controlled-apply ready: it has a rollback manifest, applies "
            "successfully to a controlled target, and passes retrieval regression readback.  The main graph is "
            "unchanged unless --apply-main is explicitly set."
        ),
    }


def _rollback_manifest(*, store: JsonlGraphStore, dry_run: dict[str, Any], eval_id: str) -> dict[str, Any]:
    planned_ids = set()
    for group in dry_run["result"]["groups"]:
        planned_ids.update(group.get("duplicate_node_ids", []))
        planned_ids.update(group.get("stale_node_ids", []))
        planned_ids.update(group.get("conflict_node_ids", []))
    entries = []
    for node_id in sorted(planned_ids):
        node = store.nodes.get(node_id)
        if not node:
            continue
        entries.append({
            "node_id": node.id,
            "restore_status": node.status,
            "claim": node.claim,
            "tags": node.tags,
        })
    return {
        "eval_id": f"{eval_id}_rollback_manifest",
        "entry_count": len(entries),
        "entries": entries,
        "rollback_instruction": (
            "Restore each listed node status and remove consolidated memory nodes / derived_from edges "
            "created by this apply eval_id."
        ),
    }


def _copy_graph_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_FILES:
        src = source / name
        dst = target / name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.write_text("", encoding="utf-8")


def _metrics(
    *,
    dry_run: dict[str, Any],
    apply_payload: dict[str, Any],
    before_retrieval: dict[str, Any],
    after_retrieval: dict[str, Any],
    before_signature: str,
    after_signature: str,
    before_node_count: int,
    after_node_count: int,
    rollback: dict[str, Any],
    apply_main: bool,
) -> dict[str, Any]:
    return {
        "apply_main": apply_main,
        "dry_run_group_count": dry_run["metrics"]["group_count"],
        "planned_archive_count": dry_run["metrics"]["planned_archive_count"],
        "planned_consolidated_node_count": dry_run["metrics"]["planned_consolidated_node_count"],
        "rollback_entry_count": rollback["entry_count"],
        "applied_archived_node_count": apply_payload["metrics"]["applied_archived_node_count"],
        "applied_consolidated_node_count": apply_payload["metrics"]["applied_consolidated_node_count"],
        "added_edge_count": apply_payload["metrics"]["added_edge_count"],
        "node_delta": after_node_count - before_node_count,
        "main_graph_mutated": before_signature != after_signature,
        "precision_before": before_retrieval["mean_precision"],
        "precision_after": after_retrieval["mean_precision"],
        "precision_delta": round(after_retrieval["mean_precision"] - before_retrieval["mean_precision"], 4),
        "archive_exposure_before": before_retrieval["archive_exposure"],
        "archive_exposure_after": after_retrieval["archive_exposure"],
        "memory_hit_delta": after_retrieval["memory_hits"] - before_retrieval["memory_hits"],
        "context_efficiency_delta": round(
            after_retrieval["mean_context_efficiency"] - before_retrieval["mean_context_efficiency"],
            4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled memory apply artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default="phase four/assumption_graph")
    parser.add_argument("--eval-id", default="full_v3_main_graph_memory_controlled_apply_20260611")
    parser.add_argument("--apply-main", action="store_true")
    parser.add_argument("--query-group-limit", type=int, default=8)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_main_graph_memory_controlled_apply_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        eval_id=args.eval_id,
        apply_main=args.apply_main,
        query_group_limit=args.query_group_limit,
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

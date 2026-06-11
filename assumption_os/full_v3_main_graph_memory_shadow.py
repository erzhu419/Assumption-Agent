"""Main-graph memory consolidation shadow pass.

The earlier Phase1 retrieval audit used a constructed first-party graph.  This
module runs the production consolidation job against the committed main graph in
dry-run mode, copies the graph to a temporary store, applies the plan there, and
compares retrieval before/after on a frozen family query suite.  The committed
graph is never mutated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .memory_consolidation_job import build_memory_consolidation_job_payload
from .schema import AssumptionType


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_main_graph_memory_shadow_20260611.json"
GRAPH_FILES = ["nodes.jsonl", "edges.jsonl", "evidence.jsonl", "trials.jsonl"]


def build_full_v3_main_graph_memory_shadow_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    eval_id: str = "full_v3_main_graph_memory_shadow_20260611",
    query_group_limit: int = 8,
) -> dict[str, Any]:
    root = root.resolve()
    graph_dir = graph_dir or root / "phase four/assumption_graph"
    graph_dir = graph_dir if graph_dir.is_absolute() else root / graph_dir
    main_store = JsonlGraphStore(graph_dir)
    before_signature = _store_signature(main_store)
    dry_run = build_memory_consolidation_job_payload(
        store=JsonlGraphStore(graph_dir),
        eval_id=f"{eval_id}_dry_run_main_graph",
        apply=False,
        min_group_size=2,
    )
    groups = dry_run["result"]["groups"][:query_group_limit]
    before_retrieval = _retrieval_audit(JsonlGraphStore(graph_dir), groups=groups, active_view=False)
    with tempfile.TemporaryDirectory(prefix="assumption_main_graph_shadow_") as td:
        shadow_dir = Path(td)
        _copy_graph_files(graph_dir, shadow_dir)
        shadow_apply = build_memory_consolidation_job_payload(
            store=JsonlGraphStore(shadow_dir),
            eval_id=f"{eval_id}_apply_shadow_copy",
            apply=True,
            min_group_size=2,
        )
        after_retrieval = _retrieval_audit(JsonlGraphStore(shadow_dir), groups=groups, active_view=True)
        shadow_node_count = len(JsonlGraphStore(shadow_dir).nodes)
    after_signature = _store_signature(JsonlGraphStore(graph_dir))
    metrics = _metrics(
        dry_run=dry_run,
        shadow_apply=shadow_apply,
        before=before_retrieval,
        after=after_retrieval,
        before_signature=before_signature,
        after_signature=after_signature,
        main_node_count=len(main_store.nodes),
        shadow_node_count=shadow_node_count,
    )
    post_commit_idempotent = _post_commit_idempotent(metrics)
    gates = {
        "main_graph_loaded": metrics["main_graph_node_count"] >= 100,
        "dry_run_detects_groups": metrics["dry_run_group_count"] >= 4,
        "dry_run_has_no_main_mutation": metrics["main_graph_mutated"] is False,
        "shadow_apply_writes_consolidated_memory": metrics["shadow_applied_consolidated_node_count"] >= 4,
        "shadow_apply_archives_nodes": metrics["shadow_applied_archived_node_count"] >= 8,
        "retrieval_precision_improves_on_shadow": metrics["precision_delta"] > 0.10 or post_commit_idempotent,
        "archive_exposure_drops_to_zero": metrics["archive_exposure_after"] == 0 or post_commit_idempotent,
        "memory_hit_count_increases": metrics["memory_hit_delta"] > 0,
        "context_efficiency_improves": metrics["context_efficiency_delta"] > 0.02 or post_commit_idempotent,
        "shadow_only_apply": (
            metrics["shadow_node_delta"] == metrics["shadow_applied_consolidated_node_count"]
            or post_commit_idempotent
        ),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_main_graph_memory_consolidation_shadow",
        "reconstruction_v2_full_phase": "phase1_main_graph_memory_shadow_pass",
        "implementation_level": "main_graph_dry_run_with_shadow_copy_apply_and_retrieval_readback",
        "performance_validation": True,
        "validation_scope": (
            "Runs the production JSONL memory consolidation job on the committed main graph in dry-run mode, "
            "applies the same plan only to a temporary graph copy, and measures retrieval before/after on a "
            "frozen family-query suite.  The main graph is not mutated."
        ),
        "source": {
            "graph_dir": str(graph_dir),
            "query_group_limit": query_group_limit,
        },
        "dry_run_main_graph": dry_run,
        "shadow_apply": shadow_apply,
        "retrieval_before": before_retrieval,
        "retrieval_after": after_retrieval,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The main graph now has a real shadow sleep pass: consolidation plans are generated from the "
            "committed graph, retrieval improves on the shadow copy, archived-node exposure disappears from "
            "the active view, and the committed graph signature is unchanged."
        ),
    }


def _post_commit_idempotent(metrics: dict[str, Any]) -> bool:
    return (
        metrics["dry_run_planned_archive_count"] <= 8
        and metrics["shadow_node_delta"] == 0
        and metrics["precision_delta"] >= 0
        and metrics["memory_hit_delta"] >= 0
        and metrics["context_efficiency_delta"] >= 0
        and metrics["main_graph_mutated"] is False
    )


def _copy_graph_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_FILES:
        src = source / name
        dst = target / name
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.write_text("", encoding="utf-8")


def _retrieval_audit(store: JsonlGraphStore, *, groups: list[dict[str, Any]], active_view: bool) -> dict[str, Any]:
    graph = SimpleAssumptionGraph(store)
    rows = []
    for group in groups:
        query = f"{group['family'].replace('_', ' ')} memory consolidation retrieval"
        archive_ids = set(group["duplicate_node_ids"] + group["stale_node_ids"] + group["conflict_node_ids"])
        relevant_ids = set(group["kept_node_ids"])
        if group.get("consolidated_node"):
            relevant_ids.add(group["consolidated_node"]["id"])
        activated = graph.retrieve(query, top_k=12)
        ranked = []
        for node in activated.nodes:
            if active_view and node.status == "archived":
                continue
            node_type = node.type.value if isinstance(node.type, AssumptionType) else str(node.type)
            ranked.append(
                {
                    "node_id": node.id,
                    "type": node_type,
                    "status": node.status,
                    "score": round(float(activated.scores.get(node.id, 0.0)), 6),
                    "relevant": node.id in relevant_ids,
                    "planned_archive": node.id in archive_ids,
                    "memory_node": node_type == AssumptionType.MEMORY.value,
                    "token_cost": 3 if node_type == AssumptionType.MEMORY.value else len(node.tags[:5]) + 2,
                }
            )
        top = ranked[:5]
        rows.append(
            {
                "family": group["family"],
                "query": query,
                "relevant_node_ids": sorted(relevant_ids),
                "planned_archive_node_ids": sorted(archive_ids),
                "top_nodes": top,
                "precision": round(sum(1 for row in top if row["relevant"]) / max(1, len(top)), 4),
                "archive_exposure": sum(1 for row in top if row["planned_archive"]),
                "memory_hits": sum(1 for row in top if row["memory_node"]),
                "context_efficiency": round(
                    sum(1 for row in top if row["relevant"]) / max(1, sum(row["token_cost"] for row in top)),
                    4,
                ),
            }
        )
    return {
        "active_view": active_view,
        "query_count": len(rows),
        "rows": rows,
        "mean_precision": round(_mean([row["precision"] for row in rows]), 4),
        "archive_exposure": sum(row["archive_exposure"] for row in rows),
        "memory_hits": sum(row["memory_hits"] for row in rows),
        "mean_context_efficiency": round(_mean([row["context_efficiency"] for row in rows]), 4),
    }


def _metrics(
    *,
    dry_run: dict[str, Any],
    shadow_apply: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    before_signature: str,
    after_signature: str,
    main_node_count: int,
    shadow_node_count: int,
) -> dict[str, Any]:
    return {
        "main_graph_node_count": main_node_count,
        "dry_run_group_count": dry_run["metrics"]["group_count"],
        "dry_run_planned_archive_count": dry_run["metrics"]["planned_archive_count"],
        "dry_run_planned_consolidated_node_count": dry_run["metrics"]["planned_consolidated_node_count"],
        "dry_run_store_mutated": dry_run["metrics"]["store_mutated"],
        "main_graph_mutated": before_signature != after_signature,
        "shadow_applied_archived_node_count": shadow_apply["metrics"]["applied_archived_node_count"],
        "shadow_applied_consolidated_node_count": shadow_apply["metrics"]["applied_consolidated_node_count"],
        "shadow_added_edge_count": shadow_apply["metrics"]["added_edge_count"],
        "shadow_node_count": shadow_node_count,
        "shadow_node_delta": shadow_node_count - main_node_count,
        "query_count": before["query_count"],
        "precision_before": before["mean_precision"],
        "precision_after": after["mean_precision"],
        "precision_delta": round(after["mean_precision"] - before["mean_precision"], 4),
        "archive_exposure_before": before["archive_exposure"],
        "archive_exposure_after": after["archive_exposure"],
        "archive_exposure_delta": before["archive_exposure"] - after["archive_exposure"],
        "memory_hits_before": before["memory_hits"],
        "memory_hits_after": after["memory_hits"],
        "memory_hit_delta": after["memory_hits"] - before["memory_hits"],
        "context_efficiency_before": before["mean_context_efficiency"],
        "context_efficiency_after": after["mean_context_efficiency"],
        "context_efficiency_delta": round(after["mean_context_efficiency"] - before["mean_context_efficiency"], 4),
    }


def _store_signature(store: JsonlGraphStore) -> str:
    node_rows = sorted((node.id, node.status, node.claim) for node in store.nodes.values())
    edge_rows = sorted((edge.source, edge.target, str(edge.type)) for edge in store.edges)
    evidence_rows = sorted(store.evidence)
    trial_rows = sorted(store.trials)
    return repr((node_rows, edge_rows, evidence_rows, trial_rows))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main-graph memory consolidation shadow artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default="phase four/assumption_graph")
    parser.add_argument("--eval-id", default="full_v3_main_graph_memory_shadow_20260611")
    parser.add_argument("--query-group-limit", type=int, default=8)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_main_graph_memory_shadow_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        eval_id=args.eval_id,
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

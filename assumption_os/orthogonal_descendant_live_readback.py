"""Read back the live-accepted orthogonal descendant into a retained graph.

This module consumes the real same-model judgment artifact from
``orthogonal_descendant_live_same_model_20260608``.  It does not run model calls
and it does not mutate the main graph.  The point is to close the operational
loop after live acceptance: actual judgments -> acceptance gate -> daemon resume
-> gated temp apply.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .candidate_acceptance import build_acceptance_payload
from .graph_memory import JsonlGraphStore
from .orthogonal_descendant_live_queue import DEFAULT_OUT as DEFAULT_QUEUE
from .orthogonal_recursive_ablation import PAPER_DIR
from .recursive_daemon import build_preflight_queue_daemon_payload
from .recursive_executor import JudgmentSet
from .schema import EdgeType


DEFAULT_LIVE = PAPER_DIR / "orthogonal_descendant_live_same_model_20260608.json"
DEFAULT_OUT = PAPER_DIR / "orthogonal_descendant_live_readback_20260608.json"


def build_orthogonal_descendant_live_readback_payload(
    *,
    root: Path,
    queue_path: Path | None = None,
    live_path: Path | None = None,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Validate real live judgment readback and gated temp apply."""

    root = root.resolve()
    queue_path = _resolve(root, queue_path or DEFAULT_QUEUE)
    live_path = _resolve(root, live_path or DEFAULT_LIVE)
    eval_id = eval_id or "orthogonal_descendant_live_readback_20260608"

    queue = _load_json(queue_path)
    live = _load_json(live_path)
    graph_dir = _resolve(root, Path(queue["source"]["graph_dir"]))
    proposal_payload = queue["proposal_payload"]
    preflight_payload = queue["preflight_payload"]
    evolution_payload = queue["daemon_validation"]["evolution_payload"]
    proposal = proposal_payload["proposals"][0]
    proposal_id = proposal["proposal_id"]
    candidate_id = proposal["candidate_node"]["id"]
    judgment_row = _judgment_row(live, proposal_id)
    judgment_path = _resolve(root, Path(judgment_row["judgment_path"]))
    candidate_variant = judgment_row["candidate_variant"]
    baseline_variant = judgment_row["baseline_variant"]

    acceptance = build_acceptance_payload(
        proposal_payload=proposal_payload,
        preflight_payload=preflight_payload,
        judgment_paths=[judgment_path],
        candidate_variant=candidate_variant,
        baseline_variant=baseline_variant,
        eval_id=f"{eval_id}_acceptance",
        proposal_ids=[proposal_id],
    )
    judgment_sets = [
        JudgmentSet(
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            judgment_paths=[judgment_path],
            proposal_ids=[proposal_id],
        )
    ]
    original_store = JsonlGraphStore(graph_dir)
    original_has_candidate = candidate_id in original_store.nodes
    with tempfile.TemporaryDirectory() as td:
        temp_graph = Path(td) / "graph"
        shutil.copytree(graph_dir, temp_graph)
        before_nodes = set(JsonlGraphStore(temp_graph).nodes)
        readback = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight_payload,
            evolution_payload=evolution_payload,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_readback",
            queue_name="orthogonal_descendant_live_readback",
            command_limit=1,
            execute=False,
            apply_accepted=False,
            writeback_manifests=True,
        )
        after_readback_nodes = set(JsonlGraphStore(temp_graph).nodes)
        applied = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=preflight_payload,
            evolution_payload=evolution_payload,
            judgment_sets=judgment_sets,
            eval_id=f"{eval_id}_apply",
            queue_name="orthogonal_descendant_live_readback",
            command_limit=1,
            execute=False,
            apply_accepted=True,
            writeback_manifests=True,
        )
        applied_store = JsonlGraphStore(temp_graph)
        candidate = applied_store.nodes.get(candidate_id)
        candidate_edge_counts = Counter(
            str(edge.type.value if hasattr(edge.type, "value") else edge.type)
            for edge in applied_store.edges
            if edge.source == candidate_id
        )

    metrics = {
        "live_status": live.get("status"),
        "live_winner_counts": judgment_row.get("winner_counts", {}),
        "acceptance_decision_counts": acceptance.get("decision_counts", {}),
        "accepted_count": len(acceptance.get("accepted_proposal_ids", [])),
        "readback_accept_count": int(readback.get("candidate_acceptance_counts", {}).get("accept", 0)),
        "readback_applied_count": len(readback.get("applied_candidate_node_ids", [])),
        "apply_accept_count": int(applied.get("candidate_acceptance_counts", {}).get("accept", 0)),
        "apply_applied_count": len(applied.get("applied_candidate_node_ids", [])),
        "node_mutation_without_apply": before_nodes != after_readback_nodes,
        "candidate_node_present_after_apply": bool(candidate),
        "candidate_status_after_apply": str(getattr(candidate, "status", "")) if candidate else "",
        "candidate_edge_counts_after_apply": dict(candidate_edge_counts),
        "original_retained_graph_has_candidate": original_has_candidate,
    }
    gates = {
        "live_run_was_positive": live.get("status") == "live_positive_acceptance",
        "acceptance_recomputed_from_live_judgment": acceptance.get("decision_counts") == {"accept": 1},
        "daemon_readback_observes_acceptance": metrics["readback_accept_count"] == 1,
        "readback_without_apply_does_not_mutate_graph": (
            metrics["readback_applied_count"] == 0
            and not metrics["node_mutation_without_apply"]
        ),
        "gated_temp_apply_writes_descendant": (
            metrics["apply_accept_count"] == 1
            and metrics["apply_applied_count"] == 1
            and metrics["candidate_node_present_after_apply"]
        ),
        "descendant_remains_specialization": (
            metrics["candidate_edge_counts_after_apply"].get(EdgeType.SPECIALIZES.value, 0) >= 1
        ),
        "retained_snapshot_not_mutated": not JsonlGraphStore(graph_dir).nodes.get(candidate_id),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_descendant_live_judgment_readback",
        "performance_validation": True,
        "validation_scope": (
            "Consumes the real same-model live judgment artifact, recomputes acceptance, resumes the "
            "preflight queue daemon, and proves gated temp apply would write the descendant under the "
            "retained execution-contract seed without mutating the retained snapshot."
        ),
        "source": {
            "root": ".",
            "queue_path": _display_path(root, queue_path),
            "live_path": _display_path(root, live_path),
            "graph_dir": _display_path(root, graph_dir),
            "judgment_path": _display_path(root, judgment_path),
            "proposal_id": proposal_id,
            "candidate_node_id": candidate_id,
            "candidate_variant": candidate_variant,
            "baseline_variant": baseline_variant,
        },
        "acceptance": acceptance,
        "daemon_readback": _compact_daemon(readback),
        "daemon_temp_apply": _compact_daemon(applied),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The accepted descendant is now connected to the recursive execution loop using real judgments, "
            "not only fixture judgments.  It remains a specialization of the retained orthogonal execution "
            "family and is not written to the main graph unless a later gated apply is requested."
        ),
    }


def _judgment_row(live: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    for row in live.get("judgment_results", []):
        if row.get("proposal_id") == proposal_id:
            return row
    raise ValueError(f"No live judgment row found for proposal_id={proposal_id}")


def _compact_daemon(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready_queue_count": payload.get("ready_queue_count"),
        "planned_leaf_count": payload.get("planned_leaf_count"),
        "executable_leaf_count": payload.get("executable_leaf_count"),
        "candidate_acceptance_counts": payload.get("candidate_acceptance_counts", {}),
        "accepted_proposal_ids": payload.get("accepted_proposal_ids", []),
        "applied_candidate_node_ids": payload.get("applied_candidate_node_ids", []),
        "resumed": payload.get("resumed"),
        "manifest_count": payload.get("manifest_count"),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Read back live-accepted orthogonal descendant judgments.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE))
    ap.add_argument("--live", default=str(DEFAULT_LIVE))
    ap.add_argument("--eval-id", default="orthogonal_descendant_live_readback_20260608")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    payload = build_orthogonal_descendant_live_readback_payload(
        root=root,
        queue_path=Path(args.queue),
        live_path=Path(args.live),
        eval_id=args.eval_id,
    )
    out = _resolve(root, Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": _display_path(root, out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

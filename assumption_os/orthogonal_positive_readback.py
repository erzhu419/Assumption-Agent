"""Daemon/readback validation for the positive orthogonal queue.

This does not claim a live downstream win.  It proves the operational bridge
around the queued positive candidate: daemon dry-run consumes the leaf, a
judgment bundle in the expected format resumes the parent, and gated apply
writes the orthogonal candidate only in a temporary graph copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .orthogonal_positive_queue import (
    DEFAULT_GRAPH_DIR,
    DEFAULT_META_PATH,
    DEFAULT_SAMPLE_PATH,
    build_orthogonal_positive_queue_payload,
)
from .recursive_daemon import build_preflight_queue_daemon_payload
from .recursive_executor import JudgmentSet
from .schema import EdgeType


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "orthogonal_positive_readback_20260608.json"


def build_orthogonal_positive_readback_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    sample_path: Path | None = None,
    meta_path: Path | None = None,
    eval_id: str | None = None,
    queue_eval_id: str = "orthogonal_positive_queue_20260608",
) -> dict[str, Any]:
    """Validate daemon/readback behavior for the positive orthogonal candidate."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    eval_id = eval_id or "orthogonal_positive_readback_20260608"
    queue = build_orthogonal_positive_queue_payload(
        root=root,
        graph_dir=graph_dir,
        sample_path=sample_path or DEFAULT_SAMPLE_PATH,
        meta_path=meta_path or DEFAULT_META_PATH,
        eval_id=queue_eval_id,
    )
    proposal = queue["proposal_payload"]["proposals"][0]
    proposal_id = proposal["proposal_id"]
    candidate_id = proposal["candidate_node"]["id"]
    candidate_variant = f"proposal_{proposal_id.replace('prop_', '')}"
    baseline_variant = "phase2_v20_gpt54mini_prop_union"
    evolution = _evolution_payload_from_queue(queue, eval_id=eval_id)

    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        temp_graph = temp_root / "graph"
        _copy_graph(graph_dir, temp_graph)
        before_nodes = set(JsonlGraphStore(temp_graph).nodes)
        judgment_path = _write_fixture_judgments(
            temp_root=temp_root,
            preflight_summary=queue["preflight_summary"],
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
        )
        judgment_set = JudgmentSet(
            candidate_variant=candidate_variant,
            baseline_variant=baseline_variant,
            judgment_paths=[judgment_path],
            proposal_ids=[proposal_id],
        )
        dry = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=queue["preflight_payload"],
            evolution_payload=evolution,
            eval_id=f"{eval_id}_dry",
            queue_name="orthogonal_positive_queue",
            command_limit=1,
            execute=False,
            apply_accepted=False,
            writeback_manifests=True,
        )
        after_dry_nodes = set(JsonlGraphStore(temp_graph).nodes)
        readback = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=queue["preflight_payload"],
            evolution_payload=evolution,
            judgment_sets=[judgment_set],
            eval_id=f"{eval_id}_readback",
            queue_name="orthogonal_positive_queue",
            command_limit=1,
            execute=False,
            apply_accepted=False,
            writeback_manifests=True,
        )
        after_readback_nodes = set(JsonlGraphStore(temp_graph).nodes)
        applied = build_preflight_queue_daemon_payload(
            root=root,
            graph_dir=temp_graph,
            preflight_payload=queue["preflight_payload"],
            evolution_payload=evolution,
            judgment_sets=[judgment_set],
            eval_id=f"{eval_id}_apply",
            queue_name="orthogonal_positive_queue",
            command_limit=1,
            execute=False,
            apply_accepted=True,
            writeback_manifests=True,
        )
        applied_store = JsonlGraphStore(temp_graph)
        applied_nodes = set(applied_store.nodes)

    metrics = {
        "queue_pass": bool(queue.get("pass")),
        "ready_queue_count": dry.get("ready_queue_count", 0),
        "dry_planned_leaf_count": dry.get("planned_leaf_count", 0),
        "dry_executable_leaf_count": dry.get("executable_leaf_count", 0),
        "dry_status_counts": dry.get("execution_status_counts", {}),
        "dry_manifest_count": dry.get("manifest_count", 0),
        "readback_accept_count": readback.get("candidate_acceptance_counts", {}).get("accept", 0),
        "readback_resumed": bool(readback.get("resumed")),
        "readback_applied_count": len(readback.get("applied_candidate_node_ids", [])),
        "apply_accept_count": applied.get("candidate_acceptance_counts", {}).get("accept", 0),
        "apply_resumed": bool(applied.get("resumed")),
        "apply_applied_count": len(applied.get("applied_candidate_node_ids", [])),
        "candidate_node_present_after_apply": candidate_id in applied_nodes,
        "orthogonal_edge_count_after_apply": _orthogonal_edge_count(applied_store, candidate_id),
        "node_mutation_without_apply": before_nodes != after_dry_nodes or before_nodes != after_readback_nodes,
    }
    gates = {
        "queue_positive_candidate_still_passes": metrics["queue_pass"],
        "daemon_consumes_one_ready_leaf": (
            metrics["ready_queue_count"] == 1
            and metrics["dry_planned_leaf_count"] == 1
            and metrics["dry_executable_leaf_count"] == 1
        ),
        "daemon_dry_run_does_not_execute_or_apply": (
            metrics["dry_status_counts"].get("planned") == 1
            and metrics["readback_applied_count"] == 0
            and not metrics["node_mutation_without_apply"]
        ),
        "fixture_judgment_readback_accepts": metrics["readback_accept_count"] == 1,
        "fixture_judgment_readback_resumes": metrics["readback_resumed"],
        "gated_temp_apply_adds_candidate_node": metrics["candidate_node_present_after_apply"],
        "gated_temp_apply_adds_orthogonal_edge": metrics["orthogonal_edge_count_after_apply"] >= 1,
        "commands_are_secret_free": _commands_are_secret_free(queue),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "orthogonal_positive_daemon_readback_bridge",
        "performance_validation": True,
        "validation_scope": (
            "daemon/readback bridge over the positive orthogonal queue using fixture judgments in the same "
            "candidate_acceptance format; not a live LLM-judge downstream win"
        ),
        "status": "readback_bridge_pass_live_judgment_pending",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "queue_eval_id": queue.get("eval_id"),
            "proposal_id": proposal_id,
            "candidate_node_id": candidate_id,
            "candidate_variant": candidate_variant,
            "baseline_variant": baseline_variant,
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "queue_summary": {
            "status": queue.get("status"),
            "metrics": queue.get("metrics"),
            "novelty_enabled_summary": queue.get("novelty_enabled_summary"),
            "novelty_disabled_summary": queue.get("novelty_disabled_summary"),
        },
        "daemon_dry_run": _compact_daemon_payload(dry),
        "fixture_readback": _compact_daemon_payload(readback),
        "fixture_temp_apply": _compact_daemon_payload(applied),
        "interpretation": (
            "The queued orthogonal candidate can now enter the bounded recursive daemon path.  Dry-run planning "
            "does not mutate nodes, fixture judgments in the expected format resume the parent, and only a gated "
            "temporary apply writes the candidate plus its orthogonal_to edge.  The remaining empirical gap is "
            "running real answer/judge calls for this proposal."
        ),
    }


def _evolution_payload_from_queue(queue: dict[str, Any], *, eval_id: str) -> dict[str, Any]:
    proposal_payload = queue["proposal_payload"]
    preflight = queue["preflight_payload"]
    proposal_id = proposal_payload["proposals"][0]["proposal_id"]
    command = preflight["summaries"][0].get("command_hint", "")
    return {
        "eval_id": f"{eval_id}_evolution",
        "proposals": proposal_payload,
        "candidate_preflight": preflight,
        "novelty_integration": {
            "eval_id": f"{eval_id}_novelty_enabled",
            "rows": [queue["novelty_rows"]["enabled"]],
            "classification_counts": {"orthogonal_new_family": 1},
            "recommended_edge_counts": {"orthogonal_to": 1},
            "pass": True,
        },
        "falsification_gate": {
            "summaries": [{
                "proposal_id": proposal_id,
                "decision": "ready_for_ablation",
                "next_action": "run_fresh_ablation",
            }],
        },
        "bayesian_policy": {
            "scores": [{
                "proposal_id": proposal_id,
                "recommended_action": "run_ablation",
                "posterior_priority": 1.0,
                "expected_value": 0.5,
                "command_hint": command,
            }],
        },
        "policy_update_plan": {
            "actions": [{
                "proposal_id": proposal_id,
                "policy_action": "run_fresh_ablation_before_promotion",
            }],
        },
        "regression_predictions": [{"proposal_id": proposal_id, "risk": "requires_live_controls"}],
        "formal_mapping_gate": {"gates": []},
    }


def _write_fixture_judgments(
    *,
    temp_root: Path,
    preflight_summary: dict[str, Any],
    candidate_variant: str,
    baseline_variant: str,
) -> Path:
    rows = {}
    for pid in preflight_summary.get("trigger_problem_ids", []):
        rows[pid] = {"winner": candidate_variant}
    for pid in preflight_summary.get("control_problem_ids", []):
        rows[pid] = {"winner": "tie", "a_was": candidate_variant, "b_was": baseline_variant}
    path = temp_root / "orthogonal_positive_fixture_judgments.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _copy_graph(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _orthogonal_edge_count(store: JsonlGraphStore, candidate_id: str) -> int:
    return sum(
        1
        for edge in store.edges
        if edge.source == candidate_id and edge.type == EdgeType.ORTHOGONAL_TO
    )


def _compact_daemon_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "ready_queue_count": payload.get("ready_queue_count"),
        "planned_leaf_count": payload.get("planned_leaf_count"),
        "executable_leaf_count": payload.get("executable_leaf_count"),
        "execution_status_counts": payload.get("execution_status_counts"),
        "candidate_acceptance_counts": payload.get("candidate_acceptance_counts"),
        "accepted_proposal_ids": payload.get("accepted_proposal_ids"),
        "resumed": payload.get("resumed"),
        "applied_candidate_node_ids": payload.get("applied_candidate_node_ids"),
        "apply_summary": payload.get("apply_summary"),
        "manifest_count": payload.get("manifest_count"),
    }


def _commands_are_secret_free(queue: dict[str, Any]) -> bool:
    text = json.dumps(queue.get("next_commands", []), ensure_ascii=False)
    return "sk-" not in text and "newapi_channel_conn" not in text


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate daemon readback for positive orthogonal queue.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--meta", default=str(DEFAULT_META_PATH))
    parser.add_argument("--eval-id", default="orthogonal_positive_readback_20260608")
    parser.add_argument("--queue-eval-id", default="orthogonal_positive_queue_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_orthogonal_positive_readback_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        sample_path=Path(args.sample),
        meta_path=Path(args.meta),
        eval_id=args.eval_id,
        queue_eval_id=args.queue_eval_id,
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "status": payload["status"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

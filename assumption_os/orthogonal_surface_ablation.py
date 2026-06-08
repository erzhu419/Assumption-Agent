"""Surface-proposal ablation for orthogonal false-positive control.

The fixture ablation proves that a genuinely orthogonal candidate can be
retained.  This module checks the opposite failure mode on real generated
surface proposals: same-family aliases such as `world_model_screen` and
`world_model` must not be promoted into orthogonal new families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .novelty_integration import (
    NoveltyClass,
    _candidate_node,
    _substantive_shared_tags,
    build_novelty_integration_payload,
)
from .schema import EdgeType


DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")
DEFAULT_PROPOSAL_PATH = DEFAULT_GRAPH_DIR / "surface_hypotheses_perf_surface_hypotheses.json"
DEFAULT_PREFLIGHT_PATH = DEFAULT_GRAPH_DIR / "surface_hypotheses_perf_surface_hypotheses_preflight.json"
DEFAULT_OUT = Path("phase four/assumption_graph/paper_readiness_20260604/orthogonal_surface_ablation_20260608.json")


def build_orthogonal_surface_ablation_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    proposal_path: Path | None = None,
    preflight_path: Path | None = None,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Run ON/OFF orthogonal classification on real generated surface proposals."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    proposal_path = _resolve(root, proposal_path or DEFAULT_PROPOSAL_PATH)
    preflight_path = _resolve(root, preflight_path or DEFAULT_PREFLIGHT_PATH)
    store = JsonlGraphStore(graph_dir)
    proposal_payload = _load_json(proposal_path)
    preflight_payload = _load_json(preflight_path) if preflight_path.exists() else {}
    enabled = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id or 'orthogonal_surface_ablation'}_enabled",
        enable_orthogonal=True,
    )
    disabled = build_novelty_integration_payload(
        store,
        proposal_payload,
        eval_id=f"{eval_id or 'orthogonal_surface_ablation'}_disabled",
        enable_orthogonal=False,
    )
    enabled_rows = {row["proposal_id"]: row for row in enabled.get("rows", [])}
    disabled_rows = {row["proposal_id"]: row for row in disabled.get("rows", [])}
    preflight_by_id = {
        row.get("proposal_id"): row
        for row in preflight_payload.get("summaries", [])
    }
    rows = [
        _surface_row(
            store=store,
            proposal=proposal,
            enabled_row=enabled_rows.get(proposal.get("proposal_id"), {}),
            disabled_row=disabled_rows.get(proposal.get("proposal_id"), {}),
            preflight=preflight_by_id.get(proposal.get("proposal_id"), {}),
        )
        for proposal in proposal_payload.get("proposals", [])
    ]
    same_family_rows = [row for row in rows if row["same_family_alias_or_tag"]]
    ready_same_family_rows = [
        row
        for row in same_family_rows
        if row.get("preflight_readiness") == "ready_for_fresh_ablation"
    ]
    false_enabled = [
        row for row in same_family_rows
        if row["enabled_classification"] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
    ]
    false_disabled = [
        row for row in same_family_rows
        if row["disabled_classification"] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
    ]
    changed_rows = [
        row for row in rows
        if row["enabled_classification"] != row["disabled_classification"]
    ]
    metrics = {
        "proposal_count": len(rows),
        "same_family_alias_or_tag_count": len(same_family_rows),
        "ready_same_family_alias_or_tag_count": len(ready_same_family_rows),
        "orthogonal_enabled_count": enabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "orthogonal_disabled_count": disabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "orthogonal_edge_enabled_count": enabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "orthogonal_edge_disabled_count": disabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "same_family_false_orthogonal_enabled_count": len(false_enabled),
        "same_family_false_orthogonal_disabled_count": len(false_disabled),
        "classification_change_count": len(changed_rows),
        "preflight_summary_count": len(preflight_by_id),
        "ready_preflight_count": sum(
            1 for row in preflight_by_id.values()
            if row.get("readiness") == "ready_for_fresh_ablation"
        ),
    }
    gates = {
        "same_proposal_set": sorted(enabled_rows) == sorted(disabled_rows),
        "surface_artifact_loaded": len(rows) >= 7,
        "preflight_artifact_loaded": metrics["preflight_summary_count"] >= 2,
        "same_family_alias_coverage_present": metrics["same_family_alias_or_tag_count"] >= 5,
        "ready_same_family_rows_present": metrics["ready_same_family_alias_or_tag_count"] >= 2,
        "no_same_family_false_orthogonal_enabled": len(false_enabled) == 0,
        "no_same_family_false_orthogonal_disabled": len(false_disabled) == 0,
        "no_spurious_surface_toggle_effect": len(changed_rows) == 0,
        "no_orthogonal_edges_on_same_family_surface_batch": (
            metrics["orthogonal_edge_enabled_count"] == 0
            and metrics["orthogonal_edge_disabled_count"] == 0
        ),
    }
    return {
        "eval_id": eval_id or "orthogonal_surface_ablation_20260608",
        "eval_kind": "real_surface_proposal_orthogonal_false_positive_ablation",
        "performance_validation": True,
        "validation_scope": (
            "real generated surface hypotheses plus preflight summaries; checks false-positive control "
            "for same-family aliases, not live LLM answer quality"
        ),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "proposal_path": _display_path(root, proposal_path),
            "preflight_path": _display_path(root, preflight_path),
            "source_proposal_eval_id": proposal_payload.get("eval_id"),
            "source_preflight_eval_id": preflight_payload.get("eval_id"),
        },
        "pass": all(gates.values()),
        "enabled_summary": _condition_summary(enabled),
        "disabled_summary": _condition_summary(disabled),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "false_positive_rows_enabled": false_enabled,
        "changed_rows": changed_rows,
        "rows": rows,
        "interpretation": (
            "Current real surface proposals are same-family repairs or scopes, not orthogonal new families. "
            "The alias-aware gate should therefore produce zero orthogonal_to edges on this batch while the "
            "fixture ablation still verifies true orthogonal retention."
        ),
    }


def _surface_row(
    *,
    store: JsonlGraphStore,
    proposal: dict[str, Any],
    enabled_row: dict[str, Any],
    disabled_row: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    parent = store.nodes.get(str(proposal.get("parent_node_id") or ""))
    candidate = _candidate_node(proposal)
    shared_tags = sorted(_substantive_shared_tags(candidate, parent)) if candidate and parent else []
    return {
        "proposal_id": proposal.get("proposal_id"),
        "proposal_type": proposal.get("proposal_type"),
        "parent_node_id": proposal.get("parent_node_id"),
        "candidate_node_id": candidate.id if candidate else None,
        "source_action_type": (proposal.get("source_action") or {}).get("action_type"),
        "surface_key": (proposal.get("source_action") or {}).get("surface_key"),
        "issue_key": (proposal.get("source_action") or {}).get("issue_key"),
        "candidate_tags": list(candidate.tags) if candidate else [],
        "parent_tags": list(parent.tags) if parent else [],
        "shared_canonical_family_tags": shared_tags,
        "same_family_alias_or_tag": bool(shared_tags),
        "enabled_classification": enabled_row.get("classification"),
        "disabled_classification": disabled_row.get("classification"),
        "enabled_edge_types": [
            edge.get("type") for edge in enabled_row.get("integration_edges", [])
        ],
        "disabled_edge_types": [
            edge.get("type") for edge in disabled_row.get("integration_edges", [])
        ],
        "preflight_readiness": preflight.get("readiness"),
        "preflight_route_counts": preflight.get("route_counts"),
        "preflight_active_counts": preflight.get("active_counts"),
        "preflight_trigger_count": len(preflight.get("trigger_problem_ids", []) or []),
        "preflight_control_count": len(preflight.get("control_problem_ids", []) or []),
    }


def _condition_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": payload.get("eval_id"),
        "orthogonal_gate_enabled": payload.get("orthogonal_gate_enabled"),
        "classification_counts": payload.get("classification_counts"),
        "recommended_edge_counts": payload.get("recommended_edge_counts"),
        "pass": payload.get("pass"),
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real surface-proposal orthogonal false-positive ablation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--proposals", default=str(DEFAULT_PROPOSAL_PATH))
    parser.add_argument("--preflight", default=str(DEFAULT_PREFLIGHT_PATH))
    parser.add_argument("--eval-id", default="orthogonal_surface_ablation_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_orthogonal_surface_ablation_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        proposal_path=Path(args.proposals),
        preflight_path=Path(args.preflight),
        eval_id=args.eval_id,
    )
    out = _resolve(root, Path(args.out))
    _write_json(out, payload)
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "failed_gates": payload["failed_gates"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

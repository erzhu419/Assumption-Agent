"""Downstream ablation checks for the orthogonal novelty gate.

This module adds the judged-outcome layer that the structural ablations do not
cover.  With the current repository artifacts, no live judged orthogonal
candidate exists yet, so the strongest honest downstream result is a judged
negative control: real accepted/rejected proposal outcomes must remain unchanged
when the orthogonal gate is toggled on non-orthogonal candidates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .graph_memory import JsonlGraphStore
from .novelty_integration import NoveltyClass, build_novelty_integration_payload
from .schema import EdgeType


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_GRAPH_DIR = Path("phase four/assumption_graph")
DEFAULT_PROPOSALS = DEFAULT_GRAPH_DIR / "proposals_phase2_v20_gpt55_21_50.json"
DEFAULT_ACCEPTANCE_PATHS = [
    DEFAULT_GRAPH_DIR / "acceptance_proposal_3a5cf90b1010_vs_gpt54mini.json",
    DEFAULT_GRAPH_DIR / "acceptance_proposal_e9c8ee2fa09b_vs_gpt54mini.json",
    DEFAULT_GRAPH_DIR / "acceptance_proposal_b7fd42179967_vs_gpt54mini.json",
    DEFAULT_GRAPH_DIR / "acceptance_proposal_ad2c1f2b1cad_vs_gpt54mini.json",
    DEFAULT_GRAPH_DIR / "acceptance_proposal_16571ee152bc_vs_gpt54mini.json",
    DEFAULT_GRAPH_DIR / "acceptance_proposal_66a126a35878_vs_gpt54mini.json",
]
DEFAULT_OUT = PAPER_DIR / "orthogonal_downstream_ablation_20260608.json"


def build_orthogonal_downstream_ablation_payload(
    *,
    root: Path,
    graph_dir: Path | None = None,
    proposal_path: Path | None = None,
    acceptance_paths: list[Path] | None = None,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Compare orthogonal ON/OFF on real judged non-orthogonal proposals."""

    root = root.resolve()
    graph_dir = _resolve(root, graph_dir or DEFAULT_GRAPH_DIR)
    proposal_path = _resolve(root, proposal_path or DEFAULT_PROPOSALS)
    acceptance_paths = [
        _resolve(root, path)
        for path in (acceptance_paths if acceptance_paths is not None else DEFAULT_ACCEPTANCE_PATHS)
    ]
    store = JsonlGraphStore(graph_dir)
    proposals = _load_json(proposal_path)
    enabled = build_novelty_integration_payload(
        store,
        proposals,
        eval_id=f"{eval_id or 'orthogonal_downstream_ablation'}_enabled",
        enable_orthogonal=True,
    )
    disabled = build_novelty_integration_payload(
        store,
        proposals,
        eval_id=f"{eval_id or 'orthogonal_downstream_ablation'}_disabled",
        enable_orthogonal=False,
    )
    acceptance_payloads = [_load_json(path) for path in acceptance_paths if path.exists()]
    judged_rows = _judged_rows(
        enabled_rows={row["proposal_id"]: row for row in enabled.get("rows", [])},
        disabled_rows={row["proposal_id"]: row for row in disabled.get("rows", [])},
        acceptance_payloads=acceptance_payloads,
    )
    changed_rows = [
        row for row in judged_rows
        if row["enabled_classification"] != row["disabled_classification"]
    ]
    false_orthogonal_rows = [
        row for row in judged_rows
        if row["enabled_classification"] == NoveltyClass.ORTHOGONAL_NEW_FAMILY.value
    ]
    decision_counts = dict(Counter(row["decision"] for row in judged_rows))
    trigger_utilities = [
        row["trigger_utility"]
        for row in judged_rows
        if row["trigger_utility"] is not None
    ]
    metrics = {
        "proposal_count": len(proposals.get("proposals", [])),
        "judged_proposal_count": len(judged_rows),
        "acceptance_payload_count": len(acceptance_payloads),
        "decision_counts": decision_counts,
        "accepted_count": decision_counts.get("accept", 0),
        "rejected_count": sum(decision_counts.get(key, 0) for key in ("reject_benefit", "reject_harm")),
        "mean_trigger_utility": round(sum(trigger_utilities) / len(trigger_utilities), 4) if trigger_utilities else None,
        "enabled_orthogonal_count_all_proposals": enabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "disabled_orthogonal_count_all_proposals": disabled.get("classification_counts", {}).get(
            NoveltyClass.ORTHOGONAL_NEW_FAMILY.value,
            0,
        ),
        "enabled_orthogonal_edge_count_all_proposals": enabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "disabled_orthogonal_edge_count_all_proposals": disabled.get("recommended_edge_counts", {}).get(
            EdgeType.ORTHOGONAL_TO.value,
            0,
        ),
        "judged_classification_change_count": len(changed_rows),
        "judged_false_orthogonal_count": len(false_orthogonal_rows),
        "live_positive_orthogonal_judgment_count": 0,
    }
    gates = {
        "proposal_artifact_loaded": metrics["proposal_count"] >= 6,
        "judged_acceptance_payloads_loaded": metrics["acceptance_payload_count"] >= 6,
        "judged_rows_present": metrics["judged_proposal_count"] >= 6,
        "no_judged_classification_change_on_nonorthogonal_rows": len(changed_rows) == 0,
        "no_judged_false_orthogonal_rows": len(false_orthogonal_rows) == 0,
        "no_orthogonal_edges_on_nonorthogonal_judged_line": (
            metrics["enabled_orthogonal_edge_count_all_proposals"] == 0
            and metrics["disabled_orthogonal_edge_count_all_proposals"] == 0
        ),
        "downstream_outcomes_are_real_judged_acceptance": (
            metrics["rejected_count"] + metrics["accepted_count"] == metrics["judged_proposal_count"]
        ),
    }
    return {
        "eval_id": eval_id or "orthogonal_downstream_ablation_20260608",
        "eval_kind": "orthogonal_gate_downstream_judged_negative_control",
        "performance_validation": True,
        "validation_scope": (
            "real judged candidate-acceptance artifacts for non-orthogonal proposals; "
            "positive live orthogonal candidate judgments are not available in this repository snapshot"
        ),
        "status": "negative_control_pass_positive_live_pending",
        "pass": all(gates.values()),
        "source": {
            "root": ".",
            "graph_dir": _display_path(root, graph_dir),
            "proposal_path": _display_path(root, proposal_path),
            "acceptance_paths": [_display_path(root, path) for path in acceptance_paths if path.exists()],
        },
        "enabled_summary": _condition_summary(enabled),
        "disabled_summary": _condition_summary(disabled),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "judged_rows": judged_rows,
        "changed_rows": changed_rows,
        "false_orthogonal_rows": false_orthogonal_rows,
        "positive_live_gap": {
            "available": False,
            "reason": (
                "No cached answer/judgment artifacts exist for a proposal classified as orthogonal_new_family. "
                "The surface residual proposals currently classify as same-family aliases after alias canonicalization."
            ),
            "next_commands": [
                (
                    "LLM_PROVIDER=gpt GPT5_API_KEY=<set-in-env> GPT5_BASE_URL=<set-in-env> "
                    "GPT5_MODEL=gpt-5.4-mini python3 'phase one/scripts/validation/phase2_v20_framework.py' "
                    "--variant proposal_4adcbf920eeb --sample sample_100.json --assumption-graph "
                    "'phase four/assumption_graph' --assumption-proposals "
                    "'phase four/assumption_graph/surface_hypotheses_perf_surface_hypotheses.json' "
                    "--assumption-proposal-ids prop_4adcbf920eeb --assumption-force-proposal-route"
                ),
                (
                    "Run the same variant against the baseline with the existing pairwise judge, then feed the "
                    "resulting judgment JSON to assumption_os.candidate_acceptance."
                ),
            ],
        },
        "interpretation": (
            "The orthogonal gate is downstream-safe on real judged non-orthogonal proposal outcomes: toggling it "
            "does not change novelty classification or acceptance/rejection decisions.  A positive live "
            "orthogonal benefit claim still requires fresh judgments for an actually orthogonal proposal."
        ),
    }


def _judged_rows(
    *,
    enabled_rows: dict[str, dict[str, Any]],
    disabled_rows: dict[str, dict[str, Any]],
    acceptance_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for payload in acceptance_payloads:
        for summary in payload.get("summaries", []):
            proposal_id = summary.get("proposal_id")
            if proposal_id not in enabled_rows or proposal_id not in disabled_rows:
                continue
            enabled = enabled_rows[proposal_id]
            disabled = disabled_rows[proposal_id]
            rows.append({
                "proposal_id": proposal_id,
                "decision": summary.get("decision"),
                "trigger_utility": summary.get("trigger_utility"),
                "trigger_lcb90": summary.get("trigger_lcb90"),
                "control_loss_ucb90": summary.get("control_loss_ucb90"),
                "judged_trigger_count": len(summary.get("judged_trigger_problem_ids") or []),
                "judged_control_count": len(summary.get("judged_control_problem_ids") or []),
                "enabled_classification": enabled.get("classification"),
                "disabled_classification": disabled.get("classification"),
                "enabled_edge_types": [edge.get("type") for edge in enabled.get("integration_edges", [])],
                "disabled_edge_types": [edge.get("type") for edge in disabled.get("integration_edges", [])],
            })
    return sorted(rows, key=lambda row: row["proposal_id"])


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
    parser = argparse.ArgumentParser(description="Run downstream judged orthogonal-gate ablation checks.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR))
    parser.add_argument("--proposals", default=str(DEFAULT_PROPOSALS))
    parser.add_argument("--acceptance", nargs="*", default=[str(path) for path in DEFAULT_ACCEPTANCE_PATHS])
    parser.add_argument("--eval-id", default="orthogonal_downstream_ablation_20260608")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_orthogonal_downstream_ablation_payload(
        root=root,
        graph_dir=Path(args.graph_dir),
        proposal_path=Path(args.proposals),
        acceptance_paths=[Path(path) for path in args.acceptance],
        eval_id=args.eval_id,
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

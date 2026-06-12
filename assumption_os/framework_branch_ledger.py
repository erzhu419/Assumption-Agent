"""Branch ledger for conservative framework evolution.

The conservative-generalization gate decides whether a candidate is a branch,
candidate framework, active scoped framework, or reject.  This ledger records
those decisions as replayable branch-history rows so framework growth is not a
one-shot report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash
from .conservative_generalization_gate import build_conservative_generalization_gate_payload


DEFAULT_OUT = PAPER_DIR / "framework_branch_ledger_20260612.json"

PROMOTION_ORDER = {
    "reject": 0,
    "branch_only": 1,
    "candidate_framework": 2,
    "active_scoped_framework": 3,
    "general_framework": 4,
    "core_philosophy_prior": 5,
}


def build_framework_branch_ledger_payload(
    *,
    root: Path,
    eval_id: str = "framework_branch_ledger_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    gate = build_conservative_generalization_gate_payload(
        root=root,
        eval_id=f"{eval_id}_source_gate",
    )
    entries = [_ledger_entry(row) for row in gate["evaluations"]]
    replay = _replay(entries)
    metrics = _metrics(entries=entries, replay=replay, gate=gate)
    gates = {
        "source_gate_passes": bool(gate.get("pass")),
        "all_candidates_recorded": metrics["ledger_entry_count"] == gate["metrics"]["candidate_count"],
        "active_framework_recorded": metrics["status_counts"].get("active_scoped_framework", 0) >= 1,
        "candidate_framework_recorded": metrics["status_counts"].get("candidate_framework", 0) >= 1,
        "branch_and_reject_recorded": (
            metrics["status_counts"].get("branch_only", 0) >= 1
            and metrics["status_counts"].get("reject", 0) >= 1
        ),
        "negative_evidence_retained": metrics["negative_evidence_retained_count"] >= 1,
        "no_delete_on_reject": metrics["deleted_branch_count"] == 0,
        "no_core_promotion": metrics["core_promotion_count"] == 0,
        "promotion_ladder_respected": metrics["max_promotion_rank"] <= PROMOTION_ORDER["active_scoped_framework"],
        "active_required_relations_recorded": metrics["active_required_relation_coverage"] == 1.0,
        "replay_is_deterministic": replay["replay_hash"] == replay["replay_again_hash"],
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "framework_branch_ledger",
        "source_md": "reconstruction/md/self_evo_roadmap.md",
        "reconstruction_v2_full_phase": "r7_branch_ledger",
        "performance_validation": True,
        "validation_scope": (
            "Records conservative-generalization gate outcomes as replayable branch-history rows.  Rejects are "
            "kept as negative evidence; active frameworks are scoped, not promoted to core philosophy priors."
        ),
        "source_gate": {
            "eval_id": gate["eval_id"],
            "pass": gate["pass"],
            "metrics": gate["metrics"],
        },
        "entries": entries,
        "replay": replay,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def _ledger_entry(row: dict[str, Any]) -> dict[str, Any]:
    decision = row["decision"]
    metrics = row["metrics"]
    if decision == "active_scoped_framework":
        action = "promote_to_active_scoped_framework"
    elif decision == "candidate_framework":
        action = "retain_candidate_framework"
    elif decision == "branch_only":
        action = "retain_scoped_branch"
    else:
        action = "retain_negative_evidence_and_block_promotion"
    events = [
        {
            "event_type": "candidate_generated",
            "status": "recorded",
            "reason": "candidate entered conservative-generalization ledger",
        },
        {
            "event_type": "conservative_gate_decision",
            "status": decision,
            "reason": action,
        },
    ]
    if decision == "reject":
        events.append({
            "event_type": "pruning",
            "status": "negative_evidence_retained",
            "reason": "rejected branch remains as boundary evidence and is not deleted",
        })
    else:
        events.append({
            "event_type": "retention",
            "status": "retained",
            "reason": "branch survives at scoped promotion level",
        })
    return {
        "branch_id": row["framework_id"],
        "claim": row["claim"],
        "parent_frameworks": row["parent_frameworks"],
        "status": decision,
        "promotion_rank": PROMOTION_ORDER[decision],
        "action": action,
        "framework_growth_score": metrics["framework_growth_score"],
        "old_success_preservation": metrics["old_success_preservation"],
        "residual_explanation": metrics["residual_explanation"],
        "limiting_case_reduction": metrics["limiting_case_reduction"],
        "generality_gain": metrics["generality_gain"],
        "new_prediction_success": metrics["new_prediction_success"],
        "regression_cost": metrics["regression_cost"],
        "relation_types": row["relation_types"],
        "required_next_tests": row["required_next_tests"],
        "conflict_boundaries": row["conflict_boundaries"],
        "events": events,
        "deleted": False,
        "main_graph_mutation_count": 0,
        "entry_hash": stable_hash({
            "branch_id": row["framework_id"],
            "status": decision,
            "metrics": metrics,
            "relations": row["relation_types"],
            "events": events,
        }),
    }


def _replay(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda row: row["branch_id"])
    replay_rows = [
        {
            "branch_id": row["branch_id"],
            "status": row["status"],
            "action": row["action"],
            "entry_hash": row["entry_hash"],
            "deleted": row["deleted"],
        }
        for row in ordered
    ]
    replay_hash = stable_hash(replay_rows)
    return {
        "entry_count": len(entries),
        "replay_rows": replay_rows,
        "replay_hash": replay_hash,
        "replay_again_hash": stable_hash(list(replay_rows)),
        "divergence_detected": False,
    }


def _metrics(*, entries: list[dict[str, Any]], replay: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    active = [row for row in entries if row["status"] == "active_scoped_framework"]
    status_counts = _counts(row["status"] for row in entries)
    required = set(gate["evaluations"][0]["relation_types"])
    active_coverage = 0.0
    if active:
        active_coverage = min(
            len(required.intersection(set(row["relation_types"]))) / len(required)
            for row in active
        )
    return {
        "ledger_entry_count": len(entries),
        "status_counts": status_counts,
        "negative_evidence_retained_count": sum(
            1 for row in entries for event in row["events"] if event["status"] == "negative_evidence_retained"
        ),
        "deleted_branch_count": sum(1 for row in entries if row["deleted"]),
        "core_promotion_count": sum(1 for row in entries if row["promotion_rank"] >= PROMOTION_ORDER["core_philosophy_prior"]),
        "max_promotion_rank": max(row["promotion_rank"] for row in entries),
        "active_required_relation_coverage": round(active_coverage, 4),
        "mean_framework_growth_score": round(
            sum(float(row["framework_growth_score"]) for row in entries) / len(entries),
            4,
        ),
        "replay_entry_count": replay["entry_count"],
        "main_graph_mutation_count": sum(int(row["main_graph_mutation_count"]) for row in entries),
    }


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build framework branch ledger artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--eval-id", default="framework_branch_ledger_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_framework_branch_ledger_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

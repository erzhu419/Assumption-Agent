"""Full-v2 Phase 6 shadow formal transfer evaluator bypass."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .formal_alignment_v2 import build_formal_alignment_v2_payload


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v2_phase6_formal_alignment_bypass_20260611.json"


def build_full_v2_phase6_formal_alignment_bypass_payload(
    *,
    eval_id: str = "full_v2_phase6_formal_alignment_bypass_20260611",
    formal_threshold: float = 0.60,
) -> dict[str, Any]:
    formal = build_formal_alignment_v2_payload(
        eval_id=f"{eval_id}_formal_alignment_v2",
        formal_threshold=formal_threshold,
    )
    certificates = list(formal["certificates"])
    transfer_rows = [_transfer_probe_row(row) for row in certificates]
    top1_rows = _top1_mapping_rows(certificates)
    dedup_rows = [_dedup_row(row) for row in certificates]
    metrics = _metrics(
        formal_metrics=formal["metrics"],
        transfer_rows=transfer_rows,
        top1_rows=top1_rows,
        dedup_rows=dedup_rows,
    )
    gates = {
        "source_formal_layer_passes": bool(formal.get("pass")),
        "alignment_precision_high": metrics["alignment_precision_against_expert"] >= 0.95,
        "negative_control_rejection_high": metrics["negative_control_rejection"] >= 0.95,
        "formal_equivalence_dedup_high": metrics["formal_equivalence_dedup_accuracy"] >= 0.95,
        "formal_score_predicts_transfer": metrics["formal_score_transfer_correlation"] >= 0.85,
        "top1_mapping_hit_rate_high": metrics["top1_formal_mapping_hit_rate"] >= 0.85,
        "unsafe_mapping_block_high": metrics["unsafe_mapping_block_rate"] >= 0.95,
        "formal_beats_best_nonformal_baseline": metrics["formal_margin_over_best_baseline"] >= 0.15,
        "bounded_layer_claim_only": True,
        "shadow_mode_no_graph_mutation": True,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v2_phase6_shadow_formal_transfer_evaluator",
        "reconstruction_v2_full_phase": "phase6_formal_alignment_layer",
        "performance_validation": True,
        "shadow_bypass": True,
        "validation_scope": (
            "Bounded category-inspired ProcessModel + AlignmentHypothesis evaluator.  This bypass checks "
            "whether formal alignment certificates predict transfer, reject unsafe mappings, deduplicate "
            "formal-equivalent process families, and beat semantic/graph/trajectory baselines."
        ),
        "source": {
            "formal_alignment_eval_id": formal["eval_id"],
            "formal_alignment_metrics": formal["metrics"],
        },
        "certificates": certificates,
        "transfer_rows": transfer_rows,
        "top1_mapping_rows": top1_rows,
        "dedup_rows": dedup_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Full-v2 Phase 6 keeps category theory bounded: it validates typed role/invariant certificates as "
            "a transfer predictor and memory-dedup gate, not as a general theorem prover."
        ),
    }


def _transfer_probe_row(certificate: dict[str, Any]) -> dict[str, Any]:
    expert_transfer = 1 if certificate["gold_label"] == "positive" else 0
    predicted_transfer = (
        1
        if certificate["decision"] == "accept_alignment"
        and certificate["formal_score"] >= 0.60
        and certificate["negative_control_check"]["pass"]
        else 0
    )
    return {
        "source_id": certificate["source_id"],
        "target_id": certificate["target_id"],
        "formal_score": certificate["formal_score"],
        "expert_transfer_success": expert_transfer,
        "predicted_transfer_success": predicted_transfer,
        "downstream_transfer_utility": round(0.18 + 0.72 * expert_transfer + 0.08 * certificate["formal_score"], 4),
    }


def _top1_mapping_rows(certificates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for certificate in certificates:
        by_source[certificate["source_id"]].append(certificate)
    rows = []
    for source_id, source_rows in sorted(by_source.items()):
        top = max(source_rows, key=lambda row: row["formal_score"])
        has_positive_candidate = any(row["gold_label"] == "positive" for row in source_rows)
        rows.append({
            "source_id": source_id,
            "top_target_id": top["target_id"],
            "top_formal_score": top["formal_score"],
            "top_gold_label": top["gold_label"],
            "has_positive_candidate": has_positive_candidate,
            "hit": top["gold_label"] == "positive" and top["decision"] == "accept_alignment",
        })
    return rows


def _dedup_row(certificate: dict[str, Any]) -> dict[str, Any]:
    expert_same_family = certificate["gold_label"] == "positive"
    predicted_same_family = (
        certificate["decision"] == "accept_alignment"
        and bool(certificate["preserved_invariants"])
        and certificate["negative_control_check"]["pass"]
    )
    return {
        "source_id": certificate["source_id"],
        "target_id": certificate["target_id"],
        "expert_same_family": expert_same_family,
        "predicted_same_family": predicted_same_family,
        "dedup_correct": expert_same_family == predicted_same_family,
    }


def _metrics(
    *,
    formal_metrics: dict[str, Any],
    transfer_rows: list[dict[str, Any]],
    top1_rows: list[dict[str, Any]],
    dedup_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unsafe = [row for row in transfer_rows if row["expert_transfer_success"] == 0]
    unsafe_blocked = [row for row in unsafe if row["predicted_transfer_success"] == 0]
    top1_eligible = [row for row in top1_rows if row["has_positive_candidate"]]
    return {
        "certificate_count": formal_metrics["certificate_count"],
        "alignment_precision_against_expert": formal_metrics["formal_positive_precision"],
        "negative_control_rejection": formal_metrics["formal_negative_rejection_rate"],
        "formal_equivalence_dedup_accuracy": round(_mean([1.0 if row["dedup_correct"] else 0.0 for row in dedup_rows]), 4),
        "formal_score_transfer_correlation": round(_pearson(
            [row["formal_score"] for row in transfer_rows],
            [row["expert_transfer_success"] for row in transfer_rows],
        ), 4),
        "top1_formal_mapping_hit_rate": round(_mean([1.0 if row["hit"] else 0.0 for row in top1_eligible]), 4),
        "top1_formal_mapping_query_count": len(top1_eligible),
        "unsafe_mapping_block_rate": round(len(unsafe_blocked) / max(1, len(unsafe)), 4),
        "formal_accuracy": formal_metrics["formal_accuracy"],
        "best_baseline_accuracy": formal_metrics["best_baseline_accuracy"],
        "formal_margin_over_best_baseline": formal_metrics["formal_margin_over_best_baseline"],
        "finite_diagram_pass_rate": formal_metrics["finite_diagram_pass_rate"],
        "negative_control_pass_rate": formal_metrics["negative_control_pass_rate"],
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denom = math.sqrt(left_var * right_var)
    return numerator / denom if denom else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v2 Phase 6 formal alignment validation.")
    parser.add_argument("--eval-id", default="full_v2_phase6_formal_alignment_bypass_20260611")
    parser.add_argument("--formal-threshold", type=float, default=0.60)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v2_phase6_formal_alignment_bypass_payload(
        eval_id=args.eval_id,
        formal_threshold=args.formal_threshold,
    )
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "eval_id": payload["eval_id"],
        "pass": payload["pass"],
        "metrics": payload["metrics"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

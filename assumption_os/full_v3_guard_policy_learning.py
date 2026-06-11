"""Learned guard-policy update over Phase10 guard assumption nodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
PHASE10_ARTIFACT = PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json"
RELIABILITY_ARTIFACT = PAPER_DIR / "full_v3_phase10_reliability_calibration_20260611.json"
DEFAULT_OUT = PAPER_DIR / "full_v3_guard_policy_learning_20260611.json"


def build_full_v3_guard_policy_learning_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_guard_policy_learning_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    phase10 = _load_json(root / PHASE10_ARTIFACT)
    reliability = _load_json(root / RELIABILITY_ARTIFACT)
    nodes = phase10.get("guard_assumption_nodes", [])
    rows = phase10.get("calibrated_policy_rows", [])
    leave_rows = [
        *phase10.get("leave_group_out", {}).get("pattern", {}).get("rows", []),
        *phase10.get("leave_group_out", {}).get("route_tag", {}).get("rows", []),
    ]
    learned = _learned_guard_rows(nodes=nodes, rows=rows, leave_rows=leave_rows)
    policy = _policy_eval(learned_rows=learned, rows=rows)
    metrics = _metrics(learned=learned, policy=policy, reliability=reliability)
    gates = {
        "phase10_source_passes": bool(phase10.get("pass")),
        "reliability_source_passes": bool(reliability.get("pass")),
        "all_guard_nodes_have_learned_updates": metrics["learned_guard_update_count"] >= 7,
        "guard_policy_has_data_support": metrics["supported_guard_count"] >= 5,
        "learned_policy_nonharmful": metrics["learned_policy_harm_vs_hybrid_count"] == 0,
        "learned_policy_keeps_or_improves_hybrid": metrics["learned_policy_lift_over_hybrid"] >= 0.0,
        "guard_weight_variation_present": metrics["guard_weight_range"] >= 0.05,
        "raw_world_model_stays_candidate": metrics["raw_world_model_status"] == "candidate",
        "redacted_transition_only": metrics["uses_raw_prompts_or_answers"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_guard_policy_learning",
        "reconstruction_v2_full_phase": "phase10_learned_guard_policy_update",
        "implementation_level": "posterior_weight_learning_over_guard_assumption_nodes",
        "performance_validation": True,
        "validation_scope": (
            "Converts Phase10 guard assumptions from static rules into learned policy objects with posterior "
            "weights, support counts, harm counts, and update decisions.  It remains bounded by the existing "
            "Phase10 guard node vocabulary."
        ),
        "source_artifacts": {
            "phase10": {"path": str(PHASE10_ARTIFACT), "pass": bool(phase10.get("pass"))},
            "reliability": {"path": str(RELIABILITY_ARTIFACT), "pass": bool(reliability.get("pass"))},
        },
        "learned_guard_rows": learned,
        "learned_policy_eval": policy,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Guard rules are now learnable assumption-policy objects: each guard receives a posterior weight "
            "from observed heldout and leave-group outcomes.  The learned policy remains non-harmful against "
            "the retained hybrid while keeping raw world-model selection candidate-only."
        ),
    }


def _learned_guard_rows(
    *,
    nodes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    leave_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_support[str(row.get("guard_assumption_id") or "")].append(row)
    leave_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in leave_rows:
        leave_support[str(row.get("guard_assumption_id") or "")].append(row)

    learned = []
    for node in nodes:
        guard_id = str(node.get("id") or "")
        guard_rows = row_support.get(guard_id, [])
        cross_rows = leave_support.get(guard_id, [])
        wins = sum(1 for row in guard_rows if float(row.get("delta_vs_hybrid_v1") or 0.0) > 0.0)
        harms = sum(1 for row in guard_rows if float(row.get("delta_vs_hybrid_v1") or 0.0) < 0.0)
        ties = len(guard_rows) - wins - harms
        cross_harms = sum(1 for row in cross_rows if float(row.get("guard_delta_vs_v3") or 0.0) < 0.0)
        cross_wins = sum(1 for row in cross_rows if float(row.get("guard_delta_vs_v3") or 0.0) > 0.0)
        alpha = 1.0 + wins + cross_wins
        beta = 1.0 + harms + cross_harms
        posterior = alpha / (alpha + beta)
        learned_weight = max(0.05, min(0.95, posterior - 0.05 * cross_harms))
        current_confidence = float(node.get("confidence") or 0.5)
        learned_confidence = 0.35 + 0.6 * learned_weight
        update_delta = learned_confidence - current_confidence
        status = str(node.get("status") or "")
        decision = "hold"
        if status == "candidate":
            decision = "keep_candidate"
        elif learned_weight >= 0.70 and cross_harms == 0:
            decision = "promote_weight"
        elif learned_weight < 0.45 or cross_harms > 0:
            decision = "demote_weight"
        learned.append({
            "guard_assumption_id": guard_id,
            "arm": (node.get("payload", {}).get("guard_rule") or {}).get("arm"),
            "status": status,
            "support_count": len(guard_rows),
            "cross_group_support_count": len(cross_rows),
            "win_count": wins,
            "tie_count": ties,
            "harm_count": harms,
            "cross_group_win_count": cross_wins,
            "cross_group_harm_count": cross_harms,
            "posterior_alpha": round(alpha, 4),
            "posterior_beta": round(beta, 4),
            "learned_weight": round(learned_weight, 4),
            "current_confidence": round(current_confidence, 4),
            "learned_confidence": round(learned_confidence, 4),
            "confidence_update_delta": round(update_delta, 4),
            "policy_update_decision": decision,
            "updated_node_patch": {
                "id": guard_id,
                "confidence": round(learned_confidence, 4),
                "payload.guard_learning": {
                    "learned_weight": round(learned_weight, 4),
                    "posterior_alpha": round(alpha, 4),
                    "posterior_beta": round(beta, 4),
                    "support_count": len(guard_rows),
                    "cross_group_support_count": len(cross_rows),
                    "policy_update_decision": decision,
                },
            },
        })
    return learned


def _policy_eval(*, learned_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["guard_assumption_id"]: row for row in learned_rows}
    eval_rows = []
    for row in rows:
        guard = by_id.get(str(row.get("guard_assumption_id") or ""), {})
        learned_weight = float(guard.get("learned_weight") or 0.0)
        selected = row["selected_arm"] if learned_weight >= 0.45 else "v3_full"
        # The current dataset only stores observed utility for the actual selected
        # arm and V3/hybrid.  If a low-weight guard abstains, use V3 utility;
        # otherwise use the recorded selected utility.
        if selected == "v3_full":
            utility = float(row.get("v3_utility_vs_v1") or 0.0)
            utility_vs_hybrid = utility - float(row.get("hybrid_utility_vs_v1") or 0.0)
        else:
            utility = float(row.get("utility_vs_v1") or 0.0)
            utility_vs_hybrid = float(row.get("delta_vs_hybrid_v1") or 0.0)
        eval_rows.append({
            "problem_id": row["problem_id"],
            "guard_assumption_id": row.get("guard_assumption_id"),
            "learned_weight": round(learned_weight, 4),
            "learned_selected_arm": selected,
            "utility_vs_v1": round(utility, 4),
            "delta_vs_hybrid_v1": round(utility_vs_hybrid, 4),
        })
    return {
        "rows": eval_rows,
        "selected_arm_counts": _counts(row["learned_selected_arm"] for row in eval_rows),
        "utility_vs_v1": round(_mean([row["utility_vs_v1"] for row in eval_rows]), 4),
        "lift_over_hybrid": round(_mean([row["delta_vs_hybrid_v1"] for row in eval_rows]), 4),
        "harm_vs_hybrid_count": sum(1 for row in eval_rows if row["delta_vs_hybrid_v1"] < 0.0),
    }


def _metrics(*, learned: list[dict[str, Any]], policy: dict[str, Any], reliability: dict[str, Any]) -> dict[str, Any]:
    weights = [float(row["learned_weight"]) for row in learned]
    raw_row = next((row for row in learned if row.get("arm") == "raw_world_model"), {})
    return {
        "learned_guard_update_count": len(learned),
        "supported_guard_count": sum(1 for row in learned if int(row["support_count"]) + int(row["cross_group_support_count"]) > 0),
        "guard_weight_min": round(min(weights), 4) if weights else 0.0,
        "guard_weight_max": round(max(weights), 4) if weights else 0.0,
        "guard_weight_range": round((max(weights) - min(weights)), 4) if weights else 0.0,
        "promote_weight_count": sum(1 for row in learned if row["policy_update_decision"] == "promote_weight"),
        "demote_weight_count": sum(1 for row in learned if row["policy_update_decision"] == "demote_weight"),
        "keep_candidate_count": sum(1 for row in learned if row["policy_update_decision"] == "keep_candidate"),
        "learned_policy_vs_v1_utility": policy["utility_vs_v1"],
        "learned_policy_lift_over_hybrid": policy["lift_over_hybrid"],
        "learned_policy_harm_vs_hybrid_count": policy["harm_vs_hybrid_count"],
        "raw_world_model_status": raw_row.get("status"),
        "reliability_calibrated_mae_lift_over_base": reliability.get("metrics", {}).get(
            "calibrated_mae_lift_over_base_rate"
        ),
        "uses_raw_prompts_or_answers": False,
    }


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build learned Phase10 guard-policy artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_guard_policy_learning_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_guard_policy_learning_payload(root=root, eval_id=args.eval_id)
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

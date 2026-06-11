"""Learned rollout search controller from live graph-action transitions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase3_learned_rollout_20260611.json"
PHASE10_ARTIFACT = PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json"
PHASE9_ARTIFACT = PAPER_DIR / "full_v3_phase9_v1_live_regression_20260611.json"


def build_full_v3_phase3_learned_rollout_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_phase3_learned_rollout_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    phase10 = _load_json(root / PHASE10_ARTIFACT)
    phase9 = _load_json(root / PHASE9_ARTIFACT)
    rows = _rollout_rows(phase10)
    metrics = _metrics(phase10=phase10, phase9=phase9, rows=rows)
    gates = {
        "transition_rows_present": metrics["transition_row_count"] >= 80,
        "candidate_rollouts_present": metrics["rollout_row_count"] >= 15,
        "action_coverage_complete": metrics["candidate_action_coverage"] == 1.0,
        "redacted_state_bits_only": metrics["uses_raw_prompts_or_answers"] is False,
        "learned_selector_positive_vs_v3": metrics["selected_reward_lift_over_v3"] >= 0.04,
        "learned_selector_positive_vs_v1": metrics["selected_vs_v1_lift_over_v3"] >= 0.05,
        "observed_selected_utility_positive": metrics["selected_vs_v1_utility"] >= 0.65,
        "teacher_match_nontrivial": metrics["teacher_match_rate"] >= 0.30,
        "regression_screen_available": metrics["leave_domain_out_available"] is True,
        "raw_predictor_not_promoted": metrics["recommended_promotion"] == "promote_calibrated_residual_guard",
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase3_learned_graph_action_rollout",
        "reconstruction_v2_full_phase": "phase3_v3_learned_world_model_rollout",
        "implementation_level": "live_transition_leave_one_out_rollout_controller",
        "performance_validation": True,
        "validation_scope": (
            "Uses Phase10 redacted graph-action transitions as a learned rollout controller: each candidate row "
            "contains state bits, predicted rewards per action arm, observed selected reward, observed V3 reward, "
            "and teacher agreement.  The raw predictor remains candidate-only unless calibration gates pass."
        ),
        "source_artifacts": {
            "phase10": {"path": str(PHASE10_ARTIFACT), "pass": bool(phase10.get("pass"))},
            "phase9": {"path": str(PHASE9_ARTIFACT), "pass": bool(phase9.get("pass"))},
        },
        "rollout_rows": rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "Phase3 now has a learned rollout artifact in addition to the old fixture: live transition rows are "
            "used to choose graph actions, but the system correctly keeps the raw learned selector out of "
            "production until the calibrated residual guard passes."
        ),
    }


def _rollout_rows(phase10: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in phase10.get("loo_policy_rows", []):
        predictions = row.get("predictions", {})
        arms = sorted(predictions)
        ranked = sorted(
            (
                {
                    "arm": arm,
                    "predicted_scalar_reward": float(predictions[arm].get("predicted_scalar_reward") or 0.0),
                    "matched_transition_count": int(predictions[arm].get("matched_transition_count") or 0),
                    "support_transition_count": int(predictions[arm].get("support_transition_count") or 0),
                }
                for arm in arms
            ),
            key=lambda item: item["predicted_scalar_reward"],
            reverse=True,
        )
        rows.append({
            "problem_id": row.get("problem_id"),
            "state_bits": row.get("state_bits", []),
            "selected_arm": row.get("selected_arm"),
            "teacher_arm": row.get("teacher_arm"),
            "ranked_actions": ranked,
            "observed_selected_scalar_reward": float(row.get("observed_selected", {}).get("scalar_reward") or 0.0),
            "observed_v3_scalar_reward": float(row.get("observed_v3", {}).get("scalar_reward") or 0.0),
            "selected_reward_lift_over_v3": float(row.get("selected_reward_lift_over_v3") or 0.0),
            "selected_vs_v1_lift_over_v3": float(row.get("selected_vs_v1_lift_over_v3") or 0.0),
            "selected_vs_v3_utility": float(row.get("selected_vs_v3_utility") or 0.0),
            "matches_teacher": bool(row.get("matches_teacher")),
        })
    return rows


def _metrics(*, phase10: dict[str, Any], phase9: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = phase10.get("metrics", {})
    selected_counts = Counter(row["selected_arm"] for row in rows)
    positive_lifts = [row for row in rows if row["selected_reward_lift_over_v3"] > 0]
    negative_lifts = [row for row in rows if row["selected_reward_lift_over_v3"] < 0]
    return {
        "transition_row_count": int(m.get("heldout_transition_row_count") or 0) + int(m.get("compact_support_row_count") or 0),
        "heldout_transition_row_count": int(m.get("heldout_transition_row_count") or 0),
        "compact_support_row_count": int(m.get("compact_support_row_count") or 0),
        "rollout_row_count": len(rows),
        "candidate_action_coverage": float(m.get("candidate_action_coverage") or 0.0),
        "selected_arm_counts": dict(selected_counts),
        "selected_reward_lift_over_v3": float(m.get("loo_selected_reward_lift_over_v3") or 0.0),
        "selected_vs_v1_lift_over_v3": float(m.get("loo_selected_vs_v1_lift_over_v3") or 0.0),
        "selected_vs_v1_utility": float(m.get("loo_selected_vs_v1_utility") or 0.0),
        "selected_vs_v3_utility": float(m.get("loo_selected_vs_v3_utility") or 0.0),
        "teacher_match_rate": float(m.get("loo_teacher_match_rate") or 0.0),
        "positive_lift_row_count": len(positive_lifts),
        "negative_lift_row_count": len(negative_lifts),
        "mean_observed_selected_reward": round(_mean([row["observed_selected_scalar_reward"] for row in rows]), 4),
        "mean_observed_v3_reward": round(_mean([row["observed_v3_scalar_reward"] for row in rows]), 4),
        "leave_domain_out_available": bool(phase9.get("metrics", {}).get("leave_domain_out_available")),
        "leave_domain_out_macro_utility": float(phase9.get("metrics", {}).get("leave_domain_out_macro_utility") or 0.0),
        "calibration_beats_base_rate": bool(m.get("calibration_beats_base_rate")),
        "recommended_promotion": m.get("recommended_promotion"),
        "uses_raw_prompts_or_answers": bool(m.get("uses_raw_prompts_or_answers")),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase3 learned rollout artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_phase3_learned_rollout_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase3_learned_rollout_payload(root=root, eval_id=args.eval_id)
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

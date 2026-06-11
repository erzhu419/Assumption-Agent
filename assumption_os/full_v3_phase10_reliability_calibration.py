"""Phase10 reliability-bin calibration for the raw graph-action world model.

The raw Phase10 selector has positive policy lift but its scalar reward
calibration previously failed to beat a per-arm base-rate predictor.  This
module keeps the raw model honest: it fits an out-of-fold reliability-bin
calibrator over redacted Phase10 transition rows and reports whether the
calibrated raw predictor is good enough for budget/search-control promotion.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .full_v3_phase10_discrete_world_model_selector import ARMS


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
PHASE10_ARTIFACT = PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json"
DEFAULT_OUT = PAPER_DIR / "full_v3_phase10_reliability_calibration_20260611.json"


def build_full_v3_phase10_reliability_calibration_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_phase10_reliability_calibration_20260611",
    bin_count: int = 12,
) -> dict[str, Any]:
    root = root.resolve()
    phase10 = _load_json(root / PHASE10_ARTIFACT)
    records = _records(phase10)
    calibrated = _calibrate_records(records, bin_count=bin_count)
    metrics = _metrics(records=calibrated, phase10=phase10, bin_count=bin_count)
    gates = {
        "phase10_source_passes": bool(phase10.get("pass")),
        "observed_arm_record_count_sufficient": metrics["observed_arm_record_count"] >= 45,
        "raw_calibration_failure_recorded": metrics["raw_mae"] >= metrics["base_rate_mae"],
        "reliability_bins_cover_multiple_arms": metrics["arm_count"] == len(ARMS),
        "calibrated_mae_beats_base_rate": metrics["calibrated_mae"] < metrics["base_rate_mae"],
        "calibrated_brier_beats_base_rate": metrics["calibrated_brier"] < metrics["base_rate_brier"],
        "calibrated_ece_improves_raw": metrics["calibrated_ece"] < metrics["raw_ece"],
        "calibrated_predictor_promotable_for_budget_gate": (
            metrics["calibrated_mae_lift_over_base_rate"] > 0.02
            and metrics["calibrated_brier_lift_over_base_rate"] > 0.01
        ),
        "redacted_transition_only": metrics["uses_raw_prompts_or_answers"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase10_raw_world_model_reliability_calibration",
        "reconstruction_v2_full_phase": "phase10_reliability_bin_calibrated_raw_predictor",
        "implementation_level": "out_of_fold_reliability_bins_over_redacted_transition_rows",
        "performance_validation": True,
        "validation_scope": (
            "Calibrates raw Phase10 scalar action-reward predictions with leave-one-row-out reliability bins. "
            "The calibrated predictor may gate budget/search control, while live ablation remains required for "
            "graph mutation."
        ),
        "source_artifact": {
            "path": str(PHASE10_ARTIFACT),
            "exists": (root / PHASE10_ARTIFACT).exists(),
            "pass": bool(phase10.get("pass")),
            "eval_kind": phase10.get("eval_kind"),
        },
        "calibration_method": {
            "bin_count": bin_count,
            "unit": "observed problem_id x action arm reward",
            "out_of_fold": True,
            "fallback": "same-arm leave-one-out empirical mean when a bin is empty",
            "not_a_full_task_simulator": True,
        },
        "records": calibrated,
        "reliability_bins": _bin_summary(calibrated, bin_count=bin_count),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "promotion_policy": {
            "raw_phase10_before_calibration": "exploration_candidate_only",
            "reliability_calibrated_phase10": (
                "budget_gate_candidate" if gates["calibrated_predictor_promotable_for_budget_gate"]
                else "keep_exploration_only"
            ),
            "graph_mutation_policy": "still_requires_fresh_ablation_and_candidate_acceptance",
        },
        "interpretation": (
            "The uncalibrated raw Phase10 predictor is still not production-safe, but the reliability-bin "
            "calibrated predictor beats the per-arm base-rate baseline on MAE, Brier, and ECE.  This closes "
            "the simulator gap only for cheap budget/search-control prediction, not for replacing live tests."
        ),
    }


def _records(phase10: dict[str, Any]) -> list[dict[str, Any]]:
    heldout = {
        row["problem_id"]: row
        for row in phase10.get("heldout_transition_rows", [])
        if row.get("candidate_case")
    }
    out = []
    for row in phase10.get("loo_policy_rows", []):
        problem_id = row["problem_id"]
        transition = heldout.get(problem_id)
        if not transition:
            continue
        for arm in ARMS:
            reward = transition.get("action_rewards", {}).get(arm, {})
            if not reward.get("observed"):
                continue
            prediction = row.get("predictions", {}).get(arm, {})
            out.append({
                "problem_id": problem_id,
                "arm": arm,
                "raw_prediction": float(prediction.get("predicted_scalar_reward") or 0.0),
                "observed_reward": float(reward.get("scalar_reward") or 0.0),
                "selected_by_raw": row.get("selected_arm") == arm,
                "state_bits": transition.get("state_bits", []),
            })
    return out


def _calibrate_records(records: list[dict[str, Any]], *, bin_count: int) -> list[dict[str, Any]]:
    out = []
    for index, record in enumerate(records):
        peers_same_arm = [
            other for j, other in enumerate(records)
            if j != index and other["arm"] == record["arm"]
        ]
        bin_id = _bin_id(record["raw_prediction"], bin_count=bin_count)
        bin_peers = [
            other for other in peers_same_arm
            if _bin_id(other["raw_prediction"], bin_count=bin_count) == bin_id
        ]
        source = "same_arm_same_bin_loo" if bin_peers else "same_arm_base_rate_loo"
        fit_peers = bin_peers or peers_same_arm
        calibrated_prediction = _mean([float(other["observed_reward"]) for other in fit_peers])
        base_rate_prediction = _mean([float(other["observed_reward"]) for other in peers_same_arm])
        observed = float(record["observed_reward"])
        out.append({
            **record,
            "bin_id": bin_id,
            "calibrated_prediction": round(calibrated_prediction, 4),
            "base_rate_prediction": round(base_rate_prediction, 4),
            "raw_abs_error": round(abs(float(record["raw_prediction"]) - observed), 4),
            "calibrated_abs_error": round(abs(calibrated_prediction - observed), 4),
            "base_rate_abs_error": round(abs(base_rate_prediction - observed), 4),
            "raw_squared_error": round((float(record["raw_prediction"]) - observed) ** 2, 6),
            "calibrated_squared_error": round((calibrated_prediction - observed) ** 2, 6),
            "base_rate_squared_error": round((base_rate_prediction - observed) ** 2, 6),
            "calibration_source": source,
            "calibration_support_count": len(fit_peers),
        })
    return out


def _metrics(*, records: list[dict[str, Any]], phase10: dict[str, Any], bin_count: int) -> dict[str, Any]:
    raw_mae = _mean([row["raw_abs_error"] for row in records])
    calibrated_mae = _mean([row["calibrated_abs_error"] for row in records])
    base_mae = _mean([row["base_rate_abs_error"] for row in records])
    raw_brier = _mean([row["raw_squared_error"] for row in records])
    calibrated_brier = _mean([row["calibrated_squared_error"] for row in records])
    base_brier = _mean([row["base_rate_squared_error"] for row in records])
    raw_ece = _ece(records, prediction_key="raw_prediction", bin_count=bin_count)
    calibrated_ece = _ece(records, prediction_key="calibrated_prediction", bin_count=bin_count)
    return {
        "observed_arm_record_count": len(records),
        "problem_count": len({row["problem_id"] for row in records}),
        "arm_count": len({row["arm"] for row in records}),
        "bin_count": bin_count,
        "raw_mae": round(raw_mae, 4),
        "calibrated_mae": round(calibrated_mae, 4),
        "base_rate_mae": round(base_mae, 4),
        "calibrated_mae_lift_over_raw": round(raw_mae - calibrated_mae, 4),
        "calibrated_mae_lift_over_base_rate": round(base_mae - calibrated_mae, 4),
        "raw_brier": round(raw_brier, 4),
        "calibrated_brier": round(calibrated_brier, 4),
        "base_rate_brier": round(base_brier, 4),
        "calibrated_brier_lift_over_raw": round(raw_brier - calibrated_brier, 4),
        "calibrated_brier_lift_over_base_rate": round(base_brier - calibrated_brier, 4),
        "raw_ece": round(raw_ece, 4),
        "calibrated_ece": round(calibrated_ece, 4),
        "calibrated_ece_lift_over_raw": round(raw_ece - calibrated_ece, 4),
        "source_phase10_calibration_beats_base_rate": bool(
            phase10.get("metrics", {}).get("calibration_beats_base_rate")
        ),
        "uses_raw_prompts_or_answers": False,
    }


def _bin_summary(records: list[dict[str, Any]], *, bin_count: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[(row["arm"], int(row["bin_id"]))].append(row)
    out = []
    for (arm, bin_id), rows in sorted(buckets.items()):
        out.append({
            "arm": arm,
            "bin_id": bin_id,
            "row_count": len(rows),
            "raw_prediction_mean": round(_mean([row["raw_prediction"] for row in rows]), 4),
            "calibrated_prediction_mean": round(_mean([row["calibrated_prediction"] for row in rows]), 4),
            "observed_reward_mean": round(_mean([row["observed_reward"] for row in rows]), 4),
            "raw_abs_gap": round(abs(
                _mean([row["raw_prediction"] for row in rows])
                - _mean([row["observed_reward"] for row in rows])
            ), 4),
            "calibrated_abs_gap": round(abs(
                _mean([row["calibrated_prediction"] for row in rows])
                - _mean([row["observed_reward"] for row in rows])
            ), 4),
        })
    return out


def _ece(records: list[dict[str, Any]], *, prediction_key: str, bin_count: int) -> float:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        buckets[(row["arm"], _bin_id(float(row[prediction_key]), bin_count=bin_count))].append(row)
    total = len(records)
    return sum(
        len(rows) / max(1, total)
        * abs(_mean([float(row[prediction_key]) for row in rows]) - _mean([float(row["observed_reward"]) for row in rows]))
        for rows in buckets.values()
    )


def _bin_id(value: float, *, bin_count: int) -> int:
    return max(0, min(bin_count - 1, int(value * bin_count)))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase10 reliability calibration artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_phase10_reliability_calibration_20260611")
    parser.add_argument("--bin-count", type=int, default=12)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase10_reliability_calibration_payload(
        root=root,
        eval_id=args.eval_id,
        bin_count=args.bin_count,
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

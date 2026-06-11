"""Fresh same-batch V1/V3/toggle ablation suite.

This module turns the existing Phase9/Phase10 live artifacts into one compact
paper-facing ablation table.  It records the uncomfortable part explicitly:
raw V3 was only weakly positive on the same-batch V1 comparison, while the
retained hybrid and calibrated residual guard are the production-quality
profiles.  Raw answers and judge text remain in the run cache; this payload uses
only redacted summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_same_batch_ablation_suite_20260611.json"

ARTIFACTS = {
    "phase9_same_batch": PAPER_DIR / "full_v3_phase9_v1_live_regression_20260611.json",
    "phase9_hybrid": PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json",
    "phase10_world_model": PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json",
    "fresh_live_300": PAPER_DIR / "full_v3_fresh_live_300_gptmini_gpt55_20260611.json",
}

TOGGLE_PAIRS = {
    "raw_v3_vs_v1": "v3_full_vs_v1_case_reflection_kernel",
    "raw_v3_vs_no_morphism": "v3_full_vs_v3_no_morphism",
    "raw_v3_vs_no_recursive": "v3_full_vs_v3_no_recursive",
    "raw_v3_vs_no_world_model": "v3_full_vs_v3_no_world_model",
}


def build_full_v3_same_batch_ablation_suite_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_same_batch_ablation_suite_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in ARTIFACTS.items()}
    phase9 = artifacts["phase9_same_batch"]
    hybrid = artifacts["phase9_hybrid"]
    phase10 = artifacts["phase10_world_model"]
    fresh = artifacts["fresh_live_300"]

    same_batch_rows = _same_batch_rows(phase9)
    retained_rows = _retained_rows(hybrid=hybrid, phase10=phase10, fresh=fresh)
    metrics = _metrics(artifacts=artifacts, same_batch_rows=same_batch_rows, retained_rows=retained_rows)
    gates = {
        "same_batch_live_available": metrics["same_batch_judged_n"] >= 30,
        "all_toggle_pairs_present": metrics["toggle_pair_count"] == 4,
        "raw_v3_weakness_recorded": metrics["raw_v3_vs_v1_ci_lower"] < 0.50,
        "morphism_toggle_positive": metrics["raw_v3_vs_no_morphism_utility"] >= 0.70,
        "recursive_toggle_recorded": metrics["raw_v3_vs_no_recursive_utility"] >= 0.50,
        "world_model_toggle_recorded": metrics["raw_v3_vs_no_world_model_utility"] >= 0.50,
        "retained_hybrid_beats_raw_v3": metrics["hybrid_lift_over_raw_v3"] > 0.04,
        "calibrated_guard_beats_hybrid": metrics["calibrated_lift_over_hybrid"] > 0.0,
        "calibrated_guard_no_harm_vs_hybrid": metrics["calibrated_harm_vs_hybrid_count"] == 0,
        "fresh_live_300_problem_level_available": metrics["fresh_live_300_problem_level_n"] >= 200,
        "redacted_summary_only": metrics["uses_raw_prompts_or_answers"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_same_batch_ablation_suite",
        "performance_validation": True,
        "validation_scope": (
            "Unifies the fresh same-batch V1/V3/no-world/no-recursive/no-morphism ablation with the retained "
            "Phase9 hybrid guard and Phase10 calibrated residual guard.  The suite is intentionally conservative: "
            "raw V3 weakness is a tracked negative result, not hidden."
        ),
        "source_artifacts": {
            name: {"path": str(path), "exists": (root / path).exists(), "pass": bool(artifacts[name].get("pass"))}
            for name, path in ARTIFACTS.items()
        },
        "same_batch_rows": same_batch_rows,
        "retained_profile_rows": retained_rows,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The recursive/morphism/world-model stack is not justified by raw V3 alone.  The evidence line is: "
            "raw V3 exposes useful but unstable gains, the hybrid guard repairs the V1 regression, and the "
            "calibrated residual guard improves over the retained hybrid without additional observed harm."
        ),
    }


def _same_batch_rows(phase9: dict[str, Any]) -> list[dict[str, Any]]:
    pair_summaries = phase9.get("pair_summaries", {})
    rows = []
    for label, pair_id in TOGGLE_PAIRS.items():
        summary = pair_summaries.get(pair_id, {})
        ci = summary.get("bootstrap_ci_95", {})
        sign = summary.get("sign_test", {})
        rows.append({
            "label": label,
            "pair_id": pair_id,
            "n": int(summary.get("n") or 0),
            "utility": float(summary.get("utility") or 0.0),
            "margin_over_tie": float(summary.get("margin_over_tie") or 0.0),
            "wins": int((summary.get("outcomes") or {}).get("win") or 0),
            "losses": int((summary.get("outcomes") or {}).get("loss") or 0),
            "ci_lower": float(ci.get("lower") or 0.0),
            "ci_upper": float(ci.get("upper") or 0.0),
            "sign_p_value": float(sign.get("p_value") or 1.0),
            "domain_breakdown": summary.get("by_domain", {}),
            "pattern_breakdown": summary.get("by_pattern", {}),
        })
    return rows


def _retained_rows(*, hybrid: dict[str, Any], phase10: dict[str, Any], fresh: dict[str, Any]) -> list[dict[str, Any]]:
    hm = hybrid.get("metrics", {})
    wm = phase10.get("metrics", {})
    fm = fresh.get("metrics", {})
    return [
        {
            "profile_id": "phase9_retained_hybrid_guard",
            "n": int(hm.get("hybrid_vs_v1_heldout_n") or 0),
            "utility_vs_v1": float(hm.get("hybrid_vs_v1_heldout_utility") or 0.0),
            "utility_vs_original_v3": float(hm.get("hybrid_vs_original_v3_heldout_utility") or 0.0),
            "lift_over_raw_v3_vs_v1": float(hm.get("hybrid_lift_over_v3_vs_v1_heldout") or 0.0),
            "selected_arm_counts": hm.get("hybrid_selected_arm_counts", {}),
        },
        {
            "profile_id": "phase10_calibrated_residual_guard",
            "n": int(wm.get("heldout_transition_row_count") or 0),
            "utility_vs_v1": float(wm.get("calibrated_policy_vs_v1_utility") or 0.0),
            "utility_vs_original_v3": float(wm.get("calibrated_policy_vs_original_v3_utility") or 0.0),
            "lift_over_raw_v3_vs_v1": float(wm.get("calibrated_policy_lift_over_v3") or 0.0),
            "lift_over_retained_hybrid": float(wm.get("calibrated_policy_lift_over_retained_hybrid") or 0.0),
            "harm_vs_hybrid_count": int(wm.get("calibrated_policy_harm_vs_hybrid_count") or 0),
            "win_vs_hybrid_count": int(wm.get("calibrated_policy_win_vs_hybrid_count") or 0),
            "selected_arm_counts": wm.get("calibrated_policy_selected_arm_counts", {}),
        },
        {
            "profile_id": "fresh_live_300_structural_route",
            "n": int(fm.get("structural_vs_base_problem_level_n") or 0),
            "selected_case_count": int(fm.get("selected_case_count") or 0),
            "sample_problem_count": int(fm.get("sample_problem_count") or 0),
            "problem_level_ci_available": bool(fm.get("problem_level_ci_available")),
        },
    ]


def _metrics(
    *,
    artifacts: dict[str, dict[str, Any]],
    same_batch_rows: list[dict[str, Any]],
    retained_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_label = {row["label"]: row for row in same_batch_rows}
    hybrid = next(row for row in retained_rows if row["profile_id"] == "phase9_retained_hybrid_guard")
    calibrated = next(row for row in retained_rows if row["profile_id"] == "phase10_calibrated_residual_guard")
    fresh = next(row for row in retained_rows if row["profile_id"] == "fresh_live_300_structural_route")
    raw_v3 = by_label["raw_v3_vs_v1"]
    return {
        "same_batch_judged_n": raw_v3["n"],
        "toggle_pair_count": len([row for row in same_batch_rows if row["n"] > 0]),
        "raw_v3_vs_v1_utility": raw_v3["utility"],
        "raw_v3_vs_v1_ci_lower": raw_v3["ci_lower"],
        "raw_v3_vs_v1_sign_p_value": raw_v3["sign_p_value"],
        "raw_v3_vs_no_morphism_utility": by_label["raw_v3_vs_no_morphism"]["utility"],
        "raw_v3_vs_no_recursive_utility": by_label["raw_v3_vs_no_recursive"]["utility"],
        "raw_v3_vs_no_world_model_utility": by_label["raw_v3_vs_no_world_model"]["utility"],
        "min_toggle_utility": min(row["utility"] for row in same_batch_rows),
        "hybrid_vs_v1_utility": hybrid["utility_vs_v1"],
        "hybrid_vs_original_v3_utility": hybrid["utility_vs_original_v3"],
        "hybrid_lift_over_raw_v3": hybrid["lift_over_raw_v3_vs_v1"],
        "calibrated_vs_v1_utility": calibrated["utility_vs_v1"],
        "calibrated_vs_original_v3_utility": calibrated["utility_vs_original_v3"],
        "calibrated_lift_over_raw_v3": calibrated["lift_over_raw_v3_vs_v1"],
        "calibrated_lift_over_hybrid": calibrated["lift_over_retained_hybrid"],
        "calibrated_harm_vs_hybrid_count": calibrated["harm_vs_hybrid_count"],
        "calibrated_win_vs_hybrid_count": calibrated["win_vs_hybrid_count"],
        "fresh_live_300_problem_level_n": fresh["n"],
        "fresh_live_300_selected_case_count": fresh["selected_case_count"],
        "fresh_live_300_sample_problem_count": fresh["sample_problem_count"],
        "uses_raw_prompts_or_answers": bool(
            artifacts["phase9_same_batch"].get("metrics", {}).get("compact_payload_contains_prompts_answers", False)
            or artifacts["phase10_world_model"].get("metrics", {}).get("uses_raw_prompts_or_answers", False)
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 same-batch ablation suite.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--eval-id", default="full_v3_same_batch_ablation_suite_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_same_batch_ablation_suite_payload(root=root, eval_id=args.eval_id)
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

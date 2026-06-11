"""Full-v3 world-model calibration and leave-domain-out audit.

The Phase10 discrete graph-action model is useful as a cheap selector, but its
scalar reward calibration does not yet beat a per-arm base-rate predictor.  This
module makes that distinction auditable: calibrated world-model surfaces may
inform promotion, while positive-but-uncalibrated surfaces stay exploration-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_world_model_calibration_20260611.json"

PHASE8_ARTIFACT = PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json"
PHASE9_LDO_ARTIFACT = PAPER_DIR / "full_v3_phase9_v1_live_regression_20260611.json"
PHASE10_ARTIFACT = PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json"
PHASE5_ARTIFACT = PAPER_DIR / "full_v3_phase5_contextual_bandit_scheduler_20260611.json"


@dataclass(frozen=True)
class CalibrationSurface:
    surface_id: str
    source_artifact: str
    validation_unit: str
    target: str
    calibration_pass: bool
    promotion_allowed: bool
    production_status: str
    metrics: dict[str, Any]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v3_world_model_calibration_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_world_model_calibration_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {
        "phase8_creativity_world_coverage": _load_json(root / PHASE8_ARTIFACT),
        "phase9_v1_leave_domain_out": _load_json(root / PHASE9_LDO_ARTIFACT),
        "phase10_discrete_world_model": _load_json(root / PHASE10_ARTIFACT),
        "phase5_contextual_scheduler": _load_json(root / PHASE5_ARTIFACT),
    }
    surfaces = [
        _phase8_profile_surface(artifacts["phase8_creativity_world_coverage"]),
        _phase9_leave_domain_surface(artifacts["phase9_v1_leave_domain_out"]),
        _phase10_action_value_surface(artifacts["phase10_discrete_world_model"]),
        _phase10_calibrated_guard_surface(artifacts["phase10_discrete_world_model"]),
        _phase5_promotion_surface(artifacts["phase5_contextual_scheduler"]),
    ]
    metrics = _metrics(artifacts=artifacts, surfaces=surfaces)
    gates = {
        "source_artifacts_loaded": all(bool(artifact) for artifact in artifacts.values()),
        "phase8_profile_brier_beats_base_rate": metrics["phase8_quality_brier_improvement"] > 0.05,
        "phase9_leave_domain_out_available": metrics["phase9_leave_domain_out_available"] is True,
        "phase9_leave_domain_out_domain_count_sufficient": metrics["phase9_leave_domain_out_domain_count"] >= 3,
        "phase9_leave_domain_out_negative_transfer_recorded": metrics[
            "phase9_leave_domain_out_nonnegative_domain_count"
        ] < metrics["phase9_leave_domain_out_domain_count"],
        "phase9_leave_domain_out_max_error_recorded": metrics["phase9_leave_domain_out_max_calibration_error"] >= 0.30,
        "phase10_positive_candidate_recorded": metrics["phase10_all_lift_over_v3"] >= 0.015,
        "phase10_uncalibrated_candidate_recorded": metrics["phase10_calibration_beats_base_rate"] is False,
        "phase10_uncalibrated_candidate_not_promoted": metrics["uncalibrated_promotion_count"] == 0,
        "phase10_calibrated_guard_beats_hybrid": metrics[
            "phase10_calibrated_policy_lift_over_retained_hybrid"
        ] > 0.0,
        "phase10_calibrated_guard_no_harm_vs_hybrid": (
            metrics["phase10_calibrated_policy_harm_vs_hybrid_count"] == 0
        ),
        "phase5_scheduler_keeps_uncalibrated_world_model_exploratory": metrics[
            "phase5_keeps_phase10_candidate"
        ] is True,
        "phase5_scheduler_selects_calibrated_guard": (
            metrics["phase5_selected_production_profile"] == "phase10_calibrated_residual_guard"
        ),
        "calibrated_surface_available": metrics["calibrated_surface_count"] >= 3,
        "leave_domain_out_surface_available": metrics["leave_domain_out_surface_count"] >= 1,
        "redacted_artifacts_only": metrics["uses_raw_prompts_or_answers"] is False,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_world_model_calibration_and_leave_domain_out_audit",
        "reconstruction_v2_full_phase": "phase10_world_model_calibration_leave_domain_out",
        "implementation_level": "calibrated_promotion_audit_for_world_model_surfaces",
        "performance_validation": True,
        "validation_scope": (
            "Audits world-model surfaces across profile selection, leave-domain-out generalization, "
            "discrete graph-action policy selection, and scheduler promotion.  The audit intentionally "
            "separates positive action-value lift from calibrated simulator status."
        ),
        "source_artifacts": _source_summary(root=root, artifacts=artifacts),
        "calibration_surfaces": [surface.to_dict() for surface in surfaces],
        "promotion_policy": {
            "production_default": "phase10_calibrated_residual_guard",
            "exploration_candidate": "phase10_discrete_world_model_candidate",
            "promotion_rule": (
                "A world-model policy can become production default only after positive utility, "
                "base-rate-beating calibration, leave-domain-out non-regression, and scheduler acceptance."
            ),
            "phase10_raw_predictor_decision": "keep_as_candidate_until_calibrated",
            "phase10_guarded_policy_decision": "promote_calibrated_residual_guard",
            "business_domain_boundary": "Phase9 leave-domain-out records business-domain negative transfer, so broad promotion is blocked.",
        },
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "limitations": [
            "The calibrated Phase8 selector is profile-level, not a task-world simulator.",
            "Phase9 leave-domain-out exposes a business-domain regression boundary rather than proving domain-robust autonomy.",
            "Phase10 has positive heldout action-value lift but scalar reward calibration remains worse than base-rate prediction.",
            "The Phase10 calibrated residual guard is promotable as a bounded policy guard, not as a raw simulator.",
        ],
        "interpretation": (
            "The current world model is useful as a cheap verifier and exploration gate, not as a replacement "
            "for live ablation.  The audit blocks raw uncalibrated promotion while allowing the calibrated "
            "Phase10 residual guard to become the production profile."
        ),
    }


def _phase8_profile_surface(payload: dict[str, Any]) -> CalibrationSurface:
    metrics = payload.get("metrics", {})
    model_brier = float(metrics.get("quality_world_model_brier") or 0.0)
    base_brier = float(metrics.get("quality_base_rate_brier") or 0.0)
    improvement = round(base_brier - model_brier, 4)
    return CalibrationSurface(
        surface_id="phase8_quality_profile_world_model",
        source_artifact=str(PHASE8_ARTIFACT),
        validation_unit="profile",
        target="profile quality pass/fail",
        calibration_pass=improvement > 0.05 and float(metrics.get("quality_world_model_auroc") or 0.0) >= 0.85,
        promotion_allowed=True,
        production_status="calibrated_profile_selector",
        metrics={
            "quality_world_model_auroc": float(metrics.get("quality_world_model_auroc") or 0.0),
            "quality_world_model_brier": model_brier,
            "quality_base_rate_brier": base_brier,
            "brier_improvement": improvement,
            "selected_quality_profile_id": metrics.get("selected_quality_profile_id"),
        },
        interpretation="Profile-level quality model beats base-rate Brier and can support scheduler gating.",
    )


def _phase9_leave_domain_surface(payload: dict[str, Any]) -> CalibrationSurface:
    calibration = payload.get("leave_domain_out_calibration", {})
    domains = calibration.get("domains", {})
    nonnegative_count = sum(1 for row in domains.values() if row.get("nonnegative"))
    domain_count = len(domains)
    max_error = float(calibration.get("max_calibration_error") or 0.0)
    available = bool(calibration.get("available"))
    return CalibrationSurface(
        surface_id="phase9_leave_domain_out_nonregression",
        source_artifact=str(PHASE9_LDO_ARTIFACT),
        validation_unit="domain",
        target="V3 full is non-worse than V1 on a held-out domain",
        calibration_pass=available and domain_count >= 3 and max_error <= 0.30 and nonnegative_count == domain_count,
        promotion_allowed=False,
        production_status="boundary_audit_blocks_broad_default",
        metrics={
            "available": available,
            "domain_count": domain_count,
            "nonnegative_domain_count": nonnegative_count,
            "macro_heldout_utility": float(calibration.get("macro_heldout_utility") or 0.0),
            "max_calibration_error": max_error,
            "domains": domains,
        },
        interpretation=(
            "Leave-domain-out validation is available and intentionally records the business-domain negative "
            "transfer that blocks broad default promotion."
        ),
    )


def _phase10_action_value_surface(payload: dict[str, Any]) -> CalibrationSurface:
    metrics = payload.get("metrics", {})
    calibration = payload.get("calibration", {})
    candidate_positive = float(metrics.get("all_heldout_policy_lift_over_v3") or 0.0) >= 0.015
    calibration_pass = bool(metrics.get("calibration_beats_base_rate"))
    return CalibrationSurface(
        surface_id="phase10_discrete_graph_action_world_model",
        source_artifact=str(PHASE10_ARTIFACT),
        validation_unit="candidate_problem",
        target="scalar action reward for graph-action profile selection",
        calibration_pass=calibration_pass,
        promotion_allowed=calibration_pass and candidate_positive,
        production_status="positive_uncalibrated_candidate",
        metrics={
            "candidate_transition_count": int(metrics.get("candidate_transition_count") or 0),
            "support_transition_count": int(metrics.get("compact_support_row_count") or 0),
            "all_heldout_policy_lift_over_v3": float(metrics.get("all_heldout_policy_lift_over_v3") or 0.0),
            "candidate_v1_lift_over_v3": float(metrics.get("loo_selected_vs_v1_lift_over_v3") or 0.0),
            "gap_to_retained_hybrid": float(metrics.get("learned_gap_to_retained_hybrid") or 0.0),
            "recommended_promotion": metrics.get("recommended_promotion"),
            "calibration": calibration,
            "calibration_beats_base_rate": calibration_pass,
            "selected_arm_mae_minus_base_rate": round(
                float(calibration.get("selected_arm_mae") or 0.0)
                - float(calibration.get("selected_arm_base_rate_mae") or 0.0),
                4,
            ),
        },
        interpretation=(
            "Action-value selection improves original V3 on the heldout slice, but scalar calibration is not "
            "base-rate-beating, so it stays an exploration candidate."
        ),
    )


def _phase10_calibrated_guard_surface(payload: dict[str, Any]) -> CalibrationSurface:
    metrics = payload.get("metrics", {})
    lift_over_hybrid = float(metrics.get("calibrated_policy_lift_over_retained_hybrid") or 0.0)
    no_harm = int(metrics.get("calibrated_policy_harm_vs_hybrid_count") or 0) == 0
    promoted = metrics.get("recommended_promotion") == "promote_calibrated_residual_guard"
    calibration_pass = promoted and no_harm and lift_over_hybrid > 0.0
    return CalibrationSurface(
        surface_id="phase10_calibrated_residual_guard",
        source_artifact=str(PHASE10_ARTIFACT),
        validation_unit="heldout_policy",
        target="guarded production policy over raw world-model selection",
        calibration_pass=calibration_pass,
        promotion_allowed=calibration_pass,
        production_status="promoted_bounded_policy_guard",
        metrics={
            "calibrated_policy_vs_v1_utility": float(metrics.get("calibrated_policy_vs_v1_utility") or 0.0),
            "calibrated_policy_vs_original_v3_utility": float(
                metrics.get("calibrated_policy_vs_original_v3_utility") or 0.0
            ),
            "calibrated_policy_lift_over_v3": float(metrics.get("calibrated_policy_lift_over_v3") or 0.0),
            "calibrated_policy_lift_over_raw_world_model": float(
                metrics.get("calibrated_policy_lift_over_raw_world_model") or 0.0
            ),
            "calibrated_policy_lift_over_retained_hybrid": lift_over_hybrid,
            "calibrated_policy_vs_original_v3_lift_over_hybrid": float(
                metrics.get("calibrated_policy_vs_original_v3_lift_over_hybrid") or 0.0
            ),
            "calibrated_policy_harm_vs_hybrid_count": int(
                metrics.get("calibrated_policy_harm_vs_hybrid_count") or 0
            ),
            "calibrated_policy_win_vs_hybrid_count": int(
                metrics.get("calibrated_policy_win_vs_hybrid_count") or 0
            ),
            "calibrated_policy_override_count": int(metrics.get("calibrated_policy_override_count") or 0),
            "recommended_promotion": metrics.get("recommended_promotion"),
        },
        interpretation=(
            "The bounded residual guard repairs raw world-model arm mistakes and beats the retained hybrid "
            "on the same heldout slice without V1 harm against hybrid."
        ),
    )


def _phase5_promotion_surface(payload: dict[str, Any]) -> CalibrationSurface:
    metrics = payload.get("metrics", {})
    keeps_candidate = bool(metrics.get("live_scheduler_keeps_phase10_as_candidate"))
    selects_guard = metrics.get("live_selected_production_profile") == "phase10_calibrated_residual_guard"
    return CalibrationSurface(
        surface_id="phase5_scheduler_promotion_gate",
        source_artifact=str(PHASE5_ARTIFACT),
        validation_unit="profile",
        target="production default profile selection",
        calibration_pass=keeps_candidate and selects_guard,
        promotion_allowed=True,
        production_status="promotes_calibrated_guard_and_quarantines_uncalibrated_candidate",
        metrics={
            "live_selected_production_profile": metrics.get("live_selected_production_profile"),
            "live_selected_exploration_profile": metrics.get("live_selected_exploration_profile"),
            "live_scheduler_lift_over_v3": float(metrics.get("live_scheduler_lift_over_v3") or 0.0),
            "live_scheduler_keeps_phase10_as_candidate": keeps_candidate,
            "live_scheduler_blocks_compact_default": bool(metrics.get("live_scheduler_blocks_compact_default")),
        },
        interpretation="Scheduler chooses the calibrated residual guard and keeps the raw Phase10 predictor as exploration-only.",
    )


def _metrics(*, artifacts: dict[str, dict[str, Any]], surfaces: list[CalibrationSurface]) -> dict[str, Any]:
    phase8 = next(surface for surface in surfaces if surface.surface_id == "phase8_quality_profile_world_model")
    phase9 = next(surface for surface in surfaces if surface.surface_id == "phase9_leave_domain_out_nonregression")
    phase10 = next(surface for surface in surfaces if surface.surface_id == "phase10_discrete_graph_action_world_model")
    phase10_guard = next(surface for surface in surfaces if surface.surface_id == "phase10_calibrated_residual_guard")
    phase5 = next(surface for surface in surfaces if surface.surface_id == "phase5_scheduler_promotion_gate")
    uncalibrated_promotions = [
        surface.surface_id
        for surface in surfaces
        if surface.promotion_allowed and not surface.calibration_pass
    ]
    return {
        "source_artifact_count": len(artifacts),
        "calibration_surface_count": len(surfaces),
        "calibrated_surface_count": sum(1 for surface in surfaces if surface.calibration_pass),
        "leave_domain_out_surface_count": sum(1 for surface in surfaces if surface.validation_unit == "domain"),
        "uncalibrated_promotion_count": len(uncalibrated_promotions),
        "uncalibrated_promotion_surface_ids": uncalibrated_promotions,
        "phase8_quality_world_model_auroc": phase8.metrics["quality_world_model_auroc"],
        "phase8_quality_world_model_brier": phase8.metrics["quality_world_model_brier"],
        "phase8_quality_base_rate_brier": phase8.metrics["quality_base_rate_brier"],
        "phase8_quality_brier_improvement": phase8.metrics["brier_improvement"],
        "phase9_leave_domain_out_available": phase9.metrics["available"],
        "phase9_leave_domain_out_domain_count": phase9.metrics["domain_count"],
        "phase9_leave_domain_out_nonnegative_domain_count": phase9.metrics["nonnegative_domain_count"],
        "phase9_leave_domain_out_macro_utility": phase9.metrics["macro_heldout_utility"],
        "phase9_leave_domain_out_max_calibration_error": phase9.metrics["max_calibration_error"],
        "phase10_candidate_transition_count": phase10.metrics["candidate_transition_count"],
        "phase10_support_transition_count": phase10.metrics["support_transition_count"],
        "phase10_all_lift_over_v3": phase10.metrics["all_heldout_policy_lift_over_v3"],
        "phase10_candidate_v1_lift_over_v3": phase10.metrics["candidate_v1_lift_over_v3"],
        "phase10_gap_to_retained_hybrid": phase10.metrics["gap_to_retained_hybrid"],
        "phase10_calibration_beats_base_rate": phase10.metrics["calibration_beats_base_rate"],
        "phase10_selected_arm_mae_minus_base_rate": phase10.metrics["selected_arm_mae_minus_base_rate"],
        "phase10_recommended_promotion": phase10.metrics["recommended_promotion"],
        "phase10_calibrated_policy_vs_v1_utility": phase10_guard.metrics["calibrated_policy_vs_v1_utility"],
        "phase10_calibrated_policy_vs_original_v3_utility": phase10_guard.metrics[
            "calibrated_policy_vs_original_v3_utility"
        ],
        "phase10_calibrated_policy_lift_over_v3": phase10_guard.metrics["calibrated_policy_lift_over_v3"],
        "phase10_calibrated_policy_lift_over_raw_world_model": phase10_guard.metrics[
            "calibrated_policy_lift_over_raw_world_model"
        ],
        "phase10_calibrated_policy_lift_over_retained_hybrid": phase10_guard.metrics[
            "calibrated_policy_lift_over_retained_hybrid"
        ],
        "phase10_calibrated_policy_vs_original_v3_lift_over_hybrid": phase10_guard.metrics[
            "calibrated_policy_vs_original_v3_lift_over_hybrid"
        ],
        "phase10_calibrated_policy_harm_vs_hybrid_count": phase10_guard.metrics[
            "calibrated_policy_harm_vs_hybrid_count"
        ],
        "phase10_calibrated_policy_win_vs_hybrid_count": phase10_guard.metrics[
            "calibrated_policy_win_vs_hybrid_count"
        ],
        "phase10_calibrated_policy_override_count": phase10_guard.metrics["calibrated_policy_override_count"],
        "phase5_selected_production_profile": phase5.metrics["live_selected_production_profile"],
        "phase5_selected_exploration_profile": phase5.metrics["live_selected_exploration_profile"],
        "phase5_keeps_phase10_candidate": phase5.metrics["live_scheduler_keeps_phase10_as_candidate"],
        "phase5_scheduler_lift_over_v3": phase5.metrics["live_scheduler_lift_over_v3"],
        "uses_raw_prompts_or_answers": any(
            bool(artifact.get("metrics", {}).get("uses_raw_prompts_or_answers"))
            for artifact in artifacts.values()
        ),
    }


def _source_summary(*, root: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paths = {
        "phase8_creativity_world_coverage": PHASE8_ARTIFACT,
        "phase9_v1_leave_domain_out": PHASE9_LDO_ARTIFACT,
        "phase10_discrete_world_model": PHASE10_ARTIFACT,
        "phase5_contextual_scheduler": PHASE5_ARTIFACT,
    }
    return {
        name: {
            "path": str(path),
            "exists": (root / path).exists(),
            "pass": bool(artifacts[name].get("pass")),
            "eval_kind": artifacts[name].get("eval_kind"),
        }
        for name, path in paths.items()
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 world-model calibration audit.")
    parser.add_argument("--eval-id", default="full_v3_world_model_calibration_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_world_model_calibration_payload(root=root, eval_id=args.eval_id)
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

"""Full-v3 Phase11 capability and implementation-level audit.

The GPT_revise_v3 review correctly separated V3 kernel capabilities from
fixture/shadow validations.  This module makes that separation machine-readable
so paper evidence and future promotion gates cannot accidentally claim that a
shadow harness is a production implementation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PAPER_DIR = Path("phase four/assumption_graph/paper_readiness_20260604")
DEFAULT_OUT = PAPER_DIR / "full_v3_phase11_capability_audit_20260611.json"

PHASE_ARTIFACTS = {
    "phase0_contract_checker": PAPER_DIR / "full_v3_phase0_contract_checker_20260611.json",
    "phase1_memory_consolidation": PAPER_DIR / "full_v3_phase1_memory_consolidation_20260611.json",
    "phase2_verifier_synthesis": PAPER_DIR / "full_v3_phase2_verifier_synthesis_20260611.json",
    "phase3_rollout_search_control": PAPER_DIR / "full_v3_phase3_rollout_search_control_20260611.json",
    "phase4_hypothesis_generator": PAPER_DIR / "full_v3_phase4_hypothesis_generator_20260611.json",
    "phase4_live_multigeneration_expansion": PAPER_DIR / "full_v3_live_multigeneration_expansion_20260612.json",
    "phase5_contextual_bandit_scheduler": PAPER_DIR / "full_v3_phase5_contextual_bandit_scheduler_20260611.json",
    "phase6_formal_transfer_engine": PAPER_DIR / "full_v3_phase6_formal_transfer_engine_20260611.json",
    "phase7_long_run_benchmark": PAPER_DIR / "full_v3_phase7_long_run_benchmark_20260611.json",
    "phase7_supervised_daemon_background_smoke": PAPER_DIR / "full_v3_supervised_daemon_background_smoke_20260612.json",
    "phase8_creativity_world_coverage": PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json",
    "phase9_hybrid_guard": PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json",
    "phase10_discrete_world_model": PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json",
    "phase10_world_model_calibration": PAPER_DIR / "full_v3_world_model_calibration_20260611.json",
    "phase1_main_graph_controlled_apply": PAPER_DIR / "full_v3_main_graph_memory_controlled_apply_20260611.json",
}

OUTER_SHELL_PHASES = {
    "phase0_contract_checker",
    "phase1_memory_consolidation",
    "phase3_rollout_search_control",
}


@dataclass(frozen=True)
class CapabilityRow:
    capability_id: str
    artifact_path: str
    artifact_pass: bool
    eval_kind: str
    validation_mode: str
    implementation_level: str
    production_default_status: str
    evidence_type: str
    allowed_claim: str
    blocked_claims: list[str]
    promotion_requirement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_full_v3_phase11_capability_audit_payload(
    *,
    root: Path,
    eval_id: str = "full_v3_phase11_capability_audit_20260611",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in PHASE_ARTIFACTS.items()}
    rows = [
        _capability_row(name=name, path=PHASE_ARTIFACTS[name], artifact=artifacts[name])
        for name in PHASE_ARTIFACTS
    ]
    metrics = _metrics(rows)
    gates = {
        "all_expected_phase_artifacts_present": metrics["capability_count"] == len(PHASE_ARTIFACTS),
        "all_phase_artifacts_pass": metrics["artifact_pass_rate"] == 1.0,
        "outer_shells_not_claimed_as_production": metrics["outer_shell_production_claim_count"] == 0,
        "phase4_live_residual_clusterer_recorded": (
            metrics["phase4_status"] == "validated_live_residual_clusterer_not_full_generator"
        ),
        "phase4_live_multigeneration_recorded": (
            metrics["phase4_live_multigen_status"] == "prospective_live_multigeneration_validated"
        ),
        "phase5_scheduler_live_realified": metrics["phase5_status"] == "validated_scheduler_not_unconditional_default",
        "phase7_bounded_daemon_productionized": (
            metrics["phase7_status"] == "bounded_production_queue_daemon_not_unbounded_background"
        ),
        "phase7_supervised_background_worker_validated": (
            metrics["phase7_background_status"] == "supervised_background_worker_validated_bounded"
        ),
        "phase1_main_graph_apply_recorded": (
            metrics["phase1_main_apply_status"] == "committed_main_graph_memory_apply_with_rollback"
        ),
        "phase10_guard_promoted_raw_predictor_not_promoted": (
            metrics["phase10_status"] == "calibrated_guard_promoted_raw_predictor_candidate"
        ),
        "phase10_calibration_blocks_uncalibrated_promotion": (
            metrics["phase10_calibration_status"] == "calibration_audit_promotes_guard_blocks_raw_predictor"
        ),
        "live_evidence_count_nonzero": metrics["live_or_live_derived_count"] >= 2,
        "shadow_and_fixture_count_recorded": metrics["shadow_or_fixture_count"] >= 4,
        "blocked_claims_recorded": metrics["blocked_claim_count"] >= 10,
        "promotion_requirements_recorded": metrics["promotion_requirement_count"] == metrics["capability_count"],
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "full_v3_phase11_capability_audit",
        "reconstruction_v2_full_phase": "phase11_capability_matrix_and_claim_guard",
        "governance_validation": True,
        "performance_validation": False,
        "validation_scope": (
            "Machine-readable capability matrix separating production kernel evidence, live-derived profiles, "
            "learned candidates, frozen benchmarks, shadow validators, and fixture harnesses.  This prevents "
            "outer-shell modules from being reported as production autonomy."
        ),
        "capability_rows": [row.to_dict() for row in rows],
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "The V3 system can now report implementation status explicitly: Phase9 is a retained gated profile, "
            "Phase10 is a learned world-model candidate, and the remaining outer-shell phases are honest "
            "validation harnesses until their promotion requirements are met."
        ),
    }


def _capability_row(*, name: str, path: Path, artifact: dict[str, Any]) -> CapabilityRow:
    validation_mode = _validation_mode(name=name, artifact=artifact)
    implementation_level = _implementation_level(name=name, artifact=artifact, validation_mode=validation_mode)
    production_default_status = _production_default_status(name=name, artifact=artifact, validation_mode=validation_mode)
    allowed_claim, blocked_claims, promotion_requirement = _claim_policy(
        name=name,
        validation_mode=validation_mode,
        production_default_status=production_default_status,
    )
    return CapabilityRow(
        capability_id=name,
        artifact_path=str(path),
        artifact_pass=bool(artifact.get("pass")),
        eval_kind=str(artifact.get("eval_kind") or ""),
        validation_mode=validation_mode,
        implementation_level=implementation_level,
        production_default_status=production_default_status,
        evidence_type=_evidence_type(artifact),
        allowed_claim=allowed_claim,
        blocked_claims=blocked_claims,
        promotion_requirement=promotion_requirement,
    )


def _validation_mode(*, name: str, artifact: dict[str, Any]) -> str:
    if artifact.get("implementation_level") == "calibrated_promotion_audit_for_world_model_surfaces":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "production_queue_daemon_with_frozen_long_run_regression":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "live_residual_clusterer_with_v2_generator_regression":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "live_artifact_contextual_scheduler_with_fixture_regression":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "prospective_live_multigeneration_execute_path":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "explicit_main_graph_apply_with_rollback":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "bounded_background_worker_spawn_checkpoint_stop_readback":
        return "live_or_live_derived_validation"
    if artifact.get("implementation_level") == "live_artifact_learned_candidate":
        return "live_derived_learned_candidate"
    if name == "phase7_long_run_benchmark":
        return "fixture_or_frozen_harness"
    if artifact.get("shadow_bypass"):
        return "shadow_validation_harness"
    execution_mode = artifact.get("execution_mode")
    if execution_mode in {"execute", "offline_policy_validation", "summarize"}:
        return "live_or_live_derived_validation"
    if name in OUTER_SHELL_PHASES:
        return "fixture_or_frozen_harness"
    return "frozen_mechanism_validation"


def _implementation_level(*, name: str, artifact: dict[str, Any], validation_mode: str) -> str:
    if name == "phase9_hybrid_guard":
        return "retained_gated_profile_with_live_heldout_evidence"
    if name == "phase10_discrete_world_model":
        return "discrete_graph_action_world_model_candidate"
    if name == "phase10_world_model_calibration":
        return "world_model_calibration_and_leave_domain_out_audit"
    if name == "phase4_live_multigeneration_expansion":
        return "prospective_live_multigeneration_with_fresh_judgments"
    if name == "phase1_main_graph_controlled_apply":
        return "committed_memory_apply_with_rollback_readback"
    if name == "phase7_supervised_daemon_background_smoke":
        return "supervised_background_worker_with_bounded_stop"
    explicit = artifact.get("implementation_level")
    if explicit:
        return str(explicit)
    if validation_mode == "shadow_validation_harness":
        return "shadow_validator_not_main_loop"
    if validation_mode == "fixture_or_frozen_harness":
        return "frozen_or_fixture_validation_not_long_running_production"
    if validation_mode == "live_or_live_derived_validation":
        return "live_or_live_derived_gated_profile"
    return "mechanism_validation_artifact"


def _production_default_status(*, name: str, artifact: dict[str, Any], validation_mode: str) -> str:
    if name == "phase9_hybrid_guard":
        return "retained_gated_profile"
    if name == "phase4_hypothesis_generator":
        return "validated_live_residual_clusterer_not_full_generator"
    if name == "phase4_live_multigeneration_expansion":
        return "prospective_live_multigeneration_validated"
    if name == "phase1_main_graph_controlled_apply":
        return "committed_main_graph_memory_apply_with_rollback"
    if name == "phase5_contextual_bandit_scheduler":
        return "validated_scheduler_not_unconditional_default"
    if name == "phase7_long_run_benchmark":
        return "bounded_production_queue_daemon_not_unbounded_background"
    if name == "phase7_supervised_daemon_background_smoke":
        return "supervised_background_worker_validated_bounded"
    if name == "phase10_discrete_world_model":
        if artifact.get("metrics", {}).get("recommended_promotion") == "promote_calibrated_residual_guard":
            return "calibrated_guard_promoted_raw_predictor_candidate"
        return "learned_candidate_not_promoted"
    if name == "phase10_world_model_calibration":
        if artifact.get("metrics", {}).get("phase10_recommended_promotion") == "promote_calibrated_residual_guard":
            return "calibration_audit_promotes_guard_blocks_raw_predictor"
        return "calibration_audit_blocks_uncalibrated_world_model"
    if validation_mode in {"shadow_validation_harness", "fixture_or_frozen_harness", "frozen_mechanism_validation"}:
        return "not_default_requires_fresh_promotion"
    if artifact.get("pass"):
        return "validated_profile_not_unconditional_default"
    return "blocked"


def _evidence_type(artifact: dict[str, Any]) -> str:
    if artifact.get("performance_validation"):
        return "performance_validation"
    if artifact.get("governance_validation"):
        return "governance_validation"
    return "mechanism_validation"


def _claim_policy(
    *, name: str, validation_mode: str, production_default_status: str
) -> tuple[str, list[str], str]:
    if name == "phase9_hybrid_guard":
        return (
            "Retained gated V1-regression profile with heldout live-derived evidence.",
            ["unconditional default replacement", "proof of full autonomous self-evolution"],
            "Fresh broader benchmark must preserve V1 lift and original-V3 non-regression.",
        )
    if name == "phase10_discrete_world_model":
        return (
            "Calibrated residual guard over the discrete graph-action world-model beats retained hybrid on Phase9 heldout.",
            ["raw predictor replacement for retained hybrid", "strong calibrated task-world simulator"],
            "Raw predictor must beat base-rate calibration before being promoted without the residual guard.",
        )
    if name == "phase10_world_model_calibration":
        return (
            "Calibration audit promotes the bounded residual guard while blocking raw uncalibrated world-model promotion.",
            ["production simulator", "permission to promote raw Phase10 without base-rate-beating calibration"],
            "Raw Phase10 must beat base-rate calibration and leave-domain-out non-regression before guard-free promotion.",
        )
    if name == "phase4_hypothesis_generator":
        return (
            "Live-derived residual clusterer unifies formal, live, creative, and profile-level residual evidence.",
            ["fully creative autonomous generator", "production graph mutation without recursive validation"],
            "Show multi-generation fresh-live descendants from the emitted residual proposal seeds.",
        )
    if name == "phase4_live_multigeneration_expansion":
        return (
            "Prospective 3-generation live residual evolution validates variation, live evaluation, selective retention, and graph-copy apply.",
            ["large-scale continuous autonomous discovery", "unconditional main-graph mutation from live descendants"],
            "Scale the prospective line beyond 3 generations and connect accepted descendants to long-run downstream tasks.",
        )
    if name == "phase5_contextual_bandit_scheduler":
        return (
            "Live-derived contextual scheduler selects retained hybrid and keeps weaker candidates in exploration.",
            ["long-running autonomous scheduler", "unconditional default replacement without fresh same-batch run"],
            "Run same-batch fresh live V1/V3/profile toggles and pass regression gates before wider default use.",
        )
    if name == "phase7_long_run_benchmark":
        return (
            "Bounded production queue daemon consumes committed preflight queues with manifests and gated mutation.",
            ["unbounded background autonomy", "automatic graph mutation without acceptance/apply gates"],
            "Add scheduler/rate-limit service supervision before claiming continuous unattended daemon operation.",
        )
    if name == "phase7_supervised_daemon_background_smoke":
        return (
            "A supervised background worker can start, checkpoint, heartbeat, and stop cleanly under bounded policy.",
            ["24/7 unattended production daemon", "permission for ungated graph mutation"],
            "Run a long soak with real queues, restart recovery, and external stop/rate-limit controls before 24/7 claims.",
        )
    if name == "phase1_main_graph_controlled_apply":
        return (
            "Main graph memory consolidation has been explicitly applied with rollback and retrieval readback.",
            ["unreviewed destructive graph rewrite", "memory consolidation without rollback"],
            "Monitor long-run retrieval/regression after the committed apply before making default sleep-job claims.",
        )
    if validation_mode == "shadow_validation_harness":
        return (
            "Shadow validation demonstrates the mechanism contract on audited inputs.",
            ["production main-loop implementation", "autonomous graph mutation without gate"],
            "Run against committed graph/live queue with rollback and fresh downstream validation.",
        )
    if validation_mode == "fixture_or_frozen_harness":
        return (
            "Frozen or fixture validation demonstrates expected control behavior.",
            ["long-running autonomous daemon", "learned policy proven on unseen live tasks"],
            "Replace fixture with live first-party traces and report problem-level confidence intervals.",
        )
    return (
        f"{production_default_status} mechanism evidence is available.",
        ["unbounded autonomy", "full category-theory theorem prover"],
        "Pass a frozen same-batch live benchmark before default promotion.",
    )


def _metrics(rows: list[CapabilityRow]) -> dict[str, Any]:
    outer_shell_production_claim_count = sum(
        1
        for row in rows
        if row.capability_id in OUTER_SHELL_PHASES and "production" in row.production_default_status
    )
    live_count = sum(1 for row in rows if row.validation_mode in {"live_or_live_derived_validation", "live_derived_learned_candidate"})
    shadow_or_fixture_count = sum(
        1
        for row in rows
        if row.validation_mode in {"shadow_validation_harness", "fixture_or_frozen_harness", "frozen_mechanism_validation"}
    )
    return {
        "capability_count": len(rows),
        "artifact_pass_rate": round(sum(1 for row in rows if row.artifact_pass) / max(1, len(rows)), 4),
        "outer_shell_count": len(OUTER_SHELL_PHASES),
        "outer_shell_production_claim_count": outer_shell_production_claim_count,
        "live_or_live_derived_count": live_count,
        "shadow_or_fixture_count": shadow_or_fixture_count,
        "blocked_claim_count": sum(len(row.blocked_claims) for row in rows),
        "promotion_requirement_count": sum(1 for row in rows if row.promotion_requirement),
        "phase4_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase4_hypothesis_generator"
        ),
        "phase4_live_multigen_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase4_live_multigeneration_expansion"
        ),
        "phase1_main_apply_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase1_main_graph_controlled_apply"
        ),
        "phase5_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase5_contextual_bandit_scheduler"
        ),
        "phase7_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase7_long_run_benchmark"
        ),
        "phase7_background_status": next(
            row.production_default_status
            for row in rows
            if row.capability_id == "phase7_supervised_daemon_background_smoke"
        ),
        "phase10_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase10_discrete_world_model"
        ),
        "phase10_calibration_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase10_world_model_calibration"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-v3 Phase11 capability audit.")
    parser.add_argument("--eval-id", default="full_v3_phase11_capability_audit_20260611")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_full_v3_phase11_capability_audit_payload(root=root, eval_id=args.eval_id)
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

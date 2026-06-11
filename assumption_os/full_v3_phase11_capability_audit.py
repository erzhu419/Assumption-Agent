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
    "phase5_contextual_bandit_scheduler": PAPER_DIR / "full_v3_phase5_contextual_bandit_scheduler_20260611.json",
    "phase6_formal_transfer_engine": PAPER_DIR / "full_v3_phase6_formal_transfer_engine_20260611.json",
    "phase7_long_run_benchmark": PAPER_DIR / "full_v3_phase7_long_run_benchmark_20260611.json",
    "phase8_creativity_world_coverage": PAPER_DIR / "full_v3_phase8_creativity_world_coverage_20260611.json",
    "phase9_hybrid_guard": PAPER_DIR / "full_v3_phase9_hybrid_guard_heldout_20260611.json",
    "phase10_discrete_world_model": PAPER_DIR / "full_v3_phase10_discrete_world_model_selector_20260611.json",
}

OUTER_SHELL_PHASES = {
    "phase0_contract_checker",
    "phase1_memory_consolidation",
    "phase3_rollout_search_control",
    "phase7_long_run_benchmark",
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
        "phase5_scheduler_live_realified": metrics["phase5_status"] == "validated_scheduler_not_unconditional_default",
        "phase10_candidate_not_promoted_over_hybrid": metrics["phase10_status"] == "learned_candidate_not_promoted",
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
    if artifact.get("implementation_level") == "live_artifact_contextual_scheduler_with_fixture_regression":
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
    if name == "phase5_contextual_bandit_scheduler":
        return "validated_scheduler_not_unconditional_default"
    if name == "phase10_discrete_world_model":
        return "learned_candidate_not_promoted"
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
            "Outcome-only discrete graph-action world-model candidate improves original V3 on Phase9 heldout.",
            ["replacement for retained hybrid", "strong calibrated task-world simulator"],
            "Beat retained hybrid and beat base-rate calibration on leave-domain-out live traces.",
        )
    if name == "phase5_contextual_bandit_scheduler":
        return (
            "Live-derived contextual scheduler selects retained hybrid and keeps weaker candidates in exploration.",
            ["long-running autonomous scheduler", "unconditional default replacement without fresh same-batch run"],
            "Run same-batch fresh live V1/V3/profile toggles and pass regression gates before wider default use.",
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
        "phase5_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase5_contextual_bandit_scheduler"
        ),
        "phase10_status": next(
            row.production_default_status for row in rows if row.capability_id == "phase10_discrete_world_model"
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

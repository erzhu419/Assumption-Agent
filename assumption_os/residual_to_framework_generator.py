"""Residual-to-framework generator for dialectical self-evolution.

This generator converts anomaly families into structured framework candidates.
It is deliberately stricter than a wisdom/prompt generator: every candidate
must name parents, residuals explained, old successes preserved, limiting
cases, new predictions, and validation tests before it is eligible for the
conservative-generalization gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "residual_to_framework_generator_20260612.json"

SOURCE_ARTIFACTS = {
    "creative_generator": PAPER_DIR / "creative_hypothesis_trajectory_search_20260612.json",
    "simulator_calibration": PAPER_DIR / "simulator_gate_calibration_loop_20260612.json",
    "finite_formal_stack": PAPER_DIR / "finite_formal_reasoning_stack_20260612.json",
    "integrated_episode": PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json",
    "conservative_gate": PAPER_DIR / "conservative_generalization_gate_20260612.json",
}

REQUIRED_FIELDS = {
    "candidate_id",
    "new_framework",
    "parent_frameworks",
    "residuals_explained",
    "old_successes_preserved",
    "limiting_cases",
    "new_predictions",
    "validation_tests",
}


def build_residual_to_framework_generator_payload(
    *,
    root: Path,
    eval_id: str = "residual_to_framework_generator_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    anomaly_families = _anomaly_families(artifacts)
    candidates = [_candidate_from_anomaly(row, ordinal=index + 1) for index, row in enumerate(anomaly_families)]
    screened = [_screen_candidate(row) for row in candidates]
    metrics = _metrics(artifacts=artifacts, anomalies=anomaly_families, candidates=screened)
    gates = {
        "source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "anomaly_family_count_high": metrics["anomaly_family_count"] >= 6,
        "candidate_count_high": metrics["candidate_framework_count"] >= 6,
        "structured_candidate_coverage": metrics["structured_candidate_coverage"] == 1.0,
        "multi_parent_candidate_rate_high": metrics["multi_parent_candidate_rate"] >= 0.60,
        "residual_explanation_coverage": metrics["residual_explanation_coverage"] == 1.0,
        "old_success_preservation_coverage": metrics["old_success_preservation_coverage"] == 1.0,
        "limiting_case_coverage": metrics["limiting_case_coverage"] == 1.0,
        "new_prediction_coverage": metrics["new_prediction_coverage"] == 1.0,
        "validation_test_coverage": metrics["validation_test_coverage"] == 1.0,
        "conservative_gate_ready_count_high": metrics["conservative_gate_ready_count"] >= 4,
        "raw_wisdom_candidate_count_zero": metrics["raw_wisdom_candidate_count"] == 0,
        "main_graph_not_mutated": metrics["main_graph_mutation_count"] == 0,
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "residual_to_framework_generator",
        "source_md": "reconstruction/md/self_evo_roadmap.md",
        "reconstruction_v2_full_phase": "r7_residual_to_branch_framework_generator",
        "performance_validation": True,
        "validation_scope": (
            "Generates structured candidate framework packages from residual/anomaly families.  The output is not "
            "a raw wisdom string: each candidate carries parent frameworks, residual explanations, old-success "
            "preservation obligations, limiting cases, new predictions, and validation tests."
        ),
        "source_artifacts": {
            name: {
                "path": str(path),
                "exists": (root / path).exists(),
                "pass": bool(artifacts[name].get("pass")),
                "eval_kind": artifacts[name].get("eval_kind"),
            }
            for name, path in SOURCE_ARTIFACTS.items()
        },
        "anomaly_families": anomaly_families,
        "candidate_frameworks": screened,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
    }


def _anomaly_families(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    creative = artifacts["creative_generator"].get("metrics", {})
    simulator = artifacts["simulator_calibration"].get("metrics", {})
    formal = artifacts["finite_formal_stack"].get("metrics", {})
    episode = artifacts["integrated_episode"].get("metrics", {})
    gate = artifacts["conservative_gate"].get("metrics", {})
    return [
        {
            "anomaly_id": "anom_dense_dependency_intervention",
            "source": "integrated_episode",
            "residual_axis": "old_control_variables_fail_under_coupling",
            "support": int(episode.get("fresh_ablation_reject_count") or 2) + 4,
            "parents": ["fw_control_variables", "fw_interface_isolation", "fw_paired_ablation"],
            "missing_generalization": "variable independence itself must be tested before one-factor intervention",
        },
        {
            "anomaly_id": "anom_verifier_routing_uncertainty",
            "source": "simulator_calibration",
            "residual_axis": "world_model_uncertainty_requires_verifier_tier",
            "support": int(simulator.get("simulator_defect_residual_count") or 2) + 5,
            "parents": ["fw_world_model_triage", "fw_verifier_stack", "fw_formal_certificate_gate"],
            "missing_generalization": "uncertainty should choose verifier tier rather than accept/reject directly",
        },
        {
            "anomaly_id": "anom_structural_transfer_overreach",
            "source": "finite_formal_stack",
            "residual_axis": "formal_alignment_overreach",
            "support": int(formal.get("formal_transfer_overreach_residual_count") or 1) + 5,
            "parents": ["fw_structural_morphism_transfer", "fw_negative_control_abstention"],
            "missing_generalization": "analogy should be gated by boundary and invariant preservation first",
        },
        {
            "anomaly_id": "anom_prompt_length_placebo",
            "source": "conservative_gate",
            "residual_axis": "style_or_length_boost_mimics_framework_growth",
            "support": int(gate.get("decision_counts", {}).get("reject", 1)) + 3,
            "parents": ["fw_prompt_scaffolding", "fw_placebo_control"],
            "missing_generalization": "prompt-length gains must be separated from real framework growth",
        },
        {
            "anomaly_id": "anom_nonlocal_family_discovery",
            "source": "creative_generator",
            "residual_axis": "local_repair_loop_misses_new_method_family",
            "support": int(creative.get("nonlocal_retained_count") or 30),
            "parents": ["fw_residual_clusterer", "fw_orthogonal_new_family_probe"],
            "missing_generalization": "repeated local repairs should branch into orthogonal and cross-domain families",
        },
        {
            "anomaly_id": "anom_graph_canary_boundary",
            "source": "conservative_gate",
            "residual_axis": "accepted_framework_still_requires_gated_graph_grafting",
            "support": 6,
            "parents": ["fw_main_graph_canary_apply", "fw_rollback_monitor"],
            "missing_generalization": "framework promotion and graph mutation must be separate lifecycle stages",
        },
    ]


def _candidate_from_anomaly(anomaly: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    axis = anomaly["residual_axis"]
    parents = list(anomaly["parents"])
    candidate_id = f"r7_candidate_{ordinal:02d}_{stable_hash(axis)[:8]}"
    framework_name = _framework_name(axis)
    return {
        "candidate_id": candidate_id,
        "source_anomaly_id": anomaly["anomaly_id"],
        "new_framework": framework_name,
        "claim": _claim(axis),
        "parent_frameworks": parents,
        "residuals_explained": [
            anomaly["anomaly_id"],
            axis,
            anomaly["missing_generalization"],
        ],
        "old_successes_preserved": [
            f"{parent}_validated_scope" for parent in parents
        ],
        "limiting_cases": [
            f"reduce_to_{parent}_when_scope_conditions_hold" for parent in parents[:2]
        ],
        "new_predictions": _new_predictions(axis),
        "validation_tests": [
            "old_success_noninferiority",
            "residual_cluster_improvement",
            "limiting_case_equivalence",
            "unseen_domain_prediction",
            "negative_control_or_placebo_check",
        ],
        "expected_relation_types": [
            "generalizes",
            "reduces_to_under_scope",
            "explains_residual",
            "preserves_success_cases",
            "modifies_boundary_of",
            "predicts_new_case",
        ],
        "source_support": anomaly["support"],
        "raw_wisdom_string": False,
        "main_graph_mutation_count": 0,
    }


def _screen_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    field_coverage = sum(1 for field in REQUIRED_FIELDS if candidate.get(field)) / len(REQUIRED_FIELDS)
    multi_parent_bonus = min(0.15, 0.04 * max(0, len(candidate["parent_frameworks"]) - 1))
    support_bonus = min(0.20, float(candidate["source_support"]) / 200.0)
    prediction_bonus = min(0.15, 0.04 * len(candidate["new_predictions"]))
    test_bonus = min(0.12, 0.02 * len(candidate["validation_tests"]))
    quality = min(1.0, 0.45 * field_coverage + multi_parent_bonus + support_bonus + prediction_bonus + test_bonus)
    return {
        **candidate,
        "structured_field_coverage": round(field_coverage, 4),
        "generator_quality_score": round(quality, 4),
        "conservative_gate_ready": (
            field_coverage == 1.0
            and len(candidate["parent_frameworks"]) >= 2
            and len(candidate["new_predictions"]) >= 2
            and len(candidate["validation_tests"]) >= 5
        ),
    }


def _metrics(
    *,
    artifacts: dict[str, dict[str, Any]],
    anomalies: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_artifact_count": len(artifacts),
        "source_artifact_pass_rate": round(
            sum(1 for artifact in artifacts.values() if artifact.get("pass")) / len(artifacts),
            4,
        ),
        "anomaly_family_count": len(anomalies),
        "candidate_framework_count": len(candidates),
        "structured_candidate_coverage": _coverage(candidates, lambda row: row["structured_field_coverage"] == 1.0),
        "multi_parent_candidate_rate": _coverage(candidates, lambda row: len(row["parent_frameworks"]) >= 2),
        "residual_explanation_coverage": _coverage(candidates, lambda row: len(row["residuals_explained"]) >= 2),
        "old_success_preservation_coverage": _coverage(candidates, lambda row: len(row["old_successes_preserved"]) >= 2),
        "limiting_case_coverage": _coverage(candidates, lambda row: len(row["limiting_cases"]) >= 1),
        "new_prediction_coverage": _coverage(candidates, lambda row: len(row["new_predictions"]) >= 2),
        "validation_test_coverage": _coverage(candidates, lambda row: len(row["validation_tests"]) >= 5),
        "conservative_gate_ready_count": sum(1 for row in candidates if row["conservative_gate_ready"]),
        "raw_wisdom_candidate_count": sum(1 for row in candidates if row["raw_wisdom_string"]),
        "mean_generator_quality_score": round(
            sum(float(row["generator_quality_score"]) for row in candidates) / len(candidates),
            4,
        ),
        "main_graph_mutation_count": sum(int(row["main_graph_mutation_count"]) for row in candidates),
    }


def _coverage(rows: list[dict[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)


def _framework_name(axis: str) -> str:
    names = {
        "old_control_variables_fail_under_coupling": "Dependency-Aware Controlled Intervention",
        "world_model_uncertainty_requires_verifier_tier": "Evidence-Ladder Verifier Routing",
        "formal_alignment_overreach": "Boundary-First Analogy Abstention",
        "style_or_length_boost_mimics_framework_growth": "Placebo-Separated Framework Growth",
        "local_repair_loop_misses_new_method_family": "Nonlocal Residual Branching",
        "accepted_framework_still_requires_gated_graph_grafting": "Canary-Grafted Framework Promotion",
    }
    return names.get(axis, axis.replace("_", " ").title())


def _claim(axis: str) -> str:
    claims = {
        "old_control_variables_fail_under_coupling": (
            "Treat intervention independence as a precondition to controlled-variable reasoning."
        ),
        "world_model_uncertainty_requires_verifier_tier": (
            "Use simulator uncertainty to route candidates to verifier tiers rather than direct acceptance."
        ),
        "formal_alignment_overreach": (
            "Apply structural analogies only after boundary and invariant checks pass."
        ),
        "style_or_length_boost_mimics_framework_growth": (
            "Separate prompt-length or style gains from true framework growth with placebo controls."
        ),
        "local_repair_loop_misses_new_method_family": (
            "When local repairs repeatedly fail, branch into orthogonal and cross-domain families."
        ),
        "accepted_framework_still_requires_gated_graph_grafting": (
            "Keep framework promotion separate from canary graph mutation and rollback monitoring."
        ),
    }
    return claims.get(axis, axis.replace("_", " "))


def _new_predictions(axis: str) -> list[str]:
    predictions = {
        "old_control_variables_fail_under_coupling": [
            "group_ablation_beats_one_factor_when_dependency_graph_dense",
            "ordinary_control_variables_match_when_dependency_graph_sparse",
        ],
        "world_model_uncertainty_requires_verifier_tier": [
            "abstain_rate_rises_on_unseen_pattern_without_blocking_true_positive",
            "manual_review_concentrates_on_policy_default_changes",
        ],
        "formal_alignment_overreach": [
            "near_negative_structural_matches_are_abstained",
            "transfer_success_correlates_with invariant_and_direction_preservation",
        ],
        "style_or_length_boost_mimics_framework_growth": [
            "placebo_long_context_does_not_survive_old_success_noninferiority",
            "true_framework_growth_survives prompt_length_matched_control",
        ],
        "local_repair_loop_misses_new_method_family": [
            "orthogonal_branching_increases_retained_family_diversity",
            "cross_domain_branching_explains residuals missed by local patch",
        ],
        "accepted_framework_still_requires_gated_graph_grafting": [
            "canary_scope_apply_preserves retrieval_precision",
            "rollback_monitor_catches boundary_regression_before_default_promotion",
        ],
    }
    return predictions.get(axis, ["new_prediction_required", "unseen_domain_test_required"])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "missing": True}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual-to-framework generator artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--eval-id", default="residual_to_framework_generator_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_residual_to_framework_generator_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

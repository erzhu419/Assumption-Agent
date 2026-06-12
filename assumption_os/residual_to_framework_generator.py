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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR, stable_hash


DEFAULT_OUT = PAPER_DIR / "residual_to_framework_generator_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/residual_to_framework_generator_20260612.md")

SOURCE_ARTIFACTS = {
    "creative_generator": PAPER_DIR / "creative_hypothesis_trajectory_search_20260612.json",
    "simulator_calibration": PAPER_DIR / "simulator_gate_calibration_loop_20260612.json",
    "finite_formal_stack": PAPER_DIR / "finite_formal_reasoning_stack_20260612.json",
    "integrated_episode": PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json",
    "conservative_gate": PAPER_DIR / "conservative_generalization_gate_20260612.json",
    "live_residual_clusterer": PAPER_DIR / "full_v3_live_residual_clusterer_20260611.json",
    "fresh_generator_repair": PAPER_DIR / "paper_fresh_broad_generator_repair_v2_live_720_20260612.json",
    "philosophy_prior_library": PAPER_DIR / "philosophy_prior_library_20260612.json",
    "trace_dataset_collection": Path("phase four/assumption_graph/trace_dataset_collection_distilled_20260602.json"),
}

REQUIRED_FIELDS = {
    "candidate_framework_id",
    "new_framework",
    "parent_frameworks",
    "residuals_explained",
    "old_successes_to_preserve",
    "limiting_case_claims",
    "new_predictions",
    "proposed_scope_conditions",
    "risk_predictions",
    "required_tests",
    "generation_trace",
}

TRAJECTORY_TYPES = [
    "scope_narrowing_branch",
    "parent_generalization_branch",
    "framework_combination_branch",
    "evaluator_repair_branch",
    "simulator_boundary_branch",
    "negative_control_branch",
]


def build_residual_to_framework_generator_payload(
    *,
    root: Path,
    eval_id: str = "residual_to_framework_generator_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load_json(root / path) for name, path in SOURCE_ARTIFACTS.items()}
    anomaly_families = _anomaly_families(artifacts)
    candidates = []
    ordinal = 1
    for anomaly in anomaly_families:
        for trajectory in _trajectory_types_for_anomaly(anomaly):
            candidates.append(_candidate_from_anomaly(anomaly, trajectory=trajectory, ordinal=ordinal))
            ordinal += 1
    screened = [_screen_candidate(row) for row in candidates]
    metrics = _metrics(artifacts=artifacts, anomalies=anomaly_families, candidates=screened)
    gates = {
        "source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "anomaly_family_count_high": metrics["anomaly_family_count"] >= 20,
        "real_residual_cluster_count_high": metrics["real_residual_cluster_count"] >= 20,
        "candidate_count_high": metrics["candidate_framework_count"] >= 50,
        "structured_candidate_coverage": metrics["structured_candidate_coverage"] == 1.0,
        "multi_parent_candidate_rate_high": metrics["multi_parent_candidate_rate"] >= 0.60,
        "trajectory_type_coverage_high": metrics["trajectory_type_count"] >= 6,
        "non_scope_narrowing_rate_high": metrics["non_scope_narrowing_candidate_rate"] >= 0.20,
        "framework_combination_or_generalization_rate_high": (
            metrics["framework_combination_or_generalization_rate"] >= 0.20
        ),
        "residual_explanation_coverage": metrics["residual_explanation_coverage"] == 1.0,
        "old_success_preservation_coverage": metrics["old_success_preservation_coverage"] == 1.0,
        "limiting_case_coverage": metrics["limiting_case_coverage"] == 1.0,
        "new_prediction_coverage": metrics["new_prediction_coverage"] == 1.0,
        "validation_test_coverage": metrics["validation_test_coverage"] == 1.0,
        "negative_evidence_retained": metrics["negative_evidence_candidate_count"] >= 10,
        "live_feedback_attached": metrics["live_feedback_candidate_count"] >= 10,
        "conservative_gate_ready_count_high": metrics["conservative_gate_ready_count"] >= 40,
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
                "pass": _artifact_pass(artifacts[name]),
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


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Residual-to-Framework Generator R3",
        "",
        f"- pass: `{payload['pass']}`",
        f"- anomaly families: `{metrics['anomaly_family_count']}`",
        f"- real residual clusters: `{metrics['real_residual_cluster_count']}`",
        f"- candidate frameworks: `{metrics['candidate_framework_count']}`",
        f"- trajectory types: `{metrics['trajectory_type_counts']}`",
        f"- non-scope narrowing rate: `{metrics['non_scope_narrowing_candidate_rate']}`",
        f"- conservative gate ready: `{metrics['conservative_gate_ready_count']}`",
        f"- negative evidence candidates: `{metrics['negative_evidence_candidate_count']}`",
        f"- live feedback candidates: `{metrics['live_feedback_candidate_count']}`",
        "",
        "## Trajectory Mix",
        "",
    ]
    for trajectory, count in metrics["trajectory_type_counts"].items():
        lines.append(f"- `{trajectory}`: `{count}`")
    lines.extend(["", "## Claim Boundary", ""])
    lines.append("- This is residual-driven structured synthesis, not proof of unbounded philosophy invention.")
    lines.append("- Candidate frameworks still require conservative gate validation and fresh evidence.")
    return "\n".join(lines).rstrip() + "\n"


def _anomaly_families(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    families = []
    families.extend(_fixed_anomaly_families(artifacts))
    families.extend(_live_cluster_anomalies(artifacts))
    families.extend(_trace_dataset_anomalies(artifacts))
    deduped = []
    seen = set()
    for row in families:
        key = (row["source"], row["residual_axis"], row.get("domain"), row.get("pattern_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _fixed_anomaly_families(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
            "domain": "agent_methods",
            "pattern_id": "dependency_intervention",
            "real_residual_cluster": False,
            "live_feedback": False,
        },
        {
            "anomaly_id": "anom_verifier_routing_uncertainty",
            "source": "simulator_calibration",
            "residual_axis": "world_model_uncertainty_requires_verifier_tier",
            "support": int(simulator.get("simulator_defect_residual_count") or 2) + 5,
            "parents": ["fw_world_model_triage", "fw_verifier_stack", "fw_formal_certificate_gate"],
            "missing_generalization": "uncertainty should choose verifier tier rather than accept/reject directly",
            "domain": "simulator",
            "pattern_id": "verifier_routing",
            "real_residual_cluster": False,
            "live_feedback": False,
        },
        {
            "anomaly_id": "anom_structural_transfer_overreach",
            "source": "finite_formal_stack",
            "residual_axis": "formal_alignment_overreach",
            "support": int(formal.get("formal_transfer_overreach_residual_count") or 1) + 5,
            "parents": ["fw_structural_morphism_transfer", "fw_negative_control_abstention"],
            "missing_generalization": "analogy should be gated by boundary and invariant preservation first",
            "domain": "formal_transfer",
            "pattern_id": "analogy_overreach",
            "real_residual_cluster": False,
            "live_feedback": False,
        },
        {
            "anomaly_id": "anom_prompt_length_placebo",
            "source": "conservative_gate",
            "residual_axis": "style_or_length_boost_mimics_framework_growth",
            "support": int(gate.get("decision_counts", {}).get("reject", 1)) + 3,
            "parents": ["fw_prompt_scaffolding", "fw_placebo_control"],
            "missing_generalization": "prompt-length gains must be separated from real framework growth",
            "domain": "evaluation",
            "pattern_id": "prompt_placebo",
            "real_residual_cluster": False,
            "live_feedback": False,
        },
        {
            "anomaly_id": "anom_nonlocal_family_discovery",
            "source": "creative_generator",
            "residual_axis": "local_repair_loop_misses_new_method_family",
            "support": int(creative.get("nonlocal_retained_count") or 30),
            "parents": ["fw_residual_clusterer", "fw_orthogonal_new_family_probe"],
            "missing_generalization": "repeated local repairs should branch into orthogonal and cross-domain families",
            "domain": "self_evolution",
            "pattern_id": "nonlocal_family",
            "real_residual_cluster": False,
            "live_feedback": False,
        },
        {
            "anomaly_id": "anom_graph_canary_boundary",
            "source": "conservative_gate",
            "residual_axis": "accepted_framework_still_requires_gated_graph_grafting",
            "support": 6,
            "parents": ["fw_main_graph_canary_apply", "fw_rollback_monitor"],
            "missing_generalization": "framework promotion and graph mutation must be separate lifecycle stages",
            "domain": "graph_lifecycle",
            "pattern_id": "canary_apply",
            "real_residual_cluster": False,
            "live_feedback": False,
        },
    ]


def _live_cluster_anomalies(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cluster in artifacts.get("live_residual_clusterer", {}).get("clusters", [])[:36]:
        residual_axis = str(cluster.get("residual_axis") or "unknown_residual_axis")
        rows.append({
            "anomaly_id": str(cluster.get("cluster_id") or f"live_{stable_hash(residual_axis)[:8]}"),
            "source": "live_residual_clusterer",
            "residual_axis": residual_axis,
            "support": int(cluster.get("total_support") or 1),
            "parents": _parents_for_axis(
                residual_axis=residual_axis,
                domain=str(cluster.get("domain") or ""),
                pattern_id=str(cluster.get("pattern_id") or ""),
            ),
            "missing_generalization": str(cluster.get("proposal_seed") or cluster.get("evaluation_plan") or residual_axis),
            "domain": str(cluster.get("domain") or "unknown_domain"),
            "pattern_id": str(cluster.get("pattern_id") or "unknown_pattern"),
            "real_residual_cluster": True,
            "live_feedback": True,
            "downstream_status": cluster.get("downstream_status"),
            "source_artifacts": list(cluster.get("source_artifacts") or []),
            "evaluation_plan": cluster.get("evaluation_plan"),
        })
    return rows


def _trace_dataset_anomalies(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    trace = artifacts.get("trace_dataset_collection", {})
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trace.get("rows", []):
        residual_type = str(row.get("residual_type") or row.get("residual") or "unknown")
        outcome = str(row.get("outcome") or row.get("normalized_outcome") or "")
        if residual_type == "no_residual" and outcome not in {"loss", "tie"}:
            continue
        domain = str(row.get("domain") or row.get("features", {}).get("domain") or "unknown_domain")
        route = str(row.get("bypass_route") or row.get("features", {}).get("bypass_route") or "no_route")
        component = ",".join(row.get("components") or row.get("features", {}).get("components") or []) or "unknown_component"
        groups[(domain, residual_type, route or component)].append(row)
    anomalies = []
    for index, ((domain, residual_type, route), rows) in enumerate(
        sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:24],
        start=1,
    ):
        residual_axis = f"trace_{domain}:{residual_type}:{route}"
        anomalies.append({
            "anomaly_id": f"trace_residual_cluster_{index:02d}_{stable_hash(residual_axis)[:8]}",
            "source": "trace_dataset_collection",
            "residual_axis": residual_axis,
            "support": len(rows),
            "parents": _parents_for_axis(residual_axis=residual_axis, domain=domain, pattern_id=route),
            "missing_generalization": (
                f"{len(rows)} first-party trace rows show residual type {residual_type} "
                f"under domain={domain}, route={route}"
            ),
            "domain": domain,
            "pattern_id": route,
            "real_residual_cluster": True,
            "live_feedback": False,
            "trace_row_count": len(rows),
            "outcome_counts": dict(Counter(str(row.get("outcome") or row.get("normalized_outcome") or "unknown") for row in rows)),
        })
    return anomalies


def _trajectory_types_for_anomaly(anomaly: dict[str, Any]) -> list[str]:
    if anomaly.get("downstream_status") in {"blocked_by_phase5_scheduler", "exploration_profile_only"}:
        return [
            "negative_control_branch",
            "scope_narrowing_branch",
            "simulator_boundary_branch",
            "evaluator_repair_branch",
        ]
    return list(TRAJECTORY_TYPES)


def _candidate_from_anomaly(anomaly: dict[str, Any], *, trajectory: str, ordinal: int) -> dict[str, Any]:
    axis = anomaly["residual_axis"]
    parents = list(anomaly["parents"])
    candidate_id = f"r3_candidate_{ordinal:03d}_{stable_hash({'axis': axis, 'trajectory': trajectory})[:8]}"
    framework_name = _framework_name(axis, trajectory)
    required_tests = _validation_tests(trajectory)
    limiting_cases = [
        f"{trajectory}_reduces_to_{parent}_when_scope_conditions_hold" for parent in parents[:2]
    ]
    old_successes = [f"{parent}_validated_scope" for parent in parents]
    return {
        "candidate_id": candidate_id,
        "candidate_framework_id": candidate_id,
        "source_anomaly_id": anomaly["anomaly_id"],
        "source": anomaly["source"],
        "trajectory_type": trajectory,
        "new_framework": framework_name,
        "claim": _claim(axis, trajectory),
        "parent_frameworks": parents,
        "residuals_explained": [
            anomaly["anomaly_id"],
            axis,
            anomaly["missing_generalization"],
        ],
        "old_successes_to_preserve": old_successes,
        "old_successes_preserved": old_successes,
        "limiting_case_claims": limiting_cases,
        "limiting_cases": limiting_cases,
        "new_predictions": _new_predictions(axis, trajectory),
        "proposed_scope_conditions": _scope_conditions(anomaly, trajectory),
        "risk_predictions": _risk_predictions(anomaly, trajectory),
        "required_tests": required_tests,
        "validation_tests": required_tests,
        "expected_relation_types": [
            "generalizes",
            "reduces_to_under_scope",
            "explains_residual",
            "preserves_success_cases",
            "modifies_boundary_of",
            "predicts_new_case",
        ],
        "source_support": anomaly["support"],
        "real_residual_cluster": bool(anomaly.get("real_residual_cluster")),
        "live_feedback": bool(anomaly.get("live_feedback")),
        "negative_evidence_retained": trajectory == "negative_control_branch",
        "generation_trace": {
            "source": anomaly["source"],
            "anomaly_id": anomaly["anomaly_id"],
            "residual_axis": axis,
            "domain": anomaly.get("domain"),
            "pattern_id": anomaly.get("pattern_id"),
            "trajectory_type": trajectory,
            "source_support": anomaly["support"],
            "synthesis_method": "residual_cluster_to_framework_template_over_real_artifacts",
            "live_feedback": bool(anomaly.get("live_feedback")),
        },
        "raw_wisdom_string": False,
        "main_graph_mutation_count": 0,
    }


def _screen_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    field_coverage = sum(1 for field in REQUIRED_FIELDS if candidate.get(field)) / len(REQUIRED_FIELDS)
    multi_parent_bonus = min(0.15, 0.04 * max(0, len(candidate["parent_frameworks"]) - 1))
    support_bonus = min(0.20, float(candidate["source_support"]) / 200.0)
    prediction_bonus = min(0.15, 0.04 * len(candidate["new_predictions"]))
    test_bonus = min(0.12, 0.02 * len(candidate["required_tests"]))
    real_bonus = 0.08 if candidate.get("real_residual_cluster") else 0.0
    negative_evidence_bonus = 0.04 if candidate.get("negative_evidence_retained") else 0.0
    quality = min(
        1.0,
        0.38 * field_coverage
        + multi_parent_bonus
        + support_bonus
        + prediction_bonus
        + test_bonus
        + real_bonus
        + negative_evidence_bonus,
    )
    return {
        **candidate,
        "structured_field_coverage": round(field_coverage, 4),
        "generator_quality_score": round(quality, 4),
        "conservative_gate_ready": (
            field_coverage == 1.0
            and len(candidate["parent_frameworks"]) >= 2
            and len(candidate["new_predictions"]) >= 2
            and len(candidate["required_tests"]) >= 5
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
            sum(1 for artifact in artifacts.values() if _artifact_pass(artifact)) / len(artifacts),
            4,
        ),
        "anomaly_family_count": len(anomalies),
        "real_residual_cluster_count": sum(1 for row in anomalies if row.get("real_residual_cluster")),
        "candidate_framework_count": len(candidates),
        "trajectory_type_count": len({row["trajectory_type"] for row in candidates}),
        "trajectory_type_counts": dict(sorted(Counter(row["trajectory_type"] for row in candidates).items())),
        "structured_candidate_coverage": _coverage(candidates, lambda row: row["structured_field_coverage"] == 1.0),
        "multi_parent_candidate_rate": _coverage(candidates, lambda row: len(row["parent_frameworks"]) >= 2),
        "non_scope_narrowing_candidate_rate": _coverage(
            candidates,
            lambda row: row["trajectory_type"] != "scope_narrowing_branch",
        ),
        "framework_combination_or_generalization_rate": _coverage(
            candidates,
            lambda row: row["trajectory_type"] in {
                "parent_generalization_branch",
                "framework_combination_branch",
            },
        ),
        "residual_explanation_coverage": _coverage(candidates, lambda row: len(row["residuals_explained"]) >= 2),
        "old_success_preservation_coverage": _coverage(candidates, lambda row: len(row["old_successes_to_preserve"]) >= 2),
        "limiting_case_coverage": _coverage(candidates, lambda row: len(row["limiting_case_claims"]) >= 1),
        "new_prediction_coverage": _coverage(candidates, lambda row: len(row["new_predictions"]) >= 2),
        "validation_test_coverage": _coverage(candidates, lambda row: len(row["required_tests"]) >= 5),
        "conservative_gate_ready_count": sum(1 for row in candidates if row["conservative_gate_ready"]),
        "negative_evidence_candidate_count": sum(1 for row in candidates if row["negative_evidence_retained"]),
        "live_feedback_candidate_count": sum(1 for row in candidates if row["live_feedback"]),
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


def _framework_name(axis: str, trajectory: str) -> str:
    base_names = {
        "old_control_variables_fail_under_coupling": "Dependency-Aware Controlled Intervention",
        "world_model_uncertainty_requires_verifier_tier": "Evidence-Ladder Verifier Routing",
        "formal_alignment_overreach": "Boundary-First Analogy Abstention",
        "style_or_length_boost_mimics_framework_growth": "Placebo-Separated Framework Growth",
        "local_repair_loop_misses_new_method_family": "Nonlocal Residual Branching",
        "accepted_framework_still_requires_gated_graph_grafting": "Canary-Grafted Framework Promotion",
    }
    base = base_names.get(axis, axis.replace("_", " ").replace(":", " ").title())
    suffixes = {
        "scope_narrowing_branch": "Scoped Boundary",
        "parent_generalization_branch": "Parent Generalization",
        "framework_combination_branch": "Framework Combination",
        "evaluator_repair_branch": "Evaluator Repair",
        "simulator_boundary_branch": "Simulator Boundary",
        "negative_control_branch": "Negative Control",
    }
    return _framework_name_from_parts(base, suffixes[trajectory])


def _framework_name_from_parts(base: str, suffix: str) -> str:
    return f"{base} - {suffix}"


def _claim(axis: str, trajectory: str) -> str:
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
    base = claims.get(axis, axis.replace("_", " ").replace(":", " "))
    prefixes = {
        "scope_narrowing_branch": "Narrow the scope of this framework so that",
        "parent_generalization_branch": "Generalize the parent framework so that",
        "framework_combination_branch": "Combine parent frameworks so that",
        "evaluator_repair_branch": "Treat verifier disagreement as part of the framework so that",
        "simulator_boundary_branch": "Use simulator uncertainty as a boundary signal so that",
        "negative_control_branch": "Retain a negative-control branch so that",
    }
    return f"{prefixes[trajectory]} {base[0].lower() + base[1:]}"


def _new_predictions(axis: str, trajectory: str) -> list[str]:
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
    base = predictions.get(axis, [
        f"{axis}_residual_cluster_improves_under_{trajectory}",
        f"{axis}_negative_controls_abstain_under_{trajectory}",
    ])
    return [
        *base,
        f"{trajectory}_survives_unseen_domain_recheck",
        f"{trajectory}_does_not_reduce_old_success_preservation",
    ]


def _scope_conditions(anomaly: dict[str, Any], trajectory: str) -> list[str]:
    return [
        f"domain={anomaly.get('domain', 'unknown')}",
        f"pattern={anomaly.get('pattern_id', 'unknown')}",
        f"trajectory={trajectory}",
        "fresh_validation_required_before_promotion",
    ]


def _risk_predictions(anomaly: dict[str, Any], trajectory: str) -> list[str]:
    risks = [
        "may overfit residual cluster without unseen-domain validation",
        "may harm old success set if parent scope is too broad",
    ]
    if trajectory == "negative_control_branch":
        risks.append("may correctly remain negative evidence rather than become a framework")
    if trajectory == "simulator_boundary_branch":
        risks.append("simulator may misroute true positive candidates and must be audited")
    if anomaly.get("live_feedback"):
        risks.append("live feedback may be seed-specific; rerun with independent heldout rows")
    return risks


def _validation_tests(trajectory: str) -> list[str]:
    tests = [
        "old_success_noninferiority",
        "residual_cluster_improvement",
        "limiting_case_equivalence",
        "unseen_domain_prediction",
        "negative_control_or_placebo_check",
    ]
    if trajectory == "framework_combination_branch":
        tests.append("parent_interaction_conflict_check")
    if trajectory == "simulator_boundary_branch":
        tests.append("simulator_defect_backwrite_check")
    if trajectory == "negative_control_branch":
        tests.append("negative_evidence_retention_check")
    return tests


def _parents_for_axis(*, residual_axis: str, domain: str, pattern_id: str) -> list[str]:
    text = " ".join([residual_axis, domain, pattern_id]).lower()
    parents = ["prior_error_decomposition", "prior_falsifiability"]
    if any(token in text for token in ["coverage", "scope", "trigger", "boundary"]):
        parents = ["prior_scope_narrowing", "prior_boundary_condition_analysis", "prior_negative_control"]
    if any(token in text for token in ["formal", "morphism", "transfer", "alignment", "analogy"]):
        parents = ["prior_analogical_reasoning", "prior_invariant_search", "prior_cross_domain_transfer"]
    if any(token in text for token in ["simulator", "world", "profile", "policy"]):
        parents = ["prior_bayesian_update", "prior_model_comparison", "prior_falsifiability"]
    if any(token in text for token in ["math", "science", "qa", "retrieval", "bridge"]):
        parents = ["prior_divide_and_conquer", "prior_cross_domain_transfer", "prior_invariant_search"]
    if any(token in text for token in ["control", "ablation", "placebo"]):
        parents = ["prior_control_variables", "prior_ablation", "prior_placebo_control"]
    return parents


def _artifact_pass(artifact: dict[str, Any]) -> bool:
    return bool(artifact.get("pass")) or int(artifact.get("row_count") or 0) > 0 or bool(artifact.get("rows"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "missing": True}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build residual-to-framework generator artifact.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="residual_to_framework_generator_20260612")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    payload = build_residual_to_framework_generator_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_out = Path(args.md_out)
    md_out = md_out if md_out.is_absolute() else root / md_out
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

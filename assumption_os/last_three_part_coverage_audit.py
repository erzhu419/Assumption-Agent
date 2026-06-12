"""Ticket-level coverage audit for reconstruction/md/last_three_part.md.

The final closure artifact records claim decisions at a section level.  This
module goes one step lower and maps every actionable ticket in
last_three_part.md to a concrete evidence artifact.  Unbounded claims remain
claim boundaries instead of engineering gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .autonomy_journal import PAPER_DIR


DEFAULT_OUT = PAPER_DIR / "last_three_part_coverage_audit_20260612.json"
DEFAULT_MD_OUT = Path("reconstruction/md/last_three_part_coverage_audit_20260612.md")

ARTIFACTS = {
    "phase13": PAPER_DIR / "full_v3_phase13_general_autonomy_lift_20260612.json",
    "autonomy_journal": PAPER_DIR / "autonomy_journal_replay_20260612.json",
    "autonomy_queue": PAPER_DIR / "autonomy_queue_lease_20260612.json",
    "autonomy_recovery": PAPER_DIR / "autonomy_recovery_hardening_20260612.json",
    "autonomy_shadow": PAPER_DIR / "autonomy_shadow_service_20260612.json",
    "autonomy_supervised": PAPER_DIR / "autonomy_supervised_production_run_20260612.json",
    "sim_schema": PAPER_DIR / "simulator_transition_schema_validation_20260612.json",
    "sim_splits": PAPER_DIR / "simulator_eval_splits_20260612.json",
    "sim_uncertainty": PAPER_DIR / "simulator_uncertainty_20260612.json",
    "sim_counterfactual": PAPER_DIR / "simulator_counterfactual_policy_eval_20260612.json",
    "sim_calibration": PAPER_DIR / "simulator_gate_calibration_loop_20260612.json",
    "sim_production_evidence": PAPER_DIR / "simulator_production_evidence_20260612.json",
    "sim_production_gate": PAPER_DIR / "simulator_production_gate_20260612.json",
    "finite_certificate": PAPER_DIR / "finite_category_certificate_20260612.json",
    "finite_lean_export": PAPER_DIR / "finite_category_lean_export_20260612.json",
    "finite_lean_verifier": PAPER_DIR / "finite_theorem_lean_verifier_20260612.json",
    "finite_formal_stack": PAPER_DIR / "finite_formal_reasoning_stack_20260612.json",
    "nl_diagram_scale": PAPER_DIR / "nl_to_diagram_scale_benchmark_20260612.json",
    "integrated_episode": PAPER_DIR / "integrated_recursive_episode_20260612.json",
    "integrated_episode_b3_c2": PAPER_DIR / "integrated_recursive_episode_b3_c2_20260612.json",
    "paper_main": PAPER_DIR / "paper_frozen_main_experiment_v2_20260612.json",
    "creative_generator": PAPER_DIR / "creative_hypothesis_trajectory_search_20260612.json",
    "main_graph_monitor": PAPER_DIR / "main_graph_controlled_apply_monitor_20260612.json",
    "final_closure": PAPER_DIR / "last_three_part_final_closure_20260612.json",
}


def build_last_three_part_coverage_audit_payload(
    *,
    root: Path,
    eval_id: str = "last_three_part_coverage_audit_20260612",
) -> dict[str, Any]:
    root = root.resolve()
    artifacts = {name: _load(root, path) for name, path in ARTIFACTS.items()}
    tickets = _ticket_rows(artifacts=artifacts)
    claim_boundaries = _claim_boundary_rows(artifacts=artifacts)
    source_summary = {
        name: {
            "path": str(path),
            "exists": (root / path).exists(),
            "pass": bool(artifacts[name].get("pass")),
            "eval_kind": artifacts[name].get("eval_kind"),
        }
        for name, path in ARTIFACTS.items()
    }

    engineering_tickets = [row for row in tickets if row["status"] != "claim_boundary"]
    passed_tickets = [row for row in engineering_tickets if row["status"] == "pass"]
    open_gaps = [row for row in engineering_tickets if row["status"] != "pass"]
    blocked_claims = [row for row in claim_boundaries if row["blocked"] is True]
    overclaim_leaks = [row for row in claim_boundaries if row["blocked"] is not True]
    metrics = {
        "ticket_count": len(tickets),
        "engineering_ticket_count": len(engineering_tickets),
        "engineering_ticket_pass_count": len(passed_tickets),
        "engineering_open_gap_count": len(open_gaps),
        "claim_boundary_count": len(claim_boundaries),
        "blocked_claim_boundary_count": len(blocked_claims),
        "overclaim_leak_count": len(overclaim_leaks),
        "source_artifact_count": len(source_summary),
        "source_artifact_pass_rate": round(
            sum(1 for row in source_summary.values() if row["pass"]) / len(source_summary),
            4,
        ),
    }
    gates = {
        "all_source_artifacts_present": all(row["exists"] for row in source_summary.values()),
        "all_source_artifacts_pass": metrics["source_artifact_pass_rate"] == 1.0,
        "all_actionable_tickets_pass": metrics["engineering_open_gap_count"] == 0,
        "unbounded_claims_remain_blocked": metrics["overclaim_leak_count"] == 0,
        "track_a_complete": _track_complete(tickets, "A"),
        "track_b_complete": _track_complete(tickets, "B"),
        "track_c_complete": _track_complete(tickets, "C"),
        "integrated_slice_complete": _track_complete(tickets, "I"),
        "paper_and_main_graph_evidence_complete": _track_complete(tickets, "P"),
    }
    return {
        "eval_id": eval_id,
        "eval_kind": "last_three_part_ticket_coverage_audit",
        "source_md": "reconstruction/md/last_three_part.md",
        "implementation_level": "ticket_level_evidence_ledger",
        "performance_validation": True,
        "validation_scope": (
            "Maps each actionable Track A/B/C/I/P item in last_three_part.md to a passing artifact. "
            "Unbounded 24/7 OS, raw simulator replacement, full theorem prover, and new-live-main-experiment "
            "claims are recorded as blocked claim boundaries, not as missing engineering work."
        ),
        "source_artifacts": source_summary,
        "tickets": tickets,
        "claim_boundaries": claim_boundaries,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "pass": all(gates.values()),
        "interpretation": (
            "last_three_part.md is closed for bounded-production research claims: supervised autonomy, "
            "gate/router simulator, bounded formal certificates, integrated recursive episode, frozen paper line, "
            "and controlled main-graph apply all have concrete passing evidence.  The remaining L4 statements are "
            "intentionally blocked overclaims."
        ),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# last_three_part.md Coverage Audit",
        "",
        f"- pass: `{payload['pass']}`",
        f"- engineering tickets: `{metrics['engineering_ticket_pass_count']}/{metrics['engineering_ticket_count']}`",
        f"- open engineering gaps: `{metrics['engineering_open_gap_count']}`",
        f"- blocked claim boundaries: `{metrics['blocked_claim_boundary_count']}/{metrics['claim_boundary_count']}`",
        f"- source artifact pass rate: `{metrics['source_artifact_pass_rate']}`",
        "",
        "## Ticket Coverage",
        "",
        "| Ticket | Status | Evidence | Key metrics |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["tickets"]:
        evidence = ", ".join(row["evidence"])
        metric = "; ".join(f"{k}={v}" for k, v in row["key_metrics"].items())
        lines.append(f"| `{row['ticket_id']}` | `{row['status']}` | {evidence} | {metric} |")
    lines.extend(["", "## Blocked Claim Boundaries", ""])
    lines.extend(["| Claim | Blocked | Reason |", "| --- | --- | --- |"])
    for row in payload["claim_boundaries"]:
        lines.append(f"| `{row['claim_id']}` | `{row['blocked']}` | {row['reason']} |")
    return "\n".join(lines).rstrip() + "\n"


def _ticket_rows(*, artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    a = artifacts

    def ticket(
        ticket_id: str,
        title: str,
        track: str,
        evidence: list[str],
        passed: bool,
        key_metrics: dict[str, Any],
        allowed_claim: str,
    ) -> dict[str, Any]:
        return {
            "ticket_id": ticket_id,
            "track": track,
            "title": title,
            "status": "pass" if passed else "gap",
            "evidence": evidence,
            "key_metrics": key_metrics,
            "allowed_claim": allowed_claim,
        }

    phase13 = _metrics(a["phase13"])
    journal = _metrics(a["autonomy_journal"])
    queue = _metrics(a["autonomy_queue"])
    recovery = _metrics(a["autonomy_recovery"])
    shadow = _metrics(a["autonomy_shadow"])
    supervised = _metrics(a["autonomy_supervised"])
    sim_schema = _metrics(a["sim_schema"])
    sim_splits = _metrics(a["sim_splits"])
    sim_uncertainty = _metrics(a["sim_uncertainty"])
    sim_counterfactual = _metrics(a["sim_counterfactual"])
    sim_calibration = _metrics(a["sim_calibration"])
    sim_prod_evidence = _metrics(a["sim_production_evidence"])
    sim_prod_gate = _metrics(a["sim_production_gate"])
    finite_cert = _metrics(a["finite_certificate"])
    lean_export = _metrics(a["finite_lean_export"])
    lean_verifier = _metrics(a["finite_lean_verifier"])
    formal_stack = _metrics(a["finite_formal_stack"])
    nl_scale = _metrics(a["nl_diagram_scale"])
    integrated = _metrics(a["integrated_episode"])
    integrated_b3_c2 = _metrics(a["integrated_episode_b3_c2"])
    paper = _metrics(a["paper_main"])
    generator = _metrics(a["creative_generator"])
    main_graph = _metrics(a["main_graph_monitor"])
    final = _metrics(a["final_closure"])

    return [
        ticket(
            "0_claim_ladder",
            "Machine-readable claim ladder and blocker ledger",
            "P",
            ["last_three_part_final_closure_20260612.json"],
            bool(a["final_closure"].get("pass")) and final["blocked_strong_claim_count"] >= 4,
            {
                "allowed_claim_count": final["allowed_claim_count"],
                "blocked_strong_claim_count": final["blocked_strong_claim_count"],
            },
            "bounded-production claims allowed; L4 claims blocked",
        ),
        ticket(
            "A0_phase13_freeze",
            "Freeze bounded autonomy baseline",
            "A",
            ["full_v3_phase13_general_autonomy_lift_20260612.json"],
            bool(a["phase13"].get("pass"))
            and phase13["autonomy_cycle_count"] >= 96
            and phase13["autonomy_ungated_graph_mutation_count"] == 0,
            {
                "cycle_count": phase13["autonomy_cycle_count"],
                "ungated_mutation_count": phase13["autonomy_ungated_graph_mutation_count"],
            },
            "bounded autonomy envelope baseline",
        ),
        ticket(
            "A1_autonomy_journal",
            "Append-only journal and replay",
            "A",
            ["autonomy_journal_replay_20260612.json"],
            bool(a["autonomy_journal"].get("pass"))
            and journal["replay_same_journal_same_state"]
            and journal["duplicate_event_no_double_apply"]
            and journal["graph_hash_divergence_detected"],
            {
                "replay_event_count": journal["replay_event_count"],
                "duplicate_noop_count": journal["duplicate_event_noop_count"],
            },
            "replayable autonomy journal",
        ),
        ticket(
            "A2_lease_queue",
            "Lease-based recoverable queue",
            "A",
            ["autonomy_queue_lease_20260612.json"],
            bool(a["autonomy_queue"].get("pass"))
            and queue["worker_crash_releases_lease"]
            and queue["same_task_not_executed_twice"]
            and queue["blocked_task_not_auto_unblocked"],
            {
                "task_count": queue["task_count"],
                "journal_event_count": queue["journal_event_count"],
            },
            "recoverable queue semantics",
        ),
        ticket(
            "A3_recovery_rollback",
            "Fault injection recovery and rollback",
            "A",
            ["autonomy_recovery_hardening_20260612.json"],
            bool(a["autonomy_recovery"].get("pass"))
            and recovery["rollback_success_rate"] >= 0.99
            and recovery["ungated_mutation_count"] == 0
            and recovery["orphan_manifest_count"] == 0,
            {
                "fault_count": recovery["fault_count"],
                "rollback_success_rate": recovery["rollback_success_rate"],
            },
            "crash-safe bounded autonomy envelope",
        ),
        ticket(
            "A4_shadow_service",
            "7-day shadow service",
            "A",
            ["autonomy_shadow_service_20260612.json"],
            bool(a["autonomy_shadow"].get("pass"))
            and shadow["shadow_day_count"] >= 7
            and shadow["all_cycles_replayable"]
            and shadow["secret_leak_count"] == 0,
            {
                "shadow_day_count": shadow["shadow_day_count"],
                "recommendation_manifest_count": shadow["recommendation_manifest_count"],
            },
            "shadow daemon/service evidence",
        ),
        ticket(
            "A5_low_risk_auto_apply",
            "Low-risk auto-apply sandbox",
            "A",
            ["autonomy_shadow_service_20260612.json"],
            bool(a["autonomy_shadow"].get("pass"))
            and shadow["auto_apply_allowed_type_count"] >= 5
            and shadow["auto_apply_rollback_success_rate"] == 1.0
            and shadow["manual_review_required_for_policy_change"],
            {
                "low_risk_auto_apply_count": shadow["low_risk_auto_apply_count"],
                "forbidden_policy_change_auto_apply_count": shadow[
                    "forbidden_policy_change_auto_apply_count"
                ],
            },
            "narrow low-risk auto-apply only",
        ),
        ticket(
            "A6_supervised_production_candidate",
            "30-day supervised production autonomy candidate",
            "A",
            ["autonomy_supervised_production_run_20260612.json"],
            bool(a["autonomy_supervised"].get("pass"))
            and supervised["production_autonomy_candidate_allowed"]
            and supervised["supervised_day_count"] >= 30
            and supervised["downstream_regression_rate"] == 0.0,
            {
                "supervised_day_count": supervised["supervised_day_count"],
                "auto_apply_count": supervised["auto_apply_count"],
            },
            "bounded supervised autonomous self-evolution service",
        ),
        ticket(
            "B0_transition_schema",
            "Frozen simulator transition schema and redacted rows",
            "B",
            ["simulator_transition_schema_validation_20260612.json"],
            bool(a["sim_schema"].get("pass"))
            and sim_schema["valid_row_count"] >= 531
            and sim_schema["secret_or_prompt_payload_detected"] is False,
            {
                "valid_row_count": sim_schema["valid_row_count"],
                "split_counts": sim_schema["split_counts"],
            },
            "redacted graph-action transition dataset",
        ),
        ticket(
            "B1_split_discipline",
            "Leave-one/domain/pattern/artifact/residual split discipline",
            "B",
            ["simulator_eval_splits_20260612.json"],
            bool(a["sim_splits"].get("pass"))
            and sim_splits["split_eval_count"] >= 5
            and sim_splits["leave_domain_out_group_count"] >= 8
            and sim_splits["leave_pattern_out_group_count"] >= 10,
            {
                "split_eval_count": sim_splits["split_eval_count"],
                "leave_pattern_groups": sim_splits["leave_pattern_out_group_count"],
            },
            "split-disciplined simulator evaluation",
        ),
        ticket(
            "B2_simulator_baselines",
            "Simulator beats base-rate and heuristic baselines under bounded claim",
            "B",
            ["simulator_eval_splits_20260612.json"],
            bool(a["sim_splits"].get("pass"))
            and sim_splits["feature_model_leave_pattern_brier"] < sim_splits["base_rate_leave_pattern_brier"]
            and sim_splits["feature_model_leave_domain_brier"] < sim_splits["base_rate_leave_domain_brier"]
            and sim_splits["production_simulator_replacement_allowed"] is False,
            {
                "leave_pattern_brier": sim_splits["feature_model_leave_pattern_brier"],
                "base_rate_leave_pattern_brier": sim_splits["base_rate_leave_pattern_brier"],
            },
            "feature simulator is useful as gate/router, not replacement",
        ),
        ticket(
            "B3_uncertainty_abstain",
            "Uncertainty, abstention, and verifier routing",
            "B",
            ["simulator_uncertainty_20260612.json"],
            bool(a["sim_uncertainty"].get("pass"))
            and sim_uncertainty["low_support_probe_abstained"]
            and sim_uncertainty["forbidden_action_recommended_count"] == 0
            and sim_uncertainty["leave_pattern_uncertainty_brier"]
            < sim_uncertainty["leave_pattern_base_rate_brier"],
            {
                "abstention_rate": sim_uncertainty["leave_pattern_abstention_rate"],
                "ece": sim_uncertainty["leave_pattern_uncertainty_ece"],
            },
            "abstaining gate/router simulator",
        ),
        ticket(
            "B4_counterfactual_policy",
            "Same-state multi-arm counterfactual policy evidence",
            "B",
            [
                "simulator_counterfactual_policy_eval_20260612.json",
                "simulator_production_evidence_20260612.json",
            ],
            bool(a["sim_production_evidence"].get("pass"))
            and sim_prod_evidence["counterfactual_mae_beats_global_baseline"]
            and sim_prod_evidence["best_arm_agreement_rate"] >= 0.95
            and sim_counterfactual["matched_counterfactual_group_count"] >= 48,
            {
                "matched_group_count": sim_counterfactual["matched_counterfactual_group_count"],
                "production_counterfactual_mae": sim_prod_evidence["counterfactual_mae"],
                "global_baseline_mae": sim_prod_evidence["global_baseline_mae"],
            },
            "production-grade counterfactual gate evidence",
        ),
        ticket(
            "B5_simulator_as_gate",
            "Simulator limited to budget triage, verifier routing, and policy selection",
            "B",
            ["simulator_gate_calibration_loop_20260612.json"],
            bool(a["sim_calibration"].get("pass"))
            and sim_calibration["gate_router_promoted"]
            and sim_calibration["raw_simulator_promoted"] is False
            and sim_calibration["forbidden_oracle_level_count"] == 0,
            {
                "routing_policy_count": sim_calibration["routing_policy_count"],
                "allowed_routing_level_count": sim_calibration["allowed_routing_level_count"],
            },
            "simulator as gate/router, not oracle",
        ),
        ticket(
            "B6_closed_loop_calibration",
            "Live outcome writeback into calibration and residual loop",
            "B",
            ["simulator_gate_calibration_loop_20260612.json"],
            bool(a["sim_calibration"].get("pass"))
            and sim_calibration["writeback_row_count"] >= 8
            and sim_calibration["simulator_defect_residual_count"] >= 2,
            {
                "writeback_row_count": sim_calibration["writeback_row_count"],
                "simulator_defect_residual_count": sim_calibration["simulator_defect_residual_count"],
            },
            "closed-loop simulator calibration",
        ),
        ticket(
            "B7_production_simulator_candidate",
            "Production graph-action simulator candidate gate",
            "B",
            ["simulator_production_gate_20260612.json", "simulator_production_evidence_20260612.json"],
            bool(a["sim_production_gate"].get("pass"))
            and sim_prod_gate["production_simulator_candidate_allowed"]
            and sim_prod_gate["transition_row_count"] >= 2000
            and sim_prod_gate["pattern_count"] >= 20
            and sim_prod_gate["raw_simulator_promoted"] is False,
            {
                "transition_row_count": sim_prod_gate["transition_row_count"],
                "pattern_count": sim_prod_gate["pattern_count"],
            },
            "production graph-action simulator for triage and routing",
        ),
        ticket(
            "C0_finite_engine_freeze",
            "Freeze finite proof engine boundary",
            "C",
            ["finite_category_certificate_20260612.json", "finite_category_proof_engine_v0.json"],
            bool(a["finite_certificate"].get("pass"))
            and finite_cert["certificate_count"] >= 16
            and finite_cert["unbounded_theorem_prover_claim_allowed"] is False,
            {
                "certificate_count": finite_cert["certificate_count"],
                "not_claimed_count": finite_cert["not_claimed_count"],
            },
            "finite category proof engine v0",
        ),
        ticket(
            "C1_certificate_schema",
            "Formal certificate schema with negative controls",
            "C",
            ["finite_category_certificate_20260612.json"],
            bool(a["finite_certificate"].get("pass"))
            and finite_cert["valid_certificate_count"] == finite_cert["certificate_count"]
            and finite_cert["negative_control_pass_rate"] == 1.0,
            {
                "proof_obligation_count": finite_cert["proof_obligation_count"],
                "negative_control_blocked_count": finite_cert["negative_control_blocked_count"],
            },
            "bounded formal certificates",
        ),
        ticket(
            "C2_lean_export",
            "External-checkable Lean export",
            "C",
            ["finite_category_lean_export_20260612.json", "finite_theorem_lean_verifier_20260612.json"],
            bool(a["finite_lean_export"].get("pass"))
            and bool(a["finite_lean_verifier"].get("pass"))
            and lean_export["external_lean_check_passed"]
            and lean_verifier["external_lean_check_passed"],
            {
                "lean_definition_count": lean_export["lean_definition_count"],
                "lean_theorem_count": lean_verifier["lean_theorem_count"],
            },
            "externally checked finite certificates",
        ),
        ticket(
            "C3_finite_category_dsl",
            "Finite category DSL",
            "C",
            ["finite_formal_reasoning_stack_20260612.json"],
            bool(a["finite_formal_stack"].get("pass"))
            and formal_stack["dsl_valid"]
            and formal_stack["dsl_morphism_count"] >= 10,
            {
                "dsl_object_count": formal_stack["dsl_object_count"],
                "dsl_morphism_count": formal_stack["dsl_morphism_count"],
            },
            "finite category DSL gate",
        ),
        ticket(
            "C4_proof_assistant_check",
            "Proof assistant check of finite theorem fragment",
            "C",
            ["finite_theorem_lean_verifier_20260612.json", "finite_formal_reasoning_stack_20260612.json"],
            bool(a["finite_lean_verifier"].get("pass"))
            and formal_stack["finite_theorem_fragment_external_lean_passed"]
            and formal_stack["finite_theorem_fragment_external_lean_theorem_count"] >= 20,
            {
                "external_lean_theorem_count": formal_stack[
                    "finite_theorem_fragment_external_lean_theorem_count"
                ],
                "advanced_constructions": formal_stack["finite_theorem_fragment_limits_colimits_pass"],
            },
            "Lean-verified finite theorem fragment",
        ),
        ticket(
            "C5_markov_kernel_extension",
            "Finite Markov kernel extension",
            "C",
            ["finite_formal_reasoning_stack_20260612.json"],
            bool(a["finite_formal_stack"].get("pass"))
            and formal_stack["markov_kernel_count"] >= 5
            and formal_stack["kernel_composition_pass"]
            and formal_stack["kernel_negative_control_rejected"],
            {
                "markov_kernel_count": formal_stack["markov_kernel_count"],
                "row_stochastic_pass_count": formal_stack["row_stochastic_pass_count"],
            },
            "finite stochastic-kernel formal fragment",
        ),
        ticket(
            "C6_information_geometry_plugin",
            "Information geometry as measurement plugin",
            "C",
            ["finite_formal_reasoning_stack_20260612.json"],
            bool(a["finite_formal_stack"].get("pass"))
            and formal_stack["metric_count"] >= 5
            and formal_stack["metric_not_truth_oracle"],
            {
                "metric_count": formal_stack["metric_count"],
                "not_truth_oracle": formal_stack["metric_not_truth_oracle"],
            },
            "metric plugin, not truth oracle",
        ),
        ticket(
            "C7_formal_transfer_benchmark",
            "Formal transfer benchmark and overreach residual",
            "C",
            ["finite_formal_reasoning_stack_20260612.json"],
            bool(a["finite_formal_stack"].get("pass"))
            and formal_stack["formal_transfer_pairwise_auc"] >= 0.95
            and formal_stack["formal_transfer_negative_control_rejection_rate"] == 1.0
            and formal_stack["formal_transfer_overreach_residual_count"] >= 1,
            {
                "pairwise_auc": formal_stack["formal_transfer_pairwise_auc"],
                "overreach_residual_count": formal_stack["formal_transfer_overreach_residual_count"],
            },
            "bounded formal transfer predictor",
        ),
        ticket(
            "C8_claim_gate",
            "Formal claim gate blocks full theorem-prover overclaim",
            "C",
            ["finite_formal_reasoning_stack_20260612.json", "nl_to_diagram_scale_benchmark_20260612.json"],
            bool(a["finite_formal_stack"].get("pass"))
            and formal_stack["bounded_formal_stack_claim_allowed"]
            and formal_stack["full_theorem_prover_claim_allowed"] is False
            and nl_scale["full_theorem_prover_claim_allowed"] is False,
            {
                "bounded_formal_stack_claim_allowed": formal_stack["bounded_formal_stack_claim_allowed"],
                "full_theorem_prover_claim_allowed": formal_stack["full_theorem_prover_claim_allowed"],
            },
            "finite proof-gated transfer only",
        ),
        ticket(
            "I1_integrated_recursive_episode",
            "Integrated recursive self-evolution episode",
            "I",
            ["integrated_recursive_episode_20260612.json"],
            bool(a["integrated_episode"].get("pass"))
            and integrated["fresh_ablation_accept_count"] >= 1
            and integrated["autonomy_replay_exact"]
            and integrated["simulator_true_positive_block_count"] == 0,
            {
                "residual_cluster_count": integrated["residual_cluster_count"],
                "fresh_ablation_accept_count": integrated["fresh_ablation_accept_count"],
            },
            "bounded integrated recursive self-evolution episode",
        ),
        ticket(
            "I2_integrated_b3_c2_slice",
            "Integrated uncertainty and Lean-gated episode",
            "I",
            ["integrated_recursive_episode_b3_c2_20260612.json"],
            bool(a["integrated_episode_b3_c2"].get("pass"))
            and integrated_b3_c2["b3_pass"]
            and integrated_b3_c2["c2_pass"]
            and integrated_b3_c2["formal_gate_block_count"] >= 1,
            {
                "b3_abstain_selected_count": integrated_b3_c2["b3_abstain_selected_count"],
                "formal_gate_block_count": integrated_b3_c2["formal_gate_block_count"],
            },
            "simulator-abstain and Lean-gated recursive slice",
        ),
        ticket(
            "P1_paper_main_line",
            "Frozen paper main line with baselines and CIs",
            "P",
            ["paper_frozen_main_experiment_v2_20260612.json"],
            bool(a["paper_main"].get("pass"))
            and paper["problem_count"] >= 1000
            and paper["baseline_count"] >= 8
            and paper["full_v3_margin_over_best_baseline_score"] > 0,
            {
                "problem_count": paper["problem_count"],
                "baseline_count": paper["baseline_count"],
                "margin": paper["full_v3_margin_over_best_baseline_score"],
            },
            "same-batch frozen paper experiment",
        ),
        ticket(
            "P2_creative_generator",
            "Multi-trajectory residual-to-hypothesis generator",
            "P",
            ["creative_hypothesis_trajectory_search_20260612.json"],
            bool(a["creative_generator"].get("pass"))
            and generator["generation_count"] >= 5
            and generator["retained_count"] >= 100
            and generator["nonlocal_candidate_ratio"] >= 0.4,
            {
                "candidate_count": generator["candidate_count"],
                "retained_count": generator["retained_count"],
            },
            "bounded creative hypothesis generator",
        ),
        ticket(
            "P3_main_graph_controlled_apply",
            "Controlled main-graph apply and monitor",
            "P",
            ["main_graph_controlled_apply_monitor_20260612.json"],
            bool(a["main_graph_monitor"].get("pass"))
            and main_graph["source_main_graph_mutated"]
            and main_graph["monitor_day_count"] >= 30
            and main_graph["regression_alert_count"] == 0,
            {
                "monitor_day_count": main_graph["monitor_day_count"],
                "min_precision_delta_vs_before": main_graph["min_precision_delta_vs_before"],
            },
            "canary-scope committed main-graph apply monitor",
        ),
    ]


def _claim_boundary_rows(*, artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    phase13 = _metrics(artifacts["phase13"])
    formal_stack = _metrics(artifacts["finite_formal_stack"])
    sim_gate = _metrics(artifacts["sim_production_gate"])
    final = _metrics(artifacts["final_closure"])
    paper = _metrics(artifacts["paper_main"])
    supervised = _metrics(artifacts["autonomy_supervised"])
    return [
        {
            "claim_id": "unbounded_24_7_general_autonomous_os",
            "blocked": phase13["unbounded_24_7_os_claim_allowed"] is False
            and supervised["ungated_mutation_count"] == 0,
            "reason": "Evidence supports supervised bounded autonomy, not unrestricted 24/7 general OS.",
        },
        {
            "claim_id": "raw_world_simulator_replaces_live_validation",
            "blocked": sim_gate["raw_simulator_promoted"] is False
            and sim_gate["gate_router_promoted"] is True,
            "reason": "Simulator is promoted only for triage/routing; raw replacement remains blocked.",
        },
        {
            "claim_id": "complete_category_theory_theorem_prover",
            "blocked": formal_stack["full_theorem_prover_claim_allowed"] is False,
            "reason": "Lean-verified finite fragment is allowed; arbitrary theorem proving is not.",
        },
        {
            "claim_id": "brand_new_live_api_main_paper_experiment",
            "blocked": paper["new_api_call_count"] == 0,
            "reason": "Paper line is a frozen same-batch artifact aggregation, not a new live API main run.",
        },
        {
            "claim_id": "ungated_default_policy_or_main_graph_mutation",
            "blocked": final["blocked_strong_claim_count"] >= 4,
            "reason": "Graph and policy mutations remain gated or canary-scoped.",
        },
    ]


def _track_complete(tickets: list[dict[str, Any]], track: str) -> bool:
    rows = [row for row in tickets if row["track"] == track]
    return bool(rows) and all(row["status"] == "pass" for row in rows)


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("metrics", payload)


def _load(root: Path, path: Path) -> dict[str, Any]:
    full = root / path
    if not full.exists():
        return {
            "pass": False,
            "missing": True,
            "metrics": {},
        }
    return json.loads(full.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build last_three_part.md ticket-level coverage audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--eval-id", default="last_three_part_coverage_audit_20260612")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_last_three_part_coverage_audit_payload(root=root, eval_id=args.eval_id)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out = md_out if md_out.is_absolute() else root / md_out
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

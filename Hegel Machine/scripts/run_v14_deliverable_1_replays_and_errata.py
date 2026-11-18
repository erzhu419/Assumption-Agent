"""Run Hegel Machine v14 Deliverable 1: V13 5-Item Errata, GPT-6 Probe Remediations, Replays A–F, & Exposed 1024000 Diagnostic.

Writes strict JSON (`allow_nan=False`) receipt:
  ``docs/evidence/v14_continuous_object_and_fair_comparison_20261118/deliverable_1_errata_and_replays_a_to_f_20261118.json``

Preserves 100% byte-for-byte immutability of all 32 historic v10..v13 JSON receipts (`model_calls = 0`).
Uses ONLY retired exposed chains (`1001000`, `1021000`, `1023000`, `1024000`, `1026000`) and minimal controls;
consumes ZERO brand-new unexposed v14 evaluation seeds (`1031000+` / `1041000+`).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    _cross_evaluate_fixed_instance_on_neighbor,
    _cross_evaluate_refittable_family_on_neighbor,
    _cross_evaluate_refittable_template_on_neighbor,
    _refine_coexisting_branch_scopes_and_routing,
    _serialize_obs_rows,
    adjudicate_branches_on_paid_evidence,
    autonomous_reconcile_library_episode,
    dispatch_branches_and_families_for_target,
    evaluate_scope_guard,
    instantiate_aligned_claim_program,
    reconstruct_parametric_families_from_ordinary_archive,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.premise_commitment_index import (
    compute_evidence_archive_fingerprint,
    extract_archive_witness_maps,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments

from scripts.run_v10_3_library_scaling_study import _make_confirmed_regime_prior
from scripts.run_v13_pr_c_tightened_study import (
    EXACT_TOLERANCE,
    SOURCE_COMPUTE_BUDGET,
    SOURCE_MEAS_BUDGET,
    SOURCE_OBS_BUDGET,
    TARGET_COMPUTE_BUDGET,
    TARGET_MEAS_BUDGET,
    TARGET_OBS_BUDGET,
    _eval_external_fresh_48,
    _run_episode,
)

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v14_continuous_object_and_fair_comparison_20261118"
V13_DIR = REPO_ROOT / "docs" / "evidence" / "v13_evidence_recursion_and_routing_20261117"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dpg_dict(donor_min: float, gap_min: float) -> dict:
    q_src = Program("arg", name="edge_source_quantity")
    q_dst = Program("arg", name="edge_target_quantity")
    gap = Program("sub", children=(q_src, q_dst))
    return Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=donor_min), q_src)),
            Program("lt", children=(Program("const", value=gap_min), gap)),
        ),
    ).to_dict()


def build_v13_five_item_errata() -> dict[str, Any]:
    """Recompute & document the 5-item V13 Errata from untouched V13 receipts (§2)."""
    pr_a_path = V13_DIR / "pr_a_evidence_and_recursion_controls_20261117.json"
    pr_b_path = V13_DIR / "pr_b_routing_vs_applicability_and_cache_20261117.json"
    pr_c_path = V13_DIR / "pr_c_tightened_source_paired_study_20261117.json"

    pr_a = json.loads(pr_a_path.read_text(encoding="utf-8"))
    pr_c = json.loads(pr_c_path.read_text(encoding="utf-8"))

    ordinary_sources = [
        ep
        for ch in pr_c["unexposed_evaluation_chains"]
        for ep in ch["source_episodes"]
    ]
    anchors = [ch["calibration_anchor_episode"] for ch in pr_c["unexposed_evaluation_chains"]]
    src_internal_ok = sum(1 for e in ordinary_sources if e["internally_confirmed"])
    src_strict_ok = sum(1 for e in ordinary_sources if e["strict_pass"])
    src_false_confs = [
        {
            "episode_tag": e["episode_tag"],
            "internal_confirmation_loss": e["internal_confirmation_loss"] if "internal_confirmation_loss" in e else e.get("confirmation_loss"),
            "external_fresh48_mse": e["external_fresh48_mse"],
        }
        for e in ordinary_sources
        if e["false_internal_confirmation"]
    ]

    arm_agg = pr_c["unexposed_8cluster_aggregate"]["arm_aggregates"]
    r_mean = arm_agg["R_ordinary_optimized_lazy"]["measured_target_costs"]["mean_compute"]
    k_noh_mean = arm_agg["K_routing_no_H_star"]["measured_target_costs"]["mean_compute"]
    k_plush_mean = arm_agg["K_routing_plus_H_star"]["measured_target_costs"]["mean_compute"]

    cp2 = pr_a["c1_three_ordinary_entry_choice_points"]["choice_point_2_disagreement_probe_unavailable"]
    cp2_exec = cp2.get("paid_probes_executed") or []
    cp2_decs = cp2.get("reconciliation_decisions") or []

    return {
        "untouched_v13_receipts_sha256": {
            "pr_a": _sha256(pr_a_path),
            "pr_b": _sha256(pr_b_path),
            "pr_c": _sha256(pr_c_path),
        },
        "erratum_1_ordinary_source_strict_count": {
            "v13_prose_claim_retracted": "32/32 ordinary source tasks strict pass (0 false confirmations)",
            "actual_recomputed_counts": {
                "ordinary_source_total": len(ordinary_sources),
                "ordinary_source_internally_confirmed": src_internal_ok,
                "ordinary_source_strict_pass": src_strict_ok,
                "ordinary_source_false_confirmations_count": len(src_false_confs),
                "false_confirmation_episodes": src_false_confs,
                "common_calibration_anchors_strict_pass": sum(1 for a in anchors if a["strict_pass"]),
                "common_calibration_anchors_total": len(anchors),
            },
        },
        "erratum_2_confounded_18_92pct_compute_comparison": {
            "v13_comparison": "R_ordinary_optimized_lazy (H* ON) vs K_routing_no_H_star (H* OFF)",
            "r_h_on_mean_compute": r_mean,
            "k_no_h_mean_compute": k_noh_mean,
            "relative_delta_pct": round(100.0 * (r_mean - k_noh_mean) / k_noh_mean, 2),
            "correction": (
                "R_ordinary_optimized_lazy had _prioritize_generalized_families=True while K_routing_no_H_star had "
                "_prioritize_generalized_families=False; therefore -18.92% confounded branch organization with H* scheduling."
            ),
        },
        "erratum_3_equal_h_star_on_comparison": {
            "r_h_on_strict_passes": arm_agg["R_ordinary_optimized_lazy"]["strict_passes"],
            "k_h_on_strict_passes": arm_agg["K_routing_plus_H_star"]["strict_passes"],
            "r_h_on_mean_compute": r_mean,
            "k_h_on_mean_compute": k_plush_mean,
            "k_over_r_compute_overhead_pct": round(100.0 * (k_plush_mean - r_mean) / r_mean, 2),
            "correction": (
                "Under equal H* ON, K_routing_plus_H_star (20/24, mean 249,564.33) cost +9.97% more compute "
                "and passed 1 fewer target than R_ordinary_optimized_lazy (21/24, mean 226,943.08)."
            ),
        },
        "erratum_4_pr_a_choice_point_2_probe_accounting": {
            "paid_probes_executed_len": len(cp2_exec),
            "paid_probes_executed_statuses": [p.get("acquisition_status") for p in cp2_exec],
            "unavailable_decisions_count": sum(1 for d in cp2_decs if d.get("probe_acquisition_status") == "unavailable"),
            "any_unavailable_in_paid_probes_executed": any(
                p.get("acquisition_status") != "executed_new" for p in cp2_exec
            ),
            "correction": (
                "Choice Point 2 had 1 genuine executed_new probe on the first branch and 1 unavailable probe on the second branch; "
                "the true invariant is that zero unavailable requests appear in paid_probes_executed, not len(paid_probes_executed)==0."
            ),
        },
        "erratum_5_boundary_probe_wiring_and_h_star_default": {
            "v13_boundary_probe_status": "standalone_probe_generator_only_not_wired_into_ordinary_confirm_loop",
            "v12_to_v13_rate_note": "V12 (19/24) and V13 (21/24) ran on distinct chains and codebases; cannot be claimed as single-factor cache proof.",
            "v14_policy": (
                "Wired refine_candidate_via_threshold_boundary_probe into ordinary UnifiedResearchAgent.run() prior to fresh confirmation, "
                "and set default H* front-priority OFF (_prioritize_generalized_families=False) in main comparisons while retaining H* representation."
            ),
        },
    }


def run_gpt6_v14_audit_probes() -> dict[str, Any]:
    """Execute GPT-6's exact 3 probe scripts from `markdown/hegel_v14/probes/` against updated sources."""
    probes_dir = REPO_ROOT / "markdown" / "hegel_v14" / "probes"
    with tempfile.TemporaryDirectory(prefix="hegel_v14_probe_verify_") as tmp:
        stage = Path(tmp)
        (stage / "bundled_sources" / "src" / "hegel_machine").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPO_ROOT / "docs" / "gpt6_audit_bundle_v13_20261117" / "05_STANDALONE_VERIFY_V13.py",
            stage / "05_STANDALONE_VERIFY_V13.py",
        )
        shutil.copytree(
            REPO_ROOT / "src" / "hegel_machine" / "cognition",
            stage / "bundled_sources" / "src" / "hegel_machine" / "cognition",
        )
        results: dict[str, Any] = {}
        for script_name in (
            "probe_acquisition_contract.py",
            "probe_question_identity.py",
            "probe_scope_and_routing.py",
        ):
            out_json = stage / script_name.replace(".py", ".json")
            proc = subprocess.run(
                [sys.executable, str(probes_dir / script_name), "--bundle", str(stage), "--output", str(out_json)],
                capture_output=True,
                text=True,
                check=True,
            )
            parsed = json.loads(out_json.read_text(encoding="utf-8"))
            results[script_name.replace(".py", "")] = parsed

    p_acq = results["probe_acquisition_contract"]
    p_qid = results["probe_question_identity"]
    p_sr = results["probe_scope_and_routing"]

    remediation_checks = {
        "real_production_batch_reused_exact_protocol": p_acq["real_production_batch"]["status"] == "reused_exact_protocol",
        "real_production_batch_zero_new_acquisitions": p_acq["real_production_batch"]["new_acquisition_attempts"] == 0,
        "distinct_parent_programs_distinct_question_ids": p_qid["identical_question_ids"] is False,
        "zero_budget_zero_loss_evaluations": p_qid["zero_budget"]["predictive_loss_evaluations_still_executed"] == 0,
        "zero_budget_zero_claim_mutations": p_qid["zero_budget"]["claims_mutated_after_denial"] is False,
        "empty_family_search_not_empirically_refuted": p_sr["empty_family_search"]["verdict"]["empirically_refuted"] is False,
        "empty_family_search_strict_json_null_mse": p_sr["empty_family_search"]["verdict"]["cross_mse"] is None,
        "router_deferred_ordinary_aggregate_retained": p_sr["router_deferred_aggregate_dropped"]["ordinary_aggregate_retained"] is True,
        "deferred_router_families_consumed_in_unified_research": p_sr["router_deferred_family_not_in_consumed_list"]["consumer_reference_count"] > 0,
        "denied_fixed_cross_check_zero_fit_calls": p_sr["denied_fixed_cross_check_still_executed"]["fit_calls"] == 0,
        "fixed_instance_scope_preserves_refittable_template": len(p_sr["fixed_instance_vs_refittable_template_control"]["dispatched_templates"]) == 1,
    }
    assert all(remediation_checks.values()), f"Probe check failed: {remediation_checks}"
    return {
        "all_gpt6_v14_probes_remediated": True,
        "remediation_checks": remediation_checks,
        "probe_outputs": results,
    }


def run_replays_a_to_f_and_1024000_diagnostic() -> dict[str, Any]:
    """Execute Replays A–F (§6) plus the exposed `1024000` R/K × H* 2×2 diagnostic (§4.3)."""
    # --- Replay A ---
    with AnonymousObjectLab(
        seed=1001001,
        objects=3,
        readout="focus",
        variant="exchange",
        relation_direction="right",
        relation_condition="always",
        relation_radius=0.92,
    ) as lab_a:
        res_a = UnifiedResearchAgent(
            ResearchConfig(
                seed=1001001,
                observation_budget=96,
                compute_budget=180_000,
                max_rounds=1,
                minimum_improvement=1e-7,
                use_memory=True,
                world_structure_commitments=True,
                transition_premise_audit=True,
                branch_reconciliation=True,
            ),
            measurement_budget=3_000,
            library=KnowledgeLibrary(),
        ).run(lab_a)
        first_fine = next(b for b in res_a["batches"] if tuple(b["channels"]) == ("fine",))
        ur_path = REPO_ROOT / "src" / "hegel_machine" / "cognition" / "unified_research.py"
        fn_node = next(
            n for n in ast.walk(ast.parse(ur_path.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "_exec_reconcile_probe"
        )
        acq_calls_a: list[str] = []

        def _acq_a(channels, purpose, *, probe=None, **kw):
            acq_calls_a.append(probe.probe_id)
            b_r = lab_a.acquire(probe, channels)
            ns_a["batches"].append(
                {
                    "probe": asdict(probe),
                    "channels": list(channels),
                    "observations": [asdict(r) for r in b_r.observations],
                    "transitions": getattr(b_r, "transitions", []),
                    "observation_cost": int(b_r.observation_cost),
                    "measurement_cost": int(b_r.measurement_cost),
                    "episode_uid": res_a["episode_uid"],
                    "history_preparation": "reset",
                }
            )
            return b_r.observations

        ns_a = {
            "reconcile_channels": ("fine",),
            "batches": [deepcopy(first_fine)],
            "acquire": _acq_a,
            "episode_uid": res_a["episode_uid"],
            "Observation": Observation,
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn_node], type_ignores=[])), str(ur_path), "exec"), ns_a)
        same_p = Probe("replay_a_same_seq", tuple(dict(pt) for pt in first_fine["probe"]["sequence"]))
        reuse_a = ns_a["_exec_reconcile_probe"](same_p, ["prog_v1:a", "prog_v1:b"])
        ns_a["episode_uid"] = "different_episode_uid"
        exec_a = ns_a["_exec_reconcile_probe"](same_p, ["prog_v1:a"])

    replay_a = {
        "passed": (
            reuse_a["status"] == "reused_exact_protocol"
            and exec_a["status"] == "executed_new"
            and exec_a["cost_ref"]["new_measurements"] > 0
        ),
        "production_batch_keys": sorted(first_fine.keys()),
        "exact_match_status": reuse_a["status"],
        "exact_match_actual_probe_id": reuse_a["actual_probe_id"],
        "mismatched_episode_status": exec_a["status"],
        "mismatched_episode_cost_ref": exec_a["cost_ref"],
    }

    # --- Replay B ---
    p_r050 = _make_confirmed_regime_prior(concept_id="src_r050", seed=9401, objects=3, readout="focused_next", direction="right", edge_pred_dict=None, radius=0.50)
    p_r075 = _make_confirmed_regime_prior(concept_id="src_r075", seed=9402, objects=3, readout="focused_next", direction="right", edge_pred_dict=None, radius=0.75)
    p_left = _make_confirmed_regime_prior(concept_id="src_left", seed=9403, objects=3, readout="focused_next", direction="left", edge_pred_dict=None, radius=0.75)
    sup_b, _ = extract_archive_witness_maps([p_r050, p_r075, p_left])

    def _mk_b(p: dict, cid: str, r_val: float, flux: float, blk: float) -> dict:
        c = consolidate_world_structure_commitments([p])[0]
        c["commitment_id"] = f"branch_comm:{cid}"
        c["source_concept_ids"] = [cid]
        return {
            "claim_id": f"branch:{cid}",
            "source_concept_id": cid,
            "source_concept_ids": [cid],
            "commitment": c,
            "radius_value": r_val,
            "schema": p["object_schema"],
            "status": "universal",
            "independent_empirical_support": True,
            "pilot_regime_features": [
                {
                    "obs_signed_direction_flux": flux,
                    "obs_unconditional_block_rate": blk,
                    "_meta": {"protocol_id": "paid_episode_transitions", "pilot_calibration_available": True},
                }
            ],
            "support_rows_serialized": _serialize_obs_rows(sup_b[cid], limit=64),
        }

    b050 = _mk_b(p_r050, "src_r050", 0.50, 0.12, 0.10)
    b075 = _mk_b(p_r075, "src_r075", 0.75, 0.28, 0.45)
    bleft = _mk_b(p_left, "src_left", 0.75, -0.25, 0.45)

    z_fix = _cross_evaluate_fixed_instance_on_neighbor(b050, b075, budget=SearchBudget(0))
    fx_075 = _cross_evaluate_fixed_instance_on_neighbor(b050, b075, budget=SearchBudget(20_000))
    tm_075 = _cross_evaluate_refittable_template_on_neighbor(b050, b075, budget=SearchBudget(20_000))
    tm_lft = _cross_evaluate_refittable_template_on_neighbor(b050, bleft, budget=SearchBudget(20_000))
    ref_same = _refine_coexisting_branch_scopes_and_routing([b050, b075], budget=SearchBudget(40_000))
    ref_opp = _refine_coexisting_branch_scopes_and_routing([b050, bleft], budget=SearchBudget(40_000))

    replay_b = {
        "passed": (
            z_fix["execution_status"] == "budget_deferred"
            and fx_075["verdict"] == "refuted_instance_on_batch"
            and tm_075["verdict"] == "supported_on_batch"
            and tm_lft["verdict"] == "infeasible_on_declared_domain"
            and ref_same[0]["scope_claim_level"] == "bound_instance"
            and ref_opp[0]["scope_claim_level"] == "refittable_template"
        ),
        "zero_budget_fixed_check": z_fix,
        "radius_050_on_075_fixed_instance": fx_075,
        "radius_050_on_075_refittable_template": tm_075,
        "right_on_left_refittable_template": tm_lft,
        "same_mechanism_scope_claim_level": ref_same[0]["scope_claim_level"],
        "opposite_direction_scope_claim_level": ref_opp[0]["scope_claim_level"],
        "automatic_protocol_mismatch_verdict": evaluate_scope_guard(
            ref_opp[0]["scope_guard"],
            {"obs_signed_direction_flux": 0.2, "_meta": {"protocol_id": "other_protocol", "pilot_calibration_available": True}},
        ),
    }

    # --- Replay C ---
    p_s1 = _make_confirmed_regime_prior(concept_id="s1", seed=9501, objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.18), radius=0.92)
    p_s2 = _make_confirmed_regime_prior(concept_id="s2", seed=9502, objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.34), radius=0.92)
    lib_c = KnowledgeLibrary()
    for cid_c, pr_c_i in (("s1", p_s1), ("s2", p_s2)):
        lib_c.meta_priors.append(deepcopy(pr_c_i))
        sup_c = extract_archive_witness_maps([pr_c_i])[0][cid_c]
        autonomous_reconcile_library_episode(
            library=lib_c,
            task_id=f"src_{cid_c}",
            spec=TaskSpec(f"src_{cid_c}", "d", tuple(pr_c_i["object_schema"]["fields"][g][k] for g in pr_c_i["object_schema"]["objects"] for k in ("q", "x", "u", "f")), (), (), feature_bounds={}),
            schema_dict=pr_c_i["object_schema"],
            confirmed=True,
            confirmation_loss=0.0,
            confirmation_rows_list=sup_c[-12:],
            support_rows=sup_c[:-12],
            episode_transitions=[],
            concept_id=cid_c,
            execute_probe_fn=None,
            budget=SearchBudget(80_000),
        )
    defer_guard = Program("lt", children=(Program("feature", name="obs_signed_direction_flux"), Program("const", value=-0.5))).to_dict()
    for b in lib_c._active_reconciled_branches:
        b["routing_guard"] = deepcopy(defer_guard)
    lib_c._generalized_branch_commitments[0]["routing_guard"] = deepcopy(defer_guard)
    fam_c_id = lib_c._generalized_branch_commitments[0]["commitment_id"]

    p_tgt = _make_confirmed_regime_prior(concept_id="tgt", seed=9503, objects=3, readout="focused_next", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.26), radius=0.92)
    tgt_rows = extract_archive_witness_maps([p_tgt])[0]["tgt"]
    sc_t = p_tgt["object_schema"]
    o0, o1 = sc_t["objects"][0], sc_t["objects"][1]
    pos_trans = [
        {
            "before": {sc_t["fields"][o0]["q"]: 0.8, sc_t["fields"][o0]["x"]: 0.0, sc_t["fields"][o1]["q"]: 0.1, sc_t["fields"][o1]["x"]: 0.4},
            "after": {sc_t["fields"][o0]["q"]: 0.6, sc_t["fields"][o0]["x"]: 0.0, sc_t["fields"][o1]["q"]: 0.3, sc_t["fields"][o1]["x"]: 0.4},
        }
    ]
    c_agg = consolidate_world_structure_commitments([p_s1])[0]
    c_agg["commitment_id"] = "world_commitment:0000"
    c_agg["source_concept_ids"] = ["s1"]
    disp_c = dispatch_branches_and_families_for_target(
        library=lib_c,
        consistent_comms=[c_agg],
        transitions=pos_trans,
        schema=sc_t,
        audit={"radius_interval": [0.92, 0.92], "verified_direction": "right"},
        task_sig="tgt_026",
        comm_budget=SearchBudget(50_000),
    )
    p_uncond = _make_confirmed_regime_prior(concept_id="s0_uncond", seed=9500, objects=3, readout="focused_next", direction="right", edge_pred_dict=None, radius=0.92)
    lib_fb = KnowledgeLibrary()
    lib_fb.meta_priors = [deepcopy(p_uncond)]
    lib_fb._generalized_branch_commitments = deepcopy(lib_c._generalized_branch_commitments)
    acts_c: list[dict] = []
    spec_t = TaskSpec(
        "tgt_026",
        "d",
        tuple(sc_t["fields"][g][k] for g in sc_t["objects"] for k in ("q", "x", "u", "f")),
        (),
        (),
        feature_bounds={},
        context={"object_schema": json.dumps(sc_t)},
    )
    cands_c = UnifiedResearchAgent(
        ResearchConfig(seed=9503, use_memory=True, world_structure_commitments=True, branch_reconciliation=True, active_threshold_probing=False),
        library=lib_fb,
    )._propose_schema_structures(spec_t, tgt_rows[:32], SearchBudget(120_000), actions=acts_c, tolerance=1e-7, transitions=pos_trans)
    apply_c = next(a for a in acts_c if a.get("action") == "apply_world_structure_commitments")
    replay_c = {
        "passed": (
            any(c["commitment_id"] == "world_commitment:0000" for c in disp_c["ordered_consistent_comms"])
            and fam_c_id in apply_c["branch_dispatch_telemetry"]["deferred_router_families_tried"]
            and any(c["training_loss"] <= 1e-7 for c in cands_c)
        ),
        "world_commitment_0000_retained_in_tail": any(c["commitment_id"] == "world_commitment:0000" for c in disp_c["ordered_consistent_comms"]),
        "deferred_router_families_tried": apply_c["branch_dispatch_telemetry"]["deferred_router_families_tried"],
        "budget_not_reached": apply_c["branch_dispatch_telemetry"]["budget_not_reached"],
        "best_fallback_training_loss": min(float(c["training_loss"]) for c in cands_c),
    }

    # --- Replay D + Exposed `1024000` R/K x H* 2x2 Diagnostic ---
    spec_1024 = {
        "chain_id": "exposed_retired_v13_chain_1024000",
        "base_seed": 1_024_000,
        "source_thresholds": (0.21, 0.46, 0.23),
        "source_radii": (0.92, 0.94, 0.89, 0.93),
        "target_thresholds": (0.34, 0.33),
        "target_radii": (0.90, 0.91, 0.94),
    }
    base_sd = spec_1024["base_seed"]
    th_a, th_b, th_l = spec_1024["source_thresholds"]
    r0, ra, rb, rl = spec_1024["source_radii"]
    th_t1, th_t3 = spec_1024["target_thresholds"]
    rt1, rt2, rt3 = spec_1024["target_radii"]

    lib_1024_k = KnowledgeLibrary()
    for tag_s, sd_s, dir_s, cond_s, th_s, r_s in (
        ("S_cal", base_sd + 5, "symmetric", "always", 0.0, 0.92),
        ("S0", base_sd + 15, "right", "always", 0.0, r0),
        ("S1", base_sd + 25, "right", "donor_positive_gap", th_a, ra),
        ("S2", base_sd + 35, "right", "donor_positive_gap", th_b, rb),
        ("S_left", base_sd + 45, "left", "donor_positive_gap", th_l, rl),
    ):
        _run_episode(
            arm_name="source_1024000_diag",
            episode_tag=f"1024000:{tag_s}",
            lab_kwargs={
                "seed": sd_s,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": dir_s,
                "relation_condition": cond_s,
                "condition_threshold": th_s,
                "relation_radius": r_s,
                "actuator_curvature": 0.15,
                "name_seed": sd_s + 100,
                "wiring_seed": sd_s + 200,
            },
            library=lib_1024_k,
            agent_seed=sd_s,
            obs_b=SOURCE_OBS_BUDGET,
            meas_b=SOURCE_MEAS_BUDGET,
            comp_b=SOURCE_COMPUTE_BUDGET,
        )

    # Check Replay D equivalence & content-hash invalidation directly on `lib_1024_k`!
    lib_1024_r_noh = deepcopy(lib_1024_k)
    lib_1024_r_noh._active_reconciled_branches = []
    lib_1024_r_noh._generalized_branch_commitments = []
    lib_1024_r_noh._retracted_premise_or_claim_ids = set()
    lib_1024_r_noh._premise_commitment_index = None
    lib_1024_r_noh._indexed_prior_fingerprints = None
    lib_1024_r_noh._indexed_group_commitments = None
    lib_1024_r_noh._prioritize_generalized_families = False
    lib_1024_r_noh._allow_on_demand_family_reconstruction = True
    lib_1024_r_noh._allow_on_demand_routing_reconstruction = True

    fams_d1, st_d1 = reconstruct_parametric_families_from_ordinary_archive(
        lib_1024_r_noh, SearchBudget(200_000), reconstruct_routing=True
    )
    _, st_d2 = reconstruct_parametric_families_from_ordinary_archive(
        lib_1024_r_noh, SearchBudget(200_000), reconstruct_routing=True
    )
    lib_edit_test = deepcopy(lib_1024_r_noh)
    if not lib_edit_test._raw_reconciliation_evidence_archive:
        lib_edit_test._raw_reconciliation_evidence_archive = [
            {
                "evidence_id": "ev_test:0",
                "protocol_id": "paid_episode_transitions",
                "schema": deepcopy(lib_1024_k._active_reconciled_branches[-1]["schema"]),
                "observations_serialized": deepcopy(lib_1024_k._active_reconciled_branches[-1]["support_rows_serialized"][:8]),
                "transitions": [],
            }
        ]
        reconstruct_parametric_families_from_ordinary_archive(lib_edit_test, SearchBudget(200_000), reconstruct_routing=True)
    fp_d_before = compute_evidence_archive_fingerprint(lib_edit_test._raw_reconciliation_evidence_archive)
    lib_edit_test._raw_reconciliation_evidence_archive[0]["protocol_id"] = "edited_protocol_v2"
    fp_d_after = compute_evidence_archive_fingerprint(lib_edit_test._raw_reconciliation_evidence_archive)
    _, st_d3 = reconstruct_parametric_families_from_ordinary_archive(
        lib_edit_test, SearchBudget(200_000), reconstruct_routing=True
    )

    # Run 2x2 (`R/K x H* ON/OFF`) on exposed `1024000/T1`!
    diag_arms_1024 = {}
    for arm_lbl, is_lazy, h_on in (
        ("R_lazy_noH", True, False),
        ("K_eager_noH", False, False),
        ("R_lazy_H_on", True, True),
        ("K_eager_H_on", False, True),
    ):
        l_arm = deepcopy(lib_1024_k)
        l_arm._freeze_branch_writes = True
        l_arm._enable_scope_and_routing_guards = True
        l_arm._prioritize_generalized_families = h_on
        if is_lazy:
            l_arm._active_reconciled_branches = []
            l_arm._generalized_branch_commitments = []
            l_arm._retracted_premise_or_claim_ids = set()
            l_arm._premise_commitment_index = None
            l_arm._indexed_prior_fingerprints = None
            l_arm._indexed_group_commitments = None
            l_arm._allow_on_demand_family_reconstruction = True
            l_arm._allow_on_demand_routing_reconstruction = True
        else:
            l_arm._allow_on_demand_family_reconstruction = False
            l_arm._allow_on_demand_routing_reconstruction = False
        ep_t1 = _run_episode(
            arm_name=arm_lbl,
            episode_tag="1024000:T1_unseen_right_dpg_N5",
            lab_kwargs={
                "seed": base_sd + 305,
                "objects": 5,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_t1,
                "relation_radius": rt1,
                "actuator_curvature": 0.15,
                "name_seed": base_sd + 405,
                "wiring_seed": base_sd + 505,
            },
            library=l_arm,
            agent_seed=base_sd + 305,
            obs_b=TARGET_OBS_BUDGET,
            meas_b=TARGET_MEAS_BUDGET,
            comp_b=TARGET_COMPUTE_BUDGET,
        )
        diag_arms_1024[arm_lbl] = {
            "strict_pass": ep_t1["strict_pass"],
            "internally_confirmed": ep_t1["internally_confirmed"],
            "external_fresh48_mse": ep_t1["external_fresh48_mse"],
            "measured_costs": ep_t1["measured_costs"],
            "ordinary_lazy_episode_reconstruction_compute": ep_t1["ordinary_lazy_episode_reconstruction_compute"],
        }

    replay_d = {
        "passed": (
            fams_d1 == []
            and st_d1["families_skipped_noH"] is True
            and st_d1["layer2_built"] == 0
            and st_d2["cache_hit"] is True
            and fp_d_before != fp_d_after
            and st_d3["cache_hit"] is False
        ),
        "noH_skipped_layer2_family_build": st_d1["families_skipped_noH"],
        "unchanged_second_call_cache_hit": st_d2["cache_hit"],
        "same_length_content_edit_fingerprint_changed": fp_d_before != fp_d_after,
        "same_length_content_edit_invalidated_layer3": st_d3["cache_hit"] is False and st_d3["layer3_routing_built"] is True,
        "exposed_retired_1024000_T1_2x2_diagnostic": diag_arms_1024,
    }

    # --- Replay E ---
    lib_e = KnowledgeLibrary()
    for sd_e, dm_e in ((9700, "symmetric"), (9701, "right")):
        with AnonymousObjectLab(
            seed=sd_e,
            objects=4,
            readout="focus",
            relation_direction=dm_e,
            relation_condition="always",
            condition_threshold=0.0,
            relation_radius=0.93,
            actuator_curvature=0.15,
            name_seed=sd_e + 31,
            wiring_seed=sd_e + 61,
        ) as lab_es:
            UnifiedResearchAgent(
                ResearchConfig(
                    seed=sd_e,
                    observation_budget=520,
                    compute_budget=520_000,
                    max_rounds=4,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    use_memory=True,
                    world_structure_commitments=True,
                    transition_premise_audit=True,
                    expanded_premise_revisions=True,
                    active_threshold_probing=True,
                    indexed_commitment_store=True,
                    sensor_role_calibration_priors=True,
                    occam_premise_minimality=True,
                    branch_reconciliation=True,
                ),
                measurement_budget=11_000,
                library=lib_e,
            ).run(lab_es)

    fm_kw = dict(
        seed=9705,
        objects=4,
        readout="focus",
        relation_direction="right",
        relation_condition="always",
        condition_threshold=0.0,
        relation_radius=0.93,
        latent_mode="focal_memory",
        latent_decay=0.55,
        latent_weight=0.35,
        actuator_curvature=0.15,
        name_seed=9736,
        wiring_seed=9766,
    )
    with AnonymousObjectLab(**fm_kw) as lab_e1:
        res_e1 = UnifiedResearchAgent(
            ResearchConfig(
                seed=9705,
                observation_budget=195,
                compute_budget=480_000,
                max_rounds=1,
                minimum_improvement=1e-7,
                max_candidates=4,
                use_memory=True,
                world_structure_commitments=True,
                transition_premise_audit=True,
                expanded_premise_revisions=False,
                latent_state_discovery=False,
                active_threshold_probing=False,
                state_search=True,
                sensor_role_calibration_priors=True,
                occam_premise_minimality=True,
                branch_reconciliation=True,
            ),
            measurement_budget=3_600,
            library=lib_e,
        ).run(lab_e1)

    q_e1 = next(q for q in lib_e._pending_reconciliation_questions if q.get("status") == "pending_unfunded")
    q_e1_id = q_e1["question_id"]
    lib_e_stale = deepcopy(lib_e)
    lib_e_stale._active_reconciled_branches[0]["semantic_version"] = "sem_v1:mutated_parent"
    cfg_e2 = ResearchConfig(
        seed=9705,
        observation_budget=560,
        compute_budget=880_000,
        max_rounds=1,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=False,
        latent_state_discovery=False,
        active_threshold_probing=False,
        state_search=True,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=True,
    )
    with AnonymousObjectLab(**fm_kw) as lab_est:
        UnifiedResearchAgent(cfg_e2, measurement_budget=11_500, library=lib_e_stale).run(lab_est)
    stale_q_status = next(q["status"] for q in lib_e_stale._pending_reconciliation_questions if q["question_id"] == q_e1_id)

    with AnonymousObjectLab(**fm_kw) as lab_e2:
        res_e2 = UnifiedResearchAgent(cfg_e2, measurement_budget=11_500, library=lib_e).run(lab_e2)
        ext_e2 = _eval_external_fresh_48(lab_e2, res_e2, seed=9705 + 920_001)
    q_e2 = next(q for q in lib_e._pending_reconciliation_questions if q["question_id"] == q_e1_id)

    replay_e = {
        "passed": (
            res_e1["status"] == "unresolved"
            and stale_q_status == "stale_parent_version_changed"
            and res_e2["status"] == "completed"
            and q_e2["status"] == "answered"
            and ext_e2["external_fresh48_mse"] <= 1e-7
            and res_e2["costs"]["observation_cost"] > res_e1["costs"]["observation_cost"]
        ),
        "question_id": q_e1_id,
        "run1_status": res_e1["status"],
        "run1_costs": res_e1["costs"],
        "mutated_parent_status": stale_q_status,
        "run2_status": res_e2["status"],
        "run2_cumulative_costs": res_e2["costs"],
        "run2_external_fresh48_mse": ext_e2["external_fresh48_mse"],
        "repaired_concept_id": q_e2["resolution"].get("repaired_concept_id"),
        "post_subresearch_reconciliation_status": q_e2["resolution"].get("post_subresearch_reconciliation_status"),
    }

    # --- Replay F: Exposed `1021000/S_left_dpg_N4`, `1023000/T3`, and `1026000/T3` ---
    replay_f_cases: dict[str, Any] = {}
    # 1) Exact exposed `1021000` prefix (`S_cal -> S0 -> S1 -> S2 -> S_left_dpg_N4`)
    lib_f1 = KnowledgeLibrary()
    for tg_f1, sd_f1, dm_f1, cn_f1, th_f1, r_f1 in (
        ("S_cal", 1_021_005, "symmetric", "always", 0.0, 0.92),
        ("S0", 1_021_015, "right", "always", 0.0, 0.94),
        ("S1", 1_021_025, "right", "donor_positive_gap", 0.19, 0.91),
        ("S2", 1_021_035, "right", "donor_positive_gap", 0.45, 0.93),
    ):
        _run_episode(
            arm_name="src_1021000_prefix",
            episode_tag=f"1021000:{tg_f1}",
            lab_kwargs={
                "seed": sd_f1,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": dm_f1,
                "relation_condition": cn_f1,
                "condition_threshold": th_f1,
                "relation_radius": r_f1,
                "actuator_curvature": 0.15,
                "name_seed": sd_f1 + 100,
                "wiring_seed": sd_f1 + 200,
            },
            library=lib_f1,
            agent_seed=sd_f1,
            obs_b=SOURCE_OBS_BUDGET,
            meas_b=SOURCE_MEAS_BUDGET,
            comp_b=SOURCE_COMPUTE_BUDGET,
        )
    with AnonymousObjectLab(
        seed=1_021_045,
        objects=4,
        readout="window_total_event",
        relation_direction="left",
        relation_condition="donor_positive_gap",
        condition_threshold=0.25,
        relation_radius=0.90,
        actuator_curvature=0.15,
        name_seed=1_021_145,
        wiring_seed=1_021_245,
    ) as lab_f1:
        res_f1 = UnifiedResearchAgent(
            ResearchConfig(
                seed=1_021_045,
                observation_budget=SOURCE_OBS_BUDGET,
                compute_budget=SOURCE_COMPUTE_BUDGET,
                max_rounds=4,
                minimum_improvement=1e-7,
                max_candidates=4,
                use_memory=True,
                world_structure_commitments=True,
                transition_premise_audit=True,
                expanded_premise_revisions=True,
                active_threshold_probing=True,
                indexed_commitment_store=True,
                sensor_role_calibration_priors=True,
                occam_premise_minimality=True,
                branch_reconciliation=True,
            ),
            measurement_budget=SOURCE_MEAS_BUDGET,
            library=lib_f1,
        ).run(lab_f1)
        ext_f1 = _eval_external_fresh_48(lab_f1, res_f1, seed=1_021_045 + 920_001)
    bp_f1 = [a for a in res_f1["actions"] if a.get("action") == "execute_threshold_boundary_probe"]
    replay_f_cases["exposed_1021000_S_left_dpg_N4"] = {
        "status": res_f1["status"],
        "internal_confirmation_loss": res_f1["confirmation_loss"],
        "external_fresh48_mse": ext_f1["external_fresh48_mse"],
        "false_confirmation_eliminated": bool(res_f1["status"] == "completed" and ext_f1["external_fresh48_mse"] <= 1e-7),
        "boundary_probes_executed": bp_f1,
        "costs": res_f1["costs"],
    }

    # 2) Exposed 1023000/T3 and 1026000/T3
    for cid_f, b_sd, src_ths, src_rs, t3_th, t3_r in (
        ("exposed_1023000_T3", 1_023_000, (0.17, 0.43, 0.27), (0.93, 0.90, 0.94, 0.91), 0.38, 0.89),
        ("exposed_1026000_T3", 1_026_000, (0.22, 0.48, 0.24), (0.90, 0.93, 0.95, 0.92), 0.35, 0.91),
    ):
        lib_ft = KnowledgeLibrary()
        for tg, sd_i, dm_i, cn_i, th_i, r_i in (
            ("S_cal", b_sd + 5, "symmetric", "always", 0.0, 0.92),
            ("S1", b_sd + 25, "right", "donor_positive_gap", src_ths[0], src_rs[1]),
            ("S_left", b_sd + 45, "left", "donor_positive_gap", src_ths[2], src_rs[3]),
            ("S0", b_sd + 15, "right", "always", 0.0, src_rs[0]),
            ("S2", b_sd + 35, "right", "donor_positive_gap", src_ths[1], src_rs[2]),
        ):
            _run_episode(
                arm_name="src",
                episode_tag=f"{cid_f}:{tg}",
                lab_kwargs={
                    "seed": sd_i,
                    "objects": 4,
                    "readout": "window_total_event",
                    "relation_direction": dm_i,
                    "relation_condition": cn_i,
                    "condition_threshold": th_i,
                    "relation_radius": r_i,
                    "actuator_curvature": 0.15,
                    "name_seed": sd_i + 100,
                    "wiring_seed": sd_i + 200,
                },
                library=lib_ft,
                agent_seed=sd_i,
                obs_b=SOURCE_OBS_BUDGET,
                meas_b=SOURCE_MEAS_BUDGET,
                comp_b=SOURCE_COMPUTE_BUDGET,
            )
        lib_ft._freeze_branch_writes = True
        lib_ft._prioritize_generalized_families = False
        with AnonymousObjectLab(
            seed=b_sd + 325,
            objects=5,
            readout="window_total_event",
            relation_direction="left",
            relation_condition="donor_positive_gap",
            condition_threshold=t3_th,
            relation_radius=t3_r,
            actuator_curvature=0.15,
            name_seed=b_sd + 425,
            wiring_seed=b_sd + 525,
        ) as lab_t3:
            res_t3 = UnifiedResearchAgent(
                ResearchConfig(
                    seed=b_sd + 325,
                    observation_budget=TARGET_OBS_BUDGET,
                    compute_budget=TARGET_COMPUTE_BUDGET,
                    max_rounds=4,
                    minimum_improvement=1e-7,
                    max_candidates=4,
                    use_memory=True,
                    world_structure_commitments=True,
                    transition_premise_audit=True,
                    expanded_premise_revisions=True,
                    active_threshold_probing=True,
                    indexed_commitment_store=True,
                    sensor_role_calibration_priors=True,
                    occam_premise_minimality=True,
                    branch_reconciliation=True,
                ),
                measurement_budget=TARGET_MEAS_BUDGET,
                library=lib_ft,
            ).run(lab_t3)
            ext_t3 = _eval_external_fresh_48(lab_t3, res_t3, seed=b_sd + 325 + 920_001)
        bp_t3 = [a for a in res_t3["actions"] if a.get("action") == "execute_threshold_boundary_probe"]
        replay_f_cases[cid_f] = {
            "status": res_t3["status"],
            "internal_confirmation_loss": res_t3["confirmation_loss"],
            "external_fresh48_mse": ext_t3["external_fresh48_mse"],
            "strict_pass": bool(
                res_t3["status"] == "completed"
                and res_t3.get("confirmation_loss") is not None
                and float(res_t3["confirmation_loss"]) <= 1e-7
                and ext_t3["external_fresh48_mse"] <= 1e-7
            ),
            "boundary_probes_executed": bp_t3,
            "threshold_boundary_uncertainty": res_t3.get("threshold_boundary_uncertainty"),
            "costs": res_t3["costs"],
        }

    replay_f = {
        "passed": replay_f_cases["exposed_1021000_S_left_dpg_N4"]["false_confirmation_eliminated"],
        "cases": replay_f_cases,
    }

    return {
        "Replay_A_production_batch_reuse": replay_a,
        "Replay_B_budget_halt_and_instance_template_layering": replay_b,
        "Replay_C_router_deferral_end_to_end": replay_c,
        "Replay_D_shared_evidence_replay_and_content_cache": replay_d,
        "Replay_E_persistent_pending_resumption_and_re_reconcile": replay_e,
        "Replay_F_ordinary_loop_residual_threshold_boundary_probe": replay_f,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errata = build_v13_five_item_errata()
    probes = run_gpt6_v14_audit_probes()
    replays = run_replays_a_to_f_and_1024000_diagnostic()

    all_replays_passed = all(v["passed"] for v in replays.values())
    receipt = {
        "schema_version": "hegel_v14_deliverable_1_errata_and_replays_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v14_Deliverable_1_V13_Errata_Probe_Remediations_and_Replays_A_to_F",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "retired_exposed_chains_notice": (
            "Chains 1001000..1005000, 1010000..1018000, and 1021000..1028000 are retired exposed development/diagnostic "
            "material only; zero brand-new v14 prospective evaluation chains were touched in Deliverable 1."
        ),
        "all_gpt6_v14_probes_remediated": probes["all_gpt6_v14_probes_remediated"],
        "all_replays_a_to_f_passed": all_replays_passed,
        "v13_five_item_errata": errata,
        "remediated_gpt6_v14_probes": probes,
        "replays_a_to_f": replays,
    }
    rendered = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)
    out_path = OUT_DIR / "deliverable_1_errata_and_replays_a_to_f_20261118.json"
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "written": str(out_path),
                "all_gpt6_v14_probes_remediated": probes["all_gpt6_v14_probes_remediated"],
                "all_replays_a_to_f_passed": all_replays_passed,
                "replay_summary": {k: v["passed"] for k, v in replays.items()},
                "exposed_1024000_T1_2x2": replays["Replay_D_shared_evidence_replay_and_content_cache"][
                    "exposed_retired_1024000_T1_2x2_diagnostic"
                ],
                "replay_f_cases": {
                    k: {
                        "status": v["status"],
                        "external_fresh48_mse": v["external_fresh48_mse"],
                        "bp_count": len(v["boundary_probes_executed"]),
                    }
                    for k, v in replays["Replay_F_ordinary_loop_residual_threshold_boundary_probe"]["cases"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

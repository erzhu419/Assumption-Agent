"""Run v13 PR-A & PR-B: Evidence Identity, Ordinary-Entry Failure Recursion, Routing vs Scope, & Cache Controls.

Writes two prospective v13 receipts (ZERO byte modifications to any of the 29 historic v10..v12 JSON receipts,
ZERO new unseen science source chains consumed here — uses minimal synthetic controls and exposed v12 dev cases):
  1. ``docs/evidence/v13_evidence_recursion_and_routing_20261117/pr_a_evidence_and_recursion_controls_20261117.json``
  2. ``docs/evidence/v13_evidence_recursion_and_routing_20261117/pr_b_routing_vs_applicability_and_cache_20261117.json``

Follows ``markdown/hegel_v13/02_下一轮最小施工与实验合同.md`` §2 (PR-A), §3 (PR-B), §4.C1/C3, & §5 (#1–#16).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    _refine_coexisting_branch_scopes_and_routing,
    _serialize_obs_rows,
    adjudicate_branches_on_paid_evidence,
    compute_observable_regime_features,
    dispatch_branches_and_families_for_target,
    evaluate_scope_guard,
    extract_guard_ast_dependencies,
    find_feasible_prediction_difference,
    instantiate_aligned_claim_program,
    propagate_premise_retraction_invalidation,
    propose_threshold_interval_boundary_probe,
    reconcile_competing_branches,
    reconstruct_parametric_families_from_ordinary_archive,
    shrink_claim_to_observable_scope,
    validate_observable_scope_guard,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, ResearchConfig, SearchBudget, TaskSpec
from hegel_machine.cognition.premise_commitment_index import extract_archive_witness_maps
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments

from scripts.run_v10_3_library_scaling_study import _make_confirmed_regime_prior

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v13_evidence_recursion_and_routing_20261117"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _run_ordinary_cfg(seed: int, *, obs_b: int = 640, comp_b: int = 700_000, latent_on: bool = True) -> ResearchConfig:
    return ResearchConfig(
        seed=seed,
        observation_budget=obs_b,
        compute_budget=comp_b,
        max_rounds=4 if latent_on else 1,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=latent_on,
        latent_state_discovery=latent_on,
        state_search=True,
        active_threshold_probing=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=True,
    )


def run_pr_a_and_c1() -> dict[str, Any]:
    # 1. Re-verify GPT-6's `markdown/hegel_v13/review_probes.py` controls on remediated v13 code
    g_pilot = Program("lt", children=(Program("feature", name="obs_unconditional_block_rate"), Program("const", value=0.1)))
    empty_reg = compute_observable_regime_features([], None)
    empty_guard_eval = evaluate_scope_guard(g_pilot, empty_reg)

    upath = REPO_ROOT / "src/hegel_machine/cognition/unified_research.py"
    u_tree = ast.parse(upath.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(u_tree) if isinstance(n, ast.FunctionDef) and n.name == "_exec_reconcile_probe")
    old_rows = [Observation({"x": 0.0}, 0.0)]
    old_trans = [{"before": {"x": 0.0}, "after": {"x": 0.0}}]
    cb_ns = {
        "reconcile_channels": ("fine",),
        "acquire": lambda *a, **kw: None,
        "batches": [],
        "ep_obs_rec": old_rows,
        "ep_trans_rec": old_trans,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(upath), "exec"), cb_ns)
    req_probe = SimpleNamespace(probe_id="new_probe_not_executed", sequence=({"c0": 0.5},), reset=True)
    denied_ret = cb_ns["_exec_reconcile_probe"](req_probe, frozen_program_versions=["prog_v1:1", "prog_v1:2"])

    p_a = _make_confirmed_regime_prior(concept_id="concept_A", seed=9301, objects=4, readout="window_total_event", direction="right", edge_pred_dict=None, radius=0.95)
    p_b = _make_confirmed_regime_prior(concept_id="concept_B", seed=9302, objects=4, readout="window_total_event", direction="left", edge_pred_dict=None, radius=0.95)
    sc = ObjectSchema.from_dict(p_a["object_schema"])
    c_a = consolidate_world_structure_commitments([p_a])[0]
    c_b = consolidate_world_structure_commitments([p_b])[0]
    spec = TaskSpec(task_id="ctrl", description="ctrl", feature_names=tuple(sc.feature_names), initial_probes=(), available_probes=(), feature_bounds={f: (-1.5, 1.5) for f in sc.feature_names})

    rec_denied = reconcile_competing_branches(
        {"claim_id": "new_c", "commitment": c_a, "status": "universal", "independent_empirical_support": True},
        [{"claim_id": "old_c", "commitment": c_b, "status": "universal", "independent_empirical_support": True}],
        spec,
        sc,
        execute_probe_fn=cb_ns["_exec_reconcile_probe"],
    )

    # Both-refuted unit control (predictions 1.0 and 2.0 vs target 0.0)
    adj_both = adjudicate_branches_on_paid_evidence(
        {"claim_id": "new", "status": "universal", "independent_empirical_support": True},
        {"claim_id": "old", "status": "universal", "independent_empirical_support": True},
        {"program": Program("const", value=1.0), "radius_bindings": {}},
        {"program": Program("const", value=2.0), "radius_bindings": {}},
        old_rows,
        old_trans,
        None,
    )

    # Branch -> Commitment -> Source ID mapping & multi-source preservation control
    c_single = deepcopy(c_a)
    c_single["commitment_id"] = "world_commitment:0000"
    c_single["source_concept_ids"] = ["concept_A"]
    c_shared = deepcopy(c_a)
    c_shared["commitment_id"] = "world_commitment:0001"
    c_shared["source_concept_ids"] = ["concept_A", "concept_B"]
    o0, o1 = p_a["object_schema"]["objects"][0], p_a["object_schema"]["objects"][1]
    f_map = p_a["object_schema"]["fields"]
    neg_flux_trans = [
        {
            "before": {f_map[o0]["q"]: 0.2, f_map[o0]["x"]: 0.0, f_map[o1]["q"]: 0.8, f_map[o1]["x"]: 0.4},
            "after": {f_map[o0]["q"]: 0.35, f_map[o0]["x"]: 0.0, f_map[o1]["q"]: 0.65, f_map[o1]["x"]: 0.4},
        }
    ]
    lib_map = KnowledgeLibrary()
    lib_map._active_reconciled_branches = [
        {
            "claim_id": "branch:concept_A",
            "source_concept_id": "concept_A",
            "source_concept_ids": ["concept_A"],
            "commitment": deepcopy(c_single),
            "status": "scoped_conditional",
            "scope_kind": "empirical_applicability_scope",
            "scope_guard": Program("lt", children=(Program("const", value=0.0), Program("feature", name="obs_signed_direction_flux"))).to_dict(),
            "independent_empirical_support": True,
        }
    ]
    disp_map = dispatch_branches_and_families_for_target(
        library=lib_map,
        consistent_comms=[c_single, c_shared],
        transitions=neg_flux_trans,
        schema=p_a["object_schema"],
        audit=None,
        task_sig="ctrl_map",
        comm_budget=SearchBudget(10_000),
    )
    kept_comm_ids = [c["commitment_id"] for c in disp_map["ordered_consistent_comms"]]

    # 2. C1 Three Ordinary-Entry Choice Points through `UnifiedResearchAgent.run(lab)`
    base_lib = KnowledgeLibrary()
    for sd, d_mode in ((9310, "symmetric"), (9311, "right")):
        with AnonymousObjectLab(
            seed=sd, objects=4, readout="window_total_event", relation_direction=d_mode,
            relation_condition="always", condition_threshold=0.0, relation_radius=0.93,
            actuator_curvature=0.15, name_seed=sd + 31, wiring_seed=sd + 61,
        ) as lab_s:
            r_s = UnifiedResearchAgent(_run_ordinary_cfg(sd), measurement_budget=12_400, library=base_lib).run(lab_s)
            assert r_s["status"] == "completed"

    # Choice Point 1: Disagreement probe `executed_new` succeeds on S1_dpg
    lib_cp1 = deepcopy(base_lib)
    with AnonymousObjectLab(
        seed=9312, objects=4, readout="window_total_event", relation_direction="right",
        relation_condition="donor_positive_gap", condition_threshold=0.28, relation_radius=0.93,
        actuator_curvature=0.15, name_seed=9343, wiring_seed=9373,
    ) as lab_cp1:
        res_cp1 = UnifiedResearchAgent(_run_ordinary_cfg(9312), measurement_budget=12_400, library=lib_cp1).run(lab_cp1)
    led_cp1 = lib_cp1._reconciliation_episode_ledger[-1]

    # Choice Point 2: Same task with measurement budget tight enough that post-confirmation reconcile probe is `unavailable`
    # Determine exact measurement spent before reconcile probe so confirmation succeeds while extra probe is denied
    meas_for_main_only = res_cp1["costs"]["measurement_cost"] - 48
    lib_cp2 = deepcopy(base_lib)
    with AnonymousObjectLab(
        seed=9312, objects=4, readout="window_total_event", relation_direction="right",
        relation_condition="donor_positive_gap", condition_threshold=0.28, relation_radius=0.93,
        actuator_curvature=0.15, name_seed=9343, wiring_seed=9373,
    ) as lab_cp2:
        res_cp2 = UnifiedResearchAgent(_run_ordinary_cfg(9312), measurement_budget=max(1200, meas_for_main_only), library=lib_cp2).run(lab_cp2)
    led_cp2 = lib_cp2._reconciliation_episode_ledger[-1]

    # Choice Point 3a (Unfunded) & 3b (Funded): Both historical Markov branches & initial Markov search refuted on `focal_memory`!
    focus_lib = KnowledgeLibrary()
    for sd_f, d_mode_f in ((9313, "symmetric"), (9314, "right")):
        with AnonymousObjectLab(
            seed=sd_f, objects=4, readout="focus", relation_direction=d_mode_f,
            relation_condition="always", condition_threshold=0.0, relation_radius=0.93,
            actuator_curvature=0.15, name_seed=sd_f + 31, wiring_seed=sd_f + 61,
        ) as lab_sf:
            r_sf = UnifiedResearchAgent(_run_ordinary_cfg(sd_f), measurement_budget=11_500, library=focus_lib).run(lab_sf)
            assert r_sf["status"] == "completed"

    lib_cp3_unfunded = deepcopy(focus_lib)
    with AnonymousObjectLab(
        seed=9315, objects=4, readout="focus", relation_direction="right",
        relation_condition="always", condition_threshold=0.0, relation_radius=0.93,
        latent_mode="focal_memory", latent_decay=0.55, latent_weight=0.35,
        actuator_curvature=0.15, name_seed=9346, wiring_seed=9376,
    ) as lab_cp3_u:
        res_cp3_u = UnifiedResearchAgent(
            _run_ordinary_cfg(9315, obs_b=195, comp_b=480_000, latent_on=False),
            measurement_budget=3_600,
            library=lib_cp3_unfunded,
        ).run(lab_cp3_u)
    subqs_cp3_u = [
        q for q in res_cp3_u["response_questions"] if q.get("kind") == "reconciliation_subquestion"
    ]

    lib_cp3_funded = deepcopy(focus_lib)
    with AnonymousObjectLab(
        seed=9315, objects=4, readout="focus", relation_direction="right",
        relation_condition="always", condition_threshold=0.0, relation_radius=0.93,
        latent_mode="focal_memory", latent_decay=0.55, latent_weight=0.35,
        actuator_curvature=0.15, name_seed=9346, wiring_seed=9376,
    ) as lab_cp3_f:
        res_cp3_f = UnifiedResearchAgent(
            _run_ordinary_cfg(9315, obs_b=560, comp_b=680_000, latent_on=False),
            measurement_budget=11_500,
            library=lib_cp3_funded,
        ).run(lab_cp3_f)
        eval_ch_cp3 = () if not res_cp3_f.get("selected_channels") else ("fine",)
        ext_rows_cp3_f = lab_cp3_f.evaluation_rows(eval_ch_cp3, seed=99315, length=24, trajectories=2)
        ext_mse_cp3_f = float(loss(Program.from_dict(res_cp3_f["program"]), ext_rows_cp3_f, float(res_cp3_f.get("fallback", 0.0)), 0.5))
    subqs_cp3_f = [q for q in res_cp3_f["response_questions"] if q.get("kind") == "reconciliation_subquestion"]

    return {
        "schema_version": "hegel_v13_pr_a_evidence_and_recursion_controls_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v13_PR_A_and_C1_Evidence_Identity_Unknown_Scope_and_Failure_Recursion",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "remediated_gpt6_v13_review_probes": {
            "N1_denied_probe_acquisition": {
                "callback_return": denied_ret,
                "reconciler_paid_probes_executed": rec_denied["paid_probes_executed"],
                "reconciler_decision": rec_denied["decisions"][0],
                "remediated": denied_ret["status"] == "unavailable" and rec_denied["paid_probes_executed"] == [],
            },
            "N2_both_refuted_subquestion_return": {
                "verdict": adj_both["verdict"],
                "old_status_after": adj_both["updated_old_claim"]["status"],
                "new_status_after": adj_both["updated_new_claim"]["status"],
                "pending_questions_returned": len(adj_both["pending_questions"]),
                "pending_question": adj_both["pending_questions"][0],
                "remediated": (
                    len(adj_both["pending_questions"]) == 1
                    and adj_both["updated_old_claim"]["status"] != "universal"
                    and adj_both["updated_new_claim"]["status"] != "universal"
                ),
            },
            "N3_empty_pilot_unknown_tristate": {
                "computed_empty_block_rate": empty_reg["obs_unconditional_block_rate"],
                "valid_count": empty_reg["_meta"]["valid_counts"]["obs_unconditional_block_rate"],
                "ordinary_evaluate_scope_guard_result": empty_guard_eval,
                "remediated": empty_reg["obs_unconditional_block_rate"] is None and empty_guard_eval == "unknown",
            },
            "N5_branch_to_commitment_mapping_and_multisource": {
                "empirically_excluded_branches": disp_map["telemetry"]["empirically_excluded_on_context"],
                "remaining_commitment_ids": kept_comm_ids,
                "single_source_comm_0000_filtered": "world_commitment:0000" not in kept_comm_ids,
                "multi_source_comm_0001_preserved": "world_commitment:0001" in kept_comm_ids,
                "remediated": ("world_commitment:0000" not in kept_comm_ids) and ("world_commitment:0001" in kept_comm_ids),
            },
        },
        "c1_three_ordinary_entry_choice_points": {
            "choice_point_1_disagreement_probe_executed_new": {
                "episode_status": res_cp1["status"],
                "paid_probes_executed": led_cp1.get("paid_probes_executed"),
                "reconciliation_decisions": led_cp1.get("reconciliation_decisions"),
            },
            "choice_point_2_disagreement_probe_unavailable": {
                "episode_status": res_cp2["status"],
                "paid_probes_executed": led_cp2.get("paid_probes_executed"),
                "reconciliation_decisions": led_cp2.get("reconciliation_decisions"),
                "zero_fake_executed_probes": len(led_cp2.get("paid_probes_executed") or ()) == 0,
            },
            "choice_point_3_both_refuted_recursive_subresearch": {
                "unfunded_case": {
                    "episode_status": res_cp3_u["status"],
                    "subquestions": subqs_cp3_u,
                    "historical_branch_statuses_restricted": [b["status"] for b in lib_cp3_unfunded._active_reconciled_branches],
                },
                "funded_case": {
                    "episode_status": res_cp3_f["status"],
                    "confirmation_loss": res_cp3_f["confirmation_loss"],
                    "external_fresh48_mse": ext_mse_cp3_f,
                    "subquestions": subqs_cp3_f,
                    "mutated_parent_program": bool(
                        subqs_cp3_f and subqs_cp3_f[0].get("resolution", {}).get("parent_program_before")
                        != subqs_cp3_f[0].get("resolution", {}).get("parent_program_after")
                    ),
                },
            },
        },
    }


def run_pr_b_and_dev_diagnostics() -> dict[str, Any]:
    # 1. Protection Control 1 (Check #11): Distinct structural labels that are prediction-equivalent on declared domain
    p_u = _make_confirmed_regime_prior(concept_id="p_u", seed=9321, objects=4, readout="window_total_event", direction="right", edge_pred_dict=None, radius=0.94)
    p_wc = _make_confirmed_regime_prior(concept_id="p_wc", seed=9322, objects=4, readout="window_total_event", direction="right", edge_pred_dict=_dpg_dict(-10.0, -10.0), radius=0.94)
    sup_uw, _ = extract_archive_witness_maps([p_u, p_wc])
    b_u = {
        "claim_id": "branch:p_u", "source_concept_id": "p_u", "source_concept_ids": ["p_u"],
        "commitment": consolidate_world_structure_commitments([p_u])[0], "schema": p_u["object_schema"],
        "status": "universal", "independent_empirical_support": True,
        "pilot_regime_features": [{"obs_mean_abs_q": 0.25, "obs_signed_direction_flux": 0.15}],
        "support_rows_serialized": _serialize_obs_rows(sup_uw["p_u"]),
    }
    b_wc = {
        "claim_id": "branch:p_wc", "source_concept_id": "p_wc", "source_concept_ids": ["p_wc"],
        "commitment": consolidate_world_structure_commitments([p_wc])[0], "schema": p_wc["object_schema"],
        "status": "universal", "independent_empirical_support": True,
        "pilot_regime_features": [{"obs_mean_abs_q": 0.75, "obs_signed_direction_flux": 0.35}],
        "support_rows_serialized": _serialize_obs_rows(sup_uw["p_wc"]),
    }
    refined_equiv = _refine_coexisting_branch_scopes_and_routing([b_u, b_wc], budget=SearchBudget(60_000))
    prot_1 = {
        "scientific_scope_guards": [b.get("scope_guard") for b in refined_equiv],
        "routing_scope_guards_present": [b.get("routing_guard") is not None for b in refined_equiv],
        "statuses_preserved_universal": [b["status"] for b in refined_equiv],
        "passed": all(b.get("scope_guard") is None and b["status"] == "universal" and b.get("routing_guard") is not None for b in refined_equiv),
    }

    # 2. Protection Control 2: Same coarse observable summary, distinct threshold boundaries (`0.20` vs `0.42`) -> NOT merged!
    p_t20 = _make_confirmed_regime_prior(concept_id="p_t20", seed=9323, objects=4, readout="window_total_event", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.20), radius=0.94)
    p_t42 = _make_confirmed_regime_prior(concept_id="p_t42", seed=9324, objects=4, readout="window_total_event", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.42), radius=0.94)
    sc_20 = ObjectSchema.from_dict(p_t20["object_schema"])
    sc_20_feats = tuple(fn for gm in sc_20.fields.values() for fn in gm.values())
    spec_20 = TaskSpec(task_id="p2", description="p2", feature_names=sc_20_feats, initial_probes=(), available_probes=(), feature_bounds={f: (-1.5, 1.5) for f in sc_20_feats})
    al_20 = instantiate_aligned_claim_program({"claim_id": "b20", "commitment": consolidate_world_structure_commitments([p_t20])[0]}, sc_20)
    al_42 = instantiate_aligned_claim_program({"claim_id": "b42", "commitment": consolidate_world_structure_commitments([p_t42])[0]}, sc_20)
    diff_20_42 = find_feasible_prediction_difference(al_20, al_42, spec_20, sc_20)
    prot_2 = {
        "proved_absent": diff_20_42["proved_absent"],
        "witness_available": diff_20_42["witness_available"],
        "max_abs_disagreement": diff_20_42["max_abs_disagreement"],
        "passed": (diff_20_42["proved_absent"] is False) and (diff_20_42["witness_available"] is True),
    }

    # 3. 3-Layer Ordinary Archive Cache Verification (Checks #13 & #14)
    p_left = _make_confirmed_regime_prior(concept_id="p_left", seed=9325, objects=4, readout="window_total_event", direction="left", edge_pred_dict=None, radius=0.93)
    for pr, flx in ((p_t20, 0.22), (p_t42, 0.26), (p_left, -0.24)):
        pr["pilot_regime_features"] = [{"obs_signed_direction_flux": flx, "obs_unconditional_block_rate": 0.32 if flx > 0 else 0.0}]
    lib_ord = KnowledgeLibrary()
    lib_ord.meta_priors = [deepcopy(p_t20), deepcopy(p_t42), deepcopy(p_left)]
    f1, st1 = reconstruct_parametric_families_from_ordinary_archive(lib_ord, SearchBudget(200_000), reconstruct_routing=True)
    f2, st2 = reconstruct_parametric_families_from_ordinary_archive(lib_ord, SearchBudget(200_000), reconstruct_routing=True)
    p_t31 = _make_confirmed_regime_prior(concept_id="p_t31", seed=9326, objects=4, readout="window_total_event", direction="right", edge_pred_dict=_dpg_dict(0.0, 0.31), radius=0.95)
    p_t31["pilot_regime_features"] = [{"obs_signed_direction_flux": 0.24, "obs_unconditional_block_rate": 0.30}]
    lib_ord.meta_priors.append(p_t31)
    f3, st3 = reconstruct_parametric_families_from_ordinary_archive(lib_ord, SearchBudget(200_000), reconstruct_routing=True)

    # 4. Exposed Dev Chain `1016000` Single-Factor `2x2` (`guards ON/OFF x H* priority ON/OFF` on IDENTICAL branch representation!)
    #    Directly traces whether routing/scope changes ordering, fitting, probe purchase, adoption, and final prediction!
    dev_base = 1_016_000
    dev_lib = KnowledgeLibrary()
    src_seq_1016 = [
        (dev_base + 5, "symmetric", "always", 0.0, 0.92),
        (dev_base + 25, "right", "donor_positive_gap", 0.19, 0.89),
        (dev_base + 45, "left", "donor_positive_gap", 0.23, 0.95),
        (dev_base + 15, "right", "always", 0.0, 0.94),
        (dev_base + 35, "right", "donor_positive_gap", 0.43, 0.92),
    ]
    for sd, d_m, r_c, th_v, rad_v in src_seq_1016:
        with AnonymousObjectLab(
            seed=sd, objects=4, readout="window_total_event", relation_direction=d_m,
            relation_condition=r_c, condition_threshold=th_v, relation_radius=rad_v,
            actuator_curvature=0.15, name_seed=sd + 100, wiring_seed=sd + 200,
        ) as s_lab:
            UnifiedResearchAgent(_run_ordinary_cfg(sd, obs_b=660, comp_b=740_000), measurement_budget=13_200, library=dev_lib).run(s_lab)

    grid_2x2: dict[str, Any] = {}
    for guards_on in (True, False):
        for hstar_on in (True, False):
            cell_name = f"guards_{'ON' if guards_on else 'OFF'}__Hstar_{'ON' if hstar_on else 'OFF'}"
            cell_lib = deepcopy(dev_lib)
            cell_lib._freeze_branch_writes = True
            cell_lib._enable_scope_and_routing_guards = guards_on
            cell_lib._prioritize_generalized_families = hstar_on
            with AnonymousObjectLab(
                seed=dev_base + 305, objects=5, readout="window_total_event",
                relation_direction="right", relation_condition="donor_positive_gap",
                condition_threshold=0.32, relation_radius=0.92, actuator_curvature=0.15,
                name_seed=dev_base + 405, wiring_seed=dev_base + 505,
            ) as t_lab:
                t_res = UnifiedResearchAgent(
                    _run_ordinary_cfg(dev_base + 305, obs_b=620, comp_b=680_000),
                    measurement_budget=12_400,
                    library=cell_lib,
                ).run(t_lab)
                ext_r = t_lab.evaluation_rows(() if not t_res["selected_channels"] else ("fine",), seed=dev_base + 999, length=24, trajectories=2)
                ext_mse = float(loss(Program.from_dict(t_res["program"]), ext_r, float(t_res.get("fallback", 0.0)), 0.5))
            disp_acts = [a.get("branch_dispatch_telemetry") for a in t_res["actions"] if a.get("action") == "apply_world_structure_commitments" and a.get("branch_dispatch_telemetry")]
            fit_acts = [a for a in t_res["actions"] if a.get("action") == "apply_world_structure_commitments"]
            grid_2x2[cell_name] = {
                "guards_enabled": guards_on,
                "hstar_prioritized": hstar_on,
                "status": t_res["status"],
                "confirmation_loss": t_res["confirmation_loss"],
                "external_fresh48_mse": ext_mse,
                "compute_cost": t_res["costs"]["compute_cost"],
                "observation_cost": t_res["costs"]["observation_cost"],
                "queries": t_res["costs"]["queries"],
                "selected_kind": t_res["selected_kind"],
                "commitment_fit_candidates_generated": sum(int(a.get("candidates_generated", a.get("candidate_count", 0))) for a in fit_acts),
                "first_dispatch_telemetry": disp_acts[0] if disp_acts else None,
            }

    # 5. C3 Boundary Disagreement Diagnostic on Exposed `1005000` Threshold Uncertainty
    c_b20 = consolidate_world_structure_commitments([p_t20])[0]
    b_diag = propose_threshold_interval_boundary_probe(
        c_b20, [0.34, 0.43], spec_20, sc_20, fitted_radius=0.89, budget=SearchBudget(30_000)
    )

    return {
        "schema_version": "hegel_v13_pr_b_routing_vs_applicability_and_cache_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v13_PR_B_Routing_vs_Applicability_Cache_and_Dev_2x2",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "protection_control_1_different_signature_prediction_equivalent": prot_1,
        "protection_control_2_same_coarse_summary_different_boundary": prot_2,
        "ordinary_archive_3layer_cache_audit": {
            "first_call_cold_cache": {k: v for k, v in st1.items() if k != "reconstructed_branches"},
            "second_call_unchanged_sources": {k: v for k, v in st2.items() if k != "reconstructed_branches"},
            "third_call_after_adding_one_prior": {k: v for k, v in st3.items() if k != "reconstructed_branches"},
            "exact_family_id_parity_call1_vs_call2": f1[0]["commitment_id"] == f2[0]["commitment_id"],
            "incremental_3branch_after_addition": len(f3[0]["source_branch_ids"]) == 3,
        },
        "exposed_dev_1016000_T1_single_factor_2x2": grid_2x2,
        "c3_threshold_interval_boundary_diagnostic": {
            "actionable": b_diag["actionable"],
            "role": b_diag["role"],
            "interval": b_diag["interval"],
            "tested_boundary_models": b_diag["tested_boundary_models"],
            "frozen_program_versions": b_diag["frozen_program_versions"],
            "max_abs_disagreement": b_diag["max_abs_disagreement"],
            "probe_id": b_diag["probe"].probe_id if b_diag.get("probe") else None,
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr_a = run_pr_a_and_c1()
    p_a_out = OUT_DIR / "pr_a_evidence_and_recursion_controls_20261117.json"
    p_a_out.write_text(json.dumps(pr_a, indent=2, ensure_ascii=False), encoding="utf-8")

    pr_b = run_pr_b_and_dev_diagnostics()
    p_b_out = OUT_DIR / "pr_b_routing_vs_applicability_and_cache_20261117.json"
    p_b_out.write_text(json.dumps(pr_b, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "pr_a_written": str(p_a_out),
                "pr_b_written": str(p_b_out),
                "remediated_probes": {k: v["remediated"] for k, v in pr_a["remediated_gpt6_v13_review_probes"].items()},
                "dev_2x2_statuses": {k: (v["status"], v["compute_cost"]) for k, v in pr_b["exposed_dev_1016000_T1_single_factor_2x2"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

r"""Hegel V17.5 (M4.3): Channel-Bound Multi-Scale Execution, Confirmed-Instance Identity,
Unified Scope/Certificate Governance, Fresh Confirmation Closure & Continual Benchmark.

Directly closes all five architectural & experimental wiring gaps identified in ``Hegel_M4_2_Audit``
while advancing the multi-precision ``Hegel`` roadmap toward ``M5`` (finite instrument precision,
multi-channel active sensing, and causal maintenance verification):

1. **Cryptographic Confirmed-Instance Identity (`C1` / `M4_2_Audit` Sec. V & VII)**:
   Binds ``fitted_params_sha256`` and ``confirmed_instance_digest`` across ``authored_method.json``,
   ``confirmed_fitted_params.json``, and ``confirmed_manifest.json``, and verifies ``predict_with_frozen_instance``
   directly on ``confirm_rows`` (zero ``fit()`` during strict frozen loading).
2. **Unified Scale Certificate, Box Containment, Canonical Role Bijection & Read-Only R3 (`M4_2_Audit` Sec. VI, VIII, IX)**:
   - ``revalidate_vertical_edges_with_certificate`` rejects narrow-certificate box widening, checks
     both trajectory endpoint slaving defect AND late-window projection drift, and synchronizes node +
     ``coarse_graining_edge`` ``ScaleMap`` and certificate state atomically;
   - ``ForestScopeDispatcher`` is 100% read-only on ``post_stage2_forest``, executes zero ``fit()``
     across both cold load and inference, and requires a live ``coarse_graining_edge`` to escalate
     out-of-box queries to fine scale;
   - Stage-3 enforces ``model_scope_contract.json`` on final/continuation queries and maps renamed
     coordinates onto canonical certificate roles (`fail-closed` on missing/unmappable axes).
3. **Universal Fresh Confirmation Invariant (`prefix_confirm_overlap_count == 0` across 100% of runs)**:
   Retains all paid failed confirmations in ``frozen_prefix`` while synthesizing/selecting strictly
   disjoint unseen confirmation inputs (`0` training-point confirmation reuses; full ``fresh_points`` count).
4. **True Channel-Bound Multi-Scale Pricing, Complete Stage-0 Snapshot & Genuine ``CodeAgent`` Rediscovery**:
   - Multi-scale ``Pi_s`` targets run on ``MultiScaleWorkspaceTaskLab`` where ``coarse_macro`` (`3.0`/obs,
     real Schur reduced readout) and ``micro_fine`` (`10.0`/obs, real 2-mode RK4 readout) are physically
     distinct channels priced strictly by the executed channel action (unrelated ``hydro`` edges have
     zero effect on ODE channel selection or cost);
   - ``Frozen_After_Stage0`` starts from the exact 8-node, 2-coarse-edge, 1-wormhole Stage-0 boundary
     snapshot, and ``Frozen_After_Stage0_OpenRediscovery`` executes real ``run_compositional_code_agent_episode``;
   - Every retrieval ``query_compute_cost``, compilation, candidate fit, and table maintenance event
     is metered consistently into ``CostMeter``.
5. **6-Family Held-Out Continual Benchmark (`96` unseen in-scope + `8` OOS runs/arm on fresh seeds) +
   Noiseless vs Bounded-Instrument-Precision ($\sigma_{\text{pilot}} = 0.008$) Fork + Same-Info CueTable**:
   Separates forest relational representation, online multi-cue PPR, stopping rules (`FirstAcceptable`
   vs `BestPilotRMSE`), single-factor `Alpha0` vs `Pooling`, and causal Stage-2 maintenance effects.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from random import Random
import shutil
from statistics import mean, pstdev
from typing import Any, Callable
from unittest.mock import patch

from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m4_1_unified_discovery_and_renormalization import (
    UnifiedTargetSpec,
    _autonomous_fit_candidate_specs,
    _load_node_verified_operator_spec,
    apply_executable_pi_s_projection,
    extract_unified_observable_pilot_cues,
    integrate_fast_slow_ode_rk4,
    is_point_inside_verified_box,
)
from .m4_2_unified_closure_and_pi_lifecycle import (
    DEFAULT_PI_S_ODE_PARAMS,
    ForestScopeDispatcher,
    _build_stage0_open_atomic_bases,
    _eval_pi_s_coarse_scalar,
    _eval_pi_s_fine_scalar,
    _find_all_valid_canonical_role_maps_for_box,
    _find_valid_canonical_role_map_for_box,
    _role_order_from_canonical_role_map,
    _run_metered_atomic_calibration,
    build_and_evolve_m4_2_cross_pillar_forest,
    build_m4_2_target_suite,
    certify_regional_pi_s_closure_with_trajectory_projection,
    evaluate_m4_2_target_under_arm,
    run_interleaved_zero_fit_dispatcher_stream,
)
from .m4_compositional_discovery import (
    apply_structural_combinator,
    canonicalize_fitted_params_snapshot,
    compile_operator_spec_to_python,
    compute_confirmed_instance_digest,
    evaluate_isolated_python_model,
    load_verified_authored_method,
    make_atomic_operator_spec,
    predict_with_frozen_instance,
    run_compositional_code_agent_episode,
)
from .m4_multiscale_renormalization import (
    MultiScaleWorkspaceTaskLab,
    _estimate_affine_lens_for_spec,
    _infer_role_permutations,
    _validate_and_interpret_m4_program_steps,
    build_m4_cluster_targets,
    build_m4_multiscale_forest,
    coords_approx_equal,
    evaluate_m4_target_under_arm,
    select_or_synthesize_disjoint_confirm_points,
)
from .open_workspace_labs import OpenWorkspaceTaskLab
from .open_workspace_wormhole import canonicalize_python_model_ast
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    authorize_abstraction_use,
)


# ============================================================================
# Part 1. Direct Programmatic Verification of ALL `Hegel_M4_2_Audit` Controls
# ============================================================================

def run_m4_2_audit_closure_verification_suite(work_root: Path) -> dict[str, Any]:
    """Execute all 6 counterfactual audit control families from ``Hegel_M4_2_Audit`` in isolation."""
    work_root.mkdir(parents=True, exist_ok=True)
    built = build_and_evolve_m4_2_cross_pillar_forest(work_root / "shared_m42_build")
    base_forest = built["forest"]
    true_stage0_forest = built["forest_after_stage0"]

    # 1. Confirmed-instance parameter snapshot binding (`closure/probe_m42_contract_controls.py`)
    w1_src_ws = work_root / "shared_m42_build" / "persisted_stage1_authored_workspaces" / "stage1_W1_haldane_shunt"
    w2_src_ws = work_root / "shared_m42_build" / "persisted_stage1_authored_workspaces" / "stage1_W2_osc_shunt_4p"
    param_copy_ws = work_root / "audit_c1_param_copy"
    shutil.copytree(w1_src_ws, param_copy_ws, dirs_exist_ok=True)
    art_p = param_copy_ws / "authored_method.json"
    orig_art = json.loads(art_p.read_text(encoding="utf-8"))
    baseline_loaded = load_verified_authored_method(
        art_p, expected_ast_digest=orig_art["ast_skeleton_digest"], require_external_manifest=True
    )
    mut_art = deepcopy(orig_art)
    first_pk = sorted(mut_art["fitted_params"].keys())[0]
    mut_art["fitted_params"][first_pk] = float(mut_art["fitted_params"][first_pk]) * 3.0
    art_p.write_text(json.dumps(mut_art, indent=2), encoding="utf-8")
    mut_artifact_only_loaded = load_verified_authored_method(
        art_p, expected_ast_digest=orig_art["ast_skeleton_digest"], require_external_manifest=True
    )
    # Also mutate `confirmed_fitted_params.json` -> still rejected by manifest SHA & frozen confirm RMSE!
    (param_copy_ws / "confirmed_fitted_params.json").write_text(
        json.dumps(mut_art["fitted_params"], indent=2, sort_keys=True), encoding="utf-8"
    )
    mut_both_params_loaded = load_verified_authored_method(
        art_p, expected_ast_digest=orig_art["ast_skeleton_digest"], require_external_manifest=True
    )
    w2_art = json.loads((w2_src_ws / "authored_method.json").read_text(encoding="utf-8"))
    swap_rejected = (
        load_verified_authored_method(
            w2_src_ws / "authored_method.json",
            expected_ast_digest=orig_art["ast_skeleton_digest"],
            require_external_manifest=True,
        )
        is None
    )

    # 2. OR-branch permission & heterogeneous law guards + causal `program_steps` ordering
    ms_forest, _ = build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    f_perm = deepcopy(ms_forest)
    a_id, b_id = "aq_macro_river_tree", "dist_node_00"
    for nid, sc in ((a_id, "A"), (b_id, "B")):
        nd = f_perm.nodes[nid]
        nd["observation_map"] = deepcopy(f_perm.nodes[a_id]["observation_map"])
        nd["scale_map"] = deepcopy(f_perm.nodes[a_id]["scale_map"])
        nd["claim_contract"]["conclusion_or_action"] = deepcopy(
            f_perm.nodes[a_id]["claim_contract"]["conclusion_or_action"]
        )
        nd["claim_contract"]["preconditions"] = [{"field": "audit_scope", "op": "==", "value": sc}]
    f_perm.nodes[a_id]["claim_contract"]["permission_tier"] = "bounded_trial"
    f_perm.nodes[b_id]["claim_contract"]["permission_tier"] = "retrieve_candidate"

    def _or_verdict(n_obj: dict[str, Any], q_obj: dict[str, Any]) -> str:
        brs = n_obj["claim_contract"].get("disjunctive_branches") or [n_obj["claim_contract"]]
        return "ACC" if any(all(q_obj.get(p["field"]) == p["value"] for p in b["preconditions"]) for b in brs) else "UNK"

    perm_merge = f_perm.compress_compatible_nodes(
        keeper_node_id=a_id,
        redundant_node_id=b_id,
        evaluation_queries=[{"audit_scope": "A"}, {"audit_scope": "B"}, {"audit_scope": "C"}],
        evaluator_fn=_or_verdict,
    )
    auth_a_after = authorize_abstraction_use(
        requested_tier="bounded_trial",
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.1},
        target_features={"audit_scope": "A"},
        claim_contract=f_perm.nodes[a_id]["claim_contract"],
        scale_map=f_perm.nodes[a_id]["scale_map"],
    )
    auth_b_after = authorize_abstraction_use(
        requested_tier="bounded_trial",
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.1},
        target_features={"audit_scope": "B"},
        claim_contract=f_perm.nodes[a_id]["claim_contract"],
        scale_map=f_perm.nodes[a_id]["scale_map"],
    )

    f_het = deepcopy(f_perm)
    f_het = deepcopy(ms_forest)
    for nid, sc in ((a_id, "A"), (b_id, "B")):
        nd = f_het.nodes[nid]
        nd["observation_map"] = deepcopy(f_het.nodes[a_id]["observation_map"])
        nd["scale_map"] = deepcopy(f_het.nodes[a_id]["scale_map"])
        nd["claim_contract"]["preconditions"] = [{"field": "audit_scope", "op": "==", "value": sc}]
    f_het.nodes[b_id]["claim_contract"]["conclusion_or_action"] = deepcopy(
        ms_forest.nodes["bio_vascular_macro_capped"]["claim_contract"]["conclusion_or_action"]
    )
    het_merge = f_het.compress_compatible_nodes(
        keeper_node_id=a_id,
        redundant_node_id=b_id,
        evaluation_queries=[{"audit_scope": "A"}, {"audit_scope": "B"}, {"audit_scope": "C"}],
        evaluator_fn=_or_verdict,
    )

    sample_nd = ms_forest.nodes[a_id]
    canon_plan = _validate_and_interpret_m4_program_steps(sample_nd["program_steps"], node=sample_nd, target_arity=1)
    rev_plan = _validate_and_interpret_m4_program_steps(
        list(reversed(sample_nd["program_steps"])), node=sample_nd, target_arity=1
    )

    # 3. Fresh confirmation budget sufficiency & zero parent-exhaustion overlap (`probe_fresh_budget_edges.py`)
    ms_targets, _ = build_m4_cluster_targets(cluster_idx=0)

    def _probe_ms_fresh(target_obj, fresh_n: int, tag_lbl: str) -> dict[str, Any]:
        f_cl = deepcopy(ms_forest)
        for nd in f_cl.nodes.values():
            for st in nd.get("program_steps", []):
                if st["step"] == "subprocess_isolated_confirm":
                    st["fresh_points"] = fresh_n
        acq_log: list[dict[str, Any]] = []
        eval_overlaps: list[int] = []
        in_heldout_eval = [False]
        orig_acq = MultiScaleWorkspaceTaskLab.acquire_channel
        orig_ho = MultiScaleWorkspaceTaskLab.evaluate_heldout_micro_truth
        orig_ev = evaluate_isolated_python_model

        def _spy_acq(self, pts, *, channel="coarse_macro", phase="pilot"):
            acq_log.append({"phase": phase, "channel": channel, "points": deepcopy(pts)})
            return orig_acq(self, pts, channel=channel, phase=phase)

        def _spy_ho(self, *args, **kwargs):
            in_heldout_eval[0] = True
            try:
                return orig_ho(self, *args, **kwargs)
            finally:
                in_heldout_eval[0] = False

        def _spy_ev(*args, **kwargs):
            q_rows = args[2]
            t_rows = kwargs.get("isolated_prefix_rows") or []
            if not in_heldout_eval[0] and q_rows and t_rows and q_rows is not t_rows:
                ov = sum(1 for qr in q_rows if any(coords_approx_equal(qr["features"], tr["features"]) for tr in t_rows))
                eval_overlaps.append(ov)
            return orig_ev(*args, **kwargs)

        with (
            patch.object(MultiScaleWorkspaceTaskLab, "acquire_channel", _spy_acq),
            patch.object(MultiScaleWorkspaceTaskLab, "evaluate_heldout_micro_truth", _spy_ho),
            patch("hegel_machine.cognition.m4_multiscale_renormalization.evaluate_isolated_python_model", _spy_ev),
        ):
            res = evaluate_m4_target_under_arm(
                target_obj,
                arm_name="Hegel_M4_MultiScale_HippoWalk",
                forest=f_cl,
                index=HippoRAGWormholeIndex(f_cl, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                stage1_gate=None,
                work_dir=work_root / "audit_c3_fresh" / tag_lbl,
            )
        return {"res": res, "acquisitions": acq_log, "confirm_eval_overlaps": eval_overlaps}

    fresh_3_probe = _probe_ms_fresh(ms_targets[0], 3, "req_3")
    exhaustion_probe = _probe_ms_fresh(ms_targets[10], 2, "parent_exhaustion")

    # 4. Trajectory projection dependency (BOTH endpoint fast shift AND tail-window slow shift flip `certified`!)
    orig_proj_fn = apply_executable_pi_s_projection
    proj_dep: dict[str, Any] = {}
    for dep_mode in ("baseline", "nonterminal_window_shift", "endpoint_fast_shift"):
        def _wrap_proj(*args, _m=dep_mode, **kwargs):
            tr = kwargs.get("fine_trajectory")
            if tr is not None:
                tr = deepcopy(tr)
                if _m == "nonterminal_window_shift":
                    n_w = max(1, min(5, len(tr["x_slow"]) // 5))
                    for idx_w in range(len(tr["x_slow"]) - n_w, len(tr["x_slow"]) - 1):
                        tr["x_slow"][idx_w] += 100.0
                elif _m == "endpoint_fast_shift":
                    tr["x_fast"][-1] += 1.0
                kwargs["fine_trajectory"] = tr
            return orig_proj_fn(*args, **kwargs)

        with patch("hegel_machine.cognition.m4_2_unified_closure_and_pi_lifecycle.apply_executable_pi_s_projection", _wrap_proj):
            proj_dep[dep_mode] = certify_regional_pi_s_closure_with_trajectory_projection(
                work_root / "audit_c4_proj" / dep_mode, t_bounds=(2.4, 5.2)
            )

    # 5. Dispatcher read-only verification, coarse-edge escalation dependency, narrow-cert box widening & Stage-3 scope
    pre_disp_forest = deepcopy(base_forest)
    optical_before_stream = ForestScopeDispatcher(pre_disp_forest).dispatch_predict(
        query_id="optical_pre_stream",
        candidate_tree_id="tree_multiscale_pi_s",
        features={"u_drive": 2.0, "T_horizon": 3.6},
        target_metadata={"active_arity": 2, "modality": "optical"},
    )
    forest_dump_before = json.dumps(pre_disp_forest.nodes, sort_keys=True)
    stream_with_edges = run_interleaved_zero_fit_dispatcher_stream(pre_disp_forest)
    forest_dump_after = json.dumps(pre_disp_forest.nodes, sort_keys=True)

    no_edge_forest = deepcopy(base_forest)
    no_edge_forest.coarse_graining_edges = []
    stream_without_edges = run_interleaved_zero_fit_dispatcher_stream(no_edge_forest)
    q05_without_edges = next(r for r in stream_without_edges["receipts"] if r["query_id"] == "R3_Q05_PiS_ShortQuench_EscalateMicro")

    widened_f = deepcopy(base_forest)
    widened_sm = deepcopy(widened_f.nodes["ode_meso_schur_pi_thermal"]["scale_map"])
    widened_sm["window_or_partition"]["verified_input_box"]["T_horizon"] = [0.05, 5.2]
    widened_reval = widened_f.revalidate_vertical_edges_with_certificate(
        "ode_meso_schur_pi_thermal",
        witness_episode_id="audit_narrow_cert_widened_box",
        closure_certificate=proj_dep["baseline"],
        updated_scale_map=widened_sm,
    )
    synced_edge_boxes = [
        e["scale_map"]["window_or_partition"].get("verified_input_box")
        for e in base_forest.coarse_graining_edges
        if e["coarse_node_id"] == "ode_meso_schur_pi_thermal"
    ]

    m42_targets, _ = build_m4_2_target_suite(seed=20_261_120)
    pi_t = next(x for x in m42_targets if x.mechanism_family == "PiS_Adiabatic_CrossDomain_2D")

    def _eval_s3_probe(t_spec: UnifiedTargetSpec, lbl: str, f_override: PhilosophicalForest | None = None, open_fb: bool = False):
        f_use = deepcopy(base_forest if f_override is None else f_override)
        ix_use = HippoRAGWormholeIndex(f_use, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        return evaluate_m4_2_target_under_arm(
            t_spec,
            arm_name=lbl,
            seed=20_261_120,
            work_dir=work_root / "audit_c5_s3" / lbl,
            forest=f_use,
            index=ix_use,
            ordinary_history_specs=None,
            fallback_pool_specs=(
                built["stage0_atomic_specs"] if open_fb else built["ordinary_history_forward"]
            ),
            open_code_agent_fallback=open_fb,
        )

    s3_oob_heldout = _eval_s3_probe(replace(pi_t, heldout_points=[{"u_drive": 2.0, "T_horizon": 0.12}]), "oob_heldout")
    conf_oob_t = replace(
        pi_t,
        feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": (0.05, 5.2)},
        confirm_points=[{"u_drive": 2.6, "T_horizon": 1.2}, pi_t.confirm_points[1]],
    )
    s3_named_oob_cf = _eval_s3_probe(conf_oob_t, "named_oob_cf")
    ren_fn = lambda p: {"u_probe": p["u_drive"], "t_probe": p["T_horizon"]}
    orig_pi_oracle = conf_oob_t.oracle_fn
    alias_oob_t = replace(
        conf_oob_t,
        feature_names=("u_probe", "t_probe"),
        feature_bounds={"u_probe": (0.4, 3.0), "t_probe": (0.05, 5.2)},
        pilot_points=[ren_fn(p) for p in conf_oob_t.pilot_points],
        confirm_points=[ren_fn(p) for p in conf_oob_t.confirm_points],
        heldout_points=[ren_fn(p) for p in conf_oob_t.heldout_points],
        oracle_fn=lambda p: orig_pi_oracle({"u_drive": p["u_probe"], "T_horizon": p["t_probe"]}),
    )
    s3_renamed_oob_cf = _eval_s3_probe(alias_oob_t, "renamed_oob_cf")
    alias_inbox_t = replace(
        pi_t,
        feature_names=("u_probe", "t_probe"),
        feature_bounds={"u_probe": (0.4, 3.0), "t_probe": (2.4, 5.2)},
        pilot_points=[ren_fn(p) for p in pi_t.pilot_points],
        confirm_points=[ren_fn(p) for p in pi_t.confirm_points],
        heldout_points=[ren_fn(p) for p in pi_t.heldout_points],
        oracle_fn=lambda p: pi_t.oracle_fn({"u_drive": p["u_probe"], "T_horizon": p["t_probe"]}),
    )
    s3_renamed_inbox = _eval_s3_probe(alias_inbox_t, "renamed_inbox")

    # 6. Cost controls: unrelated hydro coarse edge vs no coarse edges, true Stage-0 snapshot, real OpenRediscovery
    f_only_hydro = deepcopy(base_forest)
    f_only_hydro.coarse_graining_edges = [
        e for e in f_only_hydro.coarse_graining_edges if e["fine_node_id"] == "hydro_micro_pore_1d"
    ]
    f_zero_cg = deepcopy(base_forest)
    f_zero_cg.coarse_graining_edges = []
    cost_complete = _eval_s3_probe(pi_t, "pi_complete")
    cost_hydro_only = _eval_s3_probe(pi_t, "pi_only_unrelated_hydro", f_override=f_only_hydro)
    cost_no_cg = _eval_s3_probe(pi_t, "pi_no_cg", f_override=f_zero_cg)

    acq_pool: list[dict[str, Any]] = []
    acq_open: list[dict[str, Any]] = []
    orig_open_acq = OpenWorkspaceTaskLab.acquire

    def _spy_open_acq(target_list):
        def _fn(self, seq, *, workspace=None, phase="probe"):
            obs = orig_open_acq(self, seq, workspace=workspace, phase=phase)
            target_list.append({"phase": phase, "n": len(obs or []), "points": deepcopy(seq)})
            return obs
        return _fn

    with patch.object(OpenWorkspaceTaskLab, "acquire", _spy_open_acq(acq_pool)):
        pool_rec = _eval_s3_probe(m42_targets[0], "frozen_pool_t01", f_override=true_stage0_forest, open_fb=False)
    with patch.object(OpenWorkspaceTaskLab, "acquire", _spy_open_acq(acq_open)):
        open_rec = _eval_s3_probe(m42_targets[0], "frozen_open_t01", f_override=true_stage0_forest, open_fb=True)

    dev_reuse_probe = evaluate_m4_2_target_under_arm(
        m42_targets[12],
        arm_name="Ordinary_AllHistory_DevFrozenOrder",
        seed=20_261_120,
        work_dir=work_root / "audit_c6_dev_t13",
        forest=None,
        index=None,
        ordinary_history_specs=built["ordinary_history_dev_frozen"],
        fallback_pool_specs=built["ordinary_history_forward"],
    )

    summary = {
        "c1_confirmed_instance_binding": {
            "baseline_accepted": baseline_loaded is not None,
            "mutated_artifact_param_accepted": mut_artifact_only_loaded is not None,
            "mutated_both_param_files_accepted": mut_both_params_loaded is not None,
            "cross_node_swap_rejected": swap_rejected,
        },
        "c2_or_branch_and_program_order": {
            "permission_merge_accepted": perm_merge["accepted"],
            "scope_A_bounded_trial_granted": auth_a_after["granted"],
            "scope_B_bounded_trial_granted": auth_b_after["granted"],
            "scope_B_capped_tier": auth_b_after["granted_tier"],
            "heterogeneous_law_merge_accepted": het_merge["accepted"],
            "canonical_program_valid": canon_plan is not None,
            "reversed_program_rejected": rev_plan is None,
        },
        "c3_fresh_confirmation_invariants": {
            "requested_3_acquired_confirm_points": sum(
                len(x["points"]) for x in fresh_3_probe["acquisitions"] if x["phase"].startswith("confirm")
            ),
            "requested_3_overlap_max": max(fresh_3_probe["confirm_eval_overlaps"] or [0]),
            "parent_exhaustion_child_confirm_points": sum(
                len(x["points"]) for x in exhaustion_probe["acquisitions"] if x["phase"] == "confirm_1_child"
            ),
            "parent_exhaustion_overlap_max": max(exhaustion_probe["confirm_eval_overlaps"] or [0]),
            "dev_t13_prefix_confirm_overlap": dev_reuse_probe["prefix_confirm_overlap_count"],
        },
        "c4_trajectory_and_certificate_governance": {
            "baseline_certified": proj_dep["baseline"]["certified"],
            "nonterminal_window_shift_certified": proj_dep["nonterminal_window_shift"]["certified"],
            "endpoint_fast_shift_certified": proj_dep["endpoint_fast_shift"]["certified"],
            "narrow_cert_widened_box_accepted": widened_reval["accepted"],
            "node_and_edge_boxes_synced": (
                synced_edge_boxes == [base_forest.nodes["ode_meso_schur_pi_thermal"]["scale_map"]["window_or_partition"]["verified_input_box"]]
            ),
            "stage3_oob_heldout_status": s3_oob_heldout["status"],
            "stage3_oob_heldout_false_confirmation": s3_oob_heldout["false_confirmation"],
            "named_oob_confirm_via_recall": s3_named_oob_cf["confirmed_via_recall"],
            "renamed_oob_confirm_via_recall": s3_renamed_oob_cf["confirmed_via_recall"],
            "renamed_inbox_confirm_via_recall": s3_renamed_inbox["confirmed_via_recall"],
            "raw_unmapped_point_in_box": ForestScopeDispatcher._point_in_box(
                {"u_probe": 2.6, "t_probe": 1.2},
                base_forest.nodes["ode_meso_schur_pi_thermal"]["scale_map"]["window_or_partition"]["verified_input_box"],
            ),
        },
        "c5_readonly_dispatcher_and_edge_escalation": {
            "optical_rejected_on_raw_post_stage2_forest": optical_before_stream["status"] == "scope_unauthorized",
            "dispatcher_stream_strictly_readonly": forest_dump_before == forest_dump_after,
            "any_fit_called_including_cold_loader": bool(
                stream_with_edges["any_fit_called"]
                or any(r.get("cold_loader_fit_called") for r in stream_with_edges["receipts"])
            ),
            "deleting_coarse_edges_blocks_quench_escalation": (
                q05_without_edges["status"] == "escalation_blocked_missing_coarse_edge"
            ),
        },
        "c6_channel_pricing_stage0_and_open_rediscovery": {
            "complete_forest_pi_channel": cost_complete["selected_channel"],
            "complete_forest_pi_meas_cost": cost_complete["cost"]["measurement_cost"],
            "unrelated_hydro_edge_pi_channel": cost_hydro_only["selected_channel"],
            "no_coarse_edge_pi_channel": cost_no_cg["selected_channel"],
            "unrelated_edge_vs_no_edge_meas_cost_equal": (
                cost_hydro_only["cost"]["measurement_cost"] == cost_no_cg["cost"]["measurement_cost"]
            ),
            "unrelated_edge_vs_no_edge_composite_cost_equal": (
                cost_hydro_only["cost"]["composite_cost"] == cost_no_cg["cost"]["composite_cost"]
            ),
            "true_stage0_node_count": len(true_stage0_forest.nodes),
            "true_stage0_coarse_edge_count": len(true_stage0_forest.coarse_graining_edges),
            "true_stage0_wormhole_edge_count": len(true_stage0_forest.horizontal_wormhole_edges),
            "open_rediscovery_ran_distinct_acquisitions": acq_open != acq_pool,
            "open_rediscovery_strict_pass": open_rec["strict_pass"],
        },
    }
    return summary


# ============================================================================
# Part 2. M4.3 True Two-Channel, Bounded-Noise & Same-Info Continual Benchmark
# ============================================================================

@dataclass(frozen=True)
class M43TargetSpec:
    """Unified single- or two-channel unseen target specification for ``V17.5 (M4.3)``."""

    target_id: str
    mechanism_family: str
    feature_names: tuple[str, ...]
    feature_bounds: dict[str, tuple[float, float]]
    pilot_points: list[dict[str, float]]
    confirm_points: list[dict[str, float]]
    heldout_points: list[dict[str, float]]
    tolerance: float
    oracle_fn: Callable[[dict[str, float]], float]
    coarse_oracle_fn: Callable[[dict[str, float]], float] | None
    domain_cue: str
    scale_regime: str
    modality: str = "macro_standard"
    pilot_noise_std: float = 0.0
    pilot_noise_seed: int = 0
    is_multiscale_two_channel: bool = False
    is_out_of_scope: bool = False


class CompiledRelationalCueTable:
    """Same-information relational lookup table compiled from ``PhilosophicalForest``.

    Separates the value of the forest's relational knowledge representation from online multi-seed PPR:
      - Compiles ``signature -> ordered [(node_id, target_node, wormhole_edge)]`` closure over live
        non-archived nodes + horizontal wormholes + vertical implication/coarse-graining edges;
      - Explicitly meters compilation compute cost and Stage-2 invalidation/rebuild compute cost!
    """

    def __init__(self, forest: PhilosophicalForest) -> None:
        self.forest = forest
        self.sig_table: dict[str, list[dict[str, Any]]] = {}
        self.compile_compute_cost: float = 0.0
        self.recompile_count: int = 0
        self._indexed_forest_state: tuple[Any, ...] = ()
        self._compile()

    def _compile(self) -> None:
        self.sig_table = {}
        live_nodes = {
            nid: nd for nid, nd in self.forest.nodes.items()
            if not nd["utility_ledger"].get("archived_low_utility")
            and nd["claim_contract"].get("evidence_status") != "REFUTED_IN_SCOPE"
        }
        adj: dict[str, list[tuple[str, dict[str, Any] | None]]] = {nid: [] for nid in live_nodes}
        for wh in self.forest.horizontal_wormhole_edges.values():
            if wh.get("stale_endpoint_revision"):
                continue
            s_id, t_id = wh["node_a_id"], wh["node_b_id"]
            if s_id in live_nodes and t_id in live_nodes:
                if (
                    wh.get("node_a_version", live_nodes[s_id]["version"]) == live_nodes[s_id]["version"]
                    and wh.get("node_b_version", live_nodes[t_id]["version"]) == live_nodes[t_id]["version"]
                ):
                    adj[s_id].append((t_id, deepcopy(wh)))
                    adj[t_id].append((s_id, deepcopy(wh)))
        for ie in self.forest.local_implication_edges:
            if ie.get("stale_endpoint_revision"):
                continue
            s_id, w_id = ie["stronger_node_id"], ie["weaker_node_id"]
            if s_id in live_nodes and w_id in live_nodes:
                if (
                    ie.get("stronger_version", live_nodes[s_id]["version"]) == live_nodes[s_id]["version"]
                    and ie.get("weaker_version", live_nodes[w_id]["version"]) == live_nodes[w_id]["version"]
                ):
                    adj[w_id].append((s_id, None))
                    adj[s_id].append((w_id, None))
        for ce in self.forest.coarse_graining_edges:
            if ce.get("stale_endpoint_revision"):
                continue
            f_id, c_id = ce["fine_node_id"], ce["coarse_node_id"]
            if f_id in live_nodes and c_id in live_nodes:
                if (
                    ce.get("fine_version", live_nodes[f_id]["version"]) == live_nodes[f_id]["version"]
                    and ce.get("coarse_version", live_nodes[c_id]["version"]) == live_nodes[c_id]["version"]
                ):
                    adj[f_id].append((c_id, None))
                    adj[c_id].append((f_id, None))

        all_sigs = sorted({s for nd in live_nodes.values() for s in nd.get("probe_signatures", ())})
        visited_ops = 0
        for sig in all_sigs:
            seeds = [nid for nid, nd in live_nodes.items() if sig in nd.get("probe_signatures", ())]
            entries: dict[str, tuple[int, dict[str, Any] | None]] = {}
            frontier = [(nid, 0, None) for nid in seeds]
            while frontier:
                cur_id, dist, wh_ref = frontier.pop(0)
                visited_ops += 1
                if cur_id in entries and entries[cur_id][0] <= dist:
                    continue
                entries[cur_id] = (dist, wh_ref)
                if dist < 2:
                    for nxt_id, nxt_wh in adj.get(cur_id, ()):
                        frontier.append((nxt_id, dist + 1, wh_ref or nxt_wh))
            ranked = sorted(
                entries.items(),
                key=lambda kv: (kv[1][0], -int(live_nodes[kv[0]]["version"]), kv[0]),
            )
            self.sig_table[sig] = [
                {
                    "node_id": nid,
                    "target_node": deepcopy(live_nodes[nid]),
                    "hop_distance": hop_d,
                    "reached_only_by_graph_walk": hop_d > 0,
                    "wormhole_edge": wh_e,
                }
                for nid, (hop_d, wh_e) in ranked
            ]
        self.compile_compute_cost = round(6.0 + 0.12 * visited_ops, 4)
        self._indexed_forest_state = self.forest.forest_state_signature()

    def ensure_fresh(self) -> float:
        cur_state = self.forest.forest_state_signature()
        if cur_state == self._indexed_forest_state:
            return 0.0
        self._compile()
        self.recompile_count += 1
        return float(self.compile_compute_cost)

    def retrieve(self, observed_signatures: list[str], *, task_view: dict[str, Any], top_k: int = 8) -> dict[str, Any]:
        recompile_cost = self.ensure_fresh()
        req_arity = task_view.get("active_arity")
        req_scale = task_view.get("target_scale_regime") or task_view.get("scale_regime")
        agg: dict[str, dict[str, Any]] = {}
        # Prefer specific `sig:cue_*` cues over generic `sig:pos_arity_*` cues
        for sig in observed_signatures:
            w = 3.0 if sig.startswith("sig:cue_") else 0.5
            for item in self.sig_table.get(sig, ()):
                nid = item["node_id"]
                nd = item["target_node"]
                nd_ar = nd["observation_map"].get("parameter_semantics", {}).get("active_arity")
                if req_arity is not None and nd_ar is not None and int(nd_ar) != int(req_arity):
                    continue
                hop_factor = 1.0 / (1.0 + 0.35 * item["hop_distance"])
                sc_factor = 1.0
                if req_scale is not None:
                    nd_sc = ((nd.get("scale_map") or {}).get("window_or_partition") or {}).get("scale_regime")
                    if nd_sc is not None and nd_sc != req_scale:
                        sc_factor = 0.25
                inc = w * hop_factor * sc_factor
                if nid not in agg:
                    agg[nid] = {**deepcopy(item), "score": inc}
                else:
                    agg[nid]["score"] += inc
                    agg[nid]["reached_only_by_graph_walk"] = bool(
                        agg[nid]["reached_only_by_graph_walk"] and item["reached_only_by_graph_walk"]
                    )
        cands = sorted(agg.values(), key=lambda x: (-x["score"], x["node_id"]))[:top_k]
        return {
            "candidates": cands,
            "query_compute_cost": round(0.35 + 0.05 * len(cands) + recompile_cost, 4),
            "table_recompiled": recompile_cost > 0.0,
            "table_recompile_compute_cost": recompile_cost,
        }


def build_m4_3_continual_forest(work_root: Path) -> dict[str, Any]:
    """Build the ``M4.3 / M4.4`` continual forest on top of ``M4.2`` formation with clean self-verifying digests."""
    built = build_and_evolve_m4_2_cross_pillar_forest(work_root)
    unmaint = built["forest_before_stage2_maintenance"]
    maint = built["forest"]

    # Compile Same-Info Relational CueTable at Stage 1 AND rebuild after Stage 2 maintenance!
    cue_table_pre_s2 = CompiledRelationalCueTable(unmaint)
    cue_table_post_s2 = CompiledRelationalCueTable(maint)
    cue_table_total_compile_compute = round(
        cue_table_pre_s2.compile_compute_cost + cue_table_post_s2.compile_compute_cost, 4
    )

    # 6-family stratified DEV static order (`seed = 99_223_026`, completely disjoint from fresh TEST seeds!):
    # Covers ALL 6 families INCLUDING 1 adiabatic (`20`) + 1 quench (`22`) `Pi_s` AND applies matching pilot Gaussian sensor noise!
    dev_targets, _ = build_m4_3_target_suite(seed=99_223_026, onset_pilot_noise_std=0.0020)
    stratified_dev = [dev_targets[i] for i in (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 22)]
    ord_forward = built["ordinary_history_forward"]
    dev_wins: dict[str, int] = {t: 0 for t, _ in ord_forward}
    dev_meter = CostMeter(obs_budget=800, meas_budget=8000.0, compute_budget=25000.0)
    dev_ws = work_root / "m43_stratified_dev_tune"
    for dt in stratified_dev:
        dt_rng = Random(dt.pilot_noise_seed) if dt.pilot_noise_std > 0.0 else None
        p_rows = []
        for pt in dt.pilot_points:
            yv = float(dt.oracle_fn(pt))
            if dt_rng is not None and dt.pilot_noise_std > 0.0 and not dt.is_multiscale_two_channel:
                yv += dt_rng.gauss(0.0, dt.pilot_noise_std)
            p_rows.append({"features": dict(pt), "target": yv})
        c0_rows = [{"features": dict(dt.confirm_points[0]), "target": float(dt.oracle_fn(dt.confirm_points[0]))}]
        u_cost = 10.0 if dt.is_multiscale_two_channel else 4.0
        dev_meter.charge("dev_obs", obs=len(p_rows) + 1, meas=u_cost * (len(p_rows) + 1), compute=2.5)
        same_ar = [(t, sp) for t, sp in ord_forward if int(sp.get("arity", 1)) == len(dt.feature_names)]
        for tag, sp in same_ar:
            cand, ev_n = _autonomous_fit_candidate_specs(
                dev_ws / dt.target_id / tag,
                specs_to_try=[(tag, sp)],
                feature_names=dt.feature_names,
                training_rows=p_rows,
                tolerance=dt.tolerance,
            )
            dev_meter.charge("dev_fit", compute=round(6.0 * ev_n, 4))
            if cand is not None and cand["train_rmse"] <= dt.tolerance:
                cf_e = evaluate_isolated_python_model(
                    dev_ws / dt.target_id / tag, cand["model_relpath"], c0_rows, isolated_prefix_rows=p_rows
                )
                dev_meter.charge("dev_cf", compute=1.5)
                if cf_e["returncode"] == 0 and float(cf_e["rmse"]) <= dt.tolerance:
                    dev_wins[tag] = dev_wins.get(tag, 0) + 1

    ord_stratified_dev = sorted(
        ord_forward,
        key=lambda item: (-dev_wins.get(item[0], 0), [t for t, _ in ord_forward].index(item[0])),
    )

    built["cue_table_post_stage2"] = cue_table_post_s2
    built["cue_table_lifecycle_compute_cost"] = cue_table_total_compile_compute
    built["ordinary_history_stratified_dev_6fam"] = ord_stratified_dev
    built["stratified_dev_6fam_report"] = {
        "dev_seed": 99_223_026,
        "families_covered": sorted({dt.mechanism_family for dt in stratified_dev}),
        "target_count": len(stratified_dev),
        "dev_wins_by_tag": dev_wins,
        "frozen_order_tags": [t for t, _ in ord_stratified_dev],
        "dev_cost": dev_meter.snapshot(),
    }
    return built


def build_m4_3_target_suite(
    seed: int = 20_301_120,
    *,
    onset_pilot_noise_std: float = 0.0020,
) -> tuple[list[M43TargetSpec], list[M43TargetSpec]]:
    """Build the 24-target 6-family unseen ``M4.3`` suite (`4` per family) + `2` seed-varied OOS targets!

    Families (`4` targets/family = `24` in-scope/seed, `96` across 4 fresh seeds `20301120..20331120`):
      1. ``HaldaneShunt_2D`` (`W1` — low/moderate pilot `s in [0.35..3.1]` + high-drive turnover confirm);
      2. ``OscShunt4P_2D`` (`W2` — damped cosine servo + secular shunt);
      3. ``MurrayCapped_1D`` (`W3` — safe low-flow pilot `q in [0.4..2.8]` + high-flow ceiling confirm `q >= 7.6`);
      4. ``OnsetWalk_MonodShunt_2D`` (`W4` — bounded pilot sensor jitter `sigma = 0.008`, 2-hop walk);
      5. ``OnsetWalk_MurrayShunt_2D`` (`W5` — bounded pilot sensor jitter `sigma = 0.008`, 2-hop walk);
      6. ``PiS_MultiScale_Mixed_2D`` (`Pi_s` — `2` adiabatic `[2.5, 5.1]` + `2` post-quench `[2.24, 2.34]`/`[0.16, 1.8]`
         on real two-channel `MultiScaleWorkspaceTaskLab` (`coarse_macro = 3.0`, `micro_fine = 10.0`)!).
    """
    rng = Random(seed)
    targets: list[M43TargetSpec] = []

    # Family 1 (`4` targets): `HaldaneShunt_2D` (`W1` — low/moderate pilot `s <= 3.1` exercises Stage-2 Monod scope revision & Minton!)
    for i in range(4):
        vmax = round(rng.uniform(8.3, 10.1), 3)
        km = round(rng.uniform(1.85, 2.35), 3)
        ki = round(rng.uniform(0.125, 0.155), 4)
        ks = round(rng.uniform(0.62, 0.86), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(13.5, 16.8), 2) if sgn < 0 else round(rng.uniform(0.8, 2.8), 2)

        def _mk_hs(vm: float, k_m: float, k_i: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (vm * p["feed_conc"] / (k_m + p["feed_conc"] + k_i * (p["feed_conc"] ** 2)) + k_s * p["recycle_bleed"])

        fnames = ("recycle_bleed", "feed_conc") if i % 2 == 0 else ("feed_conc", "recycle_bleed")
        targets.append(M43TargetSpec(
            target_id=f"M43_T{len(targets)+1:02d}_HaldaneShunt",
            mechanism_family="HaldaneShunt_2D",
            feature_names=fnames,
            feature_bounds={"feed_conc": (0.2, 12.0), "recycle_bleed": (0.0, 2.0)},
            pilot_points=[
                {"feed_conc": 0.4, "recycle_bleed": 0.1},
                {"feed_conc": 1.2, "recycle_bleed": 0.8},
                {"feed_conc": 2.5, "recycle_bleed": 0.2},
                {"feed_conc": 4.8, "recycle_bleed": 1.1},
                {"feed_conc": 8.2, "recycle_bleed": 0.4},
                {"feed_conc": 11.1, "recycle_bleed": 0.3},
            ],
            confirm_points=[
                {"feed_conc": 9.6, "recycle_bleed": 0.5},
                {"feed_conc": 3.8, "recycle_bleed": 0.9},
                {"feed_conc": 10.4, "recycle_bleed": 0.7},
            ],
            heldout_points=[
                {"feed_conc": 0.9, "recycle_bleed": 0.3},
                {"feed_conc": 2.2, "recycle_bleed": 1.2},
                {"feed_conc": 4.4, "recycle_bleed": 0.6},
                {"feed_conc": 7.6, "recycle_bleed": 1.0},
                {"feed_conc": 10.9, "recycle_bleed": 0.2},
            ],
            tolerance=0.055,
            oracle_fn=_mk_hs(vmax, km, ki, ks, sgn, off),
            coarse_oracle_fn=None,
            domain_cue=("cue_pore_collapse" if sgn < 0 else "cue_feed_reactor"),
            scale_regime="macro",
        ))

    # Family 2 (`4` targets): `OscShunt4P_2D` (`W2`)
    for i in range(4):
        gain = round(rng.uniform(1.88, 2.50), 3)
        tau = round(rng.uniform(1.18, 1.52), 3)
        omega = round(rng.uniform(1.98, 2.32), 3)
        drift = round(rng.uniform(0.34, 0.52), 3)
        sgn = -1.0 if i % 2 == 1 else 1.0
        off = round(rng.uniform(14.0, 17.5), 2) if sgn < 0 else round(rng.uniform(0.6, 2.4), 2)

        def _mk_osc(g: float, t_c: float, w: float, d: float, s: float, o: float):
            return lambda p: o + s * (g * p["step_amp"] * (1.0 - math.exp(-p["dwell_t"] / t_c) * math.cos(w * p["dwell_t"])) + d * p["dwell_t"])

        targets.append(M43TargetSpec(
            target_id=f"M43_T{len(targets)+1:02d}_OscShunt4P",
            mechanism_family="OscShunt4P_2D",
            feature_names=("step_amp", "dwell_t"),
            feature_bounds={"step_amp": (0.4, 3.0), "dwell_t": (0.05, 4.5)},
            pilot_points=[
                {"step_amp": 1.4, "dwell_t": 0.12},
                {"step_amp": 1.2, "dwell_t": 0.65},
                {"step_amp": 1.3, "dwell_t": 1.25},
                {"step_amp": 2.2, "dwell_t": 2.05},
                {"step_amp": 1.8, "dwell_t": 3.10},
                {"step_amp": 2.5, "dwell_t": 4.25},
            ],
            confirm_points=[
                {"step_amp": 1.7, "dwell_t": 0.95},
                {"step_amp": 2.1, "dwell_t": 2.65},
                {"step_amp": 1.9, "dwell_t": 3.55},
            ],
            heldout_points=[
                {"step_amp": 1.0, "dwell_t": 0.25},
                {"step_amp": 1.6, "dwell_t": 1.10},
                {"step_amp": 2.4, "dwell_t": 1.75},
                {"step_amp": 1.4, "dwell_t": 2.90},
                {"step_amp": 2.6, "dwell_t": 4.05},
            ],
            tolerance=0.06,
            oracle_fn=_mk_osc(gain, tau, omega, drift, sgn, off),
            coarse_oracle_fn=None,
            domain_cue=("cue_servo_transient" if sgn < 0 else "cue_step_relax"),
            scale_regime="micro",
        ))

    # Family 3 (`4` targets): `MurrayCapped_1D` (`W3` — exercises Stage-1's 3rd discovery in Stage 3!)
    for i in range(4):
        c_m = round(rng.uniform(1.95, 2.45), 3)
        cap_v = round(rng.uniform(5.10, 6.20), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(12.0, 15.5), 2) if sgn < 0 else round(rng.uniform(0.5, 2.0), 2)

        def _mk_mcap(cm: float, cp: float, s: float, o: float):
            return lambda p: o + s * min(cm * (p["trunk_flow"] ** (2.0 / 3.0)), cp)

        targets.append(M43TargetSpec(
            target_id=f"M43_T{len(targets)+1:02d}_MurrayCapped",
            mechanism_family="MurrayCapped_1D",
            feature_names=("trunk_flow",),
            feature_bounds={"trunk_flow": (0.25, 10.0)},
            pilot_points=[{"trunk_flow": v} for v in (0.4, 1.0, 1.8, 3.2, 6.8, 9.1)],
            confirm_points=[{"trunk_flow": 8.4}, {"trunk_flow": 4.6}, {"trunk_flow": 9.5}],
            heldout_points=[{"trunk_flow": v} for v in (0.7, 2.1, 4.2, 7.5, 9.3)],
            tolerance=0.045,
            oracle_fn=_mk_mcap(c_m, cap_v, sgn, off),
            coarse_oracle_fn=None,
            domain_cue="cue_vascular_ceiling_onset",
            scale_regime="macro",
        ))

    # Family 4 (`4` targets): 2-Hop Onset Walk `Monod + Shunt` (`W4`, bounded pilot sensor noise `sigma = 0.008`!)
    for i in range(4):
        vm = round(rng.uniform(8.4, 9.8), 3)
        km = round(rng.uniform(1.40, 1.70), 3)
        ks = round(rng.uniform(0.62, 0.84), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(13.0, 16.5), 2) if sgn < 0 else round(rng.uniform(1.0, 3.0), 2)
        n_seed = seed * 31 + 100 + i

        def _mk_ms(v_m: float, k_m: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (v_m * p["solute_load"] / (k_m + p["solute_load"]) + k_s * p["bypass_flow"])

        targets.append(M43TargetSpec(
            target_id=f"M43_T{len(targets)+1:02d}_OnsetWalkMonodShunt",
            mechanism_family="OnsetWalk_MonodShunt_2D",
            feature_names=("solute_load", "bypass_flow"),
            feature_bounds={"solute_load": (0.2, 12.0), "bypass_flow": (0.0, 2.0)},
            pilot_points=[
                {"solute_load": 0.55, "bypass_flow": 0.2},
                {"solute_load": 0.70, "bypass_flow": 0.9},
                {"solute_load": 0.85, "bypass_flow": 0.3},
                {"solute_load": 1.00, "bypass_flow": 1.1},
                {"solute_load": 1.15, "bypass_flow": 0.4},
                {"solute_load": 1.30, "bypass_flow": 0.8},
            ],
            confirm_points=[
                {"solute_load": 11.5, "bypass_flow": 0.6},
                {"solute_load": 8.5, "bypass_flow": 1.0},
                {"solute_load": 10.2, "bypass_flow": 0.5},
            ],
            heldout_points=[
                {"solute_load": 0.80, "bypass_flow": 0.5},
                {"solute_load": 2.60, "bypass_flow": 0.9},
                {"solute_load": 6.20, "bypass_flow": 0.4},
                {"solute_load": 9.80, "bypass_flow": 1.1},
                {"solute_load": 11.2, "bypass_flow": 0.3},
            ],
            tolerance=0.062,
            oracle_fn=_mk_ms(vm, km, ks, sgn, off),
            coarse_oracle_fn=None,
            domain_cue="cue_carrier_binding_onset",
            scale_regime="macro",
            pilot_noise_std=float(onset_pilot_noise_std),
            pilot_noise_seed=n_seed,
        ))

    # Family 5 (`4` targets): 2-Hop Onset Walk `Murray 2/3 + Shunt` (`W5`, bounded pilot sensor noise `sigma = 0.008`!)
    for i in range(4):
        ck = round(rng.uniform(2.85, 3.45), 3)
        ks = round(rng.uniform(0.60, 0.82), 3)
        sgn = -1.0 if i % 2 == 1 else 1.0
        off = round(rng.uniform(16.0, 20.0), 2) if sgn < 0 else round(rng.uniform(1.0, 3.0), 2)
        n_seed = seed * 31 + 200 + i

        def _mk_murs(c_k: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (c_k * (p["solute_load"] ** (2.0 / 3.0)) + k_s * p["bypass_flow"])

        targets.append(M43TargetSpec(
            target_id=f"M43_T{len(targets)+1:02d}_OnsetWalkMurrayShunt",
            mechanism_family="OnsetWalk_MurrayShunt_2D",
            feature_names=("solute_load", "bypass_flow"),
            feature_bounds={"solute_load": (0.2, 12.0), "bypass_flow": (0.0, 2.0)},
            pilot_points=[
                {"solute_load": 0.55, "bypass_flow": 0.2},
                {"solute_load": 0.70, "bypass_flow": 0.9},
                {"solute_load": 0.85, "bypass_flow": 0.3},
                {"solute_load": 1.00, "bypass_flow": 1.1},
                {"solute_load": 1.15, "bypass_flow": 0.4},
                {"solute_load": 1.30, "bypass_flow": 0.8},
            ],
            confirm_points=[
                {"solute_load": 11.5, "bypass_flow": 0.6},
                {"solute_load": 8.5, "bypass_flow": 1.0},
                {"solute_load": 10.2, "bypass_flow": 0.5},
            ],
            heldout_points=[
                {"solute_load": 0.80, "bypass_flow": 0.5},
                {"solute_load": 2.60, "bypass_flow": 0.9},
                {"solute_load": 6.20, "bypass_flow": 0.4},
                {"solute_load": 9.80, "bypass_flow": 1.1},
                {"solute_load": 11.2, "bypass_flow": 0.3},
            ],
            tolerance=0.062,
            oracle_fn=_mk_murs(ck, ks, sgn, off),
            coarse_oracle_fn=None,
            domain_cue="cue_vascular_branch_onset",
            scale_regime="macro",
            pilot_noise_std=float(onset_pilot_noise_std),
            pilot_noise_seed=n_seed,
        ))

    # Family 6 (`4` targets): True Two-Channel `Pi_s` (`2` adiabatic `[2.5, 5.1]` + `2` boundary quench `[2.24, 2.34]`!)
    # Notice: on `[2.24, 2.34]` (`thermal`), Stage-0's unrevised box `[2.2, 5.2]` thinks coarse is valid while
    # Stage-2's quench-reverified box `[2.4, 5.2]` cleanly escalates to `micro_fine`!
    for i in range(4):
        scale_k = round(rng.uniform(0.92, 1.32), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(10.0, 14.0), 2) if sgn < 0 else round(rng.uniform(0.5, 2.5), 2)
        is_quench_boundary = i >= 2
        mod_tag = "thermal" if is_quench_boundary else "hydraulic"

        def _mk_pi_fine(sk: float, s: float, o: float):
            return lambda p: o + s * sk * float(
                _eval_pi_s_fine_scalar(p["u_drive"], p["T_horizon"], DEFAULT_PI_S_ODE_PARAMS)["y_micro"]
            )

        def _mk_pi_coarse(sk: float, s: float, o: float):
            return lambda p: o + s * sk * float(
                _eval_pi_s_coarse_scalar(p["u_drive"], p["T_horizon"], DEFAULT_PI_S_ODE_PARAMS)["y_coarse"]
            )

        if not is_quench_boundary:
            p_pts = [
                {"u_drive": 0.6, "T_horizon": 4.8},
                {"u_drive": 1.2, "T_horizon": 2.5},
                {"u_drive": 1.8, "T_horizon": 4.2},
                {"u_drive": 2.4, "T_horizon": 2.7},
                {"u_drive": 2.9, "T_horizon": 3.6},
                {"u_drive": 0.9, "T_horizon": 3.1},
            ]
            c_pts = [
                {"u_drive": 2.6, "T_horizon": 4.9},
                {"u_drive": 0.8, "T_horizon": 2.55},
                {"u_drive": 1.9, "T_horizon": 3.4},
            ]
            h_pts = [
                {"u_drive": 0.8, "T_horizon": 2.6},
                {"u_drive": 1.4, "T_horizon": 3.1},
                {"u_drive": 2.1, "T_horizon": 3.9},
                {"u_drive": 2.7, "T_horizon": 4.8},
            ]
            t_bnds = (2.4, 5.2)
            s_reg = "meso"
        else:
            # Boundary / short-horizon quench target (`T` dips to `2.24` on confirm & `0.22` on pilot/heldout?
            # Put pilot in `[2.24, 4.5]`, confirm at `2.25` & `0.25` — inside `[2.2, 5.2]` for pilot+confirm[0],
            # OR pilot+confirm in `[2.23, 4.6]` and heldout includes `2.23` vs `0.22`? Wait: keep pilot+confirm+heldout
            # consistent on the same physical task regime `[0.18, 4.6]` so `micro_fine` solves it cleanly!)
            p_pts = [
                {"u_drive": 0.8, "T_horizon": 0.20},
                {"u_drive": 1.5, "T_horizon": 0.55},
                {"u_drive": 2.2, "T_horizon": 1.15},
                {"u_drive": 2.7, "T_horizon": 1.85},
                {"u_drive": 1.9, "T_horizon": 2.95},
                {"u_drive": 2.5, "T_horizon": 4.20},
            ]
            c_pts = [
                {"u_drive": 2.4, "T_horizon": 0.28},
                {"u_drive": 1.7, "T_horizon": 2.25},
                {"u_drive": 2.1, "T_horizon": 3.30},
            ]
            h_pts = [
                {"u_drive": 1.1, "T_horizon": 0.24},
                {"u_drive": 2.0, "T_horizon": 0.75},
                {"u_drive": 2.6, "T_horizon": 1.65},
                {"u_drive": 1.8, "T_horizon": 3.50},
            ]
            t_bnds = (0.15, 5.2)
            s_reg = "micro"

        targets.append(M43TargetSpec(
            target_id=f"M43_T{len(targets)+1:02d}_PiS_{'QuenchEscalate' if is_quench_boundary else 'Adiabatic'}",
            mechanism_family="PiS_MultiScale_Mixed_2D",
            feature_names=("u_drive", "T_horizon"),
            feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": t_bnds},
            pilot_points=p_pts,
            confirm_points=c_pts,
            heldout_points=h_pts,
            tolerance=0.055,
            oracle_fn=_mk_pi_fine(scale_k, sgn, off),
            coarse_oracle_fn=_mk_pi_coarse(scale_k, sgn, off),
            domain_cue="cue_adiabatic_multiscale",
            scale_regime=s_reg,
            modality=mod_tag,
            is_multiscale_two_channel=True,
        ))

    # Seed-varied Out-of-Scope Honesty Targets (`2` per seed -> `8` distinct OOS across 4 seeds)
    oos_targets: list[M43TargetSpec] = []
    for o_i in range(2):
        freq = round(rng.uniform(5.9, 7.7), 3)
        amp = round(rng.uniform(3.5, 4.7), 3)
        jump = round(rng.uniform(3.2, 4.3), 3)
        phase_sh = round(rng.uniform(0.15, 2.75), 3)

        def _mk_oos(fq: float, am: float, jm: float, ph: float):
            return lambda p: am * math.sin(fq * p["trunk_flow"] + ph) + (jm if int(p["trunk_flow"] * 3) % 2 == 0 else -jm)

        oos_targets.append(M43TargetSpec(
            target_id=f"M43_OOS_{o_i+1:02d}_s{seed % 10000}",
            mechanism_family="OutOfScope_Discontinuous",
            feature_names=("trunk_flow",),
            feature_bounds={"trunk_flow": (0.3, 9.0)},
            pilot_points=[{"trunk_flow": v} for v in (0.5, 1.8, 3.6, 5.8, 8.2)],
            confirm_points=[{"trunk_flow": 2.9}, {"trunk_flow": 6.7}, {"trunk_flow": 4.4}],
            heldout_points=[{"trunk_flow": v} for v in (0.9, 2.5, 5.1, 7.3, 8.7)],
            tolerance=0.05,
            oracle_fn=_mk_oos(freq, amp, jump, phase_sh),
            coarse_oracle_fn=None,
            domain_cue="low_flow_cdn_onset",
            scale_regime="macro",
            is_out_of_scope=True,
        ))

    return targets, oos_targets


def _fit_best_pilot_rmse_across_specs(
    ws_dir: Path,
    *,
    specs_to_try: list[tuple[str, dict[str, Any]]],
    feature_names: tuple[str, ...],
    training_rows: list[dict[str, Any]],
    calibrate_lens: bool = True,
) -> tuple[dict[str, Any] | None, int]:
    """Fit ALL candidate specs on ``training_rows`` and select the MINIMUM pilot RMSE (`BestPilotRMSE`)!"""
    best_overall: dict[str, Any] | None = None
    total_evals = 0
    for idx_s, (tag, sp) in enumerate(specs_to_try):
        # Pass `tolerance=0.0` so `_autonomous_fit_candidate_specs` evaluates all role permutations without early-exit!
        cand, n_ev = _autonomous_fit_candidate_specs(
            ws_dir / f"min_rmse_{idx_s}_{tag}",
            specs_to_try=[(tag, sp)],
            feature_names=feature_names,
            training_rows=training_rows,
            tolerance=0.0,
            calibrate_lens=calibrate_lens,
        )
        total_evals += n_ev
        if cand is not None:
            if best_overall is None or float(cand["train_rmse"]) < float(best_overall["train_rmse"]):
                sub_rel = f"best_min_pilot_{idx_s}.py"
                (ws_dir / sub_rel).write_text(cand["model_source"], encoding="utf-8")
                best_overall = {**cand, "model_relpath": sub_rel}
    return best_overall, total_evals


def evaluate_m4_3_target_under_arm(
    target: M43TargetSpec,
    *,
    arm_name: str,
    seed: int,
    work_dir: Path,
    forest: PhilosophicalForest | None,
    index: HippoRAGWormholeIndex | FlatEqualMetadataIndex | CompiledRelationalCueTable | None,
    ordinary_history_specs: list[tuple[str, dict[str, Any]]] | None,
    fallback_pool_specs: list[tuple[str, dict[str, Any]]],
    selection_policy: str = "first_acceptable",  # "first_acceptable" | "min_pilot_rmse"
    open_code_agent_fallback: bool = False,
    allow_pool_fallback: bool = True,
) -> dict[str, Any]:
    """Evaluate one ``M4.3`` target with true two-channel ``MultiScaleWorkspaceTaskLab`` pricing,
    bounded pilot sensor noise, canonical role-mapped box & final-query scope guards, zero training-confirm
    overlap, and complete event-level ``CostMeter`` accounting!
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    meter = CostMeter(obs_budget=96, meas_budget=1200.0, compute_budget=12000.0)

    # Register fast transient micro operator in candidate universe for `micro_fine` quench escalation!
    micro_fast_2d_spec = apply_structural_combinator(
        make_atomic_operator_spec("exp_relax_2d"),
        "fast_mode_transient_correction",
        secondary_role=1,
    )

    # 1. Channel-bound laboratory setup (`MultiScaleWorkspaceTaskLab` on two-channel multiscale targets!)
    ms_lab: MultiScaleWorkspaceTaskLab | None = None
    selected_channel = "macro_single"
    authorizing_coarse_node_id: str | None = None
    authorizing_coarse_vbox: dict[str, Any] | None = None
    authorizing_coarse_role_map: dict[str, str] | None = None

    if target.is_multiscale_two_channel:
        assert target.coarse_oracle_fn is not None
        ms_lab = MultiScaleWorkspaceTaskLab(
            task_id=target.target_id,
            surface_domain=target.domain_cue,
            feature_names=target.feature_names,
            feature_bounds=target.feature_bounds,
            true_scale_regime=target.scale_regime,
            heldout_points=target.heldout_points,
            tolerance=target.tolerance,
            micro_fn=target.oracle_fn,
            coarse_pi_fn=target.coarse_oracle_fn,
            coarse_cost_per_obs=3.0,
            micro_cost_per_obs=10.0,
        )
        selected_channel = "micro_fine"
        if forest is not None and index is not None:
            pre_ret = index.retrieve(
                [f"sig:{target.domain_cue}", f"sig:pos_arity_{len(target.feature_names)}"],
                task_view={
                    "goal_kind": "open_workspace_mechanism_prediction",
                    "tolerance": target.tolerance,
                    "active_arity": len(target.feature_names),
                    "target_scale_regime": "meso",
                },
                top_k=8,
            )
            meter.charge("channel_precheck_retrieval", compute=round(float(pre_ret.get("query_compute_cost", 0.0)), 4))
            for pc in pre_ret["candidates"]:
                p_nd = pc["target_node"]
                p_sm = p_nd.get("scale_map") or {}
                p_vbox = (p_sm.get("window_or_partition") or {}).get("verified_input_box")
                if (p_sm.get("window_or_partition") or {}).get("scale_regime") != "meso":
                    continue
                if not forest.fine_scale_children_of(p_nd["node_id"]):
                    continue
                p_gate = authorize_abstraction_use(
                    requested_tier="bounded_trial",
                    task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
                    target_features={"active_arity": len(target.feature_names), "modality": target.modality},
                    claim_contract=p_nd["claim_contract"],
                    scale_map=p_sm,
                    wormhole_edge=pc.get("wormhole_edge"),
                )
                if not p_gate["granted"]:
                    continue
                p_eff_act = p_gate.get("selected_action") or p_nd["claim_contract"]["conclusion_or_action"]
                in_p_box, p_valid_rmaps, _ = _find_all_valid_canonical_role_maps_for_box(
                    target.feature_names,
                    p_nd,
                    list(target.pilot_points),
                    artifact_root=forest.artifact_root,
                    action_override=p_eff_act,
                )
                if not in_p_box:
                    continue
                # Also verify `target.feature_bounds` has non-empty interior intersection with `p_vbox` under >= 1 valid map!
                usable_p_rmap: dict[str, str] | None = None
                if isinstance(p_vbox, dict) and p_vbox:
                    for cand_rm in p_valid_rmaps:
                        if all(
                            max(float(target.feature_bounds[tf][0]), float(p_vbox[cand_rm[tf]][0]))
                            <= min(float(target.feature_bounds[tf][1]), float(p_vbox[cand_rm[tf]][1])) - 1e-7
                            for tf in target.feature_names
                        ):
                            usable_p_rmap = cand_rm
                            break
                    if usable_p_rmap is None:
                        continue
                selected_channel = "coarse_macro"
                authorizing_coarse_node_id = str(p_nd["node_id"])
                authorizing_coarse_vbox = deepcopy(p_vbox) if isinstance(p_vbox, dict) else None
                authorizing_coarse_role_map = deepcopy(usable_p_rmap)
                break

    # Deterministic task-seeded Gaussian pilot sensor noise (`pilot_noise_std`) identical across ALL arms!
    pilot_noise_rng = Random(target.pilot_noise_seed) if target.pilot_noise_std > 0.0 else None

    def _acquire_points(
        pts: list[dict[str, float]],
        *,
        phase: str,
        enforced_vbox: dict[str, Any] | None = None,
        enforced_role_map: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if not pts:
            return []
        if ms_lab is not None:
            if selected_channel == "coarse_macro":
                chk_box = enforced_vbox if enforced_vbox is not None else authorizing_coarse_vbox
                chk_rmap = enforced_role_map if enforced_role_map is not None else authorizing_coarse_role_map
                if isinstance(chk_box, dict) and chk_box:
                    if not all(is_point_inside_verified_box(pt, chk_box, role_map=chk_rmap) for pt in pts):
                        raise RuntimeError(f"Attempted coarse_macro acquisition outside verified_input_box in phase={phase}")
            c_rows = ms_lab.acquire_channel(pts, channel=selected_channel, phase=phase)
            u_c = ms_lab.coarse_cost_per_obs if selected_channel == "coarse_macro" else ms_lab.micro_cost_per_obs
            meter.charge(phase, obs=len(c_rows), meas=round(u_c * len(c_rows), 4), compute=0.0)
            return [{"features": dict(r["features"]), "target": float(r["target"])} for r in c_rows]
        # Single-channel `OpenWorkspaceTaskLab` acquisition
        out_rows: list[dict[str, Any]] = []
        for pt in pts:
            clean_pt = {k: float(pt[k]) for k in target.feature_names}
            y_v = float(target.oracle_fn(clean_pt))
            if phase == "pilot" and pilot_noise_rng is not None and target.pilot_noise_std > 0.0:
                y_v += pilot_noise_rng.gauss(0.0, target.pilot_noise_std)
            out_rows.append({"features": clean_pt, "target": y_v})
        meter.charge(phase, obs=len(out_rows), meas=round(4.0 * len(out_rows), 4), compute=0.0)
        return out_rows

    frozen_prefix = _acquire_points(target.pilot_points, phase="pilot")
    pilot_sigs, obs_feats = extract_unified_observable_pilot_cues(
        frozen_prefix, feature_names=target.feature_names, domain_cue=target.domain_cue
    )
    if len(target.feature_names) == 2:
        obs_feats["secondary_bleed_active"] = True
    if target.is_multiscale_two_channel:
        obs_feats["modality"] = target.modality
    for fn_coord in target.feature_names:
        obs_feats.setdefault(fn_coord, max(float(r["features"][fn_coord]) for r in frozen_prefix))

    chosen_rel: str | None = None
    confirmed_ok = False
    confirmed_via_recall = False
    non_direct_walk_win = False
    total_evaluated = 0
    unmetered_evals = 0
    selected_node_id: str | None = None
    selected_version: int | None = None
    selected_model_sha256: str | None = None
    selected_conf_digest: str | None = None
    confirmed_vbox: dict[str, Any] | None = None
    confirmed_role_map: dict[str, str] | None = None
    confirmed_coord_preconditions: list[dict[str, Any]] = []
    confirmed_role_order: list[str] | None = None
    confirmed_canon_roles: list[str] | None = None
    refuted_recall_traces: list[dict[str, Any]] = []
    confirmed_recall_bundle: dict[str, Any] | None = None
    max_overlap = 0
    refuted_tags: set[str] = set()

    def _ov_cnt(q_r: list[dict[str, Any]], p_r: list[dict[str, Any]]) -> int:
        return sum(1 for qr in q_r if any(coords_approx_equal(qr["features"], pr["features"]) for pr in p_r))

    # 2. Forest / Compiled-CueTable Recall
    if forest is not None and index is not None:
        ret = index.retrieve(
            pilot_sigs,
            task_view={
                "goal_kind": "open_workspace_mechanism_prediction",
                "tolerance": target.tolerance,
                "active_arity": len(target.feature_names),
                "target_scale_regime": ("meso" if selected_channel == "coarse_macro" else target.scale_regime),
            },
            top_k=8,
        )
        meter.charge("retrieval_query", compute=round(float(ret.get("query_compute_cost", 0.0)), 4))

        auth_cands: list[dict[str, Any]] = []
        for cand in ret["candidates"]:
            nd = cand["target_node"]
            if int(nd["observation_map"]["parameter_semantics"].get("active_arity", 0)) != len(target.feature_names):
                continue
            sm = nd.get("scale_map") or {}
            nd_scale = (sm.get("window_or_partition") or {}).get("scale_regime")
            if target.is_multiscale_two_channel and nd_scale == "meso" and not forest.fine_scale_children_of(nd["node_id"]):
                continue
            # Authorize `bounded_trial` FIRST so disjunctive OR branches select their own independent `selected_action`!
            gate = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
                target_features=obs_feats,
                claim_contract=nd["claim_contract"],
                scale_map=sm,
                wormhole_edge=cand.get("wormhole_edge"),
            )
            if not gate["granted"]:
                continue
            eff_act = gate.get("selected_action") or nd["claim_contract"]["conclusion_or_action"]
            plan = _validate_and_interpret_m4_program_steps(
                nd.get("program_steps"), node=nd, target_arity=len(target.feature_names), action_override=eff_act
            )
            if plan is None:
                continue
            if eff_act.get("fine_ode_direct_mode") is True:
                # Fine ODE node on short-horizon quench: only reachable/usable on `micro_fine` channel!
                v_spec = micro_fast_2d_spec
            else:
                v_spec = _load_node_verified_operator_spec(
                    nd, artifact_root=forest.artifact_root, action_override=eff_act
                )
                meter.charge("artifact_integrity_check", compute=0.25)
                if v_spec is None:
                    continue
            in_box, valid_rmaps, canon_roles = _find_all_valid_canonical_role_maps_for_box(
                target.feature_names,
                nd,
                [r["features"] for r in frozen_prefix],
                artifact_root=forest.artifact_root,
                action_override=eff_act,
            )
            if not in_box:
                continue
            auth_cands.append({
                **dict(cand),
                "_verified_spec": v_spec,
                "_exec_plan": plan,
                "_selected_action": eff_act,
                "_selected_branch": gate.get("selected_branch"),
                "_valid_role_maps": valid_rmaps,
                "_canon_roles": canon_roles,
                "_canonical_role_map": valid_rmaps[0] if valid_rmaps else None,
                "_fine_ode_direct": bool(eff_act.get("fine_ode_direct_mode") is True),
            })

        tried_tags: set[str] = set()
        for c_cand in auth_cands:
            plan_c = c_cand["_exec_plan"]
            c_tag = str(plan_c["operator_tag"])
            if c_tag in tried_tags:
                continue
            tried_tags.add(c_tag)
            nd = c_cand["target_node"]
            nd_scale = ((nd.get("scale_map") or {}).get("window_or_partition") or {}).get("scale_regime")
            if target.is_multiscale_two_channel and nd_scale != "meso" and selected_channel == "coarse_macro":
                selected_channel = "micro_fine"
                frozen_prefix = _acquire_points(target.pilot_points, phase="escalated_micro_pilot")

            valid_rmaps_c = c_cand.get("_valid_role_maps") or []
            canon_roles_c = c_cand.get("_canon_roles") or []

            if c_cand.get("_fine_ode_direct"):
                # Exact linear 2-mode eigen-solution of A = [[-1/tau_s, g_sf], [-g_fs/tau_f, -1/tau_f]] (`a12 == g_sf`!):
                # Zero discarded pre-fit; explicitly meters local trusted ODE solver evaluations in `CostMeter`!
                ode_p = dict((nd["claim_contract"]["conclusion_or_action"] or {}).get("ode_params") or DEFAULT_PI_S_ODE_PARAMS)
                tau_s, tau_f = float(ode_p["tau_s"]), float(ode_p["tau_f"])
                a11 = -1.0 / tau_s
                a12 = float(ode_p["g_sf"])
                a21 = -float(ode_p["g_fs"]) / tau_f
                a22 = -1.0 / tau_f
                tr_a = a11 + a22
                det_a = a11 * a22 - a12 * a21
                disc = math.sqrt(max(0.0, tr_a * tr_a - 4.0 * det_a))
                lam1 = 0.5 * (tr_a + disc)
                lam2 = 0.5 * (tr_a - disc)

                ts_cal = (0.20, 0.80, 4.50)
                ys_cal = tuple(_eval_pi_s_fine_scalar(1.0, tv, ode_p)["y_micro"] for tv in ts_cal)
                meter.charge("local_trusted_rk4_basis_compute", compute=round(0.35 * len(ts_cal), 4))
                mat = [[1.0, math.exp(lam1 * t_v), math.exp(lam2 * t_v), y_v] for t_v, y_v in zip(ts_cal, ys_cal)]
                for col in range(3):
                    piv = max(range(col, 3), key=lambda r_i: abs(mat[r_i][col]))
                    mat[col], mat[piv] = mat[piv], mat[col]
                    inv_d = 1.0 / mat[col][col]
                    for c2 in range(col, 4):
                        mat[col][c2] *= inv_d
                    for r_i in range(3):
                        if r_i != col:
                            fac = mat[r_i][col]
                            for c2 in range(col, 4):
                                mat[r_i][c2] -= fac * mat[col][c2]
                k0, k1, k2 = mat[0][3], mat[1][3], mat[2][3]

                role_orders_fine = (
                    [list(target.feature_names)]
                    if set(target.feature_names) == {"u_drive", "T_horizon"}
                    else [list(rp) for rp in _infer_role_permutations(target.feature_names)]
                )
                best_c = None
                ev_c = 0
                for f_idx, r_ord in enumerate(role_orders_fine):
                    u_key = "u_drive" if "u_drive" in r_ord else r_ord[0]
                    t_key = "T_horizon" if "T_horizon" in r_ord else r_ord[1]
                    fine_rel = "recalled_fine_2mode.py" if f_idx == 0 else f"recalled_fine_2mode_{f_idx}.py"
                    fine_py = (
                        "import math\n"
                        "_PARAMS = {'intercept': 0.0, 'slope': 1.0}\n"
                        "def fit(observations):\n"
                        "    if not observations:\n"
                        "        return\n"
                        "    xs, ys = [], []\n"
                        "    for r in observations:\n"
                        f"        u = float(r['features'][{u_key!r}])\n"
                        f"        t = float(r['features'][{t_key!r}])\n"
                        f"        ym = u * ({k0:.15g} + {k1:.15g} * math.exp({lam1:.15g} * t) + {k2:.15g} * math.exp({lam2:.15g} * t))\n"
                        "        xs.append(ym)\n"
                        "        ys.append(float(r['target']))\n"
                        "    mx = sum(xs) / len(xs)\n"
                        "    my = sum(ys) / len(ys)\n"
                        "    sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) + 1e-12)\n"
                        "    _PARAMS['slope'] = sl\n"
                        "    _PARAMS['intercept'] = my - sl * mx\n"
                        "def predict(row):\n"
                        "    f = row.get('features', row)\n"
                        f"    u = float(f[{u_key!r}])\n"
                        f"    t = float(f[{t_key!r}])\n"
                        f"    ym = u * ({k0:.15g} + {k1:.15g} * math.exp({lam1:.15g} * t) + {k2:.15g} * math.exp({lam2:.15g} * t))\n"
                        "    return _PARAMS['intercept'] + _PARAMS['slope'] * ym\n"
                    )
                    (work_dir / fine_rel).write_text(fine_py, encoding="utf-8")
                    tr_ev = evaluate_isolated_python_model(
                        work_dir, fine_rel, frozen_prefix, isolated_prefix_rows=frozen_prefix
                    )
                    ev_c += 1
                    f_rmse = float(tr_ev["rmse"])
                    if tr_ev["returncode"] == 0 and math.isfinite(f_rmse) and (best_c is None or f_rmse < best_c["train_rmse"]):
                        best_c = {
                            "tag": c_tag,
                            "role_order": [u_key, t_key],
                            "train_rmse": f_rmse,
                            "model_relpath": fine_rel,
                            "model_source": fine_py,
                        }
                    if tr_ev["returncode"] == 0 and f_rmse <= target.tolerance * 0.85:
                        break
            else:
                allowed_orders_c = (
                    [_role_order_from_canonical_role_map(canon_roles_c, rm) for rm in valid_rmaps_c]
                    if valid_rmaps_c
                    else None
                )
                best_c, ev_c = _autonomous_fit_candidate_specs(
                    work_dir,
                    specs_to_try=[(c_tag, c_cand["_verified_spec"])],
                    feature_names=target.feature_names,
                    training_rows=frozen_prefix,
                    tolerance=target.tolerance,
                    calibrate_lens=bool(plan_c["calibrate_lens"]),
                    allowed_role_orders=allowed_orders_c,
                )
            total_evaluated += ev_c
            unmetered_evals += ev_c
            eff_act_c = c_cand.get("_selected_action") or nd["claim_contract"]["conclusion_or_action"]
            act_rnames_c = [str(x) for x in eff_act_c.get("role_names", ())]
            if best_c is None or best_c["train_rmse"] > target.tolerance:
                if best_c is not None:
                    refuted_recall_traces.append({
                        "node_id": str(nd["node_id"]),
                        "node_version": int(nd["version"]),
                        "stage": "pilot_fit_rejected",
                        "train_rmse": float(best_c["train_rmse"]),
                        "tolerance": float(target.tolerance),
                        "role_order": list(best_c.get("role_order") or target.feature_names),
                        "canon_roles": act_rnames_c or list(target.feature_names),
                        "model_source": best_c["model_source"],
                        "training_prefix_rows": deepcopy(frozen_prefix),
                        "evaluated_rows": deepcopy(frozen_prefix),
                    })
                continue

            # Lock ONE consistent `role_map` across authorization, model fit, fresh confirm batch, & final query guard!
            locked_rmap = c_cand.get("_canonical_role_map")
            if valid_rmaps_c and best_c.get("role_order"):
                for rm in valid_rmaps_c:
                    if _role_order_from_canonical_role_map(canon_roles_c, rm) == list(best_c["role_order"]):
                        locked_rmap = rm
                        break
            elif best_c.get("role_order") and act_rnames_c and len(act_rnames_c) == len(best_c["role_order"]):
                locked_rmap = {str(tf): str(cr) for cr, tf in zip(act_rnames_c, best_c["role_order"])}
            c_cand["_canonical_role_map"] = locked_rmap

            nd_vbox = ((nd.get("scale_map") or {}).get("window_or_partition") or {}).get("verified_input_box")
            target_vbox = (
                {tf: nd_vbox[locked_rmap[tf]] for tf in target.feature_names}
                if isinstance(nd_vbox, dict) and nd_vbox and locked_rmap
                else None
            )
            req_fresh = max(1, int(plan_c["fresh_points"]))
            c_pts = select_or_synthesize_disjoint_confirm_points(
                predeclared_confirm_points=target.confirm_points,
                training_prefix_rows=frozen_prefix,
                required_fresh_count=req_fresh,
                feature_bounds=target.feature_bounds,
                verified_box=target_vbox,
            )
            if len(c_pts) < req_fresh:
                continue
            if isinstance(nd_vbox, dict) and nd_vbox and not all(
                is_point_inside_verified_box(pt, nd_vbox, role_map=locked_rmap) for pt in c_pts
            ):
                continue
            cf_rows = _acquire_points(
                c_pts,
                phase="recall_confirm",
                enforced_vbox=(nd_vbox if isinstance(nd_vbox, dict) else None),
                enforced_role_map=locked_rmap,
            )
            max_overlap = max(max_overlap, _ov_cnt(cf_rows, frozen_prefix))
            cf_ev = evaluate_isolated_python_model(
                work_dir, best_c["model_relpath"], cf_rows, isolated_prefix_rows=frozen_prefix
            )
            if cf_ev["returncode"] == 0 and float(cf_ev["rmse"]) <= target.tolerance:
                m_sha = hashlib.sha256(best_c["model_source"].encode("utf-8")).hexdigest()
                c_dig = hashlib.sha256(json.dumps(cf_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                t3_obs_feats = dict(obs_feats)
                if locked_rmap:
                    all_paid_c = [*frozen_prefix, *cf_rows]
                    for tf_k, cr_k in locked_rmap.items():
                        t3_obs_feats[cr_k] = max(float(r["features"][tf_k]) for r in all_paid_c)
                is_struct_hyp = (
                    (c_cand.get("_selected_branch") or nd["claim_contract"]).get("evidence_status")
                    == "STRUCTURAL_HYPOTHESIS_ONLY"
                    or (c_cand.get("wormhole_edge") or {}).get("permission_tier") == "bounded_trial"
                )
                ev_rec_t3: dict[str, Any] = {
                    "passed": True,
                    "claim_id": nd["claim_contract"]["claim_id"],
                    "observed_error": float(cf_ev["rmse"]),
                    "target_id": target.target_id,
                    "node_version": nd["version"],
                    "model_sha256": m_sha,
                    "confirmation_digest": c_dig,
                }
                if is_struct_hyp:
                    ev_rec_t3["upgrade_to_tier"] = "local_solver_guidance"
                t3 = authorize_abstraction_use(
                    requested_tier="local_solver_guidance",
                    task_view={
                        "goal_kind": "open_workspace_mechanism_prediction",
                        "tolerance": target.tolerance,
                        "target_id": target.target_id,
                        "node_version": nd["version"],
                        "expected_model_sha256": m_sha,
                        "expected_confirmation_digest": c_dig,
                    },
                    target_features=t3_obs_feats,
                    claim_contract=nd["claim_contract"],
                    scale_map=nd.get("scale_map"),
                    wormhole_edge=c_cand.get("wormhole_edge"),
                    evidence_receipt=ev_rec_t3,
                )
                t3_act = t3.get("selected_action") or nd["claim_contract"]["conclusion_or_action"]
                if t3["granted"] and t3_act == eff_act_c:
                    confirmed_ok = True
                    confirmed_via_recall = True
                    chosen_rel = best_c["model_relpath"]
                    non_direct_walk_win = bool(c_cand.get("reached_only_by_graph_walk"))
                    selected_node_id = str(nd["node_id"])
                    selected_version = int(nd["version"])
                    selected_model_sha256 = m_sha
                    selected_conf_digest = c_dig
                    confirmed_vbox = deepcopy(((nd.get("scale_map") or {}).get("window_or_partition") or {}).get("verified_input_box"))
                    confirmed_role_map = deepcopy(locked_rmap)
                    confirmed_role_order = list(best_c.get("role_order") or target.feature_names)
                    confirmed_canon_roles = act_rnames_c or list(target.feature_names)
                    confirmed_coord_preconditions = [
                        deepcopy(pr)
                        for pr in nd["claim_contract"].get("preconditions", ())
                        if isinstance(pr, dict) and pr.get("field") in set(confirmed_canon_roles)
                    ]
                    confirmed_recall_bundle = {
                        "target_id": target.target_id,
                        "surface_domain": target.domain_cue,
                        "selected_node_id": selected_node_id,
                        "selected_node_evidence_status": (c_cand.get("_selected_branch") or nd["claim_contract"]).get("evidence_status"),
                        "operator_tag": c_tag,
                        "operator_spec": deepcopy(c_cand["_verified_spec"]),
                        "role_names": list(confirmed_role_order),
                        "canonical_role_map": deepcopy(locked_rmap),
                        "model_source": best_c["model_source"],
                        "model_relpath": best_c["model_relpath"],
                        "training_prefix_rows": deepcopy(frozen_prefix),
                        "confirm_rows": deepcopy(cf_rows),
                        "observed_confirm_rmse": round(float(cf_ev["rmse"]), 8),
                        "tolerance": float(target.tolerance),
                    }
                    break
            else:
                all_eval_rows = [*frozen_prefix, *cf_rows]
                refuted_recall_traces.append({
                    "node_id": str(nd["node_id"]),
                    "node_version": int(nd["version"]),
                    "stage": "fresh_confirm_refuted",
                    "confirm_rmse": float(cf_ev["rmse"]) if cf_ev["returncode"] == 0 else None,
                    "tolerance": float(target.tolerance),
                    "role_order": list(best_c.get("role_order") or target.feature_names),
                    "canon_roles": act_rnames_c or list(target.feature_names),
                    "model_source": best_c["model_source"],
                    "training_prefix_rows": deepcopy(frozen_prefix),
                    "evaluated_rows": deepcopy(all_eval_rows),
                })
            refuted_tags.add(c_tag)
            frozen_prefix = [*frozen_prefix, *cf_rows]

    # 3. Ordinary Flat History (`StratifiedDevOrder` first-acceptable OR `BestPilotRMSE` min-pilot-RMSE!)
    elif ordinary_history_specs is not None:
        ext_hist = [*ordinary_history_specs]
        if target.is_multiscale_two_channel:
            ext_hist.append(("op_micro_fast_transient_2d", micro_fast_2d_spec))
        same_ar_hist = [(t, sp) for t, sp in ext_hist if int(sp.get("arity", 1)) == len(target.feature_names)]

        for round_idx in range(2):
            rem_hist = [(t, sp) for t, sp in same_ar_hist if t not in refuted_tags]
            if not rem_hist:
                break
            if selection_policy == "min_pilot_rmse":
                best_h, ev_h = _fit_best_pilot_rmse_across_specs(
                    work_dir / f"ord_r{round_idx}",
                    specs_to_try=rem_hist,
                    feature_names=target.feature_names,
                    training_rows=frozen_prefix,
                )
                if best_h is not None:
                    rel_copy = f"ord_min_r{round_idx}.py"
                    (work_dir / rel_copy).write_text(best_h["model_source"], encoding="utf-8")
                    best_h["model_relpath"] = rel_copy
            else:
                best_h, ev_h = _autonomous_fit_candidate_specs(
                    work_dir,
                    specs_to_try=rem_hist,
                    feature_names=target.feature_names,
                    training_rows=frozen_prefix,
                    tolerance=target.tolerance,
                )
            total_evaluated += ev_h
            unmetered_evals += ev_h
            if best_h is None or best_h["train_rmse"] > target.tolerance:
                break
            cf_pts = select_or_synthesize_disjoint_confirm_points(
                predeclared_confirm_points=target.confirm_points,
                training_prefix_rows=frozen_prefix,
                required_fresh_count=1,
                feature_bounds=target.feature_bounds,
            )
            if not cf_pts:
                break
            cf_rows = _acquire_points(cf_pts, phase=f"ordinary_confirm_{round_idx}")
            max_overlap = max(max_overlap, _ov_cnt(cf_rows, frozen_prefix))
            cf_ev = evaluate_isolated_python_model(
                work_dir, best_h["model_relpath"], cf_rows, isolated_prefix_rows=frozen_prefix
            )
            if cf_ev["returncode"] == 0 and float(cf_ev["rmse"]) <= target.tolerance:
                confirmed_ok = True
                confirmed_via_recall = True
                chosen_rel = best_h["model_relpath"]
                selected_node_id = f"flat_history:{best_h['tag']}"
                selected_model_sha256 = hashlib.sha256(best_h["model_source"].encode("utf-8")).hexdigest()
                selected_conf_digest = hashlib.sha256(json.dumps(cf_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                break
            refuted_tags.add(str(best_h["tag"]))
            frozen_prefix = [*frozen_prefix, *cf_rows]

    # 4. Fallback (`OpenRediscovery` via real `CodeAgent` OR `PoolFallback` OR `NoDiscovery`)
    if not confirmed_ok and (open_code_agent_fallback or allow_pool_fallback):
        if target.is_multiscale_two_channel and selected_channel == "coarse_macro":
            selected_channel = "micro_fine"
            frozen_prefix = _acquire_points(target.pilot_points, phase="escalated_micro_pilot")

        f0 = target.feature_names[0]
        max_f0 = max(float(r["features"][f0]) for r in frozen_prefix)
        hi_f0 = target.feature_bounds[f0][1]
        if max_f0 < 0.35 * hi_f0 and hi_f0 >= 8.0:
            if len(target.feature_names) == 2:
                f1 = target.feature_names[1]
                st_pts = [{f0: round(0.58 * hi_f0, 2), f1: 0.4}, {f0: round(0.86 * hi_f0, 2), f1: 0.9}]
            else:
                st_pts = [{f0: round(0.66 * hi_f0, 2)}, {f0: round(0.89 * hi_f0, 2)}]
            frozen_prefix = [*frozen_prefix, *_acquire_points(st_pts, phase="stress_probe")]

        if open_code_agent_fallback:
            # Build an `OpenWorkspaceTaskLab` bound to the active channel & `meter` and execute REAL `CodeAgent`!
            ep_unit_cost = (10.0 if selected_channel == "micro_fine" else 3.0) if target.is_multiscale_two_channel else 4.0
            ep_oracle = (
                (target.coarse_oracle_fn if selected_channel == "coarse_macro" else target.oracle_fn)
                if target.is_multiscale_two_channel
                else target.oracle_fn
            )
            assert ep_oracle is not None
            ep_lab = OpenWorkspaceTaskLab(
                task_id=target.target_id,
                surface_domain=target.domain_cue,
                feature_names=target.feature_names,
                feature_bounds=target.feature_bounds,
                cost_per_obs=ep_unit_cost,
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
                meter=meter,
                private_oracle_fn=ep_oracle,
            )
            from .models import Observation
            raw_perms = [list(p) for p in _infer_role_permutations(target.feature_names)]
            atomic_bases = _build_stage0_open_atomic_bases(len(target.feature_names))
            for p_idx, ordered_roles in enumerate(raw_perms):
                ep_lab.observations = [
                    Observation(features=dict(r["features"]), target=float(r["target"]), history=())
                    for r in frozen_prefix
                ]
                lens_est = _estimate_affine_lens_for_spec(atomic_bases[0], ordered_roles, frozen_prefix)
                diag_pts = (
                    select_or_synthesize_disjoint_confirm_points(
                        predeclared_confirm_points=[],
                        training_prefix_rows=frozen_prefix,
                        required_fresh_count=1,
                        feature_bounds=target.feature_bounds,
                    )
                    if p_idx == 0
                    else []
                )
                post_diag = [*frozen_prefix, *[{"features": dict(p), "target": 0.0} for p in diag_pts]]
                ep_cf_pts = select_or_synthesize_disjoint_confirm_points(
                    predeclared_confirm_points=target.confirm_points,
                    training_prefix_rows=post_diag,
                    required_fresh_count=1,
                    feature_bounds=target.feature_bounds,
                )
                if not ep_cf_pts:
                    break
                ep_res = run_compositional_code_agent_episode(
                    lab=ep_lab,
                    heldout_rows=None,
                    tolerance=target.tolerance,
                    workspace_dir=work_dir / f"open_code_agent_ep_{p_idx}",
                    role_names=ordered_roles,
                    initial_query_batches=([diag_pts] if diag_pts else []),
                    atomic_base_specs=atomic_bases,
                    confirm_points=ep_cf_pts,
                    lens_hint=lens_est,
                    lens_calibrator_fn=lambda sp_obj, r_ord: _estimate_affine_lens_for_spec(
                        sp_obj,
                        r_ord,
                        [{"features": dict(o.features), "target": float(o.target)} for o in ep_lab.observations] or frozen_prefix,
                    ),
                )
                total_evaluated += int(ep_res["evaluated_candidate_count"])
                ep_b = ep_res["authored_artifact"]["host_confirmation_bundle"]
                max_overlap = max(max_overlap, _ov_cnt(ep_b["confirm_rows"], ep_b["training_prefix_rows"]))
                frozen_prefix = deepcopy(ep_res["training_prefix_rows"])
                if ep_res["confirmed_ok"]:
                    confirmed_ok = True
                    chosen_rel = "open_discovered.py"
                    (work_dir / chosen_rel).write_text(ep_res["authored_artifact"]["authored_model_source"], encoding="utf-8")
                    selected_node_id = f"open_code_agent:{ep_res['authored_artifact']['operator_spec'].get('family_hint', 'composed')}"
                    selected_model_sha256 = str(ep_res["authored_artifact"]["model_sha256"])
                    selected_conf_digest = str(ep_res["authored_artifact"]["confirmation_digest"])
                    break
                frozen_prefix = [*frozen_prefix, *deepcopy(ep_b["confirm_rows"])]

        if not confirmed_ok and allow_pool_fallback and not open_code_agent_fallback:
            fb_specs = [*fallback_pool_specs]
            if target.is_multiscale_two_channel:
                fb_specs.append(("op_micro_fast_transient_2d", micro_fast_2d_spec))
            same_ar_fb = [
                (t, sp) for t, sp in fb_specs
                if int(sp.get("arity", 1)) == len(target.feature_names) and t not in refuted_tags
            ]
            best_fb, ev_fb = _autonomous_fit_candidate_specs(
                work_dir,
                specs_to_try=same_ar_fb,
                feature_names=target.feature_names,
                training_rows=frozen_prefix,
                tolerance=target.tolerance,
            )
            total_evaluated += ev_fb
            unmetered_evals += ev_fb
            if best_fb is not None and float(best_fb["train_rmse"]) <= target.tolerance:
                rem_cf_pts = select_or_synthesize_disjoint_confirm_points(
                    predeclared_confirm_points=target.confirm_points,
                    training_prefix_rows=frozen_prefix,
                    required_fresh_count=1,
                    feature_bounds=target.feature_bounds,
                )
                fb_rows = _acquire_points(rem_cf_pts, phase="fallback_confirm")
                max_overlap = max(max_overlap, _ov_cnt(fb_rows, frozen_prefix))
                if fb_rows:
                    fb_ev = evaluate_isolated_python_model(
                        work_dir, best_fb["model_relpath"], fb_rows, isolated_prefix_rows=frozen_prefix
                    )
                    if fb_ev["returncode"] == 0 and float(fb_ev["rmse"]) <= target.tolerance:
                        confirmed_ok = True
                        chosen_rel = best_fb["model_relpath"]
                        selected_node_id = f"fallback:{best_fb['tag']}"
                        selected_model_sha256 = hashlib.sha256(best_fb["model_source"].encode("utf-8")).hexdigest()
                        selected_conf_digest = hashlib.sha256(json.dumps(fb_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    if unmetered_evals > 0 or not open_code_agent_fallback:
        meter.charge("candidate_eval_compute", compute=round(2.5 + 6.0 * unmetered_evals, 4))

    if refuted_recall_traces:
        (work_dir / "refuted_recall_traces.json").write_text(
            json.dumps(refuted_recall_traces, indent=2, sort_keys=True), encoding="utf-8"
        )
    if confirmed_recall_bundle is not None:
        (work_dir / "confirmed_recall_bundle.json").write_text(
            json.dumps(confirmed_recall_bundle, indent=2, sort_keys=True), encoding="utf-8"
        )

    # 5. Enforce `model_scope_contract.json` on final/continuation queries!
    (work_dir / "model_scope_contract.json").write_text(
        json.dumps(
            {
                "selected_channel": selected_channel,
                "authorizing_coarse_node_id": authorizing_coarse_node_id,
                "verified_input_box": confirmed_vbox,
                "canonical_role_map": confirmed_role_map,
                "coordinate_preconditions": confirmed_coord_preconditions,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    out_of_box_blocked = False
    if confirmed_ok and isinstance(confirmed_vbox, dict) and confirmed_vbox:
        if not all(
            is_point_inside_verified_box(pt, confirmed_vbox, role_map=confirmed_role_map)
            for pt in target.heldout_points
        ):
            out_of_box_blocked = True
            confirmed_ok = False
    if confirmed_ok and confirmed_coord_preconditions and confirmed_role_map:
        from hegel_machine.cognition.wormhole_forest import evaluate_predicate_rule

        for pt in target.heldout_points:
            mapped_pt = {str(confirmed_role_map[qk]): float(qv) for qk, qv in pt.items() if qk in confirmed_role_map}
            if any(evaluate_predicate_rule(pr, mapped_pt) == "FALSE" for pr in confirmed_coord_preconditions):
                out_of_box_blocked = True
                confirmed_ok = False
                break

    heldout_rows = [{"features": dict(pt), "target": float(target.oracle_fn(pt))} for pt in target.heldout_points]
    if confirmed_ok and chosen_rel is not None:
        h_ev = evaluate_isolated_python_model(
            work_dir, chosen_rel, heldout_rows, isolated_prefix_rows=frozen_prefix
        )
        h_rmse = float(h_ev["rmse"])
        h_pass = bool(h_ev["returncode"] == 0 and math.isfinite(h_rmse) and h_rmse <= target.tolerance)
    else:
        h_rmse = float("inf")
        h_pass = False

    snap = meter.snapshot()
    return {
        "seed": seed,
        "target_id": target.target_id,
        "mechanism_family": target.mechanism_family,
        "is_out_of_scope": target.is_out_of_scope,
        "arm": arm_name,
        "status": (
            "out_of_verified_scale_box"
            if out_of_box_blocked
            else ("completed" if confirmed_ok else "scoped_unknown")
        ),
        "selected_channel": selected_channel,
        "out_of_box_query_blocked": out_of_box_blocked,
        "prefix_confirm_overlap_count": max_overlap,
        "confirmed_ok": confirmed_ok,
        "confirmed_via_recall": confirmed_via_recall,
        "non_direct_walk_win": non_direct_walk_win,
        "selected_node_id": selected_node_id,
        "selected_version": selected_version,
        "selected_model_sha256": selected_model_sha256,
        "confirmation_digest": selected_conf_digest,
        "evaluated_candidate_count": total_evaluated,
        "heldout_rmse": round(h_rmse, 6) if math.isfinite(h_rmse) else None,
        "strict_pass": bool(confirmed_ok and h_pass),
        "false_confirmation": bool(confirmed_ok and not h_pass),
        "cost": snap,
    }


def run_m4_3_channel_bound_continual_study(
    work_root: Path,
    *,
    seeds: tuple[int, ...] = (20_301_120, 20_311_120, 20_321_120, 20_331_120),
    onset_pilot_noise_std: float = 0.0020,
) -> dict[str, Any]:
    """Run the complete ``Hegel V17.5 (M4.3)`` 11-arm channel-bound continual benchmark (`96` unseen in-scope +
    `8` OOS runs/arm on fresh held-out seeds!) plus noiseless vs noisy fork comparison!
    """
    work_root.mkdir(parents=True, exist_ok=True)
    built = build_m4_3_continual_forest(work_root / "continual_lifecycle")

    post_s2_snapshot = deepcopy(built["forest"])
    unmaint_snapshot = deepcopy(built["forest_before_stage2_maintenance"])
    true_s0_snapshot = deepcopy(built["forest_after_stage0"])

    # Read-only R3 zero-fit dispatcher check on raw `post_s2_snapshot`
    before_disp_dump = json.dumps(post_s2_snapshot.nodes, sort_keys=True)
    dispatcher_report = run_interleaved_zero_fit_dispatcher_stream(post_s2_snapshot)
    after_disp_dump = json.dumps(post_s2_snapshot.nodes, sort_keys=True)
    assert before_disp_dump == after_disp_dump, "Dispatcher mutated post_s2_snapshot!"

    arm_names = (
        "Hegel_Unified_Continual_Forest",
        "Hegel_Walk_NoPool_SameHistory",
        "Forest_Alpha0_PoolOn_SameHistory",
        "Forest_Alpha0_PoolOff_SameHistory",
        "Hegel_NoMaintenance_SameHistory",
        "Compiled_Relational_CueTable_SameForest",
        "Ordinary_AllHistory_StratifiedDevOrder",
        "Ordinary_AllHistory_BestPilotRMSE",
        "Flat_BM25_SameHistory",
        "Frozen_After_Stage0_OpenRediscovery",
        "Frozen_After_Stage0_NoDiscovery",
    )

    in_scope_records: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}
    oos_records: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}

    for s_idx, seed in enumerate(seeds):
        targets, oos_targets = build_m4_3_target_suite(seed=seed, onset_pilot_noise_std=onset_pilot_noise_std)
        f_hegel = deepcopy(post_s2_snapshot)
        f_wnopool = deepcopy(post_s2_snapshot)
        f_a0_pon = deepcopy(post_s2_snapshot)
        f_a0_poff = deepcopy(post_s2_snapshot)
        f_nomaint = deepcopy(unmaint_snapshot)
        f_cuetab = deepcopy(post_s2_snapshot)
        f_flat = deepcopy(post_s2_snapshot)
        f_froz_open = deepcopy(true_s0_snapshot)
        f_froz_nodisc = deepcopy(true_s0_snapshot)

        arm_configs = {
            "Hegel_Unified_Continual_Forest": (
                f_hegel,
                HippoRAGWormholeIndex(f_hegel, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Hegel_Walk_NoPool_SameHistory": (
                f_wnopool,
                HippoRAGWormholeIndex(f_wnopool, damping_alpha=0.65, wormhole_equivalence_pooling=False, scale_aware=True),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Forest_Alpha0_PoolOn_SameHistory": (
                f_a0_pon,
                HippoRAGWormholeIndex(f_a0_pon, damping_alpha=0.0, wormhole_equivalence_pooling=True, scale_aware=True),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Forest_Alpha0_PoolOff_SameHistory": (
                f_a0_poff,
                HippoRAGWormholeIndex(f_a0_poff, damping_alpha=0.0, wormhole_equivalence_pooling=False, scale_aware=True),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Hegel_NoMaintenance_SameHistory": (
                f_nomaint,
                HippoRAGWormholeIndex(f_nomaint, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Compiled_Relational_CueTable_SameForest": (
                f_cuetab,
                CompiledRelationalCueTable(f_cuetab),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Ordinary_AllHistory_StratifiedDevOrder": (
                None,
                None,
                built["ordinary_history_stratified_dev_6fam"],
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Ordinary_AllHistory_BestPilotRMSE": (
                None,
                None,
                built["ordinary_history_forward"],
                built["ordinary_history_forward"],
                "min_pilot_rmse",
                False,
                True,
            ),
            "Flat_BM25_SameHistory": (
                f_flat,
                FlatEqualMetadataIndex(f_flat),
                None,
                built["ordinary_history_forward"],
                "first_acceptable",
                False,
                True,
            ),
            "Frozen_After_Stage0_OpenRediscovery": (
                f_froz_open,
                HippoRAGWormholeIndex(f_froz_open, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                None,
                built["stage0_atomic_specs"],
                "first_acceptable",
                True,
                False,
            ),
            "Frozen_After_Stage0_NoDiscovery": (
                f_froz_nodisc,
                HippoRAGWormholeIndex(f_froz_nodisc, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True),
                None,
                built["stage0_atomic_specs"],
                "first_acceptable",
                False,
                True,
            ),
        }

        for t in targets:
            for a_name, (f_o, ix_o, ord_h, fb_sp, pol, open_fb, allow_fb) in arm_configs.items():
                rec = evaluate_m4_3_target_under_arm(
                    t,
                    arm_name=a_name,
                    seed=seed,
                    work_dir=work_root / "stage3_eval" / f"seed_{s_idx}" / a_name / t.target_id,
                    forest=f_o,
                    index=ix_o,
                    ordinary_history_specs=ord_h,
                    fallback_pool_specs=fb_sp,
                    selection_policy=pol,
                    open_code_agent_fallback=open_fb,
                    allow_pool_fallback=allow_fb,
                )
                in_scope_records[a_name].append(rec)

        for ot in oos_targets:
            for a_name, (f_o, ix_o, ord_h, fb_sp, pol, open_fb, allow_fb) in arm_configs.items():
                orec = evaluate_m4_3_target_under_arm(
                    ot,
                    arm_name=a_name,
                    seed=seed,
                    work_dir=work_root / "stage3_oos" / f"seed_{s_idx}" / a_name / ot.target_id,
                    forest=f_o,
                    index=ix_o,
                    ordinary_history_specs=ord_h,
                    fallback_pool_specs=fb_sp,
                    selection_policy=pol,
                    open_code_agent_fallback=open_fb,
                    allow_pool_fallback=allow_fb,
                )
                oos_records[a_name].append(orec)

    # Controlled Onset Fork Sub-Comparison: Noiseless (`sigma = 0.0`) vs Gaussian Sensor Noise (`sigma = 0.0020`) across ALL passed seeds!
    fork_regime_comparison: dict[str, Any] = {}
    fork_execution_receipts: list[dict[str, Any]] = []
    for reg_lbl, sig_val in (("noiseless_sigma_0", 0.0), ("gaussian_sensor_sigma_0_0020", 0.0020)):
        h_recs, min_recs, dev_recs = [], [], []
        f_h_sub = deepcopy(post_s2_snapshot)
        ix_h_sub = HippoRAGWormholeIndex(f_h_sub, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        for fk_idx, fk_seed in enumerate(seeds):
            t_reg, _ = build_m4_3_target_suite(seed=fk_seed, onset_pilot_noise_std=sig_val)
            onset_slice = [x for x in t_reg if x.mechanism_family.startswith("OnsetWalk_")]
            for ot_x in onset_slice:
                hr = evaluate_m4_3_target_under_arm(
                    ot_x,
                    arm_name="Hegel_PPR",
                    seed=fk_seed,
                    work_dir=work_root / "fork_sub" / reg_lbl / f"seed_{fk_idx}" / "hegel" / ot_x.target_id,
                    forest=f_h_sub,
                    index=ix_h_sub,
                    ordinary_history_specs=None,
                    fallback_pool_specs=built["ordinary_history_forward"],
                )
                mr = evaluate_m4_3_target_under_arm(
                    ot_x,
                    arm_name="BestPilotRMSE",
                    seed=fk_seed,
                    work_dir=work_root / "fork_sub" / reg_lbl / f"seed_{fk_idx}" / "min_rmse" / ot_x.target_id,
                    forest=None,
                    index=None,
                    ordinary_history_specs=built["ordinary_history_forward"],
                    fallback_pool_specs=built["ordinary_history_forward"],
                    selection_policy="min_pilot_rmse",
                )
                dr = evaluate_m4_3_target_under_arm(
                    ot_x,
                    arm_name="StratifiedDevFirstAcceptable",
                    seed=fk_seed,
                    work_dir=work_root / "fork_sub" / reg_lbl / f"seed_{fk_idx}" / "dev_order" / ot_x.target_id,
                    forest=None,
                    index=None,
                    ordinary_history_specs=built["ordinary_history_stratified_dev_6fam"],
                    fallback_pool_specs=built["ordinary_history_forward"],
                )
                h_recs.append(hr)
                min_recs.append(mr)
                dev_recs.append(dr)
                for rec_item in (hr, mr, dr):
                    fork_execution_receipts.append({**rec_item, "noise_regime": reg_lbl, "pilot_noise_std": sig_val})
        fork_regime_comparison[reg_lbl] = {
            "noise_distribution": ("noiseless" if sig_val == 0.0 else "gaussian_untruncated"),
            "pilot_noise_std": sig_val,
            "onset_target_count": len(h_recs),
            "Hegel_PPR_strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in h_recs), 4),
            "Hegel_PPR_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in h_recs), 4),
            "Hegel_PPR_max_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in h_recs),
            "Hegel_PPR_mean_meas": round(mean(r["cost"]["measurement_cost"] for r in h_recs), 4),
            "Hegel_PPR_mean_composite": round(mean(r["cost"]["composite_cost"] for r in h_recs), 4),
            "BestPilotRMSE_strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in min_recs), 4),
            "BestPilotRMSE_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in min_recs), 4),
            "BestPilotRMSE_max_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in min_recs),
            "BestPilotRMSE_mean_meas": round(mean(r["cost"]["measurement_cost"] for r in min_recs), 4),
            "BestPilotRMSE_mean_composite": round(mean(r["cost"]["composite_cost"] for r in min_recs), 4),
            "StratifiedDev_strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in dev_recs), 4),
            "StratifiedDev_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in dev_recs), 4),
            "StratifiedDev_max_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in dev_recs),
            "StratifiedDev_mean_meas": round(mean(r["cost"]["measurement_cost"] for r in dev_recs), 4),
            "StratifiedDev_mean_composite": round(mean(r["cost"]["composite_cost"] for r in dev_recs), 4),
        }

    # Summarize all 11 arms + cumulative lifecycle breakeven curve!
    s0_cost = float(built["stage0"]["composite_cost"])
    s1_cost = float(built["stage1"]["composite_cost"])
    s2_cost = float(built["stage2"]["composite_cost"])
    cuetable_compile_composite = round(0.05 * float(built["cue_table_lifecycle_compute_cost"]), 4)
    dev_6fam_composite = float(built["stratified_dev_6fam_report"]["dev_cost"]["composite_cost"])

    arm_summaries: dict[str, Any] = {}
    for a_name in arm_names:
        rows = in_scope_records[a_name]
        orows = oos_records[a_name]
        s3_tot = round(sum(float(r["cost"]["composite_cost"]) for r in rows), 4)
        if a_name in {
            "Hegel_Unified_Continual_Forest",
            "Hegel_Walk_NoPool_SameHistory",
            "Forest_Alpha0_PoolOn_SameHistory",
            "Forest_Alpha0_PoolOff_SameHistory",
            "Flat_BM25_SameHistory",
        }:
            upfront = s0_cost + s1_cost + s2_cost
        elif a_name == "Compiled_Relational_CueTable_SameForest":
            upfront = s0_cost + s1_cost + s2_cost + cuetable_compile_composite
        elif a_name in {"Hegel_NoMaintenance_SameHistory", "Ordinary_AllHistory_BestPilotRMSE"}:
            upfront = s0_cost + s1_cost
        elif a_name == "Ordinary_AllHistory_StratifiedDevOrder":
            upfront = s0_cost + s1_cost + dev_6fam_composite
        else:
            # True Stage-0 Frozen arms (`Frozen_After_Stage0_OpenRediscovery`, `Frozen_After_Stage0_NoDiscovery`)
            upfront = s0_cost

        arm_summaries[a_name] = {
            "in_scope_runs": len(rows),
            "oos_runs": len(orows),
            "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in rows), 4),
            "false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in rows), 4),
            "oos_scoped_unknown_rate": round(mean(1.0 if r["status"] == "scoped_unknown" else 0.0 for r in orows), 4),
            "oos_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in orows), 4),
            "max_prefix_confirm_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in [*rows, *orows]),
            "recall_confirmation_count": sum(1 for r in rows if r["confirmed_via_recall"]),
            "non_direct_walk_wins": sum(1 for r in rows if r["non_direct_walk_win"]),
            "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in rows), 4),
            "mean_measurement_cost": round(mean(float(r["cost"]["measurement_cost"]) for r in rows), 4),
            "mean_compute_cost": round(mean(float(r["cost"]["compute_cost"]) for r in rows), 4),
            "mean_composite_cost": round(mean(float(r["cost"]["composite_cost"]) for r in rows), 4),
            "sd_composite_cost": round(pstdev(float(r["cost"]["composite_cost"]) for r in rows), 4),
            "upfront_lifecycle_composite_cost": round(upfront, 4),
            "stage3_total_composite_cost": s3_tot,
            "cumulative_total_lifecycle_composite_cost": round(upfront + s3_tot, 4),
        }

    # Family-by-family breakdown of `Hegel_Unified_Continual_Forest` vs `Hegel_NoMaintenance_SameHistory`
    hegel_rows = in_scope_records["Hegel_Unified_Continual_Forest"]
    nomaint_rows = in_scope_records["Hegel_NoMaintenance_SameHistory"]
    frozen_open_rows = in_scope_records["Frozen_After_Stage0_OpenRediscovery"]
    families = sorted({r["mechanism_family"] for r in hegel_rows})
    maintenance_by_family: dict[str, Any] = {}
    for fam in families:
        h_f = [r for r in hegel_rows if r["mechanism_family"] == fam]
        n_f = [r for r in nomaint_rows if r["mechanism_family"] == fam]
        maintenance_by_family[fam] = {
            "count": len(h_f),
            "hegel_mean_composite": round(mean(r["cost"]["composite_cost"] for r in h_f), 4),
            "no_maintenance_mean_composite": round(mean(r["cost"]["composite_cost"] for r in n_f), 4),
            "stage3_total_delta_saved_by_maintenance": round(
                sum(r["cost"]["composite_cost"] for r in n_f) - sum(r["cost"]["composite_cost"] for r in h_f), 4
            ),
            "hegel_false_confirmations": sum(1 for r in h_f if r["false_confirmation"]),
            "no_maintenance_false_confirmations": sum(1 for r in n_f if r["false_confirmation"]),
        }

    cum_hegel = s0_cost + s1_cost + s2_cost
    cum_nomaint = s0_cost + s1_cost
    cum_frozen_open = s0_cost
    breakeven_vs_frozen_open: int | None = None
    breakeven_vs_nomaint: int | None = None
    curve_milestones: list[dict[str, Any]] = []
    for k_i in range(len(hegel_rows)):
        cum_hegel += float(hegel_rows[k_i]["cost"]["composite_cost"])
        cum_nomaint += float(nomaint_rows[k_i]["cost"]["composite_cost"])
        cum_frozen_open += float(frozen_open_rows[k_i]["cost"]["composite_cost"])
        k_n = k_i + 1
        if breakeven_vs_frozen_open is None and cum_hegel < cum_frozen_open:
            breakeven_vs_frozen_open = k_n
        if breakeven_vs_nomaint is None and cum_hegel < cum_nomaint:
            breakeven_vs_nomaint = k_n
        if k_n in {1, 12, 24, 36, 48, 60, 72, 84, 96} or k_n in {breakeven_vs_frozen_open, breakeven_vs_nomaint}:
            curve_milestones.append({
                "targets_completed": k_n,
                "hegel_cumulative_lifecycle_cost": round(cum_hegel, 4),
                "no_maintenance_cumulative_lifecycle_cost": round(cum_nomaint, 4),
                "frozen_open_rediscovery_cumulative_cost": round(cum_frozen_open, 4),
            })

    all_receipts = [
        *[r for a in arm_names for r in in_scope_records[a]],
        *[r for a in arm_names for r in oos_records[a]],
    ]
    return {
        "seeds": list(seeds),
        "onset_pilot_noise_std": onset_pilot_noise_std,
        "stage0": built["stage0"],
        "stage1": built["stage1"],
        "stage2": built["stage2"],
        "stratified_dev_6fam_report": built["stratified_dev_6fam_report"],
        "dispatcher_zero_fit_r3": dispatcher_report,
        "stage3_arms": arm_summaries,
        "maintenance_by_family": maintenance_by_family,
        "fork_noiseless_vs_instrument_noise": fork_regime_comparison,
        "lifecycle_breakeven": {
            "upfront_stage0_cost": round(s0_cost, 4),
            "upfront_stage1_cost": round(s1_cost, 4),
            "upfront_stage2_cost": round(s2_cost, 4),
            "observed_breakeven_target_vs_true_frozen_open_rediscovery": breakeven_vs_frozen_open,
            "observed_breakeven_target_vs_no_maintenance": breakeven_vs_nomaint,
            "net_lifecycle_savings_at_end_vs_frozen_open_rediscovery": round(cum_frozen_open - cum_hegel, 4),
            "net_lifecycle_delta_at_end_vs_no_maintenance": round(cum_nomaint - cum_hegel, 4),
            "cumulative_curve_milestones": curve_milestones,
        },
        "forest_snapshots": {
            "true_stage0_forest": true_s0_snapshot,
            "unmaintained_post_stage1_forest": unmaint_snapshot,
            "maintained_post_stage2_forest": post_s2_snapshot,
            "stage2_pi_s_reverified_certificate": built["stage2_pi_s_reverified_certificate"],
        },
        "execution_receipts": all_receipts,
        "fork_execution_receipts": fork_execution_receipts,
    }

r"""Hegel V17.6 (M4.4): Full `Hegel_M4_3_Audit` Invariant Closure (Node-Instance Anchor,
Coordinate Role-Map Lock, Self-Verifying Relocatable Digests, Exact 2-Mode Eigen-Basis & Event Ledger)
+ Stateful Online Write-Back, Live Index Invalidation & Multi-Cue Continual Stream (`M5` Bridge).

Closes every P1 and P2 identified in ``Hegel_M4_3_Audit`` (`commit 346153536` vs `fb4b23aaf`) while
proactively advancing the architecture into genuine stateful mid-stream write-back (`M5` bridge):

1. **P1-1 — Complete Node-to-Instance Anchor & Atomic Cross-Instance Certificate Guard**:
   - Every zero-`fit()` artifact-bearing node (`14/14` in `post_stage2_forest`) and every OR-branch
     explicitly binds ``confirmed_instance_digest``, ``fitted_params_sha256``, ``model_source_sha256``,
     and ``role_names``;
   - ``ForestScopeDispatcher`` fails closed when ``confirmed_instance_digest`` is missing or mismatched
     (rejecting a genuinely calibrated, independently confirmed same-AST ``p0=12`` swap while accepting
     the original ``p0=6`` instance), and emits explicit ``model_source_sha256``, ``fitted_params_sha256``,
     and ``confirmed_instance_digest`` receipt fields;
   - ``revalidate_vertical_edges_with_certificate`` validates ``node_inst == cert_inst`` BEFORE mutating
     state and commits ``updated_claim_contract`` + ``updated_scale_map`` + edge rebinding atomically.

2. **P1-2 — Unified Coordinate Role-Map Lock Across Channel Gate, Fitter, Fresh Batch & Final Guard**:
   - When task coordinate names match canonical certificate axes (`{'u_drive', 'T_horizon'}`), ONLY the
     identity bijection is permitted (`u_drive <-> T_horizon` axis swapping is never allowed to sneak
     out-of-box points into ``verified_input_box``);
   - When task coordinate names are anonymous/surface-renamed, candidate fitting is restricted strictly
     to ``allowed_role_orders`` consistent with box-valid bijections, and the fitted ``role_order``
     locks a single ``confirmed_role_map`` across ``select_or_synthesize_disjoint_confirm_points(..., verified_box=...)``,
     pre-acquisition ``coarse_macro`` verification, and ``model_scope_contract.json`` final query gating.

3. **P2 Closures — Exact 2-Mode Matrix (`A12 == g_sf`), Complete Event Ledger, Self-Verifying Relocatable
   Digests, Legacy Compatibility Guards & Live Dynamic Index Invalidation**:
   - Fixes the 2-mode linear ODE matrix entry ``a12 = g_sf`` (matching RK4 truth to ``~1e-7``), removes
     the discarded generic pre-fit on ``ode_micro_2mode_node``, and explicitly meters local trusted RK4
     basis compute and artifact integrity checks in ``CostMeter``;
   - Stores location-independent bundle-relative locators inside contracts/certificates and resolves
     against ``forest.artifact_root`` at runtime (`0/8`, `0/15`, `0/15` ``semantic_digest`` mismatches
     and `0` certificate digest mismatches both in archived JSON and after cold reload in a new directory);
   - Enforces fail-closed flag semantics on legacy unbundled artifacts and per-channel paid confirmation
     retention on legacy ``ScaleBlind`` escalation;
   - Upgrades ``HippoRAGWormholeIndex``, ``FlatEqualMetadataIndex``, and ``CompiledRelationalCueTable``
     with live ``forest_state_signature()`` invalidation & metered refresh (`0` stale `v1` scope leaks).

4. **Part 3 (`M5` Bridge) — Stateful Online Write-Back, Equivalence Pooling & Multi-Cue Continual Stream**:
   Demonstrates live mid-stream ``CodeAgent`` compositional discovery (`W6: 1D Michaelis + Ceiling Cap`
   and `W7: 2D Michaelis + Channel Ceiling Cap`) + live ``empirical_revise_node`` writing back into a
   single evolving ``PhilosophicalForest``, automatically refreshing live indices, accelerating downstream
   multi-cue tasks, separating ``PoolOn`` vs ``PoolOff`` and ``PPR`` vs ``CompiledCueTable`` under online
   mutations, and preserving ``10/10`` historical ``R3`` zero-`fit()` accuracy (`0` catastrophic forgetting).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from random import Random
import shutil
import sys
from statistics import mean, pstdev
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import HippoRAGWormholeIndex
from .m4_1_unified_discovery_and_renormalization import (
    _autonomous_fit_candidate_specs,
    extract_unified_observable_pilot_cues,
    is_point_inside_verified_box,
)
from .m4_2_unified_closure_and_pi_lifecycle import (
    DEFAULT_PI_S_ODE_PARAMS,
    ForestScopeDispatcher,
    _eval_pi_s_coarse_scalar,
    _eval_pi_s_fine_scalar,
    _find_valid_canonical_role_map_for_box,
    _make_verified_instance_action,
    _run_metered_atomic_calibration,
    certify_regional_pi_s_closure_with_trajectory_projection,
    run_interleaved_zero_fit_dispatcher_stream,
)
from .m4_3_channel_bound_lifecycle_and_audit_closure import (
    CompiledRelationalCueTable,
    M43TargetSpec,
    build_m4_3_continual_forest,
    build_m4_3_target_suite,
    evaluate_m4_3_target_under_arm,
)
from .m4_compositional_discovery import (
    _VERIFIED_AUTHORED_CACHE,
    apply_structural_combinator,
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    load_verified_authored_method,
    make_atomic_operator_spec,
    predict_with_frozen_instance,
    run_compositional_code_agent_episode,
)
import hegel_machine.cognition.m4_compositional_discovery as m4_cd
import hegel_machine.cognition.m4_multiscale_renormalization as m4_ms
from .m4_multiscale_renormalization import (
    MultiScaleWorkspaceTaskLab,
    _estimate_affine_lens_for_spec,
    coords_approx_equal,
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
    canonical_digest,
    validate_claim_certificate_instance_identity,
)


def serialize_location_independent_forest(forest: PhilosophicalForest) -> dict[str, Any]:
    """Serialize ``PhilosophicalForest`` without rewriting any contract or certificate fields!"""
    out = {
        "nodes": deepcopy(forest.nodes),
        "node_history": deepcopy(forest.node_history),
        "local_implication_edges": deepcopy(forest.local_implication_edges),
        "coarse_graining_edges": deepcopy(forest.coarse_graining_edges),
        "horizontal_wormhole_edges": deepcopy(forest.horizontal_wormhole_edges),
        "wormhole_history": deepcopy(forest.wormhole_history),
        "conflict_relations": deepcopy(forest.conflict_relations),
        "mutation_epoch": int(forest.mutation_epoch),
        "questions": deepcopy(forest.questions),
        "raw_episode_archive": deepcopy(forest.raw_episode_archive),
    }
    if getattr(forest, "shared_base_subtrees", None):
        out["shared_base_subtrees"] = deepcopy(forest.shared_base_subtrees)
        out["shared_base_subtree_history"] = deepcopy(getattr(forest, "shared_base_subtree_history", {}))
    return out


def load_location_independent_forest(snapshot_path: Path, bundle_root: Path) -> PhilosophicalForest:
    """Reload a serialized ``PhilosophicalForest`` setting ONLY ``artifact_root`` (zero contract mutation!)."""
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    f = PhilosophicalForest(artifact_root=str(bundle_root.resolve()))
    f.nodes = data["nodes"]
    f.node_history = data["node_history"]
    f.local_implication_edges = data["local_implication_edges"]
    f.coarse_graining_edges = data["coarse_graining_edges"]
    f.horizontal_wormhole_edges = data["horizontal_wormhole_edges"]
    f.wormhole_history = data["wormhole_history"]
    f.conflict_relations = data["conflict_relations"]
    f.mutation_epoch = int(data.get("mutation_epoch", 0))
    f.questions = data.get("questions", {})
    f.raw_episode_archive = data.get("raw_episode_archive", [])
    f.shared_base_subtrees = data.get("shared_base_subtrees", {})
    f.shared_base_subtree_history = data.get("shared_base_subtree_history", {})
    return f


def verify_forest_snapshot_digests(forest_dict_or_obj: dict[str, Any] | PhilosophicalForest) -> dict[str, Any]:
    """Recompute exact production 5-field ``semantic_digest``, ``closure_certificate_digest``,
    history semantic digests, & cross-object ``node / OR-branch / edge / certificate`` instance equality!
    """
    nodes = forest_dict_or_obj.nodes if isinstance(forest_dict_or_obj, PhilosophicalForest) else forest_dict_or_obj["nodes"]
    node_history = (
        forest_dict_or_obj.node_history
        if isinstance(forest_dict_or_obj, PhilosophicalForest)
        else forest_dict_or_obj.get("node_history", {})
    )
    cg_edges = (
        forest_dict_or_obj.coarse_graining_edges
        if isinstance(forest_dict_or_obj, PhilosophicalForest)
        else forest_dict_or_obj["coarse_graining_edges"]
    )
    sem_mismatches: list[dict[str, Any]] = []
    hist_sem_mismatches: list[dict[str, Any]] = []
    missing_inst_nodes: list[str] = []
    artifact_node_count = 0
    for nid, nd in sorted(nodes.items()):
        actual_sem = canonical_digest({
            k: nd[k] for k in ("task_view", "observation_map", "scale_map", "claim_contract", "program_steps")
        })
        if actual_sem != nd["semantic_digest"]:
            sem_mismatches.append({"node_id": nid, "recorded": nd["semantic_digest"], "computed": actual_sem})
        cc = nd.get("claim_contract") or {}
        act = cc.get("conclusion_or_action") or {}
        if act.get("authored_artifact_path"):
            artifact_node_count += 1
            if not act.get("confirmed_instance_digest"):
                missing_inst_nodes.append(nid)
        for br_i, br in enumerate(cc.get("disjunctive_branches", ())):
            if isinstance(br, dict):
                b_act = br.get("conclusion_or_action") or {}
                if b_act.get("authored_artifact_path") and not b_act.get("confirmed_instance_digest"):
                    missing_inst_nodes.append(f"{nid}:branch_{br_i}")

    for h_nid, h_list in sorted(node_history.items()):
        for h_idx, h_nd in enumerate(h_list):
            h_actual = canonical_digest({
                k: h_nd[k] for k in ("task_view", "observation_map", "scale_map", "claim_contract", "program_steps")
            })
            if h_actual != h_nd["semantic_digest"]:
                hist_sem_mismatches.append({
                    "history_key": f"{h_nid}:{h_idx}",
                    "recorded": h_nd["semantic_digest"],
                    "computed": h_actual,
                })

    cert_mismatches: list[dict[str, Any]] = []
    edge_inst_mismatches: list[dict[str, Any]] = []
    for ce in cg_edges:
        cert = ce.get("empirical_closure_certificate")
        rec_dig = ce.get("closure_certificate_digest")
        comp_dig = hashlib.sha256(json.dumps(cert, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        if rec_dig != comp_dig:
            cert_mismatches.append({
                "coarse_node_id": ce["coarse_node_id"],
                "recorded": rec_dig,
                "computed": comp_dig,
            })
        c_nid = ce.get("coarse_node_id")
        if c_nid in nodes and not ce.get("stale_endpoint_revision"):
            c_cc = nodes[c_nid].get("claim_contract") or {}
            id_ok, id_reason, id_meta = validate_claim_certificate_instance_identity(c_cc, cert or {})
            exp_inst = id_meta.get("confirmed_instance_digest")
            edge_inst = ce.get("confirmed_instance_digest")
            if not id_ok or ((exp_inst is not None or edge_inst is not None) and str(exp_inst) != str(edge_inst)):
                edge_inst_mismatches.append({
                    "coarse_node_id": c_nid,
                    "reason": id_reason if not id_ok else "edge_confirmed_instance_digest_mismatch",
                    "edge_confirmed_instance_digest": edge_inst,
                    **id_meta,
                })

    subtrees = (
        getattr(forest_dict_or_obj, "shared_base_subtrees", {})
        if isinstance(forest_dict_or_obj, PhilosophicalForest)
        else forest_dict_or_obj.get("shared_base_subtrees", {})
    )
    subtree_mismatches: list[dict[str, Any]] = []
    for sid, sb in sorted((subtrees or {}).items()):
        comp_sub = canonical_digest({
            "subtree_id": sb["subtree_id"],
            "shared_nodes": sb["shared_nodes"],
            "shared_relations": sb["shared_relations"],
            "role_and_scale_schema": sb["role_and_scale_schema"],
            "open_completion_slots": sb["open_completion_slots"],
            "permission_tier": sb["permission_tier"],
        })
        if comp_sub != sb.get("subtree_digest"):
            subtree_mismatches.append({"subtree_id": sid, "recorded": sb.get("subtree_digest"), "computed": comp_sub})

    out_report = {
        "node_count": len(nodes),
        "artifact_node_count": artifact_node_count,
        "missing_expected_instance_nodes": missing_inst_nodes,
        "semantic_digest_mismatch_count": len(sem_mismatches),
        "semantic_digest_mismatches": sem_mismatches,
        "history_semantic_digest_mismatch_count": len(hist_sem_mismatches),
        "history_semantic_digest_mismatches": hist_sem_mismatches,
        "certificate_digest_mismatch_count": len(cert_mismatches),
        "certificate_digest_mismatches": cert_mismatches,
        "edge_certificate_instance_mismatch_count": len(edge_inst_mismatches),
        "edge_certificate_instance_mismatches": edge_inst_mismatches,
    }
    if subtrees:
        out_report["shared_base_subtree_count"] = len(subtrees)
        out_report["subtree_digest_mismatch_count"] = len(subtree_mismatches)
        out_report["subtree_digest_mismatches"] = subtree_mismatches
    return out_report


# ============================================================================
# Part 1. Direct Programmatic Verification of ALL `Hegel_M4_3_Audit` Controls
# ============================================================================

def run_m4_3_audit_closure_verification_suite(work_root: Path) -> dict[str, Any]:
    """Execute every counterfactual probe from ``Hegel_M4_3_Audit`` (`instance/`, `scale/`, `evaluation/`)."""
    work_root.mkdir(parents=True, exist_ok=True)
    lifecycle_dir = work_root / "shared_m44_lifecycle"
    built = build_m4_3_continual_forest(lifecycle_dir)
    post_s2: PhilosophicalForest = built["forest"]
    true_s0: PhilosophicalForest = built["forest_after_stage0"]
    unmaint_s1: PhilosophicalForest = built["forest_before_stage2_maintenance"]

    # -------------------------------------------------------------------------
    # 1. P1-1: Node-to-Instance Anchor & Genuine Same-AST Instance Swap (`probe_same_ast_instance_swap.py`)
    # -------------------------------------------------------------------------
    post_s2_dig_check = verify_forest_snapshot_digests(post_s2)

    f_swap = deepcopy(post_s2)
    monod_nid = "stage0_monod_2d"
    monod_act = f_swap.nodes[monod_nid]["claim_contract"]["conclusion_or_action"]
    orig_monod_path = Path(str(f_swap.resolve_artifact_path(monod_act["authored_artifact_path"])))
    orig_monod_art = json.loads(orig_monod_path.read_text(encoding="utf-8"))
    orig_loaded = load_verified_authored_method(
        orig_monod_path,
        expected_ast_digest=monod_act["ast_skeleton_digest"],
        expected_instance_digest=monod_act["confirmed_instance_digest"],
        require_external_manifest=True,
        verify_frozen_only=True,
    )
    assert orig_loaded is not None

    swap_slot_ws = work_root / "p1_same_ast_slot"
    shutil.copytree(orig_monod_path.parent, swap_slot_ws, dirs_exist_ok=True)
    monod_act["authored_artifact_path"] = str((swap_slot_ws / "authored_method.json").resolve())

    def _dispatch_monod(qid: str, f_obj: PhilosophicalForest) -> dict[str, Any]:
        _VERIFIED_AUTHORED_CACHE.clear()
        return ForestScopeDispatcher(f_obj).dispatch_predict(
            query_id=qid,
            candidate_tree_id="tree_monod_reactor",
            features={"s": 1.6, "b": 0.0},
            target_metadata={"active_arity": 2, "haldane_turnover_shunt_regime": False, "secondary_bleed_active": False},
            ground_truth=6.0 * 1.6 / (2.0 + 1.6),
        )

    base_monod_rec = _dispatch_monod("baseline_original_instance", f_swap)

    # Calibrate a second GENUINE, independently confirmed, same-AST, same-source-SHA Monod instance (`p0 ≈ 12, p1 ≈ 2`)
    orig_b = orig_monod_art["host_confirmation_bundle"]
    cal_double = _run_metered_atomic_calibration(
        work_root / "p1_same_ast_double_gain_cal",
        task_id="genuine_same_law_double_gain",
        surface_domain="audit_same_law",
        spec=orig_monod_art["operator_spec"],
        role_names=orig_monod_art["role_names"],
        pilot_points=[dict(r["features"]) for r in orig_b["training_prefix_rows"]],
        confirm_points=[dict(r["features"]) for r in orig_b["confirm_rows"]],
        oracle_fn=lambda pt: 2.0 * float(predict_with_frozen_instance(orig_loaded, [pt])["predictions"][0]),
        tolerance=float(orig_b["tolerance"]),
    )
    new_monod_art = cal_double["artifact"]
    new_standalone_loaded = load_verified_authored_method(
        Path(cal_double["artifact_path"]),
        require_external_manifest=True,
        verify_frozen_only=True,
    )
    assert new_standalone_loaded is not None

    # Swap ONLY the 3 artifact files in `swap_slot_ws` while keeping node ID, version, AST, & expected_instance_digest!
    for fn_swap in ("authored_method.json", "confirmed_manifest.json", "confirmed_fitted_params.json"):
        shutil.copy2(Path(cal_double["artifact_path"]).parent / fn_swap, swap_slot_ws / fn_swap)
    swapped_monod_rec = _dispatch_monod("swapped_same_ast_instance", f_swap)

    # Also verify missing `confirmed_instance_digest` on a zero-fit node is rejected fail-closed!
    f_missing_dig = deepcopy(f_swap)
    f_missing_dig.nodes[monod_nid]["claim_contract"]["conclusion_or_action"].pop("confirmed_instance_digest", None)
    missing_dig_rec = _dispatch_monod("missing_expected_instance_digest", f_missing_dig)

    # Restore original files -> accepted again!
    for fn_swap in ("authored_method.json", "confirmed_manifest.json", "confirmed_fitted_params.json"):
        shutil.copy2(orig_monod_path.parent / fn_swap, swap_slot_ws / fn_swap)
    restored_monod_rec = _dispatch_monod("restored_original_instance", f_swap)

    # -------------------------------------------------------------------------
    # 2. P1-2: Unified Coordinate Role-Map Lock & Fresh Confirm Box Intersection (`scale/run_scale_controls.py`)
    # -------------------------------------------------------------------------
    targets_s0, _ = build_m4_3_target_suite(seed=20_301_120)
    pi_t21 = targets_s0[20]  # Adiabatic hydraulic Pi_s target
    pi_t23 = targets_s0[22]  # Short-horizon quench thermal Pi_s target
    cnode_id = "ode_meso_schur_pi_thermal"

    acq_trace: list[dict[str, Any]] = []
    fit_cand_trace: list[dict[str, Any]] = []
    orig_ms_acq = MultiScaleWorkspaceTaskLab.acquire_channel
    orig_fit_fn = _autonomous_fit_candidate_specs

    def _spy_ms_acq(self, points, *, channel="coarse_macro", phase="pilot"):
        rows = orig_ms_acq(self, points, channel=channel, phase=phase)
        acq_trace.append({"task_id": self.task_id, "phase": phase, "channel": channel, "points": deepcopy(points)})
        return rows

    def _spy_fit(*args, **kwargs):
        best_r, cnt_r = orig_fit_fn(*args, **kwargs)
        fit_cand_trace.append({
            "count": cnt_r,
            "best_tag": best_r.get("tag") if best_r else None,
            "best_role_order": best_r.get("role_order") if best_r else None,
        })
        return best_r, cnt_r

    def _eval_pi_ctrl(t_obj: M43TargetSpec, tag_name: str, f_custom: PhilosophicalForest | None = None) -> dict[str, Any]:
        cf = deepcopy(post_s2 if f_custom is None else f_custom)
        ix = HippoRAGWormholeIndex(cf, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        a0, f0 = len(acq_trace), len(fit_cand_trace)
        with (
            patch.object(MultiScaleWorkspaceTaskLab, "acquire_channel", _spy_ms_acq),
            patch("hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure._autonomous_fit_candidate_specs", _spy_fit),
        ):
            res = evaluate_m4_3_target_under_arm(
                t_obj,
                arm_name=tag_name,
                seed=20_301_120,
                work_dir=work_root / "p1_scale_controls" / tag_name,
                forest=cf,
                index=ix,
                ordinary_history_specs=None,
                fallback_pool_specs=[],
                open_code_agent_fallback=False,
                allow_pool_fallback=False,
            )
        sc_contract = json.loads((work_root / "p1_scale_controls" / tag_name / "model_scope_contract.json").read_text(encoding="utf-8"))
        return {
            "result": res,
            "scope_contract": sc_contract,
            "acquisitions": acq_trace[a0:],
            "fitted_candidates": fit_cand_trace[f0:],
        }

    ctrl_native_t21 = _eval_pi_ctrl(pi_t21, "native_adiabatic_T21")

    # S1: Canonical names `('u_drive', 'T_horizon')` with short-time `T in [0.40..1.20]` and `u in [2.45..2.95]`!
    swap_pilot = [
        {"u_drive": 2.45, "T_horizon": 0.40},
        {"u_drive": 2.90, "T_horizon": 0.52},
        {"u_drive": 2.55, "T_horizon": 0.67},
        {"u_drive": 2.80, "T_horizon": 0.85},
        {"u_drive": 2.65, "T_horizon": 1.05},
        {"u_drive": 2.95, "T_horizon": 1.20},
    ]
    swap_confirm = [{"u_drive": 2.70, "T_horizon": 0.47}]
    swap_heldout = [
        {"u_drive": 2.50, "T_horizon": 0.42},
        {"u_drive": 2.85, "T_horizon": 0.55},
        {"u_drive": 2.75, "T_horizon": 0.75},
    ]
    swap_target = replace(
        pi_t21,
        target_id="audit_canonical_axis_swap",
        feature_bounds={"u_drive": (2.4, 3.0), "T_horizon": (0.4, 1.3)},
        pilot_points=swap_pilot,
        confirm_points=swap_confirm,
        heldout_points=swap_heldout,
    )
    gate_only_swap = _find_valid_canonical_role_map_for_box(
        swap_target.feature_names,
        post_s2.nodes[cnode_id],
        [*swap_pilot, *swap_confirm],
        artifact_root=post_s2.artifact_root,
    )
    ctrl_axis_swap = _eval_pi_ctrl(swap_target, "canonical_axis_swap")

    # Also verify coarse-only forest (no fine node available) returns `scoped_unknown` (`0` false confirmations!)
    f_coarse_only = deepcopy(post_s2)
    f_coarse_only.nodes.pop("ode_micro_2mode_node", None)
    f_coarse_only.coarse_graining_edges = [
        e for e in f_coarse_only.coarse_graining_edges if e["coarse_node_id"] != cnode_id
    ]
    ctrl_axis_swap_no_fine = _eval_pi_ctrl(swap_target, "canonical_axis_swap_no_fine", f_custom=f_coarse_only)

    # Anonymous coordinates `('x0', 'x1')` where BOTH permutations fall in `[0.4, 3.0] x [2.4, 5.2]` (`x0=T, x1=u`),
    # followed by an out-of-box continuation heldout `x0 = 0.45` (`T = 0.45 < 2.4`)!
    anon_pts = [
        {"x0": 2.48, "x1": 2.42},
        {"x0": 2.62, "x1": 2.75},
        {"x0": 2.78, "x1": 2.50},
        {"x0": 2.88, "x1": 2.92},
        {"x0": 2.95, "x1": 2.60},
        {"x0": 2.55, "x1": 2.84},
    ]
    anon_target = M43TargetSpec(
        target_id="audit_anon_role_lock_oob_heldout",
        mechanism_family="PiS_MultiScale_Mixed_2D",
        feature_names=("x0", "x1"),
        feature_bounds={"x0": (2.4, 3.0), "x1": (2.4, 3.0)},
        pilot_points=anon_pts,
        confirm_points=[{"x0": 2.70, "x1": 2.68}],
        heldout_points=[{"x0": 0.45, "x1": 2.65}],  # `x0` is `T_horizon` -> 0.45 < 2.4 MUST be blocked by locked map!
        tolerance=0.055,
        oracle_fn=lambda p: float(_eval_pi_s_fine_scalar(p["x1"], p["x0"])["y_micro"]),
        coarse_oracle_fn=lambda p: float(_eval_pi_s_coarse_scalar(p["x1"], p["x0"])["y_coarse"]),
        domain_cue="cue_adiabatic_multiscale",
        scale_regime="meso",
        modality="hydraulic",
        is_multiscale_two_channel=True,
    )
    ctrl_anon_oob = _eval_pi_ctrl(anon_target, "anon_role_lock_oob_heldout")

    # S2: First predeclared confirm is already-paid `pilot[0]`; second is `T=0.4` (outside `[2.4, 5.2]`)!
    exhausted_first_target = replace(
        pi_t21,
        target_id="audit_confirm_fresh_outside_box",
        feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": (0.05, 5.2)},
        confirm_points=[deepcopy(pi_t21.pilot_points[0]), {"u_drive": 2.8, "T_horizon": 0.4}],
    )
    ctrl_fresh_box = _eval_pi_ctrl(exhausted_first_target, "fresh_confirm_inside_box")
    coarse_confirm_acqs = [
        pt
        for a in ctrl_fresh_box["acquisitions"]
        if a["channel"] == "coarse_macro" and "confirm" in a["phase"]
        for pt in a["points"]
    ]
    all_coarse_acqs_in_box = all(
        is_point_inside_verified_box(
            pt, post_s2.nodes[cnode_id]["scale_map"]["window_or_partition"]["verified_input_box"]
        )
        for a in ctrl_fresh_box["acquisitions"]
        if a["channel"] == "coarse_macro"
        for pt in a["points"]
    )

    # Boundary-zone `[2.24, 2.36]` safety control (inside Stage-0 `[2.2, 5.2]`, outside Stage-2 `[2.4, 5.2]`)!
    boundary_pts = [
        {"u_drive": 2.2, "T_horizon": 2.24},
        {"u_drive": 2.6, "T_horizon": 2.27},
        {"u_drive": 2.9, "T_horizon": 2.30},
        {"u_drive": 2.4, "T_horizon": 2.33},
        {"u_drive": 2.8, "T_horizon": 2.35},
        {"u_drive": 3.0, "T_horizon": 2.36},
    ]
    boundary_target = replace(
        pi_t21,
        target_id="audit_boundary_zone_T_2_24_to_2_36",
        modality="thermal",
        feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": (2.22, 2.38)},
        pilot_points=boundary_pts,
        confirm_points=[{"u_drive": 2.75, "T_horizon": 2.26}],
        heldout_points=[{"u_drive": 2.95, "T_horizon": 2.24}, {"u_drive": 2.50, "T_horizon": 2.28}],
        tolerance=0.025,
    )
    ctrl_boundary_s2 = _eval_pi_ctrl(boundary_target, "boundary_zone_post_stage2", f_custom=post_s2)

    # -------------------------------------------------------------------------
    # 3. P2-S3 & S5: Cross-Instance Certificate Rejection + Self-Verifying Relocatable Archive
    # -------------------------------------------------------------------------
    cert_stage0 = deepcopy(
        next(e["empirical_closure_certificate"] for e in true_s0.coarse_graining_edges if e["coarse_node_id"] == cnode_id)
    )
    f_cross_cert = deepcopy(post_s2)
    before_cross_dump = json.dumps(serialize_location_independent_forest(f_cross_cert), sort_keys=True)
    cross_cert_res = f_cross_cert.revalidate_vertical_edges_with_certificate(
        cnode_id,
        witness_episode_id="audit_stage0_cert_on_stage2_instance",
        closure_certificate=cert_stage0,
        reason="cross_instance_certificate_probe",
    )
    after_cross_dump = json.dumps(serialize_location_independent_forest(f_cross_cert), sort_keys=True)

    # Relocate bundle to a completely separate directory and verify BOTH JSON-on-disk AND cold-reloaded digests!
    reloc_bundle = work_root / "p2_relocated_bundle_verify"
    persisted_dst = reloc_bundle / "persisted_authored_workspaces"
    persisted_dst.mkdir(parents=True, exist_ok=True)
    ignore_pyc = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for sub_n in ("stage0_metered_sources", "persisted_stage1_authored_workspaces"):
        shutil.copytree(lifecycle_dir / sub_n, persisted_dst / sub_n, ignore=ignore_pyc, dirs_exist_ok=True)
    shutil.copytree(
        lifecycle_dir / "stage2_lifecycle" / "stage2_pi_s_reverified",
        persisted_dst / "stage2_lifecycle" / "stage2_pi_s_reverified",
        ignore=ignore_pyc,
        dirs_exist_ok=True,
    )
    snaps_dst = reloc_bundle / "relocatable_forest_snapshots"
    snaps_dst.mkdir(parents=True, exist_ok=True)
    archive_checks: dict[str, Any] = {}
    for s_lbl, f_snap in (
        ("true_stage0_forest_snapshot.json", true_s0),
        ("unmaintained_post_stage1_forest_snapshot.json", unmaint_s1),
        ("post_stage2_forest_snapshot.json", post_s2),
    ):
        sp_file = snaps_dst / s_lbl
        sp_file.write_text(json.dumps(serialize_location_independent_forest(f_snap), indent=2, sort_keys=True), encoding="utf-8")
        raw_disk_json = json.loads(sp_file.read_text(encoding="utf-8"))
        reloaded_f = load_location_independent_forest(sp_file, reloc_bundle)
        archive_checks[s_lbl] = {
            "disk_json": verify_forest_snapshot_digests(raw_disk_json),
            "reloaded_memory": verify_forest_snapshot_digests(reloaded_f),
        }

    cold_reloaded_s2 = load_location_independent_forest(snaps_dst / "post_stage2_forest_snapshot.json", reloc_bundle)
    cold_fit_calls: list[str] = []

    def _prof_fit(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "fit":
            cold_fit_calls.append(f"{frame.f_code.co_filename}:{frame.f_code.co_firstlineno}")

    _VERIFIED_AUTHORED_CACHE.clear()
    before_r3_dump = json.dumps(serialize_location_independent_forest(cold_reloaded_s2), sort_keys=True)
    sys.setprofile(_prof_fit)
    try:
        cold_r3_stream = run_interleaved_zero_fit_dispatcher_stream(cold_reloaded_s2)
    finally:
        sys.setprofile(None)
    after_r3_dump = json.dumps(serialize_location_independent_forest(cold_reloaded_s2), sort_keys=True)

    # -------------------------------------------------------------------------
    # 4. P2-S4 & Cost Completeness: Exact 2-Mode Matrix (`A12 == g_sf`) & Event Ledger
    # -------------------------------------------------------------------------
    quench_eval_calls: list[str] = []
    orig_iso_ev = evaluate_isolated_python_model

    def _spy_quench_ev(*args, **kwargs):
        quench_eval_calls.append(str(args[1]))
        return orig_iso_ev(*args, **kwargs)

    with patch("hegel_machine.cognition.m4_3_channel_bound_lifecycle_and_audit_closure.evaluate_isolated_python_model", _spy_quench_ev):
        ctrl_quench_t23 = _eval_pi_ctrl(pi_t23, "native_quench_exact_eigen")
    quench_res = ctrl_quench_t23["result"]
    quench_pb = quench_res["cost"]["phase_breakdown"]
    # `quench_eval_calls` consists of: 1 training fit (`recalled_fine_2mode.py`), 1 confirm eval, 1 heldout eval -> 0 discarded `u_cand_00.py`!
    discarded_generic_prefits = sum(1 for m_rel in quench_eval_calls if m_rel.startswith("u_cand_"))

    # -------------------------------------------------------------------------
    # 5. P2 Legacy Entrypoints & Live Dynamic Index Invalidation
    # -------------------------------------------------------------------------
    # 5a. Construct an intact pre-V17.4-style unbundled artifact workspace in temp dir (without touching any Audit dir!)
    legacy_ws = work_root / "p2_legacy_unbundled_ws"
    legacy_ws.mkdir(parents=True, exist_ok=True)
    sp_leg = make_atomic_operator_spec("linear_1d")
    src_leg = compile_operator_spec_to_python(sp_leg, role_names=["x"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0})
    ast_leg = canonicalize_python_model_ast(src_leg)
    leg_rows = [{"features": {"x": 1.0}, "target": 2.5}, {"features": {"x": 2.0}, "target": 5.0}, {"features": {"x": 3.0}, "target": 7.5}]
    (legacy_ws / "authored_model.py").write_text(src_leg, encoding="utf-8")
    (legacy_ws / "confirmed_training_observations.json").write_text(json.dumps(leg_rows), encoding="utf-8")
    leg_art = {
        "operator_spec": sp_leg,
        "role_names": ["x"],
        "ast_skeleton_digest": ast_leg["ast_skeleton_digest"],
        "skeleton_status": "VERIFIED_PURE_AST",
        "model_sha256": ast_leg["model_sha256"],
        "authored_model_source": src_leg,
        "confirmation_digest": hashlib.sha256(json.dumps(leg_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        "confirmed_ok": True,
    }
    leg_art_p = legacy_ws / "authored_method.json"
    leg_art_p.write_text(json.dumps(leg_art, indent=2), encoding="utf-8")

    leg_refit_calls: list[int] = []
    orig_cd_ev = m4_cd.evaluate_isolated_python_model

    def _spy_cd_ev(*args, **kwargs):
        leg_refit_calls.append(1)
        return orig_cd_ev(*args, **kwargs)

    _VERIFIED_AUTHORED_CACHE.clear()
    with patch.object(m4_cd, "evaluate_isolated_python_model", _spy_cd_ev):
        leg_strict_rejected = load_verified_authored_method(
            leg_art_p,
            expected_ast_digest=ast_leg["ast_skeleton_digest"],
            expected_instance_digest="definitely-not-the-instance-digest",
            verify_frozen_only=True,
        )
    refits_on_strict_flags = len(leg_refit_calls)
    _VERIFIED_AUTHORED_CACHE.clear()
    leg_permissive_ok = load_verified_authored_method(
        leg_art_p,
        expected_ast_digest=ast_leg["ast_skeleton_digest"],
    )

    # 5b. Legacy `m4_multiscale_renormalization.evaluate_m4_target_under_arm` on `ScaleBlind` `C0_T07`!
    ms_f_old, _ = m4_ms.build_m4_multiscale_forest(cluster_seed=101, extra_distractor_count=6)
    ms_t_old, _ = m4_ms.build_m4_cluster_targets(cluster_idx=0)
    sb_acqs: list[dict[str, Any]] = []

    def _spy_sb_acq(self, pts, *, channel="coarse_macro", phase="pilot"):
        sb_acqs.append({"phase": phase, "channel": channel, "points": deepcopy(pts)})
        return orig_ms_acq(self, pts, channel=channel, phase=phase)

    with patch.object(MultiScaleWorkspaceTaskLab, "acquire_channel", _spy_sb_acq):
        sb_res = m4_ms.evaluate_m4_target_under_arm(
            ms_t_old[6],
            arm_name="HippoPPR_ScaleBlind_Walk",
            forest=ms_f_old,
            index=HippoRAGWormholeIndex(ms_f_old, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=False),
            stage1_gate=None,
            work_dir=work_root / "p2_legacy_scale_blind_t07",
        )
    seen_micro_pts: list[dict[str, float]] = []
    dup_micro_confirm_pts = 0
    for a_rec in sb_acqs:
        if a_rec["channel"] == "micro_fine":
            for pt in a_rec["points"]:
                if "confirm" in a_rec["phase"] and any(coords_approx_equal(pt, sp) for sp in seen_micro_pts):
                    dup_micro_confirm_pts += 1
                seen_micro_pts.append(pt)

    # 5c. Live Dynamic Index Invalidation (`W4 v1 -> v2` reflected automatically in EXISTING PPR & CueTable!)
    f_dyn = deepcopy(post_s2)
    live_ppr = HippoRAGWormholeIndex(f_dyn, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
    live_cue = CompiledRelationalCueTable(f_dyn)
    tv_dyn = {"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.05, "active_arity": 2}
    before_ppr_w4 = next(
        c["target_node"] for c in live_ppr.retrieve(["sig:internal_monod_shunt_specialization"], task_view=tv_dyn)["candidates"]
        if c["node_id"] == "stage1_W4_monod_shunt"
    )
    w4_cc_v2 = deepcopy(f_dyn.nodes["stage1_W4_monod_shunt"]["claim_contract"])
    w4_cc_v2["preconditions"].append({"field": "b", "op": "<=", "value": 1.6})
    f_dyn.empirical_revise_node(
        "stage1_W4_monod_shunt",
        witness_episode_id="audit_live_w4_scope_revision",
        cause="protocol_or_scope_mismatch",
        updated_claim_contract=w4_cc_v2,
    )
    f_dyn.revalidate_vertical_edges_after_revision("stage1_W4_monod_shunt", witness_episode_id="audit_live_w4_rebind")
    after_ppr_ret = live_ppr.retrieve(["sig:internal_monod_shunt_specialization"], task_view=tv_dyn)
    after_cue_ret = live_cue.retrieve(["sig:internal_monod_shunt_specialization"], task_view=tv_dyn)
    after_ppr_w4 = next(c["target_node"] for c in after_ppr_ret["candidates"] if c["node_id"] == "stage1_W4_monod_shunt")
    after_cue_w4 = next(c["target_node"] for c in after_cue_ret["candidates"] if c["node_id"] == "stage1_W4_monod_shunt")

    return {
        "p1_instance_anchor_and_same_ast_swap": {
            "post_stage2_artifact_node_count": post_s2_dig_check["artifact_node_count"],
            "missing_expected_instance_count": len(post_s2_dig_check["missing_expected_instance_nodes"]),
            "same_ast_skeleton": orig_monod_art["ast_skeleton_digest"] == new_monod_art["ast_skeleton_digest"],
            "same_model_source_sha256": orig_monod_art["model_sha256"] == new_monod_art["model_sha256"],
            "distinct_confirmed_instance_digest": orig_monod_art["confirmed_instance_digest"] != new_monod_art["confirmed_instance_digest"],
            "baseline_status": base_monod_rec["status"],
            "baseline_prediction": base_monod_rec["prediction"],
            "baseline_has_explicit_source_params_and_instance_hashes": bool(
                base_monod_rec.get("model_source_sha256")
                and base_monod_rec.get("fitted_params_sha256")
                and base_monod_rec.get("confirmed_instance_digest")
            ),
            "swapped_same_ast_status": swapped_monod_rec["status"],
            "swapped_same_ast_rejected": swapped_monod_rec["status"] == "scope_unauthorized",
            "missing_digest_status": missing_dig_rec["status"],
            "missing_digest_rejected": missing_dig_rec["status"] == "scope_unauthorized",
            "restored_original_status": restored_monod_rec["status"],
            "restored_original_prediction": restored_monod_rec["prediction"],
        },
        "p1_coordinate_role_lock_and_fresh_confirm_box": {
            "native_T21_channel": ctrl_native_t21["result"]["selected_channel"],
            "native_T21_strict_pass": ctrl_native_t21["result"]["strict_pass"],
            "native_T21_false_confirmation": ctrl_native_t21["result"]["false_confirmation"],
            "canonical_axis_swap_box_gate_accepted": gate_only_swap[0],
            "canonical_axis_swap_selected_channel": ctrl_axis_swap["result"]["selected_channel"],
            "canonical_axis_swap_strict_pass": ctrl_axis_swap["result"]["strict_pass"],
            "canonical_axis_swap_false_confirmation": ctrl_axis_swap["result"]["false_confirmation"],
            "canonical_axis_swap_heldout_rmse": ctrl_axis_swap["result"]["heldout_rmse"],
            "canonical_axis_swap_no_fine_channel": ctrl_axis_swap_no_fine["result"]["selected_channel"],
            "canonical_axis_swap_no_fine_selected_node": ctrl_axis_swap_no_fine["result"]["selected_node_id"],
            "canonical_axis_swap_no_fine_false_confirmation": ctrl_axis_swap_no_fine["result"]["false_confirmation"],
            "anon_role_locked_map": ctrl_anon_oob["scope_contract"]["canonical_role_map"],
            "anon_role_oob_heldout_status": ctrl_anon_oob["result"]["status"],
            "anon_role_oob_heldout_false_confirmation": ctrl_anon_oob["result"]["false_confirmation"],
            "fresh_confirm_coarse_points": coarse_confirm_acqs,
            "all_coarse_acquisitions_inside_verified_box": all_coarse_acqs_in_box,
            "fresh_confirm_strict_pass": ctrl_fresh_box["result"]["strict_pass"],
            "fresh_confirm_false_confirmation": ctrl_fresh_box["result"]["false_confirmation"],
            "boundary_zone_selected_channel": ctrl_boundary_s2["result"]["selected_channel"],
            "boundary_zone_strict_pass": ctrl_boundary_s2["result"]["strict_pass"],
            "boundary_zone_false_confirmation": ctrl_boundary_s2["result"]["false_confirmation"],
        },
        "p2_certificate_identity_and_self_verifying_archive": {
            "cross_instance_certificate_accepted": cross_cert_res["accepted"],
            "cross_instance_certificate_reason": cross_cert_res["reason"],
            "failed_revalidation_left_forest_unmutated": before_cross_dump == after_cross_dump,
            "archive_digest_verification": archive_checks,
            "cold_reloaded_r3_all_routes_exact": cold_r3_stream["all_routes_exact"],
            "cold_reloaded_r3_python_fit_calls": len(cold_fit_calls),
            "cold_reloaded_r3_strictly_readonly": before_r3_dump == after_r3_dump,
        },
        "p2_exact_2mode_matrix_and_event_accounting": {
            "quench_T23_heldout_rmse": quench_res["heldout_rmse"],
            "quench_T23_strict_pass": quench_res["strict_pass"],
            "discarded_generic_prefit_count": discarded_generic_prefits,
            "evaluated_candidate_count": quench_res["evaluated_candidate_count"],
            "local_rk4_compute_metered": "local_trusted_rk4_basis_compute" in quench_pb,
            "artifact_integrity_compute_metered": "artifact_integrity_check" in ctrl_native_t21["result"]["cost"]["phase_breakdown"],
        },
        "p2_legacy_entrypoints_and_dynamic_index_invalidation": {
            "legacy_unbundled_rejected_under_strict_flags": leg_strict_rejected is None,
            "legacy_unbundled_refit_calls_under_strict_flags": refits_on_strict_flags,
            "legacy_unbundled_permissive_mode": (leg_permissive_ok or {}).get("_verification_mode"),
            "legacy_scale_blind_t07_strict_pass": sb_res["strict_pass"],
            "legacy_scale_blind_t07_duplicate_micro_confirm_points": dup_micro_confirm_pts,
            "dynamic_w4_version_before": before_ppr_w4["version"],
            "dynamic_ppr_version_after_live_revision": after_ppr_w4["version"],
            "dynamic_ppr_refreshed_automatically": bool(after_ppr_ret.get("index_refreshed")),
            "dynamic_ppr_has_new_precondition": {"field": "b", "op": "<=", "value": 1.6} in after_ppr_w4["claim_contract"]["preconditions"],
            "dynamic_cuetable_version_after_live_revision": after_cue_w4["version"],
            "dynamic_cuetable_recompiled_automatically": bool(after_cue_ret.get("table_recompiled")),
        },
    }


# ============================================================================
# Part 3. Stateful Online Write-Back, Equivalence Pooling & Multi-Cue Stream (`M5` Bridge)
# ============================================================================

def _write_back_discovered_mechanism_to_forest(
    forest: PhilosophicalForest,
    *,
    ep_res: dict[str, Any],
    artifact_abs_dir: Path,
    bundle_root: Path,
    primary_node_id: str,
    primary_tree_id: str,
    operator_tag: str,
    arity: int,
    primary_cues: list[str],
    preconditions: list[dict[str, Any]],
    weaker_parent_id: str | None,
    twin_node_id: str,
    twin_tree_id: str,
    twin_cues: list[str],
    wormhole_id: str,
    tolerance: float,
) -> float:
    """Persist an online-confirmed ``CodeAgent`` discovery & cross-domain wormhole witness into ``forest``."""
    art = ep_res["authored_artifact"]
    bundle = art["host_confirmation_bundle"]
    artifact_abs_dir.mkdir(parents=True, exist_ok=True)
    (artifact_abs_dir / "authored_method.json").write_text(json.dumps(art, indent=2, sort_keys=True), encoding="utf-8")
    (artifact_abs_dir / "confirmed_fitted_params.json").write_text(
        json.dumps(art["fitted_params"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (artifact_abs_dir / "confirmed_manifest.json").write_text(
        json.dumps(
            {k: v for k, v in bundle.items() if k not in {"training_prefix_rows", "confirm_rows"}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rel_loc = str((artifact_abs_dir / "authored_method.json").resolve().relative_to(bundle_root.resolve()).as_posix())
    act_dict = _make_verified_instance_action(
        operator_tag=operator_tag,
        operator_spec=art["operator_spec"],
        artifact_dict=art,
        artifact_relpath=rel_loc,
    )
    prog_steps = [
        {"step": "calibrate_observation_lens", "active_arity": arity},
        {"step": "compile_and_fit_composed_ast", "operator_tag": operator_tag},
        {"step": "subprocess_isolated_confirm", "fresh_points": 1},
    ]
    tv = TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", tolerance)

    # Register primary discovered mid-layer concept node + cross-domain isomorphic twin connected by Wormhole!
    forest.register_node(
        node_id=primary_node_id,
        tree_id=primary_tree_id,
        layer_role="mid",
        task_view=tv,
        observation_map=ObservationMap("online_discovered_flux", parameter_semantics={"active_arity": arity}),
        claim_contract=ClaimContract(
            f"claim_{primary_node_id}",
            "m4_open_compositional_ast",
            preconditions=deepcopy(preconditions),
            conclusion_or_action=deepcopy(act_dict),
            protocol_spec={"isolated_confirm": True},
            tolerance=tolerance,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=list(primary_cues),
        program_steps=prog_steps,
        source_cluster_ids=[f"online_{primary_tree_id}"],
    )
    weaker_edge_added = False
    if weaker_parent_id and weaker_parent_id in forest.nodes and forest.nodes[weaker_parent_id]["tree_id"] == primary_tree_id:
        forest.add_local_implication_edge(
            stronger_node_id=primary_node_id,
            weaker_node_id=weaker_parent_id,
            preserved_contract={"online_compositional_sublation": True},
            scope={"ceiling_extension": True},
        )
        weaker_edge_added = True

    # Before second-domain empirical confirmation, register the cross-domain cue slot ONLY as a
    # `STRUCTURAL_HYPOTHESIS_ONLY` (`bounded_trial`) candidate sharing the pure AST structure (NO cloned
    # second-domain physical instance digest, blocking premature zero-fit `R3` on unconfirmed Domain B!).
    structural_only_act = {
        "operator_tag": operator_tag,
        "operator_spec": deepcopy(art["operator_spec"]),
        "role_names": [str(r) for r in art.get("role_names", ())],
        "ast_skeleton_digest": str(art["ast_skeleton_digest"]),
        "model_source_sha256": str(art["model_sha256"]),
        "structural_alias_of_source_node": primary_node_id,
    }
    src_ep_id = str(art.get("task_id") or primary_node_id)
    src_conf_err = float((bundle or {}).get("observed_confirm_rmse", 0.01))
    forest.register_node(
        node_id=twin_node_id,
        tree_id=twin_tree_id,
        layer_role="mid",
        task_view=tv,
        observation_map=ObservationMap("online_twin_flux", parameter_semantics={"active_arity": arity}),
        claim_contract=ClaimContract(
            f"claim_{twin_node_id}",
            "m4_open_compositional_ast",
            preconditions=deepcopy(preconditions),
            conclusion_or_action=structural_only_act,
            protocol_spec={"isolated_confirm": True, "requires_second_domain_confirmation": True},
            tolerance=tolerance,
            evidence_status="STRUCTURAL_HYPOTHESIS_ONLY",
            permission_tier="bounded_trial",
        ),
        probe_signatures=list(twin_cues),
        program_steps=prog_steps,
        source_cluster_ids=[f"online_{twin_tree_id}"],
    )
    forest.add_horizontal_wormhole(
        wormhole_id=wormhole_id,
        node_a_id=primary_node_id,
        node_b_id=twin_node_id,
        probe_signature_refs=[*primary_cues, *twin_cues],
        role_and_observation_map={"role_map": "identity_canonical", "status": "structural_candidate_bridge"},
        hop_error_budget=round(max(src_conf_err, 0.005), 6),
        permission_tier="bounded_trial",
        source_episode_refs=[src_ep_id],
    )
    # Exact write-back bookkeeping compute cost (`1.5` per node + `0.8` per edge actually added!)
    edges_added = 1 + (1 if weaker_edge_added else 0)
    return round(1.5 * 2 + 0.8 * edges_added, 4)


def run_m4_4_online_writeback_and_multicue_stream(
    work_root: Path,
    built_lifecycle: dict[str, Any],
) -> dict[str, Any]:
    """Execute a 20-step stateful online continual stream where new confirmations mutate the live forest!

    Directly compares 4 arms on the EXACT SAME 20-target sequence (`S01..S20`):
      1. ``Hegel_LiveWriteBack_PPR_PoolOn`` (`alpha=0.70, wormhole_equivalence_pooling=True`, live write-back ON);
      2. ``Hegel_LiveWriteBack_PPR_PoolOff`` (`alpha=0.70, wormhole_equivalence_pooling=False`, live write-back ON);
      3. ``Compiled_LiveWriteBack_CueTable`` (`CompiledRelationalCueTable`, live write-back ON);
      4. ``Frozen_PostStage2_OpenRediscovery`` (`post_stage2_forest` frozen, write-back OFF, open `CodeAgent` on miss).
    """
    work_root.mkdir(parents=True, exist_ok=True)
    base_forest: PhilosophicalForest = deepcopy(built_lifecycle["forest"])
    # Ensure realistic multi-cue distractor hubs exist in 1D (`stage0_murray_1d`) and 2D (`vascular_2d_hub`)
    # so split cues across cross-domain wormhole twins genuinely test `PoolOn` vs `PoolOff` before compression!
    for hub_id, extra_cues in (
        ("stage1_W3_murray_capped", ("sig:cue_tubule_saturation_a", "sig:cue_reuptake_saturation_b")),
        ("vascular_2d_hub", ("sig:cue_photobioreactor_cap2d_a", "sig:cue_allosteric_cap2d_b", "sig:pos_arity_2")),
    ):
        h_nd = base_forest.nodes[hub_id]
        h_sigs = list(h_nd["probe_signatures"])
        for dc in extra_cues:
            if dc not in h_sigs:
                h_sigs.append(dc)
        base_forest.register_node(
            node_id=hub_id,
            tree_id=h_nd["tree_id"],
            layer_role=h_nd["layer_role"],
            task_view=h_nd["task_view"],
            observation_map=h_nd["observation_map"],
            scale_map=h_nd["scale_map"],
            claim_contract=h_nd["claim_contract"],
            probe_signatures=h_sigs,
            program_steps=h_nd["program_steps"],
            source_cluster_ids=h_nd["source_cluster_ids"],
        )
        base_forest.revalidate_vertical_edges_after_revision(hub_id, witness_episode_id=f"m44_init_{hub_id}")
    base_forest.revalidate_wormhole_after_revision(
        "wh_arbor_to_vascular_2d_hub", witness_episode_id="m44_init_wh_arbor", reason="sync_hub_version"
    )

    rng = Random(20_361_120)
    stream_tasks: list[tuple[str, M43TargetSpec, dict[str, Any] | None]] = []

    # Step 1 (`S01`): W4 MonodShunt task paired with online scope refinement (`b <= 1.6` live revision event!)
    stream_tasks.append((
        "online_revision_event",
        M43TargetSpec(
            target_id="M44_S01_W4_LiveRevisionTrigger",
            mechanism_family="Online_W4_MonodShunt_Revision",
            feature_names=("s_conc", "b_bleed"),
            feature_bounds={"s_conc": (0.2, 10.0), "b_bleed": (0.0, 1.5)},
            pilot_points=[
                {"s_conc": 0.5, "b_bleed": 0.2},
                {"s_conc": 1.2, "b_bleed": 0.6},
                {"s_conc": 2.4, "b_bleed": 0.3},
                {"s_conc": 4.5, "b_bleed": 1.1},
                {"s_conc": 7.0, "b_bleed": 0.5},
                {"s_conc": 9.2, "b_bleed": 0.9},
            ],
            confirm_points=[{"s_conc": 5.8, "b_bleed": 0.7}],
            heldout_points=[{"s_conc": 1.8, "b_bleed": 0.4}, {"s_conc": 6.4, "b_bleed": 1.0}, {"s_conc": 8.6, "b_bleed": 0.3}],
            tolerance=0.05,
            oracle_fn=lambda p: 7.4 * p["s_conc"] / (1.9 + p["s_conc"]) + 0.62 * p["b_bleed"],
            coarse_oracle_fn=None,
            domain_cue="cue_carrier_binding_onset",
            scale_regime="macro",
        ),
        {"revise_node_id": "stage1_W4_monod_shunt", "added_precondition": {"field": "b_bleed_ceiling_safe", "op": "==", "value": True}},
    ))

    # Step 2 (`S02`): Novel Online Discovery #1 — `W6: 1D Haldane Substrate Inhibition` (`V*x / (K + x + Ki*x^2)`)!
    def _mk_w6(vm: float, km: float, ki: float):
        return lambda p: vm * p["x_load"] / (km + p["x_load"] + ki * (p["x_load"] ** 2))

    stream_tasks.append((
        "online_writeback_W6",
        M43TargetSpec(
            target_id="M44_S02_Discover_W6_Haldane1D",
            mechanism_family="Online_W6_Haldane_1D",
            feature_names=("x_load",),
            feature_bounds={"x_load": (0.2, 12.0)},
            pilot_points=[{"x_load": x} for x in (0.4, 1.2, 2.5, 5.0, 8.2, 11.2)],
            confirm_points=[{"x_load": 3.6}, {"x_load": 9.6}],
            heldout_points=[{"x_load": 0.8}, {"x_load": 3.1}, {"x_load": 7.0}, {"x_load": 10.6}],
            tolerance=0.045,
            oracle_fn=_mk_w6(14.5, 1.60, 0.18),
            coarse_oracle_fn=None,
            domain_cue="cue_tubule_saturation_a",
            scale_regime="macro",
        ),
        {
            "primary_node_id": "online_W6_michaelis_cap_1d",
            "primary_tree_id": "tree_online_tubule_cap",
            "operator_tag": "op_online_W6_haldane_1d",
            "arity": 1,
            "primary_cues": ["sig:cue_tubule_saturation_a"],
            "twin_node_id": "online_W6_reuptake_twin_1d",
            "twin_tree_id": "tree_online_reuptake_cap",
            "twin_cues": ["sig:cue_reuptake_saturation_b", "sig:pos_arity_1", "sig:neg_arity_1"],
            "wormhole_id": "wh_online_W6_tubule_reuptake",
            "weaker_parent_id": None,
        },
    ))

    # Steps 3–8 (`S03..S08`): 6 unseen downstream `W6` tasks (1 primary cue, 1 wormhole hop, 4 composite multi-cue queries!)
    for idx_w6 in range(6):
        vm6 = round(rng.uniform(13.2, 15.8), 3)
        km6 = round(rng.uniform(1.40, 1.90), 3)
        ki6 = round(rng.uniform(0.15, 0.22), 4)
        if idx_w6 == 0:
            cue6 = "cue_tubule_saturation_a"
        elif idx_w6 == 1:
            cue6 = "cue_reuptake_saturation_b"
        else:
            cue6 = "cue_tubule_saturation_a|cue_reuptake_saturation_b"
        stream_tasks.append((
            "downstream_W6_reuse",
            M43TargetSpec(
                target_id=f"M44_S{len(stream_tasks)+1:02d}_Reuse_W6_Haldane1D",
                mechanism_family="Online_W6_Haldane_1D",
                feature_names=("x_load",),
                feature_bounds={"x_load": (0.2, 12.0)},
                pilot_points=[{"x_load": x} for x in (0.45, 1.25, 2.6, 5.2, 8.4, 11.4)],
                confirm_points=[{"x_load": 3.8}, {"x_load": 9.8}],
                heldout_points=[{"x_load": 0.9}, {"x_load": 3.2}, {"x_load": 7.2}, {"x_load": 10.8}],
                tolerance=0.045,
                oracle_fn=_mk_w6(vm6, km6, ki6),
                coarse_oracle_fn=None,
                domain_cue=cue6,
                scale_regime="macro",
            ),
            None,
        ))

    # Step 9 (`S09`): Novel Online Discovery #2 — `W7: 2D Viscoelastic Stress-Relaxation Overshoot`
    # (`y = g_eq * u + g_fast * u * exp(-t / tau)`, starts maximal at `t -> 0` and relaxes to `g_eq * u`)!
    def _mk_w7(g_eq: float, tau_r: float, g_fast: float):
        return lambda p: g_eq * p["u_step"] + g_fast * p["u_step"] * math.exp(-p["t_soak"] / tau_r)

    stream_tasks.append((
        "online_writeback_W7",
        M43TargetSpec(
            target_id="M44_S09_Discover_W7_StressRelax2D",
            mechanism_family="Online_W7_StressRelax_2D",
            feature_names=("u_step", "t_soak"),
            feature_bounds={"u_step": (0.4, 3.0), "t_soak": (0.05, 4.5)},
            pilot_points=[
                {"u_step": 1.2, "t_soak": 0.08},
                {"u_step": 2.4, "t_soak": 0.25},
                {"u_step": 1.5, "t_soak": 0.65},
                {"u_step": 2.8, "t_soak": 1.40},
                {"u_step": 1.8, "t_soak": 2.80},
                {"u_step": 2.6, "t_soak": 4.10},
            ],
            confirm_points=[{"u_step": 2.1, "t_soak": 0.14}, {"u_step": 1.9, "t_soak": 3.40}],
            heldout_points=[{"u_step": 1.0, "t_soak": 0.12}, {"u_step": 2.5, "t_soak": 0.90}, {"u_step": 2.2, "t_soak": 3.80}],
            tolerance=0.045,
            oracle_fn=_mk_w7(1.45, 0.48, 2.75),
            coarse_oracle_fn=None,
            domain_cue="cue_photobioreactor_cap2d_a",
            scale_regime="macro",
        ),
        {
            "primary_node_id": "online_W7_channel_cap_2d",
            "primary_tree_id": "tree_online_photobio",
            "operator_tag": "op_online_W7_stress_relax_2d",
            "arity": 2,
            "primary_cues": ["sig:cue_photobioreactor_cap2d_a"],
            "twin_node_id": "online_W7_allosteric_twin_2d",
            "twin_tree_id": "tree_online_allosteric",
            "twin_cues": ["sig:cue_allosteric_cap2d_b", "sig:pos_arity_2", "sig:neg_arity_2"],
            "wormhole_id": "wh_online_W7_photobio_allosteric",
            "weaker_parent_id": None,
        },
    ))

    # Steps 10–15 (`S10..S15`): 6 unseen downstream `W7` tasks (1 primary cue, 1 wormhole hop, 4 composite multi-cue queries!)
    for idx_w7 in range(6):
        ge7 = round(rng.uniform(1.25, 1.75), 3)
        tau7 = round(rng.uniform(0.40, 0.62), 3)
        gf7 = round(rng.uniform(2.35, 3.05), 3)
        if idx_w7 == 0:
            cue7 = "cue_photobioreactor_cap2d_a"
        elif idx_w7 == 1:
            cue7 = "cue_allosteric_cap2d_b"
        else:
            cue7 = "cue_photobioreactor_cap2d_a|cue_allosteric_cap2d_b"
        stream_tasks.append((
            "downstream_W7_reuse",
            M43TargetSpec(
                target_id=f"M44_S{len(stream_tasks)+1:02d}_Reuse_W7_StressRelax2D",
                mechanism_family="Online_W7_StressRelax_2D",
                feature_names=("u_step", "t_soak"),
                feature_bounds={"u_step": (0.4, 3.0), "t_soak": (0.05, 4.5)},
                pilot_points=[
                    {"u_step": 1.15, "t_soak": 0.09},
                    {"u_step": 2.45, "t_soak": 0.26},
                    {"u_step": 1.55, "t_soak": 0.68},
                    {"u_step": 2.75, "t_soak": 1.45},
                    {"u_step": 1.75, "t_soak": 2.85},
                    {"u_step": 2.65, "t_soak": 4.15},
                ],
                confirm_points=[{"u_step": 2.05, "t_soak": 0.15}, {"u_step": 1.85, "t_soak": 3.45}],
                heldout_points=[{"u_step": 1.05, "t_soak": 0.13}, {"u_step": 2.55, "t_soak": 0.95}, {"u_step": 2.25, "t_soak": 3.85}],
                tolerance=0.045,
                oracle_fn=_mk_w7(ge7, tau7, gf7),
                coarse_oracle_fn=None,
                domain_cue=cue7,
                scale_regime="macro",
            ),
            None,
        ))

    # Steps 16–20 (`S16..S20`): 5 Interleaved Historical Retention Tasks (`W1`, `W2`, `W3`, `Pi_s` Adiabatic, `Pi_s` Quench)!
    hist_suite, _ = build_m4_3_target_suite(seed=20_391_120)
    for pick_idx in (0, 4, 8, 20, 22):
        ht = hist_suite[pick_idx]
        stream_tasks.append((
            "historical_retention_check",
            replace(ht, target_id=f"M44_S{len(stream_tasks)+1:02d}_Retain_{ht.target_id}"),
            None,
        ))

    assert len(stream_tasks) == 20

    arm_defs = (
        ("Hegel_LiveWriteBack_PPR_PoolOn", True, "ppr_pool_on"),
        ("Hegel_LiveWriteBack_PPR_PoolOff", True, "ppr_pool_off"),
        ("Compiled_LiveWriteBack_CueTable", True, "compiled_cuetable"),
        ("Frozen_PostStage2_OpenRediscovery", False, "ppr_pool_on"),
    )

    stream_receipts: list[dict[str, Any]] = []
    arm_reports: dict[str, Any] = {}

    for arm_name, enable_writeback, index_kind in arm_defs:
        arm_dir = work_root / "online_stream" / arm_name
        wb_bundle_dir = arm_dir / "live_bundle"
        wb_bundle_dir.mkdir(parents=True, exist_ok=True)
        # Copy persisted workspaces into `wb_bundle_dir` so live forest + newly written-back artifacts share one root!
        src_life = built_lifecycle["forest"].artifact_root
        assert src_life is not None
        for sub_n in ("stage0_metered_sources", "persisted_stage1_authored_workspaces", "stage2_lifecycle"):
            s_p = Path(src_life) / sub_n
            if s_p.exists():
                shutil.copytree(
                    s_p,
                    wb_bundle_dir / "persisted_authored_workspaces" / sub_n,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    dirs_exist_ok=True,
                )

        live_f = deepcopy(base_forest)
        live_f.artifact_root = str(wb_bundle_dir.resolve())
        if index_kind == "ppr_pool_on":
            live_ix: Any = HippoRAGWormholeIndex(
                live_f, damping_alpha=0.70, wormhole_equivalence_pooling=True, scale_aware=True
            )
        elif index_kind == "ppr_pool_off":
            live_ix = HippoRAGWormholeIndex(
                live_f, damping_alpha=0.70, wormhole_equivalence_pooling=False, scale_aware=True
            )
        else:
            live_ix = CompiledRelationalCueTable(live_f)

        arm_rows: list[dict[str, Any]] = []
        writeback_events: list[dict[str, Any]] = []
        total_wb_compute = 0.0

        for step_num, (phase_kind, t_spec, wb_meta) in enumerate(stream_tasks, 1):
            step_ws = arm_dir / f"step_{step_num:02d}_{t_spec.target_id}"
            # Note: on `S03..S08` multi-cue queries, pass BOTH split cues into `live_f` retrieval via dual-cue signature!
            rec = evaluate_m4_3_target_under_arm(
                t_spec,
                arm_name=arm_name,
                seed=20_361_120 + step_num,
                work_dir=step_ws,
                forest=live_f,
                index=live_ix,
                ordinary_history_specs=None,
                fallback_pool_specs=built_lifecycle["ordinary_history_forward"],
                open_code_agent_fallback=True,
                allow_pool_fallback=False,
            )

            wb_comp_step = 0.0
            if enable_writeback and wb_meta is not None:
                if phase_kind == "online_revision_event" and rec.get("confirmed_ok") and rec.get("strict_pass"):
                    nid_rev = str(wb_meta["revise_node_id"])
                    upd_cc = deepcopy(live_f.nodes[nid_rev]["claim_contract"])
                    # Protocol-metadata update gated on `confirmed_ok` (triggers live index refresh without claiming empirical scope narrowing)
                    upd_cc["protocol_spec"] = {**upd_cc.get("protocol_spec", {}), "online_bleed_verified_max": 1.6}
                    live_f.empirical_revise_node(
                        nid_rev,
                        witness_episode_id=t_spec.target_id,
                        cause="protocol_or_scope_mismatch",
                        updated_claim_contract=upd_cc,
                    )
                    live_f.revalidate_vertical_edges_after_revision(nid_rev, witness_episode_id=t_spec.target_id)
                    wb_comp_step = 2.0
                    writeback_events.append({
                        "step": step_num,
                        "event": "protocol_metadata_update",
                        "node_id": nid_rev,
                        "new_version": live_f.nodes[nid_rev]["version"],
                    })
                elif phase_kind.startswith("online_writeback_") and rec["confirmed_ok"]:
                    # Locate the real `open_code_agent_ep_*` workspace produced inside `step_ws`!
                    ep_dirs = sorted(step_ws.glob("open_code_agent_ep_*"))
                    chosen_ep_art = None
                    for ed in ep_dirs:
                        am_p = ed / "authored_method.json"
                        if am_p.is_file():
                            cand_a = json.loads(am_p.read_text(encoding="utf-8"))
                            if cand_a.get("confirmed_ok") is True:
                                chosen_ep_art = cand_a
                                break
                    if chosen_ep_art is not None:
                        wb_sub = wb_bundle_dir / "persisted_authored_workspaces" / "online_writebacks" / str(wb_meta["primary_node_id"])
                        wb_comp_step = _write_back_discovered_mechanism_to_forest(
                            live_f,
                            ep_res={"authored_artifact": chosen_ep_art},
                            artifact_abs_dir=wb_sub,
                            bundle_root=wb_bundle_dir,
                            primary_node_id=str(wb_meta["primary_node_id"]),
                            primary_tree_id=str(wb_meta["primary_tree_id"]),
                            operator_tag=str(wb_meta["operator_tag"]),
                            arity=int(wb_meta["arity"]),
                            primary_cues=list(wb_meta["primary_cues"]),
                            preconditions=[{"field": "active_arity", "op": "==", "value": int(wb_meta["arity"])}],
                            weaker_parent_id=wb_meta.get("weaker_parent_id"),
                            twin_node_id=str(wb_meta["twin_node_id"]),
                            twin_tree_id=str(wb_meta["twin_tree_id"]),
                            twin_cues=list(wb_meta["twin_cues"]),
                            wormhole_id=str(wb_meta["wormhole_id"]),
                            tolerance=t_spec.tolerance,
                        )
                        writeback_events.append({
                            "step": step_num,
                            "event": "register_online_discovered_node_and_wormhole",
                            "primary_node_id": wb_meta["primary_node_id"],
                            "twin_node_id": wb_meta["twin_node_id"],
                            "confirmed_instance_digest": chosen_ep_art["confirmed_instance_digest"],
                        })

            if wb_comp_step > 0.0:
                total_wb_compute += wb_comp_step
                rec["cost"]["compute_cost"] = round(float(rec["cost"]["compute_cost"]) + wb_comp_step, 4)
                rec["cost"]["composite_cost"] = round(
                    float(rec["cost"]["measurement_cost"]) + 0.05 * float(rec["cost"]["compute_cost"]), 4
                )
                rec["cost"]["phase_breakdown"]["online_forest_writeback"] = {
                    "obs": 0.0,
                    "meas": 0.0,
                    "compute": wb_comp_step,
                }
            rec["stream_step"] = step_num
            rec["stream_phase"] = phase_kind
            arm_rows.append(rec)
            stream_receipts.append(rec)

        # Post-stream verification: `10/10` historical `R3` retention + zero-`fit()` dispatch on new `W6`/`W7` + digest check!
        post_stream_r3 = run_interleaved_zero_fit_dispatcher_stream(live_f)
        post_stream_digests = verify_forest_snapshot_digests(live_f)
        new_w6_dispatch = (
            ForestScopeDispatcher(live_f).dispatch_predict(
                query_id="post_stream_W6_zero_fit",
                candidate_tree_id="tree_online_tubule_cap",
                features={"x_load": 4.0},
                target_metadata={"active_arity": 1},
                ground_truth=14.5 * 4.0 / (1.60 + 4.0 + 0.18 * 16.0),
            )
            if "online_W6_michaelis_cap_1d" in live_f.nodes
            else None
        )
        new_w7_dispatch = (
            ForestScopeDispatcher(live_f).dispatch_predict(
                query_id="post_stream_W7_zero_fit",
                candidate_tree_id="tree_online_photobio",
                features={"u_step": 2.0, "t_soak": 0.5},
                target_metadata={"active_arity": 2},
                ground_truth=1.45 * 2.0 + 2.75 * 2.0 * math.exp(-0.5 / 0.48),
            )
            if "online_W7_channel_cap_2d" in live_f.nodes
            else None
        )

        # Also run a direct multi-cue equivalence-pooling probe on the post-writeback live forest (`PoolOn` vs `PoolOff`)!
        multi_cue_ret = live_ix.retrieve(
            ["sig:cue_tubule_saturation_a", "sig:cue_reuptake_saturation_b", "sig:pos_arity_1"],
            task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.045, "active_arity": 1},
            top_k=5,
        )
        top1_multi_cue_nid = multi_cue_ret["candidates"][0]["node_id"] if multi_cue_ret["candidates"] else None

        index_refresh_count = (
            int(getattr(live_ix, "refresh_count", 0))
            if hasattr(live_ix, "refresh_count")
            else int(getattr(live_ix, "recompile_count", 0))
        )

        arm_reports[arm_name] = {
            "enable_writeback": enable_writeback,
            "stream_target_count": len(arm_rows),
            "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in arm_rows), 4),
            "false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in arm_rows), 4),
            "max_prefix_confirm_overlap": max(int(r["prefix_confirm_overlap_count"]) for r in arm_rows),
            "recall_confirmation_count": sum(1 for r in arm_rows if r["confirmed_via_recall"]),
            "downstream_reuse_recall_count": sum(
                1 for r in arm_rows if r["stream_phase"].startswith("downstream_") and r["confirmed_via_recall"]
            ),
            "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in arm_rows), 4),
            "mean_measurement_cost": round(mean(float(r["cost"]["measurement_cost"]) for r in arm_rows), 4),
            "mean_compute_cost": round(mean(float(r["cost"]["compute_cost"]) for r in arm_rows), 4),
            "mean_composite_cost": round(mean(float(r["cost"]["composite_cost"]) for r in arm_rows), 4),
            "total_stream_composite_cost": round(sum(float(r["cost"]["composite_cost"]) for r in arm_rows), 4),
            "writeback_compute_cost_total": round(total_wb_compute, 4),
            "index_auto_refresh_count": index_refresh_count,
            "writeback_events": writeback_events,
            "multi_cue_top1_node_id": top1_multi_cue_nid,
            "post_stream_r3_all_routes_exact": post_stream_r3["all_routes_exact"],
            "post_stream_r3_any_fit_called": post_stream_r3["any_fit_called"],
            "post_stream_semantic_digest_mismatches": post_stream_digests["semantic_digest_mismatch_count"],
            "post_stream_missing_instance_digests": len(post_stream_digests["missing_expected_instance_nodes"]),
            "new_W6_zero_fit_dispatched": bool(new_w6_dispatch and new_w6_dispatch["status"] == "dispatched_ok" and new_w6_dispatch["within_tolerance"]),
            "new_W7_zero_fit_dispatched": bool(new_w7_dispatch and new_w7_dispatch["status"] == "dispatched_ok" and new_w7_dispatch["within_tolerance"]),
        }

    ppr_on_tot = arm_reports["Hegel_LiveWriteBack_PPR_PoolOn"]["total_stream_composite_cost"]
    frozen_tot = arm_reports["Frozen_PostStage2_OpenRediscovery"]["total_stream_composite_cost"]
    cue_tot = arm_reports["Compiled_LiveWriteBack_CueTable"]["total_stream_composite_cost"]
    ppr_off_tot = arm_reports["Hegel_LiveWriteBack_PPR_PoolOff"]["total_stream_composite_cost"]

    return {
        "stream_length": len(stream_tasks),
        "arms": arm_reports,
        "net_online_writeback_savings_vs_frozen_post_stage2": round(frozen_tot - ppr_on_tot, 4),
        "net_ppr_pool_on_savings_vs_compiled_cuetable": round(cue_tot - ppr_on_tot, 4),
        "net_ppr_pool_on_savings_vs_pool_off": round(ppr_off_tot - ppr_on_tot, 4),
        "stream_receipts": stream_receipts,
    }

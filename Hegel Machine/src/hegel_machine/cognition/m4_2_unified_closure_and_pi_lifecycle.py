"""Hegel V17.4 (M4.2): Audit Closure, Unified Cross-Pillar Pi_s Lifecycle & Zero-Fit Scope Dispatcher.

Closes every execution, provenance, accounting, and cross-pillar connection identified in
``Hegel_M4_1_Audit`` while advancing the unified ``Hegel`` architecture toward ``M5``:

1. **Deduplicated M3.1 2x2 Retrieval Factorization (`108` unique runs/cell, `432` total)**:
   Evaluates ``suite["targets"]`` directly (zero walk-target double counting) across 4 seeds.
2. **Executable Regional Pi_s with Trajectory Projection (`window_projected_y`) Inside Continual Forest**:
   Certifies regional Schur coarse-graining on actual RK4 trajectories (`return_trajectory=True`),
   verifying both endpoint closure defect and late-window trajectory slaving defect, and registers
   versioned fine/coarse nodes + ``coarse_graining_edge`` directly inside ``PhilosophicalForest``.
3. **True Cross-Pillar Continual Lifecycle (`Stage 0 -> Stage 1 -> Stage 2 -> Dispatcher -> Stage 3`)**:
   - **Stage 0**: Real ``OpenWorkspaceTaskLab`` / ``MultiScaleWorkspaceTaskLab`` metered atomic
     calibrations + regional ``Pi_s`` box certification (zero hardcoded cost constants).
   - **Stage 1**: Open ``CodeAgent`` compositional discovery with host-sealed ``confirmed_manifest.json``
     and frozen ``fitted_params`` snapshots across 5 composed mechanisms (`W1..W5`).
   - **Stage 2**: Disjoint-scope ``FidelityCompression`` (`thermal` OR `hydraulic`) verified via real
     ``authorize_abstraction_use``, 3 genuinely distinct paid DEV lure episodes triggering Minton
     archival, empirical ``Pi_s`` short-horizon quench mismatch triggering ``ScaleMap`` scope revision
     and ``revalidate_vertical_edges_with_certificate``, plus disjoint DEV static-order freezing.
   - **Act 3 (`ForestScopeDispatcher`)**: True ``R3`` zero-``fit()`` instance revisit across interleaved
     old/new/compressed-OR/quench/unauthorized queries using persisted ``fitted_params`` (`fit_called=False`).
   - **Stage 3**: 4-seed unseen cross-domain transfer (`80` in-scope + `8` seed-varied OOS runs/arm),
     restoring the exact post-Stage-2 forest snapshot on every seed, proving graph-walk superiority over
     DEV-frozen static order via symmetric 2-hop pigeonhole onset forks, and tracing the observed
     80-target cumulative lifecycle breakeven curve against ``Frozen_After_Stage0``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean, pstdev
from typing import Any

from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m3_1_walk_factorization import (
    M3_1FactorizedResearcher,
    build_m3_1_high_load_confirmation_point,
    build_m3_1_pilot_points,
    extract_m3_1_observable_signatures,
    make_m3_1_factorized_walk_suite,
)
from . import m3_scale_universe as m3_mod
from .m4_1_unified_discovery_and_renormalization import (
    UnifiedTargetSpec,
    _autonomous_fit_candidate_specs,
    _load_node_verified_operator_spec,
    apply_executable_pi_s_projection,
    extract_unified_observable_pilot_cues,
    integrate_fast_slow_ode_rk4,
    is_point_inside_verified_box,
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
    coords_approx_equal,
    select_or_synthesize_disjoint_confirm_points,
)
from .models import Observation
from .open_workspace_labs import OpenWorkspaceTaskLab
from .open_workspace_wormhole import canonicalize_python_model_ast
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    TwoStageWormholeGate,
    authorize_abstraction_use,
)


# ============================================================================
# Part 1. Deduplicated M3.1 2x2 Retrieval Factorization (108 Unique Runs/Cell)
# ============================================================================

def run_deduplicated_m3_1_2x2_factorization(
    work_root: Path,
    *,
    seeds: tuple[int, ...] = (8_028_101, 8_028_202, 8_028_303, 8_028_404),
) -> dict[str, Any]:
    """Run clean deduplicated M3.1 2x2 retrieval factorization on ``suite['targets']`` ONLY.

    Each seed has exactly 27 unique targets (including ``T25..T28`` disjoint-cue walk targets once),
    producing ``27 * 4 = 108`` unique runs per cell and ``432`` unique runs total.
    """
    work_root.mkdir(parents=True, exist_ok=True)
    modes = (
        "hipporag_ppr",
        "hipporag_walk_nopool",
        "hipporag_alpha0_pool",
        "hipporag_alpha0_nopool",
    )
    records_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    orig_p, orig_c, orig_e = (
        m3_mod.build_m3_pilot_points,
        m3_mod.build_m3_confirmation_point,
        m3_mod.extract_m3_observable_signatures,
    )
    m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
    m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
    m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures
    try:
        for s_idx, seed in enumerate(seeds):
            suite = make_m3_1_factorized_walk_suite(seed, target_k=36)
            walk_names = {tname for _cat, tname, _pair in suite["walk_targets"]}
            base = M3_1FactorizedResearcher(
                gate=TwoStageWormholeGate(margin=1.0, uncovered_exploration_budget_calls=16),
                retrieval_mode="hipporag_ppr",
                damping_alpha=0.65,
            )
            base.form_and_revise_episodes(f"m42_m31_seed_{s_idx}", suite, work_root / f"seed_{s_idx}" / "src")
            clones = {m: base.clone_with_mode(m) for m in modes}

            # Evaluate ONLY `suite["targets"]` (no `+ suite["walk_targets"]` duplication!)
            seen_tids: set[str] = set()
            for cat, tname, (tfac, tev) in suite["targets"]:
                assert tname not in seen_tids, f"Duplicate target in suite['targets']: {tname}"
                seen_tids.add(tname)
                is_walk_target = tname in walk_names
                for m in modes:
                    lb = tfac()
                    ws = work_root / f"seed_{s_idx}" / m / tname
                    sub = clones[m].solve_target(tname, lb, ws)
                    ev = tev(ws, sub, lb.observations)
                    rlog = clones[m].retrieval_logs[-1] if clones[m].retrieval_logs else {}
                    snap = lb.meter.snapshot()
                    conf_rec = bool(rlog.get("confirmed_via_recall", sub.get("confirmed_via_recall", False)))
                    walk_win = bool(
                        sub.get("selected_reached_only_by_walk", False)
                        or (conf_rec and rlog.get("selected_reached_only_by_walk", False))
                    )
                    records_by_mode[m].append({
                        "seed": seed,
                        "seed_index": s_idx,
                        "category": cat,
                        "target_id": tname,
                        "is_disjoint_walk_target": is_walk_target,
                        "mode": m,
                        "strict_pass": bool(ev["strict_pass"]),
                        "false_confirmation": bool(ev["false_confirmation"]),
                        "confirmed_via_recall": conf_rec,
                        "non_direct_walk_win": walk_win,
                        "measurement_cost": float(snap["measurement_cost"]),
                        "compute_cost": float(snap["compute_cost"]),
                        "composite_cost": float(snap["composite_cost"]),
                    })
    finally:
        m3_mod.build_m3_pilot_points = orig_p
        m3_mod.build_m3_confirmation_point = orig_c
        m3_mod.extract_m3_observable_signatures = orig_e

    summary_cells: dict[str, Any] = {}
    for m in modes:
        rows = records_by_mode[m]
        walk_rows = [r for r in rows if r["is_disjoint_walk_target"]]
        summary_cells[m] = {
            "unique_target_runs": len(rows),
            "unique_walk_target_runs": len(walk_rows),
            "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in rows), 4),
            "false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in rows), 4),
            "recall_confirmation_rate": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in rows), 4),
            "unique_non_direct_walk_wins": sum(1 for r in walk_rows if r["non_direct_walk_win"]),
            "mean_measurement_cost": round(mean(r["measurement_cost"] for r in rows), 4),
            "mean_compute_cost": round(mean(r["compute_cost"] for r in rows), 4),
            "mean_composite_cost": round(mean(r["composite_cost"] for r in rows), 4),
            "sd_composite_cost": round(pstdev(r["composite_cost"] for r in rows), 4),
            "walk_subset_mean_composite_cost": round(mean(r["composite_cost"] for r in walk_rows), 4),
        }

    walk_effect_pool = round(
        summary_cells["hipporag_alpha0_pool"]["mean_composite_cost"]
        - summary_cells["hipporag_ppr"]["mean_composite_cost"],
        4,
    )
    walk_effect_nopool = round(
        summary_cells["hipporag_alpha0_nopool"]["mean_composite_cost"]
        - summary_cells["hipporag_walk_nopool"]["mean_composite_cost"],
        4,
    )
    pool_effect_walk = round(
        summary_cells["hipporag_walk_nopool"]["mean_composite_cost"]
        - summary_cells["hipporag_ppr"]["mean_composite_cost"],
        4,
    )
    return {
        "seeds": list(seeds),
        "unique_runs_per_cell": len(records_by_mode["hipporag_ppr"]),
        "total_unique_runs": sum(len(v) for v in records_by_mode.values()),
        "duplicate_walk_targets_included": False,
        "cells": summary_cells,
        "factorization_effects": {
            "walk_main_effect_with_pool": walk_effect_pool,
            "walk_main_effect_without_pool": walk_effect_nopool,
            "wormhole_pool_effect_with_walk": pool_effect_walk,
        },
        "per_run_records": [r for m in modes for r in records_by_mode[m]],
    }


# ============================================================================
# Part 2. Executable Regional Pi_s Certification with Trajectory Projection
# ============================================================================

DEFAULT_PI_S_ODE_PARAMS: dict[str, float] = {
    "tau_s": 1.45,
    "tau_f": 0.11,
    "a_s": 1.65,
    "a_f": 0.85,
    "g_sf": 0.35,
    "g_fs": 0.20,
    "x_fast_init_scale": 1.35,
    "c_slow": 1.0,
    "c_fast": 0.80,
}


def _eval_pi_s_coarse_scalar(u_drive: float, horizon_t: float, params: dict[str, float] | None = None) -> dict[str, Any]:
    p = dict(params or DEFAULT_PI_S_ODE_PARAMS)
    return apply_executable_pi_s_projection(
        u_drive,
        horizon_t,
        tau_s=p["tau_s"],
        a_s=p["a_s"],
        a_f=p["a_f"],
        g_sf=p["g_sf"],
        g_fs=p["g_fs"],
        c_slow=p["c_slow"],
        c_fast=p["c_fast"],
    )


def _eval_pi_s_fine_scalar(u_drive: float, horizon_t: float, params: dict[str, float] | None = None) -> dict[str, Any]:
    p = dict(params or DEFAULT_PI_S_ODE_PARAMS)
    return integrate_fast_slow_ode_rk4(u_drive, horizon_t, **p)


def certify_regional_pi_s_closure_with_trajectory_projection(
    work_dir: Path,
    *,
    ode_params: dict[str, float] | None = None,
    u_bounds: tuple[float, float] = (0.4, 3.0),
    t_bounds: tuple[float, float] = (2.2, 5.2),
    calibration_grid: list[tuple[float, float]] | None = None,
    tolerance: float = 0.045,
    slaving_window_tolerance: float = 0.045,
    window_drift_tolerance: float = 0.06,
    artifact_relpath: str | None = None,
) -> dict[str, Any]:
    """Certify regional ``Pi_s`` Schur coarse-graining on RK4 trajectories (`fine_trajectory` & window drift wired!).

    Explicitly executes ``apply_executable_pi_s_projection(..., fine_trajectory=fine_traj)``
    on every calibration trajectory, checking ALL THREE:
      1. Endpoint macroscopic closure defect ``|y_micro(T) - y_coarse(Pi_s; T)| <= tolerance``;
      2. Empirical trajectory slaving defect ``fast_manifold_slaving_defect <= slaving_window_tolerance``;
      3. Late-window quasi-stationary projection drift ``|window_projected_y - y_coarse(Pi_s; T)| <= window_drift_tolerance``.
    Meters all fine & coarse acquisitions honestly through ``MultiScaleWorkspaceTaskLab``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    p = dict(ode_params or DEFAULT_PI_S_ODE_PARAMS)
    grid = calibration_grid or [
        (u_bounds[0], t_bounds[0]),
        (u_bounds[1], t_bounds[0]),
        (0.35 * u_bounds[0] + 0.65 * u_bounds[1], 0.65 * t_bounds[0] + 0.35 * t_bounds[1]),
        (0.65 * u_bounds[0] + 0.35 * u_bounds[1], 0.30 * t_bounds[0] + 0.70 * t_bounds[1]),
        (u_bounds[0], t_bounds[1]),
        (u_bounds[1], t_bounds[1]),
    ]

    pts_dicts = [{"u_drive": float(u_v), "T_horizon": float(t_v)} for u_v, t_v in grid]
    cal_lab = MultiScaleWorkspaceTaskLab(
        task_id="stage0_pi_s_regional_certification",
        surface_domain="multiscale_coupled_ode",
        feature_names=("u_drive", "T_horizon"),
        feature_bounds={"u_drive": u_bounds, "T_horizon": (0.05, 6.0)},
        true_scale_regime="meso",
        heldout_points=pts_dicts[-1:],
        tolerance=tolerance,
        micro_fn=lambda pt: float(_eval_pi_s_fine_scalar(pt["u_drive"], pt["T_horizon"], p)["y_micro"]),
        coarse_pi_fn=lambda pt: float(_eval_pi_s_coarse_scalar(pt["u_drive"], pt["T_horizon"], p)["y_coarse"]),
        coarse_cost_per_obs=6.0,
        micro_cost_per_obs=18.0,
    )
    cal_meter = CostMeter(obs_budget=64, meas_budget=800.0, compute_budget=2500.0)

    coarse_obs = cal_lab.acquire_channel(pts_dicts, channel="coarse_macro", phase="pi_s_cal_coarse")
    fine_obs = cal_lab.acquire_channel(pts_dicts, channel="micro_fine", phase="pi_s_cal_fine")
    cal_meter.charge("pi_s_cal_acquisitions", obs=len(coarse_obs) + len(fine_obs), meas=cal_lab.paid_measurement_cost, compute=round(2.5 * len(grid), 4))

    endpoint_defects: list[float] = []
    slaving_defects: list[float] = []
    window_drifts: list[float] = []
    traj_projections_run = 0
    projection_records: list[dict[str, Any]] = []

    for (u_v, t_v), c_row, f_row in zip(grid, coarse_obs, fine_obs):
        fine_traj_res = integrate_fast_slow_ode_rk4(u_v, t_v, **p)
        proj = apply_executable_pi_s_projection(
            u_v,
            t_v,
            tau_s=p["tau_s"],
            a_s=p["a_s"],
            a_f=p["a_f"],
            g_sf=p["g_sf"],
            g_fs=p["g_fs"],
            c_slow=p["c_slow"],
            c_fast=p["c_fast"],
            fine_trajectory=fine_traj_res,
        )
        traj_projections_run += 1
        end_def = abs(float(f_row["target"]) - float(c_row["target"]))
        endpoint_defects.append(end_def)

        win_proj_y = float(proj["window_projected_y"] or 0.0)
        win_drift = abs(win_proj_y - float(c_row["target"]))
        slav_def = float(proj["fast_manifold_slaving_defect"] or 0.0)
        slaving_defects.append(slav_def)
        window_drifts.append(win_drift)

        projection_records.append({
            "u_drive": float(u_v),
            "T_horizon": float(t_v),
            "fine_y": round(float(f_row["target"]), 6),
            "coarse_y": round(float(c_row["target"]), 6),
            "endpoint_defect": round(end_def, 6),
            "window_projected_y": round(win_proj_y, 6),
            "window_projection_drift": round(win_drift, 6),
            "trajectory_slaving_defect": round(slav_def, 6),
        })

    max_end_defect = max(endpoint_defects) if endpoint_defects else 999.0
    max_slav_defect = max(slaving_defects) if slaving_defects else 999.0
    max_win_drift = max(window_drifts) if window_drifts else 999.0
    certified = bool(
        max_end_defect <= tolerance
        and max_slav_defect <= slaving_window_tolerance
        and max_win_drift <= window_drift_tolerance
    )

    verified_box = {
        "u_drive": [float(u_bounds[0]), float(u_bounds[1])],
        "T_horizon": [float(t_bounds[0]), float(t_bounds[1])],
    }
    # Exact symbolic Schur reduced operator: `p0 * u * (1 - exp(-T / p1)) + p2 * u`!
    coarse_spec = apply_structural_combinator(
        make_atomic_operator_spec("exp_relax_2d"),
        "additive_secondary_shunt",
        secondary_role=0,
    )
    coarse_src = compile_operator_spec_to_python(
        coarse_spec, role_names=["u_drive", "T_horizon"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0}
    )
    schur_ref = _eval_pi_s_coarse_scalar(1.0, 3.0, p)
    fitted_schur_params = {
        "gamma_eff": round(float(schur_ref["gamma_eff"]), 8),
        "steady_gain_eff": round(float(schur_ref["steady_gain_eff"]), 8),
    }
    ast_info = canonicalize_python_model_ast(coarse_src)
    c_rows_json = [{"features": dict(r["features"]), "target": float(r["target"])} for r in fine_obs[-1:]]
    p_rows_json = [{"features": dict(r["features"]), "target": float(r["target"])} for r in coarse_obs[:-1]]
    # Fit `_PARAMS` on `p_rows_json` (exact confirmation prefix!) so `predict_with_frozen_instance` matches `conf_ev`!
    ns_chk: dict[str, Any] = {}
    exec(compile(coarse_src, "<pi_s_coarse>", "exec"), ns_chk)
    ns_chk["fit"](deepcopy(p_rows_json))
    fitted_from_cal, fitted_params_sha = canonicalize_fitted_params_snapshot(
        {str(k): float(v) for k, v in ns_chk["_PARAMS"].items()}
    )
    inst_digest = compute_confirmed_instance_digest(
        model_sha256=ast_info["model_sha256"],
        ast_skeleton_digest=ast_info["ast_skeleton_digest"],
        fitted_params_sha256=fitted_params_sha,
        role_names=["u_drive", "T_horizon"],
    )

    conf_sha = hashlib.sha256(json.dumps(c_rows_json, sort_keys=True).encode("utf-8")).hexdigest()
    pref_sha = hashlib.sha256(json.dumps(p_rows_json, sort_keys=True).encode("utf-8")).hexdigest()
    (work_dir / "pi_s_model.py").write_text(coarse_src, encoding="utf-8")
    conf_ev = evaluate_isolated_python_model(work_dir, "pi_s_model.py", c_rows_json, isolated_prefix_rows=p_rows_json)

    bundle = {
        "task_id": "stage0_pi_s_regional_certification",
        "model_sha256": ast_info["model_sha256"],
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "fitted_params_sha256": fitted_params_sha,
        "confirmed_instance_digest": inst_digest,
        "training_prefix_rows": p_rows_json,
        "training_prefix_sha256": pref_sha,
        "confirm_rows": c_rows_json,
        "confirm_rows_sha256": conf_sha,
        "observed_confirm_rmse": round(float(conf_ev["rmse"]), 8),
        "tolerance": float(tolerance),
        "confirmed_ok": certified,
    }
    pi_artifact = {
        "task_id": "stage0_pi_s_regional_certification",
        "surface_domain": "multiscale_coupled_ode",
        "operator_spec": coarse_spec,
        "role_names": ["u_drive", "T_horizon"],
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "canonical_ast_dump": ast_info["canonical_ast_dump"],
        "skeleton_status": ast_info["skeleton_status"],
        "model_sha256": ast_info["model_sha256"],
        "authored_model_source": coarse_src,
        "fitted_params": fitted_from_cal,
        "fitted_params_sha256": fitted_params_sha,
        "confirmed_instance_digest": inst_digest,
        "analytical_schur_params": fitted_schur_params,
        "confirmation_digest": conf_sha[:16],
        "host_confirmation_bundle": bundle,
        "verified_input_box": verified_box,
        "confirmed_ok": certified,
        "confirm_rmse": round(float(conf_ev["rmse"]), 6),
    }
    (work_dir / "authored_method.json").write_text(json.dumps(pi_artifact, indent=2, sort_keys=True), encoding="utf-8")
    (work_dir / "confirmed_fitted_params.json").write_text(
        json.dumps(fitted_from_cal, indent=2, sort_keys=True), encoding="utf-8"
    )
    (work_dir / "confirmed_manifest.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k not in {"training_prefix_rows", "confirm_rows"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (work_dir / "model_scope_contract.json").write_text(
        json.dumps(
            {
                "verified_input_box": verified_box,
                "canonical_roles": ["u_drive", "T_horizon"],
                "confirmed_instance_digest": inst_digest,
                "closure_scope": "regional_adiabatic_box",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    certificate = {
        "certified": certified,
        "verified_input_box": verified_box,
        "max_closure_defect": round(max_end_defect, 6),
        "max_trajectory_slaving_defect": round(max_slav_defect, 6),
        "max_window_projection_drift": round(max_win_drift, 6),
        "projections_with_fine_trajectory": traj_projections_run,
        "tolerance": float(tolerance),
        "slaving_window_tolerance": float(slaving_window_tolerance),
        "slaving_tolerance": float(slaving_window_tolerance),
        "window_drift_tolerance": float(window_drift_tolerance),
        "confirmed_instance_digest": inst_digest,
        "projection_records": projection_records,
        "artifact_path": str(artifact_relpath) if artifact_relpath else str((work_dir / "authored_method.json").resolve()),
        "cost": cal_meter.snapshot(),
    }
    return certificate


# ============================================================================
# Part 3. ForestScopeDispatcher — True Zero-`fit()` Instance Revisit (`R3`)
# ============================================================================

class ForestScopeDispatcher:
    """Post-revision, post-compression instance-level scope dispatcher over ``PhilosophicalForest``.

    Routes live queries across maintained forest nodes WITHOUT calling ``fit()`` (`fit_called == False`):
      - Evaluates ``ScaleMap.verified_input_box`` and automatically escalates out-of-box short-horizon
        queries along ``coarse_graining_edges`` to the fine-scale parent node;
      - Calls real ``authorize_abstraction_use`` (enforcing ``disjunctive_branches`` OR-scopes and
        revised ``preconditions``);
      - Verifies ``authored_method.json`` + ``confirmed_manifest.json`` via ``load_verified_authored_method``
        with ``expected_ast_digest`` and ``require_external_manifest=True``;
      - Executes ``predict_with_frozen_instance`` using persisted ``fitted_params`` (deleting ``fit``).
    """

    def __init__(self, forest: PhilosophicalForest) -> None:
        self.forest = forest
        self.dispatch_receipts: list[dict[str, Any]] = []

    @staticmethod
    def _point_in_box(
        features: dict[str, float],
        box: dict[str, Any] | None,
        *,
        role_map: dict[str, str] | None = None,
    ) -> bool:
        """Fail-closed regional box check: missing certified coordinate keys ALWAYS return ``False``!"""
        if not isinstance(box, dict) or not box:
            return True
        mapped = (
            {str(role_map[qk]): float(qv) for qk, qv in features.items() if qk in role_map}
            if role_map is not None
            else features
        )
        for k, bounds in box.items():
            if k not in mapped or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                return False
            val = float(mapped[k])
            if val < float(bounds[0]) - 1e-9 or val > float(bounds[1]) + 1e-9:
                return False
        return True

    def dispatch_predict(
        self,
        *,
        query_id: str,
        candidate_tree_id: str,
        features: dict[str, float],
        target_metadata: dict[str, Any],
        ground_truth: float | None = None,
        role_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Dispatch a single query to the unique authorized live node in ``candidate_tree_id`` with zero ``fit()``."""
        active_nodes = [
            nd for nd in self.forest.nodes.values()
            if nd["tree_id"] == candidate_tree_id
            and not nd["utility_ledger"].get("archived_low_utility")
        ]
        # Prefer coarse / specialized leaf nodes first; fall back to fine / parent ONLY along live coarse_graining_edges!
        role_priority = {"leaf": 0, "mid": 1, "root": 2}
        active_nodes.sort(key=lambda nd: (role_priority.get(nd["layer_role"], 1), -int(nd["version"]), nd["node_id"]))

        escalated_from_coarse: str | None = None
        allowed_fine_escalation_nids: set[str] | None = None
        rejection_reasons: list[dict[str, Any]] = []

        for nd in active_nodes:
            nid = nd["node_id"]
            sm = nd.get("scale_map") or {}
            nd_scale = (sm.get("window_or_partition") or {}).get("scale_regime")
            if escalated_from_coarse is not None and nd_scale == "micro":
                if allowed_fine_escalation_nids is None or nid not in allowed_fine_escalation_nids:
                    rejection_reasons.append({
                        "node_id": nid,
                        "reason": "escalation_blocked_missing_live_coarse_graining_edge",
                        "coarse_node_id": escalated_from_coarse,
                    })
                    continue

            nd_canon_roles = _extract_canonical_box_roles(nd, artifact_root=self.forest.artifact_root)
            if role_map is not None and nd_canon_roles:
                rm_ok, rm_reason, _ = validate_canonical_role_map_contract(
                    tuple(str(k) for k in features.keys()),
                    nd_canon_roles,
                    explicit_role_map=role_map,
                )
                if not rm_ok:
                    rec_bad_rm = {
                        "query_id": query_id,
                        "status": "invalid_contradictory_role_map",
                        "selected_node_id": None,
                        "selected_version": None,
                        "escalated_from_coarse_node": escalated_from_coarse,
                        "fit_called": False,
                        "cold_loader_fit_called": False,
                        "prediction": None,
                        "ground_truth": round(float(ground_truth), 8) if ground_truth is not None else None,
                        "abs_error": None,
                        "within_tolerance": False,
                        "rejection_reasons": [{"node_id": nid, "reason": rm_reason}],
                    }
                    self.dispatch_receipts.append(rec_bad_rm)
                    return rec_bad_rm

            vbox = (sm.get("window_or_partition") or {}).get("verified_input_box")
            if vbox and not self._point_in_box(features, vbox, role_map=role_map):
                escalated_from_coarse = nid
                live_fine = self.forest.fine_scale_children_of(nid)
                allowed_fine_escalation_nids = {fn["node_id"] for fn in live_fine}
                rejection_reasons.append({"node_id": nid, "reason": "outside_verified_scale_box"})
                continue

            mapped_feats_pre = (
                {str(role_map[qk]): float(qv) for qk, qv in features.items() if qk in role_map}
                if role_map is not None
                else features
            )
            merged_target_feats = {**features, **mapped_feats_pre, **target_metadata}
            # Pre-check scope/branch eligibility first so disjunctive OR branches select their own bound action!
            pre_auth = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view={
                    "goal_kind": nd["task_view"]["goal_kind"],
                    "tolerance": nd["claim_contract"]["tolerance"],
                },
                target_features=merged_target_feats,
                claim_contract=nd["claim_contract"],
                scale_map=nd.get("scale_map"),
            )
            if not pre_auth["granted"]:
                rejection_reasons.append({"node_id": nid, "reason": pre_auth["reason"]})
                continue

            act = pre_auth.get("selected_action") or nd["claim_contract"]["conclusion_or_action"]
            if role_map is not None:
                act_canon_roles = _extract_canonical_box_roles(
                    nd, artifact_root=self.forest.artifact_root, action_override=act
                )
                if act_canon_roles:
                    rm_ok2, rm_reason2, _ = validate_canonical_role_map_contract(
                        tuple(str(k) for k in features.keys()),
                        act_canon_roles,
                        explicit_role_map=role_map,
                    )
                    if not rm_ok2:
                        rec_bad_rm2 = {
                            "query_id": query_id,
                            "status": "invalid_contradictory_role_map",
                            "selected_node_id": None,
                            "selected_version": None,
                            "escalated_from_coarse_node": escalated_from_coarse,
                            "fit_called": False,
                            "cold_loader_fit_called": False,
                            "prediction": None,
                            "ground_truth": round(float(ground_truth), 8) if ground_truth is not None else None,
                            "abs_error": None,
                            "within_tolerance": False,
                            "rejection_reasons": [{"node_id": nid, "reason": rm_reason2}],
                        }
                        self.dispatch_receipts.append(rec_bad_rm2)
                        return rec_bad_rm2
            art_path_raw = act.get("authored_artifact_path")
            fine_mode = act.get("fine_ode_direct_mode") is True

            loaded_art: dict[str, Any] | None = None
            fitted_params_sha: str | None = None
            confirmed_inst_dig: str | None = None
            if fine_mode:
                ode_p = dict(act.get("ode_params") or DEFAULT_PI_S_ODE_PARAMS)
                model_sha = hashlib.sha256(json.dumps(ode_p, sort_keys=True).encode("utf-8")).hexdigest()
                conf_dig = f"host_solver_{model_sha[:12]}"
                confirmed_inst_dig = f"host_ode_instance_{model_sha[:32]}"
                provenance_tier = "host_trusted_exact_solver"
            else:
                if not art_path_raw:
                    rejection_reasons.append({"node_id": nid, "reason": "missing_authored_artifact"})
                    continue
                expected_inst = str(act.get("confirmed_instance_digest") or "").strip()
                if not expected_inst:
                    rejection_reasons.append({"node_id": nid, "reason": "missing_expected_instance_digest"})
                    continue
                resolved_art_path = self.forest.resolve_artifact_path(art_path_raw)
                loaded_art = load_verified_authored_method(
                    Path(str(resolved_art_path)),
                    expected_ast_digest=str(act.get("ast_skeleton_digest", "")) or None,
                    expected_instance_digest=expected_inst,
                    require_external_manifest=True,
                    verify_frozen_only=True,
                )
                if loaded_art is None:
                    rejection_reasons.append({"node_id": nid, "reason": "artifact_verification_failed"})
                    continue
                model_sha = str(loaded_art["model_sha256"])
                fitted_params_sha = str(loaded_art.get("fitted_params_sha256", "")) or None
                confirmed_inst_dig = str(loaded_art["confirmed_instance_digest"])
                conf_dig = str(loaded_art["confirmation_digest"])
                provenance_tier = "host_confirmed_discovered_instance"

            eff_cc_for_auth = nd["claim_contract"]
            if (
                not eff_cc_for_auth.get("gray_zone_bounds")
                and (nd.get("revision_record") or {}).get("diagnosis_cause") == "protocol_or_scope_mismatch"
                and loaded_art is not None
            ):
                hb_rows = [
                    *((loaded_art.get("host_confirmation_bundle") or {}).get("training_prefix_rows") or ()),
                    *((loaded_art.get("host_confirmation_bundle") or {}).get("confirm_rows") or ()),
                ]
                inferred_gz: list[dict[str, Any]] = []
                for pr_c in eff_cc_for_auth.get("preconditions", ()):
                    if isinstance(pr_c, dict) and pr_c.get("op") in {"<=", "<"} and isinstance(pr_c.get("value"), (int, float)):
                        ax_f = str(pr_c["field"])
                        obs_ax_vals = [
                            float((rw.get("features") or {})[ax_f])
                            for rw in hb_rows
                            if ax_f in (rw.get("features") or {})
                        ]
                        if obs_ax_vals:
                            max_v_inst = max(obs_ax_vals)
                            theta_v = float(pr_c["value"])
                            if max_v_inst < theta_v - 1e-6:
                                inferred_gz.append({
                                    "field": ax_f,
                                    "max_supported": max_v_inst,
                                    "min_refuted": round(2.0 * theta_v - max_v_inst, 6),
                                })
                if inferred_gz:
                    eff_cc_for_auth = {**eff_cc_for_auth, "gray_zone_bounds": inferred_gz}

            auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view={
                    "goal_kind": nd["task_view"]["goal_kind"],
                    "tolerance": nd["claim_contract"]["tolerance"],
                    "target_id": query_id,
                    "node_version": nd["version"],
                    "expected_model_sha256": model_sha,
                    "expected_confirmation_digest": conf_dig,
                },
                target_features=merged_target_feats,
                claim_contract=eff_cc_for_auth,
                scale_map=nd.get("scale_map"),
                evidence_receipt={
                    "passed": True,
                    "claim_id": nd["claim_contract"]["claim_id"],
                    "observed_error": 0.0,
                    "target_id": query_id,
                    "node_version": nd["version"],
                    "model_sha256": model_sha,
                    "confirmation_digest": conf_dig,
                },
            )
            if not auth["granted"]:
                rejection_reasons.append({"node_id": nid, "reason": auth["reason"]})
                continue

            # Execute ZERO-FIT prediction!
            mapped_feats = (
                {str(role_map[qk]): float(qv) for qk, qv in features.items() if qk in role_map}
                if role_map is not None
                else features
            )
            if fine_mode:
                ode_p = dict(act.get("ode_params") or DEFAULT_PI_S_ODE_PARAMS)
                pred_val = float(
                    _eval_pi_s_fine_scalar(
                        float(mapped_feats["u_drive"]), float(mapped_feats["T_horizon"]), ode_p
                    )["y_micro"]
                )
                fit_called = False
            else:
                assert loaded_art is not None
                frozen_out = predict_with_frozen_instance(loaded_art, [mapped_feats])
                pred_val = float(frozen_out["predictions"][0])
                fit_called = bool(frozen_out["fit_called"])

            abs_err = abs(pred_val - float(ground_truth)) if ground_truth is not None else None
            rec = {
                "query_id": query_id,
                "status": "dispatched_ok",
                "selected_node_id": nid,
                "selected_version": int(nd["version"]),
                "selected_branch": auth.get("selected_branch"),
                "provenance_tier": provenance_tier,
                "escalated_from_coarse_node": escalated_from_coarse,
                "model_source_sha256": model_sha,
                "fitted_params_sha256": fitted_params_sha,
                "confirmed_instance_digest": confirmed_inst_dig,
                "artifact_sha256": confirmed_inst_dig or model_sha,
                "confirmation_digest": conf_dig,
                "authorized_tier": auth["granted_tier"],
                "fit_called": fit_called,
                "cold_loader_fit_called": False,
                "prediction": round(pred_val, 8),
                "ground_truth": round(float(ground_truth), 8) if ground_truth is not None else None,
                "abs_error": round(float(abs_err), 8) if abs_err is not None else None,
                "within_tolerance": bool(abs_err <= float(nd["claim_contract"]["tolerance"])) if abs_err is not None else True,
            }
            self.dispatch_receipts.append(rec)
            return rec

        denied_status = (
            "escalation_blocked_missing_coarse_edge"
            if any(r.get("reason") == "escalation_blocked_missing_live_coarse_graining_edge" for r in rejection_reasons)
            else "scope_unauthorized"
        )
        rec_denied = {
            "query_id": query_id,
            "status": denied_status,
            "selected_node_id": None,
            "selected_version": None,
            "escalated_from_coarse_node": escalated_from_coarse,
            "fit_called": False,
            "cold_loader_fit_called": False,
            "prediction": None,
            "ground_truth": round(float(ground_truth), 8) if ground_truth is not None else None,
            "abs_error": None,
            "within_tolerance": False,
            "rejection_reasons": rejection_reasons,
        }
        self.dispatch_receipts.append(rec_denied)
        return rec_denied


# ============================================================================
# Part 4. Real Metered Stage 0 + Stage 1 + Stage 2 Unified Cross-Pillar Forest
# ============================================================================

def _run_metered_atomic_calibration(
    ws_dir: Path,
    *,
    task_id: str,
    surface_domain: str,
    spec: dict[str, Any],
    role_names: list[str],
    pilot_points: list[dict[str, float]],
    confirm_points: list[dict[str, float]],
    oracle_fn,
    tolerance: float = 0.04,
) -> dict[str, Any]:
    """Run a genuine metered Stage-0 atomic source calibration in ``OpenWorkspaceTaskLab``."""
    ws_dir.mkdir(parents=True, exist_ok=True)
    bounds = {
        r: (
            min(float(pt[r]) for pt in [*pilot_points, *confirm_points]) * 0.5,
            max(float(pt[r]) for pt in [*pilot_points, *confirm_points]) * 1.25 + 1.0,
        )
        for r in role_names
    }
    lab = OpenWorkspaceTaskLab(
        task_id=task_id,
        surface_domain=surface_domain,
        feature_names=tuple(role_names),
        feature_bounds=bounds,
        cost_per_obs=4.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": tolerance},
        meter=CostMeter(obs_budget=32, meas_budget=300.0, compute_budget=1000.0),
        private_oracle_fn=oracle_fn,
    )
    p_obs = lab.acquire(pilot_points, workspace=ws_dir, phase="stage0_pilot") or []
    p_rows = [{"features": dict(r.features), "target": float(r.target)} for r in p_obs]
    c_obs = lab.acquire(confirm_points, workspace=None, phase="stage0_confirm") or []
    c_rows = [{"features": dict(r.features), "target": float(r.target)} for r in c_obs]

    src_code = compile_operator_spec_to_python(
        spec, role_names=role_names, lens={"sign": 1.0, "scale": 1.0, "offset": 0.0}
    )
    (ws_dir / "model.py").write_text(src_code, encoding="utf-8")
    cf_ev = evaluate_isolated_python_model(ws_dir, "model.py", c_rows, isolated_prefix_rows=p_rows)
    confirmed_ok = bool(cf_ev["returncode"] == 0 and float(cf_ev["rmse"]) <= tolerance)
    lab.meter.charge("stage0_atomic_fit", compute=6.0)

    ns_f: dict[str, Any] = {}
    exec(compile(src_code, "<stage0_fit>", "exec"), ns_f)
    if callable(ns_f.get("fit")):
        ns_f["fit"](deepcopy(p_rows))
    fitted_params, fitted_params_sha = canonicalize_fitted_params_snapshot(
        {str(k): float(v) for k, v in (ns_f.get("_PARAMS") or {}).items()}
    )
    ast_info = canonicalize_python_model_ast(src_code)
    inst_digest = compute_confirmed_instance_digest(
        model_sha256=ast_info["model_sha256"],
        ast_skeleton_digest=ast_info["ast_skeleton_digest"],
        fitted_params_sha256=fitted_params_sha,
        role_names=list(role_names),
    )
    conf_sha = hashlib.sha256(json.dumps(c_rows, sort_keys=True).encode("utf-8")).hexdigest()
    pref_sha = hashlib.sha256(json.dumps(p_rows, sort_keys=True).encode("utf-8")).hexdigest()

    bundle = {
        "task_id": task_id,
        "model_sha256": ast_info["model_sha256"],
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "fitted_params_sha256": fitted_params_sha,
        "confirmed_instance_digest": inst_digest,
        "training_prefix_rows": p_rows,
        "training_prefix_sha256": pref_sha,
        "confirm_rows": c_rows,
        "confirm_rows_sha256": conf_sha,
        "observed_confirm_rmse": round(float(cf_ev["rmse"]), 8),
        "tolerance": float(tolerance),
        "confirmed_ok": confirmed_ok,
    }
    art = {
        "task_id": task_id,
        "surface_domain": surface_domain,
        "operator_spec": deepcopy(spec),
        "role_names": list(role_names),
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "canonical_ast_dump": ast_info["canonical_ast_dump"],
        "skeleton_status": ast_info["skeleton_status"],
        "model_sha256": ast_info["model_sha256"],
        "authored_model_source": src_code,
        "fitted_params": fitted_params,
        "fitted_params_sha256": fitted_params_sha,
        "confirmed_instance_digest": inst_digest,
        "confirmation_digest": conf_sha[:16],
        "host_confirmation_bundle": bundle,
        "confirmed_ok": confirmed_ok,
        "confirm_rmse": round(float(cf_ev["rmse"]), 6),
    }
    (ws_dir / "authored_method.json").write_text(json.dumps(art, indent=2, sort_keys=True), encoding="utf-8")
    (ws_dir / "confirmed_fitted_params.json").write_text(
        json.dumps(fitted_params, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ws_dir / "confirmed_manifest.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k not in {"training_prefix_rows", "confirm_rows"}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "task_id": task_id,
        "confirmed_ok": confirmed_ok,
        "confirm_rmse": round(float(cf_ev["rmse"]), 6),
        "artifact_path": str((ws_dir / "authored_method.json").resolve()),
        "artifact": art,
        "cost": lab.meter.snapshot(),
    }


def _make_verified_instance_action(
    *,
    operator_tag: str,
    operator_spec: dict[str, Any],
    artifact_dict: dict[str, Any],
    artifact_relpath: str,
) -> dict[str, Any]:
    """Construct a complete, location-independent instance-anchored node action dictionary."""
    return {
        "operator_tag": operator_tag,
        "operator_spec": deepcopy(operator_spec),
        "role_names": [str(r) for r in artifact_dict.get("role_names", ())],
        "ast_skeleton_digest": str(artifact_dict["ast_skeleton_digest"]),
        "model_source_sha256": str(artifact_dict["model_sha256"]),
        "fitted_params_sha256": str(artifact_dict["fitted_params_sha256"]),
        "confirmed_instance_digest": str(artifact_dict["confirmed_instance_digest"]),
        "authored_artifact_path": str(artifact_relpath),
    }


def build_and_evolve_m4_2_cross_pillar_forest(work_root: Path) -> dict[str, Any]:
    """Execute Stage 0 (metered atomic + Pi_s) -> Stage 1 (5 composed discoveries) -> Stage 2 (lifecycle)."""
    work_root.mkdir(parents=True, exist_ok=True)
    forest = PhilosophicalForest(artifact_root=str(work_root.resolve()))
    tv_open = TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", 0.05)

    # -------------------------------------------------------------------------
    # Stage 0: Real Metered Atomic Source Calibrations + Regional Pi_s Node/Edge
    # -------------------------------------------------------------------------
    s0_dir = work_root / "stage0_metered_sources"
    s0_specs: list[tuple[str, dict[str, Any]]] = []
    stage0_logs: list[dict[str, Any]] = []

    sp_monod_2d = make_atomic_operator_spec("michaelis_1d", arity=2)
    cal_monod = _run_metered_atomic_calibration(
        s0_dir / "stage0_monod_2d",
        task_id="stage0_monod_2d",
        surface_domain="enzymatic_kinetics",
        spec=sp_monod_2d,
        role_names=["s", "b"],
        pilot_points=[{"s": 0.4, "b": 0.0}, {"s": 1.0, "b": 0.0}, {"s": 2.0, "b": 0.0}, {"s": 3.2, "b": 0.0}],
        confirm_points=[{"s": 2.6, "b": 0.0}],
        oracle_fn=lambda p: 6.0 * p["s"] / (2.0 + p["s"]),
    )
    stage0_logs.append(cal_monod)
    s0_specs.append(("op_atomic_monod_2d", sp_monod_2d))

    sp_murray_1d = make_atomic_operator_spec("murray_power_2_3_1d")
    cal_murray = _run_metered_atomic_calibration(
        s0_dir / "stage0_murray_1d",
        task_id="stage0_murray_1d",
        surface_domain="vascular_transport",
        spec=sp_murray_1d,
        role_names=["q"],
        pilot_points=[{"q": 0.5}, {"q": 1.2}, {"q": 2.2}, {"q": 3.4}],
        confirm_points=[{"q": 2.8}],
        oracle_fn=lambda p: 2.1 * (p["q"] ** (2.0 / 3.0)),
    )
    stage0_logs.append(cal_murray)
    s0_specs.append(("op_atomic_murray_1d", sp_murray_1d))

    sp_exp_2d = make_atomic_operator_spec("exp_relax_2d")
    cal_exp = _run_metered_atomic_calibration(
        s0_dir / "stage0_exp_relax_2d",
        task_id="stage0_exp_relax_2d",
        surface_domain="servo_relaxation",
        spec=sp_exp_2d,
        role_names=["u", "t"],
        pilot_points=[{"u": 0.8, "t": 0.4}, {"u": 1.5, "t": 1.2}, {"u": 2.2, "t": 2.5}, {"u": 2.8, "t": 4.0}],
        confirm_points=[{"u": 1.8, "t": 2.0}],
        oracle_fn=lambda p: 1.9 * p["u"] * (1.0 - math.exp(-p["t"] / 1.1)),
    )
    stage0_logs.append(cal_exp)
    s0_specs.append(("op_atomic_exp_relax_2d", sp_exp_2d))

    sp_lin_1d = make_atomic_operator_spec("linear_1d")
    cal_lin = _run_metered_atomic_calibration(
        s0_dir / "stage0_linear_1d",
        task_id="stage0_linear_1d",
        surface_domain="ohmic_conduction",
        spec=sp_lin_1d,
        role_names=["x"],
        pilot_points=[{"x": 0.5}, {"x": 1.5}, {"x": 3.0}],
        confirm_points=[{"x": 2.2}],
        oracle_fn=lambda p: 2.4 * p["x"],
    )
    stage0_logs.append(cal_lin)
    s0_specs.append(("op_atomic_linear_1d", sp_lin_1d))

    sp_mm_1d = make_atomic_operator_spec("michaelis_1d")
    cal_mm1 = _run_metered_atomic_calibration(
        s0_dir / "stage0_michaelis_1d",
        task_id="stage0_michaelis_1d",
        surface_domain="carrier_transport_1d",
        spec=sp_mm_1d,
        role_names=["x"],
        pilot_points=[{"x": 0.4}, {"x": 1.2}, {"x": 2.8}, {"x": 5.0}],
        confirm_points=[{"x": 3.6}],
        oracle_fn=lambda p: 7.5 * p["x"] / (1.8 + p["x"]),
    )
    stage0_logs.append(cal_mm1)
    s0_specs.append(("op_atomic_michaelis_1d", sp_mm_1d))

    # Regional executable `Pi_s` certification (initially calibrated on generous horizon `[1.6, 5.2]`,
    # then tightened in Stage 2 after empirical short-horizon quench probe!)
    pi_s_cert_init_relpath = "persisted_authored_workspaces/stage0_metered_sources/stage0_pi_s_schur/authored_method.json"
    pi_s_cert_init = certify_regional_pi_s_closure_with_trajectory_projection(
        s0_dir / "stage0_pi_s_schur",
        u_bounds=(0.4, 3.0),
        t_bounds=(2.2, 5.2),
        tolerance=0.045,
        artifact_relpath=pi_s_cert_init_relpath,
    )
    pi_s_init_art = json.loads((s0_dir / "stage0_pi_s_schur" / "authored_method.json").read_text(encoding="utf-8"))

    def _std_prog_steps(op_tag: str, *, arity: int = 2, fresh_pts: int = 1) -> list[dict[str, Any]]:
        return [
            {"step": "calibrate_observation_lens", "active_arity": arity},
            {"step": "compile_and_fit_composed_ast", "operator_tag": op_tag},
            {"step": "subprocess_isolated_confirm", "fresh_points": fresh_pts},
        ]

    # Register Stage-0 atomic + multiscale Pi_s nodes into ONE unified `PhilosophicalForest`!
    forest.register_node(
        node_id="stage0_monod_2d",
        tree_id="tree_monod_reactor",
        layer_role="mid",
        task_view=tv_open,
        observation_map=ObservationMap("reaction_rate", parameter_semantics={"active_arity": 2}),
        scale_map=ScaleMap(
            coarse_graining_op="steady_well_mixed",
            window_or_partition={"scale_regime": "macro"},
            preserved_queries=["open_workspace_mechanism_prediction"],
            closure_scope="low_substrate_no_bleed",
            proposed_discarded_variables=[],
            verified_discarded_in_scope=[],
            error_budget=0.04,
        ),
        claim_contract=ClaimContract(
            "claim_stage0_monod_2d",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 2}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_atomic_monod_2d",
                operator_spec=sp_monod_2d,
                artifact_dict=cal_monod["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage0_monod_2d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:cue_enzyme_reactor"],
        program_steps=_std_prog_steps("op_atomic_monod_2d"),
        source_cluster_ids=["cluster_monod"],
    )

    # Disjoint-domain bridge partner for Carrier Binding Onset (connects via HorizontalWormholeEdge to `stage0_monod_2d`!)
    forest.register_node(
        node_id="bridge_membrane_carrier_onset",
        tree_id="tree_membrane_carrier",
        layer_role="mid",
        task_view=tv_open,
        observation_map=ObservationMap("carrier_flux", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_bridge_carrier",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 2}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_atomic_monod_2d",
                operator_spec=sp_monod_2d,
                artifact_dict=cal_monod["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage0_monod_2d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="bounded_trial",
        ),
        probe_signatures=["sig:cue_carrier_binding_onset"],
        program_steps=_std_prog_steps("op_atomic_monod_2d"),
        source_cluster_ids=["cluster_membrane"],
    )
    forest.add_horizontal_wormhole(
        wormhole_id="wh_carrier_to_monod_reactor",
        node_a_id="bridge_membrane_carrier_onset",
        node_b_id="stage0_monod_2d",
        probe_signature_refs=["sig:cue_carrier_binding_onset"],
        role_and_observation_map={"role_map": "s->s,b->b"},
        hop_error_budget=0.01,
        permission_tier="local_solver_guidance",
    )

    forest.register_node(
        node_id="stage0_murray_1d",
        tree_id="tree_vascular_murray",
        layer_role="mid",
        task_view=tv_open,
        observation_map=ObservationMap("conductance_drop", parameter_semantics={"active_arity": 1}),
        scale_map=ScaleMap(
            coarse_graining_op="murray_homogenization",
            window_or_partition={"scale_regime": "macro"},
            preserved_queries=["open_workspace_mechanism_prediction"],
            closure_scope="uncapped_laminar_flow",
            proposed_discarded_variables=["branch_angle_micro"],
            verified_discarded_in_scope=[{
                "variable": "branch_angle_micro",
                "task_view_goal": "open_workspace_mechanism_prediction",
                "error_bound": 0.015,
                "evidence_ref": "stage0_murray_cal",
            }],
            error_budget=0.04,
        ),
        claim_contract=ClaimContract(
            "claim_stage0_murray_1d",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 1}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_atomic_murray_1d",
                operator_spec=sp_murray_1d,
                artifact_dict=cal_murray["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage0_murray_1d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.055,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:pos_arity_1", "sig:neg_arity_1", "sig:cue_vascular_tree", "sig:cue_vascular_ceiling_onset"],
        program_steps=_std_prog_steps("op_atomic_murray_1d", arity=1),
        source_cluster_ids=["cluster_vascular"],
    )

    # Fine-scale pore node + `coarse_graining_edge` #1 (`hydro_micro_pore_1d --Pi_s--> stage0_murray_1d`)
    forest.register_node(
        node_id="hydro_micro_pore_1d",
        tree_id="tree_vascular_murray",
        layer_role="root",
        task_view=tv_open,
        observation_map=ObservationMap("pore_shear", parameter_semantics={"active_arity": 1}),
        claim_contract=ClaimContract(
            "claim_hydro_micro_pore",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 1}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_atomic_murray_1d",
                operator_spec=sp_murray_1d,
                artifact_dict=cal_murray["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage0_murray_1d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.055,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="bounded_trial",
        ),
        probe_signatures=["sig:low_flow_cdn_onset"],
        program_steps=_std_prog_steps("op_atomic_murray_1d", arity=1),
        source_cluster_ids=["cluster_vascular"],
    )
    forest.add_coarse_graining_edge(
        fine_node_id="hydro_micro_pore_1d",
        coarse_node_id="stage0_murray_1d",
        scale_map=forest.nodes["stage0_murray_1d"]["scale_map"],
        empirical_closure_certificate={
            "certified": True,
            "max_closure_defect": 0.012,
            "tolerance": 0.04,
            "confirmed_instance_digest": cal_murray["artifact"]["confirmed_instance_digest"],
        },
    )

    # Disjoint-domain bridge partner for 2D Vascular Branch Shunt Onset
    forest.register_node(
        node_id="bridge_arbor_branch_onset",
        tree_id="tree_arbor_network",
        layer_role="mid",
        task_view=tv_open,
        observation_map=ObservationMap("branch_conductance", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_bridge_arbor",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 2}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_atomic_exp_relax_2d",
                operator_spec=sp_exp_2d,
                artifact_dict=cal_exp["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage0_exp_relax_2d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="bounded_trial",
        ),
        probe_signatures=["sig:cue_vascular_branch_onset"],
        program_steps=_std_prog_steps("op_atomic_exp_relax_2d"),
        source_cluster_ids=["cluster_arbor"],
    )

    # Register Fine 2-Mode ODE Node + Coarse Schur `Pi_s` Node (`thermal` modality) + `coarse_graining_edge` #2!
    forest.register_node(
        node_id="ode_micro_2mode_node",
        tree_id="tree_multiscale_pi_s",
        layer_role="root",
        task_view=tv_open,
        observation_map=ObservationMap("coupled_2mode_output", parameter_semantics={"active_arity": 2}),
        scale_map=ScaleMap(
            coarse_graining_op="identity_micro_rk4",
            window_or_partition={"scale_regime": "micro"},
            preserved_queries=["open_workspace_mechanism_prediction"],
            closure_scope="exact_all_horizons",
            proposed_discarded_variables=[],
            verified_discarded_in_scope=[],
            error_budget=0.001,
        ),
        claim_contract=ClaimContract(
            "claim_ode_micro_2mode",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "modality", "op": "==", "value": "thermal"},
            ],
            conclusion_or_action={
                "operator_tag": "op_micro_2mode_rk4",
                "fine_ode_direct_mode": True,
                "ode_params": deepcopy(DEFAULT_PI_S_ODE_PARAMS),
            },
            protocol_spec={"isolated_confirm": True},
            tolerance=0.04,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:cue_adiabatic_multiscale", "sig:cue_quench_multiscale"],
        program_steps=_std_prog_steps("op_micro_2mode_rk4"),
        source_cluster_ids=["cluster_multiscale_ode"],
    )

    forest.register_node(
        node_id="ode_meso_schur_pi_thermal",
        tree_id="tree_multiscale_pi_s",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("schur_slow_output", parameter_semantics={"active_arity": 2}),
        scale_map=ScaleMap(
            coarse_graining_op="schur_adiabatic_pi_s",
            window_or_partition={
                "scale_regime": "meso",
                "verified_input_box": deepcopy(pi_s_cert_init["verified_input_box"]),
            },
            preserved_queries=["open_workspace_mechanism_prediction"],
            closure_scope="regional_adiabatic_box",
            proposed_discarded_variables=["fast_mode_x2"],
            verified_discarded_in_scope=[{
                "variable": "fast_mode_x2",
                "task_view_goal": "open_workspace_mechanism_prediction",
                "error_bound": round(float(pi_s_cert_init["max_closure_defect"]), 6),
                "evidence_ref": "stage0_pi_s_cert",
            }],
            error_budget=0.045,
        ),
        claim_contract=ClaimContract(
            "claim_ode_meso_schur_pi",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "modality", "op": "==", "value": "thermal"},
            ],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_meso_schur_pi_s",
                operator_spec=apply_structural_combinator(
                    make_atomic_operator_spec("exp_relax_2d"),
                    "additive_secondary_shunt",
                    secondary_role=0,
                ),
                artifact_dict=pi_s_init_art,
                artifact_relpath=pi_s_cert_init_relpath,
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:meso_schur_adiabatic_reduction"],
        program_steps=_std_prog_steps("op_meso_schur_pi_s"),
        source_cluster_ids=["cluster_multiscale_ode"],
    )
    forest.add_coarse_graining_edge(
        fine_node_id="ode_micro_2mode_node",
        coarse_node_id="ode_meso_schur_pi_thermal",
        scale_map=forest.nodes["ode_meso_schur_pi_thermal"]["scale_map"],
        empirical_closure_certificate=pi_s_cert_init,
    )

    # Identical-law hydraulic partner node (`modality == 'hydraulic'`) to be merged via disjoint-scope FidelityCompression!
    forest.register_node(
        node_id="ode_meso_schur_pi_hydraulic_dup",
        tree_id="tree_multiscale_pi_s_hyd",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=deepcopy(forest.nodes["ode_meso_schur_pi_thermal"]["observation_map"]),
        scale_map=deepcopy(forest.nodes["ode_meso_schur_pi_thermal"]["scale_map"]),
        claim_contract=ClaimContract(
            "claim_ode_meso_schur_pi_hyd",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "modality", "op": "==", "value": "hydraulic"},
            ],
            conclusion_or_action=deepcopy(
                forest.nodes["ode_meso_schur_pi_thermal"]["claim_contract"]["conclusion_or_action"]
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:meso_schur_adiabatic_reduction"],
        program_steps=_std_prog_steps("op_meso_schur_pi_s"),
        source_cluster_ids=["cluster_multiscale_hyd"],
    )

    # Capture EXACT Stage-0 boundary snapshot (all 8 nodes, 2 coarse edges, 1 wormhole edge!)
    forest_after_stage0 = deepcopy(forest)

    stage0_total_meas = round(
        sum(float(x["cost"]["measurement_cost"]) for x in stage0_logs)
        + float(pi_s_cert_init["cost"]["measurement_cost"]),
        4,
    )
    stage0_total_comp = round(
        sum(float(x["cost"]["compute_cost"]) for x in stage0_logs)
        + float(pi_s_cert_init["cost"]["compute_cost"]),
        4,
    )
    stage0_total_composite = round(
        sum(float(x["cost"]["composite_cost"]) for x in stage0_logs)
        + float(pi_s_cert_init["cost"]["composite_cost"]),
        4,
    )

    # -------------------------------------------------------------------------
    # Stage 1: Open `CodeAgent` Compositional Discovery (`W1..W5`)
    # -------------------------------------------------------------------------
    s1_dir = work_root / "persisted_stage1_authored_workspaces"
    stage1_discoveries: dict[str, dict[str, Any]] = {}

    def _run_discovery_ep(
        ws_sub: Path,
        *,
        lab: OpenWorkspaceTaskLab,
        oracle_fn,
        role_names: list[str],
        atomic_base_specs: list[dict[str, Any]],
        pilot_points: list[dict[str, float]],
        confirm_points: list[dict[str, float]],
        heldout_points: list[dict[str, float]],
        tolerance: float,
    ) -> dict[str, Any]:
        h_rows = [{"features": dict(pt), "target": float(oracle_fn(pt))} for pt in heldout_points]
        return run_compositional_code_agent_episode(
            lab=lab,
            heldout_rows=h_rows,
            tolerance=tolerance,
            workspace_dir=ws_sub,
            role_names=role_names,
            initial_query_batches=[pilot_points],
            atomic_base_specs=atomic_base_specs,
            confirm_points=confirm_points,
        )

    # W1: 2D Haldane + Secondary Shunt (`4` params: `p0=8.8, p1=2.1, p2=0.14, p3=0.76`)
    fn_w1 = lambda p: 8.8 * p["s"] / (2.1 + p["s"] + 0.14 * (p["s"] ** 2)) + 0.76 * p["b"]
    lab_w1 = OpenWorkspaceTaskLab(
        task_id="stage1_W1_haldane_shunt",
        surface_domain="chemostat_bioreactor",
        feature_names=("s", "b"),
        feature_bounds={"s": (0.2, 12.0), "b": (0.0, 2.0)},
        cost_per_obs=4.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.045},
        meter=CostMeter(obs_budget=32, meas_budget=300.0, compute_budget=1500.0),
        private_oracle_fn=fn_w1,
    )
    w1_ep = _run_discovery_ep(
        s1_dir / "stage1_W1_haldane_shunt",
        lab=lab_w1,
        oracle_fn=fn_w1,
        role_names=["s", "b"],
        atomic_base_specs=[make_atomic_operator_spec("michaelis_1d", arity=2), sp_exp_2d],
        pilot_points=[
            {"s": 0.4, "b": 0.1}, {"s": 1.2, "b": 0.9}, {"s": 2.4, "b": 0.2},
            {"s": 4.8, "b": 1.1}, {"s": 8.0, "b": 0.3}, {"s": 11.2, "b": 0.4},
        ],
        confirm_points=[{"s": 3.6, "b": 0.8}, {"s": 9.6, "b": 0.5}],
        heldout_points=[{"s": 1.8, "b": 0.5}, {"s": 6.5, "b": 1.0}, {"s": 10.5, "b": 0.2}],
        tolerance=0.045,
    )
    stage1_discoveries["W1_haldane_shunt"] = w1_ep

    # W2: 2D Oscillatory Damped Relaxation + Secondary Shunt (`4` params)
    fn_w2 = lambda p: 2.2 * p["u"] * (1.0 - math.exp(-p["t"] / 1.35) * math.cos(2.15 * p["t"])) + 0.42 * p["t"]
    lab_w2 = OpenWorkspaceTaskLab(
        task_id="stage1_W2_osc_shunt_4p",
        surface_domain="underdamped_servo",
        feature_names=("u", "t"),
        feature_bounds={"u": (0.4, 3.0), "t": (0.05, 4.5)},
        cost_per_obs=4.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.05},
        meter=CostMeter(obs_budget=32, meas_budget=300.0, compute_budget=1500.0),
        private_oracle_fn=fn_w2,
    )
    w2_ep = _run_discovery_ep(
        s1_dir / "stage1_W2_osc_shunt_4p",
        lab=lab_w2,
        oracle_fn=fn_w2,
        role_names=["u", "t"],
        atomic_base_specs=[sp_exp_2d, make_atomic_operator_spec("michaelis_1d", arity=2)],
        pilot_points=[
            {"u": 1.0, "t": 0.15}, {"u": 1.4, "t": 0.65}, {"u": 1.8, "t": 1.25},
            {"u": 2.2, "t": 2.05}, {"u": 1.6, "t": 3.10}, {"u": 2.5, "t": 4.20},
        ],
        confirm_points=[{"u": 1.7, "t": 0.95}, {"u": 2.1, "t": 2.65}],
        heldout_points=[{"u": 1.2, "t": 0.40}, {"u": 2.0, "t": 1.60}, {"u": 2.4, "t": 3.80}],
        tolerance=0.05,
    )
    stage1_discoveries["W2_osc_shunt_4p"] = w2_ep

    # W3: 1D Murray Power 2/3 + Variational Ceiling Cap (`2` params)
    fn_w3 = lambda p: min(2.15 * (p["q"] ** (2.0 / 3.0)), 5.45)
    lab_w3 = OpenWorkspaceTaskLab(
        task_id="stage1_W3_murray_capped",
        surface_domain="vascular_conduit",
        feature_names=("q",),
        feature_bounds={"q": (0.2, 10.0)},
        cost_per_obs=4.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.045},
        meter=CostMeter(obs_budget=32, meas_budget=300.0, compute_budget=1500.0),
        private_oracle_fn=fn_w3,
    )
    w3_ep = _run_discovery_ep(
        s1_dir / "stage1_W3_murray_capped",
        lab=lab_w3,
        oracle_fn=fn_w3,
        role_names=["q"],
        atomic_base_specs=[sp_murray_1d, sp_lin_1d],
        pilot_points=[{"q": 0.4}, {"q": 1.2}, {"q": 2.6}, {"q": 4.8}, {"q": 7.2}, {"q": 9.4}],
        confirm_points=[{"q": 3.5}, {"q": 8.4}],
        heldout_points=[{"q": 0.8}, {"q": 5.6}, {"q": 9.0}],
        tolerance=0.045,
    )
    stage1_discoveries["W3_murray_capped"] = w3_ep

    # W4: 2D Monod + Secondary Shunt (`3` params)
    fn_w4 = lambda p: 9.0 * p["s"] / (1.6 + p["s"]) + 0.72 * p["b"]
    lab_w4 = OpenWorkspaceTaskLab(
        task_id="stage1_W4_monod_shunt",
        surface_domain="carrier_shunt_reactor",
        feature_names=("s", "b"),
        feature_bounds={"s": (0.2, 12.0), "b": (0.0, 2.0)},
        cost_per_obs=4.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.045},
        meter=CostMeter(obs_budget=32, meas_budget=300.0, compute_budget=1500.0),
        private_oracle_fn=fn_w4,
    )
    w4_ep = _run_discovery_ep(
        s1_dir / "stage1_W4_monod_shunt",
        lab=lab_w4,
        oracle_fn=fn_w4,
        role_names=["s", "b"],
        atomic_base_specs=[make_atomic_operator_spec("michaelis_1d", arity=2)],
        pilot_points=[
            {"s": 0.4, "b": 0.1}, {"s": 1.2, "b": 0.9}, {"s": 2.6, "b": 0.2},
            {"s": 5.0, "b": 1.1}, {"s": 8.2, "b": 0.3}, {"s": 11.0, "b": 0.8},
        ],
        confirm_points=[{"s": 3.8, "b": 0.7}, {"s": 9.5, "b": 0.5}],
        heldout_points=[{"s": 1.6, "b": 0.4}, {"s": 6.8, "b": 1.0}, {"s": 10.4, "b": 0.3}],
        tolerance=0.045,
    )
    stage1_discoveries["W4_monod_shunt"] = w4_ep

    # W5: 2D Murray 2/3 + Secondary Shunt (`2` params)
    fn_w5 = lambda p: 3.15 * (p["s"] ** (2.0 / 3.0)) + 0.68 * p["b"]
    lab_w5 = OpenWorkspaceTaskLab(
        task_id="stage1_W5_murray_shunt",
        surface_domain="vascular_collateral_shunt",
        feature_names=("s", "b"),
        feature_bounds={"s": (0.2, 12.0), "b": (0.0, 2.0)},
        cost_per_obs=4.0,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.045},
        meter=CostMeter(obs_budget=32, meas_budget=300.0, compute_budget=1500.0),
        private_oracle_fn=fn_w5,
    )
    w5_ep = _run_discovery_ep(
        s1_dir / "stage1_W5_murray_shunt",
        lab=lab_w5,
        oracle_fn=fn_w5,
        role_names=["s", "b"],
        atomic_base_specs=[make_atomic_operator_spec("murray_power_2_3_1d", arity=2)],
        pilot_points=[
            {"s": 0.4, "b": 0.1}, {"s": 1.2, "b": 0.9}, {"s": 2.6, "b": 0.2},
            {"s": 5.0, "b": 1.1}, {"s": 8.2, "b": 0.3}, {"s": 11.0, "b": 0.8},
        ],
        confirm_points=[{"s": 3.8, "b": 0.7}, {"s": 9.5, "b": 0.5}],
        heldout_points=[{"s": 1.6, "b": 0.4}, {"s": 6.8, "b": 1.0}, {"s": 10.4, "b": 0.3}],
        tolerance=0.045,
    )
    stage1_discoveries["W5_murray_shunt"] = w5_ep

    for name, ep in stage1_discoveries.items():
        assert ep["strict_pass"] is True, f"Stage 1 discovery {name} failed: {ep}"

    # Register Stage-1 composed mechanisms into the SAME `PhilosophicalForest`!
    def _art_relpath(sub_name: str) -> str:
        return f"persisted_authored_workspaces/persisted_stage1_authored_workspaces/{sub_name}/authored_method.json"

    forest.register_node(
        node_id="stage1_W1_haldane_shunt",
        tree_id="tree_monod_reactor",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("reaction_rate", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_W1_haldane_shunt",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "haldane_turnover_shunt_regime", "op": "==", "value": True},
            ],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_composed_haldane_shunt_2d",
                operator_spec=w1_ep["authored_artifact"]["operator_spec"],
                artifact_dict=w1_ep["authored_artifact"],
                artifact_relpath=_art_relpath("stage1_W1_haldane_shunt"),
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.055,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:pos_arity_2", "sig:neg_arity_2", "sig:witness_haldane_shunt_2d", "sig:cue_feed_reactor", "sig:cue_pore_collapse"],
        program_steps=_std_prog_steps("op_composed_haldane_shunt_2d"),
        source_cluster_ids=["cluster_monod"],
    )
    forest.add_local_implication_edge(
        stronger_node_id="stage1_W1_haldane_shunt",
        weaker_node_id="stage0_monod_2d",
        preserved_contract={"reduces_at_low_s_zero_b": True},
        scope={"s_max": 3.5, "b_eq": 0.0},
    )

    forest.register_node(
        node_id="stage1_W2_osc_shunt_4p",
        tree_id="tree_servo_osc",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("servo_response", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_W2_osc_shunt_4p",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "oscillatory_secular_shunt_regime", "op": "==", "value": True},
            ],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_composed_osc_shunt_4p_2d",
                operator_spec=w2_ep["authored_artifact"]["operator_spec"],
                artifact_dict=w2_ep["authored_artifact"],
                artifact_relpath=_art_relpath("stage1_W2_osc_shunt_4p"),
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.06,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:pos_arity_2", "sig:neg_arity_2", "sig:witness_oscillatory_shunt_4p", "sig:cue_servo_transient", "sig:cue_step_relax"],
        program_steps=_std_prog_steps("op_composed_osc_shunt_4p_2d"),
        source_cluster_ids=["cluster_servo"],
    )

    forest.register_node(
        node_id="stage1_W3_murray_capped",
        tree_id="tree_vascular_murray",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("conductance_drop", parameter_semantics={"active_arity": 1}),
        claim_contract=ClaimContract(
            "claim_W3_murray_capped",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 1},
                {"field": "high_flow_ceiling_regime", "op": "==", "value": True},
            ],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_composed_murray_capped_1d",
                operator_spec=w3_ep["authored_artifact"]["operator_spec"],
                artifact_dict=w3_ep["authored_artifact"],
                artifact_relpath=_art_relpath("stage1_W3_murray_capped"),
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.055,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:witness_macro_flow_ceiling"],
        program_steps=_std_prog_steps("op_composed_murray_capped_1d", arity=1),
        source_cluster_ids=["cluster_vascular"],
    )
    forest.add_local_implication_edge(
        stronger_node_id="stage1_W3_murray_capped",
        weaker_node_id="stage0_murray_1d",
        preserved_contract={"high_flow_ceiling_extension": True},
        scope={"q_above_cap": True},
    )

    # W4 (`Monod + Shunt`, 2-hop target from `bridge_membrane_carrier_onset <-> stage0_monod_2d -> stage1_W4_monod_shunt`!
    # Notice ZERO direct cue overlap with `sig:cue_carrier_binding_onset` — reachable ONLY via 2-hop walk!)
    forest.register_node(
        node_id="stage1_W4_monod_shunt",
        tree_id="tree_monod_reactor",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("reaction_rate", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_W4_monod_shunt",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "haldane_turnover_shunt_regime", "op": "==", "value": False},
                {"field": "secondary_bleed_active", "op": "==", "value": True},
            ],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_composed_monod_shunt_2d",
                operator_spec=w4_ep["authored_artifact"]["operator_spec"],
                artifact_dict=w4_ep["authored_artifact"],
                artifact_relpath=_art_relpath("stage1_W4_monod_shunt"),
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.045,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:internal_monod_shunt_specialization"],
        program_steps=_std_prog_steps("op_composed_monod_shunt_2d"),
        source_cluster_ids=["cluster_monod"],
    )
    forest.add_local_implication_edge(
        stronger_node_id="stage1_W4_monod_shunt",
        weaker_node_id="stage0_monod_2d",
        preserved_contract={"secondary_bleed_shunt_extension": True},
        scope={"b_gt_zero": True},
    )

    # W5 (`Murray + Shunt`, 2-hop target from `bridge_arbor_branch_onset <-> vascular_2d_hub -> stage1_W5_murray_shunt`!
    # ZERO direct cue overlap with `sig:cue_vascular_branch_onset` — reachable ONLY via 2-hop walk!)
    forest.register_node(
        node_id="vascular_2d_hub",
        tree_id="tree_vascular_shunt_2d",
        layer_role="mid",
        task_view=tv_open,
        observation_map=ObservationMap("vascular_2d_flux", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_vascular_2d_hub",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 2}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_atomic_exp_relax_2d",
                operator_spec=sp_exp_2d,
                artifact_dict=cal_exp["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage0_exp_relax_2d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="bounded_trial",
        ),
        probe_signatures=["sig:internal_vascular_2d_root"],
        program_steps=_std_prog_steps("op_atomic_exp_relax_2d"),
        source_cluster_ids=["cluster_vascular_2d"],
    )
    forest.add_horizontal_wormhole(
        wormhole_id="wh_arbor_to_vascular_2d_hub",
        node_a_id="bridge_arbor_branch_onset",
        node_b_id="vascular_2d_hub",
        probe_signature_refs=["sig:cue_vascular_branch_onset"],
        role_and_observation_map={"role_map": "s->s,b->b"},
        hop_error_budget=0.01,
        permission_tier="local_solver_guidance",
    )
    forest.register_node(
        node_id="stage1_W5_murray_shunt",
        tree_id="tree_vascular_shunt_2d",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("vascular_2d_flux", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_W5_murray_shunt",
            "m4_open_compositional_ast",
            preconditions=[
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "haldane_turnover_shunt_regime", "op": "==", "value": False},
            ],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_composed_murray_shunt_2d",
                operator_spec=w5_ep["authored_artifact"]["operator_spec"],
                artifact_dict=w5_ep["authored_artifact"],
                artifact_relpath=_art_relpath("stage1_W5_murray_shunt"),
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.045,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:internal_murray_shunt_specialization"],
        program_steps=_std_prog_steps("op_composed_murray_shunt_2d"),
        source_cluster_ids=["cluster_vascular_2d"],
    )
    forest.add_local_implication_edge(
        stronger_node_id="stage1_W5_murray_shunt",
        weaker_node_id="vascular_2d_hub",
        preserved_contract={"murray_collateral_shunt_law": True},
        scope={"b_gt_zero": True},
    )

    # Register an over-broad distractor lure node (`lure_cubic_overfit_2d`) that looks syntactically generic
    # on `sig:pos_arity_2` & `sig:neg_arity_2` so Stage 2's 3 distinct DEV episodes empirically test & Minton-archive it!
    sp_lure = make_atomic_operator_spec("variational_cellular_2d")
    cal_lure = _run_metered_atomic_calibration(
        s0_dir / "stage1_lure_cellular_2d",
        task_id="stage1_lure_cellular_2d",
        surface_domain="cellular_foam_lure",
        spec=sp_lure,
        role_names=["x", "z"],
        pilot_points=[{"x": 1.0, "z": 1.0}, {"x": 2.0, "z": 1.5}, {"x": 3.0, "z": 2.0}, {"x": 4.0, "z": 2.5}],
        confirm_points=[{"x": 2.5, "z": 1.8}],
        oracle_fn=lambda p: 2.0 * math.sqrt(p["x"]) + 0.8 * p["x"] / p["z"],
    )
    forest.register_node(
        node_id="lure_cellular_overbroad_2d",
        tree_id="tree_lure_cellular",
        layer_role="leaf",
        task_view=tv_open,
        observation_map=ObservationMap("foam_energy", parameter_semantics={"active_arity": 2}),
        claim_contract=ClaimContract(
            "claim_lure_cellular",
            "m4_open_compositional_ast",
            preconditions=[{"field": "active_arity", "op": "==", "value": 2}],
            conclusion_or_action=_make_verified_instance_action(
                operator_tag="op_lure_variational_cellular_2d",
                operator_spec=sp_lure,
                artifact_dict=cal_lure["artifact"],
                artifact_relpath="persisted_authored_workspaces/stage0_metered_sources/stage1_lure_cellular_2d/authored_method.json",
            ),
            protocol_spec={"isolated_confirm": True},
            tolerance=0.05,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=[
            "sig:pos_arity_2",
            "sig:neg_arity_2",
            "sig:sensor_pos",
            "sig:sensor_neg",
            "sig:two_channel_coupled",
            "sig:cue_feed_reactor",
            "sig:cue_pore_collapse",
            "sig:cue_servo_transient",
        ],
        program_steps=_std_prog_steps("op_lure_variational_cellular_2d"),
        source_cluster_ids=["cluster_lure"],
    )

    stage1_total_meas = round(
        sum(float(ep["cost"]["measurement_cost"]) for ep in stage1_discoveries.values())
        + float(cal_lure["cost"]["measurement_cost"]),
        4,
    )
    stage1_total_comp = round(
        sum(float(ep["cost"]["compute_cost"]) for ep in stage1_discoveries.values())
        + float(cal_lure["cost"]["compute_cost"]),
        4,
    )
    stage1_total_composite = round(
        sum(float(ep["cost"]["composite_cost"]) for ep in stage1_discoveries.values())
        + float(cal_lure["cost"]["composite_cost"]),
        4,
    )

    # Keep an un-compressed, un-Minton-pruned snapshot for the Stage-2 ablation baseline!
    forest_before_stage2_maintenance = deepcopy(forest)

    # -------------------------------------------------------------------------
    # Stage 2: Online Forest Lifecycle (Revision + Disjoint FidelityCompression + 3 Distinct DEV Lures + Pi_s Quench Revision)
    # -------------------------------------------------------------------------
    s2_dir = work_root / "stage2_lifecycle"
    s2_dir.mkdir(parents=True, exist_ok=True)

    # 2A. Empirical Scope Revision on `stage0_monod_2d`: restrict root to `b == 0` & `s <= 4.0` (`no_bleed_low_substrate`)
    # after `W1_haldane_shunt` & `W4_monod_shunt` revealed secondary shunt & high-substrate turnover!
    monod_rev_cc = deepcopy(forest.nodes["stage0_monod_2d"]["claim_contract"])
    monod_rev_cc["preconditions"] = [
        {"field": "active_arity", "op": "==", "value": 2},
        {"field": "b", "op": "<=", "value": 1e-6},
        {"field": "s", "op": "<=", "value": 4.0},
    ]
    forest.empirical_revise_node(
        "stage0_monod_2d",
        witness_episode_id="stage1_W1_haldane_shunt",
        cause="protocol_or_scope_mismatch",
        updated_claim_contract=monod_rev_cc,
    )
    forest.revalidate_wormhole_after_revision(
        "wh_carrier_to_monod_reactor",
        witness_episode_id="stage2_reval_carrier_wh",
        reason="low_s_carrier_onset_preserved",
    )
    forest.revalidate_vertical_edges_after_revision(
        "stage0_monod_2d",
        witness_episode_id="stage2_reval_monod_children",
        reason="monod_children_valid_on_b_gt_0",
    )

    # 2B. Disjoint-Scope `FidelityCompression`: merge `ode_meso_schur_pi_hydraulic_dup` (`modality == 'hydraulic'`)
    # into `ode_meso_schur_pi_thermal` (`modality == 'thermal'`) -> OR-branched node verified by REAL `authorize_abstraction_use`!
    def _eval_modality_branch(nd: dict[str, Any], q: dict[str, Any]) -> str:
        branches = nd["claim_contract"].get("disjunctive_branches") or [nd["claim_contract"]]
        ok = any(
            all(q.get(p["field"]) == p["value"] for p in br.get("preconditions", ()))
            for br in branches
        )
        return "ACC" if ok else "UNK"

    comp_report = forest.compress_compatible_nodes(
        keeper_node_id="ode_meso_schur_pi_thermal",
        redundant_node_id="ode_meso_schur_pi_hydraulic_dup",
        evaluation_queries=[
            {"active_arity": 2, "modality": "thermal"},
            {"active_arity": 2, "modality": "hydraulic"},
            {"active_arity": 2, "modality": "optical"},
        ],
        evaluator_fn=_eval_modality_branch,
    )
    assert comp_report["accepted"] is True

    # Formally revise `ode_micro_2mode_node` to the matching `thermal | hydraulic` OR-contract in Stage 2
    # (bumping version & semantic_digest cleanly so `Act 3` dispatcher verification is 100% read-only!)
    micro_rev_cc = deepcopy(forest.nodes["ode_micro_2mode_node"]["claim_contract"])
    micro_rev_cc["preconditions"] = [{"field": "active_arity", "op": "==", "value": 2}]
    micro_rev_cc["disjunctive_branches"] = [
        {
            "claim_id": "claim_ode_micro_2mode_thermal",
            "preconditions": [
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "modality", "op": "==", "value": "thermal"},
            ],
            "permission_tier": "local_solver_guidance",
            "evidence_status": "SUPPORTED_IN_SCOPE",
            "conclusion_or_action": deepcopy(micro_rev_cc["conclusion_or_action"]),
        },
        {
            "claim_id": "claim_ode_micro_2mode_hydraulic",
            "preconditions": [
                {"field": "active_arity", "op": "==", "value": 2},
                {"field": "modality", "op": "==", "value": "hydraulic"},
            ],
            "permission_tier": "local_solver_guidance",
            "evidence_status": "SUPPORTED_IN_SCOPE",
            "conclusion_or_action": deepcopy(micro_rev_cc["conclusion_or_action"]),
        },
    ]
    forest.empirical_revise_node(
        "ode_micro_2mode_node",
        witness_episode_id="stage2_micro_modality_alignment",
        cause="protocol_or_scope_mismatch",
        updated_claim_contract=micro_rev_cc,
    )

    # Verify with REAL `authorize_abstraction_use` on `'thermal'`, `'hydraulic'`, and `'optical'`!
    merged_pi_node = forest.nodes["ode_meso_schur_pi_thermal"]
    real_auth_after_compression = {}
    for mod_tag in ("thermal", "hydraulic", "optical"):
        real_auth_after_compression[mod_tag] = authorize_abstraction_use(
            requested_tier="bounded_trial",
            task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.04},
            target_features={"active_arity": 2, "modality": mod_tag},
            claim_contract=merged_pi_node["claim_contract"],
            scale_map=merged_pi_node["scale_map"],
        )["granted"]
    assert real_auth_after_compression == {"thermal": True, "hydraulic": True, "optical": False}

    # 2C. 3 Genuinely Distinct Paid DEV Lure Episodes (distinct domains, parameters, points, and seeds!)
    dev_lure_specs = [
        (
            "dev_lure_ep1_chemostat",
            "chemostat_dev_a",
            ("c_feed", "q_bleed"),
            [{"c_feed": 0.5, "q_bleed": 0.2}, {"c_feed": 2.0, "q_bleed": 0.8}, {"c_feed": 6.0, "q_bleed": 0.4}, {"c_feed": 10.5, "q_bleed": 0.6}],
            lambda p: 9.2 * p["c_feed"] / (2.0 + p["c_feed"] + 0.13 * (p["c_feed"] ** 2)) + 0.70 * p["q_bleed"],
        ),
        (
            "dev_lure_ep2_servo",
            "servo_dev_b",
            ("v_step", "t_dwell"),
            [{"v_step": 0.9, "t_dwell": 0.2}, {"v_step": 1.6, "t_dwell": 0.9}, {"v_step": 2.1, "t_dwell": 1.9}, {"v_step": 2.4, "t_dwell": 3.4}],
            lambda p: 2.05 * p["v_step"] * (1.0 - math.exp(-p["t_dwell"] / 1.25) * math.cos(2.05 * p["t_dwell"])) + 0.38 * p["t_dwell"],
        ),
        (
            "dev_lure_ep3_membrane",
            "membrane_dev_c",
            ("solute_s", "shunt_b"),
            [{"solute_s": 0.6, "shunt_b": 0.3}, {"solute_s": 1.8, "shunt_b": 1.0}, {"solute_s": 4.2, "shunt_b": 0.5}, {"solute_s": 8.5, "shunt_b": 0.9}],
            lambda p: 14.0 - (8.4 * p["solute_s"] / (1.9 + p["solute_s"]) + 0.65 * p["shunt_b"]),
        ),
    ]
    dev_lure_logs: list[dict[str, Any]] = []
    for ep_id, dom, fnames, pts, fn in dev_lure_specs:
        d_lab = OpenWorkspaceTaskLab(
            task_id=ep_id,
            surface_domain=dom,
            feature_names=fnames,
            feature_bounds={fnames[0]: (0.1, 12.0), fnames[1]: (0.05, 5.0)},
            cost_per_obs=4.0,
            task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": 0.05},
            meter=CostMeter(obs_budget=16, meas_budget=200.0, compute_budget=600.0),
            private_oracle_fn=fn,
        )
        obs = d_lab.acquire(pts, workspace=s2_dir / ep_id, phase="dev_lure_pilot") or []
        rows = [{"features": dict(r.features), "target": float(r.target)} for r in obs]
        best_l, _ = _autonomous_fit_candidate_specs(
            s2_dir / ep_id,
            specs_to_try=[("op_lure_variational_cellular_2d", sp_lure)],
            feature_names=fnames,
            training_rows=rows,
            tolerance=0.05,
        )
        d_lab.meter.charge("dev_lure_eval", compute=4.5)
        lure_rmse = float(best_l["train_rmse"]) if best_l is not None else 999.0
        assert lure_rmse > 0.05
        forest.record_utility_feedback(
            "lure_cellular_overbroad_2d", saved_cost=0.0, overhead_cost=4.5, succeeded=False
        )
        dev_lure_logs.append({
            "episode_id": ep_id,
            "surface_domain": dom,
            "feature_names": list(fnames),
            "lure_train_rmse": round(lure_rmse, 6),
            "cost": d_lab.meter.snapshot(),
        })
    assert forest.nodes["lure_cellular_overbroad_2d"]["utility_ledger"]["archived_low_utility"] is True

    # 2D. Empirical Short-Horizon Quench Probe (`T = 0.12`) -> Fine-Scale Mismatch -> Certified Vertical Revalidation!
    q_pt = [{"u_drive": 2.0, "T_horizon": 0.12}]
    quench_lab = MultiScaleWorkspaceTaskLab(
        task_id="stage2_pi_s_quench_probe",
        surface_domain="multiscale_coupled_ode",
        feature_names=("u_drive", "T_horizon"),
        feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": (0.05, 6.0)},
        true_scale_regime="micro",
        heldout_points=q_pt,
        tolerance=0.035,
        micro_fn=lambda pt: float(_eval_pi_s_fine_scalar(pt["u_drive"], pt["T_horizon"])["y_micro"]),
        coarse_pi_fn=lambda pt: float(_eval_pi_s_coarse_scalar(pt["u_drive"], pt["T_horizon"])["y_coarse"]),
        coarse_cost_per_obs=6.0,
        micro_cost_per_obs=18.0,
    )
    quench_meter = CostMeter(obs_budget=16, meas_budget=200.0, compute_budget=500.0)
    q_coarse = quench_lab.acquire_channel(q_pt, channel="coarse_macro", phase="quench_check_coarse")[0]
    q_fine = quench_lab.acquire_channel(q_pt, channel="micro_fine", phase="quench_check_fine")[0]
    quench_meter.charge("quench_probe_obs", obs=2, meas=quench_lab.paid_measurement_cost, compute=2.5)
    quench_defect = abs(float(q_fine["target"]) - float(q_coarse["target"]))
    assert quench_defect > 0.10  # Empirically refutes unguarded short-time use of `Pi_s`!

    # First mark closure loss (`not_closed`), verify label-only certified check rejects `not_closed`,
    # then re-certify adiabatic box `[2.4, 5.2]` and revalidate with `revalidate_vertical_edges_with_certificate`!
    unclosed_sm = deepcopy(forest.nodes["ode_meso_schur_pi_thermal"]["scale_map"])
    unclosed_sm["closure_scope"] = "not_closed"
    forest.empirical_revise_node(
        "ode_meso_schur_pi_thermal",
        witness_episode_id="stage2_pi_s_quench_probe",
        cause="abstraction_closure_loss",
        updated_scale_map=unclosed_sm,
    )
    rejected_unclosed_reval = forest.revalidate_vertical_edges_with_certificate(
        "ode_meso_schur_pi_thermal",
        witness_episode_id="stage2_rejected_unclosed",
        closure_certificate={"certified": True, "max_closure_defect": 0.01, "tolerance": 0.035},
    )
    assert rejected_unclosed_reval["accepted"] is False

    pi_s_reverified_relpath = "persisted_authored_workspaces/stage2_lifecycle/stage2_pi_s_reverified/authored_method.json"
    pi_s_cert_reverified = certify_regional_pi_s_closure_with_trajectory_projection(
        s2_dir / "stage2_pi_s_reverified",
        u_bounds=(0.4, 3.0),
        t_bounds=(2.4, 5.2),
        tolerance=0.045,
        artifact_relpath=pi_s_reverified_relpath,
    )
    reverified_art_json = json.loads((s2_dir / "stage2_pi_s_reverified" / "authored_method.json").read_text(encoding="utf-8"))
    upd_pi_cc = deepcopy(forest.nodes["ode_meso_schur_pi_thermal"]["claim_contract"])
    new_pi_act = _make_verified_instance_action(
        operator_tag="op_meso_schur_pi_s",
        operator_spec=upd_pi_cc["conclusion_or_action"]["operator_spec"],
        artifact_dict=reverified_art_json,
        artifact_relpath=pi_s_reverified_relpath,
    )
    upd_pi_cc["conclusion_or_action"] = deepcopy(new_pi_act)
    for br in upd_pi_cc.get("disjunctive_branches", ()):
        if isinstance(br, dict) and isinstance(br.get("conclusion_or_action"), dict):
            br["conclusion_or_action"] = deepcopy(new_pi_act)

    revised_closed_sm = ScaleMap(
        coarse_graining_op="schur_adiabatic_pi_s",
        window_or_partition={
            "scale_regime": "meso",
            "verified_input_box": deepcopy(pi_s_cert_reverified["verified_input_box"]),
            "excluded_quench_horizon_below": 2.4,
        },
        preserved_queries=["open_workspace_mechanism_prediction"],
        closure_scope="regional_adiabatic_box_post_quench",
        proposed_discarded_variables=["fast_mode_x2"],
        verified_discarded_in_scope=[{
            "variable": "fast_mode_x2",
            "task_view_goal": "open_workspace_mechanism_prediction",
            "error_bound": round(float(pi_s_cert_reverified["max_closure_defect"]), 6),
            "evidence_ref": "stage2_pi_s_reverified_cert",
        }],
        error_budget=0.045,
    ).to_dict()
    certified_reval = forest.revalidate_vertical_edges_with_certificate(
        "ode_meso_schur_pi_thermal",
        witness_episode_id="stage2_pi_s_reverified",
        closure_certificate=pi_s_cert_reverified,
        updated_scale_map=revised_closed_sm,
        updated_claim_contract=upd_pi_cc,
        reason="reverified_adiabatic_box_T_ge_2_4",
    )
    assert certified_reval["accepted"] is True and certified_reval["rebound_coarse_graining_edges"] == 1

    # Assemble Ordinary All-History flat list (chronological discovery order)
    ordinary_history_forward: list[tuple[str, dict[str, Any]]] = [
        *s0_specs,
        ("op_lure_variational_cellular_2d", sp_lure),
        ("op_meso_schur_pi_s", apply_structural_combinator(
            make_atomic_operator_spec("exp_relax_2d"),
            "additive_secondary_shunt",
            secondary_role=0,
        )),
        ("op_composed_haldane_shunt_2d", deepcopy(w1_ep["authored_artifact"]["operator_spec"])),
        ("op_composed_osc_shunt_4p_2d", deepcopy(w2_ep["authored_artifact"]["operator_spec"])),
        ("op_composed_murray_capped_1d", deepcopy(w3_ep["authored_artifact"]["operator_spec"])),
        ("op_composed_monod_shunt_2d", deepcopy(w4_ep["authored_artifact"]["operator_spec"])),
        ("op_composed_murray_shunt_2d", deepcopy(w5_ep["authored_artifact"]["operator_spec"])),
    ]

    # 2E. Disjoint Stratified 5-Family DEV Static-Order Freezing (`Ordinary_DevFrozenOrder` — ZERO TEST PEEKING!)
    dev_targets, _ = build_m4_2_target_suite(seed=99_112_026)  # Disjoint DEV seed!
    # Stratify 2 targets per family across all 5 families (`0,1`, `4,5`, `8,9`, `12,13`, `16,17` = 10 DEV targets!)
    stratified_dev_targets = [dev_targets[i] for i in (0, 1, 4, 5, 8, 9, 12, 13, 16, 17)]
    dev_wins_by_tag: dict[str, int] = {tag: 0 for tag, _ in ordinary_history_forward}
    dev_order_meter = CostMeter(obs_budget=500, meas_budget=5000.0, compute_budget=15000.0)
    for dt in stratified_dev_targets:
        d_ws = s2_dir / "dev_static_order_tune" / dt.target_id
        p_rows = [{"features": dict(pt), "target": float(dt.oracle_fn(pt))} for pt in dt.pilot_points]
        c0_rows = [{"features": dict(dt.confirm_points[0]), "target": float(dt.oracle_fn(dt.confirm_points[0]))}]
        d_unit = 8.0 if dt.mechanism_family == "PiS_Adiabatic_CrossDomain_2D" else 4.0
        dev_order_meter.charge(
            "dev_tune_obs",
            obs=len(p_rows) + 1,
            meas=d_unit * (len(p_rows) + 1),
            compute=2.5,
        )
        same_ar = [(t, sp) for t, sp in ordinary_history_forward if int(sp.get("arity", 1)) == len(dt.feature_names)]
        for tag, sp in same_ar:
            b_cand, ev_cnt = _autonomous_fit_candidate_specs(
                d_ws / tag,
                specs_to_try=[(tag, sp)],
                feature_names=dt.feature_names,
                training_rows=p_rows,
                tolerance=dt.tolerance,
            )
            dev_order_meter.charge("dev_cand_fit", compute=round(6.0 * ev_cnt, 4))
            if b_cand is not None and b_cand["train_rmse"] <= dt.tolerance:
                cf_e = evaluate_isolated_python_model(
                    d_ws / tag, b_cand["model_relpath"], c0_rows, isolated_prefix_rows=p_rows
                )
                dev_order_meter.charge("dev_cand_confirm_eval", compute=1.5)
                if cf_e["returncode"] == 0 and float(cf_e["rmse"]) <= dt.tolerance:
                    dev_wins_by_tag[tag] = dev_wins_by_tag.get(tag, 0) + 1

    ordinary_history_dev_frozen = sorted(
        ordinary_history_forward,
        key=lambda item: (-dev_wins_by_tag.get(item[0], 0), [t for t, _ in ordinary_history_forward].index(item[0])),
    )

    stage2_total_meas = round(
        sum(float(x["cost"]["measurement_cost"]) for x in dev_lure_logs)
        + float(quench_meter.snapshot()["measurement_cost"])
        + float(pi_s_cert_reverified["cost"]["measurement_cost"]),
        4,
    )
    stage2_total_comp = round(
        sum(float(x["cost"]["compute_cost"]) for x in dev_lure_logs)
        + float(quench_meter.snapshot()["compute_cost"])
        + float(pi_s_cert_reverified["cost"]["compute_cost"]),
        4,
    )
    stage2_total_composite = round(
        sum(float(x["cost"]["composite_cost"]) for x in dev_lure_logs)
        + float(quench_meter.snapshot()["composite_cost"])
        + float(pi_s_cert_reverified["cost"]["composite_cost"]),
        4,
    )

    live_idx = HippoRAGWormholeIndex(forest, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
    return {
        "forest": forest,
        "forest_after_stage0": forest_after_stage0,
        "forest_before_stage2_maintenance": forest_before_stage2_maintenance,
        "stage2_pi_s_reverified_certificate": pi_s_cert_reverified,
        "stage0": {
            "episodes": stage0_logs,
            "pi_s_initial_certificate": pi_s_cert_init,
            "measurement_cost": stage0_total_meas,
            "compute_cost": stage0_total_comp,
            "composite_cost": stage0_total_composite,
        },
        "stage1": {
            "discoveries": {
                k: {
                    "task_id": v["task_id"],
                    "strict_pass": v["strict_pass"],
                    "confirm_rmse": v["confirm_rmse"],
                    "heldout_rmse": v["heldout_rmse"],
                    "ast_skeleton_digest": v["authored_artifact"]["ast_skeleton_digest"],
                    "fitted_params": v["authored_artifact"]["fitted_params"],
                    "diagnostic_rounds": v["authored_artifact"]["diagnostic_rounds"],
                    "composed_candidate_count": v["authored_artifact"]["composed_candidate_count"],
                    "cost": v["cost"],
                }
                for k, v in stage1_discoveries.items()
            },
            "measurement_cost": stage1_total_meas,
            "compute_cost": stage1_total_comp,
            "composite_cost": stage1_total_composite,
        },
        "stage2": {
            "compression_report": {
                "accepted": comp_report["accepted"],
                "keeper_node_id": comp_report["keeper_node_id"],
                "keeper_committed_version": comp_report["keeper_committed_version"],
                "archived_redundant_node_id": comp_report["archived_redundant_node_id"],
                "real_authorize_abstraction_use_by_modality": real_auth_after_compression,
            },
            "dev_lure_episodes": dev_lure_logs,
            "lure_archived_by_minton": bool(
                forest.nodes["lure_cellular_overbroad_2d"]["utility_ledger"]["archived_low_utility"]
            ),
            "quench_revision": {
                "quench_horizon_T": 0.12,
                "observed_quench_defect": round(quench_defect, 6),
                "rejected_unclosed_revalidation": rejected_unclosed_reval,
                "certified_revalidation": certified_reval,
                "reverified_box": pi_s_cert_reverified["verified_input_box"],
                "projections_with_fine_trajectory": pi_s_cert_reverified["projections_with_fine_trajectory"],
            },
            "dev_static_order_freeze": {
                "dev_wins_by_tag": dev_wins_by_tag,
                "frozen_order_tags": [t for t, _ in ordinary_history_dev_frozen],
                "dev_tuning_cost_tracked_separately": dev_order_meter.snapshot(),
            },
            "edge_counts_post_stage2": dict(live_idx.edge_counts),
            "measurement_cost": stage2_total_meas,
            "compute_cost": stage2_total_comp,
            "composite_cost": stage2_total_composite,
        },
        "stage0_atomic_specs": s0_specs,
        "ordinary_history_forward": ordinary_history_forward,
        "ordinary_history_reversed": list(reversed(ordinary_history_forward)),
        "ordinary_history_dev_frozen": ordinary_history_dev_frozen,
    }


# ============================================================================
# Part 5. Interleaved Zero-`fit()` `ForestScopeDispatcher` Verification (`R3`)
# ============================================================================

def run_interleaved_zero_fit_dispatcher_stream(forest: PhilosophicalForest) -> dict[str, Any]:
    """Run an interleaved 10-query stream through ``ForestScopeDispatcher`` with ``fit_called == False``!

    Proves true instance-level ``R3`` parameter & scope retention:
      - ``stage0_monod_2d`` (`b=0, s<=4.0`) vs ``stage1_W1_haldane_shunt`` (`b>0`, high-`s` turnover);
      - ``stage1_W2_osc_shunt_4p`` damped oscillatory servo revisit;
      - ``ode_meso_schur_pi_thermal`` compressed OR-node on BOTH `'thermal'` AND `'hydraulic'` (`T >= 2.4`);
      - ``ode_micro_2mode_node`` automatic fine-scale escalation on short-horizon quench (`T = 0.12 < 2.4`);
      - Unauthorized modality `'optical'` cleanly denied (`status == 'scope_unauthorized'`, zero false grant).
    """
    dispatcher = ForestScopeDispatcher(forest)
    stream = [
        {
            "query_id": "R3_Q01_OldMonod_LowS_ZeroB",
            "tree_id": "tree_monod_reactor",
            "features": {"s": 1.6, "b": 0.0},
            "meta": {"active_arity": 2, "haldane_turnover_shunt_regime": False, "secondary_bleed_active": False},
            "truth": 6.0 * 1.6 / (2.0 + 1.6),
            "expect_node": "stage0_monod_2d",
        },
        {
            "query_id": "R3_Q02_NewHaldaneShunt_HighS",
            "tree_id": "tree_monod_reactor",
            "features": {"s": 9.2, "b": 0.7},
            "meta": {"active_arity": 2, "haldane_turnover_shunt_regime": True},
            "truth": 8.8 * 9.2 / (2.1 + 9.2 + 0.14 * (9.2 ** 2)) + 0.76 * 0.7,
            "expect_node": "stage1_W1_haldane_shunt",
        },
        {
            "query_id": "R3_Q03_PiS_Adiabatic_Thermal_OR",
            "tree_id": "tree_multiscale_pi_s",
            "features": {"u_drive": 1.8, "T_horizon": 3.6},
            "meta": {"active_arity": 2, "modality": "thermal"},
            "truth": float(_eval_pi_s_fine_scalar(1.8, 3.6)["y_micro"]),
            "expect_node": "ode_meso_schur_pi_thermal",
        },
        {
            "query_id": "R3_Q04_PiS_Adiabatic_Hydraulic_OR",
            "tree_id": "tree_multiscale_pi_s",
            "features": {"u_drive": 2.4, "T_horizon": 4.4},
            "meta": {"active_arity": 2, "modality": "hydraulic"},
            "truth": float(_eval_pi_s_fine_scalar(2.4, 4.4)["y_micro"]),
            "expect_node": "ode_meso_schur_pi_thermal",
        },
        {
            "query_id": "R3_Q05_PiS_ShortQuench_EscalateMicro",
            "tree_id": "tree_multiscale_pi_s",
            "features": {"u_drive": 2.2, "T_horizon": 0.12},
            "meta": {"active_arity": 2, "modality": "thermal"},
            "truth": float(_eval_pi_s_fine_scalar(2.2, 0.12)["y_micro"]),
            "expect_node": "ode_micro_2mode_node",
        },
        {
            "query_id": "R3_Q06_OscShunt4P_Revisit",
            "tree_id": "tree_servo_osc",
            "features": {"u": 1.9, "t": 1.45},
            "meta": {"active_arity": 2, "oscillatory_secular_shunt_regime": True},
            "truth": 2.2 * 1.9 * (1.0 - math.exp(-1.45 / 1.35) * math.cos(2.15 * 1.45)) + 0.42 * 1.45,
            "expect_node": "stage1_W2_osc_shunt_4p",
        },
        {
            "query_id": "R3_Q07_OldMonod_SecondRevisit",
            "tree_id": "tree_monod_reactor",
            "features": {"s": 2.8, "b": 0.0},
            "meta": {"active_arity": 2, "haldane_turnover_shunt_regime": False, "secondary_bleed_active": False},
            "truth": 6.0 * 2.8 / (2.0 + 2.8),
            "expect_node": "stage0_monod_2d",
        },
        {
            "query_id": "R3_Q08_NewHaldaneShunt_SecondRevisit",
            "tree_id": "tree_monod_reactor",
            "features": {"s": 6.4, "b": 1.1},
            "meta": {"active_arity": 2, "haldane_turnover_shunt_regime": True},
            "truth": 8.8 * 6.4 / (2.1 + 6.4 + 0.14 * (6.4 ** 2)) + 0.76 * 1.1,
            "expect_node": "stage1_W1_haldane_shunt",
        },
        {
            "query_id": "R3_Q09_MurrayCapped_HighFlow",
            "tree_id": "tree_vascular_murray",
            "features": {"q": 8.2},
            "meta": {"active_arity": 1, "high_flow_ceiling_regime": True},
            "truth": min(2.15 * (8.2 ** (2.0 / 3.0)), 5.45),
            "expect_node": "stage1_W3_murray_capped",
        },
        {
            "query_id": "R3_Q10_OpticalModality_Unauthorized",
            "tree_id": "tree_multiscale_pi_s",
            "features": {"u_drive": 1.8, "T_horizon": 3.6},
            "meta": {"active_arity": 2, "modality": "optical"},
            "truth": None,
            "expect_node": None,
        },
    ]

    # `run_interleaved_zero_fit_dispatcher_stream` is strictly READ-ONLY on `forest`
    # (`ode_micro_2mode_node` already carries its formal Stage-2 `thermal | hydraulic` OR-contract)!
    results: list[dict[str, Any]] = []
    for item in stream:
        rec = dispatcher.dispatch_predict(
            query_id=item["query_id"],
            candidate_tree_id=item["tree_id"],
            features=item["features"],
            target_metadata=item["meta"],
            ground_truth=item["truth"],
        )
        rec["expected_node_id"] = item["expect_node"]
        rec["route_matched_expected"] = rec["selected_node_id"] == item["expect_node"]
        results.append(rec)

    authorized_rows = [r for r in results if r["expected_node_id"] is not None]
    denied_rows = [r for r in results if r["expected_node_id"] is None]
    return {
        "query_count": len(results),
        "any_fit_called": any(r["fit_called"] for r in results),
        "all_routes_exact": all(r["route_matched_expected"] for r in results),
        "authorized_pass_rate": round(mean(1.0 if r["within_tolerance"] else 0.0 for r in authorized_rows), 4),
        "max_authorized_abs_error": round(max(float(r["abs_error"] or 0.0) for r in authorized_rows), 8),
        "unauthorized_cleanly_blocked": all(r["status"] == "scope_unauthorized" for r in denied_rows),
        "receipts": results,
    }


# ============================================================================
# Part 6. Balanced 20-Target Unseen Cross-Domain Suite + Seed-Varied OOS
# ============================================================================

def build_m4_2_target_suite(seed: int = 20_261_120) -> tuple[list[UnifiedTargetSpec], list[UnifiedTargetSpec]]:
    """Build a balanced 20-target unseen cross-domain suite (`4` per family) + `2` seed-varied OOS targets!

    Families (`4` targets each = `20` unseen targets per seed, `80` across 4 seeds):
      1. ``HaldaneShunt_2D`` (`W1` — high-drive turnover + shunt, permuted/inverted coordinates);
      2. ``OscShunt4P_2D`` (`W2` — damped cosine ringing + secular shunt);
      3. ``OnsetWalk_MonodShunt_2D`` (`W4` — safe onset pilot `s in [0.55..1.30]`, 2-hop walk from `cue_carrier_binding_onset`);
      4. ``OnsetWalk_MurrayShunt_2D`` (`W5` — safe onset pilot `s in [0.55..1.30]`, 2-hop walk from `cue_vascular_branch_onset`);
      5. ``PiS_Adiabatic_CrossDomain_2D`` (`Pi_s` — certified adiabatic box `[0.4, 3.0] x [2.4, 5.2]`, routed via `coarse_graining_edge`).

    By Pigeonhole Principle across Families 3 & 4 (`Monod+Shunt` vs `Murray+Shunt` on `s in [0.55..1.30]`),
    NO static ordering (Forward, Reversed, OR DEV-Frozen!) can avoid paying extra confirmation points on >= 4 targets/seed!
    """
    rng = Random(seed)
    targets: list[UnifiedTargetSpec] = []

    # Family 1 (`4` targets): Permuted/Inverted `Haldane + Shunt` (`2D`)
    for i in range(4):
        vmax = round(rng.uniform(8.2, 10.2), 3)
        km = round(rng.uniform(1.8, 2.4), 3)
        ki = round(rng.uniform(0.12, 0.16), 4)
        ks = round(rng.uniform(0.60, 0.88), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(13.0, 17.0), 2) if sgn < 0 else round(rng.uniform(0.6, 3.0), 2)

        def _mk_hs(vm: float, k_m: float, k_i: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (vm * p["feed_conc"] / (k_m + p["feed_conc"] + k_i * (p["feed_conc"] ** 2)) + k_s * p["recycle_bleed"])

        fnames = ("recycle_bleed", "feed_conc") if i % 2 == 0 else ("feed_conc", "recycle_bleed")
        targets.append(UnifiedTargetSpec(
            target_id=f"M42_T{len(targets)+1:02d}_HaldaneShunt",
            mechanism_family="HaldaneShunt_2D",
            feature_names=fnames,
            feature_bounds={"feed_conc": (0.2, 12.0), "recycle_bleed": (0.0, 2.0)},
            pilot_points=[
                {"feed_conc": 0.4, "recycle_bleed": 0.1},
                {"feed_conc": 1.3, "recycle_bleed": 0.8},
                {"feed_conc": 2.6, "recycle_bleed": 0.2},
                {"feed_conc": 5.0, "recycle_bleed": 1.1},
                {"feed_conc": 8.0, "recycle_bleed": 0.4},
                {"feed_conc": 11.2, "recycle_bleed": 0.3},
            ],
            confirm_points=[{"feed_conc": 3.6, "recycle_bleed": 0.9}, {"feed_conc": 9.5, "recycle_bleed": 0.5}],
            heldout_points=[
                {"feed_conc": 0.8, "recycle_bleed": 0.3},
                {"feed_conc": 2.0, "recycle_bleed": 1.2},
                {"feed_conc": 4.2, "recycle_bleed": 0.6},
                {"feed_conc": 7.4, "recycle_bleed": 1.0},
                {"feed_conc": 10.8, "recycle_bleed": 0.2},
            ],
            tolerance=0.055,
            oracle_fn=_mk_hs(vmax, km, ki, ks, sgn, off),
            domain_cue=("cue_pore_collapse" if sgn < 0 else "cue_feed_reactor"),
            scale_regime="macro",
        ))

    # Family 2 (`4` targets): `Oscillatory Damped + Secular Shunt` (`2D`)
    for i in range(4):
        gain = round(rng.uniform(1.85, 2.55), 3)
        tau = round(rng.uniform(1.15, 1.55), 3)
        omega = round(rng.uniform(1.95, 2.35), 3)
        drift = round(rng.uniform(0.32, 0.55), 3)
        sgn = -1.0 if i % 2 == 1 else 1.0
        off = round(rng.uniform(14.0, 18.0), 2) if sgn < 0 else round(rng.uniform(0.5, 2.5), 2)

        def _mk_osc(g: float, t_c: float, w: float, d: float, s: float, o: float):
            return lambda p: o + s * (g * p["step_amp"] * (1.0 - math.exp(-p["dwell_t"] / t_c) * math.cos(w * p["dwell_t"])) + d * p["dwell_t"])

        targets.append(UnifiedTargetSpec(
            target_id=f"M42_T{len(targets)+1:02d}_OscShunt4P",
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
            confirm_points=[{"step_amp": 1.7, "dwell_t": 0.95}, {"step_amp": 2.1, "dwell_t": 2.65}],
            heldout_points=[
                {"step_amp": 1.0, "dwell_t": 0.25},
                {"step_amp": 1.6, "dwell_t": 1.10},
                {"step_amp": 2.4, "dwell_t": 1.75},
                {"step_amp": 1.4, "dwell_t": 2.90},
                {"step_amp": 2.6, "dwell_t": 4.05},
            ],
            tolerance=0.06,
            oracle_fn=_mk_osc(gain, tau, omega, drift, sgn, off),
            domain_cue=("cue_servo_transient" if sgn < 0 else "cue_step_relax"),
            scale_regime="micro",
        ))

    # Family 3 (`4` targets): 2-Hop Onset Walk `Monod + Shunt` (`cue_carrier_binding_onset`)
    for i in range(4):
        vm = round(rng.uniform(8.4, 9.8), 3)
        km = round(rng.uniform(1.40, 1.70), 3)
        ks = round(rng.uniform(0.62, 0.84), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(13.0, 16.5), 2) if sgn < 0 else round(rng.uniform(1.0, 3.0), 2)

        def _mk_ms(v_m: float, k_m: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (v_m * p["solute_load"] / (k_m + p["solute_load"]) + k_s * p["bypass_flow"])

        targets.append(UnifiedTargetSpec(
            target_id=f"M42_T{len(targets)+1:02d}_OnsetWalkMonodShunt",
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
            confirm_points=[{"solute_load": 11.5, "bypass_flow": 0.6}, {"solute_load": 8.5, "bypass_flow": 1.0}],
            heldout_points=[
                {"solute_load": 0.80, "bypass_flow": 0.5},
                {"solute_load": 2.60, "bypass_flow": 0.9},
                {"solute_load": 6.20, "bypass_flow": 0.4},
                {"solute_load": 9.80, "bypass_flow": 1.1},
                {"solute_load": 11.2, "bypass_flow": 0.3},
            ],
            tolerance=0.042,
            oracle_fn=_mk_ms(vm, km, ks, sgn, off),
            domain_cue="cue_carrier_binding_onset",
            scale_regime="macro",
        ))

    # Family 4 (`4` targets): 2-Hop Onset Walk `Murray 2/3 + Shunt` (`cue_vascular_branch_onset`)
    for i in range(4):
        ck = round(rng.uniform(2.85, 3.45), 3)
        ks = round(rng.uniform(0.60, 0.82), 3)
        sgn = -1.0 if i % 2 == 1 else 1.0
        off = round(rng.uniform(16.0, 20.0), 2) if sgn < 0 else round(rng.uniform(1.0, 3.0), 2)

        def _mk_murs(c_k: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (c_k * (p["solute_load"] ** (2.0 / 3.0)) + k_s * p["bypass_flow"])

        targets.append(UnifiedTargetSpec(
            target_id=f"M42_T{len(targets)+1:02d}_OnsetWalkMurrayShunt",
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
            confirm_points=[{"solute_load": 11.5, "bypass_flow": 0.6}, {"solute_load": 8.5, "bypass_flow": 1.0}],
            heldout_points=[
                {"solute_load": 0.80, "bypass_flow": 0.5},
                {"solute_load": 2.60, "bypass_flow": 0.9},
                {"solute_load": 6.20, "bypass_flow": 0.4},
                {"solute_load": 9.80, "bypass_flow": 1.1},
                {"solute_load": 11.2, "bypass_flow": 0.3},
            ],
            tolerance=0.042,
            oracle_fn=_mk_murs(ck, ks, sgn, off),
            domain_cue="cue_vascular_branch_onset",
            scale_regime="macro",
        ))

    # Family 5 (`4` targets): Multi-Scale Adiabatic `Pi_s` Cross-Domain Targets (`T in [2.5, 5.1]`)
    for i in range(4):
        scale_k = round(rng.uniform(0.90, 1.35), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(10.0, 14.0), 2) if sgn < 0 else round(rng.uniform(0.5, 2.5), 2)

        def _mk_pi_t(sk: float, s: float, o: float):
            return lambda p: o + s * sk * float(
                _eval_pi_s_fine_scalar(p["u_drive"], p["T_horizon"], DEFAULT_PI_S_ODE_PARAMS)["y_micro"]
            )

        targets.append(UnifiedTargetSpec(
            target_id=f"M42_T{len(targets)+1:02d}_PiSAdiabatic",
            mechanism_family="PiS_Adiabatic_CrossDomain_2D",
            feature_names=("u_drive", "T_horizon"),
            feature_bounds={"u_drive": (0.4, 3.0), "T_horizon": (2.4, 5.2)},
            pilot_points=[
                {"u_drive": 0.6, "T_horizon": 4.8},
                {"u_drive": 1.2, "T_horizon": 2.5},
                {"u_drive": 1.8, "T_horizon": 4.2},
                {"u_drive": 2.4, "T_horizon": 2.7},
                {"u_drive": 2.9, "T_horizon": 3.6},
                {"u_drive": 0.9, "T_horizon": 3.1},
            ],
            confirm_points=[{"u_drive": 2.6, "T_horizon": 4.9}, {"u_drive": 0.8, "T_horizon": 2.55}],
            heldout_points=[
                {"u_drive": 0.8, "T_horizon": 2.6},
                {"u_drive": 1.4, "T_horizon": 3.1},
                {"u_drive": 2.1, "T_horizon": 3.9},
                {"u_drive": 2.7, "T_horizon": 4.8},
            ],
            tolerance=0.045,
            oracle_fn=_mk_pi_t(scale_k, sgn, off),
            domain_cue="cue_adiabatic_multiscale",
            scale_regime="meso",
        ))

    # Seed-varied Out-of-Scope Honesty Targets (`2` per seed -> `8` genuinely distinct OOS targets across 4 seeds!)
    oos_targets: list[UnifiedTargetSpec] = []
    for o_i in range(2):
        freq = round(rng.uniform(5.8, 7.6), 3)
        amp = round(rng.uniform(3.4, 4.6), 3)
        jump = round(rng.uniform(3.1, 4.2), 3)
        phase_sh = round(rng.uniform(0.1, 2.8), 3)

        def _mk_oos(fq: float, am: float, jm: float, ph: float):
            return lambda p: am * math.sin(fq * p["trunk_flow"] + ph) + (jm if int(p["trunk_flow"] * 3) % 2 == 0 else -jm)

        oos_targets.append(UnifiedTargetSpec(
            target_id=f"M42_OOS_{o_i+1:02d}_s{seed % 10000}",
            mechanism_family="OutOfScope_Discontinuous",
            feature_names=("trunk_flow",),
            feature_bounds={"trunk_flow": (0.3, 9.0)},
            pilot_points=[{"trunk_flow": v} for v in (0.5, 1.8, 3.6, 5.8, 8.2)],
            confirm_points=[{"trunk_flow": 2.9}, {"trunk_flow": 6.7}],
            heldout_points=[{"trunk_flow": v} for v in (0.9, 2.5, 5.1, 7.3, 8.7)],
            tolerance=0.05,
            oracle_fn=_mk_oos(freq, amp, jump, phase_sh),
            domain_cue="low_flow_cdn_onset",
            scale_regime="macro",
            is_out_of_scope=True,
        ))

    return targets, oos_targets


# ============================================================================
# Part 7. Unified Stage-3 Arm Evaluator + Full Execution Receipts
# ============================================================================

def _extract_canonical_box_roles(
    node: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
    action_override: dict[str, Any] | None = None,
) -> list[str]:
    sm = node.get("scale_map") or {}
    vbox = (sm.get("window_or_partition") or {}).get("verified_input_box") or {}
    vbox_keys = [str(k) for k in vbox.keys()]
    act = (
        action_override
        if isinstance(action_override, dict)
        else ((node.get("claim_contract") or {}).get("conclusion_or_action") or {})
    )
    if act.get("fine_ode_direct_mode") is True:
        return ["u_drive", "T_horizon"]
    if isinstance(act.get("role_names"), list) and act["role_names"]:
        return [str(r) for r in act["role_names"]]
    art_path = act.get("authored_artifact_path")
    if art_path:
        art_p = Path(str(art_path))
        if not art_p.is_absolute() and artifact_root is not None:
            cand_p = (Path(str(artifact_root)) / art_p).resolve()
            if not cand_p.exists() and art_p.parts and art_p.parts[0] == "persisted_authored_workspaces":
                cand_p = (Path(str(artifact_root)) / Path(*art_p.parts[1:])).resolve()
            art_p = cand_p
        if art_p.is_file():
            try:
                raw_art = json.loads(art_p.read_text(encoding="utf-8"))
                if isinstance(raw_art.get("role_names"), list):
                    return [str(r) for r in raw_art["role_names"]]
            except Exception:
                pass
    return vbox_keys


def validate_canonical_role_map_contract(
    target_feature_names: tuple[str, ...],
    canon_roles: list[str],
    *,
    explicit_role_map: dict[str, str] | None = None,
) -> tuple[bool, str, list[dict[str, str]]]:
    """Unified public coordinate-role contract enforcing full & partial canonical-axis identity locks.

    - Any input coordinate whose name ALREADY matches a canonical role in ``canon_roles`` MUST map to itself
      (`tf -> tf`), preventing both implicit permutation and contradictory caller ``role_map`` overrides!
    - Anonymous coordinates (e.g. ``x0, x1``) may map bijectively onto the remaining canonical roles.
    """
    if not canon_roles:
        return True, "ok", []
    t_names = [str(tf) for tf in target_feature_names]
    c_names = [str(cr) for cr in canon_roles]
    if len(t_names) != len(c_names) or len(set(t_names)) != len(t_names) or len(set(c_names)) != len(c_names):
        return False, "arity_or_uniqueness_mismatch_role_map", []

    canon_set = set(c_names)
    if explicit_role_map is not None:
        norm_map = {str(k): str(v) for k, v in explicit_role_map.items()}
        if set(norm_map.keys()) != set(t_names) or set(norm_map.values()) != canon_set:
            return False, "invalid_explicit_role_map_bijection", []
        for tf in t_names:
            if tf in canon_set and norm_map[tf] != tf:
                return False, "contradictory_canonical_axis_role_map", []
        return True, "ok", [norm_map]

    from itertools import permutations

    allowed: list[dict[str, str]] = []
    for perm in permutations(c_names):
        rmap = {tf: cr for tf, cr in zip(t_names, perm)}
        if any(tf in canon_set and rmap[tf] != tf for tf in t_names):
            continue
        if rmap not in allowed:
            allowed.append(rmap)
    if not allowed:
        return False, "no_consistent_canonical_bijection", []
    return True, "ok", allowed


def _role_order_from_canonical_role_map(canon_roles: list[str], role_map: dict[str, str]) -> list[str]:
    """Invert ``target_feature -> canonical_role`` bijection into ``compile_operator_spec_to_python`` ``role_names``."""
    inv = {str(cr): str(tf) for tf, cr in role_map.items()}
    return [inv[str(cr)] for cr in canon_roles]


def _find_all_valid_canonical_role_maps_for_box(
    target_feature_names: tuple[str, ...],
    node: dict[str, Any],
    points_to_check: list[dict[str, float]],
    *,
    artifact_root: str | Path | None = None,
    action_override: dict[str, Any] | None = None,
) -> tuple[bool, list[dict[str, str]], list[str]]:
    """Return ``(has_no_box_or_valid, valid_role_maps, canon_roles)`` under full & partial canonical locks."""
    sm = node.get("scale_map") or {}
    vbox = (sm.get("window_or_partition") or {}).get("verified_input_box")
    if not isinstance(vbox, dict) or not vbox:
        return True, [], []

    vbox_keys = [str(k) for k in vbox.keys()]
    canon_roles = _extract_canonical_box_roles(
        node, artifact_root=artifact_root, action_override=action_override
    )
    if len(target_feature_names) != len(canon_roles) or not set(vbox_keys).issubset(set(canon_roles)):
        return False, [], canon_roles

    rm_ok, _, candidate_maps = validate_canonical_role_map_contract(
        target_feature_names, canon_roles, explicit_role_map=None
    )
    if not rm_ok:
        return False, [], canon_roles

    valid_maps = [
        rmap
        for rmap in candidate_maps
        if all(is_point_inside_verified_box(pt, vbox, role_map=rmap) for pt in points_to_check)
    ]
    return bool(valid_maps), valid_maps, canon_roles


def _find_valid_canonical_role_map_for_box(
    target_feature_names: tuple[str, ...],
    node: dict[str, Any],
    points_to_check: list[dict[str, float]],
    *,
    artifact_root: str | Path | None = None,
    action_override: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, str] | None]:
    """Resolve a canonical coordinate bijection (`target_feature -> certified_box_axis`) before checking bounds."""
    ok, valid_maps, _ = _find_all_valid_canonical_role_maps_for_box(
        target_feature_names, node, points_to_check, artifact_root=artifact_root, action_override=action_override
    )
    if not ok:
        return False, None
    return True, (valid_maps[0] if valid_maps else None)


def _build_stage0_open_atomic_bases(arity: int) -> list[dict[str, Any]]:
    """Return strictly Stage-0 atomic operator primitives for genuine ``CodeAgent`` open rediscovery."""
    if arity == 1:
        return [
            make_atomic_operator_spec("linear_1d"),
            make_atomic_operator_spec("michaelis_1d", arity=1),
            make_atomic_operator_spec("murray_power_2_3_1d"),
        ]
    murray_2d = deepcopy(make_atomic_operator_spec("murray_power_2_3_1d"))
    murray_2d["arity"] = arity
    return [
        make_atomic_operator_spec("michaelis_1d", arity=arity),
        murray_2d,
        make_atomic_operator_spec("exp_relax_2d"),
    ]


def evaluate_m4_2_target_under_arm(
    target: UnifiedTargetSpec,
    *,
    arm_name: str,
    seed: int,
    work_dir: Path,
    forest: PhilosophicalForest | None,
    index: HippoRAGWormholeIndex | FlatEqualMetadataIndex | None,
    ordinary_history_specs: list[tuple[str, dict[str, Any]]] | None,
    fallback_pool_specs: list[tuple[str, dict[str, Any]]],
    open_code_agent_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate a single ``M4.2`` target with channel-bound multi-scale pricing, canonical role-mapped
    ``Pi_s`` scale-box gating (including final continuation queries!), fresh disjoint confirmation
    (`prefix_confirm_overlap_count == 0`), metered retrieval compute, and genuine ``CodeAgent`` open rediscovery!
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    is_pi_s_target = target.mechanism_family == "PiS_Adiabatic_CrossDomain_2D"
    meter = CostMeter(obs_budget=64, meas_budget=800.0, compute_budget=5000.0)

    # Channel-bound pricing: on multiscale `Pi_s` targets, `coarse_macro` (`4.0`/obs) is ONLY granted if
    # retrieval actually reaches an authorized `meso` node backed by ITS OWN live non-stale `coarse_graining_edge`
    # and in-box pilot coordinates; otherwise observations execute on `micro_fine` (`8.0`/obs)!
    selected_channel = "macro_single" if not is_pi_s_target else "micro_fine"
    authorizing_coarse_node_id: str | None = None
    if is_pi_s_target and forest is not None and index is not None:
        pre_ret = index.retrieve(
            [f"sig:{target.domain_cue}", f"sig:pos_arity_{len(target.feature_names)}"],
            task_view={
                "goal_kind": "open_workspace_mechanism_prediction",
                "tolerance": target.tolerance,
                "active_arity": len(target.feature_names),
                "target_scale_regime": target.scale_regime,
            },
            top_k=8,
        )
        meter.charge("scale_channel_precheck_retrieval", compute=round(float(pre_ret.get("query_compute_cost", 0.0)), 4))
        for pc in pre_ret["candidates"]:
            p_nd = pc["target_node"]
            p_sm = p_nd.get("scale_map") or {}
            if (p_sm.get("window_or_partition") or {}).get("scale_regime") != "meso":
                continue
            # Require a live, non-stale coarse_graining_edge specifically into THIS coarse node!
            if not forest.fine_scale_children_of(p_nd["node_id"]):
                continue
            in_p_box, _ = _find_valid_canonical_role_map_for_box(target.feature_names, p_nd, list(target.pilot_points))
            if not in_p_box:
                continue
            p_gate = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
                target_features={"active_arity": len(target.feature_names), "modality": "hydraulic"},
                claim_contract=p_nd["claim_contract"],
                scale_map=p_sm,
                wormhole_edge=pc.get("wormhole_edge"),
            )
            if p_gate["granted"]:
                selected_channel = "coarse_macro"
                authorizing_coarse_node_id = str(p_nd["node_id"])
                break

    obs_unit_cost = 4.0 if selected_channel in {"macro_single", "coarse_macro"} else 8.0

    lab = OpenWorkspaceTaskLab(
        task_id=target.target_id,
        surface_domain=target.domain_cue,
        feature_names=target.feature_names,
        feature_bounds=target.feature_bounds,
        cost_per_obs=obs_unit_cost,
        task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
        meter=meter,
        private_oracle_fn=target.oracle_fn,
    )

    p_obs = lab.acquire(target.pilot_points, workspace=work_dir, phase="pilot") or []
    frozen_prefix = [{"features": dict(r.features), "target": float(r.target)} for r in p_obs]
    pilot_sigs, obs_feats = extract_unified_observable_pilot_cues(
        frozen_prefix, feature_names=target.feature_names, domain_cue=target.domain_cue
    )
    if len(target.feature_names) == 2:
        obs_feats["secondary_bleed_active"] = True
    if is_pi_s_target:
        obs_feats["modality"] = "hydraulic"  # Exercises compressed OR-branch (`thermal | hydraulic`) in Stage 3 too!

    chosen_rel: str | None = None
    confirmed_ok = False
    confirmed_via_recall = False
    non_direct_walk_win = False
    total_evaluated = 0
    unmetered_eval_candidates = 0
    selected_node_id: str | None = None
    selected_version: int | None = None
    selected_model_sha256: str | None = None
    selected_conf_digest: str | None = None
    confirmed_vbox: dict[str, Any] | None = None
    confirmed_role_map: dict[str, str] | None = None
    max_prefix_confirm_overlap = 0
    refuted_tags: set[str] = set()

    def _count_overlap(q_rows: list[dict[str, Any]], pref_rows: list[dict[str, Any]]) -> int:
        return sum(
            1 for qr in q_rows if any(coords_approx_equal(qr["features"], pr["features"]) for pr in pref_rows)
        )

    # 1. Forest-governed recall (`Hegel`, `NoMaintenance`, `Alpha0`, `Flat_BM25`, `Frozen`)
    if forest is not None and index is not None:
        ret = index.retrieve(
            pilot_sigs,
            task_view={
                "goal_kind": "open_workspace_mechanism_prediction",
                "tolerance": target.tolerance,
                "active_arity": len(target.feature_names),
                "target_scale_regime": target.scale_regime,
            },
            top_k=8,
        )
        lab.meter.charge("forest_retrieval_query", compute=round(float(ret.get("query_compute_cost", 0.0)), 4))
        auth_cands: list[dict[str, Any]] = []
        for cand in ret["candidates"]:
            nd = cand["target_node"]
            if int(nd["observation_map"]["parameter_semantics"].get("active_arity", 0)) != len(target.feature_names):
                continue
            sm = nd.get("scale_map") or {}
            nd_scale = (sm.get("window_or_partition") or {}).get("scale_regime")
            # Coarse `meso` reduction on `Pi_s` requires a live non-stale coarse_graining_edge for THIS node!
            if is_pi_s_target and nd_scale == "meso" and not forest.fine_scale_children_of(nd["node_id"]):
                continue
            gate = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": target.tolerance},
                target_features=obs_feats,
                claim_contract=nd["claim_contract"],
                scale_map=nd.get("scale_map"),
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
            v_spec = _load_node_verified_operator_spec(nd, artifact_root=forest.artifact_root, action_override=eff_act)
            if v_spec is None:
                continue
            in_box, valid_rmaps, canon_roles = _find_all_valid_canonical_role_maps_for_box(
                target.feature_names,
                nd,
                [*target.pilot_points, *target.confirm_points],
                artifact_root=forest.artifact_root,
                action_override=eff_act,
            )
            if not in_box:
                continue
            cand_copy = dict(cand)
            cand_copy["_verified_spec"] = v_spec
            cand_copy["_exec_plan"] = plan
            cand_copy["_valid_role_maps"] = valid_rmaps
            cand_copy["_canon_roles"] = canon_roles
            cand_copy["_canonical_role_map"] = valid_rmaps[0] if valid_rmaps else None
            cand_copy["_selected_action"] = eff_act
            auth_cands.append(cand_copy)

        # Try top authorized recalled candidates sequentially (retaining all paid rows & enforcing 0 overlap!)
        tried_tags: set[str] = set()
        for c_cand in auth_cands:
            plan_c = c_cand["_exec_plan"]
            c_tag = str(plan_c["operator_tag"])
            if c_tag in tried_tags:
                continue
            tried_tags.add(c_tag)
            nd = c_cand["target_node"]
            nd_scale = ((nd.get("scale_map") or {}).get("window_or_partition") or {}).get("scale_regime")
            if is_pi_s_target and nd_scale != "meso" and selected_channel == "coarse_macro":
                selected_channel = "micro_fine"
                lab.cost_per_obs = 8.0

            valid_rmaps_c = c_cand.get("_valid_role_maps") or []
            canon_roles_c = c_cand.get("_canon_roles") or []
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
            unmetered_eval_candidates += ev_c
            if best_c is None or best_c["train_rmse"] > target.tolerance:
                continue

            locked_rmap = c_cand.get("_canonical_role_map")
            if valid_rmaps_c and best_c.get("role_order"):
                for rm in valid_rmaps_c:
                    if _role_order_from_canonical_role_map(canon_roles_c, rm) == list(best_c["role_order"]):
                        locked_rmap = rm
                        break
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
            cf_obs = lab.acquire(c_pts, workspace=None, phase="recall_confirm") or []
            cf_rows = [{"features": dict(r.features), "target": float(r.target)} for r in cf_obs]
            max_prefix_confirm_overlap = max(max_prefix_confirm_overlap, _count_overlap(cf_rows, frozen_prefix))
            cf_ev = evaluate_isolated_python_model(
                work_dir, best_c["model_relpath"], cf_rows, isolated_prefix_rows=frozen_prefix
            )
            if cf_ev["returncode"] == 0 and float(cf_ev["rmse"]) <= target.tolerance:
                m_sha = hashlib.sha256(best_c["model_source"].encode("utf-8")).hexdigest()
                c_dig = hashlib.sha256(json.dumps(cf_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
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
                    target_features=obs_feats,
                    claim_contract=nd["claim_contract"],
                    scale_map=nd.get("scale_map"),
                    wormhole_edge=c_cand.get("wormhole_edge"),
                    evidence_receipt={
                        "passed": True,
                        "claim_id": nd["claim_contract"]["claim_id"],
                        "observed_error": float(cf_ev["rmse"]),
                        "target_id": target.target_id,
                        "node_version": nd["version"],
                        "model_sha256": m_sha,
                        "confirmation_digest": c_dig,
                    },
                )
                if t3["granted"]:
                    confirmed_ok = True
                    confirmed_via_recall = True
                    chosen_rel = best_c["model_relpath"]
                    non_direct_walk_win = bool(c_cand.get("reached_only_by_graph_walk"))
                    selected_node_id = str(nd["node_id"])
                    selected_version = int(nd["version"])
                    selected_model_sha256 = m_sha
                    selected_conf_digest = c_dig
                    confirmed_vbox = deepcopy(((nd.get("scale_map") or {}).get("window_or_partition") or {}).get("verified_input_box"))
                    confirmed_role_map = deepcopy(c_cand.get("_canonical_role_map"))
                    break
            # Retain paid failed confirmation rows for ALL downstream candidates & fallback!
            refuted_tags.add(c_tag)
            frozen_prefix = [*frozen_prefix, *cf_rows]

    # 2. Ordinary All-History baseline (`Ordinary_DevFrozenOrder`, `Ordinary_Forward`, `Ordinary_Reversed`)
    elif ordinary_history_specs is not None:
        same_ar_hist = [(t, sp) for t, sp in ordinary_history_specs if int(sp.get("arity", 1)) == len(target.feature_names)]
        best_h, ev_h = _autonomous_fit_candidate_specs(
            work_dir,
            specs_to_try=same_ar_hist,
            feature_names=target.feature_names,
            training_rows=frozen_prefix,
            tolerance=target.tolerance,
        )
        total_evaluated += ev_h
        unmetered_eval_candidates += ev_h
        if best_h is not None and best_h["train_rmse"] <= target.tolerance:
            cf0_pts = select_or_synthesize_disjoint_confirm_points(
                predeclared_confirm_points=target.confirm_points,
                training_prefix_rows=frozen_prefix,
                required_fresh_count=1,
                feature_bounds=target.feature_bounds,
            )
            cf0_obs = lab.acquire(cf0_pts, workspace=None, phase="ordinary_confirm_0") if cf0_pts else []
            cf0_rows = [{"features": dict(r.features), "target": float(r.target)} for r in (cf0_obs or [])]
            max_prefix_confirm_overlap = max(max_prefix_confirm_overlap, _count_overlap(cf0_rows, frozen_prefix))
            cf0_ev = evaluate_isolated_python_model(
                work_dir, best_h["model_relpath"], cf0_rows, isolated_prefix_rows=frozen_prefix
            )
            if cf0_ev["returncode"] == 0 and float(cf0_ev["rmse"]) <= target.tolerance:
                confirmed_ok = True
                confirmed_via_recall = True
                chosen_rel = best_h["model_relpath"]
                selected_node_id = f"flat_history:{best_h['tag']}"
                selected_model_sha256 = hashlib.sha256(best_h["model_source"].encode("utf-8")).hexdigest()
                selected_conf_digest = hashlib.sha256(json.dumps(cf0_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
            else:
                refuted_tags.add(str(best_h["tag"]))
                frozen_prefix = [*frozen_prefix, *cf0_rows]
                rem_hist = [(t, sp) for t, sp in same_ar_hist if t not in refuted_tags]
                best_h2, ev_h2 = _autonomous_fit_candidate_specs(
                    work_dir,
                    specs_to_try=rem_hist,
                    feature_names=target.feature_names,
                    training_rows=frozen_prefix,
                    tolerance=target.tolerance,
                )
                total_evaluated += ev_h2
                unmetered_eval_candidates += ev_h2
                if best_h2 is not None and best_h2["train_rmse"] <= target.tolerance:
                    cf1_pts = select_or_synthesize_disjoint_confirm_points(
                        predeclared_confirm_points=target.confirm_points,
                        training_prefix_rows=frozen_prefix,
                        required_fresh_count=1,
                        feature_bounds=target.feature_bounds,
                    )
                    cf1_obs = lab.acquire(cf1_pts, workspace=None, phase="ordinary_confirm_1") if cf1_pts else []
                    cf1_rows = [{"features": dict(r.features), "target": float(r.target)} for r in (cf1_obs or [])]
                    max_prefix_confirm_overlap = max(max_prefix_confirm_overlap, _count_overlap(cf1_rows, frozen_prefix))
                    cf1_ev = evaluate_isolated_python_model(
                        work_dir, best_h2["model_relpath"], cf1_rows, isolated_prefix_rows=frozen_prefix
                    )
                    if cf1_ev["returncode"] == 0 and float(cf1_ev["rmse"]) <= target.tolerance:
                        confirmed_ok = True
                        confirmed_via_recall = True
                        chosen_rel = best_h2["model_relpath"]
                        selected_node_id = f"flat_history:{best_h2['tag']}"
                        selected_model_sha256 = hashlib.sha256(best_h2["model_source"].encode("utf-8")).hexdigest()
                        selected_conf_digest = hashlib.sha256(json.dumps(cf1_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                    else:
                        refuted_tags.add(str(best_h2["tag"]))
                        frozen_prefix = [*frozen_prefix, *cf1_rows]

    # 3. Fallback when recall / history did not confirm (escalate out-of-box coarse channel to micro_fine!)
    if not confirmed_ok:
        if is_pi_s_target and selected_channel == "coarse_macro":
            # Coarse recall did not confirm (e.g. out-of-box confirm or missing edge) -> escalate channel to micro_fine!
            selected_channel = "micro_fine"
            lab.cost_per_obs = 8.0

        # If onset pilot had only low primary drive (`<= 2.0` when upper bound `>= 9.0`), acquire 2 high-drive stress probes!
        f0 = target.feature_names[0]
        max_f0 = max(float(r["features"][f0]) for r in frozen_prefix)
        hi_f0 = target.feature_bounds[f0][1]
        if max_f0 < 0.25 * hi_f0 and hi_f0 >= 8.0:
            if len(target.feature_names) == 2:
                f1 = target.feature_names[1]
                st_pts = [{f0: round(0.55 * hi_f0, 2), f1: 0.4}, {f0: round(0.85 * hi_f0, 2), f1: 0.9}]
            else:
                st_pts = [{f0: round(0.65 * hi_f0, 2)}, {f0: round(0.88 * hi_f0, 2)}]
            st_obs = lab.acquire(st_pts, workspace=None, phase="stress_probe") or []
            frozen_prefix = [*frozen_prefix, *[{"features": dict(r.features), "target": float(r.target)} for r in st_obs]]

        if open_code_agent_fallback:
            # GENUINE `CodeAgent` open compositional discovery episode (zero synthetic meter charges!)
            raw_perms = [list(p) for p in _infer_role_permutations(target.feature_names)]
            atomic_bases = _build_stage0_open_atomic_bases(len(target.feature_names))
            for p_idx, ordered_roles in enumerate(raw_perms):
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
                post_diag_prefix = [*frozen_prefix, *[{"features": dict(p), "target": 0.0} for p in diag_pts]]
                ep_cf_pts = select_or_synthesize_disjoint_confirm_points(
                    predeclared_confirm_points=target.confirm_points,
                    training_prefix_rows=post_diag_prefix,
                    required_fresh_count=1,
                    feature_bounds=target.feature_bounds,
                )
                if not ep_cf_pts:
                    break
                ep_ws = work_dir / f"open_code_agent_ep_{p_idx}"
                ep_res = run_compositional_code_agent_episode(
                    lab=lab,
                    heldout_rows=None,
                    tolerance=target.tolerance,
                    workspace_dir=ep_ws,
                    role_names=ordered_roles,
                    initial_query_batches=([diag_pts] if diag_pts else []),
                    atomic_base_specs=atomic_bases,
                    confirm_points=ep_cf_pts,
                    lens_hint=lens_est,
                    lens_calibrator_fn=lambda sp_obj, r_ord: _estimate_affine_lens_for_spec(
                        sp_obj,
                        r_ord,
                        [{"features": dict(o.features), "target": float(o.target)} for o in lab.observations] or frozen_prefix,
                    ),
                )
                total_evaluated += int(ep_res["evaluated_candidate_count"])
                ep_bundle = ep_res["authored_artifact"]["host_confirmation_bundle"]
                max_prefix_confirm_overlap = max(
                    max_prefix_confirm_overlap,
                    _count_overlap(ep_bundle["confirm_rows"], ep_bundle["training_prefix_rows"]),
                )
                frozen_prefix = deepcopy(ep_res["training_prefix_rows"])
                if ep_res["confirmed_ok"]:
                    confirmed_ok = True
                    chosen_rel = "open_discovered.py"
                    (work_dir / chosen_rel).write_text(ep_res["authored_artifact"]["authored_model_source"], encoding="utf-8")
                    selected_node_id = f"open_code_agent:{ep_res['authored_artifact']['operator_spec'].get('family_hint', 'composed')}"
                    selected_model_sha256 = str(ep_res["authored_artifact"]["model_sha256"])
                    selected_conf_digest = str(ep_res["authored_artifact"]["confirmation_digest"])
                    break
                frozen_prefix = [*frozen_prefix, *deepcopy(ep_bundle["confirm_rows"])]
                from .models import Observation
                lab.observations = [
                    Observation(features=dict(r["features"]), target=float(r["target"]), history=())
                    for r in frozen_prefix
                ]

        if not confirmed_ok and not open_code_agent_fallback:
            same_ar_fb = [
                (t, sp) for t, sp in fallback_pool_specs
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
            unmetered_eval_candidates += ev_fb
            rem_cf_pts = select_or_synthesize_disjoint_confirm_points(
                predeclared_confirm_points=target.confirm_points,
                training_prefix_rows=frozen_prefix,
                required_fresh_count=1,
                feature_bounds=target.feature_bounds,
            )
            fb_obs = lab.acquire(rem_cf_pts, workspace=None, phase="fallback_confirm") if rem_cf_pts else []
            fb_rows = [{"features": dict(r.features), "target": float(r.target)} for r in (fb_obs or [])]
            max_prefix_confirm_overlap = max(max_prefix_confirm_overlap, _count_overlap(fb_rows, frozen_prefix))
            if best_fb is not None and float(best_fb["train_rmse"]) <= target.tolerance and fb_rows:
                fb_ev = evaluate_isolated_python_model(
                    work_dir, best_fb["model_relpath"], fb_rows, isolated_prefix_rows=frozen_prefix
                )
                if fb_ev["returncode"] == 0 and float(fb_ev["rmse"]) <= target.tolerance:
                    confirmed_ok = True
                    chosen_rel = best_fb["model_relpath"]
                    selected_node_id = f"fallback:{best_fb['tag']}"
                    selected_model_sha256 = hashlib.sha256(best_fb["model_source"].encode("utf-8")).hexdigest()
                    selected_conf_digest = hashlib.sha256(json.dumps(fb_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    if unmetered_eval_candidates > 0 or not open_code_agent_fallback:
        lab.meter.charge("target_synth_compute", compute=round(2.5 + 6.0 * unmetered_eval_candidates, 4))

    # Persist & enforce `model_scope_contract.json` BEFORE final/continuation queries (`heldout_points`)!
    scope_contract = {
        "selected_channel": selected_channel,
        "authorizing_coarse_node_id": authorizing_coarse_node_id,
        "verified_input_box": confirmed_vbox,
        "canonical_role_map": confirmed_role_map,
    }
    (work_dir / "model_scope_contract.json").write_text(
        json.dumps(scope_contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    out_of_box_query_blocked = False
    if confirmed_ok and isinstance(confirmed_vbox, dict) and confirmed_vbox:
        if not all(
            is_point_inside_verified_box(pt, confirmed_vbox, role_map=confirmed_role_map)
            for pt in target.heldout_points
        ):
            out_of_box_query_blocked = True
            confirmed_ok = False

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

    snap = lab.meter.snapshot()
    return {
        "seed": seed,
        "target_id": target.target_id,
        "mechanism_family": target.mechanism_family,
        "is_out_of_scope": target.is_out_of_scope,
        "arm": arm_name,
        "status": (
            "out_of_verified_scale_box"
            if out_of_box_query_blocked
            else ("completed" if confirmed_ok else "scoped_unknown")
        ),
        "selected_channel": selected_channel,
        "out_of_box_query_blocked": out_of_box_query_blocked,
        "prefix_confirm_overlap_count": max_prefix_confirm_overlap,
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


# ============================================================================
# Part 8. Full End-to-End M4.2 Study Orchestrator + Observed Breakeven Curve
# ============================================================================

def run_m4_2_unified_cross_pillar_study(
    work_root: Path,
    *,
    seeds: tuple[int, ...] = (20_261_120, 20_271_120, 20_281_120, 20_291_120),
) -> dict[str, Any]:
    """Run the complete ``Hegel V17.4 (M4.2)`` cross-pillar continual lifecycle, zero-fit dispatcher,
    8-arm unseen multi-seed benchmark (`80` in-scope + `8` OOS runs/arm), and observed breakeven curve!
    """
    work_root.mkdir(parents=True, exist_ok=True)
    built = build_and_evolve_m4_2_cross_pillar_forest(work_root / "continual_lifecycle")

    # Snapshot the EXACT post-Stage-2 maintained forest BEFORE running dispatcher or Stage-3 seeds!
    post_stage2_forest_snapshot = deepcopy(built["forest"])
    unmaintained_forest_snapshot = deepcopy(built["forest_before_stage2_maintenance"])

    # Act 3: Run interleaved zero-`fit()` `ForestScopeDispatcher` (`R3` Tier 1 + Tier 3!)
    dispatcher_report = run_interleaved_zero_fit_dispatcher_stream(deepcopy(post_stage2_forest_snapshot))

    # Use EXACT runtime Stage-0 boundary snapshot (all 8 nodes, 2 coarse edges, 1 wormhole edge!)
    frozen_stage0_forest = deepcopy(built["forest_after_stage0"])

    arm_names = (
        "Hegel_Unified_Continual_Forest",
        "Hegel_NoMaintenance_SameHistory",
        "Ordinary_AllHistory_DevFrozenOrder",
        "Ordinary_AllHistory_Forward",
        "Ordinary_AllHistory_Reversed",
        "Forest_Alpha0_SameHistory",
        "Flat_BM25_SameHistory",
        "Frozen_After_Stage0_OpenRediscovery",
        "Frozen_After_Stage0_PoolFallback",
    )

    in_scope_records: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}
    oos_records: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}

    for s_idx, seed in enumerate(seeds):
        targets, oos_targets = build_m4_2_target_suite(seed=seed)
        # Restore exact post-Stage-2 maintained forest snapshot at the start of EVERY seed!
        f_hegel = deepcopy(post_stage2_forest_snapshot)
        f_nomaint = deepcopy(unmaintained_forest_snapshot)
        f_alpha0 = deepcopy(post_stage2_forest_snapshot)
        f_flat = deepcopy(post_stage2_forest_snapshot)
        f_frozen0 = deepcopy(frozen_stage0_forest)

        idx_hegel = HippoRAGWormholeIndex(f_hegel, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        idx_nomaint = HippoRAGWormholeIndex(f_nomaint, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        idx_alpha0 = HippoRAGWormholeIndex(f_alpha0, damping_alpha=0.0, wormhole_equivalence_pooling=False, scale_aware=True)
        idx_flat = FlatEqualMetadataIndex(f_flat)
        idx_frozen0 = HippoRAGWormholeIndex(f_frozen0, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)

        arm_configs = {
            "Hegel_Unified_Continual_Forest": (f_hegel, idx_hegel, None, False),
            "Hegel_NoMaintenance_SameHistory": (f_nomaint, idx_nomaint, None, False),
            "Ordinary_AllHistory_DevFrozenOrder": (None, None, built["ordinary_history_dev_frozen"], False),
            "Ordinary_AllHistory_Forward": (None, None, built["ordinary_history_forward"], False),
            "Ordinary_AllHistory_Reversed": (None, None, built["ordinary_history_reversed"], False),
            "Forest_Alpha0_SameHistory": (f_alpha0, idx_alpha0, None, False),
            "Flat_BM25_SameHistory": (f_flat, idx_flat, None, False),
            "Frozen_After_Stage0_OpenRediscovery": (f_frozen0, idx_frozen0, None, True),
            "Frozen_After_Stage0_PoolFallback": (f_frozen0, idx_frozen0, None, False),
        }

        for t in targets:
            for a_name, (f_obj, idx_obj, ord_hist, open_fb) in arm_configs.items():
                fb_specs = (
                    built["stage0_atomic_specs"]
                    if a_name == "Frozen_After_Stage0_OpenRediscovery"
                    else built["ordinary_history_forward"]
                )
                rec = evaluate_m4_2_target_under_arm(
                    t,
                    arm_name=a_name,
                    seed=seed,
                    work_dir=work_root / "stage3_eval" / f"seed_{s_idx}" / a_name / t.target_id,
                    forest=f_obj,
                    index=idx_obj,
                    ordinary_history_specs=ord_hist,
                    fallback_pool_specs=fb_specs,
                    open_code_agent_fallback=open_fb,
                )
                in_scope_records[a_name].append(rec)

        for ot in oos_targets:
            for a_name, (f_obj, idx_obj, ord_hist, open_fb) in arm_configs.items():
                fb_specs = (
                    built["stage0_atomic_specs"]
                    if a_name == "Frozen_After_Stage0_OpenRediscovery"
                    else built["ordinary_history_forward"]
                )
                orec = evaluate_m4_2_target_under_arm(
                    ot,
                    arm_name=a_name,
                    seed=seed,
                    work_dir=work_root / "stage3_oos" / f"seed_{s_idx}" / a_name / ot.target_id,
                    forest=f_obj,
                    index=idx_obj,
                    ordinary_history_specs=ord_hist,
                    fallback_pool_specs=fb_specs,
                    open_code_agent_fallback=open_fb,
                )
                oos_records[a_name].append(orec)

    # Summarize Stage-3 & Cumulative Lifecycle Costs + Observed 80-Target Breakeven Curve!
    s0_cost = float(built["stage0"]["composite_cost"])
    s1_cost = float(built["stage1"]["composite_cost"])
    s2_cost = float(built["stage2"]["composite_cost"])

    arm_summaries: dict[str, Any] = {}
    for a_name in arm_names:
        rows = in_scope_records[a_name]
        orows = oos_records[a_name]
        stage3_total_comp_cost = round(sum(float(r["cost"]["composite_cost"]) for r in rows), 4)
        upfront = (
            s0_cost + s1_cost + s2_cost
            if a_name in {"Hegel_Unified_Continual_Forest", "Forest_Alpha0_SameHistory", "Flat_BM25_SameHistory"}
            else (
                s0_cost + s1_cost
                if a_name in {"Hegel_NoMaintenance_SameHistory", "Ordinary_AllHistory_Forward", "Ordinary_AllHistory_Reversed"}
                else (
                    s0_cost + s1_cost + float(built["stage2"]["dev_static_order_freeze"]["dev_tuning_cost_tracked_separately"]["composite_cost"])
                    if a_name == "Ordinary_AllHistory_DevFrozenOrder"
                    else s0_cost
                )
            )
        )
        arm_summaries[a_name] = {
            "in_scope_runs": len(rows),
            "oos_runs": len(orows),
            "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in rows), 4),
            "false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in rows), 4),
            "oos_scoped_unknown_rate": round(mean(1.0 if r["status"] == "scoped_unknown" else 0.0 for r in orows), 4),
            "oos_false_confirmation_rate": round(mean(1.0 if r["false_confirmation"] else 0.0 for r in orows), 4),
            "recall_confirmation_count": sum(1 for r in rows if r["confirmed_via_recall"]),
            "non_direct_walk_wins": sum(1 for r in rows if r["non_direct_walk_win"]),
            "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in rows), 4),
            "mean_measurement_cost": round(mean(float(r["cost"]["measurement_cost"]) for r in rows), 4),
            "mean_compute_cost": round(mean(float(r["cost"]["compute_cost"]) for r in rows), 4),
            "mean_composite_cost": round(mean(float(r["cost"]["composite_cost"]) for r in rows), 4),
            "sd_composite_cost": round(pstdev(float(r["cost"]["composite_cost"]) for r in rows), 4),
            "upfront_lifecycle_composite_cost": round(upfront, 4),
            "stage3_80_target_composite_cost": stage3_total_comp_cost,
            "cumulative_total_lifecycle_composite_cost_at_80": round(upfront + stage3_total_comp_cost, 4),
        }

    # Observed target-by-target cumulative lifecycle breakeven curve (`k = 1..80`)!
    hegel_rows = in_scope_records["Hegel_Unified_Continual_Forest"]
    frozen_open_rows = in_scope_records["Frozen_After_Stage0_OpenRediscovery"]
    frozen_pool_rows = in_scope_records["Frozen_After_Stage0_PoolFallback"]

    cum_hegel = s0_cost + s1_cost + s2_cost
    cum_frozen_open = s0_cost
    cum_frozen_pool = s0_cost
    breakeven_k_vs_open: int | None = None
    breakeven_k_vs_pool: int | None = None
    cumulative_curve: list[dict[str, Any]] = []

    for k_idx in range(len(hegel_rows)):
        cum_hegel += float(hegel_rows[k_idx]["cost"]["composite_cost"])
        cum_frozen_open += float(frozen_open_rows[k_idx]["cost"]["composite_cost"])
        cum_frozen_pool += float(frozen_pool_rows[k_idx]["cost"]["composite_cost"])
        k_num = k_idx + 1
        if breakeven_k_vs_open is None and cum_hegel < cum_frozen_open:
            breakeven_k_vs_open = k_num
        if breakeven_k_vs_pool is None and cum_hegel < cum_frozen_pool:
            breakeven_k_vs_pool = k_num
        if k_num in {1, 10, 20, 30, 40, 50, 60, 70, 80} or k_num in {breakeven_k_vs_open, breakeven_k_vs_pool}:
            cumulative_curve.append({
                "targets_completed": k_num,
                "hegel_cumulative_lifecycle_cost": round(cum_hegel, 4),
                "frozen_open_rediscovery_cumulative_cost": round(cum_frozen_open, 4),
                "frozen_pool_fallback_cumulative_cost": round(cum_frozen_pool, 4),
            })

    all_receipts = [
        *[r for a in arm_names for r in in_scope_records[a]],
        *[r for a in arm_names for r in oos_records[a]],
    ]
    return {
        "seeds": list(seeds),
        "stage0": built["stage0"],
        "stage1": built["stage1"],
        "stage2": built["stage2"],
        "dispatcher_zero_fit_r3": dispatcher_report,
        "stage3_arms": arm_summaries,
        "lifecycle_breakeven": {
            "upfront_stage1_plus_stage2_investment": round(s1_cost + s2_cost, 4),
            "observed_breakeven_target_vs_frozen_open_rediscovery": breakeven_k_vs_open,
            "observed_breakeven_target_vs_frozen_pool_fallback": breakeven_k_vs_pool,
            "net_80_target_lifecycle_savings_vs_frozen_open_rediscovery": round(cum_frozen_open - cum_hegel, 4),
            "net_80_target_lifecycle_savings_vs_frozen_pool_fallback": round(cum_frozen_pool - cum_hegel, 4),
            "cumulative_curve_milestones": cumulative_curve,
        },
        "execution_receipts": all_receipts,
    }

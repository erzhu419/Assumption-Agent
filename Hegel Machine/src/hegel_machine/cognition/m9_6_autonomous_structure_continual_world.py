"""Hegel M9.6 Autonomous Guard & Multi-Parent Structure Induction + Cross-Domain Wormhole Continual World.

Addresses ALL 5 Scientific Next-Generation Requirements from `Hegel_M9_5_Audit` (§4.1–§4.5):
  1. **Zero Author-Hardcoded Regime Guards (`§4.1` + `§4.5`)**:
     - `Stage S` registers `C0` with `precondition_guard = None` (`structure_origin = 'unconstrained_initial_source'`).
     - When `C0` is empirically falsified on a counterexample batch (`Public Gate == False`), `gpt-6.2-sol` (`propose_and_contrastively_validate_autonomous_split`)
       must propose structured `(c0_guard, c1_guard, violated_component_id, dependent_upstream_mechanism_ids)` over a uniform 12-feature
       domain-agnostic structural observable bank containing **myopic & Frobenius-norm distractors**!
     - Accepted ONLY if it passes the **Contrastive Public Witness Check** (`c0_guard(W_pos) == TRUE`, `c0_guard(W_neg) == FALSE`,
       `c1_guard(W_neg) == TRUE`, `c1_guard(W_pos) == FALSE`, `0` private peek!) + **Perturbation Parent Sensitivity Check**!
  2. **True Multi-Parent (`parent_ids`) Diamond Mechanism Graph (`§4.2`)**:
     - Constitutive mechanisms depend directly on `parent_ids = (wing_a_id, wing_b_id)` (NO single-parent composite wrapper!), alongside
       single-wing monitors `aux_wing_a_monitor` (`(wing_a_id,)`) and `aux_wing_b_monitor` (`(wing_b_id,)`) plus `sentinel_shared_acoustic_root`!
     - `assemble_multiparent_parameter_payload` reads ONLY the parents declared in `binding.parent_parameter_mechanism_ids`!
  3. **Non-Markovian Unloaded Residual-Damage / Post-Vortex Memory Event `U` (`§4.3`)**:
     - On `U`, present-step hold stress (`Alpha`) or present 2D potential separability (`Beta`) looks 100% subcritical/laminar (matching `R`!),
       while **past history** (`past_stress` earlier overstress pulse / `past_vortex_circulation_trace` prior vortex circulation) carries latent
       regime memory! Directly tests whether the model-proposed guard captures true temporal state rather than myopic snapshot thresholding!
  4. **Cross-Domain Partial Substructure Reuse (`§4.4`, `allow_horizontal_wormholes = True`)**:
     - Alpha `T` (`T_shared_kkt`) requires coupling 1D viscoelastic Prony creep (`(tau_1, tau_2)`, `predicted_strain`) with a nuisance-profiled
       **Box-Simplex QP + Exact Dual KKT Certificate** (`allocation_weights`, `kkt_certificate`) — sharing the exact KKT optimization operator
       substructure with Beta while differing completely in forward physics!
  5. **Equal-Capability Head-to-Head Comparison (`Flat_Multiversion_Validated` vs `Full_Mechanism_Graph_Autonomous`, `max_pre_llm_probes = 1`)**:
     - Both arms receive 100% of `M9.5.1` engineering fixes (`F01` 1-hop canonical export, `F02` prefix `0444` `readonly_dependencies`,
       `F03` validated per-wing `2/3–1/3` holdout-MSE OLS & multi-version `{C0, C1, ...}` retention, `F14` `public_gate_verified`-only selection).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import CLEAN_SCIPY_SANDBOX_IMAGE
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TypedMechanism,
    evaluate_committed_task_model,
)
from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY
from .m9_4_continual_mechanism_binding_world import (
    M94BindingRegistry,
    MechanismExecutableBinding,
    grade_m94_artifact_map,
    validate_and_stage_readonly_dependencies_setup_py,
    verify_binding_identity_and_foreign_keys,
)
from .m9_5_equal_tool_continual_attribution_world import (
    ADAPTER_PRECALIBRATED_SENTINEL,
    execute_single_pre_llm_public_gate_probe_in_container,
    make_continual_archive_entry,
    render_parameterized_input_adapter,
    update_flat_validated_wing_parameter_snapshot_from_public_item,
)

M96_ARMS: tuple[str, ...] = (
    "Flat_Multiversion_Validated",
    "Full_Mechanism_Graph_Autonomous",
)


@dataclass(frozen=True)
class M96WorldSpec:
    world_id: str
    pair_cluster_id: str  # 'pair_01'..'pair_04' (disjoint cross-family Stage S forest pair)
    family: str  # 'alpha' | 'beta'
    split: str  # 'dev' | 'confirmation'
    world_seed: int
    schedule_events: tuple[str, ...]  # Causally valid 8-step permutation of ('N','G1','T','C','R','U','G2','M')
    v2_gains: tuple[float, float, float, float]
    v2_offsets: tuple[float, float, float, float]
    v3_wing_a_gains: tuple[float, float]
    v3_wing_a_offsets: tuple[float, float]


def _make_m96_world_spec(
    world_id: str,
    pair_cluster_id: str,
    family: str,
    split: str,
    world_seed: int,
    early_order: tuple[str, str, str],
    mid_order: tuple[str, str],
) -> M96WorldSpec:
    rng = np.random.default_rng(world_seed)
    sgn_v2 = np.array([1.0, -1.0, 1.0, -1.0]) if (world_seed // 1000) % 2 == 0 else np.array([-1.0, 1.0, -1.0, 1.0])
    v2_g = tuple(round(float(1.0 + sgn_v2[i] * rng.uniform(0.19, 0.31)), 4) for i in range(4))
    v2_b = tuple(round(float(sgn_v2[i] * rng.uniform(0.09, 0.16)), 4) for i in range(4))
    v3_ga = tuple(round(float(1.0 - sgn_v2[i] * rng.uniform(0.21, 0.33)), 4) for i in range(2))
    v3_ba = tuple(round(float(-sgn_v2[i] * rng.uniform(0.10, 0.17)), 4) for i in range(2))
    sched = (*early_order, "C", *mid_order, "G2", "M")
    assert set(sched) == {"N", "G1", "T", "C", "R", "U", "G2", "M"} and len(sched) == 8
    return M96WorldSpec(
        world_id=world_id,
        pair_cluster_id=pair_cluster_id,
        family=family,
        split=split,
        world_seed=world_seed,
        schedule_events=sched,
        v2_gains=v2_g,  # type: ignore[arg-type]
        v2_offsets=v2_b,  # type: ignore[arg-type]
        v3_wing_a_gains=v3_ga,  # type: ignore[arg-type]
        v3_wing_a_offsets=v3_ba,  # type: ignore[arg-type]
    )


DEV_M96_WORLD_SPECS: tuple[M96WorldSpec, ...] = (
    _make_m96_world_spec("dev_alpha_m96_00", "dev_pair_00", "alpha", "dev", 61000, ("T", "G1", "N"), ("R", "U")),
    _make_m96_world_spec("dev_beta_m96_00", "dev_pair_00", "beta", "dev", 62000, ("N", "T", "G1"), ("U", "R")),
)

# Balanced bidirectional schedules (`("R", "U")` AND `("U", "R")`, `("G1", "N")` AND `("N", "G1")`) across 4 disjoint pairs (`8` fresh worlds)
CONFIRMATION_M96_WORLD_SPECS: tuple[M96WorldSpec, ...] = (
    _make_m96_world_spec("alpha_m96_w01", "pair_01", "alpha", "confirmation", 971000, ("T", "N", "G1"), ("R", "U")),
    _make_m96_world_spec("alpha_m96_w02", "pair_02", "alpha", "confirmation", 972000, ("N", "T", "G1"), ("U", "R")),
    _make_m96_world_spec("alpha_m96_w03", "pair_03", "alpha", "confirmation", 973000, ("N", "G1", "T"), ("R", "U")),
    _make_m96_world_spec("alpha_m96_w04", "pair_04", "alpha", "confirmation", 974000, ("T", "G1", "N"), ("U", "R")),
    _make_m96_world_spec("beta_m96_w01", "pair_01", "beta", "confirmation", 981000, ("T", "N", "G1"), ("R", "U")),
    _make_m96_world_spec("beta_m96_w02", "pair_02", "beta", "confirmation", 982000, ("N", "T", "G1"), ("U", "R")),
    _make_m96_world_spec("beta_m96_w03", "pair_03", "beta", "confirmation", 983000, ("N", "G1", "T"), ("R", "U")),
    _make_m96_world_spec("beta_m96_w04", "pair_04", "beta", "confirmation", 984000, ("T", "G1", "N"), ("U", "R")),
)


# ============================================================================
# 1. Physics Generators (`Alpha` & `Beta` with Non-Markovian `U` & Shared KKT `T`)
# ============================================================================

_M96_ALPHA_GEN_PY = _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY + r'''
def _solve_alpha_box_simplex_exact(S_tilde, r_y, lam_w, u_box, b_tot):
    from scipy.optimize import minimize
    M = S_tilde.shape[1]
    H = S_tilde.T @ S_tilde + float(lam_w) * np.eye(M)
    f = S_tilde.T @ r_y
    w0 = np.full(M, float(b_tot) / M, dtype=float)
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        w0,
        jac=lambda w: H @ w - f,
        bounds=[(0.0, float(u_box))] * M,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - float(b_tot), "jac": lambda w: np.ones(M)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 400},
    )
    w = np.clip(res.x, 0.0, float(u_box))
    lo, hi = float(np.min(w) - u_box), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, float(u_box)))) > float(b_tot):
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, float(u_box))
    for _ in range(8):
        free = (w > 1e-11) & (w < float(u_box) - 1e-11)
        f_idx = np.flatnonzero(free)
        if f_idx.size >= 2:
            rem = float(b_tot) - float(np.sum(w[~free]))
            Hf = H[np.ix_(f_idx, f_idx)]
            rhs_f = f[f_idx] - H[np.ix_(f_idx, np.flatnonzero(~free))] @ w[~free]
            aug = np.block([[Hf, -np.ones((f_idx.size, 1))], [np.ones((1, f_idx.size)), np.zeros((1, 1))]])
            sol = np.linalg.solve(aug, np.concatenate([rhs_f, [rem]]))
            wf = sol[:-1]
            if np.all((wf >= -1e-12) & (wf <= float(u_box) + 1e-12)):
                w[f_idx] = np.clip(wf, 0.0, float(u_box))
                break
    r_vec = S_tilde @ w - r_y
    j_val = 0.5 * float(r_vec @ r_vec) + 0.5 * float(lam_w) * float(w @ w)
    g = H @ w - f
    cand_nus = np.unique(g)
    best_nu = float(cand_nus[0])
    best_lb = -1e30
    for nu_c in cand_nus:
        lb_c = j_val - float(g @ w) + float(nu_c) * float(b_tot) + float(u_box) * float(np.sum(np.minimum(0.0, g - float(nu_c))))
        if lb_c > best_lb:
            best_lb = lb_c
            best_nu = float(nu_c)
    return w, {
        "weights": w.tolist(),
        "nu_sum": float(best_nu),
        "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
        "lam_upper": np.maximum(0.0, best_nu - g).tolist(),
        "primal_objective": float(j_val),
        "dual_lower_bound": float(min(best_lb, j_val)),
    }

def _gen_m96_alpha_cohort(
    seed,
    taus_pair,
    theta_vec,
    *,
    stress_gains=(1.0, 1.0, 1.0, 1.0),
    stress_biases=(0.0, 0.0, 0.0, 0.0),
    cal_mode="full_40",
    n_train=200,
    n_test=80,
    H=18,
    K=10,
    dt_mode="base_var",
    preload_scale=0.0,
    yield_stress=2.95,
    vp_theta=(0.0, 0.0),
    unloaded_residual_damage_U=False,
    shared_kkt_transfer_T=False,
    glitch_frac=0.08,
):
    rng = np.random.default_rng(seed)
    taus = np.asarray(taus_pair, dtype=float)
    theta = np.asarray(theta_vec, dtype=float)
    s_gains = np.asarray(stress_gains, dtype=float)
    s_biases = np.asarray(stress_biases, dtype=float)

    if dt_mode == "base_var":
        past_dt = np.round(rng.uniform(0.18, 0.85, size=H), 4)
        eval_times = np.round(np.linspace(0.25, 5.20, K), 4)
    else:
        past_dt = np.round(np.geomspace(0.09, 1.25, num=H) * rng.uniform(0.88, 1.12, size=H), 4)
        eval_times = np.round(np.geomspace(0.18, 6.40, num=K), 4)

    if cal_mode == "full_40":
        cal_labs = np.repeat(np.arange(4, dtype=int), 10)
        cal_cmd = rng.uniform(-2.2, 2.4, size=40)
        cal_true = s_gains[cal_labs] * cal_cmd + s_biases[cal_labs] + rng.normal(scale=1e-4, size=40)
    elif cal_mode == "sparse_spot_lab0":
        cal_labs = np.array([0, 0], dtype=int)
        cal_cmd = np.array([-1.35, 1.45], dtype=float)
        cal_true = s_gains[cal_labs] * cal_cmd + s_biases[cal_labs] + rng.normal(scale=1e-4, size=2)
    elif cal_mode == "scoped_wing_a_plus_emi_wing_b":
        la = np.repeat(np.array([0, 1], dtype=int), 12)
        ca = rng.uniform(-2.1, 2.3, size=24)
        ta = s_gains[la] * ca + s_biases[la] + rng.normal(scale=1e-4, size=24)
        lb = np.array([2, 3, 2, 3, 2, 3], dtype=int)
        cb = np.array([-1.65, 1.55, 1.25, -1.35, -0.95, 1.15], dtype=float)
        emi = np.array([4.85, -4.70, 4.55, -4.60, 0.0, 0.0], dtype=float)
        tb = s_gains[lb] * cb + s_biases[lb] + emi
        cal_labs = np.concatenate([la, lb])
        cal_cmd = np.concatenate([ca, cb])
        cal_true = np.concatenate([ta, tb])
    else:
        raise ValueError(f"Unknown cal_mode: {cal_mode}")

    p_probs = np.array([[0.42, 0.33, 0.25], [0.22, 0.48, 0.30], [0.30, 0.24, 0.46], [0.26, 0.36, 0.38]])
    gains = np.array([0.82, 1.28, 0.94, 1.42])
    offsets = np.array([-0.65, 0.48, -0.25, 0.80])

    def _make_panel(n_rows):
        labs = rng.integers(0, 4, size=n_rows)
        strata = np.array([rng.choice(3, p=p_probs[c]) for c in labs], dtype=int)
        labs[:16] = np.tile(np.arange(4, dtype=int), 4)
        Z = rng.normal(loc=0.25 * (strata[:, None] - 1.0), scale=1.0, size=(n_rows, 4))
        R_lat = rng.normal(size=(n_rows, 32))
        X_raw = offsets[labs, None] + gains[labs, None] * Z
        R_raw = offsets[labs, None] + gains[labs, None] * R_lat
        X_raw[rng.random(X_raw.shape) < 0.025] = np.nan
        R_raw[rng.random(R_raw.shape) < 0.040] = np.nan
        return labs, strata, Z, X_raw, R_raw

    tr_lab, tr_st, tr_Z, tr_X, tr_R = _make_panel(n_train)
    te_lab, te_st, te_Z, te_X, te_R = _make_panel(n_test)

    ys = float(yield_stress)
    has_vp = abs(float(vp_theta[0])) > 0.0 or abs(float(vp_theta[1])) > 0.0

    def _sim(Z_lat, labs_arr, *, is_test_split=False, add_glitches=False):
        N = Z_lat.shape[0]
        init_mem = rng.normal(scale=preload_scale, size=(N, 2)) if preload_scale > 0.0 else np.zeros((N, 2), dtype=float)
        hist_true = rng.normal(scale=1.45, size=(N, H))
        for s in range(1, H):
            hist_true[:, s] = 0.52 * hist_true[:, s - 1] + math.sqrt(1.0 - 0.52 ** 2) * hist_true[:, s]
        s0 = rng.normal(scale=0.35, size=N)
        hold_true = rng.uniform(-2.05, 2.15, size=N)

        if has_vp and unloaded_residual_damage_U and is_test_split:
            # NON-MARKOVIAN UNLOADED RESIDUAL DAMAGE (`U`, `§4.3`):
            # Present hold stress `&` trailing history steps `[-4:]` are 100% SUB-YIELD (`<= 0.72 * ys`),
            # indistinguishable from `R` under any myopic snapshot guard — EXCEPT early history `[:8]` experienced
            # heavy overstress excursion (`[ys + 0.65, ys + 1.45]`), accumulating permanent plastic-work memory `D_hist > 0`!
            sgn_h = np.where(rng.random(N) < 0.52, 1.0, -1.0)
            hold_true = sgn_h * rng.uniform(0.42 * ys, 0.72 * ys, size=N)
            hist_true = np.clip(hist_true * 0.45, -0.70 * ys, 0.70 * ys)
            sgn_early = np.where(rng.random(N) < 0.55, 1.0, -1.0)
            hist_true[:, :8] = sgn_early[:, None] * rng.uniform(ys + 0.65, ys + 1.45, size=(N, 8))
        elif has_vp:
            hi_idx = np.arange(N // 2)
            sgn = np.where(rng.random(len(hi_idx)) < 0.55, 1.0, -1.0)
            hold_true[hi_idx] = sgn * rng.uniform(ys + 0.35, ys + 1.30, size=len(hi_idx))
            hist_true[hi_idx, -6:] = sgn[:, None] * rng.uniform(ys + 0.25, ys + 1.15, size=(len(hi_idx), 6))
            # Quarter of training rows carry unloaded early-history overstress so BOTH `vp_theta[0]` and `vp_theta[1]` are strongly identifiable!
            u_idx = np.arange(N // 2, (3 * N) // 4)
            sgn_u = np.where(rng.random(len(u_idx)) < 0.5, 1.0, -1.0)
            hold_true[u_idx] = sgn_u * rng.uniform(0.45 * ys, 0.74 * ys, size=len(u_idx))
            hist_true[u_idx] = np.clip(hist_true[u_idx] * 0.45, -0.72 * ys, 0.72 * ys)
            hist_true[u_idx, :7] = sgn_u[:, None] * rng.uniform(ys + 0.55, ys + 1.35, size=(len(u_idx), 7))
            sub_idx = np.arange((3 * N) // 4, N)
            hold_true[sub_idx] = rng.uniform(-0.82 * ys, 0.82 * ys, size=len(sub_idx))
            hist_true[sub_idx] = np.clip(hist_true[sub_idx] * 0.48, -0.82 * ys, 0.82 * ys)
        else:
            hold_true = np.clip(hold_true, -0.82 * ys, 0.82 * ys)
            hist_true = np.clip(hist_true, -0.82 * ys, 0.82 * ys)

        hist_cmd = (hist_true - s_biases[labs_arr, None]) / s_gains[labs_arr, None]
        hold_cmd = (hold_true - s_biases[labs_arr]) / s_gains[labs_arr]

        z_cur = init_mem.copy()
        for s in range(H):
            dec = np.exp(-past_dt[s] / taus)
            z_cur = z_cur * dec[None, :] + (1.0 - dec[None, :]) * hist_true[:, s:s + 1]

        gap1 = hold_true - z_cur[:, 0]
        gap2 = hold_true - z_cur[:, 1]
        e1 = 1.0 - np.exp(-eval_times[None, :] / taus[0])
        e2 = 1.0 - np.exp(-eval_times[None, :] / taus[1])
        drift = theta[2] + Z_lat @ theta[3:7]

        ov_hold = (np.maximum(0.0, np.abs(hold_true) - ys) ** 2) * np.sign(hold_true)
        ov_hist = np.sum((np.maximum(0.0, np.abs(hist_true) - ys) ** 2) * past_dt[None, :], axis=1)
        vp_rate_hold = float(vp_theta[0]) * ov_hold
        vp_dmg_surge = float(vp_theta[1]) * ov_hist * hold_true

        clean = (
            s0[:, None]
            + theta[0] * gap1[:, None] * e1
            + theta[1] * gap2[:, None] * e2
            + (drift + vp_rate_hold)[:, None] * eval_times[None, :]
            + vp_dmg_surge[:, None] * e1
        )
        true_rate0 = theta[0] * gap1 / taus[0] + theta[1] * gap2 / taus[1] + drift + vp_rate_hold + vp_dmg_surge / taus[0]
        rate_obs = true_rate0 + rng.normal(scale=0.010, size=N)
        obs = clean + rng.normal(scale=0.009, size=clean.shape)
        if add_glitches and glitch_frac > 0.0:
            n_gl = max(1, int(float(glitch_frac) * obs.size))
            g_flat = rng.choice(obs.size, size=n_gl, replace=False)
            obs.ravel()[g_flat] += rng.choice([-1.0, 1.0], size=n_gl) * rng.uniform(0.75, 1.20, size=n_gl)
        return init_mem, hist_cmd, s0, hold_cmd, rate_obs, true_rate0, clean, obs, hold_true, hist_true, ov_hist

    tr_im, tr_hc, tr_s0, tr_h0c, tr_ro, _, _, tr_ob, _, _, _ = _sim(tr_Z, tr_lab, is_test_split=False, add_glitches=True)
    te_im, te_hc, te_s0, te_h0c, te_ro, te_rt, te_cl, _, te_ht, te_hist_t, te_ov_hist = _sim(te_Z, te_lab, is_test_split=True, add_glitches=False)

    ds = {
        "past_dt": past_dt,
        "eval_times": eval_times,
        "yield_stress": ys,
        "calibration_load_cells": {"lab_id": cal_labs, "cmd_stress": cal_cmd, "true_stress": cal_true},
        "train": {
            "lab_id": tr_lab, "stratum": tr_st, "coupon_features_raw": tr_X, "ref_coupons": tr_R,
            "past_stress_cmd": tr_hc, "strain_0": tr_s0, "stress_hold_cmd": tr_h0c, "instant_rate_obs": tr_ro,
            "strain_curves_obs": tr_ob,
        },
        "test": {
            "lab_id": te_lab, "stratum": te_st, "coupon_features_raw": te_X, "ref_coupons": te_R,
            "past_stress_cmd": te_hc, "strain_0": te_s0, "stress_hold_cmd": te_h0c, "instant_rate_obs": te_ro,
            "residual_plastic_state": np.zeros(len(te_lab), dtype=float),
        },
    }
    gt = {
        "taus": taus,
        "theta": theta,
        "vp_theta": np.asarray(vp_theta, dtype=float),
        "stress_gains": s_gains,
        "stress_biases": s_biases,
        "te_clean": te_cl,
        "te_true_rate0": te_rt,
        "te_hold_true": te_ht,
        "te_hist_true": te_hist_t,
    }
    if preload_scale > 0.0:
        ds["train"]["initial_memory_states"] = tr_im
        ds["test"]["initial_memory_states"] = te_im

    if shared_kkt_transfer_T:
        # Cross-Domain Box-Simplex KKT Compliance Allocation (`§4.4`):
        # Shares exact nuisance-profiled Box-Simplex QP + Dual KKT Certificate operator with Beta!
        M_al = 12
        P_z = 5
        S_alloc_modes = rng.normal(scale=0.55, size=(n_test, M_al))
        Z_nuis = rng.normal(size=(n_test, P_z))
        lam_z = 0.65
        lam_w = 0.04
        u_box = 1.0
        b_tot = 2.60
        comp_gain_true = 1.0 + 0.25 * np.tanh(te_cl[:, -1] - te_s0)
        S_mat = S_alloc_modes * comp_gain_true[:, None]
        w_true = np.zeros(M_al, dtype=float)
        raw_w = np.array([0.86, 0.74, 0.56, 0.44], dtype=float)
        w_true[[1, 4, 7, 10]] = raw_w * (b_tot / float(np.sum(raw_w)))
        gamma_z = rng.normal(scale=0.55, size=P_z)
        y_alloc_obs = S_mat @ w_true + Z_nuis @ gamma_z + rng.normal(scale=0.010, size=n_test)
        Pz = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(P_z), Z_nuis.T)
        r_al = y_alloc_obs - Z_nuis @ (Pz @ y_alloc_obs)
        S_til = S_mat - Z_nuis @ (Pz @ S_mat)
        w_opt, cert_opt = _solve_alpha_box_simplex_exact(S_til, r_al, lam_w, u_box, b_tot)
        ds["S_alloc_modes"] = S_alloc_modes
        ds["Z_nuisance"] = Z_nuis
        ds["y_alloc_obs"] = y_alloc_obs
        ds["nuisance_ridge"] = lam_z
        ds["source_ridge"] = lam_w
        ds["u_box"] = u_box
        ds["total_budget"] = b_tot
        gt["w_alloc_opt"] = w_opt
        gt["cert_alloc_opt"] = cert_opt
    return ds, gt
'''


_M96_BETA_GEN_PY = _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY + r'''
def _build_1d_fvm_advdiff_robin(L, n, kappa, vel, robin_gamma, diag_add=None):
    h = float(L) / int(n)
    kap = np.asarray(kappa, dtype=float).ravel()
    v = np.asarray(vel, dtype=float).ravel()
    rg = float(robin_gamma)
    A = np.zeros((n, n), dtype=float)
    kh = (2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])) / (h * h)
    idx = np.arange(n - 1)
    A[idx, idx] += kh
    A[idx + 1, idx + 1] += kh
    A[idx, idx + 1] -= kh
    A[idx + 1, idx] -= kh
    A[0, 0] += (rg * kap[0]) / (h * (rg * 0.5 * h + kap[0]))
    A[n - 1, n - 1] += (rg * kap[n - 1]) / (h * (rg * 0.5 * h + kap[n - 1]))
    for i in range(n):
        vi = float(v[i])
        if vi >= 0.0:
            A[i, i] += vi / h
            if i > 0:
                A[i, i - 1] -= vi / h
        else:
            A[i, i] += (-vi) / h
            if i < n - 1:
                A[i, i + 1] -= (-vi) / h
    if diag_add is not None:
        A += np.diag(np.asarray(diag_add, dtype=float).ravel())
    return A

def _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, b_tot):
    from scipy.optimize import minimize
    M = S_tilde.shape[1]
    H = S_tilde.T @ S_tilde + float(lam_w) * np.eye(M)
    f = S_tilde.T @ r_y
    w0 = np.full(M, float(b_tot) / M, dtype=float)
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        w0,
        jac=lambda w: H @ w - f,
        bounds=[(0.0, float(u_box))] * M,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - float(b_tot), "jac": lambda w: np.ones(M)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 400},
    )
    w = np.clip(res.x, 0.0, float(u_box))
    lo, hi = float(np.min(w) - u_box), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, float(u_box)))) > float(b_tot):
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, float(u_box))
    for _ in range(8):
        free = (w > 1e-11) & (w < float(u_box) - 1e-11)
        f_idx = np.flatnonzero(free)
        if f_idx.size >= 2:
            rem = float(b_tot) - float(np.sum(w[~free]))
            Hf = H[np.ix_(f_idx, f_idx)]
            rhs_f = f[f_idx] - H[np.ix_(f_idx, np.flatnonzero(~free))] @ w[~free]
            aug = np.block([[Hf, -np.ones((f_idx.size, 1))], [np.ones((1, f_idx.size)), np.zeros((1, 1))]])
            sol = np.linalg.solve(aug, np.concatenate([rhs_f, [rem]]))
            wf = sol[:-1]
            if np.all((wf >= -1e-12) & (wf <= float(u_box) + 1e-12)):
                w[f_idx] = np.clip(wf, 0.0, float(u_box))
                break
    r_vec = S_tilde @ w - r_y
    j_val = 0.5 * float(r_vec @ r_vec) + 0.5 * float(lam_w) * float(w @ w)
    g = H @ w - f
    cand_nus = np.unique(g)
    best_nu = float(cand_nus[0])
    best_lb = -1e30
    for nu_c in cand_nus:
        lb_c = j_val - float(g @ w) + float(nu_c) * float(b_tot) + float(u_box) * float(np.sum(np.minimum(0.0, g - float(nu_c))))
        if lb_c > best_lb:
            best_lb = lb_c
            best_nu = float(nu_c)
    return w, {
        "weights": w.tolist(),
        "nu_sum": float(best_nu),
        "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
        "lam_upper": np.maximum(0.0, best_nu - g).tolist(),
        "primal_objective": float(j_val),
        "dual_lower_bound": float(min(best_lb, j_val)),
    }

def _gen_m96_beta_study(
    seed,
    *,
    nx=40,
    ny=36,
    K_mu=8,
    M=12,
    vel_sign=1.0,
    robin_pair=(1.35, 1.15),
    bank_gains=(1.0, 1.0, 1.0, 1.0),
    bank_offsets=(0.0, 0.0, 0.0, 0.0),
    cal_mode="full_48",
    multi_budget=False,
    vortex_regime=False,
    unloaded_post_vortex_memory_U=False,
    obs_noise_scale=0.011,
):
    rng = np.random.default_rng(seed)
    b_gains = np.asarray(bank_gains, dtype=float)
    b_offs = np.asarray(bank_offsets, dtype=float)

    if cal_mode == "full_48":
        cal_banks = np.repeat(np.arange(4, dtype=int), 12)
        cal_ref = rng.uniform(-1.2, 1.6, size=48)
        cal_raw = b_gains[cal_banks] * cal_ref + b_offs[cal_banks] + rng.normal(scale=1e-8, size=48)
    elif cal_mode == "sparse_spot_bank0":
        cal_banks = np.array([0, 0], dtype=int)
        cal_ref = np.array([-0.95, 1.25], dtype=float)
        cal_raw = b_gains[cal_banks] * cal_ref + b_offs[cal_banks] + rng.normal(scale=1e-8, size=2)
    elif cal_mode == "scoped_wing_a_plus_emi_wing_b":
        ba = np.repeat(np.array([0, 1], dtype=int), 12)
        ra = rng.uniform(-1.2, 1.6, size=24)
        ya = b_gains[ba] * ra + b_offs[ba] + rng.normal(scale=1e-8, size=24)
        bb = np.array([2, 3, 2, 3, 2, 3], dtype=int)
        rb = np.array([-1.15, 1.35, 0.95, -1.05, -0.75, 0.85], dtype=float)
        emi = np.array([3.60, -3.50, 3.30, -3.40, 0.0, 0.0], dtype=float)
        yb = b_gains[bb] * rb + b_offs[bb] + emi
        cal_banks = np.concatenate([ba, bb])
        cal_ref = np.concatenate([ra, rb])
        cal_raw = np.concatenate([ya, yb])
    else:
        raise ValueError(f"Unknown cal_mode: {cal_mode}")

    Lx, Ly = 2.2, 1.9
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    kappa_x = 0.42 + 0.15 * np.cos(math.pi * xs / Lx) + 0.04 * rng.uniform(-1.0, 1.0, size=nx)
    kappa_y = 0.38 + 0.14 * np.sin(math.pi * ys / Ly) + 0.04 * rng.uniform(-1.0, 1.0, size=ny)
    vel_x = float(vel_sign) * (0.48 + 0.12 * np.sin(2.0 * math.pi * xs / Lx))
    vel_y = float(vel_sign) * (-0.42 + 0.10 * np.cos(2.0 * math.pi * ys / Ly))
    rx_base = np.round(0.12 + 0.05 * np.cos(math.pi * xs / Lx), 6)
    ry_base = np.round(0.10 + 0.04 * np.sin(math.pi * ys / Ly), 6)
    mu_shifts = np.round(np.geomspace(0.08, 3.40, num=K_mu), 4)

    X2, Y2 = np.meshgrid(xs, ys, indexing="ij")
    centers_x = rng.uniform(0.18 * Lx, 0.82 * Lx, size=M)
    centers_y = rng.uniform(0.18 * Ly, 0.82 * Ly, size=M)
    source_modes = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        g_m = np.exp(-((X2 - centers_x[m]) ** 2 + (Y2 - centers_y[m]) ** 2) / (2.0 * (0.21 ** 2)))
        source_modes[m] = g_m / max(float(np.linalg.norm(g_m)), 1e-12)

    N_sens = 36
    si = rng.integers(2, nx - 2, size=N_sens)
    sj = rng.integers(2, ny - 2, size=N_sens)
    sensor_ij = np.column_stack([si, sj])
    sens_bank = np.arange(N_sens, dtype=int) % 4
    row_bank_id = np.tile(sens_bank, K_mu)

    # Unified temporal circulation history trace `past_vortex_circulation_trace` (`shape (12,)` across ALL Beta tasks!)
    if vortex_regime and not unloaded_post_vortex_memory_U:
        # Active Supercritical Vortex Regime (`C` / `M`): non-separable 2D potential + active circulation `> 1.0` -> `1.45 * kappa`
        vortex_trace = np.round(np.linspace(0.85, 1.55, 12) + rng.uniform(-0.04, 0.04, size=12), 4)
        V_2d = (
            4.6 * np.exp(-((X2 - 0.52 * Lx) ** 2 + (Y2 - 0.48 * Ly) ** 2) / 0.14) * np.sin(2.6 * math.pi * X2 / Lx) * np.sin(2.4 * math.pi * Y2 / Ly)
            + 3.2 * np.exp(-((X2 - 0.35 * Lx - 0.45 * Y2) ** 2) / 0.08)
        )
        V_2d = V_2d - float(np.min(V_2d)) + 0.15
        eddy_active = True
    elif unloaded_post_vortex_memory_U:
        # NON-MARKOVIAN POST-VORTEX EDDY MEMORY (`U`, `§4.3`):
        # Present `V_reaction_2d` has relaxed back to 100% ADDITIVELY SEPARABLE `v_x(x) + v_y(y)` (`nonseparable_ratio == 0.0`!)
        # AND terminal `vortex_trace[-1] == 0.04` (< 0.25), EXCEPT earlier `vortex_trace[:7]` exceeded critical circulation `0.50`
        # (`peak ~ 1.38`), sustaining lingering turbulent eddy diffusivity `1.45 * kappa`!
        vortex_trace = np.array([1.22, 1.38, 1.31, 1.18, 0.96, 0.78, 0.62, 0.21, 0.14, 0.09, 0.06, 0.04], dtype=float)
        vx_part = 0.28 + 0.16 * np.cos(2.0 * math.pi * xs / Lx)
        vy_part = 0.24 + 0.13 * np.sin(2.0 * math.pi * ys / Ly)
        V_2d = vx_part[:, None] + vy_part[None, :]
        eddy_active = True
    else:
        # Laminar Separable Regime (`S, N, G1, T, R, G2`): `nonseparable_ratio == 0.0` AND `max(vortex_trace) <= 0.12 < 0.50` -> `1.00 * kappa`
        vortex_trace = np.round(rng.uniform(0.02, 0.11, size=12), 4)
        vx_part = 0.28 + 0.16 * np.cos(2.0 * math.pi * xs / Lx)
        vy_part = 0.24 + 0.13 * np.sin(2.0 * math.pi * ys / Ly)
        V_2d = vx_part[:, None] + vy_part[None, :]
        eddy_active = False

    if not eddy_active:
        ax_eff = rx_base + np.mean(V_2d, axis=1)
        by_eff = ry_base + (np.mean(V_2d, axis=0) - float(np.mean(V_2d)))
        Ax = _build_1d_fvm_advdiff_robin(Lx, nx, kappa_x, vel_x, robin_pair[0], diag_add=ax_eff)
        By = _build_1d_fvm_advdiff_robin(Ly, ny, kappa_y, vel_y, robin_pair[1], diag_add=by_eff)
        lx, Vx = np.linalg.eig(Ax)
        ly, Vy = np.linalg.eig(By)
        Phi_hat = np.einsum("ia,mab,bj->mij", np.linalg.inv(Vx), source_modes, np.linalg.inv(Vy).T, optimize=True)
        denom = lx[:, None, None] + ly[None, :, None] + mu_shifts[None, None, :]
        U_all = np.real(np.einsum("ai,mijk,jb->mkab", Vx, Phi_hat[:, :, :, None] / denom[None, :, :, :], Vy.T, optimize=True))
    else:
        # Notice: `kron(Ax(1.45*kappa, rx), Iy) + kron(Ix, By(1.45*kappa, ry)) + diag(V_2d.ravel())` is exact for BOTH
        # non-separable `C/M` AND separable post-vortex `U` whenever `1.45 * kappa` eddy diffusivity is active!
        from scipy import sparse
        from scipy.sparse.linalg import splu
        Ax_v = _build_1d_fvm_advdiff_robin(Lx, nx, 1.45 * kappa_x, vel_x, robin_pair[0], diag_add=rx_base)
        By_v = _build_1d_fvm_advdiff_robin(Ly, ny, 1.45 * kappa_y, vel_y, robin_pair[1], diag_add=ry_base)
        K_base = (
            sparse.kron(sparse.csc_matrix(Ax_v), sparse.eye(ny, format="csc"), format="csc")
            + sparse.kron(sparse.eye(nx, format="csc"), sparse.csc_matrix(By_v), format="csc")
            + sparse.diags(V_2d.ravel(), format="csc")
        )
        U_all = np.zeros((M, K_mu, nx, ny), dtype=float)
        rhs_mat = source_modes.reshape(M, nx * ny).T
        for k in range(K_mu):
            lu_k = splu((K_base + float(mu_shifts[k]) * sparse.eye(nx * ny, format="csc")).tocsc())
            U_all[:, k, :, :] = lu_k.solve(rhs_mat).T.reshape(M, nx, ny)

    S_stack = np.vstack([U_all[:, k, si, sj].T for k in range(K_mu)])
    n_rows = S_stack.shape[0]
    P_z = 5
    Z_nuis = rng.normal(size=(n_rows, P_z))
    lam_z = 0.65
    lam_w = 0.04
    u_box = 1.0
    b_tot = 2.60

    w_true = np.zeros(M, dtype=float)
    raw_w = np.array([0.88, 0.72, 0.58, 0.42], dtype=float)
    w_true[[1, 4, 7, 10]] = raw_w * (b_tot / float(np.sum(raw_w)))
    gamma_true = rng.normal(scale=0.65, size=P_z)
    y_true_cal = S_stack @ w_true + Z_nuis @ gamma_true + rng.normal(scale=float(obs_noise_scale), size=n_rows)
    y_obs_raw = b_gains[row_bank_id] * y_true_cal + b_offs[row_bank_id]

    Pz_mat = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(P_z), Z_nuis.T)
    r_y = y_true_cal - Z_nuis @ (Pz_mat @ y_true_cal)
    S_tilde = S_stack - Z_nuis @ (Pz_mat @ S_stack)

    study = {
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny,
        "kappa_x": kappa_x, "vel_x": vel_x, "robin_x": float(robin_pair[0]), "reaction_x": rx_base,
        "kappa_y": kappa_y, "vel_y": vel_y, "robin_y": float(robin_pair[1]), "reaction_y": ry_base,
        "V_reaction_2d": V_2d,
        "past_vortex_circulation_trace": vortex_trace,
        "critical_vortex_circulation": 0.50,
        "mu_shifts": mu_shifts,
        "source_modes": source_modes,
        "sensor_ij": sensor_ij,
        "row_bank_id": row_bank_id,
        "sensor_calibration_sweeps": {"bank_id": cal_banks, "ref_signal": cal_ref, "raw_readout": cal_raw},
        "Z_nuisance": Z_nuis,
        "y_obs_raw": y_obs_raw,
        "nuisance_ridge": lam_z,
        "source_ridge": lam_w,
        "u_box": u_box,
    }
    if multi_budget:
        budget_grid = np.array([1.40, 2.60, 3.80], dtype=float)
        study["budget_grid"] = budget_grid
        w_grid, certs_grid = [], []
        for bg in budget_grid:
            wb, cb = _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, float(bg))
            w_grid.append(wb); certs_grid.append(cb)
        term_field = np.tensordot(w_grid[1], U_all[:, -1, :, :], axes=(0, 0))
        gt = {
            "S_tilde": S_tilde, "r_y": r_y,
            "weights_by_budget": np.vstack(w_grid), "certs_by_budget": certs_grid,
            "terminal_field_mid_budget": term_field,
        }
    else:
        study["total_budget"] = b_tot
        w_opt, cert_opt = _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, b_tot)
        term_f = np.tensordot(w_opt, U_all[:, -1, :, :], axes=(0, 0))
        gt = {
            "S_tilde": S_tilde, "r_y": r_y,
            "w_opt": w_opt, "cert_opt": cert_opt,
            "terminal_field_A": term_f,
        }
    return study, gt
'''


# ============================================================================
# 2. Task Builders (`Neutral Anonymous Task IDs`, Unified Contracts + `U` + `T_kkt`)
# ============================================================================


def _build_m96_alpha_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
    is_shared_kkt_T: bool = False,
) -> HoldoutOODTask:
    artifact_name = f"{short_name}_solver.py"
    entry_name = f"solve_{short_name}"
    pkl_name = f"public_{short_name}.pkl"
    mod_stem = Path(artifact_name).stem

    if not is_shared_kkt_T:
        schema_keys = '(("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("viscoplastic_weights", (2,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,)))'
        val_extra = ""
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau_rel = 0.0; max_mae = 0.0; max_rate_mae = 0.0; max_vp_err = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "hereditary_weights", "viscoplastic_weights", "predicted_strain", "initial_rate_pred"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    hw = np.asarray(out["hereditary_weights"], dtype=float)
    vw = np.asarray(out["viscoplastic_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ir = np.asarray(out["initial_rate_pred"], dtype=float)
    if et.shape != (2,) or hw.shape != (7,) or vw.shape != (2,) or ps.shape != gt["te_clean"].shape or ir.shape != gt["te_true_rate0"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(hw)) and np.all(np.isfinite(vw)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ir)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    if float(np.linalg.norm(hw[:2])) <= 1e-4 or float(np.min(hw[:2])) <= 1e-4:
        valid = False; rows.append({{"seed": sd, "error": "degenerate_prony_weights"}}); continue
    ys_v = float(ds["yield_stress"])
    z_m = np.zeros((len(ds["test"]["lab_id"]), 2), dtype=float)
    for s_i, dts in enumerate(ds["past_dt"]):
        dec = np.exp(-dts / et); z_m = z_m * dec[None, :] + (1.0 - dec[None, :]) * gt["te_hist_true"][:, s_i:s_i+1]
    gap_m = gt["te_hold_true"][:, None] - z_m
    ov_h = (np.maximum(0.0, np.abs(gt["te_hold_true"]) - ys_v) ** 2) * np.sign(gt["te_hold_true"])
    d_hi = np.sum((np.maximum(0.0, np.abs(gt["te_hist_true"]) - ys_v) ** 2) * ds["past_dt"][None, :], axis=1)
    vp_c = vw[0] * ov_h[:, None] * ds["eval_times"][None, :] + vw[1] * (d_hi * gt["te_hold_true"])[:, None] * (-np.expm1(-ds["eval_times"][None, :] / et[0]))
    vp_r = vw[0] * ov_h + vw[1] * (d_hi * gt["te_hold_true"]) / et[0]
    pr_p = hw[0] * gap_m[:, 0:1] * (-np.expm1(-ds["eval_times"][None, :] / et[0])) + hw[1] * gap_m[:, 1:2] * (-np.expm1(-ds["eval_times"][None, :] / et[1]))
    rem_c = (ps - ds["test"]["strain_0"][:, None]) - pr_p - vp_c
    rem_r = ir - (hw[0] * gap_m[:, 0] / et[0] + hw[1] * gap_m[:, 1] / et[1]) - vp_r
    if float(np.max(np.abs(rem_c - rem_r[:, None] * ds["eval_times"][None, :]))) > 0.08:
        valid = False; rows.append({{"seed": sd, "error": "weights_prediction_handoff_mismatch"}}); continue
    if float(np.max(np.abs(hw - gt["theta"]))) > 0.14:
        valid = False; rows.append({{"seed": sd, "error": "tail_coefficients_gt_handoff_mismatch"}}); continue
    vp_err = float(np.linalg.norm(vw - gt["vp_theta"]) / max(float(np.linalg.norm(gt["vp_theta"])), 1.0))
    tau_rel = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    s_mae = float(np.mean(np.abs(ps - gt["te_clean"])))
    r_mae = float(np.mean(np.abs(ir - gt["te_true_rate0"])))
    max_tau_rel = max(max_tau_rel, tau_rel); max_mae = max(max_mae, s_mae); max_rate_mae = max(max_rate_mae, r_mae); max_vp_err = max(max_vp_err, vp_err)
    rows.append({{"seed": sd, "tau_rel": round(tau_rel, 6), "strain_mae": round(s_mae, 6), "rate_mae": round(r_mae, 6), "vp_err": round(vp_err, 6)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau_rel <= 0.10 and max_mae <= 0.070 and max_rate_mae <= 0.055 and max_vp_err <= 0.16 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau_rel, 6), "max_strain_mae": round(max_mae, 6), "max_rate_mae": round(max_rate_mae, 6), "max_vp_err": round(max_vp_err, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    else:
        schema_keys = '(("estimated_taus", (2,)), ("predicted_strain", (n_te, k_t)), ("allocation_weights", (12,)))'
        val_extra = (
            '    if not errs:\n'
            '        ps_v = np.asarray(out["predicted_strain"], dtype=float)\n'
            '        aw_v = np.asarray(out["allocation_weights"], dtype=float)\n'
            '        Z_v = np.asarray(ds["Z_nuisance"], dtype=float)\n'
            '        Pz_v = np.linalg.solve(Z_v.T @ Z_v + float(ds["nuisance_ridge"]) * np.eye(Z_v.shape[1]), Z_v.T)\n'
            '        cg_v = 1.0 + 0.25 * np.tanh(ps_v[:, -1] - np.asarray(ds["test"]["strain_0"], dtype=float))\n'
            '        Sm_v = np.asarray(ds["S_alloc_modes"], dtype=float) * cg_v[:, None]\n'
            '        ya_v = np.asarray(ds["y_alloc_obs"], dtype=float).ravel()\n'
            '        chk_v = verify_box_simplex_kkt_certificate(Sm_v - Z_v @ (Pz_v @ Sm_v), ya_v - Z_v @ (Pz_v @ ya_v), float(ds["source_ridge"]), float(ds["u_box"]), float(ds["total_budget"]), out.get("kkt_certificate"), gap_tol=2e-4)\n'
            '        if not chk_v["valid"]: errs.append(f"kkt_invalid:{chk_v[\'reason\']}")\n'
            '        elif float(np.max(np.abs(aw_v - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("aw_cert_mismatch")\n'
        )
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau = 0.0; max_s = 0.0; max_wr = 0.0; max_gap = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "predicted_strain", "allocation_weights", "kkt_certificate"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    aw = np.asarray(out["allocation_weights"], dtype=float)
    if et.shape != (2,) or ps.shape != gt["te_clean"].shape or aw.shape != gt["w_alloc_opt"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(aw)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    Z_v = np.asarray(ds["Z_nuisance"], dtype=float)
    Pz_v = np.linalg.solve(Z_v.T @ Z_v + float(ds["nuisance_ridge"]) * np.eye(Z_v.shape[1]), Z_v.T)
    cg_v = 1.0 + 0.25 * np.tanh(ps[:, -1] - np.asarray(ds["test"]["strain_0"], dtype=float))
    Sm_v = np.asarray(ds["S_alloc_modes"], dtype=float) * cg_v[:, None]
    ya_v = np.asarray(ds["y_alloc_obs"], dtype=float).ravel()
    chk_k = verify_box_simplex_kkt_certificate(Sm_v - Z_v @ (Pz_v @ Sm_v), ya_v - Z_v @ (Pz_v @ ya_v), float(ds["source_ridge"]), float(ds["u_box"]), float(ds["total_budget"]), out["kkt_certificate"], gap_tol=2e-4)
    if not chk_k["valid"] or float(np.max(np.abs(aw - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False; rows.append({{"seed": sd, "error": f"kkt:{{chk_k.get('reason')}}"}}); continue
    tr = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    sm = float(np.mean(np.abs(ps - gt["te_clean"])))
    wr = float(np.linalg.norm(aw - gt["w_alloc_opt"]) / np.linalg.norm(gt["w_alloc_opt"]))
    max_tau = max(max_tau, tr); max_s = max(max_s, sm); max_wr = max(max_wr, wr); max_gap = max(max_gap, float(chk_k["gap"]))
    rows.append({{"seed": sd, "tau_rel": round(tr, 6), "strain_mae": round(sm, 6), "alloc_w_rel": round(wr, 6), "kkt_gap": round(float(chk_k["gap"]), 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau <= 0.10 and max_s <= 0.075 and max_wr <= 0.045 and max_gap <= 1e-5 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau, 6), "max_strain_mae": round(max_s, 6), "max_alloc_w_rel": round(max_wr, 6), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''

    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave=wave_label,
        scientific_domain="Multi-Wing Polymer Coupon Hereditary Creep & Continual Calibration Stream",
        wormhole_family="fam_hereditary_viscoelastic_memory",
        declared_artifacts=(f"/holdout_task/{artifact_name}",),
        declared_callable_entrypoints=(entry_name,),
        instruction_text=instruction_text,
        public_setup_py=(
            "import json, math, os, pickle, sys\nimport numpy as np\n"
            + _M96_ALPHA_GEN_PY
            + f"\npub_ds, pub_gt = {pub_call_expr}\n"
            f'with open("/holdout_task/{pkl_name}", "wb") as fh:\n'
            '    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)\n'
            'open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\\nimport numpy as np\\n" + '
            + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY)
            + f' + """\n_d = os.path.dirname(os.path.abspath(__file__))\n'
            f'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "{artifact_name}")\n'
            'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f'ds = pickle.load(open(os.path.join(_d, "{pkl_name}"), "rb"))[0]["dataset"]\n'
            f"out = mod.{entry_name}(ds)\n"
            'n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])\n'
            "errs = []\n"
            'if not isinstance(out, dict): errs.append("not_dict")\n'
            "else:\n"
            f"    for k, shp in {schema_keys}:\n"
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    if not errs and "estimated_taus" in out:\n'
            '        ts = np.asarray(out["estimated_taus"], dtype=float)\n'
            '        if not (0.05 < ts[0] < ts[1] < 25.0): errs.append("taus_not_positive_ordered")\n'
            + val_extra
            + 'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
            "sys.exit(0 if len(errs) == 0 else 1)\n"
            '""")\n'
        ),
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            f"import {mod_stem} as cand_mod\n"
            + _M96_ALPHA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def _build_m96_beta_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
    is_multibudget_T: bool = False,
) -> HoldoutOODTask:
    artifact_name = f"{short_name}_solver.py"
    entry_name = f"solve_{short_name}"
    pkl_name = f"public_{short_name}.pkl"
    mod_stem = Path(artifact_name).stem

    if not is_multibudget_T:
        val_check = (
            '    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):\n'
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"), gap_tol=2e-4)\n'
            '    if not chk["valid"]: errs.append(f"kkt_invalid:{chk[\'reason\']}")\n'
            '    elif "weights" in out and float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("weights_cert_mismatch")\n'
        )
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd, st, gt in cases:
    out = cand_mod.{entry_name}(st)
    if not isinstance(out, dict) or not {{"weights", "terminal_field", "kkt_certificate"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    w = np.asarray(out["weights"], dtype=float)
    tf = np.asarray(out["terminal_field"], dtype=float)
    if w.shape != gt["w_opt"].shape or tf.shape != gt["terminal_field_A"].shape or not (np.all(np.isfinite(w)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({{"seed": sd, "error": "shape_or_nan"}}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
    if not chk["valid"] or float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False; rows.append({{"seed": sd, "error": f"kkt:{{chk.get('reason')}}"}}); continue
    wr = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({{"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "kkt_gap": round(float(chk["gap"]), 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 2e-4 and max_f_rel <= 2e-4 and max_gap <= 1e-5 and el <= 18.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    else:
        val_check = (
            '    B = len(st["budget_grid"])\n'
            '    for k, shp in (("weights_by_budget", (B, M)), ("terminal_field_mid_budget", (nx, ny))):\n'
            '        if k not in out: errs.append(f"missing:{k}")\n'
            "        else:\n"
            "            a = np.asarray(out[k], dtype=float)\n"
            '            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            '    certs = out.get("kkt_certificates")\n'
            '    if not isinstance(certs, list) or len(certs) != B: errs.append("bad_certs_list")\n'
            "    elif not errs:\n"
            '        wb = np.asarray(out["weights_by_budget"], dtype=float)\n'
            '        for bi, bg in enumerate(st["budget_grid"]):\n'
            '            chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi], gap_tol=2e-4)\n'
            '            if not chk["valid"]: errs.append(f"kkt_b{bi}:{chk[\'reason\']}")\n'
            '            elif float(np.max(np.abs(wb[bi] - np.asarray(certs[bi]["weights"], dtype=float)))) > 1e-5: errs.append(f"wb_cert_mismatch_b{bi}")\n'
        )
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd, st, gt in cases:
    out = cand_mod.{entry_name}(st)
    if not isinstance(out, dict) or not {{"weights_by_budget", "terminal_field_mid_budget", "kkt_certificates"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    wb = np.asarray(out["weights_by_budget"], dtype=float)
    tf = np.asarray(out["terminal_field_mid_budget"], dtype=float)
    certs = out["kkt_certificates"]
    if wb.shape != gt["weights_by_budget"].shape or tf.shape != gt["terminal_field_mid_budget"].shape or not isinstance(certs, list) or len(certs) != 3:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(wb)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    ok_all = True
    for bi, bg in enumerate(st["budget_grid"]):
        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi], gap_tol=2e-4)
        if not chk["valid"] or float(np.max(np.abs(wb[bi] - np.asarray(certs[bi]["weights"], dtype=float)))) > 1e-5:
            ok_all = False; rows.append({{"seed": sd, "error": f"kkt_b{{bi}}"}}); break
        max_gap = max(max_gap, float(chk["gap"]))
    if not ok_all:
        valid = False; continue
    wr = float(np.linalg.norm(wb - gt["weights_by_budget"]) / np.linalg.norm(gt["weights_by_budget"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_mid_budget"]) / np.linalg.norm(gt["terminal_field_mid_budget"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr)
    rows.append({{"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "max_kkt_gap": round(max_gap, 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 2e-4 and max_f_rel <= 2e-4 and max_gap <= 1e-5 and el <= 18.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''

    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave=wave_label,
        scientific_domain="Multi-Wing Tracer Reaction-Transport & Continual Sensor Tomography Stream",
        wormhole_family="fam_kronecker_sylvester_and_kkt_certification",
        declared_artifacts=(f"/holdout_task/{artifact_name}",),
        declared_callable_entrypoints=(entry_name,),
        instruction_text=instruction_text,
        public_setup_py=(
            "import json, math, os, pickle, sys\nimport numpy as np\n"
            + _M96_BETA_GEN_PY
            + f"\npub_st, pub_gt = {pub_call_expr}\n"
            f'with open("/holdout_task/{pkl_name}", "wb") as fh:\n'
            '    pickle.dump([{"study": pub_st, "gt": pub_gt}], fh)\n'
            'open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\\nimport numpy as np\\n" + '
            + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY)
            + ' + """\n'
            "_d = os.path.dirname(os.path.abspath(__file__))\n"
            f'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "{artifact_name}")\n'
            'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f'item = pickle.load(open(os.path.join(_d, "{pkl_name}"), "rb"))[0]\n'
            'st, gt = item["study"], item["gt"]\n'
            f"out = mod.{entry_name}(st)\n"
            'M, nx, ny = st["source_modes"].shape\n'
            "errs = []\n"
            'if not isinstance(out, dict): errs.append("not_dict")\n'
            "else:\n"
            + val_check
            + 'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
            "sys.exit(0 if len(errs) == 0 else 1)\n"
            '""")\n'
        ),
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            f"import {mod_stem} as cand_mod\n"
            + _M96_BETA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def build_m96_world_tasks(spec: M96WorldSpec) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]]]:
    """Instantiate `(stage_S_task, 8_anonymous_stream_tasks, hidden_schedule_manifest)` for `M9.6` (`N, G1, T, C, R, U, G2, M`)."""
    rng = np.random.default_rng(spec.world_seed + 91)
    wid = spec.world_id
    v1_g, v1_b = (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)
    v2_g, v2_b = spec.v2_gains, spec.v2_offsets
    v3_g = (spec.v3_wing_a_gains[0], spec.v3_wing_a_gains[1], v2_g[2], v2_g[3])
    v3_b = (spec.v3_wing_a_offsets[0], spec.v3_wing_a_offsets[1], v2_b[2], v2_b[3])

    if spec.family == "alpha":
        cal_hdr = (
            "1. MULTI-WING LOAD-CELL & REFERENCE-COUPON CALIBRATION (`Wing A = labs 0,1`, `Wing B = labs 2,3`):\n"
            "   - `dataset['calibration_load_cells']` provides in-batch checks `{'lab_id', 'cmd_stress', 'true_stress'}` obeying\n"
            "     `true_stress = stress_gain[c] * cmd_stress + stress_bias[c]` (`c in 0..3`; reconcile sparse spot checks or wing EMI spikes\n"
            "     against `/tmp/hegel_continual_archive/archive_calibration_history.json` when needed!).\n"
            "   - Convert `past_stress_cmd` and `stress_hold_cmd` into calibrated physical `past_stress` and `stress_hold_0`.\n"
            "   - Standardize `coupon_features_raw` `(N, 4)` per lab via median column mean/std (`ddof=0`) of `ref_coupons` (`train + test`),\n"
            "     imputing `NaN`s with `(lab, stratum)` cell-balanced column means -> `X_aligned` `(N, 4)`.\n"
        )
        src_sn = f"{wid}_s00"
        t1_s = round(float(rng.uniform(0.50, 0.64)), 3)
        t2_s = round(float(rng.uniform(3.60, 4.25)), 3)
        th_s = [round(float(rng.uniform(0.65, 0.73)), 3), round(float(rng.uniform(0.52, 0.58)), 3), 0.09, 0.16, -0.13, 0.11, -0.08]
        src_task = _build_m96_alpha_task(
            task_id=f"{wid}_s00__coupon_creep_baseline",
            short_name=src_sn,
            wave_label=f"{wid}_source",
            instruction_text=(
                f"Implement `/holdout_task/{src_sn}_solver.py` exposing `def solve_{src_sn}(dataset: dict) -> dict:`\n\n"
                + cal_hdr
                + "2. SUB-YIELD 2-BRANCH HEREDITARY CREEP (`z_m(0) = 0`, `|past_stress|, |stress_hold_0| < dataset['yield_stress']`, ~8% Cauchy glitches on `strain_curves_obs`):\n"
                "   - Propagate `z_m <- exp(-past_dt[s]/tau_m)*z_m + (1 - exp(-past_dt[s]/tau_m))*past_stress[:, s]` (`0.20 <= tau_1 < tau_2 <= 8.0`).\n"
                "   - Sub-yield hold law: `eps(t_k) = strain_0 + theta_0*(stress_hold_0 - z_1)*(1 - exp(-t_k/tau_1)) + theta_1*(stress_hold_0 - z_2)*(1 - exp(-t_k/tau_2)) + (theta_2 + X_aligned @ theta_{3:7})*t_k`,\n"
                "     `instant_rate_obs` at `t=0+`, and sub-yield `viscoplastic_weights = np.zeros(2)`.\n"
                "3. RETURN unified dict `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'viscoplastic_weights': (2,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
                f"- Public dataset: `/holdout_task/public_{src_sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
            ),
            pub_call_expr=f"_gen_m96_alpha_cohort({spec.world_seed + 1}, ({t1_s}, {t2_s}), {th_s!r})",
            priv_cases_py=(
                f"    ({spec.world_seed + 501}, *_gen_m96_alpha_cohort({spec.world_seed + 501}, ({round(t1_s - 0.03, 3)}, {round(t2_s - 0.15, 3)}), [0.71, 0.55, 0.08, 0.17, -0.14, 0.12, -0.08])),\n"
                f"    ({spec.world_seed + 502}, *_gen_m96_alpha_cohort({spec.world_seed + 502}, ({round(t1_s + 0.05, 3)}, {round(t2_s + 0.25, 3)}), [0.64, 0.58, 0.10, -0.15, 0.13, -0.11, 0.09])),"
            ),
        )

        stream_tasks: list[HoldoutOODTask] = []
        manifest: list[dict[str, Any]] = []
        for idx, ev in enumerate(spec.schedule_events, start=1):
            sn = f"{wid}_batch_{idx:02d}"
            tid = f"{sn}__polymer_coupon_campaign_batch"
            sd_pub = spec.world_seed + 100 * idx + 11
            sd_p1 = spec.world_seed + 100 * idx + 61
            sd_p2 = spec.world_seed + 100 * idx + 62
            t1_e = round(float(rng.uniform(0.47, 0.65)), 3)
            t2_e = round(float(rng.uniform(3.50, 4.35)), 3)
            th_e = [round(float(rng.uniform(0.65, 0.73)), 3), round(float(rng.uniform(0.52, 0.58)), 3), 0.08, 0.15, -0.13, 0.11, -0.09]
            th_p1 = [0.71, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08]
            th_p2 = [0.64, 0.58, 0.08, -0.14, 0.13, -0.10, 0.09]

            if ev == "T":
                tk = _build_m96_alpha_task(
                    task_id=tid,
                    short_name=sn,
                    wave_label=f"{wid}_step_{idx:02d}",
                    is_shared_kkt_T=True,
                    instruction_text=(
                        f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(dataset: dict) -> dict:`\n\n"
                        + cal_hdr
                        + "2. PRE-LOADED HEREDITARY CREEP (`H = 22`, `K = 12`) + BOX-SIMPLEX KKT COMPLIANCE ALLOCATION (`M = 12`):\n"
                        "   - Initialize `z_m` from `initial_memory_states` `(N, 2)` before propagating across `past_dt`, fit `(tau_1, tau_2)` and `theta`,\n"
                        "     and compute `predicted_strain` `(N_test, K)`.\n"
                        "   - Form modulated mode matrix `S_mat = dataset['S_alloc_modes'] * (1.0 + 0.25 * tanh(predicted_strain[:, -1] - test['strain_0']))[:, None]`,\n"
                        "     profile out `Z = dataset['Z_nuisance']` with ridge `nuisance_ridge` (`Pz = solve(Z.T @ Z + nuisance_ridge * I, Z.T)`,\n"
                        "     `S_tilde = S_mat - Z @ (Pz @ S_mat)`, `r_y = y_alloc_obs - Z @ (Pz @ y_alloc_obs)`), and solve the constrained Box-Simplex QP:\n"
                        "     minimize `0.5*||S_tilde @ w - r_y||_2^2 + 0.5*source_ridge*||w||_2^2` subject to `0 <= w <= u_box` and `sum(w) == total_budget`!\n"
                        "   - Return `{'estimated_taus': (2,), 'predicted_strain': (N_test, K), 'allocation_weights': (12,), 'kkt_certificate': {'weights', 'nu_sum', 'lam_lower', 'lam_upper', 'primal_objective', 'dual_lower_bound'}}`.\n"
                        f"- Public dataset: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                    ),
                    pub_call_expr=f'_gen_m96_alpha_cohort({sd_pub}, ({t1_e}, {t2_e}), {th_e!r}, H=22, K=12, dt_mode="geom_var", preload_scale=2.10, shared_kkt_transfer_T=True)',
                    priv_cases_py=(
                        f'    ({sd_p1}, *_gen_m96_alpha_cohort({sd_p1}, (0.46, 3.40), {th_p1!r}, H=22, K=12, dt_mode="geom_var", preload_scale=2.10, shared_kkt_transfer_T=True)),\n'
                        f'    ({sd_p2}, *_gen_m96_alpha_cohort({sd_p2}, (0.71, 4.70), {th_p2!r}, H=22, K=12, dt_mode="geom_var", preload_scale=2.10, shared_kkt_transfer_T=True)),'
                    ),
                )
                cat = "cross_domain_kkt_transfer"
            else:
                u_flag = False
                if ev == "N":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v1_g, v1_b, "full_40", 2.95, (0.0, 0.0), 0.12, "immediate_cal_control"
                elif ev == "G1":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "full_40", 2.95, (0.0, 0.0), 0.08, "immediate_cal_control"
                elif ev == "C":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "full_40", 1.52, (0.25, 0.22), 0.08, "counterexample_discovery"
                elif ev == "R":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "sparse_spot_lab0", 2.95, (0.0, 0.0), 0.08, "autonomous_guard_and_multiparent"
                elif ev == "U":
                    # Non-Markovian Unloaded Residual Damage (`|stress_hold_0| <= 0.72 * ys`, early `|past_stress| > ys`!)
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "sparse_spot_lab0", 1.50, (0.25, 0.24), 0.08, "non_markovian_unloaded_history"
                    u_flag = True
                elif ev == "G2":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v3_g, v3_b, "scoped_wing_a_plus_emi_wing_b", 2.95, (0.0, 0.0), 0.08, "autonomous_guard_and_multiparent"
                elif ev == "M":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v3_g, v3_b, "sparse_spot_lab0", 1.50, (0.24, 0.21), 0.08, "autonomous_guard_and_multiparent"
                else:
                    raise ValueError(ev)

                tk = _build_m96_alpha_task(
                    task_id=tid,
                    short_name=sn,
                    wave_label=f"{wid}_step_{idx:02d}",
                    instruction_text=(
                        f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(dataset: dict) -> dict:`\n\n"
                        + cal_hdr
                        + "2. UNIFIED VISCOELASTIC / HISTORY-COUPLED VISCOPLASTIC CONSTITUTIVE CONTRACT (`sigma_y = dataset['yield_stress']`):\n"
                        "   - Calibrate physical `past_stress` `(N, H)` and `stress_hold_0` `(N,)` across Labs `0..3` (reconciling `calibration_load_cells`\n"
                        "     with `/tmp/hegel_continual_archive/archive_calibration_history.json` when in-batch sweeps are sparse or contain EMI spikes!).\n"
                        "   - Whenever BOTH the entire loading history `|past_stress|` AND current hold `|stress_hold_0|` remain `<= sigma_y`, the 2-branch\n"
                        "     linear Prony law holds (`viscoplastic_weights = [0.0, 0.0]`).\n"
                        "   - Whenever EITHER current hold `|stress_hold_0| > sigma_y` OR ANY prior step in `|past_stress| > sigma_y` (including unloaded\n"
                        "     specimens whose present `|stress_hold_0| < sigma_y` after prior overstress history!), active hold overstress\n"
                        "     `ov_hold = (maximum(0, |stress_hold_0| - sigma_y)^2)*sign(stress_hold_0)` adds `+ theta_7 * ov_hold * t_k`, AND accumulated\n"
                        "     plastic-work history `D_hist = sum((maximum(0, |past_stress| - sigma_y)^2)*past_dt, axis=1)` adds\n"
                        "     `+ theta_8 * (D_hist * stress_hold_0)*(1 - exp(-t_k/tau_1))` (with matching `t=0+` terms in `instant_rate_obs`).\n"
                        "3. RETURN unified dict `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'viscoplastic_weights': (2,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
                        f"- Public dataset: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                    ),
                    pub_call_expr=(
                        f"_gen_m96_alpha_cohort({sd_pub}, ({t1_e}, {t2_e}), {th_e!r}, stress_gains={g_use!r}, stress_biases={b_use!r}, "
                        f'cal_mode="{cm}", yield_stress={ys_e}, vp_theta={vp_e!r}, unloaded_residual_damage_U={u_flag!r}, glitch_frac={gl_f})'
                    ),
                    priv_cases_py=(
                        f'    ({sd_p1}, *_gen_m96_alpha_cohort({sd_p1}, (0.50, 3.60), {th_p1!r}, stress_gains={g_use!r}, stress_biases={b_use!r}, cal_mode="{cm}", yield_stress={ys_e}, vp_theta={vp_e!r}, unloaded_residual_damage_U={u_flag!r}, glitch_frac={gl_f})),\n'
                        f'    ({sd_p2}, *_gen_m96_alpha_cohort({sd_p2}, (0.68, 4.45), {th_p2!r}, stress_gains={g_use!r}, stress_biases={b_use!r}, cal_mode="{cm}", yield_stress={ys_e}, vp_theta={vp_e!r}, unloaded_residual_damage_U={u_flag!r}, glitch_frac={gl_f})),'
                    ),
                )
            stream_tasks.append(tk)
            manifest.append({"step_index": idx, "task_id": tid, "short_name": sn, "causal_event": ev, "event_category": cat})
        return src_task, tuple(stream_tasks), manifest

    # Beta family
    cal_hdr_b = (
        "1. MULTI-WING SENSOR-BANK CALIBRATION (`Wing A = banks 0,1`, `Wing B = banks 2,3`) & 1D FVM STENCIL:\n"
        "   - `study['sensor_calibration_sweeps']` gives in-batch sweeps `{'bank_id', 'ref_signal', 'raw_readout'}` obeying\n"
        "     `raw_readout = bank_gain[b] * ref_signal + bank_offset[b]` (`b in 0..3`; reconcile sparse checks or wing EMI spikes\n"
        "     with `/tmp/hegel_continual_archive/archive_calibration_history.json` when needed!).\n"
        "   - Invert via `row_bank_id`: `y_obs = (y_obs_raw - bank_offset[row_bank_id]) / bank_gain[row_bank_id]`.\n"
        "   - 1D FVM operator `_build_1d(L, n, kappa, vel, robin, diag_add)`: harmonic interior diffusion `(2*k_i*k_{i+1}/(k_i+k_{i+1}))/h^2`,\n"
        "     Robin boundary `(rg*k_b)/(h*(rg*0.5*h+k_b))` at `0, n-1`, and first-order upwind advection.\n"
    )
    src_sn = f"{wid}_s00"
    src_task = _build_m96_beta_task(
        task_id=f"{wid}_s00__tracer_tomography_baseline",
        short_name=src_sn,
        wave_label=f"{wid}_source",
        instruction_text=(
            f"Implement `/holdout_task/{src_sn}_solver.py` exposing `def solve_{src_sn}(study: dict) -> dict:`\n\n"
            + cal_hdr_b
            + "2. LAMINAR SEPARABLE MULTI-SHIFT REACTION-TRANSPORT & BOX-SIMPLEX KKT (`40 x 36`, `M = 12`, `K_mu = 8`):\n"
            "   - In quiescent laminar transport (`V_reaction_2d(x,y) = v_x(x) + v_y(y)` separable AND `max(past_vortex_circulation_trace) <= critical_vortex_circulation`),\n"
            "     solve `(A_x(1.0*kappa_x) + diag(v_x)) U_{m,k} + U_{m,k} (B_y(1.0*kappa_y) + diag(v_y)).T + mu_shifts[k]*U_{m,k} = source_modes[m]`,\n"
            "     profile `Z_nuisance`, and minimize `0.5*||S_tilde @ w - r_y||_2^2 + 0.5*source_ridge*||w||_2^2` on `{0 <= w <= u_box, sum(w) == total_budget}`.\n"
            "   - Return unified dict `{'weights': (M,), 'terminal_field': sum_m w_m * U_{m, -1} (nx, ny), 'kkt_certificate': {'weights', 'nu_sum', 'lam_lower', 'lam_upper', 'primal_objective', 'dual_lower_bound'}}`.\n"
            f"- Public study: `/holdout_task/public_{src_sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m96_beta_study({spec.world_seed + 1})",
        priv_cases_py=(
            f"    ({spec.world_seed + 501}, *_gen_m96_beta_study({spec.world_seed + 501})),\n"
            f"    ({spec.world_seed + 502}, *_gen_m96_beta_study({spec.world_seed + 502})),"
        ),
    )

    stream_tasks = []
    manifest = []
    for idx, ev in enumerate(spec.schedule_events, start=1):
        sn = f"{wid}_batch_{idx:02d}"
        tid = f"{sn}__tracer_survey_campaign_batch"
        sd_pub = spec.world_seed + 100 * idx + 11
        sd_p1 = spec.world_seed + 100 * idx + 61
        sd_p2 = spec.world_seed + 100 * idx + 62

        if ev == "T":
            tk = _build_m96_beta_task(
                task_id=tid,
                short_name=sn,
                wave_label=f"{wid}_step_{idx:02d}",
                is_multibudget_T=True,
                instruction_text=(
                    f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(study: dict) -> dict:`\n\n"
                    + cal_hdr_b
                    + "2. ANISOTROPIC LAMINAR GRID (`44 x 36`, `M = 14`, `K_mu = 10`) & 3-BUDGET KKT PARETO (`study['budget_grid']`):\n"
                    "   - Solve laminar separable reaction-transport (`1.0 * kappa`), profile `Z_nuisance`,\n"
                    "     and return `{'weights_by_budget': (3, M), 'terminal_field_mid_budget': (nx, ny) for b=1, 'kkt_certificates': [cert_0, cert_1, cert_2]}`.\n"
                    f"- Public study: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                ),
                pub_call_expr=f"_gen_m96_beta_study({sd_pub}, nx=44, ny=36, K_mu=10, M=14, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)",
                priv_cases_py=(
                    f"    ({sd_p1}, *_gen_m96_beta_study({sd_p1}, nx=44, ny=36, K_mu=10, M=14, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),\n"
                    f"    ({sd_p2}, *_gen_m96_beta_study({sd_p2}, nx=44, ny=36, K_mu=10, M=14, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),"
                ),
            )
            cat = "cross_domain_kkt_transfer"
        else:
            u_b = False
            if ev == "N":
                g_use, b_use, cm, vx_f, ns_sc, cat = v1_g, v1_b, "full_48", False, 0.022, "immediate_cal_control"
            elif ev == "G1":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "full_48", False, 0.011, "immediate_cal_control"
            elif ev == "C":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "full_48", True, 0.011, "counterexample_discovery"
            elif ev == "R":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "sparse_spot_bank0", False, 0.011, "autonomous_guard_and_multiparent"
            elif ev == "U":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "sparse_spot_bank0", False, 0.011, "non_markovian_unloaded_history"
                u_b = True
            elif ev == "G2":
                g_use, b_use, cm, vx_f, ns_sc, cat = v3_g, v3_b, "scoped_wing_a_plus_emi_wing_b", False, 0.011, "autonomous_guard_and_multiparent"
            elif ev == "M":
                g_use, b_use, cm, vx_f, ns_sc, cat = v3_g, v3_b, "sparse_spot_bank0", True, 0.011, "autonomous_guard_and_multiparent"
            else:
                raise ValueError(ev)

            # UNIFIED Neutral Constitutive Specification Across ALL Non-`T` Beta Steps (`0` per-step hint leak!)
            unified_beta_spec = (
                "2. UNIFIED LAMINAR / EDDY-MEMORY REACTION-TRANSPORT & BOX-SIMPLEX KKT (`40 x 36`, `M = 12`, `K_mu = 8`):\n"
                "   - Inspect BOTH `V_reaction_2d(x, y)` AND the temporal circulation history `past_vortex_circulation_trace` (`critical_vortex_circulation = study['critical_vortex_circulation']`):\n"
                "     * When `V_reaction_2d` is additively separable (`||V - v_x - v_y||_F / ||V||_F <= 1e-6`) AND the ENTIRE history `max(past_vortex_circulation_trace) <= critical_vortex_circulation`,\n"
                "       the flow is quiescent laminar (`1.00 * kappa_x`, `1.00 * kappa_y`, separable Sylvester).\n"
                "     * Whenever EITHER `V_reaction_2d` has non-separable 2D cross-coupling (`> 1e-4`) OR ANY prior step in `past_vortex_circulation_trace > critical_vortex_circulation`\n"
                "       (including post-vortex relaxation batches where `V_reaction_2d` is separable and terminal circulation is low, but earlier vortex history sustains turbulent eddy diffusivity!),\n"
                "       diffusivity is `1.45 * kappa_x`, `1.45 * kappa_y` governed by `(kron(A_x(1.45*kappa_x, reaction_x), I_y) + kron(I_x, B_y(1.45*kappa_y, reaction_y)) + diag(V_reaction_2d.ravel() + mu_shifts[k])) u_{m,k} = source_modes[m].ravel()` (C-order)!\n"
                "   - Profile `Z_nuisance` and return unified dict `{'weights': (M,), 'terminal_field': (nx, ny), 'kkt_certificate': dict}`.\n"
            )
            tk = _build_m96_beta_task(
                task_id=tid,
                short_name=sn,
                wave_label=f"{wid}_step_{idx:02d}",
                instruction_text=(
                    f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(study: dict) -> dict:`\n\n"
                    + cal_hdr_b
                    + unified_beta_spec
                    + f"- Public study: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                ),
                pub_call_expr=(
                    f'_gen_m96_beta_study({sd_pub}, bank_gains={g_use!r}, bank_offsets={b_use!r}, cal_mode="{cm}", '
                    f"vortex_regime={vx_f!r}, unloaded_post_vortex_memory_U={u_b!r}, obs_noise_scale={ns_sc})"
                ),
                priv_cases_py=(
                    f'    ({sd_p1}, *_gen_m96_beta_study({sd_p1}, bank_gains={g_use!r}, bank_offsets={b_use!r}, cal_mode="{cm}", vortex_regime={vx_f!r}, unloaded_post_vortex_memory_U={u_b!r}, obs_noise_scale={ns_sc})),\n'
                    f'    ({sd_p2}, *_gen_m96_beta_study({sd_p2}, bank_gains={g_use!r}, bank_offsets={b_use!r}, cal_mode="{cm}", vortex_regime={vx_f!r}, unloaded_post_vortex_memory_U={u_b!r}, obs_noise_scale={ns_sc})),'
                ),
            )
        stream_tasks.append(tk)
        manifest.append({"step_index": idx, "task_id": tid, "short_name": sn, "causal_event": ev, "event_category": cat})
    return src_task, tuple(stream_tasks), manifest


def expected_m96_public_output_keys(task: HoldoutOODTask) -> list[str]:
    val_txt = task.public_setup_py.split("validate_public_schema.py", 1)[-1]
    known = [
        "estimated_taus", "hereditary_weights", "viscoplastic_weights", "predicted_strain", "initial_rate_pred",
        "allocation_weights", "kkt_certificate",
        "weights", "terminal_field",
        "weights_by_budget", "terminal_field_mid_budget", "kkt_certificates",
    ]
    return sorted([k for k in known if f'"{k}"' in val_txt])


def m96_task_to_frontier_task(
    task: HoldoutOODTask,
    *,
    continual_context_block: str = "",
    allowed_readonly_dependencies: tuple[str, ...] = (),
) -> m6f.FrontierBenchmarkTask:
    full_inst = f"{continual_context_block.strip()}\n\n{task.instruction_text}" if continual_context_block.strip() else task.instruction_text
    pkl_asset = f"/holdout_task/public_{task.short_name}.pkl"
    req_assets = tuple(dict.fromkeys([pkl_asset, "/holdout_task/validate_public_schema.py", *allowed_readonly_dependencies]))
    return m6f.FrontierBenchmarkTask(
        task_id=task.task_id,
        short_name=task.short_name,
        benchmark="Hegel-M96-Autonomous-Structure-Continual-World",
        wave=task.wave,
        scientific_domain=task.scientific_domain,
        wormhole_family=task.wormhole_family,
        instruction_text=full_inst,
        declared_artifacts=task.declared_artifacts,
        declared_callable_entrypoints=task.declared_callable_entrypoints,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        agent_cpus=2,
        agent_memory_mb=3072,
        verifier_cpus=2,
        verifier_memory_mb=2048,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
        holdout_public_setup_py=task.public_setup_py,
        required_public_assets=req_assets,
        allowed_readonly_dependencies=tuple(allowed_readonly_dependencies),
    )


# ============================================================================
# 3. Domain-Agnostic 12-Feature Structural Bank & Autonomous Guard Induction
# ============================================================================

M96_CANDIDATE_GUARD_FEATURES: tuple[str, ...] = (
    # Snapshot vs Full-History Signal Features (includes myopic snapshot trap!)
    "max_hold_signal_over_declared_threshold_excess",
    "max_history_and_hold_plus_residual_state_excess",
    "temporal_history_integral_over_threshold",
    # Spatial / Operator / Temporal Trace Features (includes Frobenius norm & terminal-only traps!)
    "matrix_2d_frobenius_norm",
    "matrix_2d_nonseparable_additive_residual_ratio",
    "terminal_circulation_over_critical_excess",
    "max_history_circulation_over_critical_excess",
    "combined_nonseparable_or_vortex_history_excess",
    # Nuisance / Geometry Distractors
    "boundary_robin_asymmetry_ratio",
    "advection_peclet_proxy",
    "nuisance_design_condition_number",
    "input_panel_missingness_fraction",
)


def compute_generic_public_structural_features(
    public_item: dict[str, Any],
    param_payload: dict[str, Any],
) -> dict[str, float | None]:
    """Compute the 12 candidate structural features purely from `public_item` + `param_payload` (`0` private peek, `0` hardcoded rule!).
    Missing/inapplicable features for a given data dictionary cleanly return `None` (which evaluates to `'UNKNOWN'` if referenced by a guard!).
    """
    feats: dict[str, float | None] = {k: None for k in M96_CANDIDATE_GUARD_FEATURES}

    if "dataset" in public_item and isinstance(public_item["dataset"], dict):
        ds = public_item["dataset"]
        gains = np.asarray(param_payload.get("stress_gains", [1.0, 1.0, 1.0, 1.0]), dtype=float)
        biases = np.asarray(param_payload.get("stress_biases", [0.0, 0.0, 0.0, 0.0]), dtype=float)
        ys = ds.get("yield_stress")
        p_dt = np.asarray(ds.get("past_dt", []), dtype=float)
        tr_d, te_d = ds.get("train") or {}, ds.get("test") or {}
        if ys is not None and math.isfinite(float(ys)) and float(ys) > 0.0 and "stress_hold_cmd" in te_d and "stress_hold_cmd" in tr_d:
            ys_f = float(ys)
            tr_l = np.asarray(tr_d["lab_id"], dtype=int)
            te_l = np.asarray(te_d["lab_id"], dtype=int)
            h_tr = np.asarray(tr_d["stress_hold_cmd"], dtype=float) * gains[tr_l] + biases[tr_l]
            h_te = np.asarray(te_d["stress_hold_cmd"], dtype=float) * gains[te_l] + biases[te_l]
            if np.all(np.isfinite(h_tr)) and np.all(np.isfinite(h_te)):
                # Myopic trap feature: inspects ONLY present `stress_hold_0` on target `test` split (or max across hold), missing unloaded `past_stress`!
                feats["max_hold_signal_over_declared_threshold_excess"] = round(
                    float(max(0.0, float(np.max(np.abs(h_te))) - ys_f)), 8
                )
            if "past_stress_cmd" in tr_d and "past_stress_cmd" in te_d:
                p_tr = np.asarray(tr_d["past_stress_cmd"], dtype=float) * gains[tr_l, None] + biases[tr_l, None]
                p_te = np.asarray(te_d["past_stress_cmd"], dtype=float) * gains[te_l, None] + biases[te_l, None]
                res_pl = np.asarray(te_d.get("residual_plastic_state", np.zeros(len(te_l))), dtype=float)
                if np.all(np.isfinite(p_tr)) and np.all(np.isfinite(p_te)) and np.all(np.isfinite(res_pl)):
                    pk_all = max(
                        float(np.max(np.abs(h_tr))),
                        float(np.max(np.abs(h_te))),
                        float(np.max(np.abs(p_tr))),
                        float(np.max(np.abs(p_te))),
                    )
                    feats["max_history_and_hold_plus_residual_state_excess"] = round(
                        float(max(0.0, pk_all - ys_f) + float(np.max(np.abs(res_pl)))), 8
                    )
                    if len(p_dt) == p_te.shape[1]:
                        d_int_te = float(np.max(np.sum((np.maximum(0.0, np.abs(p_te) - ys_f) ** 2) * p_dt[None, :], axis=1)))
                        d_int_tr = float(np.max(np.sum((np.maximum(0.0, np.abs(p_tr) - ys_f) ** 2) * p_dt[None, :], axis=1)))
                        h_ex = max(0.0, max(float(np.max(np.abs(h_te))), float(np.max(np.abs(h_tr)))) - ys_f)
                        feats["temporal_history_integral_over_threshold"] = round(
                            max(d_int_te, d_int_tr) + h_ex + float(np.max(np.abs(res_pl))), 8
                        )
        if "coupon_features_raw" in te_d:
            xr = np.asarray(te_d["coupon_features_raw"], dtype=float)
            feats["input_panel_missingness_fraction"] = round(float(np.mean(~np.isfinite(xr))), 6)
        if "Z_nuisance" in ds:
            zn = np.asarray(ds["Z_nuisance"], dtype=float)
            feats["nuisance_design_condition_number"] = round(float(np.linalg.cond(zn)), 6)

    if "study" in public_item and isinstance(public_item["study"], dict):
        st = public_item["study"]
        if "V_reaction_2d" in st and st["V_reaction_2d"] is not None:
            v2 = np.asarray(st["V_reaction_2d"], dtype=float)
            if v2.ndim == 2 and np.all(np.isfinite(v2)):
                vn = float(np.linalg.norm(v2, ord="fro"))
                v_add = np.mean(v2, axis=1, keepdims=True) + np.mean(v2, axis=0, keepdims=True) - float(np.mean(v2))
                nonsep = float(np.linalg.norm(v2 - v_add, ord="fro") / max(vn, 1e-12))
                feats["matrix_2d_frobenius_norm"] = round(vn, 8)
                feats["matrix_2d_nonseparable_additive_residual_ratio"] = round(nonsep, 8)
            else:
                nonsep = None
        else:
            nonsep = None
        if "past_vortex_circulation_trace" in st and "critical_vortex_circulation" in st:
            vt = np.asarray(st["past_vortex_circulation_trace"], dtype=float).ravel()
            crit_v = float(st["critical_vortex_circulation"])
            if vt.size > 0 and np.all(np.isfinite(vt)) and math.isfinite(crit_v):
                # Myopic terminal-only trap vs full-history circulation excess!
                term_ex = max(0.0, float(vt[-1]) - crit_v)
                hist_ex = max(0.0, float(np.max(vt)) - crit_v)
                feats["terminal_circulation_over_critical_excess"] = round(term_ex, 8)
                feats["max_history_circulation_over_critical_excess"] = round(hist_ex, 8)
                if nonsep is not None:
                    feats["combined_nonseparable_or_vortex_history_excess"] = round(max(float(nonsep), hist_ex), 8)
        if "robin_x" in st and "robin_y" in st:
            rx, ry = float(st["robin_x"]), float(st["robin_y"])
            feats["boundary_robin_asymmetry_ratio"] = round(abs(rx - ry) / max(abs(rx) + abs(ry), 1e-12), 6)
        if "vel_x" in st and "kappa_x" in st:
            vx = np.asarray(st["vel_x"], dtype=float)
            kx = np.asarray(st["kappa_x"], dtype=float)
            feats["advection_peclet_proxy"] = round(float(np.mean(np.abs(vx) / np.maximum(kx, 1e-6))), 6)
        if "Z_nuisance" in st:
            zn = np.asarray(st["Z_nuisance"], dtype=float)
            feats["nuisance_design_condition_number"] = round(float(np.linalg.cond(zn)), 6)
    return feats


def evaluate_m96_autonomous_guard(
    guard_spec: dict[str, Any] | None,
    feature_map: dict[str, float | None],
    *,
    allow_open_feature_keys: bool = False,
) -> dict[str, Any]:
    """Strict 3-valued (`TRUE / FALSE / UNKNOWN`) Kleene fail-closed evaluator for model-proposed guards (`M9.6` + `M9.6.1` `all_of` conjunctions).
    - When `guard_spec` is `None` or empty (`unconstrained_initial_source` before any counterexample has falsified `C0`),
      returns `state = 'TRUE'` with `guard_mode = 'unconstrained_prior_to_counterexample'`.
    - Supports `{"all_of": [clause1, clause2, ...]}` under exact Kleene 3-valued logic (`FALSE` if any clause is `FALSE`,
      else `UNKNOWN` if any clause is `UNKNOWN`, else `TRUE`) so multi-conflict branches (`C0 -> C1 -> C2`) refine intervals cleanly!
    """
    if not guard_spec:
        return {"state": "TRUE", "guard_mode": "unconstrained_prior_to_counterexample", "measured_value": None}
    if isinstance(guard_spec.get("all_of"), list) and guard_spec["all_of"]:
        sub_evals = [
            evaluate_m96_autonomous_guard(cl, feature_map, allow_open_feature_keys=allow_open_feature_keys)
            for cl in guard_spec["all_of"]
        ]
        states = [s["state"] for s in sub_evals]
        if "FALSE" in states:
            return {"state": "FALSE", "clauses": sub_evals}
        if "UNKNOWN" in states:
            return {"state": "UNKNOWN", "clauses": sub_evals}
        return {"state": "TRUE", "clauses": sub_evals}
    fname = str(guard_spec.get("feature_name") or "")
    op = str(guard_spec.get("op") or "")
    thr_raw = guard_spec.get("threshold")
    valid_feat = (fname in feature_map) if allow_open_feature_keys else (fname in M96_CANDIDATE_GUARD_FEATURES)
    if (not valid_feat) or op not in {"<=", "<", ">=", ">"} or thr_raw is None:
        return {"state": "UNKNOWN", "reason": f"invalid_guard_spec:{fname}:{op}", "measured_value": None}
    val = feature_map.get(fname)
    if val is None or not math.isfinite(float(val)) or not math.isfinite(float(thr_raw)):
        return {"state": "UNKNOWN", "reason": f"missing_or_nonfinite_feature:{fname}", "measured_value": None}
    v_f = float(val)
    t_f = float(thr_raw)
    if op == "<=":
        ok = v_f <= t_f
    elif op == "<":
        ok = v_f < t_f
    elif op == ">=":
        ok = v_f >= t_f
    else:
        ok = v_f > t_f
    return {
        "state": "TRUE" if ok else "FALSE",
        "feature_name": fname,
        "op": op,
        "threshold": t_f,
        "measured_value": v_f,
    }


M96_AUTONOMOUS_SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "mechanistic_diagnosis",
        "violated_component_id",
        "preserved_component_ids",
        "dependent_upstream_mechanism_ids",
        "c0_guard",
        "c1_guard",
    ],
    "properties": {
        "mechanistic_diagnosis": {"type": "string"},
        "violated_component_id": {"type": "string"},
        "preserved_component_ids": {"type": "array", "items": {"type": "string"}},
        "dependent_upstream_mechanism_ids": {"type": "array", "items": {"type": "string"}},
        "c0_guard": {
            "type": "object",
            "additionalProperties": False,
            "required": ["feature_name", "op", "threshold"],
            "properties": {
                "feature_name": {"type": "string", "enum": list(M96_CANDIDATE_GUARD_FEATURES)},
                "op": {"type": "string", "enum": ["<=", "<", ">=", ">"]},
                "threshold": {"type": "number"},
            },
        },
        "c1_guard": {
            "type": "object",
            "additionalProperties": False,
            "required": ["feature_name", "op", "threshold"],
            "properties": {
                "feature_name": {"type": "string", "enum": list(M96_CANDIDATE_GUARD_FEATURES)},
                "op": {"type": "string", "enum": ["<=", "<", ">=", ">"]},
                "threshold": {"type": "number"},
            },
        },
    },
}


def verify_contrastive_public_witness_guards(
    *,
    c0_guard: dict[str, Any],
    c1_guard: dict[str, Any],
    positive_c0_witness_features: list[dict[str, float | None]],
    negative_c0_witness_features: list[dict[str, float | None]],
) -> dict[str, Any]:
    """Host Contrastive Public Witness Verifier (`0` private peek, `0` hardcoded feature name!).
    Accepts `(c0_guard, c1_guard)` iff:
      1. On EVERY public witness `w_pos` where `C0` passed the public scientific gate:
         `c0_guard(w_pos) == 'TRUE'` AND `c1_guard(w_pos) == 'FALSE'`.
      2. On EVERY public witness `w_neg` where `C0` failed the public scientific gate and `C1` passed:
         `c0_guard(w_neg) == 'FALSE'` AND `c1_guard(w_neg) == 'TRUE'`.
    """
    pos_evals = []
    neg_evals = []
    ok = bool(positive_c0_witness_features and negative_c0_witness_features)
    reasons: list[str] = []
    for idx_p, fp in enumerate(positive_c0_witness_features):
        r0 = evaluate_m96_autonomous_guard(c0_guard, fp)
        r1 = evaluate_m96_autonomous_guard(c1_guard, fp)
        pos_evals.append({"witness_idx": idx_p, "c0": r0, "c1": r1})
        if r0["state"] != "TRUE" or r1["state"] != "FALSE":
            ok = False
            reasons.append(f"pos_witness_{idx_p}_failed:c0={r0['state']},c1={r1['state']}")
    for idx_n, fn in enumerate(negative_c0_witness_features):
        r0 = evaluate_m96_autonomous_guard(c0_guard, fn)
        r1 = evaluate_m96_autonomous_guard(c1_guard, fn)
        neg_evals.append({"witness_idx": idx_n, "c0": r0, "c1": r1})
        if r0["state"] != "FALSE" or r1["state"] != "TRUE":
            ok = False
            reasons.append(f"neg_witness_{idx_n}_failed:c0={r0['state']},c1={r1['state']}")
    return {
        "contrastively_valid": ok,
        "reasons": reasons,
        "positive_witness_evaluations": pos_evals,
        "negative_witness_evaluations": neg_evals,
    }


def verify_upstream_parent_perturbation_sensitivity(
    *,
    family: str,
    candidate_upstream_ids: list[str],
    proposed_upstream_ids: list[str],
    c1_canonical_code: str,
    c1_entrypoint: str,
    public_item: dict[str, Any],
    base_param_payload: dict[str, Any],
    intervention_fn_by_upstream_id: dict[str, Any] | None = None,
    run_in_container: bool = False,
) -> dict[str, Any]:
    """Mechanically test which upstream calibration mechanisms in `candidate_upstream_ids`
    actually affect `C1`'s public predictions when perturbed (`0` private peek!).

    `M9.6.1` `N04/W03` Hardening:
      - Supports explicit `intervention_fn_by_upstream_id` metadata (so arbitrary anonymized parent IDs and genuinely
        wired nuisance parents can be intervened without relying on `'wing_a' in uid` string naming!).
      - Supports `run_in_container=True` (executing untrusted candidate strictly inside `CLEAN_SCIPY_SANDBOX_IMAGE`
        `--network none` with pure-JSON typed-array output instead of host `python -I`!).
      - Compares ALL primary numerical output arrays across full test slices (not only the first 64 elements!).
    """
    import pickle as _pkl
    import shlex as _shlex
    import sys as _sys
    from . import m6_frontier_real_benchmarks as _m6f
    from .m7_autogenous_forest_pretraining import CLEAN_SCIPY_SANDBOX_IMAGE as _SANDBOX_IMG
    from .m9_4_1_shared_verification_kernel import decode_pure_json_numerical_text

    w_py = (
        "import importlib.util, json, math, pickle, sys, numpy as np\n"
        "s = importlib.util.spec_from_file_location('_m', sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "r = m._probe_sens_fn(pickle.loads(open(sys.argv[2], 'rb').read()))\n"
        "vecs = []\n"
        "for k in ('predicted_strain', 'initial_rate_pred', 'terminal_field', 'weights', 'weights_by_budget', 'allocation_weights'):\n"
        "    if isinstance(r, dict) and k in r and r[k] is not None:\n"
        "        a = np.asarray(r[k], dtype=float).ravel()\n"
        "        vecs.extend(a.tolist())\n"
        "arr = np.asarray(vecs, dtype=float)\n"
        "open(sys.argv[3], 'w').write(json.dumps({'__nd__': True, 'dtype': 'float64', 'shape': [int(arr.size)], 'data': arr.tolist()}, allow_nan=False))\n"
    )

    sandbox_cid: str | None = None
    if run_in_container:
        sandbox_cid = _m6f.start_detached_container(_SANDBOX_IMG, memory_limit="1024m", cpus="1.5", network_mode="none")

    try:
        def _run_c1_with_payload(pay: dict[str, Any], inp_override: dict[str, Any] | None = None) -> dict[str, Any]:
            adapted = render_parameterized_input_adapter(
                family=family,
                source_code=c1_canonical_code,
                source_entrypoint=c1_entrypoint,
                target_entrypoint="_probe_sens_fn",
                param_payload=pay,
            )
            inp = deepcopy(inp_override if inp_override is not None else (public_item["dataset"] if family == "alpha" else public_item["study"]))
            with tempfile.TemporaryDirectory(prefix="m96_sens_") as td:
                td_p = Path(td)
                (td_p / "c.py").write_text(adapted, encoding="utf-8")
                (td_p / "in.pkl").write_bytes(_pkl.dumps(inp))
                (td_p / "w.py").write_text(w_py, encoding="utf-8")
                if sandbox_cid is not None:
                    _m6f.exec_container_bash(sandbox_cid, "rm -rf /tmp/_sens && mkdir -p /tmp/_sens", timeout_sec=10)
                    for fn_s in ("c.py", "in.pkl", "w.py"):
                        subprocess.run(["docker", "cp", str(td_p / fn_s), f"{sandbox_cid}:/tmp/_sens/{fn_s}"], check=True, capture_output=True, timeout=15)
                    ex = _m6f.exec_container_bash(
                        sandbox_cid,
                        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /tmp/_sens/w.py /tmp/_sens/c.py /tmp/_sens/in.pkl /tmp/_sens/out.json && cat /tmp/_sens/out.json",
                        timeout_sec=45,
                    )
                    if ex["exit_code"] != 0:
                        raise RuntimeError(f"Container sensitivity worker failed: {ex['stderr'][-240:]}")
                    raw_txt = ex["stdout"].strip().splitlines()[-1]
                else:
                    subprocess.run([_sys.executable, "-I", str(td_p / "w.py"), str(td_p / "c.py"), str(td_p / "in.pkl"), str(td_p / "out.json")], check=True, timeout=45)
                    raw_txt = (td_p / "out.json").read_text(encoding="utf-8")
                return {"vec": np.asarray(decode_pure_json_numerical_text(raw_txt), dtype=float)}

        base_inp = deepcopy(public_item["dataset"] if family == "alpha" else public_item["study"])
        base_vec = _run_c1_with_payload(base_param_payload, base_inp)["vec"]
        sens_by_id: dict[str, float] = {}
        empirically_active: list[str] = []
        for uid in candidate_upstream_ids:
            pert_pay = deepcopy(base_param_payload)
            pert_inp = deepcopy(base_inp)
            if isinstance(intervention_fn_by_upstream_id, dict) and uid in intervention_fn_by_upstream_id:
                pert_pay, pert_inp = intervention_fn_by_upstream_id[uid](pert_pay, pert_inp)
            elif "wing_a" in uid:
                kg = "stress_gains" if family == "alpha" else "bank_gains"
                pert_pay[kg][0] = float(pert_pay[kg][0]) * 1.18
                pert_pay[kg][1] = float(pert_pay[kg][1]) * 0.84
            elif "wing_b" in uid:
                kg = "stress_gains" if family == "alpha" else "bank_gains"
                pert_pay[kg][2] = float(pert_pay[kg][2]) * 1.18
                pert_pay[kg][3] = float(pert_pay[kg][3]) * 0.84
            p_vec = _run_c1_with_payload(pert_pay, pert_inp)["vec"]
            diff = float(np.max(np.abs(p_vec - base_vec)))
            sens_by_id[uid] = round(diff, 8)
            if diff > 1e-5:
                empirically_active.append(uid)
    finally:
        if sandbox_cid is not None:
            _m6f.stop_container_safely(sandbox_cid)

    prop_clean = [u for u in proposed_upstream_ids if u in candidate_upstream_ids]
    exact_match = set(prop_clean) == set(empirically_active)
    return {
        "proposed_upstream_ids": prop_clean,
        "empirically_sensitive_upstream_ids": empirically_active,
        "sensitivity_max_diff_by_upstream_id": sens_by_id,
        "proposal_matches_empirical_sensitivity": exact_match,
        "executed_in_isolated_container": bool(run_in_container),
    }


def propose_and_contrastively_validate_autonomous_split(
    *,
    client: Any,
    stage_subdir: str,
    family: str,
    task_instruction_text: str,
    c0_code: str,
    c1_code: str,
    c1_entrypoint: str,
    positive_c0_witness_features: list[dict[str, float | None]],
    negative_c0_witness_features: list[dict[str, float | None]],
    current_public_item: dict[str, Any],
    current_param_payload: dict[str, Any],
    candidate_upstream_ids: list[str],
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Invoke `gpt-6.2-sol` (`M9.6` `§4.1 + §4.2 + §4.5`) to autonomously propose:
      1. `c0_guard` & `c1_guard` over the 12-feature bank (`0` host `if family == ...` guard hardcoding!),
      2. `violated_component_id` & `preserved_component_ids`,
      3. `dependent_upstream_mechanism_ids` from `candidate_upstream_ids` (including `sentinel_shared_acoustic_root` distractor!).
    Validates strictly via `verify_contrastive_public_witness_guards` + `verify_upstream_parent_perturbation_sensitivity` (`0` private peek!).
    """
    sys_prompt = (
        "You are the Hegel Autonomous Mechanistic Structure & Regime Guard Induction Compiler.\n"
        "A prior base solver `C0` passed the public scientific gate on earlier batches (`positive_c0_witnesses`), "
        "but was empirically FALSIFIED on new batch(es) (`negative_c0_witnesses`), where extended solver `C1` succeeded.\n"
        "Carefully read the physical task specification and compare ALL 12 candidate structural features:\n"
        "  - Beware MYOPIC snapshot traps (e.g., checking only current hold stress or terminal circulation when the physics law "
        "depends on accumulated past loading history or full past circulation history!).\n"
        "  - Beware magnitude traps (e.g., Frobenius norm of an additively separable potential `V(x,y)=v_x(x)+v_y(y)` is positive even in laminar flow!).\n"
        "  - Select ONLY the upstream calibration mechanisms in `candidate_upstream_mechanism_ids` that physically calibrate this domain's inputs/readouts "
        "(exclude unrelated sentinel roots!).\n"
        "Return strict JSON matching the schema."
    )
    user_payload = {
        "task_specification": task_instruction_text,
        "candidate_guard_feature_names": list(M96_CANDIDATE_GUARD_FEATURES),
        "candidate_upstream_mechanism_ids": candidate_upstream_ids,
        "positive_c0_witness_feature_vectors": positive_c0_witness_features,
        "negative_c0_witness_feature_vectors": negative_c0_witness_features,
        "c0_solver_excerpt": c0_code[:3500],
        "c1_solver_excerpt": c1_code[:3500],
    }
    conv = [{"role": "user", "content": json.dumps(user_payload, indent=2)}]
    call_receipts: list[dict[str, Any]] = []
    proposals_log: list[dict[str, Any]] = []

    for att in range(1, max_attempts + 1):
        res = client.call_structured(
            system_prompt=sys_prompt,
            conversation_turns=conv,
            output_schema=M96_AUTONOMOUS_SPLIT_SCHEMA,
            stage_subdir=f"{stage_subdir}_guard_ind_att{att}",
        )
        call_receipts.append(res)
        if not res.get("ok") or not isinstance(res.get("parsed"), dict):
            proposals_log.append({"attempt": att, "ok": False, "error": res.get("error")})
            continue
        parsed = res["parsed"]
        w_chk = verify_contrastive_public_witness_guards(
            c0_guard=parsed["c0_guard"],
            c1_guard=parsed["c1_guard"],
            positive_c0_witness_features=positive_c0_witness_features,
            negative_c0_witness_features=negative_c0_witness_features,
        )
        p_chk = verify_upstream_parent_perturbation_sensitivity(
            family=family,
            candidate_upstream_ids=candidate_upstream_ids,
            proposed_upstream_ids=list(parsed["dependent_upstream_mechanism_ids"]),
            c1_canonical_code=c1_code,
            c1_entrypoint=c1_entrypoint,
            public_item=current_public_item,
            base_param_payload=current_param_payload,
        )
        proposals_log.append(
            {
                "attempt": att,
                "proposed": parsed,
                "contrastive_witness_check": w_chk,
                "parent_sensitivity_check": p_chk,
            }
        )
        if w_chk["contrastively_valid"] and p_chk["proposal_matches_empirical_sensitivity"]:
            return {
                "accepted": True,
                "structure_origin": "publicly_contrastively_validated",
                "attempts_used": att,
                "c0_guard": parsed["c0_guard"],
                "c1_guard": parsed["c1_guard"],
                "violated_component_id": str(parsed["violated_component_id"]).strip() or "constitutive_regime_law",
                "preserved_component_ids": [str(x) for x in parsed["preserved_component_ids"]],
                "validated_parent_mechanism_ids": p_chk["empirically_sensitive_upstream_ids"],
                "mechanistic_diagnosis": str(parsed["mechanistic_diagnosis"]),
                "proposals_log": proposals_log,
                "provider_call_receipts": call_receipts,
            }
        conv.append({"role": "assistant", "content": json.dumps(parsed)})
        conv.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "contrastive_or_sensitivity_rejection": {
                            "contrastive_reasons": w_chk["reasons"],
                            "empirical_parent_sensitivity": p_chk["sensitivity_max_diff_by_upstream_id"],
                        },
                        "instruction": "Fix `c0_guard`, `c1_guard`, and `dependent_upstream_mechanism_ids` so all positive/negative public witnesses separate cleanly and only empirically sensitive upstream parents are linked.",
                    }
                ),
            }
        )

    return {
        "accepted": False,
        "structure_origin": "rejected_contrastive_validation",
        "attempts_used": len(call_receipts),
        "proposals_log": proposals_log,
        "provider_call_receipts": call_receipts,
    }


# ============================================================================
# 4. True Multi-Parent `(wing_a, wing_b)` Diamond `PersistentMechanismGraph`
# ============================================================================


def initialize_m96_multiparent_mechanism_graph(
    *,
    family: str,
    forest: Any | None = None,
) -> tuple[PersistentMechanismGraph, M8VariationalPlanner, M94BindingRegistry]:
    """Initialize TRUE Multi-Parent (`parent_ids = (wing_a, wing_b)`) `PersistentMechanismGraph` (`§4.2`):
    - NO single-parent `facility_composite_calibration` wrapper!
    - Registers independent roots `wing_a_id`, `wing_b_id`, and `sentinel_shared_acoustic_root`.
    - Registers single-wing monitors `aux_wing_a_monitor` (`parent_ids=(wing_a_id,)`) and `aux_wing_b_monitor` (`parent_ids=(wing_b_id,)`)
      plus `sentinel_acoustic_calibration` (`parent_ids=(sentinel_id,)`) so selective propagation on `G2` (`wing_a` updated, `wing_b` retained)
      provably updates `aux_wing_a_monitor` + multi-parent `{C0, C1}` bindings while maintaining `0.0` drift on `aux_wing_b_monitor` & `sentinel`!
    """
    graph = PersistentMechanismGraph(forest=forest)
    planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)

    rng_s = np.random.default_rng(96001)
    xs_tr, zs_tr = rng_s.uniform(-1.5, 1.5, 36), rng_s.uniform(-1.0, 1.0, 36)
    ys_tr = 1.65 * xs_tr - 0.45 * zs_tr + rng_s.normal(scale=0.01, size=36)
    xs_va, zs_va = rng_s.uniform(-1.5, 1.5, 20), rng_s.uniform(-1.0, 1.0, 20)
    ys_va = 1.65 * xs_va - 0.45 * zs_va + rng_s.normal(scale=0.01, size=20)
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="sentinel_shared_acoustic_root",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("freq_norm", "temp_norm"),
            shared_weights={"freq_norm": 1.65, "temp_norm": -0.45},
            provenance_note="Unrelated acoustic sentinel (`0.0` drift across all revisions).",
        ),
        tree_id="tree_m96_sentinel",
    )
    planner.register_task(
        task_id="sentinel_acoustic_calibration",
        shared_upstream_id="sentinel_shared_acoustic_root",
        feature_names=("freq_norm", "temp_norm"),
        train_data={"x": {"freq_norm": xs_tr, "temp_norm": zs_tr}, "y": ys_tr},
        val_data={"x": {"freq_norm": xs_va, "temp_norm": zs_va}, "y": ys_va},
        initial_local_shrinkage=50.0,
    )

    if family == "alpha":
        wa_id, wb_id = "alpha_wing_a_calibration", "alpha_wing_b_calibration"
        cols_a = ("cmd_l0", "cmd_l1", "ind_l0", "ind_l1")
        cols_b = ("cmd_l2", "cmd_l3", "ind_l2", "ind_l3")
        w_a0 = {"cmd_l0": 1.0, "cmd_l1": 1.0, "ind_l0": 0.0, "ind_l1": 0.0}
        w_b0 = {"cmd_l2": 1.0, "cmd_l3": 1.0, "ind_l2": 0.0, "ind_l3": 0.0}
        t_id = "tree_m94_chain_alpha"
    else:
        wa_id, wb_id = "beta_wing_a_calibration", "beta_wing_b_calibration"
        cols_a = ("ref_b0", "ref_b1", "ind_b0", "ind_b1")
        cols_b = ("ref_b2", "ref_b3", "ind_b2", "ind_b3")
        w_a0 = {"ref_b0": 1.0, "ref_b1": 1.0, "ind_b0": 0.0, "ind_b1": 0.0}
        w_b0 = {"ref_b2": 1.0, "ref_b3": 1.0, "ind_b2": 0.0, "ind_b3": 0.0}
        t_id = "tree_m94_chain_beta"

    graph.register_mechanism(
        TypedMechanism(mechanism_id=wa_id, parent_id=None, role="shared_upstream_premise", feature_names=cols_a, version=1, shared_weights=dict(w_a0)),
        tree_id=t_id,
    )
    graph.register_mechanism(
        TypedMechanism(mechanism_id=wb_id, parent_id=None, role="shared_upstream_premise", feature_names=cols_b, version=1, shared_weights=dict(w_b0)),
        tree_id=t_id,
    )

    # Independent 3rd calibration bench (`aux_bench_c_calibration` + committed `aux_bench_c_monitor` task -> must maintain `0.0` drift!)
    xw = np.linspace(-1.2, 1.2, 16)
    cols_c = ("bench_c0", "bench_c1")
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="aux_bench_c_calibration",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=cols_c,
            version=1,
            shared_weights={"bench_c0": 1.15, "bench_c1": -0.20},
        ),
        tree_id="tree_m96_bench_c",
    )
    d_aux_c = {"x": {"bench_c0": xw, "bench_c1": np.ones_like(xw)}, "y": 1.15 * xw - 0.20}
    planner.register_task(
        task_id="aux_bench_c_monitor",
        shared_upstream_id="aux_bench_c_calibration",
        feature_names=cols_c,
        train_data=d_aux_c,
        val_data=deepcopy(d_aux_c),
        initial_local_shrinkage=50.0,
    )

    reg = M94BindingRegistry(graph)
    reg.parameter_epochs[wa_id] = {"parameter_mechanism_id": wa_id, "version": 1, "payload": dict(w_a0)}
    reg.parameter_epochs[wb_id] = {"parameter_mechanism_id": wb_id, "version": 1, "payload": dict(w_b0)}
    reg.sync_to_graph()
    return graph, planner, reg


def assemble_multiparent_parameter_payload(
    *,
    family: str,
    graph: PersistentMechanismGraph,
    parent_mechanism_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Assemble the 4-channel calibration payload strictly from the linked `parent_mechanism_ids` (`§4.2`).
    If a wing parent is NOT linked in `parent_mechanism_ids`, its channels remain at uncalibrated nominal `(1.0, 0.0)`!
    """
    linked = set(parent_mechanism_ids)
    if family == "alpha":
        gains = [1.0, 1.0, 1.0, 1.0]
        biases = [0.0, 0.0, 0.0, 0.0]
        if "alpha_wing_a_calibration" in linked and "alpha_wing_a_calibration" in graph.mechanisms:
            wa = graph.mechanisms["alpha_wing_a_calibration"].shared_weights
            gains[0], gains[1] = float(wa["cmd_l0"]), float(wa["cmd_l1"])
            biases[0], biases[1] = float(wa["ind_l0"]), float(wa["ind_l1"])
        if "alpha_wing_b_calibration" in linked and "alpha_wing_b_calibration" in graph.mechanisms:
            wb = graph.mechanisms["alpha_wing_b_calibration"].shared_weights
            gains[2], gains[3] = float(wb["cmd_l2"]), float(wb["cmd_l3"])
            biases[2], biases[3] = float(wb["ind_l2"]), float(wb["ind_l3"])
        return {"stress_gains": gains, "stress_biases": biases, "linked_parents": sorted(linked)}

    gains = [1.0, 1.0, 1.0, 1.0]
    offs = [0.0, 0.0, 0.0, 0.0]
    if "beta_wing_a_calibration" in linked and "beta_wing_a_calibration" in graph.mechanisms:
        wa = graph.mechanisms["beta_wing_a_calibration"].shared_weights
        gains[0], gains[1] = float(wa["ref_b0"]), float(wa["ref_b1"])
        offs[0], offs[1] = float(wa["ind_b0"]), float(wa["ind_b1"])
    if "beta_wing_b_calibration" in linked and "beta_wing_b_calibration" in graph.mechanisms:
        wb = graph.mechanisms["beta_wing_b_calibration"].shared_weights
        gains[2], gains[3] = float(wb["ref_b2"]), float(wb["ref_b3"])
        offs[2], offs[3] = float(wb["ind_b2"]), float(wb["ind_b3"])
    return {"bank_gains": gains, "bank_offsets": offs, "linked_parents": sorted(linked)}


def run_m96_multiparent_calibration_update(
    *,
    family: str,
    episode_id: str,
    public_item: dict[str, Any],
    graph: PersistentMechanismGraph,
    planner: M8VariationalPlanner,
    registry: M94BindingRegistry,
) -> dict[str, Any]:
    """Evaluate Wing A and Wing B independently via `M8VariationalPlanner` (`AncestorReweight` vs `NoChange`)
    and propagate ONLY along actual `parent_ids` edges (`minimal_affected_task_ids`), verifying selective isolation!
    """
    sent_tm = graph.active_task_models["sentinel_acoustic_calibration"]
    sent_va = graph.task_datasets["sentinel_acoustic_calibration"]["val"]
    sent_before = np.asarray(evaluate_committed_task_model(sent_tm, graph, sent_va)["predictions"], dtype=float)

    aux_c_tm = graph.active_task_models["aux_bench_c_monitor"]
    aux_c_va = graph.task_datasets["aux_bench_c_monitor"]["val"]
    aux_c_before = np.asarray(evaluate_committed_task_model(aux_c_tm, graph, aux_c_va)["predictions"], dtype=float)

    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        ch_ids = np.asarray(cal["lab_id"], dtype=int).ravel()
        x_in = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        y_out = np.asarray(cal["true_stress"], dtype=float).ravel()
        wing_specs = [
            ("wing_a", "alpha_wing_a_calibration", (0, 1), ("cmd_l0", "cmd_l1", "ind_l0", "ind_l1")),
            ("wing_b", "alpha_wing_b_calibration", (2, 3), ("cmd_l2", "cmd_l3", "ind_l2", "ind_l3")),
        ]
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        ch_ids = np.asarray(cal["bank_id"], dtype=int).ravel()
        x_in = np.asarray(cal["ref_signal"], dtype=float).ravel()
        y_out = np.asarray(cal["raw_readout"], dtype=float).ravel()
        wing_specs = [
            ("wing_a", "beta_wing_a_calibration", (0, 1), ("ref_b0", "ref_b1", "ind_b0", "ind_b1")),
            ("wing_b", "beta_wing_b_calibration", (2, 3), ("ref_b2", "ref_b3", "ind_b2", "ind_b3")),
        ]

    wing_receipts: dict[str, dict[str, Any]] = {}
    updated_binding_ids_union: list[str] = []
    affected_tasks_by_wing: dict[str, list[str]] = {}

    for w_tag, mech_id, (c0, c1), cols in wing_specs:
        w_before = deepcopy(graph.mechanisms[mech_id].shared_weights)
        m_wing = (ch_ids == c0) | (ch_ids == c1)
        n_w = int(np.sum(m_wing))
        if n_w < 6:
            wing_receipts[w_tag] = {
                "mechanism_id": mech_id,
                "n_rows": n_w,
                "selected_action": "NoChange_SparseEvidenceRetained",
                "version": int(graph.mechanisms[mech_id].version),
                "max_weight_drift": 0.0,
            }
            continue
        ch_w = ch_ids[m_wing]
        x_w = x_in[m_wing]
        y_w = y_out[m_wing]
        feats = {
            cols[0]: x_w * (ch_w == c0).astype(float),
            cols[1]: x_w * (ch_w == c1).astype(float),
            cols[2]: (ch_w == c0).astype(float),
            cols[3]: (ch_w == c1).astype(float),
        }
        n_tr = max(4, (2 * n_w) // 3)
        tr_d = {"x": {k: v[:n_tr] for k, v in feats.items()}, "y": y_w[:n_tr]}
        va_d = {"x": {k: v[n_tr:] for k, v in feats.items()}, "y": y_w[n_tr:]}
        tid_w = f"cal_{family}_{w_tag}_{episode_id}"
        for old_t in [t for t in list(graph.active_task_models.keys()) if t.startswith(f"cal_{family}_{w_tag}_")]:
            graph.active_task_models.pop(old_t, None)
            graph.task_datasets.pop(old_t, None)
            graph.mechanisms[mech_id].dependent_task_ids.discard(old_t)
        for old_m in [m for m in list(graph.mechanisms.keys()) if f"cal_{family}_{w_tag}_" in m and m != mech_id]:
            graph.mechanisms.pop(old_m, None)
        planner.register_task(
            task_id=tid_w,
            shared_upstream_id=mech_id,
            feature_names=cols,
            train_data=tr_d,
            val_data=va_d,
            initial_local_shrinkage=1e6,
        )
        rev = planner.evaluate_and_revise_task(tid_w, budget_remaining=1.0)

        act = str(rev["selected_candidate"]["action_type"])
        w_after = dict(graph.mechanisms[mech_id].shared_weights)
        w_drift = max(abs(float(w_after[k]) - float(w_before[k])) for k in cols)
        if act == "AncestorReweight":
            affected_tasks_by_wing[w_tag] = graph.minimal_affected_task_ids(mech_id)
            ub = registry.update_shared_parameter_epoch(
                parameter_mechanism_id=mech_id,
                new_version=int(graph.mechanisms[mech_id].version),
                parameter_payload=w_after,
                affected_chain_id=family,
                evidence_note=f"multiparent_reweight_{w_tag}_on_{episode_id}",
            )
            for b_i in ub:
                if b_i not in updated_binding_ids_union:
                    updated_binding_ids_union.append(b_i)
        else:
            registry.parameter_epochs[mech_id] = {
                "parameter_mechanism_id": mech_id,
                "version": int(graph.mechanisms[mech_id].version),
                "payload": w_after,
            }
            registry.sync_to_graph()

        wing_receipts[w_tag] = {
            "mechanism_id": mech_id,
            "n_rows": n_w,
            "selected_action": act,
            "val_log_predictive_gain": round(float(rev["selected_candidate"]["val_log_predictive_gain"]), 4),
            "version": int(graph.mechanisms[mech_id].version),
            "max_weight_drift": round(w_drift, 8),
        }

    all_wings = ["alpha_wing_a_calibration", "alpha_wing_b_calibration"] if family == "alpha" else ["beta_wing_a_calibration", "beta_wing_b_calibration"]
    full_payload = assemble_multiparent_parameter_payload(family=family, graph=graph, parent_mechanism_ids=all_wings)

    sent_after = np.asarray(evaluate_committed_task_model(sent_tm, graph, sent_va)["predictions"], dtype=float)
    aux_c_after = np.asarray(evaluate_committed_task_model(aux_c_tm, graph, aux_c_va)["predictions"], dtype=float)
    return {
        "family": family,
        "episode_id": episode_id,
        "wing_receipts": wing_receipts,
        "affected_tasks_by_wing": affected_tasks_by_wing,
        "updated_binding_ids": updated_binding_ids_union,
        "multiparent_fanout_count": len(updated_binding_ids_union),
        "parameter_payload": full_payload,
        "aux_bench_c_monitor_drift": round(float(np.max(np.abs(aux_c_after - aux_c_before))), 10),
        "sentinel_max_abs_prediction_drift": round(float(np.max(np.abs(sent_after - sent_before))), 10),
    }


def run_m96_equal_capability_pre_llm_trial(
    *,
    arm: str,
    family: str,
    task: HoldoutOODTask,
    public_item: dict[str, Any],
    prior_archive_entries: list[dict[str, Any]],
    flat_param_snapshot: dict[str, Any] | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
    readonly_dependencies: dict[str, str] | None = None,
    expected_dependency_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Equal-capability (`max_pre_llm_probes = 1`) pre-LLM execution trial for `M9.6`:
    - `Flat_Multiversion_Validated`: Retains all `public_gate_verified` canonical solvers `{C0, C1, ...}`, applies validated
      per-wing `2/3–1/3` holdout-MSE OLS 1-hop (`0` double-wrap!), and runs `1` clean public probe.
    - `Full_Mechanism_Graph_Autonomous`: Evaluates ONLY **model-proposed & contrastively-validated** autonomous guards (`or None` prior
      to first counterexample), assembles calibration strictly from `binding.parent_parameter_mechanism_ids`, and runs `1` clean public probe!
    """
    req_keys = expected_m96_public_output_keys(task)
    target_ep = task.declared_callable_entrypoints[0]

    if arm == "Flat_Multiversion_Validated":
        compat_raw = [
            e
            for e in reversed(prior_archive_entries)
            if sorted(e.get("output_schema_keys") or []) == req_keys
            and e.get("public_qualification_status", "public_gate_verified") == "public_gate_verified"
        ]
        dedup_compat: list[dict[str, Any]] = []
        seen_shas: set[str] = set()
        for ent in compat_raw:
            can_c = str(ent.get("canonical_algorithm_code") or ent["submitted_code"])
            c_sha = hashlib.sha256(can_c.encode("utf-8")).hexdigest()
            if c_sha not in seen_shas:
                seen_shas.add(c_sha)
                dedup_compat.append(ent)

        if not dedup_compat:
            return {
                "arm": arm,
                "accepted_pre_llm": False,
                "pre_llm_probes_used": 0,
                "retained_distinct_versions": 0,
                "reason": "no_publicly_verified_schema_compatible_prior_solver",
            }
        chosen_ent = dedup_compat[0]
        can_code = str(chosen_ent.get("canonical_algorithm_code") or chosen_ent["submitted_code"])
        can_ep = str(chosen_ent.get("canonical_entrypoint") or chosen_ent["entrypoint"])
        clean_snap = {k: v for k, v in (flat_param_snapshot or {}).items() if k != "wing_validation_decisions"}
        cand_code = render_parameterized_input_adapter(
            family=family,
            source_code=can_code,
            source_entrypoint=can_ep,
            target_entrypoint=target_ep,
            param_payload=clean_snap,
        )
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=task,
            candidate_code=cand_code,
            readonly_dependencies=readonly_dependencies,
            expected_dependency_sha256=expected_dependency_sha256,
        )
        return {
            "arm": arm,
            "accepted_pre_llm": pr["accepted"],
            "pre_llm_probes_used": 1,
            "retained_distinct_versions": len(dedup_compat),
            "selected_source_short_name": chosen_ent["short_name"],
            "selected_canonical_algorithm_code": can_code,
            "selected_canonical_entrypoint": can_ep,
            "parameter_payload": deepcopy(clean_snap),
            "candidate_code": (cand_code if pr["accepted"] else None),
            "probe_result": pr,
            "reason": ("accepted_by_shared_public_gate" if pr["accepted"] else "rejected_by_shared_public_scientific_gate"),
        }

    # Arm 2: `Full_Mechanism_Graph_Autonomous`
    assert graph is not None and registry is not None
    chain_bindings = [b for b in registry.bindings.values() if b.chain_id == family]
    shuf_seed = int(hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:8], 16)
    rng_ord = np.random.default_rng(shuf_seed)
    perm_idx = rng_ord.permutation(len(chain_bindings)).tolist() if chain_bindings else []
    shuffled_bindings = [chain_bindings[i] for i in perm_idx]

    guard_evals: list[dict[str, Any]] = []
    eligible: list[tuple[MechanismExecutableBinding, dict[str, Any], dict[str, float | None]]] = []
    for b in shuffled_bindings:
        parent_ids_b = list(dict.fromkeys([b.parameter_mechanism_id, *b.parent_parameter_mechanism_ids]))
        b_payload = assemble_multiparent_parameter_payload(family=family, graph=graph, parent_mechanism_ids=parent_ids_b)
        feat_vec = compute_generic_public_structural_features(public_item, b_payload)
        g_res = evaluate_m96_autonomous_guard(b.precondition_guard or None, feat_vec)
        schema_ok = sorted(b.output_schema_keys) == req_keys
        guard_evals.append(
            {
                "binding_id": b.binding_id,
                "branch_tag": b.branch_tag,
                "structure_origin": b.structure_origin,
                "parent_parameter_mechanism_ids": parent_ids_b,
                "guard_result": g_res,
                "schema_compatible": schema_ok,
            }
        )
        if g_res.get("state") == "TRUE" and schema_ok:
            eligible.append((b, b_payload, feat_vec))

    if not eligible:
        all_wings = ["alpha_wing_a_calibration", "alpha_wing_b_calibration"] if family == "alpha" else ["beta_wing_a_calibration", "beta_wing_b_calibration"]
        fallback_pay = assemble_multiparent_parameter_payload(family=family, graph=graph, parent_mechanism_ids=all_wings)
        return {
            "arm": arm,
            "accepted_pre_llm": False,
            "pre_llm_probes_used": 0,
            "parameter_payload": fallback_pay,
            "public_structural_features": compute_generic_public_structural_features(public_item, fallback_pay),
            "shuffled_candidate_order": [b.binding_id for b in shuffled_bindings],
            "guard_evaluations": guard_evals,
            "reason": (
                "interface_adaptation_required"
                if all(not ev["schema_compatible"] for ev in guard_evals)
                else "autonomous_guard_rejected_all_branches"
            ),
        }

    chosen_b, chosen_pay, chosen_feats = eligible[0]
    can_b_code = chosen_b.canonical_algorithm_code or chosen_b.parameterized_closure_code
    can_b_ep = chosen_b.canonical_entrypoint or chosen_b.parameterized_entrypoint
    cand_code = render_parameterized_input_adapter(
        family=family,
        source_code=can_b_code,
        source_entrypoint=can_b_ep,
        target_entrypoint=target_ep,
        param_payload={k: v for k, v in chosen_pay.items() if k != "linked_parents"},
        binding=chosen_b,
        graph=graph,
        registry=registry,
    )
    pr = execute_single_pre_llm_public_gate_probe_in_container(
        task=task,
        candidate_code=cand_code,
        readonly_dependencies=readonly_dependencies,
        expected_dependency_sha256=expected_dependency_sha256,
    )
    return {
        "arm": arm,
        "accepted_pre_llm": pr["accepted"],
        "pre_llm_probes_used": 1,
        "selected_binding_id": chosen_b.binding_id,
        "selected_branch_tag": chosen_b.branch_tag,
        "selected_structure_origin": chosen_b.structure_origin,
        "selected_canonical_algorithm_code": can_b_code,
        "selected_canonical_entrypoint": can_b_ep,
        "parameter_payload": deepcopy(chosen_pay),
        "public_structural_features": chosen_feats,
        "shuffled_candidate_order": [b.binding_id for b in shuffled_bindings],
        "guard_evaluations": guard_evals,
        "candidate_code": (cand_code if pr["accepted"] else None),
        "probe_result": pr,
        "reason": ("accepted_by_shared_public_gate" if pr["accepted"] else "bound_closure_rejected_by_public_gate"),
    }


# ============================================================================
# 5. Reference Dev-Seed Solvers (`0`-LLM Unit Testing on `DEV_M96_WORLD_SPECS` ONLY)
# ============================================================================


def build_m96_reference_dev_solver_code(family: str, entrypoint: str, branch: str) -> str:
    """Reference `C0` (`Stage S`), `C1` (`Stage C` / `U` / `M`), and `T_kkt` solvers used ONLY in `0`-LLM dev-world unit tests!"""
    from .m9_5_equal_tool_continual_attribution_world import build_reference_dev_solver_code

    if branch in {"C0", "C1"}:
        return build_reference_dev_solver_code(family, entrypoint, branch)

    assert family == "alpha" and branch == "T_kkt"
    return f'''from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def {entrypoint}(dataset: dict) -> dict:
    tr, te = dataset["train"], dataset["test"]
    past_dt = np.asarray(dataset["past_dt"], dtype=float)
    eval_t = np.asarray(dataset["eval_times"], dtype=float)
    cal = dataset["calibration_load_cells"]
    c_l = np.asarray(cal["lab_id"], dtype=int).ravel()
    c_x = np.asarray(cal["cmd_stress"], dtype=float).ravel()
    c_y = np.asarray(cal["true_stress"], dtype=float).ravel()
    s_g = np.ones(4, dtype=float); s_b = np.zeros(4, dtype=float)
    for lb in range(4):
        m = c_l == lb
        if int(np.sum(m)) >= 2:
            s_g[lb], s_b[lb] = np.linalg.lstsq(np.column_stack([c_x[m], np.ones(int(np.sum(m)))]), c_y[m], rcond=None)[0]
    labs = np.concatenate([tr["lab_id"], te["lab_id"]])
    strata = np.concatenate([tr["stratum"], te["stratum"]])
    X_raw = np.vstack([tr["coupon_features_raw"], te["coupon_features_raw"]])
    R_raw = np.vstack([tr["ref_coupons"], te["ref_coupons"]])
    X_std = X_raw.copy()
    for c in range(4):
        rc = R_raw[labs == c]
        cm, cs = np.nanmean(rc, axis=0), np.nanstd(rc, axis=0, ddof=0)
        v = np.isfinite(cm) & np.isfinite(cs) & (cs > 1e-8)
        X_std[labs == c] = (X_std[labs == c] - float(np.median(cm[v]))) / float(np.median(cs[v]))
    w_all = np.zeros(len(labs), dtype=float)
    for c in range(4):
        for s in range(3):
            ms = (labs == c) & (strata == s)
            if np.any(ms): w_all[ms] = 1.0 / float(np.sum(ms))
    w_all /= float(np.mean(w_all))
    for j in range(4):
        fin = np.isfinite(X_std[:, j])
        X_std[~fin, j] = float(np.sum(w_all[fin] * X_std[fin, j]) / max(float(np.sum(w_all[fin])), 1e-12))
    ntr = len(tr["lab_id"])
    X_tr, X_te = X_std[:ntr], X_std[ntr:]
    tr_l, te_l = np.asarray(tr["lab_id"], dtype=int), np.asarray(te["lab_id"], dtype=int)
    tr_ps = np.asarray(tr["past_stress_cmd"], dtype=float) * s_g[tr_l, None] + s_b[tr_l, None]
    tr_h0 = np.asarray(tr["stress_hold_cmd"], dtype=float) * s_g[tr_l] + s_b[tr_l]
    te_ps = np.asarray(te["past_stress_cmd"], dtype=float) * s_g[te_l, None] + s_b[te_l, None]
    te_h0 = np.asarray(te["stress_hold_cmd"], dtype=float) * s_g[te_l] + s_b[te_l]
    tr_im = np.asarray(tr["initial_memory_states"], dtype=float)
    te_im = np.asarray(te["initial_memory_states"], dtype=float)
    y_c = np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]
    y_r = np.asarray(tr["instant_rate_obs"], dtype=float)
    K = y_c.shape[1]

    def _prop(ps_a, im_a, t1, t2):
        taus = np.array([t1, t2], dtype=float)
        z = im_a.copy()
        for si, dt in enumerate(past_dt):
            dec = np.exp(-float(dt) / taus)
            z = z * dec[None, :] + (1.0 - dec[None, :]) * ps_a[:, si:si + 1]
        return z

    def _fit(t1, t2, iters=7):
        z = _prop(tr_ps, tr_im, t1, t2)
        gap = tr_h0[:, None] - z
        e1 = 1.0 - np.exp(-eval_t[None, :] / t1); e2 = 1.0 - np.exp(-eval_t[None, :] / t2)
        Dc = np.zeros((ntr * K, 7), dtype=float)
        Dc[:, 0] = (gap[:, 0:1] * e1).ravel(); Dc[:, 1] = (gap[:, 1:2] * e2).ravel()
        Dc[:, 2] = np.broadcast_to(eval_t[None, :], (ntr, K)).ravel()
        for j in range(4): Dc[:, 3 + j] = (X_tr[:, j:j + 1] * eval_t[None, :]).ravel()
        Dr = np.column_stack([gap[:, 0] / t1, gap[:, 1] / t2, np.ones(ntr), X_tr])
        D_all = np.vstack([Dc, 1.8 * Dr]); y_all = np.concatenate([y_c.ravel(), 1.8 * y_r])
        w = np.ones_like(y_all); th = np.zeros(7); s2 = 0.045 ** 2
        for _ in range(iters):
            sw = np.sqrt(w)
            th, *_ = np.linalg.lstsq(D_all * sw[:, None], y_all * sw, rcond=None)
            res = y_all - D_all @ th
            lc = float(np.sum(np.log1p((res[:ntr * K] ** 2) / s2)) + 0.5 * np.sum((res[ntr * K:] ** 2) / (0.025 ** 2)))
            w[:ntr * K] = 1.0 / (1.0 + (res[:ntr * K] ** 2) / s2)
            w[ntr * K:] = 1.0 / (1.0 + 0.2 * (res[ntr * K:] ** 2) / (0.025 ** 2))
        return lc, th

    best = (1e30, 0.55, 3.8, np.zeros(7))
    for t1 in np.geomspace(0.30, 1.25, 9):
        for t2 in np.geomspace(max(t1 * 1.9, 1.9), 6.2, 9):
            lv, th = _fit(float(t1), float(t2), iters=5)
            if lv < best[0]: best = (lv, float(t1), float(t2), th)
    ro = minimize(lambda u: _fit(float(np.exp(u[0])), float(np.exp(u[0]) + np.exp(u[1])), iters=8)[0],
                  x0=np.array([np.log(best[1]), np.log(best[2] - best[1])]), method="Nelder-Mead", options={{"maxiter": 85}})
    t1h = float(np.exp(ro.x[0])); t2h = float(t1h + np.exp(ro.x[1]))
    _, hw = _fit(t1h, t2h, iters=12)
    z_te = _prop(te_ps, te_im, t1h, t2h); gap_te = te_h0[:, None] - z_te; dr_te = hw[2] + X_te @ hw[3:7]
    ps_pred = (
        np.asarray(te["strain_0"], dtype=float)[:, None]
        + hw[0] * gap_te[:, 0:1] * (1.0 - np.exp(-eval_t[None, :] / t1h))
        + hw[1] * gap_te[:, 1:2] * (1.0 - np.exp(-eval_t[None, :] / t2h))
        + dr_te[:, None] * eval_t[None, :]
    )

    Z = np.asarray(dataset["Z_nuisance"], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + float(dataset["nuisance_ridge"]) * np.eye(Z.shape[1]), Z.T)
    cg = 1.0 + 0.25 * np.tanh(ps_pred[:, -1] - np.asarray(te["strain_0"], dtype=float))
    Sm = np.asarray(dataset["S_alloc_modes"], dtype=float) * cg[:, None]
    ya = np.asarray(dataset["y_alloc_obs"], dtype=float).ravel()
    S_ti = Sm - Z @ (Pz @ Sm); r_y = ya - Z @ (Pz @ ya)
    M = S_ti.shape[1]; lw, ub, bt = float(dataset["source_ridge"]), float(dataset["u_box"]), float(dataset["total_budget"])
    H = S_ti.T @ S_ti + lw * np.eye(M); f = S_ti.T @ r_y
    res_w = minimize(lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w), np.full(M, bt / M), jac=lambda w: H @ w - f,
                     bounds=[(0.0, ub)] * M, constraints=[{{"type": "eq", "fun": lambda w: float(np.sum(w)) - bt, "jac": lambda w: np.ones(M)}}],
                     method="SLSQP", options={{"ftol": 1e-14, "maxiter": 400}})
    w = np.clip(res_w.x, 0.0, ub)
    lo, hi = float(np.min(w) - ub), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, ub))) > bt: lo = mid
        else: hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, ub)
    rv = S_ti @ w - r_y; jv = 0.5 * float(rv @ rv) + 0.5 * lw * float(w @ w); g = H @ w - f
    best_nu, best_lb = float(g[0]), -1e30
    for nc in np.unique(g):
        lbc = jv - float(g @ w) + float(nc) * bt + ub * float(np.sum(np.minimum(0.0, g - float(nc))))
        if lbc > best_lb: best_lb, best_nu = lbc, float(nc)
    return {{"estimated_taus": np.array([t1h, t2h]), "predicted_strain": ps_pred, "allocation_weights": w,
            "kkt_certificate": {{"weights": w.tolist(), "nu_sum": best_nu, "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
                                 "lam_upper": np.maximum(0.0, best_nu - g).tolist(), "primal_objective": jv, "dual_lower_bound": float(min(best_lb, jv))}}}}
'''


# ============================================================================
# 6. `M9.6.1` Evidence-Bound Witnesses, Empirical Coverage & Immutable Branch Lineage (`N07/W04/W05`)
# ============================================================================


M96_LINEAGE_FOREST_KEY = "__m9_6_immutable_branch_lineage__"


def classify_public_probe_failure(probe_result: dict[str, Any]) -> dict[str, str]:
    """Strict 4-tier fail-closed failure classifier (`A09` / `C1.3`):
    Priority order:
      1. `INFRA_FAILURE` -> `failure_kind = 'timeout_or_infra'` (NEVER becomes a scientific negative witness!)
      2. `SCHEMA_FAILURE` -> `failure_kind = 'schema_or_interface'`
      3. `DOMAIN_REJECTION` -> `failure_kind = 'domain_precondition'` (explicit structured domain precondition rejection ONLY)
      4. `SCIENTIFIC_FAILURE` -> `failure_kind = 'scientific_or_numerical'` (schema valid, invariant/residual check failed)
    """
    sch_out = str(probe_result.get("schema_stdout") or "")
    sch_err = str(probe_result.get("schema_stderr") or "")
    sci_rec = probe_result.get("scientific_probe_receipt")
    sci_err = json.dumps(sci_rec, sort_keys=True) if isinstance(sci_rec, dict) else ""
    blob = f"{sch_out}\n{sch_err}\n{sci_err}"
    blob_low = blob.lower()

    sch_rc = probe_result.get("schema_exit_code")
    setup_ok = probe_result.get("setup_ok", True)
    timed_out = bool(
        probe_result.get("timed_out")
        or "[timeout:" in blob_low
        or "timed out" in blob_low
        or "timeoutexpired" in blob_low
    )
    infra_signal = sch_rc in {124, 125, 126, 127, 137, 139, 143} or (isinstance(sch_rc, int) and sch_rc < 0)
    infra_exc = any(
        tok in blob for tok in ("MemoryError", "BrokenPipeError", "ConnectionResetError", "No space left on device")
    )
    scientific_status = sci_rec.get("execution_status") if isinstance(sci_rec, dict) else None
    if (timed_out or (setup_ok is False) or infra_signal or infra_exc
            or scientific_status not in {None, "completed"}):
        return {"tier": "INFRA_FAILURE", "failure_kind": "timeout_or_infra"}

    if probe_result.get("accepted") is True:
        return {"tier": "NONE", "failure_kind": "none"}

    # Explicit structured domain precondition rejection (checked before generic schema failure ONLY when no syntax/import/type crash!)
    has_code_bug = any(
        tok in blob
        for tok in (
            "SyntaxError",
            "ImportError",
            "ModuleNotFoundError",
            "AttributeError",
            "NameError",
            "TypeError",
            "KeyError",
            "IndexError",
            "ZeroDivisionError",
            '"not_dict"',
            '"missing:',
            '"bad:',
        )
    )
    has_explicit_domain_rejection = (not has_code_bug) and any(
        tok in blob_low
        for tok in (
            "domain_precondition_rejected",
            "domainpreconditionerror",
            '"domain_applicable": false',
            "does not explain",
            "inapplicable_regime",
            "domain_precondition",
            "regime is supported",
            "unsupported regime",
            "outside applicable",
        )
    )
    if not isinstance(probe_result.get("schema_valid"), bool):
        return {"tier": "INFRA_FAILURE", "failure_kind": "timeout_or_infra"}
    if not bool(probe_result.get("schema_valid")):
        if has_explicit_domain_rejection:
            return {"tier": "DOMAIN_REJECTION", "failure_kind": "domain_precondition"}
        return {"tier": "SCHEMA_FAILURE", "failure_kind": "schema_or_interface"}

    if not isinstance(sci_rec, dict) or not (
        sci_rec.get("scientific_invariants_verified") is False or sci_rec.get("ok") is False
    ):
        return {"tier": "INFRA_FAILURE", "failure_kind": "timeout_or_infra"}
    if has_explicit_domain_rejection:
        return {"tier": "DOMAIN_REJECTION", "failure_kind": "domain_precondition"}
    return {"tier": "SCIENTIFIC_FAILURE", "failure_kind": "scientific_or_numerical"}


def compute_canonical_numeric_item_sha256(obj: Any) -> str:
    """Pickle-protocol & NumPy-module-path independent canonical SHA-256 (`R12`: identical across NumPy 1.x and 2.x)."""
    h = hashlib.sha256()

    def _feed(v: Any) -> None:
        if v is None:
            h.update(b"N;")
        elif isinstance(v, bool):
            h.update(b"B1;" if v else b"B0;")
        elif isinstance(v, (int, np.integer)):
            h.update(f"I:{int(v)};".encode("utf-8"))
        elif isinstance(v, (float, np.floating)):
            fv = float(v)
            if math.isnan(fv):
                h.update(b"F:NaN;")
            elif math.isinf(fv):
                h.update(b"F:+Inf;" if fv > 0 else b"F:-Inf;")
            else:
                # Normalize -0.0 -> 0.0 and pack IEEE-754 float64 little-endian
                fv = 0.0 if fv == 0.0 else fv
                h.update(b"F:" + np.asarray([fv], dtype="<f8").tobytes() + b";")
        elif isinstance(v, str):
            eb = v.encode("utf-8")
            h.update(f"S:{len(eb)}:".encode("utf-8") + eb + b";")
        elif isinstance(v, bytes):
            h.update(f"Y:{len(v)}:".encode("utf-8") + v + b";")
        elif isinstance(v, np.ndarray):
            h.update(f"A:{v.dtype.kind}:{v.shape}:".encode("utf-8"))
            if v.dtype.kind == "f":
                arr = np.asarray(v, dtype=np.float64)
                nan_mask = np.isnan(arr)
                clean = np.where(nan_mask, 0.0, arr)
                clean = np.where(clean == 0.0, 0.0, clean).astype("<f8")
                h.update(clean.tobytes("C"))
                h.update(np.packbits(nan_mask.ravel()).tobytes())
            elif v.dtype.kind in {"i", "u"}:
                h.update(np.asarray(v, dtype="<i8").tobytes("C"))
            elif v.dtype.kind == "b":
                h.update(np.asarray(v, dtype=np.uint8).tobytes("C"))
            else:
                for elem in v.ravel().tolist():
                    _feed(elem)
            h.update(b";")
        elif isinstance(v, dict):
            h.update(f"D:{len(v)}:".encode("utf-8"))
            for k in sorted(v.keys(), key=lambda x: str(x)):
                _feed(str(k))
                _feed(v[k])
            h.update(b";")
        elif isinstance(v, (list, tuple)):
            tag = "L" if isinstance(v, list) else "T"
            h.update(f"{tag}:{len(v)}:".encode("utf-8"))
            for item in v:
                _feed(item)
            h.update(b";")
        else:
            eb = repr(v).encode("utf-8")
            h.update(f"R:{len(eb)}:".encode("utf-8") + eb + b";")

    _feed(obj)
    return h.hexdigest()


VALID_WITNESS_EXECUTION_STATUSES: frozenset[str] = frozenset(
    {"completed", "timeout", "infrastructure_error", "not_executed"}
)
VALID_WITNESS_EVIDENCE_ORIGINS: frozenset[str] = frozenset(
    {"direct_component_probe", "joint_task_probe", "compatibility_inference", "legacy_unspecified"}
)
CURRENT_LINEAGE_SCHEMAS: frozenset[str] = frozenset(
    {"hegel_m9_7_1_lineage_state_v1", "hegel_m9_8_1_lineage_state_v2", "hegel_m9_9_1_lineage_state_v3"}
)


@dataclass
class SolverEvaluationWitness:
    """Evidence-bound evaluation witness (`M9.6.1` `N07` + `M9.7.1` `A09/A10` + `M9.8.1` `R04/R05/R07/R12`)."""

    witness_id: str
    binding_id: str
    solver_code_sha256: str
    parent_parameter_versions: dict[str, int]
    public_input_sha256: str
    verifier_version: str
    result: str  # 'passed' | 'failed' | 'not_tested'
    failure_kind: str  # 'none' | 'scientific_or_numerical' | 'domain_precondition' | 'schema_or_interface' | 'timeout_or_infra' | 'not_executed' | 'unverified_legacy'
    public_structural_features: dict[str, float | None]
    probe_wall_ms: float = 0.0
    witness_content_sha256: str = ""
    execution_status: str = "completed"
    evidence_origin: str = "direct_component_probe"
    metric_value: float | None = None
    probe_id: str = ""
    setup_source_sha256: str = ""
    canonical_numeric_sha256: str = ""
    dependency_bundle_sha256: str = ""
    verification_trust_state: str = "verified"  # 'verified' | 'unverified_legacy'
    witness_schema_version: int = 1

    def compute_expected_content_sha256(self) -> str:
        payload: dict[str, Any] = {
            "witness_id": self.witness_id,
            "binding_id": self.binding_id,
            "solver_code_sha256": self.solver_code_sha256,
            "parent_parameter_versions": {str(k): int(v) for k, v in sorted(self.parent_parameter_versions.items())},
            "public_input_sha256": self.public_input_sha256,
            "verifier_version": self.verifier_version,
            "result": self.result,
            "failure_kind": self.failure_kind,
            "public_structural_features": {
                str(k): (None if v is None else round(float(v), 10))
                for k, v in sorted(self.public_structural_features.items())
            },
        }
        if int(self.witness_schema_version) >= 2:
            payload.update(
                {
                    "witness_schema_version": int(self.witness_schema_version),
                    "execution_status": self.execution_status,
                    "evidence_origin": self.evidence_origin,
                    "metric_value": (None if self.metric_value is None else round(float(self.metric_value), 10)),
                    "probe_id": self.probe_id,
                    "setup_source_sha256": self.setup_source_sha256,
                    "canonical_numeric_sha256": self.canonical_numeric_sha256,
                    "dependency_bundle_sha256": self.dependency_bundle_sha256,
                    "verification_trust_state": self.verification_trust_state,
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def is_verified_positive_witness(self) -> bool:
        return (
            self.verification_trust_state == "verified"
            and self.execution_status == "completed"
            and self.evidence_origin in {"direct_component_probe", "joint_task_probe", "legacy_unspecified"}
            and self.result == "passed"
        )

    def is_genuine_scientific_counterexample(self) -> bool:
        return (
            self.verification_trust_state == "verified"
            and self.execution_status == "completed"
            and self.evidence_origin in {"direct_component_probe", "joint_task_probe", "legacy_unspecified"}
            and self.result == "failed"
            and self.failure_kind in {"scientific_or_numerical", "domain_precondition"}
        )

    def to_dict(self) -> dict[str, Any]:
        c_sha = self.witness_content_sha256 or self.compute_expected_content_sha256()
        out: dict[str, Any] = {
            "witness_id": self.witness_id,
            "binding_id": self.binding_id,
            "solver_code_sha256": self.solver_code_sha256,
            "parent_parameter_versions": dict(self.parent_parameter_versions),
            "public_input_sha256": self.public_input_sha256,
            "verifier_version": self.verifier_version,
            "result": self.result,
            "failure_kind": self.failure_kind,
            "public_structural_features": dict(self.public_structural_features),
            "probe_wall_ms": round(float(self.probe_wall_ms), 2),
            "witness_content_sha256": c_sha,
        }
        if int(self.witness_schema_version) >= 2:
            out.update(
                {
                    "witness_schema_version": int(self.witness_schema_version),
                    "execution_status": self.execution_status,
                    "evidence_origin": self.evidence_origin,
                    "metric_value": (None if self.metric_value is None else round(float(self.metric_value), 10)),
                    "probe_id": self.probe_id,
                    "setup_source_sha256": self.setup_source_sha256,
                    "canonical_numeric_sha256": self.canonical_numeric_sha256,
                    "dependency_bundle_sha256": self.dependency_bundle_sha256,
                    "verification_trust_state": self.verification_trust_state,
                }
            )
        return out

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        require_content_sha256: bool = True,
        allow_legacy_unverified_downgrade: bool = False,
    ) -> SolverEvaluationWitness:
        schema_v = int(d.get("witness_schema_version", 1) or 1)
        stored_sha = str(d.get("witness_content_sha256") or "")
        exec_st = str(d.get("execution_status") or ("completed" if d.get("result") in {"passed", "failed"} else "not_executed"))
        ev_orig = str(d.get("evidence_origin") or ("legacy_unspecified" if schema_v == 1 else "direct_component_probe"))
        if exec_st not in VALID_WITNESS_EXECUTION_STATUSES:
            raise ValueError(f"Invalid execution_status `{exec_st}` on witness `{d.get('witness_id')}` (`R04/R05`)")
        if ev_orig not in VALID_WITNESS_EVIDENCE_ORIGINS:
            raise ValueError(f"Invalid evidence_origin `{ev_orig}` on witness `{d.get('witness_id')}` (`R04`)")
        raw_mv = d.get("metric_value")
        mv = None if raw_mv is None else float(raw_mv)
        if mv is not None and not math.isfinite(mv):
            raise ValueError(f"Non-finite metric_value `{raw_mv}` on witness `{d.get('witness_id')}` (`R05`)")

        obj = cls(
            witness_id=str(d["witness_id"]),
            binding_id=str(d["binding_id"]),
            solver_code_sha256=str(d["solver_code_sha256"]),
            parent_parameter_versions={str(k): int(v) for k, v in (d.get("parent_parameter_versions") or {}).items()},
            public_input_sha256=str(d["public_input_sha256"]),
            verifier_version=str(d["verifier_version"]),
            result=str(d["result"]),
            failure_kind=str(d["failure_kind"]),
            public_structural_features={
                str(k): (None if v is None else float(v))
                for k, v in (d.get("public_structural_features") or {}).items()
            },
            probe_wall_ms=float(d.get("probe_wall_ms", 0.0) or 0.0),
            witness_content_sha256=stored_sha,
            execution_status=exec_st,
            evidence_origin=ev_orig,
            metric_value=mv,
            probe_id=str(d.get("probe_id") or ""),
            setup_source_sha256=str(d.get("setup_source_sha256") or ""),
            canonical_numeric_sha256=str(d.get("canonical_numeric_sha256") or ""),
            dependency_bundle_sha256=str(d.get("dependency_bundle_sha256") or ""),
            verification_trust_state=str(d.get("verification_trust_state") or "verified"),
            witness_schema_version=schema_v,
        )
        if not stored_sha:
            if allow_legacy_unverified_downgrade and not require_content_sha256:
                obj.verification_trust_state = "unverified_legacy"
                obj.result = "not_tested"
                obj.failure_kind = "unverified_legacy"
                obj.execution_status = "not_executed"
                obj.witness_schema_version = max(int(obj.witness_schema_version), 2)
                obj.witness_content_sha256 = obj.compute_expected_content_sha256()
                return obj
            raise ValueError(f"Missing witness_content_sha256 on SolverEvaluationWitness `{obj.witness_id}` (`R07/A10`)")
        exp_sha = obj.compute_expected_content_sha256()
        if stored_sha != exp_sha:
            raise ValueError(f"Tampered SolverEvaluationWitness `{obj.witness_id}` (`R07/A10`)")
        if allow_legacy_unverified_downgrade and not require_content_sha256:
            obj.verification_trust_state = "unverified_legacy"
            obj.result = "not_tested"
            obj.failure_kind = "unverified_legacy"
            obj.execution_status = "not_executed"
            obj.witness_schema_version = max(int(obj.witness_schema_version), 2)
            obj.witness_content_sha256 = obj.compute_expected_content_sha256()
        return obj


@dataclass
class EmpiricalCoverageReceipt:
    """Finite-witness content-addressed empirical coverage receipt (`M9.6.1` `§5` + `M9.7.1` `A08/A10` + `M9.8.1` `R06/R12`)."""

    receipt_id: str
    new_binding_id: str
    new_solver_sha256: str
    old_binding_id: str
    old_solver_sha256: str
    witness_ids: list[str]
    parameter_versions_by_witness: list[dict[str, int]]
    verifier_version: str
    status: str  # 'supported_on_witnesses' | 'refuted' | 'unknown' | 'unknown_requires_reverification'
    passed_witness_count: int
    evaluated_witness_count: int
    total_cpu_probe_wall_ms: float
    mean_probe_wall_ms_ratio: float
    witness_input_sha256s: list[str] = field(default_factory=list)
    parameter_payload_sha256s: list[str] = field(default_factory=list)
    receipt_schema_version: int = 1
    witness_setup_source_sha256s: list[str] = field(default_factory=list)
    witness_canonical_numeric_sha256s: list[str] = field(default_factory=list)
    dependency_bundle_sha256s: list[str] = field(default_factory=list)
    probe_ids: list[str] = field(default_factory=list)

    def compute_expected_receipt_id(self) -> str:
        payload: dict[str, Any] = {
            "new_binding_id": self.new_binding_id,
            "new_solver_sha256": self.new_solver_sha256,
            "old_binding_id": self.old_binding_id,
            "old_solver_sha256": self.old_solver_sha256,
            "witness_ids": list(self.witness_ids),
            "witness_input_sha256s": list(self.witness_input_sha256s),
            "parameter_versions_by_witness": [
                {str(k): int(v) for k, v in sorted(pv.items())} for pv in self.parameter_versions_by_witness
            ],
            "parameter_payload_sha256s": list(self.parameter_payload_sha256s),
            "verifier_version": self.verifier_version,
            "status": self.status,
            "passed_witness_count": int(self.passed_witness_count),
            "evaluated_witness_count": int(self.evaluated_witness_count),
        }
        if int(self.receipt_schema_version) >= 2:
            payload.update(
                {
                    "receipt_schema_version": int(self.receipt_schema_version),
                    "witness_setup_source_sha256s": list(self.witness_setup_source_sha256s),
                    "witness_canonical_numeric_sha256s": list(self.witness_canonical_numeric_sha256s),
                    "dependency_bundle_sha256s": list(self.dependency_bundle_sha256s),
                    "probe_ids": list(self.probe_ids),
                }
            )
        return "ecov_" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]

    def verify_integrity(self, *, allow_legacy_unverified: bool = False) -> None:
        valid_statuses = {"supported_on_witnesses", "refuted", "unknown", "unknown_requires_reverification"}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid EmpiricalCoverageReceipt status `{self.status}` (`R06/A08`)")
        if self.status == "unknown_requires_reverification" and int(self.passed_witness_count) != 0:
            raise ValueError(
                f"EmpiricalCoverageReceipt `{self.receipt_id}` with unknown_requires_reverification must have passed_witness_count == 0 (`Q08`)"
            )
        n_ev = int(self.evaluated_witness_count)
        n_pa = int(self.passed_witness_count)
        if n_ev < 0 or n_pa < 0 or n_pa > n_ev:
            raise ValueError(f"Invalid witness counts passed={n_pa}, evaluated={n_ev} on `{self.receipt_id}` (`R06`)")
        if len(self.witness_ids) != n_ev:
            raise ValueError(f"witness_ids length {len(self.witness_ids)} != evaluated_witness_count {n_ev} (`R06`)")
        if len(self.parameter_versions_by_witness) != n_ev:
            raise ValueError(
                f"parameter_versions_by_witness length {len(self.parameter_versions_by_witness)} != {n_ev} (`R06`)"
            )
        if (self.witness_input_sha256s or int(self.receipt_schema_version) >= 2) and len(self.witness_input_sha256s) != n_ev:
            raise ValueError(f"witness_input_sha256s length {len(self.witness_input_sha256s)} != {n_ev} (`R06`)")
        if (self.parameter_payload_sha256s or int(self.receipt_schema_version) >= 2) and len(self.parameter_payload_sha256s) != n_ev:
            raise ValueError(f"parameter_payload_sha256s length {len(self.parameter_payload_sha256s)} != {n_ev} (`R06`)")
        if int(self.receipt_schema_version) >= 2:
            if self.witness_setup_source_sha256s and len(self.witness_setup_source_sha256s) != n_ev:
                raise ValueError("witness_setup_source_sha256s length mismatch (`R12`)")
            if self.witness_canonical_numeric_sha256s and len(self.witness_canonical_numeric_sha256s) != n_ev:
                raise ValueError("witness_canonical_numeric_sha256s length mismatch (`R12`)")
        if self.status == "supported_on_witnesses" and (n_ev < 1 or n_pa != n_ev):
            raise ValueError(
                f"EmpiricalCoverageReceipt `{self.receipt_id}` claims supported_on_witnesses with {n_pa}/{n_ev} (`R06`)"
            )
        if allow_legacy_unverified and self.status == "unknown_requires_reverification":
            return
        exp = self.compute_expected_receipt_id()
        if (
            not isinstance(self.receipt_id, str)
            or not self.receipt_id.startswith("ecov_")
            or len(self.receipt_id) != 29
            or self.receipt_id != exp
        ):
            raise ValueError(
                f"Tampered EmpiricalCoverageReceipt (or malformed receipt_id) `{self.receipt_id}` != `{exp}` (`R06/A08`)"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "receipt_id": self.receipt_id,
            "new_binding_id": self.new_binding_id,
            "new_solver_sha256": self.new_solver_sha256,
            "old_binding_id": self.old_binding_id,
            "old_solver_sha256": self.old_solver_sha256,
            "witness_ids": list(self.witness_ids),
            "witness_input_sha256s": list(self.witness_input_sha256s),
            "parameter_versions_by_witness": deepcopy(self.parameter_versions_by_witness),
            "parameter_payload_sha256s": list(self.parameter_payload_sha256s),
            "verifier_version": self.verifier_version,
            "status": self.status,
            "passed_witness_count": self.passed_witness_count,
            "evaluated_witness_count": self.evaluated_witness_count,
            "total_cpu_probe_wall_ms": round(float(self.total_cpu_probe_wall_ms), 2),
            "mean_probe_wall_ms_ratio": round(float(self.mean_probe_wall_ms_ratio), 4),
            "zero_llm_calls_but_nonzero_cpu": True,
        }
        if int(self.receipt_schema_version) >= 2:
            out.update(
                {
                    "receipt_schema_version": int(self.receipt_schema_version),
                    "witness_setup_source_sha256s": list(self.witness_setup_source_sha256s),
                    "witness_canonical_numeric_sha256s": list(self.witness_canonical_numeric_sha256s),
                    "dependency_bundle_sha256s": list(self.dependency_bundle_sha256s),
                    "probe_ids": list(self.probe_ids),
                }
            )
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, allow_legacy_downgrade: bool = False) -> EmpiricalCoverageReceipt:
        schema_v = int(d.get("receipt_schema_version", 1) or 1)
        obj = cls(
            receipt_id=str(d.get("receipt_id") or ""),
            new_binding_id=str(d["new_binding_id"]),
            new_solver_sha256=str(d["new_solver_sha256"]),
            old_binding_id=str(d["old_binding_id"]),
            old_solver_sha256=str(d["old_solver_sha256"]),
            witness_ids=[str(x) for x in (d.get("witness_ids") or [])],
            parameter_versions_by_witness=[
                {str(k): int(v) for k, v in pv.items()} for pv in (d.get("parameter_versions_by_witness") or [])
            ],
            verifier_version=str(d["verifier_version"]),
            status=str(d["status"]),
            passed_witness_count=int(d["passed_witness_count"]),
            evaluated_witness_count=int(d["evaluated_witness_count"]),
            total_cpu_probe_wall_ms=float(d.get("total_cpu_probe_wall_ms", 0.0) or 0.0),
            mean_probe_wall_ms_ratio=float(d.get("mean_probe_wall_ms_ratio", 1.0) or 1.0),
            witness_input_sha256s=[str(x) for x in (d.get("witness_input_sha256s") or [])],
            parameter_payload_sha256s=[str(x) for x in (d.get("parameter_payload_sha256s") or [])],
            receipt_schema_version=schema_v,
            witness_setup_source_sha256s=[str(x) for x in (d.get("witness_setup_source_sha256s") or [])],
            witness_canonical_numeric_sha256s=[str(x) for x in (d.get("witness_canonical_numeric_sha256s") or [])],
            dependency_bundle_sha256s=[str(x) for x in (d.get("dependency_bundle_sha256s") or [])],
            probe_ids=[str(x) for x in (d.get("probe_ids") or [])],
        )
        if allow_legacy_downgrade:
            obj.status = "unknown_requires_reverification"
            obj.passed_witness_count = 0
            obj.receipt_id = obj.compute_expected_receipt_id()
            obj.verify_integrity()
            return obj
        obj.verify_integrity()
        return obj


class ImmutableBranchLineageController:
    """Maintains non-destructive multi-branch lineage (`b00 -> b01 -> b02 -> restore b00/b01`, `M9.6.1` `N07/W04/W05` + `M9.7.1` `A10` + `M9.8.1` `R04–R07`)."""

    def __init__(self, *, world_id: str, family: str, graph: PersistentMechanismGraph, registry: M94BindingRegistry) -> None:
        self.world_id = world_id
        self.family = family
        self.graph = graph
        self.registry = registry
        self.witness_ledger: list[SolverEvaluationWitness] = []
        self.coverage_receipts: list[EmpiricalCoverageReceipt] = []
        self._branch_seq = len([b for b in registry.bindings.values() if b.chain_id == family])
        self._hydrate_from_graph()

    def _lineage_store_key(self) -> str:
        return f"{self.world_id}::{self.family}"

    def _verify_witness_foreign_key(self, w: SolverEvaluationWitness) -> None:
        b = self.registry.bindings.get(w.binding_id)
        if b is None:
            raise ValueError(
                f"Lineage witness `{w.witness_id}` references unknown binding_id `{w.binding_id}` (`R07/A10`)"
            )
        auth_sha = b.authoritative_executed_code_sha256()
        if auth_sha and w.solver_code_sha256 != auth_sha:
            raise ValueError(
                f"Lineage witness `{w.witness_id}` solver_code_sha256 `{w.solver_code_sha256[:16]}` != "
                f"binding `{w.binding_id}` authoritative SHA `{auth_sha[:16]}` (`R07/A10`)"
            )

    def _hydrate_from_graph(self) -> None:
        root_blob = self.graph.forest.questions.get(M96_LINEAGE_FOREST_KEY)
        if not isinstance(root_blob, dict):
            return
        blob_schema = str(root_blob.get("schema") or "")
        entry = (root_blob.get("streams") or {}).get(self._lineage_store_key())
        if not isinstance(entry, dict):
            return
        w_raw = list(entry.get("witness_ledger") or [])
        c_raw = list(entry.get("coverage_receipts") or [])
        b_seq = int(entry.get("branch_seq", self._branch_seq))

        if (not blob_schema) or (blob_schema in CURRENT_LINEAGE_SCHEMAS):
            stored_sha = str(entry.get("state_sha256") or "")
            if not stored_sha:
                raise ValueError(
                    f"Missing state_sha256 on current-schema lineage checkpoint `{self._lineage_store_key()}` (`R07/A10`)"
                )
            expected_sha = hashlib.sha256(
                json.dumps(
                    {"branch_seq": b_seq, "witness_ledger": w_raw, "coverage_receipts": c_raw},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if stored_sha != expected_sha:
                raise ValueError(f"Tampered lineage checkpoint state on `{self._lineage_store_key()}` (`R07/A10`)")
            loaded_w = [SolverEvaluationWitness.from_dict(x, require_content_sha256=True) for x in w_raw]
            for w in loaded_w:
                self._verify_witness_foreign_key(w)
            loaded_c = [EmpiricalCoverageReceipt.from_dict(x, allow_legacy_downgrade=False) for x in c_raw]
            for c in loaded_c:
                if c.old_binding_id not in self.registry.bindings:
                    raise ValueError(
                        f"EmpiricalCoverageReceipt `{c.receipt_id}` references unknown old_binding_id `{c.old_binding_id}` (`R07`)"
                    )
            self.witness_ledger = loaded_w
            self.coverage_receipts = loaded_c
            self._branch_seq = max(self._branch_seq, b_seq)
            return

        # Explicit legacy pre-M9.7.1 schema migration (`Q08` idempotent `load -> migrate -> save -> reload` round-trip):
        # Downgrades trust/coverage to non-authoritative states AND recomputes canonical digests so `sync_to_graph()` + subsequent reloads succeed cleanly!
        migrated_w: list[SolverEvaluationWitness] = []
        for x in w_raw:
            bid = str(x.get("binding_id") or "")
            if bid not in self.registry.bindings:
                continue
            w_leg = SolverEvaluationWitness.from_dict(
                x, require_content_sha256=False, allow_legacy_unverified_downgrade=True
            )
            w_leg.verification_trust_state = "unverified_legacy"
            w_leg.result = "not_tested"
            w_leg.failure_kind = "unverified_legacy"
            w_leg.execution_status = "not_executed"
            w_leg.witness_schema_version = max(int(w_leg.witness_schema_version), 2)
            w_leg.witness_content_sha256 = w_leg.compute_expected_content_sha256()
            migrated_w.append(w_leg)
        migrated_c: list[EmpiricalCoverageReceipt] = []
        for x in c_raw:
            if str(x.get("old_binding_id") or "") not in self.registry.bindings:
                continue
            c_leg = EmpiricalCoverageReceipt.from_dict(x, allow_legacy_downgrade=True)
            c_leg.status = "unknown_requires_reverification"
            c_leg.passed_witness_count = 0
            c_leg.receipt_id = c_leg.compute_expected_receipt_id()
            migrated_c.append(c_leg)
        self.witness_ledger = migrated_w
        self.coverage_receipts = migrated_c
        self._branch_seq = max(self._branch_seq, b_seq)

    def _serialize_stream(self) -> dict[str, Any]:
        w_ser = [w.to_dict() for w in self.witness_ledger]
        c_ser = [c.to_dict() for c in self.coverage_receipts]
        for c_obj in self.coverage_receipts:
            c_obj.verify_integrity()
        st_sha = hashlib.sha256(
            json.dumps({"branch_seq": int(self._branch_seq), "witness_ledger": w_ser, "coverage_receipts": c_ser}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "world_id": self.world_id,
            "family": self.family,
            "branch_seq": int(self._branch_seq),
            "witness_ledger": w_ser,
            "coverage_receipts": c_ser,
            "state_sha256": st_sha,
        }

    def sync_to_graph(self) -> None:
        root_blob = self.graph.forest.questions.get(M96_LINEAGE_FOREST_KEY) or {}
        streams = dict(root_blob.get("streams") or {})
        key = self._lineage_store_key()
        if root_blob.get("schema") not in CURRENT_LINEAGE_SCHEMAS:
            # The schema applies to the whole root. Migrate siblings before publishing it.
            for sibling_key in tuple(streams):
                if sibling_key != key:
                    wid, family = sibling_key.rsplit("::", 1)
                    sibling = ImmutableBranchLineageController(
                        world_id=wid, family=family, graph=self.graph, registry=self.registry
                    )
                    streams[sibling_key] = sibling._serialize_stream()
        streams[key] = self._serialize_stream()
        self.graph.forest.questions[M96_LINEAGE_FOREST_KEY] = {
            **root_blob, "schema": "hegel_m9_8_1_lineage_state_v2", "streams": streams,
        }
        self.graph.sync_authoritative_state_to_forest()

    def record_coverage_receipt(self, receipt: EmpiricalCoverageReceipt) -> EmpiricalCoverageReceipt:
        receipt.verify_integrity()
        if receipt.old_binding_id not in self.registry.bindings:
            raise ValueError(f"EmpiricalCoverageReceipt references unknown old_binding_id `{receipt.old_binding_id}` (`R07`)")
        self.coverage_receipts.append(receipt)
        self.sync_to_graph()
        return receipt

    def record_tested_witness(
        self,
        *,
        binding: MechanismExecutableBinding,
        public_input_sha256: str,
        passed: bool,
        failure_kind: str,
        features: dict[str, float | None],
        probe_wall_ms: float = 0.0,
        verifier_version: str = "hegel_shared_math_handoff_kernel_v3_m961_three_layer",
        execution_status: str = "completed",
        evidence_origin: str = "joint_task_probe",
        metric_value: float | None = None,
        probe_id: str = "",
        setup_source_sha256: str = "",
        canonical_numeric_sha256: str = "",
        dependency_bundle_sha256: str = "",
        witness_schema_version: int = 2,
    ) -> SolverEvaluationWitness:
        if execution_status not in VALID_WITNESS_EXECUTION_STATUSES:
            raise ValueError(f"Invalid execution_status `{execution_status}` (`R04/R05`)")
        if evidence_origin not in VALID_WITNESS_EVIDENCE_ORIGINS or evidence_origin == "legacy_unspecified":
            raise ValueError(f"Invalid evidence_origin `{evidence_origin}` (`R04`)")
        if evidence_origin == "compatibility_inference" or execution_status == "not_executed":
            raise ValueError(
                "Refusing to record unexecuted compatibility_inference inside tested witness_ledger (`R04`)!"
            )
        if execution_status in {"timeout", "infrastructure_error"} and (
            passed or failure_kind in {"scientific_or_numerical", "domain_precondition"}
        ):
            raise ValueError(
                f"Refusing to record `{execution_status}` as passed or scientific failure (`R05`)!"
            )
        if metric_value is not None and not math.isfinite(float(metric_value)):
            raise ValueError(f"Non-finite metric_value `{metric_value}` rejected (`R05`)")
        wid = f"wit_{len(self.witness_ledger) + 1:04d}_{binding.binding_id}_{public_input_sha256[:8]}"
        eff_res = "not_tested" if execution_status in {"timeout", "infrastructure_error"} else ("passed" if passed else "failed")
        eff_fk = "timeout_or_infra" if execution_status in {"timeout", "infrastructure_error"} else ("none" if passed else failure_kind)
        w = SolverEvaluationWitness(
            witness_id=wid,
            binding_id=binding.binding_id,
            solver_code_sha256=binding.authoritative_executed_code_sha256(),
            parent_parameter_versions=dict(binding.parent_parameter_versions),
            public_input_sha256=public_input_sha256,
            verifier_version=verifier_version,
            result=eff_res,
            failure_kind=eff_fk,
            public_structural_features=deepcopy(features),
            probe_wall_ms=probe_wall_ms,
            execution_status=execution_status,
            evidence_origin=evidence_origin,
            metric_value=metric_value,
            probe_id=probe_id,
            setup_source_sha256=setup_source_sha256,
            canonical_numeric_sha256=canonical_numeric_sha256,
            dependency_bundle_sha256=dependency_bundle_sha256,
            verification_trust_state="verified",
            witness_schema_version=witness_schema_version,
        )
        w.witness_content_sha256 = w.compute_expected_content_sha256()
        self._verify_witness_foreign_key(w)
        self.witness_ledger.append(w)
        self.sync_to_graph()
        return w

    def record_tested_witness_from_probe_receipt(
        self,
        *,
        binding: MechanismExecutableBinding,
        probe_receipt: dict[str, Any],
        public_input_sha256: str,
        features: dict[str, float | None],
        evidence_origin: str,
        setup_source_sha256: str = "",
        canonical_numeric_sha256: str = "",
        dependency_bundle_sha256: str = "",
    ) -> SolverEvaluationWitness | None:
        """Construct a `SolverEvaluationWitness` strictly from an executed `probe_receipt` (`Q01` — callers cannot pass bare `passed=False`!).
        Returns `None` if `execution_status` is `'not_executed'`, `'timeout'`, or `'infrastructure_error'` so unexecuted/infra probes
        never enter the tested scientific witness ledger.
        """
        if not isinstance(probe_receipt, dict):
            raise TypeError("record_tested_witness_from_probe_receipt requires a dict `probe_receipt` (`Q01`)")
        pid = str(probe_receipt.get("probe_id") or "")
        exec_st = probe_receipt.get("execution_status")
        if exec_st in {"not_executed", "timeout", "infrastructure_error"}:
            return None
        if exec_st != "completed":
            raise ValueError("Probe receipt lacks an explicit completed execution status (Q01)")
        if not pid:
            raise ValueError("Executed `probe_receipt` must carry a non-empty `probe_id` (`Q01`)")
        receipt_numeric = probe_receipt.get("canonical_numeric_sha256")
        if (probe_receipt.get("public_input_sha256") != public_input_sha256
                or not receipt_numeric or (canonical_numeric_sha256 and receipt_numeric != canonical_numeric_sha256)):
            raise ValueError("Probe receipt and witness public input fingerprints disagree (Q01)")
        canonical_numeric_sha256 = receipt_numeric
        code_sha = binding.authoritative_executed_code_sha256()
        if evidence_origin == "direct_component_probe":
            if (probe_receipt.get("component_probe_executed") is not True
                    or probe_receipt.get("evidence_origin") != evidence_origin
                    or probe_receipt.get("candidate_code_sha256") != code_sha):
                raise ValueError("Direct probe receipt does not identify the executed binding (Q01)")
            rmse_v = probe_receipt.get("chosen_up_holdout_rmse")
            m_val = None if rmse_v is None else float(rmse_v)
            tier = probe_receipt.get("failure_tier")
            f_kind = probe_receipt.get("failure_kind")
            if tier == "SCHEMA_FAILURE" and f_kind == "schema_or_interface" and m_val is None:
                passed_b = False
            elif (probe_receipt.get("nonfinite_component_output") is True and m_val is None
                    and (tier, f_kind) == ("SCIENTIFIC_FAILURE", "scientific_or_numerical")):
                passed_b = False
            elif m_val is not None and math.isfinite(m_val) and m_val >= 0:
                passed_b = m_val <= 0.03
                expected = ("NONE", "none") if passed_b else ("SCIENTIFIC_FAILURE", "scientific_or_numerical")
                if (tier, f_kind) != expected:
                    raise ValueError("Direct probe verdict disagrees with the measured holdout RMSE (Q01)")
            else:
                raise ValueError("Direct probe receipt lacks a valid measured outcome (Q01)")
            w_ms = float(probe_receipt.get("probe_wall_ms") or probe_receipt.get("wall_ms") or 0.0)
        elif evidence_origin == "joint_task_probe":
            if (code_sha not in (probe_receipt.get("component_code_sha256s") or [])
                    or not isinstance(probe_receipt.get("accepted"), bool)):
                raise ValueError("Joint probe receipt does not identify the executed binding and verdict (Q01)")
            cls = classify_public_probe_failure(probe_receipt)
            if cls["tier"] == "INFRA_FAILURE":
                return None
            passed_b = bool(probe_receipt.get("accepted"))
            f_kind = "none" if passed_b else cls["failure_kind"]
            w_ms = float(probe_receipt.get("wall_ms") or probe_receipt.get("probe_wall_ms") or 0.0)
            m_val = None
        else:
            raise ValueError(f"Unsupported `evidence_origin` `{evidence_origin}` for ProbeReceipt witness (`Q01`)")
        return self.record_tested_witness(
            binding=binding,
            public_input_sha256=public_input_sha256,
            passed=passed_b,
            failure_kind=f_kind,
            features=features,
            probe_wall_ms=w_ms,
            execution_status="completed",
            evidence_origin=evidence_origin,
            metric_value=m_val,
            probe_id=pid,
            setup_source_sha256=setup_source_sha256,
            canonical_numeric_sha256=canonical_numeric_sha256,
            dependency_bundle_sha256=dependency_bundle_sha256,
        )

    def positive_witnesses_for(self, binding_id: str) -> list[SolverEvaluationWitness]:
        return [w for w in self.witness_ledger if w.binding_id == binding_id and w.is_verified_positive_witness()]

    def scientific_negative_witnesses_for(self, binding_id: str) -> list[SolverEvaluationWitness]:
        return [w for w in self.witness_ledger if w.binding_id == binding_id and w.is_genuine_scientific_counterexample()]

    def mint_immutable_conflict_branch(
        self,
        *,
        failed_binding_id: str,
        new_canonical_code: str,
        new_entrypoint: str,
        forest_node_id: str,
        qualification_protocol: str,
        output_schema_keys: list[str],
        validated_parent_ids: list[str],
        new_applicability_guard: dict[str, Any],
        updated_failed_branch_applicability_guard: dict[str, Any] | None,
        violated_component_id: str,
        preserved_component_ids: list[str],
        selection_priority: int,
        environment_digest: str = "hegel_clean_scipy_sandbox_v2",
        evidence_digest: str = "",
    ) -> MechanismExecutableBinding:
        if failed_binding_id not in self.registry.bindings:
            raise KeyError(f"Unknown failed_binding_id `{failed_binding_id}`")
        if not self.scientific_negative_witnesses_for(failed_binding_id):
            raise ValueError(
                f"Refusing to split `{failed_binding_id}` without a genuine tested scientific negative witness (`N07`)"
            )
        idx = self._branch_seq
        self._branch_seq += 1
        b_id = f"bind_{self.world_id}_b{idx:02d}"
        m_id = f"mech_{self.world_id}_b{idx:02d}"
        if b_id in self.registry.bindings:
            raise ValueError(f"Immutable branch ID collision `{b_id}`")

        code_sha = hashlib.sha256(new_canonical_code.encode("utf-8")).hexdigest()
        c_sha = hashlib.sha256(json.dumps(sorted(output_schema_keys)).encode("utf-8")).hexdigest()
        q_rec = self.registry.issue_qualification_receipt(
            qualification_protocol=qualification_protocol,
            verified_code_sha256=code_sha,
            forest_node_id=forest_node_id,
            contract_schema_sha256=c_sha,
            environment_digest=environment_digest,
            evidence_digest=evidence_digest,
            issued_reason=f"immutable_conflict_branch_from_{failed_binding_id}",
        )
        p_vers = {pid: int(self.graph.mechanisms[pid].version) for pid in validated_parent_ids}
        new_b = MechanismExecutableBinding(
            binding_id=b_id,
            chain_id=self.family,
            branch_tag=f"C{idx}_branch",
            mechanism_id=m_id,
            mechanism_version=1,
            forest_node_id=forest_node_id,
            component_id=f"{violated_component_id}_b{idx:02d}",
            preserved_component_ids=list(preserved_component_ids),
            violated_component_id=violated_component_id,
            module_stem=f"{self.world_id}_b{idx:02d}",
            module_digest=code_sha,
            function_name=new_entrypoint,
            qualification_id=qualification_protocol,
            qualification_protocol=qualification_protocol,
            qualification_receipt_id=q_rec.receipt_id,
            qualification_status="verified",
            parameter_mechanism_id=validated_parent_ids[0],
            parent_parameter_mechanism_ids=list(validated_parent_ids),
            parent_parameter_versions=p_vers,
            parameter_version=p_vers[validated_parent_ids[0]],
            structure_origin="publicly_contrastively_validated",
            environment_digest=environment_digest,
            evidence_receipt_id=evidence_digest,
            precondition_guard=deepcopy(new_applicability_guard),
            applicability_guard=deepcopy(new_applicability_guard),
            selection_priority=int(selection_priority),
            output_schema_keys=list(output_schema_keys),
            canonical_algorithm_code=new_canonical_code,
            canonical_entrypoint=new_entrypoint,
            canonical_algorithm_sha256=code_sha,
            input_coordinate_contract="raw_measured",
            parameterized_closure_code=new_canonical_code,
            parameterized_entrypoint=new_entrypoint,
            parameterized_closure_sha256=code_sha,
        )
        # Transactional update: validate & register `new_b` BEFORE updating `failed_b`'s applicability scope (NEVER overwriting `failed_b`'s code!)
        self.registry.register_binding(new_b, reason=f"mint_immutable_branch_from_{failed_binding_id}", strict_receipt_check=True)
        if updated_failed_branch_applicability_guard is not None:
            old_b = self.registry.bindings[failed_binding_id]
            old_b.applicability_guard = deepcopy(updated_failed_branch_applicability_guard)
            old_b.precondition_guard = deepcopy(updated_failed_branch_applicability_guard)
            self.registry.sync_to_graph()
        self.sync_to_graph()
        return new_b

    def select_branch_deterministically(
        self,
        *,
        req_schema_keys: list[str],
        feature_evaluator: Any,
    ) -> dict[str, Any]:
        """Deterministic selection separating `applicability_guard` (overlap allowed!) from `selection_priority` (`N07/W04` — NO random tie-break!)."""
        candidates = sorted(
            [b for b in self.registry.bindings.values() if b.chain_id == self.family and b.qualification_status == "verified"],
            key=lambda x: (-int(x.selection_priority), x.binding_id),
        )
        eval_rows = []
        applicable: list[tuple[MechanismExecutableBinding, dict[str, float | None]]] = []
        any_unknown = False
        for b in candidates:
            if sorted(b.output_schema_keys) != sorted(req_schema_keys):
                eval_rows.append({"binding_id": b.binding_id, "schema_ok": False, "guard_state": "SCHEMA_MISMATCH"})
                continue
            feats = feature_evaluator(b)
            eff_g = b.applicability_guard or b.precondition_guard or None
            g_res = evaluate_m96_autonomous_guard(eff_g, feats, allow_open_feature_keys=True)
            st = str(g_res.get("state") or "UNKNOWN")
            eval_rows.append(
                {
                    "binding_id": b.binding_id,
                    "branch_tag": b.branch_tag,
                    "selection_priority": b.selection_priority,
                    "schema_ok": True,
                    "guard_state": st,
                }
            )
            if st == "TRUE":
                applicable.append((b, feats))
            elif st == "UNKNOWN":
                any_unknown = True

        if applicable:
            chosen_b, chosen_f = applicable[0]
            return {
                "selected": chosen_b,
                "selected_features": chosen_f,
                "overlapping_applicable_ids": [b.binding_id for b, _ in applicable],
                "evaluations": eval_rows,
                "status": ("selected_from_overlap_by_priority" if len(applicable) > 1 else "selected_unique_applicable"),
            }
        return {
            "selected": None,
            "selected_features": None,
            "overlapping_applicable_ids": [],
            "evaluations": eval_rows,
            "status": ("unknown_requires_probe" if any_unknown else "no_applicable_branch"),
        }


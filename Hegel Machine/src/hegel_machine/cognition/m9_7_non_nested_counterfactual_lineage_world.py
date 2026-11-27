"""Hegel M9.7: Non-Nested Multi-Conflict Lineage, Strict Present-Matched Counterfactual Memory & Demand-Gated Continual World.

Addresses 100% of `Hegel_M9_6_Audit` scientific & semantic findings (`N01–N11`):
  1. **Genuine Mutual Non-Subsumption Across 3 Regimes (`C0`, `C1`, `C2`, `§7.3`)**:
     - Every single-regime solver (`C0`, `C1`, `C2`) passes its own regime (`<= 0.070` MAE on Alpha; `<= 2e-4` rel-L2 on Beta)
       and FAILS the other two regimes by `4x–50x` margin unless an explicit conditional dispatcher or multi-branch guard is used!
     - `Flat_EqualInfo_Multiversion` is 100% unhandicapped: receives identical prefix-locked `0444` calibration history,
       identical `/tmp/hegel_continual_archive/` verified canonical solvers, identical validated per-channel OLS, and full
       freedom to write unified/piecewise solvers!
  2. **Strict Present-Matched Counterfactual History Pair `(R0, U_iso)` (`N06 / §7.4`)**:
     - `R0` (`Step 2`, governed by `C0`) and `U_iso` (`Step 5`, governed by `C1`) share **byte-for-byte identical** present
       inputs, thresholds (`crit_stress = 2.65`, `critical_vortex_circulation = 0.50`), hold vectors, feature matrices,
       calibration sweeps, and terminal `t = 0-` snapshots (`max_abs_present_diff == 0.0`!), differing ONLY in prior trajectory!
  3. **Multi-Conflict Immutable Lineage & Restoration (`C0 -> C1 -> R0 -> C2 -> R1 -> U_iso -> G_perm -> M_couple`, `N07`)**:
     - Tests `C1` conflict, return to `C0` (`R0`), second independent conflict `C2`, return to `C1` (`R1`), strict counterfactual
       memory `U_iso`, selective permuted upstream shift + active wired nuisance shift + outlier spike (`G_perm`), and
       unseen multi-parent + counterfactual history composition (`M_couple`).
  4. **Anonymized Permuted Multi-Parent Calibration + Genuinely Wired Nuisance Parent (`N05 / W03`)**:
     - Channels `{0, 1, 2, 3}` are randomly permuted per world onto opaque parent IDs `cal_node_p0`, `cal_node_p1`;
       `cal_node_p2` genuinely scales `aux_telemetry` inside the candidate's input dict while having `0.0` causal effect
       on physical truth; perturbation sensitivity runs strictly inside an isolated Docker container (`run_in_container=True`).
  5. **Neutral Primitive Reducer Guard DSL (`obs_01..obs_08` + Kleene-3 `all_of`, `N05 / W01`)**:
     - Zero pre-cooked composite formula names and zero prompt hints naming distractors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile
from time import monotonic
from typing import Any

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import CLEAN_SCIPY_SANDBOX_IMAGE
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TaskModel,
    TypedMechanism,
    evaluate_committed_task_model,
)
from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY
from .m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    decode_pure_json_numerical_text,
    evaluate_three_layer_scientific_contract,
)
from .m9_4_continual_mechanism_binding_world import (
    M94BindingRegistry,
    MechanismExecutableBinding,
    validate_and_stage_readonly_dependencies_setup_py,
)
from .m9_5_equal_tool_continual_attribution_world import (
    execute_single_pre_llm_public_gate_probe_in_container,
    m95_task_to_frontier_task,
    render_parameterized_input_adapter,
)
from .m9_6_autonomous_structure_continual_world import (
    EmpiricalCoverageReceipt,
    ImmutableBranchLineageController,
    evaluate_m96_autonomous_guard,
    verify_upstream_parent_perturbation_sensitivity,
)

M97_ARMS: tuple[str, ...] = (
    "Flat_EqualInfo_Multiversion",
    "Hegel_DemandGated_Graph",
)

M97_PRIMITIVE_OBSERVABLE_NAMES: tuple[str, ...] = (
    "obs_01_past_series_exceedance_integral",
    "obs_02_present_snapshot_exceedance_max",
    "obs_03_boundary_or_banding_control_scalar",
    "obs_04_terminal_step_level",
    "obs_05_spatial_or_panel_anisotropy_ratio",
    "obs_06_aux_telemetry_rms",
    "obs_07_input_frobenius_scale",
    "obs_08_declared_threshold_metadata",
)


@dataclass(frozen=True)
class M97WorldSpec:
    world_id: str
    cluster_id: str  # 1-to-1 independent cluster (`0` shared pair compilation across worlds!)
    family: str  # 'alpha' | 'beta'
    split: str  # 'dev' | 'confirmation'
    world_seed: int
    channel_perm: tuple[int, int, int, int]  # `cal_node_p0` owns `perm[:2]`, `cal_node_p1` owns `perm[2:]`
    v2_gains: tuple[float, float, float, float]
    v2_offsets: tuple[float, float, float, float]
    v3_p0_gains: tuple[float, float]
    v3_p0_offsets: tuple[float, float]
    schedule_events: tuple[str, ...] = ("C1", "R0", "C2", "R1", "U_iso", "G_perm", "M_couple")


# Pre-freeze unit tests touch ONLY `DEV_M97_WORLD_SPECS` (`0` confirmation seed touch!)
DEV_M97_WORLD_SPECS: tuple[M97WorldSpec, ...] = (
    M97WorldSpec(
        world_id="dev_alpha_m97_00",
        cluster_id="dev_cluster_a0",
        family="alpha",
        split="dev",
        world_seed=980100,
        channel_perm=(2, 0, 3, 1),
        v2_gains=(1.32, 0.78, 1.18, 0.84),
        v2_offsets=(0.14, -0.16, -0.11, 0.12),
        v3_p0_gains=(0.74, 1.38),
        v3_p0_offsets=(0.19, -0.21),
    ),
    M97WorldSpec(
        world_id="dev_beta_m97_00",
        cluster_id="dev_cluster_b0",
        family="beta",
        split="dev",
        world_seed=980200,
        channel_perm=(1, 3, 0, 2),
        v2_gains=(1.28, 0.81, 1.19, 0.86),
        v2_offsets=(0.13, -0.14, 0.09, -0.11),
        v3_p0_gains=(1.36, 0.75),
        v3_p0_offsets=(-0.18, 0.16),
    ),
)

# `6` Completely Independent Confirmation Clusters (`n = 6` -> `6/6` two-sided sign-flip `p = 0.03125 < 0.05`!)
CONFIRMATION_M97_WORLD_SPECS: tuple[M97WorldSpec, ...] = (
    M97WorldSpec(
        world_id="alpha_m97_c01",
        cluster_id="cluster_01_alpha",
        family="alpha",
        split="confirmation",
        world_seed=991100,
        channel_perm=(3, 1, 0, 2),
        v2_gains=(1.34, 0.77, 1.21, 0.83),
        v2_offsets=(0.15, -0.17, -0.12, 0.11),
        v3_p0_gains=(1.39, 0.72),
        v3_p0_offsets=(-0.19, 0.18),
    ),
    M97WorldSpec(
        world_id="alpha_m97_c02",
        cluster_id="cluster_02_alpha",
        family="alpha",
        split="confirmation",
        world_seed=991200,
        channel_perm=(1, 2, 3, 0),
        v2_gains=(0.79, 1.31, 0.85, 1.24),
        v2_offsets=(-0.14, 0.16, 0.10, -0.13),
        v3_p0_gains=(0.74, 1.37),
        v3_p0_offsets=(0.20, -0.17),
    ),
    M97WorldSpec(
        world_id="alpha_m97_c03",
        cluster_id="cluster_03_alpha",
        family="alpha",
        split="confirmation",
        world_seed=991300,
        channel_perm=(0, 3, 2, 1),
        v2_gains=(1.29, 0.82, 1.17, 0.76),
        v2_offsets=(0.12, -0.15, 0.14, -0.11),
        v3_p0_gains=(0.73, 1.36),
        v3_p0_offsets=(-0.18, 0.17),
    ),
    M97WorldSpec(
        world_id="beta_m97_c01",
        cluster_id="cluster_04_beta",
        family="beta",
        split="confirmation",
        world_seed=992100,
        channel_perm=(2, 1, 3, 0),
        v2_gains=(1.31, 0.79, 1.22, 0.85),
        v2_offsets=(0.14, -0.13, -0.10, 0.12),
        v3_p0_gains=(0.76, 1.35),
        v3_p0_offsets=(0.17, -0.16),
    ),
    M97WorldSpec(
        world_id="beta_m97_c02",
        cluster_id="cluster_05_beta",
        family="beta",
        split="confirmation",
        world_seed=992200,
        channel_perm=(3, 0, 1, 2),
        v2_gains=(0.81, 1.33, 0.86, 1.20),
        v2_offsets=(-0.15, 0.14, 0.11, -0.12),
        v3_p0_gains=(1.38, 0.74),
        v3_p0_offsets=(-0.17, 0.19),
    ),
    M97WorldSpec(
        world_id="beta_m97_c03",
        cluster_id="cluster_06_beta",
        family="beta",
        split="confirmation",
        world_seed=992300,
        channel_perm=(1, 0, 2, 3),
        v2_gains=(1.27, 0.78, 1.18, 0.84),
        v2_offsets=(0.13, -0.16, -0.09, 0.11),
        v3_p0_gains=(0.75, 1.34),
        v3_p0_offsets=(0.18, -0.15),
    ),
)


# ============================================================================
# 1. Embedded Self-Contained Cohort Generators (`3` Non-Nested Regimes + Strict `(R0, U_iso)` Twins!)
# ============================================================================

_M97_ALPHA_GEN_PY = r'''
def _gen_m97_alpha_cohort(
    present_seed,
    taus_pair,
    theta_vec,
    *,
    history_seed=None,
    stress_gains=(1.0, 1.0, 1.0, 1.0),
    stress_biases=(0.0, 0.0, 0.0, 0.0),
    aux_gain=1.0,
    cal_mode="full_40",
    p0_channels=(0, 1),
    p1_channels=(2, 3),
    n_train=180,
    n_test=72,
    H=18,
    K=10,
    crit_stress=2.65,
    regime_kind="C0",  # 'C0' | 'C1' | 'C2'
    regime_weights=(0.0, 0.0),
    glitch_frac=0.07,
):
    """Generates Alpha rheology cohort where `(R0, U_iso)` called with the SAME `present_seed` share
    BYTE-FOR-BYTE IDENTICAL present inputs (`crit_stress`, `shear_banding_index`, `stress_hold_cmd`, `strain_0`,
    `coupon_features_raw`, `ref_coupons`, `lab_id`, `stratum`, `eval_times`, `past_dt`, `calibration_load_cells`,
    `aux_chamber_telemetry`, AND final past snapshot `past_stress_cmd[:, -1]`!), differing ONLY in `past_stress_cmd[:, :-1]`!
    """
    rng_p = np.random.default_rng(int(present_seed))
    rng_h = np.random.default_rng(int(present_seed + 777 if history_seed is None else history_seed))
    taus = np.asarray(taus_pair, dtype=float)
    theta = np.asarray(theta_vec, dtype=float)
    rw = np.asarray(regime_weights, dtype=float)
    s_gains = np.asarray(stress_gains, dtype=float)
    s_biases = np.asarray(stress_biases, dtype=float)
    cs = float(crit_stress)
    sb_idx = 0.82 if regime_kind == "C2" else 0.12

    past_dt = np.round(rng_p.uniform(0.20, 0.80, size=H), 4)
    eval_times = np.round(np.linspace(0.30, 5.20, K), 4)

    if cal_mode == "full_40":
        cal_labs = np.repeat(np.arange(4, dtype=int), 10)
        cal_cmd = np.round(rng_p.uniform(-2.1, 2.2, size=40), 6)
        cal_true = s_gains[cal_labs] * cal_cmd + s_biases[cal_labs] + rng_p.normal(scale=1e-4, size=40)
    elif cal_mode == "sparse_spot_ch0":
        ch0 = int(p0_channels[0])
        cal_labs = np.array([ch0, ch0], dtype=int)
        cal_cmd = np.array([-1.30, 1.40], dtype=float)
        cal_true = s_gains[cal_labs] * cal_cmd + s_biases[cal_labs] + rng_p.normal(scale=1e-4, size=2)
    elif cal_mode == "permuted_p0_shift_plus_p1_emi":
        la = np.repeat(np.asarray(p0_channels, dtype=int), 12)
        ca = rng_p.uniform(-2.0, 2.2, size=24)
        ta = s_gains[la] * ca + s_biases[la] + rng_p.normal(scale=1e-4, size=24)
        p1a, p1b = int(p1_channels[0]), int(p1_channels[1])
        lb = np.array([p1a, p1b, p1a, p1b, p1a, p1b], dtype=int)
        cb = np.array([-1.60, 1.50, 1.20, -1.30, -0.90, 1.10], dtype=float)
        emi = np.array([4.90, -4.75, 4.60, -4.55, 0.0, 0.0], dtype=float)
        tb = s_gains[lb] * cb + s_biases[lb] + emi
        cal_labs = np.concatenate([la, lb])
        cal_cmd = np.concatenate([ca, cb])
        cal_true = np.concatenate([ta, tb])
    else:
        raise ValueError(f"Unknown cal_mode: {cal_mode}")

    p_probs = np.array([[0.42, 0.33, 0.25], [0.22, 0.48, 0.30], [0.30, 0.24, 0.46], [0.26, 0.36, 0.38]])
    f_gains = np.array([0.82, 1.28, 0.94, 1.42])
    f_offs = np.array([-0.65, 0.48, -0.25, 0.80])

    def _make_present_panel(n_rows):
        labs = rng_p.integers(0, 4, size=n_rows)
        labs[:16] = np.tile(np.arange(4, dtype=int), 4)
        strata = np.array([rng_p.choice(3, p=p_probs[c]) for c in labs], dtype=int)
        Z = rng_p.normal(loc=0.25 * (strata[:, None] - 1.0), scale=1.0, size=(n_rows, 4))
        R_lat = rng_p.normal(size=(n_rows, 32))
        X_raw = f_offs[labs, None] + f_gains[labs, None] * Z
        R_raw = f_offs[labs, None] + f_gains[labs, None] * R_lat
        X_raw[rng_p.random(X_raw.shape) < 0.025] = np.nan
        R_raw[rng_p.random(R_raw.shape) < 0.040] = np.nan
        s0 = np.round(rng_p.normal(scale=0.32, size=n_rows), 6)
        # Present hold stress is ALWAYS strictly sub-critical (`[-0.76*cs, 0.76*cs]`) so `(R0, U_iso)` share IDENTICAL `hold_true`!
        hold_true = np.round(rng_p.uniform(-0.76 * cs, 0.76 * cs, size=n_rows), 6)
        last_past_true = np.round(rng_p.uniform(-0.65 * cs, 0.65 * cs, size=n_rows), 6)
        aux_tel = np.round(float(aux_gain) * rng_p.normal(scale=1.1, size=(n_rows, 3)), 6)
        return labs, strata, Z, X_raw, R_raw, s0, hold_true, last_past_true, aux_tel

    tr_p = _make_present_panel(n_train)
    te_p = _make_present_panel(n_test)

    def _sim_from_history(panel_tup, *, add_glitches=False):
        labs_a, strata_a, Z_lat, X_r, R_r, s0, hold_true, last_p_true, aux_tel = panel_tup
        N = len(labs_a)
        hist_true = np.zeros((N, H), dtype=float)
        if regime_kind == "C1":
            # Prior loading crossed `crit_stress` on `s in 0..H-3`, then unloaded cleanly to `last_p_true` at `H-1`!
            raw_h = rng_h.uniform(-0.72 * cs, 0.72 * cs, size=(N, H - 1))
            hi_rows = np.arange((2 * N) // 3)
            sgn = np.where(rng_h.random(len(hi_rows)) < 0.52, 1.0, -1.0)
            raw_h[hi_rows, 3:11] = sgn[:, None] * rng_h.uniform(cs + 0.45, cs + 1.35, size=(len(hi_rows), 8))
            hist_true[:, :-1] = raw_h
            hist_true[:, -1] = last_p_true
        else:
            raw_h = rng_h.normal(scale=0.42 * cs, size=(N, H - 1))
            for s in range(1, H - 1):
                raw_h[:, s] = 0.48 * raw_h[:, s - 1] + math.sqrt(1.0 - 0.48 ** 2) * raw_h[:, s]
            hist_true[:, :-1] = np.clip(raw_h, -0.78 * cs, 0.78 * cs)
            hist_true[:, -1] = last_p_true

        hist_cmd = (hist_true - s_biases[labs_a, None]) / s_gains[labs_a, None]
        hold_cmd = (hold_true - s_biases[labs_a]) / s_gains[labs_a]

        z_cur = np.zeros((N, 2), dtype=float)
        for s in range(H):
            dec = np.exp(-past_dt[s] / taus)
            z_cur = z_cur * dec[None, :] + (1.0 - dec[None, :]) * hist_true[:, s : s + 1]
        gap0 = hold_true - z_cur[:, 0]
        gap1 = hold_true - z_cur[:, 1]
        e0 = 1.0 - np.exp(-eval_times[None, :] / taus[0])
        e1 = 1.0 - np.exp(-eval_times[None, :] / taus[1])
        drift = theta[2] + Z_lat @ theta[3:7]
        prony_c = theta[0] * gap0[:, None] * e0 + theta[1] * gap1[:, None] * e1
        prony_r = theta[0] * gap0 / taus[0] + theta[1] * gap1 / taus[1]

        if regime_kind == "C0":
            clean = s0[:, None] + prony_c + drift[:, None] * eval_times[None, :]
            true_rate0 = prony_r + drift
        elif regime_kind == "C1":
            d_cryst = np.sum((np.maximum(0.0, np.abs(hist_true) - cs) ** 2) * past_dt[None, :], axis=1)
            act = np.tanh(d_cryst)
            phi1 = np.log1p(eval_times)
            clean = (
                s0[:, None]
                + prony_c
                + rw[0] * (act * hold_true)[:, None] * e0
                + (drift + rw[1] * act * z_cur[:, 0])[:, None] * phi1[None, :]
            )
            true_rate0 = prony_r + rw[0] * (act * hold_true) / taus[0] + drift + rw[1] * act * z_cur[:, 0]
        elif regime_kind == "C2":
            phi2 = eval_times / (1.0 + 0.85 * eval_times)
            sl_f = np.sign(hold_true) * np.sqrt(np.maximum(np.abs(hold_true), 1e-12))
            clean = (
                s0[:, None]
                + prony_c
                + (drift + rw[0] * sl_f)[:, None] * phi2[None, :]
                + rw[1] * (gap0 - gap1)[:, None] * e0
            )
            true_rate0 = prony_r + drift + rw[0] * sl_f + rw[1] * (gap0 - gap1) / taus[0]
        else:
            raise ValueError(f"Unknown regime_kind: {regime_kind}")

        obs = clean + rng_h.normal(scale=0.015, size=clean.shape)
        if add_glitches and glitch_frac > 0.0:
            gm = rng_h.random(obs.shape) < glitch_frac
            obs = obs + gm * (0.42 * rng_h.standard_cauchy(size=obs.shape))
        rate_obs = true_rate0 + rng_h.normal(scale=0.018, size=N)
        split_d = {
            "lab_id": labs_a,
            "stratum": strata_a,
            "coupon_features_raw": X_r,
            "ref_coupons": R_r,
            "past_stress_cmd": hist_cmd,
            "stress_hold_cmd": hold_cmd,
            "strain_0": s0,
            "aux_chamber_telemetry": aux_tel,
        }
        return split_d, obs, rate_obs, clean, true_rate0, hist_true, hold_true

    tr_d, tr_obs, tr_robs, _, _, _, _ = _sim_from_history(tr_p, add_glitches=True)
    tr_d["strain_curves_obs"] = tr_obs
    tr_d["instant_rate_obs"] = tr_robs
    te_d, _, _, te_clean, te_rate0, te_hist_t, te_hold_t = _sim_from_history(te_p, add_glitches=False)

    dataset = {
        "past_dt": past_dt,
        "eval_times": eval_times,
        "crit_stress": cs,
        "yield_stress": cs,
        "shear_banding_index": sb_idx,
        "calibration_load_cells": {"lab_id": cal_labs, "cmd_stress": cal_cmd, "true_stress": cal_true},
        "train": tr_d,
        "test": te_d,
    }
    gt = {
        "taus": taus,
        "theta": theta,
        "vp_theta": rw,
        "stress_gains": s_gains,
        "stress_biases": s_biases,
        "te_clean": te_clean,
        "te_true_rate0": te_rate0,
        "te_hist_true": te_hist_t,
        "te_hold_true": te_hold_t,
    }
    return dataset, gt
'''

_M97_BETA_GEN_PY = (
    _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY
    + r'''
def _solve_box_simplex_exact_m97(S_tilde, r_y, lam_w, u_box, total_budget):
    from scipy.optimize import minimize

    M = S_tilde.shape[1]
    H = S_tilde.T @ S_tilde + float(lam_w) * np.eye(M)
    f = S_tilde.T @ r_y
    b_tot = float(total_budget)
    ub = float(u_box)
    w0 = np.full(M, b_tot / M, dtype=float)
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        w0,
        jac=lambda w: H @ w - f,
        bounds=[(0.0, ub)] * M,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - b_tot, "jac": lambda w: np.ones(M)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 400},
    )
    w = np.clip(res.x, 0.0, ub)
    lo, hi = float(np.min(w) - ub), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, ub))) > b_tot:
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, ub)
    for _ in range(8):
        free = (w > 1e-11) & (w < ub - 1e-11)
        f_idx = np.flatnonzero(free)
        if f_idx.size >= 2:
            rem = b_tot - float(np.sum(w[~free]))
            Hf = H[np.ix_(f_idx, f_idx)]
            rhs_f = f[f_idx] - H[np.ix_(f_idx, np.flatnonzero(~free))] @ w[~free]
            aug = np.block([[Hf, -np.ones((f_idx.size, 1))], [np.ones((1, f_idx.size)), np.zeros((1, 1))]])
            sol = np.linalg.solve(aug, np.concatenate([rhs_f, [rem]]))
            wf = sol[:-1]
            if np.all((wf >= -1e-12) & (wf <= ub + 1e-12)):
                w[f_idx] = np.clip(wf, 0.0, ub)
                break
    r_v = S_tilde @ w - r_y
    j_p = 0.5 * float(r_v @ r_v) + 0.5 * float(lam_w) * float(w @ w)
    g = H @ w - f
    best_nu, best_lb = float(g[0]), -1e30
    for nc in np.unique(g):
        lbc = j_p - float(g @ w) + float(nc) * b_tot + ub * float(np.sum(np.minimum(0.0, g - float(nc))))
        if lbc > best_lb:
            best_lb, best_nu = lbc, float(nc)
    cert = {
        "weights": w.tolist(),
        "nu_sum": best_nu,
        "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
        "lam_upper": np.maximum(0.0, best_nu - g).tolist(),
        "primal_objective": j_p,
        "dual_lower_bound": float(min(best_lb, j_p)),
    }
    return w, cert


def _gen_m97_beta_study(
    present_seed,
    *,
    history_seed=None,
    nx=18,
    ny=16,
    bank_gains=(1.0, 1.0, 1.0, 1.0),
    bank_offsets=(0.0, 0.0, 0.0, 0.0),
    aux_gain=1.0,
    cal_mode="full_48",
    p0_channels=(0, 1),
    p1_channels=(2, 3),
    robin_pair=(0.85, 0.75),
    regime_kind="C0",  # 'C0' | 'C1' | 'C2'
    obs_noise_scale=2.5e-4,
):
    """Generates Beta 2D reaction-transport study where `(R0, U_iso)` called with the SAME `present_seed` share
    BYTE-FOR-BYTE IDENTICAL present study dicts (`Lx, Ly, nx, ny, kappa, vel, robin, V_reaction_2d, recirculation_loop_mode,
    critical_vortex_circulation, terminal circulation trace[-1], source_modes, sensor_ij, row_bank_id, Z_nuisance,
    sensor_calibration_sweeps, aux_manifold_telemetry`!), differing ONLY in `past_vortex_circulation_trace[:-1]`!
    """
    rng_p = np.random.default_rng(int(present_seed))
    rng_h = np.random.default_rng(int(present_seed + 888 if history_seed is None else history_seed))
    b_gains = np.asarray(bank_gains, dtype=float)
    b_offs = np.asarray(bank_offsets, dtype=float)
    Lx, Ly = 1.5, 1.3
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)

    kappa_x = 0.095 + 0.025 * np.cos(math.pi * xs / Lx)
    vel_x = 0.24 * (1.0 - 0.25 * xs / Lx)
    rx_base = 0.30 + 0.16 * (xs / Lx) ** 2
    kappa_y = 0.085 + 0.020 * np.sin(math.pi * ys / Ly)
    vel_y = 0.20 * (1.0 + 0.20 * ys / Ly)
    ry_base = 0.26 + 0.14 * np.cos(0.5 * math.pi * ys / Ly)
    V_2d = rx_base[:, None] + ry_base[None, :]
    recirc_mode = 0.85 if regime_kind == "C2" else 0.0
    crit_vort = 0.50

    term_vort = round(float(rng_p.uniform(0.05, 0.14)), 6)
    vortex_trace = np.zeros(16, dtype=float)
    if regime_kind == "C1":
        vortex_trace[:-1] = np.round(rng_h.uniform(0.12, 0.34, size=15), 6)
        vortex_trace[4:9] = np.round(rng_h.uniform(0.68, 1.05, size=5), 6)
        vortex_trace[-1] = term_vort
    else:
        vortex_trace[:-1] = np.round(rng_h.uniform(0.06, 0.34, size=15), 6)
        vortex_trace[-1] = term_vort

    def _build_1d_op(n, h, kap, vel, rob, mode):
        A = np.zeros((n, n), dtype=float)
        for i in range(1, n - 1):
            kh_p = 0.5 * (kap[i] + kap[i + 1])
            kh_m = 0.5 * (kap[i] + kap[i - 1])
            A[i, i - 1] = -kh_m / (h * h) - max(vel[i], 0.0) / h
            A[i, i + 1] = -kh_p / (h * h) - max(-vel[i], 0.0) / h
            A[i, i] = (kh_p + kh_m) / (h * h) + abs(vel[i]) / h
        if mode == "C2":
            # Periodic wrap-around recirculation manifold (`0 <-> n-1`)
            for i in (0, n - 1):
                im = (i - 1) % n
                ip = (i + 1) % n
                kh_p = 0.5 * (kap[i] + kap[ip])
                kh_m = 0.5 * (kap[i] + kap[im])
                A[i, im] += -kh_m / (h * h) - max(vel[i], 0.0) / h
                A[i, ip] += -kh_p / (h * h) - max(-vel[i], 0.0) / h
                A[i, i] += (kh_p + kh_m) / (h * h) + abs(vel[i]) / h
        elif mode == "C1":
            # Fouled-membrane sealed zero-flux Neumann boundary (`rob == 0.0`!)
            A[0, 0] = kap[0] / (h * h); A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h); A[n - 1, n - 2] = -kap[-1] / (h * h)
        else:
            # `C0`: Open outflow Robin boundary (`+ rob / h` active!)
            A[0, 0] = kap[0] / (h * h) + rob / h; A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h) + rob / h; A[n - 1, n - 2] = -kap[-1] / (h * h)
        return A

    Ax = _build_1d_op(nx, dx, kappa_x, vel_x, float(robin_pair[0]), regime_kind)
    Ay = _build_1d_op(ny, dy, kappa_y, vel_y, float(robin_pair[1]), regime_kind)
    V_eff = V_2d.copy()
    if regime_kind == "C1":
        v_mem = float(np.sum(np.maximum(0.0, vortex_trace - crit_vort) ** 2))
        wall_mask = np.exp(-((xs[:, None] / (0.25 * Lx)) ** 2)) + np.exp(-(((Lx - xs[:, None]) / (0.25 * Lx)) ** 2))
        V_eff = V_eff + (0.95 + 0.55 * math.tanh(v_mem)) * wall_mask * (0.6 + 0.4 * np.sin(math.pi * ys[None, :] / Ly))
    elif regime_kind == "C2":
        V_eff = V_eff + 0.75 * recirc_mode * np.cos(2.0 * math.pi * xs[:, None] / Lx) * np.cos(2.0 * math.pi * ys[None, :] / Ly)

    L_full = np.kron(np.eye(ny), Ax) + np.kron(Ay, np.eye(nx)) + np.diag(V_eff.ravel(order="F"))

    M = 12
    K_mu = 4
    mu_shifts = np.array([0.00, 0.22, 0.52, 0.90], dtype=float)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    centers = [(0.22 + 0.28 * (m % 4), 0.24 + 0.34 * (m // 4)) for m in range(M)]
    modes = np.empty((M, nx, ny), dtype=float)
    for m, (cx, cy) in enumerate(centers):
        g = np.exp(-(((XX - cx * Lx) / 0.24) ** 2 + ((YY - cy * Ly) / 0.24) ** 2))
        modes[m] = g / np.linalg.norm(g)

    n_sens = 36
    si = rng_p.integers(1, nx - 1, size=n_sens)
    sj = rng_p.integers(1, ny - 1, size=n_sens)
    sensor_ij = np.column_stack([si, sj])
    sens_bank = np.arange(n_sens, dtype=int) % 4
    row_bank_id = np.tile(sens_bank, K_mu)
    aux_tel = np.round(float(aux_gain) * rng_p.normal(scale=1.0, size=(K_mu * n_sens, 3)), 6)

    if cal_mode == "full_48":
        cal_banks = np.repeat(np.arange(4, dtype=int), 12)
        cal_ref = np.round(rng_p.uniform(-1.5, 2.2, size=48), 6)
        cal_raw = b_gains[cal_banks] * cal_ref + b_offs[cal_banks] + rng_p.normal(scale=1e-7, size=48)
    elif cal_mode == "sparse_spot_ch0":
        ch0 = int(p0_channels[0])
        cal_banks = np.array([ch0, ch0], dtype=int)
        cal_ref = np.array([-1.10, 1.55], dtype=float)
        cal_raw = b_gains[cal_banks] * cal_ref + b_offs[cal_banks] + rng_p.normal(scale=1e-7, size=2)
    elif cal_mode == "permuted_p0_shift_plus_p1_emi":
        ba = np.repeat(np.asarray(p0_channels, dtype=int), 12)
        ra = rng_p.uniform(-1.5, 2.1, size=24)
        ya = b_gains[ba] * ra + b_offs[ba] + rng_p.normal(scale=1e-7, size=24)
        p1a, p1b = int(p1_channels[0]), int(p1_channels[1])
        bb = np.array([p1a, p1b, p1a, p1b, p1a, p1b], dtype=int)
        rb = np.array([-1.35, 1.45, 1.10, -1.15, -0.80, 1.25], dtype=float)
        emi = np.array([5.10, -4.90, 4.75, -4.80, 0.0, 0.0], dtype=float)
        yb = b_gains[bb] * rb + b_offs[bb] + emi
        cal_banks = np.concatenate([ba, bb])
        cal_ref = np.concatenate([ra, rb])
        cal_raw = np.concatenate([ya, yb])
    else:
        raise ValueError(f"Unknown cal_mode: {cal_mode}")

    U_all = np.empty((M, K_mu, nx, ny), dtype=float)
    S_blocks = []
    B_rhs = np.column_stack([modes[m].ravel(order="F") for m in range(M)])
    I_f = np.eye(nx * ny, dtype=float)
    for k_i, mu in enumerate(mu_shifts):
        sol_f = np.linalg.solve(L_full + float(mu) * I_f, B_rhs)
        for m in range(M):
            U_all[m, k_i] = sol_f[:, m].reshape((nx, ny), order="F")
        S_blocks.append(U_all[:, k_i, si, sj].T)
    S_stack = np.vstack(S_blocks)

    n_rows = K_mu * n_sens
    P_z = 5
    Z_nuis = rng_p.normal(size=(n_rows, P_z))
    lam_z = 0.65
    lam_w = 0.04
    u_box = 1.0
    b_tot = 2.60

    w_true = np.zeros(M, dtype=float)
    raw_w = np.array([0.88, 0.72, 0.58, 0.42], dtype=float)
    w_true[[1, 4, 7, 10]] = raw_w * (b_tot / float(np.sum(raw_w)))
    gamma_true = rng_p.normal(scale=0.65, size=P_z)
    y_true_cal = S_stack @ w_true + Z_nuis @ gamma_true + rng_h.normal(scale=float(obs_noise_scale), size=n_rows)
    y_obs_raw = b_gains[row_bank_id] * y_true_cal + b_offs[row_bank_id]

    Pz_mat = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(P_z), Z_nuis.T)
    r_y = y_true_cal - Z_nuis @ (Pz_mat @ y_true_cal)
    S_tilde = S_stack - Z_nuis @ (Pz_mat @ S_stack)

    w_opt, cert_opt = _solve_box_simplex_exact_m97(S_tilde, r_y, lam_w, u_box, b_tot)
    term_f = np.tensordot(w_opt, U_all[:, -1, :, :], axes=(0, 0))
    study = {
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny,
        "kappa_x": kappa_x, "vel_x": vel_x, "robin_x": float(robin_pair[0]), "reaction_x": rx_base,
        "kappa_y": kappa_y, "vel_y": vel_y, "robin_y": float(robin_pair[1]), "reaction_y": ry_base,
        "V_reaction_2d": V_2d,
        "recirculation_loop_mode": recirc_mode,
        "past_vortex_circulation_trace": vortex_trace,
        "critical_vortex_circulation": crit_vort,
        "mu_shifts": mu_shifts,
        "source_modes": modes,
        "sensor_ij": sensor_ij,
        "row_bank_id": row_bank_id,
        "sensor_calibration_sweeps": {"bank_id": cal_banks, "ref_signal": cal_ref, "raw_readout": cal_raw},
        "Z_nuisance": Z_nuis,
        "y_obs_raw": y_obs_raw,
        "aux_manifold_telemetry": aux_tel,
        "nuisance_ridge": lam_z,
        "source_ridge": lam_w,
        "u_box": u_box,
        "total_budget": b_tot,
    }
    gt = {"S_tilde": S_tilde, "r_y": r_y, "w_opt": w_opt, "cert_opt": cert_opt, "terminal_field_A": term_f}
    return study, gt
'''
)


# ============================================================================
# 2. Task Builders (`Unified Schema Across C0/C1/C2` + Three-Layer Private Verifier!)
# ============================================================================


def _build_m97_alpha_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
) -> HoldoutOODTask:
    artifact_name = f"{short_name}_solver.py"
    entry_name = f"solve_{short_name}"
    pkl_name = f"public_{short_name}.pkl"
    mod_stem = Path(artifact_name).stem
    schema_keys = '(("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("regime_mode_weights", (2,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,)))'

    priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau_rel = 0.0; max_mae = 0.0; max_rate_mae = 0.0; max_rw_err = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "hereditary_weights", "regime_mode_weights", "predicted_strain", "initial_rate_pred"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    hw = np.asarray(out["hereditary_weights"], dtype=float)
    rw = np.asarray(out["regime_mode_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ir = np.asarray(out["initial_rate_pred"], dtype=float)
    if et.shape != (2,) or hw.shape != (7,) or rw.shape != (2,) or ps.shape != gt["te_clean"].shape or ir.shape != gt["te_true_rate0"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(hw)) and np.all(np.isfinite(rw)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ir)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    if float(np.linalg.norm(hw[:2])) <= 1e-4 or float(np.min(hw[:2])) <= 1e-4:
        valid = False; rows.append({{"seed": sd, "error": "degenerate_prony_weights"}}); continue
    tau_rel = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    s_mae = float(np.mean(np.abs(ps - gt["te_clean"])))
    r_mae = float(np.mean(np.abs(ir - gt["te_true_rate0"])))
    th_err = float(np.max(np.abs(hw - gt["theta"])))
    rw_err = float(np.linalg.norm(rw - gt["vp_theta"]) / max(float(np.linalg.norm(gt["vp_theta"])), 1.0))
    if th_err > 0.14:
        valid = False; rows.append({{"seed": sd, "error": "theta_7coeff_gt_mismatch", "theta_err": round(th_err, 6), "strain_mae": round(s_mae, 6)}}); continue
    max_tau_rel = max(max_tau_rel, tau_rel); max_mae = max(max_mae, s_mae); max_rate_mae = max(max_rate_mae, r_mae); max_rw_err = max(max_rw_err, rw_err)
    rows.append({{"seed": sd, "tau_rel": round(tau_rel, 6), "strain_mae": round(s_mae, 6), "rate_mae": round(r_mae, 6), "rw_err": round(rw_err, 6), "theta_err": round(th_err, 6)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau_rel <= 0.10 and max_mae <= 0.070 and max_rate_mae <= 0.055 and max_rw_err <= 0.16 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau_rel, 6), "max_strain_mae": round(max_mae, 6), "max_rate_mae": round(max_rate_mae, 6), "max_rw_err": round(max_rw_err, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave=wave_label,
        scientific_domain="Multi-Channel Polymer Rheology & Non-Nested Phase Continual Stream",
        wormhole_family="fam_hereditary_memory_and_non_nested_rheology",
        declared_artifacts=(f"/holdout_task/{artifact_name}",),
        declared_callable_entrypoints=(entry_name,),
        instruction_text=instruction_text,
        public_setup_py=(
            "import json, math, os, pickle, sys\nimport numpy as np\n"
            + _M97_ALPHA_GEN_PY
            + f"\npub_ds, pub_gt = {pub_call_expr}\n"
            f'with open("/holdout_task/{pkl_name}", "wb") as fh:\n'
            '    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)\n'
            'open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys\nimport numpy as np\n'
            "_d = os.path.dirname(os.path.abspath(__file__))\n"
            f'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "{artifact_name}")\n'
            'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f'item = pickle.load(open(os.path.join(_d, "{pkl_name}"), "rb"))[0]\n'
            'ds = item["dataset"]\n'
            f"out = mod.{entry_name}(ds)\n"
            'n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])\n'
            "errs = []\n"
            'if not isinstance(out, dict): errs.append("not_dict")\n'
            "else:\n"
            f"    for k, shp in {schema_keys}:\n"
            '        if k not in out: errs.append(f"missing:{k}"); continue\n'
            "        a = np.asarray(out[k], dtype=float)\n"
            '        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
            'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
            "sys.exit(0 if len(errs) == 0 else 1)\n"
            '""")\n'
        ),
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            f"import {mod_stem} as cand_mod\n"
            + _M97_ALPHA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def _build_m97_beta_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
) -> HoldoutOODTask:
    artifact_name = f"{short_name}_solver.py"
    entry_name = f"solve_{short_name}"
    pkl_name = f"public_{short_name}.pkl"
    mod_stem = Path(artifact_name).stem

    val_check = (
        '    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):\n'
        '        if k not in out: errs.append(f"missing:{k}"); continue\n'
        '        a = np.asarray(out[k], dtype=float)\n'
        '        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
        '    if "kkt_certificate" not in out: errs.append("missing:kkt_certificate")\n'
        '    elif not errs:\n'
        '        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)\n'
        '        if not chk["valid"]: errs.append(f"kkt_invalid:{chk[\'reason\']}")\n'
        '        elif float(np.max(np.abs(np.asarray(out["weights"], dtype=float) - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5: errs.append("w_cert_mismatch")\n'
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
    w_rel = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    f_rel = float(np.linalg.norm(tf - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    if not chk["valid"] or float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False; rows.append({{"seed": sd, "error": f"kkt:{{chk['reason']}}", "w_rel_l2": round(w_rel, 8), "field_rel_l2": round(f_rel, 8)}}); continue
    max_w_rel = max(max_w_rel, w_rel); max_f_rel = max(max_f_rel, f_rel); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({{"seed": sd, "w_rel_l2": round(w_rel, 8), "field_rel_l2": round(f_rel, 8), "kkt_gap": round(float(chk["gap"]), 10)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 1.5e-3 and max_f_rel <= 1.5e-3 and max_gap <= 1e-4 and el <= 18.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
'''
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave=wave_label,
        scientific_domain="2D Multi-Boundary Reaction-Transport & Counterfactual Vortex Continual Stream",
        wormhole_family="fam_kronecker_boundary_regimes_and_kkt",
        declared_artifacts=(f"/holdout_task/{artifact_name}",),
        declared_callable_entrypoints=(entry_name,),
        instruction_text=instruction_text,
        public_setup_py=(
            "import json, math, os, pickle, sys\nimport numpy as np\n"
            + _M97_BETA_GEN_PY
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
            + _M97_BETA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def build_m97_world_tasks(spec: M97WorldSpec) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]]]:
    """Instantiate `(stage_S_task, 7_anonymous_stream_tasks, hidden_manifest)` for `M9.7`:
    `Stage S (C0) -> Step 1 (C1) -> Step 2 (R0) -> Step 3 (C2) -> Step 4 (R1) -> Step 5 (U_iso) -> Step 6 (G_perm) -> Step 7 (M_couple)`.
    Note: `R0` (Step 2) and `U_iso` (Step 5) use the EXACT SAME `present_seed` (`sd_r0_present`), guaranteeing `0.0` present-input difference!
    """
    rng = np.random.default_rng(spec.world_seed + 53)
    wid = spec.world_id
    p0_ch = (spec.channel_perm[0], spec.channel_perm[1])
    p1_ch = (spec.channel_perm[2], spec.channel_perm[3])

    v2_g, v2_b = list(spec.v2_gains), list(spec.v2_offsets)
    v3_g = list(v2_g)
    v3_b = list(v2_b)
    v3_g[p0_ch[0]], v3_g[p0_ch[1]] = spec.v3_p0_gains[0], spec.v3_p0_gains[1]
    v3_b[p0_ch[0]], v3_b[p0_ch[1]] = spec.v3_p0_offsets[0], spec.v3_p0_offsets[1]
    v2_g_t, v2_b_t = tuple(v2_g), tuple(v2_b)
    v3_g_t, v3_b_t = tuple(v3_g), tuple(v3_b)

    # Shared `present_seed` for `(R0, U_iso)` counterfactual pair (`N06`)!
    sd_r0_pub = spec.world_seed + 2501
    sd_r0_p1 = spec.world_seed + 2561
    sd_r0_p2 = spec.world_seed + 2562

    if spec.family == "alpha":
        cal_hdr = (
            "1. LOAD-CELL & PANEL ALIGNMENT (`4` channels `0..3`; note `aux_chamber_telemetry` is uncoupled environmental telemetry):\n"
            "   - `dataset['calibration_load_cells']` provides in-batch `{lab_id, cmd_stress, true_stress}` obeying `true_stress = gain[c]*cmd_stress + bias[c]`.\n"
            "     Fit `(gain[c], bias[c])` directly from in-batch rows whenever a channel has `>= 4` clean points; consult `/tmp/hegel_continual_archive/archive_calibration_history.json` only when a channel has `< 2` in-batch points or exhibits transient `> 3.0` EMI spikes.\n"
            "   - Convert `past_stress_cmd` and `stress_hold_cmd` into physical `past_stress` and `stress_hold_0`.\n"
            "   - Standardize `coupon_features_raw` `(N, 4)` per lab via median column mean/std (`ddof=0`) of `ref_coupons` (`train + test`),\n"
            "     imputing `NaN`s with `(lab, stratum)` cell-balanced column means -> `X_aligned` `(N, 4)`.\n"
            "   - Propagate 2-branch memory `z_m` (`z_m(0) = 0`): `z_m <- exp(-past_dt[s]/tau_m)*z_m + (1 - exp(-past_dt[s]/tau_m))*past_stress[:, s]`,\n"
            "     `gap_m = stress_hold_0 - z_m`, `drift = hw[2] + X_aligned @ hw[3:7]`, `Prony(t) = hw[0]*gap_0*(1-exp(-t/tau_0)) + hw[1]*gap_1*(1-exp(-t/tau_1))`.\n"
        )
        ret_hdr = (
            "3. RETURN unified dict `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'regime_mode_weights': (2,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
        )
        src_sn = f"{wid}_s00"
        t1_s = round(float(rng.uniform(0.50, 0.63)), 3)
        t2_s = round(float(rng.uniform(3.60, 4.20)), 3)
        th_s = [0.69, 0.55, 0.14, 0.17, -0.14, 0.12, -0.09]
        src_task = _build_m97_alpha_task(
            task_id=f"{wid}_s00__rheology_campaign_baseline",
            short_name=src_sn,
            wave_label=f"{wid}_source",
            instruction_text=(
                f"Implement `/holdout_task/{src_sn}_solver.py` exposing `def solve_{src_sn}(dataset: dict) -> dict:`\n\n"
                + cal_hdr
                + "2. AMORPHOUS MELT LINEAR FLOW REGIME (`shear_banding_index <= 0.50` and past `|past_stress| <= dataset['crit_stress']`, ~7% Cauchy glitches on `strain_curves_obs`):\n"
                "   - Linear terminal flow envelope `phi_0(t_k) = t_k`: `eps(t_k) = strain_0 + Prony(t_k) + drift * t_k`,\n"
                "     `initial_rate_pred = hw[0]*gap_0/tau_0 + hw[1]*gap_1/tau_1 + drift`, and `regime_mode_weights = np.zeros(2)`.\n"
                + ret_hdr
                + f"- Public dataset: `/holdout_task/public_{src_sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
            ),
            pub_call_expr=f"_gen_m97_alpha_cohort({spec.world_seed + 11}, ({t1_s}, {t2_s}), {th_s!r}, p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind='C0')",
            priv_cases_py=(
                f"    ({spec.world_seed + 511}, *_gen_m97_alpha_cohort({spec.world_seed + 511}, ({round(t1_s - 0.03, 3)}, {round(t2_s - 0.15, 3)}), [0.71, 0.54, 0.15, 0.16, -0.14, 0.12, -0.08], p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind='C0')),\n"
                f"    ({spec.world_seed + 512}, *_gen_m97_alpha_cohort({spec.world_seed + 512}, ({round(t1_s + 0.04, 3)}, {round(t2_s + 0.20, 3)}), [0.66, 0.57, 0.13, -0.15, 0.13, -0.11, 0.09], p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind='C0')),"
            ),
        )

        c1_body = (
            "2. STRAIN-CRYSTALLIZED NETWORK PLATEAU REGIME (`shear_banding_index <= 0.50` and past work-damage `D_cryst = sum_s max(0, |past_stress[:, s]| - crit_stress)**2 * past_dt[s] > 0`, `act = tanh(D_cryst)`):\n"
            "   - Crystallization arrests linear terminal flow into logarithmic plateau `phi_1(t_k) = log(1 + t_k)` (`phi_1'(0) = 1`):\n"
            "     `eps(t_k) = strain_0 + Prony(t_k) + rw[0]*(act * stress_hold_0)*(1 - exp(-t_k/tau_0)) + (drift + rw[1]*act*z_0) * log(1 + t_k)`,\n"
            "     `initial_rate_pred = hw[0]*gap_0/tau_0 + hw[1]*gap_1/tau_1 + rw[0]*(act * stress_hold_0)/tau_0 + drift + rw[1]*act*z_0`.\n"
        )
        c2_body = (
            "2. SHEAR-BANDING JAMMED SLIP REGIME (`dataset['shear_banding_index'] > 0.50`):\n"
            "   - High-frequency shear banding saturates terminal flow along `phi_2(t_k) = t_k / (1.0 + 0.85 * t_k)` (`phi_2'(0) = 1`) with slip `s_h = sign(stress_hold_0) * sqrt(|stress_hold_0|)`:\n"
            "     `eps(t_k) = strain_0 + Prony(t_k) + (drift + rw[0]*s_h) * phi_2(t_k) + rw[1]*(gap_0 - gap_1)*(1 - exp(-t_k/tau_0))`,\n"
            "     `initial_rate_pred = hw[0]*gap_0/tau_0 + hw[1]*gap_1/tau_1 + drift + rw[0]*s_h + rw[1]*(gap_0 - gap_1)/tau_0`.\n"
        )
        c0_body = (
            "2. AMORPHOUS MELT LINEAR FLOW REGIME (`shear_banding_index <= 0.50` and `D_cryst == 0`):\n"
            "   - Linear terminal flow `phi_0(t_k) = t_k`: `eps(t_k) = strain_0 + Prony(t_k) + drift * t_k`,\n"
            "     `initial_rate_pred = hw[0]*gap_0/tau_0 + hw[1]*gap_1/tau_1 + drift`, and `regime_mode_weights = np.zeros(2)`.\n"
        )

        stream_tasks: list[HoldoutOODTask] = []
        manifest: list[dict[str, Any]] = []
        for idx, ev in enumerate(spec.schedule_events, start=1):
            sn = f"{wid}_batch_{idx:02d}"
            tid = f"{sn}__rheology_campaign_batch"
            t1_e = 0.54 if ev in {"R0", "U_iso"} else round(float(rng.uniform(0.49, 0.64)), 3)
            t2_e = 3.90 if ev in {"R0", "U_iso"} else round(float(rng.uniform(3.55, 4.25)), 3)
            th_e = [0.68, 0.55, 0.14, 0.16, -0.13, 0.12, -0.09]
            th_p1 = [0.71, 0.54, 0.15, 0.16, -0.14, 0.12, -0.08]
            th_p2 = [0.65, 0.57, 0.13, -0.15, 0.13, -0.11, 0.09]
            sd_pub = sd_r0_pub if ev in {"R0", "U_iso"} else (spec.world_seed + 100 * idx + 21)
            sd_p1 = sd_r0_p1 if ev in {"R0", "U_iso"} else (spec.world_seed + 100 * idx + 71)
            sd_p2 = sd_r0_p2 if ev in {"R0", "U_iso"} else (spec.world_seed + 100 * idx + 72)

            if ev == "C1":
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_40", "C1", (0.28, 0.24), 1.0, 1000, c1_body, "first_non_nested_conflict"
            elif ev == "R0":
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_40", "C0", (0.0, 0.0), 1.0, 2000, c0_body, "base_regime_restoration"
            elif ev == "C2":
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_40", "C2", (0.31, -0.22), 1.0, 3000, c2_body, "second_independent_conflict"
            elif ev == "R1":
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "sparse_spot_ch0", "C1", (0.26, 0.22), 1.0, 4000, c1_body, "post_c2_first_branch_restoration"
            elif ev == "U_iso":
                # EXACT SAME `sd_pub / sd_p1 / sd_p2`, `g_u`, `b_u`, `cal_m`, `aux_g` as `R0`, ONLY `history_seed` & `C1` history differ!
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_40", "C1", (0.27, 0.23), 1.0, 9500, c1_body, "strict_present_matched_counterfactual_memory"
            elif ev == "G_perm":
                # `p0` shifts to `v3`, `p1` gets EMI spikes, AND wired nuisance `aux_gain` jumps to `3.60` on `C2`!
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v3_g_t, v3_b_t, "permuted_p0_shift_plus_p1_emi", "C2", (0.29, -0.21), 3.60, 6000, c2_body, "permuted_parent_shift_and_wired_nuisance"
            elif ev == "M_couple":
                g_u, b_u, cal_m, rk, rw_v, aux_g, h_off, body, cat = v3_g_t, v3_b_t, "sparse_spot_ch0", "C1", (0.27, 0.25), 3.60, 7000, c1_body, "multiparent_v3_plus_counterfactual_c1"
            else:
                raise ValueError(f"Unknown event {ev}")

            kw = (
                f"stress_gains={g_u}, stress_biases={b_u}, aux_gain={aux_g}, cal_mode={cal_m!r}, "
                f"p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind={rk!r}, regime_weights={rw_v}"
            )
            tk = _build_m97_alpha_task(
                task_id=tid,
                short_name=sn,
                wave_label=f"{wid}_step_{idx:02d}",
                instruction_text=(
                    f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(dataset: dict) -> dict:`\n\n"
                    + cal_hdr
                    + body
                    + ret_hdr
                    + f"- Public dataset: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                ),
                pub_call_expr=f"_gen_m97_alpha_cohort({sd_pub}, ({t1_e}, {t2_e}), {th_e!r}, history_seed={sd_pub + h_off}, {kw})",
                priv_cases_py=(
                    f"    ({sd_p1}, *_gen_m97_alpha_cohort({sd_p1}, (0.51, 3.65), {th_p1!r}, history_seed={sd_p1 + h_off}, {kw})),\n"
                    f"    ({sd_p2}, *_gen_m97_alpha_cohort({sd_p2}, (0.66, 4.35), {th_p2!r}, history_seed={sd_p2 + h_off}, {kw})),"
                ),
            )
            stream_tasks.append(tk)
            manifest.append({"step": idx, "short_name": sn, "task_id": tid, "causal_event": ev, "true_regime": rk, "event_category": cat})
        return src_task, tuple(stream_tasks), manifest

    # Family Beta (`C0` Open Robin vs `C1` Post-Vortex Sealed Neumann + Wall Sorption vs `C2` Periodic Recirculation)
    cal_b_hdr = (
        "1. SENSOR-BANK CALIBRATION (`4` banks `0..3`; note `aux_manifold_telemetry` is uncoupled auxiliary telemetry):\n"
        "   - `study['sensor_calibration_sweeps']` provides `{bank_id, ref_signal, raw_readout}` obeying `raw_readout = gain[b]*ref_signal + offset[b]`.\n"
        "     Fit `(gain[b], offset[b])` directly from in-batch rows whenever a bank has `>= 4` clean points; consult `/tmp/hegel_continual_archive/archive_calibration_history.json` only when a bank has `< 2` in-batch points or exhibits transient `> 3.0` EMI spikes,\n"
        "     inverting `y_cal = (y_obs_raw - offset[row_bank_id]) / gain[row_bank_id]`.\n"
        "   - Project out `Z_nuisance` (`lam_z = study['nuisance_ridge']`): `r_y = y_cal - Z_nuis @ solve(Z_nuis.T @ Z_nuis + lam_z*I, Z_nuis.T @ y_cal)`\n"
        "     (and identically on stacked forward sensor matrix `S_stack` -> `S_tilde`).\n"
    )
    kkt_b_hdr = (
        "3. BOX-SIMPLEX KKT SOURCE INVERSION: Minimize `0.5*||S_tilde @ w - r_y||_2^2 + 0.5*source_ridge*||w||_2^2` s.t. `0 <= w <= u_box`, `sum(w) == total_budget`.\n"
        "   Return `{'weights': (12,), 'terminal_field': (nx, ny) at mu_shifts[-1], 'kkt_certificate': {'weights', 'nu_sum', 'lam_lower', 'lam_upper', 'primal_objective', 'dual_lower_bound'}}`.\n"
    )
    src_sn = f"{wid}_s00"
    src_task = _build_m97_beta_task(
        task_id=f"{wid}_s00__reactor_tomography_baseline",
        short_name=src_sn,
        wave_label=f"{wid}_source",
        instruction_text=(
            f"Implement `/holdout_task/{src_sn}_solver.py` exposing `def solve_{src_sn}(study: dict) -> dict:`\n\n"
            + cal_b_hdr
            + "2. OPEN OUTFLOW ROBIN REACTOR (`recirculation_loop_mode == 0` and `max(past_vortex_circulation_trace) <= critical_vortex_circulation`):\n"
            "   - Interior 1D operators `Ax`, `Ay` use centered face-averaged diffusion `-(kh_m*u_{i-1} - (kh_p+kh_m)*u_i + kh_p*u_{i+1})/h^2`\n"
            "     and upwind advection (`-max(v_i,0)*u_{i-1}/h - max(-v_i,0)*u_{i+1}/h + |v_i|*u_i/h`), with OPEN ROBIN boundaries:\n"
            "     `A[0,0] = kap[0]/h^2 + rob/h`, `A[0,1] = -kap[0]/h^2`; `A[n-1,n-1] = kap[-1]/h^2 + rob/h`, `A[n-1,n-2] = -kap[-1]/h^2`.\n"
            "   - Assemble `L_full = kron(I_ny, Ax) + kron(Ay, I_nx) + diag(V_reaction_2d.ravel(order='F'))` and solve `(L_full + mu*I) u = mode`.\n"
            + kkt_b_hdr
            + f"- Public study: `/holdout_task/public_{src_sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m97_beta_study({spec.world_seed + 11}, p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind='C0')",
        priv_cases_py=(
            f"    ({spec.world_seed + 511}, *_gen_m97_beta_study({spec.world_seed + 511}, p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind='C0')),\n"
            f"    ({spec.world_seed + 512}, *_gen_m97_beta_study({spec.world_seed + 512}, p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind='C0')),"
        ),
    )

    b_c0_body = (
        "2. OPEN OUTFLOW ROBIN REACTOR (`recirculation_loop_mode == 0` and `max(past_vortex_circulation_trace) <= critical_vortex_circulation`):\n"
        "   - Open Robin boundary loss active (`A[0,0] = kap[0]/h^2 + rob/h`, `A[n-1,n-1] = kap[-1]/h^2 + rob/h`) with `V_eff = V_reaction_2d`.\n"
    )
    b_c1_body = (
        "2. FOULED-MEMBRANE SEALED NEUMANN + WALL SORPTION REACTOR (`recirculation_loop_mode == 0` and `v_mem = sum(max(0, past_vortex_circulation_trace - critical_vortex_circulation)**2) > 0`):\n"
        "   - Past vortex excursion fouls outflow ports into SEALED ZERO-FLUX NEUMANN boundaries (`A[0,0] = kap[0]/h^2`, `A[0,1] = -kap[0]/h^2`,\n"
        "     `A[n-1,n-1] = kap[-1]/h^2`, `A[n-1,n-2] = -kap[-1]/h^2` — Robin loss `rob/h` vanishes!) AND adds wall sorption sink:\n"
        "     `wall_mask = exp(-(xs[:,None]/(0.25*Lx))**2) + exp(-((Lx-xs[:,None])/(0.25*Lx))**2)`,\n"
        "     `V_eff = V_reaction_2d + (0.95 + 0.55*tanh(v_mem)) * wall_mask * (0.6 + 0.4*sin(pi*ys[None,:]/Ly))`.\n"
    )
    b_c2_body = (
        "2. PERIODIC RECIRCULATION LOOP REACTOR (`study['recirculation_loop_mode'] > 0.50`):\n"
        "   - Closed recirculation manifold wraps boundary indices `i in {0, n-1}` PERIODICALLY (`im = (i - 1) % n`, `ip = (i + 1) % n`:\n"
        "     `A[i, im] += -kh_m/h^2 - max(vel[i],0)/h`, `A[i, ip] += -kh_p/h^2 - max(-vel[i],0)/h`, `A[i, i] += (kh_p+kh_m)/h^2 + |vel[i]|/h`)\n"
        "     with `V_eff = V_reaction_2d + 0.75 * recirculation_loop_mode * cos(2*pi*xs[:,None]/Lx) * cos(2*pi*ys[None,:]/Ly)`.\n"
    )

    stream_tasks = []
    manifest = []
    for idx, ev in enumerate(spec.schedule_events, start=1):
        sn = f"{wid}_batch_{idx:02d}"
        tid = f"{sn}__reactor_survey_batch"
        sd_pub = sd_r0_pub if ev in {"R0", "U_iso"} else (spec.world_seed + 100 * idx + 21)
        sd_p1 = sd_r0_p1 if ev in {"R0", "U_iso"} else (spec.world_seed + 100 * idx + 71)
        sd_p2 = sd_r0_p2 if ev in {"R0", "U_iso"} else (spec.world_seed + 100 * idx + 72)

        if ev == "C1":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_48", "C1", 1.0, 1000, b_c1_body, "first_non_nested_conflict"
        elif ev == "R0":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_48", "C0", 1.0, 2000, b_c0_body, "base_regime_restoration"
        elif ev == "C2":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_48", "C2", 1.0, 3000, b_c2_body, "second_independent_conflict"
        elif ev == "R1":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "sparse_spot_ch0", "C1", 1.0, 4000, b_c1_body, "post_c2_first_branch_restoration"
        elif ev == "U_iso":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v2_g_t, v2_b_t, "full_48", "C1", 1.0, 9500, b_c1_body, "strict_present_matched_counterfactual_memory"
        elif ev == "G_perm":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v3_g_t, v3_b_t, "permuted_p0_shift_plus_p1_emi", "C2", 3.60, 6000, b_c2_body, "permuted_parent_shift_and_wired_nuisance"
        elif ev == "M_couple":
            g_u, b_u, cal_m, rk, aux_g, h_off, body, cat = v3_g_t, v3_b_t, "sparse_spot_ch0", "C1", 3.60, 7000, b_c1_body, "multiparent_v3_plus_counterfactual_c1"
        else:
            raise ValueError(f"Unknown event {ev}")

        kw = f"bank_gains={g_u}, bank_offsets={b_u}, aux_gain={aux_g}, cal_mode={cal_m!r}, p0_channels={p0_ch}, p1_channels={p1_ch}, regime_kind={rk!r}"
        tk = _build_m97_beta_task(
            task_id=tid,
            short_name=sn,
            wave_label=f"{wid}_step_{idx:02d}",
            instruction_text=(
                f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(study: dict) -> dict:`\n\n"
                + cal_b_hdr
                + body
                + kkt_b_hdr
                + f"- Public study: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
            ),
            pub_call_expr=f"_gen_m97_beta_study({sd_pub}, history_seed={sd_pub + h_off}, {kw})",
            priv_cases_py=(
                f"    ({sd_p1}, *_gen_m97_beta_study({sd_p1}, history_seed={sd_p1 + h_off}, {kw})),\n"
                f"    ({sd_p2}, *_gen_m97_beta_study({sd_p2}, history_seed={sd_p2 + h_off}, {kw})),"
            ),
        )
        stream_tasks.append(tk)
        manifest.append({"step": idx, "short_name": sn, "task_id": tid, "causal_event": ev, "true_regime": rk, "event_category": cat})
    return src_task, tuple(stream_tasks), manifest


# ============================================================================
# 3. Neutral Primitive Observable Reducers (`obs_01..obs_08`) & Anonymized Permuted Parent Graph
# ============================================================================


def compute_m97_primitive_observables(
    family: str,
    public_item: dict[str, Any],
    param_payload: dict[str, Any],
) -> dict[str, float | None]:
    """Compute neutral primitive reducer statistics (`obs_01..obs_08`, `N05` — `0` pre-cooked composite OR/sum formulas!)."""
    if family == "alpha":
        ds = public_item["dataset"]
        gains = np.asarray(param_payload["stress_gains"], dtype=float)
        biases = np.asarray(param_payload["stress_biases"], dtype=float)
        p_dt = np.asarray(ds["past_dt"], dtype=float)
        cs = float(ds["crit_stress"])
        sb = float(ds.get("shear_banding_index", 0.0))

        te_l = np.asarray(ds["test"]["lab_id"], dtype=int)
        tr_l = np.asarray(ds["train"]["lab_id"], dtype=int)
        ps_te = np.asarray(ds["test"]["past_stress_cmd"], dtype=float) * gains[te_l, None] + biases[te_l, None]
        ps_tr = np.asarray(ds["train"]["past_stress_cmd"], dtype=float) * gains[tr_l, None] + biases[tr_l, None]
        h0_te = np.asarray(ds["test"]["stress_hold_cmd"], dtype=float) * gains[te_l] + biases[te_l]
        h0_tr = np.asarray(ds["train"]["stress_hold_cmd"], dtype=float) * gains[tr_l] + biases[tr_l]

        d_te = np.sum((np.maximum(0.0, np.abs(ps_te) - cs) ** 2) * p_dt[None, :], axis=1)
        d_tr = np.sum((np.maximum(0.0, np.abs(ps_tr) - cs) ** 2) * p_dt[None, :], axis=1)
        obs_01 = float(max(np.max(d_te), np.max(d_tr)))
        obs_02 = float(max(np.max(np.abs(h0_te)) - cs, np.max(np.abs(h0_tr)) - cs))
        obs_03 = sb
        obs_04 = float(np.max(np.abs(ps_te[:, -1])))
        obs_05 = float(np.nanstd(ds["test"]["coupon_features_raw"]))
        obs_06 = float(np.sqrt(np.mean(np.asarray(ds["test"]["aux_chamber_telemetry"], dtype=float) ** 2)))
        obs_07 = float(np.linalg.norm(h0_te) / math.sqrt(len(h0_te)))
        obs_08 = cs
    else:
        st = public_item["study"]
        vt = np.asarray(st["past_vortex_circulation_trace"], dtype=float)
        cv = float(st["critical_vortex_circulation"])
        rc = float(st.get("recirculation_loop_mode", 0.0))
        v2d = np.asarray(st["V_reaction_2d"], dtype=float)
        r_add = v2d - v2d.mean(axis=1, keepdims=True) - v2d.mean(axis=0, keepdims=True) + float(v2d.mean())

        obs_01 = float(np.sum(np.maximum(0.0, vt - cv) ** 2))
        obs_02 = float(vt[-1] - cv)
        obs_03 = rc
        obs_04 = float(vt[-1])
        obs_05 = float(np.linalg.norm(r_add) / max(float(np.linalg.norm(v2d)), 1e-12))
        obs_06 = float(np.sqrt(np.mean(np.asarray(st["aux_manifold_telemetry"], dtype=float) ** 2)))
        obs_07 = float(np.linalg.norm(v2d))
        obs_08 = cv

    return {
        "obs_01_past_series_exceedance_integral": round(obs_01, 8),
        "obs_02_present_snapshot_exceedance_max": round(obs_02, 8),
        "obs_03_boundary_or_banding_control_scalar": round(obs_03, 8),
        "obs_04_terminal_step_level": round(obs_04, 8),
        "obs_05_spatial_or_panel_anisotropy_ratio": round(obs_05, 8),
        "obs_06_aux_telemetry_rms": round(obs_06, 8),
        "obs_07_input_frobenius_scale": round(obs_07, 8),
        "obs_08_declared_threshold_metadata": round(obs_08, 8),
    }


def initialize_m97_permuted_parent_graph(
    *,
    spec: M97WorldSpec,
    forest: Any,
) -> tuple[PersistentMechanismGraph, M8VariationalPlanner, M94BindingRegistry, dict[str, Any]]:
    """Initialize `PersistentMechanismGraph` with anonymized permuted parents:
    - `cal_node_p0`: owns `spec.channel_perm[:2]`
    - `cal_node_p1`: owns `spec.channel_perm[2:]`
    - `cal_node_p2`: genuinely wired nuisance parent scaling `aux_chamber_telemetry` / `aux_manifold_telemetry`!
    - `cal_node_p3`: independent auxiliary monitor root (`0.0` drift sentinel)
    """
    graph = PersistentMechanismGraph(forest=forest)
    planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)

    p0_ch = list(spec.channel_perm[:2])
    p1_ch = list(spec.channel_perm[2:])
    for pid, chs in (("cal_node_p0", p0_ch), ("cal_node_p1", p1_ch)):
        w_init = {f"g_ch{chs[0]}": 1.0, f"b_ch{chs[0]}": 0.0, f"g_ch{chs[1]}": 1.0, f"b_ch{chs[1]}": 0.0}
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id=pid,
                parent_id=None,
                role="shared_upstream_premise",
                feature_names=tuple(w_init.keys()),
                version=1,
                shared_weights=w_init,
                provenance_note=f"Anonymized calibration parent `{pid}` mapped to channels `{chs}`.",
            ),
            tree_id=f"tree_{spec.world_id}",
        )

    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="cal_node_p2",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("aux_gain", "aux_offset"),
            version=1,
            shared_weights={"aux_gain": 1.0, "aux_offset": 0.0},
            provenance_note="Genuinely wired auxiliary telemetry parent `cal_node_p2`.",
        ),
        tree_id=f"tree_{spec.world_id}_aux",
    )

    rng_s = np.random.default_rng(spec.world_seed + 9)
    xs_tr, zs_tr = rng_s.uniform(-1.4, 1.4, size=28), rng_s.uniform(-1.0, 1.0, size=28)
    ys_tr = 1.55 * xs_tr - 0.20 * zs_tr + rng_s.normal(scale=0.01, size=28)
    xs_va, zs_va = rng_s.uniform(-1.4, 1.4, size=16), rng_s.uniform(-1.0, 1.0, size=16)
    ys_va = 1.55 * xs_va - 0.20 * zs_va + rng_s.normal(scale=0.01, size=16)
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="cal_node_p3",
            parent_id=None,
            role="shared_upstream_premise",
            feature_names=("x_mon", "z_mon"),
            version=1,
            shared_weights={"x_mon": 1.55, "z_mon": -0.20},
            provenance_note="Independent auxiliary monitor root `cal_node_p3`.",
        ),
        tree_id="tree_m97_sentinel",
    )
    planner.register_task(
        task_id="sentinel_p3_monitor",
        shared_upstream_id="cal_node_p3",
        feature_names=("x_mon", "z_mon"),
        train_data={"x": {"x_mon": xs_tr, "z_mon": zs_tr}, "y": ys_tr},
        val_data={"x": {"x_mon": xs_va, "z_mon": zs_va}, "y": ys_va},
        initial_local_shrinkage=50.0,
    )

    reg = M94BindingRegistry(graph)
    for pid in ("cal_node_p0", "cal_node_p1", "cal_node_p2", "cal_node_p3"):
        reg.parameter_epochs[pid] = {"parameter_mechanism_id": pid, "version": 1, "payload": deepcopy(graph.mechanisms[pid].shared_weights)}
    reg.sync_to_graph()

    parent_wiring_meta = {
        "cal_node_p0": {"channels": p0_ch, "input_target": "physical_calibration_channels"},
        "cal_node_p1": {"channels": p1_ch, "input_target": "physical_calibration_channels"},
        "cal_node_p2": {"channels": [], "input_target": ("aux_chamber_telemetry" if spec.family == "alpha" else "aux_manifold_telemetry")},
        "cal_node_p3": {"channels": [], "input_target": "unrelated_sentinel_task"},
    }
    return graph, planner, reg, parent_wiring_meta


def assemble_m97_parameter_payload(
    *,
    spec: M97WorldSpec,
    graph: PersistentMechanismGraph,
    parent_ids: list[str],
) -> dict[str, Any]:
    gains = [1.0, 1.0, 1.0, 1.0]
    offs = [0.0, 0.0, 0.0, 0.0]
    p0_ch = spec.channel_perm[:2]
    p1_ch = spec.channel_perm[2:]
    if "cal_node_p0" in parent_ids and "cal_node_p0" in graph.mechanisms:
        w0 = graph.mechanisms["cal_node_p0"].shared_weights
        for ch in p0_ch:
            gains[ch] = float(w0[f"g_ch{ch}"])
            offs[ch] = float(w0[f"b_ch{ch}"])
    if "cal_node_p1" in parent_ids and "cal_node_p1" in graph.mechanisms:
        w1 = graph.mechanisms["cal_node_p1"].shared_weights
        for ch in p1_ch:
            gains[ch] = float(w1[f"g_ch{ch}"])
            offs[ch] = float(w1[f"b_ch{ch}"])
    if spec.family == "alpha":
        return {"stress_gains": gains, "stress_biases": offs}
    return {"bank_gains": gains, "bank_offsets": offs}


def build_m97_intervention_functions(spec: M97WorldSpec) -> dict[str, Any]:
    """Explicit executable intervention metadata for all 4 candidate parents (`cal_node_p0..p3`, `N05/W03`):
    - `cal_node_p0` / `cal_node_p1`: perturb their assigned permuted physical channels!
    - `cal_node_p2`: genuinely perturbs `aux_chamber_telemetry` / `aux_manifold_telemetry` inside the candidate's input!
    - `cal_node_p3`: perturbs unrelated sentinel root (`0` effect on Alpha/Beta input).
    """
    kg = "stress_gains" if spec.family == "alpha" else "bank_gains"
    p0_ch = spec.channel_perm[:2]
    p1_ch = spec.channel_perm[2:]

    def _int_p0(pay: dict[str, Any], inp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        pay[kg][p0_ch[0]] = float(pay[kg][p0_ch[0]]) * 1.22
        pay[kg][p0_ch[1]] = float(pay[kg][p0_ch[1]]) * 0.81
        return pay, inp

    def _int_p1(pay: dict[str, Any], inp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        pay[kg][p1_ch[0]] = float(pay[kg][p1_ch[0]]) * 1.22
        pay[kg][p1_ch[1]] = float(pay[kg][p1_ch[1]]) * 0.81
        return pay, inp

    def _int_p2_wired_aux(pay: dict[str, Any], inp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if spec.family == "alpha":
            for sp_k in ("train", "test"):
                inp[sp_k]["aux_chamber_telemetry"] = np.asarray(inp[sp_k]["aux_chamber_telemetry"], dtype=float) * 4.5 + 2.2
        else:
            inp["aux_manifold_telemetry"] = np.asarray(inp["aux_manifold_telemetry"], dtype=float) * 4.5 + 2.2
        return pay, inp

    def _int_p3_sentinel(pay: dict[str, Any], inp: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return pay, inp

    return {
        "cal_node_p0": _int_p0,
        "cal_node_p1": _int_p1,
        "cal_node_p2": _int_p2_wired_aux,
        "cal_node_p3": _int_p3_sentinel,
    }


def update_m97_permuted_calibration_graph(
    *,
    spec: M97WorldSpec,
    episode_id: str,
    public_item: dict[str, Any],
    graph: PersistentMechanismGraph,
    planner: M8VariationalPlanner,
    registry: M94BindingRegistry,
) -> dict[str, Any]:
    """Run holdout-validated `M8` `AncestorReweight` across `cal_node_p0`, `cal_node_p1`, and wired `cal_node_p2`."""
    sent_tm = graph.active_task_models["sentinel_p3_monitor"]
    sent_va = graph.task_datasets["sentinel_p3_monitor"]["val"]
    sent_before = np.asarray(evaluate_committed_task_model(sent_tm, graph, sent_va)["predictions"], dtype=float)

    if spec.family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        ch_arr = np.asarray(cal["lab_id"], dtype=int).ravel()
        x_arr = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        y_arr = np.asarray(cal["true_stress"], dtype=float).ravel()
        aux_rms = float(np.sqrt(np.mean(np.asarray(public_item["dataset"]["test"]["aux_chamber_telemetry"], dtype=float) ** 2)))
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        ch_arr = np.asarray(cal["bank_id"], dtype=int).ravel()
        x_arr = np.asarray(cal["ref_signal"], dtype=float).ravel()
        y_arr = np.asarray(cal["raw_readout"], dtype=float).ravel()
        aux_rms = float(np.sqrt(np.mean(np.asarray(public_item["study"]["aux_manifold_telemetry"], dtype=float) ** 2)))

    # Update `cal_node_p2` if `aux_telemetry` shifted (`G_perm`, `M_couple`) — proving selective parent vector isolation!
    p2_mech = graph.mechanisms["cal_node_p2"]
    if abs(aux_rms - float(p2_mech.shared_weights["aux_gain"])) > 0.8:
        p2_mech.version += 1
        p2_mech.shared_weights["aux_gain"] = round(aux_rms, 4)
        registry.update_shared_parameter_epoch(
            parameter_mechanism_id="cal_node_p2",
            new_version=p2_mech.version,
            parameter_payload=dict(p2_mech.shared_weights),
            affected_chain_id=spec.family,
            evidence_note=f"aux_telemetry_shift_{episode_id}",
        )

    parent_receipts: dict[str, Any] = {}
    updated_bindings: list[str] = []
    for pid, chs in (("cal_node_p0", spec.channel_perm[:2]), ("cal_node_p1", spec.channel_perm[2:])):
        m_w = (ch_arr == chs[0]) | (ch_arr == chs[1])
        n_w = int(np.sum(m_w))
        if n_w < 6:
            parent_receipts[pid] = {"selected_action": "SkippedSparseSpotCheck", "n_rows": n_w, "version": graph.mechanisms[pid].version}
            continue
        cw = ch_arr[m_w]
        xw = x_arr[m_w]
        yw = y_arr[m_w]
        m0 = (cw == chs[0]).astype(float)
        m1 = (cw == chs[1]).astype(float)
        f_names = (f"g_ch{chs[0]}", f"b_ch{chs[0]}", f"g_ch{chs[1]}", f"b_ch{chs[1]}")
        cols = {f_names[0]: xw * m0, f_names[1]: m0, f_names[2]: xw * m1, f_names[3]: m1}
        n_tr = max(4, (2 * n_w) // 3)
        tr_d = {"x": {k: v[:n_tr] for k, v in cols.items()}, "y": yw[:n_tr]}
        va_d = {"x": {k: v[n_tr:] for k, v in cols.items()}, "y": yw[n_tr:]}
        tid_w = f"cal_m97_{pid}_{episode_id}"
        for old_t in [t for t in list(graph.active_task_models.keys()) if t.startswith(f"cal_m97_{pid}_")]:
            graph.active_task_models.pop(old_t, None)
            graph.task_datasets.pop(old_t, None)
            graph.mechanisms[pid].dependent_task_ids.discard(old_t)
        for old_m in [m for m in list(graph.mechanisms.keys()) if f"cal_m97_{pid}_" in m and m != pid]:
            graph.mechanisms.pop(old_m, None)
        w_before = dict(graph.mechanisms[pid].shared_weights)
        planner.register_task(
            task_id=tid_w,
            shared_upstream_id=pid,
            feature_names=f_names,
            train_data=tr_d,
            val_data=va_d,
            initial_local_shrinkage=1e6,
        )
        rev = planner.evaluate_and_revise_task(tid_w, budget_remaining=1.0)
        act = str(rev["selected_candidate"]["action_type"])
        w_after = dict(graph.mechanisms[pid].shared_weights)
        drift = max(abs(float(w_after[k]) - float(w_before[k])) for k in f_names)
        if act == "AncestorReweight":
            ub = registry.update_shared_parameter_epoch(
                parameter_mechanism_id=pid,
                new_version=int(graph.mechanisms[pid].version),
                parameter_payload=w_after,
                affected_chain_id=spec.family,
                evidence_note=f"reweight_{pid}_{episode_id}",
            )
            for b_i in ub:
                if b_i not in updated_bindings:
                    updated_bindings.append(b_i)
        parent_receipts[pid] = {
            "selected_action": act,
            "n_rows": n_w,
            "val_log_predictive_gain": round(float(rev["selected_candidate"]["val_log_predictive_gain"]), 4),
            "version": int(graph.mechanisms[pid].version),
            "max_weight_drift": round(drift, 8),
        }

    sent_after = np.asarray(evaluate_committed_task_model(sent_tm, graph, sent_va)["predictions"], dtype=float)
    return {
        "episode_id": episode_id,
        "parent_receipts": parent_receipts,
        "cal_node_p2_version": int(graph.mechanisms["cal_node_p2"].version),
        "updated_binding_ids": updated_bindings,
        "sentinel_p3_drift": round(float(np.max(np.abs(sent_after - sent_before))), 10),
    }


def update_m97_flat_validated_channel_snapshot(
    *,
    family: str,
    public_item: dict[str, Any],
    prior_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Identical per-channel `2/3–1/3` holdout-validated OLS calibration estimator for `Flat_EqualInfo_Multiversion`!"""
    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        ch_a = np.asarray(cal["lab_id"], dtype=int).ravel()
        x_a = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        y_a = np.asarray(cal["true_stress"], dtype=float).ravel()
        gains = list(prior_snapshot["stress_gains"])
        offs = list(prior_snapshot["stress_biases"])
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        ch_a = np.asarray(cal["bank_id"], dtype=int).ravel()
        x_a = np.asarray(cal["ref_signal"], dtype=float).ravel()
        y_a = np.asarray(cal["raw_readout"], dtype=float).ravel()
        gains = list(prior_snapshot["bank_gains"])
        offs = list(prior_snapshot["bank_offsets"])

    for ch in range(4):
        m = ch_a == ch
        n_m = int(np.sum(m))
        if n_m >= 3:
            xm, ym = x_a[m], y_a[m]
            n_tr = max(2, (2 * n_m) // 3)
            gc, bc = np.linalg.lstsq(np.column_stack([xm[:n_tr], np.ones(n_tr)]), ym[:n_tr], rcond=None)[0]
            va_mse = float(np.mean((gc * xm[n_tr:] + bc - ym[n_tr:]) ** 2))
            pr_mse = float(np.mean((gains[ch] * xm[n_tr:] + offs[ch] - ym[n_tr:]) ** 2))
            if va_mse <= min(1e-2, pr_mse + 1e-5):
                gains[ch], offs[ch] = round(float(gc), 8), round(float(bc), 8)
    if family == "alpha":
        return {"stress_gains": gains, "stress_biases": offs}
    return {"bank_gains": gains, "bank_offsets": offs}


# ============================================================================
# 4. Multi-Branch Contrastive Guard & Containerized Parent Sensitivity Induction (`N05 + N07`)
# ============================================================================

_M97_CLAUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["feature_name", "op", "threshold"],
    "properties": {
        "feature_name": {"type": "string", "enum": list(M97_PRIMITIVE_OBSERVABLE_NAMES)},
        "op": {"type": "string", "enum": ["<=", "<", ">=", ">"]},
        "threshold": {"type": "number"},
    },
}

M97_GUARD_EXPR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["all_of"],
    "properties": {
        "all_of": {"type": "array", "minItems": 1, "maxItems": 3, "items": _M97_CLAUSE_SCHEMA},
    },
}


def build_m97_multibranch_induction_schema(existing_branch_ids: list[str], candidate_parent_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "mechanistic_diagnosis",
            "violated_component_id",
            "preserved_component_ids",
            "dependent_upstream_parent_ids",
            "new_branch_applicability_guard",
            "existing_branch_guards",
        ],
        "properties": {
            "mechanistic_diagnosis": {"type": "string"},
            "violated_component_id": {"type": "string"},
            "preserved_component_ids": {"type": "array", "items": {"type": "string"}},
            "dependent_upstream_parent_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(candidate_parent_ids)},
            },
            "new_branch_applicability_guard": M97_GUARD_EXPR_SCHEMA,
            "existing_branch_guards": {
                "type": "object",
                "additionalProperties": False,
                "required": list(existing_branch_ids),
                "properties": {bid: M97_GUARD_EXPR_SCHEMA for bid in existing_branch_ids},
            },
        },
    }


def verify_m97_multibranch_witness_partition(
    *,
    branch_guards: dict[str, dict[str, Any]],
    positive_witness_features_by_branch: dict[str, list[dict[str, float | None]]],
) -> dict[str, Any]:
    """Verify that across all historical tested positive witnesses of every retained branch `b_k`:
    - `b_k`'s guard evaluates to `'TRUE'` on `100%` of `b_k`'s own positive witnesses;
    - Every other non-covering branch `b_j` (`j != k`) evaluates to `'FALSE'` on `b_k`'s positive witnesses!
    """
    reasons: list[str] = []
    for owner_bid, pos_list in positive_witness_features_by_branch.items():
        for w_i, feat_map in enumerate(pos_list):
            for eval_bid, g_spec in branch_guards.items():
                res = evaluate_m96_autonomous_guard(g_spec, feat_map, allow_open_feature_keys=True)
                if eval_bid == owner_bid and res["state"] != "TRUE":
                    reasons.append(f"{owner_bid}_pos_{w_i}:own_guard_{eval_bid}={res['state']}")
                elif eval_bid != owner_bid and res["state"] != "FALSE":
                    reasons.append(f"{owner_bid}_pos_{w_i}:other_guard_{eval_bid}={res['state']}_overlap")
    return {"valid": len(reasons) == 0, "reasons": reasons}


def evaluate_empirical_coverage_on_branch_witnesses(
    *,
    family: str,
    new_binding_id: str,
    new_canonical_code: str,
    new_entrypoint: str,
    old_binding: MechanismExecutableBinding,
    old_positive_tasks_and_payloads: list[tuple[str, HoldoutOODTask, dict[str, Any], dict[str, str], dict[str, str]]],
) -> EmpiricalCoverageReceipt:
    """Execute `new_canonical_code` on frozen representative positive witnesses of `old_binding` inside isolated Docker
    containers (`0` LLM calls, explicit CPU/wall-ms accounting!) to check whether `new` empirically covers `old` (`§5`).
    """
    t0 = monotonic()
    passed_cnt = 0
    wit_ids: list[str] = []
    p_vers_list: list[dict[str, int]] = []
    for wit_id, w_task, w_pay, ro_deps, ro_shas in old_positive_tasks_and_payloads:
        wit_ids.append(wit_id)
        p_vers_list.append(dict(old_binding.parent_parameter_versions))
        cand_c = render_parameterized_input_adapter(
            family=family,
            source_code=new_canonical_code,
            source_entrypoint=new_entrypoint,
            target_entrypoint=w_task.declared_callable_entrypoints[0],
            param_payload=w_pay,
        )
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=w_task,
            candidate_code=cand_c,
            readonly_dependencies=ro_deps,
            expected_dependency_sha256=ro_shas,
        )
        if pr["accepted"]:
            passed_cnt += 1
    el_ms = round((monotonic() - t0) * 1000.0, 2)
    n_eval = len(old_positive_tasks_and_payloads)
    status = "unknown" if n_eval == 0 else ("supported_on_witnesses" if passed_cnt == n_eval else "refuted")
    new_sha = hashlib.sha256(new_canonical_code.encode("utf-8")).hexdigest()
    old_sha = old_binding.authoritative_executed_code_sha256()
    rid = "ecov_" + hashlib.sha256(f"{new_sha}:{old_sha}:{wit_ids}:{status}".encode("utf-8")).hexdigest()[:16]
    return EmpiricalCoverageReceipt(
        receipt_id=rid,
        new_binding_id=new_binding_id,
        new_solver_sha256=new_sha,
        old_binding_id=old_binding.binding_id,
        old_solver_sha256=old_sha,
        witness_ids=wit_ids,
        parameter_versions_by_witness=p_vers_list,
        verifier_version=SHARED_KERNEL_VERSION,
        status=status,
        passed_witness_count=passed_cnt,
        evaluated_witness_count=n_eval,
        total_cpu_probe_wall_ms=el_ms,
        mean_probe_wall_ms_ratio=1.0,
    )


def propose_and_validate_m97_multibranch_split(
    *,
    client: Any,
    stage_subdir: str,
    spec: M97WorldSpec,
    task_instruction_text: str,
    existing_bindings: list[MechanismExecutableBinding],
    new_branch_id_placeholder: str,
    new_code: str,
    new_entrypoint: str,
    positive_witness_features_by_branch: dict[str, list[dict[str, float | None]]],
    current_public_item: dict[str, Any],
    current_param_payload: dict[str, Any],
    parent_wiring_meta: dict[str, Any],
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Model-proposed multi-branch guard & permuted-parent induction (`0` prompt hints naming distractors, `run_in_container=True`!)."""
    existing_ids = [b.binding_id for b in existing_bindings]
    cand_parents = sorted(parent_wiring_meta.keys())
    schema = build_m97_multibranch_induction_schema(existing_ids, cand_parents)
    interventions = build_m97_intervention_functions(spec)

    sys_prompt = (
        "You are a scientific mechanism-lineage compiler. Given the task specification, existing branch solver(s), "
        "a newly verified branch solver, the measured primitive observable vectors `obs_01..obs_08` across tested "
        "positive witnesses of each branch, and the candidate calibration parent wiring table, propose:\n"
        "1. `dependent_upstream_parent_ids`: the exact subset of upstream calibration parents whose interventions actually change "
        "the new solver's physical predictions.\n"
        "2. `new_branch_applicability_guard` and `existing_branch_guards`: Kleene `all_of` conjunctions (`1..3` clauses over `obs_01..obs_08`) "
        "that cleanly partition all tested branch witness sets."
    )

    convo = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_specification": task_instruction_text,
                    "candidate_parent_wiring": parent_wiring_meta,
                    "existing_branches": [
                        {"binding_id": b.binding_id, "branch_tag": b.branch_tag, "current_guard": b.applicability_guard or b.precondition_guard}
                        for b in existing_bindings
                    ],
                    "new_solver_code_head": new_code[:2800],
                    "tested_positive_witnesses_by_branch": positive_witness_features_by_branch,
                },
                indent=2,
            ),
        }
    ]
    call_receipts = []
    attempts_log = []
    for att in range(1, max_attempts + 1):
        res = client.call_structured(
            system_prompt=sys_prompt,
            conversation_turns=convo,
            output_schema=schema,
            stage_subdir=f"{stage_subdir}_m97_ind_att{att}",
        )
        call_receipts.append(res)
        if not res.get("ok") or not isinstance(res.get("parsed"), dict):
            attempts_log.append({"attempt": att, "ok": False, "reason": "llm_call_failed"})
            continue
        parsed = res["parsed"]
        all_guards = {
            **{bid: parsed["existing_branch_guards"][bid] for bid in existing_ids},
            new_branch_id_placeholder: parsed["new_branch_applicability_guard"],
        }
        part_chk = verify_m97_multibranch_witness_partition(
            branch_guards=all_guards,
            positive_witness_features_by_branch=positive_witness_features_by_branch,
        )
        sens_chk = verify_upstream_parent_perturbation_sensitivity(
            family=spec.family,
            candidate_upstream_ids=cand_parents,
            proposed_upstream_ids=list(parsed["dependent_upstream_parent_ids"]),
            c1_canonical_code=new_code,
            c1_entrypoint=new_entrypoint,
            public_item=current_public_item,
            base_param_payload=current_param_payload,
            intervention_fn_by_upstream_id=interventions,
            run_in_container=True,
        )
        attempts_log.append({"attempt": att, "partition_check": part_chk, "sensitivity_check": sens_chk})
        if part_chk["valid"] and sens_chk["proposal_matches_empirical_sensitivity"]:
            return {
                "accepted": True,
                "attempts_used": att,
                "validated_parent_ids": list(sens_chk["empirically_sensitive_upstream_ids"]),
                "new_branch_guard": parsed["new_branch_applicability_guard"],
                "existing_branch_guards": parsed["existing_branch_guards"],
                "violated_component_id": str(parsed["violated_component_id"]),
                "preserved_component_ids": list(parsed["preserved_component_ids"]),
                "mechanistic_diagnosis": str(parsed["mechanistic_diagnosis"]),
                "attempts_log": attempts_log,
                "provider_call_receipts": call_receipts,
            }
        convo.append({"role": "assistant", "content": json.dumps(parsed)})
        convo.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "host_verification_rejected": True,
                        "witness_partition_check": part_chk,
                        "container_sensitivity_check": sens_chk,
                        "instruction": "Fix guard thresholds/clauses or parent subset to match empirical witness values and container sensitivity.",
                    }
                ),
            }
        )
    return {"accepted": False, "attempts_used": max_attempts, "attempts_log": attempts_log, "provider_call_receipts": call_receipts}


# ============================================================================
# 5. Reference Dev-Seed Solvers (`0`-LLM Unit Testing on `DEV_M97_WORLD_SPECS` ONLY)
# ============================================================================


def build_m97_reference_dev_solver_code(family: str, entrypoint: str, regime_kind: str) -> str:
    """Reference `C0`, `C1`, `C2` solvers used ONLY in `0`-LLM dev-world tests (`DEV_M97_WORLD_SPECS`)!"""
    assert regime_kind in {"C0", "C1", "C2"}
    if family == "alpha":
        return f'''from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def {entrypoint}(dataset: dict) -> dict:
    tr, te = dataset["train"], dataset["test"]
    past_dt = np.asarray(dataset["past_dt"], dtype=float)
    eval_t = np.asarray(dataset["eval_times"], dtype=float)
    cs = float(dataset["crit_stress"])
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
        cm, csd = np.nanmean(rc, axis=0), np.nanstd(rc, axis=0, ddof=0)
        v = np.isfinite(cm) & np.isfinite(csd) & (csd > 1e-8)
        X_std[labs == c] = (X_std[labs == c] - float(np.median(cm[v]))) / float(np.median(csd[v]))
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
    y_c = np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]
    y_r = np.asarray(tr["instant_rate_obs"], dtype=float)
    K = y_c.shape[1]
    rk = {regime_kind!r}

    def _prop(ps_a, t1, t2):
        taus = np.array([t1, t2], dtype=float)
        z = np.zeros((len(ps_a), 2), dtype=float)
        for si, dt in enumerate(past_dt):
            dec = np.exp(-float(dt) / taus)
            z = z * dec[None, :] + (1.0 - dec[None, :]) * ps_a[:, si : si + 1]
        return z

    def _design(ps_a, h0_a, X_a, t1, t2):
        n = len(h0_a)
        z = _prop(ps_a, t1, t2)
        gap = h0_a[:, None] - z
        e0 = 1.0 - np.exp(-eval_t[None, :] / t1)
        e1 = 1.0 - np.exp(-eval_t[None, :] / t2)
        if rk == "C0":
            phi = eval_t
            Dc = np.zeros((n * K, 7), dtype=float)
            Dc[:, 0] = (gap[:, 0:1] * e0).ravel(); Dc[:, 1] = (gap[:, 1:2] * e1).ravel()
            Dc[:, 2] = np.broadcast_to(phi[None, :], (n, K)).ravel()
            for j in range(4): Dc[:, 3 + j] = (X_a[:, j : j + 1] * phi[None, :]).ravel()
            Dr = np.column_stack([gap[:, 0] / t1, gap[:, 1] / t2, np.ones(n), X_a])
            return Dc, Dr
        elif rk == "C1":
            phi = np.log1p(eval_t)
            d_cr = np.sum((np.maximum(0.0, np.abs(ps_a) - cs) ** 2) * past_dt[None, :], axis=1)
            act = np.tanh(d_cr)
            Dc = np.zeros((n * K, 9), dtype=float)
            Dc[:, 0] = (gap[:, 0:1] * e0).ravel(); Dc[:, 1] = (gap[:, 1:2] * e1).ravel()
            Dc[:, 2] = np.broadcast_to(phi[None, :], (n, K)).ravel()
            for j in range(4): Dc[:, 3 + j] = (X_a[:, j : j + 1] * phi[None, :]).ravel()
            Dc[:, 7] = ((act * h0_a)[:, None] * e0).ravel()
            Dc[:, 8] = ((act * z[:, 0])[:, None] * phi[None, :]).ravel()
            Dr = np.column_stack([gap[:, 0] / t1, gap[:, 1] / t2, np.ones(n), X_a, (act * h0_a) / t1, act * z[:, 0]])
            return Dc, Dr
        else:
            phi = eval_t / (1.0 + 0.85 * eval_t)
            sl = np.sign(h0_a) * np.sqrt(np.maximum(np.abs(h0_a), 1e-12))
            Dc = np.zeros((n * K, 9), dtype=float)
            Dc[:, 0] = (gap[:, 0:1] * e0).ravel(); Dc[:, 1] = (gap[:, 1:2] * e1).ravel()
            Dc[:, 2] = np.broadcast_to(phi[None, :], (n, K)).ravel()
            for j in range(4): Dc[:, 3 + j] = (X_a[:, j : j + 1] * phi[None, :]).ravel()
            Dc[:, 7] = (sl[:, None] * phi[None, :]).ravel()
            Dc[:, 8] = ((gap[:, 0] - gap[:, 1])[:, None] * e0).ravel()
            Dr = np.column_stack([gap[:, 0] / t1, gap[:, 1] / t2, np.ones(n), X_a, sl, (gap[:, 0] - gap[:, 1]) / t1])
            return Dc, Dr

    def _fit(t1, t2, iters=7):
        Dc, Dr = _design(tr_ps, tr_h0, X_tr, t1, t2)
        D_all = np.vstack([Dc, 1.8 * Dr]); y_all = np.concatenate([y_c.ravel(), 1.8 * y_r])
        w = np.ones_like(y_all); p = np.zeros(D_all.shape[1]); s2 = 0.042 ** 2
        for _ in range(iters):
            sw = np.sqrt(w)
            p, *_ = np.linalg.lstsq(D_all * sw[:, None], y_all * sw, rcond=None)
            res = y_all - D_all @ p
            lc = float(np.sum(np.log1p((res[:ntr * K] ** 2) / s2)) + 0.5 * np.sum((res[ntr * K:] ** 2) / (0.022 ** 2)))
            w[:ntr * K] = 1.0 / (1.0 + (res[:ntr * K] ** 2) / s2)
            w[ntr * K:] = 1.0 / (1.0 + 0.2 * (res[ntr * K:] ** 2) / (0.022 ** 2))
        return lc, p

    best = (1e30, 0.55, 3.8, None)
    for t1 in np.geomspace(0.32, 1.15, 8):
        for t2 in np.geomspace(max(t1 * 2.0, 2.0), 5.8, 8):
            lv, pp = _fit(float(t1), float(t2), iters=5)
            if lv < best[0]: best = (lv, float(t1), float(t2), pp)
    ro = minimize(lambda u: _fit(float(np.exp(u[0])), float(np.exp(u[0]) + np.exp(u[1])), iters=8)[0],
                  x0=np.array([np.log(best[1]), np.log(best[2] - best[1])]), method="Nelder-Mead", options={{"maxiter": 80}})
    t1h = float(np.exp(ro.x[0])); t2h = float(t1h + np.exp(ro.x[1]))
    _, p_hat = _fit(t1h, t2h, iters=12)
    hw = p_hat[:7]
    rw = np.zeros(2, dtype=float) if rk == "C0" else p_hat[7:9]
    Dc_te, Dr_te = _design(te_ps, te_h0, X_te, t1h, t2h)
    ps_pred = np.asarray(te["strain_0"], dtype=float)[:, None] + (Dc_te @ p_hat).reshape((len(te_h0), K))
    ir_pred = Dr_te @ p_hat
    return {{"estimated_taus": np.array([t1h, t2h]), "hereditary_weights": hw, "regime_mode_weights": rw,
            "predicted_strain": ps_pred, "initial_rate_pred": ir_pred}}
'''

    return f'''from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

def {entrypoint}(study: dict) -> dict:
    Lx, Ly = float(study["Lx"]), float(study["Ly"])
    nx, ny = int(study["nx"]), int(study["ny"])
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    xs = np.linspace(0.0, Lx, nx); ys = np.linspace(0.0, Ly, ny)
    cal = study["sensor_calibration_sweeps"]
    cb = np.asarray(cal["bank_id"], dtype=int).ravel()
    cr = np.asarray(cal["ref_signal"], dtype=float).ravel()
    cy = np.asarray(cal["raw_readout"], dtype=float).ravel()
    bg = np.ones(4, dtype=float); bo = np.zeros(4, dtype=float)
    for b in range(4):
        m = cb == b
        if int(np.sum(m)) >= 2:
            bg[b], bo[b] = np.linalg.lstsq(np.column_stack([cr[m], np.ones(int(np.sum(m)))]), cy[m], rcond=None)[0]
    rb = np.asarray(study["row_bank_id"], dtype=int).ravel()
    y_cal = (np.asarray(study["y_obs_raw"], dtype=float).ravel() - bo[rb]) / bg[rb]
    rk = {regime_kind!r}

    def _op1d(n, h, kap, vel, rob):
        A = np.zeros((n, n), dtype=float)
        for i in range(1, n - 1):
            kp = 0.5 * (kap[i] + kap[i + 1]); km = 0.5 * (kap[i] + kap[i - 1])
            A[i, i - 1] = -km / (h * h) - max(vel[i], 0.0) / h
            A[i, i + 1] = -kp / (h * h) - max(-vel[i], 0.0) / h
            A[i, i] = (kp + km) / (h * h) + abs(vel[i]) / h
        if rk == "C2":
            for i in (0, n - 1):
                im = (i - 1) % n; ip = (i + 1) % n
                kp = 0.5 * (kap[i] + kap[ip]); km = 0.5 * (kap[i] + kap[im])
                A[i, im] += -km / (h * h) - max(vel[i], 0.0) / h
                A[i, ip] += -kp / (h * h) - max(-vel[i], 0.0) / h
                A[i, i] += (kp + km) / (h * h) + abs(vel[i]) / h
        elif rk == "C1":
            A[0, 0] = kap[0] / (h * h); A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h); A[n - 1, n - 2] = -kap[-1] / (h * h)
        else:
            A[0, 0] = kap[0] / (h * h) + rob / h; A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h) + rob / h; A[n - 1, n - 2] = -kap[-1] / (h * h)
        return A

    Ax = _op1d(nx, dx, np.asarray(study["kappa_x"], dtype=float), np.asarray(study["vel_x"], dtype=float), float(study["robin_x"]))
    Ay = _op1d(ny, dy, np.asarray(study["kappa_y"], dtype=float), np.asarray(study["vel_y"], dtype=float), float(study["robin_y"]))
    V_eff = np.asarray(study["V_reaction_2d"], dtype=float).copy()
    if rk == "C1":
        vt = np.asarray(study["past_vortex_circulation_trace"], dtype=float)
        cv = float(study["critical_vortex_circulation"])
        v_mem = float(np.sum(np.maximum(0.0, vt - cv) ** 2))
        wm = np.exp(-((xs[:, None] / (0.25 * Lx)) ** 2)) + np.exp(-(((Lx - xs[:, None]) / (0.25 * Lx)) ** 2))
        V_eff = V_eff + (0.95 + 0.55 * math.tanh(v_mem)) * wm * (0.6 + 0.4 * np.sin(math.pi * ys[None, :] / Ly))
    elif rk == "C2":
        rc = float(study["recirculation_loop_mode"])
        V_eff = V_eff + 0.75 * rc * np.cos(2.0 * math.pi * xs[:, None] / Lx) * np.cos(2.0 * math.pi * ys[None, :] / Ly)

    L_f = np.kron(np.eye(ny), Ax) + np.kron(Ay, np.eye(nx)) + np.diag(V_eff.ravel(order="F"))
    modes = np.asarray(study["source_modes"], dtype=float)
    M = modes.shape[0]
    B_rhs = np.column_stack([modes[m].ravel(order="F") for m in range(M)])
    sij = np.asarray(study["sensor_ij"], dtype=int)
    si, sj = sij[:, 0], sij[:, 1]
    mu_s = np.asarray(study["mu_shifts"], dtype=float)
    U_last = np.empty((M, nx, ny), dtype=float)
    S_blks = []
    I_f = np.eye(nx * ny, dtype=float)
    for k_i, mu in enumerate(mu_s):
        sf = np.linalg.solve(L_f + float(mu) * I_f, B_rhs)
        U_k = np.empty((M, nx, ny), dtype=float)
        for m in range(M): U_k[m] = sf[:, m].reshape((nx, ny), order="F")
        if k_i == len(mu_s) - 1: U_last = U_k
        S_blks.append(U_k[:, si, sj].T)
    S_st = np.vstack(S_blks)
    Z = np.asarray(study["Z_nuisance"], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + float(study["nuisance_ridge"]) * np.eye(Z.shape[1]), Z.T)
    S_ti = S_st - Z @ (Pz @ S_st); r_y = y_cal - Z @ (Pz @ y_cal)
    lw, ub, bt = float(study["source_ridge"]), float(study["u_box"]), float(study["total_budget"])
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
    tf = np.tensordot(w, U_last, axes=(0, 0))
    return {{"weights": w, "terminal_field": tf,
            "kkt_certificate": {{"weights": w.tolist(), "nu_sum": best_nu, "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
                                 "lam_upper": np.maximum(0.0, best_nu - g).tolist(), "primal_objective": jv, "dual_lower_bound": float(min(best_lb, jv))}}}}
'''


def expected_m97_public_output_keys(family: str) -> list[str]:
    if family == "alpha":
        return sorted(["estimated_taus", "hereditary_weights", "regime_mode_weights", "predicted_strain", "initial_rate_pred"])
    return sorted(["weights", "terminal_field", "kkt_certificate"])


def m97_task_to_frontier_task(
    task: HoldoutOODTask,
    *,
    continual_context_block: str = "",
    allowed_readonly_dependencies: tuple[str, ...] = (),
) -> m6f.FrontierBenchmarkTask:
    base_ft = m95_task_to_frontier_task(
        task,
        continual_context_block=continual_context_block,
        allowed_readonly_dependencies=allowed_readonly_dependencies,
    )
    return m6f.FrontierBenchmarkTask(**{**base_ft.__dict__, "benchmark": "Hegel-M97-NonNested-Counterfactual-Lineage-World"})


def run_m97_equal_capability_pre_llm_trial(
    *,
    arm: str,
    spec: M97WorldSpec,
    task: HoldoutOODTask,
    public_item: dict[str, Any],
    public_input_sha256: str,
    prior_archive_entries: list[dict[str, Any]],
    flat_param_snapshot: dict[str, Any] | None = None,
    lineage_ctrl: ImmutableBranchLineageController | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
    readonly_dependencies: dict[str, str] | None = None,
    expected_dependency_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Equal-capability (`max_pre_llm_probes = 1`) pre-LLM execution trial for `M9.7`:
    - `Flat_EqualInfo_Multiversion`: Selects latest `public_gate_verified` canonical solver, applies 1-hop validated
      per-channel OLS calibration (`0` double-wrap!), and runs `1` isolated container probe.
    - `Hegel_DemandGated_Graph`: Evaluates `ImmutableBranchLineageController.select_branch_deterministically(...)`
      (separating `applicability_guard` from `selection_priority`), verifies content-addressed `QualificationReceipt` +
      full `parent_parameter_versions` vector, runs `1` isolated container probe, and records an evidence-bound
      `SolverEvaluationWitness` ONLY for the branch actually tested (`0` fabricated negative witnesses!).
    """
    req_keys = expected_m97_public_output_keys(spec.family)
    target_ep = task.declared_callable_entrypoints[0]

    if arm == "Flat_EqualInfo_Multiversion":
        compat = [
            e
            for e in reversed(prior_archive_entries)
            if sorted(e.get("output_schema_keys") or []) == req_keys
            and e.get("public_qualification_status", "public_gate_verified") == "public_gate_verified"
        ]
        dedup: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ent in compat:
            can_c = str(ent.get("canonical_algorithm_code") or ent["submitted_code"])
            h_c = hashlib.sha256(can_c.encode("utf-8")).hexdigest()
            if h_c not in seen:
                seen.add(h_c)
                dedup.append(ent)
        if not dedup:
            return {"arm": arm, "accepted_pre_llm": False, "pre_llm_probes_used": 0, "reason": "no_compatible_prior_solver"}
        chosen_ent = dedup[0]
        can_code = str(chosen_ent.get("canonical_algorithm_code") or chosen_ent["submitted_code"])
        can_ep = str(chosen_ent.get("canonical_entrypoint") or chosen_ent["entrypoint"])
        cand_code = render_parameterized_input_adapter(
            family=spec.family,
            source_code=can_code,
            source_entrypoint=can_ep,
            target_entrypoint=target_ep,
            param_payload=flat_param_snapshot,
        )
        t_pr = monotonic()
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=task,
            candidate_code=cand_code,
            readonly_dependencies=readonly_dependencies,
            expected_dependency_sha256=expected_dependency_sha256,
        )
        pr_ms = round((monotonic() - t_pr) * 1000.0, 2)
        return {
            "arm": arm,
            "accepted_pre_llm": pr["accepted"],
            "pre_llm_probes_used": 1,
            "probe_wall_ms": pr_ms,
            "retained_distinct_versions": len(dedup),
            "selected_source_short_name": chosen_ent["short_name"],
            "selected_canonical_algorithm_code": can_code,
            "selected_canonical_entrypoint": can_ep,
            "parameter_payload": deepcopy(flat_param_snapshot),
            "candidate_code": (cand_code if pr["accepted"] else None),
            "probe_result": pr,
            "reason": ("accepted_by_shared_public_gate" if pr["accepted"] else "rejected_by_shared_public_scientific_gate"),
        }

    # Arm 2: `Hegel_DemandGated_Graph`
    assert lineage_ctrl is not None and graph is not None and registry is not None

    def _feat_for_binding(b: MechanismExecutableBinding) -> dict[str, float | None]:
        pids = list(dict.fromkeys([b.parameter_mechanism_id, *b.parent_parameter_mechanism_ids]))
        pay_b = assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=pids)
        return compute_m97_primitive_observables(spec.family, public_item, pay_b)

    sel = lineage_ctrl.select_branch_deterministically(req_schema_keys=req_keys, feature_evaluator=_feat_for_binding)
    chosen_b: MechanismExecutableBinding | None = sel["selected"]
    if chosen_b is None:
        # Bounded 1-probe fallback on highest-priority verified branch when guards return `no_applicable_branch` / `unknown`,
        # so any split is ALWAYS grounded in a genuine tested scientific failure witness (`N07`)!
        ver_bs = sorted(
            [b for b in registry.bindings.values() if b.chain_id == spec.family and b.qualification_status == "verified"],
            key=lambda x: (-int(x.selection_priority), x.binding_id),
        )
        if not ver_bs:
            return {"arm": arm, "accepted_pre_llm": False, "pre_llm_probes_used": 0, "reason": "no_verified_bindings"}
        chosen_b = ver_bs[0]
        chosen_feats = _feat_for_binding(chosen_b)
        sel_mode = f"bounded_probe_on_{sel['status']}"
    else:
        chosen_feats = sel["selected_features"]
        sel_mode = str(sel["status"])

    pids_c = list(dict.fromkeys([chosen_b.parameter_mechanism_id, *chosen_b.parent_parameter_mechanism_ids]))
    chosen_pay = assemble_m97_parameter_payload(spec=spec, graph=graph, parent_ids=pids_c)
    can_b_code = chosen_b.canonical_algorithm_code or chosen_b.parameterized_closure_code
    can_b_ep = chosen_b.canonical_entrypoint or chosen_b.parameterized_entrypoint
    cand_code = render_parameterized_input_adapter(
        family=spec.family,
        source_code=can_b_code,
        source_entrypoint=can_b_ep,
        target_entrypoint=target_ep,
        param_payload=chosen_pay,
        binding=chosen_b,
        graph=graph,
        registry=registry,
    )
    t_pr = monotonic()
    pr = execute_single_pre_llm_public_gate_probe_in_container(
        task=task,
        candidate_code=cand_code,
        readonly_dependencies=readonly_dependencies,
        expected_dependency_sha256=expected_dependency_sha256,
    )
    pr_ms = round((monotonic() - t_pr) * 1000.0, 2)
    f_kind = "none" if pr["accepted"] else ("scientific_or_numerical" if pr.get("schema_valid") else "schema_or_interface")
    wit = lineage_ctrl.record_tested_witness(
        binding=chosen_b,
        public_input_sha256=public_input_sha256,
        passed=bool(pr["accepted"]),
        failure_kind=f_kind,
        features=chosen_feats,
        probe_wall_ms=pr_ms,
    )
    return {
        "arm": arm,
        "accepted_pre_llm": pr["accepted"],
        "pre_llm_probes_used": 1,
        "probe_wall_ms": pr_ms,
        "selection_mode": sel_mode,
        "selected_binding_id": chosen_b.binding_id,
        "selected_branch_tag": chosen_b.branch_tag,
        "selected_structure_origin": chosen_b.structure_origin,
        "selected_canonical_algorithm_code": can_b_code,
        "selected_canonical_entrypoint": can_b_ep,
        "tested_witness": wit.to_dict(),
        "parameter_payload": deepcopy(chosen_pay),
        "primitive_observables": chosen_feats,
        "guard_evaluations": sel["evaluations"],
        "candidate_code": (cand_code if pr["accepted"] else None),
        "probe_result": pr,
        "reason": ("accepted_by_shared_public_gate" if pr["accepted"] else f"tested_branch_{chosen_b.binding_id}_failed_{f_kind}"),
    }


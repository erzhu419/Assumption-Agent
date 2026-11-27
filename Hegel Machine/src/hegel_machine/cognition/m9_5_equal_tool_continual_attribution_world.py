"""Hegel M9.5 Equal-Tool Three-Arm Multi-World Continual Attribution Benchmark.

Closes `M9.4` Audit Findings (`A01–A10`, `E01–E09`, `C01–C02`) on Disjoint Multi-World Continual Streams:
  1. **Equal-Tool Pre-LLM Execution Service (`max_pre_llm_probes = 1` Across ALL 3 Arms)**:
     - `Archive_AutoReplay`: Stateless raw replay of most-recent schema-compatible archived `.py` deliverable (`0` persistent
       parameter memory, `0` guards; `1` pre-LLM public-gate probe; falls back to `gpt-6.2-sol` if rejected).
     - `Flat_ParameterizedCache`: Flat latest-sweep `lstsq` parameter cache (retaining prior lab/bank values when a lab/bank has `< 2`
       rows in the current batch, without `M8` variational validation gating and without physical guards) + most-recent
       schema-compatible solver (`1` pre-LLM public-gate probe; falls back to `gpt-6.2-sol` if rejected).
     - `Full_Mechanism_Graph`: Wing-scoped `PersistentMechanismGraph` (`wing_a`, `wing_b`, plus `0.0`-drift `sentinel`) governed by
       `M8VariationalPlanner` (`AncestorReweight` vs `NoChange`, propagating `v3` across `fanout = 2` `{C0, C1}` bindings on `G2`) +
       cryptographically shuffled 3-valued `ObservablePreconditionGuard` branch selection (`1` pre-LLM public-gate probe; falls back
       to `gpt-6.2-sol` if rejected).
  2. **100% Unified Input & Output Schemas Across Non-`T` Steps (`S, N, G1, C, R, C', G2, M`)**:
     - Eliminates `RouteBySchema` (`A02`): `C0` and `C1` accept the EXACT same input keys and emit the EXACT same output dict keys
       and array shapes on both Alpha and Beta!
     - Pure calibration-only input adapter (`0` PDE/potential decomposition lines in `render_parameterized_input_adapter`!).
  3. **Strict Dev vs Confirmation Separation (`Zero Post-Freeze Tuning`)**:
     - `DEV_WORLD_SPECS` (`2` worlds: `dev_alpha_00`, `dev_beta_00`) used ONLY for pre-freeze unit & integration tests.
     - `CONFIRMATION_WORLD_SPECS` (`8` truly independent worlds: `alpha_w01..04`, `beta_w01..04`, each with its own fresh RNG seed,
       randomized physical parameters, independent blind `Stage S` LLM run, and causally-valid randomized anonymous `batch_01..08` schedule).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    ObservablePreconditionGuard,
    evaluate_observable_precondition_guard,
)
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

M95_ARMS: tuple[str, ...] = (
    "Archive_AutoReplay",
    "Flat_ParameterizedCache",
    "Full_Mechanism_Graph",
)
M95_EXTENDED_ARMS: tuple[str, ...] = (
    "Archive_AutoReplay",
    "Flat_ParameterizedCache",
    "Flat_Multiversion_Validated",
    "Full_Mechanism_Graph",
)
ADAPTER_PRECALIBRATED_SENTINEL = "_HEGEL_ADAPTER_PRECALIBRATED_DO_NOT_REWRAP"


@dataclass(frozen=True)
class M95WorldSpec:
    world_id: str
    family: str  # 'alpha' | 'beta'
    split: str  # 'dev' | 'confirmation'
    world_seed: int
    schedule_events: tuple[str, ...]  # Causally valid 8-step permutation of ('T','N','G1','C','R','C_prime','G2','M')
    v2_gains: tuple[float, float, float, float]
    v2_offsets: tuple[float, float, float, float]
    v3_wing_a_gains: tuple[float, float]
    v3_wing_a_offsets: tuple[float, float]


def _make_world_spec(world_id: str, family: str, split: str, world_seed: int, early_order: tuple[str, str, str], mid_order: tuple[str, str]) -> M95WorldSpec:
    rng = np.random.default_rng(world_seed)
    # Distinct nontrivial v2 full-facility calibration shift (`|gain - 1| >= 0.16`)
    sgn_v2 = np.array([1.0, -1.0, 1.0, -1.0]) if (world_seed // 1000) % 2 == 0 else np.array([-1.0, 1.0, -1.0, 1.0])
    v2_g = tuple(round(float(1.0 + sgn_v2[i] * rng.uniform(0.18, 0.30)), 4) for i in range(4))
    v2_b = tuple(round(float(sgn_v2[i] * rng.uniform(0.08, 0.15)), 4) for i in range(4))
    # Distinct v3 Wing-A (`channels 0, 1`) recalibration shift flipping sign relative to v2!
    v3_ga = tuple(round(float(1.0 - sgn_v2[i] * rng.uniform(0.20, 0.32)), 4) for i in range(2))
    v3_ba = tuple(round(float(-sgn_v2[i] * rng.uniform(0.10, 0.17)), 4) for i in range(2))
    sched = (*early_order, "C", *mid_order, "G2", "M")
    assert set(sched) == {"T", "N", "G1", "C", "R", "C_prime", "G2", "M"} and len(sched) == 8
    return M95WorldSpec(
        world_id=world_id,
        family=family,
        split=split,
        world_seed=world_seed,
        schedule_events=sched,
        v2_gains=v2_g,  # type: ignore[arg-type]
        v2_offsets=v2_b,  # type: ignore[arg-type]
        v3_wing_a_gains=v3_ga,  # type: ignore[arg-type]
        v3_wing_a_offsets=v3_ba,  # type: ignore[arg-type]
    )


DEV_WORLD_SPECS: tuple[M95WorldSpec, ...] = (
    _make_world_spec("dev_alpha_00", "alpha", "dev", 51000, ("T", "N", "G1"), ("R", "C_prime")),
    _make_world_spec("dev_beta_00", "beta", "dev", 52000, ("N", "T", "G1"), ("C_prime", "R")),
)

CONFIRMATION_WORLD_SPECS: tuple[M95WorldSpec, ...] = (
    _make_world_spec("alpha_w01", "alpha", "confirmation", 951000, ("T", "N", "G1"), ("R", "C_prime")),
    _make_world_spec("alpha_w02", "alpha", "confirmation", 952000, ("N", "T", "G1"), ("C_prime", "R")),
    _make_world_spec("alpha_w03", "alpha", "confirmation", 953000, ("N", "G1", "T"), ("R", "C_prime")),
    _make_world_spec("alpha_w04", "alpha", "confirmation", 954000, ("T", "G1", "N"), ("C_prime", "R")),
    _make_world_spec("beta_w01", "beta", "confirmation", 961000, ("T", "N", "G1"), ("R", "C_prime")),
    _make_world_spec("beta_w02", "beta", "confirmation", 962000, ("N", "T", "G1"), ("C_prime", "R")),
    _make_world_spec("beta_w03", "beta", "confirmation", 963000, ("N", "G1", "T"), ("R", "C_prime")),
    _make_world_spec("beta_w04", "beta", "confirmation", 964000, ("T", "G1", "N"), ("C_prime", "R")),
)


# ============================================================================
# 1. Unified-Schema Physics Generators (`Alpha` & `Beta`, Anonymous Task IDs)
# ============================================================================

_M95_ALPHA_GEN_PY = r'''
def _gen_m95_alpha_cohort(
    seed,
    taus_pair,
    theta_vec,
    *,
    stress_gains=(1.0, 1.0, 1.0, 1.0),
    stress_biases=(0.0, 0.0, 0.0, 0.0),
    cal_mode="full_40",
    v2_wing_b_gains=None,
    v2_wing_b_biases=None,
    n_train=200,
    n_test=80,
    H=18,
    K=10,
    dt_mode="base_var",
    preload_scale=0.0,
    yield_stress=2.95,
    vp_theta=(0.0, 0.0),
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
        # Sparse in-batch check: ONLY 2 points on Lab 0 (`0` in-batch rows for Labs 1, 2, 3 -> requires persistent/historical calibration!)
        cal_labs = np.array([0, 0], dtype=int)
        cal_cmd = np.array([-1.35, 1.45], dtype=float)
        cal_true = s_gains[cal_labs] * cal_cmd + s_biases[cal_labs] + rng.normal(scale=1e-4, size=2)
    elif cal_mode == "scoped_wing_a_plus_emi_wing_b":
        # Wing A (`labs 0, 1`): 24 clean recalibrated v3 rows; Wing B (`labs 2, 3`): 6 spot rows where first 4 suffer +/- 4.8 EMI spike
        # while trailing validation pair [4:6] sits cleanly at Wing B's unchanged v2 law!
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
        # Ensure balanced representation across all 4 labs (`Wing A` and `Wing B`)
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

    def _sim(Z_lat, labs_arr, *, n_twins=0, add_glitches=False):
        N = Z_lat.shape[0]
        init_mem = rng.normal(scale=preload_scale, size=(N, 2)) if preload_scale > 0.0 else np.zeros((N, 2), dtype=float)
        hist_true = rng.normal(scale=1.45, size=(N, H))
        for s in range(1, H):
            hist_true[:, s] = 0.52 * hist_true[:, s - 1] + math.sqrt(1.0 - 0.52 ** 2) * hist_true[:, s]
        s0 = rng.normal(scale=0.35, size=N)
        hold_true = rng.uniform(-2.05, 2.15, size=N)
        if has_vp:
            hi_idx = np.arange(N // 2)
            sgn = np.where(rng.random(len(hi_idx)) < 0.55, 1.0, -1.0)
            hold_true[hi_idx] = sgn * rng.uniform(ys + 0.35, ys + 1.30, size=len(hi_idx))
            hist_true[hi_idx, -6:] = sgn[:, None] * rng.uniform(ys + 0.25, ys + 1.15, size=(len(hi_idx), 6))
            sub_idx = np.arange(N // 2, N)
            hold_true[sub_idx] = rng.uniform(-0.86 * ys, 0.86 * ys, size=len(sub_idx))
            hist_true[sub_idx] = np.clip(hist_true[sub_idx] * 0.50, -0.88 * ys, 0.88 * ys)
        else:
            hold_true = np.clip(hold_true, -0.84 * ys, 0.84 * ys)
            hist_true = np.clip(hist_true, -0.84 * ys, 0.84 * ys)

        if n_twins > 0:
            for k in range(n_twins):
                ia, ib = N - 2 * n_twins + 2 * k, N - 2 * n_twins + 2 * k + 1
                labs_arr[ib] = labs_arr[ia]
                Z_lat[ib] = Z_lat[ia]
                s0[ib] = s0[ia]
                init_mem[ib] = -init_mem[ia]
                hist_true[ia] = np.clip(np.linspace(1.8, -1.3, H), -0.84 * ys, 0.84 * ys)
                hist_true[ib] = np.clip(np.linspace(-1.7, 1.5, H), -0.84 * ys, 0.84 * ys)
                hold_true[ia] = hold_true[ib] = 0.72 * ys

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
        crit = s0 + rng.uniform(0.15, 0.58, size=N)
        ts_full = np.concatenate([[0.0], eval_times])
        ys_full = np.column_stack([s0, clean])
        cross = np.full(N, float(eval_times[-1]), dtype=float)
        for i in range(N):
            c = float(crit[i])
            if ys_full[i, 0] >= c:
                cross[i] = 0.0
                continue
            for ki in range(len(ts_full) - 1):
                if ys_full[i, ki + 1] >= c:
                    dy = max(ys_full[i, ki + 1] - ys_full[i, ki], 1e-12)
                    cross[i] = float(ts_full[ki] + (c - ys_full[i, ki]) / dy * (ts_full[ki + 1] - ts_full[ki]))
                    break
        secant = (clean[:, -1] - s0) / np.maximum(np.abs(hold_true), 0.5)
        return init_mem, hist_cmd, s0, hold_cmd, rate_obs, true_rate0, clean, obs, crit, cross, secant, hold_true, hist_true

    tr_im, tr_hc, tr_s0, tr_h0c, tr_ro, tr_rt, tr_cl, tr_ob, _, _, _, _, _ = _sim(tr_Z, tr_lab, n_twins=0, add_glitches=True)
    te_im, te_hc, te_s0, te_h0c, te_ro, te_rt, te_cl, _, te_cr, te_cx, te_sec, te_ht, te_hist_t = _sim(te_Z, te_lab, n_twins=8, add_glitches=False)
    for k in range(8):
        ia = len(te_Z) - 16 + 2 * k
        ib = ia + 1
        te_st[ib] = te_st[ia]
        te_X[ia] = offsets[te_lab[ia]] + gains[te_lab[ia]] * te_Z[ia]
        te_X[ib] = offsets[te_lab[ib]] + gains[te_lab[ib]] * te_Z[ib]

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
    if preload_scale > 0.0:
        ds["train"]["initial_memory_states"] = tr_im
        ds["test"]["initial_memory_states"] = te_im
        ds["test"]["critical_strain"] = te_cr
    gt = {
        "taus": taus,
        "theta": theta,
        "vp_theta": np.asarray(vp_theta, dtype=float),
        "stress_gains": s_gains,
        "stress_biases": s_biases,
        "te_clean": te_cl,
        "te_true_rate0": te_rt,
        "te_cross": te_cx,
        "te_secant": te_sec,
        "te_hold_true": te_ht,
        "te_hist_true": te_hist_t,
    }
    return ds, gt
'''


_M95_BETA_GEN_PY = _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY + r'''
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
    ll = np.maximum(0.0, g - best_nu)
    lu = np.maximum(0.0, best_nu - g)
    return w, {
        "weights": w.tolist(),
        "nu_sum": float(best_nu),
        "lam_lower": ll.tolist(),
        "lam_upper": lu.tolist(),
        "primal_objective": float(j_val),
        "dual_lower_bound": float(min(best_lb, j_val)),
    }

def _gen_m95_beta_study(
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

    if not vortex_regime:
        # Laminar Separable Regime (`nonseparable_2d_additive_residual_ratio <= 1e-12`, `1.00 * kappa`)
        vx_part = 0.28 + 0.16 * np.cos(2.0 * math.pi * xs / Lx)
        vy_part = 0.24 + 0.13 * np.sin(2.0 * math.pi * ys / Ly)
        V_2d = vx_part[:, None] + vy_part[None, :]
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
        # Supercritical Vortex-Coupled Dispersion Regime (`nonseparable_ratio > 0.02`, turbulent eddy diffusivity `1.45 * kappa` + 2D non-separable V_reaction_2d)
        from scipy import sparse
        from scipy.sparse.linalg import splu
        V_2d = (
            4.6 * np.exp(-((X2 - 0.52 * Lx) ** 2 + (Y2 - 0.48 * Ly) ** 2) / 0.14) * np.sin(2.6 * math.pi * X2 / Lx) * np.sin(2.4 * math.pi * Y2 / Ly)
            + 3.2 * np.exp(-((X2 - 0.35 * Lx - 0.45 * Y2) ** 2) / 0.08)
        )
        V_2d = V_2d - float(np.min(V_2d)) + 0.15
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
    act_m = [1, 4, 7, 10]
    raw_w = np.array([0.88, 0.72, 0.58, 0.42], dtype=float)
    w_true[act_m] = raw_w * (b_tot / float(np.sum(raw_w)))
    gamma_true = rng.normal(scale=0.65, size=P_z)
    y_true_cal = S_stack @ w_true + Z_nuis @ gamma_true + rng.normal(scale=float(obs_noise_scale), size=n_rows)
    y_obs_raw = b_gains[row_bank_id] * y_true_cal + b_offs[row_bank_id]

    Pz_mat = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(P_z), Z_nuis.T)
    r_y = y_true_cal - Z_nuis @ (Pz_mat @ y_true_cal)
    S_tilde = S_stack - Z_nuis @ (Pz_mat @ S_stack)

    # Identical 17 input keys across ALL non-T Beta tasks (`S, N, G1, C, R, C', G2, M`)!
    study = {
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny,
        "kappa_x": kappa_x, "vel_x": vel_x, "robin_x": float(robin_pair[0]), "reaction_x": rx_base,
        "kappa_y": kappa_y, "vel_y": vel_y, "robin_y": float(robin_pair[1]), "reaction_y": ry_base,
        "V_reaction_2d": V_2d,
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
# 2. Task Builders (`Neutral Anonymous Task IDs`, Unified Contracts)
# ============================================================================


def _build_m95_alpha_task(
    *,
    task_id: str,
    short_name: str,
    wave_label: str,
    instruction_text: str,
    pub_call_expr: str,
    priv_cases_py: str,
    is_transfer_T: bool = False,
) -> HoldoutOODTask:
    artifact_name = f"{short_name}_solver.py"
    entry_name = f"solve_{short_name}"
    pkl_name = f"public_{short_name}.pkl"
    mod_stem = Path(artifact_name).stem

    if not is_transfer_T:
        # UNIFIED 5-KEY OUTPUT CONTRACT ACROSS `S, N, G1, C, R, C_prime, G2, M`!
        schema_keys = '(("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("viscoplastic_weights", (2,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,)))'
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
        schema_keys = '(("estimated_taus", (2,)), ("predicted_strain", (n_te, k_t)), ("secant_compliance", (n_te,)), ("hazard_crossing_times", (n_te,)))'
        priv_body = f'''
t0 = time.monotonic()
valid = True; max_tau = 0.0; max_s = 0.0; max_sec = 0.0; max_cx = 0.0; rows = []
for sd, ds, gt in cases:
    out = cand_mod.{entry_name}(ds)
    if not isinstance(out, dict) or not {{"estimated_taus", "predicted_strain", "secant_compliance", "hazard_crossing_times"}} <= set(out.keys()):
        valid = False; rows.append({{"seed": sd, "error": "missing_keys"}}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    sc = np.asarray(out["secant_compliance"], dtype=float)
    cx = np.asarray(out["hazard_crossing_times"], dtype=float)
    if et.shape != (2,) or ps.shape != gt["te_clean"].shape or sc.shape != gt["te_secant"].shape or cx.shape != gt["te_cross"].shape:
        valid = False; rows.append({{"seed": sd, "error": "shape_mismatch"}}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(sc)) and np.all(np.isfinite(cx)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({{"seed": sd, "error": "non_finite"}}); continue
    cal_d = ds["calibration_load_cells"]
    c_l = np.asarray(cal_d["lab_id"], dtype=int).ravel()
    c_x = np.asarray(cal_d["cmd_stress"], dtype=float).ravel()
    c_y = np.asarray(cal_d["true_stress"], dtype=float).ravel()
    g_ols = np.ones(4, dtype=float); b_ols = np.zeros(4, dtype=float)
    for lb_i in range(4):
        m_i = c_l == lb_i
        if int(np.sum(m_i)) >= 2:
            g_ols[lb_i], b_ols[lb_i] = np.linalg.lstsq(np.column_stack([c_x[m_i], np.ones(int(np.sum(m_i)))]), c_y[m_i], rcond=None)[0]
    te_lb = np.asarray(ds["test"]["lab_id"], dtype=int).ravel()
    te_h0_ols = np.asarray(ds["test"]["stress_hold_cmd"], dtype=float) * g_ols[te_lb] + b_ols[te_lb]
    sec_exp = (ps[:, -1] - ds["test"]["strain_0"]) / np.maximum(np.abs(te_h0_ols), 0.5)
    if float(np.max(np.abs(sc - sec_exp))) > 1e-3:
        valid = False; rows.append({{"seed": sd, "error": "secant_handoff_mismatch"}}); continue
    tr = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    sm = float(np.mean(np.abs(ps - gt["te_clean"])))
    scm = float(np.mean(np.abs(sc - gt["te_secant"])))
    cxm = float(np.mean(np.abs(cx - gt["te_cross"])))
    max_tau = max(max_tau, tr); max_s = max(max_s, sm); max_sec = max(max_sec, scm); max_cx = max(max_cx, cxm)
    rows.append({{"seed": sd, "tau_rel": round(tr, 6), "strain_mae": round(sm, 6), "secant_mae": round(scm, 6), "cross_mae": round(cxm, 6)}})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau <= 0.10 and max_s <= 0.075 and max_sec <= 0.095 and max_cx <= 0.28 and el <= 20.0)
print(json.dumps({{"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau, 6), "max_strain_mae": round(max_s, 6), "max_secant_mae": round(max_sec, 6), "max_crossing_mae": round(max_cx, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}}, allow_nan=False))
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
            + _M95_ALPHA_GEN_PY
            + f"\npub_ds, pub_gt = {pub_call_expr}\n"
            f'with open("/holdout_task/{pkl_name}", "wb") as fh:\n'
            '    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)\n'
            f'open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys\n'
            "import numpy as np\n"
            "_d = os.path.dirname(os.path.abspath(__file__))\n"
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
            'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
            "sys.exit(0 if len(errs) == 0 else 1)\n"
            '""")\n'
        ),
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            f"import {mod_stem} as cand_mod\n"
            + _M95_ALPHA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def _build_m95_beta_task(
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
        # UNIFIED 3-KEY OUTPUT CONTRACT ACROSS `S, N, G1, C, R, C_prime, G2, M`!
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
            + _M95_BETA_GEN_PY
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
            + _M95_BETA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def build_world_tasks(spec: M95WorldSpec) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]]]:
    """Instantiate `(stage_S_task, 8_anonymous_stream_tasks, hidden_schedule_manifest)` for `spec`."""
    rng = np.random.default_rng(spec.world_seed + 77)
    wid = spec.world_id

    v1_g = (1.0, 1.0, 1.0, 1.0)
    v1_b = (0.0, 0.0, 0.0, 0.0)
    v2_g = spec.v2_gains
    v2_b = spec.v2_offsets
    # Wing A (`0, 1`) shifts to v3 while Wing B (`2, 3`) stays at v2!
    v3_g = (spec.v3_wing_a_gains[0], spec.v3_wing_a_gains[1], v2_g[2], v2_g[3])
    v3_b = (spec.v3_wing_a_offsets[0], spec.v3_wing_a_offsets[1], v2_b[2], v2_b[3])

    if spec.family == "alpha":
        cal_hdr = (
            "1. MULTI-WING LOAD-CELL & REFERENCE-COUPON CALIBRATION (`Wing A = labs 0,1`, `Wing B = labs 2,3`):\n"
            "   - `dataset['calibration_load_cells']` provides in-batch load-cell checks `{'lab_id', 'cmd_stress', 'true_stress'}` obeying\n"
            "     `true_stress = stress_gain[c] * cmd_stress + stress_bias[c]` (`c in 0..3`; note: some continual batches carry only sparse\n"
            "     spot checks or occasional EMI outliers on a wing, so check `/tmp/hegel_continual_archive/` when in-batch rows are sparse/noisy!).\n"
            "   - Convert `past_stress_cmd` and `stress_hold_cmd` into calibrated physical `past_stress` and `stress_hold_0`.\n"
            "   - Standardize `coupon_features_raw` `(N, 4)` per lab via median column mean/std (`ddof=0`) of `ref_coupons` (`train + test`),\n"
            "     imputing `NaN`s with `(lab, stratum)` cell-balanced column means -> `X_aligned` `(N, 4)`.\n"
        )
        src_sn = f"{wid}_s00"
        t1_s = round(float(rng.uniform(0.50, 0.64)), 3)
        t2_s = round(float(rng.uniform(3.60, 4.25)), 3)
        th_s = [round(float(rng.uniform(0.65, 0.73)), 3), round(float(rng.uniform(0.52, 0.58)), 3), 0.09, 0.16, -0.13, 0.11, -0.08]
        src_task = _build_m95_alpha_task(
            task_id=f"{wid}_s00__coupon_creep_baseline",
            short_name=src_sn,
            wave_label=f"{wid}_source",
            instruction_text=(
                f"Implement `/holdout_task/{src_sn}_solver.py` exposing `def solve_{src_sn}(dataset: dict) -> dict:`\n\n"
                + cal_hdr
                + "2. SUB-YIELD 2-BRANCH HEREDITARY CREEP (`z_m(0) = 0`, `|stress| < dataset['yield_stress']`, ~8% Cauchy glitches on `strain_curves_obs`):\n"
                "   - Propagate `z_m <- exp(-past_dt[s]/tau_m)*z_m + (1 - exp(-past_dt[s]/tau_m))*past_stress[:, s]` (`0.20 <= tau_1 < tau_2 <= 8.0`).\n"
                "   - Sub-yield hold law: `eps(t_k) = strain_0 + theta_0*(stress_hold_0 - z_1)*(1 - exp(-t_k/tau_1)) + theta_1*(stress_hold_0 - z_2)*(1 - exp(-t_k/tau_2)) + (theta_2 + X_aligned @ theta_{3:7})*t_k`,\n"
                "     `instant_rate_obs` at `t=0+`, and sub-yield `viscoplastic_weights = np.zeros(2)`.\n"
                "3. RETURN unified dict `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'viscoplastic_weights': (2,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
                f"- Public dataset: `/holdout_task/public_{src_sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
            ),
            pub_call_expr=f"_gen_m95_alpha_cohort({spec.world_seed + 1}, ({t1_s}, {t2_s}), {th_s!r})",
            priv_cases_py=(
                f"    ({spec.world_seed + 501}, *_gen_m95_alpha_cohort({spec.world_seed + 501}, ({round(t1_s - 0.03, 3)}, {round(t2_s - 0.15, 3)}), [0.71, 0.55, 0.08, 0.17, -0.14, 0.12, -0.08])),\n"
                f"    ({spec.world_seed + 502}, *_gen_m95_alpha_cohort({spec.world_seed + 502}, ({round(t1_s + 0.05, 3)}, {round(t2_s + 0.25, 3)}), [0.64, 0.58, 0.10, -0.15, 0.13, -0.11, 0.09])),"
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
                tk = _build_m95_alpha_task(
                    task_id=tid,
                    short_name=sn,
                    wave_label=f"{wid}_step_{idx:02d}",
                    is_transfer_T=True,
                    instruction_text=(
                        f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(dataset: dict) -> dict:`\n\n"
                        + cal_hdr
                        + "2. PRE-LOADED MEMORY + SECANT COMPLIANCE & HAZARD CROSSING (`H = 22`, `K = 12`):\n"
                        "   - Initialize `z_m` from `initial_memory_states` `(N, 2)` before propagating across `past_dt`.\n"
                        "   - Return `{'estimated_taus': (2,), 'predicted_strain': (N_test, K), 'secant_compliance': (predicted_strain[:, -1] - test['strain_0']) / np.maximum(np.abs(stress_hold_0), 0.5), 'hazard_crossing_times': (N_test,)}`.\n"
                        f"- Public dataset: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                    ),
                    pub_call_expr=f'_gen_m95_alpha_cohort({sd_pub}, ({t1_e}, {t2_e}), {th_e!r}, H=22, K=12, dt_mode="geom_var", preload_scale=2.10)',
                    priv_cases_py=(
                        f'    ({sd_p1}, *_gen_m95_alpha_cohort({sd_p1}, (0.45, 3.35), {th_p1!r}, H=22, K=12, dt_mode="geom_var", preload_scale=2.10)),\n'
                        f'    ({sd_p2}, *_gen_m95_alpha_cohort({sd_p2}, (0.72, 4.75), {th_p2!r}, H=22, K=12, dt_mode="geom_var", preload_scale=2.10)),'
                    ),
                )
                cat = "adaptation_discovery"
            else:
                if ev == "N":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v1_g, v1_b, "full_40", 2.95, (0.0, 0.0), 0.13, "immediate_cal_negative_control"
                elif ev == "G1":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "full_40", 2.95, (0.0, 0.0), 0.08, "immediate_cal_negative_control"
                elif ev == "C":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "full_40", 1.52, (0.25, 0.17), 0.08, "adaptation_discovery"
                elif ev == "R":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "sparse_spot_lab0", 2.95, (0.0, 0.0), 0.08, "continual_history_guard_fanout"
                elif ev == "C_prime":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v2_g, v2_b, "sparse_spot_lab0", 1.48, (0.27, 0.18), 0.08, "continual_history_guard_fanout"
                elif ev == "G2":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v3_g, v3_b, "scoped_wing_a_plus_emi_wing_b", 2.95, (0.0, 0.0), 0.08, "continual_history_guard_fanout"
                elif ev == "M":
                    g_use, b_use, cm, ys_e, vp_e, gl_f, cat = v3_g, v3_b, "sparse_spot_lab0", 1.50, (0.24, 0.16), 0.08, "continual_history_guard_fanout"
                else:
                    raise ValueError(ev)

                tk = _build_m95_alpha_task(
                    task_id=tid,
                    short_name=sn,
                    wave_label=f"{wid}_step_{idx:02d}",
                    instruction_text=(
                        f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(dataset: dict) -> dict:`\n\n"
                        + cal_hdr
                        + "2. UNIFIED VISCOELASTIC / THRESHOLDED VISCOPLASTIC CONSTITUTIVE CONTRACT (`sigma_y = dataset['yield_stress']`):\n"
                        "   - Calibrate physical `past_stress` and `stress_hold_0` across Labs `0..3` (reconciling `calibration_load_cells` with\n"
                        "     `/tmp/hegel_continual_archive/` when in-batch sweeps are sparse or contain EMI glitches on a wing!).\n"
                        "   - Below `sigma_y` (`|past_stress| <= sigma_y` and `|stress_hold_0| <= sigma_y`), the 2-branch linear Prony law holds with `viscoplastic_weights = [0.0, 0.0]`.\n"
                        "   - Above `sigma_y`, quadratic hold overstress `ov_hold = (maximum(0, |stress_hold_0| - sigma_y)^2)*sign(stress_hold_0)` adds `+ theta_7 * ov_hold * t_k`,\n"
                        "     and plastic-work history `D_hist = sum((maximum(0, |past_stress| - sigma_y)^2)*past_dt, axis=1)` adds `+ theta_8 * (D_hist * stress_hold_0)*(1 - exp(-t_k/tau_1))`\n"
                        "     (and matching `t=0+` terms in `instant_rate_obs`).\n"
                        "3. RETURN unified dict `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'viscoplastic_weights': (2,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
                        f"- Public dataset: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                    ),
                    pub_call_expr=(
                        f"_gen_m95_alpha_cohort({sd_pub}, ({t1_e}, {t2_e}), {th_e!r}, stress_gains={g_use!r}, stress_biases={b_use!r}, "
                        f'cal_mode="{cm}", yield_stress={ys_e}, vp_theta={vp_e!r}, glitch_frac={gl_f})'
                    ),
                    priv_cases_py=(
                        f"    ({sd_p1}, *_gen_m95_alpha_cohort({sd_p1}, (0.50, 3.60), {th_p1!r}, stress_gains={g_use!r}, stress_biases={b_use!r}, cal_mode=\"{cm}\", yield_stress={ys_e}, vp_theta={vp_e!r}, glitch_frac={gl_f})),\n"
                        f"    ({sd_p2}, *_gen_m95_alpha_cohort({sd_p2}, (0.68, 4.45), {th_p2!r}, stress_gains={g_use!r}, stress_biases={b_use!r}, cal_mode=\"{cm}\", yield_stress={ys_e}, vp_theta={vp_e!r}, glitch_frac={gl_f})),"
                    ),
                )
            stream_tasks.append(tk)
            manifest.append({"step_index": idx, "task_id": tid, "short_name": sn, "causal_event": ev, "event_category": cat})
        return src_task, tuple(stream_tasks), manifest

    # Beta family
    cal_hdr_b = (
        "1. MULTI-WING SENSOR-BANK CALIBRATION (`Wing A = banks 0,1`, `Wing B = banks 2,3`) & 1D FVM STENCIL:\n"
        "   - `study['sensor_calibration_sweeps']` gives in-batch reference sweeps `{'bank_id', 'ref_signal', 'raw_readout'}` obeying\n"
        "     `raw_readout = bank_gain[b] * ref_signal + bank_offset[b]` (`b in 0..3`; note: some continual batches carry only sparse\n"
        "     spot checks or occasional EMI spikes on a wing, so reconcile with `/tmp/hegel_continual_archive/` when needed!).\n"
        "   - Invert via `row_bank_id`: `y_obs = (y_obs_raw - bank_offset[row_bank_id]) / bank_gain[row_bank_id]`.\n"
        "   - 1D FVM operator `_build_1d(L, n, kappa, vel, robin, diag_add)`: harmonic interior diffusion `(2*k_i*k_{i+1}/(k_i+k_{i+1}))/h^2`,\n"
        "     Robin boundary `(rg*k_b)/(h*(rg*0.5*h+k_b))` at `0, n-1`, and first-order upwind advection.\n"
    )
    src_sn = f"{wid}_s00"
    src_task = _build_m95_beta_task(
        task_id=f"{wid}_s00__tracer_tomography_baseline",
        short_name=src_sn,
        wave_label=f"{wid}_source",
        instruction_text=(
            f"Implement `/holdout_task/{src_sn}_solver.py` exposing `def solve_{src_sn}(study: dict) -> dict:`\n\n"
            + cal_hdr_b
            + "2. LAMINAR SEPARABLE MULTI-SHIFT REACTION-TRANSPORT & BOX-SIMPLEX KKT (`40 x 36`, `M = 12`, `K_mu = 8`):\n"
            "   - In this laminar separable survey (`V_reaction_2d(x, y) = v_x(x) + v_y(y)`), solve `(A_x + diag(v_x)) U_{m,k} + U_{m,k} (B_y + diag(v_y)).T + mu_shifts[k]*U_{m,k} = source_modes[m]`\n"
            "     (using `1.0 * kappa_x`, `1.0 * kappa_y`, `v_x = mean(V_reaction_2d, axis=1)`, `v_y = mean(V_reaction_2d, axis=0) - mean(V_reaction_2d)`),\n"
            "     profile `Z_nuisance` (`nuisance_ridge`), and minimize `0.5*||S_tilde @ w - r_y||_2^2 + 0.5*source_ridge*||w||_2^2` on `{0 <= w <= u_box, sum(w) == total_budget}`.\n"
            "   - Return unified dict `{'weights': (M,), 'terminal_field': sum_m w_m * U_{m, -1} (nx, ny), 'kkt_certificate': {'weights', 'nu_sum', 'lam_lower', 'lam_upper', 'primal_objective', 'dual_lower_bound'}}`.\n"
            f"- Public study: `/holdout_task/public_{src_sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        pub_call_expr=f"_gen_m95_beta_study({spec.world_seed + 1})",
        priv_cases_py=(
            f"    ({spec.world_seed + 501}, *_gen_m95_beta_study({spec.world_seed + 501})),\n"
            f"    ({spec.world_seed + 502}, *_gen_m95_beta_study({spec.world_seed + 502})),"
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
            tk = _build_m95_beta_task(
                task_id=tid,
                short_name=sn,
                wave_label=f"{wid}_step_{idx:02d}",
                is_multibudget_T=True,
                instruction_text=(
                    f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(study: dict) -> dict:`\n\n"
                    + cal_hdr_b
                    + "2. ANISOTROPIC LAMINAR GRID (`44 x 36`, `M = 14`, `K_mu = 10`) & 3-BUDGET KKT PARETO (`study['budget_grid']`):\n"
                    "   - Solve the laminar separable reaction-transport system (`V_reaction_2d(x,y) = v_x(x) + v_y(y)`, `1.0 * kappa`), profile `Z_nuisance`,\n"
                    "     and return `{'weights_by_budget': (3, M), 'terminal_field_mid_budget': (nx, ny) for b=1, 'kkt_certificates': [cert_0, cert_1, cert_2]}`.\n"
                    f"- Public study: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                ),
                pub_call_expr=f"_gen_m95_beta_study({sd_pub}, nx=44, ny=36, K_mu=10, M=14, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)",
                priv_cases_py=(
                    f"    ({sd_p1}, *_gen_m95_beta_study({sd_p1}, nx=44, ny=36, K_mu=10, M=14, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),\n"
                    f"    ({sd_p2}, *_gen_m95_beta_study({sd_p2}, nx=44, ny=36, K_mu=10, M=14, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)),"
                ),
            )
            cat = "adaptation_discovery"
        else:
            if ev == "N":
                g_use, b_use, cm, vx_f, ns_sc, cat = v1_g, v1_b, "full_48", False, 0.022, "immediate_cal_negative_control"
            elif ev == "G1":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "full_48", False, 0.011, "immediate_cal_negative_control"
            elif ev == "C":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "full_48", True, 0.011, "adaptation_discovery"
            elif ev == "R":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "sparse_spot_bank0", False, 0.011, "continual_history_guard_fanout"
            elif ev == "C_prime":
                g_use, b_use, cm, vx_f, ns_sc, cat = v2_g, v2_b, "sparse_spot_bank0", True, 0.011, "continual_history_guard_fanout"
            elif ev == "G2":
                g_use, b_use, cm, vx_f, ns_sc, cat = v3_g, v3_b, "scoped_wing_a_plus_emi_wing_b", False, 0.011, "continual_history_guard_fanout"
            elif ev == "M":
                g_use, b_use, cm, vx_f, ns_sc, cat = v3_g, v3_b, "sparse_spot_bank0", True, 0.011, "continual_history_guard_fanout"
            else:
                raise ValueError(ev)

            regime_body = (
                "2. SUPERCRITICAL VORTEX-COUPLED DISPERSION REGIME (`40 x 36`, `M = 12`, `K_mu = 8`):\n"
                "   - In this supercritical vortex survey (`V_reaction_2d(x, y)` has strong non-separable 2D cross-coupling `||V - v_x - v_y||_F / ||V||_F > 1e-3`),\n"
                "     vortex eddy mixing enhances diffusivity to `kappa_eff_x = 1.45 * kappa_x` and `kappa_eff_y = 1.45 * kappa_y`, governed by the full 2D operator\n"
                "     `(kron(A_x(1.45*kappa_x, reaction_x), I_y) + kron(I_x, B_y(1.45*kappa_y, reaction_y)) + diag(V_reaction_2d.ravel() + mu_shifts[k])) u_{m,k} = source_modes[m].ravel()` (C-order)!\n"
                "   - Profile `Z_nuisance` and return unified dict `{'weights': (M,), 'terminal_field': (nx, ny), 'kkt_certificate': dict}`.\n"
                if vx_f
                else "2. LAMINAR SEPARABLE REACTION-TRANSPORT REGIME (`40 x 36`, `M = 12`, `K_mu = 8`):\n"
                "   - Inspect `V_reaction_2d(x, y)`: in this laminar separable survey (`||V - v_x - v_y||_F / ||V||_F <= 1e-6`), diffusivity is nominal `1.00 * kappa_x`, `1.00 * kappa_y`\n"
                "     and `(A_x + diag(v_x)) U_{m,k} + U_{m,k} (B_y + diag(v_y)).T + mu_shifts[k]*U_{m,k} = source_modes[m]`.\n"
                "   - Profile `Z_nuisance` and return unified dict `{'weights': (M,), 'terminal_field': (nx, ny), 'kkt_certificate': dict}`.\n"
            )
            tk = _build_m95_beta_task(
                task_id=tid,
                short_name=sn,
                wave_label=f"{wid}_step_{idx:02d}",
                instruction_text=(
                    f"Implement `/holdout_task/{sn}_solver.py` exposing `def solve_{sn}(study: dict) -> dict:`\n\n"
                    + cal_hdr_b
                    + regime_body
                    + f"- Public study: `/holdout_task/public_{sn}.pkl`. Validator: `python3 /holdout_task/validate_public_schema.py`."
                ),
                pub_call_expr=(
                    f'_gen_m95_beta_study({sd_pub}, bank_gains={g_use!r}, bank_offsets={b_use!r}, cal_mode="{cm}", '
                    f"vortex_regime={vx_f!r}, obs_noise_scale={ns_sc})"
                ),
                priv_cases_py=(
                    f'    ({sd_p1}, *_gen_m95_beta_study({sd_p1}, bank_gains={g_use!r}, bank_offsets={b_use!r}, cal_mode="{cm}", vortex_regime={vx_f!r}, obs_noise_scale={ns_sc})),\n'
                    f'    ({sd_p2}, *_gen_m95_beta_study({sd_p2}, bank_gains={g_use!r}, bank_offsets={b_use!r}, cal_mode="{cm}", vortex_regime={vx_f!r}, obs_noise_scale={ns_sc})),'
                ),
            )
        stream_tasks.append(tk)
        manifest.append({"step_index": idx, "task_id": tid, "short_name": sn, "causal_event": ev, "event_category": cat})
    return src_task, tuple(stream_tasks), manifest


def m95_task_to_frontier_task(
    task: HoldoutOODTask,
    *,
    continual_context_block: str = "",
    allowed_readonly_dependencies: tuple[str, ...] = (),
) -> m6f.FrontierBenchmarkTask:
    full_inst = f"{continual_context_block.strip()}\n\n{task.instruction_text}" if continual_context_block.strip() else task.instruction_text
    pkl_fn = getattr(task, "public_pickle_filename", "") or f"public_{task.short_name}.pkl"
    pkl_asset = f"/holdout_task/{pkl_fn}"
    req_assets = tuple(dict.fromkeys([pkl_asset, "/holdout_task/validate_public_schema.py", *allowed_readonly_dependencies]))
    return m6f.FrontierBenchmarkTask(
        task_id=task.task_id,
        short_name=task.short_name,
        benchmark="Hegel-M95-Equal-Tool-Continual-World",
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
        host_only_public_gt_json=str(getattr(task, "host_only_public_gt_json", "") or ""),
    )


# ============================================================================
# 3. Wing-Scoped `M8` Continual Graph + Equal-Tool Pre-LLM Execution Service
# ============================================================================


def initialize_m95_wing_scoped_mechanism_graph(
    *,
    family: str,
    forest: Any | None = None,
) -> tuple[PersistentMechanismGraph, M8VariationalPlanner, M94BindingRegistry]:
    """Initialize Wing-A (`0,1`), Wing-B (`2,3`), Composite Facility Parameter Root, and `0.0`-Drift Acoustic Sentinel."""
    graph = PersistentMechanismGraph(forest=forest)
    planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)

    rng_s = np.random.default_rng(95001)
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
            provenance_note="Unrelated acoustic sentinel (`0.0` collateral drift across all revisions).",
        ),
        tree_id="tree_m95_sentinel",
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
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id="alpha_wing_a_calibration",
                parent_id=None,
                role="shared_upstream_premise",
                feature_names=("cmd_l0", "cmd_l1", "ind_l0", "ind_l1"),
                version=1,
                shared_weights={"cmd_l0": 1.0, "cmd_l1": 1.0, "ind_l0": 0.0, "ind_l1": 0.0},
                provenance_note="Alpha Facility Wing A (`labs 0, 1`) load-cell calibration.",
            ),
            tree_id="tree_m94_chain_alpha",
        )
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id="alpha_wing_b_calibration",
                parent_id=None,
                role="shared_upstream_premise",
                feature_names=("cmd_l2", "cmd_l3", "ind_l2", "ind_l3"),
                version=1,
                shared_weights={"cmd_l2": 1.0, "cmd_l3": 1.0, "ind_l2": 0.0, "ind_l3": 0.0},
                provenance_note="Alpha Facility Wing B (`labs 2, 3`) load-cell calibration.",
            ),
            tree_id="tree_m94_chain_alpha",
        )
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id="alpha_facility_composite_calibration",
                parent_id="alpha_wing_a_calibration",
                role="shared_upstream_premise",
                feature_names=("cmd_l0", "cmd_l1", "cmd_l2", "cmd_l3"),
                version=1,
                shared_weights={"cmd_l0": 1.0, "cmd_l1": 1.0, "cmd_l2": 1.0, "cmd_l3": 1.0},
                provenance_note="Composite `(Wing A, Wing B)` calibration epoch bound to Alpha constitutive branches.",
            ),
            tree_id="tree_m94_chain_alpha",
        )
        reg = M94BindingRegistry(graph)
        reg.update_shared_parameter_epoch(
            parameter_mechanism_id="alpha_facility_composite_calibration",
            new_version=1,
            parameter_payload={"stress_gains": [1.0, 1.0, 1.0, 1.0], "stress_biases": [0.0, 0.0, 0.0, 0.0], "wing_versions": {"wing_a": 1, "wing_b": 1}},
            affected_chain_id="alpha",
            evidence_note="initial_stage_S_nominal",
        )
    else:
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id="beta_wing_a_calibration",
                parent_id=None,
                role="shared_upstream_premise",
                feature_names=("ref_b0", "ref_b1", "ind_b0", "ind_b1"),
                version=1,
                shared_weights={"ref_b0": 1.0, "ref_b1": 1.0, "ind_b0": 0.0, "ind_b1": 0.0},
                provenance_note="Beta Sensor Wing A (`banks 0, 1`) calibration.",
            ),
            tree_id="tree_m94_chain_beta",
        )
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id="beta_wing_b_calibration",
                parent_id=None,
                role="shared_upstream_premise",
                feature_names=("ref_b2", "ref_b3", "ind_b2", "ind_b3"),
                version=1,
                shared_weights={"ref_b2": 1.0, "ref_b3": 1.0, "ind_b2": 0.0, "ind_b3": 0.0},
                provenance_note="Beta Sensor Wing B (`banks 2, 3`) calibration.",
            ),
            tree_id="tree_m94_chain_beta",
        )
        graph.register_mechanism(
            TypedMechanism(
                mechanism_id="beta_facility_composite_calibration",
                parent_id="beta_wing_a_calibration",
                role="shared_upstream_premise",
                feature_names=("ref_b0", "ref_b1", "ref_b2", "ref_b3"),
                version=1,
                shared_weights={"ref_b0": 1.0, "ref_b1": 1.0, "ref_b2": 1.0, "ref_b3": 1.0},
                provenance_note="Composite `(Wing A, Wing B)` calibration epoch bound to Beta transport branches.",
            ),
            tree_id="tree_m94_chain_beta",
        )
        reg = M94BindingRegistry(graph)
        reg.update_shared_parameter_epoch(
            parameter_mechanism_id="beta_facility_composite_calibration",
            new_version=1,
            parameter_payload={"bank_gains": [1.0, 1.0, 1.0, 1.0], "bank_offsets": [0.0, 0.0, 0.0, 0.0], "wing_versions": {"wing_a": 1, "wing_b": 1}},
            affected_chain_id="beta",
            evidence_note="initial_stage_S_nominal",
        )
    return graph, planner, reg


def run_m95_wing_scoped_calibration_update(
    *,
    family: str,
    episode_id: str,
    public_item: dict[str, Any],
    graph: PersistentMechanismGraph,
    planner: M8VariationalPlanner,
    registry: M94BindingRegistry,
) -> dict[str, Any]:
    """Uniform task-ID-blind Wing-A & Wing-B `M8VariationalPlanner` check (`AncestorReweight` vs `NoChange`).

    - Evaluates Wing A (`0,1`) and Wing B (`2,3`) independently whenever `n_wing >= 6`.
    - Rejects unvalidated EMI-corrupted spot reweights on unaffected sibling wings (`NoChange` -> `0.0` sibling drift!).
    - Propagates accepted wing updates across ALL dependent constitutive branches (`fanout = 2` updating `{C0, C1}` simultaneously!).
    """
    sentinel_tm = graph.active_task_models["sentinel_acoustic_calibration"]
    sentinel_va = graph.task_datasets["sentinel_acoustic_calibration"]["val"]
    sentinel_before = np.asarray(evaluate_committed_task_model(sentinel_tm, graph, sentinel_va)["predictions"], dtype=float)

    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        ch_ids = np.asarray(cal["lab_id"], dtype=int).ravel()
        x_in = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        y_out = np.asarray(cal["true_stress"], dtype=float).ravel()
        wing_specs = [
            ("wing_a", "alpha_wing_a_calibration", (0, 1), ("cmd_l0", "cmd_l1", "ind_l0", "ind_l1")),
            ("wing_b", "alpha_wing_b_calibration", (2, 3), ("cmd_l2", "cmd_l3", "ind_l2", "ind_l3")),
        ]
        comp_id = "alpha_facility_composite_calibration"
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        ch_ids = np.asarray(cal["bank_id"], dtype=int).ravel()
        x_in = np.asarray(cal["ref_signal"], dtype=float).ravel()
        y_out = np.asarray(cal["raw_readout"], dtype=float).ravel()
        wing_specs = [
            ("wing_a", "beta_wing_a_calibration", (0, 1), ("ref_b0", "ref_b1", "ind_b0", "ind_b1")),
            ("wing_b", "beta_wing_b_calibration", (2, 3), ("ref_b2", "ref_b3", "ind_b2", "ind_b3")),
        ]
        comp_id = "beta_facility_composite_calibration"

    wing_receipts: dict[str, dict[str, Any]] = {}
    any_reweighted = False
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
        # Contiguous two-thirds / one-third split preserves temporal check ordering (catching transient initial EMI spikes!)
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
        if act == "AncestorReweight":
            any_reweighted = True
        w_after = graph.mechanisms[mech_id].shared_weights
        w_drift = max(abs(float(w_after[k]) - float(w_before[k])) for k in cols)
        wing_receipts[w_tag] = {
            "mechanism_id": mech_id,
            "n_rows": n_w,
            "selected_action": act,
            "val_log_predictive_gain": round(float(rev["selected_candidate"]["val_log_predictive_gain"]), 4),
            "version": int(graph.mechanisms[mech_id].version),
            "max_weight_drift": round(w_drift, 8),
        }

    if family == "alpha":
        wa = graph.mechanisms["alpha_wing_a_calibration"].shared_weights
        wb = graph.mechanisms["alpha_wing_b_calibration"].shared_weights
        payload: dict[str, Any] = {
            "stress_gains": [float(wa["cmd_l0"]), float(wa["cmd_l1"]), float(wb["cmd_l2"]), float(wb["cmd_l3"])],
            "stress_biases": [float(wa["ind_l0"]), float(wa["ind_l1"]), float(wb["ind_l2"]), float(wb["ind_l3"])],
            "wing_versions": {
                "wing_a": int(graph.mechanisms["alpha_wing_a_calibration"].version),
                "wing_b": int(graph.mechanisms["alpha_wing_b_calibration"].version),
            },
        }
    else:
        wa = graph.mechanisms["beta_wing_a_calibration"].shared_weights
        wb = graph.mechanisms["beta_wing_b_calibration"].shared_weights
        payload = {
            "bank_gains": [float(wa["ref_b0"]), float(wa["ref_b1"]), float(wb["ref_b2"]), float(wb["ref_b3"])],
            "bank_offsets": [float(wa["ind_b0"]), float(wa["ind_b1"]), float(wb["ind_b2"]), float(wb["ind_b3"])],
            "wing_versions": {
                "wing_a": int(graph.mechanisms["beta_wing_a_calibration"].version),
                "wing_b": int(graph.mechanisms["beta_wing_b_calibration"].version),
            },
        }

    updated_bids: list[str] = []
    if any_reweighted:
        graph.mechanisms[comp_id].version = int(graph.mechanisms[comp_id].version) + 1
        updated_bids = registry.update_shared_parameter_epoch(
            parameter_mechanism_id=comp_id,
            new_version=int(graph.mechanisms[comp_id].version),
            parameter_payload=payload,
            affected_chain_id=family,
            evidence_note=f"wing_scoped_reweight_on_{episode_id}",
        )
    else:
        registry.parameter_epochs[comp_id] = {
            "parameter_mechanism_id": comp_id,
            "version": int(graph.mechanisms[comp_id].version),
            "payload": payload,
            "evidence_note": f"retained_on_{episode_id}",
        }
        registry.sync_to_graph()

    sentinel_after = np.asarray(evaluate_committed_task_model(sentinel_tm, graph, sentinel_va)["predictions"], dtype=float)
    sentinel_drift = float(np.max(np.abs(sentinel_after - sentinel_before)))
    return {
        "family": family,
        "episode_id": episode_id,
        "composite_mechanism_id": comp_id,
        "composite_version": int(graph.mechanisms[comp_id].version),
        "any_wing_reweighted": any_reweighted,
        "wing_receipts": wing_receipts,
        "parameter_payload": payload,
        "updated_binding_ids": updated_bids,
        "fanout_count": len(updated_bids),
        "sentinel_max_abs_prediction_drift": sentinel_drift,
    }


def update_flat_parameter_snapshot_from_public_item(
    *,
    family: str,
    public_item: dict[str, Any],
    prior_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Strong `Flat_ParameterizedCache` parameter updater (`A03`):
    - Retains prior per-channel `(gain, offset)` whenever a channel has `< 2` rows in the current batch (so sparse `R / C' / M`
      batches preserve prior channels cleanly!).
    - Refits any channel with `>= 2` rows via standard flat `np.linalg.lstsq` (without `M8` variational validation gating).
    """
    snap = deepcopy(prior_snapshot)
    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        ids = np.asarray(cal["lab_id"], dtype=int).ravel()
        xs = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        ys = np.asarray(cal["true_stress"], dtype=float).ravel()
        g_out = list(snap["stress_gains"])
        b_out = list(snap["stress_biases"])
        for c in range(4):
            m = ids == c
            if int(np.sum(m)) >= 2:
                gn, bs = np.linalg.lstsq(np.column_stack([xs[m], np.ones(int(np.sum(m)))]), ys[m], rcond=None)[0]
                g_out[c] = float(gn)
                b_out[c] = float(bs)
        return {"stress_gains": g_out, "stress_biases": b_out}

    cal = public_item["study"]["sensor_calibration_sweeps"]
    ids = np.asarray(cal["bank_id"], dtype=int).ravel()
    xs = np.asarray(cal["ref_signal"], dtype=float).ravel()
    ys = np.asarray(cal["raw_readout"], dtype=float).ravel()
    g_out = list(snap["bank_gains"])
    b_out = list(snap["bank_offsets"])
    for c in range(4):
        m = ids == c
        if int(np.sum(m)) >= 2:
            gn, bs = np.linalg.lstsq(np.column_stack([xs[m], np.ones(int(np.sum(m)))]), ys[m], rcond=None)[0]
            g_out[c] = float(gn)
            b_out[c] = float(bs)
    return {"bank_gains": g_out, "bank_offsets": b_out}


def update_flat_validated_wing_parameter_snapshot_from_public_item(
    *,
    family: str,
    public_item: dict[str, Any],
    prior_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """`Flat_Multiversion_Validated` per-wing `2/3–1/3` holdout-MSE OLS updater (`M9.5.1` `F03` — zero `M8` graph/planner!).
    - Evaluates Wing A (`0, 1`) and Wing B (`2, 3`) independently when `n_wing >= 6`.
    - Fits OLS on contiguous `[:2*n_w//3]` and accepts the wing update iff trailing `[2*n_w//3:]` validation MSE strictly improves
      over retaining `prior_snapshot` (`mse_new < 0.5 * mse_prior` and `mse_new <= 0.05`).
    """
    snap = deepcopy(prior_snapshot)
    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        ids = np.asarray(cal["lab_id"], dtype=int).ravel()
        xs = np.asarray(cal["cmd_stress"], dtype=float).ravel()
        ys = np.asarray(cal["true_stress"], dtype=float).ravel()
        g_cur = list(snap["stress_gains"])
        b_cur = list(snap["stress_biases"])
        key_g, key_b = "stress_gains", "stress_biases"
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        ids = np.asarray(cal["bank_id"], dtype=int).ravel()
        xs = np.asarray(cal["ref_signal"], dtype=float).ravel()
        ys = np.asarray(cal["raw_readout"], dtype=float).ravel()
        g_cur = list(snap["bank_gains"])
        b_cur = list(snap["bank_offsets"])
        key_g, key_b = "bank_gains", "bank_offsets"

    wing_decisions: dict[str, dict[str, Any]] = {}
    for w_tag, (c0, c1) in (("wing_a", (0, 1)), ("wing_b", (2, 3))):
        m_w = (ids == c0) | (ids == c1)
        n_w = int(np.sum(m_w))
        if n_w < 6:
            wing_decisions[w_tag] = {"n_rows": n_w, "accepted": False, "reason": "sparse_retained"}
            continue
        id_w = ids[m_w]
        x_w = xs[m_w]
        y_w = ys[m_w]
        n_tr = max(4, (2 * n_w) // 3)
        id_tr, x_tr, y_tr = id_w[:n_tr], x_w[:n_tr], y_w[:n_tr]
        id_va, x_va, y_va = id_w[n_tr:], x_w[n_tr:], y_w[n_tr:]
        cand_g = {c0: g_cur[c0], c1: g_cur[c1]}
        cand_b = {c0: b_cur[c0], c1: b_cur[c1]}
        for ch in (c0, c1):
            mc = id_tr == ch
            if int(np.sum(mc)) >= 2:
                gn, bs = np.linalg.lstsq(np.column_stack([x_tr[mc], np.ones(int(np.sum(mc)))]), y_tr[mc], rcond=None)[0]
                cand_g[ch], cand_b[ch] = float(gn), float(bs)
        pred_prior = np.array([g_cur[ch] * xv + b_cur[ch] for ch, xv in zip(id_va, x_va)], dtype=float)
        pred_new = np.array([cand_g[ch] * xv + cand_b[ch] for ch, xv in zip(id_va, x_va)], dtype=float)
        mse_prior = float(np.mean((y_va - pred_prior) ** 2))
        mse_new = float(np.mean((y_va - pred_new) ** 2))
        accept_w = bool(mse_new + 1e-9 < mse_prior and mse_new <= 0.05)
        if accept_w:
            g_cur[c0], g_cur[c1] = cand_g[c0], cand_g[c1]
            b_cur[c0], b_cur[c1] = cand_b[c0], cand_b[c1]
        wing_decisions[w_tag] = {
            "n_rows": n_w,
            "accepted": accept_w,
            "val_mse_prior": round(mse_prior, 8),
            "val_mse_new": round(mse_new, 8),
        }
    return {key_g: g_cur, key_b: b_cur, "wing_validation_decisions": wing_decisions}


def make_continual_archive_entry(
    *,
    short_name: str,
    task_id: str,
    canonical_entrypoint: str,
    canonical_algorithm_code: str,
    output_schema_keys: list[str],
    public_qualification_status: str = "public_gate_verified",
    delivered_artifact_code: str | None = None,
    delivered_entrypoint: str | None = None,
) -> dict[str, Any]:
    """Construct a dual-export continual archive entry (`M9.5.1` `F01 + F14`):
    - `canonical_algorithm_code`: ALWAYS `input_coordinate_contract == 'raw_measured'` (NEVER pre-calibrated adapter code!).
    - `delivered_artifact_code`: The rendered episode deliverable (which may wrap `canonical_algorithm_code` 1-hop).
    - `public_qualification_status`: `'public_gate_verified'` or `'public_gate_failed'` (NEVER post-hoc private verifier status!).
    """
    if ADAPTER_PRECALIBRATED_SENTINEL in canonical_algorithm_code:
        raise ValueError(
            f"Refusing to store pre-calibrated adapter deliverable as `canonical_algorithm_code` on `{short_name}` (`F01` invariant)!"
        )
    if public_qualification_status not in {"public_gate_verified", "public_gate_failed"}:
        raise ValueError(f"Invalid public_qualification_status `{public_qualification_status}` (`F14` — post-hoc private status forbidden!)")
    deliv_code = delivered_artifact_code if delivered_artifact_code is not None else canonical_algorithm_code
    deliv_ep = delivered_entrypoint if delivered_entrypoint is not None else canonical_entrypoint
    return {
        "short_name": short_name,
        "task_id": task_id,
        "entrypoint": canonical_entrypoint,
        "canonical_entrypoint": canonical_entrypoint,
        "canonical_algorithm_code": canonical_algorithm_code,
        "canonical_sha256": hashlib.sha256(canonical_algorithm_code.encode("utf-8")).hexdigest(),
        "input_coordinate_contract": "raw_measured",
        "submitted_code": canonical_algorithm_code,
        "delivered_entrypoint": deliv_ep,
        "delivered_artifact_code": deliv_code,
        "delivered_sha256": hashlib.sha256(deliv_code.encode("utf-8")).hexdigest(),
        "output_schema_keys": list(output_schema_keys),
        "public_qualification_status": public_qualification_status,
    }


def render_parameterized_input_adapter(
    *,
    family: str,
    source_code: str,
    source_entrypoint: str,
    target_entrypoint: str,
    param_payload: dict[str, Any] | None,
    binding: MechanismExecutableBinding | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
) -> str:
    """Shared Equal-Tool Input Adapter (`0` PDE/constitutive solver lines!, `M9.5.1` `F01` 1-Hop Canonical Export):
    - Fail-closed rejects double-wrapping (`ADAPTER_PRECALIBRATED_SENTINEL in source_code` -> `ValueError`!).
    - If `param_payload is None` (`Archive_AutoReplay`), emits a pure 2-line entrypoint alias (`target_entrypoint = source_entrypoint`).
    - If `param_payload is not None`, applies ONLY the 1-hop affine sensor/load-cell calibration onto `canonical_algorithm_code`
      and stamps `_HEGEL_ADAPTER_PRECALIBRATED_DO_NOT_REWRAP = True`.
    """
    import base64

    if binding is not None:
        verify_binding_identity_and_foreign_keys(binding, graph=graph, registry=registry, require_closure_code=True)
    if param_payload is not None and ADAPTER_PRECALIBRATED_SENTINEL in source_code:
        raise ValueError(
            "Refusing to wrap an already-adapted deliverable (`_HEGEL_ADAPTER_PRECALIBRATED_DO_NOT_REWRAP` detected — `F01` double-wrap forbidden)!"
        )

    b64_src = base64.b64encode(source_code.encode("utf-8")).decode("ascii")
    src_sha = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    if param_payload is None:
        return (
            f'"""M9.5 Stateless Archive Auto-Replay Alias (`source_sha256={src_sha[:16]}`)."""\n'
            "from __future__ import annotations\n"
            "import base64, types\n\n"
            f"_SRC_B64 = {b64_src!r}\n"
            f"_SRC_EP = {source_entrypoint!r}\n"
            "_mod = types.ModuleType('_hegel_m95_stateless_replay')\n"
            "exec(compile(base64.b64decode(_SRC_B64).decode('utf-8'), '<stateless_replay>', 'exec'), _mod.__dict__)\n"
            f"{target_entrypoint} = getattr(_mod, _SRC_EP)\n"
        )

    if family == "alpha":
        gains_lit = [round(float(x), 8) for x in param_payload["stress_gains"]]
        biases_lit = [round(float(x), 8) for x in param_payload["stress_biases"]]
        return (
            f'"""M9.5.1 Canonical 1-Hop Calibration Input Adapter (`canonical_sha256={src_sha[:16]}`)."""\n'
            "from __future__ import annotations\n"
            "import base64, copy, types\n"
            "import numpy as np\n\n"
            f"{ADAPTER_PRECALIBRATED_SENTINEL} = True\n"
            f"_SRC_B64 = {b64_src!r}\n"
            f"_SRC_EP = {source_entrypoint!r}\n"
            f"_GAINS = np.array({gains_lit!r}, dtype=float)\n"
            f"_BIASES = np.array({biases_lit!r}, dtype=float)\n"
            "_mod = types.ModuleType('_hegel_m95_alpha_bound')\n"
            "exec(compile(base64.b64decode(_SRC_B64).decode('utf-8'), '<alpha_bound>', 'exec'), _mod.__dict__)\n"
            "_fn = getattr(_mod, _SRC_EP)\n\n"
            f"def {target_entrypoint}(dataset: dict) -> dict:\n"
            "    ds2 = copy.deepcopy(dataset)\n"
            "    for sp_k in ('train', 'test'):\n"
            "        sp = dict(ds2[sp_k])\n"
            "        lb = np.asarray(sp['lab_id'], dtype=int).ravel()\n"
            "        sp['past_stress_cmd'] = np.asarray(sp['past_stress_cmd'], dtype=float) * _GAINS[lb, None] + _BIASES[lb, None]\n"
            "        sp['stress_hold_cmd'] = np.asarray(sp['stress_hold_cmd'], dtype=float) * _GAINS[lb] + _BIASES[lb]\n"
            "        ds2[sp_k] = sp\n"
            "    ref_cmd = np.tile(np.array([-1.5, -0.5, 0.5, 1.5], dtype=float), 4)\n"
            "    ds2['calibration_load_cells'] = {\n"
            "        'lab_id': np.repeat(np.arange(4, dtype=int), 4),\n"
            "        'cmd_stress': ref_cmd,\n"
            "        'true_stress': ref_cmd.copy(),\n"
            "    }\n"
            "    return _fn(ds2)\n"
        )

    gains_b = [round(float(x), 8) for x in param_payload["bank_gains"]]
    offs_b = [round(float(x), 8) for x in param_payload["bank_offsets"]]
    return (
        f'"""M9.5.1 Canonical 1-Hop Calibration Input Adapter (`canonical_sha256={src_sha[:16]}`)."""\n'
        "from __future__ import annotations\n"
        "import base64, copy, types\n"
        "import numpy as np\n\n"
        f"{ADAPTER_PRECALIBRATED_SENTINEL} = True\n"
        f"_SRC_B64 = {b64_src!r}\n"
        f"_SRC_EP = {source_entrypoint!r}\n"
        f"_GAINS = np.array({gains_b!r}, dtype=float)\n"
        f"_OFFS = np.array({offs_b!r}, dtype=float)\n"
        "_mod = types.ModuleType('_hegel_m95_beta_bound')\n"
        "exec(compile(base64.b64decode(_SRC_B64).decode('utf-8'), '<beta_bound>', 'exec'), _mod.__dict__)\n"
        "_fn = getattr(_mod, _SRC_EP)\n\n"
        f"def {target_entrypoint}(study: dict) -> dict:\n"
        "    st2 = copy.deepcopy(study)\n"
        "    raw_y = np.asarray(st2['y_obs_raw'], dtype=float).ravel()\n"
        "    rb = np.asarray(st2['row_bank_id'], dtype=int).ravel()\n"
        "    st2['y_obs_raw'] = (raw_y - _OFFS[rb]) / _GAINS[rb]\n"
        "    ref_sig = np.tile(np.array([-1.0, -0.3, 0.4, 1.1], dtype=float), 4)\n"
        "    st2['sensor_calibration_sweeps'] = {\n"
        "        'bank_id': np.repeat(np.arange(4, dtype=int), 4),\n"
        "        'ref_signal': ref_sig,\n"
        "        'raw_readout': ref_sig.copy(),\n"
        "    }\n"
        "    return _fn(st2)\n"
    )


def extract_m95_guard_observables(family: str, public_item: dict[str, Any], param_payload: dict[str, Any]) -> dict[str, Any]:
    if family == "alpha":
        ds = public_item["dataset"]
        gains = np.asarray(param_payload["stress_gains"], dtype=float)
        biases = np.asarray(param_payload["stress_biases"], dtype=float)
        te_l = np.asarray(ds["test"]["lab_id"], dtype=int)
        tr_l = np.asarray(ds["train"]["lab_id"], dtype=int)
        cal_past_te = np.asarray(ds["test"]["past_stress_cmd"], dtype=float) * gains[te_l, None] + biases[te_l, None]
        cal_hold_te = np.asarray(ds["test"]["stress_hold_cmd"], dtype=float) * gains[te_l] + biases[te_l]
        cal_past_tr = np.asarray(ds["train"]["past_stress_cmd"], dtype=float) * gains[tr_l, None] + biases[tr_l, None]
        cal_hold_tr = np.asarray(ds["train"]["stress_hold_cmd"], dtype=float) * gains[tr_l] + biases[tr_l]
        obs: dict[str, Any] = {
            "past_stress": np.vstack([cal_past_tr, cal_past_te]),
            "stress_hold_0": np.concatenate([cal_hold_tr, cal_hold_te]),
            "train": {"past_stress": cal_past_tr, "stress_hold_0": cal_hold_tr},
            "test": {"past_stress": cal_past_te, "stress_hold_0": cal_hold_te},
            "eval_times": np.asarray(ds["eval_times"], dtype=float),
        }
        if "yield_stress" in ds:
            obs["yield_stress"] = float(ds["yield_stress"])
        if "residual_plastic_state" in ds.get("test", {}):
            obs["residual_plastic_state"] = np.asarray(ds["test"]["residual_plastic_state"], dtype=float)
        return obs
    st = public_item["study"]
    return {"V_pocket_channel_b": np.asarray(st["V_reaction_2d"], dtype=float)}


def expected_public_output_keys(task: HoldoutOODTask) -> list[str]:
    val_txt = task.public_setup_py.split("validate_public_schema.py", 1)[-1]
    known = [
        "estimated_taus", "hereditary_weights", "viscoplastic_weights", "predicted_strain", "initial_rate_pred",
        "secant_compliance", "hazard_crossing_times",
        "weights", "terminal_field", "kkt_certificate",
        "weights_by_budget", "terminal_field_mid_budget", "kkt_certificates",
    ]
    return sorted([k for k in known if f'"{k}"' in val_txt])


def execute_single_pre_llm_public_gate_probe_in_container(
    *,
    task: HoldoutOODTask,
    candidate_code: str,
    readonly_dependencies: dict[str, str] | None = None,
    expected_dependency_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Identical air-gapped public schema + pure-JSON isolated-worker `ScientificProbeReceipt` gate (`M9.5.1` `F02 + F06`)!
    - Stages ONLY prefix-locked `readonly_dependencies` (`0444`, SHA-256 verified) and masks unmanifested `/tmp/hegel_continual_archive/*`!
    """
    import shlex

    ro_snip, ro_shas = validate_and_stage_readonly_dependencies_setup_py(
        readonly_dependencies, expected_dependency_sha256=expected_dependency_sha256
    )
    allowed_ro_paths = tuple(sorted(ro_shas.keys()))
    ft = m95_task_to_frontier_task(task, allowed_readonly_dependencies=allowed_ro_paths)
    art_path = task.declared_artifacts[0]
    cid = m6f.start_detached_container(CLEAN_SCIPY_SANDBOX_IMAGE, memory_limit="2048m", cpus="2", network_mode="none")
    try:
        with tempfile.TemporaryDirectory(prefix="m95_probe_") as td:
            td_p = Path(td)
            (td_p / "_setup.py").write_text(task.public_setup_py + ro_snip, encoding="utf-8")
            (td_p / "cand.py").write_text(candidate_code, encoding="utf-8")
            m6f.exec_container_bash(cid, "mkdir -p /holdout_task", timeout_sec=10)
            subprocess.run(["docker", "cp", str(td_p / "_setup.py"), f"{cid}:/tmp/_setup.py"], check=True, timeout=15)
            setup_r = m6f.exec_container_bash(
                cid,
                "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -I /tmp/_setup.py && rm -f /tmp/_setup.py",
                timeout_sec=120,
            )
            setup_timed_out = bool(setup_r.get("timed_out")) or setup_r["exit_code"] == 124 or "[TIMEOUT:" in setup_r.get("stderr", "")
            setup_ok = bool(setup_r["exit_code"] == 0 and not setup_timed_out)
            # Clean any unmanifested files under `/tmp/hegel_continual_archive/` BEFORE running schema validator OR scientific probe (`F02`)!
            clean_unman_py = (
                "import json, os\n"
                f"allowed = set({json.dumps(list(allowed_ro_paths))})\n"
                "d = '/tmp/hegel_continual_archive'\n"
                "if os.path.isdir(d):\n"
                "    for fn in list(os.listdir(d)):\n"
                "        fp = os.path.join(d, fn)\n"
                "        if fp not in allowed and os.path.isfile(fp):\n"
                "            os.chmod(fp, 0o644); os.unlink(fp)\n"
            )
            m6f.exec_container_bash(cid, "python3 -I -c " + shlex.quote(clean_unman_py), timeout_sec=15)
            pub_shas_py = (
                "import hashlib, json, os\n"
                f"ps = {json.dumps(list(ft.required_public_assets))}\n"
                "print(json.dumps({p: hashlib.sha256(open(p,'rb').read()).hexdigest() for p in ps if os.path.isfile(p)}, sort_keys=True))\n"
            )
            sh_r = m6f.exec_container_bash(cid, "python3 -I -c " + shlex.quote(pub_shas_py), timeout_sec=15)
            pub_lines = [ln for ln in sh_r["stdout"].strip().splitlines() if ln.strip()]
            pub_shas = json.loads(pub_lines[-1]) if (setup_ok and pub_lines) else {}
            subprocess.run(["docker", "cp", str(td_p / "cand.py"), f"{cid}:{art_path}"], check=True, timeout=15)

        cand_bundle_sha = m6f.compute_declared_artifacts_digest(cid, ft.declared_artifacts)
        sch_r = (
            m6f.exec_container_bash(
                cid,
                "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I /holdout_task/validate_public_schema.py",
                timeout_sec=45,
            )
            if setup_ok
            else {"exit_code": int(setup_r["exit_code"] or 1), "stdout": "", "stderr": setup_r.get("stderr", ""), "timed_out": setup_timed_out}
        )
        sch_timed_out = bool(sch_r.get("timed_out")) or sch_r["exit_code"] == 124 or "[TIMEOUT:" in sch_r.get("stderr", "")
        sch_ok = bool(setup_ok and not sch_timed_out and sch_r["exit_code"] == 0 and '"schema_valid": true' in sch_r["stdout"].lower())
        sci_rec = (
            m6f.execute_host_trusted_scientific_probe(
                agent_cid=cid,
                task=ft,
                current_candidate_sha256=cand_bundle_sha,
                public_setup_files_sha256=pub_shas,
            )
            if sch_ok
            else None
        )
        sci_timed_out = bool(
            isinstance(sci_rec, dict)
            and (
                bool(sci_rec.get("timed_out"))
                or "[timeout:" in str(sci_rec.get("worker_stderr") or "").lower()
                or "timed out" in str(sci_rec.get("error") or "").lower()
            )
        )
        timed_out_any = bool(setup_timed_out or sch_timed_out or sci_timed_out)
        ok_all = bool(sch_ok and not timed_out_any and isinstance(sci_rec, dict) and sci_rec.get("scientific_invariants_verified") is True)
        return {
            "accepted": ok_all,
            "setup_ok": setup_ok,
            "setup_exit_code": int(setup_r["exit_code"]),
            "schema_valid": sch_ok,
            "schema_exit_code": int(sch_r["exit_code"]),
            "timed_out": timed_out_any,
            "schema_stdout": sch_r["stdout"][-240:],
            "schema_stderr": sch_r["stderr"][-320:],
            "candidate_bundle_sha256": cand_bundle_sha,
            "candidate_raw_sha256": hashlib.sha256(candidate_code.encode("utf-8")).hexdigest(),
            "readonly_dependencies_sha256": ro_shas,
            "scientific_probe_receipt": sci_rec,
        }
    finally:
        m6f.stop_container_safely(cid)


def run_equal_tool_pre_llm_trial(
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
    max_pre_llm_probes: int = 1,
) -> dict[str, Any]:
    """Execute pre-LLM candidate probe(s) (`default max_pre_llm_probes = 1`) under identical public gate rules across all arms (`M9.5.1` `F01–F03, F14`)!
    - Online auto-replay filters strictly by `public_qualification_status == 'public_gate_verified'` (`F14` — zero private oracle!).
    - Always adapts 1-hop from `canonical_algorithm_code` (`input_coordinate_contract == 'raw_measured'`, `F01` — zero double-wrapping!).
    """
    req_keys = expected_public_output_keys(task)
    target_ep = task.declared_callable_entrypoints[0]
    probe_budget = max(1, int(max_pre_llm_probes))

    if arm in {"Archive_AutoReplay", "Flat_ParameterizedCache", "Flat_Multiversion_Validated"}:
        compat_raw = [
            e
            for e in reversed(prior_archive_entries)
            if sorted(e.get("output_schema_keys") or []) == req_keys
            and e.get("public_qualification_status", "public_gate_verified") == "public_gate_verified"
        ]
        # Deduplicate distinct canonical solver versions (preserving recency order `[most_recent, older_distinct, ...]`)
        dedup_compat: list[dict[str, Any]] = []
        seen_can_shas: set[str] = set()
        for ent in compat_raw:
            can_code = str(ent.get("canonical_algorithm_code") or ent["submitted_code"])
            can_sha = hashlib.sha256(can_code.encode("utf-8")).hexdigest()
            if can_sha not in seen_can_shas:
                seen_can_shas.add(can_sha)
                dedup_compat.append(ent)

        if not dedup_compat:
            return {
                "arm": arm,
                "accepted_pre_llm": False,
                "pre_llm_probes_used": 0,
                "reason": "no_publicly_verified_schema_compatible_prior_solver",
                "required_output_keys": req_keys,
            }

        cand_list = dedup_compat[: (probe_budget if arm == "Flat_Multiversion_Validated" else 1)]
        param_arg = None if arm == "Archive_AutoReplay" else flat_param_snapshot
        probe_history: list[dict[str, Any]] = []
        for chosen_ent in cand_list:
            can_code = str(chosen_ent.get("canonical_algorithm_code") or chosen_ent["submitted_code"])
            can_ep = str(chosen_ent.get("canonical_entrypoint") or chosen_ent["entrypoint"])
            cand_code = render_parameterized_input_adapter(
                family=family,
                source_code=can_code,
                source_entrypoint=can_ep,
                target_entrypoint=target_ep,
                param_payload=param_arg,
            )
            pr = execute_single_pre_llm_public_gate_probe_in_container(
                task=task,
                candidate_code=cand_code,
                readonly_dependencies=readonly_dependencies,
                expected_dependency_sha256=expected_dependency_sha256,
            )
            probe_history.append({"short_name": chosen_ent["short_name"], "accepted": pr["accepted"], "probe_result": pr})
            if pr["accepted"]:
                return {
                    "arm": arm,
                    "accepted_pre_llm": True,
                    "pre_llm_probes_used": len(probe_history),
                    "selected_source_short_name": chosen_ent["short_name"],
                    "selected_canonical_algorithm_code": can_code,
                    "selected_canonical_entrypoint": can_ep,
                    "parameter_payload": deepcopy(param_arg),
                    "candidate_code": cand_code,
                    "probe_result": pr,
                    "probe_history": probe_history,
                    "reason": "accepted_by_shared_public_gate",
                }
        last_ent = cand_list[-1]
        last_pr = probe_history[-1]["probe_result"]
        return {
            "arm": arm,
            "accepted_pre_llm": False,
            "pre_llm_probes_used": len(probe_history),
            "selected_source_short_name": last_ent["short_name"],
            "parameter_payload": deepcopy(param_arg),
            "candidate_code": None,
            "probe_result": last_pr,
            "probe_history": probe_history,
            "reason": "rejected_by_shared_public_scientific_gate",
        }

    # Arm 4: `Full_Mechanism_Graph` (`PersistentMechanismGraph` + shuffled 3-valued `ObservablePreconditionGuard` + 1 probe)
    assert graph is not None and registry is not None
    comp_id = "alpha_facility_composite_calibration" if family == "alpha" else "beta_facility_composite_calibration"
    param_ep = registry.parameter_epochs[comp_id]
    param_payload = param_ep["payload"]
    obs_ctx = extract_m95_guard_observables(family, public_item, param_payload)

    chain_bindings = [b for b in registry.bindings.values() if b.chain_id == family]
    shuf_seed = int(hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:8], 16)
    rng_ord = np.random.default_rng(shuf_seed)
    perm_idx = rng_ord.permutation(len(chain_bindings)).tolist() if chain_bindings else []
    shuffled_bindings = [chain_bindings[i] for i in perm_idx]

    guard_evals: list[dict[str, Any]] = []
    eligible: list[MechanismExecutableBinding] = []
    for b in shuffled_bindings:
        g_res = evaluate_observable_precondition_guard(b.precondition_guard, obs_ctx)
        schema_ok = sorted(b.output_schema_keys) == req_keys
        guard_evals.append(
            {
                "binding_id": b.binding_id,
                "branch_tag": b.branch_tag,
                "parameter_version": b.parameter_version,
                "guard_result": g_res,
                "schema_compatible": schema_ok,
            }
        )
        if g_res.get("state") == "TRUE" and schema_ok:
            eligible.append(b)

    if not eligible:
        return {
            "arm": arm,
            "accepted_pre_llm": False,
            "pre_llm_probes_used": 0,
            "parameter_mechanism_id": comp_id,
            "parameter_version": int(param_ep["version"]),
            "parameter_payload": deepcopy(param_payload),
            "shuffled_candidate_order": [b.binding_id for b in shuffled_bindings],
            "guard_evaluations": guard_evals,
            "reason": (
                "interface_adaptation_required"
                if all(not ev["schema_compatible"] for ev in guard_evals)
                else "precondition_guard_rejected_all_branches"
            ),
        }

    chosen_b = eligible[0]
    can_b_code = chosen_b.canonical_algorithm_code or chosen_b.parameterized_closure_code
    can_b_ep = chosen_b.canonical_entrypoint or chosen_b.parameterized_entrypoint
    cand_code = render_parameterized_input_adapter(
        family=family,
        source_code=can_b_code,
        source_entrypoint=can_b_ep,
        target_entrypoint=target_ep,
        param_payload=param_payload,
        binding=chosen_b,
        graph=graph,
        registry=registry,
    )
    chosen_b.adapter_code_sha256 = hashlib.sha256(cand_code.encode("utf-8")).hexdigest()
    chosen_b.parameter_epoch_sha256 = hashlib.sha256(json.dumps(param_payload, sort_keys=True).encode("utf-8")).hexdigest()
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
        "selected_canonical_algorithm_code": can_b_code,
        "selected_canonical_entrypoint": can_b_ep,
        "parameter_mechanism_id": comp_id,
        "parameter_version": int(param_ep["version"]),
        "parameter_payload": deepcopy(param_payload),
        "shuffled_candidate_order": [b.binding_id for b in shuffled_bindings],
        "guard_evaluations": guard_evals,
        "candidate_code": (cand_code if pr["accepted"] else None),
        "probe_result": pr,
        "reason": ("accepted_by_shared_public_gate" if pr["accepted"] else "bound_closure_rejected_by_public_gate"),
    }


# ============================================================================
# 4. Reference Dev-Seed Solvers (`0`-LLM Unit Testing on `DEV_WORLD_SPECS` ONLY)
# ============================================================================


def build_reference_dev_solver_code(family: str, entrypoint: str, branch: str) -> str:
    """Reference `C0` (`Stage S`) and `C1` (`Stage C`) solvers used ONLY in deterministic `0`-LLM dev-world unit tests!"""
    if family == "alpha":
        vp_fit_flag = "True" if branch == "C1" else "False"
        return f'''from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def {entrypoint}(dataset: dict) -> dict:
    tr, te = dataset["train"], dataset["test"]
    past_dt = np.asarray(dataset["past_dt"], dtype=float)
    eval_t = np.asarray(dataset["eval_times"], dtype=float)
    ys = float(dataset["yield_stress"])
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
    y_c = np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]
    y_r = np.asarray(tr["instant_rate_obs"], dtype=float)
    K = y_c.shape[1]
    fit_vp = {vp_fit_flag}

    ov_h_tr = (np.maximum(0.0, np.abs(tr_h0) - ys) ** 2) * np.sign(tr_h0)
    d_hi_tr = np.sum((np.maximum(0.0, np.abs(tr_ps) - ys) ** 2) * past_dt[None, :], axis=1)

    def _prop(ps_a, t1, t2):
        taus = np.array([t1, t2], dtype=float)
        z = np.zeros((ps_a.shape[0], 2), dtype=float)
        for si, dt in enumerate(past_dt):
            dec = np.exp(-float(dt) / taus)
            z = z * dec[None, :] + (1.0 - dec[None, :]) * ps_a[:, si:si + 1]
        return z

    p_dim = 9 if fit_vp else 7
    def _fit(t1, t2, iters=7):
        z = _prop(tr_ps, t1, t2)
        gap = tr_h0[:, None] - z
        e1 = 1.0 - np.exp(-eval_t[None, :] / t1)
        e2 = 1.0 - np.exp(-eval_t[None, :] / t2)
        Dc = np.zeros((ntr * K, p_dim), dtype=float)
        Dc[:, 0] = (gap[:, 0:1] * e1).ravel()
        Dc[:, 1] = (gap[:, 1:2] * e2).ravel()
        Dc[:, 2] = np.broadcast_to(eval_t[None, :], (ntr, K)).ravel()
        for j in range(4): Dc[:, 3 + j] = (X_tr[:, j:j + 1] * eval_t[None, :]).ravel()
        Dr = [gap[:, 0] / t1, gap[:, 1] / t2, np.ones(ntr), X_tr]
        if fit_vp:
            Dc[:, 7] = (ov_h_tr[:, None] * eval_t[None, :]).ravel()
            Dc[:, 8] = ((d_hi_tr * tr_h0)[:, None] * e1).ravel()
            Dr.extend([ov_h_tr, (d_hi_tr * tr_h0) / t1])
        D_all = np.vstack([Dc, 1.8 * np.column_stack(Dr)])
        y_all = np.concatenate([y_c.ravel(), 1.8 * y_r])
        w = np.ones_like(y_all)
        th = np.zeros(p_dim, dtype=float)
        s2 = 0.045 ** 2
        for _ in range(iters):
            sw = np.sqrt(w)
            th, *_ = np.linalg.lstsq(D_all * sw[:, None], y_all * sw, rcond=None)
            res = y_all - D_all @ th
            lc = float(np.sum(np.log1p((res[:ntr * K] ** 2) / s2)) + 0.5 * np.sum((res[ntr * K:] ** 2) / (0.025 ** 2)))
            w[:ntr * K] = 1.0 / (1.0 + (res[:ntr * K] ** 2) / s2)
            w[ntr * K:] = 1.0 / (1.0 + 0.2 * (res[ntr * K:] ** 2) / (0.025 ** 2))
        return lc, th

    best = (1e30, 0.55, 3.8, np.zeros(p_dim))
    for t1 in np.geomspace(0.30, 1.25, 9):
        for t2 in np.geomspace(max(t1 * 1.9, 1.9), 6.2, 9):
            lv, th = _fit(float(t1), float(t2), iters=5)
            if lv < best[0]: best = (lv, float(t1), float(t2), th)
    ro = minimize(lambda u: _fit(float(np.exp(u[0])), float(np.exp(u[0]) + np.exp(u[1])), iters=8)[0],
                  x0=np.array([np.log(best[1]), np.log(best[2] - best[1])]), method="Nelder-Mead", options={{"maxiter": 85}})
    t1h = float(np.exp(ro.x[0])); t2h = float(t1h + np.exp(ro.x[1]))
    _, th_full = _fit(t1h, t2h, iters=12)
    hw = th_full[:7]
    vw = th_full[7:9] if fit_vp else np.zeros(2, dtype=float)
    z_te = _prop(te_ps, t1h, t2h)
    gap_te = te_h0[:, None] - z_te
    dr_te = hw[2] + X_te @ hw[3:7]
    ov_h_te = (np.maximum(0.0, np.abs(te_h0) - ys) ** 2) * np.sign(te_h0)
    d_hi_te = np.sum((np.maximum(0.0, np.abs(te_ps) - ys) ** 2) * past_dt[None, :], axis=1)
    ps_pred = (
        np.asarray(te["strain_0"], dtype=float)[:, None]
        + hw[0] * gap_te[:, 0:1] * (1.0 - np.exp(-eval_t[None, :] / t1h))
        + hw[1] * gap_te[:, 1:2] * (1.0 - np.exp(-eval_t[None, :] / t2h))
        + (dr_te + vw[0] * ov_h_te)[:, None] * eval_t[None, :]
        + vw[1] * (d_hi_te * te_h0)[:, None] * (1.0 - np.exp(-eval_t[None, :] / t1h))
    )
    ir_pred = hw[0] * gap_te[:, 0] / t1h + hw[1] * gap_te[:, 1] / t2h + dr_te + vw[0] * ov_h_te + vw[1] * (d_hi_te * te_h0) / t1h
    return {{"estimated_taus": np.array([t1h, t2h]), "hereditary_weights": hw, "viscoplastic_weights": vw, "predicted_strain": ps_pred, "initial_rate_pred": ir_pred}}
'''

    vortex_flag = "True" if branch == "C1" else "False"
    return f'''from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize

def _b1d(L, n, kap, vel, rg, d_add=None):
    h = float(L) / int(n)
    kap = np.asarray(kap, dtype=float).ravel(); v = np.asarray(vel, dtype=float).ravel()
    A = np.zeros((n, n), dtype=float)
    kh = (2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])) / (h * h)
    idx = np.arange(n - 1)
    A[idx, idx] += kh; A[idx + 1, idx + 1] += kh; A[idx, idx + 1] -= kh; A[idx + 1, idx] -= kh
    A[0, 0] += (rg * kap[0]) / (h * (rg * 0.5 * h + kap[0]))
    A[n - 1, n - 1] += (rg * kap[n - 1]) / (h * (rg * 0.5 * h + kap[n - 1]))
    for i in range(n):
        vi = float(v[i])
        if vi >= 0.0:
            A[i, i] += vi / h
            if i > 0: A[i, i - 1] -= vi / h
        else:
            A[i, i] += (-vi) / h
            if i < n - 1: A[i, i + 1] -= (-vi) / h
    if d_add is not None: A += np.diag(np.asarray(d_add, dtype=float).ravel())
    return A

def {entrypoint}(study: dict) -> dict:
    if "budget_grid" in study:
        raise ValueError("Single-budget solver requires multi-budget Pareto adaptation.")
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

    nx, ny = int(study["nx"]), int(study["ny"])
    V2 = np.asarray(study["V_reaction_2d"], dtype=float)
    mus = np.asarray(study["mu_shifts"], dtype=float)
    modes = np.asarray(study["source_modes"], dtype=float)
    M = modes.shape[0]
    si, sj = np.asarray(study["sensor_ij"], dtype=int)[:, 0], np.asarray(study["sensor_ij"], dtype=int)[:, 1]
    is_vortex = {vortex_flag}
    if not is_vortex:
        ax = np.asarray(study["reaction_x"], dtype=float) + np.mean(V2, axis=1)
        by = np.asarray(study["reaction_y"], dtype=float) + (np.mean(V2, axis=0) - float(np.mean(V2)))
        Ax = _b1d(study["Lx"], nx, study["kappa_x"], study["vel_x"], float(study["robin_x"]), ax)
        By = _b1d(study["Ly"], ny, study["kappa_y"], study["vel_y"], float(study["robin_y"]), by)
        lx, Vx = np.linalg.eig(Ax); ly, Vy = np.linalg.eig(By)
        Phi = np.einsum("ia,mab,bj->mij", np.linalg.inv(Vx), modes, np.linalg.inv(Vy).T, optimize=True)
        den = lx[:, None, None] + ly[None, :, None] + mus[None, None, :]
        U = np.real(np.einsum("ai,mijk,jb->mkab", Vx, Phi[:, :, :, None] / den[None, :, :, :], Vy.T, optimize=True))
    else:
        Ax = _b1d(study["Lx"], nx, 1.45 * np.asarray(study["kappa_x"], dtype=float), study["vel_x"], float(study["robin_x"]), study["reaction_x"])
        By = _b1d(study["Ly"], ny, 1.45 * np.asarray(study["kappa_y"], dtype=float), study["vel_y"], float(study["robin_y"]), study["reaction_y"])
        Kb = sparse.kron(sparse.csc_matrix(Ax), sparse.eye(ny, format="csc"), format="csc") + sparse.kron(sparse.eye(nx, format="csc"), sparse.csc_matrix(By), format="csc") + sparse.diags(V2.ravel(), format="csc")
        U = np.zeros((M, len(mus), nx, ny), dtype=float)
        rhs = modes.reshape(M, nx * ny).T
        for k in range(len(mus)):
            U[:, k, :, :] = splu((Kb + float(mus[k]) * sparse.eye(nx * ny, format="csc")).tocsc()).solve(rhs).T.reshape(M, nx, ny)

    S_st = np.vstack([U[:, k, si, sj].T for k in range(len(mus))])
    Z = np.asarray(study["Z_nuisance"], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + float(study["nuisance_ridge"]) * np.eye(Z.shape[1]), Z.T)
    r_y = y_cal - Z @ (Pz @ y_cal)
    S_ti = S_st - Z @ (Pz @ S_st)
    lw, ub, bt = float(study["source_ridge"]), float(study["u_box"]), float(study["total_budget"])
    H = S_ti.T @ S_ti + lw * np.eye(M); f = S_ti.T @ r_y
    res = minimize(lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w), np.full(M, bt / M), jac=lambda w: H @ w - f,
                   bounds=[(0.0, ub)] * M, constraints=[{{"type": "eq", "fun": lambda w: float(np.sum(w)) - bt, "jac": lambda w: np.ones(M)}}],
                   method="SLSQP", options={{"ftol": 1e-14, "maxiter": 400}})
    w = np.clip(res.x, 0.0, ub)
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
    return {{"weights": w, "terminal_field": np.tensordot(w, U[:, -1, :, :], axes=(0, 0)),
            "kkt_certificate": {{"weights": w.tolist(), "nu_sum": best_nu, "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
                                 "lam_upper": np.maximum(0.0, best_nu - g).tolist(), "primal_objective": jv, "dual_lower_bound": float(min(best_lb, jv))}}}}
'''

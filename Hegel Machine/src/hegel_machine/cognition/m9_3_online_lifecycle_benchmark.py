"""Hegel M9.3 Online Source Formation, Pre-Target Qualification Freeze, and Post-Freeze Lifecycle Benchmark (`Hegel-M93-Lifecycle`).

Inherits the `v10.2` lifecycle stages (`Stage S: Blind Source Formation -> Freeze Before Target Exposure -> Stage T: Transfer Successor + Stage C: Localized Premise Counterexample -> Stage R: Held-Out Original-Condition Recovery`) on real containerized Python/NumPy/SciPy scientific discovery tasks:

- **Chain 1 (`Multi-Scale Variable-Step Hereditary Viscoelasticity & Derivative-Nullspace Spectrum Identification`)**:
  1. `src01__variable_step_prony_spectrum_and_hereditary_creep_source` (`Stage S`):
     Unknown 2-branch Prony relaxation spectrum `(tau_1, tau_2)` (`0 < tau_1 < tau_2`) under variable-step loading `past_dt`,
     multi-lab reference-coupon alignment, Cauchy sensor glitches, and exact physical ODE initial-rate consistency (`dot{eps}(0+)`).
  2. `suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer` (`Stage T`, post-freeze):
     Shifts sampling schedule `past_dt`, supplies non-zero pre-loaded `initial_memory_states` `(N, 2)`, shifts latent `(tau_1, tau_2)`,
     and changes terminal readouts to effective secant creep compliance & independent-threshold first-passage crossing time.
  3. `suc01_C__viscoplastic_bingham_overstress_counterexample_revision` (`Stage C`, post-freeze localized premise counterexample):
     Above `yield_stress` (`|sigma| > sigma_y`), Perzyna viscoplastic overstress (`sign(sigma)*max(0, |sigma|-sigma_y)^2`) and
     cumulative plastic-work softening (`sum_s max(0, |past_stress_s|-sigma_y)^2 * dt_s`) break pure linear superposition on high-load
     trajectories while preserving 100% of the sub-yield linear Prony hereditary law & multi-lab panel calibration!
  4. `rec01_R__original_linear_hereditary_heldout_recovery` (`Stage R`, post-counterexample original-condition recovery):
     Unseen held-out seeds of the original pure-linear variable-step hereditary regime verifying zero overwrite of `generation=0`
     `src01` `L1` primitives and clean scope-gated recall after `suc01_C` bifurcation.

- **Chain 2 (`Separable 2D Kronecker Sylvester Multi-Shift Reaction-Transport & Box-Simplex KKT Certificate`)**:
  5. `src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source` (`Stage S`):
     Multi-shift (`K_mu = 10`) 2D separable finite-volume advection-diffusion-Robin Sylvester family `A_x U_{m,k} + U_{m,k} B_y^T + mu_k U_{m,k} = Phi_m`
     (`140` 2D PDE solves) coupled to nuisance-profiled Box-Simplex source tomography (`0 <= w <= u_box, sum(w) == B_tot`) with an exact
     tangential KKT convex duality lower-bound certificate `L_KKT(w, nu) = J(w) - g^T w + nu*B_tot + u_box*sum(min(0, g - nu))`.
  6. `suc02_T__shifted_robin_anisotropic_kronecker_and_multibudget_kkt_transfer` (`Stage T`, post-freeze):
     Shifts grid dimensions (`48 x 36`), reverses advection direction, shifts Robin boundary impedance, increases `K_mu = 12` & `M = 16`,
     and requires multi-budget KKT Pareto certification across `3` total budgets `budget_grid` + terminal 2D concentration field reconstruction.
  7. `suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision` (`Stage C`, post-freeze localized premise counterexample):
     Dual-channel tracer survey where Channel A obeys separable Kronecker transport (`V_coupling == 0`) while Channel B includes a localized
     2D non-separable reaction-vortex pocket `V_pocket(x, y) * U` breaking Kronecker separability on Channel B while preserving Channel A
     spectral reduction, nuisance profiling, and joint Box-Simplex KKT certification!
  8. `rec02_R__original_separable_kronecker_kkt_heldout_recovery` (`Stage R`, post-counterexample original-condition recovery):
     Unseen held-out seeds of the original separable Kronecker + Box-Simplex KKT regime verifying zero overwrite of `generation=0`
     `src02` `L1` primitives and clean scope-gated recall after `suc02_C` bifurcation.

Zero artificial solver bans; every task ships a real `public_setup_py` + `/holdout_task/validate_public_schema.py` verified pre-Turn-1 (`A2`).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    HoldoutOODTask,
    ensure_clean_scipy_sandbox_image,
    run_holdout_private_verifier,
)

# Shared standalone Box-Simplex KKT Convex Duality Certificate Validator (`0` duplicated math between public schema & private verifier!)
_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY = r'''
def verify_box_simplex_kkt_certificate(S_tilde, r_y, lam_w, u_box, b_tot, cert, *, gap_tol=1e-6, feas_tol=1e-7):
    """Independently verify primal feasibility (`0 <= w <= u_box`, `sum(w) == b_tot`) AND exact convex tangential KKT lower bound
    `L_KKT(w, nu) = J(w) - g^T w + nu * b_tot + u_box * sum_m min(0.0, g_m - nu) <= J(w^*)` (`0 <= J(w) - L_KKT(w, nu) <= gap_tol`).
    """
    if not isinstance(cert, dict):
        return {"valid": False, "reason": "cert_not_dict", "gap": 1e9}
    req = {"weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"}
    if not req.issubset(cert.keys()):
        return {"valid": False, "reason": f"missing_keys:{sorted(req - set(cert.keys()))}", "gap": 1e9}
    S = np.asarray(S_tilde, dtype=float)
    ry = np.asarray(r_y, dtype=float).ravel()
    M = S.shape[1]
    try:
        w = np.asarray(cert["weights"], dtype=float).ravel()
        ll = np.asarray(cert["lam_lower"], dtype=float).ravel()
        lu = np.asarray(cert["lam_upper"], dtype=float).ravel()
        nu = float(cert["nu_sum"])
        p_rep = float(cert["primal_objective"])
        d_rep = float(cert["dual_lower_bound"])
    except Exception as exc:
        return {"valid": False, "reason": f"cast_error:{exc}", "gap": 1e9}
    if w.shape != (M,) or ll.shape != (M,) or lu.shape != (M,):
        return {"valid": False, "reason": f"shape_mismatch:{w.shape}", "gap": 1e9}
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(ll)) and np.all(np.isfinite(lu)) and math.isfinite(nu) and math.isfinite(p_rep) and math.isfinite(d_rep)):
        return {"valid": False, "reason": "non_finite_values", "gap": 1e9}
    u_b = float(u_box)
    bt = float(b_tot)
    lw = float(lam_w)
    if float(np.min(w)) < -feas_tol or float(np.max(w)) > u_b + feas_tol:
        return {"valid": False, "reason": "box_violation", "gap": 1e9}
    if abs(float(np.sum(w)) - bt) > feas_tol * max(1.0, bt):
        return {"valid": False, "reason": f"budget_sum_violation:{abs(float(np.sum(w)) - bt):.3e}", "gap": 1e9}
    if float(np.min(ll)) < -feas_tol or float(np.min(lu)) < -feas_tol:
        return {"valid": False, "reason": "negative_bound_multipliers", "gap": 1e9}

    w_feas = np.clip(w, 0.0, u_b)
    res = S @ w_feas - ry
    j_true = 0.5 * float(res @ res) + 0.5 * lw * float(w_feas @ w_feas)
    g = S.T @ res + lw * w_feas
    # Unconditional convex supporting-hyperplane + box-Lagrange lower bound valid for ANY `(w_feas, nu)`:
    l_kkt = j_true - float(g @ w_feas) + nu * bt + u_b * float(np.sum(np.minimum(0.0, g - nu)))
    stat_res = float(np.max(np.abs(g - nu - ll + lu)))
    if abs(p_rep - j_true) > max(gap_tol, 1e-6 * max(1.0, abs(j_true))):
        return {"valid": False, "reason": f"reported_primal_mismatch:{abs(p_rep - j_true):.3e}", "gap": 1e9}
    if d_rep > l_kkt + gap_tol:
        return {"valid": False, "reason": f"unsound_dual_lower_bound:{d_rep:.8f}>{l_kkt:.8f}", "gap": 1e9}
    if stat_res > 1e-4:
        return {"valid": False, "reason": f"kkt_stationarity_violation:{stat_res:.3e}", "gap": 1e9}
    eff_lb = min(d_rep, l_kkt)
    gap = j_true - eff_lb
    if gap < -1e-9 or gap > gap_tol:
        return {"valid": False, "reason": f"duality_gap_too_large:{gap:.3e}", "gap": float(gap)}
    return {"valid": True, "reason": "ok", "true_primal": float(j_true), "verified_dual_lb": float(l_kkt), "gap": float(max(0.0, gap))}
'''


# ============================================================================
# CHAIN 1 GENERATOR SNIPPET (Shared structure, distinct parameters per stage)
# ============================================================================

_CHAIN1_CORE_GEN_PY = r'''
def _gen_chain1_cohort(seed, taus_true, theta_true, *, H=28, K=10, dt_mode="mild_var", preload_scale=0.0, yield_stress=None, vp_theta=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    taus = np.asarray(taus_true, dtype=float)
    theta = np.asarray(theta_true, dtype=float)
    offsets = np.array([3.1, -3.4, 3.6, -2.9], dtype=float)
    gains = np.array([1.32, 0.79, 1.41, 0.85], dtype=float)
    tr_cells = [[20, 8, 8], [8, 22, 8], [8, 8, 22], [18, 9, 9]]   # 148 train
    te_cells = [[12, 5, 5], [5, 13, 5], [5, 5, 13], [11, 6, 5]]   # 90 test
    s_means = np.array([[-0.95, 0.75, -0.55, 0.45], [0.10, -0.15, 0.20, -0.10], [1.05, -0.80, 0.65, -0.50]])

    if dt_mode == "mild_var":
        past_dt = np.round(rng.uniform(0.12, 0.42, size=H), 4)
        eval_times = np.array([0.20, 0.45, 0.80, 1.25, 1.85, 2.60, 3.60, 4.90, 6.50, 8.50], dtype=float)[:K]
    else:
        past_dt = np.round(np.geomspace(0.025, 0.115, num=H) * rng.uniform(0.90, 1.10, size=H), 4)
        eval_times = np.array([0.15, 0.35, 0.65, 1.00, 1.45, 2.00, 2.65, 3.45, 4.35, 5.45, 6.65, 7.90], dtype=float)[:K]

    def _mk_panel(cells):
        labs, sts, Zs, Xs, Rs = [], [], [], [], []
        for c in range(4):
            for s in range(3):
                m = cells[c][s]
                z = s_means[s] + rng.normal(scale=0.72, size=(m, 4))
                x = offsets[c] + gains[c] * z
                r = offsets[c] + gains[c] * rng.normal(size=(m, 20))
                labs.append(np.full(m, c, dtype=int)); sts.append(np.full(m, s, dtype=int))
                Zs.append(z); Xs.append(x); Rs.append(r)
        p = rng.permutation(sum(sum(r) for r in cells))
        Z = np.vstack(Zs)[p]; X = np.vstack(Xs)[p]; R = np.vstack(Rs)[p]
        Xn = X.copy(); Xn[rng.uniform(size=X.shape) < 0.025] = np.nan
        Rn = R.copy(); Rn[rng.uniform(size=R.shape) < 0.04] = np.nan
        return np.concatenate(labs)[p], np.concatenate(sts)[p], Z, Xn, Rn

    tr_lab, tr_st, tr_Z, tr_X, tr_R = _mk_panel(tr_cells)
    te_lab, te_st, te_Z, te_X, te_R = _mk_panel(te_cells)

    def _sim(Z, n_twins=0, add_glitches=False):
        n = len(Z)
        t_grid = np.cumsum(past_dt)
        init_mem = rng.normal(scale=preload_scale, size=(n, 2)) if preload_scale > 0.0 else np.zeros((n, 2), dtype=float)
        amp_scale = 2.45 if yield_stress is not None else 1.85
        hist = (
            rng.uniform(-amp_scale, amp_scale, size=(n, 1)) * np.sin(rng.uniform(0.5, 1.8, size=(n, 1)) * t_grid[None, :] + rng.uniform(0, 6.28, size=(n, 1)))
            + rng.normal(scale=0.55, size=(n, H))
        )
        hold0 = rng.uniform(0.55, (2.75 if yield_stress is not None else 2.15), size=n) * rng.choice([-1.0, 1.0], size=n)
        if yield_stress is not None:
            # Ensure ~45% strictly sub-yield coupons and ~55% overstressed coupons for crisp counterexample contrast
            sub_mask = np.arange(n) % 2 == 0
            hist[sub_mask] = np.clip(hist[sub_mask], -0.92 * yield_stress, 0.92 * yield_stress)
            hold0[sub_mask] = np.clip(hold0[sub_mask], -0.92 * yield_stress, 0.92 * yield_stress)
            over_idx = np.flatnonzero(~sub_mask)
            hold0[over_idx] = rng.choice([-1.0, 1.0], size=len(over_idx)) * rng.uniform(yield_stress + 0.35, yield_stress + 1.25, size=len(over_idx))
            hist[over_idx, -6:] = rng.choice([-1.0, 1.0], size=(len(over_idx), 1)) * rng.uniform(yield_stress + 0.30, yield_stress + 1.20, size=(len(over_idx), 6))

        # Construct exact nullspace twin pairs in test (`same present state & initial derivative, opposite past history!`)
        if n_twins > 0:
            dec1, dec2 = np.exp(-past_dt / taus[0]), np.exp(-past_dt / taus[1])
            w1 = np.array([(1.0 - dec1[s]) * np.prod(dec1[s + 1 :]) for s in range(H)])
            w2 = np.array([(1.0 - dec2[s]) * np.prod(dec2[s + 1 :]) for s in range(H)])
            M_w = np.vstack([w1, w2])
            c1, c2 = theta[0] / taus[0], theta[1] / taus[1]
            v_null = np.array([c2, -c1], dtype=float)
            v_null /= max(np.linalg.norm(v_null), 1e-12)
            delta_h = np.linalg.lstsq(M_w, 2.2 * v_null, rcond=None)[0]
            if yield_stress is not None:
                # Keep twin pairs sub-yield so linear derivative nullspace holds identically
                base_h = np.zeros(H, dtype=float)
                delta_h = delta_h * min(1.0, 0.88 * yield_stress / max(float(np.max(np.abs(delta_h))), 1e-9))
            else:
                base_h = 0.25 * np.sin(0.8 * t_grid)
            for k in range(n_twins):
                ia, ib = n - 2 * n_twins + 2 * k, n - 2 * n_twins + 2 * k + 1
                Z[ib] = Z[ia].copy()
                init_mem[ib] = init_mem[ia].copy()
                hold0[ia] = hold0[ib] = (0.85 * yield_stress if yield_stress is not None else 1.45)
                hist[ia] = base_h + delta_h
                hist[ib] = base_h - delta_h

        z = init_mem.copy()
        for s in range(H):
            dec = np.exp(-past_dt[s] / taus)
            z = z * dec[None, :] + (1.0 - dec[None, :]) * hist[:, s : s + 1]

        s0 = 0.22 * hold0 + 0.14 * z[:, 0] + 0.10 * z[:, 1] + rng.normal(scale=0.02, size=n)
        if n_twins > 0:
            for k in range(n_twins):
                ia, ib = n - 2 * n_twins + 2 * k, n - 2 * n_twins + 2 * k + 1
                s0[ib] = s0[ia]

        e1 = 1.0 - np.exp(-eval_times / taus[0])
        e2 = 1.0 - np.exp(-eval_times / taus[1])
        drift = theta[2] + Z @ theta[3:7]
        clean = (
            s0[:, None]
            + theta[0] * (hold0 - z[:, 0])[:, None] * e1[None, :]
            + theta[1] * (hold0 - z[:, 1])[:, None] * e2[None, :]
            + drift[:, None] * eval_times[None, :]
        )
        true_rate0 = theta[0] * (hold0 - z[:, 0]) / taus[0] + theta[1] * (hold0 - z[:, 1]) / taus[1] + drift

        if yield_stress is not None:
            ys = float(yield_stress)
            ov_hold = (np.maximum(0.0, np.abs(hold0) - ys) ** 2) * np.sign(hold0)
            d_hist = np.sum((np.maximum(0.0, np.abs(hist) - ys) ** 2) * past_dt[None, :], axis=1)
            clean = clean + vp_theta[0] * ov_hold[:, None] * eval_times[None, :] + vp_theta[1] * (d_hist * hold0)[:, None] * e1[None, :]
            true_rate0 = true_rate0 + vp_theta[0] * ov_hold + vp_theta[1] * (d_hist * hold0) / taus[0]

        rate_obs = true_rate0 + rng.normal(scale=0.010, size=n)
        if n_twins > 0:
            for k in range(n_twins):
                ia, ib = n - 2 * n_twins + 2 * k, n - 2 * n_twins + 2 * k + 1
                rate_obs[ib] = rate_obs[ia]

        obs = clean + rng.normal(scale=0.008, size=clean.shape)
        if add_glitches:
            n_gl = max(1, int(0.08 * obs.size))
            g_flat = rng.choice(obs.size, size=n_gl, replace=False)
            obs.ravel()[g_flat] += rng.choice([-1.0, 1.0], size=n_gl) * rng.uniform(0.75, 1.15, size=n_gl)

        # Independent hazard threshold (ZERO future terminal strain leakage!)
        crit = s0 + rng.uniform(0.15, 0.58, size=n)
        ts_full = np.concatenate([[0.0], eval_times])
        ys_full = np.column_stack([s0, clean])
        cross = np.full(n, float(eval_times[-1]), dtype=float)
        for i in range(n):
            c = float(crit[i])
            if ys_full[i, 0] >= c:
                cross[i] = 0.0
                continue
            for ki in range(len(ts_full) - 1):
                if ys_full[i, ki + 1] >= c:
                    dy = max(ys_full[i, ki + 1] - ys_full[i, ki], 1e-12)
                    cross[i] = float(ts_full[ki] + (c - ys_full[i, ki]) / dy * (ts_full[ki + 1] - ts_full[ki]))
                    break
        secant = (clean[:, -1] - s0) / np.maximum(np.abs(hold0), 0.5)
        return init_mem, hist, s0, hold0, rate_obs, true_rate0, clean, obs, crit, cross, secant

    tr_im, tr_h, tr_s0, tr_h0, tr_ro, tr_rt, tr_cl, tr_ob, _, _, _ = _sim(tr_Z, n_twins=0, add_glitches=True)
    te_im, te_h, te_s0, te_h0, te_ro, te_rt, te_cl, _, te_cr, te_cx, te_sec = _sim(te_Z, n_twins=8, add_glitches=False)
    for k in range(8):
        ia, ib = len(te_Z) - 16 + 2 * k, len(te_Z) - 16 + 2 * k + 1
        te_X[ib] = offsets[te_lab[ib]] + gains[te_lab[ib]] * te_Z[ib]

    ds = {
        "past_dt": past_dt,
        "eval_times": eval_times,
        "train": {
            "lab_id": tr_lab, "stratum": tr_st, "coupon_features_raw": tr_X, "ref_coupons": tr_R,
            "past_stress": tr_h, "strain_0": tr_s0, "stress_hold_0": tr_h0, "instant_rate_obs": tr_ro,
            "strain_curves_obs": tr_ob,
        },
        "test": {
            "lab_id": te_lab, "stratum": te_st, "coupon_features_raw": te_X, "ref_coupons": te_R,
            "past_stress": te_h, "strain_0": te_s0, "stress_hold_0": te_h0, "instant_rate_obs": te_ro,
        },
    }
    if preload_scale > 0.0:
        ds["train"]["initial_memory_states"] = tr_im
        ds["test"]["initial_memory_states"] = te_im
        ds["test"]["critical_strain"] = te_cr
    if yield_stress is not None:
        ds["yield_stress"] = float(yield_stress)
    gt = {
        "taus": taus,
        "theta": theta,
        "vp_theta": np.asarray(vp_theta, dtype=float),
        "te_Z": te_Z,
        "te_clean": te_cl,
        "te_true_rate0": te_rt,
        "te_cross": te_cx,
        "te_secant": te_sec,
    }
    return ds, gt
'''


# ============================================================================
# CHAIN 2 GENERATOR SNIPPET (Shared 1D/2D Kronecker Sylvester + Box-Simplex KKT)
# ============================================================================

_CHAIN2_CORE_GEN_PY = _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY + r'''
def _build_1d_fvm_advdiff_robin(L, n, kappa, vel, robin_gamma):
    h = float(L) / int(n)
    kap = np.asarray(kappa, dtype=float).ravel()
    v = np.asarray(vel, dtype=float).ravel()
    rg = float(robin_gamma)
    A = np.zeros((n, n), dtype=float)
    # Harmonic face diffusion: (1/h^2) * 2*k_i*k_{i+1}/(k_i + k_{i+1})
    kh = (2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])) / (h * h)
    idx = np.arange(n - 1)
    A[idx, idx] += kh
    A[idx + 1, idx + 1] += kh
    A[idx, idx + 1] -= kh
    A[idx + 1, idx] -= kh
    # Robin boundaries at cell 0 and cell n-1: (1/h) * (rg * k_b) / (rg * 0.5 * h + k_b)
    A[0, 0] += (rg * kap[0]) / (h * (rg * 0.5 * h + kap[0]))
    A[n - 1, n - 1] += (rg * kap[n - 1]) / (h * (rg * 0.5 * h + kap[n - 1]))
    # First-order cell upwind advection `(v * du/dx)` with zero inflow exterior
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
    # Exact Euclidean projection polish onto `{0 <= w <= u_box, sum(w) == b_tot}` via 1D bisection + Newton/L-BFGS active polish
    w = np.clip(res.x, 0.0, float(u_box))
    lo, hi = float(np.min(w) - u_box), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, float(u_box)))) > float(b_tot):
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, float(u_box))
    # Active-set subspace KKT polish to machine precision (`< 1e-12` duality gap!)
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
    k_ord = min(M - 1, max(0, int(math.floor(float(b_tot) / float(u_box)))))
    # Evaluate exact best `nu` across all gradient breakpoints `g`
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
    cert = {
        "weights": w.tolist(),
        "nu_sum": best_nu,
        "lam_lower": ll.tolist(),
        "lam_upper": lu.tolist(),
        "primal_objective": float(j_val),
        "dual_lower_bound": float(min(best_lb, j_val)),
    }
    return w, cert

def _gen_chain2_study(seed, *, nx=44, ny=40, K_mu=10, M=14, vel_sign=1.0, robin_pair=(1.35, 1.15), multi_budget=False, nonseparable_channel_b=False):
    rng = np.random.default_rng(seed)
    Lx, Ly = 2.2, 1.9
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    kappa_x = 0.42 + 0.16 * np.cos(math.pi * xs / Lx) + 0.05 * rng.uniform(-1.0, 1.0, size=nx)
    kappa_y = 0.38 + 0.15 * np.sin(math.pi * ys / Ly) + 0.05 * rng.uniform(-1.0, 1.0, size=ny)
    vel_x = float(vel_sign) * (0.48 + 0.12 * np.sin(2.0 * math.pi * xs / Lx))
    vel_y = float(vel_sign) * (-0.42 + 0.10 * np.cos(2.0 * math.pi * ys / Ly))
    Ax = _build_1d_fvm_advdiff_robin(Lx, nx, kappa_x, vel_x, robin_pair[0])
    By = _build_1d_fvm_advdiff_robin(Ly, ny, kappa_y, vel_y, robin_pair[1])
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

    # Spectral Kronecker solve for separable operator `Ax U + U By^T + mu_k U = Phi_m`
    lx, Vx = np.linalg.eig(Ax)
    ly, Vy = np.linalg.eig(By)
    Vx_inv = np.linalg.inv(Vx)
    Vy_inv_T = np.linalg.inv(Vy).T
    Phi_hat = np.einsum("ia,mab,bj->mij", Vx_inv, source_modes, Vy_inv_T)
    denom = lx[:, None, None] + ly[None, :, None] + mu_shifts[None, None, :]  # (nx, ny, K_mu)
    U_hat = Phi_hat[:, :, :, None] / denom[None, :, :, :]
    U_sep = np.real(np.einsum("ai,mijk,jb->mkab", Vx, U_hat, Vy.T))  # (M, K_mu, nx, ny)

    S_sep_blocks = [U_sep[:, k, si, sj].T for k in range(K_mu)]  # each (N_sens, M)
    S_sep = np.vstack(S_sep_blocks)  # (K_mu * N_sens, M)

    V_pocket = None
    U_chB = None
    if nonseparable_channel_b:
        from scipy import sparse
        from scipy.sparse.linalg import splu
        # Strongly non-separable 2D vortex-cross pocket `V_pocket(x, y)` (rank >= 3, additive separable projection error ~0.25!)
        V_pocket = (
            4.8 * np.exp(-((X2 - 0.52 * Lx) ** 2 + (Y2 - 0.48 * Ly) ** 2) / 0.14) * np.sin(2.6 * math.pi * X2 / Lx) * np.sin(2.4 * math.pi * Y2 / Ly)
            + 3.4 * np.exp(-((X2 - 0.35 * Lx - 0.45 * Y2) ** 2) / 0.08)
        )
        V_pocket = V_pocket - float(np.min(V_pocket)) + 0.15  # strictly positive non-separable reaction pocket!
        K_base = sparse.kron(sparse.csc_matrix(Ax), sparse.eye(ny, format="csc"), format="csc") + sparse.kron(sparse.eye(nx, format="csc"), sparse.csc_matrix(By), format="csc") + sparse.diags(V_pocket.ravel(), format="csc")
        U_chB = np.zeros((M, K_mu, nx, ny), dtype=float)
        rhs_mat = source_modes.reshape(M, nx * ny).T
        for k in range(K_mu):
            lu_k = splu((K_base + float(mu_shifts[k]) * sparse.eye(nx * ny, format="csc")).tocsc())
            sol_k = lu_k.solve(rhs_mat)  # (nx*ny, M)
            U_chB[:, k, :, :] = sol_k.T.reshape(M, nx, ny)
        S_chB = np.vstack([U_chB[:, k, si, sj].T for k in range(K_mu)])
        S_stack = np.vstack([S_sep, S_chB])
    else:
        S_stack = S_sep

    n_rows = S_stack.shape[0]
    P_z = 5
    Z_nuis = rng.normal(size=(n_rows, P_z))
    lam_z = 0.65
    lam_w = 0.04
    u_box = 1.0
    b_tot = 2.60

    w_true = np.zeros(M, dtype=float)
    act_m = [1, 4, 8, 11]
    raw_w = np.array([0.88, 0.72, 0.58, 0.42], dtype=float)
    w_true[act_m] = raw_w * (b_tot / float(np.sum(raw_w)))
    gamma_true = rng.normal(scale=0.65, size=P_z)
    y_obs = S_stack @ w_true + Z_nuis @ gamma_true + rng.normal(scale=0.012, size=n_rows)

    Pz_mat = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(P_z), Z_nuis.T)
    r_y = y_obs - Z_nuis @ (Pz_mat @ y_obs)
    S_tilde = S_stack - Z_nuis @ (Pz_mat @ S_stack)

    study = {
        "Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny,
        "kappa_x": kappa_x, "vel_x": vel_x, "robin_x": float(robin_pair[0]),
        "kappa_y": kappa_y, "vel_y": vel_y, "robin_y": float(robin_pair[1]),
        "mu_shifts": mu_shifts,
        "source_modes": source_modes,
        "sensor_ij": sensor_ij,
        "Z_nuisance": Z_nuis,
        "y_obs": y_obs,
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
        term_field = np.tensordot(w_grid[1], U_sep[:, -1, :, :], axes=(0, 0))
        gt = {
            "Ax": Ax, "By": By, "S_tilde": S_tilde, "r_y": r_y,
            "weights_by_budget": np.vstack(w_grid), "certs_by_budget": certs_grid,
            "terminal_field_mid_budget": term_field,
        }
    else:
        study["total_budget"] = b_tot
        if nonseparable_channel_b:
            study["V_pocket_channel_b"] = V_pocket
        w_opt, cert_opt = _solve_box_simplex_exact(S_tilde, r_y, lam_w, u_box, b_tot)
        term_A = np.tensordot(w_opt, U_sep[:, -1, :, :], axes=(0, 0))
        term_B = np.tensordot(w_opt, U_chB[:, -1, :, :], axes=(0, 0)) if nonseparable_channel_b else None
        gt = {
            "Ax": Ax, "By": By, "S_tilde": S_tilde, "r_y": r_y,
            "w_opt": w_opt, "cert_opt": cert_opt,
            "terminal_field_A": term_A, "terminal_field_B": term_B,
        }
    return study, gt
'''


# ============================================================================
# 8 M9.3 LIFECYCLE TASKS (`2 Source + 4 Successor [T, C] + 2 Recovery [R]`)
# ============================================================================

M93_LIFECYCLE_TASKS: tuple[HoldoutOODTask, ...] = (
    # -------------------------------------------------------------------------
    # 1. SRC01 (`Stage S` Source — Chain 1): Variable-Step Prony Spectrum & Creep
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="src01__variable_step_prony_spectrum_and_hereditary_creep_source",
        short_name="src01_variable_step_prony_hereditary_source",
        wave="m93_stage_S_source",
        scientific_domain="Multi-Timescale Variable-Step Hereditary Viscoelastic Spectrum Identification & Twin Creep Forecasting",
        wormhole_family="fam_hereditary_viscoelastic_memory",
        declared_artifacts=("/holdout_task/hereditary_prony_source_solver.py",),
        declared_callable_entrypoints=("solve_hereditary_prony_source",),
        instruction_text=(
            "Implement `/holdout_task/hereditary_prony_source_solver.py` exposing:\n"
            "`def solve_hereditary_prony_source(dataset: dict) -> dict:`\n\n"
            "Multi-Lab Variable-Step Hereditary Viscoelastic Spectrum & Derivative-Consistent Creep Contract:\n"
            "1. MULTI-LAB REFERENCE-COUPON ALIGNMENT:\n"
            "   - `dataset['train']` and `dataset['test']` share `'lab_id'` (`C = 4` labs `0..3`), `'stratum'` (`S = 3` strata `0..2`),\n"
            "     `'coupon_features_raw'` (`(N, 4)` float64 with ~2.5% `NaN`s), and `'ref_coupons'` (`(N, 20)` reference standards drawn from\n"
            "     `N(0, I_20)` per lab before lab affine shift/scale, with ~4% `NaN`s).\n"
            "   - Concatenate train+test panels to estimate each lab's scalar `offset[c]` (median across valid ref columns of column mean) and\n"
            "     `gain[c]` (median across valid ref columns of column `ddof=0` std), standardize `(X_raw - offset[c]) / gain[c]`, compute\n"
            "     cell-balanced `(lab, stratum)` weights (`mean == 1.0`), and impute `NaN`s with weighted column means -> `X_aligned` (`(N, 4)`).\n"
            "2. VARIABLE-STEP 2-BRANCH PRONY HEREDITARY MEMORY & UNKNOWN SPECTRUM `(tau_1, tau_2)`:\n"
            "   - `dataset['past_dt']` (`shape (H,)`, `dt_s > 0`) gives variable step durations of piecewise-constant past stress `past_stress` (`(N, H)`).\n"
            "   - Starting from zero pre-history memory `z_m = 0.0` (`m in {1, 2}`), each interval `s = 0..H-1` updates:\n"
            "     `z_m <- exp(-past_dt[s] / tau_m) * z_m + (1.0 - exp(-past_dt[s] / tau_m)) * past_stress[:, s]`, with UNKNOWN `0.20 <= tau_1 < tau_2 <= 8.0`!\n"
            "   - During hold `[0, T]` at `eval_times` (`shape (K,)`) under constant hold stress `stress_hold_0` (`sigma_0`), starting from `strain_0` (`eps_0`):\n"
            "     `eps(t_k) = eps_0 + theta_0*(sigma_0 - z_1)*(1 - exp(-t_k/tau_1)) + theta_1*(sigma_0 - z_2)*(1 - exp(-t_k/tau_2)) + (theta_2 + X_aligned @ theta_{3:7}) * t_k`.\n"
            "   - Physical ODE initial creep rate at `t = 0+` (`instant_rate_obs` is noisy observation of `dot{eps}(0+)`):\n"
            "     `dot{eps}(0+) = theta_0*(sigma_0 - z_1)/tau_1 + theta_1*(sigma_0 - z_2)/tau_2 + theta_2 + X_aligned @ theta_{3:7}`.\n"
            "   - ~8% of `train['strain_curves_obs']` entries suffer heavy Cauchy sensor glitches (`+-0.95`)! Jointly or profile-identify `(tau_1, tau_2)`\n"
            "     (`tau_1 < tau_2`) and `theta` (`shape (7,)`) using robust Cauchy/Lorentzian IRLS (plus `instant_rate_obs` consistency).\n"
            "3. RETURN DICT:\n"
            "   - `'estimated_taus'`: `(2,)` float array `[tau_1, tau_2]` (`tau_1 < tau_2`, rel error `<= 0.10`),\n"
            "   - `'hereditary_weights'`: `(7,)` float array `theta_0..theta_6`,\n"
            "   - `'predicted_strain'`: `(N_test, K)` float array (MAE `<= 0.065`, twin-pair MAE `<= 0.065`),\n"
            "   - `'initial_rate_pred'`: `(N_test,)` float array `dot{eps}(0+)` (MAE `<= 0.040`).\n"
            "- Public dataset: `/holdout_task/public_src01_datasets.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN1_CORE_GEN_PY + r'''
pub_ds, pub_gt = _gen_chain1_cohort(9301, (0.60, 4.00), [0.68, 0.54, 0.09, 0.16, -0.13, 0.11, -0.09])
with open("/holdout_task/public_src01_datasets.pkl", "wb") as fh:
    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "hereditary_prony_source_solver.py")
spec = importlib.util.spec_from_file_location("cand_src01", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = pickle.load(open(os.path.join(_d, "public_src01_datasets.pkl"), "rb"))[0]["dataset"]
out = mod.solve_hereditary_prony_source(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        ts = np.asarray(out["estimated_taus"], dtype=float)
        if not (0.05 < ts[0] < ts[1] < 25.0): errs.append("taus_not_positive_ordered")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport hereditary_prony_source_solver\n" + _CHAIN1_CORE_GEN_PY + r'''
cases = [
    (99501, (0.55, 3.80), [0.72, 0.56, 0.08, 0.17, -0.14, 0.12, -0.08]),
    (99502, (0.66, 4.35), [0.64, 0.58, 0.10, -0.15, 0.13, -0.11, 0.09]),
]
t0 = time.monotonic()
valid = True; max_tau_rel = 0.0; max_mae = 0.0; max_twin_mae = 0.0; max_rate_mae = 0.0; rows = []
for sd, taus_t, th_t in cases:
    ds, gt = _gen_chain1_cohort(sd, taus_t, th_t)
    out = hereditary_prony_source_solver.solve_hereditary_prony_source(ds)
    if not isinstance(out, dict) or not {"estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ir = np.asarray(out["initial_rate_pred"], dtype=float)
    hw = np.asarray(out["hereditary_weights"], dtype=float)
    if et.shape != (2,) or hw.shape != (7,) or ps.shape != gt["te_clean"].shape or ir.shape != gt["te_true_rate0"].shape:
        valid = False; rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(hw)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ir)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({"seed": sd, "error": "non_finite_or_unordered_taus"}); continue
    tau_rel = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    s_mae = float(np.mean(np.abs(ps - gt["te_clean"])))
    tw_mae = float(np.mean(np.abs(ps[-16:] - gt["te_clean"][-16:])))
    r_mae = float(np.mean(np.abs(ir - gt["te_true_rate0"])))
    max_tau_rel = max(max_tau_rel, tau_rel); max_mae = max(max_mae, s_mae); max_twin_mae = max(max_twin_mae, tw_mae); max_rate_mae = max(max_rate_mae, r_mae)
    rows.append({"seed": sd, "tau_rel": round(tau_rel, 6), "strain_mae": round(s_mae, 6), "twin_mae": round(tw_mae, 6), "rate_mae": round(r_mae, 6)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau_rel <= 0.10 and max_mae <= 0.065 and max_twin_mae <= 0.065 and max_rate_mae <= 0.040 and el <= 20.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau_rel, 6), "max_strain_mae": round(max_mae, 6), "max_twin_mae": round(max_twin_mae, 6), "max_rate_mae": round(max_rate_mae, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 2. SUC01_T (`Stage T` Post-Freeze Transfer — Chain 1)
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer",
        short_name="suc01_T_preloaded_hereditary_secant_hazard",
        wave="m93_stage_T_transfer",
        scientific_domain="Pre-Loaded Non-Uniform Hereditary Viscoelasticity, Secant Compliance & First-Passage Hazard Crossing",
        wormhole_family="fam_hereditary_viscoelastic_memory",
        declared_artifacts=("/holdout_task/preloaded_hereditary_secant_hazard_solver.py",),
        declared_callable_entrypoints=("solve_preloaded_hereditary_transfer",),
        instruction_text=(
            "Implement `/holdout_task/preloaded_hereditary_secant_hazard_solver.py` exposing:\n"
            "`def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:`\n\n"
            "Pre-Loaded Non-Uniform Hereditary Viscoelasticity, Secant Compliance & Hazard Crossing Contract:\n"
            "1. MULTI-LAB REFERENCE-COUPON ALIGNMENT:\n"
            "   - Same 4-lab x 3-stratum coupon alignment + cell-balanced imputation (`coupon_features_raw` `(N, 4)` + `ref_coupons` `(N, 20)`).\n"
            "2. NON-ZERO INITIAL MEMORY STATES + LOG-SPACED VARIABLE `past_dt` (`H = 22`, `K = 12`):\n"
            "   - Unlike unloaded specimens, each trajectory begins at `t_{-H}` with NON-ZERO pre-existing internal branch stresses\n"
            "     `initial_memory_states` (`shape (N, 2)`, `[z_1(-T_hist), z_2(-T_hist)]`)! Propagate `z_m` from `initial_memory_states[:, m]`\n"
            "     across `s = 0..H-1` via `z_m <- exp(-past_dt[s]/tau_m)*z_m + (1 - exp(-past_dt[s]/tau_m))*past_stress[:, s]` with UNKNOWN\n"
            "     `0.20 <= tau_1 < tau_2 <= 9.0` and ~8% Cauchy sensor glitches on `train['strain_curves_obs']`.\n"
            "   - Same hold-phase constitutive law & physical initial derivative `dot{eps}(0+)` (`instant_rate_obs` provided).\n"
            "3. SHIFTED TERMINAL READOUTS (`Secant Creep Compliance` + `First-Passage Hazard Crossing`):\n"
            "   - `'estimated_taus'`: `(2,)` float array `[tau_1, tau_2]` (`tau_1 < tau_2`, rel error `<= 0.10`),\n"
            "   - `'predicted_strain'`: `(N_test, K)` float array (MAE `<= 0.075`),\n"
            "   - `'secant_compliance'`: `(N_test,)` float array `(predicted_strain[:, -1] - test['strain_0']) / np.maximum(np.abs(test['stress_hold_0']), 0.5)` (MAE `<= 0.095`),\n"
            "   - `'hazard_crossing_times'`: `(N_test,)` float array giving the earliest `t in [0, eval_times[-1]]` where piecewise-linear\n"
            "     interpolated strain `(0, strain_0) -> (eval_times, predicted_strain)` reaches `test['critical_strain']` (`0.0` if `strain_0 >= crit`,\n"
            "     `eval_times[-1]` if never reached; MAE `<= 0.28`).\n"
            "- Public dataset: `/holdout_task/public_suc01_T_datasets.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN1_CORE_GEN_PY + r'''
pub_ds, pub_gt = _gen_chain1_cohort(9311, (0.46, 3.35), [0.74, 0.52, 0.08, 0.15, -0.12, 0.10, -0.08], H=22, K=12, dt_mode="geom_var", preload_scale=2.20)
with open("/holdout_task/public_suc01_T_datasets.pkl", "wb") as fh:
    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "preloaded_hereditary_secant_hazard_solver.py")
spec = importlib.util.spec_from_file_location("cand_suc01_T", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = pickle.load(open(os.path.join(_d, "public_suc01_T_datasets.pkl"), "rb"))[0]["dataset"]
out = mod.solve_preloaded_hereditary_transfer(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("predicted_strain", (n_te, k_t)), ("secant_compliance", (n_te,)), ("hazard_crossing_times", (n_te,))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport preloaded_hereditary_secant_hazard_solver\n" + _CHAIN1_CORE_GEN_PY + r'''
cases = [
    (99511, (0.42, 3.15), [0.76, 0.50, 0.09, -0.16, 0.13, -0.10, 0.08]),
    (99512, (0.78, 5.10), [0.62, 0.60, 0.07, 0.14, -0.15, 0.11, -0.09]),
]
t0 = time.monotonic()
valid = True; max_tau = 0.0; max_s = 0.0; max_sec = 0.0; max_cx = 0.0; rows = []
for sd, taus_t, th_t in cases:
    ds, gt = _gen_chain1_cohort(sd, taus_t, th_t, H=22, K=12, dt_mode="geom_var", preload_scale=2.20)
    out = preloaded_hereditary_secant_hazard_solver.solve_preloaded_hereditary_transfer(ds)
    if not isinstance(out, dict) or not {"estimated_taus", "predicted_strain", "secant_compliance", "hazard_crossing_times"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    sc = np.asarray(out["secant_compliance"], dtype=float)
    cx = np.asarray(out["hazard_crossing_times"], dtype=float)
    if et.shape != (2,) or ps.shape != gt["te_clean"].shape or sc.shape != gt["te_secant"].shape or cx.shape != gt["te_cross"].shape:
        valid = False; rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(sc)) and np.all(np.isfinite(cx)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({"seed": sd, "error": "non_finite"}); continue
    tr = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    sm = float(np.mean(np.abs(ps - gt["te_clean"])))
    scm = float(np.mean(np.abs(sc - gt["te_secant"])))
    cxm = float(np.mean(np.abs(cx - gt["te_cross"])))
    max_tau = max(max_tau, tr); max_s = max(max_s, sm); max_sec = max(max_sec, scm); max_cx = max(max_cx, cxm)
    rows.append({"seed": sd, "tau_rel": round(tr, 6), "strain_mae": round(sm, 6), "secant_mae": round(scm, 6), "cross_mae": round(cxm, 6)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau <= 0.10 and max_s <= 0.075 and max_sec <= 0.095 and max_cx <= 0.28 and el <= 20.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau, 6), "max_strain_mae": round(max_s, 6), "max_secant_mae": round(max_sec, 6), "max_crossing_mae": round(max_cx, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 3. SUC01_C (`Stage C` Localized Premise Counterexample — Chain 1)
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="suc01_C__viscoplastic_bingham_overstress_counterexample_revision",
        short_name="suc01_C_viscoplastic_overstress_counterexample",
        wave="m93_stage_C_counterexample",
        scientific_domain="Localized Premise Counterexample: Bingham-Perzyna Viscoplastic Overstress & Plastic-Work Softening Above Yield",
        wormhole_family="fam_hereditary_viscoelastic_memory",
        declared_artifacts=("/holdout_task/viscoplastic_overstress_hereditary_solver.py",),
        declared_callable_entrypoints=("solve_viscoplastic_hereditary_revision",),
        instruction_text=(
            "Implement `/holdout_task/viscoplastic_overstress_hereditary_solver.py` exposing:\n"
            "`def solve_viscoplastic_hereditary_revision(dataset: dict) -> dict:`\n\n"
            "Localized Premise Counterexample — Sub-Yield Linear Prony Memory + Above-Yield Perzyna Overstress Contract:\n"
            "1. PRESERVED SUBSTRUCTURE:\n"
            "   - Same 4-lab x 3-stratum coupon alignment + cell-balanced imputation (`X_aligned` `(N, 4)`).\n"
            "   - On sub-yield loading (`|past_stress| <= sigma_y` and `|stress_hold_0| <= sigma_y`, where `sigma_y = dataset['yield_stress']`),\n"
            "     the material obeys the EXACT 2-branch variable-step linear Prony law (`z_1, z_2` with unknown `0.20 <= tau_1 < tau_2 <= 8.0`).\n"
            "2. LOCALIZED ABOVE-YIELD VISCOPLASTIC PREMISE VIOLATION (`|sigma| > sigma_y`):\n"
            "   - Whenever stress exceeds `sigma_y = float(dataset['yield_stress'])`, two irreversible viscoplastic mechanisms activate:\n"
            "     * Signed quadratic hold overstress `ov_hold = (np.maximum(0.0, np.abs(stress_hold_0) - sigma_y) ** 2) * np.sign(stress_hold_0)`\n"
            "       drives steady Perzyna viscoplastic creep `theta_7 * ov_hold * t_k`;\n"
            "     * Cumulative plastic-work history `D_hist = np.sum((np.maximum(0.0, np.abs(past_stress) - sigma_y) ** 2) * past_dt[None, :], axis=1)`\n"
            "       drives transient damage-softened surge `theta_8 * (D_hist * stress_hold_0) * (1 - exp(-t_k / tau_1))`!\n"
            "   - Note: `instant_rate_obs` at `t = 0+` includes `+ theta_7 * ov_hold + theta_8 * (D_hist * stress_hold_0) / tau_1`, and ~8% of\n"
            "     `train['strain_curves_obs']` entries contain Cauchy glitches (`+-0.95`). Pure linear Prony (`theta_7 = theta_8 = 0`) FAILS\n"
            "     badly on the overstressed cohort!\n"
            "3. RETURN DICT:\n"
            "   - `'estimated_taus'`: `(2,)` float array `[tau_1, tau_2]` (`tau_1 < tau_2`, rel error `<= 0.10`),\n"
            "   - `'viscoplastic_weights'`: `(2,)` float array `[theta_7, theta_8]` (rel L2 `<= 0.14`),\n"
            "   - `'predicted_strain'`: `(N_test, K)` float array (overall MAE `<= 0.070`, overstressed-cohort MAE `<= 0.080`, sub-yield MAE `<= 0.065`).\n"
            "- Public dataset: `/holdout_task/public_suc01_C_datasets.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN1_CORE_GEN_PY + r'''
pub_ds, pub_gt = _gen_chain1_cohort(9321, (0.58, 3.90), [0.68, 0.54, 0.08, 0.15, -0.12, 0.11, -0.09], yield_stress=1.55, vp_theta=(0.24, 0.16))
with open("/holdout_task/public_suc01_C_datasets.pkl", "wb") as fh:
    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "viscoplastic_overstress_hereditary_solver.py")
spec = importlib.util.spec_from_file_location("cand_suc01_C", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = pickle.load(open(os.path.join(_d, "public_suc01_C_datasets.pkl"), "rb"))[0]["dataset"]
out = mod.solve_viscoplastic_hereditary_revision(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("viscoplastic_weights", (2,)), ("predicted_strain", (n_te, k_t))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport viscoplastic_overstress_hereditary_solver\n" + _CHAIN1_CORE_GEN_PY + r'''
cases = [
    (99521, (0.52, 3.65), [0.70, 0.55, 0.09, 0.16, -0.13, 0.10, -0.08], 1.50, (0.26, 0.18)),
    (99522, (0.68, 4.45), [0.65, 0.57, 0.08, -0.14, 0.12, -0.11, 0.09], 1.60, (0.22, 0.15)),
]
t0 = time.monotonic()
valid = True; max_tau = 0.0; max_vp = 0.0; max_all = 0.0; max_over = 0.0; max_sub = 0.0; rows = []
for sd, taus_t, th_t, ys_t, vp_t in cases:
    ds, gt = _gen_chain1_cohort(sd, taus_t, th_t, yield_stress=ys_t, vp_theta=vp_t)
    out = viscoplastic_overstress_hereditary_solver.solve_viscoplastic_hereditary_revision(ds)
    if not isinstance(out, dict) or not {"estimated_taus", "viscoplastic_weights", "predicted_strain"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    vw = np.asarray(out["viscoplastic_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    if et.shape != (2,) or vw.shape != (2,) or ps.shape != gt["te_clean"].shape:
        valid = False; rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(vw)) and np.all(np.isfinite(ps)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({"seed": sd, "error": "non_finite"}); continue
    over_mask = (np.abs(ds["test"]["stress_hold_0"]) > ys_t) | (np.max(np.abs(ds["test"]["past_stress"]), axis=1) > ys_t)
    sub_mask = ~over_mask
    tr = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    vr = float(np.linalg.norm(vw - gt["vp_theta"]) / np.linalg.norm(gt["vp_theta"]))
    am = float(np.mean(np.abs(ps - gt["te_clean"])))
    om = float(np.mean(np.abs(ps[over_mask] - gt["te_clean"][over_mask])))
    sm = float(np.mean(np.abs(ps[sub_mask] - gt["te_clean"][sub_mask])))
    max_tau = max(max_tau, tr); max_vp = max(max_vp, vr); max_all = max(max_all, am); max_over = max(max_over, om); max_sub = max(max_sub, sm)
    rows.append({"seed": sd, "tau_rel": round(tr, 6), "vp_rel_l2": round(vr, 6), "all_mae": round(am, 6), "over_mae": round(om, 6), "sub_mae": round(sm, 6)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau <= 0.10 and max_vp <= 0.14 and max_all <= 0.070 and max_over <= 0.080 and max_sub <= 0.065 and el <= 20.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau, 6), "max_vp_rel_l2": round(max_vp, 6), "max_all_mae": round(max_all, 6), "max_overstress_mae": round(max_over, 6), "max_subyield_mae": round(max_sub, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 4. REC01_R (`Stage R` Original-Condition Recovery — Chain 1)
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="rec01_R__original_linear_hereditary_heldout_recovery",
        short_name="rec01_R_original_linear_hereditary_recovery",
        wave="m93_stage_R_recovery",
        scientific_domain="Original-Condition Return Check: Held-Out Pure Linear Variable-Step Hereditary Spectrum & Creep",
        wormhole_family="fam_hereditary_viscoelastic_memory",
        declared_artifacts=("/holdout_task/hereditary_prony_recovery_solver.py",),
        declared_callable_entrypoints=("solve_hereditary_prony_recovery",),
        instruction_text=(
            "Implement `/holdout_task/hereditary_prony_recovery_solver.py` exposing:\n"
            "`def solve_hereditary_prony_recovery(dataset: dict) -> dict:`\n\n"
            "Held-Out Original-Condition Recovery (Pure Linear Variable-Step 2-Branch Prony Hereditary Creep, ZERO Viscoplastic Overstress):\n"
            "- Same pure-linear multi-lab coupon alignment (`4` labs, `3` strata, `coupon_features_raw` `(N, 4)` + `ref_coupons` `(N, 20)`),\n"
            "  variable-step `past_dt` (`(H,)`), unknown `0.20 <= tau_1 < tau_2 <= 8.0`, ~8% Cauchy glitches on `train['strain_curves_obs']`,\n"
            "  and exact ODE initial rate `dot{eps}(0+)` (`instant_rate_obs`). Note: there is NO `yield_stress` or viscoplastic overstress in\n"
            "  this original linear regime — verify that pure linear Prony memory recovers `(tau_1, tau_2)` and twin strain curves cleanly!\n"
            "- Return `{'estimated_taus': (2,), 'hereditary_weights': (7,), 'predicted_strain': (N_test, K), 'initial_rate_pred': (N_test,)}`.\n"
            "- Public dataset: `/holdout_task/public_rec01_R_datasets.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN1_CORE_GEN_PY + r'''
pub_ds, pub_gt = _gen_chain1_cohort(9391, (0.50, 3.60), [0.71, 0.53, 0.08, -0.15, 0.12, 0.10, -0.09])
with open("/holdout_task/public_rec01_R_datasets.pkl", "wb") as fh:
    pickle.dump([{"dataset": pub_ds, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "hereditary_prony_recovery_solver.py")
spec = importlib.util.spec_from_file_location("cand_rec01_R", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = pickle.load(open(os.path.join(_d, "public_rec01_R_datasets.pkl"), "rb"))[0]["dataset"]
out = mod.solve_hereditary_prony_recovery(ds)
n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport hereditary_prony_recovery_solver\n" + _CHAIN1_CORE_GEN_PY + r'''
cases = [
    (99591, (0.48, 3.55), [0.73, 0.55, 0.09, 0.16, -0.13, 0.11, -0.08]),
    (99592, (0.71, 4.65), [0.63, 0.59, 0.08, -0.14, 0.14, -0.10, 0.09]),
]
t0 = time.monotonic()
valid = True; max_tau_rel = 0.0; max_mae = 0.0; max_rate_mae = 0.0; rows = []
for sd, taus_t, th_t in cases:
    ds, gt = _gen_chain1_cohort(sd, taus_t, th_t)
    out = hereditary_prony_recovery_solver.solve_hereditary_prony_recovery(ds)
    if not isinstance(out, dict) or not {"estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ir = np.asarray(out["initial_rate_pred"], dtype=float)
    if et.shape != (2,) or ps.shape != gt["te_clean"].shape or ir.shape != gt["te_true_rate0"].shape or not (np.all(np.isfinite(et)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ir)) and 0.05 < et[0] < et[1]):
        valid = False; rows.append({"seed": sd, "error": "bad_output"}); continue
    tau_rel = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    s_mae = float(np.mean(np.abs(ps - gt["te_clean"])))
    r_mae = float(np.mean(np.abs(ir - gt["te_true_rate0"])))
    max_tau_rel = max(max_tau_rel, tau_rel); max_mae = max(max_mae, s_mae); max_rate_mae = max(max_rate_mae, r_mae)
    rows.append({"seed": sd, "tau_rel": round(tau_rel, 6), "strain_mae": round(s_mae, 6), "rate_mae": round(r_mae, 6)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau_rel <= 0.10 and max_mae <= 0.065 and max_rate_mae <= 0.040 and el <= 20.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau_rel, 6), "max_strain_mae": round(max_mae, 6), "max_rate_mae": round(max_rate_mae, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 5. SRC02 (`Stage S` Source — Chain 2): Kronecker Sylvester & Box-Simplex KKT
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source",
        short_name="src02_kronecker_sylvester_box_simplex_kkt_source",
        wave="m93_stage_S_source",
        scientific_domain="Separable 2D Kronecker Sylvester Multi-Shift Reaction-Transport & Certified Box-Simplex KKT Tomography",
        wormhole_family="fam_kronecker_sylvester_and_kkt_certification",
        declared_artifacts=("/holdout_task/kronecker_box_simplex_kkt_source_solver.py",),
        declared_callable_entrypoints=("solve_kronecker_box_simplex_source",),
        instruction_text=(
            "Implement `/holdout_task/kronecker_box_simplex_kkt_source_solver.py` exposing:\n"
            "`def solve_kronecker_box_simplex_source(study: dict) -> dict:`\n\n"
            "Separable 2D Kronecker Sylvester Multi-Shift Transport & Certified Box-Simplex KKT Contract:\n"
            "1. STAGE 1 (`1D Cell-Centered Advection-Diffusion-Robin Stencil & Multi-Shift Sylvester Family`):  \n"
            "   - On a 2D `nx x ny` (`44 x 40`) grid over `[0, Lx] x [0, Ly]`, build 1D operators `A_x` (`(nx, nx)` from `Lx, nx, kappa_x, vel_x, robin_x`)\n"
            "     and `B_y` (`(ny, ny)` from `Ly, ny, kappa_y, vel_y, robin_y`) with cell step `h = L / n`:\n"
            "     * Interior face harmonic diffusion (`i = 0..n-2`): `kh_i = (2*kappa[i]*kappa[i+1]/(kappa[i]+kappa[i+1])) / h^2`, adding `+kh_i`\n"
            "       to `M[i,i]` and `M[i+1,i+1]`, and `-kh_i` to `M[i,i+1]` and `M[i+1,i]`.\n"
            "     * Robin boundary impedance at `b in {0, n-1}`: add `(robin * kappa[b]) / (h * (robin * 0.5 * h + kappa[b]))` to `M[b,b]`.\n"
            "     * First-order cell upwind advection (`i = 0..n-1`, zero inflow exterior): if `vel[i] >= 0`, add `vel[i]/h` to `M[i,i]` and\n"
            "       (if `i > 0`) `-vel[i]/h` to `M[i,i-1]`; else add `(-vel[i])/h` to `M[i,i]` and (if `i < n-1`) `-(-vel[i])/h` to `M[i,i+1]`.\n"
            "   - Across ALL `M = 14` `source_modes` (`(M, nx, ny)`) and `K_mu = 10` `mu_shifts` (`(K_mu,)`), solve the Sylvester equation:\n"
            "     `A_x @ U_{m,k} + U_{m,k} @ B_y.T + mu_shifts[k] * U_{m,k} = source_modes[m]` (`140` 2D solves; Hint: 1D spectral decomposition\n"
            "     of `A_x` and `B_y` solves all `M * K_mu` shifts simultaneously to `< 1e-12`!). Stack sensor readouts `U_{:, k, si, sj}.T`\n"
            "     (`sensor_ij` `(N_sens, 2)`) for `k = 0..K_mu-1` into `S_stack` (`shape (K_mu * N_sens, M)`).\n"
            "2. STAGE 2 (`Nuisance Ridge Profiling & Proof-Carrying Box-Simplex KKT Certificate`):\n"
            "   - Project out `Z = study['Z_nuisance']` (`ridge = study['nuisance_ridge']`): `P_z = solve(Z.T @ Z + ridge*I, Z.T)`,\n"
            "     `r_y = y_obs - Z @ (P_z @ y_obs)`, `S_tilde = S_stack - Z @ (P_z @ S_stack)`.\n"
            "   - Minimize `J(w) = 0.5 * ||S_tilde @ w - r_y||_2^2 + 0.5 * lam_w * ||w||_2^2` (`lam_w = study['source_ridge']`) over the\n"
            "     box-simplex `{0 <= w_m <= u_box, sum_m w_m == total_budget}` (`u_box = study['u_box']`, `B_tot = study['total_budget']`).\n"
            "   - Compute gradient `g = S_tilde.T @ (S_tilde @ w - r_y) + lam_w * w` and optimal equality multiplier `nu_sum in R` maximizing the\n"
            "     exact supporting-hyperplane dual lower bound `L_KKT(w, nu) = J(w) - g @ w + nu * B_tot + u_box * sum(min(0, g - nu))`,\n"
            "     with `lam_lower = maximum(0, g - nu_sum)`, `lam_upper = maximum(0, nu_sum - g)` (`0 <= J(w) - L_KKT <= 1e-6`).\n"
            "3. RETURN `{'weights': (M,), 'terminal_field': sum_m w_m * U_{m, K_mu-1} (nx, ny), 'kkt_certificate': {'weights': list, 'nu_sum': float, 'lam_lower': list, 'lam_upper': list, 'primal_objective': float, 'dual_lower_bound': float}}`.\n"
            "- Public study: `/holdout_task/public_src02_studies.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN2_CORE_GEN_PY + r'''
pub_st, pub_gt = _gen_chain2_study(9331)
with open("/holdout_task/public_src02_studies.pkl", "wb") as fh:
    pickle.dump([{"study": pub_st, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\nimport numpy as np\n" + ''' + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY) + r''' + """
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "kronecker_box_simplex_kkt_source_solver.py")
spec = importlib.util.spec_from_file_location("cand_src02", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_src02_studies.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_kronecker_box_simplex_source(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"))
    if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport kronecker_box_simplex_kkt_source_solver\n" + _CHAIN2_CORE_GEN_PY + r'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd in (99601, 99602):
    st, gt = _gen_chain2_study(sd)
    out = kronecker_box_simplex_kkt_source_solver.solve_kronecker_box_simplex_source(st)
    if not isinstance(out, dict) or not {"weights", "terminal_field", "kkt_certificate"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    w = np.asarray(out["weights"], dtype=float)
    tf = np.asarray(out["terminal_field"], dtype=float)
    if w.shape != gt["w_opt"].shape or tf.shape != gt["terminal_field_A"].shape or not (np.all(np.isfinite(w)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({"seed": sd, "error": "shape_or_nan"}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"])
    if not chk["valid"]:
        valid = False; rows.append({"seed": sd, "error": f"kkt:{chk['reason']}"}); continue
    wr = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "kkt_gap": round(float(chk["gap"]), 10)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 1e-4 and max_f_rel <= 1e-4 and max_gap <= 1e-6 and el <= 18.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 6. SUC02_T (`Stage T` Post-Freeze Transfer — Chain 2)
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="suc02_T__shifted_robin_anisotropic_kronecker_and_multibudget_kkt_transfer",
        short_name="suc02_T_multibudget_kronecker_kkt_transfer",
        wave="m93_stage_T_transfer",
        scientific_domain="Anisotropic Reversed-Advection Kronecker Sylvester Transport & Multi-Budget KKT Pareto Certification",
        wormhole_family="fam_kronecker_sylvester_and_kkt_certification",
        declared_artifacts=("/holdout_task/multibudget_kronecker_kkt_transfer_solver.py",),
        declared_callable_entrypoints=("solve_multibudget_kronecker_transfer",),
        instruction_text=(
            "Implement `/holdout_task/multibudget_kronecker_kkt_transfer_solver.py` exposing:\n"
            "`def solve_multibudget_kronecker_transfer(study: dict) -> dict:`\n\n"
            "Shifted Geometry/Reversed-Advection Kronecker Sylvester & Multi-Budget KKT Pareto Contract:\n"
            "1. SHIFTED SEPARABLE 2D OPERATOR (`48 x 36`, `M = 16` sources, `K_mu = 12` shifts -> `192` 2D Sylvester PDE solves):\n"
            "   - Same 1D harmonic-diffusion + upwind-advection + Robin boundary stencil for `A_x` (`48 x 48`) and `B_y` (`36 x 36`),\n"
            "     solving `A_x @ U_{m,k} + U_{m,k} @ B_y.T + mu_shifts[k] * U_{m,k} = source_modes[m]` and stacking sensor readouts\n"
            "     `U_{:, k, si, sj}.T` across `k = 0..K_mu-1` -> `S_stack` (`(K_mu * N_sens, M)`).\n"
            "2. MULTI-BUDGET PARETO BOX-SIMPLEX KKT CERTIFICATION (`budget_grid` of length `B = 3`):\n"
            "   - Profile out `Z_nuisance` (`ridge = study['nuisance_ridge']`) -> `(S_tilde, r_y)`.\n"
            "   - For EACH total budget `B_b = float(study['budget_grid'][b])` (`b = 0, 1, 2`), solve the box-simplex QP\n"
            "     `min_{0 <= w <= u_box, sum(w) == B_b} 0.5*||S_tilde @ w - r_y||_2^2 + 0.5*lam_w*||w||_2^2` and emit an exact KKT\n"
            "     tangential dual certificate (`0 <= J(w) - L_KKT <= 1e-6`).\n"
            "3. RETURN `{'weights_by_budget': (3, M), 'terminal_field_mid_budget': sum_m w_mid[m]*U_{m, K_mu-1} (nx, ny) for b=1, 'kkt_certificates': [cert_0, cert_1, cert_2]}`.\n"
            "- Public study: `/holdout_task/public_suc02_T_studies.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN2_CORE_GEN_PY + r'''
pub_st, pub_gt = _gen_chain2_study(9341, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)
with open("/holdout_task/public_suc02_T_studies.pkl", "wb") as fh:
    pickle.dump([{"study": pub_st, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\nimport numpy as np\n" + ''' + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY) + r''' + """
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "multibudget_kronecker_kkt_transfer_solver.py")
spec = importlib.util.spec_from_file_location("cand_suc02_T", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_suc02_T_studies.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_multibudget_kronecker_transfer(st)
M, nx, ny = st["source_modes"].shape
B = len(st["budget_grid"])
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights_by_budget", (B, M)), ("terminal_field_mid_budget", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    certs = out.get("kkt_certificates")
    if not isinstance(certs, list) or len(certs) != B: errs.append("bad_certs_list")
    else:
        for bi, bg in enumerate(st["budget_grid"]):
            chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi])
            if not chk["valid"]: errs.append(f"kkt_b{bi}:{chk['reason']}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport multibudget_kronecker_kkt_transfer_solver\n" + _CHAIN2_CORE_GEN_PY + r'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd in (99611, 99612):
    st, gt = _gen_chain2_study(sd, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True)
    out = multibudget_kronecker_kkt_transfer_solver.solve_multibudget_kronecker_transfer(st)
    if not isinstance(out, dict) or not {"weights_by_budget", "terminal_field_mid_budget", "kkt_certificates"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    wb = np.asarray(out["weights_by_budget"], dtype=float)
    tf = np.asarray(out["terminal_field_mid_budget"], dtype=float)
    certs = out["kkt_certificates"]
    if wb.shape != gt["weights_by_budget"].shape or tf.shape != gt["terminal_field_mid_budget"].shape or not isinstance(certs, list) or len(certs) != 3:
        valid = False; rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(wb)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({"seed": sd, "error": "non_finite"}); continue
    ok_all = True
    for bi, bg in enumerate(st["budget_grid"]):
        chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], float(bg), certs[bi])
        if not chk["valid"]:
            ok_all = False; rows.append({"seed": sd, "error": f"kkt_b{bi}:{chk['reason']}"}); break
        max_gap = max(max_gap, float(chk["gap"]))
    if not ok_all:
        valid = False; continue
    wr = float(np.linalg.norm(wb - gt["weights_by_budget"]) / np.linalg.norm(gt["weights_by_budget"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_mid_budget"]) / np.linalg.norm(gt["terminal_field_mid_budget"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr)
    rows.append({"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "max_kkt_gap": round(max_gap, 10)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 1e-4 and max_f_rel <= 1e-4 and max_gap <= 1e-6 and el <= 18.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 7. SUC02_C (`Stage C` Localized Premise Counterexample — Chain 2)
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision",
        short_name="suc02_C_nonseparable_vortex_pocket_counterexample",
        wave="m93_stage_C_counterexample",
        scientific_domain="Localized Premise Counterexample: Dual-Channel Separable + Non-Separable 2D Vortex Pocket & Certified Box-Simplex KKT",
        wormhole_family="fam_kronecker_sylvester_and_kkt_certification",
        declared_artifacts=("/holdout_task/coupled_channel_vortex_kkt_solver.py",),
        declared_callable_entrypoints=("solve_coupled_channel_vortex_inversion",),
        instruction_text=(
            "Implement `/holdout_task/coupled_channel_vortex_kkt_solver.py` exposing:\n"
            "`def solve_coupled_channel_vortex_inversion(study: dict) -> dict:`\n\n"
            "Localized Premise Counterexample — Dual-Channel Tracer Survey (`Separable Channel A` + `Non-Separable Vortex Pocket Channel B`):\n"
            "1. PRESERVED vs VIOLATED SEPARABILITY (`nx x ny = 36 x 32`, `M = 12` sources, `K_mu = 8` shifts):\n"
            "   - Same 1D operators `A_x` (`36 x 36`) and `B_y` (`32 x 32`).\n"
            "   - **Channel A (Separable)**: Solves `A_x @ U^A_{m,k} + U^A_{m,k} @ B_y.T + mu_shifts[k] * U^A_{m,k} = source_modes[m]`\n"
            "     (Kronecker spectral reduction holds identically!).\n"
            "   - **Channel B (Non-Separable 2D Reaction-Vortex Pocket Counterexample)**: Solves\n"
            "     `A_x @ U^B_{m,k} + U^B_{m,k} @ B_y.T + (V_pocket_channel_b + mu_shifts[k]) * U^B_{m,k} = source_modes[m]`, where\n"
            "     `V_pocket_channel_b` (`(nx, ny)`) is a strongly non-separable 2D cross-potential (`V(x,y) != v_x(x) + v_y(y)` — rank-1\n"
            "     additive separable approximation FAILS badly!). Assemble the exact 2D sparse operator `kron(Ax, I_y) + kron(I_x, By) + diag(V_pocket.ravel() + mu_k)`\n"
            "     (C-order `reshape(nx * ny)`) on Channel B.\n"
            "   - Stack sensor readouts `[S_A; S_B]` (`shape (2 * K_mu * N_sens, M)` where `S_A` stacks `k=0..K_mu-1` then `S_B` stacks `k=0..K_mu-1`).\n"
            "2. PRESERVED NUISANCE PROFILING & BOX-SIMPLEX KKT CERTIFICATE:\n"
            "   - Profile out `Z_nuisance` (`ridge = study['nuisance_ridge']`) -> `(S_tilde, r_y)` and solve the exact Box-Simplex QP\n"
            "     `{0 <= w <= u_box, sum(w) == total_budget}` with KKT certificate `kkt_certificate` (`0 <= J(w) - L_KKT <= 1e-6`).\n"
            "3. RETURN `{'weights': (M,), 'terminal_field_channel_a': sum_m w_m * U^A_{m, K_mu-1} (nx, ny), 'terminal_field_channel_b': sum_m w_m * U^B_{m, K_mu-1} (nx, ny), 'kkt_certificate': dict}`.\n"
            "- Public study: `/holdout_task/public_suc02_C_studies.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN2_CORE_GEN_PY + r'''
pub_st, pub_gt = _gen_chain2_study(9351, nx=36, ny=32, K_mu=8, M=12, nonseparable_channel_b=True)
with open("/holdout_task/public_suc02_C_studies.pkl", "wb") as fh:
    pickle.dump([{"study": pub_st, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\nimport numpy as np\n" + ''' + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY) + r''' + """
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "coupled_channel_vortex_kkt_solver.py")
spec = importlib.util.spec_from_file_location("cand_suc02_C", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_suc02_C_studies.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_coupled_channel_vortex_inversion(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field_channel_a", (nx, ny)), ("terminal_field_channel_b", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"))
    if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport coupled_channel_vortex_kkt_solver\n" + _CHAIN2_CORE_GEN_PY + r'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_fa_rel = 0.0; max_fb_rel = 0.0; max_gap = 0.0; rows = []
for sd in (99621, 99622):
    st, gt = _gen_chain2_study(sd, nx=36, ny=32, K_mu=8, M=12, nonseparable_channel_b=True)
    out = coupled_channel_vortex_kkt_solver.solve_coupled_channel_vortex_inversion(st)
    if not isinstance(out, dict) or not {"weights", "terminal_field_channel_a", "terminal_field_channel_b", "kkt_certificate"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    w = np.asarray(out["weights"], dtype=float)
    fa = np.asarray(out["terminal_field_channel_a"], dtype=float)
    fb = np.asarray(out["terminal_field_channel_b"], dtype=float)
    if w.shape != gt["w_opt"].shape or fa.shape != gt["terminal_field_A"].shape or fb.shape != gt["terminal_field_B"].shape:
        valid = False; rows.append({"seed": sd, "error": "shape_mismatch"}); continue
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(fa)) and np.all(np.isfinite(fb))):
        valid = False; rows.append({"seed": sd, "error": "non_finite"}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"])
    if not chk["valid"]:
        valid = False; rows.append({"seed": sd, "error": f"kkt:{chk['reason']}"}); continue
    wr = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    far = float(np.linalg.norm(fa - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    fbr = float(np.linalg.norm(fb - gt["terminal_field_B"]) / np.linalg.norm(gt["terminal_field_B"]))
    max_w_rel = max(max_w_rel, wr); max_fa_rel = max(max_fa_rel, far); max_fb_rel = max(max_fb_rel, fbr); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({"seed": sd, "w_rel_l2": round(wr, 8), "fa_rel_l2": round(far, 8), "fb_rel_l2": round(fbr, 8), "kkt_gap": round(float(chk["gap"]), 10)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 1e-4 and max_fa_rel <= 1e-4 and max_fb_rel <= 1e-4 and max_gap <= 1e-6 and el <= 18.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_a_rel_l2": round(max_fa_rel, 8), "max_field_b_rel_l2": round(max_fb_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),

    # -------------------------------------------------------------------------
    # 8. REC02_R (`Stage R` Original-Condition Recovery — Chain 2)
    # -------------------------------------------------------------------------
    HoldoutOODTask(
        task_id="rec02_R__original_separable_kronecker_kkt_heldout_recovery",
        short_name="rec02_R_original_separable_kronecker_recovery",
        wave="m93_stage_R_recovery",
        scientific_domain="Original-Condition Return Check: Held-Out Pure Separable 2D Kronecker Sylvester & Box-Simplex KKT",
        wormhole_family="fam_kronecker_sylvester_and_kkt_certification",
        declared_artifacts=("/holdout_task/kronecker_box_simplex_kkt_recovery_solver.py",),
        declared_callable_entrypoints=("solve_kronecker_box_simplex_recovery",),
        instruction_text=(
            "Implement `/holdout_task/kronecker_box_simplex_kkt_recovery_solver.py` exposing:\n"
            "`def solve_kronecker_box_simplex_recovery(study: dict) -> dict:`\n\n"
            "Held-Out Original-Condition Recovery (Pure Separable 2D Kronecker Sylvester `44 x 40`, `M = 14`, `K_mu = 10` + Box-Simplex KKT):\n"
            "- Same pure separable single-channel `A_x @ U_{m,k} + U_{m,k} @ B_y.T + mu_shifts[k] * U_{m,k} = source_modes[m]` (`0` non-separable\n"
            "  pocket!), `Z_nuisance` ridge profiling, and Box-Simplex KKT dual optimality certificate (`0 <= J(w) - L_KKT <= 1e-6`).\n"
            "- Return `{'weights': (M,), 'terminal_field': (nx, ny), 'kkt_certificate': dict}`.\n"
            "- Public study: `/holdout_task/public_rec02_R_studies.pkl`. Schema validator: `python3 /holdout_task/validate_public_schema.py`."
        ),
        public_setup_py="import json, math, os, pickle, sys\nimport numpy as np\n" + _CHAIN2_CORE_GEN_PY + r'''
pub_st, pub_gt = _gen_chain2_study(9392)
with open("/holdout_task/public_rec02_R_studies.pkl", "wb") as fh:
    pickle.dump([{"study": pub_st, "gt": pub_gt}], fh)

open("/holdout_task/validate_public_schema.py", "w").write("import importlib.util, json, math, os, pickle, sys\nimport numpy as np\n" + ''' + repr(_SHARED_BOX_SIMPLEX_KKT_CHECKER_PY) + r''' + """
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "kronecker_box_simplex_kkt_recovery_solver.py")
spec = importlib.util.spec_from_file_location("cand_rec02_R", cand_p)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
item = pickle.load(open(os.path.join(_d, "public_rec02_R_studies.pkl"), "rb"))[0]
st, gt = item["study"], item["gt"]
out = mod.solve_kronecker_box_simplex_recovery(st)
M, nx, ny = st["source_modes"].shape
errs = []
if not isinstance(out, dict): errs.append("not_dict")
else:
    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):
        if k not in out: errs.append(f"missing:{k}")
        else:
            a = np.asarray(out[k], dtype=float)
            if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out.get("kkt_certificate"))
    if not chk["valid"]: errs.append(f"kkt_invalid:{chk['reason']}")
print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))
sys.exit(0 if len(errs) == 0 else 1)
""")
''',
        private_verifier_py="import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\nimport kronecker_box_simplex_kkt_recovery_solver\n" + _CHAIN2_CORE_GEN_PY + r'''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd in (99691, 99692):
    st, gt = _gen_chain2_study(sd)
    out = kronecker_box_simplex_kkt_recovery_solver.solve_kronecker_box_simplex_recovery(st)
    if not isinstance(out, dict) or not {"weights", "terminal_field", "kkt_certificate"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "missing_keys"}); continue
    w = np.asarray(out["weights"], dtype=float)
    tf = np.asarray(out["terminal_field"], dtype=float)
    if w.shape != gt["w_opt"].shape or tf.shape != gt["terminal_field_A"].shape or not (np.all(np.isfinite(w)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({"seed": sd, "error": "shape_or_nan"}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"])
    if not chk["valid"]:
        valid = False; rows.append({"seed": sd, "error": f"kkt:{chk['reason']}"}); continue
    wr = float(np.linalg.norm(w - gt["w_opt"]) / np.linalg.norm(gt["w_opt"]))
    fr = float(np.linalg.norm(tf - gt["terminal_field_A"]) / np.linalg.norm(gt["terminal_field_A"]))
    max_w_rel = max(max_w_rel, wr); max_f_rel = max(max_f_rel, fr); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({"seed": sd, "w_rel_l2": round(wr, 8), "field_rel_l2": round(fr, 8), "kkt_gap": round(float(chk["gap"]), 10)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 1e-4 and max_f_rel <= 1e-4 and max_gap <= 1e-6 and el <= 18.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_weights_rel_l2": round(max_w_rel, 8), "max_field_rel_l2": round(max_f_rel, 8), "max_kkt_gap": round(max_gap, 10), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
''',
    ),
)

M93_TASKS_BY_ID: dict[str, HoldoutOODTask] = {t.task_id: t for t in M93_LIFECYCLE_TASKS}

_M93_PUBLIC_FIXTURES_BY_ID: dict[str, tuple[str, ...]] = {
    "src01__variable_step_prony_spectrum_and_hereditary_creep_source": (
        "/holdout_task/public_src01_datasets.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "suc01_T__nonuniform_preloaded_viscoelastic_secant_and_hazard_transfer": (
        "/holdout_task/public_suc01_T_datasets.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "suc01_C__viscoplastic_bingham_overstress_counterexample_revision": (
        "/holdout_task/public_suc01_C_datasets.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "rec01_R__original_linear_hereditary_heldout_recovery": (
        "/holdout_task/public_rec01_R_datasets.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source": (
        "/holdout_task/public_src02_studies.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "suc02_T__shifted_robin_anisotropic_kronecker_and_multibudget_kkt_transfer": (
        "/holdout_task/public_suc02_T_studies.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision": (
        "/holdout_task/public_suc02_C_studies.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
    "rec02_R__original_separable_kronecker_kkt_heldout_recovery": (
        "/holdout_task/public_rec02_R_studies.pkl",
        "/holdout_task/validate_public_schema.py",
    ),
}


def m93_lifecycle_task_to_frontier_task(ht: HoldoutOODTask) -> m6f.FrontierBenchmarkTask:
    ensure_clean_scipy_sandbox_image()
    return m6f.FrontierBenchmarkTask(
        task_id=ht.task_id,
        short_name=ht.short_name,
        benchmark="Hegel-M93-Lifecycle",
        wave=ht.wave,
        scientific_domain=ht.scientific_domain,
        wormhole_family=ht.wormhole_family,
        instruction_text=ht.instruction_text,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        declared_artifacts=ht.declared_artifacts,
        declared_callable_entrypoints=ht.declared_callable_entrypoints,
        holdout_public_setup_py=ht.public_setup_py,
        required_public_assets=_M93_PUBLIC_FIXTURES_BY_ID[ht.task_id],
        private_verifier_callable=run_m93_lifecycle_private_verifier_for_container,
        agent_cpus=2,
        agent_memory_mb=2560,
        verifier_cpus=2,
        verifier_memory_mb=2560,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
    )


def get_m93_source_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [m93_lifecycle_task_to_frontier_task(t) for t in M93_LIFECYCLE_TASKS if t.wave == "m93_stage_S_source"]


def get_m93_successor_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [
        m93_lifecycle_task_to_frontier_task(t)
        for t in M93_LIFECYCLE_TASKS
        if t.wave in {"m93_stage_T_transfer", "m93_stage_C_counterexample"}
    ]


def get_m93_recovery_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [m93_lifecycle_task_to_frontier_task(t) for t in M93_LIFECYCLE_TASKS if t.wave == "m93_stage_R_recovery"]


def run_m93_lifecycle_private_verifier_by_id(task_id: str, submitted_tar_path: Path | None) -> dict[str, Any]:
    ht = M93_TASKS_BY_ID[task_id]
    rep = run_holdout_private_verifier(ht, submitted_tar_path)
    rep["benchmark"] = "Hegel-M93-Lifecycle"
    return rep


def run_m93_lifecycle_private_verifier_for_container(task_id: str, agent_cid: str) -> dict[str, Any]:
    ht = M93_TASKS_BY_ID[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_m93_ver_") as td:
        tar_p = Path(td) / "submitted.tar"
        m6f.export_declared_artifacts_tar(agent_cid, ht.declared_artifacts, tar_p)
        rep = run_holdout_private_verifier(ht, tar_p)
        rep["benchmark"] = "Hegel-M93-Lifecycle"
        return rep


def evaluate_direct_source_float_plugin_controls() -> dict[str, Any]:
    """`v10.2` `C2` Lifecycle Non-Triviality Check:
    Verify that plugging fitted `src01`/`src02` numerical floats directly into `suc01_T`, `suc01_C`, `suc02_T`, `suc02_C`
    fails by orders of magnitude above target tolerances (proving operator-structure reuse rather than float memorization!).
    """
    import math as _math

    ns1: dict[str, Any] = {"np": np, "math": _math}
    exec(compile(_CHAIN1_CORE_GEN_PY, "<chain1_ctrl>", "exec"), ns1)
    _, src01_gt = ns1["_gen_chain1_cohort"](9301, (0.60, 4.00), [0.68, 0.54, 0.09, 0.16, -0.13, 0.11, -0.09])
    suc01_t_ds, suc01_t_gt = ns1["_gen_chain1_cohort"](
        99512, (0.78, 5.10), [0.62, 0.60, 0.07, 0.14, -0.15, 0.11, -0.09], H=22, K=12, dt_mode="geom_var", preload_scale=2.20
    )
    suc01_c_ds, suc01_c_gt = ns1["_gen_chain1_cohort"](
        99521, (0.52, 3.65), [0.70, 0.55, 0.09, 0.16, -0.13, 0.10, -0.08], yield_stress=1.50, vp_theta=(0.26, 0.18)
    )

    def _naive_plug_c1(ds: dict[str, Any], te_Z: np.ndarray) -> np.ndarray:
        taus = src01_gt["taus"]
        th = src01_gt["theta"]
        z = np.zeros((len(te_Z), 2), dtype=float)
        for s, dts in enumerate(ds["past_dt"]):
            dec = np.exp(-dts / taus)
            z = z * dec[None, :] + (1.0 - dec[None, :]) * ds["test"]["past_stress"][:, s : s + 1]
        et = ds["eval_times"]
        h0 = ds["test"]["stress_hold_0"]
        s0 = ds["test"]["strain_0"]
        return (
            s0[:, None]
            + th[0] * (h0 - z[:, 0])[:, None] * (1.0 - np.exp(-et / taus[0]))[None, :]
            + th[1] * (h0 - z[:, 1])[:, None] * (1.0 - np.exp(-et / taus[1]))[None, :]
            + (th[2] + te_Z @ th[3:7])[:, None] * et[None, :]
        )

    mae_plug_suc01_t = float(np.mean(np.abs(_naive_plug_c1(suc01_t_ds, suc01_t_gt["te_Z"]) - suc01_t_gt["te_clean"])))
    mae_plug_suc01_c = float(np.mean(np.abs(_naive_plug_c1(suc01_c_ds, suc01_c_gt["te_Z"]) - suc01_c_gt["te_clean"])))

    ns2: dict[str, Any] = {"np": np, "math": _math}
    exec(compile(_CHAIN2_CORE_GEN_PY, "<chain2_ctrl>", "exec"), ns2)
    _, src02_gt = ns2["_gen_chain2_study"](9331)
    suc02_t_st, suc02_t_gt = ns2["_gen_chain2_study"](
        99611, nx=48, ny=36, K_mu=12, M=16, vel_sign=-1.15, robin_pair=(0.75, 2.05), multi_budget=True
    )
    suc02_c_st, suc02_c_gt = ns2["_gen_chain2_study"](99621, nx=36, ny=32, K_mu=8, M=12, nonseparable_channel_b=True)

    w_pad_t = np.zeros(16, dtype=float)
    w_pad_t[:14] = src02_gt["w_opt"]
    rel_plug_suc02_t = float(
        np.linalg.norm(np.tile(w_pad_t, (3, 1)) - suc02_t_gt["weights_by_budget"]) / np.linalg.norm(suc02_t_gt["weights_by_budget"])
    )
    w_trunc_c = src02_gt["w_opt"][:12]
    rel_plug_suc02_c = float(np.linalg.norm(w_trunc_c - suc02_c_gt["w_opt"]) / np.linalg.norm(suc02_c_gt["w_opt"]))
    _ = (suc02_t_st, suc02_c_st)

    all_fail = bool(
        mae_plug_suc01_t > 0.15
        and mae_plug_suc01_c > 0.25
        and rel_plug_suc02_t > 0.30
        and rel_plug_suc02_c > 0.30
    )
    return {
        "direct_source_float_plugin_fails_all_successors": all_fail,
        "suc01_T_plugin_strain_mae_vs_tol_0_024": round(mae_plug_suc01_t, 6),
        "suc01_C_plugin_strain_mae_vs_tol_0_025": round(mae_plug_suc01_c, 6),
        "suc02_T_plugin_weights_rel_l2_vs_tol_1e4": round(rel_plug_suc02_t, 6),
        "suc02_C_plugin_weights_rel_l2_vs_tol_1e4": round(rel_plug_suc02_c, 6),
    }

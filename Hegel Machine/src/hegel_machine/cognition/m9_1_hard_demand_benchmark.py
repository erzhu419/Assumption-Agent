"""Hegel M9.1 Multi-Stage & 3D Scale-Bottleneck Scientific Benchmark (`Hegel-M91-HardDemand-2`).

Evaluated alongside longitudinal `M9.0a` tasks (`vr02`, `vr04`) under a tight symmetric `18`-turn primary budget (`0` skill
naming hints in prompts) to test `demand_triggered_zero_tax` (`0` Turn-1 manifest token tax + event-driven on-demand hint):

1. `hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline`:
   3-stage coupled multi-center clinical biomarker discovery & global optimality certification:
   - Stage 1: Multi-center (`C = 6` centers x `S = 3` strata) reference-control-panel affine alignment (`12%` missing `NaN`s)
     and `(center, stratum)` cell-balanced missing-channel imputation (`MAE <= 0.12`).
   - Stage 2: Cauchy-outlier-robust (`10%` heavy assay glitches) stratum-adjusted linear potency estimation (`rel L2 <= 0.08`).
   - Stage 3: Exact global symmetric-box (`[-M, M]^p`) L0-regularized ridge selection at `p in {20, 21, 22}` (`1.05M..4.19M`
     supports, positive root relaxation gap `> 0.40`) with polynomial-time proof-carrying leaf witnesses (`gap <= 1e-5`).
2. `hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography`:
   True 3D (`28 x 28 x 28 = 21,952 unknowns`) high-contrast (`4,000x`, `kappa in [0.02, 80.0]`) diffusion-reaction with
   series Robin dissipation on all 6 domain walls across `M = 8` 3D source modes (`16` 3D solves across `2` private instances,
   student solver wall-clock `< 14.0s`, relative PDE residual `<= 1e-7`) + SVD-whitened ridge nuisance amplitude profiling.
   (Unpreconditioned CG stalls at `rel_res ~ 0.11 >> 1e-7`, and 16 refactorizations via `spsolve` take `~31s > 14.0s`.)
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import (
    CLEAN_SCIPY_SANDBOX_IMAGE,
    HoldoutOODTask,
    ensure_clean_scipy_sandbox_image,
    run_holdout_private_verifier,
)


HARD_DEMAND_M91_TASKS: tuple[HoldoutOODTask, ...] = (
    # =========================================================================
    # HD01: 3-Stage Coupled Multi-Center Panel + Cauchy Fit + High-p Box-L0 Cert
    # =========================================================================
    HoldoutOODTask(
        task_id="hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline",
        short_name="hd01_multicenter_cauchy_certified_l0",
        wave="m91_hard_demand",
        scientific_domain="Multi-Center Assay Deconfounding, Robust Regression & Proof-Carrying L0 Certification",
        wormhole_family="fam_sparse_cert_and_combinatorial_pruning",
        declared_artifacts=("/holdout_task/multicenter_certified_l0.py",),
        declared_callable_entrypoints=("solve_multicenter_certified_l0",),
        instruction_text=(
            "Implement `/holdout_task/multicenter_certified_l0.py` exposing:\n"
            "`def solve_multicenter_certified_l0(study: dict) -> dict:`\n\n"
            "Three-Stage Coupled Multi-Center Deconfounding, Outlier-Robust Potency, & Certified Sparse Selection Contract:\n"
            "1. STAGE 1 (`Reference-Panel Scanner Alignment & Cell-Balanced Imputation` on `study['cohort']`):\n"
            "   - `study['cohort']` contains `'center_id'` (`shape (N,)`, `C = 6` centers), `'stratum'` (`shape (N,)`, `S = 3` strata `0, 1, 2`,\n"
            "     every `(center, stratum)` cell nonempty), `'X_raw'` (`shape (N, D)`, `D = 12` biomarkers with ~5% `NaN` missing entries),\n"
            "     `'controls'` (`shape (N, Q)`, `Q = 6` reference control standards with ~12% `NaN`s, drawn from `N(0, I_6)` in every center),\n"
            "     and `'y_potency'` (`shape (N,)`).\n"
            "   - In each center `c`, both `'X_raw'` and `'controls'` undergo `raw = scanner_offset[c] + scanner_gain[c] * latent` (`scanner_gain[c] > 0`).\n"
            "     Estimate `scanner_offset[c]` as the median across valid control columns (`>= 2` finite obs) of the column mean, and\n"
            "     `scanner_gain[c]` as the median across valid control columns of the column population std (`ddof=0`), then compute\n"
            "     `Z_aligned = (X_raw - scanner_offset[center_id]) / scanner_gain[center_id]`.\n"
            "   - Compute inverse-cell-frequency weights `w_cell` (`mean == 1.0`) giving equal total weight to each `(center, stratum)` cell,\n"
            "     and impute any `NaN` in `Z_aligned` with its `w_cell`-weighted finite column mean -> `'aligned_biomarkers'` (`shape (N, D)`, `MAE <= 0.12`).\n"
            "2. STAGE 2 (`Cauchy-Glitch-Robust Stratum-Adjusted Potency Fit`):\n"
            "   - `'y_potency'` obeys `b_0 + b_1*(stratum==1) + b_2*(stratum==2) + Z_true @ beta_potency + noise`,\n"
            "     corrupted on ~10% of rows by heavy Cauchy assay spikes (`+-6.0..11.0`)!\n"
            "   - Fit `[1, (stratum==1), (stratum==2), aligned_biomarkers]` against `'y_potency'` using Cauchy/Lorentzian IRLS\n"
            "     and return `'robust_potency_weights'` (`shape (D,)`, the `D` biomarker slopes, `rel L2 error <= 0.08`).\n"
            "3. STAGE 3 (`Proof-Carrying Global Symmetric-Box L0-Ridge Certification` on `study['sparse_selection']`):\n"
            "   - Given `X` (`shape (n_s, p)`, `p in [19, 22]` — up to `4,194,304` supports, blocking brute-force `2^p` enumeration!),\n"
            "     `y` (`shape (n_s,)`), `lam2 > 0`, `lam0 > 0`, and `M > 0`, globally minimize over `w in [-M, M]^p`:\n"
            "     `F(w) = 0.5 * ||X @ w - y||_2^2 + 0.5 * lam2 * ||w||_2^2 + lam0 * count(|w_j| > 1e-8)`.\n"
            "   - Return `'certified_selection'` dict containing:\n"
            "     - `'weights'`: `(p,)` in `[-M, M]^p`, `'objective'`: float `F(weights)`,\n"
            "     - `'certified_lower_bound'`: float `L` (`0 <= F(weights) - L <= 1e-5`),\n"
            "     - `'subcube_leaves'`: list (`1 <= len <= 2048`) partitioning `{0, 1}^p`, each leaf dict having:\n"
            "       * `'fixed_zero'`: length-`p` boolean sequence (`list[bool]` or `bool` `ndarray` of shape `(p,)`),\n"
            "       * `'fixed_active'`: length-`p` boolean sequence (`shape (p,)`, disjoint from `'fixed_zero'`),\n"
            "       * `'lb'`: finite float lower bound on `F(w)` over that subcube,\n"
            "       * AND EITHER `('minorant_diagonal', 'minorant_center')` (`shape (p,)` each, `d >= 0` with `X.T @ X - diag(d) >= 0`) OR `'dual_nu'` (`shape (n_s,)`).\n"
            "       (Minimal 1-leaf example: `[{'fixed_zero': [False]*p, 'fixed_active': [False]*p, 'lb': L, 'minorant_diagonal': d.tolist(), 'minorant_center': c.tolist()}]`.)\n"
            "- Return `{'aligned_biomarkers': (N, D), 'robust_potency_weights': (D,), 'certified_selection': dict}`.\n"
            "- Public development studies (with ground truth) are in `/holdout_task/public_multicenter_l0_studies.pkl`."
        ),
        public_setup_py=r'''
import math, pickle, numpy as np
from scipy.optimize import lsq_linear

def _make_study(seed, block_sizes):
    rng = np.random.default_rng(seed)
    D, Q = 12, 6
    offsets = np.array([3.9, -3.7, 4.2, -4.0, 3.5, -3.6], dtype=float)
    gains = np.array([1.38, 0.78, 1.44, 0.82, 1.28, 0.76], dtype=float)
    beta_pot = np.array([1.4, -1.2, 0.95, -0.85, 0.70, -0.65, 0.55, -0.50, 0.45, -0.40, 0.35, -0.30], dtype=float)
    strat_b = np.array([-0.9, 0.2, 1.1], dtype=float)

    cids, strata, Z_list, X_list, ctrl_list, y_list = [], [], [], [], [], []
    for c in range(6):
        n_c = 54
        s_base = np.array([0, 1, 2] * 5, dtype=int)
        s_rest = rng.choice([0, 1, 2], size=n_c - 15, p=[0.55, 0.30, 0.15])
        s = rng.permutation(np.concatenate([s_base, s_rest]))
        z_shift = 0.45 if (c % 2 == 0) else -0.45
        Z = rng.normal(z_shift, 1.0, size=(n_c, D))
        X_r = offsets[c] + gains[c] * Z
        X_r[rng.uniform(size=X_r.shape) < 0.05] = np.nan
        ctrl = offsets[c] + gains[c] * rng.normal(0.0, 1.0, size=(n_c, Q))
        ctrl[rng.uniform(size=ctrl.shape) < 0.12] = np.nan
        y_p = 0.4 + strat_b[s] + Z @ beta_pot + rng.normal(0.0, 0.04, size=n_c)
        sp = rng.choice(n_c, size=5, replace=False)
        y_p[sp] += rng.choice([-1.0, 1.0], size=5) * rng.uniform(6.5, 10.5, size=5)
        cids.append(np.full(n_c, c, dtype=int)); strata.append(s)
        Z_list.append(Z); X_list.append(X_r); ctrl_list.append(ctrl); y_list.append(y_p)

    # Stage 3 sparse selection problem
    n_per_block = 22
    B = len(block_sizes)
    n_tot, p_tot = B * n_per_block, sum(block_sizes)
    X_blk = np.zeros((n_tot, p_tot), dtype=float)
    y_blk = np.zeros(n_tot, dtype=float)
    lam2, lam0, M_box = 0.14, 0.22, 1.10
    r0 = c0 = 0
    for b_idx, pb in enumerate(block_sizes):
        Q_b, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(n_per_block, pb)), mode="reduced")
        tG = np.eye(pb, dtype=float)
        for j in range(pb - 1):
            sgn = 1.0 if ((j + b_idx) % 2 == 0) else -1.0
            tG[j, j + 1] = 0.26 * sgn; tG[j + 1, j] = 0.26 * sgn
        Xb = Q_b @ np.linalg.cholesky(tG).T
        wb = np.zeros(pb, dtype=float)
        wb[1] = 0.88 if (b_idx % 2 == 0) else -0.86
        wb[2] = 0.60 if (tG[1, 2] > 0) else -0.60
        if pb >= 7:
            wb[pb - 2] = -0.76 if (b_idx % 2 == 0) else 0.78
        yb = Xb @ wb + 0.48 * Xb[:, 0] - 0.44 * Xb[:, min(4, pb - 1)] + rng.normal(0.0, 0.05, size=n_per_block)
        X_blk[r0:r0+n_per_block, c0:c0+pb] = Xb; y_blk[r0:r0+n_per_block] = yb
        r0 += n_per_block; c0 += pb
    rp = rng.permutation(n_tot); cp = rng.permutation(p_tot)
    Xs = X_blk[rp][:, cp]; ys = y_blk[rp]
    exact_obj = 0.0
    r0 = c0 = 0
    sl2 = math.sqrt(lam2)
    for pb in block_sizes:
        Xb = X_blk[r0:r0+n_per_block, c0:c0+pb]; yb = y_blk[r0:r0+n_per_block]
        bv = 0.5 * float(yb @ yb)
        for mask in range(1, 1 << pb):
            idx = [j for j in range(pb) if (mask >> j) & 1]
            k = len(idx); Xsub = Xb[:, idx]
            ws = lsq_linear(np.vstack([Xsub, sl2 * np.eye(k)]), np.concatenate([yb, np.zeros(k)]), bounds=(-M_box, M_box), tol=1e-12).x
            rr = Xsub @ ws - yb
            val = 0.5 * float(rr @ rr) + 0.5 * lam2 * float(ws @ ws) + lam0 * k
            if val < bv: bv = val
        exact_obj += bv
        r0 += n_per_block; c0 += pb

    return {
        "study": {
            "cohort": {
                "center_id": np.concatenate(cids),
                "stratum": np.concatenate(strata),
                "X_raw": np.vstack(X_list),
                "controls": np.vstack(ctrl_list),
                "y_potency": np.concatenate(y_list),
            },
            "sparse_selection": {"X": Xs, "y": ys, "lam2": lam2, "lam0": lam0, "M": M_box},
        },
        "gt": {
            "Z_true": np.vstack(Z_list),
            "beta_potency": beta_pot,
            "exact_l0_objective": float(exact_obj),
        },
    }

pub = [_make_study(9101, (9, 10)), _make_study(9102, (10, 10))]
with open("/holdout_task/public_multicenter_l0_studies.pkl", "wb") as fh:
    pickle.dump(pub, fh)
''',
        private_verifier_py=r'''
import json, math, sys, time
import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, "/holdout_task")
import multicenter_certified_l0

def _is_real(val):
    if isinstance(val, (bool, np.bool_)):
        return False
    if not isinstance(val, (int, float, np.number)):
        return False
    return math.isfinite(float(val))

def _make_priv(seed, block_sizes):
    rng = np.random.default_rng(seed)
    D, Q = 12, 6
    offsets = np.array([4.1, -3.9, 4.3, -4.1, 3.6, -3.8], dtype=float)
    gains = np.array([1.42, 0.76, 1.46, 0.80, 1.32, 0.74], dtype=float)
    beta_pot = np.array([1.45, -1.25, 0.90, -0.80, 0.72, -0.66, 0.58, -0.52, 0.44, -0.38, 0.34, -0.28], dtype=float)
    strat_b = np.array([-0.95, 0.25, 1.15], dtype=float)
    cids, strata, Z_list, X_list, ctrl_list, y_list = [], [], [], [], [], []
    for c in range(6):
        n_c = 60
        s_base = np.array([0, 1, 2] * 5, dtype=int)
        s_rest = rng.choice([0, 1, 2], size=n_c - 15, p=[0.56, 0.29, 0.15])
        s = rng.permutation(np.concatenate([s_base, s_rest]))
        z_shift = 0.46 if (c % 2 == 0) else -0.46
        Z = rng.normal(z_shift, 1.0, size=(n_c, D))
        X_r = offsets[c] + gains[c] * Z
        X_r[rng.uniform(size=X_r.shape) < 0.05] = np.nan
        ctrl = offsets[c] + gains[c] * rng.normal(0.0, 1.0, size=(n_c, Q))
        ctrl[rng.uniform(size=ctrl.shape) < 0.12] = np.nan
        y_p = 0.35 + strat_b[s] + Z @ beta_pot + rng.normal(0.0, 0.04, size=n_c)
        sp = rng.choice(n_c, size=6, replace=False)
        y_p[sp] += rng.choice([-1.0, 1.0], size=6) * rng.uniform(6.8, 11.0, size=6)
        cids.append(np.full(n_c, c, dtype=int)); strata.append(s)
        Z_list.append(Z); X_list.append(X_r); ctrl_list.append(ctrl); y_list.append(y_p)

    n_per_block = 22
    B = len(block_sizes)
    n_tot, p_tot = B * n_per_block, sum(block_sizes)
    X_blk = np.zeros((n_tot, p_tot), dtype=float)
    y_blk = np.zeros(n_tot, dtype=float)
    lam2, lam0, M_box = 0.14, 0.22, 1.10
    r0 = c0 = 0
    for b_idx, pb in enumerate(block_sizes):
        Q_b, _ = np.linalg.qr(rng.normal(0.0, 1.0, size=(n_per_block, pb)), mode="reduced")
        tG = np.eye(pb, dtype=float)
        for j in range(pb - 1):
            sgn = 1.0 if ((j + b_idx) % 2 == 0) else -1.0
            tG[j, j + 1] = 0.26 * sgn; tG[j + 1, j] = 0.26 * sgn
        Xb = Q_b @ np.linalg.cholesky(tG).T
        wb = np.zeros(pb, dtype=float)
        wb[1] = 0.88 if (b_idx % 2 == 0) else -0.86
        wb[2] = 0.60 if (tG[1, 2] > 0) else -0.60
        if pb >= 7:
            wb[pb - 2] = -0.76 if (b_idx % 2 == 0) else 0.78
        yb = Xb @ wb + 0.48 * Xb[:, 0] - 0.44 * Xb[:, min(4, pb - 1)] + rng.normal(0.0, 0.05, size=n_per_block)
        X_blk[r0:r0+n_per_block, c0:c0+pb] = Xb; y_blk[r0:r0+n_per_block] = yb
        r0 += n_per_block; c0 += pb
    rp = rng.permutation(n_tot); cp = rng.permutation(p_tot)
    Xs = X_blk[rp][:, cp]; ys = y_blk[rp]
    exact_obj = 0.0
    r0 = c0 = 0
    sl2 = math.sqrt(lam2)
    for pb in block_sizes:
        Xb = X_blk[r0:r0+n_per_block, c0:c0+pb]; yb = y_blk[r0:r0+n_per_block]
        bv = 0.5 * float(yb @ yb)
        for mask in range(1, 1 << pb):
            idx = [j for j in range(pb) if (mask >> j) & 1]
            k = len(idx); Xsub = Xb[:, idx]
            ws = lsq_linear(np.vstack([Xsub, sl2 * np.eye(k)]), np.concatenate([yb, np.zeros(k)]), bounds=(-M_box, M_box), tol=1e-12).x
            rr = Xsub @ ws - yb
            val = 0.5 * float(rr @ rr) + 0.5 * lam2 * float(ws @ ws) + lam0 * k
            if val < bv: bv = val
        exact_obj += bv
        r0 += n_per_block; c0 += pb

    X_raw_full = np.vstack(X_list)
    obs_mask = np.isfinite(X_raw_full)
    return {
        "cohort": {
            "center_id": np.concatenate(cids),
            "stratum": np.concatenate(strata),
            "X_raw": X_raw_full,
            "controls": np.vstack(ctrl_list),
            "y_potency": np.concatenate(y_list),
        },
        "sparse_selection": {"X": Xs, "y": ys, "lam2": lam2, "lam0": lam0, "M": M_box},
    }, np.vstack(Z_list), obs_mask, beta_pot, float(exact_obj)

def _verify_cert(prob, out, exact_opt):
    if not isinstance(out, dict) or not {"weights", "objective", "certified_lower_bound", "subcube_leaves"} <= set(out.keys()):
        return False, "missing_cert_keys", 1e9
    X_arr, y_arr = prob["X"], prob["y"]
    n, p = X_arr.shape
    M, ridge, rho = float(prob["M"]), float(prob["lam2"]), float(prob["lam0"])
    if not (_is_real(out["objective"]) and _is_real(out["certified_lower_bound"])):
        return False, "bad_scalars", 1e9
    rep_obj, cert_lb = float(out["objective"]), float(out["certified_lower_bound"])
    w = np.asarray(out["weights"], dtype=float)
    if w.shape != (p,) or not np.all(np.isfinite(w)) or np.min(w) < -M - 1e-7 or np.max(w) > M + 1e-7:
        return False, "bad_weights", 1e9
    wc = np.clip(w, -M, M); sup = np.abs(wc) > 1e-8; we = np.where(sup, wc, 0.0)
    rr = X_arr @ we - y_arr
    true_obj = 0.5 * float(rr @ rr) + 0.5 * ridge * float(we @ we) + rho * int(np.sum(sup))
    if abs(rep_obj - true_obj) > 1e-6 * max(1.0, abs(true_obj)) or true_obj > exact_opt + 1e-5:
        return False, "primal_suboptimal_or_inconsistent", true_obj - exact_opt
    leaves = out["subcube_leaves"]
    if not isinstance(leaves, list) or not (1 <= len(leaves) <= 2048):
        return False, "bad_leaf_count", 1e9
    L = len(leaves)
    z_masks, a_masks, parsed = [], [], []
    meas = 0
    for lf in leaves:
        if not isinstance(lf, dict) or not {"fixed_zero", "fixed_active", "lb"} <= set(lf.keys()) or not _is_real(lf["lb"]):
            return False, "bad_leaf", 1e9
        fz = np.asarray(lf["fixed_zero"], dtype=bool); fa = np.asarray(lf["fixed_active"], dtype=bool)
        if fz.shape != (p,) or fa.shape != (p,) or bool(np.any(fz & fa)):
            return False, "bad_masks", 1e9
        zm = am = 0
        for j in range(p):
            if fz[j]: zm |= (1 << j)
            elif fa[j]: am |= (1 << j)
        z_masks.append(zm); a_masks.append(am)
        meas += 1 << (p - int(np.sum(fz) + np.sum(fa)))
        parsed.append((fz, fa, float(lf["lb"]), lf))
    zu = np.array(z_masks, dtype=np.uint64); au = np.array(a_masks, dtype=np.uint64)
    sep = (zu[:, None] & au[None, :]) | (au[:, None] & zu[None, :])
    np.fill_diagonal(sep, np.uint64(1))
    if bool(np.any(sep == 0)) or meas != (1 << p):
        return False, "invalid_partition", 1e9
    G_mat = X_arr.T @ X_arr
    sc_eig = max(1.0, float(np.max(np.diag(G_mat))))
    psd_cache = {}
    v_lbs = []
    for fz, fa, clb, rlf in parsed:
        free = ~(fz | fa)
        vlb = float("-inf"); ok_w = False
        if rlf.get("minorant_diagonal") is not None and rlf.get("minorant_center") is not None:
            dv = np.asarray(rlf["minorant_diagonal"], dtype=float).ravel()
            cv = np.asarray(rlf["minorant_center"], dtype=float).ravel()
            if dv.shape == (p,) and cv.shape == (p,) and np.all(np.isfinite(dv)) and np.all(np.isfinite(cv)) and np.all(dv >= -1e-12):
                dc_clean = np.maximum(dv, 0.0); dk = dc_clean.tobytes()
                if dk not in psd_cache:
                    Gd = G_mat - np.diag(dc_clean)
                    me = float(np.linalg.eigvalsh(0.5 * (Gd + Gd.T))[0])
                    sc = max(sc_eig, float(np.max(dc_clean)))
                    psd_cache[dk] = (me >= -1e-8 * sc, max(0.0, -me) + 2e-15 * sc if me < 0.0 else 0.0)
                is_p, eg = psd_cache[dk]
                if is_p:
                    pslack = 0.5 * eg * float(np.sum((M + np.abs(cv)) ** 2)) if eg > 0.0 else 0.0
                    rv = y_arr - (X_arr @ cv); gv = X_arr.T @ rv; dc = dc_clean * cv
                    ct = 0.5 * float(rv @ rv) + float(gv @ cv) + 0.5 * float(dc @ cv)
                    zv = np.abs(gv + dc); crv = ridge + dc_clean
                    gains = np.where((crv > 0.0) & (zv < M * crv), 0.5 * zv * zv / np.maximum(crv, 1e-30), M * zv - 0.5 * crv * M * M)
                    onv = rho - gains
                    vlb = float(ct + np.sum(onv[fa]) + np.sum(np.minimum(onv[free], 0.0)) - pslack)
                    ok_w = math.isfinite(vlb)
        elif rlf.get("dual_nu") is not None:
            nu = np.asarray(rlf["dual_nu"], dtype=float).ravel()
            if nu.shape == (n,) and np.all(np.isfinite(nu)):
                al = np.abs(X_arr.T @ nu)
                phi = np.where(al < M * ridge, 0.5 * al * al / ridge, M * al - 0.5 * ridge * M * M)
                ond = rho - phi
                vlb = float(-0.5 * float(nu @ nu) + float(nu @ y_arr) + np.sum(ond[fa]) + np.sum(np.minimum(ond[free], 0.0)))
                ok_w = math.isfinite(vlb)
        if (not ok_w) or (clb > vlb + 1e-6 * max(1.0, abs(vlb))):
            return False, "unsound_witness", 1e9
        v_lbs.append(vlb)
    if cert_lb > min(x[2] for x in parsed) + 1e-6:
        return False, "cert_lb_gt_min_leaf", 1e9
    gap = true_obj - min(cert_lb, min(v_lbs))
    if not (-1e-6 <= (true_obj - cert_lb) <= 1e-5 and gap <= 1.2e-5):
        return False, f"gap_too_large({gap:.6e})", gap
    return True, "ok", gap

student_elapsed = 0.0
valid = True
max_mae = 0.0
max_w_err = 0.0
max_gap = 0.0
notes = []
instance_diagnostics = []
for sd, bsz in ((99301, (10, 10)), (99302, (10, 11)), (99303, (11, 11))):
    study, gt_Z, obs_m, gt_beta, exact_opt = _make_priv(sd, bsz)
    tc0 = time.monotonic()
    out = multicenter_certified_l0.solve_multicenter_certified_l0(study)
    el_i = time.monotonic() - tc0
    student_elapsed += el_i
    if not isinstance(out, dict) or not {"aligned_biomarkers", "robust_potency_weights", "certified_selection"} <= set(out.keys()):
        valid = False
        notes.append(f"{sd}:missing_keys")
        instance_diagnostics.append({"seed": sd, "ok": False, "reason": "missing_keys", "elapsed_sec": round(float(el_i), 4)})
        continue
    Za = np.asarray(out["aligned_biomarkers"], dtype=float)
    bw = np.asarray(out["robust_potency_weights"], dtype=float)
    if Za.shape != gt_Z.shape or bw.shape != gt_beta.shape or not (np.all(np.isfinite(Za)) and np.all(np.isfinite(bw))):
        valid = False
        notes.append(f"{sd}:shape_or_nan")
        instance_diagnostics.append({"seed": sd, "ok": False, "reason": "shape_or_nan", "elapsed_sec": round(float(el_i), 4)})
        continue
    mae_i = float(np.mean(np.abs(Za[obs_m] - gt_Z[obs_m])))
    w_err_i = float(np.linalg.norm(bw - gt_beta) / np.linalg.norm(gt_beta))
    max_mae = max(max_mae, mae_i)
    max_w_err = max(max_w_err, w_err_i)
    ok_c, note_c, gap_c = _verify_cert(study["sparse_selection"], out["certified_selection"], exact_opt)
    notes.append(f"{sd}:{note_c}")
    if gap_c < 1e6:
        max_gap = max(max_gap, float(gap_c))
    else:
        max_gap = max(max_gap, 1e9)
    instance_diagnostics.append({
        "seed": sd,
        "ok": bool(ok_c and mae_i <= 0.12 and w_err_i <= 0.08),
        "cert_status": note_c,
        "aligned_mae": round(mae_i, 6),
        "potency_rel_l2": round(w_err_i, 6),
        "certified_gap": (round(float(gap_c), 9) if gap_c < 1e6 else -1.0),
        "elapsed_sec": round(float(el_i), 4),
    })
    if not ok_c:
        valid = False

passed = bool(valid and max_mae <= 0.12 and max_w_err <= 0.08 and max_gap <= 1.2e-5 and student_elapsed < 22.0)
print(json.dumps({
    "strict_pass": passed,
    "contract_version": "hd01_explicit_bool_mask_v2",
    "max_aligned_mae": round(float(max_mae), 6),
    "max_potency_weight_rel_l2": round(float(max_w_err), 6),
    "max_certified_gap": (round(float(max_gap), 9) if max_gap < 1e6 else -1.0),
    "status_notes": notes,
    "instance_diagnostics": instance_diagnostics,
    "student_elapsed_sec": round(float(student_elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # HD02: True 3D 21,952-Unknowns High-Contrast Robin AMG-CG + Nuisance Tomography
    # =========================================================================
    HoldoutOODTask(
        task_id="hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography",
        short_name="hd02_3d_robin_amg_tomography",
        wave="m91_hard_demand",
        scientific_domain="3D High-Contrast Algebraic Multigrid Elliptic PDE & Nuisance-Profiled Source Tomography",
        wormhole_family="fam_multiscale_pde_and_robust_assimilation",
        declared_artifacts=("/holdout_task/robin3d_amg_tomography.py",),
        declared_callable_entrypoints=("solve_3d_robin_tomography",),
        instruction_text=(
            "Implement `/holdout_task/robin3d_amg_tomography.py` exposing:\n"
            "`def solve_3d_robin_tomography(problem: dict) -> dict:`\n\n"
            "True 3D (`28 x 28 x 28 = 21,952 unknowns`) High-Contrast (`4,000x`) Robin Elliptic PDE & Profiled Tomography Specification:\n"
            "- Steady 3D isotropic high-contrast (`kappa in [0.02, 80.0]`) diffusion-reaction on `[0, Lx] x [0, Ly] x [0, Lz]`\n"
            "  with uniform cell-centered 3D grid `(nx, ny, nz) = (28, 28, 28)` (`21,952 unknowns`, `hx = Lx/nx`, `hy = Ly/ny`, `hz = Lz/nz`, `vol = hx*hy*hz`):\n"
            "  `-div(kappa(x, y, z) grad u) + sigma(x, y, z) * u = q(x, y, z)`\n"
            "  subject to homogeneous external Robin dissipation `(-kappa grad u) . n = gamma_robin * u` on all SIX exterior cube faces.\n"
            "- Exact 3D cell-centered finite-volume stencil (`21,952 x 21,952` SPD matrix `A` in C-order `ravel()`):\n"
            "  1. Interior faces perpendicular to axis `a in {0, 1, 2}` (step `h_a`, face area `area_a = vol / h_a`) use harmonic face conductivity\n"
            "     `k_f = 2 * k_0 * k_1 / (k_0 + k_1)`, contributing `(area_a / h_a) * k_f * (u_0 - u_1)`.\n"
            "  2. Exterior boundary faces (`index == 0` or `index == n_a - 1` along each axis `a in {0, 1, 2}`) use series half-cell Robin conductance:\n"
            "     contributing `area_a * (gamma_robin * k_b) / (gamma_robin * 0.5 * h_a + k_b) * u_b` to the diagonal!\n"
            "     (WARNING: zero-ghost Dirichlet assembly violates Robin boundary conditions!)\n"
            "  3. Cell reaction + source balance: `(face outflows) + vol * sigma[i,j,k] * u[i,j,k] = vol * q[i,j,k]`.\n"
            "- COMPUTATIONAL SCALE & CONDITIONING WARNING (`21,952 unknowns` in 3D, `16` total 3D solves across `2` held-out problems, `< 14.0s` solver budget):\n"
            "  - Calling `scipy.sparse.linalg.spsolve(A, rhs)` repeatedly inside the `M = 8` RHS loop refactors 3D sparse LU 16 times (`~31s > 14.0s` timeout!),\n"
            "    and unpreconditioned / Jacobi conjugate gradient stalls at `~1e-1` relative residual on `4,000x` conductivity contrast!\n"
            "  - Build a reusable algebraic multigrid / smoothed-aggregation preconditioner + residual-checked CG (or reuse a single sparse factorization)\n"
            "    so all `M = 8` `basis_fields` (`shape (8, 28, 28, 28)`) satisfy relative PDE residual `||A u_m - b_m||_2 / ||b_m||_2 <= 1e-7` within `< 14.0s`!\n"
            "- Given `problem` keys `('Lx', 'Ly', 'Lz', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ijk', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs')`:\n"
            "  - Restrict `basis_fields` to `sensor_ijk` (`shape (K_s, 3)`) -> `S` (`shape (K_s, M)`), with `Z = nuisance_design` (`shape (K_s, P)`),\n"
            "    `lam_z = nuisance_ridge`, `lam_s = source_ridge`, and `Y = Y_obs` (`shape (K_s, Q_exp)`).\n"
            "  - Return:\n"
            "    - `'basis_fields'`: `(M, nx, ny, nz)` (`rel PDE residual <= 1e-7`),\n"
            "    - `'single_source_profiled_amplitudes'`: `(M, Q_exp)` minimizing `0.5*||Z @ gamma + alpha_m * S[:, m] - Y[:, q]||_2^2 + 0.5*lam_z*||gamma||_2^2` (unpenalized `alpha_m`),\n"
            "    - `'joint_source_weights'`: `(M, Q_exp)` minimizing `0.5*||Z @ gamma + S @ w - Y[:, q]||_2^2 + 0.5*lam_z*||gamma||_2^2 + 0.5*lam_s*||w||_2^2`.\n"
            "- Public 3D development case (`20 x 20 x 20` for fast interactive inspection) is in `/holdout_task/public_robin3d_cases.pkl`."
        ),
        public_setup_py=r'''
import pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

def _assemble_3d_robin(Lx, Ly, Lz, kappa, sigma, gamma_robin):
    nx, ny, nz = kappa.shape
    steps = (Lx / nx, Ly / ny, Lz / nz)
    vol = steps[0] * steps[1] * steps[2]
    N = kappa.size
    ids = np.arange(N).reshape(nx, ny, nz)
    diag = vol * sigma.ravel().astype(float)
    rows, cols, data = [], [], []
    for axis, h_a in enumerate(steps):
        area_a = vol / h_a
        i0 = np.take(ids, np.arange(kappa.shape[axis] - 1), axis=axis).ravel()
        i1 = np.take(ids, np.arange(1, kappa.shape[axis]), axis=axis).ravel()
        k0 = kappa.ravel()[i0]; k1 = kappa.ravel()[i1]
        w = (area_a / h_a) * (2.0 * k0 * k1 / (k0 + k1))
        diag[i0] += w; diag[i1] += w
        rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-w, -w])
        for end in (0, -1):
            ib = np.take(ids, end, axis=axis).ravel()
            kb = kappa.ravel()[ib]
            diag[ib] += area_a * (gamma_robin * kb) / (gamma_robin * 0.5 * h_a + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _make_3d_case(seed, n=20):
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz = 1.2, 1.1, 1.0
    hx, hy, hz = Lx / n, Ly / n, Lz / n
    vol = hx * hy * hz
    xs = (np.arange(n) + 0.5) * hx; ys = (np.arange(n) + 0.5) * hy; zs = (np.arange(n) + 0.5) * hz
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    log_k = 1.7 * np.sin(2.0 * np.pi * XX / Lx) * np.cos(2.0 * np.pi * YY / Ly) * np.sin(1.5 * np.pi * ZZ / Lz)
    kappa = np.clip(np.exp(log_k), 0.03, 12.0)
    inc = ((XX - 0.45*Lx)**2 + (YY - 0.52*Ly)**2 + (ZZ - 0.48*Lz)**2 < 0.05)
    kappa[inc] = 80.0
    kappa[(XX < 0.25*Lx) & (YY > 0.70*Ly) & (ZZ < 0.40*Lz)] = 0.02
    sigma = 0.05 + 0.03 * (XX / Lx + YY / Ly + ZZ / Lz)
    gamma_robin = 1.50
    M = 8
    ctrs = rng.uniform([0.18*Lx, 0.18*Ly, 0.18*Lz], [0.82*Lx, 0.82*Ly, 0.82*Lz], size=(M, 3))
    cand = np.zeros((M, n, n, n), dtype=float)
    for m in range(M):
        r2 = (XX - ctrs[m,0])**2 + (YY - ctrs[m,1])**2 + (ZZ - ctrs[m,2])**2
        cand[m] = np.exp(-r2 / (2.0 * (0.12**2)))
    A = _assemble_3d_robin(Lx, Ly, Lz, kappa, sigma, gamma_robin)
    lu = splu(A.tocsc())
    B_rhs = vol * cand.reshape(M, -1).T
    bf = lu.solve(B_rhs).T.reshape(M, n, n, n)
    K_s = 320
    fc = rng.choice(n**3, size=K_s, replace=False)
    s_ijk = np.column_stack([fc // (n*n), (fc // n) % n, fc % n])
    sx, sy, sz = xs[s_ijk[:,0]], ys[s_ijk[:,1]], zs[s_ijk[:,2]]
    Z = np.column_stack([np.ones(K_s), sx/Lx - 0.5, sy/Ly - 0.5, sz/Lz - 0.5, np.sin(np.pi*sx/Lx)*np.cos(np.pi*sy/Ly)])
    S = bf[:, s_ijk[:,0], s_ijk[:,1], s_ijk[:,2]].T
    lam_z, lam_s = 1.5e-3, 2.5e-2
    Q_exp = 4
    Y_obs = Z @ rng.normal(0.0, 0.6, size=(5, Q_exp)) + S @ rng.normal(0.0, 0.9, size=(M, Q_exp)) + rng.normal(0.0, 2e-3, size=(K_s, Q_exp))
    Pz = np.linalg.solve(Z.T @ Z + lam_z * np.eye(5), Z.T)
    Ry = Y_obs - Z @ (Pz @ Y_obs); Rs = S - Z @ (Pz @ S)
    prof = (S.T @ Ry) / np.sum(S * Rs, axis=0)[:, None]
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(5), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(M)]])
    jw = np.linalg.solve(H_aug, np.vstack([Z.T @ Y_obs, S.T @ Y_obs]))[5:, :]
    return {
        "problem": {
            "Lx": Lx, "Ly": Ly, "Lz": Lz, "kappa": kappa, "sigma": sigma, "gamma_robin": gamma_robin,
            "candidate_sources": cand, "sensor_ijk": s_ijk, "nuisance_design": Z,
            "nuisance_ridge": lam_z, "source_ridge": lam_s, "Y_obs": Y_obs,
        },
        "gt": {"basis_fields": bf, "single_source_profiled_amplitudes": prof, "joint_source_weights": jw},
    }

pub = [_make_3d_case(9201, n=20)]
with open("/holdout_task/public_robin3d_cases.pkl", "wb") as fh:
    pickle.dump(pub, fh)
''',
        private_verifier_py=r'''
import json, math, sys, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

sys.path.insert(0, "/holdout_task")
import robin3d_amg_tomography

def _assemble_3d_robin(Lx, Ly, Lz, kappa, sigma, gamma_robin):
    nx, ny, nz = kappa.shape
    steps = (Lx / nx, Ly / ny, Lz / nz)
    vol = steps[0] * steps[1] * steps[2]
    N = kappa.size
    ids = np.arange(N).reshape(nx, ny, nz)
    diag = vol * sigma.ravel().astype(float)
    rows, cols, data = [], [], []
    for axis, h_a in enumerate(steps):
        area_a = vol / h_a
        i0 = np.take(ids, np.arange(kappa.shape[axis] - 1), axis=axis).ravel()
        i1 = np.take(ids, np.arange(1, kappa.shape[axis]), axis=axis).ravel()
        k0 = kappa.ravel()[i0]; k1 = kappa.ravel()[i1]
        w = (area_a / h_a) * (2.0 * k0 * k1 / (k0 + k1))
        diag[i0] += w; diag[i1] += w
        rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-w, -w])
        for end in (0, -1):
            ib = np.take(ids, end, axis=axis).ravel()
            kb = kappa.ravel()[ib]
            diag[ib] += area_a * (gamma_robin * kb) / (gamma_robin * 0.5 * h_a + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _make_priv_3d(seed, n=28):
    rng = np.random.default_rng(seed)
    Lx, Ly, Lz = 1.25, 1.15, 1.05
    hx, hy, hz = Lx / n, Ly / n, Lz / n
    vol = hx * hy * hz
    xs = (np.arange(n) + 0.5) * hx; ys = (np.arange(n) + 0.5) * hy; zs = (np.arange(n) + 0.5) * hz
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    log_k = 1.8 * np.cos(2.0 * np.pi * XX / Lx) * np.sin(2.0 * np.pi * YY / Ly) * np.cos(1.5 * np.pi * ZZ / Lz)
    kappa = np.clip(np.exp(log_k), 0.025, 14.0)
    inc = ((XX - 0.48*Lx)**2 + (YY - 0.50*Ly)**2 + (ZZ - 0.52*Lz)**2 < 0.055)
    kappa[inc] = 80.0
    kappa[(XX > 0.72*Lx) & (YY < 0.30*Ly) & (ZZ > 0.62*Lz)] = 0.02
    sigma = 0.05 + 0.035 * (XX / Lx + YY / Ly + ZZ / Lz)
    gamma_robin = 1.40
    M = 8
    ctrs = rng.uniform([0.18*Lx, 0.18*Ly, 0.18*Lz], [0.82*Lx, 0.82*Ly, 0.82*Lz], size=(M, 3))
    cand = np.zeros((M, n, n, n), dtype=float)
    for m in range(M):
        r2 = (XX - ctrs[m,0])**2 + (YY - ctrs[m,1])**2 + (ZZ - ctrs[m,2])**2
        cand[m] = np.exp(-r2 / (2.0 * (0.115**2)))
    A = _assemble_3d_robin(Lx, Ly, Lz, kappa, sigma, gamma_robin)
    lu = splu(A.tocsc())
    B_rhs = vol * cand.reshape(M, -1).T
    bf = lu.solve(B_rhs).T.reshape(M, n, n, n)
    K_s = 480
    fc = rng.choice(n**3, size=K_s, replace=False)
    s_ijk = np.column_stack([fc // (n*n), (fc // n) % n, fc % n])
    sx, sy, sz = xs[s_ijk[:,0]], ys[s_ijk[:,1]], zs[s_ijk[:,2]]
    Z = np.column_stack([np.ones(K_s), sx/Lx - 0.5, sy/Ly - 0.5, sz/Lz - 0.5, np.sin(np.pi*sx/Lx)*np.cos(np.pi*sy/Ly)])
    S = bf[:, s_ijk[:,0], s_ijk[:,1], s_ijk[:,2]].T
    lam_z, lam_s = 1.5e-3, 2.5e-2
    Q_exp = 4
    Y_obs = Z @ rng.normal(0.0, 0.6, size=(5, Q_exp)) + S @ rng.normal(0.0, 0.9, size=(M, Q_exp)) + rng.normal(0.0, 2e-3, size=(K_s, Q_exp))
    Pz = np.linalg.solve(Z.T @ Z + lam_z * np.eye(5), Z.T)
    Ry = Y_obs - Z @ (Pz @ Y_obs); Rs = S - Z @ (Pz @ S)
    prof = (S.T @ Ry) / np.sum(S * Rs, axis=0)[:, None]
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(5), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(M)]])
    jw = np.linalg.solve(H_aug, np.vstack([Z.T @ Y_obs, S.T @ Y_obs]))[5:, :]
    return {
        "Lx": Lx, "Ly": Ly, "Lz": Lz, "kappa": kappa, "sigma": sigma, "gamma_robin": gamma_robin,
        "candidate_sources": cand, "sensor_ijk": s_ijk, "nuisance_design": Z,
        "nuisance_ridge": lam_z, "source_ridge": lam_s, "Y_obs": Y_obs,
    }, A, vol, bf, prof, jw

student_elapsed = 0.0
valid = True
max_pde_res = 0.0
max_field_err = 0.0
max_prof_err = 0.0
max_joint_err = 0.0

for sd in (99311, 99312):
    prob, A, vol, gt_bf, gt_prof, gt_jw = _make_priv_3d(sd, n=28)
    tc0 = time.monotonic()
    out = robin3d_amg_tomography.solve_3d_robin_tomography({k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in prob.items()})
    student_elapsed += time.monotonic() - tc0
    if not isinstance(out, dict) or not {"basis_fields", "single_source_profiled_amplitudes", "joint_source_weights"} <= set(out.keys()):
        valid = False; break
    bf = np.asarray(out["basis_fields"], dtype=float)
    pa = np.asarray(out["single_source_profiled_amplitudes"], dtype=float)
    jw = np.asarray(out["joint_source_weights"], dtype=float)
    if bf.shape != gt_bf.shape or pa.shape != gt_prof.shape or jw.shape != gt_jw.shape:
        valid = False; break
    if not (np.all(np.isfinite(bf)) and np.all(np.isfinite(pa)) and np.all(np.isfinite(jw))):
        valid = False; break
    for m in range(gt_bf.shape[0]):
        rhs = vol * prob["candidate_sources"][m].ravel()
        res_rel = float(np.linalg.norm(A @ bf[m].ravel() - rhs) / max(np.linalg.norm(rhs), 1e-12))
        max_pde_res = max(max_pde_res, res_rel)
    max_field_err = max(max_field_err, float(np.linalg.norm(bf - gt_bf) / np.linalg.norm(gt_bf)))
    max_prof_err = max(max_prof_err, float(np.linalg.norm(pa - gt_prof) / np.linalg.norm(gt_prof)))
    max_joint_err = max(max_joint_err, float(np.linalg.norm(jw - gt_jw) / np.linalg.norm(gt_jw)))

passed = bool(
    valid
    and max_pde_res <= 1e-7
    and max_field_err <= 1e-5
    and max_prof_err <= 1e-5
    and max_joint_err <= 1e-5
    and student_elapsed < 14.0
)
print(json.dumps({
    "strict_pass": passed,
    "all_valid": valid,
    "max_pde_rel_residual": round(float(max_pde_res), 10),
    "max_field_rel_l2": round(float(max_field_err), 10),
    "max_profiled_amp_rel_l2": round(float(max_prof_err), 10),
    "max_joint_weight_rel_l2": round(float(max_joint_err), 10),
    "student_elapsed_sec": round(float(student_elapsed), 4),
}, allow_nan=False))
''',
    ),
)


HARD_DEMAND_M91_BY_ID: dict[str, HoldoutOODTask] = {t.task_id: t for t in HARD_DEMAND_M91_TASKS}


def hard_demand_task_to_frontier_task(ht: HoldoutOODTask) -> m6f.FrontierBenchmarkTask:
    ensure_clean_scipy_sandbox_image()
    return m6f.FrontierBenchmarkTask(
        task_id=ht.task_id,
        short_name=ht.short_name,
        benchmark="Hegel-M91-HardDemand-2",
        wave=ht.wave,
        scientific_domain=ht.scientific_domain,
        wormhole_family=ht.wormhole_family,
        instruction_text=ht.instruction_text,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        declared_artifacts=ht.declared_artifacts,
        declared_callable_entrypoints=ht.declared_callable_entrypoints,
        holdout_public_setup_py=ht.public_setup_py,
        agent_cpus=2,
        agent_memory_mb=2560,
        verifier_cpus=2,
        verifier_memory_mb=2560,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
    )


def get_hard_demand_m91_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [hard_demand_task_to_frontier_task(t) for t in HARD_DEMAND_M91_TASKS]


def run_hard_demand_private_verifier_by_id(task_id: str, submitted_tar_path: Path | None) -> dict[str, Any]:
    ht = HARD_DEMAND_M91_BY_ID[task_id]
    rep = run_holdout_private_verifier(ht, submitted_tar_path)
    rep["benchmark"] = "Hegel-M91-HardDemand-2"
    return rep


def run_hard_demand_private_verifier_for_container(task_id: str, agent_cid: str) -> dict[str, Any]:
    ht = HARD_DEMAND_M91_BY_ID[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_hd_m91_ver_") as td:
        tar_p = Path(td) / "submitted.tar"
        m6f.export_declared_artifacts_tar(agent_cid, ht.declared_artifacts, tar_p)
        rep = run_holdout_private_verifier(ht, tar_p)
        rep["benchmark"] = "Hegel-M91-HardDemand-2"
        return rep

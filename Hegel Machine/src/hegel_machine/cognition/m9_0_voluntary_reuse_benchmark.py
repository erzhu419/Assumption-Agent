"""Hegel M9.0a Controlled Voluntary Executable Reuse & Knowledge-Isolation Benchmark (`Hegel-M90a-VoluntaryReuse-4`).

Designed strictly after `M8.9.1` (`Commit 1: be2a3b29fb978f19d94b9b9b8a19956ba96d3857`) on frozen `v5` (`6` qualified `L1`
modules, `17` functions) with ZERO skill/module naming hints in prompts and 4 stratified applicability regimes:

1. `vr01__proof_carrying_box_l0_minorant_bnb` (`Exact Subproblem Fit + Non-Trivial Search`):
   Symmetric-box (`[-M, M]^p`) L0-regularized ridge regression at `p in {22, 23, 24}` (`4.19M .. 16.78M` supports, blocking
   brute-force 2^p enumeration) with strictly positive root relaxation gap and polynomial-time proof-carrying leaf witnesses
   (`minorant_diagonal` + `minorant_center` OR Fenchel perspective `dual_nu`) + exact `O(L^2)` bitmask partition verification.
2. `vr02__large_robin_spd_preconditioned_cg_multi_rhs` (`Partial Subroutine Fit + >4,000-DOF Iterative PDE`):
   `72 x 72 = 5,184`-unknowns high-contrast (`4` orders of magnitude) cell-centered diffusion-reaction with series Robin
   boundary conditions (mismatching zero-ghost Dirichlet assembly while matching AMG/Galerkin preconditioning + residual-checked
   CG across `10` candidate source modes and SVD-whitened ridge nuisance amplitude profiling).
3. `vr03__prehistory_viscoelastic_control_panel_calibration` (`Temporal Pre-Window History + Grouped Control Panel Composition`):
   Multi-center viscoelastic rheometry cohort (`C = 6` centers x `S = 3` strata, Simpson's paradox + missing `NaN` entries)
   combining non-zero pre-window hereditary strain history + Cauchy telemetry spike de-spiking with multi-column reference
   control panel alignment and cell-balanced logistic risk calibration.
4. `vr04__indefinite_helmholtz_wrapped_phase_inversion` (`Precondition-Violated Negative Control`):
   2D complex symmetric indefinite acoustic Helmholtz scattering (`k0 = 11.5`, strictly indefinite real part + absorbing
   Sommerfeld radiation boundary, violating all real/SPD/CG preconditions) with wrapped modulo-`2*pi` receiver phase telemetry
   and outlier spikes.
"""

from __future__ import annotations

import math
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


def _is_valid_real_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return False
    if not isinstance(val, (int, float, np.number)):
        return False
    return math.isfinite(float(val))


def verify_symmetric_box_l0_proof_certificate(
    *,
    X: np.ndarray,
    y: np.ndarray,
    lam2: float,
    lam0: float,
    M_box: float,
    weights: Any,
    reported_objective: Any,
    certified_lower_bound: Any,
    subcube_leaves: Any,
    gap_tol: float = 1e-5,
    obj_tol: float = 1e-6,
    leaf_soundness_tol: float = 1e-6,
    max_leaves: int = 2048,
) -> dict[str, Any]:
    """Sound polynomial-time verifier for symmetric-box `[-M, M]^p` L0-regularized ridge certificates:
    - Exact `O(L^2)` bitmask disjointness + exact big-integer `sum 2^(p - |Z_r| - |A_r|) == 2^p` completeness (`1 <= L <= max_leaves`).
    - Exact `O(n*p + p^3)` proof-carrying leaf lower-bound check via EITHER:
      (a) PSD diagonal minorant witness `('minorant_diagonal', 'minorant_center')` checking `X.T @ X - diag(d) >= 0`, OR
      (b) Fenchel perspective dual witness `'dual_nu'` (`shape (n,)`).
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n, p = X_arr.shape
    M = float(M_box)
    ridge = float(lam2)
    rho = float(lam0)

    def _fail(reason: str, **extra: Any) -> dict[str, Any]:
        base = {
            "strict_pass": False,
            "solution_feasible": False,
            "partition_valid": False,
            "leaf_bounds_valid": False,
            "certificate_valid": False,
            "gap_satisfied": False,
            "objective_consistent": False,
            "reason": reason,
        }
        base.update(extra)
        return base

    if not (_is_valid_real_scalar(reported_objective) and _is_valid_real_scalar(certified_lower_bound)):
        return _fail("non_finite_or_bool_scalar_fields")
    rep_obj = float(reported_objective)
    cert_lb = float(certified_lower_bound)

    if isinstance(weights, (bool, str, bytes)) or not hasattr(weights, "__len__"):
        return _fail("invalid_weights_container")
    try:
        if any(isinstance(v, (bool, np.bool_)) for v in weights):
            return _fail("bool_entry_in_weights")
        w = np.asarray(weights, dtype=float)
    except Exception:
        return _fail("weights_cast_error")
    if w.shape != (p,) or not np.all(np.isfinite(w)):
        return _fail("weights_shape_or_nan")
    if float(np.min(w)) < -M - 1e-7 or float(np.max(w)) > M + 1e-7:
        return _fail("weights_violate_symmetric_box")

    w_clip = np.clip(w, -M, M)
    support = np.abs(w_clip) > 1e-8
    w_eval = np.where(support, w_clip, 0.0)
    resid = X_arr @ w_eval - y_arr
    true_obj = 0.5 * float(resid @ resid) + 0.5 * ridge * float(w_eval @ w_eval) + rho * int(np.sum(support))
    obj_consistent = abs(rep_obj - true_obj) <= obj_tol * max(1.0, abs(true_obj))

    if not isinstance(subcube_leaves, list) or not (1 <= len(subcube_leaves) <= max_leaves):
        return _fail(
            f"invalid_leaf_list_length({len(subcube_leaves) if isinstance(subcube_leaves, list) else -1})",
            solution_feasible=True,
            objective_consistent=obj_consistent,
            true_objective=true_obj,
        )

    L = len(subcube_leaves)
    z_masks: list[int] = []
    a_masks: list[int] = []
    parsed_leaves: list[tuple[np.ndarray, np.ndarray, float, dict[str, Any]]] = []
    measure_sum = 0
    for idx_l, lf in enumerate(subcube_leaves):
        if not isinstance(lf, dict) or not {"fixed_zero", "fixed_active", "lb"} <= set(lf.keys()):
            return _fail(f"missing_leaf_keys_at_{idx_l}", solution_feasible=True, objective_consistent=obj_consistent)
        if not _is_valid_real_scalar(lf["lb"]):
            return _fail(f"invalid_leaf_lb_at_{idx_l}", solution_feasible=True, objective_consistent=obj_consistent)
        try:
            fz = np.asarray(lf["fixed_zero"], dtype=bool)
            fa = np.asarray(lf["fixed_active"], dtype=bool)
        except Exception:
            return _fail(f"invalid_leaf_masks_at_{idx_l}", solution_feasible=True, objective_consistent=obj_consistent)
        if fz.shape != (p,) or fa.shape != (p,) or bool(np.any(fz & fa)):
            return _fail(f"overlapping_or_bad_shape_masks_at_{idx_l}", solution_feasible=True, objective_consistent=obj_consistent)
        zm = 0
        am = 0
        for j in range(p):
            if fz[j]:
                zm |= 1 << j
            elif fa[j]:
                am |= 1 << j
        z_masks.append(zm)
        a_masks.append(am)
        fixed_cnt = int(np.sum(fz) + np.sum(fa))
        measure_sum += 1 << (p - fixed_cnt)
        parsed_leaves.append((fz, fa, float(lf["lb"]), lf))

    # Exact pairwise disjointness check via vectorized uint64 bitmasks (`p <= 62`):
    z_u64 = np.array(z_masks, dtype=np.uint64)
    a_u64 = np.array(a_masks, dtype=np.uint64)
    sep = (z_u64[:, None] & a_u64[None, :]) | (a_u64[:, None] & z_u64[None, :])
    np.fill_diagonal(sep, np.uint64(1))
    if bool(np.any(sep == 0)):
        bad_i, bad_q = np.argwhere(sep == 0)[0]
        return _fail(
            f"overlapping_leaves({int(bad_i)},{int(bad_q)})",
            solution_feasible=True,
            objective_consistent=obj_consistent,
        )
    if measure_sum != (1 << p):
        return _fail(
            f"incomplete_partition_measure({measure_sum}!={1 << p})",
            solution_feasible=True,
            objective_consistent=obj_consistent,
        )

    # Cache minorant PSD checks by raw byte digest so shared `minorant_diagonal` across leaves runs `eigvalsh` ONCE!
    G_mat = X_arr.T @ X_arr
    col_sq = np.diag(G_mat)
    scale_eig = max(1.0, float(np.max(col_sq)) if p > 0 else 1.0)
    psd_cache: dict[bytes, tuple[bool, float, str]] = {}

    leaf_reports: list[dict[str, Any]] = []
    all_leaf_bounds_sound = True
    first_unsound_reason = ""
    verified_leaf_lbs: list[float] = []

    for idx_l, (fz, fa, claimed_lb, raw_lf) in enumerate(parsed_leaves):
        free = ~(fz | fa)
        v_lb = float("-inf")
        mode_used = "missing_witness"
        witness_ok = False

        if raw_lf.get("minorant_diagonal") is not None and raw_lf.get("minorant_center") is not None:
            mode_used = "psd_diagonal_minorant"
            try:
                d_vec = np.asarray(raw_lf["minorant_diagonal"], dtype=float).ravel()
                c_vec = np.asarray(raw_lf["minorant_center"], dtype=float).ravel()
            except Exception:
                d_vec = np.array([])
                c_vec = np.array([])
            if d_vec.shape == (p,) and c_vec.shape == (p,) and np.all(np.isfinite(d_vec)) and np.all(np.isfinite(c_vec)) and np.all(d_vec >= -1e-12):
                d_clean = np.maximum(d_vec, 0.0)
                d_key = d_clean.tobytes()
                if d_key not in psd_cache:
                    G_diff = G_mat - np.diag(d_clean)
                    min_eig = float(np.linalg.eigvalsh(0.5 * (G_diff + G_diff.T))[0])
                    sc = max(scale_eig, float(np.max(d_clean)))
                    if min_eig < -1e-8 * sc:
                        psd_cache[d_key] = (False, 0.0, f"not_psd(min_eig={min_eig:.3e})")
                    else:
                        neg_eig = max(0.0, -min_eig)
                        psd_cache[d_key] = (True, neg_eig + 2e-15 * sc, "ok")
                is_psd, eig_guard, _ = psd_cache[d_key]
                if is_psd:
                    psd_slack = 0.5 * eig_guard * float(np.sum((M + np.abs(c_vec)) ** 2)) if eig_guard > 0.0 else 0.0
                    r_vec = y_arr - (X_arr @ c_vec)
                    g_vec = X_arr.T @ r_vec
                    dc = d_clean * c_vec
                    const_term = 0.5 * float(r_vec @ r_vec) + float(g_vec @ c_vec) + 0.5 * float(dc @ c_vec)
                    z_vec = np.abs(g_vec + dc)
                    curv = ridge + d_clean
                    gains = np.empty(p, dtype=float)
                    for j in range(p):
                        if curv[j] > 0.0 and z_vec[j] < M * curv[j]:
                            gains[j] = 0.5 * (z_vec[j] * z_vec[j]) / curv[j]
                        else:
                            gains[j] = M * z_vec[j] - 0.5 * curv[j] * (M * M)
                    on_val = rho - gains
                    v_lb = float(const_term + np.sum(on_val[fa]) + np.sum(np.minimum(on_val[free], 0.0)) - psd_slack)
                    witness_ok = math.isfinite(v_lb)

        elif raw_lf.get("dual_nu") is not None:
            mode_used = "fenchel_perspective_dual"
            try:
                nu = np.asarray(raw_lf["dual_nu"], dtype=float).ravel()
            except Exception:
                nu = np.array([])
            if nu.shape == (n,) and np.all(np.isfinite(nu)):
                alpha = np.abs(X_arr.T @ nu)
                phi = np.empty(p, dtype=float)
                for j in range(p):
                    if ridge > 0.0 and alpha[j] < M * ridge:
                        phi[j] = 0.5 * (alpha[j] * alpha[j]) / ridge
                    else:
                        phi[j] = M * alpha[j] - 0.5 * ridge * (M * M)
                on_dual = rho - phi
                v_lb = float(-0.5 * float(nu @ nu) + float(nu @ y_arr) + np.sum(on_dual[fa]) + np.sum(np.minimum(on_dual[free], 0.0)))
                witness_ok = math.isfinite(v_lb)

        tol_l = leaf_soundness_tol * max(1.0, abs(v_lb) if math.isfinite(v_lb) else 1.0)
        is_sound = bool(witness_ok and (claimed_lb <= v_lb + tol_l))
        if not is_sound:
            all_leaf_bounds_sound = False
            if not first_unsound_reason:
                first_unsound_reason = f"unsound_or_missing_leaf_witness(leaf={idx_l},mode={mode_used},claimed={claimed_lb:.6f},verified={v_lb})"
        verified_leaf_lbs.append(v_lb if witness_ok else float("-inf"))
        leaf_reports.append(
            {
                "leaf_index": idx_l,
                "mode": mode_used,
                "claimed_lb": round(claimed_lb, 8),
                "verified_lb": (round(v_lb, 8) if math.isfinite(v_lb) else None),
                "sound": is_sound,
            }
        )

    min_claimed_leaf = min(x[2] for x in parsed_leaves)
    min_verified_leaf = min(verified_leaf_lbs)
    cert_lb_consistent = cert_lb <= min_claimed_leaf + leaf_soundness_tol * max(1.0, abs(min_claimed_leaf))
    if not cert_lb_consistent:
        all_leaf_bounds_sound = False
        if not first_unsound_reason:
            first_unsound_reason = f"certified_lb_exceeds_min_leaf_lb({cert_lb:.6f}>{min_claimed_leaf:.6f})"

    effective_verified_global_lb = min(cert_lb, min_verified_leaf)
    reported_gap = true_obj - cert_lb
    verified_gap = true_obj - effective_verified_global_lb
    gap_ok = bool((-1e-6 <= reported_gap <= gap_tol) and (verified_gap <= gap_tol + 2.0 * leaf_soundness_tol))
    certificate_valid = bool(all_leaf_bounds_sound and gap_ok and obj_consistent)

    return {
        "strict_pass": certificate_valid,
        "solution_feasible": True,
        "partition_valid": True,
        "leaf_bounds_valid": all_leaf_bounds_sound,
        "certificate_valid": certificate_valid,
        "gap_satisfied": gap_ok,
        "objective_consistent": obj_consistent,
        "reason": ("ok" if certificate_valid else (first_unsound_reason or "gap_or_objective_mismatch")),
        "leaf_count": L,
        "true_objective": round(true_obj, 9),
        "certified_lower_bound": round(cert_lb, 9),
        "verified_global_lower_bound": (round(effective_verified_global_lb, 9) if math.isfinite(effective_verified_global_lb) else None),
        "reported_gap": round(reported_gap, 9),
        "verified_gap": (round(verified_gap, 9) if math.isfinite(verified_gap) else None),
        "leaves_preview": leaf_reports[:8],
    }


def make_vr01_instance(seed: int, block_sizes: tuple[int, ...]) -> dict[str, Any]:
    """Generate a `p = sum(block_sizes)` symmetric-box L0-ridge problem with:
    1. Row- and column-permuted coupled tridiagonal blocks (`gram_diagonal_minorant(X) > 0` strictly!),
    2. Competing correlated predictors creating a strictly positive unbranched root relaxation gap, and
    3. Tractable exact ground truth via per-block enumeration (`<= 2 * 2^11 = 4096` small ridge solves) for audit!
    """
    from scipy.optimize import lsq_linear

    rng = np.random.default_rng(seed)
    n_per_block = 22
    B = len(block_sizes)
    n_tot = B * n_per_block
    p_tot = sum(block_sizes)
    X_blk = np.zeros((n_tot, p_tot), dtype=float)
    y_blk = np.zeros(n_tot, dtype=float)

    lam2 = 0.14
    lam0 = 0.22
    M_box = 1.10

    r0 = 0
    c0 = 0
    for b_idx, pb in enumerate(block_sizes):
        # Construct orthonormal carrier Q_b (n_per_block x pb) and tridiagonal mixing so
        # X_b.T @ X_b has unit diagonal and off-diagonal |G_jk| == 0.26 on |j-k|==1 (d_gersh >= 0.48!)
        G_raw = rng.normal(0.0, 1.0, size=(n_per_block, pb))
        Q_b, _ = np.linalg.qr(G_raw, mode="reduced")
        target_G = np.eye(pb, dtype=float)
        for j in range(pb - 1):
            sgn = 1.0 if ((j + b_idx) % 2 == 0) else -1.0
            target_G[j, j + 1] = 0.26 * sgn
            target_G[j + 1, j] = 0.26 * sgn
        L_chol = np.linalg.cholesky(target_G)
        Xb = Q_b @ L_chol.T

        w_b = np.zeros(pb, dtype=float)
        w_b[1] = 0.88 if (b_idx % 2 == 0) else -0.86
        w_b[2] = 0.60 if (target_G[1, 2] > 0) else -0.60
        if pb >= 7:
            w_b[pb - 2] = -0.76 if (b_idx % 2 == 0) else 0.78
        y_b = (
            Xb @ w_b
            + 0.48 * Xb[:, 0]
            - 0.44 * Xb[:, min(4, pb - 1)]
            + rng.normal(0.0, 0.05, size=n_per_block)
        )
        X_blk[r0 : r0 + n_per_block, c0 : c0 + pb] = Xb
        y_blk[r0 : r0 + n_per_block] = y_b
        r0 += n_per_block
        c0 += pb

    row_perm = rng.permutation(n_tot)
    col_perm = rng.permutation(p_tot)
    X = X_blk[row_perm][:, col_perm]
    y = y_blk[row_perm]

    # Compute exact global optimum via exact block-wise enumeration (`<= 3 * 256` tiny box-ridge QPs)
    best_w_blk = np.zeros(p_tot, dtype=float)
    exact_obj = 0.0
    r0 = 0
    c0 = 0
    sqrt_l2 = math.sqrt(lam2)
    for pb in block_sizes:
        Xb = X_blk[r0 : r0 + n_per_block, c0 : c0 + pb]
        yb = y_blk[r0 : r0 + n_per_block]
        best_val = 0.5 * float(yb @ yb)
        best_wb = np.zeros(pb, dtype=float)
        for mask in range(1, 1 << pb):
            idx = [j for j in range(pb) if (mask >> j) & 1]
            k = len(idx)
            Xs = Xb[:, idx]
            X_aug = np.vstack([Xs, sqrt_l2 * np.eye(k)])
            y_aug = np.concatenate([yb, np.zeros(k)])
            res = lsq_linear(X_aug, y_aug, bounds=(-M_box, M_box), tol=1e-12)
            ws = res.x
            rr = Xs @ ws - yb
            val = 0.5 * float(rr @ rr) + 0.5 * lam2 * float(ws @ ws) + lam0 * k
            if val < best_val:
                best_val = val
                best_wb = np.zeros(pb, dtype=float)
                best_wb[idx] = ws
        best_w_blk[c0 : c0 + pb] = best_wb
        exact_obj += best_val
        r0 += n_per_block
        c0 += pb

    inv_col_perm = np.argsort(col_perm)
    exact_w = best_w_blk[inv_col_perm]
    return {
        "problem": {
            "X": X,
            "y": y,
            "lam2": lam2,
            "lam0": lam0,
            "M": M_box,
        },
        "gt": {
            "exact_objective": float(exact_obj),
            "exact_weights": exact_w,
        },
    }


VOLUNTARY_REUSE_M90A_TASKS: tuple[HoldoutOODTask, ...] = (
    # =========================================================================
    # VR01: Proof-Carrying High-p Symmetric-Box L0-Ridge Branch-and-Bound
    # =========================================================================
    HoldoutOODTask(
        task_id="vr01__proof_carrying_box_l0_minorant_bnb",
        short_name="vr01_proof_carrying_box_l0",
        wave="m90a_voluntary_reuse",
        scientific_domain="Combinatorial Global Optimization & Proof-Carrying Dual/Minorant Certificates",
        wormhole_family="fam_sparse_cert_and_combinatorial_pruning",
        declared_artifacts=("/holdout_task/proof_box_l0.py",),
        declared_callable_entrypoints=("solve_proof_carrying_box_l0",),
        instruction_text=(
            "Implement `/holdout_task/proof_box_l0.py` exposing:\n"
            "`def solve_proof_carrying_box_l0(problem: dict) -> dict:`\n\n"
            "Mathematical Optimization & Proof-Carrying Certificate Contract:\n"
            "- Given `X` (`shape (n, p)`, `p in [19, 22]` — `2^p` up to `4,194,304` supports, so brute-force `2^p` support enumeration is intractable!),\n"
            "  `y` (`shape (n,)`), `lam2 > 0` (`float`), `lam0 > 0` (`float`), and symmetric box bound `M > 0` (`float`),\n"
            "  globally minimize over `w in [-M, M]^p`:\n"
            "  `F(w) = 0.5 * ||X @ w - y||_2^2 + 0.5 * lam2 * ||w||_2^2 + lam0 * ||w||_0`,\n"
            "  where `||w||_0 = count(|w_j| > 1e-8)`.\n"
            "- Return a dictionary with:\n"
            "  1. `'weights'`: 1D float array `(p,)` in `[-M, M]^p` (with inactive coordinates set to `0.0`),\n"
            "  2. `'objective'`: float `F(weights)`,\n"
            "  3. `'certified_lower_bound'`: float `L <= F(weights)` satisfying `0 <= F(weights) - L <= 1e-5`,\n"
            "  4. `'subcube_leaves'`: non-empty list (`1 <= len(subcube_leaves) <= 2048`) of pairwise-disjoint subcubes partitioning `{0, 1}^p`.\n"
            "     Each leaf dict MUST contain:\n"
            "     - `'fixed_zero'`: boolean list/array `(p,)` (`w_j == 0`),\n"
            "     - `'fixed_active'`: boolean list/array `(p,)` (`w_j != 0`, disjoint from `'fixed_zero'`),\n"
            "     - `'lb'`: float lower bound (`certified_lower_bound <= min_leaf lb + 1e-6`), AND\n"
            "     - A machine-checkable polynomial-time dual witness proving `'lb'` on `[-M, M]^p` under `(fixed_zero, fixed_active)` —\n"
            "       provide EITHER:\n"
            "       (A) `'minorant_diagonal'` (`1D array (p,)`, `d >= 0` with `X.T @ X - diag(d)` positive semidefinite) AND `'minorant_center'` (`1D array (p,)`),\n"
            "           certifying `0.5*||y - X c||_2^2 + (y - X c).T @ X @ c + 0.5*(d * c) @ c + sum_{j in A}(lam0 - gain_j) + sum_{j in U} min(0, lam0 - gain_j)`\n"
            "           where `gain_j = max_{0 <= b <= M} (|(X.T @ (y - X c))_j + d_j * c_j| * b - 0.5 * (lam2 + d_j) * b^2)`, OR\n"
            "       (B) `'dual_nu'` (`1D array (n,)`), certifying the Fenchel perspective dual lower bound\n"
            "           `-0.5*||nu||_2^2 + nu.T @ y + sum_{j in A}(lam0 - phi_j) + sum_{j in U} min(0, lam0 - phi_j)`\n"
            "           where `phi_j = max_{0 <= b <= M} (|(X.T @ nu)_j| * b - 0.5 * lam2 * b^2)`.\n"
            "- Public development instances (including exact optimal objectives for validation) are in `/holdout_task/public_proof_box_l0_cases.pkl`."
        ),
        public_setup_py=r'''
import math, pickle, numpy as np
from scipy.optimize import lsq_linear

def _make_case(seed, block_sizes):
    rng = np.random.default_rng(seed)
    n_per_block = 22
    B = len(block_sizes)
    n_tot, p_tot = B * n_per_block, sum(block_sizes)
    X_blk = np.zeros((n_tot, p_tot), dtype=float)
    y_blk = np.zeros(n_tot, dtype=float)
    lam2, lam0, M_box = 0.14, 0.22, 1.10
    r0 = c0 = 0
    for b_idx, pb in enumerate(block_sizes):
        G_raw = rng.normal(0.0, 1.0, size=(n_per_block, pb))
        Q_b, _ = np.linalg.qr(G_raw, mode="reduced")
        target_G = np.eye(pb, dtype=float)
        for j in range(pb - 1):
            sgn = 1.0 if ((j + b_idx) % 2 == 0) else -1.0
            target_G[j, j + 1] = 0.26 * sgn
            target_G[j + 1, j] = 0.26 * sgn
        L_chol = np.linalg.cholesky(target_G)
        Xb = Q_b @ L_chol.T
        w_b = np.zeros(pb, dtype=float)
        w_b[1] = 0.88 if (b_idx % 2 == 0) else -0.86
        w_b[2] = 0.60 if (target_G[1, 2] > 0) else -0.60
        if pb >= 7:
            w_b[pb - 2] = -0.76 if (b_idx % 2 == 0) else 0.78
        y_b = Xb @ w_b + 0.48 * Xb[:, 0] - 0.44 * Xb[:, min(4, pb - 1)] + rng.normal(0.0, 0.05, size=n_per_block)
        X_blk[r0:r0+n_per_block, c0:c0+pb] = Xb
        y_blk[r0:r0+n_per_block] = y_b
        r0 += n_per_block; c0 += pb
    row_perm = rng.permutation(n_tot)
    col_perm = rng.permutation(p_tot)
    X = X_blk[row_perm][:, col_perm]
    y = y_blk[row_perm]
    best_w_blk = np.zeros(p_tot, dtype=float)
    exact_obj = 0.0
    r0 = c0 = 0
    sqrt_l2 = math.sqrt(lam2)
    for pb in block_sizes:
        Xb = X_blk[r0:r0+n_per_block, c0:c0+pb]
        yb = y_blk[r0:r0+n_per_block]
        best_val = 0.5 * float(yb @ yb)
        best_wb = np.zeros(pb, dtype=float)
        for mask in range(1, 1 << pb):
            idx = [j for j in range(pb) if (mask >> j) & 1]
            k = len(idx)
            Xs = Xb[:, idx]
            res = lsq_linear(np.vstack([Xs, sqrt_l2 * np.eye(k)]), np.concatenate([yb, np.zeros(k)]), bounds=(-M_box, M_box), tol=1e-12)
            ws = res.x
            rr = Xs @ ws - yb
            val = 0.5 * float(rr @ rr) + 0.5 * lam2 * float(ws @ ws) + lam0 * k
            if val < best_val:
                best_val = val
                best_wb = np.zeros(pb, dtype=float); best_wb[idx] = ws
        best_w_blk[c0:c0+pb] = best_wb
        exact_obj += best_val
        r0 += n_per_block; c0 += pb
    return {
        "problem": {"X": X, "y": y, "lam2": lam2, "lam0": lam0, "M": M_box},
        "gt": {"exact_objective": float(exact_obj), "exact_weights": best_w_blk[np.argsort(col_perm)]},
    }

cases = [_make_case(501, (9, 10)), _make_case(502, (10, 10))]
with open("/holdout_task/public_proof_box_l0_cases.pkl", "wb") as fh:
    pickle.dump(cases, fh)
''',
        private_verifier_py=r'''
import json, math, sys, time
import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, "/holdout_task")
import proof_box_l0

def _is_real(val):
    if isinstance(val, (bool, np.bool_)):
        return False
    if not isinstance(val, (int, float, np.number)):
        return False
    return math.isfinite(float(val))

def _make_priv(seed, block_sizes):
    rng = np.random.default_rng(seed)
    n_per_block = 22
    B = len(block_sizes)
    n_tot, p_tot = B * n_per_block, sum(block_sizes)
    X_blk = np.zeros((n_tot, p_tot), dtype=float)
    y_blk = np.zeros(n_tot, dtype=float)
    lam2, lam0, M_box = 0.14, 0.22, 1.10
    r0 = c0 = 0
    for b_idx, pb in enumerate(block_sizes):
        G_raw = rng.normal(0.0, 1.0, size=(n_per_block, pb))
        Q_b, _ = np.linalg.qr(G_raw, mode="reduced")
        target_G = np.eye(pb, dtype=float)
        for j in range(pb - 1):
            sgn = 1.0 if ((j + b_idx) % 2 == 0) else -1.0
            target_G[j, j + 1] = 0.26 * sgn
            target_G[j + 1, j] = 0.26 * sgn
        L_chol = np.linalg.cholesky(target_G)
        Xb = Q_b @ L_chol.T
        w_b = np.zeros(pb, dtype=float)
        w_b[1] = 0.88 if (b_idx % 2 == 0) else -0.86
        w_b[2] = 0.60 if (target_G[1, 2] > 0) else -0.60
        if pb >= 7:
            w_b[pb - 2] = -0.76 if (b_idx % 2 == 0) else 0.78
        y_b = Xb @ w_b + 0.48 * Xb[:, 0] - 0.44 * Xb[:, min(4, pb - 1)] + rng.normal(0.0, 0.05, size=n_per_block)
        X_blk[r0:r0+n_per_block, c0:c0+pb] = Xb
        y_blk[r0:r0+n_per_block] = y_b
        r0 += n_per_block; c0 += pb
    row_perm = rng.permutation(n_tot)
    col_perm = rng.permutation(p_tot)
    X = X_blk[row_perm][:, col_perm]
    y = y_blk[row_perm]
    exact_obj = 0.0
    r0 = c0 = 0
    sqrt_l2 = math.sqrt(lam2)
    for pb in block_sizes:
        Xb = X_blk[r0:r0+n_per_block, c0:c0+pb]
        yb = y_blk[r0:r0+n_per_block]
        best_val = 0.5 * float(yb @ yb)
        for mask in range(1, 1 << pb):
            idx = [j for j in range(pb) if (mask >> j) & 1]
            k = len(idx)
            Xs = Xb[:, idx]
            res = lsq_linear(np.vstack([Xs, sqrt_l2 * np.eye(k)]), np.concatenate([yb, np.zeros(k)]), bounds=(-M_box, M_box), tol=1e-12)
            ws = res.x
            rr = Xs @ ws - yb
            val = 0.5 * float(rr @ rr) + 0.5 * lam2 * float(ws @ ws) + lam0 * k
            if val < best_val:
                best_val = val
        exact_obj += best_val
        r0 += n_per_block; c0 += pb
    return {"X": X, "y": y, "lam2": lam2, "lam0": lam0, "M": M_box}, float(exact_obj)

def _verify_cert(prob, out, exact_opt):
    if not isinstance(out, dict) or not {"weights", "objective", "certified_lower_bound", "subcube_leaves"} <= set(out.keys()):
        return False, "missing_top_keys", 0.0, 0
    X_arr = prob["X"]; y_arr = prob["y"]
    n, p = X_arr.shape
    M = float(prob["M"]); ridge = float(prob["lam2"]); rho = float(prob["lam0"])
    if not (_is_real(out["objective"]) and _is_real(out["certified_lower_bound"])):
        return False, "non_finite_scalars", 0.0, 0
    rep_obj = float(out["objective"]); cert_lb = float(out["certified_lower_bound"])
    w_raw = out["weights"]
    if isinstance(w_raw, (bool, str, bytes)) or not hasattr(w_raw, "__len__") or any(isinstance(v, (bool, np.bool_)) for v in w_raw):
        return False, "bad_weights", 0.0, 0
    w = np.asarray(w_raw, dtype=float)
    if w.shape != (p,) or not np.all(np.isfinite(w)) or np.min(w) < -M - 1e-7 or np.max(w) > M + 1e-7:
        return False, "weights_out_of_box", 0.0, 0
    w_clip = np.clip(w, -M, M)
    sup = np.abs(w_clip) > 1e-8
    w_eval = np.where(sup, w_clip, 0.0)
    rr = X_arr @ w_eval - y_arr
    true_obj = 0.5 * float(rr @ rr) + 0.5 * ridge * float(w_eval @ w_eval) + rho * int(np.sum(sup))
    if abs(rep_obj - true_obj) > 1e-6 * max(1.0, abs(true_obj)):
        return False, "objective_mismatch", 0.0, 0
    if true_obj > exact_opt + 1e-5:
        return False, f"suboptimal_primal({true_obj:.6f}>{exact_opt:.6f})", true_obj - exact_opt, 0
    leaves = out["subcube_leaves"]
    if not isinstance(leaves, list) or not (1 <= len(leaves) <= 2048):
        return False, "bad_leaf_count", 0.0, 0
    L = len(leaves)
    z_masks, a_masks, parsed = [], [], []
    meas = 0
    for lf in leaves:
        if not isinstance(lf, dict) or not {"fixed_zero", "fixed_active", "lb"} <= set(lf.keys()) or not _is_real(lf["lb"]):
            return False, "bad_leaf_schema", 0.0, L
        fz = np.asarray(lf["fixed_zero"], dtype=bool)
        fa = np.asarray(lf["fixed_active"], dtype=bool)
        if fz.shape != (p,) or fa.shape != (p,) or bool(np.any(fz & fa)):
            return False, "bad_leaf_masks", 0.0, L
        zm = am = 0
        for j in range(p):
            if fz[j]: zm |= (1 << j)
            elif fa[j]: am |= (1 << j)
        z_masks.append(zm); a_masks.append(am)
        meas += 1 << (p - int(np.sum(fz) + np.sum(fa)))
        parsed.append((fz, fa, float(lf["lb"]), lf))
    z_u64 = np.array(z_masks, dtype=np.uint64)
    a_u64 = np.array(a_masks, dtype=np.uint64)
    sep = (z_u64[:, None] & a_u64[None, :]) | (a_u64[:, None] & z_u64[None, :])
    np.fill_diagonal(sep, np.uint64(1))
    if bool(np.any(sep == 0)):
        return False, "overlapping_leaves", 0.0, L
    if meas != (1 << p):
        return False, "incomplete_partition", 0.0, L

    G_mat = X_arr.T @ X_arr
    scale_eig = max(1.0, float(np.max(np.diag(G_mat))))
    psd_cache = {}
    v_lbs = []
    for fz, fa, clb, rlf in parsed:
        free = ~(fz | fa)
        vlb = float("-inf"); ok_w = False
        if rlf.get("minorant_diagonal") is not None and rlf.get("minorant_center") is not None:
            d_v = np.asarray(rlf["minorant_diagonal"], dtype=float).ravel()
            c_v = np.asarray(rlf["minorant_center"], dtype=float).ravel()
            if d_v.shape == (p,) and c_v.shape == (p,) and np.all(np.isfinite(d_v)) and np.all(np.isfinite(c_v)) and np.all(d_v >= -1e-12):
                dc_clean = np.maximum(d_v, 0.0)
                dk = dc_clean.tobytes()
                if dk not in psd_cache:
                    Gd = G_mat - np.diag(dc_clean)
                    me = float(np.linalg.eigvalsh(0.5 * (Gd + Gd.T))[0])
                    sc = max(scale_eig, float(np.max(dc_clean)))
                    psd_cache[dk] = (me >= -1e-8 * sc, max(0.0, -me) + 2e-15 * sc if me < 0.0 else 0.0)
                is_p, eg = psd_cache[dk]
                if is_p:
                    pslack = 0.5 * eg * float(np.sum((M + np.abs(c_v)) ** 2)) if eg > 0.0 else 0.0
                    rv = y_arr - (X_arr @ c_v); gv = X_arr.T @ rv; dc = dc_clean * c_v
                    ct = 0.5 * float(rv @ rv) + float(gv @ c_v) + 0.5 * float(dc @ c_v)
                    zv = np.abs(gv + dc); cv = ridge + dc_clean
                    gains = np.where((cv > 0.0) & (zv < M * cv), 0.5 * zv * zv / np.maximum(cv, 1e-30), M * zv - 0.5 * cv * M * M)
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
            return False, f"unsound_leaf_witness(claimed={clb:.6f},verified={vlb})", 0.0, L
        v_lbs.append(vlb)
    min_clb = min(x[2] for x in parsed)
    if cert_lb > min_clb + 1e-6 * max(1.0, abs(min_clb)):
        return False, "cert_lb_gt_min_leaf", 0.0, L
    eff_lb = min(cert_lb, min(v_lbs))
    gap = true_obj - eff_lb
    if not (-1e-6 <= (true_obj - cert_lb) <= 1e-5 and gap <= 1.2e-5):
        return False, f"gap_too_large({gap:.6e})", gap, L
    return True, "ok", gap, L

t0 = time.monotonic()
all_ok = True
notes = []
max_gap = 0.0
leaf_counts = []
for sd, bsz in ((99201, (10, 10)), (99202, (10, 11)), (99203, (11, 11))):
    prob, exact_opt = _make_priv(sd, bsz)
    out = proof_box_l0.solve_proof_carrying_box_l0({k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in prob.items()})
    ok_i, note_i, gap_i, lc_i = _verify_cert(prob, out, exact_opt)
    notes.append(f"{sd}:{note_i}")
    max_gap = max(max_gap, float(gap_i))
    leaf_counts.append(int(lc_i))
    if not ok_i:
        all_ok = False
        break
elapsed = time.monotonic() - t0
passed = bool(all_ok and len(leaf_counts) == 3 and elapsed < 20.0)
print(json.dumps({
    "strict_pass": passed,
    "max_verified_gap": round(float(max_gap), 9),
    "leaf_counts": leaf_counts,
    "status_notes": notes,
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # VR02: 5,184-Unknowns High-Contrast Robin SPD Multi-RHS & Nuisance Profiling
    # =========================================================================
    HoldoutOODTask(
        task_id="vr02__large_robin_spd_preconditioned_cg_multi_rhs",
        short_name="vr02_large_robin_spd_profiler",
        wave="m90a_voluntary_reuse",
        scientific_domain="Large-Scale High-Contrast Elliptic PDE Preconditioning & Nuisance-Profiled Source Matching",
        wormhole_family="fam_multiscale_pde_and_robust_assimilation",
        declared_artifacts=("/holdout_task/robin_spd_profiler.py",),
        declared_callable_entrypoints=("solve_robin_spd_and_profile_sources",),
        instruction_text=(
            "Implement `/holdout_task/robin_spd_profiler.py` exposing:\n"
            "`def solve_robin_spd_and_profile_sources(problem: dict) -> dict:`\n\n"
            "Large-Scale (`72 x 72 = 5,184 unknowns`) High-Contrast Robin Elliptic PDE & Profiled Matching Specification:\n"
            "- Steady 2D isotropic high-contrast (`4 orders of magnitude`, `kappa in [0.02, 80.0]`) diffusion-reaction on `[0, Lx] x [0, Ly]`\n"
            "  with uniform cell-centered grid `(nx, ny) = (72, 72)` (`5,184 unknowns`, `hx = Lx / nx`, `hy = Ly / ny`):\n"
            "  `-div(kappa(x, y) grad u) + sigma(x, y) * u = q(x, y)`\n"
            "  subject to homogeneous external Robin boundary dissipation `(-kappa grad u) . n = gamma_robin * u` on all four domain walls.\n"
            "- Exact cell-centered finite-volume stencil (`5,184 x 5,184` symmetric positive-definite system `A u = b` in C-order `ravel()`):\n"
            "  1. Interior x-faces `(i+1/2, j)` use harmonic conductivity `k_f = 2*kappa[i,j]*kappa[i+1,j]/(kappa[i,j]+kappa[i+1,j])`,\n"
            "     contributing `(hy / hx) * k_f * (u[i,j] - u[i+1,j])` (and symmetrically on interior y-faces `(i, j+1/2)` with `(hx / hy)`).\n"
            "  2. Exterior boundary faces (`i=0`, `i=nx-1`, `j=0`, `j=ny-1`) use series half-cell Robin conductance:\n"
            "     x-boundary faces contribute `hy * (gamma_robin * kappa_b) / (gamma_robin * 0.5 * hx + kappa_b) * u_b`,\n"
            "     y-boundary faces contribute `hx * (gamma_robin * kappa_b) / (gamma_robin * 0.5 * hy + kappa_b) * u_b`.\n"
            "     (WARNING: zero-ghost Dirichlet stencils `kappa_b / h` violate the Robin boundary flux!)\n"
            "  3. Cell reaction + source balance: `(face outflows) + (hx * hy) * sigma[i,j] * u[i,j] = (hx * hy) * q[i,j]`.\n"
            "- Given `problem` keys `('Lx', 'Ly', 'kappa', 'sigma', 'gamma_robin', 'candidate_sources', 'sensor_ij', 'nuisance_design', 'nuisance_ridge', 'source_ridge', 'Y_obs')`:\n"
            "  - `candidate_sources` has shape `(M, nx, ny)` (`M = 10` candidate source modes). Solve `A u_m = b_m` for all `m = 0..M-1`\n"
            "    to obtain `basis_fields` (`shape (M, nx, ny)`, relative PDE residual `<= 1e-8`).\n"
            "  - Let `S` (`shape (K_s, M)`) be `basis_fields[:, sensor_ij[:, 0], sensor_ij[:, 1]].T` at the `K_s` sensors,\n"
            "    `Z = nuisance_design` (`shape (K_s, P)`), `lam_z = nuisance_ridge` (`float`), `lam_s = source_ridge` (`float`), and `Y = Y_obs` (`shape (K_s, Q_exp)`).\n"
            "  - Compute TWO inverse deliverables across all `Q_exp` experiments:\n"
            "    (a) `'single_source_profiled_amplitudes'` (`shape (M, Q_exp)`): for each single candidate column `s_m = S[:, m]` (`m = 0..M-1`)\n"
            "        fitted jointly with nuisance `Z` by minimizing `0.5 * ||Z @ gamma + alpha_m * s_m - Y[:, q]||_2^2 + 0.5 * lam_z * ||gamma||_2^2`\n"
            "        (NO penalty on scalar `alpha_m`!), return the exact profiled scalar amplitude `alpha_m` for each `(m, q)`.\n"
            "    (b) `'joint_source_weights'` (`shape (M, Q_exp)`): minimizing\n"
            "        `0.5 * ||Z @ gamma + S @ w - Y[:, q]||_2^2 + 0.5 * lam_z * ||gamma||_2^2 + 0.5 * lam_s * ||w||_2^2`\n"
            "        jointly over `(gamma, w)`, return `w` (`shape (M, Q_exp)`).\n"
            "- Return `{'basis_fields': (M, nx, ny), 'single_source_profiled_amplitudes': (M, Q_exp), 'joint_source_weights': (M, Q_exp)}`.\n"
            "- Public test cases are in `/holdout_task/public_robin_spd_cases.pkl`."
        ),
        public_setup_py=r'''
import pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def _assemble_robin_spd(Lx, Ly, kappa, sigma, gamma_robin):
    nx, ny = kappa.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    rows, cols, data = [], [], []
    diag = (hx * hy) * sigma.ravel().astype(float)

    # Interior x-faces
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    k0, k1 = kappa[:-1, :].ravel(), kappa[1:, :].ravel()
    wx = (hy / hx) * (2.0 * k0 * k1 / (k0 + k1))
    diag[i0] += wx; diag[i1] += wx
    rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-wx, -wx])

    # Interior y-faces
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    ky0, ky1 = kappa[:, :-1].ravel(), kappa[:, 1:].ravel()
    wy = (hx / hy) * (2.0 * ky0 * ky1 / (ky0 + ky1))
    diag[j0] += wy; diag[j1] += wy
    rows.extend([j0, j1]); cols.extend([j1, j0]); data.extend([-wy, -wy])

    # Robin x-boundaries
    for xb in (0, -1):
        ib = ids[xb, :].ravel(); kb = kappa[xb, :].ravel()
        diag[ib] += hy * (gamma_robin * kb) / (gamma_robin * 0.5 * hx + kb)
    # Robin y-boundaries
    for yb in (0, -1):
        jb = ids[:, yb].ravel(); kb = kappa[:, yb].ravel()
        diag[jb] += hx * (gamma_robin * kb) / (gamma_robin * 0.5 * hy + kb)

    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _make_case(seed, nx=72, ny=72):
    rng = np.random.default_rng(seed)
    Lx, Ly = 1.4, 1.2
    hx, hy = Lx / nx, Ly / ny
    xs = (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    # 4-order-of-magnitude high-contrast channels + inclusions
    log_k = 1.8 * np.sin(2.5 * np.pi * XX / Lx) * np.cos(2.0 * np.pi * YY / Ly)
    inclusion = ((XX - 0.42 * Lx)**2 + (YY - 0.55 * Ly)**2 < 0.035) | ((XX - 0.75 * Lx)**2 + (YY - 0.28 * Ly)**2 < 0.025)
    kappa = np.clip(np.exp(log_k), 0.03, 12.0)
    kappa[inclusion] = 75.0
    kappa[(XX < 0.22 * Lx) & (YY > 0.72 * Ly)] = 0.022
    sigma = 0.06 + 0.04 * (XX / Lx) + 0.03 * (YY / Ly)
    gamma_robin = 1.65

    M = 10
    centers = rng.uniform([0.15 * Lx, 0.15 * Ly], [0.85 * Lx, 0.85 * Ly], size=(M, 2))
    cand = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        r2 = (XX - centers[m, 0])**2 + (YY - centers[m, 1])**2
        cand[m] = np.exp(-r2 / (2.0 * (0.09**2)))

    A = _assemble_robin_spd(Lx, Ly, kappa, sigma, gamma_robin)
    basis_fields = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        rhs = (hx * hy) * cand[m].ravel()
        basis_fields[m] = spsolve(A, rhs).reshape(nx, ny)

    K_s = 360
    flat_choices = rng.choice(nx * ny, size=K_s, replace=False)
    sensor_ij = np.column_stack([flat_choices // ny, flat_choices % ny])
    sx, sy = xs[sensor_ij[:, 0]], ys[sensor_ij[:, 1]]
    Z = np.column_stack([np.ones(K_s), (sx - 0.5 * Lx) / Lx, (sy - 0.5 * Ly) / Ly, np.sin(np.pi * sx / Lx) * np.cos(np.pi * sy / Ly)])
    S = basis_fields[:, sensor_ij[:, 0], sensor_ij[:, 1]].T

    lam_z = 1e-3
    lam_s = 2.5e-2
    Q_exp = 5
    true_w = rng.normal(0.0, 0.9, size=(M, Q_exp))
    true_g = rng.normal(0.0, 0.6, size=(4, Q_exp))
    Y_obs = Z @ true_g + S @ true_w + rng.normal(0.0, 2e-3, size=(K_s, Q_exp))

    # Exact reference profiled single-source amplitudes & joint ridge weights
    P_z = np.linalg.solve(Z.T @ Z + lam_z * np.eye(4), Z.T)
    R_y = Y_obs - Z @ (P_z @ Y_obs)
    R_s = S - Z @ (P_z @ S)
    # Note: S.T @ (I - Z (Z^T Z + lam_z I)^-1 Z^T) S = S.T @ R_s!
    denom = np.sum(S * R_s, axis=0)
    prof_amp = (S.T @ R_y) / denom[:, None]

    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(4), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(M)]])
    rhs_aug = np.vstack([Z.T @ Y_obs, S.T @ Y_obs])
    sol_aug = np.linalg.solve(H_aug, rhs_aug)
    joint_w = sol_aug[4:, :]

    return {
        "problem": {
            "Lx": Lx, "Ly": Ly, "kappa": kappa, "sigma": sigma, "gamma_robin": gamma_robin,
            "candidate_sources": cand, "sensor_ij": sensor_ij, "nuisance_design": Z,
            "nuisance_ridge": lam_z, "source_ridge": lam_s, "Y_obs": Y_obs,
        },
        "gt": {
            "basis_fields": basis_fields,
            "single_source_profiled_amplitudes": prof_amp,
            "joint_source_weights": joint_w,
        },
    }

pub = [_make_case(601), _make_case(602)]
with open("/holdout_task/public_robin_spd_cases.pkl", "wb") as fh:
    pickle.dump(pub, fh)
''',
        private_verifier_py=r'''
import json, math, sys, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, "/holdout_task")
import robin_spd_profiler

def _assemble_robin_spd(Lx, Ly, kappa, sigma, gamma_robin):
    nx, ny = kappa.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    rows, cols, data = [], [], []
    diag = (hx * hy) * sigma.ravel().astype(float)
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    k0, k1 = kappa[:-1, :].ravel(), kappa[1:, :].ravel()
    wx = (hy / hx) * (2.0 * k0 * k1 / (k0 + k1))
    diag[i0] += wx; diag[i1] += wx
    rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-wx, -wx])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    ky0, ky1 = kappa[:, :-1].ravel(), kappa[:, 1:].ravel()
    wy = (hx / hy) * (2.0 * ky0 * ky1 / (ky0 + ky1))
    diag[j0] += wy; diag[j1] += wy
    rows.extend([j0, j1]); cols.extend([j1, j0]); data.extend([-wy, -wy])
    for xb in (0, -1):
        ib = ids[xb, :].ravel(); kb = kappa[xb, :].ravel()
        diag[ib] += hy * (gamma_robin * kb) / (gamma_robin * 0.5 * hx + kb)
    for yb in (0, -1):
        jb = ids[:, yb].ravel(); kb = kappa[:, yb].ravel()
        diag[jb] += hx * (gamma_robin * kb) / (gamma_robin * 0.5 * hy + kb)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _make_priv(seed):
    rng = np.random.default_rng(seed)
    nx = ny = 72
    Lx, Ly = 1.5, 1.3
    hx, hy = Lx / nx, Ly / ny
    xs = (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    log_k = 1.9 * np.cos(2.2 * np.pi * XX / Lx) * np.sin(2.4 * np.pi * YY / Ly)
    inc = ((XX - 0.36 * Lx)**2 + (YY - 0.62 * Ly)**2 < 0.032) | ((XX - 0.72 * Lx)**2 + (YY - 0.31 * Ly)**2 < 0.028)
    kappa = np.clip(np.exp(log_k), 0.025, 14.0)
    kappa[inc] = 80.0
    kappa[(XX > 0.74 * Lx) & (YY > 0.70 * Ly)] = 0.020
    sigma = 0.05 + 0.045 * (XX / Lx) + 0.035 * (YY / Ly)
    gamma_robin = 1.45
    M = 10
    centers = rng.uniform([0.14 * Lx, 0.14 * Ly], [0.86 * Lx, 0.86 * Ly], size=(M, 2))
    cand = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        cand[m] = np.exp(-((XX - centers[m, 0])**2 + (YY - centers[m, 1])**2) / (2.0 * (0.085**2)))
    A = _assemble_robin_spd(Lx, Ly, kappa, sigma, gamma_robin)
    basis_fields = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        basis_fields[m] = spsolve(A, (hx * hy) * cand[m].ravel()).reshape(nx, ny)
    K_s = 360
    fc = rng.choice(nx * ny, size=K_s, replace=False)
    sensor_ij = np.column_stack([fc // ny, fc % ny])
    sx, sy = xs[sensor_ij[:, 0]], ys[sensor_ij[:, 1]]
    Z = np.column_stack([np.ones(K_s), (sx - 0.5 * Lx) / Lx, (sy - 0.5 * Ly) / Ly, np.sin(np.pi * sx / Lx) * np.cos(np.pi * sy / Ly)])
    S = basis_fields[:, sensor_ij[:, 0], sensor_ij[:, 1]].T
    lam_z, lam_s = 1.5e-3, 2.0e-2
    Q_exp = 5
    Y_obs = Z @ rng.normal(0.0, 0.65, size=(4, Q_exp)) + S @ rng.normal(0.0, 0.95, size=(M, Q_exp)) + rng.normal(0.0, 2e-3, size=(K_s, Q_exp))
    P_z = np.linalg.solve(Z.T @ Z + lam_z * np.eye(4), Z.T)
    R_y = Y_obs - Z @ (P_z @ Y_obs)
    R_s = S - Z @ (P_z @ S)
    prof_amp = (S.T @ R_y) / np.sum(S * R_s, axis=0)[:, None]
    H_aug = np.block([[Z.T @ Z + lam_z * np.eye(4), Z.T @ S], [S.T @ Z, S.T @ S + lam_s * np.eye(M)]])
    joint_w = np.linalg.solve(H_aug, np.vstack([Z.T @ Y_obs, S.T @ Y_obs]))[4:, :]
    return {
        "Lx": Lx, "Ly": Ly, "kappa": kappa, "sigma": sigma, "gamma_robin": gamma_robin,
        "candidate_sources": cand, "sensor_ij": sensor_ij, "nuisance_design": Z,
        "nuisance_ridge": lam_z, "source_ridge": lam_s, "Y_obs": Y_obs,
    }, A, basis_fields, prof_amp, joint_w

t0 = time.monotonic()
valid = True
max_pde_res = 0.0
max_field_err = 0.0
max_prof_err = 0.0
max_joint_err = 0.0

for sd in (99211, 99212, 99213):
    prob, A, gt_bf, gt_prof, gt_jw = _make_priv(sd)
    out = robin_spd_profiler.solve_robin_spd_and_profile_sources({k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in prob.items()})
    if not isinstance(out, dict) or not {"basis_fields", "single_source_profiled_amplitudes", "joint_source_weights"} <= set(out.keys()):
        valid = False; break
    bf = np.asarray(out["basis_fields"], dtype=float)
    pa = np.asarray(out["single_source_profiled_amplitudes"], dtype=float)
    jw = np.asarray(out["joint_source_weights"], dtype=float)
    if bf.shape != gt_bf.shape or pa.shape != gt_prof.shape or jw.shape != gt_jw.shape:
        valid = False; break
    if not (np.all(np.isfinite(bf)) and np.all(np.isfinite(pa)) and np.all(np.isfinite(jw))):
        valid = False; break
    hx, hy = prob["Lx"] / 72, prob["Ly"] / 72
    for m in range(gt_bf.shape[0]):
        rhs = (hx * hy) * prob["candidate_sources"][m].ravel()
        res_rel = float(np.linalg.norm(A @ bf[m].ravel() - rhs) / max(np.linalg.norm(rhs), 1e-12))
        max_pde_res = max(max_pde_res, res_rel)
    max_field_err = max(max_field_err, float(np.linalg.norm(bf - gt_bf) / np.linalg.norm(gt_bf)))
    max_prof_err = max(max_prof_err, float(np.linalg.norm(pa - gt_prof) / np.linalg.norm(gt_prof)))
    max_joint_err = max(max_joint_err, float(np.linalg.norm(jw - gt_jw) / np.linalg.norm(gt_jw)))

elapsed = time.monotonic() - t0
passed = bool(
    valid
    and max_pde_res <= 1e-6
    and max_field_err <= 1e-5
    and max_prof_err <= 1e-5
    and max_joint_err <= 1e-5
    and elapsed < 25.0
)
print(json.dumps({
    "strict_pass": passed,
    "all_valid": valid,
    "max_pde_rel_residual": round(float(max_pde_res), 10),
    "max_field_rel_l2": round(float(max_field_err), 10),
    "max_profiled_amp_rel_l2": round(float(max_prof_err), 10),
    "max_joint_weight_rel_l2": round(float(max_joint_err), 10),
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # VR03: Pre-Window Viscoelastic De-Spiking + Reference Control Panel Calibration
    # =========================================================================
    HoldoutOODTask(
        task_id="vr03__prehistory_viscoelastic_control_panel_calibration",
        short_name="vr03_prehistory_viscoelastic_panel",
        wave="m90a_voluntary_reuse",
        scientific_domain="Viscoelastic Hereditary Memory Extraction, Reference Control Panel Alignment & Simpson-Deconfounded Risk",
        wormhole_family="fam_cohort_deconfounding_and_reference_balancing",
        declared_artifacts=("/holdout_task/viscoelastic_panel_calibration.py",),
        declared_callable_entrypoints=("calibrate_viscoelastic_cohort",),
        instruction_text=(
            "Implement `/holdout_task/viscoelastic_panel_calibration.py` exposing:\n"
            "`def calibrate_viscoelastic_cohort(cohort: dict) -> dict:`\n\n"
            "Two-Stage Hereditary Rheometry Extraction & Multi-Center Control-Panel Deconfounding Specification:\n"
            "- A multi-center biomaterials study evaluates specimens across `C = 6` hospital/rheometry centers (`0..3` in `'train'`, `4..5` in `'query'`)\n"
            "  and `S = 3` clinical strata (`0, 1, 2`, with every `(center, stratum)` cell nonempty in `'train'`).\n"
            "- STAGE 1 (`Per-Specimen Irregular-dt Pre-Window Prony Extraction with Cauchy Glitches`):\n"
            "  - Shared relaxation constants `taus = cohort['taus']` (`[tau_1, tau_2]`, `len = 2`).\n"
            "  - Each specimen `i` in `'train'` or `'query'` has a rheometer trace dict `traces[i]` with:\n"
            "    `past_timestamps`, `past_strain_history` (`len >= 3`, `past_timestamps[-1] < timestamps[0]`),\n"
            "    `timestamps` (`len = T = 36`, strictly increasing irregular grid), `strain_u` (`shape (T,)`), and `stress_obs` (`shape (T,)`).\n"
            "  - Roll out the exact zero-order-hold (ZOH) exponential memory states `m_r` (`r in {0, 1}`) starting at\n"
            "    `m_r = past_strain_history[0]` at `past_timestamps[0]`, stepping through `k = 1 .. len(past_timestamps)-1` via\n"
            "    `a = exp(-(past_timestamps[k] - past_timestamps[k-1]) / tau_r)`, `m_r <- a * m_r + (1 - a) * past_strain_history[k]`,\n"
            "    and then across main window `k = 0 .. T-1` (`dt_0 = timestamps[0] - past_timestamps[-1]`, `dt_k = timestamps[k] - timestamps[k-1]`):\n"
            "    `a_k = exp(-dt_k / tau_r)`, `m_r[k] = a_k * m_r_prev + (1 - a_k) * strain_u[k]`.\n"
            "  - Clean specimen stress follows `stress_clean[k] = c_intercept + X_raw[i, 0]*strain_u[k] + X_raw[i, 1]*m_1[k] + X_raw[i, 2]*m_2[k]`,\n"
            "    where ~10% of `stress_obs` timepoints suffer heavy Cauchy telemetry glitches (`+-4.5..9.0`)!\n"
            "    (Ignoring `past_strain_history` or using non-robust OLS severely corrupts `X_raw[i, 0..2]`!)\n"
            "  - Additionally, `missing_channel_mask` (`shape (N, 3)`, boolean) marks ~5% dropped sensor channels where `X_raw[i, d]` MUST be treated as `NaN`.\n"
            "- STAGE 2 (`Shared Scanner Affine Control-Panel Alignment & Simpson-Deconfounded Risk Calibration`):\n"
            "  - In center `c`, BOTH the specimen's raw viscoelastic biomarker vector `X_raw[i, :]` (`shape (3,)`) AND its companion\n"
            "    `Q = 5` reference-standard control panel `controls[i, :]` (`shape (N, 5)`, ~12% `NaN` missing entries, with `Z_control ~ N(0, I_5)`\n"
            "    in every center) undergo the SAME unknown center scanner transform:\n"
            "    `raw = scanner_offset[c] + scanner_gain[c] * latent` (`scanner_gain[c] > 0`).\n"
            "    (Across each center `c`, each valid control column `q` has latent population mean `0` and population std `1`, so\n"
            "    `scanner_offset[c]` and `scanner_gain[c]` are identified by the median across control columns of the per-column control mean and population std!)\n"
            "  - SIMPSON'S PARADOX: Training centers `0, 1` (`scanner_offset ~ +4.0`) have ~80% failure prevalence (`y = 1`) while\n"
            "    centers `2, 3` (`scanner_offset ~ -4.0`) have ~20% failure prevalence, and strata `0, 1, 2` have imbalanced sample counts!\n"
            "  - True failure log-odds obey `beta_0 + stratum_offset[s] + Z_latent @ beta` (`Z_latent` shape `(N, 3)`).\n"
            "    Impute any `NaN` in aligned `Z_latent` using cell-balanced training column means.\n"
            "- Return:\n"
            "  - `'aligned_query_moduli'`: 2D float array `(N_q, 3)` estimating `Z_latent` on `'query'` (`MAE <= 0.22`),\n"
            "  - `'query_probabilities'`: 1D float array `(N_q,)` in `(0, 1)` (`within-center AUROC >= 0.88`, `Brier <= 0.160`),\n"
            "  - `'modulus_weights'`: 1D float array `(3,)` of fitted latent biomarker log-odds weights.\n"
            "- Public development cohorts are in `/holdout_task/public_viscoelastic_cohort.pkl`."
        ),
        public_setup_py=r'''
import math, pickle, numpy as np

def _rollout_prony(taus, past_t, past_u, t, u):
    T = len(t)
    M = np.zeros((len(taus), T), dtype=float)
    for r, tau in enumerate(taus):
        m = float(past_u[0])
        for k in range(1, len(past_t)):
            a = math.exp(-(past_t[k] - past_t[k - 1]) / tau)
            m = a * m + (1.0 - a) * float(past_u[k])
        for k in range(T):
            dt = (t[0] - past_t[-1]) if k == 0 else (t[k] - t[k - 1])
            a = math.exp(-dt / tau)
            m = a * m + (1.0 - a) * float(u[k])
            M[r, k] = m
    return M

def _make_cohort(seed):
    rng = np.random.default_rng(seed)
    taus = [0.55, 2.60]
    D, Q = 3, 5
    offsets = np.array([4.1, 3.8, -4.0, -3.7, -3.5, 3.9], dtype=float)
    gains = np.array([1.40, 0.78, 1.30, 0.82, 1.48, 0.75], dtype=float)
    beta = np.array([2.15, -1.95, 1.80], dtype=float)
    strat_off = np.array([-0.85, 0.05, 0.90], dtype=float)

    def _build_split(center_specs):
        cids, strata, Z_all, ctrl_all, y_all, p_all, mask_all, traces = [], [], [], [], [], [], [], []
        for c, n_c, z_bias in center_specs:
            # Guarantee every stratum {0, 1, 2} appears >= 5 times in each center!
            s_base = np.array([0, 1, 2] * 5, dtype=int)
            s_rest = rng.choice([0, 1, 2], size=n_c - len(s_base), p=[0.58, 0.27, 0.15])
            s = rng.permutation(np.concatenate([s_base, s_rest]))
            Z = rng.normal(z_bias, 1.0, size=(n_c, D))
            logits = -0.05 + strat_off[s] + Z @ beta
            probs = 1.0 / (1.0 + np.exp(-logits))
            y = (rng.uniform(size=n_c) < probs).astype(int)
            X_raw = offsets[c] + gains[c] * Z
            Z_ctrl = rng.normal(0.0, 1.0, size=(n_c, Q))
            ctrl = offsets[c] + gains[c] * Z_ctrl
            ctrl[rng.uniform(size=ctrl.shape) < 0.12] = np.nan
            mm = rng.uniform(size=(n_c, D)) < 0.05
            for i in range(n_c):
                past_t = np.cumsum(np.concatenate([[0.0], rng.uniform(0.18, 0.45, size=5)]))
                # Strong pre-window step/hold loading so ignoring past_strain_history causes huge bias!
                past_u = rng.choice([-1.0, 1.0]) * np.array([0.2, 1.8, 2.4, 2.6, 2.5, 2.3], dtype=float)
                dt_main = rng.uniform(0.08, 0.26, size=36)
                t = past_t[-1] + np.cumsum(dt_main)
                u = 0.9 * np.sin(1.7 * (t - t[0])) + 0.55 * np.cos(3.1 * (t - t[0])) + 0.35 * ((t - t[0]) / (t[-1] - t[0]))
                M_pr = _rollout_prony(taus, past_t, past_u, t, u)
                c0 = float(rng.uniform(-1.2, 1.2))
                stress = c0 + X_raw[i, 0] * u + X_raw[i, 1] * M_pr[0] + X_raw[i, 2] * M_pr[1] + rng.normal(0.0, 0.015, size=len(t))
                gl_idx = rng.choice(len(t), size=4, replace=False)
                stress[gl_idx] += rng.choice([-1.0, 1.0], size=4) * rng.uniform(4.8, 8.5, size=4)
                traces.append({
                    "past_timestamps": past_t, "past_strain_history": past_u,
                    "timestamps": t, "strain_u": u, "stress_obs": stress,
                })
            cids.append(np.full(n_c, c, dtype=int))
            strata.append(s); Z_all.append(Z); ctrl_all.append(ctrl)
            y_all.append(y); p_all.append(probs); mask_all.append(mm)
        return {
            "center_id": np.concatenate(cids),
            "stratum": np.concatenate(strata),
            "controls": np.vstack(ctrl_all),
            "missing_channel_mask": np.vstack(mask_all),
            "traces": traces,
            "y": np.concatenate(y_all),
            "gt_Z": np.vstack(Z_all),
            "gt_p": np.concatenate(p_all),
        }

    tr = _build_split([(0, 72, 0.48), (1, 72, 0.44), (2, 72, -0.48), (3, 72, -0.44)])
    qr = _build_split([(4, 90, 0.40), (5, 90, -0.40)])
    return {
        "cohort": {
            "taus": taus,
            "train": {k: tr[k] for k in ("center_id", "stratum", "controls", "missing_channel_mask", "traces", "y")},
            "query": {k: qr[k] for k in ("center_id", "stratum", "controls", "missing_channel_mask", "traces")},
        },
        "gt": {"Z_query": qr["gt_Z"], "y_query": qr["y"], "prob_query": qr["gt_p"], "beta": beta},
    }

pub = [_make_cohort(701), _make_cohort(702)]
with open("/holdout_task/public_viscoelastic_cohort.pkl", "wb") as fh:
    pickle.dump(pub, fh)
''',
        private_verifier_py=r'''
import json, math, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/holdout_task")
import viscoelastic_panel_calibration

def _rollout_prony(taus, past_t, past_u, t, u):
    T = len(t)
    M = np.zeros((len(taus), T), dtype=float)
    for r, tau in enumerate(taus):
        m = float(past_u[0])
        for k in range(1, len(past_t)):
            a = math.exp(-(past_t[k] - past_t[k - 1]) / tau)
            m = a * m + (1.0 - a) * float(past_u[k])
        for k in range(T):
            dt = (t[0] - past_t[-1]) if k == 0 else (t[k] - t[k - 1])
            a = math.exp(-dt / tau)
            m = a * m + (1.0 - a) * float(u[k])
            M[r, k] = m
    return M

def _make_cohort(seed):
    rng = np.random.default_rng(seed)
    taus = [0.55, 2.60]
    D, Q = 3, 5
    offsets = np.array([4.2, 3.9, -4.1, -3.8, -3.6, 4.0], dtype=float)
    gains = np.array([1.42, 0.76, 1.32, 0.84, 1.50, 0.74], dtype=float)
    beta = np.array([2.15, -1.95, 1.80], dtype=float)
    strat_off = np.array([-0.85, 0.05, 0.90], dtype=float)

    def _build_split(center_specs):
        cids, strata, Z_all, ctrl_all, y_all, mask_all, traces = [], [], [], [], [], [], []
        for c, n_c, z_bias in center_specs:
            s_base = np.array([0, 1, 2] * 5, dtype=int)
            s_rest = rng.choice([0, 1, 2], size=n_c - len(s_base), p=[0.58, 0.27, 0.15])
            s = rng.permutation(np.concatenate([s_base, s_rest]))
            Z = rng.normal(z_bias, 1.0, size=(n_c, D))
            logits = -0.05 + strat_off[s] + Z @ beta
            probs = 1.0 / (1.0 + np.exp(-logits))
            y = (rng.uniform(size=n_c) < probs).astype(int)
            X_raw = offsets[c] + gains[c] * Z
            Z_ctrl = rng.normal(0.0, 1.0, size=(n_c, Q))
            ctrl = offsets[c] + gains[c] * Z_ctrl
            ctrl[rng.uniform(size=ctrl.shape) < 0.12] = np.nan
            mm = rng.uniform(size=(n_c, D)) < 0.05
            for i in range(n_c):
                past_t = np.cumsum(np.concatenate([[0.0], rng.uniform(0.18, 0.45, size=5)]))
                past_u = rng.choice([-1.0, 1.0]) * np.array([0.2, 1.8, 2.4, 2.6, 2.5, 2.3], dtype=float)
                dt_main = rng.uniform(0.08, 0.26, size=36)
                t = past_t[-1] + np.cumsum(dt_main)
                u = 0.9 * np.sin(1.7 * (t - t[0])) + 0.55 * np.cos(3.1 * (t - t[0])) + 0.35 * ((t - t[0]) / (t[-1] - t[0]))
                M_pr = _rollout_prony(taus, past_t, past_u, t, u)
                c0 = float(rng.uniform(-1.2, 1.2))
                stress = c0 + X_raw[i, 0] * u + X_raw[i, 1] * M_pr[0] + X_raw[i, 2] * M_pr[1] + rng.normal(0.0, 0.015, size=len(t))
                gl_idx = rng.choice(len(t), size=4, replace=False)
                stress[gl_idx] += rng.choice([-1.0, 1.0], size=4) * rng.uniform(4.8, 8.5, size=4)
                traces.append({
                    "past_timestamps": past_t, "past_strain_history": past_u,
                    "timestamps": t, "strain_u": u, "stress_obs": stress,
                })
            cids.append(np.full(n_c, c, dtype=int))
            strata.append(s); Z_all.append(Z); ctrl_all.append(ctrl)
            y_all.append(y); mask_all.append(mm)
        return {
            "center_id": np.concatenate(cids),
            "stratum": np.concatenate(strata),
            "controls": np.vstack(ctrl_all),
            "missing_channel_mask": np.vstack(mask_all),
            "traces": traces,
            "y": np.concatenate(y_all),
            "gt_Z": np.vstack(Z_all),
        }

    tr = _build_split([(0, 76, 0.48), (1, 76, 0.44), (2, 76, -0.48), (3, 76, -0.44)])
    qr = _build_split([(4, 90, 0.40), (5, 90, -0.40)])
    return {
        "taus": taus,
        "train": {k: tr[k] for k in ("center_id", "stratum", "controls", "missing_channel_mask", "traces", "y")},
        "query": {k: qr[k] for k in ("center_id", "stratum", "controls", "missing_channel_mask", "traces")},
    }, qr["gt_Z"], qr["y"], qr["center_id"]

t0 = time.monotonic()
valid = True
z_maes, min_aurocs, briers = [], [], []
for sd in (99221, 99222, 99223):
    coh, gt_Z, gt_y, q_cid = _make_cohort(sd)
    out = viscoelastic_panel_calibration.calibrate_viscoelastic_cohort(coh)
    if not isinstance(out, dict) or not {"aligned_query_moduli", "query_probabilities", "modulus_weights"} <= set(out.keys()):
        valid = False; break
    Zq = np.asarray(out["aligned_query_moduli"], dtype=float)
    pq = np.asarray(out["query_probabilities"], dtype=float)
    mw = np.asarray(out["modulus_weights"], dtype=float)
    if Zq.shape != gt_Z.shape or pq.shape != gt_y.shape or mw.shape != (3,):
        valid = False; break
    if not (np.all(np.isfinite(Zq)) and np.all(np.isfinite(pq)) and np.all(np.isfinite(mw))):
        valid = False; break
    if np.any(pq < 0.0) or np.any(pq > 1.0):
        valid = False; break
    # Evaluate MAE on non-imputed observed channels (and overall)
    obs_m = ~coh["query"]["missing_channel_mask"]
    z_maes.append(float(np.mean(np.abs(Zq[obs_m] - gt_Z[obs_m]))))
    briers.append(float(np.mean((pq - gt_y) ** 2)))
    c_aucs = [float(roc_auc_score(gt_y[q_cid == c], pq[q_cid == c])) for c in (4, 5)]
    min_aurocs.append(min(c_aucs))

elapsed = time.monotonic() - t0
max_mae = max(z_maes) if (valid and len(z_maes) == 3) else float("nan")
worst_auc = min(min_aurocs) if (valid and len(min_aurocs) == 3) else float("nan")
max_brier = max(briers) if (valid and len(briers) == 3) else float("nan")
passed = bool(
    valid
    and math.isfinite(max_mae) and max_mae <= 0.22
    and math.isfinite(worst_auc) and worst_auc >= 0.88
    and math.isfinite(max_brier) and max_brier <= 0.160
    and elapsed < 25.0
)
print(json.dumps({
    "strict_pass": passed,
    "all_valid": valid,
    "max_aligned_moduli_mae": (round(float(max_mae), 5) if math.isfinite(max_mae) else -1.0),
    "worst_within_center_auroc": (round(float(worst_auc), 5) if math.isfinite(worst_auc) else -1.0),
    "max_brier_score": (round(float(max_brier), 5) if math.isfinite(max_brier) else -1.0),
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # VR04: Indefinite Complex Helmholtz Scattering & Wrapped-Phase Inversion
    # =========================================================================
    HoldoutOODTask(
        task_id="vr04__indefinite_helmholtz_wrapped_phase_inversion",
        short_name="vr04_indefinite_helmholtz_phase",
        wave="m90a_voluntary_reuse",
        scientific_domain="Indefinite Acoustic Helmholtz Scattering & Wrapped-Phase Receiver Calibration",
        wormhole_family="fam_indefinite_wave_scattering_negative_control",
        declared_artifacts=("/holdout_task/helmholtz_phase_inversion.py",),
        declared_callable_entrypoints=("invert_helmholtz_sources_and_phase",),
        instruction_text=(
            "Implement `/holdout_task/helmholtz_phase_inversion.py` exposing:\n"
            "`def invert_helmholtz_sources_and_phase(problem: dict) -> dict:`\n\n"
            "Indefinite Complex Acoustic Helmholtz & Wrapped-Phase Telemetry Specification:\n"
            "- Time-harmonic 2D acoustic pressure `u(x, y) in C` (`complex128`) on `[0, Lx] x [0, Ly]` (`(nx, ny) = (32, 32)` uniform cell-centered grid,\n"
            "  `hx = Lx / nx`, `hy = Ly / ny`) obeys the INDEFINITE Helmholtz equation above cavity resonance (`k0 ~ 11.5`, real part has ~15 negative eigenvalues!):\n"
            "  `-Laplacian(u) - (k0 ** 2) * n_sq(x, y) * u = sum_{m=0}^{M-1} a_m * basis[m, i, j]`\n"
            "  with outgoing first-order Sommerfeld absorbing radiation boundary condition `(-grad u) . n = -1j * k0 * beta_imp * u` on all four exterior walls.\n"
            "- Exact cell-centered finite-volume stencil (`1,024 x 1,024` complex symmetric non-Hermitian indefinite matrix `H` in C-order `ravel()`):\n"
            "  1. Interior x-faces `(i+1/2, j)` contribute `(hy / hx) * (u[i,j] - u[i+1,j])`; interior y-faces `(i, j+1/2)` contribute `(hx / hy) * (u[i,j] - u[i,j+1])`.\n"
            "  2. Each exterior boundary face of cell `(i, j)` contributes `-length_face * (1j * k0 * beta_imp) * u[i,j]` (`length_face = hy` on `i=0, nx-1`; `hx` on `j=0, ny-1`).\n"
            "  3. Interior Helmholtz wave term contributes `-(hx * hy) * (k0 ** 2) * n_sq[i,j] * u[i,j]`, with RHS `(hx * hy) * q[i,j]`.\n"
            "  (WARNING: `H` is complex and strictly INDEFINITE — real-only SPD / Cholesky / conjugate-gradient routines diverge or fail preconditions!)\n"
            "- Receiver Polar Telemetry & Wrapped-Phase Ambiguity:\n"
            "  - All receiver channels share an unknown global oscillator phase offset `global_phase_rad in (-pi, pi]`:\n"
            "    clean complex measurement is `z_clean = exp(1j * global_phase_rad) * u_true`, reported in polar form as\n"
            "    `amplitude_obs = |z_obs|` and `wrapped_phase_obs = atan2(Im(z_obs), Re(z_obs)) in (-pi, pi]`!\n"
            "  - `problem['pilot_calibration']` provides `K_p = 12` clean reference pilot measurements\n"
            "    `{'known_field_re': (K_p,), 'known_field_im': (K_p,), 'amplitude_obs': (K_p,), 'wrapped_phase_obs': (K_p,)}`\n"
            "    (where `known_complex_field = known_field_re + 1j * known_field_im`) under the SAME `global_phase_rad`\n"
            "    (uniquely resolving global phase without `pi` sign ambiguity).\n"
            "  - `problem['survey']` provides `{'sensor_ij': (K_s, 2), 'amplitude_obs': (K_s,), 'wrapped_phase_obs': (K_s,), 'tikhonov_alpha': float}`\n"
            "    for the unknown real source vector `source_amplitudes` (`shape (M,)`, `M = 6`), where ~7% of survey sensors suffer heavy\n"
            "    complex impulse/multipath spikes (`outlier amplitude/phase jumps`)!\n"
            "- Return:\n"
            "  - `'source_amplitudes'`: 1D float array `(M,)` (`rel L2 error <= 0.030`),\n"
            "  - `'global_phase_rad'`: float in `(-pi, pi]` (`circular abs error <= 0.020 rad`),\n"
            "  - `'predicted_field_abs'`: 2D float array `(nx, ny)` equal to `|u(x, y)|` (`rel L2 error <= 0.030`).\n"
            "- Public test cases are in `/holdout_task/public_helmholtz_cases.pkl`."
        ),
        public_setup_py=r'''
import math, pickle, numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def _assemble_helmholtz(Lx, Ly, k0, n_sq, beta_imp):
    nx, ny = n_sq.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    rows, cols, data = [], [], []
    diag = -(hx * hy) * (k0 ** 2) * n_sq.ravel().astype(np.complex128)

    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    wx = np.full(i0.size, hy / hx, dtype=np.complex128)
    diag[i0] += wx; diag[i1] += wx
    rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-wx, -wx])

    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    wy = np.full(j0.size, hx / hy, dtype=np.complex128)
    diag[j0] += wy; diag[j1] += wy
    rows.extend([j0, j1]); cols.extend([j1, j0]); data.extend([-wy, -wy])

    for xb in (0, -1):
        diag[ids[xb, :].ravel()] += -hy * (1j * k0 * beta_imp)
    for yb in (0, -1):
        diag[ids[:, yb].ravel()] += -hx * (1j * k0 * beta_imp)

    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _make_case(seed):
    rng = np.random.default_rng(seed)
    nx = ny = 32
    Lx, Ly = 1.0, 1.0
    hx, hy = Lx / nx, Ly / ny
    xs = (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    k0 = 11.5
    beta_imp = 0.85
    n_sq = 1.0 + 0.95 * np.exp(-((XX - 0.48)**2 + (YY - 0.52)**2) / (2.0 * (0.18**2))) - 0.20 * np.cos(2.0 * np.pi * XX)

    M = 6
    ctrs = np.array([[0.22, 0.24], [0.78, 0.24], [0.24, 0.76], [0.76, 0.76], [0.50, 0.28], [0.50, 0.72]], dtype=float)
    basis = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        basis[m] = 25.0 * np.exp(-((XX - ctrs[m, 0])**2 + (YY - ctrs[m, 1])**2) / (2.0 * (0.065**2)))

    H = _assemble_helmholtz(Lx, Ly, k0, n_sq, beta_imp)
    G_full = np.zeros((M, nx, ny), dtype=np.complex128)
    for m in range(M):
        G_full[m] = spsolve(H, (hx * hy) * basis[m].ravel()).reshape(nx, ny)

    amps = rng.uniform(-1.5, 1.6, size=M)
    amps[0] = 1.35
    phi0 = float(rng.uniform(-2.7, 2.7))
    u_true = np.tensordot(amps, G_full, axes=(0, 0))

    # Pilot calibration
    K_p = 12
    known_pilot = (rng.normal(0.0, 1.0, size=K_p) + 1j * rng.normal(0.0, 1.0, size=K_p)) * 0.5 + 0.8
    z_pilot = np.exp(1j * phi0) * known_pilot + (rng.normal(0.0, 2e-3, size=K_p) + 1j * rng.normal(0.0, 2e-3, size=K_p))

    # Survey with ~7% complex spikes & branch-cut phase wrapping
    K_s = 140
    fc = rng.choice(nx * ny, size=K_s, replace=False)
    s_ij = np.column_stack([fc // ny, fc % ny])
    z_surv = np.exp(1j * phi0) * u_true[s_ij[:, 0], s_ij[:, 1]] + (rng.normal(0.0, 1.5e-3, size=K_s) + 1j * rng.normal(0.0, 1.5e-3, size=K_s))
    sp_idx = rng.choice(K_s, size=10, replace=False)
    z_surv[sp_idx] += (rng.choice([-1.0, 1.0], size=10) * rng.uniform(0.35, 0.75, size=10)) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=10))

    return {
        "problem": {
            "Lx": Lx, "Ly": Ly, "k0": k0, "beta_imp": beta_imp, "n_sq": n_sq, "basis": basis,
            "pilot_calibration": {
                "known_field_re": known_pilot.real.copy(),
                "known_field_im": known_pilot.imag.copy(),
                "amplitude_obs": np.abs(z_pilot),
                "wrapped_phase_obs": np.angle(z_pilot),
            },
            "survey": {
                "sensor_ij": s_ij,
                "amplitude_obs": np.abs(z_surv),
                "wrapped_phase_obs": np.angle(z_surv),
                "tikhonov_alpha": 1e-6,
            },
        },
        "gt": {
            "source_amplitudes": amps,
            "global_phase_rad": phi0,
            "predicted_field_abs": np.abs(u_true),
        },
    }

pub = [_make_case(801), _make_case(802)]
with open("/holdout_task/public_helmholtz_cases.pkl", "wb") as fh:
    pickle.dump(pub, fh)
''',
        private_verifier_py=r'''
import json, math, sys, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, "/holdout_task")
import helmholtz_phase_inversion

def _assemble_helmholtz(Lx, Ly, k0, n_sq, beta_imp):
    nx, ny = n_sq.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    ids = np.arange(N).reshape(nx, ny)
    rows, cols, data = [], [], []
    diag = -(hx * hy) * (k0 ** 2) * n_sq.ravel().astype(np.complex128)
    i0, i1 = ids[:-1, :].ravel(), ids[1:, :].ravel()
    wx = np.full(i0.size, hy / hx, dtype=np.complex128)
    diag[i0] += wx; diag[i1] += wx
    rows.extend([i0, i1]); cols.extend([i1, i0]); data.extend([-wx, -wx])
    j0, j1 = ids[:, :-1].ravel(), ids[:, 1:].ravel()
    wy = np.full(j0.size, hx / hy, dtype=np.complex128)
    diag[j0] += wy; diag[j1] += wy
    rows.extend([j0, j1]); cols.extend([j1, j0]); data.extend([-wy, -wy])
    for xb in (0, -1):
        diag[ids[xb, :].ravel()] += -hy * (1j * k0 * beta_imp)
    for yb in (0, -1):
        diag[ids[:, yb].ravel()] += -hx * (1j * k0 * beta_imp)
    rows.append(ids.ravel()); cols.append(ids.ravel()); data.append(diag)
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))

def _make_priv(seed):
    rng = np.random.default_rng(seed)
    nx = ny = 32
    Lx, Ly = 1.05, 0.95
    hx, hy = Lx / nx, Ly / ny
    xs = (np.arange(nx) + 0.5) * hx
    ys = (np.arange(ny) + 0.5) * hy
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    k0 = 11.8
    beta_imp = 0.80
    n_sq = 1.05 + 0.90 * np.exp(-((XX - 0.52 * Lx)**2 + (YY - 0.47 * Ly)**2) / (2.0 * (0.17**2))) - 0.18 * np.sin(2.0 * np.pi * YY / Ly)
    M = 6
    base_ctrs = np.array(
        [[0.22 * Lx, 0.24 * Ly], [0.78 * Lx, 0.24 * Ly], [0.24 * Lx, 0.76 * Ly],
         [0.76 * Lx, 0.76 * Ly], [0.50 * Lx, 0.28 * Ly], [0.50 * Lx, 0.72 * Ly]],
        dtype=float,
    )
    ctrs = base_ctrs + rng.uniform(-0.035, 0.035, size=(M, 2))
    basis = np.zeros((M, nx, ny), dtype=float)
    for m in range(M):
        basis[m] = 25.0 * np.exp(-((XX - ctrs[m, 0])**2 + (YY - ctrs[m, 1])**2) / (2.0 * (0.068**2)))
    H = _assemble_helmholtz(Lx, Ly, k0, n_sq, beta_imp)
    G_full = np.zeros((M, nx, ny), dtype=np.complex128)
    for m in range(M):
        G_full[m] = spsolve(H, (hx * hy) * basis[m].ravel()).reshape(nx, ny)
    amps = rng.uniform(-1.6, 1.6, size=M); amps[1] = -1.25
    phi0 = float(rng.uniform(-2.85, 2.85))
    u_true = np.tensordot(amps, G_full, axes=(0, 0))
    K_p = 12
    known_pilot = (rng.normal(0.0, 1.0, size=K_p) + 1j * rng.normal(0.0, 1.0, size=K_p)) * 0.5 + 0.8
    z_pilot = np.exp(1j * phi0) * known_pilot + (rng.normal(0.0, 2e-3, size=K_p) + 1j * rng.normal(0.0, 2e-3, size=K_p))
    K_s = 140
    fc = rng.choice(nx * ny, size=K_s, replace=False)
    s_ij = np.column_stack([fc // ny, fc % ny])
    z_surv = np.exp(1j * phi0) * u_true[s_ij[:, 0], s_ij[:, 1]] + (rng.normal(0.0, 1.5e-3, size=K_s) + 1j * rng.normal(0.0, 1.5e-3, size=K_s))
    sp_idx = rng.choice(K_s, size=10, replace=False)
    z_surv[sp_idx] += (rng.choice([-1.0, 1.0], size=10) * rng.uniform(0.35, 0.75, size=10)) * np.exp(1j * rng.uniform(-np.pi, np.pi, size=10))
    return {
        "Lx": Lx, "Ly": Ly, "k0": k0, "beta_imp": beta_imp, "n_sq": n_sq, "basis": basis,
        "pilot_calibration": {
            "known_field_re": known_pilot.real.copy(),
            "known_field_im": known_pilot.imag.copy(),
            "amplitude_obs": np.abs(z_pilot),
            "wrapped_phase_obs": np.angle(z_pilot),
        },
        "survey": {
            "sensor_ij": s_ij,
            "amplitude_obs": np.abs(z_surv),
            "wrapped_phase_obs": np.angle(z_surv),
            "tikhonov_alpha": 1e-6,
        },
    }, amps, phi0, np.abs(u_true)

t0 = time.monotonic()
valid = True
max_amp_err = 0.0
max_phase_err = 0.0
max_field_err = 0.0
for sd in (99231, 99232, 99233):
    prob, gt_a, gt_p, gt_f = _make_priv(sd)
    out = helmholtz_phase_inversion.invert_helmholtz_sources_and_phase(prob)
    if not isinstance(out, dict) or not {"source_amplitudes", "global_phase_rad", "predicted_field_abs"} <= set(out.keys()):
        valid = False; break
    sa = np.asarray(out["source_amplitudes"], dtype=float)
    gp = out["global_phase_rad"]
    pf = np.asarray(out["predicted_field_abs"], dtype=float)
    if isinstance(gp, (bool, np.bool_)) or not isinstance(gp, (int, float, np.number)) or not math.isfinite(float(gp)):
        valid = False; break
    if sa.shape != gt_a.shape or pf.shape != gt_f.shape or not (np.all(np.isfinite(sa)) and np.all(np.isfinite(pf))):
        valid = False; break
    gp_f = float(gp)
    circ_err = abs(math.atan2(math.sin(gp_f - gt_p), math.cos(gp_f - gt_p)))
    max_phase_err = max(max_phase_err, circ_err)
    max_amp_err = max(max_amp_err, float(np.linalg.norm(sa - gt_a) / np.linalg.norm(gt_a)))
    max_field_err = max(max_field_err, float(np.linalg.norm(pf - gt_f) / np.linalg.norm(gt_f)))

elapsed = time.monotonic() - t0
passed = bool(
    valid
    and max_amp_err <= 0.030
    and max_phase_err <= 0.020
    and max_field_err <= 0.030
    and elapsed < 20.0
)
print(json.dumps({
    "strict_pass": passed,
    "all_valid": valid,
    "max_source_rel_l2": round(float(max_amp_err), 6),
    "max_phase_abs_err_rad": round(float(max_phase_err), 6),
    "max_field_abs_rel_l2": round(float(max_field_err), 6),
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),
)


VOLUNTARY_REUSE_M90A_BY_ID: dict[str, HoldoutOODTask] = {t.task_id: t for t in VOLUNTARY_REUSE_M90A_TASKS}


def voluntary_reuse_task_to_frontier_task(ht: HoldoutOODTask) -> m6f.FrontierBenchmarkTask:
    ensure_clean_scipy_sandbox_image()
    return m6f.FrontierBenchmarkTask(
        task_id=ht.task_id,
        short_name=ht.short_name,
        benchmark="Hegel-M90a-VoluntaryReuse-4",
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
        agent_memory_mb=2048,
        verifier_cpus=2,
        verifier_memory_mb=2048,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
    )


def get_voluntary_reuse_m90a_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [voluntary_reuse_task_to_frontier_task(t) for t in VOLUNTARY_REUSE_M90A_TASKS]


def run_voluntary_reuse_private_verifier_by_id(task_id: str, submitted_tar_path: Path | None) -> dict[str, Any]:
    ht = VOLUNTARY_REUSE_M90A_BY_ID[task_id]
    rep = run_holdout_private_verifier(ht, submitted_tar_path)
    rep["benchmark"] = "Hegel-M90a-VoluntaryReuse-4"
    return rep


def run_voluntary_reuse_private_verifier_for_container(task_id: str, agent_cid: str) -> dict[str, Any]:
    ht = VOLUNTARY_REUSE_M90A_BY_ID[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_vr_m90a_ver_") as td:
        tar_p = Path(td) / "submitted.tar"
        m6f.export_declared_artifacts_tar(agent_cid, ht.declared_artifacts, tar_p)
        rep = run_holdout_private_verifier(ht, tar_p)
        rep["benchmark"] = "Hegel-M90a-VoluntaryReuse-4"
        return rep

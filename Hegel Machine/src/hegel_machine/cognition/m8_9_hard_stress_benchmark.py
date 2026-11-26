"""Hegel M8.9 Hard-Stress Scientific Benchmark (`Hegel-M89-HardStress-4`).

Constructs 4 high-discrimination scientific inverse, optimization, hereditary-memory, and multi-center
calibration stress tasks (`hs01..hs04`) designed specifically to break ceiling saturation (`8/8 vs 8/8`)
and directly test:
  1. `hs01__correlated_box_l0_subcube_certificate`:
     Correlated (`rho ~ 0.72`) box-constrained `L0 + ridge` sparse regression (`p=14..15`) where the root
     relaxation has a large non-zero gap (`> 0.12`) and collinear decoy features defeat greedy single-feature
     screening, requiring a **replayable disjoint subcube partition certificate** (`certificate_leaves`) in
     which EVERY leaf's `leaf_lower_bound` is verified against the true convex box-QP relaxation on that leaf!
  2. `hs02__rotated_tensor_robin_darcy_inversion`:
     2D steady-state **full rotated SPD tensor** (`K_xy != 0`) diffusion-reaction with **Robin boundary fluxes**
     and Cauchy-corrupted + baseline-shifted sparse sensors — testing negative-transfer veto of zero-ghost
     scalar grid stencils (`autogenous_prim_conservative_spd_solve.conservative_grid_operator`) while rewarding
     robust affine nuisance + source inversion.
  3. `hs03__irregular_dt_hereditary_memory_deconvolution`:
     Irregular-`dt` two-mode Prony viscoelastic hereditary memory identification and regularized input-driver
     deconvolution with **non-zero pre-window `past_driver_history`** and telemetry Cauchy spikes.
  4. `hs04__missing_panel_multicenter_stratum_calibration`:
     Multi-center biomarker assay calibration under severe **Simpson's-paradox center prevalence confounding**,
     clinical stratum imbalance, and **partially missing (`NaN`) / sparse reference control panels** (including
     one center-marker pair with `< 2` finite reference controls, directly exercising the `REQ:` precondition
     of `autogenous_prim_reference_panel_balancing`).

Epistemic provenance note (`Hegel_M8_8_Audit` §九):
- This suite is an **internal frozen-code stress benchmark** committed before the `M8.9` live evaluation run
  (not an external third-party benchmark).
- Public container setup scripts (`public_setup_py`) use ONLY public seeds (`20261131..20261134`) and contain
  ZERO private verifier seeds (`99101..99104`) or evaluation thresholds beyond the task specification.
- Every private verifier executes inside a fresh air-gapped `CLEAN_SCIPY_SANDBOX_IMAGE` container (`network_mode="none"`,
  no `/tmp` copy) and enforces strict `dict`/type/non-`bool`/shape/`np.isfinite` and `allow_nan=False` JSON checks.
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


HARD_STRESS_M89_TASKS: tuple[HoldoutOODTask, ...] = (
    # =========================================================================
    # HS01: Correlated Box-L0 Regression with Replayable Subcube Certificate
    # =========================================================================
    HoldoutOODTask(
        task_id="hs01__correlated_box_l0_subcube_certificate",
        short_name="hs01_correlated_box_l0_cert",
        wave="m89_hard_stress",
        scientific_domain="Global Combinatorial Optimization & Replayable Dual Subcube Certificates",
        wormhole_family="fam_sparse_cert_and_combinatorial_pruning",
        declared_artifacts=("/holdout_task/certified_box_l0.py",),
        declared_callable_entrypoints=("solve_certified_box_l0",),
        instruction_text=(
            "Implement `/holdout_task/certified_box_l0.py` exposing:\n"
            "`def solve_certified_box_l0(problem: dict) -> dict:`\n\n"
            "Mathematical Specification:\n"
            "- Given `problem` containing:\n"
            "  - `'X'`: 2D float array `(n, p)` (`p <= 16`) with strongly correlated columns and collinear decoys,\n"
            "  - `'y'`: 1D float array `(n,)`,\n"
            "  - `'ridge'`: float `lambda_2 >= 0.0`,\n"
            "  - `'l0_penalty'`: float `lambda_0 > 0.0`,\n"
            "  - `'box'`: float `M > 0.0`,\n"
            "  - `'gap_tol'`: float (`1e-6`),\n"
            "  - `'max_leaves'`: int (`512`).\n"
            "- Minimize over `w in [-M, M]^p`:\n"
            "  `J(w) = 0.5 * ||X @ w - y||_2^2 + 0.5 * ridge * ||w||_2^2 + l0_penalty * count_nonzero(|w| > 1e-10)`.\n"
            "- Return a dictionary with:\n"
            "  - `'weights'`: 1D list/array of length `p` in `[-M, M]` achieving global optimality within `gap_tol`,\n"
            "  - `'objective'`: float `J(weights)` (`|objective - J(weights)| <= 1e-7`),\n"
            "  - `'certified_lower_bound'`: float global lower bound satisfying `0 <= objective - certified_lower_bound <= gap_tol`,\n"
            "  - `'certificate_leaves'`: non-empty list (`1 <= len <= max_leaves` — a single valid root certificate leaf `len == 1` IS accepted whenever mathematically sound!) of dicts\n"
            "    `{'fixed_zero': list[int], 'fixed_nonzero': list[int], 'leaf_lower_bound': float}`\n"
            "    (each leaf may optionally also include `'dual_nu': list[float]` of length `n` or `'minorant_diagonal'`/`'minorant_center'` as a proof-carrying witness)\n"
            "    forming an **exact disjoint exhaustive partition** of `{0, 1}^p` such that:\n"
            "    1. For every leaf, `fixed_zero` and `fixed_nonzero` are disjoint subsets of `{0, ..., p-1}`,\n"
            "    2. The subcubes `{z in {0,1}^p : z[fixed_zero]=0, z[fixed_nonzero]=1}` are pairwise disjoint and sum in measure to `2^p`,\n"
            "    3. Every leaf's `'leaf_lower_bound'` is a **sound mathematical lower bound** on the true discrete subcube infimum\n"
            "       `min_{w: |w[fixed_zero]|<=1e-10, |w[fixed_nonzero]|>1e-10, ||w||_inf <= M} J(w)`\n"
            "       (warning: fake lower bounds derived from heuristic trial points or non-PSD diagonals `G - diag(d) not PSD` exceed the subcube infimum and FAIL verification!),\n"
            "    4. `certified_lower_bound <= min(leaf['leaf_lower_bound'] for leaf in certificate_leaves) + 1e-8`.\n"
            "- Public development instances are saved in `/holdout_task/public_box_l0_cases.pkl`.\n"
            "- Must solve 3 private instances (`p in {14, 15}`) within `< 20s` total using standard Python/NumPy/SciPy."
        ),
        public_setup_py=r'''
import pickle, numpy as np

def _make_corr_case(seed, n, p, ridge, l0_pen, box):
    rng = np.random.default_rng(seed)
    z_lat = rng.normal(size=(n, 4))
    X = np.zeros((n, p), dtype=float)
    for j in range(p):
        X[:, j] = 0.72 * z_lat[:, j % 4] + 0.69 * rng.normal(size=n)
    # Plant a collinear decoy: feature 3 ~ 0.5*(feature 0 + feature 1) + small noise
    X[:, 3] = 0.52 * X[:, 0] + 0.52 * X[:, 1] + 0.14 * rng.normal(size=n)
    X /= np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-8) / np.sqrt(n)
    w_true = np.zeros(p, dtype=float)
    w_true[0] = 0.78 * box
    w_true[1] = 0.74 * box
    w_true[6] = -0.68 * box
    w_true[10] = 0.55 * box
    y = X @ w_true + rng.normal(0.0, 0.16, size=n)
    return {
        "X": X, "y": y, "ridge": float(ridge), "l0_penalty": float(l0_pen),
        "box": float(box), "gap_tol": 1e-6, "max_leaves": 512,
    }

cases = [
    _make_corr_case(20261131, 44, 14, 0.08, 0.32, 1.2),
    _make_corr_case(20261132, 48, 14, 0.04, 0.28, 1.1),
]
with open("/holdout_task/public_box_l0_cases.pkl", "wb") as f:
    pickle.dump({"cases": cases}, f)
''',
        private_verifier_py=r'''
import itertools, json, math, sys, time, numpy as np
from scipy.optimize import lsq_linear
sys.path.insert(0, "/holdout_task")
import certified_box_l0 as mod

def _make_corr_case(seed, n, p, ridge, l0_pen, box):
    rng = np.random.default_rng(seed)
    z_lat = rng.normal(size=(n, 4))
    X = np.zeros((n, p), dtype=float)
    for j in range(p):
        X[:, j] = 0.74 * z_lat[:, j % 4] + 0.67 * rng.normal(size=n)
    X[:, 4] = 0.53 * X[:, 0] + 0.51 * X[:, 2] + 0.13 * rng.normal(size=n)
    X /= np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-8) / np.sqrt(n)
    w_true = np.zeros(p, dtype=float)
    w_true[0] = 0.80 * box
    w_true[2] = 0.72 * box
    w_true[7] = -0.66 * box
    w_true[11] = 0.58 * box
    y = X @ w_true + rng.normal(0.0, 0.15, size=n)
    return {
        "X": X, "y": y, "ridge": float(ridge), "l0_penalty": float(l0_pen),
        "box": float(box), "gap_tol": 1e-6, "max_leaves": 512,
    }

def _eval_exact_obj(X, y, w, ridge, l0_pen):
    r = X @ w - y
    k0 = int(np.sum(np.abs(w) > 1e-10))
    return float(0.5 * float(r @ r) + 0.5 * ridge * float(w @ w) + l0_pen * k0)

class _ExactSupportBoxQPReference:
    def __init__(self, X, y, ridge, rho, M):
        self.X, self.y = X, y
        self.ridge, self.rho, self.M = float(ridge), float(rho), float(M)
        self.p = X.shape[1]
        self.Q = X.T @ X + self.ridge * np.eye(self.p)
        self.h = X.T @ y
        self.c = 0.5 * float(y @ y)
        self.masks = np.arange(1 << self.p, dtype=np.int64)
        self.lb = np.zeros(1 << self.p, dtype=float)
        self.ub = np.full(1 << self.p, float("inf"), dtype=float)
        self.refined = {0}
        self.roundoff_guard = 1e-9
        self.inactive_guard = 1e-10 * float(np.sum(np.abs(self.h) + self.M * np.sum(np.abs(self.Q), axis=1)))
        self.lb[0] = self.ub[0] = self.c
        for k in range(1, self.p + 1):
            combos = list(itertools.combinations(range(self.p), k))
            for off in range(0, len(combos), 256):
                ids = np.asarray(combos[off : off + 256], dtype=int)
                mats = self.Q[ids[:, :, None], ids[:, None, :]]
                rhs = self.h[ids]
                if self.ridge > 1e-12:
                    sol = np.linalg.solve(mats, rhs[..., None])[..., 0]
                    masks = np.sum(1 << ids, axis=1)
                    vals = self.c - 0.5 * np.sum(rhs * sol, axis=1) + self.rho * k
                    self.lb[masks] = vals - self.roundoff_guard
                    within = np.all(np.abs(sol) <= self.M, axis=1)
                    for row, mask in enumerate(masks):
                        if within[row]:
                            w = np.zeros(self.p, dtype=float)
                            w[ids[row]] = sol[row]
                            r = self.X @ w - self.y
                            self.ub[mask] = 0.5 * float(r @ r) + 0.5 * self.ridge * float(w @ w) + self.rho * k
                            self.refined.add(int(mask))
                else:
                    masks = np.sum(1 << ids, axis=1)
                    for row, mask in enumerate(masks):
                        sub_x = self.X[:, ids[row]]
                        sol_r, *_ = np.linalg.lstsq(sub_x, self.y, rcond=None)
                        r_un = sub_x @ sol_r - self.y
                        val_un = 0.5 * float(r_un @ r_un) + self.rho * k
                        self.lb[int(mask)] = val_un - self.roundoff_guard
                        if bool(np.all(np.abs(sol_r) <= self.M)):
                            self.ub[int(mask)] = val_un
                            self.refined.add(int(mask))

    def refine(self, mask):
        ids = np.flatnonzero((int(mask) >> np.arange(self.p)) & 1)
        if len(ids) == 0:
            self.refined.add(0)
            return
        a = np.vstack([self.X[:, ids], math.sqrt(self.ridge) * np.eye(len(ids))]) if self.ridge > 0.0 else self.X[:, ids]
        b = np.r_[self.y, np.zeros(len(ids))] if self.ridge > 0.0 else self.y
        fit = lsq_linear(a, b, bounds=(-self.M, self.M), method="bvls", tol=1e-13, max_iter=1000)
        v = np.clip(fit.x, -self.M, self.M)
        w = np.zeros(self.p, dtype=float); w[ids] = v
        r = self.X @ w - self.y
        smooth = 0.5 * float(r @ r) + 0.5 * self.ridge * float(w @ w)
        grad = self.Q[np.ix_(ids, ids)] @ v - self.h[ids]
        primal = smooth + self.rho * len(ids)
        dual = primal - float(grad @ v) - self.M * float(np.sum(np.abs(grad)))
        self.lb[int(mask)] = max(float(self.lb[int(mask)]), dual - self.roundoff_guard)
        self.ub[int(mask)] = min(float(self.ub[int(mask)]), primal)
        self.refined.add(int(mask))

    def interval(self, b0, b1):
        z = sum(1 << int(j) for j in b0)
        o = sum(1 << int(j) for j in b1)
        where = np.flatnonzero(((self.masks & z) == 0) & ((self.masks & o) == o))
        while True:
            winner = int(where[np.argmin(self.lb[where])])
            if not math.isfinite(float(np.min(self.ub[where]))):
                self.refine(winner)
            upper = float(np.min(self.ub[where]))
            if self.lb[winner] >= upper - 1e-8 or winner in self.refined:
                return float(self.lb[winner] - self.inactive_guard), upper
            self.refine(winner)

def _verify_partition_and_leaves(X, y, ridge, l0_pen, box, leaves, max_leaves, ref=None):
    p = X.shape[1]
    if ref is None:
        ref = _ExactSupportBoxQPReference(X, y, ridge, l0_pen, box)
    if not isinstance(leaves, list) or len(leaves) < 1 or len(leaves) > max_leaves:
        return False, "invalid_leaf_count", float("nan"), ref
    parsed = []
    tot_measure = 0
    min_leaf_lb = float("inf")
    for lf in leaves:
        if not isinstance(lf, dict):
            return False, "non_dict_leaf", float("nan"), ref
        raw_b0 = lf.get("fixed_zero")
        raw_b1 = lf.get("fixed_nonzero")
        raw_lb = lf.get("leaf_lower_bound")
        if not isinstance(raw_b0, (list, tuple)) or not isinstance(raw_b1, (list, tuple)):
            return False, "invalid_b0_b1_list", float("nan"), ref
        if isinstance(raw_lb, bool) or not isinstance(raw_lb, (int, float)):
            return False, "non_numeric_leaf_lb", float("nan"), ref
        lb_f = float(raw_lb)
        if not math.isfinite(lb_f):
            return False, "nonfinite_leaf_lb", float("nan"), ref
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in [*raw_b0, *raw_b1]):
            return False, "non_int_indices", float("nan"), ref
        b0 = frozenset(int(v) for v in raw_b0)
        b1 = frozenset(int(v) for v in raw_b1)
        if len(b0) != len(raw_b0) or len(b1) != len(raw_b1):
            return False, "duplicate_leaf_indices", float("nan"), ref
        if (b0 & b1) or any((idx < 0 or idx >= p) for idx in (b0 | b1)):
            return False, "overlapping_or_oob_indices", float("nan"), ref
        parsed.append((b0, b1, lb_f))
        tot_measure += 1 << (p - len(b0) - len(b1))
        min_leaf_lb = min(min_leaf_lb, lb_f)
    if tot_measure != (1 << p):
        return False, f"measure_mismatch({tot_measure}!={1<<p})", float("nan"), ref
    for i in range(len(parsed)):
        b0_i, b1_i, _ = parsed[i]
        for j in range(i + 1, len(parsed)):
            b0_j, b1_j, _ = parsed[j]
            if not ((b0_i & b1_j) or (b1_i & b0_j)):
                return False, f"overlapping_leaves({i},{j})", float("nan"), ref
    for idx_l, (b0_i, b1_i, lb_i) in enumerate(parsed):
        true_lb, true_ub = ref.interval(b0_i, b1_i)
        if lb_i > true_lb + 1e-6 or lb_i > true_ub + 1e-6:
            return False, f"unsound_leaf_lower_bound(leaf={idx_l},lb={lb_i:.6f}>true_lb={true_lb:.6f})", float("nan"), ref
    return True, "ok", min_leaf_lb, ref

t0 = time.monotonic()
priv_specs = [
    (99101, 46, 14, 0.06, 0.30, 1.2),
    (99102, 50, 14, 0.03, 0.26, 1.15),
    (99103, 52, 15, 0.07, 0.34, 1.25),
]
all_valid = True
all_solution_correct = True
all_partition_valid = True
all_leaf_bounds_valid = True
all_certificate_valid = True
max_gap = 0.0
max_leaf_count = 0
min_leaf_count = 10**9
status_notes = []

for seed_i, n_i, p_i, r_i, l0_i, box_i in priv_specs:
    prob = _make_corr_case(seed_i, n_i, p_i, r_i, l0_i, box_i)
    ref_i = _ExactSupportBoxQPReference(prob["X"], prob["y"], r_i, l0_i, box_i)
    indep_glob_lb, _ = ref_i.interval(frozenset(), frozenset())
    res = mod.solve_certified_box_l0({
        "X": prob["X"].copy(), "y": prob["y"].copy(),
        "ridge": prob["ridge"], "l0_penalty": prob["l0_penalty"],
        "box": prob["box"], "gap_tol": prob["gap_tol"], "max_leaves": prob["max_leaves"],
    })
    if not isinstance(res, dict):
        all_valid = all_solution_correct = all_partition_valid = all_leaf_bounds_valid = all_certificate_valid = False
        status_notes.append("non_dict_return"); break
    raw_w = res.get("weights")
    if raw_w is None or isinstance(raw_w, (bool, str, bytes, dict)) or np.asarray(raw_w).dtype == bool:
        all_valid = all_solution_correct = all_partition_valid = all_leaf_bounds_valid = all_certificate_valid = False
        status_notes.append("invalid_weights_type"); break
    w = np.asarray(raw_w, dtype=float)
    raw_obj = res.get("objective")
    raw_lb = res.get("certified_lower_bound")
    leaves = res.get("certificate_leaves")
    if (
        w.shape != (p_i,)
        or not bool(np.all(np.isfinite(w)))
        or isinstance(raw_obj, bool) or not isinstance(raw_obj, (int, float))
        or isinstance(raw_lb, bool) or not isinstance(raw_lb, (int, float))
    ):
        all_valid = all_solution_correct = all_partition_valid = all_leaf_bounds_valid = all_certificate_valid = False
        status_notes.append("invalid_weights_or_scalars"); break
    rep_obj = float(raw_obj)
    cert_lb = float(raw_lb)
    if not (math.isfinite(rep_obj) and math.isfinite(cert_lb)):
        all_valid = all_solution_correct = all_partition_valid = all_leaf_bounds_valid = all_certificate_valid = False
        status_notes.append("nonfinite_scalars"); break
    if float(np.max(np.abs(w))) > box_i + 1e-7:
        all_valid = all_solution_correct = False
        status_notes.append("box_violated"); break
    true_j = _eval_exact_obj(prob["X"], prob["y"], w, r_i, l0_i)
    if abs(rep_obj - true_j) > 1e-6:
        all_valid = all_solution_correct = False
        status_notes.append(f"obj_mismatch({rep_obj}!={true_j})"); break
    if true_j - indep_glob_lb > prob["gap_tol"] + 1e-6:
        all_valid = all_solution_correct = False
        status_notes.append(f"suboptimal_vs_exact_reference({true_j}>{indep_glob_lb})"); break
    part_ok, part_msg, min_leaf_lb, _ = _verify_partition_and_leaves(
        prob["X"], prob["y"], r_i, l0_i, box_i, leaves, prob["max_leaves"], ref=ref_i
    )
    if not part_ok:
        all_valid = all_certificate_valid = False
        if part_msg.startswith("unsound_leaf"):
            all_leaf_bounds_valid = False
        else:
            all_partition_valid = all_leaf_bounds_valid = False
        status_notes.append(part_msg); break
    if cert_lb > min_leaf_lb + 1e-8 or cert_lb > true_j + 1e-7:
        all_valid = all_certificate_valid = False
        status_notes.append(f"unsound_cert_lb({cert_lb}>{min(min_leaf_lb, true_j)})"); break
    gap = max(0.0, true_j - cert_lb)
    max_gap = max(max_gap, gap)
    max_leaf_count = max(max_leaf_count, len(leaves))
    min_leaf_count = min(min_leaf_count, len(leaves))
    if gap > prob["gap_tol"] + 1e-7:
        all_valid = False; status_notes.append(f"gap_exceeded({gap})"); break

elapsed = time.monotonic() - t0
ok = bool(
    all_valid
    and all_solution_correct
    and all_certificate_valid
    and math.isfinite(max_gap)
    and max_gap <= 1.1e-6
    and max_leaf_count >= 1
    and max_leaf_count <= 512
    and elapsed < 20.0
)
print(json.dumps({
    "strict_pass": ok,
    "all_valid": all_valid,
    "solution_correct": all_solution_correct,
    "partition_valid": all_partition_valid,
    "leaf_bounds_valid": all_leaf_bounds_valid,
    "certificate_valid": all_certificate_valid,
    "max_gap": round(float(max_gap), 9) if math.isfinite(max_gap) else -1.0,
    "min_leaf_count": int(min_leaf_count if min_leaf_count < 10**9 else 0),
    "max_leaf_count": int(max_leaf_count),
    "status_notes": status_notes,
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # HS02: Rotated Full-Tensor Darcy-Robin Source + Robust Offset Inversion
    # =========================================================================
    HoldoutOODTask(
        task_id="hs02__rotated_tensor_robin_darcy_inversion",
        short_name="hs02_rotated_tensor_robin_darcy",
        wave="m89_hard_stress",
        scientific_domain="Anisotropic Tensor Elliptic PDE & Outlier-Robust Inverse Source Identification",
        wormhole_family="fam_multiscale_pde_and_robust_assimilation",
        declared_artifacts=("/holdout_task/tensor_darcy_robin.py",),
        declared_callable_entrypoints=("infer_tensor_darcy_sources",),
        instruction_text=(
            "Implement `/holdout_task/tensor_darcy_robin.py` exposing:\n"
            "`def infer_tensor_darcy_sources(problem: dict) -> dict:`\n\n"
            "Physical & Numerical Discretization Specification:\n"
            "- Steady-state 2D rotated anisotropic tensor diffusion-reaction on `[0, Lx] x [0, Ly]` (`(nx, ny)` uniform cell-centered grid,\n"
            "  `hx = Lx / nx`, `hy = Ly / ny`, cell centers `x_i = (i + 0.5)*hx`, `y_j = (j + 0.5)*hy`):\n"
            "  `-div(K(x, y) grad u) + sigma(x, y) * u = sum_{m=0}^{M-1} s_m * basis[m, i, j]`\n"
            "  with symmetric positive-definite tensor `K = [[k_xx, k_xy], [k_xy, k_yy]]` (`k_xy != 0`!) and Robin boundary flux\n"
            "  `(-K grad u) . n = gamma_robin * (u - u_ambient)` on all four domain boundaries.\n"
            "- Exact cell-centered finite-volume stencil used by `problem` (verify against `/holdout_task/public_tensor_cases.pkl`):\n"
            "  1. Normal fluxes across interior x-faces `(i+1/2, j)` use harmonic `k_xx_f = 2*k_xx[i,j]*k_xx[i+1,j]/(k_xx[i,j]+k_xx[i+1,j])`,\n"
            "     contributing `(hy / hx) * k_xx_f * (u[i,j] - u[i+1,j])` (and symmetrically for y-faces with `k_yy` and `(hx / hy)`).\n"
            "  2. Exterior boundary faces (e.g. `i=0` left or `i=nx-1` right) use series Robin conductance\n"
            "     `g_x = hy * (gamma_robin * k_xx_b) / (gamma_robin * (0.5 * hx) + k_xx_b)` contributing `g_x * (u_b - u_ambient)`\n"
            "     (and symmetrically on `j=0, ny-1` with `hx`, `0.5 * hy`, `k_yy_b`).\n"
            "  3. Transverse off-diagonal flux `-d_x(k_xy d_y u) - d_y(k_xy d_x u)` at cell `(i, j)` uses clamped-boundary cell differences:\n"
            "     define `u_clamped(p, q) = u[clip(p, 0, nx-1), clip(q, 0, ny-1)]`; then for each interior x-face `(i+1/2, j)` with\n"
            "     `k_xy_fx = 0.5 * (k_xy[i, j] + k_xy[i+1, j])`, transverse flux from `(i, j)` to `(i+1, j)` is\n"
            "     `F_xy_x(i+1/2, j) = -k_xy_fx * 0.25 * ((u_clamped(i, j+1) - u_clamped(i, j-1)) + (u_clamped(i+1, j+1) - u_clamped(i+1, j-1)))`,\n"
            "     and for each interior y-face `(i, j+1/2)` with `k_xy_fy = 0.5 * (k_xy[i, j] + k_xy[i, j+1])`,\n"
            "     `F_xy_y(i, j+1/2) = -k_xy_fy * 0.25 * ((u_clamped(i+1, j) - u_clamped(i-1, j)) + (u_clamped(i+1, j+1) - u_clamped(i-1, j+1)))`;\n"
            "     exterior boundary faces have zero transverse tangential flux (`0`).\n"
            "  4. Cell reaction + source balance: `(normal + transverse face outflows) + (hx * hy) * sigma[i,j] * u[i,j] = (hx * hy) * q[i,j]`.\n"
            "- Observation model (`sensor_ij` shape `(K_s, 2)`, `y_obs` shape `(K_s,)`):\n"
            "  `y_obs[k] = u[i_k, j_k] + sensor_offset + noise_k`, where ~10% of sensors suffer heavy Cauchy outlier spikes (`+-1.5..4.0`)!\n"
            "- Given `problem` keys `('Lx', 'Ly', 'k_xx', 'k_xy', 'k_yy', 'sigma', 'gamma_robin', 'u_ambient', 'basis', 'sensor_ij', 'y_obs', 'tikhonov_alpha')`:\n"
            "  recover `source_weights` (`shape (M,)`), scalar `sensor_offset` (`float`), and full `predicted_field` (`shape (nx, ny)`).\n"
            "- Return `{'source_weights': 1D array (M,), 'sensor_offset': float, 'predicted_field': 2D array (nx, ny)}`.\n"
            "- Note: scalar zero-ghost Dirichlet operators do NOT match this rotated-tensor Robin PDE!\n"
            "- Public test cases (including ground-truth forward fields for stencil verification) are in `/holdout_task/public_tensor_cases.pkl`."
        ),
        public_setup_py=r'''
import pickle, numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def _assemble_tensor_robin(Lx, Ly, k_xx, k_xy, k_yy, sigma, gamma_r, u_amb):
    nx, ny = k_xx.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    A = lil_matrix((N, N), dtype=float)
    b_amb = np.zeros(N, dtype=float)
    def idx(i, j):
        return int(np.clip(i, 0, nx - 1)) * ny + int(np.clip(j, 0, ny - 1))
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            A[row, row] += hx * hy * float(sigma[i, j])
            # x-faces
            if i > 0:
                kf = 2.0 * k_xx[i, j] * k_xx[i - 1, j] / (k_xx[i, j] + k_xx[i - 1, j])
                c = (hy / hx) * kf
                A[row, row] += c; A[row, idx(i - 1, j)] -= c
                # transverse on (i - 1/2, j): outflow from (i,j) is -F_xy_x(i-1/2, j)
                kxy_f = 0.5 * (k_xy[i - 1, j] + k_xy[i, j])
                for p, sgn in ((i - 1, 1.0), (i, 1.0)):
                    A[row, idx(p, j + 1)] += 0.25 * kxy_f * sgn
                    A[row, idx(p, j - 1)] -= 0.25 * kxy_f * sgn
            else:
                g = hy * (gamma_r * k_xx[i, j]) / (gamma_r * 0.5 * hx + k_xx[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if i < nx - 1:
                kf = 2.0 * k_xx[i, j] * k_xx[i + 1, j] / (k_xx[i, j] + k_xx[i + 1, j])
                c = (hy / hx) * kf
                A[row, row] += c; A[row, idx(i + 1, j)] -= c
                kxy_f = 0.5 * (k_xy[i, j] + k_xy[i + 1, j])
                for p, sgn in ((i, -1.0), (i + 1, -1.0)):
                    A[row, idx(p, j + 1)] += 0.25 * kxy_f * sgn
                    A[row, idx(p, j - 1)] -= 0.25 * kxy_f * sgn
            else:
                g = hy * (gamma_r * k_xx[i, j]) / (gamma_r * 0.5 * hx + k_xx[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            # y-faces
            if j > 0:
                kf = 2.0 * k_yy[i, j] * k_yy[i, j - 1] / (k_yy[i, j] + k_yy[i, j - 1])
                c = (hx / hy) * kf
                A[row, row] += c; A[row, idx(i, j - 1)] -= c
                kxy_f = 0.5 * (k_xy[i, j - 1] + k_xy[i, j])
                for q, sgn in ((j - 1, 1.0), (j, 1.0)):
                    A[row, idx(i + 1, q)] += 0.25 * kxy_f * sgn
                    A[row, idx(i - 1, q)] -= 0.25 * kxy_f * sgn
            else:
                g = hx * (gamma_r * k_yy[i, j]) / (gamma_r * 0.5 * hy + k_yy[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if j < ny - 1:
                kf = 2.0 * k_yy[i, j] * k_yy[i, j + 1] / (k_yy[i, j] + k_yy[i, j + 1])
                c = (hx / hy) * kf
                A[row, row] += c; A[row, idx(i, j + 1)] -= c
                kxy_f = 0.5 * (k_xy[i, j] + k_xy[i, j + 1])
                for q, sgn in ((j, -1.0), (j + 1, -1.0)):
                    A[row, idx(i + 1, q)] += 0.25 * kxy_f * sgn
                    A[row, idx(i - 1, q)] -= 0.25 * kxy_f * sgn
            else:
                g = hx * (gamma_r * k_yy[i, j]) / (gamma_r * 0.5 * hy + k_yy[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
    return A.tocsc(), b_amb, hx * hy

def _make_case(seed, nx=22, ny=22):
    rng = np.random.default_rng(seed)
    Lx, Ly = 1.2, 1.0
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    theta = 0.55 + 0.35 * np.sin(np.pi * XX / Lx) * np.cos(np.pi * YY / Ly)
    l1 = 1.8 + 0.7 * np.cos(1.7 * XX)
    l2 = 0.35 + 0.15 * np.sin(1.9 * YY)
    c, s = np.cos(theta), np.sin(theta)
    k_xx = c * c * l1 + s * s * l2
    k_yy = s * s * l1 + c * c * l2
    k_xy = c * s * (l1 - l2)
    sigma = 0.25 + 0.15 * (XX / Lx) ** 2
    centers = [(0.26, 0.24), (0.94, 0.24), (0.26, 0.76), (0.94, 0.76), (0.60, 0.50)]
    basis = np.stack([np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * 0.11 ** 2)) for cx, cy in centers], axis=0)
    w_true = rng.uniform(0.8, 2.8, size=len(centers))
    gamma_r = 1.65
    u_amb = 0.40
    A, b_amb, vol = _assemble_tensor_robin(Lx, Ly, k_xx, k_xy, k_yy, sigma, gamma_r, u_amb)
    q = np.tensordot(w_true, basis, axes=(0, 0))
    u_true = spsolve(A, b_amb + vol * q.ravel()).reshape(nx, ny)
    all_ij = np.array([(i, j) for i in range(1, nx - 1, 2) for j in range(1, ny - 1, 2)], dtype=int)
    pick = rng.choice(len(all_ij), size=64, replace=False)
    sensor_ij = all_ij[pick]
    offset_true = float(rng.uniform(-1.4, 1.8))
    y_obs = u_true[sensor_ij[:, 0], sensor_ij[:, 1]] + offset_true + rng.normal(0.0, 0.0012, size=len(sensor_ij))
    out_idx = rng.choice(len(sensor_ij), size=5, replace=False)
    y_obs[out_idx] += rng.choice([-1.0, 1.0], size=5) * rng.uniform(1.8, 3.6, size=5)
    return {
        "problem": {
            "Lx": Lx, "Ly": Ly, "k_xx": k_xx, "k_xy": k_xy, "k_yy": k_yy, "sigma": sigma,
            "gamma_robin": gamma_r, "u_ambient": u_amb, "basis": basis,
            "sensor_ij": sensor_ij, "y_obs": y_obs, "tikhonov_alpha": 1e-8,
        },
        "gt": {"source_weights": w_true, "sensor_offset": offset_true, "u_true": u_true},
    }

with open("/holdout_task/public_tensor_cases.pkl", "wb") as f:
    pickle.dump({"cases": [_make_case(20261133), _make_case(20261134)]}, f)
''',
        private_verifier_py=r'''
import json, math, sys, time, numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
sys.path.insert(0, "/holdout_task")
import tensor_darcy_robin as mod

def _assemble_tensor_robin(Lx, Ly, k_xx, k_xy, k_yy, sigma, gamma_r, u_amb):
    nx, ny = k_xx.shape
    hx, hy = Lx / nx, Ly / ny
    N = nx * ny
    A = lil_matrix((N, N), dtype=float)
    b_amb = np.zeros(N, dtype=float)
    def idx(i, j):
        return int(np.clip(i, 0, nx - 1)) * ny + int(np.clip(j, 0, ny - 1))
    for i in range(nx):
        for j in range(ny):
            row = i * ny + j
            A[row, row] += hx * hy * float(sigma[i, j])
            if i > 0:
                kf = 2.0 * k_xx[i, j] * k_xx[i - 1, j] / (k_xx[i, j] + k_xx[i - 1, j])
                c = (hy / hx) * kf
                A[row, row] += c; A[row, idx(i - 1, j)] -= c
                kxy_f = 0.5 * (k_xy[i - 1, j] + k_xy[i, j])
                for p, sgn in ((i - 1, 1.0), (i, 1.0)):
                    A[row, idx(p, j + 1)] += 0.25 * kxy_f * sgn
                    A[row, idx(p, j - 1)] -= 0.25 * kxy_f * sgn
            else:
                g = hy * (gamma_r * k_xx[i, j]) / (gamma_r * 0.5 * hx + k_xx[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if i < nx - 1:
                kf = 2.0 * k_xx[i, j] * k_xx[i + 1, j] / (k_xx[i, j] + k_xx[i + 1, j])
                c = (hy / hx) * kf
                A[row, row] += c; A[row, idx(i + 1, j)] -= c
                kxy_f = 0.5 * (k_xy[i, j] + k_xy[i + 1, j])
                for p, sgn in ((i, -1.0), (i + 1, -1.0)):
                    A[row, idx(p, j + 1)] += 0.25 * kxy_f * sgn
                    A[row, idx(p, j - 1)] -= 0.25 * kxy_f * sgn
            else:
                g = hy * (gamma_r * k_xx[i, j]) / (gamma_r * 0.5 * hx + k_xx[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if j > 0:
                kf = 2.0 * k_yy[i, j] * k_yy[i, j - 1] / (k_yy[i, j] + k_yy[i, j - 1])
                c = (hx / hy) * kf
                A[row, row] += c; A[row, idx(i, j - 1)] -= c
                kxy_f = 0.5 * (k_xy[i, j - 1] + k_xy[i, j])
                for q, sgn in ((j - 1, 1.0), (j, 1.0)):
                    A[row, idx(i + 1, q)] += 0.25 * kxy_f * sgn
                    A[row, idx(i - 1, q)] -= 0.25 * kxy_f * sgn
            else:
                g = hx * (gamma_r * k_yy[i, j]) / (gamma_r * 0.5 * hy + k_yy[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
            if j < ny - 1:
                kf = 2.0 * k_yy[i, j] * k_yy[i, j + 1] / (k_yy[i, j] + k_yy[i, j + 1])
                c = (hx / hy) * kf
                A[row, row] += c; A[row, idx(i, j + 1)] -= c
                kxy_f = 0.5 * (k_xy[i, j] + k_xy[i, j + 1])
                for q, sgn in ((j, -1.0), (j + 1, -1.0)):
                    A[row, idx(i + 1, q)] += 0.25 * kxy_f * sgn
                    A[row, idx(i - 1, q)] -= 0.25 * kxy_f * sgn
            else:
                g = hx * (gamma_r * k_yy[i, j]) / (gamma_r * 0.5 * hy + k_yy[i, j])
                A[row, row] += g; b_amb[row] += g * u_amb
    return A.tocsc(), b_amb, hx * hy

t0 = time.monotonic()
valid = True
w_rel_errs = []
u_rel_errs = []
off_errs = []

for seed_i, (nx, ny) in ((99105, (24, 22)), (99106, (22, 26)), (99107, (24, 24))):
    rng = np.random.default_rng(seed_i)
    Lx, Ly = 1.3, 1.1
    xs = (np.arange(nx) + 0.5) * (Lx / nx)
    ys = (np.arange(ny) + 0.5) * (Ly / ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    theta = 0.62 + 0.32 * np.cos(np.pi * XX / Lx) * np.sin(np.pi * YY / Ly)
    l1 = 2.1 + 0.6 * np.sin(1.5 * XX)
    l2 = 0.32 + 0.14 * np.cos(1.8 * YY)
    c, s = np.cos(theta), np.sin(theta)
    k_xx = c * c * l1 + s * s * l2
    k_yy = s * s * l1 + c * c * l2
    k_xy = c * s * (l1 - l2)
    sigma = 0.22 + 0.18 * (YY / Ly) ** 2
    centers = [(0.26, 0.25), (1.04, 0.25), (0.26, 0.85), (1.04, 0.85), (0.65, 0.55)]
    basis = np.stack([np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * 0.11 ** 2)) for cx, cy in centers], axis=0)
    w_true = rng.uniform(0.8, 2.9, size=len(centers))
    gamma_r = 1.50 + float(rng.uniform(-0.25, 0.35))
    u_amb = float(rng.uniform(-0.3, 0.7))
    A, b_amb, vol = _assemble_tensor_robin(Lx, Ly, k_xx, k_xy, k_yy, sigma, gamma_r, u_amb)
    q = np.tensordot(w_true, basis, axes=(0, 0))
    u_true = spsolve(A, b_amb + vol * q.ravel()).reshape(nx, ny)
    all_ij = np.array([(i, j) for i in range(1, nx - 1, 2) for j in range(1, ny - 1, 2)], dtype=int)
    pick = rng.choice(len(all_ij), size=min(76, len(all_ij)), replace=False)
    sensor_ij = all_ij[pick]
    offset_true = float(rng.uniform(-2.0, 2.0))
    y_obs = u_true[sensor_ij[:, 0], sensor_ij[:, 1]] + offset_true + rng.normal(0.0, 0.0012, size=len(sensor_ij))
    out_idx = rng.choice(len(sensor_ij), size=6, replace=False)
    y_obs[out_idx] += rng.choice([-1.0, 1.0], size=6) * rng.uniform(2.0, 4.2, size=6)

    res = mod.infer_tensor_darcy_sources({
        "Lx": Lx, "Ly": Ly, "k_xx": k_xx.copy(), "k_xy": k_xy.copy(), "k_yy": k_yy.copy(),
        "sigma": sigma.copy(), "gamma_robin": gamma_r, "u_ambient": u_amb,
        "basis": basis.copy(), "sensor_ij": sensor_ij.copy(), "y_obs": y_obs.copy(), "tikhonov_alpha": 1e-8,
    })
    if not isinstance(res, dict):
        valid = False; break
    w_hat = np.asarray(res.get("source_weights", []), dtype=float)
    u_hat = np.asarray(res.get("predicted_field", []), dtype=float)
    raw_off = res.get("sensor_offset")
    if (
        w_hat.shape != w_true.shape
        or u_hat.shape != u_true.shape
        or not bool(np.all(np.isfinite(w_hat)))
        or not bool(np.all(np.isfinite(u_hat)))
        or isinstance(raw_off, bool) or not isinstance(raw_off, (int, float))
        or not math.isfinite(float(raw_off))
    ):
        valid = False; break
    w_rel_errs.append(float(np.linalg.norm(w_hat - w_true) / max(np.linalg.norm(w_true), 1e-9)))
    u_rel_errs.append(float(np.linalg.norm(u_hat - u_true) / max(np.linalg.norm(u_true), 1e-9)))
    off_errs.append(abs(float(raw_off) - offset_true))

elapsed = time.monotonic() - t0
max_w = max(w_rel_errs) if (valid and len(w_rel_errs) == 3) else float("nan")
max_u = max(u_rel_errs) if (valid and len(u_rel_errs) == 3) else float("nan")
max_o = max(off_errs) if (valid and len(off_errs) == 3) else float("nan")
ok = bool(
    valid
    and math.isfinite(max_w) and max_w <= 0.045
    and math.isfinite(max_u) and max_u <= 0.015
    and math.isfinite(max_o) and max_o <= 0.020
    and elapsed < 20.0
)
print(json.dumps({
    "strict_pass": ok,
    "all_valid": valid,
    "max_weight_rel_l2": round(float(max_w), 6) if math.isfinite(max_w) else -1.0,
    "max_field_rel_l2": round(float(max_u), 6) if math.isfinite(max_u) else -1.0,
    "max_offset_abs_err": round(float(max_o), 6) if math.isfinite(max_o) else -1.0,
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # HS03: Irregular-dt Hereditary Memory Calibration & Driver Deconvolution
    # =========================================================================
    HoldoutOODTask(
        task_id="hs03__irregular_dt_hereditary_memory_deconvolution",
        short_name="hs03_hereditary_memory_deconv",
        wave="m89_hard_stress",
        scientific_domain="Viscoelastic Hereditary Memory Identification & Irregular-dt Deconvolution",
        wormhole_family="fam_temporal_memory_and_stage_deconfounding",
        declared_artifacts=("/holdout_task/hereditary_deconv.py",),
        declared_callable_entrypoints=("deconvolve_hereditary_driver",),
        instruction_text=(
            "Implement `/holdout_task/hereditary_deconv.py` exposing:\n"
            "`def deconvolve_hereditary_driver(problem: dict) -> dict:`\n\n"
            "Exact Hereditary Prony Physics & Pre-Window Memory Specification:\n"
            "- A linear viscoelastic sensor with given relaxation time constants `taus = [tau_1, tau_2]` (`len=2`)\n"
            "  evolves on irregular strictly increasing timestamps via exact zero-order-hold (ZOH) exponential memory:\n"
            "  Given pre-window history `(past_timestamps, past_driver_history)` (`len >= 2`, `past_timestamps[-1] < timestamps[0]`)\n"
            "  starting from `m_r(past_timestamps[0]) = past_driver_history[0]`, step through `k = 1 .. len(past_timestamps)-1`:\n"
            "  `dt = past_timestamps[k] - past_timestamps[k-1]`, `a = exp(-dt / tau_r)`,\n"
            "  `m_r <- a * m_r + (1 - a) * past_driver_history[k]`;\n"
            "  then on main window `k = 0 .. T-1` (with `dt_0 = timestamps[0] - past_timestamps[-1]` and `dt_k = timestamps[k] - timestamps[k-1]` for `k >= 1`):\n"
            "  `a_k = exp(-dt_k / tau_r)`, `m_r[k] = a_k * m_r_prev + (1 - a_k) * u[k]`.\n"
            "- Clean response: `y_clean[k] = E_0 * u[k] + E_1 * m_1[k] + E_2 * m_2[k] + drift_slope * (timestamps[k] - timestamps[0]) + drift_offset`.\n"
            "- `problem` provides:\n"
            "  1. `'taus'`: `[tau_1, tau_2]`,\n"
            "  2. `'calibration_run'`: `{'past_timestamps', 'past_driver_history', 'timestamps', 'driver_u', 'y_obs'}` where ~8% of `y_obs` samples have heavy Cauchy telemetry glitches (`+-2.5..5.0`) -> robustly identify `(E_0, E_1, E_2)` (shared material moduli) + calibration drift!\n"
            "  3. `'blind_run'`: `{'past_timestamps', 'past_driver_history', 'timestamps', 'y_obs', 'drift_slope', 'drift_offset', 'smoothness_lambda'}` where `u[0..T-1]` is unknown (smooth + ramp/plateau), `drift_slope` & `drift_offset` are given calibrated instrument drift constants, `y_obs` is low-noise (`sigma ~ 0.008`, NO outliers in `blind_run`), and `u` minimizes\n"
            "     `0.5 * ||y_pred(u) - y_obs||_2^2 + 0.5 * smoothness_lambda * sum_{k=1}^{T-1} ((u[k] - u[k-1]) / sqrt(timestamps[k] - timestamps[k-1]))^2`!\n"
            "- Warning: ignoring `past_driver_history` sets `m_r(past_timestamps[-1]) = 0` and severely corrupts early `u[k]` and `(E_0, E_1, E_2)`!\n"
            "- Return `{'moduli': [E_0, E_1, E_2], 'recovered_driver': 1D array (T,), 'predicted_blind_response': 1D array (T,)}`.\n"
            "- Public test cases are in `/holdout_task/public_hereditary_cases.pkl`."
        ),
        public_setup_py=r'''
import math, pickle, numpy as np

def _rollout_memory(taus, past_t, past_u, t, u):
    T = len(t)
    M = np.zeros((len(taus), T), dtype=float)
    for r, tau in enumerate(taus):
        mr = float(past_u[0])
        for k in range(1, len(past_t)):
            a = math.exp(-(float(past_t[k] - past_t[k - 1])) / float(tau))
            mr = a * mr + (1.0 - a) * float(past_u[k])
        for k in range(T):
            dt = float(t[0] - past_t[-1]) if k == 0 else float(t[k] - t[k - 1])
            a = math.exp(-dt / float(tau))
            mr = a * mr + (1.0 - a) * float(u[k])
            M[r, k] = mr
    return M

def _make_case(seed):
    rng = np.random.default_rng(seed)
    taus = [0.65, 3.80]
    E_true = np.array([1.75, 0.95, 1.35], dtype=float)
    past_t_c = np.cumsum(np.r_[-6.0, rng.uniform(0.18, 0.42, size=18)])
    t_c = past_t_c[-1] + np.cumsum(rng.uniform(0.05, 0.35, size=85))
    past_u_c = 1.6 * np.sin(0.7 * past_t_c) + 1.2
    u_c = np.where(t_c < t_c[35], 1.8, -0.9) + 0.45 * np.cos(1.3 * t_c)
    Mc = _rollout_memory(taus, past_t_c, past_u_c, t_c, u_c)
    d_sl_c, d_of_c = 0.14, -0.55
    y_c = E_true[0] * u_c + E_true[1] * Mc[0] + E_true[2] * Mc[1] + d_sl_c * (t_c - t_c[0]) + d_of_c + rng.normal(0.0, 0.005, size=len(t_c))
    gl = rng.choice(len(t_c), size=7, replace=False)
    y_c[gl] += rng.choice([-1.0, 1.0], size=7) * rng.uniform(2.8, 4.8, size=7)

    past_t_b = np.cumsum(np.r_[-7.0, rng.uniform(0.15, 0.40, size=22)])
    t_b = past_t_b[-1] + np.cumsum(rng.uniform(0.06, 0.30, size=80))
    past_u_b = np.full_like(past_t_b, 2.4)  # strong non-zero pre-load!
    s_norm = (t_b - t_b[0]) / max(float(t_b[-1] - t_b[0]), 1e-9)
    u_b = 0.9 * np.sin(2.0 * math.pi * s_norm) + 0.65 * np.tanh(5.0 * (s_norm - 0.45))
    Mb = _rollout_memory(taus, past_t_b, past_u_b, t_b, u_b)
    d_sl_b, d_of_b = -0.09, 0.42
    y_b_clean = E_true[0] * u_b + E_true[1] * Mb[0] + E_true[2] * Mb[1] + d_sl_b * (t_b - t_b[0]) + d_of_b
    y_b = y_b_clean + rng.normal(0.0, 0.006, size=len(t_b))
    return {
        "problem": {
            "taus": taus,
            "calibration_run": {
                "past_timestamps": past_t_c, "past_driver_history": past_u_c,
                "timestamps": t_c, "driver_u": u_c, "y_obs": y_c,
            },
            "blind_run": {
                "past_timestamps": past_t_b, "past_driver_history": past_u_b,
                "timestamps": t_b, "y_obs": y_b,
                "drift_slope": d_sl_b, "drift_offset": d_of_b, "smoothness_lambda": 0.015,
            },
        },
        "gt": {"moduli": E_true, "u_blind": u_b, "y_blind_clean": y_b_clean},
    }

with open("/holdout_task/public_hereditary_cases.pkl", "wb") as f:
    pickle.dump({"cases": [_make_case(20261135), _make_case(20261136)]}, f)
''',
        private_verifier_py=r'''
import json, math, sys, time, numpy as np
sys.path.insert(0, "/holdout_task")
import hereditary_deconv as mod

def _rollout_memory(taus, past_t, past_u, t, u):
    T = len(t)
    M = np.zeros((len(taus), T), dtype=float)
    for r, tau in enumerate(taus):
        mr = float(past_u[0])
        for k in range(1, len(past_t)):
            a = math.exp(-(float(past_t[k] - past_t[k - 1])) / float(tau))
            mr = a * mr + (1.0 - a) * float(past_u[k])
        for k in range(T):
            dt = float(t[0] - past_t[-1]) if k == 0 else float(t[k] - t[k - 1])
            a = math.exp(-dt / float(tau))
            mr = a * mr + (1.0 - a) * float(u[k])
            M[r, k] = mr
    return M

t0 = time.monotonic()
valid = True
e_errs = []
u_rmses = []
y_rmses = []

for seed_i, taus, E_true in (
    (99108, [0.55, 3.40], np.array([1.90, 1.05, 1.45])),
    (99109, [0.80, 4.50], np.array([1.60, 0.85, 1.65])),
    (99110, [0.45, 2.90], np.array([2.15, 1.15, 1.10])),
):
    rng = np.random.default_rng(seed_i)
    past_t_c = np.cumsum(np.r_[-6.5, rng.uniform(0.16, 0.40, size=20)])
    t_c = past_t_c[-1] + np.cumsum(rng.uniform(0.05, 0.34, size=90))
    past_u_c = 1.8 * np.cos(0.6 * past_t_c) - 1.1
    u_c = np.where(t_c < t_c[40], -1.6, 1.4) + 0.55 * np.sin(1.4 * t_c)
    Mc = _rollout_memory(taus, past_t_c, past_u_c, t_c, u_c)
    d_sl_c, d_of_c = float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.8, 0.8))
    y_c = E_true[0] * u_c + E_true[1] * Mc[0] + E_true[2] * Mc[1] + d_sl_c * (t_c - t_c[0]) + d_of_c + rng.normal(0.0, 0.005, size=len(t_c))
    gl = rng.choice(len(t_c), size=8, replace=False)
    y_c[gl] += rng.choice([-1.0, 1.0], size=8) * rng.uniform(3.0, 5.2, size=8)

    past_t_b = np.cumsum(np.r_[-7.5, rng.uniform(0.15, 0.38, size=24)])
    t_b = past_t_b[-1] + np.cumsum(rng.uniform(0.05, 0.28, size=84))
    past_u_b = np.full_like(past_t_b, float(rng.choice([-2.6, 2.7])))
    s_norm = (t_b - t_b[0]) / max(float(t_b[-1] - t_b[0]), 1e-9)
    u_b = 0.85 * np.cos(2.0 * math.pi * s_norm) - 0.70 * np.tanh(5.0 * (s_norm - 0.52))
    Mb = _rollout_memory(taus, past_t_b, past_u_b, t_b, u_b)
    d_sl_b, d_of_b = float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.6, 0.6))
    y_b_clean = E_true[0] * u_b + E_true[1] * Mb[0] + E_true[2] * Mb[1] + d_sl_b * (t_b - t_b[0]) + d_of_b
    y_b = y_b_clean + rng.normal(0.0, 0.006, size=len(t_b))

    res = mod.deconvolve_hereditary_driver({
        "taus": list(taus),
        "calibration_run": {
            "past_timestamps": past_t_c.copy(), "past_driver_history": past_u_c.copy(),
            "timestamps": t_c.copy(), "driver_u": u_c.copy(), "y_obs": y_c.copy(),
        },
        "blind_run": {
            "past_timestamps": past_t_b.copy(), "past_driver_history": past_u_b.copy(),
            "timestamps": t_b.copy(), "y_obs": y_b.copy(),
            "drift_slope": d_sl_b, "drift_offset": d_of_b, "smoothness_lambda": 0.015,
        },
    })
    if not isinstance(res, dict):
        valid = False; break
    e_hat = np.asarray(res.get("moduli", []), dtype=float)
    u_hat = np.asarray(res.get("recovered_driver", []), dtype=float)
    y_hat = np.asarray(res.get("predicted_blind_response", []), dtype=float)
    if (
        e_hat.shape != (3,)
        or u_hat.shape != u_b.shape
        or y_hat.shape != y_b_clean.shape
        or not bool(np.all(np.isfinite(e_hat)))
        or not bool(np.all(np.isfinite(u_hat)))
        or not bool(np.all(np.isfinite(y_hat)))
    ):
        valid = False; break
    e_errs.append(float(np.linalg.norm(e_hat - E_true) / np.linalg.norm(E_true)))
    u_rmses.append(float(np.sqrt(np.mean((u_hat - u_b) ** 2))))
    y_rmses.append(float(np.sqrt(np.mean((y_hat - y_b_clean) ** 2))))

elapsed = time.monotonic() - t0
max_e = max(e_errs) if (valid and len(e_errs) == 3) else float("nan")
max_u = max(u_rmses) if (valid and len(u_rmses) == 3) else float("nan")
max_y = max(y_rmses) if (valid and len(y_rmses) == 3) else float("nan")
ok = bool(
    valid
    and math.isfinite(max_e) and max_e <= 0.025
    and math.isfinite(max_u) and max_u <= 0.055
    and math.isfinite(max_y) and max_y <= 0.020
    and elapsed < 15.0
)
print(json.dumps({
    "strict_pass": ok,
    "all_valid": valid,
    "max_moduli_rel_l2": round(float(max_e), 6) if math.isfinite(max_e) else -1.0,
    "max_driver_rmse": round(float(max_u), 6) if math.isfinite(max_u) else -1.0,
    "max_response_rmse": round(float(max_y), 6) if math.isfinite(max_y) else -1.0,
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),

    # =========================================================================
    # HS04: Missing-Reference-Panel Multi-Center Simpson-Confounded Calibration
    # =========================================================================
    HoldoutOODTask(
        task_id="hs04__missing_panel_multicenter_stratum_calibration",
        short_name="hs04_missing_panel_multicenter",
        wave="m89_hard_stress",
        scientific_domain="Biostatistics, Missing Reference Control Alignment & Simpson-Confounded Calibration",
        wormhole_family="fam_cohort_deconfounding_and_reference_balancing",
        declared_artifacts=("/holdout_task/multicenter_calibration.py",),
        declared_callable_entrypoints=("calibrate_multicenter_assay",),
        instruction_text=(
            "Implement `/holdout_task/multicenter_calibration.py` exposing:\n"
            "`def calibrate_multicenter_assay(dataset: dict) -> dict:`\n\n"
            "Assay Measurement & Confounding Specification:\n"
            "- A multi-center clinical assay measures `D = 8` continuous biomarkers across `C = 6` hospital centers.\n"
            "- Raw measured value in center `c` for biomarker `d` is `X_raw[i, d] = center_shift[c, d] + center_scale[c, d] * Z_true[i, d]`,\n"
            "  where healthy pooled reference controls (`is_reference_control == True`) are drawn from `Z_control ~ N(0, I_D)` across ALL centers\n"
            "  (so latent standardized coordinates have reference control median `~ 0` and robust MAD scale `~ 1` in every center!).\n"
            "- CRITICAL DATA EDGE CASE (`Missing / Sparse Reference Controls`): ~15% of entries in `reference_controls['X']` are `NaN` (missing),\n"
            "  AND small query centers may have `< 2` finite reference control observations in 1–2 biomarker columns!\n"
            "  (If a `(center, col)` has `< 2` finite controls or zero spread, shrink/fallback gracefully toward that center's median shift/scale\n"
            "  across its well-observed columns or pooled reference statistics; also impute any `NaN` in patient `X` after alignment as `0.0`.)\n"
            "- SIMPSON'S PARADOX & STRATUM IMBALANCE:\n"
            "  - Training centers `0..3` have strong opposite center shifts (`+3.5` vs `-3.5`) correlated with positive case prevalence (`82%` vs `18%`),\n"
            "    so unaligned pooled models learn spurious center shifts and collapse on held-out query centers `4..5`!\n"
            "  - Clinical `stratum` (`0, 1, 2`) is also imbalanced; true log-odds follow `beta_0 + stratum_offset[s] + Z_true @ beta`.\n"
            "- `dataset` contains:\n"
            "  - `'reference_controls'`: `{'center_id': (N_ref,), 'X': (N_ref, D)}` (contains `NaN`s & sparse center-column counts!),\n"
            "  - `'train'`: `{'center_id': (N_tr,), 'stratum': (N_tr,), 'X': (N_tr, D), 'y': (N_tr,)}` (`y in {0, 1}`),\n"
            "  - `'query'`: `{'center_id': (N_q,), 'stratum': (N_q,), 'X': (N_q, D)}`.\n"
            "- Return:\n"
            "  - `'aligned_query_features'`: 2D float array `(N_q, D)` estimating latent `Z_true` on `query` (`MAE <= 0.24`),\n"
            "  - `'query_probabilities'`: 1D float array `(N_q,)` in `(0, 1)` (`within-center AUROC >= 0.88`, `Brier <= 0.160`),\n"
            "  - `'feature_weights'`: 1D float array `(D,)` of fitted biomarker log-odds weights.\n"
            "- Public development cohorts are in `/holdout_task/public_multicenter_assay.pkl`."
        ),
        public_setup_py=r'''
import pickle, numpy as np

def _make_cohort(seed):
    rng = np.random.default_rng(seed)
    D = 8
    C = 6
    # Center shifts intentionally correlated with train prevalence (Simpson's paradox!)
    base_c_shift = np.array([3.4, 3.2, -3.3, -3.2, -2.8, 2.9], dtype=float)[:, None]
    c_shift = base_c_shift + rng.normal(0.0, 0.06, size=(C, D))
    c_scale = np.array([1.35, 0.80, 1.25, 0.85, 1.45, 0.75], dtype=float)[:, None] * rng.uniform(0.97, 1.03, size=(C, D))
    beta = np.array([1.25, -1.05, 0.90, -0.80, 0.65, -0.60, 0.55, 0.55], dtype=float)
    strat_off = np.array([-0.80, 0.05, 0.85], dtype=float)

    # Reference controls
    ref_counts = [80, 75, 80, 75, 70, 52]
    ref_cid = np.concatenate([np.full(n, c, dtype=int) for c, n in enumerate(ref_counts)])
    Z_ref = rng.normal(0.0, 1.0, size=(len(ref_cid), D))
    X_ref = c_shift[ref_cid] + c_scale[ref_cid] * Z_ref
    miss_ref = rng.uniform(size=X_ref.shape) < 0.14
    X_ref[miss_ref] = np.nan
    # Sparse control edge case: center 5 col 7 has ONLY 1 finite control!
    c5_rows = np.flatnonzero(ref_cid == 5)
    X_ref[c5_rows[1:], 7] = np.nan
    X_ref[c5_rows[0], 7] = c_shift[5, 7] + c_scale[5, 7] * 0.1

    def _sample_center(c, n, target_pos_bias):
        s = rng.choice([0, 1, 2], size=n, p=[0.55, 0.30, 0.15])
        Z = rng.normal(target_pos_bias, 1.0, size=(n, D))
        logits = -0.1 + strat_off[s] + Z @ beta
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(int)
        X = c_shift[c] + c_scale[c] * Z
        X[rng.uniform(size=X.shape) < 0.03] = np.nan
        return np.full(n, c, dtype=int), s, Z, X, y, probs

    tr_parts = [_sample_center(0, 95, 0.45), _sample_center(1, 95, 0.42), _sample_center(2, 95, -0.45), _sample_center(3, 95, -0.42)]
    q_parts = [_sample_center(4, 90, 0.40), _sample_center(5, 90, -0.38)]
    tr_cid = np.concatenate([p[0] for p in tr_parts])
    tr_s = np.concatenate([p[1] for p in tr_parts])
    tr_Z = np.vstack([p[2] for p in tr_parts])
    tr_X = np.vstack([p[3] for p in tr_parts])
    tr_y = np.concatenate([p[4] for p in tr_parts])

    q_cid = np.concatenate([p[0] for p in q_parts])
    q_s = np.concatenate([p[1] for p in q_parts])
    q_Z = np.vstack([p[2] for p in q_parts])
    q_X = np.vstack([p[3] for p in q_parts])
    q_y = np.concatenate([p[4] for p in q_parts])
    q_p = np.concatenate([p[5] for p in q_parts])
    return {
        "dataset": {
            "reference_controls": {"center_id": ref_cid, "X": X_ref},
            "train": {"center_id": tr_cid, "stratum": tr_s, "X": tr_X, "y": tr_y},
            "query": {"center_id": q_cid, "stratum": q_s, "X": q_X},
        },
        "gt": {"Z_query": q_Z, "y_query": q_y, "prob_query": q_p, "beta": beta},
    }

with open("/holdout_task/public_multicenter_assay.pkl", "wb") as f:
    pickle.dump({"cohorts": [_make_cohort(20261137), _make_cohort(20261138)]}, f)
''',
        private_verifier_py=r'''
import json, math, sys, time, numpy as np
sys.path.insert(0, "/holdout_task")
import multicenter_calibration as mod

def _auroc(y_true, scores):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)))

def _make_priv_cohort(seed):
    rng = np.random.default_rng(seed)
    D, C = 8, 6
    base_c_shift = np.array([3.5, 3.3, -3.4, -3.3, -2.9, 3.0], dtype=float)[:, None]
    c_shift = base_c_shift + rng.normal(0.0, 0.04, size=(C, D))
    c_scale = np.array([1.40, 0.78, 1.30, 0.82, 1.50, 0.72], dtype=float)[:, None] * rng.uniform(0.98, 1.02, size=(C, D))
    beta = np.array([1.75, -1.50, 1.25, -1.15, 0.95, -0.80, 0.80, 0.85], dtype=float)
    strat_off = np.array([-0.85, 0.0, 0.90], dtype=float)

    ref_counts = [110, 105, 110, 105, 95, 75]
    ref_cid = np.concatenate([np.full(n, c, dtype=int) for c, n in enumerate(ref_counts)])
    Z_ref = rng.normal(0.0, 1.0, size=(len(ref_cid), D))
    X_ref = c_shift[ref_cid] + c_scale[ref_cid] * Z_ref
    X_ref[rng.uniform(size=X_ref.shape) < 0.14] = np.nan
    c5_rows = np.flatnonzero(ref_cid == 5)
    X_ref[c5_rows[1:], 7] = np.nan
    X_ref[c5_rows[0], 7] = c_shift[5, 7]
    c4_rows = np.flatnonzero(ref_cid == 4)
    X_ref[c4_rows[:], 6] = np.nan  # 0 finite controls in (center 4, col 6)!

    def _sample_center(c, n, bias):
        s = rng.choice([0, 1, 2], size=n, p=[0.56, 0.29, 0.15])
        Z = rng.normal(bias, 1.0, size=(n, D))
        logits = -0.05 + strat_off[s] + Z @ beta
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < probs).astype(int)
        X = c_shift[c] + c_scale[c] * Z
        X[rng.uniform(size=X.shape) < 0.03] = np.nan
        return np.full(n, c, dtype=int), s, Z, X, y

    tr_parts = [_sample_center(0, 130, 0.46), _sample_center(1, 130, 0.44), _sample_center(2, 130, -0.46), _sample_center(3, 130, -0.44)]
    q_parts = [_sample_center(4, 160, 0.42), _sample_center(5, 160, -0.40)]
    return (
        {
            "reference_controls": {"center_id": ref_cid, "X": X_ref},
            "train": {
                "center_id": np.concatenate([p[0] for p in tr_parts]),
                "stratum": np.concatenate([p[1] for p in tr_parts]),
                "X": np.vstack([p[3] for p in tr_parts]),
                "y": np.concatenate([p[4] for p in tr_parts]),
            },
            "query": {
                "center_id": np.concatenate([p[0] for p in q_parts]),
                "stratum": np.concatenate([p[1] for p in q_parts]),
                "X": np.vstack([p[3] for p in q_parts]),
            },
        },
        np.vstack([p[2] for p in q_parts]),
        np.concatenate([p[4] for p in q_parts]),
        np.concatenate([p[0] for p in q_parts]),
    )

t0 = time.monotonic()
valid = True
z_maes = []
min_center_aurocs = []
briers = []

for seed_i in (99111, 99112, 99113):
    ds, Z_q, y_q, cid_q = _make_priv_cohort(seed_i)
    res = mod.calibrate_multicenter_assay(ds)
    if not isinstance(res, dict):
        valid = False; break
    Z_hat = np.asarray(res.get("aligned_query_features", []), dtype=float)
    p_hat = np.asarray(res.get("query_probabilities", []), dtype=float)
    w_hat = np.asarray(res.get("feature_weights", []), dtype=float)
    if (
        Z_hat.shape != Z_q.shape
        or p_hat.shape != y_q.shape
        or w_hat.shape != (8,)
        or not bool(np.all(np.isfinite(Z_hat)))
        or not bool(np.all(np.isfinite(p_hat)))
        or not bool(np.all(np.isfinite(w_hat)))
        or bool(np.any((p_hat < 0.0) | (p_hat > 1.0)))
    ):
        valid = False; break
    # Ignore non-missable observed entries when scoring Z MAE so imputed 3% patient NaNs don't dominate
    obs_mask = np.isfinite(ds["query"]["X"])
    z_maes.append(float(np.mean(np.abs(Z_hat[obs_mask] - Z_q[obs_mask]))))
    c_aucs = [_auroc(y_q[cid_q == c], p_hat[cid_q == c]) for c in (4, 5)]
    min_center_aurocs.append(min(c_aucs))
    briers.append(float(np.mean((p_hat - y_q) ** 2)))

elapsed = time.monotonic() - t0
max_z_mae = max(z_maes) if (valid and len(z_maes) == 3) else float("nan")
worst_auc = min(min_center_aurocs) if (valid and len(min_center_aurocs) == 3) else float("nan")
max_brier = max(briers) if (valid and len(briers) == 3) else float("nan")
ok = bool(
    valid
    and math.isfinite(max_z_mae) and max_z_mae <= 0.24
    and math.isfinite(worst_auc) and worst_auc >= 0.88
    and math.isfinite(max_brier) and max_brier <= 0.160
    and elapsed < 15.0
)
print(json.dumps({
    "strict_pass": ok,
    "all_valid": valid,
    "max_aligned_z_mae": round(float(max_z_mae), 5) if math.isfinite(max_z_mae) else -1.0,
    "worst_within_center_auroc": round(float(worst_auc), 5) if math.isfinite(worst_auc) else -1.0,
    "max_brier_score": round(float(max_brier), 5) if math.isfinite(max_brier) else -1.0,
    "elapsed_sec": round(float(elapsed), 4),
}, allow_nan=False))
''',
    ),
)


HARD_STRESS_M89_BY_ID: dict[str, HoldoutOODTask] = {t.task_id: t for t in HARD_STRESS_M89_TASKS}


def hard_stress_task_to_frontier_task(ht: HoldoutOODTask) -> m6f.FrontierBenchmarkTask:
    ensure_clean_scipy_sandbox_image()
    return m6f.FrontierBenchmarkTask(
        task_id=ht.task_id,
        short_name=ht.short_name,
        benchmark="Hegel-M89-HardStress-4",
        wave=ht.wave,
        scientific_domain=ht.scientific_domain,
        wormhole_family=ht.wormhole_family,
        instruction_text=ht.instruction_text,
        agent_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        verifier_image_tag=CLEAN_SCIPY_SANDBOX_IMAGE,
        declared_artifacts=ht.declared_artifacts,
        holdout_public_setup_py=ht.public_setup_py,
        agent_cpus=2,
        agent_memory_mb=2048,
        verifier_cpus=2,
        verifier_memory_mb=2048,
        verifier_timeout_sec=120,
        verifier_network_mode="none",
    )


def get_hard_stress_m89_frontier_tasks() -> list[m6f.FrontierBenchmarkTask]:
    return [hard_stress_task_to_frontier_task(t) for t in HARD_STRESS_M89_TASKS]


def run_hard_stress_private_verifier_by_id(task_id: str, submitted_tar_path: Path | None) -> dict[str, Any]:
    ht = HARD_STRESS_M89_BY_ID[task_id]
    rep = run_holdout_private_verifier(ht, submitted_tar_path)
    rep["benchmark"] = "Hegel-M89-HardStress-4"
    return rep


def run_hard_stress_private_verifier_for_container(task_id: str, agent_cid: str) -> dict[str, Any]:
    ht = HARD_STRESS_M89_BY_ID[task_id]
    with tempfile.TemporaryDirectory(prefix="hegel_hs_m89_ver_") as td:
        tar_p = Path(td) / "submitted.tar"
        m6f.export_declared_artifacts_tar(agent_cid, ht.declared_artifacts, tar_p)
        return run_hard_stress_private_verifier_by_id(task_id, tar_p if tar_p.is_file() else None)

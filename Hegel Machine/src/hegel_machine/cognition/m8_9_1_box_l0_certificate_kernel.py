"""Hegel M8.9.1 Sound Box-L0 Subcube Partition & Proof-Carrying Certificate Kernel (`R1`).

Separates four orthogonal mathematical properties for Box-Constrained `L0 + Ridge` optimization
`min_{w in [-M, M]^p} 0.5*||X w - y||_2^2 + 0.5*ridge*||w||_2^2 + l0_penalty * count_nonzero(|w| > active_tol)`:

1. `solution_correct`: Candidate `weights` is feasible, finite, matches reported `objective`, and achieves global
   optimality within `gap_tol` (when exact support reference is available or certified by a valid dual proof).
2. `partition_valid`: `certificate_leaves` (`1 <= len <= max_leaves` — valid 1-leaf root certificates ARE accepted!)
   forms a pairwise-disjoint, measure-complete (`sum 2^{p - |B0| - |B1|} == 2^p`) partition of `{0, 1}^p`.
3. `leaf_bounds_valid`: Every leaf's `leaf_lower_bound` is mathematically sound:
   - Mode A (`dual_nu` Fenchel Dual Witness, arbitrary `p`): Checker independently evaluates
     `L_dual(nu; B0, B1) = nu^T y - 0.5*||nu||_2^2 + rho*|B1| - sum_{j in B1} h(c_j) - sum_{j in F} max(0, h(c_j) - rho) - thresh_guard`
     with `c = X^T nu` and `h(c_j) = max_{|w| <= M} (c_j*w - 0.5*ridge*w^2)` (handling both `ridge > 0` and `ridge == 0`),
     requiring `claimed_leaf_lb <= L_dual + sound_tol`.
   - Mode B (`minorant_diagonal` + `minorant_center` PSD Minorant Witness, arbitrary `p`): Checker independently verifies
     `X^T X - diag(d) >= -1e-11 * I` (with conservative symmetric-box slack compensation) and evaluates the exact separable
     coordinate-wise lower bound on `(B0, B1)` minus `thresh_guard`, requiring `claimed_leaf_lb <= L_minorant + sound_tol`.
   - Mode C (Exact All-Support + Convex-QP Tangent Reference, `p <= max_enum_p`, default `16`): Checker enumerates all `2^p`
     supports using vectorized ridge + BVLS convex tangent dual bounds (`primal - grad^T v - M*||grad||_1 - thresh_guard`),
     requiring `claimed_leaf_lb <= true_subcube_lb + sound_tol` AND `claimed_leaf_lb <= true_subcube_ub + sound_tol`.
   - Any unproved scalar `leaf_lower_bound` on `p > max_enum_p` fails closed!
4. `gap_satisfied` & `certificate_valid`: `certified_lower_bound <= min_leaf_lb + 1e-8` AND
   `0 <= true_objective - certified_lower_bound <= gap_tol + 1e-7`.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear


def eval_exact_box_l0_objective(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    ridge: float,
    l0_penalty: float,
    *,
    active_tol: float = 1e-10,
) -> float:
    r = X @ w - y
    k0 = int(np.sum(np.abs(w) > active_tol))
    return float(0.5 * float(r @ r) + 0.5 * float(ridge) * float(w @ w) + float(l0_penalty) * k0)


def compute_inactive_threshold_lipschitz_guard(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float,
    box: float,
    *,
    active_tol: float = 1e-10,
    b0_set: frozenset[int] = frozenset(),
    b1_set: frozenset[int] = frozenset(),
) -> float:
    """Upper-bound smooth objective reduction `f(w_0) - f(w)` from coordinates `I = {j : |w_j| <= active_tol}` treated as zero in L0 count.

    For `w` in subcube `(B0, B1)` with `w_0` zeroing `I` (`B0 <= I <= {0..p-1} \\ B1`), PSD convexity of `Q = X^T X + ridge*I` gives
    `f(w) - f(w_0) >= grad f(w_0)^T (w - w_0)`. Since `w_{0,k} = 0` for all `k in I` (so `k != j` and `k not in B0`),
    `|grad_j f(w_0)| <= |h_j| + box * sum_{k != j, k not in B0} |Q_{jk}|` for every `j in I` (`j not in B1`).
    """
    p = X.shape[1]
    if p == 0 or active_tol <= 0.0:
        return 0.0
    Q_abs = np.abs(X.T @ X + float(ridge) * np.eye(p))
    np.fill_diagonal(Q_abs, 0.0)
    if b0_set:
        z_idx = [j for j in b0_set if 0 <= j < p]
        if z_idx:
            Q_abs[:, z_idx] = 0.0
    h_abs = np.abs(X.T @ y)
    per_coord = h_abs + float(box) * np.sum(Q_abs, axis=1)
    if b1_set:
        a_idx = [j for j in b1_set if 0 <= j < p]
        if a_idx:
            per_coord[a_idx] = 0.0
    return float(active_tol * float(np.sum(per_coord)))


def evaluate_fenchel_dual_leaf_lower_bound(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float,
    l0_penalty: float,
    box: float,
    b0_set: frozenset[int],
    b1_set: frozenset[int],
    dual_nu: np.ndarray,
    *,
    active_tol: float = 1e-10,
    roundoff_guard: float = 1e-9,
) -> tuple[bool, str, float]:
    """Independently compute weak-duality Fenchel conjugate lower bound `L(nu; B0, B1)` for arbitrary dimension `p`."""
    n, p = X.shape
    nu = np.asarray(dual_nu, dtype=float)
    if nu.shape != (n,) or not bool(np.all(np.isfinite(nu))):
        return False, "invalid_dual_nu_shape_or_nonfinite", float("nan")
    lam = float(ridge)
    rho = float(l0_penalty)
    M = float(box)
    if lam < 0.0 or rho < 0.0 or M <= 0.0:
        return False, "invalid_problem_hyperparameters", float("nan")

    c = X.T @ nu
    if not bool(np.all(np.isfinite(c))):
        return False, "nonfinite_dual_correlations", float("nan")

    if lam > 1e-15:
        w_star = np.clip(c / lam, -M, M)
        h_vec = c * w_star - 0.5 * lam * (w_star * w_star)
    else:
        h_vec = M * np.abs(c)

    pen_sum = 0.0
    for j in range(p):
        if j in b0_set:
            continue
        hj = float(h_vec[j])
        if j in b1_set:
            pen_sum += hj
        else:
            if hj > rho:
                pen_sum += hj - rho

    thresh_guard = compute_inactive_threshold_lipschitz_guard(
        X, y, lam, M, active_tol=active_tol, b0_set=b0_set, b1_set=b1_set
    )
    lb = (
        float(nu @ y)
        - 0.5 * float(nu @ nu)
        + rho * len(b1_set)
        - pen_sum
        - thresh_guard
        - roundoff_guard
    )
    return True, "ok_fenchel_dual", float(lb)


def evaluate_psd_minorant_leaf_lower_bound(
    X: np.ndarray,
    y: np.ndarray,
    ridge: float,
    l0_penalty: float,
    box: float,
    b0_set: frozenset[int],
    b1_set: frozenset[int],
    diagonal: np.ndarray,
    center: np.ndarray,
    *,
    active_tol: float = 1e-10,
    roundoff_guard: float = 1e-9,
) -> tuple[bool, str, float]:
    """Independently verify `X^T X - diag(d) >= 0` (with symmetric-box slack compensation) and evaluate subcube lower bound."""
    p = X.shape[1]
    d = np.asarray(diagonal, dtype=float)
    c0 = np.asarray(center, dtype=float)
    if d.shape != (p,) or c0.shape != (p,) or not (bool(np.all(np.isfinite(d))) and bool(np.all(np.isfinite(c0)))):
        return False, "invalid_minorant_diagonal_or_center", float("nan")
    if float(np.min(d)) < -1e-12:
        return False, "negative_minorant_diagonal", float("nan")
    d = np.maximum(d, 0.0)
    lam = float(ridge)
    rho = float(l0_penalty)
    M = float(box)
    if M <= 0.0 or lam < 0.0 or rho < 0.0:
        return False, "invalid_problem_hyperparameters", float("nan")

    G = X.T @ X
    scale_eig = max(1.0, float(np.max(np.diag(G))) if G.size else 1.0, float(np.max(d)) if d.size else 1.0)
    min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (G + G.T) - np.diag(d))))
    if min_eig < -1e-9 * scale_eig:
        return False, f"minorant_not_psd(min_eig={min_eig:.3e})", float("nan")
    psd_slack = (0.5 * (abs(min_eig) + 2e-15 * scale_eig) * float(np.sum((M + np.abs(c0)) ** 2))) if min_eig < 0.0 else 0.0

    r0 = y - X @ c0
    g0 = X.T @ r0
    const_term = 0.5 * float(r0 @ r0)
    coord_sum = 0.0
    for j in range(p):
        cj = float(c0[j])
        dj = float(d[j])
        gj = float(g0[j])
        # Coordinate quadratic at w_j: 0.5*dj*(w_j - cj)^2 - gj*(w_j - cj) + 0.5*lam*w_j^2
        # = 0.5*(dj + lam)*w_j^2 - (dj*cj + gj)*w_j + 0.5*dj*cj^2 + gj*cj
        q_const = 0.5 * dj * cj * cj + gj * cj
        val_zero = q_const
        curv = dj + lam
        lin = dj * cj + gj
        if curv > 1e-15:
            w_opt = float(np.clip(lin / curv, -M, M))
        else:
            w_opt = M if lin >= 0.0 else -M
        val_nz = 0.5 * curv * w_opt * w_opt - lin * w_opt + q_const + rho
        if j in b0_set:
            coord_sum += val_zero
        elif j in b1_set:
            coord_sum += val_nz
        else:
            coord_sum += min(val_zero, val_nz)

    thresh_guard = compute_inactive_threshold_lipschitz_guard(
        X, y, lam, M, active_tol=active_tol, b0_set=b0_set, b1_set=b1_set
    )
    lb = const_term + coord_sum - psd_slack - thresh_guard - roundoff_guard
    return True, "ok_psd_minorant", float(lb)


class ExactAllSupportBoxQPReference:
    """Exact `2^p` support interval reference using vectorized ridge + BVLS convex tangent lower bounds (`p <= 16`)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ridge: float,
        l0_penalty: float,
        box: float,
        *,
        active_tol: float = 1e-10,
        roundoff_guard: float = 1e-9,
    ) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.ridge = float(ridge)
        self.rho = float(l0_penalty)
        self.M = float(box)
        self.p = self.X.shape[1]
        self.Q = self.X.T @ self.X + self.ridge * np.eye(self.p)
        self.h = self.X.T @ self.y
        self.c = 0.5 * float(self.y @ self.y)
        self.masks = np.arange(1 << self.p, dtype=np.int64)
        self.lb = np.zeros(1 << self.p, dtype=float)
        self.ub = np.full(1 << self.p, float("inf"), dtype=float)
        self.refined: set[int] = {0}
        self.roundoff_guard = float(roundoff_guard)
        self.inactive_threshold_guard = compute_inactive_threshold_lipschitz_guard(
            self.X, self.y, self.ridge, self.M, active_tol=active_tol
        )
        self.lb[0] = self.ub[0] = self.c
        self.refinement_count = 0

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
                    # Possibly singular or ridge == 0: use lstsq and defer tight box bound to BVLS `refine` when needed
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

    def refine(self, mask: int) -> None:
        ids = np.flatnonzero((int(mask) >> np.arange(self.p)) & 1)
        if len(ids) == 0:
            self.refined.add(0)
            return
        if self.ridge > 0.0:
            a = np.vstack([self.X[:, ids], math.sqrt(self.ridge) * np.eye(len(ids))])
            b = np.r_[self.y, np.zeros(len(ids))]
        else:
            a = self.X[:, ids]
            b = self.y
        fit = lsq_linear(a, b, bounds=(-self.M, self.M), method="bvls", tol=1e-13, max_iter=1000)
        v = np.clip(fit.x, -self.M, self.M)
        w = np.zeros(self.p, dtype=float)
        w[ids] = v
        r = self.X @ w - self.y
        smooth = 0.5 * float(r @ r) + 0.5 * self.ridge * float(w @ w)
        grad = self.Q[np.ix_(ids, ids)] @ v - self.h[ids]
        primal = smooth + self.rho * len(ids)
        dual = primal - float(grad @ v) - self.M * float(np.sum(np.abs(grad)))
        self.lb[int(mask)] = max(float(self.lb[int(mask)]), dual - self.roundoff_guard)
        self.ub[int(mask)] = min(float(self.ub[int(mask)]), primal)
        self.refined.add(int(mask))
        self.refinement_count += 1

    def mask_filter(self, b0: frozenset[int], b1: frozenset[int]) -> np.ndarray:
        z = sum(1 << int(j) for j in b0)
        o = sum(1 << int(j) for j in b1)
        return ((self.masks & z) == 0) & ((self.masks & o) == o)

    def interval(self, b0: frozenset[int] = frozenset(), b1: frozenset[int] = frozenset()) -> tuple[float, float]:
        where = np.flatnonzero(self.mask_filter(b0, b1))
        while True:
            winner = int(where[np.argmin(self.lb[where])])
            # Ensure at least one upper bound is finite
            if not math.isfinite(float(np.min(self.ub[where]))):
                self.refine(winner)
            upper = float(np.min(self.ub[where]))
            if self.lb[winner] >= upper - 1e-8 or winner in self.refined:
                return float(self.lb[winner] - self.inactive_threshold_guard), upper
            self.refine(winner)


def verify_box_l0_subcube_certificate(
    problem: dict[str, Any],
    answer: Any,
    *,
    max_enum_p: int = 16,
    sound_tol: float = 1e-6,
    active_tol: float = 1e-10,
    reference: ExactAllSupportBoxQPReference | None = None,
) -> dict[str, Any]:
    """Full sound verification of a Box-L0 solution + disjoint subcube certificate (`R1`).

    Separates `solution_correct`, `partition_valid`, `leaf_bounds_valid`, `gap_satisfied`, and `certificate_valid`,
    accepts `1 <= len(certificate_leaves) <= max_leaves`, and rejects fake lower bounds & suboptimal feasible points!
    """
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    n, p = X.shape
    ridge = float(problem["ridge"])
    l0_pen = float(problem["l0_penalty"])
    box = float(problem["box"])
    gap_tol = float(problem.get("gap_tol", 1e-6))
    max_leaves = int(problem.get("max_leaves", 512))

    def _fail(reason: str, **extra: Any) -> dict[str, Any]:
        base = {
            "strict_pass": False,
            "solution_correct": False,
            "partition_valid": False,
            "leaf_bounds_valid": False,
            "gap_satisfied": False,
            "certificate_valid": False,
            "reason": reason,
            "leaf_count": 0,
            "true_objective": None,
            "certified_lower_bound": None,
            "reported_gap": None,
        }
        base.update(extra)
        return base

    if not isinstance(answer, dict):
        return _fail("non_dict_return")

    raw_w = answer.get("weights")
    raw_obj = answer.get("objective")
    raw_cert_lb = answer.get("certified_lower_bound")
    leaves = answer.get("certificate_leaves")

    if raw_w is None or isinstance(raw_w, (bool, str, bytes, dict)):
        return _fail("invalid_weights_container")
    try:
        arr_w_obj = np.asarray(raw_w)
        if arr_w_obj.dtype == bool:
            return _fail("bool_weights_rejected")
        w = np.asarray(raw_w, dtype=float)
    except Exception:
        return _fail("unparseable_weights")

    if (
        w.shape != (p,)
        or not bool(np.all(np.isfinite(w)))
        or isinstance(raw_obj, bool)
        or not isinstance(raw_obj, (int, float))
        or isinstance(raw_cert_lb, bool)
        or not isinstance(raw_cert_lb, (int, float))
    ):
        return _fail("invalid_weights_or_nonfinite_scalars")

    rep_obj = float(raw_obj)
    cert_lb = float(raw_cert_lb)
    if not (math.isfinite(rep_obj) and math.isfinite(cert_lb)):
        return _fail("nonfinite_objective_or_certified_lb")
    if float(np.max(np.abs(w))) > box + 1e-7:
        return _fail("box_constraint_violated")

    true_obj = eval_exact_box_l0_objective(X, y, w, ridge, l0_pen, active_tol=active_tol)
    obj_consistent = abs(rep_obj - true_obj) <= 1e-7

    # 1. Partition validation (`1 <= len(leaves) <= max_leaves` — valid 1-leaf certificates ACCEPTED!)
    partition_valid = True
    part_reason = "ok"
    parsed_leaves: list[tuple[frozenset[int], frozenset[int], float, dict[str, Any]]] = []
    if not isinstance(leaves, list) or len(leaves) < 1 or len(leaves) > max_leaves:
        partition_valid = False
        part_reason = "invalid_leaf_count"
    else:
        tot_measure = 0
        for lf in leaves:
            if not isinstance(lf, dict):
                partition_valid = False
                part_reason = "non_dict_leaf"
                break
            raw_b0 = lf.get("fixed_zero")
            raw_b1 = lf.get("fixed_nonzero")
            raw_llb = lf.get("leaf_lower_bound")
            if not isinstance(raw_b0, (list, tuple)) or not isinstance(raw_b1, (list, tuple)):
                partition_valid = False
                part_reason = "invalid_b0_b1_list"
                break
            if isinstance(raw_llb, bool) or not isinstance(raw_llb, (int, float)) or not math.isfinite(float(raw_llb)):
                partition_valid = False
                part_reason = "nonfinite_or_bool_leaf_lb"
                break
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in [*raw_b0, *raw_b1]):
                partition_valid = False
                part_reason = "non_int_leaf_indices"
                break
            b0 = frozenset(int(v) for v in raw_b0)
            b1 = frozenset(int(v) for v in raw_b1)
            if len(b0) != len(raw_b0) or len(b1) != len(raw_b1):
                partition_valid = False
                part_reason = "duplicate_leaf_indices"
                break
            if (b0 & b1) or any((idx < 0 or idx >= p) for idx in (b0 | b1)):
                partition_valid = False
                part_reason = "overlapping_or_oob_indices"
                break
            parsed_leaves.append((b0, b1, float(raw_llb), lf))
            tot_measure += 1 << (p - len(b0) - len(b1))

        if partition_valid and tot_measure != (1 << p):
            partition_valid = False
            part_reason = f"measure_mismatch({tot_measure}!={1 << p})"
        if partition_valid:
            for i in range(len(parsed_leaves)):
                b0_i, b1_i, _, _ = parsed_leaves[i]
                for j in range(i + 1, len(parsed_leaves)):
                    b0_j, b1_j, _, _ = parsed_leaves[j]
                    if not ((b0_i & b1_j) or (b1_i & b0_j)):
                        partition_valid = False
                        part_reason = f"overlapping_leaves({i},{j})"
                        break
                if not partition_valid:
                    break

    ref_obj: ExactAllSupportBoxQPReference | None = reference
    if ref_obj is None and p <= max_enum_p:
        ref_obj = ExactAllSupportBoxQPReference(X, y, ridge, l0_pen, box, active_tol=active_tol)

    indep_global_lb: float | None = None
    indep_global_ub: float | None = None
    if ref_obj is not None:
        indep_global_lb, indep_global_ub = ref_obj.interval()

    # 2. Leaf lower-bound mathematical soundness check
    leaf_bounds_valid = bool(partition_valid and len(parsed_leaves) >= 1)
    leaf_reason = "ok" if leaf_bounds_valid else part_reason
    leaf_diagnostics: list[dict[str, Any]] = []
    min_leaf_lb = float("inf")

    if partition_valid:
        for idx_l, (b0_i, b1_i, claimed_lb, raw_lf) in enumerate(parsed_leaves):
            min_leaf_lb = min(min_leaf_lb, claimed_lb)
            verified_lb: float | None = None
            verified_mode = "unverified"
            witness_msg = ""

            if "dual_nu" in raw_lf and raw_lf["dual_nu"] is not None:
                ok_d, msg_d, lb_d = evaluate_fenchel_dual_leaf_lower_bound(
                    X, y, ridge, l0_pen, box, b0_i, b1_i, raw_lf["dual_nu"], active_tol=active_tol
                )
                if not ok_d:
                    leaf_bounds_valid = False
                    leaf_reason = f"leaf_{idx_l}_{msg_d}"
                else:
                    verified_lb = lb_d
                    verified_mode = "fenchel_dual_witness"
                    witness_msg = msg_d
            elif "minorant_diagonal" in raw_lf and raw_lf["minorant_diagonal"] is not None:
                c_wit = raw_lf.get("minorant_center", np.zeros(p, dtype=float))
                ok_m, msg_m, lb_m = evaluate_psd_minorant_leaf_lower_bound(
                    X, y, ridge, l0_pen, box, b0_i, b1_i, raw_lf["minorant_diagonal"], c_wit, active_tol=active_tol
                )
                if not ok_m:
                    leaf_bounds_valid = False
                    leaf_reason = f"leaf_{idx_l}_{msg_m}"
                else:
                    verified_lb = lb_m
                    verified_mode = "psd_minorant_witness"
                    witness_msg = msg_m

            ref_low: float | None = None
            ref_up: float | None = None
            if ref_obj is not None:
                ref_low, ref_up = ref_obj.interval(b0_i, b1_i)
                if verified_lb is None:
                    verified_lb = ref_low
                    verified_mode = "exact_support_convex_qp_reference"

            if verified_lb is None:
                leaf_bounds_valid = False
                leaf_reason = f"leaf_{idx_l}_missing_proof_witness_above_max_enum_p({p}>{max_enum_p})"
                sound_i = False
            else:
                sound_vs_wit = claimed_lb <= verified_lb + sound_tol
                sound_vs_ref = True if ref_up is None else (claimed_lb <= ref_up + sound_tol and claimed_lb <= (ref_low or verified_lb) + sound_tol)
                sound_i = bool(sound_vs_wit and sound_vs_ref)
                if not sound_i:
                    leaf_bounds_valid = False
                    leaf_reason = (
                        f"unsound_leaf_lower_bound(leaf={idx_l},claimed={claimed_lb:.6f}>verified={verified_lb:.6f})"
                    )

            leaf_diagnostics.append(
                {
                    "leaf_index": idx_l,
                    "claimed_lb": claimed_lb,
                    "verified_lb": verified_lb,
                    "reference_ub": ref_up,
                    "verification_mode": verified_mode,
                    "witness_note": witness_msg,
                    "sound": sound_i,
                }
            )

    cert_consistent = bool(partition_valid and cert_lb <= min_leaf_lb + 1e-8 and cert_lb <= true_obj + 1e-7)
    certificate_valid = bool(partition_valid and leaf_bounds_valid and cert_consistent)

    reported_gap = true_obj - cert_lb
    gap_satisfied = bool(-1e-8 <= reported_gap <= gap_tol + 1e-7)

    if indep_global_lb is not None:
        solution_correct = bool(obj_consistent and (true_obj - indep_global_lb <= max(gap_tol, sound_tol) + 1e-6))
    else:
        # Above enumeration limit `p > max_enum_p`, global optimality is established iff a valid certificate closes the gap!
        solution_correct = bool(obj_consistent and certificate_valid and gap_satisfied)

    strict_pass = bool(solution_correct and certificate_valid and gap_satisfied)
    reason_final = (
        "ok"
        if strict_pass
        else (
            "objective_report_mismatch"
            if not obj_consistent
            else (
                part_reason
                if not partition_valid
                else (
                    leaf_reason
                    if not leaf_bounds_valid
                    else (
                        "certified_lb_exceeds_min_leaf_or_objective"
                        if not cert_consistent
                        else ("suboptimal_solution_vs_reference" if not solution_correct else f"gap_exceeded({reported_gap:.3e})")
                    )
                )
            )
        )
    )
    return {
        "strict_pass": strict_pass,
        "solution_correct": solution_correct,
        "partition_valid": partition_valid,
        "leaf_bounds_valid": leaf_bounds_valid,
        "certificate_valid": certificate_valid,
        "gap_satisfied": gap_satisfied,
        "objective_consistent": obj_consistent,
        "reason": reason_final,
        "leaf_count": len(leaves) if isinstance(leaves, list) else 0,
        "true_objective": round(true_obj, 10),
        "certified_lower_bound": round(cert_lb, 10),
        "independent_global_lower_bound": (round(indep_global_lb, 10) if indep_global_lb is not None else None),
        "independent_global_upper_bound": (round(indep_global_ub, 10) if indep_global_ub is not None else None),
        "reported_gap": round(max(0.0, reported_gap), 10),
        "leaves": leaf_diagnostics,
    }


CANONICAL_BOX_L0_CERTIFICATE_SCHEMA_VERSION = "hegel_box_l0_cert_schema_v2"


def normalize_and_validate_box_l0_public_schema(
    cert_dict: Any,
    p: int,
    *,
    box: float | None = None,
    max_leaves: int = 2048,
    allow_index_list_compat: bool = False,
) -> dict[str, Any]:
    """Pure PUBLIC schema & partition checker for symmetric-box L0 certificates (`M9.1.1` `C1`).

    Contains ZERO private data/labels and may be shipped verbatim inside `/holdout_task/` so ALL
    experimental arms (`Bare`, `Knowledge`, `Executable`) share identical deliverable format knowledge!
    Canonical leaf mask representation: `fixed_zero` and `fixed_active` as length-`p` boolean sequences
    (`encoding == 'bool_mask_p'`). When `allow_index_list_compat=True` (used ONLY for explicit retrospective
    sensitivity regrading of ambiguous v1 specs or when `encoding == 'index_list'` is declared), converts
    unambiguous duplicate-free integer index lists in `[0, p)` into canonical `(p,)` boolean masks & index lists.
    """
    if not isinstance(cert_dict, dict):
        return {"schema_valid": False, "partition_valid": False, "reason": "certificate_not_a_dict", "canonical": None}

    req_top = {"weights", "objective", "certified_lower_bound", "subcube_leaves"}
    if not (req_top <= set(cert_dict.keys())):
        missing = sorted(req_top - set(cert_dict.keys()))
        return {"schema_valid": False, "partition_valid": False, "reason": f"missing_keys:{missing}", "canonical": None}

    raw_obj = cert_dict.get("objective")
    raw_clb = cert_dict.get("certified_lower_bound")
    if (
        isinstance(raw_obj, (bool, np.bool_))
        or isinstance(raw_clb, (bool, np.bool_))
        or not isinstance(raw_obj, (int, float, np.number))
        or not isinstance(raw_clb, (int, float, np.number))
        or not (math.isfinite(float(raw_obj)) and math.isfinite(float(raw_clb)))
    ):
        return {"schema_valid": False, "partition_valid": False, "reason": "nonfinite_or_bool_scalars", "canonical": None}

    try:
        w = np.asarray(cert_dict["weights"], dtype=float)
    except Exception:
        return {"schema_valid": False, "partition_valid": False, "reason": "weights_not_numeric_array", "canonical": None}
    if w.shape != (p,) or not bool(np.all(np.isfinite(w))):
        return {"schema_valid": False, "partition_valid": False, "reason": f"bad_weights_shape_or_nan(expected=({p},))", "canonical": None}
    if box is not None and (float(np.min(w)) < -float(box) - 1e-7 or float(np.max(w)) > float(box) + 1e-7):
        return {"schema_valid": False, "partition_valid": False, "reason": "weights_exceed_box_bounds", "canonical": None}

    leaves = cert_dict.get("subcube_leaves")
    if not isinstance(leaves, list) or not (1 <= len(leaves) <= int(max_leaves)):
        return {"schema_valid": False, "partition_valid": False, "reason": f"bad_leaf_count(expected_1_to_{max_leaves})", "canonical": None}

    decl_enc = str(cert_dict.get("encoding") or "").strip()
    canonical_leaves: list[dict[str, Any]] = []
    encodings_seen: set[str] = set()
    tot_measure = 0
    parsed_sets: list[tuple[frozenset[int], frozenset[int]]] = []

    def _parse_mask_or_indices(raw_seq: Any) -> tuple[bool, str, list[int], str]:
        if isinstance(raw_seq, np.ndarray):
            if raw_seq.ndim != 1:
                return False, "mask_not_1d", [], ""
            if raw_seq.dtype == np.bool_:
                if raw_seq.shape != (p,):
                    return False, f"bool_mask_wrong_length({len(raw_seq)}!={p})", [], ""
                return True, "ok", [int(i) for i in np.flatnonzero(raw_seq)], "bool_mask_p"
            seq_list = raw_seq.tolist()
        elif isinstance(raw_seq, (list, tuple)):
            seq_list = list(raw_seq)
        else:
            return False, "leaf_mask_not_sequence", [], ""

        if len(seq_list) == p and all(isinstance(x, (bool, np.bool_)) for x in seq_list) and decl_enc != "index_list":
            return True, "ok", [i for i, flag in enumerate(seq_list) if bool(flag)], "bool_mask_p"

        if not (allow_index_list_compat or decl_enc == "index_list"):
            return False, f"expected_length_{p}_bool_mask_got_non_bool_or_wrong_len", [], ""

        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in seq_list):
            return False, "non_int_in_index_list", [], ""
        ints = [int(x) for x in seq_list]
        if len(set(ints)) != len(ints) or any(ix < 0 or ix >= p for ix in ints):
            return False, "duplicate_or_oob_index_in_leaf", [], ""
        return True, "ok", sorted(ints), "index_list"

    for idx_l, lf in enumerate(leaves):
        if not isinstance(lf, dict):
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_not_dict", "canonical": None}
        raw_fz = lf.get("fixed_zero")
        raw_fa = lf.get("fixed_active") if "fixed_active" in lf else lf.get("fixed_nonzero")
        raw_lb = lf.get("lb") if "lb" in lf else lf.get("leaf_lower_bound")
        if raw_fz is None or raw_fa is None or raw_lb is None:
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_missing_required_fields", "canonical": None}
        if isinstance(raw_lb, (bool, np.bool_)) or not isinstance(raw_lb, (int, float, np.number)) or not math.isfinite(float(raw_lb)):
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_bad_lb", "canonical": None}

        ok0, rsn0, idx0, enc0 = _parse_mask_or_indices(raw_fz)
        if not ok0:
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_fixed_zero_{rsn0}", "canonical": None}
        ok1, rsn1, idx1, enc1 = _parse_mask_or_indices(raw_fa)
        if not ok1:
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_fixed_active_{rsn1}", "canonical": None}
        encodings_seen.update({enc0, enc1})

        s0, s1 = frozenset(idx0), frozenset(idx1)
        if s0 & s1:
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_overlapping_zero_and_active", "canonical": None}

        has_diag = lf.get("minorant_diagonal") is not None and lf.get("minorant_center") is not None
        has_dual = lf.get("dual_nu") is not None
        if not (has_diag or has_dual):
            return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_missing_witness_fields", "canonical": None}

        can_lf: dict[str, Any] = {
            "fixed_zero": idx0,
            "fixed_nonzero": idx1,
            "fixed_zero_bool": [j in s0 for j in range(p)],
            "fixed_active_bool": [j in s1 for j in range(p)],
            "leaf_lower_bound": float(raw_lb),
        }
        if has_diag:
            dv = np.asarray(lf["minorant_diagonal"], dtype=float).ravel()
            cv = np.asarray(lf["minorant_center"], dtype=float).ravel()
            if dv.shape != (p,) or cv.shape != (p,) or not (np.all(np.isfinite(dv)) and np.all(np.isfinite(cv))):
                return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_bad_minorant_shape_or_nan", "canonical": None}
            can_lf["minorant_diagonal"] = dv.tolist()
            can_lf["minorant_center"] = cv.tolist()
        if has_dual:
            nu = np.asarray(lf["dual_nu"], dtype=float).ravel()
            if nu.ndim != 1 or nu.size == 0 or not np.all(np.isfinite(nu)):
                return {"schema_valid": False, "partition_valid": False, "reason": f"leaf_{idx_l}_bad_dual_nu", "canonical": None}
            can_lf["dual_nu"] = nu.tolist()

        canonical_leaves.append(can_lf)
        parsed_sets.append((s0, s1))
        tot_measure += 1 << (p - len(s0) - len(s1))

    if tot_measure != (1 << p):
        return {
            "schema_valid": True,
            "partition_valid": False,
            "reason": f"measure_mismatch({tot_measure}!={1 << p})",
            "canonical": None,
        }

    if p <= 62:
        zu = np.array([sum(1 << j for j in s0) for s0, _ in parsed_sets], dtype=np.uint64)
        au = np.array([sum(1 << j for j in s1) for _, s1 in parsed_sets], dtype=np.uint64)
        sep = (zu[:, None] & au[None, :]) | (au[:, None] & zu[None, :])
        np.fill_diagonal(sep, np.uint64(1))
        if bool(np.any(sep == 0)):
            return {"schema_valid": True, "partition_valid": False, "reason": "overlapping_leaves", "canonical": None}
    else:
        for i in range(len(parsed_sets)):
            s0_i, s1_i = parsed_sets[i]
            for j in range(i + 1, len(parsed_sets)):
                s0_j, s1_j = parsed_sets[j]
                if not ((s0_i & s1_j) or (s1_i & s0_j)):
                    return {"schema_valid": True, "partition_valid": False, "reason": f"overlapping_leaves({i},{j})", "canonical": None}

    return {
        "schema_valid": True,
        "partition_valid": True,
        "schema_version": CANONICAL_BOX_L0_CERTIFICATE_SCHEMA_VERSION,
        "encodings_seen": sorted(encodings_seen),
        "leaf_count": len(canonical_leaves),
        "reason": "ok",
        "canonical": {
            "weights": w.tolist(),
            "objective": float(raw_obj),
            "certified_lower_bound": float(raw_clb),
            "subcube_leaves": canonical_leaves,
        },
    }


def verify_eq02_style_box_l0_certificate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    box: float,
    l0_penalty: float,
    ridge: float,
    cert_dict: Any,
    dual_nu_sign: float = -1.0,
    active_tol: float = 1e-8,
    gap_tol: float = 1e-5,
    sound_tol: float = 1e-5,
    max_leaves: int = 2048,
    allow_index_list_compat: bool = False,
) -> dict[str, Any]:
    """Centralized `M9.2.1` `A1` adapter mapping `subcube_leaves` (`bool[p]` `fixed_zero`/`fixed_active`, `lb`)
    onto the single canonical `verify_box_l0_subcube_certificate` kernel (`0` duplicated dual/minorant math!).

    Note on `dual_nu_sign`:
    - `evaluate_fenchel_dual_leaf_lower_bound` uses `+nu^T y - 0.5*||nu||_2^2 - ...`.
    - Tasks whose prompt writes `-nu^T y - 0.5*||nu||_2^2 - ...` (`eq02` / `hd01` convention) pass `dual_nu_sign = -1.0`;
      tasks using `+nu^T y` pass `dual_nu_sign = +1.0`.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    _, p = X_arr.shape
    pub_chk = normalize_and_validate_box_l0_public_schema(
        cert_dict,
        p,
        box=float(box),
        max_leaves=int(max_leaves),
        allow_index_list_compat=allow_index_list_compat,
    )
    if not (pub_chk.get("schema_valid") and pub_chk.get("partition_valid")):
        return {
            "strict_pass": False,
            "schema_valid": bool(pub_chk.get("schema_valid")),
            "partition_valid": bool(pub_chk.get("partition_valid")),
            "leaf_bounds_valid": False,
            "certificate_valid": False,
            "gap_satisfied": False,
            "reason": str(pub_chk.get("reason") or "public_schema_or_partition_invalid"),
            "reported_gap": 1e9,
            "leaf_count": 0,
        }
    can = pub_chk["canonical"]
    mapped_leaves: list[dict[str, Any]] = []
    sgn = 1.0 if float(dual_nu_sign) >= 0.0 else -1.0
    for lf in can["subcube_leaves"]:
        mlf: dict[str, Any] = {
            "fixed_zero": list(lf["fixed_zero"]),
            "fixed_nonzero": list(lf["fixed_nonzero"]),
            "leaf_lower_bound": float(lf["leaf_lower_bound"]),
        }
        if "minorant_diagonal" in lf and "minorant_center" in lf:
            mlf["minorant_diagonal"] = list(lf["minorant_diagonal"])
            mlf["minorant_center"] = list(lf["minorant_center"])
        if "dual_nu" in lf:
            mlf["dual_nu"] = (sgn * np.asarray(lf["dual_nu"], dtype=float)).tolist()
        mapped_leaves.append(mlf)
    mapped_ans = {
        "weights": list(can["weights"]),
        "objective": float(can["objective"]),
        "certified_lower_bound": float(can["certified_lower_bound"]),
        "certificate_leaves": mapped_leaves,
    }
    res = verify_box_l0_subcube_certificate(
        {
            "X": X_arr,
            "y": y_arr,
            "ridge": float(ridge),
            "l0_penalty": float(l0_penalty),
            "box": float(box),
            "gap_tol": float(gap_tol),
            "max_leaves": int(max_leaves),
        },
        mapped_ans,
        max_enum_p=0,
        sound_tol=float(sound_tol),
        active_tol=float(active_tol),
    )
    res["schema_valid"] = True
    return res


def get_public_schema_validator_only_source() -> str:
    """Return standalone pure-public source defining `CANONICAL_BOX_L0_CERTIFICATE_SCHEMA_VERSION` and
    `normalize_and_validate_box_l0_public_schema` extracted verbatim from this module (`0` private verifier logic).
    """
    import ast as _ast
    from pathlib import Path as _Path

    src = _Path(__file__).read_text(encoding="utf-8")
    tree = _ast.parse(src)
    kept_segments: list[str] = [
        "from __future__ import annotations\nimport math\nfrom typing import Any\nimport numpy as np\n"
    ]
    for node in tree.body:
        if isinstance(node, _ast.Assign) and any(
            isinstance(t, _ast.Name) and t.id == "CANONICAL_BOX_L0_CERTIFICATE_SCHEMA_VERSION" for t in node.targets
        ):
            seg = _ast.get_source_segment(src, node)
            if seg:
                kept_segments.append(seg)
        elif isinstance(node, _ast.FunctionDef) and node.name == "normalize_and_validate_box_l0_public_schema":
            seg = _ast.get_source_segment(src, node)
            if seg:
                kept_segments.append(seg)
    return "\n\n".join(kept_segments) + "\n"



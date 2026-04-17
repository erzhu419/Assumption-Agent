"""
Blackwell partial order: A ≽_B B iff ∃ stochastic M such that K_B = M K_A.

We check feasibility via a linear program (one per column pair).
Relaxed: we allow a small residual ‖M K_A - K_B‖₁ ≤ tol so numerical slack
doesn't produce spurious "not dominated" answers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linprog


def blackwell_feasible(K_a: np.ndarray, K_b: np.ndarray, tol: float = 0.05) -> Tuple[bool, float]:
    """
    Check whether K_b = M K_a for some row-stochastic M.

    Formulated as: min 1-norm residual ‖K_b - M K_a‖_1 subject to M>=0, rows sum to 1.

    Returns (feasible, residual). feasible=True means residual <= tol.
    """
    # Dimensions
    nx_a, na_a = K_a.shape
    nx_b, na_b = K_b.shape
    if na_a != na_b:
        raise ValueError("action dims must match")
    # To keep LP small, we reduce to: minimize sum of positive/negative slacks
    # row-wise over K_b. This is O(nx_b × (nx_a + 2*na_a)) variables.

    # Variables per row of K_b (separately; iterate over rows).
    # For each row index r of K_b (target):
    #   decision var m_r (length nx_a)  subject to m_r >= 0 and sum(m_r) = 1
    #   residual slacks u_r, v_r (length na_a) >= 0 with  K_a^T m_r - K_b[r] = u_r - v_r
    # minimize sum(u_r + v_r).
    # Aggregate row residuals for return.

    total_residual = 0.0
    K_aT = K_a.T  # (na, nx_a)

    for r in range(nx_b):
        target = K_b[r]  # (na,)

        nvar = nx_a + 2 * na_a  # m, u, v
        c = np.concatenate([np.zeros(nx_a), np.ones(2 * na_a)])

        # Equality: K_aT @ m - u + v == target  (for each action)
        A_eq = np.zeros((na_a + 1, nvar))
        A_eq[:na_a, :nx_a] = K_aT
        A_eq[:na_a, nx_a:nx_a + na_a] = -np.eye(na_a)
        A_eq[:na_a, nx_a + na_a:] = np.eye(na_a)
        # sum m = 1
        A_eq[na_a, :nx_a] = 1.0

        b_eq = np.concatenate([target, [1.0]])
        bounds = [(0, None)] * nvar

        try:
            res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                          method="highs", options={"time_limit": 5})
            if not res.success:
                return False, float("inf")
            total_residual += float(res.fun)
        except Exception:
            return False, float("inf")

    total_residual /= max(nx_b, 1)  # average per row
    return total_residual <= tol, float(total_residual)


def blackwell_relation(K_a: np.ndarray, K_b: np.ndarray, tol: float = 0.05) -> str:
    """Return one of: 'a_dominates_b', 'b_dominates_a', 'both_dominate' (equivalent),
    'neither'."""
    a_over_b, r_ab = blackwell_feasible(K_a, K_b, tol)
    b_over_a, r_ba = blackwell_feasible(K_b, K_a, tol)
    if a_over_b and b_over_a:
        return "both_dominate"
    if a_over_b:
        return "a_dominates_b"
    if b_over_a:
        return "b_dominates_a"
    return "neither"

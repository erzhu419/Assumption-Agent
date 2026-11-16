"""Symbolic Algebraic Elimination, Emergent Invariants & Certified Reachability/Impossibility Proofs (v10.2).

Implements exact rational certificate checking, solver synthesis, and premise-guided discrepancy
repair over ``fractions.Fraction`` and ``sympy.Poly(..., domain='QQ')`` with strict trust-boundary
and domain-legality semantics (remediating v10.2 audit findings F01, F02, F05, F06, F07, F10, F11,
F12, F13, F14):

1. Unified Task & Domain Semantics (``validate_control_task_semantics``):
   - Enforces consistent dimensionality, finite rationalizable parameters (explicitly reporting any
     float-to-rational rounding delta), non-negative ``step_cost`` and ``fuel_weights``, and complete
     ``goal_matrix`` / ``goal_rhs`` across ``BudgetedControlLab``, ``prove_task_solvability``, and
     ``SymbolicProofKernel``.
   - Distinguishes three explicit ``state_bounds_mode`` regimes:
     * ``"physical_legality"`` (default when ``state_bounds`` is declared on ``task``): enforced
       identically by ``BudgetedControlLab``, the witness planner, the witness verifier, and the
       potential verifier.
     * ``"analysis_region"``: ``state_bounds`` is NOT enforced by execution/witnesses; unconditional
       impossibility requires an exact rational reachable-box enclosure proof
       (``verify_reachable_box_enclosure``) or zero state-bound Farkas multipliers. Otherwise the
       kernel returns ``status = "conditional_impossibility_domain_unverified"`` (``verified = False``).
     * ``"unbounded"`` (when ``state_bounds`` is ``None``): never invents a default box; only
       globally valid affine potentials with zero state-bound multipliers can certify impossibility.

2. Honest Formal Artifact Status (``emit_lean4_certificate``):
   - Trusted checker base is ``python_fractions_and_sympy_QQ``.
   - Emitted Lean 4 text is explicitly tagged ``lean_status = "draft_only"`` (not Lean-compiled)
     and never labeled ``proof_verified`` without real Lean theorem compilation.

3. Budgeted Symbolic Proof & Equal-Tool Discrepancy Repair:
   - Threads optional ``budget`` (any ``SearchBudget`` supporting ``.spend(int) -> bool``) across
     invariant synthesis, kernel verification, LP/Farkas proving, rollout witness certification,
     and multi-mechanism discrepancy repair.
   - Separates ``model_origin`` (``"supplied_model"`` vs ``"learned_model"``), ``declared_premises``,
     ``empirically_supported_premises``, ``refuted_premises``, and ``formal_obligations``.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from random import Random
from typing import Any

import numpy as np
from scipy.optimize import linprog
import sympy as sp


TRUSTED_CHECKER_BASE = "python_fractions_and_sympy_QQ"

VALID_DIRECTIONS = frozenset({"right", "left", "symmetric"})
VALID_MODEL_ORIGINS = frozenset({"supplied_model", "learned_model"})
VALID_STATE_BOUNDS_MODES = frozenset({"physical_legality", "analysis_region", "unbounded"})


def _spend_budget(budget: Any | None, units: int) -> tuple[bool, int]:
    """Deduct ``units`` from ``budget`` if provided; return ``(allowed, spent_units)``."""
    amt = max(1, int(units))
    if budget is None:
        return True, amt
    ok = bool(budget.spend(amt))
    return (True, amt) if ok else (False, 0)


def _to_frac_with_delta(
    val: float | int | str | Fraction, max_den: int = 10_000_000
) -> tuple[Fraction, float, bool]:
    """Convert ``val`` to ``Fraction`` while rejecting booleans/NaN/Inf and tracking rounding delta."""
    if isinstance(val, bool) or val is None:
        raise ValueError(f"Invalid numeric parameter: {val!r}")
    if isinstance(val, Fraction):
        return val, 0.0, True
    if isinstance(val, int):
        return Fraction(val), 0.0, True
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in {"nan", "inf", "-inf", "+inf", "infinity", "-infinity"}:
            raise ValueError(f"Non-finite string parameter: {val!r}")
        return Fraction(s), 0.0, True
    if isinstance(val, (float, np.floating)):
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite float parameter: {val!r}")
        exact_bin = Fraction(fval)
        approx = exact_bin.limit_denominator(max_den)
        delta = float(abs(approx - exact_bin))
        is_exact = approx == exact_bin
        return approx, delta, is_exact
    raise ValueError(f"Unsupported numeric type: {type(val).__name__}")


def _to_frac(val: float | int | str | Fraction, max_den: int = 10_000_000) -> Fraction:
    fr, _, _ = _to_frac_with_delta(val, max_den=max_den)
    return fr


def _to_sympy_rat(val: float | int | str | Fraction) -> sp.Rational:
    fr = _to_frac(val)
    return sp.Rational(fr.numerator, fr.denominator)


def validate_control_task_semantics(
    model_spec: dict,
    task: dict,
    *,
    lab_state_bounds: tuple[float, float] | None = None,
    lab_state_bounds_mode: str | None = None,
) -> dict[str, Any]:
    """Validate dimensions, finiteness, non-negative costs, goal completeness, and domain mode."""
    if not isinstance(model_spec, dict) or not isinstance(task, dict):
        return {"valid": False, "status": "invalid_certificate", "reason": "invalid_spec_or_task_type"}

    try:
        raw_n = model_spec.get("objects")
        if isinstance(raw_n, bool) or raw_n is None or int(raw_n) < 1 or int(raw_n) != raw_n:
            return {"valid": False, "status": "invalid_certificate", "reason": "invalid_object_count"}
        n = int(raw_n)
    except (TypeError, ValueError):
        return {"valid": False, "status": "invalid_certificate", "reason": "invalid_object_count"}

    model_origin = str(model_spec.get("model_origin", "supplied_model"))
    if model_origin not in VALID_MODEL_ORIGINS:
        return {
            "valid": False,
            "status": "invalid_certificate",
            "reason": f"invalid_model_origin:{model_origin}",
        }

    direction = str(model_spec.get("direction", "right"))
    if direction not in VALID_DIRECTIONS:
        return {
            "valid": False,
            "status": "invalid_certificate",
            "reason": f"invalid_direction:{direction}",
        }

    max_delta = 0.0
    any_rounded = False

    def conv(v: Any) -> Fraction:
        nonlocal max_delta, any_rounded
        fr, dlt, ex = _to_frac_with_delta(v)
        if not ex:
            any_rounded = True
            max_delta = max(max_delta, dlt)
        return fr

    try:
        raw_positions = model_spec.get("positions", [0] * n)
        if len(raw_positions) != n:
            return {"valid": False, "status": "invalid_certificate", "reason": "positions_dim_mismatch"}
        positions = [conv(v) for v in raw_positions]
        gamma = conv(model_spec.get("coupling", 0))
        radius = conv(model_spec.get("radius", 1))
        if radius <= 0:
            return {"valid": False, "status": "invalid_certificate", "reason": "nonpositive_radius"}
        sink_rate = conv(model_spec.get("sink_rate", 0))
        port_gain = conv(model_spec.get("boundary_port_gain", 0))
        port_thresh = conv(model_spec.get("boundary_port_threshold", Fraction(3, 2)))

        raw_w0 = task.get("initial_state")
        if not isinstance(raw_w0, (list, tuple)) or len(raw_w0) != n:
            return {"valid": False, "status": "invalid_certificate", "reason": "initial_state_dim_mismatch"}
        w0 = [conv(v) for v in raw_w0]

        raw_cb = task.get("control_bounds")
        if not isinstance(raw_cb, (list, tuple)) or len(raw_cb) != 2:
            return {"valid": False, "status": "invalid_certificate", "reason": "invalid_control_bounds"}
        u_lo, u_hi = conv(raw_cb[0]), conv(raw_cb[1])
        if u_lo > u_hi:
            return {"valid": False, "status": "invalid_certificate", "reason": "inverted_control_bounds"}

        budget_val = conv(task["budget"])
        if budget_val < 0:
            return {"valid": False, "status": "invalid_certificate", "reason": "negative_budget"}

        step_cost = conv(task.get("step_cost", 0))
        if step_cost < 0:
            return {"valid": False, "status": "invalid_certificate", "reason": "negative_step_cost"}

        raw_fw = task.get("fuel_weights", [1] * n)
        if not isinstance(raw_fw, (list, tuple)) or len(raw_fw) != n:
            return {"valid": False, "status": "invalid_certificate", "reason": "fuel_weights_dim_mismatch"}
        fuel_weights = [conv(v) for v in raw_fw]
        if any(fw < 0 for fw in fuel_weights):
            return {"valid": False, "status": "invalid_certificate", "reason": "negative_fuel_weight"}

        raw_gm = task.get("goal_matrix")
        raw_gr = task.get("goal_rhs")
        if (
            not isinstance(raw_gm, (list, tuple))
            or not isinstance(raw_gr, (list, tuple))
            or len(raw_gm) == 0
            or len(raw_gm) != len(raw_gr)
        ):
            return {"valid": False, "status": "invalid_certificate", "reason": "incomplete_or_truncated_goal"}
        a_goal: list[list[Fraction]] = []
        for r_idx, row in enumerate(raw_gm):
            if not isinstance(row, (list, tuple)) or len(row) != n:
                return {
                    "valid": False,
                    "status": "invalid_certificate",
                    "reason": f"goal_matrix_row_{r_idx}_dim_mismatch",
                }
            a_goal.append([conv(v) for v in row])
        b_goal = [conv(v) for v in raw_gr]

        raw_sb = task.get("state_bounds", lab_state_bounds)
        raw_mode = task.get("state_bounds_mode", lab_state_bounds_mode)
        if raw_sb is None:
            if raw_mode in {"physical_legality", "analysis_region"}:
                return {
                    "valid": False,
                    "status": "invalid_certificate",
                    "reason": f"state_bounds_mode_{raw_mode}_requires_state_bounds",
                }
            state_bounds_mode = "unbounded"
            w_lo, w_hi = None, None
        else:
            if not isinstance(raw_sb, (list, tuple)) or len(raw_sb) != 2:
                return {"valid": False, "status": "invalid_certificate", "reason": "invalid_state_bounds"}
            w_lo, w_hi = conv(raw_sb[0]), conv(raw_sb[1])
            if w_lo > w_hi:
                return {"valid": False, "status": "invalid_certificate", "reason": "inverted_state_bounds"}
            state_bounds_mode = str(raw_mode) if raw_mode is not None else "physical_legality"
            if state_bounds_mode not in VALID_STATE_BOUNDS_MODES or state_bounds_mode == "unbounded":
                return {
                    "valid": False,
                    "status": "invalid_certificate",
                    "reason": f"invalid_state_bounds_mode:{state_bounds_mode}",
                }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "valid": False,
            "status": "invalid_certificate",
            "reason": f"parameter_validation_error:{exc}",
        }

    if state_bounds_mode == "physical_legality" and w_lo is not None and w_hi is not None:
        for i, val in enumerate(w0):
            if not (w_lo <= val <= w_hi):
                return {
                    "valid": False,
                    "status": "invalid_certificate",
                    "reason": f"initial_state_outside_physical_state_bounds_at_obj_{i}",
                }

    # Build exact rational linear dynamics w' = A w + B u + c
    a_mat = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    b_mat = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    c_vec = [Fraction(0) for _ in range(n)]

    if gamma != 0:
        if direction in {"right", "left"}:
            sgn = Fraction(1 if direction == "right" else -1)
            for src in range(n):
                for dst in range(n):
                    if src != dst and Fraction(0) < sgn * (positions[dst] - positions[src]) < radius:
                        a_mat[src][src] -= gamma
                        a_mat[dst][src] += gamma
        else:
            for i in range(n):
                for j in range(n):
                    if i != j and abs(positions[j] - positions[i]) < radius:
                        a_mat[i][i] -= gamma
                        a_mat[i][j] += gamma

    if sink_rate != 0:
        for i in range(n):
            a_mat[i][i] -= sink_rate

    if port_gain != 0:
        for i in range(n):
            if abs(positions[i]) >= port_thresh:
                b_mat[i][i] += port_gain

    declared_premises = list(
        model_spec.get(
            "premise_ids",
            ["P_conservation", "P_interaction_direction", "P_locality_radius", "P_control_input_gain"],
        )
    )
    empirically_supported = (
        list(model_spec.get("empirically_supported_premises", ()))
        if model_origin == "learned_model"
        else []
    )
    refuted_premises = list(model_spec.get("refuted_premises", ()))

    rational_summary = {
        "objects": n,
        "direction": direction,
        "positions": [str(v) for v in positions],
        "coupling": str(gamma),
        "radius": str(radius),
        "sink_rate": str(sink_rate),
        "boundary_port_gain": str(port_gain),
        "boundary_port_threshold": str(port_thresh),
        "initial_state": [str(v) for v in w0],
        "control_bounds": [str(u_lo), str(u_hi)],
        "state_bounds": [str(w_lo), str(w_hi)] if w_lo is not None and w_hi is not None else None,
        "state_bounds_mode": state_bounds_mode,
        "budget": str(budget_val),
        "step_cost": str(step_cost),
        "fuel_weights": [str(v) for v in fuel_weights],
        "goal_matrix": [[str(v) for v in row] for row in a_goal],
        "goal_rhs": [str(v) for v in b_goal],
        "rationalization_changed_values": bool(any_rounded),
        "max_rationalization_abs_delta": float(max_delta),
    }

    return {
        "valid": True,
        "n": n,
        "model_origin": model_origin,
        "direction": direction,
        "positions": positions,
        "gamma": gamma,
        "radius": radius,
        "sink_rate": sink_rate,
        "port_gain": port_gain,
        "port_thresh": port_thresh,
        "w0": w0,
        "u_lo": u_lo,
        "u_hi": u_hi,
        "w_lo": w_lo,
        "w_hi": w_hi,
        "state_bounds_mode": state_bounds_mode,
        "budget": budget_val,
        "step_cost": step_cost,
        "fuel_weights": fuel_weights,
        "a_goal": a_goal,
        "b_goal": b_goal,
        "a_mat": a_mat,
        "b_mat": b_mat,
        "c_vec": c_vec,
        "declared_premises": declared_premises,
        "empirically_supported_premises": empirically_supported,
        "refuted_premises": refuted_premises,
        "rational_model": rational_summary,
    }


def verify_reachable_box_enclosure(
    sem: dict[str, Any],
    mu_step: list[Fraction] | None = None,
    *,
    budget: Any | None = None,
) -> dict[str, Any]:
    """Verify in exact ``Fraction`` whether ``state_bounds`` encloses all budget-reachable trajectories."""
    n = sem["n"]
    w_lo, w_hi = sem["w_lo"], sem["w_hi"]
    w0 = sem["w0"]
    a_mat, b_mat, c_vec = sem["a_mat"], sem["b_mat"], sem["c_vec"]
    u_lo, u_hi = sem["u_lo"], sem["u_hi"]
    b_tot = sem["budget"]
    step_cost = sem["step_cost"]
    fw = sem["fuel_weights"]

    ok_b, spent = _spend_budget(budget, 4 * n * n + 4 * n)
    if not ok_b:
        return {"enclosed": False, "reason": "budget_exhausted", "compute_cost": 0}

    uses_hi = [bool(mu_step is None or mu_step[i] > 0) for i in range(n)]
    uses_lo = [bool(mu_step is None or mu_step[n + i] > 0) for i in range(n)]
    if not any(uses_hi) and not any(uses_lo):
        return {
            "enclosed": True,
            "method": "global_zero_state_multipliers",
            "compute_cost": spent,
        }

    if w_lo is None or w_hi is None:
        return {
            "enclosed": False,
            "reason": "undeclared_state_bounds_with_nonzero_state_multipliers",
            "compute_cost": spent,
        }

    if any(not (w_lo <= w0[i] <= w_hi) for i in range(n)):
        return {
            "enclosed": False,
            "reason": "initial_state_outside_analysis_region",
            "compute_cost": spent,
        }

    # Criterion 1: Global L1 contraction enclosure (||A||_1 <= 1, c == 0, positive fuel on active B columns)
    col_l1_a = [sum(abs(a_mat[r][j]) for r in range(n)) for j in range(n)]
    col_l1_b = [sum(abs(b_mat[r][j]) for r in range(n)) for j in range(n)]
    zero_c = all(v == 0 for v in c_vec)
    active_b_funded = all(fw[j] > 0 for j in range(n) if col_l1_b[j] > 0)
    if max(col_l1_a, default=Fraction(0)) <= Fraction(1) and zero_c and active_b_funded:
        kappa = max(
            (col_l1_b[j] / fw[j] for j in range(n) if col_l1_b[j] > 0),
            default=Fraction(0),
        )
        max_l1 = sum(abs(v) for v in w0) + kappa * b_tot
        monotone_nonneg = (
            all(w0[i] >= 0 for i in range(n))
            and u_lo >= 0
            and all(a_mat[i][j] >= 0 for i in range(n) for j in range(n))
            and all(b_mat[i][j] >= 0 for i in range(n) for j in range(n))
        )
        reach_lo = Fraction(0) if monotone_nonneg else -max_l1
        reach_hi = max_l1
        hi_ok = all(reach_hi <= w_hi for i in range(n) if uses_hi[i])
        lo_ok = all(reach_lo >= w_lo for i in range(n) if uses_lo[i])
        if hi_ok and lo_ok and reach_lo >= w_lo and reach_hi <= w_hi:
            return {
                "enclosed": True,
                "method": "exact_l1_contraction_enclosure",
                "reachable_interval": [str(reach_lo), str(reach_hi)],
                "compute_cost": spent,
            }

    # Criterion 2: Finite-horizon exact interval propagation when step_cost > 0 caps horizon K <= floor(B / step_cost)
    if step_cost > 0:
        k_max = int(b_tot // step_cost)
        if k_max <= 64:
            ok_k, sp_k = _spend_budget(budget, max(1, k_max) * 2 * n * n)
            spent += sp_k
            if not ok_k:
                return {"enclosed": False, "reason": "budget_exhausted", "compute_cost": spent}
            cur_lo = list(w0)
            cur_hi = list(w0)
            all_inside = True
            for step_idx in range(k_max):
                rem_fuel = b_tot - step_idx * step_cost
                u_eff_lo = [
                    max(u_lo, -(rem_fuel / fw[j])) if fw[j] > 0 else u_lo for j in range(n)
                ]
                u_eff_hi = [
                    min(u_hi, rem_fuel / fw[j]) if fw[j] > 0 else u_hi for j in range(n)
                ]
                nxt_lo: list[Fraction] = []
                nxt_hi: list[Fraction] = []
                for i in range(n):
                    lo_i = c_vec[i]
                    hi_i = c_vec[i]
                    for j in range(n):
                        aij = a_mat[i][j]
                        lo_i += aij * (cur_lo[j] if aij >= 0 else cur_hi[j])
                        hi_i += aij * (cur_hi[j] if aij >= 0 else cur_lo[j])
                        bij = b_mat[i][j]
                        lo_i += bij * (u_eff_lo[j] if bij >= 0 else u_eff_hi[j])
                        hi_i += bij * (u_eff_hi[j] if bij >= 0 else u_eff_lo[j])
                    nxt_lo.append(lo_i)
                    nxt_hi.append(hi_i)
                cur_lo, cur_hi = nxt_lo, nxt_hi
                if any(cur_lo[i] < w_lo or cur_hi[i] > w_hi for i in range(n)):
                    all_inside = False
                    break
            if all_inside:
                return {
                    "enclosed": True,
                    "method": f"finite_horizon_interval_propagation_K{k_max}",
                    "compute_cost": spent,
                }

    return {
        "enclosed": False,
        "reason": "analysis_region_not_proved_to_enclose_budget_reachable_trajectories",
        "compute_cost": spent,
    }


# ---------------------------------------------------------------------------
# 1. Exact Rational & SymPy QQ Proof Checker + Draft Lean 4 Exporter
# ---------------------------------------------------------------------------


class SymbolicProofKernel:
    """Exact ``QQ`` polynomial & rational Farkas certificate verifier (trusted base: Python Fraction + SymPy QQ)."""

    @staticmethod
    def verify_emergent_invariant_certificate(cert: dict, *, budget: Any | None = None) -> dict:
        """Reconstruct symbolic transitions on declared finite arities and verify ``Delta I == S_exo`` in ``QQ``."""
        if cert.get("kind") != "emergent_invariant_certificate":
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "wrong_certificate_kind",
                "compute_cost": 0,
            }

        mech = cert.get("system_class")
        params = cert.get("exact_parameters") or {}
        inv_weights = cert.get("invariant_weights") or {}
        exo_weights = cert.get("exogenous_input_weights") or {}
        test_arities = list(cert.get("verified_arities") or cert.get("verified_on_finite_arities") or [3, 5])
        if not test_arities or any(int(a) < 2 for a in test_arities):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "invalid_test_arities",
                "compute_cost": 0,
            }

        total_spent = 0
        checked_arities: list[int] = []
        formal_obligations: list[str] = []

        for raw_n in test_arities:
            n = int(raw_n)
            ok_b, sp_u = _spend_budget(budget, 12 * n * n + 20)
            total_spent += sp_u
            if not ok_b:
                return {
                    "verified": False,
                    "status": "unknown",
                    "reason": "budget_exhausted",
                    "compute_cost": total_spent,
                }

            try:
                if mech == "coupled_state_reservoir_pomdp":
                    alpha = _to_sympy_rat(params["alpha"])
                    eta = _to_sympy_rat(params["eta"])
                    c_u = _to_sympy_rat(params.get("c_u", 1))
                    gamma = _to_sympy_rat(params.get("gamma", Fraction(1, 5)))
                    if alpha == 1:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "singular_unit_pole",
                            "compute_cost": total_spent,
                        }

                    qs = sp.symbols(f"q0:{n}")
                    zs = sp.symbols(f"z0:{n}")
                    us = sp.symbols(f"u0:{n}")
                    edges = [
                        [sp.Symbol(f"e_{i}_{j}") if i != j else sp.Integer(0) for j in range(n)]
                        for i in range(n)
                    ]
                    phi = [
                        sum(gamma * edges[j][i] * qs[j] for j in range(n) if j != i)
                        - sum(gamma * edges[i][j] * qs[i] for j in range(n) if j != i)
                        for i in range(n)
                    ]
                    q_next = [qs[i] + us[i] + phi[i] + eta * zs[i] for i in range(n)]
                    z_next = [alpha * zs[i] + c_u * us[i] for i in range(n)]

                    w_q = _to_sympy_rat(inv_weights.get("sum_q", 0))
                    w_z = _to_sympy_rat(inv_weights.get("sum_z", 0))
                    if w_q == 0 and w_z == 0:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "trivial_zero_invariant",
                            "compute_cost": total_spent,
                        }

                    inv_before = w_q * sum(qs) + w_z * sum(zs)
                    inv_after = w_q * sum(q_next) + w_z * sum(z_next)
                    exo_drive = _to_sympy_rat(exo_weights.get("sum_u", 0)) * sum(us)

                    all_syms = [
                        *qs,
                        *zs,
                        *us,
                        *(edges[i][j] for i in range(n) for j in range(n) if i != j),
                    ]
                    diff_poly = sp.Poly(
                        sp.expand(inv_after - inv_before - exo_drive), *all_syms, domain="QQ"
                    )
                    if not diff_poly.is_zero:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": f"nonzero_polynomial_residual_at_N_{n}",
                            "compute_cost": total_spent,
                        }
                    checked_arities.append(n)
                    formal_obligations = ["pairwise_flux_cancellation_on_tested_arities", "alpha != 1"]

                elif mech == "two_species_reaction_transport":
                    nu = _to_sympy_rat(params["stoichiometric_ratio"])
                    kf = _to_sympy_rat(params["k_forward"])
                    kb = _to_sympy_rat(params["k_backward"])
                    ga = _to_sympy_rat(params.get("gamma_a", Fraction(1, 5)))
                    gb = _to_sympy_rat(params.get("gamma_b", Fraction(1, 7)))

                    a_s = sp.symbols(f"a0:{n}")
                    b_s = sp.symbols(f"b0:{n}")
                    ua_s = sp.symbols(f"ua0:{n}")
                    ub_s = sp.symbols(f"ub0:{n}")
                    edges = [
                        [sp.Symbol(f"e_{i}_{j}") if i != j else sp.Integer(0) for j in range(n)]
                        for i in range(n)
                    ]
                    phi_a = [
                        sum(ga * edges[j][i] * a_s[j] for j in range(n) if j != i)
                        - sum(ga * edges[i][j] * a_s[i] for j in range(n) if j != i)
                        for i in range(n)
                    ]
                    phi_b = [
                        sum(gb * edges[j][i] * b_s[j] for j in range(n) if j != i)
                        - sum(gb * edges[i][j] * b_s[i] for j in range(n) if j != i)
                        for i in range(n)
                    ]
                    rxn = [kf * a_s[i] - kb * b_s[i] for i in range(n)]
                    a_next = [a_s[i] + ua_s[i] + phi_a[i] - rxn[i] for i in range(n)]
                    b_next = [b_s[i] + ub_s[i] + phi_b[i] + nu * rxn[i] for i in range(n)]

                    w_a = _to_sympy_rat(inv_weights.get("sum_a", 0))
                    w_b = _to_sympy_rat(inv_weights.get("sum_b", 0))
                    if w_a == 0 and w_b == 0:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "trivial_zero_invariant",
                            "compute_cost": total_spent,
                        }

                    inv_before = w_a * sum(a_s) + w_b * sum(b_s)
                    inv_after = w_a * sum(a_next) + w_b * sum(b_next)
                    exo_drive = (
                        _to_sympy_rat(exo_weights.get("sum_ua", 0)) * sum(ua_s)
                        + _to_sympy_rat(exo_weights.get("sum_ub", 0)) * sum(ub_s)
                    )
                    all_syms = [
                        *a_s,
                        *b_s,
                        *ua_s,
                        *ub_s,
                        *(edges[i][j] for i in range(n) for j in range(n) if i != j),
                    ]
                    diff_poly = sp.Poly(
                        sp.expand(inv_after - inv_before - exo_drive), *all_syms, domain="QQ"
                    )
                    if not diff_poly.is_zero:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": f"nonzero_stoichiometric_residual_at_N_{n}",
                            "compute_cost": total_spent,
                        }
                    checked_arities.append(n)
                    formal_obligations = [
                        "species_pairwise_flux_cancellation_on_tested_arities",
                        "stoichiometric_reaction_nullspace",
                    ]

                elif mech in {"manifold_neumann_conservation", "manifold_dirichlet_gauss_green"}:
                    m_sites = max(4, n)
                    d_rat = _to_sympy_rat(params["diffusion"])
                    adv_rat = _to_sympy_rat(params.get("advection", 0))
                    qs = sp.symbols(f"q0:{m_sites}")
                    us = sp.symbols(f"u0:{m_sites}")
                    is_dir = mech == "manifold_dirichlet_gauss_green"
                    q_next = []
                    for k in range(m_sites):
                        if k == 0:
                            lap = (qs[1] - qs[0]) if not is_dir else (qs[1] - 2 * qs[0])
                            adv_t = qs[0]
                        elif k == m_sites - 1:
                            lap = (
                                (qs[m_sites - 2] - qs[m_sites - 1])
                                if not is_dir
                                else (qs[m_sites - 2] - 2 * qs[m_sites - 1])
                            )
                            adv_t = -qs[m_sites - 2]
                        else:
                            lap = qs[k - 1] - 2 * qs[k] + qs[k + 1]
                            adv_t = qs[k] - qs[k - 1]
                        q_next.append(qs[k] + us[k] + d_rat * lap - adv_rat * adv_t)

                    w_q = _to_sympy_rat(inv_weights.get("sum_q", 0))
                    if w_q == 0:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "trivial_zero_invariant",
                            "compute_cost": total_spent,
                        }
                    expected_rhs = _to_sympy_rat(exo_weights.get("sum_u", 0)) * sum(us) + _to_sympy_rat(
                        exo_weights.get("boundary_q", 0)
                    ) * (qs[0] + qs[m_sites - 1])
                    diff_poly = sp.Poly(
                        sp.expand(w_q * (sum(q_next) - sum(qs)) - expected_rhs),
                        *qs,
                        *us,
                        domain="QQ",
                    )
                    if not diff_poly.is_zero:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": f"manifold_balance_failed_at_N_{m_sites}",
                            "compute_cost": total_spent,
                        }
                    checked_arities.append(m_sites)
                    formal_obligations = [
                        f"discrete_1d_stencil_balance_boundary={'dirichlet' if is_dir else 'neumann'}"
                    ]

                elif mech == "directional_moment_monotonicity":
                    gamma_r = _to_sympy_rat(params["gamma"])
                    if gamma_r <= 0:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "nonpositive_coupling",
                            "compute_cost": total_spent,
                        }
                    dom_ob = cert.get("domain_obligations")
                    if not isinstance(dom_ob, dict):
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "missing_explicit_domain_obligations_for_directional_monotonicity",
                            "compute_cost": total_spent,
                        }
                    if dom_ob.get("donor_state_cone") != "nonnegative":
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "donor_state_cone_must_be_nonnegative",
                            "compute_cost": total_spent,
                        }
                    expected_dir = dom_ob.get("direction", "right")
                    expected_ineq = dom_ob.get("moment_change_sign", "nonnegative")
                    if (expected_dir == "right" and expected_ineq != "nonnegative") or (
                        expected_dir == "left" and expected_ineq != "nonpositive"
                    ) or expected_dir not in {"right", "left"}:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "incompatible_direction_and_moment_sign_obligation",
                            "compute_cost": total_spent,
                        }
                    if dom_ob.get("active_edge_displacement_sign") != (
                        "x_dst_minus_x_src_nonnegative"
                        if expected_dir == "right"
                        else "x_src_minus_x_dst_nonnegative"
                    ):
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": "missing_active_edge_displacement_sign_obligation",
                            "compute_cost": total_spent,
                        }

                    # If concrete positions/edges are also provided on cert, verify every active edge obeys direction!
                    if "positions" in dom_ob and "active_edges" in dom_ob:
                        pos_r = [_to_frac(v) for v in dom_ob["positions"]]
                        sgn_d = Fraction(1 if expected_dir == "right" else -1)
                        for src_i, dst_j in dom_ob["active_edges"]:
                            if sgn_d * (pos_r[int(dst_j)] - pos_r[int(src_i)]) < 0:
                                return {
                                    "verified": False,
                                    "status": "invalid_certificate",
                                    "reason": f"concrete_active_edge_{src_i}_{dst_j}_violates_direction_{expected_dir}",
                                    "compute_cost": total_spent,
                                }

                    qs = sp.symbols(f"q0:{n}")
                    xs = sp.symbols(f"x0:{n}")
                    edges = [
                        [sp.Symbol(f"e_{i}_{j}") if i != j else sp.Integer(0) for j in range(n)]
                        for i in range(n)
                    ]
                    phi = [
                        sum(gamma_r * edges[j][i] * qs[j] for j in range(n) if j != i)
                        - sum(gamma_r * edges[i][j] * qs[i] for j in range(n) if j != i)
                        for i in range(n)
                    ]
                    q_next = [qs[i] + phi[i] for i in range(n)]
                    m1_before = sum(xs[i] * qs[i] for i in range(n))
                    m1_after = sum(xs[i] * q_next[i] for i in range(n))
                    sos_rhs = sum(
                        gamma_r * edges[i][j] * (xs[j] - xs[i]) * qs[i]
                        for i in range(n)
                        for j in range(n)
                        if i != j
                    )
                    all_syms = [
                        *qs,
                        *xs,
                        *(edges[i][j] for i in range(n) for j in range(n) if i != j),
                    ]
                    diff_poly = sp.Poly(
                        sp.expand((m1_after - m1_before) - sos_rhs), *all_syms, domain="QQ"
                    )
                    if not diff_poly.is_zero:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": f"moment_identity_failed_at_N_{n}",
                            "compute_cost": total_spent,
                        }

                    # Cone-positivity check over nonnegative cone generators (q_i >= 0, e_{ij} >= 0, gap_{ij} >= 0)
                    gaps = [
                        [sp.Symbol(f"gap_{i}_{j}") if i != j else sp.Integer(0) for j in range(n)]
                        for i in range(n)
                    ]
                    dir_sign = sp.Integer(1 if expected_dir == "right" else -1)
                    cone_expr = sum(
                        dir_sign * gamma_r * edges[i][j] * (dir_sign * gaps[i][j]) * qs[i]
                        for i in range(n)
                        for j in range(n)
                        if i != j
                    )
                    # Note: for expected_dir == 'right', x_j - x_i = +gap_{ij} >= 0 -> +gamma * e_{ij} * gap_{ij} * q_i >= 0;
                    # for expected_dir == 'left', x_j - x_i = -gap_{ij} <= 0 -> -(Delta M_1) = +gamma * e_{ij} * gap_{ij} * q_i >= 0.
                    signed_delta_in_cone = sum(
                        gamma_r * edges[i][j] * gaps[i][j] * qs[i]
                        for i in range(n)
                        for j in range(n)
                        if i != j
                    )
                    cone_syms = [
                        *qs,
                        *(edges[i][j] for i in range(n) for j in range(n) if i != j),
                        *(gaps[i][j] for i in range(n) for j in range(n) if i != j),
                    ]
                    cone_poly = sp.Poly(sp.expand(signed_delta_in_cone), *cone_syms, domain="QQ")
                    if any(coeff <= 0 for coeff in cone_poly.coeffs()) or cone_expr != signed_delta_in_cone:
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": f"cone_positivity_obligation_failed_at_N_{n}",
                            "compute_cost": total_spent,
                        }
                    checked_arities.append(n)
                    formal_obligations = [
                        "gamma > 0",
                        "forall i: q_i >= 0",
                        f"forall i!=j: e_{{i,j}} >= 0 and (e_{{i,j}} > 0 -> {'x_j - x_i >= 0' if expected_dir == 'right' else 'x_i - x_j >= 0'})",
                    ]
                else:
                    return {
                        "verified": False,
                        "status": "invalid_certificate",
                        "reason": f"unknown_system_class:{mech}",
                        "compute_cost": total_spent,
                    }
            except (KeyError, ValueError, TypeError) as exc:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"invalid_certificate_parameters:{exc}",
                    "compute_cost": total_spent,
                }

        return {
            "verified": True,
            "status": (
                "conditional_monotonicity_verified_on_finite_arities_under_cone_obligations"
                if mech == "directional_moment_monotonicity"
                else "polynomial_identity_verified_on_finite_arities"
            ),
            "kernel": "sympy_exact_QQ_polynomial_kernel",
            "trusted_computing_base": TRUSTED_CHECKER_BASE,
            "checked_arities": checked_arities,
            "verified_on_finite_arities": checked_arities,
            "all_arity_proved": False,
            "formal_obligations": formal_obligations,
            "zero_polynomial_residual": True,
            "compute_cost": total_spent,
        }

    @staticmethod
    def verify_solvability_certificate(
        cert: dict, model_spec: dict, task: dict, *, budget: Any | None = None
    ) -> dict:
        """Independently check either a Reachability Witness or an Impossibility Certificate in ``Fraction``."""
        sem = validate_control_task_semantics(model_spec, task)
        if not sem["valid"]:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": sem["reason"],
                "compute_cost": 0,
            }
        kind = cert.get("kind")
        if kind == "reachability_witness_certificate":
            return SymbolicProofKernel._verify_reachability_witness(cert, sem, budget=budget)
        if kind == "potential_lower_bound_certificate":
            return SymbolicProofKernel._verify_potential_lower_bound(cert, sem, budget=budget)
        if kind == "invariant_obstruction_certificate":
            return SymbolicProofKernel._verify_invariant_obstruction(cert, sem, budget=budget)
        return {
            "verified": False,
            "status": "invalid_certificate",
            "reason": f"unknown_certificate_kind:{kind}",
            "compute_cost": 0,
        }

    @staticmethod
    def _exact_linear_dynamics(
        model_spec: dict,
    ) -> tuple[list[list[Fraction]], list[list[Fraction]], list[Fraction]]:
        """Build exact rational ``(A_mat, B_mat, c_vec)`` for ``w' = A w + B u + c`` from ``model_spec``."""
        n = int(model_spec["objects"])
        a_mat = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
        b_mat = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
        c_vec = [Fraction(0) for _ in range(n)]

        positions = [_to_frac(v) for v in model_spec.get("positions", [0] * n)]
        gamma = _to_frac(model_spec.get("coupling", 0))
        radius = _to_frac(model_spec.get("radius", 1))
        direction = model_spec.get("direction", "right")
        sink_rate = _to_frac(model_spec.get("sink_rate", 0))

        if gamma != 0:
            if direction in {"right", "left"}:
                sgn = Fraction(1 if direction == "right" else -1)
                for src in range(n):
                    for dst in range(n):
                        if src != dst and Fraction(0) < sgn * (positions[dst] - positions[src]) < radius:
                            a_mat[src][src] -= gamma
                            a_mat[dst][src] += gamma
            else:
                for i in range(n):
                    for j in range(n):
                        if i != j and abs(positions[j] - positions[i]) < radius:
                            a_mat[i][i] -= gamma
                            a_mat[i][j] += gamma

        if sink_rate != 0:
            for i in range(n):
                a_mat[i][i] -= sink_rate

        port_gain = _to_frac(model_spec.get("boundary_port_gain", 0))
        port_thresh = _to_frac(model_spec.get("boundary_port_threshold", Fraction(3, 2)))
        if port_gain != 0:
            for i in range(n):
                if abs(positions[i]) >= port_thresh:
                    b_mat[i][i] += port_gain

        return a_mat, b_mat, c_vec

    @staticmethod
    def _verify_reachability_witness(
        cert: dict, sem: dict[str, Any], *, budget: Any | None = None
    ) -> dict:
        n = sem["n"]
        a_mat, b_mat, c_vec = sem["a_mat"], sem["b_mat"], sem["c_vec"]
        w_cur = list(sem["w0"])
        u_lo, u_hi = sem["u_lo"], sem["u_hi"]
        w_lo, w_hi = sem["w_lo"], sem["w_hi"]
        enforce_state_box = sem["state_bounds_mode"] == "physical_legality"
        budget_limit = sem["budget"]
        step_cost = sem["step_cost"]
        fuel_weights = sem["fuel_weights"]

        actions = cert.get("action_sequence") or ()
        if not actions:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "empty_action_sequence",
                "compute_cost": 0,
            }

        ok_b, spent = _spend_budget(budget, len(actions) * (2 * n * n + 4 * n) + len(sem["a_goal"]) * n)
        if not ok_b:
            return {
                "verified": False,
                "status": "unknown",
                "reason": "budget_exhausted",
                "compute_cost": 0,
            }

        total_cost = Fraction(0)
        trace_exact: list[list[str]] = [[str(v) for v in w_cur]]
        for step_idx, a_raw in enumerate(actions):
            if not isinstance(a_raw, (list, tuple)) or len(a_raw) != n:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"invalid_action_dim_at_{step_idx}",
                    "compute_cost": spent,
                }
            try:
                u_vec = [_to_frac(v) for v in a_raw]
            except ValueError as exc:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"invalid_action_value_at_{step_idx}:{exc}",
                    "compute_cost": spent,
                }
            for i, u_val in enumerate(u_vec):
                if not (u_lo <= u_val <= u_hi):
                    return {
                        "verified": False,
                        "status": "invalid_certificate",
                        "reason": f"control_bound_violated_at_step_{step_idx}_obj_{i}",
                        "compute_cost": spent,
                    }
            total_cost += step_cost + sum(fuel_weights[i] * abs(u_vec[i]) for i in range(n))
            w_nxt = [
                sum(a_mat[i][j] * w_cur[j] for j in range(n))
                + sum(b_mat[i][j] * u_vec[j] for j in range(n))
                + c_vec[i]
                for i in range(n)
            ]
            if enforce_state_box and w_lo is not None and w_hi is not None:
                for i, w_val in enumerate(w_nxt):
                    if not (w_lo <= w_val <= w_hi):
                        return {
                            "verified": False,
                            "status": "invalid_certificate",
                            "reason": f"state_bound_violated_at_step_{step_idx}_obj_{i}",
                            "violating_state": [str(v) for v in w_nxt],
                            "state_bounds": [str(w_lo), str(w_hi)],
                            "state_bounds_mode": sem["state_bounds_mode"],
                            "compute_cost": spent,
                        }
            w_cur = w_nxt
            trace_exact.append([str(v) for v in w_cur])

        if total_cost > budget_limit:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "budget_exceeded",
                "exact_cost": float(total_cost),
                "budget": float(budget_limit),
                "compute_cost": spent,
            }

        for r_idx, (g_row, g_rhs) in enumerate(zip(sem["a_goal"], sem["b_goal"])):
            lhs = sum(g_row[j] * w_cur[j] for j in range(n))
            if lhs > g_rhs:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"goal_inequality_{r_idx}_violated",
                    "lhs": float(lhs),
                    "rhs": float(g_rhs),
                    "compute_cost": spent,
                }

        return {
            "verified": True,
            "status": "witness_verified_in_declared_model",
            "kernel": "exact_rational_trace_verifier",
            "trusted_computing_base": TRUSTED_CHECKER_BASE,
            "model_origin": sem["model_origin"],
            "state_bounds_mode": sem["state_bounds_mode"],
            "rational_model": sem["rational_model"],
            "exact_total_cost": float(total_cost),
            "exact_total_cost_rational": str(total_cost),
            "steps": len(actions),
            "trace_steps": len(trace_exact),
            "compute_cost": spent,
        }

    @staticmethod
    def _verify_potential_lower_bound(
        cert: dict, sem: dict[str, Any], *, budget: Any | None = None
    ) -> dict:
        """Verify AWM Eq. (3) Potential Certificate with unified state-domain soundness."""
        n = sem["n"]
        a_mat, b_mat, c_vec = sem["a_mat"], sem["b_mat"], sem["c_vec"]
        u_lo, u_hi = sem["u_lo"], sem["u_hi"]
        w_lo, w_hi = sem["w_lo"], sem["w_hi"]
        sb_mode = sem["state_bounds_mode"]
        budget_limit = sem["budget"]
        step_cost = sem["step_cost"]
        fuel_weights = sem["fuel_weights"]
        a_goal, b_goal = sem["a_goal"], sem["b_goal"]

        ok_b, spent = _spend_budget(budget, 4 * n * n + 10 * n + len(a_goal) * n)
        if not ok_b:
            return {
                "verified": False,
                "status": "unknown",
                "reason": "budget_exhausted",
                "compute_cost": 0,
            }

        try:
            alpha_0 = _to_frac(cert["alpha_0"])
            alpha = [_to_frac(v) for v in cert["alpha"]]
            mu_step = [_to_frac(v) for v in cert.get("farkas_mu_step", ())]
            mu_goal = [_to_frac(v) for v in cert.get("farkas_mu_goal", ())]
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": f"invalid_potential_certificate_fields:{exc}",
                "compute_cost": spent,
            }

        if len(alpha) != n:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "alpha_dim_mismatch",
                "compute_cost": spent,
            }
        if len(mu_step) != 6 * n:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "invalid_mu_step_length",
                "compute_cost": spent,
            }
        if any(m < 0 for m in mu_step):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "negative_farkas_step_multiplier",
                "compute_cost": spent,
            }

        uses_state_bounds = any(mu_step[i] > 0 or mu_step[n + i] > 0 for i in range(n))
        if sb_mode == "unbounded" and uses_state_bounds:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "nonzero_state_bound_multiplier_without_declared_state_bounds",
                "compute_cost": spent,
            }

        for i in range(n):
            grad_w_i = alpha[i] - sum(a_mat[r][i] * alpha[r] for r in range(n))
            farkas_w_i = mu_step[i] - mu_step[n + i]
            if grad_w_i != farkas_w_i:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"farkas_step_w_gradient_mismatch_at_{i}",
                    "lhs": float(grad_w_i),
                    "rhs": float(farkas_w_i),
                    "compute_cost": spent,
                }

        for i in range(n):
            grad_u_i = -sum(b_mat[r][i] * alpha[r] for r in range(n))
            farkas_u_i = (
                mu_step[2 * n + i]
                - mu_step[3 * n + i]
                + mu_step[4 * n + i]
                - mu_step[5 * n + i]
            )
            if grad_u_i != farkas_u_i:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"farkas_step_u_gradient_mismatch_at_{i}",
                    "lhs": float(grad_u_i),
                    "rhs": float(farkas_u_i),
                    "compute_cost": spent,
                }

        for i in range(n):
            farkas_s_i = -mu_step[4 * n + i] - mu_step[5 * n + i]
            if -fuel_weights[i] > farkas_s_i:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"farkas_step_s_dual_exceeds_fuel_weight_at_{i}",
                    "compute_cost": spent,
                }

        eff_w_hi = w_hi if w_hi is not None else Fraction(0)
        eff_w_lo = w_lo if w_lo is not None else Fraction(0)
        const_step = -sum(alpha[i] * c_vec[i] for i in range(n)) - step_cost
        farkas_step_bound = sum(
            mu_step[i] * eff_w_hi
            + mu_step[n + i] * (-eff_w_lo)
            + mu_step[2 * n + i] * u_hi
            + mu_step[3 * n + i] * (-u_lo)
            for i in range(n)
        )
        if const_step + farkas_step_bound > Fraction(0):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "step_inductive_inequality_not_nonpositive",
                "upper_bound": float(const_step + farkas_step_bound),
                "compute_cost": spent,
            }

        if len(mu_goal) != len(a_goal) or any(m < 0 for m in mu_goal):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "invalid_or_negative_mu_goal",
                "compute_cost": spent,
            }

        for j in range(n):
            col_dot = sum(a_goal[r][j] * mu_goal[r] for r in range(len(a_goal)))
            if col_dot != alpha[j]:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"farkas_goal_gradient_mismatch_at_{j}",
                    "compute_cost": spent,
                }
        goal_upper = alpha_0 + sum(b_goal[r] * mu_goal[r] for r in range(len(b_goal)))
        if goal_upper > Fraction(0):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "goal_potential_positive",
                "goal_upper": float(goal_upper),
                "compute_cost": spent,
            }

        w0 = sem["w0"]
        v_w0 = alpha_0 + sum(alpha[i] * w0[i] for i in range(n))
        if v_w0 <= budget_limit:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "initial_potential_does_not_exceed_budget",
                "v_w0": float(v_w0),
                "budget": float(budget_limit),
                "compute_cost": spent,
            }

        # Check analysis-region enclosure obligation when state_bounds is NOT a physical legality rule!
        enclosure_info = None
        if sb_mode == "analysis_region":
            enclosure_info = verify_reachable_box_enclosure(sem, mu_step, budget=budget)
            spent += int(enclosure_info.get("compute_cost", 0))
            if not enclosure_info["enclosed"]:
                return {
                    "verified": False,
                    "conditional_box_farkas_valid": True,
                    "enclosure_verified": False,
                    "status": "conditional_impossibility_domain_unverified",
                    "reason": enclosure_info["reason"],
                    "kernel": "exact_rational_farkas_potential_kernel",
                    "trusted_computing_base": TRUSTED_CHECKER_BASE,
                    "model_origin": sem["model_origin"],
                    "state_bounds_mode": sb_mode,
                    "rational_model": sem["rational_model"],
                    "conditional_min_budget": float(v_w0),
                    "conditional_min_budget_rational": str(v_w0),
                    "budget": float(budget_limit),
                    "compute_cost": spent,
                }

        formal_obs = [
            "step_farkas_inequality_verified",
            "goal_farkas_inequality_verified",
            "initial_potential_strictly_exceeds_budget",
        ]
        if uses_state_bounds:
            formal_obs.append(
                "state_bounds_enforced_as_physical_legality"
                if sb_mode == "physical_legality"
                else f"analysis_region_enclosure:{enclosure_info['method']}"
            )
        else:
            formal_obs.append("global_R^N_validity_zero_state_multipliers")

        return {
            "verified": True,
            "status": "impossibility_verified_in_declared_model_with_domain_obligations",
            "kernel": "exact_rational_farkas_potential_kernel",
            "trusted_computing_base": TRUSTED_CHECKER_BASE,
            "model_origin": sem["model_origin"],
            "state_bounds_mode": sb_mode,
            "uses_state_bounds": uses_state_bounds,
            "enclosure_verification": enclosure_info,
            "formal_obligations": formal_obs,
            "environment_coverage_verified": False,
            "rational_model": sem["rational_model"],
            "certified_min_budget": float(v_w0),
            "certified_min_budget_rational": str(v_w0),
            "budget": float(budget_limit),
            "strict_gap": float(v_w0 - budget_limit),
            "compute_cost": spent,
        }

    @staticmethod
    def _verify_invariant_obstruction(
        cert: dict, sem: dict[str, Any], *, budget: Any | None = None
    ) -> dict:
        n = sem["n"]
        a_mat, b_mat, c_vec = sem["a_mat"], sem["b_mat"], sem["c_vec"]
        fuel_weights = sem["fuel_weights"]
        a_goal, b_goal = sem["a_goal"], sem["b_goal"]

        ok_b, spent = _spend_budget(budget, 3 * n * n + 4 * n)
        if not ok_b:
            return {
                "verified": False,
                "status": "unknown",
                "reason": "budget_exhausted",
                "compute_cost": 0,
            }

        try:
            inv_vec = [_to_frac(v) for v in cert["invariant_vector"]]
            y_goal = [_to_frac(v) for v in cert["farkas_y_goal"]]
            sign = Fraction(int(cert.get("obstruction_sign", 1)))
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": f"invalid_invariant_obstruction_fields:{exc}",
                "compute_cost": spent,
            }

        if len(inv_vec) != n or all(v == 0 for v in inv_vec):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "invalid_invariant_vector",
                "compute_cost": spent,
            }

        for j in range(n):
            lhs = sum(a_mat[i][j] * inv_vec[i] for i in range(n))
            if lhs != inv_vec[j]:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"not_autonomous_invariant_at_{j}",
                    "compute_cost": spent,
                }
        if sum(inv_vec[i] * c_vec[i] for i in range(n)) != 0:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "nonzero_affine_drift",
                "compute_cost": spent,
            }

        delta_per_fuel = Fraction(0)
        for j in range(n):
            b_proj = abs(sum(b_mat[i][j] * inv_vec[i] for i in range(n)))
            if b_proj > 0:
                if fuel_weights[j] <= 0:
                    return {
                        "verified": False,
                        "status": "invalid_certificate",
                        "reason": "unbounded_zero_cost_control",
                        "compute_cost": spent,
                    }
                delta_per_fuel = max(delta_per_fuel, b_proj / fuel_weights[j])

        if len(y_goal) != len(a_goal) or any(y < 0 for y in y_goal):
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "invalid_y_goal",
                "compute_cost": spent,
            }

        for j in range(n):
            if sum(a_goal[r][j] * y_goal[r] for r in range(len(a_goal))) != -sign * inv_vec[j]:
                return {
                    "verified": False,
                    "status": "invalid_certificate",
                    "reason": f"goal_farkas_mismatch_at_{j}",
                    "compute_cost": spent,
                }

        goal_lb = -sum(b_goal[r] * y_goal[r] for r in range(len(b_goal)))
        w0 = sem["w0"]
        init_val = sign * sum(inv_vec[i] * w0[i] for i in range(n))
        budget_limit = sem["budget"]
        max_change = delta_per_fuel * budget_limit
        if goal_lb - init_val <= max_change:
            return {
                "verified": False,
                "status": "invalid_certificate",
                "reason": "invariant_gap_not_strictly_larger_than_budget_reach",
                "compute_cost": spent,
            }

        return {
            "verified": True,
            "status": "impossibility_verified_in_declared_model_with_domain_obligations",
            "kernel": "exact_rational_invariant_obstruction_kernel",
            "trusted_computing_base": TRUSTED_CHECKER_BASE,
            "model_origin": sem["model_origin"],
            "rational_model": sem["rational_model"],
            "initial_invariant": float(init_val),
            "goal_lower_bound": float(goal_lb),
            "max_reachable_shift": float(max_change),
            "compute_cost": spent,
        }


def emit_lean4_certificate(cert: dict) -> str:
    """Emit a draft Lean 4 stub for ``cert`` (explicitly marked ``draft_only``, NOT Lean-compiled!)."""
    prem_comment = ", ".join(cert.get("used_premise_ids", ())) or "core_transition_commitment"
    kind = cert.get("kind", "certificate")
    cid = str(cert.get("certificate_id", "hegel_v10_cert")).replace(":", "_").replace("-", "_")
    header = (
        f"-- Lean Status: draft_only (not Lean-compiled; proof_verified=false in Lean)\n"
        f"-- Formal check performed by: {TRUSTED_CHECKER_BASE}\n"
        f"-- Referenced Model Premises: [{prem_comment}]\n"
    )
    if kind == "emergent_invariant_certificate":
        sys_cls = cert.get("system_class", "unknown")
        if sys_cls == "coupled_state_reservoir_pomdp":
            return (
                header
                + f"namespace HegelV10DraftStubs\n\n"
                f"def {cid}_invariant (sum_q sum_z : Rat) : Rat :=\n"
                f"  ({cert['invariant_weights'].get('sum_q', '1')}) * sum_q + "
                f"({cert['invariant_weights'].get('sum_z', '0')}) * sum_z\n\n"
                f"-- Draft macro-moment step balance stub (not compiled via lean):\n"
                f"-- Verified on finite arities {cert.get('verified_arities')} in SymPy QQ.\n"
                f"end HegelV10DraftStubs\n"
            )
        return (
            header
            + f"namespace HegelV10DraftStubs\n\n"
            f"-- Draft invariant/monotonicity specification for system_class={sys_cls}\n"
            f"-- Formula: {cert.get('formula', '')}\n"
            f"-- Verified on finite arities {cert.get('verified_arities')} in SymPy QQ.\n"
            f"end HegelV10DraftStubs\n"
        )
    if kind == "potential_lower_bound_certificate":
        alpha_terms = " + ".join(
            f"({_to_frac(a)}) * w{i}" for i, a in enumerate(cert.get("alpha", ()))
        ) or "0"
        return (
            header
            + f"namespace HegelV10DraftStubs\n\n"
            f"def {cid}_V ({' '.join(f'w{i}' for i in range(len(cert.get('alpha', ()))))} : Rat) : Rat :=\n"
            f"  ({_to_frac(cert.get('alpha_0', 0))}) + {alpha_terms}\n\n"
            f"-- Draft definition only (no formal Lean proof compiled).\n"
            f"-- Checked in Python Fraction arithmetic: V(w_0) = {cert.get('certified_min_budget_rational')} vs B = {cert.get('budget')}.\n"
            f"end HegelV10DraftStubs\n"
        )
    return (
        header
        + f"namespace HegelV10DraftStubs\n"
        f"-- Draft reachability witness stub ({len(cert.get('action_sequence', ()))} steps); checked in Python Fraction arithmetic.\n"
        f"end HegelV10DraftStubs\n"
    )


# ---------------------------------------------------------------------------
# 2. Cross-Commitment Symbolic Flux Elimination & Emergent Invariant Synthesis
# ---------------------------------------------------------------------------


def synthesize_emergent_invariants(system_spec: dict, *, budget: Any | None = None) -> list[dict]:
    """Symbolically eliminate internal fluxes via ``sympy.Matrix.nullspace()`` over ``QQ``."""
    sys_class = system_spec["system_class"]
    model_origin = str(system_spec.get("model_origin", "supplied_model"))
    certs: list[dict] = []

    ok_b, null_spent = _spend_budget(budget, 40)
    if not ok_b:
        return []

    if sys_class == "coupled_state_reservoir_pomdp":
        alpha_r = _to_sympy_rat(system_spec["alpha"])
        eta_r = _to_sympy_rat(system_spec["eta"])
        cu_r = _to_sympy_rat(system_spec.get("c_u", 1))
        gamma_r = _to_sympy_rat(system_spec.get("gamma", Fraction(1, 5)))

        f_int = sp.Matrix([[eta_r], [alpha_r - 1]])
        null_vecs = f_int.T.nullspace()
        for idx, vec in enumerate(null_vecs):
            wq = sp.Rational(vec[0])
            wz = sp.Rational(vec[1])
            if wq != 0:
                wz = wz / wq
                wq = sp.Rational(1)
            exo_u = wq * sp.Rational(1) + wz * cu_r
            cert = {
                "certificate_id": f"emergent_inv:pomdp:{idx}",
                "kind": "emergent_invariant_certificate",
                "system_class": sys_class,
                "model_origin": model_origin,
                "declared_premises": [
                    "P_conservation_spatial_exchange",
                    "P_markov_state:transition_memory",
                ],
                "empirically_supported_premises": (
                    list(system_spec.get("empirically_supported_premises", ()))
                    if model_origin == "learned_model"
                    else []
                ),
                "used_premise_ids": [
                    "P_conservation_spatial_exchange",
                    "P_markov_state:transition_memory",
                ],
                "exact_parameters": {
                    "alpha": str(alpha_r),
                    "eta": str(eta_r),
                    "c_u": str(cu_r),
                    "gamma": str(gamma_r),
                },
                "invariant_weights": {"sum_q": str(wq), "sum_z": str(wz)},
                "exogenous_input_weights": {"sum_u": str(exo_u)},
                "verified_arities": [3, 5, 8],
                "verified_on_finite_arities": [3, 5, 8],
                "all_arity_proved": False,
                "lean_status": "draft_only",
                "lean_compiled": False,
                "proof_verified_by": TRUSTED_CHECKER_BASE,
                "formula": f"({wq}) * sum_i q_i + ({wz}) * sum_i z_i",
            }
            ver = SymbolicProofKernel.verify_emergent_invariant_certificate(cert, budget=budget)
            cert["compute_cost"] = null_spent + int(ver.get("compute_cost", 0))
            cert["kernel_verification"] = ver
            cert["lean4_module"] = emit_lean4_certificate(cert)
            if ver["verified"]:
                certs.append(cert)

    elif sys_class == "two_species_reaction_transport":
        nu_r = _to_sympy_rat(system_spec["stoichiometric_ratio"])
        kf_r = _to_sympy_rat(system_spec.get("k_forward", Fraction(1, 4)))
        kb_r = _to_sympy_rat(system_spec.get("k_backward", Fraction(1, 8)))
        f_int = sp.Matrix([[-sp.Rational(1)], [nu_r]])
        for idx, vec in enumerate(f_int.T.nullspace()):
            wa, wb = sp.Rational(vec[0]), sp.Rational(vec[1])
            if wb != 0:
                wa = wa / wb
                wb = sp.Rational(1)
            cert = {
                "certificate_id": f"emergent_inv:stoich:{idx}",
                "kind": "emergent_invariant_certificate",
                "system_class": sys_class,
                "model_origin": model_origin,
                "declared_premises": [
                    "P_species_a_spatial_conservation",
                    "P_species_b_spatial_conservation",
                    "P_reversible_reaction_stoichiometry",
                ],
                "empirically_supported_premises": (
                    list(system_spec.get("empirically_supported_premises", ()))
                    if model_origin == "learned_model"
                    else []
                ),
                "used_premise_ids": [
                    "P_species_a_spatial_conservation",
                    "P_species_b_spatial_conservation",
                    "P_reversible_reaction_stoichiometry",
                ],
                "exact_parameters": {
                    "stoichiometric_ratio": str(nu_r),
                    "k_forward": str(kf_r),
                    "k_backward": str(kb_r),
                },
                "invariant_weights": {"sum_a": str(wa), "sum_b": str(wb)},
                "exogenous_input_weights": {"sum_ua": str(wa), "sum_ub": str(wb)},
                "verified_arities": [3, 5, 7],
                "verified_on_finite_arities": [3, 5, 7],
                "all_arity_proved": False,
                "lean_status": "draft_only",
                "lean_compiled": False,
                "proof_verified_by": TRUSTED_CHECKER_BASE,
                "formula": f"({wa}) * sum_i a_i + ({wb}) * sum_i b_i",
            }
            ver = SymbolicProofKernel.verify_emergent_invariant_certificate(cert, budget=budget)
            cert["compute_cost"] = null_spent + int(ver.get("compute_cost", 0))
            cert["kernel_verification"] = ver
            cert["lean4_module"] = emit_lean4_certificate(cert)
            if ver["verified"]:
                certs.append(cert)

    elif sys_class in {"manifold_neumann_conservation", "manifold_dirichlet_gauss_green"}:
        d_r = _to_sympy_rat(system_spec["diffusion"])
        adv_r = _to_sympy_rat(system_spec.get("advection", 0))
        is_dir = sys_class == "manifold_dirichlet_gauss_green"
        cert = {
            "certificate_id": f"emergent_inv:{sys_class}",
            "kind": "emergent_invariant_certificate",
            "system_class": sys_class,
            "model_origin": model_origin,
            "declared_premises": [
                "P_continuum_topology",
                f"P_boundary_condition:{'dirichlet' if is_dir else 'neumann'}",
                "P_pde_stencil_conservation",
            ],
            "empirically_supported_premises": (
                list(system_spec.get("empirically_supported_premises", ()))
                if model_origin == "learned_model"
                else []
            ),
            "used_premise_ids": [
                "P_continuum_topology",
                f"P_boundary_condition:{'dirichlet' if is_dir else 'neumann'}",
                "P_pde_stencil_conservation",
            ],
            "exact_parameters": {"diffusion": str(d_r), "advection": str(adv_r)},
            "invariant_weights": {"sum_q": "1"},
            "exogenous_input_weights": {
                "sum_u": "1",
                "boundary_q": str(-d_r if is_dir else sp.Rational(0)),
            },
            "verified_arities": [4, 6, 9],
            "verified_on_finite_arities": [4, 6, 9],
            "all_arity_proved": False,
            "lean_status": "draft_only",
            "lean_compiled": False,
            "proof_verified_by": TRUSTED_CHECKER_BASE,
            "formula": (
                f"Delta(sum_k q_k) = sum_k u_k - ({d_r}) * (q_0 + q_{{N-1}})"
                if is_dir
                else "Delta(sum_k q_k) = sum_k u_k"
            ),
        }
        ver = SymbolicProofKernel.verify_emergent_invariant_certificate(cert, budget=budget)
        cert["compute_cost"] = null_spent + int(ver.get("compute_cost", 0))
        cert["kernel_verification"] = ver
        cert["lean4_module"] = emit_lean4_certificate(cert)
        if ver["verified"]:
            certs.append(cert)

    elif sys_class == "directional_moment_monotonicity":
        gamma_r = _to_sympy_rat(system_spec.get("gamma", Fraction(1, 4)))
        direction = str(system_spec.get("direction", "right"))
        sign_str = "nonnegative" if direction == "right" else "nonpositive"
        disp_ob = (
            "x_dst_minus_x_src_nonnegative"
            if direction == "right"
            else "x_src_minus_x_dst_nonnegative"
        )
        dom_ob = {
            "donor_state_cone": str(system_spec.get("donor_state_cone", "nonnegative")),
            "direction": direction,
            "moment_change_sign": sign_str,
            "active_edge_displacement_sign": disp_ob,
        }
        if "positions" in system_spec and "active_edges" in system_spec:
            dom_ob["positions"] = list(system_spec["positions"])
            dom_ob["active_edges"] = list(system_spec["active_edges"])
        cert = {
            "certificate_id": "emergent_inv:directional_moment",
            "kind": "emergent_invariant_certificate",
            "system_class": sys_class,
            "model_origin": model_origin,
            "declared_premises": [
                "P_conservation",
                f"P_interaction_direction:{direction}",
                "P_nonnegative_donor_state",
            ],
            "empirically_supported_premises": (
                list(system_spec.get("empirically_supported_premises", ()))
                if model_origin == "learned_model"
                else []
            ),
            "used_premise_ids": [
                "P_conservation",
                f"P_interaction_direction:{direction}",
                "P_nonnegative_donor_state",
            ],
            "domain_obligations": dom_ob,
            "exact_parameters": {"gamma": str(gamma_r)},
            "invariant_weights": {"first_spatial_moment_xq": "1"},
            "exogenous_input_weights": {"positive_cone_sum_of_products": str(gamma_r)},
            "verified_arities": [3, 5, 7],
            "verified_on_finite_arities": [3, 5, 7],
            "all_arity_proved": False,
            "lean_status": "draft_only",
            "lean_compiled": False,
            "proof_verified_by": TRUSTED_CHECKER_BASE,
            "formula": (
                f"Delta(sum_i x_i q_i) = ({gamma_r}) * sum_{{i!=j}} e^+_{{ij}} (x_j - x_i) q_i "
                f"{'>= 0' if direction == 'right' else '<= 0'} (under donor_state_cone=nonnegative)"
            ),
        }
        ver = SymbolicProofKernel.verify_emergent_invariant_certificate(cert, budget=budget)
        cert["compute_cost"] = null_spent + int(ver.get("compute_cost", 0))
        cert["kernel_verification"] = ver
        cert["lean4_module"] = emit_lean4_certificate(cert)
        if ver["verified"]:
            certs.append(cert)

    return certs


# ---------------------------------------------------------------------------
# 3. Budgeted Control Apparatus, Solvability Prover, Learned-Model Bridge & Repair
# ---------------------------------------------------------------------------


class BudgetedControlLab:
    """Interactive multi-object physical control apparatus with unified legality semantics."""

    def __init__(
        self,
        *,
        objects: int = 4,
        positions: tuple[float, ...] = (-1.8, -0.6, 0.2, 1.8),
        coupling: float = 0.25,
        radius: float = 1.0,
        direction: str = "right",
        sink_rate: float = 0.0,
        boundary_port_gain: float = 0.0,
        boundary_port_threshold: float = 1.5,
        control_bounds: tuple[float, float] = (-0.5, 0.5),
        state_bounds: tuple[float, float] | None = None,
        state_bounds_mode: str | None = None,
        step_cost: float = 0.0,
    ):
        if int(objects) < 1 or len(positions) != int(objects):
            raise ValueError("objects/positions dimension mismatch")
        if float(step_cost) < 0.0:
            raise ValueError("step_cost must be non-negative")
        if float(control_bounds[0]) > float(control_bounds[1]):
            raise ValueError("control_bounds inverted")
        if state_bounds is not None and float(state_bounds[0]) > float(state_bounds[1]):
            raise ValueError("state_bounds inverted")
        self.objects = int(objects)
        self.positions = tuple(float(x) for x in positions)
        self.coupling = float(coupling)
        self.radius = float(radius)
        self.direction = direction
        self.sink_rate = float(sink_rate)
        self.boundary_port_gain = float(boundary_port_gain)
        self.boundary_port_threshold = float(boundary_port_threshold)
        self.control_bounds = (float(control_bounds[0]), float(control_bounds[1]))
        self.state_bounds = (
            (float(state_bounds[0]), float(state_bounds[1])) if state_bounds is not None else None
        )
        self.state_bounds_mode = (
            (state_bounds_mode or "physical_legality") if state_bounds is not None else "unbounded"
        )
        self.step_cost = float(step_cost)
        self.probe_queries = 0

    def step(
        self,
        q_state: list[float] | tuple[float, ...],
        u_action: list[float] | tuple[float, ...],
        *,
        state_bounds: tuple[float, float] | None = None,
        state_bounds_mode: str | None = None,
    ) -> list[float]:
        lo, hi = self.control_bounds
        if len(q_state) != self.objects or len(u_action) != self.objects:
            raise ValueError("state/action dimension mismatch")
        if any(not math.isfinite(float(v)) for v in (*q_state, *u_action)):
            raise ValueError("non-finite state or action")
        if any(not (lo - 1e-12 <= float(u) <= hi + 1e-12) for u in u_action):
            raise ValueError("action outside control bounds")

        eff_sb = state_bounds if state_bounds is not None else self.state_bounds
        eff_mode = (
            state_bounds_mode
            if state_bounds_mode is not None
            else ("physical_legality" if state_bounds is not None else self.state_bounds_mode)
        )
        if eff_sb is not None and eff_mode == "physical_legality":
            s_lo, s_hi = float(eff_sb[0]), float(eff_sb[1])
            if any(not (s_lo - 1e-12 <= float(v) <= s_hi + 1e-12) for v in q_state):
                raise ValueError("initial/pre-step state outside physical state_bounds")

        q = [float(v) for v in q_state]
        u = [float(v) for v in u_action]
        flux = [0.0] * self.objects
        if self.direction in {"right", "left"}:
            sgn = 1.0 if self.direction == "right" else -1.0
            for src in range(self.objects):
                for dst in range(self.objects):
                    if src != dst and 0.0 < sgn * (self.positions[dst] - self.positions[src]) < self.radius:
                        tr = self.coupling * q[src]
                        flux[src] -= tr
                        flux[dst] += tr
        else:
            for i in range(self.objects):
                for j in range(self.objects):
                    if i != j and abs(self.positions[j] - self.positions[i]) < self.radius:
                        flux[i] += self.coupling * (q[j] - q[i])

        out = []
        for i in range(self.objects):
            eff_u = u[i] * (
                1.0 + self.boundary_port_gain
                if abs(self.positions[i]) >= self.boundary_port_threshold
                else 1.0
            )
            out.append(q[i] + eff_u + flux[i] - self.sink_rate * q[i])

        if eff_sb is not None and eff_mode == "physical_legality":
            s_lo, s_hi = float(eff_sb[0]), float(eff_sb[1])
            if any(not (s_lo - 1e-12 <= float(v) <= s_hi + 1e-12) for v in out):
                raise ValueError("post-step state outside physical state_bounds")
        return out

    def probe_transition(
        self,
        q_state: list[float],
        u_action: list[float],
        *,
        budget: Any | None = None,
    ) -> dict:
        ok_b, spent = _spend_budget(budget, 2 * self.objects * self.objects + 4)
        if not ok_b:
            raise RuntimeError("SearchBudget exhausted before probe_transition")
        self.probe_queries += 1
        after = self.step(q_state, u_action)
        cost = self.step_cost + sum(abs(float(v)) for v in u_action)
        return {
            "before_q": list(q_state),
            "u": list(u_action),
            "positions": list(self.positions),
            "after_q": after,
            "cost": float(cost),
            "compute_cost": spent,
        }

    def execute_trace(
        self,
        initial_state: list[float],
        actions: list[list[float]],
        task: dict,
        *,
        budget: Any | None = None,
    ) -> dict:
        model_proxy = {
            "objects": self.objects,
            "positions": list(self.positions),
            "coupling": self.coupling,
            "radius": self.radius,
            "direction": self.direction,
            "sink_rate": self.sink_rate,
            "boundary_port_gain": self.boundary_port_gain,
            "boundary_port_threshold": self.boundary_port_threshold,
        }
        task_merged = {**task, "initial_state": initial_state}
        sem = validate_control_task_semantics(
            model_proxy,
            task_merged,
            lab_state_bounds=self.state_bounds,
            lab_state_bounds_mode=self.state_bounds_mode if self.state_bounds is not None else None,
        )
        if not sem["valid"]:
            return {
                "valid_semantics": False,
                "reason": sem["reason"],
                "final_state": None,
                "total_cost": None,
                "within_budget": False,
                "state_bounds_respected": False,
                "goal_reached": False,
                "winnable_verified_on_lab": False,
            }

        ok_b, spent = _spend_budget(budget, max(1, len(actions)) * (2 * self.objects + 2))
        if not ok_b:
            return {
                "valid_semantics": True,
                "reason": "budget_exhausted",
                "final_state": None,
                "total_cost": None,
                "within_budget": False,
                "state_bounds_respected": False,
                "goal_reached": False,
                "winnable_verified_on_lab": False,
                "compute_cost": 0,
            }

        cur = [float(v) for v in initial_state]
        total_cost = 0.0
        fuel_w = [float(v) for v in sem["fuel_weights"]]
        eff_step_cost = float(sem["step_cost"])
        enforce_sb = sem["state_bounds_mode"] == "physical_legality"
        sb_tuple = (
            (float(sem["w_lo"]), float(sem["w_hi"]))
            if sem["w_lo"] is not None and sem["w_hi"] is not None
            else None
        )
        state_legality_ok = True
        legality_violation_step: int | None = None

        for step_idx, u_t in enumerate(actions):
            if len(u_t) != self.objects:
                return {
                    "valid_semantics": False,
                    "reason": f"action_dim_mismatch_at_{step_idx}",
                    "final_state": cur,
                    "total_cost": float(total_cost),
                    "within_budget": False,
                    "state_bounds_respected": False,
                    "goal_reached": False,
                    "winnable_verified_on_lab": False,
                    "compute_cost": spent,
                }
            total_cost += eff_step_cost + sum(
                fuel_w[i] * abs(float(u_t[i])) for i in range(self.objects)
            )
            try:
                cur = self.step(
                    cur,
                    u_t,
                    state_bounds=sb_tuple if enforce_sb else None,
                    state_bounds_mode="physical_legality" if enforce_sb else "unbounded",
                )
            except ValueError as exc:
                if "state_bounds" in str(exc):
                    state_legality_ok = False
                    legality_violation_step = step_idx
                    # Compute unconstrained post-step state for diagnostic transparency while rejecting legality!
                    cur = self.step(cur, u_t, state_bounds=None, state_bounds_mode="unbounded")
                else:
                    return {
                        "valid_semantics": False,
                        "reason": str(exc),
                        "final_state": cur,
                        "total_cost": float(total_cost),
                        "within_budget": False,
                        "state_bounds_respected": False,
                        "goal_reached": False,
                        "winnable_verified_on_lab": False,
                        "compute_cost": spent,
                    }

        a_g = sem["a_goal"]
        b_g = sem["b_goal"]
        goal_met = all(
            sum(float(a_g[r][j]) * cur[j] for j in range(self.objects)) <= float(b_g[r]) + 1e-9
            for r in range(len(a_g))
        )
        within_b = bool(total_cost <= float(sem["budget"]) + 1e-9)
        return {
            "valid_semantics": True,
            "state_bounds_mode": sem["state_bounds_mode"],
            "state_bounds_respected": state_legality_ok,
            "legality_violation_step": legality_violation_step,
            "final_state": cur,
            "total_cost": float(total_cost),
            "within_budget": within_b,
            "goal_reached": goal_met,
            "winnable_verified_on_lab": bool(state_legality_ok and goal_met and within_b),
            "compute_cost": spent,
        }


def prove_task_solvability(
    model_spec: dict,
    task: dict,
    *,
    max_horizon: int = 8,
    certificate_id: str = "solvability_cert",
    budget: Any | None = None,
) -> dict:
    """Prove either Reachability (witness ``a_bar``) or Impossibility (Farkas Potential) under unified domain semantics."""
    sem = validate_control_task_semantics(model_spec, task)
    if not sem["valid"]:
        return {
            "certificate_id": f"{certificate_id}:invalid",
            "kind": "invalid_certificate",
            "verdict": "invalid_certificate",
            "status": "invalid_certificate",
            "kernel_verification": {
                "verified": False,
                "status": "invalid_certificate",
                "reason": sem["reason"],
            },
            "compute_cost": 0,
        }

    n = sem["n"]
    a_frac, b_frac, c_frac = sem["a_mat"], sem["b_mat"], sem["c_vec"]
    a_np = np.array([[float(v) for v in row] for row in a_frac], dtype=float)
    b_np = np.array([[float(v) for v in row] for row in b_frac], dtype=float)
    c_np = np.array([float(v) for v in c_frac], dtype=float)

    w0 = np.array([float(v) for v in sem["w0"]], dtype=float)
    u_lo, u_hi = float(sem["u_lo"]), float(sem["u_hi"])
    budget_limit = float(sem["budget"])
    step_cost = float(sem["step_cost"])
    fuel_w = np.array([float(v) for v in sem["fuel_weights"]], dtype=float)
    a_goal = np.array([[float(v) for v in row] for row in sem["a_goal"]], dtype=float)
    b_goal = np.array([float(v) for v in sem["b_goal"]], dtype=float)
    sb_mode = sem["state_bounds_mode"]
    enforce_state_box = sb_mode == "physical_legality" and sem["w_lo"] is not None

    total_spent = 0

    # ------------------------------------------------------------------
    # Phase A: Search for Minimum-Cost Reachability Witness across K = 1..max_horizon
    # ------------------------------------------------------------------
    best_witness = None
    for k_steps in range(1, max_horizon + 1):
        ok_b, sp_lp = _spend_budget(budget, k_steps * (8 * n * n + 12 * n))
        total_spent += sp_lp
        if not ok_b:
            return {
                "certificate_id": f"{certificate_id}:budget_exhausted",
                "kind": "unproved",
                "verdict": "unknown",
                "status": "unknown",
                "kernel_verification": {
                    "verified": False,
                    "status": "unknown",
                    "reason": "budget_exhausted",
                },
                "compute_cost": total_spent,
            }

        a_pows = [np.eye(n)]
        for _ in range(k_steps):
            a_pows.append(a_pows[-1] @ a_np)
        w_free = a_pows[k_steps] @ w0 + sum(a_pows[k_steps - 1 - t] @ c_np for t in range(k_steps))

        num_u = k_steps * n
        c_obj = np.concatenate([np.zeros(num_u), np.tile(fuel_w, k_steps)])

        g_blocks = [a_goal @ a_pows[k_steps - 1 - t] @ b_np for t in range(k_steps)]
        row_goal = np.hstack([*g_blocks, np.zeros((len(b_goal), num_u))])
        rhs_goal = b_goal - (a_goal @ w_free)

        eye_u = np.eye(num_u)
        row_abs1 = np.hstack([eye_u, -eye_u])
        row_abs2 = np.hstack([-eye_u, -eye_u])
        ub_blocks = [row_goal, row_abs1, row_abs2]
        rhs_blocks = [rhs_goal, np.zeros(2 * num_u)]

        # Enforce intermediate & terminal state_bounds identically when physical_legality holds!
        if enforce_state_box:
            w_lo_f, w_hi_f = float(sem["w_lo"]), float(sem["w_hi"])
            for step_t in range(1, k_steps + 1):
                wf_t = a_pows[step_t] @ w0 + sum(
                    a_pows[step_t - 1 - s] @ c_np for s in range(step_t)
                )
                step_cols = []
                for s in range(k_steps):
                    if s < step_t:
                        step_cols.append(a_pows[step_t - 1 - s] @ b_np)
                    else:
                        step_cols.append(np.zeros((n, n)))
                dyn_mat_t = np.hstack([*step_cols, np.zeros((n, num_u))])
                ub_blocks.append(dyn_mat_t)
                rhs_blocks.append(np.full(n, w_hi_f) - wf_t)
                ub_blocks.append(-dyn_mat_t)
                rhs_blocks.append(wf_t - np.full(n, w_lo_f))

        a_ub = np.vstack(ub_blocks)
        b_ub = np.concatenate(rhs_blocks)

        bounds_lp = [(u_lo, u_hi)] * num_u + [(0.0, None)] * num_u
        res_lp = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds_lp, method="highs")
        if res_lp.success:
            min_cost_k = k_steps * step_cost + float(res_lp.fun)
            if min_cost_k <= budget_limit + 1e-11:
                u_sol = res_lp.x[:num_u].reshape((k_steps, n))
                rat_actions = [
                    [str(_to_frac(float(u_sol[t, i]), max_den=1_000_000)) for i in range(n)]
                    for t in range(k_steps)
                ]
                cand_cert = {
                    "certificate_id": f"{certificate_id}:witness_k{k_steps}",
                    "kind": "reachability_witness_certificate",
                    "verdict": "winnable",
                    "status": "witness_verified_in_declared_model",
                    "model_origin": sem["model_origin"],
                    "declared_premises": sem["declared_premises"],
                    "empirically_supported_premises": sem["empirically_supported_premises"],
                    "refuted_premises": sem["refuted_premises"],
                    "used_premise_ids": list(sem["declared_premises"]),
                    "action_sequence": rat_actions,
                    "horizon": k_steps,
                    "budget": budget_limit,
                    "state_bounds_mode": sb_mode,
                    "lean_status": "draft_only",
                    "lean_compiled": False,
                    "proof_verified_by": TRUSTED_CHECKER_BASE,
                }
                ver = SymbolicProofKernel.verify_solvability_certificate(
                    cand_cert, model_spec, task, budget=budget
                )
                total_spent += int(ver.get("compute_cost", 0))
                if ver["verified"]:
                    cand_cert["kernel_verification"] = ver
                    cand_cert["lean4_module"] = emit_lean4_certificate(cand_cert)
                    cand_cert["compute_cost"] = total_spent
                    best_witness = cand_cert
                    break

    if best_witness is not None:
        return best_witness

    # ------------------------------------------------------------------
    # Phase B: Synthesize Farkas Potential Certificate (respecting state_bounds_mode!)
    # ------------------------------------------------------------------
    ok_p, sp_pot = _spend_budget(budget, 16 * n * n + 24 * n)
    total_spent += sp_pot
    if not ok_p:
        return {
            "certificate_id": f"{certificate_id}:budget_exhausted",
            "kind": "unproved",
            "verdict": "unknown",
            "status": "unknown",
            "kernel_verification": {
                "verified": False,
                "status": "unknown",
                "reason": "budget_exhausted",
            },
            "compute_cost": total_spent,
        }

    has_box = sem["w_lo"] is not None and sem["w_hi"] is not None
    w_lo_f = float(sem["w_lo"]) if has_box else 0.0
    w_hi_f = float(sem["w_hi"]) if has_box else 0.0
    m_g = len(b_goal)

    idx_a0 = 0
    idx_alpha = slice(1, 1 + n)
    idx_mg = slice(1 + 7 * n, 1 + 7 * n + m_g)
    total_vars = 1 + 7 * n + m_g

    c_pot = np.zeros(total_vars)
    c_pot[idx_a0] = -1.0
    c_pot[idx_alpha] = -w0

    eq_rows = []
    eq_rhs = []
    i_minus_at = np.eye(n) - a_np.T
    for i in range(n):
        row = np.zeros(total_vars)
        row[idx_alpha] = i_minus_at[i]
        row[1 + n + i] = -1.0
        row[1 + 2 * n + i] = 1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)

    neg_bt = -b_np.T
    for i in range(n):
        row = np.zeros(total_vars)
        row[idx_alpha] = neg_bt[i]
        row[1 + 3 * n + i] = -1.0
        row[1 + 4 * n + i] = 1.0
        row[1 + 5 * n + i] = -1.0
        row[1 + 6 * n + i] = 1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)

    for j in range(n):
        row = np.zeros(total_vars)
        row[1 + j] = -1.0
        row[idx_mg] = a_goal[:, j]
        eq_rows.append(row)
        eq_rhs.append(0.0)

    ub_rows = []
    ub_rhs = []
    for i in range(n):
        row = np.zeros(total_vars)
        row[1 + 5 * n + i] = 1.0
        row[1 + 6 * n + i] = 1.0
        ub_rows.append(row)
        ub_rhs.append(fuel_w[i])

    row_sc = np.zeros(total_vars)
    row_sc[idx_alpha] = -c_np
    for i in range(n):
        row_sc[1 + n + i] = w_hi_f
        row_sc[1 + 2 * n + i] = -w_lo_f
        row_sc[1 + 3 * n + i] = u_hi
        row_sc[1 + 4 * n + i] = -u_lo
    ub_rows.append(row_sc)
    ub_rhs.append(step_cost)

    row_gc = np.zeros(total_vars)
    row_gc[idx_a0] = 1.0
    row_gc[idx_mg] = b_goal
    ub_rows.append(row_gc)
    ub_rhs.append(0.0)

    # When state_bounds is None ("unbounded"), lock mu_w_hi == 0 and mu_w_lo == 0 (never invent a box!).
    state_mult_bounds = [(0.0, None)] * (2 * n) if has_box else [(0.0, 0.0)] * (2 * n)
    bounds_pot = (
        [(None, None)] * (1 + n)
        + state_mult_bounds
        + [(0.0, None)] * (4 * n + m_g)
    )
    res_pot = linprog(
        c_pot,
        A_ub=np.array(ub_rows),
        b_ub=np.array(ub_rhs),
        A_eq=np.array(eq_rows),
        b_eq=np.array(eq_rhs),
        bounds=bounds_pot,
        method="highs",
    )

    if res_pot.success:
        sol = res_pot.x
        mg_frac = [_to_frac(max(0.0, float(v)), max_den=100_000) for v in sol[idx_mg]]
        a_g_frac = sem["a_goal"]
        b_g_frac = sem["b_goal"]
        alpha_frac = [
            sum(a_g_frac[r][j] * mg_frac[r] for r in range(m_g)) for j in range(n)
        ]
        alpha0_frac = -sum(b_g_frac[r] * mg_frac[r] for r in range(m_g))

        ms_frac = [Fraction(0)] * (6 * n)
        for i in range(n):
            gw = alpha_frac[i] - sum(a_frac[r][i] * alpha_frac[r] for r in range(n))
            if gw >= 0:
                ms_frac[i] = gw
            else:
                ms_frac[n + i] = -gw

            gu = -sum(b_frac[r][i] * alpha_frac[r] for r in range(n))
            fw_i = sem["fuel_weights"][i]
            if gu >= 0:
                use_abs = min(gu, fw_i)
                ms_frac[4 * n + i] = use_abs
                ms_frac[2 * n + i] = gu - use_abs
            else:
                use_abs = min(-gu, fw_i)
                ms_frac[5 * n + i] = use_abs
                ms_frac[3 * n + i] = (-gu) - use_abs

        is_uniform_alpha = len(set(alpha_frac)) == 1
        used_premises = []
        if is_uniform_alpha and sem["sink_rate"] == 0:
            used_premises.append("P_conservation")
            if any(abs(v) >= sem["port_thresh"] for v in sem["positions"]) or sem["port_gain"] != 0:
                used_premises.append("P_control_input_gain")
        else:
            used_premises.extend(sem["declared_premises"])
            if any(ms_frac[n + i] > 0 for i in range(n)):
                used_premises.append("P_nonnegative_state_domain")

        pot_cert = {
            "certificate_id": f"{certificate_id}:potential_impossibility",
            "kind": "potential_lower_bound_certificate",
            "verdict": "impossible",
            "model_origin": sem["model_origin"],
            "declared_premises": sem["declared_premises"],
            "empirically_supported_premises": sem["empirically_supported_premises"],
            "refuted_premises": sem["refuted_premises"],
            "used_premise_ids": list(dict.fromkeys(used_premises)),
            "unused_premise_ids": [
                p
                for p in (
                    "P_conservation",
                    "P_interaction_direction",
                    "P_position_ownership",
                    "P_locality_radius",
                    "P_focus_membership",
                    "P_temporal_readout",
                    "P_control_input_gain",
                )
                if p not in used_premises
            ],
            "alpha_0": str(alpha0_frac),
            "alpha": [str(v) for v in alpha_frac],
            "farkas_mu_step": [str(v) for v in ms_frac],
            "farkas_mu_goal": [str(v) for v in mg_frac],
            "budget": budget_limit,
            "state_bounds_mode": sb_mode,
            "lean_status": "draft_only",
            "lean_compiled": False,
            "proof_verified_by": TRUSTED_CHECKER_BASE,
            "environment_coverage_verified": False,
        }
        ver = SymbolicProofKernel.verify_solvability_certificate(
            pot_cert, model_spec, task, budget=budget
        )
        total_spent += int(ver.get("compute_cost", 0))
        pot_cert["kernel_verification"] = ver
        pot_cert["status"] = ver.get("status", "unknown")
        pot_cert["compute_cost"] = total_spent
        if ver["verified"]:
            pot_cert["certified_min_budget"] = ver["certified_min_budget"]
            pot_cert["certified_min_budget_rational"] = ver["certified_min_budget_rational"]
            pot_cert["lean4_module"] = emit_lean4_certificate(pot_cert)
            return pot_cert
        if ver.get("status") == "conditional_impossibility_domain_unverified":
            pot_cert["verdict"] = "conditional_impossibility_domain_unverified"
            pot_cert["conditional_min_budget"] = ver.get("conditional_min_budget")
            pot_cert["lean4_module"] = emit_lean4_certificate(pot_cert)
            return pot_cert

    return {
        "certificate_id": f"{certificate_id}:unproved",
        "kind": "unproved",
        "verdict": "unknown",
        "status": "unknown",
        "model_origin": sem["model_origin"],
        "kernel_verification": {
            "verified": False,
            "status": "unknown",
            "reason": "no_witness_or_linear_potential_found",
        },
        "compute_cost": total_spent,
    }


def certify_rollout_baseline(
    lab: BudgetedControlLab,
    model_spec: dict,
    task: dict,
    *,
    max_trials: int = 250,
    horizon: int = 5,
    seed: int = 4242,
    budget: Any | None = None,
) -> dict[str, Any]:
    """Empirical rollout search equipped with the SAME ``SymbolicProofKernel`` witness checker (F07)."""
    rng = Random(seed)
    u_lo, u_hi = float(task["control_bounds"][0]), float(task["control_bounds"][1])
    b_val = float(task["budget"])
    n = lab.objects
    spent_total = 0

    for trial in range(max_trials):
        steps = (trial % horizon) + 1
        per_step_cap = min(abs(u_hi), max(abs(u_lo), b_val / max(1, steps * n)))
        lo_s = max(u_lo, -per_step_cap)
        hi_s = min(u_hi, per_step_cap)
        seq = [[rng.uniform(lo_s, hi_s) for _ in range(n)] for _ in range(steps)]
        out = lab.execute_trace(task["initial_state"], seq, task, budget=budget)
        spent_total += int(out.get("compute_cost", 0))
        if out.get("reason") == "budget_exhausted":
            return {
                "verdict": "budget_exhausted",
                "status": "unknown",
                "certified_proof": False,
                "trials_spent": trial,
                "witness_cost": None,
                "compute_cost": spent_total,
            }
        if out["winnable_verified_on_lab"]:
            rat_seq = [[str(_to_frac(v, max_den=1_000_000)) for v in row] for row in seq]
            cand_cert = {
                "certificate_id": f"rollout_witness_t{trial}",
                "kind": "reachability_witness_certificate",
                "action_sequence": rat_seq,
            }
            ver = SymbolicProofKernel.verify_solvability_certificate(
                cand_cert, model_spec, task, budget=budget
            )
            spent_total += int(ver.get("compute_cost", 0))
            return {
                "verdict": "winnable",
                "status": ver.get("status", "unknown"),
                "certified_proof": bool(ver.get("verified")),
                "kernel_verification": ver,
                "trials_spent": trial + 1,
                "witness_cost": out["total_cost"],
                "compute_cost": spent_total,
            }

    return {
        "verdict": "unproved_timeout",
        "status": "unknown",
        "certified_proof": False,
        "trials_spent": max_trials,
        "witness_cost": None,
        "compute_cost": spent_total,
    }


def extract_control_model_from_learned_commitment(
    commitment: dict,
    audit_report: dict,
    *,
    positions: list[float] | tuple[float, ...],
    sink_rate: float = 0.0,
) -> dict[str, Any]:
    """Bridge a runtime-learned ``WorldStructureCommitment`` + paid ``audit_report`` to ``model_spec`` (F14)."""
    params = commitment.get("radius_parameters") or ()
    direction = params[0]["direction"] if params else (audit_report.get("verified_direction") or "right")
    r_iv = audit_report.get("radius_interval") or (
        params[0]["source_intervals"][-1]
        if params and params[0].get("source_intervals")
        else None
    )
    radius = (
        0.5 * (float(r_iv[0]) + float(r_iv[1]))
        if r_iv and len(r_iv) == 2
        else float(params[0]["source_values"][-1]) if params and params[0].get("source_values") else 1.0
    )
    matches = [
        m
        for items in (audit_report.get("matches_by_direction") or {}).values()
        for m in items
    ]
    coupling = float(matches[0]["coupling_estimate"]) if matches else 0.25
    supported = list(audit_report.get("supported_premises") or ())
    return {
        "objects": len(positions),
        "positions": [float(x) for x in positions],
        "coupling": round(coupling, 6),
        "radius": round(radius, 6),
        "direction": direction,
        "sink_rate": float(sink_rate),
        "model_origin": "learned_model",
        "source_commitment_id": commitment.get("commitment_id"),
        "premise_ids": [p["premise_id"] for p in commitment.get("support_premises", ())]
        or ["P_conservation", "P_interaction_direction", "P_locality_radius"],
        "empirically_supported_premises": supported,
        "refuted_premises": [
            r.get("premise_id") for r in (audit_report.get("refuted_premises") or ()) if isinstance(r, dict)
        ],
    }


def repair_proof_from_discrepancy(
    lab: BudgetedControlLab,
    initial_model_spec: dict,
    premise_graph: dict,
    task: dict,
    initial_certificate: dict,
    *,
    discrepancy_feedback: dict,
    use_dependency_localization: bool = True,
    budget: Any | None = None,
) -> dict:
    """Equal-tool discrepancy-triggered premise audit & re-proof loop (supports single-factor ablation F05/F06).

    Handles 4 distinct physical discrepancy mechanisms with the SAME probe/estimation repertoire:
      1. ``boundary_port_gain`` (``P_control_input_gain`` / ``P_conservation``)
      2. ``bulk_sink_dissipation`` (``P_conservation``)
      3. ``coupling_shift`` (``P_locality_radius`` / ``P_exchange_coupling``)
      4. ``direction_reversal`` (``P_interaction_direction``)

    When ``use_dependency_localization=True``, probes ONLY premises in ``initial_certificate['used_premise_ids']``
    first before falling back to unused premises. When ``use_dependency_localization=False``, runs the exact same
    diagnostic probe families in a fixed untargeted order across all candidate premises!
    """
    if not discrepancy_feedback.get("oracle_winnable") or initial_certificate.get("verdict") != "impossible":
        return {"repaired": False, "reason": "no_impossibility_discrepancy", "compute_cost": 0}

    suspect_premises = list(initial_certificate.get("used_premise_ids", ()))
    ignored_premises = list(initial_certificate.get("unused_premise_ids", ()))

    n = int(initial_model_spec["objects"])
    u_lo, u_hi = float(task["control_bounds"][0]), float(task["control_bounds"][1])
    w_bounds = task.get("state_bounds")
    w_lo = float(w_bounds[0]) if w_bounds is not None else -10.0
    w_hi = float(w_bounds[1]) if w_bounds is not None else 10.0
    alpha = [float(_to_frac(v)) for v in initial_certificate["alpha"]]
    alpha_0 = float(_to_frac(initial_certificate["alpha_0"]))

    def _eval_v(w_vec: list[float]) -> float:
        return alpha_0 + sum(alpha[i] * w_vec[i] for i in range(n))

    probe_log: list[dict] = []
    spent_total = 0
    violating_transition = None
    repaired_fields: dict[str, Any] = {}
    diagnosed_mechanism: str | None = None
    refuted_premise_ids: list[str] = []

    # Shared equal-tool diagnostic probe families:
    #   Family 1 ("selector_and_readout_check"): zero-state / neutral-state baseline probes across objects
    #   Family 2 ("pairwise_exchange_direction_and_coupling"): single-donor q-state probes testing direction & coupling
    #   Family 3 ("bulk_conservation_and_sink"): interior state retention probes testing sink_rate
    #   Family 4 ("control_impulse_port_gain"): single-actuator impulse probes testing boundary_port_gain
    if use_dependency_localization:
        ordered_families: list[str] = []
        if any(p in suspect_premises for p in ("P_control_input_gain", "P_conservation")):
            ordered_families.extend(["control_impulse_port_gain", "bulk_conservation_and_sink"])
        if any(p in suspect_premises for p in ("P_interaction_direction", "P_locality_radius")):
            ordered_families.append("pairwise_exchange_direction_and_coupling")
        if any(p in suspect_premises for p in ("P_focus_membership", "P_position_ownership", "P_temporal_readout")):
            ordered_families.append("selector_and_readout_check")
        for fallback_fam in (
            "control_impulse_port_gain",
            "bulk_conservation_and_sink",
            "pairwise_exchange_direction_and_coupling",
            "selector_and_readout_check",
        ):
            if fallback_fam not in ordered_families:
                ordered_families.append(fallback_fam)
    else:
        # Untargeted order without certificate dependency cue: checks selector/position, pairwise exchange,
        # bulk conservation, and control input gain in canonical order.
        ordered_families = [
            "selector_and_readout_check",
            "pairwise_exchange_direction_and_coupling",
            "bulk_conservation_and_sink",
            "control_impulse_port_gain",
        ]

    q_safe = max(0.0, w_lo) if w_lo <= 0.0 <= w_hi else w_lo
    q_zero = [q_safe] * n
    q_excite = min(0.5, max(0.1, 0.25 * (w_hi - q_safe)))
    u_test_mag = min(0.4, max(0.0, u_hi)) if u_hi > 0 else max(-0.4, u_lo)

    for fam in ordered_families:
        if diagnosed_mechanism is not None:
            break

        if fam == "selector_and_readout_check":
            # Probe neutral zero-action step at initial state + per-object small state perturbations
            obs0 = lab.probe_transition(list(task["initial_state"]), [0.0] * n, budget=budget)
            spent_total += int(obs0["compute_cost"])
            step_drop0 = _eval_v(obs0["before_q"]) - _eval_v(obs0["after_q"])
            rec0 = {
                "probe_family": fam,
                "tested_object": -1,
                "potential_drop": float(step_drop0),
                "action_cost": float(obs0["cost"]),
                "violates_potential_step_inequality": bool(step_drop0 > obs0["cost"] + 1e-7),
            }
            probe_log.append(rec0)
            for idx in range(n):
                q_pert = [q_safe + (0.5 * q_excite if i == idx else 0.0) for i in range(n)]
                obs_s = lab.probe_transition(q_pert, [0.0] * n, budget=budget)
                spent_total += int(obs_s["compute_cost"])
                probe_log.append(
                    {
                        "probe_family": fam,
                        "tested_object": idx,
                        "potential_drop": float(_eval_v(obs_s["before_q"]) - _eval_v(obs_s["after_q"])),
                        "action_cost": float(obs_s["cost"]),
                        "violates_potential_step_inequality": False,
                    }
                )

        elif fam == "pairwise_exchange_direction_and_coupling":
            pos = list(initial_model_spec["positions"])
            init_r = float(initial_model_spec.get("radius", 1.0))
            init_dir = str(initial_model_spec.get("direction", "right"))
            init_c = float(initial_model_spec.get("coupling", 0.25))
            obs_base = lab.probe_transition(q_zero, [0.0] * n, budget=budget)
            spent_total += int(obs_base["compute_cost"])
            probe_log.append(
                {
                    "probe_family": fam,
                    "tested_object": -1,
                    "potential_drop": 0.0,
                    "action_cost": float(obs_base["cost"]),
                    "violates_potential_step_inequality": False,
                }
            )
            inferred_dir = None
            inferred_couplings: list[float] = []
            for src in range(n):
                q_vec = [q_safe + (q_excite if i == src else 0.0) for i in range(n)]
                obs = lab.probe_transition(q_vec, [0.0] * n, budget=budget)
                spent_total += int(obs["compute_cost"])
                deltas = [
                    (obs["after_q"][i] - obs_base["after_q"][i]) - (q_excite if i == src else 0.0)
                    for i in range(n)
                ]
                step_drop = _eval_v(obs["before_q"]) - _eval_v(obs["after_q"])
                viol = bool(step_drop > obs["cost"] + 1e-7)
                rec = {
                    "probe_family": fam,
                    "tested_object": src,
                    "potential_drop": float(step_drop),
                    "action_cost": float(obs["cost"]),
                    "violates_potential_step_inequality": viol,
                }
                probe_log.append(rec)
                if viol and violating_transition is None:
                    violating_transition = rec
                for dst in range(n):
                    if dst != src and deltas[dst] > 1e-5 and abs(pos[dst] - pos[src]) < init_r + 1e-9:
                        obs_dir = "right" if pos[dst] > pos[src] else "left"
                        inferred_dir = obs_dir if inferred_dir is None else inferred_dir
                        inferred_couplings.append(deltas[dst] / q_excite)

            if inferred_dir is not None and inferred_dir != init_dir and violating_transition is not None:
                diagnosed_mechanism = "direction_reversal"
                repaired_fields["direction"] = inferred_dir
                if inferred_couplings:
                    repaired_fields["coupling"] = round(sum(inferred_couplings) / len(inferred_couplings), 6)
                refuted_premise_ids = ["P_interaction_direction"]
            elif inferred_couplings:
                mean_c = sum(inferred_couplings) / len(inferred_couplings)
                if abs(mean_c - init_c) > 1e-3 and violating_transition is not None:
                    diagnosed_mechanism = "coupling_shift"
                    repaired_fields["coupling"] = round(mean_c, 6)
                    refuted_premise_ids = ["P_locality_radius"]

        elif fam == "bulk_conservation_and_sink":
            init_sink = float(initial_model_spec.get("sink_rate", 0.0))
            q_unif = [q_safe + q_excite] * n
            obs_u = lab.probe_transition(q_unif, [0.0] * n, budget=budget)
            spent_total += int(obs_u["compute_cost"])
            sum_before = sum(obs_u["before_q"])
            sum_after = sum(obs_u["after_q"])
            est_sink = (sum_before - sum_after) / sum_before if abs(sum_before) > 1e-9 else 0.0
            step_drop = _eval_v(obs_u["before_q"]) - _eval_v(obs_u["after_q"])
            viol = bool(step_drop > obs_u["cost"] + 1e-7)
            rec = {
                "probe_family": fam,
                "tested_object": -1,
                "estimated_sink_rate": float(est_sink),
                "potential_drop": float(step_drop),
                "action_cost": float(obs_u["cost"]),
                "violates_potential_step_inequality": viol,
            }
            probe_log.append(rec)
            if abs(est_sink - init_sink) > 1e-4 and not viol and u_test_mag != 0.0:
                obs_act = lab.probe_transition(q_unif, [u_test_mag] + [0.0] * (n - 1), budget=budget)
                spent_total += int(obs_act["compute_cost"])
                sd_act = _eval_v(obs_act["before_q"]) - _eval_v(obs_act["after_q"])
                viol_act = bool(sd_act > obs_act["cost"] + 1e-7)
                rec_act = {
                    "probe_family": fam,
                    "tested_object": 0,
                    "estimated_sink_rate": float(est_sink),
                    "potential_drop": float(sd_act),
                    "action_cost": float(obs_act["cost"]),
                    "violates_potential_step_inequality": viol_act,
                }
                probe_log.append(rec_act)
                if viol_act:
                    viol = True
                    rec = rec_act
            if abs(est_sink - init_sink) > 1e-4 and viol:
                violating_transition = violating_transition or rec
                diagnosed_mechanism = "bulk_sink_dissipation"
                repaired_fields["sink_rate"] = round(est_sink, 6)
                refuted_premise_ids = ["P_conservation"]

        elif fam == "control_impulse_port_gain":
            obs_z = lab.probe_transition(q_zero, [0.0] * n, budget=budget) if q_safe != 0.0 else None
            if obs_z is not None:
                spent_total += int(obs_z["compute_cost"])
            gains_by_obj: list[tuple[int, float, float]] = []
            for idx in range(n):
                u_vec = [u_test_mag if i == idx else 0.0 for i in range(n)]
                obs = lab.probe_transition(q_zero, u_vec, budget=budget)
                spent_total += int(obs["compute_cost"])
                v_before = _eval_v(obs["before_q"])
                v_after = _eval_v(obs["after_q"])
                step_drop = v_before - v_after
                base_after = obs_z["after_q"][idx] if obs_z is not None else 0.0
                gain_i = (obs["after_q"][idx] - base_after) / u_test_mag - 1.0
                gains_by_obj.append((idx, abs(obs["positions"][idx]), gain_i))
                rec = {
                    "probe_family": fam,
                    "tested_object": idx,
                    "position": obs["positions"][idx],
                    "potential_drop": float(step_drop),
                    "action_cost": float(obs["cost"]),
                    "violates_potential_step_inequality": bool(step_drop > obs["cost"] + 1e-7),
                    "excess_gain": float(gain_i),
                }
                probe_log.append(rec)
                if rec["violates_potential_step_inequality"] and violating_transition is None:
                    violating_transition = rec

            pos_gains = [g for _, _, g in gains_by_obj if g > 1e-4]
            zero_pos = [pos_abs for _, pos_abs, g in gains_by_obj if g <= 1e-4]
            active_pos = [pos_abs for _, pos_abs, g in gains_by_obj if g > 1e-4]
            if pos_gains and active_pos and violating_transition is not None:
                inferred_port_gain = sum(pos_gains) / len(pos_gains)
                inferred_port_threshold = (
                    0.5 * (max(zero_pos) + min(active_pos)) if zero_pos else min(active_pos) - 0.05
                )
                diagnosed_mechanism = "boundary_port_gain"
                repaired_fields["boundary_port_gain"] = float(round(inferred_port_gain, 6))
                repaired_fields["boundary_port_threshold"] = float(round(inferred_port_threshold, 6))
                refuted_premise_ids = ["P_conservation", "P_control_input_gain"]

    if violating_transition is None or diagnosed_mechanism is None:
        return {
            "repaired": False,
            "reason": "no_potential_step_violation_found_in_probes",
            "use_dependency_localization": use_dependency_localization,
            "audited_suspect_premises": suspect_premises,
            "targeted_probe_count": len(probe_log),
            "probe_log": probe_log,
            "compute_cost": spent_total,
        }

    updated_graph = deepcopy(premise_graph)
    for node in updated_graph.get("nodes", ()):
        if node["id"] in set(refuted_premise_ids):
            node["status"] = f"revised_{diagnosed_mechanism}"
            node["diagnosed_parameters"] = deepcopy(repaired_fields)
            node["counterexample"] = violating_transition

    repaired_model_spec = {
        **deepcopy(initial_model_spec),
        **repaired_fields,
        "refuted_premises": list(refuted_premise_ids),
        "premise_ids": [
            *(p for p in suspect_premises if p not in set(refuted_premise_ids)),
            f"P_revised_{diagnosed_mechanism}",
        ],
    }

    repaired_cert = prove_task_solvability(
        repaired_model_spec,
        task,
        certificate_id="repaired_solvability_cert",
        budget=budget,
    )
    spent_total += int(repaired_cert.get("compute_cost", 0))

    lab_replay = None
    if repaired_cert.get("kind") == "reachability_witness_certificate":
        float_actions = [[float(_to_frac(v)) for v in step] for step in repaired_cert["action_sequence"]]
        lab_replay = lab.execute_trace(task["initial_state"], float_actions, task, budget=budget)
        spent_total += int(lab_replay.get("compute_cost", 0))

    sub_budget = (
        round(0.8 * float(lab_replay["total_cost"]), 4)
        if lab_replay and lab_replay.get("total_cost") is not None
        else round(0.5 * float(task["budget"]), 4)
    )
    tight_lower_cert = prove_task_solvability(
        repaired_model_spec,
        {**deepcopy(task), "budget": sub_budget},
        certificate_id="repaired_tight_lower_bound",
        budget=budget,
    )
    spent_total += int(tight_lower_cert.get("compute_cost", 0))

    return {
        "repaired": bool(
            repaired_cert.get("kernel_verification", {}).get("verified")
            and (lab_replay is None or lab_replay.get("winnable_verified_on_lab"))
        ),
        "diagnosed_mechanism": diagnosed_mechanism,
        "use_dependency_localization": use_dependency_localization,
        "audited_suspect_premises": suspect_premises,
        "refuted_premises": refuted_premise_ids,
        "untouched_premises": ignored_premises,
        "targeted_probe_count": len(probe_log),
        "measured_probe_queries": len(probe_log),
        "violating_transition_witness": violating_transition,
        "repaired_model_spec": repaired_model_spec,
        "updated_premise_graph": updated_graph,
        "initial_certified_min_budget": initial_certificate.get("certified_min_budget"),
        "repaired_reachability_certificate": repaired_cert,
        "real_lab_execution_verification": lab_replay,
        "repaired_tight_impossibility_certificate": tight_lower_cert,
        "true_certified_min_budget": tight_lower_cert.get("certified_min_budget"),
        "compute_cost": spent_total,
    }

"""Domain-Family Plugins for Hegel V16.6 (Strictly Decoupled from Core Engine).

1. ``OrderedBoundaryPlugin``: Discrete ordered sequence diagnostics, distinguishing
   ``some_adjacent_flip`` (general opposite-endpoint bracket) from
   ``earliest_global_threshold`` (monotone sequence specialization), plus same-endpoint
   and stochastic/unbuildable handling.
2. ``BottleneckSaturationPlugin``: Open-grammar curve/bottleneck synthesis in original
   readout coordinates (linear, sqrt/log unbounded concave, rational saturation
   ``y = u / (alpha*u + beta)``, hard/tied ``min`` bottlenecks, sensor clipping,
   congestion rollover) + CEGAR joint multi-variable probe upgrade.
3. ``RelaxationFamilyPlugin``: Optional physical/stateful relaxation adapter ``(u, z, p, F)``
   separating local differential Schur stiffness bounds from multi-mode quadratic
   monotone step attenuation, zero-resistance conserved initial linkage, ``psi = u - L*p``,
   and SLS Kelvin-series vs Maxwell-parallel parameter mappings.
"""

from __future__ import annotations

import math
from typing import Any


# ============================================================================
# 1. Ordered Boundary Family Plugin
# ============================================================================

class OrderedBoundaryPlugin:
    """Shared primitives and contract checkers for ordered discrete boundary tasks."""

    @staticmethod
    def opposite_endpoint_bisect(
        query_fn,
        low_idx: int,
        high_idx: int,
        *,
        low_val: int | None = None,
        high_val: int | None = None,
        budget_remaining_fn=None,
    ) -> dict[str, Any]:
        """Find SOME adjacent sign/label flip in O(log N) when endpoints differ.

        Handles ``SKIP`` (unbuildable midpoint) via bounded local offset search
        without fabricating an answer if all interior points are unbuildable.
        """
        v_lo = query_fn(low_idx) if low_val is None else low_val
        v_hi = query_fn(high_idx) if high_val is None else high_val
        if v_lo == "SKIP" or v_hi == "SKIP":
            return {"status": "PROTOCOL_ERROR", "reason": "endpoint_unbuildable", "boundary": None}
        if v_lo == v_hi:
            return {
                "status": "NOT_DETECTED_UNDER_PROTOCOL",
                "reason": "endpoints_have_identical_label_no_ivp_guarantee",
                "boundary": None,
            }
        lo, hi = int(low_idx), int(high_idx)
        non_monotone_witness = False
        visited: dict[int, Any] = {lo: v_lo, hi: v_hi}
        while hi - lo > 1:
            if budget_remaining_fn is not None and budget_remaining_fn() <= 0:
                return {"status": "UNKNOWN", "reason": "budget_exhausted_mid_bisect", "boundary": None}
            mid = (lo + hi) // 2
            probe_idx = mid
            val = query_fn(probe_idx)
            visited[probe_idx] = val
            if val == "SKIP":
                # Try nearest untested offsets inside (lo, hi)
                resolved = False
                for delta in range(1, (hi - lo)):
                    for cand in (mid - delta, mid + delta):
                        if lo < cand < hi and cand not in visited:
                            c_val = query_fn(cand)
                            visited[cand] = c_val
                            if c_val != "SKIP":
                                probe_idx, val, resolved = cand, c_val, True
                                break
                    if resolved:
                        break
                if not resolved:
                    # Every interior point in (lo, hi) was tested and returned SKIP:
                    # (lo, hi) is the minimal testable sign-change bridge across SKIPs!
                    if all(visited.get(k) == "SKIP" for k in range(lo + 1, hi)):
                        return {
                            "status": "SUPPORTED_IN_SCOPE",
                            "guarantee_kind": "some_adjacent_flip",
                            "boundary_right_index": hi,
                            "boundary_pair": (lo, hi),
                            "bridged_unbuildable_interior": True,
                            "non_monotone_detected": non_monotone_witness,
                            "visited_count": len(visited),
                        }
                    return {"status": "UNKNOWN", "reason": "interior_points_unbuildable", "boundary": (lo, hi)}
            # Check if any visited points reveal non-monotonicity relative to (v_lo -> v_hi)
            ordered_pts = [visited[k] for k in sorted(visited) if visited[k] != "SKIP"]
            transitions = sum(1 for a, b in zip(ordered_pts, ordered_pts[1:]) if a != b)
            if transitions > 1:
                non_monotone_witness = True
            if val != v_lo:
                hi, v_hi = probe_idx, val
            else:
                lo, v_lo = probe_idx, val
        return {
            "status": "SUPPORTED_IN_SCOPE",
            "guarantee_kind": "some_adjacent_flip",
            "boundary_right_index": hi,
            "boundary_pair": (lo, hi),
            "non_monotone_detected": non_monotone_witness,
            "visited_count": len(visited),
        }

    @staticmethod
    def verify_or_find_earliest_flip(
        query_fn,
        n_length: int,
        candidate_right_idx: int | None = None,
        *,
        monotone_certificate_valid: bool = False,
        min_anomaly_width: int = 1,
        known_observations: dict[int, int | str] | None = None,
        budget_remaining_fn=None,
    ) -> dict[str, Any]:
        """Sound earliest-boundary verifier: never claims earliest without prefix coverage or certificate!

        1. If ``monotone_certificate_valid`` is True and ``candidate_right_idx`` was found by
           valid opposite-endpoint bisection, verifying ``(candidate_right_idx - 1, candidate_right_idx)``
           certifies the unique global threshold.
        2. Otherwise, if ``min_anomaly_width == W >= 1`` is guaranteed by explicit contract,
           scanning prefix with stride ``W`` cannot miss an anomaly block of width ``>= W``.
           Default ``W = 1`` (arbitrary non-monotone sequence!) scans every prefix index
           ``1, 2, ...`` until the first flip ``v[i] != v[0]`` is directly observed!
        3. If budget runs out before certifying the prefix up to a flip, returns
           ``UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT`` (never fabricating an earliest claim!).
        """
        cache: dict[int, int | str] = dict(known_observations or {})

        def get_val(idx: int) -> int | str | None:
            if idx in cache:
                return cache[idx]
            if budget_remaining_fn is not None and budget_remaining_fn() <= 0:
                return None
            val = query_fn(idx)
            if val is not None:
                cache[idx] = val
            return val

        v0 = get_val(0)
        if v0 is None or v0 == "SKIP":
            return {"status": "PROTOCOL_ERROR", "boundary_right_index": None}

        if monotone_certificate_valid and candidate_right_idx is not None and candidate_right_idx >= 1:
            vl = get_val(candidate_right_idx - 1)
            vr = get_val(candidate_right_idx)
            if vl is not None and vr is not None and vl != "SKIP" and vr != "SKIP" and vl == v0 and vr != v0:
                return {
                    "status": "SUPPORTED_IN_SCOPE",
                    "guarantee_kind": "earliest_global_threshold",
                    "certificate_basis": "verified_monotone_premise",
                    "boundary_right_index": candidate_right_idx,
                    "earlier_than_bisect_candidate": False,
                }

        stride = max(1, int(min_anomaly_width))
        upper_limit = candidate_right_idx if (candidate_right_idx is not None and candidate_right_idx < n_length) else (n_length - 1)

        if stride == 1:
            last_non_skip = v0
            for idx in range(1, upper_limit + 1):
                v_cur = get_val(idx)
                if v_cur is None:
                    return {
                        "status": "UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT",
                        "reason": "prefix_coverage_incomplete_under_budget",
                        "boundary_right_index": None,
                        "fallback_any_flip_candidate": candidate_right_idx,
                    }
                if v_cur == "SKIP":
                    continue
                if v_cur != v0:
                    return {
                        "status": "SUPPORTED_IN_SCOPE",
                        "guarantee_kind": "earliest_global_threshold",
                        "certificate_basis": "complete_prefix_observation_certificate",
                        "boundary_right_index": idx,
                        "bridged_skip_before_flip": cache.get(idx - 1) == "SKIP",
                        "earlier_than_bisect_candidate": (candidate_right_idx is not None and idx < candidate_right_idx),
                    }
                last_non_skip = v_cur
            # If candidate_right_idx was None (e.g. same-endpoint sequence) and full scan found no flip:
            return {"status": "NOT_DETECTED_UNDER_PROTOCOL", "reason": "no_flip_in_certified_prefix", "boundary_right_index": None}

        # Declared min_anomaly_width >= 2 contract path
        prev_idx = 0
        for idx in range(min(stride, upper_limit), upper_limit + 1, stride):
            v_cur = get_val(idx)
            if v_cur is None:
                return {
                    "status": "UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT",
                    "reason": "strided_prefix_incomplete_under_budget",
                    "boundary_right_index": None,
                }
            if v_cur == "SKIP":
                continue
            if v_cur != v0:
                for fine_idx in range(prev_idx + 1, idx + 1):
                    vf = get_val(fine_idx)
                    if vf is None:
                        return {"status": "UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT", "boundary_right_index": None}
                    if vf != "SKIP" and vf != v0:
                        return {
                            "status": "SUPPORTED_IN_SCOPE",
                            "guarantee_kind": "earliest_global_threshold",
                            "certificate_basis": f"min_anomaly_width_{stride}_certificate",
                            "boundary_right_index": fine_idx,
                            "earlier_than_bisect_candidate": (candidate_right_idx is not None and fine_idx < candidate_right_idx),
                        }
            prev_idx = idx
        return {"status": "UNKNOWN", "reason": "earliest_not_confirmed", "boundary_right_index": None}


# ============================================================================
# 2. Bottleneck & Saturation Family Plugin
# ============================================================================

class BottleneckSaturationPlugin:
    """Shared curve-family fitting (in original readout scale!) and joint bottleneck probing."""

    @staticmethod
    def fit_rational_saturation_original_scale(points: list[tuple[float, float]]) -> dict[str, Any]:
        """Fit ``y(u) = u / (alpha * u + beta)`` scored strictly in ORIGINAL y-scale!

        Uses reciprocal linear regression only as an Algebraic Warm-Start proposal,
        followed by Gauss-Newton / coordinate refinement in original ``y`` space so
        small-y reciprocal noise is never amplified or hidden.
        """
        valid = [(float(u), float(y)) for u, y in points if u > 0 and y > 0]
        if len(valid) < 2:
            return {"status": "insufficient_points", "alpha": None, "beta": None, "rmse": float("inf")}
        # Initial proposal via weighted reciprocal least squares (weight ~ y^4)
        xs = [1.0 / u for u, _ in valid]
        zs = [1.0 / y for _, y in valid]
        ws = [max(1e-8, y ** 4) for _, y in valid]
        w_sum = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / w_sum
        mz = sum(w * z for w, z in zip(ws, zs)) / w_sum
        denom = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        beta = sum(w * (x - mx) * (z - mz) for w, x, z in zip(ws, xs, zs)) / denom if denom > 1e-12 else 0.1
        alpha = mz - beta * mx
        alpha = max(1e-6, alpha)
        beta = max(1e-6, beta)

        def predict(u: float, a: float, b: float) -> float:
            return u / (a * u + b)

        def mse_orig(a: float, b: float) -> float:
            return sum((predict(u, a, b) - y) ** 2 for u, y in valid) / len(valid)

        # Damped Gauss-Newton refinement directly on original-scale MSE
        for _ in range(40):
            cur_loss = mse_orig(alpha, beta)
            g_a = g_b = h_aa = h_ab = h_bb = 0.0
            for u, y in valid:
                d = alpha * u + beta
                pred = u / d
                err = pred - y
                ja = -(u * u) / (d * d)
                jb = -u / (d * d)
                g_a += err * ja
                g_b += err * jb
                h_aa += ja * ja
                h_ab += ja * jb
                h_bb += jb * jb
            h_aa += 1e-6
            h_bb += 1e-6
            det = h_aa * h_bb - h_ab * h_ab
            if det <= 1e-15:
                break
            da = (h_bb * g_a - h_ab * g_b) / det
            db = (h_aa * g_b - h_ab * g_a) / det
            step = 1.0
            while step >= 1e-4:
                na = max(1e-7, alpha - step * da)
                nb = max(1e-7, beta - step * db)
                if mse_orig(na, nb) <= cur_loss:
                    alpha, beta = na, nb
                    break
                step *= 0.5
        rmse = math.sqrt(mse_orig(alpha, beta))
        return {
            "status": "ok",
            "family": "rational_saturation",
            "num_params": 2,
            "alpha": alpha,
            "beta": beta,
            "asymptote": 1.0 / alpha,
            "rmse_original_scale": rmse,
            "predict": lambda u, a=alpha, b=beta: float(u) / (a * float(u) + b),
        }

    @staticmethod
    def fit_competing_1d_families(points: list[tuple[float, float]]) -> list[dict[str, Any]]:
        """Open-grammar 1D candidate competition (shared equally with O_strong & O_HExA!)."""
        pts = [(float(u), float(y)) for u, y in points if u > 0]
        if len(pts) < 2:
            return []
        candidates: list[dict[str, Any]] = []

        # 1. Rational saturation y = u / (alpha*u + beta)
        rat = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts)
        if rat["status"] == "ok":
            candidates.append(rat)

        # 2. Linear affine y = a*u + b
        mu = sum(u for u, _ in pts) / len(pts)
        my = sum(y for _, y in pts) / len(pts)
        du = sum((u - mu) ** 2 for u, _ in pts)
        slope = sum((u - mu) * (y - my) for u, y in pts) / du if du > 1e-12 else 0.0
        intercept = my - slope * mu
        rmse_lin = math.sqrt(sum(((slope * u + intercept) - y) ** 2 for u, y in pts) / len(pts))
        candidates.append({
            "status": "ok", "family": "linear_affine", "num_params": 2,
            "slope": slope, "intercept": intercept, "rmse_original_scale": rmse_lin,
            "predict": lambda u, s=slope, c=intercept: s * float(u) + c,
        })

        # 3. Unbounded square-root concave y = a*sqrt(u) + b
        su = [math.sqrt(u) for u, _ in pts]
        msu = sum(su) / len(su)
        dsu = sum((s - msu) ** 2 for s in su)
        a_sq = sum((s - msu) * (y - my) for s, (_, y) in zip(su, pts)) / dsu if dsu > 1e-12 else 0.0
        b_sq = my - a_sq * msu
        rmse_sq = math.sqrt(sum(((a_sq * math.sqrt(u) + b_sq) - y) ** 2 for u, y in pts) / len(pts))
        candidates.append({
            "status": "ok", "family": "unbounded_sqrt_concave", "num_params": 2,
            "a": a_sq, "b": b_sq, "rmse_original_scale": rmse_sq,
            "predict": lambda u, a=a_sq, b=b_sq: a * math.sqrt(max(0.0, float(u))) + b,
        })

        # 4. Piecewise hard ceiling / clipping min(a*u, ceiling)
        max_y = max(y for _, y in pts)
        top_pts = [y for _, y in pts if y >= 0.97 * max_y]
        ceil_est = sum(top_pts) / len(top_pts)
        low_pts = [(u, y) for u, y in pts if y < 0.95 * ceil_est]
        k_est = (
            sum(u * y for u, y in low_pts) / max(1e-12, sum(u * u for u, _ in low_pts))
            if low_pts else (max_y / max(pts[0][0], 1e-6))
        )
        rmse_clip = math.sqrt(sum((min(k_est * u, ceil_est) - y) ** 2 for u, y in pts) / len(pts))
        candidates.append({
            "status": "ok", "family": "piecewise_hard_ceiling", "num_params": 2,
            "slope": k_est, "ceiling": ceil_est, "rmse_original_scale": rmse_clip,
            "predict": lambda u, k=k_est, c=ceil_est: min(k * float(u), c),
        })

        # 5. Non-monotone congestion rollover y = u / (alpha*u + beta + gamma*u^2)
        if len(pts) >= 3:
            # Since u / y = beta + alpha*u + gamma*u^2, solve weighted 3x3 least squares (w = y^4)
            basis = [(1.0, u, u * u) for u, _ in pts]
            targets = [u / max(y, 1e-6) for u, y in pts]
            weights = [max(1e-8, y ** 4) for _, y in pts]
            ata = [
                [sum(w * b[i] * b[j] for w, b in zip(weights, basis)) + (1e-8 if i == j else 0.0) for j in range(3)]
                for i in range(3)
            ]
            aty = [sum(w * b[i] * t for w, b, t in zip(weights, basis, targets)) for i in range(3)]
            aug = [ata[i] + [aty[i]] for i in range(3)]
            for col in range(3):
                piv = max(range(col, 3), key=lambda r: abs(aug[r][col]))
                aug[col], aug[piv] = aug[piv], aug[col]
                if abs(aug[col][col]) > 1e-12:
                    for r in range(col + 1, 3):
                        fac = aug[r][col] / aug[col][col]
                        for c in range(col, 4):
                            aug[r][c] -= fac * aug[col][c]
            sol = [0.0, 0.0, 0.0]
            for r in range(2, -1, -1):
                if abs(aug[r][r]) > 1e-12:
                    sol[r] = (aug[r][3] - sum(aug[r][c] * sol[c] for c in range(r + 1, 3))) / aug[r][r]
            b_r = max(1e-6, sol[0])
            a_r = max(1e-6, sol[1])
            g_r = max(0.0, sol[2])
            rmse_r = math.sqrt(sum(
                (u / (a_r * u + b_r + g_r * u * u) - y) ** 2 for u, y in pts
            ) / len(pts))
            candidates.append({
                "status": "ok", "family": "congestion_rollover", "num_params": 3,
                "alpha": a_r, "beta": b_r, "gamma": g_r,
                "rmse_original_scale": rmse_r,
                "predict": lambda u, a=a_r, b=b_r, g=g_r: (
                    float(u) / (a * float(u) + b + g * float(u) ** 2)
                ),
            })

        candidates.sort(key=lambda item: item["rmse_original_scale"] + 0.003 * item.get("num_params", 2))
        return candidates

    @staticmethod
    def diagnose_multi_resource_bottleneck(
        probe_fn,
        base_point: dict[str, float],
        *,
        step_delta: float = 1.0,
        allow_joint_probe_upgrade: bool = True,
        noise_tol: float = 1e-3,
    ) -> dict[str, Any]:
        """Probe single-axis sensitivities; on zero single-axis gains, trigger CEGAR joint probe!"""
        y0 = probe_fn(base_point)
        single_gains: dict[str, float] = {}
        for key in sorted(base_point):
            trial = dict(base_point)
            trial[key] = trial[key] + step_delta
            single_gains[key] = probe_fn(trial) - y0

        positive_axes = [k for k, g in single_gains.items() if g > noise_tol]
        if positive_axes:
            limiting_axis = max(positive_axes, key=lambda k: single_gains[k])
            return {
                "status": "SINGLE_LIMITING_FACTOR",
                "limiting_axes": [limiting_axis],
                "single_axis_gains": single_gains,
                "joint_probe_triggered": False,
                "base_output": y0,
            }
        if not allow_joint_probe_upgrade:
            return {
                "status": "ALL_SINGLE_AXIS_ZERO_GAIN",
                "limiting_axes": [],
                "single_axis_gains": single_gains,
                "joint_probe_triggered": False,
                "base_output": y0,
            }
        # CEGAR Probe Refinement: test joint increment across all zero-gain channels!
        joint_point = {k: v + step_delta for k, v in base_point.items()}
        joint_gain = probe_fn(joint_point) - y0
        if joint_gain > noise_tol:
            return {
                "status": "TIED_JOINT_BOTTLENECK_DETECTED",
                "limiting_axes": sorted(base_point.keys()),
                "single_axis_gains": single_gains,
                "joint_gain": joint_gain,
                "joint_probe_triggered": True,
                "base_output": y0,
            }
        return {
            "status": "TRUE_GLOBAL_CEILING_OR_CLIPPING",
            "limiting_axes": [],
            "single_axis_gains": single_gains,
            "joint_gain": joint_gain,
            "joint_probe_triggered": True,
            "base_output": y0,
        }


# ============================================================================
# 3. Relaxation & Conjugate Port Family Plugin (Optional Physics/Stateful Adapter)
# ============================================================================

class RelaxationFamilyPlugin:
    """Strictly scoped relaxation family contracts (never forced on non-relaxation tasks)."""

    @staticmethod
    def recover_total_flux_linkage(external_flux_u: float, inductance_l: float, conjugate_p_minus_i: float) -> float:
        """Exact sign-corrected flux recovery: since psi = u + L*I and p = -I, psi = u - L*p."""
        return float(external_flux_u) - float(inductance_l) * float(conjugate_p_minus_i)

    @staticmethod
    def local_schur_stiffness_bound(fuu: float, fuz: float, fzz: float) -> dict[str, Any]:
        """Compute local differential equilibrium Schur complement K_rel = F_uu - F_uz^2 / F_zz."""
        if fzz <= 0:
            return {"valid": False, "reason": "fzz_not_positive_definite"}
        k_rel = float(fuu) - (float(fuz) ** 2) / float(fzz)
        joint_pd = (fuu > 0) and (fuu * fzz - fuz * fuz >= -1e-12)
        return {
            "valid": True,
            "k_frozen": float(fuu),
            "k_relaxed": k_rel,
            "attenuated_le_frozen": k_rel <= float(fuu) + 1e-12,
            "nonnegative_relaxed": bool(joint_pd and k_rel >= -1e-12),
            "scope": "local_differential_equilibrium_only_not_finite_step_monotone",
        }

    @staticmethod
    def sls_parameters_by_topology(k1: float, k2: float, eta: float, *, topology: str) -> dict[str, float]:
        """Distinguish Kelvin-series vs Maxwell-parallel Standard Linear Solid realizations."""
        if topology == "kelvin_series":
            # Spring k1 in series with (dashpot eta || spring k2)
            return {
                "tau": eta / (k1 + k2),
                "rho": k1 / (k1 + k2),
                "frozen_stiffness": k1,
                "relaxed_stiffness": (k1 * k2) / (k1 + k2),
            }
        if topology == "maxwell_parallel":
            # Maxwell arm (spring k1 -- dashpot eta) in parallel with spring k2
            return {
                "tau": eta / k1,
                "rho": k1 / (k1 + k2),
                "frozen_stiffness": k1 + k2,
                "relaxed_stiffness": k2,
            }
        raise ValueError(f"Unknown SLS topology: {topology}")

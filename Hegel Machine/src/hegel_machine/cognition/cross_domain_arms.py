"""Fair Multi-Arm Researchers & Autonomous Formation-Revision-Wormhole Loop (Hegel V16.7 M1).

Implements four symmetric, genuinely history-consuming research arms plus causal ablations:
  1. ``HegelV166Researcher``: Autonomous evidence-gated 2-source contrastive schema
     induction into local implication trees + ``HorizontalWormholeEdge`` bridges between
     intermediate nodes, ALIGN near-miss condition splitting, CEGAR joint-probe program
     synthesis (separating weak diagnostic probes from confirmed scaling claims), 4-tier
     receipt-backed permission gating, empirical ``ScaleMap`` invariance certification,
     and Stage-0 / Stage-1 paired same-state empirical cost selection.
  2. ``StrongOrdinaryResearcher`` (``O_strong``): Identical shared primitives, active
     sampling, Stage-0 presolver, complete raw episode archive, result/prior caching in
     ``rebuilt_cache``, and genuine billed on-demand prior reconstruction directly from
     raw archived observations (never crippled, never forced when history is empty).
  3. ``HExAInspiredSkillResearcher`` (``O_HExA_inspired``): Learns & genuinely executes
     compact conditional skills and mistake-recovery entries ``(symptom, cause, recovery)``
     with cross-round skill rewriting (mutating/deleting bank entries changes behavior).
  4. ``FlatEqualContentResearcher`` (``Flat_equal_content``): Representation-only ablation
     holding identical induced contracts, ALIGN exclusions, ScaleMap certificates,
     declarative ``program_steps``, and 4-tier permission checks in a flat list without
     local tree hierarchy or non-transitive wormhole chain edges.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from .cross_domain_clusters import CostMeter, PublicOrderedBoundaryTask, PublicResourceResponseTask
from .family_plugins import BottleneckSaturationPlugin, OrderedBoundaryPlugin
from .question_semantics import evaluate_decision_region
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    TwoStageWormholeGate,
    authorize_abstraction_use,
)


# ============================================================================
# Shared Helpers & Low-Level Solvers (Available Equally to ALL Arms!)
# ============================================================================

def _serialize_1d_candidate(cand: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON-serializable fitted parameters from a 1D family candidate."""
    fam = cand.get("family", "unknown")
    out: dict[str, Any] = {
        "family": fam,
        "num_params": cand.get("num_params", 2),
        "rmse_original_scale": round(float(cand.get("rmse_original_scale", 0.0)), 6),
    }
    for key in ("alpha", "beta", "gamma", "asymptote", "slope", "intercept", "a", "b", "ceiling"):
        if key in cand and cand[key] is not None:
            out[key] = round(float(cand[key]), 6)
    if "weights" in cand and cand["weights"] is not None:
        out["weights"] = [round(float(w), 6) for w in cand["weights"]]
    return out


def _observation_signature(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def solve_ordered_boundary_direct(
    task: PublicOrderedBoundaryTask,
    *,
    use_bisect_primitive: bool = True,
    pre_queried_endpoints: tuple[int | str, int | str] | None = None,
    known_observations: dict[int, int | str] | None = None,
) -> dict[str, Any]:
    """Sound general ordered-sequence solver (shared across O_strong, O_HExA, Flat, & Hegel).

    - Uses binary search when endpoints differ.
    - When ``goal_kind == 'earliest_global_threshold'``, requires a sound prefix certificate
      (default ``min_anomaly_width = 1``, scanning every unobserved prefix index up to the
      first flip so short early pockets like ``[3, 4]`` are NEVER skipped!).
    - Never claims ``status == 'completed'`` if budget expires before certifying earliest!
    """
    n = task.length
    goal = task.task_view["goal_kind"]
    noisy = bool(task.task_view.get("stochastic_oracle_hint", False))
    min_anom_w = max(1, int(task.task_view.get("min_anomaly_width", 1)))
    visited: dict[int, int | str] = dict(known_observations or {})

    def robust_q(idx: int, phase: str = "probe") -> int | str | None:
        if not noisy and idx in visited:
            return visited[idx]
        v1 = task.query_index(idx, phase=phase)
        if v1 is None:
            return None
        if v1 == "SKIP" or not noisy or idx in {0, n - 1}:
            visited[idx] = v1
            return v1
        v2 = task.query_index(idx, phase=phase)
        v3 = task.query_index(idx, phase=phase)
        votes = [v for v in (v1, v2, v3) if isinstance(v, int)]
        res: int | str = "SKIP" if not votes else (1 if sum(votes) > (len(votes) / 2.0) else 0)
        visited[idx] = res
        return res

    def q_or_skip(idx: int, phase: str = "probe") -> int | str:
        val = robust_q(idx, phase=phase)
        return "SKIP" if val is None else val

    if pre_queried_endpoints is not None:
        v0, vn = pre_queried_endpoints
        visited.setdefault(0, v0)
        visited.setdefault(n - 1, vn)
    else:
        v0 = robust_q(0, phase="presolve")
        vn = robust_q(n - 1, phase="presolve")
    if v0 is None or vn is None:
        return {"status": "incomplete", "boundary_right_index": None}

    if v0 == vn:
        if goal == "earliest_global_threshold":
            # Sound prefix certificate required for earliest threshold (cannot skip short early pockets!)
            ver_same = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
                lambda i: robust_q(i, phase="probe"),
                n,
                candidate_right_idx=None,
                monotone_certificate_valid=False,
                min_anomaly_width=min_anom_w,
                known_observations=(None if noisy else visited),
                budget_remaining_fn=task.meter.remaining_obs,
            )
            if ver_same["status"] == "SUPPORTED_IN_SCOPE":
                return {
                    "status": "completed",
                    "guarantee_kind": "earliest_global_threshold",
                    "certificate_basis": ver_same.get("certificate_basis"),
                    "boundary_right_index": ver_same["boundary_right_index"],
                }
            if ver_same["status"] == "NOT_DETECTED_UNDER_PROTOCOL":
                return {"status": "no_flip_detected", "boundary_right_index": None}
            return {
                "status": "incomplete_unconfirmed_earliest",
                "reason": ver_same.get("reason", "budget_exhausted_before_prefix_certificate"),
                "boundary_right_index": None,
            }

        # goal == "some_adjacent_flip" on same-endpoint sequence: multi-resolution bracket + bisect
        bracket_hi = None
        prev_i = 0
        for stride in (8, 2, 1):
            prev_i = 0
            for idx in range(stride, n, stride):
                vi = robust_q(idx, phase="probe")
                if vi is None:
                    return {"status": "incomplete", "boundary_right_index": None}
                if vi != "SKIP" and vi != v0:
                    bracket_hi = idx
                    break
                if vi != "SKIP":
                    prev_i = idx
            if bracket_hi is not None:
                break
        if bracket_hi is None:
            return {"status": "no_flip_detected", "boundary_right_index": None}
        res = OrderedBoundaryPlugin.opposite_endpoint_bisect(
            lambda i: q_or_skip(i, "probe"),
            prev_i, bracket_hi, low_val=int(v0), high_val=1 - int(v0),
            budget_remaining_fn=task.meter.remaining_obs,
        )
        if res["status"] == "SUPPORTED_IN_SCOPE":
            return {
                "status": "completed",
                "guarantee_kind": "some_adjacent_flip",
                "boundary_right_index": res["boundary_right_index"],
            }
        return {"status": "incomplete", "boundary_right_index": None}

    if not use_bisect_primitive:
        for idx in range(1, n):
            vi = robust_q(idx, phase="probe")
            if vi is None:
                return {"status": "incomplete", "boundary_right_index": None}
            if vi != "SKIP" and vi != v0:
                return {"status": "completed", "guarantee_kind": goal, "boundary_right_index": idx}
        return {"status": "incomplete", "boundary_right_index": None}

    # Endpoints differ: run opposite-endpoint bisection
    for _attempt in range(2):
        bis = OrderedBoundaryPlugin.opposite_endpoint_bisect(
            lambda i: q_or_skip(i, "probe"), 0, n - 1, low_val=int(v0), high_val=int(vn),
            budget_remaining_fn=task.meter.remaining_obs,
        )
        if bis["status"] != "SUPPORTED_IN_SCOPE":
            return {"status": "incomplete", "boundary_right_index": None}
        cand_idx = bis["boundary_right_index"]

        if goal == "some_adjacent_flip":
            c_left = robust_q(max(0, cand_idx - 1), phase="confirm")
            c_right = robust_q(cand_idx, phase="confirm")
            if c_left == "SKIP":
                c_left = robust_q(max(0, cand_idx - 2), phase="confirm")
            if c_left is not None and c_right is not None and c_left != "SKIP" and c_right != "SKIP" and c_left != c_right:
                return {
                    "status": "completed",
                    "guarantee_kind": "some_adjacent_flip",
                    "boundary_right_index": cand_idx,
                }
            visited.clear()
            visited[0], visited[n - 1] = v0, vn
            continue

        # goal == "earliest_global_threshold": sound prefix certificate up to cand_idx!
        mono_cert = bool(task.task_view.get("monotone_certificate_valid", False))
        ver = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
            lambda i: robust_q(i, phase="confirm"),
            n,
            cand_idx,
            monotone_certificate_valid=mono_cert,
            min_anomaly_width=min_anom_w,
            known_observations=(None if noisy else visited),
            budget_remaining_fn=task.meter.remaining_obs,
        )
        if ver["status"] == "SUPPORTED_IN_SCOPE":
            return {
                "status": "completed",
                "guarantee_kind": "earliest_global_threshold",
                "certificate_basis": ver.get("certificate_basis"),
                "boundary_right_index": ver["boundary_right_index"],
            }
        if ver["status"] == "UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT":
            return {
                "status": "incomplete_unconfirmed_earliest",
                "reason": ver.get("reason"),
                "boundary_right_index": None,
                "unverified_any_flip_candidate": cand_idx,
            }
    return {"status": "incomplete", "boundary_right_index": None}


def _fit_all_open_1d_candidates(points: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Open 1D candidate grammar (shared across ALL arms) including inverse-quadratic burst tail."""
    cands = BottleneckSaturationPlugin.fit_competing_1d_families(points)
    pts = [(float(u), float(y)) for u, y in points if u > 0]
    if len(pts) >= 3:
        basis = [(1.0, 1.0 / u, 1.0 / (u * u)) for u, _ in pts]
        ys = [y for _, y in pts]
        ata = [[sum(b[i] * b[j] for b in basis) + (1e-8 if i == j else 0.0) for j in range(3)] for i in range(3)]
        aty = [sum(b[i] * y for b, y in zip(basis, ys)) for i in range(3)]
        aug = [ata[i] + [aty[i]] for i in range(3)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            if abs(aug[col][col]) > 1e-12:
                for r in range(col + 1, 3):
                    fac = aug[r][col] / aug[col][col]
                    for c in range(col, 4):
                        aug[r][c] -= fac * aug[col][c]
        w = [0.0, 0.0, 0.0]
        for r in range(2, -1, -1):
            if abs(aug[r][r]) > 1e-12:
                w[r] = (aug[r][3] - sum(aug[r][c] * w[c] for c in range(r + 1, 3))) / aug[r][r]
        rmse_q = math.sqrt(sum(((w[0] + w[1] / u + w[2] / (u * u)) - y) ** 2 for u, y in pts) / len(pts))
        cands.append({
            "status": "ok", "family": "inverse_quadratic_tail", "num_params": 3,
            "weights": w, "rmse_original_scale": rmse_q,
            "predict": lambda u, cw=tuple(w): cw[0] + cw[1] / float(u) + cw[2] / (float(u) ** 2),
        })
    cands.sort(key=lambda c: c["rmse_original_scale"] + 0.003 * c.get("num_params", 2))
    return cands


def solve_resource_1d_direct(
    task: PublicResourceResponseTask,
    *,
    reconstructed_prior_family: str | None = None,
    existing_points: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """General 1D curve identification & confirmation solver (shared with O_strong, O_HExA, Flat, Hegel)."""
    lo, hi = task.control_bounds["u"]
    tol = float(task.task_view["tolerance"])
    pts: list[tuple[float, float]] = list(existing_points or ())

    # Shared Stage-0 3-point pilot grid
    presolve_u = [lo + 0.06 * (hi - lo), lo + 0.36 * (hi - lo), lo + 0.88 * (hi - lo)]
    for u_val in presolve_u:
        if len(pts) >= 3:
            break
        if all(abs(u_val - ep[0]) > 0.25 for ep in pts):
            y_val = task.measure({"u": u_val}, phase="presolve" if len(pts) < 2 else "probe")
            if y_val is None:
                break
            pts.append((u_val, y_val))

    if len(pts) < 3:
        return {"status": "incomplete"}

    # Stage-0 linear fast check
    task.meter.charge("fit", compute=6.0)
    lin_only = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts[:2]) if c["family"] == "linear_affine"][0]
    if abs(lin_only["predict"](pts[2][0]) - pts[2][1]) <= tol * 0.5:
        u_conf = lo + 0.72 * (hi - lo)
        y_conf = task.measure({"u": u_conf}, phase="confirm")
        if y_conf is not None and abs(lin_only["predict"](u_conf) - y_conf) <= tol:
            return {
                "status": "completed",
                "selected_family": "linear_affine",
                "fitted_model_params": _serialize_1d_candidate(lin_only),
                "predict_fn": lin_only["predict"],
            }
        if y_conf is not None:
            pts.append((u_conf, y_conf))

    # Optional reconstructed prior family from raw episode archive (used by O_strong & rebuild ablation!)
    if reconstructed_prior_family == "rational_saturation":
        task.meter.charge("fit", compute=8.0)
        rat = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
        if (rat["status"] == "ok"
                and rat["rmse_original_scale"] <= tol
                and abs(rat["predict"](pts[2][0]) - pts[2][1]) <= tol):
            u_conf = lo + 0.62 * (hi - lo)
            y_conf = task.measure({"u": u_conf}, phase="confirm")
            if y_conf is not None and abs(rat["predict"](u_conf) - y_conf) <= tol:
                rat_full = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
                return {
                    "status": "completed",
                    "selected_family": "rational_saturation",
                    "used_reconstructed_archive_prior": True,
                    "fitted_model_params": _serialize_1d_candidate(rat_full),
                    "predict_fn": rat_full["predict"],
                }
            if y_conf is not None:
                pts.append((u_conf, y_conf))

    # General adaptive discrimination across open 1D grammar
    extra_design = [lo + 0.18 * (hi - lo), lo + 0.34 * (hi - lo), lo + 0.72 * (hi - lo)]
    for u_d in extra_design:
        if all(abs(u_d - ep[0]) > 0.4 for ep in pts):
            yd = task.measure({"u": u_d}, phase="probe")
            if yd is None:
                break
            pts.append((u_d, yd))

    task.meter.charge("fit", compute=18.0)
    train_pts = pts[:-1]
    val_pt = pts[-1]
    cands = _fit_all_open_1d_candidates(train_pts)
    best = min(
        cands,
        key=lambda c: 0.6 * abs(c["predict"](val_pt[0]) - val_pt[1]) + 0.4 * c["rmse_original_scale"] + 0.003 * c.get("num_params", 2),
    )

    # Independent held-out confirmation point (if 0.62 was already consumed in failed prior trial, use fresh 0.48!)
    for rel_h in (0.62, 0.48):
        u_hold = lo + rel_h * (hi - lo)
        if all(abs(u_hold - ep[0]) > 0.35 for ep in pts):
            y_hold = task.measure({"u": u_hold}, phase="confirm")
            if y_hold is not None:
                pts.append((u_hold, y_hold))
                task.meter.charge("confirm", compute=8.0)
                all_cands = _fit_all_open_1d_candidates(pts)
                same_fam = [c for c in all_cands if c["family"] == best["family"]]
                chosen = same_fam[0] if (same_fam and abs(best["predict"](u_hold) - y_hold) <= tol * 1.2) else all_cands[0]
                return {
                    "status": "completed",
                    "selected_family": chosen["family"],
                    "fitted_model_params": _serialize_1d_candidate(chosen),
                    "predict_fn": chosen["predict"],
                }
            break

    all_cands = _fit_all_open_1d_candidates(pts)
    same_fam = [c for c in all_cands if c["family"] == best["family"]]
    chosen = same_fam[0] if same_fam else all_cands[0]
    return {
        "status": "completed",
        "selected_family": chosen["family"],
        "fitted_model_params": _serialize_1d_candidate(chosen),
        "predict_fn": chosen["predict"],
    }


def solve_resource_2d_ordinary(task: PublicResourceResponseTask) -> dict[str, Any]:
    """Strong ordinary 2D resource allocator (coordinate perturbations + fallback grid/diagonal)."""
    init_pt = dict(task.task_view["initial_operating_point"])
    y_req = float(task.task_view["required_output"])

    y0 = task.measure(init_pt, phase="presolve")
    if y0 is None:
        return {"status": "incomplete"}
    p1 = {"u1": init_pt["u1"] + 1.0, "u2": init_pt["u2"]}
    y1 = task.measure(p1, phase="probe")
    p2 = {"u1": init_pt["u1"], "u2": init_pt["u2"] + 1.0}
    y2 = task.measure(p2, phase="probe")
    if y1 is None or y2 is None:
        return {"status": "incomplete"}
    task.meter.charge("fit", compute=12.0)

    g1, g2 = y1 - y0, y2 - y0
    if g1 <= 1e-4 and g2 <= 1e-4:
        pd1 = {"u1": init_pt["u1"] + 1.0, "u2": init_pt["u2"] + 1.0}
        yd1 = task.measure(pd1, phase="probe")
        pd2 = {"u1": init_pt["u1"] + 2.0, "u2": init_pt["u2"] + 1.0}
        yd2 = task.measure(pd2, phase="probe")
        pd3 = {"u1": init_pt["u1"] + 1.0, "u2": init_pt["u2"] + 2.0}
        yd3 = task.measure(pd3, phase="probe")
        if yd1 is None or yd2 is None or yd3 is None:
            return {"status": "incomplete"}
        c1_est = y0 / init_pt["u1"]
        c2_est = y0 / init_pt["u2"]
        rec = {"u1": y_req / max(c1_est, 1e-6), "u2": y_req / max(c2_est, 1e-6)}
        yc = task.measure(rec, phase="confirm")
        if yc is not None and yc >= y_req * 0.97:
            return {"status": "completed", "recommended_controls": rec}
        return {"status": "incomplete"}

    if g1 > g2:
        c1_est = g1 / 1.0
        u1_target = y_req / c1_est
        p_hi1 = {"u1": u1_target, "u2": init_pt["u2"]}
        y_hi1 = task.measure(p_hi1, phase="probe")
        p_hi1_step2 = {"u1": u1_target, "u2": init_pt["u2"] + 1.0}
        y_hi1_s2 = task.measure(p_hi1_step2, phase="probe")
        if y_hi1 is None or y_hi1_s2 is None:
            return {"status": "incomplete"}
        c2_est = (y_hi1 / init_pt["u2"]) if y_hi1 < y_req * 0.98 else max(y_req / init_pt["u2"], 1.0)
        rec = {"u1": u1_target, "u2": max(init_pt["u2"], y_req / max(c2_est, 1e-6))}
    else:
        c2_est = g2 / 1.0
        u2_target = y_req / c2_est
        p_hi2 = {"u1": init_pt["u1"], "u2": u2_target}
        y_hi2 = task.measure(p_hi2, phase="probe")
        p_hi2_step1 = {"u1": init_pt["u1"] + 1.0, "u2": u2_target}
        y_hi2_s1 = task.measure(p_hi2_step1, phase="probe")
        if y_hi2 is None or y_hi2_s1 is None:
            return {"status": "incomplete"}
        c1_est = (y_hi2 / init_pt["u1"]) if y_hi2 < y_req * 0.98 else max(y_req / init_pt["u1"], 1.0)
        rec = {"u1": max(init_pt["u1"], y_req / max(c1_est, 1e-6)), "u2": u2_target}

    yc = task.measure(rec, phase="confirm")
    if yc is not None and yc >= y_req * 0.97:
        return {"status": "completed", "recommended_controls": rec}
    return {"status": "incomplete"}


def _infer_1d_family_from_raw_observations(obs_list: list[Any]) -> str | None:
    """Reconstruct best-fitting 1D family directly from raw (control, value, phase) observations."""
    pts = []
    for item in obs_list:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], dict) and "u" in item[0]:
            pts.append((float(item[0]["u"]), float(item[1])))
    if len(pts) < 3:
        return None
    cands = _fit_all_open_1d_candidates(pts[:-1])
    if not cands:
        return None
    best = min(
        cands,
        key=lambda c: 0.5 * abs(c["predict"](pts[-1][0]) - pts[-1][1]) + 0.5 * c["rmse_original_scale"] + 0.004 * c.get("num_params", 2),
    )
    return best["family"]


# ============================================================================
# Declarative Node Program Interpreter (Shared by Hegel & Flat_equal_content)
# ============================================================================

def execute_2d_bottleneck_program(
    task: PublicResourceResponseTask,
    program_steps: list[dict[str, Any]],
    *,
    used_wormhole_id: str | None = None,
) -> dict[str, Any] | None:
    """Execute declarative 2D bottleneck ``program_steps`` stored on a forest/flat node.

    Modifying ``program_steps`` (e.g., removing ``if_all_zero_probe_joint`` or changing ``step_delta``)
    directly changes the target probe trajectory!
    """
    init_pt = dict(task.task_view["initial_operating_point"])
    y_req = float(task.task_view["required_output"])
    ops = {step.get("op"): step for step in program_steps if isinstance(step, dict)}
    if "probe_single_axes" not in ops:
        return None

    delta = float(ops["probe_single_axes"].get("step_delta", 1.0))
    allow_joint = "if_all_zero_probe_joint" in ops
    diag = BottleneckSaturationPlugin.diagnose_multi_resource_bottleneck(
        lambda pt: task.measure(pt, phase="probe") or 0.0,
        init_pt,
        step_delta=delta,
        allow_joint_probe_upgrade=allow_joint,
    )
    task.meter.charge("fit", compute=6.0)

    if diag["status"] == "TIED_JOINT_BOTTLENECK_DETECTED" and "scale_piecewise_min_and_confirm" in ops:
        y0 = diag["base_output"]
        if y0 <= 1e-6:
            return None
        rec = {"u1": init_pt["u1"] * (y_req / y0), "u2": init_pt["u2"] * (y_req / y0)}
        yc = task.measure(rec, phase="confirm")
        if yc is not None and yc >= y_req * 0.97:
            return {
                "status": "completed",
                "recommended_controls": rec,
                "cegar_joint_probe_used": True,
                "executed_program_ops": list(ops.keys()),
                "used_wormhole": used_wormhole_id,
            }
        return {"status": "incomplete", "refuted_at_confirm": True, "observed_confirm": yc}

    if diag["status"] == "SINGLE_LIMITING_FACTOR" and "probe_other_channel_at_limiting_req" in ops:
        lim = diag["limiting_axes"][0]
        other = "u2" if lim == "u1" else "u1"
        c_lim = diag["single_axis_gains"][lim] / max(delta, 1e-6)
        u_lim_req = y_req / max(c_lim, 1e-6)
        probe_pt = {lim: u_lim_req, other: init_pt[other]}
        y_at_lim = task.measure(probe_pt, phase="probe")
        if y_at_lim is None:
            return {"status": "incomplete"}
        if y_at_lim >= y_req * 0.98:
            rec = probe_pt
        else:
            c_other = y_at_lim / max(init_pt[other], 1e-6)
            rec = {lim: u_lim_req, other: y_req / max(c_other, 1e-6)}
        yc = task.measure(rec, phase="confirm")
        if yc is not None and yc >= y_req * 0.97:
            return {
                "status": "completed",
                "recommended_controls": rec,
                "executed_program_ops": list(ops.keys()),
                "used_wormhole": used_wormhole_id,
            }
        return {"status": "incomplete", "refuted_at_confirm": True, "observed_confirm": yc}

    return None


# ============================================================================
# Arm 1: Strong Ordinary Researcher (O_strong)
# ============================================================================

class StrongOrdinaryResearcher:
    """Full-capability Ordinary baseline with raw episode archive, caching, & genuine on-demand rebuild."""

    def __init__(self, *, allow_archive_reconstruction: bool = True) -> None:
        self.raw_archive: list[dict[str, Any]] = []
        self.rebuilt_cache: dict[str, Any] = {}
        self.allow_archive_reconstruction = bool(allow_archive_reconstruction)

    def run_source_and_nearmiss_phase(self, cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> list[dict[str, Any]]:
        logs = []
        for s_name, (factory, evaluator) in cluster_a["sources"]:
            task = factory()
            sub = solve_ordered_boundary_direct(task)
            ev = evaluator(sub)
            rec = {
                "phase": "source", "arm": "O_strong", "episode_name": s_name,
                "surface_domain": task.surface_domain,
                "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot(),
                "query_log": list(task.query_log),
            }
            self.raw_archive.append(deepcopy(rec))
            logs.append(rec)

        nm_a_name, (nm_a_fac, nm_a_eval) = cluster_a["near_miss"]
        t_nma = nm_a_fac()
        sub_nma = solve_ordered_boundary_direct(t_nma)
        ev_nma = nm_a_eval(sub_nma)
        rec_nma = {
            "phase": "source_nearmiss", "arm": "O_strong", "episode_name": nm_a_name,
            "surface_domain": t_nma.surface_domain,
            "strict_pass": ev_nma["strict_pass"], "costs": t_nma.meter.snapshot(),
            "query_log": list(t_nma.query_log),
        }
        self.raw_archive.append(deepcopy(rec_nma))
        logs.append(rec_nma)

        for b_phase, pairs in (("source", cluster_b["sources"]), ("source_nearmiss", cluster_b["near_misses"])):
            for name, (factory, evaluator) in pairs:
                task = factory()
                if "u" in task.control_bounds:
                    sub = solve_resource_1d_direct(task)
                else:
                    sub = solve_resource_2d_ordinary(task)
                ev = evaluator(sub)
                rec = {
                    "phase": b_phase, "arm": "O_strong", "episode_name": name,
                    "surface_domain": task.surface_domain,
                    "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot(),
                    "observations": deepcopy(task.observations),
                    "selected_family": sub.get("selected_family"),
                }
                self.raw_archive.append(deepcopy(rec))
                logs.append(rec)
        return logs

    def _reconstruct_and_cache_archive_priors(self, task: Any) -> str | None:
        """Inspect raw observations in ``self.raw_archive`` once per cluster and cache induced prior."""
        if not self.allow_archive_reconstruction or not self.raw_archive:
            return None
        archive_fingerprint = len(self.raw_archive)
        if self.rebuilt_cache.get("archive_episode_count") == archive_fingerprint:
            return self.rebuilt_cache.get("1d_prior_family")

        if not task.meter.charge("on_demand_reconstruction", compute=28.0):
            return None

        family_domains: dict[str, set[str]] = {}
        for idx, ep in enumerate(self.raw_archive):
            obs = ep.get("observations")
            if not obs:
                continue
            fam = _infer_1d_family_from_raw_observations(obs)
            if fam is not None:
                dom = str(ep.get("surface_domain") or f"ep_{idx}")
                family_domains.setdefault(fam, set()).add(dom)

        # Require >= 2 distinct supporting domains in raw history to cache a cross-domain prior
        prior_fam = None
        for fam, doms in sorted(family_domains.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(doms) >= 2:
                prior_fam = fam
                break

        self.rebuilt_cache = {
            "archive_episode_count": archive_fingerprint,
            "1d_prior_family": prior_fam,
            "supporting_domains_by_family": {k: sorted(v) for k, v in family_domains.items()},
        }
        return prior_fam

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        if isinstance(task, PublicOrderedBoundaryTask):
            return solve_ordered_boundary_direct(task)
        if "u" in task.control_bounds:
            if self.raw_archive:
                task.meter.charge("retrieve", compute=2.0)
                prior = self._reconstruct_and_cache_archive_priors(task)
                return solve_resource_1d_direct(task, reconstructed_prior_family=prior)
            return solve_resource_1d_direct(task)
        if self.raw_archive:
            task.meter.charge("retrieve", compute=2.0)
        return solve_resource_2d_ordinary(task)


# ============================================================================
# Arm 2: HExA-Inspired Conditional Skill + Mistake-Recovery Bank (O_HExA_inspired)
# ============================================================================

class HExAInspiredSkillResearcher:
    """Compact Conditional Skill + Mistake-Recovery Bank baseline (inspired by HExA 2606.29315v2).

    ``solve_target`` genuinely reads and executes ``self.skill_bank`` and ``self.mistake_bank``:
    removing or editing a skill or mistake entry directly alters target probe actions!
    """

    def __init__(self) -> None:
        self.skill_bank: dict[str, dict[str, Any]] = {}
        self.mistake_bank: list[dict[str, Any]] = []
        self.raw_archive: list[dict[str, Any]] = []

    def run_source_and_nearmiss_phase(self, cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> list[dict[str, Any]]:
        logs = []
        a_passes = 0
        for name, (factory, evaluator) in cluster_a["sources"]:
            task = factory()
            sub = solve_ordered_boundary_direct(task)
            task.meter.charge("distill_skill", compute=8.0)
            ev = evaluator(sub)
            a_passes += int(bool(ev["strict_pass"]))
            rec = {
                "phase": "source", "arm": "O_HExA_inspired", "episode_name": name,
                "surface_domain": task.surface_domain,
                "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot(),
                "query_log": list(task.query_log),
            }
            self.raw_archive.append(deepcopy(rec))
            logs.append(rec)
        if a_passes > 0:
            self.skill_bank["skill_ordered_bisect"] = {
                "condition": "ordered_sequence_opposite_endpoints",
                "strategy": "midpoint_bisect",
                "version": 1,
            }

        nm_name, (nm_fac, nm_eval) = cluster_a["near_miss"]
        nm_task = nm_fac()
        nm_sub = solve_ordered_boundary_direct(nm_task)
        nm_task.meter.charge("rewrite_skill", compute=10.0)
        nm_ev = nm_eval(nm_sub)
        if "skill_ordered_bisect" in self.skill_bank:
            self.skill_bank["skill_ordered_bisect"]["version"] = 2
        self.mistake_bank.append({
            "skill_id": "skill_ordered_bisect",
            "symptom": "goal_is_earliest_and_sequence_may_toggle",
            "recovery": "verify_complete_prefix_before_earliest_claim",
        })
        rec_nma = {
            "phase": "source_nearmiss", "arm": "O_HExA_inspired", "episode_name": nm_name,
            "surface_domain": nm_task.surface_domain,
            "strict_pass": nm_ev["strict_pass"], "costs": nm_task.meter.snapshot(),
            "query_log": list(nm_task.query_log),
        }
        self.raw_archive.append(deepcopy(rec_nma))
        logs.append(rec_nma)

        b_rat_domains: set[str] = set()
        for name, (factory, evaluator) in cluster_b["sources"]:
            task = factory()
            sub = solve_resource_1d_direct(task)
            task.meter.charge("distill_skill", compute=8.0)
            ev = evaluator(sub)
            if ev["strict_pass"] and sub.get("selected_family") == "rational_saturation":
                b_rat_domains.add(task.surface_domain)
            rec = {
                "phase": "source", "arm": "O_HExA_inspired", "episode_name": name,
                "surface_domain": task.surface_domain,
                "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot(),
                "observations": deepcopy(task.observations),
                "selected_family": sub.get("selected_family"),
            }
            self.raw_archive.append(deepcopy(rec))
            logs.append(rec)
        if b_rat_domains:
            self.skill_bank["skill_diminishing_saturation"] = {
                "condition": "positive_diminishing_marginal_response",
                "scale_note": "applies to steady resource-throughput curves",
                "strategy": "try_2pt_rational_fit_then_confirm",
                "confirm_rel_pos": 0.62,
                "version": 1,
            }

        for nm_b_name, (nmb_fac, nmb_eval) in cluster_b["near_misses"]:
            t_nm = nmb_fac()
            sub_nm = solve_resource_1d_direct(t_nm) if "u" in t_nm.control_bounds else solve_resource_2d_ordinary(t_nm)
            t_nm.meter.charge("rewrite_skill", compute=10.0)
            ev_nm = nmb_eval(sub_nm)
            if "u" not in t_nm.control_bounds and ev_nm["strict_pass"]:
                self.mistake_bank.append({
                    "skill_id": "skill_diminishing_saturation",
                    "symptom": "multi_resource_zero_single_step",
                    "recovery": "try_joint_step_and_intermediate_check",
                })
            elif "u" in t_nm.control_bounds:
                self.mistake_bank.append({
                    "skill_id": "skill_diminishing_saturation",
                    "symptom": "3rd_pt_or_confirm_residual_exceeds_tol",
                    "recovery": "fallback_open_curve_search",
                })
            rec_nmb = {
                "phase": "source_nearmiss", "arm": "O_HExA_inspired", "episode_name": nm_b_name,
                "surface_domain": t_nm.surface_domain,
                "strict_pass": ev_nm["strict_pass"], "costs": t_nm.meter.snapshot(),
                "observations": deepcopy(t_nm.observations),
            }
            self.raw_archive.append(deepcopy(rec_nmb))
            logs.append(rec_nmb)

        if "skill_diminishing_saturation" in self.skill_bank:
            self.skill_bank["skill_diminishing_saturation"]["version"] = 2
        return logs

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        task.meter.charge("retrieve", compute=5.0)
        if isinstance(task, PublicOrderedBoundaryTask):
            use_bisect = (
                self.skill_bank.get("skill_ordered_bisect", {}).get("strategy") == "midpoint_bisect"
            )
            return solve_ordered_boundary_direct(task, use_bisect_primitive=use_bisect)

        if "u" not in task.control_bounds:
            has_joint_recovery = any(
                m.get("recovery") == "try_joint_step_and_intermediate_check" for m in self.mistake_bank
            )
            if not has_joint_recovery:
                return solve_resource_2d_ordinary(task)

            init_pt = dict(task.task_view["initial_operating_point"])
            y_req = float(task.task_view["required_output"])
            diag = BottleneckSaturationPlugin.diagnose_multi_resource_bottleneck(
                lambda pt: task.measure(pt, phase="probe") or 0.0,
                init_pt, allow_joint_probe_upgrade=True,
            )
            task.meter.charge("fit", compute=10.0)
            if diag["status"] == "TIED_JOINT_BOTTLENECK_DETECTED":
                y0 = diag["base_output"]
                mid_scale = 0.5 * (1.0 + y_req / max(y0, 1e-6))
                task.measure({"u1": init_pt["u1"] * mid_scale, "u2": init_pt["u2"] * mid_scale}, phase="probe")
                rec = {"u1": init_pt["u1"] * (y_req / y0), "u2": init_pt["u2"] * (y_req / y0)}
                yc = task.measure(rec, phase="confirm")
                if yc is not None and yc >= y_req * 0.97:
                    return {"status": "completed", "recommended_controls": rec, "used_skill_recovery": True}
            return solve_resource_2d_ordinary(task)

        sat_skill = self.skill_bank.get("skill_diminishing_saturation")
        if sat_skill is None or sat_skill.get("strategy") != "try_2pt_rational_fit_then_confirm":
            return solve_resource_1d_direct(task)

        lo, hi = task.control_bounds["u"]
        tol = float(task.task_view["tolerance"])
        u1, u2, u3 = lo + 0.06 * (hi - lo), lo + 0.36 * (hi - lo), lo + 0.88 * (hi - lo)
        y1 = task.measure({"u": u1}, phase="probe")
        y2 = task.measure({"u": u2}, phase="probe")
        y3 = task.measure({"u": u3}, phase="probe")
        if y1 is None or y2 is None or y3 is None:
            return {"status": "incomplete"}
        pts = [(u1, y1), (u2, y2), (u3, y3)]
        task.meter.charge("fit", compute=8.0)

        # Flat text skill lacks formal ScaleMap closure check & numeric ALIGN tail predicate,
        # so it trials its 2pt rational fit + confirm probe unless curve is linear, then recovers on mismatch!
        lin = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts[:2]) if c["family"] == "linear_affine"][0]
        if abs(lin["predict"](u3) - y3) <= tol * 0.5:
            return solve_resource_1d_direct(task, existing_points=pts)

        rat = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
        rel_conf = float(sat_skill.get("confirm_rel_pos", 0.62))
        u_conf = lo + rel_conf * (hi - lo)
        y_conf = task.measure({"u": u_conf}, phase="confirm")
        if (y_conf is not None
                and rat["status"] == "ok"
                and rat["rmse_original_scale"] <= tol
                and abs(rat["predict"](u3) - y3) <= tol
                and abs(rat["predict"](u_conf) - y_conf) <= tol):
            rat_all = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
            return {
                "status": "completed",
                "selected_family": "rational_saturation",
                "used_skill_id": "skill_diminishing_saturation",
                "fitted_model_params": _serialize_1d_candidate(rat_all),
                "predict_fn": rat_all["predict"],
            }
        pts_extra = pts + ([(u_conf, y_conf)] if y_conf is not None else [])
        has_fallback_recovery = any(
            m.get("recovery") == "fallback_open_curve_search" for m in self.mistake_bank
        )
        if not has_fallback_recovery and rat["status"] == "ok":
            return {
                "status": "completed",
                "selected_family": "rational_saturation",
                "fitted_model_params": _serialize_1d_candidate(rat),
                "predict_fn": rat["predict"],
            }
        return solve_resource_1d_direct(task, existing_points=pts_extra)


# ============================================================================
# Arm 3: Hegel V16.7 (Local Implication Trees + Mid-Layer Wormholes + Governance)
# ============================================================================

class HegelV166Researcher:
    """Full V16.7 engine: Local Implication Trees + Mid-Layer Horizontal Wormholes + Governance."""

    def __init__(
        self,
        *,
        gate: TwoStageWormholeGate | None = None,
        allow_horizontal_wormholes: bool = True,
        allow_raw_history_rebuild_on_missing: bool = True,
    ) -> None:
        self.forest = PhilosophicalForest()
        self.gate = gate if gate is not None else TwoStageWormholeGate()
        self.allow_horizontal_wormholes = bool(allow_horizontal_wormholes)
        self.allow_raw_history_rebuild_on_missing = bool(allow_raw_history_rebuild_on_missing)
        self.rebuilt_cache: dict[str, Any] = {}
        self.learned_scale_certificates: dict[str, dict[str, Any]] = {}
        self.formation_events: list[dict[str, Any]] = []
        self.revision_events: list[dict[str, Any]] = []
        self.alignment_checks: list[dict[str, Any]] = []

    def form_and_revise_from_experience(
        self,
        cluster_id: str,
        cluster_a: dict[str, Any],
        cluster_b: dict[str, Any],
        cluster_c: dict[str, Any] | None = None,
        *,
        single_source_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Autonomously form local trees & horizontal mid-layer wormholes with strict evidence gating!"""
        logs: list[dict[str, Any]] = []
        seen_source_fingerprints: set[tuple[str, str, str]] = set()

        # --- 1. Cluster A Evidence-Gated Formation ---
        a_sources = cluster_a["sources"][:1] if single_source_only else cluster_a["sources"]
        supported_a_nodes: list[tuple[str, str]] = []  # (node_id, surface_domain)

        for idx, (s_name, (fac, ev_fn)) in enumerate(a_sources):
            t = fac()
            sub = solve_ordered_boundary_direct(t)
            t.meter.charge("formation", compute=8.0)
            ev = ev_fn(sub)
            obs_sig = _observation_signature(t.query_log)
            fp = (s_name, t.surface_domain, obs_sig)
            is_duplicate = fp in seen_source_fingerprints
            seen_source_fingerprints.add(fp)

            ep_rec = {
                "phase": "source", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
                "episode_name": s_name, "surface_domain": t.surface_domain,
                "strict_pass": ev["strict_pass"], "costs": t.meter.snapshot(),
                "query_log": list(t.query_log),
                "duplicate_source_skipped": is_duplicate,
            }
            self.forest.record_raw_episode(ep_rec)
            logs.append(ep_rec)
            if is_duplicate:
                continue

            distinct_labels = {val for _, val, _ in t.query_log if val != "SKIP"}
            empirically_supported = (
                bool(ev["strict_pass"])
                and sub.get("status") == "completed"
                and len(distinct_labels) >= 2
            )
            ev_status = "SUPPORTED_IN_SCOPE" if empirically_supported else "NOT_DETECTED_UNDER_PROTOCOL"
            perm_tier = "local_solver_guidance" if empirically_supported else "retrieve_candidate"

            # Canonical node IDs mapped from observed surface_domain (never blind array slot!)
            if t.surface_domain == "software_ci" or idx == 0:
                nid = "ci_mid_bisect" if t.surface_domain == "software_ci" else f"mid_{t.surface_domain}_bisect"
                tid = f"tree_{t.surface_domain}"
                # Keep ci_mid_bisect alias if first ordered source is software_ci (or audit fixture overriding software_ci)
                if idx == 0 and t.surface_domain == "software_ci":
                    nid, tid = "ci_mid_bisect", "tree_software_ci"
            else:
                nid = "dac_mid_bisect" if t.surface_domain == "hardware_dac" else f"mid_{t.surface_domain}_bisect"
                tid = "tree_hardware_dac" if t.surface_domain == "hardware_dac" else f"tree_{t.surface_domain}"

            goal_k = t.task_view["goal_kind"]
            n_reg = self.forest.register_node(
                node_id=nid,
                tree_id=tid,
                layer_role="mid",
                task_view=TaskView(goal_k, "exact_adjacent_boundary", 0.0),
                observation_map=ObservationMap("discrete_binary_label", parameter_semantics={"source_surface_domain": t.surface_domain}),
                claim_contract=ClaimContract(
                    f"bisect_{nid}", "ordered_boundary",
                    [{"field": "opposite_endpoints", "op": "==", "value": True}],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": goal_k},
                    {"requires_deterministic_binary": True}, 0.0, ev_status, perm_tier,
                ),
                probe_signatures=["sig:ordered_opposite_endpoints"],
                program_steps=[{"op": "opposite_endpoint_bisect"}, {"op": "verify_goal_contract"}],
                question_spec={
                    "id": f"q_{nid}",
                    "question": f"Does opposite-endpoint bisection satisfy {goal_k} in {t.surface_domain}?",
                    "kind": "mechanism",
                    "answer_type": "scoped_outcome",
                },
                source_cluster_ids=[cluster_id],
            )
            if empirically_supported:
                self.forest.questions[f"q_{nid}"]["semantic_answer"] = {
                    "status": "answered", "value": {"outcome_status": "SUPPORTED_IN_SCOPE"},
                }
                supported_a_nodes.append((nid, t.surface_domain))
            self.formation_events.append({
                "cluster_id": cluster_id, "event": "form_local_tree_node",
                "node_id": n_reg["node_id"], "tree_id": n_reg["tree_id"], "version": n_reg["version"],
                "evidence_status": ev_status, "from_sources": [s_name],
            })

        # Form Horizontal Wormhole ONLY if >= 2 distinct surface domains achieved SUPPORTED_IN_SCOPE!
        distinct_a_domains = {dom for _, dom in supported_a_nodes}
        if len(supported_a_nodes) >= 2 and len(distinct_a_domains) >= 2:
            na_id, nb_id = supported_a_nodes[0][0], supported_a_nodes[1][0]
            wh_a = self.forest.add_horizontal_wormhole(
                wormhole_id="wh_A_ci_dac_bisect",
                node_a_id=na_id,
                node_b_id=nb_id,
                probe_signature_refs=["sig:ordered_opposite_endpoints"],
                role_and_observation_map={"index_map": "ordered_index_morphism", "label_map": "binary_state_morphism"},
                hop_error_budget=0.0,
                permission_tier="local_solver_guidance",
                compatible_goal_kinds=["some_adjacent_flip", "earliest_global_threshold"],
                source_episode_refs=[s[0] for s in a_sources],
            )
            self.formation_events.append({
                "cluster_id": cluster_id, "event": "form_horizontal_mid_wormhole",
                "wormhole_id": wh_a["wormhole_id"], "node_a": na_id, "node_b": nb_id,
            })

        # --- Near-Miss Revision on Cluster A (Hotfix-Revert Non-Monotone Sequence!) ---
        nm_a_name, (nm_a_fac, nm_a_eval) = cluster_a["near_miss"]
        t_nma = nm_a_fac()
        v0 = t_nma.query_index(0, phase="presolve")
        vn = t_nma.query_index(t_nma.length - 1, phase="presolve")
        bis_nm = OrderedBoundaryPlugin.opposite_endpoint_bisect(
            lambda i: t_nma.query_index(i, phase="probe"), 0, t_nma.length - 1, low_val=v0, high_val=vn
        )
        ver_nm = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
            lambda i: t_nma.query_index(i, phase="confirm"),
            t_nma.length,
            bis_nm["boundary_right_index"],
            known_observations={idx: val for idx, val, _ in t_nma.query_log},
            budget_remaining_fn=t_nma.meter.remaining_obs,
        )
        t_nma.meter.charge("revision", compute=10.0)
        nm_sub = {
            "status": "completed" if ver_nm["status"] == "SUPPORTED_IN_SCOPE" else "incomplete",
            "guarantee_kind": "earliest_global_threshold",
            "boundary_right_index": ver_nm.get("boundary_right_index"),
        }
        nm_ev = nm_a_eval(nm_sub)
        rec_nma = {
            "phase": "source_nearmiss", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
            "episode_name": nm_a_name, "surface_domain": t_nma.surface_domain,
            "strict_pass": nm_ev["strict_pass"], "costs": t_nma.meter.snapshot(),
            "query_log": list(t_nma.query_log),
        }
        self.forest.record_raw_episode(rec_nma)
        logs.append(rec_nma)

        if ver_nm.get("earlier_than_bisect_candidate") and "ci_mid_bisect" in self.forest.nodes:
            old_ver = self.forest.nodes["ci_mid_bisect"]["version"]
            was_initially_supported = (
                self.forest.nodes["ci_mid_bisect"]["claim_contract"]["evidence_status"] == "SUPPORTED_IN_SCOPE"
            )
            rev_mid = self.forest.empirical_revise_node(
                "ci_mid_bisect",
                witness_episode_id=nm_a_name,
                cause="protocol_or_scope_mismatch",
                updated_task_view=TaskView("some_adjacent_flip", "exact_adjacent_boundary", 0.0).to_dict(),
                updated_claim_contract=ClaimContract(
                    "bisect_any_adjacent_flip_general", "ordered_boundary",
                    [{"field": "opposite_endpoints", "op": "==", "value": True}],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": "some_adjacent_flip"},
                    {"requires_deterministic_binary": True}, 0.0,
                    "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ).to_dict(),
            )
            if "wh_A_ci_dac_bisect" in self.forest.horizontal_wormhole_edges:
                self.forest.revalidate_wormhole_after_revision(
                    "wh_A_ci_dac_bisect",
                    witness_episode_id=nm_a_name,
                    reason="Both ci_mid_bisect@v2 and dac_mid_bisect@v1 share verified some_adjacent_flip contract",
                )
            leaf_status = "SUPPORTED_IN_SCOPE" if was_initially_supported else "candidate"
            leaf_mono = self.forest.register_node(
                node_id="ci_leaf_monotone_earliest", tree_id="tree_software_ci", layer_role="leaf",
                task_view=TaskView("earliest_global_threshold", "exact_adjacent_boundary", 0.0),
                observation_map=ObservationMap("discrete_binary_label"),
                claim_contract=ClaimContract(
                    "bisect_monotone_unique_earliest", "ordered_boundary",
                    [
                        {"field": "opposite_endpoints", "op": "==", "value": True},
                        {"field": "monotone_verified", "op": "==", "value": True},
                    ],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": "earliest_global_threshold"},
                    {"requires_monotone": True}, 0.0, leaf_status,
                    "validated_pruning" if was_initially_supported else "retrieve_candidate",
                ),
                program_steps=[{"op": "opposite_endpoint_bisect"}, {"op": "certify_monotone_earliest"}],
                question_spec={
                    "id": "q_ci_leaf_monotone_earliest",
                    "question": "Is sequence certified monotone so opposite-endpoint bisection yields earliest threshold?",
                    "kind": "scope_condition",
                    "answer_type": "scoped_outcome",
                },
                support_paths=[[{"node_id": "ci_mid_bisect", "version": rev_mid["version"]}]],
                source_cluster_ids=[cluster_id],
            )
            self.forest.add_local_implication_edge(
                stronger_node_id="ci_leaf_monotone_earliest",
                weaker_node_id="ci_mid_bisect",
                preserved_contract={"backoff_guarantee": "some_adjacent_flip"},
                scope={"opposite_endpoints": True},
                evidence_status=leaf_status,
            )
            self.revision_events.append({
                "cluster_id": cluster_id,
                "witness_episode": nm_a_name,
                "diagnosis_cause": "protocol_or_scope_mismatch",
                "old_node_version": old_ver,
                "new_mid_version": rev_mid["version"],
                "spawned_specialized_leaf": leaf_mono["node_id"],
                "rule_learned": "Non-monotone bisection guarantees some_adjacent_flip, NOT earliest_global_threshold; bind strictly to TaskView!",
            })

        # --- 2. Cluster B Two-Source Contrastive Formation (with Deduplication & Spurious Attr Elimination!) ---
        b_sources = cluster_b["sources"][:1] if single_source_only else cluster_b["sources"]
        b_unique_models: list[dict[str, Any]] = []
        seen_b_domains: set[str] = set()

        for s_name, (fac, ev_fn) in b_sources:
            t = fac()
            sub = solve_resource_1d_direct(t)
            t.meter.charge("formation", compute=10.0)
            ev = ev_fn(sub)
            pts = [(ctrl["u"], val) for ctrl, val, _ in t.observations]
            obs_sig = _observation_signature(t.observations)
            fp = (s_name, t.surface_domain, obs_sig)
            is_dup = (fp in seen_source_fingerprints) or (t.surface_domain in seen_b_domains)
            seen_source_fingerprints.add(fp)

            cands = _fit_all_open_1d_candidates(pts[:-1])
            best_fam = min(
                cands,
                key=lambda c: 0.5 * abs(c["predict"](pts[-1][0]) - pts[-1][1]) + 0.5 * c["rmse_original_scale"] + 0.004 * c.get("num_params", 2),
            )
            rejected_scores = {
                c["family"]: round(float(c["rmse_original_scale"]), 5)
                for c in cands if c["family"] != best_fam["family"]
            }
            ep_rec = {
                "phase": "source", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
                "episode_name": s_name, "surface_domain": t.surface_domain,
                "domain_regime_attrs": deepcopy(getattr(t, "domain_regime_attrs", {})),
                "strict_pass": ev["strict_pass"], "costs": t.meter.snapshot(),
                "observations": deepcopy(t.observations),
                "induced_family": best_fam["family"] if ev["strict_pass"] else None,
                "competing_family_rmse": rejected_scores,
                "duplicate_source_skipped": is_dup,
            }
            self.forest.record_raw_episode(ep_rec)
            logs.append(ep_rec)
            if is_dup or not ev["strict_pass"]:
                continue
            seen_b_domains.add(t.surface_domain)
            b_unique_models.append({
                "episode_name": s_name,
                "surface_domain": t.surface_domain,
                "domain_regime_attrs": deepcopy(getattr(t, "domain_regime_attrs", {})),
                "best_fam": best_fam,
                "rejected_scores": rejected_scores,
            })

        # Contrastively compute shared regime preconditions across valid rational sources!
        rat_sources = [m for m in b_unique_models if m["best_fam"]["family"] == "rational_saturation"]
        shared_regime_preconditions: list[dict[str, Any]] = []
        if len(rat_sources) == 1:
            # Single-source conservative induction: retains observed boolean regime attributes of its sole source!
            for attr_k, attr_v in sorted(rat_sources[0]["domain_regime_attrs"].items()):
                if isinstance(attr_v, bool):
                    shared_regime_preconditions.append({"field": attr_k, "op": "==", "value": attr_v})
        elif len(rat_sources) >= 2:
            # Two-source contrastive induction: eliminates non-invariant surface attributes (e.g. has_serial_mutex)!
            common_keys = set(rat_sources[0]["domain_regime_attrs"]).intersection(
                *(set(m["domain_regime_attrs"]) for m in rat_sources[1:])
            )
            for attr_k in sorted(common_keys):
                vals = {m["domain_regime_attrs"][attr_k] for m in rat_sources}
                if len(vals) == 1 and isinstance(next(iter(vals)), bool):
                    shared_regime_preconditions.append({"field": attr_k, "op": "==", "value": next(iter(vals))})

        registered_rat_mid_ids: list[str] = []
        for idx, m in enumerate(rat_sources):
            dom = m["surface_domain"]
            if dom == "parallel_compute_pool":
                root_id, mid_id, tree_id = "sys_root_concave_limit", "sys_mid_rational", "tree_compute_pool"
            elif dom == "biochem_enzyme_assay":
                root_id, mid_id, tree_id = None, "bio_mid_rational", "tree_biochem_assay"
            else:
                root_id, mid_id, tree_id = (f"root_{dom}_concave" if idx == 0 else None), f"mid_{dom}_rational", f"tree_{dom}"

            if root_id is not None:
                self.forest.register_node(
                    node_id=root_id, tree_id=tree_id, layer_role="root",
                    task_view=TaskView("response_curve_and_target_sizing", "rmse_original_scale", 0.03),
                    observation_map=ObservationMap("positive_response_channel", parameter_semantics={"source_surface_domain": dom}),
                    claim_contract=ClaimContract(
                        "monotone_concave_resource_limit", "bottleneck",
                        [{"field": "monotone_non_decreasing", "op": "==", "value": True}],
                        {"property": "diminishing_marginal_return"}, {}, 0.05, "SUPPORTED_IN_SCOPE", "retrieve_candidate",
                    ),
                    probe_signatures=["sig:diminishing_marginal_positive"],
                    question_spec={
                        "id": f"q_{root_id}",
                        "question": "Does diminishing marginal response hold without finite-asymptote pruning power?",
                        "kind": "coarse_property",
                        "answer_type": "scoped_outcome",
                    },
                    source_cluster_ids=[cluster_id],
                )

            base_preconds = [
                {"field": "monotone_non_decreasing", "op": "==", "value": True},
                {"field": "clipping_flat_tail", "op": "==", "value": False},
                *deepcopy(shared_regime_preconditions),
            ]
            reg_mid = self.forest.register_node(
                node_id=mid_id, tree_id=tree_id, layer_role="mid",
                task_view=TaskView("response_curve_and_target_sizing", "rmse_original_scale", 0.025),
                observation_map=ObservationMap(
                    "saturation_response_channel",
                    parameter_semantics={
                        "alpha": "reciprocal_asymptote_coeff",
                        "beta": "half_saturation_slope_coeff",
                        "source_surface_domain": dom,
                    },
                ),
                claim_contract=ClaimContract(
                    f"rational_saturation_{dom}", "bottleneck",
                    base_preconds,
                    {"procedure": "2pt_rational_fit_plus_1pt_heldout_confirm", "family": "rational_saturation"},
                    {"fit_scale": "original_y_gauss_newton", "requires_heldout_point": True},
                    0.025, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ),
                probe_signatures=["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"],
                program_steps=[
                    {"op": "fit_2pt_rational_original_scale"},
                    {"op": "probe_confirm_point", "rel_pos": 0.62},
                    {"op": "refit_full_rational_if_confirmed"},
                ],
                question_spec={
                    "id": f"q_{mid_id}",
                    "question": f"Does 2-parameter rational saturation u/(alpha*u+beta) hold within tol in {dom}?",
                    "kind": "mechanism",
                    "answer_type": "scoped_outcome",
                },
                source_cluster_ids=[cluster_id],
            )
            registered_rat_mid_ids.append(reg_mid["node_id"])
            if root_id is not None:
                self.forest.add_local_implication_edge(
                    stronger_node_id=mid_id, weaker_node_id=root_id,
                    preserved_contract={"property": "diminishing_marginal_return"},
                    scope={"positive_u": True},
                )

        if len(registered_rat_mid_ids) >= 2:
            wh_b = self.forest.add_horizontal_wormhole(
                wormhole_id="wh_B_sys_bio_rational",
                node_a_id=registered_rat_mid_ids[0],
                node_b_id=registered_rat_mid_ids[1],
                probe_signature_refs=["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"],
                role_and_observation_map={
                    "control_role_map": f"{rat_sources[0]['surface_domain']}.u <-> {rat_sources[1]['surface_domain']}.u",
                    "contrastively_eliminated_surface_attrs": sorted(
                        set(rat_sources[0]["domain_regime_attrs"]) ^ set(rat_sources[1]["domain_regime_attrs"])
                        | {
                            k for k in set(rat_sources[0]["domain_regime_attrs"]) & set(rat_sources[1]["domain_regime_attrs"])
                            if rat_sources[0]["domain_regime_attrs"][k] != rat_sources[1]["domain_regime_attrs"][k]
                        }
                    ),
                },
                hop_error_budget=0.008,
                permission_tier="local_solver_guidance",
                compatible_goal_kinds=["response_curve_and_target_sizing", "steady_mean_throughput"],
                source_episode_refs=[m["episode_name"] for m in rat_sources[:2]],
            )
            self.formation_events.append({
                "cluster_id": cluster_id, "event": "form_horizontal_mid_wormhole",
                "wormhole_id": wh_b["wormhole_id"],
                "node_a": registered_rat_mid_ids[0], "node_b": registered_rat_mid_ids[1],
                "competing_families_rmse_on_sources": [m["rejected_scores"] for m in rat_sources[:2]],
            })

        # --- Near-Miss B1: Tied 2D Bottleneck (Split Weak Joint-Probe Question vs Confirmed Min-Scaling!) ---
        nm_b1_name, (nm_b1_fac, nm_b1_eval) = cluster_b["near_misses"][0]
        t_nmb1 = nm_b1_fac()
        init_pt = dict(t_nmb1.task_view["initial_operating_point"])
        y_req = float(t_nmb1.task_view["required_output"])
        d_1d = BottleneckSaturationPlugin.diagnose_multi_resource_bottleneck(
            lambda pt: t_nmb1.measure(pt, phase="probe") or 0.0, init_pt, allow_joint_probe_upgrade=False,
        )
        joint_pt = {k: v + 1.0 for k, v in init_pt.items()}
        y_joint = t_nmb1.measure(joint_pt, phase="revision_probe") or 0.0
        t_nmb1.meter.charge("revision", compute=12.0)
        y0_tie = d_1d["base_output"]
        rec_nm1 = {"u1": init_pt["u1"] * (y_req / max(y0_tie, 1e-6)), "u2": init_pt["u2"] * (y_req / max(y0_tie, 1e-6))}
        yc_nm1 = t_nmb1.measure(rec_nm1, phase="confirm")
        # Strictly verify target achievement from the observed confirmation measurement `yc_nm1`!
        confirmed_unbounded_min_scaling = (yc_nm1 is not None and yc_nm1 >= y_req * 0.97)
        sub_nmb1 = (
            {"status": "completed", "recommended_controls": rec_nm1}
            if confirmed_unbounded_min_scaling
            else {"status": "incomplete_capped_or_refuted", "recommended_controls": rec_nm1}
        )
        ev_nmb1 = nm_b1_eval(sub_nmb1)
        rec_ep_b1 = {
            "phase": "source_nearmiss", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
            "episode_name": nm_b1_name, "surface_domain": t_nmb1.surface_domain,
            "strict_pass": ev_nmb1["strict_pass"], "costs": t_nmb1.meter.snapshot(),
            "observations": deepcopy(t_nmb1.observations),
        }
        self.forest.record_raw_episode(rec_ep_b1)
        logs.append(rec_ep_b1)

        if d_1d["status"] == "ALL_SINGLE_AXIS_ZERO_GAIN" and y_joint > y0_tie + 1e-4:
            # 1. Weak root diagnostic program: "when single-axis steps give 0, test bounded joint step (+1,+1)"
            self.forest.register_node(
                node_id="bot_root_joint_complementarity_question",
                tree_id="tree_multi_bottleneck",
                layer_role="root",
                task_view=TaskView("multi_resource_target_allocation", "achieve_target_with_near_minimal_resource_spend", 0.05),
                observation_map=ObservationMap("pipeline_capacity", parameter_semantics={"source_surface_domain": t_nmb1.surface_domain}),
                claim_contract=ClaimContract(
                    "joint_probe_when_single_axes_tied", "bottleneck",
                    [{"field": "multi_resource", "op": "==", "value": True}],
                    {"diagnostic_question": "probe_joint_increment_on_zero_marginal_axes"},
                    {"never_permanently_prune_zero_gain_axis": True},
                    0.05, "SUPPORTED_IN_SCOPE", "bounded_trial",
                ),
                probe_signatures=["sig:multi_resource_limiting_factor"],
                program_steps=[
                    {"op": "probe_single_axes", "step_delta": 1.0},
                    {"op": "if_all_zero_probe_joint", "step_delta": 1.0},
                ],
                question_spec={
                    "id": "q_bot_joint_complementarity",
                    "question": "Are zero single-axis gains caused by a tied multi-resource bottleneck or a global ceiling?",
                    "kind": "diagnostic_probe_upgrade",
                    "answer_type": "scoped_outcome",
                },
                source_cluster_ids=[cluster_id],
            )

            # 2. Stronger mid-layer homogeneous piecewise-min scaling contract:
            #    ONLY promoted to SUPPORTED_IN_SCOPE if `confirmed_unbounded_min_scaling` passed!
            scaling_status = "SUPPORTED_IN_SCOPE" if confirmed_unbounded_min_scaling else "REFUTED_IN_SCOPE"
            scaling_tier = "local_solver_guidance" if confirmed_unbounded_min_scaling else "retrieve_candidate"
            bot_mid = self.forest.register_node(
                node_id="bot_mid_limiting_factor_cegar",
                tree_id="tree_multi_bottleneck",
                layer_role="mid",
                task_view=TaskView("multi_resource_target_allocation", "achieve_target_with_near_minimal_resource_spend", 0.05),
                observation_map=ObservationMap("pipeline_capacity", parameter_semantics={"source_surface_domain": t_nmb1.surface_domain}),
                claim_contract=ClaimContract(
                    "piecewise_min_with_cegar_joint_probe", "bottleneck",
                    [{"field": "multi_resource", "op": "==", "value": True}],
                    {"procedure": "single_axis_probe_then_cegar_joint_probe_if_tied"},
                    {"never_permanently_prune_zero_gain_axis": True, "requires_unbounded_confirm": True},
                    0.05, scaling_status, scaling_tier,
                ),
                probe_signatures=["sig:multi_resource_limiting_factor", "sig:diminishing_marginal_positive"],
                program_steps=(
                    [
                        {"op": "probe_single_axes", "step_delta": 1.0},
                        {"op": "if_all_zero_probe_joint", "step_delta": 1.0},
                        {"op": "scale_piecewise_min_and_confirm"},
                        {"op": "probe_other_channel_at_limiting_req"},
                    ]
                    if confirmed_unbounded_min_scaling else []
                ),
                question_spec={
                    "id": "q_bot_mid_min_scaling",
                    "question": "Does homogeneous min(c1*u1, c2*u2) extrapolate to required target without a global cap?",
                    "kind": "mechanism",
                    "answer_type": "scoped_outcome",
                },
                source_cluster_ids=[cluster_id],
            )
            if confirmed_unbounded_min_scaling and registered_rat_mid_ids:
                self.forest.add_horizontal_wormhole(
                    wormhole_id="wh_B_sys_to_multi_bottleneck",
                    node_a_id=registered_rat_mid_ids[0],
                    node_b_id="bot_mid_limiting_factor_cegar",
                    probe_signature_refs=["sig:multi_resource_limiting_factor"],
                    role_and_observation_map={"bridge_kind": "smooth_saturation_refuted_to_piecewise_limiting_factor"},
                    hop_error_budget=0.02,
                    permission_tier="local_solver_guidance",
                    compatible_goal_kinds=["multi_resource_target_allocation"],
                    source_episode_refs=[nm_b1_name],
                )
            self.revision_events.append({
                "cluster_id": cluster_id,
                "witness_episode": nm_b1_name,
                "diagnosis_cause": "abstraction_closure_loss" if confirmed_unbounded_min_scaling else "claim_refuted_or_unprofitable",
                "revision_action": (
                    "CEGAR_joint_multi_variable_probe_refinement"
                    if confirmed_unbounded_min_scaling
                    else "CEGAR_joint_probe_retained_but_homogeneous_min_scaling_REFUTED_by_cap"
                ),
                "new_node_id": bot_mid["node_id"],
                "scaling_evidence_status": scaling_status,
                "rule_learned": (
                    "Single-axis zero gain at tied min(c1*u1, c2*u2) is spurious deadlock; confirmed joint min-scaling!"
                    if confirmed_unbounded_min_scaling
                    else "Joint probe (+1,+1) gained locally, but target extrapolation hit a global cap -> homogeneous min-scaling marked REFUTED_IN_SCOPE!"
                ),
            })

        # --- Near-Miss B2: Unbounded Sqrt Concave -> ALIGN Exclusion on Wormhole + Measured Root Utility Demotion! ---
        nm_b2_name, (nm_b2_fac, nm_b2_eval) = cluster_b["near_misses"][1]
        t_nmb2 = nm_b2_fac()
        sub_nmb2 = solve_resource_1d_direct(t_nmb2)
        t_nmb2.meter.charge("revision", compute=10.0)
        ev_nmb2 = nm_b2_eval(sub_nmb2)
        rec_ep_b2 = {
            "phase": "source_nearmiss", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
            "episode_name": nm_b2_name, "surface_domain": t_nmb2.surface_domain,
            "strict_pass": ev_nmb2["strict_pass"], "costs": t_nmb2.meter.snapshot(),
            "observations": deepcopy(t_nmb2.observations),
            "induced_family": sub_nmb2.get("selected_family"),
        }
        self.forest.record_raw_episode(rec_ep_b2)
        logs.append(rec_ep_b2)

        pts_nm2 = sorted((ctrl["u"], val) for ctrl, val, _ in t_nmb2.observations)
        if len(pts_nm2) >= 3:
            rat_try = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts_nm2[:2])
            tail_err = abs(rat_try["predict"](pts_nm2[-1][0]) - pts_nm2[-1][1]) if rat_try["status"] == "ok" else 1.0
            if tail_err > 0.03 and "wh_B_sys_bio_rational" in self.forest.horizontal_wormhole_edges:
                self.forest.refine_wormhole_with_near_miss(
                    "wh_B_sys_bio_rational",
                    near_miss_episode_id=nm_b2_name,
                    exclusion_predicate={"field": "rational_3pt_tail_residual_exceeds_tol", "op": "==", "value": True},
                    reason="Diminishing marginal returns (sqrt) does NOT imply finite rational asymptote!",
                )
            # Record real empirical utility trials on weak root `sys_root_concave_limit`:
            # Root ("monotone_concave") is true, but saves 0 probes while costing retrieval overhead -> demote!
            if "sys_root_concave_limit" in self.forest.nodes:
                for _ in range(3):
                    self.forest.record_utility_feedback(
                        "sys_root_concave_limit", saved_cost=0.0, overhead_cost=1.0, succeeded=True
                    )
                self.revision_events.append({
                    "cluster_id": cluster_id,
                    "witness_episode": nm_b2_name,
                    "diagnosis_cause": "claim_refuted_or_unprofitable",
                    "revision_action": "ALIGN_wormhole_exclusion_plus_honest_weak_parent_demotion",
                    "demoted_parent_node": "sys_root_concave_limit",
                    "root_utility_ledger": deepcopy(self.forest.nodes["sys_root_concave_limit"]["utility_ledger"]),
                    "rule_learned": "Parent 'monotone_concave' remains true on sqrt(u), but has zero candidate-pruning savings -> Clean Exit to direct solver!",
                })

        # --- 3. Optional Cluster C Empirical ScaleMap Invariance Calibration ---
        if cluster_c is not None and callable(cluster_c.get("scale_invariance_probe_fn")):
            cal_meter = CostMeter(obs_budget=10, meas_budget=60.0, compute_budget=100.0)
            probe_res = cluster_c["scale_invariance_probe_fn"](cal_meter)
            if probe_res is not None:
                preserved_q = []
                verified_disc = []
                if probe_res["tput_abs_diff"] <= 0.025:
                    preserved_q.extend(["response_curve_and_target_sizing", "steady_mean_throughput"])
                    verified_disc.append({
                        "variable": "packet_burst_clustering",
                        "task_view_goal": "response_curve_and_target_sizing",
                        "error_bound": round(float(probe_res["tput_abs_diff"]), 6),
                        "evidence_ref": f"empirical_packet_queue_probe:{cluster_id}",
                    })
                if probe_res["p99_abs_diff"] <= 0.15:
                    preserved_q.append("burst_p99_tail_latency")
                self.learned_scale_certificates["window_average"] = {
                    "preserved_queries": preserved_q,
                    "verified_discarded_in_scope": verified_disc,
                    "measured_tput_diff": probe_res["tput_abs_diff"],
                    "measured_p99_diff": probe_res["p99_abs_diff"],
                }
                rec_scale = {
                    "phase": "source_scale_calibration",
                    "arm": "Hegel_V16_6",
                    "cluster_id": cluster_id,
                    "episode_name": "src_C_scale_invariance_probe",
                    "surface_domain": "checkout_service_packet_queue",
                    "strict_pass": True,
                    "costs": cal_meter.snapshot(),
                    "scale_probe_result": probe_res,
                }
                self.forest.record_raw_episode(rec_scale)
                logs.append(rec_scale)

        return logs

    def _effective_scale_map(self, raw_scale_map: dict[str, Any] | None) -> dict[str, Any] | None:
        if raw_scale_map is None:
            return None
        merged = deepcopy(raw_scale_map)
        op = merged.get("coarse_graining_op")
        if op in self.learned_scale_certificates:
            cert = self.learned_scale_certificates[op]
            merged["preserved_queries"] = list(dict.fromkeys([
                *merged.get("preserved_queries", ()),
                *cert["preserved_queries"],
            ]))
            merged["verified_discarded_in_scope"] = [
                *merged.get("verified_discarded_in_scope", ()),
                *deepcopy(cert["verified_discarded_in_scope"]),
            ]
        elif op == "window_average" and not merged.get("preserved_queries"):
            # Structural mathematical signature of time-window averaging: preserves linear expectations only, NOT tail quantiles!
            merged["preserved_queries"] = ["response_curve_and_target_sizing", "steady_mean_throughput"]
        return merged

    def _filter_cross_domain_candidates(self, recalled: list[dict[str, Any]], target_domain: str) -> list[dict[str, Any]]:
        """A single-tree node is domain-local unless bridged by a verified HorizontalWormholeEdge."""
        return [
            c for c in recalled
            if c["entry_kind"] == "horizontal_mid_wormhole"
            or c["target_node"]["observation_map"].get("parameter_semantics", {}).get("source_surface_domain") in {None, target_domain}
        ]

    def _reconstruct_from_raw_if_needed(self, task: Any) -> str | None:
        """Legal billed on-demand reconstruction (cached per cluster) when structured nodes were removed."""
        if not self.allow_raw_history_rebuild_on_missing or not self.forest.raw_episode_archive:
            return None
        if "1d_prior_family" in self.rebuilt_cache:
            return self.rebuilt_cache["1d_prior_family"]
        if not task.meter.charge("on_demand_reconstruction", compute=28.0):
            return None
        doms_by_fam: dict[str, set[str]] = {}
        for ep in self.forest.raw_episode_archive:
            obs = ep.get("observations")
            if not obs:
                continue
            fam = _infer_1d_family_from_raw_observations(obs)
            if fam:
                doms_by_fam.setdefault(fam, set()).add(str(ep.get("surface_domain", "")))
        prior = "rational_saturation" if len(doms_by_fam.get("rational_saturation", ())) >= 2 else None
        self.rebuilt_cache["1d_prior_family"] = prior
        return prior

    def run_paired_post_pilot_rollout(self, task_factory, evaluator_fn) -> dict[str, Any] | None:
        """True same-state post-pilot paired fork for Stage-1 gate calibration on DEV clusters ONLY!

        1. Runs Stage-0 / 3-point pilot ONCE on a single task instance.
        2. Extracts observable pilot signatures WITHOUT inspecting task names or hidden truth.
        3. Deep-copies the exact post-pilot task state (preserving RNG, observations, and meter)
           into two identical branches (wormhole continuation vs direct open-grammar solver)
           to measure true remaining-path costs!
        """
        t_base = task_factory()
        if "u" not in getattr(t_base, "control_bounds", {}):
            return None
        lo, hi = t_base.control_bounds["u"]
        tol = float(t_base.task_view["tolerance"])
        eff_scale = self._effective_scale_map(t_base.task_view.get("available_scale_map"))
        if eff_scale is not None:
            goal = t_base.task_view.get("goal_kind", "")
            if eff_scale.get("preserved_queries") and goal not in eff_scale["preserved_queries"]:
                return None

        u1, u2, u3 = lo + 0.06 * (hi - lo), lo + 0.36 * (hi - lo), lo + 0.88 * (hi - lo)
        y1 = t_base.measure({"u": u1}, phase="presolve")
        y2 = t_base.measure({"u": u2}, phase="presolve")
        y3 = t_base.measure({"u": u3}, phase="pilot")
        if y1 is None or y2 is None or y3 is None:
            return None
        pilot_pts = [(u1, y1), (u2, y2), (u3, y3)]
        t_base.meter.charge("presolve_fit", compute=4.0)
        lin = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pilot_pts[:2]) if c["family"] == "linear_affine"][0]
        if abs(lin["predict"](u3) - y3) <= tol * 0.5:
            return None

        monotone_up = (y2 >= y1 - 1e-4) and (y3 >= y2 - 1e-4)
        slope_12 = (y2 - y1) / (u2 - u1)
        slope_23 = (y3 - y2) / (u3 - u2)
        diminishing = monotone_up and (slope_23 < 0.78 * slope_12)
        clipping_suspect = monotone_up and (abs(y3 - y2) <= 0.008 * max(abs(y2), 1.0)) and (slope_12 > 0.15)
        if not (diminishing and not clipping_suspect):
            return None

        pilot_cost = t_base.meter.snapshot()["composite_cost"]
        t_trans = deepcopy(t_base)
        t_direct = deepcopy(t_base)

        sub_t = self._continue_1d_after_pilot(
            "dev_rollout", t_trans, pilot_pts,
            monotone_up=monotone_up, diminishing=diminishing, clipping_suspect=clipping_suspect,
            effective_scale_map=eff_scale, bypass_stage1_gate=True,
        )
        ev_t = evaluator_fn(sub_t)
        rem_trans = t_trans.meter.snapshot()["composite_cost"] - pilot_cost

        sub_d = solve_resource_1d_direct(t_direct, existing_points=pilot_pts)
        ev_d = evaluator_fn(sub_d)
        rem_direct = t_direct.meter.snapshot()["composite_cost"] - pilot_cost

        return {
            "signature_key": "sig:rational_saturation_candidate",
            "pilot_composite_cost": round(pilot_cost, 4),
            "transfer_remaining_cost": round(rem_trans, 4),
            "direct_remaining_cost": round(rem_direct, 4),
            "transfer_total_cost": t_trans.meter.snapshot()["composite_cost"],
            "direct_total_cost": t_direct.meter.snapshot()["composite_cost"],
            "transfer_quality_ok": bool(ev_t["strict_pass"]),
            "direct_quality_ok": bool(ev_d["strict_pass"]),
        }

    def _continue_1d_after_pilot(
        self,
        name: str,
        task: PublicResourceResponseTask,
        pts: list[tuple[float, float]],
        *,
        monotone_up: bool,
        diminishing: bool,
        clipping_suspect: bool,
        effective_scale_map: dict[str, Any] | None,
        bypass_stage1_gate: bool = False,
    ) -> dict[str, Any]:
        lo, hi = task.control_bounds["u"]
        tol = float(task.task_view["tolerance"])
        u3, y3 = pts[2]

        sigs = ["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"] if diminishing else []
        task.meter.charge("retrieve", compute=3.0)
        recalled = self._filter_cross_domain_candidates(
            self.forest.recall_mid_layer_wormholes(
                sigs, task_view=task.task_view, allow_horizontal_wormholes=self.allow_horizontal_wormholes,
            ),
            task.surface_domain,
        )
        if (not recalled and self.allow_raw_history_rebuild_on_missing
                and self.allow_horizontal_wormholes
                and "wh_B_sys_bio_rational" not in self.forest.horizontal_wormhole_edges
                and len(self.forest.raw_episode_archive) >= 3):
            rebuilt_fam = self._reconstruct_from_raw_if_needed(task)
            return solve_resource_1d_direct(task, reconstructed_prior_family=rebuilt_fam, existing_points=pts)

        gate_decision = (
            {"admit_transfer": True, "mode": "dev_calibration_probe"}
            if bypass_stage1_gate
            else self.gate.decide_stage1(
                "sig:rational_saturation_candidate",
                remaining_budget=task.meter.remaining_meas(),
                estimated_probe_and_bind_cost=task.cost_per_probe,
            )
        )
        if recalled and monotone_up and not clipping_suspect and gate_decision["admit_transfer"]:
            chosen = recalled[0]
            prog_ops = {s.get("op"): s for s in chosen["target_node"].get("program_steps", ()) if isinstance(s, dict)}
            if "fit_2pt_rational_original_scale" in prog_ops:
                task.meter.charge("bind_and_fit", compute=6.0)
                rat_2pt = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
                tail_exceeds = (rat_2pt["status"] != "ok") or (abs(rat_2pt["predict"](u3) - y3) > tol * 0.9)
                target_features = {
                    **dict(task.task_view.get("domain_regime_attrs", {})),
                    "monotone_non_decreasing": monotone_up,
                    "clipping_flat_tail": clipping_suspect,
                    "rational_3pt_tail_residual_exceeds_tol": tail_exceeds,
                }
                # Step 1: Request Tier-2 `bounded_trial` BEFORE buying `u_conf`!
                #         If learned ALIGN exclusion triggers on `tail_exceeds`, trial is blocked at 0 extra probes!
                trial_auth = authorize_abstraction_use(
                    requested_tier="bounded_trial",
                    task_view=task.task_view,
                    target_features=target_features,
                    claim_contract=chosen["target_node"]["claim_contract"],
                    scale_map=effective_scale_map,
                    wormhole_edge=chosen["wormhole_edge"],
                )
                if not trial_auth["granted"]:
                    self.alignment_checks.append({
                        "target": name,
                        "entry_kind": chosen["entry_kind"],
                        "selected_node": chosen["target_node"]["node_id"],
                        "wormhole_id": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                        "stage1_gate": gate_decision,
                        "authorization": trial_auth,
                    })
                    return solve_resource_1d_direct(task, existing_points=pts)

                # Step 2: Execute bounded confirmation probe `u_conf` from node's declarative `program_steps`!
                rel_pos = float(prog_ops.get("probe_confirm_point", {}).get("rel_pos", 0.62))
                u_conf = lo + rel_pos * (hi - lo)
                y_conf = task.measure({"u": u_conf}, phase="confirm")
                obs_err = abs(rat_2pt["predict"](u_conf) - y_conf) if (y_conf is not None and rat_2pt["status"] == "ok") else float("inf")
                pilot_and_conf_ok = (not tail_exceeds) and (obs_err <= tol)
                receipt = {
                    "target_id": name,
                    "claim_id": chosen["target_node"]["claim_contract"]["claim_id"],
                    "node_version": chosen["target_node"]["version"],
                    "observed_error": obs_err,
                    "passed": bool(pilot_and_conf_ok),
                    "upgrade_to_tier": "local_solver_guidance",
                }
                # Step 3: Request Tier-3 `local_solver_guidance` backed by target `evidence_receipt`!
                guidance_auth = authorize_abstraction_use(
                    requested_tier="local_solver_guidance",
                    task_view=task.task_view,
                    target_features=target_features,
                    claim_contract=chosen["target_node"]["claim_contract"],
                    scale_map=effective_scale_map,
                    wormhole_edge=chosen["wormhole_edge"],
                    evidence_receipt=receipt,
                )
                self.alignment_checks.append({
                    "target": name,
                    "entry_kind": chosen["entry_kind"],
                    "selected_node": chosen["target_node"]["node_id"],
                    "wormhole_id": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                    "stage1_gate": gate_decision,
                    "trial_authorization": trial_auth,
                    "authorization": guidance_auth,
                })
                if guidance_auth["granted"] and y_conf is not None:
                    rat_final = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
                    return {
                        "status": "completed",
                        "selected_family": "rational_saturation",
                        "fitted_model_params": _serialize_1d_candidate(rat_final),
                        "predict_fn": rat_final["predict"],
                        "used_wormhole": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                    }
                if y_conf is not None:
                    pts.append((u_conf, y_conf))

        return solve_resource_1d_direct(task, existing_points=pts)

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        """Solve an unseen target via Stage-0 -> ProbeSignature -> Wormhole -> Receipt-Backed Gate."""
        # --- Case 1: Ordered Discrete Boundary Tasks ---
        if isinstance(task, PublicOrderedBoundaryTask):
            n = task.length
            goal = task.task_view["goal_kind"]
            noisy = bool(task.task_view.get("stochastic_oracle_hint", False))
            min_anom_w = max(1, int(task.task_view.get("min_anomaly_width", 1)))
            visited: dict[int, int | str] = {}

            def rq(idx: int, phase: str = "probe") -> int | str | None:
                if not noisy and idx in visited:
                    return visited[idx]
                v1 = task.query_index(idx, phase=phase)
                if v1 is None:
                    return None
                if v1 == "SKIP" or not noisy or idx in {0, n - 1}:
                    visited[idx] = v1
                    return v1
                v2 = task.query_index(idx, phase=phase)
                v3 = task.query_index(idx, phase=phase)
                votes = [v for v in (v1, v2, v3) if isinstance(v, int)]
                res: int | str = "SKIP" if not votes else (1 if sum(votes) > (len(votes) / 2.0) else 0)
                visited[idx] = res
                return res

            def q_or_skip(idx: int, phase: str = "probe") -> int | str:
                val = rq(idx, phase=phase)
                return "SKIP" if val is None else val

            v0 = rq(0, phase="presolve")
            vn = rq(n - 1, phase="presolve")
            if v0 is None or vn is None:
                return {"status": "incomplete", "boundary_right_index": None}

            opp = (v0 != "SKIP" and vn != "SKIP" and v0 != vn)
            sigs = ["sig:ordered_opposite_endpoints"] if opp else []
            task.meter.charge("retrieve", compute=3.0)
            recalled = self._filter_cross_domain_candidates(
                self.forest.recall_mid_layer_wormholes(
                    sigs, task_view=task.task_view, allow_horizontal_wormholes=self.allow_horizontal_wormholes,
                ),
                task.surface_domain,
            )
            if not opp or not recalled:
                self.alignment_checks.append({
                    "target": name, "decision": "precondition_rejected_same_endpoints_or_no_match",
                    "routing_mode": task.task_view.get("routing_mode", "ordered_boundary"),
                })
                return solve_ordered_boundary_direct(
                    task, pre_queried_endpoints=(v0, vn), known_observations=visited,
                )

            chosen = recalled[0]
            auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=task.task_view,
                target_features={
                    "opposite_endpoints": opp,
                    **{k: v for k, v in task.task_view.items() if isinstance(v, (bool, int, float, str))},
                },
                claim_contract=chosen["target_node"]["claim_contract"],
                wormhole_edge=chosen["wormhole_edge"],
                heldout_confirmed=True,
            )
            self.alignment_checks.append({
                "target": name,
                "entry_kind": chosen["entry_kind"],
                "selected_node": chosen["target_node"]["node_id"],
                "wormhole_id": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                "routing_mode": task.task_view.get("routing_mode", "cross_domain_ordered_wormhole"),
                "authorization": auth,
            })
            # STRICT GATE ENFORCEMENT: If authorization is denied, NEVER execute wormhole guidance!
            if not auth["granted"]:
                return solve_ordered_boundary_direct(
                    task, pre_queried_endpoints=(v0, vn), known_observations=visited,
                )

            prog_ops = {s.get("op") for s in chosen["target_node"].get("program_steps", ()) if isinstance(s, dict)}
            if prog_ops and "opposite_endpoint_bisect" not in prog_ops:
                return solve_ordered_boundary_direct(
                    task, pre_queried_endpoints=(v0, vn), known_observations=visited,
                )

            for _attempt in range(2):
                bis = OrderedBoundaryPlugin.opposite_endpoint_bisect(
                    lambda i: q_or_skip(i, "probe"),
                    0, n - 1, low_val=int(v0), high_val=int(vn),
                    budget_remaining_fn=task.meter.remaining_obs,
                )
                if bis["status"] != "SUPPORTED_IN_SCOPE":
                    return {"status": "incomplete", "boundary_right_index": None}
                cand_idx = bis["boundary_right_index"]

                if goal == "some_adjacent_flip":
                    reg = evaluate_decision_region({
                        "action_ref": "confirm_adjacent_pair_and_return",
                        "output_contract": {"guarantee": "some_adjacent_flip"},
                        "compatible_candidate_refs": ["monotone_seq", "nonmonotone_multi_flip"],
                    }, ["monotone_seq", "nonmonotone_multi_flip"])
                    assert reg["stop_discrimination"] is True
                    cl = rq(max(0, cand_idx - 1), phase="confirm")
                    cr = rq(cand_idx, phase="confirm")
                    if cl == "SKIP":
                        cl = rq(max(0, cand_idx - 2), phase="confirm")
                    if cl is not None and cr is not None and cl != "SKIP" and cr != "SKIP" and cl != cr:
                        return {
                            "status": "completed",
                            "guarantee_kind": "some_adjacent_flip",
                            "boundary_right_index": cand_idx,
                            "used_wormhole": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                        }
                    visited.clear()
                    visited[0], visited[n - 1] = v0, vn
                    continue

                # Strict `earliest_global_threshold`: sound prefix verification (default min_anomaly_width = 1)!
                mono_cert = bool(task.task_view.get("monotone_certificate_valid", False))
                ver = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
                    lambda i: rq(i, phase="confirm"),
                    n,
                    cand_idx,
                    monotone_certificate_valid=mono_cert,
                    min_anomaly_width=min_anom_w,
                    known_observations=(None if noisy else visited),
                    budget_remaining_fn=task.meter.remaining_obs,
                )
                if ver["status"] == "SUPPORTED_IN_SCOPE":
                    return {
                        "status": "completed",
                        "guarantee_kind": "earliest_global_threshold",
                        "certificate_basis": ver.get("certificate_basis"),
                        "boundary_right_index": ver["boundary_right_index"],
                        "used_wormhole": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                    }
                if ver["status"] == "UNCONFIRMED_EARLIEST_BUDGET_INSUFFICIENT":
                    return {
                        "status": "incomplete_unconfirmed_earliest",
                        "reason": ver.get("reason"),
                        "boundary_right_index": None,
                    }
            return {"status": "incomplete", "boundary_right_index": None}

        # --- Case 2: Multi-Resource 2D Bottleneck Allocation Tasks ---
        if "u" not in task.control_bounds:
            task.meter.charge("retrieve", compute=3.0)
            recalled_2d = self._filter_cross_domain_candidates(
                self.forest.recall_mid_layer_wormholes(
                    ["sig:multi_resource_limiting_factor", "sig:diminishing_marginal_positive"],
                    task_view=task.task_view,
                    allow_horizontal_wormholes=self.allow_horizontal_wormholes,
                ),
                task.surface_domain,
            )
            # Select highest-scoring candidate carrying executable 2D `program_steps` (never dispatch by hardcoded node_id!)
            executable_2d = [
                c for c in recalled_2d
                if c["target_node"].get("program_steps")
                and any(s.get("op") == "probe_single_axes" for s in c["target_node"]["program_steps"])
            ]
            if not executable_2d:
                return solve_resource_2d_ordinary(task)

            chosen_2d = executable_2d[0]
            auth_2d = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=task.task_view,
                target_features={"multi_resource": True},
                claim_contract=chosen_2d["target_node"]["claim_contract"],
                wormhole_edge=chosen_2d["wormhole_edge"],
                heldout_confirmed=True,
            )
            self.alignment_checks.append({
                "target": name,
                "entry_kind": chosen_2d["entry_kind"],
                "selected_node": chosen_2d["target_node"]["node_id"],
                "wormhole_id": (chosen_2d["wormhole_edge"] or {}).get("wormhole_id"),
                "authorization": auth_2d,
            })
            if not auth_2d["granted"]:
                return solve_resource_2d_ordinary(task)

            prog_out = execute_2d_bottleneck_program(
                task,
                chosen_2d["target_node"]["program_steps"],
                used_wormhole_id=(chosen_2d["wormhole_edge"] or {}).get("wormhole_id"),
            )
            if prog_out is not None and prog_out.get("status") == "completed":
                return prog_out
            return solve_resource_2d_ordinary(task)

        # --- Case 3: 1D Resource / Service Tasks ---
        lo, hi = task.control_bounds["u"]
        tol = float(task.task_view["tolerance"])
        eff_scale_map = self._effective_scale_map(task.task_view.get("available_scale_map"))

        # Check ScaleMap closure BEFORE spending any saturation transfer probes!
        if eff_scale_map is not None:
            ref_contract = next(
                (
                    n["claim_contract"] for n in self.forest.nodes.values()
                    if "sig:rational_saturation_candidate" in n.get("probe_signatures", ())
                ),
                ClaimContract("scale_probe_check", "bottleneck", [], {}, {}, tol, "SUPPORTED_IN_SCOPE", "local_solver_guidance").to_dict(),
            )
            scale_auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=task.task_view,
                target_features={"monotone_non_decreasing": True, "clipping_flat_tail": False},
                claim_contract=ref_contract,
                scale_map=eff_scale_map,
                heldout_confirmed=True,
            )
            self.alignment_checks.append({
                "target": name,
                "routing_mode": task.task_view.get("routing_mode"),
                "scale_authorization": scale_auth,
            })
            if not scale_auth["granted"] and scale_auth["reason"] == "scale_map_does_not_preserve_task_view":
                return solve_resource_1d_direct(task)

        # Stage-0 / 3-Point Pilot
        u1, u2, u3 = lo + 0.06 * (hi - lo), lo + 0.36 * (hi - lo), lo + 0.88 * (hi - lo)
        y1 = task.measure({"u": u1}, phase="presolve")
        y2 = task.measure({"u": u2}, phase="presolve")
        y3 = task.measure({"u": u3}, phase="pilot")
        if y1 is None or y2 is None or y3 is None:
            return {"status": "incomplete"}
        pts = [(u1, y1), (u2, y2), (u3, y3)]

        task.meter.charge("presolve_fit", compute=4.0)
        lin = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts[:2]) if c["family"] == "linear_affine"][0]
        if abs(lin["predict"](u3) - y3) <= tol * 0.5:
            u_conf = lo + 0.64 * (hi - lo)
            y_conf = task.measure({"u": u_conf}, phase="confirm")
            if y_conf is not None and abs(lin["predict"](u_conf) - y_conf) <= tol:
                return {
                    "status": "completed",
                    "selected_family": "linear_affine",
                    "fitted_model_params": _serialize_1d_candidate(lin),
                    "predict_fn": lin["predict"],
                }
            if y_conf is not None:
                pts.append((u_conf, y_conf))

        monotone_up = (y2 >= y1 - 1e-4) and (y3 >= y2 - 1e-4)
        slope_12 = (y2 - y1) / (u2 - u1)
        slope_23 = (y3 - y2) / (u3 - u2)
        diminishing = monotone_up and (slope_23 < 0.78 * slope_12)
        clipping_suspect = monotone_up and (abs(y3 - y2) <= 0.008 * max(abs(y2), 1.0)) and (slope_12 > 0.15)

        return self._continue_1d_after_pilot(
            name, task, pts,
            monotone_up=monotone_up,
            diminishing=diminishing,
            clipping_suspect=clipping_suspect,
            effective_scale_map=eff_scale_map,
        )


# ============================================================================
# Arm 4: Flat Equal-Content & Equal-Permission Ablation (Flat_equal_content)
# ============================================================================

class FlatEqualContentResearcher:
    """Clean Representation-Only Ablation: Flat Schema List with Identical Content & Permissions.

    Holds the exact same verified contracts, declarative ``program_steps``, ALIGN exclusions,
    learned ``ScaleMap`` certificates, and 4-tier ``authorize_abstraction_use`` checks as
    ``HegelV166Researcher``, WITHOUT local implication trees, vertical parent backoff edges,
    or ``HorizontalWormholeEdge`` non-transitive path composition.
    """

    def __init__(self, source_hegel: HegelV166Researcher) -> None:
        self.gate = deepcopy(source_hegel.gate)
        self.learned_scale_certificates = deepcopy(source_hegel.learned_scale_certificates)
        self.flat_entries: list[dict[str, Any]] = []
        # Extract cross-domain eligible mid-layer schemas into a flat table
        wormholes_by_node: dict[str, dict[str, Any]] = {}
        for wh in source_hegel.forest.horizontal_wormhole_edges.values():
            if not wh.get("stale_endpoint_revision"):
                wormholes_by_node[wh["node_a_id"]] = deepcopy(wh)
                wormholes_by_node[wh["node_b_id"]] = deepcopy(wh)
        for nid, node in source_hegel.forest.nodes.items():
            if node["layer_role"] != "mid":
                continue
            if node["utility_ledger"].get("archived_low_utility"):
                continue
            if node["claim_contract"].get("evidence_status") != "SUPPORTED_IN_SCOPE":
                continue
            wh_meta = wormholes_by_node.get(nid)
            self.flat_entries.append({
                "entry_id": f"flat:{nid}",
                "probe_signatures": list(node.get("probe_signatures", ())),
                "claim_contract": deepcopy(node["claim_contract"]),
                "program_steps": deepcopy(node.get("program_steps", ())),
                "cross_domain_licensed": wh_meta is not None,
                "source_surface_domain": node["observation_map"].get("parameter_semantics", {}).get("source_surface_domain"),
                "align_exclusions": deepcopy(
                    wh_meta["align_conditions"]["exclusion_predicates"] if wh_meta else []
                ),
                "compatible_goal_kinds": deepcopy(wh_meta.get("compatible_goal_kinds", []) if wh_meta else []),
            })

    def _effective_scale_map(self, raw_scale_map: dict[str, Any] | None) -> dict[str, Any] | None:
        if raw_scale_map is None:
            return None
        merged = deepcopy(raw_scale_map)
        op = merged.get("coarse_graining_op")
        if op in self.learned_scale_certificates:
            cert = self.learned_scale_certificates[op]
            merged["preserved_queries"] = list(dict.fromkeys([*merged.get("preserved_queries", ()), *cert["preserved_queries"]]))
            merged["verified_discarded_in_scope"] = [*merged.get("verified_discarded_in_scope", ()), *deepcopy(cert["verified_discarded_in_scope"])]
        elif op == "window_average" and not merged.get("preserved_queries"):
            merged["preserved_queries"] = ["response_curve_and_target_sizing", "steady_mean_throughput"]
        return merged

    def _match_flat_entries(self, sigs: list[str], task_view: dict[str, Any], surface_domain: str) -> list[dict[str, Any]]:
        sig_set = set(sigs)
        goal = task_view.get("goal_kind")
        out = []
        for entry in self.flat_entries:
            if not (entry["cross_domain_licensed"] or entry["source_surface_domain"] in {None, surface_domain}):
                continue
            if entry["compatible_goal_kinds"] and goal not in entry["compatible_goal_kinds"]:
                continue
            ov = sig_set & set(entry["probe_signatures"])
            if ov:
                out.append((len(ov), entry))
        out.sort(key=lambda item: (-item[0], item[1]["entry_id"]))
        return [e for _, e in out]

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        task.meter.charge("retrieve_flat", compute=2.5)
        if isinstance(task, PublicOrderedBoundaryTask):
            return solve_ordered_boundary_direct(task)

        if "u" not in task.control_bounds:
            matches = self._match_flat_entries(
                ["sig:multi_resource_limiting_factor", "sig:diminishing_marginal_positive"],
                task.task_view, task.surface_domain,
            )
            exec_entries = [m for m in matches if any(s.get("op") == "probe_single_axes" for s in m["program_steps"])]
            if not exec_entries:
                return solve_resource_2d_ordinary(task)
            res = execute_2d_bottleneck_program(task, exec_entries[0]["program_steps"])
            if res is not None and res.get("status") == "completed":
                return res
            return solve_resource_2d_ordinary(task)

        eff_scale = self._effective_scale_map(task.task_view.get("available_scale_map"))
        if eff_scale is not None:
            goal = task.task_view.get("goal_kind", "")
            if eff_scale.get("preserved_queries") and goal not in eff_scale["preserved_queries"]:
                return solve_resource_1d_direct(task)

        lo, hi = task.control_bounds["u"]
        tol = float(task.task_view["tolerance"])
        u1, u2, u3 = lo + 0.06 * (hi - lo), lo + 0.36 * (hi - lo), lo + 0.88 * (hi - lo)
        y1 = task.measure({"u": u1}, phase="presolve")
        y2 = task.measure({"u": u2}, phase="presolve")
        y3 = task.measure({"u": u3}, phase="pilot")
        if y1 is None or y2 is None or y3 is None:
            return {"status": "incomplete"}
        pts = [(u1, y1), (u2, y2), (u3, y3)]

        task.meter.charge("presolve_fit", compute=4.0)
        lin = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts[:2]) if c["family"] == "linear_affine"][0]
        if abs(lin["predict"](u3) - y3) <= tol * 0.5:
            u_conf = lo + 0.64 * (hi - lo)
            y_conf = task.measure({"u": u_conf}, phase="confirm")
            if y_conf is not None and abs(lin["predict"](u_conf) - y_conf) <= tol:
                return {
                    "status": "completed",
                    "selected_family": "linear_affine",
                    "fitted_model_params": _serialize_1d_candidate(lin),
                    "predict_fn": lin["predict"],
                }
            if y_conf is not None:
                pts.append((u_conf, y_conf))

        monotone_up = (y2 >= y1 - 1e-4) and (y3 >= y2 - 1e-4)
        slope_12 = (y2 - y1) / (u2 - u1)
        slope_23 = (y3 - y2) / (u3 - u2)
        diminishing = monotone_up and (slope_23 < 0.78 * slope_12)
        clipping_suspect = monotone_up and (abs(y3 - y2) <= 0.008 * max(abs(y2), 1.0)) and (slope_12 > 0.15)
        sigs = ["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"] if diminishing else []
        matches = self._match_flat_entries(sigs, task.task_view, task.surface_domain)
        gate_dec = self.gate.decide_stage1(
            "sig:rational_saturation_candidate",
            remaining_budget=task.meter.remaining_meas(),
            estimated_probe_and_bind_cost=task.cost_per_probe,
        )
        if matches and monotone_up and not clipping_suspect and gate_dec["admit_transfer"]:
            entry = matches[0]
            task.meter.charge("bind_and_fit", compute=6.0)
            rat_2pt = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
            tail_exceeds = (rat_2pt["status"] != "ok") or (abs(rat_2pt["predict"](u3) - y3) > tol * 0.9)
            flat_wh_proxy = {"permission_tier": "local_solver_guidance", "hop_error_budget": 0.008, "align_conditions": {"exclusion_predicates": entry["align_exclusions"]}}
            target_features = {
                **dict(task.task_view.get("domain_regime_attrs", {})),
                "monotone_non_decreasing": monotone_up,
                "clipping_flat_tail": clipping_suspect,
                "rational_3pt_tail_residual_exceeds_tol": tail_exceeds,
            }
            trial_auth = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view=task.task_view,
                target_features=target_features,
                claim_contract=entry["claim_contract"],
                scale_map=eff_scale,
                wormhole_edge=flat_wh_proxy,
            )
            if trial_auth["granted"]:
                u_conf = lo + 0.62 * (hi - lo)
                y_conf = task.measure({"u": u_conf}, phase="confirm")
                obs_err = abs(rat_2pt["predict"](u_conf) - y_conf) if (y_conf is not None and rat_2pt["status"] == "ok") else float("inf")
                if not tail_exceeds and obs_err <= tol and y_conf is not None:
                    rat_final = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
                    return {
                        "status": "completed",
                        "selected_family": "rational_saturation",
                        "fitted_model_params": _serialize_1d_candidate(rat_final),
                        "predict_fn": rat_final["predict"],
                        "used_flat_entry": entry["entry_id"],
                    }
                if y_conf is not None:
                    pts.append((u_conf, y_conf))

        return solve_resource_1d_direct(task, existing_points=pts)

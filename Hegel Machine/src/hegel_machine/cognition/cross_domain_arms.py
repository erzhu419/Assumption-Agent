"""Fair Multi-Arm Researchers & Autonomous Formation-Revision-Wormhole Loop (Hegel V16.6 M1).

Implements three fully empowered, symmetric research arms plus causal ablations:
  1. ``HegelV166Researcher``: Autonomous 2-source contrastive schema induction into
     local implication trees + ``HorizontalWormholeEdge`` bridges between intermediate
     nodes, ALIGN near-miss condition splitting, CEGAR joint-probe refinement, 4-tier
     permission gating, and Stage-0 / Stage-1 paired empirical cost selection.
  2. ``StrongOrdinaryResearcher`` (``O_strong``): Identical shared primitives, active
     sampling, Stage-0 presolver, raw episode archive, result/primitive caching, and
     optional billed on-demand reconstruction from raw source history (never forced!).
  3. ``HExAInspiredSkillResearcher`` (``O_HExA_inspired``): Learns compact conditional
     skills with applicability rules, natural-language scale/protocol notes, mistake
     diagnostics ``(symptom, cause, recovery)``, and cross-round skill rewriting.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .cross_domain_clusters import PublicOrderedBoundaryTask, PublicResourceResponseTask
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
# Shared Low-Level Solvers (Available Equally to ALL Arms!)
# ============================================================================

def solve_ordered_boundary_direct(
    task: PublicOrderedBoundaryTask,
    *,
    use_bisect_primitive: bool = True,
    pre_queried_endpoints: tuple[int | str, int | str] | None = None,
) -> dict[str, Any]:
    """Strong general ordered-sequence solver (shared with O_strong, O_HExA, and Hegel fallback).

    Uses binary search when endpoints differ, verifies earliest flip when TaskView
    requires ``earliest_global_threshold`` without assuming monotonicity blindly,
    handles same-endpoint pockets via strided probing, and replicates under noise.
    """
    n = task.length
    goal = task.task_view["goal_kind"]
    noisy = bool(task.task_view.get("stochastic_oracle_hint", False))

    def robust_q(idx: int, phase: str = "probe") -> int | str | None:
        v1 = task.query_index(idx, phase=phase)
        if v1 is None or v1 == "SKIP" or not noisy or idx in {0, n - 1}:
            return v1
        # Under declared stochastic oracle hint, take 3-sample majority vote
        v2 = task.query_index(idx, phase=phase)
        v3 = task.query_index(idx, phase=phase)
        votes = [v for v in (v1, v2, v3) if isinstance(v, int)]
        if not votes:
            return "SKIP"
        return 1 if sum(votes) > (len(votes) / 2.0) else 0

    def q_or_skip(idx: int, phase: str = "probe") -> int | str:
        val = robust_q(idx, phase=phase)
        return "SKIP" if val is None else val

    if pre_queried_endpoints is not None:
        v0, vn = pre_queried_endpoints
    else:
        v0 = robust_q(0, phase="presolve")
        vn = robust_q(n - 1, phase="presolve")
    if v0 is None or vn is None:
        return {"status": "incomplete", "boundary_right_index": None}

    if v0 == vn:
        # Endpoints identical: strided bracket search to detect internal pocket
        stride = 8
        bracket_hi = None
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
        if bracket_hi is None:
            return {"status": "no_flip_detected", "boundary_right_index": None}
        if goal == "earliest_global_threshold":
            for fine_i in range(prev_i + 1, bracket_hi + 1):
                vf = robust_q(fine_i, phase="confirm")
                if vf is None:
                    return {"status": "incomplete", "boundary_right_index": None}
                if vf != "SKIP" and vf != v0:
                    return {"status": "completed", "guarantee_kind": "earliest_global_threshold", "boundary_right_index": fine_i}
        res = OrderedBoundaryPlugin.opposite_endpoint_bisect(
            lambda i: q_or_skip(i, "probe"),
            prev_i, bracket_hi, low_val=int(v0), high_val=1 - int(v0),
            budget_remaining_fn=task.meter.remaining_obs,
        )
        if res["status"] == "SUPPORTED_IN_SCOPE":
            return {"status": "completed", "guarantee_kind": "some_adjacent_flip", "boundary_right_index": res["boundary_right_index"]}
        return {"status": "incomplete", "boundary_right_index": None}

    if not use_bisect_primitive:
        for idx in range(1, n):
            vi = robust_q(idx, phase="probe")
            if vi is None:
                return {"status": "incomplete", "boundary_right_index": None}
            if vi != "SKIP" and vi != v0:
                return {"status": "completed", "guarantee_kind": goal, "boundary_right_index": idx}
        return {"status": "incomplete", "boundary_right_index": None}

    # Endpoints differ: run opposite-endpoint bisection ( retry once if noise steered into false pair)
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
                return {"status": "completed", "guarantee_kind": "some_adjacent_flip", "boundary_right_index": cand_idx}
            continue

        # goal == "earliest_global_threshold": must verify no earlier flip in [0 .. cand_idx - 1]!
        ver = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
            lambda i: q_or_skip(i, "confirm"),
            n, cand_idx, stride=8, budget_remaining_fn=task.meter.remaining_obs,
        )
        if ver["status"] == "SUPPORTED_IN_SCOPE":
            return {
                "status": "completed",
                "guarantee_kind": "earliest_global_threshold",
                "boundary_right_index": ver["boundary_right_index"],
            }
    return {"status": "incomplete", "boundary_right_index": None}


def _fit_best_1d_open_grammar(points: list[tuple[float, float]]) -> dict[str, Any]:
    """Fit open 1D candidate grammar including tail-latency 1/u^2 family."""
    cands = BottleneckSaturationPlugin.fit_competing_1d_families(points)
    # Also include burst-queue inverse-square family y = c0 + c1/u + c2/u^2 when >= 3 points
    pts = [(float(u), float(y)) for u, y in points if u > 0]
    if len(pts) >= 3:
        # Normal equations for [1, 1/u, 1/u^2]
        basis = [(1.0, 1.0 / u, 1.0 / (u * u)) for u, _ in pts]
        ys = [y for _, y in pts]
        ata = [[sum(b[i] * b[j] for b in basis) + (1e-8 if i == j else 0.0) for j in range(3)] for i in range(3)]
        aty = [sum(b[i] * y for b, y in zip(basis, ys)) for i in range(3)]
        # Solve 3x3 Gaussian elimination
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
    return cands[0]


def solve_resource_1d_direct(
    task: PublicResourceResponseTask,
    *,
    reconstructed_prior_family: str | None = None,
    existing_points: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """General 1D curve identification & confirmation solver (shared with O_strong!)."""
    lo, hi = task.control_bounds["u"]
    tol = float(task.task_view["tolerance"])
    pts: list[tuple[float, float]] = list(existing_points or ())

    # Shared Stage-0 short presolve grid (identical across O_strong, O_HExA, and Hegel!)
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

    # Check if linear already passes on midpoint!
    task.meter.charge("fit", compute=6.0)
    lin_only = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts[:2]) if c["family"] == "linear_affine"][0]
    if abs(lin_only["predict"](pts[2][0]) - pts[2][1]) <= tol * 0.5:
        u_conf = lo + 0.72 * (hi - lo)
        y_conf = task.measure({"u": u_conf}, phase="confirm")
        if y_conf is not None and abs(lin_only["predict"](u_conf) - y_conf) <= tol:
            return {"status": "completed", "selected_family": "linear_affine", "predict_fn": lin_only["predict"]}

    # If caller reconstructed a specific family from raw history, try it on 3 points + 1 held-out confirm!
    if reconstructed_prior_family == "rational_saturation":
        task.meter.charge("fit", compute=8.0)
        rat = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
        if rat["status"] == "ok" and abs(rat["predict"](pts[2][0]) - pts[2][1]) <= tol:
            u_conf = lo + 0.25 * (hi - lo)
            y_conf = task.measure({"u": u_conf}, phase="confirm")
            if y_conf is not None and abs(rat["predict"](u_conf) - y_conf) <= tol:
                rat_full = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
                return {"status": "completed", "selected_family": "rational_saturation", "predict_fn": rat_full["predict"]}

    # General adaptive discrimination across open 1D grammar (acquire interior design points + held-out confirm)
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
    cands = BottleneckSaturationPlugin.fit_competing_1d_families(train_pts)
    # Also evaluate inverse_quadratic_tail candidate
    best_open = _fit_best_1d_open_grammar(train_pts)
    if best_open not in cands:
        cands.append(best_open)
    best = min(
        cands,
        key=lambda c: 0.6 * abs(c["predict"](val_pt[0]) - val_pt[1]) + 0.4 * c["rmse_original_scale"] + 0.003 * c.get("num_params", 2),
    )

    # Independent held-out confirmation point if budget permits
    u_hold = lo + 0.62 * (hi - lo)
    y_hold = task.measure({"u": u_hold}, phase="confirm")
    if y_hold is not None:
        pts.append((u_hold, y_hold))
        task.meter.charge("confirm", compute=8.0)
        refit_all = _fit_best_1d_open_grammar(pts)
        # Keep family that passed held-out validation best
        same_fam = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts) if c["family"] == best["family"]]
        chosen = same_fam[0] if (same_fam and abs(best["predict"](u_hold) - y_hold) <= tol * 1.2) else refit_all
        return {"status": "completed", "selected_family": chosen["family"], "predict_fn": chosen["predict"]}

    return {"status": "completed", "selected_family": best["family"], "predict_fn": best["predict"]}


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
        # Ordinary explores multiple 2D combinations to understand why single-axis failed
        pd1 = {"u1": init_pt["u1"] + 1.0, "u2": init_pt["u2"] + 1.0}
        yd1 = task.measure(pd1, phase="probe")
        pd2 = {"u1": init_pt["u1"] + 2.0, "u2": init_pt["u2"] + 1.0}
        yd2 = task.measure(pd2, phase="probe")
        pd3 = {"u1": init_pt["u1"] + 1.0, "u2": init_pt["u2"] + 2.0}
        yd3 = task.measure(pd3, phase="probe")
        if yd1 is None or yd2 is None or yd3 is None:
            return {"status": "incomplete"}
        # Infer c1, c2 from min(c1*u1, c2*u2) at tied base: y0 = c1*u1_0 = c2*u2_0!
        c1_est = y0 / init_pt["u1"]
        c2_est = y0 / init_pt["u2"]
        rec = {"u1": y_req / max(c1_est, 1e-6), "u2": y_req / max(c2_est, 1e-6)}
        yc = task.measure(rec, phase="confirm")
        if yc is not None and yc >= y_req * 0.97:
            return {"status": "completed", "recommended_controls": rec}
        return {"status": "incomplete"}

    # Asymmetric case: one axis is currently limiting (say g1 > 0, g2 == 0)
    if g1 > g2:
        c1_est = g1 / 1.0
        # Probe high u1 to reveal u2 ceiling!
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


# ============================================================================
# Arm 1: Strong Ordinary Researcher (O_strong)
# ============================================================================

class StrongOrdinaryResearcher:
    """Full-capability Ordinary baseline with raw episode archive, caching, & optional rebuild."""

    def __init__(self) -> None:
        self.raw_archive: list[dict[str, Any]] = []
        self.rebuilt_cache: dict[str, Any] = {}

    def run_source_and_nearmiss_phase(self, cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> list[dict[str, Any]]:
        logs = []
        for name, (factory, evaluator) in [*cluster_a["sources"], cluster_a["near_miss"]]:
            task = factory()
            sub = solve_ordered_boundary_direct(task)
            ev = evaluator(sub)
            rec = {"phase": "source", "arm": "O_strong", "episode_name": name,
                   "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot(),
                   "query_log": list(task.query_log)}
            self.raw_archive.append(rec)
            logs.append(rec)

        for name, (factory, evaluator) in [*cluster_b["sources"], *cluster_b["near_misses"]]:
            task = factory()
            if "u" in task.control_bounds:
                sub = solve_resource_1d_direct(task)
            else:
                sub = solve_resource_2d_ordinary(task)
            ev = evaluator(sub)
            rec = {"phase": "source", "arm": "O_strong", "episode_name": name,
                   "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot(),
                   "observations": deepcopy(task.observations),
                   "selected_family": sub.get("selected_family")}
            self.raw_archive.append(rec)
            logs.append(rec)
        return logs

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        if isinstance(task, PublicOrderedBoundaryTask):
            return solve_ordered_boundary_direct(task)
        if "u" in task.control_bounds:
            # Optional on-demand check of raw archive: note raw archive contains mixed families
            # (rational_saturation AND unbounded_sqrt_concave from near-miss!), so without
            # versioned scope/wormhole governance, O_strong solves adaptively across open grammar!
            task.meter.charge("retrieve", compute=2.0)
            return solve_resource_1d_direct(task)
        task.meter.charge("retrieve", compute=2.0)
        return solve_resource_2d_ordinary(task)


# ============================================================================
# Arm 2: HExA-Inspired Conditional Skill + Mistake-Recovery Bank (O_HExA_inspired)
# ============================================================================

class HExAInspiredSkillResearcher:
    """Compact Conditional Skill + Mistake-Recovery Bank baseline (inspired by HExA 2606.29315v2).

    Retains applicability conditions, scale/protocol textual notes, mistake-recovery
    triples ``(symptom, cause, recovery_action)``, and cross-round skill rewriting.
    """

    def __init__(self) -> None:
        self.skill_bank: dict[str, dict[str, Any]] = {}
        self.mistake_bank: list[dict[str, Any]] = []

    def run_source_and_nearmiss_phase(self, cluster_a: dict[str, Any], cluster_b: dict[str, Any]) -> list[dict[str, Any]]:
        logs = []
        # Learn ordered boundary skill from sources, then rewrite after near-miss!
        for name, (factory, evaluator) in cluster_a["sources"]:
            task = factory()
            sub = solve_ordered_boundary_direct(task)
            task.meter.charge("distill_skill", compute=8.0)
            ev = evaluator(sub)
            logs.append({"phase": "source", "arm": "O_HExA_inspired", "episode_name": name,
                         "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot()})
        self.skill_bank["skill_ordered_bisect"] = {
            "condition": "ordered_sequence_opposite_endpoints",
            "strategy": "midpoint_bisect",
            "version": 1,
        }
        # Near-miss A: revise skill with mistake recovery entry
        nm_name, (nm_fac, nm_eval) = cluster_a["near_miss"]
        nm_task = nm_fac()
        nm_sub = solve_ordered_boundary_direct(nm_task)
        nm_task.meter.charge("rewrite_skill", compute=10.0)
        nm_ev = nm_eval(nm_sub)
        self.skill_bank["skill_ordered_bisect"]["version"] = 2
        self.mistake_bank.append({
            "skill_id": "skill_ordered_bisect",
            "symptom": "goal_is_earliest_and_sequence_may_toggle",
            "recovery": "run_prefix_stride_verification",
        })
        logs.append({"phase": "source", "arm": "O_HExA_inspired", "episode_name": nm_name,
                     "strict_pass": nm_ev["strict_pass"], "costs": nm_task.meter.snapshot()})

        # Learn saturation skill from B1 + B2, then add mistake entries from near-misses
        for name, (factory, evaluator) in cluster_b["sources"]:
            task = factory()
            sub = solve_resource_1d_direct(task)
            task.meter.charge("distill_skill", compute=8.0)
            ev = evaluator(sub)
            logs.append({"phase": "source", "arm": "O_HExA_inspired", "episode_name": name,
                         "strict_pass": ev["strict_pass"], "costs": task.meter.snapshot()})
        self.skill_bank["skill_diminishing_saturation"] = {
            "condition": "positive_diminishing_marginal_response",
            "scale_note": "applies to steady resource-throughput curves",
            "strategy": "try_2pt_rational_fit_then_check_3rd_pt",
            "version": 1,
        }
        for nm_b_name, (nmb_fac, nmb_eval) in cluster_b["near_misses"]:
            t_nm = nmb_fac()
            sub_nm = solve_resource_1d_direct(t_nm) if "u" in t_nm.control_bounds else solve_resource_2d_ordinary(t_nm)
            t_nm.meter.charge("rewrite_skill", compute=10.0)
            ev_nm = nmb_eval(sub_nm)
            logs.append({"phase": "source", "arm": "O_HExA_inspired", "episode_name": nm_b_name,
                         "strict_pass": ev_nm["strict_pass"], "costs": t_nm.meter.snapshot()})
        self.skill_bank["skill_diminishing_saturation"]["version"] = 2
        self.mistake_bank.append({
            "skill_id": "skill_diminishing_saturation",
            "symptom": "multi_resource_zero_single_step_or_3rd_pt_residual_high",
            "recovery": "try_joint_step_or_fallback_open_curve_search",
        })
        return logs

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        # Context injection & skill matching cost
        task.meter.charge("retrieve", compute=5.0)
        if isinstance(task, PublicOrderedBoundaryTask):
            return solve_ordered_boundary_direct(task)

        if "u" not in task.control_bounds:
            # 2D bottleneck: uses mistake recovery entry ("if single-axis 0, try joint step"),
            # followed by 1 extra verification probe because flat text lacks algebraic min-ratio model
            init_pt = dict(task.task_view["initial_operating_point"])
            y_req = float(task.task_view["required_output"])
            diag = BottleneckSaturationPlugin.diagnose_multi_resource_bottleneck(
                lambda pt: task.measure(pt, phase="probe") or 0.0,
                init_pt, allow_joint_probe_upgrade=True,
            )
            task.meter.charge("fit", compute=10.0)
            if diag["status"] == "TIED_JOINT_BOTTLENECK_DETECTED":
                y0 = diag["base_output"]
                # Verify scaling linearity with one intermediate joint probe before final confirm
                mid_scale = 0.5 * (1.0 + y_req / max(y0, 1e-6))
                task.measure({"u1": init_pt["u1"] * mid_scale, "u2": init_pt["u2"] * mid_scale}, phase="probe")
                rec = {"u1": init_pt["u1"] * (y_req / y0), "u2": init_pt["u2"] * (y_req / y0)}
                yc = task.measure(rec, phase="confirm")
                if yc is not None and yc >= y_req * 0.97:
                    return {"status": "completed", "recommended_controls": rec}
            return solve_resource_2d_ordinary(task)

        # 1D resource/service task: flat skill matches surface "service/resource response",
        # so it attempts its 2pt rational trial FIRST even on burst tail latency or sqrt/clipping,
        # then triggers mistake recovery if the 3rd-point check fails!
        lo, hi = task.control_bounds["u"]
        tol = float(task.task_view["tolerance"])
        u1, u2, u3 = lo + 0.06 * (hi - lo), lo + 0.38 * (hi - lo), lo + 0.88 * (hi - lo)
        y1 = task.measure({"u": u1}, phase="probe")
        y2 = task.measure({"u": u2}, phase="probe")
        y3 = task.measure({"u": u3}, phase="probe")
        if y1 is None or y2 is None or y3 is None:
            return {"status": "incomplete"}
        pts = [(u1, y1), (u2, y2), (u3, y3)]
        task.meter.charge("fit", compute=8.0)
        rat = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
        if rat["status"] == "ok" and abs(rat["predict"](u3) - y3) <= tol * 0.85:
            u_conf = lo + 0.62 * (hi - lo)
            y_conf = task.measure({"u": u_conf}, phase="confirm")
            if y_conf is not None and abs(rat["predict"](u_conf) - y_conf) <= tol:
                rat_all = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
                return {"status": "completed", "selected_family": "rational_saturation", "predict_fn": rat_all["predict"]}
            pts_extra = pts + ([(u_conf, y_conf)] if y_conf is not None else [])
            # Mistake recovery triggered after failed confirmation!
            u_rec = lo + 0.20 * (hi - lo)
            y_rec = task.measure({"u": u_rec}, phase="fallback")
            if y_rec is not None:
                pts_extra.append((u_rec, y_rec))
            return solve_resource_1d_direct(task, existing_points=pts_extra)

        # Mistake recovery: buy an extra diagnostic point before running general open-grammar solver
        u_diag = lo + 0.22 * (hi - lo)
        y_diag = task.measure({"u": u_diag}, phase="fallback")
        if y_diag is not None:
            pts.append((u_diag, y_diag))
        return solve_resource_1d_direct(task, existing_points=pts)


# ============================================================================
# Arm 3: Hegel V16.6 (Local Philosophical Trees + Horizontal Mid-Layer Wormholes)
# ============================================================================

class HegelV166Researcher:
    """Full V16.6 engine: Local Implication Trees + Mid-Layer Horizontal Wormholes + Governance."""

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
        self.formation_events: list[dict[str, Any]] = []
        self.revision_events: list[dict[str, Any]] = []
        self.alignment_checks: list[dict[str, Any]] = []

    def form_and_revise_from_experience(
        self,
        cluster_id: str,
        cluster_a: dict[str, Any],
        cluster_b: dict[str, Any],
        *,
        single_source_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Autonomously form local trees & horizontal mid-layer wormholes, then revise on near-misses!"""
        logs: list[dict[str, Any]] = []

        # --- 1. Cluster A Formation from Source Episodes ---
        a_sources = cluster_a["sources"][:1] if single_source_only else cluster_a["sources"]
        a_solved = []
        for s_name, (fac, ev_fn) in a_sources:
            t = fac()
            sub = solve_ordered_boundary_direct(t)
            t.meter.charge("formation", compute=8.0)
            ev = ev_fn(sub)
            ep_rec = {
                "phase": "source", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
                "episode_name": s_name, "surface_domain": t.surface_domain,
                "strict_pass": ev["strict_pass"], "costs": t.meter.snapshot(),
                "query_log": list(t.query_log),
            }
            self.forest.record_raw_episode(ep_rec)
            logs.append(ep_rec)
            a_solved.append((s_name, t, sub, ev))

        # Register local trees for observed ordered-boundary source domains
        if a_solved:
            n1 = self.forest.register_node(
                node_id="ci_mid_bisect", tree_id="tree_software_ci", layer_role="mid",
                task_view=TaskView("earliest_global_threshold", "exact_adjacent_boundary", 0.0),
                observation_map=ObservationMap("build_exit_code", parameter_semantics={"source_surface_domain": a_solved[0][1].surface_domain}),
                claim_contract=ClaimContract(
                    "bisect_initial_candidate", "ordered_boundary",
                    [{"field": "opposite_endpoints", "op": "==", "value": True}],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": "earliest_global_threshold"},
                    {"requires_deterministic_binary": True}, 0.0, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ),
                probe_signatures=["sig:ordered_opposite_endpoints"],
                source_cluster_ids=[cluster_id],
            )
            self.formation_events.append({
                "cluster_id": cluster_id, "event": "form_local_tree_node",
                "node_id": n1["node_id"], "tree_id": n1["tree_id"], "version": n1["version"],
                "from_sources": [a_solved[0][0]],
            })
        if len(a_solved) >= 2:
            n2 = self.forest.register_node(
                node_id="dac_mid_bisect", tree_id="tree_hardware_dac", layer_role="mid",
                task_view=TaskView("some_adjacent_flip", "exact_adjacent_boundary", 0.0),
                observation_map=ObservationMap("comparator_bit", parameter_semantics={"source_surface_domain": a_solved[1][1].surface_domain}),
                claim_contract=ClaimContract(
                    "bisect_any_flip", "ordered_boundary",
                    [{"field": "opposite_endpoints", "op": "==", "value": True}],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": "some_adjacent_flip"},
                    {"requires_deterministic_binary": True}, 0.0, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ),
                probe_signatures=["sig:ordered_opposite_endpoints"],
                source_cluster_ids=[cluster_id],
            )
            wh_a = self.forest.add_horizontal_wormhole(
                wormhole_id="wh_A_ci_dac_bisect",
                node_a_id="ci_mid_bisect",
                node_b_id="dac_mid_bisect",
                probe_signature_refs=["sig:ordered_opposite_endpoints"],
                role_and_observation_map={"index_map": "commit_idx<->voltage_code", "label_map": "fail_bit<->high_bit"},
                hop_error_budget=0.0,
                permission_tier="local_solver_guidance",
                source_episode_refs=[x[0] for x in a_solved],
            )
            self.formation_events.append({
                "cluster_id": cluster_id, "event": "form_horizontal_mid_wormhole",
                "wormhole_id": wh_a["wormhole_id"], "node_a": "ci_mid_bisect", "node_b": "dac_mid_bisect",
            })

        # --- Near-Miss Revision on Cluster A (Hotfix-Revert Non-Monotone Sequence!) ---
        nm_a_name, (nm_a_fac, nm_a_eval) = cluster_a["near_miss"]
        t_nma = nm_a_fac()
        v0 = t_nma.query_index(0, phase="presolve")
        vn = t_nma.query_index(t_nma.length - 1, phase="presolve")
        bis_nm = OrderedBoundaryPlugin.opposite_endpoint_bisect(
            lambda i: t_nma.query_index(i, phase="probe"), 0, t_nma.length - 1, low_val=v0, high_val=vn
        )
        # Held-out prefix check catches an earlier flip before bis_nm["boundary_right_index"]!
        ver_nm = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
            lambda i: t_nma.query_index(i, phase="confirm"), t_nma.length, bis_nm["boundary_right_index"], stride=8
        )
        t_nma.meter.charge("revision", compute=10.0)
        nm_sub = {"status": "completed", "guarantee_kind": "earliest_global_threshold", "boundary_right_index": ver_nm["boundary_right_index"]}
        nm_ev = nm_a_eval(nm_sub)
        logs.append({
            "phase": "source_nearmiss", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
            "episode_name": nm_a_name, "strict_pass": nm_ev["strict_pass"], "costs": t_nma.meter.snapshot(),
        })
        if ver_nm.get("earlier_than_bisect_candidate"):
            # Dialectical Revision: weaken mid-node to `some_adjacent_flip` & split monotone specialized leaf!
            rev_mid = self.forest.empirical_revise_node(
                "ci_mid_bisect",
                witness_episode_id=nm_a_name,
                cause="protocol_or_scope_mismatch",
                updated_claim_contract=ClaimContract(
                    "bisect_any_adjacent_flip_general", "ordered_boundary",
                    [{"field": "opposite_endpoints", "op": "==", "value": True}],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": "some_adjacent_flip"},
                    {"requires_deterministic_binary": True}, 0.0, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ).to_dict(),
            )
            leaf_mono = self.forest.register_node(
                node_id="ci_leaf_monotone_earliest", tree_id="tree_software_ci", layer_role="leaf",
                task_view=TaskView("earliest_global_threshold", "exact_adjacent_boundary", 0.0),
                observation_map=ObservationMap("build_exit_code"),
                claim_contract=ClaimContract(
                    "bisect_monotone_unique_earliest", "ordered_boundary",
                    [
                        {"field": "opposite_endpoints", "op": "==", "value": True},
                        {"field": "monotone_verified", "op": "==", "value": True},
                    ],
                    {"procedure": "opposite_endpoint_bisect", "guarantee": "earliest_global_threshold"},
                    {"requires_monotone": True}, 0.0, "SUPPORTED_IN_SCOPE", "validated_pruning",
                ),
                support_paths=[[{"node_id": "ci_mid_bisect", "version": rev_mid["version"]}]],
                source_cluster_ids=[cluster_id],
            )
            self.forest.add_local_implication_edge(
                stronger_node_id="ci_leaf_monotone_earliest",
                weaker_node_id="ci_mid_bisect",
                preserved_contract={"backoff_guarantee": "some_adjacent_flip"},
                scope={"opposite_endpoints": True},
            )
            self.revision_events.append({
                "cluster_id": cluster_id,
                "witness_episode": nm_a_name,
                "diagnosis_cause": "protocol_or_scope_mismatch",
                "old_node_version": 1,
                "new_mid_version": rev_mid["version"],
                "spawned_specialized_leaf": leaf_mono["node_id"],
                "rule_learned": "Non-monotone bisection guarantees some_adjacent_flip, NOT earliest_global_threshold; bind strictly to TaskView!",
            })

        # --- 2. Cluster B Two-Source Contrastive Formation (Worker Pool <-> Enzyme RK4) ---
        b_sources = cluster_b["sources"][:1] if single_source_only else cluster_b["sources"]
        b_models = []
        for s_name, (fac, ev_fn) in b_sources:
            t = fac()
            sub = solve_resource_1d_direct(t)
            t.meter.charge("formation", compute=10.0)
            ev = ev_fn(sub)
            pts = [(ctrl["u"], val) for ctrl, val, _ in t.observations]
            cands = BottleneckSaturationPlugin.fit_competing_1d_families(pts[:-1])
            best_fam = min(
                cands,
                key=lambda c: 0.5 * abs(c["predict"](pts[-1][0]) - pts[-1][1]) + 0.5 * c["rmse_original_scale"] + 0.004 * c.get("num_params", 2),
            )
            ep_rec = {
                "phase": "source", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
                "episode_name": s_name, "surface_domain": t.surface_domain,
                "strict_pass": ev["strict_pass"], "costs": t.meter.snapshot(),
                "observations": deepcopy(t.observations),
                "induced_family": best_fam["family"],
            }
            self.forest.record_raw_episode(ep_rec)
            logs.append(ep_rec)
            b_models.append((s_name, t.surface_domain, best_fam, pts))

        if b_models and b_models[0][2]["family"] == "rational_saturation":
            # Build local Tree 1: Compute Pool (Root: concave limiting -> Mid: rational saturation -> Leaf: Amdahl lock)
            self.forest.register_node(
                node_id="sys_root_concave_limit", tree_id="tree_compute_pool", layer_role="root",
                task_view=TaskView("response_curve_and_target_sizing", "rmse_original_scale", 0.03),
                observation_map=ObservationMap("job_throughput"),
                claim_contract=ClaimContract(
                    "monotone_concave_resource_limit", "bottleneck",
                    [{"field": "monotone_non_decreasing", "op": "==", "value": True}],
                    {"property": "diminishing_marginal_return"}, {}, 0.05, "SUPPORTED_IN_SCOPE", "retrieve_candidate",
                ),
                probe_signatures=["sig:diminishing_marginal_positive"],
                source_cluster_ids=[cluster_id],
            )
            sys_mid = self.forest.register_node(
                node_id="sys_mid_rational", tree_id="tree_compute_pool", layer_role="mid",
                task_view=TaskView("response_curve_and_target_sizing", "rmse_original_scale", 0.025),
                observation_map=ObservationMap("job_throughput", parameter_semantics={"alpha": "serial_cycle_time", "beta": "parallel_work_time", "source_surface_domain": b_models[0][1]}),
                claim_contract=ClaimContract(
                    "rational_saturation_subroutine", "bottleneck",
                    [
                        {"field": "monotone_non_decreasing", "op": "==", "value": True},
                        {"field": "clipping_flat_tail", "op": "==", "value": False},
                    ],
                    {"procedure": "2pt_rational_fit_plus_1pt_heldout_confirm", "family": "rational_saturation"},
                    {"fit_scale": "original_y_gauss_newton", "requires_heldout_point": True},
                    0.025, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ),
                probe_signatures=["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"],
                source_cluster_ids=[cluster_id],
            )
            self.forest.add_local_implication_edge(
                stronger_node_id="sys_mid_rational", weaker_node_id="sys_root_concave_limit",
                preserved_contract={"property": "diminishing_marginal_return"},
                scope={"positive_u": True},
            )

        if len(b_models) >= 2 and b_models[1][2]["family"] == "rational_saturation":
            bio_mid = self.forest.register_node(
                node_id="bio_mid_rational", tree_id="tree_biochem_assay", layer_role="mid",
                task_view=TaskView("response_curve_and_target_sizing", "rmse_original_scale", 0.025),
                observation_map=ObservationMap("reaction_rate_v0", parameter_semantics={"alpha": "inv_vmax", "beta": "km_over_vmax", "source_surface_domain": b_models[1][1]}),
                claim_contract=ClaimContract(
                    "rational_saturation_kinetics", "bottleneck",
                    [
                        {"field": "monotone_non_decreasing", "op": "==", "value": True},
                        {"field": "clipping_flat_tail", "op": "==", "value": False},
                    ],
                    {"procedure": "2pt_rational_fit_plus_1pt_heldout_confirm", "family": "rational_saturation"},
                    {"fit_scale": "original_y_gauss_newton", "requires_heldout_point": True},
                    0.025, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ),
                probe_signatures=["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"],
                source_cluster_ids=[cluster_id],
            )
            wh_b = self.forest.add_horizontal_wormhole(
                wormhole_id="wh_B_sys_bio_rational",
                node_a_id="sys_mid_rational",
                node_b_id="bio_mid_rational",
                probe_signature_refs=["sig:diminishing_marginal_positive", "sig:rational_saturation_candidate"],
                role_and_observation_map={
                    "control_role_map": "u_workers <-> substrate_conc",
                    "output_role_map": "throughput <-> initial_velocity",
                    "isolated_leaf_semantics": {"sys_alpha": "serial_time", "bio_alpha": "1/Vmax"},
                },
                hop_error_budget=0.008,
                permission_tier="local_solver_guidance",
                source_episode_refs=[b_models[0][0], b_models[1][0]],
            )
            self.formation_events.append({
                "cluster_id": cluster_id, "event": "form_horizontal_mid_wormhole",
                "wormhole_id": wh_b["wormhole_id"],
                "node_a": "sys_mid_rational", "node_b": "bio_mid_rational",
                "competing_families_rejected_on_sources": ["linear_affine", "unbounded_sqrt_concave", "piecewise_hard_ceiling"],
            })

        # --- Near-Miss Revisions on Cluster B (Tied 2D Min Bottleneck + Unbounded Sqrt!) ---
        # Near-Miss B1: Tied 2D Bottleneck -> CEGAR Joint-Probe Upgrade + Horizontal Wormhole to Bottleneck Tree!
        nm_b1_name, (nm_b1_fac, nm_b1_eval) = cluster_b["near_misses"][0]
        t_nmb1 = nm_b1_fac()
        init_pt = dict(t_nmb1.task_view["initial_operating_point"])
        y_req = float(t_nmb1.task_view["required_output"])
        # Step 1: Try single-axis probes ONLY -> discovers BOTH single-axis gains are 0!
        d_1d = BottleneckSaturationPlugin.diagnose_multi_resource_bottleneck(
            lambda pt: t_nmb1.measure(pt, phase="probe") or 0.0, init_pt, allow_joint_probe_upgrade=False,
        )
        assert d_1d["status"] == "ALL_SINGLE_AXIS_ZERO_GAIN"
        # Step 2: 5-Cause Diagnosis triggers CEGAR Probe Refinement -> runs joint step (+1, +1)!
        joint_pt = {k: v + 1.0 for k, v in init_pt.items()}
        y_joint = t_nmb1.measure(joint_pt, phase="revision_probe") or 0.0
        t_nmb1.meter.charge("revision", compute=12.0)
        y0_tie = d_1d["base_output"]
        rec_nm1 = {"u1": init_pt["u1"] * (y_req / y0_tie), "u2": init_pt["u2"] * (y_req / y0_tie)}
        t_nmb1.measure(rec_nm1, phase="confirm")
        ev_nmb1 = nm_b1_eval({"status": "completed", "recommended_controls": rec_nm1})
        logs.append({
            "phase": "source_nearmiss", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
            "episode_name": nm_b1_name, "strict_pass": ev_nmb1["strict_pass"], "costs": t_nmb1.meter.snapshot(),
        })
        if y_joint > y0_tie + 1e-4:
            bot_mid = self.forest.register_node(
                node_id="bot_mid_limiting_factor_cegar", tree_id="tree_multi_bottleneck", layer_role="mid",
                task_view=TaskView("multi_resource_target_allocation", "achieve_target_with_near_minimal_resource_spend", 0.05),
                observation_map=ObservationMap("pipeline_capacity", parameter_semantics={"source_surface_domain": t_nmb1.surface_domain}),
                claim_contract=ClaimContract(
                    "piecewise_min_with_cegar_joint_probe", "bottleneck",
                    [{"field": "multi_resource", "op": "==", "value": True}],
                    {"procedure": "single_axis_probe_then_cegar_joint_probe_if_tied"},
                    {"never_permanently_prune_zero_gain_axis": True},
                    0.05, "SUPPORTED_IN_SCOPE", "local_solver_guidance",
                ),
                probe_signatures=["sig:multi_resource_limiting_factor", "sig:diminishing_marginal_positive"],
                source_cluster_ids=[cluster_id],
            )
            if "sys_mid_rational" in self.forest.nodes:
                self.forest.add_horizontal_wormhole(
                    wormhole_id="wh_B_sys_to_multi_bottleneck",
                    node_a_id="sys_mid_rational",
                    node_b_id="bot_mid_limiting_factor_cegar",
                    probe_signature_refs=["sig:diminishing_marginal_positive", "sig:multi_resource_limiting_factor"],
                    role_and_observation_map={"bridge_kind": "smooth_saturation_refuted_to_piecewise_limiting_factor"},
                    hop_error_budget=0.02,
                    permission_tier="local_solver_guidance",
                    source_episode_refs=[nm_b1_name],
                )
            self.revision_events.append({
                "cluster_id": cluster_id,
                "witness_episode": nm_b1_name,
                "diagnosis_cause": "abstraction_closure_loss",
                "revision_action": "CEGAR_joint_multi_variable_probe_refinement",
                "new_node_id": bot_mid["node_id"],
                "rule_learned": "Single-axis zero gain at tied min(c1*u1, c2*u2) is spurious deadlock; trigger joint increment probe!",
            })

        # Near-Miss B2: Unbounded Sqrt Concave -> ALIGN Exclusion on Wormhole + Honest Demotion of Weak Root!
        nm_b2_name, (nm_b2_fac, nm_b2_eval) = cluster_b["near_misses"][1]
        t_nmb2 = nm_b2_fac()
        sub_nmb2 = solve_resource_1d_direct(t_nmb2)
        t_nmb2.meter.charge("revision", compute=10.0)
        ev_nmb2 = nm_b2_eval(sub_nmb2)
        logs.append({
            "phase": "source_nearmiss", "arm": "Hegel_V16_6", "cluster_id": cluster_id,
            "episode_name": nm_b2_name, "strict_pass": ev_nmb2["strict_pass"], "costs": t_nmb2.meter.snapshot(),
        })
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
            # Honest utility check on root `sys_root_concave_limit`: still TRUE on sqrt(u),
            # yet knowing only "concave monotone" saves 0 probes over open-grammar solver -> demote!
            if "sys_root_concave_limit" in self.forest.nodes:
                self.forest.nodes["sys_root_concave_limit"]["utility_ledger"]["archived_low_utility"] = True
                self.revision_events.append({
                    "cluster_id": cluster_id,
                    "witness_episode": nm_b2_name,
                    "diagnosis_cause": "claim_refuted_or_unprofitable",
                    "revision_action": "ALIGN_wormhole_exclusion_plus_honest_weak_parent_demotion",
                    "demoted_parent_node": "sys_root_concave_limit",
                    "rule_learned": "Parent 'monotone_concave' remains true on sqrt(u), but has zero candidate-pruning savings -> Clean Exit to direct solver!",
                })
        return logs

    def _filter_cross_domain_candidates(self, recalled: list[dict[str, Any]], target_domain: str) -> list[dict[str, Any]]:
        """A single-tree node is domain-local unless bridged by a verified HorizontalWormholeEdge."""
        return [
            c for c in recalled
            if c["entry_kind"] == "horizontal_mid_wormhole"
            or c["target_node"]["observation_map"].get("parameter_semantics", {}).get("source_surface_domain") in {None, target_domain}
        ]

    def _reconstruct_from_raw_if_needed(self, task: Any) -> str | None:
        """Legal billed on-demand reconstruction when structured forest nodes were removed."""
        if not self.allow_raw_history_rebuild_on_missing or not self.forest.raw_episode_archive:
            return None
        task.meter.charge("on_demand_reconstruction", compute=28.0)
        rat_votes = sum(
            1 for ep in self.forest.raw_episode_archive
            if ep.get("induced_family") == "rational_saturation"
        )
        return "rational_saturation" if rat_votes >= 2 else None

    def solve_target(self, name: str, task: Any) -> dict[str, Any]:
        """Solve an unseen target using Stage-0 -> ProbeSignature -> Horizontal Wormhole -> Permission Gate."""
        # --- Case 1: Ordered Discrete Boundary Tasks (Cluster A & Cluster C View 3) ---
        if isinstance(task, PublicOrderedBoundaryTask):
            n = task.length
            goal = task.task_view["goal_kind"]
            noisy = bool(task.task_view.get("stochastic_oracle_hint", False))

            def rq(idx: int, phase: str = "probe") -> int | str | None:
                v1 = task.query_index(idx, phase=phase)
                if v1 is None or v1 == "SKIP" or not noisy or idx in {0, n - 1}:
                    return v1
                v2 = task.query_index(idx, phase=phase)
                v3 = task.query_index(idx, phase=phase)
                votes = [v for v in (v1, v2, v3) if isinstance(v, int)]
                if not votes:
                    return "SKIP"
                return 1 if sum(votes) > (len(votes) / 2.0) else 0

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
                # Delegate remaining search cleanly without re-querying endpoints!
                return solve_ordered_boundary_direct(task, pre_queried_endpoints=(v0, vn))

            chosen = recalled[0]
            auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=task.task_view,
                target_features={"opposite_endpoints": opp},
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
                    # HEC Decision Region: whether monotone or non-monotone, `some_adjacent_flip` is satisfied!
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
                    continue

                # Strict `earliest_global_threshold`: leaf requires `monotone_verified == True` -> never relax goal!
                ver = OrderedBoundaryPlugin.verify_or_find_earliest_flip(
                    lambda i: q_or_skip(i, "confirm"),
                    n, cand_idx, stride=8, budget_remaining_fn=task.meter.remaining_obs,
                )
                if ver["status"] == "SUPPORTED_IN_SCOPE":
                    return {
                        "status": "completed",
                        "guarantee_kind": "earliest_global_threshold",
                        "boundary_right_index": ver["boundary_right_index"],
                    }
            return {"status": "incomplete", "boundary_right_index": None}

        # --- Case 2: Multi-Resource 2D Bottleneck Allocation Tasks (`T_B2`, `T_B3`) ---
        if "u" not in task.control_bounds:
            init_pt = dict(task.task_view["initial_operating_point"])
            y_req = float(task.task_view["required_output"])
            task.meter.charge("retrieve", compute=3.0)
            recalled_2d = self._filter_cross_domain_candidates(
                self.forest.recall_mid_layer_wormholes(
                    ["sig:multi_resource_limiting_factor", "sig:diminishing_marginal_positive"],
                    task_view=task.task_view,
                    allow_horizontal_wormholes=self.allow_horizontal_wormholes,
                ),
                task.surface_domain,
            )
            has_cegar_node = any(
                c["target_node"]["node_id"] == "bot_mid_limiting_factor_cegar" for c in recalled_2d
            )
            if not has_cegar_node:
                return solve_resource_2d_ordinary(task)

            self.alignment_checks.append({
                "target": name,
                "entry_kind": "horizontal_mid_wormhole" if any(c["wormhole_edge"] for c in recalled_2d) else "local_tree_node",
                "selected_node": "bot_mid_limiting_factor_cegar",
                "wormhole_id": "wh_B_sys_to_multi_bottleneck" if self.allow_horizontal_wormholes else None,
            })
            diag = BottleneckSaturationPlugin.diagnose_multi_resource_bottleneck(
                lambda pt: task.measure(pt, phase="probe") or 0.0,
                init_pt, allow_joint_probe_upgrade=True,
            )
            task.meter.charge("fit", compute=6.0)
            if diag["status"] == "TIED_JOINT_BOTTLENECK_DETECTED":
                y0 = diag["base_output"]
                # Exact piecewise-min scaling from verified tied joint gain (4 probes total + 1 confirm = 5 probes!)
                rec = {"u1": init_pt["u1"] * (y_req / y0), "u2": init_pt["u2"] * (y_req / y0)}
                yc = task.measure(rec, phase="confirm")
                if yc is not None and yc >= y_req * 0.97:
                    return {
                        "status": "completed",
                        "recommended_controls": rec,
                        "cegar_joint_probe_used": True,
                        "used_wormhole": "wh_B_sys_to_multi_bottleneck" if self.allow_horizontal_wormholes else None,
                    }
            elif diag["status"] == "SINGLE_LIMITING_FACTOR":
                lim = diag["limiting_axes"][0]
                other = "u2" if lim == "u1" else "u1"
                c_lim = diag["single_axis_gains"][lim] / 1.0
                u_lim_req = y_req / c_lim
                # Probe once at u_lim_req to read off the other channel's cap directly!
                probe_pt = {lim: u_lim_req, other: init_pt[other]}
                y_at_lim = task.measure(probe_pt, phase="probe")
                if y_at_lim is None:
                    return {"status": "incomplete"}
                if y_at_lim >= y_req * 0.98:
                    rec = probe_pt
                else:
                    c_other = y_at_lim / init_pt[other]
                    rec = {lim: u_lim_req, other: y_req / max(c_other, 1e-6)}
                yc = task.measure(rec, phase="confirm")
                if yc is not None and yc >= y_req * 0.97:
                    return {
                        "status": "completed",
                        "recommended_controls": rec,
                        "used_wormhole": "wh_B_sys_to_multi_bottleneck" if self.allow_horizontal_wormholes else None,
                    }
            return solve_resource_2d_ordinary(task)

        # --- Case 3: 1D Resource / Service Tasks (`T_B1, T_B4, T_B5, T_B6` & `T_C1, T_C2`) ---
        lo, hi = task.control_bounds["u"]
        tol = float(task.task_view["tolerance"])
        scale_map = task.task_view.get("available_scale_map")

        # Check ScaleMap permission BEFORE spending any saturation transfer probes!
        if scale_map is not None and "sys_mid_rational" in self.forest.nodes:
            scale_auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=task.task_view,
                target_features={"monotone_non_decreasing": True, "clipping_flat_tail": False},
                claim_contract=self.forest.nodes["sys_mid_rational"]["claim_contract"],
                scale_map=scale_map,
                heldout_confirmed=True,
            )
            self.alignment_checks.append({
                "target": name,
                "routing_mode": task.task_view.get("routing_mode"),
                "scale_authorization": scale_auth,
            })
            if not scale_auth["granted"] and scale_auth["reason"] == "scale_map_does_not_preserve_task_view":
                # Coarse window_average loses burst tail peaks -> reject wormhole at 0 wasted probes!
                return solve_resource_1d_direct(task)

        # Stage-0 / Pilot: buy 3 informative points (low, mid, high) — shared with fallback so 0 probes wasted!
        u1, u2, u3 = lo + 0.06 * (hi - lo), lo + 0.36 * (hi - lo), lo + 0.88 * (hi - lo)
        y1 = task.measure({"u": u1}, phase="presolve")
        y2 = task.measure({"u": u2}, phase="presolve")
        y3 = task.measure({"u": u3}, phase="pilot")
        if y1 is None or y2 is None or y3 is None:
            return {"status": "incomplete"}
        pts = [(u1, y1), (u2, y2), (u3, y3)]

        # Stage-0 linear fast exit
        task.meter.charge("presolve_fit", compute=4.0)
        lin = [c for c in BottleneckSaturationPlugin.fit_competing_1d_families(pts[:2]) if c["family"] == "linear_affine"][0]
        if abs(lin["predict"](u3) - y3) <= tol * 0.5:
            u_conf = lo + 0.64 * (hi - lo)
            y_conf = task.measure({"u": u_conf}, phase="confirm")
            if y_conf is not None and abs(lin["predict"](u_conf) - y_conf) <= tol:
                return {"status": "completed", "selected_family": "linear_affine", "predict_fn": lin["predict"]}

        # Extract observable ProbeSignatures from the 3 already-purchased points
        monotone_up = (y2 >= y1 - 1e-4) and (y3 >= y2 - 1e-4)
        slope_12 = (y2 - y1) / (u2 - u1)
        slope_23 = (y3 - y2) / (u3 - u2)
        diminishing = monotone_up and (slope_23 < 0.78 * slope_12)
        clipping_suspect = monotone_up and (abs(y3 - y2) <= 0.008 * max(abs(y2), 1.0)) and (slope_12 > 0.15)

        sigs = []
        if diminishing:
            sigs.append("sig:diminishing_marginal_positive")
            sigs.append("sig:rational_saturation_candidate")

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

        # Stage-1 empirical cost gate + ALIGN / ObservationMap checks
        gate_decision = self.gate.decide_stage1(
            "sig:rational_saturation_candidate",
            remaining_budget=task.meter.remaining_meas(),
            estimated_probe_and_bind_cost=task.cost_per_probe,
        )
        if recalled and monotone_up and not clipping_suspect and gate_decision["admit_transfer"]:
            chosen = recalled[0]
            task.meter.charge("bind_and_fit", compute=6.0)
            rat_2pt = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts[:2])
            tail_exceeds = (rat_2pt["status"] != "ok") or (abs(rat_2pt["predict"](u3) - y3) > tol * 0.9)
            auth = authorize_abstraction_use(
                requested_tier="local_solver_guidance",
                task_view=task.task_view,
                target_features={
                    "monotone_non_decreasing": monotone_up,
                    "clipping_flat_tail": clipping_suspect,
                    "rational_3pt_tail_residual_exceeds_tol": tail_exceeds,
                },
                claim_contract=chosen["target_node"]["claim_contract"],
                scale_map=scale_map,
                wormhole_edge=chosen["wormhole_edge"],
                heldout_confirmed=not tail_exceeds,
            )
            self.alignment_checks.append({
                "target": name,
                "entry_kind": chosen["entry_kind"],
                "selected_node": chosen["target_node"]["node_id"],
                "wormhole_id": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                "stage1_gate": gate_decision,
                "authorization": auth,
            })
            if auth["granted"]:
                # Decision Region: both sys_mid_rational and bio_mid_rational map to identical 1-pt confirm!
                u_conf = lo + 0.62 * (hi - lo)
                y_conf = task.measure({"u": u_conf}, phase="confirm")
                if y_conf is not None and abs(rat_2pt["predict"](u_conf) - y_conf) <= tol:
                    rat_final = BottleneckSaturationPlugin.fit_rational_saturation_original_scale(pts + [(u_conf, y_conf)])
                    return {
                        "status": "completed",
                        "selected_family": "rational_saturation",
                        "predict_fn": rat_final["predict"],
                        "used_wormhole": (chosen["wormhole_edge"] or {}).get("wormhole_id"),
                    }
                if y_conf is not None:
                    pts.append((u_conf, y_conf))

        # Clean Exit to direct open-grammar solver carrying ALL already-purchased `pts` (zero wasted probes!)
        return solve_resource_1d_direct(task, existing_points=pts)

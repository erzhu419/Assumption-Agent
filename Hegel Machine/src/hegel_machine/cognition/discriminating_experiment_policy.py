"""Experience-induced discriminating experiment-selection policy M1 (v11 Deliverable C).

Separates world-knowledge learning (``K0 -> K1``) from research-method induction (``M0 -> M1``)
as specified in ``02_下一轮实施与验收.md`` §3.2–3.4:

  - **``M0`` (``developer_default_M0``)**:
    Standard space-filling / random multi-control probe policy across ``spec.feature_bounds``.
    Unhandicapped and deterministic per seed.
  - **``M1`` (``runtime_induced_from_dev_episodes``)**:
    Induced from real development episode probe rollouts (strictly disjoint from evaluation
    source chains!) by measuring how observable pre-execution probe features
    ``(pairwise_disagreement_margin, ensemble_variance, control_sparsity, displacement_scale,
    sign_contrast)`` predict empirical candidate-elimination rate per observation row
    ``eliminated_fraction / probe_rows``.
  - **Exploration archive vs default promotion (§3.4 / TRUST)**:
    Retains all evaluated candidate method rules (including ties) in ``exploration_archive``,
    promoting ``M1`` only if independent ``dev_selection`` episodes support an empirical gain,
    and emits explicit ``ConsumptionEvent`` records whenever ``M1`` alters probe selection.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from math import isfinite, sqrt
from random import Random
import time
from typing import Any

from .continuous_state import _linear_fit
from .language import Program
from .models import Observation, Probe, SearchBudget, TaskSpec
from .structure_lab import ObjectSchema
from .world_structure_commitments import _as_schema


def fit_empirical_q_actuator_maps(
    records: list[dict],
    schema: ObjectSchema | dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit public paid-data forward/inverse ``q_cmd <-> q_sensor`` maps per object (zero private access!)."""
    sc = _as_schema(schema)
    fwd_fns: dict[str, Any] = {}
    inv_fns: dict[str, Any] = {}
    if sc is None or not records:
        return fwd_fns, inv_fns

    for grp in sc.objects:
        qf = sc.fields[grp].get("q")
        qc = (sc.controls.get(grp) or {}).get("q")
        if not qf or not qc:
            continue
        distinct_pts: dict[float, float] = {}
        for r in records:
            ctls = r.get("controls") or {}
            bef = r.get("before") or {}
            if qc in ctls and qf in bef:
                distinct_pts[round(float(ctls[qc]), 6)] = float(bef[qf])
        if len(distinct_pts) >= 5 and (max(distinct_pts) - min(distinct_pts)) >= 0.35:
            c_vals = list(distinct_pts.keys())
            y_vals = list(distinct_pts.values())
            coeffs_q = tuple(
                float(x)
                for x in _linear_fit(
                    [c_vals, [c * c for c in c_vals], [c * c * c for c in c_vals]],
                    y_vals,
                )
            )

            def _fwd(val: float, _cf: tuple[float, ...] = coeffs_q) -> float:
                v = float(val)
                return float(_cf[0] + _cf[1] * v + _cf[2] * (v * v) + _cf[3] * (v * v * v))

            def _inv(
                target_q: float,
                lo_q: float = -1.0,
                hi_q: float = 1.0,
                _f: Any = _fwd,
            ) -> float:
                tq = float(target_q)
                flo, fhi = _f(lo_q), _f(hi_q)
                if flo <= fhi:
                    if tq <= flo:
                        return float(lo_q)
                    if tq >= fhi:
                        return float(hi_q)
                    a, b = float(lo_q), float(hi_q)
                    for _ in range(28):
                        m = 0.5 * (a + b)
                        if _f(m) < tq:
                            a = m
                        else:
                            b = m
                    return 0.5 * (a + b)
                return max(float(lo_q), min(float(hi_q), tq))

            fwd_fns[grp] = _fwd
            inv_fns[grp] = _inv
    return fwd_fns, inv_fns


def _simulate_proxy_trajectory_for_probe(
    probe: Probe,
    spec: TaskSpec,
    schema: ObjectSchema | dict,
    controls_by_object: dict[str, dict[str, str]] | None,
    *,
    q_forward_fns: dict[str, Any] | None = None,
) -> list[Observation]:
    """Build an observable kinematic state proxy from public control sequence ``probe.sequence``.

    Uses ONLY public ``probe.sequence``, ``spec.feature_bounds``, discovered ``schema``
    role bindings, and optional paid-data ``q_forward_fns`` (zero private lab access!) so
    candidate hypotheses ``h in H`` can be evaluated pre-execution to compute observable
    disagreement features.
    """
    sc = _as_schema(schema)
    if sc is None or not probe.sequence:
        return []
    objs = list(sc.objects)
    n = len(objs)
    hist_frames: list[dict[str, float]] = []
    out_obs: list[Observation] = []

    for st_idx, pt in enumerate(probe.sequence):
        frame: dict[str, float] = {}
        q_cur = [0.0] * n
        x_cur = [float(i) * 0.4 for i in range(n)]
        u_cur = [0.0] * n
        focus_idx = 0
        if sc.focus_control in pt:
            focus_idx = max(0, min(n - 1, round(float(pt[sc.focus_control]))))
        f_cur = [1.0 if i == focus_idx else 0.0 for i in range(n)]

        if controls_by_object:
            for idx_o, g in enumerate(objs):
                cmap = controls_by_object.get(g) or {}
                if cmap.get("q") in pt:
                    raw_qc = float(pt[cmap["q"]])
                    q_cur[idx_o] = (
                        float(q_forward_fns[g](raw_qc))
                        if (q_forward_fns and g in q_forward_fns)
                        else raw_qc
                    )
                if cmap.get("u") in pt:
                    u_cur[idx_o] = float(pt[cmap["u"]])
                if cmap.get("x") in pt:
                    x_cur[idx_o] = float(pt[cmap["x"]])
        else:
            vals = list(pt.values())
            for idx_o in range(n):
                q_cur[idx_o] = float(vals[idx_o % max(1, len(vals))])

        for idx_o, g in enumerate(objs):
            fmap = sc.fields[g]
            frame[fmap["q"]] = float(q_cur[idx_o])
            if "u" in fmap:
                frame[fmap["u"]] = float(u_cur[idx_o])
            if "x" in fmap:
                frame[fmap["x"]] = float(x_cur[idx_o])
            if "f" in fmap:
                frame[fmap["f"]] = float(f_cur[idx_o])

        obs_row = Observation(
            dict(frame),
            0.0,
            tuple(hist_frames[-4:]),
        )
        hist_frames.append(frame)
        out_obs.append(obs_row)
    return out_obs


def compute_observable_probe_features(
    probe: Probe,
    candidate_programs: list[Program],
    spec: TaskSpec,
    schema: ObjectSchema | dict,
    controls_by_object: dict[str, dict[str, str]] | None = None,
    *,
    q_forward_fns: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute strictly pre-execution observable features of ``probe`` w.r.t. ``candidate_programs``."""
    sc = _as_schema(schema)
    bounds = dict(spec.feature_bounds)
    pts = probe.sequence
    if not pts:
        return {
            "pairwise_disagreement_margin": 0.0,
            "ensemble_std": 0.0,
            "disagreement_step_fraction": 0.0,
            "control_sparsity": 0.0,
            "displacement_scale": 1.0,
            "sign_contrast": 0.0,
        }

    neutral_count = 0
    total_entries = 0
    rel_scales: list[float] = []
    pos_drives = 0
    neg_drives = 0

    for pt in pts:
        for cname, val in pt.items():
            if cname not in bounds:
                continue
            lo, hi = bounds[cname]
            mid = 0.5 * (lo + hi)
            half = max(1e-6, 0.5 * (hi - lo))
            rel = abs(float(val) - mid) / half
            total_entries += 1
            if rel <= 0.08:
                neutral_count += 1
            elif rel < 0.98:
                rel_scales.append(rel)
            if float(val) - mid > 0.1 * half:
                pos_drives += 1
            elif float(val) - mid < -0.1 * half:
                neg_drives += 1

    # Compute normalized pair separation among non-co-located active objects!
    pair_seps: list[float] = []
    co_located_bystanders = 0
    total_obj_steps = 0
    if sc is not None and controls_by_object and len(sc.objects) >= 2:
        for pt in pts:
            non_parked_x: list[float] = []
            x_half = 2.0
            for g in sc.objects:
                cmap = controls_by_object.get(g) or {}
                xc = cmap.get("x")
                if xc and xc in pt and xc in bounds:
                    lo, hi = bounds[xc]
                    x_half = max(1e-6, 0.5 * (hi - lo))
                    xv = float(pt[xc])
                    total_obj_steps += 1
                    if abs(xv - 0.5 * (lo + hi)) / x_half < 0.96:
                        non_parked_x.append(xv)
                    else:
                        co_located_bystanders += 1
            if len(non_parked_x) >= 2:
                min_dx = min(
                    abs(non_parked_x[i] - non_parked_x[j])
                    for i in range(len(non_parked_x))
                    for j in range(i + 1, len(non_parked_x))
                )
                pair_seps.append(min_dx / x_half)

    # Sparsity reflects EITHER neutral controls OR cleanly isolated co-located bystander parking!
    sparsity = max(
        neutral_count / max(1, total_entries),
        co_located_bystanders / max(1, total_obj_steps),
    )
    disp_scale = (
        sum(pair_seps) / len(pair_seps)
        if pair_seps
        else (sum(rel_scales) / len(rel_scales) if rel_scales else 0.5)
    )
    tot_signed = pos_drives + neg_drives
    sign_contrast = (
        2.0 * min(pos_drives, neg_drives) / tot_signed if tot_signed > 0 else 0.0
    )

    proxy_rows = _simulate_proxy_trajectory_for_probe(
        probe, spec, schema, controls_by_object, q_forward_fns=q_forward_fns
    )
    if len(candidate_programs) < 2 or not proxy_rows:
        return {
            "pairwise_disagreement_margin": 0.0,
            "ensemble_std": 0.0,
            "disagreement_step_fraction": 0.0,
            "control_sparsity": float(sparsity),
            "displacement_scale": float(disp_scale),
            "sign_contrast": float(sign_contrast),
        }

    preds_by_prog: list[list[float]] = []
    for prog in candidate_programs[:12]:
        row_preds = []
        for r in proxy_rows:
            v = float(prog.predict(r))
            row_preds.append(v if isfinite(v) else 0.0)
        preds_by_prog.append(row_preds)

    m = len(preds_by_prog)
    l_len = len(proxy_rows)
    max_pair_diff = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            mean_d = sum(abs(preds_by_prog[i][t] - preds_by_prog[j][t]) for t in range(l_len)) / l_len
            if mean_d > max_pair_diff:
                max_pair_diff = mean_d

    stds: list[float] = []
    disagree_steps = 0
    for t in range(l_len):
        col = [preds_by_prog[i][t] for i in range(m)]
        if (max(col) - min(col)) > 1e-6:
            disagree_steps += 1
        mu = sum(col) / m
        var = sum((x - mu) ** 2 for x in col) / m
        stds.append(sqrt(max(0.0, var)))
    ens_std = sum(stds) / len(stds) if stds else 0.0
    disagree_frac = disagree_steps / max(1, l_len)

    return {
        "pairwise_disagreement_margin": float(max_pair_diff),
        "ensemble_std": float(ens_std),
        "disagreement_step_fraction": float(disagree_frac),
        "control_sparsity": float(sparsity),
        "displacement_scale": float(disp_scale),
        "sign_contrast": float(sign_contrast),
    }


def generate_candidate_probe_menu(
    spec: TaskSpec,
    schema: ObjectSchema | dict,
    controls_by_object: dict[str, dict[str, str]] | None,
    *,
    seed: int = 11,
    probe_length: int = 6,
    audit_hint: dict[str, Any] | None = None,
    q_inverse_fns: dict[str, Any] | None = None,
) -> list[Probe]:
    """Generate a diverse menu of legal public control ``Probe``s spanning random & structured styles."""
    sc = _as_schema(schema)
    bounds = dict(spec.feature_bounds)
    ctl_names = list(bounds.keys())
    rng = Random(seed)
    menu: list[Probe] = []

    # 1. Default space-filling / full-bound random multi-control probes (what M0 uses by default!)
    for k in range(4):
        pts = []
        for _ in range(probe_length):
            pts.append({c: rng.uniform(bounds[c][0], bounds[c][1]) for c in ctl_names})
        menu.append(Probe(f"menu_random_fullbound:{k}", tuple(pts)))

    # 2. Moderate-scale random multi-control probes
    for k in range(3):
        pts = []
        for _ in range(probe_length):
            pt = {}
            for c in ctl_names:
                lo, hi = bounds[c]
                mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                pt[c] = mid + rng.uniform(-0.35, 0.35) * half
            pts.append(pt)
        menu.append(Probe(f"menu_random_subbound:{k}", tuple(pts)))

    objs = list(sc.objects) if sc is not None else []
    if controls_by_object and len(objs) >= 2:
        c0 = controls_by_object.get(objs[0], {})
        c1 = controls_by_object.get(objs[1], {})

        def _inv_q(obj_name: str, target_q: float) -> float:
            qc = (controls_by_object.get(obj_name) or {}).get("q")
            lo_q, hi_q = bounds.get(qc, (-1.0, 1.0)) if qc else (-1.0, 1.0)
            if q_inverse_fns and obj_name in q_inverse_fns:
                return float(q_inverse_fns[obj_name](target_q, lo_q, hi_q))
            return max(float(lo_q), min(float(hi_q), float(target_q)))

        bystander_q_sweep = (-0.88, 0.86, -0.46, 0.48, -0.74, 0.76)
        impulse_sweep = (-0.21, 0.19, -0.13, 0.16, -0.09, 0.22)

        def _populate_co_located_bystanders_and_impulses(
            pt: dict[str, float], st: int, sign_dir: float
        ) -> None:
            # Co-locate all bystander objects at the opposite far boundary with synchronous
            # full-range q & u sweeps: dx_byst == 0 and dq_byst == 0 -> zero cross-talk under
            # right/left/symmetric while exciting all 4N nonlinear actuator & sensor maps!
            byst_q_s = bystander_q_sweep[st % len(bystander_q_sweep)]
            for io in range(2, len(objs)):
                g_b = objs[io]
                ci = controls_by_object.get(g_b, {})
                if ci.get("x") in bounds:
                    lo_x, hi_x = bounds[ci["x"]]
                    pt[ci["x"]] = hi_x if sign_dir >= 0.0 else lo_x
                if ci.get("q") in bounds:
                    pt[ci["q"]] = _inv_q(g_b, byst_q_s)
                if ci.get("u") in bounds:
                    lo_u, hi_u = bounds[ci["u"]]
                    u_v = impulse_sweep[(st + io) % len(impulse_sweep)]
                    pt[ci["u"]] = max(lo_u, min(hi_u, u_v))
            for io_a, c_act in ((0, c0), (1, c1)):
                if c_act.get("u") in bounds:
                    lo_u, hi_u = bounds[c_act["u"]]
                    u_v = impulse_sweep[(st + io_a) % len(impulse_sweep)]
                    pt[c_act["u"]] = max(lo_u, min(hi_u, u_v))

        # 3. Broad uncalibrated pair-isolated probes across fixed scales rho in {0.14, 0.28, 0.55, 0.88}
        bracket_profiles = [
            ((0.82, 0.0), (-0.65, 0.15), (0.20, 0.05), (0.42, 0.05), (0.05, 0.78), (0.31, 0.0)),
            ((0.55, 0.50), (0.70, 0.10), (-0.45, -0.10), (0.26, 0.0), (0.0, -0.60), (0.48, 0.05)),
            ((0.85, 0.05), (0.75, 0.10), (0.90, 0.0), (0.80, 0.05), (0.70, 0.0), (0.88, 0.10)),
            ((0.50, 0.50), (-0.40, -0.40), (0.30, 0.30), (0.70, 0.70), (-0.20, -0.20), (0.60, 0.60)),
        ]
        for rho_idx, rho in enumerate((0.14, 0.28, 0.55, 0.88)):
            for prof_idx, prof in enumerate(bracket_profiles):
                pts = []
                for st in range(probe_length):
                    pt = {c: 0.5 * (bounds[c][0] + bounds[c][1]) for c in ctl_names}
                    if sc is not None and sc.focus_control in bounds:
                        flo, fhi = bounds[sc.focus_control]
                        pt[sc.focus_control] = max(flo, min(fhi, float(st % 2)))
                    sign_x = 1.0 if (st < max(1, (2 * probe_length) // 3)) else -1.0
                    _populate_co_located_bystanders_and_impulses(pt, st, sign_x)
                    if c0.get("x") in bounds and c1.get("x") in bounds:
                        lo_x0, hi_x0 = bounds[c0["x"]]
                        lo_x1, hi_x1 = bounds[c1["x"]]
                        span_x = hi_x0 - lo_x0
                        x0_anchor = (
                            (lo_x0 + 0.12 * span_x)
                            if sign_x >= 0.0
                            else (hi_x0 - 0.12 * span_x)
                        )
                        pt[c0["x"]] = max(lo_x0, min(hi_x0, x0_anchor))
                        pt[c1["x"]] = max(
                            lo_x1,
                            min(hi_x1, x0_anchor + sign_x * rho * (0.5 * span_x)),
                        )
                    q0_s, q1_s = prof[st % len(prof)]
                    if c0.get("q") in bounds:
                        pt[c0["q"]] = _inv_q(objs[0], q0_s)
                    if c1.get("q") in bounds:
                        pt[c1["q"]] = _inv_q(objs[1], q1_s)
                    pts.append(pt)
                menu.append(Probe(f"menu_pair_scale_{rho_idx}_prof_{prof_idx}", tuple(pts)))

        # 4. Version-space interval-bisecting probes built strictly from public `audit_hint` (if available)
        if isinstance(audit_hint, dict):
            v_dir = audit_hint.get("verified_direction")
            sign_dir = -1.0 if v_dir == "left" else 1.0
            r_iv = audit_hint.get("radius_interval")
            r_lo, r_hi = (
                (max(0.05, float(r_iv[0])), min(1.75, float(r_iv[1])))
                if (isinstance(r_iv, (list, tuple)) and len(r_iv) == 2)
                else (0.35, 1.35)
            )
            if r_hi <= r_lo + 1e-6:
                r_hi = r_lo + 0.02
            t_iv = audit_hint.get("verified_threshold_interval")
            occam_g = audit_hint.get("occam_minimality_gate") or {}
            has_t_iv = (
                isinstance(audit_hint.get("verified_edge_condition"), dict)
                and isinstance(t_iv, (list, tuple))
                and len(t_iv) == 2
                and not bool(occam_g.get("boundary_test_needed"))
            )

            quantile_sets = (
                ((0.25, 0.50, 0.75), 0.84),
                ((0.16, 0.48, 0.82), 0.90),
                ((0.32, 0.60, 0.88), 0.78),
            )
            for b_idx, (q_fracs, q_hi_base) in enumerate(quantile_sets):
                pts_b = []
                for st in range(probe_length):
                    pt = {c: 0.5 * (bounds[c][0] + bounds[c][1]) for c in ctl_names}
                    if sc is not None and sc.focus_control in bounds:
                        flo, fhi = bounds[sc.focus_control]
                        pt[sc.focus_control] = max(flo, min(fhi, float(st % 2)))
                    _populate_co_located_bystanders_and_impulses(pt, st, sign_dir)

                    lo_x0, hi_x0 = bounds.get(c0.get("x", ""), (-2.0, 2.0))
                    lo_x1, hi_x1 = bounds.get(c1.get("x", ""), (-2.0, 2.0))
                    span_x = hi_x0 - lo_x0
                    x0_anchor = (
                        (lo_x0 + 0.12 * span_x)
                        if sign_dir >= 0.0
                        else (hi_x0 - 0.12 * span_x)
                    )

                    half_len = max(1, probe_length // 2)
                    if st < half_len:
                        # Pure radius-interval bisection: hold gap safely above threshold version space!
                        frac_r = q_fracs[st % len(q_fracs)]
                        dx_step = r_lo + frac_r * (r_hi - r_lo)
                        q0_s = q_hi_base
                        safe_gap = (
                            min(1.65, max(0.65, float(t_iv[1]) + 0.22))
                            if has_t_iv
                            else 0.85
                        )
                        q1_s = max(-0.92, min(0.20, q0_s - safe_gap))
                    else:
                        # Pure threshold / Occam-boundary bisection: hold dx safely inside r_lo!
                        dx_step = max(0.08, r_lo - 0.04)
                        sub_i = st - half_len
                        if has_t_iv:
                            t_lo_f, t_hi_f = float(t_iv[0]), float(t_iv[1])
                            frac_t = q_fracs[sub_i % len(q_fracs)]
                            g_target = t_lo_f + frac_t * (t_hi_f - t_lo_f)
                            q0_s = min(0.94, max(0.35, 0.5 * g_target + 0.32))
                            q1_s = max(-0.94, min(0.90, q0_s - g_target))
                        else:
                            # Occam boundary probe: negative donor + positive-gap sweep
                            boundary_pairs = (
                                (-0.22, -0.90),
                                (0.62, 0.46),
                                (0.76, 0.40),
                            )
                            q0_s, q1_s = boundary_pairs[sub_i % len(boundary_pairs)]

                    if c0.get("x") in bounds:
                        pt[c0["x"]] = max(lo_x0, min(hi_x0, x0_anchor))
                    if c1.get("x") in bounds:
                        pt[c1["x"]] = max(lo_x1, min(hi_x1, x0_anchor + sign_dir * dx_step))
                    if c0.get("q") in bounds:
                        pt[c0["q"]] = _inv_q(objs[0], q0_s)
                    if c1.get("q") in bounds:
                        pt[c1["q"]] = _inv_q(objs[1], q1_s)
                    pts_b.append(pt)
                menu.append(Probe(f"menu_interval_bisect:{b_idx}", tuple(pts_b)))
    return menu


def build_generic_schema_candidate_hypotheses(
    schema: ObjectSchema | dict,
    *,
    audit_hint: dict[str, Any] | None = None,
) -> list[Program]:
    """Build relational response templates spanning the surviving version space (shared by K0 and K1!)."""
    from .premise_dependency_audit import (
        _replace_const_at_path,
        describe_comparator_threshold_parameter,
        edge_condition_predicate,
    )
    from .world_structure_commitments import instantiate_commitment_terms

    sc = _as_schema(schema)
    if sc is None:
        return []

    q_src = Program("arg", name="edge_source_quantity")
    q_dst = Program("arg", name="edge_target_quantity")
    gap = Program("sub", children=(q_src, q_dst))

    specs: list[tuple[str, str | dict, float]] = []
    if isinstance(audit_hint, dict) and (
        audit_hint.get("verified_direction") or audit_hint.get("radius_interval")
    ):
        v_dir = audit_hint.get("verified_direction")
        dirs = [v_dir] if v_dir in {"right", "left"} else ["right", "left"]
        r_iv = audit_hint.get("radius_interval")
        r_lo, r_hi = (
            (max(0.05, float(r_iv[0])), min(1.75, float(r_iv[1])))
            if (isinstance(r_iv, (list, tuple)) and len(r_iv) == 2)
            else (0.45, 1.25)
        )
        r_mid = 0.5 * (r_lo + r_hi)
        r_safe = max(0.08, r_lo - 0.04)
        v_cond = audit_hint.get("verified_edge_condition")
        t_iv = audit_hint.get("verified_threshold_interval")
        occam_g = audit_hint.get("occam_minimality_gate") or {}

        if (
            isinstance(v_cond, dict)
            and isinstance(t_iv, (list, tuple))
            and len(t_iv) == 2
            and not bool(occam_g.get("boundary_test_needed"))
        ):
            base_pred = Program.from_dict(v_cond)
            desc = describe_comparator_threshold_parameter(base_pred)
            t_lo, t_hi = float(t_iv[0]), float(t_iv[1])
            d0 = dirs[0]
            if desc is not None:
                c_path = tuple(desc["path"])
                t_mid_pred = _replace_const_at_path(
                    base_pred, c_path, 0.5 * (t_lo + t_hi)
                ).to_dict()
                for rf in (0.15, 0.38, 0.62, 0.85):
                    specs.append((d0, t_mid_pred, r_lo + rf * (r_hi - r_lo)))
                for tf in (0.15, 0.38, 0.62, 0.85):
                    tp_d = _replace_const_at_path(
                        base_pred, c_path, t_lo + tf * (t_hi - t_lo)
                    ).to_dict()
                    specs.append((d0, tp_d, r_safe))
            else:
                for rf in (0.20, 0.50, 0.80):
                    specs.append((d0, v_cond, r_lo + rf * (r_hi - r_lo)))
        else:
            for d0 in dirs:
                specs.append((d0, "unconditional", r_lo + 0.20 * (r_hi - r_lo)))
                specs.append((d0, "unconditional", r_lo + 0.80 * (r_hi - r_lo)))
                specs.append((d0, "source_greater", r_mid))
                for th_c in (0.18, 0.34, 0.50):
                    specs.append(
                        (
                            d0,
                            Program(
                                "and",
                                children=(
                                    Program("lt", children=(Program("const", value=0.0), q_src)),
                                    Program("lt", children=(Program("const", value=th_c), gap)),
                                ),
                            ).to_dict(),
                            r_safe,
                        )
                    )

    if not specs:
        specs = [
            ("right", "unconditional", 0.85),
            ("left", "unconditional", 0.85),
            ("right", "source_greater", 0.85),
            ("left", "source_greater", 0.85),
            (
                "right",
                Program(
                    "and",
                    children=(
                        Program("lt", children=(Program("const", value=0.0), q_src)),
                        Program("lt", children=(Program("const", value=0.22), gap)),
                    ),
                ).to_dict(),
                0.85,
            ),
            (
                "right",
                Program(
                    "and",
                    children=(
                        Program("lt", children=(Program("const", value=0.0), q_src)),
                        Program("lt", children=(Program("const", value=0.45), gap)),
                    ),
                ).to_dict(),
                0.85,
            ),
        ]

    fq_sum = {
        "kind": "all_object_sum",
        "roles": ["f", "q"],
        "summand": Program(
            "mul",
            children=(
                Program("feature", name="object_role:f"),
                Program("feature", name="object_role:q"),
            ),
        ).to_dict(),
    }
    fu_sum = {
        "kind": "all_object_sum",
        "roles": ["f", "u"],
        "summand": Program(
            "mul",
            children=(
                Program("feature", name="object_role:f"),
                Program("feature", name="object_role:u"),
            ),
        ).to_dict(),
    }

    progs: list[Program] = []
    for d_mode, ec_spec, r_val in specs:
        pred_d = (
            ec_spec
            if isinstance(ec_spec, dict)
            else edge_condition_predicate(ec_spec).to_dict()
        )
        comm = {
            "commitment_id": f"generic:{d_mode}",
            "radius_parameters": [
                {"symbol": "r_0", "direction": d_mode, "edge_condition": ec_spec, "source_values": [r_val]}
            ],
            "role_terms": [
                fq_sum,
                fu_sum,
                {
                    "kind": "all_relation_sum",
                    "direction": d_mode,
                    "symbol": "r_0",
                    "predicate": pred_d,
                },
            ],
        }
        terms = instantiate_commitment_terms(
            comm,
            sc,
            radius_bindings={"r_0": r_val},
            predicate_overrides={"r_0": ec_spec},
        )
        if terms and len(terms) >= 3:
            progs.append(
                Program(
                    "add",
                    children=(
                        terms[0],
                        Program("mul", children=(Program("const", value=0.22), terms[2])),
                    ),
                )
            )
    return progs


def frozen_default_method_m0() -> dict[str, Any]:
    """Return frozen baseline experiment policy ``M0`` (``developer_default_M0``)."""
    return {
        "method_id": "experiment_policy:M0_default_space_filling",
        "author_origin": "developer_default_M0",
        "mode": "default_space_filling_first",
        "feature_weights": {
            "pairwise_disagreement_margin": 0.0,
            "ensemble_std": 0.0,
            "disagreement_step_fraction": 0.0,
            "control_sparsity": 0.0,
            "displacement_scale": 0.0,
            "sign_contrast": 0.0,
        },
        "preferred_displacement_scale": 1.0,
        "semantic_version": "M0_v11_frozen",
    }


def select_probe_under_method(
    method_spec: dict[str, Any],
    probe_menu: list[Probe],
    candidate_programs: list[Program],
    spec: TaskSpec,
    schema: ObjectSchema | dict,
    controls_by_object: dict[str, dict[str, str]] | None = None,
    *,
    round_index: int = 0,
    exclude_probe_ids: set[str] | None = None,
    q_forward_fns: dict[str, Any] | None = None,
    budget: SearchBudget | None = None,
) -> tuple[Probe, dict[str, Any]]:
    """Select next ``Probe`` under ``method_spec`` (``M0`` or learned ``M1``) and emit ``ConsumptionEvent``."""
    assert probe_menu, "probe_menu must be non-empty"
    default_probe = probe_menu[round_index % min(4, len(probe_menu))]

    if method_spec.get("mode") == "default_space_filling_first":
        if budget is not None:
            budget.spend(1)
        return default_probe, {
            "method_id": method_spec["method_id"],
            "author_origin": method_spec["author_origin"],
            "selected_probe_id": default_probe.probe_id,
            "default_m0_probe_id": default_probe.probe_id,
            "changed_probe_vs_m0": False,
            "consumption_event": None,
        }

    weights = method_spec.get("feature_weights") or {}
    pref_scale = float(method_spec.get("preferred_displacement_scale", 0.35))
    best_score = -1e18
    best_probe = default_probe
    best_feats: dict[str, float] = {}
    excluded = exclude_probe_ids or set()
    eligible_menu = [p for p in probe_menu if p.probe_id not in excluded] or probe_menu

    for p in eligible_menu:
        if budget is not None and not budget.spend(2):
            break
        feats = compute_observable_probe_features(
            p,
            candidate_programs,
            spec,
            schema,
            controls_by_object,
            q_forward_fns=q_forward_fns,
        )
        scale_proximity = -abs(feats["displacement_scale"] - pref_scale)
        score = (
            float(weights.get("pairwise_disagreement_margin", 0.0))
            * feats["pairwise_disagreement_margin"]
            + float(weights.get("ensemble_std", 0.0)) * feats["ensemble_std"]
            + float(weights.get("disagreement_step_fraction", 0.0))
            * feats.get("disagreement_step_fraction", 0.0)
            + float(weights.get("control_sparsity", 0.0)) * feats["control_sparsity"]
            + float(weights.get("sign_contrast", 0.0)) * feats["sign_contrast"]
            + float(weights.get("scale_proximity", 0.0)) * scale_proximity
        )
        if score > best_score:
            best_score = score
            best_probe = p
            best_feats = feats

    changed = best_probe.probe_id != default_probe.probe_id
    cons_ev = {
        "object_id": method_spec["method_id"],
        "semantic_version": method_spec.get("semantic_version"),
        "author_origin": method_spec.get("author_origin"),
        "consumption_site": "select_discriminating_experiment_probe",
        "generated_candidate_ids": [],
        "probe_before": default_probe.probe_id,
        "probe_after": best_probe.probe_id,
        "selected_probe_features": best_feats,
        "score": float(best_score),
    }
    return best_probe, {
        "method_id": method_spec["method_id"],
        "author_origin": method_spec["author_origin"],
        "selected_probe_id": best_probe.probe_id,
        "default_m0_probe_id": default_probe.probe_id,
        "changed_probe_vs_m0": changed,
        "consumption_event": cons_ev,
    }


def induce_discriminating_experiment_method(
    dev_rollout_records: list[dict[str, Any]],
    dev_selection_evaluator: Any,
    *,
    generation_tag: str = "M1",
    parent_method_id: str = "experiment_policy:M0_default_space_filling",
    budget: SearchBudget | None = None,
) -> dict[str, Any]:
    """Induce an executable probe-selection policy ``M1`` from real dev rollout outcomes (§3.3–3.4)."""
    t0 = time.perf_counter()
    start = budget.spent if budget is not None else 0
    if not dev_rollout_records:
        m0 = frozen_default_method_m0()
        return {
            "promoted_method": m0,
            "promoted": False,
            "baseline_m0_dev_selection": None,
            "promoted_dev_selection": None,
            "learned_regression_weights": {},
            "learned_preferred_displacement_scale": 1.0,
            "exploration_archive": [{"method": m0, "dev_selection_metrics": None}],
            "formation_compute_cost": 0,
            "wall_seconds": 0.0,
        }

    feat_names = [
        "pairwise_disagreement_margin",
        "ensemble_std",
        "disagreement_step_fraction",
        "control_sparsity",
        "sign_contrast",
    ]
    cols = [
        [float(rec["probe_features"].get(fn, 0.0)) for rec in dev_rollout_records]
        for fn in feat_names
    ]
    targets = [float(rec["empirical_elimination_per_row"]) for rec in dev_rollout_records]
    if budget is not None:
        budget.spend(len(dev_rollout_records) * (len(feat_names) + 2))

    # 1. Fit linear regression from scale-free observable probe features -> empirical elimination rate!
    fitted_coeffs = _linear_fit(cols, targets)
    reg_weights = {fn: round(float(w), 6) for fn, w in zip(feat_names, fitted_coeffs[1:])}

    sorted_by_gain = sorted(
        dev_rollout_records,
        key=lambda r: float(r["empirical_elimination_per_row"]),
        reverse=True,
    )
    top_k = sorted_by_gain[: max(1, len(sorted_by_gain) // 4)]
    learned_scale = round(
        sum(float(r["probe_features"].get("displacement_scale", 0.35)) for r in top_k)
        / len(top_k),
        4,
    )

    source_ep_ids = sorted({str(r["episode_id"]) for r in dev_rollout_records})

    # 2. Construct candidate method family (regression-induced rule + scale-free coverage rule + ablation + M0)
    m0_base = frozen_default_method_m0()
    candidates_to_screen: list[dict[str, Any]] = [
        m0_base,
        {
            "method_id": f"experiment_policy:{generation_tag}_regression_induced",
            "author_origin": "runtime_induced_from_dev_episodes",
            "parent_method_id": parent_method_id,
            "source_dev_episode_ids": source_ep_ids,
            "mode": "learned_linear_probe_scorer",
            "feature_weights": {**reg_weights, "scale_proximity": 0.0},
            "preferred_displacement_scale": learned_scale,
        },
        {
            "method_id": f"experiment_policy:{generation_tag}_version_space_coverage_induced",
            "author_origin": "runtime_induced_from_dev_episodes",
            "parent_method_id": parent_method_id,
            "source_dev_episode_ids": source_ep_ids,
            "mode": "learned_linear_probe_scorer",
            "feature_weights": {
                "pairwise_disagreement_margin": max(0.05, reg_weights["pairwise_disagreement_margin"]),
                "ensemble_std": max(0.05, reg_weights["ensemble_std"]),
                "disagreement_step_fraction": max(0.20, reg_weights["disagreement_step_fraction"]),
                "control_sparsity": max(0.10, reg_weights["control_sparsity"]),
                "sign_contrast": max(0.0, reg_weights["sign_contrast"]),
                "scale_proximity": 0.0,
            },
            "preferred_displacement_scale": learned_scale,
        },
        {
            "method_id": f"experiment_policy:{generation_tag}_unfocused_ensemble_std_only",
            "author_origin": "runtime_induced_from_dev_episodes",
            "parent_method_id": parent_method_id,
            "source_dev_episode_ids": source_ep_ids,
            "mode": "learned_linear_probe_scorer",
            "feature_weights": {
                "pairwise_disagreement_margin": 0.0,
                "ensemble_std": 1.0,
                "disagreement_step_fraction": 0.0,
                "control_sparsity": 0.0,
                "sign_contrast": 0.0,
                "scale_proximity": 0.0,
            },
            "preferred_displacement_scale": 0.85,
        },
    ]

    for c_m in candidates_to_screen:
        if "semantic_version" not in c_m:
            c_m["semantic_version"] = hashlib.sha256(
                json.dumps(c_m, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]

    # 3. Evaluate on independent `dev_selection` tasks; keep ALL in `exploration_archive` (TRUST / §3.4)!
    exploration_archive: list[dict[str, Any]] = []
    m0_eval = None
    best_cand = m0_base
    best_eval = None

    for cand in candidates_to_screen:
        ev = dev_selection_evaluator(cand)
        rec_entry = {
            "method": deepcopy(cand),
            "dev_selection_metrics": ev,
        }
        exploration_archive.append(rec_entry)
        if cand["method_id"] == m0_base["method_id"]:
            m0_eval = ev
            if best_eval is None:
                best_eval = ev
        else:
            if best_eval is None or (
                ev["passes"],
                -ev["mean_observations"],
                -ev["mean_compute"],
            ) > (
                best_eval["passes"],
                -best_eval["mean_observations"],
                -best_eval["mean_compute"],
            ):
                best_cand = cand
                best_eval = ev

    promoted = bool(
        best_cand["method_id"] != m0_base["method_id"]
        and m0_eval is not None
        and best_eval is not None
        and (
            best_eval["passes"] > m0_eval["passes"]
            or (
                best_eval["passes"] == m0_eval["passes"]
                and best_eval["mean_observations"] < m0_eval["mean_observations"] - 1.0
            )
        )
    )
    chosen = deepcopy(best_cand) if promoted else deepcopy(m0_base)
    return {
        "promoted": promoted,
        "promoted_method": chosen,
        "baseline_m0_dev_selection": m0_eval,
        "promoted_dev_selection": best_eval if promoted else m0_eval,
        "learned_regression_weights": reg_weights,
        "learned_preferred_displacement_scale": learned_scale,
        "exploration_archive": exploration_archive,
        "formation_compute_cost": (budget.spent - start) if budget is not None else 0,
        "wall_seconds": round(time.perf_counter() - t0, 4),
    }

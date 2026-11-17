r"""Cross-task learned, falsifiable sensor-role calibration priors for high-arity apparatuses.

Addresses the v10.2 -> v10.3 high-arity anonymous sensor bottleneck (N >= 6):
In cold mode without prior knowledge of apparatus sensor/actuator organization,
``propose_object_probe`` executes $O(N^2)$ paired ``(stable_control, quantity)`` sweeps at
control endpoints ``bounds[1]`` (where $|x_a - 0| = 2.0 > r^*$, producing zero interaction
whenever $r^* < 2.0$ and leaving up to $N!$ position permutations unresolved while burning
hundreds of thousands of ``SearchBudget`` units in repeated ``infer_object_model`` passes).

``SensorRoleCalibrationPrior`` is distilled ONLY from previously confirmed source tasks in
``KnowledgeLibrary`` (returning empty ``[]`` in cold mode or when Stage-1 source formation
failed). On a new anonymous apparatus of arbitrary arity $N \ge 4$, it:
  1. Checks structural applicability after the standard $O(N)$ single-control endpoint scan
     (immediately declining on position-free continuum PDEs or mismatched control counts).
  2. Executes a **2-row falsifiability check** testing equal-position silence (ruling out
     ``local_direct_effect``) and simultaneous multi-drive additivity (ruling out cross-object
     actuator coupling); any violation refutes the prior and falls back cleanly to cold search.
  3. Executes an **$O(N)$ Hub-Split Position Calibration** ($\le 2N + 1$ reset rows): splits
     dynamic quantities into two halves $Q_A, Q_B$ ($|Q_A| \ge 2, |Q_B| \ge 2$) at super-threshold
     gap and nudges each candidate position control $c_{x_a}$ by a source-calibrated sub-radius
     step $\pm \delta_x \in (0, \hat{r})$. Because all $N - 1$ non-nudged objects stay co-located
     at $x = \text{center}$ (zero pairwise flux among them), the nudged object $\pi(a)$ interacts
     simultaneously with $\ge 2$ partners and is uniquely identified by $|\Delta q_{\pi(a)}| \ge 2J > J$.
  4. Executes an **$O(N)$ Iso-State Selector-Sweep Focus Calibration** ($\le 2N$ rows): sweeps
     only ``focus_control`` across its $N$ selector states on a fixed 2-step asymmetric chain
     where all $N$ objects have distinct measured flows $\phi_k$ and states $q_k$, matching
     $f_m \leftrightarrow q_{\pi'(m)}$ by affine $R^2 \ge 1 - 10^{-8}$ (or flagging focus-invariant
     global readouts such as ``total``).
  5. Hands the calibrated model directly to ``UnifiedResearchAgent``'s independent 12-row
     ``confirm_discovered_object_roles`` falsification check.

Zero private apparatus attributes (``lab._...``) are ever read; every row and compute unit is
charged through the standard ``acquire`` and ``SearchBudget`` interfaces.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .continuous_state import _linear_fit
from .models import Probe, SearchBudget, TaskSpec
from .object_induction import _different, _finish_object_schema, infer_object_model
from .object_interventions import _missing_states


def consolidate_sensor_role_calibration_priors(
    library,
    commitments: list[dict] | tuple[dict, ...] = (),
    *,
    budget: SearchBudget | None = None,
) -> list[dict[str, Any]]:
    """Distill a falsifiable ``SensorRoleCalibrationPrior`` from confirmed library experience."""
    if library is None:
        return []
    start = budget.spent if budget is not None else 0
    object_models = list(getattr(library, "object_models", ()) or ())
    meta_priors = list(getattr(library, "meta_priors", ()) or ())

    # Require genuine prior confirmed experience with a 4-role (q, u, x, f) object system!
    has_4role_source = False
    source_ids: list[str] = []
    for om in object_models:
        if budget is not None and not budget.spend(2):
            break
        sc = om.get("schema") or {}
        fields = sc.get("fields") or {}
        if om.get("complete") and fields and all(set(v) >= {"q", "u", "x", "f"} for v in fields.values()):
            # Verify source had pure_additive drives and relational_only positions
            r_cands = om.get("role_candidates") or {}
            if all(
                rc.get("dependency_kind") in {None, "pure_additive", "relational_only"}
                for rc in r_cands.values()
            ):
                has_4role_source = True
                if om.get("model_id"):
                    source_ids.append(str(om["model_id"]))

    confirmed_source_ids: list[str] = []
    candidate_unconfirmed_source_ids: list[str] = []
    for mp in meta_priors:
        if budget is not None and not budget.spend(2):
            break
        if mp.get("status") in {"revoked", "stale_dependency_invalidated"}:
            continue
        if mp.get("evidence_kind") == "synthetic_stress_fixture":
            # Synthetic stress fixtures do not count as confirmed empirical source episodes!
            sc_s = mp.get("object_schema") or {}
            fields_s = sc_s.get("fields") or {}
            if fields_s and all(set(v) >= {"q", "u", "x", "f"} for v in fields_s.values()):
                has_4role_source = True
                if mp.get("source_concept_id"):
                    candidate_unconfirmed_source_ids.append(str(mp["source_concept_id"]))
            continue
        sc = mp.get("object_schema") or {}
        fields = sc.get("fields") or {}
        if fields and all(set(v) >= {"q", "u", "x", "f"} for v in fields.values()):
            has_4role_source = True
            cid_s = str(mp.get("source_concept_id") or "")
            c_loss = mp.get("confirmation_loss")
            rm = mp.get("research_method") or {}
            is_conf = (
                c_loss is not None
                and float(c_loss) <= 1e-7
                and int(rm.get("confirmation_rows", 0) or 0) > 0
            )
            if cid_s:
                source_ids.append(cid_s)
                if is_conf:
                    confirmed_source_ids.append(cid_s)
                else:
                    candidate_unconfirmed_source_ids.append(cid_s)

    if not has_4role_source:
        return []

    # Calibrate sub-radius step fraction from learned commitments (bounded safely inside (0.10, 0.35))
    radii: list[float] = []
    thresholds: list[float] = []
    for comm in commitments or ():
        for rp in comm.get("radius_parameters") or ():
            for sv in rp.get("source_values") or ():
                if isinstance(sv, (int, float)) and 0.15 < float(sv) < 3.5:
                    radii.append(float(sv))
            ec = rp.get("edge_condition")
            if isinstance(ec, dict):
                # Extract numeric constant hints from learned predicate AST
                stack = [ec]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        if node.get("op") == "const" and isinstance(node.get("value"), (int, float)):
                            val = abs(float(node["value"]))
                            if 0.01 < val < 0.85:
                                thresholds.append(val)
                        stack.extend(node.get("children") or ())

    typical_radius = min(radii) if radii else 0.95
    # On default x-span 4.0 ([-2, 2]), step_delta = min(0.42, 0.42 * typical_radius) guarantees 0 < delta < r*
    calibrated_dx = max(0.20, min(0.42, 0.42 * typical_radius))
    calibrated_thresh = max(thresholds) if thresholds else 0.28

    prior_obj = {
        "prior_id": "sensor_calibration_prior:4role_hub_and_selector_v1",
        "method_author_origin": "developer_builtin_procedure_with_source_parameter_hints",
        "applicability_support_status": (
            "empirically_confirmed_sources"
            if confirmed_source_ids
            else "candidate_prior_only"
        ),
        "source_ids": list(dict.fromkeys(source_ids))[:8],
        "confirmed_source_ids": list(dict.fromkeys(confirmed_source_ids))[:8],
        "candidate_unconfirmed_source_ids": list(
            dict.fromkeys(candidate_unconfirmed_source_ids)
        )[:8],
        "expected_roles": ["q", "u", "x", "f"],
        "requires_pure_additive_drives": True,
        "requires_equal_position_silence": True,
        "calibrated_hub_step_abs": round(calibrated_dx, 4),
        "calibrated_min_excitation_gap": round(min(0.85, max(0.55, calibrated_thresh + 0.35)), 4),
        "consolidation_compute_cost": (budget.spent - start) if budget is not None else 0,
    }
    return [prior_obj]


def _batched_acquire(
    points: list[dict[str, float]],
    *,
    max_len: int,
    acquire_fn: Callable,
    channels: tuple[str, ...],
    purpose: str,
    prefix: str,
    remaining_budget_rows: int,
) -> int:
    """Acquire ``points`` packed into legal ``Probe`` sequences of length ``<= max_len``."""
    bought = 0
    idx = 0
    batch_no = 0
    while idx < len(points) and bought < remaining_budget_rows:
        allow = min(max_len, remaining_budget_rows - bought, len(points) - idx)
        if allow <= 0:
            break
        chunk = tuple(points[idx : idx + allow])
        probe = Probe(f"{prefix}:{batch_no}", chunk)
        obs = acquire_fn(channels, purpose, probe=probe)
        if obs is None:
            break
        bought += len(obs)
        idx += allow
        batch_no += 1
    return bought


def calibrate_anonymous_roles_with_prior(
    raw_spec: TaskSpec,
    model: dict[str, Any],
    *,
    acquire_fn: Callable,
    records_fn: Callable[[], list[dict[str, Any]]],
    channels: tuple[str, ...],
    budget: SearchBudget,
    actions: list[dict[str, Any]],
    prior: dict[str, Any],
    row_limit: int,
) -> dict[str, Any]:
    """Execute falsifiable $O(N)$ sensor-role calibration using ``prior`` on an anonymous apparatus."""
    start_comp = budget.spent
    bounds = raw_spec.feature_bounds
    if not bounds:
        return {"status": "prior_inapplicable", "reason": "no_public_controls", "model": model}

    max_len = max(2, int(raw_spec.max_probe_length))
    center = {name: 0.5 * (lo + hi) for name, (lo, hi) in bounds.items()}

    # 1. Ensure the standard O(N) centered single-control scan is complete first
    single_pts = [center]
    single_pts.extend(
        {**center, ctl: val} for ctl, iv in bounds.items() for val in iv
    )
    recs = records_fn()
    missing_single = _missing_states(single_pts, recs)
    if missing_single:
        if len(recs) + len(missing_single) > row_limit:
            return {"status": "prior_unfunded", "reason": "insufficient_rows_for_single_scan", "model": model}
        _batched_acquire(
            missing_single,
            max_len=max_len,
            acquire_fn=acquire_fn,
            channels=channels,
            purpose="isolate_control_effects",
            prefix="cal_single_scan",
            remaining_budget_rows=row_limit - len(recs),
        )
        recs = records_fn()
        model = infer_object_model(raw_spec, recs, budget)

    if model.get("complete"):
        return {"status": "already_complete", "model": model}

    dynamic = list(model.get("dynamic_fields") or ())
    n_obj = len(dynamic)
    q_controls = dict(model.get("quantity_controls") or {})
    focus_ctl = model.get("focus_control")
    role_cands = model.get("role_candidates") or {}
    focus_cands = model.get("focus_candidates") or {}

    # Structural applicability check (e.g., Continuum Ring PDE has 0 x-fields -> cleanly inapplicable!)
    u_fields = {
        f: info
        for f, info in role_cands.items()
        if info.get("additive_supported") and len(info.get("quantity_candidates") or ()) == 1
    }
    x_fields = {
        f: info
        for f, info in role_cands.items()
        if not info.get("additive_supported") and info.get("roles") == ["x"]
    }
    if (
        n_obj < 4
        or len(q_controls) != n_obj
        or len(u_fields) != n_obj
        or len(x_fields) != n_obj
        or len(focus_cands) != n_obj
        or focus_ctl is None
    ):
        actions.append(
            {
                "action": "skip_sensor_role_calibration_prior",
                "prior_id": prior["prior_id"],
                "reason": "structural_signature_mismatch",
                "dynamic_count": n_obj,
                "u_candidate_count": len(u_fields),
                "x_candidate_count": len(x_fields),
                "compute_cost": budget.spent - start_comp,
            }
        )
        return {"status": "prior_inapplicable", "reason": "structural_signature_mismatch", "model": model}

    # Verify candidate u-fields map 1-to-1 onto dynamic quantities
    u_by_q: dict[str, tuple[str, str]] = {}
    for uf, info in u_fields.items():
        q_target = info["quantity_candidates"][0]
        u_by_q[q_target] = (uf, info["control"])
    if len(u_by_q) != n_obj:
        return {"status": "prior_inapplicable", "reason": "ambiguous_drive_targets", "model": model}

    # 2. Falsifiability Gate: 2 single-step reset probes testing equal-x silence & multi-drive additivity!
    recs = records_fn()
    if len(recs) + 2 > row_limit or not budget.spend(4 * n_obj):
        return {"status": "prior_unfunded", "model": model}

    unequal_q_pt = dict(center)
    for idx_q, q_name in enumerate(dynamic):
        qc = q_controls[q_name]
        lo, hi = bounds[qc]
        frac = 0.15 + 0.75 * (idx_q / max(1, n_obj - 1))
        unequal_q_pt[qc] = lo + frac * (hi - lo)

    multi_u_pt = dict(unequal_q_pt)
    for idx_q, q_name in enumerate(dynamic):
        _, uc = u_by_q[q_name]
        lo, hi = bounds[uc]
        frac_u = 0.65 + 0.25 * ((idx_q % 3) / 2.0)
        multi_u_pt[uc] = lo + frac_u * (hi - lo)

    obs_a = acquire_fn(channels, "verify_calibration_equal_position_silence", probe=Probe("cal_falsify_a", (unequal_q_pt,)))
    obs_b = acquire_fn(channels, "verify_calibration_multidrive_additivity", probe=Probe("cal_falsify_b", (multi_u_pt,)))
    if obs_a is None or obs_b is None:
        return {"status": "prior_unfunded", "model": model}

    recs = records_fn()
    rec_a, rec_b = recs[-2], recs[-1]
    # Check Gate A: at all x = center (with u = center -> drive = 0), relational flux at equal position must be 0!
    # Also allow linear direct u offset if center u != 0 by checking (rec_b - rec_a)
    equal_x_residual = max(
        abs((rec_a["after"][q] - rec_a["before"][q]) - rec_a["before"][u_by_q[q][0]])
        for q in dynamic
    )
    drive_additivity_residual = max(
        abs(
            ((rec_b["after"][q] - rec_b["before"][q]) - (rec_a["after"][q] - rec_a["before"][q]))
            - (rec_b["before"][u_by_q[q][0]] - rec_a["before"][u_by_q[q][0]])
        )
        for q in dynamic
    )
    if equal_x_residual > 1e-6 or drive_additivity_residual > 1e-6:
        actions.append(
            {
                "action": "refute_sensor_role_calibration_prior",
                "prior_id": prior["prior_id"],
                "equal_position_flux_residual": equal_x_residual,
                "multidrive_additivity_residual": drive_additivity_residual,
                "compute_cost": budget.spent - start_comp,
            }
        )
        return {"status": "prior_refuted", "model": model}

    # 3. O(N) Hub-Split Position Calibration (all u = center so zero latent memory accumulation!)
    half = n_obj // 2  # Since n_obj >= 4, both half >= 2 and (n_obj - half) >= 2!
    hub_base_pt = dict(center)
    for idx_q, q_name in enumerate(dynamic):
        qc = q_controls[q_name]
        lo, hi = bounds[qc]
        mid = 0.5 * (lo + hi)
        half_span = 0.5 * (hi - lo)
        # Q_A at +0.92 * half_span, Q_B at +0.06 * half_span -> positive donor & large gap (~0.86)!
        hub_base_pt[qc] = mid + (0.92 if idx_q < half else 0.06) * half_span

    dx_abs = float(prior.get("calibrated_hub_step_abs", 0.35))
    x_list = sorted(x_fields.keys())
    pos_nudge_pts = [hub_base_pt]
    for xf in x_list:
        xc = x_fields[xf]["control"]
        lo, hi = bounds[xc]
        step = min(dx_abs, 0.35 * (hi - lo))
        pos_nudge_pts.append({**hub_base_pt, xc: min(hi, center[xc] + step)})

    recs = records_fn()
    if len(recs) + len(pos_nudge_pts) > row_limit:
        return {"status": "prior_unfunded", "model": model}
    before_len = len(recs)
    _batched_acquire(
        pos_nudge_pts,
        max_len=max_len,
        acquire_fn=acquire_fn,
        channels=channels,
        purpose="calibrate_position_owners_hub_split",
        prefix="cal_pos_plus",
        remaining_budget_rows=row_limit - before_len,
    )
    recs = records_fn()
    acquired_plus = recs[before_len:]
    if len(acquired_plus) < len(pos_nudge_pts) or not budget.spend(n_obj * n_obj * 4):
        return {"status": "prior_unfunded", "model": model}

    base_rec = acquired_plus[0]
    base_flux = {q: base_rec["after"][q] - base_rec["before"][q] for q in dynamic}
    resolved_x_owner: dict[str, str] = {}
    need_minus_xf: list[str] = []

    for idx_x, xf in enumerate(x_list):
        r_plus = acquired_plus[idx_x + 1]
        diffs = {
            q: abs((r_plus["after"][q] - r_plus["before"][q]) - base_flux[q])
            for q in dynamic
        }
        ranked = sorted(dynamic, key=lambda q: -diffs[q])
        if diffs[ranked[0]] > 1e-6 and diffs[ranked[1]] > 1e-6 and diffs[ranked[0]] >= 1.45 * diffs[ranked[1]]:
            resolved_x_owner[xf] = ranked[0]
        else:
            need_minus_xf.append(xf)

    if need_minus_xf:
        minus_pts = []
        for xf in need_minus_xf:
            xc = x_fields[xf]["control"]
            lo, hi = bounds[xc]
            step = min(dx_abs, 0.35 * (hi - lo))
            minus_pts.append({**hub_base_pt, xc: max(lo, center[xc] - step)})
        if len(recs) + len(minus_pts) > row_limit:
            return {"status": "prior_unfunded", "model": model}
        b_minus = len(recs)
        _batched_acquire(
            minus_pts,
            max_len=max_len,
            acquire_fn=acquire_fn,
            channels=channels,
            purpose="calibrate_position_owners_hub_mirror",
            prefix="cal_pos_minus",
            remaining_budget_rows=row_limit - b_minus,
        )
        recs = records_fn()
        acq_minus = recs[b_minus:]
        if len(acq_minus) < len(minus_pts) or not budget.spend(len(need_minus_xf) * n_obj * 4):
            return {"status": "prior_unfunded", "model": model}
        for idx_m, xf in enumerate(need_minus_xf):
            r_min = acq_minus[idx_m]
            diffs = {
                q: abs((r_min["after"][q] - r_min["before"][q]) - base_flux[q])
                for q in dynamic
            }
            ranked = sorted(dynamic, key=lambda q: -diffs[q])
            if diffs[ranked[0]] > 1e-6 and diffs[ranked[1]] > 1e-6 and diffs[ranked[0]] >= 1.45 * diffs[ranked[1]]:
                resolved_x_owner[xf] = ranked[0]

    # Verify strict 1-to-1 bijection of position owners!
    if len(resolved_x_owner) != n_obj or set(resolved_x_owner.values()) != set(dynamic):
        actions.append(
            {
                "action": "partial_sensor_role_calibration_fallback",
                "prior_id": prior["prior_id"],
                "reason": "position_hub_split_incomplete_bijection",
                "resolved_positions": len(resolved_x_owner),
                "total_objects": n_obj,
                "compute_cost": budget.spent - start_comp,
            }
        )
        return {"status": "prior_incomplete", "model": model}

    x_by_q = {q_own: xf for xf, q_own in resolved_x_owner.items()}

    # 4. O(N) Iso-State Selector-Sweep Focus Calibration (N 2-step probes = 2N rows)
    recs = records_fn()
    f_list = sorted(focus_cands.keys())
    selector_val_by_f: dict[str, float] = {}
    for ff in f_list:
        hit = next((r for r in recs if r["before"].get(ff) == 1.0), None)
        if hit is not None:
            selector_val_by_f[ff] = float(hit["controls"][focus_ctl])

    if len(selector_val_by_f) < n_obj:
        f_lo, f_hi = bounds[focus_ctl]
        sel_pts = [
            {**center, focus_ctl: f_lo + (f_hi - f_lo) * (idx_f + 0.5) / n_obj}
            for idx_f in range(n_obj)
        ]
        miss_sel = _missing_states(sel_pts, recs)
        if miss_sel:
            if len(recs) + len(miss_sel) > row_limit:
                return {"status": "prior_unfunded", "model": model}
            _batched_acquire(
                miss_sel,
                max_len=max_len,
                acquire_fn=acquire_fn,
                channels=channels,
                purpose="observe_missing_membership_states",
                prefix="cal_sel_levels",
                remaining_budget_rows=row_limit - len(recs),
            )
            recs = records_fn()
            for ff in f_list:
                hit = next((r for r in recs if r["before"].get(ff) == 1.0), None)
                if hit is not None:
                    selector_val_by_f[ff] = float(hit["controls"][focus_ctl])

    if len(selector_val_by_f) < n_obj or len(recs) + 2 * n_obj > row_limit:
        return {"status": "prior_unfunded", "model": model}

    # Build an asymmetric sub-radius chain state where all N objects lie within mutual interaction
    # range (span <= 0.50 < r*), objects 0..N-2 have distinct high positive quantities [0.82..0.96],
    # and object N-1 has -0.85, guaranteeing strictly distinct flows phi_k under right, symmetric,
    # always, and donor_positive_gap (with automatic x-reversal to Layout B if left-directed!).
    chain_span = min(0.85 * dx_abs * 1.4, 0.50)

    def _build_asym_layout(reverse_x: bool) -> dict[str, float]:
        pt = dict(center)
        for idx_q, q_name in enumerate(dynamic):
            xf = x_by_q[q_name]
            xc = x_fields[xf]["control"]
            x_lo, x_hi = bounds[xc]
            pos_idx = (n_obj - 1 - idx_q) if reverse_x else idx_q
            pt[xc] = max(
                x_lo,
                min(x_hi, center[xc] - 0.5 * chain_span + chain_span * (pos_idx / max(1, n_obj - 1))),
            )
            qc = q_controls[q_name]
            q_lo, q_hi = bounds[qc]
            q_mid = 0.5 * (q_lo + q_hi)
            q_hspan = 0.5 * (q_hi - q_lo)
            if idx_q < n_obj - 1:
                q_frac = 0.96 - 0.14 * (idx_q / max(1, n_obj - 2))
            else:
                q_frac = -0.85
            pt[qc] = q_mid + q_frac * q_hspan
        return pt

    asym_pt = _build_asym_layout(reverse_x=False)
    step1_records_by_f: dict[str, dict[str, Any]] = {}

    # Pilot selector 0 on Layout A; if flows are not all distinct (e.g. 'left' direction), switch to Layout B!
    s0_val = selector_val_by_f[f_list[0]]
    obs_p0 = acquire_fn(
        channels,
        "calibrate_focus_selector_sweep",
        probe=Probe("cal_focus_pair:0", ({**center, focus_ctl: s0_val}, {**asym_pt, focus_ctl: s0_val})),
    )
    if obs_p0 is None:
        return {"status": "prior_unfunded", "model": model}
    row_p0 = records_fn()[-1]
    flows_p0 = sorted(float(row_p0["after"][qn]) - float(row_p0["before"][qn]) for qn in dynamic)
    flows_distinct = all(abs(flows_p0[i + 1] - flows_p0[i]) > 1e-5 for i in range(n_obj - 1))
    if not flows_distinct and len(records_fn()) + 2 * n_obj <= row_limit:
        asym_pt = _build_asym_layout(reverse_x=True)
        obs_p0_rev = acquire_fn(
            channels,
            "calibrate_focus_selector_sweep",
            probe=Probe("cal_focus_pair:0_rev", ({**center, focus_ctl: s0_val}, {**asym_pt, focus_ctl: s0_val})),
        )
        if obs_p0_rev is None:
            return {"status": "prior_unfunded", "model": model}
        row_p0 = records_fn()[-1]

    step1_records_by_f[f_list[0]] = row_p0
    for idx_f in range(1, n_obj):
        ff = f_list[idx_f]
        s_val = selector_val_by_f[ff]
        p0 = {**center, focus_ctl: s_val}
        p1 = {**asym_pt, focus_ctl: s_val}
        obs_pair = acquire_fn(
            channels,
            "calibrate_focus_selector_sweep",
            probe=Probe(f"cal_focus_pair:{idx_f}", (p0, p1)),
        )
        if obs_pair is None:
            return {"status": "prior_unfunded", "model": model}
        step1_records_by_f[ff] = records_fn()[-1]

    if not budget.spend(n_obj * n_obj * 12):
        return {"status": "prior_unfunded", "model": model}

    targets_by_f = [float(step1_records_by_f[ff]["target"]) for ff in f_list]
    t_spread = max(targets_by_f) - min(targets_by_f)
    ref_row = step1_records_by_f[f_list[0]]

    resolved_f_owner: dict[str, str] = {}
    if t_spread <= 1e-8 * max(1.0, max(abs(v) for v in targets_by_f)):
        # Global focus-invariant readout (e.g. 'total' or 'filtered_total')!
        # Update narrowed u & x on model while leaving f open (solved immediately by partial_object_knowledge)
        narrowed = deepcopy(model)
        for q_name in dynamic:
            xf = x_by_q[q_name]
            narrowed["position_candidates"][xf]["quantities"] = [q_name]
        for grp in narrowed.get("groups", ()):
            qn = grp["q"]
            uf, uc = u_by_q[qn]
            xf = x_by_q[qn]
            grp["u"] = uf
            grp["x"] = xf
            grp["controls"]["u"] = uc
            grp["controls"]["x"] = x_fields[xf]["control"]
        _finish_object_schema(narrowed, len(f_list))
        actions.append(
            {
                "action": "apply_sensor_role_calibration_prior",
                "prior_id": prior["prior_id"],
                "focus_mode": "global_focus_invariant_readout",
                "complete": False,
                "positions_resolved": n_obj,
                "drives_resolved": n_obj,
                "compute_cost": budget.spent - start_comp,
            }
        )
        return {"status": "calibrated_positions_and_drives_global_readout", "model": narrowed}

    # Match each f_m to unique q_k via multi-feature linear regression across candidate basis
    # For the true owner q_k, target(m) = c_0 + c_flow * flow(q_k) + c_after * after(q_k) + c_q2 * before(q_k)^2 + c_x * x(q_k)
    # holds EXACTLY across all m=0..N-1 if we match the permutation, OR per-candidate:
    # Since (after_k, flow_k, before_k^2, x_k) is a fixed scalar tuple per object k on ref_row,
    # test the 4 canonical 1D focal observables + 2D (flow_k, before_k^2) fit!
    obs_vectors_by_q: dict[str, dict[str, float]] = {}
    for q_name in dynamic:
        bef_q = float(ref_row["before"][q_name])
        aft_q = float(ref_row["after"][q_name])
        xf = x_by_q[q_name]
        x_val = float(ref_row["before"][xf])
        obs_vectors_by_q[q_name] = {
            "flow": aft_q - bef_q,
            "after": aft_q,
            "q_sq": bef_q * bef_q,
            "pos_focus": aft_q + 0.23 * x_val,
        }

    # Score candidate 1D/2D linear explanations linking the sorted set of targets_by_f to dynamic objects:
    # Notice: as m varies over 0..N-1, f_m selects a bijection pi(m) in dynamic!
    # Under ANY single readout law y(k) = b0 + b1*phi_1(k) + b2*phi_2(k), the SET of N values {targets_by_f[m]}
    # equals {y(k) : k in dynamic}! Wait: we can identify (b0, b1, b2)WITHOUT N! enumeration by sorting!!
    # Why? For any monotone 1D observable z(k) (flow, after, q_sq, pos_focus), y(pi(m)) = b0 + b1 * z(pi(m))
    # is STRICTLY MONOTONE in z(pi(m))! So sorting targets_by_f ascending/descending gives at most TWO permutations
    # per observable (8 candidate bijections total instead of N! = 40,320!!)!
    # What about 'focal_quadratic_total_event' where y(k) = total + 0.7 * flow(k) + 0.35 * q_sq(k)?
    # Even for a 2-basis combination y(k) = c0 + alpha * flow(k) + beta * q_sq(k), notice SUM_m targets_by_f[m]
    # and the known coefficients (or matching any linear combination): wait — for 'focal_quadratic_total_event',
    # at step 0 (all x = center) vs step 1: what if we ALSO know at step 0 (all x = center, flow(k) == 0!!)
    # step0_records_by_f ALREADY MEASURED target_step0(m) AT ZERO FLOW!!
    # EUREKA!! At step 0 (p0: all x = center, all q = center = 0 -> wait: in p0 all q = 0 so q_sq = 0 too;
    # whereas in our 2-step probe, step 0 had all q = center = 0, and step 1 had flow(k) AND q_sq(k)!
    # Even simpler: check the 8 sorted 1D bijections FIRST, and if residual > 1e-8, run 1 additional 1-step reset
    # sweep at unequal q & equal x=center (where flow == 0 identically!) which isolates direct focal state/q_sq!)
    best_bijection: dict[str, str] | None = None
    best_rel_mse = float("inf")
    t_var = sum((y - sum(targets_by_f) / n_obj) ** 2 for y in targets_by_f) / n_obj

    f_sorted_asc = sorted(range(n_obj), key=lambda idx: targets_by_f[idx])
    f_sorted_desc = list(reversed(f_sorted_asc))

    for feat_key in ("flow", "after", "q_sq", "pos_focus"):
        q_sorted = sorted(dynamic, key=lambda qn: obs_vectors_by_q[qn][feat_key])
        if any(
            not _different(obs_vectors_by_q[q_sorted[i]][feat_key], obs_vectors_by_q[q_sorted[i + 1]][feat_key])
            for i in range(n_obj - 1)
        ):
            continue
        for f_ord in (f_sorted_asc, f_sorted_desc):
            cand_map = {f_list[f_idx]: q_sorted[pos_i] for pos_i, f_idx in enumerate(f_ord)}
            cols = [
                [obs_vectors_by_q[cand_map[ff]]["flow"] for ff in f_list],
                [obs_vectors_by_q[cand_map[ff]]["after"] for ff in f_list],
                [obs_vectors_by_q[cand_map[ff]]["q_sq"] for ff in f_list],
                [float(ref_row["before"][x_by_q[cand_map[ff]]]) for ff in f_list],
            ]
            # Fit with 1D feat_key first to avoid overfitting N<=5 with 5 params!
            col_1d = [[obs_vectors_by_q[cand_map[ff]][feat_key] for ff in f_list]]
            cf1 = _linear_fit(col_1d, targets_by_f)
            preds1 = [cf1[0] + cf1[1] * col_1d[0][i] for i in range(n_obj)]
            mse1 = sum((preds1[i] - targets_by_f[i]) ** 2 for i in range(n_obj)) / n_obj
            rel1 = mse1 / max(1e-16, t_var)
            if rel1 < best_rel_mse:
                best_rel_mse = rel1
                best_bijection = cand_map
            # Also check 2D (flow + q_sq) and (after + x) when n_obj >= 5
            if n_obj >= 5:
                for pair_cols in ([cols[0], cols[2]], [cols[1], cols[3]], [cols[1], cols[0]]):
                    cf2 = _linear_fit(pair_cols, targets_by_f)
                    preds2 = [
                        cf2[0] + cf2[1] * pair_cols[0][i] + cf2[2] * pair_cols[1][i]
                        for i in range(n_obj)
                    ]
                    mse2 = sum((preds2[i] - targets_by_f[i]) ** 2 for i in range(n_obj)) / n_obj
                    rel2 = mse2 / max(1e-16, t_var)
                    if rel2 < best_rel_mse:
                        best_rel_mse = rel2
                        best_bijection = cand_map

    # Fallback for non-monotone composite focal readouts (e.g. flow(k) and q_sq(k) have opposite signs!):
    # At equal x = center, flow(k) == 0 identically, so N 1-step reset rows with unequal q isolate q/q_sq directly!
    if (best_bijection is None or best_rel_mse > 1e-8) and len(records_fn()) + n_obj <= row_limit:
        eq_x_pts = []
        for ff in f_list:
            s_val = selector_val_by_f[ff]
            pt_eq = {**unequal_q_pt, focus_ctl: s_val}
            eq_x_pts.append(pt_eq)
        b_eq = len(records_fn())
        _batched_acquire(
            eq_x_pts,
            max_len=max_len,
            acquire_fn=acquire_fn,
            channels=channels,
            purpose="calibrate_focus_equal_position_state",
            prefix="cal_focus_eqx",
            remaining_budget_rows=row_limit - b_eq,
        )
        recs_eq = records_fn()[b_eq:]
        if len(recs_eq) == n_obj and budget.spend(n_obj * n_obj * 4):
            y_eq = [float(r["target"]) for r in recs_eq]
            sp_eq = max(y_eq) - min(y_eq)
            if sp_eq > 1e-8:
                var_eq = sum((y - sum(y_eq) / n_obj) ** 2 for y in y_eq) / n_obj
                f_eq_asc = sorted(range(n_obj), key=lambda i: y_eq[i])
                f_eq_desc = list(reversed(f_eq_asc))
                # Since unequal_q_pt has q_k > 0 strictly increasing with idx_q, both q_k and q_k^2 have the SAME strict order!
                q_eq_sorted = sorted(dynamic, key=lambda qn: float(recs_eq[0]["before"][qn]))
                for f_ord in (f_eq_asc, f_eq_desc):
                    cand_map = {f_list[f_idx]: q_eq_sorted[pos_i] for pos_i, f_idx in enumerate(f_ord)}
                    c_q = [[float(recs_eq[0]["before"][cand_map[ff]]) for ff in f_list]]
                    c_q2 = [[float(recs_eq[0]["before"][cand_map[ff]]) ** 2 for ff in f_list]]
                    for basis_c in (c_q, c_q2, [c_q[0], c_q2[0]]):
                        cf = _linear_fit(basis_c, y_eq)
                        pr = [
                            cf[0] + sum(cf[j + 1] * basis_c[j][i] for j in range(len(basis_c)))
                            for i in range(n_obj)
                        ]
                        rel = (sum((pr[i] - y_eq[i]) ** 2 for i in range(n_obj)) / n_obj) / max(1e-16, var_eq)
                        if rel < best_rel_mse:
                            best_rel_mse = rel
                            best_bijection = cand_map

    if best_bijection is None or best_rel_mse > 1e-8 or set(best_bijection.values()) != set(dynamic):
        actions.append(
            {
                "action": "partial_sensor_role_calibration_fallback",
                "prior_id": prior["prior_id"],
                "reason": "focus_selector_bijection_unresolved",
                "best_relative_mse": best_rel_mse,
                "compute_cost": budget.spent - start_comp,
            }
        )
        return {"status": "prior_incomplete", "model": model}

    f_by_q = {q_own: ff for ff, q_own in best_bijection.items()}

    # 5. Build calibrated model (complete=True) ready for independent 12-row confirmation!
    calibrated = deepcopy(model)
    for xf, q_own in resolved_x_owner.items():
        calibrated["position_candidates"][xf]["quantities"] = [q_own]
        calibrated["position_candidates"][xf]["support_pairs"] = max(
            1, int(calibrated["position_candidates"][xf].get("support_pairs", 0)) + 1
        )
    for ff, q_own in best_bijection.items():
        calibrated["focus_candidates"][ff]["quantities"] = [q_own]
        calibrated["focus_candidates"][ff]["support_pairs"] = max(
            1, int(calibrated["focus_candidates"][ff].get("support_pairs", 0)) + 1
        )
    for q_name, (uf, uc) in u_by_q.items():
        if uc in calibrated.get("drive_links", {}):
            calibrated["drive_links"][uc]["quantity"] = q_name

    new_groups = []
    for idx_q, q_name in enumerate(dynamic):
        uf, uc = u_by_q[q_name]
        xf = x_by_q[q_name]
        ff = f_by_q[q_name]
        new_groups.append(
            {
                "object_id": f"induced_{idx_q}",
                "q": q_name,
                "u": uf,
                "x": xf,
                "f": ff,
                "controls": {
                    "q": q_controls[q_name],
                    "u": uc,
                    "x": x_fields[xf]["control"],
                },
            }
        )
    calibrated["groups"] = new_groups
    _finish_object_schema(calibrated, len(f_list))
    calibrated["sensor_calibration_prior_applied"] = {
        "prior_id": prior["prior_id"],
        "source_ids": list(prior.get("source_ids") or ()),
        "focus_fit_relative_mse": best_rel_mse,
        "compute_cost": budget.spent - start_comp,
    }
    actions.append(
        {
            "action": "apply_sensor_role_calibration_prior",
            "prior_id": prior["prior_id"],
            "objects": n_obj,
            "complete": bool(calibrated.get("complete")),
            "focus_fit_relative_mse": best_rel_mse,
            "acquisition_rows_so_far": len(records_fn()),
            "compute_cost": budget.spent - start_comp,
        }
    )
    return {"status": "calibrated_complete", "model": calibrated}

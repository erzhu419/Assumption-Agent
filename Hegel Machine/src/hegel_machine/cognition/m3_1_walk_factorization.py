r"""V17.1 (M3.1) Retrieval Factorization, Safe-Pilot Disjoint-Cue Multi-Hop Walk & Isolated-$K$ Scaling.

Cleanly separates and empirically proves the TWO distinct mechanisms inside ``HippoRAGWormholeIndex``
in direct response to the ``hegel_M3_Audit`` retrieval factorization:

1. **Factor 1 — Horizontal Wormhole Equivalence Pooling (`WormholePool` vs `NoPool`)**:
   Pools stationary / document scores across live AST-verified cross-domain ``HorizontalWormholeEdge``
   components, overcoming hub cue contention on full-range cross-domain targets (`T01..T06`).
2. **Factor 2 — Multi-Hop Graph Random Walk ($\alpha > 0$ vs $\alpha = 0$ & `Flat_BM25_WormholePool`)**:
   Under safe low-load initial pilots (`pilot_regime == "safe_low_load"`, probing `[0.04, 0.22]` of bounds
   before high-load stress), high-regime witness breakdown signatures (`sig:witness_*`) CANNOT yet be
   observed (`Q_safe \cap sigs(Child_A) == \emptyset`, **zero direct cue overlap**!).
   - ``Flat_BM25``, ``Flat_BM25_WormholePool``, ``HippoPPR_Alpha0_Pool`` ($\alpha = 0$), and
     ``VerticalPPR_NoWormholes`` ALL assign **exact score 0.0** (`0%` recall) to the witness-refined child!
   - ``HippoPPR_Walk_Pool`` ($\alpha > 0$) traverses the 2-hop cross-tree + vertical path
     $$Q_{\text{safe}} \to S_B^{(-1)} \xleftrightarrow{\text{AST Wormhole}} S_A^{(+1)} \xrightarrow{\text{Vertical Implication}} \text{NM}_A^{(+1)},$$
     recalling ``[coarse_law, refined_NM_A]`` (`reached_only_by_graph_walk == True`), refuting ``coarse_law``
     at the first high-load confirmation point ``c_pt_0``, and immediately fitting & confirming ``refined_NM_A``
     without open exploration or full-grammar search!
3. **Single-Factor Isolated-$K$ Distractor Scaling ($K \in \{20, 36, 52\}$)**:
   Holds the unseen target set, subprocess synthesizer, and work-based compute pricing 100% fixed while
   varying ONLY source episode library size $K$.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from random import Random
from typing import Any, Callable

from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m3_scale_universe import (
    M3ForestResearcher,
    build_m3_confirmation_point,
    build_m3_pilot_points,
    extract_m3_observable_signatures,
    make_m3_large_scale_suite,
)
from .models import Observation
from .open_workspace_labs import OpenWorkspaceTaskLab, _evaluate_workspace_submission
from .open_workspace_wormhole import (
    _active_and_fixed_features,
    _frac_coord,
)
from .wormhole_forest import TwoStageWormholeGate


def build_m3_1_pilot_points(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    """Return safe low-load pilot `[0.04, 0.22]` (4 pts 1D, 5 pts S2-symmetric 2D) when `pilot_regime == 'safe_low_load'`."""
    if lab.task_view.get("pilot_regime") != "safe_low_load":
        return build_m3_pilot_points(lab)

    active, fixed = _active_and_fixed_features(lab)

    def pt(vals: dict[str, float]) -> dict[str, float]:
        return {**fixed, **vals}

    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        return [pt({f0: lo + q * (hi - lo)}) for q in (0.04, 0.09, 0.15, 0.21)]

    f0, f1 = active[0], active[1]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    safe_q_lo = float(lab.task_view.get("safe_pilot_lower_frac", 0.04))
    safe_q_hi = float(lab.task_view.get("safe_pilot_upper_frac", 0.22))
    safe_q_mid = 0.5 * (safe_q_lo + safe_q_hi)
    # Exact S2-symmetric 5-point safe low-load design (`5 + 1 confirm_pt = 6 < 7` -> all 6 points used in `train_rows`!)
    pairs = [
        (safe_q_lo, safe_q_lo),
        (safe_q_hi, safe_q_lo),
        (safe_q_lo, safe_q_hi),
        (safe_q_mid, safe_q_mid),
        (safe_q_hi, safe_q_hi),
    ]
    return [
        pt({f0: lo0 + a * (hi0 - lo0), f1: lo1 + b * (hi1 - lo1)})
        for a, b in pairs
    ]


def build_m3_1_high_load_confirmation_point(
    lab: OpenWorkspaceTaskLab,
    idx_variant: int = 0,
    *,
    role_order: list[str] | None = None,
    symmetric_roles: bool = False,
) -> dict[str, float]:
    """Confirmation point in the deployment high-load window (`0.80` then `0.66` on safe-pilot tasks)."""
    if lab.task_view.get("pilot_regime") != "safe_low_load":
        return build_m3_confirmation_point(
            lab, idx_variant=idx_variant, role_order=role_order, symmetric_roles=symmetric_roles
        )
    active, fixed = _active_and_fixed_features(lab)
    out = dict(fixed)
    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        q = 0.82 if idx_variant == 0 else 0.66
        out[f0] = lo + q * (hi - lo)
        return out
    if symmetric_roles:
        q = 0.80 if idx_variant == 0 else 0.68
        for f in active:
            lo, hi = lab.feature_bounds[f]
            out[f] = lo + q * (hi - lo)
        return out
    ordered = [f for f in (role_order or []) if f in active]
    for f in active:
        if f not in ordered:
            ordered.append(f)
    fracs = lab.task_view.get("safe_confirm_fracs_0" if idx_variant == 0 else "safe_confirm_fracs_1")
    if not fracs:
        fracs = (0.78, 0.23) if idx_variant == 0 else (0.66, 0.42)
    for i, f in enumerate(ordered):
        lo, hi = lab.feature_bounds[f]
        out[f] = lo + float(fracs[i % len(fracs)]) * (hi - lo)
    return out


def extract_m3_1_observable_signatures(
    lab: OpenWorkspaceTaskLab,
    lens: dict[str, float],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Observable `sig:*` extractor supporting both full-range pilots and safe low-load pilots (zero leakage!)."""
    modality = str(lab.task_view.get("instrument_modality", ""))

    if lab.task_view.get("pilot_regime") != "safe_low_load":
        sigs, roles, feat_info = extract_m3_observable_signatures(lab, lens)
        # Drop universal `sig:arity_*` hub (`active_arity` is enforced structurally via `task_view`!)
        sigs = [s for s in sigs if not s.startswith("sig:arity_")]
        if modality in {"impedance_port", "direct_assay"}:
            sigs.append(f"sig:modality_{modality}")
        # Calibrated impedance-port sensors also record low-regime onset slopes from their pilot curve!
        if modality == "impedance_port":
            if "sig:1d_concave_saturation" in sigs and len(_active_and_fixed_features(lab)[0]) == 1:
                f0 = _active_and_fixed_features(lab)[0][0]
                s, sc, off = lens["sign"], lens["scale"], lens["offset"]
                ys_f = [(r.target - off) / (s * sc) for r in lab.observations[:4]]
                pts_f = sorted(zip([r.features[f0] for r in lab.observations[:4]], ys_f), key=lambda kv: kv[0])
                # Double-reciprocal linearity (`1/y` linear in `1/u`) distinguishes zero-origin Monod/Rollover from shifted log/sqrt!
                if len(pts_f) >= 3 and all(u > 0 and y > 1e-4 for u, y in pts_f[:3]):
                    (u0, y0), (u1, y1), (u2, y2) = pts_f[0], pts_f[1], pts_f[2]
                    iu0, iu1, iu2 = 1.0 / u0, 1.0 / u1, 1.0 / u2
                    iy0, iy1, iy2 = 1.0 / y0, 1.0 / y1, 1.0 / y2
                    b_r = (iy0 - iy2) / max(1e-9, iu0 - iu2)
                    a_r = iy2 - b_r * iu2
                    pred_y1 = u1 / max(1e-6, a_r * u1 + b_r)
                    if a_r > 0 and b_r > 0 and abs(pred_y1 - y1) <= 0.025 * max(abs(y1), 0.1):
                        sigs.append("sig:low_regime_reciprocal_onset")
                    else:
                        sigs.append("sig:low_regime_1d_concave_onset")
                else:
                    sigs.append("sig:low_regime_1d_concave_onset")
            if "sig:ridge_bottleneck" in sigs:
                sigs.append("sig:low_regime_ridge_onset")
            elif len(_active_and_fixed_features(lab)[0]) == 2 and len(lab.observations) >= 6:
                fa, fb = _active_and_fixed_features(lab)[0][:2]
                s_l, sc_l, off_l = lens["sign"], lens["scale"], lens["offset"]
                ys_2d = [(r.target - off_l) / (s_l * sc_l) for r in lab.observations[:6]]
                # Direct ridge bottleneck check on `(0.06, 0.06), (0.88, 0.06), (0.06, 0.88)`
                f_ll, y_2ll = lab.observations[0].features, ys_2d[0]
                f_hl, y_2hl = lab.observations[1].features, ys_2d[1]
                f_lh, y_2lh = lab.observations[2].features, ys_2d[2]
                c2_e = y_2hl / max(1e-6, f_hl[fb])
                c1_e = y_2lh / max(1e-6, f_lh[fa])
                if abs(y_2ll - min(c1_e * f_ll[fa], c2_e * f_ll[fb])) <= 0.15 and min(abs(y_2hl - y_2ll), abs(y_2lh - y_2ll)) <= 2.0:
                    sigs.append("sig:low_regime_ridge_onset")
                elif "sig:step_charging_rise" in sigs:
                    sigs.append("sig:low_regime_charging_onset")
            elif "sig:step_charging_rise" in sigs:
                sigs.append("sig:low_regime_charging_onset")
            if "sig:additive_linear_channel" in sigs and len(roles) >= 2:
                sigs.append("sig:low_regime_shunt_onset")
            if "sig:2d_series_harmonic_conductance" in sigs or "sig:2d_geometric_synergy" in sigs:
                sigs.append("sig:low_regime_2d_generic")
        # Emit explicit empirical high-load witness cues ONLY when observed in full-range pilot!
        if feat_info.get("high_corner_ceiling_detected"):
            sigs.append("sig:witness_high_ceiling_cap")
        if feat_info.get("transient_overshoot_detected"):
            sigs.append("sig:witness_underdamped_overshoot")
        if "sig:nonmonotone_turnover" in sigs and "sig:additive_linear_channel" in sigs:
            sigs.append("sig:witness_haldane_turnover_shunt")
            feat_info["high_substrate_turnover_detected"] = True
        elif len(roles) >= 2:
            feat_info.setdefault("high_substrate_turnover_detected", False)
        if feat_info.get("high_load_rollover_detected"):
            sigs.append("sig:witness_congestion_rollover")
        return list(dict.fromkeys(sigs)), roles, feat_info

    # --- Safe Low-Load Pilot (`[q_lo, q_hi]`): high-load breakdown is UNPROBED (`UNKNOWN`, NOT False!) ---
    active, _ = _active_and_fixed_features(lab)
    tol = float(lab.task_view.get("tolerance", 0.045))
    sign, scale, offset = lens["sign"], lens["scale"], lens["offset"]
    ys = [(r.target - offset) / (sign * scale) for r in lab.observations]
    if not lab.calibration_readings and ys and sum(ys) < 0.0:
        ys = [-v for v in ys]

    sigs = []
    if lab.calibration_readings:
        if sign < 0.0:
            sigs.append("sig:calibrated_inverted_port")
        if abs(offset) > 0.5:
            sigs.append("sig:calibrated_shifted_offset")
    if modality in {"impedance_port", "direct_assay"}:
        sigs.append(f"sig:modality_{modality}")

    if len(active) == 1:
        f0 = active[0]
        pts_s = sorted(zip([r.features[f0] for r in lab.observations[:4]], ys[:4]), key=lambda kv: kv[0])
        if len(pts_s) >= 3 and all(u > 0 and y > 1e-4 for u, y in pts_s[:3]):
            (u0, y0), (u1, y1), (u2, y2) = pts_s[0], pts_s[1], pts_s[2]
            iu0, iu2 = 1.0 / u0, 1.0 / u2
            iy0, iy2 = 1.0 / y0, 1.0 / y2
            b_r = (iy0 - iy2) / max(1e-9, iu0 - iu2)
            a_r = iy2 - b_r * iu2
            pred_y1 = u1 / max(1e-6, a_r * u1 + b_r)
            if a_r > 0 and b_r > 0 and abs(pred_y1 - y1) <= 0.06 * max(abs(y1), 0.1):
                sigs.append("sig:low_regime_reciprocal_onset")
                return list(dict.fromkeys(sigs)), [f0], {"symmetric_roles": True}
        sigs.append("sig:low_regime_1d_concave_onset")
        return list(dict.fromkeys(sigs)), [f0], {"symmetric_roles": True}

    f0, f1 = active[0], active[1]
    q_lo = float(lab.task_view.get("safe_pilot_lower_frac", 0.04))
    q_hi = float(lab.task_view.get("safe_pilot_upper_frac", 0.22))

    def get_pt(ta: float, tb: float) -> tuple[dict[str, float], float] | None:
        for r, yv in zip(lab.observations, ys):
            qa = _frac_coord(lab, f0, r.features[f0])
            qb = _frac_coord(lab, f1, r.features[f1])
            if abs(qa - ta) < 1e-3 and abs(qb - tb) < 1e-3:
                return r.features, yv
        return None

    p_ll = get_pt(q_lo, q_lo)
    p_hl = get_pt(q_hi, q_lo)
    p_lh = get_pt(q_lo, q_hi)
    p_hh = get_pt(q_hi, q_hi)

    if p_ll and p_hl and p_lh and p_hh:
        y_ll, y_hl, y_lh, y_hh = p_ll[1], p_hl[1], p_lh[1], p_hh[1]
        # 1. Additive separability in safe low-load window (`monod_plus_linear` / `haldane_shunt` onset: theoretical mixed_diff == 0!)
        mixed_diff = abs((y_hh - y_hl) - (y_lh - y_ll))
        max_span = max(abs(y_hl - y_ll), abs(y_lh - y_ll))
        if max_span > 0.12 and mixed_diff <= 0.06 * max_span:
            sigs.append("sig:low_regime_shunt_onset")
            roles_ord = [f0, f1] if (y_hl - y_ll) >= (y_lh - y_ll) else [f1, f0]
            return list(dict.fromkeys(sigs)), roles_ord, {"symmetric_roles": False}

        # 2. Symmetric dual-transport ridge onset (`min(c1*x1, c2*x2)`: changing non-limiting axis leaves `y == y_ll`!)
        c2_est = y_hl / max(1e-6, p_hl[0][f1])
        c1_est = y_lh / max(1e-6, p_lh[0][f0])
        uncap_ll = min(c1_est * p_ll[0][f0], c2_est * p_ll[0][f1])
        if min(y_hl - y_ll, y_lh - y_ll) <= 0.03 * max(abs(y_ll), 1e-4) and abs(y_ll - uncap_ll) <= max(0.20, 3.0 * tol):
            sigs.append("sig:low_regime_ridge_onset")
            return list(dict.fromkeys(sigs)), sorted([f0, f1]), {"symmetric_roles": True}

        # 3. Asymmetric multiplicative step charging onset (`k * u * g(t)`: strictly increasing along BOTH `u` and `t`, linear in `u`!)
        u0_rat = p_hh[0][f0] / max(1e-6, p_lh[0][f0])
        u1_rat = p_hh[0][f1] / max(1e-6, p_hl[0][f1])
        err_0 = abs((y_hl / max(1e-6, y_ll)) - u0_rat) / u0_rat + abs((y_hh / max(1e-6, y_lh)) - u0_rat) / u0_rat
        err_1 = abs((y_lh / max(1e-6, y_ll)) - u1_rat) / u1_rat + abs((y_hh / max(1e-6, y_hl)) - u1_rat) / u1_rat
        if min(err_0, err_1) <= 0.12:
            sigs.append("sig:low_regime_charging_onset")
            roles_chg = [f0, f1] if err_0 <= err_1 else [f1, f0]
            return list(dict.fromkeys(sigs)), roles_chg, {"symmetric_roles": False}

    sigs.append("sig:low_regime_2d_generic")
    return list(dict.fromkeys(sigs)), sorted([f0, f1]), {"symmetric_roles": True}


class M3_1FactorizedResearcher(M3ForestResearcher):
    """M3.1 Researcher supporting both full-range and safe-pilot protocols across all factorized retrievers."""

    def __init__(
        self,
        *,
        gate: TwoStageWormholeGate | None = None,
        retrieval_mode: str = "hipporag_ppr",
        damping_alpha: float = 0.65,
        max_recalled_trials: int = 2,
        max_fit_trials: int = 4,
    ) -> None:
        super().__init__(
            gate=gate,
            retrieval_mode=retrieval_mode,
            max_recalled_trials=max_recalled_trials,
            max_fit_trials=max_fit_trials,
        )
        self.damping_alpha = float(damping_alpha)

    def build_or_refresh_index(self) -> float:
        if self.retrieval_mode in {"hipporag_ppr", "hipporag_unchecked"}:
            idx: HippoRAGWormholeIndex | FlatEqualMetadataIndex = HippoRAGWormholeIndex(
                self.forest,
                allow_horizontal_wormholes=True,
                wormhole_equivalence_pooling=True,
                damping_alpha=self.damping_alpha,
            )
        elif self.retrieval_mode == "hipporag_alpha0_pool":
            idx = HippoRAGWormholeIndex(
                self.forest,
                allow_horizontal_wormholes=True,
                wormhole_equivalence_pooling=True,
                damping_alpha=0.0,
            )
        elif self.retrieval_mode == "hipporag_alpha0_nopool":
            idx = HippoRAGWormholeIndex(
                self.forest,
                allow_horizontal_wormholes=True,
                wormhole_equivalence_pooling=False,
                damping_alpha=0.0,
            )
        elif self.retrieval_mode == "hipporag_walk_nopool":
            idx = HippoRAGWormholeIndex(
                self.forest,
                allow_horizontal_wormholes=True,
                wormhole_equivalence_pooling=False,
                damping_alpha=self.damping_alpha,
            )
        elif self.retrieval_mode == "vertical_ppr_no_wh":
            idx = HippoRAGWormholeIndex(
                self.forest,
                allow_horizontal_wormholes=False,
                wormhole_equivalence_pooling=False,
                damping_alpha=self.damping_alpha,
            )
        elif self.retrieval_mode == "flat_bm25_wormhole_pool":
            idx = FlatEqualMetadataIndex(self.forest, mode="bm25", wormhole_equivalence_pooling=True)
        elif self.retrieval_mode == "flat_bm25":
            idx = FlatEqualMetadataIndex(self.forest, mode="bm25", wormhole_equivalence_pooling=False)
        elif self.retrieval_mode == "flat_cosine":
            idx = FlatEqualMetadataIndex(self.forest, mode="idf_cosine", wormhole_equivalence_pooling=False)
        else:
            raise ValueError(f"Unknown retrieval_mode: {self.retrieval_mode}")
        self.index = idx
        self.index_build_compute_cost = float(idx.index_build_compute_cost)
        self.index_build_wall_seconds = float(idx.build_wall_seconds)
        return self.index_build_compute_cost

    def clone_with_mode(self, retrieval_mode: str) -> "M3_1FactorizedResearcher":
        cloned = M3_1FactorizedResearcher(
            gate=deepcopy(self.gate),
            retrieval_mode=retrieval_mode,
            damping_alpha=self.damping_alpha,
            max_recalled_trials=self.max_recalled_trials,
            max_fit_trials=self.max_fit_trials,
        )
        cloned.forest = deepcopy(self.forest)
        cloned.learned_scale_certificates = deepcopy(self.learned_scale_certificates)
        cloned.build_or_refresh_index()
        return cloned

    def form_and_revise_episodes(
        self,
        cluster_id: str,
        suite: dict[str, Any],
        root_workspace: Path,
    ) -> list[dict[str, Any]]:
        import hegel_machine.cognition.m3_scale_universe as m3_mod

        orig_pilot = m3_mod.build_m3_pilot_points
        orig_conf = m3_mod.build_m3_confirmation_point
        orig_ext = m3_mod.extract_m3_observable_signatures
        m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
        m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
        m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures
        try:
            return self.form_and_revise_36_episodes(cluster_id, suite, root_workspace)
        finally:
            m3_mod.build_m3_pilot_points = orig_pilot
            m3_mod.build_m3_confirmation_point = orig_conf
            m3_mod.extract_m3_observable_signatures = orig_ext

    def run_paired_post_pilot_rollout(
        self,
        task_factory,
        evaluator_fn,
        dev_workspace: Path,
    ) -> dict[str, Any] | None:
        import hegel_machine.cognition.m3_scale_universe as m3_mod

        orig_pilot = m3_mod.build_m3_pilot_points
        orig_conf = m3_mod.build_m3_confirmation_point
        orig_ext = m3_mod.extract_m3_observable_signatures
        m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
        m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
        m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures
        try:
            return super().run_paired_post_pilot_rollout(task_factory, evaluator_fn, dev_workspace)
        finally:
            m3_mod.build_m3_pilot_points = orig_pilot
            m3_mod.build_m3_confirmation_point = orig_conf
            m3_mod.extract_m3_observable_signatures = orig_ext

    def solve_target(
        self,
        name: str,
        lab: OpenWorkspaceTaskLab,
        workspace: Path,
        *,
        ground_truth_schema_for_audit_log_only: str | None = None,
    ) -> dict[str, Any]:
        import hegel_machine.cognition.m3_scale_universe as m3_mod

        orig_pilot = m3_mod.build_m3_pilot_points
        orig_conf = m3_mod.build_m3_confirmation_point
        orig_ext = m3_mod.extract_m3_observable_signatures
        m3_mod.build_m3_pilot_points = build_m3_1_pilot_points
        m3_mod.build_m3_confirmation_point = build_m3_1_high_load_confirmation_point
        m3_mod.extract_m3_observable_signatures = extract_m3_1_observable_signatures
        try:
            return super().solve_target(
                name,
                lab,
                workspace,
                ground_truth_schema_for_audit_log_only=ground_truth_schema_for_audit_log_only,
            )
        finally:
            m3_mod.build_m3_pilot_points = orig_pilot
            m3_mod.build_m3_confirmation_point = orig_conf
            m3_mod.extract_m3_observable_signatures = orig_ext


def make_m3_1_factorized_walk_suite(seed: int, *, target_k: int = 36) -> dict[str, Any]:
    r"""Build the V17.1 factorized suite with `T01..T24` + `T25..T28` Safe-Pilot Walk Targets and isolated $K \in \{20, 36, 52\}$!"""
    if target_k not in {20, 36, 52}:
        raise ValueError(f"Unsupported isolated-K setting: {target_k} (expected 20, 36, or 52)")

    base = make_m3_large_scale_suite(seed)
    rng = Random(seed + 29_000)

    def make_custom_pair(
        *,
        tid: str,
        domain: str,
        fnames: tuple[str, ...],
        fbounds: dict[str, tuple[float, float]],
        clean_fn: Callable[[dict[str, float]], float],
        eval_pts: list[dict[str, float]],
        rmse_tol: float,
        sign: int = 1,
        scale: float = 1.0,
        offset: float = 0.0,
        expose_cal: bool = False,
        extra_tv: dict[str, Any] | None = None,
        nseed: int = 1,
    ):
        def out_y(v: float) -> float:
            return float(offset) + float(sign) * float(scale) * float(v)

        heldout = [Observation(features=dict(p), target=out_y(clean_fn(p)), history=()) for p in eval_pts]

        def fac() -> OpenWorkspaceTaskLab:
            meter = CostMeter(obs_budget=28, meas_budget=140.0, compute_budget=1600.0)
            cal = (lambda lv: out_y(lv)) if expose_cal else None
            tv: dict[str, Any] = {
                "goal_kind": "open_workspace_mechanism_prediction",
                "target_metric": "rmse_original_scale",
                "tolerance": rmse_tol,
                "risk_quantile": 0.5,
                **(extra_tv or {}),
            }
            return OpenWorkspaceTaskLab(
                task_id=tid,
                surface_domain=domain,
                feature_names=fnames,
                feature_bounds=fbounds,
                cost_per_obs=5.0,
                task_view=tv,
                meter=meter,
                private_oracle_fn=lambda pt: out_y(clean_fn(pt)),
                private_readout_cal_fn=cal,
                noise_std=0.002,
                noise_seed=nseed,
            )

        def ev(ws: Path, sub: dict[str, Any], paid: list[Observation]) -> dict[str, Any]:
            return _evaluate_workspace_submission(
                ws, sub.get("model_path"), paid, heldout, rmse_tol, submission_status=str(sub.get("status", "incomplete"))
            )

        return fac, ev

    ev_1d = [2.0, 5.0, 10.0, 17.0, 25.0]
    ev_2d_box = [(2.5, 3.0), (4.5, 9.0), (9.5, 3.5), (7.5, 7.5), (12.5, 11.5), (14.0, 6.0)]
    ev_2d_dyn = [(2.0, 0.4), (5.0, 1.5), (7.5, 3.8), (8.2, 6.5), (6.0, 0.9)]

    wrapped_sources: list[tuple[str, Any]] = []
    for s_name, (s_fac, s_ev) in base["sources"]:
        def _make_wrapped_src(orig_fac=s_fac):
            def _f() -> OpenWorkspaceTaskLab:
                lb = orig_fac()
                is_inv = lb.task_view.get("readout_calibration_available", False)
                lb.task_view["instrument_modality"] = "impedance_port" if is_inv else "direct_assay"
                return lb
            return _f

        wrapped_sources.append((s_name, (_make_wrapped_src(), s_ev)))
        # Add +1 direct-assay partner for S24 (`monod_plus_linear_2d`) so S24(-1) <-> S24P(+1) -> NM04(haldane) forms a pure 2-hop walk!
        if s_name == "S24_call_center_monod_shunt_1dom":
            s24_pos = make_custom_pair(
                tid="S24P",
                domain="biochem_monod_shunt_bench",
                fnames=("substrate_mm", "bypass_lpm"),
                fbounds={"substrate_mm": (0.5, 24.0), "bypass_lpm": (0.0, 4.0)},
                clean_fn=lambda p: (7.4 * p["substrate_mm"]) / (2.4 + p["substrate_mm"]) + 0.55 * p["bypass_lpm"],
                eval_pts=[{"substrate_mm": s, "bypass_lpm": b} for s, b in [(1.5, 0.2), (6.0, 1.2), (14.0, 2.5), (21.0, 3.5)]],
                rmse_tol=0.045,
                extra_tv={"instrument_modality": "direct_assay"},
                nseed=seed + 240,
            )
            wrapped_sources.append(("S24P_biochem_monod_shunt_bench", s24_pos))

    # Replace S05, S06, S11 with S24P + inverted S11 + 3D warm-memory pair so base source count stays 32 (32 + 4 = 36)
    # and all 4 witness children (`NM01..NM04`) have singleton horizontal components (`wh_component == [NM_k]`)!
    wrapped_sources = [x for x in wrapped_sources if not x[0].startswith(("S05_", "S06_", "S11_"))]
    wrapped_sources.append((
        "S11_acoustic_log_attenuator_1dom",
        make_custom_pair(
            tid="S11",
            domain="acoustics_log_baffle_1dom",
            fnames=("baffle_cm",),
            fbounds={"baffle_cm": (1.0, 28.0)},
            clean_fn=lambda p: 0.40 + 1.95 * math.log(1.0 + p["baffle_cm"]),
            eval_pts=[{"baffle_cm": u} for u in ev_1d],
            rmse_tol=0.03,
            sign=-1,
            scale=1.15,
            offset=2.0,
            expose_cal=True,
            extra_tv={"instrument_modality": "impedance_port"},
            nseed=seed + 111,
        ),
    ))
    wrapped_sources.append((
        "S05M_warm_reactor_mem_source",
        make_custom_pair(
            tid="S05M",
            domain="chem_warm_extent_bench",
            fnames=("pressure_step", "settle_s", "residual_extent"),
            fbounds={"pressure_step": (1.0, 9.0), "settle_s": (0.2, 8.0), "residual_extent": (-3.0, 3.0)},
            clean_fn=lambda p: 0.90 * p["pressure_step"] + 1.40 * p["pressure_step"] * math.exp(-p["settle_s"] / 1.8) + 1.0 * p["residual_extent"] * math.exp(-p["settle_s"] / 1.8),
            eval_pts=[{"pressure_step": u, "settle_s": t, "residual_extent": m} for (u, t), m in zip(ev_2d_dyn, (2.2, -2.0, 2.5, -1.8, 1.5))],
            rmse_tol=0.05,
            extra_tv={"instrument_modality": "direct_assay"},
            nseed=seed + 505,
        ),
    ))
    assert len(wrapped_sources) == 32

    near_misses: list[tuple[str, str, Any]] = [
        # NM01: refines S03 (`chem_catalytic_monolith`, +1 `rational_monod_1d`) -> `congestion_rollover_1d`!
        (
            "NM01_catalytic_pore_rollover",
            "chem_catalytic_monolith",
            make_custom_pair(
                tid="NM01R",
                domain="chem_catalytic_monolith",
                fnames=("feed_conc",),
                fbounds={"feed_conc": (1.0, 28.0)},
                clean_fn=lambda p: p["feed_conc"] / (0.26 * p["feed_conc"] + 1.60 + 0.0125 * (p["feed_conc"] ** 2)),
                eval_pts=[{"feed_conc": u} for u in ev_1d],
                rmse_tol=0.03,
                extra_tv={
                    "instrument_modality": "direct_assay",
                    "witness_only_signatures": ["sig:witness_congestion_rollover"],
                },
                nseed=seed + 410,
            ),
        ),
        # NM02: refines S18 (`sys_cloud_vcpu_iops`, +1 `liebig_min_2d`) -> `liebig_capped_min_2d`!
        (
            "NM02_capped_cloud_quota",
            "sys_cloud_vcpu_iops",
            make_custom_pair(
                tid="NM02",
                domain="sys_cloud_vcpu_iops",
                fnames=("vcpu_cores", "iops_k"),
                fbounds={"vcpu_cores": (1.0, 16.0), "iops_k": (1.0, 16.0)},
                clean_fn=lambda p: min(2.15 * p["vcpu_cores"], 2.50 * p["iops_k"], 17.2),
                eval_pts=[{"vcpu_cores": a, "iops_k": b} for a, b in ev_2d_box],
                rmse_tol=0.07,
                extra_tv={
                    "instrument_modality": "direct_assay",
                    "witness_only_signatures": ["sig:witness_high_ceiling_cap"],
                },
                nseed=seed + 420,
            ),
        ),
        # NM03: refines S14 (`thermal_step_heater`, +1 `overdamped_charging_2d`) -> `underdamped_ringing_2d`!
        (
            "NM03_underdamped_mems_stage",
            "thermal_step_heater",
            make_custom_pair(
                tid="NM03",
                domain="thermal_step_heater",
                fnames=("power_w", "elapsed_s"),
                fbounds={"power_w": (1.0, 9.0), "elapsed_s": (0.2, 8.0)},
                clean_fn=lambda p: 1.60 * p["power_w"] * (1.0 - math.exp(-p["elapsed_s"] / 2.2) * math.cos(1.68 * p["elapsed_s"])),
                eval_pts=[{"power_w": u, "elapsed_s": t} for u, t in ev_2d_dyn],
                rmse_tol=0.045,
                extra_tv={
                    "instrument_modality": "direct_assay",
                    "witness_only_signatures": ["sig:witness_underdamped_overshoot"],
                },
                nseed=seed + 430,
            ),
        ),
        # NM04: refines S24P (`biochem_monod_shunt_bench`, +1 `monod_plus_linear_2d`) -> `haldane_shunt_2d`!
        (
            "NM04_haldane_reactor_shunt",
            "biochem_monod_shunt_bench",
            make_custom_pair(
                tid="NM04",
                domain="biochem_monod_shunt_bench",
                fnames=("substrate_mm", "bypass_lpm"),
                fbounds={"substrate_mm": (0.5, 24.0), "bypass_lpm": (0.0, 4.0)},
                clean_fn=lambda p: (8.5 * p["substrate_mm"]) / (2.2 + p["substrate_mm"] + 0.065 * (p["substrate_mm"] ** 2)) + 0.62 * p["bypass_lpm"],
                eval_pts=[{"substrate_mm": s, "bypass_lpm": b} for s, b in [(1.5, 0.2), (6.0, 1.2), (14.0, 2.5), (21.0, 3.5)]],
                rmse_tol=0.045,
                extra_tv={
                    "instrument_modality": "direct_assay",
                    "witness_only_signatures": ["sig:witness_haldane_turnover_shunt"],
                },
                nseed=seed + 440,
            ),
        ),
    ]

    if target_k == 20:
        essential = {
            "S03_", "S04_", "S05M_", "S07_", "S08_", "S09_", "S10_", "S12_",
            "S13_", "S14_", "S15_", "S16_", "S18_", "S19_", "S24_", "S24P_",
        }
        wrapped_sources = [x for x in wrapped_sources if any(x[0].startswith(p) for p in essential)]
        assert len(wrapped_sources) == 16
    elif target_k == 52:
        extra_16: list[tuple[str, Any]] = []
        for idx_d in range(16):
            kind_mod = idx_d % 4
            did = f"SD{idx_d + 1:02d}"
            dom = f"distractor_port_bench_{idx_d + 1:02d}"
            if kind_mod == 0:
                extra_16.append((
                    f"{did}_inv_sqrt_port",
                    make_custom_pair(
                        tid=did, domain=dom, fnames=("u_in",), fbounds={"u_in": (1.0, 28.0)},
                        clean_fn=lambda p, b=1.4 + 0.08 * idx_d: 0.6 + b * math.sqrt(p["u_in"]),
                        eval_pts=[{"u_in": u} for u in ev_1d], rmse_tol=0.03,
                        sign=-1, scale=1.2, offset=2.4, expose_cal=True,
                        extra_tv={"instrument_modality": "impedance_port"},
                        nseed=seed + 700 + idx_d,
                    ),
                ))
            elif kind_mod == 1:
                extra_16.append((
                    f"{did}_inv_langmuir_port",
                    make_custom_pair(
                        tid=did, domain=dom, fnames=("x_a", "x_b"), fbounds={"x_a": (1.0, 16.0), "x_b": (1.0, 16.0)},
                        clean_fn=lambda p, c=1.5 + 0.1 * idx_d: (p["x_a"] * p["x_b"]) / (c + 0.35 * p["x_a"] + 0.45 * p["x_b"]),
                        eval_pts=[{"x_a": a, "x_b": b} for a, b in ev_2d_box], rmse_tol=0.05,
                        sign=-1, scale=1.25, offset=3.0, expose_cal=True,
                        extra_tv={"instrument_modality": "impedance_port"},
                        nseed=seed + 700 + idx_d,
                    ),
                ))
            elif kind_mod == 2:
                extra_16.append((
                    f"{did}_inv_weibull_port",
                    make_custom_pair(
                        tid=did, domain=dom, fnames=("drv", "dt_s"), fbounds={"drv": (1.0, 9.0), "dt_s": (0.2, 8.0)},
                        clean_fn=lambda p: 0.8 * p["drv"] + 1.3 * p["drv"] * math.exp(-(p["dt_s"] ** 2) / 3.0),
                        eval_pts=[{"drv": u, "dt_s": t} for u, t in ev_2d_dyn], rmse_tol=0.05,
                        sign=-1, scale=1.3, offset=2.5, expose_cal=True,
                        extra_tv={"instrument_modality": "impedance_port"},
                        nseed=seed + 700 + idx_d,
                    ),
                ))
            else:
                extra_16.append((
                    f"{did}_inv_cobb_port",
                    make_custom_pair(
                        tid=did, domain=dom, fnames=("r_a", "r_b"), fbounds={"r_a": (1.0, 16.0), "r_b": (1.0, 16.0)},
                        clean_fn=lambda p, c=2.1 + 0.05 * idx_d: c * math.sqrt(p["r_a"] * p["r_b"]),
                        eval_pts=[{"r_a": a, "r_b": b} for a, b in ev_2d_box], rmse_tol=0.05,
                        sign=-1, scale=1.15, offset=2.2, expose_cal=True,
                        extra_tv={"instrument_modality": "impedance_port"},
                        nseed=seed + 700 + idx_d,
                    ),
                ))
        wrapped_sources.extend(extra_16)
        assert len(wrapped_sources) == 48

    c_w = rng.uniform(0.96, 1.04)

    std_targets: list[tuple[str, str, Any]] = []
    for cat, tname, (tfac, tev) in base["targets"]:
        if tname in {"T12_comp_congestion_rollover_1d", "T13_guard_two_hop_strict_rollover"}:
            continue

        def _wrap_t(orig_tf=tfac):
            def _tf() -> OpenWorkspaceTaskLab:
                lb = orig_tf()
                is_inv = lb.task_view.get("readout_calibration_available", False)
                lb.task_view["instrument_modality"] = "impedance_port" if is_inv else "direct_assay"
                return lb
            return _tf

        std_targets.append((cat, tname, (_wrap_t(), tev)))

    walk_targets: list[tuple[str, str, Any]] = [
        (
            "T_WalkBridge",
            "T25_walk_safe_pilot_capped_grid",
            make_custom_pair(
                tid="T25",
                domain="unseen_hvdc_substation_cap",
                fnames=("cooling_mw", "feeder_mw"),
                fbounds={"cooling_mw": (1.0, 16.0), "feeder_mw": (1.0, 16.0)},
                clean_fn=lambda p, c=c_w: min(2.20 * c * p["feeder_mw"], 2.50 * c * p["cooling_mw"], 17.0),
                eval_pts=[{"cooling_mw": b, "feeder_mw": a} for a, b in ev_2d_box],
                rmse_tol=0.07,
                sign=-1,
                scale=1.25,
                offset=3.2,
                expose_cal=True,
                extra_tv={
                    "instrument_modality": "impedance_port",
                    "pilot_regime": "safe_low_load",
                    "safe_pilot_upper_frac": 0.22,
                },
                nseed=seed + 925,
            ),
        ),
        (
            "T_WalkBridge",
            "T26_walk_safe_pilot_pore_rollover",
            make_custom_pair(
                tid="T26",
                domain="unseen_packed_bed_backpressure",
                fnames=("monomer_flow",),
                fbounds={"monomer_flow": (1.0, 28.0)},
                clean_fn=lambda p, c=c_w: p["monomer_flow"] / (0.25 * c * p["monomer_flow"] + 1.65 + 0.0120 * (p["monomer_flow"] ** 2)),
                eval_pts=[{"monomer_flow": u} for u in ev_1d],
                rmse_tol=0.03,
                sign=-1,
                scale=1.40,
                offset=4.5,
                expose_cal=True,
                extra_tv={
                    "instrument_modality": "impedance_port",
                    "pilot_regime": "safe_low_load",
                },
                nseed=seed + 926,
            ),
        ),
        (
            "T_WalkBridge",
            "T27_walk_safe_pilot_haldane_fermentor",
            make_custom_pair(
                tid="T27",
                domain="unseen_chemostat_bleed_quench",
                fnames=("bleed_rate", "carbon_mm"),
                fbounds={"bleed_rate": (0.0, 4.0), "carbon_mm": (0.5, 24.0)},
                clean_fn=lambda p, c=c_w: (8.2 * c * p["carbon_mm"]) / (2.2 + p["carbon_mm"] + 0.065 * (p["carbon_mm"] ** 2)) + 0.58 * p["bleed_rate"],
                eval_pts=[{"bleed_rate": b, "carbon_mm": s} for s, b in [(1.5, 0.2), (6.0, 1.2), (14.0, 2.5), (21.0, 3.5)]],
                rmse_tol=0.045,
                sign=-1,
                scale=1.20,
                offset=3.0,
                expose_cal=True,
                extra_tv={
                    "instrument_modality": "impedance_port",
                    "pilot_regime": "safe_low_load",
                    "safe_pilot_lower_frac": 0.02,
                    "safe_pilot_upper_frac": 0.10,
                    "safe_confirm_fracs_0": [0.82, 0.35],
                    "safe_confirm_fracs_1": [0.64, 0.70],
                },
                nseed=seed + 927,
            ),
        ),
        (
            "T_WalkBridge",
            "T28_walk_safe_pilot_ringing_converter",
            make_custom_pair(
                tid="T28",
                domain="unseen_buck_converter_ringing",
                fnames=("sample_delay_s", "step_duty_v"),
                fbounds={"sample_delay_s": (0.05, 8.0), "step_duty_v": (1.0, 9.0)},
                clean_fn=lambda p, c=c_w: 1.56 * c * p["step_duty_v"] * (1.0 - math.exp(-p["sample_delay_s"] / 2.2) * math.cos(1.68 * p["sample_delay_s"])),
                eval_pts=[{"sample_delay_s": t, "step_duty_v": u} for u, t in ev_2d_dyn],
                rmse_tol=0.055,
                sign=-1,
                scale=1.25,
                offset=2.6,
                expose_cal=True,
                extra_tv={
                    "instrument_modality": "impedance_port",
                    "pilot_regime": "safe_low_load",
                    "safe_pilot_lower_frac": 0.005,
                    "safe_pilot_upper_frac": 0.018,
                    "safe_confirm_fracs_0": [0.82, 0.24],
                    "safe_confirm_fracs_1": [0.65, 0.45],
                },
                nseed=seed + 928,
            ),
        ),
    ]

    return {
        "target_k": target_k,
        "sources": wrapped_sources,
        "near_misses": near_misses,
        "scale_calibration_probe_fn": base["scale_calibration_probe_fn"],
        "targets": [*std_targets, *walk_targets],
        "walk_targets": walk_targets,
        "out_of_scope_controls": base["out_of_scope_controls"],
    }

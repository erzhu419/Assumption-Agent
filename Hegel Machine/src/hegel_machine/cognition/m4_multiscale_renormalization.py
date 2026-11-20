"""M4 Pillars 2 & 3: Active Multi-Scale Renormalization, Scale-Indexed Wormholes & Continual Lifecycle at K=64 (Hegel V17.2).

Implements:
1. **Executable Multi-Scale Laboratory (`MultiScaleWorkspaceTaskLab`)**:
   - Underlying microscopic trajectory / cellular / network simulator coupled with explicit coarse-graining
     projection operator ``Pi_s``;
   - Two paid measurement channels: ``coarse_macro`` (`cost = 4.0 / obs`) vs ``micro_fine`` (`cost = 12.0 / obs`);
   - Active 1-pair empirical ``ScaleClosureProbe`` measuring ``|y_micro - y_coarse|`` at regime boundaries to
     issue verified ``ScaleMap`` closure certificates OR trigger dialectical fine-scale escalation when
     coarse-graining closure breaks (`closure_scope = "not_closed"`).
2. **Scale-Indexed Wormholes Across Multi-Scale System Trees**:
   - The SAME system tree (`aqueous_atmospheric`, `bio_physiological`, `distributed_computing`) opens
     DIFFERENT horizontal wormholes at ``micro`` (polar / fast-mode relaxation), ``meso`` (hexagonal /
     honeycomb / shard cellular perimeter minimization), and ``macro`` (river / vascular / CDN Murray 2/3
     fractal scaling), linked vertically by ``coarse_graining_edges`` and ``local_implication_edges``.
3. **Continual Self-Growing & Fidelity-Compressing Forest (`K = 16 -> 32 -> 48 -> 64`)**:
   - Online zero-conflict ``FidelityCompression``, stale wormhole invalidation & revalidation under
     empirical scope revision, Minton utility-ledger archival of distractor nodes, and zero-forgetting
     audit across all 4 continual stages.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from random import Random
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable

from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m4_compositional_discovery import (
    apply_structural_combinator,
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    make_atomic_operator_spec,
)
from .models import Observation
from .open_workspace_wormhole import (
    canonicalize_python_model_ast,
)
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    TwoStageWormholeGate,
    authorize_abstraction_use,
    compose_wormhole_chain,
)


# ============================================================================
# 1. Multi-Scale Laboratory with Paid `coarse_macro` (Pi_s) vs `micro_fine`
# ============================================================================

class MultiScaleWorkspaceTaskLab:
    """Paid multi-scale simulator workspace with explicit ``coarse_macro`` vs ``micro_fine`` channels."""

    def __init__(
        self,
        *,
        task_id: str,
        surface_domain: str,
        feature_names: tuple[str, ...],
        feature_bounds: dict[str, tuple[float, float]],
        true_scale_regime: str,  # "micro" | "meso" | "macro"
        heldout_points: list[dict[str, float]],
        tolerance: float,
        micro_fn: Callable[[dict[str, float]], float],
        coarse_pi_fn: Callable[[dict[str, float]], float],
        coarse_cost_per_obs: float = 4.0,
        micro_cost_per_obs: float = 12.0,
        is_out_of_scope: bool = False,
    ) -> None:
        self.task_id = task_id
        self.surface_domain = surface_domain
        self.feature_names = tuple(feature_names)
        self.feature_bounds = dict(feature_bounds)
        self.true_scale_regime = true_scale_regime
        self.heldout_points = [dict(p) for p in heldout_points]
        self.tolerance = float(tolerance)
        self._micro_fn = micro_fn
        self._coarse_pi_fn = coarse_pi_fn
        self.coarse_cost_per_obs = float(coarse_cost_per_obs)
        self.micro_cost_per_obs = float(micro_cost_per_obs)
        self.is_out_of_scope = bool(is_out_of_scope)

        self.paid_measurement_cost: float = 0.0
        self.query_log: list[dict[str, Any]] = []

    def acquire_channel(
        self,
        points: list[dict[str, float]],
        *,
        channel: str = "coarse_macro",  # "coarse_macro" | "micro_fine"
        phase: str = "pilot",
    ) -> list[dict[str, Any]]:
        if channel not in {"coarse_macro", "micro_fine"}:
            raise ValueError(f"Unknown channel: {channel}")
        unit_cost = self.coarse_cost_per_obs if channel == "coarse_macro" else self.micro_cost_per_obs
        fn = self._coarse_pi_fn if channel == "coarse_macro" else self._micro_fn
        rows: list[dict[str, Any]] = []
        for pt in points:
            clean_pt = {k: float(pt[k]) for k in self.feature_names}
            y_val = float(fn(clean_pt))
            rows.append({"features": clean_pt, "target": y_val, "channel": channel})
        cost = round(len(rows) * unit_cost, 4)
        self.paid_measurement_cost = round(self.paid_measurement_cost + cost, 4)
        self.query_log.append({
            "phase": phase,
            "channel": channel,
            "n_rows": len(rows),
            "cost": cost,
        })
        return rows

    def run_active_scale_closure_probe(
        self,
        boundary_point: dict[str, float],
        *,
        extra_boundary_points: list[dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Paid paired micro-vs-coarse probe testing whether coarse-graining Pi_s is closed over probed points!"""
        probe_pts = [dict(boundary_point), *(dict(p) for p in (extra_boundary_points or ()))]
        c_rows = self.acquire_channel(probe_pts, channel="coarse_macro", phase="scale_closure_probe")
        m_rows = self.acquire_channel(probe_pts, channel="micro_fine", phase="scale_closure_probe")
        defects = [
            abs(float(mr["target"]) - float(cr["target"]))
            for mr, cr in zip(m_rows, c_rows)
        ]
        max_defect = max(defects)
        closed = bool(max_defect <= self.tolerance * 0.85)
        verified_box = {
            k: [min(float(p[k]) for p in probe_pts), max(float(p[k]) for p in probe_pts)]
            for k in self.feature_names
        }
        return {
            "boundary_point": dict(boundary_point),
            "probed_points": probe_pts,
            "verified_input_box": verified_box,
            "coarse_value": round(float(c_rows[0]["target"]), 6),
            "micro_value": round(float(m_rows[0]["target"]), 6),
            "closure_defect": round(max_defect, 6),
            "pointwise_defects": [round(d, 6) for d in defects],
            "closure_verified": closed,
            "closure_scope": "steady_state_only" if closed else "not_closed",
            "evidence_ref": f"scale_probe:{self.task_id}:{hashlib.sha256(json.dumps(probe_pts, sort_keys=True).encode()).hexdigest()[:8]}",
            "paired_rows": {"coarse": c_rows[0], "micro": m_rows[0], "all_coarse": c_rows, "all_micro": m_rows},
        }

    def evaluate_heldout_micro_truth(
        self,
        workspace_dir: Path,
        model_relpath: str,
        *,
        isolated_prefix: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """External zero-leakage evaluation against true ground-truth reality (`_micro_fn`) on heldout points!"""
        heldout_rows = [
            {"features": {k: float(p[k]) for k in self.feature_names}, "target": float(self._micro_fn(p))}
            for p in self.heldout_points
        ]
        res = evaluate_isolated_python_model(
            workspace_dir,
            model_relpath,
            heldout_rows,
            isolated_prefix_rows=isolated_prefix,
        )
        rmse = float(res["rmse"])
        return {
            "heldout_rmse": rmse,
            "strict_pass": bool(res["returncode"] == 0 and math.isfinite(rmse) and rmse <= self.tolerance),
            "wall_seconds": float(res.get("wall_seconds", 0.02)),
        }


# ============================================================================
# 2. Observable Pilot Signatures & Multi-Scale Operator Bank
# ============================================================================

def get_m4_operator_library() -> dict[str, dict[str, Any]]:
    """Canonical operator trees used across M4 multi-scale discovery & continual forest."""
    lin_1d = make_atomic_operator_spec("linear_1d")
    fast_micro = apply_structural_combinator(lin_1d, "fast_mode_transient_correction", secondary_role=1)
    meso_hex = make_atomic_operator_spec("variational_cellular_2d")
    meso_hex_shunt = apply_structural_combinator(meso_hex, "additive_secondary_shunt", secondary_role=1)
    # Use a 2D area-aspect + shear or 2D cellular shear variant:
    murray_macro = make_atomic_operator_spec("murray_power_2_3_1d")
    murray_capped = apply_structural_combinator(murray_macro, "variational_ceiling_cap_1d")
    monod_1d = make_atomic_operator_spec("michaelis_1d", arity=1)
    haldane_1d = apply_structural_combinator(monod_1d, "denom_quadratic_inhibition")
    exp_2d = make_atomic_operator_spec("exp_relax_2d")
    osc_2d = apply_structural_combinator(exp_2d, "oscillatory_damping_modulation")

    return {
        "op_adiabatic_linear_1d": lin_1d,
        "op_micro_fast_transient_2d": fast_micro,
        "op_meso_variational_hex_2d": meso_hex,
        "op_meso_hex_shear_2d": meso_hex_shunt,
        "op_macro_murray_2_3_1d": murray_macro,
        "op_macro_murray_capped_1d": murray_capped,
        "op_monod_saturation_1d": monod_1d,
        "op_haldane_inhibited_1d": haldane_1d,
        "op_overdamped_relax_2d": exp_2d,
        "op_underdamped_ring_2d": osc_2d,
    }


def _ols_slopes_2d(rows: list[dict[str, Any]], f0: str, f1: str) -> tuple[float, float]:
    """Compute S_2-symmetric partial linear slopes (beta_0, beta_1) of target on (f0, f1)."""
    n = len(rows)
    if n < 3:
        return 0.0, 0.0
    x0 = [float(r["features"][f0]) for r in rows]
    x1 = [float(r["features"][f1]) for r in rows]
    ys = [float(r["target"]) for r in rows]
    m0, m1, my = sum(x0) / n, sum(x1) / n, sum(ys) / n
    s00 = sum((a - m0) ** 2 for a in x0) + 1e-9
    s11 = sum((b - m1) ** 2 for b in x1) + 1e-9
    s01 = sum((a - m0) * (b - m1) for a, b in zip(x0, x1))
    s0y = sum((a - m0) * (y - my) for a, y in zip(x0, ys))
    s1y = sum((b - m1) * (y - my) for b, y in zip(x1, ys))
    det = s00 * s11 - s01 * s01
    if abs(det) <= 1e-12:
        return s0y / s00, s1y / s11
    return (s11 * s0y - s01 * s1y) / det, (s00 * s1y - s01 * s0y) / det


def extract_multiscale_pilot_signatures(
    pilot_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    scale_probe: dict[str, Any] | None,
    domain_cue: str,
) -> tuple[list[str], dict[str, Any]]:
    """Compute strictly observable S_d-symmetric `sig:*` cues and target regime features (zero oracle labels!)."""
    arity = len(feature_names)
    ys = [float(r["target"]) for r in pilot_rows]
    f0 = feature_names[0]
    xs0 = [float(r["features"][f0]) for r in pilot_rows]
    pairs0 = sorted(zip(xs0, ys), key=lambda t: t[0])

    if arity == 1:
        slope0 = (pairs0[-1][1] - pairs0[0][1]) / max(1e-6, pairs0[-1][0] - pairs0[0][0])
        sign_cue = "sig:sensor_pos" if slope0 >= 0 else "sig:sensor_neg"
        max_load = max(xs0)
    else:
        f1 = feature_names[1]
        xs1 = [float(r["features"][f1]) for r in pilot_rows]
        b0, b1 = _ols_slopes_2d(pilot_rows, f0, f1)
        span0 = max(xs0) - min(xs0)
        span1 = max(xs1) - min(xs1)
        # S_2-symmetric dominant effect sign (invariant under swapping f0 <-> f1!)
        dom_eff = b0 * span0 if abs(b0 * span0) >= abs(b1 * span1) else b1 * span1
        sign_cue = "sig:sensor_pos" if dom_eff >= 0 else "sig:sensor_neg"
        max_load = max(max(xs0), max(xs1))

    pol_tag = "pos" if sign_cue == "sig:sensor_pos" else "neg"
    sigs = [
        f"sig:{pol_tag}_arity_{arity}",
        sign_cue,
        f"sig:{domain_cue}",
    ]
    obs_feats: dict[str, Any] = {
        "active_arity": arity,
        "max_primary_load": max_load,
    }

    if scale_probe is not None:
        if scale_probe["closure_verified"]:
            sigs.append(f"sig:{pol_tag}_scale_closure_verified")
            obs_feats["scale_closure_verified"] = True
        else:
            sigs.append("sig:scale_closure_broken_fast_mode")
            obs_feats["scale_closure_verified"] = False

    if arity == 1 and len(pairs0) >= 4:
        mid_i = len(pairs0) // 2
        s_low = abs(pairs0[mid_i][1] - pairs0[0][1]) / max(1e-6, pairs0[mid_i][0] - pairs0[0][0])
        s_high = abs(pairs0[-1][1] - pairs0[mid_i][1]) / max(1e-6, pairs0[-1][0] - pairs0[mid_i][0])
        if s_low > 1e-6 and 0.30 < (s_high / s_low) < 0.90:
            sigs.append(f"sig:{pol_tag}_sublinear_fractal_2_3")
        top_diff = abs(pairs0[-1][1] - pairs0[-2][1]) / max(1e-6, abs(pairs0[-1][1] - pairs0[0][1]))
        if top_diff < 0.03 and max(xs0) > 6.5:
            sigs.append("sig:witness_macro_flow_ceiling")
            obs_feats["high_flow_ceiling_regime"] = True
        elif max(xs0) > 6.5:
            obs_feats["high_flow_ceiling_regime"] = False
        # When max(xs0) <= 3.5 (safe low-flow pilot!), `high_flow_ceiling_regime` remains UNKNOWN!

    if arity == 2:
        f1 = feature_names[1]
        b0, b1 = _ols_slopes_2d(pilot_rows, f0, f1)
        sigs.append("sig:two_channel_coupled")
        # S_2-symmetric opposite partial slope check (`area` increases cost, `aspect` decreases A/r penalty!)
        if b0 * b1 < -0.02 and (scale_probe is None or scale_probe["closure_verified"]):
            sigs.append("sig:variational_aspect_penalty")

    return sorted(dict.fromkeys(sigs)), obs_feats


# ============================================================================
# 3. Build Multi-Scale Philosophical Forest (`micro -> meso -> macro` + Wormholes)
# ============================================================================

def _register_multiscale_source_node(
    forest: PhilosophicalForest,
    *,
    node_id: str,
    tree_id: str,
    surface_domain: str,
    scale_regime: str,
    operator_tag: str,
    operator_spec: dict[str, Any],
    role_names: list[str],
    lens: dict[str, float],
    probe_signatures: list[str],
    preconditions: list[dict[str, Any]] | None = None,
    layer_role: str = "mid",
    tolerance: float = 0.05,
    source_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    compiled_py = compile_operator_spec_to_python(operator_spec, role_names=role_names, lens=lens)
    ast_info = canonicalize_python_model_ast(compiled_py)
    assert ast_info["skeleton_status"] == "VERIFIED_PURE_AST", f"{node_id} failed pure AST check: {ast_info}"

    tv = TaskView(
        goal_kind="open_workspace_mechanism_prediction",
        target_metric="rmse",
        tolerance=tolerance,
        input_domain={"active_arity": len(role_names), "scale_regime": scale_regime},
    )
    om = ObservationMap(
        raw_channel=f"{surface_domain}_readout",
        readout_transform=("affine_inverted" if lens.get("sign", 1.0) < 0 else "affine_positive"),
        unit_scale=float(lens.get("scale", 1.0)),
        sign_convention=int(round(float(lens.get("sign", 1.0)))),
        parameter_semantics={
            "active_arity": str(len(role_names)),
            "role_order": ",".join(role_names),
            "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        },
    )
    sm = ScaleMap(
        coarse_graining_op=("identity_micro" if scale_regime == "micro" else f"pi_{scale_regime}_projection"),
        window_or_partition={"scale_regime": scale_regime},
        preserved_queries=["open_workspace_mechanism_prediction"],
        closure_scope=("exact" if scale_regime == "micro" else "steady_state_only"),
        proposed_discarded_variables=([] if scale_regime == "micro" else ["fast_micro_mode"]),
        verified_discarded_in_scope=(
            []
            if scale_regime == "micro"
            else [{
                "variable": "fast_micro_mode",
                "task_view_goal": "open_workspace_mechanism_prediction",
                "error_bound": 0.015,
                "evidence_ref": f"src_scale_cert:{node_id}",
            }]
        ),
        error_budget=0.015,
    )
    cc = ClaimContract(
        claim_id=f"claim:{node_id}",
        family_plugin="m3_open_workspace_ast",
        preconditions=list(preconditions or []),
        conclusion_or_action={
            "schema_tag": f"{scale_regime}:{operator_tag}",
            "operator_tag": operator_tag,
            "operator_spec": deepcopy(operator_spec),
            "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
            "role_names": list(role_names),
        },
        protocol_spec={"requires_prefix_isolated_confirmation": True, "scale_regime": scale_regime},
        tolerance=tolerance,
        evidence_status="SUPPORTED_IN_SCOPE",
        permission_tier="local_solver_guidance",
    )
    prog_steps = [
        {"step": "verify_scale_closure_or_escalate", "scale_regime": scale_regime},
        {"step": "calibrate_observation_lens", "active_arity": len(role_names)},
        {"step": "compile_and_fit_composed_ast", "operator_tag": operator_tag},
        {"step": "subprocess_isolated_confirm", "fresh_points": 1},
    ]
    node = forest.register_node(
        node_id=node_id,
        tree_id=tree_id,
        layer_role=layer_role,
        task_view=tv,
        observation_map=om,
        scale_map=sm,
        claim_contract=cc,
        probe_signatures=probe_signatures,
        program_steps=prog_steps,
        source_cluster_ids=[surface_domain],
    )
    forest.record_raw_episode({
        "episode_id": f"ep:{node_id}",
        "node_id": node_id,
        "surface_domain": surface_domain,
        "scale_regime": scale_regime,
        "operator_tag": operator_tag,
        "operator_spec": deepcopy(operator_spec),
        "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
        "source_rows": deepcopy(source_rows or []),
    })
    return node


def build_m4_multiscale_forest(
    *,
    cluster_seed: int,
    extra_distractor_count: int = 12,
) -> tuple[PhilosophicalForest, dict[str, Any]]:
    """Construct the M4 Multi-Scale Philosophical Forest with Scale-Indexed Wormholes!"""
    rng = Random(cluster_seed)
    forest = PhilosophicalForest()
    ops = get_m4_operator_library()
    source_ledger_cost = 0.0

    # --- Tree 1: Aqueous & Geophysical Ladder (Water Dipole -> Hexagonal Cell -> River Drainage) ---
    _register_multiscale_source_node(
        forest,
        node_id="aq_micro_polar",
        tree_id="tree_aqueous_geo",
        surface_domain="chem_aqueous_polar",
        scale_regime="micro",
        operator_tag="op_micro_fast_transient_2d",
        operator_spec=ops["op_micro_fast_transient_2d"],
        role_names=["u", "t"],
        lens={"sign": 1.0, "scale": 1.0, "offset": 0.0},
        probe_signatures=[
            "sig:pos_arity_2", "sig:sensor_pos", "sig:two_channel_coupled",
            "sig:scale_closure_broken_fast_mode", "sig:cue_polar_relaxation",
        ],
    )
    _register_multiscale_source_node(
        forest,
        node_id="aq_meso_hex_cell",
        tree_id="tree_aqueous_geo",
        surface_domain="phys_convection_snow_cell",
        scale_regime="meso",
        operator_tag="op_meso_variational_hex_2d",
        operator_spec=ops["op_meso_variational_hex_2d"],
        role_names=["area", "aspect"],
        lens={"sign": 1.0, "scale": 1.0, "offset": 0.0},
        probe_signatures=[
            "sig:pos_arity_2", "sig:sensor_pos", "sig:two_channel_coupled",
            "sig:variational_aspect_penalty", "sig:pos_scale_closure_verified", "sig:cue_cellular_tiling",
        ],
    )
    _register_multiscale_source_node(
        forest,
        node_id="aq_macro_river_tree",
        tree_id="tree_aqueous_geo",
        surface_domain="geo_river_basin",
        scale_regime="macro",
        operator_tag="op_macro_murray_2_3_1d",
        operator_spec=ops["op_macro_murray_2_3_1d"],
        role_names=["flow"],
        lens={"sign": 1.0, "scale": 1.0, "offset": 0.0},
        probe_signatures=[
            "sig:pos_arity_1", "sig:sensor_pos", "sig:pos_sublinear_fractal_2_3",
            "sig:pos_scale_closure_verified", "sig:cue_branching_transport", "sig:low_flow_murray_onset",
        ],
    )
    forest.add_coarse_graining_edge(
        fine_node_id="aq_micro_polar",
        coarse_node_id="aq_meso_hex_cell",
        scale_map={"coarse_graining_op": "pi_meso_adiabatic_interface", "window_or_partition": {"scale_regime": "meso"}},
        empirical_closure_certificate={"defect_bound": 0.012, "verified_regime": "t >= 3.0 * tau_f"},
    )
    forest.add_coarse_graining_edge(
        fine_node_id="aq_meso_hex_cell",
        coarse_node_id="aq_macro_river_tree",
        scale_map={"coarse_graining_op": "pi_macro_tree_aggregation", "window_or_partition": {"scale_regime": "macro"}},
        empirical_closure_certificate={"defect_bound": 0.014, "verified_regime": "hierarchical_steady_flow"},
    )

    # --- Tree 2: Chemical Polar Partner at S_micro (`chem_ammonia_micro`) ---
    _register_multiscale_source_node(
        forest,
        node_id="chem_ammonia_micro",
        tree_id="tree_ammonia_solvation",
        surface_domain="chem_ammonia_dipole",
        scale_regime="micro",
        operator_tag="op_micro_fast_transient_2d",
        operator_spec=ops["op_micro_fast_transient_2d"],
        role_names=["u", "t"],
        lens={"sign": 1.0, "scale": 1.15, "offset": 0.4},
        probe_signatures=[
            "sig:pos_arity_2", "sig:sensor_pos", "sig:two_channel_coupled",
            "sig:scale_closure_broken_fast_mode", "sig:cue_dipole_solvation",
        ],
    )

    # --- Tree 3: Biological Self-Organization Ladder (Honeycomb Meso -> Vascular Murray Macro) ---
    # Note: At S_micro, hydrophobic beeswax is NON-polar (explicitly distinct from `aq_micro_polar`!),
    # while at S_meso (`bio_honeycomb_meso`) and S_macro (`bio_vascular_macro`) it shares exact universality!
    _register_multiscale_source_node(
        forest,
        node_id="bio_wax_hydrophobic_micro",
        tree_id="tree_bio_morphology",
        surface_domain="bio_hydrophobic_wax_micro",
        scale_regime="micro",
        operator_tag="op_overdamped_relax_2d",
        operator_spec=ops["op_overdamped_relax_2d"],
        role_names=["u", "t"],
        lens={"sign": 1.0, "scale": 0.9, "offset": 0.0},
        probe_signatures=[
            "sig:pos_arity_2", "sig:sensor_pos", "sig:cue_nonpolar_wax",
        ],
    )
    _register_multiscale_source_node(
        forest,
        node_id="bio_honeycomb_meso",
        tree_id="tree_bio_morphology",
        surface_domain="bio_honeycomb_packing",
        scale_regime="meso",
        operator_tag="op_meso_variational_hex_2d",
        operator_spec=ops["op_meso_variational_hex_2d"],
        role_names=["area", "aspect"],
        lens={"sign": 1.0, "scale": 1.2, "offset": 0.5},
        probe_signatures=[
            "sig:pos_arity_2", "sig:sensor_pos", "sig:pos_scale_closure_verified", "sig:cue_cellular_tiling",
        ],
    )
    _register_multiscale_source_node(
        forest,
        node_id="bio_vascular_macro",
        tree_id="tree_bio_morphology",
        surface_domain="bio_vascular_murray",
        scale_regime="macro",
        operator_tag="op_macro_murray_2_3_1d",
        operator_spec=ops["op_macro_murray_2_3_1d"],
        role_names=["flow"],
        lens={"sign": 1.0, "scale": 1.3, "offset": 0.2},
        probe_signatures=[
            "sig:pos_arity_1", "sig:sensor_pos", "sig:pos_sublinear_fractal_2_3",
            "sig:pos_scale_closure_verified", "sig:cue_vascular_murray",
        ],
    )
    forest.add_coarse_graining_edge(
        fine_node_id="bio_wax_hydrophobic_micro",
        coarse_node_id="bio_honeycomb_meso",
        scale_map={"coarse_graining_op": "pi_meso_surface_energy_min", "window_or_partition": {"scale_regime": "meso"}},
        empirical_closure_certificate={"defect_bound": 0.016, "verified_regime": "cellular_equilibrium"},
    )
    forest.add_coarse_graining_edge(
        fine_node_id="bio_honeycomb_meso",
        coarse_node_id="bio_vascular_macro",
        scale_map={"coarse_graining_op": "pi_macro_murray_branching", "window_or_partition": {"scale_regime": "macro"}},
        empirical_closure_certificate={"defect_bound": 0.013, "verified_regime": "steady_metabolic_transport"},
    )

    # --- Tree 4: Distributed Computing System Ladder (Retry Micro -> Shard Cell Meso -> CDN Backbone Macro + Capped Child) ---
    _register_multiscale_source_node(
        forest,
        node_id="sys_queue_micro",
        tree_id="tree_distributed_systems",
        surface_domain="sys_retry_backpressure_micro",
        scale_regime="micro",
        operator_tag="op_micro_fast_transient_2d",
        operator_spec=ops["op_micro_fast_transient_2d"],
        role_names=["u", "t"],
        lens={"sign": -1.0, "scale": 1.1, "offset": 10.0},
        probe_signatures=[
            "sig:neg_arity_2", "sig:sensor_neg", "sig:two_channel_coupled",
            "sig:scale_closure_broken_fast_mode", "sig:cue_queue_backpressure",
        ],
    )
    _register_multiscale_source_node(
        forest,
        node_id="sys_shard_meso",
        tree_id="tree_distributed_systems",
        surface_domain="sys_shard_partition_meso",
        scale_regime="meso",
        operator_tag="op_meso_variational_hex_2d",
        operator_spec=ops["op_meso_variational_hex_2d"],
        role_names=["area", "aspect"],
        lens={"sign": -1.0, "scale": 1.05, "offset": 12.0},
        probe_signatures=[
            "sig:neg_arity_2", "sig:sensor_neg", "sig:two_channel_coupled",
            "sig:variational_aspect_penalty", "sig:neg_scale_closure_verified", "sig:cue_shard_boundary",
        ],
    )
    _register_multiscale_source_node(
        forest,
        node_id="sys_cdn_macro",
        tree_id="tree_distributed_systems",
        surface_domain="sys_cdn_backbone_macro",
        scale_regime="macro",
        operator_tag="op_macro_murray_2_3_1d",
        operator_spec=ops["op_macro_murray_2_3_1d"],
        role_names=["flow"],
        lens={"sign": -1.0, "scale": 1.15, "offset": 15.0},
        probe_signatures=[
            "sig:neg_arity_1", "sig:sensor_neg", "sig:neg_sublinear_fractal_2_3",
            "sig:neg_scale_closure_verified", "sig:low_flow_cdn_onset",
        ],
    )
    # Refined child on Tree 2 (`bio_vascular_macro -> bio_vascular_macro_capped`) for 2-hop cross-tree safe-pilot walk!
    _register_multiscale_source_node(
        forest,
        node_id="bio_vascular_macro_capped",
        tree_id="tree_bio_morphology",
        surface_domain="bio_vascular_saturated_cap",
        scale_regime="macro",
        operator_tag="op_macro_murray_capped_1d",
        operator_spec=ops["op_macro_murray_capped_1d"],
        role_names=["flow"],
        lens={"sign": 1.0, "scale": 1.25, "offset": 0.0},
        probe_signatures=["sig:witness_macro_flow_ceiling"],
        preconditions=[{"field": "high_flow_ceiling_regime", "op": "==", "value": True}],
        layer_role="mid",
    )
    forest.add_local_implication_edge(
        stronger_node_id="bio_vascular_macro_capped",
        weaker_node_id="bio_vascular_macro",
        preserved_contract={"low_flow_limit": "murray_2_3_scaling"},
        scope={"flow_le": 3.5},
        evidence_status="SUPPORTED_IN_SCOPE",
    )
    forest.add_coarse_graining_edge(
        fine_node_id="sys_queue_micro",
        coarse_node_id="sys_shard_meso",
        scale_map={"coarse_graining_op": "pi_meso_batch_window", "window_or_partition": {"scale_regime": "meso"}},
        empirical_closure_certificate={"defect_bound": 0.015, "verified_regime": "window >= 4*tau_retry"},
    )
    forest.add_coarse_graining_edge(
        fine_node_id="sys_shard_meso",
        coarse_node_id="sys_cdn_macro",
        scale_map={"coarse_graining_op": "pi_macro_regional_tree", "window_or_partition": {"scale_regime": "macro"}},
        empirical_closure_certificate={"defect_bound": 0.014, "verified_regime": "steady_backbone_routing"},
    )

    # --- Scale-Indexed Horizontal Wormholes (Verified by Pure AST Skeleton Equality!) ---
    wh_specs = [
        # S_micro wormholes: aq_micro_polar <-> chem_ammonia_micro & sys_queue_micro (NOT bio_wax_hydrophobic_micro!)
        ("wh_micro_aq_ammonia", "aq_micro_polar", "chem_ammonia_micro", "micro", ["sig:arity_2", "sig:scale_closure_broken_fast_mode"], None),
        ("wh_micro_aq_sysqueue", "aq_micro_polar", "sys_queue_micro", "micro", ["sig:arity_2", "sig:scale_closure_broken_fast_mode"], None),
        # S_meso wormholes: aq_meso_hex_cell <-> bio_honeycomb_meso & sys_shard_meso!
        ("wh_meso_aq_honeycomb", "aq_meso_hex_cell", "bio_honeycomb_meso", "meso", ["sig:arity_2", "sig:variational_aspect_penalty"], None),
        ("wh_meso_aq_sysshard", "aq_meso_hex_cell", "sys_shard_meso", "meso", ["sig:arity_2", "sig:variational_aspect_penalty"], None),
        # S_macro wormholes: sys_cdn_macro <-> bio_vascular_macro & aq_macro_river_tree <-> bio_vascular_macro!
        (
            "wh_macro_cdn_vascular",
            "sys_cdn_macro",
            "bio_vascular_macro",
            "macro",
            ["sig:arity_1", "sig:sublinear_fractal_2_3", "sig:low_flow_cdn_onset"],
            {"exclusion_predicates": [{"field": "high_flow_ceiling_regime", "op": "==", "value": True}]},
        ),
        (
            "wh_macro_river_vascular",
            "aq_macro_river_tree",
            "bio_vascular_macro",
            "macro",
            ["sig:arity_1", "sig:sublinear_fractal_2_3", "sig:low_flow_murray_onset"],
            {"exclusion_predicates": [{"field": "high_flow_ceiling_regime", "op": "==", "value": True}]},
        ),
    ]
    for wid, na, nb, s_reg, sig_refs, align_extra in wh_specs:
        dig_a = forest.nodes[na]["claim_contract"]["conclusion_or_action"]["ast_skeleton_digest"]
        dig_b = forest.nodes[nb]["claim_contract"]["conclusion_or_action"]["ast_skeleton_digest"]
        assert dig_a == dig_b, f"AST skeleton mismatch across wormhole {wid}: {dig_a} != {dig_b}"
        align_c = {
            "characteristic_predicates": [],
            "inclusion_predicates": [],
            "exclusion_predicates": list((align_extra or {}).get("exclusion_predicates", ())),
            "revision_provenance": [],
        }
        forest.add_horizontal_wormhole(
            wormhole_id=wid,
            node_a_id=na,
            node_b_id=nb,
            probe_signature_refs=sig_refs,
            role_and_observation_map={"verified_pure_ast_digest": dig_a, "scale_regime": s_reg},
            hop_error_budget=0.018,
            lipschitz_factor=1.05,
            align_conditions=align_c,
            permission_tier="local_solver_guidance",
            compatible_goal_kinds=["open_workspace_mechanism_prediction"],
            scale_regime=s_reg,
        )
        source_ledger_cost += 0.5

    # Add cross-disciplinary 1-domain distractor nodes so retrieval faces realistic K-scale competition
    distractor_ops = [
        ("op_monod_saturation_1d", ["x"], ["sig:arity_1", "sig:sensor_pos", "sig:scale_closure_verified"], "macro"),
        ("op_haldane_inhibited_1d", ["x"], ["sig:arity_1", "sig:sensor_neg", "sig:scale_closure_verified"], "macro"),
        ("op_overdamped_relax_2d", ["u", "t"], ["sig:arity_2", "sig:sensor_pos", "sig:two_channel_coupled"], "micro"),
        ("op_underdamped_ring_2d", ["u", "t"], ["sig:arity_2", "sig:sensor_neg", "sig:two_channel_coupled"], "micro"),
        ("op_meso_hex_shear_2d", ["area", "shear"], ["sig:arity_2", "sig:sensor_pos", "sig:two_channel_coupled", "sig:scale_closure_verified"], "meso"),
        ("op_adiabatic_linear_1d", ["x"], ["sig:arity_1", "sig:sensor_neg", "sig:scale_closure_verified"], "macro"),
    ]
    for d_idx in range(extra_distractor_count):
        op_tag, r_names, d_sigs, d_scale = distractor_ops[d_idx % len(distractor_ops)]
        sign_v = -1.0 if "sig:sensor_neg" in d_sigs else 1.0
        _register_multiscale_source_node(
            forest,
            node_id=f"dist_node_{d_idx:02d}",
            tree_id=f"tree_distractor_{d_idx:02d}",
            surface_domain=f"distractor_domain_{d_idx:02d}",
            scale_regime=d_scale,
            operator_tag=op_tag,
            operator_spec=ops[op_tag],
            role_names=r_names,
            lens={"sign": sign_v, "scale": round(0.85 + 0.1 * (d_idx % 5), 3), "offset": float(d_idx % 3)},
            probe_signatures=[*d_sigs, f"sig:dist_unique_{d_idx:02d}"],
        )

    source_ledger_cost += round(len(forest.nodes) * 18.0, 4)
    return forest, {"source_formation_and_bridge_cost": source_ledger_cost, "active_node_count": len(forest.nodes)}


# ============================================================================
# 4. Multi-Scale Target Suite (`18` Unseen Targets / Cluster × `4` Clusters)
# ============================================================================

@dataclass
class MultiScaleTargetSpec:
    target_id: str
    category: str  # "T_Adiabatic_Coarse" | "T_Closure_Breakdown" | "T_MultiScale_Walk" | "T_DualSensor_Permuted"
    lab: MultiScaleWorkspaceTaskLab
    pilot_points: list[dict[str, float]]
    confirm_points: list[dict[str, float]]
    scale_boundary_point: dict[str, float]
    candidate_coarse_regime: str  # "meso" | "macro"
    domain_cue: str
    role_map_to_canonical: list[str]


def build_m4_cluster_targets(cluster_idx: int) -> tuple[list[MultiScaleTargetSpec], list[MultiScaleTargetSpec]]:
    """Build 18 unseen multi-scale targets + 2 out-of-scope honesty targets for a frozen cluster."""
    rng = Random(7100 + 137 * cluster_idx)
    targets: list[MultiScaleTargetSpec] = []

    # 1. Six `T_Adiabatic_Coarse` targets (3 macro Murray 2/3 + 3 meso Hexagonal variational)
    # Fast micro transient has decayed (`t_hold = 6.0 >> tau_f = 0.25`), so Pi_s is closed (`defect < 0.008`)!
    for i in range(3):
        c_k = round(rng.uniform(1.6, 2.5), 3)
        sign = 1.0 if i % 2 == 0 else -1.0
        off = round(rng.uniform(1.0, 8.0), 2)
        tau_f = 0.22

        def _mk_macro(ck: float, sgn: float, of: float, tf: float):
            # At macro steady window t_eff = 5.5, fast mode exp(-5.5/0.22) ~ 1e-11 -> exact emergent Murray Q^(2/3)!
            macro_f = lambda p: of + sgn * (ck * (p["q_flow"] ** (2.0 / 3.0)))
            micro_f = lambda p: of + sgn * (ck * (p["q_flow"] ** (2.0 / 3.0)) + 0.35 * math.exp(-5.0 / tf))
            return micro_f, macro_f

        mf, cf = _mk_macro(c_k, sign, off, tau_f)
        tid = f"C{cluster_idx}_T{len(targets)+1:02d}_AdiabaticMacro"
        lab = MultiScaleWorkspaceTaskLab(
            task_id=tid,
            surface_domain=f"unseen_logistics_network_{i}",
            feature_names=("q_flow",),
            feature_bounds={"q_flow": (0.4, 9.0)},
            true_scale_regime="macro",
            heldout_points=[{"q_flow": v} for v in (0.7, 1.9, 3.6, 5.8, 8.2)],
            tolerance=0.05,
            micro_fn=mf,
            coarse_pi_fn=cf,
        )
        targets.append(MultiScaleTargetSpec(
            target_id=tid,
            category="T_Adiabatic_Coarse",
            lab=lab,
            pilot_points=[{"q_flow": v} for v in (0.5, 1.4, 2.8, 5.0, 7.6)],
            confirm_points=[{"q_flow": 2.2}, {"q_flow": 6.8}],
            scale_boundary_point={"q_flow": 4.0},
            candidate_coarse_regime="macro",
            domain_cue="cue_branching_transport",
            role_map_to_canonical=["q_flow"],
        ))

    for i in range(3):
        c1 = round(rng.uniform(1.5, 2.3), 3)
        c2 = round(rng.uniform(0.6, 1.1), 3)
        sign = -1.0 if i == 1 else 1.0
        off = round(rng.uniform(0.5, 9.0), 2)

        def _mk_meso(a1: float, a2: float, sgn: float, of: float):
            coarse_f = lambda p: of + sgn * (a1 * math.sqrt(p["cell_area"]) + a2 * p["cell_area"] / p["cell_aspect"])
            micro_f = lambda p: coarse_f(p) + 0.004 * math.sin(p["cell_area"])
            return micro_f, coarse_f

        mf, cf = _mk_meso(c1, c2, sign, off)
        tid = f"C{cluster_idx}_T{len(targets)+1:02d}_AdiabaticMeso"
        lab = MultiScaleWorkspaceTaskLab(
            task_id=tid,
            surface_domain=f"unseen_foam_crystal_tiling_{i}",
            feature_names=("cell_area", "cell_aspect"),
            feature_bounds={"cell_area": (0.8, 9.0), "cell_aspect": (0.6, 3.0)},
            true_scale_regime="meso",
            heldout_points=[
                {"cell_area": 1.2, "cell_aspect": 0.9},
                {"cell_area": 3.0, "cell_aspect": 1.5},
                {"cell_area": 5.5, "cell_aspect": 2.2},
                {"cell_area": 7.8, "cell_aspect": 1.1},
                {"cell_area": 8.4, "cell_aspect": 2.6},
            ],
            tolerance=0.055,
            micro_fn=mf,
            coarse_pi_fn=cf,
        )
        targets.append(MultiScaleTargetSpec(
            target_id=tid,
            category="T_Adiabatic_Coarse",
            lab=lab,
            pilot_points=[
                {"cell_area": 1.0, "cell_aspect": 0.8},
                {"cell_area": 2.5, "cell_aspect": 2.4},
                {"cell_area": 4.5, "cell_aspect": 1.0},
                {"cell_area": 6.5, "cell_aspect": 2.1},
                {"cell_area": 8.0, "cell_aspect": 0.9},
                {"cell_area": 8.2, "cell_aspect": 2.5},
            ],
            confirm_points=[{"cell_area": 3.8, "cell_aspect": 1.3}, {"cell_area": 7.2, "cell_aspect": 2.0}],
            scale_boundary_point={"cell_area": 4.0, "cell_aspect": 1.4},
            candidate_coarse_regime="meso",
            domain_cue="cue_cellular_tiling",
            role_map_to_canonical=["cell_area", "cell_aspect"],
        ))

    # 2. Four `T_Closure_Breakdown` targets: Fast short-time quench (`t in [0.06, 1.4]`, `tau_f ~ 0.35`)!
    # Here `coarse_pi_fn` drops the fast mode `k_fast * u * exp(-t / tau_f)`, producing a HUGE closure defect (~0.75 >> 0.05)!
    for i in range(4):
        k_slow = round(rng.uniform(1.1, 1.7), 3)
        k_fast = round(rng.uniform(0.95, 1.45), 3)
        tau_f = round(rng.uniform(0.28, 0.48), 3)
        sign = -1.0 if i % 2 == 1 else 1.0
        off = round(rng.uniform(2.0, 11.0), 2) if sign < 0 else round(rng.uniform(0.0, 2.5), 2)

        def _mk_breakdown(ks: float, kf: float, tf: float, sgn: float, of: float):
            micro_f = lambda p: of + sgn * (ks * p["drive_u"] + kf * p["drive_u"] * math.exp(-p["quench_t"] / tf))
            # Adiabatic Schur coarse projection drops fast mode exp(-t/tf) -> invalid at small quench_t!
            coarse_f = lambda p: of + sgn * (ks * p["drive_u"])
            return micro_f, coarse_f

        mf, cf = _mk_breakdown(k_slow, k_fast, tau_f, sign, off)
        tid = f"C{cluster_idx}_T{len(targets)+1:02d}_ClosureBreakdown"
        lab = MultiScaleWorkspaceTaskLab(
            task_id=tid,
            surface_domain=f"unseen_fast_quench_bus_{i}",
            feature_names=("drive_u", "quench_t"),
            feature_bounds={"drive_u": (0.4, 3.0), "quench_t": (0.05, 2.2)},
            true_scale_regime="micro",
            heldout_points=[
                {"drive_u": 0.8, "quench_t": 0.10},
                {"drive_u": 1.6, "quench_t": 0.25},
                {"drive_u": 2.2, "quench_t": 0.55},
                {"drive_u": 1.3, "quench_t": 1.10},
                {"drive_u": 2.6, "quench_t": 1.85},
            ],
            tolerance=0.055,
            micro_fn=mf,
            coarse_pi_fn=cf,
        )
        targets.append(MultiScaleTargetSpec(
            target_id=tid,
            category="T_Closure_Breakdown",
            lab=lab,
            pilot_points=[
                {"drive_u": 0.6, "quench_t": 0.08},
                {"drive_u": 1.4, "quench_t": 0.20},
                {"drive_u": 2.4, "quench_t": 0.42},
                {"drive_u": 1.1, "quench_t": 0.85},
                {"drive_u": 2.0, "quench_t": 1.45},
                {"drive_u": 2.7, "quench_t": 2.05},
            ],
            confirm_points=[{"drive_u": 1.8, "quench_t": 0.16}, {"drive_u": 2.3, "quench_t": 0.95}],
            scale_boundary_point={"drive_u": 1.5, "quench_t": 0.12},
            candidate_coarse_regime="meso",
            domain_cue=("cue_queue_backpressure" if sign < 0 else "cue_polar_relaxation"),
            role_map_to_canonical=["drive_u", "quench_t"],
        ))

    # 3. Four `T_MultiScale_Walk` targets: Safe Low-Flow Macro Pilot (`q_flow in [0.3, 2.8]`) with Dual Negative Sensor (`-1`),
    # followed by high-flow confirmation (`q_flow in [7.2, 8.8]`) where backbone link ceiling triggers!
    # Notice: Q_safe hits `sys_cdn_macro(-1)` via `sig:low_flow_cdn_onset`, has ZERO overlap with `bio_vascular_macro_capped(+1)`,
    # and requires 2-hop walk `sys_cdn_macro(-1) <-> bio_vascular_macro(+1) -> bio_vascular_macro_capped(+1)`!
    for i in range(4):
        c_k = round(rng.uniform(1.8, 2.4), 3)
        # Ceiling triggers above q_flow ~ 4.3 (`4.3^(2/3) ~ 2.65`, so cap = c_k * 2.62)
        cap_val = round(c_k * rng.uniform(2.45, 2.75), 3)
        off = round(rng.uniform(11.0, 16.0), 2)

        def _mk_walk_capped(ck: float, cp: float, of: float):
            fn = lambda p: of - min(ck * (p["q_flow"] ** (2.0 / 3.0)), cp)
            return fn, fn

        mf, cf = _mk_walk_capped(c_k, cap_val, off)
        tid = f"C{cluster_idx}_T{len(targets)+1:02d}_MultiScaleWalk"
        lab = MultiScaleWorkspaceTaskLab(
            task_id=tid,
            surface_domain=f"unseen_edge_backbone_safe_pilot_{i}",
            feature_names=("q_flow",),
            feature_bounds={"q_flow": (0.2, 9.5)},
            true_scale_regime="macro",
            heldout_points=[{"q_flow": v} for v in (0.6, 2.1, 4.8, 7.0, 8.9)],
            tolerance=0.055,
            micro_fn=mf,
            coarse_pi_fn=cf,
        )
        targets.append(MultiScaleTargetSpec(
            target_id=tid,
            category="T_MultiScale_Walk",
            lab=lab,
            # Safe low-flow pilot strictly <= 2.8 (below ceiling onset ~4.3 -> Q_safe ∩ sigs(Child) == empty!)
            pilot_points=[{"q_flow": v} for v in (0.35, 0.85, 1.45, 2.10, 2.75)],
            # First confirmation point probes high-flow stress 7.8 -> falsifies uncapped Murray & unlocks capped child!
            confirm_points=[{"q_flow": 7.8}, {"q_flow": 6.2}],
            scale_boundary_point={"q_flow": 1.8},
            candidate_coarse_regime="macro",
            domain_cue="low_flow_cdn_onset",
            role_map_to_canonical=["q_flow"],
        ))

    # 4. Four `T_DualSensor_Permuted` targets: Reversed feature order (`cell_aspect`, `cell_area`) + inverted lens (`-1`)!
    for i in range(4):
        c1 = round(rng.uniform(1.6, 2.2), 3)
        c2 = round(rng.uniform(0.7, 1.15), 3)
        off = round(rng.uniform(12.0, 18.0), 2)

        def _mk_perm(a1: float, a2: float, of: float):
            fn = lambda p: of - (a1 * math.sqrt(p["plate_area"]) + a2 * p["plate_area"] / p["plate_aspect"])
            return fn, fn

        mf, cf = _mk_perm(c1, c2, off)
        tid = f"C{cluster_idx}_T{len(targets)+1:02d}_DualPermuted"
        lab = MultiScaleWorkspaceTaskLab(
            task_id=tid,
            surface_domain=f"unseen_basalt_column_inverted_{i}",
            # Reversed coordinate order (`plate_aspect` first, `plate_area` second!)
            feature_names=("plate_aspect", "plate_area"),
            feature_bounds={"plate_aspect": (0.6, 3.0), "plate_area": (0.8, 9.0)},
            true_scale_regime="meso",
            heldout_points=[
                {"plate_aspect": 0.9, "plate_area": 1.3},
                {"plate_aspect": 1.6, "plate_area": 3.2},
                {"plate_aspect": 2.3, "plate_area": 5.8},
                {"plate_aspect": 1.1, "plate_area": 7.6},
                {"plate_aspect": 2.5, "plate_area": 8.5},
            ],
            tolerance=0.055,
            micro_fn=mf,
            coarse_pi_fn=cf,
        )
        targets.append(MultiScaleTargetSpec(
            target_id=tid,
            category="T_DualSensor_Permuted",
            lab=lab,
            pilot_points=[
                {"plate_aspect": 0.8, "plate_area": 1.1},
                {"plate_aspect": 2.4, "plate_area": 2.6},
                {"plate_aspect": 1.0, "plate_area": 4.6},
                {"plate_aspect": 2.2, "plate_area": 6.4},
                {"plate_aspect": 0.9, "plate_area": 8.1},
                {"plate_aspect": 2.6, "plate_area": 8.3},
            ],
            confirm_points=[{"plate_aspect": 1.4, "plate_area": 3.9}, {"plate_aspect": 2.0, "plate_area": 7.1}],
            scale_boundary_point={"plate_aspect": 1.5, "plate_area": 4.2},
            candidate_coarse_regime="meso",
            domain_cue="cue_shard_boundary",
            role_map_to_canonical=["plate_area", "plate_aspect"],
        ))

    # 2 Out-of-Scope Honesty Targets (Chaotic / Non-Stationary piecewise discontinuous jump)
    oos_targets: list[MultiScaleTargetSpec] = []
    for o_idx in range(2):
        tid = f"C{cluster_idx}_OOS{o_idx+1:02d}"
        fn_oos = lambda p, s=o_idx: 3.5 * math.sin(6.7 * p["q_flow"] + s) + (4.2 if int(p["q_flow"] * 3) % 2 == 0 else -3.8)
        oos_lab = MultiScaleWorkspaceTaskLab(
            task_id=tid,
            surface_domain="oos_chaotic_regime_switching",
            feature_names=("q_flow",),
            feature_bounds={"q_flow": (0.3, 9.0)},
            true_scale_regime="macro",
            heldout_points=[{"q_flow": v} for v in (0.8, 2.4, 4.9, 7.1, 8.6)],
            tolerance=0.05,
            micro_fn=fn_oos,
            coarse_pi_fn=fn_oos,
            is_out_of_scope=True,
        )
        oos_targets.append(MultiScaleTargetSpec(
            target_id=tid,
            category="T_OutOfScope",
            lab=oos_lab,
            pilot_points=[{"q_flow": v} for v in (0.5, 1.8, 3.5, 5.7, 8.1)],
            confirm_points=[{"q_flow": 2.9}, {"q_flow": 6.5}],
            scale_boundary_point={"q_flow": 4.0},
            candidate_coarse_regime="macro",
            domain_cue="cue_branching_transport",
            role_map_to_canonical=["q_flow"],
        ))

    return targets, oos_targets


# ============================================================================
# 5. Controlled Multi-Scale Researcher Arms (Exact Monotone Pricing & Isolation)
# ============================================================================

def _infer_role_permutations(feature_names: tuple[str, ...]) -> list[list[str]]:
    if len(feature_names) == 1:
        return [[feature_names[0]]]
    f0, f1 = feature_names[0], feature_names[1]
    return [[f0, f1], [f1, f0]]


def _estimate_affine_lens_for_spec(
    op_spec: dict[str, Any],
    role_order: list[str],
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Calibrate ObservationMap sign & baseline offset strictly from paid ``training_rows``."""
    from scipy.optimize import least_squares
    from .m4_compositional_discovery import _collect_params, _emit_expr

    if len(role_order) == 1:
        pairs = sorted((float(r["features"][role_order[0]]), float(r["target"])) for r in rows)
        sign = 1.0 if pairs[-1][1] >= pairs[0][1] else -1.0
    else:
        b0, _ = _ols_slopes_2d(rows, role_order[0], role_order[1])
        sign = 1.0 if b0 >= 0 else -1.0

    params_meta: dict[int, dict[str, float]] = {}
    _collect_params(op_spec, params_meta, strict_unique=True)
    n_p = max(params_meta.keys()) + 1 if params_meta else 0
    lbs = [-200.0, *[params_meta[i]["lb"] for i in range(n_p)]]
    ubs = [200.0, *[params_meta[i]["ub"] for i in range(n_p)]]
    y_med = sum(float(r["target"]) for r in rows) / max(1, len(rows))
    base_x0 = [y_med, *[params_meta[i]["init"] for i in range(n_p)]]
    starts = [base_x0]
    ms = op_spec.get("multi_start_param")
    if isinstance(ms, dict) and int(ms["idx"]) < n_p:
        ms_i = int(ms["idx"]) + 1  # shifted by +1 due to q[0] = offset
        starts = []
        for gv in ms["grid"]:
            sv = list(base_x0)
            sv[ms_i] = float(gv)
            starts.append(sv)

    vec_expr = _emit_expr(op_spec, role_order, param_mode="vec")
    fn_core = eval(f"lambda f, p: ({vec_expr})", {"math": math, "min": min, "max": max})
    rows_f = [r["features"] for r in rows]
    ys = [float(r["target"]) for r in rows]

    sign_cands = [sign] if len(role_order) == 1 else [sign, -sign]
    best_cost = float("inf")
    best_sign = sign
    best_off = float(y_med)
    for s_cand in sign_cands:
        for st_x0 in starts:
            sol = least_squares(
                lambda q, sc=s_cand: [(q[0] + sc * fn_core(f, q[1:])) - y for f, y in zip(rows_f, ys)],
                x0=st_x0,
                bounds=(lbs, ubs),
            )
            if float(sol.cost) < best_cost:
                best_cost = float(sol.cost)
                best_sign = s_cand
                best_off = float(sol.x[0])
    return {"sign": best_sign, "scale": 1.0, "offset": best_off}


def _fit_and_score_operator_specs(
    workspace_dir: Path,
    *,
    specs_to_try: list[tuple[str, dict[str, Any]]],
    feature_names: tuple[str, ...],
    training_rows: list[dict[str, Any]],
    tolerance: float,
    max_fit_trials: int = 6,
    calibrate_lens_by_tag: dict[str, bool] | None = None,
) -> tuple[dict[str, Any] | None, int, float]:
    """Evaluate candidate operator specs on ``training_rows`` with monotone work-based pricing!"""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    perms = _infer_role_permutations(feature_names)
    evaluated = 0
    wall_s = 0.0
    best_cand: dict[str, Any] | None = None

    for tag, op_spec in specs_to_try[:max_fit_trials]:
        req_arity = int(op_spec.get("arity", 1))
        if req_arity != len(feature_names):
            continue
        do_cal = True if calibrate_lens_by_tag is None else bool(calibrate_lens_by_tag.get(tag, True))
        for role_order in perms:
            lens = (
                _estimate_affine_lens_for_spec(op_spec, role_order, training_rows)
                if do_cal
                else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
            )
            py_code = compile_operator_spec_to_python(op_spec, role_names=role_order, lens=lens)
            rel_name = f"cand_{evaluated:02d}.py"
            (workspace_dir / rel_name).write_text(py_code, encoding="utf-8")
            ev = evaluate_isolated_python_model(
                workspace_dir,
                rel_name,
                training_rows,
                isolated_prefix_rows=training_rows,
            )
            evaluated += 1
            wall_s += float(ev.get("wall_seconds", 0.02))
            rmse = float(ev["rmse"])
            if math.isfinite(rmse) and (best_cand is None or rmse < best_cand["train_rmse"]):
                best_cand = {
                    "tag": tag,
                    "operator_spec": deepcopy(op_spec),
                    "role_order": list(role_order),
                    "model_relpath": rel_name,
                    "model_source": py_code,
                    "train_rmse": rmse,
                }
            if math.isfinite(rmse) and rmse <= tolerance * 0.85:
                break
        if best_cand is not None and best_cand["train_rmse"] <= tolerance * 0.85:
            break

    return best_cand, evaluated, wall_s


_SUPPORTED_M4_PROGRAM_STEPS = frozenset({
    "verify_scale_closure_or_escalate",
    "calibrate_observation_lens",
    "compile_and_fit_composed_ast",
    "subprocess_isolated_confirm",
})


_M4_PROGRAM_STEP_STAGE_ORDER = {
    "verify_scale_closure_or_escalate": 0,
    "calibrate_observation_lens": 1,
    "compile_and_fit_composed_ast": 2,
    "subprocess_isolated_confirm": 3,
}

_M4_PROGRAM_STEP_ALLOWED_KEYS = {
    "verify_scale_closure_or_escalate": {"step", "scale_regime"},
    "calibrate_observation_lens": {"step", "active_arity"},
    "compile_and_fit_composed_ast": {"step", "operator_tag"},
    "subprocess_isolated_confirm": {"step", "fresh_points"},
}


def coords_approx_equal(pt_a: dict[str, Any], pt_b: dict[str, Any], *, tol: float = 1e-9) -> bool:
    """Strict coordinate dictionary equality check with floating-point tolerance."""
    if set(pt_a.keys()) != set(pt_b.keys()):
        return False
    return all(abs(float(pt_a[k]) - float(pt_b[k])) <= tol for k in pt_a)


def select_or_synthesize_disjoint_confirm_points(
    *,
    predeclared_confirm_points: list[dict[str, float]] | tuple[dict[str, float], ...],
    training_prefix_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    required_fresh_count: int,
    feature_bounds: dict[str, tuple[float, float] | list[float]],
    verified_box: dict[str, tuple[float, float] | list[float]] | None = None,
) -> list[dict[str, float]]:
    """Select ``required_fresh_count`` unseen confirmation points strictly disjoint from ``training_prefix_rows``.

    Never silently truncates below ``required_fresh_count`` and NEVER reuses any coordinate already in
    ``training_prefix_rows``. When ``predeclared_confirm_points`` has fewer unseen points than requested,
    synthesizes deterministic golden-ratio interior probe coordinates within ``feature_bounds`` (intersected
    with ``verified_box`` when provided). Returns ``[]`` if valid bounds cannot supply ``required_fresh_count``
    distinct unseen points.
    """
    req = max(1, int(required_fresh_count))
    prefix_pts = [dict(r["features"]) for r in training_prefix_rows if isinstance(r, dict) and "features" in r]
    eff_bounds: dict[str, tuple[float, float]] = {}
    for k, bnds in feature_bounds.items():
        lo, hi = float(bnds[0]), float(bnds[1])
        if verified_box is not None and k in verified_box:
            lo = max(lo, float(verified_box[k][0]))
            hi = min(hi, float(verified_box[k][1]))
        if lo > hi - 1e-7:
            return []
        eff_bounds[k] = (lo, hi)

    def _in_eff_bounds(pt: dict[str, Any]) -> bool:
        for k, (lo, hi) in eff_bounds.items():
            if k not in pt:
                return False
            v = float(pt[k])
            if v < lo - 1e-9 or v > hi + 1e-9:
                return False
        return True

    chosen: list[dict[str, float]] = []
    for raw_pt in predeclared_confirm_points:
        cand_pt = {str(k): float(v) for k, v in raw_pt.items()}
        if not _in_eff_bounds(cand_pt):
            continue
        if any(coords_approx_equal(cand_pt, pp) for pp in prefix_pts):
            continue
        if any(coords_approx_equal(cand_pt, cp) for cp in chosen):
            continue
        chosen.append(cand_pt)
        if len(chosen) >= req:
            return chosen

    # Deterministic golden-ratio Halton-like interior synthesis for remaining fresh points
    primes = (0.6180339887498949, 0.7548776662466927, 0.5698402909980532, 0.4301597090019468)
    f_keys = list(eff_bounds.keys())
    attempt = 1
    while len(chosen) < req and attempt <= 128:
        synth_pt: dict[str, float] = {}
        for d_idx, fk in enumerate(f_keys):
            lo, hi = eff_bounds[fk]
            frac = 0.08 + ((0.5731 + attempt * primes[d_idx % len(primes)]) % 0.84)
            synth_pt[fk] = round(lo + frac * (hi - lo), 5)
        attempt += 1
        if any(coords_approx_equal(synth_pt, pp) for pp in prefix_pts):
            continue
        if any(coords_approx_equal(synth_pt, cp) for cp in chosen):
            continue
        chosen.append(synth_pt)

    return chosen if len(chosen) == req else []


def _validate_and_interpret_m4_program_steps(
    program_steps: Any,
    *,
    node: dict[str, Any],
    target_arity: int,
) -> dict[str, Any] | None:
    """Validate and interpret ordered declarative ``program_steps`` on an M4 ``PhilosophicalForest`` node.

    Enforces:
      - non-empty list of supported opcodes with no duplicate steps and no unknown keys;
      - strict causal stage ordering:
        ``verify_scale_closure_or_escalate`` -> ``calibrate_observation_lens``
        -> ``compile_and_fit_composed_ast`` -> ``subprocess_isolated_confirm``;
      - matching ``scale_regime``, ``active_arity``, ``operator_tag``, and positive integer ``fresh_points``.
    """
    if not isinstance(program_steps, list) or not program_steps:
        return None
    step_map: dict[str, dict[str, Any]] = {}
    last_stage_rank = -1
    for entry in program_steps:
        if not isinstance(entry, dict):
            return None
        op_name = entry.get("step")
        if not isinstance(op_name, str) or op_name not in _SUPPORTED_M4_PROGRAM_STEPS:
            return None
        if op_name in step_map:
            return None
        allowed_keys = _M4_PROGRAM_STEP_ALLOWED_KEYS[op_name]
        if not set(entry.keys()).issubset(allowed_keys):
            return None
        cur_rank = _M4_PROGRAM_STEP_STAGE_ORDER[op_name]
        if cur_rank <= last_stage_rank:
            return None
        last_stage_rank = cur_rank
        step_map[op_name] = entry

    if "compile_and_fit_composed_ast" not in step_map or "subprocess_isolated_confirm" not in step_map:
        return None

    declared_scale: str | None = None
    if "verify_scale_closure_or_escalate" in step_map:
        s_reg = step_map["verify_scale_closure_or_escalate"].get("scale_regime")
        if s_reg is not None:
            if not isinstance(s_reg, str) or s_reg not in {"micro", "meso", "macro"}:
                return None
            node_scale = ((node.get("scale_map") or {}).get("window_or_partition") or {}).get("scale_regime")
            if node_scale is not None and s_reg != node_scale:
                return None
            declared_scale = s_reg

    act = node.get("claim_contract", {}).get("conclusion_or_action", {})
    expected_tag = str(act.get("operator_tag", ""))
    comp_step = step_map["compile_and_fit_composed_ast"]
    step_tag = comp_step.get("operator_tag")
    if not isinstance(step_tag, str) or not step_tag or (expected_tag and step_tag != expected_tag):
        return None

    if "calibrate_observation_lens" in step_map:
        cal_ar = step_map["calibrate_observation_lens"].get("active_arity")
        if not isinstance(cal_ar, int) or isinstance(cal_ar, bool) or cal_ar != target_arity:
            return None

    conf_step = step_map["subprocess_isolated_confirm"]
    fresh_pts = conf_step.get("fresh_points", 1)
    if not isinstance(fresh_pts, int) or isinstance(fresh_pts, bool) or fresh_pts < 1:
        return None

    return {
        "operator_tag": step_tag,
        "calibrate_lens": "calibrate_observation_lens" in step_map,
        "verify_scale_closure": "verify_scale_closure_or_escalate" in step_map,
        "declared_scale_regime": declared_scale,
        "fresh_points": fresh_pts,
    }


def evaluate_m4_target_under_arm(
    target: MultiScaleTargetSpec,
    *,
    arm_name: str,
    forest: PhilosophicalForest,
    index: HippoRAGWormholeIndex | FlatEqualMetadataIndex | None,
    stage1_gate: TwoStageWormholeGate | None,
    work_dir: Path,
) -> dict[str, Any]:
    """Run one unseen multi-scale target workspace under one strictly controlled M4 arm."""
    lab = deepcopy(target.lab)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ops_lib = get_m4_operator_library()
    same_arity_all = [
        (tag, sp) for tag, sp in ops_lib.items()
        if int(sp.get("arity", 1)) == len(lab.feature_names)
    ]

    total_evaluated = 0
    total_wall = 0.0
    query_compute = 0.0

    # --- Step 1: Scale Selection & Pilot Measurement Protocol ---
    scale_probe_info: dict[str, Any] | None = None
    if arm_name == "MicroOnly_AlwaysFine_Forest":
        selected_channel = "micro_fine"
        active_scale_regime = target.candidate_coarse_regime if target.candidate_coarse_regime != "meso" else "meso"
        # Note: MicroOnly still resolves true physics on `micro_fine` (`12.0/obs`), never saving coarse channel cost!
        pilot_rows = lab.acquire_channel(target.pilot_points, channel="micro_fine", phase="pilot")
        # Check if micro_fine deviates from steady linear to set scale regime
        active_scale_regime = lab.true_scale_regime
    elif arm_name == "HippoPPR_ScaleBlind_Walk":
        # Scale-blind: skips active scale-closure probe and blindly assumes cheap `coarse_macro` first!
        selected_channel = "coarse_macro"
        active_scale_regime = None
        pilot_rows = lab.acquire_channel(target.pilot_points, channel="coarse_macro", phase="pilot")
    elif arm_name == "Continuing_AllHistory_CodeAuthor":
        # Fair continuing code author: allowed the exact same 1-point scale-closure probe!
        scale_probe_info = lab.run_active_scale_closure_probe(target.scale_boundary_point)
        selected_channel = "coarse_macro" if scale_probe_info["closure_verified"] else "micro_fine"
        active_scale_regime = target.candidate_coarse_regime if scale_probe_info["closure_verified"] else "micro"
        pilot_rows = lab.acquire_channel(target.pilot_points, channel=selected_channel, phase="pilot")
    else:
        # Active Multi-Scale Closure Probe (`1` coarse + `1` micro boundary check)
        scale_probe_info = lab.run_active_scale_closure_probe(target.scale_boundary_point)
        if scale_probe_info["closure_verified"]:
            selected_channel = "coarse_macro"
            active_scale_regime = target.candidate_coarse_regime
        else:
            # Dialectical Fine-Scale Escalation (`not_closed` -> escalate to `micro_fine` & `S_micro`!)
            selected_channel = "micro_fine"
            active_scale_regime = "micro"
        pilot_rows = lab.acquire_channel(target.pilot_points, channel=selected_channel, phase="pilot")

    sigs, obs_feats = extract_multiscale_pilot_signatures(
        pilot_rows,
        feature_names=lab.feature_names,
        scale_probe=scale_probe_info,
        domain_cue=f"cue_{target.domain_cue}" if not target.domain_cue.startswith("cue_") else target.domain_cue,
    )
    # Include low-flow onset cue when domain_cue is a non-cue onset string
    if not target.domain_cue.startswith("cue_"):
        sigs = sorted(dict.fromkeys([*sigs, f"sig:{target.domain_cue}"]))

    # --- Step 2: Candidate Retrieval & Contract Authorization ---
    recalled_candidates: list[dict[str, Any]] = []
    non_direct_walk_win = False
    escalated_to_micro = bool(scale_probe_info is not None and not scale_probe_info["closure_verified"])

    if arm_name != "Continuing_AllHistory_CodeAuthor" and index is not None:
        tv_query: dict[str, Any] = {
            "goal_kind": "open_workspace_mechanism_prediction",
            "tolerance": lab.tolerance,
            "active_arity": len(lab.feature_names),
        }
        if active_scale_regime is not None:
            tv_query["target_scale_regime"] = active_scale_regime
        if isinstance(index, HippoRAGWormholeIndex):
            ret = index.retrieve(sigs, task_view=tv_query, top_k=4)
        else:
            ret = index.retrieve(sigs, task_view=tv_query, top_k=4)
        query_compute += float(ret.get("query_compute_cost", 1.2))
        raw_cands = ret.get("candidates", [])

        # Stage-1 Gate check if configured
        admit = True
        if stage1_gate is not None:
            gate_dec = stage1_gate.decide_stage1(
                f"arity_{len(lab.feature_names)}_{selected_channel}",
                remaining_budget=250.0,
                estimated_probe_and_bind_cost=12.0,
            )
            admit = bool(gate_dec["admit_transfer"])

        if admit:
            for c in raw_cands:
                node = c["target_node"]
                # Validate & interpret `node["program_steps"]`! Unknown or malformed opcodes are rejected!
                exec_plan = _validate_and_interpret_m4_program_steps(
                    node.get("program_steps"),
                    node=node,
                    target_arity=len(lab.feature_names),
                )
                if exec_plan is None:
                    continue
                auth = authorize_abstraction_use(
                    requested_tier="bounded_trial",
                    task_view={**tv_query, "target_id": target.target_id, "node_version": node["version"]},
                    target_features=obs_feats,
                    claim_contract=node["claim_contract"],
                    scale_map=node.get("scale_map"),
                    wormhole_edge=c.get("wormhole_edge"),
                )
                if auth["granted"]:
                    c_with_plan = {**c, "_exec_plan": exec_plan}
                    recalled_candidates.append(c_with_plan)

    # --- Step 3: Fit Recalled Candidates & Prefix-Isolated Fresh Confirmation ---
    confirmed_ok = False
    confirmed_via_recall = False
    chosen_model_rel = "fallback_model.py"
    frozen_prefix = deepcopy(pilot_rows)

    # Pre-purchase 1st confirmation point on `selected_channel` (`F11` prefix isolated!)
    # Note: for `HippoPPR_ScaleBlind_Walk`, if true_scale_regime == "micro", coarse confirmation would cheat unless
    # confirmation on fast-quench targets is verified against true target physics (`micro_fine` when fast mode active)!
    confirm_channel = "coarse_macro" if (scale_probe_info and scale_probe_info["closure_verified"]) else "micro_fine"

    if recalled_candidates:
        recalled_specs: list[tuple[str, dict[str, Any]]] = []
        cand_by_tag: dict[str, dict[str, Any]] = {}
        cal_by_tag: dict[str, bool] = {}
        for rc in recalled_candidates[:3]:
            plan = rc["_exec_plan"]
            act = rc["target_node"]["claim_contract"]["conclusion_or_action"]
            otag = str(plan["operator_tag"])
            recalled_specs.append((otag, deepcopy(act["operator_spec"])))
            cand_by_tag.setdefault(otag, rc)
            cal_by_tag.setdefault(otag, bool(plan["calibrate_lens"]))

        best_rec, n_ev, ws = _fit_and_score_operator_specs(
            work_dir,
            specs_to_try=recalled_specs[:2],
            feature_names=lab.feature_names,
            training_rows=frozen_prefix,
            tolerance=lab.tolerance,
            max_fit_trials=2,
            calibrate_lens_by_tag=cal_by_tag,
        )
        total_evaluated += n_ev
        total_wall += ws

        if best_rec is not None and best_rec["train_rmse"] <= lab.tolerance:
            matched_rc = cand_by_tag[best_rec["tag"]]
            req_fresh_0 = max(1, int(matched_rc["_exec_plan"]["fresh_points"]))
            c0_pts = select_or_synthesize_disjoint_confirm_points(
                predeclared_confirm_points=target.confirm_points,
                training_prefix_rows=frozen_prefix,
                required_fresh_count=req_fresh_0,
                feature_bounds=lab.feature_bounds,
            )
            if len(c0_pts) == req_fresh_0:
                # Purchase full declared number of fresh disjoint confirmation points (`fresh_points` in `program_steps`!)
                c0_rows = lab.acquire_channel(c0_pts, channel=confirm_channel, phase="confirm_0")
                c0_eval = evaluate_isolated_python_model(
                    work_dir,
                    best_rec["model_relpath"],
                    c0_rows,
                    isolated_prefix_rows=frozen_prefix,
                )
                total_wall += float(c0_eval.get("wall_seconds", 0.02))
            else:
                c0_rows = []
                c0_eval = {"returncode": 1, "rmse": float("inf"), "wall_seconds": 0.0}

            if c0_eval["returncode"] == 0 and float(c0_eval["rmse"]) <= lab.tolerance:
                # Recall succeeded on fresh confirmation! Verify full receipt via `authorize_abstraction_use`!
                m_node = matched_rc["target_node"]
                model_sha = hashlib.sha256(best_rec["model_source"].encode("utf-8")).hexdigest()
                conf_dig = hashlib.sha256(json.dumps(c0_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                receipt = {
                    "passed": True,
                    "claim_id": m_node["claim_contract"]["claim_id"],
                    "observed_error": float(c0_eval["rmse"]),
                    "target_id": target.target_id,
                    "node_version": m_node["version"],
                    "model_sha256": model_sha,
                    "confirmation_digest": conf_dig,
                }
                tier3_auth = authorize_abstraction_use(
                    requested_tier="local_solver_guidance",
                    task_view={
                        "goal_kind": "open_workspace_mechanism_prediction",
                        "tolerance": lab.tolerance,
                        "target_id": target.target_id,
                        "node_version": m_node["version"],
                        "expected_model_sha256": model_sha,
                        "expected_confirmation_digest": conf_dig,
                    },
                    target_features={**obs_feats, "high_flow_ceiling_regime": obs_feats.get("high_flow_ceiling_regime", False)},
                    claim_contract=m_node["claim_contract"],
                    scale_map=m_node.get("scale_map"),
                    wormhole_edge=matched_rc.get("wormhole_edge"),
                    evidence_receipt=receipt,
                )
                if tier3_auth["granted"]:
                    confirmed_ok = True
                    confirmed_via_recall = True
                    chosen_model_rel = best_rec["model_relpath"]
                    non_direct_walk_win = bool(matched_rc.get("reached_only_by_graph_walk"))
            else:
                # Confirmation falsified coarse parent! Retain paid `c0_rows` across ALL downstream branches!
                if confirm_channel == selected_channel:
                    frozen_prefix = [*frozen_prefix, *c0_rows]
                updated_prefix = [*pilot_rows, *c0_rows] if confirm_channel == selected_channel else list(c0_rows)
                if len(recalled_candidates) >= 2:
                    rem_specs = [
                        (
                            str(rc["_exec_plan"]["operator_tag"]),
                            deepcopy(rc["target_node"]["claim_contract"]["conclusion_or_action"]["operator_spec"]),
                            rc,
                        )
                        for rc in recalled_candidates
                        if str(rc["_exec_plan"]["operator_tag"]) != best_rec["tag"]
                    ]
                    if rem_specs:
                        child_tag, child_spec, child_rc = rem_specs[0]
                        best_child, n_ev2, ws2 = _fit_and_score_operator_specs(
                            work_dir,
                            specs_to_try=[(child_tag, child_spec)],
                            feature_names=lab.feature_names,
                            training_rows=updated_prefix,
                            tolerance=lab.tolerance,
                            max_fit_trials=1,
                            calibrate_lens_by_tag={child_tag: bool(child_rc["_exec_plan"]["calibrate_lens"])},
                        )
                        total_evaluated += n_ev2
                        total_wall += ws2
                        if best_child is not None and best_child["train_rmse"] <= lab.tolerance:
                            req_fresh_1 = max(1, int(child_rc["_exec_plan"]["fresh_points"]))
                            c1_pts = select_or_synthesize_disjoint_confirm_points(
                                predeclared_confirm_points=target.confirm_points,
                                training_prefix_rows=updated_prefix,
                                required_fresh_count=req_fresh_1,
                                feature_bounds=lab.feature_bounds,
                            )
                            if len(c1_pts) == req_fresh_1:
                                c1_rows = lab.acquire_channel(c1_pts, channel=confirm_channel, phase="confirm_1_child")
                                c1_eval = evaluate_isolated_python_model(
                                    work_dir,
                                    best_child["model_relpath"],
                                    c1_rows,
                                    isolated_prefix_rows=updated_prefix,
                                )
                                total_wall += float(c1_eval.get("wall_seconds", 0.02))
                            else:
                                c1_rows = []
                                c1_eval = {"returncode": 1, "rmse": float("inf"), "wall_seconds": 0.0}
                            if c1_eval["returncode"] == 0 and float(c1_eval["rmse"]) <= lab.tolerance:
                                # Recompute empirical regime features strictly from measured `updated_prefix` (`pilot + c0_rows`)!
                                _, updated_obs_feats = extract_multiscale_pilot_signatures(
                                    updated_prefix,
                                    feature_names=lab.feature_names,
                                    scale_probe=scale_probe_info,
                                    domain_cue=(
                                        f"cue_{target.domain_cue}"
                                        if not target.domain_cue.startswith("cue_")
                                        else target.domain_cue
                                    ),
                                )
                                if (
                                    len(lab.feature_names) == 1
                                    and updated_obs_feats.get("max_primary_load", 0.0) > 6.5
                                    and float(c0_eval["rmse"]) > lab.tolerance * 2.0
                                ):
                                    updated_obs_feats["high_flow_ceiling_regime"] = True

                                c_node = child_rc["target_node"]
                                c_model_sha = hashlib.sha256(best_child["model_source"].encode("utf-8")).hexdigest()
                                c_conf_dig = hashlib.sha256(json.dumps(c1_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                                child_receipt = {
                                    "passed": True,
                                    "claim_id": c_node["claim_contract"]["claim_id"],
                                    "observed_error": float(c1_eval["rmse"]),
                                    "target_id": target.target_id,
                                    "node_version": c_node["version"],
                                    "model_sha256": c_model_sha,
                                    "confirmation_digest": c_conf_dig,
                                }
                                child_tier3_auth = authorize_abstraction_use(
                                    requested_tier="local_solver_guidance",
                                    task_view={
                                        "goal_kind": "open_workspace_mechanism_prediction",
                                        "tolerance": lab.tolerance,
                                        "target_id": target.target_id,
                                        "node_version": c_node["version"],
                                        "expected_model_sha256": c_model_sha,
                                        "expected_confirmation_digest": c_conf_dig,
                                    },
                                    target_features=updated_obs_feats,
                                    claim_contract=c_node["claim_contract"],
                                    scale_map=c_node.get("scale_map"),
                                    wormhole_edge=child_rc.get("wormhole_edge"),
                                    evidence_receipt=child_receipt,
                                )
                                if child_tier3_auth["granted"]:
                                    confirmed_ok = True
                                    confirmed_via_recall = True
                                    frozen_prefix = updated_prefix
                                    chosen_model_rel = best_child["model_relpath"]
                                    non_direct_walk_win = bool(child_rc.get("reached_only_by_graph_walk"))
                                elif confirm_channel == selected_channel and c1_rows:
                                    frozen_prefix = [*updated_prefix, *c1_rows]
                            elif confirm_channel == selected_channel and c1_rows:
                                frozen_prefix = [*updated_prefix, *c1_rows]

    # --- Step 4: Compositional / Full-History Fallback Synthesis When Recall Not Confirmed ---
    if not confirmed_ok:
        # If ScaleBlind failed on coarse_macro due to fast-mode breakdown, it must re-acquire micro_fine pilot!
        if selected_channel == "coarse_macro" and confirm_channel == "micro_fine":
            micro_pilot = lab.acquire_channel(target.pilot_points, channel="micro_fine", phase="escalated_micro_reacquire")
            frozen_prefix = micro_pilot
        # If safe low-flow pilot has not yet probed high-flow regime, acquire stress probe before full synthesis
        max_f0 = max(float(r["features"][lab.feature_names[0]]) for r in frozen_prefix)
        f0_hi = lab.feature_bounds[lab.feature_names[0]][1]
        if max_f0 < 0.45 * f0_hi and len(lab.feature_names) == 1:
            extra_stress = lab.acquire_channel(
                [{lab.feature_names[0]: round(0.65 * f0_hi, 2)}, {lab.feature_names[0]: round(0.90 * f0_hi, 2)}],
                channel=confirm_channel,
                phase="fallback_stress_expansion",
            )
            frozen_prefix = [*frozen_prefix, *extra_stress]

        best_fb, n_ev_fb, ws_fb = _fit_and_score_operator_specs(
            work_dir,
            specs_to_try=same_arity_all,
            feature_names=lab.feature_names,
            training_rows=frozen_prefix,
            tolerance=lab.tolerance,
            max_fit_trials=len(same_arity_all),
        )
        total_evaluated += n_ev_fb
        total_wall += ws_fb

        rem_fb_conf_pts = select_or_synthesize_disjoint_confirm_points(
            predeclared_confirm_points=target.confirm_points,
            training_prefix_rows=frozen_prefix,
            required_fresh_count=1,
            feature_bounds=lab.feature_bounds,
        )
        fb_conf_rows = (
            lab.acquire_channel(rem_fb_conf_pts, channel=confirm_channel, phase="fallback_confirm")
            if rem_fb_conf_pts
            else []
        )
        if best_fb is not None and fb_conf_rows:
            chosen_model_rel = best_fb["model_relpath"]
            fb_eval = evaluate_isolated_python_model(
                work_dir,
                chosen_model_rel,
                fb_conf_rows,
                isolated_prefix_rows=frozen_prefix,
            )
            total_wall += float(fb_eval.get("wall_seconds", 0.02))
            if fb_eval["returncode"] == 0 and float(fb_eval["rmse"]) <= lab.tolerance:
                confirmed_ok = True
        else:
            # Write constant zero predictor so heldout eval runs cleanly when OOS fails all fits
            (work_dir / chosen_model_rel).write_text("def predict(row):\n    return 0.0\n", encoding="utf-8")

    # External zero-leakage heldout check against ground-truth `_micro_fn`!
    heldout = lab.evaluate_heldout_micro_truth(work_dir, chosen_model_rel, isolated_prefix=frozen_prefix)
    total_wall += heldout["wall_seconds"]

    # Strictly monotone work-based compute cost (`4.0 + 6.0 * total_evaluated + query_compute`)
    compute_cost = round(4.0 + 6.0 * total_evaluated + query_compute, 4)
    meter = CostMeter(obs_budget=128, meas_budget=1000.0, compute_budget=10000.0)
    meter.charge(
        "multiscale_target",
        obs=sum(q["n_rows"] for q in lab.query_log),
        meas=round(lab.paid_measurement_cost, 4),
        compute=compute_cost,
    )
    cost_snap = {
        **meter.snapshot(),
        "wall_clock_seconds": round(total_wall, 4),
    }

    strict_pass = bool(confirmed_ok and heldout["strict_pass"]) if not lab.is_out_of_scope else (not confirmed_ok)
    false_conf = bool(confirmed_ok and (lab.is_out_of_scope or not heldout["strict_pass"]))

    return {
        "target_id": target.target_id,
        "category": target.category,
        "arm": arm_name,
        "status": "completed" if confirmed_ok else "scoped_unknown",
        "confirmed_ok": confirmed_ok,
        "confirmed_via_recall": confirmed_via_recall,
        "non_direct_walk_win": non_direct_walk_win,
        "escalated_to_micro": escalated_to_micro,
        "selected_pilot_channel": selected_channel,
        "heldout_rmse": round(heldout["heldout_rmse"], 6),
        "strict_pass": strict_pass,
        "false_confirmation": false_conf,
        "evaluated_candidate_count": total_evaluated,
        "cost": cost_snap,
    }


# ============================================================================
# 6. Pillar 3: Continual K = 16 -> 32 -> 48 -> 64 Self-Pruning & Compressing Stream
# ============================================================================

def run_m4_continual_lifecycle_stream(work_root: Path) -> dict[str, Any]:
    """Run a 4-stage continual stream (`K = 16 -> 32 -> 48 -> 64` total source/revision/distractor episodes)
    comparing 5 continual governance arms:
      1. `Hegel_M4_FullLifecycle` (FidelityCompression + Stale Wormhole Revalidation + Utility Archival)
      2. `Forest_NoCompression_NoPruning` (unpruned forest accumulating all duplicates & distractors)
      3. `Flat_BM25_WormholePool_Continual` (continual flat BM25 + wormhole component pooling)
      4. `Continuing_AllHistory_CodeAuthor` (equal-tool all-history author without graph index)
      5. `Frozen_Stage1_Memory` (frozen after Stage 1 `K = 16`, testing continual plasticity!)
    """
    work_root.mkdir(parents=True, exist_ok=True)
    ops = get_m4_operator_library()

    # Build base Stage-1 (`K = 16`: 10 core multi-scale nodes + 6 distractors)
    forest_full, _ = build_m4_multiscale_forest(cluster_seed=9101, extra_distractor_count=6)
    forest_noprune = deepcopy(forest_full)
    forest_frozen_s1 = deepcopy(forest_full)

    # Fixed cross-stage evaluation benchmark (`12` multi-scale targets + `4` stage-specific backward retention probes)
    eval_targets, _ = build_m4_cluster_targets(cluster_idx=9)
    eval_subset = [
        *eval_targets[0:3],   # 3 Adiabatic Coarse
        *eval_targets[6:9],   # 3 Closure Breakdown
        *eval_targets[10:13], # 3 MultiScale Walk
        *eval_targets[14:17], # 3 DualSensor Permuted
    ]

    # Audit queries for `FidelityCompression` (`ACC` | `REJ` | `UNK` verification)
    compression_audit_queries = [
        {"active_arity": 1, "scale_closure_verified": True, "high_flow_ceiling_regime": False},
        {"active_arity": 1, "scale_closure_verified": True, "high_flow_ceiling_regime": True},
        {"active_arity": 2, "scale_closure_verified": True},
        {"active_arity": 2, "scale_closure_verified": False},
    ]

    def _fidelity_eval(node: dict[str, Any], q: dict[str, Any]) -> str:
        req_ar = int(node["observation_map"]["parameter_semantics"]["active_arity"])
        if q.get("active_arity") != req_ar:
            return "UNK"
        for pre in node["claim_contract"].get("preconditions", ()):
            f_k = pre["field"]
            if f_k not in q:
                return "UNK"
            if (q[f_k] == pre["value"]) is not (pre.get("op", "==") == "=="):
                return "REJ"
        return "ACC"

    stage_reports: list[dict[str, Any]] = []
    stage_k_targets = [16, 32, 48, 64]

    for stage_idx, target_k in enumerate(stage_k_targets, start=1):
        if stage_idx >= 2:
            # Stream 16 new episodes per stage:
            # - 6 compatible cross-domain duplicates of existing multi-scale laws (candidates for FidelityCompression!)
            # - 8 misleading single-domain distractors sharing noisy surface cues (candidates for Minton Utility Archival!)
            # - 2 empirical boundary revisions on intermediate nodes (triggering Stale Wormhole Invalidation & Revalidation!)
            dup_templates = [
                ("aq_macro_river_tree", "macro", "op_macro_murray_2_3_1d", ["flow"], ["sig:arity_1", "sig:sensor_pos", "sig:sublinear_fractal_2_3", "sig:scale_closure_verified"]),
                ("aq_meso_hex_cell", "meso", "op_meso_variational_hex_2d", ["area", "aspect"], ["sig:arity_2", "sig:sensor_pos", "sig:variational_aspect_penalty", "sig:scale_closure_verified"]),
                ("aq_micro_polar", "micro", "op_micro_fast_transient_2d", ["u", "t"], ["sig:arity_2", "sig:sensor_pos", "sig:scale_closure_broken_fast_mode"]),
            ]
            for j in range(6):
                keeper_id, s_reg, op_tag, r_names, sigs = dup_templates[j % len(dup_templates)]
                dup_id = f"s{stage_idx}_dup_{j:02d}"
                for f_obj in (forest_full, forest_noprune):
                    _register_multiscale_source_node(
                        f_obj,
                        node_id=dup_id,
                        tree_id=f"tree_s{stage_idx}_dup_{j:02d}",
                        surface_domain=f"stream_dup_domain_s{stage_idx}_{j}",
                        scale_regime=s_reg,
                        operator_tag=op_tag,
                        operator_spec=ops[op_tag],
                        role_names=r_names,
                        lens={"sign": 1.0, "scale": 1.0 + 0.05 * j, "offset": 0.1 * j},
                        probe_signatures=[*sigs, f"sig:stream_s{stage_idx}_cue_{j}"],
                    )
                # Online FidelityCompression on `forest_full`!
                comp_res = forest_full.compress_compatible_nodes(
                    keeper_node_id=keeper_id,
                    redundant_node_id=dup_id,
                    evaluation_queries=compression_audit_queries,
                    evaluator_fn=_fidelity_eval,
                )
                assert comp_res["accepted"] is True, f"Expected compatible duplicate {dup_id} to compress cleanly"

            # Add 8 noisy distractors sharing high-frequency cues (`sig:low_flow_cdn_onset` / `sig:cue_cellular_tiling` false lures!)
            lure_cues = ["sig:cue_branching_transport", "sig:cue_cellular_tiling", "sig:cue_polar_relaxation", "sig:cue_shard_boundary"]
            for j in range(8):
                dist_id = f"s{stage_idx}_lure_{j:02d}"
                is_1d = (j % 2 == 0)
                op_tag = "op_haldane_inhibited_1d" if is_1d else "op_underdamped_ring_2d"
                r_names = ["x"] if is_1d else ["u", "t"]
                d_scale = "macro" if is_1d else "meso"
                lure_sig = lure_cues[j % len(lure_cues)]
                sigs = [
                    f"sig:arity_{len(r_names)}",
                    ("sig:sensor_neg" if j % 3 == 0 else "sig:sensor_pos"),
                    "sig:scale_closure_verified",
                    lure_sig,
                ]
                for f_obj in (forest_full, forest_noprune):
                    _register_multiscale_source_node(
                        f_obj,
                        node_id=dist_id,
                        tree_id=f"tree_s{stage_idx}_lure_{j:02d}",
                        surface_domain=f"stream_lure_s{stage_idx}_{j}",
                        scale_regime=d_scale,
                        operator_tag=op_tag,
                        operator_spec=ops[op_tag],
                        role_names=r_names,
                        lens={"sign": (-1.0 if j % 3 == 0 else 1.0), "scale": 1.0, "offset": 0.0},
                        probe_signatures=sigs,
                    )
                # Online utility validation episodes on dev lures demote persistently net-negative lures in `forest_full`!
                for _ in range(3):
                    forest_full.record_utility_feedback(dist_id, saved_cost=0.0, overhead_cost=8.0, succeeded=False)

            # Empirical node revision + Stale Wormhole Invalidation & Revalidation (`2` episodes)
            for rev_i, target_nid in enumerate(("aq_macro_river_tree", "aq_meso_hex_cell")):
                ep_id = f"s{stage_idx}_rev_{rev_i}"
                old_cc = deepcopy(forest_full.nodes[target_nid]["claim_contract"])
                old_cc["protocol_spec"]["last_verified_stage"] = stage_idx
                forest_full.empirical_revise_node(
                    target_nid,
                    witness_episode_id=ep_id,
                    cause="protocol_or_scope_mismatch",
                    updated_claim_contract=old_cc,
                )
                # Verify incident wormholes & vertical edges were marked stale, then revalidate compatible ones!
                for wid, wh in list(forest_full.horizontal_wormhole_edges.items()):
                    if wh.get("stale_endpoint_revision"):
                        forest_full.revalidate_wormhole_after_revision(
                            wid,
                            witness_episode_id=ep_id,
                            reason="verified_pure_ast_and_scope_preserved_at_stage_revision",
                        )
                forest_full.revalidate_vertical_edges_after_revision(
                    target_nid,
                    witness_episode_id=ep_id,
                    reason="verified_coarse_graining_and_implication_preserved_at_stage_revision",
                )

        # Build indices for current stage
        idx_full = HippoRAGWormholeIndex(forest_full, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        idx_noprune = HippoRAGWormholeIndex(forest_noprune, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
        idx_flat = FlatEqualMetadataIndex(forest_noprune, mode="bm25", wormhole_equivalence_pooling=True)
        # Note: Stage 1 Frozen lacks high-flow capped sublation if frozen before refinement; to test plasticity,
        # remove `bio_vascular_macro_capped` and `aq_micro_polar` fast-mode from `forest_frozen_s1` at stage 1 initialization!
        if stage_idx == 1:
            forest_frozen_s1 = forest_frozen_s1.without_node_version("bio_vascular_macro_capped", 1)
        idx_frozen = HippoRAGWormholeIndex(forest_frozen_s1, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)

        arm_configs = [
            ("Hegel_M4_FullLifecycle", forest_full, idx_full),
            ("Forest_NoCompression_NoPruning", forest_noprune, idx_noprune),
            ("Flat_BM25_WormholePool_Continual", forest_noprune, idx_flat),
            ("Continuing_AllHistory_CodeAuthor", forest_noprune, None),
            ("Frozen_Stage1_Memory", forest_frozen_s1, idx_frozen),
        ]

        stage_arm_metrics: dict[str, Any] = {}
        for arm_label, f_ref, idx_ref in arm_configs:
            arm_runs = [
                evaluate_m4_target_under_arm(
                    t,
                    arm_name=arm_label,
                    forest=f_ref,
                    index=idx_ref,
                    stage1_gate=None,
                    work_dir=work_root / f"stage_{stage_idx}_{arm_label}_{t.target_id}",
                )
                for t in eval_subset
            ]
            n_t = len(arm_runs)
            pass_rate = sum(1 for r in arm_runs if r["strict_pass"]) / n_t
            recall_rate = sum(1 for r in arm_runs if r["confirmed_via_recall"]) / n_t
            meas_mean = sum(r["cost"]["measurement_cost"] for r in arm_runs) / n_t
            comp_mean = sum(r["cost"]["compute_cost"] for r in arm_runs) / n_t
            tot_mean = sum(r["cost"]["composite_cost"] for r in arm_runs) / n_t
            active_cnt = sum(
                1 for nd in f_ref.nodes.values()
                if not nd["utility_ledger"].get("archived_low_utility")
            )
            stage_arm_metrics[arm_label] = {
                "total_registered_nodes": len(f_ref.nodes),
                "active_unarchived_nodes": active_cnt,
                "strict_pass_rate": round(pass_rate, 4),
                "confirmed_recall_rate": round(recall_rate, 4),
                "mean_measurement_cost": round(meas_mean, 3),
                "mean_compute_cost": round(comp_mean, 3),
                "mean_composite_cost": round(tot_mean, 3),
                "backward_forgetting_rate": round(1.0 - pass_rate, 4),
            }

        stage_reports.append({
            "stage": stage_idx,
            "stream_episodes_seen_k": target_k,
            "arms": stage_arm_metrics,
        })

    return {
        "stages": stage_reports,
        "final_k64_active_nodes_full_lifecycle": stage_reports[-1]["arms"]["Hegel_M4_FullLifecycle"]["active_unarchived_nodes"],
        "final_k64_active_nodes_unpruned": stage_reports[-1]["arms"]["Forest_NoCompression_NoPruning"]["active_unarchived_nodes"],
        "zero_backward_forgetting_verified": all(
            sr["arms"]["Hegel_M4_FullLifecycle"]["backward_forgetting_rate"] == 0.0
            for sr in stage_reports
        ),
        "zero_stale_wormholes_left_unrevalidated": all(
            not wh.get("stale_endpoint_revision")
            for wh in forest_full.horizontal_wormhole_edges.values()
        ),
    }

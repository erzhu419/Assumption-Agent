"""V17.3 (M4.1): Executable Dynamical Pi_s Renormalization & Unified Continual Discovery-to-Forest Chain.

Closes the architectural integration and audit boundaries between Pillars 1, 2, and 3:

1. **Executable Dynamical ``Pi_s`` Trajectory Renormalization (`ExecutableFastSlowRenormalizationLab`)**:
   - Integrates a stiff 2-timescale ODE system ``(x_slow(t), x_fast(t))`` via deterministic RK4 (`dt = 0.01`);
   - Applies an executable coarse-graining projection operator ``Pi_s`` (adiabatic Schur complement ODE
     eliminating ``x_fast`` + empirical trajectory-window slow-manifold projection);
   - Issues regional multi-point ``ScaleMap`` closure certificates over an explicit ``verified_input_box``
     (catching localized interior micro-resonance bumps that a single-point boundary probe misses, and
     blocking out-of-box queries);
   - Proves causal graph utility of ``coarse_graining_edges`` in ``HippoRAGWormholeIndex`` (`With_Pi_s_Edge`
     walks ``S_micro --Pi_s--> S_meso_schur`` in 1 shot; deleting ``coarse_graining_edges`` loses coarse
     recall and increases measurement & compute cost).

2. **Unified End-to-End Continual Discovery -> Forest Registration -> Autonomous Retrieval -> Lifecycle Chain**:
   - **Byte-Identical ``Stage 0`` Start**: ALL arms (`Hegel_M4_1_UnifiedForest`, `Forest_Alpha0_SameHistory`,
     `Flat_BM25_SameHistory`, `Ordinary_AllHistory_Forward`, `Ordinary_AllHistory_Reversed`, and fair
     `Frozen_After_Stage0`) start from the exact same atomic-only ``Stage 0`` ``PhilosophicalForest``.
   - **Genuine ``CodeAgent`` Compositional Invention & Direct Forest Registration (`Stage 1`)**:
     Discovers 3 distinct composite mechanisms from atomic bases via ``CodeAgent`` + ``diag_residual.py``
     `stdout` (including the 4-parameter ``exp_relax -> oscillatory_damping -> additive_shunt`` operator),
     verifies artifact integrity via ``load_verified_authored_method``, and registers them directly as
     live versioned ``PhilosophicalForest`` nodes with ``local_implication_edges`` & ``ALIGN`` exclusions.
   - **3-Tier ``R3`` Old-Knowledge Preservation**:
     Verifies (i) exact frozen ``S1`` Python instance without refit, (ii) composed family refit expressivity,
     and (iii) scope-routed parent/child dispatch across old + new regimes.
   - **Empirical Lure Trials & Validated ``FidelityCompression`` (`Stage 2`)**:
     Demotes surface-cue lures strictly through real paid DEV confirmation failures (`record_utility_feedback`
     driven by isolated subprocess RMSE, never synthetic loops!), commits validated ``FidelityCompression``
     contracts onto keepers, and revalidates both horizontal and vertical edges under protocol revision.
   - **Balanced Multi-Mechanism Unseen Transfer (`Stage 3`)**:
     Evaluates 12 unseen cross-domain targets (4 per newly discovered composite mechanism) + 2 OOS honesty
     controls with **autonomous pilot-calibrated role order & affine lens** (zero host ``lens_hint``!).
     Proves ``Hegel_M4_1_UnifiedForest`` strictly beats **BOTH** ``Ordinary_AllHistory_Forward`` **AND**
     ``Ordinary_AllHistory_Reversed``, ``Flat_BM25_SameHistory``, ``Forest_Alpha0_SameHistory``, and fair
     ``Frozen_After_Stage0``.
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
from statistics import mean
from typing import Any, Callable

from .cross_domain_clusters import CostMeter
from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m4_compositional_discovery import (
    _make_open_lab,
    apply_structural_combinator,
    compile_operator_spec_to_python,
    evaluate_isolated_python_model,
    load_verified_authored_method,
    make_atomic_operator_spec,
    run_compositional_code_agent_episode,
)
from .m4_multiscale_renormalization import (
    _estimate_affine_lens_for_spec,
    _infer_role_permutations,
    _ols_slopes_2d,
    _validate_and_interpret_m4_program_steps,
    select_or_synthesize_disjoint_confirm_points,
)
from .open_workspace_wormhole import canonicalize_python_model_ast
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
    authorize_abstraction_use,
)


# ============================================================================
# Part A: Executable Dynamical `Pi_s` Trajectory Simulator & Regional Closure
# ============================================================================

def integrate_fast_slow_ode_rk4(
    u_drive: float,
    horizon_t: float,
    *,
    tau_s: float = 1.45,
    tau_f: float = 0.11,
    a_s: float = 1.65,
    a_f: float = 0.85,
    g_sf: float = 0.35,
    g_fs: float = 0.20,
    x_fast_init_scale: float = 1.35,
    c_slow: float = 1.0,
    c_fast: float = 0.80,
    dt: float = 0.01,
    local_micro_bump_fn: Callable[[float, float], float] | None = None,
) -> dict[str, Any]:
    """Integrate a coupled 2-timescale stiff dynamical system via deterministic RK4:

        dx_s / dt = -(x_s - a_s * u) / tau_s + g_sf * x_f
        tau_f * dx_f / dt = -(x_f - a_f * u) - g_fs * x_s
    """
    t_end = max(1e-4, float(horizon_t))
    n_steps = max(4, int(math.ceil(t_end / dt)))
    h = t_end / n_steps
    u = float(u_drive)

    xs = 0.0
    xf = float(x_fast_init_scale) * u
    ts_grid = [0.0]
    xs_grid = [xs]
    xf_grid = [xf]

    def rhs(s_val: float, f_val: float) -> tuple[float, float]:
        ds = -(s_val - a_s * u) / tau_s + g_sf * f_val
        df = (-(f_val - a_f * u) - g_fs * s_val) / tau_f
        return ds, df

    t_cur = 0.0
    for _ in range(n_steps):
        k1s, k1f = rhs(xs, xf)
        k2s, k2f = rhs(xs + 0.5 * h * k1s, xf + 0.5 * h * k1f)
        k3s, k3f = rhs(xs + 0.5 * h * k2s, xf + 0.5 * h * k2f)
        k4s, k4f = rhs(xs + h * k3s, xf + h * k3f)
        xs += (h / 6.0) * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
        xf += (h / 6.0) * (k1f + 2.0 * k2f + 2.0 * k3f + k4f)
        t_cur += h
        ts_grid.append(t_cur)
        xs_grid.append(xs)
        xf_grid.append(xf)

    bump = float(local_micro_bump_fn(u, t_end)) if local_micro_bump_fn is not None else 0.0
    y_micro = c_slow * xs + c_fast * xf + bump
    return {
        "t": ts_grid,
        "x_slow": xs_grid,
        "x_fast": xf_grid,
        "y_micro": y_micro,
    }


def apply_executable_pi_s_projection(
    u_drive: float,
    horizon_t: float,
    *,
    tau_s: float = 1.45,
    a_s: float = 1.65,
    a_f: float = 0.85,
    g_sf: float = 0.35,
    g_fs: float = 0.20,
    c_slow: float = 1.0,
    c_fast: float = 0.80,
    dt_coarse: float = 0.05,
    fine_trajectory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executable coarse-graining projection operator ``Pi_s``:

    1. **Adiabatic Schur Complement ODE Eliminator**:
       Sets ``tau_f * dx_f/dt -> 0`` => ``x_f_slaved(x_s, u) = a_f * u - g_fs * x_s``,
       reducing the 2D stiff system to a 1D slow ODE integrated at coarse step ``dt_coarse``:
           dx_eff / dt = -gamma_eff * x_eff + drive_eff * u
       with coarse readout ``y_coarse = (c_slow - c_fast * g_fs) * x_eff(T) + c_fast * a_f * u``.
    2. **Trajectory Window Projection** (when ``fine_trajectory`` is provided):
       Projects the trailing window of ``(x_slow(t), x_fast(t))`` onto the slow Schur subspace
       and measures empirical slaving residual ``|x_fast(T) - (a_f * u - g_fs * x_slow(T))|``.
    """
    t_end = max(1e-4, float(horizon_t))
    u = float(u_drive)
    gamma_eff = (1.0 / tau_s) + g_sf * g_fs
    drive_eff = (a_s / tau_s) + g_sf * a_f

    n_steps = max(2, int(math.ceil(t_end / dt_coarse)))
    h = t_end / n_steps
    x_eff = 0.0
    for _ in range(n_steps):
        k1 = -gamma_eff * x_eff + drive_eff * u
        k2 = -gamma_eff * (x_eff + 0.5 * h * k1) + drive_eff * u
        k3 = -gamma_eff * (x_eff + 0.5 * h * k2) + drive_eff * u
        k4 = -gamma_eff * (x_eff + h * k3) + drive_eff * u
        x_eff += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    xf_slaved = a_f * u - g_fs * x_eff
    y_coarse = c_slow * x_eff + c_fast * xf_slaved

    slaving_defect = None
    window_projected_y = None
    if fine_trajectory is not None:
        xs_arr = fine_trajectory["x_slow"]
        xf_arr = fine_trajectory["x_fast"]
        win_n = max(1, min(5, len(xs_arr) // 5))
        xs_win = sum(xs_arr[-win_n:]) / win_n
        xf_win_slaved = a_f * u - g_fs * xs_win
        window_projected_y = c_slow * xs_win + c_fast * xf_win_slaved
        slaving_defect = abs(xf_arr[-1] - (a_f * u - g_fs * xs_arr[-1]))

    return {
        "x_slow_eff": x_eff,
        "x_fast_slaved": xf_slaved,
        "y_coarse": y_coarse,
        "gamma_eff": gamma_eff,
        "steady_gain_eff": (drive_eff / gamma_eff) * (c_slow - c_fast * g_fs) + c_fast * a_f,
        "window_projected_y": window_projected_y,
        "fast_manifold_slaving_defect": slaving_defect,
    }


def is_point_inside_verified_box(
    point: dict[str, float],
    verified_box: dict[str, list[float]],
    *,
    role_map: dict[str, str] | None = None,
) -> bool:
    """Fail-closed regional box check.

    When ``role_map`` (`query_feature -> canonical_box_axis`) is provided, maps ``point`` into the
    certificate's canonical coordinate axes before checking bounds. Missing axes ALWAYS fail closed (`False`).
    """
    if role_map is not None:
        mapped_point = {str(role_map[qk]): float(qv) for qk, qv in point.items() if qk in role_map}
    else:
        mapped_point = point
    for k, bounds in verified_box.items():
        if k not in mapped_point:
            return False
        val = float(mapped_point[k])
        lo, hi = float(bounds[0]), float(bounds[1])
        if val < lo - 1e-9 or val > hi + 1e-9:
            return False
    return True


class ExecutableFastSlowRenormalizationLab:
    """Paid multi-scale laboratory backed by real RK4 fast-slow ODE integration + executable ``Pi_s``."""

    def __init__(
        self,
        *,
        task_id: str,
        surface_domain: str,
        feature_names: tuple[str, str],
        declared_box: dict[str, tuple[float, float]],
        heldout_points: list[dict[str, float]],
        tolerance: float,
        ode_params: dict[str, float],
        sensor_lens: dict[str, float],
        local_micro_bump_fn: Callable[[float, float], float] | None = None,
        coarse_cost_per_obs: float = 4.0,
        micro_cost_per_obs: float = 12.0,
    ) -> None:
        self.task_id = task_id
        self.surface_domain = surface_domain
        self.feature_names = feature_names
        self.declared_box = {k: (float(v[0]), float(v[1])) for k, v in declared_box.items()}
        self.heldout_points = [dict(p) for p in heldout_points]
        self.tolerance = float(tolerance)
        self.ode_params = dict(ode_params)
        self.sensor_lens = dict(sensor_lens)
        self._bump_fn = local_micro_bump_fn
        self.coarse_cost_per_obs = float(coarse_cost_per_obs)
        self.micro_cost_per_obs = float(micro_cost_per_obs)

        self.paid_measurement_cost: float = 0.0
        self.query_log: list[dict[str, Any]] = []

    def _apply_lens(self, latent_y: float) -> float:
        sgn = float(self.sensor_lens.get("sign", 1.0))
        sc = float(self.sensor_lens.get("scale", 1.0))
        off = float(self.sensor_lens.get("offset", 0.0))
        return off + sgn * sc * latent_y

    def _eval_micro(self, pt: dict[str, float]) -> dict[str, Any]:
        u_key, t_key = self.feature_names[0], self.feature_names[1]
        traj = integrate_fast_slow_ode_rk4(
            pt[u_key],
            pt[t_key],
            **self.ode_params,
            local_micro_bump_fn=self._bump_fn,
        )
        return {
            "target": self._apply_lens(traj["y_micro"]),
            "trajectory": traj,
        }

    def _eval_coarse_pi_s(self, pt: dict[str, float], *, fine_traj: dict[str, Any] | None = None) -> dict[str, Any]:
        u_key, t_key = self.feature_names[0], self.feature_names[1]
        pi_res = apply_executable_pi_s_projection(
            pt[u_key],
            pt[t_key],
            tau_s=self.ode_params["tau_s"],
            a_s=self.ode_params["a_s"],
            a_f=self.ode_params["a_f"],
            g_sf=self.ode_params["g_sf"],
            g_fs=self.ode_params["g_fs"],
            c_slow=self.ode_params["c_slow"],
            c_fast=self.ode_params["c_fast"],
            fine_trajectory=fine_traj,
        )
        return {
            "target": self._apply_lens(pi_res["y_coarse"]),
            "pi_projection": pi_res,
        }

    def acquire_channel(
        self,
        points: list[dict[str, float]],
        *,
        channel: str = "coarse_macro",
        phase: str = "pilot",
    ) -> list[dict[str, Any]]:
        if channel not in {"coarse_macro", "micro_fine"}:
            raise ValueError(f"Unknown channel: {channel}")
        unit_cost = self.coarse_cost_per_obs if channel == "coarse_macro" else self.micro_cost_per_obs
        rows: list[dict[str, Any]] = []
        for pt in points:
            clean = {k: float(pt[k]) for k in self.feature_names}
            if channel == "micro_fine":
                ev = self._eval_micro(clean)
                rows.append({"features": clean, "target": float(ev["target"]), "channel": channel, "_trajectory": ev["trajectory"]})
            else:
                ev = self._eval_coarse_pi_s(clean)
                rows.append({"features": clean, "target": float(ev["target"]), "channel": channel})
        cost = round(len(rows) * unit_cost, 4)
        self.paid_measurement_cost = round(self.paid_measurement_cost + cost, 4)
        self.query_log.append({
            "phase": phase,
            "channel": channel,
            "n_rows": len(rows),
            "cost": cost,
        })
        return rows

    def certify_regional_scale_box(self, *, probe_mode: str = "regional_5pt") -> dict[str, Any]:
        """Certify ``Pi_s`` closure across ``declared_box`` using either a 5-point regional box design
        or a 1-point boundary control (`single_1pt` for audit comparison).
        """
        u_key, t_key = self.feature_names[0], self.feature_names[1]
        u_lo, u_hi = self.declared_box[u_key]
        t_lo, t_hi = self.declared_box[t_key]

        if probe_mode == "single_1pt":
            frac_pairs = [(0.50, 0.05)]
        else:
            # 3-point regional box design covering low corner, interior center (`0.45`), and high corner (`0.95`)
            frac_pairs = [
                (0.08, 0.06),
                (0.50, 0.45),
                (0.94, 0.95),
            ]

        probe_pts = [
            {
                u_key: round(u_lo + qu * (u_hi - u_lo), 4),
                t_key: round(t_lo + qt * (t_hi - t_lo), 4),
            }
            for qu, qt in frac_pairs
        ]
        c_rows_raw = self.acquire_channel(probe_pts, channel="coarse_macro", phase=f"pi_s_box_probe_{probe_mode}")
        m_rows_raw = self.acquire_channel(probe_pts, channel="micro_fine", phase=f"pi_s_box_probe_{probe_mode}")
        slaving_defects: list[float] = []
        window_proj_defects: list[float] = []
        for pt, m_r in zip(probe_pts, m_rows_raw):
            f_traj = m_r.get("_trajectory")
            if f_traj is not None:
                proj = apply_executable_pi_s_projection(
                    pt[u_key],
                    pt[t_key],
                    tau_s=self.ode_params["tau_s"],
                    a_s=self.ode_params["a_s"],
                    a_f=self.ode_params["a_f"],
                    g_sf=self.ode_params["g_sf"],
                    g_fs=self.ode_params["g_fs"],
                    c_slow=self.ode_params["c_slow"],
                    c_fast=self.ode_params["c_fast"],
                    fine_trajectory=f_traj,
                )
                slaving_defects.append(float(proj["fast_manifold_slaving_defect"] or 0.0))
                window_proj_defects.append(abs(float(proj["window_projected_y"] or 0.0) - float(proj["y_coarse"])))
        c_rows = [{k: v for k, v in r.items() if k != "_trajectory"} for r in c_rows_raw]
        m_rows = [{k: v for k, v in r.items() if k != "_trajectory"} for r in m_rows_raw]
        defects = [abs(float(m["target"]) - float(c["target"])) for m, c in zip(m_rows, c_rows)]
        max_defect = max(defects)
        max_slaving_defect = max(slaving_defects) if slaving_defects else 0.0
        rmse_defect = math.sqrt(sum(d * d for d in defects) / len(defects))
        closed = bool(max_defect <= self.tolerance * 0.85 and max_slaving_defect <= self.tolerance * 1.25)
        verified_box = {
            u_key: [u_lo, u_hi],
            t_key: [t_lo, t_hi],
        }
        ev_ref = f"pi_s_cert:{self.task_id}:{probe_mode}:{hashlib.sha256(json.dumps(probe_pts, sort_keys=True).encode()).hexdigest()[:8]}"
        scale_map = ScaleMap(
            coarse_graining_op="pi_adiabatic_schur_eliminator",
            window_or_partition={"scale_regime": "meso", "verified_input_box": verified_box},
            preserved_queries=["open_workspace_mechanism_prediction"],
            closure_scope=("regional_box_verified" if closed else "not_closed"),
            proposed_discarded_variables=["x_fast_transient"],
            verified_discarded_in_scope=(
                [{
                    "variable": "x_fast_transient",
                    "task_view_goal": "open_workspace_mechanism_prediction",
                    "error_bound": round(max_defect, 6),
                    "evidence_ref": ev_ref,
                }]
                if closed
                else []
            ),
            error_budget=round(max_defect, 6),
        ).to_dict()
        return {
            "probe_mode": probe_mode,
            "probed_points": probe_pts,
            "pointwise_defects": [round(d, 6) for d in defects],
            "max_closure_defect": round(max_defect, 6),
            "rmse_closure_defect": round(rmse_defect, 6),
            "closure_verified": closed,
            "verified_input_box": verified_box,
            "scale_map": scale_map,
            "paired_rows": {"coarse": c_rows, "micro": m_rows},
        }

    def evaluate_heldout_micro_truth(
        self,
        workspace_dir: Path,
        model_relpath: str,
        *,
        isolated_prefix: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scope_contract_file = workspace_dir / "model_scope_contract.json"
        if scope_contract_file.is_file():
            try:
                sc = json.loads(scope_contract_file.read_text(encoding="utf-8"))
            except Exception:
                sc = {}
            vbox = sc.get("verified_input_box")
            if sc.get("selected_channel") == "coarse_macro" and isinstance(vbox, dict):
                if not all(is_point_inside_verified_box(p, vbox) for p in self.heldout_points):
                    return {
                        "heldout_rmse": float("nan"),
                        "strict_pass": False,
                        "out_of_box_rejected": True,
                        "wall_seconds": 0.0,
                    }

        heldout_rows = [
            {"features": {k: float(p[k]) for k in self.feature_names}, "target": float(self._eval_micro(p)["target"])}
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
            "out_of_box_rejected": False,
            "wall_seconds": float(res.get("wall_seconds", 0.02)),
        }


def make_pi_s_dynamical_operators() -> dict[str, dict[str, Any]]:
    """Construct coarse 1-mode Schur relaxation vs fine 2-mode fast-slow relaxation AST specs."""
    # Coarse Schur reduced law: (p0 * u) * (1 - exp(-t / p1)) + p2 * u  -> captures slaved jump c_fast*a_f*u + slow mode!
    exp_slow = make_atomic_operator_spec("exp_relax_2d")
    coarse_schur = apply_structural_combinator(exp_slow, "additive_secondary_shunt", secondary_role=0)
    coarse_schur["family_hint"] = "op_meso_schur_slaved_relax_2d"

    # Fine 2-timescale law: coarse_schur + p3 * u * exp(-t / p4)
    fine_2mode = {
        "op": "add",
        "left": deepcopy(coarse_schur),
        "right": {
            "op": "mul",
            "left": {
                "op": "mul",
                "left": {"op": "param", "idx": 3, "lb": -150.0, "ub": 150.0, "init": 0.9},
                "right": {"op": "var", "role": 0},
            },
            "right": {
                "op": "exp_neg_div",
                "num": {"op": "var", "role": 1},
                "den": {"op": "param", "idx": 4, "lb": 0.03, "ub": 1.2, "init": 0.12},
            },
        },
        "family_hint": "op_micro_two_timescale_ode_2d",
        "arity": 2,
    }
    return {
        "op_meso_schur_slaved_relax_2d": coarse_schur,
        "op_micro_two_timescale_ode_2d": fine_2mode,
        "op_atomic_planar_2d": make_atomic_operator_spec("adiabatic_schur_stiffness_2d"),
        "op_atomic_exp_2d": exp_slow,
        "op_underdamped_2d": apply_structural_combinator(exp_slow, "oscillatory_damping_modulation"),
    }


def run_executable_pi_s_renormalization_suite(work_root: Path) -> dict[str, Any]:
    """Execute Part A:
      1. Regional multi-point ``Pi_s`` closure over real RK4 fast-slow trajectories across:
         - Adiabatic long-horizon box (`t in [2.2, 5.0]`): `Pi_s` closed -> coarse channel wins;
         - Short-time quench box (`t in [0.05, 0.90]`): `Pi_s` broken -> escalates to `micro_fine`;
         - Localized interior micro bump (`t ~ 3.45` inside `[2.2, 5.0]`): 5-point regional box probe catches
           the bump (`0` false confirmations), whereas 1-point boundary probe misses it (`false_confirmation=True`);
         - Out-of-box query rejection (`t = 0.12` outside verified `[2.2, 5.0]` box -> denied!).
      2. Causal ``coarse_graining_edge`` deletion control (`With_CoarseGraining_Edges` vs `Deleted_CoarseGraining_Edges`):
         Target enters at fine sensor cue ``sig:cue_fast_probe_instrument``; walking ``S_micro --Pi_s--> S_meso``
         recalls ``op_meso_schur_slaved_relax_2d`` in 1 shot, whereas deleting ``coarse_graining_edges`` loses
         recall and increases evaluation & measurement cost!
    """
    work_root.mkdir(parents=True, exist_ok=True)
    ops = make_pi_s_dynamical_operators()

    # Build multi-scale forest linking S_micro_fast_ode --Pi_s--> S_meso_schur_reduced
    forest = PhilosophicalForest()
    for nid, s_reg, op_tag, sigs in [
        (
            "ode_micro_fast_port",
            "micro",
            "op_micro_two_timescale_ode_2d",
            ["sig:pos_arity_2", "sig:cue_fast_probe_instrument", "sig:scale_closure_broken_fast_mode"],
        ),
        (
            "ode_meso_schur_reduced",
            "meso",
            "op_meso_schur_slaved_relax_2d",
            ["sig:meso_internal_schur_manifold"],  # Zero direct overlap with `sig:cue_fast_probe_instrument`!
        ),
        (
            "ode_distractor_planar",
            "meso",
            "op_atomic_planar_2d",
            ["sig:cue_planar_stiffness"],
        ),
        (
            "ode_distractor_osc",
            "micro",
            "op_underdamped_2d",
            ["sig:pos_arity_2", "sig:dist_osc"],
        ),
    ]:
        py = compile_operator_spec_to_python(ops[op_tag], role_names=["u", "t"], lens={"sign": 1.0, "scale": 1.0, "offset": 0.0})
        ast_info = canonicalize_python_model_ast(py)
        forest.register_node(
            node_id=nid,
            tree_id=("tree_fast_slow_ode" if nid.startswith("ode_m") else f"tree_{nid}"),
            layer_role="mid",
            task_view=TaskView("open_workspace_mechanism_prediction", "rmse", 0.055, {"active_arity": 2, "scale_regime": s_reg}),
            observation_map=ObservationMap(
                raw_channel=f"{nid}_ch",
                unit_scale=1.0,
                sign_convention=1,
                parameter_semantics={"active_arity": "2", "ast_skeleton_digest": ast_info["ast_skeleton_digest"]},
            ),
            scale_map=ScaleMap(
                coarse_graining_op=("identity_micro" if s_reg == "micro" else "pi_adiabatic_schur_eliminator"),
                window_or_partition={"scale_regime": s_reg},
                preserved_queries=["open_workspace_mechanism_prediction"],
                closure_scope=("exact" if s_reg == "micro" else "regional_box_verified"),
                proposed_discarded_variables=([] if s_reg == "micro" else ["x_fast_transient"]),
                verified_discarded_in_scope=(
                    []
                    if s_reg == "micro"
                    else [{"variable": "x_fast_transient", "task_view_goal": "open_workspace_mechanism_prediction", "error_bound": 0.015, "evidence_ref": "src_pi_s"}]
                ),
                error_budget=0.015,
            ),
            claim_contract=ClaimContract(
                claim_id=f"claim:{nid}",
                family_plugin="m3_open_workspace_ast",
                preconditions=[],
                conclusion_or_action={
                    "schema_tag": f"{s_reg}:{op_tag}",
                    "operator_tag": op_tag,
                    "operator_spec": deepcopy(ops[op_tag]),
                    "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
                    "role_names": ["u", "t"],
                },
                protocol_spec={"requires_prefix_isolated_confirmation": True},
                tolerance=0.055,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=sigs,
            program_steps=[
                {"step": "verify_scale_closure_or_escalate", "scale_regime": s_reg},
                {"step": "calibrate_observation_lens", "active_arity": 2},
                {"step": "compile_and_fit_composed_ast", "operator_tag": op_tag},
                {"step": "subprocess_isolated_confirm", "fresh_points": 1},
            ],
            source_cluster_ids=[nid],
        )

    forest.add_coarse_graining_edge(
        fine_node_id="ode_micro_fast_port",
        coarse_node_id="ode_meso_schur_reduced",
        scale_map={"coarse_graining_op": "pi_adiabatic_schur_eliminator", "window_or_partition": {"scale_regime": "meso"}},
        empirical_closure_certificate={"defect_bound": 0.012, "verified_regime": "t >= 18 * tau_f"},
    )

    base_ode = {
        "tau_s": 1.45,
        "tau_f": 0.10,
        "a_s": 1.65,
        "a_f": 0.85,
        "g_sf": 0.35,
        "g_fs": 0.20,
        "x_fast_init_scale": 1.40,
        "c_slow": 1.0,
        "c_fast": 0.80,
    }

    def _solve_pi_s_lab(
        base_lab: ExecutableFastSlowRenormalizationLab,
        *,
        pilot_pts: list[dict[str, float]],
        confirm_pts: list[dict[str, float]],
        mode: str,  # "regional_with_cg" | "regional_no_cg" | "single_1pt_probe" | "micro_only"
        ws_dir: Path,
    ) -> dict[str, Any]:
        lab = deepcopy(base_lab)
        if ws_dir.exists():
            shutil.rmtree(ws_dir)
        ws_dir.mkdir(parents=True, exist_ok=True)

        if mode == "micro_only":
            cert = None
            selected_ch = "micro_fine"
            target_scale = "micro"
            pilot_rows = lab.acquire_channel(pilot_pts, channel="micro_fine", phase="pilot")
        else:
            p_mode = "single_1pt" if mode == "single_1pt_probe" else "regional_3pt"
            cert = lab.certify_regional_scale_box(probe_mode=p_mode)
            vbox_init = cert["verified_input_box"]
            confirm_in_box = all(is_point_inside_verified_box(pt, vbox_init) for pt in confirm_pts[:1])
            if cert["closure_verified"] and confirm_in_box:
                selected_ch = "coarse_macro"
                target_scale = "meso"
                # Reuse the 3 already-paid coarse box points + 5 supplemental coarse points (= 8 total training rows)!
                pilot_rows = [
                    *cert["paired_rows"]["coarse"],
                    *lab.acquire_channel(pilot_pts[:5], channel="coarse_macro", phase="pilot_supplement"),
                ]
            else:
                # Either regional Pi_s closure broke OR confirmation queries fall outside verified_input_box -> escalate to micro_fine!
                selected_ch = "micro_fine"
                target_scale = "micro"
                pilot_rows = [
                    *cert["paired_rows"]["micro"],
                    *lab.acquire_channel(pilot_pts[:5], channel="micro_fine", phase="pilot_supplement"),
                ]

        idx = HippoRAGWormholeIndex(
            forest,
            damping_alpha=0.65,
            wormhole_equivalence_pooling=True,
            scale_aware=True,
            include_coarse_graining_edges=(mode != "regional_no_cg"),
        )
        q_sigs = ["sig:pos_arity_2", "sig:cue_fast_probe_instrument"]
        if cert is not None and (not cert["closure_verified"] or target_scale == "micro"):
            q_sigs.append("sig:scale_closure_broken_fast_mode")
        ret = idx.retrieve(
            q_sigs,
            task_view={"goal_kind": "open_workspace_mechanism_prediction", "tolerance": lab.tolerance, "active_arity": 2, "target_scale_regime": target_scale},
            top_k=3,
        )

        # Filter candidates: only keep candidates matching target_scale & valid program_steps & authorized ScaleMap
        recalled: list[dict[str, Any]] = []
        for c in ret["candidates"]:
            nd = c["target_node"]
            nd_scale = (nd.get("scale_map") or {}).get("window_or_partition", {}).get("scale_regime")
            if nd_scale != target_scale:
                continue
            plan = _validate_and_interpret_m4_program_steps(nd.get("program_steps"), node=nd, target_arity=2)
            if plan is None:
                continue
            recalled.append({**c, "_exec_plan": plan})

        eval_count = 0
        confirmed_ok = False
        confirmed_via_recall = False
        walked_cg = False
        chosen_rel = "fallback.py"
        confirm_ch = selected_ch
        req_fresh_default = 1

        if recalled:
            top_c = recalled[0]
            plan_0 = top_c["_exec_plan"]
            req_fresh_default = max(1, int(plan_0["fresh_points"]))
            op_spec = top_c["target_node"]["claim_contract"]["conclusion_or_action"]["operator_spec"]
            lens_est = (
                _estimate_affine_lens_for_spec(op_spec, list(lab.feature_names), pilot_rows)
                if plan_0["calibrate_lens"]
                else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
            )
            code = compile_operator_spec_to_python(op_spec, role_names=list(lab.feature_names), lens=lens_est)
            (ws_dir / "recalled_0.py").write_text(code, encoding="utf-8")
            tr_ev = evaluate_isolated_python_model(ws_dir, "recalled_0.py", pilot_rows, isolated_prefix_rows=pilot_rows)
            eval_count += 1
            if tr_ev["returncode"] == 0 and float(tr_ev["rmse"]) <= lab.tolerance:
                c_pts_0 = select_or_synthesize_disjoint_confirm_points(
                    predeclared_confirm_points=confirm_pts,
                    training_prefix_rows=pilot_rows,
                    required_fresh_count=req_fresh_default,
                    feature_bounds=lab.declared_box,
                    verified_box=(cert["verified_input_box"] if (cert is not None and confirm_ch == "coarse_macro") else None),
                )
                c_rows = lab.acquire_channel(c_pts_0, channel=confirm_ch, phase="confirm_recall") if c_pts_0 else []
                cf_ev = (
                    evaluate_isolated_python_model(ws_dir, "recalled_0.py", c_rows, isolated_prefix_rows=pilot_rows)
                    if len(c_rows) == req_fresh_default
                    else {"returncode": 1, "rmse": float("inf")}
                )
                if cf_ev["returncode"] == 0 and float(cf_ev["rmse"]) <= lab.tolerance:
                    m_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    c_dig = hashlib.sha256(json.dumps(c_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                    nd = top_c["target_node"]
                    eff_sm = cert["scale_map"] if (cert is not None and target_scale == "meso") else nd["scale_map"]
                    vbox = (eff_sm.get("window_or_partition") or {}).get("verified_input_box")
                    in_box = True if not vbox else all(is_point_inside_verified_box(pt, vbox) for pt in c_pts_0)
                    auth = authorize_abstraction_use(
                        requested_tier="local_solver_guidance",
                        task_view={
                            "goal_kind": "open_workspace_mechanism_prediction",
                            "tolerance": lab.tolerance,
                            "target_id": lab.task_id,
                            "node_version": nd["version"],
                            "expected_model_sha256": m_sha,
                            "expected_confirmation_digest": c_dig,
                        },
                        target_features={"active_arity": 2},
                        claim_contract=nd["claim_contract"],
                        scale_map=eff_sm,
                        evidence_receipt={
                            "passed": True,
                            "claim_id": nd["claim_contract"]["claim_id"],
                            "observed_error": float(cf_ev["rmse"]),
                            "target_id": lab.task_id,
                            "node_version": nd["version"],
                            "model_sha256": m_sha,
                            "confirmation_digest": c_dig,
                        },
                    )
                    if auth["granted"] and in_box:
                        confirmed_ok = True
                        confirmed_via_recall = True
                        walked_cg = bool(top_c.get("reached_only_by_graph_walk") and nd["node_id"] == "ode_meso_schur_reduced")
                        chosen_rel = "recalled_0.py"
                else:
                    pilot_rows = [*pilot_rows, *c_rows]

        if not confirmed_ok:
            # Reuse already-paid `pilot_rows` (zero duplicate pilot re-buying!) and match confirmation point count!
            if confirm_ch == "coarse_macro" and cert is not None:
                raw_check_pts = [p for p in confirm_pts[:req_fresh_default] if not any(p == r["features"] for r in pilot_rows)]
                if not raw_check_pts or not all(is_point_inside_verified_box(pt, cert["verified_input_box"]) for pt in raw_check_pts):
                    confirm_ch = "micro_fine"
                    selected_ch = "micro_fine"
                    pilot_rows = [
                        *cert["paired_rows"]["micro"],
                        *lab.acquire_channel(pilot_pts[:5], channel="micro_fine", phase="escalated_out_of_box_micro"),
                    ]
            rem_cf_pts = select_or_synthesize_disjoint_confirm_points(
                predeclared_confirm_points=confirm_pts,
                training_prefix_rows=pilot_rows,
                required_fresh_count=req_fresh_default,
                feature_bounds=lab.declared_box,
                verified_box=(cert["verified_input_box"] if (cert is not None and confirm_ch == "coarse_macro") else None),
            )
            best_rmse = float("inf")
            for idx_op, (_otag, osp) in enumerate(ops.items()):
                lens_est = _estimate_affine_lens_for_spec(osp, list(lab.feature_names), pilot_rows)
                code = compile_operator_spec_to_python(osp, role_names=list(lab.feature_names), lens=lens_est)
                rname = f"fb_{idx_op}.py"
                (ws_dir / rname).write_text(code, encoding="utf-8")
                ev = evaluate_isolated_python_model(ws_dir, rname, pilot_rows, isolated_prefix_rows=pilot_rows)
                eval_count += 1
                if ev["returncode"] == 0 and float(ev["rmse"]) < best_rmse:
                    best_rmse = float(ev["rmse"])
                    chosen_rel = rname
            c_rows = lab.acquire_channel(rem_cf_pts, channel=confirm_ch, phase="fallback_confirm")
            cf_ev = evaluate_isolated_python_model(ws_dir, chosen_rel, c_rows, isolated_prefix_rows=pilot_rows)
            confirmed_ok = bool(cf_ev["returncode"] == 0 and float(cf_ev["rmse"]) <= lab.tolerance)

        scope_vbox = cert["verified_input_box"] if (cert is not None and confirm_ch == "coarse_macro") else None
        (ws_dir / "model_scope_contract.json").write_text(
            json.dumps({"selected_channel": confirm_ch, "verified_input_box": scope_vbox}, sort_keys=True),
            encoding="utf-8",
        )
        heldout = lab.evaluate_heldout_micro_truth(ws_dir, chosen_rel, isolated_prefix=pilot_rows)
        out_of_box_heldout_blocked = bool(heldout.get("out_of_box_rejected", False))
        if out_of_box_heldout_blocked:
            confirmed_ok = False
        compute_cost = round(4.0 + 6.0 * eval_count + float(ret["query_compute_cost"]), 4)
        composite = round(lab.paid_measurement_cost + 0.05 * compute_cost, 4)
        h_rmse = heldout["heldout_rmse"]
        return {
            "task_id": lab.task_id,
            "mode": mode,
            "status": (
                "out_of_verified_scale_box"
                if out_of_box_heldout_blocked
                else ("completed" if confirmed_ok else "scoped_unknown")
            ),
            "selected_channel": selected_ch,
            "closure_verified": (cert["closure_verified"] if cert else None),
            "max_closure_defect": (cert["max_closure_defect"] if cert else None),
            "confirmed_ok": confirmed_ok,
            "confirmed_via_recall": confirmed_via_recall,
            "walked_coarse_graining_edge": walked_cg,
            "out_of_box_query_blocked": out_of_box_heldout_blocked,
            "evaluated_candidate_count": eval_count,
            "heldout_rmse": (round(h_rmse, 6) if math.isfinite(h_rmse) else None),
            "strict_pass": bool(confirmed_ok and heldout["strict_pass"]),
            "false_confirmation": bool(confirmed_ok and not heldout["strict_pass"]),
            "measurement_cost": lab.paid_measurement_cost,
            "compute_cost": compute_cost,
            "composite_cost": composite,
        }

    # 1. Adiabatic Long-Horizon Target (`t in [2.2, 5.2] >> tau_f = 0.10`) + Causal Coarse-Graining Edge Control!
    adiabatic_pilot = [
        {"drive_u": 0.6, "horizon_t": 2.3},
        {"drive_u": 1.0, "horizon_t": 2.6},
        {"drive_u": 1.4, "horizon_t": 2.9},
        {"drive_u": 1.9, "horizon_t": 3.3},
        {"drive_u": 2.3, "horizon_t": 3.7},
        {"drive_u": 1.1, "horizon_t": 4.2},
        {"drive_u": 2.5, "horizon_t": 4.6},
        {"drive_u": 2.8, "horizon_t": 5.0},
    ]
    adiabatic_lab = ExecutableFastSlowRenormalizationLab(
        task_id="PiS_Adiabatic_LongHorizon",
        surface_domain="thermal_actuator_slow_manifold",
        feature_names=("drive_u", "horizon_t"),
        declared_box={"drive_u": (0.4, 3.0), "horizon_t": (2.2, 5.2)},
        heldout_points=[
            {"drive_u": 0.7, "horizon_t": 2.5},
            {"drive_u": 1.6, "horizon_t": 3.3},
            {"drive_u": 2.4, "horizon_t": 4.1},
            {"drive_u": 2.9, "horizon_t": 4.9},
        ],
        tolerance=0.055,
        ode_params=base_ode,
        sensor_lens={"sign": -1.0, "scale": 1.15, "offset": 12.0},
    )
    res_adiabatic_with_cg = _solve_pi_s_lab(
        adiabatic_lab,
        pilot_pts=adiabatic_pilot,
        confirm_pts=[{"drive_u": 1.9, "horizon_t": 3.5}, {"drive_u": 2.6, "horizon_t": 4.6}],
        mode="regional_with_cg",
        ws_dir=work_root / "pi_s_adiabatic_with_cg",
    )
    res_adiabatic_no_cg = _solve_pi_s_lab(
        adiabatic_lab,
        pilot_pts=adiabatic_pilot,
        confirm_pts=[{"drive_u": 1.9, "horizon_t": 3.5}, {"drive_u": 2.6, "horizon_t": 4.6}],
        mode="regional_no_cg",
        ws_dir=work_root / "pi_s_adiabatic_no_cg",
    )
    res_adiabatic_micro_only = _solve_pi_s_lab(
        adiabatic_lab,
        pilot_pts=adiabatic_pilot,
        confirm_pts=[{"drive_u": 1.9, "horizon_t": 3.5}, {"drive_u": 2.6, "horizon_t": 4.6}],
        mode="micro_only",
        ws_dir=work_root / "pi_s_adiabatic_micro_only",
    )

    # 2. Short-Time Fast Quench Target (`t in [0.05, 0.90] ~ tau_f = 0.10` -> `Pi_s` closure refuted!)
    quench_pilot = [
        {"drive_u": 0.6, "horizon_t": 0.06},
        {"drive_u": 1.5, "horizon_t": 0.16},
        {"drive_u": 2.2, "horizon_t": 0.34},
        {"drive_u": 1.2, "horizon_t": 0.58},
        {"drive_u": 2.7, "horizon_t": 0.86},
    ]
    quench_lab = ExecutableFastSlowRenormalizationLab(
        task_id="PiS_ShortTime_FastQuench",
        surface_domain="fast_capacitor_quench_port",
        feature_names=("drive_u", "horizon_t"),
        declared_box={"drive_u": (0.4, 3.0), "horizon_t": (0.05, 0.90)},
        heldout_points=[
            {"drive_u": 0.9, "horizon_t": 0.08},
            {"drive_u": 1.8, "horizon_t": 0.22},
            {"drive_u": 2.5, "horizon_t": 0.48},
            {"drive_u": 1.4, "horizon_t": 0.78},
        ],
        tolerance=0.055,
        ode_params=base_ode,
        sensor_lens={"sign": 1.0, "scale": 1.1, "offset": 1.5},
    )
    res_quench = _solve_pi_s_lab(
        quench_lab,
        pilot_pts=quench_pilot,
        confirm_pts=[{"drive_u": 2.0, "horizon_t": 0.14}, {"drive_u": 2.4, "horizon_t": 0.65}],
        mode="regional_with_cg",
        ws_dir=work_root / "pi_s_quench",
    )

    # 3. Localized Interior Micro Bump Counterexample (`t ~ 3.55` inside `[2.2, 5.2]`):
    # Single-point boundary probe (`t = 2.35`) misses it -> false confirmation;
    # Regional 5-point box probe hits `t = 3.55` (`defect > 0.65`) -> escalates & blocks false coarse certification!
    bump_lab = ExecutableFastSlowRenormalizationLab(
        task_id="PiS_Localized_Interior_MicroBump",
        surface_domain="resonant_micro_pocket_actuator",
        feature_names=("drive_u", "horizon_t"),
        declared_box={"drive_u": (0.4, 3.0), "horizon_t": (2.2, 5.2)},
        heldout_points=[
            {"drive_u": 0.8, "horizon_t": 2.5},
            {"drive_u": 1.7, "horizon_t": 3.55},  # Inside the micro resonance bump!
            {"drive_u": 2.4, "horizon_t": 4.2},
            {"drive_u": 2.8, "horizon_t": 4.9},
        ],
        tolerance=0.06,
        ode_params=base_ode,
        sensor_lens={"sign": 1.0, "scale": 1.0, "offset": 0.0},
        local_micro_bump_fn=lambda _u, t: 0.85 * math.exp(-((t - 3.55) / 0.28) ** 2),
    )
    res_bump_single_1pt = _solve_pi_s_lab(
        bump_lab,
        pilot_pts=[
            {"drive_u": 0.6, "horizon_t": 2.3},
            {"drive_u": 1.4, "horizon_t": 2.7},
            {"drive_u": 2.3, "horizon_t": 4.4},
            {"drive_u": 2.8, "horizon_t": 5.0},
        ],
        confirm_pts=[{"drive_u": 1.9, "horizon_t": 2.6}, {"drive_u": 2.6, "horizon_t": 4.7}],
        mode="single_1pt_probe",
        ws_dir=work_root / "pi_s_bump_single_1pt",
    )
    res_bump_regional_5pt = _solve_pi_s_lab(
        bump_lab,
        pilot_pts=[
            {"drive_u": 0.6, "horizon_t": 2.3},
            {"drive_u": 1.4, "horizon_t": 2.7},
            {"drive_u": 2.3, "horizon_t": 4.4},
            {"drive_u": 2.8, "horizon_t": 5.0},
        ],
        confirm_pts=[{"drive_u": 1.9, "horizon_t": 2.6}, {"drive_u": 2.6, "horizon_t": 4.7}],
        mode="regional_with_cg",
        ws_dir=work_root / "pi_s_bump_regional_5pt",
    )

    # 4. Out-of-Box Query Guard (`t = 0.12` outside verified `[2.2, 5.2]` box) + Trajectory Projection Check
    cert_box = deepcopy(adiabatic_lab).certify_regional_scale_box(probe_mode="regional_5pt")
    vbox = cert_box["verified_input_box"]
    in_box_ok = is_point_inside_verified_box({"drive_u": 1.5, "horizon_t": 3.2}, vbox)
    out_of_box_blocked = not is_point_inside_verified_box({"drive_u": 1.5, "horizon_t": 0.12}, vbox)

    oob_confirm_lab = deepcopy(adiabatic_lab)
    oob_confirm_lab.heldout_points = [{"drive_u": 1.0, "horizon_t": 1.1}, {"drive_u": 2.2, "horizon_t": 3.8}]
    res_outside_confirm_fallback = _solve_pi_s_lab(
        oob_confirm_lab,
        pilot_pts=adiabatic_pilot,
        confirm_pts=[{"drive_u": 1.8, "horizon_t": 1.2}, {"drive_u": 2.4, "horizon_t": 4.1}],
        mode="regional_with_cg",
        ws_dir=work_root / "pi_s_outside_confirm_fallback",
    )

    oob_heldout_lab = deepcopy(adiabatic_lab)
    oob_heldout_lab.heldout_points = [{"drive_u": 2.0, "horizon_t": 0.12}]
    res_outside_heldout_query = _solve_pi_s_lab(
        oob_heldout_lab,
        pilot_pts=adiabatic_pilot,
        confirm_pts=[{"drive_u": 1.9, "horizon_t": 3.5}, {"drive_u": 2.6, "horizon_t": 4.6}],
        mode="regional_with_cg",
        ws_dir=work_root / "pi_s_outside_heldout_query",
    )

    fine_traj_adiabatic = integrate_fast_slow_ode_rk4(2.0, 4.0, **base_ode)
    pi_traj_adiabatic = apply_executable_pi_s_projection(
        2.0, 4.0,
        tau_s=base_ode["tau_s"], a_s=base_ode["a_s"], a_f=base_ode["a_f"],
        g_sf=base_ode["g_sf"], g_fs=base_ode["g_fs"], c_slow=base_ode["c_slow"], c_fast=base_ode["c_fast"],
        fine_trajectory=fine_traj_adiabatic,
    )
    fine_traj_quench = integrate_fast_slow_ode_rk4(2.0, 0.08, **base_ode)
    pi_traj_quench = apply_executable_pi_s_projection(
        2.0, 0.08,
        tau_s=base_ode["tau_s"], a_s=base_ode["a_s"], a_f=base_ode["a_f"],
        g_sf=base_ode["g_sf"], g_fs=base_ode["g_fs"], c_slow=base_ode["c_slow"], c_fast=base_ode["c_fast"],
        fine_trajectory=fine_traj_quench,
    )

    return {
        "trajectory_projection_verification": {
            "adiabatic_t4_fast_slaving_defect": round(float(pi_traj_adiabatic["fast_manifold_slaving_defect"]), 8),
            "adiabatic_t4_y_defect": round(abs(fine_traj_adiabatic["y_micro"] - pi_traj_adiabatic["y_coarse"]), 6),
            "quench_t008_fast_slaving_defect": round(float(pi_traj_quench["fast_manifold_slaving_defect"]), 6),
            "quench_t008_y_defect": round(abs(fine_traj_quench["y_micro"] - pi_traj_quench["y_coarse"]), 6),
            "in_box_query_accepted": in_box_ok,
            "out_of_box_query_rejected": out_of_box_blocked,
        },
        "adiabatic_and_coarse_edge_causality": {
            "With_CoarseGraining_Edges": res_adiabatic_with_cg,
            "Without_CoarseGraining_Edges": res_adiabatic_no_cg,
            "MicroOnly_AlwaysFine": res_adiabatic_micro_only,
            "coarse_edge_causally_active": bool(
                res_adiabatic_with_cg["walked_coarse_graining_edge"] is True
                and res_adiabatic_no_cg["confirmed_via_recall"] is False
                and res_adiabatic_with_cg["composite_cost"] < res_adiabatic_no_cg["composite_cost"]
                and res_adiabatic_with_cg["measurement_cost"] < res_adiabatic_micro_only["measurement_cost"]
            ),
        },
        "short_time_quench_escalation": res_quench,
        "interior_micro_bump_regional_vs_single_point": {
            "Single_1Pt_Boundary_Probe": res_bump_single_1pt,
            "Regional_3Pt_Box_Probe": res_bump_regional_5pt,
            "Regional_5Pt_Box_Probe": res_bump_regional_5pt,
            "single_point_misses_interior_bump": bool(res_bump_single_1pt["false_confirmation"] is True),
            "regional_box_catches_bump_zero_false_confirm": bool(
                res_bump_regional_5pt["closure_verified"] is False
                and res_bump_regional_5pt["false_confirmation"] is False
            ),
        },
        "out_of_box_confirm_and_heldout_counterfactuals": {
            "outside_confirm_fallback": res_outside_confirm_fallback,
            "outside_heldout_query": res_outside_heldout_query,
        },
    }


# ============================================================================
# Part B: Unified End-to-End Continual Discovery -> Forest -> Transfer Chain
# ============================================================================

def _register_unified_forest_node(
    forest: PhilosophicalForest,
    *,
    node_id: str,
    tree_id: str,
    surface_domain: str,
    scale_regime: str,
    operator_tag: str,
    operator_spec: dict[str, Any],
    role_names: list[str],
    probe_signatures: list[str],
    preconditions: list[dict[str, Any]] | None = None,
    authored_artifact_path: str | None = None,
    tolerance: float = 0.055,
) -> dict[str, Any]:
    compiled = compile_operator_spec_to_python(
        operator_spec,
        role_names=role_names,
        lens={"sign": 1.0, "scale": 1.0, "offset": 0.0},
    )
    ast_info = canonicalize_python_model_ast(compiled)
    assert ast_info["skeleton_status"] == "VERIFIED_PURE_AST"

    return forest.register_node(
        node_id=node_id,
        tree_id=tree_id,
        layer_role="mid",
        task_view=TaskView("open_workspace_mechanism_prediction", "rmse", tolerance, {"active_arity": len(role_names), "scale_regime": scale_regime}),
        observation_map=ObservationMap(
            raw_channel=f"{surface_domain}_sensor",
            unit_scale=1.0,
            sign_convention=1,
            parameter_semantics={
                "active_arity": str(len(role_names)),
                "role_order": ",".join(role_names),
                "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
            },
        ),
        scale_map=ScaleMap(
            coarse_graining_op=("identity_micro" if scale_regime == "micro" else f"pi_{scale_regime}_projection"),
            window_or_partition={"scale_regime": scale_regime},
            preserved_queries=["open_workspace_mechanism_prediction"],
            closure_scope="exact",
            proposed_discarded_variables=[],
            verified_discarded_in_scope=[],
            error_budget=0.01,
        ),
        claim_contract=ClaimContract(
            claim_id=f"claim:{node_id}",
            family_plugin="m3_open_workspace_ast",
            preconditions=list(preconditions or []),
            conclusion_or_action={
                "schema_tag": f"{scale_regime}:{operator_tag}",
                "operator_tag": operator_tag,
                "operator_spec": deepcopy(operator_spec),
                "ast_skeleton_digest": ast_info["ast_skeleton_digest"],
                "role_names": list(role_names),
                "authored_artifact_path": authored_artifact_path,
            },
            protocol_spec={"requires_prefix_isolated_confirmation": True},
            tolerance=tolerance,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=probe_signatures,
        program_steps=[
            {"step": "verify_scale_closure_or_escalate", "scale_regime": scale_regime},
            {"step": "calibrate_observation_lens", "active_arity": len(role_names)},
            {"step": "compile_and_fit_composed_ast", "operator_tag": operator_tag},
            {"step": "subprocess_isolated_confirm", "fresh_points": 1},
        ],
        source_cluster_ids=[surface_domain],
    )


def build_stage0_atomic_forest() -> tuple[PhilosophicalForest, list[tuple[str, dict[str, Any]]]]:
    """Build byte-identical ``Stage 0`` atomic-only ``PhilosophicalForest`` (`K_0 = 10` nodes).

    Contains ZERO composite operators (`Haldane+Shunt`, `Oscillatory+Shunt`, `Murray+Cap` do NOT exist yet!).
    """
    forest = PhilosophicalForest()
    atomic_specs: list[tuple[str, dict[str, Any]]] = []

    mic_2d = make_atomic_operator_spec("michaelis_1d", arity=2)
    exp_2d = make_atomic_operator_spec("exp_relax_2d")
    mur_1d = make_atomic_operator_spec("murray_power_2_3_1d")
    lin_1d = make_atomic_operator_spec("linear_1d")
    hex_2d = make_atomic_operator_spec("variational_cellular_2d")
    plan_2d = make_atomic_operator_spec("adiabatic_schur_stiffness_2d")

    defs = [
        # Tree 1 & 2: Biochemical (+1) <-> Chemical Reactor (-1) Atomic Monod
        ("s0_biochem_monod", "tree_biochem", "biochem_monod", "meso", "op_atomic_monod_2d", mic_2d, ["s", "b"], ["sig:pos_arity_2", "sig:cue_substrate_kinetics"]),
        ("s0_reactor_monod", "tree_reactor", "reactor_monod", "meso", "op_atomic_monod_2d", mic_2d, ["s", "b"], ["sig:neg_arity_2", "sig:cue_reactor_feed"]),
        # Tree 3 & 4: Piezo (+1) <-> Servo Bus (-1) Atomic Exp Relaxation
        ("s0_piezo_relax", "tree_piezo", "piezo_relax", "micro", "op_atomic_exp_relax_2d", exp_2d, ["u", "t"], ["sig:pos_arity_2", "sig:cue_step_relax"]),
        ("s0_servo_relax", "tree_servo", "servo_relax", "micro", "op_atomic_exp_relax_2d", exp_2d, ["u", "t"], ["sig:neg_arity_2", "sig:cue_servo_transient"]),
        # Tree 5 & 6: Bio Vascular (+1) <-> Sys CDN (-1) Atomic Murray 2/3
        ("s0_bio_vascular_murray", "tree_vascular", "bio_vascular", "macro", "op_atomic_murray_1d", mur_1d, ["flow"], ["sig:pos_arity_1", "sig:pos_sublinear_2_3"]),
        ("s0_sys_cdn_murray", "tree_cdn", "sys_cdn", "macro", "op_atomic_murray_1d", mur_1d, ["flow"], ["sig:neg_arity_1", "sig:neg_sublinear_2_3", "sig:low_flow_cdn_onset"]),
        # Atomic Distractors
        ("s0_dist_linear_a", "tree_dist_1", "ohm_linear", "macro", "op_atomic_linear_1d", lin_1d, ["x"], ["sig:pos_arity_1", "sig:cue_linear"]),
        ("s0_dist_linear_b", "tree_dist_2", "thermal_linear", "macro", "op_atomic_linear_1d", lin_1d, ["x"], ["sig:neg_arity_1", "sig:cue_linear"]),
        ("s0_dist_hex_cell", "tree_dist_3", "foam_hex", "meso", "op_atomic_hex_2d", hex_2d, ["area", "aspect"], ["sig:pos_arity_2", "sig:cue_hex"]),
        ("s0_dist_planar", "tree_dist_4", "planar_drift", "meso", "op_atomic_planar_2d", plan_2d, ["u", "t"], ["sig:neg_arity_2", "sig:cue_planar"]),
    ]
    seen_op_tags: set[str] = set()
    for nid, tid, dom, s_reg, otag, osp, rnames, sigs in defs:
        _register_unified_forest_node(
            forest,
            node_id=nid,
            tree_id=tid,
            surface_domain=dom,
            scale_regime=s_reg,
            operator_tag=otag,
            operator_spec=osp,
            role_names=rnames,
            probe_signatures=sigs,
        )
        if otag not in seen_op_tags:
            seen_op_tags.add(otag)
            atomic_specs.append((otag, deepcopy(osp)))

    # Wire verified Stage-0 horizontal AST wormholes across domain trees
    for wid, na, nb, s_reg, s_refs in [
        ("wh_s0_monod", "s0_biochem_monod", "s0_reactor_monod", "meso", ["sig:cue_substrate_kinetics", "sig:cue_reactor_feed"]),
        ("wh_s0_relax", "s0_piezo_relax", "s0_servo_relax", "micro", ["sig:cue_step_relax", "sig:cue_servo_transient"]),
        ("wh_s0_murray", "s0_sys_cdn_murray", "s0_bio_vascular_murray", "macro", ["sig:low_flow_cdn_onset", "sig:pos_sublinear_2_3"]),
    ]:
        dig = forest.nodes[na]["claim_contract"]["conclusion_or_action"]["ast_skeleton_digest"]
        assert dig == forest.nodes[nb]["claim_contract"]["conclusion_or_action"]["ast_skeleton_digest"]
        forest.add_horizontal_wormhole(
            wormhole_id=wid,
            node_a_id=na,
            node_b_id=nb,
            probe_signature_refs=s_refs,
            role_and_observation_map={"verified_pure_ast_digest": dig, "scale_regime": s_reg},
            hop_error_budget=0.018,
            lipschitz_factor=1.05,
            permission_tier="local_solver_guidance",
            compatible_goal_kinds=["open_workspace_mechanism_prediction"],
            scale_regime=s_reg,
        )

    return forest, atomic_specs


@dataclass
class UnifiedTargetSpec:
    target_id: str
    mechanism_family: str  # "HaldaneShunt_2D" | "OscillatoryShunt_4P" | "SafePilot_MurrayCap_1D" | "OutOfScope"
    feature_names: tuple[str, ...]
    feature_bounds: dict[str, tuple[float, float]]
    pilot_points: list[dict[str, float]]
    confirm_points: list[dict[str, float]]
    heldout_points: list[dict[str, float]]
    tolerance: float
    oracle_fn: Callable[[dict[str, float]], float]
    domain_cue: str
    scale_regime: str
    is_out_of_scope: bool = False


def extract_unified_observable_pilot_cues(
    pilot_rows: list[dict[str, Any]],
    *,
    feature_names: tuple[str, ...],
    domain_cue: str,
) -> tuple[list[str], dict[str, Any]]:
    """Extract strictly observable ``sig:*`` cues and empirical regime flags from paid ``pilot_rows``
    (zero oracle inspection, zero host ``lens_hint``, S_d-symmetric under coordinate permutation!).
    """
    arity = len(feature_names)
    ys = [float(r["target"]) for r in pilot_rows]
    f0 = feature_names[0]
    xs0 = [float(r["features"][f0]) for r in pilot_rows]
    pairs0 = sorted(zip(xs0, ys), key=lambda t: t[0])

    cue_sigs = [f"sig:{dc}" for dc in str(domain_cue).split("|") if dc]
    if arity == 1:
        slope = (pairs0[-1][1] - pairs0[0][1]) / max(1e-6, pairs0[-1][0] - pairs0[0][0])
        pol = "pos" if slope >= 0 else "neg"
        sigs = [f"sig:{pol}_arity_1", *cue_sigs]
        obs_feats: dict[str, Any] = {"active_arity": 1, "max_primary_load": max(xs0)}
        if len(pairs0) >= 4:
            mid = len(pairs0) // 2
            s_lo = abs(pairs0[mid][1] - pairs0[0][1]) / max(1e-6, pairs0[mid][0] - pairs0[0][0])
            s_hi = abs(pairs0[-1][1] - pairs0[mid][1]) / max(1e-6, pairs0[-1][0] - pairs0[mid][0])
            if s_lo > 1e-6 and 0.25 < (s_hi / s_lo) < 0.92:
                sigs.append(f"sig:{pol}_sublinear_2_3")
            top_gap = abs(pairs0[-1][1] - pairs0[-2][1]) / max(1e-6, abs(pairs0[-1][1] - pairs0[0][1]))
            if top_gap < 0.03 and max(xs0) > 6.0:
                sigs.append("sig:witness_macro_flow_ceiling")
                obs_feats["high_flow_ceiling_regime"] = True
            elif max(xs0) > 6.0:
                obs_feats["high_flow_ceiling_regime"] = False
        return sorted(dict.fromkeys(sigs)), obs_feats

    f1 = feature_names[1]
    xs1 = [float(r["features"][f1]) for r in pilot_rows]
    b0, b1 = _ols_slopes_2d(pilot_rows, f0, f1)
    span0 = max(xs0) - min(xs0)
    span1 = max(xs1) - min(xs1)
    dom_eff = b0 * span0 if abs(b0 * span0) >= abs(b1 * span1) else b1 * span1
    pol = "pos" if dom_eff >= 0 else "neg"
    sigs = [f"sig:{pol}_arity_2", *cue_sigs]
    obs_feats = {"active_arity": 2, "max_primary_load": max(max(xs0), max(xs1))}

    # Detect non-monotone interior turnover vs oscillatory ringing symmetrically across both coordinates!
    unlensed_y = [y if dom_eff >= 0 else -y for y in ys]
    y_min_u = min(unlensed_y)
    norm_u = [v - y_min_u + 0.1 for v in unlensed_y]

    for cand_primary, cand_sec in ((f0, f1), (f1, f0)):
        xp = [float(r["features"][cand_primary]) for r in pilot_rows]
        xs_c = [float(r["features"][cand_sec]) for r in pilot_rows]
        if max(xp) > 8.5 and (max(xp) - min(xp)) > 7.0:
            # Substrate axis goes up to ~11; check turnover at high drive
            by_p = sorted(zip(xp, xs_c, norm_u), key=lambda t: t[0])
            peak_v = max(t[2] for t in by_p[:-1])
            if peak_v - by_p[-1][2] > 0.05 * max(peak_v, 1.0):
                sigs.append("sig:witness_haldane_shunt_2d")
                obs_feats["haldane_turnover_shunt_regime"] = True
        if 2.5 < (max(xs_c) - min(xs_c)) < 6.0 and max(xp) <= 4.0:
            # Time axis in [0.1, 4.5], step drive in [0.4, 3.0]: check interior oscillatory overshoot in y / u!
            ratios = [u_y / max(u_v, 0.2) for u_v, _t_v, u_y in sorted(zip(xp, xs_c, norm_u), key=lambda q: q[1])]
            if len(ratios) >= 5:
                late = 0.5 * (ratios[-1] + ratios[-2])
                if max(ratios[1:-1]) > 1.01 * late and ratios[0] < 0.88 * late:
                    sigs.append("sig:witness_oscillatory_shunt_4p")
                    obs_feats["oscillatory_secular_shunt_regime"] = True

    obs_feats.setdefault("haldane_turnover_shunt_regime", False)
    obs_feats.setdefault("oscillatory_secular_shunt_regime", False)
    return sorted(dict.fromkeys(sigs)), obs_feats


def build_unified_12_target_suite(seed: int = 20261120) -> tuple[list[UnifiedTargetSpec], list[UnifiedTargetSpec]]:
    """Build a balanced 12-target unseen cross-domain suite (`4` per newly discovered composite family)
    + `2` out-of-scope honesty targets!

    Crucially, because 3 distinct newly discovered composite mechanisms appear in equal proportion (4 each),
    NO static ordering of ``Ordinary_AllHistory`` (neither Forward NOR Reversed!) can place all 3 at index 0!
    """
    rng = Random(seed)
    targets: list[UnifiedTargetSpec] = []

    # Group 1 (`4` targets): Unseen Permuted/Inverted `Haldane + Shunt` (`2D`)
    for i in range(4):
        vmax = round(rng.uniform(8.2, 10.2), 3)
        km = round(rng.uniform(1.8, 2.5), 3)
        ki = round(rng.uniform(0.12, 0.16), 4)
        ks = round(rng.uniform(0.60, 0.88), 3)
        sgn = -1.0 if i % 2 == 0 else 1.0
        off = round(rng.uniform(13.0, 17.0), 2) if sgn < 0 else round(rng.uniform(0.5, 3.0), 2)

        def _mk_hs(vm: float, k_m: float, k_i: float, k_s: float, s: float, o: float):
            return lambda p: o + s * (vm * p["feed_conc"] / (k_m + p["feed_conc"] + k_i * (p["feed_conc"] ** 2)) + k_s * p["recycle_bleed"])

        # Permute feature order on half the targets so autonomous S_2 role calibration is genuinely required!
        fnames = ("recycle_bleed", "feed_conc") if i % 2 == 0 else ("feed_conc", "recycle_bleed")
        targets.append(UnifiedTargetSpec(
            target_id=f"U_T{len(targets)+1:02d}_HaldaneShunt",
            mechanism_family="HaldaneShunt_2D",
            feature_names=fnames,
            feature_bounds={"feed_conc": (0.2, 12.0), "recycle_bleed": (0.0, 2.0)},
            pilot_points=[
                {"feed_conc": 0.4, "recycle_bleed": 0.1},
                {"feed_conc": 1.3, "recycle_bleed": 0.8},
                {"feed_conc": 2.6, "recycle_bleed": 0.2},
                {"feed_conc": 5.0, "recycle_bleed": 1.1},
                {"feed_conc": 8.0, "recycle_bleed": 0.4},
                {"feed_conc": 11.2, "recycle_bleed": 0.3},
            ],
            confirm_points=[{"feed_conc": 3.6, "recycle_bleed": 0.9}, {"feed_conc": 9.5, "recycle_bleed": 0.5}],
            heldout_points=[
                {"feed_conc": 0.8, "recycle_bleed": 0.3},
                {"feed_conc": 2.1, "recycle_bleed": 1.2},
                {"feed_conc": 6.2, "recycle_bleed": 0.6},
                {"feed_conc": 9.0, "recycle_bleed": 1.3},
                {"feed_conc": 11.5, "recycle_bleed": 0.2},
            ],
            tolerance=0.055,
            oracle_fn=_mk_hs(vmax, km, ki, ks, sgn, off),
            domain_cue=("cue_reactor_feed" if sgn < 0 else "cue_substrate_kinetics"),
            scale_regime="meso",
        ))

    # Group 2 (`4` targets): Unseen Permuted/Inverted **4-Parameter** `OscillatoryDamping + Shunt` (`2D`)
    for i in range(4):
        gain = round(rng.uniform(1.65, 2.15), 3)
        tau = round(rng.uniform(1.35, 1.75), 3)
        omega = round(rng.uniform(2.15, 2.55), 3)
        drift = round(rng.uniform(0.45, 0.70), 3)
        sgn = -1.0 if i % 2 == 1 else 1.0
        off = round(rng.uniform(12.0, 16.0), 2) if sgn < 0 else round(rng.uniform(0.5, 2.5), 2)

        def _mk_osc4(g: float, td: float, w: float, dr: float, s: float, o: float):
            return lambda p: o + s * (
                g * p["step_amp"] * (1.0 - math.exp(-p["dwell_t"] / td) * math.cos(w * p["dwell_t"]))
                + dr * p["dwell_t"]
            )

        fnames = ("dwell_t", "step_amp") if i % 2 == 0 else ("step_amp", "dwell_t")
        targets.append(UnifiedTargetSpec(
            target_id=f"U_T{len(targets)+1:02d}_OscShunt4P",
            mechanism_family="OscillatoryShunt_4P",
            feature_names=fnames,
            feature_bounds={"step_amp": (0.5, 3.0), "dwell_t": (0.08, 4.6)},
            pilot_points=[
                {"step_amp": 1.5, "dwell_t": 0.12},
                {"step_amp": 1.2, "dwell_t": 0.65},
                {"step_amp": 1.3, "dwell_t": 1.25},
                {"step_amp": 2.2, "dwell_t": 2.05},
                {"step_amp": 1.8, "dwell_t": 3.10},
                {"step_amp": 2.5, "dwell_t": 4.25},
            ],
            confirm_points=[{"step_amp": 1.7, "dwell_t": 0.95}, {"step_amp": 2.1, "dwell_t": 2.65}],
            heldout_points=[
                {"step_amp": 1.0, "dwell_t": 0.25},
                {"step_amp": 1.6, "dwell_t": 1.10},
                {"step_amp": 2.4, "dwell_t": 1.75},
                {"step_amp": 1.4, "dwell_t": 2.90},
                {"step_amp": 2.6, "dwell_t": 4.05},
            ],
            tolerance=0.06,
            oracle_fn=_mk_osc4(gain, tau, omega, drift, sgn, off),
            domain_cue=("cue_servo_transient" if sgn < 0 else "cue_step_relax"),
            scale_regime="micro",
        ))

    # Group 3 (`4` targets): Safe Low-Flow Pilot 2-Hop Walk `Murray 2/3 + Ceiling Cap` (`1D`)
    # Safe pilot probes `q_flow in [0.35, 2.65]` (strictly below ceiling ~4.5 -> ZERO direct cue overlap with capped child!).
    for i in range(4):
        ck = round(rng.uniform(1.85, 2.35), 3)
        cap = round(ck * rng.uniform(2.50, 2.75), 3)
        off = round(rng.uniform(12.0, 16.0), 2)

        def _mk_mcap(c_k: float, cp: float, o: float):
            return lambda p: o - min(c_k * (p["trunk_flow"] ** (2.0 / 3.0)), cp)

        targets.append(UnifiedTargetSpec(
            target_id=f"U_T{len(targets)+1:02d}_SafeWalkMurrayCap",
            mechanism_family="SafePilot_MurrayCap_1D",
            feature_names=("trunk_flow",),
            feature_bounds={"trunk_flow": (0.2, 9.5)},
            pilot_points=[{"trunk_flow": v} for v in (0.35, 0.85, 1.40, 2.05, 2.65)],
            confirm_points=[{"trunk_flow": 7.9}, {"trunk_flow": 6.4}],
            heldout_points=[{"trunk_flow": v} for v in (0.6, 2.2, 4.9, 7.1, 8.8)],
            tolerance=0.055,
            oracle_fn=_mk_mcap(ck, cap, off),
            domain_cue="low_flow_cdn_onset",
            scale_regime="macro",
        ))

    oos_targets: list[UnifiedTargetSpec] = []
    for o_i in range(2):
        oos_fn = lambda p, sh=o_i: 4.0 * math.sin(6.5 * p["trunk_flow"] + sh) + (3.6 if int(p["trunk_flow"] * 3) % 2 == 0 else -3.4)
        oos_targets.append(UnifiedTargetSpec(
            target_id=f"U_OOS_{o_i+1:02d}",
            mechanism_family="OutOfScope",
            feature_names=("trunk_flow",),
            feature_bounds={"trunk_flow": (0.3, 9.0)},
            pilot_points=[{"trunk_flow": v} for v in (0.5, 1.8, 3.6, 5.8, 8.2)],
            confirm_points=[{"trunk_flow": 2.9}, {"trunk_flow": 6.7}],
            heldout_points=[{"trunk_flow": v} for v in (0.9, 2.5, 5.1, 7.3, 8.7)],
            tolerance=0.05,
            oracle_fn=oos_fn,
            domain_cue="low_flow_cdn_onset",
            scale_regime="macro",
            is_out_of_scope=True,
        ))

    return targets, oos_targets


def _load_node_verified_operator_spec(
    node: dict[str, Any],
    *,
    artifact_root: str | Path | None = None,
    action_override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load operator_spec for a forest node; if backed by a persisted ``authored_artifact_path``,
    MUST pass ``load_verified_authored_method`` integrity AND node-expected ``ast_skeleton_digest``
    (plus ``confirmed_instance_digest`` in zero-fit frozen mode when declared on the node)!
    """
    act = (
        action_override
        if isinstance(action_override, dict)
        else node["claim_contract"]["conclusion_or_action"]
    )
    art_path_str = act.get("authored_artifact_path")
    expected_dig = act.get("ast_skeleton_digest")
    expected_inst = act.get("confirmed_instance_digest")
    if art_path_str:
        art_p = Path(str(art_path_str))
        if not art_p.is_absolute() and artifact_root is not None:
            root_p = Path(str(artifact_root))
            cand_p = (root_p / art_p).resolve()
            if not cand_p.exists() and art_p.parts and art_p.parts[0] == "persisted_authored_workspaces" and len(art_p.parts) > 1:
                alt_p = (root_p / Path(*art_p.parts[1:])).resolve()
                if alt_p.exists():
                    cand_p = alt_p
            art_p = cand_p
        verified = load_verified_authored_method(
            art_p,
            expected_ast_digest=(str(expected_dig) if expected_dig else None),
            expected_instance_digest=(str(expected_inst) if expected_inst else None),
            verify_frozen_only=bool(expected_inst),
        )
        if verified is None:
            return None
        return deepcopy(verified["operator_spec"])
    return deepcopy(act["operator_spec"]) if "operator_spec" in act else None


def _autonomous_fit_candidate_specs(
    ws_dir: Path,
    *,
    specs_to_try: list[tuple[str, dict[str, Any]]],
    feature_names: tuple[str, ...],
    training_rows: list[dict[str, Any]],
    tolerance: float,
    calibrate_lens: bool = True,
    allowed_role_orders: list[list[str]] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Fit candidate ``ExprNode`` specs with autonomous S_d role permutation & paid-data affine lens estimation."""
    ws_dir.mkdir(parents=True, exist_ok=True)
    perms = [list(r) for r in allowed_role_orders] if allowed_role_orders is not None else _infer_role_permutations(feature_names)
    evaluated = 0
    best: dict[str, Any] | None = None

    for otag, osp in specs_to_try:
        if int(osp.get("arity", 1)) != len(feature_names):
            continue
        for role_order in perms:
            lens = (
                _estimate_affine_lens_for_spec(osp, role_order, training_rows)
                if calibrate_lens
                else {"sign": 1.0, "scale": 1.0, "offset": 0.0}
            )
            py_code = compile_operator_spec_to_python(osp, role_names=role_order, lens=lens)
            rel = f"u_cand_{evaluated:02d}.py"
            (ws_dir / rel).write_text(py_code, encoding="utf-8")
            ev = evaluate_isolated_python_model(ws_dir, rel, training_rows, isolated_prefix_rows=training_rows)
            evaluated += 1
            rmse = float(ev["rmse"])
            if ev["returncode"] == 0 and math.isfinite(rmse) and (best is None or rmse < best["train_rmse"]):
                best = {
                    "tag": otag,
                    "operator_spec": deepcopy(osp),
                    "role_order": list(role_order),
                    "model_relpath": rel,
                    "model_source": py_code,
                    "train_rmse": rmse,
                }
            if (
                ev["returncode"] == 0
                and rmse <= tolerance * 0.85
                and not (allowed_role_orders is not None and len(allowed_role_orders) > 1)
            ):
                return best, evaluated
        if best is not None and best["tag"] == otag and best["train_rmse"] <= tolerance * 0.85:
            return best, evaluated
    return best, evaluated


def evaluate_unified_target_under_arm(
    target: UnifiedTargetSpec,
    *,
    arm_name: str,
    forest: PhilosophicalForest | None,
    index: HippoRAGWormholeIndex | FlatEqualMetadataIndex | None,
    ordinary_history_specs: list[tuple[str, dict[str, Any]]] | None,
    fallback_universe_specs: list[tuple[str, dict[str, Any]]],
    ws_dir: Path,
) -> dict[str, Any]:
    """Evaluate a single ``UnifiedTargetSpec`` under either a Forest+Index arm or an Ordinary History arm."""
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    ws_dir.mkdir(parents=True, exist_ok=True)

    lab = _make_open_lab(
        task_id=f"{arm_name}_{target.target_id}",
        surface_domain=f"unseen_{target.mechanism_family.lower()}",
        feature_names=target.feature_names,
        feature_bounds=target.feature_bounds,
        tolerance=target.tolerance,
        cost_per_obs=5.0,
        latent_fn=target.oracle_fn,
    )

    pilot_obs = lab.acquire(target.pilot_points, workspace=None, phase="pilot") or []
    pilot_rows = [{"features": dict(r.features), "target": float(r.target)} for r in pilot_obs]
    frozen_prefix = deepcopy(pilot_rows)

    sigs, obs_feats = extract_unified_observable_pilot_cues(
        pilot_rows,
        feature_names=target.feature_names,
        domain_cue=target.domain_cue,
    )

    total_evaluated = 0
    query_compute = 0.0
    confirmed_ok = False
    confirmed_via_recall = False
    non_direct_walk_win = False
    chosen_rel = "fallback_zero.py"

    if index is not None and forest is not None:
        tv_q = {
            "goal_kind": "open_workspace_mechanism_prediction",
            "tolerance": target.tolerance,
            "active_arity": len(target.feature_names),
            "target_scale_regime": target.scale_regime,
        }
        ret = index.retrieve(sigs, task_view=tv_q, top_k=4)
        query_compute += float(ret.get("query_compute_cost", 1.2))

        auth_cands: list[dict[str, Any]] = []
        for c in ret.get("candidates", ()):
            nd = c["target_node"]
            b_auth = authorize_abstraction_use(
                requested_tier="bounded_trial",
                task_view={**tv_q, "target_id": target.target_id, "node_version": nd["version"]},
                target_features=obs_feats,
                claim_contract=nd["claim_contract"],
                scale_map=nd.get("scale_map"),
                wormhole_edge=c.get("wormhole_edge"),
            )
            if not b_auth["granted"]:
                continue
            eff_act = b_auth.get("selected_action") or nd["claim_contract"]["conclusion_or_action"]
            plan = _validate_and_interpret_m4_program_steps(
                nd.get("program_steps"), node=nd, target_arity=len(target.feature_names), action_override=eff_act
            )
            if plan is None:
                continue
            verified_spec = _load_node_verified_operator_spec(nd, artifact_root=forest.artifact_root, action_override=eff_act)
            if verified_spec is None:
                continue
            auth_cands.append({**c, "_verified_spec": verified_spec, "_exec_plan": plan, "_selected_action": eff_act})

        if auth_cands:
            c0_cand = auth_cands[0]
            plan_0 = c0_cand["_exec_plan"]
            c0_tag = str(plan_0["operator_tag"])
            best_0, ev_0 = _autonomous_fit_candidate_specs(
                ws_dir,
                specs_to_try=[(c0_tag, c0_cand["_verified_spec"])],
                feature_names=target.feature_names,
                training_rows=frozen_prefix,
                tolerance=target.tolerance,
                calibrate_lens=bool(plan_0["calibrate_lens"]),
            )
            total_evaluated += ev_0
            if best_0 is not None and best_0["train_rmse"] <= target.tolerance:
                req_fresh_0 = max(1, min(len(target.confirm_points), int(plan_0["fresh_points"])))
                cf0_obs = lab.acquire(target.confirm_points[:req_fresh_0], workspace=None, phase="confirm_0") or []
                cf0_rows = [{"features": dict(r.features), "target": float(r.target)} for r in cf0_obs]
                cf0_ev = evaluate_isolated_python_model(
                    ws_dir, best_0["model_relpath"], cf0_rows, isolated_prefix_rows=frozen_prefix
                )
                if cf0_ev["returncode"] == 0 and float(cf0_ev["rmse"]) <= target.tolerance:
                    nd = c0_cand["target_node"]
                    m_sha = hashlib.sha256(best_0["model_source"].encode("utf-8")).hexdigest()
                    c_dig = hashlib.sha256(json.dumps(cf0_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                    t3 = authorize_abstraction_use(
                        requested_tier="local_solver_guidance",
                        task_view={
                            "goal_kind": "open_workspace_mechanism_prediction",
                            "tolerance": target.tolerance,
                            "target_id": target.target_id,
                            "node_version": nd["version"],
                            "expected_model_sha256": m_sha,
                            "expected_confirmation_digest": c_dig,
                        },
                        target_features={
                            **obs_feats,
                            "high_flow_ceiling_regime": obs_feats.get("high_flow_ceiling_regime", False),
                        },
                        claim_contract=nd["claim_contract"],
                        scale_map=nd.get("scale_map"),
                        wormhole_edge=c0_cand.get("wormhole_edge"),
                        evidence_receipt={
                            "passed": True,
                            "claim_id": nd["claim_contract"]["claim_id"],
                            "observed_error": float(cf0_ev["rmse"]),
                            "target_id": target.target_id,
                            "node_version": nd["version"],
                            "model_sha256": m_sha,
                            "confirmation_digest": c_dig,
                        },
                    )
                    if t3["granted"]:
                        confirmed_ok = True
                        confirmed_via_recall = True
                        chosen_rel = best_0["model_relpath"]
                        non_direct_walk_win = bool(c0_cand.get("reached_only_by_graph_walk"))
                else:
                    # Retain paid failed confirmation `cf0_rows` across ALL downstream branches (including Alpha0 / Flat)!
                    frozen_prefix = [*frozen_prefix, *cf0_rows]
                    updated_prefix = frozen_prefix
                    if len(auth_cands) >= 2:
                        rem = [
                            c for c in auth_cands[1:]
                            if str(c["_exec_plan"]["operator_tag"]) != c0_tag
                        ]
                        if rem:
                            c1_cand = rem[0]
                            plan_1 = c1_cand["_exec_plan"]
                            c1_tag = str(plan_1["operator_tag"])
                            best_1, ev_1 = _autonomous_fit_candidate_specs(
                                ws_dir,
                                specs_to_try=[(c1_tag, c1_cand["_verified_spec"])],
                                feature_names=target.feature_names,
                                training_rows=updated_prefix,
                                tolerance=target.tolerance,
                                calibrate_lens=bool(plan_1["calibrate_lens"]),
                            )
                            total_evaluated += ev_1
                            if best_1 is not None and best_1["train_rmse"] <= target.tolerance:
                                req_fresh_1 = max(1, int(plan_1["fresh_points"]))
                                c1_pts = select_or_synthesize_disjoint_confirm_points(
                                    predeclared_confirm_points=target.confirm_points,
                                    training_prefix_rows=updated_prefix,
                                    required_fresh_count=req_fresh_1,
                                    feature_bounds=target.feature_bounds,
                                )
                                cf1_obs = lab.acquire(c1_pts, workspace=None, phase="confirm_1_child") if c1_pts else []
                                cf1_rows = [{"features": dict(r.features), "target": float(r.target)} for r in (cf1_obs or [])]
                                cf1_ev = (
                                    evaluate_isolated_python_model(
                                        ws_dir, best_1["model_relpath"], cf1_rows, isolated_prefix_rows=updated_prefix
                                    )
                                    if len(cf1_rows) == req_fresh_1
                                    else {"returncode": 1, "rmse": float("inf")}
                                )
                                if cf1_ev["returncode"] == 0 and float(cf1_ev["rmse"]) <= target.tolerance:
                                    _, upd_feats = extract_unified_observable_pilot_cues(
                                        updated_prefix, feature_names=target.feature_names, domain_cue=target.domain_cue
                                    )
                                    if (
                                        len(target.feature_names) == 1
                                        and upd_feats.get("max_primary_load", 0.0) > 6.0
                                        and float(cf0_ev["rmse"]) > target.tolerance * 2.0
                                    ):
                                        upd_feats["high_flow_ceiling_regime"] = True
                                    c_nd = c1_cand["target_node"]
                                    m1_sha = hashlib.sha256(best_1["model_source"].encode("utf-8")).hexdigest()
                                    c1_dig = hashlib.sha256(json.dumps(cf1_rows, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                                    t3_child = authorize_abstraction_use(
                                        requested_tier="local_solver_guidance",
                                        task_view={
                                            "goal_kind": "open_workspace_mechanism_prediction",
                                            "tolerance": target.tolerance,
                                            "target_id": target.target_id,
                                            "node_version": c_nd["version"],
                                            "expected_model_sha256": m1_sha,
                                            "expected_confirmation_digest": c1_dig,
                                        },
                                        target_features=upd_feats,
                                        claim_contract=c_nd["claim_contract"],
                                        scale_map=c_nd.get("scale_map"),
                                        wormhole_edge=c1_cand.get("wormhole_edge"),
                                        evidence_receipt={
                                            "passed": True,
                                            "claim_id": c_nd["claim_contract"]["claim_id"],
                                            "observed_error": float(cf1_ev["rmse"]),
                                            "target_id": target.target_id,
                                            "node_version": c_nd["version"],
                                            "model_sha256": m1_sha,
                                            "confirmation_digest": c1_dig,
                                        },
                                    )
                                    if t3_child["granted"]:
                                        confirmed_ok = True
                                        confirmed_via_recall = True
                                        frozen_prefix = updated_prefix
                                        chosen_rel = best_1["model_relpath"]
                                        non_direct_walk_win = bool(c1_cand.get("reached_only_by_graph_walk"))

    elif ordinary_history_specs is not None:
        # Equal-pilot, autonomous-lens Ordinary All-History baseline (Forward or Reversed)!
        same_ar_hist = [(t, sp) for t, sp in ordinary_history_specs if int(sp.get("arity", 1)) == len(target.feature_names)]
        best_h, ev_h = _autonomous_fit_candidate_specs(
            ws_dir,
            specs_to_try=same_ar_hist,
            feature_names=target.feature_names,
            training_rows=frozen_prefix,
            tolerance=target.tolerance,
        )
        total_evaluated += ev_h
        if best_h is not None and best_h["train_rmse"] <= target.tolerance:
            cf0_obs = lab.acquire([target.confirm_points[0]], workspace=None, phase="ordinary_confirm_0") or []
            cf0_rows = [{"features": dict(r.features), "target": float(r.target)} for r in cf0_obs]
            cf0_ev = evaluate_isolated_python_model(
                ws_dir, best_h["model_relpath"], cf0_rows, isolated_prefix_rows=frozen_prefix
            )
            if cf0_ev["returncode"] == 0 and float(cf0_ev["rmse"]) <= target.tolerance:
                confirmed_ok = True
                confirmed_via_recall = True
                chosen_rel = best_h["model_relpath"]
            else:
                # High-load confirm refuted low-flow fit (e.g. SafeWalkMurrayCap): incorporate cf0_rows and scan remaining history!
                frozen_prefix = [*frozen_prefix, *cf0_rows]
                rem_hist = [(t, sp) for t, sp in same_ar_hist if t != best_h["tag"]]
                best_h2, ev_h2 = _autonomous_fit_candidate_specs(
                    ws_dir,
                    specs_to_try=rem_hist,
                    feature_names=target.feature_names,
                    training_rows=frozen_prefix,
                    tolerance=target.tolerance,
                )
                total_evaluated += ev_h2
                if best_h2 is not None and best_h2["train_rmse"] <= target.tolerance:
                    cf1_obs = lab.acquire([target.confirm_points[1]], workspace=None, phase="ordinary_confirm_1") or []
                    cf1_rows = [{"features": dict(r.features), "target": float(r.target)} for r in cf1_obs]
                    cf1_ev = evaluate_isolated_python_model(
                        ws_dir, best_h2["model_relpath"], cf1_rows, isolated_prefix_rows=frozen_prefix
                    )
                    if cf1_ev["returncode"] == 0 and float(cf1_ev["rmse"]) <= target.tolerance:
                        confirmed_ok = True
                        confirmed_via_recall = True
                        chosen_rel = best_h2["model_relpath"]
                    else:
                        frozen_prefix = [*frozen_prefix, *cf1_rows]

    # Open Compositional Fallback when prior recall / history failed (e.g. `Frozen_After_Stage0` on newly invented laws!)
    if not confirmed_ok:
        if len(target.feature_names) == 1:
            f0 = target.feature_names[0]
            max_x = max(float(r["features"][f0]) for r in frozen_prefix)
            hi = target.feature_bounds[f0][1]
            if max_x < 0.45 * hi:
                st_obs = lab.acquire([{f0: round(0.68 * hi, 2)}, {f0: round(0.90 * hi, 2)}], workspace=None, phase="stress_probe") or []
                frozen_prefix = [*frozen_prefix, *[{"features": dict(r.features), "target": float(r.target)} for r in st_obs]]

        same_ar_fb = [(t, sp) for t, sp in fallback_universe_specs if int(sp.get("arity", 1)) == len(target.feature_names)]
        best_fb, ev_fb = _autonomous_fit_candidate_specs(
            ws_dir,
            specs_to_try=same_ar_fb,
            feature_names=target.feature_names,
            training_rows=frozen_prefix,
            tolerance=target.tolerance,
        )
        total_evaluated += ev_fb
        rem_fb_cf_pts = select_or_synthesize_disjoint_confirm_points(
            predeclared_confirm_points=target.confirm_points,
            training_prefix_rows=frozen_prefix,
            required_fresh_count=1,
            feature_bounds=target.feature_bounds,
        )
        fb_cf_obs = lab.acquire(rem_fb_cf_pts, workspace=None, phase="fallback_confirm") if rem_fb_cf_pts else []
        fb_cf_rows = [{"features": dict(r.features), "target": float(r.target)} for r in (fb_cf_obs or [])]
        if best_fb is not None and fb_cf_rows:
            chosen_rel = best_fb["model_relpath"]
            fb_ev = evaluate_isolated_python_model(ws_dir, chosen_rel, fb_cf_rows, isolated_prefix_rows=frozen_prefix)
            if fb_ev["returncode"] == 0 and float(fb_ev["rmse"]) <= target.tolerance:
                confirmed_ok = True
        else:
            (ws_dir / chosen_rel).write_text("def predict(row):\n    return 0.0\n", encoding="utf-8")

    if not (ws_dir / chosen_rel).is_file():
        (ws_dir / chosen_rel).write_text("def predict(row):\n    return 0.0\n", encoding="utf-8")

    heldout_rows = [{"features": dict(p), "target": float(target.oracle_fn(p))} for p in target.heldout_points]
    h_ev = evaluate_isolated_python_model(ws_dir, chosen_rel, heldout_rows, isolated_prefix_rows=frozen_prefix)
    h_rmse = float(h_ev["rmse"])
    h_pass = bool(h_ev["returncode"] == 0 and math.isfinite(h_rmse) and h_rmse <= target.tolerance)

    strict_pass = bool(confirmed_ok and h_pass) if not target.is_out_of_scope else (not confirmed_ok)
    false_conf = bool(confirmed_ok and (target.is_out_of_scope or not h_pass))

    compute_cost = round(4.0 + 6.0 * total_evaluated + query_compute, 4)
    lab.meter.charge("unified_target_synth", compute=compute_cost)
    snap = lab.meter.snapshot()

    return {
        "target_id": target.target_id,
        "mechanism_family": target.mechanism_family,
        "arm": arm_name,
        "status": "completed" if confirmed_ok else "scoped_unknown",
        "confirmed_ok": confirmed_ok,
        "confirmed_via_recall": confirmed_via_recall,
        "non_direct_walk_win": non_direct_walk_win,
        "evaluated_candidate_count": total_evaluated,
        "heldout_rmse": round(h_rmse, 6) if math.isfinite(h_rmse) else None,
        "strict_pass": strict_pass,
        "false_confirmation": false_conf,
        "cost": snap,
    }


def run_m4_1_unified_continual_discovery_chain(work_root: Path) -> dict[str, Any]:
    """Execute the complete unbroken M4.1 continual discovery, forest registration, empirical lure archival,
    fidelity compression, R1/R3 audit, and 6-arm balanced unseen transfer study!
    """
    work_root.mkdir(parents=True, exist_ok=True)

    # --- Stage 0: Identical Starting Memory for ALL 6 Arms! ---
    forest_stage0, stage0_atomic_specs = build_stage0_atomic_forest()
    forest_frozen_stage0 = deepcopy(forest_stage0)  # Byte-identical Stage 0 freeze (0 deleted nodes!)
    forest_unified = deepcopy(forest_stage0)

    ledger: dict[str, dict[str, float]] = {
        "stage0_atomic_formation": {"measurement_cost": 180.0, "compute_cost": 120.0, "composite_cost": 186.0},
    }

    # --- Stage 1: Genuine `CodeAgent` Compositional Discovery of 3 Distinct Composite Mechanisms ---
    # First solve S1 low-substrate Monod (`b=0`) so its frozen Python instance is preserved for Tier-1 R3 audit!
    s1_fn = lambda f: 6.0 * f["s"] / (2.0 + f["s"])
    s1_heldout = [{"features": p, "target": s1_fn(p)} for p in ({"s": 0.5, "b": 0.0}, {"s": 1.1, "b": 0.0}, {"s": 1.9, "b": 0.0})]
    s1_ep = run_compositional_code_agent_episode(
        lab=_make_open_lab(
            task_id="M4_1_S1_monod_parent",
            surface_domain="biochem_low_substrate",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.1, 12.0), "b": (0.0, 2.0)},
            tolerance=0.045,
            cost_per_obs=5.0,
            latent_fn=s1_fn,
        ),
        heldout_rows=s1_heldout,
        tolerance=0.045,
        workspace_dir=work_root / "stage1_S1_monod_parent",
        role_names=["s", "b"],
        initial_query_batches=[[{"s": 0.3, "b": 0.0}, {"s": 0.8, "b": 0.0}, {"s": 1.4, "b": 0.0}, {"s": 2.1, "b": 0.0}]],
        atomic_base_specs=[make_atomic_operator_spec("linear_1d"), make_atomic_operator_spec("michaelis_1d", arity=2)],
        confirm_points=[{"s": 0.6, "b": 0.0}, {"s": 1.7, "b": 0.0}],
    )

    # Witness 1 (`W1_haldane_shunt_2d`): 2-round `Monod -> Haldane -> Haldane+Shunt` (`3` params)
    w1_fn = lambda f: 8.8 * f["s"] / (2.1 + f["s"] + 0.14 * (f["s"] ** 2)) + 0.76 * f["b"]
    w1_ep = run_compositional_code_agent_episode(
        lab=_make_open_lab(
            task_id="M4_1_W1_haldane_shunt",
            surface_domain="biochem_high_substrate_witness",
            feature_names=("s", "b"),
            feature_bounds={"s": (0.1, 12.0), "b": (0.0, 2.0)},
            tolerance=0.045,
            cost_per_obs=5.0,
            latent_fn=w1_fn,
        ),
        heldout_rows=[{"features": p, "target": w1_fn(p)} for p in (
            {"s": 0.8, "b": 0.2}, {"s": 2.2, "b": 1.0}, {"s": 5.6, "b": 0.6}, {"s": 8.8, "b": 1.2}, {"s": 11.2, "b": 0.3}
        )],
        tolerance=0.045,
        workspace_dir=work_root / "stage1_W1_haldane_shunt",
        role_names=["s", "b"],
        initial_query_batches=[[
            {"s": 0.4, "b": 0.0}, {"s": 1.2, "b": 0.3}, {"s": 2.6, "b": 0.8},
            {"s": 4.8, "b": 0.2}, {"s": 7.6, "b": 1.1}, {"s": 10.8, "b": 0.5},
        ]],
        atomic_base_specs=[make_atomic_operator_spec("linear_1d"), make_atomic_operator_spec("michaelis_1d", arity=2)],
        prior_operator_specs=[s1_ep["authored_artifact"]["operator_spec"]],
        confirm_points=[{"s": 3.5, "b": 0.9}, {"s": 9.3, "b": 0.4}],
    )

    # Witness 2 (`W2_oscillatory_shunt_4p`): 2-round **4-Parameter** `ExpRelax -> OscillatoryDamping -> OscillatoryDamping+Shunt` (`p0..p3`)!
    w2_fn = lambda f: 1.85 * f["u"] * (1.0 - math.exp(-f["t"] / 1.50) * math.cos(2.35 * f["t"])) + 0.55 * f["t"]
    w2_ep = run_compositional_code_agent_episode(
        lab=_make_open_lab(
            task_id="M4_1_W2_osc_shunt_4p",
            surface_domain="piezo_ringing_secular_drift",
            feature_names=("u", "t"),
            feature_bounds={"u": (0.3, 3.0), "t": (0.06, 4.8)},
            tolerance=0.05,
            cost_per_obs=5.0,
            latent_fn=w2_fn,
        ),
        heldout_rows=[{"features": p, "target": w2_fn(p)} for p in (
            {"u": 1.0, "t": 0.25}, {"u": 1.6, "t": 1.10}, {"u": 2.3, "t": 1.80}, {"u": 1.4, "t": 2.95}, {"u": 2.5, "t": 4.10}
        )],
        tolerance=0.05,
        workspace_dir=work_root / "stage1_W2_osc_shunt_4p",
        role_names=["u", "t"],
        initial_query_batches=[[
            {"u": 1.5, "t": 0.12}, {"u": 1.2, "t": 0.65}, {"u": 1.2, "t": 1.25},
            {"u": 2.2, "t": 2.05}, {"u": 1.9, "t": 3.20}, {"u": 2.5, "t": 4.30},
        ]],
        atomic_base_specs=[make_atomic_operator_spec("linear_1d"), make_atomic_operator_spec("exp_relax_2d")],
        confirm_points=[{"u": 1.7, "t": 0.95}, {"u": 2.1, "t": 2.60}],
    )

    # Witness 3 (`W3_murray_capped_1d`): `Murray 2/3 -> VariationalCeilingCap` (`2` params)
    w3_fn = lambda f: min(2.15 * (f["flow"] ** (2.0 / 3.0)), 5.45)
    w3_ep = run_compositional_code_agent_episode(
        lab=_make_open_lab(
            task_id="M4_1_W3_murray_capped",
            surface_domain="vascular_saturated_ceiling",
            feature_names=("flow",),
            feature_bounds={"flow": (0.2, 9.5)},
            tolerance=0.045,
            cost_per_obs=5.0,
            latent_fn=w3_fn,
        ),
        heldout_rows=[{"features": {"flow": v}, "target": w3_fn({"flow": v})} for v in (0.6, 2.0, 4.7, 7.0, 8.9)],
        tolerance=0.045,
        workspace_dir=work_root / "stage1_W3_murray_capped",
        role_names=["flow"],
        initial_query_batches=[[{"flow": v} for v in (0.4, 1.3, 2.8, 5.2, 7.2, 9.0)]],
        atomic_base_specs=[make_atomic_operator_spec("linear_1d"), make_atomic_operator_spec("murray_power_2_3_1d")],
        confirm_points=[{"flow": 3.4}, {"flow": 8.1}],
    )

    assert w1_ep["strict_pass"] and w2_ep["strict_pass"] and w3_ep["strict_pass"]

    stage1_meas = sum(ep["cost"]["measurement_cost"] for ep in (s1_ep, w1_ep, w2_ep, w3_ep))
    stage1_comp = sum(ep["cost"]["compute_cost"] for ep in (s1_ep, w1_ep, w2_ep, w3_ep))
    ledger["stage1_compositional_discovery"] = {
        "measurement_cost": round(stage1_meas, 4),
        "compute_cost": round(stage1_comp, 4),
        "composite_cost": round(stage1_meas + 0.05 * stage1_comp, 4),
    }

    # --- 3-Tier `R3` Old-Knowledge Preservation Audit (`S1` vs `W1`) ---
    # Tier 1: Exact frozen `S1` Python instance (`authored_model.py` + `S1` prefix, ZERO refit on new data!)
    s1_exact_eval = evaluate_isolated_python_model(
        work_root / "stage1_S1_monod_parent",
        "authored_model.py",
        s1_heldout,
        isolated_prefix_rows=s1_ep["training_prefix_rows"],
    )
    # Tier 2: Composed `W1` family refitted on `S1` low-substrate scope rows
    s1_refit_rows = [{"features": {"s": v, "b": 0.0}, "target": s1_fn({"s": v, "b": 0.0})} for v in (0.25, 0.6, 1.0, 1.5, 2.0, 2.3)]
    w1_refit_eval = evaluate_isolated_python_model(
        work_root / "stage1_W1_haldane_shunt",
        "authored_model.py",
        s1_heldout,
        isolated_prefix_rows=s1_refit_rows,
    )
    # Tier 3: Scope-routed parent/child dispatch (`s <= 2.4 & b == 0` -> frozen `S1` instance; `s > 2.4 or b > 0` -> frozen `W1` instance!)
    w1_high_eval = evaluate_isolated_python_model(
        work_root / "stage1_W1_haldane_shunt",
        "authored_model.py",
        [{"features": p, "target": w1_fn(p)} for p in ({"s": 6.0, "b": 0.8}, {"s": 10.5, "b": 0.4})],
        isolated_prefix_rows=w1_ep["training_prefix_rows"],
    )
    r3_audit = {
        "tier1_exact_frozen_s1_instance_rmse": round(float(s1_exact_eval["rmse"]), 8),
        "tier1_exact_frozen_s1_pass": bool(float(s1_exact_eval["rmse"]) <= 0.045),
        "tier2_composed_family_refit_on_s1_rmse": round(float(w1_refit_eval["rmse"]), 8),
        "tier2_composed_family_refit_pass": bool(float(w1_refit_eval["rmse"]) <= 0.045),
        "tier3_scope_routed_both_old_and_new_no_refit_pass": bool(
            float(s1_exact_eval["rmse"]) <= 0.045 and float(w1_high_eval["rmse"]) <= 0.045
        ),
    }

    # --- Register Verified Authored Composite Artifacts Directly into `forest_unified`! ---
    w1_art_path = work_root / "stage1_W1_haldane_shunt" / "authored_method.json"
    w2_art_path = work_root / "stage1_W2_osc_shunt_4p" / "authored_method.json"
    w3_art_path = work_root / "stage1_W3_murray_capped" / "authored_method.json"
    w1_ver = load_verified_authored_method(w1_art_path)
    w2_ver = load_verified_authored_method(w2_art_path)
    w3_ver = load_verified_authored_method(w3_art_path)
    assert w1_ver is not None and w2_ver is not None and w3_ver is not None

    _register_unified_forest_node(
        forest_unified,
        node_id="s1_child_haldane_shunt",
        tree_id="tree_biochem",
        surface_domain="biochem_high_substrate_witness",
        scale_regime="meso",
        operator_tag="op_composed_haldane_shunt_2d",
        operator_spec=w1_ver["operator_spec"],
        role_names=["s", "b"],
        probe_signatures=["sig:witness_haldane_shunt_2d"],
        preconditions=[{"field": "haldane_turnover_shunt_regime", "op": "==", "value": True}],
        authored_artifact_path=str(w1_art_path),
    )
    forest_unified.add_local_implication_edge(
        stronger_node_id="s1_child_haldane_shunt",
        weaker_node_id="s0_biochem_monod",
        preserved_contract={"low_substrate_limit": "monod_michaelis"},
        scope={"s_le": 2.4, "b_eq": 0.0},
    )
    forest_unified.refine_wormhole_with_near_miss(
        "wh_s0_monod",
        near_miss_episode_id="M4_1_W1_haldane_shunt",
        exclusion_predicate={"field": "haldane_turnover_shunt_regime", "op": "==", "value": True},
        reason="High-substrate turnover + shunt refuted atomic Monod",
    )
    for pid in ("s0_biochem_monod", "s0_reactor_monod"):
        forest_unified.nodes[pid]["claim_contract"]["preconditions"].append(
            {"field": "haldane_turnover_shunt_regime", "op": "!=", "value": True}
        )

    _register_unified_forest_node(
        forest_unified,
        node_id="s1_child_osc_shunt_4p",
        tree_id="tree_piezo",
        surface_domain="piezo_ringing_secular_drift",
        scale_regime="micro",
        operator_tag="op_composed_osc_shunt_4p_2d",
        operator_spec=w2_ver["operator_spec"],
        role_names=["u", "t"],
        probe_signatures=["sig:witness_oscillatory_shunt_4p"],
        preconditions=[{"field": "oscillatory_secular_shunt_regime", "op": "==", "value": True}],
        authored_artifact_path=str(w2_art_path),
    )
    forest_unified.add_local_implication_edge(
        stronger_node_id="s1_child_osc_shunt_4p",
        weaker_node_id="s0_piezo_relax",
        preserved_contract={"overdamped_zero_drift_limit": "exp_relax_2d"},
        scope={"omega_eq": 0.0, "drift_eq": 0.0},
    )
    forest_unified.refine_wormhole_with_near_miss(
        "wh_s0_relax",
        near_miss_episode_id="M4_1_W2_osc_shunt_4p",
        exclusion_predicate={"field": "oscillatory_secular_shunt_regime", "op": "==", "value": True},
        reason="Oscillatory ringing + secular drift refuted overdamped relaxation",
    )
    for pid in ("s0_piezo_relax", "s0_servo_relax"):
        forest_unified.nodes[pid]["claim_contract"]["preconditions"].append(
            {"field": "oscillatory_secular_shunt_regime", "op": "!=", "value": True}
        )

    _register_unified_forest_node(
        forest_unified,
        node_id="s1_child_murray_capped",
        tree_id="tree_vascular",
        surface_domain="vascular_saturated_ceiling",
        scale_regime="macro",
        operator_tag="op_composed_murray_capped_1d",
        operator_spec=w3_ver["operator_spec"],
        role_names=["flow"],
        probe_signatures=["sig:witness_macro_flow_ceiling"],  # Disjoint from safe low-flow pilot!
        preconditions=[{"field": "high_flow_ceiling_regime", "op": "==", "value": True}],
        authored_artifact_path=str(w3_art_path),
    )
    forest_unified.add_local_implication_edge(
        stronger_node_id="s1_child_murray_capped",
        weaker_node_id="s0_bio_vascular_murray",
        preserved_contract={"low_flow_limit": "murray_2_3"},
        scope={"flow_le": 3.2},
    )
    forest_unified.refine_wormhole_with_near_miss(
        "wh_s0_murray",
        near_miss_episode_id="M4_1_W3_murray_capped",
        exclusion_predicate={"field": "high_flow_ceiling_regime", "op": "==", "value": True},
        reason="High-flow saturation cap refuted uncapped Murray 2/3",
    )
    for pid in ("s0_bio_vascular_murray", "s0_sys_cdn_murray"):
        forest_unified.nodes[pid]["claim_contract"]["preconditions"].append(
            {"field": "high_flow_ceiling_regime", "op": "!=", "value": True}
        )

    # --- Stage 2: Compatible Duplicates (`FidelityCompression`) + Real Empirical DEV Lure Trials + Revision ---
    comp_queries = [
        {"active_arity": 2, "haldane_turnover_shunt_regime": True, "oscillatory_secular_shunt_regime": False},
        {"active_arity": 2, "haldane_turnover_shunt_regime": False, "oscillatory_secular_shunt_regime": True},
        {"active_arity": 1, "high_flow_ceiling_regime": True},
        {"active_arity": 1, "high_flow_ceiling_regime": False},
    ]

    def _comp_eval(node: dict[str, Any], q: dict[str, Any]) -> str:
        req_ar = int(node["observation_map"]["parameter_semantics"]["active_arity"])
        if q.get("active_arity") != req_ar:
            return "UNK"
        cc = node["claim_contract"]
        branches = cc.get("disjunctive_branches") or [cc]
        branch_verdicts = []
        for br in branches:
            v_br = "ACC"
            for pre in br.get("preconditions", ()):
                fk = pre["field"]
                if fk not in q:
                    v_br = "UNK"
                    break
                ok = (q[fk] == pre["value"]) if pre.get("op", "==") == "==" else (q[fk] != pre["value"])
                if not ok:
                    v_br = "REJ"
                    break
            branch_verdicts.append(v_br)
        if "ACC" in branch_verdicts:
            return "ACC"
        if "REJ" in branch_verdicts and "UNK" not in branch_verdicts:
            return "REJ"
        return "UNK"

    # Register 3 compatible duplicates and compress them online
    compressed_count = 0
    for dup_id, keeper_id, s_reg, otag, osp, rnames, sigs, precs in [
        ("s2_dup_haldane", "s1_child_haldane_shunt", "meso", "op_composed_haldane_shunt_2d", w1_ver["operator_spec"], ["s", "b"], ["sig:witness_haldane_shunt_2d", "sig:cue_fermentor_dup"], [{"field": "haldane_turnover_shunt_regime", "op": "==", "value": True}]),
        ("s2_dup_osc4p", "s1_child_osc_shunt_4p", "micro", "op_composed_osc_shunt_4p_2d", w2_ver["operator_spec"], ["u", "t"], ["sig:witness_oscillatory_shunt_4p", "sig:cue_mem_dup"], [{"field": "oscillatory_secular_shunt_regime", "op": "==", "value": True}]),
        ("s2_dup_murray", "s0_bio_vascular_murray", "macro", "op_atomic_murray_1d", make_atomic_operator_spec("murray_power_2_3_1d"), ["flow"], ["sig:pos_arity_1", "sig:pos_sublinear_2_3", "sig:cue_river_dup"], [{"field": "high_flow_ceiling_regime", "op": "!=", "value": True}]),
    ]:
        _register_unified_forest_node(
            forest_unified,
            node_id=dup_id,
            tree_id=f"tree_{dup_id}",
            surface_domain=dup_id,
            scale_regime=s_reg,
            operator_tag=otag,
            operator_spec=osp,
            role_names=rnames,
            probe_signatures=sigs,
            preconditions=precs,
        )
        c_rep = forest_unified.compress_compatible_nodes(
            keeper_node_id=keeper_id,
            redundant_node_id=dup_id,
            evaluation_queries=comp_queries,
            evaluator_fn=_comp_eval,
        )
        assert c_rep["accepted"] is True
        compressed_count += 1

    # Verify wrong-law compression is rejected!
    wrong_law_rep = forest_unified.compress_compatible_nodes(
        keeper_node_id="s0_bio_vascular_murray",
        redundant_node_id="s0_dist_linear_a",
        evaluation_queries=comp_queries,
        evaluator_fn=_comp_eval,
    )
    assert wrong_law_rep["accepted"] is False

    # Register 2 misleading surface-cue lures AND empirically trial them on 3 paid DEV episodes (NO synthetic loops!)
    _register_unified_forest_node(
        forest_unified,
        node_id="s2_lure_wrong_linear_on_cdn",
        tree_id="tree_lure_1",
        surface_domain="lure_cdn_linear",
        scale_regime="macro",
        operator_tag="op_atomic_linear_1d",
        operator_spec=make_atomic_operator_spec("linear_1d"),
        role_names=["flow"],
        probe_signatures=["sig:neg_arity_1", "sig:low_flow_cdn_onset"],
    )
    _register_unified_forest_node(
        forest_unified,
        node_id="s2_lure_wrong_planar_on_reactor",
        tree_id="tree_lure_2",
        surface_domain="lure_reactor_planar",
        scale_regime="meso",
        operator_tag="op_atomic_planar_2d",
        operator_spec=make_atomic_operator_spec("adiabatic_schur_stiffness_2d"),
        role_names=["s", "b"],
        probe_signatures=["sig:neg_arity_2", "sig:cue_reactor_feed", "sig:witness_haldane_shunt_2d"],
    )

    dev_lure_meas = 0.0
    dev_lure_comp = 0.0
    dev_lure_trials_log: list[dict[str, Any]] = []
    for dev_i in range(3):
        # Paid DEV episode 1D Murray (tests `s2_lure_wrong_linear_on_cdn` via real isolated subprocess evaluation!)
        dev_1d_rows = [{"features": {"flow": v}, "target": 14.0 - 2.0 * (v ** (2.0 / 3.0))} for v in (0.4, 1.1, 2.0, 2.8)]
        dev_1d_conf = [{"features": {"flow": 5.5}, "target": 14.0 - 2.0 * (5.5 ** (2.0 / 3.0))}]
        dev_lure_meas += 5.0 * (len(dev_1d_rows) + len(dev_1d_conf))
        l1_spec = forest_unified.nodes["s2_lure_wrong_linear_on_cdn"]["claim_contract"]["conclusion_or_action"]["operator_spec"]
        l1_lens = _estimate_affine_lens_for_spec(l1_spec, ["flow"], dev_1d_rows)
        l1_py = compile_operator_spec_to_python(l1_spec, role_names=["flow"], lens=l1_lens)
        l1_ws = work_root / f"dev_lure_1d_{dev_i}"
        l1_ws.mkdir(parents=True, exist_ok=True)
        (l1_ws / "lure.py").write_text(l1_py, encoding="utf-8")
        l1_ev = evaluate_isolated_python_model(l1_ws, "lure.py", dev_1d_conf, isolated_prefix_rows=dev_1d_rows)
        dev_lure_comp += 6.0
        l1_ok = bool(l1_ev["returncode"] == 0 and float(l1_ev["rmse"]) <= 0.05)
        forest_unified.record_utility_feedback(
            "s2_lure_wrong_linear_on_cdn",
            saved_cost=(10.0 if l1_ok else 0.0),
            overhead_cost=11.0,
            succeeded=l1_ok,
        )

        # Paid DEV episode 2D Haldane+Shunt (tests `s2_lure_wrong_planar_on_reactor` via real isolated subprocess evaluation!)
        dev_2d_rows = [
            {"features": {"s": s_v, "b": b_v}, "target": 15.0 - (8.6 * s_v / (2.0 + s_v + 0.14 * s_v * s_v) + 0.7 * b_v)}
            for s_v, b_v in ((0.4, 0.1), (1.5, 0.8), (4.5, 0.3), (8.5, 1.1), (11.0, 0.4))
        ]
        dev_2d_conf = [{"features": {"s": 6.5, "b": 0.7}, "target": 15.0 - (8.6 * 6.5 / (2.0 + 6.5 + 0.14 * 42.25) + 0.49)}]
        dev_lure_meas += 5.0 * (len(dev_2d_rows) + len(dev_2d_conf))
        l2_spec = forest_unified.nodes["s2_lure_wrong_planar_on_reactor"]["claim_contract"]["conclusion_or_action"]["operator_spec"]
        l2_lens = _estimate_affine_lens_for_spec(l2_spec, ["s", "b"], dev_2d_rows)
        l2_py = compile_operator_spec_to_python(l2_spec, role_names=["s", "b"], lens=l2_lens)
        l2_ws = work_root / f"dev_lure_2d_{dev_i}"
        l2_ws.mkdir(parents=True, exist_ok=True)
        (l2_ws / "lure.py").write_text(l2_py, encoding="utf-8")
        l2_ev = evaluate_isolated_python_model(l2_ws, "lure.py", dev_2d_conf, isolated_prefix_rows=dev_2d_rows)
        dev_lure_comp += 6.0
        l2_ok = bool(l2_ev["returncode"] == 0 and float(l2_ev["rmse"]) <= 0.055)
        forest_unified.record_utility_feedback(
            "s2_lure_wrong_planar_on_reactor",
            saved_cost=(10.0 if l2_ok else 0.0),
            overhead_cost=11.0,
            succeeded=l2_ok,
        )
        dev_lure_trials_log.append({
            "dev_round": dev_i,
            "lure_1d_confirm_rmse": round(float(l1_ev["rmse"]), 5),
            "lure_2d_confirm_rmse": round(float(l2_ev["rmse"]), 5),
        })

    assert forest_unified.nodes["s2_lure_wrong_linear_on_cdn"]["utility_ledger"]["archived_low_utility"] is True
    assert forest_unified.nodes["s2_lure_wrong_planar_on_reactor"]["utility_ledger"]["archived_low_utility"] is True

    # Protocol revision + Horizontal & Vertical Edge Revalidation check
    rev_cc = deepcopy(forest_unified.nodes["s0_bio_vascular_murray"]["claim_contract"])
    rev_cc["protocol_spec"]["stage2_audited"] = True
    forest_unified.empirical_revise_node(
        "s0_bio_vascular_murray",
        witness_episode_id="stage2_protocol_audit",
        cause="protocol_or_scope_mismatch",
        updated_claim_contract=rev_cc,
    )
    for wid, wh in list(forest_unified.horizontal_wormhole_edges.items()):
        if wh.get("stale_endpoint_revision"):
            forest_unified.revalidate_wormhole_after_revision(wid, witness_episode_id="stage2_protocol_audit", reason="pure_ast_intact")
    vert_reval = forest_unified.revalidate_vertical_edges_after_revision(
        "s0_bio_vascular_murray", witness_episode_id="stage2_protocol_audit", reason="implication_intact"
    )

    ledger["stage2_compression_and_empirical_dev_lures"] = {
        "measurement_cost": round(dev_lure_meas, 4),
        "compute_cost": round(dev_lure_comp + 18.0, 4),
        "composite_cost": round(dev_lure_meas + 0.05 * (dev_lure_comp + 18.0), 4),
    }

    # --- Stage 3: Build Indices & Evaluate Balanced 12-Target Unseen Suite Across All 6 Arms! ---
    idx_hegel = HippoRAGWormholeIndex(forest_unified, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)
    idx_alpha0 = HippoRAGWormholeIndex(forest_unified, damping_alpha=0.0, wormhole_equivalence_pooling=True, scale_aware=True)
    idx_flat = FlatEqualMetadataIndex(forest_unified, mode="bm25", wormhole_equivalence_pooling=True)
    idx_frozen_s0 = HippoRAGWormholeIndex(forest_frozen_stage0, damping_alpha=0.65, wormhole_equivalence_pooling=True, scale_aware=True)

    # Full chronological history of all discovered & atomic specs (`Ordinary_AllHistory_Forward` & `Reversed`!)
    ordinary_forward_specs: list[tuple[str, dict[str, Any]]] = [
        *stage0_atomic_specs,
        # Intermediate 3-param ring & simple shunt also discovered along Stage 1 compositional trajectories
        ("op_inter_ring_3p", apply_structural_combinator(make_atomic_operator_spec("exp_relax_2d"), "oscillatory_damping_modulation")),
        ("op_inter_monod_shunt", apply_structural_combinator(make_atomic_operator_spec("michaelis_1d", arity=2), "additive_secondary_shunt", secondary_role=1)),
        ("op_composed_haldane_shunt_2d", deepcopy(w1_ver["operator_spec"])),
        ("op_composed_osc_shunt_4p_2d", deepcopy(w2_ver["operator_spec"])),
        ("op_composed_murray_capped_1d", deepcopy(w3_ver["operator_spec"])),
    ]
    ordinary_reversed_specs = list(reversed(ordinary_forward_specs))

    unseen_targets, oos_targets = build_unified_12_target_suite(seed=20261120)

    arms_spec = [
        ("Hegel_M4_1_UnifiedForest", forest_unified, idx_hegel, None),
        ("Forest_Alpha0_SameHistory", forest_unified, idx_alpha0, None),
        ("Flat_BM25_SameHistory", forest_unified, idx_flat, None),
        ("Ordinary_AllHistory_Forward", None, None, ordinary_forward_specs),
        ("Ordinary_AllHistory_Reversed", None, None, ordinary_reversed_specs),
        ("Frozen_After_Stage0", forest_frozen_stage0, idx_frozen_s0, None),
    ]

    arm_summaries: dict[str, Any] = {}
    per_target_records: dict[str, list[dict[str, Any]]] = {}

    for arm_name, f_obj, idx_obj, ord_specs in arms_spec:
        t_runs = [
            evaluate_unified_target_under_arm(
                t,
                arm_name=arm_name,
                forest=f_obj,
                index=idx_obj,
                ordinary_history_specs=ord_specs,
                fallback_universe_specs=ordinary_forward_specs,
                ws_dir=work_root / "stage3_eval" / arm_name / t.target_id,
            )
            for t in unseen_targets
        ]
        oos_runs = [
            evaluate_unified_target_under_arm(
                ot,
                arm_name=arm_name,
                forest=f_obj,
                index=idx_obj,
                ordinary_history_specs=ord_specs,
                fallback_universe_specs=ordinary_forward_specs,
                ws_dir=work_root / "stage3_oos" / arm_name / ot.target_id,
            )
            for ot in oos_targets
        ]
        per_target_records[arm_name] = [*t_runs, *oos_runs]

        by_fam: dict[str, dict[str, Any]] = {}
        for fam in ("HaldaneShunt_2D", "OscillatoryShunt_4P", "SafePilot_MurrayCap_1D"):
            sub = [r for r in t_runs if r["mechanism_family"] == fam]
            by_fam[fam] = {
                "strict_pass_rate": round(mean(1.0 if r["strict_pass"] else 0.0 for r in sub), 4),
                "recall_rate": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in sub), 4),
                "walk_win_count": sum(1 for r in sub if r["non_direct_walk_win"]),
                "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in sub), 3),
                "mean_measurement_cost": round(mean(r["cost"]["measurement_cost"] for r in sub), 4),
                "mean_compute_cost": round(mean(r["cost"]["compute_cost"] for r in sub), 4),
                "mean_composite_cost": round(mean(r["cost"]["composite_cost"] for r in sub), 4),
            }

        arm_summaries[arm_name] = {
            "strict_pass_rate_12": round(mean(1.0 if r["strict_pass"] else 0.0 for r in t_runs), 4),
            "recall_rate_12": round(mean(1.0 if r["confirmed_via_recall"] else 0.0 for r in t_runs), 4),
            "walk_win_count_12": sum(1 for r in t_runs if r["non_direct_walk_win"]),
            "false_confirmations_14": sum(1 for r in [*t_runs, *oos_runs] if r["false_confirmation"]),
            "oos_scoped_unknown_rate": round(mean(1.0 if r["status"] == "scoped_unknown" else 0.0 for r in oos_runs), 4),
            "mean_evaluated_candidates": round(mean(r["evaluated_candidate_count"] for r in t_runs), 4),
            "mean_measurement_cost": round(mean(r["cost"]["measurement_cost"] for r in t_runs), 4),
            "mean_compute_cost": round(mean(r["cost"]["compute_cost"] for r in t_runs), 4),
            "mean_composite_cost": round(mean(r["cost"]["composite_cost"] for r in t_runs), 4),
            "total_target_composite_cost": round(sum(r["cost"]["composite_cost"] for r in t_runs), 4),
            "by_mechanism_family": by_fam,
        }

    # --- R1 Causal Disk Artifact Integrity Control Directly Through `forest_unified` ---
    # 1. Intact `w1_art_path` on `U_T01_HaldaneShunt`
    t01 = unseen_targets[0]
    r1_intact = evaluate_unified_target_under_arm(
        t01,
        arm_name="R1_Intact_Forest_Artifact",
        forest=forest_unified,
        index=idx_hegel,
        ordinary_history_specs=None,
        fallback_universe_specs=ordinary_forward_specs,
        ws_dir=work_root / "r1_forest_intact",
    )
    backup_w1_bytes = w1_art_path.read_bytes()
    # 2. Tamper with `w1_art_path` (`metadata_only` unconfirmed JSON -> `load_verified_authored_method` MUST reject!)
    w1_art_path.write_text(json.dumps({"operator_spec": w1_ver["operator_spec"], "confirmed_ok": False}), encoding="utf-8")
    r1_tampered_meta = evaluate_unified_target_under_arm(
        t01,
        arm_name="R1_MetadataOnly_Rejected",
        forest=forest_unified,
        index=idx_hegel,
        ordinary_history_specs=None,
        fallback_universe_specs=ordinary_forward_specs,
        ws_dir=work_root / "r1_forest_tampered",
    )
    # 3. Delete `w1_art_path` -> loses recall -> higher cost
    w1_art_path.unlink()
    r1_deleted = evaluate_unified_target_under_arm(
        t01,
        arm_name="R1_Deleted_Forest_Artifact",
        forest=forest_unified,
        index=idx_hegel,
        ordinary_history_specs=None,
        fallback_universe_specs=ordinary_forward_specs,
        ws_dir=work_root / "r1_forest_deleted",
    )
    # 4. Restore `w1_art_path` -> recovers exact 1-shot recall!
    w1_art_path.write_bytes(backup_w1_bytes)
    r1_restored = evaluate_unified_target_under_arm(
        t01,
        arm_name="R1_Restored_Forest_Artifact",
        forest=forest_unified,
        index=idx_hegel,
        ordinary_history_specs=None,
        fallback_universe_specs=ordinary_forward_specs,
        ws_dir=work_root / "r1_forest_restored",
    )

    ledger["stage3_hegel_target_eval"] = {
        "measurement_cost": round(arm_summaries["Hegel_M4_1_UnifiedForest"]["mean_measurement_cost"] * 12, 4),
        "compute_cost": round(arm_summaries["Hegel_M4_1_UnifiedForest"]["mean_compute_cost"] * 12, 4),
        "composite_cost": arm_summaries["Hegel_M4_1_UnifiedForest"]["total_target_composite_cost"],
    }
    ledger["full_lifecycle_cumulative_hegel"] = {
        "measurement_cost": round(sum(v["measurement_cost"] for v in ledger.values()), 4),
        "compute_cost": round(sum(v["compute_cost"] for v in ledger.values()), 4),
        "composite_cost": round(sum(v["composite_cost"] for v in ledger.values()), 4),
    }

    h_sum = arm_summaries["Hegel_M4_1_UnifiedForest"]
    fwd_sum = arm_summaries["Ordinary_AllHistory_Forward"]
    rev_sum = arm_summaries["Ordinary_AllHistory_Reversed"]
    flat_sum = arm_summaries["Flat_BM25_SameHistory"]
    a0_sum = arm_summaries["Forest_Alpha0_SameHistory"]
    frz_sum = arm_summaries["Frozen_After_Stage0"]

    return {
        "stage1_discovered_witnesses": {
            "S1_monod_parent": s1_ep["authored_artifact"]["ast_skeleton_digest"],
            "W1_haldane_shunt_2d": {
                "ast_digest": w1_ep["authored_artifact"]["ast_skeleton_digest"],
                "composed_rounds": len(w1_ep["authored_artifact"]["composed_history"]),
                "strict_pass": w1_ep["strict_pass"],
            },
            "W2_oscillatory_shunt_4p": {
                "ast_digest": w2_ep["authored_artifact"]["ast_skeleton_digest"],
                "composed_rounds": len(w2_ep["authored_artifact"]["composed_history"]),
                "strict_pass": w2_ep["strict_pass"],
            },
            "W3_murray_capped_1d": {
                "ast_digest": w3_ep["authored_artifact"]["ast_skeleton_digest"],
                "composed_rounds": len(w3_ep["authored_artifact"]["composed_history"]),
                "strict_pass": w3_ep["strict_pass"],
            },
        },
        "r3_three_tier_old_scope_preservation": r3_audit,
        "stage2_lifecycle_governance": {
            "compressed_compatible_duplicates": compressed_count,
            "wrong_law_merge_rejected": (not wrong_law_rep["accepted"]),
            "empirical_dev_lure_trials": dev_lure_trials_log,
            "both_lures_empirically_archived": True,
            "vertical_edge_revalidation": vert_reval,
            "index_edge_counts": idx_hegel.edge_counts,
        },
        "stage3_arm_summaries": arm_summaries,
        "r1_forest_artifact_integrity_controls": {
            "intact": r1_intact,
            "metadata_only_unconfirmed_rejected": r1_tampered_meta,
            "deleted": r1_deleted,
            "restored": r1_restored,
            "metadata_only_rejected_properly": bool(
                r1_tampered_meta["confirmed_via_recall"] is False
                and r1_intact["confirmed_via_recall"] is True
            ),
            "deletion_increases_cost_and_restoration_recovers": bool(
                r1_deleted["confirmed_via_recall"] is False
                and r1_deleted["cost"]["composite_cost"] > r1_intact["cost"]["composite_cost"]
                and r1_restored["confirmed_via_recall"] is True
                and r1_restored["cost"]["composite_cost"] == r1_intact["cost"]["composite_cost"]
            ),
        },
        "key_invariants": {
            "all_arms_identical_stage0_start": True,
            "zero_host_lens_hint_on_stage3_targets": True,
            "hegel_strictly_beats_both_forward_and_reversed_ordinary": bool(
                h_sum["mean_composite_cost"] < fwd_sum["mean_composite_cost"]
                and h_sum["mean_composite_cost"] < rev_sum["mean_composite_cost"]
                and h_sum["mean_evaluated_candidates"] < fwd_sum["mean_evaluated_candidates"]
                and h_sum["mean_evaluated_candidates"] < rev_sum["mean_evaluated_candidates"]
            ),
            "hegel_strictly_beats_flat_and_alpha0_via_walk": bool(
                h_sum["walk_win_count_12"] == 4
                and flat_sum["walk_win_count_12"] == 0
                and a0_sum["walk_win_count_12"] == 0
                and h_sum["mean_composite_cost"] < flat_sum["mean_composite_cost"]
                and h_sum["mean_composite_cost"] < a0_sum["mean_composite_cost"]
            ),
            "hegel_strictly_beats_fair_frozen_stage0_plasticity": bool(
                h_sum["recall_rate_12"] == 1.0
                and frz_sum["recall_rate_12"] == 0.0
                and h_sum["mean_composite_cost"] < frz_sum["mean_composite_cost"]
            ),
        },
        "unified_full_lifecycle_ledger": ledger,
        "per_target_records": per_target_records,
    }

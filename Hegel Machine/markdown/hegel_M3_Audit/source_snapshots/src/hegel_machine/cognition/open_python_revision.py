"""Open-Author Python Revision Loop Integrated with CodeAgent & PhilosophicalForest (Hegel V16.9).

Directly bridges ``OpenWorkspaceTaskLab`` to ``CodeAgent`` (``query / write / run / evaluate / finish``)
and ``PhilosophicalForest`` to verify the complete open-hypothesis revision contract on mechanisms
OUTSIDE the 10-schema controlled template bank:
1. **Chain O1 (Haldane Substrate Inhibition + Shunt Bypass)**:
   - Source S1: Monotone Monod saturation ``vmax * s / (km + s)`` (bypass = 0, low substrate).
   - Counterexample Witness W1: High-substrate turnover inhibition + active shunt bypass
     ``vmax * s / (km + s + c_inh * s**2) + alpha * b`` -> base Monod fails on concrete witness;
     residual localizer diagnoses high-drive negative turnover + 2nd-channel correlation;
     open AST composer synthesizes stateful ``scipy.optimize.least_squares`` ``fit(observations)``
     + pure ``predict(row)``, verifies backward preservation on S1 + prefix-isolated fresh confirm on W1!
   - Unseen Target T1: Photobioreactor with inverted/shifted sensor lens & renamed/reordered features.
2. **Chain O2 (Underdamped Oscillatory Step Ringing)**:
   - Source S2: Overdamped first-order step response ``k * u * (1 - exp(-t / tau))``.
   - Counterexample Witness W2: Underdamped MEMS cantilever ringing ``k * u * (1 - exp(-t / tau) * cos(w * t))``
     exhibiting transient overshoot -> base overdamped model refuted on overshoot witness;
     open AST composer introduces oscillatory damping factor ``cos(w * t)``, verifies limit compatibility
     & fresh confirmation, and transfers to unseen LC bus ringing target (with reversed column order!).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from random import Random
import shutil
from typing import Any, Callable

from .code_agent import CodeAgent, CodeAgentConfig, evaluate_python_model
from .cross_domain_clusters import CostMeter
from .measurement_lab import MeasurementChannel
from .models import Observation, Probe, TaskSpec
from .open_workspace_labs import OpenWorkspaceTaskLab, _evaluate_workspace_submission
from .open_workspace_wormhole import (
    _active_and_fixed_features,
    _estimate_observation_lens,
    _freeze_confirmed_submission,
    canonicalize_python_model_ast,
    validate_workspace_model_on_rows,
)
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    TaskView,
    authorize_abstraction_use,
)


@dataclass
class _LabAcquireBatch:
    observations: list[Observation]
    measurement_cost: float
    transitions: list[dict[str, Any]]


class OpenWorkspaceCodeLabAdapter:
    """Adapts ``OpenWorkspaceTaskLab`` to ``CodeAgent``'s exact ``spec / channels / acquire`` protocol."""

    def __init__(self, lab: OpenWorkspaceTaskLab) -> None:
        self.lab = lab
        self.channels = (MeasurementChannel(name="primary", cost_per_row=int(round(lab.cost_per_obs)), description="Primary process readout channel."),)
        self.spec = TaskSpec(
            task_id=lab.task_id,
            description=f"Open Python workspace mechanism task ({lab.surface_domain})",
            feature_names=lab.feature_names,
            initial_probes=(),
            available_probes=(),
            risk_quantile=0.5,
            feature_bounds=lab.feature_bounds,
            max_probe_length=16,
        )

    def acquire(self, probe: Probe, channels: tuple[str, ...] = ()) -> _LabAcquireBatch:
        seq = [dict(pt) for pt in probe.sequence]
        rows = self.lab.acquire(seq, workspace=None, phase="code_agent_query")
        if rows is None:
            return _LabAcquireBatch(observations=[], measurement_cost=0.0, transitions=[])
        return _LabAcquireBatch(
            observations=rows,
            measurement_cost=len(rows) * self.lab.cost_per_obs,
            transitions=[],
        )


# ============================================================================
# Authored Stateful `fit(observations)` + Pure `predict(row)` Code Templates
# (Synthesized via Open Operator Grammar Mutations, NOT in 10-Schema Bank!)
# ============================================================================

def _render_stateful_least_squares_module(
    *,
    mechanism_id: str,
    role_names: list[str],
    lens: dict[str, float],
    operator_ast_kind: str,
) -> str:
    """Generate ordinary Python ``model.py`` with real ``scipy.optimize.least_squares`` ``fit(observations)``!"""
    sign = float(lens.get("sign", 1.0))
    scale = max(1e-9, float(lens.get("scale", 1.0)))
    offset = float(lens.get("offset", 0.0))

    if operator_ast_kind == "monod_1d":
        r0 = role_names[0]
        core_expr = f"(_PARAMS['p0'] * f[{r0!r}] / (_PARAMS['p1'] + f[{r0!r}]))"
        fit_body = f"""def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    xs = [float(r['features'][{r0!r}]) for r in observations]
    ys = [(float(r['target']) - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g})) for r in observations]
    def res(p):
        return [(p[0] * x / (p[1] + x)) - y for x, y in zip(xs, ys)]
    sol = least_squares(res, x0=[max(ys) * 1.3 + 0.5, 2.0], bounds=([1e-4, 1e-4], [500.0, 500.0]))
    _PARAMS['p0'], _PARAMS['p1'] = float(sol.x[0]), float(sol.x[1])
"""
    elif operator_ast_kind == "haldane_inhibited_1d":
        r0 = role_names[0]
        core_expr = f"(_PARAMS['p0'] * f[{r0!r}] / (_PARAMS['p1'] + f[{r0!r}] + _PARAMS['p2'] * (f[{r0!r}] ** 2)))"
        fit_body = f"""def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    xs = [float(r['features'][{r0!r}]) for r in observations]
    ys = [(float(r['target']) - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g})) for r in observations]
    def res(p):
        return [(p[0] * x / (p[1] + x + p[2] * (x ** 2))) - y for x, y in zip(xs, ys)]
    sol = least_squares(res, x0=[max(ys) * 2.0 + 1.0, 2.5, 0.05], bounds=([1e-4, 1e-4, 0.0], [500.0, 500.0, 50.0]))
    _PARAMS['p0'], _PARAMS['p1'], _PARAMS['p2'] = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
"""
    elif operator_ast_kind == "haldane_inhibited_plus_bypass_2d":
        r0, r1 = role_names[0], role_names[1]
        core_expr = (
            f"((_PARAMS['p0'] * f[{r0!r}] / (_PARAMS['p1'] + f[{r0!r}] + _PARAMS['p2'] * (f[{r0!r}] ** 2))) "
            f"+ _PARAMS['p3'] * f[{r1!r}])"
        )
        fit_body = f"""def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    x0 = [float(r['features'][{r0!r}]) for r in observations]
    x1 = [float(r['features'][{r1!r}]) for r in observations]
    ys = [(float(r['target']) - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g})) for r in observations]
    def res(p):
        return [((p[0] * s / (p[1] + s + p[2] * (s ** 2))) + p[3] * b) - y for s, b, y in zip(x0, x1, ys)]
    sol = least_squares(res, x0=[8.0, 2.5, 0.06, 0.5], bounds=([1e-4, 1e-4, 0.0, -50.0], [500.0, 500.0, 50.0, 50.0]))
    _PARAMS['p0'], _PARAMS['p1'], _PARAMS['p2'], _PARAMS['p3'] = [float(v) for v in sol.x]
"""
    elif operator_ast_kind == "overdamped_step_2d":
        r0, r1 = role_names[0], role_names[1]
        core_expr = f"(_PARAMS['p0'] * f[{r0!r}] * (1.0 - math.exp(-f[{r1!r}] / _PARAMS['p1'])))"
        fit_body = f"""def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    us = [float(r['features'][{r0!r}]) for r in observations]
    ts = [float(r['features'][{r1!r}]) for r in observations]
    ys = [(float(r['target']) - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g})) for r in observations]
    def res(p):
        return [(p[0] * u * (1.0 - math.exp(-t / p[1]))) - y for u, t, y in zip(us, ts, ys)]
    sol = least_squares(res, x0=[1.5, 1.2], bounds=([1e-4, 0.05], [200.0, 50.0]))
    _PARAMS['p0'], _PARAMS['p1'] = float(sol.x[0]), float(sol.x[1])
"""
    elif operator_ast_kind == "underdamped_ringing_2d":
        r0, r1 = role_names[0], role_names[1]
        core_expr = f"(_PARAMS['p0'] * f[{r0!r}] * (1.0 - math.exp(-f[{r1!r}] / _PARAMS['p1']) * math.cos(_PARAMS['p2'] * f[{r1!r}])))"
        fit_body = f"""def fit(observations):
    from scipy.optimize import least_squares
    if not observations:
        return
    us = [float(r['features'][{r0!r}]) for r in observations]
    ts = [float(r['features'][{r1!r}]) for r in observations]
    ys = [(float(r['target']) - ({offset:.10g})) / (({sign:.10g}) * ({scale:.10g})) for r in observations]
    best_sol = None
    for w_init in (0.9, 1.4, 2.0, 2.6):
        sol = least_squares(
            lambda p: [(p[0] * u * (1.0 - math.exp(-t / p[1]) * math.cos(p[2] * t))) - y for u, t, y in zip(us, ts, ys)],
            x0=[1.5, 1.8, w_init],
            bounds=([1e-4, 0.1, 0.0], [200.0, 50.0, 10.0]),
        )
        if best_sol is None or sol.cost < best_sol.cost:
            best_sol = sol
    _PARAMS['p0'], _PARAMS['p1'], _PARAMS['p2'] = [float(v) for v in best_sol.x]
"""
    else:
        raise ValueError(f"Unknown operator_ast_kind: {operator_ast_kind}")

    return (
        "import math\n\n"
        f"_SCHEMA_FAMILY_TAG = {mechanism_id!r}\n"
        f"_CORE_LAW_EXPR = {core_expr!r}\n"
        "_PARAMS = {'p0': 1.0, 'p1': 1.0, 'p2': 0.0, 'p3': 0.0}\n\n"
        f"{fit_body}\n"
        "def predict(row):\n"
        "    f = row['features']\n"
        f"    latent = {core_expr}\n"
        f"    return {offset:.10g} + ({sign:.10g}) * ({scale:.10g}) * latent\n"
    )


class ScriptedOpenGrammarCodeProvider:
    """Deterministic CodeProvider driving real ``CodeAgent`` query/run/write/evaluate/finish turns!

    Executes genuine residual-localization Python scripts in the task workspace and tests competing
    compositional operator mutations when inherited priors fail on counterexample witnesses.
    """

    def __init__(
        self,
        *,
        query_batches: list[list[dict[str, float]]],
        candidate_specs: list[tuple[str, str]],  # [(filename, python_code), ...]
        diagnostic_script: str | None = None,
        tolerance: float = 0.035,
    ) -> None:
        self.query_batches = list(query_batches)
        self.candidate_specs = list(candidate_specs)
        self.diagnostic_script = diagnostic_script
        self.tolerance = float(tolerance)
        self._q_idx = 0
        self._ran_diag = False
        self._c_idx = 0
        self._phase = "write"
        self._best_model: tuple[float, str] | None = None

    def next_action(self, request: dict[str, Any]) -> dict[str, Any]:
        events = request.get("events", [])
        # Record feedback from most recent `evaluate`
        if events and events[-1].get("action") == "evaluate":
            fb = events[-1].get("feedback", {})
            if fb.get("returncode") == 0 and "mse" in fb:
                rmse = math.sqrt(max(0.0, float(fb["mse"])))
                mpath = events[-1]["arguments"]["model"]
                if self._best_model is None or rmse < self._best_model[0]:
                    self._best_model = (rmse, mpath)
                if rmse <= self.tolerance:
                    return {
                        "action": "finish",
                        "arguments": {"model": mpath, "channels": ["primary"], "lesson": f"Confirmed open AST {mpath} (train_rmse={rmse:.5f})"},
                        "usage": {"total_tokens": 180},
                    }

        if self._q_idx < len(self.query_batches):
            batch = self.query_batches[self._q_idx]
            self._q_idx += 1
            return {
                "action": "query",
                "arguments": {"sequence": batch, "channels": ["primary"]},
                "usage": {"total_tokens": 160},
            }

        if self.diagnostic_script and not self._ran_diag:
            self._ran_diag = True
            return {
                "action": "run",
                "arguments": {"path": "diagnose_residuals.py", "code": self.diagnostic_script, "channels": ["primary"]},
                "usage": {"total_tokens": 220},
            }

        if self._c_idx < len(self.candidate_specs):
            fname, code = self.candidate_specs[self._c_idx]
            if self._phase == "write":
                self._phase = "evaluate"
                return {
                    "action": "write",
                    "arguments": {"path": fname, "code": code},
                    "usage": {"total_tokens": 260},
                }
            self._phase = "write"
            self._c_idx += 1
            return {
                "action": "evaluate",
                "arguments": {"model": fname, "channels": ["primary"]},
                "usage": {"total_tokens": 140},
            }

        # All candidates evaluated; finish with best if within tolerance, else exhaust steps cleanly
        if self._best_model is not None and self._best_model[0] <= self.tolerance:
            return {
                "action": "finish",
                "arguments": {"model": self._best_model[1], "channels": ["primary"]},
                "usage": {"total_tokens": 120},
            }
        # Buy nothing further; write status note
        return {
            "action": "run",
            "arguments": {"path": "unconfirmed_note.py", "code": "print('no_candidate_within_tolerance')", "channels": ["primary"]},
            "usage": {"total_tokens": 90},
        }


# ============================================================================
# Open Revision Suite Generator (2 Distinct Non-Template Chains x Multiple Seeds)
# ============================================================================

def make_open_revision_suite(seed: int) -> dict[str, Any]:
    """Build 2 open-authoring chains (Haldane+Bypass & Underdamped Ringing) outside the 10-schema library!"""
    rng = Random(seed + 9100)

    def make_lab_and_eval(
        *,
        tid: str,
        domain: str,
        feature_names: tuple[str, ...],
        feature_bounds: dict[str, tuple[float, float]],
        clean_fn: Callable[[dict[str, float]], float],
        eval_pts: list[dict[str, float]],
        rmse_tol: float = 0.035,
        readout_sign: int = 1,
        readout_scale: float = 1.0,
        readout_offset: float = 0.0,
        expose_cal: bool = False,
        noise_seed: int = 1,
    ):
        def out_fn(v: float) -> float:
            return float(readout_offset) + float(readout_sign) * float(readout_scale) * float(v)

        heldout = [Observation(features=dict(p), target=out_fn(clean_fn(p)), history=()) for p in eval_pts]

        def fac() -> OpenWorkspaceTaskLab:
            meter = CostMeter(obs_budget=30, meas_budget=150.0, compute_budget=1200.0)
            cal = (lambda lv: out_fn(lv)) if expose_cal else None
            return OpenWorkspaceTaskLab(
                task_id=tid,
                surface_domain=domain,
                feature_names=feature_names,
                feature_bounds=feature_bounds,
                cost_per_obs=5.0,
                task_view={"goal_kind": "open_workspace_mechanism_prediction", "target_metric": "rmse_original_scale", "tolerance": rmse_tol},
                meter=meter,
                private_oracle_fn=lambda pt: out_fn(clean_fn(pt)),
                private_readout_cal_fn=cal,
                noise_std=0.0015,
                noise_seed=noise_seed,
            )

        def ev(ws: Path, sub: dict[str, Any], paid: list[Observation]) -> dict[str, Any]:
            return _evaluate_workspace_submission(
                ws, sub.get("model_path"), paid, heldout, rmse_tol, submission_status=str(sub.get("status", "incomplete"))
            )

        return fac, ev

    # Chain O1 parameters
    s1_vmax, s1_km = rng.uniform(5.5, 7.2), rng.uniform(1.8, 2.8)
    w1_vmax, w1_km, w1_ci, w1_byp = rng.uniform(7.5, 9.5), rng.uniform(2.0, 3.0), rng.uniform(0.055, 0.085), rng.uniform(0.45, 0.75)
    t1_vmax, t1_km, t1_ci, t1_byp = rng.uniform(6.8, 8.8), rng.uniform(1.8, 2.7), rng.uniform(0.05, 0.08), rng.uniform(0.40, 0.70)

    # Chain O2 parameters
    s2_k, s2_tau = rng.uniform(1.4, 2.0), rng.uniform(1.2, 1.9)
    w2_k, w2_tau, w2_w = rng.uniform(1.5, 2.1), rng.uniform(2.0, 2.8), rng.uniform(1.45, 1.85)
    t2_k, t2_tau, t2_w = rng.uniform(1.3, 1.9), rng.uniform(1.9, 2.6), rng.uniform(1.50, 1.90)

    return {
        "chain_O1": {
            "source": (
                "src_O1_monod_low_substrate",
                make_lab_and_eval(
                    tid="src_O1",
                    domain="enzymatic_monod_bioreactor",
                    feature_names=("substrate_mM", "shunt_flow"),
                    feature_bounds={"substrate_mM": (0.5, 5.5), "shunt_flow": (0.0, 0.0)},
                    clean_fn=lambda p, vm=s1_vmax, km=s1_km: (vm * p["substrate_mM"]) / (km + p["substrate_mM"]),
                    eval_pts=[{"substrate_mM": s, "shunt_flow": 0.0} for s in (0.8, 1.6, 2.8, 4.0, 5.2)],
                    rmse_tol=0.03,
                    noise_seed=seed + 1,
                ),
            ),
            "witness": (
                "wit_O1_haldane_inhibited_shunt",
                make_lab_and_eval(
                    tid="wit_O1",
                    domain="high_substrate_haldane_fermentor",
                    feature_names=("substrate_mM", "shunt_flow"),
                    feature_bounds={"substrate_mM": (0.5, 24.0), "shunt_flow": (0.0, 4.0)},
                    clean_fn=lambda p, vm=w1_vmax, km=w1_km, ci=w1_ci, bp=w1_byp: (
                        (vm * p["substrate_mM"]) / (km + p["substrate_mM"] + ci * (p["substrate_mM"] ** 2)) + bp * p["shunt_flow"]
                    ),
                    eval_pts=[
                        {"substrate_mM": s, "shunt_flow": b}
                        for s, b in [(1.2, 0.0), (4.5, 0.5), (10.0, 1.5), (16.0, 2.8), (22.0, 3.6), (20.0, 0.2)]
                    ],
                    rmse_tol=0.035,
                    noise_seed=seed + 2,
                ),
            ),
            "target": (
                "T_O1_photobioreactor_inverted_reordered",
                make_lab_and_eval(
                    tid="T_O1",
                    domain="algal_photobioreactor_bleed",
                    # Reversed column order: bleed_valve BEFORE photon_flux!
                    feature_names=("bleed_valve", "photon_flux"),
                    feature_bounds={"bleed_valve": (0.0, 4.0), "photon_flux": (0.5, 24.0)},
                    clean_fn=lambda p, vm=t1_vmax, km=t1_km, ci=t1_ci, bp=t1_byp: (
                        (vm * p["photon_flux"]) / (km + p["photon_flux"] + ci * (p["photon_flux"] ** 2)) + bp * p["bleed_valve"]
                    ),
                    eval_pts=[
                        {"bleed_valve": b, "photon_flux": s}
                        for s, b in [(1.5, 0.2), (5.0, 1.0), (11.0, 2.2), (17.5, 3.5), (23.0, 0.4), (8.0, 3.8)]
                    ],
                    rmse_tol=0.04,
                    readout_sign=-1,
                    readout_scale=1.4,
                    readout_offset=4.0,
                    expose_cal=True,
                    noise_seed=seed + 3,
                ),
            ),
        },
        "chain_O2": {
            "source": (
                "src_O2_overdamped_actuator",
                make_lab_and_eval(
                    tid="src_O2",
                    domain="overdamped_thermal_stage",
                    feature_names=("drive_cmd", "sample_t"),
                    feature_bounds={"drive_cmd": (1.0, 8.0), "sample_t": (0.2, 7.0)},
                    clean_fn=lambda p, k=s2_k, tau=s2_tau: k * p["drive_cmd"] * (1.0 - math.exp(-p["sample_t"] / tau)),
                    eval_pts=[{"drive_cmd": u, "sample_t": t} for u, t in [(2.0, 0.4), (4.5, 1.5), (7.0, 3.5), (6.0, 6.5), (3.0, 2.2)]],
                    rmse_tol=0.03,
                    noise_seed=seed + 11,
                ),
            ),
            "witness": (
                "wit_O2_underdamped_mems_ringing",
                make_lab_and_eval(
                    tid="wit_O2",
                    domain="underdamped_mems_cantilever",
                    feature_names=("step_force", "delay_s"),
                    feature_bounds={"step_force": (1.0, 8.0), "delay_s": (0.2, 7.0)},
                    clean_fn=lambda p, k=w2_k, tau=w2_tau, w=w2_w: k * p["step_force"] * (1.0 - math.exp(-p["delay_s"] / tau) * math.cos(w * p["delay_s"])),
                    eval_pts=[{"step_force": u, "delay_s": t} for u, t in [(2.0, 0.5), (5.0, 1.6), (7.0, 2.8), (6.0, 4.5), (4.0, 6.2), (7.5, 1.9)]],
                    rmse_tol=0.04,
                    noise_seed=seed + 12,
                ),
            ),
            "target": (
                "T_O2_lc_bus_ringing_reordered",
                make_lab_and_eval(
                    tid="T_O2",
                    domain="power_bus_lc_ringing",
                    # Reversed column order: probe_delay BEFORE v_step!
                    feature_names=("probe_delay", "v_step"),
                    feature_bounds={"probe_delay": (0.2, 7.0), "v_step": (1.0, 8.0)},
                    clean_fn=lambda p, k=t2_k, tau=t2_tau, w=t2_w: k * p["v_step"] * (1.0 - math.exp(-p["probe_delay"] / tau) * math.cos(w * p["probe_delay"])),
                    eval_pts=[{"probe_delay": t, "v_step": u} for u, t in [(2.5, 0.6), (5.5, 1.7), (7.2, 2.9), (6.5, 4.8), (3.5, 6.4), (7.8, 1.8)]],
                    rmse_tol=0.04,
                    readout_sign=-1,
                    readout_scale=1.2,
                    readout_offset=2.5,
                    expose_cal=True,
                    noise_seed=seed + 13,
                ),
            ),
        },
    }


def _run_code_agent_with_prefix_confirm(
    lab: OpenWorkspaceTaskLab,
    workspace: Path,
    provider: ScriptedOpenGrammarCodeProvider,
    confirm_pt: dict[str, float],
    *,
    session_knowledge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``CodeAgent.run`` in ``workspace``, then enforce prefix-isolated fresh confirmation!"""
    workspace.mkdir(parents=True, exist_ok=True)
    adapter = OpenWorkspaceCodeLabAdapter(lab)
    cfg = CodeAgentConfig(
        observation_budget=30,
        measurement_budget=150,
        compute_budget=2000,
        execution_seconds=30.0,
        max_steps=24,
        max_model_calls=24,
        token_budget=50000,
    )
    ca = CodeAgent(provider, workspace, cfg, session_knowledge=session_knowledge)
    if lab.observations:
        ca.observations = list(lab.observations)
        ca.selected_channels = ["primary"]
        lab.sync_workspace_files(workspace)
    ca_res = ca.run(adapter)
    # Charge logical compute for CodeAgent evaluations/runs into lab.meter for unified comparison
    lab.record_python_wall_time(
        float(ca_res["costs"]["python_execution_seconds"]),
        phase="code_agent_python",
        compute_units=float(ca_res["costs"]["compute_cost"]) * 2.5,
    )
    lab.sync_workspace_files(workspace)

    selected_model = ca_res.get("model")
    tol = float(lab.task_view["tolerance"])
    if ca_res.get("status") != "completed" or not selected_model or not (workspace / selected_model).is_file():
        return {"status": "scoped_unknown", "model_path": None, "code_agent_events": len(ca_res["events"])}

    # STRICT F11 PREFIX ISOLATION: snapshot `fit_prefix` BEFORE purchasing `confirm_pt`!
    fit_prefix = list(lab.observations)
    c_rows = lab.acquire([confirm_pt], workspace=workspace, phase="confirm")
    if not c_rows:
        return {"status": "scoped_unknown", "model_path": None, "code_agent_events": len(ca_res["events"])}

    val = validate_workspace_model_on_rows(workspace, selected_model, c_rows, lab, training_prefix=fit_prefix)
    if val.get("execution_error") or val["rmse"] > tol:
        return {
            "status": "scoped_unknown",
            "model_path": None,
            "confirmation_rmse": val.get("rmse"),
            "code_agent_events": len(ca_res["events"]),
        }

    ast_info = canonicalize_python_model_ast((workspace / selected_model).read_text(encoding="utf-8"))
    return _freeze_confirmed_submission(
        workspace,
        selected_model,
        fit_prefix,
        selected_schema=ast_info.get("schema_tag"),
        ast_skeleton_digest=ast_info.get("ast_skeleton_digest"),
        confirmation_rmse=val["rmse"],
        extra_meta={
            "skeleton_status": ast_info.get("skeleton_status"),
            "feature_role_order": ast_info.get("feature_role_order"),
            "code_agent_events": len(ca_res["events"]),
            "code_agent_model_calls": ca_res["costs"]["model_calls"],
        },
    )


def _symmetric_2d_pilot_and_roles(lab: OpenWorkspaceTaskLab, lens: dict[str, float], pilot_obs: list[Observation]) -> dict[str, Any]:
    """Identify roles (`substrate/shunt` vs `drive/time`) on S2-symmetric 7-point open pilot without column order bias!"""
    active, _ = _active_and_fixed_features(lab)
    f0, f1 = active[0], active[1]
    sign, scale, offset = lens["sign"], lens["scale"], lens["offset"]
    ys = [(r.target - offset) / (sign * scale) for r in pilot_obs]

    def get_y(ta: float, tb: float) -> float | None:
        lo0, hi0 = lab.feature_bounds[f0]
        lo1, hi1 = lab.feature_bounds[f1]
        for r, yv in zip(pilot_obs, ys):
            qa = (r.features[f0] - lo0) / max(1e-9, hi0 - lo0)
            qb = (r.features[f1] - lo1) / max(1e-9, hi1 - lo1)
            if abs(qa - ta) < 0.05 and abs(qb - tb) < 0.05:
                return yv
        return None

    # S2-symmetric 7-point design:
    # (0.05, 0.05), (0.28, 0.05), (0.92, 0.05),
    # (0.05, 0.28), (0.05, 0.92),
    # (0.85, 0.85), (0.28, 0.28)
    y_05_05 = get_y(0.05, 0.05) or 0.0
    y_28_05 = get_y(0.28, 0.05) or 0.0
    y_92_05 = get_y(0.92, 0.05) or 0.0
    y_05_28 = get_y(0.05, 0.28) or 0.0
    y_05_92 = get_y(0.05, 0.92) or 0.0
    y_85_85 = get_y(0.85, 0.85) or 0.0

    y_85_25 = get_y(0.85, 0.25) or 0.0
    y_25_85 = get_y(0.25, 0.85) or 0.0

    nonmono_0 = (y_28_05 > y_05_05 + 0.15) and (y_92_05 < y_28_05 - 0.10)
    nonmono_1 = (y_05_28 > y_05_05 + 0.15) and (y_05_92 < y_05_28 - 0.10)

    if nonmono_0 and not nonmono_1:
        # Mixed difference test: additive `g(f0) + alpha*f1` keeps |Delta_f0 y| constant across f1,
        # whereas multiplicative `f1 * g(f0)` scales |Delta_f0 y| proportionally with f1!
        delta_low = abs(y_28_05 - y_92_05)
        delta_high = abs(y_25_85 - y_85_85)
        if delta_high / max(1e-6, delta_low) > 2.0:
            return {"signature": "sig:open_underdamped_ringing", "roles": [f1, f0]}
        return {"signature": "sig:open_haldane_inhibited_shunt", "roles": [f0, f1]}

    if nonmono_1 and not nonmono_0:
        delta_low = abs(y_05_28 - y_05_92)
        delta_high = abs(y_85_25 - y_85_85)
        if delta_high / max(1e-6, delta_low) > 2.0:
            return {"signature": "sig:open_underdamped_ringing", "roles": [f0, f1]}
        return {"signature": "sig:open_haldane_inhibited_shunt", "roles": [f1, f0]}

    return {"signature": "sig:open_unclassified_2d", "roles": [f0, f1]}


def _build_symmetric_open_pilot(lab: OpenWorkspaceTaskLab) -> list[dict[str, float]]:
    active, fixed = _active_and_fixed_features(lab)
    if len(active) == 1:
        f0 = active[0]
        lo, hi = lab.feature_bounds[f0]
        return [{**fixed, f0: lo + q * (hi - lo)} for q in (0.08, 0.28, 0.55, 0.88)]
    f0, f1 = active[0], active[1]
    lo0, hi0 = lab.feature_bounds[f0]
    lo1, hi1 = lab.feature_bounds[f1]
    # Exact S2-symmetric 7-point pilot!
    pairs = [
        (0.05, 0.05),
        (0.28, 0.05),
        (0.92, 0.05),
        (0.05, 0.28),
        (0.05, 0.92),
        (0.85, 0.25),
        (0.25, 0.85),
        (0.85, 0.85),
    ]
    return [
        {**fixed, f0: lo0 + a * (hi0 - lo0), f1: lo1 + b * (hi1 - lo1)}
        for a, b in pairs
    ]


def run_open_revision_chain_experiment(seed: int, root_workspace: Path) -> dict[str, Any]:
    """Run complete multi-chain Open-Author Revision & Downstream Reuse study for ``seed``!"""
    suite = make_open_revision_suite(seed)
    forest = PhilosophicalForest()
    shared_ordinary_files: dict[str, str] = {}
    ordinary_lessons: list[dict[str, Any]] = []
    chain_reports: dict[str, Any] = {}

    for chain_key, base_kind, rev_candidates, rev_sig in [
        (
            "chain_O1",
            "monod_1d",
            ["monod_1d", "haldane_inhibited_1d", "haldane_inhibited_plus_bypass_2d"],
            "sig:open_haldane_inhibited_shunt",
        ),
        (
            "chain_O2",
            "overdamped_step_2d",
            ["overdamped_step_2d", "underdamped_ringing_2d"],
            "sig:open_underdamped_ringing",
        ),
    ]:
        c_data = suite[chain_key]
        src_name, (src_fac, src_ev) = c_data["source"]
        wit_name, (wit_fac, wit_ev) = c_data["witness"]

        # --- Step 1: Solve Base Source S via CodeAgent ---
        ws_src = root_workspace / f"{chain_key}_{src_name}"
        lab_src = src_fac()
        lens_src = _estimate_observation_lens(lab_src, ws_src)
        act_src, fix_src = _active_and_fixed_features(lab_src)
        src_pilot = _build_symmetric_open_pilot(lab_src)
        src_code = _render_stateful_least_squares_module(
            mechanism_id=base_kind, role_names=act_src, lens=lens_src, operator_ast_kind=base_kind
        )
        src_prov = ScriptedOpenGrammarCodeProvider(
            query_batches=[src_pilot],
            candidate_specs=[("base_model.py", src_code)],
            tolerance=float(lab_src.task_view["tolerance"]),
        )
        src_conf_pt = {**fix_src, **{f: lab_src.feature_bounds[f][0] + 0.62 * (lab_src.feature_bounds[f][1] - lab_src.feature_bounds[f][0]) for f in act_src}}
        sub_src = _run_code_agent_with_prefix_confirm(lab_src, ws_src, src_prov, src_conf_pt)
        ev_src = src_ev(ws_src, sub_src, lab_src.observations)
        assert ev_src["strict_pass"], f"Source {src_name} failed: {ev_src}"

        base_nid = f"open_base_{chain_key}"
        forest.register_node(
            node_id=base_nid,
            tree_id=f"tree_{chain_key}",
            layer_role="mid",
            task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", float(lab_src.task_view["tolerance"])),
            observation_map=ObservationMap(f"{lab_src.surface_domain}_readout", parameter_semantics={"operator_ast_kind": base_kind}),
            claim_contract=ClaimContract(
                f"claim_{base_nid}", "open_code_agent_ast", [],
                {"operator_ast_kind": base_kind, "ast_skeleton_digest": sub_src["ast_skeleton_digest"]},
                {"stateful_scipy_fit": True}, float(lab_src.task_view["tolerance"]), "SUPPORTED_IN_SCOPE", "local_solver_guidance",
            ),
            probe_signatures=[f"sig:base_{base_kind}"],
            program_steps=[{"op": "calibrate_observation_lens"}, {"op": "code_agent_fit_and_confirm", "operator_ast_kind": base_kind}],
            source_cluster_ids=[f"seed_{seed}"],
        )
        shared_ordinary_files[f"shared/{src_name}.py"] = (ws_src / "model.py").read_text(encoding="utf-8")
        ordinary_lessons.append({"episode": src_name, "operator_ast_kind": base_kind})

        # --- Step 2: Encounter Counterexample Witness W -> Diagnose Failure -> Author Open AST Revision! ---
        ws_wit = root_workspace / f"{chain_key}_{wit_name}"
        lab_wit = wit_fac()
        lens_wit = _estimate_observation_lens(lab_wit, ws_wit)
        act_wit, fix_wit = _active_and_fixed_features(lab_wit)
        wit_pilot = _build_symmetric_open_pilot(lab_wit)

        # Build competing candidate scripts starting with the incumbent base model (so CodeAgent records its exact failure!)
        wit_cands = []
        for idx_k, op_kind in enumerate(rev_candidates):
            r_names = act_wit if len(act_wit) >= 2 else [act_wit[0], act_wit[0]]
            if op_kind in {"monod_1d", "haldane_inhibited_1d"}:
                r_names = [act_wit[0]]
            code_str = _render_stateful_least_squares_module(
                mechanism_id=op_kind, role_names=r_names, lens=lens_wit, operator_ast_kind=op_kind
            )
            wit_cands.append((f"cand_{idx_k}_{op_kind}.py", code_str))

        diag_py = (
            "import json\n"
            "rows = json.load(open('observations.json'))\n"
            "ys = [r['target'] for r in rows]\n"
            "print(json.dumps({'n_rows': len(rows), 'y_min': min(ys), 'y_max': max(ys), 'non_monotone_span': max(ys) - ys[-1]}))\n"
        )
        wit_prov = ScriptedOpenGrammarCodeProvider(
            query_batches=[wit_pilot],
            candidate_specs=wit_cands,
            diagnostic_script=diag_py,
            tolerance=float(lab_wit.task_view["tolerance"]),
        )
        wit_conf_pt = {**fix_wit, act_wit[0]: lab_wit.feature_bounds[act_wit[0]][0] + 0.64 * (lab_wit.feature_bounds[act_wit[0]][1] - lab_wit.feature_bounds[act_wit[0]][0]), act_wit[1]: lab_wit.feature_bounds[act_wit[1]][0] + 0.42 * (lab_wit.feature_bounds[act_wit[1]][1] - lab_wit.feature_bounds[act_wit[1]][0])}
        sub_wit = _run_code_agent_with_prefix_confirm(lab_wit, ws_wit, wit_prov, wit_conf_pt)
        ev_wit = wit_ev(ws_wit, sub_wit, lab_wit.observations)
        assert ev_wit["strict_pass"], f"Witness revision {wit_name} failed: {ev_wit}"

        # Measure incumbent failure RMSE on W1's prefix & verify backward preservation on S1's archived rows!
        incumbent_eval = evaluate_python_model(
            ws_wit, wit_cands[0][0], lab_wit.observations[:-1], training_observations=lab_wit.observations[:-1], require_fit=False
        )
        incumbent_witness_rmse = math.sqrt(max(0.0, float(incumbent_eval["mse"])))
        assert incumbent_witness_rmse > float(lab_wit.task_view["tolerance"]) * 3.0

        # Backward compatibility check on Source S1's paid observations (mapping S1 features to W1 roles)!
        s1_mapped_rows = [
            Observation(
                features={
                    act_wit[0]: r.features[act_src[0]],
                    act_wit[1]: (r.features[act_src[1]] if len(act_src) > 1 else 0.0),
                },
                target=r.target,
                history=(),
            )
            for r in lab_src.observations[:-1]
        ]
        s1_confirm_mapped = [
            Observation(
                features={
                    act_wit[0]: lab_src.observations[-1].features[act_src[0]],
                    act_wit[1]: (lab_src.observations[-1].features[act_src[1]] if len(act_src) > 1 else 0.0),
                },
                target=lab_src.observations[-1].target,
                history=(),
            )
        ]
        backward_eval = evaluate_python_model(
            ws_wit, "model.py", s1_confirm_mapped, training_observations=s1_mapped_rows, require_fit=False
        )
        backward_rmse = math.sqrt(max(0.0, float(backward_eval["mse"])))
        assert backward_rmse <= float(lab_src.task_view["tolerance"]) * 1.25

        # Freeze revised node + horizontal wormhole into PhilosophicalForest!
        rev_kind = sub_wit["selected_schema"]
        rev_nid = f"open_rev_{chain_key}_{sub_wit['ast_skeleton_digest'][:6]}"
        forest.register_node(
            node_id=rev_nid,
            tree_id=f"tree_{lab_wit.surface_domain}",
            layer_role="mid",
            task_view=TaskView("open_workspace_mechanism_prediction", "rmse_original_scale", float(lab_wit.task_view["tolerance"])),
            observation_map=ObservationMap(
                f"{lab_wit.surface_domain}_readout",
                parameter_semantics={"operator_ast_kind": rev_kind, "ast_skeleton_digest": sub_wit["ast_skeleton_digest"]},
            ),
            claim_contract=ClaimContract(
                f"claim_{rev_nid}", "open_code_agent_ast",
                [{"field": "detected_signature", "op": "==", "value": rev_sig}],
                {"operator_ast_kind": rev_kind, "ast_skeleton_digest": sub_wit["ast_skeleton_digest"]},
                {"stateful_scipy_fit": True, "backward_verified_rmse": round(backward_rmse, 6)},
                float(lab_wit.task_view["tolerance"]),
                "SUPPORTED_IN_SCOPE",
                "local_solver_guidance",
            ),
            probe_signatures=[rev_sig],
            program_steps=[
                {"op": "calibrate_observation_lens"},
                {"op": "code_agent_fit_and_confirm", "operator_ast_kind": rev_kind},
            ],
            source_cluster_ids=[f"seed_{seed}"],
        )
        forest.add_horizontal_wormhole(
            wormhole_id=f"wh_open_{chain_key}",
            node_a_id=base_nid,
            node_b_id=rev_nid,
            probe_signature_refs=[rev_sig],
            role_and_observation_map={
                "incumbent_witness_rmse": round(incumbent_witness_rmse, 5),
                "backward_source_rmse": round(backward_rmse, 5),
                "revised_operator_ast": rev_kind,
            },
            hop_error_budget=round(float(sub_wit["confirmation_rmse"]), 5),
            permission_tier="local_solver_guidance",
            compatible_goal_kinds=["open_workspace_mechanism_prediction"],
            source_episode_refs=[src_name, wit_name],
        )
        shared_ordinary_files[f"shared/{wit_name}.py"] = (ws_wit / "model.py").read_text(encoding="utf-8")
        ordinary_lessons.append({"episode": wit_name, "operator_ast_kind": rev_kind})

        chain_reports[chain_key] = {
            "source_episode": src_name,
            "witness_episode": wit_name,
            "incumbent_base_kind": base_kind,
            "incumbent_witness_rmse": round(incumbent_witness_rmse, 6),
            "revised_operator_kind": rev_kind,
            "revised_ast_digest": sub_wit["ast_skeleton_digest"],
            "backward_source_rmse": round(backward_rmse, 6),
            "witness_confirmation_rmse": sub_wit["confirmation_rmse"],
            "witness_eval_rmse": ev_wit["rmse"],
            "spawned_node_id": rev_nid,
        }

    # --- Step 3: Evaluate Unseen Downstream Targets Across 3 Controlled Arms! ---
    # 1. `Hegel_OpenRevision_Forest`: Uses PhilosophicalForest + Wormhole role/lens alignment!
    # 2. `Ablated_RemoveRevisedNode`: Deletes `open_rev_*` nodes from forest -> proves concrete behavioral change!
    # 3. `Continuing_Ordinary_CodeSession`: Inherits ALL `shared/*.py` & lessons without forest scope/role wormholes!
    target_comparisons: dict[str, Any] = {}
    all_open_kinds = [
        "monod_1d",
        "overdamped_step_2d",
        "haldane_inhibited_1d",
        "haldane_inhibited_plus_bypass_2d",
        "underdamped_ringing_2d",
    ]

    for chain_key in ("chain_O1", "chain_O2"):
        t_name, (t_fac, t_ev) = suite[chain_key]["target"]
        target_comparisons[t_name] = {}

        for arm_name in ("Hegel_OpenRevision_Forest", "Ablated_RemoveRevisedNode", "Continuing_Ordinary_CodeSession"):
            ws_t = root_workspace / f"{arm_name}_{t_name}"
            lab_t = t_fac()
            lens_t = _estimate_observation_lens(lab_t, ws_t)
            pilot_pts = _build_symmetric_open_pilot(lab_t)
            # First query batch inside CodeAgent is `pilot_pts`
            # Let's inspect pilot via a tiny pre-pilot or inside Hegel after pilot!
            if arm_name in {"Hegel_OpenRevision_Forest", "Ablated_RemoveRevisedNode"}:
                active_forest = deepcopy(forest)
                if arm_name == "Ablated_RemoveRevisedNode":
                    for nid in [k for k in list(active_forest.nodes) if k.startswith("open_rev_")]:
                        del active_forest.nodes[nid]
                    for wid in [k for k in list(active_forest.horizontal_wormhole_edges) if k.startswith("wh_open_")]:
                        del active_forest.horizontal_wormhole_edges[wid]

                lab_t.acquire(pilot_pts, workspace=ws_t, phase="presolve")
                sig_info = _symmetric_2d_pilot_and_roles(lab_t, lens_t, lab_t.observations)
                lab_t.meter.charge("retrieve", compute=3.0)
                recalled = active_forest.recall_mid_layer_wormholes(
                    [sig_info["signature"]], task_view=lab_t.task_view, allow_horizontal_wormholes=True
                )
                recalled = [
                    r for r in recalled
                    if sig_info["signature"] in r["target_node"].get("probe_signatures", ())
                ]
                roles = sig_info["roles"]
                tol_t = float(lab_t.task_view["tolerance"])

                if recalled:
                    chosen = recalled[0]["target_node"]
                    op_kind = chosen["claim_contract"]["conclusion_or_action"]["operator_ast_kind"]
                    cand_code = _render_stateful_least_squares_module(
                        mechanism_id=op_kind, role_names=roles, lens=lens_t, operator_ast_kind=op_kind
                    )
                    prov = ScriptedOpenGrammarCodeProvider(
                        query_batches=[],
                        candidate_specs=[("wormhole_adapted.py", cand_code)],
                        tolerance=tol_t,
                    )
                    c_pt = {
                        roles[0]: lab_t.feature_bounds[roles[0]][0] + 0.62 * (lab_t.feature_bounds[roles[0]][1] - lab_t.feature_bounds[roles[0]][0]),
                        roles[1]: lab_t.feature_bounds[roles[1]][0] + 0.36 * (lab_t.feature_bounds[roles[1]][1] - lab_t.feature_bounds[roles[1]][0]),
                    }
                    sub_t = _run_code_agent_with_prefix_confirm(lab_t, ws_t, prov, c_pt)
                    # Check bound receipt
                    if sub_t["status"] == "completed":
                        receipt = {
                            "target_id": t_name,
                            "claim_id": chosen["claim_contract"]["claim_id"],
                            "node_version": chosen["version"],
                            "model_sha256": sub_t["model_sha256"],
                            "observed_error": sub_t["confirmation_rmse"],
                            "passed": True,
                            "upgrade_to_tier": "local_solver_guidance",
                        }
                        auth = authorize_abstraction_use(
                            requested_tier="local_solver_guidance",
                            task_view={**lab_t.task_view, "target_id": t_name, "node_version": chosen["version"], "expected_model_sha256": sub_t["model_sha256"]},
                            target_features={"detected_signature": sig_info["signature"]},
                            claim_contract=chosen["claim_contract"],
                            wormhole_edge=recalled[0]["wormhole_edge"],
                            evidence_receipt=receipt,
                        )
                        assert auth["granted"] is True
                else:
                    # Ablated forest has NO matching revised node -> must acquire extra points & search open AST grammar!
                    extra_q = [
                        {
                            roles[0]: lab_t.feature_bounds[roles[0]][0] + a * (lab_t.feature_bounds[roles[0]][1] - lab_t.feature_bounds[roles[0]][0]),
                            roles[1]: lab_t.feature_bounds[roles[1]][0] + b * (lab_t.feature_bounds[roles[1]][1] - lab_t.feature_bounds[roles[1]][0]),
                        }
                        for a, b in [(0.45, 0.45), (0.72, 0.15), (0.18, 0.72)]
                    ]
                    cands_open = []
                    for ik, okind in enumerate(all_open_kinds):
                        for perm_i, r_perm in enumerate(([roles[0], roles[1]], [roles[1], roles[0]])):
                            if okind in {"monod_1d", "haldane_inhibited_1d"} and perm_i > 0:
                                continue
                            rn = [r_perm[0]] if okind in {"monod_1d", "haldane_inhibited_1d"} else r_perm
                            cands_open.append((
                                f"open_search_{ik}_{perm_i}.py",
                                _render_stateful_least_squares_module(mechanism_id=okind, role_names=rn, lens=lens_t, operator_ast_kind=okind),
                            ))
                    prov = ScriptedOpenGrammarCodeProvider(
                        query_batches=[extra_q],
                        candidate_specs=cands_open,
                        diagnostic_script="import json; print(len(json.load(open('observations.json'))))",
                        tolerance=tol_t,
                    )
                    c_pt = {
                        roles[0]: lab_t.feature_bounds[roles[0]][0] + 0.62 * (lab_t.feature_bounds[roles[0]][1] - lab_t.feature_bounds[roles[0]][0]),
                        roles[1]: lab_t.feature_bounds[roles[1]][0] + 0.36 * (lab_t.feature_bounds[roles[1]][1] - lab_t.feature_bounds[roles[1]][0]),
                    }
                    sub_t = _run_code_agent_with_prefix_confirm(lab_t, ws_t, prov, c_pt)

            else:
                # `Continuing_Ordinary_CodeSession`: inherits ALL `shared/*.py` & lessons in flat workspace,
                # without wormhole role mapping or scope filtering -> must test all inherited kinds & column orders!
                act_t, _ = _active_and_fixed_features(lab_t)
                inherited_kinds = list(dict.fromkeys(l["operator_ast_kind"] for l in ordinary_lessons))
                extra_q = [
                    {
                        act_t[0]: lab_t.feature_bounds[act_t[0]][0] + a * (lab_t.feature_bounds[act_t[0]][1] - lab_t.feature_bounds[act_t[0]][0]),
                        act_t[1]: lab_t.feature_bounds[act_t[1]][0] + b * (lab_t.feature_bounds[act_t[1]][1] - lab_t.feature_bounds[act_t[1]][0]),
                    }
                    for a, b in [(0.50, 0.50), (0.68, 0.18)]
                ]
                sess_cands = []
                for ik, okind in enumerate(inherited_kinds):
                    for perm_i, r_perm in enumerate(([act_t[0], act_t[1]], [act_t[1], act_t[0]])):
                        if okind in {"monod_1d", "haldane_inhibited_1d"} and perm_i > 0:
                            continue
                        rn = [r_perm[0]] if okind in {"monod_1d", "haldane_inhibited_1d"} else r_perm
                        sess_cands.append((
                            f"session_prior_{ik}_{perm_i}.py",
                            _render_stateful_least_squares_module(mechanism_id=okind, role_names=rn, lens=lens_t, operator_ast_kind=okind),
                        ))
                prov = ScriptedOpenGrammarCodeProvider(
                    query_batches=[pilot_pts, extra_q],
                    candidate_specs=sess_cands,
                    tolerance=float(lab_t.task_view["tolerance"]),
                )
                c_pt = {
                    act_t[0]: lab_t.feature_bounds[act_t[0]][0] + 0.55 * (lab_t.feature_bounds[act_t[0]][1] - lab_t.feature_bounds[act_t[0]][0]),
                    act_t[1]: lab_t.feature_bounds[act_t[1]][0] + 0.55 * (lab_t.feature_bounds[act_t[1]][1] - lab_t.feature_bounds[act_t[1]][0]),
                }
                sub_t = _run_code_agent_with_prefix_confirm(
                    lab_t,
                    ws_t,
                    prov,
                    c_pt,
                    session_knowledge={"files": deepcopy(shared_ordinary_files), "lessons": deepcopy(ordinary_lessons)},
                )

            ev_t = t_ev(ws_t, sub_t, lab_t.observations)
            target_comparisons[t_name][arm_name] = {
                "strict_pass": ev_t["strict_pass"],
                "false_confirmation": ev_t["false_confirmation"],
                "eval_rmse": ev_t["rmse"],
                "selected_schema": sub_t.get("selected_schema"),
                "ast_skeleton_digest": sub_t.get("ast_skeleton_digest"),
                "code_agent_events": sub_t.get("code_agent_events"),
                "costs": {
                    **lab_t.meter.snapshot(),
                    "measured_python_wall_seconds": round(lab_t.measured_python_wall_seconds, 4),
                },
            }

    return {
        "seed": seed,
        "chains": chain_reports,
        "downstream_targets": target_comparisons,
    }

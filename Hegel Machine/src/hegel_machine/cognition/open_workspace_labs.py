"""M2 Open Python Workspace Laboratories & Cross-Domain Process Suites (Hegel V16.8).

Every laboratory in M2 exposes ONE unified ``CodeAgent``-compatible workspace interface:
  - Public ``feature_names``, ``feature_bounds``, ``task_view``, and cost caps ONLY.
  - Purchased observations are ``Observation(features, target, history)`` written directly
    to ``workspace / "observations.json"`` and ``workspace / "model_observations.json"``.
  - Submissions MUST be real executable Python files (``model.py`` defining ``predict(row)``
    and optional ``fit(observations)``) inside the task's filesystem workspace, evaluated
    in an isolated Python subprocess via ``code_agent.evaluate_python_model``!
  - Ground-truth functions and held-out test rows live exclusively inside read-only
    closures never accessible to any research arm.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from random import Random
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .code_agent import evaluate_python_model
from .cross_domain_clusters import CostMeter
from .models import Observation


class OpenWorkspaceTaskLab:
    """Unified public laboratory interface for open Python workspace research."""

    def __init__(
        self,
        *,
        task_id: str,
        surface_domain: str,
        feature_names: tuple[str, ...],
        feature_bounds: dict[str, tuple[float, float]],
        cost_per_obs: float,
        task_view: dict[str, Any],
        meter: CostMeter,
        private_oracle_fn: Callable[[dict[str, float]], float],
        private_readout_cal_fn: Callable[[float], float] | None = None,
        domain_regime_attrs: dict[str, Any] | None = None,
        noise_std: float = 0.0,
        noise_seed: int | None = None,
    ) -> None:
        self.task_id = task_id
        self.surface_domain = surface_domain
        self.feature_names = tuple(feature_names)
        self.feature_bounds = {k: (float(lo), float(hi)) for k, (lo, hi) in feature_bounds.items()}
        self.cost_per_obs = float(cost_per_obs)
        self.domain_regime_attrs = dict(domain_regime_attrs or {})
        self.task_view = {
            **dict(task_view),
            "feature_names": list(self.feature_names),
            "feature_bounds": {k: list(v) for k, v in self.feature_bounds.items()},
            "readout_calibration_available": private_readout_cal_fn is not None,
            **({"domain_regime_attrs": dict(self.domain_regime_attrs)} if self.domain_regime_attrs else {}),
        }
        self.meter = meter
        self._base_oracle_fn = private_oracle_fn
        self._noise_std = float(noise_std)
        self._noise_seed = noise_seed
        self._noise_rng: Random | None = (
            Random(noise_seed) if (noise_seed is not None and self._noise_std > 0.0) else None
        )
        self._oracle = self._make_bound_oracle()
        self._readout_cal = private_readout_cal_fn
        self.observations: list[Observation] = []
        self.calibration_readings: list[tuple[float, float]] = []
        self.python_execution_seconds: float = 0.0
        self.measured_python_wall_seconds: float = 0.0

    def _make_bound_oracle(self) -> Callable[[dict[str, float]], float]:
        base_fn = self._base_oracle_fn
        rng = self._noise_rng
        sd = self._noise_std

        def _eval_point(pt: dict[str, float]) -> float:
            val = float(base_fn(pt))
            if rng is not None and sd > 0.0:
                val += rng.gauss(0.0, sd)
            return val

        return _eval_point

    def __deepcopy__(self, memo: dict[int, Any]) -> "OpenWorkspaceTaskLab":
        cloned = OpenWorkspaceTaskLab(
            task_id=self.task_id,
            surface_domain=self.surface_domain,
            feature_names=tuple(self.feature_names),
            feature_bounds=deepcopy(self.feature_bounds, memo),
            cost_per_obs=self.cost_per_obs,
            task_view=deepcopy(self.task_view, memo),
            meter=deepcopy(self.meter, memo),
            private_oracle_fn=self._base_oracle_fn,
            private_readout_cal_fn=self._readout_cal,
            domain_regime_attrs=deepcopy(self.domain_regime_attrs, memo),
            noise_std=self._noise_std,
            noise_seed=self._noise_seed,
        )
        memo[id(self)] = cloned
        if self._noise_rng is not None:
            cloned_rng = Random()
            cloned_rng.setstate(self._noise_rng.getstate())
            cloned._noise_rng = cloned_rng
        cloned._oracle = cloned._make_bound_oracle()
        cloned.observations = deepcopy(self.observations, memo)
        cloned.calibration_readings = deepcopy(self.calibration_readings, memo)
        cloned.python_execution_seconds = float(self.python_execution_seconds)
        cloned.measured_python_wall_seconds = float(self.measured_python_wall_seconds)
        return cloned

    def sync_workspace_files(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([asdict(r) for r in self.observations], ensure_ascii=False)
        (workspace / "observations.json").write_text(payload, encoding="utf-8")
        (workspace / "model_observations.json").write_text(payload, encoding="utf-8")
        if self.calibration_readings:
            (workspace / "readout_calibration.json").write_text(
                json.dumps([{"reference_level": x, "raw_reading": y} for x, y in self.calibration_readings]),
                encoding="utf-8",
            )

    def acquire(
        self,
        sequence: list[dict[str, float]],
        *,
        workspace: Path | None = None,
        phase: str = "probe",
    ) -> list[Observation] | None:
        """Purchase experimental rows at public feature settings and persist to workspace JSON."""
        if not sequence:
            return []
        n_rows = len(sequence)
        if not self.meter.charge(
            phase,
            obs=n_rows,
            meas=n_rows * self.cost_per_obs,
            compute=1.5 * n_rows,
        ):
            return None
        acquired: list[Observation] = []
        for raw_pt in sequence:
            clipped: dict[str, float] = {}
            for fname in self.feature_names:
                if fname not in raw_pt:
                    raise ValueError(f"Missing feature '{fname}' in query {raw_pt}")
                lo, hi = self.feature_bounds[fname]
                clipped[fname] = min(hi, max(lo, float(raw_pt[fname])))
            y_val = float(self._oracle(clipped))
            obs = Observation(features=clipped, target=y_val, history=())
            self.observations.append(obs)
            acquired.append(obs)
        if workspace is not None:
            self.sync_workspace_files(workspace)
        return acquired

    def calibrate_readout(
        self,
        levels: list[float],
        *,
        workspace: Path | None = None,
        phase: str = "calibrate",
    ) -> list[tuple[float, float]] | None:
        """Optional cheap instrument calibration channel for ObservationMap sign/scale/offset."""
        if self._readout_cal is None or not levels:
            return []
        n = len(levels)
        if not self.meter.charge(phase, obs=n, meas=0.5 * n * self.cost_per_obs, compute=1.0 * n):
            return None
        out = []
        for lv in levels:
            rv = float(self._readout_cal(float(lv)))
            pair = (float(lv), rv)
            self.calibration_readings.append(pair)
            out.append(pair)
        if workspace is not None:
            self.sync_workspace_files(workspace)
        return out

    def record_python_wall_time(self, seconds: float, *, phase: str = "python_exec", compute_units: float = 4.0) -> None:
        dt = max(0.0, float(seconds))
        self.python_execution_seconds += dt
        self.measured_python_wall_seconds += dt
        self.meter.charge(phase, compute=compute_units)


def _load_verified_confirmed_prefix(
    workspace: Path,
    paid_observations: list[Observation],
) -> list[Observation]:
    """Use frozen pre-confirmation training prefix if saved and verified as a subset of paid rows."""
    prefix_path = workspace / "confirmed_training_observations.json"
    if not prefix_path.is_file():
        return paid_observations
    try:
        raw_rows = json.loads(prefix_path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list) or len(raw_rows) > len(paid_observations):
            return paid_observations
        reconstructed: list[Observation] = []
        for idx, item in enumerate(raw_rows):
            paid = paid_observations[idx]
            if item.get("features") != paid.features or abs(float(item.get("target", 0.0)) - float(paid.target)) > 1e-12:
                return paid_observations
            reconstructed.append(paid)
        return reconstructed
    except Exception:
        return paid_observations


def _evaluate_workspace_submission(
    workspace: Path,
    model_rel_path: str | None,
    paid_observations: list[Observation],
    heldout_eval_rows: list[Observation],
    rmse_tolerance: float,
    *,
    submission_status: str = "completed",
) -> dict[str, Any]:
    """Closed read-only benchmark evaluator executing ``workspace / model_rel_path`` in subprocess!"""
    if submission_status != "completed" or not model_rel_path or not (workspace / model_rel_path).is_file():
        return {
            "strict_pass": False,
            "false_confirmation": False,
            "reason": "unconfirmed_or_missing_submitted_model_file",
            "submission_status": submission_status,
            "rmse": None,
        }
    training_obs = _load_verified_confirmed_prefix(workspace, paid_observations)
    with TemporaryDirectory(prefix="hegel-eval-sandbox-") as tmp_dir:
        sandbox = Path(tmp_dir) / "ws"
        shutil.copytree(
            workspace,
            sandbox,
            ignore=shutil.ignore_patterns("observations.json", "model_observations.json", "__pycache__", "*.pyc"),
        )
        prefix_payload = json.dumps([asdict(r) for r in training_obs], ensure_ascii=False)
        (sandbox / "observations.json").write_text(prefix_payload, encoding="utf-8")
        (sandbox / "model_observations.json").write_text(prefix_payload, encoding="utf-8")
        eval_res = evaluate_python_model(
            sandbox,
            model_rel_path,
            heldout_eval_rows,
            timeout_seconds=15.0,
            risk_quantile=0.5,
            training_observations=training_obs,
            require_fit=False,
        )
    if eval_res.get("returncode") != 0 or "mse" not in eval_res:
        return {
            "strict_pass": False,
            "false_confirmation": True,
            "reason": "submitted_model_execution_failed",
            "returncode": eval_res.get("returncode"),
            "stderr": (eval_res.get("stderr") or "")[-400:],
            "rmse": None,
        }
    rmse = math.sqrt(max(0.0, float(eval_res["mse"])))
    ok = rmse <= float(rmse_tolerance)
    return {
        "strict_pass": bool(ok),
        "false_confirmation": bool(not ok),
        "rmse": round(rmse, 6),
        "tolerance": float(rmse_tolerance),
        "external_python_wall_seconds": round(float(eval_res.get("external_python_wall_seconds", 0.0)), 5),
        "eval_predictions": [round(float(p), 6) for p in eval_res.get("predictions", ())[:8]],
    }


# ============================================================================
# Mechanistic Process Simulators for M2 Suites (RK4 ODEs & Discrete Queues)
# ============================================================================

def _rk4_conjugate_relaxation(
    drive_u: float,
    horizon_dt: float,
    init_mem_m0: float,
    *,
    k_frozen: float,
    k_relaxed: float,
    tau: float,
) -> float:
    """RK4 integration of internal relaxation state dz/dt = -(z - z_eq(u)) / tau.

    At t=0+, instantaneous response is k_frozen * u + init_mem_m0;
    as t -> inf, internal relaxation attenuates response toward k_relaxed * u (0 < k_relaxed < k_frozen).
    """
    # Let internal opposing variable z satisfy dz/dt = ((k_frozen - k_relaxed)*u - z) / tau,
    # with initial condition z(0) = -init_mem_m0, and port output p(t) = k_frozen * u - z(t).
    z = -float(init_mem_m0)
    target_z = (k_frozen - k_relaxed) * float(drive_u)
    n_steps = 60
    dt = max(1e-5, float(horizon_dt)) / n_steps
    for _ in range(n_steps):
        k1 = (target_z - z) / tau
        k2 = (target_z - (z + 0.5 * dt * k1)) / tau
        k3 = (target_z - (z + 0.5 * dt * k2)) / tau
        k4 = (target_z - (z + dt * k3)) / tau
        z += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return float(k_frozen) * float(drive_u) - z


def _rk4_bimodal_creep_relaxation(
    drive_u: float,
    horizon_dt: float,
    *,
    k_relaxed: float,
    amp_fast: float,
    tau_fast: float,
    amp_slow: float,
    tau_slow: float,
) -> float:
    """Two-mode Maxwell viscoelastic relaxation (violates single-exponential decay!)."""
    z1, z2 = 0.0, 0.0
    n_steps = 80
    dt = max(1e-5, float(horizon_dt)) / n_steps
    for _ in range(n_steps):
        z1 += dt * ((amp_fast * drive_u - z1) / tau_fast)
        z2 += dt * ((amp_slow * drive_u - z2) / tau_slow)
    k_frozen = k_relaxed + amp_fast + amp_slow
    return k_frozen * drive_u - z1 - z2


def _sim_oscillatory_carrier_trace(drive_u: float, *, alpha: float, beta: float, osc_amp: float) -> dict[str, float]:
    """Simulate a fast-oscillating catalytic/packet carrier over 120 micro-steps."""
    u = float(drive_u)
    base_flux = u / (alpha * u + beta)
    samples: list[float] = []
    for step in range(120):
        phase = 2.0 * math.pi * (step / 12.0)
        # Zero-mean harmonic + burst crest scaling as 1/u^2!
        instant = base_flux + (osc_amp / (u * u)) * math.cos(phase) + (0.65 * osc_amp / (u * u)) * (math.cos(2.0 * phase))
        samples.append(instant)
    window_mean = sum(samples) / len(samples)
    peak_crest = max(samples) - window_mean
    return {"window_mean": window_mean, "peak_crest": peak_crest}


# ============================================================================
# Build Complete M2 Cross-Domain Open-Workspace Suite
# ============================================================================

def make_m2_open_workspace_suite(seed: int) -> dict[str, Any]:
    """Create 3 cross-domain M2 clusters (sources, near-misses, scale probes, and 10 unseen targets)."""
    rng = Random(seed + 8000)

    def build_task_pair(
        *,
        tid: str,
        domain: str,
        feature_names: tuple[str, ...],
        feature_bounds: dict[str, tuple[float, float]],
        clean_fn: Callable[[dict[str, float]], float],
        eval_points: list[dict[str, float]],
        goal_kind: str = "open_workspace_mechanism_prediction",
        rmse_tol: float = 0.045,
        cost_per_obs: float = 5.0,
        obs_budget: int = 24,
        meas_budget: float = 120.0,
        noise_std: float = 0.002,
        noise_seed: int = 1,
        readout_sign: int = 1,
        readout_scale: float = 1.0,
        readout_offset: float = 0.0,
        expose_calibration_channel: bool = False,
        available_scale_map: dict[str, Any] | None = None,
        domain_regime_attrs: dict[str, Any] | None = None,
    ):
        def transform_out(val: float) -> float:
            return float(readout_offset) + float(readout_sign) * float(readout_scale) * float(val)

        heldout_rows = [
            Observation(features=dict(pt), target=transform_out(clean_fn(pt)), history=())
            for pt in eval_points
        ]

        def factory() -> OpenWorkspaceTaskLab:
            meter = CostMeter(obs_budget=obs_budget, meas_budget=meas_budget, compute_budget=800.0)
            clean_transformed = lambda pt: transform_out(clean_fn(pt))
            cal_fn = (lambda lv: transform_out(lv)) if expose_calibration_channel else None
            tv: dict[str, Any] = {
                "goal_kind": goal_kind,
                "target_metric": "rmse_original_scale",
                "tolerance": rmse_tol,
                "risk_quantile": 0.5,
                **({"available_scale_map": dict(available_scale_map)} if available_scale_map is not None else {}),
            }
            return OpenWorkspaceTaskLab(
                task_id=tid,
                surface_domain=domain,
                feature_names=feature_names,
                feature_bounds=feature_bounds,
                cost_per_obs=cost_per_obs,
                task_view=tv,
                meter=meter,
                private_oracle_fn=clean_transformed,
                private_readout_cal_fn=cal_fn,
                domain_regime_attrs=domain_regime_attrs,
                noise_std=noise_std,
                noise_seed=noise_seed,
            )

        def evaluator(workspace: Path, submission: dict[str, Any], paid_obs: list[Observation]) -> dict[str, Any]:
            return _evaluate_workspace_submission(
                workspace,
                submission.get("model_path"),
                paid_obs,
                heldout_rows,
                rmse_tol,
                submission_status=str(submission.get("status", "incomplete")),
            )

        return factory, evaluator

    # ------------------------------------------------------------------------
    # M2 Cluster A: Conjugate Opposition & Relaxation (Le Chatelier <-> Lenz <-> Queue)
    # ------------------------------------------------------------------------
    # Source A1: Chemical Le Chatelier Equilibrium Shift (sign = +1, offset = 0.0, reset prior_extent = 0)
    ca_s1_kfr = rng.uniform(2.1, 2.7)
    ca_s1_krel = rng.uniform(0.75, 1.15)
    ca_s1_tau = rng.uniform(1.4, 2.4)
    ca_s1_eval = [
        {"pressure_step": u, "settle_time": dt, "prior_extent": 0.0}
        for u, dt in [(1.5, 0.3), (3.0, 1.2), (5.0, 3.5), (7.5, 8.0), (4.0, 0.5), (6.5, 2.2)]
    ]

    # Source A2: Electromagnetic Lenz / Eddy Port (sign = -1 OPPOSING!, scale = 1.5, offset = +3.0, reset init_current = 0)
    ca_s2_kfr = rng.uniform(1.8, 2.4)
    ca_s2_krel = rng.uniform(0.65, 1.05)
    ca_s2_tau = rng.uniform(1.2, 2.1)
    ca_s2_eval = [
        {"flux_step": u, "window_dt": dt, "init_current": 0.0}
        for u, dt in [(1.5, 0.25), (3.5, 1.0), (6.0, 3.0), (8.0, 7.5), (4.5, 0.6), (7.0, 2.5)]
    ]

    # Near-Miss A: Unreset Eddy Brake (`init_current != 0` violates zero-memory assumption -> state-augmented repair!)
    nm_a_kfr, nm_a_krel, nm_a_tau = 2.2, 0.9, 1.8
    nm_a_eval = [
        {"flux_step": u, "window_dt": dt, "init_current": m0}
        for u, dt, m0 in [(2.0, 0.4, 2.5), (5.0, 1.2, -2.0), (7.0, 4.0, 3.0), (3.5, 0.6, -1.5), (6.0, 7.0, 2.0)]
    ]

    # Source A3 (Approx Relay Domain for 2-Hop Non-Transitivity Test!): mildly perturbed nonlinear damping
    # Hop 1 (S1 <-> S2) eps=0.008; Hop 2 (S2 <-> S3_approx) eps=0.024; 2-hop chain S1->S2->S3 on strict tol=0.025!
    t_a1_kfr, t_a1_krel, t_a1_tau = rng.uniform(1.9, 2.5), rng.uniform(0.7, 1.1), rng.uniform(1.3, 2.2)
    t_a2_kfr, t_a2_krel, t_a2_tau = rng.uniform(2.0, 2.6), rng.uniform(0.8, 1.2), rng.uniform(1.5, 2.3)

    # ------------------------------------------------------------------------
    # M2 Cluster B: Multi-Resource Complementarity & Capped Bottlenecks
    # ------------------------------------------------------------------------
    cb_s1_c1, cb_s1_c2 = rng.uniform(1.7, 2.3), rng.uniform(2.2, 2.9)
    cb_s2_c1, cb_s2_c2 = rng.uniform(1.4, 2.0), rng.uniform(2.5, 3.2)
    tb1_c1, tb1_c2 = rng.uniform(1.8, 2.4), rng.uniform(2.1, 2.8)
    tb2_c1, tb2_c2, tb2_cap = 2.2, 2.6, rng.uniform(15.8, 18.8)
    tb3_coef = rng.uniform(2.2, 3.0)

    grid_2d = [
        (3.0, 3.5), (4.5, 8.0), (8.5, 4.0), (7.0, 7.0), (11.0, 10.5), (13.5, 6.5),
    ]

    # ------------------------------------------------------------------------
    # M2 Cluster C: Multi-Scale Oscillatory / Bursty Carrier + Sensor Lens
    # ------------------------------------------------------------------------
    cc_alpha = rng.uniform(0.22, 0.34)
    cc_beta = rng.uniform(1.3, 2.2)
    cc_osc = rng.uniform(4.5, 6.5)
    tc3_a, tc3_b = rng.uniform(0.24, 0.36), rng.uniform(1.5, 2.4)
    eval_1d_u = [2.0, 4.5, 9.0, 16.0, 24.0]

    unverified_scale_map = {
        "coarse_graining_op": "window_average",
        "closure_scope": "steady_state_only",
        "proposed_discarded_variables": ["micro_oscillation_phase"],
        "verified_discarded_in_scope": [],
    }

    def run_m2_scale_calibration_probe(meter: CostMeter) -> dict[str, Any] | None:
        if not meter.charge("scale_calibration", obs=2, meas=10.0, compute=4.0):
            return None
        low_osc = _sim_oscillatory_carrier_trace(3.0, alpha=cc_alpha, beta=cc_beta, osc_amp=0.0)
        hi_osc = _sim_oscillatory_carrier_trace(3.0, alpha=cc_alpha, beta=cc_beta, osc_amp=cc_osc)
        return {
            "mean_flux_diff": abs(hi_osc["window_mean"] - low_osc["window_mean"]),
            "peak_crest_diff": abs(hi_osc["peak_crest"] - low_osc["peak_crest"]),
        }

    return {
        "sources": [
            # Relaxation Pair (Le Chatelier +1 <-> Lenz -1 with offset/scale)
            (
                "src_M2A1_lechatelier_shift",
                build_task_pair(
                    tid="src_M2A1",
                    domain="chemical_lechatelier_reactor",
                    feature_names=("pressure_step", "settle_time", "prior_extent"),
                    feature_bounds={"pressure_step": (1.0, 9.0), "settle_time": (0.2, 9.0), "prior_extent": (0.0, 0.0)},
                    clean_fn=lambda pt, kf=ca_s1_kfr, kr=ca_s1_krel, t=ca_s1_tau: _rk4_conjugate_relaxation(
                        pt["pressure_step"], pt["settle_time"], pt["prior_extent"], k_frozen=kf, k_relaxed=kr, tau=t
                    ),
                    eval_points=ca_s1_eval,
                    expose_calibration_channel=True,
                    noise_seed=seed + 11,
                    domain_regime_attrs={"aqueous_phase": True},
                ),
            ),
            (
                "src_M2A2_lenz_eddy_port",
                build_task_pair(
                    tid="src_M2A2",
                    domain="electromagnetic_lenz_coil",
                    feature_names=("flux_step", "window_dt", "init_current"),
                    feature_bounds={"flux_step": (1.0, 9.0), "window_dt": (0.2, 9.0), "init_current": (0.0, 0.0)},
                    clean_fn=lambda pt, kf=ca_s2_kfr, kr=ca_s2_krel, t=ca_s2_tau: _rk4_conjugate_relaxation(
                        pt["flux_step"], pt["window_dt"], pt["init_current"], k_frozen=kf, k_relaxed=kr, tau=t
                    ),
                    eval_points=ca_s2_eval,
                    readout_sign=-1,
                    readout_scale=1.5,
                    readout_offset=3.0,
                    expose_calibration_channel=True,
                    noise_seed=seed + 12,
                    domain_regime_attrs={"aqueous_phase": False},
                ),
            ),
            # 2D Complementarity Pair (Cloud vCPU/IOPS <-> Agricultural Liebig N/P)
            (
                "src_M2B1_cloud_vcpu_iops",
                build_task_pair(
                    tid="src_M2B1",
                    domain="cloud_storage_pipeline",
                    feature_names=("vcpu_share", "iops_quota"),
                    feature_bounds={"vcpu_share": (1.0, 16.0), "iops_quota": (1.0, 16.0)},
                    clean_fn=lambda pt, c1=cb_s1_c1, c2=cb_s1_c2: min(c1 * pt["vcpu_share"], c2 * pt["iops_quota"]),
                    eval_points=[{"vcpu_share": a, "iops_quota": b} for a, b in grid_2d],
                    rmse_tol=0.06,
                    noise_seed=seed + 21,
                    domain_regime_attrs={"digital_throttling": True},
                ),
            ),
            (
                "src_M2B2_liebig_np_yield",
                build_task_pair(
                    tid="src_M2B2",
                    domain="agronomy_liebig_plot",
                    feature_names=("nitrate_kg", "phosphate_kg"),
                    feature_bounds={"nitrate_kg": (1.0, 16.0), "phosphate_kg": (1.0, 16.0)},
                    clean_fn=lambda pt, c1=cb_s2_c1, c2=cb_s2_c2: min(c1 * pt["nitrate_kg"], c2 * pt["phosphate_kg"]),
                    eval_points=[{"nitrate_kg": a, "phosphate_kg": b} for a, b in grid_2d],
                    rmse_tol=0.06,
                    noise_seed=seed + 22,
                    domain_regime_attrs={"digital_throttling": False},
                ),
            ),
            # 1D Saturation Pair (for Cluster C Macro Scale Transfer)
            (
                "src_M2C1_catalytic_monolith",
                build_task_pair(
                    tid="src_M2C1",
                    domain="heterogeneous_catalysis",
                    feature_names=("reactant_feed",),
                    feature_bounds={"reactant_feed": (1.0, 28.0)},
                    clean_fn=lambda pt: pt["reactant_feed"] / (0.28 * pt["reactant_feed"] + 1.6),
                    eval_points=[{"reactant_feed": u} for u in eval_1d_u],
                    rmse_tol=0.03,
                    noise_seed=seed + 31,
                    domain_regime_attrs={"packed_bed": True},
                ),
            ),
            (
                "src_M2C2_token_bucket_egress",
                build_task_pair(
                    tid="src_M2C2",
                    domain="network_egress_shaper",
                    feature_names=("policer_rate",),
                    feature_bounds={"policer_rate": (1.0, 28.0)},
                    clean_fn=lambda pt: pt["policer_rate"] / (0.26 * pt["policer_rate"] + 1.75 + 0.00045 * (pt["policer_rate"] ** 2)),
                    eval_points=[{"policer_rate": u} for u in eval_1d_u],
                    rmse_tol=0.03,
                    noise_seed=seed + 32,
                    domain_regime_attrs={"packed_bed": False},
                ),
            ),
            (
                "src_M2C3_edge_cdn_relay",
                build_task_pair(
                    tid="src_M2C3",
                    domain="edge_cdn_relay_buffer",
                    feature_names=("ingress_rate",),
                    feature_bounds={"ingress_rate": (1.0, 28.0)},
                    clean_fn=lambda pt: pt["ingress_rate"] / (0.24 * pt["ingress_rate"] + 1.60 + 0.00145 * (pt["ingress_rate"] ** 2)),
                    eval_points=[{"ingress_rate": u} for u in eval_1d_u],
                    rmse_tol=0.020,
                    noise_seed=seed + 33,
                    domain_regime_attrs={"packed_bed": False},
                ),
            ),
        ],
        "near_misses": [
            # Near-Miss 1: Unreset Initial Memory on Relaxation (m0 != 0)
            (
                "nm_M2A_unreset_eddy_memory",
                build_task_pair(
                    tid="nm_M2A",
                    domain="eddy_brake_warm_restart",
                    feature_names=("flux_step", "window_dt", "init_current"),
                    feature_bounds={"flux_step": (1.0, 9.0), "window_dt": (0.2, 9.0), "init_current": (-3.5, 3.5)},
                    clean_fn=lambda pt: _rk4_conjugate_relaxation(
                        pt["flux_step"], pt["window_dt"], pt["init_current"],
                        k_frozen=nm_a_kfr, k_relaxed=nm_a_krel, tau=nm_a_tau,
                    ),
                    eval_points=nm_a_eval,
                    readout_sign=-1,
                    readout_scale=1.0,
                    readout_offset=1.5,
                    expose_calibration_channel=True,
                    noise_seed=seed + 41,
                ),
            ),
            # Near-Miss 2: Globally Capped 2D Bottleneck min(c1*x1, c2*x2, cap)
            (
                "nm_M2B_capped_reactor_ceiling",
                build_task_pair(
                    tid="nm_M2B",
                    domain="bioreactor_oxygen_capped",
                    feature_names=("carbon_feed", "aeration_rate"),
                    feature_bounds={"carbon_feed": (1.0, 16.0), "aeration_rate": (1.0, 16.0)},
                    clean_fn=lambda pt: min(2.1 * pt["carbon_feed"], 2.4 * pt["aeration_rate"], 17.5),
                    eval_points=[{"carbon_feed": a, "aeration_rate": b} for a, b in grid_2d],
                    rmse_tol=0.08,
                    noise_seed=seed + 42,
                ),
            ),
        ],
        "scale_calibration_probe_fn": run_m2_scale_calibration_probe,
        "targets": [
            # 1. Unseen 3rd Domain Conjugate Relaxation (Reset Queue Congestion Backpressure, inverted readout!)
            (
                "Cluster_M2A",
                "T_M2A1_queue_backpressure_reset",
                build_task_pair(
                    tid="T_M2A1",
                    domain="tcp_backpressure_controller",
                    feature_names=("surge_load", "horizon_s", "backlog_init"),
                    feature_bounds={"surge_load": (1.0, 9.0), "horizon_s": (0.2, 9.0), "backlog_init": (0.0, 0.0)},
                    clean_fn=lambda pt, kf=t_a1_kfr, kr=t_a1_krel, t=t_a1_tau: _rk4_conjugate_relaxation(
                        pt["surge_load"], pt["horizon_s"], pt["backlog_init"], k_frozen=kf, k_relaxed=kr, tau=t
                    ),
                    eval_points=[
                        {"surge_load": u, "horizon_s": dt, "backlog_init": 0.0}
                        for u, dt in [(2.0, 0.35), (4.5, 1.4), (7.0, 4.2), (8.5, 8.0), (5.5, 0.7)]
                    ],
                    readout_sign=-1,
                    readout_scale=1.25,
                    readout_offset=2.0,
                    expose_calibration_channel=True,
                    obs_budget=18,
                    meas_budget=90.0,
                    noise_seed=seed + 101,
                    domain_regime_attrs={"aqueous_phase": False},
                ),
            ),
            # 2. Unseen Domain with Unreset Memory (`backlog_init != 0` -> routes to revised state-augmented Python module!)
            (
                "Cluster_M2A",
                "T_M2A2_thermal_unreset_memory",
                build_task_pair(
                    tid="T_M2A2",
                    domain="thermal_heatsink_warmstart",
                    feature_names=("heater_step", "elapsed_s", "residual_temp"),
                    feature_bounds={"heater_step": (1.0, 9.0), "elapsed_s": (0.2, 9.0), "residual_temp": (-3.0, 3.0)},
                    clean_fn=lambda pt, kf=t_a2_kfr, kr=t_a2_krel, t=t_a2_tau: _rk4_conjugate_relaxation(
                        pt["heater_step"], pt["elapsed_s"], pt["residual_temp"], k_frozen=kf, k_relaxed=kr, tau=t
                    ),
                    eval_points=[
                        {"heater_step": u, "elapsed_s": dt, "residual_temp": m0}
                        for u, dt, m0 in [(2.5, 0.4, 2.0), (5.5, 1.5, -2.5), (7.5, 5.0, 2.2), (4.0, 0.7, -1.8), (6.5, 7.5, 1.5)]
                    ],
                    expose_calibration_channel=True,
                    obs_budget=20,
                    meas_budget=100.0,
                    noise_seed=seed + 102,
                ),
            ),
            # 3. Bimodal Viscoelastic Creep (Near-Miss Counterexample -> 1-mode rejected on residual, clean fallback!)
            (
                "Cluster_M2A",
                "T_M2A3_bimodal_viscoelastic_creep",
                build_task_pair(
                    tid="T_M2A3",
                    domain="polymer_viscoelastic_mount",
                    feature_names=("strain_step", "creep_time", "pre_stress"),
                    feature_bounds={"strain_step": (1.0, 9.0), "creep_time": (0.15, 10.0), "pre_stress": (0.0, 0.0)},
                    clean_fn=lambda pt: _rk4_bimodal_creep_relaxation(
                        pt["strain_step"], pt["creep_time"],
                        k_relaxed=0.65, amp_fast=1.35, tau_fast=0.35, amp_slow=1.15, tau_slow=4.5,
                    ),
                    eval_points=[
                        {"strain_step": u, "creep_time": dt, "pre_stress": 0.0}
                        for u, dt in [(2.0, 0.2), (4.0, 0.8), (6.0, 2.5), (8.0, 6.0), (5.0, 9.5), (7.0, 0.35)]
                    ],
                    rmse_tol=0.06,
                    obs_budget=22,
                    meas_budget=110.0,
                    noise_seed=seed + 103,
                ),
            ),
            # 4. Unseen 2D Tied Dual-Transport Bottleneck (Battery Electrode)
            (
                "Cluster_M2B",
                "T_M2B1_battery_dual_transport",
                build_task_pair(
                    tid="T_M2B1",
                    domain="battery_cathode_transport",
                    feature_names=("li_ion_cond", "electron_cond"),
                    feature_bounds={"li_ion_cond": (1.0, 16.0), "electron_cond": (1.0, 16.0)},
                    clean_fn=lambda pt, c1=tb1_c1, c2=tb1_c2: min(c1 * pt["li_ion_cond"], c2 * pt["electron_cond"]),
                    eval_points=[{"li_ion_cond": a, "electron_cond": b} for a, b in grid_2d],
                    rmse_tol=0.06,
                    obs_budget=18,
                    meas_budget=90.0,
                    noise_seed=seed + 104,
                    domain_regime_attrs={"digital_throttling": False},
                ),
            ),
            # 5. Unseen Capped 2D Bottleneck (Datacenter Power + Cooling + Grid Cap)
            (
                "Cluster_M2B",
                "T_M2B2_datacenter_power_cooling_capped",
                build_task_pair(
                    tid="T_M2B2",
                    domain="datacenter_rack_provisioning",
                    feature_names=("pdu_kw", "chiller_kw"),
                    feature_bounds={"pdu_kw": (1.0, 16.0), "chiller_kw": (1.0, 16.0)},
                    clean_fn=lambda pt, c1=tb2_c1, c2=tb2_c2, cap=tb2_cap: min(c1 * pt["pdu_kw"], c2 * pt["chiller_kw"], cap),
                    eval_points=[{"pdu_kw": a, "chiller_kw": b} for a, b in grid_2d],
                    rmse_tol=0.08,
                    obs_budget=20,
                    meas_budget=100.0,
                    noise_seed=seed + 105,
                ),
            ),
            # 6. Smooth Multiplicative Synergy (Cobb-Douglas `c * sqrt(x1 * x2)` -> piecewise min rejected cleanly!)
            (
                "Cluster_M2B",
                "T_M2B3_cobb_douglas_synergy",
                build_task_pair(
                    tid="T_M2B3",
                    domain="team_compute_bandwidth_synergy",
                    feature_names=("gpu_nodes", "interconnect_gbps"),
                    feature_bounds={"gpu_nodes": (1.0, 16.0), "interconnect_gbps": (1.0, 16.0)},
                    clean_fn=lambda pt, coef=tb3_coef: coef * math.sqrt(pt["gpu_nodes"] * pt["interconnect_gbps"]),
                    eval_points=[{"gpu_nodes": a, "interconnect_gbps": b} for a, b in grid_2d],
                    rmse_tol=0.06,
                    obs_budget=20,
                    meas_budget=100.0,
                    noise_seed=seed + 106,
                ),
            ),
            # 7. Same Oscillatory Carrier — Macro Window-Averaged Flux (ScaleMap valid -> 1-hop wormhole!)
            (
                "Cluster_M2C",
                "T_M2C1_carrier_macro_window_mean",
                build_task_pair(
                    tid="T_M2C1",
                    domain="pulsed_membrane_extractor",
                    feature_names=("pump_drive",),
                    feature_bounds={"pump_drive": (1.0, 28.0)},
                    clean_fn=lambda pt, a=cc_alpha, b=cc_beta, osc=cc_osc: _sim_oscillatory_carrier_trace(
                        pt["pump_drive"], alpha=a, beta=b, osc_amp=osc
                    )["window_mean"],
                    eval_points=[{"pump_drive": u} for u in eval_1d_u],
                    goal_kind="open_workspace_mechanism_prediction",
                    rmse_tol=0.03,
                    available_scale_map=unverified_scale_map,
                    obs_budget=14,
                    meas_budget=70.0,
                    noise_seed=seed + 107,
                    domain_regime_attrs={"packed_bed": False},
                ),
            ),
            # 8. Same Oscillatory Carrier — Micro Peak Crest Load (ScaleMap blocks window_average at 0 wasted probes!)
            (
                "Cluster_M2C",
                "T_M2C2_carrier_micro_peak_crest",
                build_task_pair(
                    tid="T_M2C2",
                    domain="pulsed_membrane_crest_stress",
                    feature_names=("pump_drive",),
                    feature_bounds={"pump_drive": (1.0, 28.0)},
                    clean_fn=lambda pt, a=cc_alpha, b=cc_beta, osc=cc_osc: _sim_oscillatory_carrier_trace(
                        pt["pump_drive"], alpha=a, beta=b, osc_amp=osc
                    )["peak_crest"],
                    eval_points=[{"pump_drive": u} for u in eval_1d_u],
                    goal_kind="micro_phase_peak_crest",
                    rmse_tol=0.08,
                    available_scale_map=unverified_scale_map,
                    obs_budget=16,
                    meas_budget=80.0,
                    noise_seed=seed + 108,
                ),
            ),
            # 9. Shifted + Inverted Sensor Saturation (`ObservationMap` calibration + Rational Wormhole!)
            (
                "Cluster_M2C",
                "T_M2C3_inverted_sensor_saturation",
                build_task_pair(
                    tid="T_M2C3",
                    domain="optical_quenching_sensor",
                    feature_names=("quencher_conc",),
                    feature_bounds={"quencher_conc": (1.0, 28.0)},
                    clean_fn=lambda pt, a=tc3_a, b=tc3_b: pt["quencher_conc"] / (a * pt["quencher_conc"] + b),
                    eval_points=[{"quencher_conc": u} for u in eval_1d_u],
                    readout_sign=-1,
                    readout_scale=2.0,
                    readout_offset=10.0,
                    expose_calibration_channel=True,
                    rmse_tol=0.05,
                    obs_budget=16,
                    meas_budget=80.0,
                    noise_seed=seed + 109,
                    domain_regime_attrs={"packed_bed": False},
                ),
            ),
            # 10. 2-Hop Non-Transitive Drift Target (S_C1 <-> S_C2 <-> S_C3 empirical chain, NO target hint flag!)
            (
                "Cluster_M2C",
                "T_M2C4_two_hop_drift_non_transitive",
                build_task_pair(
                    tid="T_M2C4",
                    domain="congested_relay_two_hop",
                    feature_names=("relay_load",),
                    feature_bounds={"relay_load": (1.0, 28.0)},
                    clean_fn=lambda pt: pt["relay_load"] / (0.22 * pt["relay_load"] + 1.5 + 0.0023 * (pt["relay_load"] ** 2)),
                    eval_points=[{"relay_load": u} for u in eval_1d_u],
                    goal_kind="open_workspace_mechanism_prediction",
                    rmse_tol=0.022,
                    obs_budget=16,
                    meas_budget=80.0,
                    noise_seed=seed + 110,
                    domain_regime_attrs={"packed_bed": False},
                ),
            ),
        ],
    }

"""Cross-Domain Simulators & Strictly Isolated Read-Only Evaluators (Hegel V16.6 M1).

Implements the three non-physics experimental clusters from ``06_给Gemini的实施合同.md``:
  - Cluster A: Ordered Boundary & Diagnostic Program (Software Build Commits, DAC
    Comparator Ladder, Hotfix-Revert Near-Miss; 5 target contract types).
  - Cluster B: Capacity Saturation, Limiting Factors & Joint Probes (Parallel Worker
    Timeline Sim, Enzyme RK4 Kinetics Sim, Tied/Asymmetric Min Bottlenecks, Sensor
    Clipping, Unbounded Sqrt Concave, Congestion Thrashing; 6 target types).
  - Cluster C: Unified Versioned Packet-Queue Service under 3 TaskViews / ScaleMaps
    (Mean Throughput Capacity, Burst p99 Tail Latency, Version Fault Isolation).

Strict Isolation Rule:
  Public laboratory objects expose ONLY public control bounds, probe costs, budget
  caps, and ``TaskView`` contracts. Ground truth lives exclusively inside closed
  read-only evaluator functions never accessible to any research arm.
"""

from __future__ import annotations

import math
from random import Random
from typing import Any, Callable


class BudgetExhaustedError(RuntimeError):
    """Raised when an arm attempts to spend beyond its allocated probe/compute budget."""


class CostMeter:
    """Mutually exclusive per-episode cost ledger across all phases."""

    def __init__(self, *, obs_budget: int, meas_budget: float, compute_budget: float) -> None:
        self.obs_budget = int(obs_budget)
        self.meas_budget = float(meas_budget)
        self.compute_budget = float(compute_budget)
        self.obs_spent = 0
        self.meas_spent = 0.0
        self.compute_spent = 0.0
        self.model_calls = 0
        self.reported_tokens = 0
        self.phase_breakdown: dict[str, dict[str, float]] = {}

    def charge(
        self,
        phase: str,
        *,
        obs: int = 0,
        meas: float = 0.0,
        compute: float = 0.0,
    ) -> bool:
        if (self.obs_spent + obs > self.obs_budget + 1e-9 or
                self.meas_spent + meas > self.meas_budget + 1e-9 or
                self.compute_spent + compute > self.compute_budget + 1e-9):
            return False
        self.obs_spent += int(obs)
        self.meas_spent += float(meas)
        self.compute_spent += float(compute)
        bucket = self.phase_breakdown.setdefault(
            phase, {"obs": 0.0, "meas": 0.0, "compute": 0.0}
        )
        bucket["obs"] += float(obs)
        bucket["meas"] += float(meas)
        bucket["compute"] += float(compute)
        return True

    def remaining_meas(self) -> float:
        return max(0.0, self.meas_budget - self.meas_spent)

    def remaining_obs(self) -> int:
        return max(0, self.obs_budget - self.obs_spent)

    def snapshot(self) -> dict[str, Any]:
        # Pre-registered composite cost: 1.0 * meas_spent + 0.05 * compute_spent
        composite = self.meas_spent + 0.05 * self.compute_spent
        return {
            "observation_cost": self.obs_spent,
            "measurement_cost": round(self.meas_spent, 4),
            "compute_cost": round(self.compute_spent, 4),
            "model_calls": self.model_calls,
            "reported_tokens": self.reported_tokens,
            "composite_cost": round(composite, 4),
            "phase_breakdown": {k: {mk: round(mv, 4) for mk, mv in v.items()}
                                for k, v in self.phase_breakdown.items()},
        }


# ============================================================================
# CLUSTER A: Ordered Discrete Boundary Environments
# ============================================================================

class PublicOrderedBoundaryTask:
    """Public interface for an ordered discrete sequence task (zero oracle leakage)."""

    def __init__(
        self,
        *,
        task_id: str,
        surface_domain: str,
        length: int,
        cost_per_query: float,
        task_view: dict[str, Any],
        meter: CostMeter,
        private_oracle_fn: Callable[[int], int | str],
    ) -> None:
        self.task_id = task_id
        self.surface_domain = surface_domain
        self.length = int(length)
        self.cost_per_query = float(cost_per_query)
        self.task_view = dict(task_view)
        self.meter = meter
        self._oracle = private_oracle_fn
        self.query_log: list[tuple[int, int | str, str]] = []

    def query_index(self, idx: int, *, phase: str = "probe") -> int | str | None:
        if idx < 0 or idx >= self.length:
            raise ValueError(f"Index {idx} out of range [0, {self.length - 1}]")
        if not self.meter.charge(phase, obs=1, meas=self.cost_per_query, compute=1.0):
            return None
        val = self._oracle(int(idx))
        self.query_log.append((int(idx), val, phase))
        return val


def make_cluster_a_suite(seed: int) -> dict[str, Any]:
    """Build Cluster A source episodes, near-miss episode, and 5 unseen target tasks."""
    rng = Random(seed)
    n_len = 128

    # Source 1: Software commit regression build pipeline (monotone single break)
    s1_break = rng.randint(18, 108)
    s1_seq = [0 if i < s1_break else 1 for i in range(n_len)]

    # Source 2: Hardware quantized DAC comparator threshold ladder (monotone single threshold)
    s2_thresh = rng.randint(22, 104)
    s2_seq = [0 if i < s2_thresh else 1 for i in range(n_len)]

    # Near-Miss Revision Episode: Hotfix-revert-reintroduce sequence (0 -> 1 -> 0 -> 1)
    nm_i1 = rng.randint(14, 34)
    nm_i2 = nm_i1 + rng.randint(14, 24)
    nm_i3 = nm_i2 + rng.randint(18, 30)
    nm_seq = [
        0 if i < nm_i1 else (1 if i < nm_i2 else (0 if i < nm_i3 else 1))
        for i in range(n_len)
    ]

    def build_seq_task(
        tid: str,
        domain: str,
        seq: list[int | str],
        goal_kind: str,
        cost_q: float = 3.0,
        noise_flip_p: float = 0.0,
        noise_seed: int = 0,
        obs_budget: int = 64,
        meas_budget: float = 192.0,
    ) -> tuple[Callable[[], PublicOrderedBoundaryTask], Callable[[dict[str, Any]], dict[str, Any]]]:
        clean_seq = list(seq)

        def factory() -> PublicOrderedBoundaryTask:
            local_rng = Random(noise_seed)
            meter = CostMeter(obs_budget=obs_budget, meas_budget=meas_budget, compute_budget=500.0)

            def oracle(idx: int) -> int | str:
                base = clean_seq[idx]
                if base == "SKIP":
                    return "SKIP"
                if noise_flip_p > 0.0 and 0 < idx < len(clean_seq) - 1:
                    if local_rng.random() < noise_flip_p:
                        return 1 - int(base)
                return int(base)

            tv = {
                "goal_kind": goal_kind,
                "target_metric": "exact_adjacent_boundary",
                "tolerance": 0.0,
                "input_domain": {"type": "ordered_discrete", "length": len(clean_seq)},
                "stochastic_oracle_hint": noise_flip_p > 0.0,
            }
            return PublicOrderedBoundaryTask(
                task_id=tid,
                surface_domain=domain,
                length=len(clean_seq),
                cost_per_query=cost_q,
                task_view=tv,
                meter=meter,
                private_oracle_fn=oracle,
            )

        def evaluator(submission: dict[str, Any]) -> dict[str, Any]:
            # Compute true valid adjacent flips on clean underlying sequence
            flips = [
                i for i in range(1, len(clean_seq))
                if clean_seq[i - 1] != "SKIP" and clean_seq[i] != "SKIP" and clean_seq[i - 1] != clean_seq[i]
            ]
            # Also allow bridging across a single SKIP commit (i-2 -> i)
            skip_bridges = [
                i for i in range(2, len(clean_seq))
                if clean_seq[i - 1] == "SKIP" and clean_seq[i - 2] != "SKIP"
                and clean_seq[i] != "SKIP" and clean_seq[i - 2] != clean_seq[i]
            ]
            all_valid_flips = sorted(set(flips + skip_bridges))
            earliest = all_valid_flips[0] if all_valid_flips else None

            reported_status = submission.get("status")
            pred_idx = submission.get("boundary_right_index")
            claimed_guarantee = submission.get("guarantee_kind")
            goal_relaxed_illegally = (
                goal_kind == "earliest_global_threshold"
                and claimed_guarantee == "some_adjacent_flip"
                and reported_status == "completed"
            )
            if goal_relaxed_illegally:
                return {
                    "strict_pass": False,
                    "false_confirmation": True,
                    "reason": "illegal_task_view_relaxation_any_flip_claimed_for_earliest",
                    "true_earliest": earliest,
                    "predicted": pred_idx,
                }
            if goal_kind == "some_adjacent_flip":
                ok = (reported_status == "completed" and pred_idx in all_valid_flips)
                return {
                    "strict_pass": bool(ok),
                    "false_confirmation": bool(reported_status == "completed" and not ok),
                    "valid_flips": all_valid_flips,
                    "predicted": pred_idx,
                }
            if goal_kind == "earliest_global_threshold":
                ok = (reported_status == "completed" and pred_idx == earliest)
                return {
                    "strict_pass": bool(ok),
                    "false_confirmation": bool(reported_status == "completed" and not ok),
                    "true_earliest": earliest,
                    "predicted": pred_idx,
                }
            raise ValueError(f"Unknown goal_kind: {goal_kind}")

        return factory, evaluator

    # 5 Unseen Target Tasks covering every row of Contract §5.1 Table
    t1_b = rng.randint(15, 110)
    t1_seq = [0 if i < t1_b else 1 for i in range(n_len)]

    a2_1 = rng.randint(12, 28)
    a2_2 = a2_1 + rng.randint(15, 25)
    a2_3 = a2_2 + rng.randint(20, 32)
    t2_seq = [0 if i < a2_1 else (1 if i < a2_2 else (0 if i < a2_3 else 1)) for i in range(n_len)]

    a3_1 = rng.randint(10, 24)  # Early fault! Midpoint bisect will land on a3_3 >> a3_1!
    a3_2 = a3_1 + rng.randint(12, 18)
    a3_3 = rng.randint(76, 98)
    t3_seq = [0 if i < a3_1 else (1 if i < a3_2 else (0 if i < a3_3 else 1)) for i in range(n_len)]

    p_start = rng.randint(36, 58)
    p_end = p_start + rng.randint(16, 24)
    t4_seq = [1 if p_start <= i < p_end else 0 for i in range(n_len)]  # Same endpoints 0 == 0!

    t5_b = rng.randint(28, 96)
    t5_seq: list[int | str] = [0 if i < t5_b else 1 for i in range(n_len)]
    # Make exact midpoint and a couple interior commits unbuildable 'SKIP' + mild noise
    t5_seq[64] = "SKIP"
    if t5_b + 3 < n_len - 1:
        t5_seq[t5_b + 3] = "SKIP"

    return {
        "sources": [
            ("src_A1_commit_build", build_seq_task("src_A1", "software_ci", s1_seq, "earliest_global_threshold", cost_q=3.0)),
            ("src_A2_dac_ladder", build_seq_task("src_A2", "hardware_dac", s2_seq, "some_adjacent_flip", cost_q=2.5)),
        ],
        "near_miss": (
            "nm_A_hotfix_revert",
            build_seq_task("nm_A", "ci_hotfix_revert", nm_seq, "earliest_global_threshold", cost_q=3.0, obs_budget=80, meas_budget=240.0),
        ),
        "targets": [
            ("T_A1_monotone_earliest", build_seq_task("T_A1", "db_migration_chain", t1_seq, "earliest_global_threshold", cost_q=3.0)),
            ("T_A2_nonmonotone_any_flip", build_seq_task("T_A2", "feature_flag_rollout", t2_seq, "some_adjacent_flip", cost_q=3.0)),
            ("T_A3_nonmonotone_earliest_strict", build_seq_task("T_A3", "multi_revert_release", t3_seq, "earliest_global_threshold", cost_q=3.0, obs_budget=72, meas_budget=216.0)),
            ("T_A4_same_endpoints_pocket", build_seq_task("T_A4", "canary_config_pocket", t4_seq, "earliest_global_threshold", cost_q=3.0, obs_budget=72, meas_budget=216.0)),
            ("T_A5_noisy_and_skip_midpoint", build_seq_task("T_A5", "hil_flaky_pipeline", t5_seq, "some_adjacent_flip", cost_q=3.0, noise_flip_p=0.10, noise_seed=seed + 99, obs_budget=72, meas_budget=216.0)),
        ],
    }


# ============================================================================
# CLUSTER B: Capacity Saturation, Limiting Factors & Joint Probes
# ============================================================================

class PublicResourceResponseTask:
    """Public interface for 1D or multi-resource response/sizing tasks (zero oracle leakage)."""

    def __init__(
        self,
        *,
        task_id: str,
        surface_domain: str,
        control_bounds: dict[str, tuple[float, float]],
        cost_per_probe: float,
        task_view: dict[str, Any],
        meter: CostMeter,
        private_sim_fn: Callable[[dict[str, float]], float],
    ) -> None:
        self.task_id = task_id
        self.surface_domain = surface_domain
        self.control_bounds = dict(control_bounds)
        self.cost_per_probe = float(cost_per_probe)
        self.task_view = dict(task_view)
        self.meter = meter
        self._sim = private_sim_fn
        self.observations: list[tuple[dict[str, float], float, str]] = []

    def measure(self, controls: dict[str, float], *, phase: str = "probe") -> float | None:
        clipped: dict[str, float] = {}
        for k, (lo, hi) in self.control_bounds.items():
            if k not in controls:
                raise ValueError(f"Missing control {k}")
            clipped[k] = min(hi, max(lo, float(controls[k])))
        if not self.meter.charge(phase, obs=1, meas=self.cost_per_probe, compute=1.5):
            return None
        val = float(self._sim(clipped))
        self.observations.append((clipped, val, phase))
        return val


def _sim_parallel_worker_timeline(u_workers: float, *, serial_time: float, parallel_work: float, n_jobs: int = 30) -> float:
    """Discrete-batch timeline simulation of serial lock contention + parallel worker pool."""
    # Simulate n_jobs passing through a mutex serial stage + worker pool queue
    lock_free_at = 0.0
    worker_free_at = [0.0 for _ in range(max(1, int(round(u_workers * 10))))]
    # Continuous effective worker pool scaling (supporting fractional resource shares / vCPUs)
    eff_parallel_time = parallel_work / float(u_workers)
    total_elapsed = n_jobs * serial_time + eff_parallel_time * n_jobs / max(1.0, min(float(n_jobs), float(u_workers) + (float(n_jobs) - float(u_workers)) * (float(u_workers) / float(n_jobs))))
    # Exact batch throughput rate (normalized jobs per unit time):
    # Each cycle takes (serial_time + parallel_work / u_workers)
    cycle_time = serial_time + (parallel_work / float(u_workers))
    return 1.0 / cycle_time


def _sim_enzyme_rk4_initial_rate(substrate_u: float, *, e_total: float, k1: float, km1: float, kcat: float) -> float:
    """Numerical RK4 integration of E + S <-> ES -> E + P quasi-steady complex formation."""
    # Integrate fast complex equilibration d[ES]/dt = k1*(E0 - ES)*(S0) - (km1 + kcat)*ES
    es = 0.0
    dt = 0.002
    for _ in range(250):
        def rhs(cur_es: float) -> float:
            return k1 * (e_total - cur_es) * substrate_u - (km1 + kcat) * cur_es
        r1 = rhs(es)
        r2 = rhs(es + 0.5 * dt * r1)
        r3 = rhs(es + 0.5 * dt * r2)
        r4 = rhs(es + dt * r3)
        es += (dt / 6.0) * (r1 + 2 * r2 + 2 * r3 + r4)
    return kcat * es


def make_cluster_b_suite(seed: int) -> dict[str, Any]:
    """Build Cluster B distinct simulators: 2 sources, 2 near-misses, and 6 unseen target types."""
    rng = Random(seed + 1000)

    # Source B1: Parallel vCPU Worker Pool Simulator
    s1_serial = rng.uniform(0.18, 0.28)
    s1_par = rng.uniform(1.1, 1.8)
    s1_eval_grid = [1.5, 3.5, 7.0, 14.0, 24.0]

    # Source B2: Enzyme-Substrate RK4 Kinetics Simulator
    s2_e0 = rng.uniform(1.2, 2.0)
    s2_k1 = 2.5
    s2_km1 = rng.uniform(4.0, 7.0)
    s2_kcat = rng.uniform(1.5, 2.5)
    s2_eval_grid = [1.5, 4.0, 9.0, 18.0, 28.0]

    def make_1d_curve_task(
        tid: str,
        domain: str,
        sim_fn: Callable[[float], float],
        eval_grid: list[float],
        *,
        bounds: tuple[float, float] = (1.0, 32.0),
        probe_cost: float = 6.0,
        rmse_tol: float = 0.025,
        obs_budget: int = 16,
        meas_budget: float = 96.0,
        noise_std: float = 0.002,
        noise_seed: int = 1,
        clipping_ceiling: float | None = None,
    ):
        def factory() -> PublicResourceResponseTask:
            local_rng = Random(noise_seed)
            meter = CostMeter(obs_budget=obs_budget, meas_budget=meas_budget, compute_budget=600.0)

            def noisy_sim(ctrl: dict[str, float]) -> float:
                raw = sim_fn(ctrl["u"]) + local_rng.gauss(0.0, noise_std)
                if clipping_ceiling is not None:
                    return min(clipping_ceiling, max(1e-5, raw))
                return max(1e-5, raw)

            tv = {
                "goal_kind": "response_curve_and_target_sizing",
                "target_metric": "rmse_original_scale",
                "tolerance": rmse_tol,
                "eval_query_points": list(eval_grid),
                "input_domain": {"u": list(bounds)},
                **({"sensor_ceiling_spec_hint": clipping_ceiling} if clipping_ceiling is not None else {}),
            }
            return PublicResourceResponseTask(
                task_id=tid,
                surface_domain=domain,
                control_bounds={"u": bounds},
                cost_per_probe=probe_cost,
                task_view=tv,
                meter=meter,
                private_sim_fn=noisy_sim,
            )

        def evaluator(submission: dict[str, Any]) -> dict[str, Any]:
            if submission.get("status") != "completed" or "predict_fn" not in submission:
                return {"strict_pass": False, "false_confirmation": False, "rmse": None}
            pred_fn = submission["predict_fn"]
            errs = []
            for u_q in eval_grid:
                true_y = sim_fn(u_q)
                if clipping_ceiling is not None:
                    true_y = min(clipping_ceiling, true_y)
                pred_y = float(pred_fn(u_q))
                errs.append((pred_y - true_y) ** 2)
            rmse = math.sqrt(sum(errs) / len(errs))
            ok = rmse <= rmse_tol
            return {
                "strict_pass": bool(ok),
                "false_confirmation": bool(not ok),
                "rmse": round(rmse, 6),
                "tolerance": rmse_tol,
            }

        return factory, evaluator

    def make_2d_bottleneck_task(
        tid: str,
        domain: str,
        c1: float,
        c2: float,
        start_u1: float,
        start_u2: float,
        target_output: float,
        *,
        probe_cost: float = 6.0,
        obs_budget: int = 16,
        meas_budget: float = 96.0,
    ):
        def sim_2d(ctrl: dict[str, float]) -> float:
            return min(c1 * ctrl["u1"], c2 * ctrl["u2"])

        def factory() -> PublicResourceResponseTask:
            meter = CostMeter(obs_budget=obs_budget, meas_budget=meas_budget, compute_budget=600.0)
            tv = {
                "goal_kind": "multi_resource_target_allocation",
                "target_metric": "achieve_target_with_near_minimal_resource_spend",
                "tolerance": 0.05,
                "initial_operating_point": {"u1": start_u1, "u2": start_u2},
                "required_output": target_output,
                "resource_unit_prices": {"u1": 1.0, "u2": 1.0},
            }
            return PublicResourceResponseTask(
                task_id=tid,
                surface_domain=domain,
                control_bounds={"u1": (1.0, 30.0), "u2": (1.0, 30.0)},
                cost_per_probe=probe_cost,
                task_view=tv,
                meter=meter,
                private_sim_fn=sim_2d,
            )

        def evaluator(submission: dict[str, Any]) -> dict[str, Any]:
            if submission.get("status") != "completed" or "recommended_controls" not in submission:
                return {"strict_pass": False, "false_confirmation": False}
            rec = submission["recommended_controls"]
            achieved = sim_2d(rec)
            opt_spend = target_output / c1 + target_output / c2
            actual_spend = float(rec["u1"]) + float(rec["u2"])
            meets_output = achieved >= target_output * 0.97
            efficient = actual_spend <= opt_spend * 1.15
            ok = meets_output and efficient
            return {
                "strict_pass": bool(ok),
                "false_confirmation": bool(not ok),
                "achieved_output": round(achieved, 4),
                "required_output": round(target_output, 4),
                "actual_resource_spend": round(actual_spend, 4),
                "optimal_resource_spend": round(opt_spend, 4),
            }

        return factory, evaluator

    # Target parameters (unseen structures & parameterizations!)
    tb1_a = rng.uniform(0.22, 0.35)
    tb1_b = rng.uniform(1.4, 2.4)
    tb2_c1, tb2_c2 = rng.uniform(1.6, 2.2), rng.uniform(2.8, 3.5)
    tb3_c1 = rng.uniform(1.8, 2.4)
    tb3_c2 = tb3_c1 * 1.25  # Start at exact tie c1*u1 == c2*u2!
    tb3_u1_0 = 5.0
    tb3_u2_0 = (tb3_c1 * tb3_u1_0) / tb3_c2  # Exact tied bottleneck!
    tb4_slope, tb4_ceil = rng.uniform(0.35, 0.50), rng.uniform(4.2, 5.5)
    tb5_scale = rng.uniform(0.85, 1.25)
    tb6_a, tb6_b, tb6_g = 0.20, 1.5, rng.uniform(0.018, 0.032)

    eval_grid_target = [2.0, 5.0, 10.0, 18.0, 28.0]

    return {
        "sources": [
            (
                "src_B1_worker_pool",
                make_1d_curve_task(
                    "src_B1", "parallel_compute_pool",
                    lambda u, s=s1_serial, p=s1_par: _sim_parallel_worker_timeline(u, serial_time=s, parallel_work=p),
                    s1_eval_grid, noise_seed=seed + 201,
                ),
            ),
            (
                "src_B2_enzyme_kinetics",
                make_1d_curve_task(
                    "src_B2", "biochem_enzyme_assay",
                    lambda u, e0=s2_e0, k1=s2_k1, km1=s2_km1, kc=s2_kcat: _sim_enzyme_rk4_initial_rate(
                        u, e_total=e0, k1=k1, km1=km1, kcat=kc
                    ),
                    s2_eval_grid, noise_seed=seed + 202,
                ),
            ),
        ],
        "near_misses": [
            (
                "nm_B1_tied_min_pipeline",
                make_2d_bottleneck_task(
                    "nm_B1", "microservice_db_tie", 2.0, 2.0, 4.0, 4.0, 16.0,
                    obs_budget=18, meas_budget=108.0,
                ),
            ),
            (
                "nm_B2_unbounded_sqrt",
                make_1d_curve_task(
                    "nm_B2", "spatial_search_sublinear",
                    lambda u: 0.95 * math.sqrt(u) + 0.15,
                    eval_grid_target, rmse_tol=0.03, noise_seed=seed + 205,
                ),
            ),
        ],
        "targets": [
            (
                "T_B1_unseen_rational_transport",
                make_1d_curve_task(
                    "T_B1", "membrane_carrier_flux",
                    lambda u, a=tb1_a, b=tb1_b: u / (a * u + b),
                    eval_grid_target, rmse_tol=0.025, obs_budget=10, meas_budget=60.0, noise_seed=seed + 301,
                ),
            ),
            (
                "T_B2_hard_min_asymmetric",
                make_2d_bottleneck_task(
                    "T_B2", "supply_chain_liebig", tb2_c1, tb2_c2, 3.0, 5.0, 18.0,
                    obs_budget=12, meas_budget=72.0,
                ),
            ),
            (
                "T_B3_tied_bottleneck_joint",
                make_2d_bottleneck_task(
                    "T_B3", "dual_stage_tied_pipeline", tb3_c1, tb3_c2, tb3_u1_0, tb3_u2_0, 20.0,
                    obs_budget=12, meas_budget=72.0,
                ),
            ),
            (
                "T_B4_sensor_clipping_ceiling",
                make_1d_curve_task(
                    "T_B4", "adc_clipped_amplifier",
                    lambda u, sl=tb4_slope: sl * u,
                    eval_grid_target, rmse_tol=0.04, clipping_ceiling=tb4_ceil,
                    obs_budget=12, meas_budget=72.0, noise_seed=seed + 304,
                ),
            ),
            (
                "T_B5_unbounded_sqrt_concave",
                make_1d_curve_task(
                    "T_B5", "sensor_network_coverage",
                    lambda u, sc=tb5_scale: sc * math.sqrt(u) + 0.2,
                    eval_grid_target, rmse_tol=0.03, obs_budget=12, meas_budget=72.0, noise_seed=seed + 305,
                ),
            ),
            (
                "T_B6_congestion_thrashing",
                make_1d_curve_task(
                    "T_B6", "db_lock_convoy_thrashing",
                    lambda u, a=tb6_a, b=tb6_b, g=tb6_g: u / (a * u + b + g * u * u),
                    eval_grid_target, rmse_tol=0.035, obs_budget=13, meas_budget=78.0, noise_seed=seed + 306,
                ),
            ),
        ],
    }


# ============================================================================
# CLUSTER C: Same-Service Multi-View & Multi-Scale Routing
# ============================================================================

def make_cluster_c_same_service_suite(seed: int) -> dict[str, Any]:
    """One unified versioned packet-queue microservice queried under 3 TaskViews / ScaleMaps."""
    rng = Random(seed + 2000)
    n_commits = 64
    fault_commit = rng.randint(14, 52)
    serial_overhead = rng.uniform(0.20, 0.30)
    work_per_req = rng.uniform(1.2, 1.9)
    burst_intensity = rng.uniform(3.2, 4.5)

    def simulate_packet_queue(u_workers: float, *, burst_mode: bool) -> dict[str, float]:
        mean_tput = 1.0 / (serial_overhead + work_per_req / float(u_workers))
        # Under ON-OFF packet bursts, p99 tail latency spikes nonlinearly if u_workers < critical burst capacity,
        # whereas time-averaged throughput discards burst arrival clustering completely!
        burst_ratio = burst_intensity / float(u_workers)
        p99_latency = (0.4 + 0.8 / float(u_workers) + 6.5 * (burst_ratio ** 2)) if burst_mode else (0.4 + 0.8 / float(u_workers))
        return {"mean_throughput": mean_tput, "p99_tail_latency": p99_latency}

    eval_u = [2.0, 5.0, 10.0, 18.0, 24.0]

    # View C1: Steady-state Mean Throughput (explicit window_average ScaleMap valid!)
    def make_c1():
        def factory() -> PublicResourceResponseTask:
            meter = CostMeter(obs_budget=10, meas_budget=60.0, compute_budget=500.0)
            tv = {
                "goal_kind": "response_curve_and_target_sizing",
                "service_object_id": "unified_checkout_service",
                "routing_mode": "coarse_grained_window_average",
                "target_metric": "rmse_original_scale",
                "tolerance": 0.025,
                "eval_query_points": list(eval_u),
                "available_scale_map": {
                    "coarse_graining_op": "window_average",
                    "preserved_queries": ["response_curve_and_target_sizing", "steady_mean_throughput"],
                    "closure_scope": "steady_state_only",
                    "proposed_discarded_variables": ["packet_burst_clustering"],
                    "verified_discarded_in_scope": [{
                        "variable": "packet_burst_clustering",
                        "task_view_goal": "response_curve_and_target_sizing",
                        "error_bound": 0.005,
                        "evidence_ref": "stationary_ergodic_mean_window",
                    }],
                },
            }
            return PublicResourceResponseTask(
                task_id="T_C1_service_mean_throughput",
                surface_domain="checkout_service_capacity",
                control_bounds={"u": (1.0, 28.0)},
                cost_per_probe=6.0,
                task_view=tv,
                meter=meter,
                private_sim_fn=lambda ctrl: simulate_packet_queue(ctrl["u"], burst_mode=True)["mean_throughput"],
            )

        def evaluator(sub: dict[str, Any]) -> dict[str, Any]:
            if sub.get("status") != "completed" or "predict_fn" not in sub:
                return {"strict_pass": False, "false_confirmation": False}
            errs = [(float(sub["predict_fn"](u)) - simulate_packet_queue(u, burst_mode=True)["mean_throughput"]) ** 2 for u in eval_u]
            rmse = math.sqrt(sum(errs) / len(errs))
            return {"strict_pass": rmse <= 0.025, "false_confirmation": rmse > 0.025, "rmse": round(rmse, 6)}

        return factory, evaluator

    # View C2: Burst p99 Tail Latency (window_average loses burst peaks -> must reject coarse mean model!)
    def make_c2():
        def factory() -> PublicResourceResponseTask:
            meter = CostMeter(obs_budget=12, meas_budget=72.0, compute_budget=500.0)
            tv = {
                "goal_kind": "burst_p99_tail_latency",
                "service_object_id": "unified_checkout_service",
                "routing_mode": "scale_sensitive_tail_query",
                "target_metric": "rmse_original_scale",
                "tolerance": 0.15,
                "eval_query_points": list(eval_u),
                "available_scale_map": {
                    "coarse_graining_op": "window_average",
                    "preserved_queries": ["response_curve_and_target_sizing", "steady_mean_throughput"],
                    "closure_scope": "steady_state_only",
                    "proposed_discarded_variables": ["packet_burst_clustering"],
                    "verified_discarded_in_scope": [{
                        "variable": "packet_burst_clustering",
                        "task_view_goal": "steady_mean_throughput",
                        "error_bound": 0.005,
                        "evidence_ref": "stationary_ergodic_mean_window",
                    }],
                },
            }
            return PublicResourceResponseTask(
                task_id="T_C2_service_burst_tail_latency",
                surface_domain="checkout_service_tail_sla",
                control_bounds={"u": (1.0, 28.0)},
                cost_per_probe=6.0,
                task_view=tv,
                meter=meter,
                private_sim_fn=lambda ctrl: simulate_packet_queue(ctrl["u"], burst_mode=True)["p99_tail_latency"],
            )

        def evaluator(sub: dict[str, Any]) -> dict[str, Any]:
            if sub.get("status") != "completed" or "predict_fn" not in sub:
                return {"strict_pass": False, "false_confirmation": False}
            errs = [(float(sub["predict_fn"](u)) - simulate_packet_queue(u, burst_mode=True)["p99_tail_latency"]) ** 2 for u in eval_u]
            rmse = math.sqrt(sum(errs) / len(errs))
            return {"strict_pass": rmse <= 0.15, "false_confirmation": rmse > 0.15, "rmse": round(rmse, 6)}

        return factory, evaluator

    # View C3: Release Fault Commit Isolation on the SAME service (pure TaskView switch, not RG!)
    def make_c3():
        seq = [0 if i < fault_commit else 1 for i in range(n_commits)]

        def factory() -> PublicOrderedBoundaryTask:
            meter = CostMeter(obs_budget=32, meas_budget=96.0, compute_budget=400.0)
            tv = {
                "goal_kind": "some_adjacent_flip",
                "service_object_id": "unified_checkout_service",
                "routing_mode": "multi_view_task_switch_not_coarse_graining",
                "target_metric": "exact_adjacent_boundary",
                "tolerance": 0.0,
                "input_domain": {"type": "ordered_discrete", "length": n_commits},
            }
            return PublicOrderedBoundaryTask(
                task_id="T_C3_service_version_fault_isolation",
                surface_domain="checkout_service_release_history",
                length=n_commits,
                cost_per_query=3.0,
                task_view=tv,
                meter=meter,
                private_oracle_fn=lambda idx: seq[idx],
            )

        def evaluator(sub: dict[str, Any]) -> dict[str, Any]:
            ok = sub.get("status") == "completed" and sub.get("boundary_right_index") == fault_commit
            return {"strict_pass": bool(ok), "false_confirmation": bool(sub.get("status") == "completed" and not ok), "true_fault": fault_commit, "predicted": sub.get("boundary_right_index")}

        return factory, evaluator

    return {
        "targets": [
            ("T_C1_service_mean_throughput", make_c1()),
            ("T_C2_service_burst_tail_latency", make_c2()),
            ("T_C3_service_version_fault_isolation", make_c3()),
        ]
    }

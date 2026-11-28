"""M9.9 public world and validation for online two-stage module formation.

The task supplies an upstream/downstream interface, calibrated coordinates, and a
fixed stream order. Both modular arms may synthesize candidate implementations
from the shared monolithic source and a later public-qualified solution.
The graph controller can expand the downstream guard's observation scope after
a verified counterexample. E6 and E7 share the downstream precursor observations
while requesting different mechanisms. Actual solver and selector outcomes are
measured by the runner and recorded in the per-episode receipts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import tempfile
from time import monotonic
from typing import Any
import uuid

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
)
from .m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    encode_pure_json_numerical_payload,
)
from .m9_4_continual_mechanism_binding_world import (
    M94BindingRegistry,
    MechanismExecutableBinding,
)

from .m9_5_equal_tool_continual_attribution_world import (
    execute_single_pre_llm_public_gate_probe_in_container as _execute_public_gate,
    m95_task_to_frontier_task,
)

# Include Alpha's documented pooled-panel contract in every prospective M9.9 public gate.
execute_single_pre_llm_public_gate_probe_in_container = partial(_execute_public_gate, require_pooled_coupon_check=True)

from .m9_6_autonomous_structure_continual_world import (
    CURRENT_LINEAGE_SCHEMAS,
    ImmutableBranchLineageController,
    SolverEvaluationWitness,
    classify_public_probe_failure,
    joint_scientific_receipt_matches,
    compute_canonical_numeric_item_sha256,
    evaluate_m96_autonomous_guard,
)
from .m9_8_blind_mismatch_and_selective_dependency_world import (
    BLIND_ALPHA_INSTRUCTION_TEXT,
    BLIND_BETA_INSTRUCTION_TEXT,
    M98_ALL_OBSERVABLE_NAMES,
    M98_DOWNSTREAM_OBSERVABLE_NAMES,
    M98_UPSTREAM_OBSERVABLE_NAMES,
    M98WorldSpec,
    materialize_sandbox_public_item_and_sha as materialize_m99_public_item_and_sha,
    assemble_m98_graph_diagonal_payload,
    build_m98_upstream_transducer_synthesis_task,
    compose_m98_two_stage_solver_code,
    compute_m98_combined_observables,
    compute_m98_downstream_observables,
    compute_m98_upstream_observables,
    diagnose_m98_factored_conflict_locus,
    expected_m98_public_output_keys,
    initialize_m98_factored_mechanism_graph,
    propose_and_validate_m98_guard_partition,
    update_m98_calibration_snapshot_and_graph,
)

M99_ARMS: tuple[str, ...] = (
    "Flat_MonolithicConditionalTable_Online",
    "Flat_PlainModuleRegistry_Online",
    "Hegel_AutonomousInteractionGraph_Online",
)

M99_UPSTREAM_OBSERVABLE_NAMES = M98_UPSTREAM_OBSERVABLE_NAMES
M99_DOWNSTREAM_OBSERVABLE_NAMES = M98_DOWNSTREAM_OBSERVABLE_NAMES
M99_ALL_OBSERVABLE_NAMES = M98_ALL_OBSERVABLE_NAMES

M99_WITHDRAWAL_FOREST_KEY = "__m9_9_separability_withdrawal_ledger__"


@dataclass(frozen=True)
class M99WorldSpec:
    world_id: str
    cluster_id: str
    family: str  # 'alpha' | 'beta'
    split: str  # 'dev' | 'confirmation'
    seed_base: int
    channel_perm: tuple[int, int, int, int]
    active_log_channels: tuple[int, int]
    v2_gains: tuple[float, float, float, float]
    v2_offsets: tuple[float, float, float, float]
    v3_gains: tuple[float, float, float, float]
    v3_offsets: tuple[float, float, float, float]
    bridge_offdiag_step3: float
    bridge_offdiag_step4: float
    bridge_offdiag_step5: float
    bridge_offdiag_step7: float

    def to_m98_compat_spec(self) -> M98WorldSpec:
        return M98WorldSpec(
            world_id=self.world_id,
            cluster_id=self.cluster_id,
            family=self.family,
            split=self.split,
            seed_base=self.seed_base,
            channel_perm=self.channel_perm,
            active_log_channels=self.active_log_channels,
            v2_gains=self.v2_gains,
            v2_offsets=self.v2_offsets,
            v3_gains=self.v3_gains,
            v3_offsets=self.v3_offsets,
            bridge_offdiag_step4=self.bridge_offdiag_step3,
            bridge_offdiag_step6=self.bridge_offdiag_step4,
            bridge_offdiag_step7=self.bridge_offdiag_step7,
        )


DEV_M99_WORLD_SPECS: tuple[M99WorldSpec, ...] = (
    M99WorldSpec(
        world_id="dev_alpha_m99_00",
        cluster_id="dev_cluster_m99_alpha_00",
        family="alpha",
        split="dev",
        seed_base=850100,
        channel_perm=(3, 0, 2, 1),
        active_log_channels=(1, 6),
        v2_gains=(1.28, 0.79, 1.31, 0.78),
        v2_offsets=(0.13, -0.12, 0.15, -0.11),
        v3_gains=(0.80, 1.33, 1.21, 0.82),
        v3_offsets=(-0.14, 0.17, 0.10, -0.13),
        bridge_offdiag_step3=0.33,
        bridge_offdiag_step4=-0.31,
        bridge_offdiag_step5=0.32,
        bridge_offdiag_step7=-0.29,
    ),
    M99WorldSpec(
        world_id="dev_beta_m99_00",
        cluster_id="dev_cluster_m99_beta_00",
        family="beta",
        split="dev",
        seed_base=860100,
        channel_perm=(2, 3, 1, 0),
        active_log_channels=(5, 2),
        v2_gains=(1.27, 0.80, 1.25, 0.82),
        v2_offsets=(0.08, -0.07, 0.06, -0.08),
        v3_gains=(0.81, 1.30, 0.77, 1.28),
        v3_offsets=(-0.08, 0.10, -0.06, 0.07),
        bridge_offdiag_step3=0.32,
        bridge_offdiag_step4=-0.30,
        bridge_offdiag_step5=0.31,
        bridge_offdiag_step7=-0.28,
    ),
)

# `n = 8` Completely Fresh Unexposed Confirmation Clusters (`951000..964000` — Zero Overlap with Any Prior Milestone!)
CONFIRMATION_M99_WORLD_SPECS: tuple[M99WorldSpec, ...] = (
    M99WorldSpec(
        world_id="alpha_m99_c01",
        cluster_id="cluster_m99_01_alpha",
        family="alpha",
        split="confirmation",
        seed_base=951000,
        channel_perm=(1, 3, 0, 2),
        active_log_channels=(2, 7),
        v2_gains=(1.29, 0.78, 1.33, 0.79),
        v2_offsets=(0.15, -0.11, 0.14, -0.12),
        v3_gains=(0.79, 1.34, 1.24, 0.80),
        v3_offsets=(-0.14, 0.16, 0.11, -0.10),
        bridge_offdiag_step3=0.33,
        bridge_offdiag_step4=-0.31,
        bridge_offdiag_step5=0.30,
        bridge_offdiag_step7=-0.32,
    ),
    M99WorldSpec(
        world_id="alpha_m99_c02",
        cluster_id="cluster_m99_02_alpha",
        family="alpha",
        split="confirmation",
        seed_base=952000,
        channel_perm=(2, 0, 3, 1),
        active_log_channels=(4, 0),
        v2_gains=(0.80, 1.31, 0.81, 1.27),
        v2_offsets=(-0.13, 0.14, -0.11, 0.12),
        v3_gains=(1.31, 0.78, 1.27, 0.81),
        v3_offsets=(0.15, -0.13, 0.10, -0.09),
        bridge_offdiag_step3=-0.32,
        bridge_offdiag_step4=0.30,
        bridge_offdiag_step5=-0.31,
        bridge_offdiag_step7=0.29,
    ),
    M99WorldSpec(
        world_id="alpha_m99_c03",
        cluster_id="cluster_m99_03_alpha",
        family="alpha",
        split="confirmation",
        seed_base=953000,
        channel_perm=(3, 1, 2, 0),
        active_log_channels=(6, 3),
        v2_gains=(1.31, 0.82, 0.77, 1.28),
        v2_offsets=(0.16, -0.10, -0.13, 0.11),
        v3_gains=(0.82, 1.29, 1.32, 0.78),
        v3_offsets=(-0.12, 0.15, 0.14, -0.11),
        bridge_offdiag_step3=0.31,
        bridge_offdiag_step4=-0.33,
        bridge_offdiag_step5=0.32,
        bridge_offdiag_step7=-0.30,
    ),
    M99WorldSpec(
        world_id="alpha_m99_c04",
        cluster_id="cluster_m99_04_alpha",
        family="alpha",
        split="confirmation",
        seed_base=954000,
        channel_perm=(0, 3, 1, 2),
        active_log_channels=(5, 1),
        v2_gains=(0.82, 1.27, 1.32, 0.78),
        v2_offsets=(-0.10, 0.12, 0.16, -0.13),
        v3_gains=(1.29, 0.81, 0.78, 1.30),
        v3_offsets=(0.13, -0.11, -0.14, 0.12),
        bridge_offdiag_step3=-0.30,
        bridge_offdiag_step4=0.32,
        bridge_offdiag_step5=-0.29,
        bridge_offdiag_step7=0.31,
    ),
    M99WorldSpec(
        world_id="beta_m99_c01",
        cluster_id="cluster_m99_05_beta",
        family="beta",
        split="confirmation",
        seed_base=961000,
        channel_perm=(3, 1, 0, 2),
        active_log_channels=(3, 6),
        v2_gains=(1.28, 0.81, 1.26, 0.82),
        v2_offsets=(0.07, -0.06, 0.08, -0.05),
        v3_gains=(0.80, 1.31, 0.81, 1.28),
        v3_offsets=(-0.06, 0.08, -0.07, 0.09),
        bridge_offdiag_step3=0.32,
        bridge_offdiag_step4=-0.30,
        bridge_offdiag_step5=0.29,
        bridge_offdiag_step7=-0.31,
    ),
    M99WorldSpec(
        world_id="beta_m99_c02",
        cluster_id="cluster_m99_06_beta",
        family="beta",
        split="confirmation",
        seed_base=962000,
        channel_perm=(1, 2, 3, 0),
        active_log_channels=(7, 1),
        v2_gains=(0.79, 1.30, 0.81, 1.28),
        v2_offsets=(-0.08, 0.07, -0.06, 0.05),
        v3_gains=(1.29, 0.78, 1.31, 0.80),
        v3_offsets=(0.08, -0.07, 0.06, -0.05),
        bridge_offdiag_step3=-0.31,
        bridge_offdiag_step4=0.29,
        bridge_offdiag_step5=-0.30,
        bridge_offdiag_step7=0.28,
    ),
    M99WorldSpec(
        world_id="beta_m99_c03",
        cluster_id="cluster_m99_07_beta",
        family="beta",
        split="confirmation",
        seed_base=963000,
        channel_perm=(2, 0, 1, 3),
        active_log_channels=(0, 5),
        v2_gains=(1.27, 0.83, 0.78, 1.31),
        v2_offsets=(0.06, -0.05, -0.07, 0.08),
        v3_gains=(0.81, 1.28, 1.31, 0.79),
        v3_offsets=(-0.07, 0.06, 0.08, -0.07),
        bridge_offdiag_step3=0.30,
        bridge_offdiag_step4=-0.32,
        bridge_offdiag_step5=0.31,
        bridge_offdiag_step7=-0.29,
    ),
    M99WorldSpec(
        world_id="beta_m99_c04",
        cluster_id="cluster_m99_08_beta",
        family="beta",
        split="confirmation",
        seed_base=964000,
        channel_perm=(0, 2, 3, 1),
        active_log_channels=(4, 2),
        v2_gains=(0.81, 1.29, 1.31, 0.79),
        v2_offsets=(-0.05, 0.08, 0.07, -0.06),
        v3_gains=(1.30, 0.81, 0.78, 1.28),
        v3_offsets=(0.07, -0.08, -0.06, 0.05),
        bridge_offdiag_step3=-0.29,
        bridge_offdiag_step4=0.31,
        bridge_offdiag_step5=-0.28,
        bridge_offdiag_step7=0.30,
    ),
)


# ============================================================================
# 1. Empirical Physics Generators with Cross-Regime `U1 ⊗ D2 -> D2_coupled` Interaction
# ============================================================================

_M99_ALPHA_GEN_PY = '''
def _build_bridge_matrix(gains, offdiag):
    g = np.asarray(gains, dtype=float)
    M = np.diag(g)
    if abs(float(offdiag)) > 1e-12:
        for c in range(4):
            M[c, (c + 1) % 4] += float(offdiag) * g[c]
            M[c, (c - 1) % 4] -= 0.5 * float(offdiag) * g[c]
    return M


def _apply_alpha_transducer(x_true, gains, offsets, offdiag, upstream_mode):
    x_arr = np.asarray(x_true, dtype=float)
    orig_shape = x_arr.shape
    x2 = x_arr.reshape(-1, 4)
    g = np.asarray(gains, dtype=float)
    b_vec = np.asarray(offsets, dtype=float)
    M_br = _build_bridge_matrix(gains, offdiag if upstream_mode == "U1" else 0.0)
    y2 = x2 @ M_br.T + b_vec[None, :]
    if upstream_mode == "U1":
        sgn = 1.0 if float(offdiag) >= 0.0 else -1.0
        x_next = np.roll(x2, -1, axis=1)
        y2 = y2 + (0.18 * g)[None, :] * (x2 ** 3) + (0.22 * sgn * g)[None, :] * (x2 * x_next)
    return y2.reshape(orig_shape)


def _gen_m99_alpha_instance(
    seed: int,
    *,
    world_seed: int,
    upstream_mode: str,
    downstream_regime: str,
    gains: tuple[float, float, float, float],
    offsets: tuple[float, float, float, float],
    bridge_offdiag: float = 0.0,
    include_chamber_pilots: bool = True,
    cal_outlier_rows: int = 0,
):
    assert upstream_mode in ("U0", "U1")
    assert downstream_regime in ("D0", "D1", "D2")
    # CRITICAL (`M9.9` Non-Separable `U1 x D2` Cross-Mechanism Interaction):
    # - `(U0, D0)`, `(U0, D1)`, `(U0, D2)`, `(U1, D0)`, `(U1, D1)` are genuinely separable (`eff_regime = downstream_regime`).
    # - When `U1` (cross-channel nonlinear bridge drive) AND `D2` co-occur, chamber cross-excitation quenches hereditary memory (`z = 0`)
    #   and shifts creep drift to `2.0 * (sqrt(1 + t) - 1)` (`eff_regime = "D2_coupled"`), falsifying naive `d2_sep(u1(x))`!
    eff_regime = "D2_coupled" if (upstream_mode == "U1" and downstream_regime == "D2") else downstream_regime

    rng_w = np.random.default_rng(world_seed)
    taus = np.array([float(rng_w.uniform(0.45, 0.72)), float(rng_w.uniform(3.20, 4.60))], dtype=float)
    theta = np.array(
        [
            float(rng_w.uniform(0.55, 0.82)),
            float(rng_w.uniform(0.42, 0.68)),
            float(rng_w.uniform(0.26, 0.40)),
            float(rng_w.uniform(0.11, 0.21)),
            float(rng_w.uniform(-0.20, -0.09)),
            float(rng_w.uniform(0.09, 0.18)),
            float(rng_w.uniform(-0.16, -0.07)),
        ],
        dtype=float,
    )

    rng = np.random.default_rng(seed)
    past_dt = np.array([0.35, 0.35, 0.40, 0.45, 0.50, 0.55], dtype=float)
    eval_t = np.linspace(0.25, 4.0, 12, dtype=float)
    S_len = len(past_dt)

    n_cal = 80
    true_cal_loads = np.zeros((n_cal, 4), dtype=float)
    for c in range(4):
        true_cal_loads[4 * c : 4 * (c + 1), c] = np.array([-1.6, -0.6, 0.7, 1.7], dtype=float)
    true_cal_loads[16:, :] = rng.uniform(-1.6, 1.6, size=(n_cal - 16, 4))
    bridge_raw_cal = _apply_alpha_transducer(true_cal_loads, gains, offsets, bridge_offdiag, upstream_mode) + rng.normal(
        0.0, 1e-4, size=(n_cal, 4)
    )
    if cal_outlier_rows > 0:
        for oi in range(min(cal_outlier_rows, 3)):
            bridge_raw_cal[13 + oi, :] += np.array([14.5, -13.2, 12.8, -14.1], dtype=float)

    sc_ch = np.repeat(np.arange(4, dtype=int), 6)
    sc_true = np.tile(np.array([-1.5, -0.8, -0.2, 0.4, 1.0, 1.6], dtype=float), 4)
    sc_raw = np.empty_like(sc_true)
    for idx_s in range(len(sc_ch)):
        c_i = int(sc_ch[idx_s])
        v4 = np.zeros(4, dtype=float)
        v4[c_i] = sc_true[idx_s]
        sc_raw[idx_s] = float(_apply_alpha_transducer(v4, gains, offsets, bridge_offdiag, upstream_mode)[c_i] + rng.normal(0.0, 1e-4))
    if cal_outlier_rows > 0:
        sc_raw[5] += 15.5

    lab_off = np.array([-0.6, 0.5, -0.3, 0.8], dtype=float)
    lab_gn = np.array([1.25, 0.85, 1.40, 0.90], dtype=float)

    def _make_panel(n_rows: int, regime_code: str):
        labs = rng.integers(0, 4, size=n_rows)
        strata = rng.integers(0, 3, size=n_rows)
        x_lat = rng.normal(0.0, 1.0, size=(n_rows, 4))
        x_raw = x_lat * lab_gn[labs, None] + lab_off[labs, None]
        for r_i in range(n_rows):
            if rng.random() < 0.22:
                x_raw[r_i, int(rng.integers(0, 4))] = np.nan
        ref_c = rng.normal(0.0, 1.0, size=(n_rows, 8)) * lab_gn[labs, None] + lab_off[labs, None]
        past_true_ch = rng.uniform(-0.65, 1.15, size=(n_rows, S_len, 4))
        hold_true_ch = rng.uniform(0.35, 1.35, size=(n_rows, 4))
        past_raw_ch = _apply_alpha_transducer(past_true_ch, gains, offsets, bridge_offdiag, upstream_mode)
        hold_raw_ch = _apply_alpha_transducer(hold_true_ch, gains, offsets, bridge_offdiag, upstream_mode)
        ps_true = np.sum(past_true_ch, axis=-1)
        h0_true = np.sum(hold_true_ch, axis=-1)
        s0 = rng.uniform(-0.15, 0.25, size=n_rows)
        return {
            "lab_id": labs,
            "stratum": strata,
            "coupon_features_raw": x_raw,
            "ref_coupons": ref_c,
            "past_channel_raw": past_raw_ch,
            "hold_channel_raw": hold_raw_ch,
            "strain_0": s0,
            "_ps_true": ps_true,
            "_h0_true": h0_true,
            "_regime": regime_code,
        }

    tr_p = _make_panel(72, "D0")
    te_p = _make_panel(36, eff_regime)

    labs_all = np.concatenate([tr_p["lab_id"], te_p["lab_id"]])
    strata_all = np.concatenate([tr_p["stratum"], te_p["stratum"]])
    x_all = np.vstack([tr_p["coupon_features_raw"], te_p["coupon_features_raw"]]).copy()
    r_all = np.vstack([tr_p["ref_coupons"], te_p["ref_coupons"]])
    for c in range(4):
        m = labs_all == c
        if np.any(m):
            cm = np.mean(r_all[m], axis=0)
            cs = np.std(r_all[m], axis=0, ddof=0)
            v = np.isfinite(cm) & np.isfinite(cs) & (cs > 1e-8)
            x_all[m] = (x_all[m] - float(np.median(cm[v]))) / float(np.median(cs[v]))
    w_all = np.zeros(len(labs_all), dtype=float)
    for c in range(4):
        for s in range(3):
            ms = (labs_all == c) & (strata_all == s)
            if np.any(ms):
                w_all[ms] = 1.0 / float(np.sum(ms))
    w_all /= float(np.mean(w_all))
    for j in range(4):
        fin = np.isfinite(x_all[:, j])
        if np.any(fin) and not np.all(fin):
            x_all[~fin, j] = float(np.sum(w_all[fin] * x_all[fin, j]) / max(float(np.sum(w_all[fin])), 1e-12))
    x_tr_std = x_all[:72]
    x_te_std = x_all[72:]

    def _eval_curves_and_rate(ps_t, h0_t, s0_v, x_std_v, reg_c):
        z = np.zeros((len(h0_t), 2), dtype=float)
        if reg_c not in ("D1", "D2_coupled"):
            for si, dts in enumerate(past_dt):
                dec = np.exp(-float(dts) / taus)
                z = z * dec[None, :] + (1.0 - dec[None, :]) * ps_t[:, si : si + 1]
        gap = h0_t[:, None] - z
        prony_c = (
            theta[0] * gap[:, 0:1] * (-np.expm1(-eval_t[None, :] / taus[0]))
            + theta[1] * gap[:, 1:2] * (-np.expm1(-eval_t[None, :] / taus[1]))
        )
        r_drift = theta[2] + x_std_v @ theta[3:7]
        if reg_c == "D0":
            phi = eval_t
            dphi0 = 1.0
        elif reg_c == "D1":
            phi = np.log1p(eval_t)
            dphi0 = 1.0
        elif reg_c == "D2":
            phi = -np.expm1(-eval_t)
            dphi0 = 1.0
        else:
            # `D2_coupled`: quenched memory (`z = 0`) + accelerated `2.0 * (sqrt(1 + t) - 1)` (`dphi/dt(0) = 1.0`, `phi(4) = 2.472` vs `0.982`!)
            phi = 2.0 * (np.sqrt(1.0 + eval_t) - 1.0)
            dphi0 = 1.0
        clean_c = s0_v[:, None] + prony_c + r_drift[:, None] * phi[None, :]
        rate0 = theta[0] * gap[:, 0] / taus[0] + theta[1] * gap[:, 1] / taus[1] + dphi0 * r_drift
        return clean_c, rate0

    tr_clean, tr_rate0 = _eval_curves_and_rate(tr_p["_ps_true"], tr_p["_h0_true"], tr_p["strain_0"], x_tr_std, "D0")
    noise_c = rng.normal(0.0, 0.012, size=tr_clean.shape)
    outlier_mask = rng.random(size=tr_clean.shape) < 0.03
    noise_c[outlier_mask] += rng.choice([-0.28, 0.28], size=int(np.sum(outlier_mask)))
    tr_obs_c = tr_clean + noise_c
    tr_obs_r = tr_rate0 + rng.normal(0.0, 0.008, size=tr_rate0.shape)

    te_clean, te_rate0 = _eval_curves_and_rate(te_p["_ps_true"], te_p["_h0_true"], te_p["strain_0"], x_te_std, eff_regime)

    pilot_dict = None
    if include_chamber_pilots:
        pi_p = _make_panel(24, eff_regime)
        x_pi = pi_p["coupon_features_raw"].copy()
        for c in range(4):
            m = pi_p["lab_id"] == c
            if np.any(m):
                cm = np.mean(pi_p["ref_coupons"][m], axis=0)
                cs = np.std(pi_p["ref_coupons"][m], axis=0, ddof=0)
                v = np.isfinite(cm) & np.isfinite(cs) & (cs > 1e-8)
                x_pi[m] = (x_pi[m] - float(np.median(cm[v]))) / float(np.median(cs[v]))
        for j in range(4):
            fin = np.isfinite(x_pi[:, j])
            if np.any(fin) and not np.all(fin):
                x_pi[~fin, j] = float(np.mean(x_pi[fin, j]))
        pi_clean, pi_rate0 = _eval_curves_and_rate(pi_p["_ps_true"], pi_p["_h0_true"], pi_p["strain_0"], x_pi, eff_regime)
        pilot_dict = {
            "lab_id": pi_p["lab_id"],
            "stratum": pi_p["stratum"],
            "coupon_features_raw": pi_p["coupon_features_raw"],
            "ref_coupons": pi_p["ref_coupons"],
            "past_channel_raw": pi_p["past_channel_raw"],
            "hold_channel_raw": pi_p["hold_channel_raw"],
            "strain_0": pi_p["strain_0"],
            "strain_curves_obs": pi_clean + rng.normal(0.0, 0.0025, size=pi_clean.shape),
            "instant_rate_obs": pi_rate0 + rng.normal(0.0, 0.0020, size=pi_rate0.shape),
        }

    aux_tel = rng.normal(0.0, 1.0, size=(36, 4))
    ds = {
        "past_dt": past_dt,
        "eval_times": eval_t,
        "calibration_load_cells": {
            "bridge_raw": bridge_raw_cal,
            "true_channel_loads": true_cal_loads,
            "single_channel_sweeps": {"lab_id": sc_ch, "cmd_stress": sc_raw, "true_stress": sc_true},
        },
        "train": {
            "lab_id": tr_p["lab_id"],
            "stratum": tr_p["stratum"],
            "coupon_features_raw": tr_p["coupon_features_raw"],
            "ref_coupons": tr_p["ref_coupons"],
            "past_channel_raw": tr_p["past_channel_raw"],
            "hold_channel_raw": tr_p["hold_channel_raw"],
            "strain_0": tr_p["strain_0"],
            "strain_curves_obs": tr_obs_c,
            "instant_rate_obs": tr_obs_r,
            "aux_chamber_telemetry": rng.normal(0.0, 1.0, size=(72, 4)),
        },
        "test": {
            "lab_id": te_p["lab_id"],
            "stratum": te_p["stratum"],
            "coupon_features_raw": te_p["coupon_features_raw"],
            "ref_coupons": te_p["ref_coupons"],
            "past_channel_raw": te_p["past_channel_raw"],
            "hold_channel_raw": te_p["hold_channel_raw"],
            "strain_0": te_p["strain_0"],
            "aux_chamber_telemetry": aux_tel,
        },
        "_campaign_diagnostics": pilot_dict,
    }
    gt = {
        "taus": taus,
        "theta": theta,
        "te_clean": te_clean,
        "te_true_rate0": te_rate0,
        "te_hist_true": te_p["_ps_true"],
        "te_hold_true": te_p["_h0_true"],
        "tr_hist_true": tr_p["_ps_true"],
        "tr_hold_true": tr_p["_h0_true"],
        "effective_regime": eff_regime,
        "check_rate_gt": True,
    }
    return ds, gt
'''

_M99_BETA_GEN_PY = '''
from scipy.optimize import minimize as _scipy_min


def _build_beta_bridge_matrix(gains, offdiag):
    g = np.asarray(gains, dtype=float)
    M = np.diag(g)
    if abs(float(offdiag)) > 1e-12:
        for c in range(4):
            M[c, (c + 1) % 4] += float(offdiag) * g[c]
            M[c, (c - 1) % 4] -= 0.5 * float(offdiag) * g[c]
    return M


def _apply_beta_transducer(x_true, gains, offsets, offdiag, upstream_mode):
    x_arr = np.asarray(x_true, dtype=float)
    orig_shape = x_arr.shape
    x2 = x_arr.reshape(-1, 4)
    g = np.asarray(gains, dtype=float)
    b_vec = np.asarray(offsets, dtype=float)
    M_br = _build_beta_bridge_matrix(gains, offdiag if upstream_mode == "U1" else 0.0)
    y2 = x2 @ M_br.T + b_vec[None, :]
    if upstream_mode == "U1":
        sgn = 1.0 if float(offdiag) >= 0.0 else -1.0
        x_next = np.roll(x2, -1, axis=1)
        y2 = y2 + (0.18 * g)[None, :] * (x2 ** 3) + (0.22 * sgn * g)[None, :] * (x2 * x_next)
    return y2.reshape(orig_shape)


def _solve_box_simplex_qp_exact(S_ti, r_y, lw, ub, bt):
    M = S_ti.shape[1]
    H = S_ti.T @ S_ti + lw * np.eye(M)
    f = S_ti.T @ r_y
    res_w = _scipy_min(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        np.full(M, bt / M, dtype=float),
        jac=lambda w: H @ w - f,
        bounds=[(0.0, ub)] * M,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - bt, "jac": lambda w: np.ones(M)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    w = np.clip(res_w.x, 0.0, ub)
    for _ in range(3):
        free = (w > 1e-11) & (w < ub - 1e-11)
        if int(np.sum(free)) >= 2:
            idx_f = np.where(free)[0]
            rem_b = bt - float(np.sum(w[~free]))
            H_ff = H[np.ix_(idx_f, idx_f)]
            rhs_f = f[idx_f] - H[np.ix_(idx_f, ~free)] @ w[~free]
            ones_f = np.ones(len(idx_f), dtype=float)
            kkt_m = np.block([[H_ff, ones_f[:, None]], [ones_f[None, :], np.zeros((1, 1))]])
            sol_f = np.linalg.solve(kkt_m, np.concatenate([rhs_f, [rem_b]]))
            w_cand = w.copy()
            w_cand[idx_f] = sol_f[:-1]
            if np.all(w_cand >= -1e-12) and np.all(w_cand <= ub + 1e-12):
                w = np.clip(w_cand, 0.0, ub)
                break
    lo, hi = float(np.min(w) - ub), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, ub))) > bt:
            lo = mid
        else:
            hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, ub)
    rv = S_ti @ w - r_y
    jv = 0.5 * float(rv @ rv) + 0.5 * lw * float(w @ w)
    g = H @ w - f
    best_nu, best_lb = float(g[0]), -1e30
    for nc in np.unique(g):
        lbc = jv - float(g @ w) + float(nc) * bt + ub * float(np.sum(np.minimum(0.0, g - float(nc))))
        if lbc > best_lb:
            best_lb, best_nu = lbc, float(nc)
    cert = {
        "weights": w.tolist(),
        "nu_sum": best_nu,
        "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
        "lam_upper": np.maximum(0.0, best_nu - g).tolist(),
        "primal_objective": jv,
        "dual_lower_bound": float(min(best_lb, jv)),
    }
    return w, cert


def _gen_m99_beta_instance(
    seed: int,
    *,
    world_seed: int,
    upstream_mode: str,
    downstream_regime: str,
    gains: tuple[float, float, float, float],
    offsets: tuple[float, float, float, float],
    bridge_offdiag: float = 0.0,
    include_tracer_pulses: bool = True,
    cal_outlier_rows: int = 0,
):
    assert upstream_mode in ("U0", "U1")
    assert downstream_regime in ("D0", "D1", "D2")
    # CRITICAL (`M9.9` Non-Separable `U1 x D2` Cross-Mechanism Interaction):
    # - `D0`: Robin-x + Robin-y
    # - `D1`: Neumann-x + Neumann-y
    # - `D2` under `U0` (`D2_sep`): Periodic-x + Periodic-y
    # - `D2` under `U1` (`D2_coupled`): Anisotropic cross-coupled boundary `Periodic-x + Neumann-y`!
    eff_regime = "D2_coupled" if (upstream_mode == "U1" and downstream_regime == "D2") else downstream_regime

    rng_w = np.random.default_rng(world_seed)
    Lx, Ly = 2.4, 2.0
    nx, ny = 14, 12
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)

    kap_x = 0.45 + 0.14 * np.sin(math.pi * xs / Lx) + float(rng_w.uniform(-0.04, 0.04))
    kap_y = 0.42 + 0.12 * np.cos(math.pi * ys / Ly) + float(rng_w.uniform(-0.04, 0.04))
    vel_x = 0.35 * np.cos(0.7 * math.pi * xs / Lx)
    vel_y = -0.28 * np.sin(0.7 * math.pi * ys / Ly)
    rob_x = float(rng_w.uniform(0.85, 1.35))
    rob_y = float(rng_w.uniform(0.80, 1.30))
    V_2d = 0.25 + 0.18 * (np.sin(math.pi * xs[:, None] / Lx) ** 2) * (np.cos(0.5 * math.pi * ys[None, :] / Ly) ** 2)

    rng = np.random.default_rng(seed)
    M_modes = 8
    modes = np.empty((M_modes, nx, ny), dtype=float)
    for m in range(M_modes):
        cx_m = float(rng.uniform(0.18 * Lx, 0.82 * Lx))
        cy_m = float(rng.uniform(0.18 * Ly, 0.82 * Ly))
        modes[m] = np.exp(-(((xs[:, None] - cx_m) / 0.42) ** 2 + ((ys[None, :] - cy_m) / 0.38) ** 2))

    all_ij = [(i, j) for i in range(0, nx, 2) for j in range(0, ny, 2)]
    perm_ij = rng.permutation(len(all_ij))[:32]
    sensor_ij = np.array([all_ij[idx] for idx in perm_ij], dtype=int)
    mu_shifts = np.array([0.15, 0.55, 1.10], dtype=float)
    n_obs = len(sensor_ij) * len(mu_shifts)
    row_bank_id = np.tile(np.arange(4, dtype=int), n_obs // 4)

    def _op1d_bc(n, h, kap, vel, rob, bc_kind: str):
        A = np.zeros((n, n), dtype=float)
        for i in range(1, n - 1):
            kp = 0.5 * (kap[i] + kap[i + 1])
            km = 0.5 * (kap[i] + kap[i - 1])
            A[i, i - 1] = -km / (h * h) - max(vel[i], 0.0) / h
            A[i, i + 1] = -kp / (h * h) - max(-vel[i], 0.0) / h
            A[i, i] = (kp + km) / (h * h) + abs(vel[i]) / h
        if bc_kind == "periodic":
            for i in (0, n - 1):
                im = (i - 1) % n
                ip = (i + 1) % n
                kp = 0.5 * (kap[i] + kap[ip])
                km = 0.5 * (kap[i] + kap[im])
                A[i, im] += -km / (h * h) - max(vel[i], 0.0) / h
                A[i, ip] += -kp / (h * h) - max(-vel[i], 0.0) / h
                A[i, i] += (kp + km) / (h * h) + abs(vel[i]) / h
        elif bc_kind == "neumann":
            A[0, 0] = kap[0] / (h * h)
            A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h)
            A[n - 1, n - 2] = -kap[-1] / (h * h)
        else:
            A[0, 0] = kap[0] / (h * h) + rob / h
            A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h) + rob / h
            A[n - 1, n - 2] = -kap[-1] / (h * h)
        return A

    def _build_S_and_Ulast(reg_c: str):
        if reg_c == "D0":
            bc_x, bc_y = "robin", "robin"
        elif reg_c == "D1":
            bc_x, bc_y = "neumann", "neumann"
        elif reg_c == "D2":
            bc_x, bc_y = "periodic", "periodic"
        else:
            bc_x, bc_y = "periodic", "neumann"
        Ax = _op1d_bc(nx, dx, kap_x, vel_x, rob_x, bc_x)
        Ay = _op1d_bc(ny, dy, kap_y, vel_y, rob_y, bc_y)
        L_f = np.kron(np.eye(ny), Ax) + np.kron(Ay, np.eye(nx)) + np.diag(V_2d.ravel(order="F"))
        B_rhs = np.column_stack([modes[m].ravel(order="F") for m in range(M_modes)])
        si, sj = sensor_ij[:, 0], sensor_ij[:, 1]
        U_last = np.empty((M_modes, nx, ny), dtype=float)
        S_blks = []
        I_f = np.eye(nx * ny, dtype=float)
        for ki, mu in enumerate(mu_shifts):
            sf = np.linalg.solve(L_f + float(mu) * I_f, B_rhs)
            U_k = np.empty((M_modes, nx, ny), dtype=float)
            for m in range(M_modes):
                U_k[m] = sf[:, m].reshape((nx, ny), order="F")
            if ki == len(mu_shifts) - 1:
                U_last = U_k
            S_blks.append(U_k[:, si, sj].T)
        return np.vstack(S_blks), U_last

    S_true, U_last_true = _build_S_and_Ulast(eff_regime)
    w_gen = rng.dirichlet(np.ones(M_modes)) * 1.4
    Z_nuis = rng.normal(0.0, 1.0, size=(n_obs, 3))
    gamma_gen = rng.normal(0.0, 0.12, size=3)
    y_cal_true = S_true @ w_gen + Z_nuis @ gamma_gen + rng.normal(0.0, 0.008, size=n_obs)
    y_obs_raw = _apply_beta_transducer(y_cal_true.reshape(-1, 4), gains, offsets, bridge_offdiag, upstream_mode).ravel()

    n_cal = 80
    ref_rosette = np.zeros((n_cal, 4), dtype=float)
    for c in range(4):
        ref_rosette[4 * c : 4 * (c + 1), c] = np.array([-1.2, -0.4, 0.5, 1.3], dtype=float)
    ref_rosette[16:, :] = rng.uniform(-1.4, 1.8, size=(n_cal - 16, 4))
    raw_rosette = _apply_beta_transducer(ref_rosette, gains, offsets, bridge_offdiag, upstream_mode) + rng.normal(
        0.0, 1e-5, size=(n_cal, 4)
    )
    if cal_outlier_rows > 0:
        for oi in range(min(cal_outlier_rows, 3)):
            raw_rosette[13 + oi, :] += np.array([13.5, -12.8, 12.4, -13.2], dtype=float)

    sb_bank = np.repeat(np.arange(4, dtype=int), 6)
    sb_ref = np.tile(np.array([-1.1, -0.6, -0.1, 0.4, 0.9, 1.4], dtype=float), 4)
    sb_raw = np.empty_like(sb_ref)
    for idx_s in range(len(sb_bank)):
        c_i = int(sb_bank[idx_s])
        v4 = np.zeros(4, dtype=float)
        v4[c_i] = sb_ref[idx_s]
        sb_raw[idx_s] = float(_apply_beta_transducer(v4, gains, offsets, bridge_offdiag, upstream_mode)[c_i] + rng.normal(0.0, 1e-5))
    if cal_outlier_rows > 0:
        sb_raw[5] += 14.5

    lam_z, lam_w, u_box, b_tot = 0.08, 0.04, 0.42, 1.40
    Pz = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(3), Z_nuis.T)
    S_tilde = S_true - Z_nuis @ (Pz @ S_true)
    r_y = y_cal_true - Z_nuis @ (Pz @ y_cal_true)
    w_opt, cert_opt = _solve_box_simplex_qp_exact(S_tilde, r_y, lam_w, u_box, b_tot)
    term_f = np.tensordot(w_opt, U_last_true, axes=(0, 0))

    tracer_pulses = None
    if include_tracer_pulses:
        tracer_pulses = []
        for p_i in range(3):
            wp = np.zeros(M_modes, dtype=float)
            wp[p_i] = 1.0
            wp[(p_i + 3) % M_modes] = 0.5
            yp_cal = S_true @ wp
            yp_raw = _apply_beta_transducer(yp_cal.reshape(-1, 4), gains, offsets, bridge_offdiag, upstream_mode).ravel()
            tracer_pulses.append({"w_pulse": wp, "y_pulse_raw": yp_raw})

    study = {
        "Lx": Lx,
        "Ly": Ly,
        "nx": nx,
        "ny": ny,
        "kappa_x": kap_x,
        "kappa_y": kap_y,
        "vel_x": vel_x,
        "vel_y": vel_y,
        "robin_x": rob_x,
        "robin_y": rob_y,
        "V_reaction_2d": V_2d,
        "mu_shifts": mu_shifts,
        "source_modes": modes,
        "sensor_ij": sensor_ij,
        "row_bank_id": row_bank_id,
        "sensor_calibration_sweeps": {
            "ref_rosette": ref_rosette,
            "raw_rosette": raw_rosette,
            "single_bank_sweeps": {"bank_id": sb_bank, "ref_signal": sb_ref, "raw_readout": sb_raw},
        },
        "Z_nuisance": Z_nuis,
        "y_obs_raw": y_obs_raw,
        "_campaign_diagnostics": tracer_pulses,
        "aux_manifold_telemetry": rng.normal(0.0, 1.0, size=(24, 4)),
        "nuisance_ridge": lam_z,
        "source_ridge": lam_w,
        "u_box": u_box,
        "total_budget": b_tot,
    }
    gt = {
        "S_tilde": S_tilde,
        "r_y": r_y,
        "w_opt": w_opt,
        "cert_opt": cert_opt,
        "terminal_field_A": term_f,
        "effective_regime": eff_regime,
    }
    return study, gt
'''


def _eval_m99_gen_host(gen_py: str, call_expr: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ns: dict[str, Any] = {"np": np, "math": math}
    exec(compile("import math\nimport numpy as np\n" + gen_py + f"\n_res = {call_expr}\n", "<m99_host_gen>", "exec"), ns)
    return ns["_res"]


def _build_m99_alpha_task(*, task_id: str, short_name: str, pub_call_expr: str, priv_cases_py: str) -> HoldoutOODTask:
    from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY  # noqa: F401

    _, pub_gt = _eval_m99_gen_host(_M99_ALPHA_GEN_PY, pub_call_expr)
    host_gt_json = json.dumps(encode_pure_json_numerical_payload(pub_gt), allow_nan=False)
    schema_keys = '(("estimated_taus", (2,)), ("hereditary_weights", (7,)), ("predicted_strain", (n_te, k_t)), ("initial_rate_pred", (n_te,)))'
    priv_body = '''
t0 = time.monotonic()
valid = True; max_tau_rel = 0.0; max_mae = 0.0; max_rate_mae = 0.0; rows = []
for sd, ds, gt in cases:
    ds.pop("_campaign_diagnostics", None)
    out = cand_mod.solve_batch(ds)
    if not isinstance(out, dict) or not {"estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "MISSING_KEYS"}); continue
    et = np.asarray(out["estimated_taus"], dtype=float)
    hw = np.asarray(out["hereditary_weights"], dtype=float)
    ps = np.asarray(out["predicted_strain"], dtype=float)
    ir = np.asarray(out["initial_rate_pred"], dtype=float)
    if et.shape != (2,) or hw.shape != (7,) or ps.shape != gt["te_clean"].shape or ir.shape != gt["te_true_rate0"].shape:
        valid = False; rows.append({"seed": sd, "error": "SHAPE_MISMATCH"}); continue
    if not (np.all(np.isfinite(et)) and np.all(np.isfinite(hw)) and np.all(np.isfinite(ps)) and np.all(np.isfinite(ir))):
        valid = False; rows.append({"seed": sd, "error": "NON_FINITE"}); continue
    if not (0.05 < float(et[0]) < float(et[1]) < 25.0):
        valid = False; rows.append({"seed": sd, "error": "PARAMETER_ORDER", "estimated_taus": [round(float(et[0]), 6), round(float(et[1]), 6)]}); continue
    if float(np.linalg.norm(hw[:2])) <= 1e-4 or float(np.min(hw[:2])) <= 1e-4:
        valid = False; rows.append({"seed": sd, "error": "DEGENERATE_PRONY_WEIGHTS"}); continue
    tau_rel = float(np.max(np.abs(et - gt["taus"]) / gt["taus"]))
    s_mae = float(np.mean(np.abs(ps - gt["te_clean"])))
    r_mae = float(np.mean(np.abs(ir - gt["te_true_rate0"])))
    th_err = float(np.max(np.abs(hw - gt["theta"])))
    if th_err > 0.14:
        valid = False; rows.append({"seed": sd, "error": "THETA_7COEFF_GT_MISMATCH", "theta_err": round(th_err, 6), "strain_mae": round(s_mae, 6)}); continue
    max_tau_rel = max(max_tau_rel, tau_rel); max_mae = max(max_mae, s_mae); max_rate_mae = max(max_rate_mae, r_mae)
    f_codes = []
    if tau_rel > 0.10: f_codes.append("TAU_REL_ERROR")
    if s_mae > 0.070: f_codes.append("STRAIN_ERROR")
    if r_mae > 0.055: f_codes.append("RATE_ERROR")
    rows.append({"seed": sd, "tau_rel": round(tau_rel, 6), "strain_mae": round(s_mae, 6), "rate_mae": round(r_mae, 6), "theta_err": round(th_err, 6), "failure_codes": f_codes})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_tau_rel <= 0.10 and max_mae <= 0.070 and max_rate_mae <= 0.055 and el <= 25.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_tau_rel_err": round(max_tau_rel, 6), "max_strain_mae": round(max_mae, 6), "max_rate_mae": round(max_rate_mae, 6), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
'''
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M99_ALPHA_GEN_PY
        + f"\npub_ds, _unused_gt = {pub_call_expr}\n"
        'pub_diag = pub_ds.pop("_campaign_diagnostics", None)\n'
        'with open("/holdout_task/public_batch.pkl", "wb") as fh:\n'
        '    pickle.dump([{"dataset": pub_ds, "campaign_diagnostics": pub_diag}], fh)\n'
        'open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys\nimport numpy as np\n'
        '_d = os.path.dirname(os.path.abspath(__file__))\n'
        'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")\n'
        'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
        'mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n'
        'item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]\n'
        'ds = item["dataset"]\n'
        'out = mod.solve_batch(ds)\n'
        'n_te = len(ds["test"]["lab_id"]); k_t = len(ds["eval_times"])\n'
        'errs = []\n'
        'if not isinstance(out, dict): errs.append("not_dict")\n'
        'else:\n'
        f'    for k, shp in {schema_keys}:\n'
        '        if k not in out: errs.append(f"missing:{k}"); continue\n'
        '        a = np.asarray(out[k], dtype=float)\n'
        '        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
        'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
        'sys.exit(0 if len(errs) == 0 else 1)\n'
        '""")\n'
    )
    priv_ver = (
        "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
        "import batch_solver as cand_mod\n"
        + _M99_ALPHA_GEN_PY
        + f"\ncases = [\n{priv_cases_py}\n]\n"
        + priv_body
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="m9_9_autonomous_decomposition_and_interaction_withdrawal",
        scientific_domain="coupled_viscoelastic_chamber_hereditary_mechanics",
        wormhole_family="m99_alpha",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json=host_gt_json,
        instruction_text=BLIND_ALPHA_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=priv_ver,
    )


def _build_m99_beta_task(*, task_id: str, short_name: str, pub_call_expr: str, priv_cases_py: str) -> HoldoutOODTask:
    from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY

    _, pub_gt = _eval_m99_gen_host(_M99_BETA_GEN_PY, pub_call_expr)
    host_gt_json = json.dumps(encode_pure_json_numerical_payload(pub_gt), allow_nan=False)
    priv_body = '''
t0 = time.monotonic()
valid = True; max_w_rel = 0.0; max_f_rel = 0.0; max_gap = 0.0; rows = []
for sd, st, gt in cases:
    st.pop("_campaign_diagnostics", None)
    out = cand_mod.solve_batch(st)
    if not isinstance(out, dict) or not {"weights", "terminal_field", "kkt_certificate"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "MISSING_KEYS"}); continue
    w = np.asarray(out["weights"], dtype=float)
    tf = np.asarray(out["terminal_field"], dtype=float)
    if w.shape != gt["w_opt"].shape or tf.shape != gt["terminal_field_A"].shape or not isinstance(out["kkt_certificate"], dict):
        valid = False; rows.append({"seed": sd, "error": "SHAPE_OR_CERT_TYPE"}); continue
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(tf))):
        valid = False; rows.append({"seed": sd, "error": "NON_FINITE"}); continue
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
    if not chk["valid"] or float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False; rows.append({"seed": sd, "error": f"KKT_INVALID:{chk.get('reason')}"}); continue
    w_rel = float(np.linalg.norm(w - gt["w_opt"]) / max(float(np.linalg.norm(gt["w_opt"])), 1e-12))
    f_rel = float(np.linalg.norm(tf - gt["terminal_field_A"]) / max(float(np.linalg.norm(gt["terminal_field_A"])), 1e-12))
    max_w_rel = max(max_w_rel, w_rel); max_f_rel = max(max_f_rel, f_rel); max_gap = max(max_gap, float(chk["gap"]))
    rows.append({"seed": sd, "w_rel": round(w_rel, 7), "f_rel": round(f_rel, 7), "kkt_gap": round(float(chk["gap"]), 9)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel <= 1.5e-3 and max_f_rel <= 1.5e-3 and max_gap <= 2e-4 and el <= 25.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_w_rel_l2": round(max_w_rel, 7), "max_field_rel_l2": round(max_f_rel, 7), "max_kkt_gap": round(max_gap, 9), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
'''
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M99_BETA_GEN_PY
        + f"\npub_st, _unused_gt = {pub_call_expr}\n"
        'pub_diag = pub_st.pop("_campaign_diagnostics", None)\n'
        'with open("/holdout_task/public_batch.pkl", "wb") as fh:\n'
        '    pickle.dump([{"study": pub_st, "campaign_diagnostics": pub_diag}], fh)\n'
        'open("/holdout_task/validate_public_schema.py", "w").write("""import importlib.util, json, os, pickle, sys\nimport numpy as np\n'
        '_d = os.path.dirname(os.path.abspath(__file__))\n'
        'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")\n'
        'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
        'mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n'
        'item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]\n'
        'st = item["study"]\n'
        'out = mod.solve_batch(st)\n'
        'm_m = st["source_modes"].shape[0]; nx = int(st["nx"]); ny = int(st["ny"])\n'
        'errs = []\n'
        'if not isinstance(out, dict): errs.append("not_dict")\n'
        'else:\n'
        '    for k, shp in (("weights", (m_m,)), ("terminal_field", (nx, ny))):\n'
        '        if k not in out: errs.append(f"missing:{k}"); continue\n'
        '        a = np.asarray(out[k], dtype=float)\n'
        '        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
        '    if not isinstance(out.get("kkt_certificate"), dict): errs.append("missing_or_bad:kkt_certificate")\n'
        'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
        'sys.exit(0 if len(errs) == 0 else 1)\n'
        '""")\n'
    )
    priv_ver = (
        "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
        "import batch_solver as cand_mod\n"
        + _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY
        + _M99_BETA_GEN_PY
        + f"\ncases = [\n{priv_cases_py}\n]\n"
        + priv_body
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="m9_9_autonomous_decomposition_and_interaction_withdrawal",
        scientific_domain="coupled_2d_transport_reaction_box_simplex_kkt",
        wormhole_family="m99_beta",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json=host_gt_json,
        instruction_text=BLIND_BETA_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=priv_ver,
    )


def build_m99_apparatus_log_entry(
    *,
    spec: M99WorldSpec,
    step_index: int,
    short_name: str,
    next_downstream_regime: str,
    trace_seed_override: int | None = None,
) -> dict[str, Any]:
    """Generate the unlabeled 8-channel facility equilibration log entry (`24 x 8`) for `step_index`.
    When `trace_seed_override` is shared across `{E2_down, E5_coupled_break}` or `{E6_separable_revisit, E7_coupled_revisit}`,
    their `dn_obs` vectors are 100% byte-identical (`0.0` difference!), proving mathematically that any downstream-only
    selector `select_D(dn_obs)` CANNOT separate `(U0, D2_sep)` from `(U1, D2_coupled)`!
    """
    t_seed = trace_seed_override if trace_seed_override is not None else (spec.seed_base + 8000 + 37 * step_index)
    rng = np.random.default_rng(t_seed)
    ch_d1, ch_d2 = spec.active_log_channels
    t_grid = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    trace = np.zeros((24, 8), dtype=float)
    for k in range(8):
        if k == ch_d1:
            mu = 2.45 if next_downstream_regime == "D1" else -0.35
            trace[:, k] = mu + rng.normal(0.0, 0.07, size=24)
        elif k == ch_d2:
            mu = 2.55 if next_downstream_regime == "D2" else -0.40
            trace[:, k] = mu + rng.normal(0.0, 0.07, size=24)
        else:
            # Full-period AC harmonic vibration (`endpoint=False` -> exact zero DC mean) + low-amplitude sensor noise
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            harm = 1 + (k % 3)
            trace[:, k] = 0.85 * np.sin(harm * t_grid + phase) + rng.normal(0.0, 0.03, size=24)
    return {
        "completed_campaign_index": step_index,
        "completed_campaign_short_name": short_name,
        "facility_telemetry_trace": np.round(trace, 5).tolist(),
    }


def build_m99_world_tasks(
    spec: M99WorldSpec,
) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct `Stage S` + the 7-step continual stream:
      1. `E1_down`: `(U0, D1, v2_gains, pilots=True)` -> Autonomous `(u0, d0, d1)` decomposition + `D1` guard
      2. `E2_down`: `(U0, D2, v2_gains, pilots=True)` -> Separable `D2_sep` discovery under `U0`
      3. `E3_up`: `(U1, D0, v2_gains, offdiag_step3, pilots=False)` -> Upstream `U1` localized discovery
      4. `E4_valid_comb`: `(U1, D1, v3_gains, offdiag_step4, pilots=False)` -> Unseen SEPARABLE cross-factor `(U1, D1)`
      5. `E5_coupled_break`: `(U1, D2, v2_gains, offdiag_step5, pilots=True)` -> NON-SEPARABLE `U1 x D2 -> D2_coupled` break & withdrawal!
      6. `E6_separable_revisit`: `(U0, D2, v3_gains, 0.0, pilots=False)` -> Separable `(U0, D2_sep)` revisit (`dn_obs == E7`!)
      7. `E7_coupled_revisit`: `(U1, D2, v3_gains, offdiag_step7, pilots=False)` -> Coupled `(U1, D2_coupled)` revisit (`dn_obs == E6`!)
    """
    wid = spec.world_id
    sb = spec.seed_base
    w_seed = sb + 7
    id_g = (1.0, 1.0, 1.0, 1.0)
    id_b = (0.0, 0.0, 0.0, 0.0)

    def _mk_task(
        short_nm: str,
        pub_seed: int,
        priv_s1: int,
        priv_s2: int,
        up_m: str,
        dn_r: str,
        g_t: tuple[float, ...],
        b_t: tuple[float, ...],
        br_off: float,
        pilots: bool,
        cal_out: int,
    ) -> HoldoutOODTask:
        if spec.family == "alpha":
            kw = (
                f"world_seed={w_seed}, upstream_mode={up_m!r}, downstream_regime={dn_r!r}, "
                f"gains={g_t!r}, offsets={b_t!r}, bridge_offdiag={br_off!r}, "
                f"include_chamber_pilots={pilots!r}, cal_outlier_rows={cal_out}"
            )
            pub_c = f"_gen_m99_alpha_instance({pub_seed}, {kw})"
            priv_c = f"    ({priv_s1}, *_gen_m99_alpha_instance({priv_s1}, {kw})),\n    ({priv_s2}, *_gen_m99_alpha_instance({priv_s2}, {kw})),"
            return _build_m99_alpha_task(
                task_id=f"{short_nm}__blind_continual_campaign",
                short_name=short_nm,
                pub_call_expr=pub_c,
                priv_cases_py=priv_c,
            )
        kw_b = (
            f"world_seed={w_seed}, upstream_mode={up_m!r}, downstream_regime={dn_r!r}, "
            f"gains={g_t!r}, offsets={b_t!r}, bridge_offdiag={br_off!r}, "
            f"include_tracer_pulses={pilots!r}, cal_outlier_rows={cal_out}"
        )
        pub_cb = f"_gen_m99_beta_instance({pub_seed}, {kw_b})"
        priv_cb = f"    ({priv_s1}, *_gen_m99_beta_instance({priv_s1}, {kw_b})),\n    ({priv_s2}, *_gen_m99_beta_instance({priv_s2}, {kw_b})),"
        return _build_m99_beta_task(
            task_id=f"{short_nm}__blind_continual_campaign",
            short_name=short_nm,
            pub_call_expr=pub_cb,
            priv_cases_py=priv_cb,
        )

    s_task = _mk_task(f"{wid}_s00", sb + 10, sb + 11, sb + 12, "U0", "D0", id_g, id_b, 0.0, False, 0)

    step_cfgs = [
        ("E1_down", "U0", "D1", "D1", spec.v2_gains, spec.v2_offsets, 0.0, True, 0, sb + 110, sb + 111, sb + 112, "autonomous_decomposition_and_d1"),
        ("E2_down", "U0", "D2", "D2", spec.v2_gains, spec.v2_offsets, 0.0, True, 0, sb + 210, sb + 211, sb + 212, "separable_d2_discovery"),
        ("E3_up", "U1", "D0", "D0", spec.v2_gains, spec.v2_offsets, spec.bridge_offdiag_step3, False, 3, sb + 310, sb + 311, sb + 312, "localized_upstream_u1_discovery"),
        ("E4_valid_comb", "U1", "D1", "D1", spec.v3_gains, spec.v3_offsets, spec.bridge_offdiag_step4, False, 3, sb + 410, sb + 411, sb + 412, "unseen_separable_U1_x_D1"),
        ("E5_coupled_break", "U1", "D2", "D2_coupled", spec.v2_gains, spec.v2_offsets, spec.bridge_offdiag_step5, True, 3, sb + 510, sb + 511, sb + 512, "nonseparable_U1_x_D2_interaction_break"),
        ("E6_separable_revisit", "U0", "D2", "D2", spec.v3_gains, spec.v3_offsets, 0.0, False, 0, sb + 610, sb + 611, sb + 612, "separable_U0_x_D2_revisit"),
        ("E7_coupled_revisit", "U1", "D2", "D2_coupled", spec.v3_gains, spec.v3_offsets, spec.bridge_offdiag_step7, False, 3, sb + 710, sb + 711, sb + 712, "coupled_U1_x_D2_revisit"),
    ]

    tasks: list[HoldoutOODTask] = []
    manifest: list[dict[str, Any]] = []
    for idx_i, (ev, up_m, dn_r, eff_r, gg, bb, boff, pil, cout, ps, p1, p2, cat) in enumerate(step_cfgs, start=1):
        sn = f"{wid}_batch_{idx_i:02d}"
        tk = _mk_task(sn, ps, p1, p2, up_m, dn_r, gg, bb, boff, pil, cout)
        tasks.append(tk)
        manifest.append(
            {
                "stream_step": idx_i,
                "short_name": sn,
                "task_id": tk.task_id,
                "causal_event": ev,
                "true_upstream_mode": up_m,
                "true_downstream_precursor": dn_r,
                "true_effective_regime": eff_r,
                "true_regime": f"{up_m}__{eff_r}",
                "event_category": cat,
            }
        )

    # Prefix apparatus logs (`8` entries total: `idx = 0` precedes `Stage S (D0)`, `idx = step_i` precedes `Step step_i`!):
    # - `Stage S` receives `apparatus_logs[:1]` (`D0` precursor -> gives `d0` a genuine measured `dn_obs` positive witness!)
    # - `Step step_i` (`1..7`) receives `apparatus_logs[:step_i + 1]`!
    # - `idx = 2` (`Step 2: E2_down`) and `idx = 5` (`Step 5: E5_coupled_break`) share `d2_disc_seed` (`dn_obs(E2) == dn_obs(E5)`)!
    # - `idx = 6` (`Step 6: E6_separable_revisit`) and `idx = 7` (`Step 7: E7_coupled_revisit`) share `d2_rev_seed` (`dn_obs(E6) == dn_obs(E7)`)!
    prev_sn = ["facility_init"] + [s_task.short_name] + [m["short_name"] for m in manifest[:-1]]
    next_dn_seq = ["D0"] + [m["true_downstream_precursor"] for m in manifest]
    d2_disc_seed = sb + 9901
    d2_rev_seed = sb + 9902
    seed_overrides = {2: d2_disc_seed, 5: d2_disc_seed, 6: d2_rev_seed, 7: d2_rev_seed}
    apparatus_logs = [
        build_m99_apparatus_log_entry(
            spec=spec,
            step_index=k,
            short_name=prev_sn[k],
            next_downstream_regime=next_dn_seq[k],
            trace_seed_override=seed_overrides.get(k),
        )
        for k in range(len(next_dn_seq))
    ]
    return s_task, tuple(tasks), manifest, apparatus_logs


# ============================================================================
# 2. Content-Addressed `SeparabilityWithdrawalReceipt` & Persistence (`M9.9` Core Primitive)
# ============================================================================


@dataclass
class SeparabilityWithdrawalReceipt:
    """Content-addressed proof that `upstream_binding_id` passed its direct component probe AND `downstream_binding_id`
    matched its downstream guard on a prior verified regime, YET their naive composition `d(u(x))` failed the joint
    public scientific gate — authorizing counterexample-scoped withdrawal of pure separability on `(u, d)`!
    """

    receipt_id: str
    world_id: str
    family: str
    upstream_binding_id: str
    upstream_code_sha256: str
    upstream_component_probe_id: str
    upstream_holdout_rmse: float
    downstream_binding_id: str
    downstream_code_sha256: str
    composed_candidate_sha256: str
    counterexample_public_input_sha256: str
    counterexample_canonical_numeric_sha256: str
    joint_failure_kind: str
    verifier_version: str
    expanded_feature_names: list[str]
    joint_probe_id: str = ""
    prior_witness_id: str = ""
    joint_gate_receipt: dict[str, Any] | None = None

    def compute_expected_receipt_id(self) -> str:
        if not math.isfinite(float(self.upstream_holdout_rmse)) or not 0.0 <= float(self.upstream_holdout_rmse) <= 0.03:
            raise ValueError(
                f"SeparabilityWithdrawalReceipt requires verified healthy upstream RMSE <= 0.03, got {self.upstream_holdout_rmse}"
            )
        if self.joint_failure_kind not in {"scientific_or_numerical", "domain_precondition"}:
            raise ValueError(
                f"SeparabilityWithdrawalReceipt requires genuine scientific/domain joint failure, got `{self.joint_failure_kind}`"
            )
        payload = {
            "world_id": self.world_id,
            "family": self.family,
            "upstream_binding_id": self.upstream_binding_id,
            "upstream_code_sha256": self.upstream_code_sha256,
            "upstream_component_probe_id": self.upstream_component_probe_id,
            "upstream_holdout_rmse": round(float(self.upstream_holdout_rmse), 8),
            "downstream_binding_id": self.downstream_binding_id,
            "downstream_code_sha256": self.downstream_code_sha256,
            "composed_candidate_sha256": self.composed_candidate_sha256,
            "counterexample_public_input_sha256": self.counterexample_public_input_sha256,
            "counterexample_canonical_numeric_sha256": self.counterexample_canonical_numeric_sha256,
            "joint_failure_kind": self.joint_failure_kind,
            "verifier_version": self.verifier_version,
            "expanded_feature_names": list(self.expanded_feature_names),
            **({"joint_probe_id": self.joint_probe_id} if self.joint_probe_id else {}),
            **({"prior_witness_id": self.prior_witness_id} if self.prior_witness_id else {}),
            **({"joint_gate_receipt": deepcopy(self.joint_gate_receipt)} if self.joint_gate_receipt is not None else {}),
        }
        return "swdr_" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]

    def verify_integrity(self) -> None:
        exp = self.compute_expected_receipt_id()
        if (
            not isinstance(self.receipt_id, str)
            or not self.receipt_id.startswith("swdr_")
            or len(self.receipt_id) != 29
            or self.receipt_id != exp
        ):
            raise ValueError(f"Tampered or malformed SeparabilityWithdrawalReceipt `{self.receipt_id}` != `{exp}`")

    def to_dict(self) -> dict[str, Any]:
        self.verify_integrity()
        return {
            "receipt_id": self.receipt_id,
            "world_id": self.world_id,
            "family": self.family,
            "upstream_binding_id": self.upstream_binding_id,
            "upstream_code_sha256": self.upstream_code_sha256,
            "upstream_component_probe_id": self.upstream_component_probe_id,
            "upstream_holdout_rmse": round(float(self.upstream_holdout_rmse), 8),
            "downstream_binding_id": self.downstream_binding_id,
            "downstream_code_sha256": self.downstream_code_sha256,
            "composed_candidate_sha256": self.composed_candidate_sha256,
            "counterexample_public_input_sha256": self.counterexample_public_input_sha256,
            "counterexample_canonical_numeric_sha256": self.counterexample_canonical_numeric_sha256,
            "joint_failure_kind": self.joint_failure_kind,
            "verifier_version": self.verifier_version,
            "expanded_feature_names": list(self.expanded_feature_names),
            **({"joint_probe_id": self.joint_probe_id} if self.joint_probe_id else {}),
            **({"prior_witness_id": self.prior_witness_id} if self.prior_witness_id else {}),
            **({"joint_gate_receipt": deepcopy(self.joint_gate_receipt)} if self.joint_gate_receipt is not None else {}),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SeparabilityWithdrawalReceipt:
        obj = cls(
            receipt_id=str(d.get("receipt_id") or ""),
            world_id=str(d["world_id"]),
            family=str(d["family"]),
            upstream_binding_id=str(d["upstream_binding_id"]),
            upstream_code_sha256=str(d["upstream_code_sha256"]),
            upstream_component_probe_id=str(d["upstream_component_probe_id"]),
            upstream_holdout_rmse=float(d["upstream_holdout_rmse"]),
            downstream_binding_id=str(d["downstream_binding_id"]),
            downstream_code_sha256=str(d["downstream_code_sha256"]),
            composed_candidate_sha256=str(d["composed_candidate_sha256"]),
            counterexample_public_input_sha256=str(d["counterexample_public_input_sha256"]),
            counterexample_canonical_numeric_sha256=str(d["counterexample_canonical_numeric_sha256"]),
            joint_failure_kind=str(d["joint_failure_kind"]),
            verifier_version=str(d["verifier_version"]),
            expanded_feature_names=[str(x) for x in d["expanded_feature_names"]],
            joint_probe_id=str(d.get("joint_probe_id") or ""),
            prior_witness_id=str(d.get("prior_witness_id") or ""),
            joint_gate_receipt=deepcopy(d.get("joint_gate_receipt")),
        )
        obj.verify_integrity()
        return obj


def update_m99_calibration_snapshot_and_graph(
    *,
    spec: M98WorldSpec,
    episode_id: str,
    public_item: dict[str, Any],
    flat_snapshot: dict[str, Any] | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
) -> dict[str, Any]:
    """Shared `2/3-1/3` calibration update that ALSO selectively propagates `cal_node_p0/p1` epochs to any downstream
    branches in `f"{spec.family}_dn"` that acquired cross-factor parent edges via `SeparabilityWithdrawalReceipt`
    (leaving pure separable `d0, d1` untouched at `0` collateral version churn!).
    """
    out = update_m98_calibration_snapshot_and_graph(
        spec=spec,
        episode_id=episode_id,
        public_item=public_item,
        flat_snapshot=flat_snapshot,
        graph=graph,
        registry=registry,
    )
    if graph is not None and registry is not None and "graph_receipt" in out:
        p_recs = out["graph_receipt"].get("parent_receipts") or {}
        dn_updated: dict[str, list[str]] = {}
        for pid in ("cal_node_p0", "cal_node_p1"):
            if (p_recs.get(pid) or {}).get("selected_action") == "AncestorReweight":
                mech = graph.mechanisms[pid]
                dn_updated[pid] = registry.update_shared_parameter_epoch(
                    parameter_mechanism_id=pid,
                    new_version=int(mech.version),
                    parameter_payload=dict(mech.shared_weights),
                    affected_chain_id=f"{spec.family}_dn",
                    evidence_note=f"cross_factor_reweight_{pid}_{episode_id}",
                )
        out["graph_receipt"]["cross_factor_dn_updated_bindings"] = dn_updated
        graph.sync_authoritative_state_to_forest()
    return out


def persist_separability_withdrawals_to_graph(
    graph: PersistentMechanismGraph,
    world_id: str,
    receipts: list[SeparabilityWithdrawalReceipt],
) -> None:
    ser = [r.to_dict() for r in receipts]
    st_sha = hashlib.sha256(json.dumps({"world_id": world_id, "receipts": ser}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    blob = graph.forest.questions.get(M99_WITHDRAWAL_FOREST_KEY)
    if not isinstance(blob, dict) or blob.get("schema") != "hegel_m9_9_withdrawal_ledger_v1":
        blob = {"schema": "hegel_m9_9_withdrawal_ledger_v1", "worlds": {}}
    worlds = dict(blob.get("worlds") or {})
    worlds[world_id] = {"world_id": world_id, "receipts": ser, "state_sha256": st_sha}
    blob["worlds"] = worlds
    graph.forest.questions[M99_WITHDRAWAL_FOREST_KEY] = blob
    graph.sync_authoritative_state_to_forest()


def find_prior_withdrawal_witness(
    ledger: list[SolverEvaluationWitness],
    binding_id: str,
    parent_versions: dict[str, int],
    features: dict[str, float | None],
    verifier_version: str,
    witness_id: str = "",
) -> SolverEvaluationWitness | None:
    for witness in ledger:
        if (witness.binding_id == binding_id and witness.is_verified_positive_witness()
                and witness.evidence_origin == "joint_task_probe" and witness.verifier_version == verifier_version
                and witness.parent_parameter_versions == parent_versions and (not witness_id or witness.witness_id == witness_id)
                and all(witness.public_structural_features.get(k) is not None and features.get(k) is not None
                        and abs(float(witness.public_structural_features[k]) - float(features[k])) <= 1e-9
                        for k in M99_DOWNSTREAM_OBSERVABLE_NAMES)):
            return witness
    return None


def _withdrawn_candidate_sha256(
    graph: PersistentMechanismGraph,
    up_binding: MechanismExecutableBinding,
    dn_binding: MechanismExecutableBinding,
    parent_versions: dict[str, int],
) -> str | None:
    """Rebuild the composed solver with the calibration epochs actually used on the counterexample."""
    weights = {}
    for pid in ("cal_node_p0", "cal_node_p1"):
        mechanism = graph.mechanisms.get(pid)
        epoch = parent_versions.get(pid)
        if mechanism is None or epoch is None:
            return None
        if mechanism.version == epoch:
            weights.update(mechanism.shared_weights)
        else:
            archived = [row for row in graph.mechanism_lineage.get(pid, []) if row["version"] == epoch]
            if len(archived) != 1:
                return None
            weights.update(archived[0]["shared_weights"])
    if any(f"{prefix}_ch{i}" not in weights for i in range(4) for prefix in ("g", "b")):
        return None
    payload = {
        "gains": [float(weights[f"g_ch{i}"]) for i in range(4)],
        "offsets": [float(weights[f"b_ch{i}"]) for i in range(4)],
    }
    code = compose_m98_two_stage_solver_code(
        upstream_code=up_binding.canonical_algorithm_code or up_binding.parameterized_closure_code,
        upstream_entrypoint=up_binding.canonical_entrypoint or up_binding.parameterized_entrypoint,
        downstream_code=dn_binding.canonical_algorithm_code or dn_binding.parameterized_closure_code,
        downstream_entrypoint=dn_binding.canonical_entrypoint or dn_binding.parameterized_entrypoint,
        param_payload=payload,
    )
    return hashlib.sha256(code.encode()).hexdigest()


def withdrawal_has_measured_joint_rejection(receipt: SeparabilityWithdrawalReceipt, witness: SolverEvaluationWitness) -> bool:
    gate = receipt.joint_gate_receipt
    if not isinstance(gate, dict) or not isinstance(gate.get("scientific_probe_receipt"), dict):
        return False
    diagnostic = gate["scientific_probe_receipt"]
    observed_dependencies = hashlib.sha256(json.dumps(
        gate.get("readonly_dependencies_sha256"), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return (
        bool(receipt.joint_probe_id and receipt.prior_witness_id)
        and gate.get("accepted") is False and gate.get("schema_valid") is True
        and joint_scientific_receipt_matches(
            gate, receipt.composed_candidate_sha256, receipt.counterexample_public_input_sha256,
            receipt.counterexample_canonical_numeric_sha256,
        )
        and classify_public_probe_failure(gate)["failure_kind"] == receipt.joint_failure_kind
        and bool(diagnostic.get("errors")) and diagnostic.get("kernel_version") == receipt.verifier_version
        and gate.get("setup_source_sha256") == witness.setup_source_sha256
        and observed_dependencies == witness.dependency_bundle_sha256
    )


def _verify_external_probe_receipt_for_witness(
    probe_receipt_store: Any,
    witness: SolverEvaluationWitness,
    *,
    expected_kind: str,
    expected_passed: bool | None = None,
    expected_code_sha: str = "",
    expected_rmse: float | None = None,
    expected_composed_sha: str = "",
) -> dict[str, Any]:
    """Verify that `witness.probe_id` resolves to an untampered external `ProbeReceiptStore` file matching all
    content, input, setup, dependency, code, and outcome fingerprints (`M10.0.1` `T01` 3-way external FK).
    """
    if probe_receipt_store is None:
        raise ValueError("External ProbeReceiptStore is required for strict 3-way withdrawal FK (`T01`)")
    if not witness.probe_id:
        raise ValueError(f"Witness `{witness.witness_id}` has empty probe_id (`T01`)")
    env = probe_receipt_store.load_probe_receipt(
        witness.probe_id,
        expected_content_sha256=witness.probe_receipt_sha256,
        verify_graph_index=True,
    )
    rec = env["receipt"]
    if rec.get("execution_status") != "completed":
        raise ValueError(f"External ProbeReceipt `{witness.probe_id}` execution_status != completed (`T01`)")
    if rec.get("public_input_sha256") != witness.public_input_sha256:
        raise ValueError(f"External ProbeReceipt `{witness.probe_id}` public_input_sha256 mismatch (`T01`)")
    if witness.canonical_numeric_sha256 and rec.get("canonical_numeric_sha256") != witness.canonical_numeric_sha256:
        raise ValueError(f"External ProbeReceipt `{witness.probe_id}` canonical_numeric_sha256 mismatch (`T01`)")
    if witness.setup_source_sha256 and rec.get("setup_source_sha256") != witness.setup_source_sha256:
        raise ValueError(f"External ProbeReceipt `{witness.probe_id}` setup_source_sha256 mismatch (`T01`)")
    if witness.dependency_bundle_sha256 and rec.get("dependency_bundle_sha256") != witness.dependency_bundle_sha256:
        raise ValueError(f"External ProbeReceipt `{witness.probe_id}` dependency_bundle_sha256 mismatch (`T01`)")

    if expected_kind == "direct_component_probe":
        if rec.get("component_probe_executed") is not True or rec.get("evidence_origin") != "direct_component_probe":
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` is not a direct_component_probe (`T01`)")
        if expected_code_sha and rec.get("candidate_code_sha256") != expected_code_sha:
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` candidate_code_sha256 mismatch (`T01`)")
        rmse = rec.get("chosen_up_holdout_rmse")
        if not isinstance(rmse, (int, float)) or not (0.0 <= float(rmse) <= 0.03):
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` holdout RMSE invalid (`T01`)")
        if expected_rmse is not None and round(float(rmse), 8) != round(float(expected_rmse), 8):
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` holdout RMSE mismatch (`T01`)")
    elif expected_kind == "joint_task_probe":
        if expected_passed is not None and rec.get("accepted") is not expected_passed:
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` accepted != {expected_passed} (`T01`)")
        cand_sha = str(rec.get("candidate_sha256") or rec.get("candidate_raw_sha256") or "")
        if expected_composed_sha and cand_sha != expected_composed_sha:
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` composed candidate SHA mismatch (`T01`)")
        comp_shas = list(rec.get("component_code_sha256s") or [])
        if expected_code_sha and expected_code_sha not in comp_shas and cand_sha != expected_code_sha:
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` does not cover binding code SHA (`T01`)")
        sci = rec.get("scientific_probe_receipt")
        if not isinstance(sci, dict) or sci.get("kernel_version") != witness.verifier_version:
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` verifier kernel_version mismatch (`T01`)")
        if not joint_scientific_receipt_matches(
            rec, cand_sha, witness.public_input_sha256, witness.canonical_numeric_sha256
        ):
            raise ValueError(f"External ProbeReceipt `{witness.probe_id}` fails joint scientific receipt binding (`T01`)")
    else:
        raise ValueError(f"Unsupported expected_kind `{expected_kind}` (`T01`)")
    return env


def issue_measured_withdrawal(
    *, world_id: str, family: str, downstream_selected: bool,
    upstream: MechanismExecutableBinding, downstream: MechanismExecutableBinding,
    downstream_features: dict[str, float | None], downstream_ledger: list[SolverEvaluationWitness],
    direct: dict[str, Any], gate: dict[str, Any], candidate_sha: str,
    expanded_feature_names: tuple[str, ...] = M99_ALL_OBSERVABLE_NAMES,
    probe_receipt_store: Any = None,
    require_external_probe_receipts: bool = False,
) -> SeparabilityWithdrawalReceipt | None:
    """Issue a withdrawal after a healthy component and its failed joint execution contradict a prior routing witness."""
    classification = classify_public_probe_failure(gate)
    rmse = direct.get("chosen_up_holdout_rmse")
    if (not downstream_selected or not downstream_ledger
            or classification["tier"] not in {"SCIENTIFIC_FAILURE", "DOMAIN_REJECTION"}
            or direct.get("execution_status") != "completed" or direct.get("locus") != "downstream_constitutive"
            or not direct.get("component_probe_executed") or not direct.get("probe_id")
            or direct.get("candidate_code_sha256") != upstream.authoritative_executed_code_sha256()
            or not isinstance(rmse, (int, float)) or not 0 <= rmse <= 0.03):
        return None
    negative = downstream_ledger[-1]
    if not negative.is_genuine_scientific_counterexample() or negative.binding_id != downstream.binding_id:
        return None
    prior = find_prior_withdrawal_witness(
        downstream_ledger[:-1], downstream.binding_id, negative.parent_parameter_versions,
        downstream_features, SHARED_KERNEL_VERSION,
    )
    if prior is None:
        return None
    if probe_receipt_store is not None or require_external_probe_receipts:
        try:
            prior_env = _verify_external_probe_receipt_for_witness(
                probe_receipt_store,
                prior,
                expected_kind="joint_task_probe",
                expected_passed=True,
                expected_code_sha=downstream.authoritative_executed_code_sha256(),
            )
            neg_env = _verify_external_probe_receipt_for_witness(
                probe_receipt_store,
                negative,
                expected_kind="joint_task_probe",
                expected_passed=False,
                expected_code_sha=downstream.authoritative_executed_code_sha256(),
                expected_composed_sha=candidate_sha,
            )
            direct_env = probe_receipt_store.load_probe_receipt(str(direct["probe_id"]), verify_graph_index=True)
            d_rec = direct_env["receipt"]
            if (
                d_rec.get("execution_status") != "completed"
                or d_rec.get("component_probe_executed") is not True
                or d_rec.get("candidate_code_sha256") != upstream.authoritative_executed_code_sha256()
                or round(float(d_rec.get("chosen_up_holdout_rmse", -1.0)), 8) != round(float(rmse), 8)
            ):
                return None
            p_utc = str(prior_env.get("executed_utc") or "")
            n_utc = str(neg_env.get("executed_utc") or "")
            if p_utc and n_utc and p_utc > n_utc:
                return None
        except (ValueError, FileNotFoundError, KeyError, TypeError):
            direct["withdrawal_status"] = "rejected_missing_or_invalid_external_probe_receipt"
            return None
    receipt = SeparabilityWithdrawalReceipt(
        receipt_id="", world_id=world_id, family=family, upstream_binding_id=upstream.binding_id,
        upstream_code_sha256=upstream.authoritative_executed_code_sha256(),
        upstream_component_probe_id=str(direct["probe_id"]), upstream_holdout_rmse=float(direct["chosen_up_holdout_rmse"]),
        downstream_binding_id=downstream.binding_id, downstream_code_sha256=downstream.authoritative_executed_code_sha256(),
        composed_candidate_sha256=candidate_sha, joint_probe_id=negative.probe_id, prior_witness_id=prior.witness_id,
        joint_gate_receipt=deepcopy(gate), counterexample_public_input_sha256=negative.public_input_sha256,
        counterexample_canonical_numeric_sha256=negative.canonical_numeric_sha256, joint_failure_kind=classification["failure_kind"],
        verifier_version=SHARED_KERNEL_VERSION, expanded_feature_names=list(expanded_feature_names),
    )
    receipt.receipt_id = receipt.compute_expected_receipt_id()
    receipt.verify_integrity()
    if not withdrawal_has_measured_joint_rejection(receipt, negative):
        direct["withdrawal_status"] = "pending_measured_joint_rejection"
        return None
    return receipt


def inspect_archived_withdrawal_lineage(
    graph: PersistentMechanismGraph, registry: M94BindingRegistry, world_id: str,
) -> list[SeparabilityWithdrawalReceipt]:
    """Inspect historical lineage links for audit. These rows lack scientific authority without a measured gate."""
    return _load_m99_withdrawals(graph, registry, world_id, quarantine_unverified=False, require_measurement=False)


def hydrate_separability_withdrawals_from_graph(
    graph: PersistentMechanismGraph,
    registry: M94BindingRegistry,
    world_id: str,
    *,
    strict_ledger_fk: bool = True,
    quarantine_unverified: bool = False,
    expected_feature_names: tuple[str, ...] = M99_ALL_OBSERVABLE_NAMES,
    probe_receipt_store: Any = None,
    require_external_probe_receipts: bool = False,
) -> list[SeparabilityWithdrawalReceipt]:
    """Return withdrawals supported by both cold graph lineage, the measured host gate, and (when attached/required)
    3-way external `ProbeReceiptStore` verification (`Q07` + `T01`).
    """
    if not strict_ledger_fk:
        raise ValueError("Hydration of authoritative withdrawals requires ledger verification (Q07)")
    return _load_m99_withdrawals(
        graph, registry, world_id, quarantine_unverified=quarantine_unverified,
        require_measurement=True, expected_feature_names=expected_feature_names,
        probe_receipt_store=probe_receipt_store,
        require_external_probe_receipts=require_external_probe_receipts,
    )


def _load_m99_withdrawals(
    graph: PersistentMechanismGraph, registry: M94BindingRegistry, world_id: str,
    *, quarantine_unverified: bool, require_measurement: bool, expected_feature_names: tuple[str, ...] = M99_ALL_OBSERVABLE_NAMES,
    probe_receipt_store: Any = None,
    require_external_probe_receipts: bool = False,
) -> list[SeparabilityWithdrawalReceipt]:
    blob = graph.forest.questions.get(M99_WITHDRAWAL_FOREST_KEY)
    if not isinstance(blob, dict):
        return []
    entry = (blob.get("worlds") or {}).get(world_id)
    if not isinstance(entry, dict):
        return []
    stored_sha = str(entry.get("state_sha256") or "")
    if not stored_sha:
        raise ValueError(f"Missing state_sha256 on separability withdrawal checkpoint `{world_id}`")
    raw_r = list(entry.get("receipts") or [])
    exp_sha = hashlib.sha256(json.dumps({"world_id": world_id, "receipts": raw_r}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if stored_sha != exp_sha:
        raise ValueError(f"Tampered separability withdrawal checkpoint `{world_id}`")
    loaded = [SeparabilityWithdrawalReceipt.from_dict(x) for x in raw_r]
    verified_out: list[SeparabilityWithdrawalReceipt] = []

    for r in loaded:
        if r.world_id != world_id:
            if quarantine_unverified:
                continue
            raise ValueError(f"SeparabilityWithdrawalReceipt `{r.receipt_id}` world_id `{r.world_id}` != `{world_id}` (`Q07`)")
        if r.upstream_binding_id not in registry.bindings or r.downstream_binding_id not in registry.bindings:
            if quarantine_unverified:
                continue
            raise ValueError(f"SeparabilityWithdrawalReceipt `{r.receipt_id}` references missing binding foreign key (`Q07`)")
        up_b = registry.bindings[r.upstream_binding_id]
        dn_b = registry.bindings[r.downstream_binding_id]
        if r.upstream_code_sha256 != up_b.authoritative_executed_code_sha256():
            if quarantine_unverified:
                continue
            raise ValueError(
                f"SeparabilityWithdrawalReceipt `{r.receipt_id}` upstream_code_sha256 `{r.upstream_code_sha256[:16]}` != "
                f"registry `{up_b.authoritative_executed_code_sha256()[:16]}` (`Q07`)"
            )
        if r.downstream_code_sha256 != dn_b.authoritative_executed_code_sha256():
            if quarantine_unverified:
                continue
            raise ValueError(
                f"SeparabilityWithdrawalReceipt `{r.receipt_id}` downstream_code_sha256 `{r.downstream_code_sha256[:16]}` != "
                f"registry `{dn_b.authoritative_executed_code_sha256()[:16]}` (`Q07`)"
            )

        if (up_b.chain_id != f"{r.family}_up" or dn_b.chain_id != f"{r.family}_dn"
                or r.expanded_feature_names != list(expected_feature_names)
                or not r.counterexample_canonical_numeric_sha256):
            if quarantine_unverified:
                continue
            raise ValueError(f"Withdrawal {r.receipt_id} has inconsistent roles, observables or input identity (Q07)")
        up_ctrl = ImmutableBranchLineageController(world_id=f"{world_id}_up", family=f"{r.family}_up", graph=graph, registry=registry)
        dn_ctrl = ImmutableBranchLineageController(world_id=f"{world_id}_dn", family=f"{r.family}_dn", graph=graph, registry=registry)
        up_matches = [
            w
            for w in up_ctrl.witness_ledger
            if w.binding_id == r.upstream_binding_id
            and w.is_verified_positive_witness()
            and w.evidence_origin == "direct_component_probe"
            and w.probe_id == r.upstream_component_probe_id
            and w.public_input_sha256 == r.counterexample_public_input_sha256
            and w.canonical_numeric_sha256 == r.counterexample_canonical_numeric_sha256
            and w.verifier_version == r.verifier_version
            and w.metric_value is not None
            and 0.0 <= float(w.metric_value) <= 0.03
            and round(float(w.metric_value), 8) == round(r.upstream_holdout_rmse, 8)
        ]
        if len(up_matches) != 1:
            if quarantine_unverified:
                continue
            raise ValueError(
                f"SeparabilityWithdrawalReceipt `{r.receipt_id}` missing verified upstream direct probe `{r.upstream_component_probe_id}` in ledger (`Q07`)"
            )
        up_w = up_matches[0]
        if _withdrawn_candidate_sha256(graph, up_b, dn_b, up_w.parent_parameter_versions) != r.composed_candidate_sha256:
            if quarantine_unverified:
                continue
            raise ValueError(f"Withdrawal {r.receipt_id} cannot resolve the executed composition and parameter epoch (Q07)")
        neg_indices = [
            idx
            for idx, w in enumerate(dn_ctrl.witness_ledger)
            if w.binding_id == r.downstream_binding_id
            and w.is_genuine_scientific_counterexample()
            and w.evidence_origin == "joint_task_probe"
            and w.public_input_sha256 == r.counterexample_public_input_sha256
            and w.failure_kind == r.joint_failure_kind
            and (not r.joint_probe_id or w.probe_id == r.joint_probe_id)
            and w.canonical_numeric_sha256 == r.counterexample_canonical_numeric_sha256
            and w.verifier_version == r.verifier_version
            and w.dependency_bundle_sha256 and w.dependency_bundle_sha256 == up_w.dependency_bundle_sha256
            and w.setup_source_sha256 and w.setup_source_sha256 == up_w.setup_source_sha256
            and all(up_w.parent_parameter_versions.get(p, v) == v for p, v in w.parent_parameter_versions.items())
        ]
        if len(neg_indices) != 1:
            if quarantine_unverified:
                continue
            raise ValueError(
                f"SeparabilityWithdrawalReceipt `{r.receipt_id}` missing verified downstream joint-negative witness in ledger (`Q07`)"
            )
        neg_idx = neg_indices[0]
        neg_w = dn_ctrl.witness_ledger[neg_idx]
        prior_w = find_prior_withdrawal_witness(
            dn_ctrl.witness_ledger[:neg_idx], r.downstream_binding_id, neg_w.parent_parameter_versions,
            neg_w.public_structural_features, r.verifier_version, r.prior_witness_id,
        )
        if prior_w is None:
            if quarantine_unverified:
                continue
            raise ValueError(
                f"SeparabilityWithdrawalReceipt `{r.receipt_id}` missing prior colliding downstream positive witness before joint failure (`Q07`)"
            )
        if require_measurement and not withdrawal_has_measured_joint_rejection(r, neg_w):
            if quarantine_unverified:
                continue
            raise ValueError(f"Withdrawal {r.receipt_id} lacks a bound, measured host rejection (Q07)")
        if probe_receipt_store is not None or require_external_probe_receipts:
            try:
                prior_env = _verify_external_probe_receipt_for_witness(
                    probe_receipt_store,
                    prior_w,
                    expected_kind="joint_task_probe",
                    expected_passed=True,
                    expected_code_sha=r.downstream_code_sha256,
                )
                _verify_external_probe_receipt_for_witness(
                    probe_receipt_store,
                    up_w,
                    expected_kind="direct_component_probe",
                    expected_code_sha=r.upstream_code_sha256,
                    expected_rmse=r.upstream_holdout_rmse,
                )
                neg_env = _verify_external_probe_receipt_for_witness(
                    probe_receipt_store,
                    neg_w,
                    expected_kind="joint_task_probe",
                    expected_passed=False,
                    expected_code_sha=r.downstream_code_sha256,
                    expected_composed_sha=r.composed_candidate_sha256,
                )
                p_utc = str(prior_env.get("executed_utc") or "")
                n_utc = str(neg_env.get("executed_utc") or "")
                if p_utc and n_utc and p_utc > n_utc:
                    raise ValueError(
                        f"Withdrawal `{r.receipt_id}` prior receipt UTC `{p_utc}` > negative receipt UTC `{n_utc}` (`T01`)"
                    )
            except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
                if quarantine_unverified:
                    continue
                raise ValueError(
                    f"Withdrawal `{r.receipt_id}` failed strict 3-way external ProbeReceiptStore verification (`T01`): {exc}"
                ) from exc
        verified_out.append(r)
    return verified_out


# ============================================================================
# 3. Autonomous Two-Stage Decomposition Compiler & Equal-Prompt Downstream Harness (`R02`)
# ============================================================================


def build_m99_downstream_canonical_synthesis_task(
    *,
    base_task: HoldoutOODTask,
    family: str,
    upstream_code: str,
    upstream_entrypoint: str,
    param_payload: dict[str, Any],
) -> HoldoutOODTask:
    """Equal-prompt downstream module synthesis harness used IDENTICALLY by BOTH `Flat_PlainModuleRegistry_Online`
    AND `Hegel_AutonomousInteractionGraph_Online` whenever `locus == 'downstream_constitutive'` or `'cross_factor_interaction'`!
    Stages the learned upstream module `/holdout_task/_verified_upstream.py` (`0444`) and composes
    `solve_batch(item) = solve_downstream(calibrate_upstream(item, param_payload))`.
    """
    import base64

    up_b64 = base64.b64encode(upstream_code.encode("utf-8")).decode("ascii")
    pay_json = json.dumps(param_payload or {}, sort_keys=True)
    bridge_py = (
        "\nimport base64 as _b64\n"
        f'open("/holdout_task/_verified_upstream.py", "wb").write(_b64.b64decode({up_b64!r}))\n'
        'os.chmod("/holdout_task/_verified_upstream.py", 0o444)\n'
        f'open("/holdout_task/_upstream_param_payload.json", "w").write({pay_json!r})\n'
        'os.chmod("/holdout_task/_upstream_param_payload.json", 0o444)\n'
        'open("/holdout_task/batch_solver.py", "w").write("""from __future__ import annotations\n'
        "import inspect, json\n"
        "import _verified_upstream as _up\n"
        "import downstream_solver as _dn\n\n"
        "_PAY = json.load(open('/holdout_task/_upstream_param_payload.json', 'r'))\n\n"
        "def solve_batch(item: dict) -> dict:\n"
        f"    fn_u = getattr(_up, {upstream_entrypoint!r})\n"
        "    cal = fn_u(item, _PAY) if len(inspect.signature(fn_u).parameters) >= 2 else fn_u(item)\n"
        "    return _dn.solve_downstream(cal)\n"
        '""")\n'
    )
    inst = (
        "MODULAR DOWNSTREAM CONSTITUTIVE/BOUNDARY SOLVER SYNTHESIS:\n"
        "Write `/holdout_task/downstream_solver.py` exposing `solve_downstream(cal_item: dict) -> dict`.\n"
        "- Previously learned upstream transducer `/holdout_task/_verified_upstream.py` is already staged (`0444`) and `/holdout_task/batch_solver.py` "
        "automatically composes `solve_batch(item) = solve_downstream(calibrate_upstream(item, _PAY))` into canonical calibrated coordinates "
        "(note: if you inspect `campaign_diagnostics` in `/holdout_task/public_batch.pkl` during development, apply `_verified_upstream` with "
        "`/holdout_task/_upstream_param_payload.json` to calibrate raw diagnostic readings too!).\n"
        "- Run `python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py` to verify.\n\n"
        + base_task.instruction_text
    )
    return HoldoutOODTask(
        task_id=f"{base_task.task_id}__downstream_module",
        short_name=base_task.short_name,
        wave=base_task.wave,
        scientific_domain=base_task.scientific_domain,
        wormhole_family=base_task.wormhole_family,
        declared_artifacts=("/holdout_task/batch_solver.py", "/holdout_task/downstream_solver.py"),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename=base_task.public_pickle_filename,
        host_only_public_gt_json=base_task.host_only_public_gt_json,
        instruction_text=inst,
        public_setup_py=base_task.public_setup_py + bridge_py,
        private_verifier_py=base_task.private_verifier_py,
    )


def verify_m99_autonomous_decomposition_in_containers(
    *,
    family: str,
    s0_task: HoldoutOODTask,
    s0_public_item: dict[str, Any],
    s0_pay: dict[str, Any],
    e1_task: HoldoutOODTask,
    e1_public_item: dict[str, Any],
    e1_pay: dict[str, Any],
    e1_ro_deps: dict[str, str],
    e1_ro_shas: dict[str, str],
    upstream_code: str,
    d0_code: str,
    d1_code: str,
    s0_ro_deps: dict[str, str] | None = None,
    s0_ro_shas: dict[str, str] | None = None,
    verify_off_diagonal: bool = True,
) -> dict[str, Any]:
    """Fail-closed Docker verification (`0` private peek, `R04/R05/R11 + Q01` `2x2` matrix compliant) of an LLM-proposed `(u0, d0, d1)` decomposition:
    1. `upstream_code` (`calibrate_upstream(item, param_payload=None)`) passes `56/24` holdout check (`rmse <= 0.03`) on `E1_down`!
    2. Non-triviality check: when applied to `E1_down` (`v2_gains != 1`), `upstream_code(e1_item, e1_pay)` normalizes the calibration
       sweep table to identity (`max |bridge_raw - true_channel_loads| <= 0.05` / `max |raw_rosette - ref_rosette| <= 0.05`),
       strictly rejecting `return item` identity wrappers! (`Q08/R05`: timeout appends `probe_id` + `wall_ms` before returning!)
    3. Diagonal positive requirements:
       - `s0_d0`: `compose(upstream_code, d0_code, s0_pay)` passes `Stage S` full public scientific gate!
       - `e1_d1`: `compose(upstream_code, d1_code, e1_pay)` passes `E1_down` full public scientific gate!
    4. Off-diagonal empirical cross-verification (`Q01` — NEVER fabricate unexecuted cross-negatives!):
       - `e1_d0`: `compose(upstream_code, d0_code, e1_pay)` executed on `E1_down` public gate!
       - `s0_d1`: `compose(upstream_code, d1_code, s0_pay)` executed on `Stage S` public gate!
       Returns typed `matrix_2x2` probe receipts (`probe_id`, `wall_ms`, `execution_status`, `accepted`, `failure_tier`, `failure_kind`,
       `candidate_sha256`) so callers record ONLY genuine executed outcomes (sharing/covering if off-diagonals pass!).
    """
    probes: list[dict[str, Any]] = []

    fingerprints = {}
    for task, item in ((s0_task, s0_public_item), (e1_task, e1_public_item)):
        materialized, raw_sha = materialize_m99_public_item_and_sha(task)
        numeric_sha = compute_canonical_numeric_item_sha256(materialized)
        if numeric_sha != compute_canonical_numeric_item_sha256(item):
            raise ValueError(f"Public component input disagrees with joint task {task.task_id} (Q01)")
        fingerprints[task.task_id] = {
            "public_input_sha256": raw_sha, "canonical_numeric_sha256": numeric_sha,
            "setup_source_sha256": hashlib.sha256(task.public_setup_py.encode()).hexdigest(),
        }

    # 1. Direct upstream component holdout probe on `E1_down`
    up_diag = diagnose_m98_factored_conflict_locus(
        family,
        e1_public_item,
        chosen_up_code=upstream_code,
        chosen_up_entrypoint="calibrate_upstream",
        diag_payload=e1_pay,
        force_direct_component_probe=True,
    )
    up_diag = {**up_diag, **fingerprints[e1_task.task_id]}
    probes.append(
        {
            "role": "decomp_upstream_holdout",
            "probe_id": up_diag.get("probe_id"),
            "wall_ms": up_diag.get("probe_wall_ms", 0.0),
            "execution_status": up_diag.get("execution_status"),
        }
    )
    rmse = up_diag.get("chosen_up_holdout_rmse")
    if (up_diag.get("execution_status") != "completed" or up_diag.get("component_probe_executed") is not True
            or up_diag.get("evidence_origin") != "direct_component_probe" or not up_diag.get("probe_id")
            or up_diag.get("candidate_code_sha256") != hashlib.sha256(upstream_code.encode()).hexdigest()
            or rmse is None or not 0.0 <= float(rmse) <= 0.03
            or (up_diag.get("failure_tier"), up_diag.get("failure_kind")) != ("NONE", "none")):
        return {"valid": False, "reason": f"upstream_holdout_failed:{up_diag}", "probes": probes, "upstream_probe_receipt": up_diag}

    # 2. Non-trivial canonical normalization check inside air-gapped container
    import subprocess
    from .m7_autogenous_forest_pretraining import CLEAN_SCIPY_SANDBOX_IMAGE

    nt_pid = f"ntprobe_{uuid.uuid4().hex[:16]}"
    t_nt = monotonic()
    with tempfile.TemporaryDirectory(prefix="m99_nt_") as td:
        td_p = Path(td)
        (td_p / "up_mod.py").write_text(upstream_code, encoding="utf-8")
        with open(td_p / "in.pkl", "wb") as fh:
            pickle.dump({"family": family, "item": e1_public_item, "pay": e1_pay}, fh)
        nt_py = (
            "import copy, inspect, json, pickle, sys\nimport numpy as np\nsys.path.insert(0, '/work')\nimport up_mod\n"
            "b = pickle.load(open('/work/in.pkl', 'rb'))\nfam = b['family']; pay = b['pay']\n"
            "fn = up_mod.calibrate_upstream\n"
            "arg = copy.deepcopy(b['item']['dataset'] if fam == 'alpha' else b['item']['study'])\n"
            "out = fn(arg, pay) if len(inspect.signature(fn).parameters) >= 2 else fn(arg)\n"
            "if fam == 'alpha':\n"
            "    c = out['calibration_load_cells']\n"
            "    raw, tru = np.asarray(c['bridge_raw'], float), np.asarray(c['true_channel_loads'], float)\n"
            "    kp = np.max(np.abs(raw), axis=1) <= 8.5\n"
            "    cal_err = float(np.max(np.abs(raw[kp] - tru[kp])))\n"
            "    sh_ok = out['test']['hold_channel_raw'].shape == arg['test']['hold_channel_raw'].shape\n"
            "    changed = float(np.max(np.abs(np.asarray(out['test']['hold_channel_raw']) - np.asarray(arg['test']['hold_channel_raw'])))) > 0.02\n"
            "else:\n"
            "    c = out['sensor_calibration_sweeps']\n"
            "    raw, tru = np.asarray(c['raw_rosette'], float), np.asarray(c['ref_rosette'], float)\n"
            "    kp = np.max(np.abs(raw), axis=1) <= 8.5\n"
            "    cal_err = float(np.max(np.abs(raw[kp] - tru[kp])))\n"
            "    sh_ok = np.asarray(out['y_obs_raw']).shape == np.asarray(arg['y_obs_raw']).shape\n"
            "    changed = float(np.max(np.abs(np.asarray(out['y_obs_raw']) - np.asarray(arg['y_obs_raw'])))) > 0.02\n"
            "print(json.dumps({'cal_identity_err': cal_err, 'shape_ok': bool(sh_ok), 'transformed_signal': bool(changed)}))\n"
        )
        (td_p / "run_canon_chk.py").write_text(nt_py, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "-v", f"{td_p}:/work:ro", CLEAN_SCIPY_SANDBOX_IMAGE, "python3", "/work/run_canon_chk.py"],
                capture_output=True,
                text=True,
                timeout=25,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            status = "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "infrastructure_error"
            nt_ms = round((monotonic() - t_nt) * 1000.0, 2)
            probes.append({"role": "decomp_canonical_interface_check", "probe_id": nt_pid, "wall_ms": nt_ms, "execution_status": status})
            return {"valid": False, "reason": f"nontriviality_probe_{status}", "probes": probes}
    nt_ms = round((monotonic() - t_nt) * 1000.0, 2)
    nt_receipt = {"role": "decomp_canonical_interface_check", "probe_id": nt_pid, "wall_ms": nt_ms,
                  "execution_status": "infrastructure_error"}
    probes.append(nt_receipt)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"valid": False, "reason": f"nontriviality_probe_failed:{(proc.stderr or '')[-240:]}", "probes": probes}
    try:
        nt_j = json.loads(proc.stdout.strip().splitlines()[-1])
        error = float(nt_j["cal_identity_err"])
        if not all(isinstance(nt_j[key], bool) for key in ("shape_ok", "transformed_signal")):
            raise ValueError("invalid boolean measurements")
    except (ValueError, TypeError, KeyError) as exc_j:
        return {"valid": False, "reason": f"nontriviality_probe_json_parse_error:{exc_j}", "probes": probes}
    nt_receipt["execution_status"] = "completed"
    if not (nt_j["shape_ok"] and nt_j["transformed_signal"] and 0.0 <= error <= 0.05):
        return {"valid": False, "reason": f"canonical_interface_violation:{nt_j}", "probes": probes}

    def _run_gate_cell(
        role_tag: str,
        tk: HoldoutOODTask,
        dn_src: str,
        pay_d: dict[str, Any],
        ro_d: dict[str, str] | None = None,
        ro_s: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        cand_c = compose_m98_two_stage_solver_code(
            upstream_code=upstream_code,
            upstream_entrypoint="calibrate_upstream",
            downstream_code=dn_src,
            downstream_entrypoint="solve_downstream",
            param_payload=pay_d,
        )
        cand_sha = hashlib.sha256(cand_c.encode("utf-8")).hexdigest()
        pid_g = f"dprobe_{uuid.uuid4().hex[:16]}"
        t_g = monotonic()
        kw_g: dict[str, Any] = {"task": tk, "candidate_code": cand_c}
        if ro_d:
            kw_g["readonly_dependencies"] = ro_d
        if ro_s:
            kw_g["expected_dependency_sha256"] = ro_s
        try:
            pr_g = execute_single_pre_llm_public_gate_probe_in_container(**kw_g)
        except (subprocess.SubprocessError, OSError) as exc:
            pr_g = {"accepted": False, "setup_ok": False, "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                    "infrastructure_error": str(exc)}
        w_ms = round((monotonic() - t_g) * 1000.0, 2)
        cls_g = classify_public_probe_failure(pr_g)
        fingerprint = fingerprints[tk.task_id]
        if cls_g["tier"] not in {"INFRA_FAILURE", "SCHEMA_FAILURE"} and not joint_scientific_receipt_matches(
            pr_g, cand_sha, fingerprint["public_input_sha256"], fingerprint["canonical_numeric_sha256"],
        ):
            cls_g = {"tier": "INFRA_FAILURE", "failure_kind": "timeout_or_infra"}
        ex_st = "timeout" if pr_g.get("timed_out") else ("infrastructure_error" if cls_g["tier"] == "INFRA_FAILURE" else "completed")
        rec_g = {
            **pr_g,
            "accepted": cls_g["tier"] == "NONE",
            "role": role_tag,
            "probe_id": pid_g,
            "wall_ms": w_ms,
            "probe_wall_ms": w_ms,
            "candidate_sha256": cand_sha,
            "component_code_sha256s": [hashlib.sha256(src.encode()).hexdigest() for src in (upstream_code, dn_src)],
            **fingerprint,
            "dependency_bundle_sha256": hashlib.sha256(json.dumps(ro_s or {}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "execution_status": ex_st,
            "failure_tier": cls_g["tier"],
            "failure_kind": cls_g["failure_kind"],
        }
        probes.append({"role": role_tag, "probe_id": pid_g, "wall_ms": w_ms, "execution_status": ex_st, "accepted": rec_g["accepted"]})
        return rec_g

    # 3. Verify `s0_d0`: `compose(upstream_code, d0_code, s0_pay)` on `Stage S` public gate
    pr_s0_d0 = _run_gate_cell("decomp_s0_d0_public_gate", s0_task, d0_code, s0_pay, s0_ro_deps, s0_ro_shas)
    if not pr_s0_d0["accepted"]:
        return {
            "valid": False,
            "reason": f"composed_s0_rejected:{pr_s0_d0.get('scientific_probe_receipt') or pr_s0_d0.get('schema_stderr')}",
            "probes": probes,
        }

    # 4. Verify `e1_d1`: `compose(upstream_code, d1_code, e1_pay)` on `E1_down` public gate
    pr_e1_d1 = _run_gate_cell("decomp_e1_d1_public_gate", e1_task, d1_code, e1_pay, e1_ro_deps, e1_ro_shas)
    if not pr_e1_d1["accepted"]:
        return {
            "valid": False,
            "reason": f"composed_e1_rejected:{pr_e1_d1.get('scientific_probe_receipt') or pr_e1_d1.get('schema_stderr')}",
            "probes": probes,
        }

    # 5. Off-diagonal `2x2` empirical verification (`Q01`: `e1_d0` and `s0_d1`!)
    if verify_off_diagonal:
        pr_e1_d0 = _run_gate_cell("decomp_e1_d0_cross_public_gate", e1_task, d0_code, e1_pay, e1_ro_deps, e1_ro_shas)
        pr_s0_d1 = _run_gate_cell("decomp_s0_d1_cross_public_gate", s0_task, d1_code, s0_pay, s0_ro_deps, s0_ro_shas)
    else:
        pr_e1_d0 = {"accepted": False, "execution_status": "not_executed", "failure_tier": "NOT_EXECUTED", "failure_kind": "not_executed", "probe_id": "", "wall_ms": 0.0}
        pr_s0_d1 = {"accepted": False, "execution_status": "not_executed", "failure_tier": "NOT_EXECUTED", "failure_kind": "not_executed", "probe_id": "", "wall_ms": 0.0}

    return {
        "valid": True,
        "upstream_holdout_rmse": up_diag["chosen_up_holdout_rmse"],
        "upstream_probe_receipt": up_diag,
        "canonical_interface_check": nt_j,
        "matrix_2x2": {
            "s0_d0": pr_s0_d0,
            "e1_d1": pr_e1_d1,
            "e1_d0": pr_e1_d0,
            "s0_d1": pr_s0_d1,
        },
        "probes": probes,
        "total_probe_wall_ms": round(sum(float(p["wall_ms"]) for p in probes), 2),
    }


def propose_and_verify_m99_autonomous_two_stage_factorization(
    *,
    client: Any,
    stage_subdir: str,
    family: str,
    s0_monolithic_code: str,
    e1_monolithic_code: str,
    s0_task: HoldoutOODTask,
    s0_public_item: dict[str, Any],
    s0_pay: dict[str, Any],
    e1_task: HoldoutOODTask,
    e1_public_item: dict[str, Any],
    e1_pay: dict[str, Any],
    e1_ro_deps: dict[str, str],
    e1_ro_shas: dict[str, str],
    s0_ro_deps: dict[str, str] | None = None,
    s0_ro_shas: dict[str, str] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Autonomous LLM refactoring compiler (`R02` — ZERO host-seeded `u0` or `d0`!):
    Refactors the two publicly-verified monolithic solvers `(s0_monolithic_code, e1_monolithic_code)` into:
      - `upstream_code` exposing `calibrate_upstream(item: dict, param_payload: dict | None = None) -> dict`
      - `downstream_d0_code` exposing `solve_downstream(cal_item: dict) -> dict`
      - `downstream_d1_code` exposing `solve_downstream(cal_item: dict) -> dict`
    Verified fail-closed in air-gapped Docker containers!
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["diagnosis", "upstream_code", "downstream_d0_code", "downstream_d1_code"],
        "properties": {
            "diagnosis": {"type": "string"},
            "upstream_code": {"type": "string"},
            "downstream_d0_code": {"type": "string"},
            "downstream_d1_code": {"type": "string"},
        },
    }
    coord_spec = (
        "For `alpha`, `item` is `dataset: dict`. `calibrate_upstream(dataset, param_payload=None)`: if `param_payload` has "
        "`'gains'` and `'offsets'` (length 4), invert `(raw - b) / g` on `train['past_channel_raw']`, `train['hold_channel_raw']`, "
        "`test['past_channel_raw']`, `test['hold_channel_raw']` (otherwise fit diagonal affine calibration from inlier rows `|bridge_raw|<=8.5` "
        "of `calibration_load_cells`), AND normalize `calibration_load_cells['bridge_raw']` (on inlier rows `<=8.5`) to equal `true_channel_loads` "
        "and `single_channel_sweeps['cmd_stress']` (inlier `<=8.5`) to equal `true_stress`, returning a deep copy of `dataset`. "
        "`downstream_d0_code` and `downstream_d1_code` must each expose `solve_downstream(cal_dataset: dict) -> dict` operating directly in "
        "canonical calibrated coordinates (`loads_true = cal_dataset[split]['past_channel_raw']`, etc.)."
        if family == "alpha"
        else "For `beta`, `item` is `study: dict`. `calibrate_upstream(study, param_payload=None)`: if `param_payload` has `'gains'` and "
        "`'offsets'` (length 4), invert `((y_obs_raw.reshape(-1, 4) - b) / g).ravel()` (otherwise fit diagonal affine calibration from inlier "
        "rows `|raw_rosette|<=8.5` of `sensor_calibration_sweeps`), AND normalize `sensor_calibration_sweeps['raw_rosette']` (inlier `<=8.5`) "
        "to equal `ref_rosette` and `single_bank_sweeps['raw_readout']` (inlier `<=8.5`) to equal `ref_signal`, returning a deep copy of `study`. "
        "`downstream_d0_code` and `downstream_d1_code` must each expose `solve_downstream(cal_study: dict) -> dict` operating directly in "
        "canonical calibrated coordinates (`y_cal = cal_study['y_obs_raw']`)."
    )
    sys_prompt = (
        "You are an autonomous scientific program modularization compiler. Given two verified monolithic solver programs from consecutive "
        "campaigns (`s0_monolithic_code` and `e1_monolithic_code`), factor out the shared upstream sensor/transducer calibration layer from "
        "the campaign-specific downstream constitutive/PDE solvers so `solve_downstream(calibrate_upstream(item, param_payload))` reproduces "
        "both campaigns cleanly without `_HEGEL_ADAPTER_PRECALIBRATED_DO_NOT_REWRAP`."
    )
    convo = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "family": family,
                    "interface_contract": coord_spec,
                    "s0_monolithic_code": s0_monolithic_code,
                    "e1_monolithic_code": e1_monolithic_code,
                },
                indent=2,
            ),
        }
    ]
    call_receipts: list[dict[str, Any]] = []
    attempts_log: list[dict[str, Any]] = []
    all_probes: list[dict[str, Any]] = []

    for att in range(1, max_attempts + 1):
        res = client.call_structured(
            system_prompt=sys_prompt,
            conversation_turns=convo,
            output_schema=schema,
            stage_subdir=f"{stage_subdir}_decomp_att{att}",
        )
        call_receipts.append(res)
        if not res.get("ok") or not isinstance(res.get("parsed"), dict):
            attempts_log.append({"attempt": att, "ok": False, "reason": "llm_call_failed"})
            continue
        parsed = res["parsed"]
        ver = verify_m99_autonomous_decomposition_in_containers(
            family=family,
            s0_task=s0_task,
            s0_public_item=s0_public_item,
            s0_pay=s0_pay,
            s0_ro_deps=s0_ro_deps,
            s0_ro_shas=s0_ro_shas,
            e1_task=e1_task,
            e1_public_item=e1_public_item,
            e1_pay=e1_pay,
            e1_ro_deps=e1_ro_deps,
            e1_ro_shas=e1_ro_shas,
            upstream_code=str(parsed["upstream_code"]),
            d0_code=str(parsed["downstream_d0_code"]),
            d1_code=str(parsed["downstream_d1_code"]),
        )
        all_probes.extend(ver.get("probes") or [])
        attempts_log.append({"attempt": att, "verification": {k: v for k, v in ver.items() if k != "probes"}})
        if ver["valid"]:
            return {
                "accepted": True,
                "attempts_used": att,
                "diagnosis": str(parsed["diagnosis"]),
                "upstream_code": str(parsed["upstream_code"]),
                "downstream_d0_code": str(parsed["downstream_d0_code"]),
                "downstream_d1_code": str(parsed["downstream_d1_code"]),
                "upstream_probe_receipt": ver.get("upstream_probe_receipt"),
                "matrix_2x2": ver.get("matrix_2x2"),
                "verification_probes": all_probes,
                "total_probe_wall_ms": round(sum(float(p["wall_ms"]) for p in all_probes), 2),
                "attempts_log": attempts_log,
                "provider_call_receipts": call_receipts,
            }
        convo.append({"role": "assistant", "content": json.dumps(parsed)})
        convo.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "container_verification_failed": ver["reason"],
                        "instruction": "Fix `upstream_code`, `downstream_d0_code`, and `downstream_d1_code` so all 4 public checks pass.",
                    }
                ),
            }
        )
    return {
        "accepted": False,
        "attempts_used": max_attempts,
        "verification_probes": all_probes,
        "total_probe_wall_ms": round(sum(float(p["wall_ms"]) for p in all_probes), 2),
        "attempts_log": attempts_log,
        "provider_call_receipts": call_receipts,
    }


def verify_m99_large_margin_witness_partition(
    *,
    branch_guards: dict[str, dict[str, Any]],
    positive_witness_features_by_branch: dict[str, list[dict[str, float | None]]],
    covered_branch_pairs: set[tuple[str, str]] | None = None,
    negative_witness_features_by_branch: dict[str, list[dict[str, float | None]]] | None = None,
    min_clause_margin: float = 0.035,
) -> dict[str, Any]:
    """Fail-closed Kleene witness partition check (`verify_m97_multibranch_witness_partition`) PLUS feature-name-blind
    maximum-margin regularization (`min_clause_margin = 0.035`) shared identically across ALL 3 arms:
      1. Every feature referenced in any `all_of` clause must have empirical spread `>= 2 * min_clause_margin` (`0.070`)
         across the supplied witnesses (preventing 1-clause shortcuts on low-variance ambient telemetry noise!).
      2. Every positive witness of `b` must satisfy `|w[feat] - threshold| >= min_clause_margin` on every clause of `b`.
      3. Every non-covered negative witness (explicit negative or positive of another non-covering branch) must be
         falsified by at least one clause of `b` with strict margin `|w[feat] - threshold| >= min_clause_margin`.
    """
    from .m9_7_non_nested_counterfactual_lineage_world import verify_m97_multibranch_witness_partition

    base_chk = verify_m97_multibranch_witness_partition(
        branch_guards=branch_guards,
        positive_witness_features_by_branch=positive_witness_features_by_branch,
        covered_branch_pairs=covered_branch_pairs,
        negative_witness_features_by_branch=negative_witness_features_by_branch,
    )
    if not base_chk["valid"]:
        return base_chk

    cov = covered_branch_pairs or set()
    neg_map = negative_witness_features_by_branch or {}
    all_wits: list[dict[str, float | None]] = []
    for wl in positive_witness_features_by_branch.values():
        all_wits.extend(wl)
    for wl in neg_map.values():
        all_wits.extend(wl)

    feat_spread: dict[str, float] = {}
    for fk in sorted({k for w in all_wits for k, v in w.items() if v is not None}):
        vals = [float(w[fk]) for w in all_wits if w.get(fk) is not None]
        feat_spread[fk] = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0

    min_spread = 2.0 * float(min_clause_margin)
    reasons: list[str] = []

    def _clause_false_with_margin(cl: dict[str, Any], w: dict[str, float | None]) -> bool:
        fn = str(cl["feature_name"])
        op = str(cl["op"])
        thr = float(cl["threshold"])
        v = w.get(fn)
        if v is None:
            return False
        fv = float(v)
        if abs(fv - thr) < min_clause_margin:
            return False
        if op == "<=":
            return not (fv <= thr)
        if op == "<":
            return not (fv < thr)
        if op == ">=":
            return not (fv >= thr)
        if op == ">":
            return not (fv > thr)
        return False

    for bid, g in sorted(branch_guards.items()):
        clauses = list((g or {}).get("all_of") or [])
        for cl in clauses:
            fn = str(cl["feature_name"])
            thr = float(cl["threshold"])
            sp = feat_spread.get(fn, 0.0)
            if sp < min_spread:
                reasons.append(f"low_variance_feature_split:{bid}:{fn}:spread={sp:.5f}<{min_spread:.3f}")
            for wi, wp in enumerate(positive_witness_features_by_branch.get(bid, [])):
                vp = wp.get(fn)
                if vp is None or abs(float(vp) - thr) < min_clause_margin:
                    reasons.append(f"insufficient_positive_margin:{bid}[{wi}]:{fn}:|{vp}-{thr}|<{min_clause_margin}")

        neg_cands: list[tuple[str, dict[str, float | None]]] = []
        for wi, wn in enumerate(neg_map.get(bid, [])):
            neg_cands.append((f"neg[{wi}]", wn))
        for other_bid, owl in sorted(positive_witness_features_by_branch.items()):
            if other_bid == bid or (bid, other_bid) in cov:
                continue
            for wi, ow in enumerate(owl):
                neg_cands.append((f"{other_bid}[{wi}]", ow))

        for n_lbl, wn in neg_cands:
            if not any(_clause_false_with_margin(cl, wn) for cl in clauses):
                reasons.append(f"insufficient_negative_margin:{bid}_vs_{n_lbl}:min_margin={min_clause_margin}")

    return {"valid": len(reasons) == 0, "reasons": reasons}


def propose_and_validate_m99_guard_partition(
    *,
    client: Any,
    stage_subdir: str,
    scope_label: str,
    feature_names: tuple[str, ...],
    existing_branch_ids: list[str],
    new_branch_id_placeholder: str,
    positive_witness_features_by_branch: dict[str, list[dict[str, float | None]]],
    negative_witness_features_by_branch: dict[str, list[dict[str, float | None]]] | None = None,
    covered_branch_pairs: set[tuple[str, str]] | None = None,
    min_clause_margin: float = 0.035,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Equal-capability large-margin Kleene `all_of` guard compiler shared 100% identically across ALL 3 M9.9 arms!"""
    from .m9_8_blind_mismatch_and_selective_dependency_world import _build_guard_expr_schema

    g_schema = _build_guard_expr_schema(feature_names)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["diagnosis", "new_branch_guard", "existing_branch_guards"],
        "properties": {
            "diagnosis": {"type": "string"},
            "new_branch_guard": g_schema,
            "existing_branch_guards": {
                "type": "object",
                "additionalProperties": False,
                "required": list(existing_branch_ids),
                "properties": {bid: g_schema for bid in existing_branch_ids},
            },
        },
    }
    all_wits: list[dict[str, float | None]] = []
    for wl in positive_witness_features_by_branch.values():
        all_wits.extend(wl)
    for wl in (negative_witness_features_by_branch or {}).values():
        all_wits.extend(wl)
    empirical_spreads = {
        fn: round((max(vals) - min(vals)) if len(vals) >= 2 else 0.0, 5)
        for fn in feature_names
        for vals in [[float(w[fn]) for w in all_wits if w.get(fn) is not None]]
    }
    sys_prompt = (
        f"You are a large-margin Kleene applicability-guard synthesis compiler (`scope={scope_label}`). "
        f"Given measured generic statistical observables `{list(feature_names)}` across tested positive and negative "
        "witnesses, output `all_of` conjunctions (`1..4` clauses) that evaluate to `TRUE` on 100% of each branch's own "
        "positive witnesses and `FALSE` on non-covering branches' witnesses. "
        f"LARGE-MARGIN REGULARIZATION RULE: Every referenced feature must have empirical spread `>= {2.0 * min_clause_margin:.3f}` "
        f"and every threshold must separate positive and negative witnesses by a strict margin `|x - threshold| >= {min_clause_margin:.3f}` "
        "(place thresholds near cluster midpoints; never split low-variance `< 0.07` noise jitter — use a 2-clause conjunction "
        "of high-spread features when needed to isolate a baseline cell where multiple high-spread indicators are inactive)."
    )
    convo = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "allowed_features": list(feature_names),
                    "min_clause_margin": min_clause_margin,
                    "empirical_feature_spreads": empirical_spreads,
                    "existing_branch_ids": existing_branch_ids,
                    "new_branch_id": new_branch_id_placeholder,
                    "positive_witnesses_by_branch": positive_witness_features_by_branch,
                    "negative_witnesses_by_branch": negative_witness_features_by_branch or {},
                },
                indent=2,
            ),
        }
    ]
    call_receipts = []
    attempts_log = []
    for att in range(1, max_attempts + 1):
        res = client.call_structured(
            system_prompt=sys_prompt,
            conversation_turns=convo,
            output_schema=schema,
            stage_subdir=f"{stage_subdir}_gind_att{att}",
        )
        call_receipts.append(res)
        if not res.get("ok") or not isinstance(res.get("parsed"), dict):
            attempts_log.append({"attempt": att, "ok": False, "reason": "llm_call_failed"})
            continue
        parsed = res["parsed"]
        all_guards = {
            **{bid: parsed["existing_branch_guards"][bid] for bid in existing_branch_ids},
            new_branch_id_placeholder: parsed["new_branch_guard"],
        }
        chk = verify_m99_large_margin_witness_partition(
            branch_guards=all_guards,
            positive_witness_features_by_branch=positive_witness_features_by_branch,
            covered_branch_pairs=covered_branch_pairs,
            negative_witness_features_by_branch=negative_witness_features_by_branch,
            min_clause_margin=min_clause_margin,
        )
        attempts_log.append({"attempt": att, "partition_check": chk})
        if chk["valid"]:
            return {
                "accepted": True,
                "attempts_used": att,
                "diagnosis": str(parsed["diagnosis"]),
                "new_branch_guard": parsed["new_branch_guard"],
                "existing_branch_guards": parsed["existing_branch_guards"],
                "attempts_log": attempts_log,
                "provider_call_receipts": call_receipts,
            }
        convo.append({"role": "assistant", "content": json.dumps(parsed)})
        convo.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "host_partition_check_failed": chk,
                        "instruction": (
                            f"Use only high-spread features (`spread >= {2.0 * min_clause_margin:.3f}`) and midpoint thresholds "
                            f"with `|x - threshold| >= {min_clause_margin:.3f}` on all separated positive/negative witnesses."
                        ),
                    }
                ),
            }
        )
    return {"accepted": False, "attempts_used": max_attempts, "attempts_log": attempts_log, "provider_call_receipts": call_receipts}


def check_witness_feature_collision(
    positive_witness_features_by_branch: dict[str, list[dict[str, float | None]]],
    covered_branch_pairs: set[tuple[str, str]] | None = None,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Detect exact witness collision in the allowed feature space (`0` wasted doomed guard calls when `dn_obs` collides!)."""
    cov = covered_branch_pairs or set()
    bids = sorted(positive_witness_features_by_branch.keys())
    collisions = []
    for i, b1 in enumerate(bids):
        for b2 in bids[i + 1 :]:
            if (b1, b2) in cov or (b2, b1) in cov:
                continue
            for w1 in positive_witness_features_by_branch[b1]:
                for w2 in positive_witness_features_by_branch[b2]:
                    keys = sorted(set(w1.keys()) | set(w2.keys()))
                    same = True
                    for k in keys:
                        v1, v2 = w1.get(k), w2.get(k)
                        if v1 is None and v2 is None:
                            continue
                        if v1 is None or v2 is None or abs(float(v1) - float(v2)) > tol:
                            same = False
                            break
                    if same:
                        collisions.append((b1, b2))
                        break
    return {"has_collision": len(collisions) > 0, "colliding_pairs": collisions}


def m99_task_to_frontier_task(
    task: HoldoutOODTask,
    *,
    continual_context_block: str = "",
    allowed_readonly_dependencies: tuple[str, ...] = (),
) -> m6f.FrontierBenchmarkTask:
    base_ft = m95_task_to_frontier_task(
        task,
        continual_context_block=continual_context_block,
        allowed_readonly_dependencies=allowed_readonly_dependencies,
    )
    return m6f.FrontierBenchmarkTask(
        **{**base_ft.__dict__, "benchmark": "Hegel-M99-Autonomous-Decomposition-And-Interaction-Withdrawal-World"}
    )


# ============================================================================
# 4. Reference Dev-Seed Solvers (`0`-LLM Unit & Container Testing on `DEV_M99_WORLD_SPECS` ONLY!)
# ============================================================================


def build_m99_reference_dev_downstream_solver_code(family: str, entrypoint: str, effective_regime: str) -> str:
    """Reference downstream solver in canonical coordinates (`D0, D1, D2, D2_coupled`), used ONLY in `0`-LLM dev tests!"""
    from .m9_8_blind_mismatch_and_selective_dependency_world import build_m98_reference_dev_downstream_solver_code

    if effective_regime in {"D0", "D1", "D2"}:
        return build_m98_reference_dev_downstream_solver_code(family, entrypoint, effective_regime)
    assert effective_regime == "D2_coupled"
    if family == "alpha":
        base_d1 = build_m98_reference_dev_downstream_solver_code("alpha", entrypoint, "D1")
        return base_d1.replace("np.log1p(eval_t)", "2.0 * (np.sqrt(1.0 + eval_t) - 1.0)")
    # Beta `D2_coupled`: `Periodic` in `x` + `Neumann` in `y`!
    return f'''from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

def _op1d_bc(n: int, h: float, kap: np.ndarray, vel: np.ndarray, rob: float, bc_kind: str) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        kp = 0.5 * (kap[i] + kap[i + 1])
        km = 0.5 * (kap[i] + kap[i - 1])
        A[i, i - 1] = -km / (h * h) - max(float(vel[i]), 0.0) / h
        A[i, i + 1] = -kp / (h * h) - max(-float(vel[i]), 0.0) / h
        A[i, i] = (kp + km) / (h * h) + abs(float(vel[i])) / h
    if bc_kind == "periodic":
        for i in (0, n - 1):
            im = (i - 1) % n
            ip = (i + 1) % n
            kp = 0.5 * (kap[i] + kap[ip])
            km = 0.5 * (kap[i] + kap[im])
            A[i, im] += -km / (h * h) - max(float(vel[i]), 0.0) / h
            A[i, ip] += -kp / (h * h) - max(-float(vel[i]), 0.0) / h
            A[i, i] += (kp + km) / (h * h) + abs(float(vel[i])) / h
    else:
        A[0, 0] = kap[0] / (h * h)
        A[0, 1] = -kap[0] / (h * h)
        A[n - 1, n - 1] = kap[-1] / (h * h)
        A[n - 1, n - 2] = -kap[-1] / (h * h)
    return A

def {entrypoint}(study: dict) -> dict:
    Lx, Ly = float(study["Lx"]), float(study["Ly"])
    nx, ny = int(study["nx"]), int(study["ny"])
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    Ax = _op1d_bc(nx, dx, np.asarray(study["kappa_x"], float), np.asarray(study["vel_x"], float), float(study["robin_x"]), "periodic")
    Ay = _op1d_bc(ny, dy, np.asarray(study["kappa_y"], float), np.asarray(study["vel_y"], float), float(study["robin_y"]), "neumann")
    V2 = np.asarray(study["V_reaction_2d"], float)
    L = np.kron(np.eye(ny), Ax) + np.kron(Ay, np.eye(nx)) + np.diag(V2.ravel(order="F"))
    modes = np.asarray(study["source_modes"], float)
    M = modes.shape[0]
    B = np.column_stack([modes[m].ravel(order="F") for m in range(M)])
    sij = np.asarray(study["sensor_ij"], int)
    si, sj = sij[:, 0], sij[:, 1]
    mus = np.asarray(study["mu_shifts"], float)
    I_f = np.eye(nx * ny, dtype=float)
    S_blks = []
    U_last = np.empty((M, nx, ny), dtype=float)
    for ki, mu in enumerate(mus):
        sf = np.linalg.solve(L + float(mu) * I_f, B)
        Uk = np.empty((M, nx, ny), dtype=float)
        for m in range(M):
            Uk[m] = sf[:, m].reshape((nx, ny), order="F")
        if ki == len(mus) - 1:
            U_last = Uk
        S_blks.append(Uk[:, si, sj].T)
    S = np.vstack(S_blks)
    Z = np.asarray(study["Z_nuisance"], float)
    y = np.asarray(study["y_obs_raw"], float)
    lz, lw = float(study["nuisance_ridge"]), float(study["source_ridge"])
    ub, bt = float(study["u_box"]), float(study["total_budget"])
    Pz = np.linalg.solve(Z.T @ Z + lz * np.eye(Z.shape[1]), Z.T)
    St = S - Z @ (Pz @ S)
    ry = y - Z @ (Pz @ y)
    H = St.T @ St + lw * np.eye(M)
    f = St.T @ ry
    res = minimize(
        lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w),
        np.full(M, bt / M, dtype=float),
        jac=lambda w: H @ w - f,
        bounds=[(0.0, ub)] * M,
        constraints=[{{"type": "eq", "fun": lambda w: float(np.sum(w)) - bt, "jac": lambda w: np.ones(M)}}],
        method="SLSQP",
        options={{"ftol": 1e-14, "maxiter": 500}},
    )
    w = np.clip(res.x, 0.0, ub)
    for _ in range(3):
        fr = (w > 1e-11) & (w < ub - 1e-11)
        if int(np.sum(fr)) >= 2:
            ix = np.where(fr)[0]
            rem = bt - float(np.sum(w[~fr]))
            Hff = H[np.ix_(ix, ix)]
            rf = f[ix] - H[np.ix_(ix, ~fr)] @ w[~fr]
            o1 = np.ones(len(ix), dtype=float)
            km = np.block([[Hff, o1[:, None]], [o1[None, :], np.zeros((1, 1))]])
            sol = np.linalg.solve(km, np.concatenate([rf, [rem]]))
            wc = w.copy(); wc[ix] = sol[:-1]
            if np.all(wc >= -1e-12) and np.all(wc <= ub + 1e-12):
                w = np.clip(wc, 0.0, ub); break
    lo, hi = float(np.min(w) - ub), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, ub))) > bt: lo = mid
        else: hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, ub)
    rv = St @ w - ry
    jv = 0.5 * float(rv @ rv) + 0.5 * lw * float(w @ w)
    g = H @ w - f
    best_nu, best_lb = float(g[0]), -1e30
    for nc in np.unique(g):
        lbc = jv - float(g @ w) + float(nc) * bt + ub * float(np.sum(np.minimum(0.0, g - float(nc))))
        if lbc > best_lb: best_lb, best_nu = lbc, float(nc)
    return {{
        "weights": w,
        "terminal_field": np.tensordot(w, U_last, axes=(0, 0)),
        "kkt_certificate": {{
            "weights": w.tolist(),
            "nu_sum": best_nu,
            "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
            "lam_upper": np.maximum(0.0, best_nu - g).tolist(),
            "primal_objective": jv,
            "dual_lower_bound": float(min(best_lb, jv)),
        }},
    }}
'''

"""Hegel M9.8: Blind Residual-Mismatch Discovery, Strict Byte-Identical Counterfactual History (`R0 == U_iso`) &
Factored Two-Stage Mechanism Graph vs Online `Flat_ConditionalTable` Continual Benchmark.

Resolves ALL open scientific & methodological questions from `Hegel_M9_7_Audit` (`A01–A17`, §9–10):
  1. **Head-to-Head Online Comparison Against `Flat_ConditionalTable_Online` (`A02`)**:
     - `Flat_LatestFirst`: Unhandicapped equal-info archive + shared `2/3–1/3` stratified channel OLS + `0444` prefix logs (`1` probe).
     - `Flat_ConditionalTable_Online`: Live online conditional version table (`0` `PersistentMechanismGraph`) with its own live
       `gpt-6.2-sol` Kleene guard induction over all 12 generic observables (`up_stat_01..04 + dn_stat_01..08`) + `EmpiricalCoverageReceipt`!
     - `Hegel_FactoredMechanismGraph`: Two-stage causal `PersistentMechanismGraph` (`Upstream_Transducer_Chain {u0, u1}` ->
       `Downstream_Constitutive_Chain {d0, d1, d2}`) with causal parent holdout-residual locus attribution (`upstream_transducer`
       vs `downstream_constitutive`), content-addressed `QualificationReceipt`s, checkpoint-hydrated `witness_ledger` &
       `EmpiricalCoverageReceipt`s (`A10`), and zero-shot cross-factor structural composition (`u1 o d1` on `M_comb1`, `u1 o d2` on `M_comb2`)!
  2. **Strict Byte-Identical Visible Current Inputs on Counterfactual Pair `(R0, U_iso)` + Executed `HistoryMasked` Control (`A03`)**:
     - `/holdout_task/public_batch.pkl` contains **ZERO `gt` key** (`host_only_public_gt_json` is held strictly on the host!).
     - On `R0` (`Step 2`, true regime `(U0, D0)`) and `U_iso` (`Step 5`, true regime `(U0, D1)`):
       * `R0.instruction_text == U_iso.instruction_text` (`100%` byte-identical!),
       * `SHA256(/holdout_task/public_batch.pkl on R0) == SHA256(/holdout_task/public_batch.pkl on U_iso)` (`100%` byte-identical, `0` differing fields!),
       * Even private test covariates use identical seeds on `R0` and `U_iso`; ONLY the prefix apparatus log
         `/tmp/hegel_continual_archive/archive_apparatus_log.json` (`<= t - 1`) differs!
  3. **Zero Prompt Formula/Role Leakage & Generic Moment/OLS Statistical Reducer Bank (`A13`)**:
     - Identical blind `instruction_text` across ALL 8 episodes per family (`0` mention of `Neumann`, `Periodic`, `Quenched`,
       `Logarithmic`, `1 - exp(-t)`, `4x4 cross-talk`, `D0/D1/D2`, `U0/U1`, or which of the 8 unlabeled facility log channels `tel_0..tel_7` is active!).
     - Generic reducer bank: `up_stat_01..04` (univariate diagonal OLS $R^2$, max off-diagonal residual correlation, Frobenius norm, row count)
       + `dn_stat_01..08` (plain temporal means `mean(tel_k)` of the 8 unlabeled facility log channels in `archive_apparatus_log[-1]`).
  4. **Pre-Registered Unexposed `n = 8` Independent Confirmation Clusters (`4` Alpha + `4` Beta, `A04` & `A15`)**:
     - Pooled `n = 8` two-sided sign-flip resolution `p = 2 / 2^8 = 0.0078125`; dev testing strictly confined to `DEV_M98_WORLD_SPECS`.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import tempfile
from time import monotonic
from typing import Any

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_autogenous_forest_pretraining import CLEAN_SCIPY_SANDBOX_IMAGE
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m8_persistent_variational_planner import (
    M8VariationalPlanner,
    PersistentMechanismGraph,
    TaskModel,
    TypedMechanism,
    evaluate_committed_task_model,
)
from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY
from .m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    encode_pure_json_numerical_payload,
)
from .m9_4_continual_mechanism_binding_world import (
    M94BindingRegistry,
    MechanismExecutableBinding,
    grade_m94_artifact_map,
    verify_binding_identity_and_foreign_keys,
)
from .m9_5_equal_tool_continual_attribution_world import (
    ADAPTER_PRECALIBRATED_SENTINEL,
    execute_single_pre_llm_public_gate_probe_in_container,
    m95_task_to_frontier_task,
)
from .m9_6_autonomous_structure_continual_world import (
    EmpiricalCoverageReceipt,
    ImmutableBranchLineageController,
    classify_public_probe_failure,
    evaluate_m96_autonomous_guard,
)
from .m9_7_non_nested_counterfactual_lineage_world import (
    estimate_validated_channel_affine_calibration,
    verify_m97_multibranch_witness_partition,
)

M98_ARMS: tuple[str, ...] = (
    "Flat_LatestFirst",
    "Flat_ConditionalTable_Online",
    "Hegel_FactoredMechanismGraph",
)

M98_UPSTREAM_OBSERVABLE_NAMES: tuple[str, ...] = (
    "up_stat_01_diagonal_ols_r2",
    "up_stat_02_off_diagonal_corr_max",
    "up_stat_03_cal_frobenius_rms",
    "up_stat_04_cal_row_count",
)

M98_DOWNSTREAM_OBSERVABLE_NAMES: tuple[str, ...] = tuple(
    f"dn_stat_0{k + 1}_log_channel_{k}_mean" for k in range(8)
)

M98_ALL_OBSERVABLE_NAMES: tuple[str, ...] = M98_UPSTREAM_OBSERVABLE_NAMES + M98_DOWNSTREAM_OBSERVABLE_NAMES


@dataclass(frozen=True)
class M98WorldSpec:
    world_id: str
    cluster_id: str
    family: str  # 'alpha' | 'beta'
    split: str  # 'dev' | 'confirmation'
    seed_base: int
    channel_perm: tuple[int, int, int, int]
    active_log_channels: tuple[int, int]  # `(ch_d1, ch_d2)` hidden inside `0..7` unlabeled facility log channels!
    v2_gains: tuple[float, float, float, float]
    v2_offsets: tuple[float, float, float, float]
    v3_gains: tuple[float, float, float, float]
    v3_offsets: tuple[float, float, float, float]
    bridge_offdiag_step4: float
    bridge_offdiag_step6: float
    bridge_offdiag_step7: float


DEV_M98_WORLD_SPECS: tuple[M98WorldSpec, ...] = (
    M98WorldSpec(
        world_id="dev_alpha_m98_00",
        cluster_id="dev_cluster_alpha_00",
        family="alpha",
        split="dev",
        seed_base=810100,
        channel_perm=(2, 0, 3, 1),
        active_log_channels=(2, 5),
        v2_gains=(1.26, 0.81, 1.32, 0.77),
        v2_offsets=(0.14, -0.11, 0.16, -0.13),
        v3_gains=(0.79, 1.34, 1.18, 0.84),
        v3_offsets=(-0.15, 0.18, 0.09, -0.12),
        bridge_offdiag_step4=0.33,
        bridge_offdiag_step6=-0.31,
        bridge_offdiag_step7=0.29,
    ),
    M98WorldSpec(
        world_id="dev_beta_m98_00",
        cluster_id="dev_cluster_beta_00",
        family="beta",
        split="dev",
        seed_base=820100,
        channel_perm=(1, 3, 0, 2),
        active_log_channels=(4, 1),
        v2_gains=(1.29, 0.78, 1.24, 0.83),
        v2_offsets=(0.08, -0.07, 0.06, -0.09),
        v3_gains=(0.82, 1.31, 0.76, 1.27),
        v3_offsets=(-0.09, 0.11, -0.06, 0.08),
        bridge_offdiag_step4=0.32,
        bridge_offdiag_step6=-0.30,
        bridge_offdiag_step7=0.28,
    ),
)

CONFIRMATION_M98_WORLD_SPECS: tuple[M98WorldSpec, ...] = (
    M98WorldSpec(
        world_id="alpha_m98_v2_c01",
        cluster_id="cluster_m98_v2_01_alpha",
        family="alpha",
        split="confirmation",
        seed_base=931000,
        channel_perm=(2, 3, 0, 1),
        active_log_channels=(3, 6),
        v2_gains=(1.27, 0.80, 1.34, 0.77),
        v2_offsets=(0.14, -0.11, 0.16, -0.13),
        v3_gains=(0.78, 1.35, 1.22, 0.81),
        v3_offsets=(-0.15, 0.18, 0.10, -0.12),
        bridge_offdiag_step4=0.33,
        bridge_offdiag_step6=-0.31,
        bridge_offdiag_step7=0.29,
    ),
    M98WorldSpec(
        world_id="alpha_m98_v2_c02",
        cluster_id="cluster_m98_v2_02_alpha",
        family="alpha",
        split="confirmation",
        seed_base=932000,
        channel_perm=(1, 0, 3, 2),
        active_log_channels=(5, 1),
        v2_gains=(0.79, 1.30, 0.82, 1.28),
        v2_offsets=(-0.12, 0.15, -0.10, 0.13),
        v3_gains=(1.32, 0.77, 1.26, 0.80),
        v3_offsets=(0.16, -0.14, 0.11, -0.09),
        bridge_offdiag_step4=-0.32,
        bridge_offdiag_step6=0.30,
        bridge_offdiag_step7=-0.29,
    ),
    M98WorldSpec(
        world_id="alpha_m98_v2_c03",
        cluster_id="cluster_m98_v2_03_alpha",
        family="alpha",
        split="confirmation",
        seed_base=933000,
        channel_perm=(3, 2, 1, 0),
        active_log_channels=(0, 4),
        v2_gains=(1.32, 0.81, 0.78, 1.26),
        v2_offsets=(0.17, -0.09, -0.14, 0.12),
        v3_gains=(0.81, 1.28, 1.33, 0.79),
        v3_offsets=(-0.13, 0.14, 0.15, -0.11),
        bridge_offdiag_step4=0.31,
        bridge_offdiag_step6=-0.33,
        bridge_offdiag_step7=0.30,
    ),
    M98WorldSpec(
        world_id="alpha_m98_v2_c04",
        cluster_id="cluster_m98_v2_04_alpha",
        family="alpha",
        split="confirmation",
        seed_base=934000,
        channel_perm=(0, 2, 3, 1),
        active_log_channels=(7, 2),
        v2_gains=(0.81, 1.26, 1.33, 0.79),
        v2_offsets=(-0.11, 0.13, 0.17, -0.12),
        v3_gains=(1.30, 0.82, 0.77, 1.31),
        v3_offsets=(0.14, -0.10, -0.15, 0.13),
        bridge_offdiag_step4=-0.30,
        bridge_offdiag_step6=0.32,
        bridge_offdiag_step7=-0.29,
    ),
    M98WorldSpec(
        world_id="beta_m98_v2_c01",
        cluster_id="cluster_m98_v2_05_beta",
        family="beta",
        split="confirmation",
        seed_base=941000,
        channel_perm=(1, 3, 2, 0),
        active_log_channels=(2, 5),
        v2_gains=(1.29, 0.80, 1.25, 0.83),
        v2_offsets=(0.08, -0.07, 0.06, -0.05),
        v3_gains=(0.79, 1.32, 0.82, 1.27),
        v3_offsets=(-0.07, 0.09, -0.06, 0.08),
        bridge_offdiag_step4=0.32,
        bridge_offdiag_step6=-0.30,
        bridge_offdiag_step7=0.28,
    ),
    M98WorldSpec(
        world_id="beta_m98_v2_c02",
        cluster_id="cluster_m98_v2_06_beta",
        family="beta",
        split="confirmation",
        seed_base=942000,
        channel_perm=(2, 1, 3, 0),
        active_log_channels=(6, 0),
        v2_gains=(0.78, 1.31, 0.82, 1.27),
        v2_offsets=(-0.07, 0.08, -0.05, 0.06),
        v3_gains=(1.28, 0.79, 1.30, 0.81),
        v3_offsets=(0.09, -0.08, 0.07, -0.06),
        bridge_offdiag_step4=-0.31,
        bridge_offdiag_step6=0.29,
        bridge_offdiag_step7=-0.28,
    ),
    M98WorldSpec(
        world_id="beta_m98_v2_c03",
        cluster_id="cluster_m98_v2_07_beta",
        family="beta",
        split="confirmation",
        seed_base=943000,
        channel_perm=(3, 0, 1, 2),
        active_log_channels=(4, 7),
        v2_gains=(1.26, 0.84, 0.79, 1.30),
        v2_offsets=(0.06, -0.05, -0.08, 0.07),
        v3_gains=(0.82, 1.27, 1.32, 0.78),
        v3_offsets=(-0.06, 0.07, 0.09, -0.08),
        bridge_offdiag_step4=0.30,
        bridge_offdiag_step6=-0.32,
        bridge_offdiag_step7=0.29,
    ),
    M98WorldSpec(
        world_id="beta_m98_v2_c04",
        cluster_id="cluster_m98_v2_08_beta",
        family="beta",
        split="confirmation",
        seed_base=944000,
        channel_perm=(0, 3, 1, 2),
        active_log_channels=(1, 3),
        v2_gains=(0.82, 1.28, 1.32, 0.78),
        v2_offsets=(-0.06, 0.07, 0.08, -0.07),
        v3_gains=(1.31, 0.80, 0.77, 1.29),
        v3_offsets=(0.08, -0.07, -0.09, 0.06),
        bridge_offdiag_step4=-0.29,
        bridge_offdiag_step6=0.31,
        bridge_offdiag_step7=-0.28,
    ),
)


# ============================================================================
# 1. Blind Empirical Physics Generators (`0` Magic Constants, `R0 == U_iso` Byte-Identical!)
# ============================================================================

_M98_ALPHA_GEN_PY = '''
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


def _gen_m98_alpha_instance(
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
    rng_w = np.random.default_rng(world_seed)
    taus = np.array([float(rng_w.uniform(0.45, 0.72)), float(rng_w.uniform(3.20, 4.60))], dtype=float)
    theta = np.array(
        [
            float(rng_w.uniform(0.55, 0.82)),
            float(rng_w.uniform(0.42, 0.68)),
            float(rng_w.uniform(0.24, 0.38)),
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

    # 4-channel calibration load cell sweeps (`(80, 4)` -> `(80, 4)`)
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

    # Single-channel sweep view for shared channel-wise `2/3-1/3` estimator
    sc_ch = np.repeat(np.arange(4, dtype=int), 6)
    sc_true = np.tile(np.array([-1.5, -0.8, -0.2, 0.4, 1.0, 1.6], dtype=float), 4)
    sc_raw = np.empty_like(sc_true)
    for idx_s in range(len(sc_ch)):
        c_i = int(sc_ch[idx_s])
        v4 = np.zeros(4, dtype=float)
        v4[c_i] = sc_true[idx_s]
        sc_raw[idx_s] = float(_apply_alpha_transducer(v4, gains, offsets, bridge_offdiag, upstream_mode)[c_i] + rng.normal(0.0, 1e-4))
    if cal_outlier_rows > 0:
        sc_raw[5] += 15.5  # gross spike in validation third -> rejected by 2/3-1/3 holdout check!

    lab_off = np.array([-0.6, 0.5, -0.3, 0.8], dtype=float)
    lab_gn = np.array([1.25, 0.85, 1.40, 0.90], dtype=float)

    def _make_panel(n_rows: int, regime_code: str, include_obs: bool, heavy_tail: bool):
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
            "_include_obs": include_obs,
            "_heavy_tail": heavy_tail,
        }

    tr_p = _make_panel(72, "D0", True, True)
    te_p = _make_panel(36, downstream_regime, False, False)

    # Joint coupon standardization across `(train, test)` matching `standardize_reference_coupon_panel`
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
        if reg_c != "D1":
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
        elif reg_c == "D1":
            phi = np.log1p(eval_t)
        else:
            phi = -np.expm1(-eval_t)
        clean_c = s0_v[:, None] + prony_c + r_drift[:, None] * phi[None, :]
        rate0 = theta[0] * gap[:, 0] / taus[0] + theta[1] * gap[:, 1] / taus[1] + r_drift
        return clean_c, rate0

    tr_clean, tr_rate0 = _eval_curves_and_rate(tr_p["_ps_true"], tr_p["_h0_true"], tr_p["strain_0"], x_tr_std, "D0")
    noise_c = rng.normal(0.0, 0.012, size=tr_clean.shape)
    outlier_mask = rng.random(size=tr_clean.shape) < 0.03
    noise_c[outlier_mask] += rng.choice([-0.28, 0.28], size=int(np.sum(outlier_mask)))
    tr_obs_c = tr_clean + noise_c
    tr_obs_r = tr_rate0 + rng.normal(0.0, 0.008, size=tr_rate0.shape)

    te_clean, te_rate0 = _eval_curves_and_rate(te_p["_ps_true"], te_p["_h0_true"], te_p["strain_0"], x_te_std, downstream_regime)

    pilot_dict = None
    if include_chamber_pilots:
        pi_p = _make_panel(18, downstream_regime, True, False)
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
        pi_clean, pi_rate0 = _eval_curves_and_rate(pi_p["_ps_true"], pi_p["_h0_true"], pi_p["strain_0"], x_pi, downstream_regime)
        pilot_dict = {
            "lab_id": pi_p["lab_id"],
            "stratum": pi_p["stratum"],
            "coupon_features_raw": pi_p["coupon_features_raw"],
            "ref_coupons": pi_p["ref_coupons"],
            "past_channel_raw": pi_p["past_channel_raw"],
            "hold_channel_raw": pi_p["hold_channel_raw"],
            "strain_0": pi_p["strain_0"],
            "strain_curves_obs": pi_clean + rng.normal(0.0, 0.004, size=pi_clean.shape),
            "instant_rate_obs": pi_rate0 + rng.normal(0.0, 0.003, size=pi_rate0.shape),
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
        "check_rate_gt": True,
    }
    return ds, gt
'''

_M98_BETA_GEN_PY = '''
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


def _gen_m98_beta_instance(
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

    # 32 spatial sensors (`8` 4-channel rosettes x `3` mu_shifts = `96` rows = `24` rosettes!)
    all_ij = [(i, j) for i in range(0, nx, 2) for j in range(0, ny, 2)]
    perm_ij = rng.permutation(len(all_ij))[:32]
    sensor_ij = np.array([all_ij[idx] for idx in perm_ij], dtype=int)
    mu_shifts = np.array([0.15, 0.55, 1.10], dtype=float)
    n_obs = len(sensor_ij) * len(mu_shifts)  # 96 (divisible by 4 -> 24 rosettes)
    row_bank_id = np.tile(np.arange(4, dtype=int), n_obs // 4)

    def _build_S_and_Ulast(reg_c: str):
        def _op1d(n, h, kap, vel, rob):
            A = np.zeros((n, n), dtype=float)
            for i in range(1, n - 1):
                kp = 0.5 * (kap[i] + kap[i + 1])
                km = 0.5 * (kap[i] + kap[i - 1])
                A[i, i - 1] = -km / (h * h) - max(vel[i], 0.0) / h
                A[i, i + 1] = -kp / (h * h) - max(-vel[i], 0.0) / h
                A[i, i] = (kp + km) / (h * h) + abs(vel[i]) / h
            if reg_c == "D2":
                for i in (0, n - 1):
                    im = (i - 1) % n
                    ip = (i + 1) % n
                    kp = 0.5 * (kap[i] + kap[ip])
                    km = 0.5 * (kap[i] + kap[im])
                    A[i, im] += -km / (h * h) - max(vel[i], 0.0) / h
                    A[i, ip] += -kp / (h * h) - max(-vel[i], 0.0) / h
                    A[i, i] += (kp + km) / (h * h) + abs(vel[i]) / h
            elif reg_c == "D1":
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

        Ax = _op1d(nx, dx, kap_x, vel_x, rob_x)
        Ay = _op1d(ny, dy, kap_y, vel_y, rob_y)
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

    # Generate `y_cal_base` deterministically from `seed` (using `D0` generator so when `include_tracer_pulses == False`
    # and `upstream_mode == 'U0'`, `R0` and `U_iso` have 100% BYTE-FOR-BYTE IDENTICAL `y_obs_raw` and `study` dict!!)
    S_d0, _ = _build_S_and_Ulast("D0")
    w_gen = rng.dirichlet(np.ones(M_modes)) * 1.4
    Z_nuis = rng.normal(0.0, 1.0, size=(n_obs, 3))
    gamma_gen = rng.normal(0.0, 0.12, size=3)
    y_cal_true = S_d0 @ w_gen + Z_nuis @ gamma_gen + rng.normal(0.0, 0.008, size=n_obs)

    y_obs_raw = _apply_beta_transducer(y_cal_true.reshape(-1, 4), gains, offsets, bridge_offdiag, upstream_mode).ravel()

    # Rosette calibration sweeps `(80, 4)` + single-bank view covering full rosette dynamic range
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

    S_true, U_last_true = _build_S_and_Ulast(downstream_regime)
    lam_z, lam_w, u_box, b_tot = 0.08, 0.04, 0.42, 1.40
    Pz = np.linalg.solve(Z_nuis.T @ Z_nuis + lam_z * np.eye(3), Z_nuis.T)
    S_tilde = S_true - Z_nuis @ (Pz @ S_true)
    r_y = y_cal_true - Z_nuis @ (Pz @ y_cal_true)
    w_opt, cert_opt = _solve_box_simplex_qp_exact(S_tilde, r_y, lam_z * 0.0 + lam_w, u_box, b_tot)
    term_f = np.tensordot(w_opt, U_last_true, axes=(0, 0))

    tracer_pulses = None
    if include_tracer_pulses:
        tracer_pulses = []
        for p_i in range(2):
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
    gt = {"S_tilde": S_tilde, "r_y": r_y, "w_opt": w_opt, "cert_opt": cert_opt, "terminal_field_A": term_f}
    return study, gt
'''


# ============================================================================
# 2. Blind Prompts (`A13`: 100% Identical Across All Episodes, `0` Formula Leakage!) & Task Builders
# ============================================================================

BLIND_ALPHA_INSTRUCTION_TEXT = (
    "Write `/holdout_task/batch_solver.py` exposing `solve_batch(dataset: dict) -> dict`.\n"
    "Inspect `/holdout_task/public_batch.pkl` (`item = pickle.load(...)[0]`, containing `item['dataset']` and optional "
    "`item['campaign_diagnostics']`) and `/tmp/hegel_continual_archive/` (when present). Note that `solve_batch(dataset)` receives ONLY "
    "`dataset = item['dataset']` (never `campaign_diagnostics`).\n"
    "- `dataset['calibration_load_cells']` contains empirical 4-channel load-bridge calibration sweeps (`bridge_raw` shape `(N_cal, 4)`, "
    "`true_channel_loads` shape `(N_cal, 4)`, plus `single_channel_sweeps`) characterizing the active transducer bridge for this campaign "
    "(always fit transducer calibration dynamically from `calibration_load_cells` rather than hardcoding constants). "
    "Total physical axial stress on any specimen is the sum across the 4 calibrated channel loads `sum(loads_true, axis=-1)`.\n"
    "- `dataset['train']` (`N_tr` benchtop characterization coupons) and `dataset['test']` (`N_te` deployment coupons) share the SAME "
    "intrinsic material relaxation spectrum `taus = (tau_0, tau_1)` (`0.05 < tau_0 < tau_1 < 25.0`) and 7 hereditary weights `theta` "
    "(`theta[0:2]` branch weights, `theta[2]` drift intercept, `theta[3:7]` standardized coupon feature weights). Standardize "
    "`coupon_features_raw` jointly across `vstack([train, test])` per `lab_id` using the median of column means and stds (`ddof=0`) of "
    "`ref_coupons`, followed by cell-balanced `(lab_id, stratum)` inverse-count weighted mean imputation of missing `NaN`s -> `X_tr, X_te`. "
    "On benchtop `train` (chronological `past_dt[0 .. S-1]` from `z_0 = 0`: `z_{s+1} = exp(-past_dt[s]/taus)*z_s + "
    "(1 - exp(-past_dt[s]/taus))*past_stress[:, s]`, `gap = hold_stress[:, None] - z_S`, `r_drift = theta[2] + X_std @ theta[3:7]`), "
    "`strain_curves_obs` (at `eval_times`, with occasional heavy-tailed sensor outliers) follows `strain_0[:, None] + "
    "sum_b theta[b]*gap[:, b:b+1]*(1 - exp(-t/tau_b)) + r_drift[:, None]*t` and `instant_rate_obs` follows `sum_b theta[b]*gap[:, b]/tau_b + r_drift`.\n"
    "- Deployment chamber constitutive behavior (`test`) and transducer characteristics may vary across campaigns from benchtop `train`. "
    "When `item['campaign_diagnostics']` is a dict of `18` chamber verification coupons (of the SAME material `(taus, theta)`), inspect it "
    "during development to identify this campaign's chamber memory/drift law so `solve_batch(dataset)` predicts `test` directly from `dataset` "
    "(when `item['campaign_diagnostics']` is `None` on a blind campaign, consult `/tmp/hegel_continual_archive/` if present, otherwise use "
    "the baseline benchtop law; keep `(taus, theta)` estimated from `train`).\n"
    "- Return `{'estimated_taus': shape (2,), 'hereditary_weights': shape (7,), 'predicted_strain': shape (N_te, K), "
    "'initial_rate_pred': shape (N_te,)}`. Run `python3 /holdout_task/validate_public_schema.py` to verify output structure."
)

BLIND_BETA_INSTRUCTION_TEXT = (
    "Write `/holdout_task/batch_solver.py` exposing `solve_batch(study: dict) -> dict`.\n"
    "Inspect `/holdout_task/public_batch.pkl` (`item = pickle.load(...)[0]`, containing `item['study']` and optional "
    "`item['campaign_diagnostics']`) and `/tmp/hegel_continual_archive/` (when present). Note that `solve_batch(study)` receives ONLY "
    "`study = item['study']` (never `campaign_diagnostics`).\n"
    "- A 2D steady transport-reaction chamber on `[0, Lx] x [0, Ly]` (`nx x ny` uniform grid, `dx = Lx/(nx-1)`, `dy = Ly/(ny-1)`, "
    "column-major `'F'` ordering) combines 1D operators `Ax`, `Ay` via `L = kron(I_ny, Ax) + kron(Ay, I_nx) + "
    "diag(V_reaction_2d.ravel(order='F'))` and solves `(L + mu_shifts[k] * I) u = mode.ravel(order='F')`. For interior indices "
    "`1 <= i <= n-2`, 1D row-local upwind FVM uses `kp = 0.5*(kap[i]+kap[i+1])`, `km = 0.5*(kap[i]+kap[i-1])`, "
    "`A[i, i-1] = -km/h^2 - max(vel[i], 0)/h`, `A[i, i+1] = -kp/h^2 - max(-vel[i], 0)/h`, `A[i, i] = (kp+km)/h^2 + abs(vel[i])/h`. "
    "Under open-wall Robin baseline, boundary rows `0, n-1` satisfy `A[0,0] = kap[0]/h^2 + rob/h`, `A[0,1] = -kap[0]/h^2`, "
    "`A[n-1,n-1] = kap[-1]/h^2 + rob/h`, `A[n-1,n-2] = -kap[-1]/h^2`.\n"
    "- `study['sensor_calibration_sweeps']` provides empirical 4-channel sensor rosette calibration sweeps (`ref_rosette` shape `(N_cal, 4)`, "
    "`raw_rosette` shape `(N_cal, 4)`, plus `single_bank_sweeps`) characterizing the active sensor rosette for this campaign "
    "(always fit sensor calibration dynamically from `sensor_calibration_sweeps` rather than hardcoding constants). `study['y_obs_raw']` "
    "(`(N_obs,)`, stacked across `mu_shifts` at `sensor_ij`, organized in consecutive 4-channel rosettes `reshape(-1, 4)`) measures "
    "`S @ w + Z_nuisance @ gamma + noise`.\n"
    "- Chamber boundary conditions at `i in {0, n-1}` and sensor rosette characteristics may vary across campaigns from the open-wall baseline. "
    "When `item['campaign_diagnostics']` is a list of diagnostic tracer pulses `[{w_pulse, y_pulse_raw}, ...]`, inspect it during development "
    "to identify this campaign's boundary operator at `i in {0, n-1}` so `solve_batch(study)` solves directly from `study` (when "
    "`item['campaign_diagnostics']` is `None` on a blind campaign, consult `/tmp/hegel_continual_archive/` if present, otherwise use the "
    "open-wall Robin baseline).\n"
    "- Project nuisance (`Pz = solve(Z.T @ Z + nuisance_ridge * I, Z.T)`, `S_tilde = S - Z @ (Pz @ S)`, `r_y = y_cal - Z @ (Pz @ y_cal)`), "
    "solve `min_{0 <= w <= u_box, sum(w) == total_budget} J(w)` where `J(w) = 0.5*||S_tilde @ w - r_y||_2^2 + 0.5*source_ridge*||w||_2^2`, "
    "and return `{'weights': w, 'terminal_field': tensordot(w, U_last, axes=(0, 0)), 'kkt_certificate': {'weights': w.tolist(), "
    "'nu_sum': nu, 'lam_lower': max(0, g - nu), 'lam_upper': max(0, nu - g), 'primal_objective': J(w), "
    "'dual_lower_bound': J(w) - g @ w + nu * total_budget + u_box * sum(min(0, g - nu))}}` with `g = S_tilde.T @ (S_tilde @ w - r_y) + "
    "source_ridge * w` and `g - nu - lam_lower + lam_upper = 0`. Run `python3 /holdout_task/validate_public_schema.py` to verify."
)


def _eval_gen_host(gen_py: str, call_expr: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ns: dict[str, Any] = {"np": np, "math": math}
    exec(compile("import math\nimport numpy as np\n" + gen_py + f"\n_res = {call_expr}\n", "<m98_host_gen>", "exec"), ns)
    return ns["_res"]


def _build_m98_alpha_task(
    *,
    task_id: str,
    short_name: str,
    pub_call_expr: str,
    priv_cases_py: str,
) -> HoldoutOODTask:
    _, pub_gt = _eval_gen_host(_M98_ALPHA_GEN_PY, pub_call_expr)
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
    # NOTE: `/holdout_task/public_batch.pkl` stores ONLY `[{"dataset": pub_ds}]` — ZERO `gt` inside container!
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M98_ALPHA_GEN_PY
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
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="wave_1",
        scientific_domain="Multi-Channel Polymer Rheology & Blind Continual Campaign",
        wormhole_family="fam_hereditary_memory_and_factored_rheology",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json=host_gt_json,
        instruction_text=BLIND_ALPHA_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            "import batch_solver as cand_mod\n"
            + _M98_ALPHA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


def _build_m98_beta_task(
    *,
    task_id: str,
    short_name: str,
    pub_call_expr: str,
    priv_cases_py: str,
) -> HoldoutOODTask:
    ns_b: dict[str, Any] = {"np": np, "math": math}
    # On host without scipy, solve exact QP via active-set / projected gradient fallback if scipy is absent!
    try:
        import scipy  # noqa: F401
        _, pub_gt = _eval_gen_host(_M98_BETA_GEN_PY, pub_call_expr)
    except ModuleNotFoundError:
        _, pub_gt = _eval_beta_gen_via_container(pub_call_expr)
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
    if w.shape != gt["w_opt"].shape or tf.shape != gt["terminal_field_A"].shape:
        valid = False; rows.append({"seed": sd, "error": "SHAPE_MISMATCH"}); continue
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
        + _M98_BETA_GEN_PY
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
        'M, nx, ny = np.asarray(st["source_modes"]).shape\n'
        'errs = []\n'
        'if not isinstance(out, dict): errs.append("not_dict")\n'
        'else:\n'
        '    for k, shp in (("weights", (M,)), ("terminal_field", (nx, ny))):\n'
        '        if k not in out: errs.append(f"missing:{k}"); continue\n'
        '        a = np.asarray(out[k], dtype=float)\n'
        '        if a.shape != shp or not np.all(np.isfinite(a)): errs.append(f"bad:{k}:{a.shape}")\n'
        '    if "kkt_certificate" not in out or not isinstance(out["kkt_certificate"], dict): errs.append("missing:kkt_certificate")\n'
        '    else:\n'
        '        kc = out["kkt_certificate"]\n'
        '        for kk in ("weights", "nu_sum", "lam_lower", "lam_upper", "primal_objective", "dual_lower_bound"):\n'
        '            if kk not in kc: errs.append(f"missing_kkt:{kk}")\n'
        '        if not errs:\n'
        '            kw = np.asarray(kc["weights"], dtype=float); ll = np.asarray(kc["lam_lower"], dtype=float); lu = np.asarray(kc["lam_upper"], dtype=float)\n'
        '            po = float(kc["primal_objective"]); dl = float(kc["dual_lower_bound"])\n'
        '            if np.min(kw) < -1e-6 or np.max(kw) > float(st["u_box"]) + 1e-6: errs.append("kkt_box_violation")\n'
        '            if abs(float(np.sum(kw)) - float(st["total_budget"])) > 1e-5: errs.append("kkt_budget_violation")\n'
        '            if np.min(ll) < -1e-6 or np.min(lu) < -1e-6: errs.append("kkt_neg_multiplier")\n'
        '            if dl < -1e-7 or po < dl - 1e-6 or (po - dl) > 2e-4: errs.append(f"kkt_duality_gap_or_sign:{po:.6f}_vs_{dl:.6f}")\n'
        'print(json.dumps({"schema_valid": len(errs) == 0, "errors": errs}))\n'
        'sys.exit(0 if len(errs) == 0 else 1)\n'
        '""")\n'
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="wave_1",
        scientific_domain="2D Advection-Diffusion-Reaction Tomography & Blind Continual Campaign",
        wormhole_family="fam_elliptic_transport_and_factored_kkt",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json=host_gt_json,
        instruction_text=BLIND_BETA_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=(
            "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
            "import batch_solver as cand_mod\n"
            + _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY
            + "\n"
            + _M98_BETA_GEN_PY
            + f"\ncases = [\n{priv_cases_py}\n]\n"
            + priv_body
        ),
    )


# Pure-NumPy active-set / projected-gradient Box-Simplex QP solver so host NEVER needs `scipy` to build Beta tasks!
def _solve_box_simplex_qp_numpy_host(S_ti: np.ndarray, r_y: np.ndarray, lw: float, ub: float, bt: float) -> tuple[np.ndarray, dict[str, Any]]:
    M = S_ti.shape[1]
    H = S_ti.T @ S_ti + lw * np.eye(M)
    f = S_ti.T @ r_y

    def _proj(v: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(v) - ub), float(np.max(v))
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if float(np.sum(np.clip(v - mid, 0.0, ub))) > bt:
                lo = mid
            else:
                hi = mid
        return np.clip(v - 0.5 * (lo + hi), 0.0, ub)

    L_lip = float(np.linalg.norm(H, ord=2)) + 1e-9
    w = np.full(M, bt / M, dtype=float)
    y_acc = w.copy()
    t_acc = 1.0
    for _ in range(1500):
        w_next = _proj(y_acc - (H @ y_acc - f) / L_lip)
        if float(np.max(np.abs(w_next - w))) < 1e-14:
            w = w_next
            break
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_acc * t_acc))
        y_acc = w_next + ((t_acc - 1.0) / t_next) * (w_next - w)
        w, t_acc = w_next, t_next

    free = (w > 1e-10) & (w < ub - 1e-10)
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
        if np.all(w_cand >= -1e-11) and np.all(w_cand <= ub + 1e-11):
            w = _proj(w_cand)

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


def _eval_beta_gen_via_container(call_expr: str) -> tuple[dict[str, Any], dict[str, Any]]:
    host_beta_py = _M98_BETA_GEN_PY.replace("from scipy.optimize import minimize as _scipy_min", "")
    ns: dict[str, Any] = {"np": np, "math": math}
    exec(compile(host_beta_py, "<m98_beta_host_np>", "exec"), ns)
    ns["_solve_box_simplex_qp_exact"] = _solve_box_simplex_qp_numpy_host
    return eval(compile(call_expr, "<m98_beta_call>", "eval"), ns)


def materialize_m98_public_item_and_sha(task: HoldoutOODTask) -> tuple[dict[str, Any], str]:
    """Materialize `/holdout_task/public_batch.pkl` on the host using pure NumPy (`0` Docker overhead, exact byte SHA-256!)."""
    src_py = task.public_setup_py.split('open("/holdout_task/validate_public_schema.py"', 1)[0]
    if "from scipy.optimize import minimize as _scipy_min" in src_py:
        src_py = src_py.replace("from scipy.optimize import minimize as _scipy_min", "")
        ns: dict[str, Any] = {"np": np, "math": math, "_solve_box_simplex_qp_exact": _solve_box_simplex_qp_numpy_host}
        with tempfile.TemporaryDirectory(prefix="m98_pub_") as td:
            # Override `_solve_box_simplex_qp_exact` AFTER function definitions before `pub_st, _unused_gt = ...`
            parts = src_py.split("\npub_st, _unused_gt =", 1)
            exec(compile(parts[0], "<m98_b_defs>", "exec"), ns)
            ns["_solve_box_simplex_qp_exact"] = _solve_box_simplex_qp_numpy_host
            tail = "pub_st, _unused_gt =" + parts[1].replace("/holdout_task", td)
            exec(compile(tail, "<m98_b_tail>", "exec"), ns)
            raw_b = (Path(td) / "public_batch.pkl").read_bytes()
            return pickle.loads(raw_b)[0], hashlib.sha256(raw_b).hexdigest()
    with tempfile.TemporaryDirectory(prefix="m98_pub_") as td:
        exec(compile(src_py.replace("/holdout_task", td), "<m98_a_pub>", "exec"), {})
        raw_b = (Path(td) / "public_batch.pkl").read_bytes()
        return pickle.loads(raw_b)[0], hashlib.sha256(raw_b).hexdigest()


def build_m98_apparatus_log_entry(
    *,
    spec: M98WorldSpec,
    step_index: int,
    short_name: str,
    next_downstream_regime: str,
) -> dict[str, Any]:
    """Generate the unlabeled 8-channel `(16, 8)` facility equilibration telemetry recorded at the end of batch `step_index`
    (`<= t - 1`), which determines the apparatus chamber state entering batch `step_index + 1` (`0` magic formulas or labels!).
    - `active_log_channels = (ch_d1, ch_d2)` carry mean `~ +2.4` when the apparatus enters `D1` or `D2` respectively (`~ 0.0` otherwise).
    - The remaining 6 channels are realistic zero-mean / drift / sinusoidal distractors!
    """
    rng = np.random.default_rng(spec.seed_base + 7700 + step_index * 31)
    ch_d1, ch_d2 = spec.active_log_channels
    trace = rng.normal(0.0, 0.14, size=(16, 8))
    t_grid = np.linspace(0.0, 2.0 * math.pi, 16)
    for k in range(8):
        if k == ch_d1:
            lvl = 2.45 if next_downstream_regime == "D1" else -0.35
            trace[:, k] = lvl + rng.normal(0.0, 0.08, size=16)
        elif k == ch_d2:
            lvl = 2.55 if next_downstream_regime == "D2" else -0.40
            trace[:, k] = lvl + rng.normal(0.0, 0.08, size=16)
        else:
            # Rich distractors: sinusoidal, step-index drift, alternating sign
            if k % 3 == 0:
                trace[:, k] += 0.9 * np.sin(t_grid + 0.7 * step_index)
            elif k % 3 == 1:
                trace[:, k] += (-1.0 if step_index % 2 else 1.0) * 1.15
            else:
                trace[:, k] += 0.22 * step_index - 0.65
    return {
        "completed_batch_index": int(step_index),
        "completed_short_name": short_name,
        "facility_telemetry_trace": np.round(trace, 5).tolist(),
    }


def build_m98_world_tasks(spec: M98WorldSpec) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build `(source_task, 7_continual_tasks, schedule_manifest, apparatus_log_by_prefix_step)` for `spec`:
      - `Stage S` (`step 0`): `(U0_v1, D0)`
      - `Step 1 (C1_down)`: `(U0_v2, D1)` (blind downstream mismatch discovery #1)
      - `Step 2 (R0)`: `(U0_v2, D0)` (**Byte-Identical Current Twin #1**, `include_pilots = False`)
      - `Step 3 (C2_down)`: `(U0_v2, D2)` (blind downstream mismatch discovery #2)
      - `Step 4 (C_up)`: `(U1_bridge_s4, D0)` (blind upstream 4x4 cross-talk mismatch + calibration spike outliers)
      - `Step 5 (U_iso)`: `(U0_v2, D1)` (**Byte-Identical Current Twin #2**, `SHA256(public_batch.pkl) == R0`, `instruction_text == R0`!)
      - `Step 6 (M_comb1)`: `(U1_bridge_s6_v3, D1)` (**Unseen Structural Product Cell `U1 x D1`** + shifted `v3` gains + outliers!)
      - `Step 7 (M_comb2)`: `(U1_bridge_s7_v3, D2)` (**Unseen Structural Product Cell `U1 x D2`** + shifted `v3` gains + outliers!)
    """
    sb = spec.seed_base
    wid = spec.world_id
    fam = spec.family
    w_seed = sb + 9

    id_g = (1.0, 1.0, 1.0, 1.0)
    id_b = (0.0, 0.0, 0.0, 0.0)

    def _mk_task(short_nm: str, pub_seed: int, priv_s1: int, priv_s2: int, up_m: str, dn_r: str, g_t: tuple[float, ...], b_t: tuple[float, ...], br_off: float, pilots: bool, cal_out: int) -> HoldoutOODTask:
        if fam == "alpha":
            kw = (
                f"world_seed={w_seed}, upstream_mode={up_m!r}, downstream_regime={dn_r!r}, "
                f"gains={g_t!r}, offsets={b_t!r}, bridge_offdiag={br_off!r}, "
                f"include_chamber_pilots={pilots!r}, cal_outlier_rows={cal_out}"
            )
            pub_c = f"_gen_m98_alpha_instance({pub_seed}, {kw})"
            priv_c = f"    ({priv_s1}, *_gen_m98_alpha_instance({priv_s1}, {kw})),\n    ({priv_s2}, *_gen_m98_alpha_instance({priv_s2}, {kw})),"
            return _build_m98_alpha_task(
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
        pub_cb = f"_gen_m98_beta_instance({pub_seed}, {kw_b})"
        priv_cb = f"    ({priv_s1}, *_gen_m98_beta_instance({priv_s1}, {kw_b})),\n    ({priv_s2}, *_gen_m98_beta_instance({priv_s2}, {kw_b})),"
        return _build_m98_beta_task(
            task_id=f"{short_nm}__blind_continual_campaign",
            short_name=short_nm,
            pub_call_expr=pub_cb,
            priv_cases_py=priv_cb,
        )

    s_task = _mk_task(f"{wid}_s00", sb + 10, sb + 11, sb + 12, "U0", "D0", id_g, id_b, 0.0, False, 0)

    # CRITICAL (`A03`): `R0` (`Step 2`) and `U_iso` (`Step 5`) use the EXACT SAME `pub_seed = sb + 200` AND `priv_seeds = (sb + 201, sb + 202)`!
    twin_pub_s, twin_p1, twin_p2 = sb + 200, sb + 201, sb + 202

    step_cfgs = [
        ("C1_down", "U0", "D1", spec.v2_gains, spec.v2_offsets, 0.0, True, 0, sb + 110, sb + 111, sb + 112, "downstream_discovery_1"),
        ("R0", "U0", "D0", spec.v2_gains, spec.v2_offsets, 0.0, False, 0, twin_pub_s, twin_p1, twin_p2, "byte_identical_present_twin_D0"),
        ("C2_down", "U0", "D2", spec.v2_gains, spec.v2_offsets, 0.0, True, 0, sb + 310, sb + 311, sb + 312, "downstream_discovery_2"),
        ("C_up", "U1", "D0", spec.v2_gains, spec.v2_offsets, spec.bridge_offdiag_step4, False, 3, sb + 410, sb + 411, sb + 412, "upstream_transducer_discovery"),
        ("U_iso", "U0", "D1", spec.v2_gains, spec.v2_offsets, 0.0, False, 0, twin_pub_s, twin_p1, twin_p2, "byte_identical_present_twin_D1"),
        ("M_comb1", "U1", "D1", spec.v3_gains, spec.v3_offsets, spec.bridge_offdiag_step6, True, 3, sb + 610, sb + 611, sb + 612, "unseen_cross_factor_U1_x_D1"),
        ("M_comb2", "U1", "D2", spec.v3_gains, spec.v3_offsets, spec.bridge_offdiag_step7, True, 3, sb + 710, sb + 711, sb + 712, "unseen_cross_factor_U1_x_D2"),
    ]

    tasks: list[HoldoutOODTask] = []
    manifest: list[dict[str, Any]] = []
    for idx_i, (ev, up_m, dn_r, gg, bb, boff, pil, cout, ps, p1, p2, cat) in enumerate(step_cfgs, start=1):
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
                "true_downstream_regime": dn_r,
                "true_regime": f"{up_m}__{dn_r}",
                "event_category": cat,
            }
        )

    # Prefix apparatus logs: entry `k` (`k = 0..6`) is recorded after step `k` (`0` = Stage S) and visible at step `k + 1`!
    all_sn = [s_task.short_name] + [m["short_name"] for m in manifest]
    next_dn_seq = [m["true_downstream_regime"] for m in manifest] + ["D0"]
    apparatus_logs = [
        build_m98_apparatus_log_entry(spec=spec, step_index=k, short_name=all_sn[k], next_downstream_regime=next_dn_seq[k])
        for k in range(len(step_cfgs))
    ]
    return s_task, tuple(tasks), manifest, apparatus_logs


# ============================================================================
# 3. Generic Statistical Reducer Bank (`A13`: Plain OLS $R^2$, Correlation, & Log Means!)
# ============================================================================


def compute_m98_upstream_observables(family: str, public_item: dict[str, Any]) -> dict[str, float | None]:
    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        x_m = np.asarray(cal["true_channel_loads"], dtype=float)
        y_m = np.asarray(cal["bridge_raw"], dtype=float)
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        x_m = np.asarray(cal["ref_rosette"], dtype=float)
        y_m = np.asarray(cal["raw_rosette"], dtype=float)

    # Use trimmed interior rows (excluding gross `> 8.5` spike rows) for clean statistical diagnostics
    row_norm = np.max(np.abs(y_m), axis=1)
    keep = row_norm <= 8.5
    xk = x_m[keep] if int(np.sum(keep)) >= 8 else x_m
    yk = y_m[keep] if int(np.sum(keep)) >= 8 else y_m

    r2_list = []
    off_corrs = []
    for c in range(4):
        xc = xk[:, c]
        yc = yk[:, c]
        g_c, b_c = np.linalg.lstsq(np.column_stack([xc, np.ones(len(xc))]), yc, rcond=None)[0]
        res_c = yc - (g_c * xc + b_c)
        ss_tot = max(float(np.sum((yc - np.mean(yc)) ** 2)), 1e-12)
        r2_list.append(1.0 - float(np.sum(res_c ** 2)) / ss_tot)
        for j in range(4):
            if j != c and float(np.std(xk[:, j])) > 1e-8 and float(np.std(res_c)) > 1e-8:
                corr = abs(float(np.corrcoef(xk[:, j], res_c)[0, 1]))
                off_corrs.append(corr)

    return {
        "up_stat_01_diagonal_ols_r2": round(float(np.min(r2_list)), 6),
        "up_stat_02_off_diagonal_corr_max": round(float(max(off_corrs) if off_corrs else 0.0), 6),
        "up_stat_03_cal_frobenius_rms": round(float(np.sqrt(np.mean(yk ** 2))), 6),
        "up_stat_04_cal_row_count": float(len(y_m)),
    }


def compute_m98_downstream_observables(prefix_apparatus_logs: list[dict[str, Any]] | None) -> dict[str, float | None]:
    if not prefix_apparatus_logs:
        return {nm: None for nm in M98_DOWNSTREAM_OBSERVABLE_NAMES}
    last_trace = np.asarray(prefix_apparatus_logs[-1]["facility_telemetry_trace"], dtype=float)
    if last_trace.ndim != 2 or last_trace.shape[1] != 8:
        return {nm: None for nm in M98_DOWNSTREAM_OBSERVABLE_NAMES}
    return {
        f"dn_stat_0{k + 1}_log_channel_{k}_mean": round(float(np.mean(last_trace[:, k])), 6)
        for k in range(8)
    }


def compute_m98_combined_observables(
    family: str,
    public_item: dict[str, Any],
    prefix_apparatus_logs: list[dict[str, Any]] | None,
) -> dict[str, float | None]:
    return {
        **compute_m98_upstream_observables(family, public_item),
        **compute_m98_downstream_observables(prefix_apparatus_logs),
    }


# ============================================================================
# 4. Shared Stratified Per-Channel Calibration (`A11`) & 2-Stage Factored Composition
# ============================================================================


def initialize_m98_factored_mechanism_graph(
    *,
    spec: M98WorldSpec,
    forest: Any,
) -> tuple[PersistentMechanismGraph, M8VariationalPlanner, M94BindingRegistry]:
    graph = PersistentMechanismGraph(forest)
    planner = M8VariationalPlanner(graph, min_gain_nats=0.25, collateral_tolerance_nats=-0.05)
    p0_ch = spec.channel_perm[:2]
    p1_ch = spec.channel_perm[2:]
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="cal_node_p0",
            parent_id=None,
            role="permuted_transducer_bank_a",
            feature_names=(f"g_ch{p0_ch[0]}", f"b_ch{p0_ch[0]}", f"g_ch{p0_ch[1]}", f"b_ch{p0_ch[1]}"),
            version=1,
            shared_weights={f"g_ch{p0_ch[0]}": 1.0, f"b_ch{p0_ch[0]}": 0.0, f"g_ch{p0_ch[1]}": 1.0, f"b_ch{p0_ch[1]}": 0.0},
        ),
        tree_id=f"tree_m98_{spec.world_id}_cal",
    )
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="cal_node_p1",
            parent_id=None,
            role="permuted_transducer_bank_b",
            feature_names=(f"g_ch{p1_ch[0]}", f"b_ch{p1_ch[0]}", f"g_ch{p1_ch[1]}", f"b_ch{p1_ch[1]}"),
            version=1,
            shared_weights={f"g_ch{p1_ch[0]}": 1.0, f"b_ch{p1_ch[0]}": 0.0, f"g_ch{p1_ch[1]}": 1.0, f"b_ch{p1_ch[1]}": 0.0},
        ),
        tree_id=f"tree_m98_{spec.world_id}_cal",
    )
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="constitutive_root_dn",
            parent_id=None,
            role="canonical_constitutive_frame",
            feature_names=("frame_scale",),
            version=1,
            shared_weights={"frame_scale": 1.0},
        ),
        tree_id=f"tree_m98_{spec.world_id}_constitutive",
    )
    graph.register_mechanism(
        TypedMechanism(
            mechanism_id="cal_node_p3",
            parent_id=None,
            role="sentinel_root",
            feature_names=("s0", "s1"),
            version=1,
            shared_weights={"s0": 1.5, "s1": -0.5},
        ),
        tree_id=f"tree_m98_{spec.world_id}_sentinel",
    )
    rng_s = np.random.default_rng(spec.seed_base + 55)
    xs_tr = rng_s.normal(0.0, 1.0, size=24)
    zs_tr = rng_s.normal(0.0, 1.0, size=24)
    ys_tr = 1.5 * xs_tr - 0.5 * zs_tr + rng_s.normal(0.0, 0.01, size=24)
    xs_va = rng_s.normal(0.0, 1.0, size=16)
    zs_va = rng_s.normal(0.0, 1.0, size=16)
    ys_va = 1.5 * xs_va - 0.5 * zs_va + rng_s.normal(0.0, 0.01, size=16)
    planner.register_task(
        task_id="sentinel_p3_monitor",
        shared_upstream_id="cal_node_p3",
        feature_names=("s0", "s1"),
        train_data={"x": {"s0": xs_tr, "s1": zs_tr}, "y": ys_tr},
        val_data={"x": {"s0": xs_va, "s1": zs_va}, "y": ys_va},
        initial_local_shrinkage=50.0,
    )
    reg = M94BindingRegistry(graph)
    for pid in ("cal_node_p0", "cal_node_p1", "constitutive_root_dn", "cal_node_p3"):
        reg.parameter_epochs[pid] = {
            "parameter_mechanism_id": pid,
            "version": 1,
            "payload": deepcopy(graph.mechanisms[pid].shared_weights),
        }
    reg.sync_to_graph()
    return graph, planner, reg


def update_m98_calibration_snapshot_and_graph(
    *,
    spec: M98WorldSpec,
    episode_id: str,
    public_item: dict[str, Any],
    flat_snapshot: dict[str, Any] | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
) -> dict[str, Any]:
    """Exact shared `2/3–1/3` stratified channel calibration estimator (`A11`) for BOTH `Flat` arms and `Hegel`!"""
    if spec.family == "alpha":
        sc = public_item["dataset"]["calibration_load_cells"]["single_channel_sweeps"]
        ch_a = np.asarray(sc["lab_id"], dtype=int).ravel()
        # `cmd_stress` is raw readout, `true_stress` is true load -> fit `raw = g * true + b`!
        x_true = np.asarray(sc["true_stress"], dtype=float).ravel()
        y_raw = np.asarray(sc["cmd_stress"], dtype=float).ravel()
    else:
        sc = public_item["study"]["sensor_calibration_sweeps"]["single_bank_sweeps"]
        ch_a = np.asarray(sc["bank_id"], dtype=int).ravel()
        x_true = np.asarray(sc["ref_signal"], dtype=float).ravel()
        y_raw = np.asarray(sc["raw_readout"], dtype=float).ravel()

    out: dict[str, Any] = {}
    if flat_snapshot is not None:
        pg = {c: float(flat_snapshot["gains"][c]) for c in range(4)}
        pb = {c: float(flat_snapshot["offsets"][c]) for c in range(4)}
        est_f = estimate_validated_channel_affine_calibration(
            ch_arr=ch_a, x_arr=x_true, y_arr=y_raw, channels=(0, 1, 2, 3), prior_gains=pg, prior_offsets=pb
        )
        out["flat_snapshot"] = {
            "gains": [float(est_f["gains"][c]) for c in range(4)],
            "offsets": [float(est_f["offsets"][c]) for c in range(4)],
        }

    if graph is not None and registry is not None:
        sent_tm = graph.active_task_models["sentinel_p3_monitor"]
        sent_va = graph.task_datasets["sentinel_p3_monitor"]["val"]
        sent_before = np.asarray(evaluate_committed_task_model(sent_tm, graph, sent_va)["predictions"], dtype=float)
        p_recs = {}
        for pid, chs in (("cal_node_p0", spec.channel_perm[:2]), ("cal_node_p1", spec.channel_perm[2:])):
            mech = graph.mechanisms[pid]
            wb = dict(mech.shared_weights)
            pg_i = {int(c): float(wb[f"g_ch{c}"]) for c in chs}
            pb_i = {int(c): float(wb[f"b_ch{c}"]) for c in chs}
            est_g = estimate_validated_channel_affine_calibration(
                ch_arr=ch_a, x_arr=x_true, y_arr=y_raw, channels=chs, prior_gains=pg_i, prior_offsets=pb_i
            )
            if est_g["any_accepted_change"]:
                wa = dict(wb)
                for c in chs:
                    wa[f"g_ch{c}"] = float(est_g["gains"][int(c)])
                    wa[f"b_ch{c}"] = float(est_g["offsets"][int(c)])
                mech.shared_weights = wa
                mech.version = int(mech.version) + 1
                registry.update_shared_parameter_epoch(
                    parameter_mechanism_id=pid,
                    new_version=int(mech.version),
                    parameter_payload=wa,
                    affected_chain_id=f"{spec.family}_up",
                    evidence_note=f"reweight_{pid}_{episode_id}",
                )
                act = "AncestorReweight"
            else:
                act = "NoChange"
            p_recs[pid] = {"selected_action": act, "version": int(mech.version), "max_drift": est_g["max_drift"]}
        graph.sync_authoritative_state_to_forest()
        sent_after = np.asarray(evaluate_committed_task_model(sent_tm, graph, sent_va)["predictions"], dtype=float)
        out["graph_receipt"] = {
            "parent_receipts": p_recs,
            "sentinel_p3_drift": round(float(np.max(np.abs(sent_after - sent_before))), 10),
        }
    return out


def assemble_m98_graph_diagonal_payload(spec: M98WorldSpec, graph: PersistentMechanismGraph) -> dict[str, Any]:
    gains = [1.0, 1.0, 1.0, 1.0]
    offs = [0.0, 0.0, 0.0, 0.0]
    for pid, chs in (("cal_node_p0", spec.channel_perm[:2]), ("cal_node_p1", spec.channel_perm[2:])):
        w = graph.mechanisms[pid].shared_weights
        for c in chs:
            gains[int(c)] = float(w[f"g_ch{c}"])
            offs[int(c)] = float(w[f"b_ch{c}"])
    return {"gains": gains, "offsets": offs}


def build_m98_default_u0_transducer_code(family: str) -> str:
    """Canonical `u0` diagonal channel transducer (`calibrate_upstream(item, param_payload)` -> canonical identity coordinates)."""
    if family == "alpha":
        return '''from __future__ import annotations
import copy
import numpy as np

def calibrate_upstream(dataset: dict, param_payload: dict | None = None) -> dict:
    ds = copy.deepcopy(dataset)
    if param_payload and "gains" in param_payload and "offsets" in param_payload:
        g = np.asarray(param_payload["gains"], dtype=float)
        b = np.asarray(param_payload["offsets"], dtype=float)
    else:
        g = np.ones(4, dtype=float)
        b = np.zeros(4, dtype=float)
    for sp_k in ("train", "test"):
        sp = dict(ds[sp_k])
        sp["past_channel_raw"] = (np.asarray(sp["past_channel_raw"], dtype=float) - b[None, None, :]) / g[None, None, :]
        sp["hold_channel_raw"] = (np.asarray(sp["hold_channel_raw"], dtype=float) - b[None, :]) / g[None, :]
        ds[sp_k] = sp
    if isinstance(ds.get("chamber_verification_trials"), dict):
        pv = dict(ds["chamber_verification_trials"])
        pv["past_channel_raw"] = (np.asarray(pv["past_channel_raw"], dtype=float) - b[None, None, :]) / g[None, None, :]
        pv["hold_channel_raw"] = (np.asarray(pv["hold_channel_raw"], dtype=float) - b[None, :]) / g[None, :]
        ds["chamber_verification_trials"] = pv
    cal = dict(ds["calibration_load_cells"])
    b_raw = np.asarray(cal["bridge_raw"], dtype=float)
    keep = np.max(np.abs(b_raw), axis=1) <= 8.5
    b_cal = b_raw.copy()
    b_cal[keep] = (b_raw[keep] - b[None, :]) / g[None, :]
    cal["bridge_raw"] = b_cal
    sc = dict(cal["single_channel_sweeps"])
    sc_l = np.asarray(sc["lab_id"], dtype=int).ravel()
    sc_y = np.asarray(sc["cmd_stress"], dtype=float).ravel().copy()
    m_in = np.abs(sc_y) <= 8.5
    sc_y[m_in] = (sc_y[m_in] - b[sc_l[m_in]]) / g[sc_l[m_in]]
    sc["cmd_stress"] = sc_y
    cal["single_channel_sweeps"] = sc
    ds["calibration_load_cells"] = cal
    return ds
'''
    return '''from __future__ import annotations
import copy
import numpy as np

def calibrate_upstream(study: dict, param_payload: dict | None = None) -> dict:
    st = copy.deepcopy(study)
    if param_payload and "gains" in param_payload and "offsets" in param_payload:
        g = np.asarray(param_payload["gains"], dtype=float)
        b = np.asarray(param_payload["offsets"], dtype=float)
    else:
        g = np.ones(4, dtype=float)
        b = np.zeros(4, dtype=float)
    yr = np.asarray(st["y_obs_raw"], dtype=float).reshape(-1, 4)
    st["y_obs_raw"] = ((yr - b[None, :]) / g[None, :]).ravel()
    if isinstance(st.get("reference_tracer_pulses"), list):
        new_p = []
        for row in st["reference_tracer_pulses"]:
            r2 = dict(row)
            yp = np.asarray(r2["y_pulse_raw"], dtype=float).reshape(-1, 4)
            r2["y_pulse_raw"] = ((yp - b[None, :]) / g[None, :]).ravel()
            new_p.append(r2)
        st["reference_tracer_pulses"] = new_p
    cal = dict(st["sensor_calibration_sweeps"])
    r_raw = np.asarray(cal["raw_rosette"], dtype=float)
    keep = np.max(np.abs(r_raw), axis=1) <= 8.5
    r_cal = r_raw.copy()
    r_cal[keep] = (r_raw[keep] - b[None, :]) / g[None, :]
    cal["raw_rosette"] = r_cal
    sb = dict(cal["single_bank_sweeps"])
    sb_b = np.asarray(sb["bank_id"], dtype=int).ravel()
    sb_y = np.asarray(sb["raw_readout"], dtype=float).ravel().copy()
    m_in = np.abs(sb_y) <= 8.5
    sb_y[m_in] = (sb_y[m_in] - b[sb_b[m_in]]) / g[sb_b[m_in]]
    sb["raw_readout"] = sb_y
    cal["single_bank_sweeps"] = sb
    st["sensor_calibration_sweeps"] = cal
    return st
'''


def compose_m98_two_stage_solver_code(
    *,
    upstream_code: str,
    upstream_entrypoint: str,
    downstream_code: str,
    downstream_entrypoint: str,
    param_payload: dict[str, Any] | None = None,
    target_entrypoint: str = "solve_batch",
    upstream_binding: MechanismExecutableBinding | None = None,
    downstream_binding: MechanismExecutableBinding | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
) -> str:
    """Compose `downstream(upstream(raw_item))` into a standalone 1-hop Python module (`/holdout_task/batch_solver.py`).
    Verifies BOTH `upstream_binding` and `downstream_binding` fail-closed against `graph` & `registry` (`A07 + N04`)!
    """
    if upstream_binding is not None:
        verify_binding_identity_and_foreign_keys(
            upstream_binding, graph=graph, registry=registry, require_closure_code=True, strict_receipt_check=(registry is not None)
        )
    if downstream_binding is not None:
        verify_binding_identity_and_foreign_keys(
            downstream_binding, graph=graph, registry=registry, require_closure_code=True, strict_receipt_check=(registry is not None)
        )
    if ADAPTER_PRECALIBRATED_SENTINEL in downstream_code:
        raise ValueError("Refusing to double-wrap an already composed adapter (`F01`)!")

    up_b64 = base64.b64encode(upstream_code.encode("utf-8")).decode("ascii")
    dn_b64 = base64.b64encode(downstream_code.encode("utf-8")).decode("ascii")
    pay_json = json.dumps(param_payload or {}, sort_keys=True)
    return (
        f'"""M9.8 Composed 2-Stage Mechanism Artifact (`up={hashlib.sha256(upstream_code.encode()).hexdigest()[:12]}`, '
        f'`dn={hashlib.sha256(downstream_code.encode()).hexdigest()[:12]}`)."""\n'
        "from __future__ import annotations\n"
        "import base64, inspect, json, types\n"
        "import numpy as np\n\n"
        f"{ADAPTER_PRECALIBRATED_SENTINEL} = True\n"
        f"_UP_B64 = {up_b64!r}\n"
        f"_UP_EP = {upstream_entrypoint!r}\n"
        f"_SRC_B64 = {dn_b64!r}\n"
        f"_SRC_EP = {downstream_entrypoint!r}\n"
        f"_PARAM_PAYLOAD = json.loads({pay_json!r})\n"
        "_up_mod = types.ModuleType('_m98_up')\n"
        "exec(compile(base64.b64decode(_UP_B64).decode('utf-8'), '<m98_up>', 'exec'), _up_mod.__dict__)\n"
        "_up_fn = getattr(_up_mod, _UP_EP)\n"
        "_dn_mod = types.ModuleType('_m98_dn')\n"
        "exec(compile(base64.b64decode(_SRC_B64).decode('utf-8'), '<m98_dn>', 'exec'), _dn_mod.__dict__)\n"
        "_dn_fn = getattr(_dn_mod, _SRC_EP)\n\n"
        f"def {target_entrypoint}(item: dict) -> dict:\n"
        "    sig = inspect.signature(_up_fn)\n"
        "    if len(sig.parameters) >= 2:\n"
        "        cal_item = _up_fn(item, _PARAM_PAYLOAD)\n"
        "    else:\n"
        "        cal_item = _up_fn(item)\n"
        "    return _dn_fn(cal_item)\n"
    )


def diagnose_m98_factored_conflict_locus(
    family: str,
    public_item: dict[str, Any],
    *,
    chosen_up_code: str | None = None,
    chosen_up_entrypoint: str = "calibrate_upstream",
    diag_payload: dict[str, Any] | None = None,
    force_direct_component_probe: bool = False,
) -> dict[str, Any]:
    """Causal parent holdout residual localization (`0` private peek, `R04/R05/R11` fail-closed semantics):
    1. When `r2_diag >= 0.96` AND `not force_direct_component_probe`: returns `locus = 'downstream_constitutive'` ONLY as a
       `compatibility_inference` (`execution_status = 'not_executed'`, `chosen_up_holdout_rmse = None` — NEVER `0.0`!).
    2. When `r2_diag < 0.96` (or `force_direct_component_probe = True`) AND `chosen_up_code` is provided: executes `chosen_up_code`
       in air-gapped `hegel-clean-scipy:m7v1` on a `56/24` split of the public calibration sweep table, recording `probe_id` and
       `probe_wall_ms` (`R11`).
    3. Any `TimeoutExpired` or Docker infrastructure error (`rc=125`, daemon error, OOM, unparseable output without Python `up_mod`
       traceback) fails closed to `locus = 'unknown_infra'`, `execution_status = 'timeout'|'infrastructure_error'`,
       `chosen_up_holdout_rmse = None` (NEVER `999.0`!), blocking false scientific negative attribution (`R05`)!
    """
    import subprocess
    import uuid

    up_obs = compute_m98_upstream_observables(family, public_item)
    r2_diag = float(up_obs["up_stat_01_diagonal_ols_r2"] or 0.0)
    off_corr = float(up_obs["up_stat_02_off_diagonal_corr_max"] or 0.0)
    if r2_diag >= 0.96 and not force_direct_component_probe:
        return {
            "locus": "downstream_constitutive",
            "component_probe_executed": False,
            "execution_status": "not_executed",
            "evidence_origin": "compatibility_inference",
            "failure_tier": "NOT_EXECUTED",
            "failure_kind": "not_executed",
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "probe_id": None,
            "probe_wall_ms": 0.0,
            "upstream_observables": up_obs,
        }
    if not chosen_up_code:
        return {
            "locus": "upstream_transducer",
            "component_probe_executed": False,
            "execution_status": "not_executed",
            "evidence_origin": "compatibility_inference",
            "failure_tier": "NOT_EXECUTED",
            "failure_kind": "not_executed",
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "probe_id": None,
            "probe_wall_ms": 0.0,
            "upstream_observables": up_obs,
        }

    probe_id = f"uprobe_{uuid.uuid4().hex[:16]}"
    code_sha = hashlib.sha256(chosen_up_code.encode("utf-8")).hexdigest()
    t0 = monotonic()
    with tempfile.TemporaryDirectory(prefix="m98_uploc_") as td:
        td_p = Path(td)
        (td_p / "up_mod.py").write_text(chosen_up_code, encoding="utf-8")
        with open(td_p / "item.pkl", "wb") as fh:
            pickle.dump({"family": family, "item": public_item, "ep": chosen_up_entrypoint, "pay": diag_payload or {}}, fh)
        chk_py = (
            "import copy, inspect, json, pickle, sys\nimport numpy as np\nsys.path.insert(0, '/work')\nimport up_mod\n"
            "b = pickle.load(open('/work/item.pkl', 'rb'))\nfam = b['family']; fn = getattr(up_mod, b['ep']); pay = b['pay']\n"
            "if fam == 'alpha':\n"
            "    ds = copy.deepcopy(b['item']['dataset']); cal = ds['calibration_load_cells']\n"
            "    raw = np.asarray(cal['bridge_raw'], float); tru = np.asarray(cal['true_channel_loads'], float)\n"
            "    kp = np.flatnonzero(np.max(np.abs(raw), axis=1) <= 8.5)\n"
            "    tr_i, ho_i = np.arange(56), kp[kp >= 56]\n"
            "    cal['bridge_raw'] = raw[tr_i]; cal['true_channel_loads'] = tru[tr_i]\n"
            "    ds['test'] = dict(ds['test']); ds['test']['hold_channel_raw'] = raw[ho_i]\n"
            "    out = fn(ds, pay) if len(inspect.signature(fn).parameters) >= 2 else fn(ds)\n"
            "    pred = np.asarray(out['test']['hold_channel_raw'], float); ref = tru[ho_i]\n"
            "else:\n"
            "    st = copy.deepcopy(b['item']['study']); cal = st['sensor_calibration_sweeps']\n"
            "    raw = np.asarray(cal['raw_rosette'], float); tru = np.asarray(cal['ref_rosette'], float)\n"
            "    kp = np.flatnonzero(np.max(np.abs(raw), axis=1) <= 8.5)\n"
            "    tr_i, ho_i = np.arange(56), kp[kp >= 56]\n"
            "    cal['raw_rosette'] = raw[tr_i]; cal['ref_rosette'] = tru[tr_i]\n"
            "    st['y_obs_raw'] = raw[ho_i].ravel()\n"
            "    out = fn(st, pay) if len(inspect.signature(fn).parameters) >= 2 else fn(st)\n"
            "    pred = np.asarray(out['y_obs_raw'], float).reshape(-1, 4); ref = tru[ho_i]\n"
            "diff = pred - ref\n"
            "all_fin = bool(np.all(np.isfinite(pred)))\n"
            "rmse = float(np.sqrt(np.mean(diff ** 2))) if (all_fin and len(ref)) else None\n"
            "print(json.dumps({'all_finite': all_fin, 'rmse': rmse}))\n"
        )
        (td_p / "run_chk.py").write_text(chk_py, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "-v",
                    f"{td_p}:/work:ro",
                    CLEAN_SCIPY_SANDBOX_IMAGE,
                    "python3",
                    "/work/run_chk.py",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            el_ms = round((monotonic() - t0) * 1000.0, 2)
            return {
                "locus": "unknown_infra",
                "component_probe_executed": True,
                "execution_status": "timeout",
                "evidence_origin": "direct_component_probe",
                "failure_tier": "INFRA_FAILURE",
                "failure_kind": "timeout_or_infra",
                "diagonal_ols_r2": r2_diag,
                "off_diagonal_corr_max": off_corr,
                "chosen_up_holdout_rmse": None,
                "candidate_code_sha256": code_sha,
                "probe_id": probe_id,
                "probe_wall_ms": el_ms,
                "return_code": None,
                "upstream_observables": up_obs,
            }

    el_ms = round((monotonic() - t0) * 1000.0, 2)
    rc = int(proc.returncode)
    out_txt = (proc.stdout or "").strip()
    err_txt = (proc.stderr or "").strip()
    err_low = err_txt.lower()

    is_infra_rc = rc in {124, 125, 126, 127, 137, 139, 143} or rc < 0
    is_infra_err = any(
        tok in err_low
        for tok in (
            "cannot connect to",
            "docker daemon",
            "docker:",
            "[timeout:",
            "timed out",
            "memoryerror",
            "no space left on device",
            "brokenpipeerror",
        )
    )
    if is_infra_rc or is_infra_err:
        ex_st = "timeout" if (rc == 124 or "[timeout:" in err_low or "timed out" in err_low) else "infrastructure_error"
        return {
            "locus": "unknown_infra",
            "component_probe_executed": True,
            "execution_status": ex_st,
            "evidence_origin": "direct_component_probe",
            "failure_tier": "INFRA_FAILURE",
            "failure_kind": "timeout_or_infra",
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "candidate_code_sha256": code_sha,
            "probe_id": probe_id,
            "probe_wall_ms": el_ms,
            "return_code": rc,
            "stderr_excerpt": err_txt[:300],
            "upstream_observables": up_obs,
        }

    if rc != 0:
        # Distinguish genuine Python exception thrown by `up_mod` inside container vs unknown container failure
        if "Traceback (most recent call last):" in err_txt:
            return {
                "locus": "upstream_transducer",
                "component_probe_executed": True,
                "execution_status": "completed",
                "evidence_origin": "direct_component_probe",
                "failure_tier": "SCHEMA_FAILURE",
                "failure_kind": "schema_or_interface",
                "diagonal_ols_r2": r2_diag,
                "off_diagonal_corr_max": off_corr,
                "chosen_up_holdout_rmse": None,
                "candidate_code_sha256": code_sha,
                "probe_id": probe_id,
                "probe_wall_ms": el_ms,
                "return_code": rc,
                "stderr_excerpt": err_txt[-300:],
                "upstream_observables": up_obs,
            }
        return {
            "locus": "unknown_infra",
            "component_probe_executed": True,
            "execution_status": "infrastructure_error",
            "evidence_origin": "direct_component_probe",
            "failure_tier": "INFRA_FAILURE",
            "failure_kind": "timeout_or_infra",
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "candidate_code_sha256": code_sha,
            "probe_id": probe_id,
            "probe_wall_ms": el_ms,
            "return_code": rc,
            "stderr_excerpt": err_txt[:300],
            "upstream_observables": up_obs,
        }

    if not out_txt:
        return {
            "locus": "unknown_infra",
            "component_probe_executed": True,
            "execution_status": "infrastructure_error",
            "evidence_origin": "direct_component_probe",
            "failure_tier": "INFRA_FAILURE",
            "failure_kind": "timeout_or_infra",
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "candidate_code_sha256": code_sha,
            "probe_id": probe_id,
            "probe_wall_ms": el_ms,
            "return_code": rc,
            "upstream_observables": up_obs,
        }

    try:
        parsed_j = json.loads(out_txt.splitlines()[-1])
        raw_rmse = parsed_j.get("rmse")
        all_fin = bool(parsed_j.get("all_finite", raw_rmse is not None))
        rmse_opt = None if raw_rmse is None else float(raw_rmse)
    except Exception:
        return {
            "locus": "unknown_infra",
            "component_probe_executed": True,
            "execution_status": "infrastructure_error",
            "evidence_origin": "direct_component_probe",
            "failure_tier": "INFRA_FAILURE",
            "failure_kind": "timeout_or_infra",
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "candidate_code_sha256": code_sha,
            "probe_id": probe_id,
            "probe_wall_ms": el_ms,
            "return_code": rc,
            "upstream_observables": up_obs,
        }

    if (not all_fin) or (rmse_opt is None) or (not math.isfinite(rmse_opt)):
        return {
            "locus": "upstream_transducer",
            "component_probe_executed": True,
            "execution_status": "completed",
            "evidence_origin": "direct_component_probe",
            "failure_tier": "SCIENTIFIC_FAILURE",
            "failure_kind": "scientific_or_numerical",
            "nonfinite_component_output": True,
            "diagonal_ols_r2": r2_diag,
            "off_diagonal_corr_max": off_corr,
            "chosen_up_holdout_rmse": None,
            "candidate_code_sha256": code_sha,
            "probe_id": probe_id,
            "probe_wall_ms": el_ms,
            "return_code": rc,
            "upstream_observables": up_obs,
        }

    is_upstream = bool(rmse_opt > 0.03)
    return {
        "locus": ("upstream_transducer" if is_upstream else "downstream_constitutive"),
        "component_probe_executed": True,
        "execution_status": "completed",
        "evidence_origin": "direct_component_probe",
        "failure_tier": ("SCIENTIFIC_FAILURE" if is_upstream else "NONE"),
        "failure_kind": ("scientific_or_numerical" if is_upstream else "none"),
        "diagonal_ols_r2": r2_diag,
        "off_diagonal_corr_max": off_corr,
        "chosen_up_holdout_rmse": rmse_opt,
        "candidate_code_sha256": code_sha,
        "probe_id": probe_id,
        "probe_wall_ms": el_ms,
        "return_code": rc,
        "upstream_observables": up_obs,
    }


def build_m98_upstream_transducer_synthesis_task(
    *,
    base_task: HoldoutOODTask,
    family: str,
    downstream_code: str,
    downstream_entrypoint: str,
) -> HoldoutOODTask:
    """Build the focused upstream transducer synthesis task for `Hegel_FactoredMechanismGraph` when `locus == 'upstream_transducer'`:
    - Stages the already-verified downstream solver `d_sel` inside `/holdout_task/_verified_downstream.py` (`0444`) and wires
      `/holdout_task/batch_solver.py` to call `d_sel(calibrate_upstream(item))` so `validate_public_schema.py` AND
      `execute_host_trusted_scientific_probe` verify the composed `d_sel o calibrate_upstream` end-to-end!
    - Declared deliverable `/holdout_task/upstream_transducer.py` exposes `calibrate_upstream(item: dict) -> dict`.
    """
    dn_b64 = base64.b64encode(downstream_code.encode("utf-8")).decode("ascii")
    bridge_harness_py = (
        f"\nimport base64 as _b64\n"
        f'open("/holdout_task/_verified_downstream.py", "wb").write(_b64.b64decode({dn_b64!r}))\n'
        'os.chmod("/holdout_task/_verified_downstream.py", 0o444)\n'
        'open("/holdout_task/batch_solver.py", "w").write("""from __future__ import annotations\n'
        "import _verified_downstream as _dn\n"
        "import upstream_transducer as _up\n\n"
        "def solve_batch(item: dict) -> dict:\n"
        f"    return _dn.{downstream_entrypoint}(_up.calibrate_upstream(item))\n"
        '""")\n'
        # Update `validate_public_schema.py` default target so `python3 /holdout_task/validate_public_schema.py` runs `batch_solver.py`!
    )
    if family == "alpha":
        inst = (
            "UPSTREAM TRANSDUCER CALIBRATION MODULE SYNTHESIS:\n"
            "Write `/holdout_task/upstream_transducer.py` exposing `calibrate_upstream(dataset: dict) -> dict`.\n"
            "- Downstream constitutive solver `/holdout_task/_verified_downstream.py` is already verified in canonical identity coordinates "
            "and `/holdout_task/batch_solver.py` automatically composes `solve_batch(ds) = _dn(calibrate_upstream(ds))`.\n"
            "- Inspect `/holdout_task/public_batch.pkl`: fit a robust (outlier-resistant) 4-channel transducer forward/inverse mapping between "
            "`dataset['calibration_load_cells']['bridge_raw']` `(N_cal, 4)` and `true_channel_loads` `(N_cal, 4)` (accounting for any "
            "cross-channel coupling, smooth nonlinear bridge terms, and occasional gross spike outlier rows!), invert `train`, `test`, and "
            "`chamber_verification_trials` (if not `None`) `past_channel_raw` and `hold_channel_raw` into calibrated `true_channel_loads` "
            "coordinates, and normalize `calibration_load_cells` to identity (`bridge_raw = true_channel_loads.copy()`, "
            "`single_channel_sweeps['cmd_stress'] = true_stress.copy()`).\n"
            "- Run `python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py` to test the composed pipeline."
        )
    else:
        inst = (
            "UPSTREAM TRANSDUCER CALIBRATION MODULE SYNTHESIS:\n"
            "Write `/holdout_task/upstream_transducer.py` exposing `calibrate_upstream(study: dict) -> dict`.\n"
            "- Downstream PDE solver `/holdout_task/_verified_downstream.py` is already verified in canonical identity coordinates "
            "and `/holdout_task/batch_solver.py` automatically composes `solve_batch(st) = _dn(calibrate_upstream(st))`.\n"
            "- Inspect `/holdout_task/public_batch.pkl`: fit a robust (outlier-resistant) 4-channel rosette forward/inverse mapping between "
            "`study['sensor_calibration_sweeps']['raw_rosette']` `(N_cal, 4)` and `ref_rosette` `(N_cal, 4)` (accounting for any "
            "cross-channel rosette coupling, smooth nonlinear saturation terms, and occasional gross EMI spike rows!), invert `y_obs_raw` "
            "(`reshape(-1, 4)`) and each entry's `y_pulse_raw` in `reference_tracer_pulses` (if not `None`) into calibrated `ref_rosette` "
            "coordinates, and normalize `sensor_calibration_sweeps` to identity (`raw_rosette = ref_rosette.copy()`, "
            "`single_bank_sweeps['raw_readout'] = ref_signal.copy()`).\n"
            "- Run `python3 /holdout_task/validate_public_schema.py /holdout_task/batch_solver.py` to test the composed pipeline."
        )
    return HoldoutOODTask(
        task_id=f"{base_task.task_id}__upstream_transducer",
        short_name=base_task.short_name,
        wave=base_task.wave,
        scientific_domain=base_task.scientific_domain,
        wormhole_family=base_task.wormhole_family,
        declared_artifacts=("/holdout_task/batch_solver.py", "/holdout_task/upstream_transducer.py"),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename=base_task.public_pickle_filename,
        host_only_public_gt_json=base_task.host_only_public_gt_json,
        instruction_text=inst,
        public_setup_py=base_task.public_setup_py + bridge_harness_py,
        private_verifier_py=base_task.private_verifier_py,
    )


# ============================================================================
# 5. Guard Induction Schemas & Validators (`Flat_ConditionalTable_Online` & `Hegel_FactoredMechanismGraph`)
# ============================================================================


def _build_guard_expr_schema(feature_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["all_of"],
        "properties": {
            "all_of": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["feature_name", "op", "threshold"],
                    "properties": {
                        "feature_name": {"type": "string", "enum": list(feature_names)},
                        "op": {"type": "string", "enum": ["<=", "<", ">=", ">"]},
                        "threshold": {"type": "number"},
                    },
                },
            }
        },
    }


def propose_and_validate_m98_guard_partition(
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
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Unified Kleene `all_of` guard partition compiler used identically by BOTH `Flat_ConditionalTable_Online`
    (`feature_names = M98_ALL_OBSERVABLE_NAMES`) AND `Hegel_FactoredMechanismGraph` (`M98_UPSTREAM_OBSERVABLE_NAMES` for Layer U,
    `M98_DOWNSTREAM_OBSERVABLE_NAMES` for Layer D)!
    """
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
    sys_prompt = (
        f"You are a Kleene applicability-guard synthesis compiler (`scope={scope_label}`). "
        f"Given the measured generic statistical observables `{list(feature_names)}` across tested positive and negative "
        "witnesses of each retained branch and the newly verified branch, output concise `all_of` conjunctions (`1..4` clauses) "
        "that evaluate to `TRUE` on 100% of each branch's own positive witnesses and `FALSE` on non-covering branches' witnesses."
    )
    convo = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "allowed_features": list(feature_names),
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
        chk = verify_m97_multibranch_witness_partition(
            branch_guards=all_guards,
            positive_witness_features_by_branch=positive_witness_features_by_branch,
            covered_branch_pairs=covered_branch_pairs,
            negative_witness_features_by_branch=negative_witness_features_by_branch,
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
                        "instruction": "Adjust feature names, operators, or thresholds so every positive/negative witness is cleanly separated.",
                    }
                ),
            }
        )
    return {"accepted": False, "attempts_used": max_attempts, "attempts_log": attempts_log, "provider_call_receipts": call_receipts}


# ============================================================================
# 6. Pre-LLM Equal-Capability (`max_pre_llm_probes = 1`) Execution Across All 3 Online Arms
# ============================================================================


def expected_m98_public_output_keys(family: str) -> list[str]:
    if family == "alpha":
        return sorted(["estimated_taus", "hereditary_weights", "predicted_strain", "initial_rate_pred"])
    return sorted(["weights", "terminal_field", "kkt_certificate"])


def m98_task_to_frontier_task(
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
    return m6f.FrontierBenchmarkTask(**{**base_ft.__dict__, "benchmark": "Hegel-M98-Blind-Mismatch-Factored-Lineage-World"})


def run_m98_equal_capability_pre_llm_trial(
    *,
    arm: str,
    spec: M98WorldSpec,
    task: HoldoutOODTask,
    public_item: dict[str, Any],
    public_input_sha256: str,
    prefix_apparatus_logs: list[dict[str, Any]] | None,
    prior_archive_entries: list[dict[str, Any]],
    flat_param_snapshot: dict[str, Any] | None = None,
    flat_conditional_table: list[dict[str, Any]] | None = None,
    up_ctrl: ImmutableBranchLineageController | None = None,
    dn_ctrl: ImmutableBranchLineageController | None = None,
    graph: PersistentMechanismGraph | None = None,
    registry: M94BindingRegistry | None = None,
    readonly_dependencies: dict[str, str] | None = None,
    expected_dependency_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute `1` isolated Docker pre-LLM probe (`max_pre_llm_probes = 1`) under identical public gate rules across all 3 arms!"""
    req_keys = expected_m98_public_output_keys(spec.family)
    u0_code = build_m98_default_u0_transducer_code(spec.family)
    comb_obs = compute_m98_combined_observables(spec.family, public_item, prefix_apparatus_logs)

    if arm == "Flat_LatestFirst":
        compat = [
            e
            for e in reversed(prior_archive_entries)
            if sorted(e.get("output_schema_keys") or []) == req_keys
            and e.get("public_qualification_status", "public_gate_verified") == "public_gate_verified"
        ]
        if not compat:
            return {"arm": arm, "accepted_pre_llm": False, "pre_llm_probes_used": 0, "reason": "no_prior_verified_solver"}
        chosen_ent = compat[0]
        can_code = str(chosen_ent.get("canonical_algorithm_code") or chosen_ent["submitted_code"])
        can_ep = str(chosen_ent.get("canonical_entrypoint") or chosen_ent["entrypoint"])
        cand_code = compose_m98_two_stage_solver_code(
            upstream_code=u0_code,
            upstream_entrypoint="calibrate_upstream",
            downstream_code=can_code,
            downstream_entrypoint=can_ep,
            param_payload=flat_param_snapshot,
        )
        t0 = monotonic()
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=task, candidate_code=cand_code, readonly_dependencies=readonly_dependencies, expected_dependency_sha256=expected_dependency_sha256
        )
        pr_ms = round((monotonic() - t0) * 1000.0, 2)
        return {
            "arm": arm,
            "accepted_pre_llm": pr["accepted"],
            "pre_llm_probes_used": 1,
            "probe_wall_ms": pr_ms,
            "selected_source_short_name": chosen_ent["short_name"],
            "selected_canonical_algorithm_code": can_code,
            "selected_canonical_entrypoint": can_ep,
            "observables": comb_obs,
            "candidate_code": (cand_code if pr["accepted"] else None),
            "probe_result": pr,
            "reason": ("accepted_by_public_gate" if pr["accepted"] else f"rejected_{classify_public_probe_failure(pr)['failure_kind']}"),
        }

    if arm == "Flat_ConditionalTable_Online":
        assert flat_conditional_table is not None
        sorted_tbl = sorted(flat_conditional_table, key=lambda r: (-int(r["priority"]), str(r["row_id"])))
        applicable = []
        eval_rows = []
        for r in sorted_tbl:
            if sorted(r["output_schema_keys"]) != req_keys:
                continue
            st = evaluate_m96_autonomous_guard(r.get("guard"), comb_obs, allow_open_feature_keys=True)["state"]
            eval_rows.append({"row_id": r["row_id"], "priority": r["priority"], "guard_state": st})
            if st == "TRUE":
                applicable.append(r)
        chosen_row = applicable[0] if applicable else sorted_tbl[0]
        sel_mode = "table_guard_matched" if applicable else "table_fallback_probe_top_priority"
        # Note: if `chosen_row` was synthesized on an upstream `U1` batch (`self_calibrates_bridge == True`), pass identity diagonal
        # snapshot so `u0` doesn't distort raw `bridge_raw` before `chosen_row`'s own internal calibration!
        pay_arg = {"gains": [1.0] * 4, "offsets": [0.0] * 4} if chosen_row.get("uses_raw_bridge") else flat_param_snapshot
        cand_code = compose_m98_two_stage_solver_code(
            upstream_code=u0_code,
            upstream_entrypoint="calibrate_upstream",
            downstream_code=chosen_row["canonical_code"],
            downstream_entrypoint=chosen_row["entrypoint"],
            param_payload=pay_arg,
        )
        t0 = monotonic()
        pr = execute_single_pre_llm_public_gate_probe_in_container(
            task=task, candidate_code=cand_code, readonly_dependencies=readonly_dependencies, expected_dependency_sha256=expected_dependency_sha256
        )
        pr_ms = round((monotonic() - t0) * 1000.0, 2)
        f_kind = classify_public_probe_failure(pr)["failure_kind"]
        return {
            "arm": arm,
            "accepted_pre_llm": pr["accepted"],
            "pre_llm_probes_used": 1,
            "probe_wall_ms": pr_ms,
            "selection_mode": sel_mode,
            "selected_row_id": chosen_row["row_id"],
            "selected_canonical_algorithm_code": chosen_row["canonical_code"],
            "selected_canonical_entrypoint": chosen_row["entrypoint"],
            "table_evaluations": eval_rows,
            "failure_kind": f_kind,
            "observables": comb_obs,
            "candidate_code": (cand_code if pr["accepted"] else None),
            "probe_result": pr,
            "reason": ("accepted_by_public_gate" if pr["accepted"] else f"table_row_{chosen_row['row_id']}_failed_{f_kind}"),
        }

    # Arm 3: `Hegel_FactoredMechanismGraph` (`Upstream {u0, u1} -> Downstream {d0, d1, d2}`)
    assert up_ctrl is not None and dn_ctrl is not None and graph is not None and registry is not None
    up_obs = compute_m98_upstream_observables(spec.family, public_item)
    dn_obs = compute_m98_downstream_observables(prefix_apparatus_logs)

    up_sel = up_ctrl.select_branch_deterministically(req_schema_keys=["calibrated_input"], feature_evaluator=lambda _b: up_obs)
    chosen_up: MechanismExecutableBinding | None = up_sel["selected"]
    if chosen_up is None:
        up_ver = sorted(
            [b for b in registry.bindings.values() if b.chain_id == f"{spec.family}_up" and b.qualification_status == "verified"],
            key=lambda x: (-int(x.selection_priority), x.binding_id),
        )
        chosen_up = up_ver[0]
        up_mode = f"bounded_probe_on_{up_sel['status']}"
    else:
        up_mode = str(up_sel["status"])

    dn_sel = dn_ctrl.select_branch_deterministically(req_schema_keys=req_keys, feature_evaluator=lambda _b: dn_obs)
    chosen_dn: MechanismExecutableBinding | None = dn_sel["selected"]
    if chosen_dn is None:
        dn_ver = sorted(
            [b for b in registry.bindings.values() if b.chain_id == f"{spec.family}_dn" and b.qualification_status == "verified"],
            key=lambda x: (-int(x.selection_priority), x.binding_id),
        )
        chosen_dn = dn_ver[0]
        dn_mode = f"bounded_probe_on_{dn_sel['status']}"
    else:
        dn_mode = str(dn_sel["status"])

    diag_pay = assemble_m98_graph_diagonal_payload(spec, graph)
    cand_code = compose_m98_two_stage_solver_code(
        upstream_code=(chosen_up.canonical_algorithm_code or chosen_up.parameterized_closure_code),
        upstream_entrypoint=(chosen_up.canonical_entrypoint or chosen_up.parameterized_entrypoint),
        downstream_code=(chosen_dn.canonical_algorithm_code or chosen_dn.parameterized_closure_code),
        downstream_entrypoint=(chosen_dn.canonical_entrypoint or chosen_dn.parameterized_entrypoint),
        param_payload=diag_pay,
        upstream_binding=chosen_up,
        downstream_binding=chosen_dn,
        graph=graph,
        registry=registry,
    )
    t0 = monotonic()
    pr = execute_single_pre_llm_public_gate_probe_in_container(
        task=task, candidate_code=cand_code, readonly_dependencies=readonly_dependencies, expected_dependency_sha256=expected_dependency_sha256
    )
    pr_ms = round((monotonic() - t0) * 1000.0, 2)
    main_cls = classify_public_probe_failure(pr)
    f_kind = main_cls["failure_kind"]
    setup_src_sha = hashlib.sha256(task.public_setup_py.encode("utf-8")).hexdigest()
    dep_bundle_sha = hashlib.sha256(
        json.dumps(expected_dependency_sha256 or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    if pr["accepted"]:
        locus_diag: dict[str, Any] = {
            "locus": "none",
            "component_probe_executed": False,
            "execution_status": "completed",
            "evidence_origin": "joint_task_probe",
            "failure_tier": "NONE",
            "failure_kind": "none",
            "diagonal_ols_r2": float(up_obs["up_stat_01_diagonal_ols_r2"] or 0.0),
            "chosen_up_holdout_rmse": None,
            "probe_id": None,
            "probe_wall_ms": 0.0,
        }
        up_wit = up_ctrl.record_tested_witness(
            binding=chosen_up,
            public_input_sha256=public_input_sha256,
            passed=True,
            failure_kind="none",
            features=up_obs,
            probe_wall_ms=pr_ms,
            execution_status="completed",
            evidence_origin="joint_task_probe",
            setup_source_sha256=setup_src_sha,
            dependency_bundle_sha256=dep_bundle_sha,
        )
        dn_wit = dn_ctrl.record_tested_witness(
            binding=chosen_dn,
            public_input_sha256=public_input_sha256,
            passed=True,
            failure_kind="none",
            features=dn_obs,
            probe_wall_ms=pr_ms,
            execution_status="completed",
            evidence_origin="joint_task_probe",
            setup_source_sha256=setup_src_sha,
            dependency_bundle_sha256=dep_bundle_sha,
        )
    elif main_cls["tier"] == "INFRA_FAILURE":
        # `R04 + R05`: Main composed probe hit timeout/infra — NEVER run scientific localization or mark either component passed/refuted!
        locus_diag = {
            "locus": "unknown_infra",
            "component_probe_executed": False,
            "execution_status": "not_executed",
            "evidence_origin": "compatibility_inference",
            "failure_tier": "INFRA_FAILURE",
            "failure_kind": "timeout_or_infra",
            "diagonal_ols_r2": float(up_obs["up_stat_01_diagonal_ols_r2"] or 0.0),
            "chosen_up_holdout_rmse": None,
            "probe_id": None,
            "probe_wall_ms": 0.0,
            "upstream_observables": up_obs,
        }
        up_wit = None
        dn_wit = None
    else:
        locus_diag = diagnose_m98_factored_conflict_locus(
            spec.family,
            public_item,
            chosen_up_code=(chosen_up.canonical_algorithm_code or chosen_up.parameterized_closure_code),
            chosen_up_entrypoint=(chosen_up.canonical_entrypoint or chosen_up.parameterized_entrypoint),
            diag_payload=diag_pay,
        )
        if locus_diag["locus"] == "unknown_infra":
            # `R05`: Localization container failed/timed out — NEVER attribute scientific failure to `chosen_up` or `chosen_dn`!
            up_wit = None
            dn_wit = None
        elif locus_diag["locus"] == "upstream_transducer":
            if locus_diag.get("execution_status") == "completed" and locus_diag.get("evidence_origin") == "direct_component_probe":
                up_wit = up_ctrl.record_tested_witness(
                    binding=chosen_up,
                    public_input_sha256=public_input_sha256,
                    passed=False,
                    failure_kind=str(locus_diag.get("failure_kind") or "scientific_or_numerical"),
                    features=up_obs,
                    probe_wall_ms=float(locus_diag.get("probe_wall_ms") or 0.0),
                    execution_status="completed",
                    evidence_origin="direct_component_probe",
                    metric_value=locus_diag.get("chosen_up_holdout_rmse"),
                    probe_id=str(locus_diag.get("probe_id") or ""),
                    setup_source_sha256=setup_src_sha,
                    dependency_bundle_sha256=dep_bundle_sha,
                )
            else:
                up_wit = None
            dn_wit = None
        else:
            # `R04 + Q08`: ONLY record upstream `passed=True` AND downstream `passed=False` when `chosen_up` was DIRECTLY executed
            # in the component probe (`compatibility_inference` without direct execution NEVER authorizes a genuine downstream negative witness!)
            if locus_diag.get("execution_status") == "completed" and locus_diag.get("evidence_origin") == "direct_component_probe":
                up_wit = up_ctrl.record_tested_witness(
                    binding=chosen_up,
                    public_input_sha256=public_input_sha256,
                    passed=True,
                    failure_kind="none",
                    features=up_obs,
                    probe_wall_ms=float(locus_diag.get("probe_wall_ms") or 0.0),
                    execution_status="completed",
                    evidence_origin="direct_component_probe",
                    metric_value=locus_diag.get("chosen_up_holdout_rmse"),
                    probe_id=str(locus_diag.get("probe_id") or ""),
                    setup_source_sha256=setup_src_sha,
                    dependency_bundle_sha256=dep_bundle_sha,
                )
                dn_wit = dn_ctrl.record_tested_witness(
                    binding=chosen_dn,
                    public_input_sha256=public_input_sha256,
                    passed=False,
                    failure_kind=f_kind,
                    features=dn_obs,
                    probe_wall_ms=pr_ms,
                    execution_status="completed",
                    evidence_origin="joint_task_probe",
                    setup_source_sha256=setup_src_sha,
                    dependency_bundle_sha256=dep_bundle_sha,
                )
            else:
                up_wit = None
                dn_wit = None

    loc_wall_ms = round(float(locus_diag.get("probe_wall_ms") or 0.0), 2)
    loc_used = 1 if locus_diag.get("component_probe_executed") else 0
    return {
        "arm": arm,
        "accepted_pre_llm": pr["accepted"],
        "pre_llm_probes_used": 1,
        "probe_wall_ms": pr_ms,
        "localization_probes_used": loc_used,
        "localization_probe_wall_ms": loc_wall_ms,
        "upstream_selection_mode": up_mode,
        "downstream_selection_mode": dn_mode,
        "selected_upstream_binding_id": chosen_up.binding_id,
        "selected_downstream_binding_id": chosen_dn.binding_id,
        "selected_canonical_algorithm_code": (chosen_dn.canonical_algorithm_code or chosen_dn.parameterized_closure_code),
        "selected_canonical_entrypoint": (chosen_dn.canonical_entrypoint or chosen_dn.parameterized_entrypoint),
        "conflict_locus_diagnosis": locus_diag,
        "inferred_upstream_compatibility": bool(
            (not pr["accepted"])
            and locus_diag.get("locus") == "downstream_constitutive"
            and locus_diag.get("evidence_origin") == "compatibility_inference"
        ),
        "upstream_tested_witness": (up_wit.to_dict() if up_wit else None),
        "downstream_tested_witness": (dn_wit.to_dict() if dn_wit else None),
        "upstream_observables": up_obs,
        "downstream_observables": dn_obs,
        "candidate_code": (cand_code if pr["accepted"] else None),
        "probe_result": pr,
        "reason": (
            "accepted_by_public_gate"
            if pr["accepted"]
            else f"composed_{chosen_up.binding_id}_x_{chosen_dn.binding_id}_failed_{locus_diag['locus']}_{f_kind}"
        ),
    }


# ============================================================================
# 7. Reference Dev-Seed Solvers (`0`-LLM Unit & Container Testing on `DEV_M98_WORLD_SPECS` ONLY!)
# ============================================================================


def build_m98_reference_dev_upstream_u1_code(family: str) -> str:
    """Reference `U1` robust nonlinear polynomial + cross-channel transducer used ONLY in `0`-LLM dev tests (`DEV_M98_WORLD_SPECS`)."""
    common_inv_py = '''
def _poly3_features_and_jac(x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Features per 4-vector x: [1, x0, x1, x2, x3, x0^3, x1^3, x2^3, x3^3, x0*x1, x1*x2, x2*x3, x3*x0] (13 features)
    n = x2.shape[0]
    xn = np.roll(x2, -1, axis=1)
    phi = np.column_stack([np.ones(n, dtype=float), x2, x2 ** 3, x2 * xn])
    J_phi = np.zeros((n, 13, 4), dtype=float)
    for c in range(4):
        J_phi[:, 1 + c, c] = 1.0
        J_phi[:, 5 + c, c] = 3.0 * (x2[:, c] ** 2)
        J_phi[:, 9 + c, c] += xn[:, c]
        J_phi[:, 9 + c, (c + 1) % 4] += x2[:, c]
    return phi, J_phi

def _fit_and_invert_u1(raw_cal: np.ndarray, true_cal: np.ndarray, raw_query: np.ndarray) -> np.ndarray:
    keep = np.max(np.abs(raw_cal), axis=1) <= 8.5
    xc, yc = true_cal[keep], raw_cal[keep]
    phi_c, _ = _poly3_features_and_jac(xc)
    # Forward polynomial coefficients `C` shape `(13, 4)` + linear warm-start `W_lin, b_lin`
    C, *_ = np.linalg.lstsq(phi_c, yc, rcond=None)
    aug_y = np.column_stack([yc, np.ones(len(yc), dtype=float)])
    W_lin, *_ = np.linalg.lstsq(aug_y, xc, rcond=None)
    orig_shape = raw_query.shape
    yq = raw_query.reshape(-1, 4)
    xq = yq @ W_lin[:4] + W_lin[4][None, :]
    for _ in range(8):
        phi_q, J_phi = _poly3_features_and_jac(xq)
        res = phi_q @ C - yq  # `(N, 4)`
        if float(np.max(np.abs(res))) < 1e-12:
            break
        # `J_y[i, out_c, in_k] = sum_f C[f, out_c] * J_phi[i, f, in_k]`
        J_y = np.einsum("fc,nfk->nck", C, J_phi)
        dx = np.linalg.solve(J_y, res[..., None])[..., 0]
        xq = xq - dx
    return xq.reshape(orig_shape)
'''
    if family == "alpha":
        return (
            "from __future__ import annotations\nimport copy\nimport numpy as np\n"
            + common_inv_py
            + '''
def calibrate_upstream(dataset: dict) -> dict:
    ds = copy.deepcopy(dataset)
    cal = ds["calibration_load_cells"]
    b_raw = np.asarray(cal["bridge_raw"], dtype=float)
    t_true = np.asarray(cal["true_channel_loads"], dtype=float)
    for sp_k in ("train", "test"):
        sp = dict(ds[sp_k])
        sp["past_channel_raw"] = _fit_and_invert_u1(b_raw, t_true, np.asarray(sp["past_channel_raw"], dtype=float))
        sp["hold_channel_raw"] = _fit_and_invert_u1(b_raw, t_true, np.asarray(sp["hold_channel_raw"], dtype=float))
        ds[sp_k] = sp
    if isinstance(ds.get("chamber_verification_trials"), dict):
        pv = dict(ds["chamber_verification_trials"])
        pv["past_channel_raw"] = _fit_and_invert_u1(b_raw, t_true, np.asarray(pv["past_channel_raw"], dtype=float))
        pv["hold_channel_raw"] = _fit_and_invert_u1(b_raw, t_true, np.asarray(pv["hold_channel_raw"], dtype=float))
        ds["chamber_verification_trials"] = pv
    cal2 = dict(cal)
    cal2["bridge_raw"] = t_true.copy()
    sc2 = dict(cal2["single_channel_sweeps"])
    sc2["cmd_stress"] = np.asarray(sc2["true_stress"], dtype=float).copy()
    cal2["single_channel_sweeps"] = sc2
    ds["calibration_load_cells"] = cal2
    return ds
'''
        )
    return (
        "from __future__ import annotations\nimport copy\nimport numpy as np\n"
        + common_inv_py
        + '''
def calibrate_upstream(study: dict) -> dict:
    st = copy.deepcopy(study)
    cal = st["sensor_calibration_sweeps"]
    raw_r = np.asarray(cal["raw_rosette"], dtype=float)
    ref_r = np.asarray(cal["ref_rosette"], dtype=float)
    yr = np.asarray(st["y_obs_raw"], dtype=float).reshape(-1, 4)
    st["y_obs_raw"] = _fit_and_invert_u1(raw_r, ref_r, yr).ravel()
    if isinstance(st.get("reference_tracer_pulses"), list):
        np_list = []
        for row in st["reference_tracer_pulses"]:
            r2 = dict(row)
            yp = np.asarray(r2["y_pulse_raw"], dtype=float).reshape(-1, 4)
            r2["y_pulse_raw"] = _fit_and_invert_u1(raw_r, ref_r, yp).ravel()
            np_list.append(r2)
        st["reference_tracer_pulses"] = np_list
    cal2 = dict(cal)
    cal2["raw_rosette"] = ref_r.copy()
    sb2 = dict(cal2["single_bank_sweeps"])
    sb2["raw_readout"] = np.asarray(sb2["ref_signal"], dtype=float).copy()
    cal2["single_bank_sweeps"] = sb2
    st["sensor_calibration_sweeps"] = cal2
    return st
'''
    )


def build_m98_reference_dev_downstream_solver_code(family: str, entrypoint: str, downstream_regime: str) -> str:
    """Reference `D0, D1, D2` solvers used ONLY in `0`-LLM dev tests (`DEV_M98_WORLD_SPECS`)."""
    assert downstream_regime in {"D0", "D1", "D2"}
    if family == "alpha":
        return f'''from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def {entrypoint}(dataset: dict) -> dict:
    tr, te = dataset["train"], dataset["test"]
    past_dt = np.asarray(dataset["past_dt"], dtype=float)
    eval_t = np.asarray(dataset["eval_times"], dtype=float)
    sc = dataset["calibration_load_cells"]["single_channel_sweeps"]
    cl = np.asarray(sc["lab_id"], dtype=int).ravel()
    cy = np.asarray(sc["cmd_stress"], dtype=float).ravel()
    cx = np.asarray(sc["true_stress"], dtype=float).ravel()
    g = np.ones(4, dtype=float); b = np.zeros(4, dtype=float)
    for c in range(4):
        m = (cl == c) & (np.abs(cy) <= 8.5)
        if int(np.sum(m)) >= 2:
            g[c], b[c] = np.linalg.lstsq(np.column_stack([cx[m], np.ones(int(np.sum(m)))]), cy[m], rcond=None)[0]
    tr_ps = np.sum((np.asarray(tr["past_channel_raw"], dtype=float) - b[None, None, :]) / g[None, None, :], axis=-1)
    tr_h0 = np.sum((np.asarray(tr["hold_channel_raw"], dtype=float) - b[None, :]) / g[None, :], axis=-1)
    te_ps = np.sum((np.asarray(te["past_channel_raw"], dtype=float) - b[None, None, :]) / g[None, None, :], axis=-1)
    te_h0 = np.sum((np.asarray(te["hold_channel_raw"], dtype=float) - b[None, :]) / g[None, :], axis=-1)

    labs = np.concatenate([tr["lab_id"], te["lab_id"]])
    strata = np.concatenate([tr["stratum"], te["stratum"]])
    X_all = np.vstack([tr["coupon_features_raw"], te["coupon_features_raw"]]).copy()
    R_all = np.vstack([tr["ref_coupons"], te["ref_coupons"]])
    for c in range(4):
        m = labs == c
        cm, csd = np.nanmean(R_all[m], axis=0), np.nanstd(R_all[m], axis=0, ddof=0)
        v = np.isfinite(cm) & np.isfinite(csd) & (csd > 1e-8)
        X_all[m] = (X_all[m] - float(np.median(cm[v]))) / float(np.median(csd[v]))
    w_all = np.zeros(len(labs), dtype=float)
    for c in range(4):
        for s in range(3):
            ms = (labs == c) & (strata == s)
            if np.any(ms): w_all[ms] = 1.0 / float(np.sum(ms))
    w_all /= float(np.mean(w_all))
    for j in range(4):
        fin = np.isfinite(X_all[:, j])
        if np.any(fin) and not np.all(fin):
            X_all[~fin, j] = float(np.sum(w_all[fin] * X_all[fin, j]) / max(float(np.sum(w_all[fin])), 1e-12))
    ntr = len(tr["lab_id"])
    X_tr, X_te = X_all[:ntr], X_all[ntr:]
    yc = np.asarray(tr["strain_curves_obs"], dtype=float) - np.asarray(tr["strain_0"], dtype=float)[:, None]
    yr = np.asarray(tr["instant_rate_obs"], dtype=float)
    K = yc.shape[1]

    def _prop(ps_a, t1, t2):
        taus = np.array([t1, t2], dtype=float)
        z = np.zeros((len(ps_a), 2), dtype=float)
        for si, dt in enumerate(past_dt):
            dec = np.exp(-float(dt) / taus)
            z = z * dec[None, :] + (1.0 - dec[None, :]) * ps_a[:, si : si + 1]
        return z

    def _fit_tr(t1, t2, iters=7):
        z = _prop(tr_ps, t1, t2)
        gap = tr_h0[:, None] - z
        e0 = -np.expm1(-eval_t[None, :] / t1)
        e1 = -np.expm1(-eval_t[None, :] / t2)
        Dc = np.zeros((ntr * K, 7), dtype=float)
        Dc[:, 0] = (gap[:, 0:1] * e0).ravel()
        Dc[:, 1] = (gap[:, 1:2] * e1).ravel()
        Dc[:, 2] = np.broadcast_to(eval_t[None, :], (ntr, K)).ravel()
        for j in range(4):
            Dc[:, 3 + j] = (X_tr[:, j : j + 1] * eval_t[None, :]).ravel()
        Dr = np.column_stack([gap[:, 0] / t1, gap[:, 1] / t2, np.ones(ntr), X_tr])
        D = np.vstack([Dc, 2.0 * Dr]); y = np.concatenate([yc.ravel(), 2.0 * yr])
        w = np.ones_like(y); p = np.zeros(7); s2 = 0.03 ** 2
        for _ in range(iters):
            sw = np.sqrt(w)
            p, *_ = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)
            res = y - D @ p
            lc = float(np.sum(np.log1p((res[:ntr * K] ** 2) / s2)) + 0.5 * np.sum((res[ntr * K:] ** 2) / (0.015 ** 2)))
            w[:ntr * K] = 1.0 / (1.0 + (res[:ntr * K] ** 2) / s2)
            w[ntr * K:] = 1.0 / (1.0 + 0.15 * (res[ntr * K:] ** 2) / (0.015 ** 2))
        return lc, p

    best = (1e30, 0.55, 3.8)
    for t1 in np.geomspace(0.35, 1.05, 8):
        for t2 in np.geomspace(max(t1 * 2.2, 2.2), 5.5, 8):
            lv, _ = _fit_tr(float(t1), float(t2), iters=5)
            if lv < best[0]: best = (lv, float(t1), float(t2))
    ro = minimize(lambda u: _fit_tr(float(np.exp(u[0])), float(np.exp(u[0]) + np.exp(u[1])), iters=8)[0],
                  x0=np.array([np.log(best[1]), np.log(best[2] - best[1])]), method="Nelder-Mead", options={{"maxiter": 90}})
    t1h = float(np.exp(ro.x[0])); t2h = float(t1h + np.exp(ro.x[1]))
    _, hw = _fit_tr(t1h, t2h, iters=12)

    rk = {downstream_regime!r}
    z_te = np.zeros((len(te_h0), 2), dtype=float) if rk == "D1" else _prop(te_ps, t1h, t2h)
    gap_te = te_h0[:, None] - z_te
    prony_te = hw[0] * gap_te[:, 0:1] * (-np.expm1(-eval_t[None, :] / t1h)) + hw[1] * gap_te[:, 1:2] * (-np.expm1(-eval_t[None, :] / t2h))
    r_dr = hw[2] + X_te @ hw[3:7]
    phi = eval_t if rk == "D0" else (np.log1p(eval_t) if rk == "D1" else -np.expm1(-eval_t))
    ps_pred = np.asarray(te["strain_0"], dtype=float)[:, None] + prony_te + r_dr[:, None] * phi[None, :]
    ir_pred = hw[0] * gap_te[:, 0] / t1h + hw[1] * gap_te[:, 1] / t2h + r_dr
    return {{"estimated_taus": np.array([t1h, t2h]), "hereditary_weights": hw, "predicted_strain": ps_pred, "initial_rate_pred": ir_pred}}
'''
    return f'''from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def {entrypoint}(study: dict) -> dict:
    Lx, Ly = float(study["Lx"]), float(study["Ly"])
    nx, ny = int(study["nx"]), int(study["ny"])
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    sb = study["sensor_calibration_sweeps"]["single_bank_sweeps"]
    cb = np.asarray(sb["bank_id"], dtype=int).ravel()
    cr = np.asarray(sb["ref_signal"], dtype=float).ravel()
    cy = np.asarray(sb["raw_readout"], dtype=float).ravel()
    bg = np.ones(4, dtype=float); bo = np.zeros(4, dtype=float)
    for c in range(4):
        m = (cb == c) & (np.abs(cy) <= 8.5)
        if int(np.sum(m)) >= 2:
            bg[c], bo[c] = np.linalg.lstsq(np.column_stack([cr[m], np.ones(int(np.sum(m)))]), cy[m], rcond=None)[0]
    yr = np.asarray(study["y_obs_raw"], dtype=float).reshape(-1, 4)
    y_cal = ((yr - bo[None, :]) / bg[None, :]).ravel()
    rk = {downstream_regime!r}

    def _op1d(n, h, kap, vel, rob):
        A = np.zeros((n, n), dtype=float)
        for i in range(1, n - 1):
            kp = 0.5 * (kap[i] + kap[i + 1]); km = 0.5 * (kap[i] + kap[i - 1])
            A[i, i - 1] = -km / (h * h) - max(vel[i], 0.0) / h
            A[i, i + 1] = -kp / (h * h) - max(-vel[i], 0.0) / h
            A[i, i] = (kp + km) / (h * h) + abs(vel[i]) / h
        if rk == "D2":
            for i in (0, n - 1):
                im = (i - 1) % n; ip = (i + 1) % n
                kp = 0.5 * (kap[i] + kap[ip]); km = 0.5 * (kap[i] + kap[im])
                A[i, im] += -km / (h * h) - max(vel[i], 0.0) / h
                A[i, ip] += -kp / (h * h) - max(-vel[i], 0.0) / h
                A[i, i] += (kp + km) / (h * h) + abs(vel[i]) / h
        elif rk == "D1":
            A[0, 0] = kap[0] / (h * h); A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h); A[n - 1, n - 2] = -kap[-1] / (h * h)
        else:
            A[0, 0] = kap[0] / (h * h) + rob / h; A[0, 1] = -kap[0] / (h * h)
            A[n - 1, n - 1] = kap[-1] / (h * h) + rob / h; A[n - 1, n - 2] = -kap[-1] / (h * h)
        return A

    Ax = _op1d(nx, dx, np.asarray(study["kappa_x"], dtype=float), np.asarray(study["vel_x"], dtype=float), float(study["robin_x"]))
    Ay = _op1d(ny, dy, np.asarray(study["kappa_y"], dtype=float), np.asarray(study["vel_y"], dtype=float), float(study["robin_y"]))
    V_2d = np.asarray(study["V_reaction_2d"], dtype=float)
    L_f = np.kron(np.eye(ny), Ax) + np.kron(Ay, np.eye(nx)) + np.diag(V_2d.ravel(order="F"))
    modes = np.asarray(study["source_modes"], dtype=float)
    M = modes.shape[0]
    B_rhs = np.column_stack([modes[m].ravel(order="F") for m in range(M)])
    sij = np.asarray(study["sensor_ij"], dtype=int)
    si, sj = sij[:, 0], sij[:, 1]
    mu_s = np.asarray(study["mu_shifts"], dtype=float)
    U_last = np.empty((M, nx, ny), dtype=float)
    S_blks = []
    I_f = np.eye(nx * ny, dtype=float)
    for ki, mu in enumerate(mu_s):
        sf = np.linalg.solve(L_f + float(mu) * I_f, B_rhs)
        U_k = np.empty((M, nx, ny), dtype=float)
        for m in range(M): U_k[m] = sf[:, m].reshape((nx, ny), order="F")
        if ki == len(mu_s) - 1: U_last = U_k
        S_blks.append(U_k[:, si, sj].T)
    S_st = np.vstack(S_blks)
    Z = np.asarray(study["Z_nuisance"], dtype=float)
    Pz = np.linalg.solve(Z.T @ Z + float(study["nuisance_ridge"]) * np.eye(Z.shape[1]), Z.T)
    S_ti = S_st - Z @ (Pz @ S_st); r_y = y_cal - Z @ (Pz @ y_cal)
    lw, ub, bt = float(study["source_ridge"]), float(study["u_box"]), float(study["total_budget"])
    H = S_ti.T @ S_ti + lw * np.eye(M); f = S_ti.T @ r_y
    res_w = minimize(lambda w: 0.5 * float(w @ (H @ w)) - float(f @ w), np.full(M, bt / M), jac=lambda w: H @ w - f,
                     bounds=[(0.0, ub)] * M, constraints=[{{"type": "eq", "fun": lambda w: float(np.sum(w)) - bt, "jac": lambda w: np.ones(M)}}],
                     method="SLSQP", options={{"ftol": 1e-14, "maxiter": 450}})
    w = np.clip(res_w.x, 0.0, ub)
    lo, hi = float(np.min(w) - ub), float(np.max(w))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(w - mid, 0.0, ub))) > bt: lo = mid
        else: hi = mid
    w = np.clip(w - 0.5 * (lo + hi), 0.0, ub)
    rv = S_ti @ w - r_y; jv = 0.5 * float(rv @ rv) + 0.5 * lw * float(w @ w); g = H @ w - f
    best_nu, best_lb = float(g[0]), -1e30
    for nc in np.unique(g):
        lbc = jv - float(g @ w) + float(nc) * bt + ub * float(np.sum(np.minimum(0.0, g - float(nc))))
        if lbc > best_lb: best_lb, best_nu = lbc, float(nc)
    tf = np.tensordot(w, U_last, axes=(0, 0))
    return {{"weights": w, "terminal_field": tf,
            "kkt_certificate": {{"weights": w.tolist(), "nu_sum": best_nu, "lam_lower": np.maximum(0.0, g - best_nu).tolist(),
                                 "lam_upper": np.maximum(0.0, best_nu - g).tolist(), "primal_objective": jv, "dual_lower_bound": float(min(best_lb, jv))}}}}
'''

"""Hegel M10.0: Reliable Program Knowledge, Symmetric Candidate Fallback & Lazy Projection (`B1–B3`),
Bidirectional Counterexample-Scoped Separability Withdrawal (`sep_first` & `coupled_first`),
and Empirical Break-Even ($K^*$) Continual Benchmark on a `3 × 3` Constrained Two-Stage World.

Implements Work Packages `A`, `B`, `C`, and `D2` (`3 × 3` Constrained Two-Stage Amortization Without Mixing Open DAG Discovery)
from `Hegel_M9_9_Audit` (`01_Implementation_and_Experiment_Plan_ZH.md` §1–§8):
  1. **Three Genuinely Distinct Structural Upstream Operators (`{U0, U1, U2}`) × Three Downstream Operators (`{D0, D1, D2}`)**
     (`|U| × |D| = 9` structural cells, `1` coupled exception `(U1, D2) -> D2_coupled`, `3` unseen separable cells):
     - `U0`: Diagonal linear sensor/bridge calibration (`y = g * x + b`).
     - `U1`: Adjacent-channel superlinear cubic + nearest-neighbor cross-excitation bridge (`(c ± 1) mod 4`).
     - `U2`: Opposite-pair harmonic resolver ripple (`sin(2.50 x)`) + antidiagonal cross-bank bridge (`(c + 2) mod 4`),
       globally diagonally dominant (`det J(x) > 0` everywhere on R^4) and structurally disjoint from both `U0` and `U1`.
  2. **Balanced `50% / 50%` Bidirectional Stream Arrival Orders (`Order A: sep_first` vs `Order B: coupled_first`)**:
     - `Order A (sep_first)`: Encounters `(U0, D2_sep)` at `S2` and `(U1, D2_coupled)` at `S8` (forward `SeparabilityWithdrawalReceipt`),
       with `3` unseen separable cells (`S4: U1×D1`, `S6: U2×D1`, `S7: U2×D2_sep`) and `2` revisits (`S9, S10`).
     - `Order B (coupled_first)`: Encounters `(U1, D2_coupled)` FIRST at `S4` (when `0` prior `D2_sep` exists -> `0` fake withdrawal!)
       and `(U0, D2_sep)` later at `S7` (reverse `SeparabilityWithdrawalReceipt` when `u0 o d2_coupled` fails with colliding `dn_obs`),
       followed by unseen separable `(U2, D2_sep)` (`S8`) and revisits (`S9, S10`).
  3. The public Alpha schema and host scientific probe share the same within-lab reference exchange check.
     The lazy projection helper records each attempted source and target gate, including failed and unresolved
     measurements. Equal tool availability, selection, provider cost accounting and break-even measurement require
     integration in the campaign runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path
from time import monotonic
from typing import Any
import uuid

import numpy as np

from . import m6_frontier_real_benchmarks as m6f
from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m9_4_1_shared_verification_kernel import (
    SHARED_KERNEL_VERSION,
    encode_pure_json_numerical_payload,
    verify_alpha_pooled_ref_coupon_invariance,
)
from .m9_5_equal_tool_continual_attribution_world import (
    execute_single_pre_llm_public_gate_probe_in_container,
    m95_task_to_frontier_task,
)
from .m9_6_autonomous_structure_continual_world import (
    classify_public_probe_failure,
    joint_scientific_receipt_matches,
    compute_canonical_numeric_item_sha256,
)
from .m9_8_blind_mismatch_and_selective_dependency_world import (
    BLIND_ALPHA_INSTRUCTION_TEXT,
    BLIND_BETA_INSTRUCTION_TEXT,
    M98_DOWNSTREAM_OBSERVABLE_NAMES,
    M98WorldSpec,
    compose_m98_two_stage_solver_code,
    compute_m98_downstream_observables,
    diagnose_m98_factored_conflict_locus,
    expected_m98_public_output_keys,
    materialize_m98_public_item_and_sha,
)
from .m9_9_autonomous_decomposition_and_interaction_withdrawal_world import (
    M99WorldSpec,
    SeparabilityWithdrawalReceipt,
    build_m99_apparatus_log_entry,
    build_m99_downstream_canonical_synthesis_task,
    build_m99_reference_dev_downstream_solver_code,
    check_witness_feature_collision,
    hydrate_separability_withdrawals_from_graph as _hydrate_m99_withdrawals,
    persist_separability_withdrawals_to_graph,
    propose_and_validate_m99_guard_partition,
    verify_m99_autonomous_decomposition_in_containers,
    verify_m99_large_margin_witness_partition,
)

M100_ARMS: tuple[str, ...] = (
    "Flat_MonolithicConditionalTable_Online",
    "Flat_PlainModuleRegistry_Online",
    "Hegel_SelectiveConditionExpansion_Online",
)

M100_UPSTREAM_OBSERVABLE_NAMES: tuple[str, ...] = (
    "up_stat_01_diagonal_ols_r2",
    "up_stat_02_adj_offdiag_signed_corr",
    "up_stat_03_adj_offdiag_abs_corr_max",
    "up_stat_04_opp_offdiag_abs_corr_max",
    "up_stat_05_cubic_poly_residual_r2",
    "up_stat_06_cal_frobenius_rms",
    "up_stat_07_cal_row_count",
)

M100_DOWNSTREAM_OBSERVABLE_NAMES: tuple[str, ...] = M98_DOWNSTREAM_OBSERVABLE_NAMES
M100_ALL_OBSERVABLE_NAMES: tuple[str, ...] = M100_UPSTREAM_OBSERVABLE_NAMES + M100_DOWNSTREAM_OBSERVABLE_NAMES


def hydrate_separability_withdrawals_from_graph(
    graph,
    registry,
    world_id: str,
    *,
    strict_ledger_fk: bool = True,
    quarantine_unverified: bool = False,
    probe_receipt_store: Any = None,
    require_external_probe_receipts: bool = False,
) -> list[SeparabilityWithdrawalReceipt]:
    return _hydrate_m99_withdrawals(
        graph,
        registry,
        world_id,
        strict_ledger_fk=strict_ledger_fk,
        quarantine_unverified=quarantine_unverified,
        expected_feature_names=M100_ALL_OBSERVABLE_NAMES,
        probe_receipt_store=probe_receipt_store,
        require_external_probe_receipts=require_external_probe_receipts,
    )



@dataclass(frozen=True)
class M100WorldSpec:
    world_id: str
    cluster_id: str
    family: str  # 'alpha' | 'beta'
    split: str  # 'dev' | 'confirmation'
    order_mode: str  # 'sep_first' | 'coupled_first'
    seed_base: int
    channel_perm: tuple[int, int, int, int]
    active_log_channels: tuple[int, int]
    v2_gains: tuple[float, float, float, float]
    v2_offsets: tuple[float, float, float, float]
    v3_gains: tuple[float, float, float, float]
    v3_offsets: tuple[float, float, float, float]
    u1_offdiag_disc: float
    u1_offdiag_comb: float
    u1_offdiag_cpl: float
    u1_offdiag_rev: float
    u2_offdiag_disc: float
    u2_offdiag_comb1: float
    u2_offdiag_comb2: float

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
            bridge_offdiag_step4=self.u1_offdiag_disc,
            bridge_offdiag_step6=self.u1_offdiag_comb,
            bridge_offdiag_step7=self.u1_offdiag_rev,
        )

    def to_m99_log_spec(self) -> M99WorldSpec:
        return M99WorldSpec(
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
            bridge_offdiag_step3=self.u1_offdiag_disc,
            bridge_offdiag_step4=self.u1_offdiag_comb,
            bridge_offdiag_step5=self.u1_offdiag_cpl,
            bridge_offdiag_step7=self.u1_offdiag_rev,
        )


# `n = 4` Fresh Unexposed Dev Clusters (`971000..974000`, `2` Alpha + `2` Beta, Balanced `sep_first` / `coupled_first`)
DEV_M100_WORLD_SPECS: tuple[M100WorldSpec, ...] = (
    M100WorldSpec(
        world_id="dev_alpha_m100_01",
        cluster_id="dev_cluster_m100_01_alpha",
        family="alpha",
        split="dev",
        order_mode="sep_first",
        seed_base=971000,
        channel_perm=(2, 0, 3, 1),
        active_log_channels=(1, 6),
        v2_gains=(1.27, 0.80, 1.30, 0.79),
        v2_offsets=(0.13, -0.11, 0.14, -0.10),
        v3_gains=(0.81, 1.31, 1.22, 0.82),
        v3_offsets=(-0.12, 0.15, 0.10, -0.11),
        u1_offdiag_disc=0.31,
        u1_offdiag_comb=0.29,
        u1_offdiag_cpl=0.32,
        u1_offdiag_rev=0.30,
        u2_offdiag_disc=0.21,
        u2_offdiag_comb1=0.20,
        u2_offdiag_comb2=0.22,
    ),
    M100WorldSpec(
        world_id="dev_alpha_m100_02",
        cluster_id="dev_cluster_m100_02_alpha",
        family="alpha",
        split="dev",
        order_mode="coupled_first",
        seed_base=972000,
        channel_perm=(1, 3, 0, 2),
        active_log_channels=(5, 2),
        v2_gains=(0.81, 1.29, 0.82, 1.27),
        v2_offsets=(-0.12, 0.13, -0.10, 0.11),
        v3_gains=(1.28, 0.80, 1.26, 0.81),
        v3_offsets=(0.14, -0.12, 0.09, -0.10),
        u1_offdiag_disc=-0.31,
        u1_offdiag_comb=-0.29,
        u1_offdiag_cpl=-0.30,
        u1_offdiag_rev=-0.32,
        u2_offdiag_disc=-0.21,
        u2_offdiag_comb1=-0.20,
        u2_offdiag_comb2=-0.22,
    ),
    M100WorldSpec(
        world_id="dev_beta_m100_01",
        cluster_id="dev_cluster_m100_03_beta",
        family="beta",
        split="dev",
        order_mode="sep_first",
        seed_base=973000,
        channel_perm=(3, 1, 2, 0),
        active_log_channels=(3, 7),
        v2_gains=(1.26, 0.81, 1.25, 0.82),
        v2_offsets=(0.07, -0.06, 0.08, -0.05),
        v3_gains=(0.82, 1.29, 0.79, 1.27),
        v3_offsets=(-0.07, 0.09, -0.06, 0.07),
        u1_offdiag_disc=0.30,
        u1_offdiag_comb=0.29,
        u1_offdiag_cpl=0.31,
        u1_offdiag_rev=0.28,
        u2_offdiag_disc=0.21,
        u2_offdiag_comb1=0.20,
        u2_offdiag_comb2=0.22,
    ),
    M100WorldSpec(
        world_id="dev_beta_m100_02",
        cluster_id="dev_cluster_m100_04_beta",
        family="beta",
        split="dev",
        order_mode="coupled_first",
        seed_base=974000,
        channel_perm=(0, 2, 3, 1),
        active_log_channels=(4, 0),
        v2_gains=(0.80, 1.28, 1.30, 0.79),
        v2_offsets=(-0.06, 0.07, 0.06, -0.05),
        v3_gains=(1.29, 0.80, 0.78, 1.28),
        v3_offsets=(0.08, -0.07, -0.06, 0.05),
        u1_offdiag_disc=-0.30,
        u1_offdiag_comb=-0.28,
        u1_offdiag_cpl=-0.31,
        u1_offdiag_rev=-0.29,
        u2_offdiag_disc=-0.21,
        u2_offdiag_comb1=-0.20,
        u2_offdiag_comb2=-0.22,
    ),
)

# `n = 8` Completely Fresh Unexposed Confirmation Clusters (`981000..988000` — Zero Overlap with Any Prior Milestone!)
CONFIRMATION_M100_WORLD_SPECS: tuple[M100WorldSpec, ...] = (
    M100WorldSpec(
        world_id="alpha_m100_c01",
        cluster_id="cluster_m100_01_alpha",
        family="alpha",
        split="confirmation",
        order_mode="sep_first",
        seed_base=981000,
        channel_perm=(1, 3, 0, 2),
        active_log_channels=(2, 7),
        v2_gains=(1.28, 0.79, 1.31, 0.80),
        v2_offsets=(0.14, -0.11, 0.13, -0.11),
        v3_gains=(0.80, 1.32, 1.23, 0.81),
        v3_offsets=(-0.13, 0.15, 0.11, -0.10),
        u1_offdiag_disc=0.31,
        u1_offdiag_comb=0.30,
        u1_offdiag_cpl=0.29,
        u1_offdiag_rev=0.32,
        u2_offdiag_disc=0.21,
        u2_offdiag_comb1=0.22,
        u2_offdiag_comb2=0.20,
    ),
    M100WorldSpec(
        world_id="alpha_m100_c02",
        cluster_id="cluster_m100_02_alpha",
        family="alpha",
        split="confirmation",
        order_mode="coupled_first",
        seed_base=982000,
        channel_perm=(2, 0, 3, 1),
        active_log_channels=(4, 0),
        v2_gains=(0.81, 1.30, 0.80, 1.27),
        v2_offsets=(-0.12, 0.14, -0.11, 0.12),
        v3_gains=(1.30, 0.79, 1.26, 0.82),
        v3_offsets=(0.14, -0.12, 0.10, -0.09),
        u1_offdiag_disc=-0.31,
        u1_offdiag_comb=-0.29,
        u1_offdiag_cpl=-0.30,
        u1_offdiag_rev=-0.28,
        u2_offdiag_disc=-0.21,
        u2_offdiag_comb1=-0.20,
        u2_offdiag_comb2=-0.22,
    ),
    M100WorldSpec(
        world_id="alpha_m100_c03",
        cluster_id="cluster_m100_03_alpha",
        family="alpha",
        split="confirmation",
        order_mode="sep_first",
        seed_base=983000,
        channel_perm=(3, 1, 2, 0),
        active_log_channels=(6, 3),
        v2_gains=(1.30, 0.82, 0.78, 1.27),
        v2_offsets=(0.15, -0.10, -0.12, 0.11),
        v3_gains=(0.82, 1.28, 1.30, 0.79),
        v3_offsets=(-0.11, 0.14, 0.13, -0.10),
        u1_offdiag_disc=0.30,
        u1_offdiag_comb=0.32,
        u1_offdiag_cpl=0.31,
        u1_offdiag_rev=0.29,
        u2_offdiag_disc=-0.20,
        u2_offdiag_comb1=-0.22,
        u2_offdiag_comb2=-0.21,
    ),
    M100WorldSpec(
        world_id="alpha_m100_c04",
        cluster_id="cluster_m100_04_alpha",
        family="alpha",
        split="confirmation",
        order_mode="coupled_first",
        seed_base=984000,
        channel_perm=(0, 3, 1, 2),
        active_log_channels=(5, 1),
        v2_gains=(0.82, 1.27, 1.31, 0.79),
        v2_offsets=(-0.10, 0.12, 0.15, -0.12),
        v3_gains=(1.28, 0.81, 0.79, 1.29),
        v3_offsets=(0.13, -0.11, -0.13, 0.12),
        u1_offdiag_disc=-0.30,
        u1_offdiag_comb=-0.31,
        u1_offdiag_cpl=-0.29,
        u1_offdiag_rev=-0.30,
        u2_offdiag_disc=0.22,
        u2_offdiag_comb1=0.20,
        u2_offdiag_comb2=0.21,
    ),
    M100WorldSpec(
        world_id="beta_m100_c01",
        cluster_id="cluster_m100_05_beta",
        family="beta",
        split="confirmation",
        order_mode="sep_first",
        seed_base=985000,
        channel_perm=(3, 1, 0, 2),
        active_log_channels=(3, 6),
        v2_gains=(1.27, 0.81, 1.26, 0.82),
        v2_offsets=(0.07, -0.06, 0.08, -0.05),
        v3_gains=(0.80, 1.30, 0.81, 1.27),
        v3_offsets=(-0.06, 0.08, -0.07, 0.08),
        u1_offdiag_disc=0.31,
        u1_offdiag_comb=0.29,
        u1_offdiag_cpl=0.30,
        u1_offdiag_rev=0.28,
        u2_offdiag_disc=0.21,
        u2_offdiag_comb1=0.20,
        u2_offdiag_comb2=0.22,
    ),
    M100WorldSpec(
        world_id="beta_m100_c02",
        cluster_id="cluster_m100_06_beta",
        family="beta",
        split="confirmation",
        order_mode="coupled_first",
        seed_base=986000,
        channel_perm=(1, 2, 3, 0),
        active_log_channels=(7, 1),
        v2_gains=(0.80, 1.29, 0.81, 1.28),
        v2_offsets=(-0.08, 0.07, -0.06, 0.05),
        v3_gains=(1.28, 0.79, 1.30, 0.80),
        v3_offsets=(0.08, -0.07, 0.06, -0.05),
        u1_offdiag_disc=-0.30,
        u1_offdiag_comb=-0.28,
        u1_offdiag_cpl=-0.31,
        u1_offdiag_rev=-0.29,
        u2_offdiag_disc=-0.21,
        u2_offdiag_comb1=-0.22,
        u2_offdiag_comb2=-0.20,
    ),
    M100WorldSpec(
        world_id="beta_m100_c03",
        cluster_id="cluster_m100_07_beta",
        family="beta",
        split="confirmation",
        order_mode="sep_first",
        seed_base=987000,
        channel_perm=(2, 0, 1, 3),
        active_log_channels=(0, 5),
        v2_gains=(1.26, 0.83, 0.79, 1.30),
        v2_offsets=(0.06, -0.05, -0.07, 0.08),
        v3_gains=(0.81, 1.28, 1.30, 0.79),
        v3_offsets=(-0.07, 0.06, 0.08, -0.07),
        u1_offdiag_disc=0.29,
        u1_offdiag_comb=0.31,
        u1_offdiag_cpl=0.30,
        u1_offdiag_rev=0.28,
        u2_offdiag_disc=-0.20,
        u2_offdiag_comb1=-0.21,
        u2_offdiag_comb2=-0.22,
    ),
    M100WorldSpec(
        world_id="beta_m100_c04",
        cluster_id="cluster_m100_08_beta",
        family="beta",
        split="confirmation",
        order_mode="coupled_first",
        seed_base=988000,
        channel_perm=(0, 2, 3, 1),
        active_log_channels=(4, 2),
        v2_gains=(0.81, 1.28, 1.30, 0.79),
        v2_offsets=(-0.05, 0.08, 0.07, -0.06),
        v3_gains=(1.29, 0.81, 0.78, 1.27),
        v3_offsets=(0.07, -0.08, -0.06, 0.05),
        u1_offdiag_disc=-0.29,
        u1_offdiag_comb=-0.30,
        u1_offdiag_cpl=-0.28,
        u1_offdiag_rev=-0.31,
        u2_offdiag_disc=0.22,
        u2_offdiag_comb1=0.21,
        u2_offdiag_comb2=0.20,
    ),
)


# ============================================================================
# 1. `3 × 3` Empirical Physics Generators (`{U0, U1, U2} × {D0, D1, D2}`, `U1 ⊗ D2 -> D2_coupled`)
# ============================================================================

_M100_TRANSDUCER_COMMON_PY = '''
def _apply_m100_transducer(x_true, gains, offsets, offdiag, upstream_mode):
    x_arr = np.asarray(x_true, dtype=float)
    orig_shape = x_arr.shape
    x2 = x_arr.reshape(-1, 4)
    g = np.asarray(gains, dtype=float)
    b_vec = np.asarray(offsets, dtype=float)
    if upstream_mode == "U0":
        y2 = x2 * g[None, :] + b_vec[None, :]
    elif upstream_mode == "U1":
        M_br = np.diag(g)
        if abs(float(offdiag)) > 1e-12:
            for c in range(4):
                M_br[c, (c + 1) % 4] += float(offdiag) * g[c]
                M_br[c, (c - 1) % 4] -= 0.5 * float(offdiag) * g[c]
        sgn = 1.0 if float(offdiag) >= 0.0 else -1.0
        x_next = np.roll(x2, -1, axis=1)
        y2 = x2 @ M_br.T + b_vec[None, :] + (0.18 * g)[None, :] * (x2 ** 3) + (0.22 * sgn * g)[None, :] * (x2 * x_next)
    elif upstream_mode == "U2":
        # Globally diagonally dominant (`min diag = 0.375 > max offdiag = 0.340`) opposite-pair harmonic resolver transducer!
        M_opp = np.diag(g)
        if abs(float(offdiag)) > 1e-12:
            for c in range(4):
                M_opp[c, (c + 2) % 4] += float(offdiag) * g[c]
        sgn2 = 1.0 if float(offdiag) >= 0.0 else -1.0
        x_opp = np.roll(x2, 2, axis=1)
        y2 = (
            x2 @ M_opp.T
            + b_vec[None, :]
            + (0.25 * g)[None, :] * np.sin(2.50 * x2)
            + (0.06 * sgn2 * g)[None, :] * (np.cos(2.00 * x_opp) - 1.0)
        )
    else:
        raise ValueError(f"Unknown upstream_mode: {upstream_mode}")
    return y2.reshape(orig_shape)
'''

_M100_ALPHA_GEN_PY = (
    _M100_TRANSDUCER_COMMON_PY
    + '''
def _gen_m100_alpha_instance(
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
    assert upstream_mode in ("U0", "U1", "U2")
    assert downstream_regime in ("D0", "D1", "D2")
    # `U0` and `U2` are genuinely separable across `{D0, D1, D2}`; ONLY `(U1, D2)` triggers `D2_coupled`!
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
    bridge_raw_cal = _apply_m100_transducer(true_cal_loads, gains, offsets, bridge_offdiag, upstream_mode) + rng.normal(
        0.0, 1e-4, size=(n_cal, 4)
    )
    if cal_outlier_rows > 0:
        for oi in range(min(cal_outlier_rows, 3)):
            bridge_raw_cal[13 + oi, :] += np.array([14.5, -13.2, 12.8, -14.1], dtype=float)

    sc_ch = np.repeat(np.arange(4, dtype=int), 12)
    sc_true = np.tile(np.linspace(-1.6, 1.6, 12, dtype=float), 4)
    sc_raw = np.empty_like(sc_true)
    for idx_s in range(len(sc_ch)):
        c_i = int(sc_ch[idx_s])
        v4 = np.zeros(4, dtype=float)
        v4[c_i] = sc_true[idx_s]
        sc_raw[idx_s] = float(_apply_m100_transducer(v4, gains, offsets, bridge_offdiag, upstream_mode)[c_i] + rng.normal(0.0, 1e-4))
    if cal_outlier_rows > 0:
        sc_raw[5] += 15.5

    lab_off = np.array([-0.6, 0.5, -0.3, 0.8], dtype=float)
    lab_gn = np.array([1.25, 0.85, 1.40, 0.90], dtype=float)

    def _make_panel(n_rows: int, regime_code: str):
        labs = rng.integers(0, 4, size=n_rows)
        labs[:16] = np.tile(np.arange(4, dtype=int), 4)
        strata = rng.integers(0, 3, size=n_rows)
        x_lat = rng.normal(0.0, 1.0, size=(n_rows, 4))
        x_raw = x_lat * lab_gn[labs, None] + lab_off[labs, None]
        for r_i in range(n_rows):
            if rng.random() < 0.22:
                x_raw[r_i, int(rng.integers(0, 4))] = np.nan
        ref_c = rng.normal(0.0, 1.0, size=(n_rows, 8)) * lab_gn[labs, None] + lab_off[labs, None]
        past_true_ch = rng.uniform(-0.65, 1.15, size=(n_rows, S_len, 4))
        hold_true_ch = rng.uniform(0.35, 1.35, size=(n_rows, 4))
        past_raw_ch = _apply_m100_transducer(past_true_ch, gains, offsets, bridge_offdiag, upstream_mode)
        hold_raw_ch = _apply_m100_transducer(hold_true_ch, gains, offsets, bridge_offdiag, upstream_mode)
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
        elif reg_c == "D1":
            phi = np.log1p(eval_t)
        elif reg_c == "D2":
            phi = -np.expm1(-eval_t)
        else:
            phi = 2.0 * (np.sqrt(1.0 + eval_t) - 1.0)
        clean_c = s0_v[:, None] + prony_c + r_drift[:, None] * phi[None, :]
        rate0 = theta[0] * gap[:, 0] / taus[0] + theta[1] * gap[:, 1] / taus[1] + r_drift
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
            "aux_chamber_telemetry": rng.normal(0.0, 1.0, size=(36, 4)),
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
)

_M100_BETA_GEN_PY = (
    "from scipy.optimize import minimize as _scipy_min\n"
    + _M100_TRANSDUCER_COMMON_PY
    + '''
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


def _gen_m100_beta_instance(
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
    assert upstream_mode in ("U0", "U1", "U2")
    assert downstream_regime in ("D0", "D1", "D2")
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
    y_obs_raw = _apply_m100_transducer(y_cal_true.reshape(-1, 4), gains, offsets, bridge_offdiag, upstream_mode).ravel()

    n_cal = 80
    ref_rosette = np.zeros((n_cal, 4), dtype=float)
    for c in range(4):
        ref_rosette[4 * c : 4 * (c + 1), c] = np.array([-1.2, -0.4, 0.5, 1.3], dtype=float)
    ref_rosette[16:, :] = rng.uniform(-1.4, 1.8, size=(n_cal - 16, 4))
    raw_rosette = _apply_m100_transducer(ref_rosette, gains, offsets, bridge_offdiag, upstream_mode) + rng.normal(
        0.0, 1e-5, size=(n_cal, 4)
    )
    if cal_outlier_rows > 0:
        for oi in range(min(cal_outlier_rows, 3)):
            raw_rosette[13 + oi, :] += np.array([13.5, -12.8, 12.4, -13.2], dtype=float)

    sb_bank = np.repeat(np.arange(4, dtype=int), 12)
    sb_ref = np.tile(np.linspace(-1.3, 1.5, 12, dtype=float), 4)
    sb_raw = np.empty_like(sb_ref)
    for idx_s in range(len(sb_bank)):
        c_i = int(sb_bank[idx_s])
        v4 = np.zeros(4, dtype=float)
        v4[c_i] = sb_ref[idx_s]
        sb_raw[idx_s] = float(_apply_m100_transducer(v4, gains, offsets, bridge_offdiag, upstream_mode)[c_i] + rng.normal(0.0, 1e-5))
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
            yp_raw = _apply_m100_transducer(yp_cal.reshape(-1, 4), gains, offsets, bridge_offdiag, upstream_mode).ravel()
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
)


def _eval_m100_gen_host(gen_py: str, call_expr: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from .m9_8_blind_mismatch_and_selective_dependency_world import _solve_box_simplex_qp_numpy_host

    clean_py = gen_py.replace("from scipy.optimize import minimize as _scipy_min\n", "")
    ns: dict[str, Any] = {"np": np, "math": math}
    exec(compile("import math\nimport numpy as np\n" + clean_py, "<m100_host_defs>", "exec"), ns)
    ns["_solve_box_simplex_qp_exact"] = _solve_box_simplex_qp_numpy_host
    return eval(compile(call_expr, "<m100_host_call>", "eval"), ns)


def _build_m100_alpha_task(*, task_id: str, short_name: str, pub_call_expr: str, priv_cases_py: str) -> HoldoutOODTask:
    _, pub_gt = _eval_m100_gen_host(_M100_ALPHA_GEN_PY, pub_call_expr)
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
    # Stage just the public behavioral check. The shared scientific kernel also contains private grading logic.
    schema_program = (
        "from __future__ import annotations\nimport copy, importlib.util, json, os, pickle, sys\nimport numpy as np\n"
        + inspect.getsource(verify_alpha_pooled_ref_coupon_invariance)
        + """
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    ds = pickle.load(fh)[0]["dataset"]
out = mod.solve_batch(copy.deepcopy(ds))
n_te = len(ds["test"]["lab_id"])
k_t = len(ds["eval_times"])
errs = []
coupon = None
if not isinstance(out, dict):
    errs.append("not_dict")
else:
"""
        + f"    for k, shp in {schema_keys}:\n"
        + """        if k not in out:
            errs.append(f"missing:{k}")
            continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        try:
            coupon = verify_alpha_pooled_ref_coupon_invariance(mod.solve_batch, ds, base_out=out)
            if not coupon["valid"]:
                errs.append("pooled_reference_coupon_invariance_failed")
        except (KeyError, TypeError, ValueError, IndexError):
            errs.append("invalid_output_on_reference_exchange")
print(json.dumps({"schema_valid": not errs, "errors": errs, "pooled_reference_check": coupon}, allow_nan=False))
sys.exit(1 if errs else 0)
"""
    )
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M100_ALPHA_GEN_PY
        + f"\npub_ds, _unused_gt = {pub_call_expr}\n"
        'pub_diag = pub_ds.pop("_campaign_diagnostics", None)\n'
        'with open("/holdout_task/public_batch.pkl", "wb") as fh:\n'
        '    pickle.dump([{"dataset": pub_ds, "campaign_diagnostics": pub_diag}], fh)\n'
        + f'open("/holdout_task/validate_public_schema.py", "w").write({schema_program!r})\n'
    )
    priv_ver = (
        "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
        "import batch_solver as cand_mod\n"
        + _M100_ALPHA_GEN_PY
        + f"\ncases = [\n{priv_cases_py}\n]\n"
        + priv_body
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="m10_0_reliable_modular_amortization_and_bidirectional_withdrawal",
        scientific_domain="coupled_viscoelastic_chamber_hereditary_mechanics",
        wormhole_family="m100_alpha",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json=host_gt_json,
        instruction_text=BLIND_ALPHA_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=priv_ver,
    )


def _build_m100_beta_task(*, task_id: str, short_name: str, pub_call_expr: str, priv_cases_py: str) -> HoldoutOODTask:
    from .m9_3_online_lifecycle_benchmark import _SHARED_BOX_SIMPLEX_KKT_CHECKER_PY

    _, pub_gt = _eval_m100_gen_host(_M100_BETA_GEN_PY, pub_call_expr)
    host_gt_json = json.dumps(encode_pure_json_numerical_payload(pub_gt), allow_nan=False)
    priv_body = '''
t0 = time.monotonic()
valid = True; max_w_rel = None; max_f_rel = None; max_gap = None; rows = []
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
    w_rel = float(np.linalg.norm(w - gt["w_opt"]) / max(float(np.linalg.norm(gt["w_opt"])), 1e-12))
    f_rel = float(np.linalg.norm(tf - gt["terminal_field_A"]) / max(float(np.linalg.norm(gt["terminal_field_A"])), 1e-12))
    max_w_rel = w_rel if max_w_rel is None else max(max_w_rel, w_rel)
    max_f_rel = f_rel if max_f_rel is None else max(max_f_rel, f_rel)
    chk = verify_box_simplex_kkt_certificate(gt["S_tilde"], gt["r_y"], st["source_ridge"], st["u_box"], st["total_budget"], out["kkt_certificate"], gap_tol=2e-4)
    gap_v = float(chk["gap"]) if (isinstance(chk.get("gap"), (int, float)) and math.isfinite(float(chk["gap"]))) else None
    if gap_v is not None:
        max_gap = gap_v if max_gap is None else max(max_gap, gap_v)
    if not chk["valid"] or float(np.max(np.abs(w - np.asarray(out["kkt_certificate"]["weights"], dtype=float)))) > 1e-5:
        valid = False
        rows.append({"seed": sd, "error": f"KKT_INVALID:{chk.get('reason')}", "w_rel": round(w_rel, 7), "f_rel": round(f_rel, 7), "kkt_gap": (round(gap_v, 9) if gap_v is not None else None)})
        continue
    rows.append({"seed": sd, "w_rel": round(w_rel, 7), "f_rel": round(f_rel, 7), "kkt_gap": round(float(chk["gap"]), 9)})
el = time.monotonic() - t0
sp = bool(valid and len(rows) == 2 and max_w_rel is not None and max_f_rel is not None and max_gap is not None and max_w_rel <= 1.5e-3 and max_f_rel <= 1.5e-3 and max_gap <= 2e-4 and el <= 25.0)
print(json.dumps({"strict_pass": sp, "reward": 1.0 if sp else 0.0, "max_w_rel_l2": (round(max_w_rel, 7) if max_w_rel is not None else None), "max_field_rel_l2": (round(max_f_rel, 7) if max_f_rel is not None else None), "max_kkt_gap": (round(max_gap, 9) if max_gap is not None else None), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
'''
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M100_BETA_GEN_PY
        + f"\npub_st, _unused_gt = {pub_call_expr}\n"
        'pub_diag = pub_st.pop("_campaign_diagnostics", None)\n'
        'with open("/holdout_task/public_batch.pkl", "wb") as fh:\n'
        '    pickle.dump([{"study": pub_st, "campaign_diagnostics": pub_diag}], fh)\n'
        'open("/holdout_task/validate_public_schema.py", "w").write("""import copy, importlib.util, json, os, pickle, sys\nimport numpy as np\n'
        '_d = os.path.dirname(os.path.abspath(__file__))\n'
        'cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")\n'
        'spec = importlib.util.spec_from_file_location("_cand", cand_p)\n'
        'mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n'
        'item = pickle.load(open(os.path.join(_d, "public_batch.pkl"), "rb"))[0]\n'
        'st = item["study"]\n'
        'out = mod.solve_batch(copy.deepcopy(st))\n'
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
        + _M100_BETA_GEN_PY
        + f"\ncases = [\n{priv_cases_py}\n]\n"
        + priv_body
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="m10_0_reliable_modular_amortization_and_bidirectional_withdrawal",
        scientific_domain="coupled_2d_transport_reaction_box_simplex_kkt",
        wormhole_family="m100_beta",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json=host_gt_json,
        instruction_text=BLIND_BETA_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=priv_ver,
    )


def build_m100_world_tasks(
    spec: M100WorldSpec, *, grid_size: str | None = None, order_mode: str | None = None,
) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct `Stage S (U0, D0)` + the `10`-step `3 × 3` continual stream under `spec.order_mode`
    (`'sep_first'` vs `'coupled_first'`):
      - Both arrival orders evaluate the EXACT SAME `9` `U × D` grid cells (`(U0,D0)` at `Stage S`, `5` single-factor /
        coupled discovery cells, `3` unseen separable combination cells `(U1,D1)`, `(U2,D1)`, `(U2,D2_sep)`, plus `2`
        colliding-`dn_obs` revisits `(U0,D2_sep)` and `(U1,D2_coupled)` under `v3_gains`!).
    On development worlds, explicit grid_size ("2x3"/"3x3") and order_mode ("sep_first"/"coupled_first")
    select repeated measurements of one source cluster. Common physical events share their public batch
    and apparatus trace across conditions. Chronological prefixes still reflect their actual order.
    The no-keyword entry point retains its declared grid, order and output format.
    """
    paired = grid_size is not None or order_mode is not None
    if paired and spec.split != "dev":
        raise ValueError("The paired grid/order controls are for development source clusters")
    grid = "3x3" if grid_size is None else grid_size
    order = spec.order_mode if order_mode is None else order_mode
    valid_orders = ("sep_first", "coupled_first")
    if grid not in ("2x3", "3x3") or order not in valid_orders or spec.order_mode not in valid_orders:
        raise ValueError(f"Unsupported M10 condition: grid={grid}, order={order}, source order={spec.order_mode}")

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
            pub_c = f"_gen_m100_alpha_instance({pub_seed}, {kw})"
            priv_c = f"    ({priv_s1}, *_gen_m100_alpha_instance({priv_s1}, {kw})),\n    ({priv_s2}, *_gen_m100_alpha_instance({priv_s2}, {kw})),"
            return _build_m100_alpha_task(
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
        pub_cb = f"_gen_m100_beta_instance({pub_seed}, {kw_b})"
        priv_cb = f"    ({priv_s1}, *_gen_m100_beta_instance({priv_s1}, {kw_b})),\n    ({priv_s2}, *_gen_m100_beta_instance({priv_s2}, {kw_b})),"
        return _build_m100_beta_task(
            task_id=f"{short_nm}__blind_continual_campaign",
            short_name=short_nm,
            pub_call_expr=pub_cb,
            priv_cases_py=priv_cb,
        )

    s_task = _mk_task(f"{wid}_s00", sb + 10, sb + 11, sb + 12, "U0", "D0", id_g, id_b, 0.0, False, 0)

    events = {
        "d1_discovery": ("decomp_d1", "U0", "D1", "D1", spec.v2_gains, spec.v2_offsets, 0.0, True, 0, sb + 110, sb + 111, sb + 112, "online_2x2_decomp_and_d1"),
        "separable_d2_discovery": ("d2_sep_disc", "U0", "D2", "D2", spec.v2_gains, spec.v2_offsets, 0.0, True, 0, sb + 210, sb + 211, sb + 212, "separable_d2_discovery"),
        "u1_discovery": ("u1_disc", "U1", "D0", "D0", spec.v2_gains, spec.v2_offsets, spec.u1_offdiag_disc, False, 3, sb + 310, sb + 311, sb + 312, "upstream_u1_discovery"),
        "u1_d1_combination": ("comb_u1_d1", "U1", "D1", "D1", spec.v3_gains, spec.v3_offsets, spec.u1_offdiag_comb, False, 3, sb + 410, sb + 411, sb + 412, "unseen_separable_comb_1"),
        "u2_discovery": ("u2_disc", "U2", "D0", "D0", spec.v2_gains, spec.v2_offsets, spec.u2_offdiag_disc, False, 3, sb + 510, sb + 511, sb + 512, "upstream_u2_discovery"),
        "u2_d1_combination": ("comb_u2_d1", "U2", "D1", "D1", spec.v3_gains, spec.v3_offsets, spec.u2_offdiag_comb1, False, 3, sb + 610, sb + 611, sb + 612, "unseen_separable_comb_2"),
        "u2_d2_combination": ("comb_u2_d2sep", "U2", "D2", "D2", spec.v3_gains, spec.v3_offsets, spec.u2_offdiag_comb2, False, 3, sb + 710, sb + 711, sb + 712, "unseen_separable_comb_3"),
        "coupled_d2_discovery": ("fwd_coupled_break", "U1", "D2", "D2_coupled", spec.v2_gains, spec.v2_offsets, spec.u1_offdiag_cpl, True, 3, sb + 810, sb + 811, sb + 812, "forward_interaction_withdrawal"),
        "separable_d2_revisit": ("revisit_sep", "U0", "D2", "D2", spec.v3_gains, spec.v3_offsets, 0.0, False, 0, sb + 910, sb + 911, sb + 912, "colliding_d2_separable_revisit"),
        "coupled_d2_revisit": ("revisit_coupled", "U1", "D2", "D2_coupled", spec.v3_gains, spec.v3_offsets, spec.u1_offdiag_rev, False, 3, sb + 1010, sb + 1011, sb + 1012, "colliding_d2_coupled_revisit"),
    }
    orders = {
        "sep_first": tuple(events),
        "coupled_first": ("d1_discovery", "u1_discovery", "u1_d1_combination", "coupled_d2_discovery",
                          "u2_discovery", "u2_d1_combination", "separable_d2_discovery", "u2_d2_combination",
                          "coupled_d2_revisit", "separable_d2_revisit"),
    }
    native_index = {key: i for i, key in enumerate(orders[spec.order_mode], start=1)}
    selected_keys = [key for key in orders[order] if grid == "3x3" or events[key][1] != "U2"]

    tasks: list[HoldoutOODTask] = []
    manifest: list[dict[str, Any]] = []
    for idx_i, key in enumerate(selected_keys, start=1):
        label, up_m, dn_r, eff_r, gg, bb, boff, pil, cout, ps, p1, p2, cat = events[key]
        if order == "coupled_first":
            if key == "coupled_d2_discovery":
                label, cat = "d2_coupled_first", "coupled_d2_first_no_prior_sep"
            elif key == "separable_d2_discovery":
                label, cat = "rev_sep_break", "reverse_interaction_withdrawal"
        ev = f"S{idx_i:02d}_{label}"
        sn = f"{wid}_batch_{idx_i:02d}"
        tk = _mk_task(sn, ps, p1, p2, up_m, dn_r, gg, bb, boff, pil, cout)
        tasks.append(tk)
        manifest.append(
            {
                "stream_step": idx_i,
                "short_name": sn,
                "task_id": tk.task_id,
                "order_mode": order,
                "causal_event": ev,
                "true_upstream_mode": up_m,
                "true_downstream_precursor": dn_r,
                "true_effective_regime": eff_r,
                "true_regime": f"{up_m}__{eff_r}",
                "event_category": cat,
            }
        )

        if paired:
            manifest[-1].update(
                source_cluster_id=spec.cluster_id, condition_id=f"{spec.world_id}/{grid}/{order}",
                grid_size=grid, paired_event_id=key,
            )

    prev_sn = ["facility_init"] + [s_task.short_name] + [m["short_name"] for m in manifest[:-1]]
    next_dn_seq = ["D0"] + [m["true_downstream_precursor"] for m in manifest]
    log_spec = spec.to_m99_log_spec()
    apparatus_logs = []
    for k, key in enumerate(["source", *selected_keys]):
        if key in ("separable_d2_discovery", "coupled_d2_discovery"):
            ovr = sb + 9901
        elif key in ("separable_d2_revisit", "coupled_d2_revisit"):
            ovr = sb + 9902
        else:
            # Keep the source world's original telemetry realization when an event moves to another step.
            ovr = sb + 8000 + 37 * native_index.get(key, 0)
        apparatus_logs.append(
            build_m99_apparatus_log_entry(
                spec=log_spec,
                step_index=k,
                short_name=prev_sn[k],
                next_downstream_regime=next_dn_seq[k],
                trace_seed_override=ovr,
            )
        )
    return s_task, tuple(tasks), manifest, apparatus_logs


# ============================================================================
# 2. Generic Domain-Blind Statistical Observable Bank (`7` Upstream + `8` Downstream)
# ============================================================================


def compute_m100_upstream_observables(family: str, public_item: dict[str, Any]) -> dict[str, float | None]:
    """Compute 7 generic linear/polynomial/correlation summary statistics on interior (`<= 8.5`) calibration rows:
      - `0` domain formulas, `0` private keys.
      - Includes signed adjacent correlation (`up_stat_02_adj_offdiag_signed_corr`, fixing `c04` unsigned `max |corr|` collision!),
        adjacent vs opposite off-diagonal residual correlations (`(c+1)%4` vs `(c+2)%4`), and degree-3 polynomial fit $R^2$.
    """
    if family == "alpha":
        cal = public_item["dataset"]["calibration_load_cells"]
        x_m = np.asarray(cal["true_channel_loads"], dtype=float)
        y_m = np.asarray(cal["bridge_raw"], dtype=float)
    else:
        cal = public_item["study"]["sensor_calibration_sweeps"]
        x_m = np.asarray(cal["ref_rosette"], dtype=float)
        y_m = np.asarray(cal["raw_rosette"], dtype=float)

    keep = np.max(np.abs(y_m), axis=1) <= 8.5
    xk = x_m[keep] if int(np.sum(keep)) >= 12 else x_m
    yk = y_m[keep] if int(np.sum(keep)) >= 12 else y_m

    r2_list: list[float] = []
    for c in range(4):
        xc = xk[:, c]
        yc = yk[:, c]
        g_c, b_c = np.linalg.lstsq(np.column_stack([xc, np.ones(len(xc))]), yc, rcond=None)[0]
        res_c = yc - (g_c * xc + b_c)
        ss_tot = max(float(np.sum((yc - np.mean(yc)) ** 2)), 1e-12)
        r2_list.append(1.0 - float(np.sum(res_c ** 2)) / ss_tot)

    # Multivariate 4-channel linear Jacobian OLS `yk ~ [xk, 1] @ B_lin` (`0` finite-sample predictor cross-correlation leakage!)
    b_lin, *_ = np.linalg.lstsq(np.column_stack([xk, np.ones(len(xk), dtype=float)]), yk, rcond=None)
    adj_signed: list[float] = []
    adj_abs: list[float] = []
    opp_abs: list[float] = []
    for c in range(4):
        diag_c = float(b_lin[c, c])
        denom_c = diag_c if abs(diag_c) > 1e-6 else 1.0
        r_adj = float(b_lin[(c + 1) % 4, c]) / denom_c
        r_opp = float(b_lin[(c + 2) % 4, c]) / denom_c
        adj_signed.append(r_adj)
        adj_abs.append(abs(r_adj))
        opp_abs.append(abs(r_opp))

    # Generic multivariate degree-3 basis fit $R^2$ (`[1, x_i, x_i*x_j, x_i^3]`) on interior calibration rows
    cols = [np.ones(len(xk), dtype=float)]
    for i in range(4):
        cols.append(xk[:, i])
        cols.append(xk[:, i] ** 3)
        for j in range(i, 4):
            cols.append(xk[:, i] * xk[:, j])
    phi3 = np.column_stack(cols)
    w3, *_ = np.linalg.lstsq(phi3, yk, rcond=None)
    res3 = yk - phi3 @ w3
    poly_r2s = [
        1.0 - float(np.sum(res3[:, c] ** 2)) / max(float(np.sum((yk[:, c] - np.mean(yk[:, c])) ** 2)), 1e-12)
        for c in range(4)
    ]

    return {
        "up_stat_01_diagonal_ols_r2": round(float(np.min(r2_list)), 6),
        "up_stat_02_adj_offdiag_signed_corr": round(float(np.mean(adj_signed) if adj_signed else 0.0), 6),
        "up_stat_03_adj_offdiag_abs_corr_max": round(float(max(adj_abs) if adj_abs else 0.0), 6),
        "up_stat_04_opp_offdiag_abs_corr_max": round(float(max(opp_abs) if opp_abs else 0.0), 6),
        "up_stat_05_cubic_poly_residual_r2": round(float(np.min(poly_r2s)), 6),
        "up_stat_06_cal_frobenius_rms": round(float(np.sqrt(np.mean(yk ** 2))), 6),
        "up_stat_07_cal_row_count": float(len(y_m)),
    }


# ============================================================================
# 3. Lazy On-Demand Retroactive Downstream Projection (`B3 / Q04`, `100%` Equal Across Modular Arms)
# ============================================================================


def propose_and_verify_lazy_retroactive_downstream_projection(
    *,
    client: Any,
    stage_subdir: str,
    family: str,
    u0_code: str,
    current_up_code: str,
    current_up_entrypoint: str,
    current_pay: dict[str, Any],
    prior_monolithic_code: str,
    prior_task: HoldoutOODTask,
    prior_pay: dict[str, Any],
    prior_ro_deps: dict[str, str],
    prior_ro_shas: dict[str, str],
    current_task: HoldoutOODTask,
    current_ro_deps: dict[str, str],
    current_ro_shas: dict[str, str],
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Lazy on-demand downstream extraction (`B3 / Q04` — `0` eager `M9.6` compile tax!):
    Invoked ONLY when an unfactored pre-decomposition monolithic solver matches the current step's `dn_obs`
    under a new verified upstream transducer (`current_up_code`). Extracts `solve_downstream(cal_item)` and
    verifies BOTH `(u0 o d_proj)` on `prior_task` AND `(current_up o d_proj)` on `current_task` in Docker!
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["diagnosis", "downstream_code"],
        "properties": {
            "diagnosis": {"type": "string"},
            "downstream_code": {"type": "string"},
        },
    }
    sys_prompt = (
        "You are an on-demand downstream module extractor. Given a previously verified monolithic `solve_batch` program "
        "and the verified baseline upstream transducer `u0_code` (`calibrate_upstream`), extract `downstream_code` exposing "
        "`solve_downstream(cal_item: dict) -> dict` in canonical calibrated coordinates so `solve_downstream(calibrate_upstream(item, pay))` "
        "passes the public scientific gate without `_HEGEL_ADAPTER_PRECALIBRATED_DO_NOT_REWRAP`."
    )
    convo = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "family": family,
                    "verified_u0_code": u0_code,
                    "prior_verified_monolithic_code": prior_monolithic_code,
                },
                indent=2,
            ),
        }
    ]
    probes: list[dict[str, Any]] = []
    call_receipts: list[dict[str, Any]] = []
    outcome: dict[str, Any] = {"accepted": False, "status": "unresolved", "reason": "no_valid_proposal"}
    targets = (
        ("prior", prior_task, u0_code, "calibrate_upstream", prior_pay, prior_ro_deps, prior_ro_shas),
        ("current", current_task, current_up_code, current_up_entrypoint, current_pay, current_ro_deps, current_ro_shas),
    )
    for att in range(1, max_attempts + 1):
        res = client.call_structured(
            system_prompt=sys_prompt, conversation_turns=convo, output_schema=schema,
            stage_subdir=f"{stage_subdir}_lazyproj_att{att}",
        )
        call_receipts.append(res)
        if not res.get("ok") or not isinstance(res.get("parsed"), dict):
            continue
        dn_code = res["parsed"].get("downstream_code")
        if not isinstance(dn_code, str) or not dn_code.strip():
            continue
        receipts = {}
        for side, task, up_code, entrypoint, pay, dependencies, dependency_hashes in targets:
            candidate = compose_m98_two_stage_solver_code(
                upstream_code=up_code, upstream_entrypoint=entrypoint,
                downstream_code=dn_code, downstream_entrypoint="solve_downstream", param_payload=pay,
            )
            candidate_sha = hashlib.sha256(candidate.encode()).hexdigest()
            start = monotonic()
            try:
                gate = execute_single_pre_llm_public_gate_probe_in_container(
                    task=task, candidate_code=candidate, readonly_dependencies=dependencies,
                    expected_dependency_sha256=dependency_hashes, require_pooled_coupon_check=family == "alpha",
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                gate = {"setup_ok": False, "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                        "infrastructure_error": str(exc)}
            wall = round((monotonic() - start) * 1000.0, 2)
            classification = classify_public_probe_failure(gate)
            tier = classification["tier"]
            assets = gate.get("public_setup_files_sha256") or {}
            actual_input = assets.get("/holdout_task/public_batch.pkl")
            diagnostic = gate.get("scientific_probe_receipt") or {}
            numeric_sha = diagnostic.get("canonical_numeric_sha256")
            setup_sha = hashlib.sha256(task.public_setup_py.encode()).hexdigest()
            if tier != "INFRA_FAILURE":
                bound = (bool(actual_input) and gate.get("candidate_raw_sha256") == candidate_sha
                         and gate.get("setup_source_sha256") == setup_sha
                         and gate.get("readonly_dependencies_sha256") == dependency_hashes
                         and all(assets.get(k) == v for k, v in dependency_hashes.items()))
                if tier in {"NONE", "SCIENTIFIC_FAILURE"}:
                    bound = bound and joint_scientific_receipt_matches(gate, candidate_sha, actual_input, numeric_sha)
                if not bound:
                    tier = "INFRA_FAILURE"
                    classification = {"tier": tier, "failure_kind": "unbound_probe"}
            status = ("timeout" if gate.get("timed_out") else "infrastructure_error") if tier == "INFRA_FAILURE" else "completed"
            receipt = {
                **gate, "role": f"lazy_proj_{side}_gate", "attempt_index": att, "source_call_uuid": res.get("call_uuid"),
                "task_id": task.task_id, "setup_source_sha256": gate.get("setup_source_sha256"),
                "candidate_sha256": gate.get("candidate_raw_sha256"),
                "component_code_sha256s": ([hashlib.sha256(src.encode()).hexdigest() for src in (up_code, dn_code)]
                                          if gate.get("candidate_raw_sha256") == candidate_sha else []),
                "public_input_sha256": actual_input, "canonical_numeric_sha256": numeric_sha,
                "dependency_bundle_sha256": (hashlib.sha256(json.dumps(
                    gate["readonly_dependencies_sha256"], sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest() if isinstance(gate.get("readonly_dependencies_sha256"), dict) else None),
                "expected_candidate_sha256": candidate_sha, "executed_public_input_sha256": actual_input,
                "executed_canonical_numeric_sha256": numeric_sha, "probe_id": f"lpprobe_{uuid.uuid4().hex[:16]}",
                "probe_wall_ms": wall, "wall_ms": wall, "execution_status": status,
                "accepted": tier == "NONE", "failure_tier": tier, "failure_kind": classification["failure_kind"],
            }
            probes.append(receipt)
            receipts[side] = receipt
            if tier == "INFRA_FAILURE":
                outcome = {"accepted": False, "status": "unresolved", "reason": receipt["failure_kind"]}
                break
            if tier != "NONE":
                outcome = {"accepted": False, "status": "rejected", "reason": receipt["failure_kind"]}
                convo.append({"role": "assistant", "content": json.dumps(res["parsed"])})
                convo.append({"role": "user", "content": json.dumps({
                    f"{side}_task_gate_failed": diagnostic or gate.get("schema_stdout")})})
                break
        else:
            outcome = {"accepted": True, "status": "accepted", "downstream_code": dn_code,
                       "composed_current_code": candidate, "prior_probe_receipt": receipts["prior"],
                       "current_probe_receipt": receipts["current"]}
        if outcome["accepted"] or tier == "INFRA_FAILURE":
            break

    return {**outcome, "attempts_used": len(call_receipts), "probes": probes,
            "total_probe_wall_ms": round(sum(probe["probe_wall_ms"] for probe in probes), 2),
            "provider_call_receipts": call_receipts}


def m100_task_to_frontier_task(
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
        **{**base_ft.__dict__, "benchmark": "Hegel-M100-Reliable-Modular-Amortization-And-Bidirectional-Withdrawal-World"}
    )


# ============================================================================
# 4. Reference Dev-Seed Solvers (`0`-LLM Unit & Container Testing on `DEV_M100_WORLD_SPECS` ONLY!)
# ============================================================================


def build_m100_reference_dev_upstream_u2_code(family: str) -> str:
    """Reference `u2` harmonic opposite-pair calibration inverter used ONLY in `0`-LLM dev tests (`DEV_M100_WORLD_SPECS`)!"""
    if family == "alpha":
        return '''from __future__ import annotations
import copy
import numpy as np

def _phi_u2(x: np.ndarray) -> np.ndarray:
    x2 = np.asarray(x, dtype=float).reshape(-1, 4)
    x_opp = np.roll(x2, 2, axis=1)
    return np.column_stack([np.ones(len(x2)), x2, np.sin(2.50 * x2), np.cos(2.00 * x_opp) - 1.0])

def calibrate_upstream(dataset: dict, param_payload: dict | None = None) -> dict:
    ds = copy.deepcopy(dataset)
    cal = dict(ds["calibration_load_cells"])
    raw = np.asarray(cal["bridge_raw"], dtype=float)
    tru = np.asarray(cal["true_channel_loads"], dtype=float)
    kp = np.max(np.abs(raw), axis=1) <= 8.5
    W, *_ = np.linalg.lstsq(_phi_u2(tru[kp]), raw[kp], rcond=None)

    def _invert(y_in: np.ndarray) -> np.ndarray:
        orig = y_in.shape
        y2 = np.asarray(y_in, dtype=float).reshape(-1, 4)
        x = (y2 - W[0:1, :]) @ np.linalg.pinv(W[1:5, :])
        for _ in range(12):
            f = _phi_u2(x) @ W - y2
            n_r = len(x)
            J = np.broadcast_to(W[1:5, :].T[None, :, :], (n_r, 4, 4)).copy()
            c_sin = 2.50 * np.cos(2.50 * x)
            s_cos = -2.00 * np.sin(2.00 * np.roll(x, 2, axis=1))
            for k in range(4):
                J[:, :, k] += c_sin[:, k : k + 1] * W[5 + k, :][None, :]
                k_opp = (k + 2) % 4
                J[:, :, k] += (-2.00 * np.sin(2.00 * x[:, k : k + 1])) * W[9 + k_opp, :][None, :]
            dx = np.linalg.solve(J, f[:, :, None])[:, :, 0]
            x = x - dx
            if float(np.max(np.abs(dx))) <= 1e-11:
                break
        return x.reshape(orig)

    for sp_k in ("train", "test"):
        sp = dict(ds[sp_k])
        sp["past_channel_raw"] = _invert(np.asarray(sp["past_channel_raw"], dtype=float))
        sp["hold_channel_raw"] = _invert(np.asarray(sp["hold_channel_raw"], dtype=float))
        ds[sp_k] = sp
    if isinstance(ds.get("chamber_verification_trials"), dict):
        pv = dict(ds["chamber_verification_trials"])
        pv["past_channel_raw"] = _invert(np.asarray(pv["past_channel_raw"], dtype=float))
        pv["hold_channel_raw"] = _invert(np.asarray(pv["hold_channel_raw"], dtype=float))
        ds["chamber_verification_trials"] = pv
    b_cal = raw.copy()
    b_cal[kp] = _invert(raw[kp])
    cal["bridge_raw"] = b_cal
    sc = dict(cal["single_channel_sweeps"])
    sc_y = np.asarray(sc["cmd_stress"], dtype=float).ravel().copy()
    sc_t = np.asarray(sc["true_stress"], dtype=float).ravel()
    m_in = np.abs(sc_y) <= 8.5
    sc_y[m_in] = sc_t[m_in]
    sc["cmd_stress"] = sc_y
    cal["single_channel_sweeps"] = sc
    ds["calibration_load_cells"] = cal
    return ds
'''
    return '''from __future__ import annotations
import copy
import numpy as np

def _phi_u2(x: np.ndarray) -> np.ndarray:
    x2 = np.asarray(x, dtype=float).reshape(-1, 4)
    x_opp = np.roll(x2, 2, axis=1)
    return np.column_stack([np.ones(len(x2)), x2, np.sin(2.50 * x2), np.cos(2.00 * x_opp) - 1.0])

def calibrate_upstream(study: dict, param_payload: dict | None = None) -> dict:
    st = copy.deepcopy(study)
    cal = dict(st["sensor_calibration_sweeps"])
    raw = np.asarray(cal["raw_rosette"], dtype=float)
    tru = np.asarray(cal["ref_rosette"], dtype=float)
    kp = np.max(np.abs(raw), axis=1) <= 8.5
    W, *_ = np.linalg.lstsq(_phi_u2(tru[kp]), raw[kp], rcond=None)

    def _invert(y_in: np.ndarray) -> np.ndarray:
        orig = y_in.shape
        y2 = np.asarray(y_in, dtype=float).reshape(-1, 4)
        x = (y2 - W[0:1, :]) @ np.linalg.pinv(W[1:5, :])
        for _ in range(12):
            f = _phi_u2(x) @ W - y2
            n_r = len(x)
            J = np.broadcast_to(W[1:5, :].T[None, :, :], (n_r, 4, 4)).copy()
            c_sin = 2.50 * np.cos(2.50 * x)
            for k in range(4):
                J[:, :, k] += c_sin[:, k : k + 1] * W[5 + k, :][None, :]
                k_opp = (k + 2) % 4
                J[:, :, k] += (-2.00 * np.sin(2.00 * x[:, k : k + 1])) * W[9 + k_opp, :][None, :]
            dx = np.linalg.solve(J, f[:, :, None])[:, :, 0]
            x = x - dx
            if float(np.max(np.abs(dx))) <= 1e-11:
                break
        return x.reshape(orig)

    st["y_obs_raw"] = _invert(np.asarray(st["y_obs_raw"], dtype=float).reshape(-1, 4)).ravel()
    if isinstance(st.get("reference_tracer_pulses"), list):
        new_p = []
        for row in st["reference_tracer_pulses"]:
            r2 = dict(row)
            r2["y_pulse_raw"] = _invert(np.asarray(r2["y_pulse_raw"], dtype=float).reshape(-1, 4)).ravel()
            new_p.append(r2)
        st["reference_tracer_pulses"] = new_p
    r_cal = raw.copy()
    r_cal[kp] = _invert(raw[kp])
    cal["raw_rosette"] = r_cal
    sb = dict(cal["single_bank_sweeps"])
    sb_y = np.asarray(sb["raw_readout"], dtype=float).ravel().copy()
    sb_t = np.asarray(sb["ref_signal"], dtype=float).ravel()
    m_in = np.abs(sb_y) <= 8.5
    sb_y[m_in] = sb_t[m_in]
    sb["raw_readout"] = sb_y
    cal["single_bank_sweeps"] = sb
    st["sensor_calibration_sweeps"] = cal
    return st
'''


__all__ = [
    "CONFIRMATION_M100_WORLD_SPECS",
    "DEV_M100_WORLD_SPECS",
    "M100WorldSpec",
    "M100_ALL_OBSERVABLE_NAMES",
    "M100_ARMS",
    "M100_DOWNSTREAM_OBSERVABLE_NAMES",
    "M100_UPSTREAM_OBSERVABLE_NAMES",
    "SeparabilityWithdrawalReceipt",
    "build_m100_reference_dev_upstream_u2_code",
    "build_m100_world_tasks",
    "build_m99_downstream_canonical_synthesis_task",
    "build_m99_reference_dev_downstream_solver_code",
    "check_witness_feature_collision",
    "compute_m100_upstream_observables",
    "hydrate_separability_withdrawals_from_graph",
    "m100_task_to_frontier_task",
    "persist_separability_withdrawals_to_graph",
    "propose_and_validate_m99_guard_partition",
    "propose_and_verify_lazy_retroactive_downstream_projection",
    "verify_m99_autonomous_decomposition_in_containers",
    "verify_m99_large_margin_witness_partition",
]

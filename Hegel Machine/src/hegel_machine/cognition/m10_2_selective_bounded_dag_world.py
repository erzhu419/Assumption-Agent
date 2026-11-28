"""Hegel `M10.2`: Evidence-Gated Selective Bounded Computational DAG Discovery & Maintenance (`H = 16`, De-Scaffolded, Equal-Probe-Cap).

Closes ALL design & attribution gaps identified in `Hegel_M10_1_Audit` (`U01–U12`, `01_Implementation_and_Experiment_Plan_ZH.md` §六–七):
  1. **Zero Prompt Split-Scaffolding & Neutral Observable Names (`U06 / §七`)**:
     - Observables are named strictly `stat_00..stat_17` (ZERO `pa/pb/pc/lin_r2/tel_ch` role hints!).
     - Prompts provide ZERO `panel_a -> panel_b -> panel_c['v1']/['v2']` decomposition recipes or observable-to-action
       mapping rules; structure proposals rely purely on generic Python dataflow/AST inspection of `source_code` and
       statistical shift patterns across `stat_00..stat_17`.
  2. **100% Equal Pre-LLM Probe Budget Across All 3 Arms (`MAX_PRE_LLM_PROBES_PER_STEP = 4`, `E1`)**:
     - `Flat_MonolithicArchive_Online`, `Eager_BoundedDAG_Online`, and `Hegel_SelectiveBoundedDAG_Online` share the
       EXACT same `max_pre_llm_probes = 4` container CV budget, `M10.1.1` feasible-domain router, and borderline-CV
       (`0.026 < y_mae <= 0.038`) 2-candidate public comparison rule!
  3. **Zero-LLM Evidence-Based Structural Investment & Maintenance Gate (`evaluate_m102_selective_investment_gate`, `E2`)**:
     - Single-factor causal difference between `Eager_BoundedDAG_Online` and `Hegel_SelectiveBoundedDAG_Online`:
       evaluates (a) Python AST multi-subtable numerical stage separability of `source_code`, (b) localized vs global
       shift ratio across `stat_00..stat_13` groups, and (c) remaining-horizon payback & consecutive failed-surgery
       strike-out before spending tokens on structure proposal/maintenance!
  4. **Pre-Registered `H = 16` Continual Horizon (`H = 4, 8, 12` Nested Checkpoints, `H = 16` Single Primary Endpoint, `E4.3`)**:
     - Wave 1 (`S01..S08`) + Wave 2 (`S09..S16`) introducing genuine Wave-2 operator regimes (`A2`, `B2`, `L2`, `R2`, `M2..M4`),
       unseen 3-way/4-way recombinations, periodic cross-stage coupling shifts, and colliding revisits (ZERO duplicate padding!).
     - Dedicated `UNIT_TEST_M102_WORLD_SPECS` (`7201000..`), `DEV_M102_WORLD_SPECS` (`1141000..`), and
       `CONFIRMATION_M102_WORLD_SPECS` (`1151000..`) with strict seed isolation (`U01`).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .m7_holdout_ood_blind_benchmark import HoldoutOODTask
from .m9_4_1_shared_verification_kernel import SHARED_KERNEL_VERSION
from .m10_1_unknown_cut_dag_discovery_world import (
    BLIND_M101_INSTRUCTION_TEXT,
    M101_MAX_ACTIVE_NODES,
    M101_MAX_DEPTH,
    M101_MAX_FANOUT,
    M101_OBSERVABLE_NAMES,
    M101_OUTPUT_KEYS,
    _M101_WORLD_GEN_PY,
    compute_m101_neutral_observables,
)

M102_ARMS: tuple[str, ...] = (
    "Flat_MonolithicArchive_Online",
    "Eager_BoundedDAG_Online",
    "Hegel_SelectiveBoundedDAG_Online",
)

M102_HORIZON_STEPS = 16
M102_NESTED_CHECKPOINTS: tuple[int, ...] = (4, 8, 12, 16)
M102_MAX_PRE_LLM_PROBES_PER_STEP = 4
M102_BORDERLINE_CV_COMPARE_THRESHOLD = 0.026

# Strictly role-blind observable names (`stat_00..stat_17`, zero `pa/pb/pc` substrings!)
M102_OBSERVABLE_NAMES: tuple[str, ...] = tuple(f"stat_{i:02d}" for i in range(18))

M102_DAG_LEDGER_FOREST_KEY = "__m10_2_selective_bounded_dag_state__"


def compute_m102_blind_observables(study: dict[str, Any]) -> dict[str, float | None]:
    """Compute role-blind `stat_00..stat_17` numerical summary vector (`0` stage-name hints in keys!)."""
    raw = compute_m101_neutral_observables(study)
    return {
        f"stat_{idx:02d}": raw[old_k]
        for idx, old_k in enumerate(M101_OBSERVABLE_NAMES)
    }


# ============================================================================
# 1. Extended Wave-2 (`A2, B2, L2, R2, M2..M4`) Physics Generator for `H = 16`
# ============================================================================

_M102_WORLD_GEN_EXTENSION_PY = (
    _M101_WORLD_GEN_PY
    + r"""
_ORIG_M101_APPLY_PC_BRANCH = _m101_apply_pc_branch
def _m101_apply_pc_branch(z: np.ndarray, mode: str, W_mat: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    z2 = np.asarray(z, dtype=float)
    if mode in ("L2", "H2"):
        # Smooth saturation + cyclic channel harmonic coupling (20 linear params in W_mat, b_vec; non-nested with L0/L1!)
        act = np.tanh(1.75 * z2) + 0.30 * np.sin(2.1 * np.roll(z2, 1, axis=1))
        return 1.25 * (act @ W_mat.T) + b_vec[None, :]
    if mode == "R2":
        # Smooth algebraic normalization + cyclic bilinear bridge (non-nested with R0 softplus and R1 even cosine!)
        act = z2 / np.sqrt(1.0 + 0.85 * (z2 ** 2)) + 0.28 * (z2 * np.roll(z2, 1, axis=1))
        return 1.25 * (act @ W_mat.T) + b_vec[None, :]
    return _ORIG_M101_APPLY_PC_BRANCH(z2, mode, W_mat, b_vec)


def _gen_m102_study(
    *,
    world_seed: int,
    batch_seed: int,
    topology_kind: str,
    stage_cfg: dict,
    telemetry_vec: tuple[float, float, float, float],
):
    tk_map = {
        "monolithic_coupled": "monolithic",
        "skip_k4": "skip_dag_k4",
    }
    tk_norm = tk_map.get(str(topology_kind), str(topology_kind))
    norm_cfg = {
        "pa": str(stage_cfg.get("pa") or stage_cfg.get("a") or "A0"),
        "pb": str(stage_cfg.get("pb") or stage_cfg.get("b") or "B0"),
        "pc1": str(stage_cfg.get("pc1") or stage_cfg.get("l") or stage_cfg.get("h") or "L0"),
        "pc2": str(stage_cfg.get("pc2") or stage_cfg.get("r") or "R0"),
        "rd": str(stage_cfg.get("rd") or stage_cfg.get("c") or "C0"),
        "cross_coupled": bool(stage_cfg.get("cross_coupled", stage_cfg.get("coupled", False))),
    }
    mono_m = str(stage_cfg.get("core") or stage_cfg.get("mono_mode") or "M0")
    eff_w_seed = int(world_seed)
    if tk_norm == "monolithic" and mono_m in ("M3", "M4"):
        offset_map = {"M3": (203, "M1"), "M4": (307, "M2")}
        off, base_m = offset_map[mono_m]
        eff_w_seed += off
        norm_cfg["core"] = base_m
    else:
        norm_cfg["core"] = mono_m

    return _gen_m101_study(
        int(batch_seed),
        world_seed=eff_w_seed,
        topology_kind=tk_norm,
        stage_cfg=norm_cfg,
        telemetry_vec=list(telemetry_vec),
    )
"""
)


@dataclass(frozen=True)
class M102WorldSpec:
    world_id: str
    cluster_id: str
    split: str  # 'unit_test_synthetic' | 'dev' | 'confirmation'
    stratum: str  # 'stratum_a_seen_topology' | 'stratum_b1_unseen_static_topology' | 'stratum_b2_shift_order_and_regime_switch'
    topology_class: str  # hidden analytics label ONLY!
    seed_base: int
    telemetry_channels: tuple[int, int, int]


# Dedicated Synthetic Unit-Test World Specs (`7201000..`, NEVER overlapping `dev` or `confirmation`! `U01`)
UNIT_TEST_M102_WORLD_SPECS: tuple[M102WorldSpec, ...] = (
    M102WorldSpec(
        world_id="ut_m102_mono",
        cluster_id="ut_cluster_m102_01",
        split="unit_test_synthetic",
        stratum="stratum_a_seen_topology",
        topology_class="monolithic_coupled",
        seed_base=7201000,
        telemetry_channels=(0, 1, 2),
    ),
    M102WorldSpec(
        world_id="ut_m102_k3",
        cluster_id="ut_cluster_m102_02",
        split="unit_test_synthetic",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_variable_cut",
        seed_base=7202000,
        telemetry_channels=(1, 2, 3),
    ),
    M102WorldSpec(
        world_id="ut_m102_dia",
        cluster_id="ut_cluster_m102_03",
        split="unit_test_synthetic",
        stratum="stratum_a_seen_topology",
        topology_class="diamond_k4_shared_prefix",
        seed_base=7203000,
        telemetry_channels=(2, 0, 3),
    ),
)

# `n = 4` Dev Calibration Clusters (`1141000..1144000`, `H = 16`)
DEV_M102_WORLD_SPECS: tuple[M102WorldSpec, ...] = (
    M102WorldSpec(
        world_id="dev_m102_c01",
        cluster_id="dev_cluster_m102_01",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="monolithic_coupled",
        seed_base=1141000,
        telemetry_channels=(0, 2, 1),
    ),
    M102WorldSpec(
        world_id="dev_m102_c02",
        cluster_id="dev_cluster_m102_02",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_variable_cut",
        seed_base=1142000,
        telemetry_channels=(1, 3, 2),
    ),
    M102WorldSpec(
        world_id="dev_m102_c03",
        cluster_id="dev_cluster_m102_03",
        split="dev",
        stratum="stratum_a_seen_topology",
        topology_class="diamond_k4_shared_prefix",
        seed_base=1143000,
        telemetry_channels=(2, 1, 3),
    ),
    M102WorldSpec(
        world_id="dev_m102_c04",
        cluster_id="dev_cluster_m102_04",
        split="dev",
        stratum="stratum_b2_shift_order_and_regime_switch",
        topology_class="diamond_k4_coupled_first",
        seed_base=1144000,
        telemetry_channels=(3, 0, 2),
    ),
)

# `n = 8` Strictly Unexposed Confirmation Clusters (`1151000..1158000`, `H = 16`):
CONFIRMATION_M102_WORLD_SPECS: tuple[M102WorldSpec, ...] = (
    M102WorldSpec(
        world_id="conf_m102_a01",
        cluster_id="conf_cluster_m102_a01",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="monolithic_coupled",
        seed_base=1151000,
        telemetry_channels=(1, 2, 0),
    ),
    M102WorldSpec(
        world_id="conf_m102_a02",
        cluster_id="conf_cluster_m102_a02",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_variable_cut",
        seed_base=1152000,
        telemetry_channels=(0, 3, 1),
    ),
    M102WorldSpec(
        world_id="conf_m102_a03",
        cluster_id="conf_cluster_m102_a03",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="chain_k3_reversed_cut",
        seed_base=1153000,
        telemetry_channels=(2, 3, 0),
    ),
    M102WorldSpec(
        world_id="conf_m102_a04",
        cluster_id="conf_cluster_m102_a04",
        split="confirmation",
        stratum="stratum_a_seen_topology",
        topology_class="diamond_k4_shared_prefix",
        seed_base=1154000,
        telemetry_channels=(3, 1, 2),
    ),
    M102WorldSpec(
        world_id="conf_m102_b01",
        cluster_id="conf_cluster_m102_b01",
        split="confirmation",
        stratum="stratum_b1_unseen_static_topology",
        topology_class="chain_k4_deep_cascade",
        seed_base=1155000,
        telemetry_channels=(0, 1, 3),
    ),
    M102WorldSpec(
        world_id="conf_m102_b02",
        cluster_id="conf_cluster_m102_b02",
        split="confirmation",
        stratum="stratum_b1_unseen_static_topology",
        topology_class="asymmetric_skip_dag_k4",
        seed_base=1156000,
        telemetry_channels=(1, 3, 0),
    ),
    M102WorldSpec(
        world_id="conf_m102_b03",
        cluster_id="conf_cluster_m102_b03",
        split="confirmation",
        stratum="stratum_b2_shift_order_and_regime_switch",
        topology_class="diamond_k4_coupled_first",
        seed_base=1157000,
        telemetry_channels=(2, 0, 1),
    ),
    M102WorldSpec(
        world_id="conf_m102_b04",
        cluster_id="conf_cluster_m102_b04",
        split="confirmation",
        stratum="stratum_b2_shift_order_and_regime_switch",
        topology_class="hybrid_mono_to_split_shift",
        seed_base=1158000,
        telemetry_channels=(3, 2, 1),
    ),
)


def _tel_vec_m102(channels: tuple[int, int, int], f0: int, f1: int, f2: int) -> tuple[float, float, float, float]:
    v = [0.22, 0.22, 0.22, 0.22]
    v[channels[0]] = 1.25 if f0 >= 1 else 0.22
    if f0 == 2:
        v[channels[0]] = 2.15
    v[channels[1]] = 1.25 if f1 >= 1 else 0.22
    if f1 == 2:
        v[channels[1]] = 2.15
    v[channels[2]] = 1.25 if f2 >= 1 else 0.22
    if f2 == 2:
        v[channels[2]] = 2.15
    return (v[0], v[1], v[2], v[3])


def _schedule_for_m102_topology(spec: M102WorldSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Construct `Stage S` + `16` balanced continual steps (`Wave 1: S01..S08` + `Wave 2: S09..S16` with new `A2/B2/L2/R2` regimes!)."""
    ch = spec.telemetry_channels
    tc = spec.topology_class

    if tc == "monolithic_coupled":
        s0 = {"event": "S00_source", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)}
        steps = [
            {"event": "S01_mono_shift_1", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M1"}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S02_mono_revisit_0", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)},
            {"event": "S03_mono_shift_2", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M2"}, "tel": _tel_vec_m102(ch, 0, 1, 0)},
            {"event": "S04_mono_revisit_1", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M1"}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S05_mono_shift_3", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M3"}, "tel": _tel_vec_m102(ch, 1, 1, 0)},
            {"event": "S06_mono_revisit_2", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M2"}, "tel": _tel_vec_m102(ch, 0, 1, 0)},
            {"event": "S07_mono_revisit_0b", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)},
            {"event": "S08_mono_revisit_3", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M3"}, "tel": _tel_vec_m102(ch, 1, 1, 0)},
            {"event": "S09_mono_shift_4", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M4"}, "tel": _tel_vec_m102(ch, 0, 0, 1)},
            {"event": "S10_mono_revisit_1b", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M1"}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S11_mono_revisit_4", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M4"}, "tel": _tel_vec_m102(ch, 0, 0, 1)},
            {"event": "S12_mono_revisit_2b", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M2"}, "tel": _tel_vec_m102(ch, 0, 1, 0)},
            {"event": "S13_mono_revisit_3b", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M3"}, "tel": _tel_vec_m102(ch, 1, 1, 0)},
            {"event": "S14_mono_revisit_0c", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)},
            {"event": "S15_mono_revisit_4b", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M4"}, "tel": _tel_vec_m102(ch, 0, 0, 1)},
            {"event": "S16_mono_revisit_1c", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M1"}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
        ]
        return s0, steps

    if tc in {"chain_k3_variable_cut", "chain_k3_reversed_cut"}:
        rev = (tc == "chain_k3_reversed_cut")
        s0 = {"event": "S00_source", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 0)}
        w1 = [
            {"event": "S01_shift_head" if rev else "S01_shift_mid", "topology_kind": "chain_k3", "stage_cfg": {"a": ("A1" if rev else "A0"), "b": ("B0" if rev else "B1"), "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, (1 if rev else 0), (0 if rev else 1), 0)},
            {"event": "S02_shift_tail", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B0", "c": "C1", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 1)},
            {"event": "S03_shift_mid" if rev else "S03_shift_head", "topology_kind": "chain_k3", "stage_cfg": {"a": ("A0" if rev else "A1"), "b": ("B1" if rev else "B0"), "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, (0 if rev else 1), (1 if rev else 0), 0)},
            {"event": "S04_unseen_comb_1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 0)},
            {"event": "S05_unseen_comb_2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B1", "c": "C1", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 1)},
            {"event": "S06_interaction_coupled", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B1", "c": "C1", "coupled": True}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S07_revisit_sep_s04", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 0)},
            {"event": "S08_revisit_coupled_s06", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B1", "c": "C1", "coupled": True}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
        ]
        w2 = [
            {"event": "S09_wave2_shift_a2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 0, 0)},
            {"event": "S10_unseen_comb_a2_b1_c0", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 0)},
            {"event": "S11_unseen_comb_a2_b0_c1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B0", "c": "C1", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 0, 1)},
            {"event": "S12_wave2_shift_b2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 2, 0)},
            {"event": "S13_unseen_comb_a1_b2_c1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B2", "c": "C1", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 2, 1)},
            {"event": "S14_wave2_coupled_a2_b1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B1", "c": "C1", "coupled": True}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
            {"event": "S15_unseen_comb_a2_b2_c0", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 2, 0)},
            {"event": "S16_revisit_sep_s10", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 0)},
        ]
        return s0, w1 + w2

    if tc in {"diamond_k4_shared_prefix", "diamond_k4_coupled_first"}:
        cpl_first = (tc == "diamond_k4_coupled_first")
        s0 = {"event": "S00_source", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L0", "r": "R0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 0)}
        w1 = [
            {"event": "S01_coupled_shock" if cpl_first else "S01_shift_left", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L1", "r": ("R1" if cpl_first else "R0"), "c": "C0", "coupled": cpl_first}, "tel": _tel_vec_m102(ch, 0, 1, (1 if cpl_first else 0))},
            {"event": "S02_shift_right", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L0", "r": "R1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 1)},
            {"event": "S03_shift_prefix", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A1", "l": "L0", "r": "R0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S04_shift_left_clean" if cpl_first else "S04_unseen_comb_lr", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L1", "r": ("R0" if cpl_first else "R1"), "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, (0 if cpl_first else 1))},
            {"event": "S05_unseen_comb_a1_l1_r1", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A1", "l": "L1", "r": "R1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S06_unseen_comb_a0_l1_r1" if cpl_first else "S06_interaction_coupled", "topology_kind": "diamond_k4", "stage_cfg": {"a": ("A0" if cpl_first else "A1"), "l": "L1", "r": "R1", "c": "C0", "coupled": (not cpl_first)}, "tel": _tel_vec_m102(ch, (0 if cpl_first else 1), 1, 1)},
            {"event": "S07_revisit_sep", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L1", "r": "R1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 1)},
            {"event": "S08_revisit_coupled", "topology_kind": "diamond_k4", "stage_cfg": {"a": ("A0" if cpl_first else "A1"), "l": "L1", "r": "R1", "c": "C0", "coupled": True}, "tel": _tel_vec_m102(ch, (0 if cpl_first else 1), 1, 1)},
        ]
        w2 = [
            {"event": "S09_wave2_shift_l2", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L2", "r": "R0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 2, 0)},
            {"event": "S10_unseen_comb_a1_l2_r1", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A1", "l": "L2", "r": "R1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 2, 1)},
            {"event": "S11_wave2_shift_a2", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A2", "l": "L0", "r": "R0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 0, 0)},
            {"event": "S12_unseen_comb_a2_l1_r1", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A2", "l": "L1", "r": "R1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
            {"event": "S13_wave2_shift_r2", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A0", "l": "L0", "r": "R2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 2)},
            {"event": "S14_unseen_comb_a2_l2_r2", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A2", "l": "L2", "r": "R2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 2, 2)},
            {"event": "S15_unseen_comb_a1_l0_r2", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A1", "l": "L0", "r": "R2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 0, 2)},
            {"event": "S16_revisit_sep_a2_l1_r1", "topology_kind": "diamond_k4", "stage_cfg": {"a": "A2", "l": "L1", "r": "R1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
        ]
        return s0, w1 + w2

    if tc in {"chain_k4_deep_cascade", "asymmetric_skip_dag_k4"}:
        tk = "chain_k4" if tc == "chain_k4_deep_cascade" else "skip_k4"
        s0 = {"event": "S00_source", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B0", "h": "H0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 0)}
        steps = [
            {"event": "S01_shift_b1", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B1", "h": "H0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 0)},
            {"event": "S02_shift_h1", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B0", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 1)},
            {"event": "S03_shift_a1", "topology_kind": tk, "stage_cfg": {"a": "A1", "b": "B0", "h": "H0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S04_unseen_comb_a0_b1_h1", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B1", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 1)},
            {"event": "S05_unseen_comb_a1_b1_h1", "topology_kind": tk, "stage_cfg": {"a": "A1", "b": "B1", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S06_interaction_coupled", "topology_kind": tk, "stage_cfg": {"a": "A1", "b": "B1", "h": "H1", "c": "C1", "coupled": True}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S07_revisit_sep_a0_b1_h1", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B1", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 1)},
            {"event": "S08_revisit_coupled", "topology_kind": tk, "stage_cfg": {"a": "A1", "b": "B1", "h": "H1", "c": "C1", "coupled": True}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S09_wave2_shift_a2", "topology_kind": tk, "stage_cfg": {"a": "A2", "b": "B0", "h": "H0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 0, 0)},
            {"event": "S10_unseen_comb_a2_b1_h1", "topology_kind": tk, "stage_cfg": {"a": "A2", "b": "B1", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
            {"event": "S11_wave2_shift_h2", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B0", "h": "H2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 0, 2)},
            {"event": "S12_unseen_comb_a1_b1_h2", "topology_kind": tk, "stage_cfg": {"a": "A1", "b": "B1", "h": "H2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 2)},
            {"event": "S13_wave2_shift_b2", "topology_kind": tk, "stage_cfg": {"a": "A0", "b": "B2", "h": "H0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 2, 0)},
            {"event": "S14_unseen_comb_a2_b2_h2", "topology_kind": tk, "stage_cfg": {"a": "A2", "b": "B2", "h": "H2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 2, 2)},
            {"event": "S15_unseen_comb_a2_b2_h1", "topology_kind": tk, "stage_cfg": {"a": "A2", "b": "B2", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 2, 1)},
            {"event": "S16_revisit_sep_a2_b1_h1", "topology_kind": tk, "stage_cfg": {"a": "A2", "b": "B1", "h": "H1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
        ]
        return s0, steps

    if tc == "hybrid_mono_to_split_shift":
        s0 = {"event": "S00_source_mono", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)}
        steps = [
            {"event": "S01_mono_shift_1", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M1"}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S02_mono_revisit_0", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)},
            {"event": "S03_switch_to_sep_base", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 0)},
            {"event": "S04_sep_shift_b1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 1, 1)},
            {"event": "S05_sep_shift_a1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 0)},
            {"event": "S06_unseen_comb_a1_b1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S07_revisit_mono_1", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M1"}, "tel": _tel_vec_m102(ch, 1, 0, 0)},
            {"event": "S08_revisit_sep_a1_b1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 1, 1)},
            {"event": "S09_sep_wave2_shift_a2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B0", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 0)},
            {"event": "S10_unseen_comb_a2_b1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
            {"event": "S11_sep_wave2_shift_b2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A0", "b": "B2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 0, 2, 0)},
            {"event": "S12_unseen_comb_a1_b2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A1", "b": "B2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 1, 2, 0)},
            {"event": "S13_unseen_comb_a2_b2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 2, 0)},
            {"event": "S14_revisit_mono_0", "topology_kind": "monolithic_coupled", "stage_cfg": {"mono_mode": "M0"}, "tel": _tel_vec_m102(ch, 0, 0, 0)},
            {"event": "S15_revisit_sep_a2_b1", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B1", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 1, 1)},
            {"event": "S16_revisit_sep_a2_b2", "topology_kind": "chain_k3", "stage_cfg": {"a": "A2", "b": "B2", "c": "C0", "coupled": False}, "tel": _tel_vec_m102(ch, 2, 2, 0)},
        ]
        return s0, steps

    raise ValueError(f"Unknown M10.2 topology_class: {tc}")


def _build_m102_task(
    *,
    task_id: str,
    short_name: str,
    world_seed: int,
    pub_seed: int,
    priv_s1: int,
    priv_s2: int,
    topology_kind: str,
    stage_cfg: dict[str, Any],
    telemetry_vec: tuple[float, float, float, float],
) -> HoldoutOODTask:
    pub_call = (
        f"_gen_m102_study(world_seed={world_seed}, batch_seed={pub_seed}, "
        f"topology_kind={topology_kind!r}, stage_cfg={stage_cfg!r}, telemetry_vec={telemetry_vec!r})"
    )
    priv_cases = (
        f"  ({priv_s1}, *_gen_m102_study(world_seed={world_seed}, batch_seed={priv_s1}, topology_kind={topology_kind!r}, stage_cfg={stage_cfg!r}, telemetry_vec={telemetry_vec!r})),\n"
        f"  ({priv_s2}, *_gen_m102_study(world_seed={world_seed}, batch_seed={priv_s2}, topology_kind={topology_kind!r}, stage_cfg={stage_cfg!r}, telemetry_vec={telemetry_vec!r})),"
    )
    schema_py = """from __future__ import annotations
import copy, importlib.util, json, os, pickle, sys
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
cand_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_d, "batch_solver.py")
spec = importlib.util.spec_from_file_location("_cand", cand_p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with open(os.path.join(_d, "public_batch.pkl"), "rb") as fh:
    st = pickle.load(fh)[0]["study"]
out = mod.solve_batch(copy.deepcopy(st))
n_te = len(st["test"]["x"])
k_t = len(st["eval_times"])
errs = []
if not isinstance(out, dict):
    errs.append("not_dict")
else:
    for k, shp in (("y_hat", (n_te,)), ("state_pred", (n_te, k_t)), ("diagnostics_vec", (4,))):
        if k not in out:
            errs.append(f"missing:{k}"); continue
        a = np.asarray(out[k], dtype=float)
        if a.shape != shp or not np.all(np.isfinite(a)):
            errs.append(f"bad:{k}:{a.shape}")
    if not errs:
        yh = np.asarray(out["y_hat"], dtype=float)
        sp = np.asarray(out["state_pred"], dtype=float)
        dv = np.asarray(out["diagnostics_vec"], dtype=float)
        exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
        if float(np.max(np.abs(dv - exp_dv))) > 1e-3:
            errs.append("diagnostics_vec_handoff_mismatch")
    if not errs:
        st_cv = copy.deepcopy(st)
        n_tr = len(st["train"]["x"])
        s_idx = max(40, n_tr - 20)
        vx = st["train"]["x"][s_idx:].copy()
        vy = st["train"]["y"][s_idx:].copy()
        vt = st["train"]["traj"][s_idx:].copy()
        st_cv["train"] = {
            "x": st["train"]["x"][:s_idx].copy(),
            "y": st["train"]["y"][:s_idx].copy(),
            "traj": st["train"]["traj"][:s_idx].copy(),
            "summary": np.array([float(np.mean(st["train"]["y"][:s_idx])), float(np.std(st["train"]["y"][:s_idx])), float(np.mean(st["train"]["traj"][:s_idx, -1])), float(np.std(st["train"]["traj"][:s_idx, -1]))], dtype=float),
        }
        st_cv["test"] = {"x": vx}
        ocv = mod.solve_batch(st_cv)
        cy = np.asarray(ocv["y_hat"], dtype=float)
        ct = np.asarray(ocv["state_pred"], dtype=float)
        if cy.shape != vy.shape or ct.shape != vt.shape or not (np.all(np.isfinite(cy)) and np.all(np.isfinite(ct))):
            errs.append("cv_shape_or_nonfinite")
        else:
            ym = float(np.mean(np.abs(cy - vy)))
            tm = float(np.mean(np.abs(ct - vt)))
            if ym > 0.038:
                errs.append(f"public_cv_y_mae_exceeded:{ym:.6f}")
            if tm > 0.038:
                errs.append(f"public_cv_traj_mae_exceeded:{tm:.6f}")
print(json.dumps({"schema_valid": not errs, "errors": errs}, allow_nan=False))
sys.exit(1 if errs else 0)
"""
    pub_setup = (
        "import json, math, os, pickle, sys\nimport numpy as np\n"
        + _M102_WORLD_GEN_EXTENSION_PY
        + f"\npub_st, _unused_gt = {pub_call}\n"
        'with open("/holdout_task/public_batch.pkl", "wb") as fh:\n'
        '    pickle.dump([{"study": pub_st}], fh)\n'
        + f'open("/holdout_task/validate_public_schema.py", "w").write({schema_py!r})\n'
    )
    priv_ver = (
        "import json, math, os, sys, time\nimport numpy as np\nsys.path.insert(0, '/holdout_task')\n"
        "import batch_solver as cand_mod\n"
        + _M102_WORLD_GEN_EXTENSION_PY
        + f"\ncases = [\n{priv_cases}\n]\n"
        + """
t0 = time.monotonic()
valid = True; max_y_mae = None; max_tr_mae = None; max_dv_mae = None; rows = []
for sd, st, gt in cases:
    out = cand_mod.solve_batch(st)
    if not isinstance(out, dict) or not {"y_hat", "state_pred", "diagnostics_vec"} <= set(out.keys()):
        valid = False; rows.append({"seed": sd, "error": "MISSING_KEYS"}); continue
    yh = np.asarray(out["y_hat"], dtype=float)
    sp = np.asarray(out["state_pred"], dtype=float)
    dv = np.asarray(out["diagnostics_vec"], dtype=float)
    if yh.shape != gt["y_clean"].shape or sp.shape != gt["traj_clean"].shape or dv.shape != (4,):
        valid = False; rows.append({"seed": sd, "error": "SHAPE_MISMATCH"}); continue
    if not (np.all(np.isfinite(yh)) and np.all(np.isfinite(sp)) and np.all(np.isfinite(dv))):
        valid = False; rows.append({"seed": sd, "error": "NON_FINITE"}); continue
    y_mae = float(np.mean(np.abs(yh - gt["y_clean"])))
    tr_mae = float(np.mean(np.abs(sp - gt["traj_clean"])))
    dv_mae = float(np.mean(np.abs(dv - gt["summary_clean"])))
    exp_dv = np.array([float(np.mean(yh)), float(np.std(yh)), float(np.mean(sp[:, -1])), float(np.std(sp[:, -1]))], dtype=float)
    handoff_err = float(np.max(np.abs(dv - exp_dv)))
    max_y_mae = y_mae if max_y_mae is None else max(max_y_mae, y_mae)
    max_tr_mae = tr_mae if max_tr_mae is None else max(max_tr_mae, tr_mae)
    max_dv_mae = dv_mae if max_dv_mae is None else max(max_dv_mae, dv_mae)
    if handoff_err > 1e-3:
        valid = False; rows.append({"seed": sd, "error": "HANDOFF_MISMATCH", "y_mae": round(y_mae, 6), "tr_mae": round(tr_mae, 6)}); continue
    rows.append({"seed": sd, "y_mae": round(y_mae, 6), "tr_mae": round(tr_mae, 6), "dv_mae": round(dv_mae, 6)})
el = time.monotonic() - t0
sp_ok = bool(valid and len(rows) == 2 and max_y_mae is not None and max_tr_mae is not None and max_dv_mae is not None and max_y_mae <= 0.045 and max_tr_mae <= 0.045 and max_dv_mae <= 0.055 and el <= 25.0)
print(json.dumps({"strict_pass": sp_ok, "reward": 1.0 if sp_ok else 0.0, "max_y_mae": (round(max_y_mae, 6) if max_y_mae is not None else None), "max_traj_mae": (round(max_tr_mae, 6) if max_tr_mae is not None else None), "max_diag_mae": (round(max_dv_mae, 6) if max_dv_mae is not None else None), "student_elapsed_sec": round(el, 4), "instance_diagnostics": rows}, allow_nan=False))
"""
    )
    return HoldoutOODTask(
        task_id=task_id,
        short_name=short_name,
        wave="m10_2_selective_bounded_dag_discovery_h16",
        scientific_domain="blind_multi_stage_scientific_computation_graph",
        wormhole_family="m102_blind_dag",
        declared_artifacts=("/holdout_task/batch_solver.py",),
        declared_callable_entrypoints=("solve_batch",),
        public_pickle_filename="public_batch.pkl",
        host_only_public_gt_json="",
        instruction_text=BLIND_M101_INSTRUCTION_TEXT,
        public_setup_py=pub_setup,
        private_verifier_py=priv_ver,
    )


def build_m102_world_tasks(
    spec: M102WorldSpec,
) -> tuple[HoldoutOODTask, tuple[HoldoutOODTask, ...], list[dict[str, Any]]]:
    """Construct `Stage S` + `16` continual stream tasks (`H = 16`, `H = 4, 8, 12` nested checkpoints)."""
    s0_cfg, step_cfgs = _schedule_for_m102_topology(spec)
    assert len(step_cfgs) == M102_HORIZON_STEPS
    sb = spec.seed_base
    w_seed = sb + 7
    s_task = _build_m102_task(
        task_id=f"{spec.world_id}_batch_00__blind_campaign",
        short_name=f"{spec.world_id}_batch_00",
        world_seed=w_seed,
        pub_seed=sb + 10,
        priv_s1=sb + 20,
        priv_s2=sb + 30,
        topology_kind=s0_cfg["topology_kind"],
        stage_cfg=s0_cfg["stage_cfg"],
        telemetry_vec=s0_cfg["tel"],
    )
    stream_tasks: list[HoldoutOODTask] = []
    manifest: list[dict[str, Any]] = []
    for idx, sc in enumerate(step_cfgs, start=1):
        tk = _build_m102_task(
            task_id=f"{spec.world_id}_batch_{idx:02d}__blind_campaign",
            short_name=f"{spec.world_id}_batch_{idx:02d}",
            world_seed=w_seed,
            pub_seed=sb + 100 * idx + 10,
            priv_s1=sb + 100 * idx + 20,
            priv_s2=sb + 100 * idx + 30,
            topology_kind=sc["topology_kind"],
            stage_cfg=sc["stage_cfg"],
            telemetry_vec=sc["tel"],
        )
        stream_tasks.append(tk)
        manifest.append(
            {
                "step": idx,
                "task_id": tk.task_id,
                "short_name": tk.short_name,
                "event": sc["event"],
                "causal_event": sc["event"],
                "topology_kind": sc["topology_kind"],
                "stage_cfg": deepcopy(sc["stage_cfg"]),
                "telemetry_vec": list(sc["tel"]),
            }
        )
    return s_task, tuple(stream_tasks), manifest


# ============================================================================
# 2. Zero-LLM Evidence-Based Structural Investment & Maintenance Gate (`E2`)
# ============================================================================


def inspect_monolithic_code_ast_separability(source_code: str) -> dict[str, Any]:
    """Pure Python AST dataflow inspection (`0` LLM calls, `0` domain-truth leakage!).

    Counts how many distinct non-target top-level sub-tables of `study` (`k not in {'train', 'test', 'eval_times', 'telemetry_log'}`)
    are actually indexed AND used in numerical solver calls (`lstsq`, `solve`, `minimize`, `least_squares`, `curve_fit`, `inv`, `pinv`).
    When a monolithic solver directly fits `train['x'] -> y, traj` without using `>= 1` calibration sub-table in numerical estimation
    (or uses `0` auxiliary sub-tables), decomposing it into `K >= 2` nodes cannot factorize independent calibration stages!
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {"parse_ok": False, "aux_keys_read": [], "solver_call_count": 0, "ast_separable": False}

    core_keys = {"train", "test", "eval_times", "telemetry_log", "telemetry_vec", "__input__"}
    io_keys = {"x", "y", "traj", "summary", "y_hat", "state_pred", "diagnostics_vec"}
    aux_keys: set[str] = set()
    solver_names = {
        "lstsq",
        "solve",
        "minimize",
        "least_squares",
        "curve_fit",
        "root",
        "fsolve",
        "pinv",
        "inv",
        "eigh",
        "svd",
        "qr",
        "polyfit",
    }

    trees_to_scan = [tree]
    # Also scan embedded `_NODE_SOURCES` if `source_code` is a compiled DAG wrapper
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_NODE_SOURCES" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            try:
                ns_map = json.loads(node.value.value)
                if isinstance(ns_map, dict):
                    for sub_src in ns_map.values():
                        if isinstance(sub_src, str):
                            trees_to_scan.append(ast.parse(sub_src))
            except Exception:
                pass

    helper_solver_fns: set[str] = set()
    for tr in trees_to_scan:
        for fn_def in ast.walk(tr):
            if isinstance(fn_def, ast.FunctionDef) and fn_def.name not in {"solve_batch", "run_node"}:
                for sub in ast.walk(fn_def):
                    if isinstance(sub, ast.Call):
                        f = sub.func
                        nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                        if nm in solver_names:
                            helper_solver_fns.add(fn_def.name)

    effective_solver_calls = 0
    for tr in trees_to_scan:
        for node in ast.walk(tr):
            if isinstance(node, ast.Subscript):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    sval = sl.value
                    if (
                        sval not in core_keys
                        and sval not in io_keys
                        and not sval.startswith("node_")
                        and len(sval) <= 32
                    ):
                        aux_keys.add(sval)
            elif isinstance(node, ast.Call):
                fn = node.func
                fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                if fn_name in solver_names or fn_name in helper_solver_fns:
                    effective_solver_calls += 1
                elif fn_name == "get" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    sval = node.args[0].value
                    if sval not in core_keys and sval not in io_keys and not sval.startswith("node_"):
                        aux_keys.add(sval)

    separable = len(aux_keys) >= 1 and (effective_solver_calls >= 2 or len(aux_keys) >= 2)
    return {
        "parse_ok": True,
        "aux_keys_read": sorted(aux_keys),
        "solver_call_count": effective_solver_calls,
        "ast_separable": bool(separable),
    }


def evaluate_m102_selective_investment_gate(
    *,
    phase: str,  # 'initial_decomposition' | 'graph_maintenance_edit'
    source_code: str,
    ref_obs: dict[str, float | None],
    target_obs: dict[str, float | None],
    step_index: int,
    total_horizon: int,
    consecutive_failed_dag_searches: int,
    total_failed_dag_searches: int,
    solver_free_dag_reuse_hits: int,
) -> dict[str, Any]:
    """Deterministic `0`-LLM Evidence-Based Structural Investment & Maintenance Policy (`E2`).

    Decides whether opening paid DAG decomposition / graph-edit search has positive expected return vs `keep_monolithic`:
      1. Remaining-horizon payback rule: if `remaining_steps <= 1`, a brand-new decomposition cannot amortize its proposal+validation tax.
      2. AST separability rule (`initial_decomposition`): if `source_code` does not perform multi-stage numerical solves across
         auxiliary data sub-tables (`ast_separable == False`), retain `K = 1` (`keep_monolithic`) at `0` LLM cost!
      3. Shift-localization rule (`initial_decomposition`): group `stat_00..stat_11` into 3 generic 4-stat blocks (`G0=0..3`, `G1=4..7`,
         `G2=8..11`) and `G_out=12..13`. If `0` of `(G0, G1, G2)` shifted (`max_delta < 0.06`, i.e., pure readout/monolithic shift) OR
         ALL `3/3` of `(G0, G1, G2)` shifted simultaneously on Step 1 (`coupled_first` shock), defer decomposition to monolithic solve!
      4. Strike-out rule (`graph_maintenance_edit`): if `consecutive_failed_dag_searches >= 1` OR
         (`total_failed_dag_searches >= 2` and `solver_free_dag_reuse_hits == 0`), suspend graph surgery for this step and solve
         monolithically while preserving all existing verified DAGs in the archive!
    """
    rem_steps = int(total_horizon) - int(step_index)
    # Four scale-invariant R^2 / nonlinear-gain statistical blocks over `stat_00..stat_11`
    # (excludes finite-sample RMS `stat_03/stat_07` so unshifted 24-row draws have jitter < 0.025 << 0.08!)
    blocks = (
        ("G0", ("stat_00", "stat_02")),
        ("G1", ("stat_04", "stat_05", "stat_06")),
        ("G2", ("stat_08", "stat_09")),
        ("G3", ("stat_10", "stat_11")),
    )
    shifted_blocks: list[str] = []
    for b_nm, keys in blocks:
        deltas = [
            abs(float(target_obs[k]) - float(ref_obs[k]))
            for k in keys
            if target_obs.get(k) is not None and ref_obs.get(k) is not None
        ]
        if deltas and max(deltas) >= 0.08:
            shifted_blocks.append(b_nm)

    if phase == "initial_decomposition":
        if rem_steps <= 1:
            return {
                "invest_in_structure": False,
                "permanent_source_verdict": False,
                "reason": "insufficient_remaining_horizon_for_amortization",
                "remaining_steps": rem_steps,
                "shifted_blocks": shifted_blocks,
            }
        ast_info = inspect_monolithic_code_ast_separability(source_code)
        if not ast_info["ast_separable"]:
            return {
                "invest_in_structure": False,
                "permanent_source_verdict": True,
                "reason": "source_ast_not_multi_stage_separable",
                "ast_info": ast_info,
                "shifted_blocks": shifted_blocks,
            }
        if len(shifted_blocks) == 0:
            return {
                "invest_in_structure": False,
                "permanent_source_verdict": False,
                "reason": "no_auxiliary_subtable_shift_keep_monolithic",
                "ast_info": ast_info,
                "shifted_blocks": shifted_blocks,
            }
        if len(shifted_blocks) >= 2 and int(step_index) == 1:
            return {
                "invest_in_structure": False,
                "permanent_source_verdict": False,
                "reason": "simultaneous_multi_block_shock_defer_initial_split",
                "ast_info": ast_info,
                "shifted_blocks": shifted_blocks,
            }
        return {
            "invest_in_structure": True,
            "permanent_source_verdict": False,
            "reason": "ast_separable_and_localized_shift_evidence",
            "ast_info": ast_info,
            "shifted_blocks": shifted_blocks,
        }

    # `phase == 'graph_maintenance_edit'`
    if consecutive_failed_dag_searches >= 1:
        return {
            "invest_in_structure": False,
            "permanent_source_verdict": False,
            "reason": "cooldown_after_recent_failed_dag_surgery",
            "consecutive_failed_dag_searches": consecutive_failed_dag_searches,
            "shifted_blocks": shifted_blocks,
        }
    if total_failed_dag_searches >= 3 and solver_free_dag_reuse_hits == 0:
        return {
            "invest_in_structure": False,
            "permanent_source_verdict": False,
            "reason": "negative_net_maintenance_return_suspend_surgery",
            "total_failed_dag_searches": total_failed_dag_searches,
            "shifted_blocks": shifted_blocks,
        }
    return {
        "invest_in_structure": True,
        "permanent_source_verdict": False,
        "reason": "maintenance_budget_approved",
        "shifted_blocks": shifted_blocks,
    }


# ============================================================================
# 3. De-Scaffolded Prompt Builders (`U06 / §七`: ZERO `panel_a/b/c` Coaching!)
# ============================================================================

DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE = (
    "You are a domain-agnostic Python dataflow decomposition architect.\n"
    "Given a verified monolithic `solve_batch(study)` Python program (`source_monolithic_code`) and role-blind summary "
    "statistics (`stat_00..stat_17`) comparing the verified source batch against a new shifted batch where `source_monolithic_code` "
    "failed 52/20 public cross-validation, decide whether `source_monolithic_code` admits a clean bounded acyclic DAG factorization.\n"
    "HARD TOPOLOGY & RUNTIME INVARIANTS:\n"
    "- Bounds: `1 <= active_nodes <= {max_nodes}`, acyclic, `depth <= {max_depth}`, `max_fanout <= {max_fanout}`.\n"
    "- Choose `keep_monolithic` (`nodes=[]`, `edges=[]`, `sink_node_id=''`, `node_modules=[]`) whenever `source_monolithic_code` "
    "forms a single tightly coupled estimation routine without independent reusable sub-transforms (`K = 1` is ALWAYS legal and zero-penalty).\n"
    "- Choose `split_node` or `extract_module` ONLY when `source_monolithic_code` genuinely computes distinct sequential or parallel "
    "numerical sub-transformations over sub-structures of `study`.\n"
    "- Neutral node IDs ONLY: `node_00`, `node_01`, ...; every node reading raw `study` MUST declare edge `['__input__', node_id]`.\n"
    "- Each node module defines `def run_node(inputs: dict) -> dict:` where `inputs` maps each declared incoming source ID "
    "(`'__input__'` or predecessor `'node_XX'`) to its output dictionary.\n"
    "- STRICT CAUSAL EDGE CONSUMPTION & NOVEL NUMERIC OUTPUT: Every node MUST actually read every incoming key declared in `edges` "
    "(no dead/unconsumed incoming edges!), and every non-sink node MUST return a non-empty dict containing newly computed `np.ndarray` "
    "numerical features/parameters (NEVER wrap or echo `inputs['__input__']`!).\n"
    "- SHARED COMPUTATION: When two downstream nodes consume the same upstream computation, wire one shared upstream node (`fan-out <= {max_fanout}`) "
    "so it executes ONCE (`exec_count == 1`).\n"
    "- The decomposed DAG MUST reproduce `source_monolithic_code`'s accuracy on the verified source batch (`<= 0.038` public CV)."
)

DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a domain-agnostic computational-DAG edit planner selecting the minimal valid graph action to adapt an archive of "
    "verified DAGs and callable Python node versions to a new shifted batch.\n"
    "HARD BOUNDS: `1 <= len(new_nodes) <= {max_nodes}`, acyclic, `depth <= {max_depth}`, `max_fanout <= {max_fanout}`.\n"
    "LEGAL ACTIONS:\n"
    "- `reuse_subgraph`: Keep an existing multi-node DAG topology (`base_dag_id`), freeze all unchanged nodes via `reused_node_versions`, "
    "and list ONLY the `1` (or `2`) shifted node ID(s) in `editable_node_ids` to re-synthesize after inspecting each node's Python `code` "
    "and `stat_00..stat_17` deltas.\n"
    "- `split_node` / `extract_module`: Refine a coarse node into separate nodes (`<= {max_nodes}`) when its Python `code` bundles two "
    "independent sub-computations and only one shifted.\n"
    "- `rewire_edge`: Freeze reusable upstream/branch nodes, add a direct cross-node edge (`<= {max_fanout}` fan-out) into a downstream/sink "
    "node when cross-stage coupling falsifies an earlier cut despite previously seen marginal statistics, and re-synthesize ONLY the target node "
    "(ensuring the target node's `run_node` actually reads the newly wired incoming key!).\n"
    "- `merge_nodes`: Fuse coupled nodes into fewer nodes (or `K = 1`) when `rewire_edge` would exceed `max_fanout` or cuts no longer separate.\n"
    "- `keep_monolithic`: Solve directly as a single `K = 1` node (`new_nodes=['node_00']`, `new_edges=[['__input__','node_00']]`, "
    "`new_sink_node_id='node_00'`, `reused_node_versions=[]`, `editable_node_ids=['node_00']`).\n"
    "PARTITION INVARIANT: `reused_node_versions` and `editable_node_ids` must be non-empty (`editable_node_ids`), disjoint, and their union "
    "must equal `new_nodes`. Every node reading `study` must include `['__input__', node_id]` in `new_edges`."
)


__all__ = [
    "CONFIRMATION_M102_WORLD_SPECS",
    "DEV_M102_WORLD_SPECS",
    "DE_SCAFFOLDED_EDIT_PLANNER_SYSTEM_PROMPT_TEMPLATE",
    "DE_SCAFFOLDED_INITIAL_DECOMP_SYSTEM_PROMPT_TEMPLATE",
    "M102WorldSpec",
    "M102_ARMS",
    "M102_BORDERLINE_CV_COMPARE_THRESHOLD",
    "M102_DAG_LEDGER_FOREST_KEY",
    "M102_HORIZON_STEPS",
    "M102_MAX_PRE_LLM_PROBES_PER_STEP",
    "M102_NESTED_CHECKPOINTS",
    "M102_OBSERVABLE_NAMES",
    "UNIT_TEST_M102_WORLD_SPECS",
    "build_m102_world_tasks",
    "compute_m102_blind_observables",
    "evaluate_m102_selective_investment_gate",
    "inspect_monolithic_code_ast_separability",
]

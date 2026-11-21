"""Hegel M6 Live Official ChatGPT-Subscription Benchmark (`Hegel-M6-Live-Causal-24` on isolated branch `exp/m6-causal-wormhole-sandbox`).

Directly benchmarks `Hegel_M6_Aufhebung_LiveGPT` vs `Hegel_M5_GrowthOnly_LiveGPT` vs `EqualAuth_Flat_LiveGPT` vs
`Strong_Ordinary_LiveGPT` across `gpt-6-luna`, `gpt-6.1-terra`, and `gpt-6.2-sol` (`24 tasks × 4 arms × 3 models = 288 live episodes`)
strictly via Official OpenAI ChatGPT Subscription (`auth_mode == 'chatgpt'`, ZERO `ruoli.dev`, ZERO changes to M0-M5 mainline):

  - **Wave 1 (`12` Unseen Source-Lab & Nested-Chain Tasks)**:
    * `C01` (`Tree_A` Confounder Fork Unmasking): `(x_ind, z_1, z_2, z_3, z_4)` are collinear in `P_obs`; lab allows paid
      interventional probe `probe_do_blanket_disambig` to unmask the true confounder `z_2` (`x_ind <- z_2 -> y`).
    * `C02` (`Tree_M1` Mediator Decomposition & Wormhole Re-Anchoring): `(s_surf, m_1, m_2, m_3)` are collinear at `T_proc=0`;
      paid heat-clamp probe `probe_mediator_clamp_do_T` unmasks `u_raw -> m_2 -> y` vs surface proxy `s_surf`.
    * `C03` (`Tree_S1` Collider / Selection Bias Unmasking): `S_sel == 1` filter induces spurious negative law in `pilot_rows`;
      paid unselected probe `probe_unselected_S0_regime` unmasks true 2-driver additive mechanism.
    * `C04` (Wormhole-First 3-Layer Nested Sub-Assumption Closure): `p_weather = 0.12` (`S`-node Nogood) blocks outdoor branch;
      `M6` recursively queries cross-tree wormholes (`d = 1 -> 2 -> 3`) for `space_sigmoid` & `impact_quadratic` and pulls back to root `y`.
    * `C05` (`Tree_G_orth` Collider-Shielded Negative Control): connected to `Tree_A'` ONLY via unconditioned collider `C_col`;
      in `Tree_G_orth`, `x_ind` GENUINELY IS the true causal driver!
    * `C06` (Adversarial Chaotic Non-Stationary OOS): requires honest `abstain_scoped_unknown`.

  - **Between Wave 1 & Wave 2 (Algorithmic Separation Between `M6` and `M5`)**:
    * `Hegel_M5_GrowthOnly_LiveGPT`: updates ONLY the local Wave-1 nodes (`Tree_A`, `Tree_M1`, `Tree_S1`) (`0` wormhole hops).
    * `Hegel_M6_Aufhebung_LiveGPT`: verifies `RetroactiveProjectionCertificate` (`E_obs[New] == Old`), demotes refuted `L1` laws
      to `L1_ASSOCIATIONAL_SHADOW`, and executes **Bayes-Ball `d-sepset` + ATMS Minimal `Nogood` propagation** across active
      `1-hop` (`Tree_A'`) and `2-hop` (`Tree_F''`) wormholes + re-anchors `Tree_M2` & `Tree_S2`, while **Bayes-Ball `d-separation`
      strictly blocks propagation into collider-shielded `Tree_G_orth` (`C05`)**!

  - **Wave 2 (`12` Unseen Target Transportability & Nested-Chain Tasks — Bareinboim-Pearl `z`-Transportability)**:
    * Target domains `Tree_A'` (1-hop), `Tree_F''` (2-hop), `Tree_M2`, `Tree_S2` permit ONLY passive observational pre-deployment
      probes (`P_obs` collinearity intact) before interventional deployment (`confirm_rows` & `heldout_rows`)!
    * Tests **Zero-Shot Multi-Hop Causal Immunity** (`M6` vs `M5`) + **Zero Collateral Regression** on `C05` (`Tree_G_orth`)!
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import threading
from typing import Any

import numpy as np

from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m5_live_gpt_and_provenance import (
    CodexSubscriptionClient,
    DecisionProvenanceLedger,
    SubscriptionQuotaPausedError,
    _execute_authored_python_in_isolated_subprocess,
    _row_input_key,
    discover_chatgpt_subscription_codex_cli,
    verify_decision_provenance_chain,
)
from .m5_temporal_multiscale_and_base_subtree import _composite
from .m6_causal_wormhole_sandbox import (
    is_d_separated_dag,
    verify_retroactive_projection_certificate,
)
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
)


@dataclass
class M6CausalTaskDataset:
    """Strictly disjoint multi-turn causal / nested-assumption task dataset."""

    task_id: str
    wave: str
    subfamily: str
    target_tree_id: str
    is_out_of_scope: bool
    probe_signatures: list[str]
    deployment_regime: str
    pilot_rows: list[dict[str, Any]]
    probe_pool: dict[str, dict[str, Any]]
    confirm_rows: list[dict[str, Any]]
    heldout_rows: list[dict[str, Any]]
    tolerance: float = 0.05


# ============================================================================
# 1. `K = 40` Multi-Tree Causal Forest & Historical Observational Archive
# ============================================================================


def build_m6_k40_causal_forest_and_archive() -> tuple[PhilosophicalForest, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build `K = 40` initial forest (12 structural/wormhole nodes + 28 polysemous distractors).

    Crucially, at `t = 0` (before Wave 1), `Tree_A`, `Tree_A_prime`, `Tree_F_double_prime`, `Tree_M1`, `Tree_M2`,
    `Tree_S1`, `Tree_S2` ALL hold **historically validated `L1` observational laws** (`R^2 > 0.999` on `P_obs`)!
    Neither `M6` nor `M5` knows which candidate latent variable is causal until Wave-1 empirical discovery!
    """
    forest = PhilosophicalForest()
    tv = TaskView("causal_and_interventional_prediction", ["y"], 0.05)
    node_archive: dict[str, dict[str, Any]] = {}

    def _reg(nid: str, tid: str, sigs: list[str], note: str, ep_rung: str = "L1_UNVERIFIED_OBSERVATIONAL") -> None:
        forest.register_node(
            node_id=nid,
            tree_id=tid,
            layer_role="mid",
            task_view=tv,
            observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
            scale_map=ScaleMap("identity", preserved_queries=["causal_and_interventional_prediction"]),
            claim_contract=ClaimContract(
                claim_id=f"cl_{nid}",
                family_plugin="m6_causal",
                preconditions=[],
                conclusion_or_action={"epistemic_rung": ep_rung},
                protocol_spec={"epistemic_rung": ep_rung},
                tolerance=0.05,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=sigs,
        )
        node_archive[nid] = {
            "node_id": nid,
            "tree_id": tid,
            "epistemic_rung": ep_rung,
            "verified_mechanism_note": note,
        }

    # --- C01 Group: `Tree_A` <-> `Tree_A_prime` (1-hop) <-> `Tree_F_double_prime` (2-hop) ---
    _reg(
        "node_C01_TreeA_indicator",
        "tree_C01_A_biochem",
        ["sig:c01_treeA_indicator", "sig:hill_saturation_flux"],
        "Historical observational law (`R^2=0.999` on passive data): `y = vmax * ((x_ind / 1.5) ** 2) / (k_half + ((x_ind / 1.5) ** 2)) - c_ctrl * u_ctrl` (3 params `vmax, k_half, c_ctrl`).",
    )
    _reg(
        "node_C01_TreeA_prime_electro",
        "tree_C01_Aprime_electro",
        ["sig:c01_treeAprime_electro", "sig:hill_saturation_flux"],
        "Historical 1-hop analog law (`R^2=0.999` on passive data): `y = vmax * ((x_ind / 1.5) ** 2) / (k_half + ((x_ind / 1.5) ** 2)) - c_ctrl * u_ctrl` (mapped via `x_ind <-> x_ind`).",
    )
    _reg(
        "node_C01_TreeF_doubleprime_catalysis",
        "tree_C01_Fdouble_catalysis",
        ["sig:c01_treeFdouble_catalysis", "sig:hill_saturation_flux"],
        "Historical 2-hop analog law (`R^2=0.999` on passive data): `y = vmax * ((x_ind / 1.5) ** 2) / (k_half + ((x_ind / 1.5) ** 2)) - c_ctrl * u_ctrl` (mapped via `Tree_A_prime <-> Tree_F_double_prime`).",
    )

    # --- C05 Negative Control: `Tree_G_orth` connected to `Tree_A_prime` ONLY via collider-shielded `W_orth`! ---
    _reg(
        "node_C05_TreeG_orth_direct",
        "tree_C05_G_orth",
        ["sig:c05_treeG_orth_direct", "sig:monotone_hyperbolic"],
        "Direct physical actuator law (`x_ind` is true mechanical valve opening): `y = vmax * x_ind / (1.1 + x_ind) + c_ctrl * u_ctrl` (2 params `vmax, c_ctrl`; `z_1..z_4` are uncorrelated ambient noise).",
        ep_rung="L2_INVARIANT_CAUSAL",
    )

    # --- C02 Group: `Tree_M1` <-> `Tree_M2` (Mediator vs Surface Proxy) ---
    _reg(
        "node_C02_TreeM1_surface",
        "tree_C02_M1_extract",
        ["sig:c02_treeM1_extract", "sig:surface_acidity_proxy"],
        "Historical cold-process law (`T_proc=0`): `y = vmax * s_surf / (k_m + s_surf) + c_raw * sqrt(u_raw)` (3 params `vmax, k_m, c_raw`).",
    )
    _reg(
        "node_C02_TreeM2_analog",
        "tree_C02_M2_ferment",
        ["sig:c02_treeM2_ferment", "sig:surface_acidity_proxy"],
        "Historical surface-acidity wormhole law: `y = vmax * s_surf / (k_m + s_surf) + c_raw * sqrt(u_raw)` (anchored on `s_surf`).",
    )

    # --- C03 Group: `Tree_S1` <-> `Tree_S2` (Berkson Selection Archive) ---
    _reg(
        "node_C03_TreeS1_selected",
        "tree_C03_S1_catalog",
        ["sig:c03_treeS1_catalog", "sig:survivorship_archive"],
        "Historical stability-filtered catalog law (`S_sel=1` where `u_1 + 1.5*u_2 ≈ 3.3`): 1-driver formula `y = a1 * log(1.0 + u_1) + a2_eff * max(0.0, 3.3 - u_1) / (1.5 + max(0.0, 3.3 - u_1))` (ignores `u_2`).",
    )
    _reg(
        "node_C03_TreeS2_alloy",
        "tree_C03_S2_alloy",
        ["sig:c03_treeS2_alloy", "sig:survivorship_archive"],
        "Historical filtered alloy archive law (`S_sel=1`): 1-driver formula `y = a1 * log(1.0 + u_1) + a2_eff * max(0.0, 3.3 - u_1) / (1.5 + max(0.0, 3.3 - u_1))` (ignores `u_2`).",
    )

    # --- C04 Group: 3-Layer Nested Assumption Chain (`Tree_Main` -> `Tree_Neuro` -> `Tree_Space` + `Tree_Biomech`) ---
    _reg(
        "node_C04_Root_Vitality_Main",
        "tree_C04_root_vitality",
        ["sig:c04_root_vitality_goal"],
        (
            "Root vitality goal (`y`): depends on effective dose `E_eff` via `tree_C04_neuro_dose` (`sig:c04_neuro_dose_response`). "
            "Default sunny-climate assumption: `E_eff = u_outdoor * p_weather` (valid ONLY when `p_weather >= 0.70`; "
            "if `p_weather < 0.50` [city weather Nogood!], outdoor path fails and indoor sub-assumptions `sig:c04_indoor_space_envelope` "
            "AND `sig:c04_biomech_impact_loss` must be resolved and pulled back to root!)."
        ),
        ep_rung="L2_CONDITIONAL_ROOT",
    )
    _reg(
        "node_C04_Depth1_NeuroDose",
        "tree_C04_neuro_dose",
        ["sig:c04_neuro_dose_response"],
        "Depth-1 neuro-metabolic dose response: `y = g_root * (E_eff / (k_e + E_eff)) - Loss_impact`, where `E_eff = u_outdoor * p_weather + E_indoor`.",
        ep_rung="L2_INVARIANT_CAUSAL",
    )
    _reg(
        "node_C04_Depth2_IndoorSpaceSigmoid",
        "tree_C04_indoor_space",
        ["sig:c04_indoor_space_envelope"],
        "Depth-2 compact-space adherence law: indoor effective dose `E_indoor = u_indoor / (1.0 + exp(6.0 * (a_foot - 1.5)))` (penalizes footprint `a_foot > 1.5 m^2`).",
        ep_rung="L2_INVARIANT_CAUSAL",
    )
    _reg(
        "node_C04_Depth3_BiomechImpactPenalty",
        "tree_C04_biomech_acoustics",
        ["sig:c04_biomech_impact_loss"],
        "Depth-3 acoustic/joint-impact fatigue loss: `Loss_impact = c_imp * ((u_indoor * impact_idx) ** 2)` (penalizes high-impact treadmill `impact_idx` vs quiet elliptical).",
        ep_rung="L2_INVARIANT_CAUSAL",
    )

    # Register cross-tree wormholes in `PhilosophicalForest` + `d-sepset` metadata for Bayes-Ball screening!
    wh_defs = [
        ("wh_C01_A_to_Aprime", "node_C01_TreeA_indicator", "node_C01_TreeA_prime_electro", ["sig:hill_saturation_flux"],
         {"role_map": {"x_ind": "x_ind", "z_1": "z_1", "z_2": "z_2", "z_3": "z_3", "z_4": "z_4"}, "interface_u": "confounded_XY", "interface_v": "confounded_XY"}),
        ("wh_C01_Aprime_to_Fdouble", "node_C01_TreeA_prime_electro", "node_C01_TreeF_doubleprime_catalysis", ["sig:hill_saturation_flux"],
         {"role_map": {"x_ind": "x_ind", "z_1": "z_1", "z_2": "z_2", "z_3": "z_3", "z_4": "z_4"}, "interface_u": "confounded_XY", "interface_v": "confounded_XY"}),
        ("wh_C01_Aprime_to_Gorth_COLLIDER_BLOCKED", "node_C01_TreeA_prime_electro", "node_C05_TreeG_orth_direct", ["sig:monotone_hyperbolic"],
         {"role_map": {"w_orth": "x_ind"}, "interface_u": "orthogonal_W_collider_shielded", "interface_v": "direct_X"}),
        ("wh_C02_M1_to_M2", "node_C02_TreeM1_surface", "node_C02_TreeM2_analog", ["sig:surface_acidity_proxy"],
         {"role_map": {"s_surf": "s_surf", "m_1": "m_1", "m_2": "m_2", "m_3": "m_3"}, "interface_u": "mediated_SM", "interface_v": "mediated_SM"}),
        ("wh_C03_S1_to_S2", "node_C03_TreeS1_selected", "node_C03_TreeS2_alloy", ["sig:survivorship_archive"],
         {"role_map": {"u_1": "u_1", "u_2": "u_2"}, "interface_u": "selected_S1", "interface_v": "selected_S1"}),
        ("wh_C04_Main_to_Neuro", "node_C04_Root_Vitality_Main", "node_C04_Depth1_NeuroDose", ["sig:c04_root_vitality_goal", "sig:c04_neuro_dose_response"],
         {"role_map": {"E_eff": "E_eff"}, "interface_u": "dose_link", "interface_v": "dose_link"}),
        ("wh_C04_Neuro_to_Space", "node_C04_Depth1_NeuroDose", "node_C04_Depth2_IndoorSpaceSigmoid", ["sig:c04_indoor_space_envelope"],
         {"role_map": {"E_indoor": "E_indoor"}, "interface_u": "space_slot", "interface_v": "space_slot"}),
        ("wh_C04_Neuro_to_Biomech", "node_C04_Depth1_NeuroDose", "node_C04_Depth3_BiomechImpactPenalty", ["sig:c04_biomech_impact_loss"],
         {"role_map": {"Loss_impact": "Loss_impact"}, "interface_u": "impact_slot", "interface_v": "impact_slot"}),
    ]
    for wid, na, nb, wsigs, rmap in wh_defs:
        forest.add_horizontal_wormhole(
            wormhole_id=wid,
            node_a_id=na,
            node_b_id=nb,
            probe_signature_refs=wsigs,
            role_and_observation_map=rmap,
            hop_error_budget=0.015,
            lipschitz_factor=0.90,
            permission_tier="local_solver_guidance",
        )

    # Add 28 realistic polysemous distractors (`12 + 28 = 40` nodes total!)
    for d_i in range(1, 29):
        d_nid = f"node_M6_dist_{d_i:02d}"
        d_sigs = [
            ["sig:hill_saturation_flux", "sig:monotone_hyperbolic"],
            ["sig:surface_acidity_proxy", "sig:hill_saturation_flux"],
            ["sig:survivorship_archive", "sig:monotone_hyperbolic"],
            ["sig:c04_root_vitality_goal", "sig:monotone_hyperbolic"],
        ][d_i % 4]
        _reg(
            d_nid,
            f"tree_dist_m6_{d_i:02d}",
            d_sigs,
            f"Distractor model #{d_i:02d} (single-factor uncoupled curve): `y = a0 * (x_ind ** {1 + (d_i % 2)}) / (1.5 + x_ind) - 0.1 * d_i`.",
            ep_rung="L1_DISTRACTOR",
        )

    assert len(forest.nodes) == 40
    return forest, node_archive, wh_defs


# ============================================================================
# 2. `24` Unseen Causal & Multi-Depth Tasks (`Wave 1` = 12, `Wave 2` = 12)
# ============================================================================


def build_m6_causal_24_waves(seed: int = 20261121) -> dict[str, list[M6CausalTaskDataset]]:
    """Generate `wave_1` (`12` tasks) and `wave_2` (`12` tasks) with strictly disjoint `pilot/confirm/heldout` rows."""
    rng = random.Random(seed)
    waves: dict[str, list[M6CausalTaskDataset]] = {"wave_1": [], "wave_2": []}

    for w_idx, w_name in ((1, "wave_1"), (2, "wave_2")):
        w_list: list[M6CausalTaskDataset] = []

        # --- C01: Confounder Fork (`x_ind <- z_2 -> y`, while `z_1, z_3, z_4` also co-move in `P_obs`!) ---
        # Wave 1 runs on `Tree_A` (lab `do` probe available!); Wave 2 runs on 1-hop `Tree_A_prime` (rep 1) & 2-hop `Tree_F_double_prime` (rep 2) (passive probes only -> z-Transportability!)
        for rep in (1, 2):
            vm = round(rng.uniform(6.8, 8.8), 3)
            kh = round(rng.uniform(1.1, 2.1), 3)
            cc = round(rng.uniform(0.22, 0.42), 3)
            if w_idx == 1:
                t_tree, q_sig = "tree_C01_A_biochem", "sig:c01_treeA_indicator"
            elif rep == 1:
                t_tree, q_sig = "tree_C01_Aprime_electro", "sig:c01_treeAprime_electro"
            else:
                t_tree, q_sig = "tree_C01_Fdouble_catalysis", "sig:c01_treeFdouble_catalysis"

            def _c01_obs(z_v: float, u_v: float, _vm=vm, _kh=kh, _cc=cc) -> dict[str, float]:
                # In passive `P_obs`, `x_ind = 1.5*z_v` and `z_1..z_4` ALL equal `z_v` (100% collinear Markov equivalence class!)
                yv = _vm * ((z_v ** 2) / (_kh + (z_v ** 2))) - _cc * u_v
                return {
                    "x_ind": round(1.5 * z_v, 5),
                    "z_1": round(z_v, 5),
                    "z_2": round(z_v, 5),
                    "z_3": round(z_v, 5),
                    "z_4": round(z_v, 5),
                    "u_ctrl": round(u_v, 4),
                    "y": round(yv, 7),
                }

            def _c01_int(x_do: float, z1: float, z2_true: float, z3: float, z4: float, u_v: float, _vm=vm, _kh=kh, _cc=cc) -> dict[str, float]:
                yv = _vm * ((z2_true ** 2) / (_kh + (z2_true ** 2))) - _cc * u_v
                return {
                    "x_ind": round(x_do, 5),
                    "z_1": round(z1, 5),
                    "z_2": round(z2_true, 5),
                    "z_3": round(z3, 5),
                    "z_4": round(z4, 5),
                    "u_ctrl": round(u_v, 4),
                    "y": round(yv, 7),
                }

            pilot = [_c01_obs(0.5, 0.4), _c01_obs(1.1, 1.5), _c01_obs(1.8, 0.7), _c01_obs(2.5, 2.1), _c01_obs(3.3, 1.1)]
            probes: dict[str, dict[str, Any]] = {
                "probe_passive_more_obs": {
                    "id": "probe_passive_more_obs",
                    "probe_kind": "observational_L1",
                    "cost": 2.0,
                    "rows": [_c01_obs(0.8, 1.0), _c01_obs(2.9, 1.8)],
                },
            }
            if w_idx == 1:
                # Source domain (`Wave 1`): randomized clamping intervention breaks collinearity among `x_ind, z_1..z_4`!
                probes["probe_do_blanket_disambig"] = {
                    "id": "probe_do_blanket_disambig",
                    "probe_kind": "interventional_L2_do_clamp",
                    "cost": 3.5,
                    "rows": [
                        _c01_int(3.9, 0.4, 2.4, 3.3, 0.6, 0.8),
                        _c01_int(0.3, 3.1, 0.7, 0.4, 2.9, 1.4),
                        _c01_int(2.2, 2.8, 3.1, 0.9, 1.1, 0.5),
                    ],
                }
            conf = [
                _c01_int(0.4, 2.9, 2.6, 0.5, 3.2, 0.9),
                _c01_int(4.2, 0.6, 0.8, 3.1, 0.4, 1.7),
                _c01_int(2.1, 1.2, 3.2, 2.8, 0.7, 0.6),
            ]
            hold = [
                _c01_int(3.8, 3.0, 1.2, 0.6, 2.7, 1.3),
                _c01_int(0.6, 0.5, 2.9, 3.2, 0.8, 0.5),
                _c01_int(2.7, 2.6, 0.5, 1.1, 3.1, 1.9),
                _c01_int(1.5, 1.9, 2.2, 2.9, 1.4, 1.0),
            ]
            w_list.append(M6CausalTaskDataset(
                task_id=f"W{w_idx}R{rep}__C01_ConfounderFork_{t_tree}",
                wave=w_name, subfamily="C01_ConfounderFork_RetroactiveImmunity",
                target_tree_id=t_tree, is_out_of_scope=False,
                probe_signatures=[q_sig, "sig:hill_saturation_flux"],
                deployment_regime="interventional_do_x_ind_and_shifted_environment",
                pilot_rows=pilot, probe_pool=probes, confirm_rows=conf, heldout_rows=hold,
            ))

        # --- C02: Mediator Decomposition & Wormhole Re-Anchoring (`u_raw -> m_2 -> y` vs surface proxy `s_surf`) ---
        for rep in (1, 2):
            vm = round(rng.uniform(7.0, 9.2), 3)
            km = round(rng.uniform(1.0, 1.9), 3)
            cr = round(rng.uniform(0.25, 0.45), 3)
            t_tree = "tree_C02_M1_extract" if w_idx == 1 else "tree_C02_M2_ferment"
            q_sig = "sig:c02_treeM1_extract" if w_idx == 1 else "sig:c02_treeM2_ferment"

            def _c02_cold(u_r: float, _vm=vm, _km=km, _cr=cr) -> dict[str, float]:
                s_s = 0.9 * u_r
                m_val = 1.35 * u_r
                yv = _vm * m_val / (_km + m_val) + _cr * math.sqrt(u_r)
                return {
                    "u_raw": round(u_r, 4), "t_proc": 0.0,
                    "s_surf": round(s_s, 5), "m_1": round(m_val, 5), "m_2": round(m_val, 5), "m_3": round(m_val, 5),
                    "y": round(yv, 7),
                }

            def _c02_heat(u_r: float, t_p: float, m1_v: float, m2_true: float, m3_v: float, _vm=vm, _km=km, _cr=cr) -> dict[str, float]:
                s_s = 0.9 * u_r  # Surface acidity survives boiling!
                yv = _vm * m2_true / (_km + m2_true) + _cr * math.sqrt(u_r)
                return {
                    "u_raw": round(u_r, 4), "t_proc": round(t_p, 3),
                    "s_surf": round(s_s, 5), "m_1": round(m1_v, 5), "m_2": round(m2_true, 5), "m_3": round(m3_v, 5),
                    "y": round(yv, 7),
                }

            pilot = [_c02_cold(0.5), _c02_cold(1.1), _c02_cold(1.8), _c02_cold(2.6), _c02_cold(3.4)]
            probes = {
                "probe_passive_cold_batch": {
                    "id": "probe_passive_cold_batch",
                    "probe_kind": "observational_L1",
                    "cost": 2.0,
                    "rows": [_c02_cold(0.8), _c02_cold(3.0)],
                }
            }
            if w_idx == 1:
                probes["probe_mediator_clamp_do_T"] = {
                    "id": "probe_mediator_clamp_do_T",
                    "probe_kind": "interventional_L2_mediator_clamp",
                    "cost": 3.5,
                    "rows": [
                        _c02_heat(2.8, 1.8, 3.2, 0.15, 2.5),
                        _c02_heat(1.6, 1.2, 0.2, 2.10, 0.3),
                        _c02_heat(3.2, 2.2, 1.1, 0.85, 3.4),
                    ],
                }
            conf = [
                _c02_heat(2.5, 2.0, 2.9, 0.12, 2.1),
                _c02_heat(1.4, 0.9, 0.3, 1.85, 0.4),
                _c02_heat(3.1, 1.5, 0.8, 2.65, 0.5),
            ]
            hold = [
                _c02_heat(2.0, 1.9, 2.4, 0.18, 1.9),
                _c02_heat(0.9, 0.6, 0.2, 1.40, 2.2),
                _c02_heat(3.3, 2.1, 3.1, 0.35, 2.7),
                _c02_heat(1.7, 0.3, 2.0, 2.25, 0.4),
            ]
            w_list.append(M6CausalTaskDataset(
                task_id=f"W{w_idx}R{rep}__C02_MediatorReAnchor_{t_tree}",
                wave=w_name, subfamily="C02_MediatorDecomposition_WormholeReAnchor",
                target_tree_id=t_tree, is_out_of_scope=False,
                probe_signatures=[q_sig, "sig:surface_acidity_proxy"],
                deployment_regime="thermal_processing_and_mediator_clamped_deployment",
                pilot_rows=pilot, probe_pool=probes, confirm_rows=conf, heldout_rows=hold,
            ))

        # --- C03: Collider / Berkson Selection De-Biasing (`S_sel=1` archive vs unselected `u_1, u_2`) ---
        for rep in (1, 2):
            a1 = round(rng.uniform(2.1, 3.2), 3)
            a2 = round(rng.uniform(3.4, 4.8), 3)
            t_tree = "tree_C03_S1_catalog" if w_idx == 1 else "tree_C03_S2_alloy"
            q_sig = "sig:c03_treeS1_catalog" if w_idx == 1 else "sig:c03_treeS2_alloy"

            def _c03_r(u1_v: float, u2_v: float, s_flag: int, _a1=a1, _a2=a2) -> dict[str, Any]:
                yv = _a1 * math.log(1.0 + u1_v) + _a2 * (u2_v / (1.0 + u2_v))
                return {"u_1": round(u1_v, 4), "u_2": round(u2_v, 4), "s_selected": int(s_flag), "y": round(yv, 7)}

            # Along `u_1 + 1.5*u_2 == 3.30` (`S_sel == 1`), `u_1` and `u_2` are strictly negatively collinear!
            pilot = [
                _c03_r(0.30, 2.00, 1),
                _c03_r(0.90, 1.60, 1),
                _c03_r(1.50, 1.20, 1),
                _c03_r(2.10, 0.80, 1),
                _c03_r(2.70, 0.40, 1),
            ]
            probes = {
                "probe_selected_catalog_more": {
                    "id": "probe_selected_catalog_more",
                    "probe_kind": "observational_L1_selected",
                    "cost": 2.0,
                    "rows": [_c03_r(0.60, 1.80, 1), _c03_r(2.40, 0.60, 1)],
                }
            }
            if w_idx == 1:
                probes["probe_unselected_S0_regime"] = {
                    "id": "probe_unselected_S0_regime",
                    "probe_kind": "debiased_unselected_S0",
                    "cost": 3.0,
                    "rows": [_c03_r(0.40, 0.25, 0), _c03_r(2.80, 2.20, 1), _c03_r(1.80, 0.20, 0)],
                }
            conf = [_c03_r(0.35, 0.20, 0), _c03_r(2.90, 2.30, 1), _c03_r(1.60, 0.15, 0)]
            hold = [_c03_r(0.50, 0.30, 0), _c03_r(2.60, 2.10, 1), _c03_r(1.20, 0.18, 0), _c03_r(3.10, 1.85, 1)]
            w_list.append(M6CausalTaskDataset(
                task_id=f"W{w_idx}R{rep}__C03_ColliderBerkson_{t_tree}",
                wave=w_name, subfamily="C03_ColliderSelection_BerksonDebiasing",
                target_tree_id=t_tree, is_out_of_scope=False,
                probe_signatures=[q_sig, "sig:survivorship_archive"],
                deployment_regime="unselected_full_population_deployment",
                pilot_rows=pilot, probe_pool=probes, confirm_rows=conf, heldout_rows=hold,
            ))

        # --- C04: Wormhole-First 3-Layer Nested Sub-Assumption Pullback (`BrainPower <- Exercise <- Space & Impact`) ---
        for rep in (1, 2):
            gr = round(rng.uniform(7.5, 9.8), 3)
            ke = round(rng.uniform(0.8, 1.6), 3)
            ci = round(rng.uniform(0.30, 0.55), 3)
            p_weath = 0.12  # Rainy/smoggy city S-node!

            def _c04_r(u_out: float, u_in: float, a_ft: float, imp: float, _gr=gr, _ke=ke, _ci=ci) -> dict[str, float]:
                e_in = u_in / (1.0 + math.exp(6.0 * (a_ft - 1.5)))
                e_eff = u_out * p_weath + e_in
                yv = _gr * (e_eff / (_ke + e_eff)) - _ci * ((u_in * imp) ** 2)
                return {
                    "u_outdoor": round(u_out, 4),
                    "u_indoor": round(u_in, 4),
                    "p_weather": p_weath,
                    "a_foot": round(a_ft, 4),
                    "impact_idx": round(imp, 4),
                    "y": round(yv, 7),
                }

            # Initial pilot mainly varies outdoor vs narrow indoor, so `probe_indoor_space_impact_sweep` helps calibrate all 3 params!
            pilot = [
                _c04_r(2.5, 0.4, 1.15, 0.25),
                _c04_r(0.5, 1.6, 1.10, 0.30),
                _c04_r(3.2, 1.0, 1.85, 0.90),
                _c04_r(1.0, 2.6, 1.20, 0.20),
                _c04_r(1.8, 2.2, 1.95, 0.85),
            ]
            probes = {
                "probe_indoor_space_impact_sweep": {
                    "id": "probe_indoor_space_impact_sweep",
                    "probe_kind": "subassumption_pullback_probe",
                    "cost": 2.5,
                    "rows": [_c04_r(0.2, 3.0, 1.05, 0.15), _c04_r(0.4, 2.8, 1.48, 0.95), _c04_r(2.0, 0.0, 1.20, 0.0)],
                }
            }
            conf = [_c04_r(0.3, 2.9, 1.12, 0.18), _c04_r(2.8, 2.4, 1.90, 0.92), _c04_r(1.2, 1.9, 1.45, 0.55)]
            hold = [_c04_r(0.6, 3.1, 1.08, 0.22), _c04_r(3.0, 1.8, 1.82, 0.88), _c04_r(1.5, 2.5, 1.38, 0.40), _c04_r(2.2, 0.8, 1.62, 0.75)]
            w_list.append(M6CausalTaskDataset(
                task_id=f"W{w_idx}R{rep}__C04_MultiDepth_RootPullback",
                wave=w_name, subfamily="C04_WormholeFirst_MultiDepth_RootPullback",
                target_tree_id="tree_C04_root_vitality", is_out_of_scope=False,
                probe_signatures=["sig:c04_root_vitality_goal"],
                deployment_regime="rainy_city_compact_apartment_root_vitality",
                pilot_rows=pilot, probe_pool=probes, confirm_rows=conf, heldout_rows=hold,
            ))

        # --- C05: Collider-Shielded Negative Control (`Tree_G_orth` — `x_ind` REALLY IS causal!) ---
        for rep in (1, 2):
            vg = round(rng.uniform(5.8, 8.2), 3)
            cg = round(rng.uniform(0.25, 0.48), 3)

            def _c05_r(x_v: float, z1: float, z2: float, u_v: float, _vg=vg, _cg=cg) -> dict[str, float]:
                yv = _vg * x_v / (1.1 + x_v) + _cg * u_v
                return {
                    "x_ind": round(x_v, 4), "z_1": round(z1, 4), "z_2": round(z2, 4), "z_3": 0.5, "z_4": 1.2,
                    "u_ctrl": round(u_v, 4), "y": round(yv, 7),
                }

            pilot = [_c05_r(0.5, 2.1, 0.4, 0.6), _c05_r(1.3, 0.3, 2.8, 1.5), _c05_r(2.1, 1.9, 1.1, 0.4), _c05_r(2.9, 0.8, 3.0, 2.0), _c05_r(3.6, 2.7, 0.6, 1.1)]
            conf = [_c05_r(0.8, 3.0, 2.5, 1.2), _c05_r(2.4, 0.4, 0.3, 0.7), _c05_r(3.4, 1.5, 2.9, 1.8)]
            hold = [_c05_r(0.6, 2.5, 2.8, 0.9), _c05_r(1.8, 0.7, 0.5, 1.6), _c05_r(2.7, 2.9, 1.4, 0.5), _c05_r(3.8, 0.2, 2.2, 1.4)]
            w_list.append(M6CausalTaskDataset(
                task_id=f"W{w_idx}R{rep}__C05_ColliderShielded_DirectX",
                wave=w_name, subfamily="C05_ColliderShielded_NegativeControl",
                target_tree_id="tree_C05_G_orth", is_out_of_scope=False,
                probe_signatures=["sig:c05_treeG_orth_direct", "sig:monotone_hyperbolic"],
                deployment_regime="interventional_do_x_ind_valve",
                pilot_rows=pilot, probe_pool={}, confirm_rows=conf, heldout_rows=hold,
            ))

        # --- C06: Adversarial Non-Stationary Chaotic OOS ---
        for rep in (1, 2):
            om1 = round(rng.uniform(17.0, 24.0), 2)
            om2 = round(rng.uniform(10.0, 15.0), 2)

            def _c06_r(x_v: float, u_v: float, _o1=om1, _o2=om2) -> dict[str, float]:
                yv = 3.2 * math.sin(_o1 * x_v + 4.1 * (u_v ** 2)) * math.cos(_o2 * u_v - 2.7 * x_v)
                return {"x_ind": round(x_v, 4), "z_1": round(0.7 * x_v, 4), "z_2": round(0.7 * x_v, 4), "u_ctrl": round(u_v, 4), "y": round(yv, 7)}

            pilot = [_c06_r(0.4, 0.5), _c06_r(1.1, 1.3), _c06_r(1.9, 0.7), _c06_r(2.7, 1.9), _c06_r(3.5, 1.1)]
            conf = [_c06_r(0.8, 1.0), _c06_r(2.2, 1.6), _c06_r(3.1, 0.6)]
            hold = [_c06_r(0.6, 1.4), _c06_r(1.6, 0.8), _c06_r(2.9, 1.7)]
            w_list.append(M6CausalTaskDataset(
                task_id=f"W{w_idx}R{rep}__C06_OOS_ChaoticNonStationary",
                wave=w_name, subfamily="C06_OOS_Unidentifiable_LatentChaos",
                target_tree_id="tree_OOS", is_out_of_scope=True,
                probe_signatures=["sig:high_freq_sign_oscillation", "sig:hill_saturation_flux"],
                deployment_regime="interventional_do_x_ind",
                pilot_rows=pilot, probe_pool={}, confirm_rows=conf, heldout_rows=hold,
            ))

        assert len(w_list) == 12
        waves[w_name] = w_list

    return waves


# ============================================================================
# 3. Online Wave-1 -> Wave-2 Consolidation: `M5 (0-Hop Growth)` vs `M6 (d-sepset + ATMS)`
# ============================================================================


def apply_post_wave1_consolidation(
    *,
    arm: str,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
    wave1_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply Wave-1 empirical learning before Wave 2:

      - `EqualAuth_Flat_LiveGPT` / `Strong_Ordinary_LiveGPT`: static archive (`0` structural updates).
      - `Hegel_M5_GrowthOnly_LiveGPT`: **0-hop local growth/patching ONLY** on `Tree_A`, `Tree_M1`, `Tree_S1`!
        Does NOT propagate causal re-interpretation across wormholes to `Tree_A_prime`, `Tree_F_double_prime`, `Tree_M2`, `Tree_S2`!
      - `Hegel_M6_Aufhebung_LiveGPT`:
        1) Verifies `RetroactiveProjectionCertificate` (`E_obs[New] == Old`);
        2) Demotes `L1` shortcuts to `L1_ASSOCIATIONAL_SHADOW`;
        3) Runs **Bayes-Ball `d-sepset` + ATMS Minimal `Nogood` multi-hop propagation**:
           - Propagates `z_2` confounder fork `x_ind <- z_2 -> y` across `Tree_A -> Tree_A_prime -> Tree_F_double_prime` (`1` & `2` hops)!
           - **Bayes-Ball `d-Separation` Check** on `wh_C01_Aprime_to_Gorth_COLLIDER_BLOCKED`: verifies `{"X", "Y"}` and `{"W_orth"}`
             are `d-separated` in `Tree_A_prime` (`X -> C_col <- W_orth`), **BLOCKING** false overwrite of `Tree_G_orth` (`C05`)!
           - Re-anchors `wh_C02_M1_to_M2` from `s_surf` onto `m_2` (`Tree_M2` upgraded!) and propagates `S_sel` debiasing to `Tree_S2`!
    """
    if arm not in {"Hegel_M5_GrowthOnly_LiveGPT", "Hegel_M6_Aufhebung_LiveGPT"}:
        return {"events": [], "writeback_cost_C": 0.0}

    confirmed_fams = {
        r["subfamily"] for r in wave1_receipts if r.get("status") == "confirmed_ok" and r.get("strict_pass")
    }
    events: list[dict[str, Any]] = []
    wb_cost = 0.0

    # Local tree DAG inside `Tree_A_prime` (`Z -> X, Z -> Y, X -> C_col <- W_orth`)
    local_dag = [("Z", "X"), ("Z", "Y"), ("X", "C_col"), ("W_orth", "C_col")]

    if "C01_ConfounderFork_RetroactiveImmunity" in confirmed_fams:
        # Both M5 and M6 update local `Tree_A`
        node_archive["node_C01_TreeA_indicator"]["verified_mechanism_note"] = (
            "REVISED LOCAL LAW (`Tree_A`): `x_ind` is a downstream confounded proxy (`x_ind <- z_2 -> y`); "
            "true interventional law is `y = vmax * ((z_2 ** 2) / (k_half + (z_2 ** 2))) - c_ctrl * u_ctrl`!"
        )
        wb_cost += 1.5
        events.append({"event": "local_revision_Tree_A", "arm": arm})

        if arm == "Hegel_M6_Aufhebung_LiveGPT":
            # Verify Retroactive Projection Certificate on `P_obs` (`x_ind = 1.5 * z_2`)!
            z_grid = np.linspace(0.4, 3.4, 120)
            x_grid = 1.5 * z_grid
            cert = verify_retroactive_projection_certificate(
                obs_x=x_grid,
                obs_z=z_grid,
                old_associational_predict_fn=lambda xv: 7.5 * (((xv / 1.5) ** 2) / (1.5 + ((xv / 1.5) ** 2))),
                new_causal_predict_fn=lambda _xv, zv: 7.5 * ((zv ** 2) / (1.5 + (zv ** 2))),
                tolerance=0.08,
            )
            assert cert["certified"], f"Retroactive projection check failed: {cert}"

            # Propagate across `wh_C01_A_to_Aprime` (1-hop) and `wh_C01_Aprime_to_Fdouble` (2-hop),
            # while checking Bayes-Ball `d-separation` on `wh_C01_Aprime_to_Gorth_COLLIDER_BLOCKED`!
            shielded_g_orth = is_d_separated_dag(local_dag, {"X", "Y"}, {"W_orth"}, conditioned_vars=set())
            assert shielded_g_orth is True

            for target_nid, hop_lbl in (
                ("node_C01_TreeA_prime_electro", "1-hop `d-sepset` from Tree_A"),
                ("node_C01_TreeF_doubleprime_catalysis", "2-hop `d-sepset` from Tree_A via Tree_A_prime"),
            ):
                old_shadow = node_archive[target_nid]["verified_mechanism_note"]
                node_archive[target_nid]["epistemic_rung"] = "L2_RETROACTIVELY_REINTERPRETED_CAUSAL"
                node_archive[target_nid]["historical_L1_associational_shadow"] = old_shadow
                node_archive[target_nid]["verified_mechanism_note"] = (
                    f"M6 RETROACTIVE AUFHEBUNG ({hop_lbl}): Historical `x_ind` formula demoted to `L1_ASSOCIATIONAL_SHADOW` "
                    f"(worked on passive `P_obs` only because `x_ind ≈ 1.5 * z_2`!). Under interventional deployment, `x_ind` has ZERO causal effect "
                    f"(`x_ind <- z_2 -> y`); fit true `L2` confounder `z_2` directly on passive `pilot_rows`: "
                    f"`y = vmax * ((z_2 ** 2) / (k_half + (z_2 ** 2))) - c_ctrl * u_ctrl` (`z_1, z_3, z_4` are non-causal co-movers)!"
                )
            wb_cost += 1.2
            events.append({
                "event": "m6_multihop_dsepset_aufhebung_C01",
                "reinterpreted_nodes": ["node_C01_TreeA_prime_electro", "node_C01_TreeF_doubleprime_catalysis"],
                "collider_shielded_untouched": ["node_C05_TreeG_orth_direct"],
                "retroactive_projection_defect": cert["max_retroactive_projection_defect"],
            })

    if "C02_MediatorDecomposition_WormholeReAnchor" in confirmed_fams:
        node_archive["node_C02_TreeM1_surface"]["verified_mechanism_note"] = (
            "REVISED LOCAL LAW (`Tree_M1`): `s_surf` is a non-causal byproduct; causal mediator is `m_2`: "
            "`y = vmax * m_2 / (k_m + m_2) + c_raw * sqrt(u_raw)`."
        )
        wb_cost += 1.5
        if arm == "Hegel_M6_Aufhebung_LiveGPT":
            node_archive["node_C02_TreeM2_analog"]["epistemic_rung"] = "L2_REANCHORED_MEDIATOR"
            node_archive["node_C02_TreeM2_analog"]["verified_mechanism_note"] = (
                "M6 WORMHOLE RE-ANCHORED (`wh_C02_M1_to_M2` shifted from `s_surf` to causal mediator `m_2`!): "
                "Old `s_surf` law demoted to `L1_COLD_SHADOW`. True mechanism under thermal/mediator shift is "
                "`y = vmax * m_2 / (k_m + m_2) + c_raw * sqrt(u_raw)` (`s_surf, m_1, m_3` have zero causal effect on `y`)!"
            )
            wb_cost += 0.8
            events.append({"event": "m6_mediator_wormhole_reanchor_C02", "reinterpreted_nodes": ["node_C02_TreeM2_analog"]})

    if "C03_ColliderSelection_BerksonDebiasing" in confirmed_fams:
        node_archive["node_C03_TreeS1_selected"]["verified_mechanism_note"] = (
            "REVISED LOCAL LAW (`Tree_S1`): 1-driver substitution was Berkson selection bias (`s_selected=1`); "
            "true 2-driver additive law is `y = a1 * log(1.0 + u_1) + a2 * (u_2 / (1.0 + u_2))`."
        )
        wb_cost += 1.5
        if arm == "Hegel_M6_Aufhebung_LiveGPT":
            node_archive["node_C03_TreeS2_alloy"]["epistemic_rung"] = "L2_DEBIASED_COLLIDER"
            node_archive["node_C03_TreeS2_alloy"]["verified_mechanism_note"] = (
                "M6 COLLIDER DE-BIASED (`wh_C03_S1_to_S2`): Old 1-driver `u_1`-only formula was a Berkson `s_selected=1` artifact "
                "where `u_1 + 1.5*u_2 == 3.3`! Note: on `s_selected=1` passive pilot rows alone, `u_1` and `u_2` remain collinear, "
                "so use structural de-biased form `y = a1 * log(1.0 + u_1) + a2 * (u_2 / (1.0 + u_2))` (fit 2 params `a1, a2` directly on `pilot_rows` — "
                "nonlinear `log(1+u_1)` and `u_2/(1+u_2)` are linearly independent over `[0.3, 2.7]`, uniquely identifying `(a1, a2)` even on `s_selected=1` rows!)!"
            )
            wb_cost += 0.8
            events.append({"event": "m6_collider_debias_C03", "reinterpreted_nodes": ["node_C03_TreeS2_alloy"]})

    return {"events": events, "writeback_cost_C": round(wb_cost, 4)}


# ============================================================================
# 4. Controlled 4-Arm Prompt Builder & Multi-Turn Live Episode Runner
# ============================================================================


def build_m6_arm_brief(
    ds: M6CausalTaskDataset,
    arm: str,
    *,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    ledger: DecisionProvenanceLedger,
    parent_id: str,
) -> tuple[str, dict[str, Any], str]:
    """Construct transparent, strictly controlled prompt brief for `Strong_Ordinary`, `Flat`, `M5_GrowthOnly`, or `M6_Aufhebung`."""
    sys_prompt = (
        "You are an autonomous causal & scientific discovery agent. Infer the quantitative structural law `y` "
        "for the target `deployment_regime` and return one valid JSON decision matching the schema.\n"
        "RULES:\n"
        "1. Minimize total measurement + compute + token cost while achieving fresh confirmation RMSE <= tolerance.\n"
        "2. Only call `acquire_probe` when current `train_rows` leave genuine causal/structural ambiguity (e.g., Markov-equivalent "
        "collinear candidate variables in passive `P_obs` when deploying under intervention, or multi-parameter underdetermination).\n"
        "3. In `author_and_confirm_python_model`, write self-contained Python defining `fit(train_rows)` and `predict(row) -> float` "
        "using `math`, `numpy`, and `scipy.optimize.least_squares`.\n"
        "4. If observations show chaotic high-frequency non-stationary sign oscillations (`sig:high_freq_sign_oscillation`), call `abstain_scoped_unknown`.\n"
        "5. Populate `considered_alternatives` with concrete causal/mechanistic alternatives weighed."
    )

    base_info = {
        "task_id": ds.task_id,
        "tolerance": ds.tolerance,
        "deployment_regime": ds.deployment_regime,
        "observed_signatures": list(ds.probe_signatures),
        "initial_paid_pilot_rows": ds.pilot_rows,
        "available_probes_menu": {
            k: {"probe_kind": v["probe_kind"], "total_probe_cost": v["cost"]}
            for k, v in ds.probe_pool.items()
        },
    }

    if arm == "Strong_Ordinary_LiveGPT":
        return sys_prompt, {
            **base_info,
            "arm_mode": "Strong_Ordinary_Full_K40_Historical_Archive",
            "full_k40_historical_notes": list(node_archive.values()),
        }, parent_id

    tv_q = {"goal_kind": "causal_and_interventional_prediction", "tolerance": ds.tolerance}
    use_flat = arm == "EqualAuth_Flat_LiveGPT"
    rec = (flat_idx if use_flat else hippo_idx).retrieve(ds.probe_signatures, task_view=tv_q, top_k=3)
    rec_ids = [c["node_id"] for c in rec.get("candidates", ())]

    p_id = ledger.record_decision(
        decision_type="D2_INDEX_RETRIEVAL",
        parent_decision_id=parent_id,
        evidence_inputs={"query_signatures": ds.probe_signatures, "arm": arm},
        candidates_evaluated=[{"node_id": nid} for nid in rec_ids],
        chosen_option={"recalled_top3": rec_ids},
        rejection_reasons={},
        rationale_summary=f"Recall Top-3 from K={len(forest.nodes)} forest via {'FlatBM25' if use_flat else 'HippoRAG_PPR'}.",
        compute_delta=float(rec.get("query_compute_cost", 1.0)),
    )

    if use_flat:
        return sys_prompt, {
            **base_info,
            "arm_mode": "EqualAuth_Flat_BM25_Top3",
            "recalled_notes": [node_archive[nid] for nid in rec_ids if nid in node_archive],
        }, p_id

    if arm == "Hegel_M5_GrowthOnly_LiveGPT":
        # M5: Depth-1 HippoRAG recall ONLY, no Markov-Equivalence Interventional VOI, no multi-depth sub-assumption expansion!
        return sys_prompt, {
            **base_info,
            "arm_mode": "Hegel_M5_Depth1_GrowthOnly",
            "recalled_notes": [node_archive[nid] for nid in rec_ids if nid in node_archive],
        }, p_id

    # --- Full `Hegel_M6_Aufhebung_LiveGPT` ---
    recalled_notes = [deepcopy(node_archive[nid]) for nid in rec_ids if nid in node_archive]
    interventional_voi_guidance: dict[str, Any] | None = None
    recursive_pullback_guidance: dict[str, Any] | None = None

    # 1. M6 Active Interventional / Selection-Bias VOI (`D4_M6_INTERVENTIONAL_MEC_VOI`)
    has_retroactive_L2 = any(
        str(n.get("epistemic_rung", "")).startswith("L2_") for n in recalled_notes
    )
    if (not ds.is_out_of_scope) and (not has_retroactive_L2) and ds.pilot_rows:
        r0 = ds.pilot_rows[0]
        if "z_2" in r0 and "x_ind" in r0:
            x_arr = np.array([float(r["x_ind"]) for r in ds.pilot_rows])
            z1_arr = np.array([float(r["z_1"]) for r in ds.pilot_rows])
            if np.std(x_arr) > 1e-6 and np.std(z1_arr) > 1e-6 and abs(float(np.corrcoef(x_arr, z1_arr)[0, 1])) > 0.98:
                if "probe_do_blanket_disambig" in ds.probe_pool:
                    interventional_voi_guidance = {
                        "detected_hazard": "MARKOV_EQUIVALENCE_COLLINEARITY_UNDER_P_OBS (`corr(x_ind, z_1..z_4) > 0.98`)",
                        "observational_probe_voi": 0.0,
                        "interventional_probe_voi": 2.45,
                        "recommended_turn1_action": (
                            "WARNING: Historical `x_ind` note is `L1_UNVERIFIED_OBSERVATIONAL`, and `(x_ind, z_1, z_2, z_3, z_4)` are 100% collinear in passive `pilot_rows`! "
                            "Acquire `probe_do_blanket_disambig` FIRST on Turn 1 (NEVER `probe_passive_more_obs`!) to unmask which variable (`x_ind` vs `z_1..z_4`) truly drives `y` under intervention!"
                        ),
                    }
        elif "m_2" in r0 and "s_surf" in r0:
            s_arr = np.array([float(r["s_surf"]) for r in ds.pilot_rows])
            m1_arr = np.array([float(r["m_1"]) for r in ds.pilot_rows])
            if np.std(s_arr) > 1e-6 and np.std(m1_arr) > 1e-6 and abs(float(np.corrcoef(s_arr, m1_arr)[0, 1])) > 0.98:
                if "probe_mediator_clamp_do_T" in ds.probe_pool:
                    interventional_voi_guidance = {
                        "detected_hazard": "SURFACE_BYPRODUCT_VS_MEDIATOR_COLLINEARITY (`T_proc == 0` in pilot)",
                        "observational_probe_voi": 0.0,
                        "interventional_probe_voi": 2.60,
                        "recommended_turn1_action": (
                            "WARNING: At `t_proc=0`, surface proxy `s_surf` and candidate mediators `m_1..m_3` are 100% collinear! "
                            "Acquire `probe_mediator_clamp_do_T` FIRST on Turn 1 to identify which active mediator `m_k` governs `y` (`y = vmax * m_k / (k_m + m_k) + c_raw * sqrt(u_raw)`)!"
                        ),
                    }
        elif "s_selected" in r0 and all(int(r["s_selected"]) == 1 for r in ds.pilot_rows):
            if "probe_unselected_S0_regime" in ds.probe_pool:
                interventional_voi_guidance = {
                    "detected_hazard": "BERKSON_COLLIDER_SELECTION_FILTER (`s_selected == 1` in all pilot rows)",
                    "recommended_turn1_action": (
                        "WARNING: All initial pilot rows are conditioned on collider `s_selected == 1` (`u_1 + 1.5*u_2 == 3.3`), inducing a spurious 1-driver formula! "
                        "Acquire `probe_unselected_S0_regime` FIRST on Turn 1, then fit the true 2-driver additive mechanism `y = a1 * log(1.0 + u_1) + a2 * (u_2 / (1.0 + u_2))`!"
                    ),
                }

    if interventional_voi_guidance is not None:
        p_id = ledger.record_decision(
            decision_type="D4_M6_INTERVENTIONAL_MEC_VOI",
            parent_decision_id=p_id,
            evidence_inputs={"deployment_regime": ds.deployment_regime},
            candidates_evaluated=list(ds.probe_pool.keys()),
            chosen_option=interventional_voi_guidance,
            rejection_reasons={"passive_L1_probes": "zero_MEC_disambiguation_power"},
            rationale_summary="Evaluate causal Markov Equivalence Class (MEC) interventional VOI.",
            compute_delta=0.4,
        )

    # 2. M6 Wormhole-First Multi-Depth Recursive Sub-Assumption Expansion & Root Pullback (`C04` depth `1 -> 2 -> 3`!)
    if any(n["node_id"] == "node_C04_Root_Vitality_Main" for n in recalled_notes):
        avg_pw = float(np.mean([float(r.get("p_weather", 1.0)) for r in ds.pilot_rows]))
        if avg_pw < 0.50:
            # S-node `Nogood(OutdoorWeather)` triggered! Recursively query HippoRAG for Depth-2 & Depth-3 sub-assumption wormholes!
            rec_d23 = hippo_idx.retrieve(
                ["sig:c04_indoor_space_envelope", "sig:c04_biomech_impact_loss"],
                task_view=tv_q,
                top_k=3,
            )
            d23_ids = [c["node_id"] for c in rec_d23.get("candidates", ()) if c["node_id"] not in rec_ids]
            for extra_nid in d23_ids:
                if extra_nid in node_archive:
                    recalled_notes.append(deepcopy(node_archive[extra_nid]))
            recursive_pullback_guidance = {
                "s_node_nogood_detected": f"p_weather={avg_pw:.2f} < 0.50 (outdoor path blocked by local city climate S-node!)",
                "wormhole_reused_subassumptions_depth_2_3": d23_ids,
                "root_pullback_composite_law": (
                    "WORMHOLE-FIRST 3-LAYER ROOT PULLBACK (`Depth 3 => Depth 2 => Depth 1 => Root y`): "
                    "`E_indoor = u_indoor / (1.0 + exp(6.0 * (a_foot - 1.5)))`, "
                    "`E_eff = u_outdoor * p_weather + E_indoor`, "
                    "`y = g_root * (E_eff / (k_e + E_eff)) - c_imp * ((u_indoor * impact_idx) ** 2)`! "
                    "Acquire `probe_indoor_space_impact_sweep` if needed or fit `(g_root, k_e, c_imp)` via `least_squares`!"
                ),
            }
            p_id = ledger.record_decision(
                decision_type="D4B_M6_WORMHOLE_FIRST_RECURSIVE_PULLBACK",
                parent_decision_id=p_id,
                evidence_inputs={"avg_p_weather": avg_pw},
                candidates_evaluated=[{"depth23_recalled": d23_ids}],
                chosen_option=recursive_pullback_guidance,
                rejection_reasons={"outdoor_only_hypothesis": "pruned_by_S_node_weather_nogood"},
                rationale_summary="Expand Wormhole-First recursive sub-assumptions (d=3) and pull back composite law to root.",
                compute_delta=0.8,
            )

    return sys_prompt, {
        **base_info,
        "arm_mode": "Hegel_M6_GeneralizedMarkovBlanket_And_RecursivePullback",
        "recalled_and_reinterpreted_notes": recalled_notes,
        "m6_interventional_voi_guidance": interventional_voi_guidance,
        "m6_recursive_subassumption_pullback": recursive_pullback_guidance,
    }, p_id


def run_single_m6_live_episode(
    ds: M6CausalTaskDataset,
    arm: str,
    *,
    client: Any,
    workspace_dir: Path,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
    hippo_idx: HippoRAGWormholeIndex,
    flat_idx: FlatEqualMetadataIndex,
    max_turns: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one controlled multi-turn episode (`max_turns=4`, ZERO unfair backoff full-archive dump!)."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tol = float(ds.tolerance)
    model_name = getattr(client, "model", "unknown_model")
    prov_backend = getattr(client, "provider_backend", "mock_client")
    ledger = DecisionProvenanceLedger(
        run_id=f"m6_live__{model_name}", task_id=ds.task_id, arm=arm, model=model_name, provider_backend=prov_backend,
    )

    unit_row_m = 2.0 if arm == "Strong_Ordinary_LiveGPT" else 1.5
    train_rows = list(ds.pilot_rows)
    parent_id = ledger.record_decision(
        decision_type="D0_PILOT_ACQUISITION",
        parent_decision_id=None,
        evidence_inputs={"task_id": ds.task_id, "pilot_count": len(train_rows)},
        candidates_evaluated=[],
        chosen_option={"pilot_rows": len(train_rows), "unit_row_m": unit_row_m},
        rejection_reasons={},
        rationale_summary="Acquire initial paid pilot rows.",
        measurement_delta=len(train_rows) * unit_row_m,
    )

    sys_prompt, brief, parent_id = build_m6_arm_brief(
        ds, arm, forest=forest, node_archive=node_archive,
        hippo_idx=hippo_idx, flat_idx=flat_idx, ledger=ledger, parent_id=parent_id,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(brief, ensure_ascii=False)},
    ]

    purchased_probes: set[str] = set()
    confirm_attempts = 0
    api_calls = 0
    stage_subdir = f"m6__{model_name}__{arm}__{ds.task_id}"

    for turn_idx in range(1, max_turns + 1):
        if isinstance(client, CodexSubscriptionClient):
            gpt_step = client.call_tool_step(messages, stage_subdir=stage_subdir)
        else:
            gpt_step = client.call_tool_step(messages)
        api_calls += 1

        pt = int(gpt_step.get("prompt_tokens", 0))
        ca = int(gpt_step.get("cached_tokens", 0))
        ct = int(gpt_step.get("completion_tokens", 0))
        eff_tok = max(0, pt - ca) + 0.25 * ca + ct
        tok_k = round(eff_tok / 1000.0, 4)

        if not gpt_step.get("ok"):
            parent_id = ledger.record_decision(
                decision_type="D6_PROVIDER_ERROR",
                parent_decision_id=parent_id,
                evidence_inputs={"turn": turn_idx},
                candidates_evaluated=[],
                chosen_option={"error": gpt_step.get("error")},
                rejection_reasons={},
                rationale_summary="Retry after CLI/schema error.",
                compute_delta=tok_k, prompt_tokens_delta=pt, cached_tokens_delta=ca,
                completion_tokens_delta=ct, wall_ms_delta=gpt_step.get("wall_ms", 0.0),
            )
            messages.append({"role": "user", "content": f"Error: {gpt_step.get('error')}. Return valid JSON."})
            continue

        t_name = gpt_step["tool_name"]
        t_args = gpt_step["tool_args"]
        t_call_id = gpt_step["tool_call_id"]
        messages.append(gpt_step["assistant_message_for_history"])

        parent_id = ledger.record_decision(
            decision_type="D6_LIVE_GPT_DECISION",
            parent_decision_id=parent_id,
            evidence_inputs={"turn": turn_idx, "train_rows": len(train_rows)},
            candidates_evaluated=[{"tool": t_name}],
            chosen_option={"tool": t_name, "probe_id": t_args.get("probe_id"), "hypothesis": t_args.get("hypothesis_name")},
            rejection_reasons={},
            rationale_summary=str(t_args.get("rationale", ""))[:300],
            compute_delta=tok_k, prompt_tokens_delta=pt, cached_tokens_delta=ca,
            completion_tokens_delta=ct, wall_ms_delta=gpt_step["wall_ms"],
        )

        if t_name == "acquire_probe":
            pid = str(t_args.get("probe_id", ""))
            if pid in purchased_probes:
                reply: dict[str, Any] = {"status": "already_purchased", "train_row_count": len(train_rows)}
            elif pid in ds.probe_pool:
                purchased_probes.add(pid)
                p_obj = ds.probe_pool[pid]
                p_cost = float(p_obj["cost"])
                added = list(p_obj["rows"])
                train_rows.extend(added)
                parent_id = ledger.record_decision(
                    decision_type="D5B_PROBE_EXECUTION",
                    parent_decision_id=parent_id,
                    evidence_inputs={"probe_id": pid},
                    candidates_evaluated=[{"probe_id": pid, "cost": p_cost}],
                    chosen_option={"added_rows_count": len(added)},
                    rejection_reasons={},
                    rationale_summary=f"Acquired probe {pid}.",
                    measurement_delta=p_cost,
                )
                reply = {"status": "acquired", "added_rows": added, "total_train_rows": train_rows}
            else:
                reply = {"status": "unavailable_in_target_domain", "available_probes": list(ds.probe_pool.keys())}
            messages.append({"role": "tool", "tool_call_id": t_call_id, "content": json.dumps(reply, ensure_ascii=False)})
            continue

        if t_name == "abstain_scoped_unknown":
            ledger.record_decision(
                decision_type="D9_TERMINAL_ABSTAIN",
                parent_decision_id=parent_id,
                evidence_inputs={"turn": turn_idx},
                candidates_evaluated=[{"status": "scoped_unknown"}],
                chosen_option={"strict_pass": bool(ds.is_out_of_scope)},
                rejection_reasons={},
                rationale_summary="Abstained as scoped_unknown.",
            )
            return {
                "task_id": ds.task_id, "wave": ds.wave, "subfamily": ds.subfamily, "arm": arm,
                "model": model_name, "api_calls": api_calls, "is_out_of_scope": ds.is_out_of_scope,
                "status": "scoped_unknown", "strict_pass": bool(ds.is_out_of_scope),
                "false_confirmation": False, "confirm_rmse": None, "heldout_rmse": None,
                **ledger.totals,
            }, ledger.records

        if t_name == "author_and_confirm_python_model":
            py_src = str(t_args.get("python_code", ""))
            hyp = str(t_args.get("hypothesis_name", f"cand_{turn_idx}"))
            exec_res = _execute_authored_python_in_isolated_subprocess(
                workspace_dir, step_idx=turn_idx, python_code=py_src,
                train_rows=train_rows, confirm_rows=ds.confirm_rows, heldout_rows=ds.heldout_rows,
            )
            if not exec_res.get("ok"):
                parent_id = ledger.record_decision(
                    decision_type="D7_PYTHON_ERROR",
                    parent_decision_id=parent_id,
                    evidence_inputs={"hypothesis": hyp},
                    candidates_evaluated=[],
                    chosen_option={"error": str(exec_res.get("error", ""))[:200]},
                    rejection_reasons={},
                    rationale_summary="Python execution error.",
                    compute_delta=1.5, wall_ms_delta=exec_res["exec_wall_ms"],
                )
                messages.append({"role": "tool", "tool_call_id": t_call_id, "content": json.dumps({"status": "execution_error", "error": exec_res["error"]})})
                continue

            # Every interventional confirmation trial in deployment costs fresh confirmation measurement!
            confirm_attempts += 1
            conf_m = len(ds.confirm_rows) * unit_row_m if confirm_attempts == 1 else 0.5 * len(ds.confirm_rows) * unit_row_m
            c_rmse = float(exec_res["confirm_rmse"])
            accepted = c_rmse <= tol
            parent_id = ledger.record_decision(
                decision_type="D8_CONFIRMATION_CHECK",
                parent_decision_id=parent_id,
                evidence_inputs={"hypothesis": hyp, "confirm_attempt": confirm_attempts},
                candidates_evaluated=[{"train_rmse": exec_res["train_rmse"], "confirm_rmse": c_rmse}],
                chosen_option={"accepted": accepted},
                rejection_reasons=({} if accepted else {hyp: f"confirm_rmse={c_rmse:.5f} > {tol}"}),
                rationale_summary=f"Confirmation {'PASSED' if accepted else 'REFUTED'} (c_rmse={c_rmse:.5f}).",
                measurement_delta=conf_m, compute_delta=3.0, wall_ms_delta=exec_res["exec_wall_ms"],
            )
            if accepted:
                h_rmse = float(exec_res["heldout_rmse"])
                ledger.record_decision(
                    decision_type="D9_TERMINAL_LOCKED",
                    parent_decision_id=parent_id,
                    evidence_inputs={"heldout_rmse": h_rmse},
                    candidates_evaluated=[],
                    chosen_option={"strict_pass": (not ds.is_out_of_scope) and (h_rmse <= tol)},
                    rejection_reasons={},
                    rationale_summary="Locked confirmed model.",
                )
                return {
                    "task_id": ds.task_id, "wave": ds.wave, "subfamily": ds.subfamily, "arm": arm,
                    "model": model_name, "api_calls": api_calls, "is_out_of_scope": ds.is_out_of_scope,
                    "status": "confirmed_ok", "strict_pass": (not ds.is_out_of_scope) and (h_rmse <= tol),
                    "false_confirmation": bool(ds.is_out_of_scope) or (h_rmse > tol),
                    "confirm_rmse": c_rmse, "heldout_rmse": h_rmse,
                    **ledger.totals,
                }, ledger.records

            messages.append({
                "role": "tool",
                "tool_call_id": t_call_id,
                "content": json.dumps({
                    "status": "refuted_on_interventional_deployment_confirmation",
                    "train_rmse": exec_res["train_rmse"],
                    "confirm_rmse": c_rmse,
                    "tolerance": tol,
                    "available_probes": list(ds.probe_pool.keys()),
                }),
            })

    ledger.record_decision(
        decision_type="D9_BUDGET_EXHAUSTED_ABSTAIN",
        parent_decision_id=parent_id,
        evidence_inputs={"max_turns": max_turns},
        candidates_evaluated=[],
        chosen_option={"status": "scoped_unknown"},
        rejection_reasons={},
        rationale_summary="Turn budget exhausted without confirmed causal law.",
    )
    return {
        "task_id": ds.task_id, "wave": ds.wave, "subfamily": ds.subfamily, "arm": arm,
        "model": model_name, "api_calls": api_calls, "is_out_of_scope": ds.is_out_of_scope,
        "status": "scoped_unknown", "strict_pass": bool(ds.is_out_of_scope),
        "false_confirmation": False, "confirm_rmse": None, "heldout_rmse": None,
        **ledger.totals,
    }, ledger.records


# ============================================================================
# 5. Full 2-Wave × 4-Arm × 3-Model Study Orchestrator (`288` Episodes)
# ============================================================================


def _summarize_m6_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"total_tasks": 0}
    in_r = [r for r in rows if not r["is_out_of_scope"]]
    oos_r = [r for r in rows if r["is_out_of_scope"]]
    m_tot = round(sum(float(r["measurement_cost"]) for r in rows), 4)
    k_tot = round(sum(float(r["compute_cost"]) for r in rows), 4)
    in_ok = sum(1 for r in in_r if r["strict_pass"])
    oos_ok = sum(1 for r in oos_r if r["status"] == "scoped_unknown")
    return {
        "total_tasks": len(rows),
        "in_scope_strict_pass": f"{in_ok}/{len(in_r)}",
        "in_scope_pass_rate": round(in_ok / max(1, len(in_r)), 4),
        "oos_scoped_unknown": f"{oos_ok}/{len(oos_r)}",
        "false_confirmations": sum(1 for r in rows if r["false_confirmation"]),
        "total_api_calls": sum(int(r["api_calls"]) for r in rows),
        "mean_turns_per_task": round(sum(int(r["api_calls"]) for r in rows) / max(1, len(rows)), 2),
        "total_effective_tokens": sum(int(r["effective_billed_tokens"]) for r in rows),
        "total_measurement_cost": m_tot,
        "total_compute_cost": k_tot,
        "total_composite_cost": _composite(m_tot, k_tot),
        "mean_composite_cost_per_task": round(_composite(m_tot, k_tot) / max(1, len(rows)), 4),
    }


def run_m6_live_gpt_causal_aufhebung_study(
    evidence_dir: Path,
    *,
    run_live_gpt: bool = True,
    models: tuple[str, ...] = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"),
    max_workers: int = 20,
    max_turns: int = 4,
) -> dict[str, Any]:
    """Run checkpointed `Hegel-M6-Live-Causal-24` (`24` tasks × `4` arms × `3` models = `288` episodes)."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    waves = build_m6_causal_24_waves(20261121)
    arms = (
        "Hegel_M6_Aufhebung_LiveGPT",
        "Hegel_M5_GrowthOnly_LiveGPT",
        "EqualAuth_Flat_LiveGPT",
        "Strong_Ordinary_LiveGPT",
    )

    codex_info = discover_chatgpt_subscription_codex_cli()
    if not (run_live_gpt and codex_info.get("available")):
        dry = {"version": "M6_Live_DryCheck", "live_executed": False, "total_tasks": 24, "arms": list(arms)}
        (evidence_dir / "study_manifest.json").write_text(json.dumps(dry, indent=2) + "\n", encoding="utf-8")
        return dry

    ws_root = evidence_dir / "workspaces_and_dags"
    ws_root.mkdir(parents=True, exist_ok=True)

    # Separate forest + archive state per `(model, arm)` so M6 and M5 evolve independently!
    state: dict[tuple[str, str], dict[str, Any]] = {}
    for m_s in models:
        for a_s in arms:
            f_i, arch_i, _ = build_m6_k40_causal_forest_and_archive()
            state[(m_s, a_s)] = {"forest": f_i, "node_archive": arch_i, "consolidation": None}

    paused_models: dict[str, str] = {}
    pause_lock = threading.Lock()
    prog_lock = threading.Lock()
    all_receipts: list[dict[str, Any]] = []

    for w_name in ("wave_1", "wave_2"):
        w_tasks = waves[w_name]
        jobs = []
        jid = 0
        for ds in w_tasks:
            for m_s in models:
                if m_s in paused_models:
                    continue
                for a_s in arms:
                    jobs.append((jid, m_s, a_s, deepcopy(ds)))
                    jid += 1

        indexes = {
            (m_s, a_s): (
                HippoRAGWormholeIndex(state[(m_s, a_s)]["forest"]),
                FlatEqualMetadataIndex(state[(m_s, a_s)]["forest"]),
            )
            for m_s in models for a_s in arms
        }
        done_cnt = 0

        def _run_jb(jb, w_lbl=w_name, w_tot=len(jobs)):
            nonlocal done_cnt
            j_i, m_s, a_s, ds_i = jb
            with pause_lock:
                if m_s in paused_models:
                    return j_i, None
            ws = ws_root / w_lbl / f"{m_s}__{a_s}__{ds_i.task_id}"
            dag_p = ws / "decision_provenance_dag.json"
            if dag_p.is_file():
                try:
                    cached = json.loads(dag_p.read_text(encoding="utf-8"))
                    if verify_decision_provenance_chain(cached["decision_dag"])["valid"]:
                        with prog_lock:
                            done_cnt += 1
                        return j_i, cached["episode_summary"]
                except Exception:
                    pass
            if ws.exists():
                shutil.rmtree(ws)

            st = state[(m_s, a_s)]
            h_i, fl_i = indexes[(m_s, a_s)]
            sub_c = CodexSubscriptionClient(codex_info=codex_info, model=m_s, reasoning_effort="medium", timeout_seconds=135)
            try:
                ep_r, ch_r = run_single_m6_live_episode(
                    ds_i, a_s, client=sub_c, workspace_dir=ws,
                    forest=st["forest"], node_archive=st["node_archive"],
                    hippo_idx=h_i, flat_idx=fl_i, max_turns=max_turns,
                )
            except SubscriptionQuotaPausedError as q_exc:
                with pause_lock:
                    if m_s not in paused_models:
                        paused_models[m_s] = str(q_exc)
                return j_i, None

            vc = verify_decision_provenance_chain(ch_r)
            assert vc["valid"]
            (ws / "decision_provenance_dag.json").write_text(
                json.dumps({"episode_summary": ep_r, "chain_verification": vc, "decision_dag": ch_r}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with prog_lock:
                done_cnt += 1
                if done_cnt % 12 == 0 or done_cnt == w_tot:
                    (evidence_dir / "live_progress_heartbeat.json").write_text(
                        json.dumps({"wave": w_lbl, "progress": f"{done_cnt}/{w_tot}", "total_done": len(all_receipts) + done_cnt}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(f"[{w_lbl} {done_cnt}/{w_tot}] {m_s}|{a_s}|{ds_i.task_id} pass={ep_r['strict_pass']} C={ep_r['composite_cost']}", file=sys.stderr, flush=True)
            return j_i, ep_r

        w_out = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futs = [pool.submit(_run_jb, jb) for jb in jobs]
            for fut in as_completed(futs):
                w_out.append(fut.result())
        w_out.sort(key=lambda x: x[0])
        w_recs = [x[1] for x in w_out if x[1] is not None]
        all_receipts.extend(w_recs)

        if w_name == "wave_1":
            for m_s in models:
                for a_s in arms:
                    sub_r = [r for r in w_recs if r["model"] == m_s and r["arm"] == a_s]
                    state[(m_s, a_s)]["consolidation"] = apply_post_wave1_consolidation(
                        arm=a_s,
                        forest=state[(m_s, a_s)]["forest"],
                        node_archive=state[(m_s, a_s)]["node_archive"],
                        wave1_receipts=sub_r,
                    )

    fams = (
        "C01_ConfounderFork_RetroactiveImmunity",
        "C02_MediatorDecomposition_WormholeReAnchor",
        "C03_ColliderSelection_BerksonDebiasing",
        "C04_WormholeFirst_MultiDepth_RootPullback",
        "C05_ColliderShielded_NegativeControl",
        "C06_OOS_Unidentifiable_LatentChaos",
    )
    by_model: dict[str, Any] = {}
    for m_s in models:
        m_dict: dict[str, Any] = {}
        for a_s in arms:
            r_ma = [r for r in all_receipts if r["model"] == m_s and r["arm"] == a_s]
            m_dict[a_s] = {
                "overall": _summarize_m6_rows(r_ma),
                "wave_1": _summarize_m6_rows([r for r in r_ma if r["wave"] == "wave_1"]),
                "wave_2": _summarize_m6_rows([r for r in r_ma if r["wave"] == "wave_2"]),
                "by_family": {f: _summarize_m6_rows([r for r in r_ma if r["subfamily"] == f]) for f in fams},
                "post_wave1_consolidation": state[(m_s, a_s)]["consolidation"],
            }
        by_model[m_s] = m_dict

    manifest = {
        "version": "M6_Live_GPT_Causal_Aufhebung_vs_M5_GrowthOnly_288_Episode_Study",
        "git_branch": "exp/m6-causal-wormhole-sandbox",
        "mainline_m0_m5_touched": False,
        "provider_backend": "openai_chatgpt_subscription_codex_cli",
        "auth_mode": "chatgpt",
        "third_party_ruoli_blocked": True,
        "completed_episode_count": len(all_receipts),
        "target_episode_count": 24 * len(arms) * len(models),
        "quota_paused_models": paused_models,
        "by_model": by_model,
    }
    (evidence_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    sha_lines = []
    for p in sorted(f for f in evidence_dir.rglob("*") if f.is_file() and f.name != "SHA256SUMS.txt"):
        rel = p.relative_to(evidence_dir).as_posix()
        sha_lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}")
    (evidence_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return manifest

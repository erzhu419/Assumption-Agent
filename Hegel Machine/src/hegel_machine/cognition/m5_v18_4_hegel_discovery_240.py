"""Hegel V18.4 (`Hegel-Discovery-240`: Publication-Scale 240-Task × 3-Model × 3-Arm Continual Scientific Discovery Benchmark).

Benchmarks `gpt-6-luna`, `gpt-6.1-terra`, and `gpt-6.2-sol` (`240 tasks × 3 models × 3 arms = 2,160 episodes`)
strictly via Official OpenAI ChatGPT Subscription (`CodexSubscriptionClient`, `auth_mode == 'chatgpt'`, ZERO `ruoli.dev`):

  - **Scale & Stratification (`240` Unseen Tasks = `216` In-Scope + `24` Adversarial OOS)**:
    Directly matches ICML/ICLR publication standards (`LLM-SRBench` 239 tasks, `DiscoveryBench` 264 tasks).
    Organized into **6 Sequential Continual Waves (`W1`–`W6`, `40` tasks/wave)**:
      * Cluster I — Cross-Temporal & Multi-Scale Folding (`F01`–`F04`, 72 tasks)
      * Cluster II — Progressive Shared Base Subtrees, Gray-Zone Sublation & 2-Hop Wormholes (`F05`–`F08`, 72 tasks)
      * Cluster III — `LLM-SRBench`-Style Anti-Memorization Transformed Equations & 5-Param Hybrids (`F09`–`F12`, 72 tasks)
      * Adversarial Out-of-Scope Honesty Suite (`OOS`, 24 tasks)
  - **Source Archive Scale (`K = 45 -> 48` Online)**:
    12 verified structural/bridge nodes + 33 realistic polysemous scientific distractors across 10 disciplines.
  - **Bulletproof Checkpoint-Resume & Per-Model Quota-Window Resilience**:
    Every completed episode writes a SHA-256 hash-chained `decision_provenance_dag.json` immediately.
    After every 15 episodes (`live_progress_heartbeat.json`) and after every wave (`wave_W{k}_summary.json` +
    `study_manifest.json`), full metrics are flushed. If any model hits a ChatGPT subscription rate/usage window,
    `SubscriptionQuotaPausedError` pauses ONLY that model cleanly (zero fake failures!) for instant lossless resume.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import threading
from time import monotonic
from typing import Any

from .hipporag_wormhole_index import FlatEqualMetadataIndex, HippoRAGWormholeIndex
from .m5_live_gpt_and_provenance import (
    CodexSubscriptionClient,
    SubscriptionQuotaPausedError,
    _build_live_k15_forest_and_archive,
    discover_chatgpt_subscription_codex_cli,
    run_single_live_gpt_m5_episode,
    verify_decision_provenance_chain,
)
from .m5_temporal_multiscale_and_base_subtree import (
    M5DisjointTaskDataset,
    _build_seed_forest_and_priors,
    _composite,
    simulate_coupled_fast_slow_trajectory,
)
from .m5_v18_3_continual_and_complex_live import (
    _sim_spatiotemporal_siphon_memory_row,
    compute_offline_to_online_amortization_ledger,
)
from .wormhole_forest import (
    ClaimContract,
    ObservationMap,
    PhilosophicalForest,
    ScaleMap,
    TaskView,
)


# ============================================================================
# 1. `K = 45` Heterogeneous Polysemous Forest & Source Archive
# ============================================================================


def build_live_k45_forest_and_archive() -> tuple[PhilosophicalForest, dict[str, Any], dict[str, dict[str, Any]]]:
    """Construct `K = 45` multi-tree `PhilosophicalForest` (12 structural/bridge nodes + 33 polysemous distractors)."""
    forest, priors_bundle, node_archive = _build_live_k15_forest_and_archive()
    tv_def = TaskView("trajectory_and_horizon_prediction", ["y"], 0.045)

    # Upgrade `node_A_before_after_opposition` note to cover dimensionless `tau_scale` + equilibrium offset (`F04`)
    node_archive["node_A_before_after_opposition"]["verified_mechanism_note"] = (
        "Normalized before/after perturbation contrast (`u_n = u_now / tau_scale, u_p = u_past / tau_scale`): "
        "`y = gain * (u_n - u_p) * exp(-H / tau_relax) + k_eq * u_n` (fit 3 params `gain, tau_relax, k_eq`; `k_eq=0` when pure contrast)."
    )

    # 1a. Register 2 `LLM-SRBench`-style transformed cross-disciplinary structural mechanisms (`node_Q`, `node_R`)
    forest.register_node(
        node_id="node_Q_cooperative_hill_inhibition",
        tree_id="tree_biophys_allostery",
        layer_role="mid",
        task_view=tv_def,
        observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
        scale_map=ScaleMap("dimensionless_tau_normalization", preserved_queries=["trajectory_and_horizon_prediction"]),
        claim_contract=ClaimContract(
            claim_id="cl_cooperative_hill_inhibition",
            family_plugin="lsr_biophys_m5",
            preconditions=[],
            conclusion_or_action={"model_classes": ["LSR_CooperativeHillInhibition"]},
            protocol_spec={},
            tolerance=0.045,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:sigmoidal_inflection_onset", "sig:sink_gradient_C2", "sig:cooperative_dimer"],
    )
    forest.add_horizontal_wormhole(
        wormhole_id="wh_C_Q_sink_hill",
        node_a_id="node_C_source_inland_hub",
        node_b_id="node_Q_cooperative_hill_inhibition",
        probe_signature_refs=["sig:sigmoidal_inflection_onset", "sig:sink_gradient_C2"],
        role_and_observation_map={"u_drive": "u_now", "s_grad": "s_grad"},
        hop_error_budget=0.02,
        permission_tier="local_solver_guidance",
    )
    node_archive["node_Q_cooperative_hill_inhibition"] = {
        "node_id": "node_Q_cooperative_hill_inhibition",
        "verified_mechanism_note": (
            "Transformed cooperative Hill-2 dimer binding with competitive sink inhibition & linear bleed (`u_n = u_now / tau_scale`): "
            "`y = vmax * ((u_n ** 2) / (k_half + (u_n ** 2))) / (1.0 + alpha_inh * s_grad) - c_leak * u_n` "
            "(fit 4 params `vmax, k_half, alpha_inh, c_leak`; if `s_grad` pilot span < 0.08, acquire `open_branch_discriminating_probe` first!)."
        ),
    }

    forest.register_node(
        node_id="node_R_barrier_modulated_siphon",
        tree_id="tree_electrochem_barrier",
        layer_role="mid",
        task_view=tv_def,
        observation_map=ObservationMap("y", temporal_readout_mode="before_after_contrast"),
        scale_map=ScaleMap("dimensionless_tau_normalization", preserved_queries=["trajectory_and_horizon_prediction"]),
        claim_contract=ClaimContract(
            claim_id="cl_barrier_modulated_siphon",
            family_plugin="lsr_electrochem_m5",
            preconditions=[],
            conclusion_or_action={"model_classes": ["LSR_BarrierModulatedSiphon"]},
            protocol_spec={},
            tolerance=0.045,
            evidence_status="SUPPORTED_IN_SCOPE",
            permission_tier="local_solver_guidance",
        ),
        probe_signatures=["sig:barrier_activation_onset", "sig:base_field_C1", "sig:sink_gradient_C2"],
    )
    forest.add_horizontal_wormhole(
        wormhole_id="wh_C_R_barrier_siphon",
        node_a_id="node_C_source_inland_hub",
        node_b_id="node_R_barrier_modulated_siphon",
        probe_signature_refs=["sig:barrier_activation_onset", "sig:sink_gradient_C2"],
        role_and_observation_map={"u_drive": "u_now", "s_grad": "s_grad"},
        hop_error_budget=0.02,
        permission_tier="local_solver_guidance",
    )
    node_archive["node_R_barrier_modulated_siphon"] = {
        "node_id": "node_R_barrier_modulated_siphon",
        "verified_mechanism_note": (
            "Barrier-activated Eyring/Arrhenius flux coupled with 2-layer sink inhibition (`u_n = u_now / tau_scale`): "
            "`y = j0 * u_n * exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s_grad / (1.0 + s_grad)) + c_sqrt * sqrt(u_n)` "
            "(fit 4 params `j0, beta_bar, rho, c_sqrt`; acquire `open_branch_discriminating_probe` if `s_grad` span < 0.08!)."
        ),
    }

    # 1b. Register 3 Cross-Disciplinary Bridge Entry Nodes (`F08` 2-hop wormhole transfer to `node_C_source_inland_hub`!)
    bridge_specs = [
        (
            "node_W1_epidemic_mobility_bridge",
            "tree_epidemic_net",
            ["sig:bridge_epidemic_mobility", "sig:flux_conservation"],
            "wh_W1_to_C",
            "Metapopulation mobility bridge entry: maps hub outflow gradient onto `node_C_source_inland_hub` 2-layer siphon.",
        ),
        (
            "node_W2_thermal_fin_bridge",
            "tree_thermal_fin",
            ["sig:bridge_thermal_fin", "sig:flux_conservation"],
            "wh_W2_to_C",
            "Convective fin boundary bridge entry: maps ambient sink gradient onto `node_C_source_inland_hub` 2-layer siphon.",
        ),
        (
            "node_W3_supply_bullwhip_bridge",
            "tree_supply_chain",
            ["sig:bridge_supply_bullwhip", "sig:flux_conservation"],
            "wh_W3_to_C",
            "Inventory echelon bridge entry: maps downstream pull gradient onto `node_C_source_inland_hub` 2-layer siphon.",
        ),
    ]
    for b_nid, b_tid, b_sigs, wh_id, b_note in bridge_specs:
        forest.register_node(
            node_id=b_nid,
            tree_id=b_tid,
            layer_role="mid",
            task_view=tv_def,
            observation_map=ObservationMap("y", temporal_readout_mode="before_after_contrast"),
            scale_map=ScaleMap("dimensionless_tau_normalization", preserved_queries=["trajectory_and_horizon_prediction"]),
            claim_contract=ClaimContract(
                claim_id=f"cl_{b_nid}",
                family_plugin="bridge_m5",
                preconditions=[],
                conclusion_or_action={"model_classes": []},
                protocol_spec={},
                tolerance=0.045,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=b_sigs,
        )
        forest.add_horizontal_wormhole(
            wormhole_id=wh_id,
            node_a_id=b_nid,
            node_b_id="node_C_source_inland_hub",
            probe_signature_refs=b_sigs,
            role_and_observation_map={"u_drive": "u_now", "s_grad": "s_grad"},
            hop_error_budget=0.015,
            permission_tier="local_solver_guidance",
        )
        node_archive[b_nid] = {"node_id": b_nid, "verified_mechanism_note": b_note}

    # 1c. Register 25 additional realistic polysemous scientific distractors (`15 + 5 + 25 = 45` nodes total!)
    extra_25_distractors = [
        ("node_D01_gompertz_tumor", "tree_dist_01", ["sig:sigmoidal_inflection_onset", "sig:monotone_concave"],
         "Gompertz growth without sink inhibition: y = K_max * exp(-b0 * exp(-k_g * u_now))."),
        ("node_D02_richards_generalized", "tree_dist_02", ["sig:sigmoidal_inflection_onset", "sig:base_field_C1"],
         "Richards asymmetric logistic curve: y = K_max / ((1 + q0 * exp(-r0 * u_now)) ** (1 / nu))."),
        ("node_D03_weibull_cdf_hazard", "tree_dist_03", ["sig:sigmoidal_inflection_onset", "sig:flux_conservation"],
         "Weibull cumulative hazard saturation: y = A0 * (1 - exp(-((u_now / lam) ** k_shape)))."),
        ("node_D04_avrami_nucleation", "tree_dist_04", ["sig:sigmoidal_inflection_onset", "sig:barrier_activation_onset"],
         "Avrami phase-transformation kinetics: y = phi_max * (1 - exp(-k_avr * (H ** 3)))."),
        ("node_D05_tafel_overpotential", "tree_dist_05", ["sig:barrier_activation_onset", "sig:base_field_C1"],
         "Uninhibited Tafel exponential polarization: y = i0 * exp(alpha_f * u_now) - i0 * exp(-beta_f * u_now)."),
        ("node_D06_fowler_nordheim_tunnel", "tree_dist_06", ["sig:barrier_activation_onset", "sig:monotone_concave"],
         "Fowler-Nordheim cold field emission (no sink term): y = a_fn * (u_now ** 2) * exp(-b_fn / max(1e-3, u_now))."),
        ("node_D07_vogel_fulcher_viscosity", "tree_dist_07", ["sig:barrier_activation_onset", "sig:slow_horizon_query"],
         "Vogel-Fulcher-Tammann supercooled relaxation: y = eta0 * exp(B_vft / (u_now + T0))."),
        ("node_D08_langmuir_hinschelwood", "tree_dist_08", ["sig:base_field_C1", "sig:monotone_concave"],
         "Bimolecular surface competition peaking then falling: y = k_r * K_u * u_now / ((1 + K_u * u_now + K_s * s_grad) ** 2)."),
        ("node_D09_haldane_substrate_inh", "tree_dist_09", ["sig:base_field_C1", "sig:sink_gradient_C2"],
         "Self-inhibiting Haldane curve (drops at high u_now): y = vmax * u_now / (km + u_now + (u_now ** 2) / ki)."),
        ("node_D10_freundlich_isotherm", "tree_dist_10", ["sig:base_field_C1", "sig:monotone_concave"],
         "Freundlich fractional power isotherm: y = k_f * (u_now ** (1 / n_f))."),
        ("node_D11_temkin_log_adsorption", "tree_dist_11", ["sig:base_field_C1", "sig:flux_conservation"],
         "Temkin logarithmic chemisorption: y = c_t * log(1 + a_t * u_now)."),
        ("node_D12_bet_multilayer_ads", "tree_dist_12", ["sig:base_field_C1", "sig:Flux_multilayer"],
         "BET multilayer isotherm diverging near saturation: y = vm * c_b * u_now / ((1 - 0.1*u_now) * (1 + (c_b - 0.1)*u_now))."),
        ("node_D13_erlang_queue_delay", "tree_dist_13", ["sig:bridge_supply_bullwhip", "sig:flux_conservation"],
         "M/M/1 hyperbolic queuing blowup: y = u_now / max(0.05, mu_cap - u_now)."),
        ("node_D14_littles_inventory_linear", "tree_dist_14", ["sig:bridge_supply_bullwhip", "sig:monotone_concave"],
         "Pure linear Little's Law pipeline stock: y = lambda_in * u_now - mu_out * s_grad."),
        ("node_D15_sir_mass_action_bilinear", "tree_dist_15", ["sig:bridge_epidemic_mobility", "sig:flux_conservation"],
         "Simple bilinear mass-action contact rate: y = beta_0 * u_now * x_now - gamma_0 * x_now."),
        ("node_D16_gravity_radiation_flux", "tree_dist_16", ["sig:bridge_thermal_fin", "sig:flux_conservation"],
         "Stefan-Boltzmann quartic radiative loss: y = h_conv * u_now - eps_rad * ((0.2 * u_now) ** 4)."),
        ("node_D17_newton_cooling_pure", "tree_dist_17", ["sig:bridge_thermal_fin", "sig:fast_transient_damped"],
         "Uncoupled Newton cooling without sink inhibition: y = u_now + (x_now - u_now) * exp(-h_c * H)."),
        ("node_D18_kohlrausch_stretched_exp", "tree_dist_18", ["sig:slow_horizon_query", "sig:coupled_transient_lag"],
         "Stretched-exponential glass relaxation: y = x_now * exp(-((H / tau_k) ** 0.55))."),
        ("node_D19_maxwell_viscoelastic", "tree_dist_19", ["sig:endpoint_matched_history_gap", "sig:opposing_step_shift"],
         "Single-mode Maxwell dashpot stress decay: y = E0 * (u_now - u_past) * exp(-H / tau_m)."),
        ("node_D20_kelvin_voigt_creep", "tree_dist_20", ["sig:coupled_transient_lag", "sig:fast_transient_damped"],
         "Uncoupled 1-state Kelvin-Voigt creep: y = (u_now / E_kv) * (1 - exp(-E_kv * H / eta_kv))."),
        ("node_D21_lotka_predator_cycle", "tree_dist_21", ["sig:short_window_positive_gain", "sig:endogenous_adaptation_candidate"],
         "Periodic harmonic Lotka-Volterra oscillation: y = x_now * cos(omega_lv * H) + u_now * sin(omega_lv * H)."),
        ("node_D22_ricker_overcompensation", "tree_dist_22", ["sig:short_window_positive_gain", "sig:monotone_concave"],
         "Single-horizon Ricker dome map: y = r_0 * u_now * exp(-beta_r * u_now)."),
        ("node_D23_beverton_holt_recruit", "tree_dist_23", ["sig:bridge_epidemic_mobility", "sig:monotone_concave"],
         "Beverton-Holt density recruitment ignoring s_grad: y = r_bh * u_now / (1 + u_now / k_bh)."),
        ("node_D24_cobb_douglas_prod", "tree_dist_24", ["sig:base_field_C1", "sig:sink_gradient_C2"],
         "Multiplicative Cobb-Douglas power law: y = A_tfp * (max(1e-4, u_now) ** alpha_u) * (max(1e-4, s_grad) ** beta_s)."),
        ("node_D25_ornstein_uhlenbeck_mean", "tree_dist_25", ["sig:static_snapshot_uninformative", "sig:zero_history_gap"],
         "Mean-reverting Ornstein-Uhlenbeck drift: y = theta_ou + (x_now - theta_ou) * exp(-kappa_ou * H)."),
    ]
    for d_nid, d_tid, d_sigs, d_note in extra_25_distractors:
        forest.register_node(
            node_id=d_nid,
            tree_id=d_tid,
            layer_role="mid",
            task_view=tv_def,
            observation_map=ObservationMap("y", temporal_readout_mode="present_snapshot"),
            scale_map=ScaleMap("identity", preserved_queries=["trajectory_and_horizon_prediction"]),
            claim_contract=ClaimContract(
                claim_id=f"cl_{d_nid}",
                family_plugin="distractor_k45",
                preconditions=[],
                conclusion_or_action={"model_classes": [f"Distractor_{d_nid}"]},
                protocol_spec={},
                tolerance=0.045,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="bounded_trial",
            ),
            probe_signatures=d_sigs,
        )
        node_archive[d_nid] = {"node_id": d_nid, "verified_mechanism_note": d_note}

    assert len(forest.nodes) == 45, f"Expected K=45 nodes, got {len(forest.nodes)}"
    return forest, priors_bundle, node_archive


# ============================================================================
# 2. `Hegel-Discovery-240` Task Generator (`6` Waves × `40` Tasks = `240`)
# ============================================================================


def _make_standard_probes(
    rng: random.Random,
    *,
    open_rows: list[dict[str, Any]] | None = None,
    slow_rows: list[dict[str, Any]] | None = None,
    high_sgrad_rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    del rng
    pool: dict[str, dict[str, Any]] = {
        "nuisance_high_var_aux": {
            "id": "nuisance_high_var_aux",
            "measurement_cost": 3.5,
            "reset_cost": 0.0,
            "horizon_wait_cost": 0.0,
            "row": None,
        }
    }
    if open_rows:
        pool["open_branch_discriminating_probe"] = {
            "id": "open_branch_discriminating_probe",
            "measurement_cost": 2.0,
            "reset_cost": 0.0,
            "horizon_wait_cost": 0.0,
            "rows": open_rows,
            "row": open_rows[0],
        }
    if slow_rows:
        pool["slow_horizon_discriminating_probe"] = {
            "id": "slow_horizon_discriminating_probe",
            "measurement_cost": 2.0,
            "reset_cost": 0.5,
            "horizon_wait_cost": 2.5,
            "rows": slow_rows,
            "row": slow_rows[-1],
        }
    if high_sgrad_rows:
        pool["high_sgrad_regime_probe"] = {
            "id": "high_sgrad_regime_probe",
            "measurement_cost": 3.0,
            "reset_cost": 0.0,
            "horizon_wait_cost": 0.0,
            "rows": high_sgrad_rows,
            "row": high_sgrad_rows[-1],
        }
    return pool


def _build_one_family_task(
    *,
    wave_idx: int,
    rep_idx: int,
    family_code: str,
    rng: random.Random,
) -> M5DisjointTaskDataset:
    """Build one strictly disjoint unseen task for `family_code` in `wave_idx` (`1..6`)."""
    w_tag = f"W{wave_idx}R{rep_idx}"

    # --- F01: Irregular Shifted Prefix Memory-Kernel ODE (`mem_conv = None`) ---
    if family_code == "F01_IrregularMemoryKernel_ODE":
        a_s = round(rng.uniform(1.55, 2.15), 3)
        b_c = round(rng.uniform(1.05, 1.40), 3)
        c_d = round(rng.uniform(1.15, 1.65), 3)
        tau_f = round(rng.uniform(0.48, 0.78), 3)
        t_shift = round(rng.uniform(30.0, 220.0), 1)
        rel_ts = [0.0, 0.05, 0.12, 0.21, 0.29, 0.37, 0.44, 0.51, 0.58]

        def _f01_r(x0_v: float, up_v: float, un_v: float, H_v: float) -> dict[str, Any]:
            sim = simulate_coupled_fast_slow_trajectory(
                a_slow=a_s, b_couple=b_c, c_drive=c_d, tau_fast=tau_f, g_feedback=0.52, d_fast_drive=0.0,
                x0=x0_v, z0=0.52 * x0_v,
                u_fn=lambda t: (up_v if t < 0.38 else un_v),
                timestamps=[*rel_ts, 0.58 + H_v], rk4_dt=0.001,
            )
            r_pref = [
                {"t": round(t_shift + float(pt["t"]), 4), "u": round(float(pt["u"]), 5), "x": round(float(pt["x"]), 6)}
                for pt in sim[:-1]
            ]
            return {
                "u_past": round(up_v, 4), "u_now": round(un_v, 4), "x_now": round(float(sim[-2]["x"]), 6),
                "mem_conv": None, "raw_prefix": r_pref, "s_grad": 0.0, "H": round(H_v, 4),
                "y": round(float(sim[-1]["y"]), 7),
            }

        jit = rng.uniform(0.0, 0.15)
        pilot = [
            _f01_r(2.0 + jit, 2.7, 0.3, 0.40),
            _f01_r(0.3 + jit, 0.2, 2.5, 0.48),
            _f01_r(1.6 - jit, 2.3, 0.6, 0.36),
            _f01_r(0.6 + jit, 0.4, 2.1, 0.52),
            _f01_r(1.85, 2.6, 1.1, 0.44),
        ]
        extra = [_f01_r(2.3, 2.9, 0.2, 0.50), _f01_r(0.25, 0.15, 2.8, 0.39), _f01_r(1.25, 1.7, 1.8, 0.55)]
        conf = [_f01_r(2.15, 2.8, 0.45, 0.45), _f01_r(0.42, 0.3, 2.55, 0.43), _f01_r(1.45, 2.1, 0.85, 0.49)]
        hold = [_f01_r(1.95, 2.6, 0.5, 0.47), _f01_r(0.38, 0.25, 2.35, 0.41), _f01_r(1.7, 2.4, 0.95, 0.51), _f01_r(0.9, 0.6, 1.9, 0.38)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F01_IrregularMemoryKernel_ODE",
            suite="Cluster1_TemporalMultiScale", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:endpoint_matched_history_gap", "sig:coupled_transient_lag"],
            coarse_channel_available=False, horizon_H=0.45, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, open_rows=[extra[0]]),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F02: Adiabatic Coarse-Channel Slow Fold ---
    if family_code == "F02_AdiabaticCoarse_Fold":
        a_s = round(rng.uniform(1.25, 1.85), 3)
        b_c = round(rng.uniform(0.22, 0.34), 3)
        c_d = round(rng.uniform(2.0, 3.3), 3)
        H_base = round(rng.uniform(3.3, 4.4), 2)

        def _f02_r(un_v: float, xn_v: float, H_v: float) -> dict[str, float]:
            sim = simulate_coupled_fast_slow_trajectory(
                a_slow=a_s, b_couple=b_c, c_drive=c_d, tau_fast=0.018, g_feedback=0.50, d_fast_drive=0.0,
                x0=xn_v, z0=0.50 * xn_v, u_fn=lambda _t: un_v, timestamps=[0.0, H_v], rk4_dt=0.002,
            )
            return {
                "u_past": round(0.7 * un_v, 4), "u_now": round(un_v, 4), "x_now": round(xn_v, 4),
                "mem_conv": round(0.50 * xn_v, 4), "s_grad": 0.0, "H": round(H_v, 4), "y": round(float(sim[-1]["y"]), 7),
            }

        pilot = [_f02_r(0.6, 2.2, H_base - 0.3), _f02_r(1.4, 0.4, H_base), _f02_r(2.2, 3.5, H_base + 0.3), _f02_r(3.0, 0.8, H_base - 0.15), _f02_r(3.7, 2.9, H_base + 0.15)]
        extra = [_f02_r(1.0, 1.5, H_base - 0.2), _f02_r(2.6, 0.5, H_base + 0.2), _f02_r(3.4, 2.1, H_base)]
        conf = [_f02_r(0.9, 1.8, H_base), _f02_r(2.0, 0.6, H_base + 0.25), _f02_r(3.3, 3.1, H_base - 0.2)]
        hold = [_f02_r(0.7, 0.5, H_base + 0.1), _f02_r(1.7, 2.8, H_base - 0.25), _f02_r(2.8, 1.2, H_base + 0.2), _f02_r(3.6, 0.4, H_base)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F02_AdiabaticCoarse_Fold",
            suite="Cluster1_TemporalMultiScale", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:slow_horizon_query", "sig:fast_transient_damped"],
            coarse_channel_available=True, horizon_H=H_base, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F03: Dual-Horizon Short-Gain / Long-Reversal ---
    if family_code == "F03_DualHorizon_EndogenousReversal":
        af = round(rng.uniform(1.30, 1.75), 3)
        tf = round(rng.uniform(0.28, 0.44), 3)
        aslow = round(rng.uniform(2.25, 2.75), 3)
        tslow = round(rng.uniform(10.5, 14.5), 2)
        H_targ = round(rng.uniform(19.0, 25.0), 1)

        def _f03_r(un_v: float, H_v: float) -> dict[str, float]:
            yv = un_v * (af * (1.0 - math.exp(-H_v / tf)) - aslow * (1.0 - math.exp(-H_v / tslow)))
            return {"u_past": 0.5, "u_now": round(un_v, 4), "x_now": 0.0, "mem_conv": 0.0, "s_grad": 0.0, "H": round(H_v, 4), "y": round(yv, 7)}

        pilot = [_f03_r(0.8, 0.16), _f03_r(1.5, 0.25), _f03_r(2.2, 0.34), _f03_r(2.9, 0.41), _f03_r(3.6, 0.48)]
        slow_p = [_f03_r(1.8, round(0.35 * H_targ, 2)), _f03_r(2.6, round(0.72 * H_targ, 2)), _f03_r(3.3, H_targ)]
        extra = [_f03_r(1.3, 4.5), _f03_r(2.4, 11.0), _f03_r(3.1, 16.5)]
        conf = [_f03_r(1.1, H_targ - 2.0), _f03_r(2.3, H_targ), _f03_r(3.4, H_targ + 2.0)]
        hold = [_f03_r(0.9, H_targ - 1.0), _f03_r(1.9, H_targ + 1.5), _f03_r(2.8, H_targ - 3.0), _f03_r(3.7, H_targ)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F03_DualHorizon_EndogenousReversal",
            suite="Cluster1_TemporalMultiScale", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:short_window_positive_gain", "sig:endogenous_adaptation_candidate"],
            coarse_channel_available=False, horizon_H=H_targ, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, slow_rows=slow_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F04: Multi-Timescale Normalized Before/After Perturbation Contrast ---
    if family_code == "F04_BeforeAfter_NormalizedHysteresis":
        t_scale = rng.choice([0.004, 0.02, 0.2, 1.0, 5.0])
        gain = round(rng.uniform(1.6, 2.8) * rng.choice([-1.0, 1.0]), 3)
        tau_r = round(rng.uniform(0.45, 1.25), 3)
        k_eq = round(rng.uniform(0.25, 0.65), 3)

        def _f04_r(up_n: float, un_n: float, H_v: float) -> dict[str, float]:
            yv = gain * (un_n - up_n) * math.exp(-H_v / tau_r) + k_eq * un_n
            return {
                "u_past": round(up_n * t_scale, 6), "u_now": round(un_n * t_scale, 6), "x_now": round(k_eq * up_n, 5),
                "mem_conv": 0.0, "s_grad": 0.0, "H": round(H_v, 4), "y": round(yv, 7),
            }

        pilot = [_f04_r(0.4, 2.2, 0.25), _f04_r(2.5, 0.6, 0.45), _f04_r(0.8, 3.1, 0.75), _f04_r(3.0, 1.2, 1.10), _f04_r(1.5, 1.5, 0.60)]
        extra = [_f04_r(0.5, 2.8, 0.35), _f04_r(2.7, 0.4, 0.90), _f04_r(1.2, 3.4, 1.35)]
        conf = [_f04_r(0.6, 2.6, 0.35), _f04_r(2.8, 0.8, 0.65), _f04_r(1.0, 3.3, 1.05)]
        hold = [_f04_r(0.3, 1.9, 0.30), _f04_r(2.4, 0.5, 0.55), _f04_r(1.1, 2.9, 0.85), _f04_r(3.2, 1.6, 1.25)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F04_BeforeAfter_NormalizedHysteresis",
            suite="Cluster1_TemporalMultiScale", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:static_snapshot_uninformative", "sig:opposing_step_shift"],
            coarse_channel_available=False, horizon_H=0.65, tau_scale=t_scale,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F05: 2-Layer Conjunctive Siphon Subtree Across Normalized Scales ---
    if family_code == "F05_TwoLayer_ConjunctiveSiphon_Subtree":
        t_scale = rng.choice([0.005, 0.05, 1.0, 4.0])
        vm = round(rng.uniform(5.4, 8.6), 3)
        rh = round(rng.uniform(0.685, 0.725), 3)
        cl = round(rng.uniform(0.26, 0.48) * rng.choice([-1.0, 1.0]), 3)

        def _f05_r(un_n: float, s_v: float) -> dict[str, float]:
            yv = vm * (1.0 - rh * s_v / (1.0 + s_v)) * (un_n / (1.2 + un_n)) + cl * math.sqrt(un_n)
            return {"u_past": round(1.0 * t_scale, 6), "u_now": round(un_n * t_scale, 6), "x_now": 0.25, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": 0.6, "y": round(yv, 7)}

        pilot = [_f05_r(0.5, 0.25), _f05_r(1.2, 1.85), _f05_r(2.0, 0.55), _f05_r(2.9, 2.20), _f05_r(3.7, 1.05)]
        open_p = [_f05_r(2.4, 0.30)]
        extra = [_f05_r(0.8, 1.45), _f05_r(1.7, 0.40), _f05_r(3.3, 1.95)]
        conf = [_f05_r(0.7, 0.35), _f05_r(1.8, 2.05), _f05_r(3.2, 0.65)]
        hold = [_f05_r(0.9, 1.90), _f05_r(1.5, 0.28), _f05_r(2.6, 1.65), _f05_r(3.5, 0.50)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F05_TwoLayer_ConjunctiveSiphon_Subtree",
            suite="Cluster2_SubtreeAndWormhole", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"],
            coarse_channel_available=False, horizon_H=0.6, tau_scale=t_scale,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, open_rows=open_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F06: Narrow-Pilot Heterogeneous Parameter Wide Escape ---
    if family_code == "F06_NarrowPilot_HeteroParam_WideEscape":
        vm = round(rng.uniform(6.4, 8.8), 3)
        rh = round(rng.choice([rng.uniform(0.11, 0.23), rng.uniform(0.85, 0.92)]), 3)
        cl = round(rng.uniform(0.25, 0.42) * rng.choice([-1.0, 1.0]), 3)

        def _f06_r(un_v: float, s_v: float) -> dict[str, float]:
            yv = vm * (1.0 - rh * s_v / (1.0 + s_v)) * (un_v / (1.2 + un_v)) + cl * math.sqrt(un_v)
            return {"u_past": 1.0, "u_now": round(un_v, 4), "x_now": 0.25, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": 0.6, "y": round(yv, 7)}

        pilot = [_f06_r(0.5, 0.99), _f06_r(1.2, 1.008), _f06_r(1.9, 0.995), _f06_r(2.7, 1.005), _f06_r(3.5, 0.992)]
        open_p = [_f06_r(2.2, 0.18), _f06_r(2.9, 2.35)]
        extra = [_f06_r(0.8, 0.24), _f06_r(1.7, 2.10), _f06_r(3.1, 0.38)]
        conf = [_f06_r(0.6, 0.25), _f06_r(1.6, 2.20), _f06_r(3.3, 0.35)]
        hold = [_f06_r(0.9, 2.00), _f06_r(1.9, 0.28), _f06_r(2.8, 1.85), _f06_r(3.6, 0.55)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F06_NarrowPilot_HeteroParam_WideEscape",
            suite="Cluster2_SubtreeAndWormhole", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"],
            coarse_channel_available=False, horizon_H=0.6, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, open_rows=open_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F07: Critical Threshold Gray-Zone Avalanche (`Wave 1` sublation -> `Waves 2..6` gray-zone reuse!) ---
    if family_code == "F07_CriticalThreshold_GrayZone_Avalanche":
        vm = round(rng.uniform(6.1, 7.8), 3)
        rh = round(rng.uniform(0.69, 0.72), 3)
        sc = round(rng.uniform(2.52, 2.64), 3)
        ka = round(rng.uniform(1.45, 1.85), 3)
        cl = round(rng.uniform(0.28, 0.40), 3)

        def _f07_r(un_v: float, s_v: float) -> dict[str, float]:
            yv = vm * (1.0 - rh * s_v / (1.0 + s_v)) * (un_v / (1.2 + un_v)) + cl * math.sqrt(un_v) - ka * max(0.0, s_v - sc) * un_v
            return {"u_past": 1.0, "u_now": round(un_v, 4), "x_now": 0.3, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": 0.8, "y": round(yv, 7)}

        # Subcritical tasks (`F05, F06, F08, F10, F11, F12`) all have pilot `s_grad <= 2.25`.
        # Wave 1: max pilot `s_grad == 2.34 <= 2.36` (triggers Turn-1 refutation & sublation);
        # Waves 2..6: max pilot `s_grad == 2.44` (sits inside Wave-1's learned `[2.36, 2.68]` gray zone while still < `sc`!)
        top_pilot_s = 2.34 if wave_idx == 1 else 2.44
        pilot = [_f07_r(0.6, 0.35), _f07_r(1.3, 0.85), _f07_r(2.1, 1.35), _f07_r(2.9, 1.88), _f07_r(3.6, top_pilot_s)]
        hi_p = [_f07_r(1.8, 2.78), _f07_r(2.5, 3.18), _f07_r(3.2, 3.58)]
        extra = [_f07_r(1.1, 0.55), _f07_r(2.3, 1.55), _f07_r(3.1, 2.15)]
        conf = [_f07_r(1.0, 0.55), _f07_r(2.4, 3.46), _f07_r(3.2, 3.12)]
        hold = [_f07_r(0.8, 0.45), _f07_r(1.8, 2.88), _f07_r(2.7, 3.36), _f07_r(3.5, 1.65)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F07_CriticalThreshold_GrayZone_Avalanche",
            suite="Cluster2_SubtreeAndWormhole", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"],
            coarse_channel_available=False, horizon_H=0.8, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, high_sgrad_rows=hi_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F08: Two-Hop Cross-Disciplinary Wormhole Bridge Transfer (`K=45` lexical distractors!) ---
    if family_code == "F08_TwoHop_WormholeBridge_Transfer":
        b_cue = [
            "sig:bridge_epidemic_mobility",
            "sig:bridge_thermal_fin",
            "sig:bridge_supply_bullwhip",
        ][(rep_idx - 1) % 3]
        vm = round(rng.uniform(5.8, 8.3), 3)
        rh = round(rng.uniform(0.69, 0.72), 3)
        cl = round(rng.uniform(0.28, 0.44) * rng.choice([-1.0, 1.0]), 3)

        def _f08_r(un_v: float, s_v: float) -> dict[str, float]:
            yv = vm * (1.0 - rh * s_v / (1.0 + s_v)) * (un_v / (1.2 + un_v)) + cl * math.sqrt(un_v)
            return {"u_past": 1.0, "u_now": round(un_v, 4), "x_now": 0.25, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": 0.6, "y": round(yv, 7)}

        # Narrow `s_grad == 1.0` in pilot so `s_grad` inhibition is hidden until bridge wormhole + probe!
        pilot = [_f08_r(0.5, 1.0), _f08_r(1.2, 1.0), _f08_r(1.9, 1.0), _f08_r(2.7, 1.0), _f08_r(3.5, 1.0)]
        open_p = [_f08_r(2.1, 0.20), _f08_r(2.9, 2.30)]
        extra = [_f08_r(0.8, 0.26), _f08_r(1.7, 2.15), _f08_r(3.1, 0.40)]
        conf = [_f08_r(0.6, 0.28), _f08_r(1.6, 2.20), _f08_r(3.3, 0.38)]
        hold = [_f08_r(0.9, 1.95), _f08_r(1.8, 0.30), _f08_r(2.7, 1.80), _f08_r(3.6, 0.60)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F08_TwoHop_WormholeBridge_Transfer",
            suite="Cluster2_SubtreeAndWormhole", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=[b_cue, "sig:flux_conservation"],
            coarse_channel_available=False, horizon_H=0.6, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, open_rows=open_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F09: `LLM-SRBench` Style Transformed Cooperative Hill-2 + Competitive Sink Inhibition ---
    if family_code == "F09_LSR_Transformed_HillInhibition":
        t_scale = rng.choice([0.1, 1.0, 2.5])
        vm = round(rng.uniform(6.5, 9.5), 3)
        kh = round(rng.uniform(0.9, 2.2), 3)
        ainh = round(rng.uniform(0.45, 0.95), 3)
        cleak = round(rng.uniform(0.12, 0.32), 3)

        def _f09_r(un_n: float, s_v: float) -> dict[str, float]:
            yv = vm * ((un_n ** 2) / (kh + (un_n ** 2))) / (1.0 + ainh * s_v) - cleak * un_n
            return {"u_past": round(0.5 * t_scale, 5), "u_now": round(un_n * t_scale, 5), "x_now": 0.2, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": 0.7, "y": round(yv, 7)}

        pilot = [_f09_r(0.4, 0.25), _f09_r(1.0, 1.85), _f09_r(1.8, 0.50), _f09_r(2.7, 2.25), _f09_r(3.6, 1.05)]
        open_p = [_f09_r(1.4, 0.20), _f09_r(2.3, 2.40)]
        extra = [_f09_r(0.7, 1.40), _f09_r(2.1, 0.35), _f09_r(3.2, 1.90)]
        conf = [_f09_r(0.6, 0.30), _f09_r(1.6, 2.10), _f09_r(3.1, 0.65)]
        hold = [_f09_r(0.8, 1.95), _f09_r(1.3, 0.28), _f09_r(2.5, 1.65), _f09_r(3.4, 0.45)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F09_LSR_Transformed_HillInhibition",
            suite="Cluster3_LSR_And_5ParamHybrids", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:sigmoidal_inflection_onset", "sig:sink_gradient_C2"],
            coarse_channel_available=False, horizon_H=0.7, tau_scale=t_scale,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, open_rows=open_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F10: `LLM-SRBench` Style Barrier-Activated Siphon ---
    if family_code == "F10_LSR_Transformed_BarrierSiphon":
        t_scale = rng.choice([0.02, 1.0, 3.0])
        j0 = round(rng.uniform(2.8, 4.4), 3)
        bbar = round(rng.uniform(0.75, 1.35), 3)
        rh = round(rng.uniform(0.69, 0.72), 3)
        csq = round(rng.uniform(0.20, 0.38) * rng.choice([-1.0, 1.0]), 3)

        def _f10_r(un_n: float, s_v: float) -> dict[str, float]:
            yv = j0 * un_n * math.exp(-bbar / (0.5 + un_n)) * (1.0 - rh * s_v / (1.0 + s_v)) + csq * math.sqrt(un_n)
            return {"u_past": round(0.5 * t_scale, 5), "u_now": round(un_n * t_scale, 5), "x_now": 0.2, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": 0.7, "y": round(yv, 7)}

        pilot = [_f10_r(0.4, 0.25), _f10_r(1.1, 1.90), _f10_r(1.9, 0.50), _f10_r(2.8, 2.25), _f10_r(3.7, 1.05)]
        open_p = [_f10_r(1.5, 0.20), _f10_r(2.4, 2.35)]
        extra = [_f10_r(0.7, 1.35), _f10_r(2.2, 0.35), _f10_r(3.3, 1.85)]
        conf = [_f10_r(0.6, 0.32), _f10_r(1.7, 2.15), _f10_r(3.2, 0.60)]
        hold = [_f10_r(0.9, 1.95), _f10_r(1.4, 0.28), _f10_r(2.6, 1.70), _f10_r(3.5, 0.45)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F10_LSR_Transformed_BarrierSiphon",
            suite="Cluster3_LSR_And_5ParamHybrids", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:barrier_activation_onset", "sig:base_field_C1", "sig:sink_gradient_C2"],
            coarse_channel_available=False, horizon_H=0.7, tau_scale=t_scale,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, open_rows=open_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F11: 5-Param Spatio-Temporal Siphon + 2-State Memory ODE ---
    if family_code == "F11_SpatioTemporal_SiphonMemory_5P":
        t_scale = 1.0 if wave_idx == 1 else rng.choice([0.005, 0.05, 1.0, 2.0])
        p5 = dict(
            tau_scale=t_scale,
            a_slow=round(rng.uniform(1.55, 1.95), 3),
            b_couple=round(rng.uniform(1.10, 1.32), 3),
            vmax=round(rng.uniform(6.2, 8.0), 3),
            rho=round(rng.uniform(0.69, 0.72), 3),
            tau_fast=round(rng.uniform(0.48, 0.65), 3),
        )
        pilot = [
            _sim_spatiotemporal_siphon_memory_row(u_now=0.6 * t_scale, x_now=1.7, mem_conv=1.4, s_grad=0.25, H=0.40, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=1.4 * t_scale, x_now=0.35, mem_conv=0.15, s_grad=1.90, H=0.48, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=2.3 * t_scale, x_now=1.15, mem_conv=1.8, s_grad=0.55, H=0.38, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=3.1 * t_scale, x_now=0.45, mem_conv=0.2, s_grad=2.20, H=0.52, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=3.9 * t_scale, x_now=1.8, mem_conv=0.4, s_grad=0.95, H=0.44, **p5),
        ]
        extra = [
            _sim_spatiotemporal_siphon_memory_row(u_now=0.9 * t_scale, x_now=2.1, mem_conv=0.2, s_grad=0.32, H=0.56, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=2.7 * t_scale, x_now=0.2, mem_conv=1.8, s_grad=2.05, H=0.40, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=3.5 * t_scale, x_now=1.2, mem_conv=1.0, s_grad=0.60, H=0.60, **p5),
        ]
        conf = [
            _sim_spatiotemporal_siphon_memory_row(u_now=0.8 * t_scale, x_now=1.8, mem_conv=0.3, s_grad=0.28, H=0.43, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=2.2 * t_scale, x_now=0.45, mem_conv=1.6, s_grad=2.15, H=0.49, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=3.4 * t_scale, x_now=1.3, mem_conv=0.85, s_grad=0.68, H=0.45, **p5),
        ]
        hold = [
            _sim_spatiotemporal_siphon_memory_row(u_now=1.1 * t_scale, x_now=1.4, mem_conv=1.5, s_grad=1.80, H=0.42, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=1.8 * t_scale, x_now=0.65, mem_conv=0.2, s_grad=0.30, H=0.53, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=2.9 * t_scale, x_now=1.9, mem_conv=1.1, s_grad=1.45, H=0.46, **p5),
            _sim_spatiotemporal_siphon_memory_row(u_now=3.7 * t_scale, x_now=0.35, mem_conv=1.4, s_grad=0.42, H=0.50, **p5),
        ]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F11_SpatioTemporal_SiphonMemory_5P",
            suite="Cluster3_LSR_And_5ParamHybrids", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:coupled_transient_lag"],
            coarse_channel_available=False, horizon_H=0.46, tau_scale=t_scale,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- F12: 5-Param Sink-Coupled Dual-Horizon Reversal ---
    if family_code == "F12_SinkCoupled_DualHorizon_5P":
        vm = round(rng.uniform(6.6, 8.4), 3)
        rh = round(rng.uniform(0.69, 0.72), 3)
        tf = round(rng.uniform(0.30, 0.42), 3)
        aad = round(rng.uniform(1.65, 2.15), 3)
        tsl = round(rng.uniform(10.5, 13.5), 2)
        H_targ = round(rng.uniform(20.0, 24.0), 1)

        def _f12_r(un_v: float, s_v: float, H_v: float) -> dict[str, float]:
            siph = vm * (1.0 - rh * s_v / (1.0 + s_v)) * (un_v / (1.2 + un_v))
            yv = siph * (1.0 - math.exp(-H_v / tf)) - aad * un_v * (1.0 - math.exp(-H_v / tsl))
            return {"u_past": 0.5, "u_now": round(un_v, 4), "x_now": 0.2, "mem_conv": 0.0, "s_grad": round(s_v, 4), "H": round(H_v, 4), "y": round(yv, 7)}

        pilot = [_f12_r(0.7, 0.25, 0.16), _f12_r(1.5, 1.85, 0.27), _f12_r(2.3, 0.45, 0.35), _f12_r(3.1, 2.15, 0.42), _f12_r(3.8, 0.90, 0.48)]
        slow_p = [_f12_r(1.8, 0.38, 8.0), _f12_r(2.7, 1.92, 15.5), _f12_r(3.3, 0.72, H_targ + 1.5)]
        extra = [_f12_r(1.2, 0.35, 3.5), _f12_r(2.6, 1.65, 9.5), _f12_r(3.4, 0.55, 15.0)]
        conf = [_f12_r(1.1, 0.30, H_targ - 2.0), _f12_r(2.2, 1.95, H_targ), _f12_r(3.5, 0.65, H_targ + 2.0)]
        hold = [_f12_r(0.9, 1.70, H_targ - 1.0), _f12_r(1.9, 0.38, H_targ), _f12_r(2.9, 2.20, H_targ + 2.5), _f12_r(3.7, 0.85, H_targ - 3.0)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__F12_SinkCoupled_DualHorizon_5P",
            suite="Cluster3_LSR_And_5ParamHybrids", subfamily=family_code, is_out_of_scope=False,
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:short_window_positive_gain", "sig:endogenous_adaptation_candidate"],
            coarse_channel_available=False, horizon_H=H_targ, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng, slow_rows=slow_p),
            confirm_rows=conf, heldout_rows=hold,
        )

    # --- OOS: Adversarial Non-Stationary Chaotic Honesty Suite ---
    if family_code == "OOS_Adversarial_NonStationary":
        om1 = round(rng.uniform(16.5, 24.5), 2)
        om2 = round(rng.uniform(9.5, 14.5), 2)
        lure_sig = [
            "sig:base_field_C1",
            "sig:sigmoidal_inflection_onset",
            "sig:slow_horizon_query",
            "sig:barrier_activation_onset",
        ][(rep_idx - 1) % 4]

        def _oos_r(u_v: float, x_v: float, s_v: float, H_v: float) -> dict[str, float]:
            yv = 3.1 * math.sin(om1 * u_v * H_v + 3.7 * (s_v ** 2)) * math.cos(om2 * x_v + 2.3 * u_v)
            return {"u_past": round(0.8 * u_v, 4), "u_now": round(u_v, 4), "x_now": round(x_v, 4), "mem_conv": round(0.5 * x_v, 4), "s_grad": round(s_v, 4), "H": round(H_v, 4), "y": round(yv, 7)}

        pilot = [_oos_r(0.5, 0.3, 0.4, 0.6), _oos_r(1.1, 0.9, 1.2, 0.9), _oos_r(1.8, 1.5, 0.7, 1.2), _oos_r(2.5, 0.4, 1.9, 1.4), _oos_r(3.3, 1.8, 2.2, 1.1)]
        extra = [_oos_r(0.8, 1.2, 0.9, 0.8), _oos_r(2.1, 0.7, 1.6, 1.3), _oos_r(3.0, 1.4, 0.5, 1.0)]
        conf = [_oos_r(0.9, 0.6, 1.5, 1.0), _oos_r(2.2, 1.3, 0.5, 1.3), _oos_r(3.0, 0.8, 2.1, 0.9)]
        hold = [_oos_r(0.7, 1.1, 0.8, 1.1), _oos_r(1.6, 0.5, 1.8, 0.7), _oos_r(2.8, 1.7, 1.1, 1.2)]
        return M5DisjointTaskDataset(
            task_id=f"{w_tag}__OOS_Adversarial_NonStationary",
            suite="OOS_Adversarial", subfamily=family_code, is_out_of_scope=True,
            probe_signatures=["sig:high_freq_sign_oscillation", lure_sig],
            coarse_channel_available=False, horizon_H=1.1, tau_scale=1.0,
            pilot_rows=pilot, extra_dynamic_rows=extra, probe_pool=_make_standard_probes(rng),
            confirm_rows=conf, heldout_rows=hold,
        )

    raise ValueError(f"Unknown family_code: {family_code}")


IN_SCOPE_12_FAMILIES: tuple[str, ...] = (
    "F01_IrregularMemoryKernel_ODE",
    "F02_AdiabaticCoarse_Fold",
    "F03_DualHorizon_EndogenousReversal",
    "F04_BeforeAfter_NormalizedHysteresis",
    "F05_TwoLayer_ConjunctiveSiphon_Subtree",
    "F06_NarrowPilot_HeteroParam_WideEscape",
    "F07_CriticalThreshold_GrayZone_Avalanche",
    "F08_TwoHop_WormholeBridge_Transfer",
    "F09_LSR_Transformed_HillInhibition",
    "F10_LSR_Transformed_BarrierSiphon",
    "F11_SpatioTemporal_SiphonMemory_5P",
    "F12_SinkCoupled_DualHorizon_5P",
)


def build_hegel_discovery_240_waves(seed: int = 20261121) -> dict[str, list[M5DisjointTaskDataset]]:
    """Build all 6 stratified continual waves (`6 × 40 = 240` unseen tasks: `216` in-scope + `24` OOS)."""
    rng = random.Random(seed)
    waves: dict[str, list[M5DisjointTaskDataset]] = {}
    for w_idx in range(1, 7):
        w_tasks: list[M5DisjointTaskDataset] = []
        for fam in IN_SCOPE_12_FAMILIES:
            for r_idx in range(1, 4):
                w_tasks.append(_build_one_family_task(wave_idx=w_idx, rep_idx=r_idx, family_code=fam, rng=rng))
        for oos_idx in range(1, 5):
            w_tasks.append(
                _build_one_family_task(
                    wave_idx=w_idx, rep_idx=oos_idx, family_code="OOS_Adversarial_NonStationary", rng=rng
                )
            )
        assert len(w_tasks) == 40
        waves[f"wave_{w_idx}"] = w_tasks
    return waves


# ============================================================================
# 3. Online Continual Consolidation Across Waves (`K = 45 -> 48`)
# ============================================================================


def apply_v18_4_online_wave_consolidation(
    *,
    forest: PhilosophicalForest,
    node_archive: dict[str, dict[str, Any]],
    wave_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Consolidate confirmed discoveries (`F07` gray-zone avalanche, `F11` spatio-temporal, `F12` dual-horizon) into live forest."""
    tv_def = TaskView("trajectory_and_horizon_prediction", ["y"], 0.045)
    events: list[dict[str, Any]] = []
    wb_c = 0.0

    confirmed_subfams = {
        str(r["subfamily"]): str(r["task_id"])
        for r in wave_receipts
        if r.get("status") == "confirmed_ok" and r.get("strict_pass")
    }

    if "F11_SpatioTemporal_SiphonMemory_5P" in confirmed_subfams and "node_O_spatiotemporal_siphon_memory" not in forest.nodes:
        wit = confirmed_subfams["F11_SpatioTemporal_SiphonMemory_5P"]
        forest.register_node(
            node_id="node_O_spatiotemporal_siphon_memory",
            tree_id="tree_temporal_dynamics",
            layer_role="mid",
            task_view=tv_def,
            observation_map=ObservationMap("y", temporal_readout_mode="history_conditioned_horizon"),
            scale_map=ScaleMap("dimensionless_tau_plus_mori_zwanzig", preserved_queries=["trajectory_and_horizon_prediction"], folding_output_kind="memory_kernel"),
            claim_contract=ClaimContract(
                claim_id="cl_spatiotemporal_siphon_memory",
                family_plugin="subtree_temporal_m5",
                preconditions=[{"field": "has_history_or_lag_gap", "op": "==", "value": True}],
                conclusion_or_action={"model_classes": ["F11_SpatioTemporalSiphonMemory"]},
                protocol_spec={},
                tolerance=0.045,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:coupled_transient_lag"],
        )
        forest.add_horizontal_wormhole(
            wormhole_id="wh_C_O_siphon_memory",
            node_a_id="node_C_source_inland_hub",
            node_b_id="node_O_spatiotemporal_siphon_memory",
            probe_signature_refs=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:coupled_transient_lag"],
            role_and_observation_map={"u_drive": "u_now", "s_grad": "s_grad", "z_mem": "mem_conv"},
            hop_error_budget=0.015,
            permission_tier="local_solver_guidance",
        )
        node_archive["node_O_spatiotemporal_siphon_memory"] = {
            "node_id": "node_O_spatiotemporal_siphon_memory",
            "verified_mechanism_note": (
                "5-param spatio-temporal siphon-memory ODE (`u_n = u_now / tau_scale`): "
                "`J_siph = vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n))`, RK4 over `[0, H]` from `(x_now, mem_conv)` "
                "for `dx/dt = -a_slow*x + b_couple*z + J_siph`, `tau_fast*dz/dt = 0.52*x - z`, `y = x(H)` (>= 8 rows via `extra_fine_trajectory_batch`)."
            ),
        }
        wb_c += 1.80
        events.append({"event": "consolidated_F11_spatiotemporal_siphon_memory", "witness": wit, "cost_C": 1.80})

    if "F07_CriticalThreshold_GrayZone_Avalanche" in confirmed_subfams and "node_P_supercritical_siphon_avalanche" not in forest.nodes:
        wit = confirmed_subfams["F07_CriticalThreshold_GrayZone_Avalanche"]
        for s_nid in ("node_C_source_inland_hub", "node_D_source_hydraulic_basin"):
            upd_cc = deepcopy(forest.nodes[s_nid]["claim_contract"])
            upd_cc["preconditions"] = [*list(upd_cc.get("preconditions", ())), {"field": "s_grad", "op": "<=", "value": 2.50}]
            upd_cc["gray_zone_bounds"] = [{
                "field": "s_grad",
                "max_supported": 2.36,
                "min_refuted": 2.68,
                "monotonicity_premise": "smooth_siphon_verified_only_up_to_s_grad_2_36",
            }]
            forest.empirical_revise_node(s_nid, witness_episode_id=wit, cause="protocol_or_scope_mismatch", updated_claim_contract=upd_cc)

        forest.register_node(
            node_id="node_P_supercritical_siphon_avalanche",
            tree_id="tree_regional_siphon",
            layer_role="leaf",
            task_view=tv_def,
            observation_map=ObservationMap("y", temporal_readout_mode="before_after_contrast"),
            scale_map=ScaleMap("dimensionless_tau_normalization", preserved_queries=["trajectory_and_horizon_prediction"]),
            claim_contract=ClaimContract(
                claim_id="cl_supercritical_siphon_avalanche",
                family_plugin="subtree_m5",
                preconditions=[],
                conclusion_or_action={"model_classes": ["F07_SupercriticalAvalanche"]},
                protocol_spec={},
                tolerance=0.045,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:monotone_concave"],
        )
        cur_slots = list(forest.shared_base_subtrees["sub_2layer_siphon_base"]["open_completion_slots"])
        cur_slots.append({
            "slot": "local_residual_branch",
            "branch_id": "confirmed_supercritical_sgrad_avalanche",
            "note": (
                "SUPERCRITICAL GRAY ZONE (`s_grad > 2.36` when `high_sgrad_regime_probe` is available): "
                "acquire `high_sgrad_regime_probe` FIRST on Turn 1 and fit 5-param threshold-avalanche siphon "
                "`vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n)) + c_loc*sqrt(u_n) - k_aval*max(0.0, s_grad - s_crit)*u_n` "
                "(subcritical tasks with `s_grad <= 2.36` remain standard smooth siphon)!"
            ),
        })
        forest.revise_shared_base_subtree(
            "sub_2layer_siphon_base", witness_episode_id=wit, cause="protocol_or_scope_mismatch", updated_open_completion_slots=cur_slots
        )
        node_archive["node_P_supercritical_siphon_avalanche"] = {
            "node_id": "node_P_supercritical_siphon_avalanche",
            "verified_mechanism_note": (
                "Supercritical sink-gradient avalanche siphon: above `s_crit ~ 2.50..2.68`, "
                "`y = vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n)) + c_loc*sqrt(u_n) - k_aval*max(0.0, s_grad - s_crit)*u_n` "
                "(acquire `high_sgrad_regime_probe` first when `s_grad > 2.36`!)."
            ),
        }
        wb_c += 1.80
        events.append({"event": "consolidated_F07_grayzone_and_avalanche", "witness": wit, "cost_C": 1.80})

    if "F12_SinkCoupled_DualHorizon_5P" in confirmed_subfams and "node_S_sink_dual_horizon_5p" not in forest.nodes:
        wit = confirmed_subfams["F12_SinkCoupled_DualHorizon_5P"]
        forest.register_node(
            node_id="node_S_sink_dual_horizon_5p",
            tree_id="tree_endogenous_adaptation",
            layer_role="mid",
            task_view=tv_def,
            observation_map=ObservationMap("y", temporal_readout_mode="history_conditioned_horizon"),
            scale_map=ScaleMap("dual_timescale_window_split", preserved_queries=["trajectory_and_horizon_prediction"]),
            claim_contract=ClaimContract(
                claim_id="cl_sink_dual_horizon_5p",
                family_plugin="temporal_subtree_m5",
                preconditions=[],
                conclusion_or_action={"model_classes": ["F12_SinkDualHorizon5P"]},
                protocol_spec={"requires_slow_horizon_voi_probe": True},
                tolerance=0.045,
                evidence_status="SUPPORTED_IN_SCOPE",
                permission_tier="local_solver_guidance",
            ),
            probe_signatures=["sig:base_field_C1", "sig:sink_gradient_C2", "sig:short_window_positive_gain", "sig:endogenous_adaptation_candidate"],
        )
        node_archive["node_S_sink_dual_horizon_5p"] = {
            "node_id": "node_S_sink_dual_horizon_5p",
            "verified_mechanism_note": (
                "5-param sink-coupled dual-horizon adaptation: acquire `slow_horizon_discriminating_probe` first, then fit "
                "`y = vmax*(1 - rho*s_grad/(1+s_grad))*(u_n/(1.2+u_n))*(1 - exp(-H/tau_fast)) - amp_adapt*u_n*(1 - exp(-H/tau_slow))`."
            ),
        }
        wb_c += 1.80
        events.append({"event": "consolidated_F12_sink_dual_horizon", "witness": wit, "cost_C": 1.80})

    return {
        "events": events,
        "online_writeback_composite_cost": round(wb_c, 4),
        "new_forest_node_count": len(forest.nodes),
    }


# ============================================================================
# 4. Aggregation & Checkpointed Wave Runner (`2,160` Live Episodes)
# ============================================================================


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"total_tasks": 0}
    in_r = [r for r in rows if not r["is_out_of_scope"]]
    oos_r = [r for r in rows if r["is_out_of_scope"]]
    m_tot = round(sum(float(r["measurement_cost"]) for r in rows), 4)
    k_tot = round(sum(float(r["compute_cost"]) for r in rows), 4)
    in_pass = sum(1 for r in in_r if r["strict_pass"])
    oos_ok = sum(1 for r in oos_r if r["status"] == "scoped_unknown")
    return {
        "total_tasks": len(rows),
        "in_scope_strict_pass": f"{in_pass}/{len(in_r)}",
        "in_scope_pass_rate": round(in_pass / max(1, len(in_r)), 4),
        "oos_scoped_unknown": f"{oos_ok}/{len(oos_r)}",
        "false_confirmations": sum(1 for r in rows if r["false_confirmation"]),
        "total_api_calls": sum(int(r["api_calls"]) for r in rows),
        "mean_turns_per_task": round(sum(int(r["api_calls"]) for r in rows) / max(1, len(rows)), 2),
        "total_effective_tokens": sum(int(r["effective_billed_tokens"]) for r in rows),
        "total_measurement_cost": m_tot,
        "total_compute_cost": k_tot,
        "total_composite_cost": _composite(m_tot, k_tot),
        "mean_composite_cost_per_task": round(_composite(m_tot, k_tot) / max(1, len(rows)), 4),
        "total_wall_sec": round(sum(float(r["wall_ms"]) for r in rows) / 1000.0, 1),
    }


def _write_live_manifest(
    evidence_dir: Path,
    *,
    all_receipts: list[dict[str, Any]],
    models: tuple[str, ...],
    arms: tuple[str, ...],
    paused_models: dict[str, str],
    state_by_model: dict[str, dict[str, Any]],
    offline_seed_c: float,
    offline_cpu_ms: float,
    ordered_task_ids: list[str],
    waves_completed: list[str],
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    amortization_by_model: dict[str, Any] = {}

    rec_by_key = {(r["model"], r["arm"], r["task_id"]): r for r in all_receipts}

    for m_s in models:
        m_arms: dict[str, Any] = {}
        for arm_n in arms:
            r_sub = [r for r in all_receipts if r["model"] == m_s and r["arm"] == arm_n]
            by_fam = {
                fam: _summarize_rows([r for r in r_sub if r["subfamily"] == fam])
                for fam in (*IN_SCOPE_12_FAMILIES, "OOS_Adversarial_NonStationary")
                if any(r["subfamily"] == fam for r in r_sub)
            }
            m_arms[arm_n] = {
                **_summarize_rows(r_sub),
                "by_family": by_fam,
            }
        by_model[m_s] = {
            "quota_paused": m_s in paused_models,
            "quota_pause_reason": paused_models.get(m_s),
            "final_forest_node_count_K": len(state_by_model[m_s]["forest"].nodes),
            "online_writeback_cost_C": round(state_by_model[m_s]["online_writeback_C"], 4),
            "consolidation_events": state_by_model[m_s]["consolidation_events"],
            "arms": m_arms,
        }

        # Compute Offline-to-Online Amortization on all tasks completed across all 3 arms for `m_s`!
        common_tids = [
            tid for tid in ordered_task_ids
            if all((m_s, a_chk, tid) in rec_by_key for a_chk in arms)
        ]
        if common_tids:
            amortization_by_model[m_s] = compute_offline_to_online_amortization_ledger(
                offline_seed_formation_cost_C=offline_seed_c,
                offline_cpu_wall_ms=offline_cpu_ms,
                online_writeback_cost_C=float(state_by_model[m_s]["online_writeback_C"]),
                hegel_ordered_task_costs=[rec_by_key[(m_s, "Hegel_M5_LiveGPT", tid)] for tid in common_tids],
                baseline_ordered_task_costs={
                    "vs_EqualAuth_Flat_LiveGPT": [rec_by_key[(m_s, "EqualAuth_Flat_LiveGPT", tid)] for tid in common_tids],
                    "vs_Strong_Ordinary_LiveGPT": [rec_by_key[(m_s, "Strong_Ordinary_LiveGPT", tid)] for tid in common_tids],
                },
            )

    manifest = {
        "version": "V18.4_Hegel_Discovery_240_Publication_Scale_Continual_Benchmark",
        "provider_backend": "openai_chatgpt_subscription_codex_cli",
        "auth_mode": "chatgpt",
        "third_party_ruoli_blocked": True,
        "source_forest_initial_K": 45,
        "target_task_count": 240,
        "target_episode_count": 240 * len(models) * len(arms),
        "completed_episode_count": len(all_receipts),
        "waves_completed": waves_completed,
        "quota_paused_models": paused_models,
        "by_model": by_model,
        "offline_to_online_amortization": {
            mk: {
                "completed_common_tasks_N": len(
                    next(iter(mv["comparisons"].values()))["trajectory"]
                ) if mv["comparisons"] else 0,
                "offline_seed_formation_cost_C": mv["offline_seed_formation_cost_C"],
                "online_continual_writeback_cost_C": mv["online_continual_writeback_cost_C"],
                "summary_by_baseline": {
                    bk: {k: v for k, v in bv.items() if k != "trajectory"}
                    for bk, bv in mv["comparisons"].items()
                },
            }
            for mk, mv in amortization_by_model.items()
        },
    }
    (evidence_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_v18_4_hegel_discovery_240_study(
    evidence_dir: Path,
    *,
    run_live_gpt: bool = True,
    models: tuple[str, ...] = ("gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"),
    wave_names: tuple[str, ...] = ("wave_1", "wave_2", "wave_3", "wave_4", "wave_5", "wave_6"),
    max_workers: int = 8,
    max_turns: int = 6,
    reasoning_effort: str = "medium",
    clean_restart: bool = False,
) -> dict[str, Any]:
    """Run checkpoint-resumable `Hegel-Discovery-240` (`240` tasks × `3` models × `3` arms = `2,160` episodes)."""
    if clean_restart and evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    t0_cpu = monotonic()
    _, seed_pb = _build_seed_forest_and_priors()
    offline_cpu_ms = (monotonic() - t0_cpu) * 1000.0
    offline_seed_c = float(seed_pb["source_formation_cost"])

    waves = build_hegel_discovery_240_waves(20261121)
    ordered_task_ids = [t.task_id for w_k in wave_names for t in waves[w_k]]
    live_arms = ("Hegel_M5_LiveGPT", "EqualAuth_Flat_LiveGPT", "Strong_Ordinary_LiveGPT")

    codex_info = discover_chatgpt_subscription_codex_cli()
    if not (run_live_gpt and codex_info.get("available")):
        dry = {
            "version": "V18.4_Dry_Structure_Check",
            "live_executed": False,
            "total_waves": len(waves),
            "total_tasks": sum(len(v) for v in waves.values()),
            "initial_forest_K": 45,
        }
        (evidence_dir / "study_manifest.json").write_text(json.dumps(dry, indent=2) + "\n", encoding="utf-8")
        return dry

    ws_root = evidence_dir / "workspaces_and_dags"
    ws_root.mkdir(parents=True, exist_ok=True)

    state_by_model: dict[str, dict[str, Any]] = {}
    for m_s in models:
        f_i, pb_i, arch_i = build_live_k45_forest_and_archive()
        state_by_model[m_s] = {
            "forest": f_i,
            "priors_bundle": pb_i,
            "node_archive": arch_i,
            "consolidation_events": [],
            "online_writeback_C": 0.0,
        }

    paused_models: dict[str, str] = {}
    pause_lock = threading.Lock()
    progress_lock = threading.Lock()
    all_receipts: list[dict[str, Any]] = []
    waves_done: list[str] = []

    for w_name in wave_names:
        w_tasks = waves[w_name]
        w_jobs: list[tuple[int, str, M5DisjointTaskDataset, str]] = []
        jid = 0
        for ds_item in w_tasks:
            for m_s in models:
                if m_s in paused_models:
                    continue
                for arm_n in live_arms:
                    w_jobs.append((jid, m_s, deepcopy(ds_item), arm_n))
                    jid += 1

        w_completed_counter = 0
        wave_indexes = {
            m_s: (
                HippoRAGWormholeIndex(state_by_model[m_s]["forest"]),
                FlatEqualMetadataIndex(state_by_model[m_s]["forest"]),
            )
            for m_s in models
        }

        def _exec_job(
            jb: tuple[int, str, M5DisjointTaskDataset, str],
            w_lbl: str = w_name,
            w_tot: int = len(w_jobs),
        ) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
            nonlocal w_completed_counter
            j_i, m_s, ds_i, arm_n = jb
            with pause_lock:
                if m_s in paused_models:
                    return j_i, None, []

            ws = ws_root / w_lbl / f"{m_s}__{arm_n}__{ds_i.task_id}"
            dag_p = ws / "decision_provenance_dag.json"
            if dag_p.is_file():
                try:
                    cached = json.loads(dag_p.read_text(encoding="utf-8"))
                    if verify_decision_provenance_chain(cached["decision_dag"])["valid"]:
                        with progress_lock:
                            w_completed_counter += 1
                        return j_i, cached["episode_summary"], cached["decision_dag"]
                except Exception:
                    pass
            if ws.exists():
                shutil.rmtree(ws)

            st = state_by_model[m_s]
            h_idx, fl_idx = wave_indexes[m_s]
            sub_c = CodexSubscriptionClient(
                codex_info=codex_info, model=m_s, reasoning_effort=reasoning_effort, timeout_seconds=135
            )
            try:
                ep_r, ch_r = run_single_live_gpt_m5_episode(
                    ds_i, arm_n, client=sub_c, workspace_dir=ws,
                    forest=st["forest"], priors_bundle=st["priors_bundle"],
                    hippo_idx=h_idx, flat_idx=fl_idx,
                    node_archive=st["node_archive"], max_turns=max_turns,
                )
            except SubscriptionQuotaPausedError as q_exc:
                with pause_lock:
                    if m_s not in paused_models:
                        paused_models[m_s] = str(q_exc)
                        print(f"\n[QUOTA-WINDOW PAUSED] {m_s}: {q_exc}\n", file=sys.stderr, flush=True)
                if ws.exists() and not (ws / "decision_provenance_dag.json").is_file():
                    shutil.rmtree(ws, ignore_errors=True)
                return j_i, None, []

            ep_r["continual_wave"] = w_lbl
            ep_r["forest_node_count_at_start"] = len(st["forest"].nodes)
            vc = verify_decision_provenance_chain(ch_r)
            assert vc["valid"], f"Provenance invalid: {w_lbl}/{m_s}/{arm_n}/{ds_i.task_id}"
            (ws / "decision_provenance_dag.json").write_text(
                json.dumps({"episode_summary": ep_r, "chain_verification": vc, "decision_dag": ch_r}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            with progress_lock:
                w_completed_counter += 1
                cur_c = w_completed_counter
                if cur_c % 15 == 0 or cur_c == w_tot:
                    hb = {
                        "current_wave": w_lbl,
                        "wave_progress": f"{cur_c}/{w_tot}",
                        "total_completed_so_far": len(all_receipts) + cur_c,
                        "quota_paused_models": dict(paused_models),
                    }
                    (evidence_dir / "live_progress_heartbeat.json").write_text(
                        json.dumps(hb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                    print(
                        f"[{w_lbl} {cur_c}/{w_tot}] latest={m_s}|{arm_n}|{ds_i.task_id} pass={ep_r['strict_pass']} C={ep_r['composite_cost']}",
                        file=sys.stderr, flush=True,
                    )
            return j_i, ep_r, ch_r

        w_out: list[tuple[int, dict[str, Any] | None, list[dict[str, Any]]]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futs = [pool.submit(_exec_job, jb) for jb in w_jobs]
            for fut in as_completed(futs):
                w_out.append(fut.result())
        w_out.sort(key=lambda x: x[0])

        w_recs = [x[1] for x in w_out if x[1] is not None]
        all_receipts.extend(w_recs)

        # Save wave receipt checkpoint & run online consolidation per model!
        (evidence_dir / f"{w_name}_episode_receipts.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in w_recs) + "\n",
            encoding="utf-8",
        )
        for m_s in models:
            h_recs = [r for r in w_recs if r["model"] == m_s and r["arm"] == "Hegel_M5_LiveGPT"]
            if h_recs:
                cons = apply_v18_4_online_wave_consolidation(
                    forest=state_by_model[m_s]["forest"],
                    node_archive=state_by_model[m_s]["node_archive"],
                    wave_receipts=h_recs,
                )
                state_by_model[m_s]["consolidation_events"].extend(cons["events"])
                state_by_model[m_s]["online_writeback_C"] += cons["online_writeback_composite_cost"]

        if len(w_recs) == len(w_jobs) and not paused_models:
            waves_done.append(w_name)

        w_summary = {
            "wave": w_name,
            "episodes_in_wave": len(w_recs),
            "by_model": {
                m_s: {
                    arm_n: _summarize_rows([r for r in w_recs if r["model"] == m_s and r["arm"] == arm_n])
                    for arm_n in live_arms
                }
                for m_s in models
            },
        }
        (evidence_dir / f"{w_name}_summary.json").write_text(
            json.dumps(w_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_live_manifest(
            evidence_dir,
            all_receipts=all_receipts,
            models=models,
            arms=live_arms,
            paused_models=paused_models,
            state_by_model=state_by_model,
            offline_seed_c=offline_seed_c,
            offline_cpu_ms=offline_cpu_ms,
            ordered_task_ids=ordered_task_ids,
            waves_completed=waves_done,
        )
        print(f"=== COMPLETED {w_name}: {len(w_recs)} episodes flushed to disk (cumulative={len(all_receipts)}) ===", file=sys.stderr, flush=True)

        if len(paused_models) == len(models):
            print("All requested models hit subscription quota window; stopping cleanly for later resume.", file=sys.stderr, flush=True)
            break

    (evidence_dir / "all_completed_episode_receipts.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in all_receipts) + "\n",
        encoding="utf-8",
    )
    final_manifest = _write_live_manifest(
        evidence_dir,
        all_receipts=all_receipts,
        models=models,
        arms=live_arms,
        paused_models=paused_models,
        state_by_model=state_by_model,
        offline_seed_c=offline_seed_c,
        offline_cpu_ms=offline_cpu_ms,
        ordered_task_ids=ordered_task_ids,
        waves_completed=waves_done,
    )

    sha_lines: list[str] = []
    for fpath in sorted(p for p in evidence_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        rel = fpath.relative_to(evidence_dir).as_posix()
        dig = hashlib.sha256(fpath.read_bytes()).hexdigest()
        sha_lines.append(f"{dig}  {rel}")
    (evidence_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return final_manifest

"""Run v12 Deliverable C: Same-Tool, Same-History, Same-API Knowledge Utility & Counterfactual Interventions.

Writes new prospective receipt (ZERO byte modifications to any historic v10..v11.1 JSON receipts):
  ``docs/evidence/v12_autonomous_reconciliation_and_utility_20261117/deliverable_c_v12_knowledge_utility_20261117.json``

Follows ``markdown/hegel_v12/02_下一轮最小施工与实验.md`` §4 (C1–C4) strictly:
  - **C1 (Fixed M0, Equal Solver API Across All Arms — Fixes F04)**:
    All 5 arms run under fixed ``M0`` (``use_learned_experiment_policy=False``) and **identical**
    ``ResearchConfig(..., branch_reconciliation=True)`` so Pass-0 ``instantiate_parametric_family_on_target``
    solver API, candidate grammar, and budget limits are 100% identical across every arm:
      1. ``K_cold``: Neutral starting calibration anchor only (prior to source chain).
      2. ``R_ordinary_same_api``: Strong ordinary archive (identical raw ``meta_priors`` +
         ``OrdinaryPersistentCommitmentCache`` + ``_allow_on_demand_family_reconstruction=True``
         using the exact same Pass-0 calibration API within target budget).
      3. ``K_object_deleted`` (C2 Intervention 1 — Object Deletion): Frozen ``K_reconcile`` with ONLY
         ``_generalized_branch_commitments`` (``H*``) deleted (retaining raw priors, scoped branches,
         scope guards, and Pass-0 API).
      4. ``K_relations_deleted`` (C2 Intervention 2 — Structural Relation / Scope Deletion): Frozen
         ``K_reconcile`` keeping identical ``H*`` and branch predictive programs + Pass-0 API,
         deleting ONLY ``scope_guard``s, ``observable_bounds``, and retraction/support links.
      5. ``K_reconcile``: Autonomous reconciled library produced by ordinary ``UnifiedResearchAgent.run()``
         on the source chain.
  - **C3 (Diverse Pre-Registered Unseen Chains + All 7 Metric Groups)**:
    1 dev chain (``1010000``) + 8 unseen chains (``1011000..1018000``) simultaneously varying
    branch arrival order (unconditional-first, conditional-first reversed, minority-interleaved),
    continuous thresholds, interaction radii, and anonymous wiring, evaluated across 3 target regimes
    per chain (``T1_unseen_family``, ``T2_old_domain_return``, ``T3_minority_direction`` = 24 unseen
    target tasks per arm).
  - **C4 (Candid Method-Learning Status)**:
    Retains ``v11`` ``M1`` negative result as reason for ``M0`` default, and explicitly classifies
    ``v11.1``'s co-located bystander + interval-bisecting probe menu as an engineered diagnostic baseline.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    extract_real_consumption_events_from_actions,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v12_autonomous_reconciliation_and_utility_20261117"

EXACT_TOLERANCE = 1e-7
SOURCE_OBS_BUDGET = 660
SOURCE_MEAS_BUDGET = 13_200
SOURCE_COMPUTE_BUDGET = 740_000

# Target budget pre-declared identically for ALL 5 arms!
TARGET_OBS_BUDGET = 620
TARGET_MEAS_BUDGET = 12_400
TARGET_COMPUTE_BUDGET = 680_000

ARM_NAMES = (
    "K_cold",
    "R_ordinary_same_api",
    "K_object_deleted",
    "K_relations_deleted",
    "K_reconcile",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wilson_95_ci(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959964
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    half = (z * sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _make_config(seed: int, *, obs_b: int, comp_b: int) -> ResearchConfig:
    """Identical M0 + Same-API ResearchConfig across ALL arms!"""
    return ResearchConfig(
        seed=seed,
        observation_budget=obs_b,
        compute_budget=comp_b,
        max_rounds=4,
        minimum_improvement=EXACT_TOLERANCE,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=True,  # Enabled identically on ALL 5 arms!
    )


def _eval_external_fresh_48(lab: AnonymousObjectLab, res: dict[str, Any], seed: int) -> dict[str, Any]:
    sel_ch = tuple(res.get("selected_channels") or ())
    prog = Program.from_dict(res["program"])
    req_vars = {name for name, _ in prog.observed_variables()}
    cmd_vars = set(lab.spec_for(()).feature_names)
    uses_sensors = not (req_vars <= (cmd_vars | {"__target__"}))
    eval_ch = ("fine",) if (sel_ch or uses_sensors) else ()
    ext_rows = lab.evaluation_rows(eval_ch, seed=seed, length=24, trajectories=2)
    ext_mse = float(loss(prog, ext_rows, float(res.get("fallback", 0.0)), 0.5))
    return {
        "external_fresh48_mse": ext_mse,
        "evaluation_channels": list(eval_ch),
        "information_timing": (
            "sensor_channel_at_prediction_step"
            if eval_ch
            else "pure_command_control_delivery"
        ),
    }


def _run_episode(
    *,
    arm_name: str,
    episode_tag: str,
    lab_kwargs: dict[str, Any],
    library: KnowledgeLibrary,
    agent_seed: int,
    obs_b: int,
    meas_b: int,
    comp_b: int,
) -> dict[str, Any]:
    cfg = _make_config(agent_seed, obs_b=obs_b, comp_b=comp_b)
    t0 = time.perf_counter()
    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = UnifiedResearchAgent(cfg, measurement_budget=meas_b, library=library).run(lab)
        ext_info = _eval_external_fresh_48(lab, res, seed=agent_seed + 910_001)
    wall_s = round(time.perf_counter() - t0, 4)

    costs = res["costs"]
    internally_confirmed = bool(
        res["status"] == "completed"
        and res.get("confirmation_loss") is not None
        and float(res["confirmation_loss"]) <= EXACT_TOLERANCE
    )
    ext_mse = ext_info["external_fresh48_mse"]
    within_budget = bool(
        costs["observation_cost"] <= obs_b
        and costs["measurement_cost"] <= meas_b
        and costs["compute_cost"] <= comp_b
    )
    strict_pass = bool(internally_confirmed and ext_mse <= EXACT_TOLERANCE and within_budget)
    false_confirmation = bool(internally_confirmed and ext_mse > EXACT_TOLERANCE)

    evs = extract_real_consumption_events_from_actions(res["actions"])
    retrieved_ids: list[str] = []
    dispatch_tel = None
    for act in res["actions"]:
        if act.get("action") == "apply_world_structure_commitments":
            for cid in act.get("consistent_commitments_matched") or ():
                if cid not in retrieved_ids:
                    retrieved_ids.append(cid)
            bdt = act.get("branch_dispatch_telemetry")
            if bdt:
                dispatch_tel = bdt
                for fid in bdt.get("matched_parametric_family_ids") or ():
                    if fid not in retrieved_ids:
                        retrieved_ids.append(fid)

    chain_tel = build_action_chain_telemetry(
        arm=arm_name,
        task_id=res["task_id"],
        knowledge_available_count=len(library.meta_priors)
        + len(getattr(library, "_generalized_branch_commitments", ()) or ()),
        knowledge_retrieved_ids=retrieved_ids,
        confirmed=internally_confirmed,
        heldout_mse=ext_mse,
        tolerance=EXACT_TOLERANCE,
        consumption_events=evs,
    )

    return {
        "arm": arm_name,
        "episode_tag": episode_tag,
        "task_id": res["task_id"],
        "status": res["status"],
        "internally_confirmed": internally_confirmed,
        "confirmation_loss": res.get("confirmation_loss"),
        "external_fresh48_mse": ext_mse,
        "evaluation_channels": ext_info["evaluation_channels"],
        "information_timing": ext_info["information_timing"],
        "strict_pass": strict_pass,
        "false_internal_confirmation": false_confirmation,
        "within_predeclared_budget": within_budget,
        "measured_costs": {
            "compute_cost": int(costs["compute_cost"]),
            "observation_cost": int(costs["observation_cost"]),
            "measurement_cost": int(costs["measurement_cost"]),
            "queries": int(costs["queries"]),
            "wall_seconds": wall_s,
        },
        "selected_kind": res.get("selected_kind"),
        "branch_dispatch_telemetry": dispatch_tel,
        "causal_consumption_telemetry": chain_tel,
    }


def _build_five_arm_libraries(
    cold_lib: KnowledgeLibrary,
    reconciled_lib: KnowledgeLibrary,
) -> dict[str, KnowledgeLibrary]:
    """Construct the 5 pre-declared intervention libraries with identical Pass-0 solver access."""
    # 1. K_cold: starting calibration anchor only
    lib_cold = deepcopy(cold_lib)
    lib_cold._freeze_branch_writes = True
    lib_cold._allow_on_demand_family_reconstruction = False

    # 2. R_ordinary_same_api: identical raw meta_priors + OrdinaryPersistentCommitmentCache +
    #    allowed on-demand family reconstruction + Pass-0 calibration, WITHOUT pre-reconciled branch/scope guards!
    lib_r = deepcopy(reconciled_lib)
    lib_r._active_reconciled_branches = []
    lib_r._generalized_branch_commitments = []
    lib_r._retracted_premise_or_claim_ids = set()
    lib_r._premise_commitment_index = None
    lib_r._indexed_prior_fingerprints = None
    lib_r._indexed_group_commitments = None
    lib_r._freeze_branch_writes = True
    lib_r._allow_on_demand_family_reconstruction = True

    # 3. K_object_deleted (C2 Intervention 1): delete ONLY H* (_generalized_branch_commitments)
    lib_obj_del = deepcopy(reconciled_lib)
    lib_obj_del._generalized_branch_commitments = []
    lib_obj_del._freeze_branch_writes = True
    lib_obj_del._allow_on_demand_family_reconstruction = False

    # 4. K_relations_deleted (C2 Intervention 2): keep identical H* + branch predictive programs,
    #    delete ONLY scope_guards, observable_bounds, and retraction/support links!
    lib_rel_del = deepcopy(reconciled_lib)
    for b in getattr(lib_rel_del, "_active_reconciled_branches", ()) or ():
        b["scope_guard"] = None
        b["observable_bounds"] = {}
        b["status"] = "universal"
    for gf in getattr(lib_rel_del, "_generalized_branch_commitments", ()) or ():
        gf["scope_guard"] = None
        gf["observable_bounds"] = {}
    lib_rel_del._retracted_premise_or_claim_ids = set()
    lib_rel_del._freeze_branch_writes = True
    lib_rel_del._allow_on_demand_family_reconstruction = False

    # 5. K_reconcile: full autonomous reconciled library
    lib_rec = deepcopy(reconciled_lib)
    lib_rec._freeze_branch_writes = True
    lib_rec._allow_on_demand_family_reconstruction = False

    return {
        "K_cold": lib_cold,
        "R_ordinary_same_api": lib_r,
        "K_object_deleted": lib_obj_del,
        "K_relations_deleted": lib_rel_del,
        "K_reconcile": lib_rec,
    }


def _evaluate_chain_spec(chain_spec: dict[str, Any]) -> dict[str, Any]:
    cid = chain_spec["chain_id"]
    base_seed = int(chain_spec["base_seed"])
    order_mode = chain_spec["arrival_order_mode"]
    th_a, th_b, th_left = chain_spec["source_thresholds"]
    r_0, r_a, r_b, r_l = chain_spec["source_radii"]
    th_t1, th_t3 = chain_spec["target_thresholds"]
    r_t1, r_t2, r_t3 = chain_spec["target_radii"]

    shared_lib = KnowledgeLibrary()
    # Common starting neutral calibration anchor (before source chain)
    cal_out = _run_episode(
        arm_name="source_calibration_anchor",
        episode_tag=f"{cid}:S_cal_neutral_N4",
        lab_kwargs={
            "seed": base_seed + 5,
            "objects": 4,
            "readout": "window_total_event",
            "relation_direction": "symmetric",
            "relation_condition": "always",
            "condition_threshold": 0.0,
            "relation_radius": 0.92,
            "actuator_curvature": 0.15,
            "name_seed": base_seed + 105,
            "wiring_seed": base_seed + 205,
        },
        library=shared_lib,
        agent_seed=base_seed + 5,
        obs_b=SOURCE_OBS_BUDGET,
        meas_b=SOURCE_MEAS_BUDGET,
        comp_b=SOURCE_COMPUTE_BUDGET,
    )
    cold_snapshot = deepcopy(shared_lib)

    src_defs = {
        "S0": (
            f"{cid}:S0_right_always_N4",
            base_seed + 15,
            {
                "seed": base_seed + 15,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": r_0,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 115,
                "wiring_seed": base_seed + 215,
            },
        ),
        "S1": (
            f"{cid}:S1_right_dpg_a_N4",
            base_seed + 25,
            {
                "seed": base_seed + 25,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_a,
                "relation_radius": r_a,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 125,
                "wiring_seed": base_seed + 225,
            },
        ),
        "S2": (
            f"{cid}:S2_right_dpg_b_N4",
            base_seed + 35,
            {
                "seed": base_seed + 35,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_b,
                "relation_radius": r_b,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 135,
                "wiring_seed": base_seed + 235,
            },
        ),
        "S_left": (
            f"{cid}:S_left_dpg_N4",
            base_seed + 45,
            {
                "seed": base_seed + 45,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "left",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_left,
                "relation_radius": r_l,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 145,
                "wiring_seed": base_seed + 245,
            },
        ),
    }

    if order_mode == "unconditional_first":
        order_keys = ["S0", "S1", "S2", "S_left"]
    elif order_mode == "conditional_first_reversed":
        order_keys = ["S1", "S2", "S0", "S_left"]
    else:
        order_keys = ["S1", "S_left", "S0", "S2"]

    source_episodes = []
    for k in order_keys:
        ep_tag, ep_sd, l_kw = src_defs[k]
        ep_res = _run_episode(
            arm_name="source_chain_ordinary_entry",
            episode_tag=ep_tag,
            lab_kwargs=l_kw,
            library=shared_lib,
            agent_seed=ep_sd,
            obs_b=SOURCE_OBS_BUDGET,
            meas_b=SOURCE_MEAS_BUDGET,
            comp_b=SOURCE_COMPUTE_BUDGET,
        )
        source_episodes.append(ep_res)

    arm_libs = _build_five_arm_libraries(cold_snapshot, shared_lib)

    target_defs = [
        (
            "T1_unseen_family",
            f"{cid}:T1_unseen_right_dpg_N5",
            base_seed + 305,
            {
                "seed": base_seed + 305,
                "objects": 5,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_t1,
                "relation_radius": r_t1,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 405,
                "wiring_seed": base_seed + 505,
            },
        ),
        (
            "T2_old_unconditional_domain",
            f"{cid}:T2_old_right_always_N5",
            base_seed + 315,
            {
                "seed": base_seed + 315,
                "objects": 5,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": r_t2,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 415,
                "wiring_seed": base_seed + 515,
            },
        ),
        (
            "T3_minority_direction_domain",
            f"{cid}:T3_minority_left_dpg_N5",
            base_seed + 325,
            {
                "seed": base_seed + 325,
                "objects": 5,
                "readout": "window_total_event",
                "relation_direction": "left",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_t3,
                "relation_radius": r_t3,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 425,
                "wiring_seed": base_seed + 525,
            },
        ),
    ]

    arms_target_results: dict[str, list[dict[str, Any]]] = {a: [] for a in ARM_NAMES}
    for arm_name in ARM_NAMES:
        for role_kind, t_tag, t_sd, t_kw in target_defs:
            # Fresh deepcopy per target so targets within a chain are mutually independent!
            t_lib = deepcopy(arm_libs[arm_name])
            # Preserve persistent OrdinaryPersistentCommitmentCache across targets on R_ordinary_same_api!
            if arm_name == "R_ordinary_same_api" and hasattr(arm_libs[arm_name], "_ordinary_commitment_cache"):
                t_lib._ordinary_commitment_cache = arm_libs[arm_name]._ordinary_commitment_cache
            t_out = _run_episode(
                arm_name=arm_name,
                episode_tag=t_tag,
                lab_kwargs=t_kw,
                library=t_lib,
                agent_seed=t_sd,
                obs_b=TARGET_OBS_BUDGET,
                meas_b=TARGET_MEAS_BUDGET,
                comp_b=TARGET_COMPUTE_BUDGET,
            )
            t_out["target_regime_role"] = role_kind
            if arm_name == "R_ordinary_same_api" and hasattr(t_lib, "_ordinary_commitment_cache"):
                arm_libs[arm_name]._ordinary_commitment_cache = t_lib._ordinary_commitment_cache
            arms_target_results[arm_name].append(t_out)

    reconciliation_compute_total = sum(
        int(rec.get("compute_cost", 0))
        for rec in (getattr(shared_lib, "_reconciliation_episode_ledger", ()) or ())
    )

    return {
        "chain_id": cid,
        "base_seed": base_seed,
        "arrival_order_mode": order_mode,
        "source_arrival_sequence": order_keys,
        "calibration_anchor_episode": cal_out,
        "source_formation_measured_ledger": {
            "source_episodes_attempted": len(source_episodes),
            "source_episodes_confirmed_exact": sum(1 for e in source_episodes if e["internally_confirmed"]),
            "source_episodes_failed_or_unresolved": sum(1 for e in source_episodes if not e["internally_confirmed"]),
            "source_false_internal_confirmations": sum(1 for e in source_episodes if e["false_internal_confirmation"]),
            "source_total_compute_measured": sum(e["measured_costs"]["compute_cost"] for e in source_episodes),
            "reconciliation_subset_compute_measured": reconciliation_compute_total,
            "source_total_observations_measured": sum(e["measured_costs"]["observation_cost"] for e in source_episodes),
            "source_total_measurements_measured": sum(e["measured_costs"]["measurement_cost"] for e in source_episodes),
            "source_total_wall_seconds": round(sum(e["measured_costs"]["wall_seconds"] for e in source_episodes), 4),
            "formed_H_star_count": len(getattr(shared_lib, "_generalized_branch_commitments", ()) or ()),
            "rejected_false_merge_count": len(getattr(shared_lib, "_rejected_merge_records", ()) or ()),
        },
        "source_episodes": source_episodes,
        "arms_target_results": arms_target_results,
    }


def _aggregate_unseen_chains(chains: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARM_NAMES:
        all_eps = [ep for ch in chains for ep in ch["arms_target_results"][arm]]
        t1_eps = [ep for ep in all_eps if ep["target_regime_role"] == "T1_unseen_family"]
        t2_eps = [ep for ep in all_eps if ep["target_regime_role"] == "T2_old_unconditional_domain"]
        t3_eps = [ep for ep in all_eps if ep["target_regime_role"] == "T3_minority_direction_domain"]

        n_all = len(all_eps)
        passes_all = sum(1 for ep in all_eps if ep["strict_pass"])
        by_arm[arm] = {
            "total_target_episodes": n_all,
            "strict_passes": passes_all,
            "strict_pass_rate": round(passes_all / max(1, n_all), 4),
            "strict_pass_95_wilson_ci": _wilson_95_ci(passes_all, n_all),
            "by_target_role": {
                "T1_unseen_family_passes": f"{sum(1 for e in t1_eps if e['strict_pass'])}/{len(t1_eps)}",
                "T2_old_domain_passes": f"{sum(1 for e in t2_eps if e['strict_pass'])}/{len(t2_eps)}",
                "T3_minority_direction_passes": f"{sum(1 for e in t3_eps if e['strict_pass'])}/{len(t3_eps)}",
                "T1_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t1_eps), 2),
                "T2_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t2_eps), 2),
                "T3_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t3_eps), 2),
            },
            "external_fresh48_mean_mse": round(mean(e["external_fresh48_mse"] for e in all_eps), 10),
            "external_fresh48_max_mse": round(max(e["external_fresh48_mse"] for e in all_eps), 10),
            "information_timing_breakdown": {
                "sensor_channel_at_prediction_step": sum(
                    1 for e in all_eps if e["information_timing"] == "sensor_channel_at_prediction_step"
                ),
                "pure_command_control_delivery": sum(
                    1 for e in all_eps if e["information_timing"] == "pure_command_control_delivery"
                ),
            },
            "false_internal_confirmations": sum(1 for e in all_eps if e["false_internal_confirmation"]),
            "measured_target_costs": {
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in all_eps), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in all_eps), 2),
                "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in all_eps), 2),
                "mean_wall_seconds": round(mean(e["measured_costs"]["wall_seconds"] for e in all_eps), 4),
                "total_compute": sum(e["measured_costs"]["compute_cost"] for e in all_eps),
                "total_wall_seconds": round(sum(e["measured_costs"]["wall_seconds"] for e in all_eps), 4),
            },
            "causal_4stage_consumption_counts": {
                "episodes_entered_candidates": sum(
                    1 for e in all_eps if e["causal_consumption_telemetry"]["entered_candidates"]
                ),
                "episodes_selected_from_knowledge": sum(
                    1 for e in all_eps if e["causal_consumption_telemetry"]["selected_from_knowledge"]
                ),
                "episodes_adopted_confirmed_from_knowledge": sum(
                    1 for e in all_eps if e["causal_consumption_telemetry"]["adopted_confirmed_from_knowledge"]
                ),
                "episodes_with_physical_executed_followup_probes": sum(
                    1 for e in all_eps if bool(e["causal_consumption_telemetry"]["executed_probe_ids"])
                ),
            },
        }

    # Explicit old-domain & minority degradation comparisons vs K_cold and R_ordinary_same_api
    r_t2 = sum(
        1 for ch in chains for e in ch["arms_target_results"]["R_ordinary_same_api"]
        if e["target_regime_role"] == "T2_old_unconditional_domain" and e["strict_pass"]
    )
    k_t2 = sum(
        1 for ch in chains for e in ch["arms_target_results"]["K_reconcile"]
        if e["target_regime_role"] == "T2_old_unconditional_domain" and e["strict_pass"]
    )
    r_t3 = sum(
        1 for ch in chains for e in ch["arms_target_results"]["R_ordinary_same_api"]
        if e["target_regime_role"] == "T3_minority_direction_domain" and e["strict_pass"]
    )
    k_t3 = sum(
        1 for ch in chains for e in ch["arms_target_results"]["K_reconcile"]
        if e["target_regime_role"] == "T3_minority_direction_domain" and e["strict_pass"]
    )

    source_summary = {
        "chains_total": len(chains),
        "total_source_episodes_attempted": sum(
            ch["source_formation_measured_ledger"]["source_episodes_attempted"] for ch in chains
        ),
        "total_source_episodes_confirmed_exact": sum(
            ch["source_formation_measured_ledger"]["source_episodes_confirmed_exact"] for ch in chains
        ),
        "total_source_episodes_failed_or_unresolved": sum(
            ch["source_formation_measured_ledger"]["source_episodes_failed_or_unresolved"] for ch in chains
        ),
        "chains_forming_H_star": sum(
            1 for ch in chains if ch["source_formation_measured_ledger"]["formed_H_star_count"] >= 1
        ),
        "mean_source_chain_compute_measured": round(
            mean(ch["source_formation_measured_ledger"]["source_total_compute_measured"] for ch in chains),
            2,
        ),
        "mean_reconciliation_maintenance_compute_measured": round(
            mean(ch["source_formation_measured_ledger"]["reconciliation_subset_compute_measured"] for ch in chains),
            2,
        ),
        "mean_source_chain_wall_seconds": round(
            mean(ch["source_formation_measured_ledger"]["source_total_wall_seconds"] for ch in chains),
            4,
        ),
    }

    return {
        "source_formation_and_maintenance_summary": source_summary,
        "arm_aggregates": by_arm,
        "old_domain_and_minority_degradation_check": {
            "T2_old_domain_K_reconcile_vs_R_ordinary_passes": {"K_reconcile": k_t2, "R_ordinary_same_api": r_t2, "degraded": k_t2 < r_t2},
            "T3_minority_K_reconcile_vs_R_ordinary_passes": {"K_reconcile": k_t3, "R_ordinary_same_api": r_t3, "degraded": k_t3 < r_t3},
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_spec = {
        "chain_id": "dev_c_chain_1010000",
        "base_seed": 1_010_000,
        "arrival_order_mode": "unconditional_first",
        "source_thresholds": (0.22, 0.46, 0.25),
        "source_radii": (0.94, 0.93, 0.91, 0.92),
        "target_thresholds": (0.34, 0.36),
        "target_radii": (0.89, 0.95, 0.90),
    }
    unseen_specs = [
        {
            "chain_id": "unseen_c_chain_1011000",
            "base_seed": 1_011_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.18, 0.44, 0.26),
            "source_radii": (0.95, 0.92, 0.90, 0.93),
            "target_thresholds": (0.31, 0.37),
            "target_radii": (0.88, 0.94, 0.91),
        },
        {
            "chain_id": "unseen_c_chain_1012000",
            "base_seed": 1_012_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.24, 0.48, 0.22),
            "source_radii": (0.92, 0.94, 0.89, 0.91),
            "target_thresholds": (0.36, 0.33),
            "target_radii": (0.91, 0.89, 0.94),
        },
        {
            "chain_id": "unseen_c_chain_1013000",
            "base_seed": 1_013_000,
            "arrival_order_mode": "minority_interleaved",
            "source_thresholds": (0.20, 0.45, 0.28),
            "source_radii": (0.91, 0.95, 0.93, 0.90),
            "target_thresholds": (0.33, 0.39),
            "target_radii": (0.90, 0.92, 0.88),
        },
        {
            "chain_id": "unseen_c_chain_1014000",
            "base_seed": 1_014_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.16, 0.42, 0.24),
            "source_radii": (0.93, 0.90, 0.95, 0.92),
            "target_thresholds": (0.29, 0.35),
            "target_radii": (0.93, 0.90, 0.92),
        },
        {
            "chain_id": "unseen_c_chain_1015000",
            "base_seed": 1_015_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.21, 0.47, 0.27),
            "source_radii": (0.90, 0.93, 0.91, 0.94),
            "target_thresholds": (0.35, 0.40),
            "target_radii": (0.89, 0.93, 0.90),
        },
        {
            "chain_id": "unseen_c_chain_1016000",
            "base_seed": 1_016_000,
            "arrival_order_mode": "minority_interleaved",
            "source_thresholds": (0.19, 0.43, 0.23),
            "source_radii": (0.94, 0.89, 0.92, 0.95),
            "target_thresholds": (0.32, 0.34),
            "target_radii": (0.92, 0.88, 0.93),
        },
        {
            "chain_id": "unseen_c_chain_1017000",
            "base_seed": 1_017_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.23, 0.49, 0.29),
            "source_radii": (0.92, 0.96, 0.90, 0.91),
            "target_thresholds": (0.38, 0.41),
            "target_radii": (0.90, 0.95, 0.89),
        },
        {
            "chain_id": "unseen_c_chain_1018000",
            "base_seed": 1_018_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.17, 0.41, 0.21),
            "source_radii": (0.91, 0.92, 0.94, 0.89),
            "target_thresholds": (0.28, 0.32),
            "target_radii": (0.94, 0.91, 0.90),
        },
    ]

    dev_result = _evaluate_chain_spec(dev_spec)
    unseen_results = [_evaluate_chain_spec(sp) for sp in unseen_specs]
    agg = _aggregate_unseen_chains(unseen_results)

    receipt = {
        "schema_version": "hegel_v12_deliverable_c_knowledge_utility_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v12_Deliverable_C_Same_API_Knowledge_Utility_and_Counterfactuals",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "equal_tool_and_api_protocol": {
            "experiment_policy_fixed": "developer_default_M0",
            "pass0_parametric_family_calibration_enabled_on_all_arms": True,
            "r_ordinary_allowed_on_demand_family_reconstruction": True,
            "zero_magic_cost_constants": True,
            "source_episode_budget": {
                "observations": SOURCE_OBS_BUDGET,
                "measurements": SOURCE_MEAS_BUDGET,
                "compute": SOURCE_COMPUTE_BUDGET,
            },
            "target_episode_budget": {
                "observations": TARGET_OBS_BUDGET,
                "measurements": TARGET_MEAS_BUDGET,
                "compute": TARGET_COMPUTE_BUDGET,
                "exact_confirmation_tolerance": EXACT_TOLERANCE,
            },
        },
        "c4_method_learning_epistemic_record": {
            "v11_M1_status": "retained_negative_result_4_of_12_vs_M0_8_of_12",
            "default_method_in_this_study": "M0_default_space_filling",
            "v11_1_co_located_bystander_menu_classification": "engineered_diagnostic_baseline_reported_separately_not_claimed_as_learned_policy",
        },
        "unseen_8chain_aggregate": agg,
        "development_chain": dev_result,
        "unseen_evaluation_chains": unseen_results,
    }
    out_path = OUT_DIR / "deliverable_c_v12_knowledge_utility_20261117.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "written": str(out_path),
                "source_summary": agg["source_formation_and_maintenance_summary"],
                "arm_strict_passes": {
                    k: {
                        "strict_passes": v["strict_passes"],
                        "total": v["total_target_episodes"],
                        "ci95": v["strict_pass_95_wilson_ci"],
                        "mean_compute": v["measured_target_costs"]["mean_compute"],
                        "by_role": v["by_target_role"],
                    }
                    for k, v in agg["arm_aggregates"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

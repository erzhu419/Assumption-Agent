"""Run Hegel Machine v14 Frozen Single-Touch Prospective Evaluations (Deliverables 2 & 3).

Writes TWO strict-JSON (`allow_nan=False`) prospective receipts in
``docs/evidence/v14_continuous_object_and_fair_comparison_20261118/``:
  1. ``deliverable_2_unconfounded_noH_source_paired_study_20261118.json``
     - 8 brand-new unexposed physical source chains (`1031000–1038000`, `24` target episodes per arm).
     - Unconfounded primary comparison (`_prioritize_generalized_families = False` across ALL arms):
       `K_cold`, `R_lazy_noH` (shared raw priors + replayable `_raw_reconciliation_evidence_archive`,
       3-layer cache skipping unconsumed `H*`), and `K_eager_noH`.
     - Strict separation of common calibration anchors (`8`), ordinary source episodes (`32`), and
       target episodes (`24`/arm); separate agent-run vs external-eval wall-clock; cold-cache (`k=1`)
       vs warm-cache (`k=2, 3`); and per-chain prefix full cost differences `Delta C_c(1..3)` (§8.3).
  2. ``deliverable_3_continuous_repaired_object_study_20261118.json``
     - 6 brand-new unexposed `focal_memory` continuous-object chains (`1041000–1046000`), recording
       100% of attempted chains.
     - Question A (§7.2): Parent-task repair worthiness under identical `latent_state_discovery` API,
       identical total budgets, and identical Markov history (`A_ordinary_failure_retry_equal_api` vs
       `A_subquestion_driven_repair_equal_api`).
     - Question B (§7.2): Downstream reuse of the exact same repaired `accumulate` object (`concept_id`,
       `branch_id`, `semantic_version`) on unseen `N=5` targets (`T1` unseen `focal_memory` decay/weight/radius/wiring,
       `T2` return to old Markov domain, `T3` mismatched left-conditional negative-transfer counterexample)
       across `K_eager_noH`, `R_lazy_noH`, `K_minimal_branch_removed_raw_kept`,
       `K_minimal_repaired_object_removed_other_sources_kept`, and `K_cold` (all keeping full
       `latent_state_discovery=True` capability and identical budgets).
     - Machine-generated 7-sentence factual ledger (§7.3) per chain from the single continuous ledger.

Zero byte modifications to all 32 historic v10..v13 JSON receipts (`model_calls = 0`).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from statistics import mean, median
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import _program_semantic_version
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    extract_real_consumption_events_from_actions,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v14_continuous_object_and_fair_comparison_20261118"

EXACT_TOLERANCE = 1e-7

# Deliverable 2 frozen budgets (matching M0 / v13 source-target regime budgets)
D2_SOURCE_OBS_BUDGET = 660
D2_SOURCE_MEAS_BUDGET = 13_200
D2_SOURCE_COMPUTE_BUDGET = 740_000

D2_TARGET_OBS_BUDGET = 620
D2_TARGET_MEAS_BUDGET = 12_400
D2_TARGET_COMPUTE_BUDGET = 680_000

D2_ARMS = ("K_cold", "R_lazy_noH", "K_eager_noH")

# Deliverable 3 frozen budgets (`focal_memory` continuous-object chains)
D3_MARKOV_SRC_OBS_BUDGET = 520
D3_MARKOV_SRC_MEAS_BUDGET = 11_000
D3_MARKOV_SRC_COMPUTE_BUDGET = 540_000

D3_CONFLICT_OBS_BUDGET = 560
D3_CONFLICT_MEAS_BUDGET = 11_500
D3_CONFLICT_COMPUTE_BUDGET = 880_000

D3_TARGET_OBS_BUDGET = 580
D3_TARGET_MEAS_BUDGET = 11_800
D3_TARGET_COMPUTE_BUDGET = 840_000

D3_TARGET_ARMS = (
    "K_cold",
    "R_lazy_noH",
    "K_eager_noH",
    "K_minimal_branch_removed_raw_kept",
    "K_minimal_repaired_object_removed_other_sources_kept",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _eval_external_fresh_48(lab: AnonymousObjectLab, res: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    sel_ch = tuple(res.get("selected_channels") or ())
    prog = Program.from_dict(res["program"])
    req_vars = {name for name, _ in prog.observed_variables()}
    cmd_vars = set(lab.spec_for(()).feature_names)
    uses_sensors = not (req_vars <= (cmd_vars | {"__target__"}))
    eval_ch = ("fine",) if (sel_ch or uses_sensors) else ()
    ext_rows = lab.evaluation_rows(eval_ch, seed=seed, length=24, trajectories=2)
    ext_mse = float(loss(prog, ext_rows, float(res.get("fallback", 0.0)), 0.5))
    eval_wall_s = round(time.perf_counter() - t0, 4)
    return {
        "external_fresh48_mse": ext_mse,
        "evaluation_channels": list(eval_ch),
        "external_eval_wall_seconds": eval_wall_s,
        "information_timing": (
            "sensor_channel_at_prediction_step"
            if eval_ch
            else "pure_command_control_delivery"
        ),
    }


def _run_episode_with_telemetry(
    *,
    arm_name: str,
    episode_tag: str,
    lab_kwargs: dict[str, Any],
    library: KnowledgeLibrary,
    cfg: ResearchConfig,
    meas_b: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t_run0 = time.perf_counter()
    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = UnifiedResearchAgent(cfg, measurement_budget=meas_b, library=library).run(lab)
        agent_wall_s = round(time.perf_counter() - t_run0, 4)
        ext_info = _eval_external_fresh_48(lab, res, seed=int(cfg.seed) + 920_001)
    total_wall_s = round(agent_wall_s + ext_info["external_eval_wall_seconds"], 4)

    costs = res["costs"]
    obs_b = cfg.observation_budget
    comp_b = cfg.compute_budget
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
    unconfirmed_yet_accurate = bool((not internally_confirmed) and ext_mse <= EXACT_TOLERANCE)

    evs = extract_real_consumption_events_from_actions(res["actions"])
    retrieved_ids: list[str] = []
    dispatch_tel = None
    ord_ep_recon_compute = 0
    ord_ep_first_hit: bool | None = None
    ord_layer2_built = 0
    scope_unknown_count = 0
    scope_excluded_ids: list[str] = []
    boundary_probes: list[dict[str, Any]] = []

    for act in res["actions"]:
        if act.get("action") == "execute_threshold_boundary_probe":
            boundary_probes.append(deepcopy(act))
        if act.get("action") == "retrieve_ordinary_cached_commitments":
            ord_ep_recon_compute += int(act.get("compute_cost", 0))
            if ord_ep_first_hit is None:
                ord_ep_first_hit = bool(act.get("cache_hit", False))
        if act.get("action") == "apply_world_structure_commitments":
            for cid in act.get("consistent_commitments_matched") or ():
                if cid not in retrieved_ids:
                    retrieved_ids.append(cid)
            bdt = act.get("branch_dispatch_telemetry")
            if bdt and dispatch_tel is None:
                dispatch_tel = bdt
            if bdt:
                for u_id in bdt.get("scope_not_established") or bdt.get("scope_unknown_branch_ids") or ():
                    scope_unknown_count += 1
                for ex_id in bdt.get("empirically_excluded_on_context") or bdt.get("excluded_out_of_scope_branch_ids") or ():
                    if ex_id not in scope_excluded_ids:
                        scope_excluded_ids.append(ex_id)
                od_st = bdt.get("on_demand_archive_family_reconstruction")
                if isinstance(od_st, dict):
                    ord_ep_recon_compute += int(od_st.get("compute_spent", 0))
                    ord_layer2_built += int(od_st.get("layer2_built", 0))
                    if ord_ep_first_hit is None:
                        ord_ep_first_hit = bool(od_st.get("cache_hit", False))
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
    wrong_transfer = bool(chain_tel.get("adopted_confirmed_from_knowledge") and false_confirmation)

    summary = {
        "arm": arm_name,
        "episode_tag": episode_tag,
        "task_id": res["task_id"],
        "episode_uid": res.get("episode_uid"),
        "status": res["status"],
        "concept_id": res.get("concept_id") or ((res.get("concepts_added") or [None])[0]),
        "internally_confirmed": internally_confirmed,
        "confirmation_loss": res.get("confirmation_loss"),
        "external_fresh48_mse": ext_mse,
        "evaluation_channels": ext_info["evaluation_channels"],
        "information_timing": ext_info["information_timing"],
        "strict_pass": strict_pass,
        "false_internal_confirmation": false_confirmation,
        "unresolved_or_unconfirmed": not internally_confirmed,
        "unconfirmed_yet_accurate_on_external": unconfirmed_yet_accurate,
        "wrong_knowledge_transfer": wrong_transfer,
        "scope_unknown_events": scope_unknown_count,
        "excluded_out_of_scope_branch_ids": scope_excluded_ids,
        "boundary_probes_executed_count": len(boundary_probes),
        "threshold_boundary_uncertainty": res.get("threshold_boundary_uncertainty"),
        "within_predeclared_budget": within_budget,
        "measured_costs": {
            "compute_cost": int(costs["compute_cost"]),
            "observation_cost": int(costs["observation_cost"]),
            "measurement_cost": int(costs["measurement_cost"]),
            "queries": int(costs["queries"]),
            "agent_run_wall_seconds": agent_wall_s,
            "external_eval_wall_seconds": ext_info["external_eval_wall_seconds"],
            "total_wall_seconds_including_external_eval": total_wall_s,
        },
        "selected_kind": res.get("selected_kind"),
        "ordinary_lazy_episode_reconstruction_compute": ord_ep_recon_compute,
        "ordinary_lazy_first_call_cache_hit": ord_ep_first_hit,
        "ordinary_lazy_layer2_families_built": ord_layer2_built,
        "branch_dispatch_telemetry": dispatch_tel,
        "causal_consumption_telemetry": chain_tel,
    }
    return summary, res


# ============================================================================
# DELIVERABLE 2: Unconfounded No-H* Source-Cluster Paired Study (1031000–1038000)
# ============================================================================

def _make_d2_config(seed: int, *, obs_b: int, comp_b: int) -> ResearchConfig:
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
        branch_reconciliation=True,
    )


def _build_d2_arm_libraries(
    cold_lib: KnowledgeLibrary,
    reconciled_lib: KnowledgeLibrary,
) -> dict[str, KnowledgeLibrary]:
    """Construct the 3 unconfounded no-H* comparison arms (`_prioritize_generalized_families = False` across ALL)."""
    lib_cold = deepcopy(cold_lib)
    lib_cold._freeze_branch_writes = True
    lib_cold._allow_on_demand_family_reconstruction = False
    lib_cold._allow_on_demand_routing_reconstruction = False
    lib_cold._prioritize_generalized_families = False

    lib_r_noh = deepcopy(reconciled_lib)
    lib_r_noh._active_reconciled_branches = []
    lib_r_noh._generalized_branch_commitments = []
    lib_r_noh._retracted_premise_or_claim_ids = set()
    lib_r_noh._premise_commitment_index = None
    lib_r_noh._indexed_prior_fingerprints = None
    lib_r_noh._indexed_group_commitments = None
    lib_r_noh._freeze_branch_writes = True
    lib_r_noh._allow_on_demand_family_reconstruction = True
    lib_r_noh._allow_on_demand_routing_reconstruction = True
    lib_r_noh._enable_scope_and_routing_guards = True
    lib_r_noh._prioritize_generalized_families = False

    lib_k_noh = deepcopy(reconciled_lib)
    lib_k_noh._freeze_branch_writes = True
    lib_k_noh._allow_on_demand_family_reconstruction = False
    lib_k_noh._allow_on_demand_routing_reconstruction = False
    lib_k_noh._enable_scope_and_routing_guards = True
    lib_k_noh._prioritize_generalized_families = False

    return {
        "K_cold": lib_cold,
        "R_lazy_noH": lib_r_noh,
        "K_eager_noH": lib_k_noh,
    }


def _evaluate_d2_chain(chain_spec: dict[str, Any]) -> dict[str, Any]:
    cid = chain_spec["chain_id"]
    base_seed = int(chain_spec["base_seed"])
    order_mode = chain_spec["arrival_order_mode"]
    th_a, th_b, th_left = chain_spec["source_thresholds"]
    r_0, r_a, r_b, r_l = chain_spec["source_radii"]
    th_t1, th_t3 = chain_spec["target_thresholds"]
    r_t1, r_t2, r_t3 = chain_spec["target_radii"]

    shared_lib = KnowledgeLibrary()
    shared_lib._prioritize_generalized_families = False
    cal_out, _ = _run_episode_with_telemetry(
        arm_name="common_calibration_anchor",
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
        cfg=_make_d2_config(base_seed + 5, obs_b=D2_SOURCE_OBS_BUDGET, comp_b=D2_SOURCE_COMPUTE_BUDGET),
        meas_b=D2_SOURCE_MEAS_BUDGET,
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
        ep_res, _ = _run_episode_with_telemetry(
            arm_name="ordinary_source_episode",
            episode_tag=ep_tag,
            lab_kwargs=l_kw,
            library=shared_lib,
            cfg=_make_d2_config(ep_sd, obs_b=D2_SOURCE_OBS_BUDGET, comp_b=D2_SOURCE_COMPUTE_BUDGET),
            meas_b=D2_SOURCE_MEAS_BUDGET,
        )
        source_episodes.append(ep_res)

    arm_libs = _build_d2_arm_libraries(cold_snapshot, shared_lib)

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

    arms_target_results: dict[str, list[dict[str, Any]]] = {a: [] for a in D2_ARMS}
    for arm_name in D2_ARMS:
        for k_idx, (role_kind, t_tag, t_sd, t_kw) in enumerate(target_defs, 1):
            t_lib = deepcopy(arm_libs[arm_name])
            if arm_name == "R_lazy_noH" and hasattr(arm_libs[arm_name], "_ordinary_commitment_cache"):
                t_lib._ordinary_commitment_cache = arm_libs[arm_name]._ordinary_commitment_cache
            t_out, _ = _run_episode_with_telemetry(
                arm_name=arm_name,
                episode_tag=t_tag,
                lab_kwargs=t_kw,
                library=t_lib,
                cfg=_make_d2_config(t_sd, obs_b=D2_TARGET_OBS_BUDGET, comp_b=D2_TARGET_COMPUTE_BUDGET),
                meas_b=D2_TARGET_MEAS_BUDGET,
            )
            t_out["target_regime_role"] = role_kind
            t_out["prefix_step_index_k"] = k_idx
            t_out["cache_state_at_entry"] = "cold_cache_k1" if k_idx == 1 else f"warm_cache_k{k_idx}"
            if arm_name == "R_lazy_noH" and hasattr(t_lib, "_ordinary_commitment_cache"):
                arm_libs[arm_name]._ordinary_commitment_cache = t_lib._ordinary_commitment_cache
            arms_target_results[arm_name].append(t_out)

    reconcile_compute_total = sum(
        int(rec.get("compute_cost", 0))
        for rec in (getattr(shared_lib, "_reconciliation_episode_ledger", ()) or ())
    )
    extra_reconcile_probe_obs = sum(
        len(ev.get("observations_serialized") or ev.get("rows_serialized") or ())
        for ev in (getattr(shared_lib, "_raw_reconciliation_evidence_archive", ()) or ())
        if str(ev.get("evidence_id") or "").startswith("reconcile_probe:")
    )
    archive_batches_total = len(getattr(shared_lib, "_raw_reconciliation_evidence_archive", ()) or ())

    # §8.3 Per-chain prefix full cost:
    # Delta C_c(k) = C^R_{maintenance,c} + sum_{t<=k} C^R_{c,t} - C^K_{maintenance,c} - sum_{t<=k} C^K_{c,t}
    # Note: R_lazy_noH's on-demand reconstruction is ALREADY included inside C^R_{c,t}, so C^R_{maintenance,c} = 0.
    c_r_maint = 0
    c_k_maint = reconcile_compute_total
    r_eps = arms_target_results["R_lazy_noH"]
    k_eps = arms_target_results["K_eager_noH"]
    prefix_costs: dict[str, Any] = {}
    for k_step in (1, 2, 3):
        sum_r = sum(r_eps[i]["measured_costs"]["compute_cost"] for i in range(k_step))
        sum_k = sum(k_eps[i]["measured_costs"]["compute_cost"] for i in range(k_step))
        delta_ck = (c_r_maint + sum_r) - (c_k_maint + sum_k)
        prefix_costs[f"k_{k_step}"] = {
            "k": k_step,
            "C_R_maintenance_c": c_r_maint,
            "C_K_maintenance_c": c_k_maint,
            "sum_target_compute_R_lazy_noH": sum_r,
            "sum_target_compute_K_eager_noH": sum_k,
            "delta_C_c_k_R_minus_K": delta_ck,
            "target_only_delta_R_minus_K": sum_r - sum_k,
            "R_strict_passes_up_to_k": sum(1 for i in range(k_step) if r_eps[i]["strict_pass"]),
            "K_strict_passes_up_to_k": sum(1 for i in range(k_step) if k_eps[i]["strict_pass"]),
        }

    return {
        "chain_id": cid,
        "base_seed": base_seed,
        "arrival_order_mode": order_mode,
        "source_arrival_sequence": order_keys,
        "calibration_anchor_episode": cal_out,
        "ordinary_source_episodes": source_episodes,
        "source_formation_measured_ledger": {
            "ordinary_source_episodes_attempted": len(source_episodes),
            "ordinary_source_internally_confirmed": sum(1 for e in source_episodes if e["internally_confirmed"]),
            "ordinary_source_strict_external_passes": sum(1 for e in source_episodes if e["strict_pass"]),
            "ordinary_source_false_internal_confirmations": sum(
                1 for e in source_episodes if e["false_internal_confirmation"]
            ),
            "ordinary_source_unresolved_count": sum(1 for e in source_episodes if e["unresolved_or_unconfirmed"]),
            "source_total_compute_including_reconciliation": sum(
                e["measured_costs"]["compute_cost"] for e in source_episodes
            ),
            "source_reconciliation_maintenance_compute_C_K_maint": reconcile_compute_total,
            "source_ordinary_compute_excluding_reconciliation": sum(
                e["measured_costs"]["compute_cost"] for e in source_episodes
            )
            - reconcile_compute_total,
            "replayable_evidence_archive_batches": archive_batches_total,
            "extra_reconciliation_probe_observations": extra_reconcile_probe_obs,
            "source_total_observations": sum(e["measured_costs"]["observation_cost"] for e in source_episodes),
            "source_total_measurements": sum(e["measured_costs"]["measurement_cost"] for e in source_episodes),
            "source_agent_run_wall_seconds": round(
                sum(e["measured_costs"]["agent_run_wall_seconds"] for e in source_episodes), 4
            ),
        },
        "prefix_full_cost_comparison_R_minus_K": prefix_costs,
        "arms_target_results": arms_target_results,
    }


def _pair_clusters(chains: list[dict[str, Any]], focal_arm: str, ref_arm: str) -> dict[str, Any]:
    sd: list[int] = []
    cd: list[float] = []
    wins = losses = 0
    changed_cases: list[dict[str, Any]] = []
    for ch in chains:
        rr = ch["arms_target_results"][ref_arm]
        kk = ch["arms_target_results"][focal_arm]
        diff = sum(int(k["strict_pass"]) - int(r["strict_pass"]) for k, r in zip(kk, rr))
        dc = sum(r["measured_costs"]["compute_cost"] - k["measured_costs"]["compute_cost"] for k, r in zip(kk, rr)) / len(kk)
        sd.append(diff)
        cd.append(round(dc, 2))
        for r, k in zip(rr, kk):
            w = int(k["strict_pass"]) - int(r["strict_pass"])
            if w == 1:
                wins += 1
            elif w == -1:
                losses += 1
            if w != 0:
                changed_cases.append(
                    {
                        "chain_id": ch["chain_id"],
                        "target_role": k["target_regime_role"],
                        "outcome_delta": "benefited" if w == 1 else "harmed",
                        "focal_arm": {
                            "strict_pass": k["strict_pass"],
                            "status": k["status"],
                            "external_fresh48_mse": k["external_fresh48_mse"],
                            "compute_cost": k["measured_costs"]["compute_cost"],
                        },
                        "reference_arm": {
                            "strict_pass": r["strict_pass"],
                            "status": r["status"],
                            "external_fresh48_mse": r["external_fresh48_mse"],
                            "compute_cost": r["measured_costs"]["compute_cost"],
                        },
                    }
                )
    obs_sum = sum(sd)
    abs_vals = [abs(x) for x in sd if x != 0]
    exact_p = (
        sum(
            abs(sum(s * v for s, v in zip(signs, abs_vals))) >= abs(obs_sum) - 1e-12
            for signs in product((-1, 1), repeat=len(abs_vals))
        )
        / (2 ** len(abs_vals))
        if abs_vals
        else 1.0
    )
    return {
        "focal_arm": focal_arm,
        "reference_arm": ref_arm,
        "independent_source_clusters": len(chains),
        "targets_per_cluster": len(chains[0]["arms_target_results"][focal_arm]),
        "success_difference_per_chain": sd,
        "target_compute_saving_per_chain_mean": cd,
        "task_wins": wins,
        "task_losses": losses,
        "task_ties": len(chains) * len(chains[0]["arms_target_results"][focal_arm]) - wins - losses,
        "success_rate_difference_pp": round(
            100.0 * obs_sum / (len(chains) * len(chains[0]["arms_target_results"][focal_arm])), 2
        ),
        "source_cluster_sign_flip_two_sided_p": round(exact_p, 4),
        "mean_target_compute_saving": round(mean(cd), 2),
        "median_chain_mean_compute_saving": round(median(cd), 2),
        "changed_target_cases": changed_cases,
    }


def run_deliverable_2() -> dict[str, Any]:
    unseen_d2_specs = [
        {
            "chain_id": "unexposed_v14_chain_1031000",
            "base_seed": 1_031_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.20, 0.44, 0.24),
            "source_radii": (0.93, 0.90, 0.94, 0.91),
            "target_thresholds": (0.33, 0.35),
            "target_radii": (0.90, 0.94, 0.92),
        },
        {
            "chain_id": "unexposed_v14_chain_1032000",
            "base_seed": 1_032_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.18, 0.46, 0.22),
            "source_radii": (0.91, 0.94, 0.89, 0.93),
            "target_thresholds": (0.31, 0.37),
            "target_radii": (0.92, 0.89, 0.91),
        },
        {
            "chain_id": "unexposed_v14_chain_1033000",
            "base_seed": 1_033_000,
            "arrival_order_mode": "minority_interleaved",
            "source_thresholds": (0.22, 0.45, 0.26),
            "source_radii": (0.94, 0.91, 0.93, 0.90),
            "target_thresholds": (0.34, 0.36),
            "target_radii": (0.89, 0.92, 0.93),
        },
        {
            "chain_id": "unexposed_v14_chain_1034000",
            "base_seed": 1_034_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.17, 0.43, 0.25),
            "source_radii": (0.92, 0.93, 0.90, 0.94),
            "target_thresholds": (0.30, 0.34),
            "target_radii": (0.91, 0.93, 0.89),
        },
        {
            "chain_id": "unexposed_v14_chain_1035000",
            "base_seed": 1_035_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.21, 0.47, 0.23),
            "source_radii": (0.90, 0.92, 0.95, 0.91),
            "target_thresholds": (0.35, 0.33),
            "target_radii": (0.93, 0.90, 0.92),
        },
        {
            "chain_id": "unexposed_v14_chain_1036000",
            "base_seed": 1_036_000,
            "arrival_order_mode": "minority_interleaved",
            "source_thresholds": (0.19, 0.45, 0.27),
            "source_radii": (0.93, 0.89, 0.92, 0.94),
            "target_thresholds": (0.32, 0.38),
            "target_radii": (0.90, 0.91, 0.90),
        },
        {
            "chain_id": "unexposed_v14_chain_1037000",
            "base_seed": 1_037_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.23, 0.48, 0.21),
            "source_radii": (0.91, 0.95, 0.91, 0.92),
            "target_thresholds": (0.36, 0.32),
            "target_radii": (0.94, 0.88, 0.93),
        },
        {
            "chain_id": "unexposed_v14_chain_1038000",
            "base_seed": 1_038_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.16, 0.42, 0.26),
            "source_radii": (0.94, 0.90, 0.93, 0.89),
            "target_thresholds": (0.29, 0.36),
            "target_radii": (0.91, 0.93, 0.91),
        },
    ]

    chains: list[dict[str, Any]] = []
    for idx, sp in enumerate(unseen_d2_specs, 1):
        ch = _evaluate_d2_chain(sp)
        chains.append(ch)
        print(
            f"[Deliverable 2 {idx}/{len(unseen_d2_specs)}] {sp['chain_id']}: "
            f"src_strict={ch['source_formation_measured_ledger']['ordinary_source_strict_external_passes']}/4, "
            + ", ".join(
                f"{a}={sum(1 for e in ch['arms_target_results'][a] if e['strict_pass'])}/3"
                for a in D2_ARMS
            ),
            flush=True,
        )

    anchors = [ch["calibration_anchor_episode"] for ch in chains]
    ord_sources = [ep for ch in chains for ep in ch["ordinary_source_episodes"]]

    anchors_table = {
        "episodes_total": len(anchors),
        "internally_confirmed": sum(1 for e in anchors if e["internally_confirmed"]),
        "strict_external_passes": sum(1 for e in anchors if e["strict_pass"]),
        "false_internal_confirmations": sum(1 for e in anchors if e["false_internal_confirmation"]),
        "unresolved_or_unconfirmed": sum(1 for e in anchors if e["unresolved_or_unconfirmed"]),
        "mean_compute_cost": round(mean(e["measured_costs"]["compute_cost"] for e in anchors), 2),
        "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in anchors), 2),
        "mean_agent_run_wall_seconds_excluding_ext_eval": round(
            mean(e["measured_costs"]["agent_run_wall_seconds"] for e in anchors), 4
        ),
    }

    ordinary_sources_table = {
        "episodes_total": len(ord_sources),
        "internally_confirmed": sum(1 for e in ord_sources if e["internally_confirmed"]),
        "strict_external_passes": sum(1 for e in ord_sources if e["strict_pass"]),
        "false_internal_confirmations": sum(1 for e in ord_sources if e["false_internal_confirmation"]),
        "unresolved_or_unconfirmed": sum(1 for e in ord_sources if e["unresolved_or_unconfirmed"]),
        "boundary_probes_executed_total": sum(e["boundary_probes_executed_count"] for e in ord_sources),
        "mean_compute_cost_including_reconciliation": round(
            mean(e["measured_costs"]["compute_cost"] for e in ord_sources), 2
        ),
        "mean_reconciliation_maintenance_compute_per_chain": round(
            mean(ch["source_formation_measured_ledger"]["source_reconciliation_maintenance_compute_C_K_maint"] for ch in chains),
            2,
        ),
        "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in ord_sources), 2),
        "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in ord_sources), 2),
        "mean_agent_run_wall_seconds_excluding_ext_eval": round(
            mean(e["measured_costs"]["agent_run_wall_seconds"] for e in ord_sources), 4
        ),
    }

    by_arm: dict[str, Any] = {}
    for arm in D2_ARMS:
        eps = [ep for ch in chains for ep in ch["arms_target_results"][arm]]
        t1 = [e for e in eps if e["target_regime_role"] == "T1_unseen_family"]
        t2 = [e for e in eps if e["target_regime_role"] == "T2_old_unconditional_domain"]
        t3 = [e for e in eps if e["target_regime_role"] == "T3_minority_direction_domain"]
        cold_k1 = [e for e in eps if e["prefix_step_index_k"] == 1]
        warm_k23 = [e for e in eps if e["prefix_step_index_k"] > 1]

        by_arm[arm] = {
            "total_target_episodes": len(eps),
            "independent_source_clusters": len(chains),
            "H_star_front_priority_enabled": False,
            "internally_confirmed": sum(1 for e in eps if e["internally_confirmed"]),
            "strict_external_passes": sum(1 for e in eps if e["strict_pass"]),
            "strict_pass_rate": round(sum(1 for e in eps if e["strict_pass"]) / len(eps), 4),
            "false_internal_confirmations": sum(1 for e in eps if e["false_internal_confirmation"]),
            "unresolved_or_unconfirmed": sum(1 for e in eps if e["unresolved_or_unconfirmed"]),
            "unconfirmed_yet_accurate_count": sum(1 for e in eps if e["unconfirmed_yet_accurate_on_external"]),
            "wrong_knowledge_transfers": sum(1 for e in eps if e["wrong_knowledge_transfer"]),
            "scope_unknown_events_total": sum(e["scope_unknown_events"] for e in eps),
            "boundary_probes_executed_total": sum(e["boundary_probes_executed_count"] for e in eps),
            "by_target_role": {
                "T1_unseen_family_strict_passes": f"{sum(1 for e in t1 if e['strict_pass'])}/{len(t1)}",
                "T2_old_domain_retention_strict_passes": f"{sum(1 for e in t2 if e['strict_pass'])}/{len(t2)}",
                "T3_minority_direction_strict_passes": f"{sum(1 for e in t3 if e['strict_pass'])}/{len(t3)}",
                "T1_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t1), 2),
                "T2_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t2), 2),
                "T3_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t3), 2),
            },
            "unconditional_measured_target_costs_all_24": {
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in eps), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in eps), 2),
                "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in eps), 2),
                "mean_agent_run_wall_seconds_excluding_ext_eval": round(
                    mean(e["measured_costs"]["agent_run_wall_seconds"] for e in eps), 4
                ),
                "mean_total_wall_seconds_including_ext_eval": round(
                    mean(e["measured_costs"]["total_wall_seconds_including_external_eval"] for e in eps), 4
                ),
                "total_compute": sum(e["measured_costs"]["compute_cost"] for e in eps),
            },
            "cache_cold_vs_warm_breakdown": {
                "cold_cache_k1_mean_reconstruction_compute": round(
                    mean(e["ordinary_lazy_episode_reconstruction_compute"] for e in cold_k1), 2
                ),
                "warm_cache_k2_k3_mean_reconstruction_compute": round(
                    mean(e["ordinary_lazy_episode_reconstruction_compute"] for e in warm_k23), 2
                ),
                "warm_cache_k2_k3_first_call_hits": sum(
                    1 for e in warm_k23 if e.get("ordinary_lazy_first_call_cache_hit") is True
                ),
                "warm_cache_k2_k3_episodes": len(warm_k23),
                "unconsumed_layer2_H_star_built_total": sum(
                    e["ordinary_lazy_layer2_families_built"] for e in eps
                ),
            },
        }

    # Descriptive supplement: both-strict-pass subset between R_lazy_noH and K_eager_noH
    both_ok_r = []
    both_ok_k = []
    for ch in chains:
        for r_e, k_e in zip(ch["arms_target_results"]["R_lazy_noH"], ch["arms_target_results"]["K_eager_noH"]):
            if r_e["strict_pass"] and k_e["strict_pass"]:
                both_ok_r.append(r_e["measured_costs"]["compute_cost"])
                both_ok_k.append(k_e["measured_costs"]["compute_cost"])

    prefix_agg = {}
    for k_key in ("k_1", "k_2", "k_3"):
        deltas = [ch["prefix_full_cost_comparison_R_minus_K"][k_key]["delta_C_c_k_R_minus_K"] for ch in chains]
        tgt_deltas = [ch["prefix_full_cost_comparison_R_minus_K"][k_key]["target_only_delta_R_minus_K"] for ch in chains]
        prefix_agg[k_key] = {
            "per_chain_delta_C_c_k_R_minus_K": deltas,
            "mean_delta_C_c_k_R_minus_K": round(mean(deltas), 2),
            "median_delta_C_c_k_R_minus_K": round(median(deltas), 2),
            "chains_where_K_has_lower_full_prefix_cost_than_R": sum(1 for d in deltas if d > 0),
            "mean_target_only_delta_R_minus_K": round(mean(tgt_deltas), 2),
        }

    receipt_d2 = {
        "schema_version": "hegel_v14_deliverable_2_unconfounded_noH_source_paired_study_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v14_Deliverable_2_Unconfounded_NoH_Source_Cluster_Paired_Study",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "retired_exposed_chains_excluded": [
            "1001000..1005000",
            "1010000..1018000",
            "1021000..1028000",
        ],
        "fairness_and_protocol_guarantees": {
            "single_touch_frozen_before_running_1031000_1038000": True,
            "H_star_front_priority_disabled_across_all_arms": True,
            "R_lazy_noH_skips_unconsumed_H_star_reconstruction": True,
            "shared_raw_priors_and_replayable_reconciliation_evidence_archive": True,
            "anchors_separated_from_ordinary_sources": True,
            "wall_clock_reports_agent_run_and_external_eval_separately": True,
            "statistical_unit": "physical_source_chain_cluster_n_equals_8",
        },
        "separate_tables": {
            "common_calibration_anchors_8": anchors_table,
            "ordinary_source_episodes_32": ordinary_sources_table,
            "target_arms_24_per_arm": by_arm,
            "descriptive_both_strict_pass_subset_R_vs_K": {
                "shared_strict_pass_episodes": len(both_ok_r),
                "R_lazy_noH_mean_compute": round(mean(both_ok_r), 2) if both_ok_r else None,
                "K_eager_noH_mean_compute": round(mean(both_ok_k), 2) if both_ok_k else None,
            },
        },
        "prefix_full_cost_summary_R_minus_K": prefix_agg,
        "source_cluster_paired_comparisons": {
            "K_eager_noH_vs_R_lazy_noH": _pair_clusters(chains, "K_eager_noH", "R_lazy_noH"),
            "K_eager_noH_vs_K_cold": _pair_clusters(chains, "K_eager_noH", "K_cold"),
            "R_lazy_noH_vs_K_cold": _pair_clusters(chains, "R_lazy_noH", "K_cold"),
        },
        "unexposed_evaluation_chains": chains,
    }
    out_d2 = OUT_DIR / "deliverable_2_unconfounded_noH_source_paired_study_20261118.json"
    out_d2.write_text(json.dumps(receipt_d2, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return receipt_d2


# ============================================================================
# DELIVERABLE 3: Continuous Repaired Object End-to-End Study (1041000–1046000)
# ============================================================================

def _build_d3_target_libraries(
    cold_snapshot: KnowledgeLibrary,
    repaired_lib: KnowledgeLibrary,
    repaired_concept_id: str | None,
    repaired_semantic_version: str | None = None,
) -> dict[str, KnowledgeLibrary]:
    """Construct the 5 fair Question-B target arms (v15 §3.1–§3.3), with true versioned removal & legal rebuild."""
    if hasattr(repaired_lib, "without_object_version") and repaired_concept_id:
        lib_branch_rm = repaired_lib.without_object_version(
            repaired_concept_id, repaired_semantic_version, mode="branch_materialization"
        )
        lib_obj_rm = repaired_lib.without_object_version(
            repaired_concept_id, repaired_semantic_version, mode="repaired_version"
        )
    else:
        lib_branch_rm = deepcopy(repaired_lib)
        lib_obj_rm = deepcopy(repaired_lib)
        if repaired_concept_id:
            rep_bid = f"branch:{repaired_concept_id}"
            lib_branch_rm._active_reconciled_branches = [
                b
                for b in getattr(lib_branch_rm, "_active_reconciled_branches", ())
                if b.get("claim_id") != rep_bid
                and b.get("source_concept_id") != repaired_concept_id
                and list(b.get("source_concept_ids") or ()) != [repaired_concept_id]
            ]
            lib_obj_rm.meta_priors = [
                p for p in getattr(lib_obj_rm, "meta_priors", ()) if p.get("source_concept_id") != repaired_concept_id
            ]
            if hasattr(lib_obj_rm, "concepts"):
                if isinstance(lib_obj_rm.concepts, list):
                    lib_obj_rm.concepts = [
                        c for c in lib_obj_rm.concepts if getattr(c, "concept_id", None) != repaired_concept_id
                    ]
                elif isinstance(lib_obj_rm.concepts, dict):
                    lib_obj_rm.concepts.pop(repaired_concept_id, None)
            if hasattr(lib_obj_rm, "abstracts") and isinstance(lib_obj_rm.abstracts, list):
                lib_obj_rm.abstracts = [
                    a for a in lib_obj_rm.abstracts if getattr(a, "source_concept_id", None) != repaired_concept_id
                ]
            if hasattr(lib_obj_rm, "definitions") and isinstance(lib_obj_rm.definitions, dict):
                removed_defs = {
                    dk
                    for dk, dv in lib_obj_rm.definitions.items()
                    if getattr(dv, "source_concept_id", None) == repaired_concept_id
                }
                for dk in removed_defs:
                    del lib_obj_rm.definitions[dk]
                if hasattr(lib_obj_rm, "definition_bindings") and isinstance(lib_obj_rm.definition_bindings, dict):
                    for bk in list(lib_obj_rm.definition_bindings.keys()):
                        bval = lib_obj_rm.definition_bindings[bk]
                        tid = bval.get("template_id") if isinstance(bval, dict) else None
                        if (
                            bk in removed_defs
                            or str(bk).startswith(f"{repaired_concept_id}:")
                            or (tid and str(tid).startswith(f"{repaired_concept_id}:"))
                        ):
                            del lib_obj_rm.definition_bindings[bk]
            lib_obj_rm._active_reconciled_branches = [
                b
                for b in getattr(lib_obj_rm, "_active_reconciled_branches", ())
                if b.get("claim_id") != rep_bid
                and b.get("source_concept_id") != repaired_concept_id
                and list(b.get("source_concept_ids") or ()) != [repaired_concept_id]
            ]

    lib_cold = deepcopy(cold_snapshot)
    lib_cold._freeze_branch_writes = True
    lib_cold._allow_on_demand_family_reconstruction = False
    lib_cold._allow_on_demand_routing_reconstruction = False
    lib_cold._prioritize_generalized_families = False

    lib_k = deepcopy(repaired_lib)
    lib_k._freeze_branch_writes = True
    lib_k._allow_on_demand_family_reconstruction = False
    lib_k._allow_on_demand_routing_reconstruction = False
    lib_k._enable_scope_and_routing_guards = True
    lib_k._prioritize_generalized_families = False

    lib_r = deepcopy(repaired_lib)
    lib_r._active_reconciled_branches = []
    lib_r._generalized_branch_commitments = []
    lib_r._retracted_premise_or_claim_ids = set()
    lib_r._premise_commitment_index = None
    lib_r._indexed_prior_fingerprints = None
    lib_r._indexed_group_commitments = None
    lib_r._freeze_branch_writes = True
    lib_r._allow_on_demand_family_reconstruction = True
    lib_r._allow_on_demand_routing_reconstruction = True
    lib_r._enable_scope_and_routing_guards = True
    lib_r._prioritize_generalized_families = False

    lib_branch_rm._premise_commitment_index = None
    lib_branch_rm._indexed_prior_fingerprints = None
    lib_branch_rm._indexed_group_commitments = None
    lib_branch_rm._freeze_branch_writes = True
    lib_branch_rm._allow_on_demand_family_reconstruction = True
    lib_branch_rm._allow_on_demand_routing_reconstruction = True
    lib_branch_rm._enable_scope_and_routing_guards = True
    lib_branch_rm._prioritize_generalized_families = False

    lib_obj_rm._premise_commitment_index = None
    lib_obj_rm._indexed_prior_fingerprints = None
    lib_obj_rm._indexed_group_commitments = None
    lib_obj_rm._freeze_branch_writes = True
    lib_obj_rm._allow_on_demand_family_reconstruction = True
    lib_obj_rm._allow_on_demand_routing_reconstruction = True
    lib_obj_rm._allow_on_demand_raw_history_reconstruction = True
    lib_obj_rm._enable_scope_and_routing_guards = True
    lib_obj_rm._prioritize_generalized_families = False

    return {
        "K_cold": lib_cold,
        "R_lazy_noH": lib_r,
        "K_eager_noH": lib_k,
        "K_minimal_branch_removed_raw_kept": lib_branch_rm,
        "K_minimal_repaired_object_removed_other_sources_kept": lib_obj_rm,
    }


def _evaluate_d3_chain(spec_c: dict[str, Any]) -> dict[str, Any]:
    cid = spec_c["chain_id"]
    base_sd = int(spec_c["base_seed"])
    r_m1, r_m2, r_fm = spec_c["source_radii"]
    d_src, w_src = spec_c["conflict_focal_params"]
    d_t1, w_t1, r_t1 = spec_c["target_t1_focal_params"]
    r_t2 = spec_c["target_t2_markov_radius"]
    th_t3, r_t3 = spec_c["target_t3_counterexample_params"]

    markov_lib = KnowledgeLibrary()
    markov_lib._prioritize_generalized_families = False

    # Step 0: Neutral anchor (`S_cal`)
    cal_ep, _ = _run_episode_with_telemetry(
        arm_name="common_calibration_anchor",
        episode_tag=f"{cid}:S_cal_focus_N4",
        lab_kwargs={
            "seed": base_sd + 5,
            "objects": 4,
            "readout": "focus",
            "relation_direction": "symmetric",
            "relation_condition": "always",
            "condition_threshold": 0.0,
            "relation_radius": 0.92,
            "actuator_curvature": 0.15,
            "name_seed": base_sd + 105,
            "wiring_seed": base_sd + 205,
        },
        library=markov_lib,
        cfg=ResearchConfig(
            seed=base_sd + 5,
            observation_budget=D3_MARKOV_SRC_OBS_BUDGET,
            compute_budget=D3_MARKOV_SRC_COMPUTE_BUDGET,
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
            branch_reconciliation=True,
        ),
        meas_b=D3_MARKOV_SRC_MEAS_BUDGET,
    )
    cold_snapshot = deepcopy(markov_lib)

    # Steps 1 & 2: Ordinary Markov experiences (`S_markov_sym`, `S_markov_right`)
    markov_episodes = []
    for tag_m, off_m, dir_m, rad_m in (
        ("S_markov_sym_N4", 15, "symmetric", r_m1),
        ("S_markov_right_N4", 25, "right", r_m2),
    ):
        sd_m = base_sd + off_m
        ep_m, _ = _run_episode_with_telemetry(
            arm_name="ordinary_markov_source",
            episode_tag=f"{cid}:{tag_m}",
            lab_kwargs={
                "seed": sd_m,
                "objects": 4,
                "readout": "focus",
                "relation_direction": dir_m,
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": rad_m,
                "actuator_curvature": 0.15,
                "name_seed": sd_m + 100,
                "wiring_seed": sd_m + 200,
            },
            library=markov_lib,
            cfg=ResearchConfig(
                seed=sd_m,
                observation_budget=D3_MARKOV_SRC_OBS_BUDGET,
                compute_budget=D3_MARKOV_SRC_COMPUTE_BUDGET,
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
                branch_reconciliation=True,
            ),
            meas_b=D3_MARKOV_SRC_MEAS_BUDGET,
        )
        markov_episodes.append(ep_m)

    initial_markov_branches = [
        {
            "claim_id": b["claim_id"],
            "source_concept_id": b.get("source_concept_id"),
            "semantic_version": b.get("semantic_version"),
            "status": b.get("status"),
        }
        for b in markov_lib._active_reconciled_branches
    ]

    # Step 3 — QUESTION A (§7.2): Parent-task repair worthiness under EQUAL capability & EQUAL budget!
    sd_conf = base_sd + 55
    conflict_lab_kw = {
        "seed": sd_conf,
        "objects": 4,
        "readout": "focus",
        "relation_direction": "right",
        "relation_condition": "always",
        "condition_threshold": 0.0,
        "relation_radius": r_fm,
        "latent_mode": "focal_memory",
        "latent_decay": d_src,
        "latent_weight": w_src,
        "actuator_curvature": 0.15,
        "name_seed": sd_conf + 100,
        "wiring_seed": sd_conf + 200,
    }

    # Arm A1: Un-weakened ordinary failure-retry with full unconditional `latent_state_discovery=True`!
    lib_qA_ordinary = deepcopy(markov_lib)
    lib_qA_ordinary._active_reconciled_branches = []
    lib_qA_ordinary._generalized_branch_commitments = []
    qA_ord_summary, _ = _run_episode_with_telemetry(
        arm_name="QuestionA_ordinary_failure_retry_equal_api",
        episode_tag=f"{cid}:S_conflict_focal_memory_N4",
        lab_kwargs=conflict_lab_kw,
        library=lib_qA_ordinary,
        cfg=ResearchConfig(
            seed=sd_conf,
            observation_budget=D3_CONFLICT_OBS_BUDGET,
            compute_budget=D3_CONFLICT_COMPUTE_BUDGET,
            max_rounds=4,
            minimum_improvement=EXACT_TOLERANCE,
            max_candidates=4,
            use_memory=True,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=True,
            latent_state_discovery=True,
            active_threshold_probing=True,
            state_search=True,
            indexed_commitment_store=True,
            sensor_role_calibration_priors=True,
            occam_premise_minimality=True,
            branch_reconciliation=False,
        ),
        meas_b=D3_CONFLICT_MEAS_BUDGET,
    )

    # Arm A2: Subquestion-driven repair (`branch_reconciliation=True`; tests inherited Markov branches first,
    # opens `reconcile_subq` when both fail, unlocks the exact same `latent_state_discovery` API, confirms on
    # a fresh batch, and re-reconciles locally against historical Markov branches!).
    repaired_lib = deepcopy(markov_lib)
    qA_subq_summary, qA_subq_raw = _run_episode_with_telemetry(
        arm_name="QuestionA_subquestion_driven_repair_equal_api",
        episode_tag=f"{cid}:S_conflict_focal_memory_N4",
        lab_kwargs=conflict_lab_kw,
        library=repaired_lib,
        cfg=ResearchConfig(
            seed=sd_conf,
            observation_budget=D3_CONFLICT_OBS_BUDGET,
            compute_budget=D3_CONFLICT_COMPUTE_BUDGET,
            max_rounds=1,
            minimum_improvement=EXACT_TOLERANCE,
            max_candidates=4,
            use_memory=True,
            world_structure_commitments=True,
            transition_premise_audit=True,
            expanded_premise_revisions=False,
            latent_state_discovery=False,
            active_threshold_probing=False,
            state_search=True,
            indexed_commitment_store=True,
            sensor_role_calibration_priors=True,
            occam_premise_minimality=True,
            branch_reconciliation=True,
        ),
        meas_b=D3_CONFLICT_MEAS_BUDGET,
    )

    # Extract continuous repaired object ledger from `repaired_lib` & `qA_subq_raw` (v15 §2.2: use real production action names & fields!)
    subq_resolve_acts = [
        a for a in qA_subq_raw["actions"] if a.get("action") == "resolve_reconciliation_subquestion"
    ]
    post_rec_acts = [
        a
        for a in qA_subq_raw["actions"]
        if a.get("action") == "autonomous_branch_reconciliation" and a.get("is_post_subresearch_re_reconcile") is True
    ]
    init_rec_acts = [
        a
        for a in qA_subq_raw["actions"]
        if a.get("action") == "autonomous_branch_reconciliation"
        and not a.get("is_post_subresearch_re_reconcile")
    ]

    repaired_cid = qA_subq_summary.get("concept_id")
    if subq_resolve_acts and subq_resolve_acts[-1].get("repaired_concept_id"):
        repaired_cid = subq_resolve_acts[-1]["repaired_concept_id"]

    repaired_branch = next(
        (b for b in repaired_lib._active_reconciled_branches if b.get("claim_id") == f"branch:{repaired_cid}"),
        None,
    )
    repaired_prior = next(
        (p for p in repaired_lib.meta_priors if p.get("source_concept_id") == repaired_cid),
        None,
    )
    repaired_prog_dict = qA_subq_raw.get("program")
    repaired_sem_ver = (
        repaired_branch.get("semantic_version")
        if isinstance(repaired_branch, dict)
        else (_program_semantic_version(repaired_prog_dict) if (repaired_cid and repaired_prog_dict) else None)
    )
    has_accumulate = "accumulate" in json.dumps(repaired_prog_dict or {})
    qualified_repaired = bool(qA_subq_summary["strict_pass"] and repaired_cid and has_accumulate)

    repaired_object_record = {
        "produced_qualified_repaired_object": qualified_repaired,
        "repaired_concept_id": repaired_cid if qualified_repaired else None,
        "repaired_branch_id": f"branch:{repaired_cid}" if qualified_repaired else None,
        "repaired_semantic_version": repaired_sem_ver if qualified_repaired else None,
        "contains_accumulate_operator": has_accumulate,
        "fixed_and_rebindable_parameters": {
            "source_fitted_latent_poles": (repaired_branch or {}).get("commitment", {}).get("latent_poles"),
            "latent_state_mechanism": (repaired_branch or {}).get("commitment", {}).get("latent_state_mechanism"),
            "radius_parameters": (repaired_branch or {}).get("commitment", {}).get("radius_parameters"),
            "rebindable_on_unseen_targets": [
                "arity_N_and_anonymous_object_wiring",
                "locality_radius_r_0",
                "continuous_1d_focal_memory_decay_pole",
                "linear_readout_weights",
            ],
        },
        "initial_conflict_reconciliation_event": {
            "executed": bool(init_rec_acts),
            "epistemic_status": init_rec_acts[0].get("epistemic_status") if init_rec_acts else "missing_evidence",
            "decisions": init_rec_acts[0].get("reconciliation_decisions") if init_rec_acts else [],
            "paid_probes_executed": init_rec_acts[0].get("paid_probes_executed") if init_rec_acts else [],
            "pending_questions": init_rec_acts[0].get("pending_questions") if init_rec_acts else [],
        },
        "subquestion_resolution_event": subq_resolve_acts[-1] if subq_resolve_acts else None,
        "post_subresearch_re_reconciliation_event": {
            "executed": bool(post_rec_acts),
            "epistemic_status": post_rec_acts[-1].get("epistemic_status") if post_rec_acts else "missing_evidence",
            "decisions": post_rec_acts[-1].get("reconciliation_decisions") if post_rec_acts else [],
            "paid_probes_executed": post_rec_acts[-1].get("paid_probes_executed") if post_rec_acts else [],
            "pending_questions": post_rec_acts[-1].get("pending_questions") if post_rec_acts else [],
        },
        "repaired_prior_present_in_library": repaired_prior is not None,
    }

    # Step 4 — QUESTION B (§7.2): Downstream reuse on unseen N=5 targets (`T1`, `T2`, `T3`) across 5 fair arms!
    target_libs = _build_d3_target_libraries(cold_snapshot, repaired_lib, repaired_cid, repaired_sem_ver)
    d3_targets = [
        (
            "T1_unseen_focal_memory_N5",
            f"{cid}:T1_unseen_focal_memory_N5",
            base_sd + 305,
            {
                "seed": base_sd + 305,
                "objects": 5,
                "readout": "focus",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": r_t1,
                "latent_mode": "focal_memory",
                "latent_decay": d_t1,
                "latent_weight": w_t1,
                "actuator_curvature": 0.15,
                "name_seed": base_sd + 405,
                "wiring_seed": base_sd + 505,
            },
        ),
        (
            "T2_return_to_old_markov_N5",
            f"{cid}:T2_return_to_old_markov_N5",
            base_sd + 315,
            {
                "seed": base_sd + 315,
                "objects": 5,
                "readout": "focus",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": r_t2,
                "latent_mode": "none",
                "actuator_curvature": 0.15,
                "name_seed": base_sd + 415,
                "wiring_seed": base_sd + 515,
            },
        ),
        (
            "T3_mismatched_left_conditional_counterexample_N5",
            f"{cid}:T3_mismatched_left_dpg_N5",
            base_sd + 325,
            {
                "seed": base_sd + 325,
                "objects": 5,
                "readout": "focus",
                "relation_direction": "left",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_t3,
                "relation_radius": r_t3,
                "latent_mode": "none",
                "actuator_curvature": 0.15,
                "name_seed": base_sd + 425,
                "wiring_seed": base_sd + 525,
            },
        ),
    ]

    qB_results: dict[str, list[dict[str, Any]]] = {a: [] for a in D3_TARGET_ARMS}
    for arm_b in D3_TARGET_ARMS:
        for k_idx, (t_role, t_tag, t_sd, t_kw) in enumerate(d3_targets, 1):
            l_b = deepcopy(target_libs[arm_b])
            if arm_b in {"R_lazy_noH", "K_minimal_branch_removed_raw_kept"} and hasattr(
                target_libs[arm_b], "_ordinary_commitment_cache"
            ):
                l_b._ordinary_commitment_cache = target_libs[arm_b]._ordinary_commitment_cache
            ep_b, raw_b = _run_episode_with_telemetry(
                arm_name=arm_b,
                episode_tag=t_tag,
                lab_kwargs=t_kw,
                library=l_b,
                cfg=ResearchConfig(
                    seed=t_sd,
                    observation_budget=D3_TARGET_OBS_BUDGET,
                    compute_budget=D3_TARGET_COMPUTE_BUDGET,
                    max_rounds=4,
                    minimum_improvement=EXACT_TOLERANCE,
                    max_candidates=4,
                    use_memory=True,
                    world_structure_commitments=True,
                    transition_premise_audit=True,
                    expanded_premise_revisions=True,
                    latent_state_discovery=True,
                    active_threshold_probing=True,
                    state_search=True,
                    indexed_commitment_store=True,
                    sensor_role_calibration_priors=True,
                    occam_premise_minimality=True,
                    branch_reconciliation=(arm_b != "R_lazy_noH" or True),
                ),
                meas_b=D3_TARGET_MEAS_BUDGET,
            )
            ep_b["target_regime_role"] = t_role
            ep_b["prefix_step_index_k"] = k_idx
            # v15 §5.1: Structured 5-stage adoption check (NEVER substring matching!)
            ep_cons_events = (ep_b.get("causal_consumption_telemetry") or {}).get("consumption_events") or []
            adopted_repaired = any(
                bool(ev.get("adopted_confirmed"))
                and bool(qualified_repaired and repaired_cid)
                and (
                    repaired_cid in (ev.get("source_concept_ids") or ())
                    or str(ev.get("object_id") or "") in {f"branch_comm:{repaired_cid}", f"archive_comm:{repaired_cid}", f"recall_concept:{repaired_cid}"}
                )
                for ev in ep_cons_events
            )
            ep_b["invoked_repaired_object"] = adopted_repaired
            if arm_b in {"R_lazy_noH", "K_minimal_branch_removed_raw_kept", "K_minimal_repaired_object_removed_other_sources_kept"} and hasattr(
                l_b, "_ordinary_commitment_cache"
            ):
                target_libs[arm_b]._ordinary_commitment_cache = l_b._ordinary_commitment_cache
            qB_results[arm_b].append(ep_b)

    # Build status-aware machine-generated 7-Sentence Factual Ledger (v15 §2.2: never emit fake confirmation on failed chains!)
    m_ids = [f"{b['claim_id']}@{b['semantic_version']}" for b in initial_markov_branches]
    t1_k = qB_results["K_eager_noH"][0]
    t1_r = qB_results["R_lazy_noH"][0]
    t1_rm = qB_results["K_minimal_repaired_object_removed_other_sources_kept"][0]
    t2_k = qB_results["K_eager_noH"][1]
    t3_k = qB_results["K_eager_noH"][2]
    subq_q_id = subq_resolve_acts[-1]["question_id"] if subq_resolve_acts else "missing_evidence"
    init_decs = repaired_object_record["initial_conflict_reconciliation_event"]["decisions"]
    post_decs = repaired_object_record["post_subresearch_re_reconciliation_event"]["decisions"]
    post_ep_status = repaired_object_record["post_subresearch_re_reconciliation_event"]["epistemic_status"]

    if qualified_repaired:
        s4_text = (
            f"Subquestion `{subq_q_id}` invoked `latent_state_discovery` on paid fine rows/transitions and replaced "
            f"parent candidate `{subq_resolve_acts[-1]['parent_candidate_before'] if subq_resolve_acts else 'missing_evidence'}` "
            f"with `accumulate`-bearing candidate `{subq_resolve_acts[-1]['parent_candidate_after'] if subq_resolve_acts else 'missing_evidence'}` "
            f"(`{repaired_cid}` @ `{repaired_sem_ver}`)."
        )
        s5_text = (
            f"Repaired object `{repaired_cid}` (`{repaired_sem_ver}`) passed an independent fresh 24-row confirmation probe "
            f"(loss={qA_subq_summary['confirmation_loss']}, external48_mse={qA_subq_summary['external_fresh48_mse']}) "
            f"and completed local post-subresearch re-reconciliation (epistemic_status=`{post_ep_status}`, "
            f"decisions={json.dumps(post_decs, ensure_ascii=False)})."
        )
        s6_text = (
            f"On unseen N=5 target `{t1_k['episode_tag']}`, `K_eager_noH` structured adoption of `{repaired_cid}` was "
            f"`{t1_k['invoked_repaired_object']}` (strict_pass={t1_k['strict_pass']}, compute={t1_k['measured_costs']['compute_cost']}, "
            f"obs={t1_k['measured_costs']['observation_cost']}), whereas true version-removed arm "
            f"`K_minimal_repaired_object_removed_other_sources_kept` achieved strict_pass={t1_rm['strict_pass']}, "
            f"compute={t1_rm['measured_costs']['compute_cost']}, obs={t1_rm['measured_costs']['observation_cost']}; "
            f"old Markov `T2` passed={t2_k['strict_pass']} and mismatched left-conditional `T3` passed={t3_k['strict_pass']}."
        )
    else:
        s4_text = (
            f"Subquestion `{subq_q_id}` did NOT yield a confirmed repaired hypothesis (`status={qA_subq_summary['status']}`, "
            f"produced_qualified_repaired_object=False, subquestion_event="
            f"`{subq_resolve_acts[-1].get('status') if subq_resolve_acts else 'missing_evidence'}`)."
        )
        s5_text = (
            f"missing_evidence: No repaired object passed fresh confirmation on `{cid}:S_conflict_focal_memory_N4` "
            f"(confirmation_loss={qA_subq_summary['confirmation_loss']}, external48_mse={qA_subq_summary['external_fresh48_mse']}); "
            f"post-subresearch re-reconciliation did not confirm a new branch (`epistemic_status={post_ep_status}`)."
        )
        s6_text = (
            f"Because Question A remained `{qA_subq_summary['status']}` with no qualified repaired object, downstream target "
            f"`{t1_k['episode_tag']}` had zero repaired-object adoption (`invoked_repaired_object=False`, "
            f"strict_pass={t1_k['strict_pass']}, compute={t1_k['measured_costs']['compute_cost']})."
        )

    seven_sentences = {
        "1_prior_ordinary_experiences_and_versions": (
            f"After neutral anchor {cal_ep['episode_tag']} and ordinary Markov episodes "
            f"{markov_episodes[0]['episode_tag']} and {markov_episodes[1]['episode_tag']}, the library held "
            f"confirmed Markov branches {m_ids} before encountering {cid}:S_conflict_focal_memory_N4."
        ),
        "2_refuting_real_evidence_and_failed_versions": (
            f"On paid episode {qA_subq_summary['episode_uid']} ({cid}:S_conflict_focal_memory_N4, "
            f"decay={d_src}, weight={w_src}), paid fine-channel observations were evaluated against Markov versions {m_ids} "
            f"(initial_reconcile_status=`{repaired_object_record['initial_conflict_reconciliation_event']['epistemic_status']}`)."
        ),
        "3_reconciliation_choice_rationale": (
            f"Initial reconciliation recorded decisions={json.dumps(init_decs, ensure_ascii=False)} and subquestion=`{subq_q_id}`."
            if init_decs or subq_resolve_acts
            else f"missing_evidence: Question A halted at `{qA_subq_summary['status']}` before producing a confirmed reconciliation decision."
        ),
        "4_subanswer_parent_mutation": s4_text,
        "5_fresh_confirmation_and_historical_branch_checks": s5_text,
        "6_unseen_target_consumption_and_action_change": s6_text,
        "7_equal_capability_ordinary_method_comparison": (
            f"On Question A (`S_conflict`), equal-API ordinary failure-retry (`branch_reconciliation=False, latent_state_discovery=True`) "
            f"achieved strict_pass={qA_ord_summary['strict_pass']} at compute={qA_ord_summary['measured_costs']['compute_cost']}, "
            f"obs={qA_ord_summary['measured_costs']['observation_cost']} vs subquestion repair strict_pass={qA_subq_summary['strict_pass']} "
            f"at compute={qA_subq_summary['measured_costs']['compute_cost']}, obs={qA_subq_summary['measured_costs']['observation_cost']}; "
            f"on Question B (`T1..T3`), same-history `R_lazy_noH` achieved T1 strict_pass={t1_r['strict_pass']} "
            f"(compute={t1_r['measured_costs']['compute_cost']}, obs={t1_r['measured_costs']['observation_cost']}) "
            f"and total 3-target compute={sum(x['measured_costs']['compute_cost'] for x in qB_results['R_lazy_noH'])} vs "
            f"`K_eager_noH`={sum(x['measured_costs']['compute_cost'] for x in qB_results['K_eager_noH'])}."
        ),
    }

    return {
        "chain_id": cid,
        "base_seed": base_sd,
        "calibration_anchor_episode": cal_ep,
        "ordinary_markov_source_episodes": markov_episodes,
        "question_A_parent_repair_equal_api": {
            "ordinary_failure_retry_equal_api": qA_ord_summary,
            "subquestion_driven_repair_equal_api": qA_subq_summary,
            "compute_delta_subq_minus_ordinary": (
                qA_subq_summary["measured_costs"]["compute_cost"]
                - qA_ord_summary["measured_costs"]["compute_cost"]
            ),
            "observation_delta_subq_minus_ordinary": (
                qA_subq_summary["measured_costs"]["observation_cost"]
                - qA_ord_summary["measured_costs"]["observation_cost"]
            ),
        },
        "continuous_repaired_object_ledger": repaired_object_record,
        "arms_target_results": qB_results,
        "seven_sentence_factual_ledger": seven_sentences,
    }


def run_deliverable_3() -> dict[str, Any]:
    unseen_d3_specs = [
        {
            "chain_id": "unexposed_v14_focal_chain_1041000",
            "base_seed": 1_041_000,
            "source_radii": (0.93, 0.91, 0.92),
            "conflict_focal_params": (0.54, 0.36),
            "target_t1_focal_params": (0.68, 0.42, 0.90),
            "target_t2_markov_radius": 0.94,
            "target_t3_counterexample_params": (0.28, 0.91),
        },
        {
            "chain_id": "unexposed_v14_focal_chain_1042000",
            "base_seed": 1_042_000,
            "source_radii": (0.91, 0.94, 0.90),
            "conflict_focal_params": (0.62, 0.34),
            "target_t1_focal_params": (0.48, 0.39, 0.93),
            "target_t2_markov_radius": 0.89,
            "target_t3_counterexample_params": (0.32, 0.92),
        },
        {
            "chain_id": "unexposed_v14_focal_chain_1043000",
            "base_seed": 1_043_000,
            "source_radii": (0.94, 0.90, 0.93),
            "conflict_focal_params": (0.50, 0.38),
            "target_t1_focal_params": (0.72, 0.33, 0.91),
            "target_t2_markov_radius": 0.92,
            "target_t3_counterexample_params": (0.26, 0.90),
        },
        {
            "chain_id": "unexposed_v14_focal_chain_1044000",
            "base_seed": 1_044_000,
            "source_radii": (0.92, 0.93, 0.91),
            "conflict_focal_params": (0.58, 0.35),
            "target_t1_focal_params": (0.45, 0.41, 0.94),
            "target_t2_markov_radius": 0.90,
            "target_t3_counterexample_params": (0.34, 0.89),
        },
        {
            "chain_id": "unexposed_v14_focal_chain_1045000",
            "base_seed": 1_045_000,
            "source_radii": (0.90, 0.92, 0.94),
            "conflict_focal_params": (0.66, 0.32),
            "target_t1_focal_params": (0.53, 0.44, 0.89),
            "target_t2_markov_radius": 0.93,
            "target_t3_counterexample_params": (0.30, 0.93),
        },
        {
            "chain_id": "unexposed_v14_focal_chain_1046000",
            "base_seed": 1_046_000,
            "source_radii": (0.93, 0.89, 0.92),
            "conflict_focal_params": (0.52, 0.37),
            "target_t1_focal_params": (0.64, 0.36, 0.92),
            "target_t2_markov_radius": 0.91,
            "target_t3_counterexample_params": (0.35, 0.90),
        },
    ]

    chains_d3: list[dict[str, Any]] = []
    for idx, sp in enumerate(unseen_d3_specs, 1):
        ch = _evaluate_d3_chain(sp)
        chains_d3.append(ch)
        qa_ord = ch["question_A_parent_repair_equal_api"]["ordinary_failure_retry_equal_api"]["strict_pass"]
        qa_sub = ch["question_A_parent_repair_equal_api"]["subquestion_driven_repair_equal_api"]["strict_pass"]
        print(
            f"[Deliverable 3 {idx}/{len(unseen_d3_specs)}] {sp['chain_id']}: "
            f"QA(ord={qa_ord}, subq={qa_sub}), "
            + ", ".join(
                f"{a}={sum(1 for e in ch['arms_target_results'][a] if e['strict_pass'])}/3"
                for a in D3_TARGET_ARMS
            ),
            flush=True,
        )

    # Summarize Question A
    qa_ord_eps = [ch["question_A_parent_repair_equal_api"]["ordinary_failure_retry_equal_api"] for ch in chains_d3]
    qa_sub_eps = [ch["question_A_parent_repair_equal_api"]["subquestion_driven_repair_equal_api"] for ch in chains_d3]
    question_A_summary = {
        "chains_attempted_total": len(chains_d3),
        "chains_producing_qualified_accumulate_object": sum(
            1 for ch in chains_d3 if ch["continuous_repaired_object_ledger"]["produced_qualified_repaired_object"]
        ),
        "ordinary_failure_retry_equal_api": {
            "strict_external_passes": sum(1 for e in qa_ord_eps if e["strict_pass"]),
            "false_internal_confirmations": sum(1 for e in qa_ord_eps if e["false_internal_confirmation"]),
            "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in qa_ord_eps), 2),
            "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in qa_ord_eps), 2),
            "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in qa_ord_eps), 2),
            "mean_agent_run_wall_seconds": round(
                mean(e["measured_costs"]["agent_run_wall_seconds"] for e in qa_ord_eps), 4
            ),
        },
        "subquestion_driven_repair_equal_api": {
            "strict_external_passes": sum(1 for e in qa_sub_eps if e["strict_pass"]),
            "false_internal_confirmations": sum(1 for e in qa_sub_eps if e["false_internal_confirmation"]),
            "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in qa_sub_eps), 2),
            "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in qa_sub_eps), 2),
            "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in qa_sub_eps), 2),
            "mean_agent_run_wall_seconds": round(
                mean(e["measured_costs"]["agent_run_wall_seconds"] for e in qa_sub_eps), 4
            ),
        },
    }

    # Summarize Question B
    question_B_arms: dict[str, Any] = {}
    for arm in D3_TARGET_ARMS:
        eps = [e for ch in chains_d3 for e in ch["arms_target_results"][arm]]
        t1 = [e for e in eps if e["target_regime_role"] == "T1_unseen_focal_memory_N5"]
        t2 = [e for e in eps if e["target_regime_role"] == "T2_return_to_old_markov_N5"]
        t3 = [e for e in eps if e["target_regime_role"] == "T3_mismatched_left_conditional_counterexample_N5"]
        question_B_arms[arm] = {
            "total_target_episodes": len(eps),
            "strict_external_passes": sum(1 for e in eps if e["strict_pass"]),
            "false_internal_confirmations": sum(1 for e in eps if e["false_internal_confirmation"]),
            "unresolved_or_unconfirmed": sum(1 for e in eps if e["unresolved_or_unconfirmed"]),
            "wrong_knowledge_transfers": sum(1 for e in eps if e["wrong_knowledge_transfer"]),
            "T1_unseen_focal_memory_N5": {
                "strict_passes": f"{sum(1 for e in t1 if e['strict_pass'])}/{len(t1)}",
                "invoked_repaired_object_count": sum(1 for e in t1 if e["invoked_repaired_object"]),
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t1), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in t1), 2),
            },
            "T2_return_to_old_markov_N5": {
                "strict_passes": f"{sum(1 for e in t2 if e['strict_pass'])}/{len(t2)}",
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t2), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in t2), 2),
            },
            "T3_mismatched_left_conditional_counterexample_N5": {
                "strict_passes": f"{sum(1 for e in t3 if e['strict_pass'])}/{len(t3)}",
                "wrong_transfers": sum(1 for e in t3 if e["wrong_knowledge_transfer"]),
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t3), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in t3), 2),
            },
            "overall_measured_costs": {
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in eps), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in eps), 2),
                "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in eps), 2),
                "mean_agent_run_wall_seconds": round(
                    mean(e["measured_costs"]["agent_run_wall_seconds"] for e in eps), 4
                ),
            },
        }

    receipt_d3 = {
        "schema_version": "hegel_v14_deliverable_3_continuous_repaired_object_study_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v14_Deliverable_3_Continuous_Repaired_Object_End_to_End_Study",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "protocol_guarantees": {
            "single_continuous_object_ledger_from_markov_to_conflict_to_unseen_targets": True,
            "all_attempted_source_chains_recorded_without_cherry_picking": True,
            "question_A_equal_latent_api_and_equal_budget": True,
            "question_B_minimal_single_object_removals_preserve_other_sources_and_latent_api": True,
            "H_star_front_priority_disabled": True,
        },
        "question_A_parent_repair_summary": question_A_summary,
        "question_B_downstream_reuse_summary": {
            "arms": question_B_arms,
            "paired_cluster_comparisons": {
                "K_eager_noH_vs_R_lazy_noH": _pair_clusters(chains_d3, "K_eager_noH", "R_lazy_noH"),
                "K_eager_noH_vs_K_minimal_repaired_object_removed": _pair_clusters(
                    chains_d3, "K_eager_noH", "K_minimal_repaired_object_removed_other_sources_kept"
                ),
                "K_eager_noH_vs_K_minimal_branch_removed_raw_kept": _pair_clusters(
                    chains_d3, "K_eager_noH", "K_minimal_branch_removed_raw_kept"
                ),
                "K_eager_noH_vs_K_cold": _pair_clusters(chains_d3, "K_eager_noH", "K_cold"),
            },
        },
        "all_attempted_continuous_chains": chains_d3,
    }
    out_d3 = OUT_DIR / "deliverable_3_continuous_repaired_object_study_20261118.json"
    out_d3.write_text(json.dumps(receipt_d3, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return receipt_d3


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d2 = run_deliverable_2()
    d3 = run_deliverable_3()
    print(
        json.dumps(
            {
                "deliverable_2_summary": {
                    "ordinary_sources_32": d2["separate_tables"]["ordinary_source_episodes_32"],
                    "target_arms": {
                        k: {
                            "strict_passes": v["strict_external_passes"],
                            "false_confirmations": v["false_internal_confirmations"],
                            "mean_compute": v["unconditional_measured_target_costs_all_24"]["mean_compute"],
                            "by_role": v["by_target_role"],
                        }
                        for k, v in d2["separate_tables"]["target_arms_24_per_arm"].items()
                    },
                    "prefix_full_cost_R_minus_K": d2["prefix_full_cost_summary_R_minus_K"],
                },
                "deliverable_3_summary": {
                    "question_A": d3["question_A_parent_repair_summary"],
                    "question_B": {
                        k: {
                            "strict_passes": v["strict_external_passes"],
                            "T1": v["T1_unseen_focal_memory_N5"],
                            "T2": v["T2_return_to_old_markov_N5"],
                            "T3": v["T3_mismatched_left_conditional_counterexample_N5"],
                            "mean_compute": v["overall_measured_costs"]["mean_compute"],
                        }
                        for k, v in d3["question_B_downstream_reuse_summary"]["arms"].items()
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

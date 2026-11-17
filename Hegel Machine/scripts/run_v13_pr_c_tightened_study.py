"""Run v13 PR-C: Tightened Source-Chain-Paired Study on Unexposed Physical Chains (1021000–1028000).

Writes new prospective v13 receipt (ZERO byte modifications to any of the 29 historic v10..v12 JSON receipts):
  ``docs/evidence/v13_evidence_recursion_and_routing_20261117/pr_c_tightened_source_paired_study_20261117.json``

Follows ``markdown/hegel_v13/02_下一轮最小施工与实验合同.md`` §4.C2, §6 (Stopping Rules), & §7 strictly:
  - Retires all v12 chains (``1001000..1005000``, ``1010000..1018000``) to exposed development material.
  - Evaluates 8 brand-new unexposed physical source chains (``1021000..1028000``) under fixed ``M0``
    and identical ``ResearchConfig``, pairing outcomes at the physical **source-chain cluster** level
    (``n = 8`` clusters, 3 targets per cluster = 24 target episodes per arm).
  - All knowledge-bearing arms share **100% identical raw source history**, including raw ``meta_priors``,
    source counterexamples, and reconciliation-acquired probe data (``_raw_reconciliation_evidence_archive``):
      1. ``K_cold``: Neutral starting calibration anchor only.
      2. ``R_ordinary_optimized_lazy``: Strong optimized ordinary lazy archive with 3-layer
         ``OrdinaryPersistentCommitmentCache`` (caching per-prior extractions, anti-unified families,
         AND on-demand reconstructed routing/scope guards within target budget).
      3. ``K_routing_no_H_star``: Learned reconciled branch/routing organization, ``H*`` NOT prioritized
         (``_prioritize_generalized_families = False``).
      4. ``K_routing_plus_H_star``: Identical learned branch/routing organization + ``H*`` prioritized
         (``_prioritize_generalized_families = True``).
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
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    extract_real_consumption_events_from_actions,
)
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v13_evidence_recursion_and_routing_20261117"

EXACT_TOLERANCE = 1e-7
SOURCE_OBS_BUDGET = 660
SOURCE_MEAS_BUDGET = 13_200
SOURCE_COMPUTE_BUDGET = 740_000

TARGET_OBS_BUDGET = 620
TARGET_MEAS_BUDGET = 12_400
TARGET_COMPUTE_BUDGET = 680_000

ARM_NAMES = (
    "K_cold",
    "R_ordinary_optimized_lazy",
    "K_routing_no_H_star",
    "K_routing_plus_H_star",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_config(seed: int, *, obs_b: int, comp_b: int) -> ResearchConfig:
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
        ext_info = _eval_external_fresh_48(lab, res, seed=agent_seed + 920_001)
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
    unconfirmed_yet_accurate = bool((not internally_confirmed) and ext_mse <= EXACT_TOLERANCE)

    evs = extract_real_consumption_events_from_actions(res["actions"])
    retrieved_ids: list[str] = []
    dispatch_tel = None
    ord_ep_recon_compute = 0
    ord_ep_first_hit: bool | None = None
    for act in res["actions"]:
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
                od_st = bdt.get("on_demand_archive_family_reconstruction")
                if isinstance(od_st, dict):
                    ord_ep_recon_compute += int(od_st.get("compute_spent", 0))
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
        "unconfirmed_yet_accurate_on_external": unconfirmed_yet_accurate,
        "wrong_knowledge_transfer": wrong_transfer,
        "within_predeclared_budget": within_budget,
        "measured_costs": {
            "compute_cost": int(costs["compute_cost"]),
            "observation_cost": int(costs["observation_cost"]),
            "measurement_cost": int(costs["measurement_cost"]),
            "queries": int(costs["queries"]),
            "wall_seconds": wall_s,
        },
        "selected_kind": res.get("selected_kind"),
        "ordinary_lazy_episode_reconstruction_compute": ord_ep_recon_compute,
        "ordinary_lazy_first_call_cache_hit": ord_ep_first_hit,
        "branch_dispatch_telemetry": dispatch_tel,
        "causal_consumption_telemetry": chain_tel,
    }


def _build_four_arm_libraries(
    cold_lib: KnowledgeLibrary,
    reconciled_lib: KnowledgeLibrary,
) -> dict[str, KnowledgeLibrary]:
    """Construct the 4 tightened comparison arms sharing 100% identical raw source history."""
    # 1. K_cold: starting calibration anchor only
    lib_cold = deepcopy(cold_lib)
    lib_cold._freeze_branch_writes = True
    lib_cold._allow_on_demand_family_reconstruction = False
    lib_cold._allow_on_demand_routing_reconstruction = False

    # 2. R_ordinary_optimized_lazy: identical raw priors + counterexamples + reconciliation probe archive,
    #    with full 3-layer OrdinaryPersistentCommitmentCache (family + routing/scope reconstruction allowed!)
    lib_r = deepcopy(reconciled_lib)
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
    lib_r._prioritize_generalized_families = True

    # 3. K_routing_no_H_star: learned reconciled branch/routing organization, H* NOT prioritized
    lib_no_h = deepcopy(reconciled_lib)
    lib_no_h._freeze_branch_writes = True
    lib_no_h._allow_on_demand_family_reconstruction = False
    lib_no_h._allow_on_demand_routing_reconstruction = False
    lib_no_h._enable_scope_and_routing_guards = True
    lib_no_h._prioritize_generalized_families = False

    # 4. K_routing_plus_H_star: identical learned branch/routing organization + H* prioritized
    lib_plus_h = deepcopy(reconciled_lib)
    lib_plus_h._freeze_branch_writes = True
    lib_plus_h._allow_on_demand_family_reconstruction = False
    lib_plus_h._allow_on_demand_routing_reconstruction = False
    lib_plus_h._enable_scope_and_routing_guards = True
    lib_plus_h._prioritize_generalized_families = True

    return {
        "K_cold": lib_cold,
        "R_ordinary_optimized_lazy": lib_r,
        "K_routing_no_H_star": lib_no_h,
        "K_routing_plus_H_star": lib_plus_h,
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

    arm_libs = _build_four_arm_libraries(cold_snapshot, shared_lib)

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
            t_lib = deepcopy(arm_libs[arm_name])
            # Preserve persistent 3-layer OrdinaryPersistentCommitmentCache across targets on R_ordinary_optimized_lazy!
            if arm_name == "R_ordinary_optimized_lazy" and hasattr(arm_libs[arm_name], "_ordinary_commitment_cache"):
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
            if arm_name == "R_ordinary_optimized_lazy" and hasattr(t_lib, "_ordinary_commitment_cache"):
                arm_libs[arm_name]._ordinary_commitment_cache = t_lib._ordinary_commitment_cache
            arms_target_results[arm_name].append(t_out)

    reconcile_compute_total = sum(
        int(rec.get("compute_cost", 0))
        for rec in (getattr(shared_lib, "_reconciliation_episode_ledger", ()) or ())
    )
    extra_reconcile_probe_obs = sum(
        len(ev.get("rows_serialized") or ())
        for ev in (getattr(shared_lib, "_raw_reconciliation_evidence_archive", ()) or ())
    )
    formed_objects = [
        {
            "claim_id": b["claim_id"],
            "status": b.get("status"),
            "scope_guard": b.get("scope_guard"),
            "routing_guard": b.get("routing_guard"),
            "counterexamples_count": len(b.get("counterexamples") or ()),
        }
        for b in (getattr(shared_lib, "_active_reconciled_branches", ()) or ())
    ]
    formed_families = [
        {
            "commitment_id": gf["commitment_id"],
            "source_branch_ids": gf.get("source_branch_ids"),
            "parameter_slots": gf.get("parameter_slots"),
            "anti_unified_predicate_skeleton": gf.get("anti_unified_predicate_skeleton"),
            "source_preservation_status": gf.get("source_preservation_status"),
            "scope_guard": gf.get("scope_guard"),
            "routing_guard": gf.get("routing_guard"),
        }
        for gf in (getattr(shared_lib, "_generalized_branch_commitments", ()) or ())
    ]

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
            "reconciliation_subset_compute_measured": reconcile_compute_total,
            "extra_reconciliation_probe_observations": extra_reconcile_probe_obs,
            "source_total_observations_measured": sum(e["measured_costs"]["observation_cost"] for e in source_episodes),
            "source_total_measurements_measured": sum(e["measured_costs"]["measurement_cost"] for e in source_episodes),
            "source_total_wall_seconds": round(sum(e["measured_costs"]["wall_seconds"] for e in source_episodes), 4),
            "formed_H_star_count": len(formed_families),
            "rejected_false_merge_count": len(getattr(shared_lib, "_rejected_merge_records", ()) or ()),
        },
        "formed_branches": formed_objects,
        "formed_parametric_families": formed_families,
        "source_episodes": source_episodes,
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
        "targets_per_cluster": 3,
        "success_difference_per_chain": sd,
        "target_compute_saving_per_chain_mean": cd,
        "task_wins": wins,
        "task_losses": losses,
        "task_ties": 3 * len(chains) - wins - losses,
        "success_rate_difference_pp": round(100.0 * obs_sum / (3 * len(chains)), 2),
        "source_cluster_sign_flip_two_sided_p": round(exact_p, 4),
        "mean_target_compute_saving": round(mean(cd), 2),
        "median_chain_mean_compute_saving": round(median(cd), 2),
        "changed_target_cases": changed_cases,
    }


def _aggregate_study(chains: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in ARM_NAMES:
        eps = [ep for ch in chains for ep in ch["arms_target_results"][arm]]
        t1 = [e for e in eps if e["target_regime_role"] == "T1_unseen_family"]
        t2 = [e for e in eps if e["target_regime_role"] == "T2_old_unconditional_domain"]
        t3 = [e for e in eps if e["target_regime_role"] == "T3_minority_direction_domain"]
        n = len(eps)
        passes = sum(1 for e in eps if e["strict_pass"])
        recon_costs = [
            int(e.get("ordinary_lazy_episode_reconstruction_compute", 0))
            for e in eps
            if e.get("ordinary_lazy_first_call_cache_hit") is not None
        ]
        recon_hits = [
            bool(e.get("ordinary_lazy_first_call_cache_hit", False))
            for e in eps
            if e.get("ordinary_lazy_first_call_cache_hit") is not None
        ]
        by_arm[arm] = {
            "total_target_episodes": n,
            "independent_source_clusters": len(chains),
            "strict_passes": passes,
            "strict_pass_rate": round(passes / max(1, n), 4),
            "unconfirmed_yet_accurate_count": sum(1 for e in eps if e["unconfirmed_yet_accurate_on_external"]),
            "false_internal_confirmations": sum(1 for e in eps if e["false_internal_confirmation"]),
            "wrong_knowledge_transfers": sum(1 for e in eps if e["wrong_knowledge_transfer"]),
            "pure_command_control_deliveries": sum(
                1 for e in eps if e["information_timing"] == "pure_command_control_delivery"
            ),
            "by_target_role": {
                "T1_unseen_family_passes": f"{sum(1 for e in t1 if e['strict_pass'])}/{len(t1)}",
                "T2_old_domain_passes": f"{sum(1 for e in t2 if e['strict_pass'])}/{len(t2)}",
                "T3_minority_direction_passes": f"{sum(1 for e in t3 if e['strict_pass'])}/{len(t3)}",
                "T1_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t1), 2),
                "T2_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t2), 2),
                "T3_mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in t3), 2),
            },
            "external_fresh48_mean_mse": round(mean(e["external_fresh48_mse"] for e in eps), 10),
            "external_fresh48_max_mse": round(max(e["external_fresh48_mse"] for e in eps), 10),
            "measured_target_costs": {
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in eps), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in eps), 2),
                "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in eps), 2),
                "mean_wall_seconds": round(mean(e["measured_costs"]["wall_seconds"] for e in eps), 4),
                "total_compute": sum(e["measured_costs"]["compute_cost"] for e in eps),
                "total_wall_seconds": round(sum(e["measured_costs"]["wall_seconds"] for e in eps), 4),
            },
            "ordinary_lazy_reconstruction_summary": {
                "episodes_invoking_reconstruction": len(recon_costs),
                "cache_hits": sum(1 for h in recon_hits if h),
                "mean_first_or_cached_reconstruction_compute": round(mean(recon_costs), 2) if recon_costs else 0.0,
                "total_reconstruction_compute": sum(recon_costs),
            },
        }

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
        "total_source_false_confirmations": sum(
            ch["source_formation_measured_ledger"]["source_false_internal_confirmations"] for ch in chains
        ),
        "chains_forming_H_star": sum(
            1 for ch in chains if ch["source_formation_measured_ledger"]["formed_H_star_count"] >= 1
        ),
        "mean_source_chain_compute_measured": round(
            mean(ch["source_formation_measured_ledger"]["source_total_compute_measured"] for ch in chains), 2
        ),
        "mean_reconciliation_maintenance_compute_measured": round(
            mean(ch["source_formation_measured_ledger"]["reconciliation_subset_compute_measured"] for ch in chains), 2
        ),
        "mean_extra_reconciliation_probe_observations": round(
            mean(ch["source_formation_measured_ledger"]["extra_reconciliation_probe_observations"] for ch in chains), 2
        ),
        "mean_source_chain_wall_seconds": round(
            mean(ch["source_formation_measured_ledger"]["source_total_wall_seconds"] for ch in chains), 4
        ),
    }

    return {
        "source_formation_and_maintenance_summary": source_summary,
        "arm_aggregates": by_arm,
        "source_cluster_paired_comparisons": {
            "K_routing_no_H_star_vs_R_ordinary_optimized_lazy": _pair_clusters(
                chains, "K_routing_no_H_star", "R_ordinary_optimized_lazy"
            ),
            "K_routing_plus_H_star_vs_R_ordinary_optimized_lazy": _pair_clusters(
                chains, "K_routing_plus_H_star", "R_ordinary_optimized_lazy"
            ),
            "K_routing_plus_H_star_vs_K_routing_no_H_star": _pair_clusters(
                chains, "K_routing_plus_H_star", "K_routing_no_H_star"
            ),
            "K_routing_no_H_star_vs_K_cold": _pair_clusters(
                chains, "K_routing_no_H_star", "K_cold"
            ),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    unseen_specs = [
        {
            "chain_id": "unexposed_v13_chain_1021000",
            "base_seed": 1_021_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.19, 0.45, 0.25),
            "source_radii": (0.94, 0.91, 0.93, 0.90),
            "target_thresholds": (0.32, 0.36),
            "target_radii": (0.89, 0.95, 0.92),
        },
        {
            "chain_id": "unexposed_v13_chain_1022000",
            "base_seed": 1_022_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.23, 0.47, 0.21),
            "source_radii": (0.91, 0.95, 0.90, 0.92),
            "target_thresholds": (0.35, 0.34),
            "target_radii": (0.92, 0.88, 0.93),
        },
        {
            "chain_id": "unexposed_v13_chain_1023000",
            "base_seed": 1_023_000,
            "arrival_order_mode": "minority_interleaved",
            "source_thresholds": (0.17, 0.43, 0.27),
            "source_radii": (0.93, 0.90, 0.94, 0.91),
            "target_thresholds": (0.30, 0.38),
            "target_radii": (0.91, 0.93, 0.89),
        },
        {
            "chain_id": "unexposed_v13_chain_1024000",
            "base_seed": 1_024_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.21, 0.46, 0.23),
            "source_radii": (0.92, 0.94, 0.89, 0.93),
            "target_thresholds": (0.34, 0.33),
            "target_radii": (0.90, 0.91, 0.94),
        },
        {
            "chain_id": "unexposed_v13_chain_1025000",
            "base_seed": 1_025_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.18, 0.44, 0.26),
            "source_radii": (0.95, 0.92, 0.91, 0.89),
            "target_thresholds": (0.31, 0.39),
            "target_radii": (0.88, 0.94, 0.90),
        },
        {
            "chain_id": "unexposed_v13_chain_1026000",
            "base_seed": 1_026_000,
            "arrival_order_mode": "minority_interleaved",
            "source_thresholds": (0.22, 0.48, 0.24),
            "source_radii": (0.90, 0.93, 0.95, 0.92),
            "target_thresholds": (0.36, 0.35),
            "target_radii": (0.93, 0.89, 0.91),
        },
        {
            "chain_id": "unexposed_v13_chain_1027000",
            "base_seed": 1_027_000,
            "arrival_order_mode": "unconditional_first",
            "source_thresholds": (0.16, 0.42, 0.28),
            "source_radii": (0.93, 0.89, 0.92, 0.94),
            "target_thresholds": (0.29, 0.40),
            "target_radii": (0.91, 0.92, 0.88),
        },
        {
            "chain_id": "unexposed_v13_chain_1028000",
            "base_seed": 1_028_000,
            "arrival_order_mode": "conditional_first_reversed",
            "source_thresholds": (0.20, 0.45, 0.22),
            "source_radii": (0.92, 0.91, 0.93, 0.90),
            "target_thresholds": (0.33, 0.31),
            "target_radii": (0.94, 0.90, 0.92),
        },
    ]

    chains = []
    for idx_sp, sp in enumerate(unseen_specs, 1):
        ch_res = _evaluate_chain_spec(sp)
        chains.append(ch_res)
        print(
            f"[PR-C {idx_sp}/{len(unseen_specs)}] {sp['chain_id']} done: "
            + ", ".join(f"{a}={sum(1 for e in ch_res['arms_target_results'][a] if e['strict_pass'])}/3" for a in ARM_NAMES),
            flush=True,
        )
    agg = _aggregate_study(chains)

    receipt = {
        "schema_version": "hegel_v13_pr_c_tightened_source_paired_study_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v13_PR_C_Tightened_Unexposed_Source_Cluster_Paired_Study",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "retired_exposed_v12_chains": [
            "1001000..1005000",
            "1010000..1018000",
        ],
        "protocol_guarantees": {
            "experiment_policy_fixed": "developer_default_M0",
            "shared_raw_history_and_reconciliation_evidence": True,
            "ordinary_3layer_cache_and_routing_reconstruction_enabled": True,
            "statistical_pairing_unit": "physical_source_chain_cluster_n_equals_8",
        },
        "unexposed_8cluster_aggregate": agg,
        "unexposed_evaluation_chains": chains,
    }
    out_path = OUT_DIR / "pr_c_tightened_source_paired_study_20261117.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "written": str(out_path),
                "source_summary": agg["source_formation_and_maintenance_summary"],
                "arms": {
                    k: {
                        "strict_passes": v["strict_passes"],
                        "total": v["total_target_episodes"],
                        "mean_compute": v["measured_target_costs"]["mean_compute"],
                        "mean_wall_seconds": v["measured_target_costs"]["mean_wall_seconds"],
                        "false_confirmations": v["false_internal_confirmations"],
                        "by_role": v["by_target_role"],
                    }
                    for k, v in agg["arm_aggregates"].items()
                },
                "paired_clusters": {
                    k: {
                        "success_difference_per_chain": v["success_difference_per_chain"],
                        "mean_target_compute_saving": v["mean_target_compute_saving"],
                        "sign_flip_p": v["source_cluster_sign_flip_two_sided_p"],
                    }
                    for k, v in agg["source_cluster_paired_comparisons"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

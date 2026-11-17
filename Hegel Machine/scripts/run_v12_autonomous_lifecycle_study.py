"""Run v12 Deliverable B: Ordinary-Entry Autonomous Branch Lifecycle & Causal Removal Study.

Writes new prospective receipt (ZERO byte modifications to any historic v10..v11.1 JSON receipts):
  ``docs/evidence/v12_autonomous_reconciliation_and_utility_20261117/deliverable_b_v12_autonomous_lifecycle_20261117.json``

Follows ``markdown/hegel_v12/02_下一轮最小施工与实验.md`` §3 (B1–B4) strictly:
  - **B1 (Zero Harness Mechanism Routing)**: The harness ONLY pre-generates ``AnonymousObjectLab``
    task instances + fixed ``ResearchConfig`` budgets and invokes the single ordinary entry point
    ``UnifiedResearchAgent(cfg, measurement_budget=..., library=shared_lib).run(lab)``.
    Zero ``if relation_condition == ...`` branches, zero manual calls to ``reconcile_competing_branches``
    or ``propose_anti_unified_generalization`` or ``shrink_claim_to_observable_scope``, zero manual
    ``0.45 / 0.42`` support overrides.
  - **B2 & B3 (Runtime Reconciliation & Honest Denominator Retention)**: Every episode attempt and
    any unconfirmed/unresolved failure is retained in the denominator (zero ``assert`` statements
    dropping failed episodes or chains).
  - **B4 (Full Autonomous Chain + Causal Removal Verification)**:
    * 1 development chain (``dev_chain_1001000``, reported separately)
    * 4 contiguous pre-registered unseen evaluation chains (``1002000, 1003000, 1004000, 1005000``):
      ``S_cal_sym -> S0_always -> S0_rewired_N5 -> S1_dpg -> S2_dpg -> S3_dpg_incremental -> S_left_conflict -> S0_return_N5``
      followed by frozen-library evaluation on unseen target ``T_unseen_family_N5``, unseen old-domain
      target ``T_unseen_old_domain_N5``, and causal object-removal rerun ``T_unseen_family_N5_without_H_star``
      proving that removing ``H*`` changes real runtime candidate generation and compute behavior.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
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
EPISODE_OBS_BUDGET = 660
EPISODE_MEAS_BUDGET = 13_200
EPISODE_COMPUTE_BUDGET = 740_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_ordinary_config(seed: int) -> ResearchConfig:
    return ResearchConfig(
        seed=seed,
        observation_budget=EPISODE_OBS_BUDGET,
        compute_budget=EPISODE_COMPUTE_BUDGET,
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
    """Evaluate frozen returned program on 48 independent external test rows with explicit channel timing."""
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


def _run_ordinary_episode(
    *,
    episode_tag: str,
    lab_kwargs: dict[str, Any],
    library: KnowledgeLibrary,
    agent_seed: int,
) -> dict[str, Any]:
    cfg = _build_ordinary_config(agent_seed)
    t0 = time.perf_counter()
    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = UnifiedResearchAgent(
            cfg, measurement_budget=EPISODE_MEAS_BUDGET, library=library
        ).run(lab)
        ext_info = _eval_external_fresh_48(lab, res, seed=agent_seed + 900_001)
    wall_s = round(time.perf_counter() - t0, 4)

    costs = res["costs"]
    internally_confirmed = bool(
        res["status"] == "completed"
        and res.get("confirmation_loss") is not None
        and float(res["confirmation_loss"]) <= EXACT_TOLERANCE
    )
    ext_mse = ext_info["external_fresh48_mse"]
    within_budget = bool(
        costs["observation_cost"] <= EPISODE_OBS_BUDGET
        and costs["measurement_cost"] <= EPISODE_MEAS_BUDGET
        and costs["compute_cost"] <= EPISODE_COMPUTE_BUDGET
    )
    strict_pass = bool(internally_confirmed and ext_mse <= EXACT_TOLERANCE and within_budget)
    false_confirmation = bool(internally_confirmed and ext_mse > EXACT_TOLERANCE)

    evs = extract_real_consumption_events_from_actions(res["actions"])
    retrieved_ids = []
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
        arm="autonomous_ordinary_entry",
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


def _run_full_autonomous_chain(chain_id: str, base_seed: int, thresholds: tuple[float, float, float, float, float]) -> dict[str, Any]:
    """Run one complete autonomous chain strictly via ``UnifiedResearchAgent.run(lab)``."""
    th1, th2, th3, th_left, th_unseen = thresholds
    shared_lib = KnowledgeLibrary()

    # Pre-declared task sequence (harness passes ONLY task labs to UnifiedResearchAgent.run!)
    source_sequence = [
        (
            f"{chain_id}:S_cal_neutral_N4",
            base_seed + 10,
            {
                "seed": base_seed + 10,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "symmetric",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": 0.92,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 111,
                "wiring_seed": base_seed + 211,
            },
        ),
        (
            f"{chain_id}:S0_right_always_N4",
            base_seed + 20,
            {
                "seed": base_seed + 20,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": 0.94,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 121,
                "wiring_seed": base_seed + 221,
            },
        ),
        (
            f"{chain_id}:S0_rewired_transfer_N5",
            base_seed + 30,
            {
                "seed": base_seed + 30,
                "objects": 5,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": 0.94,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 131,
                "wiring_seed": base_seed + 231,
            },
        ),
        (
            f"{chain_id}:S1_right_dpg_t1_N4",
            base_seed + 40,
            {
                "seed": base_seed + 40,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th1,
                "relation_radius": 0.93,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 141,
                "wiring_seed": base_seed + 241,
            },
        ),
        (
            f"{chain_id}:S2_right_dpg_t2_N4",
            base_seed + 50,
            {
                "seed": base_seed + 50,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th2,
                "relation_radius": 0.91,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 151,
                "wiring_seed": base_seed + 251,
            },
        ),
        (
            f"{chain_id}:S3_right_dpg_t3_incremental_N4",
            base_seed + 60,
            {
                "seed": base_seed + 60,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th3,
                "relation_radius": 0.95,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 161,
                "wiring_seed": base_seed + 261,
            },
        ),
        (
            f"{chain_id}:S_left_conflict_N4",
            base_seed + 70,
            {
                "seed": base_seed + 70,
                "objects": 4,
                "readout": "window_total_event",
                "relation_direction": "left",
                "relation_condition": "donor_positive_gap",
                "condition_threshold": th_left,
                "relation_radius": 0.92,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 171,
                "wiring_seed": base_seed + 271,
            },
        ),
        (
            f"{chain_id}:S0_domain_return_N5",
            base_seed + 80,
            {
                "seed": base_seed + 80,
                "objects": 5,
                "readout": "window_total_event",
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": 0.96,
                "actuator_curvature": 0.15,
                "name_seed": base_seed + 181,
                "wiring_seed": base_seed + 281,
            },
        ),
    ]

    episode_records = []
    for ep_tag, ep_seed, l_kw in source_sequence:
        ep_out = _run_ordinary_episode(
            episode_tag=ep_tag,
            lab_kwargs=l_kw,
            library=shared_lib,
            agent_seed=ep_seed,
        )
        episode_records.append(ep_out)

    # Freeze library for unseen prospective target evaluation & causal H* removal check!
    frozen_with_hstar = deepcopy(shared_lib)
    frozen_with_hstar._freeze_branch_writes = True

    frozen_without_hstar = deepcopy(shared_lib)
    frozen_without_hstar._freeze_branch_writes = True
    frozen_without_hstar._generalized_branch_commitments = []

    unseen_family_kw = {
        "seed": base_seed + 310,
        "objects": 5,
        "readout": "window_total_event",
        "relation_direction": "right",
        "relation_condition": "donor_positive_gap",
        "condition_threshold": th_unseen,
        "relation_radius": 0.89,
        "actuator_curvature": 0.15,
        "name_seed": base_seed + 411,
        "wiring_seed": base_seed + 511,
    }
    unseen_old_kw = {
        "seed": base_seed + 320,
        "objects": 5,
        "readout": "window_total_event",
        "relation_direction": "right",
        "relation_condition": "always",
        "condition_threshold": 0.0,
        "relation_radius": 0.88,
        "actuator_curvature": 0.15,
        "name_seed": base_seed + 421,
        "wiring_seed": base_seed + 521,
    }

    target_with_hstar = _run_ordinary_episode(
        episode_tag=f"{chain_id}:T_unseen_family_with_H_star_N5",
        lab_kwargs=unseen_family_kw,
        library=frozen_with_hstar,
        agent_seed=base_seed + 310,
    )
    target_without_hstar = _run_ordinary_episode(
        episode_tag=f"{chain_id}:T_unseen_family_H_star_removed_N5",
        lab_kwargs=unseen_family_kw,
        library=frozen_without_hstar,
        agent_seed=base_seed + 310,
    )
    target_old_domain = _run_ordinary_episode(
        episode_tag=f"{chain_id}:T_unseen_old_domain_N5",
        lab_kwargs=unseen_old_kw,
        library=frozen_with_hstar,
        agent_seed=base_seed + 320,
    )

    active_branches_summary = [
        {
            "claim_id": b["claim_id"],
            "status": b.get("status"),
            "independent_empirical_support": b.get("independent_empirical_support"),
            "scope_shrink_features": b.get("scope_shrink_features"),
            "scope_guard": b.get("scope_guard"),
            "scope_verification": b.get("scope_verification"),
        }
        for b in (getattr(shared_lib, "_active_reconciled_branches", ()) or ())
    ]
    families_summary = [
        {
            "commitment_id": gf["commitment_id"],
            "predicate_skeleton_id": gf.get("predicate_skeleton_id"),
            "source_branch_ids": gf.get("source_branch_ids"),
            "parameter_slots": gf.get("parameter_slots"),
            "per_branch_substitutions": gf.get("per_branch_substitutions"),
            "scope_guard": gf.get("scope_guard"),
            "scope_verification": gf.get("scope_verification"),
        }
        for gf in (getattr(shared_lib, "_generalized_branch_commitments", ()) or ())
    ]

    with_sites = [
        ev.get("consumption_site")
        for ev in (target_with_hstar["causal_consumption_telemetry"].get("consumption_events") or ())
        if ev.get("event_stage") == "candidate_generation"
    ]
    without_sites = [
        ev.get("consumption_site")
        for ev in (target_without_hstar["causal_consumption_telemetry"].get("consumption_events") or ())
        if ev.get("event_stage") == "candidate_generation"
    ]
    removing_hstar_changed_behavior = bool(
        with_sites != without_sites
        or target_with_hstar["measured_costs"]["compute_cost"]
        != target_without_hstar["measured_costs"]["compute_cost"]
    )

    return {
        "chain_id": chain_id,
        "base_seed": base_seed,
        "thresholds": {
            "t1": th1,
            "t2": th2,
            "t3": th3,
            "t_left": th_left,
            "t_unseen_target": th_unseen,
        },
        "source_formation_measured_ledger": {
            "episodes_attempted": len(episode_records),
            "episodes_confirmed_exact": sum(1 for r in episode_records if r["internally_confirmed"]),
            "episodes_failed_or_unresolved": sum(1 for r in episode_records if not r["internally_confirmed"]),
            "false_internal_confirmations": sum(1 for r in episode_records if r["false_internal_confirmation"]),
            "total_compute": sum(r["measured_costs"]["compute_cost"] for r in episode_records),
            "total_observations": sum(r["measured_costs"]["observation_cost"] for r in episode_records),
            "total_measurements": sum(r["measured_costs"]["measurement_cost"] for r in episode_records),
            "total_wall_seconds": round(sum(r["measured_costs"]["wall_seconds"] for r in episode_records), 4),
        },
        "source_episodes": episode_records,
        "reconciled_library_state": {
            "active_branches": active_branches_summary,
            "generalized_families": families_summary,
            "family_evolution_history": deepcopy(getattr(shared_lib, "_family_evolution_history", [])),
            "rejected_false_merges": deepcopy(getattr(shared_lib, "_rejected_merge_records", [])),
            "reconciliation_episode_ledger": deepcopy(getattr(shared_lib, "_reconciliation_episode_ledger", [])),
        },
        "prospective_target_and_removal_checks": {
            "unseen_family_with_H_star": target_with_hstar,
            "unseen_family_H_star_removed": target_without_hstar,
            "unseen_old_domain_return": target_old_domain,
            "removing_H_star_changed_runtime_behavior": removing_hstar_changed_behavior,
            "consumption_sites_with_H_star": with_sites,
            "consumption_sites_without_H_star": without_sites,
            "compute_delta_with_minus_without_H_star": (
                target_with_hstar["measured_costs"]["compute_cost"]
                - target_without_hstar["measured_costs"]["compute_cost"]
            ),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_chain = _run_full_autonomous_chain(
        "dev_chain_1001000", 1_001_000, (0.22, 0.46, 0.34, 0.25, 0.39)
    )

    eval_configs = [
        ("eval_chain_1002000", 1_002_000, (0.18, 0.44, 0.31, 0.27, 0.37)),
        ("eval_chain_1003000", 1_003_000, (0.24, 0.48, 0.36, 0.21, 0.30)),
        ("eval_chain_1004000", 1_004_000, (0.20, 0.42, 0.33, 0.29, 0.47)),
        ("eval_chain_1005000", 1_005_000, (0.16, 0.45, 0.28, 0.23, 0.38)),
    ]
    eval_chains = [
        _run_full_autonomous_chain(cid, sd, ths)
        for cid, sd, ths in eval_configs
    ]

    summary = {
        "eval_chains_total": len(eval_chains),
        "eval_chains_formed_H_star": sum(
            1 for c in eval_chains if len(c["reconciled_library_state"]["generalized_families"]) >= 1
        ),
        "eval_chains_3branch_incremental_H_star": sum(
            1
            for c in eval_chains
            if any(
                len(gf.get("source_branch_ids") or ()) >= 3
                for gf in c["reconciled_library_state"]["generalized_families"]
            )
        ),
        "eval_chains_recorded_false_merge_rejection": sum(
            1 for c in eval_chains if len(c["reconciled_library_state"]["rejected_false_merges"]) >= 1
        ),
        "eval_unseen_family_strict_passes_with_H_star": sum(
            1 for c in eval_chains if c["prospective_target_and_removal_checks"]["unseen_family_with_H_star"]["strict_pass"]
        ),
        "eval_unseen_family_strict_passes_without_H_star": sum(
            1 for c in eval_chains if c["prospective_target_and_removal_checks"]["unseen_family_H_star_removed"]["strict_pass"]
        ),
        "eval_old_domain_strict_passes": sum(
            1 for c in eval_chains if c["prospective_target_and_removal_checks"]["unseen_old_domain_return"]["strict_pass"]
        ),
        "eval_removing_H_star_changed_behavior_count": sum(
            1 for c in eval_chains if c["prospective_target_and_removal_checks"]["removing_H_star_changed_runtime_behavior"]
        ),
        "eval_total_false_internal_confirmations": sum(
            c["source_formation_measured_ledger"]["false_internal_confirmations"]
            + int(c["prospective_target_and_removal_checks"]["unseen_family_with_H_star"]["false_internal_confirmation"])
            + int(c["prospective_target_and_removal_checks"]["unseen_old_domain_return"]["false_internal_confirmation"])
            for c in eval_chains
        ),
        "mean_unseen_target_compute_with_H_star": round(
            mean(c["prospective_target_and_removal_checks"]["unseen_family_with_H_star"]["measured_costs"]["compute_cost"] for c in eval_chains),
            2,
        ),
        "mean_unseen_target_compute_without_H_star": round(
            mean(c["prospective_target_and_removal_checks"]["unseen_family_H_star_removed"]["measured_costs"]["compute_cost"] for c in eval_chains),
            2,
        ),
    }

    receipt = {
        "schema_version": "hegel_v12_deliverable_b_autonomous_lifecycle_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v12_Deliverable_B_Ordinary_Entry_Autonomous_Branch_Lifecycle",
        "discovery_scope": "finite_open_grammar_catalog_discovery_via_ordinary_UnifiedResearchAgent_entry",
        "harness_isolation_guarantee": {
            "routes_on_hidden_relation_condition": False,
            "manual_merge_pair_lists": False,
            "manual_scope_guard_overrides": False,
            "asserts_dropping_failed_episodes": False,
        },
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "predeclared_episode_budget": {
            "observation_budget": EPISODE_OBS_BUDGET,
            "measurement_budget": EPISODE_MEAS_BUDGET,
            "compute_budget": EPISODE_COMPUTE_BUDGET,
            "exact_confirmation_tolerance": EXACT_TOLERANCE,
        },
        "unseen_evaluation_summary": summary,
        "development_chain": dev_chain,
        "unseen_evaluation_chains": eval_chains,
    }
    out_path = OUT_DIR / "deliverable_b_v12_autonomous_lifecycle_20261117.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"written": str(out_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

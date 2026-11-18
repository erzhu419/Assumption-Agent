#!/usr/bin/env python3
"""Run Single-Touch Prospective v15 Studies E1 (Same-History Organization) & E2 (End-to-End Strategy).

Strictly implements GPT-6 v15 contract (`02_下一轮最小施工与实验合同.md` §7–§11):
  - Frozen code & pre-fixed 6-cluster protocol (`1051000..1056000`, retired `1001..1046` excluded).
  - Zero online API model calls (`model_calls = 0`).
  - Zero splicing between E1 and E2:
      * E2 (`deliverable_4_prospective_e2_end_to_end_strategy_20261118.json`):
        `Ordinary_autonomous_end_to_end` forms its OWN source library `L^Ord_c` (`branch_reconciliation=False`,
        with `OrdinaryPersistentCommitmentCache`) and runs `T1..T6` on `L^Ord_c`;
        `K_autonomous_end_to_end` forms its OWN reconciled source library `L^K_c` (`branch_reconciliation=True`,
        `_prioritize_generalized_families=False`) and runs `T1..T6` on `L^K_c`, under identical `ResearchConfig`
        solver flags, APIs, and budgets.
      * E1 (`deliverable_3_prospective_e1_same_history_organization_20261118.json`):
        Uses the exact same raw history `L^K_c` + identical chronological state-transition function
        (`apply_reconciliation_step_from_evidence`) across `K_eager_persistent`, `R_same_history_lazy`
        (asserting 0 semantic first-dispatch state differences), `K_true_repaired_version_removed_raw_rebuild_allowed`
        (true list-shaped single-version removal + billed raw-history reconstruction), and `K_cold`.
  - Pre-fixed Latin-square order-balanced `k = 1, 3, 6` target prefixes across 3 target roles:
      * `Role_A_compatible_focal_memory`
      * `Role_B_return_to_old_domain`
      * `Role_C_mismatched_counterexample`
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import time
from typing import Any

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.evaluation import loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import (
    KnowledgeLibrary,
    export_library_object_inventory,
)
from hegel_machine.cognition.meta_prior_research import _sources
from hegel_machine.cognition.models import ResearchConfig
from hegel_machine.cognition.premise_commitment_index import (
    OrdinaryPersistentCommitmentCache,
    compute_semantic_version,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v15_true_intervention_and_strategy_20261118"
EXACT_TOLERANCE = 1e-7

# Pre-fixed equal budgets across Ordinary and Recursive arms (§7.2, §8.1, §8.2)
SRC_MARKOV_OBS_BUDGET = 600
SRC_MARKOV_MEAS_BUDGET = 12_200
SRC_MARKOV_COMPUTE_BUDGET = 680_000

SRC_CONFLICT_OBS_BUDGET = 600
SRC_CONFLICT_MEAS_BUDGET = 12_200
SRC_CONFLICT_COMPUTE_BUDGET = 880_000

TARGET_OBS_BUDGET = 600
TARGET_MEAS_BUDGET = 12_200
TARGET_COMPUTE_BUDGET = 840_000

DISPATCH_LIST_FIELDS = (
    "support_invalidated",
    "empirically_excluded_on_context",
    "empirically_excluded_bound_instance_only",
    "not_scheduled_by_router",
    "scope_not_established",
)

# Pre-fixed 6 unexposed prospective clusters (`1051000..1056000`) with cyclic Latin-square target ordering
PROSPECTIVE_V15_CLUSTERS: list[dict[str, Any]] = [
    {
        "cluster_id": "prospective_v15_cluster_1051000",
        "base_seed": 1_051_000,
        "target_order_pattern": ("A1", "B1", "C1", "A2", "B2", "C2"),
        "source_params": {"r_sym": 0.92, "r_right": 0.90, "th_dpg": 0.21, "r_dpg": 0.93, "d_focal": 0.55, "w_focal": 0.36, "r_focal": 0.91},
        "target_params": {
            "A1": ("Role_A_compatible_focal_memory", {"latent_decay": 0.67, "latent_weight": 0.41, "relation_radius": 0.90}),
            "B1": ("Role_B_return_to_old_domain", {"mode": "right_always", "relation_radius": 0.93}),
            "C1": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.27, "relation_radius": 0.91}),
            "A2": ("Role_A_compatible_focal_memory", {"latent_decay": 0.46, "latent_weight": 0.38, "relation_radius": 0.93}),
            "B2": ("Role_B_return_to_old_domain", {"mode": "right_dpg", "condition_threshold": 0.33, "relation_radius": 0.90}),
            "C2": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.36, "relation_radius": 0.92}),
        },
    },
    {
        "cluster_id": "prospective_v15_cluster_1052000",
        "base_seed": 1_052_000,
        "target_order_pattern": ("B1", "C1", "A1", "B2", "C2", "A2"),
        "source_params": {"r_sym": 0.91, "r_right": 0.93, "th_dpg": 0.19, "r_dpg": 0.90, "d_focal": 0.61, "w_focal": 0.34, "r_focal": 0.92},
        "target_params": {
            "A1": ("Role_A_compatible_focal_memory", {"latent_decay": 0.49, "latent_weight": 0.40, "relation_radius": 0.92}),
            "B1": ("Role_B_return_to_old_domain", {"mode": "right_always", "relation_radius": 0.89}),
            "C1": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.31, "relation_radius": 0.92}),
            "A2": ("Role_A_compatible_focal_memory", {"latent_decay": 0.69, "latent_weight": 0.35, "relation_radius": 0.90}),
            "B2": ("Role_B_return_to_old_domain", {"mode": "right_dpg", "condition_threshold": 0.31, "relation_radius": 0.93}),
            "C2": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.24, "relation_radius": 0.90}),
        },
    },
    {
        "cluster_id": "prospective_v15_cluster_1053000",
        "base_seed": 1_053_000,
        "target_order_pattern": ("C1", "A1", "B1", "C2", "A2", "B2"),
        "source_params": {"r_sym": 0.93, "r_right": 0.91, "th_dpg": 0.23, "r_dpg": 0.92, "d_focal": 0.52, "w_focal": 0.37, "r_focal": 0.90},
        "target_params": {
            "A1": ("Role_A_compatible_focal_memory", {"latent_decay": 0.64, "latent_weight": 0.37, "relation_radius": 0.91}),
            "B1": ("Role_B_return_to_old_domain", {"mode": "right_always", "relation_radius": 0.92}),
            "C1": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.29, "relation_radius": 0.90}),
            "A2": ("Role_A_compatible_focal_memory", {"latent_decay": 0.44, "latent_weight": 0.42, "relation_radius": 0.94}),
            "B2": ("Role_B_return_to_old_domain", {"mode": "right_dpg", "condition_threshold": 0.35, "relation_radius": 0.91}),
            "C2": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.38, "relation_radius": 0.93}),
        },
    },
    {
        "cluster_id": "prospective_v15_cluster_1054000",
        "base_seed": 1_054_000,
        "target_order_pattern": ("A1", "B1", "C1", "A2", "B2", "C2"),
        "source_params": {"r_sym": 0.90, "r_right": 0.92, "th_dpg": 0.18, "r_dpg": 0.94, "d_focal": 0.58, "w_focal": 0.35, "r_focal": 0.91},
        "target_params": {
            "A1": ("Role_A_compatible_focal_memory", {"latent_decay": 0.47, "latent_weight": 0.41, "relation_radius": 0.93}),
            "B1": ("Role_B_return_to_old_domain", {"mode": "right_always", "relation_radius": 0.90}),
            "C1": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.33, "relation_radius": 0.89}),
            "A2": ("Role_A_compatible_focal_memory", {"latent_decay": 0.66, "latent_weight": 0.36, "relation_radius": 0.91}),
            "B2": ("Role_B_return_to_old_domain", {"mode": "right_dpg", "condition_threshold": 0.29, "relation_radius": 0.92}),
            "C2": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.25, "relation_radius": 0.92}),
        },
    },
    {
        "cluster_id": "prospective_v15_cluster_1055000",
        "base_seed": 1_055_000,
        "target_order_pattern": ("B1", "C1", "A1", "B2", "C2", "A2"),
        "source_params": {"r_sym": 0.92, "r_right": 0.89, "th_dpg": 0.22, "r_dpg": 0.91, "d_focal": 0.56, "w_focal": 0.38, "r_focal": 0.93},
        "target_params": {
            "A1": ("Role_A_compatible_focal_memory", {"latent_decay": 0.63, "latent_weight": 0.39, "relation_radius": 0.89}),
            "B1": ("Role_B_return_to_old_domain", {"mode": "right_always", "relation_radius": 0.94}),
            "C1": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.26, "relation_radius": 0.91}),
            "A2": ("Role_A_compatible_focal_memory", {"latent_decay": 0.48, "latent_weight": 0.34, "relation_radius": 0.92}),
            "B2": ("Role_B_return_to_old_domain", {"mode": "right_dpg", "condition_threshold": 0.34, "relation_radius": 0.90}),
            "C2": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.37, "relation_radius": 0.90}),
        },
    },
    {
        "cluster_id": "prospective_v15_cluster_1056000",
        "base_seed": 1_056_000,
        "target_order_pattern": ("C1", "A1", "B1", "C2", "A2", "B2"),
        "source_params": {"r_sym": 0.91, "r_right": 0.94, "th_dpg": 0.20, "r_dpg": 0.92, "d_focal": 0.53, "w_focal": 0.36, "r_focal": 0.90},
        "target_params": {
            "A1": ("Role_A_compatible_focal_memory", {"latent_decay": 0.65, "latent_weight": 0.38, "relation_radius": 0.92}),
            "B1": ("Role_B_return_to_old_domain", {"mode": "right_always", "relation_radius": 0.91}),
            "C1": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.34, "relation_radius": 0.90}),
            "A2": ("Role_A_compatible_focal_memory", {"latent_decay": 0.45, "latent_weight": 0.40, "relation_radius": 0.91}),
            "B2": ("Role_B_return_to_old_domain", {"mode": "right_dpg", "condition_threshold": 0.32, "relation_radius": 0.93}),
            "C2": ("Role_C_mismatched_counterexample", {"condition_threshold": 0.28, "relation_radius": 0.93}),
        },
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_code_provenance() -> dict[str, Any]:
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tracked = [
        "src/hegel_machine/cognition/premise_commitment_index.py",
        "src/hegel_machine/cognition/memory.py",
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/unified_research.py",
        "scripts/run_v15_prospective_e1_e2_studies.py",
    ]
    return {
        "git_base_commit": git_head,
        "frozen_source_sha256": {rel: _sha256_file(REPO_ROOT / rel) for rel in tracked},
    }


def _make_equal_policy_cfg(seed: int, *, obs_b: int, comp_b: int, branch_reconciliation: bool) -> ResearchConfig:
    """Identical ResearchConfig across Recursive and Ordinary arms, differing ONLY in `branch_reconciliation` (§7.2)."""
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
        latent_state_discovery=True,
        active_threshold_probing=True,
        state_search=True,
        indexed_commitment_store=branch_reconciliation,
        sensor_role_calibration_priors=True,
        occam_premise_minimality=True,
        branch_reconciliation=branch_reconciliation,
    )


def _normalize_dispatch(telemetry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(telemetry, dict) or not telemetry.get("branch_identity_map"):
        return None
    identity: dict[str, str] = {}
    levels: dict[str, Any] = {}
    for item in telemetry["branch_identity_map"]:
        sources = item["source_concept_ids"]
        src = sources[0] if len(sources) == 1 else "+".join(sorted(sources))
        identity[item["branch_id"]] = src
        levels[src] = item.get("scope_claim_level")
    normalized = {
        field: sorted(identity[bid] for bid in telemetry.get(field, ()))
        for field in DISPATCH_LIST_FIELDS
    }
    normalized["scope_claim_level"] = dict(sorted(levels.items()))
    return normalized


def _decompose_episode_costs(actions: list[dict[str, Any]], total_compute: int) -> dict[str, int]:
    """7-bucket compute cost decomposition (§7.1)."""
    b_obj_and_audit = 0
    b_hist_rescore = 0
    b_discrim_exp = 0
    b_rebind_and_new_mech = 0
    b_fresh_confirm = 0
    b_branch_reconcile_and_index = 0
    b_on_demand_recon_and_consume = 0

    for act in actions:
        a_name = str(act.get("action") or "")
        c = int(act.get("compute_cost") or 0)
        if a_name in {"adopt_discovered_objects", "retain_object_ambiguity", "audit_transition_premises", "audit_stochastic_transition_premises", "research_instrument_mapping"}:
            b_obj_and_audit += c
        elif a_name in {"autonomous_branch_reconciliation"}:
            b_branch_reconcile_and_index += c
        elif a_name in {"retrieve_indexed_world_commitments", "consolidate_world_structure_commitments"}:
            b_branch_reconcile_and_index += c
        elif a_name in {"retrieve_ordinary_cached_commitments", "complete_raw_history_object_reconstruction"}:
            b_on_demand_recon_and_consume += c
        elif a_name == "apply_world_structure_commitments":
            bdt = act.get("branch_dispatch_telemetry") or {}
            od = (bdt.get("on_demand_archive_family_reconstruction") or {})
            od_c = int(od.get("compute_spent") or 0)
            b_on_demand_recon_and_consume += od_c
            b_rebind_and_new_mech += max(0, c - od_c)
        elif a_name in {"condition_inherited_relation_intervals", "execute_research_programs", "compose_relation_kernels", "research_conditional_matched_experiments", "refine_conditional_measured_boundary"}:
            b_rebind_and_new_mech += c
        elif a_name in {"plan_inherited_relation_experiment", "plan_competing_program_experiment", "propose_conditional_experiments", "plan_instrument_disagreement"}:
            b_discrim_exp += c
        elif a_name in {"compare_continuation_paid_evidence"}:
            b_hist_rescore += c
        elif "confirm" in a_name:
            b_fresh_confirm += c

    accounted = (
        b_obj_and_audit + b_hist_rescore + b_discrim_exp
        + b_rebind_and_new_mech + b_fresh_confirm
        + b_branch_reconcile_and_index + b_on_demand_recon_and_consume
    )
    residual_fit_and_confirm = max(0, int(total_compute) - accounted)
    return {
        "1_initial_object_and_premise_audit_compute": b_obj_and_audit,
        "2_historical_instance_rescoring_compute": b_hist_rescore,
        "3_discriminating_experiment_planning_compute": b_discrim_exp,
        "4_parameter_rebinding_and_mechanism_search_compute": b_rebind_and_new_mech,
        "5_confirmation_and_selection_scoring_compute": b_fresh_confirm + residual_fit_and_confirm,
        "6_branch_reconciliation_and_index_compute": b_branch_reconcile_and_index,
        "7_on_demand_or_raw_history_reconstruction_compute": b_on_demand_recon_and_consume,
        "total_compute_cost": int(total_compute),
    }


def _run_episode_v15(
    *,
    arm_name: str,
    episode_tag: str,
    lab_kwargs: dict[str, Any],
    library: KnowledgeLibrary,
    cfg: ResearchConfig,
    meas_b: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    t0 = time.perf_counter()
    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = UnifiedResearchAgent(cfg, measurement_budget=meas_b, library=library).run(lab)
        agent_wall = round(time.perf_counter() - t0, 4)
        t_ev0 = time.perf_counter()
        sel_ch = tuple(res.get("selected_channels") or ())
        prog = Program.from_dict(res["program"])
        req_vars = {nm for nm, _ in prog.observed_variables()}
        cmd_vars = set(lab.spec_for(()).feature_names)
        eval_ch = ("fine",) if (sel_ch or not (req_vars <= (cmd_vars | {"__target__"}))) else ()
        ext_rows = lab.evaluation_rows(eval_ch, seed=int(cfg.seed) + 920_001, length=24, trajectories=2)
        ext_mse = float(loss(prog, ext_rows, float(res.get("fallback", 0.0)), 0.5))
        eval_wall = round(time.perf_counter() - t_ev0, 4)

    costs = res["costs"]
    within_budget = bool(
        costs["observation_cost"] <= cfg.observation_budget
        and costs["measurement_cost"] <= meas_b
        and costs["compute_cost"] <= cfg.compute_budget
    )
    internally_confirmed = bool(
        res["status"] == "completed"
        and res.get("confirmation_loss") is not None
        and float(res["confirmation_loss"]) <= EXACT_TOLERANCE
    )
    accurate_external = bool(ext_mse <= EXACT_TOLERANCE)
    strict_pass = bool(internally_confirmed and accurate_external and within_budget)
    false_confirmation = bool(internally_confirmed and not accurate_external)
    unconfirmed_yet_accurate = bool((not internally_confirmed) and accurate_external)

    # Extract first-dispatch telemetry, raw-history reconstruction events, & 5-stage consumption attribution
    dispatch_tel = None
    scope_not_est: list[str] = []
    scope_excl: list[str] = []
    raw_recon_events: list[dict[str, Any]] = []
    resolution_cons: list[dict[str, Any]] = []
    on_demand_recon_compute = 0
    on_demand_first_hit: bool | None = None
    retained_competition = False
    deferred_confirmation = False

    for act in res["actions"]:
        an = act.get("action")
        if an == "complete_raw_history_object_reconstruction":
            raw_recon_events.append(deepcopy(act))
            on_demand_recon_compute += int(act.get("compute_cost", 0))
        elif an == "retrieve_ordinary_cached_commitments":
            on_demand_recon_compute += int(act.get("compute_cost", 0))
            if on_demand_first_hit is None:
                on_demand_first_hit = bool(act.get("cache_hit", False))
        elif an == "apply_world_structure_commitments":
            bdt = act.get("branch_dispatch_telemetry")
            if bdt and dispatch_tel is None:
                dispatch_tel = bdt
            if isinstance(bdt, dict):
                for u_id in bdt.get("scope_not_established") or ():
                    if u_id not in scope_not_est:
                        scope_not_est.append(u_id)
                for ex_id in bdt.get("empirically_excluded_on_context") or ():
                    if ex_id not in scope_excl:
                        scope_excl.append(ex_id)
                od = bdt.get("on_demand_archive_family_reconstruction")
                if isinstance(od, dict):
                    on_demand_recon_compute += int(od.get("compute_spent", 0))
                    if on_demand_first_hit is None:
                        on_demand_first_hit = bool(od.get("cache_hit", False))
        elif an == "resolve_commitment_consumption":
            for ev in act.get("consumption_events") or ():
                resolution_cons.append(deepcopy(ev))
        elif an == "retain_relation_competition":
            retained_competition = True
        elif an == "defer_frozen_confirmation":
            deferred_confirmation = True

    if strict_pass:
        fail_reason = "none_strict_pass"
    elif not within_budget:
        fail_reason = "exceeded_budget"
    elif unconfirmed_yet_accurate:
        if deferred_confirmation:
            fail_reason = "budget_reserved_before_confirmation"
        elif retained_competition:
            fail_reason = "confirmation_competition_unresolved"
        else:
            fail_reason = "confirmation_not_completed"
    elif false_confirmation:
        fail_reason = "false_internal_confirmation"
    else:
        fail_reason = "research_candidate_inaccurate_or_budget_exhausted"

    adopted_sources = sorted(
        {
            str(s)
            for ev in resolution_cons
            if ev.get("adopted_confirmed")
            for s in (ev.get("source_concept_ids") or ())
        }
    )
    wrong_transfer = bool(adopted_sources and false_confirmation)
    cid_added = res.get("concept_id") or ((res.get("concepts_added") or [None])[0])

    summary = {
        "arm": arm_name,
        "episode_tag": episode_tag,
        "task_id": res["task_id"],
        "episode_uid": res.get("episode_uid"),
        "concept_id": cid_added,
        "quality_and_deliverability_sec9_1": {
            "accurate_model_found_external": accurate_external,
            "legally_completed_status": res["status"] == "completed",
            "independent_confirmation_completed": internally_confirmed,
            "strict_pass": strict_pass,
            "false_internal_confirmation": false_confirmation,
            "unresolved_or_unconfirmed": not internally_confirmed,
            "unconfirmed_yet_accurate_on_external": unconfirmed_yet_accurate,
            "wrong_knowledge_transfer": wrong_transfer,
            "failure_stage_diagnosis": fail_reason,
        },
        "status": res["status"],
        "internally_confirmed": internally_confirmed,
        "strict_pass": strict_pass,
        "false_internal_confirmation": false_confirmation,
        "unresolved_or_unconfirmed": not internally_confirmed,
        "unconfirmed_yet_accurate_on_external": unconfirmed_yet_accurate,
        "wrong_knowledge_transfer": wrong_transfer,
        "confirmation_loss": res.get("confirmation_loss"),
        "external_fresh48_mse": ext_mse,
        "within_predeclared_budget": within_budget,
        "measured_costs": {
            "compute_cost": int(costs["compute_cost"]),
            "observation_cost": int(costs["observation_cost"]),
            "measurement_cost": int(costs["measurement_cost"]),
            "queries": int(costs["queries"]),
            "agent_run_wall_seconds": agent_wall,
            "external_eval_wall_seconds": eval_wall,
            "total_wall_seconds": round(agent_wall + eval_wall, 4),
        },
        "cost_decomposition_sec7_1": _decompose_episode_costs(res["actions"], int(costs["compute_cost"])),
        "on_demand_or_raw_reconstruction_compute": on_demand_recon_compute,
        "first_call_cache_hit": on_demand_first_hit,
        "dispatcher_scope_not_established": scope_not_est,
        "dispatcher_empirically_excluded_on_context": scope_excl,
        "branch_dispatch_telemetry": dispatch_tel,
        "raw_history_reconstruction_events": raw_recon_events,
        "structured_5stage_consumption_events": resolution_cons,
        "confirmed_adopted_source_ids": adopted_sources,
    }
    return summary, res


def _build_target_lab_kwargs(base_sd: int, step_idx: int, slot_key: str, role_name: str, params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    t_sd = base_sd + 300 + 15 * step_idx
    common = {
        "seed": t_sd,
        "objects": 5,
        "readout": "focus",
        "actuator_curvature": 0.15,
        "name_seed": t_sd + 100,
        "wiring_seed": t_sd + 200,
    }
    if role_name == "Role_A_compatible_focal_memory":
        return t_sd, {
            **common,
            "relation_direction": "right",
            "relation_condition": "always",
            "condition_threshold": 0.0,
            "relation_radius": params["relation_radius"],
            "latent_mode": "focal_memory",
            "latent_decay": params["latent_decay"],
            "latent_weight": params["latent_weight"],
        }
    if role_name == "Role_B_return_to_old_domain":
        if params["mode"] == "right_always":
            return t_sd, {
                **common,
                "relation_direction": "right",
                "relation_condition": "always",
                "condition_threshold": 0.0,
                "relation_radius": params["relation_radius"],
                "latent_mode": "none",
            }
        return t_sd, {
            **common,
            "relation_direction": "right",
            "relation_condition": "donor_positive_gap",
            "condition_threshold": params["condition_threshold"],
            "relation_radius": params["relation_radius"],
            "latent_mode": "none",
        }
    return t_sd, {
        **common,
        "relation_direction": "left",
        "relation_condition": "donor_positive_gap",
        "condition_threshold": params["condition_threshold"],
        "relation_radius": params["relation_radius"],
        "latent_mode": "none",
    }


def _sign_flip_p(vals: list[float]) -> float:
    nz = [x for x in vals if abs(x) > 1e-12]
    if not nz:
        return 1.0
    obs = abs(sum(nz))
    return sum(
        abs(sum(s * abs(x) for s, x in zip(signs, nz))) >= obs - 1e-9
        for signs in product((-1, 1), repeat=len(nz))
    ) / (2 ** len(nz))


def _evaluate_single_cluster(cluster_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cid = cluster_spec["cluster_id"]
    base_sd = int(cluster_spec["base_seed"])
    sp = cluster_spec["source_params"]

    src_defs = [
        ("S_cal_sym_N4", base_sd + 5, SRC_MARKOV_OBS_BUDGET, SRC_MARKOV_MEAS_BUDGET, SRC_MARKOV_COMPUTE_BUDGET, {"seed": base_sd + 5, "objects": 4, "readout": "focus", "relation_direction": "symmetric", "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": sp["r_sym"], "actuator_curvature": 0.15, "name_seed": base_sd + 105, "wiring_seed": base_sd + 205}),
        ("S1_markov_sym_N4", base_sd + 15, SRC_MARKOV_OBS_BUDGET, SRC_MARKOV_MEAS_BUDGET, SRC_MARKOV_COMPUTE_BUDGET, {"seed": base_sd + 15, "objects": 4, "readout": "focus", "relation_direction": "symmetric", "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": sp["r_sym"] - 0.01, "actuator_curvature": 0.15, "name_seed": base_sd + 115, "wiring_seed": base_sd + 215}),
        ("S2_markov_right_N4", base_sd + 25, SRC_MARKOV_OBS_BUDGET, SRC_MARKOV_MEAS_BUDGET, SRC_MARKOV_COMPUTE_BUDGET, {"seed": base_sd + 25, "objects": 4, "readout": "focus", "relation_direction": "right", "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": sp["r_right"], "actuator_curvature": 0.15, "name_seed": base_sd + 125, "wiring_seed": base_sd + 225}),
        ("S3_right_dpg_N4", base_sd + 35, SRC_MARKOV_OBS_BUDGET, SRC_MARKOV_MEAS_BUDGET, SRC_MARKOV_COMPUTE_BUDGET, {"seed": base_sd + 35, "objects": 4, "readout": "focus", "relation_direction": "right", "relation_condition": "donor_positive_gap", "condition_threshold": sp["th_dpg"], "relation_radius": sp["r_dpg"], "actuator_curvature": 0.15, "name_seed": base_sd + 135, "wiring_seed": base_sd + 235}),
        ("S4_conflict_focal_memory_N4", base_sd + 45, SRC_CONFLICT_OBS_BUDGET, SRC_CONFLICT_MEAS_BUDGET, SRC_CONFLICT_COMPUTE_BUDGET, {"seed": base_sd + 45, "objects": 4, "readout": "focus", "relation_direction": "right", "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": sp["r_focal"], "latent_mode": "focal_memory", "latent_decay": sp["d_focal"], "latent_weight": sp["w_focal"], "actuator_curvature": 0.15, "name_seed": base_sd + 145, "wiring_seed": base_sd + 245}),
    ]

    # ------------------------------------------------------------------------
    # 1. Recursive Source Trajectory (`L^K_c`, shared by E1 & E2-K)
    # ------------------------------------------------------------------------
    lib_k_src = KnowledgeLibrary()
    lib_k_src._prioritize_generalized_families = False
    lib_k_src._enable_open_grammar_exchange_screening = True
    k_source_eps: list[dict[str, Any]] = []
    k_s4_raw: dict[str, Any] = {}
    cold_snapshot: KnowledgeLibrary | None = None

    for idx_s, (s_tag, s_sd, o_b, m_b, c_b, s_kw) in enumerate(src_defs):
        ep_k, raw_k = _run_episode_v15(
            arm_name="K_source_formation",
            episode_tag=f"{cid}:{s_tag}",
            lab_kwargs=s_kw,
            library=lib_k_src,
            cfg=_make_equal_policy_cfg(s_sd, obs_b=o_b, comp_b=c_b, branch_reconciliation=True),
            meas_b=m_b,
        )
        k_source_eps.append(ep_k)
        if idx_s == 0:
            cold_snapshot = deepcopy(lib_k_src)
        if idx_s == 4:
            k_s4_raw = raw_k

    assert cold_snapshot is not None

    # ------------------------------------------------------------------------
    # 2. Ordinary Autonomous Source Trajectory (`L^Ord_c`, for E2 ONLY — zero splicing!)
    # ------------------------------------------------------------------------
    lib_ord_src = KnowledgeLibrary()
    lib_ord_src._prioritize_generalized_families = False
    lib_ord_src._enable_open_grammar_exchange_screening = True
    lib_ord_src._ordinary_persistent_cache = OrdinaryPersistentCommitmentCache()
    lib_ord_src._ordinary_commitment_cache = lib_ord_src._ordinary_persistent_cache
    ord_source_eps: list[dict[str, Any]] = []

    for s_tag, s_sd, o_b, m_b, c_b, s_kw in src_defs:
        ep_o, _ = _run_episode_v15(
            arm_name="Ordinary_source_formation",
            episode_tag=f"{cid}:{s_tag}",
            lab_kwargs=s_kw,
            library=lib_ord_src,
            cfg=_make_equal_policy_cfg(s_sd, obs_b=o_b, comp_b=c_b, branch_reconciliation=False),
            meas_b=m_b,
        )
        ord_source_eps.append(ep_o)

    # Inspect repaired S4 object in `lib_k_src`
    s4_ep = k_source_eps[-1]
    s4_cid = s4_ep["concept_id"]
    s4_has_acc = "accumulate" in json.dumps(k_s4_raw.get("program") or {})
    s4_qualified = bool(s4_ep["strict_pass"] and s4_cid and s4_has_acc)
    s4_prior = next((p for p in lib_k_src.meta_priors if p.get("source_concept_id") == s4_cid), None)
    s4_sem_ver = compute_semantic_version(s4_prior) if (s4_qualified and isinstance(s4_prior, dict)) else None

    s4_reconcile_acts = [
        a for a in k_s4_raw.get("actions", ()) if a.get("action") == "autonomous_branch_reconciliation"
    ]
    k_maint_compute = sum(
        int(rec.get("compute_cost", 0))
        for rec in (getattr(lib_k_src, "_reconciliation_episode_ledger", ()) or ())
    )
    k_discrim_probe_obs = sum(
        len(ev.get("observations_serialized") or ev.get("rows_serialized") or ())
        for ev in (getattr(lib_k_src, "_raw_reconciliation_evidence_archive", ()) or ())
        if ev.get("kind") == "discriminating_probe_batch"
    )

    # ------------------------------------------------------------------------
    # 3. Build Target Libraries for E1 (same raw history `L^K_c`) & E2 (`L^Ord_c`)
    # ------------------------------------------------------------------------
    # E1 Arm 1 & E2 Arm K: `K_eager_persistent`
    lib_e1_k = deepcopy(lib_k_src)
    lib_e1_k._freeze_branch_writes = True
    lib_e1_k._allow_on_demand_family_reconstruction = False
    lib_e1_k._allow_on_demand_routing_reconstruction = False
    lib_e1_k._prioritize_generalized_families = False

    # E1 Arm 2: `R_same_history_lazy` (same raw `meta_priors` + `_raw_reconciliation_evidence_archive`!)
    lib_e1_r = deepcopy(lib_k_src)
    lib_e1_r._active_reconciled_branches = []
    lib_e1_r._generalized_branch_commitments = []
    lib_e1_r._retracted_premise_or_claim_ids = set()
    lib_e1_r._premise_commitment_index = None
    lib_e1_r._indexed_prior_fingerprints = None
    lib_e1_r._indexed_group_commitments = None
    lib_e1_r._ordinary_commitment_cache = OrdinaryPersistentCommitmentCache()
    lib_e1_r._ordinary_persistent_cache = lib_e1_r._ordinary_commitment_cache
    lib_e1_r._freeze_branch_writes = True
    lib_e1_r._allow_on_demand_family_reconstruction = True
    lib_e1_r._allow_on_demand_routing_reconstruction = True
    lib_e1_r._prioritize_generalized_families = False

    # E1 Arm 3: `K_true_repaired_version_removed_raw_rebuild_allowed`
    pre_rm_inv = export_library_object_inventory(lib_k_src)
    if s4_qualified and s4_cid:
        lib_e1_rm = lib_k_src.without_object_version(s4_cid, s4_sem_ver, mode="repaired_version")
    else:
        lib_e1_rm = deepcopy(lib_k_src)
    lib_e1_rm._freeze_branch_writes = True
    lib_e1_rm._prioritize_generalized_families = False
    post_rm_inv = export_library_object_inventory(lib_e1_rm)
    if s4_qualified and s4_cid:
        assert s4_cid not in post_rm_inv["objects"]
        assert all(c.concept_id != s4_cid for c in lib_e1_rm.concepts)

    # E1 Arm 4: `K_cold`
    lib_e1_cold = deepcopy(cold_snapshot)
    lib_e1_cold._freeze_branch_writes = True
    lib_e1_cold._allow_on_demand_family_reconstruction = False
    lib_e1_cold._allow_on_demand_routing_reconstruction = False
    lib_e1_cold._prioritize_generalized_families = False

    # E2 Arm Ordinary: `Ordinary_autonomous_end_to_end` on its OWN `lib_ord_src`!
    lib_e2_ord = deepcopy(lib_ord_src)
    lib_e2_ord._freeze_branch_writes = True
    lib_e2_ord._prioritize_generalized_families = False

    # Build 6 order-balanced unseen N=5 targets `T1..T6`
    target_sequence = []
    for k_step, slot_k in enumerate(cluster_spec["target_order_pattern"], start=1):
        role_nm, t_p = cluster_spec["target_params"][slot_k]
        t_sd, t_kw = _build_target_lab_kwargs(base_sd, k_step, slot_k, role_nm, t_p)
        target_sequence.append((k_step, slot_k, role_nm, f"{cid}:T{k_step}_{slot_k}_{role_nm}_N5", t_sd, t_kw))

    e1_arm_targets: dict[str, list[dict[str, Any]]] = {
        "K_eager_persistent": [],
        "R_same_history_lazy": [],
        "K_true_repaired_version_removed_raw_rebuild_allowed": [],
        "K_cold": [],
    }
    e2_ord_targets: list[dict[str, Any]] = []

    # Run E1 arms on `T1..T6` (carrying warm cache / reconstructed raw object across `T1..T6` without target-write contamination!)
    for arm_e1, base_arm_lib in (
        ("K_eager_persistent", lib_e1_k),
        ("R_same_history_lazy", lib_e1_r),
        ("K_true_repaired_version_removed_raw_rebuild_allowed", lib_e1_rm),
        ("K_cold", lib_e1_cold),
    ):
        carried_cache = getattr(base_arm_lib, "_ordinary_commitment_cache", None)
        carried_k_idx = (
            getattr(base_arm_lib, "_premise_commitment_index", None),
            getattr(base_arm_lib, "_indexed_prior_fingerprints", None),
            getattr(base_arm_lib, "_indexed_group_commitments", None),
        )
        carried_rm_lib = deepcopy(base_arm_lib)
        for k_step, slot_k, role_nm, t_tag, t_sd, t_kw in target_sequence:
            if arm_e1 == "K_true_repaired_version_removed_raw_rebuild_allowed":
                # Keep any raw-history reconstructed source object across T1..T6 while freezing target concepts!
                run_lib = deepcopy(carried_rm_lib)
                n_concepts_before = len(run_lib.concepts)
                n_priors_before = len(run_lib.meta_priors)
            else:
                run_lib = deepcopy(base_arm_lib)
                if carried_cache is not None:
                    run_lib._ordinary_commitment_cache = carried_cache
                    run_lib._ordinary_persistent_cache = carried_cache
                if arm_e1 == "K_eager_persistent" and carried_k_idx[0] is not None:
                    run_lib._premise_commitment_index = deepcopy(carried_k_idx[0])
                    run_lib._indexed_prior_fingerprints = deepcopy(carried_k_idx[1])
                    run_lib._indexed_group_commitments = deepcopy(carried_k_idx[2])

            ep_out, _ = _run_episode_v15(
                arm_name=arm_e1,
                episode_tag=t_tag,
                lab_kwargs=t_kw,
                library=run_lib,
                cfg=_make_equal_policy_cfg(t_sd, obs_b=TARGET_OBS_BUDGET, comp_b=TARGET_COMPUTE_BUDGET, branch_reconciliation=True),
                meas_b=TARGET_MEAS_BUDGET,
            )
            ep_out["prefix_step_k"] = k_step
            ep_out["slot_id"] = slot_k
            ep_out["target_regime_role"] = role_nm
            ep_out["cache_state_at_entry"] = "cold_k1" if k_step == 1 else f"warm_k{k_step}"
            ep_out["adopted_repaired_or_rebuilt_focal_source"] = bool(
                (s4_qualified and s4_cid and s4_cid in ep_out["confirmed_adopted_source_ids"])
                or any(
                    ev.get("adopted_confirmed") and ev.get("generated_from_raw_history")
                    for ev in ep_out["structured_5stage_consumption_events"]
                )
            )
            e1_arm_targets[arm_e1].append(ep_out)

            if arm_e1 == "R_same_history_lazy":
                carried_cache = getattr(run_lib, "_ordinary_commitment_cache", None)
            elif arm_e1 == "K_eager_persistent":
                carried_k_idx = (
                    deepcopy(getattr(run_lib, "_premise_commitment_index", None)),
                    deepcopy(getattr(run_lib, "_indexed_prior_fingerprints", None)),
                    deepcopy(getattr(run_lib, "_indexed_group_commitments", None)),
                )
            elif arm_e1 == "K_true_repaired_version_removed_raw_rebuild_allowed":
                # Persist ONLY raw-history reconstructed source objects & ordinary cache, NOT target concepts!
                recon_new_priors = [
                    p for p in run_lib.meta_priors[n_priors_before:] if p.get("generated_from_raw_history") is True
                ]
                if recon_new_priors:
                    recon_cids = {p["source_concept_id"] for p in recon_new_priors}
                    for c_add in run_lib.concepts[n_concepts_before:]:
                        if c_add.concept_id in recon_cids:
                            carried_rm_lib.concepts.append(deepcopy(c_add))
                    for a_add in run_lib.abstracts:
                        if a_add.source_concept_id in recon_cids and all(x.template_id != a_add.template_id for x in carried_rm_lib.abstracts):
                            carried_rm_lib.abstracts.append(deepcopy(a_add))
                    for dk, dv in run_lib.definitions.items():
                        if dv.source_concept_id in recon_cids and dk not in carried_rm_lib.definitions:
                            carried_rm_lib.definitions[dk] = deepcopy(dv)
                    for bk, bv in run_lib.definition_bindings.items():
                        if bk not in carried_rm_lib.definition_bindings and any(bk.startswith(f"{rc}:") for rc in recon_cids):
                            carried_rm_lib.definition_bindings[bk] = deepcopy(bv)
                    carried_rm_lib.meta_priors.extend(deepcopy(recon_new_priors))
                    carried_rm_lib._reconstructed_from_raw_history_episode_ids = deepcopy(
                        getattr(run_lib, "_reconstructed_from_raw_history_episode_ids", set())
                    )
                if getattr(run_lib, "_ordinary_commitment_cache", None) is not None:
                    carried_rm_lib._ordinary_commitment_cache = run_lib._ordinary_commitment_cache
                    carried_rm_lib._ordinary_persistent_cache = run_lib._ordinary_commitment_cache

    # Run E2 `Ordinary_autonomous_end_to_end` on its OWN `lib_e2_ord` (`branch_reconciliation=False`)
    ord_carried_cache = getattr(lib_e2_ord, "_ordinary_commitment_cache", None)
    for k_step, slot_k, role_nm, t_tag, t_sd, t_kw in target_sequence:
        run_ord = deepcopy(lib_e2_ord)
        if ord_carried_cache is not None:
            run_ord._ordinary_commitment_cache = ord_carried_cache
            run_ord._ordinary_persistent_cache = ord_carried_cache
        # Freeze `meta_priors` & `concepts` snapshot so targets evaluate `L^Ord_c` cleanly across T1..T6
        saved_ord_priors = deepcopy(run_ord.meta_priors)
        saved_ord_concepts = deepcopy(run_ord.concepts)
        ep_ord, _ = _run_episode_v15(
            arm_name="Ordinary_autonomous_end_to_end",
            episode_tag=t_tag,
            lab_kwargs=t_kw,
            library=run_ord,
            cfg=_make_equal_policy_cfg(t_sd, obs_b=TARGET_OBS_BUDGET, comp_b=TARGET_COMPUTE_BUDGET, branch_reconciliation=False),
            meas_b=TARGET_MEAS_BUDGET,
        )
        ep_ord["prefix_step_k"] = k_step
        ep_ord["slot_id"] = slot_k
        ep_ord["target_regime_role"] = role_nm
        ep_ord["cache_state_at_entry"] = "cold_k1" if k_step == 1 else f"warm_k{k_step}"
        e2_ord_targets.append(ep_ord)
        # Preserve warm ordinary cache keyed to `L^Ord_c`
        run_ord.meta_priors = saved_ord_priors
        run_ord.concepts = saved_ord_concepts
        ord_carried_cache = getattr(run_ord, "_ordinary_commitment_cache", None)

    # Verify E1 `K_eager_persistent` vs `R_same_history_lazy` semantic first-dispatch parity across all 6 targets!
    dispatch_diffs = 0
    for k_ep, r_ep in zip(e1_arm_targets["K_eager_persistent"], e1_arm_targets["R_same_history_lazy"]):
        nk = _normalize_dispatch(k_ep["branch_dispatch_telemetry"])
        nr = _normalize_dispatch(r_ep["branch_dispatch_telemetry"])
        if nk != nr:
            dispatch_diffs += 1
    assert dispatch_diffs == 0, f"E1 K vs R dispatch mismatch on {cid}: {dispatch_diffs} diffs!"

    # Prefix full-cost calculations for E1 (`k = 1, 3, 6`, `R - K` including `C^K_maint` & extra probe obs)
    e1_prefixes = {}
    for k_val in (1, 3, 6):
        r_comp = sum(e["measured_costs"]["compute_cost"] for e in e1_arm_targets["R_same_history_lazy"][:k_val])
        k_comp = sum(e["measured_costs"]["compute_cost"] for e in e1_arm_targets["K_eager_persistent"][:k_val])
        rm_comp = sum(e["measured_costs"]["compute_cost"] for e in e1_arm_targets["K_true_repaired_version_removed_raw_rebuild_allowed"][:k_val])
        r_obs = sum(e["measured_costs"]["observation_cost"] for e in e1_arm_targets["R_same_history_lazy"][:k_val])
        k_obs = sum(e["measured_costs"]["observation_cost"] for e in e1_arm_targets["K_eager_persistent"][:k_val])
        e1_prefixes[f"k_{k_val}"] = {
            "R_target_compute": r_comp,
            "K_target_compute": k_comp,
            "K_source_maintenance_compute": k_maint_compute,
            "full_prefix_compute_delta_R_minus_K": r_comp - (k_comp + k_maint_compute),
            "target_only_compute_delta_R_minus_K": r_comp - k_comp,
            "removed_minus_K_target_compute_delta": rm_comp - k_comp,
            "full_prefix_obs_delta_R_minus_K": r_obs - (k_obs + k_discrim_probe_obs),
            "target_only_obs_delta_R_minus_K": r_obs - k_obs,
        }

    # Prefix full-cost calculations for E2 (`k = 0, 1, 3, 6`: entire source + target trajectory, `Ordinary - K`)
    k_src_tot_comp = sum(e["measured_costs"]["compute_cost"] for e in k_source_eps)
    o_src_tot_comp = sum(e["measured_costs"]["compute_cost"] for e in ord_source_eps)
    k_src_tot_obs = sum(e["measured_costs"]["observation_cost"] for e in k_source_eps)
    o_src_tot_obs = sum(e["measured_costs"]["observation_cost"] for e in ord_source_eps)
    k_src_tot_meas = sum(e["measured_costs"]["measurement_cost"] for e in k_source_eps)
    o_src_tot_meas = sum(e["measured_costs"]["measurement_cost"] for e in ord_source_eps)

    e2_prefixes = {
        "k_0_source_only": {
            "K_end_to_end_compute": k_src_tot_comp,
            "Ordinary_end_to_end_compute": o_src_tot_comp,
            "compute_saving_Ordinary_minus_K": o_src_tot_comp - k_src_tot_comp,
            "K_end_to_end_observations": k_src_tot_obs,
            "Ordinary_end_to_end_observations": o_src_tot_obs,
            "observation_saving_Ordinary_minus_K": o_src_tot_obs - k_src_tot_obs,
            "measurement_saving_Ordinary_minus_K": o_src_tot_meas - k_src_tot_meas,
        }
    }
    for k_val in (1, 3, 6):
        kt_c = k_src_tot_comp + sum(e["measured_costs"]["compute_cost"] for e in e1_arm_targets["K_eager_persistent"][:k_val])
        ot_c = o_src_tot_comp + sum(e["measured_costs"]["compute_cost"] for e in e2_ord_targets[:k_val])
        kt_o = k_src_tot_obs + sum(e["measured_costs"]["observation_cost"] for e in e1_arm_targets["K_eager_persistent"][:k_val])
        ot_o = o_src_tot_obs + sum(e["measured_costs"]["observation_cost"] for e in e2_ord_targets[:k_val])
        kt_m = k_src_tot_meas + sum(e["measured_costs"]["measurement_cost"] for e in e1_arm_targets["K_eager_persistent"][:k_val])
        ot_m = o_src_tot_meas + sum(e["measured_costs"]["measurement_cost"] for e in e2_ord_targets[:k_val])
        e2_prefixes[f"k_{k_val}"] = {
            "K_end_to_end_compute": kt_c,
            "Ordinary_end_to_end_compute": ot_c,
            "compute_saving_Ordinary_minus_K": ot_c - kt_c,
            "K_end_to_end_observations": kt_o,
            "Ordinary_end_to_end_observations": ot_o,
            "observation_saving_Ordinary_minus_K": ot_o - kt_o,
            "measurement_saving_Ordinary_minus_K": ot_m - kt_m,
            "K_target_strict_passes": sum(int(e["strict_pass"]) for e in e1_arm_targets["K_eager_persistent"][:k_val]),
            "Ordinary_target_strict_passes": sum(int(e["strict_pass"]) for e in e2_ord_targets[:k_val]),
        }

    # Status-aware machine-generated factual cluster ledger (§11)
    if s4_qualified:
        s4_fact = (
            f"Confirmed repaired hypothesis `{s4_cid}` (`{s4_sem_ver}`) with `accumulate` operator on `{s4_ep['episode_tag']}` "
            f"(confirmation_loss={s4_ep['confirmation_loss']}, external48_mse={s4_ep['external_fresh48_mse']}, "
            f"reconciliation_events={len(s4_reconcile_acts)})."
        )
    else:
        s4_fact = (
            f"missing_evidence: `{s4_ep['episode_tag']}` ended `{s4_ep['status']}` without forming a confirmed repaired `focal_memory` object "
            f"(confirmation_loss={s4_ep['confirmation_loss']}, external48_mse={s4_ep['external_fresh48_mse']})."
        )

    e1_cluster_rec = {
        "cluster_id": cid,
        "base_seed": base_sd,
        "target_order_pattern": list(cluster_spec["target_order_pattern"]),
        "shared_source_formation": {
            "episodes": k_source_eps,
            "source_reconciliation_maintenance_compute": k_maint_compute,
            "source_discriminating_probe_extra_observations": k_discrim_probe_obs,
            "repaired_s4_object": {
                "qualified": s4_qualified,
                "concept_id": s4_cid if s4_qualified else None,
                "semantic_version": s4_sem_ver,
                "factual_status": s4_fact,
            },
        },
        "version_removal_inventory_audit": {
            "removed_concept_id": s4_cid if s4_qualified else None,
            "removed_semantic_version": s4_sem_ver,
            "pre_removal_object_ids": sorted(pre_rm_inv["objects"].keys()),
            "post_removal_object_ids": sorted(post_rm_inv["objects"].keys()),
            "raw_evidence_batches_preserved": post_rm_inv["raw_reconciliation_evidence_batches_count"],
        },
        "k_vs_r_semantic_dispatch_differences_across_6_targets": dispatch_diffs,
        "arms_target_results": e1_arm_targets,
        "prefix_full_cost_comparison": e1_prefixes,
    }

    e2_cluster_rec = {
        "cluster_id": cid,
        "base_seed": base_sd,
        "target_order_pattern": list(cluster_spec["target_order_pattern"]),
        "K_autonomous_end_to_end": {
            "source_episodes": k_source_eps,
            "source_library_object_inventory": pre_rm_inv,
            "target_episodes": e1_arm_targets["K_eager_persistent"],
        },
        "Ordinary_autonomous_end_to_end": {
            "source_episodes": ord_source_eps,
            "source_library_object_inventory": export_library_object_inventory(lib_ord_src),
            "target_episodes": e2_ord_targets,
        },
        "end_to_end_prefix_comparison_Ordinary_minus_K": e2_prefixes,
        "factual_cluster_summary": s4_fact,
    }
    return e1_cluster_rec, e2_cluster_rec


def _summarize_arm_episodes(eps: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = {}
    for role in sorted({e["target_regime_role"] for e in eps}):
        r_eps = [e for e in eps if e["target_regime_role"] == role]
        by_role[role] = {
            "episodes": len(r_eps),
            "strict_passes": sum(int(e["strict_pass"]) for e in r_eps),
            "accurate_external": sum(int(e["quality_and_deliverability_sec9_1"]["accurate_model_found_external"]) for e in r_eps),
            "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in r_eps), 2),
            "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in r_eps), 2),
            "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in r_eps), 2),
        }
    cold_eps = [e for e in eps if e.get("prefix_step_k") == 1]
    warm_eps = [e for e in eps if (e.get("prefix_step_k") or 0) > 1]
    decomp_keys = list(eps[0]["cost_decomposition_sec7_1"].keys())
    return {
        "episodes_total": len(eps),
        "strict_passes": sum(int(e["strict_pass"]) for e in eps),
        "strict_pass_rate": round(sum(int(e["strict_pass"]) for e in eps) / len(eps), 4),
        "accurate_model_found_external": sum(int(e["quality_and_deliverability_sec9_1"]["accurate_model_found_external"]) for e in eps),
        "legally_completed": sum(int(e["quality_and_deliverability_sec9_1"]["legally_completed_status"]) for e in eps),
        "false_internal_confirmations": sum(int(e["false_internal_confirmation"]) for e in eps),
        "unresolved_or_unconfirmed": sum(int(e["unresolved_or_unconfirmed"]) for e in eps),
        "unconfirmed_yet_accurate_on_external": sum(int(e["unconfirmed_yet_accurate_on_external"]) for e in eps),
        "wrong_knowledge_transfers": sum(int(e["wrong_knowledge_transfer"]) for e in eps),
        "adopted_repaired_or_rebuilt_focal_count": sum(int(bool(e.get("adopted_repaired_or_rebuilt_focal_source"))) for e in eps),
        "raw_history_reconstructions_triggered": sum(len(e.get("raw_history_reconstruction_events") or ()) for e in eps),
        "unconditional_costs": {
            "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in eps), 2),
            "median_compute": round(median(e["measured_costs"]["compute_cost"] for e in eps), 2),
            "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in eps), 2),
            "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in eps), 2),
            "mean_agent_wall_seconds": round(mean(e["measured_costs"]["agent_run_wall_seconds"] for e in eps), 4),
        },
        "mean_cost_decomposition_sec7_1": {
            dk: round(mean(e["cost_decomposition_sec7_1"][dk] for e in eps), 2)
            for dk in decomp_keys
        },
        "cache_cold_vs_warm": {
            "cold_k1_mean_recon_compute": round(mean(e["on_demand_or_raw_reconstruction_compute"] for e in cold_eps), 2) if cold_eps else 0.0,
            "warm_k2_to_k6_mean_recon_compute": round(mean(e["on_demand_or_raw_reconstruction_compute"] for e in warm_eps), 2) if warm_eps else 0.0,
        },
        "by_target_role": by_role,
        "all_non_passing_episodes": [
            {
                "episode_tag": e["episode_tag"],
                "status": e["status"],
                "failure_stage_diagnosis": e["quality_and_deliverability_sec9_1"]["failure_stage_diagnosis"],
                "confirmation_loss": e["confirmation_loss"],
                "external_fresh48_mse": e["external_fresh48_mse"],
                "measured_costs": e["measured_costs"],
            }
            for e in eps
            if not e["strict_pass"]
        ],
    }


def main() -> None:
    t0 = time.perf_counter()
    e1_clusters: list[dict[str, Any]] = []
    e2_clusters: list[dict[str, Any]] = []

    for idx, c_spec in enumerate(PROSPECTIVE_V15_CLUSTERS, start=1):
        e1_c, e2_c = _evaluate_single_cluster(c_spec)
        e1_clusters.append(e1_c)
        e2_clusters.append(e2_c)
        k_pass = sum(int(e["strict_pass"]) for e in e1_c["arms_target_results"]["K_eager_persistent"])
        r_pass = sum(int(e["strict_pass"]) for e in e1_c["arms_target_results"]["R_same_history_lazy"])
        rm_pass = sum(int(e["strict_pass"]) for e in e1_c["arms_target_results"]["K_true_repaired_version_removed_raw_rebuild_allowed"])
        o_pass = sum(int(e["strict_pass"]) for e in e2_c["Ordinary_autonomous_end_to_end"]["target_episodes"])
        print(
            f"[v15 Prospective {idx}/{len(PROSPECTIVE_V15_CLUSTERS)}] {c_spec['cluster_id']}: "
            f"E1(K={k_pass}/6, R={r_pass}/6, K_rm={rm_pass}/6, diffs={e1_c['k_vs_r_semantic_dispatch_differences_across_6_targets']}), "
            f"E2(Ord={o_pass}/6 vs K={k_pass}/6)",
            flush=True,
        )

    # Aggregate E1
    e1_arm_summaries = {
        arm: _summarize_arm_episodes([e for c in e1_clusters for e in c["arms_target_results"][arm]])
        for arm in ("K_eager_persistent", "R_same_history_lazy", "K_true_repaired_version_removed_raw_rebuild_allowed", "K_cold")
    }
    e1_prefix_summary = {}
    for k_key in ("k_1", "k_3", "k_6"):
        c_deltas = [c["prefix_full_cost_comparison"][k_key]["full_prefix_compute_delta_R_minus_K"] for c in e1_clusters]
        t_deltas = [c["prefix_full_cost_comparison"][k_key]["target_only_compute_delta_R_minus_K"] for c in e1_clusters]
        rm_deltas = [c["prefix_full_cost_comparison"][k_key]["removed_minus_K_target_compute_delta"] for c in e1_clusters]
        o_deltas = [c["prefix_full_cost_comparison"][k_key]["full_prefix_obs_delta_R_minus_K"] for c in e1_clusters]
        e1_prefix_summary[k_key] = {
            "per_cluster_full_compute_delta_R_minus_K": c_deltas,
            "mean_full_compute_delta_R_minus_K": round(mean(c_deltas), 2),
            "median_full_compute_delta_R_minus_K": round(median(c_deltas), 2),
            "clusters_where_K_has_lower_full_compute_than_R": sum(1 for d in c_deltas if d > 0),
            "exploratory_sign_flip_p_full_compute": round(_sign_flip_p(c_deltas), 4),
            "mean_target_only_compute_delta_R_minus_K": round(mean(t_deltas), 2),
            "per_cluster_removed_minus_K_target_compute_delta": rm_deltas,
            "mean_removed_minus_K_target_compute_delta": round(mean(rm_deltas), 2),
            "mean_full_obs_delta_R_minus_K": round(mean(o_deltas), 2),
        }

    receipt_e1 = {
        "schema_version": "hegel_v15_deliverable_3_prospective_e1_same_history_organization_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v15_Deliverable_3_Prospective_E1_Same_History_Organization_And_True_Version_Removal",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "code_provenance": _source_code_provenance(),
        "protocol_and_statistical_scope": {
            "single_touch_frozen_before_running_1051000_1056000": True,
            "retired_exposed_series_excluded": ["1001..1005", "1010..1018", "1021..1028", "1031..1038", "1041..1046"],
            "statistical_unit": "independent_source_cluster_n_equals_6_descriptive_exploratory",
            "identical_state_transition_function_for_K_and_R": True,
            "total_k_vs_r_dispatch_differences_across_36_pairs": sum(
                c["k_vs_r_semantic_dispatch_differences_across_6_targets"] for c in e1_clusters
            ),
            "true_versioned_object_removal_with_legal_billed_raw_rebuild": True,
            "unconsumed_source_time_H_star_skipped": True,
        },
        "arm_summaries_36_targets_per_arm": e1_arm_summaries,
        "prefix_cost_summary_k_1_3_6": e1_prefix_summary,
        "prospective_clusters": e1_clusters,
        "total_wall_seconds": round(time.perf_counter() - t0, 3),
    }

    # Aggregate E2
    k_src_all = [e for c in e2_clusters for e in c["K_autonomous_end_to_end"]["source_episodes"]]
    o_src_all = [e for c in e2_clusters for e in c["Ordinary_autonomous_end_to_end"]["source_episodes"]]
    k_tgt_all = [e for c in e2_clusters for e in c["K_autonomous_end_to_end"]["target_episodes"]]
    o_tgt_all = [e for c in e2_clusters for e in c["Ordinary_autonomous_end_to_end"]["target_episodes"]]

    e2_prefix_summary = {}
    for k_key in ("k_0_source_only", "k_1", "k_3", "k_6"):
        c_sav = [c["end_to_end_prefix_comparison_Ordinary_minus_K"][k_key]["compute_saving_Ordinary_minus_K"] for c in e2_clusters]
        o_sav = [c["end_to_end_prefix_comparison_Ordinary_minus_K"][k_key]["observation_saving_Ordinary_minus_K"] for c in e2_clusters]
        m_sav = [c["end_to_end_prefix_comparison_Ordinary_minus_K"][k_key]["measurement_saving_Ordinary_minus_K"] for c in e2_clusters]
        e2_prefix_summary[k_key] = {
            "per_cluster_compute_saving_Ordinary_minus_K": c_sav,
            "mean_compute_saving_Ordinary_minus_K": round(mean(c_sav), 2),
            "median_compute_saving_Ordinary_minus_K": round(median(c_sav), 2),
            "clusters_where_K_cheaper_in_compute": sum(1 for x in c_sav if x > 0),
            "exploratory_sign_flip_p_compute": round(_sign_flip_p(c_sav), 4),
            "per_cluster_observation_saving_Ordinary_minus_K": o_sav,
            "mean_observation_saving_Ordinary_minus_K": round(mean(o_sav), 2),
            "clusters_where_K_cheaper_in_observations": sum(1 for x in o_sav if x > 0),
            "exploratory_sign_flip_p_observations": round(_sign_flip_p(o_sav), 4),
            "mean_measurement_saving_Ordinary_minus_K": round(mean(m_sav), 2),
        }

    k_pass_diffs = [
        sum(int(e["strict_pass"]) for e in c["K_autonomous_end_to_end"]["target_episodes"])
        - sum(int(e["strict_pass"]) for e in c["Ordinary_autonomous_end_to_end"]["target_episodes"])
        for c in e2_clusters
    ]

    receipt_e2 = {
        "schema_version": "hegel_v15_deliverable_4_prospective_e2_end_to_end_strategy_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v15_Deliverable_4_Prospective_E2_End_To_End_Strategy_Comparison",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "code_provenance": _source_code_provenance(),
        "protocol_and_statistical_scope": {
            "single_touch_frozen_before_running_1051000_1056000": True,
            "zero_splicing_each_strategy_uses_its_own_formed_source_library": True,
            "equal_research_config_and_local_api_triggers": True,
            "statistical_unit": "independent_source_cluster_n_equals_6_descriptive_exploratory",
            "target_strict_pass_diff_per_cluster_K_minus_Ordinary": k_pass_diffs,
            "exploratory_sign_flip_p_target_strict_pass": round(_sign_flip_p([float(x) for x in k_pass_diffs]), 4),
        },
        "source_formation_summary_30_episodes_per_strategy": {
            "K_autonomous_source": {
                "strict_passes": sum(int(e["strict_pass"]) for e in k_src_all),
                "episodes_total": len(k_src_all),
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in k_src_all), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in k_src_all), 2),
                "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in k_src_all), 2),
            },
            "Ordinary_autonomous_source": {
                "strict_passes": sum(int(e["strict_pass"]) for e in o_src_all),
                "episodes_total": len(o_src_all),
                "mean_compute": round(mean(e["measured_costs"]["compute_cost"] for e in o_src_all), 2),
                "mean_observations": round(mean(e["measured_costs"]["observation_cost"] for e in o_src_all), 2),
                "mean_measurements": round(mean(e["measured_costs"]["measurement_cost"] for e in o_src_all), 2),
            },
        },
        "target_summaries_36_targets_per_strategy": {
            "K_autonomous_end_to_end": _summarize_arm_episodes(k_tgt_all),
            "Ordinary_autonomous_end_to_end": _summarize_arm_episodes(o_tgt_all),
        },
        "end_to_end_prefix_summary_Ordinary_minus_K": e2_prefix_summary,
        "prospective_clusters": e2_clusters,
        "total_wall_seconds": round(time.perf_counter() - t0, 3),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_e1 = OUT_DIR / "deliverable_3_prospective_e1_same_history_organization_20261118.json"
    p_e2 = OUT_DIR / "deliverable_4_prospective_e2_end_to_end_strategy_20261118.json"
    p_e1.write_text(json.dumps(receipt_e1, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    p_e2.write_text(json.dumps(receipt_e2, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "completed",
                "deliverable_3_e1": str(p_e1),
                "deliverable_4_e2": str(p_e2),
                "e1_dispatch_differences": receipt_e1["protocol_and_statistical_scope"]["total_k_vs_r_dispatch_differences_across_36_pairs"],
                "e1_prefix_k6_mean_R_minus_K": e1_prefix_summary["k_6"]["mean_full_compute_delta_R_minus_K"],
                "e2_prefix_k6_compute_Ord_minus_K": e2_prefix_summary["k_6"]["mean_compute_saving_Ordinary_minus_K"],
                "e2_prefix_k6_obs_Ord_minus_K": e2_prefix_summary["k_6"]["mean_observation_saving_Ordinary_minus_K"],
                "wall_seconds": receipt_e2["total_wall_seconds"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run v15 Deliverables 1–4 Engineering Errata, Controls, Parity Audits & Exposed Diagnostic Replays.

Strictly enforces GPT-6 v15 contract (`02_下一轮最小施工与实验合同.md` §1–§6):
  - Zero online API model calls (`model_calls = 0`).
  - Zero byte modifications to all historic `v10..v14` JSON receipts in `docs/evidence/`.
  - Strict `allow_nan=False` JSON serialization everywhere.
  - Retires `1031000..1038000` and `1041000..1046000` to exposed development/diagnostic material only.

Outputs in `docs/evidence/v15_true_intervention_and_strategy_20261118/`:
  1. `deliverable_1_v14_errata_and_factual_reexport_20261118.json`
  2. `deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json`
"""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import subprocess
import time
from types import SimpleNamespace
from typing import Any

from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.branch_reconciliation import (
    dispatch_branches_and_families_for_target,
    reconstruct_missing_objects_from_raw_history,
    reconstruct_parametric_families_from_ordinary_archive,
    replay_evidence_batches_onto_branches,
    shrink_claim_to_observable_scope,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import (
    KnowledgeLibrary,
    export_library_object_inventory,
    remove_library_object_version,
)
from hegel_machine.cognition.models import Candidate, Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import (
    compute_evidence_archive_fingerprint,
    compute_semantic_version,
)
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from scripts.run_v14_fair_comparison_and_continuous_object import (
    D2_SOURCE_COMPUTE_BUDGET,
    D2_SOURCE_MEAS_BUDGET,
    D2_SOURCE_OBS_BUDGET,
    D2_TARGET_COMPUTE_BUDGET,
    D2_TARGET_MEAS_BUDGET,
    D2_TARGET_OBS_BUDGET,
    D3_CONFLICT_COMPUTE_BUDGET,
    D3_CONFLICT_MEAS_BUDGET,
    D3_CONFLICT_OBS_BUDGET,
    D3_MARKOV_SRC_COMPUTE_BUDGET,
    D3_MARKOV_SRC_MEAS_BUDGET,
    D3_MARKOV_SRC_OBS_BUDGET,
    D3_TARGET_COMPUTE_BUDGET,
    D3_TARGET_MEAS_BUDGET,
    D3_TARGET_OBS_BUDGET,
    EXACT_TOLERANCE,
    _build_d2_arm_libraries,
    _build_d3_target_libraries,
    _make_d2_config,
    _run_episode_with_telemetry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V14_DIR = REPO_ROOT / "docs" / "evidence" / "v14_continuous_object_and_fair_comparison_20261118"
OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v15_true_intervention_and_strategy_20261118"

EXPOSED_D2_SPECS: list[dict[str, Any]] = [
    {"chain_id": "unexposed_v14_chain_1031000", "base_seed": 1_031_000, "arrival_order_mode": "unconditional_first", "source_thresholds": (0.20, 0.44, 0.24), "source_radii": (0.93, 0.90, 0.94, 0.91), "target_thresholds": (0.33, 0.35), "target_radii": (0.90, 0.94, 0.92)},
    {"chain_id": "unexposed_v14_chain_1032000", "base_seed": 1_032_000, "arrival_order_mode": "conditional_first_reversed", "source_thresholds": (0.18, 0.46, 0.22), "source_radii": (0.91, 0.94, 0.89, 0.93), "target_thresholds": (0.31, 0.37), "target_radii": (0.92, 0.89, 0.91)},
    {"chain_id": "unexposed_v14_chain_1033000", "base_seed": 1_033_000, "arrival_order_mode": "minority_interleaved", "source_thresholds": (0.22, 0.45, 0.26), "source_radii": (0.94, 0.91, 0.93, 0.90), "target_thresholds": (0.34, 0.36), "target_radii": (0.89, 0.92, 0.93)},
    {"chain_id": "unexposed_v14_chain_1034000", "base_seed": 1_034_000, "arrival_order_mode": "unconditional_first", "source_thresholds": (0.17, 0.43, 0.25), "source_radii": (0.92, 0.93, 0.90, 0.94), "target_thresholds": (0.30, 0.34), "target_radii": (0.91, 0.93, 0.89)},
    {"chain_id": "unexposed_v14_chain_1035000", "base_seed": 1_035_000, "arrival_order_mode": "conditional_first_reversed", "source_thresholds": (0.21, 0.47, 0.23), "source_radii": (0.90, 0.92, 0.95, 0.91), "target_thresholds": (0.35, 0.33), "target_radii": (0.93, 0.90, 0.92)},
    {"chain_id": "unexposed_v14_chain_1036000", "base_seed": 1_036_000, "arrival_order_mode": "minority_interleaved", "source_thresholds": (0.19, 0.45, 0.27), "source_radii": (0.93, 0.89, 0.92, 0.94), "target_thresholds": (0.32, 0.38), "target_radii": (0.90, 0.91, 0.90)},
    {"chain_id": "unexposed_v14_chain_1037000", "base_seed": 1_037_000, "arrival_order_mode": "unconditional_first", "source_thresholds": (0.23, 0.48, 0.21), "source_radii": (0.91, 0.95, 0.91, 0.92), "target_thresholds": (0.36, 0.32), "target_radii": (0.94, 0.88, 0.93)},
    {"chain_id": "unexposed_v14_chain_1038000", "base_seed": 1_038_000, "arrival_order_mode": "conditional_first_reversed", "source_thresholds": (0.16, 0.42, 0.26), "source_radii": (0.94, 0.90, 0.93, 0.89), "target_thresholds": (0.29, 0.36), "target_radii": (0.91, 0.93, 0.91)},
]

EXPOSED_D3_SPECS: list[dict[str, Any]] = [
    {"chain_id": "unexposed_v14_focal_chain_1041000", "base_seed": 1_041_000, "source_radii": (0.93, 0.91, 0.92), "conflict_focal_params": (0.54, 0.36), "target_t1_focal_params": (0.68, 0.42, 0.90), "target_t2_markov_radius": 0.94, "target_t3_counterexample_params": (0.28, 0.91)},
    {"chain_id": "unexposed_v14_focal_chain_1042000", "base_seed": 1_042_000, "source_radii": (0.91, 0.94, 0.90), "conflict_focal_params": (0.62, 0.34), "target_t1_focal_params": (0.48, 0.39, 0.93), "target_t2_markov_radius": 0.89, "target_t3_counterexample_params": (0.32, 0.92)},
    {"chain_id": "unexposed_v14_focal_chain_1043000", "base_seed": 1_043_000, "source_radii": (0.94, 0.90, 0.93), "conflict_focal_params": (0.50, 0.38), "target_t1_focal_params": (0.72, 0.33, 0.91), "target_t2_markov_radius": 0.92, "target_t3_counterexample_params": (0.26, 0.90)},
    {"chain_id": "unexposed_v14_focal_chain_1044000", "base_seed": 1_044_000, "source_radii": (0.92, 0.93, 0.91), "conflict_focal_params": (0.58, 0.35), "target_t1_focal_params": (0.45, 0.41, 0.94), "target_t2_markov_radius": 0.90, "target_t3_counterexample_params": (0.34, 0.89)},
    {"chain_id": "unexposed_v14_focal_chain_1045000", "base_seed": 1_045_000, "source_radii": (0.90, 0.92, 0.94), "conflict_focal_params": (0.66, 0.32), "target_t1_focal_params": (0.53, 0.44, 0.89), "target_t2_markov_radius": 0.93, "target_t3_counterexample_params": (0.30, 0.93)},
    {"chain_id": "unexposed_v14_focal_chain_1046000", "base_seed": 1_046_000, "source_radii": (0.93, 0.89, 0.92), "conflict_focal_params": (0.52, 0.37), "target_t1_focal_params": (0.64, 0.36, 0.92), "target_t2_markov_radius": 0.91, "target_t3_counterexample_params": (0.35, 0.90)},
]

DISPATCH_LIST_FIELDS = (
    "support_invalidated",
    "empirically_excluded_on_context",
    "empirically_excluded_bound_instance_only",
    "not_scheduled_by_router",
    "scope_not_established",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_code_provenance() -> dict[str, Any]:
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    tracked_files = [
        "src/hegel_machine/cognition/premise_commitment_index.py",
        "src/hegel_machine/cognition/memory.py",
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/unified_research.py",
        "scripts/run_v14_fair_comparison_and_continuous_object.py",
    ]
    return {
        "git_base_commit": git_head,
        "modified_source_sha256": {rel: _sha256_file(REPO_ROOT / rel) for rel in tracked_files},
    }


def _normalize_dispatch(telemetry: dict[str, Any]) -> dict[str, Any]:
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


# ============================================================================
# DELIVERABLE 1: V14 10-Item Errata + Status-Aware Factual Re-Export (§2.1, §2.2)
# ============================================================================

def _rerun_question_a_events_for_exposed_d3_spec(spec_c: dict[str, Any]) -> dict[str, Any]:
    """Re-extract genuine `autonomous_branch_reconciliation` events on exposed D3 Question A (`§2.2`)."""
    cid = spec_c["chain_id"]
    base_sd = int(spec_c["base_seed"])
    r_m1, r_m2, r_fm = spec_c["source_radii"]
    d_src, w_src = spec_c["conflict_focal_params"]

    markov_lib = KnowledgeLibrary()
    markov_lib._prioritize_generalized_families = False

    cal_ep, _ = _run_episode_with_telemetry(
        arm_name="common_calibration_anchor",
        episode_tag=f"{cid}:S_cal_focus_N4",
        lab_kwargs={
            "seed": base_sd + 5, "objects": 4, "readout": "focus",
            "relation_direction": "symmetric", "relation_condition": "always",
            "condition_threshold": 0.0, "relation_radius": 0.92,
            "actuator_curvature": 0.15, "name_seed": base_sd + 105, "wiring_seed": base_sd + 205,
        },
        library=markov_lib,
        cfg=ResearchConfig(
            seed=base_sd + 5, observation_budget=D3_MARKOV_SRC_OBS_BUDGET, compute_budget=D3_MARKOV_SRC_COMPUTE_BUDGET,
            max_rounds=4, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True,
            world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True,
            active_threshold_probing=True, indexed_commitment_store=True, sensor_role_calibration_priors=True,
            occam_premise_minimality=True, branch_reconciliation=True,
        ),
        meas_b=D3_MARKOV_SRC_MEAS_BUDGET,
    )
    markov_eps = []
    for tag_m, off_m, dir_m, rad_m in (("S_markov_sym_N4", 15, "symmetric", r_m1), ("S_markov_right_N4", 25, "right", r_m2)):
        sd_m = base_sd + off_m
        ep_m, _ = _run_episode_with_telemetry(
            arm_name="ordinary_markov_source",
            episode_tag=f"{cid}:{tag_m}",
            lab_kwargs={
                "seed": sd_m, "objects": 4, "readout": "focus",
                "relation_direction": dir_m, "relation_condition": "always",
                "condition_threshold": 0.0, "relation_radius": rad_m,
                "actuator_curvature": 0.15, "name_seed": sd_m + 100, "wiring_seed": sd_m + 200,
            },
            library=markov_lib,
            cfg=ResearchConfig(
                seed=sd_m, observation_budget=D3_MARKOV_SRC_OBS_BUDGET, compute_budget=D3_MARKOV_SRC_COMPUTE_BUDGET,
                max_rounds=4, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True,
                world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True,
                active_threshold_probing=True, indexed_commitment_store=True, sensor_role_calibration_priors=True,
                occam_premise_minimality=True, branch_reconciliation=True,
            ),
            meas_b=D3_MARKOV_SRC_MEAS_BUDGET,
        )
        markov_eps.append(ep_m)

    prior_markov_versions = [
        {"branch_id": b["claim_id"], "source_concept_id": b["source_concept_id"], "semantic_version": b["semantic_version"]}
        for b in markov_lib._active_reconciled_branches
    ]

    sd_conf = base_sd + 35
    repaired_lib = deepcopy(markov_lib)
    qA_subq_summary, qA_subq_raw = _run_episode_with_telemetry(
        arm_name="QuestionA_subquestion_driven_repair_equal_api",
        episode_tag=f"{cid}:S_conflict_focal_memory_N4",
        lab_kwargs={
            "seed": sd_conf, "objects": 4, "readout": "focus",
            "relation_direction": "right", "relation_condition": "always",
            "condition_threshold": 0.0, "relation_radius": r_fm,
            "latent_mode": "focal_memory", "latent_decay": d_src, "latent_weight": w_src,
            "actuator_curvature": 0.15, "name_seed": sd_conf + 100, "wiring_seed": sd_conf + 200,
        },
        library=repaired_lib,
        cfg=ResearchConfig(
            seed=sd_conf, observation_budget=D3_CONFLICT_OBS_BUDGET, compute_budget=D3_CONFLICT_COMPUTE_BUDGET,
            max_rounds=1, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True,
            world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=False,
            latent_state_discovery=False, active_threshold_probing=False, state_search=True,
            indexed_commitment_store=True, sensor_role_calibration_priors=True, occam_premise_minimality=True,
            branch_reconciliation=True,
        ),
        meas_b=D3_CONFLICT_MEAS_BUDGET,
    )

    init_rec_acts = [
        a for a in qA_subq_raw["actions"]
        if a.get("action") == "autonomous_branch_reconciliation" and not a.get("is_post_subresearch_re_reconcile")
    ]
    subq_acts = [a for a in qA_subq_raw["actions"] if a.get("action") == "resolve_reconciliation_subquestion"]
    post_rec_acts = [
        a for a in qA_subq_raw["actions"]
        if a.get("action") == "autonomous_branch_reconciliation" and a.get("is_post_subresearch_re_reconcile") is True
    ]
    confirm_acqs = [
        a for a in qA_subq_raw["actions"]
        if a.get("action") == "acquire" and "confirm" in str(a.get("purpose") or "")
    ]
    rep_cid = subq_acts[-1].get("repaired_concept_id") if subq_acts else qA_subq_summary.get("concept_id")
    rep_branch = next((b for b in repaired_lib._active_reconciled_branches if b.get("source_concept_id") == rep_cid), None)
    has_acc = "accumulate" in json.dumps(qA_subq_raw.get("program") or {})
    qualified = bool(qA_subq_summary["strict_pass"] and rep_cid and has_acc)

    return {
        "prior_markov_versions": prior_markov_versions,
        "qualified_repaired_object": qualified,
        "repaired_concept_id": rep_cid if qualified else None,
        "repaired_semantic_version": (rep_branch or {}).get("semantic_version") if qualified else None,
        "initial_conflict_reconciliation_event": {
            "executed": bool(init_rec_acts),
            "epistemic_status": init_rec_acts[0].get("epistemic_status") if init_rec_acts else "missing_evidence",
            "reconciliation_decisions": init_rec_acts[0].get("reconciliation_decisions") if init_rec_acts else [],
            "paid_probes_executed": init_rec_acts[0].get("paid_probes_executed") if init_rec_acts else [],
            "pending_questions": init_rec_acts[0].get("pending_questions") if init_rec_acts else [],
        },
        "subquestion_resolution_event": subq_acts[-1] if subq_acts else None,
        "confirmation_batches": confirm_acqs,
        "post_subresearch_re_reconciliation_event": {
            "executed": bool(post_rec_acts),
            "epistemic_status": post_rec_acts[-1].get("epistemic_status") if post_rec_acts else "missing_evidence",
            "reconciliation_decisions": post_rec_acts[-1].get("reconciliation_decisions") if post_rec_acts else [],
            "paid_probes_executed": post_rec_acts[-1].get("paid_probes_executed") if post_rec_acts else [],
            "pending_questions": post_rec_acts[-1].get("pending_questions") if post_rec_acts else [],
        },
        "repaired_library": repaired_lib,
        "cold_snapshot": deepcopy(markov_lib),
    }


def run_deliverable_1_errata_and_reexport() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    d2_path = V14_DIR / "deliverable_2_unconfounded_noH_source_paired_study_20261118.json"
    d3_path = V14_DIR / "deliverable_3_continuous_repaired_object_study_20261118.json"
    d2_raw = d2_path.read_bytes()
    d3_raw = d3_path.read_bytes()
    d2_data = json.loads(d2_raw)
    d3_data = json.loads(d3_raw)

    d2_chains = d2_data["unexposed_evaluation_chains"]
    d3_chains = d3_data["all_attempted_continuous_chains"]

    # Recompute exact episode counts & table figures directly from untouched V14 JSON
    d2_all_eps = (
        [c["calibration_anchor_episode"] for c in d2_chains]
        + [e for c in d2_chains for e in c["ordinary_source_episodes"]]
        + [e for c in d2_chains for arm in c["arms_target_results"].values() for e in arm]
    )
    d3_all_eps = (
        [c["calibration_anchor_episode"] for c in d3_chains]
        + [e for c in d3_chains for e in c["ordinary_markov_source_episodes"]]
        + [c["question_A_parent_repair_equal_api"]["ordinary_failure_retry_equal_api"] for c in d3_chains]
        + [c["question_A_parent_repair_equal_api"]["subquestion_driven_repair_equal_api"] for c in d3_chains]
        + [e for c in d3_chains for arm in c["arms_target_results"].values() for e in arm]
    )
    all_232 = d2_all_eps + d3_all_eps
    d2_anchors = [c["calibration_anchor_episode"] for c in d2_chains]

    d2_meas_means = {
        arm: mean(e["measured_costs"]["measurement_cost"] for c in d2_chains for e in c["arms_target_results"][arm])
        for arm in ("K_cold", "R_lazy_noH", "K_eager_noH")
    }

    # Typed consumption audit on saved V14 D3
    typed_d3_t1_summary: dict[str, dict[str, Any]] = {}
    for arm in ("K_eager_noH", "R_lazy_noH"):
        str_true = []
        typed_gen = []
        typed_adopted = []
        for c in d3_chains:
            ob = c["continuous_repaired_object_ledger"]
            rcid = ob["repaired_concept_id"]
            if not ob["produced_qualified_repaired_object"]:
                continue
            ep_t1 = c["arms_target_results"][arm][0]
            if ep_t1["invoked_repaired_object"]:
                str_true.append(c["chain_id"])
            evs = [
                x
                for x in (ep_t1.get("causal_consumption_telemetry") or {}).get("consumption_events", [])
                if rcid in (x.get("source_concept_ids") or ())
            ]
            if any(x.get("event_stage") == "candidate_generation" for x in evs):
                typed_gen.append(c["chain_id"])
            if any(x.get("adopted_confirmed") for x in evs):
                typed_adopted.append(c["chain_id"])
        typed_d3_t1_summary[arm] = {
            "qualified_chains_n": 4,
            "reported_invoked_by_substring": str_true,
            "typed_candidate_generated_chains": typed_gen,
            "typed_adopted_confirmed_chains": typed_adopted,
            "unproved_by_v14_telemetry_chains": [x for x in str_true if x not in typed_adopted],
        }

    # Re-export the 6 exposed V14 D3 chains with real production events + status-aware narratives
    # while keeping the untouched V14 MSE & cost records intact (§2.2 Acceptance #1–#4)!
    reexported_d3_chains: list[dict[str, Any]] = []
    exposed_qa_cache: dict[str, dict[str, Any]] = {}

    for spec_c, orig_c in zip(EXPOSED_D3_SPECS, d3_chains):
        cid = orig_c["chain_id"]
        live_qa = _reextract_or_run_qa(spec_c) if False else _re_run_qa_cached(spec_c, exposed_qa_cache)
        qa_ord = orig_c["question_A_parent_repair_equal_api"]["ordinary_failure_retry_equal_api"]
        qa_subq = orig_c["question_A_parent_repair_equal_api"]["subquestion_driven_repair_equal_api"]
        qualified = bool(orig_c["continuous_repaired_object_ledger"]["produced_qualified_repaired_object"])
        rcid = orig_c["continuous_repaired_object_ledger"]["repaired_concept_id"] if qualified else None
        rver = orig_c["continuous_repaired_object_ledger"]["repaired_semantic_version"] if qualified else None

        # Re-summarize target scope fields & structured consumption from untouched saved V14 target telemetry
        reexported_targets: dict[str, list[dict[str, Any]]] = {}
        for arm, eps in orig_c["arms_target_results"].items():
            arm_list = []
            for ep in eps:
                bdt = ep.get("branch_dispatch_telemetry") or {}
                evs_all = (ep.get("causal_consumption_telemetry") or {}).get("consumption_events") or []
                typed_from_rep = [
                    ev for ev in evs_all
                    if rcid and rcid in (ev.get("source_concept_ids") or ())
                ]
                typed_gen = any(ev.get("event_stage") == "candidate_generation" for ev in typed_from_rep)
                typed_ad = any(bool(ev.get("adopted_confirmed")) for ev in typed_from_rep)
                arm_list.append(
                    {
                        "episode_tag": ep["episode_tag"],
                        "episode_uid": ep["episode_uid"],
                        "target_regime_role": ep["target_regime_role"],
                        "status": ep["status"],
                        "strict_pass": ep["strict_pass"],
                        "confirmation_loss": ep["confirmation_loss"],
                        "external_fresh48_mse": ep["external_fresh48_mse"],
                        "measured_costs": deepcopy(ep["measured_costs"]),
                        "dispatcher_scope_not_established": list(bdt.get("scope_not_established") or ()),
                        "dispatcher_empirically_excluded_on_context": list(bdt.get("empirically_excluded_on_context") or ()),
                        "dispatcher_empirically_excluded_bound_instance_only": list(
                            bdt.get("empirically_excluded_bound_instance_only") or ()
                        ),
                        "dispatcher_not_scheduled_by_router": list(bdt.get("not_scheduled_by_router") or ()),
                        "structured_repaired_source_attribution": {
                            "repaired_source_id": rcid,
                            "repaired_semantic_version": rver,
                            "legacy_substring_flag": bool(ep.get("invoked_repaired_object")),
                            "typed_candidate_generated": typed_gen,
                            "typed_confirmed_adopted": typed_ad,
                            "attribution_verdict": (
                                "confirmed_adopted"
                                if typed_ad
                                else (
                                    "unproved_by_v14_telemetry"
                                    if ep.get("invoked_repaired_object")
                                    else "not_used"
                                )
                            ),
                        },
                    }
                )
            reexported_targets[arm] = arm_list

        init_ev = live_qa["initial_conflict_reconciliation_event"]
        subq_ev = live_qa["subquestion_resolution_event"]
        post_ev = live_qa["post_subresearch_re_reconciliation_event"]
        m_vers = [f"{m['branch_id']}@{m['semantic_version']}" for m in live_qa["prior_markov_versions"]]
        subq_id = subq_ev.get("question_id") if isinstance(subq_ev, dict) else (
            init_ev["pending_questions"][0]["question_id"] if init_ev["pending_questions"] else "missing_evidence"
        )
        t1_k = reexported_targets["K_eager_noH"][0]
        t1_r = reexported_targets["R_lazy_noH"][0]
        t2_k = reexported_targets["K_eager_noH"][1]
        t3_k = reexported_targets["K_eager_noH"][2]

        if qualified:
            s4 = (
                f"Subquestion `{subq_id}` (parent_version=`{(init_ev['pending_questions'][0].get('parent_semantic_version') if init_ev['pending_questions'] else 'recorded')}`) "
                f"replaced `{subq_ev.get('parent_candidate_before') if subq_ev else 'missing_evidence'}` with "
                f"`{subq_ev.get('parent_candidate_after') if subq_ev else 'missing_evidence'}` (`{rcid}` @ `{rver}`)."
            )
            s5 = (
                f"Repaired object `{rcid}` (`{rver}`) passed independent fresh confirmation "
                f"(confirmation_loss={qa_subq['confirmation_loss']}, external48_mse={qa_subq['external_fresh48_mse']}, "
                f"confirmation_batches={json.dumps(live_qa['confirmation_batches'], ensure_ascii=False)}) "
                f"and executed local post-subresearch re-reconciliation with epistemic_status=`{post_ev['epistemic_status']}` "
                f"and decisions={json.dumps(post_ev['reconciliation_decisions'], ensure_ascii=False)}."
            )
            s6 = (
                f"On unseen N=5 target `{t1_k['episode_tag']}`, `K_eager_noH` structured adoption verdict for `{rcid}` was "
                f"`{t1_k['structured_repaired_source_attribution']['attribution_verdict']}` "
                f"(strict_pass={t1_k['strict_pass']}, compute={t1_k['measured_costs']['compute_cost']}, obs={t1_k['measured_costs']['observation_cost']}); "
                f"`T2` strict_pass={t2_k['strict_pass']}, `T3` strict_pass={t3_k['strict_pass']}. "
                f"(Note: V14's removal arm did not delete list-shaped `concepts`; true versioned removal is verified separately in v15 Deliverable 2 & E1.)"
            )
        else:
            s4 = (
                f"unresolved: Subquestion `{subq_id}` on `{cid}:S_conflict_focal_memory_N4` ended with status="
                f"`{subq_ev.get('status') if isinstance(subq_ev, dict) else qa_subq['status']}` and produced NO qualified repaired object "
                f"(`repaired_concept_id=None`, `produced_qualified_repaired_object=False`)."
            )
            s5 = (
                f"missing_evidence: Question A remained `{qa_subq['status']}` without passing fresh confirmation "
                f"(confirmation_loss={qa_subq['confirmation_loss']}, external48_mse={qa_subq['external_fresh48_mse']}); "
                f"post-subresearch re-reconciliation did NOT confirm a repaired branch (`epistemic_status={post_ev['epistemic_status']}`)."
            )
            s6 = (
                f"Because no repaired object was formed on Question A, unseen target `{t1_k['episode_tag']}` had zero repaired-object "
                f"adoption (`attribution_verdict=not_used`, strict_pass={t1_k['strict_pass']}, compute={t1_k['measured_costs']['compute_cost']})."
            )

        seven_factual = {
            "1_prior_ordinary_experiences_and_versions": (
                f"Prior Markov source episodes formed branches {m_vers} before conflict episode `{qa_subq['episode_tag']}`."
            ),
            "2_refuting_real_evidence_and_failed_versions": (
                f"On paid episode `{qa_subq['episode_uid']}` (`{qa_subq['episode_tag']}`), initial reconciliation recorded "
                f"epistemic_status=`{init_ev['epistemic_status']}` and paid_probes={json.dumps(init_ev['paid_probes_executed'], ensure_ascii=False)} "
                f"against {m_vers}."
            ),
            "3_reconciliation_choice_rationale": (
                f"Initial reconciliation decisions={json.dumps(init_ev['reconciliation_decisions'], ensure_ascii=False)} "
                f"spawned subquestion `{subq_id}`."
                if init_ev["reconciliation_decisions"] or init_ev["pending_questions"]
                else f"missing_evidence: `{qa_subq['episode_tag']}` halted at `{qa_subq['status']}` with initial_reconcile_status=`{init_ev['epistemic_status']}`."
            ),
            "4_subanswer_parent_mutation": s4,
            "5_fresh_confirmation_and_historical_branch_checks": s5,
            "6_unseen_target_consumption_and_action_change": s6,
            "7_equal_capability_ordinary_method_comparison": (
                f"Question A equal-API ordinary retry achieved strict_pass={qa_ord['strict_pass']} "
                f"(compute={qa_ord['measured_costs']['compute_cost']}, obs={qa_ord['measured_costs']['observation_cost']}) "
                f"vs subquestion policy strict_pass={qa_subq['strict_pass']} "
                f"(compute={qa_subq['measured_costs']['compute_cost']}, obs={qa_subq['measured_costs']['observation_cost']}); "
                f"on Question B (shared subquestion-derived library), `R_lazy_noH` T1 strict_pass={t1_r['strict_pass']} "
                f"(compute={t1_r['measured_costs']['compute_cost']}, attribution=`{t1_r['structured_repaired_source_attribution']['attribution_verdict']}`)."
            ),
        }
        # Enforce §2.2 Acceptance #1: Failed chains MUST NEVER contain `"None passed"` or `"passed an independent"`!
        if not qualified:
            joined_s = json.dumps(seven_factual, ensure_ascii=False)
            assert "None` passed" not in joined_s and "passed an independent" not in joined_s

        reexported_d3_chains.append(
            {
                "chain_id": cid,
                "exposed_development_reexport": True,
                "untouched_v14_mse_and_costs_preserved": True,
                "produced_qualified_repaired_object": qualified,
                "repaired_concept_id": rcid,
                "repaired_semantic_version": rver,
                "production_reconciliation_events": {
                    "initial_conflict_reconciliation": init_ev,
                    "subquestion_resolution": subq_ev,
                    "confirmation_batches": live_qa["confirmation_batches"],
                    "post_subresearch_re_reconciliation": post_ev,
                },
                "reexported_target_summaries": reexported_targets,
                "status_aware_seven_sentence_factual_ledger": seven_factual,
            }
        )

    errata_10_items = [
        {
            "item_id": "V14_ERRATA_01_EPISODE_COUNTS",
            "v14_markdown_claim": "214 prospective episodes (189 strict passes, 25 unresolved, 0 false confirmations)",
            "corrected_fact": (
                f"D2 (112) + D3 (120) = {len(all_232)} arm-episode judgments ({sum(e['strict_pass'] for e in all_232)} strict passes, "
                f"{sum(e['unresolved_or_unconfirmed'] for e in all_232)} unresolved/unconfirmed, "
                f"{sum(e['false_internal_confirmation'] for e in all_232)} false internal confirmations); "
                f"deduplicating by episode_uid yields {len({e['episode_uid'] for e in all_232})} unique episodes; "
                "214 omitted D3's 6 anchors + 12 Markov sources; paired statistical unit remains the source cluster (n=8 in D2, n=6 in D3)."
            ),
        },
        {
            "item_id": "V14_ERRATA_02_D2_ANCHOR_COSTS",
            "v14_markdown_claim": "D2 anchor mean compute 82,193.6 and wall-clock 4.31s",
            "corrected_fact": (
                f"D2 anchor mean compute = {mean(a['measured_costs']['compute_cost'] for a in d2_anchors):.3f} "
                f"and agent wall-clock = {mean(a['measured_costs']['agent_run_wall_seconds'] for a in d2_anchors):.5f}s."
            ),
        },
        {
            "item_id": "V14_ERRATA_03_D2_TARGET_MEASUREMENTS",
            "v14_markdown_claim": "Hand-transcribed D2 target mean measurement figures",
            "corrected_fact": (
                f"Exact D2 target mean measurements: K_cold = {d2_meas_means['K_cold']:.1f}, "
                f"R_lazy_noH = {d2_meas_means['R_lazy_noH']:.1f}, K_eager_noH = {d2_meas_means['K_eager_noH']:.1f}."
            ),
        },
        {
            "item_id": "V14_ERRATA_04_STRUCTURED_CONSUMPTION_VS_SUBSTRING",
            "v14_markdown_claim": "4/4 qualified D3 chains consumed the repaired object on T1",
            "corrected_fact": (
                f"V14 used substring matching (`invoked_repaired_object`). Typed `consumption_events` prove confirmed adoption "
                f"on 3/4 chains for K_eager_noH ({typed_d3_t1_summary['K_eager_noH']['typed_adopted_confirmed_chains']}) and "
                f"1/4 chains for R_lazy_noH ({typed_d3_t1_summary['R_lazy_noH']['typed_adopted_confirmed_chains']}); "
                "remaining cases are marked `unproved_by_v14_telemetry`."
            ),
        },
        {
            "item_id": "V14_ERRATA_05_FAILED_CHAIN_NARRATIVES",
            "v14_markdown_claim": "Template sentence 5 emitted `Repaired object None passed...` on failed chains 1043000 and 1045000",
            "corrected_fact": (
                "Chains 1043000 (external MSE 0.03976) and 1045000 (external MSE 0.03268) did NOT form a qualified repaired object; "
                "re-exported ledgers report `unresolved` / `missing_evidence` with zero fake confirmation claims."
            ),
        },
        {
            "item_id": "V14_ERRATA_06_TIMING_VS_FULL_POLICY_ATTRIBUTION",
            "v14_markdown_claim": "Attributed V14 D2 R vs K solely to eager vs lazy maintenance timing",
            "corrected_fact": (
                "In V14 D2, R and K differed on 24/24 first-dispatch states due to batch-vs-chronological replay and `scope_unidentified` "
                "handling, so V14 D2 compared two complete organization policies rather than timing-only; v15 E1 unifies both onto one state-transition function."
            ),
        },
        {
            "item_id": "V14_ERRATA_07_OBJECT_REMOVAL_VALIDITY",
            "v14_markdown_claim": "Claimed single-object removal in V14 D3 isolated marginal value of concept_0004",
            "corrected_fact": (
                "V14 `_build_d3_target_libraries` checked `isinstance(concepts, dict)` on list-shaped `KnowledgeLibrary.concepts`, leaving "
                "`concept_0004` in `concepts`, `abstracts`, `definitions`, and `definition_bindings`, while disabling reconstruction; "
                "no marginal object value claim is drawn from V14 D3 prior to v15's true versioned removal + billed rebuild."
            ),
        },
        {
            "item_id": "V14_ERRATA_08_WRONG_TRANSFER_AND_SCOPE_KEYS",
            "v14_markdown_claim": "Reported `wrong_transfer = 0` and `scope_unknown_events = 0`",
            "corrected_fact": (
                "`wrong_knowledge_transfer == 0` covers only false confirmations after adopting library knowledge (not extra cost or extra unresolved episodes), "
                "and V14's exporter read `scope_unknown_branch_ids`/`excluded_out_of_scope_branch_ids` instead of dispatcher's actual "
                "`scope_not_established`/`empirically_excluded_on_context`."
            ),
        },
        {
            "item_id": "V14_ERRATA_09_H_STAR_SCOPE",
            "v14_markdown_claim": "Described H* as disabled by default",
            "corrected_fact": (
                "In V14, `_prioritize_generalized_families = False` disabled target-time H* Pass-0 scheduling while source-time H* "
                "construction in `autonomous_reconcile_library_episode` still ran; v15 §7.1 skips unconsumed source-time H* construction when disabled."
            ),
        },
        {
            "item_id": "V14_ERRATA_10_NO_SPLICING_E1_AND_E2",
            "v14_markdown_claim": "Discussed Question A ordinary retry alongside Question B R_lazy_noH",
            "corrected_fact": (
                "V14 D3 Question B's `R_lazy_noH` operated on the subquestion-formed library rather than Question A ordinary retry's own library; "
                "A-ordinary and B-R cannot be spliced into a single end-to-end run, so v15 separates E1 (same history) and E2 (true end-to-end)."
            ),
        },
    ]

    receipt_d1 = {
        "schema_version": "hegel_v15_deliverable_1_v14_errata_and_factual_reexport_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v15_Deliverable_1_V14_Errata_And_Status_Aware_Factual_Reexport",
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "code_and_input_provenance": {
            **_source_code_provenance(),
            "untouched_v14_receipt_sha256": {
                "D1": _sha256_file(V14_DIR / "deliverable_1_errata_and_replays_a_to_f_20261118.json"),
                "D2": hashlib.sha256(d2_raw).hexdigest(),
                "D3": hashlib.sha256(d3_raw).hexdigest(),
            },
        },
        "errata_10_items": errata_10_items,
        "recomputed_v14_ground_truth": {
            "total_arm_episode_records": len(all_232),
            "d2_arm_episode_records": len(d2_all_eps),
            "d3_arm_episode_records": len(d3_all_eps),
            "unique_episode_uids": len({e["episode_uid"] for e in all_232}),
            "strict_passes_all": sum(e["strict_pass"] for e in all_232),
            "unresolved_or_unconfirmed_all": sum(e["unresolved_or_unconfirmed"] for e in all_232),
            "false_internal_confirmations_all": sum(e["false_internal_confirmation"] for e in all_232),
            "d2_anchor_mean_compute": mean(a["measured_costs"]["compute_cost"] for a in d2_anchors),
            "d2_anchor_mean_agent_wall_seconds": mean(a["measured_costs"]["agent_run_wall_seconds"] for a in d2_anchors),
            "d2_target_mean_measurements": d2_meas_means,
            "d3_typed_t1_consumption_summary": typed_d3_t1_summary,
        },
        "reexported_exposed_v14_d3_chains": reexported_d3_chains,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_p = OUT_DIR / "deliverable_1_v14_errata_and_factual_reexport_20261118.json"
    out_p.write_text(json.dumps(receipt_d1, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return receipt_d1, exposed_qa_cache


def _re_run_qa_cached(spec_c: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cid = spec_c["chain_id"]
    if cid not in cache:
        cache[cid] = _rerun_question_a_events_for_exposed_d3_spec(spec_c)
    return cache[cid]


# ============================================================================
# DELIVERABLE 2: True Removal Controls (§3.3), K/R Parity (§4.3), & 6 Exposed Replays (§5.2)
# ============================================================================

def _verify_gpt6_ast_probes_on_updated_code() -> dict[str, Any]:
    """Execute the exact AST-level checks from GPT-6's `probe_d3_object_removal.py` & `probe_scope_replay.py`."""
    # 1. `probe_d3_object_removal` check on updated `scripts/run_v14_fair_comparison_and_continuous_object.py`
    src_v14 = REPO_ROOT / "scripts" / "run_v14_fair_comparison_and_continuous_object.py"
    tree_v14 = ast.parse(src_v14.read_text(encoding="utf-8"))
    node_rm = next(n for n in tree_v14.body if isinstance(n, ast.FunctionDef) and n.name == "_build_d3_target_libraries")
    pre = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns_rm: dict[str, Any] = {"deepcopy": deepcopy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pre, node_rm], type_ignores=[])), str(src_v14), "exec"), ns_rm)
    ids = ["concept_0001", "concept_0004"]
    stub_lib = SimpleNamespace(
        meta_priors=[{"source_concept_id": c} for c in ids],
        concepts=[SimpleNamespace(concept_id=c, status="available") for c in ids],
        abstracts=[SimpleNamespace(template_id=f"{c}:abstract:0", source_concept_id=c, status="available") for c in ids],
        definitions={f"{c}:definition": SimpleNamespace(source_concept_id=c) for c in ids},
        definition_bindings={f"{c}:definition": {"template_id": f"{c}:abstract:0"} for c in ids},
        _active_reconciled_branches=[{"claim_id": f"branch:{c}", "source_concept_id": c, "source_concept_ids": [c]} for c in ids],
        _generalized_branch_commitments=[],
    )
    rm_out = ns_rm["_build_d3_target_libraries"](deepcopy(stub_lib), deepcopy(stub_lib), "concept_0004")[
        "K_minimal_repaired_object_removed_other_sources_kept"
    ]
    probe_removal_res = {
        "removed_id": "concept_0004",
        "meta_prior_remaining_ids": [x["source_concept_id"] for x in rm_out.meta_priors],
        "branch_remaining_ids": [x["claim_id"] for x in rm_out._active_reconciled_branches],
        "concept_remaining_ids": [x.concept_id for x in rm_out.concepts],
        "abstract_remaining_source_ids": [x.source_concept_id for x in rm_out.abstracts],
        "definition_remaining_source_ids": [x.source_concept_id for x in rm_out.definitions.values()],
        "definition_binding_keys": list(rm_out.definition_bindings),
        "allow_on_demand_raw_history_reconstruction": bool(getattr(rm_out, "_allow_on_demand_raw_history_reconstruction", False)),
    }
    assert probe_removal_res["concept_remaining_ids"] == ["concept_0001"]
    assert probe_removal_res["abstract_remaining_source_ids"] == ["concept_0001"]
    assert probe_removal_res["definition_remaining_source_ids"] == ["concept_0001"]
    assert probe_removal_res["definition_binding_keys"] == ["concept_0001:definition"]

    # 2. `probe_scope_replay` check on updated `branch_reconciliation.py` + `premise_commitment_index.py`
    reg = {"obs_signed_direction_flux": 0.5, "obs_unconditional_block_rate": 0.2, "_meta": {"protocol_id": "paid_episode_transitions", "pilot_calibration_available": True}}
    b_path = REPO_ROOT / "src" / "hegel_machine" / "cognition" / "branch_reconciliation.py"
    i_path = REPO_ROOT / "src" / "hegel_machine" / "cognition" / "premise_commitment_index.py"
    ns_sr: dict[str, Any] = {
        "Program": Program,
        "Observation": Observation,
        "deepcopy": deepcopy,
        "isfinite": math.isfinite,
        "hashlib": hashlib,
        "json": json,
        "Any": Any,
    }

    def _load_ast(path: Path, fn_names: set[str], const_names: tuple[str, ...] = ()) -> None:
        tr = ast.parse(path.read_text(encoding="utf-8"))
        nodes = []
        for n in tr.body:
            if isinstance(n, ast.FunctionDef) and n.name in fn_names:
                nodes.append(n)
            targets = []
            if isinstance(n, ast.Assign):
                targets = [x.id for x in n.targets if isinstance(x, ast.Name)]
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                targets = [n.target.id]
            if set(targets) & set(const_names):
                nodes.append(n)
        mod = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), ns_sr)

    _load_ast(
        b_path,
        {"_deserialize_obs_rows", "_serialize_obs_rows", "evaluate_scope_guard", "validate_observable_scope_guard", "extract_guard_ast_dependencies", "shrink_claim_to_observable_scope", "replay_evidence_batches_onto_branches"},
        ("FORBIDDEN_SCOPE_TOKENS", "PUBLIC_OBSERVABLE_GUARD_MANIFEST", "OBSERVABLE_REGIME_FEATURES"),
    )
    _load_ast(i_path, {"compute_evidence_archive_fingerprint"})
    ns_sr["compute_observable_regime_features"] = lambda *a, **k: deepcopy(reg)
    ns_sr["instantiate_aligned_claim_program"] = lambda b, sc: {"aligned": True, "program": Program("const", value=b["value"])}
    ns_sr["loss"] = lambda p, rows, *a: sum((p.predict(r) - r.target) ** 2 for r in rows) / len(rows)
    tmpl_calls: list[int] = []
    ns_sr["_cross_evaluate_refittable_template_on_neighbor"] = lambda *a, **k: (
        tmpl_calls.append(1) or {"evaluated": True, "empirically_refuted": False, "verdict": "supported_on_batch"}
    )
    ns_sr["_refine_coexisting_branch_scopes_and_routing"] = lambda bs, **k: bs

    bs = [
        {"claim_id": "branch:old", "source_concept_id": "old", "value": 1.0, "status": "universal", "scope_claim_level": "bound_instance", "independent_empirical_support": True, "pilot_regime_features": [deepcopy(reg)]},
        {"claim_id": "branch:new", "source_concept_id": "new", "value": 0.0, "status": "universal", "scope_claim_level": "bound_instance", "independent_empirical_support": True, "pilot_regime_features": [deepcopy(reg)]},
    ]
    ev = [{"evidence_id": "ev:new", "schema": {"defined": True}, "observations_serialized": [{"features": {}, "target": 0.0}], "transitions": [{"dummy": True}], "protocol_id": "paid_episode_transitions", "radius_value": 0.75, "commitment": {"radius_parameters": [{"direction": "right"}]}}]
    rb, stats = ns_sr["replay_evidence_batches_onto_branches"](bs, ev)
    fp = ns_sr["compute_evidence_archive_fingerprint"](ev)
    ev_r = deepcopy(ev); ev_r[0]["radius_value"] = 0.90
    ev_c = deepcopy(ev); ev_c[0]["commitment"]["radius_parameters"][0]["direction"] = "left"
    ev_d = deepcopy(ev); ev_d[0]["observations_serialized"][0]["target"] = 0.1
    probe_scope_res = {
        "replayed_old_status": rb[0]["status"],
        "template_check_calls": len(tmpl_calls),
        "old_independent_empirical_support": rb[0]["independent_empirical_support"],
        "stats": stats,
        "radius_edit_invalidates": fp != ns_sr["compute_evidence_archive_fingerprint"](ev_r),
        "commitment_edit_invalidates": fp != ns_sr["compute_evidence_archive_fingerprint"](ev_c),
        "observation_edit_invalidates": fp != ns_sr["compute_evidence_archive_fingerprint"](ev_d),
    }
    assert probe_scope_res["replayed_old_status"] == "instance_scope_unknown"
    assert probe_scope_res["template_check_calls"] == 1
    assert probe_scope_res["radius_edit_invalidates"] is True
    assert probe_scope_res["commitment_edit_invalidates"] is True
    assert probe_scope_res["observation_edit_invalidates"] is True
    return {
        "remediated_probe_d3_object_removal": probe_removal_res,
        "remediated_probe_scope_replay": probe_scope_res,
    }


def _run_three_object_removal_controls(qa_1041: dict[str, Any]) -> dict[str, Any]:
    """Run the 3 mandatory object-removal controls (§3.3) on real `KnowledgeLibrary` instances."""
    base_lib: KnowledgeLibrary = deepcopy(qa_1041["repaired_library"])
    rep_cid = str(qa_1041["repaired_concept_id"])
    rep_ver = str(qa_1041["repaired_semantic_version"])
    surviving_cid = base_lib.concepts[1].concept_id

    with AnonymousObjectLab(seed=1_041_035, objects=4, readout="focus", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.92, latent_mode="focal_memory", latent_decay=0.54, latent_weight=0.36, actuator_curvature=0.15, name_seed=1_041_135, wiring_seed=1_041_235) as lab_ref:
        fine_spec = lab_ref.spec_for(("fine",))

    sample_rows = [Observation({fn: 0.15 for fn in fine_spec.feature_names}, 0.15) for _ in range(8)]
    excl_cid, _ = base_lib.learn(
        Candidate("v15_ctrl2_excl", Program("add", children=(Program("feature", name=fine_spec.feature_names[0]), Program("const", value=0.123))), "derived_exclusive", parents=(rep_cid,)),
        fine_spec,
        sample_rows,
    )
    shared_cid, _ = base_lib.learn(
        Candidate("v15_ctrl2_shared", Program("add", children=(Program("feature", name=fine_spec.feature_names[0]), Program("const", value=0.234))), "derived_two_source", parents=(surviving_cid, rep_cid)),
        fine_spec,
        sample_rows,
    )

    pre_inventory = export_library_object_inventory(base_lib)
    pruned_lib = remove_library_object_version(base_lib, rep_cid, rep_ver, mode="repaired_version")
    post_inventory = export_library_object_inventory(pruned_lib)

    from hegel_machine.cognition.meta_prior_research import _sources

    cfg_chk = ResearchConfig(seed=1_041_999, observation_budget=120, compute_budget=60_000, use_memory=True, world_structure_commitments=True)
    recalled_after = UnifiedResearchAgent(cfg_chk, library=pruned_lib)._recall(fine_spec)
    recalled_sources = sorted(
        {
            str(s)
            for item in recalled_after
            for s in (_sources(item) if isinstance(item, Program) else [*(item.get("method_source_ids") or ()), *(item["candidate"].parents or ())])
        }
    )
    assert rep_cid not in post_inventory["objects"]
    assert excl_cid not in post_inventory["objects"]
    assert shared_cid in post_inventory["objects"]
    assert rep_cid not in recalled_sources and excl_cid not in recalled_sources

    # Control 3: Billed reconstruction from preserved raw history
    lib_for_rebuild = remove_library_object_version(qa_1041["repaired_library"], rep_cid, rep_ver, mode="repaired_version")
    rebuild_budget = SearchBudget(350_000)
    rebuild_actions: list[dict[str, Any]] = []
    rebuild_events = reconstruct_missing_objects_from_raw_history(
        lib_for_rebuild, rebuild_budget, actions=rebuild_actions, tolerance=EXACT_TOLERANCE
    )
    assert len(rebuild_events) == 1 and rebuild_events[0]["generated_from_raw_history"] is True
    rebuilt_inv = export_library_object_inventory(lib_for_rebuild)

    return {
        "control_1_true_list_shaped_version_removal": {
            "removed_concept_id": rep_cid,
            "removed_semantic_version": rep_ver,
            "pre_removal_inventory_summary": {
                "object_count": len(pre_inventory["objects"]),
                "removed_target_entry_before": pre_inventory["objects"][rep_cid],
                "raw_evidence_batches_before": pre_inventory["raw_reconciliation_evidence_batches_count"],
            },
            "post_removal_inventory_summary": {
                "object_count": len(post_inventory["objects"]),
                "removed_target_present_in_concepts": any(c.concept_id == rep_cid for c in pruned_lib.concepts),
                "removed_target_present_in_abstracts": any(a.source_concept_id == rep_cid for a in pruned_lib.abstracts),
                "removed_target_present_in_definitions": any(d.source_concept_id == rep_cid for d in pruned_lib.definitions.values()),
                "removed_target_present_in_bindings": any(k.startswith(f"{rep_cid}:") for k in pruned_lib.definition_bindings),
                "removed_target_present_in_recall": rep_cid in recalled_sources,
                "surviving_recallable_sources": recalled_sources,
                "raw_evidence_batches_after": post_inventory["raw_reconciliation_evidence_batches_count"],
            },
        },
        "control_2_exclusive_derivative_vs_two_source_shared_retention": {
            "exclusive_derivative_cid": excl_cid,
            "exclusive_derivative_removed": excl_cid not in post_inventory["objects"],
            "two_source_shared_cid": shared_cid,
            "two_source_shared_retained": shared_cid in post_inventory["objects"],
            "two_source_surviving_entry": post_inventory["objects"][shared_cid],
        },
        "control_3_billed_raw_history_reconstruction": {
            "reconstruction_event": rebuild_events[0],
            "action_trace": rebuild_actions,
            "reconstructed_object_inventory_entry": rebuilt_inv["objects"][rebuild_events[0]["reconstructed_concept_id"]],
        },
    }


def _verify_all_8_d2_source_histories_k_vs_r_parity() -> dict[str, Any]:
    """§4.3: Verify K vs R source end-state & first-dispatch parity across 1031000 and all 8 exposed D2 source histories!"""
    per_chain_parity: list[dict[str, Any]] = []
    total_pairs = 0
    total_diffs = 0
    replay_1031_t1: dict[str, Any] = {}
    replay_1037_diag: dict[str, Any] = {}

    for spec_d2 in EXPOSED_D2_SPECS:
        cid = spec_d2["chain_id"]
        base_seed = int(spec_d2["base_seed"])
        order_mode = spec_d2["arrival_order_mode"]
        th_a, th_b, th_left = spec_d2["source_thresholds"]
        r_0, r_a, r_b, r_l = spec_d2["source_radii"]
        th_t1, th_t3 = spec_d2["target_thresholds"]
        r_t1, r_t2, r_t3 = spec_d2["target_radii"]

        shared_lib = KnowledgeLibrary()
        shared_lib._prioritize_generalized_families = False
        _run_episode_with_telemetry(
            arm_name="common_calibration_anchor",
            episode_tag=f"{cid}:S_cal_neutral_N4",
            lab_kwargs={
                "seed": base_seed + 5, "objects": 4, "readout": "window_total_event",
                "relation_direction": "symmetric", "relation_condition": "always",
                "condition_threshold": 0.0, "relation_radius": 0.92,
                "actuator_curvature": 0.15, "name_seed": base_seed + 105, "wiring_seed": base_seed + 205,
            },
            library=shared_lib,
            cfg=_make_d2_config(base_seed + 5, obs_b=D2_SOURCE_OBS_BUDGET, comp_b=D2_SOURCE_COMPUTE_BUDGET),
            meas_b=D2_SOURCE_MEAS_BUDGET,
        )
        cold_snap = deepcopy(shared_lib)

        src_defs = {
            "S0": (f"{cid}:S0_right_always_N4", base_seed + 15, {"seed": base_seed + 15, "objects": 4, "readout": "window_total_event", "relation_direction": "right", "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": r_0, "actuator_curvature": 0.15, "name_seed": base_seed + 115, "wiring_seed": base_seed + 215}),
            "S1": (f"{cid}:S1_right_dpg_a_N4", base_seed + 25, {"seed": base_seed + 25, "objects": 4, "readout": "window_total_event", "relation_direction": "right", "relation_condition": "donor_positive_gap", "condition_threshold": th_a, "relation_radius": r_a, "actuator_curvature": 0.15, "name_seed": base_seed + 125, "wiring_seed": base_seed + 225}),
            "S2": (f"{cid}:S2_right_dpg_b_N4", base_seed + 35, {"seed": base_seed + 35, "objects": 4, "readout": "window_total_event", "relation_direction": "right", "relation_condition": "donor_positive_gap", "condition_threshold": th_b, "relation_radius": r_b, "actuator_curvature": 0.15, "name_seed": base_seed + 135, "wiring_seed": base_seed + 235}),
            "S_left": (f"{cid}:S_left_dpg_N4", base_seed + 45, {"seed": base_seed + 45, "objects": 4, "readout": "window_total_event", "relation_direction": "left", "relation_condition": "donor_positive_gap", "condition_threshold": th_left, "relation_radius": r_l, "actuator_curvature": 0.15, "name_seed": base_seed + 145, "wiring_seed": base_seed + 245}),
        }
        order_keys = (
            ["S0", "S1", "S2", "S_left"]
            if order_mode == "unconditional_first"
            else (["S1", "S2", "S0", "S_left"] if order_mode == "conditional_first_reversed" else ["S1", "S_left", "S0", "S2"])
        )
        src_summaries = {}
        for k_s in order_keys:
            ep_tag, ep_sd, l_kw = src_defs[k_s]
            s_out, _ = _run_episode_with_telemetry(
                arm_name="ordinary_source_episode",
                episode_tag=ep_tag,
                lab_kwargs=l_kw,
                library=shared_lib,
                cfg=_make_d2_config(ep_sd, obs_b=D2_SOURCE_OBS_BUDGET, comp_b=D2_SOURCE_COMPUTE_BUDGET),
                meas_b=D2_SOURCE_MEAS_BUDGET,
            )
            src_summaries[k_s] = s_out

        arm_libs = _build_d2_arm_libraries(cold_snap, shared_lib)
        lib_k = arm_libs["K_eager_noH"]
        lib_r = arm_libs["R_lazy_noH"]

        # Verify `partial_pending` budget interruption on `1031000`
        partial_check = None
        if cid == "unexposed_v14_chain_1031000":
            lib_r_tight = deepcopy(lib_r)
            _, st_tight = reconstruct_parametric_families_from_ordinary_archive(
                lib_r_tight, SearchBudget(35), reconstruct_routing=True, build_families=False
            )
            assert st_tight["reconstruction_status"] == "partial_pending" and st_tight["interrupted"] is True
            partial_check = {
                "reconstruction_status": st_tight["reconstruction_status"],
                "interrupted": st_tight["interrupted"],
                "completed_steps": st_tight["completed_steps"],
                "total_steps": st_tight["total_steps"],
            }

        # Reconstruct R end-state and compare source-by-source branch end-states
        lib_r_probe = deepcopy(lib_r)
        _, r_recon_stats = reconstruct_parametric_families_from_ordinary_archive(
            lib_r_probe, SearchBudget(200_000), reconstruct_routing=True, build_families=False
        )
        k_end = {
            b["source_concept_id"]: {"status": b["status"], "scope_claim_level": b.get("scope_claim_level"), "has_scope_guard": b.get("scope_guard") is not None, "has_routing_guard": b.get("routing_guard") is not None}
            for b in lib_k._active_reconciled_branches
        }
        r_end = {
            b["source_concept_id"]: {"status": b["status"], "scope_claim_level": b.get("scope_claim_level"), "has_scope_guard": b.get("scope_guard") is not None, "has_routing_guard": b.get("routing_guard") is not None}
            for b in r_recon_stats["reconstructed_branches"]
        }
        assert k_end == r_end, f"End-state mismatch on {cid}: K={k_end} vs R={r_end}"

        # Verify first-dispatch parity across T1, T2, T3
        t_specs = [
            ("T1_unseen_family", base_seed + 305, {"seed": base_seed + 305, "objects": 5, "readout": "window_total_event", "relation_direction": "right", "relation_condition": "donor_positive_gap", "condition_threshold": th_t1, "relation_radius": r_t1, "actuator_curvature": 0.15, "name_seed": base_seed + 405, "wiring_seed": base_seed + 505}),
            ("T2_old_unconditional_domain", base_seed + 315, {"seed": base_seed + 315, "objects": 5, "readout": "window_total_event", "relation_direction": "right", "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": r_t2, "actuator_curvature": 0.15, "name_seed": base_seed + 415, "wiring_seed": base_seed + 515}),
            ("T3_minority_direction_domain", base_seed + 325, {"seed": base_seed + 325, "objects": 5, "readout": "window_total_event", "relation_direction": "left", "relation_condition": "donor_positive_gap", "condition_threshold": th_t3, "relation_radius": r_t3, "actuator_curvature": 0.15, "name_seed": base_seed + 425, "wiring_seed": base_seed + 525}),
        ]
        chain_diffs = 0
        sc_ref = lib_k.meta_priors[-1]["object_schema"]
        for t_role, t_sd, t_kw in t_specs:
            with AnonymousObjectLab(**t_kw) as t_lab:
                fb_t = t_lab.spec_for(()).feature_bounds
                p_seq = tuple(
                    {
                        fn: float(lo + (0.2 + 0.3 * ((i + j) % 3)) * (hi - lo))
                        for j, (fn, (lo, hi)) in enumerate(fb_t.items())
                    }
                    for i in range(12)
                )
                acq_b = t_lab.acquire(Probe(f"parity_probe:{cid}:{t_role}", p_seq), ("fine",))
            dk = dispatch_branches_and_families_for_target(
                library=deepcopy(lib_k), consistent_comms=[], transitions=acq_b.transitions, schema=sc_ref, audit=None, task_sig=f"{cid}:{t_role}", comm_budget=SearchBudget(150_000)
            )["telemetry"]
            dr = dispatch_branches_and_families_for_target(
                library=deepcopy(lib_r), consistent_comms=[], transitions=acq_b.transitions, schema=sc_ref, audit=None, task_sig=f"{cid}:{t_role}", comm_budget=SearchBudget(150_000)
            )["telemetry"]
            nk, nr = _normalize_dispatch(dk), _normalize_dispatch(dr)
            total_pairs += 1
            if nk != nr:
                chain_diffs += 1
                total_diffs += 1

        # Full agent target replays on affected exposed cases `1031000/T1` and `1037000/T3` (§5.2)
        if cid == "unexposed_v14_chain_1031000":
            t1_k_ep, _ = _run_episode_with_telemetry(
                arm_name="K_eager_noH", episode_tag=f"{cid}:T1_unseen_right_dpg_N5", lab_kwargs=t_specs[0][2],
                library=deepcopy(lib_k), cfg=_make_d2_config(t_specs[0][1], obs_b=D2_TARGET_OBS_BUDGET, comp_b=D2_TARGET_COMPUTE_BUDGET), meas_b=D2_TARGET_MEAS_BUDGET,
            )
            t1_r_ep, _ = _run_episode_with_telemetry(
                arm_name="R_lazy_noH", episode_tag=f"{cid}:T1_unseen_right_dpg_N5", lab_kwargs=t_specs[0][2],
                library=deepcopy(lib_r), cfg=_make_d2_config(t_specs[0][1], obs_b=D2_TARGET_OBS_BUDGET, comp_b=D2_TARGET_COMPUTE_BUDGET), meas_b=D2_TARGET_MEAS_BUDGET,
            )
            nk_live = _normalize_dispatch(t1_k_ep["branch_dispatch_telemetry"])
            nr_live = _normalize_dispatch(t1_r_ep["branch_dispatch_telemetry"])
            assert nk_live == nr_live
            replay_1031_t1 = {
                "historical_v14_divergence_root_cause": (
                    "In V14, R initialized all 5 future source branches simultaneously as universal and replayed only `ev_probe` batches "
                    "while marking `scope_unidentified` branches (`concept_0001`, `concept_0005`) as global `refuted` (`support_invalidated`)."
                ),
                "v15_normalized_first_dispatch_K": nk_live,
                "v15_normalized_first_dispatch_R": nr_live,
                "first_dispatch_exact_match": nk_live == nr_live,
                "partial_pending_budget_check": partial_check,
                "K_T1_outcome": {"status": t1_k_ep["status"], "strict_pass": t1_k_ep["strict_pass"], "external_fresh48_mse": t1_k_ep["external_fresh48_mse"], "measured_costs": t1_k_ep["measured_costs"]},
                "R_T1_outcome": {"status": t1_r_ep["status"], "strict_pass": t1_r_ep["strict_pass"], "external_fresh48_mse": t1_r_ep["external_fresh48_mse"], "measured_costs": t1_r_ep["measured_costs"]},
            }

        if cid == "unexposed_v14_chain_1037000":
            t3_k_ep, _ = _run_episode_with_telemetry(
                arm_name="K_eager_noH", episode_tag=f"{cid}:T3_minority_left_dpg_N5", lab_kwargs=t_specs[2][2],
                library=deepcopy(lib_k), cfg=_make_d2_config(t_specs[2][1], obs_b=D2_TARGET_OBS_BUDGET, comp_b=D2_TARGET_COMPUTE_BUDGET), meas_b=D2_TARGET_MEAS_BUDGET,
            )
            t3_r_ep, _ = _run_episode_with_telemetry(
                arm_name="R_lazy_noH", episode_tag=f"{cid}:T3_minority_left_dpg_N5", lab_kwargs=t_specs[2][2],
                library=deepcopy(lib_r), cfg=_make_d2_config(t_specs[2][1], obs_b=D2_TARGET_OBS_BUDGET, comp_b=D2_TARGET_COMPUTE_BUDGET), meas_b=D2_TARGET_MEAS_BUDGET,
            )
            replay_1037_diag = {
                "root_cause_of_v14_unconfirmed_yet_accurate": (
                    "On `1037000/T3`, `R_lazy_noH` found the exact model (`external_fresh48_mse = 4.74e-22`, `confirmation_loss = 6.61e-22`) "
                    "but `research_inherited_relation_intervals` only cleared `pending_relation_candidates` on `plan['probe'] is None` when top-level "
                    "`world_commitment_application` was present, leaving an indistinguishable sub-cell clone that triggered `retain_relation_competition`."
                ),
                "S2_source_episode": {"status": src_summaries["S2"]["status"], "strict_pass": src_summaries["S2"]["strict_pass"], "external_fresh48_mse": src_summaries["S2"]["external_fresh48_mse"], "measured_costs": src_summaries["S2"]["measured_costs"]},
                "K_T3_outcome": {"status": t3_k_ep["status"], "strict_pass": t3_k_ep["strict_pass"], "confirmation_loss": t3_k_ep["confirmation_loss"], "external_fresh48_mse": t3_k_ep["external_fresh48_mse"], "measured_costs": t3_k_ep["measured_costs"]},
                "R_T3_outcome": {"status": t3_r_ep["status"], "strict_pass": t3_r_ep["strict_pass"], "confirmation_loss": t3_r_ep["confirmation_loss"], "external_fresh48_mse": t3_r_ep["external_fresh48_mse"], "measured_costs": t3_r_ep["measured_costs"]},
            }

        per_chain_parity.append(
            {
                "chain_id": cid,
                "arrival_order_mode": order_mode,
                "source_end_state_exact_match": k_end == r_end,
                "branch_count": len(k_end),
                "target_dispatch_differences": chain_diffs,
                "source_end_state": k_end,
            }
        )

    assert total_diffs == 0, f"Expected 0 dispatch differences across all 24 D2 pairs, got {total_diffs}"
    return {
        "d2_8_chain_parity_summary": {
            "chains_verified": len(per_chain_parity),
            "target_pairs_verified": total_pairs,
            "semantic_dispatch_differences_total": total_diffs,
            "per_chain": per_chain_parity,
        },
        "exposed_replay_1031000_source_and_T1": replay_1031_t1,
        "exposed_replay_1037000_S2_and_T3": replay_1037_diag,
    }


def _run_exposed_d3_and_d1e_replays(qa_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run the remaining §5.2 exposed dev diagnostic replays (`1041000/T1`, `1042000/T1`, `1043000 & 1045000`, `D1 Replay E`)."""
    # 1. `1041000/T1`: True versioned removal + billed raw-history reconstruction on actual unseen N=5 target
    sp_1041 = EXPOSED_D3_SPECS[0]
    qa_1041 = qa_cache[sp_1041["chain_id"]]
    rcid_41 = qa_1041["repaired_concept_id"]
    rver_41 = qa_1041["repaired_semantic_version"]
    libs_41 = _build_d3_target_libraries(qa_1041["cold_snapshot"], qa_1041["repaired_library"], rcid_41, rver_41)
    d_t1, w_t1, r_t1 = sp_1041["target_t1_focal_params"]
    t1_41_kw = {
        "seed": 1_041_305, "objects": 5, "readout": "focus", "relation_direction": "right",
        "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": r_t1,
        "latent_mode": "focal_memory", "latent_decay": d_t1, "latent_weight": w_t1,
        "actuator_curvature": 0.15, "name_seed": 1_041_405, "wiring_seed": 1_041_505,
    }
    t_cfg_41 = ResearchConfig(
        seed=1_041_305, observation_budget=D3_TARGET_OBS_BUDGET, compute_budget=D3_TARGET_COMPUTE_BUDGET,
        max_rounds=4, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True,
        world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True,
        latent_state_discovery=True, active_threshold_probing=True, state_search=True,
        indexed_commitment_store=True, sensor_role_calibration_priors=True, occam_premise_minimality=True,
        branch_reconciliation=True,
    )
    ep_41_k, _ = _run_episode_with_telemetry(
        arm_name="K_eager_noH", episode_tag="1041000:T1_unseen_focal_memory_N5", lab_kwargs=t1_41_kw,
        library=deepcopy(libs_41["K_eager_noH"]), cfg=t_cfg_41, meas_b=D3_TARGET_MEAS_BUDGET,
    )
    lib_41_rm = deepcopy(libs_41["K_minimal_repaired_object_removed_other_sources_kept"])
    pre_inv_41_rm = export_library_object_inventory(lib_41_rm)
    ep_41_rm, raw_41_rm = _run_episode_with_telemetry(
        arm_name="K_true_repaired_version_removed_raw_rebuild_allowed",
        episode_tag="1041000:T1_unseen_focal_memory_N5",
        lab_kwargs=t1_41_kw,
        library=lib_41_rm,
        cfg=t_cfg_41,
        meas_b=D3_TARGET_MEAS_BUDGET,
    )
    raw_recon_acts_41 = [
        a for a in raw_41_rm["actions"] if a.get("action") == "complete_raw_history_object_reconstruction"
    ]

    # 2. `1042000/T1`: Verify 5-stage structured `consumption_events` vs old substring `True`
    sp_1042 = EXPOSED_D3_SPECS[1]
    qa_1042 = qa_cache[sp_1042["chain_id"]]
    rcid_42 = qa_1042["repaired_concept_id"]
    rver_42 = qa_1042["repaired_semantic_version"]
    libs_42 = _build_d3_target_libraries(qa_1042["cold_snapshot"], qa_1042["repaired_library"], rcid_42, rver_42)
    d_t42, w_t42, r_t42 = sp_1042["target_t1_focal_params"]
    t1_42_kw = {
        "seed": 1_042_305, "objects": 5, "readout": "focus", "relation_direction": "right",
        "relation_condition": "always", "condition_threshold": 0.0, "relation_radius": r_t42,
        "latent_mode": "focal_memory", "latent_decay": d_t42, "latent_weight": w_t42,
        "actuator_curvature": 0.15, "name_seed": 1_042_405, "wiring_seed": 1_042_505,
    }
    t_cfg_42 = ResearchConfig(
        seed=1_042_305, observation_budget=D3_TARGET_OBS_BUDGET, compute_budget=D3_TARGET_COMPUTE_BUDGET,
        max_rounds=4, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True,
        world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True,
        latent_state_discovery=True, active_threshold_probing=True, state_search=True,
        indexed_commitment_store=True, sensor_role_calibration_priors=True, occam_premise_minimality=True,
        branch_reconciliation=True,
    )
    ep_42_k, raw_42_k = _run_episode_with_telemetry(
        arm_name="K_eager_noH", episode_tag="1042000:T1_unseen_focal_memory_N5", lab_kwargs=t1_42_kw,
        library=deepcopy(libs_42["K_eager_noH"]), cfg=t_cfg_42, meas_b=D3_TARGET_MEAS_BUDGET,
    )
    res_cons_42 = [
        ev
        for act in raw_42_k["actions"]
        if act.get("action") == "resolve_commitment_consumption"
        for ev in (act.get("consumption_events") or ())
    ]

    # 3. `D1 Replay E`: Verify persistent `pending_unfunded` continuation, cumulative budget, & post-subresearch re-reconcile unbroken!
    lib_e = KnowledgeLibrary()
    for sd_e, dir_e in ((9700, "symmetric"), (9701, "right")):
        with AnonymousObjectLab(seed=sd_e, objects=3, readout="focus", relation_direction=dir_e, relation_condition="always", condition_threshold=0.0, relation_radius=0.92, actuator_curvature=0.15, name_seed=sd_e + 100, wiring_seed=sd_e + 200) as lab_e:
            UnifiedResearchAgent(
                ResearchConfig(seed=sd_e, observation_budget=520, compute_budget=520_000, max_rounds=4, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True, world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True, active_threshold_probing=True, indexed_commitment_store=True, sensor_role_calibration_priors=True, occam_premise_minimality=True, branch_reconciliation=True),
                measurement_budget=11_000,
                library=lib_e,
            ).run(lab_e)

    conf_e_kw = dict(seed=9705, objects=3, readout="focus", relation_direction="right", relation_condition="always", condition_threshold=0.0, relation_radius=0.91, latent_mode="focal_memory", latent_decay=0.56, latent_weight=0.35, actuator_curvature=0.15, name_seed=9805, wiring_seed=9905)
    with AnonymousObjectLab(**conf_e_kw) as lab_e1:
        r_e1 = UnifiedResearchAgent(
            ResearchConfig(seed=9705, observation_budget=156, compute_budget=260_000, max_rounds=1, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True, world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=False, latent_state_discovery=False, active_threshold_probing=False, indexed_commitment_store=True, sensor_role_calibration_priors=True, occam_premise_minimality=True, branch_reconciliation=True),
            measurement_budget=4_200,
            library=lib_e,
        ).run(lab_e1)
    assert r_e1["status"] == "unresolved"
    assert any(q.get("status") in {"pending_unfunded", "pending", "pending_reconciliation"} for q in lib_e._pending_reconciliation_questions)

    with AnonymousObjectLab(**conf_e_kw) as lab_e2:
        r_e2 = UnifiedResearchAgent(
            ResearchConfig(seed=9705, observation_budget=560, compute_budget=780_000, max_rounds=2, minimum_improvement=EXACT_TOLERANCE, max_candidates=4, use_memory=True, world_structure_commitments=True, transition_premise_audit=True, expanded_premise_revisions=True, latent_state_discovery=True, active_threshold_probing=True, indexed_commitment_store=True, sensor_role_calibration_priors=True, occam_premise_minimality=True, branch_reconciliation=True),
            measurement_budget=12_000,
            library=lib_e,
        ).run(lab_e2)
    assert r_e2["status"] == "completed"
    assert r_e2["costs"]["observation_cost"] > r_e1["costs"]["observation_cost"]
    assert r_e2["costs"]["compute_cost"] > r_e1["costs"]["compute_cost"]

    return {
        "exposed_replay_1041000_T1_true_removal_and_rebuild": {
            "removed_repaired_concept_id": rcid_41,
            "removed_repaired_semantic_version": rver_41,
            "pre_run_removed_arm_object_ids": sorted(pre_inv_41_rm["objects"].keys()),
            "removed_concept_absent_pre_run": rcid_41 not in pre_inv_41_rm["objects"],
            "K_eager_noH_T1": {"status": ep_41_k["status"], "strict_pass": ep_41_k["strict_pass"], "external_fresh48_mse": ep_41_k["external_fresh48_mse"], "measured_costs": ep_41_k["measured_costs"]},
            "K_true_removed_rebuild_allowed_T1": {
                "status": ep_41_rm["status"],
                "strict_pass": ep_41_rm["strict_pass"],
                "external_fresh48_mse": ep_41_rm["external_fresh48_mse"],
                "measured_costs": ep_41_rm["measured_costs"],
                "raw_history_reconstruction_events": raw_recon_acts_41,
            },
        },
        "exposed_replay_1042000_T1_structured_5stage_attribution": {
            "repaired_concept_id": rcid_42,
            "repaired_semantic_version": rver_42,
            "status": ep_42_k["status"],
            "strict_pass": ep_42_k["strict_pass"],
            "measured_costs": ep_42_k["measured_costs"],
            "structured_episode_resolution_events": res_cons_42,
            "adopted_source_ids_proven_by_5stage_chain": sorted(
                {
                    s
                    for ev in res_cons_42
                    if ev.get("adopted_confirmed")
                    for s in (ev.get("source_concept_ids") or ())
                }
            ),
        },
        "exposed_replay_1043000_and_1045000_failed_chain_honesty": {
            cid_f: {
                "qualified_repaired_object": qa_cache[cid_f]["qualified_repaired_object"],
                "repaired_concept_id": qa_cache[cid_f]["repaired_concept_id"],
                "initial_reconciliation_status": qa_cache[cid_f]["initial_conflict_reconciliation_event"]["epistemic_status"],
                "subquestion_resolution_event": qa_cache[cid_f]["subquestion_resolution_event"],
                "post_subresearch_re_reconciliation_status": qa_cache[cid_f]["post_subresearch_re_reconciliation_event"]["epistemic_status"],
            }
            for cid_f in ("unexposed_v14_focal_chain_1043000", "unexposed_v14_focal_chain_1045000")
        },
        "exposed_replay_d1_replay_e_persistent_pending_continuation": {
            "run1_status": r_e1["status"],
            "run1_costs": r_e1["costs"],
            "run2_status": r_e2["status"],
            "run2_cumulative_costs": r_e2["costs"],
            "cumulative_accounting_preserved": bool(
                r_e2["costs"]["observation_cost"] > r_e1["costs"]["observation_cost"]
                and r_e2["costs"]["compute_cost"] > r_e1["costs"]["compute_cost"]
            ),
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    before_hashes = {p.name: _sha256_file(p) for p in sorted(V14_DIR.glob("*.json"))}

    print("[v15 Step 1/3] Running Deliverable 1 (10-item V14 Errata + Status-Aware Factual Re-Export)...", flush=True)
    d1_receipt, qa_cache = run_deliverable_1_errata_and_reexport()

    print("[v15 Step 2/3] Verifying GPT-6 AST Probes + 3 Object-Removal Controls + 8-Chain K/R State Parity...", flush=True)
    ast_probes = _verify_gpt6_ast_probes_on_updated_code()
    removal_controls = _run_three_object_removal_controls(qa_cache["unexposed_v14_focal_chain_1041000"])
    d2_parity_and_replays = _verify_all_8_d2_source_histories_k_vs_r_parity()

    print("[v15 Step 3/3] Running Remaining §5.2 Exposed Development Diagnostic Replays...", flush=True)
    d3_and_d1e_replays = _run_exposed_d3_and_d1e_replays(qa_cache)

    receipt_d2 = {
        "schema_version": "hegel_v15_deliverable_2_removal_replay_parity_and_exposed_diagnostics_v1",
        "generated_at_utc": _utc_now(),
        "deliverable": "v15_Deliverable_2_True_Removal_Controls_KR_Parity_And_Exposed_Replays",
        "exposed_development_material_only": True,
        "not_prospective_generalization_evidence": True,
        "paid_api_budget": {"model_calls": 0, "total_tokens": 0},
        "code_provenance": _source_code_provenance(),
        "remediated_gpt6_v15_probes": ast_probes,
        "three_object_removal_controls_sec3_3": removal_controls,
        "eight_chain_k_vs_r_state_and_dispatch_parity_sec4_3": d2_parity_and_replays["d2_8_chain_parity_summary"],
        "six_exposed_development_replays_sec5_2": {
            "case_1_1031000_source_and_T1_parity": d2_parity_and_replays["exposed_replay_1031000_source_and_T1"],
            "case_2_1041000_T1_true_removal_and_rebuild_cost": d3_and_d1e_replays["exposed_replay_1041000_T1_true_removal_and_rebuild"],
            "case_3_1042000_T1_structured_vs_substring_consumption": d3_and_d1e_replays["exposed_replay_1042000_T1_structured_5stage_attribution"],
            "case_4_1043000_and_1045000_unresolved_status_aware_narratives": d3_and_d1e_replays["exposed_replay_1043000_and_1045000_failed_chain_honesty"],
            "case_5_1037000_S2_and_T3_accurate_completion_fix": d2_parity_and_replays["exposed_replay_1037000_S2_and_T3"],
            "case_6_d1_replay_e_persistent_pending_resumption": d3_and_d1e_replays["exposed_replay_d1_replay_e_persistent_pending_continuation"],
        },
        "total_wall_seconds": round(time.perf_counter() - t0, 3),
    }
    out_d2 = OUT_DIR / "deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json"
    out_d2.write_text(json.dumps(receipt_d2, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    after_hashes = {p.name: _sha256_file(p) for p in sorted(V14_DIR.glob("*.json"))}
    assert before_hashes == after_hashes, "IRON RULE VIOLATION: Historic V14 JSON receipt was modified!"
    print(
        json.dumps(
            {
                "status": "completed",
                "deliverable_1": str(OUT_DIR / "deliverable_1_v14_errata_and_factual_reexport_20261118.json"),
                "deliverable_2": str(out_d2),
                "historic_v14_receipts_untouched": before_hashes == after_hashes,
                "d2_dispatch_differences_across_24_pairs": d2_parity_and_replays["d2_8_chain_parity_summary"]["semantic_dispatch_differences_total"],
                "wall_seconds": receipt_d2["total_wall_seconds"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

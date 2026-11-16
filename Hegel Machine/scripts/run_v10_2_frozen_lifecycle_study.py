"""Run v10.2 Deliverable C: Frozen 6-Stage v4 Knowledge Lifecycle Study (model_calls = 0).

Writes new receipt (zero modification to existing 13 v10 receipts):
  ``docs/evidence/v10_2_audit_remediation_20261116/frozen_v4_lifecycle_20261116.json``

Executes the full 6-stage v4 scientific lifecycle (Contract C1-C3, Behavioral Acceptance #9-#13,
#15, #16, #18, #21) across:
  - 1 Development Qualification Chain (``dev_chain_700000``, excluded from prospective aggregates)
  - 4 Pre-Registered Contiguous Prospective Physical-Source Chains (``711000, 712000, 713000, 714000``,
    zero seed cherry-picking; physical source is the sample unit ``n = 4``):

Stages per chain:
  1. Source Formation (N=4, right, donor_positive_gap): open-grammar AST discovery from anonymous
     sensors, full failed-candidate denominator retained, distinct fit/selection/confirmation metadata.
  2. Freeze Before Target Exposure: SHA256-locked proposer catalog, ResearchConfig, and source commitment.
  3. Shifted-Arity & Shifted-Parameter Transfer (N=5, right, donor_positive_gap, shifted threshold & radius;
     verifies direct source-float plug-in fails while target re-identification succeeds).
  4. Unhinted Counterexample (N=5, left, donor_positive_gap): refutes P_interaction_direction:right
     without deleting learned predicate structure; transitions C_0 to status='conditional'.
  5. Sub-Research & Conservative Scope-Bounded Revision: retains unaffected donor_positive_gap & readout
     columns while revising ONLY direction right->left to form C_1 (generation=1).
  6. Return of Original Condition (N=6, right, donor_positive_gap): verifies newer C_1 (generation=1,
     status='supported') is empirically rejected on returning right transitions (#18), while scope-gated
     C_0 (generation=0) is directly recovered via equivalence_reorganization rather than overwritten (C2).

Compares 3 equal-tool arms: ``cold_full_tools``, ``archive_full_tools`` (allowed to reconstruct
equivalent methods on demand with reconstruction compute charged!), and ``structured_lifecycle_v10``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from scripts.run_v10_phase2b_open_grammar_llm import (
    _acquire_shared_discovery_bundle,
    _build_parent_commitment_from_empty_library,
    _eval_program_heldout_48,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    build_bounded_open_grammar_patch_catalog,
    reconstruct_commitments_from_raw_archive,
    run_single_factor_dependency_ablation_arm,
)
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_terms,
    is_commitment_empirically_consistent,
)
from hegel_machine.cognition.continuous_state import _sum_terms

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _frozen_config(seed: int, *, objects: int, compute_budget: int) -> ResearchConfig:
    return ResearchConfig(
        seed=seed,
        observation_budget=400 if objects >= 6 else 380,
        compute_budget=compute_budget,
        minimum_improvement=1e-7,
        max_candidates=4,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        latent_state_discovery=True,
        higher_order_latent_discovery=True,
        nonlinear_limit_cycle_discovery=True,
        active_resonance_probing=True,
        continuum_field_discovery=True,
        manifold_pde_discovery=True,
    )


def _preregistered_chain_specs() -> list[dict]:
    """1 development qualification chain + 4 contiguous prospective physical-source chains."""
    defs = [
        ("dev_chain_700000", "development_qualification", 700000, 0.24, 1.06, 0.32, 1.15, 0.27, 1.11, 0.19, 1.19),
        ("prospective_chain_1", "frozen_prospective", 711000, 0.22, 1.05, 0.31, 1.14, 0.28, 1.12, 0.20, 1.18),
        ("prospective_chain_2", "frozen_prospective", 712000, 0.26, 1.08, 0.19, 1.16, 0.30, 1.10, 0.23, 1.17),
        ("prospective_chain_3", "frozen_prospective", 713000, 0.21, 1.04, 0.29, 1.13, 0.24, 1.14, 0.31, 1.20),
        ("prospective_chain_4", "frozen_prospective", 714000, 0.28, 1.07, 0.20, 1.15, 0.32, 1.11, 0.22, 1.16),
    ]
    out = []
    for cid, role, base_s, c1, r1, c3, r3, c4, r4, c6, r6 in defs:
        out.append(
            {
                "chain_id": cid,
                "chain_role": role,
                "base_seed": base_s,
                "stage1_source": dict(
                    seed=base_s + 401,
                    apparatus_seed=base_s + 41,
                    objects=4,
                    readout="window_total_event",
                    variant="exchange",
                    relation_direction="right",
                    relation_condition="donor_positive_gap",
                    condition_threshold=c1,
                    relation_radius=r1,
                    actuator_curvature=0.16,
                    name_seed=base_s + 501,
                    wiring_seed=base_s + 601,
                ),
                "stage3_transfer": dict(
                    seed=base_s + 501,
                    apparatus_seed=base_s + 51,
                    objects=5,
                    readout="window_total_event",
                    variant="exchange",
                    relation_direction="right",
                    relation_condition="donor_positive_gap",
                    condition_threshold=c3,
                    relation_radius=r3,
                    actuator_curvature=0.15,
                    name_seed=base_s + 511,
                    wiring_seed=base_s + 611,
                ),
                "stage4_5_counterexample_revision": dict(
                    seed=base_s + 502,
                    apparatus_seed=base_s + 52,
                    objects=5,
                    readout="window_total_event",
                    variant="exchange",
                    relation_direction="left",
                    relation_condition="donor_positive_gap",
                    condition_threshold=c4,
                    relation_radius=r4,
                    actuator_curvature=0.15,
                    name_seed=base_s + 521,
                    wiring_seed=base_s + 621,
                ),
                "stage6_return": dict(
                    seed=base_s + 601,
                    apparatus_seed=base_s + 61,
                    objects=6,
                    readout="window_total_event",
                    variant="exchange",
                    relation_direction="right",
                    relation_condition="donor_positive_gap",
                    condition_threshold=c6,
                    relation_radius=r6,
                    actuator_curvature=0.14,
                    name_seed=base_s + 531,
                    wiring_seed=base_s + 631,
                ),
            }
        )
    return out


def _check_direct_source_float_plugin_mse(source_prior: dict, source_comm: dict, target_lab_kwargs: dict) -> float | None:
    """Verify source numeric floats cannot be plugged directly into shifted target as the answer (C2)."""
    fit_steps = (source_prior.get("structure") or {}).get("fits") or [{}]
    coeffs = fit_steps[0].get("coefficients")
    if not coeffs or not source_comm.get("radius_parameters"):
        return None
    src_r = float(source_comm["radius_parameters"][0]["source_values"][-1])
    sym = source_comm["radius_parameters"][0]["symbol"]
    with AnonymousObjectLab(**target_lab_kwargs) as t_lab:
        true_sc = t_lab._evaluation_schema(("fine",))
        terms = instantiate_commitment_terms(
            source_comm, true_sc, radius_bindings={sym: src_r}
        )
        if terms is None or len(terms) != len(coeffs) - 1:
            return None
        naive_prog = _sum_terms(list(terms), coeffs[1:], coeffs[0])
        rows = t_lab.evaluation_rows(("fine",), seed=target_lab_kwargs["seed"] + 99991, length=48, trajectories=1)
        return float(loss(naive_prog, rows, 0.0, 0.5))


def _run_stage_episode_for_arm(
    *,
    arm_name: str,
    stage_name: str,
    lab_kwargs: dict,
    base_parent_lib: KnowledgeLibrary,
    base_parent_comm: dict,
    persistent_lib: KnowledgeLibrary,
    preconsolidated_count: int,
    c0_concept_id: str | None = None,
) -> dict:
    """Run a single lifecycle target/counterexample/return stage under identical full v10 + open-grammar tools."""
    t0 = time.monotonic()
    seed = int(lab_kwargs["seed"])
    n_obj = int(lab_kwargs["objects"])
    base_comp_cap = 520000 if n_obj >= 6 else 460000
    meas_cap = 10800 if n_obj >= 6 else 8600
    eval_seed = seed + 1000003

    archive_rec_compute = 0
    if arm_name == "cold_full_tools":
        active_lib = deepcopy(base_parent_lib)
        avail_k = 0
    elif arm_name == "archive_full_tools":
        rec_b = SearchBudget(60000)
        reconstructed, rec_spent = reconstruct_commitments_from_raw_archive(
            persistent_lib.meta_priors, rec_b, tolerance=1e-7
        )
        archive_rec_compute = rec_spent
        active_lib = persistent_lib
        avail_k = max(0, len(reconstructed) - 1)
    elif arm_name == "structured_lifecycle_v10":
        active_lib = persistent_lib
        avail_k = preconsolidated_count
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    cfg = _frozen_config(seed, objects=n_obj, compute_budget=base_comp_cap - archive_rec_compute)
    agent = UnifiedResearchAgent(cfg, measurement_budget=meas_cap, library=active_lib)

    og_compute = 0
    og_obs = 0
    og_meas = 0
    recovered_c0 = False
    c1_rejected_on_return = False
    reorg_type = None
    applied_gen = None
    retrieved_ids: list[str] = []

    with AnonymousObjectLab(**lab_kwargs) as lab:
        # Before running Stage 6 on structured_lifecycle_v10 / archive_full_tools, capture current commitments
        comms_before = consolidate_world_structure_commitments(active_lib.meta_priors)
        res = agent.run(lab)
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        confirmed = bool(res["status"] == "completed")

        app_meta = (res.get("research_program") or {}).get("world_commitment_application") or {}
        if app_meta:
            reorg_type = app_meta.get("reorganization_type")
            applied_gen = app_meta.get("generation")
            retrieved_ids = list(app_meta.get("source_concept_ids") or ())

        # Verify Acceptance #18 & C2 on Stage 6 (return): check whether newer 'left' commitment C_1 was rejected
        # at runtime in apply_world_structure_commitments and Stage-1 'right' commitment C_0 was recovered!
        apply_acts = [a for a in res.get("actions", ()) if a.get("action") == "apply_world_structure_commitments"]
        left_comms = [
            c
            for c in comms_before
            if (c.get("radius_parameters") or [{}])[0].get("direction") == "left"
        ]
        if left_comms and apply_acts:
            last_apply = apply_acts[-1]
            rejected_ids = set(last_apply.get("rejected_inconsistent_commitment_ids") or ())
            matched_ids = set(last_apply.get("consistent_commitments_matched") or ())
            c1_rejected_on_return = all(
                c1["commitment_id"] in rejected_ids and c1["commitment_id"] not in matched_ids
                for c1 in left_comms
            )
        if (
            stage_name == "stage6_return"
            and reorg_type == "equivalence_reorganization"
            and c0_concept_id is not None
            and c0_concept_id in retrieved_ids
        ):
            recovered_c0 = True

        # Equal-tool open-grammar fallback for cold_full_tools (or if unresolved)
        rem_comp = max(0, (base_comp_cap - archive_rec_compute) - res["costs"]["compute_cost"])
        if not confirmed and rem_comp > 5000:
            with AnonymousObjectLab(**lab_kwargs) as sub_lab:
                sub_bundle = _acquire_shared_discovery_bundle(sub_lab, seed=seed + 77)
                og_obs = sub_bundle["observation_cost"]
                og_meas = sub_bundle["measurement_cost"]
                og_out = run_single_factor_dependency_ablation_arm(
                    parent_commitment=base_parent_comm,
                    model=sub_bundle["model"],
                    records=sub_bundle["records"],
                    spec=sub_bundle["spec"],
                    training=sub_bundle["training"],
                    validation=sub_bundle["validation"],
                    compute_budget_limit=rem_comp,
                    use_support_dependency_cue=True,
                    tolerance=1e-7,
                )
                og_compute = sub_bundle["shared_induction_compute"] + og_out["compute_cost"]
                if og_out["supported"] and og_out["best_entry"] is not None:
                    prog = og_out["best_entry"]["candidate"].program
                    ch = ("fine",)
                    fb_val = initial_prediction(sub_bundle["training"], sub_bundle["spec"].risk_quantile)
                    chk_mse = _eval_program_heldout_48(sub_lab, prog, fb_val, seed + 555001)
                    og_obs += 48
                    og_meas += 48 * 3 * n_obj
                    if chk_mse is not None and chk_mse <= 1e-7:
                        confirmed = True
                        res["fallback"] = fb_val

        heldout_mse = (
            float(
                loss(
                    prog,
                    lab.evaluation_rows(ch, seed=eval_seed, length=48, trajectories=1),
                    res["fallback"],
                    0.5,
                )
            )
            if prog is not None
            else None
        )

    tot_comp = res["costs"]["compute_cost"] + archive_rec_compute + og_compute
    tot_obs = res["costs"]["observation_cost"] + og_obs
    tot_meas = res["costs"]["measurement_cost"] + og_meas

    chain_tel = build_action_chain_telemetry(
        arm=arm_name,
        task_id=f"{stage_name}:{seed}",
        knowledge_available_count=avail_k,
        knowledge_retrieved_ids=retrieved_ids,
        entered_candidate_pool=bool(retrieved_ids),
        changed_probe_or_audit=bool(retrieved_ids),
        changed_parent_commitment=bool(retrieved_ids),
        reorganization_type=reorg_type,
        confirmed=confirmed,
        heldout_mse=heldout_mse,
        tolerance=1e-7,
    )

    return {
        "stage": stage_name,
        "objects": n_obj,
        "confirmed": confirmed,
        "selected_channels": list(ch),
        "heldout_fresh48_mse": heldout_mse,
        "passed_1e7": bool(confirmed and heldout_mse is not None and heldout_mse <= 1e-7),
        "reorganization_type": reorg_type,
        "applied_generation": applied_gen,
        "recovered_generation0_commitment": recovered_c0,
        "newer_generation1_empirically_rejected": c1_rejected_on_return,
        "costs": {
            "observation_cost": tot_obs,
            "measurement_cost": tot_meas,
            "agent_compute_cost": res["costs"]["compute_cost"],
            "archive_reconstruction_compute_cost": archive_rec_compute,
            "open_grammar_compute_cost": og_compute,
            "total_compute_cost": tot_comp,
        },
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "action_chain": chain_tel,
    }


def _execute_single_chain(chain_spec: dict, base_lib: KnowledgeLibrary, base_comm: dict) -> dict:
    s1_kw = chain_spec["stage1_source"]
    s1_seed = int(s1_kw["seed"])
    eval_s1 = s1_seed + 1000003
    t_s1 = time.monotonic()

    # ------------------------------------------------------------------
    # Stage 1: Source Formation (N=4, right, donor_positive_gap)
    # ------------------------------------------------------------------
    with AnonymousObjectLab(**s1_kw) as lab1:
        bundle1 = _acquire_shared_discovery_bundle(lab1, seed=s1_seed)
        fb1 = initial_prediction(bundle1["training"], bundle1["spec"].risk_quantile)
        og1 = run_single_factor_dependency_ablation_arm(
            parent_commitment=base_comm,
            model=bundle1["model"],
            records=bundle1["records"],
            spec=bundle1["spec"],
            training=bundle1["training"],
            validation=bundle1["validation"],
            compute_budget_limit=200000,
            use_support_dependency_cue=True,
            tolerance=1e-7,
        )
        best1 = og1["best_entry"]
        prog1 = best1["candidate"].program if best1 else None
        mse1 = _eval_program_heldout_48(lab1, prog1, fb1, eval_s1) if og1["supported"] else None
        confirmed1 = bool(mse1 is not None and mse1 <= 1e-7)

        source_lib = deepcopy(base_lib)
        cons_b1 = SearchBudget(50000)
        c0_comm = None
        c0_prior = None
        if confirmed1 and best1 is not None:
            saver = UnifiedResearchAgent(_frozen_config(s1_seed, objects=4, compute_budget=300000), library=source_lib)
            saver._object_schemas[("fine",)] = deepcopy(best1["resolved_schema"])
            saved_rec = saver.remember_confirmed_selection(
                best1,
                saver._spec(lab1, ("fine",)),
                bundle1["training"],
                bundle1["validation"],
                confirmed=True,
                confirmation_loss=float(mse1),
                confirmation_rows=48,
                selected_channels=("fine",),
                confirmation_ref={
                    "kind": "lifecycle_stage1_fresh48_confirmation",
                    "chain_id": chain_spec["chain_id"],
                    "source_seed": s1_seed,
                    "eval_seed": eval_s1,
                    "confirmation_rows": 48,
                },
            )
            c0_prior = next(
                p for p in source_lib.meta_priors if p["source_concept_id"] == saved_rec["concept_id"]
            )
            comms1 = consolidate_world_structure_commitments(
                [c0_prior], budget=cons_b1, tolerance=1e-7
            )
            c0_comm = comms1[0] if comms1 else None

    s1_wall = round(time.monotonic() - t_s1, 3)
    naive_plugin_mse = (
        _check_direct_source_float_plugin_mse(c0_prior, c0_comm, chain_spec["stage3_transfer"])
        if c0_prior and c0_comm
        else None
    )

    # ------------------------------------------------------------------
    # Stage 2: Freeze Proposer, Policy, and Learned Object BEFORE Target Exposure
    # ------------------------------------------------------------------
    freeze_manifest = {
        "frozen_before_target_exposure": True,
        "proposer_catalog_sha256": _sha256_obj(build_bounded_open_grammar_patch_catalog()),
        "policy_config_sha256": _sha256_obj(asdict(_frozen_config(0, objects=5, compute_budget=320000))),
        "source_commitment_sha256": _sha256_obj(c0_comm) if c0_comm else None,
        "source_commitment_id": c0_comm["commitment_id"] if c0_comm else None,
        "source_concept_id": c0_prior["source_concept_id"] if c0_prior else None,
    }

    stage1_summary = {
        "confirmed": confirmed1,
        "fresh48_mse": mse1,
        "candidates_attempted_denominator": og1["candidates_attempted_denominator"],
        "failed_candidates_count": og1["failed_candidates_count"],
        "research_method_metadata": deepcopy(c0_prior["research_method"]) if c0_prior else None,
        "direct_source_float_plugin_on_stage3_mse": naive_plugin_mse,
        "direct_source_float_plugin_fails_1e7": bool(naive_plugin_mse is not None and naive_plugin_mse > 1e-3),
        "costs": {
            "observation_cost": bundle1["observation_cost"] + 48,
            "measurement_cost": bundle1["measurement_cost"] + 48 * 12,
            "discovery_compute_cost": bundle1["shared_induction_compute"] + og1["compute_cost"],
            "consolidation_compute_cost": cons_b1.spent,
        },
        "wall_seconds": s1_wall,
    }

    # ------------------------------------------------------------------
    # Stages 3, 4-5, and 6 Across All 3 Equal-Tool Arms
    # ------------------------------------------------------------------
    c0_cid = c0_prior["source_concept_id"] if c0_prior else None
    arms_out = {}
    for arm_name in ("cold_full_tools", "archive_full_tools", "structured_lifecycle_v10"):
        pers_lib = deepcopy(base_lib) if arm_name == "cold_full_tools" else deepcopy(source_lib)
        pre_cnt = 1 if (arm_name == "structured_lifecycle_v10" and c0_comm is not None) else 0

        s3 = _run_stage_episode_for_arm(
            arm_name=arm_name,
            stage_name="stage3_transfer",
            lab_kwargs=chain_spec["stage3_transfer"],
            base_parent_lib=base_lib,
            base_parent_comm=base_comm,
            persistent_lib=pers_lib,
            preconsolidated_count=pre_cnt,
            c0_concept_id=c0_cid,
        )
        s45 = _run_stage_episode_for_arm(
            arm_name=arm_name,
            stage_name="stage4_5_counterexample_revision",
            lab_kwargs=chain_spec["stage4_5_counterexample_revision"],
            base_parent_lib=base_lib,
            base_parent_comm=base_comm,
            persistent_lib=pers_lib,
            preconsolidated_count=pre_cnt,
            c0_concept_id=c0_cid,
        )
        # Check C_0 status after Stage 4-5 counterexample!
        c0_status_after_counterexample = None
        c0_counterexample_count = 0
        if c0_prior is not None and arm_name != "cold_full_tools":
            matched_p = next(
                (p for p in pers_lib.meta_priors if p["source_concept_id"] == c0_prior["source_concept_id"]),
                None,
            )
            if matched_p is not None:
                c0_status_after_counterexample = matched_p.get("status")
                c0_counterexample_count = len(matched_p.get("counterexamples", ()))

        s6 = _run_stage_episode_for_arm(
            arm_name=arm_name,
            stage_name="stage6_return",
            lab_kwargs=chain_spec["stage6_return"],
            base_parent_lib=base_lib,
            base_parent_comm=base_comm,
            persistent_lib=pers_lib,
            preconsolidated_count=pre_cnt + (1 if s45["passed_1e7"] else 0),
            c0_concept_id=c0_cid,
        )

        src_obs = stage1_summary["costs"]["observation_cost"] if arm_name != "cold_full_tools" else 0
        src_meas = stage1_summary["costs"]["measurement_cost"] if arm_name != "cold_full_tools" else 0
        src_comp = (
            stage1_summary["costs"]["discovery_compute_cost"]
            + (
                stage1_summary["costs"]["consolidation_compute_cost"]
                if arm_name == "structured_lifecycle_v10"
                else 0
            )
        )
        if arm_name == "cold_full_tools":
            # Cold arm runs Stage 1 independently as well with zero carryover
            src_obs = stage1_summary["costs"]["observation_cost"]
            src_meas = stage1_summary["costs"]["measurement_cost"]
            src_comp = stage1_summary["costs"]["discovery_compute_cost"]

        post_freeze_stages = [s3, s45, s6]
        arms_out[arm_name] = {
            "stage3_transfer": s3,
            "stage4_5_counterexample_revision": {
                **s45,
                "c0_status_after_counterexample": c0_status_after_counterexample,
                "c0_recorded_counterexamples": c0_counterexample_count,
            },
            "stage6_return": s6,
            "all_post_freeze_stages_passed": all(st["passed_1e7"] for st in post_freeze_stages),
            "cost_ledger_separated": {
                "stage1_source_compute_cost": src_comp,
                "stage3_transfer_compute_cost": s3["costs"]["total_compute_cost"],
                "stage4_5_revision_compute_cost": s45["costs"]["total_compute_cost"],
                "stage6_return_compute_cost": s6["costs"]["total_compute_cost"],
                "post_freeze_target_compute_sum": sum(st["costs"]["total_compute_cost"] for st in post_freeze_stages),
                "cumulative_sequence_compute_cost": src_comp
                + sum(st["costs"]["total_compute_cost"] for st in post_freeze_stages),
                "post_freeze_observation_sum": sum(st["costs"]["observation_cost"] for st in post_freeze_stages),
                "cumulative_sequence_observation_cost": src_obs
                + sum(st["costs"]["observation_cost"] for st in post_freeze_stages),
                "post_freeze_measurement_sum": sum(st["costs"]["measurement_cost"] for st in post_freeze_stages),
            },
        }

    return {
        "chain_id": chain_spec["chain_id"],
        "chain_role": chain_spec["chain_role"],
        "stage1_source_formation": stage1_summary,
        "stage2_freeze_manifest": freeze_manifest,
        "arms": arms_out,
    }


def main() -> int:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_lib, base_comm, base_info = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )

    all_chains = []
    for c_spec in _preregistered_chain_specs():
        res_c = _execute_single_chain(c_spec, base_lib, base_comm)
        all_chains.append(res_c)
        st_v10 = res_c["arms"]["structured_lifecycle_v10"]
        print(
            f"[{c_spec['chain_id']} ({c_spec['chain_role']})] "
            f"S1={res_c['stage1_source_formation']['confirmed']} | "
            f"S3={st_v10['stage3_transfer']['passed_1e7']} | "
            f"S4-5={st_v10['stage4_5_counterexample_revision']['passed_1e7']} "
            f"(C0_status={st_v10['stage4_5_counterexample_revision']['c0_status_after_counterexample']}) | "
            f"S6_return={st_v10['stage6_return']['passed_1e7']} "
            f"(recovered_C0={st_v10['stage6_return']['recovered_generation0_commitment']}, "
            f"rejected_C1={st_v10['stage6_return']['newer_generation1_empirically_rejected']})",
            flush=True,
        )

    prospective = [c for c in all_chains if c["chain_role"] == "frozen_prospective"]
    prospective_summary = {}
    for arm_name in ("cold_full_tools", "archive_full_tools", "structured_lifecycle_v10"):
        rows = [c["arms"][arm_name] for c in prospective]
        prospective_summary[arm_name] = {
            "independent_physical_source_chains": len(rows),
            "stage3_transfer_passes": sum(1 for r in rows if r["stage3_transfer"]["passed_1e7"]),
            "stage4_5_revision_passes": sum(
                1 for r in rows if r["stage4_5_counterexample_revision"]["passed_1e7"]
            ),
            "stage6_return_passes": sum(1 for r in rows if r["stage6_return"]["passed_1e7"]),
            "full_chain_passes": sum(1 for r in rows if r["all_post_freeze_stages_passed"]),
            "stage6_recovered_generation0_count": sum(
                1 for r in rows if r["stage6_return"]["recovered_generation0_commitment"]
            ),
            "stage6_rejected_incompatible_gen1_count": sum(
                1 for r in rows if r["stage6_return"]["newer_generation1_empirically_rejected"]
            ),
            "mean_post_freeze_target_compute": float(
                mean(r["cost_ledger_separated"]["post_freeze_target_compute_sum"] for r in rows)
            ),
            "mean_cumulative_sequence_compute": float(
                mean(r["cost_ledger_separated"]["cumulative_sequence_compute_cost"] for r in rows)
            ),
            "mean_post_freeze_observations": float(
                mean(r["cost_ledger_separated"]["post_freeze_observation_sum"] for r in rows)
            ),
            "mean_cumulative_sequence_observations": float(
                mean(r["cost_ledger_separated"]["cumulative_sequence_observation_cost"] for r in rows)
            ),
            "model_calls": 0,
        }

    receipt = {
        "experiment_id": "v10_2_frozen_v4_lifecycle_20261116",
        "generated_at": _utc_now(),
        "audited_anchor_commit": "a66cd82b22f444044dbda98f43782c96f833a4f0",
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "common_initial_background_formation": base_info,
        "protocol_notes": {
            "sample_unit": "independent_physical_source_chain",
            "development_chains_count": 1,
            "frozen_prospective_chains_count": len(prospective),
            "seed_selection_rule": "contiguous_preregistered_711000_to_714000_zero_cherry_picking",
        },
        "frozen_prospective_summary_by_arm": prospective_summary,
        "development_qualification_chain": next(
            c for c in all_chains if c["chain_role"] == "development_qualification"
        ),
        "prospective_chains": prospective,
    }
    out_path = OUT_DIR / "frozen_v4_lifecycle_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Frozen Prospective Summary:", json.dumps(prospective_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

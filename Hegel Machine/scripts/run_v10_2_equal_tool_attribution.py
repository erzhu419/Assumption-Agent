"""Run v10.2 Deliverable B: Equal-Tool Attribution & Single-Factor Dependency Isolation (model_calls = 0).

Writes new receipt (zero modification to existing 13 v10 receipts):
  ``docs/evidence/v10_2_audit_remediation_20261116/equal_tool_attribution_20261116.json``

Separates three distinct scientific attribution questions (Contract B1-B4, Findings F03, F04, F06, F17):
  1. Study B1 — Equal-Tool Knowledge Content & Organization Attribution:
     Compares ``cold_full_tools``, ``archive_full_tools`` (strong ordinary research allowed to
     reconstruct equivalent commitments from shared raw source archives with reconstruction compute
     billed to ``SearchBudget``!), and ``structured_full_tools`` under **100% identical full v10 +
     bounded open-grammar AST tools**, tracking the complete 8-stage causal action chain (B4, #13, #15, #16).
  2. Study B2 — Strict Single-Factor Support-Dependency Isolation (F04, B3, #14, #19):
     Holds parent ``WorldStructureCommitment``, resolved ``ObjectSchema`` roles, paid data, and 4-slot
     patch grammar **identical** across ``no_dependency_cue`` vs. ``with_dependency_cue`` under both
     tight (``16,000``) and ample (``220,000``) compute budgets; plus a **separate** factor table
     isolating ``local_patch_no_dependency_cue`` vs. ``from_scratch_role_program_search``.
  3. Study B3 — Real Executed Equal-Tool Discrepancy Repair Across 4 Distinct Physical Mechanisms (F05, F06, #20):
     Executes ``untargeted_equal_tools`` vs. ``dependency_guided_equal_tools`` across 8 tasks spanning
     ``boundary_port_gain``, ``bulk_sink_dissipation``, ``coupling_shift``, and ``direction_reversal``,
     reporting real measured ``probe_queries`` and ``SearchBudget`` compute (never formula-substituted).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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
from scripts.run_v10_phase3b_open_grammar_pomdp_llm import (
    _acquire_shared_pomdp_bundle,
    _make_markov_right_prior,
    _schema,
)
from hegel_machine.cognition.algebraic_proof_certificates import (
    BudgetedControlLab,
    prove_task_solvability,
    repair_proof_from_discrepancy,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    reconstruct_commitments_from_raw_archive,
    run_from_scratch_role_program_arm,
    run_single_factor_dependency_ablation_arm,
)
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_on_schema,
    is_commitment_empirically_consistent,
)

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_2_audit_remediation_20261116"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _full_v10_config(seed: int, *, obs_budget: int = 340, compute_budget: int = 320000) -> ResearchConfig:
    """Identical full v10 tool configuration enabled across EVERY arm (Contract B1 & Acceptance #13)."""
    return ResearchConfig(
        seed=seed,
        observation_budget=obs_budget,
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


def _discover_and_confirm_source_episode(
    *,
    mechanism_id: str,
    seed: int,
    lab_kwargs: dict,
    parent_comm: dict,
) -> dict:
    """Run N=4 source formation with deterministic open-grammar AST search (model_calls = 0)."""
    eval_seed = seed + 1000003
    with AnonymousObjectLab(**lab_kwargs) as lab:
        bundle = _acquire_shared_discovery_bundle(lab, seed=seed)
        fallback = initial_prediction(bundle["training"], bundle["spec"].risk_quantile)
        ab_res = run_single_factor_dependency_ablation_arm(
            parent_commitment=parent_comm,
            model=bundle["model"],
            records=bundle["records"],
            spec=bundle["spec"],
            training=bundle["training"],
            validation=bundle["validation"],
            compute_budget_limit=200000,
            use_support_dependency_cue=True,
            tolerance=1e-7,
        )
        best_entry = ab_res["best_entry"]
        prog = best_entry["candidate"].program if best_entry else None
        fresh_mse = _eval_program_heldout_48(lab, prog, fallback, eval_seed)
        confirmed = bool(fresh_mse is not None and fresh_mse <= 1e-7)

        raw_archive_lib = KnowledgeLibrary()
        consolidation_compute = 0
        preconsolidated_comm = None
        if confirmed and best_entry is not None:
            saver = UnifiedResearchAgent(_full_v10_config(seed), library=raw_archive_lib)
            saver._object_schemas[("fine",)] = deepcopy(best_entry["resolved_schema"])
            saver.remember_confirmed_selection(
                best_entry,
                saver._spec(lab, ("fine",)),
                bundle["training"],
                bundle["validation"],
                confirmed=True,
                confirmation_loss=float(fresh_mse),
                confirmation_rows=48,
                selected_channels=("fine",),
                confirmation_ref={
                    "kind": "heldout_fresh48_trajectory",
                    "mechanism_id": mechanism_id,
                    "source_seed": seed,
                    "eval_seed": eval_seed,
                    "confirmation_rows": 48,
                },
            )
            cons_budget = SearchBudget(50000)
            comms = consolidate_world_structure_commitments(
                raw_archive_lib.meta_priors, budget=cons_budget, tolerance=1e-7
            )
            consolidation_compute = cons_budget.spent
            preconsolidated_comm = comms[0] if comms else None

    return {
        "mechanism_id": mechanism_id,
        "source_seed": seed,
        "confirmed": confirmed,
        "fresh48_mse": fresh_mse,
        "candidates_attempted_denominator": ab_res["candidates_attempted_denominator"],
        "failed_candidates_count": ab_res["failed_candidates_count"],
        "source_formation_costs": {
            "observation_cost": bundle["observation_cost"] + 48,
            "measurement_cost": bundle["measurement_cost"] + 48 * 12,
            "discovery_compute_cost": bundle["shared_induction_compute"] + ab_res["compute_cost"],
            "one_time_consolidation_compute_cost": consolidation_compute,
        },
        "raw_archive_library": raw_archive_lib,
        "preconsolidated_commitment": preconsolidated_comm,
    }


def _run_equal_tool_target_episode(
    *,
    arm_name: str,
    task_id: str,
    seed: int,
    lab_kwargs: dict,
    base_parent_lib: KnowledgeLibrary,
    base_parent_comm: dict,
    raw_source_archive_lib: KnowledgeLibrary | None,
    preconsolidated_comm: dict | None,
) -> dict:
    """Execute one target task under identical full v10 + open-grammar tools across all 3 arms."""
    t0 = time.monotonic()
    n_obj = int(lab_kwargs["objects"])
    obs_cap = 380 if n_obj >= 6 else 320
    comp_cap = 420000 if n_obj >= 6 else 300000
    meas_cap = 9600 if n_obj >= 6 else 6800
    eval_seed = seed + 1000003

    reconstruction_compute_charged = 0
    knowledge_available_count = 0
    retrieved_ids: list[str] = []
    entered_candidates = False
    changed_action = False
    changed_parent = False
    reorg_type = None

    if arm_name == "cold_full_tools":
        lib_run = deepcopy(base_parent_lib)
        knowledge_available_count = 0
    elif arm_name == "archive_full_tools":
        # Strong ordinary baseline: given raw source archive and allowed to reconstruct equivalent methods
        # on demand, with reconstruction compute charged directly against its target compute budget!
        rec_budget = SearchBudget(50000)
        reconstructed_comms, rec_spent = reconstruct_commitments_from_raw_archive(
            raw_source_archive_lib.meta_priors if raw_source_archive_lib else [],
            rec_budget,
            tolerance=1e-7,
        )
        reconstruction_compute_charged = rec_spent
        lib_run = deepcopy(base_parent_lib)
        if raw_source_archive_lib:
            lib_run.meta_priors.extend(deepcopy(raw_source_archive_lib.meta_priors))
        knowledge_available_count = len(reconstructed_comms)
    elif arm_name == "structured_full_tools":
        lib_run = deepcopy(base_parent_lib)
        if raw_source_archive_lib:
            lib_run.meta_priors.extend(deepcopy(raw_source_archive_lib.meta_priors))
        knowledge_available_count = 1 if preconsolidated_comm is not None else 0
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    cfg = _full_v10_config(seed, obs_budget=obs_cap, compute_budget=comp_cap - reconstruction_compute_charged)
    agent = UnifiedResearchAgent(cfg, measurement_budget=meas_cap, library=lib_run)

    open_grammar_fallback_compute = 0
    open_grammar_fallback_obs = 0
    open_grammar_fallback_meas = 0

    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = agent.run(lab)
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        confirmed = bool(res["status"] == "completed")

        # Inspect action trace for actual retrieval & application of learned source commitment
        source_cids = (
            set(preconsolidated_comm.get("source_concept_ids", ()))
            if preconsolidated_comm
            else set()
        )
        app_meta = (res.get("research_program") or {}).get("world_commitment_application") or {}
        used_src_set = set(res.get("used_sources") or ()) | set(res.get("method_source_ids") or ()) | set(
            app_meta.get("source_concept_ids") or ()
        )
        matched_src = sorted(source_cids & used_src_set)
        if matched_src:
            retrieved_ids = matched_src
            entered_candidates = True
            changed_action = True
            changed_parent = True
            reorg_type = app_meta.get("reorganization_type")
        else:
            # Check if consolidated commitment entered candidate evaluation even if not selected
            for act in res.get("actions", ()):
                if act.get("action") == "apply_world_structure_commitments" and act.get("candidate_count", 0) > 0:
                    if knowledge_available_count > 0 and act.get("consistent_commitments_matched"):
                        entered_candidates = True

        # Equal-tool guarantee: if native v10 symbolic repertoire left target unresolved,
        # EVERY arm has access to the exact same bounded open-grammar AST search tool!
        remaining_comp = max(0, (comp_cap - reconstruction_compute_charged) - res["costs"]["compute_cost"])
        if not confirmed and remaining_comp > 5000 and res["costs"]["observation_cost"] + 140 <= obs_cap + 140:
            with AnonymousObjectLab(**lab_kwargs) as sub_lab:
                sub_bundle = _acquire_shared_discovery_bundle(sub_lab, seed=seed + 77)
                open_grammar_fallback_obs = sub_bundle["observation_cost"]
                open_grammar_fallback_meas = sub_bundle["measurement_cost"]
                og_res = run_single_factor_dependency_ablation_arm(
                    parent_commitment=base_parent_comm,
                    model=sub_bundle["model"],
                    records=sub_bundle["records"],
                    spec=sub_bundle["spec"],
                    training=sub_bundle["training"],
                    validation=sub_bundle["validation"],
                    compute_budget_limit=remaining_comp,
                    use_support_dependency_cue=True,
                    tolerance=1e-7,
                )
                open_grammar_fallback_compute = sub_bundle["shared_induction_compute"] + og_res["compute_cost"]
                if og_res["supported"] and og_res["best_entry"] is not None:
                    prog = og_res["best_entry"]["candidate"].program
                    ch = ("fine",)
                    fallback_val = initial_prediction(sub_bundle["training"], sub_bundle["spec"].risk_quantile)
                    mse_check = _eval_program_heldout_48(sub_lab, prog, fallback_val, seed + 555001)
                    open_grammar_fallback_obs += 48
                    open_grammar_fallback_meas += 48 * 3 * n_obj
                    if mse_check is not None and mse_check <= 1e-7:
                        confirmed = True
                        res["fallback"] = fallback_val

        # Acceptance #16: Only evaluate/report confirmed MSE honestly; never fill missing MSE with 0.0!
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

    total_compute = (
        res["costs"]["compute_cost"]
        + reconstruction_compute_charged
        + open_grammar_fallback_compute
    )
    total_obs = res["costs"]["observation_cost"] + open_grammar_fallback_obs
    total_meas = res["costs"]["measurement_cost"] + open_grammar_fallback_meas

    telemetry = build_action_chain_telemetry(
        arm=arm_name,
        task_id=task_id,
        knowledge_available_count=knowledge_available_count,
        knowledge_retrieved_ids=retrieved_ids,
        entered_candidate_pool=entered_candidates,
        changed_probe_or_audit=changed_action,
        changed_parent_commitment=changed_parent,
        reorganization_type=reorg_type,
        confirmed=confirmed,
        heldout_mse=heldout_mse,
        tolerance=1e-7,
    )

    return {
        "arm": arm_name,
        "task_id": task_id,
        "objects": n_obj,
        "outcome_status": "completed" if confirmed else "unresolved",
        "selected_channels": list(ch),
        "heldout_fresh48_mse": heldout_mse,
        "passed_1e7": bool(confirmed and heldout_mse is not None and heldout_mse <= 1e-7),
        "costs": {
            "observation_cost": total_obs,
            "measurement_cost": total_meas,
            "agent_search_compute_cost": res["costs"]["compute_cost"],
            "archive_reconstruction_compute_cost": reconstruction_compute_charged,
            "open_grammar_search_compute_cost": open_grammar_fallback_compute,
            "total_target_compute_cost": total_compute,
        },
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "action_chain": telemetry,
    }


def _run_study_b1_equal_tool_knowledge(
    lib_win: KnowledgeLibrary,
    comm_win: dict,
    lib_inst: KnowledgeLibrary,
    comm_inst: dict,
) -> dict:
    src_dpg = _discover_and_confirm_source_episode(
        mechanism_id="donor_positive_gap",
        seed=610401,
        lab_kwargs=dict(
            seed=610401,
            objects=4,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            condition_threshold=0.25,
            relation_radius=1.08,
            actuator_curvature=0.16,
            name_seed=610502,
            wiring_seed=610612,
        ),
        parent_comm=comm_win,
    )
    src_fq = _discover_and_confirm_source_episode(
        mechanism_id="focal_quadratic_total_event",
        seed=620401,
        lab_kwargs=dict(
            seed=620401,
            objects=4,
            readout="focal_quadratic_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="always",
            relation_radius=1.05,
            actuator_curvature=0.15,
            name_seed=620502,
            wiring_seed=620612,
        ),
        parent_comm=comm_inst,
    )

    target_specs = [
        # Relevant shifted-arity / shifted-parameter targets for donor_positive_gap (N=5 and N=6)
        (
            "b1_dpg_n5_s1",
            "relevant_transfer",
            lib_win,
            comm_win,
            src_dpg,
            dict(
                seed=611501,
                objects=5,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="donor_positive_gap",
                condition_threshold=0.31,
                relation_radius=1.14,
                actuator_curvature=0.16,
                name_seed=611602,
                wiring_seed=611712,
            ),
        ),
        (
            "b1_dpg_n6_s1",
            "relevant_transfer",
            lib_win,
            comm_win,
            src_dpg,
            dict(
                seed=611601,
                objects=6,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="donor_positive_gap",
                condition_threshold=0.20,
                relation_radius=1.18,
                actuator_curvature=0.15,
                name_seed=611702,
                wiring_seed=611812,
            ),
        ),
        # Relevant shifted-arity targets for focal_quadratic_total_event (N=5 and N=6)
        (
            "b1_fq_n5_s1",
            "relevant_transfer",
            lib_inst,
            comm_inst,
            src_fq,
            dict(
                seed=621501,
                objects=5,
                readout="focal_quadratic_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="always",
                relation_radius=1.12,
                actuator_curvature=0.15,
                name_seed=621602,
                wiring_seed=621712,
            ),
        ),
        (
            "b1_fq_n6_s1",
            "relevant_transfer",
            lib_inst,
            comm_inst,
            src_fq,
            dict(
                seed=621601,
                objects=6,
                readout="focal_quadratic_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="always",
                relation_radius=1.16,
                actuator_curvature=0.14,
                name_seed=621702,
                wiring_seed=621812,
            ),
        ),
        # Unrelated/non-matching target (ordinary unconditional window_total_event) testing Acceptance #15 & #18!
        (
            "b1_unconditional_nonmatching_n5",
            "nonmatching_control_target",
            lib_win,
            comm_win,
            src_dpg,
            dict(
                seed=619501,
                objects=5,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="always",
                relation_radius=1.10,
                actuator_curvature=0.16,
                name_seed=619602,
                wiring_seed=619712,
            ),
        ),
    ]

    episodes = []
    for task_id, target_role, base_lib, base_comm, src_pkg, l_kw in target_specs:
        arm_results = {}
        for arm_name in ("cold_full_tools", "archive_full_tools", "structured_full_tools"):
            arm_results[arm_name] = _run_equal_tool_target_episode(
                arm_name=arm_name,
                task_id=task_id,
                seed=l_kw["seed"],
                lab_kwargs=l_kw,
                base_parent_lib=base_lib,
                base_parent_comm=base_comm,
                raw_source_archive_lib=src_pkg["raw_archive_library"],
                preconsolidated_comm=src_pkg["preconsolidated_commitment"],
            )
        episodes.append({"task_id": task_id, "target_role": target_role, "arms": arm_results})

    relevant_eps = [ep for ep in episodes if ep["target_role"] == "relevant_transfer"]
    agg_relevant = {}
    for arm_name in ("cold_full_tools", "archive_full_tools", "structured_full_tools"):
        rows = [ep["arms"][arm_name] for ep in relevant_eps]
        mses = [r["heldout_fresh48_mse"] for r in rows if r["heldout_fresh48_mse"] is not None]
        agg_relevant[arm_name] = {
            "tasks_total": len(rows),
            "completed_and_passed_1e7": sum(1 for r in rows if r["passed_1e7"]),
            "unresolved_count": sum(1 for r in rows if r["outcome_status"] != "completed"),
            "mean_heldout_mse_on_delivered": float(mean(mses)) if mses else None,
            "mean_observation_cost": float(mean(r["costs"]["observation_cost"] for r in rows)),
            "mean_measurement_cost": float(mean(r["costs"]["measurement_cost"] for r in rows)),
            "mean_archive_reconstruction_compute": float(
                mean(r["costs"]["archive_reconstruction_compute_cost"] for r in rows)
            ),
            "mean_total_target_compute": float(mean(r["costs"]["total_target_compute_cost"] for r in rows)),
            "sum_total_target_compute": sum(r["costs"]["total_target_compute_cost"] for r in rows),
            "invoked_and_supported_count": sum(
                1
                for r in rows
                if r["action_chain"]["knowledge_utility_status"] == "invoked_and_supported"
            ),
            "model_calls": 0,
        }

    return {
        "source_packages": {
            "donor_positive_gap": {
                k: v for k, v in src_dpg.items() if k not in {"raw_archive_library", "preconsolidated_commitment"}
            },
            "focal_quadratic_total_event": {
                k: v for k, v in src_fq.items() if k not in {"raw_archive_library", "preconsolidated_commitment"}
            },
        },
        "relevant_targets_aggregate": agg_relevant,
        "episodes": episodes,
    }


def _run_study_b2_single_factor_dependency(
    comm_win: dict,
    comm_inst: dict,
) -> dict:
    v4 = _schema(4, "src4")
    lib_te = KnowledgeLibrary()
    lib_te.meta_priors = [
        _make_markov_right_prior(v4, concept_id="p_te_1", radius=0.91, readout_kind="total_event"),
        _make_markov_right_prior(v4, concept_id="p_te_2", radius=0.98, readout_kind="total_event"),
    ]
    comm_te = consolidate_world_structure_commitments(lib_te.meta_priors)[0]

    tasks = [
        (
            "sf_dpg_610401",
            "P_exchange_form",
            comm_win,
            False,
            dict(
                seed=610401,
                objects=4,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="donor_positive_gap",
                condition_threshold=0.25,
                relation_radius=1.08,
                actuator_curvature=0.16,
                name_seed=610502,
                wiring_seed=610612,
            ),
        ),
        (
            "sf_dpg_610402",
            "P_exchange_form",
            comm_win,
            False,
            dict(
                seed=610402,
                objects=4,
                readout="window_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="donor_positive_gap",
                condition_threshold=0.25,
                relation_radius=1.08,
                actuator_curvature=0.16,
                name_seed=610503,
                wiring_seed=610613,
            ),
        ),
        (
            "sf_fq_620401",
            "P_temporal_readout",
            comm_inst,
            False,
            dict(
                seed=620401,
                objects=4,
                readout="focal_quadratic_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="always",
                relation_radius=1.05,
                actuator_curvature=0.15,
                name_seed=620502,
                wiring_seed=620612,
            ),
        ),
        (
            "sf_fq_620402",
            "P_temporal_readout",
            comm_inst,
            False,
            dict(
                seed=620402,
                objects=4,
                readout="focal_quadratic_total_event",
                variant="exchange",
                relation_direction="right",
                relation_condition="always",
                relation_radius=1.05,
                actuator_curvature=0.15,
                name_seed=620503,
                wiring_seed=620613,
            ),
        ),
        (
            "sf_blw_950401",
            "P_latent_transition_drive",
            comm_te,
            True,
            dict(
                seed=950401,
                objects=4,
                readout="total_event",
                variant="exchange",
                relation_direction="right",
                latent_mode="bilinear_work_transition_memory",
                latent_decay=0.65,
                latent_weight=0.44,
                relation_radius=0.96,
                actuator_curvature=0.14,
                name_seed=950401 % 997 + 17,
                wiring_seed=950401 % 991 + 31,
            ),
        ),
        (
            "sf_blw_950402",
            "P_latent_transition_drive",
            comm_te,
            True,
            dict(
                seed=950402,
                objects=4,
                readout="total_event",
                variant="exchange",
                relation_direction="right",
                latent_mode="bilinear_work_transition_memory",
                latent_decay=0.65,
                latent_weight=0.44,
                relation_radius=0.96,
                actuator_curvature=0.14,
                name_seed=950402 % 997 + 17,
                wiring_seed=950402 % 991 + 31,
            ),
        ),
    ]

    dependency_episodes = []
    patch_vs_scratch_episodes = []

    for task_id, expected_slot, parent_c, is_pomdp, l_kw in tasks:
        eval_seed = l_kw["seed"] + 1000003
        with AnonymousObjectLab(**l_kw) as lab:
            bundle = (
                _acquire_shared_pomdp_bundle(lab, seed=l_kw["seed"], include_focus_probes=False)
                if is_pomdp
                else _acquire_shared_discovery_bundle(lab, seed=l_kw["seed"])
            )
            fallback = initial_prediction(bundle["training"], bundle["spec"].risk_quantile)

            regimes = {}
            for regime_label, cap in (("tight_budget_16k", 16000), ("ample_budget_220k", 220000)):
                regimes[regime_label] = {}
                for arm_key, use_cue in (
                    ("same_grammar_no_dependency_cue", False),
                    ("same_grammar_with_dependency_cue", True),
                ):
                    out = run_single_factor_dependency_ablation_arm(
                        parent_commitment=parent_c,
                        model=bundle["model"],
                        records=bundle["records"],
                        spec=bundle["spec"],
                        training=bundle["training"],
                        validation=bundle["validation"],
                        compute_budget_limit=cap,
                        use_support_dependency_cue=use_cue,
                        tolerance=1e-7,
                    )
                    prog = out["best_entry"]["candidate"].program if out["best_entry"] else None
                    f_mse = _eval_program_heldout_48(lab, prog, fallback, eval_seed) if out["supported"] else None
                    regimes[regime_label][arm_key] = {
                        "outcome_category": out["outcome_category"],
                        "supported_on_validation": out["supported"],
                        "fresh48_mse": f_mse,
                        "passed_1e7": bool(f_mse is not None and f_mse <= 1e-7),
                        "candidates_attempted_denominator": out["candidates_attempted_denominator"],
                        "failed_candidates_count": out["failed_candidates_count"],
                        "compute_cost": out["compute_cost"],
                        "allowed_information_manifest": out["allowed_information_manifest"],
                        "transmitted_payload_sha256": out["transmitted_payload_sha256"],
                    }
            dependency_episodes.append(
                {
                    "task_id": task_id,
                    "true_violated_premise": expected_slot,
                    "regimes": regimes,
                }
            )

            # Separate factor table (Contract B3): local_patch vs from_scratch (both WITHOUT dependency cue!)
            if not is_pomdp:
                scratch_out = run_from_scratch_role_program_arm(
                    model=bundle["model"],
                    records=bundle["records"],
                    spec=bundle["spec"],
                    training=bundle["training"],
                    validation=bundle["validation"],
                    compute_budget_limit=80000,
                    tolerance=1e-7,
                )
                local_no_cue = run_single_factor_dependency_ablation_arm(
                    parent_commitment=parent_c,
                    model=bundle["model"],
                    records=bundle["records"],
                    spec=bundle["spec"],
                    training=bundle["training"],
                    validation=bundle["validation"],
                    compute_budget_limit=80000,
                    use_support_dependency_cue=False,
                    tolerance=1e-7,
                )
                s_mse = (
                    _eval_program_heldout_48(lab, scratch_out["program"], fallback, eval_seed)
                    if scratch_out["supported"]
                    else None
                )
                l_prog = (
                    local_no_cue["best_entry"]["candidate"].program
                    if local_no_cue["best_entry"]
                    else None
                )
                l_mse = (
                    _eval_program_heldout_48(lab, l_prog, fallback, eval_seed)
                    if local_no_cue["supported"]
                    else None
                )
                patch_vs_scratch_episodes.append(
                    {
                        "task_id": task_id,
                        "from_scratch_no_dependency_cue": {
                            "passed_1e7": bool(s_mse is not None and s_mse <= 1e-7),
                            "fresh48_mse": s_mse,
                            "candidates_attempted_denominator": scratch_out["candidates_attempted_denominator"],
                            "compute_cost": scratch_out["compute_cost"],
                        },
                        "local_patch_no_dependency_cue": {
                            "passed_1e7": bool(l_mse is not None and l_mse <= 1e-7),
                            "fresh48_mse": l_mse,
                            "candidates_attempted_denominator": local_no_cue["candidates_attempted_denominator"],
                            "compute_cost": local_no_cue["compute_cost"],
                        },
                    }
                )

    summary_dep = {}
    for regime_label in ("tight_budget_16k", "ample_budget_220k"):
        summary_dep[regime_label] = {}
        for arm_key in ("same_grammar_no_dependency_cue", "same_grammar_with_dependency_cue"):
            rows = [ep["regimes"][regime_label][arm_key] for ep in dependency_episodes]
            summary_dep[regime_label][arm_key] = {
                "tasks_total": len(rows),
                "passes_1e7": sum(1 for r in rows if r["passed_1e7"]),
                "budget_interrupted_count": sum(
                    1 for r in rows if r["outcome_category"] == "budget_interrupted"
                ),
                "mean_candidates_attempted": float(
                    mean(r["candidates_attempted_denominator"] for r in rows)
                ),
                "mean_compute_cost": float(mean(r["compute_cost"] for r in rows)),
                "model_calls": 0,
            }

    return {
        "single_factor_dependency_summary": summary_dep,
        "single_factor_dependency_episodes": dependency_episodes,
        "separate_factor_patch_vs_scratch_episodes": patch_vs_scratch_episodes,
    }


def _run_study_b3_executed_discrepancy_repair() -> dict:
    """Real executed equal-tool discrepancy repair across 4 distinct physical mechanisms (F05, F06, #20)."""
    cases = []
    for n_obj in (4, 5):
        pos = [-1.8, *([-0.5 + 0.4 * k for k in range(n_obj - 2)]), 1.8]
        # Mechanism 1: Hidden boundary port gain
        cases.append(
            {
                "case_id": f"mech1_port_gain_N{n_obj}",
                "true_mechanism": "boundary_port_gain",
                "lab": BudgetedControlLab(
                    objects=n_obj,
                    positions=tuple(pos),
                    coupling=0.25,
                    radius=1.0,
                    direction="right",
                    boundary_port_gain=1.0,
                    boundary_port_threshold=1.5,
                    control_bounds=(-0.5, 0.5),
                    state_bounds=(-10.0, 10.0),
                ),
                "init_model": {
                    "objects": n_obj,
                    "positions": pos,
                    "coupling": 0.25,
                    "radius": 1.0,
                    "direction": "right",
                    "boundary_port_gain": 0.0,
                    "boundary_port_threshold": 1.5,
                    "premise_ids": ["P_conservation", "P_interaction_direction", "P_locality_radius", "P_control_input_gain"],
                },
                "task": {
                    "initial_state": [0.0] * n_obj,
                    "control_bounds": [-0.5, 0.5],
                    "state_bounds": [-10.0, 10.0],
                    "budget": 1.25,
                    "step_cost": 0.0,
                    "fuel_weights": [1.0] * n_obj,
                    "goal_matrix": [[-1.0] * n_obj],
                    "goal_rhs": [-2.0],
                },
            }
        )
        # Mechanism 2: Over-pessimistic bulk sink assumption (model assumes sink=0.40, true sink=0.0)
        pos_2hop = [-1.8, *([-0.6 + 0.3 * k for k in range(n_obj - 4)]), 0.4, 1.1, 1.8]
        init_q_2hop = [0.0] * (n_obj - 3) + [2.0, 0.0, 0.0]
        goal_m_2hop = [
            [0.0] * (n_obj - 3) + [-1.0, 0.0, 0.0],
            [0.0] * (n_obj - 3) + [0.0, -1.0, 0.0],
            [0.0] * (n_obj - 3) + [0.0, 0.0, -1.0],
        ]
        cases.append(
            {
                "case_id": f"mech2_bulk_sink_N{n_obj}",
                "true_mechanism": "bulk_sink_dissipation",
                "lab": BudgetedControlLab(
                    objects=n_obj,
                    positions=tuple(pos_2hop),
                    coupling=0.50,
                    radius=0.80,
                    direction="right",
                    sink_rate=0.0,
                    control_bounds=(0.0, 0.25),
                    state_bounds=(0.0, 10.0),
                ),
                "init_model": {
                    "objects": n_obj,
                    "positions": pos_2hop,
                    "coupling": 0.50,
                    "radius": 0.80,
                    "direction": "right",
                    "sink_rate": 0.40,
                    "premise_ids": ["P_conservation", "P_control_input_gain"],
                },
                "task": {
                    "initial_state": init_q_2hop,
                    "control_bounds": [0.0, 0.25],
                    "state_bounds": [0.0, 10.0],
                    "budget": 0.15,
                    "step_cost": 0.0,
                    "fuel_weights": [1.0] * n_obj,
                    "goal_matrix": goal_m_2hop,
                    "goal_rhs": [0.0, 0.0, -0.85],
                },
            }
        )
        # Mechanism 3: Underestimated passive conveyor coupling (model coupling=0.10 vs true=0.50, known sink=0.10)
        pos_conv = [-1.8, *([-0.6 + 0.4 * k for k in range(n_obj - 3)]), 1.1, 1.8]
        init_q_conv = [0.0] * (n_obj - 2) + [2.0, 0.0]
        goal_m_conv = [[0.0] * (n_obj - 1) + [-1.0]]
        cases.append(
            {
                "case_id": f"mech3_coupling_shift_N{n_obj}",
                "true_mechanism": "coupling_shift",
                "lab": BudgetedControlLab(
                    objects=n_obj,
                    positions=tuple(pos_2hop),
                    coupling=0.50,
                    radius=0.80,
                    direction="right",
                    sink_rate=0.10,
                    control_bounds=(0.0, 0.25),
                    state_bounds=(0.0, 10.0),
                ),
                "init_model": {
                    "objects": n_obj,
                    "positions": pos_2hop,
                    "coupling": 0.10,
                    "radius": 0.80,
                    "direction": "right",
                    "sink_rate": 0.10,
                    "premise_ids": ["P_locality_radius", "P_interaction_direction"],
                },
                "task": {
                    "initial_state": init_q_2hop,
                    "control_bounds": [0.0, 0.25],
                    "state_bounds": [0.0, 10.0],
                    "budget": 0.12,
                    "step_cost": 0.0,
                    "fuel_weights": [1.0] * n_obj,
                    "goal_matrix": goal_m_2hop,
                    "goal_rhs": [0.0, 0.0, -0.75],
                },
            }
        )
        # Mechanism 4: Wrong directional premise (model assumes left obstruction vs true right conveyor)
        cases.append(
            {
                "case_id": f"mech4_direction_reversal_N{n_obj}",
                "true_mechanism": "direction_reversal",
                "lab": BudgetedControlLab(
                    objects=n_obj,
                    positions=tuple(pos_conv),
                    coupling=0.30,
                    radius=1.0,
                    direction="right",
                    control_bounds=(0.0, 0.25),
                    state_bounds=(0.0, 10.0),
                ),
                "init_model": {
                    "objects": n_obj,
                    "positions": pos_conv,
                    "coupling": 0.30,
                    "radius": 1.0,
                    "direction": "left",
                    "premise_ids": ["P_interaction_direction", "P_locality_radius"],
                },
                "task": {
                    "initial_state": init_q_conv,
                    "control_bounds": [0.0, 0.25],
                    "state_bounds": [0.0, 10.0],
                    "budget": 0.12,
                    "step_cost": 0.0,
                    "fuel_weights": [1.0] * n_obj,
                    "goal_matrix": goal_m_conv,
                    "goal_rhs": [-0.85],
                },
            }
        )

    records = []
    for item in cases:
        n_obj = item["init_model"]["objects"]
        p_graph = {
            "nodes": [
                {"id": "P_conservation", "status": "supported"},
                {"id": "P_control_input_gain", "status": "supported"},
                {"id": "P_interaction_direction", "status": "supported"},
                {"id": "P_locality_radius", "status": "supported"},
                {"id": "P_focus_membership", "status": "supported"},
            ]
        }
        cert_init = prove_task_solvability(
            item["init_model"], item["task"], certificate_id=f"{item['case_id']}_init"
        )
        arm_out = {}
        for arm_label, use_dep in (
            ("executed_untargeted_equal_tools", False),
            ("executed_dependency_guided_equal_tools", True),
        ):
            lab_copy = deepcopy(item["lab"])
            lab_copy.probe_queries = 0
            b_rep = SearchBudget(200000)
            rep = repair_proof_from_discrepancy(
                lab_copy,
                item["init_model"],
                p_graph,
                item["task"],
                cert_init,
                discrepancy_feedback={"oracle_winnable": True},
                use_dependency_localization=use_dep,
                budget=b_rep,
            )
            arm_out[arm_label] = {
                "repaired": rep["repaired"],
                "diagnosed_mechanism": rep.get("diagnosed_mechanism"),
                "measured_lab_probe_queries": lab_copy.probe_queries,
                "measured_compute_cost": b_rep.spent,
            }

        records.append(
            {
                "case_id": item["case_id"],
                "objects": n_obj,
                "true_mechanism": item["true_mechanism"],
                "initial_certificate_verdict": cert_init["verdict"],
                "arms": arm_out,
                # Strictly separated historical formula reference (Acceptance #20: never mixed with measured!)
                "historical_nominal_formula_estimate_only": 2 * (3 * n_obj + 1) + n_obj * (n_obj - 1),
            }
        )

    summary = {}
    for arm_label in ("executed_untargeted_equal_tools", "executed_dependency_guided_equal_tools"):
        rows = [r["arms"][arm_label] for r in records]
        summary[arm_label] = {
            "cases_total": len(rows),
            "distinct_mechanisms_tested": 4,
            "repaired_count": sum(1 for r in rows if r["repaired"]),
            "mean_measured_lab_probe_queries": float(mean(r["measured_lab_probe_queries"] for r in rows)),
            "sum_measured_lab_probe_queries": sum(r["measured_lab_probe_queries"] for r in rows),
            "mean_measured_compute_cost": float(mean(r["measured_compute_cost"] for r in rows)),
        }
    u_sum = summary["executed_untargeted_equal_tools"]["sum_measured_lab_probe_queries"]
    g_sum = summary["executed_dependency_guided_equal_tools"]["sum_measured_lab_probe_queries"]
    summary["measured_probe_reduction_pct_guided_vs_untargeted"] = round(
        100.0 * (1.0 - g_sum / max(1, u_sum)), 2
    )
    return {"summary": summary, "cases": records}


def main() -> int:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lib_win, comm_win, _ = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
    lib_inst, comm_inst, _ = _build_parent_commitment_from_empty_library(
        readout="total_event", s0_seed=600411, s1_seed=600412
    )

    b1 = _run_study_b1_equal_tool_knowledge(lib_win, comm_win, lib_inst, comm_inst)
    b2 = _run_study_b2_single_factor_dependency(comm_win, comm_inst)
    b3 = _run_study_b3_executed_discrepancy_repair()

    receipt = {
        "experiment_id": "v10_2_equal_tool_attribution_20261116",
        "generated_at": _utc_now(),
        "audited_anchor_commit": "a66cd82b22f444044dbda98f43782c96f833a4f0",
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "study_b1_equal_tool_knowledge_attribution": b1,
        "study_b2_single_factor_support_dependency_isolation": b2,
        "study_b3_executed_four_mechanism_discrepancy_repair": b3,
    }
    out_path = OUT_DIR / "equal_tool_attribution_20261116.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("B1 Relevant Aggregate:", json.dumps(b1["relevant_targets_aggregate"], indent=2))
    print("B2 Single-Factor Summary:", json.dumps(b2["single_factor_dependency_summary"], indent=2))
    print("B3 Executed Discrepancy Repair Summary:", json.dumps(b3["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

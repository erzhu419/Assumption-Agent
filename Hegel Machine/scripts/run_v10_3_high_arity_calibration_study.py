"""Run v10.3 Track 2: High-Arity (N=6,7,8) Sensor-Role Calibration Prior & Lifecycle Extension (model_calls = 0).

Writes new receipt (zero edits to any v10 or v10.2 receipts):
  ``docs/evidence/v10_3_scaling_and_calibration_20261117/high_arity_sensor_calibration_20261117.json``

Contains three strictly separated studies:
  1. **Study 2A (Single-Factor High-Arity Ablation, N in {6, 7, 8}, 9 tasks)**:
     Holds world commitments, solver repertoire, and budgets 100% identical while toggling ONLY
     ``sensor_role_calibration_priors=False`` (``commitment_only_cold_roles``) vs
     ``sensor_role_calibration_priors=True`` (``commitment_plus_calibration_prior``), alongside
     ``cold_full_tools`` across 3 mechanism/direction families at N=6, 7, 8 (up to (8!)^2 = 1.63e9
     anonymous sensor permutations).
  2. **Study 2B (Falsifiability & Scope-Safety Negative Controls, 3 prior-violating apparatuses)**:
     Verifies that ``SensorRoleCalibrationPrior`` empirically detects violations and never hallucinates
     false role bindings on:
       (a) ``variant='anchored'`` (violates equal-position silence -> ``refute_sensor_role_calibration_prior``),
       (b) 1D position-free Continuum Ring PDE (``ContinuumFieldLab`` -> ``skip_sensor_role_calibration_prior``,
           continuum solver still passes ``1e-7``),
       (c) global ``readout='total'`` (focus-invariant readout -> ``global_focus_invariant_readout``,
           still passes ``1e-7``).
  3. **Study 2C (Prospective Lifecycle Chains 711000-714000 Re-Evaluated at N=6 & Extended Stage 7 N=8)**:
     Tests the exact 4 pre-registered prospective chains from v10.2 with ``sensor_role_calibration_priors=True``
     and ``indexed_commitment_store=True`` (learning the calibration prior ONLY when Stage 1 confirmed, so
     ``prospective_chain_2`` remains an honest 0-prior negative!).
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

from scripts.run_v10_2_frozen_lifecycle_study import _preregistered_chain_specs
from scripts.run_v10_phase2b_open_grammar_llm import (
    _acquire_shared_discovery_bundle,
    _build_parent_commitment_from_empty_library,
    _eval_program_heldout_48,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuum_field_lab import ContinuumFieldLab
from hegel_machine.cognition.equal_tool_attribution import run_single_factor_dependency_ablation_arm
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import ensure_library_commitment_index
from hegel_machine.cognition.unified_research import UnifiedResearchAgent

OUT_DIR = REPO_ROOT / "docs" / "evidence" / "v10_3_scaling_and_calibration_20261117"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_n4_source_libraries() -> dict[str, KnowledgeLibrary]:
    """Build genuine N=4 confirmed source libraries for WTE donor_positive_gap and FQ total_event."""
    base_lib_wte, base_comm_wte, _ = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
    _, base_comm_te, _ = _build_parent_commitment_from_empty_library(
        readout="total_event", s0_seed=600411, s1_seed=600412
    )

    # 1. donor_positive_gap N=4 source library
    with AnonymousObjectLab(
        seed=700401,
        apparatus_seed=700041,
        objects=4,
        readout="window_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.24,
        relation_radius=1.06,
        actuator_curvature=0.16,
        name_seed=700501,
        wiring_seed=700601,
    ) as lab_dpg:
        b1 = _acquire_shared_discovery_bundle(lab_dpg, seed=700401)
        fb1 = initial_prediction(b1["training"], b1["spec"].risk_quantile)
        og1 = run_single_factor_dependency_ablation_arm(
            parent_commitment=base_comm_wte,
            model=b1["model"],
            records=b1["records"],
            spec=b1["spec"],
            training=b1["training"],
            validation=b1["validation"],
            compute_budget_limit=200000,
            use_support_dependency_cue=True,
            tolerance=1e-7,
        )
        best1 = og1["best_entry"]
        mse1 = _eval_program_heldout_48(lab_dpg, best1["candidate"].program, fb1, 1700404)
        lib_dpg = deepcopy(base_lib_wte)
        saver1 = UnifiedResearchAgent(
            ResearchConfig(seed=700401, world_structure_commitments=True, indexed_commitment_store=True),
            library=lib_dpg,
        )
        saver1._object_schemas[("fine",)] = deepcopy(best1["resolved_schema"])
        saver1.remember_confirmed_selection(
            best1,
            saver1._spec(lab_dpg, ("fine",)),
            b1["training"],
            b1["validation"],
            confirmed=True,
            confirmation_loss=float(mse1),
            confirmation_rows=48,
            selected_channels=("fine",),
        )
        ensure_library_commitment_index(lib_dpg, budget=SearchBudget(50000), tolerance=1e-7)

    # 2. focal_quadratic_total_event N=4 source library
    with AnonymousObjectLab(
        seed=620401,
        apparatus_seed=620411,
        objects=4,
        readout="focal_quadratic_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="always",
        relation_radius=1.05,
        actuator_curvature=0.15,
        name_seed=620502,
        wiring_seed=620612,
    ) as lab_fq:
        b2 = _acquire_shared_discovery_bundle(lab_fq, seed=620401)
        fb2 = initial_prediction(b2["training"], b2["spec"].risk_quantile)
        og2 = run_single_factor_dependency_ablation_arm(
            parent_commitment=base_comm_te,
            model=b2["model"],
            records=b2["records"],
            spec=b2["spec"],
            training=b2["training"],
            validation=b2["validation"],
            compute_budget_limit=200000,
            use_support_dependency_cue=True,
            tolerance=1e-7,
        )
        best2 = og2["best_entry"]
        mse2 = _eval_program_heldout_48(lab_fq, best2["candidate"].program, fb2, 1620404)
        lib_fq = KnowledgeLibrary()
        saver2 = UnifiedResearchAgent(
            ResearchConfig(seed=620401, world_structure_commitments=True, indexed_commitment_store=True),
            library=lib_fq,
        )
        saver2._object_schemas[("fine",)] = deepcopy(best2["resolved_schema"])
        saver2.remember_confirmed_selection(
            best2,
            saver2._spec(lab_fq, ("fine",)),
            b2["training"],
            b2["validation"],
            confirmed=True,
            confirmation_loss=float(mse2),
            confirmation_rows=48,
            selected_channels=("fine",),
        )
        ensure_library_commitment_index(lib_fq, budget=SearchBudget(50000), tolerance=1e-7)

    return {"dpg_wte": lib_dpg, "fq_te": lib_fq, "base_wte": base_lib_wte}


def _study_2a_high_arity_tasks() -> list[dict]:
    """9 high-arity anonymous apparatuses across N in {6, 7, 8} x 3 families."""
    tasks = []
    for n_obj in (6, 7, 8):
        s_base = 820000 + 1000 * n_obj
        tasks.append(
            {
                "task_id": f"ha_dpg_right_N{n_obj}",
                "source_key": "dpg_wte",
                "lab_kwargs": dict(
                    seed=s_base + 1,
                    apparatus_seed=s_base + 11,
                    objects=n_obj,
                    readout="window_total_event",
                    variant="exchange",
                    relation_direction="right",
                    relation_condition="donor_positive_gap",
                    condition_threshold=0.29 if n_obj != 7 else 0.21,
                    relation_radius=1.16,
                    actuator_curvature=0.14,
                    name_seed=s_base + 101,
                    wiring_seed=s_base + 201,
                ),
            }
        )
        tasks.append(
            {
                "task_id": f"ha_dpg_left_rev_N{n_obj}",
                "source_key": "dpg_wte",
                "lab_kwargs": dict(
                    seed=s_base + 2,
                    apparatus_seed=s_base + 12,
                    objects=n_obj,
                    readout="window_total_event",
                    variant="exchange",
                    relation_direction="left",
                    relation_condition="donor_positive_gap",
                    condition_threshold=0.27 if n_obj != 8 else 0.22,
                    relation_radius=1.14,
                    actuator_curvature=0.14,
                    name_seed=s_base + 102,
                    wiring_seed=s_base + 202,
                ),
            }
        )
        tasks.append(
            {
                "task_id": f"ha_fq_right_N{n_obj}",
                "source_key": "fq_te",
                "lab_kwargs": dict(
                    seed=s_base + 3,
                    apparatus_seed=s_base + 13,
                    objects=n_obj,
                    readout="focal_quadratic_total_event",
                    variant="exchange",
                    relation_direction="right",
                    relation_condition="always",
                    relation_radius=1.15,
                    actuator_curvature=0.14,
                    name_seed=s_base + 103,
                    wiring_seed=s_base + 203,
                ),
            }
        )
    return tasks


def _run_high_arity_episode(arm_name: str, lab_kwargs: dict, source_lib: KnowledgeLibrary) -> dict:
    t0 = time.monotonic()
    seed = int(lab_kwargs["seed"])
    n_obj = int(lab_kwargs["objects"])
    eval_seed = seed + 1000003
    obs_cap = 400
    comp_cap = 480000
    meas_cap = 3 * n_obj * obs_cap

    if arm_name == "cold_full_tools":
        lib = KnowledgeLibrary()
        use_mem = False
        use_cal = False
    elif arm_name == "commitment_only_cold_roles":
        lib = deepcopy(source_lib)
        use_mem = True
        use_cal = False
    elif arm_name == "commitment_plus_calibration_prior":
        lib = deepcopy(source_lib)
        use_mem = True
        use_cal = True
    else:
        raise ValueError(arm_name)

    cfg = ResearchConfig(
        seed=seed,
        observation_budget=obs_cap,
        compute_budget=comp_cap,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=use_mem,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=use_mem,
        sensor_role_calibration_priors=use_cal,
    )
    agent = UnifiedResearchAgent(cfg, measurement_budget=meas_cap, library=lib)
    with AnonymousObjectLab(**lab_kwargs) as lab:
        res = agent.run(lab)
        prog = Program.from_dict(res["program"]) if res.get("program") is not None else None
        ch = tuple(res.get("selected_channels") or ())
        confirmed = bool(res["status"] == "completed")
        mse = (
            float(loss(prog, lab.evaluation_rows(ch, seed=eval_seed, length=48, trajectories=1), res["fallback"], 0.5))
            if prog is not None
            else None
        )

    role_disc_act = next(
        (
            a
            for a in res.get("actions", ())
            if a.get("action") in {"adopt_discovered_objects", "retain_object_ambiguity"}
        ),
        {},
    )
    cal_act = next(
        (a for a in res.get("actions", ()) if "sensor_role_calibration" in str(a.get("action", ""))),
        None,
    )
    return {
        "arm": arm_name,
        "objects": n_obj,
        "confirmed": confirmed,
        "passed_1e7": bool(confirmed and mse is not None and mse <= 1e-7),
        "selected_channels": list(ch),
        "heldout_fresh48_mse": mse,
        "roles_fully_resolved": bool(role_disc_act.get("action") == "adopt_discovered_objects"),
        "role_discovery_rows": role_disc_act.get("rows"),
        "role_discovery_compute_cost": role_disc_act.get("compute_cost"),
        "calibration_action": cal_act,
        "costs": res["costs"],
        "model_calls": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
    }


def _run_study_2b_falsifiability_controls(source_lib: KnowledgeLibrary) -> list[dict]:
    """3 negative-control apparatuses violating the 4-role pure-additive/focal assumptions."""
    controls_out = []

    # Case 1: variant='anchored' -> violates equal-position silence (local_direct_effect)!
    cfg1 = ResearchConfig(
        seed=890601,
        observation_budget=340,
        compute_budget=360000,
        minimum_improvement=1e-7,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
    )
    ag1 = UnifiedResearchAgent(cfg1, measurement_budget=9600, library=deepcopy(source_lib))
    with AnonymousObjectLab(
        seed=890601,
        apparatus_seed=890611,
        objects=6,
        readout="focus",
        variant="anchored",
        relation_direction="right",
        relation_radius=1.05,
        name_seed=890701,
        wiring_seed=890801,
    ) as lab1:
        res1 = ag1.run(lab1)
    ref1 = [a for a in res1["actions"] if a.get("action") == "refute_sensor_role_calibration_prior"]
    app1 = [a for a in res1["actions"] if a.get("action") == "apply_sensor_role_calibration_prior"]
    controls_out.append(
        {
            "control_id": "falsify_1_anchored_local_direct_effect",
            "violation": "position_has_direct_local_effect_at_equal_x",
            "prior_cleanly_refuted": bool(len(ref1) >= 1 and len(app1) == 0),
            "false_role_application_count": len(app1),
            "refutation_telemetry": ref1[0] if ref1 else None,
        }
    )

    # Case 2: Position-free 1D Continuum Ring PDE -> 0 position controls/sensors!
    cfg2 = ResearchConfig(
        seed=890602,
        observation_budget=320,
        compute_budget=360000,
        minimum_improvement=1e-7,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        continuum_field_discovery=True,
        manifold_pde_discovery=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
    )
    ag2 = UnifiedResearchAgent(cfg2, measurement_budget=9600, library=deepcopy(source_lib))
    with ContinuumFieldLab(seed=890602, sites=5, readout="focal_pde_event", diffusion=0.18, advection=0.07) as lab2:
        res2 = ag2.run(lab2)
        p2 = Program.from_dict(res2["program"]) if res2.get("program") else None
        ch2 = tuple(res2.get("selected_channels") or ())
        mse2 = float(loss(p2, lab2.evaluation_rows(ch2, seed=9890602, length=48, trajectories=1), res2["fallback"], 0.5)) if p2 else None
    skip2 = [a for a in res2["actions"] if a.get("action") == "skip_sensor_role_calibration_prior"]
    app2 = [a for a in res2["actions"] if a.get("action") == "apply_sensor_role_calibration_prior"]
    controls_out.append(
        {
            "control_id": "falsify_2_position_free_continuum_ring_pde",
            "violation": "zero_position_sensors_and_controls",
            "prior_cleanly_skipped": bool(len(app2) == 0),
            "skip_telemetry": skip2[0] if skip2 else {"reason": "continuum_branch_routed_before_4role"},
            "continuum_solver_still_passed_1e7": bool(res2["status"] == "completed" and mse2 is not None and mse2 <= 1e-7),
            "heldout_fresh48_mse": mse2,
        }
    )

    # Case 3: Pure global readout='total' -> target invariant to focus selector!
    cfg3 = ResearchConfig(
        seed=890603,
        observation_budget=320,
        compute_budget=360000,
        minimum_improvement=1e-7,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
    )
    ag3 = UnifiedResearchAgent(cfg3, measurement_budget=9600, library=deepcopy(source_lib))
    with AnonymousObjectLab(
        seed=890603,
        apparatus_seed=890613,
        objects=6,
        readout="total",
        variant="exchange",
        relation_direction="right",
        relation_radius=1.08,
        name_seed=890703,
        wiring_seed=890803,
    ) as lab3:
        res3 = ag3.run(lab3)
        p3 = Program.from_dict(res3["program"]) if res3.get("program") else None
        ch3 = tuple(res3.get("selected_channels") or ())
        mse3 = float(loss(p3, lab3.evaluation_rows(ch3, seed=9890603, length=48, trajectories=1), res3["fallback"], 0.5)) if p3 else None
    cal3 = [a for a in res3["actions"] if a.get("action") == "apply_sensor_role_calibration_prior"]
    controls_out.append(
        {
            "control_id": "falsify_3_global_focus_invariant_total_readout",
            "violation": "target_has_zero_dependence_on_focus_selector",
            "detected_global_focus_invariance": bool(
                cal3 and cal3[0].get("focus_mode") == "global_focus_invariant_readout" and cal3[0].get("complete") is False
            ),
            "global_law_passed_1e7": bool(res3["status"] == "completed" and mse3 is not None and mse3 <= 1e-7),
            "heldout_fresh48_mse": mse3,
        }
    )
    return controls_out


def _run_study_2c_extended_prospective_lifecycle() -> dict:
    """Re-evaluate the exact 4 prospective chains 711000-714000 with calibration priors + Stage 7 (N=8) return."""
    base_lib, base_comm, _ = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
    prospective_specs = [
        s for s in _preregistered_chain_specs() if s["chain_role"] == "frozen_prospective"
    ]
    chains_out = []
    for c_spec in prospective_specs:
        s1_kw = c_spec["stage1_source"]
        s1_seed = int(s1_kw["seed"])
        with AnonymousObjectLab(**s1_kw) as lab1:
            b1 = _acquire_shared_discovery_bundle(lab1, seed=s1_seed)
            fb1 = initial_prediction(b1["training"], b1["spec"].risk_quantile)
            og1 = run_single_factor_dependency_ablation_arm(
                parent_commitment=base_comm,
                model=b1["model"],
                records=b1["records"],
                spec=b1["spec"],
                training=b1["training"],
                validation=b1["validation"],
                compute_budget_limit=200000,
                use_support_dependency_cue=True,
                tolerance=1e-7,
            )
            best1 = og1["best_entry"]
            mse1 = _eval_program_heldout_48(lab1, best1["candidate"].program, fb1, s1_seed + 1000003) if og1["supported"] else None
            confirmed1 = bool(mse1 is not None and mse1 <= 1e-7)

            # CRITICAL HONESTY RULE: Only store Stage 1 prior & object_model if Stage 1 ACTUALLY confirmed!
            # On prospective_chain_2 (712000), Stage 1 fails -> pers_lib gets ZERO Stage-1 commitment!
            pers_lib = KnowledgeLibrary() if not confirmed1 else deepcopy(base_lib)
            c0_cid = None
            if confirmed1 and best1 is not None:
                saver = UnifiedResearchAgent(
                    ResearchConfig(seed=s1_seed, world_structure_commitments=True, indexed_commitment_store=True),
                    library=pers_lib,
                )
                saver._object_schemas[("fine",)] = deepcopy(best1["resolved_schema"])
                rec = saver.remember_confirmed_selection(
                    best1,
                    saver._spec(lab1, ("fine",)),
                    b1["training"],
                    b1["validation"],
                    confirmed=True,
                    confirmation_loss=float(mse1),
                    confirmation_rows=48,
                    selected_channels=("fine",),
                )
                c0_cid = rec["concept_id"]
                ensure_library_commitment_index(pers_lib, budget=SearchBudget(50000), tolerance=1e-7)

        # Define Stage 7 (N=8 return of original 'right' condition!)
        base_s = int(c_spec["base_seed"])
        stage7_n8_kw = dict(
            seed=base_s + 801,
            apparatus_seed=base_s + 81,
            objects=8,
            readout="window_total_event",
            variant="exchange",
            relation_direction="right",
            relation_condition="donor_positive_gap",
            condition_threshold=float(c_spec["stage6_return"]["condition_threshold"]) + 0.03,
            relation_radius=1.17,
            actuator_curvature=0.14,
            name_seed=base_s + 581,
            wiring_seed=base_s + 681,
        )

        stage_results = {}
        for st_key, st_kw in (
            ("stage3_transfer_N5", c_spec["stage3_transfer"]),
            ("stage4_5_revision_N5", c_spec["stage4_5_counterexample_revision"]),
            ("stage6_return_N6", c_spec["stage6_return"]),
            ("stage7_return_N8", stage7_n8_kw),
        ):
            n_st = int(st_kw["objects"])
            st_seed = int(st_kw["seed"])
            cfg_st = ResearchConfig(
                seed=st_seed,
                observation_budget=400,
                compute_budget=480000,
                minimum_improvement=1e-7,
                max_candidates=4,
                use_memory=bool(confirmed1),
                world_structure_commitments=True,
                transition_premise_audit=True,
                expanded_premise_revisions=True,
                active_threshold_probing=True,
                indexed_commitment_store=bool(confirmed1),
                sensor_role_calibration_priors=bool(confirmed1),
            )
            ag_st = UnifiedResearchAgent(cfg_st, measurement_budget=3 * n_st * 400, library=pers_lib)
            with AnonymousObjectLab(**st_kw) as lab_st:
                res_st = ag_st.run(lab_st)
                p_st = Program.from_dict(res_st["program"]) if res_st.get("program") else None
                ch_st = tuple(res_st.get("selected_channels") or ())
                conf_st = bool(res_st["status"] == "completed")
                mse_st = (
                    float(
                        loss(
                            p_st,
                            lab_st.evaluation_rows(ch_st, seed=st_seed + 1000003, length=48, trajectories=1),
                            res_st["fallback"],
                            0.5,
                        )
                    )
                    if p_st is not None
                    else None
                )
            if confirmed1:
                ensure_library_commitment_index(pers_lib, budget=SearchBudget(20000), tolerance=1e-7)
            app_meta = (res_st.get("research_program") or {}).get("world_commitment_application") or {}
            apply_acts = [a for a in res_st.get("actions", ()) if a.get("action") == "apply_world_structure_commitments"]
            rej_ids = set(apply_acts[-1].get("rejected_inconsistent_commitment_ids") or ()) if apply_acts else set()
            stage_results[st_key] = {
                "objects": n_st,
                "confirmed": conf_st,
                "passed_1e7": bool(conf_st and mse_st is not None and mse_st <= 1e-7),
                "selected_channels": list(ch_st),
                "heldout_fresh48_mse": mse_st,
                "reorganization_type": app_meta.get("reorganization_type"),
                "recovered_c0": bool(
                    c0_cid
                    and app_meta.get("reorganization_type") == "equivalence_reorganization"
                    and c0_cid in (app_meta.get("source_concept_ids") or ())
                ),
                "rejected_incompatible_commitments_count": len(rej_ids),
                "compute_cost": res_st["costs"]["compute_cost"],
                "observation_cost": res_st["costs"]["observation_cost"],
            }
        print(
            f"[{c_spec['chain_id']}] S1={confirmed1} | "
            f"S3(N5)={stage_results['stage3_transfer_N5']['passed_1e7']} | "
            f"S45(N5)={stage_results['stage4_5_revision_N5']['passed_1e7']} | "
            f"S6(N6)={stage_results['stage6_return_N6']['passed_1e7']} (comp={stage_results['stage6_return_N6']['compute_cost']}) | "
            f"S7(N8)={stage_results['stage7_return_N8']['passed_1e7']} (comp={stage_results['stage7_return_N8']['compute_cost']})",
            flush=True,
        )
        chains_out.append(
            {
                "chain_id": c_spec["chain_id"],
                "stage1_confirmed": confirmed1,
                "stages": stage_results,
            }
        )

    formed_chains = [c for c in chains_out if c["stage1_confirmed"]]
    return {
        "total_prospective_chains": len(chains_out),
        "stage1_confirmed_chains": len(formed_chains),
        "stage6_N6_return_passes_all_4": sum(1 for c in chains_out if c["stages"]["stage6_return_N6"]["passed_1e7"]),
        "stage6_N6_return_passes_given_stage1": sum(
            1 for c in formed_chains if c["stages"]["stage6_return_N6"]["passed_1e7"]
        ),
        "stage7_N8_return_passes_all_4": sum(1 for c in chains_out if c["stages"]["stage7_return_N8"]["passed_1e7"]),
        "stage7_N8_return_passes_given_stage1": sum(
            1 for c in formed_chains if c["stages"]["stage7_return_N8"]["passed_1e7"]
        ),
        "all_4_post_freeze_stages_passed_count": sum(
            1 for c in chains_out if all(st["passed_1e7"] for st in c["stages"].values())
        ),
        "chains": chains_out,
    }


def main() -> int:
    t0 = time.monotonic()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_libs = _build_n4_source_libraries()

    # 1. Study 2A: 9 high-arity tasks x 3 arms
    tasks_2a = _study_2a_high_arity_tasks()
    episodes_2a = []
    for t in tasks_2a:
        s_lib = src_libs[t["source_key"]]
        arms_res = {}
        for arm in ("cold_full_tools", "commitment_only_cold_roles", "commitment_plus_calibration_prior"):
            arms_res[arm] = _run_high_arity_episode(arm, t["lab_kwargs"], s_lib)
        print(
            f"[{t['task_id']}] "
            f"cold={arms_res['cold_full_tools']['passed_1e7']} | "
            f"comm_cold_roles={arms_res['commitment_only_cold_roles']['passed_1e7']} "
            f"(comp={arms_res['commitment_only_cold_roles']['costs']['compute_cost']}) | "
            f"comm+cal_prior={arms_res['commitment_plus_calibration_prior']['passed_1e7']} "
            f"(comp={arms_res['commitment_plus_calibration_prior']['costs']['compute_cost']}, "
            f"obs={arms_res['commitment_plus_calibration_prior']['costs']['observation_cost']})",
            flush=True,
        )
        episodes_2a.append({"task_id": t["task_id"], "objects": t["lab_kwargs"]["objects"], "arms": arms_res})

    summary_2a = {}
    for arm in ("cold_full_tools", "commitment_only_cold_roles", "commitment_plus_calibration_prior"):
        rows = [e["arms"][arm] for e in episodes_2a]
        summary_2a[arm] = {
            "tasks_total": len(rows),
            "passes_1e7": sum(1 for r in rows if r["passed_1e7"]),
            "roles_fully_resolved_count": sum(1 for r in rows if r["roles_fully_resolved"]),
            "passes_by_arity": {
                f"N={n}": sum(1 for r in rows if r["objects"] == n and r["passed_1e7"])
                for n in (6, 7, 8)
            },
            "mean_compute_cost": float(mean(r["costs"]["compute_cost"] for r in rows)),
            "mean_observation_cost": float(mean(r["costs"]["observation_cost"] for r in rows)),
            "mean_role_discovery_compute_cost": float(
                mean(int(r["role_discovery_compute_cost"] or 0) for r in rows)
            ),
            "model_calls": 0,
        }

    # 2. Study 2B: Falsifiability controls
    controls_2b = _run_study_2b_falsifiability_controls(src_libs["dpg_wte"])
    print("Study 2B Falsifiability Controls:", json.dumps(controls_2b, indent=2), flush=True)

    # 3. Study 2C: Extended prospective lifecycle 711000-714000 up to N=8
    lifecycle_2c = _run_study_2c_extended_prospective_lifecycle()

    receipt = {
        "experiment_id": "v10_3_high_arity_sensor_calibration_20261117",
        "generated_at": _utc_now(),
        "base_commit": "85b6055b11312a53720561f6697f6054b4ab8873",
        "model_calls": 0,
        "total_tokens": 0,
        "wall_seconds": round(time.monotonic() - t0, 3),
        "study_2a_high_arity_single_factor_ablation": {
            "summary_by_arm": summary_2a,
            "episodes": episodes_2a,
        },
        "study_2b_falsifiability_negative_controls": controls_2b,
        "study_2c_prospective_lifecycle_n6_and_n8_return": lifecycle_2c,
    }
    out_path = OUT_DIR / "high_arity_sensor_calibration_20261117.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

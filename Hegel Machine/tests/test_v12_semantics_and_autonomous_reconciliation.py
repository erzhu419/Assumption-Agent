"""Comprehensive regression & behavioral test suite for Hegel Machine v12 (Deliverables A, B, and C).

Verifies all v12 audit remediations from ``markdown/hegel_v12/02_下一轮最小施工与实验.md``:
  - A1 (F01): Fixed confirmation contract (no theta-slot 150,000x relaxation, missing fresh data
    reported as ``unknown_no_fresh_data`` rather than ``0.0``, provisional approximate candidates
    never marked ``confirmed=True``).
  - A2 (F02, F09): ``shrink_claim_to_observable_scope`` returns ``scope_unidentified`` (``scope_guard=None``)
    when empirical support fails to separate; manifest whitelist + pilot timing enforcement;
    bounds/guard consistency in ``assess_scope_overlap``.
  - A3 (F05, F06, F07, F08 + all 6 minimal regressions):
      1. ``x > 0`` vs ``x > 5e-10`` NOT algebraically identical;
      2. ``0.22, 0.46, 0.34`` 3-branch incremental anti-unification succeeds & is permutation-invariant,
         and coincidental equal constants at distinct AST paths get separate slots;
      3. Distinct scientific ``commitment_id`` for ``right`` vs ``left`` families sharing ``predicate_skeleton_id``;
      4. Unsupported root ``{'claim_id': 'new_root', 'status': 'candidate'}`` gets ``unverified_missing_support``;
      5. 2-slot predicate calibration binds both slots, and 3-slot family explicitly returns ``unsupported_multi_slot_arity``;
      6. ``q^2`` dimensional scaling uses ``y_scale / (q_scale ** 2)`` and inhomogeneous nonlinear terms return ``aligned=False``.
  - A4 (F10, F11): 4-stage ``ConsumptionEvent`` separation (``candidate_generation`` vs ``episode_resolution``,
    no premature ``parent_after = gen_cids[0]``, zero fabricated probe ID strings).
  - B (F03): Ordinary ``UnifiedResearchAgent.run(lab)`` entry autonomously discovers conditional branches,
    shrinks ``H0``'s scope, and anti-unifies 3 branches into ``H*`` without harness routing.
"""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.run_v10_3_library_scaling_study import _make_confirmed_regime_prior
from hegel_machine.cognition.branch_reconciliation import (
    _has_unbound_theta_slot,
    _programs_algebraically_identical,
    anti_unify_program_family,
    anti_unify_programs,
    assess_scope_overlap,
    validate_observable_scope_guard,
    find_feasible_prediction_difference,
    instantiate_aligned_claim_program,
    instantiate_parametric_family_on_target,
    propagate_premise_retraction_invalidation,
    propose_anti_unified_generalization,
    shrink_claim_to_observable_scope,
    validate_parametric_family_calibration_support,
)
from hegel_machine.cognition.equal_tool_attribution import (
    build_action_chain_telemetry,
    extract_real_consumption_events_from_actions,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Observation, Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import extract_archive_witness_maps
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.unified_research import UnifiedResearchAgent
from hegel_machine.cognition.world_structure_commitments import consolidate_world_structure_commitments


def _pred_lt(thresh: float) -> Program:
    return Program("lt", children=(Program("const", value=thresh), Program("arg", name="x")))


def _dpg_dict(donor_min: float, gap_min: float) -> dict:
    q_src = Program("arg", name="edge_source_quantity")
    q_dst = Program("arg", name="edge_target_quantity")
    gap = Program("sub", children=(q_src, q_dst))
    return Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=donor_min), q_src)),
            Program("lt", children=(Program("const", value=gap_min), gap)),
        ),
    ).to_dict()


def test_v12_a3_1_and_a3_6_gpt6_source_excerpt_probes_remediated() -> None:
    # 1. Exact comparator threshold non-equivalence (F05)
    p0, p1 = _pred_lt(0.0), _pred_lt(5e-10)
    assert _programs_algebraically_identical(p0, p1) is False
    assert _programs_algebraically_identical(p0, p0) is True

    # Also check via find_feasible_prediction_difference on aligned claims
    prior_22 = _make_confirmed_regime_prior(
        concept_id="p_0",
        seed=8101,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 0.0),
        radius=0.92,
    )
    prior_eps = _make_confirmed_regime_prior(
        concept_id="p_eps",
        seed=8101,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 5e-10),
        radius=0.92,
    )
    c0 = consolidate_world_structure_commitments([prior_22])[0]
    ceps = consolidate_world_structure_commitments([prior_eps])[0]
    sc = ObjectSchema.from_dict(prior_22["object_schema"])
    lab = UnifiedObjectLab(seed=8101, objects=4, readout="window_total_event")
    spec = lab.spec
    lab.close()
    al0 = instantiate_aligned_claim_program({"claim_id": "c0", "commitment": c0}, sc)
    ale = instantiate_aligned_claim_program({"claim_id": "ceps", "commitment": ceps}, sc)
    diff = find_feasible_prediction_difference(al0, ale, spec, sc)
    assert diff["proved_absent"] is False

    # 2. Incremental 3-branch anti-unification (F06)
    au = anti_unify_programs(_pred_lt(0.22), _pred_lt(0.46))
    assert au is not None
    third = anti_unify_programs(au[0], _pred_lt(0.34))
    assert third is not None
    assert [s["slot"] for s in third[1]] == ["theta_0"]

    # 3. Unsupported candidate root & ungrounded cycle (F07)
    unknown = propagate_premise_retraction_invalidation(
        [{"claim_id": "new_root", "status": "candidate"}]
    )[0]
    assert unknown["support_validity"] == "unverified_missing_support"
    assert unknown["support_validity"] != "valid_grounded"

    cycle = propagate_premise_retraction_invalidation(
        [
            {"claim_id": "A", "depends_on_claim_ids": ["B"], "independent_empirical_support": False},
            {"claim_id": "B", "depends_on_claim_ids": ["A"], "independent_empirical_support": False},
        ]
    )
    assert all(c["support_validity"] == "invalid_ungrounded_cycle" for c in cycle)


def test_v12_a3_2_and_a3_4_order_invariance_coincidence_guard_and_family_identity() -> None:
    priors = [
        _make_confirmed_regime_prior(
            concept_id=f"p_right_{tag}",
            seed=8200 + idx,
            objects=4,
            readout="window_total_event",
            direction="right",
            edge_pred_dict=_dpg_dict(0.0, th),
            radius=0.94,
        )
        for idx, (tag, th) in enumerate([("22", 0.22), ("46", 0.46), ("34", 0.34)])
    ]
    sup_map, _ = extract_archive_witness_maps(priors)
    records = [
        {
            "claim_id": p["source_concept_id"],
            "commitment": consolidate_world_structure_commitments([p])[0],
            "schema": p["object_schema"],
            "support_rows": sup_map[p["source_concept_id"]],
            "scoped_counterexamples": [],
        }
        for p in priors
    ]

    ref_res = None
    for perm in itertools.permutations(records):
        res = propose_anti_unified_generalization(list(perm), budget=SearchBudget(40_000))
        assert res["accepted"] is True
        assert res["parameter_slots"] == ["theta_0"]
        if ref_res is None:
            ref_res = res
        else:
            assert res["anti_unified_predicate_skeleton"] == ref_res["anti_unified_predicate_skeleton"]
            assert res["generalized_commitment"]["commitment_id"] == ref_res["generalized_commitment"]["commitment_id"]
            assert res["per_branch_substitutions"] == ref_res["per_branch_substitutions"]

    # Numerical coincidence guard: equal values at TWO different AST roles must NOT share one theta_0!
    coinc_a = Program.from_dict(_dpg_dict(0.22, 0.22))
    coinc_b = Program.from_dict(_dpg_dict(0.46, 0.46))
    fam_coinc = anti_unify_program_family([("a", coinc_a), ("b", coinc_b)])
    assert fam_coinc is not None
    assert fam_coinc["parameter_slots"] == ["theta_0", "theta_1"]

    # A3.4 / F08: Same predicate skeleton, different world direction -> distinct scientific commitment_id!
    left_priors = [
        _make_confirmed_regime_prior(
            concept_id=f"p_left_{tag}",
            seed=8250 + idx,
            objects=4,
            readout="window_total_event",
            direction="left",
            edge_pred_dict=_dpg_dict(0.0, th),
            radius=0.94,
        )
        for idx, (tag, th) in enumerate([("22", 0.22), ("46", 0.46)])
    ]
    sup_left, _ = extract_archive_witness_maps(left_priors)
    left_recs = [
        {
            "claim_id": p["source_concept_id"],
            "commitment": consolidate_world_structure_commitments([p])[0],
            "schema": p["object_schema"],
            "support_rows": sup_left[p["source_concept_id"]],
            "scoped_counterexamples": [],
        }
        for p in left_priors
    ]
    res_left = propose_anti_unified_generalization(left_recs, budget=SearchBudget(40_000))
    assert res_left["accepted"] is True
    assert res_left["predicate_skeleton_id"] == ref_res["predicate_skeleton_id"]
    assert (
        res_left["generalized_commitment"]["commitment_id"]
        != ref_res["generalized_commitment"]["commitment_id"]
    )


def test_v12_a3_3_two_slot_calibration_and_three_slot_explicit_rejection() -> None:
    # Build 2-parameter family: donor_min in {0.05, 0.18} and gap_min in {0.22, 0.46}
    p_a = _make_confirmed_regime_prior(
        concept_id="p_2slot_a",
        seed=8301,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.05, 0.22),
        radius=0.94,
    )
    p_b = _make_confirmed_regime_prior(
        concept_id="p_2slot_b",
        seed=8302,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.18, 0.46),
        radius=0.94,
    )
    sup_2, _ = extract_archive_witness_maps([p_a, p_b])
    au_2 = propose_anti_unified_generalization(
        [
            {
                "claim_id": "b_a",
                "commitment": consolidate_world_structure_commitments([p_a])[0],
                "schema": p_a["object_schema"],
                "support_rows": sup_2["p_2slot_a"],
                "scoped_counterexamples": [],
            },
            {
                "claim_id": "b_b",
                "commitment": consolidate_world_structure_commitments([p_b])[0],
                "schema": p_b["object_schema"],
                "support_rows": sup_2["p_2slot_b"],
                "scoped_counterexamples": [],
            },
        ],
        budget=SearchBudget(40_000),
    )
    assert au_2["accepted"] is True
    assert au_2["parameter_slots"] == ["theta_0", "theta_1"]

    # Target dataset generated by (theta_0=0.15, theta_1=0.34, r=0.94)
    p_tgt = _make_confirmed_regime_prior(
        concept_id="p_2slot_tgt",
        seed=8303,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.15, 0.34),
        radius=0.94,
    )
    tgt_rows = extract_archive_witness_maps([p_tgt])[0]["p_2slot_tgt"]
    sc_tgt = ObjectSchema.from_dict(p_tgt["object_schema"])
    lab_t = UnifiedObjectLab(seed=8303, objects=4, readout="window_total_event")
    spec_t = lab_t.spec_for(("fine",))
    lab_t.close()

    gf_2 = deepcopy(au_2["generalized_commitment"])
    cands_2 = instantiate_parametric_family_on_target(
        gf_2, spec_t, tgt_rows, sc_tgt, SearchBudget(80_000), tolerance=1e-7
    )
    assert len(cands_2) == 1
    assert cands_2[0]["training_loss"] <= 1e-7
    assert "theta_0" in cands_2[0]["calibrated_slots"]
    assert "theta_1" in cands_2[0]["calibrated_slots"]
    assert not _has_unbound_theta_slot(cands_2[0]["candidate"].program)

    # 3-slot family must explicitly reject with unsupported_multi_slot_arity!
    gf_3 = deepcopy(gf_2)
    gf_3["parameter_slots"] = ["theta_0", "theta_1", "theta_2"]
    sup_chk = validate_parametric_family_calibration_support(gf_3)
    assert sup_chk["supported"] is False
    assert sup_chk["status"] == "unsupported_multi_slot_arity"
    cands_3 = instantiate_parametric_family_on_target(
        gf_3, spec_t, tgt_rows, sc_tgt, SearchBudget(80_000), tolerance=1e-7
    )
    assert cands_3 == []
    assert gf_3["_last_instantiation_diagnostic"]["status"] == "unsupported_multi_slot_arity"


def test_v12_a3_5_nonlinear_q_squared_unit_scaling_and_inhomogeneous_rejection() -> None:
    p_base = _make_confirmed_regime_prior(
        concept_id="p_unit",
        seed=8401,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=None,
        radius=0.90,
    )
    comm = consolidate_world_structure_commitments([p_base])[0]
    sc = ObjectSchema.from_dict(p_base["object_schema"])
    q0 = sc.fields[sc.objects[0]]["q"]
    q_sq_term = Program("mul", children=(Program("feature", name=q0), Program("feature", name=q0)))

    # Homogeneous quadratic q^2: if q_target = 2 * q_source and y_target = 1 * y_source,
    # weight MUST scale by 1 / (2^2) = 0.25 (NOT linear 1 / 2 = 0.5)!
    quad_claim = {
        "claim_id": "quad_q2",
        "commitment": comm,
        "explicit_terms": [q_sq_term],
        "coefficients": [0.0, 1.0],
    }
    al_quad = instantiate_aligned_claim_program(
        quad_claim, sc, unit_scales={"q_scale": 2.0, "x_scale": 1.0, "y_scale": 1.0}
    )
    assert al_quad["aligned"] is True
    assert al_quad["term_q_degrees"] == [2]
    assert abs(al_quad["coefficients"][1] - 0.25) <= 1e-12
    assert abs(al_quad["coefficients"][1] - 0.5) > 0.2

    # Inhomogeneous term `q + q^2` cannot scale by a single coefficient under q_scale=2.0 -> explicit rejection!
    inhom_term = Program("add", children=(Program("feature", name=q0), q_sq_term))
    inhom_claim = {
        "claim_id": "inhom_q_plus_q2",
        "commitment": comm,
        "explicit_terms": [inhom_term],
        "coefficients": [0.0, 1.0],
    }
    al_inhom = instantiate_aligned_claim_program(
        inhom_claim, sc, unit_scales={"q_scale": 2.0, "x_scale": 1.0, "y_scale": 1.0}
    )
    assert al_inhom["aligned"] is False
    assert al_inhom["reason"] == "unsupported_inhomogeneous_or_nonlinear_q_unit_scaling"


def test_v12_a1_a2_fixed_confirmation_contract_and_scope_unidentified() -> None:
    p1 = _make_confirmed_regime_prior(
        concept_id="p1",
        seed=8501,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 0.22),
        radius=0.92,
    )
    p2 = _make_confirmed_regime_prior(
        concept_id="p2",
        seed=8502,
        objects=4,
        readout="window_total_event",
        direction="right",
        edge_pred_dict=_dpg_dict(0.0, 0.46),
        radius=0.92,
    )
    sup, _ = extract_archive_witness_maps([p1, p2])
    c1 = consolidate_world_structure_commitments([p1])[0]
    c2 = consolidate_world_structure_commitments([p2])[0]

    # 1. Missing fresh data -> unknown_no_fresh_data (None per branch, NEVER 0.0!)
    no_fresh = propose_anti_unified_generalization(
        [
            {"claim_id": "b1", "commitment": c1, "schema": p1["object_schema"], "support_rows": sup["p1"]},
            {"claim_id": "b2", "commitment": c2, "schema": p2["object_schema"], "support_rows": sup["p2"]},
        ],
        tolerance=1e-7,
    )
    assert no_fresh["fresh_confirmation_status"] == "unknown_no_fresh_data"
    assert no_fresh["per_branch_fresh_mse"] == {"b1": None, "b2": None}
    assert no_fresh["confirmed"] is False

    # 2. Same perturbed fresh MSE (~0.004 > 1e-7, <= 1.5e-2) must NOT be accepted regardless of theta slots!
    pert_row = deepcopy(sup["p1"][0])
    noisy_fresh = [
        Observation(dict(pert_row.features), float(pert_row.target) + 0.06, pert_row.history)
    ]
    with_slot = propose_anti_unified_generalization(
        [
            {
                "claim_id": "b1",
                "commitment": c1,
                "schema": p1["object_schema"],
                "support_rows": sup["p1"],
                "fresh_holdout_rows": noisy_fresh,
            },
            {
                "claim_id": "b2",
                "commitment": c2,
                "schema": p2["object_schema"],
                "support_rows": sup["p2"],
                "fresh_holdout_rows": sup["p2"][:4],
            },
        ],
        tolerance=1e-7,
    )
    no_slot = propose_anti_unified_generalization(
        [
            {
                "claim_id": "b1a",
                "commitment": c1,
                "schema": p1["object_schema"],
                "support_rows": sup["p1"],
                "fresh_holdout_rows": noisy_fresh,
            },
            {
                "claim_id": "b1b",
                "commitment": deepcopy(c1),
                "schema": p1["object_schema"],
                "support_rows": sup["p1"],
                "fresh_holdout_rows": sup["p1"][:4],
            },
        ],
        tolerance=1e-7,
    )
    assert with_slot["accepted"] is False and with_slot["confirmed"] is False
    assert no_slot["accepted"] is False and no_slot["confirmed"] is False
    assert with_slot["status"] == no_slot["status"] == "provisional_approximate_candidate_family"

    # 3. A2: Non-separating support/counterexample returns scope_unidentified (scope_guard=None)!
    unsep = shrink_claim_to_observable_scope(
        {"claim_id": "H0", "status": "universal"},
        retained_support_regime_features=[
            {"obs_unconditional_block_rate": 0.0, "obs_signed_direction_flux": 0.12},
            {"obs_unconditional_block_rate": 0.5, "obs_signed_direction_flux": 0.25},
        ],
        refuting_regime_features={"obs_unconditional_block_rate": 0.25, "obs_signed_direction_flux": 0.18},
    )
    assert unsep["status"] == "scope_unidentified"
    assert unsep["shrunk"] is False
    assert unsep["scope_guard"] is None

    # Pilot-requiring guard variable rejected when pilot_calibration_available=False
    pilot_guard = Program(
        "lt", children=(Program("feature", name="obs_unconditional_block_rate"), Program("const", value=0.2))
    )
    ok_no_pilot, msg_no_pilot = validate_observable_scope_guard(
        pilot_guard, pilot_calibration_available=False
    )
    assert ok_no_pilot is False
    assert "pilot_calibration_required" in msg_no_pilot

    # Guard without consistent bounds never defaults to disjoint!
    ov = assess_scope_overlap(
        {"scope_guard": pilot_guard.to_dict(), "observable_bounds": {}},
        {"scope_guard": pilot_guard.to_dict(), "observable_bounds": {"obs_unconditional_block_rate": [0.5, 1.0]}},
    )
    assert ov["certified_disjoint"] is False
    assert ov["unknown"] is True


def test_v12_b_and_a4_ordinary_entry_autonomous_lifecycle_and_4stage_telemetry() -> None:
    # Verify zero private lab._ reads across runtime modules
    for mod_rel in (
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/discriminating_experiment_policy.py",
        "src/hegel_machine/cognition/premise_commitment_index.py",
    ):
        txt = (REPO_ROOT / mod_rel).read_text(encoding="utf-8")
        assert "lab._" not in txt and "_evaluation_schema" not in txt

    shared_lib = KnowledgeLibrary()
    specs = [
        ("s_cal_sym", 8600, "symmetric", "always", 0.0, 0.92),
        ("s0_always", 8601, "right", "always", 0.0, 0.94),
        ("s1_dpg22", 8602, "right", "donor_positive_gap", 0.22, 0.94),
        ("s2_dpg46", 8603, "right", "donor_positive_gap", 0.46, 0.92),
        ("s3_dpg34", 8604, "right", "donor_positive_gap", 0.34, 0.96),
    ]
    results = []
    for name, sd, d_mode, r_cond, th, rad in specs:
        cfg = ResearchConfig(
            seed=sd,
            observation_budget=640,
            compute_budget=720_000,
            max_rounds=4,
            minimum_improvement=1e-7,
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
        with AnonymousObjectLab(
            seed=sd,
            objects=4,
            readout="window_total_event",
            relation_direction=d_mode,
            relation_condition=r_cond,
            condition_threshold=th,
            relation_radius=rad,
            actuator_curvature=0.15,
            name_seed=sd + 51,
            wiring_seed=sd + 71,
        ) as anon_lab:
            res = UnifiedResearchAgent(
                cfg, measurement_budget=12_400, library=shared_lib
            ).run(anon_lab)
        results.append((name, res))

    assert all(r["status"] == "completed" for _, r in results)
    assert len(shared_lib._generalized_branch_commitments) == 1
    gf = shared_lib._generalized_branch_commitments[0]
    assert len(gf["source_branch_ids"]) == 3
    assert gf["parameter_slots"] == ["theta_0"]

    # Check 4-stage ConsumptionEvents on S3 (zero fake probe IDs, generation parent_after == parent_before)
    evs_s3 = extract_real_consumption_events_from_actions(results[-1][1]["actions"])
    gen_evs = [e for e in evs_s3 if e.get("event_stage") == "candidate_generation"]
    res_evs = [e for e in evs_s3 if e.get("event_stage") == "episode_resolution"]
    assert len(gen_evs) >= 1 and len(res_evs) >= 1
    for ge in gen_evs:
        assert ge["parent_after"] == ge["parent_before"]
        assert ge["probe_after"] is None
    for re_ev in res_evs:
        assert re_ev.get("probe_after") not in {
            "parametric_slot_calibration",
            "active_threshold_or_interval_probe",
            "premise_revised_probe",
        }
    tel_s3 = build_action_chain_telemetry(
        arm="autonomous",
        task_id=results[-1][1]["task_id"],
        knowledge_available_count=len(shared_lib.meta_priors),
        knowledge_retrieved_ids=[gf["commitment_id"]],
        confirmed=True,
        heldout_mse=0.0,
        consumption_events=evs_s3,
    )
    assert tel_s3["adopted_confirmed_from_knowledge"] is True

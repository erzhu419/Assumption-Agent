"""Comprehensive acceptance test suite for Hegel Machine v11 (Deliverables A, B, and C).

Covers all acceptance items from ``markdown/hegel_v11/02_下一轮实施与验收.md``:
  - Deliverable A (A1, A2, A3): 6/6 semantic mutation detection, atomic incremental group commits,
    budget-interrupted resume parity, parent dependency retraction without collateral deletion,
    5-way retrieval telemetry, zero archive double-charging, ``OrdinaryPersistentCommitmentCache``,
    ``UnifiedResourceAccount``, ``synthetic_stress_fixture`` isolation, real ``ConsumptionEvent``
    non-invocation, and provisional Tier-0 Occam epistemic status.
  - Deliverable B (B1–B10): role/unit/time alignment, executable observable scope guards (rejecting
    ``task_id``/oracle tokens and never defaulting ``unknown`` to ``disjoint``), feasible
    discriminating ``Probe`` synthesis, refuted newcomer protection, universal scope shrinking,
    certified equivalence vs finite-sample coincidence, typed AST anti-unification (valid merge
    accepted, minority-degrading false merge rejected), and retraction/cycle grounding.
  - Deliverable C: runtime method induction (``author_origin='runtime_induced_from_dev_episodes'``),
    ``exploration_archive`` retention, ``ConsumptionEvent`` probe-change tracking, and zero private
    ``lab._...`` runtime leakage.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.run_v10_3_library_scaling_study import _make_confirmed_regime_prior
from hegel_machine.cognition.branch_reconciliation import (
    adjudicate_branches_on_paid_evidence,
    align_semantics_roles_units_and_time,
    assess_scope_overlap,
    find_feasible_prediction_difference,
    propose_anti_unified_generalization,
    propagate_premise_retraction_invalidation,
    validate_observable_scope_guard,
)
from hegel_machine.cognition.continuous_state import _linear_fit, _sum_terms
from hegel_machine.cognition.discriminating_experiment_policy import (
    build_generic_schema_candidate_hypotheses,
    compute_observable_probe_features,
    frozen_default_method_m0,
    generate_candidate_probe_menu,
    induce_discriminating_experiment_method,
    select_probe_under_method,
)
from hegel_machine.cognition.equal_tool_attribution import (
    UnifiedResourceAccount,
    build_action_chain_telemetry,
    extract_real_consumption_events_from_actions,
    reconstruct_commitments_from_raw_archive,
)
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import Probe, ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import (
    OrdinaryPersistentCommitmentCache,
    count_empirical_confirmation_sources,
    ensure_library_commitment_index,
)
from hegel_machine.cognition.premise_dependency_audit import audit_transition_premises
from hegel_machine.cognition.structure_lab import ObjectSchema
from hegel_machine.cognition.unified_object_lab import UnifiedObjectLab
from hegel_machine.cognition.world_structure_commitments import (
    consolidate_world_structure_commitments,
    instantiate_commitment_terms,
)


def _sample_fixture_priors() -> list[dict]:
    q_src = Program("arg", name="edge_source_quantity")
    q_dst = Program("arg", name="edge_target_quantity")
    gap = Program("sub", children=(q_src, q_dst))
    pred_22 = Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=0.0), q_src)),
            Program("lt", children=(Program("const", value=0.22), gap)),
        ),
    ).to_dict()
    pred_46 = Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=0.0), q_src)),
            Program("lt", children=(Program("const", value=0.46), gap)),
        ),
    ).to_dict()
    return [
        _make_confirmed_regime_prior(
            concept_id="p_right_uncond",
            seed=7101,
            objects=4,
            readout="window_total_event",
            direction="right",
            edge_pred_dict=None,
            radius=0.92,
        ),
        _make_confirmed_regime_prior(
            concept_id="p_right_dpg22",
            seed=7102,
            objects=4,
            readout="window_total_event",
            direction="right",
            edge_pred_dict=pred_22,
            radius=0.94,
        ),
        _make_confirmed_regime_prior(
            concept_id="p_right_dpg46",
            seed=7103,
            objects=4,
            readout="window_total_event",
            direction="right",
            edge_pred_dict=pred_46,
            radius=0.96,
        ),
        _make_confirmed_regime_prior(
            concept_id="p_left_dpg25",
            seed=7104,
            objects=4,
            readout="window_total_event",
            direction="left",
            edge_pred_dict=pred_22,
            radius=0.90,
        ),
    ]


def test_a1_gpt6_fingerprint_probe_and_versioned_group_invalidation() -> None:
    probe_py = REPO_ROOT / "markdown" / "hegel_v11" / "03_fingerprint_probe.py"
    res = subprocess.run(
        [sys.executable, str(probe_py), "--repo-root", str(REPO_ROOT), "--allow-different-source"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    report = json.loads(res.stdout)
    assert report["undetected_semantic_mutations"] == 0
    for mut_info in report["semantic_mutations"].values():
        assert mut_info["fingerprint_changed"] is True

    priors = _sample_fixture_priors()
    lib = KnowledgeLibrary(meta_priors=deepcopy(priors))
    ensure_library_commitment_index(lib, budget=SearchBudget(80_000), tolerance=1e-7)
    init_groups = len(lib._indexed_group_commitments)
    assert init_groups >= 3

    # Mutate coefficients in-place on ONE prior while keeping ID and len(meta_priors) unchanged
    old_gk = lib.meta_priors[-1].get("_cached_commitment_group_key")
    lib.meta_priors[-1]["structure"]["fits"][0]["coefficients"] = [0.0, 1.15, 1.0, 0.7, -0.7]
    ensure_library_commitment_index(lib, budget=SearchBudget(80_000), tolerance=1e-7)
    tel = lib._last_index_telemetry
    assert lib.meta_priors[-1].get("_cached_commitment_group_key") != old_gk
    assert tel["groups_reconsolidated"] == 1
    assert tel["groups_reused_from_cache"] == init_groups - 1


def test_a1_partial_budget_atomic_build_and_resume_parity() -> None:
    priors = _sample_fixture_priors()
    lib_gold = KnowledgeLibrary(meta_priors=deepcopy(priors))
    comms_gold, _, _ = ensure_library_commitment_index(
        lib_gold, budget=SearchBudget(80_000), tolerance=1e-7
    )

    lib_part = KnowledgeLibrary(meta_priors=deepcopy(priors))
    ensure_library_commitment_index(lib_part, budget=SearchBudget(120), tolerance=1e-7)
    assert lib_part._index_build_status == "partial_budget_pending"
    assert lib_part._indexed_prior_fingerprints is None
    assert lib_part._last_index_telemetry["dirty_pending_groups_count"] >= 1

    comms_resumed, _, _ = ensure_library_commitment_index(
        lib_part, budget=SearchBudget(80_000), tolerance=1e-7
    )
    assert lib_part._index_build_status == "complete"
    assert [c["commitment_id"] for c in comms_resumed] == [
        c["commitment_id"] for c in comms_gold
    ]


def test_a1_dependency_retraction_and_five_way_retrieval_telemetry() -> None:
    priors = _sample_fixture_priors()
    p_parent = deepcopy(priors[0])
    p_dep = deepcopy(priors[1])
    p_dep["source_concept_id"] = "dep_child"
    p_dep["depends_on_concept_ids"] = [p_parent["source_concept_id"]]
    p_dep["independent_empirical_support"] = False

    p_indep = deepcopy(priors[2])
    p_indep["source_concept_id"] = "indep_child"
    p_indep["depends_on_concept_ids"] = [p_parent["source_concept_id"]]
    p_indep["independent_empirical_support"] = True

    lib = KnowledgeLibrary(meta_priors=[p_parent, p_dep, p_indep])
    _, idx, _ = ensure_library_commitment_index(lib, budget=SearchBudget(80_000), tolerance=1e-7)

    # Check 5-way telemetry under tiny query budget
    _, q_stats = idx.query_consistent_commitments(
        {"verified_direction": "right", "nonconservation_mechanisms": []},
        task_id="q_test",
        budget=SearchBudget(1),
    )
    assert not (set(q_stats["not_retrieved_ids"]) & set(q_stats["empirically_refuted_ids"]))
    assert not (set(q_stats["budget_pending_ids"]) & set(q_stats["empirically_refuted_ids"]))

    # Revoke parent -> dependent child becomes stale, independent child survives!
    lib.meta_priors[0]["status"] = "revoked"
    comms_after, _, _ = ensure_library_commitment_index(
        lib, budget=SearchBudget(80_000), tolerance=1e-7
    )
    surviving_srcs = {
        s for c in comms_after for s in (c.get("source_concept_ids") or ())
    }
    assert "dep_child" not in surviving_srcs
    assert "indep_child" in surviving_srcs


def test_a2_a3_archive_single_charge_ordinary_cache_and_resource_account() -> None:
    priors = _sample_fixture_priors()
    # All 4 generated priors are tagged synthetic_stress_fixture -> 0 empirical confirmation sources!
    assert count_empirical_confirmation_sources(priors)["empirical_episode"] == 0
    priors_emp = deepcopy(priors)
    priors_emp[0]["evidence_kind"] = "empirical_episode"
    priors_emp[0]["empirical_confirmation"] = True
    assert count_empirical_confirmation_sources(priors_emp)["empirical_episode"] == 1

    # OrdinaryPersistentCommitmentCache invalidates cleanly on semantic edit
    ord_cache = OrdinaryPersistentCommitmentCache()
    c1, st1 = ord_cache.get_or_consolidate(priors, budget=SearchBudget(80_000))
    assert st1["cache_hit"] is False and len(c1) >= 3
    _, st2 = ord_cache.get_or_consolidate(priors, budget=SearchBudget(80_000))
    assert st2["cache_hit"] is True
    priors_mut = deepcopy(priors)
    priors_mut[0]["support_witness_rows"][0]["target"] += 5.0
    _, st3 = ord_cache.get_or_consolidate(priors_mut, budget=SearchBudget(80_000))
    assert st3["cache_hit"] is False

    # Archive episode cache installation
    lib_arch = KnowledgeLibrary(meta_priors=deepcopy(priors))
    reconstruct_commitments_from_raw_archive(
        lib_arch.meta_priors, SearchBudget(80_000), install_into_library=lib_arch
    )
    assert getattr(lib_arch, "_episode_archive_cache", None) is not None

    # UnifiedResourceAccount fallback denial & uninvoked ConsumptionEvent check
    acc = UnifiedResourceAccount(compute_limit=500, observation_limit=24, measurement_limit=240)
    assert acc.request_compute("main", 400) is True
    assert acc.request_compute("fallback", 200) is False
    assert acc.exceeded_or_denied is True

    tel = build_action_chain_telemetry(
        arm="structured",
        task_id="t0",
        knowledge_available_count=2,
        knowledge_retrieved_ids=["c0"],
        confirmed=False,
        heldout_mse=0.2,
        consumption_events=[],
    )
    assert tel["knowledge_utility_status"] == "retained_knowledge_not_invoked"
    assert tel["changed_action_or_probe"] is False

    # Provisional Tier-0 Occam check on positive-donor-only transitions
    lab_p = UnifiedObjectLab(seed=8801, objects=4, relation_direction="right", relation_radius=0.95)
    sc_p = json.loads(lab_p.spec_for(("fine",)).context["object_schema"])
    pts = []
    for k in range(6):
        pt = {}
        for io, g in enumerate(sc_p["objects"]):
            cm = sc_p["controls"][g]
            pt[cm["q"]] = (0.55 + 0.06 * k) if io == 0 else 0.10
            pt[cm["x"]] = 0.0 if io == 0 else (0.35 if io == 1 else 1.9)
            pt[cm["u"]] = 0.0
        pt[sc_p["focus_control"]] = float(k % 2)
        pts.append(pt)
    batch = lab_p.acquire(Probe("pos_only", tuple(pts)), ("fine",))
    recs = [
        {"controls": dict(p), **tr, "target": obs.target}
        for p, obs, tr in zip(pts, batch.observations, batch.transitions)
    ]
    lab_p.close()
    q_s = Program("arg", name="edge_source_quantity")
    cond_d = Program("lt", children=(Program("const", value=0.0), q_s)).to_dict()
    dyn_f = [sc_p["fields"][g]["q"] for g in sc_p["objects"]]
    pos_c = {
        sc_p["fields"][g]["x"]: {
            "control": sc_p["controls"][g]["x"],
            "quantities": list(dyn_f),
        }
        for g in sc_p["objects"]
    }
    aud = audit_transition_premises(
        {
            "complete": True,
            "schema": sc_p,
            "dynamic_fields": dyn_f,
            "position_candidates": pos_c,
            "groups": [sc_p["fields"][g] for g in sc_p["objects"]],
        },
        recs,
        SearchBudget(15_000),
        commitments=[
            {"commitment_id": "c_cond", "radius_parameters": [{"symbol": "r_0", "direction": "right", "edge_condition": cond_d}]},
            {"commitment_id": "c_uncond", "radius_parameters": [{"symbol": "r_0", "direction": "right", "edge_condition": "unconditional"}]},
        ],
        occam_premise_minimality=True,
    )
    assert aud["occam_minimality_gate"]["selection_epistemic_status"] == "provisional_occam_preference"
    assert aud["occam_minimality_gate"]["boundary_test_needed"] is True


def test_b1_b2_b6_role_unit_alignment_observable_scope_and_non_equivalence() -> None:
    priors = _sample_fixture_priors()
    comm0 = consolidate_world_structure_commitments([priors[0]], budget=SearchBudget(20_000))[0]
    sc = ObjectSchema.from_dict(priors[0]["object_schema"])
    sc_ren = sc.remap_features({f: f"renamed_{i}" for i, f in enumerate(sc.feature_names)})

    base_claim = {
        "claim_id": "H0_base",
        "commitment": deepcopy(comm0),
        "radius_value": 0.92,
        "coefficients": [0.0, 1.0, 1.0, 0.7, -0.7],
    }
    scaled_claim = {
        "claim_id": "H0_scaled_2x_space",
        "commitment": deepcopy(comm0),
        "radius_value": 1.84,
        "coefficients": [0.0, 1.0, 1.0, 0.7, -0.7],
    }
    al = align_semantics_roles_units_and_time(
        scaled_claim,
        [base_claim],
        target_schema=sc_ren,
        new_unit_scales={"q_scale": 1.0, "x_scale": 0.5, "y_scale": 1.0},
    )
    lab = UnifiedObjectLab(seed=7101, objects=4, readout="window_total_event")
    spec = lab.spec
    lab.close()

    diff_eq = find_feasible_prediction_difference(
        al["aligned_new"], al["compatible_candidates"][0]["aligned"], spec, sc_ren
    )
    assert diff_eq["proved_absent"] is True

    # B6: Finite-sample coincidence must NOT report proved_absent when ASTs differ!
    cutoff_claim = deepcopy(base_claim)
    cutoff_claim["claim_id"] = "H_cutoff_neg99"
    cutoff_claim["commitment"]["radius_parameters"][0]["edge_condition"] = Program(
        "lt", children=(Program("const", value=-99.0), Program("arg", name="edge_source_quantity"))
    ).to_dict()
    al_b6 = align_semantics_roles_units_and_time(cutoff_claim, [base_claim], target_schema=sc_ren)
    diff_b6 = find_feasible_prediction_difference(
        al_b6["aligned_new"], al_b6["compatible_candidates"][0]["aligned"], spec, sc_ren
    )
    assert diff_b6["proved_absent"] is False

    # B2: Forbidden non-observable scope guard rejected, unknown overlap never defaults to disjoint!
    bad_guard = Program(
        "lt", children=(Program("const", value=0.0), Program("feature", name="hidden_task_id"))
    )
    ok_bad, _ = validate_observable_scope_guard(bad_guard)
    assert ok_bad is False
    with pytest.raises(ValueError):
        assess_scope_overlap({"scope_guard": bad_guard.to_dict()}, {"scope_guard": None})

    good_guard = Program(
        "lt",
        children=(Program("const", value=0.2), Program("feature", name="obs_min_active_gap_dq")),
    )
    ov_unk = assess_scope_overlap({"scope_guard": good_guard.to_dict()}, {"scope_guard": None})
    assert ov_unk["certified_disjoint"] is False
    assert ov_unk["unknown"] is True


def test_b3_b4_b5_discriminating_probe_refuted_newcomer_and_scope_shrinking() -> None:
    lab = UnifiedObjectLab(
        seed=7201,
        objects=4,
        readout="window_total_event",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.25,
        relation_radius=0.95,
    )
    sc = ObjectSchema.from_dict(json.loads(lab.spec_for(("fine",)).context["object_schema"]))
    priors = _sample_fixture_priors()
    comm_uncond = consolidate_world_structure_commitments([priors[0]], budget=SearchBudget(20_000))[0]
    comm_dpg = consolidate_world_structure_commitments([priors[1]], budget=SearchBudget(20_000))[0]

    # Fit exact response coefficients on real lab rows for dpg(0.25)
    b_fit = lab.acquire(lab.random_probe(7202, length=32), ("fine",))
    q_s = Program("arg", name="edge_source_quantity")
    q_d = Program("arg", name="edge_target_quantity")
    pred_25 = Program(
        "and",
        children=(
            Program("lt", children=(Program("const", value=0.0), q_s)),
            Program("lt", children=(Program("const", value=0.25), Program("sub", children=(q_s, q_d)))),
        ),
    )
    comm_dpg25 = deepcopy(comm_dpg)
    comm_dpg25["radius_parameters"][0]["edge_condition"] = pred_25.to_dict()
    terms_25 = instantiate_commitment_terms(
        comm_dpg25, sc, radius_bindings={"r_0": 0.95}, predicate_overrides={"r_0": pred_25}
    )
    assert terms_25 is not None
    cols = [[float(t.predict(r)) for r in b_fit.observations] for t in terms_25]
    coeffs = _linear_fit(cols, [float(r.target) for r in b_fit.observations])

    h0_old = {
        "claim_id": "H0_uncond",
        "commitment": comm_uncond,
        "radius_value": 0.95,
        "coefficients": list(coeffs),
        "status": "universal",
    }
    h1_new = {
        "claim_id": "H1_dpg25",
        "commitment": comm_dpg25,
        "radius_value": 0.95,
        "coefficients": list(coeffs),
        "status": "universal",
    }

    al = align_semantics_roles_units_and_time(h1_new, [h0_old], target_schema=sc)
    diff = find_feasible_prediction_difference(
        al["aligned_new"],
        al["compatible_candidates"][0]["aligned"],
        lab.spec,
        sc,
        controls_by_object=sc.controls,
    )
    assert diff["witness_available"] is True

    paid_b = lab.acquire(diff["probe"], ("fine",))
    # B5: H0 fails on negative/sub-threshold probe while H1 succeeds -> H0 shrinks to observable scope!
    adj_b5 = adjudicate_branches_on_paid_evidence(
        h1_new,
        h0_old,
        al["aligned_new"],
        al["compatible_candidates"][0]["aligned"],
        paid_b.observations,
        paid_b.transitions,
        sc,
        old_support_regime_features=[
            {
                "obs_unconditional_block_rate": 0.0,
                "obs_min_active_donor_q": 0.55,
                "obs_min_active_gap_dq": 0.55,
            }
        ],
    )
    assert adj_b5["verdict"] == "old_universal_shrunk_and_conditional_branch_retained"
    assert adj_b5["updated_old_claim"]["status"] == "scoped_conditional"
    ok_g, _ = validate_observable_scope_guard(adj_b5["updated_old_claim"]["scope_guard"])
    assert ok_g is True

    # B4: Reverse roles (propose false H0_uncond against already-supported H1_dpg25) -> newcomer refuted, H1 preserved!
    adj_b4 = adjudicate_branches_on_paid_evidence(
        h0_old,
        h1_new,
        al["compatible_candidates"][0]["aligned"],
        al["aligned_new"],
        paid_b.observations,
        paid_b.transitions,
        sc,
    )
    assert adj_b4["verdict"] == "new_claim_refuted_old_preserved"
    assert adj_b4["updated_new_claim"]["status"] == "refuted"
    assert adj_b4["updated_old_claim"]["status"] == "universal"
    lab.close()


def test_b7_b8_b9_anti_unification_merge_false_merge_rejection_and_cycle_grounding() -> None:
    priors = _sample_fixture_priors()
    # Extract real consistent rows from fixture priors p_right_dpg22, p_right_dpg46, p_left_dpg25
    from hegel_machine.cognition.premise_commitment_index import extract_archive_witness_maps

    sup_map, _ = extract_archive_witness_maps(priors)
    c_22 = consolidate_world_structure_commitments([priors[1]])[0]
    c_46 = consolidate_world_structure_commitments([priors[2]])[0]
    c_left = consolidate_world_structure_commitments([priors[3]])[0]

    rec_22 = {
        "claim_id": "H1_22",
        "commitment": c_22,
        "schema": priors[1]["object_schema"],
        "support_rows": sup_map["p_right_dpg22"],
        "scoped_counterexamples": [],
    }
    rec_46 = {
        "claim_id": "H2_46",
        "commitment": c_46,
        "schema": priors[2]["object_schema"],
        "support_rows": sup_map["p_right_dpg46"],
        "scoped_counterexamples": [],
    }
    # B7: Valid parametric anti-unification
    good = propose_anti_unified_generalization([rec_22, rec_46], budget=SearchBudget(30_000))
    assert good["accepted"] is True
    assert good["parameter_slots"] == ["theta_0"]

    # B8: False merge with minority-weighted left branch MUST be rejected!
    rec_left = {
        "claim_id": "H_left_25",
        "commitment": c_left,
        "schema": priors[3]["object_schema"],
        "support_rows": sup_map["p_left_dpg25"][:4],
        "scoped_counterexamples": [],
    }
    bad = propose_anti_unified_generalization([rec_22, rec_46, rec_left], budget=SearchBudget(30_000))
    assert bad["accepted"] is False
    assert bad["status"] == "rejected_false_merge"
    assert "H_left_25" in bad["restored_branch_ids"]

    # B9: Premise retraction & ungrounded mutual cycle rejection
    checked = propagate_premise_retraction_invalidation(
        [
            {"claim_id": "P", "independent_empirical_support": True, "depends_on_claim_ids": []},
            {"claim_id": "C_dep", "independent_empirical_support": False, "depends_on_claim_ids": ["P"]},
            {"claim_id": "C_ind", "independent_empirical_support": True, "depends_on_claim_ids": ["P"]},
            {"claim_id": "Loop1", "independent_empirical_support": False, "depends_on_claim_ids": ["Loop2"]},
            {"claim_id": "Loop2", "independent_empirical_support": False, "depends_on_claim_ids": ["Loop1"]},
        ],
        retracted_premise_or_claim_ids={"P"},
    )
    by_id = {c["claim_id"]: c["support_validity"] for c in checked}
    assert by_id["P"] == "directly_retracted"
    assert by_id["C_dep"] == "invalidated_by_upstream_retraction"
    assert by_id["C_ind"] == "valid_grounded"
    assert by_id["Loop1"] == "invalid_ungrounded_cycle"
    assert by_id["Loop2"] == "invalid_ungrounded_cycle"


def test_c_runtime_method_induction_exploration_archive_and_consumption_events() -> None:
    # Verify zero private `lab._` accesses in all new v11 runtime cognition modules!
    for mod_rel in (
        "src/hegel_machine/cognition/branch_reconciliation.py",
        "src/hegel_machine/cognition/discriminating_experiment_policy.py",
        "src/hegel_machine/cognition/premise_commitment_index.py",
    ):
        txt = (REPO_ROOT / mod_rel).read_text(encoding="utf-8")
        assert "lab._" not in txt and "_evaluation_schema" not in txt

    lab = UnifiedObjectLab(seed=7301, objects=5, readout="window_total_event")
    sc = ObjectSchema.from_dict(json.loads(lab.spec_for(("fine",)).context["object_schema"]))
    cmd_spec = lab.spec_for(())
    lab.close()

    cands = build_generic_schema_candidate_hypotheses(sc)
    menu = generate_candidate_probe_menu(cmd_spec, sc, sc.controls, seed=7301, probe_length=6)
    rollouts = []
    for idx, p in enumerate(menu):
        fts = compute_observable_probe_features(p, cands, cmd_spec, sc, sc.controls)
        # Simulate dev measurements favoring high disagreement + close pair sparsity
        elim = (
            0.35 * fts["pairwise_disagreement_margin"]
            + 0.25 * fts["control_sparsity"]
            + 0.15 * fts["sign_contrast"]
            if fts["displacement_scale"] < 0.45
            else 0.01
        )
        rollouts.append(
            {
                "episode_id": f"dev_{idx % 3}",
                "probe_id": p.probe_id,
                "probe_features": fts,
                "empirical_elimination_per_row": elim,
            }
        )

    def _mock_sel_eval(m_spec: dict) -> dict:
        if m_spec.get("mode") == "default_space_filling_first":
            return {"passes": 1, "mean_observations": 120.0, "mean_compute": 40000.0}
        w_d = float((m_spec.get("feature_weights") or {}).get("pairwise_disagreement_margin", 0.0))
        return {
            "passes": 3 if w_d > 0.0 else 1,
            "mean_observations": 96.0 if w_d > 0.0 else 120.0,
            "mean_compute": 35000.0,
        }

    ind = induce_discriminating_experiment_method(
        rollouts, _mock_sel_eval, generation_tag="M1", budget=SearchBudget(10_000)
    )
    assert ind["promoted"] is True
    m1 = ind["promoted_method"]
    assert m1["author_origin"] == "runtime_induced_from_dev_episodes"
    assert len(ind["exploration_archive"]) >= 3

    chosen_p, meta = select_probe_under_method(
        m1, menu, cands, cmd_spec, sc, sc.controls, round_index=0, budget=SearchBudget(5000)
    )
    assert meta["changed_probe_vs_m0"] is True
    assert meta["consumption_event"] is not None
    assert meta["consumption_event"]["probe_before"] != meta["consumption_event"]["probe_after"]
    assert chosen_p.probe_id == meta["consumption_event"]["probe_after"]

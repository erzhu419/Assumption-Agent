"""Unit and behavioral tests for v10.3 PremiseCommitmentIndex and SensorRoleCalibrationPrior."""

from __future__ import annotations

from copy import deepcopy

from scripts.run_v10_3_library_scaling_study import _make_confirmed_regime_prior
from scripts.run_v10_phase2b_open_grammar_llm import (
    _acquire_shared_discovery_bundle,
    _build_parent_commitment_from_empty_library,
    _eval_program_heldout_48,
)
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.continuum_field_lab import ContinuumFieldLab
from hegel_machine.cognition.equal_tool_attribution import (
    build_bounded_open_grammar_patch_catalog,
    run_single_factor_dependency_ablation_arm,
)
from hegel_machine.cognition.evaluation import initial_prediction, loss
from hegel_machine.cognition.language import Program
from hegel_machine.cognition.memory import KnowledgeLibrary
from hegel_machine.cognition.models import ResearchConfig, SearchBudget
from hegel_machine.cognition.premise_commitment_index import ensure_library_commitment_index
from hegel_machine.cognition.sensor_role_calibration import consolidate_sensor_role_calibration_priors
from hegel_machine.cognition.unified_research import UnifiedResearchAgent


def test_premise_commitment_index_sublinear_lookup_and_incremental_cache():
    """Verify O(1) cache hit on unchanged priors, predicate deduplication, and narrow bucket lookup."""
    catalog = build_bounded_open_grammar_patch_catalog()
    ex_preds = [item["edge_predicate"] for item in catalog["P_exchange_form"]]

    priors = []
    for idx in range(16):
        d_dir = ("right", "left", "symmetric")[idx % 3]
        pred_d = ex_preds[idx % 4]  # intentionally repeat 4 predicates across 16 priors!
        rd = ("total_event", "window_total_event")[idx % 2]
        var = "sink" if idx == 15 else "exchange"
        priors.append(
            _make_confirmed_regime_prior(
                concept_id=f"unit_p_{idx:02d}",
                seed=930000 + 50 * idx,
                objects=4,
                readout=rd,
                direction=d_dir,
                edge_pred_dict=pred_d,
                radius=0.95,
                variant=var,
            )
        )

    lib = KnowledgeLibrary(meta_priors=priors)
    b1 = SearchBudget(200000)
    comms1, idx1, spent1 = ensure_library_commitment_index(lib, budget=b1, tolerance=1e-7)
    assert len(comms1) >= 8
    assert spent1 > 100

    # Second call with unchanged priors MUST hit O(1) cache (cost == 1)!
    b2 = SearchBudget(100)
    comms2, idx2, spent2 = ensure_library_commitment_index(lib, budget=b2, tolerance=1e-7)
    assert spent2 == 1
    assert len(comms2) == len(comms1)

    # Deduplicated audit commitments only keep 1 representative per distinct custom predicate (+ direction)!
    dedup = idx2.deduplicated_audit_commitments()
    assert len(dedup) <= 7 < len(comms2)

    # Querying for conservative + 'right' + ex_preds[0] inspects only the matching bucket!
    mock_audit = {
        "status": "verified_conditional_exchange",
        "conservation_supported": True,
        "nonconservation_mechanisms": [],
        "verified_direction": "right",
        "supported_directions": ["right"],
        "verified_edge_condition": ex_preds[0],
        "scientific_revision_triggers": [],
    }
    q_b = SearchBudget(1000)
    matched, stats = idx2.query_consistent_commitments(mock_audit, task_id="fresh_target_task", budget=q_b)
    assert len(matched) >= 1
    assert stats["candidates_inspected"] < stats["total_indexed_commitments"]


def test_sensor_role_calibration_prior_empty_guard_and_n7_transfer():
    """Empty library produces zero calibration priors; learned N=4 prior unlocks N=7 anonymous transfer."""
    assert consolidate_sensor_role_calibration_priors(KnowledgeLibrary()) == []

    base_lib, base_comm, _ = _build_parent_commitment_from_empty_library(
        readout="window_total_event", s0_seed=600401, s1_seed=600402
    )
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
    ) as lab4:
        b4 = _acquire_shared_discovery_bundle(lab4, seed=700401)
        fb4 = initial_prediction(b4["training"], b4["spec"].risk_quantile)
        og4 = run_single_factor_dependency_ablation_arm(
            parent_commitment=base_comm,
            model=b4["model"],
            records=b4["records"],
            spec=b4["spec"],
            training=b4["training"],
            validation=b4["validation"],
            compute_budget_limit=200000,
            use_support_dependency_cue=True,
            tolerance=1e-7,
        )
        best4 = og4["best_entry"]
        mse4 = _eval_program_heldout_48(lab4, best4["candidate"].program, fb4, 1700404)
        src_lib = deepcopy(base_lib)
        saver = UnifiedResearchAgent(
            ResearchConfig(seed=700401, world_structure_commitments=True, indexed_commitment_store=True),
            library=src_lib,
        )
        saver._object_schemas[("fine",)] = deepcopy(best4["resolved_schema"])
        saver.remember_confirmed_selection(
            best4,
            saver._spec(lab4, ("fine",)),
            b4["training"],
            b4["validation"],
            confirmed=True,
            confirmation_loss=float(mse4),
            confirmation_rows=48,
            selected_channels=("fine",),
        )

    assert len(consolidate_sensor_role_calibration_priors(src_lib)) == 1

    cfg7 = ResearchConfig(
        seed=827001,
        observation_budget=360,
        compute_budget=420000,
        minimum_improvement=1e-7,
        max_candidates=4,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        expanded_premise_revisions=True,
        active_threshold_probing=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
    )
    ag7 = UnifiedResearchAgent(cfg7, measurement_budget=9600, library=deepcopy(src_lib))
    with AnonymousObjectLab(
        seed=827001,
        apparatus_seed=827011,
        objects=7,
        readout="window_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.22,
        relation_radius=1.15,
        actuator_curvature=0.14,
        name_seed=827101,
        wiring_seed=827201,
    ) as lab7:
        res7 = ag7.run(lab7)
        assert res7["status"] == "completed"
        prog7 = Program.from_dict(res7["program"])
        ch7 = tuple(res7["selected_channels"])
        mse7 = float(loss(prog7, lab7.evaluation_rows(ch7, seed=9827001, length=48, trajectories=1), res7["fallback"], 0.5))
        assert mse7 <= 1e-7


def test_sensor_role_calibration_prior_falsified_on_anchored_and_skipped_on_continuum():
    """Verify prior self-refutes on local_direct_effect ('anchored') and skips on position-free PDE."""
    _, _ = _build_parent_commitment_from_empty_library(
        readout="total_event", s0_seed=600411, s1_seed=600412
    )[:2]
    src_lib = KnowledgeLibrary(
        meta_priors=[
            _make_confirmed_regime_prior(
                concept_id="cal_src_0",
                seed=940001,
                objects=4,
                readout="total_event",
                direction="right",
                edge_pred_dict=None,
                radius=1.0,
            )
        ]
    )

    cfg = ResearchConfig(
        seed=941001,
        observation_budget=260,
        compute_budget=280000,
        minimum_improvement=1e-7,
        use_memory=True,
        world_structure_commitments=True,
        transition_premise_audit=True,
        continuum_field_discovery=True,
        indexed_commitment_store=True,
        sensor_role_calibration_priors=True,
    )
    ag_anc = UnifiedResearchAgent(cfg, measurement_budget=7800, library=deepcopy(src_lib))
    with AnonymousObjectLab(
        seed=941001,
        apparatus_seed=941011,
        objects=5,
        readout="focus",
        variant="anchored",
        relation_direction="right",
        relation_radius=1.05,
        name_seed=941101,
        wiring_seed=941201,
    ) as lab_anc:
        res_anc = ag_anc.run(lab_anc)
    assert any(
        a.get("action") in {"skip_sensor_role_calibration_prior", "refute_sensor_role_calibration_prior"}
        for a in res_anc["actions"]
    )
    assert not any(a.get("action") == "apply_sensor_role_calibration_prior" for a in res_anc["actions"])

    ag_sink = UnifiedResearchAgent(cfg, measurement_budget=7800, library=deepcopy(src_lib))
    with AnonymousObjectLab(
        seed=941005,
        apparatus_seed=941015,
        objects=5,
        readout="focus",
        variant="sink",
        relation_direction="right",
        relation_radius=1.05,
        name_seed=941105,
        wiring_seed=941205,
    ) as lab_sink:
        res_sink = ag_sink.run(lab_sink)
    assert any(a.get("action") == "refute_sensor_role_calibration_prior" for a in res_sink["actions"])
    assert not any(a.get("action") == "apply_sensor_role_calibration_prior" for a in res_sink["actions"])

    ag_pde = UnifiedResearchAgent(cfg, measurement_budget=7800, library=deepcopy(src_lib))
    with ContinuumFieldLab(seed=941002, sites=4, readout="focal_pde_event", diffusion=0.18, advection=0.07) as lab_pde:
        res_pde = ag_pde.run(lab_pde)
    assert not any(a.get("action") == "apply_sensor_role_calibration_prior" for a in res_pde["actions"])

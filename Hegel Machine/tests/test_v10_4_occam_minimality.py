"""Unit & behavioral acceptance tests for v10.4 Occam Premise-Minimality Gating (model_calls = 0)."""

from __future__ import annotations

from copy import deepcopy

from scripts.run_v10_phase2b_open_grammar_llm import _acquire_shared_discovery_bundle
from hegel_machine.cognition.anonymous_object_lab import AnonymousObjectLab
from hegel_machine.cognition.equal_tool_attribution import (
    build_bounded_open_grammar_patch_catalog,
)
from hegel_machine.cognition.models import SearchBudget
from hegel_machine.cognition.open_grammar_author import validate_edge_predicate_ast
from hegel_machine.cognition.premise_dependency_audit import (
    audit_transition_premises,
    edge_condition_predicate,
)


def _make_mock_commitments_with_all_catalog_predicates() -> list[dict]:
    catalog = build_bounded_open_grammar_patch_catalog()
    comms = []
    for idx, item in enumerate(catalog["P_exchange_form"]):
        pred_prog = validate_edge_predicate_ast(item["edge_predicate"])
        comms.append(
            {
                "commitment_id": f"mock_cat_comm_{idx}_{item['patch_id']}",
                "readout_kind": "window_total_event",
                "nonconservation_mechanism": None,
                "radius_parameters": [
                    {
                        "symbol": "r_0",
                        "direction": "right",
                        "edge_condition": pred_prog.to_dict(),
                    }
                ],
            }
        )
    comms.append(
        {
            "commitment_id": "mock_uncond_comm",
            "readout_kind": "window_total_event",
            "nonconservation_mechanism": None,
            "radius_parameters": [
                {
                    "symbol": "r_0",
                    "direction": "right",
                    "edge_condition": "unconditional",
                }
            ],
        }
    )
    return comms


def _acquire_lab_model_and_records(lab: AnonymousObjectLab, seed: int = 101):
    bundle = _acquire_shared_discovery_bundle(lab, seed=seed)
    return bundle["model"], bundle["records"]


def test_occam_tier0_unconditional_skips_13_custom_predicates_and_prevents_overfitting() -> None:
    """When unconditional exchange explains 100% of transitions, Occam Tier 0 skips all 13 custom predicates."""
    comms = _make_mock_commitments_with_all_catalog_predicates()
    with AnonymousObjectLab(
        seed=915506,
        apparatus_seed=915056,
        objects=5,
        readout="window_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="always",
        relation_radius=1.18,
        actuator_curvature=0.14,
        name_seed=915606,
        wiring_seed=915706,
    ) as lab:
        model, records = _acquire_lab_model_and_records(lab, seed=915506)

    b_no_occam = SearchBudget(400000)
    res_no_occam = audit_transition_premises(
        model,
        records,
        b_no_occam,
        commitments=comms,
        tolerance=1e-7,
        occam_premise_minimality=False,
    )

    b_with_occam = SearchBudget(400000)
    res_with_occam = audit_transition_premises(
        model,
        records,
        b_with_occam,
        commitments=comms,
        tolerance=1e-7,
        occam_premise_minimality=True,
    )

    # Without Occam minimality, parameterized threshold predicates in the 13-predicate catalog
    # calibrate their cutoff below observed values and hijack unconditional exchange!
    assert isinstance(res_no_occam["verified_edge_condition"], dict)

    # With Occam minimality, Tier 0 accepts unconditional exchange immediately and skips all 13 custom predicates.
    assert res_with_occam["status"] == "verified_unconditional_exchange"
    assert res_with_occam["verified_edge_condition"] == "unconditional"
    gate = res_with_occam["occam_minimality_gate"]
    assert gate["enabled"] is True
    assert gate["unconditional_sufficient"] is True
    assert gate["selected_tier"] == "tier_0_unconditional"
    assert gate["skipped_custom_predicates"] == 13
    assert gate["screened_calibrated_custom_predicates"] == 0
    assert res_with_occam["compute_cost"] * 3 < res_no_occam["compute_cost"]


def test_occam_tier1_preserves_exact_subset_predicate_sensitivity_under_adversarial_order() -> None:
    """When donor_positive_gap (unshifted threshold 0.20) blocks edges, Tier 1 finds exact predicate before Tier 3."""
    comms = _make_mock_commitments_with_all_catalog_predicates()
    # Move donor_positive_gap to the very tail after curved hypersurface & distraction predicates
    dpg_comms = [c for c in comms if "donor_positive_gap" in c["commitment_id"]]
    other_comms = [c for c in comms if "donor_positive_gap" not in c["commitment_id"]]
    comms_adv = other_comms + dpg_comms

    with AnonymousObjectLab(
        seed=915501,
        apparatus_seed=915051,
        objects=5,
        readout="window_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.20,
        relation_radius=1.16,
        actuator_curvature=0.15,
        name_seed=915601,
        wiring_seed=915701,
    ) as lab:
        model, records = _acquire_lab_model_and_records(lab, seed=915501)

    res = audit_transition_premises(
        model,
        records,
        SearchBudget(400000),
        commitments=comms_adv,
        tolerance=1e-7,
        occam_premise_minimality=True,
    )
    assert res["status"] == "verified_unconditional_exchange"
    assert isinstance(res["verified_edge_condition"], dict)
    gate = res["occam_minimality_gate"]
    assert gate["enabled"] is True
    assert gate["unconditional_sufficient"] is False
    assert gate["selected_tier"] == "tier_1_exact_committed_predicate"
    assert gate["screened_calibrated_custom_predicates"] == 0


def test_occam_tier2_atomic_comparator_and_tier3_shifted_threshold_calibration() -> None:
    """Tier 2 recovers source_greater before parameter fitting; Tier 3 calibrates shifted threshold when needed."""
    comms = _make_mock_commitments_with_all_catalog_predicates()
    sg_prog = edge_condition_predicate("source_greater")
    comms_no_sg = [
        c
        for c in comms
        if c["radius_parameters"][0]["edge_condition"] != sg_prog.to_dict()
    ]

    with AnonymousObjectLab(
        seed=916401,
        apparatus_seed=916041,
        objects=4,
        readout="window_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="source_greater",
        relation_radius=1.08,
        name_seed=916501,
        wiring_seed=916601,
    ) as lab_sg:
        m_sg, r_sg = _acquire_lab_model_and_records(lab_sg, seed=916401)

    res_sg = audit_transition_premises(
        m_sg,
        r_sg,
        SearchBudget(300000),
        commitments=comms_no_sg,
        tolerance=1e-7,
        occam_premise_minimality=True,
    )
    assert res_sg["status"] == "verified_conditional_exchange"
    assert res_sg["verified_edge_condition"] == "source_greater"
    assert res_sg["occam_minimality_gate"]["selected_tier"] == "tier_2_atomic_comparator"
    assert res_sg["occam_minimality_gate"]["screened_calibrated_custom_predicates"] == 0

    # Tier 3: True world is donor_positive_gap with shifted threshold 0.36 (committed catalog has 0.20)
    dpg_comm = [c for c in comms if "donor_positive_gap" in c["commitment_id"]]
    with AnonymousObjectLab(
        seed=917401,
        apparatus_seed=917041,
        objects=4,
        readout="window_total_event",
        variant="exchange",
        relation_direction="right",
        relation_condition="donor_positive_gap",
        condition_threshold=0.36,
        relation_radius=1.08,
        name_seed=917501,
        wiring_seed=917601,
    ) as lab_th:
        m_th, r_th = _acquire_lab_model_and_records(lab_th, seed=917401)

    res_th = audit_transition_premises(
        m_th,
        r_th,
        SearchBudget(300000),
        commitments=dpg_comm,
        tolerance=1e-7,
        occam_premise_minimality=True,
    )
    assert res_th["status"] == "verified_unconditional_exchange"
    assert res_th["occam_minimality_gate"]["selected_tier"] == "tier_3_calibrated_custom_predicate"
    lo, hi = res_th["verified_threshold_interval"]
    assert 0.22 < lo <= hi

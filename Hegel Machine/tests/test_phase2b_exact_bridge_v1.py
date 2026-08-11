import ast
import inspect
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

import hegel_machine.phase2b_exact_bridge_v1 as exact_bridge
from hegel_machine.bootstrap import initial_theory
from hegel_machine.phase2_exit import HARD_NEGATIVE_OBSERVABLES, PASS_OBSERVABLES
from hegel_machine.phase2b_adapter import (
    AdapterEnumerationResult,
    ObservableChannelBinding,
    Phase2BAdapterRegistry,
)
from hegel_machine.phase2b_exact_bridge_v1 import (
    DEFAULT_EXACT_BRIDGE_POLICY,
    ExactBridgeDisposition,
    ExactCandidateStatus,
    ExactSelectionDisposition,
    run_exact_rational_bridge,
)
from hegel_machine.phase2b_wire import (
    NumericValue,
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    PublicEvidenceBundle,
)
from hegel_machine.schema import LawKind


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def bridge_inputs():
    theory = initial_theory()
    role_keys = tuple(
        (law.law_id, role)
        for law in theory.relation_laws
        for role in law.roles
    )
    role_ids = {key: uid(100 + index) for index, key in enumerate(role_keys)}
    entity_ids = {
        (key, variant): uid(300 + 2 * index + variant)
        for index, key in enumerate(role_keys)
        for variant in (0, 1)
    }
    observable_names = tuple(
        sorted(
            {
                observable
                for law in theory.relation_laws
                for observable in law.required_observables
            }
        )
    )
    quantity_ids = {
        observable: uid(500 + index)
        for index, observable in enumerate(observable_names)
    }
    family_ids = {
        kind: uid(800 + index) for index, kind in enumerate(LawKind)
    }
    registry = Phase2BAdapterRegistry.from_theory(
        theory,
        family_ids=family_ids,
        role_ids=role_ids,
        quantity_ids=quantity_ids,
    )

    observations = []
    observation_index = 1000
    for law in theory.relation_laws:
        for observable_name in law.required_observables:
            witness_roles = tuple(
                role
                for role, names in law.role_observable_requirements
                if observable_name in names
            )
            if not witness_roles:
                witness_roles = law.roles
            bound_wire_roles = sorted(
                role_ids[(law.law_id, role)] for role in witness_roles
            )
            for variants in product((0, 1), repeat=len(witness_roles)):
                canonical_witness = all(variant == 0 for variant in variants)
                payload = (
                    PASS_OBSERVABLES[law.kind]
                    if law.kind is LawKind.SYMMETRY and canonical_witness
                    else HARD_NEGATIVE_OBSERVABLES[law.kind]
                )
                if law.kind is LawKind.LOCALITY:
                    payload = {
                        **payload,
                        "conditional_a": (1.0, 0.0),
                        "conditional_b": (0.0, 1.0),
                    }
                raw_value = payload[observable_name]
                bound_entities = sorted(
                    entity_ids[((law.law_id, role), variant)]
                    for role, variant in zip(
                        witness_roles,
                        variants,
                        strict=True,
                    )
                )
                if type(raw_value) is bool:
                    value = {"kind": "boolean", "value": raw_value}
                    uncertainty = {"model": "not_applicable", "radius": []}
                else:
                    values = (
                        list(raw_value)
                        if isinstance(raw_value, (tuple, list))
                        else [raw_value]
                    )
                    if not values:
                        values = [0.0]
                    value = {"kind": "numeric", "values": values}
                    uncertainty = {
                        "model": "absolute_bound",
                        "radius": [0.0 for _ in values],
                    }
                observations.append(
                    {
                        "observation_id": uid(observation_index),
                        "source_channel_id": uid(990),
                        "entity_ids": bound_entities,
                        "role_candidate_ids": bound_wire_roles,
                        "quantity_id": quantity_ids[observable_name],
                        "value": value,
                        "unit_dimension": {
                            "si_exponents": [0, 0, 0, 0, 0, 0, 0]
                        },
                        "temporal_support": {
                            "clock_id": uid(991),
                            "start": 0.0,
                            "end": 1.0,
                        },
                        "spatial_support": None,
                        "uncertainty": uncertainty,
                        "provenance_sha256": f"{observation_index % 16:x}" * 64,
                        "missingness": "observed",
                    }
                )
                observation_index += 1

    root_scale_id = uid(900)
    second_scale_id = uid(902)
    mapping = {
        "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": uid(1),
        "entity_candidates": [
            {
                "entity_id": entity_ids[(key, variant)],
                "role_candidate_ids": [role_ids[key]],
            }
            for key in role_keys
            for variant in (0, 1)
        ],
        "role_ids": list(role_ids.values()),
        "quantity_ids": list(quantity_ids.values()),
        "observations": observations,
        "task_target": {
            "task_id": uid(2),
            "entity_ids": list(entity_ids.values()),
            "quantity_ids": list(quantity_ids.values()),
        },
        "aggregation_graph": {
            "scale_ids": [root_scale_id, second_scale_id],
            "root_scale_ids": [root_scale_id],
            "edges": [
                {
                    "source_scale_id": root_scale_id,
                    "target_scale_id": second_scale_id,
                    "transform_id": uid(901),
                }
            ],
        },
        "transform_catalog": [
            {
                "transform_id": uid(901),
                "operation": "identity",
                "parameters": [],
            }
        ],
        "missingness_mask": [],
    }
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    identities = {
        "role_ids": role_ids,
        "quantity_ids": quantity_ids,
        "family_ids": family_ids,
        "root_scale_id": root_scale_id,
        "second_scale_id": second_scale_id,
    }
    return theory, mapping, bundle, registry, identities


def set_numeric_observable(
    mapping,
    identities,
    observable_name,
    *,
    values,
    radii,
):
    quantity_id = identities["quantity_ids"][observable_name]
    for observation in mapping["observations"]:
        if observation["quantity_id"] == quantity_id:
            observation["value"] = {
                "kind": "numeric",
                "values": list(values),
            }
            observation["uncertainty"] = {
                "model": "absolute_bound",
                "radius": list(radii),
            }


def set_boolean_observable(mapping, identities, observable_name, *, value):
    quantity_id = identities["quantity_ids"][observable_name]
    for observation in mapping["observations"]:
        if observation["quantity_id"] == quantity_id:
            observation["value"] = {"kind": "boolean", "value": value}
            observation["uncertainty"] = {
                "model": "not_applicable",
                "radius": [],
            }


def representative_completed_evaluation(run, law_kind, root_scale_id):
    assert run.compilation is not None
    return next(
        item
        for item in run.compilation.evaluations
        if item.law_kind is law_kind
        and item.scale_hypothesis_id == root_scale_id
        and item.completed
    )


def assert_non_degenerate_residual_contains(evaluation, exact_value):
    assert evaluation.residual is not None
    assert evaluation.residual.lower_fraction < evaluation.residual.upper_fraction
    assert (
        evaluation.residual.lower_fraction
        <= exact_value
        <= evaluation.residual.upper_fraction
    )


def test_authoritative_exact_bridge_selects_from_complete_six_family_grid():
    theory, _, bundle, registry, identities = bridge_inputs()
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.COMPLETE
    assert run.uncertainty_receipt is not None
    assert run.compilation is not None
    assert run.decision is not None
    assert len(run.compilation.evaluations) == 72
    assert all(item.completed for item in run.compilation.evaluations)
    assert {item.law_kind for item in run.compilation.evaluations} == set(LawKind)
    assert {
        item.law_kind
        for item in run.compilation.evaluations
        if item.status is ExactCandidateStatus.PASS
    } == {LawKind.SYMMETRY}
    assert run.decision.disposition is ExactSelectionDisposition.ADMISSIBLE_SCALE_SET
    assert run.decision.selected_law_kind is LawKind.SYMMETRY
    assert run.decision.admissible_scale_hypothesis_ids == (
        identities["root_scale_id"],
        identities["second_scale_id"],
    )
    assert run.decision.bridge_result_id == run.compilation.result_id

    expected_tolerance = Fraction.from_float(0.01)
    assert all(
        item.tolerance is not None
        and item.tolerance.lower_fraction == expected_tolerance
        and item.tolerance.upper_fraction == expected_tolerance
        for item in run.compilation.evaluations
    )
    assert all(
        item.uncertainty_result_id == run.uncertainty_receipt.result_id
        and item.candidate_grid_commitment_id
        == run.compilation.candidate_grid_commitment_id
        and item.used_observation_compilation_ids
        for item in run.compilation.evaluations
    )


def test_non_degenerate_symmetry_interval_has_proved_exact_residual_envelope():
    theory, mapping, _, registry, identities = bridge_inputs()
    forward_quantity = identities["quantity_ids"]["forward"]
    for observation in mapping["observations"]:
        if observation["quantity_id"] == forward_quantity:
            observation["uncertainty"]["radius"] = [
                0.125 for _ in observation["value"]["values"]
            ]
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.compilation is not None
    symmetry = tuple(
        item
        for item in run.compilation.evaluations
        if item.law_kind is LawKind.SYMMETRY
    )
    assert symmetry
    assert all(item.completed for item in symmetry)
    positive = next(
        item
        for item in symmetry
        if item.role_binding
        == tuple(
            sorted(
                (
                    role,
                    next(
                        entity["entity_id"]
                        for entity in mapping["entity_candidates"]
                        if identities["role_ids"][("law_symmetry_v1", role)]
                        in entity["role_candidate_ids"]
                    ),
                )
                for role in ("source", "transformed_source")
            )
        )
    )
    assert positive.residual is not None
    assert positive.residual.lower_fraction == 0
    assert positive.residual.upper_fraction == Fraction(1, 16)
    assert positive.status is ExactCandidateStatus.INCONCLUSIVE


def test_non_degenerate_monotonicity_interval_contains_exact_point_residual():
    theory, mapping, _, registry, identities = bridge_inputs()
    set_numeric_observable(
        mapping,
        identities,
        "x_low",
        values=(0.0,),
        radii=(0.25,),
    )
    set_numeric_observable(
        mapping,
        identities,
        "x_high",
        values=(2.0,),
        radii=(0.25,),
    )
    set_numeric_observable(
        mapping,
        identities,
        "y_low",
        values=(1.0,),
        radii=(0.25,),
    )
    set_numeric_observable(
        mapping,
        identities,
        "y_high",
        values=(0.0,),
        radii=(0.25,),
    )
    set_numeric_observable(
        mapping,
        identities,
        "direction",
        values=(1.0,),
        radii=(0.0,),
    )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    evaluation = representative_completed_evaluation(
        run,
        LawKind.MONOTONICITY,
        identities["root_scale_id"],
    )
    # At the exact center point, max(0, -(0 - 1)) / max(1, 1, 0) = 1.
    assert_non_degenerate_residual_contains(evaluation, Fraction(1))


def test_non_degenerate_conservation_interval_contains_exact_point_residual():
    theory, mapping, _, registry, identities = bridge_inputs()
    for observable_name, value, radius in (
        ("storage_delta", 1.0, 0.25),
        ("inflows", 2.0, 0.25),
        ("outflows", 0.0, 0.25),
        ("sources", 0.0, 0.0),
        ("sinks", 0.0, 0.0),
    ):
        set_numeric_observable(
            mapping,
            identities,
            observable_name,
            values=(value,),
            radii=(radius,),
        )
    set_boolean_observable(
        mapping,
        identities,
        "boundary_observed",
        value=True,
    )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    evaluation = representative_completed_evaluation(
        run,
        LawKind.CONSERVATION,
        identities["root_scale_id"],
    )
    # At the exact center point, abs(1 + 0 - 2 - 0 + 0) / 2 = 1/2.
    assert_non_degenerate_residual_contains(evaluation, Fraction(1, 2))


def test_non_degenerate_complementarity_interval_contains_exact_point_residual():
    theory, mapping, _, registry, identities = bridge_inputs()
    for observable_name, value, radius in (
        ("u_empty", 0.0, 0.0),
        ("u_a", 1.0, 0.0),
        ("u_b", 1.0, 0.0),
        ("u_ab", 1.0, 0.25),
        ("expected_interaction", 1.0, 0.0),
        ("interaction_margin", 1.0, 0.0),
    ):
        set_numeric_observable(
            mapping,
            identities,
            observable_name,
            values=(value,),
            radii=(radius,),
        )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    evaluation = representative_completed_evaluation(
        run,
        LawKind.COMPLEMENTARITY,
        identities["root_scale_id"],
    )
    # Center interaction is 1 - 1 - 1 + 0 = -1, hence (1 - -1) / 1 = 2.
    assert_non_degenerate_residual_contains(evaluation, Fraction(2))


def test_non_degenerate_negative_feedback_continuous_interval_contains_point():
    theory, mapping, _, registry, identities = bridge_inputs()
    for observable_name, value, radius in (
        ("disturbance_delta", 2.0, 0.25),
        ("response_delta", -1.0, 0.25),
        ("deviation_before_response", 1.0, 0.25),
        ("deviation_after_response", 0.75, 0.25),
        ("response_margin", 0.5, 0.0),
        ("mitigation_margin", 0.5, 0.0),
    ):
        set_numeric_observable(
            mapping,
            identities,
            observable_name,
            values=(value,),
            radii=(radius,),
        )
    for observable_name in (
        "controlled_quantity_observed",
        "disturbance_precedes_response",
        "system_induced_response",
        "same_controlled_quantity",
        "local_stability_window_observed",
    ):
        set_boolean_observable(
            mapping,
            identities,
            observable_name,
            value=True,
        )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    evaluation = representative_completed_evaluation(
        run,
        LawKind.NEGATIVE_FEEDBACK,
        identities["root_scale_id"],
    )
    # The center is in the continuous branch: sign residual 0, mitigation 1/4.
    assert_non_degenerate_residual_contains(evaluation, Fraction(1, 4))


def test_non_degenerate_locality_interval_contains_exact_point_residual():
    theory, mapping, _, registry, identities = bridge_inputs()
    set_numeric_observable(
        mapping,
        identities,
        "conditional_a",
        values=(0.75, 0.25),
        radii=(0.125, 0.125),
    )
    set_numeric_observable(
        mapping,
        identities,
        "conditional_b",
        values=(0.5, 0.5),
        radii=(0.125, 0.125),
    )
    set_boolean_observable(
        mapping,
        identities,
        "blanket_observed",
        value=True,
    )
    set_boolean_observable(
        mapping,
        identities,
        "same_blanket_state",
        value=True,
    )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    evaluation = representative_completed_evaluation(
        run,
        LawKind.LOCALITY,
        identities["root_scale_id"],
    )
    # Both center vectors sum to one; their exact total variation is 1/4.
    assert_non_degenerate_residual_contains(evaluation, Fraction(1, 4))


def test_nonidentity_transform_is_full_grid_error_and_exact_selector_abstains():
    theory, mapping, _, registry, identities = bridge_inputs()
    mapping["transform_catalog"][0]["operation"] = "coarse_graining"
    mapping["transform_catalog"][0]["parameters"] = [2.0]
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.compilation is not None and run.decision is not None
    assert len(run.compilation.evaluations) == 72
    errors = tuple(
        item for item in run.compilation.evaluations if not item.completed
    )
    assert len(errors) == 36
    assert all(
        item.scale_hypothesis_id == identities["second_scale_id"]
        for item in errors
    )
    assert {item.error_code for item in errors} == {
        "unsupported_transform_semantics:coarse_graining"
    }
    assert run.decision.disposition is ExactSelectionDisposition.ABSTAIN
    assert run.decision.reason == "candidate_evaluation_error"


def test_standard_error_aborts_before_any_candidate_grid():
    theory, mapping, _, registry, _ = bridge_inputs()
    numeric = next(
        item for item in mapping["observations"] if item["value"]["kind"] == "numeric"
    )
    numeric["uncertainty"]["model"] = "standard_error"
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.uncertainty_receipt is not None
    assert run.compilation is not None
    assert run.compilation.evaluations == ()
    assert run.decision is not None
    assert run.decision.disposition is ExactSelectionDisposition.ABSTAIN


def test_unused_transform_channel_fails_before_uncertainty_compilation():
    theory, mapping, _, registry, _ = bridge_inputs()
    mapping["transform_catalog"].append(
        {
            "transform_id": uid(903),
            "operation": "identity",
            "parameters": [],
        }
    )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "unused_or_missing_transform_catalog_entry"
    assert run.uncertainty_receipt is None
    assert run.compilation is None
    assert run.decision is None


def test_vector_width_resource_limit_precedes_uncertainty_and_adapter():
    theory, mapping, _, registry, _ = bridge_inputs()
    numeric = next(
        item for item in mapping["observations"] if item["value"]["kind"] == "numeric"
    )
    width = DEFAULT_EXACT_BRIDGE_POLICY.maximum_vector_width + 1
    numeric["value"]["values"] = [0.0] * width
    numeric["uncertainty"]["radius"] = [0.0] * width
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "RESOURCE_LIMIT:vector_width"
    assert run.uncertainty_receipt is None


def test_entity_count_resource_limit_precedes_adapter_materialization(monkeypatch):
    theory, mapping, _, registry, identities = bridge_inputs()
    role_id = next(iter(identities["role_ids"].values()))
    while len(mapping["entity_candidates"]) <= (
        DEFAULT_EXACT_BRIDGE_POLICY.maximum_entity_candidate_count
    ):
        index = 20_000 + len(mapping["entity_candidates"])
        mapping["entity_candidates"].append(
            {"entity_id": uid(index), "role_candidate_ids": [role_id]}
        )

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter ran before entity-count preflight")

    def forbidden_content_root(_self):
        raise AssertionError("oversized authority was hashed before preflight")

    monkeypatch.setattr(
        exact_bridge,
        "enumerate_candidate_hypotheses",
        forbidden_adapter,
    )
    monkeypatch.setattr(
        PublicEvidenceBundle,
        "content_id",
        property(forbidden_content_root),
    )
    monkeypatch.setattr(
        type(theory),
        "version_id",
        property(forbidden_content_root),
    )
    monkeypatch.setattr(
        Phase2BAdapterRegistry,
        "registry_id",
        property(forbidden_content_root),
    )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "RESOURCE_LIMIT:entity_candidate_count"
    assert run.uncertainty_receipt is None
    assert run.bundle_content_id is None
    assert not hasattr(run, "run_id")


def test_registry_binding_and_transform_catalog_caps_precede_adapter(monkeypatch):
    theory, mapping, bundle, registry, _ = bridge_inputs()

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter ran before registry/catalog preflight")

    monkeypatch.setattr(
        exact_bridge,
        "enumerate_candidate_hypotheses",
        forbidden_adapter,
    )
    original = registry.law_bindings[0]
    extra = replace(original, law_id="extra_law", family_id=uid(48_000))
    changed_registry = replace(
        registry,
        law_bindings=registry.law_bindings + (extra,),
    )
    registry_rejection = run_exact_rational_bridge(
        bundle=bundle,
        theory=theory,
        registry=changed_registry,
    )
    assert registry_rejection.reason == (
        "RESOURCE_LIMIT:registry_law_binding_count"
    )
    assert registry_rejection.uncertainty_receipt is None

    for index in range(
        len(mapping["transform_catalog"]),
        DEFAULT_EXACT_BRIDGE_POLICY.maximum_transform_catalog_count + 1,
    ):
        mapping["transform_catalog"].append(
            {
                "transform_id": uid(50_000 + index),
                "operation": "identity",
                "parameters": [],
            }
        )
    catalog_rejection = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert catalog_rejection.reason == "RESOURCE_LIMIT:transform_catalog_count"
    assert catalog_rejection.uncertainty_receipt is None


def test_adapter_scan_work_is_bounded_before_legacy_enumeration(monkeypatch):
    theory, mapping, _, registry, identities = bridge_inputs()
    symmetry_roles = tuple(
        identities["role_ids"][("law_symmetry_v1", role)]
        for role in ("source", "transformed_source")
    )
    next_id = 30_000
    for role_id in symmetry_roles:
        for _ in range(98):
            mapping["entity_candidates"].append(
                {"entity_id": uid(next_id), "role_candidate_ids": [role_id]}
            )
            next_id += 1

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter ran before scan-work preflight")

    monkeypatch.setattr(
        exact_bridge,
        "enumerate_candidate_hypotheses",
        forbidden_adapter,
    )
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "RESOURCE_LIMIT:adapter_scan_work"
    assert run.uncertainty_receipt is None


def test_extra_registry_observable_channel_is_rejected_before_adapter(monkeypatch):
    theory, _, bundle, registry, _ = bridge_inputs()
    changed_registry = replace(
        registry,
        observable_channels=registry.observable_channels
        + (ObservableChannelBinding(uid(49_999), "covert_extra"),),
    )

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter ran before registry exactness check")

    monkeypatch.setattr(
        exact_bridge,
        "enumerate_candidate_hypotheses",
        forbidden_adapter,
    )
    with pytest.raises(
        ValueError,
        match="exact bridge observable channel registry differs",
    ):
        run_exact_rational_bridge(
            bundle=bundle,
            theory=theory,
            registry=changed_registry,
        )


def test_per_candidate_exact_operation_budget_aborts_whole_bridge():
    theory, mapping, _, registry, identities = bridge_inputs()
    width = DEFAULT_EXACT_BRIDGE_POLICY.maximum_vector_width
    vector_quantities = {
        identities["quantity_ids"]["forward"],
        identities["quantity_ids"]["transformed"],
    }
    for observation in mapping["observations"]:
        if observation["quantity_id"] in vector_quantities:
            observation["value"]["values"] = [2.0] * width
            observation["uncertainty"]["radius"] = [0.0] * width
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "RESOURCE_LIMIT:exact_operation_budget"
    assert run.uncertainty_receipt is not None
    assert run.compilation is not None
    assert run.compilation.evaluations == ()


def test_raw_role_product_resource_limit_precedes_adapter_materialization():
    theory, mapping, _, registry, identities = bridge_inputs()
    negative_law = next(
        law for law in theory.relation_laws if law.kind is LawKind.NEGATIVE_FEEDBACK
    )
    next_id = 5000
    for role in negative_law.roles:
        wire_role = identities["role_ids"][(negative_law.law_id, role)]
        for _ in range(28):
            mapping["entity_candidates"].append(
                {
                    "entity_id": uid(next_id),
                    "role_candidate_ids": [wire_role],
                }
            )
            next_id += 1
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "RESOURCE_LIMIT:raw_role_binding_scale_product"
    assert run.uncertainty_receipt is None


def test_nonunique_scale_path_fails_in_bounded_topological_preflight():
    theory, mapping, _, registry, _ = bridge_inputs()
    mapping["aggregation_graph"] = {
        "scale_ids": [uid(900), uid(902), uid(904), uid(906)],
        "root_scale_ids": [uid(900)],
        "edges": [
            {
                "source_scale_id": uid(900),
                "target_scale_id": uid(902),
                "transform_id": uid(901),
            },
            {
                "source_scale_id": uid(900),
                "target_scale_id": uid(904),
                "transform_id": uid(903),
            },
            {
                "source_scale_id": uid(902),
                "target_scale_id": uid(906),
                "transform_id": uid(905),
            },
            {
                "source_scale_id": uid(904),
                "target_scale_id": uid(906),
                "transform_id": uid(907),
            },
        ],
    }
    mapping["transform_catalog"] = [
        {
            "transform_id": transform_id,
            "operation": "identity",
            "parameters": [],
        }
        for transform_id in (uid(901), uid(903), uid(905), uid(907))
    ]
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.ABSTAIN
    assert run.reason == "nonunique_transform_path"
    assert run.uncertainty_receipt is None


def test_negative_feedback_interval_crossing_zero_branch_is_error_cell():
    theory, mapping, _, registry, identities = bridge_inputs()
    response_quantity = identities["quantity_ids"]["response_delta"]
    induced_quantity = identities["quantity_ids"]["system_induced_response"]
    for observation in mapping["observations"]:
        if observation["quantity_id"] == response_quantity:
            observation["uncertainty"]["radius"] = [1.0]
        if observation["quantity_id"] == induced_quantity:
            observation["value"]["value"] = True
    run = run_exact_rational_bridge(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert run.compilation is not None
    feedback = tuple(
        item
        for item in run.compilation.evaluations
        if item.law_kind is LawKind.NEGATIVE_FEEDBACK
    )
    assert feedback and not any(item.completed for item in feedback)
    assert {item.error_code for item in feedback} == {
        "nonuniform_domain:zero_branch_boundary_crossed"
    }


def test_exact_normalized_threshold_equality_is_pass():
    theory, _, bundle, registry, _ = bridge_inputs()
    functionals = tuple(
        replace(item, tolerance=2.0)
        if item.law_kind is LawKind.SYMMETRY
        else item
        for item in theory.violation_functionals
    )
    changed_theory = replace(theory, violation_functionals=functionals)
    changed_registry = replace(
        registry,
        theory_version_id=changed_theory.version_id,
    )
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=changed_theory,
        registry=changed_registry,
    )
    assert run.compilation is not None
    equality = tuple(
        item
        for item in run.compilation.evaluations
        if item.law_kind is LawKind.SYMMETRY
        and item.normalized_interval is not None
        and item.normalized_interval.upper_fraction == 1
    )
    assert equality
    assert all(item.status is ExactCandidateStatus.PASS for item in equality)


def test_zero_tolerance_fails_closed_per_exact_candidate():
    theory, _, bundle, registry, _ = bridge_inputs()
    functionals = tuple(
        replace(item, tolerance=0.0)
        if item.law_kind is LawKind.SYMMETRY
        else item
        for item in theory.violation_functionals
    )
    changed_theory = replace(theory, violation_functionals=functionals)
    changed_registry = replace(
        registry,
        theory_version_id=changed_theory.version_id,
    )
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=changed_theory,
        registry=changed_registry,
    )
    assert run.compilation is not None
    assert len(run.compilation.evaluations) == 72
    symmetry = tuple(
        item
        for item in run.compilation.evaluations
        if item.law_kind is LawKind.SYMMETRY
    )
    assert symmetry
    assert not any(item.completed for item in symmetry)
    assert {item.error_code for item in symmetry} == {
        "strict_positive_tolerance_not_guaranteed"
    }
    assert run.decision is not None
    assert run.decision.disposition is ExactSelectionDisposition.ABSTAIN


def test_negative_tolerance_is_rejected_by_theory_schema():
    theory, _, _, _, _ = bridge_inputs()
    symmetry = next(
        item
        for item in theory.violation_functionals
        if item.law_kind is LawKind.SYMMETRY
    )
    with pytest.raises(ValueError, match="violation tolerance cannot be negative"):
        replace(symmetry, tolerance=-0.25)


def test_authoritative_entry_rejects_external_subclasses_of_all_input_types():
    theory, _, bundle, registry, _ = bridge_inputs()

    class ExternalBundle(PublicEvidenceBundle):
        pass

    class ExternalTheory(type(theory)):
        pass

    class ExternalRegistry(Phase2BAdapterRegistry):
        pass

    external_bundle = ExternalBundle(
        **{field.name: getattr(bundle, field.name) for field in fields(bundle)}
    )
    external_theory = ExternalTheory(
        **{field.name: getattr(theory, field.name) for field in fields(theory)}
    )
    external_registry = ExternalRegistry(
        **{
            field.name: getattr(registry, field.name)
            for field in fields(registry)
        }
    )

    with pytest.raises(
        TypeError,
        match="exact bridge run requires exact evidence bundle type",
    ):
        run_exact_rational_bridge(
            bundle=external_bundle,
            theory=theory,
            registry=registry,
        )
    with pytest.raises(
        TypeError,
        match="exact bridge run requires exact theory type",
    ):
        run_exact_rational_bridge(
            bundle=bundle,
            theory=external_theory,
            registry=registry,
        )
    with pytest.raises(
        TypeError,
        match="exact bridge run requires exact adapter registry type",
    ):
        run_exact_rational_bridge(
            bundle=bundle,
            theory=theory,
            registry=external_registry,
        )


def test_nested_wire_subclass_cannot_split_content_root_from_exact_values():
    theory, _, bundle, registry, _ = bridge_inputs()
    target = next(
        observation
        for observation in bundle.observations
        if type(observation.value) is NumericValue
    )
    assert type(target.value) is NumericValue

    class RootSpoofingNumericValue(NumericValue):
        def to_mapping(self):
            return NumericValue((0.0,) * len(self.values)).to_mapping()

    zero_value = NumericValue((0.0,) * len(target.value.values))
    ordinary = replace(
        bundle,
        observations=tuple(
            replace(observation, value=zero_value)
            if observation.observation_id == target.observation_id
            else observation
            for observation in bundle.observations
        ),
    )
    spoofed_value = RootSpoofingNumericValue(
        tuple(1.0 for _ in target.value.values)
    )
    spoofed = replace(
        bundle,
        observations=tuple(
            replace(observation, value=spoofed_value)
            if observation.observation_id == target.observation_id
            else observation
            for observation in bundle.observations
        ),
    )
    assert ordinary.content_id == spoofed.content_id
    with pytest.raises(
        TypeError,
        match="not an exact frozen authority schema node",
    ):
        run_exact_rational_bridge(
            bundle=spoofed,
            theory=theory,
            registry=registry,
        )


def test_nested_theory_and_registry_subclasses_are_rejected_before_hashing():
    theory, _, bundle, registry, _ = bridge_inputs()

    class ExternalLaw(type(theory.relation_laws[0])):
        pass

    class ExternalChannel(ObservableChannelBinding):
        pass

    law = theory.relation_laws[0]
    external_law = ExternalLaw(
        **{field.name: getattr(law, field.name) for field in fields(law)}
    )
    changed_theory = replace(
        theory,
        relation_laws=(external_law, *theory.relation_laws[1:]),
    )
    channel = registry.observable_channels[0]
    external_channel = ExternalChannel(channel.quantity_id, channel.observable_id)
    changed_registry = replace(
        registry,
        observable_channels=(external_channel, *registry.observable_channels[1:]),
    )
    with pytest.raises(
        TypeError,
        match="not an exact frozen authority schema node",
    ):
        run_exact_rational_bridge(
            bundle=bundle,
            theory=changed_theory,
            registry=registry,
        )
    with pytest.raises(
        TypeError,
        match="not an exact frozen authority schema node",
    ):
        run_exact_rational_bridge(
            bundle=bundle,
            theory=theory,
            registry=changed_registry,
        )


def test_theory_text_budget_rejects_before_any_authority_content_hash(monkeypatch):
    theory, _, bundle, registry, _ = bridge_inputs()
    changed_theory = replace(
        theory,
        scope=(
            "x" * (DEFAULT_EXACT_BRIDGE_POLICY.maximum_authority_text_characters + 1),
        ),
    )

    def forbidden_content_root(_self):
        raise AssertionError("authority content root ran before tree budget")

    monkeypatch.setattr(
        PublicEvidenceBundle,
        "content_id",
        property(forbidden_content_root),
    )
    monkeypatch.setattr(
        type(theory),
        "version_id",
        property(forbidden_content_root),
    )
    monkeypatch.setattr(
        Phase2BAdapterRegistry,
        "registry_id",
        property(forbidden_content_root),
    )
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=changed_theory,
        registry=registry,
    )
    assert run.reason == "RESOURCE_LIMIT:authority_text_characters"
    assert run.bundle_content_id is None
    assert run.uncertainty_receipt is None


def test_authority_integer_bit_length_rejects_before_registry_hash(monkeypatch):
    theory, _, bundle, registry, _ = bridge_inputs()
    changed_registry = replace(
        registry,
        maximum_candidate_count=(
            1
            << (DEFAULT_EXACT_BRIDGE_POLICY.maximum_authority_integer_bit_length + 1)
        ),
    )

    def forbidden_registry_root(_self):
        raise AssertionError("registry root ran before integer bit-length budget")

    monkeypatch.setattr(
        Phase2BAdapterRegistry,
        "registry_id",
        property(forbidden_registry_root),
    )
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=theory,
        registry=changed_registry,
    )
    assert run.reason == "RESOURCE_LIMIT:authority_integer_bit_length"
    assert run.registry_id is None


def test_authoritative_output_contains_no_float_and_api_accepts_no_receipts():
    theory, _, bundle, registry, _ = bridge_inputs()
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )

    def visit(value: object) -> None:
        assert not isinstance(value, float)
        if is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                visit(item)
        elif isinstance(value, Enum):
            return

    visit(run)
    parameter_names = set(inspect.signature(run_exact_rational_bridge).parameters)
    assert parameter_names == {"bundle", "theory", "registry"}
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_exact_rational_bridge(
            bundle=bundle,
            theory=theory,
            registry=registry,
            selection_policy=exact_bridge.DEFAULT_EXACT_SELECTION_POLICY,
        )


def test_authoritative_path_uses_own_grid_commitment_not_legacy_selector(
    monkeypatch,
):
    theory, _, bundle, registry, _ = bridge_inputs()

    def forbidden_legacy_commitment(_adapter):
        raise AssertionError("legacy selector commitment was consulted")

    monkeypatch.setattr(
        AdapterEnumerationResult,
        "candidate_grid_commitment",
        property(forbidden_legacy_commitment),
    )
    run = run_exact_rational_bridge(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert run.disposition is ExactBridgeDisposition.COMPLETE
    assert exact_bridge.__all__ == ("run_exact_rational_bridge",)


def test_bridge_has_no_old_float_projection_selector_or_law_import():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_exact_bridge_v1.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        token in module
        for module in imported
        for token in ("phase2b_projection_compiler", "phase2b_selector", "laws")
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Constant) and isinstance(node.value, float)
        for node in ast.walk(tree)
    )

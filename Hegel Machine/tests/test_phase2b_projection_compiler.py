import ast
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import product
from pathlib import Path

from hegel_machine.bootstrap import initial_theory
from hegel_machine.phase2_exit import HARD_NEGATIVE_OBSERVABLES, PASS_OBSERVABLES
from hegel_machine.phase2b_adapter import Phase2BAdapterRegistry
from hegel_machine.phase2b_projection_compiler import (
    ObservationCompilationDisposition,
    ProjectionCompilationDisposition,
    compile_candidate_evaluations,
    compile_observation_absolute_bound,
)
from hegel_machine.phase2b_selector import (
    CandidateIntervalStatus,
    TypedSelectionDisposition,
    select_typed_candidate_evaluations,
)
from hegel_machine.phase2b_wire import (
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    PublicEvidenceBundle,
)
from hegel_machine.schema import LawKind


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


def compiler_inputs():
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
    observables = tuple(
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
        for index, observable in enumerate(observables)
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
                raw_value = payload[observable_name]
                bound_entities = sorted(
                    entity_ids[((law.law_id, role), variant)]
                    for role, variant in zip(witness_roles, variants, strict=True)
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
                    # The public wire has nonempty numeric vectors.  Zero-valued
                    # source/sink terms are verifier-equivalent to an empty sum in
                    # this mechanics-only fixture.
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

    scale_id = uid(900)
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
            "scale_ids": [scale_id, second_scale_id],
            "root_scale_ids": [scale_id],
            "edges": [
                {
                    "source_scale_id": scale_id,
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
    return theory, mapping, bundle, registry


def test_root_projection_compiler_closes_wire_adapter_selector_mechanics():
    theory, _, bundle, registry = compiler_inputs()
    compiled = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert compiled.disposition is ProjectionCompilationDisposition.COMPLETE
    assert compiled.reason == "complete_candidate_evaluation_grid"
    assert len(compiled.evaluations) == 72
    assert all(item.completed for item in compiled.evaluations)
    assert all(item.error_code is None for item in compiled.evaluations)
    assert {item.law_kind for item in compiled.evaluations} == set(LawKind)
    assert {
        item.law_kind
        for item in compiled.evaluations
        if item.status is CandidateIntervalStatus.PASS
    } == {LawKind.SYMMETRY}

    decision = select_typed_candidate_evaluations(
        compiled.evaluations,
        evidence_bundle=bundle,
        adapter_registry=registry,
    )
    assert decision.disposition is TypedSelectionDisposition.ADMISSIBLE_SCALE_SET
    assert decision.selected_law_kind is LawKind.SYMMETRY
    assert decision.admissible_scale_hypothesis_ids == (uid(900), uid(902))
    assert decision.candidate_grid_commitment_id == (
        compiled.candidate_grid_commitment_id
    )


def test_absolute_bound_compiler_preserves_interval_endpoints_not_midpoint():
    _, mapping, _, _ = compiler_inputs()
    observation = next(
        item
        for item in mapping["observations"]
        if item["value"]["kind"] == "numeric"
    )
    observation["value"] = {
        "kind": "interval",
        "lower": [1.0],
        "upper": [3.0],
    }
    observation["uncertainty"] = {
        "model": "absolute_bound",
        "radius": [0.5],
    }
    typed = PublicEvidenceBundle.from_mapping(mapping).observations
    compiled = compile_observation_absolute_bound(
        next(
            item
            for item in typed
            if item.observation_id == observation["observation_id"]
        )
    )
    assert compiled.disposition is ObservationCompilationDisposition.COMPILED
    assert tuple((item.lower, item.upper) for item in compiled.numeric_bounds) == (
        (0.5, 3.5),
    )


def test_absolute_bound_binary64_arithmetic_is_rounded_outward():
    _, mapping, _, _ = compiler_inputs()
    observation = next(
        item
        for item in mapping["observations"]
        if item["value"]["kind"] == "numeric"
        and len(item["value"]["values"]) == 1
    )
    observation["value"] = {"kind": "numeric", "values": [1.0]}
    observation["uncertainty"] = {
        "model": "absolute_bound",
        "radius": [0.1],
    }
    typed = PublicEvidenceBundle.from_mapping(mapping).observations
    compiled = compile_observation_absolute_bound(
        next(
            item
            for item in typed
            if item.observation_id == observation["observation_id"]
        )
    )
    bound = compiled.numeric_bounds[0]
    exact_lower = Fraction.from_float(1.0) - Fraction.from_float(0.1)
    exact_upper = Fraction.from_float(1.0) + Fraction.from_float(0.1)
    assert Fraction.from_float(bound.lower) <= exact_lower
    assert Fraction.from_float(bound.upper) >= exact_upper


def test_absolute_bound_overflow_is_a_typed_unsupported_result():
    _, mapping, _, _ = compiler_inputs()
    observation = next(
        item
        for item in mapping["observations"]
        if item["value"]["kind"] == "numeric"
        and len(item["value"]["values"]) == 1
    )
    observation["value"] = {"kind": "numeric", "values": [1e308]}
    observation["uncertainty"] = {
        "model": "absolute_bound",
        "radius": [1e308],
    }
    typed = PublicEvidenceBundle.from_mapping(mapping).observations
    compiled = compile_observation_absolute_bound(
        next(
            item
            for item in typed
            if item.observation_id == observation["observation_id"]
        )
    )
    assert compiled.disposition is ObservationCompilationDisposition.UNSUPPORTED
    assert compiled.reason == "absolute_bound_overflow"
    assert compiled.numeric_bounds == ()


def test_nondegenerate_numeric_envelopes_are_full_grid_errors_not_corner_scores():
    theory, mapping, _, registry = compiler_inputs()
    for observation in mapping["observations"]:
        if observation["value"]["kind"] == "numeric":
            observation["uncertainty"]["radius"] = [
                0.1 for _ in observation["value"]["values"]
            ]
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    compiled = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert compiled.disposition is ProjectionCompilationDisposition.COMPLETE
    assert len(compiled.evaluations) == 72
    assert not any(item.completed for item in compiled.evaluations)
    assert {
        item.error_code for item in compiled.evaluations
    } == {"nondegenerate_interval_residual_semantics_not_implemented"}

    decision = select_typed_candidate_evaluations(
        compiled.evaluations,
        evidence_bundle=bundle,
        adapter_registry=registry,
    )
    assert decision.disposition is TypedSelectionDisposition.ABSTAIN
    assert decision.reason == "candidate_evaluation_error"


def test_units_and_support_are_not_silently_discarded():
    theory, mapping, _, registry = compiler_inputs()
    for observation in mapping["observations"]:
        observation["unit_dimension"]["si_exponents"] = [1, 0, 0, 0, 0, 0, 0]
    dimensioned = compile_candidate_evaluations(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert {
        item.error_code for item in dimensioned.evaluations
    } == {"nondimensionless_unit_semantics_not_implemented"}

    theory, mapping, _, registry = compiler_inputs()
    observable_names = sorted(
        {
            observable
            for law in theory.relation_laws
            for observable in law.required_observables
        }
    )
    forward_quantity_id = uid(500 + observable_names.index("forward"))
    for observation in mapping["observations"]:
        if observation["quantity_id"] == forward_quantity_id:
            observation["temporal_support"]["clock_id"] = uid(992)
    unaligned = compile_candidate_evaluations(
        bundle=PublicEvidenceBundle.from_mapping(mapping),
        theory=theory,
        registry=registry,
    )
    assert "unaligned_temporal_support" in {
        item.error_code for item in unaligned.evaluations
    }


def test_finite_inputs_outside_safe_verifier_numeric_domain_fail_closed():
    theory, mapping, _, registry = compiler_inputs()
    for observation in mapping["observations"]:
        value = observation["value"]
        if value["kind"] == "numeric":
            value["values"] = [1e308 for _ in value["values"]]
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    compiled = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert compiled.disposition is ProjectionCompilationDisposition.COMPLETE
    assert len(compiled.evaluations) == 72
    assert {
        item.error_code for item in compiled.evaluations
    } == {"verifier_numeric_domain_unsupported"}

    decision = select_typed_candidate_evaluations(
        compiled.evaluations,
        evidence_bundle=bundle,
        adapter_registry=registry,
    )
    assert decision.disposition is TypedSelectionDisposition.ABSTAIN


def test_standard_error_aborts_the_bundle_before_any_candidate_grid_is_returned():
    theory, mapping, _, registry = compiler_inputs()
    changed = next(
        item
        for item in mapping["observations"]
        if item["value"]["kind"] == "numeric"
    )
    changed["uncertainty"]["model"] = "standard_error"
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    compiled = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert compiled.disposition is ProjectionCompilationDisposition.ABSTAIN
    assert compiled.reason == (
        "bundle_uncertainty_preflight:STANDARD_ERROR_UNSUPPORTED"
    )
    assert compiled.evaluations == ()
    assert compiled.candidate_grid_commitment_id is None


def test_nonidentity_transform_is_not_assigned_implicit_semantics():
    theory, mapping, _, registry = compiler_inputs()
    second_scale = uid(902)
    mapping["transform_catalog"][0] = {
        "transform_id": uid(901),
        "operation": "coarse_graining",
        "parameters": [2.0],
    }
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    compiled = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert len(compiled.evaluations) == 72
    errors = tuple(item for item in compiled.evaluations if not item.completed)
    assert len(errors) == 36
    assert {
        item.error_code for item in errors
    } == {"unsupported_transform_semantics:coarse_graining"}
    assert all(item.scale_hypothesis_id == second_scale for item in errors)


def test_ambiguous_witness_never_returns_a_partial_candidate_grid():
    theory, mapping, _, registry = compiler_inputs()
    duplicate = deepcopy(mapping["observations"][0])
    duplicate["observation_id"] = uid(1999)
    duplicate["provenance_sha256"] = "f" * 64
    mapping["observations"].append(duplicate)
    bundle = PublicEvidenceBundle.from_mapping(mapping)
    compiled = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    assert compiled.disposition is ProjectionCompilationDisposition.COMPLETE
    assert len(compiled.evaluations) == 72
    errors = tuple(item for item in compiled.evaluations if not item.completed)
    assert errors
    assert {item.error_code for item in errors} == {"ambiguous_observable_witness"}


def test_projection_compilation_is_public_input_order_invariant():
    theory, mapping, bundle, registry = compiler_inputs()
    expected = compile_candidate_evaluations(
        bundle=bundle,
        theory=theory,
        registry=registry,
    )
    for key in ("entity_candidates", "role_ids", "quantity_ids", "observations"):
        mapping[key] = list(reversed(mapping[key]))
    reordered = PublicEvidenceBundle.from_mapping(mapping)
    assert reordered == bundle
    assert compile_candidate_evaluations(
        bundle=reordered,
        theory=theory,
        registry=registry,
    ) == expected


def test_projection_compiler_rejects_registry_theory_shape_drift():
    theory, _, bundle, registry = compiler_inputs()
    changed_binding = replace(registry.law_bindings[0], law_id="foreign-law")
    changed_registry = replace(
        registry,
        law_bindings=(changed_binding, *registry.law_bindings[1:]),
    )
    try:
        compile_candidate_evaluations(
            bundle=bundle,
            theory=theory,
            registry=changed_registry,
        )
    except ValueError as exc:
        assert str(exc) == "projection compiler adapter law registry differs"
    else:
        raise AssertionError("registry/theory drift was accepted")


def test_projection_compiler_has_no_answer_generator_or_phase2a_fixture_import():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hegel_machine"
        / "phase2b_projection_compiler.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = ("phase2_exit", "benchmark", "generator", "evaluator")
    assert not any(
        token in module
        for module in imported
        for token in banned
    )

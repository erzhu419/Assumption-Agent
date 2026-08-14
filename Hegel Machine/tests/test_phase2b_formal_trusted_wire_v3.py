from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict, fields, make_dataclass, replace
from fractions import Fraction
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import re
import runpy
import struct
import types
import unicodedata
import uuid
from typing import get_args, get_origin, get_type_hints

import pytest

import hegel_machine.phase2b_exact_transform_core_v1 as exact_core
import hegel_machine.phase2b_formal_trusted_wire_v3 as wire


def _uuid(value: int) -> str:
    return str(uuid.UUID(int=value, version=4))


ZERO_DIGEST = "sha256:" + "0" * 64


_EXACT_CORE_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_phase2b_exact_transform_core_v1.py"))
)


def _rational(value: Fraction) -> wire.RationalValueV3:
    return wire.RationalValueV3(str(value.numerator), str(value.denominator))


def _wire_case_from_exact_core_profile(
    profile: exact_core.ExactTransformCoreProfileV1,
    *,
    index: int,
) -> wire.FormalSourceCaseV3:
    """Test-only inverse adapter from locked literal core fixtures to rich V3."""
    computed = exact_core.run_exact_transform_core_v1(profile)
    assert computed.disposition is exact_core.CoreTransformDispositionV1.COMPLETE
    all_values = tuple(
        sorted(
            (*profile.source_components, *computed.output_components),
            key=lambda item: item.ref,
        )
    )
    next_value = 1_000_000_000 + index * 100_000

    def new_uuid() -> str:
        nonlocal next_value
        next_value += 1
        return _uuid(next_value)

    bundle_id, aggregate_map_id, transform_id, task_id = (
        new_uuid() for _ in range(4)
    )
    entity_keys = tuple(
        sorted({key for item in all_values for key in item.metadata.entity_keys})
    )
    role_keys = tuple(
        sorted({key for item in all_values for key in item.metadata.role_keys})
    )
    entity_ids = {key: new_uuid() for key in entity_keys}
    role_ids = {key: new_uuid() for key in role_keys}
    scale_keys = tuple(
        sorted({item.ref.scale_ref.key for item in all_values})
    )
    scale_ids = {key: new_uuid() for key in scale_keys}

    quantities: list[exact_core.CoreQuantityMetadataV1] = []
    for item in all_values:
        if item.metadata.quantity not in quantities:
            quantities.append(item.metadata.quantity)
    quantity_ids = {value: new_uuid() for value in quantities}
    semantic_keys: dict[str, str] = {}
    unit_keys: dict[exact_core.CoreUnitBindingV1, str] = {}
    quantity_records: list[wire.FormalQuantityRecordV3] = []
    for quantity_index, quantity in enumerate(quantities):
        semantic_key = semantic_keys.setdefault(
            quantity.semantic_key, f"quantity-{len(semantic_keys)}"
        )
        if quantity.unit_binding is None:
            unit_binding = wire.FormalUnitBindingV3(
                "none", _rational(Fraction(1)), _rational(Fraction(0))
            )
            value_kind = "boolean"
        else:
            unit_binding = wire.FormalUnitBindingV3(
                unit_keys.setdefault(
                    quantity.unit_binding, f"unit-{len(unit_keys)}"
                ),
                _rational(quantity.unit_binding.scale_to_si),
                _rational(quantity.unit_binding.offset_to_si),
            )
            value_kind = "numeric"
        quantity_records.append(
            wire.FormalQuantityRecordV3(
                quantity_ids[quantity],
                semantic_key,
                quantity.si_exponents,
                unit_binding,
                value_kind,
            )
        )

    context_keys: list[
        tuple[
            exact_core.CoreTemporalSupportV1 | None,
            exact_core.CoreSpatialSupportV1 | None,
        ]
    ] = []
    for item in all_values:
        key = (item.metadata.temporal_support, item.metadata.spatial_support)
        if key not in context_keys:
            context_keys.append(key)
    context_ids = {key: new_uuid() for key in context_keys}
    clock_keys: dict[exact_core.CoreClockBindingV1, str] = {}
    frame_keys: dict[exact_core.CoreFrameBindingV1, str] = {}
    context_records: list[wire.FormalContextRecordV3] = []
    for temporal, spatial in context_keys:
        if temporal is None:
            temporal_kind = "none"
            clock_binding = "none"
            temporal_support = wire.NullableTemporalSupportV3(False, None)
        else:
            temporal_kind = "domain_epoch"
            clock_binding = clock_keys.setdefault(
                temporal.clock_binding, f"clock-{len(clock_keys)}"
            )
            temporal_support = wire.NullableTemporalSupportV3(
                True,
                wire.ExactTemporalSupportV3(
                    _rational(temporal.start), _rational(temporal.end)
                ),
            )
        if spatial is None:
            spatial_kind = "none"
            frame_binding = "none"
            coordinate_labels: tuple[str, ...] = ()
            spatial_support = wire.NullableSpatialSupportV3(False, None)
            orientation = wire.NullableTextV3(False, None)
        else:
            spatial_kind = "cartesian"
            frame_binding = frame_keys.setdefault(
                spatial.frame_binding, f"frame-{len(frame_keys)}"
            )
            coordinate_labels = tuple(
                f"axis-{dimension}" for dimension in range(len(spatial.lower))
            )
            spatial_support = wire.NullableSpatialSupportV3(
                True,
                wire.ExactSpatialSupportV3(
                    tuple(_rational(value) for value in spatial.lower),
                    tuple(_rational(value) for value in spatial.upper),
                ),
            )
            orientation = wire.NullableTextV3(
                spatial.frame_binding.orientation is not None,
                spatial.frame_binding.orientation,
            )
        context_records.append(
            wire.FormalContextRecordV3(
                context_ids[(temporal, spatial)],
                temporal_kind,
                clock_binding,
                spatial_kind,
                frame_binding,
                coordinate_labels,
                temporal_support,
                spatial_support,
                orientation,
            )
        )

    owner_values: dict[
        tuple[exact_core.CoreScaleRefV1, exact_core.CoreObservationRefV1],
        list[exact_core.CoreComponentValueV1],
    ] = {}
    owner_for_ref: dict[
        exact_core.CoreComponentRefV1,
        tuple[exact_core.CoreScaleRefV1, exact_core.CoreObservationRefV1],
    ] = {}
    coordinate_source_owner = exact_core.CoreObservationRefV1(
        "wire-v3-coordinate-source-vector"
    )
    for item in all_values:
        owner = (
            item.ref.scale_ref,
            coordinate_source_owner
            if profile.contract.operation
            is exact_core.CoreTransformOperationV1.COORDINATE_AFFINE
            and item.ref.scale_ref == profile.contract.source_scale_ref
            else item.ref.observation_ref,
        )
        owner_for_ref[item.ref] = owner
        owner_values.setdefault(owner, []).append(item)
    observation_ids = {owner: new_uuid() for owner in sorted(owner_values)}
    local_component_indices: dict[exact_core.CoreComponentRefV1, int] = {}
    observation_records: list[wire.FormalObservationRecordV3] = []
    for owner in sorted(owner_values):
        values = tuple(
            sorted(
                owner_values[owner],
                key=lambda item: (item.ref.component_index, item.ref),
            )
        )
        for local_index, item in enumerate(values):
            local_component_indices[item.ref] = local_index
        metadata = values[0].metadata
        assert all(
            item.metadata.quantity == metadata.quantity
            and item.metadata.entity_keys == metadata.entity_keys
            and item.metadata.role_keys == metadata.role_keys
            and item.metadata.temporal_support == metadata.temporal_support
            and item.metadata.spatial_support == metadata.spatial_support
            for item in values
        )
        kinds = {item.value_kind for item in values}
        assert len(kinds) == 1
        kind = values[0].value_kind
        if kind is exact_core.CoreComponentValueKindV1.NUMERIC:
            numeric_value = wire.NullableRationalVectorV3(
                True,
                tuple(_rational(item.numeric_value) for item in values),
            )
            boolean_value = wire.NullableBooleanV3(False, None)
            uncertainty_model = values[0].uncertainty_model.value
            uncertainty_radius = wire.NullableRationalVectorV3(
                True,
                tuple(_rational(item.absolute_radius) for item in values),
            )
        elif kind is exact_core.CoreComponentValueKindV1.BOOLEAN:
            assert len(values) == 1
            numeric_value = wire.NullableRationalVectorV3(False, None)
            boolean_value = wire.NullableBooleanV3(True, values[0].boolean_value)
            uncertainty_model = "not_applicable"
            uncertainty_radius = wire.NullableRationalVectorV3(False, None)
        else:
            assert len(values) == 1
            numeric_value = wire.NullableRationalVectorV3(False, None)
            boolean_value = wire.NullableBooleanV3(False, None)
            uncertainty_model = "not_applicable"
            uncertainty_radius = wire.NullableRationalVectorV3(False, None)
        observation_records.append(
            wire.FormalObservationRecordV3(
                observation_ids[owner],
                tuple(sorted(entity_ids[key] for key in metadata.entity_keys)),
                tuple(sorted(role_ids[key] for key in metadata.role_keys)),
                quantity_ids[metadata.quantity],
                context_ids[(metadata.temporal_support, metadata.spatial_support)],
                scale_ids[owner[0].key],
                tuple(
                    wire.FormalComponentDescriptorV3(
                        item.metadata.component_label,
                        item.metadata.axis.value,
                        item.metadata.value_role.value,
                    )
                    for item in values
                ),
                numeric_value,
                boolean_value,
                uncertainty_model,
                uncertainty_radius,
                ZERO_DIGEST,
            )
        )
    observation_by_owner = {
        owner: record
        for owner, record in zip(
            sorted(owner_values), observation_records, strict=True
        )
    }

    def formal_ref(
        ref: exact_core.CoreComponentRefV1,
    ) -> wire.FormalComponentRefV3:
        record = observation_by_owner[owner_for_ref[ref]]
        local_index = local_component_indices[ref]
        descriptor = record.components[local_index]
        return wire.FormalComponentRefV3(
            record.observation_id,
            record.quantity_id,
            record.context_id,
            record.scale_id,
            descriptor.label,
            local_index,
        )

    def formal_row(row: exact_core.CoreExactRowV1) -> wire.FormalExactRowV3:
        return wire.FormalExactRowV3(
            formal_ref(row.output_ref),
            tuple(
                sorted(
                    (
                        wire.FormalExactTermV3(
                            formal_ref(term.input_ref),
                            _rational(term.coefficient),
                        )
                        for term in row.terms
                    ),
                    key=lambda term: term.input_ref,
                )
            ),
            _rational(row.offset),
        )

    def formal_rows(
        rows: tuple[exact_core.CoreExactRowV1, ...],
    ) -> tuple[wire.FormalExactRowV3, ...]:
        return tuple(sorted((formal_row(row) for row in rows), key=lambda row: row.output_ref))

    certificate = profile.contract.certificate
    if hasattr(certificate, "groups"):
        groups = tuple(
            wire.FormalAggregateGroupV3(
                tuple(sorted(formal_ref(ref) for ref in group.input_refs)),
                tuple(sorted(formal_ref(ref) for ref in group.output_refs)),
                group.quotient_label,
            )
            for group in certificate.groups
        )
    else:
        groups = (
            wire.FormalAggregateGroupV3(
                tuple(sorted(formal_ref(ref) for ref in profile.contract.input_refs)),
                tuple(
                    sorted(
                        formal_ref(item.ref)
                        for item in profile.contract.output_metadata
                    )
                ),
                "all-components",
            ),
        )
    operation = profile.contract.operation.value
    reducer = (
        certificate.reducer.value if hasattr(certificate, "reducer") else "not_applicable"
    )
    missing_policy = certificate.missing_policy.value
    if type(certificate) is exact_core.CoreIdentityCertificateV1:
        formal_certificate: wire.FormalOperationCertificateV3 = (
            wire.FormalIdentityCertificateV3(
                wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
                missing_policy,
                "exact_one_to_one_identity",
            )
        )
    elif type(certificate) is exact_core.CoreUnitConversionCertificateV1:
        formal_certificate = wire.FormalUnitConversionCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            _rational(certificate.factor),
            _rational(certificate.inverse_factor),
            missing_policy,
            "target_value_equals_source_value_times_factor",
            "si_dimension_and_support_preserved",
        )
    elif type(certificate) is exact_core.CoreCoordinateAffineCertificateV1:
        formal_certificate = wire.FormalCoordinateAffineCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            frame_keys[certificate.source_frame_binding],
            frame_keys[certificate.target_frame_binding],
            certificate.dimension,
            formal_rows(certificate.inverse_rows),
            missing_policy,
            "exact_affine_box_enclosure",
            "two_sided_exact_sparse_affine_inverse",
        )
    elif type(certificate) is exact_core.CoreTemporalAggregationCertificateV1:
        formal_certificate = wire.FormalTemporalAggregationCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            reducer,
            missing_policy,
            certificate.boundary_policy.value,
            "same_clock_half_open_partition_modulo_final_endpoint",
            certificate.weighted_mean_basis.value,
        )
    elif type(certificate) is exact_core.CoreSpatialAggregationCertificateV1:
        formal_certificate = wire.FormalSpatialAggregationCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            reducer,
            missing_policy,
            certificate.boundary_policy.value,
            "same_frame_interior_disjoint_positive_volume_partition",
            certificate.weighted_mean_basis.value,
        )
    elif type(certificate) is exact_core.CoreSamplingResolutionCertificateV1:
        formal_certificate = wire.FormalSamplingResolutionCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            certificate.axis.value,
            tuple(formal_ref(ref) for ref in certificate.selected_inputs),
            tuple(formal_ref(ref) for ref in certificate.discarded_inputs),
            tuple(
                wire.FormalSamplingGridPointV3(
                    formal_ref(point.input_ref),
                    tuple(_rational(value) for value in point.coordinates),
                )
                for point in certificate.grid_points
            ),
            certificate.grid_dimension,
            (
                "none"
                if certificate.grid_frame_binding is None
                else frame_keys[certificate.grid_frame_binding]
            ),
            missing_policy,
            certificate.boundary_policy.value,
            "single_series_ordered_exact_subselection",
        )
    elif type(certificate) is exact_core.CoreEquivalentSplitMergeCertificateV1:
        formal_certificate = wire.FormalEquivalentSplitMergeCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            certificate.direction.value,
            formal_rows(certificate.inverse_rows),
            missing_policy,
            "extensive_sum_one_sided_exact_inverse",
        )
    else:
        assert type(certificate) is exact_core.CoreCoarseGrainingCertificateV1
        formal_certificate = wire.FormalCoarseGrainingCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            reducer,
            formal_rows(certificate.source_commutation_rows),
            formal_rows(certificate.target_commutation_rows),
            aggregate_map_id,
            certificate.quotient_labels,
            missing_policy,
            certificate.boundary_policy.value,
            "transform_after_source_equals_target_after_transform",
            certificate.weighted_mean_basis.value,
        )

    input_refs = tuple(sorted(formal_ref(ref) for ref in profile.contract.input_refs))
    output_refs = tuple(
        sorted(formal_ref(item.ref) for item in profile.contract.output_metadata)
    )
    graph = wire.FormalCaseGraphV3(
        wire.FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION,
        bundle_id,
        tuple(
            wire.FormalEntityRecordV3(
                entity_ids[key], tuple(sorted(role_ids.values()))
            )
            for key in entity_keys
        ),
        tuple(sorted(role_ids.values())),
        tuple(quantity_records),
        tuple(context_records),
        tuple(observation_records),
        wire.FormalTaskTargetV3(
            task_id,
            tuple(sorted(entity_ids.values())),
            tuple(sorted(quantity_ids.values())),
        ),
        wire.FormalSemanticDescriptorV3(
            "symbolic", (f"Exact typed {operation} witness.",)
        ),
        tuple(
            wire.FormalScaleRecordV3(
                scale_ids[key], key, key == profile.contract.source_scale_ref.key
            )
            for key in scale_keys
        ),
        (
            wire.FormalAggregateMapRecordV3(
                aggregate_map_id,
                scale_ids[profile.contract.source_scale_ref.key],
                scale_ids[profile.contract.target_scale_ref.key],
                transform_id,
                operation,
                reducer,
                groups,
            ),
        ),
        (
            wire.FormalTransformContractV3(
                transform_id,
                operation,
                scale_ids[profile.contract.source_scale_ref.key],
                scale_ids[profile.contract.target_scale_ref.key],
                input_refs,
                output_refs,
                formal_rows(profile.contract.kernel_rows),
                tuple(
                    sorted(
                        (
                            wire.FormalDiscreteMappingV3(
                                formal_ref(mapping.input_ref),
                                formal_ref(mapping.output_ref),
                            )
                            for mapping in profile.contract.discrete_mappings
                        ),
                        key=lambda mapping: (mapping.input_ref, mapping.output_ref),
                    )
                ),
                formal_certificate,
            ),
        ),
        wire.FormalRecognizerRegistryV3(
            "exact-transform-core-v1",
            (wire.FormalFamilyAliasV3(operation, operation),),
            (
                wire.FormalLawBindingV3(
                    operation + "-law",
                    operation,
                    operation,
                    tuple(sorted(record.observation_id for record in observation_records)),
                    tuple(sorted(role_ids.values())),
                    tuple(sorted(quantity_ids.values())),
                ),
            ),
            tuple(
                sorted(
                    (
                        wire.FormalObservableChannelV3(
                            record.observation_id, record.quantity_id
                        )
                        for record in observation_records
                    ),
                    key=lambda item: item.observation_id,
                )
            ),
            100,
        ),
        wire.FormalEligibilityMaskV3(
            tuple(sorted(record.observation_id for record in observation_records)),
            tuple(sorted(role_ids.values())),
            tuple(sorted(scale_ids.values())),
        ),
    )
    return wire.FormalSourceCaseV3(graph)


def source_case(index: int = 0, *, standard_error: bool = False) -> wire.FormalSourceCaseV3:
    base = 100_000 * (index + 1)
    bundle, entity, role, quantity, context = (_uuid(base + i) for i in range(1, 6))
    source_scale, target_scale = _uuid(base + 6), _uuid(base + 7)
    source_observation, target_observation = _uuid(base + 8), _uuid(base + 9)
    auxiliary_source, auxiliary_target = _uuid(base + 13), _uuid(base + 14)
    aggregate_map, transform, task = _uuid(base + 10), _uuid(base + 11), _uuid(base + 12)
    rational = wire.RationalValueV3
    numeric = lambda value: wire.NullableRationalVectorV3(
        True, (rational(str(value), "1"),)
    )
    observations = tuple(
        wire.FormalObservationRecordV3(
            observation_id=observation_id,
            entity_ids=(entity,),
            role_candidate_ids=(role,),
            quantity_id=quantity,
            context_id=context,
            scale_id=scale,
            components=(
                wire.FormalComponentDescriptorV3("scalar", "scalar", "intensive"),
            ),
            numeric_value=numeric(value),
            boolean_value=wire.NullableBooleanV3(False, None),
            uncertainty_model=("standard_error" if standard_error else "absolute_bound"),
            uncertainty_radius=numeric(1),
            public_provenance_commitment=ZERO_DIGEST,
        )
        for observation_id, scale, value in (
            (source_observation, source_scale, index + 1),
            (target_observation, target_scale, index + 1),
            (auxiliary_source, source_scale, index + 2),
            (auxiliary_target, target_scale, index + 2),
        )
    )
    source_ref = wire.FormalComponentRefV3(
        source_observation, quantity, context, source_scale, "scalar", 0
    )
    target_ref = wire.FormalComponentRefV3(
        target_observation, quantity, context, target_scale, "scalar", 0
    )
    graph = wire.FormalCaseGraphV3(
        schema_version=wire.FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION,
        bundle_id=bundle,
        entities=(wire.FormalEntityRecordV3(entity, (role,)),),
        role_candidate_ids=(role,),
        quantities=(
            wire.FormalQuantityRecordV3(
                quantity,
                "fixture-quantity",
                (1, 0, 0, 0, 0, 0, 0),
                wire.FormalUnitBindingV3(
                    "fixture-unit", rational("1", "1"), rational("0", "1")
                ),
                "numeric",
            ),
        ),
        contexts=(
            wire.FormalContextRecordV3(
                context,
                "relative_monotonic",
                "fixture-clock",
                "cartesian",
                "fixture-frame",
                ("x",),
                wire.NullableTemporalSupportV3(
                    True,
                    wire.ExactTemporalSupportV3(
                        rational("0", "1"), rational("10", "1")
                    ),
                ),
                wire.NullableSpatialSupportV3(
                    True,
                    wire.ExactSpatialSupportV3(
                        (rational("0", "1"),), (rational("10", "1"),)
                    ),
                ),
                wire.NullableTextV3(False, None),
            ),
        ),
        observations=observations,
        task_target=wire.FormalTaskTargetV3(task, (entity,), (quantity,)),
        semantic_descriptor=wire.FormalSemanticDescriptorV3(
            "mixed",
            (
                "A scalar quantity is preserved across an exact identity map.",
                "The source and target observations share entity, role, quantity, and context semantics.",
            ),
        ),
        scales=(
            wire.FormalScaleRecordV3(source_scale, "fine", True),
            wire.FormalScaleRecordV3(target_scale, "coarse", False),
        ),
        aggregate_maps=(
            wire.FormalAggregateMapRecordV3(
                aggregate_map,
                source_scale,
                target_scale,
                transform,
                "identity",
                "not_applicable",
                (wire.FormalAggregateGroupV3((source_ref,), (target_ref,), "q0"),),
            ),
        ),
        transforms=(
            wire.FormalTransformContractV3(
                transform,
                "identity",
                source_scale,
                target_scale,
                (source_ref,),
                (target_ref,),
                (
                    wire.FormalExactRowV3(
                        target_ref,
                        (
                            wire.FormalExactTermV3(
                                source_ref, rational("1", "1")
                            ),
                        ),
                        rational("0", "1"),
                    ),
                ),
                (),
                wire.FormalIdentityCertificateV3(
                    wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
                    "explicit_preserve",
                    "exact_one_to_one_identity",
                ),
            ),
        ),
        recognizer_registry=wire.FormalRecognizerRegistryV3(
            "theory-policy-v3",
            (wire.FormalFamilyAliasV3("identity", "identity"),),
            (
                wire.FormalLawBindingV3(
                    "identity-law",
                    "identity",
                    "identity",
                    tuple(sorted((source_observation, target_observation))),
                    (role,),
                    (quantity,),
                ),
            ),
            tuple(
                sorted(
                    (
                        wire.FormalObservableChannelV3(source_observation, quantity),
                        wire.FormalObservableChannelV3(target_observation, quantity),
                        wire.FormalObservableChannelV3(auxiliary_source, quantity),
                        wire.FormalObservableChannelV3(auxiliary_target, quantity),
                    ),
                    key=lambda item: item.observation_id,
                )
            ),
            100,
        ),
        eligibility_mask=wire.FormalEligibilityMaskV3(
            tuple(
                sorted(
                    (
                        source_observation,
                        target_observation,
                        auxiliary_source,
                        auxiliary_target,
                    )
                )
            ),
            (role,),
            tuple(sorted((source_scale, target_scale))),
        ),
    )
    return wire.FormalSourceCaseV3(graph)


def keys() -> wire.FormalWireKeySourcesV3:
    return wire.FormalWireKeySourcesV3(b"S" * 32, b"I" * 32, b"P" * 32)


def receipt() -> wire.KeySourceTranscriptReceiptV3:
    return wire.issue_key_source_transcript_receipt_v3(
        key_sources=keys(),
        latent_seed_commitments=(wire.seed_commitment_v3(b"L" * 32),),
        answer_seed_commitments=(wire.seed_commitment_v3(b"A" * 32),),
    )


def catalog(
    cases: tuple[wire.FormalSourceCaseV3, ...],
) -> tuple[
    wire.FormalGlobalTransformCatalogV3,
    wire.FormalGlobalTransformCatalogBindingReceiptV3,
]:
    return wire.issue_formal_global_transform_catalog_v3(source_cases=cases)


def unsafe_alignment_copy(
    alignment: wire.FormalPrivateAuditAlignmentV3,
    **changes: object,
) -> wire.FormalPrivateAuditAlignmentV3:
    result = object.__new__(type(alignment))
    for item in fields(alignment):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(alignment, item.name)),
        )
    return result


def unsafe_issued_copy(instance: object, **changes: object) -> object:
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return result


def unsafe_path_replace(
    value: object,
    path: tuple[str | int, ...],
    replacement: object,
) -> object:
    """Test-only recursive frozen-dataclass mutation without constructor checks."""
    if not path:
        return replacement
    head, *tail = path
    if isinstance(head, str) and hasattr(value, "__dataclass_fields__"):
        return unsafe_issued_copy(
            value,
            **{
                head: unsafe_path_replace(
                    getattr(value, head), tuple(tail), replacement
                )
            },
        )
    if isinstance(head, int) and type(value) is tuple:
        rows = list(value)
        rows[head] = unsafe_path_replace(rows[head], tuple(tail), replacement)
        return tuple(rows)
    raise TypeError(f"unsupported unsafe mutation path {path!r}")


def routed_non_id_string_paths(
    value: object,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str | int, ...], ...]:
    """Enumerate concrete non-ID string leaves from the frozen routing policy."""
    excluded_routes = {"UUID", "UUID_TUPLE", "DIGEST", "PREFIXED_DIGEST_ID"}
    result: list[tuple[str | int, ...]] = []
    if hasattr(value, "__dataclass_fields__"):
        policy = wire.FORMAL_PUBLIC_STRING_FIELD_POLICY_V3.get(
            type(value).__name__, {}
        )
        for item in fields(value):
            child = getattr(value, item.name)
            route = policy.get(item.name)
            child_path = (*path, item.name)
            if route is not None and route not in excluded_routes:
                if type(child) is tuple:
                    result.extend(
                        (*child_path, index) for index in range(len(child))
                    )
                else:
                    result.append(child_path)
            elif hasattr(child, "__dataclass_fields__") or type(child) is tuple:
                result.extend(routed_non_id_string_paths(child, child_path))
    elif type(value) is tuple:
        for index, child in enumerate(value):
            result.extend(routed_non_id_string_paths(child, (*path, index)))
    return tuple(result)


def build_success(
    cases: tuple[wire.FormalSourceCaseV3, ...] | None = None,
) -> wire.FormalTrustedWireBuildSuccessV3:
    selected = (source_case(0), source_case(1)) if cases is None else cases
    global_catalog, binding = catalog(selected)
    result = wire.build_formal_trusted_wire_batch_v3(
        source_cases=selected,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(result) is wire.FormalTrustedWireBuildSuccessV3, getattr(result, "reason", None)
    return result


def build(cases: tuple[wire.FormalSourceCaseV3, ...] | None = None) -> wire.FormalTrustedWireBatchV3:
    return build_success(cases).public_batch


def identity_variant_schedule(
    success: wire.FormalTrustedWireBuildSuccessV3,
    *,
    case_order: tuple[str, ...] | None = None,
    observation_order_by_envelope: dict[str, tuple[str, ...]] | None = None,
    renamings: tuple[wire.FormalUuidRenamingV3, ...] = (),
) -> wire.FormalVariantScheduleV3:
    batch = success.public_batch
    orders: list[wire.FormalObservationOrderV3] = []
    for row in batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(row.envelope)
        base_order = tuple(
            item.observation_id for item in decoded.public_case.graph.observations
        )
        orders.append(
            wire.FormalObservationOrderV3(
                row.envelope_id,
                (observation_order_by_envelope or {}).get(row.envelope_id, base_order),
            )
        )
    return wire.FormalVariantScheduleV3(
        batch.envelope_ids if case_order is None else case_order,
        renamings,
        tuple(sorted(orders, key=lambda item: item.envelope_id)),
    )


def total_nonmonotonic_renamings(
    success: wire.FormalTrustedWireBuildSuccessV3,
    variant: int,
) -> tuple[wire.FormalUuidRenamingV3, ...]:
    domain: dict[str, str] = {}
    for envelope in success.public_batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(envelope.envelope)
        for _, namespace, value in wire._walk_uuid_leaves_v3(
            decoded.payload["public_case"]["graph"]
        ):
            assert domain.setdefault(value, namespace) == namespace
    sources = sorted(domain)
    targets = [
        _uuid(10_000_000_000 + variant * 100_000 + index)
        for index in range(len(sources))
    ]
    targets.reverse()
    return tuple(
        sorted(
            (
                wire.FormalUuidRenamingV3(domain[source], source, target)
                for source, target in zip(sources, targets, strict=True)
            ),
            key=lambda item: (item.namespace, item.source_uuid),
        )
    )


def test_source_commitment_roundtrip() -> None:
    global_catalog, binding = catalog((source_case(),))
    assert wire.verify_formal_global_transform_catalog_binding_v3(
        catalog=global_catalog, binding_receipt=binding
    )
    value = wire.source_case_commitment_v3(
        source_case(), transform_catalog_root=binding.transform_catalog_root
    )
    assert value.startswith("sha256:") and len(value) == 71


@pytest.mark.parametrize(
    "factory_name",
    (
        "identity_profile",
        "unit_profile",
        "coordinate_profile",
        "temporal_profile",
        "spatial_profile",
        "sampling_profile",
        "split_profile",
        "coarse_profile",
    ),
)
def test_eight_operation_wire_projection_and_claimed_output_differential(
    factory_name: str,
) -> None:
    original = _EXACT_CORE_FIXTURES[factory_name]()
    original_result = exact_core.run_exact_transform_core_v1(original)
    assert original_result.disposition is exact_core.CoreTransformDispositionV1.COMPLETE
    source = _wire_case_from_exact_core_profile(
        original,
        index=(
            "identity_profile",
            "unit_profile",
            "coordinate_profile",
            "temporal_profile",
            "spatial_profile",
            "sampling_profile",
            "split_profile",
            "coarse_profile",
        ).index(factory_name),
    )
    graph = source.graph
    projected, claimed = wire._project_transform_to_exact_core_v1(
        contract=graph.transforms[0],
        aggregate=graph.aggregate_maps[0],
        observations={item.observation_id: item for item in graph.observations},
        quantities={item.quantity_id: item for item in graph.quantities},
        contexts={item.context_id: item for item in graph.contexts},
    )
    verified = exact_core.verify_exact_transform_core_outputs_v1(projected, claimed)
    assert verified.disposition is exact_core.CoreTransformDispositionV1.COMPLETE
    assert projected.contract.operation is original.contract.operation

    def signature(item: exact_core.CoreComponentValueV1) -> tuple[object, ...]:
        return (
            item.value_kind.value,
            item.numeric_value,
            item.absolute_radius,
            item.boolean_value,
        )

    assert tuple(signature(item) for item in verified.output_components) == tuple(
        signature(item) for item in original_result.output_components
    )

    target_ref = graph.transforms[0].output_refs[0]
    target_index = next(
        index
        for index, observation in enumerate(graph.observations)
        if observation.observation_id == target_ref.observation_id
    )
    target = graph.observations[target_index]
    if target.numeric_value.available:
        values = target.numeric_value.value
        radii = target.uncertainty_radius.value
        assert type(values) is tuple and type(radii) is tuple
        changed_values = list(values)
        changed_values[target_ref.component_index] = _rational(
            Fraction(int(values[target_ref.component_index].numerator), int(values[target_ref.component_index].denominator))
            + 1
        )
        changed_observations = list(graph.observations)
        changed_observations[target_index] = replace(
            target,
            numeric_value=wire.NullableRationalVectorV3(
                True, tuple(changed_values)
            ),
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, observations=tuple(changed_observations))
        changed_radii = list(radii)
        changed_radii[target_ref.component_index] = _rational(
            Fraction(
                int(radii[target_ref.component_index].numerator),
                int(radii[target_ref.component_index].denominator),
            )
            + 1
        )
        changed_observations[target_index] = replace(
            target,
            uncertainty_radius=wire.NullableRationalVectorV3(
                True, tuple(changed_radii)
            ),
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, observations=tuple(changed_observations))


@pytest.mark.parametrize(
    "factory_name",
    (
        "identity_profile",
        "unit_profile",
        "coordinate_profile",
        "temporal_profile",
        "spatial_profile",
        "sampling_profile",
        "split_profile",
        "coarse_profile",
    ),
)
def test_eight_operations_each_build_decode_and_public_replay(
    factory_name: str,
) -> None:
    factory_names = (
        "identity_profile",
        "unit_profile",
        "coordinate_profile",
        "temporal_profile",
        "spatial_profile",
        "sampling_profile",
        "split_profile",
        "coarse_profile",
    )
    source = _wire_case_from_exact_core_profile(
        _EXACT_CORE_FIXTURES[factory_name](),
        index=600 + factory_names.index(factory_name),
    )
    global_catalog, binding = catalog((source,))
    result = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(source,),
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"B" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(result) is wire.FormalTrustedWireBuildSuccessV3
    decoded = wire.decode_and_audit_formal_envelope_v3(
        result.public_batch.envelopes[0].envelope
    )
    expected_operation = {
        "identity_profile": "identity",
        "unit_profile": "unit_conversion",
        "coordinate_profile": "coordinate_affine",
        "temporal_profile": "temporal_aggregation",
        "spatial_profile": "spatial_aggregation",
        "sampling_profile": "sampling_resolution",
        "split_profile": "equivalent_split_merge",
        "coarse_profile": "coarse_graining",
    }[factory_name]
    assert decoded.public_case.graph.transforms[0].operation == expected_operation
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (decoded.envelope,),
        key_source_receipt_id=result.public_batch.key_source_receipt_id,
    )
    assert type(replay) is wire.FormalTrustedWireBatchReplayV3
    assert replay.batch_id == result.public_batch.batch_id
    assert replay.catalog_membership_verified is False


@pytest.mark.parametrize(
    "factory_name",
    (
        "identity_profile",
        "unit_profile",
        "coordinate_profile",
        "temporal_profile",
        "spatial_profile",
        "sampling_profile",
        "split_profile",
        "coarse_profile",
    ),
)
def test_eight_operation_wire_adapter_rejects_typed_certificate_attacks(
    factory_name: str,
) -> None:
    graph = _wire_case_from_exact_core_profile(
        _EXACT_CORE_FIXTURES[factory_name](), index=20 + list(
            (
                "identity_profile",
                "unit_profile",
                "coordinate_profile",
                "temporal_profile",
                "spatial_profile",
                "sampling_profile",
                "split_profile",
                "coarse_profile",
            )
        ).index(factory_name)
    ).graph
    contract = graph.transforms[0]
    certificate = contract.certificate

    if factory_name == "identity_profile":
        numeric_row = contract.kernel_rows[0]
        for attacked_row in (
            replace(
                numeric_row,
                terms=(
                    replace(
                        numeric_row.terms[0],
                        coefficient=wire.RationalValueV3("2", "1"),
                    ),
                ),
            ),
            replace(numeric_row, offset=wire.RationalValueV3("1", "1")),
        ):
            with pytest.raises(ValueError, match="exact transform core rejected"):
                replace(
                    graph,
                    transforms=(
                        replace(
                            contract,
                            kernel_rows=(
                                attacked_row,
                                *contract.kernel_rows[1:],
                            ),
                        ),
                    ),
                )
        with pytest.raises(ValueError):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        discrete_mappings=contract.discrete_mappings[:-1],
                    ),
                ),
            )
        boolean_target = next(
            observation
            for observation in graph.observations
            if observation.scale_id == contract.target_scale_id
            and observation.boolean_value.available
        )
        attacked = tuple(
            replace(
                observation,
                boolean_value=wire.NullableBooleanV3(
                    True, not boolean_target.boolean_value.value
                ),
            )
            if observation.observation_id == boolean_target.observation_id
            else observation
            for observation in graph.observations
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, observations=attacked)
    elif factory_name == "unit_profile":
        assert type(certificate) is wire.FormalUnitConversionCertificateV3
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        certificate=replace(
                            certificate,
                            inverse_factor=wire.RationalValueV3("2", "1"),
                        ),
                    ),
                ),
            )
        first_row = contract.kernel_rows[0]
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        kernel_rows=(
                            replace(
                                first_row,
                                terms=(
                                    replace(
                                        first_row.terms[0],
                                        coefficient=wire.RationalValueV3(
                                            "3", "1"
                                        ),
                                    ),
                                ),
                            ),
                            *contract.kernel_rows[1:],
                        ),
                    ),
                ),
            )
        target_ref = contract.output_refs[0]
        target_quantity_id = next(
            item.quantity_id
            for item in graph.observations
            if item.observation_id == target_ref.observation_id
        )
        quantities = tuple(
            replace(
                quantity,
                unit_binding=replace(
                    quantity.unit_binding,
                    offset_to_si=wire.RationalValueV3("1", "1"),
                ),
            )
            if quantity.quantity_id == target_quantity_id
            else quantity
            for quantity in graph.quantities
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, quantities=quantities)
    elif factory_name == "coordinate_profile":
        assert type(certificate) is wire.FormalCoordinateAffineCertificateV3
        first = certificate.inverse_rows[0]
        term = first.terms[0]
        attacked_term = replace(
            term,
            coefficient=_rational(
                Fraction(int(term.coefficient.numerator), int(term.coefficient.denominator))
                + 1
            ),
        )
        attacked_certificate = replace(
            certificate,
            inverse_rows=(replace(first, terms=(attacked_term, *first.terms[1:])), *certificate.inverse_rows[1:]),
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, transforms=(replace(contract, certificate=attacked_certificate),))
    elif factory_name == "temporal_profile":
        assert type(certificate) is wire.FormalTemporalAggregationCertificateV3
        with pytest.raises(ValueError, match="weighted-mean basis"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        certificate=replace(
                            certificate, weighted_mean_basis="duration_inferred"
                        ),
                    ),
                ),
            )
        second_ref = contract.input_refs[1]
        second_context_id = next(
            item.context_id
            for item in graph.observations
            if item.observation_id == second_ref.observation_id
        )
        contexts = tuple(
            replace(
                context,
                temporal_support=wire.NullableTemporalSupportV3(
                    True,
                    wire.ExactTemporalSupportV3(
                        wire.RationalValueV3("3", "2"),
                        context.temporal_support.value.end,
                    ),
                ),
            )
            if context.context_id == second_context_id
            else context
            for context in graph.contexts
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, contexts=contexts)
    elif factory_name == "spatial_profile":
        assert type(certificate) is wire.FormalSpatialAggregationCertificateV3
        with pytest.raises(ValueError, match="weighted-mean basis"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        certificate=replace(
                            certificate, weighted_mean_basis="volume_inferred"
                        ),
                    ),
                ),
            )
        second_ref = contract.input_refs[1]
        second_context_id = next(
            item.context_id
            for item in graph.observations
            if item.observation_id == second_ref.observation_id
        )
        contexts = []
        for context in graph.contexts:
            if context.context_id != second_context_id:
                contexts.append(context)
                continue
            support = context.spatial_support.value
            assert type(support) is wire.ExactSpatialSupportV3
            contexts.append(
                replace(
                    context,
                    spatial_support=wire.NullableSpatialSupportV3(
                        True,
                        wire.ExactSpatialSupportV3(
                            (wire.RationalValueV3("1", "2"), support.lower[1]),
                            support.upper,
                        ),
                    ),
                )
            )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(graph, contexts=tuple(contexts))
    elif factory_name == "sampling_profile":
        assert type(certificate) is wire.FormalSamplingResolutionCertificateV3
        row = contract.kernel_rows[0]
        attacked_row = replace(
            row,
            terms=(
                wire.FormalExactTermV3(
                    certificate.discarded_inputs[0],
                    wire.RationalValueV3("1", "1"),
                ),
            ),
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        kernel_rows=(attacked_row, *contract.kernel_rows[1:]),
                    ),
                ),
            )
    elif factory_name == "split_profile":
        assert type(certificate) is wire.FormalEquivalentSplitMergeCertificateV3
        row = certificate.inverse_rows[0]
        attacked = replace(
            row,
            terms=(
                replace(
                    row.terms[0], coefficient=wire.RationalValueV3("2", "1")
                ),
                *row.terms[1:],
            ),
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        certificate=replace(certificate, inverse_rows=(attacked,)),
                    ),
                ),
            )
    else:
        assert type(certificate) is wire.FormalCoarseGrainingCertificateV3
        with pytest.raises(ValueError, match="weighted-mean basis"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        certificate=replace(
                            certificate, weighted_mean_basis="support_inferred"
                        ),
                    ),
                ),
            )
        row = certificate.target_commutation_rows[0]
        attacked = replace(
            row,
            terms=(
                replace(
                    row.terms[0], coefficient=wire.RationalValueV3("2", "1")
                ),
            ),
        )
        with pytest.raises(ValueError, match="exact transform core rejected"):
            replace(
                graph,
                transforms=(
                    replace(
                        contract,
                        certificate=replace(
                            certificate, target_commutation_rows=(attacked,)
                        ),
                    ),
                ),
            )


def test_external_core_complete_is_not_an_input_and_decode_freshly_verifies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cooperative core result cannot be injected as Wire authority."""
    core_profile = _EXACT_CORE_FIXTURES["identity_profile"]()
    external_result = exact_core.run_exact_transform_core_v1(core_profile)
    assert external_result.disposition is exact_core.CoreTransformDispositionV1.COMPLETE

    cases = (source_case(0),)
    global_catalog, binding = catalog(cases)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        wire.build_formal_trusted_wire_batch_v3(
            source_cases=cases,
            global_transform_catalog=global_catalog,
            transform_catalog_binding_receipt=binding,
            run_id=b"R" * 32,
            key_sources=keys(),
            key_source_receipt=receipt(),
            exact_transform_core_result=external_result,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        wire.FormalSourceCaseV3(
            graph=cases[0].graph,
            exact_transform_core_result=external_result,  # type: ignore[call-arg]
        )

    success = build_success(cases)
    original_verify = exact_core.verify_exact_transform_core_outputs_v1
    invocations: list[
        tuple[
            exact_core.ExactTransformCoreProfileV1,
            tuple[exact_core.CoreComponentValueV1, ...],
        ]
    ] = []

    def fresh_verify_spy(
        profile: exact_core.ExactTransformCoreProfileV1,
        claimed_outputs: tuple[exact_core.CoreComponentValueV1, ...],
    ) -> exact_core.ExactTransformCoreResultV1:
        invocations.append((profile, claimed_outputs))
        return original_verify(profile, claimed_outputs)

    monkeypatch.setattr(
        wire._exact_core_v1,
        "verify_exact_transform_core_outputs_v1",
        fresh_verify_spy,
    )
    decoded = wire.decode_and_audit_formal_envelope_v3(
        success.public_batch.envelopes[0].envelope
    )
    assert decoded.envelope_id == success.public_batch.envelope_ids[0]
    assert invocations
    assert all(profile is not core_profile for profile, _ in invocations)


def test_profile_ids_and_private_public_claim_boundary_are_synchronized() -> None:
    source_path = Path(__file__).parents[1] / "src" / "hegel_machine" / "phase2b_formal_trusted_wire_v3.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(source_tree))
    profile = json.loads(
        (Path(__file__).parents[1] / "config" / "phase2b_formal_trusted_wire_v3.json").read_text(
            encoding="utf-8"
        )
    )
    closed = profile["closed_public_schema"]
    assert closed["graph_completeness_policy_id"] == wire.FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3
    assert closed["field_classification_manifest_id"] == wire.FIELD_CLASSIFICATION_MANIFEST_ID_V3
    assert profile["private_build_and_variant_surface"]["public_batch_contains_private_alignment"] is False
    assert profile["private_build_and_variant_surface"][
        "standalone_variant_capability_exported_or_accepted"
    ] is False
    assert profile["private_build_and_variant_surface"][
        "variant_api_required_argument"
    ] == "FULL_FormalTrustedWireBuildSuccessV3"
    assert profile["global_transform_catalog_binding"]["standalone_decode_or_replay_catalog_membership_verified"] is False
    assert profile["claim_boundary"]["semantic_overlap_threshold_frozen"] is False
    assert profile["claim_boundary"]["allowed"] == wire.FORMAL_TRUSTED_WIRE_V3_CLAIM_LEVEL
    assert profile["claim_boundary"]["public_key_source_transcript_verified"] is False
    assert profile["claim_boundary"]["public_catalog_membership_verified"] is False
    assert profile["key_sources"]["public_batch_transcript_verified"] is False
    assert profile["closed_public_schema"]["transform_semantics"][
        "weighted_mean_basis"
    ] == wire.FORMAL_WEIGHTED_MEAN_BASIS_V3
    descriptor_profile = profile["closed_public_schema"]["semantic_descriptor"]
    assert descriptor_profile["unicode_data_version"] == wire.UNICODE_DATA_VERSION_V3
    assert descriptor_profile["default_ignorable_codepoint_count"] == 4_173
    assert descriptor_profile["representation_or_semantic_channel_eliminated"] is False
    string_policy = profile["closed_public_schema"]["public_string_field_policy"]
    assert string_policy["unrouted_string_leaf"] == "REJECT"
    assert string_policy[
        "policy_is_runtime_immutable_and_CONTENT_BOUND_IN_GRAPH_COMPLETENESS_ID"
    ] is True
    assert profile["identifier_assignment"][
        "public_uuid_disjoint_from_complete_source_uuid_domain"
    ] is True
    assert profile["identifier_assignment"][
        "quotient_aggregate_map_id_namespace"
    ] == "aggregate_map_id"
    assert profile["envelope"]["maximum_json_nesting_depth"] == 64
    assert profile["envelope"]["maximum_json_container_tokens"] == 16_384
    assert profile["envelope"]["decoded_dto_cache_rule"].startswith(
        "FRESH_NORMALIZE_RAW_CANONICAL_PAYLOAD"
    )
    assert profile["envelope"]["batch_and_replay_cache_rule"].startswith(
        "FRESH_DECODE_EVERY_ENVELOPE"
    )
    assert profile["private_audit_alignment"][
        "maximum_uuid_mappings_per_case"
    ] == wire.MAXIMUM_PRIVATE_UUID_MAPPINGS_PER_CASE_V3
    assert profile["private_build_and_variant_surface"][
        "variant_target_uuid_disjoint_from_complete_private_source_uuid_domain"
    ] is True
    assert profile["private_build_and_variant_surface"][
        "variant_full_rename_target_uuid_disjoint_from_base_public_uuid_domain"
    ] is True
    suite_profile = profile["variant_suite_surface"]
    assert suite_profile["schema_version"] == wire.FORMAL_VARIANT_SUITE_V3_SCHEMA_VERSION
    assert suite_profile["maximum_schedules"] == wire.MAXIMUM_VARIANT_SUITE_SCHEDULES_V3
    assert suite_profile["maximum_output_envelopes"] == wire.MAXIMUM_VARIANT_SUITE_ENVELOPES_V3
    assert suite_profile["maximum_output_bytes"] == wire.MAXIMUM_VARIANT_SUITE_OUTPUT_BYTES_V3
    assert suite_profile["maximum_schedule_tokens"] == wire.MAXIMUM_VARIANT_SUITE_SCHEDULE_TOKENS_V3
    assert suite_profile["early_resource_preflight"].startswith(
        "EXACT_OUTER_TUPLES_O1_LENGTH_CAPS_AND_CHECKED_ADD"
    )
    assert suite_profile["derived_cache_rule"].startswith(
        "FRESH_RECOMPUTE_SCHEDULE_ROOTS_VARIANT_IDS"
    )
    assert suite_profile["schedule_preimage_binding"].startswith(
        "BATCH_CARRIES_EXACT_SNAPSHOTTED_SCHEDULE"
    )
    assert suite_profile["validated_handle_token_or_cache_capability_exported"] is False
    assert suite_profile[
        "public_positive_structural_semantic_transcript_catalog_origin_audit_or_execution_claim"
    ] is False
    assert suite_profile["candidate_boundary"].startswith(
        "ATOMIC_CONTENT_ADDRESSED_CANDIDATE_BYTES"
    )
    assert suite_profile["consumer_requirement"].startswith(
        "EVERY_RETURNED_ENVELOPE_MUST_PASS_DECODE_AND_AUDIT"
    )
    assert profile["key_sources"]["batch_id_binding"].endswith(
        "KEY_SOURCE_RECEIPT_ID"
    )
    rfc_path = source_path.with_name("rfc8785_v1.py")
    assert profile["canonicalization_binding"][
        "implementation_sha256"
    ] == hashlib.sha256(rfc_path.read_bytes()).hexdigest()
    core_path = source_path.with_name("phase2b_exact_transform_core_v1.py")
    core_source_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()
    assert "sha256:" + core_source_hash == wire.EXACT_TRANSFORM_CORE_SOURCE_SHA256_V3
    core_profile = profile["exact_transform_core_binding"]
    assert core_profile["schema_version"] == exact_core.EXACT_TRANSFORM_CORE_VERSION_V1
    assert core_profile["implementation_sha256"] == core_source_hash
    core_test_path = Path(__file__).with_name("test_phase2b_exact_transform_core_v1.py")
    assert core_profile["focused_test_sha256"] == hashlib.sha256(
        core_test_path.read_bytes()
    ).hexdigest()
    assert core_profile["wire_entrypoint"] == "verify_exact_transform_core_outputs_v1"
    assert core_profile["complete_disposition_is_authority_or_origin_proof"] is False

    cases = (source_case(0),)
    global_catalog, binding = catalog(cases)
    success = build_success(cases)
    assert success.public_batch.key_source_transcript_verified is False
    assert success.public_batch.key_source_independence_verified is False
    assert wire.verify_formal_trusted_wire_build_success_v3(
        build_success=success,
        run_id=b"R" * 32,
        key_sources=keys(),
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
    )
    with pytest.raises(ValueError, match="run ID"):
        wire.verify_formal_trusted_wire_build_success_v3(
            build_success=success,
            run_id=b"X" * 32,
            key_sources=keys(),
            source_cases=cases,
            global_transform_catalog=global_catalog,
            transform_catalog_binding_receipt=binding,
        )
    with pytest.raises(ValueError, match="transcript|receipt"):
        wire.verify_formal_trusted_wire_build_success_v3(
            build_success=success,
            run_id=b"R" * 32,
            key_sources=wire.FormalWireKeySourcesV3(
                b"X" * 32, keys().id_ikm, keys().padding_ikm
            ),
            source_cases=cases,
            global_transform_catalog=global_catalog,
            transform_catalog_binding_receipt=binding,
        )
    with pytest.raises(ValueError, match="source commitments"):
        wire.verify_formal_trusted_wire_build_success_v3(
            build_success=success,
            run_id=b"R" * 32,
            key_sources=keys(),
            source_cases=(source_case(1),),
            global_transform_catalog=global_catalog,
            transform_catalog_binding_receipt=binding,
        )
    foreign_catalog, foreign_binding = catalog((source_case(1),))
    with pytest.raises(ValueError, match="catalog evidence"):
        wire.verify_formal_trusted_wire_build_success_v3(
            build_success=success,
            run_id=b"R" * 32,
            key_sources=keys(),
            source_cases=cases,
            global_transform_catalog=foreign_catalog,
            transform_catalog_binding_receipt=foreign_binding,
        )
    with pytest.raises(ValueError):
        replace(success.public_batch, key_source_transcript_verified=True)


def test_build_decode_replay_fixed_envelope() -> None:
    batch = build()
    assert all(len(row.envelope) == 65_536 for row in batch.envelopes)
    replay = wire.replay_formal_trusted_wire_batch_v3(
        tuple(row.envelope for row in batch.envelopes),
        key_source_receipt_id=batch.key_source_receipt_id,
    )
    assert type(replay) is wire.FormalTrustedWireBatchReplayV3
    assert replay.batch_id == batch.batch_id
    assert replay.formal_covert_audit is False


def test_batch_id_binds_exact_key_receipt_id_on_every_replay() -> None:
    batch = build((source_case(0),))
    independent_body = {
        "batch_context": asdict(batch.batch_context),
        "envelope_ids_in_public_order": list(batch.envelope_ids),
        "key_source_receipt_id": batch.key_source_receipt_id,
        "schema_version": wire.FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,
    }
    independent_batch_id = (
        "phase2b_formal_trusted_wire_batch_v3_"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/BATCH/V3\x00"
            + wire.canonicalize_rfc8785_v1(independent_body)
        ).hexdigest()
    )
    assert batch.batch_id == independent_batch_id

    other_receipt = wire.issue_key_source_transcript_receipt_v3(
        key_sources=keys(),
        latent_seed_commitments=(wire.seed_commitment_v3(b"M" * 32),),
        answer_seed_commitments=(wire.seed_commitment_v3(b"N" * 32),),
    )
    assert other_receipt.receipt_id != batch.key_source_receipt_id
    with pytest.raises(ValueError, match="content ID"):
        replace(batch, key_source_receipt_id=other_receipt.receipt_id)

    public_only = wire.replay_formal_trusted_wire_batch_v3(
        tuple(row.envelope for row in batch.envelopes),
        key_source_receipt_id=other_receipt.receipt_id,
    )
    assert type(public_only) is wire.FormalTrustedWireBatchReplayV3
    assert public_only.batch_id != batch.batch_id
    assert public_only.key_source_transcript_verified is False
    with pytest.raises(ValueError, match="identity"):
        replace(public_only, key_source_receipt_id=batch.key_source_receipt_id)


def test_private_alignment_reorders_and_rejects_splice() -> None:
    success = build_success()
    batch = success.public_batch
    alignment = success.private_audit_alignment
    assert wire.verify_private_audit_alignment_v3(batch=batch, alignment=alignment)
    labels = (b"source-zero", b"source-one")
    issued = wire.issue_formal_private_audit_rows_v3(
        build_success=success,
        private_payloads_in_source_order=labels,
    )
    aligned = wire.align_private_rows_to_public_v3(
        batch=batch,
        alignment=alignment,
        private_rows_in_source_order=issued,
    )
    assert tuple(row.private_payload_bytes for row in aligned) == tuple(
        labels[index] for index in alignment.source_row_index_by_public_row
    )
    first = alignment.links_in_public_order[0]
    bad = unsafe_alignment_copy(
        alignment,
        links_in_public_order=(
            replace(first, public_envelope_sha256="sha256:" + "0" * 64),
        ) + alignment.links_in_public_order[1:],
    )
    with pytest.raises(ValueError):
        wire.verify_private_audit_alignment_v3(batch=batch, alignment=bad)


def test_private_uuid_projection_exact_coverage_and_attacks() -> None:
    case = source_case(0)
    success = build_success((case,))
    alignment = success.private_audit_alignment
    link = alignment.links_in_public_order[0]
    source_label = {
        "binding": {
            "entity_id": case.graph.entities[0].entity_id,
            "role_candidate_id": case.graph.role_candidate_ids[0],
            "scale_id": case.graph.scales[0].scale_id,
        },
        "answer_kind": "fixture-only",
    }
    projected_bytes = wire.project_private_source_label_to_public_v3(
        build_success=success,
        source_row_index=0,
        canonical_source_label_bytes=wire.canonicalize_rfc8785_v1(source_label),
    )
    projected = wire.decode_and_require_canonical_rfc8785_v1(projected_bytes)
    assert wire.project_private_source_labels_to_public_order_v3(
        build_success=success,
        canonical_source_label_bytes_in_source_order=(
            wire.canonicalize_rfc8785_v1(source_label),
        ),
    ) == (projected_bytes,)
    projected_rows = wire.issue_projected_private_audit_rows_v3(
        build_success=success,
        canonical_source_label_bytes_in_source_order=(
            wire.canonicalize_rfc8785_v1(source_label),
        ),
    )
    assert tuple(row.private_payload_bytes for row in projected_rows) == (
        projected_bytes,
    )
    assert projected_rows[0].source_case_commitment == link.source_case_commitment
    assert projected_rows[0].label_binding_id == link.label_binding_id
    assert projected_rows[0].semantic_truth_verified is False
    for wrong_coverage in ((), (b"{}", b"{}")):
        with pytest.raises(ValueError, match="exactly cover"):
            wire.project_private_source_labels_to_public_order_v3(
                build_success=success,
                canonical_source_label_bytes_in_source_order=wrong_coverage,
            )
    mapping = {
        row.source_uuid: row.public_uuid for row in link.source_to_public_uuid_rows
    }
    expected_source_uuid_rows = {
        (namespace, value)
        for _, namespace, value in wire._walk_uuid_leaves_v3(
            wire._profile_value_v3(case.graph)
        )
    }
    assert {
        (row.namespace, row.source_uuid)
        for row in link.source_to_public_uuid_rows
    } == expected_source_uuid_rows
    assert {row.namespace for row in link.source_to_public_uuid_rows} == set(
        wire.NAMESPACE_REGISTRY_V3
    )
    assert len(link.source_to_public_uuid_rows) > len(wire.NAMESPACE_REGISTRY_V3)
    assert projected["binding"] == {
        "entity_id": mapping[case.graph.entities[0].entity_id],
        "role_candidate_id": mapping[case.graph.role_candidate_ids[0]],
        "scale_id": mapping[case.graph.scales[0].scale_id],
    }
    for attack in (
        {"entity_id": _uuid(999_999_901)},
        {"scale_id": case.graph.entities[0].entity_id},
    ):
        with pytest.raises(ValueError):
            wire.project_private_source_label_to_public_v3(
                build_success=success,
                source_row_index=0,
                canonical_source_label_bytes=wire.canonicalize_rfc8785_v1(attack),
            )
    for embedded_attack in (
        {"note": "embedded=" + case.graph.entities[0].entity_id + "."},
        {"note": "digest=" + ZERO_DIGEST + "."},
        {"note": "raw=" + "a" * 64 + "."},
        {"field-" + case.graph.entities[0].entity_id: "value"},
    ):
        with pytest.raises(ValueError, match="embedded identifier/digest"):
            wire.project_private_source_label_to_public_v3(
                build_success=success,
                source_row_index=0,
                canonical_source_label_bytes=wire.canonicalize_rfc8785_v1(
                    embedded_attack
                ),
            )

    rows = link.source_to_public_uuid_rows
    with pytest.raises(ValueError, match="case cap"):
        replace(
            link,
            source_to_public_uuid_rows=(rows[0],)
            * (wire.MAXIMUM_PRIVATE_UUID_MAPPINGS_PER_CASE_V3 + 1),
        )
    with pytest.raises(ValueError):
        replace(link, source_to_public_uuid_rows=rows[:-1])
    with pytest.raises(ValueError):
        replace(link, source_to_public_uuid_rows=rows + (rows[0],))
    changed_row = replace(rows[0], public_uuid=_uuid(999_999_902))
    changed_rows = tuple(
        sorted((changed_row,) + rows[1:], key=lambda row: (row.namespace, row.source_uuid))
    )
    changed_label_id = wire._label_binding_id_v3(
        batch_id=alignment.batch_id,
        public_envelope_sha256=link.public_envelope_sha256,
        source_row_index=link.source_row_index,
        source_case_commitment=link.source_case_commitment,
        source_to_public_uuid_rows=changed_rows,
    )
    changed_link = replace(
        link,
        source_to_public_uuid_rows=changed_rows,
        label_binding_id=changed_label_id,
    )
    changed_links = (changed_link,)
    body = wire._alignment_body_v3(
        batch_id=alignment.batch_id,
        run_id_commitment=alignment.run_id_commitment,
        key_source_receipt_id=alignment.key_source_receipt_id,
        source_case_count=alignment.source_case_count,
        links=changed_links,
    )
    forged = unsafe_alignment_copy(
        alignment,
        links_in_public_order=changed_links,
        alignment_id=wire._alignment_id_v3(body),
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        wire.verify_private_audit_alignment_v3(
            batch=success.public_batch,
            alignment=forged,
        )


def test_standard_error_and_1025_are_atomic_empty() -> None:
    standard_cases = (source_case(0), source_case(1, standard_error=True))
    global_catalog, binding = catalog(standard_cases)
    result = wire.build_formal_trusted_wire_batch_v3(
        source_cases=standard_cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(result) is wire.FormalTrustedWireBatchRejectionV3
    assert result.reason == "STANDARD_ERROR_UNSUPPORTED"
    assert result.envelopes == () and result.batch_id is None
    global_catalog, binding = catalog((source_case(0),))
    over = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(source_case(0),) * 1_025,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(over) is wire.FormalTrustedWireBatchRejectionV3
    assert over.envelopes == () and over.batch_id is None


def test_standard_error_cannot_bypass_any_operation_literal_gate() -> None:
    factories = (
        "identity_profile",
        "unit_profile",
        "coordinate_profile",
        "temporal_profile",
        "spatial_profile",
        "sampling_profile",
        "split_profile",
        "coarse_profile",
    )
    hidden = "prefix " + _uuid(8_888_888_888) + " suffix"
    key_receipt = receipt()
    for factory_index, factory_name in enumerate(factories):
        source = _wire_case_from_exact_core_profile(
            _EXACT_CORE_FIXTURES[factory_name](), index=400 + factory_index
        )
        graph = source.graph
        standard_observations = tuple(
            replace(observation, uncertainty_model="standard_error")
            if observation.numeric_value.available
            else observation
            for observation in graph.observations
        )
        standard_graph = replace(graph, observations=standard_observations)
        standard_source = wire.FormalSourceCaseV3(standard_graph)
        global_catalog, binding = catalog((standard_source,))
        certificate = standard_graph.transforms[0].certificate
        policy = wire.FORMAL_PUBLIC_STRING_FIELD_POLICY_V3[
            type(certificate).__name__
        ]
        literal_fields = tuple(
            name
            for name, route in policy.items()
            if route in {"ENUM_LITERAL", "EXACT_LITERAL"}
        )
        assert "semantics_version" in literal_fields
        for field_name in literal_fields:
            attacked_certificate = unsafe_issued_copy(
                certificate, **{field_name: hidden}
            )
            attacked_contract = unsafe_issued_copy(
                standard_graph.transforms[0], certificate=attacked_certificate
            )
            attacked_graph = unsafe_issued_copy(
                standard_graph, transforms=(attacked_contract,)
            )
            attacked_source = unsafe_issued_copy(
                standard_source, graph=attacked_graph
            )
            result = wire.build_formal_trusted_wire_batch_v3(
                source_cases=(attacked_source,),
                global_transform_catalog=global_catalog,
                transform_catalog_binding_receipt=binding,
                run_id=b"R" * 32,
                key_sources=keys(),
                key_source_receipt=key_receipt,
            )
            assert type(result) is wire.FormalTrustedWireBatchRejectionV3
            assert result.reason == "formal_wire_preflight_or_atomic_build_failed"
            assert result.envelopes == () and result.batch_id is None


def test_no_source_uuid_or_private_alignment_in_public_bytes() -> None:
    cases = (source_case(0), source_case(1))
    success = build_success(cases)
    batch = success.public_batch
    global_catalog = success.variant_context.global_transform_catalog
    catalog_binding = success.variant_context.transform_catalog_binding_receipt
    public = b"".join(row.envelope for row in batch.envelopes)
    for case in cases:
        mapping = wire._profile_value_v3(case.graph)
        for _, _, source_id in wire._walk_uuid_leaves_v3(mapping):
            assert source_id.encode("ascii") not in public
    assert success.private_audit_alignment.alignment_id.encode("ascii") not in public
    assert batch.key_source_receipt_id.encode("ascii") not in public
    assert catalog_binding.receipt_id.encode("ascii") not in public
    for entry in global_catalog.entries:
        assert entry.exact_source_transform_use_sha256.encode("ascii") not in public
    for entry_commitment in catalog_binding.catalog_entry_commitments:
        assert entry_commitment.encode("ascii") not in public
    assert success.variant_context.context_id.encode("ascii") not in public
    public_profile = repr(asdict(batch))
    assert "private_audit_alignment" not in public_profile
    assert "variant_capability" not in public_profile
    assert "variant_context" not in public_profile


def test_variant_normalizer_recomputes_uuid_and_order_provenance() -> None:
    success = build_success((source_case(0),))
    batch = success.public_batch
    decoded = wire.decode_and_audit_formal_envelope_v3(batch.envelopes[0].envelope)
    reversed_order = tuple(
        reversed(tuple(item.observation_id for item in decoded.public_case.graph.observations))
    )
    schedule = identity_variant_schedule(
        success,
        observation_order_by_envelope={decoded.envelope_id: reversed_order},
    )
    expected = wire.make_formal_variant_payloads_v3(
        build_success=success,
        schedule=schedule,
    )[0]
    normalized = wire.normalize_formal_variant_payload_v3(
        deepcopy(expected.payload),
        build_success=success,
        schedule=schedule,
        envelope_id=decoded.envelope_id,
    )
    assert tuple(
        item["observation_id"]
        for item in normalized["public_case"]["graph"]["observations"]
    ) == reversed_order
    assert normalized["public_case"]["public_provenance_commitment"] != ZERO_DIGEST
    envelope = wire.reencode_formal_variant_payload_envelope_v3(
        payload=normalized,
        padding_bytes=expected.padding_bytes,
        build_success=success,
        schedule=schedule,
        envelope_id=decoded.envelope_id,
    )
    assert len(envelope) == 65_536
    assert wire.decode_and_audit_formal_envelope_v3(envelope).catalog_membership_verified is False


def test_variant_targets_cannot_reveal_private_source_uuid_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = build_success((source_case(0), source_case(1)))
    reveal_source_ids = tuple(
        sorted(
            (
                wire.FormalUuidRenamingV3(
                    mapping.namespace,
                    mapping.public_uuid,
                    mapping.source_uuid,
                )
                for link in success.private_audit_alignment.links_in_public_order
                for mapping in link.source_to_public_uuid_rows
            ),
            key=lambda row: (row.namespace, row.source_uuid),
        )
    )
    schedule = identity_variant_schedule(
        success, renamings=reveal_source_ids
    )
    frame_calls: list[tuple[bytes, bytes]] = []
    original_frame = wire._frame_payload_with_padding_v3

    def recording_frame(payload: bytes, padding: bytes) -> bytes:
        frame_calls.append((payload, padding))
        return original_frame(payload, padding)

    monkeypatch.setattr(wire, "_frame_payload_with_padding_v3", recording_frame)
    with pytest.raises(ValueError, match="private source UUIDs"):
        wire.make_formal_variant_payloads_v3(
            build_success=success,
            schedule=schedule,
        )
    with pytest.raises(ValueError, match="private source UUIDs"):
        wire.reencode_formal_variant_batch_v3(
            build_success=success,
            schedule=schedule,
        )
    assert frame_calls == []

    safe_renamings = total_nonmonotonic_renamings(success, 123)
    private_source_ids = {
        mapping.source_uuid
        for link in success.private_audit_alignment.links_in_public_order
        for mapping in link.source_to_public_uuid_rows
    }
    assert not (
        {row.target_uuid for row in safe_renamings} & private_source_ids
    )
    safe_schedule = identity_variant_schedule(
        success, renamings=safe_renamings
    )
    assert len(
        wire.make_formal_variant_payloads_v3(
            build_success=success,
            schedule=safe_schedule,
        )
    ) == 2


def test_variant_suite_roots_outputs_and_claim_boundary_are_exact() -> None:
    suite_signature = inspect.signature(wire.reencode_formal_variant_suite_v3)
    assert tuple(suite_signature.parameters) == ("build_success", "schedules")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in suite_signature.parameters.values()
    )
    assert "reencode_formal_variant_suite_v3" in wire.__all__
    assert not any(
        "prepared" in name.lower() or "validatedvariantsnapshot" in name.lower()
        for name in wire.__all__
    )
    success = build_success((source_case(0), source_case(1)))
    identity = identity_variant_schedule(success)
    renamed = identity_variant_schedule(
        success, renamings=total_nonmonotonic_renamings(success, 201)
    )
    reversed_observations: dict[str, tuple[str, ...]] = {}
    for row in success.public_batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(row.envelope)
        reversed_observations[row.envelope_id] = tuple(
            reversed(
                tuple(
                    observation.observation_id
                    for observation in decoded.public_case.graph.observations
                )
            )
        )
    reordered = identity_variant_schedule(
        success,
        case_order=tuple(reversed(success.public_batch.envelope_ids)),
        observation_order_by_envelope=reversed_observations,
    )
    schedules = (identity, renamed, reordered)
    sequential = tuple(
        wire.reencode_formal_variant_batch_v3(
            build_success=success, schedule=schedule
        )
        for schedule in schedules
    )
    suite = wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=schedules
    )
    assert type(suite) is wire.FormalVariantSuiteV3
    assert tuple(batch.envelopes for batch in suite.batches) == sequential
    assert len(suite.schedule_roots) == len(suite.batch_envelope_roots) == 3
    assert len(set(suite.schedule_roots)) == 3
    assert all(
        value is False
        for value in (
            suite.key_source_transcript_verified,
            suite.catalog_membership_verified,
            suite.origin_authenticated,
            suite.formal_covert_audit,
            suite.actual_execution_claimed,
        )
    )
    assert not hasattr(suite, "structural_replay_verified")
    for batch in suite.batches:
        for envelope in batch.envelopes:
            decoded = wire.decode_and_audit_formal_envelope_v3(envelope)
            assert decoded.structural_replay_verified is True
            assert decoded.catalog_membership_verified is False

    def schedule_root(schedule: wire.FormalVariantScheduleV3) -> str:
        body = {
            "case_order_envelope_ids": list(schedule.case_order_envelope_ids),
            "observation_orders": [
                {
                    "envelope_id": row.envelope_id,
                    "observation_ids": list(row.observation_ids),
                }
                for row in schedule.observation_orders
            ],
            "schema_version": wire.FORMAL_VARIANT_SUITE_V3_SCHEMA_VERSION,
            "uuid_renamings": [
                {
                    "namespace": row.namespace,
                    "source_uuid": row.source_uuid,
                    "target_uuid": row.target_uuid,
                }
                for row in schedule.uuid_renamings
            ],
        }
        return "sha256:" + hashlib.sha256(
            b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/VARIANT_SCHEDULE_ROOT/V3\x00"
            + wire.canonicalize_rfc8785_v1(body)
        ).hexdigest()

    expected_schedule_roots = tuple(schedule_root(row) for row in schedules)
    assert suite.schedule_roots == expected_schedule_roots
    expected_batch_roots: list[str] = []
    for schedule, batch in zip(schedules, suite.batches, strict=True):
        expected_variant_ids = [
            "phase2b_formal_trusted_envelope_v3_"
            + hashlib.sha256(
                b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/ENVELOPE/V3\x00"
                + envelope
            ).hexdigest()
            for envelope in batch.envelopes
        ]
        expected_envelope_sha = [
            "sha256:" + hashlib.sha256(envelope).hexdigest()
            for envelope in batch.envelopes
        ]
        body = {
            "base_envelope_ids_in_order": list(
                schedule.case_order_envelope_ids
            ),
            "envelope_sha256_in_order": expected_envelope_sha,
            "schedule_root": schedule_root(schedule),
            "schema_version": wire.FORMAL_VARIANT_SUITE_V3_SCHEMA_VERSION,
            "source_batch_id": success.public_batch.batch_id,
            "variant_envelope_ids_in_order": expected_variant_ids,
        }
        assert list(batch.variant_envelope_ids_in_order) == expected_variant_ids
        assert list(batch.envelope_sha256_in_order) == expected_envelope_sha
        expected_batch_roots.append(
            "sha256:"
            + hashlib.sha256(
                b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/VARIANT_BATCH_ENVELOPE_ROOT/V3\x00"
                + wire.canonicalize_rfc8785_v1(body)
            ).hexdigest()
        )
    assert suite.batch_envelope_roots == tuple(expected_batch_roots)
    suite_body = {
        "batch_envelope_roots": expected_batch_roots,
        "build_success_id": success.success_id,
        "schedule_roots": list(expected_schedule_roots),
        "schema_version": wire.FORMAL_VARIANT_SUITE_V3_SCHEMA_VERSION,
        "source_batch_id": success.public_batch.batch_id,
    }
    assert suite.suite_root == (
        "phase2b_formal_variant_suite_v3_"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/VARIANT_SUITE_ROOT/V3\x00"
            + wire.canonicalize_rfc8785_v1(suite_body)
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="claim boundary"):
        replace(suite, formal_covert_audit=True)

    def forge_cached_field(value: object, name: str, forged: object) -> object:
        result = object.__new__(type(value))
        for item in fields(value):
            try:
                current = getattr(value, item.name)
            except AttributeError:
                continue
            object.__setattr__(result, item.name, current)
        object.__setattr__(result, name, forged)
        return result

    first_batch = suite.batches[0]
    direct_batch = wire.FormalVariantSuiteBatchV3(
        source_batch_id=success.public_batch.batch_id,
        schedule=schedules[0],
        base_envelope_ids_in_order=schedules[0].case_order_envelope_ids,
        envelopes=first_batch.envelopes,
    )
    assert direct_batch.schedule == schedules[0]
    assert direct_batch.schedule is not schedules[0]
    assert direct_batch.schedule_root == schedule_root(schedules[0])
    assert direct_batch.batch_envelope_root == first_batch.batch_envelope_root
    direct_suite = wire.FormalVariantSuiteV3(
        schema_version=wire.FORMAL_VARIANT_SUITE_V3_SCHEMA_VERSION,
        source_batch_id=success.public_batch.batch_id,
        build_success_id=success.success_id,
        batches=(direct_batch,),
    )
    assert direct_suite.schedule_roots == (schedule_root(schedules[0]),)
    assert direct_suite.batch_envelope_roots == (
        direct_batch.batch_envelope_root,
    )
    assert "schedule_root" not in inspect.signature(
        wire.FormalVariantSuiteBatchV3
    ).parameters
    with pytest.raises(TypeError, match="unexpected keyword"):
        wire.FormalVariantSuiteBatchV3(  # type: ignore[call-arg]
            source_batch_id=success.public_batch.batch_id,
            schedule=schedules[0],
            schedule_root=ZERO_DIGEST,
            base_envelope_ids_in_order=schedules[0].case_order_envelope_ids,
            envelopes=first_batch.envelopes,
        )
    with pytest.raises(ValueError, match="schedule preimage"):
        wire.FormalVariantSuiteBatchV3(
            source_batch_id=success.public_batch.batch_id,
            schedule=schedules[0],
            base_envelope_ids_in_order=tuple(
                reversed(schedules[0].case_order_envelope_ids)
            ),
            envelopes=first_batch.envelopes,
        )
    forged_preimage = forge_cached_field(first_batch, "schedule", schedules[1])
    with pytest.raises(ValueError, match="cached schedule root"):
        forged_preimage.__post_init__()  # type: ignore[attr-defined]
    for name, forged in (
        ("schedule_root", ZERO_DIGEST),
        ("variant_envelope_ids_in_order", ("forged",) * len(first_batch.envelopes)),
        ("envelope_sha256_in_order", (ZERO_DIGEST,) * len(first_batch.envelopes)),
        ("batch_envelope_root", ZERO_DIGEST),
    ):
        forged_batch = forge_cached_field(first_batch, name, forged)
        with pytest.raises(ValueError, match="cached"):
            forged_batch.__post_init__()  # type: ignore[attr-defined]
    for name, forged in (
        ("schedule_roots", (ZERO_DIGEST,) * len(suite.batches)),
        ("batch_envelope_roots", (ZERO_DIGEST,) * len(suite.batches)),
        ("suite_root", "phase2b_formal_variant_suite_v3_" + "0" * 64),
    ):
        forged_suite = forge_cached_field(suite, name, forged)
        with pytest.raises(ValueError, match="cached"):
            forged_suite.__post_init__()  # type: ignore[attr-defined]
    for claim in (
        "key_source_transcript_verified",
        "catalog_membership_verified",
        "origin_authenticated",
        "formal_covert_audit",
        "actual_execution_claimed",
    ):
        with pytest.raises(ValueError, match="claim boundary"):
            replace(suite, **{claim: True})


def test_variant_suite_resource_preflight_is_length_first_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_envelope_id = "phase2b_formal_trusted_envelope_v3_" + "1" * 64

    def forged_schedule(
        *, uuid_rows: tuple[object, ...], observation_rows: tuple[object, ...]
    ) -> wire.FormalVariantScheduleV3:
        result = object.__new__(wire.FormalVariantScheduleV3)
        object.__setattr__(result, "case_order_envelope_ids", (base_envelope_id,))
        object.__setattr__(result, "uuid_renamings", uuid_rows)
        object.__setattr__(result, "observation_orders", observation_rows)
        return result

    class AttributeSpy:
        accesses = 0

        def __getattribute__(self, name: str) -> object:
            type(self).accesses += 1
            raise AssertionError("oversized row content was accessed: " + name)

    class UnsafeTuple(tuple):
        iterations = 0

        def __iter__(self) -> object:
            type(self).iterations += 1
            raise AssertionError("unsafe tuple was traversed")

    unsafe = forged_schedule(
        uuid_rows=UnsafeTuple((AttributeSpy(),)), observation_rows=()
    )
    with pytest.raises(TypeError, match="exact tuple"):
        unsafe.__post_init__()
    with pytest.raises(TypeError, match="containers must be tuples"):
        wire._snapshot_variant_schedules_v3((unsafe,))
    assert UnsafeTuple.iterations == AttributeSpy.accesses == 0

    outer_spies = tuple(AttributeSpy() for _ in range(81))
    with pytest.raises(ValueError, match="outside 1 through 80"):
        wire._snapshot_variant_schedules_v3(outer_spies)  # type: ignore[arg-type]
    assert AttributeSpy.accesses == 0

    monkeypatch.setattr(wire, "MAXIMUM_VARIANT_SUITE_SCHEDULE_TOKENS_V3", 5)
    oversized_uuid = forged_schedule(
        uuid_rows=tuple(AttributeSpy() for _ in range(5)), observation_rows=()
    )
    with pytest.raises(ValueError, match="token budget"):
        oversized_uuid.__post_init__()
    with pytest.raises(ValueError, match="token budget"):
        wire._snapshot_variant_schedules_v3((oversized_uuid,))
    with pytest.raises(ValueError, match="token budget"):
        wire.FormalVariantSuiteBatchV3(
            source_batch_id="phase2b_formal_trusted_wire_batch_v3_" + "1" * 64,
            schedule=oversized_uuid,
            base_envelope_ids_in_order=(base_envelope_id,),
            envelopes=(b"\x00" * wire.ENVELOPE_BYTES_V3,),
        )
    assert AttributeSpy.accesses == 0

    observation_row = object.__new__(wire.FormalObservationOrderV3)
    object.__setattr__(
        observation_row,
        "observation_ids",
        (_uuid(1), _uuid(2), _uuid(3)),
    )
    oversized_observation = forged_schedule(
        uuid_rows=(), observation_rows=(observation_row,)
    )
    with pytest.raises(ValueError, match="token budget"):
        oversized_observation.__post_init__()
    with pytest.raises(ValueError, match="token budget"):
        wire._snapshot_variant_schedules_v3((oversized_observation,))


def test_variant_suite_bounds_attacks_atomicity_and_one_time_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = build_success((source_case(0), source_case(1)))
    identity = identity_variant_schedule(success)
    safe = identity_variant_schedule(
        success, renamings=total_nonmonotonic_renamings(success, 202)
    )
    with pytest.raises(ValueError, match="outside 1 through 80"):
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=()
        )
    with pytest.raises(ValueError):
        wire.reencode_formal_variant_suite_v3(  # type: ignore[arg-type]
            build_success=success, schedules=[identity]
        )
    with pytest.raises(ValueError, match="outside 1 through 80"):
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(identity,) * 81
        )
    identity_duplicate = wire.FormalVariantScheduleV3(
        tuple(identity.case_order_envelope_ids),
        tuple(
            wire.FormalUuidRenamingV3(
                row.namespace, row.source_uuid, row.target_uuid
            )
            for row in identity.uuid_renamings
        ),
        tuple(
            wire.FormalObservationOrderV3(
                row.envelope_id, tuple(row.observation_ids)
            )
            for row in identity.observation_orders
        ),
    )
    assert identity_duplicate is not identity
    with pytest.raises(ValueError, match="repeats a schedule"):
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(identity, identity_duplicate)
        )

    explicit_base_targets: list[wire.FormalUuidRenamingV3] = []
    for envelope in success.public_batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(envelope.envelope)
        explicit_base_targets.extend(
            wire.FormalUuidRenamingV3(namespace, value, value)
            for _, namespace, value in wire._walk_uuid_leaves_v3(
                decoded.payload["public_case"]["graph"]
            )
        )
    explicit_base_schedule = identity_variant_schedule(
        success,
        renamings=tuple(
            sorted(
                set(explicit_base_targets),
                key=lambda row: (row.namespace, row.source_uuid),
            )
        ),
    )
    with pytest.raises(ValueError, match="base public UUID domain"):
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(explicit_base_schedule,)
        )
    partial = identity_variant_schedule(
        success, renamings=safe.uuid_renamings[:-1]
    )
    with pytest.raises(ValueError, match="does not exactly cover"):
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(partial,)
        )

    snapshot_calls = 0
    closure_calls = 0
    suite_closure_calls = 0
    original_snapshot = wire._snapshot_and_validate_build_success_for_variants_v3
    original_closure = wire._verify_decoded_batch_closure_v3
    original_suite_closure = wire._verify_formal_variant_suite_batch_replay_v3

    def count_snapshot(value: wire.FormalTrustedWireBuildSuccessV3) -> object:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(value)

    def count_closure(decoded: object, context: object) -> None:
        nonlocal closure_calls
        closure_calls += 1
        original_closure(decoded, context)  # type: ignore[arg-type]

    def count_suite_closure(*, replayed: object, expected_context: object) -> None:
        nonlocal suite_closure_calls
        suite_closure_calls += 1
        original_suite_closure(  # type: ignore[arg-type]
            replayed=replayed, expected_context=expected_context
        )

    monkeypatch.setattr(
        wire,
        "_snapshot_and_validate_build_success_for_variants_v3",
        count_snapshot,
    )
    monkeypatch.setattr(wire, "_verify_decoded_batch_closure_v3", count_closure)
    monkeypatch.setattr(
        wire,
        "_verify_formal_variant_suite_batch_replay_v3",
        count_suite_closure,
    )
    suite = wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=(identity, safe)
    )
    assert len(suite.batches) == 2
    assert snapshot_calls == 1
    # One closure is part of the one-time public-batch reconstruction; each
    # schedule then has one independent variant-batch closure.
    assert closure_calls >= 3
    assert suite_closure_calls == 2

    monkeypatch.setattr(
        wire,
        "_snapshot_and_validate_build_success_for_variants_v3",
        original_snapshot,
    )
    monkeypatch.setattr(wire, "_verify_decoded_batch_closure_v3", original_closure)
    monkeypatch.setattr(
        wire,
        "_verify_formal_variant_suite_batch_replay_v3",
        original_suite_closure,
    )

    exact_core_calls = 0
    original_exact_core_verify = (
        wire._exact_core_v1.verify_exact_transform_core_outputs_v1
    )

    def count_exact_core(*args: object, **kwargs: object) -> object:
        nonlocal exact_core_calls
        exact_core_calls += 1
        return original_exact_core_verify(*args, **kwargs)

    monkeypatch.setattr(
        wire._exact_core_v1,
        "verify_exact_transform_core_outputs_v1",
        count_exact_core,
    )
    wire._prepare_variant_validation_v3(success)
    preparation_exact_core_calls = exact_core_calls
    assert preparation_exact_core_calls > 0
    exact_core_calls = 0
    wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=(identity, safe)
    )
    assert exact_core_calls == preparation_exact_core_calls
    monkeypatch.setattr(
        wire._exact_core_v1,
        "verify_exact_transform_core_outputs_v1",
        original_exact_core_verify,
    )
    baseline = wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=(identity, safe)
    )
    original_prepare = wire._prepare_variant_validation_v3

    def mutate_after_schedule_snapshot(
        value: wire.FormalTrustedWireBuildSuccessV3,
    ) -> object:
        object.__setattr__(identity, "case_order_envelope_ids", ())
        return original_prepare(value)

    monkeypatch.setattr(
        wire, "_prepare_variant_validation_v3", mutate_after_schedule_snapshot
    )
    replayed = wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=(identity, safe)
    )
    assert replayed.suite_root == baseline.suite_root
    assert tuple(row.envelopes for row in replayed.batches) == tuple(
        row.envelopes for row in baseline.batches
    )
    monkeypatch.setattr(wire, "_prepare_variant_validation_v3", original_prepare)

    original_frame = wire._frame_payload_with_padding_v3
    frame_calls = 0

    def fail_in_second_schedule(payload: bytes, padding: bytes) -> bytes:
        nonlocal frame_calls
        frame_calls += 1
        if frame_calls == len(success.public_batch.envelopes) + 1:
            raise RuntimeError("injected atomic suite failure")
        return original_frame(payload, padding)

    monkeypatch.setattr(
        wire, "_frame_payload_with_padding_v3", fail_in_second_schedule
    )
    partial_result: wire.FormalVariantSuiteV3 | None = None
    with pytest.raises(RuntimeError, match="injected atomic suite failure"):
        partial_result = wire.reencode_formal_variant_suite_v3(
            build_success=success,
            schedules=(identity_variant_schedule(success), safe),
        )
    assert partial_result is None
    assert frame_calls == len(success.public_batch.envelopes) + 1
    assert inspect.isgeneratorfunction(wire.reencode_formal_variant_suite_v3) is False
    monkeypatch.setattr(wire, "_frame_payload_with_padding_v3", original_frame)

    malformed_frame_calls = 0

    def corrupt_internal_frame(payload: bytes, padding: bytes) -> bytes:
        nonlocal malformed_frame_calls
        malformed_frame_calls += 1
        framed = original_frame(payload, padding)
        if malformed_frame_calls == len(success.public_batch.envelopes) + 1:
            return framed[:-1] + bytes((framed[-1] ^ 1,))
        return framed

    monkeypatch.setattr(
        wire, "_frame_payload_with_padding_v3", corrupt_internal_frame
    )
    malformed_result: wire.FormalVariantSuiteV3 | None = None
    with pytest.raises(ValueError, match="byte/digest replay drift"):
        malformed_result = wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(safe,)
        )
    assert malformed_result is None
    assert malformed_frame_calls == len(success.public_batch.envelopes) + 1
    monkeypatch.setattr(wire, "_frame_payload_with_padding_v3", original_frame)

    monkeypatch.setattr(wire, "MAXIMUM_VARIANT_SUITE_SCHEDULE_TOKENS_V3", 1)
    with pytest.raises(ValueError, match="token budget"):
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(safe,)
        )
    monkeypatch.setattr(
        wire,
        "MAXIMUM_VARIANT_SUITE_SCHEDULE_TOKENS_V3",
        8_388_608,
    )

    def exhausted(_: wire.FormalTrustedWireBuildSuccessV3) -> object:
        raise MemoryError

    monkeypatch.setattr(wire, "_prepare_variant_validation_v3", exhausted)
    with pytest.raises(ValueError) as failure:
        wire.reencode_formal_variant_suite_v3(
            build_success=success, schedules=(safe,)
        )
    assert str(failure.value) == "bounded formal variant suite reencode rejected"


def test_variant_surface_requires_full_build_and_rejects_self_minted_context() -> None:
    assert not hasattr(wire, "FormalVariantCapabilityV3")
    assert "FormalVariantCapabilityV3" not in wire.__all__
    for function_name in (
        "make_formal_variant_payloads_v3",
        "normalize_formal_variant_payload_v3",
        "reencode_formal_variant_payload_envelope_v3",
        "reencode_formal_variant_batch_v3",
    ):
        signature = inspect.signature(getattr(wire, function_name))
        assert "build_success" in signature.parameters
        assert "capability" not in signature.parameters

    success = build_success((source_case(0),))
    schedule = identity_variant_schedule(success)
    foreign_catalog, foreign_binding = catalog((source_case(1),))
    original_context = success.variant_context
    foreign_context_id = wire._variant_context_id_v3(
        batch_id=original_context.batch_id,
        private_alignment_id=original_context.private_alignment_id,
        transform_catalog_root=foreign_binding.transform_catalog_root,
        catalog_binding_receipt_id=foreign_binding.receipt_id,
        base_envelopes=original_context.base_envelopes,
    )
    self_minted = wire._FormalVariantContextV3(
        original_context.batch_id,
        original_context.private_alignment_id,
        foreign_binding.transform_catalog_root,
        foreign_binding.receipt_id,
        foreign_catalog,
        foreign_binding,
        original_context.base_envelopes,
        foreign_context_id,
        False,
    )
    forged_success = unsafe_issued_copy(
        success,
        variant_context=self_minted,
        success_id=wire._build_success_id_v3(
            public_batch=success.public_batch,
            key_source_receipt=success.key_source_receipt,
            private_alignment=success.private_audit_alignment,
            variant_context=self_minted,
        ),
    )
    with pytest.raises(ValueError, match="context|root|catalog"):
        wire.make_formal_variant_payloads_v3(
            build_success=forged_success, schedule=schedule
        )
    with pytest.raises(TypeError, match="full build success"):
        wire.make_formal_variant_payloads_v3(  # type: ignore[arg-type]
            build_success=self_minted, schedule=schedule
        )


def test_variant_immutable_base_and_snapshot_block_toctou(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = build_success((source_case(0),))
    schedule = identity_variant_schedule(success)
    base = success.variant_context.base_envelopes[0]
    assert type(base.canonical_payload_bytes) is bytes
    assert type(base.padding_bytes) is bytes
    assert not hasattr(base, "payload")

    changed_bytes = b" " + base.canonical_payload_bytes[1:]
    changed_base = unsafe_issued_copy(
        base, canonical_payload_bytes=changed_bytes
    )
    changed_bases = (changed_base,)
    changed_context = unsafe_issued_copy(
        success.variant_context,
        base_envelopes=changed_bases,
        context_id=wire._variant_context_id_v3(
            batch_id=success.variant_context.batch_id,
            private_alignment_id=success.variant_context.private_alignment_id,
            transform_catalog_root=success.variant_context.transform_catalog_root,
            catalog_binding_receipt_id=success.variant_context.catalog_binding_receipt_id,
            base_envelopes=changed_bases,
        ),
    )
    changed_success = unsafe_issued_copy(
        success,
        variant_context=changed_context,
        success_id=wire._build_success_id_v3(
            public_batch=success.public_batch,
            key_source_receipt=success.key_source_receipt,
            private_alignment=success.private_audit_alignment,
            variant_context=changed_context,
        ),
    )
    with pytest.raises((wire.RFC8785Error, ValueError)):
        wire.make_formal_variant_payloads_v3(
            build_success=changed_success, schedule=schedule
        )

    expected = wire.make_formal_variant_payloads_v3(
        build_success=success, schedule=schedule
    )
    original_snapshot = wire._snapshot_and_validate_build_success_for_variants_v3

    def snapshot_then_mutate(
        value: wire.FormalTrustedWireBuildSuccessV3,
    ) -> object:
        snapshot = original_snapshot(value)
        object.__setattr__(
            value.variant_context.base_envelopes[0],
            "canonical_payload_bytes",
            changed_bytes,
        )
        return snapshot

    monkeypatch.setattr(
        wire,
        "_snapshot_and_validate_build_success_for_variants_v3",
        snapshot_then_mutate,
    )
    assert wire.make_formal_variant_payloads_v3(
        build_success=success, schedule=schedule
    ) == expected
    monkeypatch.setattr(
        wire,
        "_snapshot_and_validate_build_success_for_variants_v3",
        original_snapshot,
    )

    payload_success = build_success((source_case(2),))
    payload_schedule = identity_variant_schedule(payload_success)
    payload_expected = wire.make_formal_variant_payloads_v3(
        build_success=payload_success, schedule=payload_schedule
    )[0]
    mutable_payload = deepcopy(payload_expected.payload)
    original_rows = wire._validated_variant_payload_rows_v3

    def mutate_caller_after_payload_snapshot(**kwargs: object) -> object:
        mutable_payload["public_case"]["graph"]["semantic_descriptor"][  # type: ignore[index]
            "segments"
        ][0] = "post-snapshot mutation"  # type: ignore[index]
        return original_rows(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        wire,
        "_validated_variant_payload_rows_v3",
        mutate_caller_after_payload_snapshot,
    )
    normalized = wire.normalize_formal_variant_payload_v3(
        mutable_payload,
        build_success=payload_success,
        schedule=payload_schedule,
        envelope_id=payload_expected.envelope_id,
    )
    assert normalized == payload_expected.payload
    assert mutable_payload != normalized


def test_channel_projection_and_claim_boundaries() -> None:
    batch = build((source_case(0),))
    projection = batch.envelopes[0].channel_projection
    assert projection.observation_array_paths == wire.PUBLIC_OBSERVATION_ARRAY_PATHS_V3
    assert projection.missingness_channels
    assert projection.missingness_bits == tuple(
        row.missing for row in projection.missingness_channels
    )
    assert all(row.json_pointer.startswith("/public_case/graph/") for row in projection.missingness_channels)
    assert projection.missingness_bits
    assert projection.record_order_tokens
    assert projection.component_slot_order_tokens
    assert projection.eligibility_mask_tokens
    assert {row.namespace for row in projection.namespace_uuid_channels} == set(wire.NAMESPACE_REGISTRY_V3)
    assert batch.formal_covert_audit is False
    assert batch.capacity_1024_qualified is False
    assert batch.actual_execution_claimed is False


def test_id_frame_exact_and_field_classes_cover_every_leaf_once() -> None:
    run_id = bytes(range(32))
    namespace = "observation_id"
    frame = wire.id_frame_v3(run_id, namespace, 7, 2)
    assert frame == (
        wire.ID_FRAME_DOMAIN_V3
        + run_id
        + len(namespace).to_bytes(2, "big")
        + namespace.encode("ascii")
        + (7).to_bytes(8, "big")
        + b"\x02"
    )
    decoded = wire.decode_and_audit_formal_envelope_v3(build((source_case(0),)).envelopes[0].envelope)
    pointers = [row.json_pointer for row in decoded.classified_leaves]
    assert len(pointers) == len(set(pointers))
    assert {row.classification for row in decoded.classified_leaves} == {
        wire.PublicLeafClassV3.CHANNEL_ONLY,
        wire.PublicLeafClassV3.EVIDENCE_BEARING,
    }


def test_exact_uuid_field_namespace_manifest_is_literal_and_annotation_complete() -> None:
    expected = {
        "bundle_id": "bundle_id",
        "observation_id": "observation_id",
        "observation_ids": "observation_id",
        "required_observation_ids": "observation_id",
        "entity_id": "entity_id",
        "entity_ids": "entity_id",
        "role_candidate_id": "role_candidate_id",
        "role_candidate_ids": "role_candidate_id",
        "required_role_candidate_ids": "role_candidate_id",
        "quantity_id": "quantity_id",
        "quantity_ids": "quantity_id",
        "required_quantity_ids": "quantity_id",
        "context_id": "context_id",
        "context_ids": "context_id",
        "task_id": "task_id",
        "task_ids": "task_id",
        "scale_id": "scale_id",
        "scale_ids": "scale_id",
        "source_scale_id": "scale_id",
        "target_scale_id": "scale_id",
        "aggregate_map_id": "aggregate_map_id",
        "aggregate_map_ids": "aggregate_map_id",
        "quotient_aggregate_map_id": "aggregate_map_id",
        "transform_id": "transform_id",
        "transform_ids": "transform_id",
    }
    assert wire._ID_FIELD_NAMESPACE_V3 == expected
    for field_routes in wire.FORMAL_PUBLIC_STRING_FIELD_POLICY_V3.values():
        for field_name, route in field_routes.items():
            if route in {"UUID", "UUID_TUPLE"}:
                assert field_name in expected
                assert expected[field_name] in wire.NAMESPACE_REGISTRY_V3


def test_literal_catalog_root_and_private_membership_binding() -> None:
    case = source_case(0)
    global_catalog, binding = catalog((case,))

    def independent_profile(value: object) -> object:
        if type(value) in (str, int, bool, type(None)):
            return value
        if type(value) is tuple:
            return [independent_profile(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return {
                item.name: independent_profile(getattr(value, item.name))
                for item in fields(value)
            }
        raise TypeError(type(value))

    use_body = {
        "aggregate_map": independent_profile(case.graph.aggregate_maps[0]),
        "transform": independent_profile(case.graph.transforms[0]),
    }
    literal_use_hash = "sha256:" + hashlib.sha256(
        b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/SOURCE_TRANSFORM_USE/V3\x00"
        + wire.canonicalize_rfc8785_v1(use_body)
    ).hexdigest()
    assert literal_use_hash == (
        "sha256:3a8ab141a34056193df37b1dfeb067751d32268578459955e5ef97be42f5be55"
    )
    source_use = wire.FormalSourceTransformUseV3(
        case.graph.aggregate_maps[0], case.graph.transforms[0]
    )
    assert wire.formal_source_transform_use_hash_v3(source_use) == literal_use_hash
    renamed_transform_id = _uuid(999_999_777)
    renamed_source_use = wire.FormalSourceTransformUseV3(
        replace(case.graph.aggregate_maps[0], transform_id=renamed_transform_id),
        replace(case.graph.transforms[0], transform_id=renamed_transform_id),
    )
    assert wire.formal_source_transform_use_hash_v3(renamed_source_use) != literal_use_hash
    assert global_catalog.entries == (
        wire.FormalGlobalTransformCatalogEntryV3("identity", literal_use_hash),
    )
    literal_entry = {
        "operation": "identity",
        "exact_source_transform_use_sha256": literal_use_hash,
    }
    literal_catalog = {
        "entries": [literal_entry],
        "schema_version": "hegel-machine-phase2b-formal-global-transform-catalog/3",
    }
    literal_root = "sha256:" + hashlib.sha256(
        b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/TRANSFORM_CATALOG_ROOT/V3\x00"
        + wire.canonicalize_rfc8785_v1(literal_catalog)
    ).hexdigest()
    assert literal_root == (
        "sha256:ea358d4825a4df386c0f0eb819a0e2d208cc95360f4aa43543d86ff79ae847d9"
    )
    binding_domain = (
        b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/TRANSFORM_CATALOG_BINDING/V3\x00"
    )
    literal_entry_commitment = "sha256:" + hashlib.sha256(
        binding_domain + wire.canonicalize_rfc8785_v1(literal_entry)
    ).hexdigest()
    assert literal_entry_commitment == (
        "sha256:305394da828b82abb2de134d34d83cad1cd936b235e7cc12067d44779a5038b5"
    )
    literal_receipt_body = {
        "catalog_entry_commitments": [literal_entry_commitment],
        "schema_version": "hegel-machine-phase2b-formal-global-transform-catalog-binding/3",
        "transform_catalog_root": literal_root,
    }
    literal_receipt_id = (
        "phase2b_formal_global_transform_catalog_binding_v3_"
        + hashlib.sha256(
            binding_domain + wire.canonicalize_rfc8785_v1(literal_receipt_body)
        ).hexdigest()
    )
    assert literal_receipt_id == (
        "phase2b_formal_global_transform_catalog_binding_v3_"
        "b72d75d8f8e6bda7bf0a053562cd8fb9e48d5ecc5b1d8aff37542d5fbc92f7aa"
    )
    assert wire.formal_global_transform_catalog_root_v3(global_catalog) == literal_root
    assert binding.transform_catalog_root == literal_root
    assert binding.catalog_entry_commitments == (literal_entry_commitment,)
    assert binding.receipt_id == literal_receipt_id
    batch = build((case,))
    payload = wire.decode_and_audit_formal_envelope_v3(batch.envelopes[0].envelope).payload
    assert set(payload["batch_context"]) == {
        "case_count_decimal",
        "id_collision_retry_count_decimal",
        "run_id_commitment",
        "transform_catalog_root",
    }
    assert payload["batch_context"]["transform_catalog_root"] == binding.transform_catalog_root
    assert "global_transform_catalog_root" not in repr(payload)
    assert "transform_catalog" not in payload["public_case"]["graph"]
    assert batch.envelopes[0].structural_replay_verified is True
    assert wire.decode_and_audit_formal_envelope_v3(
        batch.envelopes[0].envelope
    ).catalog_membership_verified is False

    # The source catalog is deliberately UUID/content scoped rather than
    # alpha-equivalence scoped.  A separately named but otherwise admissible
    # source transform is therefore not a member of this exact catalog.
    changed_case = source_case(1)
    rejection = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(changed_case,),
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(rejection) is wire.FormalTrustedWireBatchRejectionV3
    changed_map = replace(
        case.graph.aggregate_maps[0],
        groups=(
            replace(
                case.graph.aggregate_maps[0].groups[0],
                quotient_label="q-catalog-mutation",
            ),
        ),
    )
    changed_map_case = wire.FormalSourceCaseV3(
        replace(case.graph, aggregate_maps=(changed_map,))
    )
    rejection = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(changed_map_case,),
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(rejection) is wire.FormalTrustedWireBatchRejectionV3
    forged_binding = replace(binding, transform_catalog_root=ZERO_DIGEST)
    with pytest.raises(ValueError):
        wire.verify_formal_global_transform_catalog_binding_v3(
            catalog=global_catalog,
            binding_receipt=forged_binding,
        )


def test_same_source_uuid_across_cases_and_public_uuid_across_envelopes_reject() -> None:
    first, second = source_case(0), source_case(1)
    cross_namespace_source = wire.FormalSourceCaseV3(
        replace(first.graph, bundle_id=first.graph.entities[0].entity_id)
    )
    cross_catalog, cross_binding = catalog((cross_namespace_source,))
    cross_rejection = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(cross_namespace_source,),
        global_transform_catalog=cross_catalog,
        transform_catalog_binding_receipt=cross_binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(cross_rejection) is wire.FormalTrustedWireBatchRejectionV3
    assert cross_rejection.envelopes == ()

    reused = wire.FormalSourceCaseV3(
        replace(second.graph, bundle_id=first.graph.bundle_id)
    )
    global_catalog, binding = catalog((first, reused))
    result = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(first, reused),
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(result) is wire.FormalTrustedWireBatchRejectionV3
    assert result.envelopes == ()

    batch = build((first, second))
    decoded = tuple(
        wire.decode_and_audit_formal_envelope_v3(row.envelope)
        for row in batch.envelopes
    )
    mutated = deepcopy(decoded[1].payload)
    mutated["public_case"]["graph"]["bundle_id"] = decoded[0].payload["public_case"]["graph"]["bundle_id"]
    normalized = wire._normalize_formal_payload_structure_v3(mutated)
    spliced = wire._frame_payload_with_padding_v3(
        wire.canonicalize_rfc8785_v1(normalized), decoded[1].padding_bytes
    )
    # The mutated row is internally valid and freshly provenance-bound.  Only
    # the batch replay's global UUID ownership check may reject this splice.
    spliced_decoded = wire.decode_and_audit_formal_envelope_v3(spliced)
    assert (
        spliced_decoded.public_case.graph.bundle_id
        == decoded[0].public_case.graph.bundle_id
    )
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (decoded[0].envelope, spliced),
        key_source_receipt_id=batch.key_source_receipt_id,
    )
    assert type(replay) is wire.FormalTrustedWireBatchRejectionV3
    assert replay.envelopes == ()

    trusted_spliced = wire._trusted_envelope_from_primitive_bytes_v3(spliced)
    spliced_rows = (batch.envelopes[0], trusted_spliced)
    spliced_ids = tuple(row.envelope_id for row in spliced_rows)
    spliced_batch_id = wire._batch_id_v3(
        batch.batch_context,
        batch.key_source_receipt_id,
        spliced_ids,
    )
    with pytest.raises(ValueError, match="reused across envelopes"):
        replace(
            batch,
            envelopes=spliced_rows,
            envelope_ids=spliced_ids,
            batch_id=spliced_batch_id,
        )
    baseline_replay = wire.replay_formal_trusted_wire_batch_v3(
        tuple(row.envelope for row in batch.envelopes),
        key_source_receipt_id=batch.key_source_receipt_id,
    )
    assert type(baseline_replay) is wire.FormalTrustedWireBatchReplayV3
    with pytest.raises(ValueError, match="reused across envelopes"):
        replace(
            baseline_replay,
            decoded_envelopes=(decoded[0], spliced_decoded),
            envelope_ids=spliced_ids,
            batch_id=spliced_batch_id,
        )

    single = build((source_case(2),))
    single_decoded = wire.decode_and_audit_formal_envelope_v3(
        single.envelopes[0].envelope
    )
    retry_payload = deepcopy(single_decoded.payload)
    unique_uuid_count = len(
        {
            channel.public_uuid
            for channel in single_decoded.channel_projection.namespace_uuid_channels
        }
    )
    retry_payload["batch_context"][  # type: ignore[index]
        "id_collision_retry_count_decimal"
    ] = str(unique_uuid_count * wire.MAXIMUM_UUID_COLLISION_RETRIES_V3 + 1)
    retry_normalized = wire._normalize_formal_payload_structure_v3(retry_payload)
    retry_payload_bytes = wire.canonicalize_rfc8785_v1(retry_normalized)
    retry_padding = b"r" * (
        wire.ENVELOPE_BYTES_V3
        - wire.ENVELOPE_HEADER_BYTES_V3
        - len(retry_payload_bytes)
    )
    retry_envelope = wire._frame_payload_with_padding_v3(
        retry_payload_bytes, retry_padding
    )
    retry_decoded = wire.decode_and_audit_formal_envelope_v3(retry_envelope)
    retry_trusted = wire._trusted_envelope_from_primitive_bytes_v3(retry_envelope)
    retry_ids = (retry_decoded.envelope_id,)
    retry_batch_id = wire._batch_id_v3(
        retry_decoded.batch_context,
        single.key_source_receipt_id,
        retry_ids,
    )
    with pytest.raises(ValueError, match="retry count"):
        replace(
            single,
            batch_context=retry_decoded.batch_context,
            envelopes=(retry_trusted,),
            envelope_ids=retry_ids,
            batch_id=retry_batch_id,
        )
    single_replay = wire.replay_formal_trusted_wire_batch_v3(
        (single.envelopes[0].envelope,),
        key_source_receipt_id=single.key_source_receipt_id,
    )
    assert type(single_replay) is wire.FormalTrustedWireBatchReplayV3
    with pytest.raises(ValueError, match="retry count"):
        replace(
            single_replay,
            batch_context=retry_decoded.batch_context,
            decoded_envelopes=(retry_decoded,),
            envelope_ids=retry_ids,
            batch_id=retry_batch_id,
        )
    retry_replay = wire.replay_formal_trusted_wire_batch_v3(
        (retry_envelope,),
        key_source_receipt_id=single.key_source_receipt_id,
    )
    assert type(retry_replay) is wire.FormalTrustedWireBatchRejectionV3
    assert retry_replay.envelopes == ()


def test_ascii_decimal_and_rfc_closed_payload_rejections() -> None:
    success = build_success((source_case(0),))
    batch = success.public_batch
    decoded = wire.decode_and_audit_formal_envelope_v3(batch.envelopes[0].envelope)
    with pytest.raises(ValueError):
        wire.FormalBatchContextV3(
            "０",
            decoded.batch_context.transform_catalog_root,
            "0",
            decoded.batch_context.run_id_commitment,
        )
    with pytest.raises(ValueError):
        wire.FormalBatchContextV3(
            "01",
            decoded.batch_context.transform_catalog_root,
            "0",
            decoded.batch_context.run_id_commitment,
        )

    for mutation in ("unknown", "missing", "bool", "float"):
        payload = deepcopy(decoded.payload)
        if mutation == "unknown":
            payload["public_case"]["graph"]["hidden"] = "covert"
        elif mutation == "missing":
            del payload["public_case"]["graph"]["eligibility_mask"]
        else:
            payload["public_case"]["graph"]["aggregate_maps"][0]["groups"][0]["input_refs"][0]["component_index"] = (
                True if mutation == "bool" else 0.5
            )
        with pytest.raises((TypeError, ValueError)):
            wire._normalize_formal_payload_structure_v3(payload)

    schedule = identity_variant_schedule(success)
    semantic_mutation = deepcopy(decoded.payload)
    semantic_mutation["public_case"]["graph"]["transforms"][0]["certificate"][
        "inverse_contract"
    ] = "forged-weaker-contract"
    with pytest.raises(ValueError, match="inverse-contract|semantic/root change"):
        wire.normalize_formal_variant_payload_v3(
            semantic_mutation,
            build_success=success,
            schedule=schedule,
            envelope_id=decoded.envelope_id,
        )

    version = wire.canonicalize_rfc8785_v1(
        wire.FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION
    )
    duplicate = b'{"schema_version":' + version + b"," + decoded.payload_bytes[1:]
    noncanonical = b" " + decoded.payload_bytes
    for payload_bytes in (duplicate, noncanonical):
        padding = b"x" * (
            wire.ENVELOPE_BYTES_V3
            - wire.ENVELOPE_HEADER_BYTES_V3
            - len(payload_bytes)
        )
        envelope = wire._frame_payload_with_padding_v3(payload_bytes, padding)
        with pytest.raises((wire.RFC8785Error, ValueError)):
            wire.decode_and_audit_formal_envelope_v3(envelope)


def test_standalone_decode_checks_root_structure_not_private_catalog_membership() -> None:
    trusted = build_success((source_case(0),))
    foreign = build_success((source_case(1),))
    foreign_decoded = wire.decode_and_audit_formal_envelope_v3(
        foreign.public_batch.envelopes[0].envelope
    )
    payload = deepcopy(foreign_decoded.payload)
    payload["batch_context"]["transform_catalog_root"] = (
        trusted.public_batch.batch_context.transform_catalog_root
    )
    normalized = wire._normalize_formal_payload_structure_v3(payload)
    envelope = wire._frame_payload_with_padding_v3(
        wire.canonicalize_rfc8785_v1(normalized),
        foreign_decoded.padding_bytes,
    )
    decoded = wire.decode_and_audit_formal_envelope_v3(envelope)
    assert (
        decoded.batch_context.transform_catalog_root
        == trusted.public_batch.batch_context.transform_catalog_root
    )
    assert decoded.catalog_membership_verified is False
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (envelope,), key_source_receipt_id=trusted.public_batch.key_source_receipt_id
    )
    assert type(replay) is wire.FormalTrustedWireBatchReplayV3
    assert replay.catalog_membership_verified is False

    trusted_schedule = identity_variant_schedule(trusted)
    with pytest.raises(ValueError):
        wire.normalize_formal_variant_payload_v3(
            deepcopy(normalized),
            build_success=trusted,
            schedule=trusted_schedule,
            envelope_id=trusted.public_batch.envelope_ids[0],
        )


def test_nested_dto_manifest_and_independent_exact_one_leaf_oracle() -> None:
    expected = {
        "RationalValueV3": ("numerator", "denominator"),
        "NullableTextV3": ("available", "value"),
        "NullableBooleanV3": ("available", "value"),
        "NullableRationalVectorV3": ("available", "value"),
        "ExactTemporalSupportV3": ("start", "end"),
        "NullableTemporalSupportV3": ("available", "value"),
        "ExactSpatialSupportV3": ("lower", "upper"),
        "NullableSpatialSupportV3": ("available", "value"),
        "FormalUnitBindingV3": ("semantic_key", "scale_to_si", "offset_to_si"),
        "FormalEntityRecordV3": ("entity_id", "role_candidate_ids"),
        "FormalQuantityRecordV3": (
            "quantity_id", "semantic_key", "si_exponents", "unit_binding", "value_kind",
        ),
        "FormalContextRecordV3": (
            "context_id", "temporal_reference_kind", "clock_binding",
            "spatial_reference_kind", "frame_binding", "coordinate_labels",
            "temporal_support", "spatial_support", "orientation",
        ),
        "FormalObservationRecordV3": (
            "observation_id", "entity_ids", "role_candidate_ids", "quantity_id",
            "context_id", "scale_id", "components", "numeric_value",
            "boolean_value", "uncertainty_model", "uncertainty_radius",
            "public_provenance_commitment",
        ),
        "FormalTaskTargetV3": ("task_id", "entity_ids", "quantity_ids"),
        "FormalSemanticDescriptorV3": ("kind", "segments"),
        "FormalScaleRecordV3": ("scale_id", "semantic_level", "is_root"),
        "FormalComponentDescriptorV3": ("label", "semantic_axis", "value_role"),
        "FormalComponentRefV3": (
            "observation_id", "quantity_id", "context_id", "scale_id",
            "component_axis", "component_index",
        ),
        "FormalExactTermV3": ("input_ref", "coefficient"),
        "FormalExactRowV3": ("output_ref", "terms", "offset"),
        "FormalSamplingGridPointV3": ("input_ref", "coordinates"),
        "FormalDiscreteMappingV3": ("input_ref", "output_ref"),
        "FormalAggregateGroupV3": ("input_refs", "output_refs", "quotient_label"),
        "FormalAggregateMapRecordV3": (
            "aggregate_map_id", "source_scale_id", "target_scale_id", "transform_id",
            "operation", "reducer", "groups",
        ),
        "FormalIdentityCertificateV3": (
            "semantics_version", "missing_policy", "inverse_contract",
        ),
        "FormalUnitConversionCertificateV3": (
            "semantics_version", "factor", "inverse_factor", "missing_policy",
            "orientation", "commutation_contract",
        ),
        "FormalCoordinateAffineCertificateV3": (
            "semantics_version", "source_frame_binding", "target_frame_binding",
            "dimension", "inverse_rows", "missing_policy", "support_contract",
            "inverse_contract",
        ),
        "FormalTemporalAggregationCertificateV3": (
            "semantics_version", "aggregate_map_id", "reducer", "missing_policy",
            "boundary_policy", "support_contract", "weighted_mean_basis",
        ),
        "FormalSpatialAggregationCertificateV3": (
            "semantics_version", "aggregate_map_id", "reducer", "missing_policy",
            "boundary_policy", "support_contract", "weighted_mean_basis",
        ),
        "FormalSamplingResolutionCertificateV3": (
            "semantics_version", "axis", "selected_inputs", "discarded_inputs",
            "grid_points", "grid_dimension", "grid_frame_binding", "missing_policy",
            "boundary_policy", "kernel_contract",
        ),
        "FormalEquivalentSplitMergeCertificateV3": (
            "semantics_version", "aggregate_map_id", "direction", "inverse_rows",
            "missing_policy", "equivalence_contract",
        ),
        "FormalCoarseGrainingCertificateV3": (
            "semantics_version", "aggregate_map_id", "reducer",
            "source_commutation_rows", "target_commutation_rows",
            "quotient_aggregate_map_id", "quotient_labels", "missing_policy",
            "boundary_policy", "commutation_contract", "weighted_mean_basis",
        ),
        "FormalTransformContractV3": (
            "transform_id", "operation", "source_scale_id", "target_scale_id",
            "input_refs", "output_refs", "kernel_rows", "discrete_mappings", "certificate",
        ),
        "FormalFamilyAliasV3": ("alias", "canonical_family"),
        "FormalLawBindingV3": (
            "law_name", "family", "law_kind", "required_observation_ids",
            "required_role_candidate_ids", "required_quantity_ids",
        ),
        "FormalObservableChannelV3": ("observation_id", "quantity_id"),
        "FormalRecognizerRegistryV3": (
            "theory_policy", "family_aliases", "law_bindings", "observable_channels",
            "max_candidate_count",
        ),
        "FormalEligibilityMaskV3": ("observation_ids", "role_candidate_ids", "scale_ids"),
        "FormalCaseGraphV3": (
            "schema_version", "bundle_id", "entities", "role_candidate_ids", "quantities",
            "contexts", "observations", "task_target", "semantic_descriptor", "scales", "aggregate_maps",
            "transforms", "recognizer_registry", "eligibility_mask",
        ),
        "FormalPublicCaseV3": ("schema_version", "graph", "public_provenance_commitment"),
        "FormalBatchContextV3": (
            "case_count_decimal", "transform_catalog_root",
            "id_collision_retry_count_decimal", "run_id_commitment",
        ),
        "FormalPayloadV3": (
            "batch_context", "field_classification_manifest_id",
            "graph_completeness_policy_id", "public_case", "schema_version",
        ),
    }
    assert wire.FORMAL_PUBLIC_DTO_FIELD_MANIFEST_V3 == expected
    for type_name, exact_fields in expected.items():
        if type_name == "FormalPayloadV3":
            continue
        assert tuple(item.name for item in fields(getattr(wire, type_name))) == exact_fields
    decoded = wire.decode_and_audit_formal_envelope_v3(
        build((source_case(0),)).envelopes[0].envelope
    )
    def independent_leaves(value: object, pointer: str = "") -> tuple[tuple[str, object], ...]:
        rows: list[tuple[str, object]] = []
        if isinstance(value, dict):
            for key in sorted(value):
                rows.extend(independent_leaves(value[key], pointer + "/" + key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                rows.extend(independent_leaves(item, pointer + "/" + str(index)))
        else:
            rows.append((pointer, value))
        return tuple(rows)

    independent = independent_leaves(decoded.payload)
    classified = wire.classify_public_payload_leaves_v3(decoded.payload)
    assert len(independent) == len(classified)
    assert {pointer for pointer, _ in independent} == {
        row.json_pointer for row in classified
    }
    literal_channel_last_segments = {
        "available", "case_count_decimal", "component_index",
        "field_classification_manifest_id", "graph_completeness_policy_id",
        "id_collision_retry_count_decimal", "max_candidate_count",
        "public_provenance_commitment", "run_id_commitment", "schema_version",
    }
    literal_uuid4 = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    expected_classes: dict[str, str] = {}
    for pointer, value in independent:
        last = pointer.rsplit("/", 1)[-1]
        is_channel = (
            not pointer.startswith("/public_case/graph/")
            or "/eligibility_mask/" in pointer
            or last in literal_channel_last_segments
            or value is None
            or (isinstance(value, str) and literal_uuid4.fullmatch(value) is not None)
        )
        expected_classes[pointer] = (
            "CHANNEL_ONLY" if is_channel else "EVIDENCE_BEARING"
        )
    actual_classes = {
        row.json_pointer: row.classification.value for row in classified
    }
    assert actual_classes == expected_classes
    descriptor_classes = {
        value
        for pointer, value in actual_classes.items()
        if "/semantic_descriptor/" in pointer
    }
    assert descriptor_classes == {"EVIDENCE_BEARING"}


def test_public_string_route_manifest_is_type_complete_and_fail_closed() -> None:
    """The oracle derives direct string fields from annotations, not values."""
    for data_type in (
        *wire._FORMAL_PUBLIC_GRAPH_DTO_TYPES_V3,
        wire.FormalBatchContextV3,
    ):
        direct_string_fields: set[str] = set()
        for name, annotation in get_type_hints(data_type).items():
            origin = get_origin(annotation)
            arguments = get_args(annotation)
            is_optional_string = (
                origin is types.UnionType
                and set(arguments) == {str, type(None)}
            )
            if (
                annotation is str
                or is_optional_string
                or origin is tuple and arguments == (str, Ellipsis)
            ):
                direct_string_fields.add(name)
        assert set(
            wire.FORMAL_PUBLIC_STRING_FIELD_POLICY_V3.get(
                data_type.__name__, {}
            )
        ) == direct_string_fields

    unknown_type = make_dataclass(
        "FormalUnroutedStringFixtureV3", (("hidden", str),), frozen=True
    )
    with pytest.raises(ValueError, match="not routed"):
        wire._validate_routed_public_strings_v3(unknown_type("not-routed"))
    with pytest.raises(TypeError):
        wire.FORMAL_PUBLIC_STRING_FIELD_POLICY_V3[  # type: ignore[index]
            "FormalScaleRecordV3"
        ] = {}


def test_every_routed_non_id_string_rejects_embedded_source_ids_atomically() -> None:
    factories = (
        "identity_profile",
        "unit_profile",
        "coordinate_profile",
        "temporal_profile",
        "spatial_profile",
        "sampling_profile",
        "split_profile",
        "coarse_profile",
    )
    exercised_routes: set[str] = set()
    key_receipt = receipt()
    for factory_index, factory_name in enumerate(factories):
        source = _wire_case_from_exact_core_profile(
            _EXACT_CORE_FIXTURES[factory_name](), index=300 + factory_index
        )
        graph = source.graph
        global_catalog, binding = catalog((source,))
        paths = routed_non_id_string_paths(graph)
        assert paths
        for path in paths:
            current: object = graph
            route: str | None = None
            for part in path:
                if isinstance(part, str):
                    route = wire.FORMAL_PUBLIC_STRING_FIELD_POLICY_V3.get(
                        type(current).__name__, {}
                    ).get(part, route)
                    current = getattr(current, part)
                else:
                    current = current[part]  # type: ignore[index]
            assert route is not None
            exercised_routes.add(route)
            for hidden in (
                "prefix:" + graph.bundle_id + ":suffix",
                "prefix:" + binding.transform_catalog_root + ":suffix",
            ):
                attacked_graph = unsafe_path_replace(graph, path, hidden)
                attacked_source = unsafe_issued_copy(source, graph=attacked_graph)
                result = wire.build_formal_trusted_wire_batch_v3(
                    source_cases=(attacked_source,),
                    global_transform_catalog=global_catalog,
                    transform_catalog_binding_receipt=binding,
                    run_id=b"R" * 32,
                    key_sources=keys(),
                    key_source_receipt=key_receipt,
                )
                assert type(result) is wire.FormalTrustedWireBatchRejectionV3
                assert result.envelopes == () and result.batch_id is None

    assert exercised_routes == {
        "BINDING_TOKEN",
        "CANONICAL_DECIMAL",
        "ENUM_LITERAL",
        "EXACT_LITERAL",
        "SEMANTIC_DESCRIPTOR_TEXT_TUPLE",
        "SEMANTIC_TEXT",
        "SEMANTIC_TEXT_OR_NULL",
        "SEMANTIC_TEXT_TUPLE",
    }
    for hidden in (
        "x" + "a" * 64 + "x",
        "bidi\u202etext",
        "variation\ufe0ftext",
    ):
        with pytest.raises(ValueError):
            wire._public_semantic_text_v3(hidden, "test semantic text")

    # Exact typed UUID/digest/decimal routes remain legal in their own fields.
    wire._validate_routed_public_strings_v3(source_case(0).graph)
    wire.FormalBatchContextV3("1", ZERO_DIGEST, "0", ZERO_DIGEST)


def test_reference_cycle_operation_clause_unused_transform_bool_and_float_reject() -> None:
    case = source_case(0)
    graph = case.graph
    with pytest.raises(ValueError):
        replace(
            graph,
            task_target=replace(graph.task_target, entity_ids=(_uuid(999_999_991),)),
        )
    with pytest.raises(ValueError):
        replace(
            graph,
            quantities=(
                replace(graph.quantities[0], si_exponents=(True, 0, 0, 0, 0, 0, 0)),
            ),
        )
    with pytest.raises(ValueError):
        replace(
            graph,
            transforms=(
                replace(
                    graph.transforms[0],
                    certificate=wire.FormalUnitConversionCertificateV3(
                        wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
                        wire.RationalValueV3("1", "1"),
                        wire.RationalValueV3("1", "1"),
                        "explicit_preserve",
                        "target_value_equals_source_value_times_factor",
                        "si_dimension_and_support_preserved",
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError):
        replace(
            graph,
            transforms=graph.transforms
            + (replace(graph.transforms[0], transform_id=_uuid(999_999_992)),),
        )
    bad_ref = replace(
        graph.aggregate_maps[0].groups[0].input_refs[0], component_index=0.0
    )
    with pytest.raises(ValueError):
        replace(
            graph,
            aggregate_maps=(
                replace(
                    graph.aggregate_maps[0],
                    groups=(
                        replace(
                            graph.aggregate_maps[0].groups[0],
                            input_refs=(bad_ref,),
                        ),
                    ),
                ),
            ),
        )

    forward_map, forward_transform = graph.aggregate_maps[0], graph.transforms[0]
    reverse_map_id, reverse_transform_id = _uuid(999_999_993), _uuid(999_999_994)
    source_ref = forward_transform.input_refs[0]
    target_ref = forward_transform.output_refs[0]
    reverse_map = wire.FormalAggregateMapRecordV3(
        reverse_map_id,
        forward_map.target_scale_id,
        forward_map.source_scale_id,
        reverse_transform_id,
        "identity",
        "not_applicable",
        (wire.FormalAggregateGroupV3((target_ref,), (source_ref,), "q1"),),
    )
    reverse_transform = wire.FormalTransformContractV3(
        reverse_transform_id,
        "identity",
        forward_map.target_scale_id,
        forward_map.source_scale_id,
        (target_ref,),
        (source_ref,),
        (
            wire.FormalExactRowV3(
                source_ref,
                (wire.FormalExactTermV3(target_ref, wire.RationalValueV3("1", "1")),),
                wire.RationalValueV3("0", "1"),
            ),
        ),
        (),
        wire.FormalIdentityCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            "explicit_preserve",
            "exact_one_to_one_identity",
        ),
    )
    with pytest.raises(ValueError, match="cyclic"):
        replace(
            graph,
            aggregate_maps=graph.aggregate_maps + (reverse_map,),
            transforms=graph.transforms + (reverse_transform,),
        )


def test_aggregate_group_kernel_and_reducer_exact_cross_closure() -> None:
    graph = source_case(0).graph
    rational = wire.RationalValueV3
    source_observations = tuple(
        item for item in graph.observations if item.scale_id == graph.scales[0].scale_id
    )
    target_observations = tuple(
        item for item in graph.observations if item.scale_id == graph.scales[1].scale_id
    )

    def component(observation: wire.FormalObservationRecordV3) -> wire.FormalComponentRefV3:
        return wire.FormalComponentRefV3(
            observation.observation_id,
            observation.quantity_id,
            observation.context_id,
            observation.scale_id,
            observation.components[0].label,
            0,
        )

    source_refs = tuple(component(item) for item in source_observations)
    target_refs = tuple(component(item) for item in target_observations)
    groups = tuple(
        wire.FormalAggregateGroupV3((source,), (target,), f"partition-{index}")
        for index, (source, target) in enumerate(
            zip(source_refs, target_refs, strict=True)
        )
    )
    identity_map = replace(graph.aggregate_maps[0], groups=groups)
    identity_rows = tuple(
        wire.FormalExactRowV3(
            target,
            (wire.FormalExactTermV3(source, rational("1", "1")),),
            rational("0", "1"),
        )
        for source, target in zip(source_refs, target_refs, strict=True)
    )
    identity_contract = replace(
        graph.transforms[0],
        input_refs=source_refs,
        output_refs=target_refs,
        kernel_rows=identity_rows,
    )
    valid_identity = replace(
        graph,
        aggregate_maps=(identity_map,),
        transforms=(identity_contract,),
    )
    assert wire.FormalSourceCaseV3(valid_identity).graph is valid_identity
    crossed_rows = (
        replace(
            identity_rows[0],
            terms=(wire.FormalExactTermV3(source_refs[1], rational("1", "1")),),
        ),
        replace(
            identity_rows[1],
            terms=(wire.FormalExactTermV3(source_refs[0], rational("1", "1")),),
        ),
    )
    with pytest.raises(ValueError, match="aggregate-group boundary"):
        replace(
            valid_identity,
            transforms=(replace(identity_contract, kernel_rows=crossed_rows),),
        )
    with pytest.raises(ValueError, match="exactly partition|canonical"):
        replace(
                valid_identity,
                aggregate_maps=(
                    replace(
                        identity_map,
                        groups=(
                            groups[0],
                            replace(groups[0], quotient_label="duplicate-ref-other-label"),
                        ),
                    ),
                ),
            )
    with pytest.raises(ValueError, match="quotient label"):
        replace(
            valid_identity,
            aggregate_maps=(
                replace(
                    identity_map,
                    groups=(groups[0], replace(groups[1], quotient_label=groups[0].quotient_label)),
                ),
            ),
        )


def test_coarse_group_order_cannot_drift_from_quotient_label_binding() -> None:
    graph = _wire_case_from_exact_core_profile(
        _EXACT_CORE_FIXTURES["coarse_profile"](), index=520
    ).graph
    contract = graph.transforms[0]
    certificate = contract.certificate
    assert type(certificate) is wire.FormalCoarseGrainingCertificateV3
    source_by_id = {item.observation_id: item for item in graph.observations}
    source_rows = tuple(source_by_id[ref.observation_id] for ref in contract.input_refs)
    target = source_by_id[contract.output_refs[0].observation_id]
    target_two_id = _uuid(9_990_520_001)
    target_one = replace(
        target,
        numeric_value=source_rows[0].numeric_value,
        uncertainty_radius=source_rows[0].uncertainty_radius,
    )
    target_two = replace(
        target,
        observation_id=target_two_id,
        numeric_value=source_rows[1].numeric_value,
        uncertainty_radius=source_rows[1].uncertainty_radius,
    )
    output_refs = (
        contract.output_refs[0],
        replace(contract.output_refs[0], observation_id=target_two_id),
    )
    one = wire.RationalValueV3("1", "1")
    zero = wire.RationalValueV3("0", "1")
    kernel_rows = tuple(
        wire.FormalExactRowV3(
            output_ref,
            (wire.FormalExactTermV3(input_ref, one),),
            zero,
        )
        for input_ref, output_ref in zip(
            contract.input_refs, output_refs, strict=True
        )
    )
    target_commutation = tuple(
        wire.FormalExactRowV3(
            output_ref,
            (wire.FormalExactTermV3(output_ref, one),),
            zero,
        )
        for output_ref in output_refs
    )
    groups = tuple(
        wire.FormalAggregateGroupV3(
            (input_ref,), (output_ref,), f"quotient-{index}"
        )
        for index, (input_ref, output_ref) in enumerate(
            zip(contract.input_refs, output_refs, strict=True)
        )
    )
    aggregate = replace(graph.aggregate_maps[0], groups=groups)
    changed_certificate = replace(
        certificate,
        target_commutation_rows=target_commutation,
        quotient_labels=tuple(group.quotient_label for group in groups),
    )
    changed_contract = replace(
        contract,
        output_refs=output_refs,
        kernel_rows=kernel_rows,
        certificate=changed_certificate,
    )
    registry = graph.recognizer_registry
    law = registry.law_bindings[0]
    changed_registry = replace(
        registry,
        law_bindings=(
            replace(
                law,
                required_observation_ids=tuple(
                    sorted((*law.required_observation_ids, target_two_id))
                ),
            ),
        ),
        observable_channels=tuple(
            sorted(
                (
                    *registry.observable_channels,
                    wire.FormalObservableChannelV3(
                        target_two_id, target.quantity_id
                    ),
                ),
                key=lambda row: row.observation_id,
            )
        ),
    )
    valid = replace(
        graph,
        observations=tuple(
            sorted(
                (
                    *(item for item in graph.observations if item != target),
                    target_one,
                    target_two,
                ),
                key=lambda row: row.observation_id,
            )
        ),
        aggregate_maps=(aggregate,),
        transforms=(changed_contract,),
        recognizer_registry=changed_registry,
        eligibility_mask=replace(
            graph.eligibility_mask,
            observation_ids=tuple(
                sorted((*graph.eligibility_mask.observation_ids, target_two_id))
            ),
        ),
    )
    assert wire.FormalSourceCaseV3(valid).graph is valid
    presented = wire._profile_value_v3(valid)
    presented["aggregate_maps"][0]["groups"].reverse()
    presented["transforms"][0]["certificate"]["quotient_labels"].reverse()
    canonical = wire._canonicalize_graph_mapping_v3(
        presented, preserve_observation_order=False
    )
    assert canonical == wire._profile_value_v3(valid)
    assert tuple(
        canonical["transforms"][0]["certificate"]["quotient_labels"]
    ) == tuple(
        group["quotient_label"]
        for group in canonical["aggregate_maps"][0]["groups"]
    )

    full_wire = build_success((wire.FormalSourceCaseV3(valid),))
    decoded = wire.decode_and_audit_formal_envelope_v3(
        full_wire.public_batch.envelopes[0].envelope
    )
    decoded_certificate = decoded.public_case.graph.transforms[0].certificate
    assert type(decoded_certificate) is wire.FormalCoarseGrainingCertificateV3
    assert decoded_certificate.quotient_labels == tuple(
        group.quotient_label
        for group in decoded.public_case.graph.aggregate_maps[0].groups
    )
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (decoded.envelope,),
        key_source_receipt_id=full_wire.public_batch.key_source_receipt_id,
    )
    assert type(replay) is wire.FormalTrustedWireBatchReplayV3

    with pytest.raises(ValueError, match="canonical before quotient-label"):
        replace(
            valid,
            aggregate_maps=(replace(aggregate, groups=tuple(reversed(groups))),),
        )

def test_context_uncertainty_retry_and_semantic_descriptor_bounds() -> None:
    graph = source_case(0).graph
    context = graph.contexts[0]
    with pytest.raises(ValueError, match="reference/support"):
        replace(
            graph,
            contexts=(replace(context, temporal_reference_kind="none"),),
        )
    with pytest.raises(ValueError, match="dimension"):
        replace(
            graph,
            contexts=(replace(context, coordinate_labels=("x", "y")),),
        )
    observation = graph.observations[0]
    with pytest.raises(ValueError, match="nonnegative"):
        replace(
            graph,
            observations=(
                replace(
                    observation,
                    uncertainty_radius=wire.NullableRationalVectorV3(
                        True, (wire.RationalValueV3("-1", "1"),)
                    ),
                ),
            ) + graph.observations[1:],
        )
    with pytest.raises(ValueError, match="component-complete"):
        replace(
            graph,
            observations=(
                replace(
                    observation,
                    uncertainty_radius=wire.NullableRationalVectorV3(
                        True,
                        (
                            wire.RationalValueV3("1", "1"),
                            wire.RationalValueV3("1", "1"),
                        ),
                    ),
                ),
            ) + graph.observations[1:],
        )
    with pytest.raises(ValueError, match="unitless discrete quantity"):
        replace(
            graph,
            observations=(
                replace(
                    observation,
                    components=(
                        wire.FormalComponentDescriptorV3(
                            "scalar", "scalar", "missing"
                        ),
                    ),
                    numeric_value=wire.NullableRationalVectorV3(False, None),
                    boolean_value=wire.NullableBooleanV3(False, None),
                    uncertainty_model="not_applicable",
                    uncertainty_radius=wire.NullableRationalVectorV3(False, None),
                ),
            )
            + graph.observations[1:],
        )
    with pytest.raises(ValueError, match="unitless sentinel"):
        replace(
            graph,
            quantities=graph.quantities
            + (
                wire.FormalQuantityRecordV3(
                    _uuid(999_999_949),
                    "unused-boolean",
                    (1, 0, 0, 0, 0, 0, 0),
                    wire.FormalUnitBindingV3(
                        "forbidden-unit",
                        wire.RationalValueV3("7", "1"),
                        wire.RationalValueV3("9", "1"),
                    ),
                    "boolean",
                ),
            ),
        )

    # Distinct context IDs may carry equal semantics, but identity must reject
    # any target-only drift in reference kind or coordinate-label meaning.
    target_ref = graph.transforms[0].output_refs[0]
    target_context_id = _uuid(999_999_950)
    target_context = replace(context, context_id=target_context_id)
    target_observations = tuple(
        replace(item, context_id=target_context_id)
        if item.observation_id == target_ref.observation_id
        else item
        for item in graph.observations
    )
    remapped_target_ref = replace(target_ref, context_id=target_context_id)
    split_context_graph = replace(
        graph,
        contexts=(context, target_context),
        observations=target_observations,
        aggregate_maps=(
            replace(
                graph.aggregate_maps[0],
                groups=(
                    replace(
                        graph.aggregate_maps[0].groups[0],
                        output_refs=(remapped_target_ref,),
                    ),
                ),
            ),
        ),
        transforms=(
            replace(
                graph.transforms[0],
                output_refs=(remapped_target_ref,),
                kernel_rows=(
                    replace(
                        graph.transforms[0].kernel_rows[0],
                        output_ref=remapped_target_ref,
                    ),
                ),
            ),
        ),
    )
    assert wire.FormalSourceCaseV3(split_context_graph).graph is split_context_graph
    for semantic_drift in (
        replace(target_context, temporal_reference_kind="domain_epoch"),
        replace(target_context, spatial_reference_kind="geodetic"),
        replace(target_context, coordinate_labels=("longitude",)),
    ):
        with pytest.raises(ValueError, match="full context semantics"):
            replace(
                split_context_graph,
                contexts=(context, semantic_drift),
            )
    with pytest.raises(ValueError, match="realizable schema bound"):
        wire.FormalBatchContextV3("1", ZERO_DIGEST, "999999999", ZERO_DIGEST)
    with pytest.raises(ValueError, match="already be NFC"):
        wire.FormalSemanticDescriptorV3("natural_language", ("e\u0301",))
    for unsafe_text in (
        "bad\nsegment",
        "zero\u200dwidth",
        "bidi\u202econtrol",
        "private\ue000use",
        "line\u2028separator",
    ):
        with pytest.raises(ValueError, match="invisible/private/undefined"):
            wire.FormalSemanticDescriptorV3("natural_language", (unsafe_text,))
    padded = wire.FormalSemanticDescriptorV3(
        "natural_language", ("keyword right padded ",)
    )
    assert padded.segments == ("keyword right padded ",)
    with pytest.raises(ValueError, match="segment exceeds"):
        wire.FormalSemanticDescriptorV3("natural_language", ("x" * 1_025,))
    with pytest.raises(ValueError, match="kind"):
        wire.FormalSemanticDescriptorV3("high_overlap", ("valid evidence",))
    for hidden_identifier in (
        "evidence id=" + graph.bundle_id + ".",
        "evidence root=" + ZERO_DIGEST + ".",
    ):
        with pytest.raises(ValueError, match="identifier/digest"):
            wire.FormalSemanticDescriptorV3(
                "natural_language", (hidden_identifier,)
            )

    batch = build((source_case(0),))
    decoded = wire.decode_and_audit_formal_envelope_v3(batch.envelopes[0].envelope)
    descriptor_leaves = tuple(
        row
        for row in decoded.classified_leaves
        if "/semantic_descriptor/" in row.json_pointer
    )
    assert descriptor_leaves and all(
        row.classification is wire.PublicLeafClassV3.EVIDENCE_BEARING
        for row in descriptor_leaves
    )
    assert all(
        forbidden not in decoded.payload["public_case"]["graph"]["semantic_descriptor"]
        for forbidden in (
            "challenge_label", "overlap_stratum", "private_truth", "filename",
            "source_order", "row_order",
        )
    )
    hidden = deepcopy(decoded.payload)
    hidden["public_case"]["graph"]["semantic_descriptor"]["challenge_label"] = "low"
    with pytest.raises(ValueError):
        wire._normalize_formal_payload_structure_v3(hidden)


def test_unicode_13_descriptor_profile_and_format_channel_kats() -> None:
    literal_ranges = (
        (0x00AD, 0x00AD),
        (0x034F, 0x034F),
        (0x061C, 0x061C),
        (0x115F, 0x1160),
        (0x17B4, 0x17B5),
        (0x180B, 0x180E),
        (0x200B, 0x200F),
        (0x202A, 0x202E),
        (0x2060, 0x206F),
        (0x3164, 0x3164),
        (0xFE00, 0xFE0F),
        (0xFEFF, 0xFEFF),
        (0xFFA0, 0xFFA0),
        (0xFFF0, 0xFFF8),
        (0x1BCA0, 0x1BCA3),
        (0x1D173, 0x1D17A),
        (0xE0000, 0xE0FFF),
    )
    assert unicodedata.unidata_version == wire.UNICODE_DATA_VERSION_V3 == "13.0.0"
    assert wire.UCD13_DERIVED_CORE_PROPERTIES_SHA256_V3 == (
        "sha256:a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248"
    )
    assert wire.UCD13_PROP_LIST_SHA256_V3 == (
        "sha256:485b5a3ed25dbf1f94dfa5a9b69d8b4550ffd0c33045ccc55ccfd7c80b2a40cf"
    )
    assert wire._DEFAULT_IGNORABLE_RANGES_V3 == literal_ranges
    assert sum(upper - lower + 1 for lower, upper in literal_ranges) == 4_173
    literal_table_digest = "sha256:" + hashlib.sha256(
        wire.canonicalize_rfc8785_v1([list(row) for row in literal_ranges])
    ).hexdigest()
    assert literal_table_digest == wire.DEFAULT_IGNORABLE_RANGE_TABLE_SHA256_V3 == (
        "sha256:629f6c5f9f20e8322371ff59b61ce34246a9c4afb8fd299756540058dc8a553e"
    )
    assert wire._is_default_ignorable_v3("\u180e") is True
    assert wire._is_default_ignorable_v3("\u180f") is False
    for unsafe_text in ("variation\ufe0f", "noncharacter\ufdd0"):
        with pytest.raises(ValueError):
            wire.FormalSemanticDescriptorV3("natural_language", (unsafe_text,))

    graph = source_case(0).graph

    def descriptor_channel(segment: str) -> wire.SemanticDescriptorChannelV3:
        descriptor_graph = replace(
            graph,
            semantic_descriptor=wire.FormalSemanticDescriptorV3(
                "natural_language", (segment,)
            ),
        )
        public_case = wire.FormalPublicCaseV3(
            wire.FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION,
            descriptor_graph,
            ZERO_DIGEST,
        )
        return wire._channel_projection_v3(
            public_case,
            wire.MAXIMUM_PAYLOAD_BYTES_V3,
            b"p" * wire.MINIMUM_PADDING_BYTES_V3,
        ).semantic_descriptor_channel

    # U+2000/U+2001 are rejected by the prior NFC requirement under UCD 13;
    # use two NFC-stable, equal-width Zs scalars to prove the channel records
    # the actual formatting code point rather than only category/byte length.
    for non_nfc_space in ("a\u2000b", "a\u2001b"):
        with pytest.raises(ValueError, match="already be NFC"):
            wire.FormalSemanticDescriptorV3(
                "natural_language", (non_nfc_space,)
            )
    en_space = descriptor_channel("a\u2002b")
    em_space = descriptor_channel("a\u2003b")
    comma = descriptor_channel("a,b")
    semicolon = descriptor_channel("a;b")
    assert en_space.segment_non_alnum_codepoint_sequences == ((0x2002,),)
    assert em_space.segment_non_alnum_codepoint_sequences == ((0x2003,),)
    assert comma.segment_non_alnum_codepoint_sequences == ((ord(","),),)
    assert semicolon.segment_non_alnum_codepoint_sequences == ((ord(";"),),)
    assert en_space != em_space and comma != semicolon


def test_fresh_header_cache_batch_success_and_single_envelope_uuid_replay() -> None:
    success = build_success((source_case(0),))
    batch = success.public_batch
    decoded = wire.decode_and_audit_formal_envelope_v3(batch.envelopes[0].envelope)
    other = build_success((source_case(1),))
    other_decoded = wire.decode_and_audit_formal_envelope_v3(
        other.public_batch.envelopes[0].envelope
    )
    with pytest.raises(ValueError, match="payload-spliced"):
        replace(decoded, batch_context=other_decoded.batch_context)
    with pytest.raises(ValueError, match="payload-spliced"):
        replace(decoded, public_case=other_decoded.public_case)
    arbitrary_payload_bytes = b"{}"
    arbitrary_padding = b"p" * (
        wire.ENVELOPE_BYTES_V3
        - wire.ENVELOPE_HEADER_BYTES_V3
        - len(arbitrary_payload_bytes)
    )
    arbitrary_envelope = wire._frame_payload_with_padding_v3(
        arbitrary_payload_bytes, arbitrary_padding
    )
    arbitrary_cache = unsafe_issued_copy(
        decoded,
        envelope=arbitrary_envelope,
        envelope_id="phase2b_formal_trusted_envelope_v3_"
        + hashlib.sha256(
            wire._ENVELOPE_ID_DOMAIN_V3 + arbitrary_envelope
        ).hexdigest(),
        payload={},
        payload_bytes=arbitrary_payload_bytes,
        padding_bytes=arbitrary_padding,
        payload_length=len(arbitrary_payload_bytes),
        padding_length=len(arbitrary_padding),
        payload_sha256="sha256:"
        + hashlib.sha256(arbitrary_payload_bytes).hexdigest(),
        padding_sha256="sha256:" + hashlib.sha256(arbitrary_padding).hexdigest(),
    )
    with pytest.raises(ValueError, match="closed field set"):
        arbitrary_cache.__post_init__()

    foreign_batch_id = wire._batch_id_v3(
        other.public_batch.batch_context,
        batch.key_source_receipt_id,
        batch.envelope_ids,
    )
    with pytest.raises(ValueError, match="contexts disagree"):
        replace(
            batch,
            batch_context=other.public_batch.batch_context,
            batch_id=foreign_batch_id,
        )
    replayed = wire.replay_formal_trusted_wire_batch_v3(
        (decoded.envelope,), key_source_receipt_id=batch.key_source_receipt_id
    )
    assert type(replayed) is wire.FormalTrustedWireBatchReplayV3
    with pytest.raises(ValueError, match="contexts disagree"):
        replace(
            replayed,
            batch_context=other.public_batch.batch_context,
            batch_id=foreign_batch_id,
        )

    changed_header = bytearray(decoded.envelope)
    changed_header[0] ^= 1
    with pytest.raises(ValueError, match="header"):
        replace(decoded, envelope=bytes(changed_header))

    forged_payload = deepcopy(decoded.payload)
    forged_payload["public_case"]["graph"]["semantic_descriptor"]["segments"][0] = "forged"
    forged_decoded = unsafe_issued_copy(decoded, payload=forged_payload)
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (decoded.envelope,), key_source_receipt_id=batch.key_source_receipt_id
    )
    assert type(replay) is wire.FormalTrustedWireBatchReplayV3
    with pytest.raises(ValueError, match="fresh"):
        replace(replay, decoded_envelopes=(forged_decoded,))

    spliced_success = unsafe_issued_copy(
        success,
        private_audit_alignment=other.private_audit_alignment,
    )
    with pytest.raises(ValueError):
        spliced_success._validate_issued_shape()
    foreign_alignment_id = other.private_audit_alignment.alignment_id
    forged_context_id = wire._variant_context_id_v3(
        batch_id=success.variant_context.batch_id,
        private_alignment_id=foreign_alignment_id,
        transform_catalog_root=success.variant_context.transform_catalog_root,
        catalog_binding_receipt_id=success.variant_context.catalog_binding_receipt_id,
        base_envelopes=success.variant_context.base_envelopes,
    )
    forged_context = unsafe_issued_copy(
        success.variant_context,
        private_alignment_id=foreign_alignment_id,
        context_id=forged_context_id,
    )
    forged_success = unsafe_issued_copy(
        success,
        variant_context=forged_context,
    )
    with pytest.raises(ValueError, match="batch/catalog stale"):
        forged_success._validate_issued_shape()

    payload = deepcopy(decoded.payload)
    graph_mapping = payload["public_case"]["graph"]
    shared = graph_mapping["bundle_id"]
    graph_mapping["entities"][0]["entity_id"] = shared
    for observation_row in graph_mapping["observations"]:
        observation_row["entity_ids"] = [shared]
    graph_mapping["task_target"]["entity_ids"] = [shared]
    wire._refresh_graph_provenance_v3(
        graph_mapping, decoded.batch_context.transform_catalog_root
    )
    graph_object = wire._graph_from_mapping_v3(graph_mapping)
    normalized_graph = wire._profile_value_v3(graph_object)
    payload["public_case"]["graph"] = normalized_graph
    payload["public_case"]["public_provenance_commitment"] = "sha256:" + hashlib.sha256(
        wire._CASE_PROVENANCE_DOMAIN_V3
        + wire.canonicalize_rfc8785_v1(
            {
                "transform_catalog_root": decoded.batch_context.transform_catalog_root,
                "graph": normalized_graph,
            }
        )
    ).hexdigest()
    forged_envelope = wire._frame_payload_with_padding_v3(
        wire.canonicalize_rfc8785_v1(payload), decoded.padding_bytes
    )
    with pytest.raises(ValueError, match="across namespaces"):
        wire.decode_and_audit_formal_envelope_v3(forged_envelope)


def test_input_tuple_and_record_order_are_erased_before_shuffle() -> None:
    cases = (source_case(0), source_case(1), source_case(2))
    forward = build(cases)
    reverse = build(tuple(reversed(cases)))
    assert tuple(row.envelope for row in forward.envelopes) == tuple(
        row.envelope for row in reverse.envelopes
    )
    reordered = wire.FormalSourceCaseV3(
        replace(cases[0].graph, observations=tuple(reversed(cases[0].graph.observations)))
    )
    assert build((cases[0],)).envelopes[0].envelope == build((reordered,)).envelopes[0].envelope


def test_key_independence_kat_and_overlap_rejection() -> None:
    derived = wire._derive_keys_v3(b"R" * 32, keys())
    assert tuple(value.hex() for value in derived) == (
        "edf19b5c5aa772bc5dcd0bebd6c0df7ebae92c64185d79905722f1c6a89e5eff",
        "9aebef00ea2f7fcf422c380e7598343c73f8661048fd64366226dadafbf184be",
        "6967cfbc25c0b59ec75ade73a2e7ad534fad36c14b860d893e53098016bf7a56",
    )
    assert wire._shuffle_indices_v3(8, derived[0], b"R" * 32) == (
        2, 7, 0, 5, 4, 1, 3, 6
    )
    assert wire._uuid_candidate_v3(
        derived[1], b"R" * 32, "observation_id", 7, 2
    ) == "172bd4aa-75b1-48d1-bdaf-b07fded17305"
    assert wire._padding_v3(derived[2], b"R" * 32, bytes(range(32)), 96).hex() == (
        "5cbb258bfbf092549fc7ba62f65e58f1b6c1541d001863cd75406c52420beec8"
        "f1d5ce41fab8a117aa98d70cbc3f4cfd74cc1c71e4501d708b7afe5b168385d65"
        "86c6d01cf3ebc3848a918c921eb20935aaca265ce2794de2163aa89d7f2637c"
    )
    case = source_case(0)
    global_catalog, binding = catalog((case,))
    overlap = wire.issue_key_source_transcript_receipt_v3(
        key_sources=keys(),
        latent_seed_commitments=(wire.seed_commitment_v3(b"S" * 32),),
        answer_seed_commitments=(wire.seed_commitment_v3(b"A" * 32),),
    )
    result = wire.build_formal_trusted_wire_batch_v3(
        source_cases=(case,),
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"R" * 32,
        key_sources=keys(),
        key_source_receipt=overlap,
    )
    assert type(result) is wire.FormalTrustedWireBatchRejectionV3
    assert result.envelopes == ()


@pytest.mark.parametrize("forced_collisions", (0, 1, 10, 11))
def test_collision_retry_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    forced_collisions: int,
) -> None:
    original = wire._uuid_candidate_v3

    def candidate(
        key: bytes, run_id: bytes, namespace: str, counter: int, retry: int
    ) -> str:
        if namespace == "scale_id" and counter == 1 and retry < forced_collisions:
            return original(key, run_id, namespace, 0, 0)
        return original(key, run_id, namespace, counter, retry)

    monkeypatch.setattr(wire, "_uuid_candidate_v3", candidate)
    result = build((source_case(0),)) if forced_collisions <= 10 else None
    if result is not None:
        assert result.batch_context.id_collision_retry_count_decimal == str(
            forced_collisions
        )
    else:
        case = source_case(0)
        global_catalog, binding = catalog((case,))
        rejection = wire.build_formal_trusted_wire_batch_v3(
            source_cases=(case,),
            global_transform_catalog=global_catalog,
            transform_catalog_binding_receipt=binding,
            run_id=b"R" * 32,
            key_sources=keys(),
            key_source_receipt=receipt(),
        )
        assert type(rejection) is wire.FormalTrustedWireBatchRejectionV3


def test_public_uuid_allocator_retries_against_complete_source_uuid_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = source_case(0)
    source_uuids = {
        value
        for _, _, value in wire._walk_uuid_leaves_v3(
            wire._profile_value_v3(case.graph)
        )
    }
    original = wire._uuid_candidate_v3

    def candidate(
        key: bytes, run_id: bytes, namespace: str, counter: int, retry: int
    ) -> str:
        if namespace == "bundle_id" and counter == 0 and retry == 0:
            return case.graph.bundle_id
        return original(key, run_id, namespace, counter, retry)

    monkeypatch.setattr(wire, "_uuid_candidate_v3", candidate)
    first = build_success((case,))
    second = build_success((case,))
    assert first.public_batch.batch_context.id_collision_retry_count_decimal == "1"
    assert tuple(row.envelope for row in first.public_batch.envelopes) == tuple(
        row.envelope for row in second.public_batch.envelopes
    )
    public_uuids = {
        channel.public_uuid
        for row in first.public_batch.envelopes
        for channel in row.channel_projection.namespace_uuid_channels
    }
    assert source_uuids.isdisjoint(public_uuids)


def test_alignment_constructor_and_all_stale_views_reject() -> None:
    success = build_success()
    batch = success.public_batch
    alignment = success.private_audit_alignment
    cooperative_copy = wire.FormalPrivateAuditAlignmentV3(
        batch_id=alignment.batch_id,
        run_id_commitment=alignment.run_id_commitment,
        key_source_receipt_id=alignment.key_source_receipt_id,
        source_case_count=alignment.source_case_count,
        links_in_public_order=alignment.links_in_public_order,
        alignment_id=alignment.alignment_id,
    )
    assert cooperative_copy == alignment
    first = alignment.links_in_public_order[0]
    attacks = (
        {
            "links_in_public_order": (
                replace(first, source_row_index=alignment.links_in_public_order[1].source_row_index),
                alignment.links_in_public_order[1],
            )
        },
        {"batch_id": "phase2b_formal_trusted_wire_batch_v3_" + "0" * 64},
        {
            "links_in_public_order": (
                replace(first, source_case_commitment=ZERO_DIGEST),
                alignment.links_in_public_order[1],
            )
        },
    )
    for changes in attacks:
        forged = unsafe_alignment_copy(alignment, **changes)
        with pytest.raises(ValueError):
            wire.verify_private_audit_alignment_v3(batch=batch, alignment=forged)
    private_rows = wire.issue_formal_private_audit_rows_v3(
        build_success=success,
        private_payloads_in_source_order=(b"zero", b"one"),
    )
    with pytest.raises(ValueError, match="swapped, stale"):
        wire.align_private_rows_to_public_v3(
            batch=batch,
            alignment=alignment,
            private_rows_in_source_order=tuple(reversed(private_rows)),
        )
    stale = build_success((source_case(2), source_case(3)))
    stale_rows = wire.issue_formal_private_audit_rows_v3(
        build_success=stale,
        private_payloads_in_source_order=(b"zero", b"one"),
    )
    with pytest.raises(ValueError, match="swapped, stale"):
        wire.align_private_rows_to_public_v3(
            batch=batch,
            alignment=alignment,
            private_rows_in_source_order=stale_rows,
        )
    assert success.origin_authenticated is False
    assert all(row.semantic_truth_verified is False for row in private_rows)
    assert not hasattr(wire, "_PRIVATE_ALIGNMENT_ISSUE_TOKEN_V3")


def _replace_uuid_strings(value: object, replacements: dict[str, str]) -> object:
    if type(value) is dict:
        return {key: _replace_uuid_strings(item, replacements) for key, item in value.items()}
    if type(value) is list:
        return [_replace_uuid_strings(item, replacements) for item in value]
    if type(value) is str:
        return replacements.get(value, value)
    return value


def test_32_global_renames_and_16_observation_orders_redecode() -> None:
    success = build_success((source_case(0),))
    decoded = wire.decode_and_audit_formal_envelope_v3(
        success.public_batch.envelopes[0].envelope
    )
    rename_envelope_ids: set[str] = set()
    for variant in range(32):
        renamings = total_nonmonotonic_renamings(success, variant)
        by_source = tuple(sorted(renamings, key=lambda row: row.source_uuid))
        assert tuple(row.target_uuid for row in by_source) != tuple(
            sorted(row.target_uuid for row in by_source)
        )
        schedule = identity_variant_schedule(success, renamings=renamings)
        expected = wire.make_formal_variant_payloads_v3(
            build_success=success,
            schedule=schedule,
        )[0]
        envelope = wire.reencode_formal_variant_payload_envelope_v3(
            payload=deepcopy(expected.payload),
            padding_bytes=expected.padding_bytes,
            build_success=success,
            schedule=schedule,
            envelope_id=decoded.envelope_id,
        )
        renamed = wire.decode_and_audit_formal_envelope_v3(envelope)
        rename_envelope_ids.add(renamed.envelope_id)
        renamed_values = {
            value
            for _, _, value in wire._walk_uuid_leaves_v3(
                renamed.payload["public_case"]["graph"]
            )
        }
        assert renamed_values == {row.target_uuid for row in renamings}
    assert len(rename_envelope_ids) == 32

    base_observation_ids = tuple(
        item.observation_id for item in decoded.public_case.graph.observations
    )
    observation_envelope_ids: set[str] = set()
    for expected_order in itertools.islice(
        itertools.permutations(base_observation_ids), 16
    ):
        schedule = identity_variant_schedule(
            success,
            observation_order_by_envelope={decoded.envelope_id: expected_order},
        )
        payload_row = wire.make_formal_variant_payloads_v3(
            build_success=success,
            schedule=schedule,
        )[0]
        envelope = wire.reencode_formal_variant_payload_envelope_v3(
            payload=payload_row.payload,
            padding_bytes=payload_row.padding_bytes,
            build_success=success,
            schedule=schedule,
            envelope_id=decoded.envelope_id,
        )
        replayed = wire.decode_and_audit_formal_envelope_v3(envelope)
        observation_envelope_ids.add(replayed.envelope_id)
        assert tuple(
            item["observation_id"]
            for item in replayed.payload["public_case"]["graph"]["observations"]
        ) == expected_order
    assert len(observation_envelope_ids) == 16

    five = build_success(tuple(source_case(index) for index in range(5)))
    observed_case_orders: set[tuple[str, ...]] = set()
    for case_order in itertools.islice(
        itertools.permutations(five.public_batch.envelope_ids), 32
    ):
        schedule = identity_variant_schedule(five, case_order=case_order)
        rows = wire.make_formal_variant_payloads_v3(
            build_success=five,
            schedule=schedule,
        )
        actual = tuple(row.envelope_id for row in rows)
        assert actual == case_order
        observed_case_orders.add(actual)
    assert len(observed_case_orders) == 32


def test_combined_uuid_observation_and_case_order_variant_replays() -> None:
    success = build_success((source_case(0), source_case(1)))
    reversed_observations: dict[str, tuple[str, ...]] = {}
    for row in success.public_batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(row.envelope)
        reversed_observations[row.envelope_id] = tuple(
            reversed(
                tuple(
                    observation.observation_id
                    for observation in decoded.public_case.graph.observations
                )
            )
        )
    case_order = tuple(reversed(success.public_batch.envelope_ids))
    renamings = total_nonmonotonic_renamings(success, 97)
    schedule = identity_variant_schedule(
        success,
        case_order=case_order,
        observation_order_by_envelope=reversed_observations,
        renamings=renamings,
    )
    payload_rows = wire.make_formal_variant_payloads_v3(
        build_success=success,
        schedule=schedule,
    )
    assert tuple(row.envelope_id for row in payload_rows) == case_order
    envelopes = tuple(
        wire.reencode_formal_variant_payload_envelope_v3(
            payload=deepcopy(row.payload),
            padding_bytes=row.padding_bytes,
            build_success=success,
            schedule=schedule,
            envelope_id=row.envelope_id,
        )
        for row in payload_rows
    )
    decoded = tuple(wire.decode_and_audit_formal_envelope_v3(row) for row in envelopes)
    for row, base_envelope_id in zip(decoded, case_order, strict=True):
        assert tuple(
            observation.observation_id
            for observation in row.public_case.graph.observations
        ) == tuple(
            next(
                rename.target_uuid
                for rename in renamings
                if rename.namespace == "observation_id"
                and rename.source_uuid == source_uuid
            )
            for source_uuid in reversed_observations[base_envelope_id]
        )
    replay = wire.replay_formal_trusted_wire_batch_v3(
        envelopes, key_source_receipt_id=success.public_batch.key_source_receipt_id
    )
    assert type(replay) is wire.FormalTrustedWireBatchReplayV3
    assert replay.structural_replay_verified is True
    assert replay.catalog_membership_verified is False


def test_variant_suite_rich32_exact_80_schedule_surface() -> None:
    success = build_success(tuple(source_case(index) for index in range(32)))
    base_orders: dict[str, tuple[str, ...]] = {}
    uuid_domain: dict[str, str] = {}
    for envelope in success.public_batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(envelope.envelope)
        base_orders[envelope.envelope_id] = tuple(
            observation.observation_id
            for observation in decoded.public_case.graph.observations
        )
        for _, namespace, value in wire._walk_uuid_leaves_v3(
            decoded.payload["public_case"]["graph"]
        ):
            assert uuid_domain.setdefault(value, namespace) == namespace

    def schedule(
        *,
        case_order: tuple[str, ...] | None = None,
        renamings: tuple[wire.FormalUuidRenamingV3, ...] = (),
        observation_override: tuple[str, tuple[str, ...]] | None = None,
    ) -> wire.FormalVariantScheduleV3:
        overrides = {} if observation_override is None else {
            observation_override[0]: observation_override[1]
        }
        return wire.FormalVariantScheduleV3(
            success.public_batch.envelope_ids if case_order is None else case_order,
            renamings,
            tuple(
                wire.FormalObservationOrderV3(
                    envelope_id, overrides.get(envelope_id, observation_ids)
                )
                for envelope_id, observation_ids in sorted(base_orders.items())
            ),
        )

    sources = sorted(uuid_domain)
    rename_schedules: list[wire.FormalVariantScheduleV3] = []
    for variant in range(32):
        targets = [
            _uuid(30_000_000_000 + variant * 100_000 + index)
            for index in range(len(sources))
        ]
        targets.reverse()
        renamings = tuple(
            sorted(
                (
                    wire.FormalUuidRenamingV3(
                        uuid_domain[source], source, target
                    )
                    for source, target in zip(sources, targets, strict=True)
                ),
                key=lambda row: (row.namespace, row.source_uuid),
            )
        )
        rename_schedules.append(schedule(renamings=renamings))

    first_envelope_id = success.public_batch.envelope_ids[0]
    observation_schedules = tuple(
        schedule(observation_override=(first_envelope_id, order))
        for order in itertools.islice(
            itertools.permutations(base_orders[first_envelope_id]), 1, 17
        )
    )
    envelope_ids = success.public_batch.envelope_ids
    case_orders = tuple(
        envelope_ids[offset:] + envelope_ids[:offset]
        for offset in range(1, len(envelope_ids))
    ) + (tuple(reversed(envelope_ids)),)
    case_schedules = tuple(schedule(case_order=order) for order in case_orders)
    schedules = (
        *rename_schedules,
        *observation_schedules,
        *case_schedules,
    )
    assert len(schedules) == wire.MAXIMUM_VARIANT_SUITE_SCHEDULES_V3 == 80
    suite = wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=schedules
    )
    assert len(suite.batches) == len(suite.schedule_roots) == 80
    assert len(set(suite.schedule_roots)) == 80
    assert all(len(batch.envelopes) == 32 for batch in suite.batches)
    assert sum(len(batch.envelopes) for batch in suite.batches) == 2_560
    assert sum(
        len(envelope)
        for batch in suite.batches
        for envelope in batch.envelopes
    ) == 2_560 * wire.ENVELOPE_BYTES_V3


def test_variant_suite_capacity_1024_exact_80_schedule_surface() -> None:
    success = build_success(tuple(source_case(index) for index in range(1_024)))
    base_orders: dict[str, tuple[str, ...]] = {}
    uuid_domain: dict[str, str] = {}
    for envelope in success.public_batch.envelopes:
        decoded = wire.decode_and_audit_formal_envelope_v3(envelope.envelope)
        base_orders[envelope.envelope_id] = tuple(
            observation.observation_id
            for observation in decoded.public_case.graph.observations
        )
        for _, namespace, value in wire._walk_uuid_leaves_v3(
            decoded.payload["public_case"]["graph"]
        ):
            assert uuid_domain.setdefault(value, namespace) == namespace

    def schedule(
        *,
        case_order: tuple[str, ...] | None = None,
        renamings: tuple[wire.FormalUuidRenamingV3, ...] = (),
        observation_override: tuple[str, tuple[str, ...]] | None = None,
    ) -> wire.FormalVariantScheduleV3:
        overrides = (
            {}
            if observation_override is None
            else {observation_override[0]: observation_override[1]}
        )
        return wire.FormalVariantScheduleV3(
            success.public_batch.envelope_ids
            if case_order is None
            else case_order,
            renamings,
            tuple(
                wire.FormalObservationOrderV3(
                    envelope_id, overrides.get(envelope_id, observation_ids)
                )
                for envelope_id, observation_ids in sorted(base_orders.items())
            ),
        )

    sources = sorted(uuid_domain)
    schedules: list[wire.FormalVariantScheduleV3] = []
    for variant in range(32):
        targets = [
            _uuid(40_000_000_000 + variant * 1_000_000 + index)
            for index in range(len(sources))
        ]
        targets.reverse()
        schedules.append(
            schedule(
                renamings=tuple(
                    sorted(
                        (
                            wire.FormalUuidRenamingV3(
                                uuid_domain[source], source, target
                            )
                            for source, target in zip(
                                sources, targets, strict=True
                            )
                        ),
                        key=lambda row: (row.namespace, row.source_uuid),
                    )
                )
            )
        )
    first_envelope_id = success.public_batch.envelope_ids[0]
    schedules.extend(
        schedule(observation_override=(first_envelope_id, order))
        for order in itertools.islice(
            itertools.permutations(base_orders[first_envelope_id]), 1, 17
        )
    )
    envelope_ids = success.public_batch.envelope_ids
    schedules.extend(
        schedule(case_order=envelope_ids[offset:] + envelope_ids[:offset])
        for offset in range(1, 32)
    )
    schedules.append(schedule(case_order=tuple(reversed(envelope_ids))))
    assert len(schedules) == wire.MAXIMUM_VARIANT_SUITE_SCHEDULES_V3 == 80

    suite = wire.reencode_formal_variant_suite_v3(
        build_success=success, schedules=tuple(schedules)
    )
    assert len(suite.batches) == len(suite.schedule_roots) == 80
    assert len(set(suite.schedule_roots)) == 80
    assert all(len(batch.envelopes) == 1_024 for batch in suite.batches)
    assert sum(len(batch.envelopes) for batch in suite.batches) == 81_920
    assert sum(
        len(envelope)
        for batch in suite.batches
        for envelope in batch.envelopes
    ) == wire.MAXIMUM_VARIANT_SUITE_OUTPUT_BYTES_V3
    assert all(
        value is False
        for value in (
            suite.key_source_transcript_verified,
            suite.catalog_membership_verified,
            suite.origin_authenticated,
            suite.formal_covert_audit,
            suite.actual_execution_claimed,
        )
    )


def test_payload_exact_boundary_and_plus_one() -> None:
    payload = b'"' + b"a" * (wire.MAXIMUM_PAYLOAD_BYTES_V3 - 2) + b'"'
    envelope = wire._frame_payload_with_padding_v3(payload, b"p" * 32)
    assert len(envelope) == wire.ENVELOPE_BYTES_V3
    with pytest.raises(ValueError):
        wire._frame_payload_with_padding_v3(payload + b"x", b"p" * 31)


def test_deep_bounded_json_and_memory_fail_closed_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at_container_limit = b"[" + b",".join(
        [b"[]"] * (wire.MAXIMUM_JSON_CONTAINER_TOKENS_V3 - 1)
    ) + b"]"
    above_container_limit = b"[" + b",".join(
        [b"[]"] * wire.MAXIMUM_JSON_CONTAINER_TOKENS_V3
    ) + b"]"
    assert len(above_container_limit) <= wire.MAXIMUM_PAYLOAD_BYTES_V3
    wire._bounded_json_wire_preflight_v3(at_container_limit)
    with pytest.raises(ValueError, match="container budget"):
        wire._bounded_json_wire_preflight_v3(above_container_limit)

    payload = b"[" * 65 + b"0" + b"]" * 65
    assert len(payload) < wire.MAXIMUM_PAYLOAD_BYTES_V3
    padding = b"p" * (
        wire.ENVELOPE_BYTES_V3
        - wire.ENVELOPE_HEADER_BYTES_V3
        - len(payload)
    )
    envelope = wire._frame_payload_with_padding_v3(payload, padding)
    with pytest.raises(ValueError, match="nesting"):
        wire.decode_and_audit_formal_envelope_v3(envelope)
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (envelope,), key_source_receipt_id=receipt().receipt_id
    )
    assert type(replay) is wire.FormalTrustedWireBatchRejectionV3
    assert replay.reason == "bounded_public_batch_replay_failed"
    assert replay.envelopes == ()

    valid_envelope = build((source_case(0),)).envelopes[0].envelope

    def exhausted(_: bytes) -> None:
        raise MemoryError

    monkeypatch.setattr(wire, "_bounded_json_wire_preflight_v3", exhausted)
    with pytest.raises(ValueError) as failure:
        wire.decode_and_audit_formal_envelope_v3(valid_envelope)
    assert str(failure.value) == "bounded formal envelope decode rejected"
    replay = wire.replay_formal_trusted_wire_batch_v3(
        (valid_envelope,), key_source_receipt_id=receipt().receipt_id
    )
    assert type(replay) is wire.FormalTrustedWireBatchRejectionV3
    assert replay.envelopes == ()


def test_capacity_1024_build_and_1025_reject() -> None:
    cases = tuple(source_case(index) for index in range(1_024))
    global_catalog, binding = catalog(cases)
    success = wire.build_formal_trusted_wire_batch_v3(
        source_cases=cases,
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"C" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(success) is wire.FormalTrustedWireBuildSuccessV3
    batch = success.public_batch
    assert len(batch.envelopes) == 1_024
    assert all(len(row.envelope) == wire.ENVELOPE_BYTES_V3 for row in batch.envelopes)
    assert batch.key_source_transcript_verified is False
    assert batch.key_source_independence_verified is False
    assert batch.capacity_1024_qualified is False
    assert batch.formal_covert_audit is False
    assert batch.actual_execution_claimed is False
    assert success.builder_input_transcript_consistency_checked is True
    assert success.builder_input_catalog_membership_checked is True
    assert success.private_alignment_linkage_checked is True
    assert success.origin_authenticated is False
    rejection = wire.build_formal_trusted_wire_batch_v3(
        source_cases=cases + (source_case(1_024),),
        global_transform_catalog=global_catalog,
        transform_catalog_binding_receipt=binding,
        run_id=b"C" * 32,
        key_sources=keys(),
        key_source_receipt=receipt(),
    )
    assert type(rejection) is wire.FormalTrustedWireBatchRejectionV3
    assert rejection.envelopes == () and rejection.batch_id is None

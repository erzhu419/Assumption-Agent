import inspect
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction

import pytest

import hegel_machine.phase2b_exact_transform_semantics_v1 as tx
from hegel_machine.phase2b_uncertainty_compiler import compile_bundle_uncertainty
from hegel_machine.phase2b_wire import (
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
    PublicEvidenceBundle,
    TransformOperation,
)


def uid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012x}"


ROOT_SCALE = uid(100)
TARGET_SCALE = uid(101)
TRANSFORM_ID = uid(200)
ENTITY_ID = uid(300)
ROLE_ID = uid(301)
QUANTITY_ID = uid(302)
CHANNEL_ID = uid(303)
UNIT_A = uid(304)
UNIT_B = uid(305)
CLOCK_ID = uid(306)
FRAME_A = uid(307)
FRAME_B = uid(308)


def _observation_mapping(index: int, spec: dict[str, object]) -> dict[str, object]:
    values = list(spec.get("values", [float(index + 1)]))
    temporal = spec.get("temporal")
    spatial = spec.get("spatial")
    return {
        "observation_id": uid(1_000 + index),
        "source_channel_id": CHANNEL_ID,
        "entity_ids": [ENTITY_ID],
        "role_candidate_ids": [ROLE_ID],
        "quantity_id": QUANTITY_ID,
        "value": {"kind": "numeric", "values": values},
        "unit_dimension": {"si_exponents": [0, 0, 0, 0, 0, 0, 0]},
        "temporal_support": (
            None
            if temporal is None
            else {"clock_id": CLOCK_ID, "start": temporal[0], "end": temporal[1]}
        ),
        "spatial_support": (
            None
            if spatial is None
            else {
                "frame_id": spec.get("spatial_frame", FRAME_A),
                "lower": list(spatial[0]),
                "upper": list(spatial[1]),
            }
        ),
        "uncertainty": {
            "model": "absolute_bound",
            "radius": [0.0 for _ in values],
        },
        "provenance_sha256": f"{(index + 1) % 16:x}" * 64,
        "missingness": "observed",
    }


def _base_and_metadata(
    operation: TransformOperation,
    specs: tuple[dict[str, object], ...],
) -> tuple[
    PublicEvidenceBundle,
    tuple[tx.ObservationComponentMetadata, ...],
    tuple[tx.ComponentDescriptor, ...],
]:
    observations = tuple(
        _observation_mapping(index, spec) for index, spec in enumerate(specs)
    )
    base = PublicEvidenceBundle.from_mapping(
        {
            "schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
            "bundle_id": uid(1),
            "entity_candidates": [
                {"entity_id": ENTITY_ID, "role_candidate_ids": [ROLE_ID]}
            ],
            "role_ids": [ROLE_ID],
            "quantity_ids": [QUANTITY_ID],
            "observations": list(observations),
            "task_target": {
                "task_id": uid(2),
                "entity_ids": [ENTITY_ID],
                "quantity_ids": [QUANTITY_ID],
            },
            "aggregation_graph": {
                "scale_ids": [ROOT_SCALE, TARGET_SCALE],
                "root_scale_ids": [ROOT_SCALE],
                "edges": [
                    {
                        "source_scale_id": ROOT_SCALE,
                        "target_scale_id": TARGET_SCALE,
                        "transform_id": TRANSFORM_ID,
                    }
                ],
            },
            "transform_catalog": [
                {
                    "transform_id": TRANSFORM_ID,
                    "operation": operation.value,
                    "parameters": [],
                }
            ],
            "missingness_mask": [],
        }
    )
    metadata = tuple(
        tx.ObservationComponentMetadata(
            observation_id=observation.observation_id,
            scale_id=ROOT_SCALE,
            component_ids=tuple(
                uid(2_000 + 10 * index + ordinal)
                for ordinal in range(len(spec.get("values", [float(index + 1)])))
            ),
            axis=spec.get("axis", tx.ComponentAxis.SCALAR),
            value_role=spec.get("role", tx.ComponentValueRole.INTENSIVE),
            unit_id=spec.get("unit", UNIT_A),
            coordinate_frame_id=spec.get("coordinate_frame"),
        )
        for index, (observation, spec) in enumerate(zip(base.observations, specs, strict=True))
    )
    descriptors = tuple(
        tx._root_descriptor(observation, item, ordinal)
        for observation, item in zip(base.observations, metadata, strict=True)
        for ordinal in range(len(item.component_ids))
    )
    return base, metadata, descriptors


def _derived_observation(
    base: PublicEvidenceBundle,
    descriptors: tuple[tx.ComponentDescriptor, ...],
    source_observation_ids: tuple[str, ...],
) -> tx.DerivedObservationDescriptor:
    first = descriptors[0]
    return tx.DerivedObservationDescriptor(
        scale_id=TARGET_SCALE,
        observation_id=first.ref.observation_id,
        source_channel_id=CHANNEL_ID,
        entity_ids=(ENTITY_ID,),
        role_candidate_ids=(ROLE_ID,),
        quantity_id=QUANTITY_ID,
        unit_id=first.unit_id,
        si_exponents=first.si_exponents,
        temporal_support=first.temporal_support,
        spatial_support=first.spatial_support,
        provenance_sha256="0" * 64,
        source_observation_ids=tuple(sorted(source_observation_ids)),
        value_kind=tx.ComponentValueKind.NUMERIC_INTERVAL,
        component_refs=tuple(item.ref for item in descriptors),
    )


def _sign_authority(
    base: PublicEvidenceBundle,
    metadata: tuple[tx.ObservationComponentMetadata, ...],
    contract: tx.ExactTransformContract,
) -> tx.PublicTransformEvidenceBundleV2:
    provisional = tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        metadata,
        (contract,),
    )
    receipt = compile_bundle_uncertainty(base)
    states, _ = tx._build_root_states(provisional, receipt)
    source = states[ROOT_SCALE]
    source_roots = {}
    for state in source.values():
        observation = state.observation
        source_roots.setdefault(
            observation.observation_id,
            (observation.descriptor_id, observation),
        )
    semantics_id = contract.semantics_id
    inputs_by_output = tx._output_input_refs(contract)
    signed_observations = []
    for observation in contract.output_observations:
        contributing = tuple(
            source[input_ref]
            for output_ref in observation.component_refs
            for input_ref in inputs_by_output[output_ref]
        )
        signed_observations.append(
            replace(
                observation,
                provenance_sha256=tx._expected_derived_provenance(
                    observation,
                    contributing,
                    source_roots,
                    semantics_id,
                    (TRANSFORM_ID,),
                    (semantics_id,),
                ),
            )
        )
    signed = replace(contract, output_observations=tuple(signed_observations))
    assert signed.semantics_id == semantics_id
    return replace(provisional, transform_contracts=(signed,))


def identity_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.IDENTITY,
        ({"values": [1.0, -2.0]},),
    )
    observation_id = uid(4_000)
    outputs = tuple(
        replace(
            item,
            ref=tx.ComponentRef(
                TARGET_SCALE,
                observation_id,
                ordinal,
                uid(4_100 + ordinal),
            ),
        )
        for ordinal, item in enumerate(source)
    )
    rows = tuple(
        tx.ExactSparseAffineRow(
            output.ref,
            (tx.ExactSparseTerm(input_.ref, tx.ONE),),
        )
        for input_, output in zip(source, outputs, strict=True)
    )
    observation = _derived_observation(
        base,
        outputs,
        (base.observations[0].observation_id,),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.IDENTITY,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        outputs,
        (observation,),
        rows,
        (),
        tx.IdentityTransformCertificate(),
    )
    return _sign_authority(base, metadata, contract)


def unit_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.UNIT_CONVERSION,
        ({"values": [1.0, -2.0], "unit": UNIT_A},),
    )
    observation_id = uid(4_000)
    outputs = tuple(
        replace(
            item,
            ref=tx.ComponentRef(
                TARGET_SCALE,
                observation_id,
                ordinal,
                uid(4_100 + ordinal),
            ),
            unit_id=UNIT_B,
        )
        for ordinal, item in enumerate(source)
    )
    factor = tx.ExactTransformAtom(2)
    rows = tuple(
        tx.ExactSparseAffineRow(
            output.ref,
            (tx.ExactSparseTerm(input_.ref, factor),),
        )
        for input_, output in zip(source, outputs, strict=True)
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.UNIT_CONVERSION,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        outputs,
        (
            _derived_observation(
                base,
                outputs,
                (base.observations[0].observation_id,),
            ),
        ),
        rows,
        (),
        tx.UnitConversionCertificate(
            UNIT_A,
            UNIT_B,
            factor,
            tx.ExactTransformAtom(1, 2),
        ),
    )
    return _sign_authority(base, metadata, contract)


def coordinate_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.COORDINATE_AFFINE,
        (
            {
                "values": [0.25, 0.75],
                "axis": tx.ComponentAxis.COORDINATE,
                "role": tx.ComponentValueRole.COORDINATE,
                "unit": UNIT_A,
                "coordinate_frame": FRAME_A,
                "spatial": ((0.0, 0.0), (1.0, 1.0)),
                "spatial_frame": FRAME_A,
            },
        ),
    )
    observation_id = uid(4_000)
    target_support = tx.ExactSpatialSupport(
        FRAME_B,
        (tx.ExactTransformAtom(-1), tx.ZERO),
        (tx.ONE, tx.ONE),
    )
    outputs = tuple(
        replace(
            item,
            ref=tx.ComponentRef(
                TARGET_SCALE,
                observation_id,
                ordinal,
                uid(4_100 + ordinal),
            ),
            coordinate_frame_id=FRAME_B,
            spatial_support=target_support,
        )
        for ordinal, item in enumerate(source)
    )
    # A = [[-1, 1], [0, 1]] and A^-1 = A.  This exercises both
    # cross-coordinate mixing and negative-coefficient enclosure.
    rows = (
        tx.ExactSparseAffineRow(
            outputs[0].ref,
            (
                tx.ExactSparseTerm(source[0].ref, tx.ExactTransformAtom(-1)),
                tx.ExactSparseTerm(source[1].ref, tx.ONE),
            ),
        ),
        tx.ExactSparseAffineRow(
            outputs[1].ref,
            (tx.ExactSparseTerm(source[1].ref, tx.ONE),),
        ),
    )
    inverse = (
        tx.ExactSparseAffineRow(
            source[0].ref,
            (
                tx.ExactSparseTerm(outputs[0].ref, tx.ExactTransformAtom(-1)),
                tx.ExactSparseTerm(outputs[1].ref, tx.ONE),
            ),
        ),
        tx.ExactSparseAffineRow(
            source[1].ref,
            (tx.ExactSparseTerm(outputs[1].ref, tx.ONE),),
        ),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.COORDINATE_AFFINE,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        outputs,
        (
            _derived_observation(
                base,
                outputs,
                (base.observations[0].observation_id,),
            ),
        ),
        rows,
        (),
        tx.CoordinateAffineCertificate(FRAME_A, FRAME_B, 2, inverse),
    )
    return _sign_authority(base, metadata, contract)


def temporal_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.TEMPORAL_AGGREGATION,
        (
            {
                "values": [1.0],
                "axis": tx.ComponentAxis.TEMPORAL,
                "role": tx.ComponentValueRole.EXTENSIVE,
                "temporal": (0.0, 1.0),
            },
            {
                "values": [2.0],
                "axis": tx.ComponentAxis.TEMPORAL,
                "role": tx.ComponentValueRole.EXTENSIVE,
                "temporal": (1.0, 2.0),
            },
        ),
    )
    output = replace(
        source[0],
        ref=tx.ComponentRef(TARGET_SCALE, uid(4_000), 0, uid(4_100)),
        temporal_support=tx.ExactTemporalSupport(
            CLOCK_ID,
            tx.ZERO,
            tx.ExactTransformAtom(2),
        ),
    )
    group = tx.ExactPartitionGroup(
        tuple(item.ref for item in source),
        (output.ref,),
    )
    row = tx.ExactSparseAffineRow(
        output.ref,
        tuple(tx.ExactSparseTerm(item.ref, tx.ONE) for item in source),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.TEMPORAL_AGGREGATION,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        (output,),
        (
            _derived_observation(
                base,
                (output,),
                tuple(item.observation_id for item in base.observations),
            ),
        ),
        (row,),
        (),
        tx.TemporalAggregationCertificate(tx.ReducerKind.SUM, (group,)),
    )
    return _sign_authority(base, metadata, contract)


def weighted_temporal_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.TEMPORAL_AGGREGATION,
        (
            {
                "values": [1.0],
                "axis": tx.ComponentAxis.TEMPORAL,
                "role": tx.ComponentValueRole.INTENSIVE,
                "temporal": (0.0, 1.0),
            },
            {
                "values": [2.0],
                "axis": tx.ComponentAxis.TEMPORAL,
                "role": tx.ComponentValueRole.INTENSIVE,
                "temporal": (1.0, 2.0),
            },
        ),
    )
    output = replace(
        source[0],
        ref=tx.ComponentRef(TARGET_SCALE, uid(4_000), 0, uid(4_100)),
        temporal_support=tx.ExactTemporalSupport(
            CLOCK_ID,
            tx.ZERO,
            tx.ExactTransformAtom(2),
        ),
    )
    group = tx.ExactPartitionGroup(
        tuple(item.ref for item in source),
        (output.ref,),
    )
    row = tx.ExactSparseAffineRow(
        output.ref,
        (
            tx.ExactSparseTerm(source[0].ref, tx.ExactTransformAtom(1, 4)),
            tx.ExactSparseTerm(source[1].ref, tx.ExactTransformAtom(3, 4)),
        ),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.TEMPORAL_AGGREGATION,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        (output,),
        (
            _derived_observation(
                base,
                (output,),
                tuple(item.observation_id for item in base.observations),
            ),
        ),
        (row,),
        (),
        tx.TemporalAggregationCertificate(
            tx.ReducerKind.WEIGHTED_MEAN,
            (group,),
        ),
    )
    return _sign_authority(base, metadata, contract)


def spatial_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.SPATIAL_AGGREGATION,
        (
            {
                "values": [1.0],
                "axis": tx.ComponentAxis.SPATIAL,
                "role": tx.ComponentValueRole.EXTENSIVE,
                "spatial": ((0.0, 0.0), (1.0, 1.0)),
            },
            {
                "values": [2.0],
                "axis": tx.ComponentAxis.SPATIAL,
                "role": tx.ComponentValueRole.EXTENSIVE,
                "spatial": ((1.0, 0.0), (2.0, 1.0)),
            },
        ),
    )
    target_support = tx.ExactSpatialSupport(
        FRAME_A,
        (tx.ZERO, tx.ZERO),
        (tx.ExactTransformAtom(2), tx.ONE),
    )
    output = replace(
        source[0],
        ref=tx.ComponentRef(TARGET_SCALE, uid(4_000), 0, uid(4_100)),
        spatial_support=target_support,
    )
    group = tx.ExactPartitionGroup(
        tuple(item.ref for item in source),
        (output.ref,),
    )
    row = tx.ExactSparseAffineRow(
        output.ref,
        tuple(tx.ExactSparseTerm(item.ref, tx.ONE) for item in source),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.SPATIAL_AGGREGATION,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        (output,),
        (
            _derived_observation(
                base,
                (output,),
                tuple(item.observation_id for item in base.observations),
            ),
        ),
        (row,),
        (),
        tx.SpatialAggregationCertificate(tx.ReducerKind.SUM, (group,)),
    )
    return _sign_authority(base, metadata, contract)


def sampling_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.SAMPLING_RESOLUTION,
        tuple(
            {
                "values": [float(index + 1)],
                "axis": tx.ComponentAxis.TEMPORAL,
                "role": tx.ComponentValueRole.INTENSIVE,
                "temporal": (float(index), float(index)),
            }
            for index in range(3)
        ),
    )
    selected = source[:2]
    outputs = tuple(
        replace(
            item,
            ref=tx.ComponentRef(
                TARGET_SCALE,
                uid(4_000 + index),
                0,
                uid(4_100 + index),
            ),
        )
        for index, item in enumerate(selected)
    )
    rows = tuple(
        tx.ExactSparseAffineRow(
            output.ref,
            (tx.ExactSparseTerm(input_.ref, tx.ONE),),
        )
        for input_, output in zip(selected, outputs, strict=True)
    )
    observations = tuple(
        _derived_observation(
            base,
            (output,),
            (base.observations[index].observation_id,),
        )
        for index, output in enumerate(outputs)
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.SAMPLING_RESOLUTION,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        outputs,
        observations,
        rows,
        (),
        tx.SamplingResolutionCertificate(
            tx.ComponentAxis.TEMPORAL,
            tuple(item.ref for item in selected),
            (source[2].ref,),
            ((tx.ZERO,), (tx.ONE,)),
            1,
            None,
        ),
    )
    return _sign_authority(base, metadata, contract)


def split_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.EQUIVALENT_SPLIT_MERGE,
        (
            {
                "values": [4.0],
                "axis": tx.ComponentAxis.ENTITY,
                "role": tx.ComponentValueRole.EXTENSIVE,
            },
        ),
    )
    outputs = tuple(
        replace(
            source[0],
            ref=tx.ComponentRef(
                TARGET_SCALE,
                uid(4_000),
                ordinal,
                uid(4_100 + ordinal),
            ),
        )
        for ordinal in range(2)
    )
    half = tx.ExactTransformAtom(1, 2)
    rows = tuple(
        tx.ExactSparseAffineRow(
            output.ref,
            (tx.ExactSparseTerm(source[0].ref, half),),
        )
        for output in outputs
    )
    group = tx.ExactPartitionGroup((source[0].ref,), tuple(item.ref for item in outputs))
    inverse = (
        tx.ExactSparseAffineRow(
            source[0].ref,
            tuple(tx.ExactSparseTerm(item.ref, tx.ONE) for item in outputs),
        ),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.EQUIVALENT_SPLIT_MERGE,
        ROOT_SCALE,
        TARGET_SCALE,
        (source[0].ref,),
        outputs,
        (
            _derived_observation(
                base,
                outputs,
                (base.observations[0].observation_id,),
            ),
        ),
        rows,
        (),
        tx.EquivalentSplitMergeCertificate(
            tx.SplitMergeDirection.SPLIT,
            (group,),
            inverse,
        ),
    )
    return _sign_authority(base, metadata, contract)


def merge_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.EQUIVALENT_SPLIT_MERGE,
        tuple(
            {
                "values": [value],
                "axis": tx.ComponentAxis.ENTITY,
                "role": tx.ComponentValueRole.EXTENSIVE,
            }
            for value in (1.0, 3.0)
        ),
    )
    output = replace(
        source[0],
        ref=tx.ComponentRef(TARGET_SCALE, uid(4_000), 0, uid(4_100)),
    )
    group = tx.ExactPartitionGroup(
        tuple(item.ref for item in source),
        (output.ref,),
    )
    row = tx.ExactSparseAffineRow(
        output.ref,
        tuple(tx.ExactSparseTerm(item.ref, tx.ONE) for item in source),
    )
    half = tx.ExactTransformAtom(1, 2)
    inverse = tuple(
        tx.ExactSparseAffineRow(
            item.ref,
            (tx.ExactSparseTerm(output.ref, half),),
        )
        for item in source
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.EQUIVALENT_SPLIT_MERGE,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        (output,),
        (
            _derived_observation(
                base,
                (output,),
                tuple(item.observation_id for item in base.observations),
            ),
        ),
        (row,),
        (),
        tx.EquivalentSplitMergeCertificate(
            tx.SplitMergeDirection.MERGE,
            (group,),
            inverse,
        ),
    )
    return _sign_authority(base, metadata, contract)


def coarse_authority() -> tx.PublicTransformEvidenceBundleV2:
    base, metadata, source = _base_and_metadata(
        TransformOperation.COARSE_GRAINING,
        (
            {
                "values": [1.0],
                "axis": tx.ComponentAxis.ENTITY,
                "role": tx.ComponentValueRole.EXTENSIVE,
            },
            {
                "values": [2.0],
                "axis": tx.ComponentAxis.ENTITY,
                "role": tx.ComponentValueRole.EXTENSIVE,
            },
        ),
    )
    output = replace(
        source[0],
        ref=tx.ComponentRef(TARGET_SCALE, uid(4_000), 0, uid(4_100)),
        axis=tx.ComponentAxis.COARSE,
    )
    group = tx.ExactPartitionGroup(
        tuple(item.ref for item in source),
        (output.ref,),
    )
    row = tx.ExactSparseAffineRow(
        output.ref,
        tuple(tx.ExactSparseTerm(item.ref, tx.ONE) for item in source),
    )
    source_commutation = tuple(
        tx.ExactSparseAffineRow(
            item.ref,
            (tx.ExactSparseTerm(item.ref, tx.ONE),),
        )
        for item in source
    )
    target_commutation = (
        tx.ExactSparseAffineRow(
            output.ref,
            (tx.ExactSparseTerm(output.ref, tx.ONE),),
        ),
    )
    contract = tx.ExactTransformContract(
        TRANSFORM_ID,
        TransformOperation.COARSE_GRAINING,
        ROOT_SCALE,
        TARGET_SCALE,
        tuple(item.ref for item in source),
        (output,),
        (
            _derived_observation(
                base,
                (output,),
                tuple(item.observation_id for item in base.observations),
            ),
        ),
        (row,),
        (),
        tx.CoarseGrainingCertificate(
            tx.ReducerKind.SUM,
            (group,),
            (uid(4_200),),
            source_commutation,
            target_commutation,
        ),
    )
    return _sign_authority(base, metadata, contract)


def two_step_authority() -> tx.PublicTransformEvidenceBundleV2:
    first_authority = identity_authority()
    first = first_authority.transform_contracts[0]
    final_scale = uid(102)
    second_transform = uid(201)
    mapping = first_authority.base_bundle.to_mapping()
    mapping["aggregation_graph"] = {
        "scale_ids": [ROOT_SCALE, TARGET_SCALE, final_scale],
        "root_scale_ids": [ROOT_SCALE],
        "edges": [
            {
                "source_scale_id": ROOT_SCALE,
                "target_scale_id": TARGET_SCALE,
                "transform_id": TRANSFORM_ID,
            },
            {
                "source_scale_id": TARGET_SCALE,
                "target_scale_id": final_scale,
                "transform_id": second_transform,
            },
        ],
    }
    mapping["transform_catalog"] = [
        {
            "transform_id": TRANSFORM_ID,
            "operation": "identity",
            "parameters": [],
        },
        {
            "transform_id": second_transform,
            "operation": "unit_conversion",
            "parameters": [],
        },
    ]
    base = PublicEvidenceBundle.from_mapping(mapping)
    output_observation_id = uid(6_000)
    outputs = tuple(
        replace(
            descriptor,
            ref=tx.ComponentRef(
                final_scale,
                output_observation_id,
                ordinal,
                uid(6_100 + ordinal),
            ),
            unit_id=UNIT_B,
        )
        for ordinal, descriptor in enumerate(first.output_components)
    )
    factor = tx.ExactTransformAtom(2)
    second = tx.ExactTransformContract(
        second_transform,
        TransformOperation.UNIT_CONVERSION,
        TARGET_SCALE,
        final_scale,
        tuple(item.ref for item in first.output_components),
        outputs,
        (
            tx.DerivedObservationDescriptor(
                scale_id=final_scale,
                observation_id=output_observation_id,
                source_channel_id=CHANNEL_ID,
                entity_ids=(ENTITY_ID,),
                role_candidate_ids=(ROLE_ID,),
                quantity_id=QUANTITY_ID,
                unit_id=UNIT_B,
                si_exponents=(0,) * 7,
                temporal_support=None,
                spatial_support=None,
                provenance_sha256="0" * 64,
                source_observation_ids=(base.observations[0].observation_id,),
                value_kind=tx.ComponentValueKind.NUMERIC_INTERVAL,
                component_refs=tuple(item.ref for item in outputs),
            ),
        ),
        tuple(
            tx.ExactSparseAffineRow(
                output.ref,
                (tx.ExactSparseTerm(input_.ref, factor),),
            )
            for input_, output in zip(first.output_components, outputs, strict=True)
        ),
        (),
        tx.UnitConversionCertificate(
            UNIT_A,
            UNIT_B,
            factor,
            tx.ExactTransformAtom(1, 2),
        ),
    )
    provisional = tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        first_authority.observation_metadata,
        (first, second),
    )
    receipt = compile_bundle_uncertainty(base)
    states, _ = tx._build_root_states(provisional, receipt)
    first_semantics = first.semantics_id
    first_target, _, error = tx._apply_contract(
        first,
        states[ROOT_SCALE],
        tx._KernelBudget(tx._DEFAULT_POLICY),
        contract_semantics_id=first_semantics,
        ordered_transform_path_ids=(TRANSFORM_ID,),
        ordered_contract_semantics_ids=(first_semantics,),
    )
    assert error is None and first_target is not None
    second_semantics = second.semantics_id
    source_roots = {}
    for state in first_target.values():
        observation = state.observation
        source_roots.setdefault(
            observation.observation_id,
            (observation.descriptor_id, observation),
        )
    inputs_by_output = tx._output_input_refs(second)
    provisional_observation = second.output_observations[0]
    contributing = tuple(
        first_target[input_ref]
        for output_ref in provisional_observation.component_refs
        for input_ref in inputs_by_output[output_ref]
    )
    signed_observation = replace(
        provisional_observation,
        provenance_sha256=tx._expected_derived_provenance(
            provisional_observation,
            contributing,
            source_roots,
            second_semantics,
            (TRANSFORM_ID, second_transform),
            (first_semantics, second_semantics),
        ),
    )
    signed_second = replace(second, output_observations=(signed_observation,))
    assert signed_second.semantics_id == second_semantics
    return replace(provisional, transform_contracts=(first, signed_second))


def test_identity_golden_and_authoritative_signature():
    authority = identity_authority()
    result = tx.run_exact_transform_semantics(authority)
    assert result.disposition is tx.TransformCompilationDisposition.COMPLETE
    assert result.failures == ()
    assert len(result.observations) == 2
    assert len(result.components) == 4
    target = [
        item
        for item in result.components
        if item.descriptor.ref.scale_id == TARGET_SCALE
    ]
    assert [
        (item.numeric_interval.lower_fraction, item.numeric_interval.upper_fraction)
        for item in target
    ] == [(Fraction(1), Fraction(1)), (Fraction(-2), Fraction(-2))]
    assert str(inspect.signature(tx.run_exact_transform_semantics)) == (
        "(authority: 'PublicTransformEvidenceBundleV2') -> "
        "'ExactTransformCompilation | ExactTransformPreflightRejection'"
    )


@pytest.mark.parametrize(
    "factory,expected",
    [
        (unit_authority, (Fraction(2), Fraction(-4))),
        (coordinate_authority, (Fraction(1, 2), Fraction(3, 4))),
        (temporal_authority, (Fraction(3),)),
        (spatial_authority, (Fraction(3),)),
        (sampling_authority, (Fraction(1), Fraction(2))),
        (split_authority, (Fraction(2), Fraction(2))),
        (merge_authority, (Fraction(4),)),
        (coarse_authority, (Fraction(3),)),
    ],
)
def test_nonidentity_operation_golden(factory, expected):
    result = tx.run_exact_transform_semantics(factory())
    assert result.disposition is tx.TransformCompilationDisposition.COMPLETE
    target = [
        item
        for item in result.components
        if item.descriptor.ref.scale_id == TARGET_SCALE
    ]
    assert tuple(item.numeric_interval.lower_fraction for item in target) == expected
    assert all(
        item.numeric_interval.lower_fraction == item.numeric_interval.upper_fraction
        for item in target
    )


def test_weighted_mean_uses_declared_exact_quarter_weights():
    result = tx.run_exact_transform_semantics(weighted_temporal_authority())
    assert result.disposition is tx.TransformCompilationDisposition.COMPLETE
    target = next(
        item
        for item in result.components
        if item.descriptor.ref.scale_id == TARGET_SCALE
    )
    assert (
        target.numeric_interval.lower_fraction,
        target.numeric_interval.upper_fraction,
    ) == (Fraction(7, 4), Fraction(7, 4))


def _resign(
    authority: tx.PublicTransformEvidenceBundleV2,
    contract: tx.ExactTransformContract,
) -> tx.PublicTransformEvidenceBundleV2:
    unsigned = replace(
        contract,
        output_observations=tuple(
            replace(observation, provenance_sha256="0" * 64)
            for observation in contract.output_observations
        ),
    )
    return _sign_authority(
        authority.base_bundle,
        authority.observation_metadata,
        unsigned,
    )


def _assert_atomic_failure(result, error_code: str):
    assert result.disposition is tx.TransformCompilationDisposition.ABSTAIN
    assert result.observations == ()
    assert result.components == ()
    assert result.failures[0].error_code == error_code


def test_wrapper_root_spoof_and_provenance_mutation_fail_closed():
    authority = identity_authority()

    class WrapperSpoof(tx.PublicTransformEvidenceBundleV2):
        @property
        def content_id(self):
            return "caller_chosen_root"

    spoof = WrapperSpoof(
        authority.schema_version,
        authority.base_bundle,
        authority.observation_metadata,
        authority.transform_contracts,
    )
    with pytest.raises(TypeError, match="PublicTransformEvidenceBundleV2"):
        tx.run_exact_transform_semantics(spoof)

    contract = authority.transform_contracts[0]
    changed_observation = replace(
        contract.output_observations[0],
        provenance_sha256="f" * 64,
    )
    mutated = replace(
        authority,
        transform_contracts=(
            replace(contract, output_observations=(changed_observation,)),
        ),
    )
    assert mutated.content_id != authority.content_id
    result = tx.run_exact_transform_semantics(mutated)
    _assert_atomic_failure(result, "output_observation_provenance_mismatch")


def test_legacy_parameter_conflict_and_wrong_certificate_reject_before_hash():
    authority = unit_authority()
    spec = authority.base_bundle.transform_catalog[0]
    conflicted_base = replace(
        authority.base_bundle,
        transform_catalog=(replace(spec, parameters=(1.0,)),),
    )
    conflicted = replace(authority, base_bundle=conflicted_base)
    result = tx.run_exact_transform_semantics(conflicted)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "legacy_nonidentity_parameters_forbidden"
    assert result.wrapper_content_id is None

    wrong = replace(
        authority,
        transform_contracts=(
            replace(
                authority.transform_contracts[0],
                certificate=tx.IdentityTransformCertificate(),
            ),
        ),
    )
    result = tx.run_exact_transform_semantics(wrong)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "wrong_certificate_for_operation"


def test_sum_is_not_mean_and_temporal_spatial_certificate_swap_rejects():
    authority = temporal_authority()
    contract = authority.transform_contracts[0]
    certificate = contract.certificate
    assert isinstance(certificate, tx.TemporalAggregationCertificate)
    mean_contract = replace(
        contract,
        certificate=replace(certificate, reducer=tx.ReducerKind.WEIGHTED_MEAN),
    )
    result = tx.run_exact_transform_semantics(_resign(authority, mean_contract))
    _assert_atomic_failure(result, "aggregation_reducer_kernel_mismatch")

    spatial_certificate = spatial_authority().transform_contracts[0].certificate
    swapped = replace(
        authority,
        transform_contracts=(replace(contract, certificate=spatial_certificate),),
    )
    result = tx.run_exact_transform_semantics(swapped)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "wrong_certificate_for_operation"


def test_sampling_boundary_split_direction_and_malicious_inverse_reject_atomically():
    authority = sampling_authority()
    contract = authority.transform_contracts[0]
    certificate = contract.certificate
    assert isinstance(certificate, tx.SamplingResolutionCertificate)
    wrong_grid = replace(
        certificate,
        grid_points=((tx.ZERO,), (tx.ExactTransformAtom(2),)),
    )
    result = tx.run_exact_transform_semantics(
        _resign(authority, replace(contract, certificate=wrong_grid))
    )
    _assert_atomic_failure(result, "sampling_temporal_grid_not_source_point")

    authority = split_authority()
    contract = authority.transform_contracts[0]
    certificate = contract.certificate
    assert isinstance(certificate, tx.EquivalentSplitMergeCertificate)
    wrong_direction = replace(
        certificate,
        direction=tx.SplitMergeDirection.MERGE,
    )
    result = tx.run_exact_transform_semantics(
        _resign(authority, replace(contract, certificate=wrong_direction))
    )
    _assert_atomic_failure(result, "merge_group_shape_invalid")

    bad_inverse = (
        tx.ExactSparseAffineRow(
            certificate.inverse_rows[0].output_ref,
            (
                tx.ExactSparseTerm(
                    certificate.inverse_rows[0].terms[0].input_ref,
                    tx.ExactTransformAtom(3),
                ),
                tx.ExactSparseTerm(
                    certificate.inverse_rows[0].terms[1].input_ref,
                    tx.ExactTransformAtom(-1),
                ),
            ),
        ),
    )
    malicious = replace(certificate, inverse_rows=bad_inverse)
    result = tx.run_exact_transform_semantics(
        _resign(authority, replace(contract, certificate=malicious))
    )
    _assert_atomic_failure(result, "split_inverse_is_not_exact_sum")


def test_singular_coordinate_and_resource_limit_are_atomic():
    authority = coordinate_authority()
    contract = authority.transform_contracts[0]
    singular_rows = (
        replace(
            contract.kernel_rows[0],
            terms=(tx.ExactSparseTerm(contract.input_components[1], tx.ONE),),
        ),
        contract.kernel_rows[1],
    )
    result = tx.run_exact_transform_semantics(
        _resign(authority, replace(contract, kernel_rows=singular_rows))
    )
    _assert_atomic_failure(
        result,
        "coordinate_affine_is_singular_or_inverse_mismatch",
    )

    authority = identity_authority()
    contract = authority.transform_contracts[0]
    huge = tx.ExactTransformAtom(1 << 5_000)
    row = replace(
        contract.kernel_rows[0],
        terms=(tx.ExactSparseTerm(contract.input_components[0], huge),),
    )
    limited = replace(
        authority,
        transform_contracts=(
            replace(contract, kernel_rows=(row, *contract.kernel_rows[1:])),
        ),
    )
    result = tx.run_exact_transform_semantics(limited)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "RESOURCE_LIMIT:authority_integer_bit_length"
    assert result.components == ()


def test_complete_result_tree_contains_no_float():
    result = tx.run_exact_transform_semantics(coordinate_authority())
    stack = [result]
    while stack:
        value = stack.pop()
        assert type(value) is not float
        if type(value) is tuple:
            stack.extend(value)
        elif is_dataclass(value):
            stack.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, Enum):
            continue


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("boolean", "boolean_observation_must_be_dimensionless"),
        ("missing", "missing_observation_must_be_dimensionless"),
    ],
)
def test_dimensioned_boolean_and_missing_fail_without_exception(kind, expected):
    authority = identity_authority()
    mapping = authority.base_bundle.to_mapping()
    observation = mapping["observations"][0]
    observation["unit_dimension"] = {"si_exponents": [1, 0, 0, 0, 0, 0, 0]}
    observation["uncertainty"] = {"model": "not_applicable", "radius": []}
    metadata = replace(
        authority.observation_metadata[0],
        component_ids=(authority.observation_metadata[0].component_ids[0],),
        axis=(
            tx.ComponentAxis.CONTROL
            if kind == "boolean"
            else tx.ComponentAxis.SCALAR
        ),
        value_role=(
            tx.ComponentValueRole.BOOLEAN_CONTROL
            if kind == "boolean"
            else tx.ComponentValueRole.MISSING
        ),
        unit_id=None,
    )
    if kind == "boolean":
        observation["value"] = {"kind": "boolean", "value": True}
    else:
        observation["value"] = None
        observation["missingness"] = "missing"
        mapping["missingness_mask"] = [observation["observation_id"]]
    mutated = replace(
        authority,
        base_bundle=PublicEvidenceBundle.from_mapping(mapping),
        observation_metadata=(metadata,),
    )
    result = tx.run_exact_transform_semantics(mutated)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == expected


def test_forest_rejects_two_roots_merging_into_one_target():
    second_root = uid(102)
    second_transform = uid(201)
    base, metadata, _ = _base_and_metadata(
        TransformOperation.IDENTITY,
        ({"values": [1.0]}, {"values": [2.0]}),
    )
    mapping = base.to_mapping()
    mapping["aggregation_graph"] = {
        "scale_ids": [ROOT_SCALE, second_root, TARGET_SCALE],
        "root_scale_ids": [ROOT_SCALE, second_root],
        "edges": [
            {
                "source_scale_id": ROOT_SCALE,
                "target_scale_id": TARGET_SCALE,
                "transform_id": TRANSFORM_ID,
            },
            {
                "source_scale_id": second_root,
                "target_scale_id": TARGET_SCALE,
                "transform_id": second_transform,
            },
        ],
    }
    mapping["transform_catalog"] = [
        {
            "transform_id": TRANSFORM_ID,
            "operation": "identity",
            "parameters": [],
        },
        {
            "transform_id": second_transform,
            "operation": "identity",
            "parameters": [],
        },
    ]
    base = PublicEvidenceBundle.from_mapping(mapping)
    metadata = (metadata[0], replace(metadata[1], scale_id=second_root))
    sources = tuple(
        tx._root_descriptor(observation, item, 0)
        for observation, item in zip(base.observations, metadata, strict=True)
    )
    contracts = []
    for index, (transform_id, source_scale, source) in enumerate(
        (
            (TRANSFORM_ID, ROOT_SCALE, sources[0]),
            (second_transform, second_root, sources[1]),
        )
    ):
        output = replace(
            source,
            ref=tx.ComponentRef(
                TARGET_SCALE,
                uid(5_000 + index),
                0,
                uid(5_100 + index),
            ),
        )
        contracts.append(
            tx.ExactTransformContract(
                transform_id,
                TransformOperation.IDENTITY,
                source_scale,
                TARGET_SCALE,
                (source.ref,),
                (output,),
                (
                    _derived_observation(
                        base,
                        (output,),
                        (base.observations[index].observation_id,),
                    ),
                ),
                (
                    tx.ExactSparseAffineRow(
                        output.ref,
                        (tx.ExactSparseTerm(source.ref, tx.ONE),),
                    ),
                ),
                (),
                tx.IdentityTransformCertificate(),
            )
        )
    authority = tx.PublicTransformEvidenceBundleV2(
        tx.PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        base,
        metadata,
        tuple(contracts),
    )
    result = tx.run_exact_transform_semantics(authority)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "forest_nonroot_requires_exactly_one_parent"


def test_two_step_receipts_bind_ab_order_without_path_enumeration():
    authority = two_step_authority()
    result = tx.run_exact_transform_semantics(authority)
    assert result.disposition is tx.TransformCompilationDisposition.COMPLETE
    final_scale = uid(102)
    final_cells = [
        item
        for item in result.components
        if item.descriptor.ref.scale_id == final_scale
    ]
    assert tuple(
        item.numeric_interval.lower_fraction for item in final_cells
    ) == (Fraction(2), Fraction(-4))
    assert {item.ordered_transform_path_ids for item in final_cells} == {
        (TRANSFORM_ID, uid(201))
    }
    assert all(
        len(item.ordered_contract_semantics_ids) == 2 for item in final_cells
    )


def test_sampling_v1_rejects_nonseries_discard_and_repeated_point():
    authority = sampling_authority()
    contract = authority.transform_contracts[0]
    certificate = contract.certificate
    assert isinstance(certificate, tx.SamplingResolutionCertificate)
    source_observation = authority.base_bundle.observations[2]
    changed = replace(source_observation, quantity_id=uid(999))
    base = replace(
        authority.base_bundle,
        quantity_ids=tuple(sorted((*authority.base_bundle.quantity_ids, uid(999)))),
        observations=tuple(
            changed if item.observation_id == changed.observation_id else item
            for item in authority.base_bundle.observations
        ),
    )
    mutated = replace(authority, base_bundle=base)
    result = tx.run_exact_transform_semantics(mutated)
    # Root observation identity is content-rooted; the first semantic check is
    # intentionally the single-series boundary, not a silent arbitrary drop.
    _assert_atomic_failure(result, "sampling_v1_requires_one_semantic_series")


def test_negative_coefficient_encloses_non_degenerate_uncertainty_interval():
    authority = coordinate_authority()
    mapping = authority.base_bundle.to_mapping()
    mapping["observations"][0]["uncertainty"] = {
        "model": "absolute_bound",
        "radius": [0.25, 0.125],
    }
    changed = replace(
        authority,
        base_bundle=PublicEvidenceBundle.from_mapping(mapping),
    )
    result = tx.run_exact_transform_semantics(
        _resign(changed, changed.transform_contracts[0])
    )
    assert result.disposition is tx.TransformCompilationDisposition.COMPLETE
    target = [
        item
        for item in result.components
        if item.descriptor.ref.scale_id == TARGET_SCALE
    ]
    assert (
        target[0].numeric_interval.lower_fraction,
        target[0].numeric_interval.upper_fraction,
    ) == (Fraction(1, 8), Fraction(7, 8))


@pytest.mark.parametrize(
    "first_upper,second_lower",
    [
        (0.75, 1.0),
        (1.5, 1.0),
    ],
)
def test_spatial_gap_and_overlap_are_not_exact_partitions(
    first_upper,
    second_lower,
):
    authority = spatial_authority()
    mapping = authority.base_bundle.to_mapping()
    mapping["observations"][0]["spatial_support"]["upper"][0] = first_upper
    mapping["observations"][1]["spatial_support"]["lower"][0] = second_lower
    changed = replace(
        authority,
        base_bundle=PublicEvidenceBundle.from_mapping(mapping),
    )
    result = tx.run_exact_transform_semantics(
        _resign(changed, changed.transform_contracts[0])
    )
    _assert_atomic_failure(result, "spatial_aggregation_not_exact_partition")


def test_coarse_commutation_rows_are_checked_as_exact_matrix_equality():
    authority = coarse_authority()
    contract = authority.transform_contracts[0]
    certificate = contract.certificate
    assert isinstance(certificate, tx.CoarseGrainingCertificate)
    bad_target = (
        replace(
            certificate.target_commutation_rows[0],
            terms=(
                tx.ExactSparseTerm(
                    certificate.target_commutation_rows[0].terms[0].input_ref,
                    tx.ExactTransformAtom(2),
                ),
            ),
        ),
    )
    changed_contract = replace(
        contract,
        certificate=replace(
            certificate,
            target_commutation_rows=bad_target,
        ),
    )
    result = tx.run_exact_transform_semantics(
        _resign(authority, changed_contract)
    )
    _assert_atomic_failure(
        result,
        "coarse_commutation_matrix_equality_failed",
    )


def test_metadata_and_row_count_caps_short_circuit_before_wrapper_hash():
    authority = identity_authority()
    oversized_metadata = replace(
        authority,
        observation_metadata=(authority.observation_metadata[0],) * 4_097,
    )
    result = tx.run_exact_transform_semantics(oversized_metadata)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "RESOURCE_LIMIT:metadata_count"
    assert result.wrapper_content_id is None

    contract = authority.transform_contracts[0]
    oversized_rows = replace(
        authority,
        transform_contracts=(
            replace(
                contract,
                kernel_rows=(contract.kernel_rows[0],) * 65_537,
            ),
        ),
    )
    result = tx.run_exact_transform_semantics(oversized_rows)
    assert isinstance(result, tx.ExactTransformPreflightRejection)
    assert result.reason == "RESOURCE_LIMIT:row_count"
    assert result.wrapper_content_id is None

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import runpy

import pytest

import hegel_machine.phase2b_exact_transform_core_v1 as core


F = Fraction
SOURCE = core.CoreScaleRefV1("source")
TARGET = core.CoreScaleRefV1("target")
CLOCK = core.CoreClockBindingV1("clock:domain-epoch")
FRAME_A = core.CoreFrameBindingV1("frame:a", "right-handed")
FRAME_B = core.CoreFrameBindingV1("frame:b", "right-handed")
UNIT_A = core.CoreUnitBindingV1("unit:a", F(1), F(0))
UNIT_B = core.CoreUnitBindingV1("unit:b", F(1, 100), F(0))
UNIT_OFFSET_SOURCE = core.CoreUnitBindingV1("unit:offset-source", F(2), F(3))
UNIT_OFFSET_TARGET = core.CoreUnitBindingV1("unit:offset-target", F(1), F(1))
NUMERIC_QUANTITY_A = core.CoreQuantityMetadataV1(
    "quantity:length", (1, 0, 0, 0, 0, 0, 0), UNIT_A
)
NUMERIC_QUANTITY_B = core.CoreQuantityMetadataV1(
    "quantity:length", (1, 0, 0, 0, 0, 0, 0), UNIT_B
)
NUMERIC_QUANTITY_OFFSET_SOURCE = core.CoreQuantityMetadataV1(
    "quantity:length", (1, 0, 0, 0, 0, 0, 0), UNIT_OFFSET_SOURCE
)
NUMERIC_QUANTITY_OFFSET_TARGET = core.CoreQuantityMetadataV1(
    "quantity:length", (1, 0, 0, 0, 0, 0, 0), UNIT_OFFSET_TARGET
)
DIMENSIONLESS = core.CoreQuantityMetadataV1(
    "quantity:dimensionless", (0, 0, 0, 0, 0, 0, 0),
    core.CoreUnitBindingV1("unit:one", F(1), F(0)),
)
BOOLEAN_QUANTITY = core.CoreQuantityMetadataV1(
    "quantity:boolean", (0, 0, 0, 0, 0, 0, 0), None
)
MISSING_QUANTITY = core.CoreQuantityMetadataV1(
    "quantity:missing", (0, 0, 0, 0, 0, 0, 0), None
)


def ref(
    scale: core.CoreScaleRefV1,
    observation: str,
    index: int = 0,
) -> core.CoreComponentRefV1:
    return core.CoreComponentRefV1(
        scale, core.CoreObservationRefV1(observation), index
    )


def metadata(
    component_ref: core.CoreComponentRefV1,
    *,
    label: str = "value",
    quantity: core.CoreQuantityMetadataV1 = DIMENSIONLESS,
    axis: core.CoreComponentAxisV1 = core.CoreComponentAxisV1.SCALAR,
    role: core.CoreComponentValueRoleV1 = (
        core.CoreComponentValueRoleV1.INTENSIVE
    ),
    temporal: core.CoreTemporalSupportV1 | None = None,
    spatial: core.CoreSpatialSupportV1 | None = None,
) -> core.CoreComponentMetadataV1:
    return core.CoreComponentMetadataV1(
        ref=component_ref,
        component_label=label,
        entity_keys=("entity:subject",),
        role_keys=("role:measured",),
        quantity=quantity,
        axis=axis,
        value_role=role,
        temporal_support=temporal,
        spatial_support=spatial,
    )


def numeric(
    item: core.CoreComponentMetadataV1,
    value: Fraction,
    radius: Fraction,
    *,
    uncertainty: core.CoreUncertaintyModelV1 = (
        core.CoreUncertaintyModelV1.ABSOLUTE_BOUND
    ),
) -> core.CoreComponentValueV1:
    return core.CoreComponentValueV1(
        metadata=item,
        value_kind=core.CoreComponentValueKindV1.NUMERIC,
        numeric_value=value,
        uncertainty_model=uncertainty,
        absolute_radius=radius,
    )


def boolean(
    item: core.CoreComponentMetadataV1, value: bool
) -> core.CoreComponentValueV1:
    return core.CoreComponentValueV1(
        metadata=item,
        value_kind=core.CoreComponentValueKindV1.BOOLEAN,
        boolean_value=value,
    )


def missing(
    item: core.CoreComponentMetadataV1,
) -> core.CoreComponentValueV1:
    return core.CoreComponentValueV1(
        metadata=item,
        value_kind=core.CoreComponentValueKindV1.MISSING,
    )


def row(
    output_ref: core.CoreComponentRefV1,
    terms: tuple[tuple[core.CoreComponentRefV1, Fraction], ...],
    offset: Fraction = F(0),
) -> core.CoreExactRowV1:
    return core.CoreExactRowV1(
        output_ref=output_ref,
        terms=tuple(
            core.CoreExactTermV1(input_ref, coefficient)
            for input_ref, coefficient in sorted(terms)
        ),
        offset=offset,
    )


def group(
    inputs: tuple[core.CoreComponentRefV1, ...],
    outputs: tuple[core.CoreComponentRefV1, ...],
    label: str,
) -> core.CoreAggregateGroupV1:
    return core.CoreAggregateGroupV1(
        tuple(sorted(inputs)), tuple(sorted(outputs)), label
    )


def profile(
    *,
    operation: core.CoreTransformOperationV1,
    sources: tuple[core.CoreComponentValueV1, ...],
    outputs: tuple[core.CoreComponentMetadataV1, ...],
    rows: tuple[core.CoreExactRowV1, ...],
    mappings: tuple[core.CoreDiscreteMappingV1, ...] = (),
    certificate: core.CoreOperationCertificateV1,
) -> core.ExactTransformCoreProfileV1:
    return core.ExactTransformCoreProfileV1(
        source_components=tuple(sorted(sources, key=lambda item: item.ref)),
        contract=core.CoreExactTransformContractV1(
            operation=operation,
            source_scale_ref=SOURCE,
            target_scale_ref=TARGET,
            input_refs=tuple(sorted(item.ref for item in sources)),
            output_metadata=tuple(sorted(outputs, key=lambda item: item.ref)),
            kernel_rows=tuple(sorted(rows, key=lambda item: item.output_ref)),
            discrete_mappings=tuple(sorted(mappings)),
            certificate=certificate,
        ),
    )


def complete(
    item: core.ExactTransformCoreProfileV1,
) -> tuple[core.CoreComponentValueV1, ...]:
    core.validate_exact_transform_core_v1(item)
    result = core.run_exact_transform_core_v1(item)
    assert result.disposition is core.CoreTransformDispositionV1.COMPLETE
    assert result.failure is None
    assert result.output_components
    return result.output_components


def identity_profile() -> core.ExactTransformCoreProfileV1:
    number_ref = ref(SOURCE, "a-number")
    boolean_ref = ref(SOURCE, "b-boolean")
    missing_ref = ref(SOURCE, "c-missing")
    number_meta = metadata(number_ref, quantity=DIMENSIONLESS)
    boolean_meta = metadata(
        boolean_ref,
        label="flag",
        quantity=BOOLEAN_QUANTITY,
        axis=core.CoreComponentAxisV1.CONTROL,
        role=core.CoreComponentValueRoleV1.BOOLEAN_CONTROL,
    )
    missing_meta = metadata(
        missing_ref,
        label="missing",
        quantity=MISSING_QUANTITY,
        role=core.CoreComponentValueRoleV1.MISSING,
    )
    target_number = replace(number_meta, ref=ref(TARGET, "a-number"))
    target_boolean = replace(boolean_meta, ref=ref(TARGET, "b-boolean"))
    target_missing = replace(missing_meta, ref=ref(TARGET, "c-missing"))
    return profile(
        operation=core.CoreTransformOperationV1.IDENTITY,
        sources=(
            numeric(number_meta, F(3, 2), F(1, 10)),
            boolean(boolean_meta, True),
            missing(missing_meta),
        ),
        outputs=(target_number, target_boolean, target_missing),
        rows=(row(target_number.ref, ((number_ref, F(1)),)),),
        mappings=(
            core.CoreDiscreteMappingV1(boolean_ref, target_boolean.ref),
            core.CoreDiscreteMappingV1(missing_ref, target_missing.ref),
        ),
        certificate=core.CoreIdentityCertificateV1(),
    )


def unit_profile() -> core.ExactTransformCoreProfileV1:
    source_ref = ref(SOURCE, "unit")
    target_ref = ref(TARGET, "unit")
    source_meta = metadata(source_ref, quantity=NUMERIC_QUANTITY_A)
    target_meta = metadata(target_ref, quantity=NUMERIC_QUANTITY_B)
    return profile(
        operation=core.CoreTransformOperationV1.UNIT_CONVERSION,
        sources=(numeric(source_meta, F(2), F(1, 10)),),
        outputs=(target_meta,),
        rows=(row(target_ref, ((source_ref, F(100)),)),),
        certificate=core.CoreUnitConversionCertificateV1(F(100), F(1, 100)),
    )


def offset_unit_profile() -> core.ExactTransformCoreProfileV1:
    source_ref = ref(SOURCE, "offset-unit")
    target_ref = ref(TARGET, "offset-unit")
    source_meta = metadata(
        source_ref, quantity=NUMERIC_QUANTITY_OFFSET_SOURCE
    )
    target_meta = metadata(
        target_ref, quantity=NUMERIC_QUANTITY_OFFSET_TARGET
    )
    # SI equality: 2*x + 3 = 1*y + 1, hence y = 2*x + 2.
    return profile(
        operation=core.CoreTransformOperationV1.UNIT_CONVERSION,
        sources=(numeric(source_meta, F(4), F(1, 2)),),
        outputs=(target_meta,),
        rows=(row(target_ref, ((source_ref, F(2)),), F(2)),),
        certificate=core.CoreUnitConversionCertificateV1(F(2), F(1, 2)),
    )


def coordinate_profile() -> core.ExactTransformCoreProfileV1:
    # Deliberately oppose observation-key order to component-index order.  The
    # coordinate semantics must use component_index, never ref lexicography.
    source_refs = (
        ref(SOURCE, "z-coordinate", 0),
        ref(SOURCE, "a-coordinate", 1),
    )
    target_refs = (ref(TARGET, "coordinate", 0), ref(TARGET, "coordinate", 1))
    source_support = core.CoreSpatialSupportV1(
        FRAME_A, (F(0), F(0)), (F(1), F(2))
    )
    target_support = core.CoreSpatialSupportV1(
        FRAME_B, (F(1), F(0)), (F(5), F(2))
    )
    source_metadata = tuple(
        metadata(
            item,
            label=f"coordinate-{index}",
            quantity=NUMERIC_QUANTITY_A,
            axis=core.CoreComponentAxisV1.COORDINATE,
            role=core.CoreComponentValueRoleV1.COORDINATE,
            spatial=source_support,
        )
        for index, item in enumerate(source_refs)
    )
    target_metadata = tuple(
        metadata(
            item,
            label=f"coordinate-{index}",
            quantity=NUMERIC_QUANTITY_A,
            axis=core.CoreComponentAxisV1.COORDINATE,
            role=core.CoreComponentValueRoleV1.COORDINATE,
            spatial=target_support,
        )
        for index, item in enumerate(target_refs)
    )
    forward = (
        row(
            target_refs[0],
            ((source_refs[0], F(2)), (source_refs[1], F(1))),
            F(1),
        ),
        row(target_refs[1], ((source_refs[1], F(1)),)),
    )
    inverse = tuple(sorted((
        row(
            source_refs[0],
            ((target_refs[0], F(1, 2)), (target_refs[1], F(-1, 2))),
            F(-1, 2),
        ),
        row(source_refs[1], ((target_refs[1], F(1)),)),
    ), key=lambda item: item.output_ref))
    return profile(
        operation=core.CoreTransformOperationV1.COORDINATE_AFFINE,
        sources=(
            numeric(source_metadata[0], F(1, 2), F(1, 10)),
            numeric(source_metadata[1], F(1), F(1, 5)),
        ),
        outputs=target_metadata,
        rows=forward,
        certificate=core.CoreCoordinateAffineCertificateV1(
            FRAME_A, FRAME_B, 2, inverse
        ),
    )


def temporal_profile() -> core.ExactTransformCoreProfileV1:
    source_refs = (ref(SOURCE, "temporal-a"), ref(SOURCE, "temporal-b"))
    target_ref = ref(TARGET, "temporal")
    source_metadata = (
        metadata(
            source_refs[0],
            label="flow",
            axis=core.CoreComponentAxisV1.TEMPORAL,
            role=core.CoreComponentValueRoleV1.EXTENSIVE,
            temporal=core.CoreTemporalSupportV1(CLOCK, F(0), F(1)),
        ),
        metadata(
            source_refs[1],
            label="flow",
            axis=core.CoreComponentAxisV1.TEMPORAL,
            role=core.CoreComponentValueRoleV1.EXTENSIVE,
            temporal=core.CoreTemporalSupportV1(CLOCK, F(1), F(2)),
        ),
    )
    target_metadata = metadata(
        target_ref,
        label="flow",
        axis=core.CoreComponentAxisV1.TEMPORAL,
        role=core.CoreComponentValueRoleV1.EXTENSIVE,
        temporal=core.CoreTemporalSupportV1(CLOCK, F(0), F(2)),
    )
    grouping = group(source_refs, (target_ref,), "temporal-bin")
    return profile(
        operation=core.CoreTransformOperationV1.TEMPORAL_AGGREGATION,
        sources=(
            numeric(source_metadata[0], F(1), F(1, 10)),
            numeric(source_metadata[1], F(2), F(1, 5)),
        ),
        outputs=(target_metadata,),
        rows=(row(target_ref, ((source_refs[0], F(1)), (source_refs[1], F(1)))),),
        certificate=core.CoreTemporalAggregationCertificateV1(
            core.CoreReducerV1.SUM, (grouping,)
        ),
    )


def spatial_profile() -> core.ExactTransformCoreProfileV1:
    source_refs = (ref(SOURCE, "spatial-a"), ref(SOURCE, "spatial-b"))
    target_ref = ref(TARGET, "spatial")
    source_metadata = (
        metadata(
            source_refs[0],
            label="density",
            axis=core.CoreComponentAxisV1.SPATIAL,
            spatial=core.CoreSpatialSupportV1(
                FRAME_A, (F(0), F(0)), (F(1), F(1))
            ),
        ),
        metadata(
            source_refs[1],
            label="density",
            axis=core.CoreComponentAxisV1.SPATIAL,
            spatial=core.CoreSpatialSupportV1(
                FRAME_A, (F(1), F(0)), (F(2), F(1))
            ),
        ),
    )
    target_metadata = metadata(
        target_ref,
        label="density",
        axis=core.CoreComponentAxisV1.SPATIAL,
        spatial=core.CoreSpatialSupportV1(
            FRAME_A, (F(0), F(0)), (F(2), F(1))
        ),
    )
    grouping = group(source_refs, (target_ref,), "spatial-bin")
    return profile(
        operation=core.CoreTransformOperationV1.SPATIAL_AGGREGATION,
        sources=(
            numeric(source_metadata[0], F(2), F(1, 10)),
            numeric(source_metadata[1], F(4), F(1, 5)),
        ),
        outputs=(target_metadata,),
        rows=(
            row(
                target_ref,
                ((source_refs[0], F(1, 2)), (source_refs[1], F(1, 2))),
            ),
        ),
        certificate=core.CoreSpatialAggregationCertificateV1(
            core.CoreReducerV1.WEIGHTED_MEAN, (grouping,)
        ),
    )


def sampling_profile() -> core.ExactTransformCoreProfileV1:
    source_refs = tuple(ref(SOURCE, f"sample-{name}") for name in ("a", "b", "c"))
    source_metadata = tuple(
        metadata(
            component_ref,
            label="series",
            axis=core.CoreComponentAxisV1.TEMPORAL,
            temporal=core.CoreTemporalSupportV1(CLOCK, F(index), F(index)),
        )
        for index, component_ref in enumerate(source_refs)
    )
    selected = (source_refs[0], source_refs[2])
    target_metadata = (
        replace(source_metadata[0], ref=ref(TARGET, "sample-a")),
        replace(source_metadata[2], ref=ref(TARGET, "sample-c")),
    )
    rows = (
        row(target_metadata[0].ref, ((source_refs[0], F(1)),)),
        row(target_metadata[1].ref, ((source_refs[2], F(1)),)),
    )
    return profile(
        operation=core.CoreTransformOperationV1.SAMPLING_RESOLUTION,
        sources=tuple(
            numeric(item, F(index + 1), F(index + 1, 10))
            for index, item in enumerate(source_metadata)
        ),
        outputs=target_metadata,
        rows=rows,
        certificate=core.CoreSamplingResolutionCertificateV1(
            axis=core.CoreComponentAxisV1.TEMPORAL,
            selected_inputs=selected,
            discarded_inputs=(source_refs[1],),
            grid_points=(
                core.CoreSamplingGridPointV1(source_refs[0], (F(0),)),
                core.CoreSamplingGridPointV1(source_refs[2], (F(2),)),
            ),
            grid_dimension=1,
            grid_frame_binding=None,
        ),
    )


def split_profile() -> core.ExactTransformCoreProfileV1:
    source_ref = ref(SOURCE, "split")
    target_refs = (ref(TARGET, "split-a"), ref(TARGET, "split-b"))
    source_metadata = metadata(
        source_ref,
        label="mass",
        role=core.CoreComponentValueRoleV1.EXTENSIVE,
    )
    target_metadata = tuple(
        metadata(
            item,
            label=f"mass-{index}",
            role=core.CoreComponentValueRoleV1.EXTENSIVE,
        )
        for index, item in enumerate(target_refs)
    )
    grouping = group((source_ref,), target_refs, "split-equivalence")
    rows = (
        row(target_refs[0], ((source_ref, F(1, 4)),)),
        row(target_refs[1], ((source_ref, F(3, 4)),)),
    )
    inverse = (
        row(
            source_ref,
            ((target_refs[0], F(1)), (target_refs[1], F(1))),
        ),
    )
    label_bindings = tuple(
        core.CoreOutputLabelBindingV1(
            input_refs=(source_ref,),
            output_ref=item.ref,
            output_label=item.component_label,
            quotient_label="split-equivalence",
        )
        for item in target_metadata
    )
    return profile(
        operation=core.CoreTransformOperationV1.EQUIVALENT_SPLIT_MERGE,
        sources=(numeric(source_metadata, F(8), F(1)),),
        outputs=target_metadata,
        rows=rows,
        certificate=core.CoreEquivalentSplitMergeCertificateV1(
            core.CoreSplitMergeDirectionV1.SPLIT,
            (grouping,),
            inverse,
            label_bindings,
        ),
    )


def merge_profile() -> core.ExactTransformCoreProfileV1:
    source_refs = (ref(SOURCE, "merge-a"), ref(SOURCE, "merge-b"))
    target_ref = ref(TARGET, "merge")
    source_metadata = tuple(
        metadata(
            item,
            label=f"mass-part-{index}",
            role=core.CoreComponentValueRoleV1.EXTENSIVE,
        )
        for index, item in enumerate(source_refs)
    )
    target_metadata = metadata(
        target_ref,
        label="mass-merged",
        role=core.CoreComponentValueRoleV1.EXTENSIVE,
    )
    grouping = group(source_refs, (target_ref,), "merge-equivalence")
    forward = (
        row(target_ref, ((source_refs[0], F(1)), (source_refs[1], F(1)))),
    )
    inverse = (
        row(source_refs[0], ((target_ref, F(1, 4)),)),
        row(source_refs[1], ((target_ref, F(3, 4)),)),
    )
    label_binding = core.CoreOutputLabelBindingV1(
        input_refs=source_refs,
        output_ref=target_ref,
        output_label="mass-merged",
        quotient_label="merge-equivalence",
    )
    return profile(
        operation=core.CoreTransformOperationV1.EQUIVALENT_SPLIT_MERGE,
        sources=(
            numeric(source_metadata[0], F(2), F(1, 2)),
            numeric(source_metadata[1], F(6), F(3, 2)),
        ),
        outputs=(target_metadata,),
        rows=forward,
        certificate=core.CoreEquivalentSplitMergeCertificateV1(
            core.CoreSplitMergeDirectionV1.MERGE,
            (grouping,),
            inverse,
            (label_binding,),
        ),
    )


def coarse_profile() -> core.ExactTransformCoreProfileV1:
    source_refs = (ref(SOURCE, "coarse-a"), ref(SOURCE, "coarse-b"))
    target_ref = ref(TARGET, "coarse")
    source_metadata = tuple(
        metadata(
            item,
            label="amount",
            role=core.CoreComponentValueRoleV1.EXTENSIVE,
        )
        for item in source_refs
    )
    target_metadata = metadata(
        target_ref,
        label="amount",
        axis=core.CoreComponentAxisV1.COARSE,
        role=core.CoreComponentValueRoleV1.EXTENSIVE,
    )
    transform_row = row(
        target_ref,
        ((source_refs[0], F(1)), (source_refs[1], F(1))),
    )
    source_commutation = tuple(
        row(item, ((item, F(1)),)) for item in source_refs
    )
    target_commutation = (row(target_ref, ((target_ref, F(1)),)),)
    grouping = group(source_refs, (target_ref,), "quotient-a")
    label_binding = core.CoreOutputLabelBindingV1(
        input_refs=source_refs,
        output_ref=target_ref,
        output_label=target_metadata.component_label,
        quotient_label="quotient-a",
    )
    return profile(
        operation=core.CoreTransformOperationV1.COARSE_GRAINING,
        sources=(
            numeric(source_metadata[0], F(2), F(1, 10)),
            numeric(source_metadata[1], F(4), F(1, 5)),
        ),
        outputs=(target_metadata,),
        rows=(transform_row,),
        certificate=core.CoreCoarseGrainingCertificateV1(
            reducer=core.CoreReducerV1.SUM,
            groups=(grouping,),
            quotient_labels=("quotient-a",),
            source_commutation_rows=source_commutation,
            target_commutation_rows=target_commutation,
            label_bindings=(label_binding,),
        ),
    )


def test_identity_exact_numeric_boolean_and_missing_copy() -> None:
    outputs = complete(identity_profile())
    assert tuple(item.value_kind for item in outputs) == (
        core.CoreComponentValueKindV1.NUMERIC,
        core.CoreComponentValueKindV1.BOOLEAN,
        core.CoreComponentValueKindV1.MISSING,
    )
    assert (outputs[0].numeric_value, outputs[0].absolute_radius) == (
        F(3, 2), F(1, 10)
    )
    assert outputs[1].boolean_value is True
    assert outputs[2].boolean_value is None


@pytest.mark.parametrize(
    "factory,expected",
    (
        (unit_profile, ((F(200), F(10)),)),
        (offset_unit_profile, ((F(10), F(1)),)),
        (coordinate_profile, ((F(3), F(2, 5)), (F(1), F(1, 5)))),
        (temporal_profile, ((F(3), F(3, 10)),)),
        (spatial_profile, ((F(3), F(3, 20)),)),
        (sampling_profile, ((F(1), F(1, 10)), (F(3), F(3, 10)))),
        (split_profile, ((F(2), F(1, 4)), (F(6), F(3, 4)))),
        (merge_profile, ((F(8), F(2)),)),
        (coarse_profile, ((F(6), F(3, 10)),)),
    ),
)
def test_nonidentity_operations_use_literal_exact_oracles(
    factory, expected
) -> None:
    outputs = complete(factory())
    assert tuple(
        (item.numeric_value, item.absolute_radius) for item in outputs
    ) == expected


def test_verify_fresh_run_rejects_value_radius_and_metadata_claims_atomically() -> None:
    item = unit_profile()
    expected = complete(item)
    assert core.verify_exact_transform_core_outputs_v1(
        item, expected
    ).disposition is core.CoreTransformDispositionV1.COMPLETE

    bad_value = (replace(expected[0], numeric_value=F(201)),)
    bad_radius = (replace(expected[0], absolute_radius=F(11)),)
    bad_metadata = (
        replace(
            expected[0],
            metadata=replace(expected[0].metadata, component_label="altered"),
        ),
    )
    cases = (
        (bad_value, core.CoreTransformFailureCodeV1.CLAIM_VALUE_MISMATCH),
        (bad_radius, core.CoreTransformFailureCodeV1.CLAIM_RADIUS_MISMATCH),
        (bad_metadata, core.CoreTransformFailureCodeV1.CLAIM_METADATA_MISMATCH),
    )
    for claim, code in cases:
        result = core.verify_exact_transform_core_outputs_v1(item, claim)
        assert result.disposition is core.CoreTransformDispositionV1.ABSTAIN
        assert result.output_components == ()
        assert result.failure is not None and result.failure.code is code

    standard_error_claim = (
        replace(
            expected[0],
            uncertainty_model=core.CoreUncertaintyModelV1.STANDARD_ERROR,
        ),
    )
    standard_error = core.verify_exact_transform_core_outputs_v1(
        item, standard_error_claim
    )
    assert standard_error.output_components == ()
    assert standard_error.failure is not None
    assert (
        standard_error.failure.code
        is core.CoreTransformFailureCodeV1.STANDARD_ERROR_UNSUPPORTED
    )


def test_standard_error_precedes_fraction_budget_and_is_atomic() -> None:
    item = identity_profile()
    source = list(item.source_components)
    source[0] = replace(
        source[0],
        uncertainty_model=core.CoreUncertaintyModelV1.STANDARD_ERROR,
        numeric_value=F(1 << 5000),
    )
    attacked = replace(item, source_components=tuple(source))
    result = core.run_exact_transform_core_v1(attacked)
    assert result.disposition is core.CoreTransformDispositionV1.ABSTAIN
    assert result.output_components == ()
    assert result.failure is not None
    assert (
        result.failure.code
        is core.CoreTransformFailureCodeV1.STANDARD_ERROR_UNSUPPORTED
    )


def test_fraction_resource_limit_and_bool_is_not_an_integer() -> None:
    item = identity_profile()
    source = list(item.source_components)
    source[0] = replace(source[0], numeric_value=F(1 << 5000))
    result = core.run_exact_transform_core_v1(
        replace(item, source_components=tuple(source))
    )
    assert result.disposition is core.CoreTransformDispositionV1.ABSTAIN
    assert result.output_components == ()
    assert result.failure is not None
    assert result.failure.code is core.CoreTransformFailureCodeV1.RESOURCE_LIMIT
    with pytest.raises(TypeError):
        core.CoreComponentRefV1(SOURCE, core.CoreObservationRefV1("bad"), True)


def test_profile_component_count_is_preflighted_before_recursive_expansion() -> None:
    item = unit_profile()
    attacked = replace(item)
    object.__setattr__(
        attacked,
        "source_components",
        (item.source_components[0],) * 65_537,
    )
    result = core.run_exact_transform_core_v1(attacked)
    assert result.output_components == ()
    assert result.failure is not None
    assert result.failure.code is core.CoreTransformFailureCodeV1.RESOURCE_LIMIT


def test_same_unit_and_inverse_factor_attacks_reject() -> None:
    item = unit_profile()
    source_unit = item.source_components[0].metadata.quantity
    output = replace(item.contract.output_metadata[0], quantity=source_unit)
    source_ref = item.contract.input_refs[0]
    attacked_contract = replace(
        item.contract,
        output_metadata=(output,),
        kernel_rows=(row(output.ref, ((source_ref, F(1)),)),),
        certificate=core.CoreUnitConversionCertificateV1(F(1), F(1)),
    )
    same_unit = core.run_exact_transform_core_v1(
        replace(item, contract=attacked_contract)
    )
    assert same_unit.output_components == ()
    assert same_unit.failure is not None
    assert (
        same_unit.failure.code
        is core.CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID
    )

    wrong_inverse = replace(
        item,
        contract=replace(
            item.contract,
            certificate=core.CoreUnitConversionCertificateV1(F(100), F(1, 99)),
        ),
    )
    result = core.run_exact_transform_core_v1(wrong_inverse)
    assert result.output_components == ()
    assert result.failure is not None
    assert (
        result.failure.code
        is core.CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID
    )


def test_sampling_selected_discarded_is_not_forced_into_kernel_and_overlap_rejects() -> None:
    valid = complete(sampling_profile())
    assert tuple(item.numeric_value for item in valid) == (F(1), F(3))
    item = sampling_profile()
    certificate = item.contract.certificate
    assert type(certificate) is core.CoreSamplingResolutionCertificateV1
    attacked_certificate = replace(
        certificate,
        discarded_inputs=tuple(
            sorted((certificate.selected_inputs[0], *certificate.discarded_inputs))
        ),
    )
    result = core.run_exact_transform_core_v1(
        replace(item, contract=replace(item.contract, certificate=attacked_certificate))
    )
    assert result.output_components == ()
    assert result.failure is not None
    assert (
        result.failure.code
        is core.CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID
    )

    nonpoint_sources = list(item.source_components)
    nonpoint_sources[1] = replace(
        nonpoint_sources[1],
        metadata=replace(
            nonpoint_sources[1].metadata,
            temporal_support=core.CoreTemporalSupportV1(CLOCK, F(1), F(3, 2)),
        ),
    )
    nonpoint = core.run_exact_transform_core_v1(
        replace(item, source_components=tuple(nonpoint_sources))
    )
    assert nonpoint.output_components == ()
    assert nonpoint.failure is not None
    assert (
        nonpoint.failure.code
        is core.CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID
    )

    wrong_grid_certificate = replace(
        certificate,
        grid_points=(
            core.CoreSamplingGridPointV1(
                certificate.selected_inputs[0], (F(1, 2),)
            ),
            certificate.grid_points[1],
        ),
    )
    wrong_grid = core.run_exact_transform_core_v1(
        replace(
            item,
            contract=replace(item.contract, certificate=wrong_grid_certificate),
        )
    )
    assert wrong_grid.output_components == ()
    assert wrong_grid.failure is not None
    assert (
        wrong_grid.failure.code
        is core.CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID
    )

    repeated_sources = list(item.source_components)
    repeated_sources[2] = replace(
        repeated_sources[2],
        metadata=replace(
            repeated_sources[2].metadata,
            temporal_support=core.CoreTemporalSupportV1(CLOCK, F(0), F(0)),
        ),
    )
    repeated = core.run_exact_transform_core_v1(
        replace(item, source_components=tuple(repeated_sources))
    )
    assert repeated.output_components == ()
    assert repeated.failure is not None
    assert (
        repeated.failure.code
        is core.CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID
    )


def test_nonidentity_boolean_phantoms_all_discarded_and_empty_groups_reject() -> None:
    source_ref = ref(SOURCE, "phantom-boolean")
    target_ref = ref(TARGET, "phantom-boolean")
    source_meta = metadata(
        source_ref,
        label="flag",
        quantity=BOOLEAN_QUANTITY,
        axis=core.CoreComponentAxisV1.CONTROL,
        role=core.CoreComponentValueRoleV1.BOOLEAN_CONTROL,
    )
    target_meta = replace(source_meta, ref=target_ref)
    mapping = core.CoreDiscreteMappingV1(source_ref, target_ref)
    boolean_source = (boolean(source_meta, True),)

    unit_phantom = profile(
        operation=core.CoreTransformOperationV1.UNIT_CONVERSION,
        sources=boolean_source,
        outputs=(target_meta,),
        rows=(),
        mappings=(mapping,),
        certificate=core.CoreUnitConversionCertificateV1(F(2), F(1, 2)),
    )
    unit_result = core.run_exact_transform_core_v1(unit_phantom)
    assert unit_result.output_components == ()
    assert unit_result.failure is not None
    assert (
        unit_result.failure.code
        is core.CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID
    )

    temporal_group = group((source_ref,), (target_ref,), "phantom-bin")
    temporal_phantom = profile(
        operation=core.CoreTransformOperationV1.TEMPORAL_AGGREGATION,
        sources=boolean_source,
        outputs=(target_meta,),
        rows=(),
        mappings=(mapping,),
        certificate=core.CoreTemporalAggregationCertificateV1(
            core.CoreReducerV1.SUM, (temporal_group,)
        ),
    )
    temporal_result = core.run_exact_transform_core_v1(temporal_phantom)
    assert temporal_result.output_components == ()
    assert temporal_result.failure is not None
    assert (
        temporal_result.failure.code
        is core.CoreTransformFailureCodeV1.TEMPORAL_AGGREGATION_CONTRACT_INVALID
    )

    discarded_ref = ref(SOURCE, "all-discarded")
    discarded_meta = metadata(
        discarded_ref,
        label="series",
        axis=core.CoreComponentAxisV1.TEMPORAL,
        temporal=core.CoreTemporalSupportV1(CLOCK, F(0), F(0)),
    )
    all_discarded = profile(
        operation=core.CoreTransformOperationV1.SAMPLING_RESOLUTION,
        sources=(numeric(discarded_meta, F(1), F(0)), *boolean_source),
        outputs=(target_meta,),
        rows=(),
        mappings=(mapping,),
        certificate=core.CoreSamplingResolutionCertificateV1(
            axis=core.CoreComponentAxisV1.TEMPORAL,
            selected_inputs=(),
            discarded_inputs=(discarded_ref,),
            grid_points=(),
            grid_dimension=1,
            grid_frame_binding=None,
        ),
    )
    discarded_result = core.run_exact_transform_core_v1(all_discarded)
    assert discarded_result.output_components == ()
    assert discarded_result.failure is not None
    assert (
        discarded_result.failure.code
        is core.CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID
    )

    empty_group_profile = temporal_profile()
    empty_certificate = empty_group_profile.contract.certificate
    object.__setattr__(empty_certificate, "groups", ())
    empty_result = core.run_exact_transform_core_v1(empty_group_profile)
    assert empty_result.output_components == ()
    assert empty_result.failure is not None
    assert empty_result.failure.code is core.CoreTransformFailureCodeV1.SCHEMA_INVALID


def test_coordinate_temporal_spatial_split_and_coarse_attacks_are_atomic() -> None:
    coordinate = coordinate_profile()
    coordinate_certificate = coordinate.contract.certificate
    assert type(coordinate_certificate) is core.CoreCoordinateAffineCertificateV1
    bad_inverse = replace(
        coordinate_certificate,
        inverse_rows=(
            row(
                coordinate.contract.input_refs[0],
                ((coordinate.contract.output_metadata[0].ref, F(1)),),
            ),
            row(
                coordinate.contract.input_refs[1],
                ((coordinate.contract.output_metadata[1].ref, F(1)),),
            ),
        ),
    )

    temporal = temporal_profile()
    temporal_source = list(temporal.source_components)
    temporal_source[1] = replace(
        temporal_source[1],
        metadata=replace(
            temporal_source[1].metadata,
            temporal_support=core.CoreTemporalSupportV1(
                core.CoreClockBindingV1("clock:other"), F(1), F(2)
            ),
        ),
    )

    spatial = spatial_profile()
    spatial_source = list(spatial.source_components)
    spatial_source[1] = replace(
        spatial_source[1],
        metadata=replace(
            spatial_source[1].metadata,
            spatial_support=core.CoreSpatialSupportV1(
                FRAME_A, (F(1, 2), F(0)), (F(3, 2), F(1))
            ),
        ),
    )

    split = split_profile()
    split_rows = list(split.contract.kernel_rows)
    split_rows[0] = row(
        split_rows[0].output_ref,
        ((split.contract.input_refs[0], F(-1, 4)),),
    )
    split_rows[1] = row(
        split_rows[1].output_ref,
        ((split.contract.input_refs[0], F(5, 4)),),
    )

    coarse = coarse_profile()
    coarse_certificate = coarse.contract.certificate
    assert type(coarse_certificate) is core.CoreCoarseGrainingCertificateV1
    target_ref = coarse.contract.output_metadata[0].ref
    bad_commutation = replace(
        coarse_certificate,
        target_commutation_rows=(row(target_ref, ((target_ref, F(2)),)),),
    )

    attacks = (
        replace(
            coordinate,
            contract=replace(coordinate.contract, certificate=bad_inverse),
        ),
        replace(temporal, source_components=tuple(temporal_source)),
        replace(spatial, source_components=tuple(spatial_source)),
        replace(
            split,
            contract=replace(split.contract, kernel_rows=tuple(split_rows)),
        ),
        replace(
            coarse,
            contract=replace(coarse.contract, certificate=bad_commutation),
        ),
    )
    expected_codes = (
        core.CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID,
        core.CoreTransformFailureCodeV1.TEMPORAL_AGGREGATION_CONTRACT_INVALID,
        core.CoreTransformFailureCodeV1.SPATIAL_AGGREGATION_CONTRACT_INVALID,
        core.CoreTransformFailureCodeV1.EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID,
        core.CoreTransformFailureCodeV1.COARSE_GRAINING_CONTRACT_INVALID,
    )
    for attacked, expected_code in zip(attacks, expected_codes, strict=True):
        result = core.run_exact_transform_core_v1(attacked)
        assert result.disposition is core.CoreTransformDispositionV1.ABSTAIN
        assert result.output_components == ()
        assert result.failure is not None and result.failure.code is expected_code


def test_operation_certificate_exact_type_mismatch_is_typed() -> None:
    item = identity_profile()
    attacked = replace(
        item,
        contract=replace(
            item.contract,
            certificate=core.CoreUnitConversionCertificateV1(F(1), F(1)),
        ),
    )
    with pytest.raises(core.ExactTransformCoreValidationErrorV1) as caught:
        core.validate_exact_transform_core_v1(attacked)
    assert (
        caught.value.code
        is core.CoreTransformFailureCodeV1.CERTIFICATE_TYPE_MISMATCH
    )
    result = core.run_exact_transform_core_v1(attacked)
    assert result.output_components == ()
    assert result.failure is not None
    assert (
        result.failure.code
        is core.CoreTransformFailureCodeV1.CERTIFICATE_TYPE_MISMATCH
    )


def test_certificate_groups_and_auxiliary_rows_are_canonical_unique() -> None:
    coordinate = coordinate_profile()
    coordinate_certificate = coordinate.contract.certificate
    assert type(coordinate_certificate) is core.CoreCoordinateAffineCertificateV1
    reordered = replace(
        coordinate_certificate,
        inverse_rows=tuple(reversed(coordinate_certificate.inverse_rows)),
    )
    reordered_result = core.run_exact_transform_core_v1(
        replace(
            coordinate,
            contract=replace(coordinate.contract, certificate=reordered),
        )
    )
    assert reordered_result.output_components == ()
    assert reordered_result.failure is not None
    assert (
        reordered_result.failure.code
        is core.CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID
    )

    temporal = temporal_profile()
    temporal_certificate = temporal.contract.certificate
    assert type(temporal_certificate) is core.CoreTemporalAggregationCertificateV1
    duplicated_groups = replace(
        temporal_certificate,
        groups=(temporal_certificate.groups[0], temporal_certificate.groups[0]),
    )
    duplicated_group_result = core.run_exact_transform_core_v1(
        replace(
            temporal,
            contract=replace(temporal.contract, certificate=duplicated_groups),
        )
    )
    assert duplicated_group_result.output_components == ()
    assert duplicated_group_result.failure is not None
    assert (
        duplicated_group_result.failure.code
        is core.CoreTransformFailureCodeV1.TEMPORAL_AGGREGATION_CONTRACT_INVALID
    )

    coarse = coarse_profile()
    coarse_certificate = coarse.contract.certificate
    assert type(coarse_certificate) is core.CoreCoarseGrainingCertificateV1
    duplicated_rows = replace(
        coarse_certificate,
        target_commutation_rows=(
            coarse_certificate.target_commutation_rows[0],
            coarse_certificate.target_commutation_rows[0],
        ),
    )
    duplicated_row_result = core.run_exact_transform_core_v1(
        replace(
            coarse,
            contract=replace(coarse.contract, certificate=duplicated_rows),
        )
    )
    assert duplicated_row_result.output_components == ()
    assert duplicated_row_result.failure is not None
    assert (
        duplicated_row_result.failure.code
        is core.CoreTransformFailureCodeV1.COARSE_GRAINING_CONTRACT_INVALID
    )

    split = split_profile()
    split_certificate = split.contract.certificate
    assert type(split_certificate) is core.CoreEquivalentSplitMergeCertificateV1
    for attacked_bindings in (
        tuple(reversed(split_certificate.label_bindings)),
        (
            split_certificate.label_bindings[0],
            split_certificate.label_bindings[0],
        ),
    ):
        binding_result = core.run_exact_transform_core_v1(
            replace(
                split,
                contract=replace(
                    split.contract,
                    certificate=replace(
                        split_certificate, label_bindings=attacked_bindings
                    ),
                ),
            )
        )
        assert binding_result.output_components == ()
        assert binding_result.failure is not None
        assert (
            binding_result.failure.code
            is core.CoreTransformFailureCodeV1.EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID
        )


def test_component_labels_and_coarse_quotient_bindings_are_semantic() -> None:
    unit = unit_profile()
    unit_output = replace(unit.contract.output_metadata[0], component_label="renamed")
    unit_result = core.run_exact_transform_core_v1(
        replace(
            unit,
            contract=replace(unit.contract, output_metadata=(unit_output,)),
        )
    )
    assert unit_result.output_components == ()
    assert unit_result.failure is not None
    assert (
        unit_result.failure.code
        is core.CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID
    )

    coordinate = coordinate_profile()
    coordinate_outputs = list(coordinate.contract.output_metadata)
    coordinate_outputs[0] = replace(
        coordinate_outputs[0], component_label="renamed-coordinate"
    )
    coordinate_result = core.run_exact_transform_core_v1(
        replace(
            coordinate,
            contract=replace(
                coordinate.contract, output_metadata=tuple(coordinate_outputs)
            ),
        )
    )
    assert coordinate_result.output_components == ()
    assert coordinate_result.failure is not None
    assert (
        coordinate_result.failure.code
        is core.CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID
    )

    temporal = temporal_profile()
    temporal_output = replace(
        temporal.contract.output_metadata[0], component_label="renamed-flow"
    )
    temporal_result = core.run_exact_transform_core_v1(
        replace(
            temporal,
            contract=replace(temporal.contract, output_metadata=(temporal_output,)),
        )
    )
    assert temporal_result.output_components == ()
    assert temporal_result.failure is not None
    assert (
        temporal_result.failure.code
        is core.CoreTransformFailureCodeV1.TEMPORAL_AGGREGATION_CONTRACT_INVALID
    )

    split = split_profile()
    split_certificate = split.contract.certificate
    assert type(split_certificate) is core.CoreEquivalentSplitMergeCertificateV1
    split_bindings = list(split_certificate.label_bindings)
    split_bindings[0] = replace(split_bindings[0], output_label="unbound-label")
    split_result = core.run_exact_transform_core_v1(
        replace(
            split,
            contract=replace(
                split.contract,
                certificate=replace(
                    split_certificate, label_bindings=tuple(split_bindings)
                ),
            ),
        )
    )
    assert split_result.output_components == ()
    assert split_result.failure is not None
    assert (
        split_result.failure.code
        is core.CoreTransformFailureCodeV1.EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID
    )

    coarse = coarse_profile()
    coarse_certificate = coarse.contract.certificate
    assert type(coarse_certificate) is core.CoreCoarseGrainingCertificateV1
    wrong_quotient = replace(
        coarse_certificate.label_bindings[0], quotient_label="wrong-quotient"
    )
    coarse_result = core.run_exact_transform_core_v1(
        replace(
            coarse,
            contract=replace(
                coarse.contract,
                certificate=replace(
                    coarse_certificate, label_bindings=(wrong_quotient,)
                ),
            ),
        )
    )
    assert coarse_result.output_components == ()
    assert coarse_result.failure is not None
    assert (
        coarse_result.failure.code
        is core.CoreTransformFailureCodeV1.COARSE_GRAINING_CONTRACT_INVALID
    )


def test_weighted_mean_uses_declared_coefficients_not_support_measure() -> None:
    item = spatial_profile()
    source_refs = item.contract.input_refs
    output_ref = item.contract.output_metadata[0].ref
    declared = row(
        output_ref,
        ((source_refs[0], F(1, 4)), (source_refs[1], F(3, 4))),
    )
    changed = replace(item, contract=replace(item.contract, kernel_rows=(declared,)))
    outputs = complete(changed)
    assert (outputs[0].numeric_value, outputs[0].absolute_radius) == (
        F(7, 2),
        F(7, 40),
    )
    certificate = changed.contract.certificate
    assert type(certificate) is core.CoreSpatialAggregationCertificateV1
    assert (
        certificate.weighted_mean_basis
        is core.CoreWeightedMeanBasisV1.CALLER_DECLARED_EXACT_COEFFICIENTS
    )


def test_fresh_recursive_validation_rejects_forged_nested_state_and_cycles() -> None:
    base = identity_profile()

    forged_metadata = replace(base.source_components[0].metadata)
    forged_component = replace(base.source_components[0], metadata=forged_metadata)
    object.__setattr__(forged_metadata, "entity_keys", ("z", "a"))
    sources = (forged_component, *base.source_components[1:])
    unsorted_result = core.run_exact_transform_core_v1(
        replace(base, source_components=sources)
    )
    assert unsorted_result.output_components == ()
    assert unsorted_result.failure is not None
    assert unsorted_result.failure.code is core.CoreTransformFailureCodeV1.SCHEMA_INVALID

    zero_term = replace(base.contract.kernel_rows[0].terms[0])
    object.__setattr__(zero_term, "coefficient", F(0))
    forged_row = replace(base.contract.kernel_rows[0])
    object.__setattr__(forged_row, "terms", (zero_term,))
    zero_result = core.run_exact_transform_core_v1(
        replace(
            base,
            contract=replace(base.contract, kernel_rows=(forged_row,)),
        )
    )
    assert zero_result.output_components == ()
    assert zero_result.failure is not None
    assert zero_result.failure.code is core.CoreTransformFailureCodeV1.SCHEMA_INVALID

    unit = unit_profile()
    negative_unit = replace(UNIT_A)
    object.__setattr__(negative_unit, "scale_to_si", F(-1))
    negative_quantity = replace(
        unit.source_components[0].metadata.quantity, unit_binding=negative_unit
    )
    negative_metadata = replace(
        unit.source_components[0].metadata, quantity=negative_quantity
    )
    negative_component = replace(
        unit.source_components[0], metadata=negative_metadata
    )
    negative_result = core.run_exact_transform_core_v1(
        replace(unit, source_components=(negative_component,))
    )
    assert negative_result.output_components == ()
    assert negative_result.failure is not None
    assert negative_result.failure.code is core.CoreTransformFailureCodeV1.SCHEMA_INVALID

    cyclic_metadata = replace(base.source_components[0].metadata)
    cyclic_component = replace(base.source_components[0], metadata=cyclic_metadata)
    object.__setattr__(cyclic_metadata, "quantity", cyclic_metadata)
    cyclic_result = core.run_exact_transform_core_v1(
        replace(
            base,
            source_components=(cyclic_component, *base.source_components[1:]),
        )
    )
    assert cyclic_result.output_components == ()
    assert cyclic_result.failure is not None
    assert cyclic_result.failure.code is core.CoreTransformFailureCodeV1.SCHEMA_INVALID

    huge_ref = replace(base.source_components[0].ref)
    object.__setattr__(huge_ref, "component_index", 1 << 5000)
    huge_metadata = replace(base.source_components[0].metadata, ref=huge_ref)
    huge_component = replace(base.source_components[0], metadata=huge_metadata)
    huge_result = core.run_exact_transform_core_v1(
        replace(
            base,
            source_components=(huge_component, *base.source_components[1:]),
        )
    )
    assert huge_result.output_components == ()
    assert huge_result.failure is not None
    assert huge_result.failure.code is core.CoreTransformFailureCodeV1.RESOURCE_LIMIT

    noncanonical_fraction = F(1, 2)
    object.__setattr__(noncanonical_fraction, "_numerator", 2)
    object.__setattr__(noncanonical_fraction, "_denominator", 4)
    forged_fraction_component = replace(base.source_components[0])
    object.__setattr__(
        forged_fraction_component, "numeric_value", noncanonical_fraction
    )
    forged_fraction_result = core.run_exact_transform_core_v1(
        replace(
            base,
            source_components=(
                forged_fraction_component,
                *base.source_components[1:],
            ),
        )
    )
    assert forged_fraction_result.output_components == ()
    assert forged_fraction_result.failure is not None
    assert (
        forged_fraction_result.failure.code
        is core.CoreTransformFailureCodeV1.SCHEMA_INVALID
    )


def test_claim_verifier_distinguishes_all_mismatch_and_closure_reasons() -> None:
    numeric_profile = unit_profile()
    numeric_expected = complete(numeric_profile)
    expected = numeric_expected[0]

    changed_ref_metadata = replace(
        expected.metadata, ref=ref(TARGET, "different-claim-ref")
    )
    reference_claim = (replace(expected, metadata=changed_ref_metadata),)

    boolean_metadata = metadata(
        expected.ref,
        label="claim-boolean",
        quantity=BOOLEAN_QUANTITY,
        axis=core.CoreComponentAxisV1.CONTROL,
        role=core.CoreComponentValueRoleV1.BOOLEAN_CONTROL,
    )
    kind_claim = (boolean(boolean_metadata, True),)

    missing_metadata = metadata(
        expected.ref,
        label="claim-missing",
        quantity=MISSING_QUANTITY,
        role=core.CoreComponentValueRoleV1.MISSING,
    )
    missing_claim = (missing(missing_metadata),)

    metadata_claim = (
        replace(
            expected,
            metadata=replace(expected.metadata, component_label="claim-renamed"),
        ),
    )
    value_claim = (replace(expected, numeric_value=F(201)),)
    radius_claim = (replace(expected, absolute_radius=F(11)),)

    mismatch_cases = (
        (reference_claim, core.CoreTransformFailureCodeV1.CLAIM_REFERENCE_MISMATCH),
        (kind_claim, core.CoreTransformFailureCodeV1.CLAIM_KIND_MISMATCH),
        (missing_claim, core.CoreTransformFailureCodeV1.CLAIM_MISSING_MISMATCH),
        (metadata_claim, core.CoreTransformFailureCodeV1.CLAIM_METADATA_MISMATCH),
        (value_claim, core.CoreTransformFailureCodeV1.CLAIM_VALUE_MISMATCH),
        (radius_claim, core.CoreTransformFailureCodeV1.CLAIM_RADIUS_MISMATCH),
    )
    for claim, expected_code in mismatch_cases:
        result = core.verify_exact_transform_core_outputs_v1(
            numeric_profile, claim
        )
        assert result.output_components == ()
        assert result.failure is not None and result.failure.code is expected_code

    identity = identity_profile()
    identity_expected = list(complete(identity))
    identity_expected[1] = replace(
        identity_expected[1], boolean_value=False
    )
    boolean_result = core.verify_exact_transform_core_outputs_v1(
        identity, tuple(identity_expected)
    )
    assert boolean_result.output_components == ()
    assert boolean_result.failure is not None
    assert (
        boolean_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_BOOLEAN_MISMATCH
    )

    forged_metadata = replace(expected.metadata)
    forged_claim = replace(expected, metadata=forged_metadata)
    object.__setattr__(forged_metadata, "role_keys", ["not-a-tuple"])
    forged_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile, (forged_claim,)
    )
    assert forged_result.output_components == ()
    assert forged_result.failure is not None
    assert (
        forged_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID
    )

    forged_claim_fraction = F(1, 2)
    object.__setattr__(forged_claim_fraction, "_numerator", 2)
    object.__setattr__(forged_claim_fraction, "_denominator", 4)
    forged_fraction_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile,
        (replace(expected, numeric_value=forged_claim_fraction),),
    )
    assert forged_fraction_result.output_components == ()
    assert forged_fraction_result.failure is not None
    assert (
        forged_fraction_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID
    )

    missing_fields = object.__new__(core.CoreComponentValueV1)
    missing_fields_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile, (missing_fields,)
    )
    assert missing_fields_result.output_components == ()
    assert missing_fields_result.failure is not None
    assert (
        missing_fields_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID
    )

    wrong_dataclass_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile, (numeric_profile,)
    )
    assert wrong_dataclass_result.output_components == ()
    assert wrong_dataclass_result.failure is not None
    assert (
        wrong_dataclass_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID
    )

    huge_label_claim = (
        replace(
            expected,
            metadata=replace(expected.metadata, component_label="x" * 5000),
        ),
    )
    huge_label_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile, huge_label_claim
    )
    assert huge_label_result.output_components == ()
    assert huge_label_result.failure is not None
    assert (
        huge_label_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_RESOURCE_LIMIT
    )

    huge_fraction_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile, (replace(expected, numeric_value=F(1 << 5000)),)
    )
    assert huge_fraction_result.output_components == ()
    assert huge_fraction_result.failure is not None
    assert (
        huge_fraction_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_RESOURCE_LIMIT
    )

    huge_count_result = core.verify_exact_transform_core_outputs_v1(
        numeric_profile, (expected,) * 65_537
    )
    assert huge_count_result.output_components == ()
    assert huge_count_result.failure is not None
    assert (
        huge_count_result.failure.code
        is core.CoreTransformFailureCodeV1.CLAIM_RESOURCE_LIMIT
    )


def two_numeric_identity_profile() -> core.ExactTransformCoreProfileV1:
    source_refs = (ref(SOURCE, "legacy", 0), ref(SOURCE, "legacy", 1))
    target_refs = (ref(TARGET, "legacy", 0), ref(TARGET, "legacy", 1))
    source_metadata = tuple(
        metadata(component_ref, label=f"legacy-{index}")
        for index, component_ref in enumerate(source_refs)
    )
    target_metadata = tuple(
        replace(source_metadata[index], ref=target_refs[index]) for index in range(2)
    )
    return profile(
        operation=core.CoreTransformOperationV1.IDENTITY,
        sources=(
            numeric(source_metadata[0], F(1), F(0)),
            numeric(source_metadata[1], F(-2), F(0)),
        ),
        outputs=target_metadata,
        rows=(
            row(target_refs[0], ((source_refs[0], F(1)),)),
            row(target_refs[1], ((source_refs[1], F(1)),)),
        ),
        certificate=core.CoreIdentityCertificateV1(),
    )


def test_existing_legacy_identity_fixture_differential_uses_literal_oracle() -> None:
    legacy_test = Path(__file__).with_name(
        "test_phase2b_exact_transform_semantics_v1.py"
    )
    namespace = runpy.run_path(str(legacy_test))
    legacy_module = namespace["tx"]
    legacy_result = legacy_module.run_exact_transform_semantics(
        namespace["identity_authority"]()
    )
    legacy_target_scale = namespace["TARGET_SCALE"]
    legacy_target = tuple(
        item
        for item in legacy_result.components
        if item.descriptor.ref.scale_id == legacy_target_scale
    )
    legacy_intervals = tuple(
        (
            item.numeric_interval.lower_fraction,
            item.numeric_interval.upper_fraction,
        )
        for item in legacy_target
    )

    pure_outputs = complete(two_numeric_identity_profile())
    pure_intervals = tuple(
        (
            item.numeric_value - item.absolute_radius,
            item.numeric_value + item.absolute_radius,
        )
        for item in pure_outputs
        if item.numeric_value is not None and item.absolute_radius is not None
    )
    literal_oracle = ((F(1), F(1)), (F(-2), F(-2)))
    assert legacy_intervals == literal_oracle
    assert pure_intervals == literal_oracle


def test_identity_1024_component_capacity_qualification() -> None:
    source_refs = tuple(
        ref(SOURCE, f"capacity-{index:04d}") for index in range(1024)
    )
    target_refs = tuple(
        ref(TARGET, f"capacity-{index:04d}") for index in range(1024)
    )
    source_metadata = tuple(
        metadata(component_ref, label="capacity-value")
        for component_ref in source_refs
    )
    target_metadata = tuple(
        replace(source_metadata[index], ref=target_refs[index])
        for index in range(1024)
    )
    item = profile(
        operation=core.CoreTransformOperationV1.IDENTITY,
        sources=tuple(
            numeric(source_metadata[index], F(index), F(1, 1000))
            for index in range(1024)
        ),
        outputs=target_metadata,
        rows=tuple(
            row(target_refs[index], ((source_refs[index], F(1)),))
            for index in range(1024)
        ),
        certificate=core.CoreIdentityCertificateV1(),
    )
    outputs = complete(item)
    assert len(outputs) == 1024
    assert (outputs[0].numeric_value, outputs[0].absolute_radius) == (
        F(0),
        F(1, 1000),
    )
    assert (outputs[-1].numeric_value, outputs[-1].absolute_radius) == (
        F(1023),
        F(1, 1000),
    )


def test_public_core_version_and_no_caller_budget_surface() -> None:
    assert core.EXACT_TRANSFORM_CORE_VERSION_V1 == (
        "hegel-machine-phase2b-pure-exact-transform-core/1"
    )
    assert "policy" not in core.run_exact_transform_core_v1.__annotations__
    assert "policy" not in core.validate_exact_transform_core_v1.__annotations__

"""Pure exact arithmetic core for the eight Phase-2B transform operations.

This module deliberately has no dependency on a wire DTO, UUID syntax,
content-addressing, graph traversal, or the legacy binary64/grid compiler.  A
caller first projects one transform edge into the closed dataclasses below and
then either computes the target state or verifies a claimed target state.

Numeric cells carry an exact point value and an absolute-bound radius.  For an
affine row ``y = b + sum(a_i * x_i)`` the only executable uncertainty rule is
``r_y = sum(abs(a_i) * r_i)``.  ``standard_error`` is represented so that a
wire adapter can project it, but it is rejected before any core arithmetic.

All semantic failures are bundle-atomic: ``run_exact_transform_core_v1`` and
``verify_exact_transform_core_outputs_v1`` return no output components on any
failure.  The resource policy is frozen and cannot be overridden by callers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from math import gcd
from typing import Final, TypeAlias


EXACT_TRANSFORM_CORE_VERSION_V1: Final = (
    "hegel-machine-phase2b-pure-exact-transform-core/1"
)


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be exact nonempty text")
    return value


def _tuple(value: object, name: str, *, nonempty: bool = False) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if nonempty and not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _fraction(value: object, name: str) -> Fraction:
    if type(value) is not Fraction:
        raise TypeError(f"{name} must be an exact Fraction")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its minimum")
    return value


def _sorted_unique_text(values: tuple[str, ...], name: str) -> None:
    _tuple(values, name)
    if any(type(value) is not str or not value for value in values):
        raise TypeError(f"{name} must contain exact nonempty text")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be sorted and unique")


class CoreTransformDispositionV1(str, Enum):
    COMPLETE = "complete"
    ABSTAIN = "abstain"


class CoreTransformOperationV1(str, Enum):
    IDENTITY = "identity"
    UNIT_CONVERSION = "unit_conversion"
    COORDINATE_AFFINE = "coordinate_affine"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    SPATIAL_AGGREGATION = "spatial_aggregation"
    SAMPLING_RESOLUTION = "sampling_resolution"
    EQUIVALENT_SPLIT_MERGE = "equivalent_split_merge"
    COARSE_GRAINING = "coarse_graining"


class CoreComponentAxisV1(str, Enum):
    SCALAR = "scalar"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CHANNEL = "channel"
    ENTITY = "entity"
    COORDINATE = "coordinate"
    COARSE = "coarse"
    CONTROL = "control"


class CoreComponentValueRoleV1(str, Enum):
    INTENSIVE = "intensive"
    EXTENSIVE = "extensive"
    COORDINATE = "coordinate"
    BOOLEAN_CONTROL = "boolean_control"
    MISSING = "missing"


class CoreComponentValueKindV1(str, Enum):
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    MISSING = "missing"


class CoreUncertaintyModelV1(str, Enum):
    ABSOLUTE_BOUND = "absolute_bound"
    STANDARD_ERROR = "standard_error"
    NOT_APPLICABLE = "not_applicable"


class CoreReducerV1(str, Enum):
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"


class CoreWeightedMeanBasisV1(str, Enum):
    CALLER_DECLARED_EXACT_COEFFICIENTS = (
        "caller_declared_exact_coefficients_not_duration_or_volume_inference"
    )


class CoreMissingPolicyV1(str, Enum):
    REJECT = "reject"
    EXPLICIT_PRESERVE = "explicit_preserve"


class CoreBoundaryPolicyV1(str, Enum):
    EXACT_PARTITION = "exact_partition"
    VALID_ONLY = "valid_only"


class CoreSplitMergeDirectionV1(str, Enum):
    SPLIT = "split"
    MERGE = "merge"


class CoreTransformFailureCodeV1(str, Enum):
    INVALID_PROFILE_TYPE = "INVALID_PROFILE_TYPE"
    STANDARD_ERROR_UNSUPPORTED = "STANDARD_ERROR_UNSUPPORTED"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    REFERENCE_CLOSURE_INVALID = "REFERENCE_CLOSURE_INVALID"
    CERTIFICATE_TYPE_MISMATCH = "CERTIFICATE_TYPE_MISMATCH"
    MISSING_VALUE_REJECTED = "MISSING_VALUE_REJECTED"
    DISCRETE_MAPPING_INVALID = "DISCRETE_MAPPING_INVALID"
    IDENTITY_CONTRACT_INVALID = "IDENTITY_CONTRACT_INVALID"
    UNIT_CONVERSION_CONTRACT_INVALID = "UNIT_CONVERSION_CONTRACT_INVALID"
    COORDINATE_AFFINE_CONTRACT_INVALID = "COORDINATE_AFFINE_CONTRACT_INVALID"
    TEMPORAL_AGGREGATION_CONTRACT_INVALID = "TEMPORAL_AGGREGATION_CONTRACT_INVALID"
    SPATIAL_AGGREGATION_CONTRACT_INVALID = "SPATIAL_AGGREGATION_CONTRACT_INVALID"
    SAMPLING_RESOLUTION_CONTRACT_INVALID = "SAMPLING_RESOLUTION_CONTRACT_INVALID"
    EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID = "EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID"
    COARSE_GRAINING_CONTRACT_INVALID = "COARSE_GRAINING_CONTRACT_INVALID"
    EXACT_KERNEL_EVALUATION_FAILED = "EXACT_KERNEL_EVALUATION_FAILED"
    CLAIM_REFERENCE_MISMATCH = "CLAIM_REFERENCE_MISMATCH"
    CLAIM_METADATA_MISMATCH = "CLAIM_METADATA_MISMATCH"
    CLAIM_KIND_MISMATCH = "CLAIM_KIND_MISMATCH"
    CLAIM_VALUE_MISMATCH = "CLAIM_VALUE_MISMATCH"
    CLAIM_RADIUS_MISMATCH = "CLAIM_RADIUS_MISMATCH"
    CLAIM_BOOLEAN_MISMATCH = "CLAIM_BOOLEAN_MISMATCH"
    CLAIM_MISSING_MISMATCH = "CLAIM_MISSING_MISMATCH"
    CLAIM_SCHEMA_INVALID = "CLAIM_SCHEMA_INVALID"
    CLAIM_RESOURCE_LIMIT = "CLAIM_RESOURCE_LIMIT"
    INTERNAL_CORE_FAILURE = "INTERNAL_CORE_FAILURE"


class ExactTransformCoreValidationErrorV1(ValueError):
    """Typed semantic rejection used by the validating API."""

    def __init__(self, code: CoreTransformFailureCodeV1, detail: str) -> None:
        if type(code) is not CoreTransformFailureCodeV1:
            raise TypeError("core validation error needs an exact failure code")
        _text(detail, "core validation detail")
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")


def _reject(code: CoreTransformFailureCodeV1, detail: str) -> None:
    raise ExactTransformCoreValidationErrorV1(code, detail)


@dataclass(frozen=True, order=True, slots=True)
class CoreScaleRefV1:
    key: str

    def __post_init__(self) -> None:
        _text(self.key, "scale key")


@dataclass(frozen=True, order=True, slots=True)
class CoreObservationRefV1:
    key: str

    def __post_init__(self) -> None:
        _text(self.key, "observation key")


@dataclass(frozen=True, order=True, slots=True)
class CoreComponentRefV1:
    scale_ref: CoreScaleRefV1
    observation_ref: CoreObservationRefV1
    component_index: int

    def __post_init__(self) -> None:
        if type(self.scale_ref) is not CoreScaleRefV1 or type(
            self.observation_ref
        ) is not CoreObservationRefV1:
            raise TypeError("component ref needs exact typed owner refs")
        _integer(self.component_index, "component index", minimum=0)


@dataclass(frozen=True, order=True, slots=True)
class CoreClockBindingV1:
    anchor: str

    def __post_init__(self) -> None:
        _text(self.anchor, "clock anchor")


@dataclass(frozen=True, order=True, slots=True)
class CoreFrameBindingV1:
    anchor: str
    orientation: str | None = None

    def __post_init__(self) -> None:
        _text(self.anchor, "frame anchor")
        if self.orientation is not None:
            _text(self.orientation, "frame orientation")


@dataclass(frozen=True, slots=True)
class CoreUnitBindingV1:
    semantic_key: str
    scale_to_si: Fraction
    offset_to_si: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        _text(self.semantic_key, "unit semantic key")
        _fraction(self.scale_to_si, "unit scale_to_si")
        _fraction(self.offset_to_si, "unit offset_to_si")
        if self.scale_to_si <= 0:
            raise ValueError("unit scale_to_si must be strictly positive")


@dataclass(frozen=True, slots=True)
class CoreQuantityMetadataV1:
    semantic_key: str
    si_exponents: tuple[int, int, int, int, int, int, int]
    unit_binding: CoreUnitBindingV1 | None

    def __post_init__(self) -> None:
        _text(self.semantic_key, "quantity semantic key")
        _tuple(self.si_exponents, "quantity SI exponents")
        if len(self.si_exponents) != 7 or any(
            type(value) is not int or abs(value) > 16
            for value in self.si_exponents
        ):
            raise ValueError("quantity SI exponents are invalid")
        if self.unit_binding is not None and type(
            self.unit_binding
        ) is not CoreUnitBindingV1:
            raise TypeError("quantity unit binding has wrong exact type")


@dataclass(frozen=True, slots=True)
class CoreTemporalSupportV1:
    clock_binding: CoreClockBindingV1
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        if type(self.clock_binding) is not CoreClockBindingV1:
            raise TypeError("temporal support needs an exact clock binding")
        _fraction(self.start, "temporal start")
        _fraction(self.end, "temporal end")
        if self.start > self.end:
            raise ValueError("temporal support start exceeds end")


@dataclass(frozen=True, slots=True)
class CoreSpatialSupportV1:
    frame_binding: CoreFrameBindingV1
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if type(self.frame_binding) is not CoreFrameBindingV1:
            raise TypeError("spatial support needs an exact frame binding")
        _tuple(self.lower, "spatial lower")
        _tuple(self.upper, "spatial upper")
        if not 1 <= len(self.lower) == len(self.upper) <= 4:
            raise ValueError("spatial support needs one to four dimensions")
        for lower, upper in zip(self.lower, self.upper, strict=True):
            _fraction(lower, "spatial lower endpoint")
            _fraction(upper, "spatial upper endpoint")
            if lower > upper:
                raise ValueError("spatial support lower exceeds upper")


@dataclass(frozen=True, slots=True)
class CoreComponentMetadataV1:
    ref: CoreComponentRefV1
    component_label: str
    entity_keys: tuple[str, ...]
    role_keys: tuple[str, ...]
    quantity: CoreQuantityMetadataV1
    axis: CoreComponentAxisV1
    value_role: CoreComponentValueRoleV1
    temporal_support: CoreTemporalSupportV1 | None = None
    spatial_support: CoreSpatialSupportV1 | None = None

    def __post_init__(self) -> None:
        if type(self.ref) is not CoreComponentRefV1:
            raise TypeError("component metadata ref has wrong exact type")
        _text(self.component_label, "component label")
        _sorted_unique_text(self.entity_keys, "component entity keys")
        _sorted_unique_text(self.role_keys, "component role keys")
        if not self.entity_keys or not self.role_keys:
            raise ValueError("component metadata needs entity and role semantics")
        if type(self.quantity) is not CoreQuantityMetadataV1:
            raise TypeError("component quantity metadata has wrong exact type")
        if type(self.axis) is not CoreComponentAxisV1 or type(
            self.value_role
        ) is not CoreComponentValueRoleV1:
            raise TypeError("component axis/value-role type drift")
        if self.temporal_support is not None and type(
            self.temporal_support
        ) is not CoreTemporalSupportV1:
            raise TypeError("component temporal support type drift")
        if self.spatial_support is not None and type(
            self.spatial_support
        ) is not CoreSpatialSupportV1:
            raise TypeError("component spatial support type drift")
        if self.value_role in (
            CoreComponentValueRoleV1.BOOLEAN_CONTROL,
            CoreComponentValueRoleV1.MISSING,
        ):
            if self.quantity.unit_binding is not None or self.quantity.si_exponents != (
                0,
            ) * 7:
                raise ValueError("discrete metadata must be unitless")
        elif self.quantity.unit_binding is None:
            raise ValueError("numeric metadata needs an exact unit binding")
        if self.value_role is CoreComponentValueRoleV1.COORDINATE:
            if (
                self.axis is not CoreComponentAxisV1.COORDINATE
                or self.spatial_support is None
            ):
                raise ValueError("coordinate metadata needs coordinate axis/support")


@dataclass(frozen=True, slots=True)
class CoreComponentValueV1:
    metadata: CoreComponentMetadataV1
    value_kind: CoreComponentValueKindV1
    numeric_value: Fraction | None = None
    uncertainty_model: CoreUncertaintyModelV1 = (
        CoreUncertaintyModelV1.NOT_APPLICABLE
    )
    absolute_radius: Fraction | None = None
    boolean_value: bool | None = None

    def __post_init__(self) -> None:
        if type(self.metadata) is not CoreComponentMetadataV1:
            raise TypeError("component value metadata has wrong exact type")
        if type(self.value_kind) is not CoreComponentValueKindV1 or type(
            self.uncertainty_model
        ) is not CoreUncertaintyModelV1:
            raise TypeError("component value kind/uncertainty type drift")
        if self.value_kind is CoreComponentValueKindV1.NUMERIC:
            _fraction(self.numeric_value, "numeric component value")
            if self.uncertainty_model not in (
                CoreUncertaintyModelV1.ABSOLUTE_BOUND,
                CoreUncertaintyModelV1.STANDARD_ERROR,
            ):
                raise ValueError("numeric component needs an uncertainty model")
            _fraction(self.absolute_radius, "numeric component radius")
            if self.absolute_radius < 0:
                raise ValueError("numeric component radius cannot be negative")
            if self.boolean_value is not None:
                raise ValueError("numeric component cannot carry a boolean")
            if self.metadata.value_role in (
                CoreComponentValueRoleV1.BOOLEAN_CONTROL,
                CoreComponentValueRoleV1.MISSING,
            ):
                raise ValueError("numeric payload disagrees with metadata role")
        elif self.value_kind is CoreComponentValueKindV1.BOOLEAN:
            if (
                type(self.boolean_value) is not bool
                or self.numeric_value is not None
                or self.absolute_radius is not None
                or self.uncertainty_model
                is not CoreUncertaintyModelV1.NOT_APPLICABLE
                or self.metadata.value_role
                is not CoreComponentValueRoleV1.BOOLEAN_CONTROL
            ):
                raise ValueError("boolean component payload/metadata mismatch")
        elif (
            self.numeric_value is not None
            or self.absolute_radius is not None
            or self.boolean_value is not None
            or self.uncertainty_model
            is not CoreUncertaintyModelV1.NOT_APPLICABLE
            or self.metadata.value_role is not CoreComponentValueRoleV1.MISSING
        ):
            raise ValueError("missing component payload/metadata mismatch")

    @property
    def ref(self) -> CoreComponentRefV1:
        return self.metadata.ref


@dataclass(frozen=True, slots=True)
class CoreExactTermV1:
    input_ref: CoreComponentRefV1
    coefficient: Fraction

    def __post_init__(self) -> None:
        if type(self.input_ref) is not CoreComponentRefV1:
            raise TypeError("exact term input ref has wrong type")
        _fraction(self.coefficient, "exact coefficient")


@dataclass(frozen=True, slots=True)
class CoreExactRowV1:
    output_ref: CoreComponentRefV1
    terms: tuple[CoreExactTermV1, ...]
    offset: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if type(self.output_ref) is not CoreComponentRefV1:
            raise TypeError("exact row output ref has wrong type")
        _tuple(self.terms, "exact row terms", nonempty=True)
        if any(type(term) is not CoreExactTermV1 for term in self.terms):
            raise TypeError("exact row contains a non-exact term")
        refs = tuple(term.input_ref for term in self.terms)
        if refs != tuple(sorted(refs)) or len(refs) != len(set(refs)):
            raise ValueError("exact row terms must be ref-sorted and unique")
        if any(term.coefficient == 0 for term in self.terms):
            raise ValueError("zero sparse coefficients are forbidden")
        _fraction(self.offset, "exact row offset")


@dataclass(frozen=True, order=True, slots=True)
class CoreDiscreteMappingV1:
    input_ref: CoreComponentRefV1
    output_ref: CoreComponentRefV1

    def __post_init__(self) -> None:
        if type(self.input_ref) is not CoreComponentRefV1 or type(
            self.output_ref
        ) is not CoreComponentRefV1:
            raise TypeError("discrete mapping refs have wrong exact type")


@dataclass(frozen=True, slots=True)
class CoreAggregateGroupV1:
    input_refs: tuple[CoreComponentRefV1, ...]
    output_refs: tuple[CoreComponentRefV1, ...]
    quotient_label: str

    def __post_init__(self) -> None:
        _tuple(self.input_refs, "aggregate group inputs", nonempty=True)
        _tuple(self.output_refs, "aggregate group outputs", nonempty=True)
        if any(type(ref) is not CoreComponentRefV1 for ref in self.input_refs) or any(
            type(ref) is not CoreComponentRefV1 for ref in self.output_refs
        ):
            raise TypeError("aggregate group contains a non-exact ref")
        if self.input_refs != tuple(sorted(self.input_refs)) or len(
            self.input_refs
        ) != len(set(self.input_refs)):
            raise ValueError("aggregate group inputs must be sorted unique")
        if self.output_refs != tuple(sorted(self.output_refs)) or len(
            self.output_refs
        ) != len(set(self.output_refs)):
            raise ValueError("aggregate group outputs must be sorted unique")
        _text(self.quotient_label, "aggregate quotient label")


@dataclass(frozen=True, order=True, slots=True)
class CoreSamplingGridPointV1:
    input_ref: CoreComponentRefV1
    coordinates: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if type(self.input_ref) is not CoreComponentRefV1:
            raise TypeError("sampling grid point ref has wrong type")
        _tuple(self.coordinates, "sampling coordinates", nonempty=True)
        if not 1 <= len(self.coordinates) <= 4 or any(
            type(value) is not Fraction for value in self.coordinates
        ):
            raise ValueError("sampling coordinates need one to four Fractions")


@dataclass(frozen=True, order=True, slots=True)
class CoreOutputLabelBindingV1:
    input_refs: tuple[CoreComponentRefV1, ...]
    output_ref: CoreComponentRefV1
    output_label: str
    quotient_label: str

    def __post_init__(self) -> None:
        _tuple(self.input_refs, "label binding inputs", nonempty=True)
        if any(type(ref) is not CoreComponentRefV1 for ref in self.input_refs):
            raise TypeError("label binding contains a non-exact input ref")
        if self.input_refs != tuple(sorted(self.input_refs)) or len(
            self.input_refs
        ) != len(set(self.input_refs)):
            raise ValueError("label binding inputs must be sorted unique")
        if type(self.output_ref) is not CoreComponentRefV1:
            raise TypeError("label binding output ref has wrong exact type")
        _text(self.output_label, "label binding output label")
        _text(self.quotient_label, "label binding quotient label")


@dataclass(frozen=True, slots=True)
class CoreIdentityCertificateV1:
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.EXPLICIT_PRESERVE

    def __post_init__(self) -> None:
        if type(self.missing_policy) is not CoreMissingPolicyV1:
            raise TypeError("identity missing policy type drift")


@dataclass(frozen=True, slots=True)
class CoreUnitConversionCertificateV1:
    factor: Fraction
    inverse_factor: Fraction
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.EXPLICIT_PRESERVE

    def __post_init__(self) -> None:
        _fraction(self.factor, "unit conversion factor")
        _fraction(self.inverse_factor, "unit conversion inverse factor")
        if type(self.missing_policy) is not CoreMissingPolicyV1:
            raise TypeError("unit conversion missing policy type drift")


@dataclass(frozen=True, slots=True)
class CoreCoordinateAffineCertificateV1:
    source_frame_binding: CoreFrameBindingV1
    target_frame_binding: CoreFrameBindingV1
    dimension: int
    inverse_rows: tuple[CoreExactRowV1, ...]
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.EXPLICIT_PRESERVE

    def __post_init__(self) -> None:
        if type(self.source_frame_binding) is not CoreFrameBindingV1 or type(
            self.target_frame_binding
        ) is not CoreFrameBindingV1:
            raise TypeError("coordinate certificate frame binding type drift")
        _integer(self.dimension, "coordinate dimension", minimum=1)
        if self.dimension > 4:
            raise ValueError("coordinate dimension exceeds four")
        _tuple(self.inverse_rows, "coordinate inverse rows", nonempty=True)
        if any(type(row) is not CoreExactRowV1 for row in self.inverse_rows):
            raise TypeError("coordinate inverse rows have wrong exact type")
        if type(self.missing_policy) is not CoreMissingPolicyV1:
            raise TypeError("coordinate missing policy type drift")


@dataclass(frozen=True, slots=True)
class CoreTemporalAggregationCertificateV1:
    reducer: CoreReducerV1
    groups: tuple[CoreAggregateGroupV1, ...]
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.REJECT
    boundary_policy: CoreBoundaryPolicyV1 = CoreBoundaryPolicyV1.EXACT_PARTITION
    weighted_mean_basis: CoreWeightedMeanBasisV1 = (
        CoreWeightedMeanBasisV1.CALLER_DECLARED_EXACT_COEFFICIENTS
    )

    def __post_init__(self) -> None:
        if type(self.reducer) is not CoreReducerV1:
            raise TypeError("temporal reducer type drift")
        _tuple(self.groups, "temporal groups", nonempty=True)
        if any(type(group) is not CoreAggregateGroupV1 for group in self.groups):
            raise TypeError("temporal groups contain a non-exact group")
        if type(self.missing_policy) is not CoreMissingPolicyV1 or type(
            self.boundary_policy
        ) is not CoreBoundaryPolicyV1 or type(
            self.weighted_mean_basis
        ) is not CoreWeightedMeanBasisV1:
            raise TypeError("temporal policy type drift")


@dataclass(frozen=True, slots=True)
class CoreSpatialAggregationCertificateV1:
    reducer: CoreReducerV1
    groups: tuple[CoreAggregateGroupV1, ...]
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.REJECT
    boundary_policy: CoreBoundaryPolicyV1 = CoreBoundaryPolicyV1.EXACT_PARTITION
    weighted_mean_basis: CoreWeightedMeanBasisV1 = (
        CoreWeightedMeanBasisV1.CALLER_DECLARED_EXACT_COEFFICIENTS
    )

    def __post_init__(self) -> None:
        if type(self.reducer) is not CoreReducerV1:
            raise TypeError("spatial reducer type drift")
        _tuple(self.groups, "spatial groups", nonempty=True)
        if any(type(group) is not CoreAggregateGroupV1 for group in self.groups):
            raise TypeError("spatial groups contain a non-exact group")
        if type(self.missing_policy) is not CoreMissingPolicyV1 or type(
            self.boundary_policy
        ) is not CoreBoundaryPolicyV1 or type(
            self.weighted_mean_basis
        ) is not CoreWeightedMeanBasisV1:
            raise TypeError("spatial policy type drift")


@dataclass(frozen=True, slots=True)
class CoreSamplingResolutionCertificateV1:
    axis: CoreComponentAxisV1
    selected_inputs: tuple[CoreComponentRefV1, ...]
    discarded_inputs: tuple[CoreComponentRefV1, ...]
    grid_points: tuple[CoreSamplingGridPointV1, ...]
    grid_dimension: int
    grid_frame_binding: CoreFrameBindingV1 | None
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.REJECT
    boundary_policy: CoreBoundaryPolicyV1 = CoreBoundaryPolicyV1.VALID_ONLY

    def __post_init__(self) -> None:
        if type(self.axis) is not CoreComponentAxisV1:
            raise TypeError("sampling axis type drift")
        for values, name, exact_type in (
            (self.selected_inputs, "sampling selected inputs", CoreComponentRefV1),
            (self.discarded_inputs, "sampling discarded inputs", CoreComponentRefV1),
            (self.grid_points, "sampling grid points", CoreSamplingGridPointV1),
        ):
            _tuple(values, name)
            if any(type(value) is not exact_type for value in values):
                raise TypeError(f"{name} exact type drift")
        _integer(self.grid_dimension, "sampling grid dimension", minimum=1)
        if self.grid_dimension > 4:
            raise ValueError("sampling grid dimension exceeds four")
        if self.grid_frame_binding is not None and type(
            self.grid_frame_binding
        ) is not CoreFrameBindingV1:
            raise TypeError("sampling frame binding type drift")
        if type(self.missing_policy) is not CoreMissingPolicyV1 or type(
            self.boundary_policy
        ) is not CoreBoundaryPolicyV1:
            raise TypeError("sampling policy type drift")


@dataclass(frozen=True, slots=True)
class CoreEquivalentSplitMergeCertificateV1:
    direction: CoreSplitMergeDirectionV1
    groups: tuple[CoreAggregateGroupV1, ...]
    inverse_rows: tuple[CoreExactRowV1, ...]
    label_bindings: tuple[CoreOutputLabelBindingV1, ...]
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.REJECT

    def __post_init__(self) -> None:
        if type(self.direction) is not CoreSplitMergeDirectionV1:
            raise TypeError("split/merge direction type drift")
        for values, name, exact_type in (
            (self.groups, "split/merge groups", CoreAggregateGroupV1),
            (self.inverse_rows, "split/merge inverse rows", CoreExactRowV1),
            (self.label_bindings, "split/merge label bindings", CoreOutputLabelBindingV1),
        ):
            _tuple(values, name, nonempty=True)
            if any(type(value) is not exact_type for value in values):
                raise TypeError(f"{name} exact type drift")
        if type(self.missing_policy) is not CoreMissingPolicyV1:
            raise TypeError("split/merge missing policy type drift")


@dataclass(frozen=True, slots=True)
class CoreCoarseGrainingCertificateV1:
    reducer: CoreReducerV1
    groups: tuple[CoreAggregateGroupV1, ...]
    quotient_labels: tuple[str, ...]
    source_commutation_rows: tuple[CoreExactRowV1, ...]
    target_commutation_rows: tuple[CoreExactRowV1, ...]
    label_bindings: tuple[CoreOutputLabelBindingV1, ...]
    missing_policy: CoreMissingPolicyV1 = CoreMissingPolicyV1.REJECT
    boundary_policy: CoreBoundaryPolicyV1 = CoreBoundaryPolicyV1.EXACT_PARTITION
    weighted_mean_basis: CoreWeightedMeanBasisV1 = (
        CoreWeightedMeanBasisV1.CALLER_DECLARED_EXACT_COEFFICIENTS
    )

    def __post_init__(self) -> None:
        if type(self.reducer) is not CoreReducerV1:
            raise TypeError("coarse reducer type drift")
        for values, name, exact_type in (
            (self.groups, "coarse groups", CoreAggregateGroupV1),
            (self.source_commutation_rows, "coarse source commutation rows", CoreExactRowV1),
            (self.target_commutation_rows, "coarse target commutation rows", CoreExactRowV1),
            (self.label_bindings, "coarse label bindings", CoreOutputLabelBindingV1),
        ):
            _tuple(values, name, nonempty=True)
            if any(type(value) is not exact_type for value in values):
                raise TypeError(f"{name} exact type drift")
        _sorted_unique_text(self.quotient_labels, "coarse quotient labels")
        if not self.quotient_labels:
            raise ValueError("coarse quotient labels cannot be empty")
        if type(self.missing_policy) is not CoreMissingPolicyV1 or type(
            self.boundary_policy
        ) is not CoreBoundaryPolicyV1 or type(
            self.weighted_mean_basis
        ) is not CoreWeightedMeanBasisV1:
            raise TypeError("coarse policy type drift")


CoreOperationCertificateV1: TypeAlias = (
    CoreIdentityCertificateV1
    | CoreUnitConversionCertificateV1
    | CoreCoordinateAffineCertificateV1
    | CoreTemporalAggregationCertificateV1
    | CoreSpatialAggregationCertificateV1
    | CoreSamplingResolutionCertificateV1
    | CoreEquivalentSplitMergeCertificateV1
    | CoreCoarseGrainingCertificateV1
)


_CERTIFICATE_TYPE_BY_OPERATION_V1: Final = {
    CoreTransformOperationV1.IDENTITY: CoreIdentityCertificateV1,
    CoreTransformOperationV1.UNIT_CONVERSION: CoreUnitConversionCertificateV1,
    CoreTransformOperationV1.COORDINATE_AFFINE: CoreCoordinateAffineCertificateV1,
    CoreTransformOperationV1.TEMPORAL_AGGREGATION: CoreTemporalAggregationCertificateV1,
    CoreTransformOperationV1.SPATIAL_AGGREGATION: CoreSpatialAggregationCertificateV1,
    CoreTransformOperationV1.SAMPLING_RESOLUTION: CoreSamplingResolutionCertificateV1,
    CoreTransformOperationV1.EQUIVALENT_SPLIT_MERGE: CoreEquivalentSplitMergeCertificateV1,
    CoreTransformOperationV1.COARSE_GRAINING: CoreCoarseGrainingCertificateV1,
}


@dataclass(frozen=True, slots=True)
class CoreExactTransformContractV1:
    operation: CoreTransformOperationV1
    source_scale_ref: CoreScaleRefV1
    target_scale_ref: CoreScaleRefV1
    input_refs: tuple[CoreComponentRefV1, ...]
    output_metadata: tuple[CoreComponentMetadataV1, ...]
    kernel_rows: tuple[CoreExactRowV1, ...]
    discrete_mappings: tuple[CoreDiscreteMappingV1, ...]
    certificate: CoreOperationCertificateV1

    def __post_init__(self) -> None:
        if type(self.operation) is not CoreTransformOperationV1:
            raise TypeError("core transform operation type drift")
        if type(self.source_scale_ref) is not CoreScaleRefV1 or type(
            self.target_scale_ref
        ) is not CoreScaleRefV1:
            raise TypeError("core transform scale ref type drift")
        for value, name in (
            (self.input_refs, "contract input refs"),
            (self.output_metadata, "contract output metadata"),
            (self.kernel_rows, "contract kernel rows"),
            (self.discrete_mappings, "contract discrete mappings"),
        ):
            _tuple(value, name)
        if not self.input_refs or not self.output_metadata:
            raise ValueError("core transform contract cannot be empty")
        if any(type(ref) is not CoreComponentRefV1 for ref in self.input_refs):
            raise TypeError("contract input ref exact type drift")
        if any(
            type(item) is not CoreComponentMetadataV1
            for item in self.output_metadata
        ):
            raise TypeError("contract output metadata exact type drift")
        if any(type(row) is not CoreExactRowV1 for row in self.kernel_rows):
            raise TypeError("contract kernel row exact type drift")
        if any(
            type(item) is not CoreDiscreteMappingV1
            for item in self.discrete_mappings
        ):
            raise TypeError("contract discrete mapping exact type drift")


@dataclass(frozen=True, slots=True)
class ExactTransformCoreProfileV1:
    source_components: tuple[CoreComponentValueV1, ...]
    contract: CoreExactTransformContractV1

    def __post_init__(self) -> None:
        _tuple(self.source_components, "core source components", nonempty=True)
        if any(
            type(component) is not CoreComponentValueV1
            for component in self.source_components
        ):
            raise TypeError("core source components have wrong exact type")
        if type(self.contract) is not CoreExactTransformContractV1:
            raise TypeError("core profile contract has wrong exact type")


@dataclass(frozen=True, slots=True)
class ExactTransformCoreFailureV1:
    code: CoreTransformFailureCodeV1
    detail: str

    def __post_init__(self) -> None:
        if type(self.code) is not CoreTransformFailureCodeV1:
            raise TypeError("core failure code type drift")
        _text(self.detail, "core failure detail")


@dataclass(frozen=True, slots=True)
class ExactTransformCoreResultV1:
    """Cooperative computation DTO, never an authority or proof receipt."""

    disposition: CoreTransformDispositionV1
    output_components: tuple[CoreComponentValueV1, ...]
    failure: ExactTransformCoreFailureV1 | None

    def __post_init__(self) -> None:
        if type(self.disposition) is not CoreTransformDispositionV1:
            raise TypeError("core result disposition type drift")
        _tuple(self.output_components, "core result outputs")
        if any(
            type(component) is not CoreComponentValueV1
            for component in self.output_components
        ):
            raise TypeError("core result output type drift")
        if self.disposition is CoreTransformDispositionV1.COMPLETE:
            if not self.output_components or self.failure is not None:
                raise ValueError("complete core result needs outputs only")
        elif self.output_components or type(
            self.failure
        ) is not ExactTransformCoreFailureV1:
            raise ValueError("abstaining core result must be output-empty")


@dataclass(frozen=True, slots=True)
class _CoreLimitsV1:
    maximum_components: int = 65_536
    maximum_rows: int = 65_536
    maximum_nonzeros: int = 1_000_000
    maximum_certificate_memberships: int = 1_000_000
    maximum_auxiliary_rows: int = 65_536
    maximum_auxiliary_nonzeros: int = 1_000_000
    maximum_fraction_bit_length: int = 4_096
    maximum_exact_operations: int = 1_000_000
    maximum_spatial_dimension: int = 4
    maximum_tree_nodes: int = 3_000_000
    maximum_tuple_items: int = 1_000_000
    maximum_total_tuple_items: int = 3_000_000
    maximum_text_characters_per_value: int = 4_096
    maximum_text_utf8_bytes_per_value: int = 16_384
    maximum_total_text_utf8_bytes: int = 3_000_000
    maximum_integer_bit_length: int = 4_096


_LIMITS_V1: Final = _CoreLimitsV1()


@dataclass(slots=True)
class _ArithmeticBudgetV1:
    operations: int = 0

    def _charge(self, value: Fraction) -> Fraction:
        self.operations += 1
        if self.operations > _LIMITS_V1.maximum_exact_operations:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "exact_operations")
        _check_fraction_bits(value)
        return value

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self._charge(left + right)

    def multiply(self, left: Fraction, right: Fraction) -> Fraction:
        return self._charge(left * right)

    def divide(self, left: Fraction, right: Fraction) -> Fraction:
        return self._charge(left / right)


def _check_fraction_bits(value: Fraction) -> None:
    if type(value) is not Fraction:
        _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "fraction_exact_type")
    try:
        numerator = value.numerator
        denominator = value.denominator
    except (AttributeError, TypeError) as exc:
        raise ExactTransformCoreValidationErrorV1(
            CoreTransformFailureCodeV1.SCHEMA_INVALID,
            "fraction_state_unavailable",
        ) from exc
    if type(numerator) is not int or type(denominator) is not int:
        _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "fraction_integer_state")
    if (
        numerator.bit_length() > _LIMITS_V1.maximum_fraction_bit_length
        or denominator.bit_length() > _LIMITS_V1.maximum_fraction_bit_length
    ):
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "fraction_bit_length")
    if denominator <= 0 or gcd(abs(numerator), denominator) != 1:
        _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "fraction_canonical_state")


_CORE_TREE_DATACLASS_TYPES_V1: Final = frozenset(
    {
        CoreAggregateGroupV1,
        CoreClockBindingV1,
        CoreCoarseGrainingCertificateV1,
        CoreComponentMetadataV1,
        CoreComponentRefV1,
        CoreComponentValueV1,
        CoreCoordinateAffineCertificateV1,
        CoreDiscreteMappingV1,
        CoreEquivalentSplitMergeCertificateV1,
        CoreExactRowV1,
        CoreExactTermV1,
        CoreExactTransformContractV1,
        CoreFrameBindingV1,
        CoreIdentityCertificateV1,
        CoreObservationRefV1,
        CoreOutputLabelBindingV1,
        CoreQuantityMetadataV1,
        CoreSamplingGridPointV1,
        CoreSamplingResolutionCertificateV1,
        CoreScaleRefV1,
        CoreSpatialAggregationCertificateV1,
        CoreSpatialSupportV1,
        CoreTemporalAggregationCertificateV1,
        CoreTemporalSupportV1,
        CoreUnitBindingV1,
        CoreUnitConversionCertificateV1,
        ExactTransformCoreProfileV1,
    }
)
_CORE_TREE_ENUM_TYPES_V1: Final = frozenset(
    {
        CoreBoundaryPolicyV1,
        CoreComponentAxisV1,
        CoreComponentValueKindV1,
        CoreComponentValueRoleV1,
        CoreMissingPolicyV1,
        CoreReducerV1,
        CoreSplitMergeDirectionV1,
        CoreTransformOperationV1,
        CoreUncertaintyModelV1,
        CoreWeightedMeanBasisV1,
    }
)


def _bounded_closed_tree_nodes(
    root: object,
) -> tuple[tuple[object, ...], tuple[Fraction, ...]]:
    dataclass_nodes: list[object] = []
    fraction_values: list[Fraction] = []
    stack: list[tuple[object, bool]] = [(root, False)]
    active_containers: set[int] = set()
    nodes = 0
    total_tuple_items = 0
    total_text_bytes = 0
    while stack:
        value, exiting = stack.pop()
        exact_type = type(value)
        is_container = exact_type is tuple or exact_type in _CORE_TREE_DATACLASS_TYPES_V1
        if exiting:
            identity = id(value)
            active_containers.remove(identity)
            continue
        nodes += 1
        if nodes > _LIMITS_V1.maximum_tree_nodes:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "tree_nodes")
        if is_container:
            identity = id(value)
            if identity in active_containers:
                _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "tree_cycle")
            active_containers.add(identity)
            stack.append((value, True))
        if exact_type is Fraction:
            fraction_values.append(value)
        elif exact_type is tuple:
            length = len(value)
            if length > _LIMITS_V1.maximum_tuple_items:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "tuple_items")
            total_tuple_items += length
            if total_tuple_items > _LIMITS_V1.maximum_total_tuple_items:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "total_tuple_items")
            stack.extend((item, False) for item in reversed(value))
        elif exact_type in _CORE_TREE_DATACLASS_TYPES_V1:
            dataclass_nodes.append(value)
            try:
                children = tuple(
                    getattr(value, field.name) for field in reversed(fields(value))
                )
            except (AttributeError, TypeError) as exc:
                raise ExactTransformCoreValidationErrorV1(
                    CoreTransformFailureCodeV1.SCHEMA_INVALID,
                    "dataclass_field_unavailable",
                ) from exc
            stack.extend((child, False) for child in children)
        elif exact_type in _CORE_TREE_ENUM_TYPES_V1:
            if not any(value is member for member in exact_type):
                _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "enum_member")
        elif exact_type in (bool, type(None)):
            continue
        elif exact_type is str:
            if len(value) > _LIMITS_V1.maximum_text_characters_per_value:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "text_characters")
            try:
                encoded_length = len(value.encode("utf-8"))
            except UnicodeError as exc:
                raise ExactTransformCoreValidationErrorV1(
                    CoreTransformFailureCodeV1.SCHEMA_INVALID,
                    "text_utf8_invalid",
                ) from exc
            if encoded_length > _LIMITS_V1.maximum_text_utf8_bytes_per_value:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "text_utf8_bytes")
            total_text_bytes += encoded_length
            if total_text_bytes > _LIMITS_V1.maximum_total_text_utf8_bytes:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "total_text_utf8_bytes")
        elif exact_type is int:
            if value.bit_length() > _LIMITS_V1.maximum_integer_bit_length:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "integer_bit_length")
        else:
            _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "nonclosed_tree")
    return tuple(dataclass_nodes), tuple(fraction_values)


def _check_fraction_collection(values: tuple[Fraction, ...]) -> None:
    for value in values:
        _check_fraction_bits(value)


def _revalidate_dataclass_invariants(nodes: tuple[object, ...]) -> None:
    for value in reversed(nodes):
        method = type(value).__dict__.get("__post_init__")
        if method is None:
            _reject(
                CoreTransformFailureCodeV1.SCHEMA_INVALID,
                "missing_fresh_invariant_validator",
            )
        try:
            method(value)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ExactTransformCoreValidationErrorV1(
                CoreTransformFailureCodeV1.SCHEMA_INVALID,
                "fresh_invariant:" + type(value).__name__,
            ) from exc


def _preflight_standard_error(nodes: tuple[object, ...], *, detail: str) -> None:
    for component in nodes:
        if (
            type(component) is CoreComponentValueV1
            and
            component.uncertainty_model
            is CoreUncertaintyModelV1.STANDARD_ERROR
        ):
            _reject(
                CoreTransformFailureCodeV1.STANDARD_ERROR_UNSUPPORTED,
                detail,
            )


def _groups_from_certificate(
    certificate: CoreOperationCertificateV1,
) -> tuple[CoreAggregateGroupV1, ...]:
    if type(certificate) in (
        CoreTemporalAggregationCertificateV1,
        CoreSpatialAggregationCertificateV1,
        CoreEquivalentSplitMergeCertificateV1,
        CoreCoarseGrainingCertificateV1,
    ):
        return certificate.groups
    return ()


def _auxiliary_row_collections_from_certificate(
    certificate: CoreOperationCertificateV1,
) -> tuple[tuple[CoreExactRowV1, ...], ...]:
    if type(certificate) is CoreCoordinateAffineCertificateV1:
        return (certificate.inverse_rows,)
    if type(certificate) is CoreEquivalentSplitMergeCertificateV1:
        return (certificate.inverse_rows,)
    if type(certificate) is CoreCoarseGrainingCertificateV1:
        return (
            certificate.source_commutation_rows,
            certificate.target_commutation_rows,
        )
    return ()


def _shallow_profile_resource_preflight(
    profile: ExactTransformCoreProfileV1,
) -> None:
    """Bound declared collection shapes before recursively visiting members."""

    limits = _LIMITS_V1
    try:
        source_components = profile.source_components
        contract = profile.contract
    except (AttributeError, TypeError):
        return
    if type(source_components) is tuple and len(
        source_components
    ) > limits.maximum_components:
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "component_count")
    if type(contract) is not CoreExactTransformContractV1:
        return
    try:
        input_refs = contract.input_refs
        output_metadata = contract.output_metadata
        kernel_rows = contract.kernel_rows
        discrete_mappings = contract.discrete_mappings
        certificate = contract.certificate
    except (AttributeError, TypeError):
        return
    if type(input_refs) is tuple and len(input_refs) > limits.maximum_components:
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "component_count")
    if type(output_metadata) is tuple:
        if len(output_metadata) > limits.maximum_components or (
            type(source_components) is tuple
            and len(source_components) + len(output_metadata)
            > limits.maximum_components
        ):
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "component_count")
    if type(discrete_mappings) is tuple and len(
        discrete_mappings
    ) > limits.maximum_components:
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "component_count")
    if type(kernel_rows) is tuple:
        if len(kernel_rows) > limits.maximum_rows:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "row_count")
        nonzeros = 0
        for exact_row in kernel_rows:
            if type(exact_row) is not CoreExactRowV1:
                continue
            try:
                terms = exact_row.terms
            except (AttributeError, TypeError):
                continue
            if type(terms) is tuple:
                nonzeros += len(terms)
                if nonzeros > limits.maximum_nonzeros:
                    _reject(
                        CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                        "nonzero_count",
                    )

    memberships = 0
    groups: object = ()
    if type(certificate) in (
        CoreTemporalAggregationCertificateV1,
        CoreSpatialAggregationCertificateV1,
        CoreEquivalentSplitMergeCertificateV1,
        CoreCoarseGrainingCertificateV1,
    ):
        try:
            groups = certificate.groups
        except (AttributeError, TypeError):
            groups = ()
    if type(groups) is tuple:
        if len(groups) > limits.maximum_certificate_memberships // 2:
            _reject(
                CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                "certificate_memberships",
            )
        for aggregate_group in groups:
            if type(aggregate_group) is not CoreAggregateGroupV1:
                continue
            try:
                group_inputs = aggregate_group.input_refs
                group_outputs = aggregate_group.output_refs
            except (AttributeError, TypeError):
                continue
            if type(group_inputs) is tuple and type(group_outputs) is tuple:
                memberships += len(group_inputs) + len(group_outputs)
                if memberships > limits.maximum_certificate_memberships:
                    _reject(
                        CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                        "certificate_memberships",
                    )
    if type(certificate) is CoreSamplingResolutionCertificateV1:
        for name in ("selected_inputs", "discarded_inputs", "grid_points"):
            try:
                collection = getattr(certificate, name)
            except (AttributeError, TypeError):
                continue
            if type(collection) is tuple:
                memberships += len(collection)
                if memberships > limits.maximum_certificate_memberships:
                    _reject(
                        CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                        "certificate_memberships",
                    )
    if type(certificate) in (
        CoreEquivalentSplitMergeCertificateV1,
        CoreCoarseGrainingCertificateV1,
    ):
        try:
            label_bindings = certificate.label_bindings
        except (AttributeError, TypeError):
            label_bindings = ()
        if type(label_bindings) is tuple:
            if len(label_bindings) > limits.maximum_certificate_memberships // 2:
                _reject(
                    CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                    "certificate_memberships",
                )
            for binding in label_bindings:
                if type(binding) is not CoreOutputLabelBindingV1:
                    continue
                try:
                    binding_inputs = binding.input_refs
                except (AttributeError, TypeError):
                    continue
                if type(binding_inputs) is tuple:
                    memberships += len(binding_inputs) + 1
                    if memberships > limits.maximum_certificate_memberships:
                        _reject(
                            CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                            "certificate_memberships",
                        )

    auxiliary_rows = 0
    auxiliary_nonzeros = 0
    try:
        auxiliary_collections = _auxiliary_row_collections_from_certificate(
            certificate
        )
    except (AttributeError, TypeError):
        auxiliary_collections = ()
    for collection in auxiliary_collections:
        if type(collection) is not tuple:
            continue
        auxiliary_rows += len(collection)
        if auxiliary_rows > limits.maximum_auxiliary_rows:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "auxiliary_rows")
        for exact_row in collection:
            if type(exact_row) is not CoreExactRowV1:
                continue
            try:
                terms = exact_row.terms
            except (AttributeError, TypeError):
                continue
            if type(terms) is tuple:
                auxiliary_nonzeros += len(terms)
                if auxiliary_nonzeros > limits.maximum_auxiliary_nonzeros:
                    _reject(
                        CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                        "auxiliary_nonzeros",
                    )


def _resource_preflight(profile: ExactTransformCoreProfileV1) -> None:
    contract = profile.contract
    components = len(profile.source_components) + len(contract.output_metadata)
    rows = len(contract.kernel_rows)
    limits = _LIMITS_V1
    if components > limits.maximum_components:
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "component_count")
    if rows > limits.maximum_rows:
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "row_count")
    nonzeros = 0
    for exact_row in contract.kernel_rows:
        nonzeros += len(exact_row.terms)
        if nonzeros > limits.maximum_nonzeros:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "nonzero_count")
    groups = _groups_from_certificate(contract.certificate)
    memberships = 0
    for aggregate_group in groups:
        memberships += len(aggregate_group.input_refs) + len(aggregate_group.output_refs)
        if memberships > limits.maximum_certificate_memberships:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "certificate_memberships")
    if type(contract.certificate) is CoreSamplingResolutionCertificateV1:
        memberships += (
            len(contract.certificate.selected_inputs)
            + len(contract.certificate.discarded_inputs)
            + len(contract.certificate.grid_points)
        )
    if type(contract.certificate) in (
        CoreEquivalentSplitMergeCertificateV1,
        CoreCoarseGrainingCertificateV1,
    ):
        for binding in contract.certificate.label_bindings:
            memberships += len(binding.input_refs) + 1
            if memberships > limits.maximum_certificate_memberships:
                _reject(
                    CoreTransformFailureCodeV1.RESOURCE_LIMIT,
                    "certificate_memberships",
                )
    if memberships > limits.maximum_certificate_memberships:
        _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "certificate_memberships")
    auxiliary_rows = 0
    auxiliary_nonzeros = 0
    for collection in _auxiliary_row_collections_from_certificate(
        contract.certificate
    ):
        auxiliary_rows += len(collection)
        if auxiliary_rows > limits.maximum_auxiliary_rows:
            _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "auxiliary_rows")
        for exact_row in collection:
            auxiliary_nonzeros += len(exact_row.terms)
            if auxiliary_nonzeros > limits.maximum_auxiliary_nonzeros:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "auxiliary_nonzeros")


def _canonical_ref_tuple(
    refs: tuple[CoreComponentRefV1, ...],
    *,
    name: str,
    nonempty: bool = False,
) -> None:
    if nonempty and not refs:
        _reject(CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID, name)
    if refs != tuple(sorted(refs)) or len(refs) != len(set(refs)):
        _reject(CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID, name)


def _metadata_equal_except_ref(
    left: CoreComponentMetadataV1,
    right: CoreComponentMetadataV1,
) -> bool:
    return all(
        getattr(left, field.name) == getattr(right, field.name)
        for field in fields(CoreComponentMetadataV1)
        if field.name != "ref"
    )


def _same_semantic_series(
    left: CoreComponentMetadataV1,
    right: CoreComponentMetadataV1,
    *,
    ignore_axis_support: CoreComponentAxisV1 | None = None,
) -> bool:
    common = (
        left.component_label == right.component_label
        and left.entity_keys == right.entity_keys
        and left.role_keys == right.role_keys
        and left.quantity == right.quantity
        and left.value_role == right.value_role
    )
    if not common:
        return False
    if ignore_axis_support is CoreComponentAxisV1.TEMPORAL:
        return left.spatial_support == right.spatial_support
    if ignore_axis_support is CoreComponentAxisV1.SPATIAL:
        return left.temporal_support == right.temporal_support
    return (
        left.axis == right.axis
        and left.temporal_support == right.temporal_support
        and left.spatial_support == right.spatial_support
    )


def _missing_policy(certificate: CoreOperationCertificateV1) -> CoreMissingPolicyV1:
    value = certificate.missing_policy
    if type(value) is not CoreMissingPolicyV1:
        _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "missing_policy")
    return value


_OPERATION_FAILURE_CODE_V1: Final = {
    CoreTransformOperationV1.IDENTITY: CoreTransformFailureCodeV1.IDENTITY_CONTRACT_INVALID,
    CoreTransformOperationV1.UNIT_CONVERSION: CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID,
    CoreTransformOperationV1.COORDINATE_AFFINE: CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID,
    CoreTransformOperationV1.TEMPORAL_AGGREGATION: CoreTransformFailureCodeV1.TEMPORAL_AGGREGATION_CONTRACT_INVALID,
    CoreTransformOperationV1.SPATIAL_AGGREGATION: CoreTransformFailureCodeV1.SPATIAL_AGGREGATION_CONTRACT_INVALID,
    CoreTransformOperationV1.SAMPLING_RESOLUTION: CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID,
    CoreTransformOperationV1.EQUIVALENT_SPLIT_MERGE: CoreTransformFailureCodeV1.EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID,
    CoreTransformOperationV1.COARSE_GRAINING: CoreTransformFailureCodeV1.COARSE_GRAINING_CONTRACT_INVALID,
}


def _validate_certificate_canonical(
    contract: CoreExactTransformContractV1,
) -> None:
    certificate = contract.certificate
    code = _OPERATION_FAILURE_CODE_V1[contract.operation]
    groups = _groups_from_certificate(certificate)
    if groups:
        keys = tuple(
            (group.input_refs, group.output_refs, group.quotient_label)
            for group in groups
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            _reject(code, "certificate_groups_canonical_unique")

    row_collections: tuple[tuple[CoreExactRowV1, ...], ...] = ()
    if type(certificate) is CoreCoordinateAffineCertificateV1:
        row_collections = (certificate.inverse_rows,)
    elif type(certificate) is CoreEquivalentSplitMergeCertificateV1:
        row_collections = (certificate.inverse_rows,)
    elif type(certificate) is CoreCoarseGrainingCertificateV1:
        row_collections = (
            certificate.source_commutation_rows,
            certificate.target_commutation_rows,
        )
    for rows in row_collections:
        outputs = tuple(row.output_ref for row in rows)
        if (
            rows != tuple(sorted(rows, key=lambda row: row.output_ref))
            or len(outputs) != len(set(outputs))
        ):
            _reject(code, "certificate_rows_canonical_unique")
    label_bindings: tuple[CoreOutputLabelBindingV1, ...] = ()
    if type(certificate) in (
        CoreEquivalentSplitMergeCertificateV1,
        CoreCoarseGrainingCertificateV1,
    ):
        label_bindings = certificate.label_bindings
    if label_bindings:
        binding_outputs = tuple(binding.output_ref for binding in label_bindings)
        if (
            label_bindings
            != tuple(sorted(label_bindings, key=lambda binding: binding.output_ref))
            or len(binding_outputs) != len(set(binding_outputs))
        ):
            _reject(code, "certificate_label_bindings_canonical_unique")


def _validate_output_label_bindings(
    groups: tuple[CoreAggregateGroupV1, ...],
    bindings: tuple[CoreOutputLabelBindingV1, ...],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    code: CoreTransformFailureCodeV1,
) -> None:
    by_output = {binding.output_ref: binding for binding in bindings}
    expected_outputs = {
        output_ref for aggregate_group in groups for output_ref in aggregate_group.output_refs
    }
    if len(by_output) != len(bindings) or set(by_output) != expected_outputs:
        _reject(code, "label_binding_output_exact_coverage")
    for aggregate_group in groups:
        for output_ref in aggregate_group.output_refs:
            binding = by_output[output_ref]
            if (
                binding.input_refs != aggregate_group.input_refs
                or binding.quotient_label != aggregate_group.quotient_label
                or binding.output_label != outputs[output_ref].component_label
            ):
                _reject(code, "label_binding_group_or_metadata_mismatch")


def _common_validation(
    profile: ExactTransformCoreProfileV1,
) -> tuple[
    dict[CoreComponentRefV1, CoreComponentValueV1],
    dict[CoreComponentRefV1, CoreComponentMetadataV1],
]:
    contract = profile.contract
    expected_certificate_type = _CERTIFICATE_TYPE_BY_OPERATION_V1[contract.operation]
    if type(contract.certificate) is not expected_certificate_type:
        _reject(
            CoreTransformFailureCodeV1.CERTIFICATE_TYPE_MISMATCH,
            contract.operation.value,
        )
    _validate_certificate_canonical(contract)
    source_refs = tuple(component.ref for component in profile.source_components)
    output_refs = tuple(metadata.ref for metadata in contract.output_metadata)
    _canonical_ref_tuple(source_refs, name="source_components", nonempty=True)
    _canonical_ref_tuple(contract.input_refs, name="contract_inputs", nonempty=True)
    _canonical_ref_tuple(output_refs, name="output_metadata", nonempty=True)
    if source_refs != contract.input_refs:
        _reject(
            CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
            "source_input_exact_coverage",
        )
    if any(
        ref.scale_ref != contract.source_scale_ref for ref in contract.input_refs
    ) or any(
        ref.scale_ref != contract.target_scale_ref for ref in output_refs
    ):
        _reject(
            CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
            "source_target_scale_closure",
        )
    if contract.source_scale_ref == contract.target_scale_ref:
        _reject(
            CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
            "source_target_scale_must_differ",
        )
    row_outputs = tuple(row.output_ref for row in contract.kernel_rows)
    mapping_outputs = tuple(item.output_ref for item in contract.discrete_mappings)
    if contract.kernel_rows != tuple(
        sorted(contract.kernel_rows, key=lambda row: row.output_ref)
    ) or len(row_outputs) != len(set(row_outputs)):
        _reject(CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID, "kernel_rows")
    if contract.discrete_mappings != tuple(sorted(contract.discrete_mappings)):
        _reject(
            CoreTransformFailureCodeV1.DISCRETE_MAPPING_INVALID,
            "mapping_canonical_order",
        )
    if len(mapping_outputs) != len(set(mapping_outputs)) or set(
        row_outputs
    ).intersection(mapping_outputs):
        _reject(
            CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
            "output_construction_overlap",
        )
    if set(row_outputs).union(mapping_outputs) != set(output_refs):
        _reject(
            CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
            "output_exact_coverage",
        )
    source = {component.ref: component for component in profile.source_components}
    outputs = {metadata.ref: metadata for metadata in contract.output_metadata}
    numeric_inputs = {
        ref
        for ref, component in source.items()
        if component.value_kind is CoreComponentValueKindV1.NUMERIC
    }
    discrete_inputs = set(source) - numeric_inputs
    used_numeric_inputs: set[CoreComponentRefV1] = set()
    for row in contract.kernel_rows:
        if row.output_ref not in outputs:
            _reject(
                CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
                "row_output_unresolved",
            )
        for term in row.terms:
            if term.input_ref not in numeric_inputs:
                _reject(
                    CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
                    "row_input_not_numeric_source",
                )
            used_numeric_inputs.add(term.input_ref)
    expected_used_numeric_inputs = numeric_inputs
    if contract.operation is CoreTransformOperationV1.SAMPLING_RESOLUTION:
        sampling_certificate = contract.certificate
        if type(sampling_certificate) is not CoreSamplingResolutionCertificateV1:
            _reject(
                CoreTransformFailureCodeV1.CERTIFICATE_TYPE_MISMATCH,
                contract.operation.value,
            )
        expected_used_numeric_inputs = set(sampling_certificate.selected_inputs)
    if used_numeric_inputs != expected_used_numeric_inputs:
        _reject(
            CoreTransformFailureCodeV1.REFERENCE_CLOSURE_INVALID,
            "numeric_input_exact_use",
        )
    mapping_inputs = tuple(item.input_ref for item in contract.discrete_mappings)
    if len(mapping_inputs) != len(set(mapping_inputs)) or set(
        mapping_inputs
    ) != discrete_inputs:
        _reject(
            CoreTransformFailureCodeV1.DISCRETE_MAPPING_INVALID,
            "discrete_input_exact_coverage",
        )
    if (
        _missing_policy(contract.certificate) is CoreMissingPolicyV1.REJECT
        and any(
            source[ref].value_kind is CoreComponentValueKindV1.MISSING
            for ref in discrete_inputs
        )
    ):
        _reject(
            CoreTransformFailureCodeV1.MISSING_VALUE_REJECTED,
            contract.operation.value,
        )
    for mapping in contract.discrete_mappings:
        if mapping.output_ref not in outputs:
            _reject(
                CoreTransformFailureCodeV1.DISCRETE_MAPPING_INVALID,
                "mapping_output_unresolved",
            )
        left = source.get(mapping.input_ref)
        if left is None or not _metadata_equal_except_ref(
            left.metadata, outputs[mapping.output_ref]
        ):
            _reject(
                CoreTransformFailureCodeV1.DISCRETE_MAPPING_INVALID,
                "discrete_metadata_copy",
            )
    return source, outputs


def _numeric_source_refs(
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
) -> tuple[CoreComponentRefV1, ...]:
    return tuple(
        sorted(
            ref
            for ref, component in source.items()
            if component.value_kind is CoreComponentValueKindV1.NUMERIC
        )
    )


def _numeric_output_refs(
    contract: CoreExactTransformContractV1,
) -> tuple[CoreComponentRefV1, ...]:
    discrete = {item.output_ref for item in contract.discrete_mappings}
    return tuple(
        metadata.ref
        for metadata in contract.output_metadata
        if metadata.ref not in discrete
    )


def _one_to_one_rows(
    rows: tuple[CoreExactRowV1, ...],
    source_refs: tuple[CoreComponentRefV1, ...],
    target_refs: tuple[CoreComponentRefV1, ...],
    *,
    factor: Fraction,
    offsets: dict[CoreComponentRefV1, Fraction] | None = None,
) -> bool:
    if len(rows) != len(source_refs) or len(rows) != len(target_refs):
        return False
    used: list[CoreComponentRefV1] = []
    for row in rows:
        expected_offset = Fraction(0) if offsets is None else offsets[row.output_ref]
        if (
            len(row.terms) != 1
            or row.terms[0].coefficient != factor
            or row.offset != expected_offset
        ):
            return False
        used.append(row.terms[0].input_ref)
    return set(used) == set(source_refs) and {
        row.output_ref for row in rows
    } == set(target_refs)


def _validate_identity(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
) -> None:
    certificate = contract.certificate
    if type(certificate) is not CoreIdentityCertificateV1 or certificate.missing_policy is not CoreMissingPolicyV1.EXPLICIT_PRESERVE:
        _reject(CoreTransformFailureCodeV1.IDENTITY_CONTRACT_INVALID, "certificate")
    numeric_source = _numeric_source_refs(source)
    numeric_target = _numeric_output_refs(contract)
    if not _one_to_one_rows(
        contract.kernel_rows,
        numeric_source,
        numeric_target,
        factor=Fraction(1),
    ):
        _reject(CoreTransformFailureCodeV1.IDENTITY_CONTRACT_INVALID, "kernel")
    for row in contract.kernel_rows:
        if not _metadata_equal_except_ref(
            source[row.terms[0].input_ref].metadata,
            outputs[row.output_ref],
        ):
            _reject(CoreTransformFailureCodeV1.IDENTITY_CONTRACT_INVALID, "metadata")


def _validate_unit_conversion(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    budget: _ArithmeticBudgetV1,
) -> None:
    certificate = contract.certificate
    if type(certificate) is not CoreUnitConversionCertificateV1:
        _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "certificate")
    factor = certificate.factor
    inverse = certificate.inverse_factor
    if (
        certificate.missing_policy is not CoreMissingPolicyV1.EXPLICIT_PRESERVE
        or factor <= 0
        or inverse <= 0
        or budget.multiply(factor, inverse) != 1
    ):
        _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "factor_inverse")
    numeric_source = _numeric_source_refs(source)
    numeric_target = _numeric_output_refs(contract)
    if not numeric_source or not numeric_target:
        _reject(
            CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID,
            "numeric_witness_required",
        )
    offsets: dict[CoreComponentRefV1, Fraction] = {}
    for row in contract.kernel_rows:
        if len(row.terms) != 1:
            _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "one_to_one")
        left = source[row.terms[0].input_ref].metadata
        right = outputs[row.output_ref]
        source_unit = left.quantity.unit_binding
        target_unit = right.quantity.unit_binding
        if source_unit is None or target_unit is None:
            _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "unit_missing")
        if (
            source_unit == target_unit
            or source_unit.semantic_key == target_unit.semantic_key
            or
            left.quantity.semantic_key != right.quantity.semantic_key
            or left.quantity.si_exponents != right.quantity.si_exponents
            or left.axis != right.axis
            or left.value_role != right.value_role
            or left.component_label != right.component_label
            or left.entity_keys != right.entity_keys
            or left.role_keys != right.role_keys
            or left.temporal_support != right.temporal_support
            or left.spatial_support != right.spatial_support
        ):
            _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "metadata")
        if budget.multiply(factor, target_unit.scale_to_si) != source_unit.scale_to_si:
            _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "scale_to_si")
        offsets[row.output_ref] = budget.divide(
            budget.add(source_unit.offset_to_si, -target_unit.offset_to_si),
            target_unit.scale_to_si,
        )
    if not _one_to_one_rows(
        contract.kernel_rows,
        numeric_source,
        numeric_target,
        factor=factor,
        offsets=offsets,
    ):
        _reject(CoreTransformFailureCodeV1.UNIT_CONVERSION_CONTRACT_INVALID, "kernel")


def _normalized_expression(
    coefficients: dict[CoreComponentRefV1, Fraction],
    offset: Fraction,
) -> tuple[tuple[tuple[CoreComponentRefV1, Fraction], ...], Fraction]:
    return (
        tuple(sorted((ref, value) for ref, value in coefficients.items() if value)),
        offset,
    )


def _compose_rows(
    outer: tuple[CoreExactRowV1, ...],
    inner: tuple[CoreExactRowV1, ...],
    budget: _ArithmeticBudgetV1,
) -> dict[
    CoreComponentRefV1,
    tuple[tuple[tuple[CoreComponentRefV1, Fraction], ...], Fraction],
] | None:
    inner_by_output = {row.output_ref: row for row in inner}
    if len(inner_by_output) != len(inner):
        return None
    result: dict[
        CoreComponentRefV1,
        tuple[tuple[tuple[CoreComponentRefV1, Fraction], ...], Fraction],
    ] = {}
    for outer_row in outer:
        coefficients: dict[CoreComponentRefV1, Fraction] = {}
        offset = outer_row.offset
        for outer_term in outer_row.terms:
            inner_row = inner_by_output.get(outer_term.input_ref)
            if inner_row is None:
                return None
            offset = budget.add(
                offset,
                budget.multiply(outer_term.coefficient, inner_row.offset),
            )
            for inner_term in inner_row.terms:
                coefficients[inner_term.input_ref] = budget.add(
                    coefficients.get(inner_term.input_ref, Fraction(0)),
                    budget.multiply(
                        outer_term.coefficient, inner_term.coefficient
                    ),
                )
        result[outer_row.output_ref] = _normalized_expression(coefficients, offset)
    return result


def _identity_expressions(
    refs: tuple[CoreComponentRefV1, ...],
) -> dict[
    CoreComponentRefV1,
    tuple[tuple[tuple[CoreComponentRefV1, Fraction], ...], Fraction],
]:
    return {ref: (((ref, Fraction(1)),), Fraction(0)) for ref in refs}


def _rows_two_sided_inverse(
    forward: tuple[CoreExactRowV1, ...],
    inverse: tuple[CoreExactRowV1, ...],
    source_refs: tuple[CoreComponentRefV1, ...],
    target_refs: tuple[CoreComponentRefV1, ...],
    budget: _ArithmeticBudgetV1,
) -> bool:
    return (
        _compose_rows(inverse, forward, budget) == _identity_expressions(source_refs)
        and _compose_rows(forward, inverse, budget)
        == _identity_expressions(target_refs)
    )


def _interval_for_row(
    row: CoreExactRowV1,
    intervals: dict[CoreComponentRefV1, tuple[Fraction, Fraction]],
    budget: _ArithmeticBudgetV1,
) -> tuple[Fraction, Fraction] | None:
    lower = row.offset
    upper = row.offset
    for term in row.terms:
        interval = intervals.get(term.input_ref)
        if interval is None:
            return None
        left, right = interval
        if term.coefficient >= 0:
            scaled_left = budget.multiply(term.coefficient, left)
            scaled_right = budget.multiply(term.coefficient, right)
        else:
            scaled_left = budget.multiply(term.coefficient, right)
            scaled_right = budget.multiply(term.coefficient, left)
        lower = budget.add(lower, scaled_left)
        upper = budget.add(upper, scaled_right)
    return lower, upper


def _validate_coordinate_affine(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    budget: _ArithmeticBudgetV1,
) -> None:
    certificate = contract.certificate
    if type(certificate) is not CoreCoordinateAffineCertificateV1:
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "certificate")
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    if (
        not source_refs
        or not target_refs
        or
        certificate.missing_policy is not CoreMissingPolicyV1.EXPLICIT_PRESERVE
        or len(source_refs) != certificate.dimension
        or len(target_refs) != certificate.dimension
        or len(contract.kernel_rows) != certificate.dimension
        or len(certificate.inverse_rows) != certificate.dimension
        or not _rows_two_sided_inverse(
            contract.kernel_rows,
            certificate.inverse_rows,
            source_refs,
            target_refs,
            budget,
        )
    ):
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "inverse_or_dimension")
    source_by_index = {ref.component_index: ref for ref in source_refs}
    target_by_index = {ref.component_index: ref for ref in target_refs}
    expected_indices = set(range(certificate.dimension))
    if (
        len(source_by_index) != len(source_refs)
        or len(target_by_index) != len(target_refs)
        or set(source_by_index) != expected_indices
        or set(target_by_index) != expected_indices
        or any(
            source[source_by_index[index]].metadata.component_label
            != outputs[target_by_index[index]].component_label
            for index in range(certificate.dimension)
        )
    ):
        _reject(
            CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID,
            "component_index_or_label_pairing",
        )
    source_metadata = tuple(source[ref].metadata for ref in source_refs)
    target_metadata = tuple(outputs[ref] for ref in target_refs)
    if any(
        item.axis is not CoreComponentAxisV1.COORDINATE
        or item.value_role is not CoreComponentValueRoleV1.COORDINATE
        or item.spatial_support is None
        or item.spatial_support.frame_binding
        != certificate.source_frame_binding
        for item in source_metadata
    ) or any(
        item.axis is not CoreComponentAxisV1.COORDINATE
        or item.value_role is not CoreComponentValueRoleV1.COORDINATE
        or item.spatial_support is None
        or item.spatial_support.frame_binding
        != certificate.target_frame_binding
        for item in target_metadata
    ):
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "frame_metadata")
    source_semantics = {
        (item.entity_keys, item.role_keys, item.quantity, item.temporal_support)
        for item in source_metadata
    }
    target_semantics = {
        (item.entity_keys, item.role_keys, item.quantity, item.temporal_support)
        for item in target_metadata
    }
    if len(source_semantics) != 1 or source_semantics != target_semantics:
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "semantic_series")
    source_supports = {item.spatial_support for item in source_metadata}
    target_supports = {item.spatial_support for item in target_metadata}
    if len(source_supports) != 1 or len(target_supports) != 1:
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "common_support")
    source_support = next(iter(source_supports))
    target_support = next(iter(target_supports))
    if source_support is None or target_support is None or len(
        source_support.lower
    ) != certificate.dimension or len(target_support.lower) != certificate.dimension:
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "support_dimension")
    intervals = {
        source_by_index[index]: (
            source_support.lower[index],
            source_support.upper[index],
        )
        for index in range(certificate.dimension)
    }
    rows_by_output = _row_by_output(contract)
    bounds = tuple(
        _interval_for_row(
            rows_by_output[target_by_index[index]], intervals, budget
        )
        for index in range(certificate.dimension)
    )
    if any(bound is None for bound in bounds):
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "support_ref")
    exact_bounds = tuple(bound for bound in bounds if bound is not None)
    if tuple(bound[0] for bound in exact_bounds) != target_support.lower or tuple(
        bound[1] for bound in exact_bounds
    ) != target_support.upper:
        _reject(CoreTransformFailureCodeV1.COORDINATE_AFFINE_CONTRACT_INVALID, "support_enclosure")


def _group_ref_closure(
    groups: tuple[CoreAggregateGroupV1, ...],
) -> tuple[set[CoreComponentRefV1], set[CoreComponentRefV1]] | None:
    inputs: set[CoreComponentRefV1] = set()
    outputs: set[CoreComponentRefV1] = set()
    labels: set[str] = set()
    for group in groups:
        if (
            inputs.intersection(group.input_refs)
            or outputs.intersection(group.output_refs)
            or group.quotient_label in labels
        ):
            return None
        inputs.update(group.input_refs)
        outputs.update(group.output_refs)
        labels.add(group.quotient_label)
    return inputs, outputs


def _row_by_output(
    contract: CoreExactTransformContractV1,
) -> dict[CoreComponentRefV1, CoreExactRowV1]:
    return {row.output_ref: row for row in contract.kernel_rows}


def _valid_reducer_row(
    row: CoreExactRowV1,
    group: CoreAggregateGroupV1,
    reducer: CoreReducerV1,
    budget: _ArithmeticBudgetV1,
) -> bool:
    """Validate caller-declared coefficients; never infer geometric weights."""

    if (
        group.output_refs != (row.output_ref,)
        or tuple(term.input_ref for term in row.terms) != group.input_refs
        or row.offset != 0
    ):
        return False
    coefficients = tuple(term.coefficient for term in row.terms)
    if reducer is CoreReducerV1.SUM:
        return all(value == 1 for value in coefficients)
    if reducer is not CoreReducerV1.WEIGHTED_MEAN or any(
        value <= 0 for value in coefficients
    ):
        return False
    total = Fraction(0)
    for value in coefficients:
        total = budget.add(total, value)
    return total == 1


def _exact_temporal_partition(
    supports: tuple[CoreTemporalSupportV1, ...],
    target: CoreTemporalSupportV1,
    budget: _ArithmeticBudgetV1,
) -> bool:
    if not supports or any(
        item.clock_binding != target.clock_binding for item in supports
    ):
        return False
    intervals = tuple((item.start, item.end) for item in supports)
    if any(left >= right for left, right in intervals):
        return False
    ordered = tuple(sorted(intervals))
    if (
        ordered[0][0] != target.start
        or ordered[-1][1] != target.end
        or any(
            left_right != right_left
            for (_, left_right), (right_left, _) in zip(
                ordered, ordered[1:]
            )
        )
    ):
        return False
    total = Fraction(0)
    for left, right in ordered:
        total = budget.add(total, budget.add(right, -left))
    return total == budget.add(target.end, -target.start)


def _box_volume(
    support: CoreSpatialSupportV1,
    budget: _ArithmeticBudgetV1,
) -> Fraction | None:
    volume = Fraction(1)
    for lower, upper in zip(support.lower, support.upper, strict=True):
        width = budget.add(upper, -lower)
        if width <= 0:
            return None
        volume = budget.multiply(volume, width)
    return volume


def _spatial_interiors_overlap(
    left: CoreSpatialSupportV1,
    right: CoreSpatialSupportV1,
) -> bool:
    return all(
        max(left_lower, right_lower) < min(left_upper, right_upper)
        for left_lower, left_upper, right_lower, right_upper in zip(
            left.lower,
            left.upper,
            right.lower,
            right.upper,
            strict=True,
        )
    )


def _exact_spatial_partition(
    supports: tuple[CoreSpatialSupportV1, ...],
    target: CoreSpatialSupportV1,
    budget: _ArithmeticBudgetV1,
) -> bool:
    if not supports or any(
        item.frame_binding != target.frame_binding
        or len(item.lower) != len(target.lower)
        for item in supports
    ):
        return False
    for support in supports:
        if any(
            lower < target_lower or upper > target_upper
            for lower, upper, target_lower, target_upper in zip(
                support.lower,
                support.upper,
                target.lower,
                target.upper,
                strict=True,
            )
        ):
            return False
    for index, left in enumerate(supports):
        for right in supports[index + 1 :]:
            budget.operations += 1
            if budget.operations > _LIMITS_V1.maximum_exact_operations:
                _reject(CoreTransformFailureCodeV1.RESOURCE_LIMIT, "exact_operations")
            if _spatial_interiors_overlap(left, right):
                return False
    target_volume = _box_volume(target, budget)
    if target_volume is None:
        return False
    total = Fraction(0)
    for support in supports:
        volume = _box_volume(support, budget)
        if volume is None:
            return False
        total = budget.add(total, volume)
    return total == target_volume


def _validate_aggregation(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    budget: _ArithmeticBudgetV1,
    *,
    temporal: bool,
) -> None:
    certificate = contract.certificate
    expected_type = (
        CoreTemporalAggregationCertificateV1
        if temporal
        else CoreSpatialAggregationCertificateV1
    )
    code = (
        CoreTransformFailureCodeV1.TEMPORAL_AGGREGATION_CONTRACT_INVALID
        if temporal
        else CoreTransformFailureCodeV1.SPATIAL_AGGREGATION_CONTRACT_INVALID
    )
    if type(certificate) is not expected_type:
        _reject(code, "certificate")
    if (
        type(certificate.reducer) is not CoreReducerV1
        or certificate.missing_policy is not CoreMissingPolicyV1.REJECT
        or certificate.boundary_policy
        is not CoreBoundaryPolicyV1.EXACT_PARTITION
        or certificate.weighted_mean_basis
        is not CoreWeightedMeanBasisV1.CALLER_DECLARED_EXACT_COEFFICIENTS
    ):
        _reject(code, "policy")
    if (
        not certificate.groups
        or not _numeric_source_refs(source)
        or not _numeric_output_refs(contract)
    ):
        _reject(code, "numeric_witness_required")
    closure = _group_ref_closure(certificate.groups)
    if closure != (
        set(_numeric_source_refs(source)),
        set(_numeric_output_refs(contract)),
    ):
        _reject(code, "group_closure")
    rows = _row_by_output(contract)
    expected_axis = (
        CoreComponentAxisV1.TEMPORAL
        if temporal
        else CoreComponentAxisV1.SPATIAL
    )
    expected_role = (
        CoreComponentValueRoleV1.EXTENSIVE
        if certificate.reducer is CoreReducerV1.SUM
        else CoreComponentValueRoleV1.INTENSIVE
    )
    for group in certificate.groups:
        row = rows.get(group.output_refs[0]) if len(group.output_refs) == 1 else None
        if row is None or not _valid_reducer_row(
            row, group, certificate.reducer, budget
        ):
            _reject(code, "reducer")
        input_metadata = tuple(source[ref].metadata for ref in group.input_refs)
        output = outputs[group.output_refs[0]]
        if output.axis is not expected_axis or output.value_role is not expected_role:
            _reject(code, "output_axis_role")
        if any(
            item.axis is not expected_axis
            or item.value_role is not expected_role
            or item.entity_keys != output.entity_keys
            or item.role_keys != output.role_keys
            or item.quantity != output.quantity
            or item.component_label != output.component_label
            for item in input_metadata
        ):
            _reject(code, "metadata")
        if temporal:
            supports = tuple(item.temporal_support for item in input_metadata)
            if (
                any(item is None for item in supports)
                or output.temporal_support is None
                or not _exact_temporal_partition(
                    tuple(item for item in supports if item is not None),
                    output.temporal_support,
                    budget,
                )
                or any(
                    item.spatial_support != output.spatial_support
                    for item in input_metadata
                )
            ):
                _reject(code, "temporal_partition_or_anchor")
        else:
            supports = tuple(item.spatial_support for item in input_metadata)
            if (
                any(item is None for item in supports)
                or output.spatial_support is None
                or not _exact_spatial_partition(
                    tuple(item for item in supports if item is not None),
                    output.spatial_support,
                    budget,
                )
                or any(
                    item.temporal_support != output.temporal_support
                    for item in input_metadata
                )
            ):
                _reject(code, "spatial_partition_or_anchor")


def _validate_sampling(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
) -> None:
    certificate = contract.certificate
    code = CoreTransformFailureCodeV1.SAMPLING_RESOLUTION_CONTRACT_INVALID
    if type(certificate) is not CoreSamplingResolutionCertificateV1:
        _reject(code, "certificate")
    if (
        certificate.axis
        not in (CoreComponentAxisV1.TEMPORAL, CoreComponentAxisV1.SPATIAL)
        or certificate.missing_policy is not CoreMissingPolicyV1.REJECT
        or certificate.boundary_policy is not CoreBoundaryPolicyV1.VALID_ONLY
        or type(certificate.grid_dimension) is not int
        or not 1 <= certificate.grid_dimension <= 4
    ):
        _reject(code, "policy_or_dimension")
    _canonical_ref_tuple(certificate.selected_inputs, name="sampling_selected")
    _canonical_ref_tuple(certificate.discarded_inputs, name="sampling_discarded")
    numeric_source = set(_numeric_source_refs(source))
    selected = set(certificate.selected_inputs)
    discarded = set(certificate.discarded_inputs)
    if not selected or not _numeric_output_refs(contract):
        _reject(code, "numeric_selected_output_witness_required")
    if selected.intersection(discarded) or selected.union(discarded) != numeric_source:
        _reject(code, "selected_discarded_coverage")
    if certificate.grid_points != tuple(sorted(certificate.grid_points)) or tuple(
        point.input_ref for point in certificate.grid_points
    ) != certificate.selected_inputs:
        _reject(code, "input_bound_grid_order")
    if any(
        len(point.coordinates) != certificate.grid_dimension
        for point in certificate.grid_points
    ):
        _reject(code, "grid_dimension")
    source_metadata = tuple(source[ref].metadata for ref in sorted(numeric_source))
    if any(item.axis is not certificate.axis for item in source_metadata) or (
        source_metadata
        and any(
            not _same_semantic_series(
                source_metadata[0], item, ignore_axis_support=certificate.axis
            )
            for item in source_metadata[1:]
        )
    ):
        _reject(code, "semantic_series")
    source_points: dict[CoreComponentRefV1, tuple[Fraction, ...]] = {}
    if certificate.axis is CoreComponentAxisV1.TEMPORAL:
        if certificate.grid_dimension != 1 or certificate.grid_frame_binding is not None:
            _reject(code, "temporal_grid_policy")
        clocks: set[CoreClockBindingV1] = set()
        for input_ref in sorted(numeric_source):
            support = source[input_ref].metadata.temporal_support
            if support is None or support.start != support.end:
                _reject(code, "temporal_source_not_grid_point")
            clocks.add(support.clock_binding)
            source_points[input_ref] = (support.start,)
        if len(clocks) != 1:
            _reject(code, "temporal_clock_anchor")
    else:
        if certificate.grid_frame_binding is None:
            _reject(code, "spatial_grid_policy")
        for input_ref in sorted(numeric_source):
            support = source[input_ref].metadata.spatial_support
            if (
                support is None
                or support.frame_binding != certificate.grid_frame_binding
                or len(support.lower) != certificate.grid_dimension
                or support.lower != support.upper
            ):
                _reject(code, "spatial_source_not_grid_point")
            source_points[input_ref] = support.lower
    if len(set(source_points.values())) != len(source_points):
        _reject(code, "source_grid_points_repeat")
    rows = contract.kernel_rows
    if len(rows) != len(certificate.selected_inputs) or len(rows) != len(
        _numeric_output_refs(contract)
    ):
        _reject(code, "output_count")
    grid = {point.input_ref: point.coordinates for point in certificate.grid_points}
    seen: set[CoreComponentRefV1] = set()
    for row in rows:
        if (
            len(row.terms) != 1
            or row.terms[0].coefficient != 1
            or row.offset != 0
            or row.terms[0].input_ref not in selected
            or row.terms[0].input_ref in seen
        ):
            _reject(code, "subselection_kernel")
        input_ref = row.terms[0].input_ref
        seen.add(input_ref)
        left = source[input_ref].metadata
        right = outputs[row.output_ref]
        if not _metadata_equal_except_ref(left, right) or left.axis is not certificate.axis:
            _reject(code, "metadata_copy")
        point = grid[input_ref]
        if source_points[input_ref] != point:
            _reject(code, "input_bound_grid_coordinate")
        if certificate.axis is CoreComponentAxisV1.TEMPORAL:
            support = left.temporal_support
            if (
                certificate.grid_dimension != 1
                or certificate.grid_frame_binding is not None
                or support is None
                or support.start != support.end
                or point != (support.start,)
            ):
                _reject(code, "temporal_grid_anchor")
        else:
            support = left.spatial_support
            if (
                certificate.grid_frame_binding is None
                or support is None
                or support.frame_binding != certificate.grid_frame_binding
                or support.lower != support.upper
                or point != support.lower
            ):
                _reject(code, "spatial_grid_anchor")
    if seen != selected:
        _reject(code, "selected_exact_use")


def _same_numeric_semantics(
    left: CoreComponentMetadataV1,
    right: CoreComponentMetadataV1,
) -> bool:
    return (
        left.entity_keys == right.entity_keys
        and left.role_keys == right.role_keys
        and left.quantity == right.quantity
        and left.axis == right.axis
        and left.temporal_support == right.temporal_support
        and left.spatial_support == right.spatial_support
    )


def _validate_split_merge(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    budget: _ArithmeticBudgetV1,
) -> None:
    certificate = contract.certificate
    code = CoreTransformFailureCodeV1.EQUIVALENT_SPLIT_MERGE_CONTRACT_INVALID
    if type(certificate) is not CoreEquivalentSplitMergeCertificateV1:
        _reject(code, "certificate")
    if (
        type(certificate.direction) is not CoreSplitMergeDirectionV1
        or certificate.missing_policy is not CoreMissingPolicyV1.REJECT
        or not certificate.groups
        or not certificate.inverse_rows
        or not _numeric_source_refs(source)
        or not _numeric_output_refs(contract)
        or _group_ref_closure(certificate.groups)
        != (
            set(_numeric_source_refs(source)),
            set(_numeric_output_refs(contract)),
        )
    ):
        _reject(code, "policy_or_group_closure")
    _validate_output_label_bindings(
        certificate.groups, certificate.label_bindings, outputs, code
    )
    rows = _row_by_output(contract)
    for group in certificate.groups:
        if certificate.direction is CoreSplitMergeDirectionV1.SPLIT:
            if len(group.input_refs) != 1 or len(group.output_refs) < 2:
                _reject(code, "split_shape")
            source_ref = group.input_refs[0]
            total = Fraction(0)
            for output_ref in group.output_refs:
                row = rows.get(output_ref)
                if (
                    row is None
                    or len(row.terms) != 1
                    or row.terms[0].input_ref != source_ref
                    or row.terms[0].coefficient <= 0
                    or row.offset != 0
                    or source[source_ref].metadata.value_role
                    is not CoreComponentValueRoleV1.EXTENSIVE
                    or outputs[output_ref].value_role
                    is not CoreComponentValueRoleV1.EXTENSIVE
                    or not _same_numeric_semantics(
                        source[source_ref].metadata, outputs[output_ref]
                    )
                ):
                    _reject(code, "split_kernel_or_metadata")
                total = budget.add(total, row.terms[0].coefficient)
            if total != 1:
                _reject(code, "split_positive_partition")
        else:
            if len(group.input_refs) < 2 or len(group.output_refs) != 1:
                _reject(code, "merge_shape")
            row = rows.get(group.output_refs[0])
            target = outputs[group.output_refs[0]]
            if (
                row is None
                or tuple(term.input_ref for term in row.terms) != group.input_refs
                or any(term.coefficient != 1 for term in row.terms)
                or row.offset != 0
                or target.value_role is not CoreComponentValueRoleV1.EXTENSIVE
                or any(
                    source[ref].metadata.value_role
                    is not CoreComponentValueRoleV1.EXTENSIVE
                    or not _same_numeric_semantics(source[ref].metadata, target)
                    for ref in group.input_refs
                )
            ):
                _reject(code, "merge_kernel_or_metadata")
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    inverse_by_output = {row.output_ref: row for row in certificate.inverse_rows}
    if len(inverse_by_output) != len(certificate.inverse_rows) or set(
        inverse_by_output
    ) != set(source_refs):
        _reject(code, "inverse_coverage")
    if certificate.direction is CoreSplitMergeDirectionV1.SPLIT:
        for group in certificate.groups:
            row = inverse_by_output[group.input_refs[0]]
            if (
                tuple(term.input_ref for term in row.terms) != group.output_refs
                or any(term.coefficient != 1 for term in row.terms)
                or row.offset != 0
            ):
                _reject(code, "split_inverse")
        composed = _compose_rows(certificate.inverse_rows, contract.kernel_rows, budget)
        expected = _identity_expressions(source_refs)
    else:
        for group in certificate.groups:
            total = Fraction(0)
            for input_ref in group.input_refs:
                row = inverse_by_output[input_ref]
                if (
                    len(row.terms) != 1
                    or row.terms[0].input_ref != group.output_refs[0]
                    or row.terms[0].coefficient <= 0
                    or row.offset != 0
                ):
                    _reject(code, "merge_inverse")
                total = budget.add(total, row.terms[0].coefficient)
            if total != 1:
                _reject(code, "merge_inverse_partition")
        composed = _compose_rows(contract.kernel_rows, certificate.inverse_rows, budget)
        expected = _identity_expressions(target_refs)
    if composed != expected:
        _reject(code, "one_sided_inverse")


def _validate_coarse_graining(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    budget: _ArithmeticBudgetV1,
) -> None:
    certificate = contract.certificate
    code = CoreTransformFailureCodeV1.COARSE_GRAINING_CONTRACT_INVALID
    if type(certificate) is not CoreCoarseGrainingCertificateV1:
        _reject(code, "certificate")
    _sorted_unique_text(certificate.quotient_labels, "coarse quotient labels")
    if (
        type(certificate.reducer) is not CoreReducerV1
        or certificate.missing_policy is not CoreMissingPolicyV1.REJECT
        or certificate.boundary_policy
        is not CoreBoundaryPolicyV1.EXACT_PARTITION
        or certificate.weighted_mean_basis
        is not CoreWeightedMeanBasisV1.CALLER_DECLARED_EXACT_COEFFICIENTS
        or not certificate.groups
        or not certificate.source_commutation_rows
        or not certificate.target_commutation_rows
        or not _numeric_source_refs(source)
        or not _numeric_output_refs(contract)
        or tuple(sorted(group.quotient_label for group in certificate.groups))
        != certificate.quotient_labels
        or _group_ref_closure(certificate.groups)
        != (
            set(_numeric_source_refs(source)),
            set(_numeric_output_refs(contract)),
        )
    ):
        _reject(code, "policy_group_or_quotient_closure")
    _validate_output_label_bindings(
        certificate.groups, certificate.label_bindings, outputs, code
    )
    rows = _row_by_output(contract)
    expected_role = (
        CoreComponentValueRoleV1.EXTENSIVE
        if certificate.reducer is CoreReducerV1.SUM
        else CoreComponentValueRoleV1.INTENSIVE
    )
    for group in certificate.groups:
        row = rows.get(group.output_refs[0]) if len(group.output_refs) == 1 else None
        if row is None or not _valid_reducer_row(
            row, group, certificate.reducer, budget
        ):
            _reject(code, "reducer")
        target = outputs[group.output_refs[0]]
        if target.axis is not CoreComponentAxisV1.COARSE or target.value_role is not expected_role:
            _reject(code, "output_axis_role")
        if any(
            source[ref].metadata.value_role is not expected_role
            or source[ref].metadata.entity_keys != target.entity_keys
            or source[ref].metadata.role_keys != target.role_keys
            or source[ref].metadata.quantity != target.quantity
            or source[ref].metadata.temporal_support != target.temporal_support
            or source[ref].metadata.spatial_support != target.spatial_support
            for ref in group.input_refs
        ):
            _reject(code, "metadata")
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    source_rows = certificate.source_commutation_rows
    target_rows = certificate.target_commutation_rows
    if (
        {row.output_ref for row in source_rows} != set(source_refs)
        or {
            term.input_ref for row in source_rows for term in row.terms
        } - set(source_refs)
        or {row.output_ref for row in target_rows} != set(target_refs)
        or {
            term.input_ref for row in target_rows for term in row.terms
        } - set(target_refs)
    ):
        _reject(code, "commutation_ref_closure")
    left = _compose_rows(contract.kernel_rows, source_rows, budget)
    right = _compose_rows(target_rows, contract.kernel_rows, budget)
    if left is None or left != right:
        _reject(code, "commutation_equation")


def _validate_operation(
    contract: CoreExactTransformContractV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    outputs: dict[CoreComponentRefV1, CoreComponentMetadataV1],
    budget: _ArithmeticBudgetV1,
) -> None:
    operation = contract.operation
    if operation is CoreTransformOperationV1.IDENTITY:
        _validate_identity(contract, source, outputs)
    elif operation is CoreTransformOperationV1.UNIT_CONVERSION:
        _validate_unit_conversion(contract, source, outputs, budget)
    elif operation is CoreTransformOperationV1.COORDINATE_AFFINE:
        _validate_coordinate_affine(contract, source, outputs, budget)
    elif operation is CoreTransformOperationV1.TEMPORAL_AGGREGATION:
        _validate_aggregation(contract, source, outputs, budget, temporal=True)
    elif operation is CoreTransformOperationV1.SPATIAL_AGGREGATION:
        _validate_aggregation(contract, source, outputs, budget, temporal=False)
    elif operation is CoreTransformOperationV1.SAMPLING_RESOLUTION:
        _validate_sampling(contract, source, outputs)
    elif operation is CoreTransformOperationV1.EQUIVALENT_SPLIT_MERGE:
        _validate_split_merge(contract, source, outputs, budget)
    elif operation is CoreTransformOperationV1.COARSE_GRAINING:
        _validate_coarse_graining(contract, source, outputs, budget)
    else:
        _reject(CoreTransformFailureCodeV1.SCHEMA_INVALID, "unknown_operation")


def _evaluate_numeric_row(
    row: CoreExactRowV1,
    source: dict[CoreComponentRefV1, CoreComponentValueV1],
    budget: _ArithmeticBudgetV1,
) -> tuple[Fraction, Fraction]:
    value = row.offset
    radius = Fraction(0)
    for term in row.terms:
        component = source[term.input_ref]
        if (
            component.value_kind is not CoreComponentValueKindV1.NUMERIC
            or component.numeric_value is None
            or component.absolute_radius is None
            or component.uncertainty_model
            is not CoreUncertaintyModelV1.ABSOLUTE_BOUND
        ):
            _reject(
                CoreTransformFailureCodeV1.EXACT_KERNEL_EVALUATION_FAILED,
                "nonnumeric_or_unsupported_input",
            )
        value = budget.add(
            value, budget.multiply(term.coefficient, component.numeric_value)
        )
        radius = budget.add(
            radius,
            budget.multiply(abs(term.coefficient), component.absolute_radius),
        )
    return value, radius


def _validate_and_compute(
    profile: ExactTransformCoreProfileV1,
) -> tuple[CoreComponentValueV1, ...]:
    _shallow_profile_resource_preflight(profile)
    fresh_nodes, fraction_values = _bounded_closed_tree_nodes(profile)
    _preflight_standard_error(fresh_nodes, detail="source_standard_error")
    _check_fraction_collection(fraction_values)
    _revalidate_dataclass_invariants(fresh_nodes)
    _resource_preflight(profile)
    source, output_metadata = _common_validation(profile)
    budget = _ArithmeticBudgetV1()
    _validate_operation(profile.contract, source, output_metadata, budget)
    result: list[CoreComponentValueV1] = []
    for row in profile.contract.kernel_rows:
        value, radius = _evaluate_numeric_row(row, source, budget)
        result.append(
            CoreComponentValueV1(
                metadata=output_metadata[row.output_ref],
                value_kind=CoreComponentValueKindV1.NUMERIC,
                numeric_value=value,
                uncertainty_model=CoreUncertaintyModelV1.ABSOLUTE_BOUND,
                absolute_radius=radius,
            )
        )
    for mapping in profile.contract.discrete_mappings:
        item = source[mapping.input_ref]
        result.append(
            CoreComponentValueV1(
                metadata=output_metadata[mapping.output_ref],
                value_kind=item.value_kind,
                numeric_value=None,
                uncertainty_model=CoreUncertaintyModelV1.NOT_APPLICABLE,
                absolute_radius=None,
                boolean_value=item.boolean_value,
            )
        )
    return tuple(sorted(result, key=lambda component: component.ref))


def validate_exact_transform_core_v1(profile: ExactTransformCoreProfileV1) -> None:
    """Validate one projected transform and its exact arithmetic to completion.

    The function raises :class:`ExactTransformCoreValidationErrorV1` on a
    semantic rejection and returns ``None`` only after every output cell can be
    built under the frozen resource limits.
    """

    if type(profile) is not ExactTransformCoreProfileV1:
        _reject(CoreTransformFailureCodeV1.INVALID_PROFILE_TYPE, "profile")
    try:
        _validate_and_compute(profile)
    except ExactTransformCoreValidationErrorV1:
        raise
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExactTransformCoreValidationErrorV1(
            CoreTransformFailureCodeV1.INTERNAL_CORE_FAILURE,
            "closed_profile_evaluation",
        ) from exc


def _abstention(
    code: CoreTransformFailureCodeV1,
    detail: str,
) -> ExactTransformCoreResultV1:
    return ExactTransformCoreResultV1(
        disposition=CoreTransformDispositionV1.ABSTAIN,
        output_components=(),
        failure=ExactTransformCoreFailureV1(code, detail),
    )


def run_exact_transform_core_v1(
    profile: ExactTransformCoreProfileV1,
) -> ExactTransformCoreResultV1:
    """Compute all exact outputs or return a typed output-empty abstention."""

    if type(profile) is not ExactTransformCoreProfileV1:
        return _abstention(
            CoreTransformFailureCodeV1.INVALID_PROFILE_TYPE, "profile"
        )
    try:
        outputs = _validate_and_compute(profile)
    except ExactTransformCoreValidationErrorV1 as exc:
        return _abstention(exc.code, exc.detail)
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
        return _abstention(
            CoreTransformFailureCodeV1.INTERNAL_CORE_FAILURE,
            "closed_profile_evaluation",
        )
    return ExactTransformCoreResultV1(
        disposition=CoreTransformDispositionV1.COMPLETE,
        output_components=outputs,
        failure=None,
    )


def verify_exact_transform_core_outputs_v1(
    profile: ExactTransformCoreProfileV1,
    claimed_outputs: tuple[CoreComponentValueV1, ...],
) -> ExactTransformCoreResultV1:
    """Freshly compute and exactly compare a claimed target state.

    This is the required entry point for a wire that already carries target
    values.  Reference coverage, metadata, kind, point value, absolute radius,
    and Boolean/missing payloads are compared independently.  A mismatch never
    returns the freshly computed target as a partial result.
    """

    if type(claimed_outputs) is not tuple:
        return _abstention(
            CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID,
            "claimed_output_schema",
        )
    if len(claimed_outputs) > _LIMITS_V1.maximum_components:
        return _abstention(
            CoreTransformFailureCodeV1.CLAIM_RESOURCE_LIMIT,
            "claimed_output_count",
        )
    if any(type(component) is not CoreComponentValueV1 for component in claimed_outputs):
        return _abstention(
            CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID,
            "claimed_output_item_type",
        )
    try:
        claim_nodes, claim_fractions = _bounded_closed_tree_nodes(claimed_outputs)
        _preflight_standard_error(claim_nodes, detail="claimed_standard_error")
        _check_fraction_collection(claim_fractions)
        _revalidate_dataclass_invariants(claim_nodes)
    except ExactTransformCoreValidationErrorV1 as exc:
        if exc.code is CoreTransformFailureCodeV1.STANDARD_ERROR_UNSUPPORTED:
            return _abstention(exc.code, exc.detail)
        claim_code = (
            CoreTransformFailureCodeV1.CLAIM_RESOURCE_LIMIT
            if exc.code is CoreTransformFailureCodeV1.RESOURCE_LIMIT
            else CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID
        )
        return _abstention(claim_code, "claimed_output_fresh_validation")
    except (
        ArithmeticError,
        AttributeError,
        KeyError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return _abstention(
            CoreTransformFailureCodeV1.CLAIM_SCHEMA_INVALID,
            "claimed_output_fresh_validation",
        )
    computed = run_exact_transform_core_v1(profile)
    if computed.disposition is CoreTransformDispositionV1.ABSTAIN:
        return computed
    claimed_refs = tuple(component.ref for component in claimed_outputs)
    if (
        claimed_refs != tuple(sorted(claimed_refs))
        or len(claimed_refs) != len(set(claimed_refs))
        or claimed_refs
        != tuple(component.ref for component in computed.output_components)
    ):
        return _abstention(
            CoreTransformFailureCodeV1.CLAIM_REFERENCE_MISMATCH,
            "claimed_output_refs",
        )
    for expected, claimed in zip(
        computed.output_components, claimed_outputs, strict=True
    ):
        if (
            claimed.value_kind is CoreComponentValueKindV1.MISSING
        ) != (expected.value_kind is CoreComponentValueKindV1.MISSING):
            return _abstention(
                CoreTransformFailureCodeV1.CLAIM_MISSING_MISMATCH,
                "claimed_output_missingness",
            )
        if claimed.value_kind is not expected.value_kind:
            return _abstention(
                CoreTransformFailureCodeV1.CLAIM_KIND_MISMATCH,
                "claimed_output_kind",
            )
        if claimed.metadata != expected.metadata:
            return _abstention(
                CoreTransformFailureCodeV1.CLAIM_METADATA_MISMATCH,
                "claimed_output_metadata",
            )
        if claimed.numeric_value != expected.numeric_value:
            return _abstention(
                CoreTransformFailureCodeV1.CLAIM_VALUE_MISMATCH,
                "claimed_output_value",
            )
        if (
            claimed.uncertainty_model is not expected.uncertainty_model
            or claimed.absolute_radius != expected.absolute_radius
        ):
            return _abstention(
                CoreTransformFailureCodeV1.CLAIM_RADIUS_MISMATCH,
                "claimed_output_radius",
            )
        if claimed.boolean_value != expected.boolean_value:
            return _abstention(
                CoreTransformFailureCodeV1.CLAIM_BOOLEAN_MISMATCH,
                "claimed_output_boolean",
            )
    return computed


__all__ = [
    "CoreAggregateGroupV1",
    "CoreBoundaryPolicyV1",
    "CoreClockBindingV1",
    "CoreCoarseGrainingCertificateV1",
    "CoreComponentAxisV1",
    "CoreComponentMetadataV1",
    "CoreComponentRefV1",
    "CoreComponentValueKindV1",
    "CoreComponentValueRoleV1",
    "CoreComponentValueV1",
    "CoreCoordinateAffineCertificateV1",
    "CoreDiscreteMappingV1",
    "CoreEquivalentSplitMergeCertificateV1",
    "CoreExactRowV1",
    "CoreExactTermV1",
    "CoreExactTransformContractV1",
    "CoreFrameBindingV1",
    "CoreIdentityCertificateV1",
    "CoreMissingPolicyV1",
    "CoreObservationRefV1",
    "CoreOperationCertificateV1",
    "CoreOutputLabelBindingV1",
    "CoreQuantityMetadataV1",
    "CoreReducerV1",
    "CoreSamplingGridPointV1",
    "CoreSamplingResolutionCertificateV1",
    "CoreScaleRefV1",
    "CoreSpatialAggregationCertificateV1",
    "CoreSpatialSupportV1",
    "CoreSplitMergeDirectionV1",
    "CoreTemporalAggregationCertificateV1",
    "CoreTemporalSupportV1",
    "CoreTransformDispositionV1",
    "CoreTransformFailureCodeV1",
    "CoreTransformOperationV1",
    "CoreUncertaintyModelV1",
    "CoreUnitBindingV1",
    "CoreUnitConversionCertificateV1",
    "CoreWeightedMeanBasisV1",
    "EXACT_TRANSFORM_CORE_VERSION_V1",
    "ExactTransformCoreFailureV1",
    "ExactTransformCoreProfileV1",
    "ExactTransformCoreResultV1",
    "ExactTransformCoreValidationErrorV1",
    "run_exact_transform_core_v1",
    "validate_exact_transform_core_v1",
    "verify_exact_transform_core_outputs_v1",
]

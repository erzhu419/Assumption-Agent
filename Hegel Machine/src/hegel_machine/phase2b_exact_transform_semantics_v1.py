"""Content-addressed typed transform authority and exact sparse kernels.

The Phase-2B v1 wire names transform operations but deliberately gives the
anonymous ``parameters`` vector no executable meaning.  This module does not
guess one.  It introduces a separate versioned authority wrapper whose full
canonical mapping contains component/scale metadata, exact sparse affine
operators, and an operation-specific certificate for every graph edge.

The authoritative entry point accepts only that wrapper.  It checks the exact
frozen authority tree and resource bounds before hashing, internally rebuilds
the bundle-atomic uncertainty receipt, and applies contracts in graph order.
Every invalid contract or cell aborts the whole compilation without returning
partial transformed components.  This is a narrow typed-authority and kernel
receipt; it is not integrated with the exact law bridge and does not establish
the frozen preservation-pair or Phase-2B exit claims.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from typing import Final, TypeAlias
from uuid import UUID

from .hashing import stable_hash
from .phase2b_uncertainty_compiler import (
    BundleUncertaintyCompilation,
    BundleUncertaintyDisposition,
    DEFAULT_EXACT_UNCERTAINTY_POLICY,
    FROZEN_PHASE2B_EXACT_FREEZE_ID,
    FROZEN_RATIONAL_GRID_ID,
    ObservationValueKind,
    compile_bundle_uncertainty,
)
from .phase2b_wire import (
    AggregationEdge,
    AggregationGraph,
    BooleanValue,
    EntityCandidate,
    MeasurementUncertainty,
    Missingness,
    NumericInterval,
    NumericValue,
    PublicEvidenceBundle,
    SpatialSupport,
    TaskTarget,
    TemporalSupport,
    TransformOperation,
    TransformSpec,
    TypedObservation,
    UncertaintyModel,
    UnitDimension,
)
from .schema import require_tuple


PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-transform-evidence/2"
)
EXACT_TRANSFORM_SEMANTICS_VERSION: Final = (
    "hegel-machine-phase2b-exact-sparse-transform-semantics/1"
)
def _uuid4(value: str, name: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be an opaque UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical lowercase UUIDv4")
    return value


class TransformCompilationDisposition(str, Enum):
    COMPLETE = "complete"
    ABSTAIN = "abstain"


class ComponentAxis(str, Enum):
    SCALAR = "scalar"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CHANNEL = "channel"
    ENTITY = "entity"
    COORDINATE = "coordinate"
    COARSE = "coarse"
    CONTROL = "control"


class ComponentValueRole(str, Enum):
    INTENSIVE = "intensive"
    EXTENSIVE = "extensive"
    COORDINATE = "coordinate"
    BOOLEAN_CONTROL = "boolean_control"
    MISSING = "missing"


class ComponentValueKind(str, Enum):
    NUMERIC_INTERVAL = "numeric_interval"
    BOOLEAN = "boolean"
    MISSING = "missing"


class ReducerKind(str, Enum):
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"


class MissingValuePolicy(str, Enum):
    REJECT = "reject"
    EXPLICIT_PRESERVE = "explicit_preserve"


class BoundaryPolicy(str, Enum):
    EXACT_PARTITION = "exact_partition"
    VALID_ONLY = "valid_only"


class SplitMergeDirection(str, Enum):
    SPLIT = "split"
    MERGE = "merge"


@dataclass(frozen=True, order=True, slots=True)
class ExactTransformAtom:
    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError("exact transform atom fields must be integers")
        if self.denominator <= 0:
            raise ValueError("exact transform atom denominator must be positive")
        value = Fraction(self.numerator, self.denominator)
        if (value.numerator, value.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("exact transform atom must already be reduced")

    @classmethod
    def from_fraction(cls, value: Fraction) -> "ExactTransformAtom":
        if type(value) is not Fraction:
            raise TypeError("exact transform atom requires Fraction")
        return cls(value.numerator, value.denominator)

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


ZERO: Final = ExactTransformAtom(0)
ONE: Final = ExactTransformAtom(1)


@dataclass(frozen=True, slots=True)
class ExactTransformInterval:
    lower: ExactTransformAtom
    upper: ExactTransformAtom

    def __post_init__(self) -> None:
        if type(self.lower) is not ExactTransformAtom or type(
            self.upper
        ) is not ExactTransformAtom:
            raise TypeError("exact transform interval endpoints have wrong type")
        if self.lower_fraction > self.upper_fraction:
            raise ValueError("exact transform interval lower exceeds upper")

    @classmethod
    def from_fractions(
        cls,
        lower: Fraction,
        upper: Fraction,
    ) -> "ExactTransformInterval":
        return cls(
            ExactTransformAtom.from_fraction(lower),
            ExactTransformAtom.from_fraction(upper),
        )

    @property
    def lower_fraction(self) -> Fraction:
        return self.lower.as_fraction()

    @property
    def upper_fraction(self) -> Fraction:
        return self.upper.as_fraction()


@dataclass(frozen=True, slots=True)
class ExactTemporalSupport:
    clock_id: str
    start: ExactTransformAtom
    end: ExactTransformAtom

    def __post_init__(self) -> None:
        _uuid4(self.clock_id, "exact temporal clock ID")
        if type(self.start) is not ExactTransformAtom or type(
            self.end
        ) is not ExactTransformAtom:
            raise TypeError("exact temporal endpoints have wrong type")
        if self.start.as_fraction() > self.end.as_fraction():
            raise ValueError("exact temporal support start exceeds end")

    @classmethod
    def from_wire(cls, support: TemporalSupport) -> "ExactTemporalSupport":
        if type(support) is not TemporalSupport:
            raise TypeError("exact temporal support requires exact wire type")
        return cls(
            support.clock_id,
            ExactTransformAtom.from_fraction(Fraction.from_float(support.start)),
            ExactTransformAtom.from_fraction(Fraction.from_float(support.end)),
        )


@dataclass(frozen=True, slots=True)
class ExactSpatialSupport:
    frame_id: str
    lower: tuple[ExactTransformAtom, ...]
    upper: tuple[ExactTransformAtom, ...]

    def __post_init__(self) -> None:
        _uuid4(self.frame_id, "exact spatial frame ID")
        require_tuple(self.lower, "exact spatial lower")
        require_tuple(self.upper, "exact spatial upper")
        if (
            not self.lower
            or len(self.lower) != len(self.upper)
            or len(self.lower) > 4
        ):
            raise ValueError("exact spatial bounds need one to four dimensions")
        if any(type(value) is not ExactTransformAtom for value in self.lower):
            raise TypeError("exact spatial lower contains invalid atom")
        if any(type(value) is not ExactTransformAtom for value in self.upper):
            raise TypeError("exact spatial upper contains invalid atom")
        if any(
            lower.as_fraction() > upper.as_fraction()
            for lower, upper in zip(self.lower, self.upper, strict=True)
        ):
            raise ValueError("exact spatial lower exceeds upper")

    @classmethod
    def from_wire(cls, support: SpatialSupport) -> "ExactSpatialSupport":
        if type(support) is not SpatialSupport:
            raise TypeError("exact spatial support requires exact wire type")
        return cls(
            support.frame_id,
            tuple(
                ExactTransformAtom.from_fraction(Fraction.from_float(value))
                for value in support.lower
            ),
            tuple(
                ExactTransformAtom.from_fraction(Fraction.from_float(value))
                for value in support.upper
            ),
        )


@dataclass(frozen=True, order=True, slots=True)
class ComponentRef:
    scale_id: str
    observation_id: str
    ordinal: int
    component_id: str

    def __post_init__(self) -> None:
        _uuid4(self.scale_id, "component scale ID")
        _uuid4(self.observation_id, "component observation ID")
        _uuid4(self.component_id, "component ID")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("component ordinal must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class ObservationComponentMetadata:
    observation_id: str
    scale_id: str
    component_ids: tuple[str, ...]
    axis: ComponentAxis
    value_role: ComponentValueRole
    unit_id: str | None
    coordinate_frame_id: str | None = None

    def __post_init__(self) -> None:
        _uuid4(self.observation_id, "metadata observation ID")
        _uuid4(self.scale_id, "metadata scale ID")
        require_tuple(self.component_ids, "metadata component IDs")
        if not self.component_ids:
            raise ValueError("observation metadata needs component IDs")
        for value in self.component_ids:
            _uuid4(value, "metadata component ID")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("observation metadata repeats a component ID")
        if type(self.axis) is not ComponentAxis:
            raise TypeError("observation metadata axis has wrong type")
        if type(self.value_role) is not ComponentValueRole:
            raise TypeError("observation metadata role has wrong type")
        if self.unit_id is not None:
            _uuid4(self.unit_id, "metadata unit ID")
        if self.coordinate_frame_id is not None:
            _uuid4(self.coordinate_frame_id, "metadata coordinate frame ID")
        if self.value_role is ComponentValueRole.COORDINATE:
            if self.coordinate_frame_id is None or self.unit_id is None:
                raise ValueError("coordinate metadata needs unit and frame IDs")
        elif self.coordinate_frame_id is not None:
            raise ValueError("only coordinate metadata may carry a frame ID")
        if self.value_role in (
            ComponentValueRole.BOOLEAN_CONTROL,
            ComponentValueRole.MISSING,
        ):
            if self.unit_id is not None or len(self.component_ids) != 1:
                raise ValueError("discrete metadata needs one unitless component")
        elif self.unit_id is None:
            raise ValueError("numeric metadata needs an opaque unit ID")


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    ref: ComponentRef
    axis: ComponentAxis
    value_role: ComponentValueRole
    unit_id: str | None
    si_exponents: tuple[int, int, int, int, int, int, int]
    coordinate_frame_id: str | None
    temporal_support: ExactTemporalSupport | None
    spatial_support: ExactSpatialSupport | None

    def __post_init__(self) -> None:
        if type(self.ref) is not ComponentRef:
            raise TypeError("component descriptor ref has wrong type")
        if type(self.axis) is not ComponentAxis:
            raise TypeError("component descriptor axis has wrong type")
        if type(self.value_role) is not ComponentValueRole:
            raise TypeError("component descriptor role has wrong type")
        if self.unit_id is not None:
            _uuid4(self.unit_id, "component unit ID")
        require_tuple(self.si_exponents, "component SI exponents")
        if (
            len(self.si_exponents) != 7
            or any(type(value) is not int for value in self.si_exponents)
            or any(abs(value) > 16 for value in self.si_exponents)
        ):
            raise ValueError("component SI exponents are invalid")
        if self.coordinate_frame_id is not None:
            _uuid4(self.coordinate_frame_id, "component coordinate frame ID")
        if self.temporal_support is not None and type(
            self.temporal_support
        ) is not ExactTemporalSupport:
            raise TypeError("component temporal support has wrong type")
        if self.spatial_support is not None and type(
            self.spatial_support
        ) is not ExactSpatialSupport:
            raise TypeError("component spatial support has wrong type")
        if self.value_role is ComponentValueRole.COORDINATE:
            if self.coordinate_frame_id is None or self.unit_id is None:
                raise ValueError("coordinate descriptor needs frame and unit IDs")
        elif self.coordinate_frame_id is not None:
            raise ValueError("noncoordinate descriptor cannot carry coordinate frame")
        if self.value_role in (
            ComponentValueRole.BOOLEAN_CONTROL,
            ComponentValueRole.MISSING,
        ):
            if self.unit_id is not None or self.si_exponents != (0,) * 7:
                raise ValueError("discrete descriptor must be unitless")
        elif self.unit_id is None:
            raise ValueError("numeric descriptor needs a unit ID")


@dataclass(frozen=True, slots=True)
class DerivedObservationDescriptor:
    scale_id: str
    observation_id: str
    source_channel_id: str
    entity_ids: tuple[str, ...]
    role_candidate_ids: tuple[str, ...]
    quantity_id: str
    unit_id: str | None
    si_exponents: tuple[int, int, int, int, int, int, int]
    temporal_support: ExactTemporalSupport | None
    spatial_support: ExactSpatialSupport | None
    provenance_sha256: str
    source_observation_ids: tuple[str, ...]
    value_kind: ComponentValueKind
    component_refs: tuple[ComponentRef, ...]

    def __post_init__(self) -> None:
        _uuid4(self.scale_id, "derived observation scale ID")
        _uuid4(self.observation_id, "derived observation ID")
        _uuid4(self.source_channel_id, "derived source channel ID")
        _uuid4(self.quantity_id, "derived quantity ID")
        require_tuple(self.entity_ids, "derived observation entity IDs")
        require_tuple(
            self.role_candidate_ids,
            "derived observation role candidate IDs",
        )
        require_tuple(self.si_exponents, "derived observation SI exponents")
        require_tuple(
            self.source_observation_ids,
            "derived source observation IDs",
        )
        require_tuple(self.component_refs, "derived observation component refs")
        if not self.entity_ids or not self.role_candidate_ids:
            raise ValueError("derived observation needs entity and role IDs")
        for value in self.entity_ids:
            _uuid4(value, "derived entity ID")
        for value in self.role_candidate_ids:
            _uuid4(value, "derived role candidate ID")
        if self.unit_id is not None:
            _uuid4(self.unit_id, "derived unit ID")
        if (
            len(self.si_exponents) != 7
            or any(type(value) is not int for value in self.si_exponents)
            or any(abs(value) > 16 for value in self.si_exponents)
        ):
            raise ValueError("derived observation SI exponents are invalid")
        if self.temporal_support is not None and type(
            self.temporal_support
        ) is not ExactTemporalSupport:
            raise TypeError("derived temporal support has wrong type")
        if self.spatial_support is not None and type(
            self.spatial_support
        ) is not ExactSpatialSupport:
            raise TypeError("derived spatial support has wrong type")
        if (
            type(self.provenance_sha256) is not str
            or len(self.provenance_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.provenance_sha256)
        ):
            raise ValueError("derived observation provenance must be SHA-256")
        if (
            not self.source_observation_ids
            or self.source_observation_ids
            != tuple(sorted(self.source_observation_ids))
            or len(self.source_observation_ids)
            != len(set(self.source_observation_ids))
        ):
            raise ValueError("derived source observation IDs are not canonical")
        for value in self.source_observation_ids:
            _uuid4(value, "derived source observation ID")
        if type(self.value_kind) is not ComponentValueKind:
            raise TypeError("derived observation value kind has wrong type")
        if not self.component_refs:
            raise ValueError("derived observation needs component refs")
        if any(
            ref.scale_id != self.scale_id or ref.observation_id != self.observation_id
            for ref in self.component_refs
        ):
            raise ValueError("derived component ref leaves its observation")
        if tuple(ref.ordinal for ref in self.component_refs) != tuple(
            range(len(self.component_refs))
        ):
            raise ValueError("derived observation ordinals are not contiguous")
        if self.value_kind is not ComponentValueKind.NUMERIC_INTERVAL:
            if len(self.component_refs) != 1 or self.unit_id is not None:
                raise ValueError("derived discrete observation must be scalar unitless")
        elif self.unit_id is None:
            raise ValueError("derived numeric observation needs a unit ID")

    @property
    def descriptor_id(self) -> str:
        return stable_hash(self, prefix="phase2b_derived_observation_")


@dataclass(frozen=True, slots=True)
class ExactSparseTerm:
    input_ref: ComponentRef
    coefficient: ExactTransformAtom

    def __post_init__(self) -> None:
        if type(self.input_ref) is not ComponentRef:
            raise TypeError("sparse term input ref has wrong type")
        if type(self.coefficient) is not ExactTransformAtom:
            raise TypeError("sparse term coefficient has wrong type")
        if self.coefficient == ZERO:
            raise ValueError("sparse term coefficient cannot be zero")


@dataclass(frozen=True, slots=True)
class ExactSparseAffineRow:
    output_ref: ComponentRef
    terms: tuple[ExactSparseTerm, ...]
    offset: ExactTransformAtom = ZERO

    def __post_init__(self) -> None:
        if type(self.output_ref) is not ComponentRef:
            raise TypeError("sparse row output ref has wrong type")
        require_tuple(self.terms, "sparse row terms")
        if not self.terms or any(
            type(term) is not ExactSparseTerm for term in self.terms
        ):
            raise TypeError("sparse row needs exact nonempty terms")
        if self.terms != tuple(sorted(self.terms, key=lambda item: item.input_ref)):
            raise ValueError("sparse row terms are not canonical")
        refs = tuple(term.input_ref for term in self.terms)
        if len(refs) != len(set(refs)):
            raise ValueError("sparse row repeats an input component")
        if type(self.offset) is not ExactTransformAtom:
            raise TypeError("sparse row offset has wrong type")


@dataclass(frozen=True, slots=True)
class ExactDiscreteMapping:
    input_ref: ComponentRef
    output_ref: ComponentRef

    def __post_init__(self) -> None:
        if type(self.input_ref) is not ComponentRef or type(
            self.output_ref
        ) is not ComponentRef:
            raise TypeError("discrete mapping refs have wrong type")


@dataclass(frozen=True, slots=True)
class ExactPartitionGroup:
    input_refs: tuple[ComponentRef, ...]
    output_refs: tuple[ComponentRef, ...]

    def __post_init__(self) -> None:
        require_tuple(self.input_refs, "partition input refs")
        require_tuple(self.output_refs, "partition output refs")
        if not self.input_refs or not self.output_refs:
            raise ValueError("partition group cannot be empty")
        if self.input_refs != tuple(sorted(self.input_refs)):
            raise ValueError("partition input refs are not canonical")
        if self.output_refs != tuple(sorted(self.output_refs)):
            raise ValueError("partition output refs are not canonical")
        if len(self.input_refs) != len(set(self.input_refs)) or len(
            self.output_refs
        ) != len(set(self.output_refs)):
            raise ValueError("partition group repeats a component")


@dataclass(frozen=True, slots=True)
class IdentityTransformCertificate:
    semantics_version: str = EXACT_TRANSFORM_SEMANTICS_VERSION
    missing_policy: MissingValuePolicy = MissingValuePolicy.EXPLICIT_PRESERVE
    inverse_contract: str = "exact_one_to_one_identity"


@dataclass(frozen=True, slots=True)
class UnitConversionCertificate:
    source_unit_id: str
    target_unit_id: str
    factor: ExactTransformAtom
    inverse_factor: ExactTransformAtom
    missing_policy: MissingValuePolicy = MissingValuePolicy.EXPLICIT_PRESERVE
    orientation: str = "target_value_equals_source_value_times_factor"
    commutation_contract: str = "si_dimension_and_support_preserved"

    def __post_init__(self) -> None:
        _uuid4(self.source_unit_id, "source unit ID")
        _uuid4(self.target_unit_id, "target unit ID")
        if type(self.factor) is not ExactTransformAtom or type(
            self.inverse_factor
        ) is not ExactTransformAtom:
            raise TypeError("unit conversion factors have wrong type")


@dataclass(frozen=True, slots=True)
class CoordinateAffineCertificate:
    source_frame_id: str
    target_frame_id: str
    dimension: int
    inverse_rows: tuple[ExactSparseAffineRow, ...]
    missing_policy: MissingValuePolicy = MissingValuePolicy.EXPLICIT_PRESERVE
    support_contract: str = "exact_affine_box_enclosure"
    inverse_contract: str = "two_sided_exact_sparse_affine_inverse"

    def __post_init__(self) -> None:
        _uuid4(self.source_frame_id, "source coordinate frame ID")
        _uuid4(self.target_frame_id, "target coordinate frame ID")
        if type(self.dimension) is not int or not 1 <= self.dimension <= 4:
            raise ValueError("coordinate affine dimension must be one to four")
        require_tuple(self.inverse_rows, "coordinate inverse rows")


@dataclass(frozen=True, slots=True)
class TemporalAggregationCertificate:
    reducer: ReducerKind
    groups: tuple[ExactPartitionGroup, ...]
    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT
    boundary_policy: BoundaryPolicy = BoundaryPolicy.EXACT_PARTITION
    support_contract: str = (
        "same_clock_half_open_partition_modulo_final_endpoint"
    )

    def __post_init__(self) -> None:
        if type(self.reducer) is not ReducerKind:
            raise TypeError("temporal reducer has wrong type")
        require_tuple(self.groups, "temporal aggregation groups")


@dataclass(frozen=True, slots=True)
class SpatialAggregationCertificate:
    reducer: ReducerKind
    groups: tuple[ExactPartitionGroup, ...]
    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT
    boundary_policy: BoundaryPolicy = BoundaryPolicy.EXACT_PARTITION
    support_contract: str = (
        "same_frame_interior_disjoint_positive_volume_partition"
    )

    def __post_init__(self) -> None:
        if type(self.reducer) is not ReducerKind:
            raise TypeError("spatial reducer has wrong type")
        require_tuple(self.groups, "spatial aggregation groups")


@dataclass(frozen=True, slots=True)
class SamplingResolutionCertificate:
    axis: ComponentAxis
    selected_inputs: tuple[ComponentRef, ...]
    discarded_inputs: tuple[ComponentRef, ...]
    grid_points: tuple[tuple[ExactTransformAtom, ...], ...]
    grid_dimension: int
    grid_frame_id: str | None
    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT
    boundary_policy: BoundaryPolicy = BoundaryPolicy.VALID_ONLY
    kernel_contract: str = "single_series_ordered_exact_subselection"

    def __post_init__(self) -> None:
        if self.axis not in (ComponentAxis.TEMPORAL, ComponentAxis.SPATIAL):
            raise ValueError("sampling axis must be temporal or spatial")
        require_tuple(self.selected_inputs, "sampling selected inputs")
        require_tuple(self.discarded_inputs, "sampling discarded inputs")
        require_tuple(self.grid_points, "sampling grid points")
        if type(self.grid_dimension) is not int or not 1 <= self.grid_dimension <= 4:
            raise ValueError("sampling grid dimension must be one to four")
        if self.grid_frame_id is not None:
            _uuid4(self.grid_frame_id, "sampling grid frame ID")
        for point in self.grid_points:
            require_tuple(point, "sampling grid point")
            if len(point) != self.grid_dimension or any(
                type(value) is not ExactTransformAtom for value in point
            ):
                raise ValueError("sampling grid point dimension is invalid")


@dataclass(frozen=True, slots=True)
class EquivalentSplitMergeCertificate:
    direction: SplitMergeDirection
    groups: tuple[ExactPartitionGroup, ...]
    inverse_rows: tuple[ExactSparseAffineRow, ...]
    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT
    equivalence_contract: str = "extensive_sum_one_sided_exact_inverse"

    def __post_init__(self) -> None:
        if type(self.direction) is not SplitMergeDirection:
            raise TypeError("split/merge direction has wrong type")
        require_tuple(self.groups, "split/merge groups")
        require_tuple(self.inverse_rows, "split/merge inverse rows")


@dataclass(frozen=True, slots=True)
class CoarseGrainingCertificate:
    reducer: ReducerKind
    groups: tuple[ExactPartitionGroup, ...]
    quotient_class_ids: tuple[str, ...]
    source_commutation_rows: tuple[ExactSparseAffineRow, ...]
    target_commutation_rows: tuple[ExactSparseAffineRow, ...]
    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT
    boundary_policy: BoundaryPolicy = BoundaryPolicy.EXACT_PARTITION
    commutation_contract: str = "transform_after_source_equals_target_after_transform"

    def __post_init__(self) -> None:
        if type(self.reducer) is not ReducerKind:
            raise TypeError("coarse reducer has wrong type")
        require_tuple(self.groups, "coarse groups")
        require_tuple(self.quotient_class_ids, "coarse quotient class IDs")
        require_tuple(self.source_commutation_rows, "source commutation rows")
        require_tuple(self.target_commutation_rows, "target commutation rows")
        for value in self.quotient_class_ids:
            _uuid4(value, "quotient class ID")


TransformCertificate: TypeAlias = (
    IdentityTransformCertificate
    | UnitConversionCertificate
    | CoordinateAffineCertificate
    | TemporalAggregationCertificate
    | SpatialAggregationCertificate
    | SamplingResolutionCertificate
    | EquivalentSplitMergeCertificate
    | CoarseGrainingCertificate
)


_CERTIFICATE_TYPE_BY_OPERATION: Final = {
    TransformOperation.IDENTITY: IdentityTransformCertificate,
    TransformOperation.UNIT_CONVERSION: UnitConversionCertificate,
    TransformOperation.COORDINATE_AFFINE: CoordinateAffineCertificate,
    TransformOperation.TEMPORAL_AGGREGATION: TemporalAggregationCertificate,
    TransformOperation.SPATIAL_AGGREGATION: SpatialAggregationCertificate,
    TransformOperation.SAMPLING_RESOLUTION: SamplingResolutionCertificate,
    TransformOperation.EQUIVALENT_SPLIT_MERGE: (
        EquivalentSplitMergeCertificate
    ),
    TransformOperation.COARSE_GRAINING: CoarseGrainingCertificate,
}


@dataclass(frozen=True, slots=True)
class ExactTransformContract:
    transform_id: str
    operation: TransformOperation
    source_scale_id: str
    target_scale_id: str
    input_components: tuple[ComponentRef, ...]
    output_components: tuple[ComponentDescriptor, ...]
    output_observations: tuple[DerivedObservationDescriptor, ...]
    kernel_rows: tuple[ExactSparseAffineRow, ...]
    discrete_mappings: tuple[ExactDiscreteMapping, ...]
    certificate: TransformCertificate

    def __post_init__(self) -> None:
        _uuid4(self.transform_id, "typed transform ID")
        _uuid4(self.source_scale_id, "typed source scale ID")
        _uuid4(self.target_scale_id, "typed target scale ID")
        if type(self.operation) is not TransformOperation:
            raise TypeError("typed transform operation has wrong type")
        for name in (
            "input_components",
            "output_components",
            "output_observations",
            "kernel_rows",
            "discrete_mappings",
        ):
            require_tuple(getattr(self, name), f"typed transform {name}")
        if not self.input_components or not self.output_components:
            raise ValueError("typed transform needs input and output components")
        if self.input_components != tuple(sorted(self.input_components)):
            raise ValueError("typed transform inputs are not canonical")
        if self.output_components != tuple(
            sorted(self.output_components, key=lambda item: item.ref)
        ):
            raise ValueError("typed transform outputs are not canonical")
        if self.output_observations != tuple(
            sorted(
                self.output_observations,
                key=lambda item: (item.scale_id, item.observation_id),
            )
        ):
            raise ValueError("typed output observations are not canonical")
        if self.kernel_rows != tuple(
            sorted(self.kernel_rows, key=lambda item: item.output_ref)
        ):
            raise ValueError("typed transform kernel rows are not canonical")
        if self.discrete_mappings != tuple(
            sorted(
                self.discrete_mappings,
                key=lambda item: (item.output_ref, item.input_ref),
            )
        ):
            raise ValueError("typed transform discrete mappings are not canonical")

    @property
    def contract_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_transform_contract_")

    @property
    def semantics_id(self) -> str:
        return stable_hash(
            {
                "transform_id": self.transform_id,
                "operation": self.operation,
                "source_scale_id": self.source_scale_id,
                "target_scale_id": self.target_scale_id,
                "input_components": self.input_components,
                "output_components": self.output_components,
                "output_observations": tuple(
                    _descriptor_without_provenance(item)
                    for item in self.output_observations
                ),
                "kernel_rows": self.kernel_rows,
                "discrete_mappings": self.discrete_mappings,
                "certificate": self.certificate,
            },
            prefix="phase2b_transform_contract_semantics_",
        )


@dataclass(frozen=True, slots=True)
class PublicTransformEvidenceBundleV2:
    schema_version: str
    base_bundle: PublicEvidenceBundle
    observation_metadata: tuple[ObservationComponentMetadata, ...]
    transform_contracts: tuple[ExactTransformContract, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported public transform evidence schema")
        if not isinstance(self.base_bundle, PublicEvidenceBundle):
            raise TypeError("transform authority base bundle has wrong type")
        require_tuple(self.observation_metadata, "transform observation metadata")
        require_tuple(self.transform_contracts, "transform contracts")
        if not self.observation_metadata or not self.transform_contracts:
            raise ValueError("transform authority cannot be empty")
        if self.observation_metadata != tuple(
            sorted(self.observation_metadata, key=lambda item: item.observation_id)
        ):
            raise ValueError("transform observation metadata is not canonical")
        if self.transform_contracts != tuple(
            sorted(self.transform_contracts, key=lambda item: item.transform_id)
        ):
            raise ValueError("transform contracts are not canonical")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "base_bundle": self.base_bundle.to_mapping(),
            "observation_metadata": _canonical_value(self.observation_metadata),
            "transform_contracts": _canonical_value(self.transform_contracts),
        }

    @property
    def content_id(self) -> str:
        return stable_hash(
            self.to_mapping(),
            prefix="phase2b_public_transform_evidence_",
        )


def _canonical_value(value: object) -> object:
    exact_type = type(value)
    if exact_type in (str, int, float, bool, type(None)):
        return value
    if isinstance(value, Enum):
        return value.value
    if exact_type is tuple:
        return [_canonical_value(item) for item in value]
    if exact_type in _SCHEMA_DATACLASS_TYPES:
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    raise TypeError("value is outside the canonical transform schema")


@dataclass(frozen=True, slots=True)
class _ExactTransformPolicy:
    schema_version: str = PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION
    semantics_version: str = EXACT_TRANSFORM_SEMANTICS_VERSION
    uncertainty_policy_id: str = DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id
    phase2b_exact_freeze_id: str = FROZEN_PHASE2B_EXACT_FREEZE_ID
    rational_grid_id: str = FROZEN_RATIONAL_GRID_ID
    maximum_observations: int = 4_096
    maximum_components_per_observation: int = 256
    maximum_total_root_components: int = 65_536
    maximum_scales: int = 64
    maximum_edges: int = 256
    maximum_contracts: int = 256
    maximum_rows: int = 65_536
    maximum_nonzeros: int = 1_000_000
    maximum_output_components: int = 65_536
    maximum_contract_input_refs: int = 65_536
    maximum_discrete_mappings: int = 65_536
    maximum_certificate_memberships: int = 1_000_000
    maximum_auxiliary_rows: int = 65_536
    maximum_auxiliary_nonzeros: int = 1_000_000
    maximum_path_length: int = 64
    maximum_result_cells: int = 262_144
    maximum_scale_state_work: int = 1_000_000
    maximum_exact_operations: int = 1_000_000
    maximum_fraction_bit_length: int = 4_096
    maximum_authority_nodes: int = 300_000
    maximum_authority_text_characters: int = 3_000_000
    maximum_authority_integer_bit_length: int = 4_096

    def __post_init__(self) -> None:
        if (
            self.schema_version,
            self.semantics_version,
            self.uncertainty_policy_id,
            self.phase2b_exact_freeze_id,
            self.rational_grid_id,
        ) != (
            PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
            EXACT_TRANSFORM_SEMANTICS_VERSION,
            DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id,
            FROZEN_PHASE2B_EXACT_FREEZE_ID,
            FROZEN_RATIONAL_GRID_ID,
        ):
            raise ValueError("exact transform semantic identity drift")
        if (
            self.maximum_observations,
            self.maximum_components_per_observation,
            self.maximum_total_root_components,
            self.maximum_scales,
            self.maximum_edges,
            self.maximum_contracts,
            self.maximum_rows,
            self.maximum_nonzeros,
            self.maximum_output_components,
            self.maximum_contract_input_refs,
            self.maximum_discrete_mappings,
            self.maximum_certificate_memberships,
            self.maximum_auxiliary_rows,
            self.maximum_auxiliary_nonzeros,
            self.maximum_path_length,
            self.maximum_result_cells,
            self.maximum_scale_state_work,
            self.maximum_exact_operations,
            self.maximum_fraction_bit_length,
            self.maximum_authority_nodes,
            self.maximum_authority_text_characters,
            self.maximum_authority_integer_bit_length,
        ) != (
            4_096,
            256,
            65_536,
            64,
            256,
            256,
            65_536,
            1_000_000,
            65_536,
            65_536,
            65_536,
            1_000_000,
            65_536,
            1_000_000,
            64,
            262_144,
            1_000_000,
            1_000_000,
            4_096,
            300_000,
            3_000_000,
            4_096,
        ):
            raise ValueError("exact transform resource budget drift")

    @property
    def semantics_id(self) -> str:
        return stable_hash(
            (
                self.semantics_version,
                tuple(
                    (operation, certificate.__name__)
                    for operation, certificate in sorted(
                        _CERTIFICATE_TYPE_BY_OPERATION.items(),
                        key=lambda item: item[0].value,
                    )
                ),
                "legacy_nonidentity_parameters_must_be_empty",
                "forest_unique_root_path_only",
                "derived_observation_shapes_are_operation_specific",
                "sampling_v1_single_series_distinct_points_scalar_outputs",
                "exact_sparse_affine_interval_enclosure",
            ),
            prefix="phase2b_exact_transform_semantics_",
        )

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_transform_policy_")


_DEFAULT_POLICY: Final = _ExactTransformPolicy()


class _ResourceLimit(RuntimeError):
    pass


_SCHEMA_DATACLASS_TYPES: Final = frozenset(
    {
        AggregationEdge,
        AggregationGraph,
        BooleanValue,
        CoarseGrainingCertificate,
        ComponentDescriptor,
        ComponentRef,
        CoordinateAffineCertificate,
        DerivedObservationDescriptor,
        EntityCandidate,
        EquivalentSplitMergeCertificate,
        ExactDiscreteMapping,
        ExactPartitionGroup,
        ExactSparseAffineRow,
        ExactSparseTerm,
        ExactSpatialSupport,
        ExactTemporalSupport,
        ExactTransformAtom,
        ExactTransformContract,
        IdentityTransformCertificate,
        MeasurementUncertainty,
        NumericInterval,
        NumericValue,
        ObservationComponentMetadata,
        PublicEvidenceBundle,
        PublicTransformEvidenceBundleV2,
        SamplingResolutionCertificate,
        SpatialAggregationCertificate,
        SpatialSupport,
        TaskTarget,
        TemporalAggregationCertificate,
        TemporalSupport,
        TransformSpec,
        TypedObservation,
        UnitConversionCertificate,
        UnitDimension,
    }
)
_SCHEMA_ENUM_TYPES: Final = frozenset(
    {
        BoundaryPolicy,
        ComponentAxis,
        ComponentValueKind,
        ComponentValueRole,
        Missingness,
        MissingValuePolicy,
        ReducerKind,
        SplitMergeDirection,
        TransformOperation,
        UncertaintyModel,
    }
)


@dataclass(slots=True)
class _AuthorityTreeBudget:
    policy: _ExactTransformPolicy
    nodes: int = 0
    text_characters: int = 0

    def visit(self, root: object) -> None:
        stack = [root]
        while stack:
            current = stack.pop()
            self.nodes += 1
            if self.nodes > self.policy.maximum_authority_nodes:
                raise _ResourceLimit("RESOURCE_LIMIT:authority_nodes")
            exact_type = type(current)
            if exact_type is str:
                self.text_characters += len(current)
                if (
                    self.text_characters
                    > self.policy.maximum_authority_text_characters
                ):
                    raise _ResourceLimit(
                        "RESOURCE_LIMIT:authority_text_characters"
                    )
                continue
            if exact_type is int:
                if (
                    current.bit_length()
                    > self.policy.maximum_authority_integer_bit_length
                ):
                    raise _ResourceLimit(
                        "RESOURCE_LIMIT:authority_integer_bit_length"
                    )
                continue
            if exact_type in (float, bool, type(None)):
                continue
            if exact_type in _SCHEMA_ENUM_TYPES:
                continue
            if exact_type is tuple:
                if len(current) > self.policy.maximum_authority_nodes - self.nodes:
                    raise _ResourceLimit("RESOURCE_LIMIT:authority_nodes")
                stack.extend(reversed(current))
                continue
            if exact_type in _SCHEMA_DATACLASS_TYPES:
                stack.extend(
                    getattr(current, field.name)
                    for field in reversed(fields(current))
                )
                continue
            raise TypeError("authority contains a non-exact schema node")


@dataclass(frozen=True, slots=True)
class ExactTransformedComponent:
    descriptor: ComponentDescriptor
    value_kind: ComponentValueKind
    numeric_interval: ExactTransformInterval | None
    boolean_value: bool | None
    uncertainty_compilation_ids: tuple[str, ...]
    observation_descriptor_id: str
    ordered_transform_path_ids: tuple[str, ...]
    ordered_contract_semantics_ids: tuple[str, ...]
    wrapper_content_id: str
    base_bundle_content_id: str
    uncertainty_result_id: str
    transform_policy_id: str
    transform_semantics_id: str
    contract_commitment_id: str
    graph_commitment_id: str

    def __post_init__(self) -> None:
        if type(self.descriptor) is not ComponentDescriptor:
            raise TypeError("transformed component descriptor has wrong type")
        if type(self.value_kind) is not ComponentValueKind:
            raise TypeError("transformed component value kind has wrong type")
        for name in (
            "uncertainty_compilation_ids",
            "ordered_transform_path_ids",
            "ordered_contract_semantics_ids",
        ):
            require_tuple(getattr(self, name), f"transformed component {name}")
        if self.uncertainty_compilation_ids != tuple(
            sorted(self.uncertainty_compilation_ids)
        ):
            raise ValueError("transformed component lineage is not canonical")
        if len(self.ordered_transform_path_ids) != len(
            self.ordered_contract_semantics_ids
        ):
            raise ValueError("transformed component path roots disagree")
        identities = (
            self.wrapper_content_id,
            self.observation_descriptor_id,
            self.base_bundle_content_id,
            self.uncertainty_result_id,
            self.transform_policy_id,
            self.transform_semantics_id,
            self.contract_commitment_id,
            self.graph_commitment_id,
        )
        if not all(type(value) is str and value for value in identities):
            raise ValueError("transformed component provenance is incomplete")
        if self.value_kind is ComponentValueKind.NUMERIC_INTERVAL:
            if (
                type(self.numeric_interval) is not ExactTransformInterval
                or self.boolean_value is not None
            ):
                raise ValueError("numeric transformed component has wrong payload")
        elif self.value_kind is ComponentValueKind.BOOLEAN:
            if type(self.boolean_value) is not bool or self.numeric_interval is not None:
                raise ValueError("Boolean transformed component has wrong payload")
        elif self.numeric_interval is not None or self.boolean_value is not None:
            raise ValueError("missing transformed component cannot carry a value")

    @property
    def component_receipt_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_transformed_component_")


@dataclass(frozen=True, slots=True)
class ExactTransformFailure:
    error_code: str
    transform_id: str | None = None
    scale_id: str | None = None
    component_ref: ComponentRef | None = None

    def __post_init__(self) -> None:
        if type(self.error_code) is not str or not self.error_code:
            raise ValueError("exact transform failure needs an error code")
        if self.transform_id is not None:
            _uuid4(self.transform_id, "failure transform ID")
        if self.scale_id is not None:
            _uuid4(self.scale_id, "failure scale ID")
        if self.component_ref is not None and type(
            self.component_ref
        ) is not ComponentRef:
            raise TypeError("failure component ref has wrong type")


@dataclass(frozen=True, slots=True)
class ExactTransformCompilation:
    disposition: TransformCompilationDisposition
    reason: str
    wrapper_content_id: str
    base_bundle_content_id: str
    uncertainty_result_id: str
    uncertainty_policy_id: str
    phase2b_exact_freeze_id: str
    rational_grid_id: str
    transform_policy_id: str
    transform_semantics_id: str
    contract_commitment_id: str
    graph_commitment_id: str
    uncertainty_receipt: BundleUncertaintyCompilation
    observations: tuple[DerivedObservationDescriptor, ...]
    components: tuple[ExactTransformedComponent, ...]
    failures: tuple[ExactTransformFailure, ...]

    def __post_init__(self) -> None:
        if type(self.disposition) is not TransformCompilationDisposition:
            raise TypeError("exact transform disposition has wrong type")
        require_tuple(self.observations, "exact transformed observations")
        require_tuple(self.components, "exact transformed components")
        require_tuple(self.failures, "exact transform failures")
        if type(self.uncertainty_receipt) is not BundleUncertaintyCompilation:
            raise TypeError("exact transform uncertainty receipt has wrong type")
        identities = (
            self.reason,
            self.wrapper_content_id,
            self.base_bundle_content_id,
            self.uncertainty_result_id,
            self.uncertainty_policy_id,
            self.phase2b_exact_freeze_id,
            self.rational_grid_id,
            self.transform_policy_id,
            self.transform_semantics_id,
            self.contract_commitment_id,
            self.graph_commitment_id,
        )
        if not all(type(value) is str and value for value in identities):
            raise ValueError("exact transform compilation identity is incomplete")
        if self.disposition is TransformCompilationDisposition.COMPLETE:
            if not self.observations or not self.components or self.failures:
                raise ValueError("complete exact transform result needs only cells")
        elif self.observations or self.components or not self.failures:
            raise ValueError("abstaining exact transform result cannot be partial")

    @property
    def result_id(self) -> str:
        return stable_hash(self, prefix="phase2b_exact_transform_result_")


@dataclass(frozen=True, slots=True)
class ExactTransformPreflightRejection:
    disposition: TransformCompilationDisposition
    reason: str
    base_bundle_id: str
    wrapper_schema_version: str
    transform_policy_id: str
    transform_semantics_id: str
    wrapper_content_id: None = None
    base_bundle_content_id: None = None
    uncertainty_receipt: None = None
    components: tuple[()] = ()

    def __post_init__(self) -> None:
        if self.disposition is not TransformCompilationDisposition.ABSTAIN:
            raise ValueError("transform preflight rejection must abstain")
        if not all(
            type(value) is str and value
            for value in (
                self.reason,
                self.base_bundle_id,
                self.wrapper_schema_version,
                self.transform_policy_id,
                self.transform_semantics_id,
            )
        ):
            raise ValueError("transform preflight rejection is incomplete")
        if (
            self.wrapper_content_id is not None
            or self.base_bundle_content_id is not None
            or self.uncertainty_receipt is not None
            or self.components
        ):
            raise ValueError("preflight rejection cannot carry committed receipts")


@dataclass(frozen=True, slots=True)
class _ComponentState:
    descriptor: ComponentDescriptor
    observation: DerivedObservationDescriptor
    value_kind: ComponentValueKind
    numeric_interval: ExactTransformInterval | None
    boolean_value: bool | None
    uncertainty_compilation_ids: tuple[str, ...]


@dataclass(slots=True)
class _KernelBudget:
    policy: _ExactTransformPolicy
    operations: int = 0

    def check_fraction(self, value: Fraction) -> None:
        if (
            value.numerator.bit_length()
            > self.policy.maximum_fraction_bit_length
            or value.denominator.bit_length()
            > self.policy.maximum_fraction_bit_length
        ):
            raise _ResourceLimit("RESOURCE_LIMIT:fraction_bit_length")

    def interval(
        self,
        lower: Fraction,
        upper: Fraction,
    ) -> ExactTransformInterval:
        self.operations += 1
        if self.operations > self.policy.maximum_exact_operations:
            raise _ResourceLimit("RESOURCE_LIMIT:exact_operations")
        self.check_fraction(lower)
        self.check_fraction(upper)
        return ExactTransformInterval.from_fractions(lower, upper)

    def add_fraction(self, left: Fraction, right: Fraction) -> Fraction:
        self.operations += 1
        if self.operations > self.policy.maximum_exact_operations:
            raise _ResourceLimit("RESOURCE_LIMIT:exact_operations")
        result = left + right
        self.check_fraction(result)
        return result

    def multiply_fraction(self, left: Fraction, right: Fraction) -> Fraction:
        self.operations += 1
        if self.operations > self.policy.maximum_exact_operations:
            raise _ResourceLimit("RESOURCE_LIMIT:exact_operations")
        result = left * right
        self.check_fraction(result)
        return result


def _scale_interval(
    budget: _KernelBudget,
    value: ExactTransformInterval,
    factor: Fraction,
) -> ExactTransformInterval:
    if factor >= 0:
        return budget.interval(
            budget.multiply_fraction(value.lower_fraction, factor),
            budget.multiply_fraction(value.upper_fraction, factor),
        )
    return budget.interval(
        budget.multiply_fraction(value.upper_fraction, factor),
        budget.multiply_fraction(value.lower_fraction, factor),
    )


def _add_intervals(
    budget: _KernelBudget,
    left: ExactTransformInterval,
    right: ExactTransformInterval,
) -> ExactTransformInterval:
    return budget.interval(
        budget.add_fraction(left.lower_fraction, right.lower_fraction),
        budget.add_fraction(left.upper_fraction, right.upper_fraction),
    )


def _evaluate_sparse_row(
    budget: _KernelBudget,
    row: ExactSparseAffineRow,
    source: dict[ComponentRef, _ComponentState],
) -> tuple[ExactTransformInterval, tuple[str, ...]]:
    result = budget.interval(
        row.offset.as_fraction(),
        row.offset.as_fraction(),
    )
    lineage: set[str] = set()
    for term in row.terms:
        state = source[term.input_ref]
        if (
            state.value_kind is not ComponentValueKind.NUMERIC_INTERVAL
            or state.numeric_interval is None
        ):
            raise ValueError("kernel term references a nonnumeric component")
        if term.coefficient.as_fraction() == 0:
            raise ValueError("kernel term cannot carry a zero coefficient")
        result = _add_intervals(
            budget,
            result,
            _scale_interval(
                budget,
                state.numeric_interval,
                term.coefficient.as_fraction(),
            ),
        )
        lineage.update(state.uncertainty_compilation_ids)
    return result, tuple(sorted(lineage))


def _preflight_rejection(
    reason: str,
    authority: PublicTransformEvidenceBundleV2,
    policy: _ExactTransformPolicy,
) -> ExactTransformPreflightRejection:
    return ExactTransformPreflightRejection(
        disposition=TransformCompilationDisposition.ABSTAIN,
        reason=reason,
        base_bundle_id=authority.base_bundle.bundle_id,
        wrapper_schema_version=authority.schema_version,
        transform_policy_id=policy.policy_id,
        transform_semantics_id=policy.semantics_id,
    )


def _shallow_preflight(
    authority: PublicTransformEvidenceBundleV2,
    policy: _ExactTransformPolicy,
) -> str | None:
    if type(authority.base_bundle) is not PublicEvidenceBundle:
        raise TypeError("transform base authority must have exact root type")
    if (
        type(authority.schema_version) is not str
        or type(authority.base_bundle.bundle_id) is not str
        or type(authority.base_bundle.aggregation_graph) is not AggregationGraph
        or type(authority.observation_metadata) is not tuple
        or type(authority.transform_contracts) is not tuple
    ):
        raise TypeError("transform preflight receipt scalars have wrong type")
    if len(authority.observation_metadata) > policy.maximum_observations:
        return "RESOURCE_LIMIT:metadata_count"
    if len(authority.transform_contracts) > policy.maximum_contracts:
        return "RESOURCE_LIMIT:contract_count"
    if any(
        type(item) is not ObservationComponentMetadata
        for item in authority.observation_metadata
    ):
        raise TypeError("transform metadata must have exact schema type")
    if any(
        type(item) is not ExactTransformContract
        for item in authority.transform_contracts
    ):
        raise TypeError("transform contracts must have exact schema type")
    base = authority.base_bundle
    sequences = (
        authority.observation_metadata,
        authority.transform_contracts,
        base.observations,
        base.entity_candidates,
        base.role_ids,
        base.quantity_ids,
        base.transform_catalog,
        base.missingness_mask,
        base.aggregation_graph.scale_ids,
        base.aggregation_graph.root_scale_ids,
        base.aggregation_graph.edges,
    )
    if any(type(value) is not tuple for value in sequences):
        raise TypeError("transform authority sequences must be exact tuples")
    if len(base.observations) > policy.maximum_observations:
        return "RESOURCE_LIMIT:observation_count"
    graph = base.aggregation_graph
    if len(graph.scale_ids) > policy.maximum_scales:
        return "RESOURCE_LIMIT:scale_count"
    if len(graph.edges) > policy.maximum_edges:
        return "RESOURCE_LIMIT:edge_count"
    widths = tuple(len(item.component_ids) for item in authority.observation_metadata)
    if any(width > policy.maximum_components_per_observation for width in widths):
        return "RESOURCE_LIMIT:component_width"
    if sum(widths) > policy.maximum_total_root_components:
        return "RESOURCE_LIMIT:root_component_count"
    rows = sum(len(contract.kernel_rows) for contract in authority.transform_contracts)
    outputs = sum(
        len(contract.output_components)
        for contract in authority.transform_contracts
    )
    if rows > policy.maximum_rows:
        return "RESOURCE_LIMIT:row_count"
    if outputs > policy.maximum_output_components:
        return "RESOURCE_LIMIT:output_component_count"
    nonzeros = sum(
        len(row.terms)
        for contract in authority.transform_contracts
        for row in contract.kernel_rows
    )
    if nonzeros > policy.maximum_nonzeros:
        return "RESOURCE_LIMIT:nonzero_count"
    if sum(widths) + outputs > policy.maximum_result_cells:
        return "RESOURCE_LIMIT:result_cell_count"
    return None


def _certificate_resource_counts(
    certificate: TransformCertificate,
) -> tuple[int, int, int]:
    """Return membership, auxiliary-row, and auxiliary-nonzero counts."""

    memberships = 0
    auxiliary_rows: tuple[ExactSparseAffineRow, ...] = ()
    if type(certificate) in (
        TemporalAggregationCertificate,
        SpatialAggregationCertificate,
    ):
        memberships = sum(
            len(group.input_refs) + len(group.output_refs)
            for group in certificate.groups
        )
    elif type(certificate) is SamplingResolutionCertificate:
        memberships = (
            len(certificate.selected_inputs)
            + len(certificate.discarded_inputs)
            + sum(len(point) for point in certificate.grid_points)
        )
    elif type(certificate) is CoordinateAffineCertificate:
        auxiliary_rows = certificate.inverse_rows
    elif type(certificate) is EquivalentSplitMergeCertificate:
        memberships = sum(
            len(group.input_refs) + len(group.output_refs)
            for group in certificate.groups
        )
        auxiliary_rows = certificate.inverse_rows
    elif type(certificate) is CoarseGrainingCertificate:
        memberships = sum(
            len(group.input_refs) + len(group.output_refs)
            for group in certificate.groups
        ) + len(certificate.quotient_class_ids)
        auxiliary_rows = (
            *certificate.source_commutation_rows,
            *certificate.target_commutation_rows,
        )
    return (
        memberships,
        len(auxiliary_rows),
        sum(len(row.terms) for row in auxiliary_rows),
    )


def _composition_work_bound(
    outer: tuple[ExactSparseAffineRow, ...],
    inner: tuple[ExactSparseAffineRow, ...],
    *,
    remaining: int,
) -> int | None:
    """Linearly bound the exact multiply/add work of one composition.

    ``None`` means the bound exceeds ``remaining``.  Duplicate inner outputs
    are deliberately charged using the largest duplicate row; semantic
    validation rejects the encoding later without letting this preflight turn
    into a quadratic scan.
    """

    inner_widths: dict[ComponentRef, int] = {}
    for row in inner:
        current = inner_widths.get(row.output_ref, 0)
        if len(row.terms) > current:
            inner_widths[row.output_ref] = len(row.terms)
    work = 0
    for row in outer:
        for term in row.terms:
            # factor*inner.offset + offset, then multiply/add per expanded term.
            work += 2 + 2 * inner_widths.get(term.input_ref, 0)
            if work > remaining:
                return None
    return work


def _detailed_resource_preflight(
    authority: PublicTransformEvidenceBundleV2,
    policy: _ExactTransformPolicy,
) -> str | None:
    input_refs = sum(
        len(contract.input_components)
        for contract in authority.transform_contracts
    )
    discrete = sum(
        len(contract.discrete_mappings)
        for contract in authority.transform_contracts
    )
    counts = tuple(
        _certificate_resource_counts(contract.certificate)
        for contract in authority.transform_contracts
    )
    memberships = sum(item[0] for item in counts)
    auxiliary_rows = sum(item[1] for item in counts)
    auxiliary_nonzeros = sum(item[2] for item in counts)
    if input_refs > policy.maximum_contract_input_refs:
        return "RESOURCE_LIMIT:contract_input_refs"
    if discrete > policy.maximum_discrete_mappings:
        return "RESOURCE_LIMIT:discrete_mappings"
    if memberships > policy.maximum_certificate_memberships:
        return "RESOURCE_LIMIT:certificate_memberships"
    if auxiliary_rows > policy.maximum_auxiliary_rows:
        return "RESOURCE_LIMIT:auxiliary_rows"
    if auxiliary_nonzeros > policy.maximum_auxiliary_nonzeros:
        return "RESOURCE_LIMIT:auxiliary_nonzeros"
    main_nonzeros = sum(
        len(row.terms)
        for contract in authority.transform_contracts
        for row in contract.kernel_rows
    )
    work = (
        input_refs
        + discrete
        + memberships
        + auxiliary_rows
        + auxiliary_nonzeros
        + main_nonzeros
        + sum(
            len(contract.output_components)
            + sum(
                len(item.component_refs)
                for item in contract.output_observations
            )
            for contract in authority.transform_contracts
        )
    )
    if work > policy.maximum_scale_state_work:
        return "RESOURCE_LIMIT:scale_state_work"
    composition_work = 0
    other_exact_work = 0
    for contract in authority.transform_contracts:
        certificate = contract.certificate
        if type(certificate) is CoordinateAffineCertificate:
            compositions = (
                (certificate.inverse_rows, contract.kernel_rows),
                (contract.kernel_rows, certificate.inverse_rows),
            )
        elif type(certificate) is EquivalentSplitMergeCertificate:
            forward = contract.kernel_rows
            inverse = certificate.inverse_rows
            compositions = (
                (
                    (inverse, forward)
                    if certificate.direction is SplitMergeDirection.SPLIT
                    else (forward, inverse)
                ),
            )
        elif type(certificate) is CoarseGrainingCertificate:
            compositions = (
                (contract.kernel_rows, certificate.source_commutation_rows),
                (certificate.target_commutation_rows, contract.kernel_rows),
            )
        else:
            compositions = ()
        for outer, inner in compositions:
            remaining = policy.maximum_exact_operations - composition_work
            estimated = _composition_work_bound(
                outer,
                inner,
                remaining=remaining,
            )
            if estimated is None:
                return "RESOURCE_LIMIT:composition_work"
            composition_work += estimated
        if type(certificate) is SpatialAggregationCertificate:
            for group in certificate.groups:
                count = len(group.input_refs)
                other_exact_work += count * (count - 1) // 2
                other_exact_work += (count + 1) * 4 + count
                if (
                    composition_work + other_exact_work
                    > policy.maximum_exact_operations
                ):
                    return "RESOURCE_LIMIT:spatial_partition_work"
        elif type(certificate) in (
            TemporalAggregationCertificate,
            EquivalentSplitMergeCertificate,
            CoarseGrainingCertificate,
        ):
            other_exact_work += sum(
                len(group.input_refs) for group in certificate.groups
            )
            if (
                composition_work + other_exact_work
                > policy.maximum_exact_operations
            ):
                return "RESOURCE_LIMIT:certificate_arithmetic_work"
    return None


def _exact_temporal_from_observation(
    observation: TypedObservation,
) -> ExactTemporalSupport | None:
    return (
        None
        if observation.temporal_support is None
        else ExactTemporalSupport.from_wire(observation.temporal_support)
    )


def _exact_spatial_from_observation(
    observation: TypedObservation,
) -> ExactSpatialSupport | None:
    return (
        None
        if observation.spatial_support is None
        else ExactSpatialSupport.from_wire(observation.spatial_support)
    )


def _root_descriptor(
    observation: TypedObservation,
    metadata: ObservationComponentMetadata,
    ordinal: int,
) -> ComponentDescriptor:
    return ComponentDescriptor(
        ref=ComponentRef(
            metadata.scale_id,
            metadata.observation_id,
            ordinal,
            metadata.component_ids[ordinal],
        ),
        axis=metadata.axis,
        value_role=metadata.value_role,
        unit_id=metadata.unit_id,
        si_exponents=observation.unit_dimension.si_exponents,
        coordinate_frame_id=metadata.coordinate_frame_id,
        temporal_support=_exact_temporal_from_observation(observation),
        spatial_support=_exact_spatial_from_observation(observation),
    )


def _root_observation_descriptor(
    observation: TypedObservation,
    metadata: ObservationComponentMetadata,
    value_kind: ComponentValueKind,
) -> DerivedObservationDescriptor:
    refs = tuple(
        ComponentRef(
            metadata.scale_id,
            metadata.observation_id,
            ordinal,
            component_id,
        )
        for ordinal, component_id in enumerate(metadata.component_ids)
    )
    return DerivedObservationDescriptor(
        scale_id=metadata.scale_id,
        observation_id=metadata.observation_id,
        source_channel_id=observation.source_channel_id,
        entity_ids=observation.entity_ids,
        role_candidate_ids=observation.role_candidate_ids,
        quantity_id=observation.quantity_id,
        unit_id=metadata.unit_id,
        si_exponents=observation.unit_dimension.si_exponents,
        temporal_support=_exact_temporal_from_observation(observation),
        spatial_support=_exact_spatial_from_observation(observation),
        provenance_sha256=observation.provenance_sha256,
        source_observation_ids=(observation.observation_id,),
        value_kind=value_kind,
        component_refs=refs,
    )


def _build_root_states(
    authority: PublicTransformEvidenceBundleV2,
    receipt: BundleUncertaintyCompilation,
) -> tuple[
    dict[str, dict[ComponentRef, _ComponentState]],
    dict[str, tuple[DerivedObservationDescriptor, ...]],
]:
    base = authority.base_bundle
    metadata_by_observation = {
        item.observation_id: item for item in authority.observation_metadata
    }
    compiled_by_observation = {
        item.observation_id: item for item in receipt.observations
    }
    states = {
        scale_id: {} for scale_id in base.aggregation_graph.root_scale_ids
    }
    observations_by_scale: dict[str, list[DerivedObservationDescriptor]] = {
        scale_id: [] for scale_id in base.aggregation_graph.root_scale_ids
    }
    for observation in base.observations:
        metadata = metadata_by_observation[observation.observation_id]
        compiled = compiled_by_observation[observation.observation_id]
        value_kind = (
            ComponentValueKind.NUMERIC_INTERVAL
            if compiled.value_kind is ObservationValueKind.NUMERIC_INTERVAL
            else ComponentValueKind.BOOLEAN
            if compiled.value_kind is ObservationValueKind.BOOLEAN
            else ComponentValueKind.MISSING
        )
        observation_descriptor = _root_observation_descriptor(
            observation,
            metadata,
            value_kind,
        )
        observations_by_scale[metadata.scale_id].append(observation_descriptor)
        for ordinal, component_id in enumerate(metadata.component_ids):
            descriptor = _root_descriptor(observation, metadata, ordinal)
            if compiled.value_kind is ObservationValueKind.NUMERIC_INTERVAL:
                exact = compiled.numeric_bounds[ordinal]
                state = _ComponentState(
                    descriptor,
                    observation_descriptor,
                    ComponentValueKind.NUMERIC_INTERVAL,
                    ExactTransformInterval.from_fractions(
                        exact.lower_fraction,
                        exact.upper_fraction,
                    ),
                    None,
                    (compiled.compilation_id,),
                )
            elif compiled.value_kind is ObservationValueKind.BOOLEAN:
                state = _ComponentState(
                    descriptor,
                    observation_descriptor,
                    ComponentValueKind.BOOLEAN,
                    None,
                    compiled.boolean_value,
                    (compiled.compilation_id,),
                )
            else:
                state = _ComponentState(
                    descriptor,
                    observation_descriptor,
                    ComponentValueKind.MISSING,
                    None,
                    None,
                    (compiled.compilation_id,),
                )
            if descriptor.ref in states[metadata.scale_id]:
                raise ValueError("duplicate root component ref")
            states[metadata.scale_id][descriptor.ref] = state
    return states, {
        scale_id: tuple(
            sorted(values, key=lambda item: item.observation_id)
        )
        for scale_id, values in observations_by_scale.items()
    }


def _validate_metadata(authority: PublicTransformEvidenceBundleV2) -> str | None:
    base = authority.base_bundle
    observations = {item.observation_id: item for item in base.observations}
    metadata = {item.observation_id: item for item in authority.observation_metadata}
    if len(metadata) != len(authority.observation_metadata):
        return "duplicate_observation_metadata"
    if set(metadata) != set(observations):
        return "observation_metadata_coverage_mismatch"
    roots = set(base.aggregation_graph.root_scale_ids)
    all_component_ids: set[str] = set()
    for observation_id, item in metadata.items():
        observation = observations[observation_id]
        if item.scale_id not in roots:
            return "base_observation_scale_is_not_root"
        value = observation.value
        width = (
            len(value.values)
            if isinstance(value, NumericValue)
            else len(value.lower)
            if isinstance(value, NumericInterval)
            else 1
        )
        if len(item.component_ids) != width:
            return "observation_component_coverage_mismatch"
        if any(value in all_component_ids for value in item.component_ids):
            return "component_id_reused_across_observations"
        all_component_ids.update(item.component_ids)
        if observation.missingness is Missingness.MISSING:
            if item.value_role is not ComponentValueRole.MISSING:
                return "missing_observation_metadata_role_mismatch"
            if observation.unit_dimension.si_exponents != (0,) * 7:
                return "missing_observation_must_be_dimensionless"
        elif isinstance(value, BooleanValue):
            if (
                item.value_role is not ComponentValueRole.BOOLEAN_CONTROL
                or item.axis is not ComponentAxis.CONTROL
            ):
                return "boolean_observation_metadata_role_mismatch"
            if observation.unit_dimension.si_exponents != (0,) * 7:
                return "boolean_observation_must_be_dimensionless"
        elif item.value_role in (
            ComponentValueRole.BOOLEAN_CONTROL,
            ComponentValueRole.MISSING,
        ):
            return "numeric_observation_metadata_role_mismatch"
        if item.axis is ComponentAxis.TEMPORAL and observation.temporal_support is None:
            return "temporal_axis_without_temporal_support"
        if item.axis is ComponentAxis.SPATIAL and observation.spatial_support is None:
            return "spatial_axis_without_spatial_support"
        if item.value_role is ComponentValueRole.COORDINATE and (
            observation.spatial_support is None
            or item.coordinate_frame_id != observation.spatial_support.frame_id
        ):
            return "coordinate_frame_metadata_mismatch"
    if any(not value for value in _build_metadata_scale_counts(authority).values()):
        return "root_scale_without_observation_components"
    return None


def _build_metadata_scale_counts(
    authority: PublicTransformEvidenceBundleV2,
) -> dict[str, int]:
    counts = {
        scale_id: 0
        for scale_id in authority.base_bundle.aggregation_graph.root_scale_ids
    }
    for item in authority.observation_metadata:
        if item.scale_id in counts:
            counts[item.scale_id] += len(item.component_ids)
    return counts


def _fraction_bits_valid(value: ExactTransformAtom, policy: _ExactTransformPolicy) -> bool:
    return (
        value.numerator.bit_length() <= policy.maximum_fraction_bit_length
        and value.denominator.bit_length() <= policy.maximum_fraction_bit_length
    )


def _validate_certificate_literals(
    certificate: TransformCertificate,
    policy: _ExactTransformPolicy,
) -> str | None:
    atoms: list[ExactTransformAtom] = []

    def rows_canonical(rows: tuple[ExactSparseAffineRow, ...]) -> bool:
        return (
            rows == tuple(sorted(rows, key=lambda item: item.output_ref))
            and len({row.output_ref for row in rows}) == len(rows)
            and all(
                term.coefficient != ZERO
                for row in rows
                for term in row.terms
            )
        )

    def groups_canonical(groups: tuple[ExactPartitionGroup, ...]) -> bool:
        return groups == tuple(
            sorted(
                groups,
                key=lambda item: (item.input_refs, item.output_refs),
            )
        )

    if type(certificate) is IdentityTransformCertificate:
        if (
            certificate.semantics_version != EXACT_TRANSFORM_SEMANTICS_VERSION
            or certificate.missing_policy
            is not MissingValuePolicy.EXPLICIT_PRESERVE
            or certificate.inverse_contract != "exact_one_to_one_identity"
        ):
            return "identity_certificate_literal_drift"
    elif type(certificate) is UnitConversionCertificate:
        atoms.extend((certificate.factor, certificate.inverse_factor))
        if (
            certificate.missing_policy
            is not MissingValuePolicy.EXPLICIT_PRESERVE
            or certificate.orientation
            != "target_value_equals_source_value_times_factor"
            or certificate.commutation_contract
            != "si_dimension_and_support_preserved"
            or certificate.factor.as_fraction() <= 0
            or certificate.inverse_factor.as_fraction() <= 0
            or certificate.factor.numerator
            != certificate.inverse_factor.denominator
            or certificate.factor.denominator
            != certificate.inverse_factor.numerator
        ):
            return "unit_conversion_certificate_invalid"
    elif type(certificate) is CoordinateAffineCertificate:
        if (
            certificate.missing_policy
            is not MissingValuePolicy.EXPLICIT_PRESERVE
            or certificate.support_contract != "exact_affine_box_enclosure"
            or certificate.inverse_contract
            != "two_sided_exact_sparse_affine_inverse"
            or not rows_canonical(certificate.inverse_rows)
        ):
            return "coordinate_affine_certificate_literal_drift"
        for row in certificate.inverse_rows:
            atoms.append(row.offset)
            atoms.extend(term.coefficient for term in row.terms)
    elif type(certificate) is TemporalAggregationCertificate:
        if (
            certificate.missing_policy is not MissingValuePolicy.REJECT
            or certificate.boundary_policy is not BoundaryPolicy.EXACT_PARTITION
            or certificate.support_contract
            != "same_clock_half_open_partition_modulo_final_endpoint"
            or not certificate.groups
            or not groups_canonical(certificate.groups)
        ):
            return "temporal_aggregation_certificate_literal_drift"
    elif type(certificate) is SpatialAggregationCertificate:
        if (
            certificate.missing_policy is not MissingValuePolicy.REJECT
            or certificate.boundary_policy is not BoundaryPolicy.EXACT_PARTITION
            or certificate.support_contract
            != "same_frame_interior_disjoint_positive_volume_partition"
            or not certificate.groups
            or not groups_canonical(certificate.groups)
        ):
            return "spatial_aggregation_certificate_literal_drift"
    elif type(certificate) is SamplingResolutionCertificate:
        atoms.extend(value for point in certificate.grid_points for value in point)
        exact_points = tuple(
            tuple(value.as_fraction() for value in point)
            for point in certificate.grid_points
        )
        if (
            certificate.missing_policy is not MissingValuePolicy.REJECT
            or certificate.boundary_policy is not BoundaryPolicy.VALID_ONLY
            or certificate.kernel_contract
            != "single_series_ordered_exact_subselection"
            or not certificate.selected_inputs
            or len(certificate.selected_inputs) != len(certificate.grid_points)
            or (
                certificate.axis is ComponentAxis.TEMPORAL
                and (
                    certificate.grid_dimension != 1
                    or certificate.grid_frame_id is not None
                )
            )
            or (
                certificate.axis is ComponentAxis.SPATIAL
                and certificate.grid_frame_id is None
            )
            or set(certificate.selected_inputs).intersection(
                certificate.discarded_inputs
            )
            or certificate.discarded_inputs
            != tuple(sorted(certificate.discarded_inputs))
            or len(certificate.discarded_inputs)
            != len(set(certificate.discarded_inputs))
            or any(
                left >= right
                for left, right in zip(
                    exact_points,
                    exact_points[1:],
                )
            )
        ):
            return "sampling_resolution_certificate_invalid"
    elif type(certificate) is EquivalentSplitMergeCertificate:
        if (
            certificate.missing_policy is not MissingValuePolicy.REJECT
            or certificate.equivalence_contract
            != "extensive_sum_one_sided_exact_inverse"
            or not certificate.groups
            or not certificate.inverse_rows
            or not groups_canonical(certificate.groups)
            or not rows_canonical(certificate.inverse_rows)
        ):
            return "split_merge_certificate_literal_drift"
        for row in certificate.inverse_rows:
            atoms.append(row.offset)
            atoms.extend(term.coefficient for term in row.terms)
    elif type(certificate) is CoarseGrainingCertificate:
        if (
            certificate.missing_policy is not MissingValuePolicy.REJECT
            or certificate.boundary_policy is not BoundaryPolicy.EXACT_PARTITION
            or certificate.commutation_contract
            != "transform_after_source_equals_target_after_transform"
            or not certificate.groups
            or len(certificate.groups) != len(certificate.quotient_class_ids)
            or not certificate.source_commutation_rows
            or not certificate.target_commutation_rows
            or not groups_canonical(certificate.groups)
            or not rows_canonical(certificate.source_commutation_rows)
            or not rows_canonical(certificate.target_commutation_rows)
            or len(certificate.quotient_class_ids)
            != len(set(certificate.quotient_class_ids))
            or tuple(
                (group.input_refs, group.output_refs, quotient_id)
                for group, quotient_id in zip(
                    certificate.groups,
                    certificate.quotient_class_ids,
                    strict=True,
                )
            )
            != tuple(
                sorted(
                    (
                        group.input_refs,
                        group.output_refs,
                        quotient_id,
                    )
                    for group, quotient_id in zip(
                        certificate.groups,
                        certificate.quotient_class_ids,
                        strict=True,
                    )
                )
            )
        ):
            return "coarse_graining_certificate_literal_drift"
        for row in (
            *certificate.source_commutation_rows,
            *certificate.target_commutation_rows,
        ):
            atoms.append(row.offset)
            atoms.extend(term.coefficient for term in row.terms)
    else:
        return "unknown_transform_certificate_type"
    if any(not _fraction_bits_valid(atom, policy) for atom in atoms):
        return "RESOURCE_LIMIT:certificate_fraction_bit_length"
    return None


def _validate_contract_index(
    authority: PublicTransformEvidenceBundleV2,
    policy: _ExactTransformPolicy,
) -> str | None:
    base = authority.base_bundle
    graph = base.aggregation_graph
    catalog = {item.transform_id: item for item in base.transform_catalog}
    contracts = {
        item.transform_id: item for item in authority.transform_contracts
    }
    if len(catalog) != len(base.transform_catalog):
        return "duplicate_legacy_transform_id"
    if len(contracts) != len(authority.transform_contracts):
        return "duplicate_typed_transform_id"
    if set(catalog) != set(contracts):
        return "transform_catalog_contract_id_mismatch"
    edge_ids = tuple(edge.transform_id for edge in graph.edges)
    if len(edge_ids) != len(set(edge_ids)):
        return "transform_id_reused_across_edges"
    if set(edge_ids) != set(catalog):
        return "transform_catalog_edge_id_mismatch"
    edges = {edge.transform_id: edge for edge in graph.edges}
    base_observation_ids = {item.observation_id for item in base.observations}
    derived_observation_ids = tuple(
        observation.observation_id
        for contract in authority.transform_contracts
        for observation in contract.output_observations
    )
    if (
        len(derived_observation_ids) != len(set(derived_observation_ids))
        or base_observation_ids.intersection(derived_observation_ids)
    ):
        return "derived_observation_id_not_globally_unique"
    root_component_ids = {
        component_id
        for item in authority.observation_metadata
        for component_id in item.component_ids
    }
    output_component_ids = tuple(
        descriptor.ref.component_id
        for contract in authority.transform_contracts
        for descriptor in contract.output_components
    )
    if (
        len(output_component_ids) != len(set(output_component_ids))
        or root_component_ids.intersection(output_component_ids)
    ):
        return "derived_component_id_not_globally_unique"
    for transform_id, contract in contracts.items():
        spec = catalog[transform_id]
        edge = edges[transform_id]
        if spec.operation is not contract.operation:
            return "legacy_and_typed_operation_mismatch"
        if spec.operation is not TransformOperation.IDENTITY and spec.parameters != ():
            return "legacy_nonidentity_parameters_forbidden"
        if (
            contract.source_scale_id != edge.source_scale_id
            or contract.target_scale_id != edge.target_scale_id
        ):
            return "typed_contract_edge_mismatch"
        expected_certificate = _CERTIFICATE_TYPE_BY_OPERATION.get(
            contract.operation
        )
        if expected_certificate is None or type(contract.certificate) is not (
            expected_certificate
        ):
            return "wrong_certificate_for_operation"
        literal_error = _validate_certificate_literals(
            contract.certificate,
            policy,
        )
        if literal_error is not None:
            return literal_error
        if len(contract.input_components) != len(set(contract.input_components)):
            return "typed_contract_repeats_input"
        output_refs = tuple(item.ref for item in contract.output_components)
        if len(output_refs) != len(set(output_refs)):
            return "typed_contract_repeats_output"
        if any(
            ref.scale_id != contract.source_scale_id
            for ref in contract.input_components
        ):
            return "typed_contract_input_scale_mismatch"
        if any(
            ref.scale_id != contract.target_scale_id for ref in output_refs
        ):
            return "typed_contract_output_scale_mismatch"
        row_outputs = tuple(row.output_ref for row in contract.kernel_rows)
        discrete_outputs = tuple(
            mapping.output_ref for mapping in contract.discrete_mappings
        )
        if len((*row_outputs, *discrete_outputs)) != len(
            set((*row_outputs, *discrete_outputs))
        ):
            return "typed_contract_output_written_more_than_once"
        if set((*row_outputs, *discrete_outputs)) != set(output_refs):
            return "typed_contract_output_coverage_mismatch"
        inputs = set(contract.input_components)
        if any(
            term.input_ref not in inputs
            for row in contract.kernel_rows
            for term in row.terms
        ):
            return "typed_kernel_references_undeclared_input"
        if any(
            mapping.input_ref not in inputs
            for mapping in contract.discrete_mappings
        ):
            return "typed_discrete_mapping_references_undeclared_input"
        atoms = tuple(
            atom
            for row in contract.kernel_rows
            for atom in (
                row.offset,
                *(term.coefficient for term in row.terms),
            )
        )
        if any(not _fraction_bits_valid(atom, policy) for atom in atoms):
            return "RESOURCE_LIMIT:kernel_fraction_bit_length"
        if any(
            term.coefficient == ZERO
            for row in contract.kernel_rows
            for term in row.terms
        ):
            return "zero_sparse_coefficient_forbidden"
    return None


def _forest_plan(
    authority: PublicTransformEvidenceBundleV2,
    policy: _ExactTransformPolicy,
) -> tuple[
    str | None,
    tuple[AggregationEdge, ...],
    dict[str, tuple[str, ...]],
]:
    """Validate the v1 forest restriction without enumerating DAG paths."""

    graph = authority.base_bundle.aggregation_graph
    scales = tuple(graph.scale_ids)
    roots = tuple(graph.root_scale_ids)
    if (
        len(scales) != len(set(scales))
        or len(roots) != len(set(roots))
        or not roots
        or not set(roots).issubset(scales)
    ):
        return "forest_scale_or_root_index_invalid", (), {}
    incoming: dict[str, list[AggregationEdge]] = {
        scale_id: [] for scale_id in scales
    }
    outgoing: dict[str, list[AggregationEdge]] = {
        scale_id: [] for scale_id in scales
    }
    for edge in graph.edges:
        if (
            type(edge) is not AggregationEdge
            or edge.source_scale_id not in outgoing
            or edge.target_scale_id not in incoming
            or edge.source_scale_id == edge.target_scale_id
        ):
            return "forest_edge_index_invalid", (), {}
        incoming[edge.target_scale_id].append(edge)
        outgoing[edge.source_scale_id].append(edge)
    root_set = set(roots)
    if any(incoming[root] for root in roots):
        return "forest_root_has_incoming_edge", (), {}
    for scale_id in scales:
        if scale_id not in root_set and len(incoming[scale_id]) != 1:
            return "forest_nonroot_requires_exactly_one_parent", (), {}
    for edges in outgoing.values():
        edges.sort(
            key=lambda item: (
                item.target_scale_id,
                item.transform_id,
            )
        )
    visited = set(roots)
    frontier = list(sorted(roots))
    paths = {root: () for root in roots}
    ordered_edges: list[AggregationEdge] = []
    while frontier:
        source_scale_id = frontier.pop(0)
        for edge in outgoing[source_scale_id]:
            if edge.target_scale_id in visited:
                return "forest_target_reached_more_than_once", (), {}
            path = (*paths[source_scale_id], edge.transform_id)
            if len(path) > policy.maximum_path_length:
                return "RESOURCE_LIMIT:transform_path_length", (), {}
            paths[edge.target_scale_id] = path
            visited.add(edge.target_scale_id)
            frontier.append(edge.target_scale_id)
            ordered_edges.append(edge)
    if visited != set(scales) or len(ordered_edges) != len(graph.edges):
        return "forest_has_unreachable_or_cyclic_scale", (), {}
    return None, tuple(ordered_edges), paths


def _metadata_equal_except_ref(
    left: ComponentDescriptor,
    right: ComponentDescriptor,
) -> bool:
    return (
        left.axis == right.axis
        and left.value_role == right.value_role
        and left.unit_id == right.unit_id
        and left.si_exponents == right.si_exponents
        and left.coordinate_frame_id == right.coordinate_frame_id
        and left.temporal_support == right.temporal_support
        and left.spatial_support == right.spatial_support
    )


def _missing_policy(certificate: TransformCertificate) -> MissingValuePolicy:
    return certificate.missing_policy


def _validate_common_contract(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
) -> str | None:
    if contract.input_components != tuple(sorted(source)):
        return "contract_input_is_not_complete_source_state"
    output_descriptors = {
        item.ref: item for item in contract.output_components
    }
    numeric_inputs = {
        ref
        for ref, state in source.items()
        if state.value_kind is ComponentValueKind.NUMERIC_INTERVAL
    }
    discrete_inputs = set(source) - numeric_inputs
    row_inputs = {
        term.input_ref for row in contract.kernel_rows for term in row.terms
    }
    if not row_inputs.issubset(numeric_inputs):
        return "numeric_kernel_references_discrete_input"
    mapping_inputs = tuple(
        item.input_ref for item in contract.discrete_mappings
    )
    if len(mapping_inputs) != len(set(mapping_inputs)):
        return "discrete_input_mapped_more_than_once"
    if set(mapping_inputs) != discrete_inputs:
        return "discrete_input_copy_coverage_mismatch"
    if _missing_policy(contract.certificate) is MissingValuePolicy.REJECT and any(
        source[ref].value_kind is ComponentValueKind.MISSING
        for ref in discrete_inputs
    ):
        return "missing_value_rejected_by_transform"
    for mapping in contract.discrete_mappings:
        input_state = source[mapping.input_ref]
        output = output_descriptors[mapping.output_ref]
        if not _metadata_equal_except_ref(input_state.descriptor, output):
            return "discrete_copy_metadata_mismatch"
        expected_role = (
            ComponentValueRole.BOOLEAN_CONTROL
            if input_state.value_kind is ComponentValueKind.BOOLEAN
            else ComponentValueRole.MISSING
        )
        if output.value_role is not expected_role:
            return "discrete_copy_value_role_mismatch"
    if any(
        term.coefficient.as_fraction() == 0
        for row in contract.kernel_rows
        for term in row.terms
    ):
        return "zero_sparse_coefficient_forbidden"
    return None


def _output_input_refs(
    contract: ExactTransformContract,
) -> dict[ComponentRef, tuple[ComponentRef, ...]]:
    result = {
        row.output_ref: tuple(term.input_ref for term in row.terms)
        for row in contract.kernel_rows
    }
    result.update(
        {
            mapping.output_ref: (mapping.input_ref,)
            for mapping in contract.discrete_mappings
        }
    )
    return result


def _descriptor_without_provenance(
    descriptor: DerivedObservationDescriptor,
) -> dict[str, object]:
    return {
        field.name: _canonical_value(getattr(descriptor, field.name))
        for field in fields(descriptor)
        if field.name != "provenance_sha256"
    }


def _expected_derived_provenance(
    descriptor: DerivedObservationDescriptor,
    source_states: tuple[_ComponentState, ...],
    source_observation_roots: dict[
        str,
        tuple[str, DerivedObservationDescriptor],
    ],
    contract_semantics_id: str,
    ordered_transform_path_ids: tuple[str, ...],
    ordered_contract_semantics_ids: tuple[str, ...],
) -> str:
    observation_ids = {
        state.observation.observation_id for state in source_states
    }
    observations = tuple(
        source_observation_roots[observation_id]
        for observation_id in sorted(observation_ids)
    )
    return stable_hash(
        {
            "input_observation_descriptors": tuple(
                (
                    descriptor_id,
                    observation.provenance_sha256,
                    observation.source_observation_ids,
                )
                for descriptor_id, observation in sorted(observations)
            ),
            "input_uncertainty_compilation_ids": tuple(
                sorted(
                    {
                        compilation_id
                        for state in source_states
                        for compilation_id in state.uncertainty_compilation_ids
                    }
                )
            ),
            "contract_semantics_without_provenance_id": contract_semantics_id,
            "ordered_transform_path_ids": ordered_transform_path_ids,
            "ordered_contract_semantics_ids": ordered_contract_semantics_ids,
            "output_observation": _descriptor_without_provenance(descriptor),
        },
        prefix="",
    )


def _validate_output_observations(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
    contract_semantics_id: str,
    ordered_transform_path_ids: tuple[str, ...],
    ordered_contract_semantics_ids: tuple[str, ...],
) -> str | None:
    source_observation_roots: dict[
        str,
        tuple[str, DerivedObservationDescriptor],
    ] = {}
    for state in source.values():
        observation = state.observation
        if observation.observation_id not in source_observation_roots:
            source_observation_roots[observation.observation_id] = (
                observation.descriptor_id,
                observation,
            )
    output_components = {item.ref: item for item in contract.output_components}
    observations = {
        item.observation_id: item for item in contract.output_observations
    }
    if len(observations) != len(contract.output_observations):
        return "duplicate_output_observation_id"
    observation_refs = tuple(
        ref
        for observation in contract.output_observations
        for ref in observation.component_refs
    )
    if len(observation_refs) != len(set(observation_refs)):
        return "output_observation_component_reused"
    if set(observation_refs) != set(output_components):
        return "output_observation_component_coverage_mismatch"
    input_refs_by_output = _output_input_refs(contract)
    for observation in contract.output_observations:
        descriptors = tuple(
            output_components[ref] for ref in observation.component_refs
        )
        if any(
            descriptor.unit_id != observation.unit_id
            or descriptor.si_exponents != observation.si_exponents
            or descriptor.temporal_support != observation.temporal_support
            or descriptor.spatial_support != observation.spatial_support
            for descriptor in descriptors
        ):
            return "output_observation_component_metadata_mismatch"
        contributing_refs = tuple(
            ref
            for output_ref in observation.component_refs
            for ref in input_refs_by_output[output_ref]
        )
        contributing_states = tuple(source[ref] for ref in contributing_refs)
        if not contributing_states:
            return "output_observation_without_input_lineage"
        semantic_identities = {
            (
                state.observation.source_channel_id,
                state.observation.entity_ids,
                state.observation.role_candidate_ids,
                state.observation.quantity_id,
            )
            for state in contributing_states
        }
        if len(semantic_identities) != 1:
            return "output_observation_input_semantic_identity_mismatch"
        source_channel, entity_ids, role_ids, quantity_id = next(
            iter(semantic_identities)
        )
        if (
            observation.source_channel_id != source_channel
            or observation.entity_ids != entity_ids
            or observation.role_candidate_ids != role_ids
            or observation.quantity_id != quantity_id
        ):
            return "output_observation_semantic_relabel_forbidden"
        expected_source_ids = tuple(
            sorted(
                {
                    source_id
                    for state in contributing_states
                    for source_id in state.observation.source_observation_ids
                }
            )
        )
        if observation.source_observation_ids != expected_source_ids:
            return "output_observation_source_lineage_mismatch"
        row_outputs = {row.output_ref for row in contract.kernel_rows}
        numeric_flags = tuple(
            ref in row_outputs for ref in observation.component_refs
        )
        if any(numeric_flags) and not all(numeric_flags):
            return "output_observation_mixes_numeric_and_discrete_components"
        expected_kind = (
            ComponentValueKind.NUMERIC_INTERVAL
            if all(numeric_flags)
            else source[contributing_refs[0]].value_kind
        )
        if observation.value_kind is not expected_kind:
            return "output_observation_value_kind_mismatch"
        expected_provenance = _expected_derived_provenance(
            observation,
            contributing_states,
            source_observation_roots,
            contract_semantics_id,
            ordered_transform_path_ids,
            ordered_contract_semantics_ids,
        )
        if observation.provenance_sha256 != expected_provenance:
            return "output_observation_provenance_mismatch"
    return None


def _validate_observation_shapes(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
) -> str | None:
    """Freeze which operations may regroup observable components."""

    inputs_by_output = _output_input_refs(contract)
    numeric_outputs = {row.output_ref for row in contract.kernel_rows}
    source_observations: dict[
        str,
        tuple[str, DerivedObservationDescriptor],
    ] = {}
    for state in source.values():
        observation = state.observation
        if observation.observation_id not in source_observations:
            source_observations[observation.observation_id] = (
                observation.descriptor_id,
                observation,
            )

    def source_descriptor_id(state: _ComponentState) -> str:
        return source_observations[state.observation.observation_id][0]

    numeric_observations = tuple(
        observation
        for observation in contract.output_observations
        if observation.value_kind is ComponentValueKind.NUMERIC_INTERVAL
    )
    discrete_observations = tuple(
        observation
        for observation in contract.output_observations
        if observation.value_kind is not ComponentValueKind.NUMERIC_INTERVAL
    )

    # Every discrete observation is scalar in the schema and must remain a
    # one-observation/one-component copy of its immediate source observation.
    source_discrete_descriptor_ids = {
        source_descriptor_id(state)
        for state in source.values()
        if state.value_kind is not ComponentValueKind.NUMERIC_INTERVAL
    }
    seen_discrete_descriptor_ids: set[str] = set()
    for observation in discrete_observations:
        if len(observation.component_refs) != 1:
            return "discrete_observation_shape_changed"
        input_ref = inputs_by_output[observation.component_refs[0]][0]
        source_observation = source[input_ref].observation
        if source_observation.value_kind is ComponentValueKind.NUMERIC_INTERVAL:
            return "discrete_observation_has_numeric_source"
        seen_discrete_descriptor_ids.add(
            source_observations[source_observation.observation_id][0]
        )
    if (
        seen_discrete_descriptor_ids != source_discrete_descriptor_ids
        or len(discrete_observations) != len(source_discrete_descriptor_ids)
    ):
        return "discrete_observation_copy_not_one_to_one"

    if contract.operation in (
        TransformOperation.IDENTITY,
        TransformOperation.UNIT_CONVERSION,
    ):
        source_numeric_observations = {
            descriptor_id: observation
            for descriptor_id, observation in source_observations.values()
            if observation.value_kind is ComponentValueKind.NUMERIC_INTERVAL
        }
        seen: set[str] = set()
        for observation in numeric_observations:
            input_refs = tuple(
                inputs_by_output[ref][0] for ref in observation.component_refs
            )
            source_ids = {
                source_descriptor_id(source[ref]) for ref in input_refs
            }
            if len(source_ids) != 1:
                return "one_to_one_transform_merged_observations"
            source_id = next(iter(source_ids))
            source_observation = source_numeric_observations[source_id]
            if (
                input_refs != source_observation.component_refs
                or tuple(ref.ordinal for ref in input_refs)
                != tuple(ref.ordinal for ref in observation.component_refs)
            ):
                return "one_to_one_transform_changed_observation_shape"
            seen.add(source_id)
        if seen != set(source_numeric_observations):
            return "one_to_one_transform_split_or_dropped_observation"
        return None

    if contract.operation is TransformOperation.COORDINATE_AFFINE:
        source_numeric = {
            descriptor_id: observation
            for descriptor_id, observation in source_observations.values()
            if observation.value_kind is ComponentValueKind.NUMERIC_INTERVAL
        }
        if len(source_numeric) != 1 or len(numeric_observations) != 1:
            return "coordinate_affine_requires_one_vector_observation"
        source_observation = next(iter(source_numeric.values()))
        target_observation = numeric_observations[0]
        if (
            source_observation.component_refs != _numeric_source_refs(source)
            or target_observation.component_refs != _numeric_output_refs(contract)
            or tuple(ref.ordinal for ref in source_observation.component_refs)
            != tuple(range(len(source_observation.component_refs)))
            or tuple(ref.ordinal for ref in target_observation.component_refs)
            != tuple(range(len(target_observation.component_refs)))
        ):
            return "coordinate_affine_vector_order_mismatch"
        return None

    certificate = contract.certificate
    if type(certificate) in (
        TemporalAggregationCertificate,
        SpatialAggregationCertificate,
        CoarseGrainingCertificate,
    ):
        expected = {group.output_refs for group in certificate.groups}
    elif type(certificate) is SamplingResolutionCertificate:
        expected = {(row.output_ref,) for row in contract.kernel_rows}
    elif type(certificate) is EquivalentSplitMergeCertificate:
        expected = {group.output_refs for group in certificate.groups}
    else:
        return "unknown_observation_shape_contract"
    actual = {observation.component_refs for observation in numeric_observations}
    if len(actual) != len(numeric_observations) or actual != expected:
        return "operation_specific_observation_shape_mismatch"
    if any(
        ref not in numeric_outputs
        for observation in numeric_observations
        for ref in observation.component_refs
    ):
        return "numeric_observation_shape_references_discrete_output"
    return None


def _normalized_expression(
    coefficients: dict[ComponentRef, Fraction],
    offset: Fraction,
) -> tuple[tuple[tuple[ComponentRef, Fraction], ...], Fraction]:
    return (
        tuple(sorted((ref, value) for ref, value in coefficients.items() if value)),
        offset,
    )


def _compose_affine_rows(
    budget: _KernelBudget,
    outer: tuple[ExactSparseAffineRow, ...],
    inner: tuple[ExactSparseAffineRow, ...],
) -> dict[
    ComponentRef,
    tuple[tuple[tuple[ComponentRef, Fraction], ...], Fraction],
] | None:
    inner_by_output = {row.output_ref: row for row in inner}
    if len(inner_by_output) != len(inner):
        return None
    result = {}
    for outer_row in outer:
        coefficients: dict[ComponentRef, Fraction] = {}
        offset = outer_row.offset.as_fraction()
        for outer_term in outer_row.terms:
            inner_row = inner_by_output.get(outer_term.input_ref)
            if inner_row is None:
                return None
            factor = outer_term.coefficient.as_fraction()
            offset = budget.add_fraction(
                offset,
                budget.multiply_fraction(
                    factor,
                    inner_row.offset.as_fraction(),
                ),
            )
            for inner_term in inner_row.terms:
                coefficients[inner_term.input_ref] = budget.add_fraction(
                    coefficients.get(inner_term.input_ref, Fraction(0)),
                    budget.multiply_fraction(
                        factor,
                        inner_term.coefficient.as_fraction(),
                    ),
                )
        result[outer_row.output_ref] = _normalized_expression(
            coefficients,
            offset,
        )
    return result


def _identity_expressions(
    refs: tuple[ComponentRef, ...],
) -> dict[
    ComponentRef,
    tuple[tuple[tuple[ComponentRef, Fraction], ...], Fraction],
]:
    return {
        ref: (((ref, Fraction(1)),), Fraction(0))
        for ref in refs
    }


def _rows_form_two_sided_inverse(
    budget: _KernelBudget,
    forward: tuple[ExactSparseAffineRow, ...],
    inverse: tuple[ExactSparseAffineRow, ...],
    source_refs: tuple[ComponentRef, ...],
    target_refs: tuple[ComponentRef, ...],
) -> bool:
    return (
        _compose_affine_rows(budget, inverse, forward)
        == _identity_expressions(source_refs)
        and _compose_affine_rows(budget, forward, inverse)
        == _identity_expressions(target_refs)
    )


def _group_refs(
    groups: tuple[ExactPartitionGroup, ...],
) -> tuple[set[ComponentRef], set[ComponentRef]] | None:
    inputs: set[ComponentRef] = set()
    outputs: set[ComponentRef] = set()
    for group in groups:
        if inputs.intersection(group.input_refs) or outputs.intersection(
            group.output_refs
        ):
            return None
        inputs.update(group.input_refs)
        outputs.update(group.output_refs)
    return inputs, outputs


def _row_by_output(
    contract: ExactTransformContract,
) -> dict[ComponentRef, ExactSparseAffineRow]:
    return {row.output_ref: row for row in contract.kernel_rows}


def _numeric_source_refs(
    source: dict[ComponentRef, _ComponentState],
) -> tuple[ComponentRef, ...]:
    return tuple(
        sorted(
            ref
            for ref, state in source.items()
            if state.value_kind is ComponentValueKind.NUMERIC_INTERVAL
        )
    )


def _numeric_output_refs(
    contract: ExactTransformContract,
) -> tuple[ComponentRef, ...]:
    discrete = {item.output_ref for item in contract.discrete_mappings}
    return tuple(
        item.ref
        for item in contract.output_components
        if item.ref not in discrete
    )


def _validate_one_to_one_rows(
    rows: tuple[ExactSparseAffineRow, ...],
    source_refs: tuple[ComponentRef, ...],
    target_refs: tuple[ComponentRef, ...],
    factor: Fraction,
) -> bool:
    if len(rows) != len(source_refs) or len(rows) != len(target_refs):
        return False
    used = []
    for row in rows:
        if (
            len(row.terms) != 1
            or row.offset != ZERO
            or row.terms[0].coefficient.as_fraction() != factor
        ):
            return False
        used.append(row.terms[0].input_ref)
    return set(used) == set(source_refs) and {
        row.output_ref for row in rows
    } == set(target_refs)


def _validate_identity(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
) -> str | None:
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    if not _validate_one_to_one_rows(
        contract.kernel_rows,
        source_refs,
        target_refs,
        Fraction(1),
    ):
        return "identity_kernel_is_not_exact_one_to_one"
    outputs = {item.ref: item for item in contract.output_components}
    for row in contract.kernel_rows:
        if not _metadata_equal_except_ref(
            source[row.terms[0].input_ref].descriptor,
            outputs[row.output_ref],
        ):
            return "identity_numeric_metadata_mismatch"
    return None


def _validate_unit_conversion(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
) -> str | None:
    certificate = contract.certificate
    assert type(certificate) is UnitConversionCertificate
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    factor = certificate.factor.as_fraction()
    if not source_refs or certificate.source_unit_id == certificate.target_unit_id:
        return "unit_conversion_requires_numeric_distinct_declared_units"
    if not _validate_one_to_one_rows(
        contract.kernel_rows,
        source_refs,
        target_refs,
        factor,
    ):
        return "unit_conversion_kernel_mismatch"
    outputs = {item.ref: item for item in contract.output_components}
    for row in contract.kernel_rows:
        left = source[row.terms[0].input_ref].descriptor
        right = outputs[row.output_ref]
        if (
            left.unit_id != certificate.source_unit_id
            or right.unit_id != certificate.target_unit_id
            or left.si_exponents != right.si_exponents
            or left.axis != right.axis
            or left.value_role != right.value_role
            or left.coordinate_frame_id != right.coordinate_frame_id
            or left.temporal_support != right.temporal_support
            or left.spatial_support != right.spatial_support
        ):
            return "unit_conversion_metadata_contract_mismatch"
    return None


def _validate_coordinate_affine(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
    budget: _KernelBudget,
) -> str | None:
    certificate = contract.certificate
    assert type(certificate) is CoordinateAffineCertificate
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    if (
        len(source_refs) != certificate.dimension
        or len(target_refs) != certificate.dimension
        or len(contract.kernel_rows) != certificate.dimension
        or len(certificate.inverse_rows) != certificate.dimension
    ):
        return "coordinate_affine_dimension_mismatch"
    if not _rows_form_two_sided_inverse(
        budget,
        contract.kernel_rows,
        certificate.inverse_rows,
        source_refs,
        target_refs,
    ):
        return "coordinate_affine_is_singular_or_inverse_mismatch"
    outputs = {item.ref: item for item in contract.output_components}
    source_descriptors = tuple(source[ref].descriptor for ref in source_refs)
    target_descriptors = tuple(outputs[ref] for ref in target_refs)
    if any(
        item.value_role is not ComponentValueRole.COORDINATE
        or item.axis is not ComponentAxis.COORDINATE
        or item.coordinate_frame_id != certificate.source_frame_id
        for item in source_descriptors
    ):
        return "coordinate_affine_source_metadata_mismatch"
    if any(
        item.value_role is not ComponentValueRole.COORDINATE
        or item.axis is not ComponentAxis.COORDINATE
        or item.coordinate_frame_id != certificate.target_frame_id
        for item in target_descriptors
    ):
        return "coordinate_affine_target_metadata_mismatch"
    source_units = {
        (item.unit_id, item.si_exponents) for item in source_descriptors
    }
    target_units = {
        (item.unit_id, item.si_exponents) for item in target_descriptors
    }
    if len(source_units) != 1 or source_units != target_units:
        return "coordinate_affine_unit_dimension_mismatch"
    temporal = {item.temporal_support for item in source_descriptors}
    if len(temporal) != 1 or any(
        item.temporal_support != source_descriptors[0].temporal_support
        for item in target_descriptors
    ):
        return "coordinate_affine_temporal_support_mismatch"
    source_spatial = {item.spatial_support for item in source_descriptors}
    target_spatial = {item.spatial_support for item in target_descriptors}
    if len(source_spatial) != 1 or len(target_spatial) != 1:
        return "coordinate_affine_spatial_support_not_common"
    source_support = source_descriptors[0].spatial_support
    target_support = target_descriptors[0].spatial_support
    if (
        source_support is None
        or target_support is None
        or source_support.frame_id != certificate.source_frame_id
        or target_support.frame_id != certificate.target_frame_id
        or len(source_support.lower) != certificate.dimension
        or len(target_support.lower) != certificate.dimension
    ):
        return "coordinate_affine_support_frame_or_dimension_mismatch"
    support_states = {
        ref: _ComponentState(
            descriptor=source[ref].descriptor,
            observation=source[ref].observation,
            value_kind=ComponentValueKind.NUMERIC_INTERVAL,
            numeric_interval=ExactTransformInterval(
                source_support.lower[index],
                source_support.upper[index],
            ),
            boolean_value=None,
            uncertainty_compilation_ids=source[ref].uncertainty_compilation_ids,
        )
        for index, ref in enumerate(source_refs)
    }
    expected_bounds = tuple(
        _evaluate_sparse_row(budget, row, support_states)[0]
        for row in contract.kernel_rows
    )
    if tuple(bound.lower for bound in expected_bounds) != target_support.lower or tuple(
        bound.upper for bound in expected_bounds
    ) != target_support.upper:
        return "coordinate_affine_support_enclosure_mismatch"
    return None


def _validate_reducer_row(
    row: ExactSparseAffineRow,
    group: ExactPartitionGroup,
    reducer: ReducerKind,
    budget: _KernelBudget,
) -> bool:
    if (
        group.output_refs != (row.output_ref,)
        or tuple(term.input_ref for term in row.terms) != group.input_refs
        or row.offset != ZERO
    ):
        return False
    coefficients = tuple(term.coefficient.as_fraction() for term in row.terms)
    if reducer is ReducerKind.SUM:
        return all(value == 1 for value in coefficients)
    if any(value <= 0 for value in coefficients):
        return False
    total = Fraction(0)
    for value in coefficients:
        total = budget.add_fraction(total, value)
    return total == 1


def _exact_temporal_partition(
    supports: tuple[ExactTemporalSupport, ...],
    target: ExactTemporalSupport,
    budget: _KernelBudget,
) -> bool:
    if not supports or any(item.clock_id != target.clock_id for item in supports):
        return False
    intervals = tuple(
        (item.start.as_fraction(), item.end.as_fraction()) for item in supports
    )
    target_start = target.start.as_fraction()
    target_end = target.end.as_fraction()
    if any(left >= right for left, right in intervals):
        return False
    if min(left for left, _ in intervals) != target_start or max(
        right for _, right in intervals
    ) != target_end:
        return False
    ordered = tuple(sorted(intervals))
    if any(
        left_right != right_left
        for (_, left_right), (right_left, _) in zip(
            ordered,
            ordered[1:],
        )
    ):
        return False
    total = Fraction(0)
    for left, right in intervals:
        width = budget.add_fraction(right, -left)
        total = budget.add_fraction(total, width)
    target_width = budget.add_fraction(target_end, -target_start)
    return total == target_width


def _box_volume(
    support: ExactSpatialSupport,
    budget: _KernelBudget,
) -> Fraction | None:
    result = Fraction(1)
    for lower, upper in zip(support.lower, support.upper, strict=True):
        width = budget.add_fraction(
            upper.as_fraction(),
            -lower.as_fraction(),
        )
        if width <= 0:
            return None
        result = budget.multiply_fraction(result, width)
    return result


def _spatial_interiors_overlap(
    left: ExactSpatialSupport,
    right: ExactSpatialSupport,
) -> bool:
    return all(
        max(left_lower.as_fraction(), right_lower.as_fraction())
        < min(left_upper.as_fraction(), right_upper.as_fraction())
        for left_lower, left_upper, right_lower, right_upper in zip(
            left.lower,
            left.upper,
            right.lower,
            right.upper,
            strict=True,
        )
    )


def _exact_spatial_partition(
    supports: tuple[ExactSpatialSupport, ...],
    target: ExactSpatialSupport,
    budget: _KernelBudget,
) -> bool:
    if not supports or any(
        item.frame_id != target.frame_id
        or len(item.lower) != len(target.lower)
        for item in supports
    ):
        return False
    for support in supports:
        if any(
            source_lower.as_fraction() < target_lower.as_fraction()
            or source_upper.as_fraction() > target_upper.as_fraction()
            for source_lower, source_upper, target_lower, target_upper in zip(
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
            if budget.operations > budget.policy.maximum_exact_operations:
                raise _ResourceLimit("RESOURCE_LIMIT:exact_operations")
            if _spatial_interiors_overlap(left, right):
                return False
    target_volume = _box_volume(target, budget)
    if target_volume is None:
        return False
    volume_sum = Fraction(0)
    for support in supports:
        volume = _box_volume(support, budget)
        if volume is None:
            return False
        volume_sum = budget.add_fraction(volume_sum, volume)
    return volume_sum == target_volume


def _validate_aggregation(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
    budget: _KernelBudget,
    *,
    temporal: bool,
) -> str | None:
    certificate = contract.certificate
    expected_type = (
        TemporalAggregationCertificate
        if temporal
        else SpatialAggregationCertificate
    )
    assert type(certificate) is expected_type
    grouped = _group_refs(certificate.groups)
    if grouped is None:
        return "aggregation_groups_overlap"
    source_refs = set(_numeric_source_refs(source))
    target_refs = set(_numeric_output_refs(contract))
    if grouped != (source_refs, target_refs):
        return "aggregation_groups_not_exhaustive"
    rows = _row_by_output(contract)
    outputs = {item.ref: item for item in contract.output_components}
    for group in certificate.groups:
        row = rows.get(group.output_refs[0]) if len(group.output_refs) == 1 else None
        if row is None or not _validate_reducer_row(
            row,
            group,
            certificate.reducer,
            budget,
        ):
            return "aggregation_reducer_kernel_mismatch"
        inputs = tuple(source[ref].descriptor for ref in group.input_refs)
        output = outputs[group.output_refs[0]]
        expected_role = (
            ComponentValueRole.EXTENSIVE
            if certificate.reducer is ReducerKind.SUM
            else ComponentValueRole.INTENSIVE
        )
        expected_axis = ComponentAxis.TEMPORAL if temporal else ComponentAxis.SPATIAL
        if any(
            item.axis is not expected_axis
            or item.value_role is not expected_role
            or item.unit_id != output.unit_id
            or item.si_exponents != output.si_exponents
            for item in inputs
        ) or output.axis is not expected_axis or output.value_role is not expected_role:
            return "aggregation_component_metadata_mismatch"
        if temporal:
            supports = tuple(item.temporal_support for item in inputs)
            if any(item is None for item in supports) or output.temporal_support is None:
                return "temporal_aggregation_support_missing"
            if not _exact_temporal_partition(
                supports,  # type: ignore[arg-type]
                output.temporal_support,
                budget,
            ):
                return "temporal_aggregation_not_exact_partition"
            if any(
                item.spatial_support != output.spatial_support for item in inputs
            ):
                return "temporal_aggregation_spatial_support_mismatch"
        else:
            supports = tuple(item.spatial_support for item in inputs)
            if any(item is None for item in supports) or output.spatial_support is None:
                return "spatial_aggregation_support_missing"
            if not _exact_spatial_partition(
                supports,  # type: ignore[arg-type]
                output.spatial_support,
                budget,
            ):
                return "spatial_aggregation_not_exact_partition"
            if any(
                item.temporal_support != output.temporal_support for item in inputs
            ):
                return "spatial_aggregation_temporal_support_mismatch"
    return None


def _validate_sampling(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
) -> str | None:
    certificate = contract.certificate
    assert type(certificate) is SamplingResolutionCertificate
    source_refs = set(_numeric_source_refs(source))
    if (
        len(certificate.selected_inputs)
        != len(set(certificate.selected_inputs))
        or source_refs
        != set(certificate.selected_inputs).union(certificate.discarded_inputs)
    ):
        return "sampling_selected_discarded_not_exact_coverage"
    semantic_series = {
        (
            source[ref].observation.source_channel_id,
            source[ref].observation.entity_ids,
            source[ref].observation.role_candidate_ids,
            source[ref].observation.quantity_id,
            source[ref].descriptor.value_role,
            source[ref].descriptor.unit_id,
            source[ref].descriptor.si_exponents,
            source[ref].descriptor.coordinate_frame_id,
            (
                source[ref].descriptor.spatial_support
                if certificate.axis is ComponentAxis.TEMPORAL
                else source[ref].descriptor.temporal_support
            ),
        )
        for ref in source_refs
    }
    if len(semantic_series) != 1:
        return "sampling_v1_requires_one_semantic_series"
    source_points: dict[ComponentRef, tuple[ExactTransformAtom, ...]] = {}
    temporal_clocks: set[str] = set()
    for ref in source_refs:
        descriptor = source[ref].descriptor
        if descriptor.axis is not certificate.axis:
            return "sampling_source_axis_mismatch"
        if certificate.axis is ComponentAxis.TEMPORAL:
            support = descriptor.temporal_support
            if support is None or support.start != support.end:
                return "sampling_temporal_source_is_not_grid_point"
            temporal_clocks.add(support.clock_id)
            source_points[ref] = (support.start,)
        else:
            support = descriptor.spatial_support
            if (
                support is None
                or support.frame_id != certificate.grid_frame_id
                or len(support.lower) != certificate.grid_dimension
                or support.lower != support.upper
            ):
                return "sampling_spatial_source_is_not_grid_point"
            source_points[ref] = support.lower
    if len(temporal_clocks) > 1:
        return "sampling_temporal_clock_mismatch"
    if len(set(source_points.values())) != len(source_points):
        return "sampling_v1_repeats_source_grid_point"
    rows = contract.kernel_rows
    target_refs = _numeric_output_refs(contract)
    if len(rows) != len(certificate.selected_inputs) or len(rows) != len(
        target_refs
    ):
        return "sampling_output_count_mismatch"
    outputs = {item.ref: item for item in contract.output_components}
    for row, input_ref, grid_point in zip(
        rows,
        certificate.selected_inputs,
        certificate.grid_points,
        strict=True,
    ):
        if (
            len(row.terms) != 1
            or row.terms[0].input_ref != input_ref
            or row.terms[0].coefficient != ONE
            or row.offset != ZERO
        ):
            return "sampling_kernel_is_not_exact_subselection"
        left = source[input_ref].descriptor
        right = outputs[row.output_ref]
        if left.axis is not certificate.axis or not _metadata_equal_except_ref(
            left,
            right,
        ):
            return "sampling_component_metadata_mismatch"
        if certificate.axis is ComponentAxis.TEMPORAL:
            support = left.temporal_support
            if (
                support is None
                or support.start != support.end
                or support.start != grid_point[0]
            ):
                return "sampling_temporal_grid_not_source_point"
        else:
            support = left.spatial_support
            if (
                support is None
                or support.frame_id != certificate.grid_frame_id
                or len(support.lower) != certificate.grid_dimension
                or support.lower != support.upper
                or support.lower != grid_point
            ):
                return "sampling_spatial_grid_not_source_point"
        if source_points[input_ref] != grid_point:
            return "sampling_certificate_grid_point_mismatch"
    return None


def _same_numeric_semantics(
    source: ComponentDescriptor,
    target: ComponentDescriptor,
) -> bool:
    return (
        source.axis == target.axis
        and source.unit_id == target.unit_id
        and source.si_exponents == target.si_exponents
        and source.coordinate_frame_id == target.coordinate_frame_id
        and source.temporal_support == target.temporal_support
        and source.spatial_support == target.spatial_support
    )


def _validate_split_merge(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
    budget: _KernelBudget,
) -> str | None:
    certificate = contract.certificate
    assert type(certificate) is EquivalentSplitMergeCertificate
    grouped = _group_refs(certificate.groups)
    if grouped is None or grouped != (
        set(_numeric_source_refs(source)),
        set(_numeric_output_refs(contract)),
    ):
        return "split_merge_groups_not_disjoint_exhaustive"
    rows = _row_by_output(contract)
    outputs = {item.ref: item for item in contract.output_components}
    for group in certificate.groups:
        if certificate.direction is SplitMergeDirection.SPLIT:
            if len(group.input_refs) != 1 or len(group.output_refs) < 2:
                return "split_group_shape_invalid"
            input_ref = group.input_refs[0]
            group_rows = tuple(rows.get(ref) for ref in group.output_refs)
            if any(row is None for row in group_rows):
                return "split_group_missing_kernel_row"
            coefficients = []
            for row in group_rows:
                assert row is not None
                if (
                    len(row.terms) != 1
                    or row.terms[0].input_ref != input_ref
                    or row.offset != ZERO
                    or row.terms[0].coefficient.as_fraction() <= 0
                ):
                    return "split_kernel_invalid"
                coefficients.append(row.terms[0].coefficient.as_fraction())
                if (
                    source[input_ref].descriptor.value_role
                    is not ComponentValueRole.EXTENSIVE
                    or outputs[row.output_ref].value_role
                    is not ComponentValueRole.EXTENSIVE
                    or not _same_numeric_semantics(
                        source[input_ref].descriptor,
                        outputs[row.output_ref],
                    )
                ):
                    return "split_metadata_not_extensive_equivalent"
            coefficient_sum = Fraction(0)
            for coefficient in coefficients:
                coefficient_sum = budget.add_fraction(
                    coefficient_sum,
                    coefficient,
                )
            if coefficient_sum != 1:
                return "split_coefficients_do_not_preserve_sum"
        else:
            if len(group.input_refs) < 2 or len(group.output_refs) != 1:
                return "merge_group_shape_invalid"
            row = rows.get(group.output_refs[0])
            if (
                row is None
                or tuple(term.input_ref for term in row.terms)
                != group.input_refs
                or row.offset != ZERO
                or any(term.coefficient != ONE for term in row.terms)
            ):
                return "merge_kernel_invalid"
            target = outputs[group.output_refs[0]]
            if target.value_role is not ComponentValueRole.EXTENSIVE or any(
                source[ref].descriptor.value_role
                is not ComponentValueRole.EXTENSIVE
                or not _same_numeric_semantics(source[ref].descriptor, target)
                for ref in group.input_refs
            ):
                return "merge_metadata_not_extensive_equivalent"
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    if certificate.direction is SplitMergeDirection.SPLIT:
        inverse = {row.output_ref: row for row in certificate.inverse_rows}
        if len(inverse) != len(certificate.inverse_rows) or set(inverse) != set(
            source_refs
        ):
            return "split_inverse_row_coverage_mismatch"
        for group in certificate.groups:
            row = inverse.get(group.input_refs[0])
            if (
                row is None
                or row.offset != ZERO
                or tuple(term.input_ref for term in row.terms)
                != group.output_refs
                or any(term.coefficient != ONE for term in row.terms)
            ):
                return "split_inverse_is_not_exact_sum"
        composed = _compose_affine_rows(
            budget,
            certificate.inverse_rows,
            contract.kernel_rows,
        )
        expected = _identity_expressions(source_refs)
    else:
        inverse = {row.output_ref: row for row in certificate.inverse_rows}
        if len(inverse) != len(certificate.inverse_rows) or set(inverse) != set(
            source_refs
        ):
            return "merge_inverse_row_coverage_mismatch"
        for group in certificate.groups:
            target_ref = group.output_refs[0]
            coefficient_sum = Fraction(0)
            for input_ref in group.input_refs:
                row = inverse.get(input_ref)
                if (
                    row is None
                    or row.offset != ZERO
                    or len(row.terms) != 1
                    or row.terms[0].input_ref != target_ref
                    or row.terms[0].coefficient.as_fraction() <= 0
                ):
                    return "merge_inverse_is_not_positive_partition"
                coefficient_sum = budget.add_fraction(
                    coefficient_sum,
                    row.terms[0].coefficient.as_fraction(),
                )
            if coefficient_sum != 1:
                return "merge_inverse_weights_do_not_sum_to_one"
        composed = _compose_affine_rows(
            budget,
            contract.kernel_rows,
            certificate.inverse_rows,
        )
        expected = _identity_expressions(target_refs)
    if composed != expected:
        return "split_merge_one_sided_inverse_mismatch"
    return None


def _validate_coarse_graining(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
    budget: _KernelBudget,
) -> str | None:
    certificate = contract.certificate
    assert type(certificate) is CoarseGrainingCertificate
    grouped = _group_refs(certificate.groups)
    if grouped is None or grouped != (
        set(_numeric_source_refs(source)),
        set(_numeric_output_refs(contract)),
    ):
        return "coarse_groups_not_disjoint_exhaustive"
    rows = _row_by_output(contract)
    outputs = {item.ref: item for item in contract.output_components}
    for group in certificate.groups:
        if len(group.output_refs) != 1:
            return "coarse_group_needs_one_quotient_output"
        row = rows.get(group.output_refs[0])
        if row is None or not _validate_reducer_row(
            row,
            group,
            certificate.reducer,
            budget,
        ):
            return "coarse_reducer_kernel_mismatch"
        target = outputs[group.output_refs[0]]
        expected_role = (
            ComponentValueRole.EXTENSIVE
            if certificate.reducer is ReducerKind.SUM
            else ComponentValueRole.INTENSIVE
        )
        if target.axis is not ComponentAxis.COARSE or target.value_role is not (
            expected_role
        ):
            return "coarse_output_axis_or_role_mismatch"
        inputs = tuple(source[ref].descriptor for ref in group.input_refs)
        if any(
            item.value_role is not expected_role
            or item.unit_id != target.unit_id
            or item.si_exponents != target.si_exponents
            or item.temporal_support != target.temporal_support
            or item.spatial_support != target.spatial_support
            for item in inputs
        ):
            return "coarse_input_output_metadata_mismatch"
    source_refs = _numeric_source_refs(source)
    target_refs = _numeric_output_refs(contract)
    source_rows = certificate.source_commutation_rows
    target_rows = certificate.target_commutation_rows
    if (
        {row.output_ref for row in source_rows} != set(source_refs)
        or {
            term.input_ref for row in source_rows for term in row.terms
        }
        - set(source_refs)
        or {row.output_ref for row in target_rows} != set(target_refs)
        or {
            term.input_ref for row in target_rows for term in row.terms
        }
        - set(target_refs)
    ):
        return "coarse_commutation_ref_closure_mismatch"
    left = _compose_affine_rows(
        budget,
        contract.kernel_rows,
        source_rows,
    )
    right = _compose_affine_rows(
        budget,
        target_rows,
        contract.kernel_rows,
    )
    if left is None or left != right:
        return "coarse_commutation_matrix_equality_failed"
    return None


def _apply_contract(
    contract: ExactTransformContract,
    source: dict[ComponentRef, _ComponentState],
    budget: _KernelBudget,
    *,
    contract_semantics_id: str,
    ordered_transform_path_ids: tuple[str, ...],
    ordered_contract_semantics_ids: tuple[str, ...],
) -> tuple[
    dict[ComponentRef, _ComponentState] | None,
    tuple[DerivedObservationDescriptor, ...],
    str | None,
]:
    common_error = _validate_common_contract(contract, source)
    if common_error is not None:
        return None, (), common_error
    observation_error = _validate_output_observations(
        contract,
        source,
        contract_semantics_id,
        ordered_transform_path_ids,
        ordered_contract_semantics_ids,
    )
    if observation_error is not None:
        return None, (), observation_error
    shape_error = _validate_observation_shapes(contract, source)
    if shape_error is not None:
        return None, (), shape_error

    operation = contract.operation
    if operation is TransformOperation.IDENTITY:
        operation_error = _validate_identity(contract, source)
    elif operation is TransformOperation.UNIT_CONVERSION:
        operation_error = _validate_unit_conversion(contract, source)
    elif operation is TransformOperation.COORDINATE_AFFINE:
        operation_error = _validate_coordinate_affine(contract, source, budget)
    elif operation is TransformOperation.TEMPORAL_AGGREGATION:
        operation_error = _validate_aggregation(
            contract,
            source,
            budget,
            temporal=True,
        )
    elif operation is TransformOperation.SPATIAL_AGGREGATION:
        operation_error = _validate_aggregation(
            contract,
            source,
            budget,
            temporal=False,
        )
    elif operation is TransformOperation.SAMPLING_RESOLUTION:
        operation_error = _validate_sampling(contract, source)
    elif operation is TransformOperation.EQUIVALENT_SPLIT_MERGE:
        operation_error = _validate_split_merge(contract, source, budget)
    elif operation is TransformOperation.COARSE_GRAINING:
        operation_error = _validate_coarse_graining(contract, source, budget)
    else:
        return None, (), "unknown_transform_operation"
    if operation_error is not None:
        return None, (), operation_error

    descriptors = {item.ref: item for item in contract.output_components}
    observations_by_ref = {
        ref: observation
        for observation in contract.output_observations
        for ref in observation.component_refs
    }
    target: dict[ComponentRef, _ComponentState] = {}
    try:
        for row in contract.kernel_rows:
            numeric_interval, lineage = _evaluate_sparse_row(
                budget,
                row,
                source,
            )
            target[row.output_ref] = _ComponentState(
                descriptor=descriptors[row.output_ref],
                observation=observations_by_ref[row.output_ref],
                value_kind=ComponentValueKind.NUMERIC_INTERVAL,
                numeric_interval=numeric_interval,
                boolean_value=None,
                uncertainty_compilation_ids=lineage,
            )
        for mapping in contract.discrete_mappings:
            input_state = source[mapping.input_ref]
            target[mapping.output_ref] = _ComponentState(
                descriptor=descriptors[mapping.output_ref],
                observation=observations_by_ref[mapping.output_ref],
                value_kind=input_state.value_kind,
                numeric_interval=None,
                boolean_value=input_state.boolean_value,
                uncertainty_compilation_ids=(
                    input_state.uncertainty_compilation_ids
                ),
            )
    except (KeyError, TypeError, ValueError):
        return None, (), "exact_kernel_evaluation_failed"
    if set(target) != set(descriptors):
        return None, (), "target_state_not_exactly_constructed"
    return target, contract.output_observations, None


def _contract_commitment(
    contracts: tuple[ExactTransformContract, ...],
    semantics_ids: dict[str, str],
) -> str:
    return stable_hash(
        tuple(
            (
                contract.transform_id,
                semantics_ids[contract.transform_id],
                contract.contract_id,
            )
            for contract in contracts
        ),
        prefix="phase2b_exact_transform_contract_set_",
    )


def _graph_commitment(graph: AggregationGraph) -> str:
    return stable_hash(
        {
            "graph": graph,
            "execution_contract": "forest_unique_root_path_only",
        },
        prefix="phase2b_exact_transform_graph_",
    )


def _abstaining_compilation(
    *,
    reason: str,
    failure: ExactTransformFailure,
    wrapper_content_id: str,
    base_bundle_content_id: str,
    uncertainty_receipt: BundleUncertaintyCompilation,
    contract_commitment_id: str,
    graph_commitment_id: str,
) -> ExactTransformCompilation:
    return ExactTransformCompilation(
        disposition=TransformCompilationDisposition.ABSTAIN,
        reason=reason,
        wrapper_content_id=wrapper_content_id,
        base_bundle_content_id=base_bundle_content_id,
        uncertainty_result_id=uncertainty_receipt.result_id,
        uncertainty_policy_id=DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id,
        phase2b_exact_freeze_id=FROZEN_PHASE2B_EXACT_FREEZE_ID,
        rational_grid_id=FROZEN_RATIONAL_GRID_ID,
        transform_policy_id=_DEFAULT_POLICY.policy_id,
        transform_semantics_id=_DEFAULT_POLICY.semantics_id,
        contract_commitment_id=contract_commitment_id,
        graph_commitment_id=graph_commitment_id,
        uncertainty_receipt=uncertainty_receipt,
        observations=(),
        components=(),
        failures=(failure,),
    )


def run_exact_transform_semantics(
    authority: PublicTransformEvidenceBundleV2,
) -> ExactTransformCompilation | ExactTransformPreflightRejection:
    """Compile the content-rooted v2 transform authority atomically.

    There are intentionally no caller-selectable policy, registry, receipt, or
    transform-semantics arguments.  Preflight failures are uncommitted; after
    authority hashing, every failure returns a committed abstention with no
    observations or components.
    """

    if type(authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError(
            "exact transform semantics requires PublicTransformEvidenceBundleV2"
        )
    policy = _DEFAULT_POLICY
    try:
        shallow_error = _shallow_preflight(authority, policy)
    except (AttributeError, TypeError, ValueError):
        raise TypeError("transform authority root or shallow schema is invalid")
    if shallow_error is not None:
        return _preflight_rejection(shallow_error, authority, policy)
    try:
        _AuthorityTreeBudget(policy).visit(authority)
    except _ResourceLimit as exc:
        return _preflight_rejection(str(exc), authority, policy)
    except (AttributeError, TypeError, ValueError):
        return _preflight_rejection(
            "authority_exact_tree_validation_failed",
            authority,
            policy,
        )
    detailed_error = _detailed_resource_preflight(authority, policy)
    if detailed_error is not None:
        return _preflight_rejection(detailed_error, authority, policy)
    metadata_error = _validate_metadata(authority)
    if metadata_error is not None:
        return _preflight_rejection(metadata_error, authority, policy)
    contract_error = _validate_contract_index(authority, policy)
    if contract_error is not None:
        return _preflight_rejection(contract_error, authority, policy)
    forest_error, ordered_edges, transform_paths = _forest_plan(
        authority,
        policy,
    )
    if forest_error is not None:
        return _preflight_rejection(forest_error, authority, policy)

    wrapper_content_id = authority.content_id
    base_bundle_content_id = authority.base_bundle.content_id
    graph_commitment_id = _graph_commitment(
        authority.base_bundle.aggregation_graph
    )
    uncertainty_receipt = compile_bundle_uncertainty(authority.base_bundle)
    # Contract semantics excludes derived provenance; the full commitment is
    # computed only after the attempted provenance validations below.
    semantics_ids = {
        contract.transform_id: contract.semantics_id
        for contract in authority.transform_contracts
    }
    contracts_by_id = {
        contract.transform_id: contract
        for contract in authority.transform_contracts
    }
    semantic_paths: dict[str, tuple[str, ...]] = {
        root: ()
        for root in authority.base_bundle.aggregation_graph.root_scale_ids
    }

    if uncertainty_receipt.disposition is not BundleUncertaintyDisposition.COMPLETE:
        commitment = _contract_commitment(
            authority.transform_contracts,
            semantics_ids,
        )
        return _abstaining_compilation(
            reason="bundle_uncertainty_not_complete",
            failure=ExactTransformFailure(uncertainty_receipt.reason),
            wrapper_content_id=wrapper_content_id,
            base_bundle_content_id=base_bundle_content_id,
            uncertainty_receipt=uncertainty_receipt,
            contract_commitment_id=commitment,
            graph_commitment_id=graph_commitment_id,
        )

    try:
        states_by_scale, observations_by_scale = _build_root_states(
            authority,
            uncertainty_receipt,
        )
    except (IndexError, KeyError, TypeError, ValueError):
        commitment = _contract_commitment(
            authority.transform_contracts,
            semantics_ids,
        )
        return _abstaining_compilation(
            reason="root_state_construction_failed",
            failure=ExactTransformFailure("root_state_construction_failed"),
            wrapper_content_id=wrapper_content_id,
            base_bundle_content_id=base_bundle_content_id,
            uncertainty_receipt=uncertainty_receipt,
            contract_commitment_id=commitment,
            graph_commitment_id=graph_commitment_id,
        )
    budget = _KernelBudget(policy)
    failed: tuple[str, str, str] | None = None
    try:
        for edge in ordered_edges:
            contract = contracts_by_id[edge.transform_id]
            source = states_by_scale[edge.source_scale_id]
            semantic_path = (
                *semantic_paths[edge.source_scale_id],
                semantics_ids[edge.transform_id],
            )
            target, observations, error = _apply_contract(
                contract,
                source,
                budget,
                contract_semantics_id=semantics_ids[edge.transform_id],
                ordered_transform_path_ids=transform_paths[edge.target_scale_id],
                ordered_contract_semantics_ids=semantic_path,
            )
            if error is not None or target is None:
                failed = (
                    error or "transform_contract_failed",
                    edge.transform_id,
                    edge.target_scale_id,
                )
                break
            states_by_scale[edge.target_scale_id] = target
            observations_by_scale[edge.target_scale_id] = observations
            semantic_paths[edge.target_scale_id] = semantic_path
    except _ResourceLimit as exc:
        failed = (
            str(exc),
            edge.transform_id,
            edge.target_scale_id,
        )
    commitment = _contract_commitment(
        authority.transform_contracts,
        semantics_ids,
    )
    if failed is not None:
        error_code, transform_id, scale_id = failed
        return _abstaining_compilation(
            reason="transform_bundle_atomic_rejection",
            failure=ExactTransformFailure(
                error_code,
                transform_id=transform_id,
                scale_id=scale_id,
            ),
            wrapper_content_id=wrapper_content_id,
            base_bundle_content_id=base_bundle_content_id,
            uncertainty_receipt=uncertainty_receipt,
            contract_commitment_id=commitment,
            graph_commitment_id=graph_commitment_id,
        )

    ordered_observations = tuple(
        observation
        for scale_id in sorted(observations_by_scale)
        for observation in sorted(
            observations_by_scale[scale_id],
            key=lambda item: item.observation_id,
        )
    )
    observation_ids = {
        observation.observation_id: observation.descriptor_id
        for observation in ordered_observations
    }
    cells: list[ExactTransformedComponent] = []
    for scale_id in sorted(states_by_scale):
        for ref, state in sorted(states_by_scale[scale_id].items()):
            cells.append(
                ExactTransformedComponent(
                    descriptor=state.descriptor,
                    value_kind=state.value_kind,
                    numeric_interval=state.numeric_interval,
                    boolean_value=state.boolean_value,
                    uncertainty_compilation_ids=(
                        state.uncertainty_compilation_ids
                    ),
                    observation_descriptor_id=observation_ids[
                        state.observation.observation_id
                    ],
                    ordered_transform_path_ids=transform_paths[scale_id],
                    ordered_contract_semantics_ids=semantic_paths[scale_id],
                    wrapper_content_id=wrapper_content_id,
                    base_bundle_content_id=base_bundle_content_id,
                    uncertainty_result_id=uncertainty_receipt.result_id,
                    transform_policy_id=policy.policy_id,
                    transform_semantics_id=policy.semantics_id,
                    contract_commitment_id=commitment,
                    graph_commitment_id=graph_commitment_id,
                )
            )
    return ExactTransformCompilation(
        disposition=TransformCompilationDisposition.COMPLETE,
        reason="complete_typed_exact_transform_forest",
        wrapper_content_id=wrapper_content_id,
        base_bundle_content_id=base_bundle_content_id,
        uncertainty_result_id=uncertainty_receipt.result_id,
        uncertainty_policy_id=DEFAULT_EXACT_UNCERTAINTY_POLICY.policy_id,
        phase2b_exact_freeze_id=FROZEN_PHASE2B_EXACT_FREEZE_ID,
        rational_grid_id=FROZEN_RATIONAL_GRID_ID,
        transform_policy_id=policy.policy_id,
        transform_semantics_id=policy.semantics_id,
        contract_commitment_id=commitment,
        graph_commitment_id=graph_commitment_id,
        uncertainty_receipt=uncertainty_receipt,
        observations=ordered_observations,
        components=tuple(cells),
        failures=(),
    )


__all__ = [
    "BoundaryPolicy",
    "CoarseGrainingCertificate",
    "ComponentAxis",
    "ComponentDescriptor",
    "ComponentRef",
    "ComponentValueKind",
    "ComponentValueRole",
    "CoordinateAffineCertificate",
    "DerivedObservationDescriptor",
    "EXACT_TRANSFORM_SEMANTICS_VERSION",
    "EquivalentSplitMergeCertificate",
    "ExactDiscreteMapping",
    "ExactPartitionGroup",
    "ExactSparseAffineRow",
    "ExactSparseTerm",
    "ExactSpatialSupport",
    "ExactTemporalSupport",
    "ExactTransformAtom",
    "ExactTransformCompilation",
    "ExactTransformContract",
    "ExactTransformFailure",
    "ExactTransformInterval",
    "ExactTransformPreflightRejection",
    "ExactTransformedComponent",
    "IdentityTransformCertificate",
    "MissingValuePolicy",
    "ObservationComponentMetadata",
    "PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION",
    "PublicTransformEvidenceBundleV2",
    "ReducerKind",
    "SamplingResolutionCertificate",
    "SpatialAggregationCertificate",
    "SplitMergeDirection",
    "TemporalAggregationCertificate",
    "TransformCompilationDisposition",
    "UnitConversionCertificate",
    "run_exact_transform_semantics",
]

"""Strict accepted-profile codec for native V2 transform authorities.

This module is intentionally independent of the secret batch implementation so
that the batch builder and the custodian replay can share one schema-closed
typed authority path without an import cycle.  No caller-selected schema,
policy, provenance root, or transform result is accepted.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import math
import re
import struct
from typing import Final

from .hashing import stable_hash
from .phase2b_exact_transform_semantics_v1 import (
    BoundaryPolicy,
    CoarseGrainingCertificate,
    ComponentAxis,
    ComponentDescriptor,
    ComponentRef,
    ComponentValueKind,
    ComponentValueRole,
    CoordinateAffineCertificate,
    DerivedObservationDescriptor,
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
    MissingValuePolicy,
    ObservationComponentMetadata,
    PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
    PublicTransformEvidenceBundleV2,
    ReducerKind,
    SamplingResolutionCertificate,
    SpatialAggregationCertificate,
    SplitMergeDirection,
    TemporalAggregationCertificate,
    UnitConversionCertificate,
)
from .phase2b_wire import (
    PUBLIC_EVIDENCE_SCHEMA_VERSION,
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
from .phase2b_trusted_wire_v1 import (
    MAXIMUM_ARRAY_ENTRIES,
    MAXIMUM_ASCII_STRING_BYTES,
    MAXIMUM_PROFILE_DEPTH,
    MAXIMUM_PROFILE_NODES,
    MAXIMUM_RATIONAL_BIT_LENGTH,
    MAXIMUM_SAFE_INTEGER,
    MAXIMUM_UNIQUE_UUIDS,
    MAXIMUM_UUID_OCCURRENCES,
)


TYPED_AUTHORITY_CODEC_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-typed-authority-codec/1"
)

_F64 = re.compile(r"^f64be:([0-9a-f]{16})$")
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SIGNED_DECIMAL = re.compile(r"^(?:0|-[1-9][0-9]*|[1-9][0-9]*)$")
_POSITIVE_DECIMAL = re.compile(r"^[1-9][0-9]*$")


def _mapping(value: object, name: str, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact object")
    if len(value) > MAXIMUM_ARRAY_ENTRIES:
        raise ValueError(f"{name} exceeds the accepted-profile object cap")
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(f"{name} closed schema mismatch: {';'.join(detail)}")
    return value


def _array(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact array")
    if len(value) > MAXIMUM_ARRAY_ENTRIES:
        raise ValueError(f"{name} exceeds the accepted-profile array cap")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use accepted-profile ASCII") from exc
    if len(encoded) > MAXIMUM_ASCII_STRING_BYTES:
        raise ValueError(f"{name} exceeds the accepted-profile string cap")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if abs(value) > MAXIMUM_SAFE_INTEGER:
        raise ValueError(f"{name} exceeds the accepted-profile safe-integer cap")
    return value


def _raw_profile_resource_check(root: object) -> None:
    """Bound an untrusted decoded-JCS tree before schema set operations."""

    nodes = 0
    entries = 0
    string_bytes = 0
    uuid_occurrences = 0
    unique_uuids: set[str] = set()
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAXIMUM_PROFILE_NODES:
            raise ValueError("typed authority profile exceeds the node cap")
        if depth > MAXIMUM_PROFILE_DEPTH:
            raise ValueError("typed authority profile exceeds the depth cap")
        exact_type = type(value)
        if exact_type is dict:
            if len(value) > MAXIMUM_ARRAY_ENTRIES:
                raise ValueError("typed authority object exceeds the entry cap")
            entries += len(value)
            if entries > MAXIMUM_PROFILE_NODES:
                raise ValueError("typed authority profile exceeds the total entry cap")
            for key, item in value.items():
                _string(key, "typed authority object key")
                string_bytes += len(key)
                stack.append((item, depth + 1))
        elif exact_type is list:
            if len(value) > MAXIMUM_ARRAY_ENTRIES:
                raise ValueError("typed authority array exceeds the entry cap")
            entries += len(value)
            if entries > MAXIMUM_PROFILE_NODES:
                raise ValueError("typed authority profile exceeds the total entry cap")
            stack.extend((item, depth + 1) for item in value)
        elif exact_type is str:
            _string(value, "typed authority string")
            string_bytes += len(value)
            if _UUID4.fullmatch(value) is not None:
                uuid_occurrences += 1
                unique_uuids.add(value)
                if uuid_occurrences > MAXIMUM_UUID_OCCURRENCES:
                    raise ValueError("typed authority exceeds the UUID occurrence cap")
                if len(unique_uuids) > MAXIMUM_UNIQUE_UUIDS:
                    raise ValueError("typed authority exceeds the unique UUID cap")
        elif exact_type is int:
            _integer(value, "typed authority integer")
        elif exact_type in (bool, type(None)):
            pass
        else:
            raise TypeError("typed authority profile contains a non-JCS node")
        if string_bytes > MAXIMUM_PROFILE_NODES * 256:
            raise ValueError("typed authority profile exceeds the total string cap")


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact Boolean")
    return value


def _f64(value: object, name: str) -> float:
    text = _string(value, name)
    match = _F64.fullmatch(text)
    if match is None:
        raise ValueError(f"{name} must use canonical f64be lowercase hex")
    result = struct.unpack(">d", bytes.fromhex(match.group(1)))[0]
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite binary64")
    return result


def _tuple_of(
    value: object,
    name: str,
    decoder: object,
) -> tuple[object, ...]:
    decode = decoder
    if not callable(decode):
        raise TypeError("tuple decoder must be callable")
    return tuple(decode(item, f"{name}[{index}]") for index, item in enumerate(_array(value, name)))


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    )


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    )


def _f64_tuple(value: object, name: str) -> tuple[float, ...]:
    return tuple(
        _f64(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    )


def _enum(enum_type: type[Enum], value: object, name: str) -> Enum:
    text = _string(value, name)
    try:
        return enum_type(text)
    except ValueError as exc:
        raise ValueError(f"{name} has an unknown discriminator") from exc


def _atom(value: object, name: str) -> ExactTransformAtom:
    mapping = _mapping(
        value,
        name,
        frozenset({"denominator_decimal", "numerator_decimal"}),
    )
    numerator_text = _string(mapping["numerator_decimal"], f"{name}.numerator")
    denominator_text = _string(
        mapping["denominator_decimal"], f"{name}.denominator"
    )
    if _SIGNED_DECIMAL.fullmatch(numerator_text) is None:
        raise ValueError(f"{name} numerator is not canonical decimal")
    if _POSITIVE_DECIMAL.fullmatch(denominator_text) is None:
        raise ValueError(f"{name} denominator is not canonical positive decimal")
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if (
        numerator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
        or denominator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
    ):
        raise ValueError(f"{name} exceeds the rational bit-length budget")
    return ExactTransformAtom(numerator, denominator)


def _typed_value(value: object, name: str) -> NumericValue | NumericInterval | BooleanValue:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact typed-value object")
    keys = set(value)
    if keys == {"values"}:
        return NumericValue(_f64_tuple(value["values"], f"{name}.values"))
    if keys == {"lower", "upper"}:
        return NumericInterval(
            _f64_tuple(value["lower"], f"{name}.lower"),
            _f64_tuple(value["upper"], f"{name}.upper"),
        )
    if keys == {"value"}:
        return BooleanValue(_boolean(value["value"], f"{name}.value"))
    raise ValueError(f"{name} typed-value discriminator shape is invalid")


def _unit_dimension(value: object, name: str) -> UnitDimension:
    mapping = _mapping(value, name, frozenset({"si_exponents"}))
    return UnitDimension(_integer_tuple(mapping["si_exponents"], f"{name}.si_exponents"))


def _temporal_support(value: object, name: str) -> TemporalSupport:
    mapping = _mapping(value, name, frozenset({"clock_id", "start", "end"}))
    return TemporalSupport(
        _string(mapping["clock_id"], f"{name}.clock_id"),
        _f64(mapping["start"], f"{name}.start"),
        _f64(mapping["end"], f"{name}.end"),
    )


def _spatial_support(value: object, name: str) -> SpatialSupport:
    mapping = _mapping(value, name, frozenset({"frame_id", "lower", "upper"}))
    return SpatialSupport(
        _string(mapping["frame_id"], f"{name}.frame_id"),
        _f64_tuple(mapping["lower"], f"{name}.lower"),
        _f64_tuple(mapping["upper"], f"{name}.upper"),
    )


def _uncertainty(value: object, name: str) -> MeasurementUncertainty:
    mapping = _mapping(value, name, frozenset({"model", "radius"}))
    return MeasurementUncertainty(
        _enum(UncertaintyModel, mapping["model"], f"{name}.model"),  # type: ignore[arg-type]
        _f64_tuple(mapping["radius"], f"{name}.radius"),
    )


def _entity(value: object, name: str) -> EntityCandidate:
    mapping = _mapping(value, name, frozenset({"entity_id", "role_candidate_ids"}))
    return EntityCandidate(
        _string(mapping["entity_id"], f"{name}.entity_id"),
        _string_tuple(mapping["role_candidate_ids"], f"{name}.role_candidate_ids"),
    )


def _observation(value: object, name: str) -> TypedObservation:
    mapping = _mapping(
        value,
        name,
        frozenset(
            {
                "entity_ids",
                "missingness",
                "observation_id",
                "provenance_sha256",
                "quantity_id",
                "role_candidate_ids",
                "source_channel_id",
                "spatial_support",
                "temporal_support",
                "uncertainty",
                "unit_dimension",
                "value",
            }
        ),
    )
    raw_value = mapping["value"]
    temporal = mapping["temporal_support"]
    spatial = mapping["spatial_support"]
    return TypedObservation(
        observation_id=_string(mapping["observation_id"], f"{name}.observation_id"),
        source_channel_id=_string(mapping["source_channel_id"], f"{name}.source_channel_id"),
        entity_ids=_string_tuple(mapping["entity_ids"], f"{name}.entity_ids"),
        role_candidate_ids=_string_tuple(mapping["role_candidate_ids"], f"{name}.role_candidate_ids"),
        quantity_id=_string(mapping["quantity_id"], f"{name}.quantity_id"),
        value=None if raw_value is None else _typed_value(raw_value, f"{name}.value"),
        unit_dimension=_unit_dimension(mapping["unit_dimension"], f"{name}.unit_dimension"),
        temporal_support=None if temporal is None else _temporal_support(temporal, f"{name}.temporal_support"),
        spatial_support=None if spatial is None else _spatial_support(spatial, f"{name}.spatial_support"),
        uncertainty=_uncertainty(mapping["uncertainty"], f"{name}.uncertainty"),
        provenance_sha256=_string(mapping["provenance_sha256"], f"{name}.provenance_sha256"),
        missingness=_enum(Missingness, mapping["missingness"], f"{name}.missingness"),  # type: ignore[arg-type]
    )


def _task(value: object, name: str) -> TaskTarget:
    mapping = _mapping(value, name, frozenset({"task_id", "entity_ids", "quantity_ids"}))
    return TaskTarget(
        _string(mapping["task_id"], f"{name}.task_id"),
        _string_tuple(mapping["entity_ids"], f"{name}.entity_ids"),
        _string_tuple(mapping["quantity_ids"], f"{name}.quantity_ids"),
    )


def _edge(value: object, name: str) -> AggregationEdge:
    mapping = _mapping(value, name, frozenset({"source_scale_id", "target_scale_id", "transform_id"}))
    return AggregationEdge(
        _string(mapping["source_scale_id"], f"{name}.source_scale_id"),
        _string(mapping["target_scale_id"], f"{name}.target_scale_id"),
        _string(mapping["transform_id"], f"{name}.transform_id"),
    )


def _graph(value: object, name: str) -> AggregationGraph:
    mapping = _mapping(value, name, frozenset({"scale_ids", "root_scale_ids", "edges"}))
    return AggregationGraph(
        _string_tuple(mapping["scale_ids"], f"{name}.scale_ids"),
        _string_tuple(mapping["root_scale_ids"], f"{name}.root_scale_ids"),
        tuple(_edge(item, f"{name}.edges[{index}]") for index, item in enumerate(_array(mapping["edges"], f"{name}.edges"))),
    )


def _transform_spec(value: object, name: str) -> TransformSpec:
    mapping = _mapping(value, name, frozenset({"transform_id", "operation", "parameters"}))
    return TransformSpec(
        _string(mapping["transform_id"], f"{name}.transform_id"),
        _enum(TransformOperation, mapping["operation"], f"{name}.operation"),  # type: ignore[arg-type]
        _f64_tuple(mapping["parameters"], f"{name}.parameters"),
    )


def _base_bundle(value: object, name: str) -> PublicEvidenceBundle:
    mapping = _mapping(
        value,
        name,
        frozenset(
            {
                "aggregation_graph",
                "bundle_id",
                "entity_candidates",
                "missingness_mask",
                "observations",
                "quantity_ids",
                "role_ids",
                "schema_version",
                "task_target",
                "transform_catalog",
            }
        ),
    )
    if mapping["schema_version"] != PUBLIC_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("public evidence profile schema version drift")
    return PublicEvidenceBundle(
        schema_version=_string(mapping["schema_version"], f"{name}.schema_version"),
        bundle_id=_string(mapping["bundle_id"], f"{name}.bundle_id"),
        entity_candidates=tuple(_entity(item, f"{name}.entity_candidates[{index}]") for index, item in enumerate(_array(mapping["entity_candidates"], f"{name}.entity_candidates"))),
        role_ids=_string_tuple(mapping["role_ids"], f"{name}.role_ids"),
        quantity_ids=_string_tuple(mapping["quantity_ids"], f"{name}.quantity_ids"),
        observations=tuple(_observation(item, f"{name}.observations[{index}]") for index, item in enumerate(_array(mapping["observations"], f"{name}.observations"))),
        task_target=_task(mapping["task_target"], f"{name}.task_target"),
        aggregation_graph=_graph(mapping["aggregation_graph"], f"{name}.aggregation_graph"),
        transform_catalog=tuple(_transform_spec(item, f"{name}.transform_catalog[{index}]") for index, item in enumerate(_array(mapping["transform_catalog"], f"{name}.transform_catalog"))),
        missingness_mask=_string_tuple(mapping["missingness_mask"], f"{name}.missingness_mask"),
    )


def _exact_temporal(value: object, name: str) -> ExactTemporalSupport:
    mapping = _mapping(value, name, frozenset({"clock_id", "start", "end"}))
    return ExactTemporalSupport(
        _string(mapping["clock_id"], f"{name}.clock_id"),
        _atom(mapping["start"], f"{name}.start"),
        _atom(mapping["end"], f"{name}.end"),
    )


def _exact_spatial(value: object, name: str) -> ExactSpatialSupport:
    mapping = _mapping(value, name, frozenset({"frame_id", "lower", "upper"}))
    return ExactSpatialSupport(
        _string(mapping["frame_id"], f"{name}.frame_id"),
        tuple(
            _atom(item, f"{name}.lower[{index}]")
            for index, item in enumerate(_array(mapping["lower"], f"{name}.lower"))
        ),
        tuple(
            _atom(item, f"{name}.upper[{index}]")
            for index, item in enumerate(_array(mapping["upper"], f"{name}.upper"))
        ),
    )


def _component_ref(value: object, name: str) -> ComponentRef:
    mapping = _mapping(
        value,
        name,
        frozenset({"scale_id", "observation_id", "ordinal", "component_id"}),
    )
    return ComponentRef(
        _string(mapping["scale_id"], f"{name}.scale_id"),
        _string(mapping["observation_id"], f"{name}.observation_id"),
        _integer(mapping["ordinal"], f"{name}.ordinal"),
        _string(mapping["component_id"], f"{name}.component_id"),
    )


def _metadata(value: object, name: str) -> ObservationComponentMetadata:
    mapping = _mapping(
        value,
        name,
        frozenset(
            {
                "axis",
                "component_ids",
                "coordinate_frame_id",
                "observation_id",
                "scale_id",
                "unit_id",
                "value_role",
            }
        ),
    )
    unit_id = mapping["unit_id"]
    frame_id = mapping["coordinate_frame_id"]
    return ObservationComponentMetadata(
        observation_id=_string(mapping["observation_id"], f"{name}.observation_id"),
        scale_id=_string(mapping["scale_id"], f"{name}.scale_id"),
        component_ids=_string_tuple(mapping["component_ids"], f"{name}.component_ids"),
        axis=_enum(ComponentAxis, mapping["axis"], f"{name}.axis"),  # type: ignore[arg-type]
        value_role=_enum(ComponentValueRole, mapping["value_role"], f"{name}.value_role"),  # type: ignore[arg-type]
        unit_id=None if unit_id is None else _string(unit_id, f"{name}.unit_id"),
        coordinate_frame_id=None if frame_id is None else _string(frame_id, f"{name}.coordinate_frame_id"),
    )


def _component_descriptor(value: object, name: str) -> ComponentDescriptor:
    mapping = _mapping(
        value,
        name,
        frozenset(
            {
                "axis",
                "coordinate_frame_id",
                "ref",
                "si_exponents",
                "spatial_support",
                "temporal_support",
                "unit_id",
                "value_role",
            }
        ),
    )
    unit_id = mapping["unit_id"]
    frame_id = mapping["coordinate_frame_id"]
    temporal = mapping["temporal_support"]
    spatial = mapping["spatial_support"]
    return ComponentDescriptor(
        ref=_component_ref(mapping["ref"], f"{name}.ref"),
        axis=_enum(ComponentAxis, mapping["axis"], f"{name}.axis"),  # type: ignore[arg-type]
        value_role=_enum(ComponentValueRole, mapping["value_role"], f"{name}.value_role"),  # type: ignore[arg-type]
        unit_id=None if unit_id is None else _string(unit_id, f"{name}.unit_id"),
        si_exponents=_integer_tuple(mapping["si_exponents"], f"{name}.si_exponents"),  # type: ignore[arg-type]
        coordinate_frame_id=None if frame_id is None else _string(frame_id, f"{name}.coordinate_frame_id"),
        temporal_support=None if temporal is None else _exact_temporal(temporal, f"{name}.temporal_support"),
        spatial_support=None if spatial is None else _exact_spatial(spatial, f"{name}.spatial_support"),
    )


def _derived_observation(value: object, name: str) -> DerivedObservationDescriptor:
    mapping = _mapping(
        value,
        name,
        frozenset(
            {
                "component_refs",
                "entity_ids",
                "observation_id",
                "provenance_sha256",
                "quantity_id",
                "role_candidate_ids",
                "scale_id",
                "si_exponents",
                "source_channel_id",
                "source_observation_ids",
                "spatial_support",
                "temporal_support",
                "unit_id",
                "value_kind",
            }
        ),
    )
    unit_id = mapping["unit_id"]
    temporal = mapping["temporal_support"]
    spatial = mapping["spatial_support"]
    return DerivedObservationDescriptor(
        scale_id=_string(mapping["scale_id"], f"{name}.scale_id"),
        observation_id=_string(mapping["observation_id"], f"{name}.observation_id"),
        source_channel_id=_string(mapping["source_channel_id"], f"{name}.source_channel_id"),
        entity_ids=_string_tuple(mapping["entity_ids"], f"{name}.entity_ids"),
        role_candidate_ids=_string_tuple(mapping["role_candidate_ids"], f"{name}.role_candidate_ids"),
        quantity_id=_string(mapping["quantity_id"], f"{name}.quantity_id"),
        unit_id=None if unit_id is None else _string(unit_id, f"{name}.unit_id"),
        si_exponents=_integer_tuple(mapping["si_exponents"], f"{name}.si_exponents"),  # type: ignore[arg-type]
        temporal_support=None if temporal is None else _exact_temporal(temporal, f"{name}.temporal_support"),
        spatial_support=None if spatial is None else _exact_spatial(spatial, f"{name}.spatial_support"),
        provenance_sha256=_string(mapping["provenance_sha256"], f"{name}.provenance_sha256"),
        source_observation_ids=_string_tuple(mapping["source_observation_ids"], f"{name}.source_observation_ids"),
        value_kind=_enum(ComponentValueKind, mapping["value_kind"], f"{name}.value_kind"),  # type: ignore[arg-type]
        component_refs=tuple(
            _component_ref(item, f"{name}.component_refs[{index}]")
            for index, item in enumerate(_array(mapping["component_refs"], f"{name}.component_refs"))
        ),
    )


def _sparse_term(value: object, name: str) -> ExactSparseTerm:
    mapping = _mapping(value, name, frozenset({"input_ref", "coefficient"}))
    return ExactSparseTerm(
        _component_ref(mapping["input_ref"], f"{name}.input_ref"),
        _atom(mapping["coefficient"], f"{name}.coefficient"),
    )


def _sparse_row(value: object, name: str) -> ExactSparseAffineRow:
    mapping = _mapping(value, name, frozenset({"output_ref", "terms", "offset"}))
    return ExactSparseAffineRow(
        output_ref=_component_ref(mapping["output_ref"], f"{name}.output_ref"),
        terms=tuple(
            _sparse_term(item, f"{name}.terms[{index}]")
            for index, item in enumerate(_array(mapping["terms"], f"{name}.terms"))
        ),
        offset=_atom(mapping["offset"], f"{name}.offset"),
    )


def _discrete_mapping(value: object, name: str) -> ExactDiscreteMapping:
    mapping = _mapping(value, name, frozenset({"input_ref", "output_ref"}))
    return ExactDiscreteMapping(
        _component_ref(mapping["input_ref"], f"{name}.input_ref"),
        _component_ref(mapping["output_ref"], f"{name}.output_ref"),
    )


def _partition_group(value: object, name: str) -> ExactPartitionGroup:
    mapping = _mapping(value, name, frozenset({"input_refs", "output_refs"}))
    return ExactPartitionGroup(
        tuple(
            _component_ref(item, f"{name}.input_refs[{index}]")
            for index, item in enumerate(_array(mapping["input_refs"], f"{name}.input_refs"))
        ),
        tuple(
            _component_ref(item, f"{name}.output_refs[{index}]")
            for index, item in enumerate(_array(mapping["output_refs"], f"{name}.output_refs"))
        ),
    )


def _rows(value: object, name: str) -> tuple[ExactSparseAffineRow, ...]:
    return tuple(
        _sparse_row(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    )


def _groups(value: object, name: str) -> tuple[ExactPartitionGroup, ...]:
    return tuple(
        _partition_group(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    )


def _certificate(
    operation: TransformOperation,
    value: object,
    name: str,
) -> object:
    if operation is TransformOperation.IDENTITY:
        mapping = _mapping(
            value,
            name,
            frozenset({"semantics_version", "missing_policy", "inverse_contract"}),
        )
        return IdentityTransformCertificate(
            semantics_version=_string(mapping["semantics_version"], f"{name}.semantics_version"),
            missing_policy=_enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),  # type: ignore[arg-type]
            inverse_contract=_string(mapping["inverse_contract"], f"{name}.inverse_contract"),
        )
    if operation is TransformOperation.UNIT_CONVERSION:
        mapping = _mapping(
            value,
            name,
            frozenset(
                {
                    "commutation_contract",
                    "factor",
                    "inverse_factor",
                    "missing_policy",
                    "orientation",
                    "source_unit_id",
                    "target_unit_id",
                }
            ),
        )
        return UnitConversionCertificate(
            source_unit_id=_string(mapping["source_unit_id"], f"{name}.source_unit_id"),
            target_unit_id=_string(mapping["target_unit_id"], f"{name}.target_unit_id"),
            factor=_atom(mapping["factor"], f"{name}.factor"),
            inverse_factor=_atom(mapping["inverse_factor"], f"{name}.inverse_factor"),
            missing_policy=_enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),  # type: ignore[arg-type]
            orientation=_string(mapping["orientation"], f"{name}.orientation"),
            commutation_contract=_string(mapping["commutation_contract"], f"{name}.commutation_contract"),
        )
    if operation is TransformOperation.COORDINATE_AFFINE:
        mapping = _mapping(
            value,
            name,
            frozenset(
                {
                    "dimension",
                    "inverse_contract",
                    "inverse_rows",
                    "missing_policy",
                    "source_frame_id",
                    "support_contract",
                    "target_frame_id",
                }
            ),
        )
        return CoordinateAffineCertificate(
            source_frame_id=_string(mapping["source_frame_id"], f"{name}.source_frame_id"),
            target_frame_id=_string(mapping["target_frame_id"], f"{name}.target_frame_id"),
            dimension=_integer(mapping["dimension"], f"{name}.dimension"),
            inverse_rows=_rows(mapping["inverse_rows"], f"{name}.inverse_rows"),
            missing_policy=_enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),  # type: ignore[arg-type]
            support_contract=_string(mapping["support_contract"], f"{name}.support_contract"),
            inverse_contract=_string(mapping["inverse_contract"], f"{name}.inverse_contract"),
        )
    if operation in (
        TransformOperation.TEMPORAL_AGGREGATION,
        TransformOperation.SPATIAL_AGGREGATION,
    ):
        mapping = _mapping(
            value,
            name,
            frozenset(
                {
                    "boundary_policy",
                    "groups",
                    "missing_policy",
                    "reducer",
                    "support_contract",
                }
            ),
        )
        values = {
            "reducer": _enum(ReducerKind, mapping["reducer"], f"{name}.reducer"),
            "groups": _groups(mapping["groups"], f"{name}.groups"),
            "missing_policy": _enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),
            "boundary_policy": _enum(BoundaryPolicy, mapping["boundary_policy"], f"{name}.boundary_policy"),
            "support_contract": _string(mapping["support_contract"], f"{name}.support_contract"),
        }
        if operation is TransformOperation.TEMPORAL_AGGREGATION:
            return TemporalAggregationCertificate(**values)  # type: ignore[arg-type]
        return SpatialAggregationCertificate(**values)  # type: ignore[arg-type]
    if operation is TransformOperation.SAMPLING_RESOLUTION:
        mapping = _mapping(
            value,
            name,
            frozenset(
                {
                    "axis",
                    "boundary_policy",
                    "discarded_inputs",
                    "grid_dimension",
                    "grid_frame_id",
                    "grid_points",
                    "kernel_contract",
                    "missing_policy",
                    "selected_inputs",
                }
            ),
        )
        frame_id = mapping["grid_frame_id"]
        return SamplingResolutionCertificate(
            axis=_enum(ComponentAxis, mapping["axis"], f"{name}.axis"),  # type: ignore[arg-type]
            selected_inputs=tuple(
                _component_ref(item, f"{name}.selected_inputs[{index}]")
                for index, item in enumerate(_array(mapping["selected_inputs"], f"{name}.selected_inputs"))
            ),
            discarded_inputs=tuple(
                _component_ref(item, f"{name}.discarded_inputs[{index}]")
                for index, item in enumerate(_array(mapping["discarded_inputs"], f"{name}.discarded_inputs"))
            ),
            grid_points=tuple(
                tuple(
                    _atom(atom, f"{name}.grid_points[{point_index}][{atom_index}]")
                    for atom_index, atom in enumerate(_array(point, f"{name}.grid_points[{point_index}]"))
                )
                for point_index, point in enumerate(_array(mapping["grid_points"], f"{name}.grid_points"))
            ),
            grid_dimension=_integer(mapping["grid_dimension"], f"{name}.grid_dimension"),
            grid_frame_id=None if frame_id is None else _string(frame_id, f"{name}.grid_frame_id"),
            missing_policy=_enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),  # type: ignore[arg-type]
            boundary_policy=_enum(BoundaryPolicy, mapping["boundary_policy"], f"{name}.boundary_policy"),  # type: ignore[arg-type]
            kernel_contract=_string(mapping["kernel_contract"], f"{name}.kernel_contract"),
        )
    if operation is TransformOperation.EQUIVALENT_SPLIT_MERGE:
        mapping = _mapping(
            value,
            name,
            frozenset(
                {
                    "direction",
                    "equivalence_contract",
                    "groups",
                    "inverse_rows",
                    "missing_policy",
                }
            ),
        )
        return EquivalentSplitMergeCertificate(
            direction=_enum(SplitMergeDirection, mapping["direction"], f"{name}.direction"),  # type: ignore[arg-type]
            groups=_groups(mapping["groups"], f"{name}.groups"),
            inverse_rows=_rows(mapping["inverse_rows"], f"{name}.inverse_rows"),
            missing_policy=_enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),  # type: ignore[arg-type]
            equivalence_contract=_string(mapping["equivalence_contract"], f"{name}.equivalence_contract"),
        )
    if operation is TransformOperation.COARSE_GRAINING:
        mapping = _mapping(
            value,
            name,
            frozenset(
                {
                    "boundary_policy",
                    "commutation_contract",
                    "groups",
                    "missing_policy",
                    "quotient_class_ids",
                    "reducer",
                    "source_commutation_rows",
                    "target_commutation_rows",
                }
            ),
        )
        return CoarseGrainingCertificate(
            reducer=_enum(ReducerKind, mapping["reducer"], f"{name}.reducer"),  # type: ignore[arg-type]
            groups=_groups(mapping["groups"], f"{name}.groups"),
            quotient_class_ids=_string_tuple(mapping["quotient_class_ids"], f"{name}.quotient_class_ids"),
            source_commutation_rows=_rows(mapping["source_commutation_rows"], f"{name}.source_commutation_rows"),
            target_commutation_rows=_rows(mapping["target_commutation_rows"], f"{name}.target_commutation_rows"),
            missing_policy=_enum(MissingValuePolicy, mapping["missing_policy"], f"{name}.missing_policy"),  # type: ignore[arg-type]
            boundary_policy=_enum(BoundaryPolicy, mapping["boundary_policy"], f"{name}.boundary_policy"),  # type: ignore[arg-type]
            commutation_contract=_string(mapping["commutation_contract"], f"{name}.commutation_contract"),
        )
    raise ValueError(f"{name} has no frozen certificate discriminator")


def _contract(value: object, name: str) -> ExactTransformContract:
    mapping = _mapping(
        value,
        name,
        frozenset(
            {
                "certificate",
                "discrete_mappings",
                "input_components",
                "kernel_rows",
                "operation",
                "output_components",
                "output_observations",
                "source_scale_id",
                "target_scale_id",
                "transform_id",
            }
        ),
    )
    operation = _enum(TransformOperation, mapping["operation"], f"{name}.operation")
    assert isinstance(operation, TransformOperation)
    return ExactTransformContract(
        transform_id=_string(mapping["transform_id"], f"{name}.transform_id"),
        operation=operation,
        source_scale_id=_string(mapping["source_scale_id"], f"{name}.source_scale_id"),
        target_scale_id=_string(mapping["target_scale_id"], f"{name}.target_scale_id"),
        input_components=tuple(
            _component_ref(item, f"{name}.input_components[{index}]")
            for index, item in enumerate(_array(mapping["input_components"], f"{name}.input_components"))
        ),
        output_components=tuple(
            _component_descriptor(item, f"{name}.output_components[{index}]")
            for index, item in enumerate(_array(mapping["output_components"], f"{name}.output_components"))
        ),
        output_observations=tuple(
            _derived_observation(item, f"{name}.output_observations[{index}]")
            for index, item in enumerate(_array(mapping["output_observations"], f"{name}.output_observations"))
        ),
        kernel_rows=_rows(mapping["kernel_rows"], f"{name}.kernel_rows"),
        discrete_mappings=tuple(
            _discrete_mapping(item, f"{name}.discrete_mappings[{index}]")
            for index, item in enumerate(_array(mapping["discrete_mappings"], f"{name}.discrete_mappings"))
        ),
        certificate=_certificate(operation, mapping["certificate"], f"{name}.certificate"),  # type: ignore[arg-type]
    )


_PROFILE_DATACLASS_TYPES: Final = frozenset(
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
_PROFILE_ENUM_TYPES: Final = frozenset(
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


def _typed_authority_resource_check(root: PublicTransformEvidenceBundleV2) -> None:
    """Reject polluted typed trees before profile conversion or content hashing."""

    nodes = 0
    entries = 0
    string_bytes = 0
    uuid_occurrences = 0
    unique_uuids: set[str] = set()
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAXIMUM_PROFILE_NODES:
            raise ValueError("typed authority exceeds the node cap")
        if depth > MAXIMUM_PROFILE_DEPTH:
            raise ValueError("typed authority exceeds the depth cap")
        exact_type = type(value)
        if exact_type is ExactTransformAtom:
            if type(value.numerator) is not int or type(value.denominator) is not int:
                raise TypeError("typed authority exact atom fields must be integers")
            if (
                value.numerator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
                or value.denominator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
            ):
                raise ValueError("typed authority exact atom exceeds the bit cap")
            entries += 2
            stack.append((str(value.numerator), depth + 1))
            stack.append((str(value.denominator), depth + 1))
        elif exact_type is str:
            _string(value, "typed authority string")
            string_bytes += len(value)
            if _UUID4.fullmatch(value) is not None:
                uuid_occurrences += 1
                unique_uuids.add(value)
                if uuid_occurrences > MAXIMUM_UUID_OCCURRENCES:
                    raise ValueError("typed authority exceeds the UUID occurrence cap")
                if len(unique_uuids) > MAXIMUM_UNIQUE_UUIDS:
                    raise ValueError("typed authority exceeds the unique UUID cap")
        elif exact_type is int:
            _integer(value, "typed authority integer")
        elif exact_type is float:
            if not math.isfinite(value):
                raise ValueError("typed authority contains nonfinite binary64")
        elif exact_type in (bool, type(None)):
            pass
        elif exact_type in _PROFILE_ENUM_TYPES:
            pass
        elif exact_type is tuple:
            if len(value) > MAXIMUM_ARRAY_ENTRIES:
                raise ValueError("typed authority tuple exceeds the entry cap")
            entries += len(value)
            stack.extend((item, depth + 1) for item in value)
        elif exact_type in _PROFILE_DATACLASS_TYPES:
            rows = fields(value)
            if len(rows) > MAXIMUM_ARRAY_ENTRIES:
                raise ValueError("typed authority object exceeds the entry cap")
            entries += len(rows)
            stack.extend(
                (getattr(value, item.name), depth + 1) for item in rows
            )
        else:
            raise TypeError("typed authority contains a non-schema node")
        if entries > MAXIMUM_PROFILE_NODES:
            raise ValueError("typed authority exceeds the total entry cap")
        if string_bytes > MAXIMUM_PROFILE_NODES * 256:
            raise ValueError("typed authority exceeds the total string cap")


_OPERATION_CERTIFICATE_MANIFEST: Final = (
    ("coarse_graining", "CoarseGrainingCertificate"),
    ("coordinate_affine", "CoordinateAffineCertificate"),
    ("equivalent_split_merge", "EquivalentSplitMergeCertificate"),
    ("identity", "IdentityTransformCertificate"),
    ("sampling_resolution", "SamplingResolutionCertificate"),
    ("spatial_aggregation", "SpatialAggregationCertificate"),
    ("temporal_aggregation", "TemporalAggregationCertificate"),
    ("unit_conversion", "UnitConversionCertificate"),
)
_F64_FIELD_PATH_MANIFEST: Final = (
    "/base_bundle/observations/*/spatial_support/lower/*",
    "/base_bundle/observations/*/spatial_support/upper/*",
    "/base_bundle/observations/*/temporal_support/end",
    "/base_bundle/observations/*/temporal_support/start",
    "/base_bundle/observations/*/uncertainty/radius/*",
    "/base_bundle/observations/*/value/lower/*",
    "/base_bundle/observations/*/value/upper/*",
    "/base_bundle/observations/*/value/values/*",
    "/base_bundle/transform_catalog/*/parameters/*",
)
_EXACT_ATOM_FIELD_PATH_MANIFEST: Final = (
    "/transform_contracts/*/certificate/factor",
    "/transform_contracts/*/certificate/grid_points/*/*",
    "/transform_contracts/*/certificate/inverse_factor",
    "/transform_contracts/*/certificate/inverse_rows/*/offset",
    "/transform_contracts/*/certificate/inverse_rows/*/terms/*/coefficient",
    "/transform_contracts/*/certificate/source_commutation_rows/*/offset",
    "/transform_contracts/*/certificate/source_commutation_rows/*/terms/*/coefficient",
    "/transform_contracts/*/certificate/target_commutation_rows/*/offset",
    "/transform_contracts/*/certificate/target_commutation_rows/*/terms/*/coefficient",
    "/transform_contracts/*/kernel_rows/*/offset",
    "/transform_contracts/*/kernel_rows/*/terms/*/coefficient",
    "/transform_contracts/*/output_components/*/spatial_support/lower/*",
    "/transform_contracts/*/output_components/*/spatial_support/upper/*",
    "/transform_contracts/*/output_components/*/temporal_support/end",
    "/transform_contracts/*/output_components/*/temporal_support/start",
    "/transform_contracts/*/output_observations/*/spatial_support/lower/*",
    "/transform_contracts/*/output_observations/*/spatial_support/upper/*",
    "/transform_contracts/*/output_observations/*/temporal_support/end",
    "/transform_contracts/*/output_observations/*/temporal_support/start",
)
_NULLABLE_FIELD_PATH_MANIFEST: Final = (
    "/base_bundle/observations/*/spatial_support",
    "/base_bundle/observations/*/temporal_support",
    "/base_bundle/observations/*/value",
    "/observation_metadata/*/coordinate_frame_id",
    "/observation_metadata/*/unit_id",
    "/transform_contracts/*/certificate/grid_frame_id",
    "/transform_contracts/*/output_components/*/coordinate_frame_id",
    "/transform_contracts/*/output_components/*/spatial_support",
    "/transform_contracts/*/output_components/*/temporal_support",
    "/transform_contracts/*/output_components/*/unit_id",
    "/transform_contracts/*/output_observations/*/spatial_support",
    "/transform_contracts/*/output_observations/*/temporal_support",
    "/transform_contracts/*/output_observations/*/unit_id",
)
_DATACLASS_WIRE_GRAMMAR_MANIFEST: Final = tuple(
    (
        value.__name__,
        tuple(
            (
                item.name,
                str(item.type),
                "required_serialized_field",
            )
            for item in fields(value)
        ),
    )
    for value in sorted(
        _PROFILE_DATACLASS_TYPES,
        key=lambda item: item.__name__,
    )
)
_SCALAR_AND_CONTAINER_WIRE_GRAMMAR: Final = (
    "dataclass=exact_dict_all_declared_fields_required_no_extra",
    "enum=exact_string_member_value",
    "tuple=exact_json_list_constructor_cardinality_and_canonical_order",
    "str=exact_ascii_string_max_2048_bytes",
    "int=exact_non_boolean_json_integer_abs_le_2^53_minus_1",
    "float=exact_f64be_colon_16_lowercase_hex_finite",
    "bool=exact_json_boolean",
    "none=exact_json_null_only_at_nullable_manifest_paths",
    "exact_atom=exact_two_key_decimal_string_object_reduced_positive_denominator",
    "typed_value=exact_union_shape_values_or_lower_upper_or_value",
    "root=PublicTransformEvidenceBundleV2_exact_object",
)


TYPED_AUTHORITY_SCHEMA_ID: Final = stable_hash(
    {
        "caps": {
            "maximum_array_entries": MAXIMUM_ARRAY_ENTRIES,
            "maximum_ascii_string_bytes": MAXIMUM_ASCII_STRING_BYTES,
            "maximum_profile_depth": MAXIMUM_PROFILE_DEPTH,
            "maximum_profile_nodes": MAXIMUM_PROFILE_NODES,
            "maximum_rational_bit_length": MAXIMUM_RATIONAL_BIT_LENGTH,
            "maximum_safe_integer": MAXIMUM_SAFE_INTEGER,
            "maximum_unique_uuids": MAXIMUM_UNIQUE_UUIDS,
            "maximum_uuid_occurrences": MAXIMUM_UUID_OCCURRENCES,
            "total_string_bytes": MAXIMUM_PROFILE_NODES * 256,
        },
        "codec_version": TYPED_AUTHORITY_CODEC_VERSION,
        "public_evidence_schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
        "public_transform_evidence_schema_version": (
            PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION
        ),
        "dataclass_wire_grammar": _DATACLASS_WIRE_GRAMMAR_MANIFEST,
        "enum_values": tuple(
            (
                value.__name__,
                tuple(item.value for item in value),
            )
            for value in sorted(
                _PROFILE_ENUM_TYPES,
                key=lambda item: item.__name__,
            )
        ),
        "float_representation": "f64be:16-lowercase-hex",
        "float_field_paths": _F64_FIELD_PATH_MANIFEST,
        "exact_atom_field_paths": _EXACT_ATOM_FIELD_PATH_MANIFEST,
        "nullable_field_paths": _NULLABLE_FIELD_PATH_MANIFEST,
        "operation_certificate_manifest": _OPERATION_CERTIFICATE_MANIFEST,
        "rational_representation": (
            "reduced-canonical-decimal-string-pair-positive-denominator"
        ),
        "typed_value_discriminator": (
            "exact-key-set:{values}|{lower,upper}|{value}"
        ),
        "wire_scalar_and_container_grammar": (
            _SCALAR_AND_CONTAINER_WIRE_GRAMMAR
        ),
    },
    prefix="phase2b_trusted_wire_typed_authority_schema_",
)
TYPED_AUTHORITY_CODEC_POLICY_ID: Final = stable_hash(
    {
        "accepted_root": "exact_dict_only",
        "canonicality": "decode_then_exact_profile_reencode_equality",
        "resource_preflight": "iterative_before_schema_set_or_content_hash",
        "schema_id": TYPED_AUTHORITY_SCHEMA_ID,
        "version": TYPED_AUTHORITY_CODEC_VERSION,
    },
    prefix="phase2b_trusted_wire_typed_authority_codec_policy_",
)


def _profile_value_unchecked(value: object) -> object:
    exact_type = type(value)
    if exact_type is ExactTransformAtom:
        return {
            "denominator_decimal": str(value.denominator),
            "numerator_decimal": str(value.numerator),
        }
    if exact_type is float:
        if not math.isfinite(value):
            raise ValueError("typed authority contains nonfinite binary64")
        return "f64be:" + struct.pack(">d", value).hex()
    if exact_type in (str, int, bool, type(None)):
        return value
    if exact_type in _PROFILE_ENUM_TYPES:
        return value.value
    if exact_type is tuple:
        return [_profile_value_unchecked(item) for item in value]
    if exact_type in _PROFILE_DATACLASS_TYPES:
        return {
            item.name: _profile_value_unchecked(getattr(value, item.name))
            for item in fields(value)
        }
    if is_dataclass(value):
        raise TypeError("typed authority contains a non-schema dataclass")
    raise TypeError("typed authority contains a value outside the closed schema")


def encode_typed_transform_authority_profile_v1(
    authority: PublicTransformEvidenceBundleV2,
) -> dict[str, object]:
    """Encode one exact V2 authority into its closed accepted-profile value."""

    if type(authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError("typed authority encoder requires the exact V2 type")
    _typed_authority_resource_check(authority)
    encoded = _profile_value_unchecked(authority)
    if type(encoded) is not dict:
        raise RuntimeError("typed authority profile root encoding drift")
    reconstructed = _decode_typed_transform_authority_profile_unchecked(encoded)
    if reconstructed != authority:
        raise ValueError("typed authority tree is polluted or noncanonical")
    return encoded


def _decode_typed_transform_authority_profile_unchecked(
    authority_profile: object,
) -> PublicTransformEvidenceBundleV2:
    mapping = _mapping(
        authority_profile,
        "typed authority profile",
        frozenset(
            {
                "base_bundle",
                "observation_metadata",
                "schema_version",
                "transform_contracts",
            }
        ),
    )
    schema_version = _string(
        mapping["schema_version"],
        "typed authority profile.schema_version",
    )
    if schema_version != PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("typed authority profile schema version drift")
    authority = PublicTransformEvidenceBundleV2(
        schema_version=schema_version,
        base_bundle=_base_bundle(
            mapping["base_bundle"],
            "typed authority profile.base_bundle",
        ),
        observation_metadata=tuple(
            _metadata(
                item,
                f"typed authority profile.observation_metadata[{index}]",
            )
            for index, item in enumerate(
                _array(
                    mapping["observation_metadata"],
                    "typed authority profile.observation_metadata",
                )
            )
        ),
        transform_contracts=tuple(
            _contract(
                item,
                f"typed authority profile.transform_contracts[{index}]",
            )
            for index, item in enumerate(
                _array(
                    mapping["transform_contracts"],
                    "typed authority profile.transform_contracts",
                )
            )
        ),
    )
    if _profile_value_unchecked(authority) != mapping:
        raise ValueError(
            "typed authority profile is not canonical or losslessly decodable"
        )
    return authority


def decode_typed_transform_authority_profile_v1(
    authority_profile: object,
) -> PublicTransformEvidenceBundleV2:
    """Losslessly decode one exact closed-schema accepted-profile authority."""

    _raw_profile_resource_check(authority_profile)
    return _decode_typed_transform_authority_profile_unchecked(authority_profile)


__all__ = (
    "TYPED_AUTHORITY_CODEC_VERSION",
    "TYPED_AUTHORITY_CODEC_POLICY_ID",
    "TYPED_AUTHORITY_SCHEMA_ID",
    "decode_typed_transform_authority_profile_v1",
    "encode_typed_transform_authority_profile_v1",
)

"""Public Phase-2B wire contracts with no evaluator or candidate payloads.

This module is deliberately dependency-light.  It describes the only evidence
that an isolated recognizer may receive and the only decision that it may emit.
All identifiers are opaque UUIDv4 values, mappings use exact field allowlists,
and every set-like collection is canonicalized before content addressing.

The contract starts after raw extraction but before structural projection.  It
therefore contains typed measurements and an aggregation graph, never a law
family, verifier observable, gold binding/scale, answer, or candidate-private
payload.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TypeAlias
from uuid import UUID

from .hashing import canonical_json, stable_hash


PUBLIC_EVIDENCE_SCHEMA_VERSION = "hegel-machine-phase2b-public-evidence/1"
PREDICTION_SCHEMA_VERSION = "hegel-machine-phase2b-prediction/1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _strict_mapping(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    keys = set(value)
    unknown = keys - required - optional
    missing = required - keys
    if unknown:
        raise ValueError(
            f"{name} contains unknown or forbidden fields: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            f"{name} is missing required fields: " + ", ".join(sorted(missing))
        )
    return value


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{name} must be an array")
    return tuple(value)


def _uuid4(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an opaque UUIDv4 string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be an opaque UUIDv4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical lowercase UUIDv4 string")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _uuid_tuple(
    value: object,
    *,
    name: str,
    nonempty: bool = True,
) -> tuple[str, ...]:
    items = tuple(
        _uuid4(item, name=f"{name} item")
        for item in _sequence(value, name=name)
    )
    if nonempty and not items:
        raise ValueError(f"{name} cannot be empty")
    if len(items) != len(set(items)):
        raise ValueError(f"{name} contains duplicate UUIDs")
    return tuple(sorted(items))


def _number_tuple(
    value: object,
    *,
    name: str,
    nonempty: bool = True,
) -> tuple[float, ...]:
    items = tuple(
        _number(item, name=f"{name} item")
        for item in _sequence(value, name=name)
    )
    if nonempty and not items:
        raise ValueError(f"{name} cannot be empty")
    return items


class Missingness(str, Enum):
    OBSERVED = "observed"
    MISSING = "missing"


class UncertaintyModel(str, Enum):
    ABSOLUTE_BOUND = "absolute_bound"
    STANDARD_ERROR = "standard_error"
    NOT_APPLICABLE = "not_applicable"


class TransformOperation(str, Enum):
    IDENTITY = "identity"
    TEMPORAL_AGGREGATION = "temporal_aggregation"
    SPATIAL_AGGREGATION = "spatial_aggregation"
    SAMPLING_RESOLUTION = "sampling_resolution"
    UNIT_CONVERSION = "unit_conversion"
    COORDINATE_AFFINE = "coordinate_affine"
    EQUIVALENT_SPLIT_MERGE = "equivalent_split_merge"
    COARSE_GRAINING = "coarse_graining"


class PredictionDisposition(str, Enum):
    UNIQUE_MATCH = "unique_match"
    ABSTAIN = "abstain"


class PredictionReason(str, Enum):
    UNIQUE_STRUCTURAL_MATCH = "unique_structural_match"
    NO_PASSING_CANDIDATE = "no_passing_candidate"
    MULTIPLE_STRUCTURAL_MATCHES = "multiple_structural_matches"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INCOMPLETE_CANDIDATE_COVERAGE = "incomplete_candidate_coverage"
    INSUFFICIENT_MARGIN = "insufficient_margin"
    NONIDENTIFIABLE_SCALE = "nonidentifiable_scale"
    RESOURCE_LIMIT = "resource_limit"
    VERIFIER_ERROR = "verifier_error"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True, slots=True)
class NumericValue:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _number_tuple(self.values, name="numeric value"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"kind": "numeric", "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class NumericInterval:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        lower = _number_tuple(self.lower, name="interval lower")
        upper = _number_tuple(self.upper, name="interval upper")
        if len(lower) != len(upper):
            raise ValueError("interval bounds must have equal dimensions")
        if any(left > right for left, right in zip(lower, upper, strict=True)):
            raise ValueError("interval lower bound exceeds upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_mapping(self) -> dict[str, object]:
        return {
            "kind": "interval",
            "lower": list(self.lower),
            "upper": list(self.upper),
        }


@dataclass(frozen=True, slots=True)
class BooleanValue:
    value: bool

    def __post_init__(self) -> None:
        if type(self.value) is not bool:
            raise TypeError("boolean value must be a boolean")

    def to_mapping(self) -> dict[str, object]:
        return {"kind": "boolean", "value": self.value}


TypedValue: TypeAlias = NumericValue | NumericInterval | BooleanValue


def _parse_typed_value(value: object) -> TypedValue:
    mapping = _strict_mapping(
        value,
        name="typed value",
        required=frozenset({"kind"}),
        optional=frozenset({"values", "lower", "upper", "value"}),
    )
    kind = mapping["kind"]
    if kind == "numeric":
        _strict_mapping(
            mapping,
            name="numeric value",
            required=frozenset({"kind", "values"}),
        )
        return NumericValue(
            _number_tuple(mapping["values"], name="numeric values")
        )
    if kind == "interval":
        _strict_mapping(
            mapping,
            name="interval value",
            required=frozenset({"kind", "lower", "upper"}),
        )
        return NumericInterval(
            _number_tuple(mapping["lower"], name="interval lower"),
            _number_tuple(mapping["upper"], name="interval upper"),
        )
    if kind == "boolean":
        _strict_mapping(
            mapping,
            name="boolean value",
            required=frozenset({"kind", "value"}),
        )
        return BooleanValue(mapping["value"])  # type: ignore[arg-type]
    raise ValueError("typed value kind must be numeric, interval, or boolean")


def _typed_value_mapping(value: TypedValue | None) -> dict[str, object] | None:
    return value.to_mapping() if value is not None else None


@dataclass(frozen=True, slots=True)
class UnitDimension:
    """SI base-dimension exponents in L, M, T, I, Θ, N, J order."""

    si_exponents: tuple[int, int, int, int, int, int, int]

    def __post_init__(self) -> None:
        raw = _sequence(self.si_exponents, name="unit dimension exponents")
        if len(raw) != 7:
            raise ValueError("unit dimension needs seven SI base exponents")
        exponents = tuple(
            _integer(item, name="unit dimension exponent") for item in raw
        )
        if any(abs(item) > 16 for item in exponents):
            raise ValueError("unit dimension exponent is outside the bounded schema")
        object.__setattr__(self, "si_exponents", exponents)

    @classmethod
    def from_mapping(cls, value: object) -> "UnitDimension":
        mapping = _strict_mapping(
            value,
            name="unit dimension",
            required=frozenset({"si_exponents"}),
        )
        return cls(mapping["si_exponents"])  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        return {"si_exponents": list(self.si_exponents)}


@dataclass(frozen=True, slots=True)
class TemporalSupport:
    clock_id: str
    start: float
    end: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_id", _uuid4(self.clock_id, name="clock id"))
        start = _number(self.start, name="temporal support start")
        end = _number(self.end, name="temporal support end")
        if start > end:
            raise ValueError("temporal support start exceeds end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @classmethod
    def from_mapping(cls, value: object) -> "TemporalSupport":
        mapping = _strict_mapping(
            value,
            name="temporal support",
            required=frozenset({"clock_id", "start", "end"}),
        )
        return cls(
            mapping["clock_id"],  # type: ignore[arg-type]
            mapping["start"],  # type: ignore[arg-type]
            mapping["end"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {"clock_id": self.clock_id, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class SpatialSupport:
    frame_id: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _uuid4(self.frame_id, name="frame id"))
        lower = _number_tuple(self.lower, name="spatial lower")
        upper = _number_tuple(self.upper, name="spatial upper")
        if len(lower) != len(upper) or not 1 <= len(lower) <= 4:
            raise ValueError("spatial bounds need equal dimensions between one and four")
        if any(left > right for left, right in zip(lower, upper, strict=True)):
            raise ValueError("spatial lower bound exceeds upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_mapping(cls, value: object) -> "SpatialSupport":
        mapping = _strict_mapping(
            value,
            name="spatial support",
            required=frozenset({"frame_id", "lower", "upper"}),
        )
        return cls(
            mapping["frame_id"],  # type: ignore[arg-type]
            mapping["lower"],  # type: ignore[arg-type]
            mapping["upper"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "lower": list(self.lower),
            "upper": list(self.upper),
        }


@dataclass(frozen=True, slots=True)
class MeasurementUncertainty:
    model: UncertaintyModel
    radius: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, UncertaintyModel):
            raise TypeError("uncertainty model must be an UncertaintyModel")
        radius = _number_tuple(
            self.radius,
            name="uncertainty radius",
            nonempty=False,
        )
        if any(item < 0 for item in radius):
            raise ValueError("uncertainty radius cannot be negative")
        if self.model is UncertaintyModel.NOT_APPLICABLE and radius:
            raise ValueError("not-applicable uncertainty cannot carry a radius")
        if self.model is not UncertaintyModel.NOT_APPLICABLE and not radius:
            raise ValueError("numeric uncertainty requires a radius")
        object.__setattr__(self, "radius", radius)

    @classmethod
    def from_mapping(cls, value: object) -> "MeasurementUncertainty":
        mapping = _strict_mapping(
            value,
            name="measurement uncertainty",
            required=frozenset({"model", "radius"}),
        )
        try:
            model = UncertaintyModel(mapping["model"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown uncertainty model") from exc
        return cls(
            model,
            _number_tuple(
                mapping["radius"],
                name="uncertainty radius",
                nonempty=False,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {"model": self.model.value, "radius": list(self.radius)}


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    entity_id: str
    role_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _uuid4(self.entity_id, name="entity id"))
        object.__setattr__(
            self,
            "role_candidate_ids",
            _uuid_tuple(self.role_candidate_ids, name="entity role candidate ids"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "EntityCandidate":
        mapping = _strict_mapping(
            value,
            name="entity candidate",
            required=frozenset({"entity_id", "role_candidate_ids"}),
        )
        return cls(
            mapping["entity_id"],  # type: ignore[arg-type]
            _uuid_tuple(
                mapping["role_candidate_ids"],
                name="entity role candidate ids",
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "role_candidate_ids": list(self.role_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class TypedObservation:
    observation_id: str
    source_channel_id: str
    entity_ids: tuple[str, ...]
    role_candidate_ids: tuple[str, ...]
    quantity_id: str
    value: TypedValue | None
    unit_dimension: UnitDimension
    temporal_support: TemporalSupport | None
    spatial_support: SpatialSupport | None
    uncertainty: MeasurementUncertainty
    provenance_sha256: str
    missingness: Missingness

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _uuid4(self.observation_id, name="observation id"),
        )
        object.__setattr__(
            self,
            "source_channel_id",
            _uuid4(self.source_channel_id, name="source channel id"),
        )
        object.__setattr__(
            self,
            "entity_ids",
            _uuid_tuple(self.entity_ids, name="observation entity ids"),
        )
        object.__setattr__(
            self,
            "role_candidate_ids",
            _uuid_tuple(
                self.role_candidate_ids,
                name="observation role candidate ids",
            ),
        )
        object.__setattr__(
            self,
            "quantity_id",
            _uuid4(self.quantity_id, name="quantity id"),
        )
        if self.value is not None and not isinstance(
            self.value, (NumericValue, NumericInterval, BooleanValue)
        ):
            raise TypeError("observation value is not a typed wire value")
        if not isinstance(self.unit_dimension, UnitDimension):
            raise TypeError("observation unit dimension has the wrong type")
        if self.temporal_support is not None and not isinstance(
            self.temporal_support, TemporalSupport
        ):
            raise TypeError("observation temporal support has the wrong type")
        if self.spatial_support is not None and not isinstance(
            self.spatial_support, SpatialSupport
        ):
            raise TypeError("observation spatial support has the wrong type")
        if not isinstance(self.uncertainty, MeasurementUncertainty):
            raise TypeError("observation uncertainty has the wrong type")
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, name="observation provenance"),
        )
        if not isinstance(self.missingness, Missingness):
            raise TypeError("observation missingness has the wrong type")
        if (self.missingness is Missingness.MISSING) is (self.value is not None):
            raise ValueError("missingness and typed value presence disagree")

        numeric_width: int | None
        if isinstance(self.value, NumericValue):
            numeric_width = len(self.value.values)
        elif isinstance(self.value, NumericInterval):
            numeric_width = len(self.value.lower)
        else:
            numeric_width = None
        if numeric_width is not None:
            if self.uncertainty.model is UncertaintyModel.NOT_APPLICABLE:
                raise ValueError("numeric observations require numeric uncertainty")
            if len(self.uncertainty.radius) != numeric_width:
                raise ValueError("uncertainty dimension disagrees with numeric value")
        elif self.uncertainty.model is not UncertaintyModel.NOT_APPLICABLE:
            raise ValueError("boolean or missing observations need not-applicable uncertainty")

    @classmethod
    def from_mapping(cls, value: object) -> "TypedObservation":
        mapping = _strict_mapping(
            value,
            name="typed observation",
            required=frozenset(
                {
                    "observation_id",
                    "source_channel_id",
                    "entity_ids",
                    "role_candidate_ids",
                    "quantity_id",
                    "value",
                    "unit_dimension",
                    "temporal_support",
                    "spatial_support",
                    "uncertainty",
                    "provenance_sha256",
                    "missingness",
                }
            ),
        )
        try:
            missingness = Missingness(mapping["missingness"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown missingness value") from exc
        temporal = mapping["temporal_support"]
        spatial = mapping["spatial_support"]
        return cls(
            observation_id=mapping["observation_id"],  # type: ignore[arg-type]
            source_channel_id=mapping["source_channel_id"],  # type: ignore[arg-type]
            entity_ids=_uuid_tuple(
                mapping["entity_ids"], name="observation entity ids"
            ),
            role_candidate_ids=_uuid_tuple(
                mapping["role_candidate_ids"],
                name="observation role candidate ids",
            ),
            quantity_id=mapping["quantity_id"],  # type: ignore[arg-type]
            value=(
                None
                if mapping["value"] is None
                else _parse_typed_value(mapping["value"])
            ),
            unit_dimension=UnitDimension.from_mapping(mapping["unit_dimension"]),
            temporal_support=(
                None if temporal is None else TemporalSupport.from_mapping(temporal)
            ),
            spatial_support=(
                None if spatial is None else SpatialSupport.from_mapping(spatial)
            ),
            uncertainty=MeasurementUncertainty.from_mapping(mapping["uncertainty"]),
            provenance_sha256=mapping["provenance_sha256"],  # type: ignore[arg-type]
            missingness=missingness,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "source_channel_id": self.source_channel_id,
            "entity_ids": list(self.entity_ids),
            "role_candidate_ids": list(self.role_candidate_ids),
            "quantity_id": self.quantity_id,
            "value": _typed_value_mapping(self.value),
            "unit_dimension": self.unit_dimension.to_mapping(),
            "temporal_support": (
                self.temporal_support.to_mapping()
                if self.temporal_support is not None
                else None
            ),
            "spatial_support": (
                self.spatial_support.to_mapping()
                if self.spatial_support is not None
                else None
            ),
            "uncertainty": self.uncertainty.to_mapping(),
            "provenance_sha256": self.provenance_sha256,
            "missingness": self.missingness.value,
        }


@dataclass(frozen=True, slots=True)
class TaskTarget:
    task_id: str
    entity_ids: tuple[str, ...]
    quantity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _uuid4(self.task_id, name="task id"))
        object.__setattr__(
            self,
            "entity_ids",
            _uuid_tuple(self.entity_ids, name="task target entity ids"),
        )
        object.__setattr__(
            self,
            "quantity_ids",
            _uuid_tuple(self.quantity_ids, name="task target quantity ids"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "TaskTarget":
        mapping = _strict_mapping(
            value,
            name="task target",
            required=frozenset({"task_id", "entity_ids", "quantity_ids"}),
        )
        return cls(
            mapping["task_id"],  # type: ignore[arg-type]
            _uuid_tuple(mapping["entity_ids"], name="task target entity ids"),
            _uuid_tuple(mapping["quantity_ids"], name="task target quantity ids"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "entity_ids": list(self.entity_ids),
            "quantity_ids": list(self.quantity_ids),
        }


@dataclass(frozen=True, slots=True)
class TransformSpec:
    transform_id: str
    operation: TransformOperation
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_id",
            _uuid4(self.transform_id, name="transform id"),
        )
        if not isinstance(self.operation, TransformOperation):
            raise TypeError("transform operation has the wrong type")
        parameters = _number_tuple(
            self.parameters,
            name="transform parameters",
            nonempty=False,
        )
        if len(parameters) > 4:
            raise ValueError("transform parameter vector exceeds the public bound")
        if self.operation is TransformOperation.IDENTITY and parameters:
            raise ValueError("identity transform cannot carry parameters")
        object.__setattr__(self, "parameters", parameters)

    @classmethod
    def from_mapping(cls, value: object) -> "TransformSpec":
        mapping = _strict_mapping(
            value,
            name="transform specification",
            required=frozenset({"transform_id", "operation", "parameters"}),
        )
        try:
            operation = TransformOperation(mapping["operation"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown transform operation") from exc
        return cls(
            mapping["transform_id"],  # type: ignore[arg-type]
            operation,
            _number_tuple(
                mapping["parameters"],
                name="transform parameters",
                nonempty=False,
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "transform_id": self.transform_id,
            "operation": self.operation.value,
            "parameters": list(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class AggregationEdge:
    source_scale_id: str
    target_scale_id: str
    transform_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_scale_id",
            _uuid4(self.source_scale_id, name="source scale id"),
        )
        object.__setattr__(
            self,
            "target_scale_id",
            _uuid4(self.target_scale_id, name="target scale id"),
        )
        object.__setattr__(
            self,
            "transform_id",
            _uuid4(self.transform_id, name="edge transform id"),
        )
        if self.source_scale_id == self.target_scale_id:
            raise ValueError("aggregation graph cannot contain a self edge")

    @classmethod
    def from_mapping(cls, value: object) -> "AggregationEdge":
        mapping = _strict_mapping(
            value,
            name="aggregation edge",
            required=frozenset(
                {"source_scale_id", "target_scale_id", "transform_id"}
            ),
        )
        return cls(
            mapping["source_scale_id"],  # type: ignore[arg-type]
            mapping["target_scale_id"],  # type: ignore[arg-type]
            mapping["transform_id"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_scale_id": self.source_scale_id,
            "target_scale_id": self.target_scale_id,
            "transform_id": self.transform_id,
        }


@dataclass(frozen=True, slots=True)
class AggregationGraph:
    scale_ids: tuple[str, ...]
    root_scale_ids: tuple[str, ...]
    edges: tuple[AggregationEdge, ...]

    def __post_init__(self) -> None:
        scale_ids = _uuid_tuple(self.scale_ids, name="aggregation scale ids")
        root_scale_ids = _uuid_tuple(
            self.root_scale_ids,
            name="aggregation root scale ids",
        )
        if not set(root_scale_ids).issubset(scale_ids):
            raise ValueError("aggregation root references an unknown scale")
        if not isinstance(self.edges, tuple):
            raise TypeError("aggregation graph edges must be an immutable tuple")
        if any(not isinstance(edge, AggregationEdge) for edge in self.edges):
            raise TypeError("aggregation graph contains an invalid edge")
        edges = tuple(
            sorted(
                self.edges,
                key=lambda item: (
                    item.source_scale_id,
                    item.target_scale_id,
                    item.transform_id,
                ),
            )
        )
        edge_keys = tuple(
            (edge.source_scale_id, edge.target_scale_id, edge.transform_id)
            for edge in edges
        )
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("aggregation graph repeats an edge")
        if any(
            edge.source_scale_id not in scale_ids
            or edge.target_scale_id not in scale_ids
            for edge in edges
        ):
            raise ValueError("aggregation edge references an unknown scale")

        adjacency = {scale_id: [] for scale_id in scale_ids}
        for edge in edges:
            adjacency[edge.source_scale_id].append(edge.target_scale_id)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(scale_id: str) -> None:
            if scale_id in visiting:
                raise ValueError("aggregation graph must be acyclic")
            if scale_id in visited:
                return
            visiting.add(scale_id)
            for target in adjacency[scale_id]:
                visit(target)
            visiting.remove(scale_id)
            visited.add(scale_id)

        for scale_id in scale_ids:
            visit(scale_id)

        reachable = set(root_scale_ids)
        frontier = list(root_scale_ids)
        while frontier:
            source = frontier.pop()
            for target in adjacency[source]:
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if reachable != set(scale_ids):
            raise ValueError("every aggregation scale must be reachable from a root")

        object.__setattr__(self, "scale_ids", scale_ids)
        object.__setattr__(self, "root_scale_ids", root_scale_ids)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def from_mapping(cls, value: object) -> "AggregationGraph":
        mapping = _strict_mapping(
            value,
            name="aggregation graph",
            required=frozenset({"scale_ids", "root_scale_ids", "edges"}),
        )
        return cls(
            scale_ids=_uuid_tuple(
                mapping["scale_ids"], name="aggregation scale ids"
            ),
            root_scale_ids=_uuid_tuple(
                mapping["root_scale_ids"], name="aggregation root scale ids"
            ),
            edges=tuple(
                AggregationEdge.from_mapping(item)
                for item in _sequence(mapping["edges"], name="aggregation edges")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "scale_ids": list(self.scale_ids),
            "root_scale_ids": list(self.root_scale_ids),
            "edges": [edge.to_mapping() for edge in self.edges],
        }


@dataclass(frozen=True, slots=True)
class PublicEvidenceBundle:
    schema_version: str
    bundle_id: str
    entity_candidates: tuple[EntityCandidate, ...]
    role_ids: tuple[str, ...]
    quantity_ids: tuple[str, ...]
    observations: tuple[TypedObservation, ...]
    task_target: TaskTarget
    aggregation_graph: AggregationGraph
    transform_catalog: tuple[TransformSpec, ...]
    missingness_mask: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PUBLIC_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported Phase-2B public evidence schema")
        object.__setattr__(self, "bundle_id", _uuid4(self.bundle_id, name="bundle id"))
        for name, expected_type in (
            ("entity_candidates", EntityCandidate),
            ("observations", TypedObservation),
            ("transform_catalog", TransformSpec),
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise TypeError(f"bundle {name} must be an immutable tuple")
            if not value or any(not isinstance(item, expected_type) for item in value):
                raise TypeError(f"bundle {name} contains an invalid item")

        entities = tuple(
            sorted(self.entity_candidates, key=lambda item: item.entity_id)
        )
        entity_ids = tuple(item.entity_id for item in entities)
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("public evidence repeats an entity id")
        role_ids = _uuid_tuple(self.role_ids, name="bundle role ids")
        quantity_ids = _uuid_tuple(self.quantity_ids, name="bundle quantity ids")
        observations = tuple(
            sorted(self.observations, key=lambda item: item.observation_id)
        )
        observation_ids = tuple(item.observation_id for item in observations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("public evidence repeats an observation id")
        transforms = tuple(
            sorted(self.transform_catalog, key=lambda item: item.transform_id)
        )
        transform_ids = tuple(item.transform_id for item in transforms)
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("public evidence repeats a transform id")

        if any(
            not set(entity.role_candidate_ids).issubset(role_ids)
            for entity in entities
        ):
            raise ValueError("entity candidate references an unknown role id")
        for observation in observations:
            if not set(observation.entity_ids).issubset(entity_ids):
                raise ValueError("observation references an unknown entity id")
            if not set(observation.role_candidate_ids).issubset(role_ids):
                raise ValueError("observation references an unknown role id")
            if observation.quantity_id not in quantity_ids:
                raise ValueError("observation references an unknown quantity id")
        if not isinstance(self.task_target, TaskTarget):
            raise TypeError("bundle task target has the wrong type")
        if not set(self.task_target.entity_ids).issubset(entity_ids):
            raise ValueError("task target references an unknown entity id")
        if not set(self.task_target.quantity_ids).issubset(quantity_ids):
            raise ValueError("task target references an unknown quantity id")
        if not isinstance(self.aggregation_graph, AggregationGraph):
            raise TypeError("bundle aggregation graph has the wrong type")
        if any(
            edge.transform_id not in transform_ids
            for edge in self.aggregation_graph.edges
        ):
            raise ValueError("aggregation edge references an unknown transform id")
        missingness_mask = _uuid_tuple(
            self.missingness_mask,
            name="bundle missingness mask",
            nonempty=False,
        )
        if not set(missingness_mask).issubset(observation_ids):
            raise ValueError("missingness mask references an unknown observation")
        expected_missing = {
            observation.observation_id
            for observation in observations
            if observation.missingness is Missingness.MISSING
        }
        if set(missingness_mask) != expected_missing:
            raise ValueError("missingness mask disagrees with observation missingness")

        object.__setattr__(self, "entity_candidates", entities)
        object.__setattr__(self, "role_ids", role_ids)
        object.__setattr__(self, "quantity_ids", quantity_ids)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "transform_catalog", transforms)
        object.__setattr__(self, "missingness_mask", missingness_mask)

    @classmethod
    def from_mapping(cls, value: object) -> "PublicEvidenceBundle":
        mapping = _strict_mapping(
            value,
            name="public evidence bundle",
            required=frozenset(
                {
                    "schema_version",
                    "bundle_id",
                    "entity_candidates",
                    "role_ids",
                    "quantity_ids",
                    "observations",
                    "task_target",
                    "aggregation_graph",
                    "transform_catalog",
                    "missingness_mask",
                }
            ),
        )
        return cls(
            schema_version=mapping["schema_version"],  # type: ignore[arg-type]
            bundle_id=mapping["bundle_id"],  # type: ignore[arg-type]
            entity_candidates=tuple(
                EntityCandidate.from_mapping(item)
                for item in _sequence(
                    mapping["entity_candidates"], name="entity candidates"
                )
            ),
            role_ids=_uuid_tuple(mapping["role_ids"], name="bundle role ids"),
            quantity_ids=_uuid_tuple(
                mapping["quantity_ids"], name="bundle quantity ids"
            ),
            observations=tuple(
                TypedObservation.from_mapping(item)
                for item in _sequence(mapping["observations"], name="observations")
            ),
            task_target=TaskTarget.from_mapping(mapping["task_target"]),
            aggregation_graph=AggregationGraph.from_mapping(
                mapping["aggregation_graph"]
            ),
            transform_catalog=tuple(
                TransformSpec.from_mapping(item)
                for item in _sequence(
                    mapping["transform_catalog"], name="transform catalog"
                )
            ),
            missingness_mask=_uuid_tuple(
                mapping["missingness_mask"],
                name="bundle missingness mask",
                nonempty=False,
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "PublicEvidenceBundle":
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("public evidence must be valid JSON") from exc
        return cls.from_mapping(decoded)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "entity_candidates": [
                entity.to_mapping() for entity in self.entity_candidates
            ],
            "role_ids": list(self.role_ids),
            "quantity_ids": list(self.quantity_ids),
            "observations": [
                observation.to_mapping() for observation in self.observations
            ],
            "task_target": self.task_target.to_mapping(),
            "aggregation_graph": self.aggregation_graph.to_mapping(),
            "transform_catalog": [
                transform.to_mapping() for transform in self.transform_catalog
            ],
            "missingness_mask": list(self.missingness_mask),
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_mapping())

    @property
    def content_id(self) -> str:
        return stable_hash(self.to_mapping(), prefix="phase2b_evidence_")


@dataclass(frozen=True, slots=True)
class RoleBinding:
    role_id: str
    entity_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_id", _uuid4(self.role_id, name="binding role id"))
        object.__setattr__(
            self,
            "entity_id",
            _uuid4(self.entity_id, name="binding entity id"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "RoleBinding":
        mapping = _strict_mapping(
            value,
            name="role binding",
            required=frozenset({"role_id", "entity_id"}),
        )
        return cls(
            mapping["role_id"],  # type: ignore[arg-type]
            mapping["entity_id"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {"role_id": self.role_id, "entity_id": self.entity_id}


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    schema_version: str
    bundle_id: str
    input_root_sha256: str
    protocol_sha256: str
    freeze_manifest_sha256: str
    disposition: PredictionDisposition
    reason: PredictionReason
    family_id: str | None
    binding: tuple[RoleBinding, ...]
    admissible_scale_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError("unsupported Phase-2B prediction schema")
        object.__setattr__(self, "bundle_id", _uuid4(self.bundle_id, name="bundle id"))
        for name in (
            "input_root_sha256",
            "protocol_sha256",
            "freeze_manifest_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if not isinstance(self.disposition, PredictionDisposition):
            raise TypeError("prediction disposition has the wrong type")
        if not isinstance(self.reason, PredictionReason):
            raise TypeError("prediction reason has the wrong type")
        if not isinstance(self.binding, tuple) or any(
            not isinstance(item, RoleBinding) for item in self.binding
        ):
            raise TypeError("prediction binding must be an immutable RoleBinding tuple")
        binding = tuple(sorted(self.binding, key=lambda item: item.role_id))
        role_ids = tuple(item.role_id for item in binding)
        entity_ids = tuple(item.entity_id for item in binding)
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("prediction binding repeats a role")
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("prediction binding repeats an entity")
        scales = _uuid_tuple(
            self.admissible_scale_ids,
            name="admissible scale ids",
            nonempty=False,
        )

        if self.disposition is PredictionDisposition.UNIQUE_MATCH:
            family_id = _uuid4(self.family_id, name="predicted family id")
            if not binding or not scales:
                raise ValueError(
                    "a unique match needs one family, a binding, and an admissible scale set"
                )
            if self.reason is not PredictionReason.UNIQUE_STRUCTURAL_MATCH:
                raise ValueError("unique prediction has an incompatible reason")
            object.__setattr__(self, "family_id", family_id)
        else:
            if self.family_id is not None or binding or scales:
                raise ValueError("an abstention cannot carry family, binding, or scales")
            if self.reason is PredictionReason.UNIQUE_STRUCTURAL_MATCH:
                raise ValueError("abstention has an incompatible reason")

        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "admissible_scale_ids", scales)

    @classmethod
    def from_mapping(cls, value: object) -> "PredictionBundle":
        mapping = _strict_mapping(
            value,
            name="prediction bundle",
            required=frozenset(
                {
                    "schema_version",
                    "bundle_id",
                    "input_root_sha256",
                    "protocol_sha256",
                    "freeze_manifest_sha256",
                    "disposition",
                    "reason",
                    "family_id",
                    "binding",
                    "admissible_scale_ids",
                }
            ),
        )
        try:
            disposition = PredictionDisposition(mapping["disposition"])
            reason = PredictionReason(mapping["reason"])
        except (TypeError, ValueError) as exc:
            raise ValueError("prediction disposition or reason is unknown") from exc
        return cls(
            schema_version=mapping["schema_version"],  # type: ignore[arg-type]
            bundle_id=mapping["bundle_id"],  # type: ignore[arg-type]
            input_root_sha256=mapping["input_root_sha256"],  # type: ignore[arg-type]
            protocol_sha256=mapping["protocol_sha256"],  # type: ignore[arg-type]
            freeze_manifest_sha256=mapping["freeze_manifest_sha256"],  # type: ignore[arg-type]
            disposition=disposition,
            reason=reason,
            family_id=mapping["family_id"],  # type: ignore[arg-type]
            binding=tuple(
                RoleBinding.from_mapping(item)
                for item in _sequence(mapping["binding"], name="prediction binding")
            ),
            admissible_scale_ids=_uuid_tuple(
                mapping["admissible_scale_ids"],
                name="admissible scale ids",
                nonempty=False,
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "PredictionBundle":
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("prediction bundle must be valid JSON") from exc
        return cls.from_mapping(decoded)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "input_root_sha256": self.input_root_sha256,
            "protocol_sha256": self.protocol_sha256,
            "freeze_manifest_sha256": self.freeze_manifest_sha256,
            "disposition": self.disposition.value,
            "reason": self.reason.value,
            "family_id": self.family_id,
            "binding": [item.to_mapping() for item in self.binding],
            "admissible_scale_ids": list(self.admissible_scale_ids),
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_mapping())

    @property
    def content_id(self) -> str:
        return stable_hash(self.to_mapping(), prefix="phase2b_prediction_")


__all__ = [
    "AggregationEdge",
    "AggregationGraph",
    "BooleanValue",
    "EntityCandidate",
    "MeasurementUncertainty",
    "Missingness",
    "NumericInterval",
    "NumericValue",
    "PREDICTION_SCHEMA_VERSION",
    "PUBLIC_EVIDENCE_SCHEMA_VERSION",
    "PredictionBundle",
    "PredictionDisposition",
    "PredictionReason",
    "PublicEvidenceBundle",
    "RoleBinding",
    "SpatialSupport",
    "TaskTarget",
    "TemporalSupport",
    "TransformOperation",
    "TransformSpec",
    "TypedObservation",
    "TypedValue",
    "UncertaintyModel",
    "UnitDimension",
]

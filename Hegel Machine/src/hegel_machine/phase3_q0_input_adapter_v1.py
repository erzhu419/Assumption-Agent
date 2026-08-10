"""Target-independent typed-input adapter for Phase-3A quotient replay.

The quotient evaluator needs an observation boundary that does not learn the
odd or sink answer tables.  This module therefore decodes only the two frozen
input schemas and exposes only observations that the old DSL may read.  It
deliberately has no dependency on a target, truth table, split, static-basis
fixture, or the legacy reference evaluator.

All missing observations are represented by one strict bottom value.  Missing
bits, measurements, contexts, tasks, scopes, or orientations are never filled
with ``0`` or ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Final, Sequence, TypeAlias

from .strict_cbor_v1 import canonical_cbor_decode


ADAPTER_SCHEMA_ID: Final = "hegel-phase3-q0-input-adapter/1"
DSL_VERSION: Final = "hegel-old-dsl-v1.6.0"

ODD_INPUT_SIGNATURE_ID: Final = 1
SINK_INPUT_SIGNATURE_ID: Final = 2
ODD_INPUT_TAG: Final = 0x3401
SINK_INPUT_TAG: Final = 0x3402
ODD_INPUT_SCHEMA_ID: Final = b"hegel-odd-input/1"
SINK_INPUT_SCHEMA_ID: Final = b"hegel-sink-input/1"

LEAF_SCALAR_CONST_ID: Final = 0
LEAF_BIT_AT_ID: Final = 1
LEAF_SET_SIZE_ID: Final = 2
LEAF_AGGREGATE_ID: Final = 3
LEAF_CONTEXT_FLAG_ID: Final = 4
LEAF_TASK_FLAG_ID: Final = 5
LEAF_NEW_SYMBOL_ID: Final = 6

AGGREGATE_SUM_ID: Final = 0
AGGREGATE_COUNT_NONZERO_ID: Final = 1
AGGREGATE_SIGNED_BALANCE_ID: Final = 5
ACTIVE_AGGREGATE_MAP_IDS: Final = (0, 1, 5)
TOMBSTONED_AGGREGATE_MAP_IDS: Final = (2, 3, 4)

SCOPE_ALL_OBSERVED_ID: Final = 0
SCOPE_PRIMARY_ONLY_ID: Final = 1
SCOPE_BOUNDARY_ONLY_ID: Final = 2
SCOPE_CONTROL_VOLUME_ALL_OBSERVED_ID: Final = 3
SCOPE_IDS: Final = (0, 1, 2, 3)

QUANTITY_Q0_ID: Final = 0
QUANTITY_Q1_ID: Final = 1
QUANTITY_IDS: Final = (0, 1)
CONTEXT_IDS: Final = (0, 1, 2, 3)
TASK_IDS: Final = (0, 1)

ACTIVE_RATIONAL_PARAMETER_IDS: Final = (1, 3, 5)
TOMBSTONED_RATIONAL_PARAMETER_IDS: Final = (0, 2, 4, 6)
RATIONAL_PARAMETER_VALUES: Final = (
    Fraction(-2, 1),
    Fraction(-1, 1),
    Fraction(-1, 2),
    Fraction(0, 1),
    Fraction(1, 2),
    Fraction(1, 1),
    Fraction(2, 1),
)

RATIONAL_VALUE_GRID: Final = frozenset(
    Fraction(numerator, denominator)
    for numerator in range(-64, 65)
    for denominator in range(1, 9)
)

REJECT_TYPED_INPUT_PREFIX: Final = "REJECT_TYPED_INPUT_PREFIX"
REJECT_ODD_SET_SIZE: Final = "REJECT_ODD_SET_SIZE"
REJECT_ODD_BIT_COUNT: Final = "REJECT_ODD_BIT_COUNT"
REJECT_ODD_BIT_TYPE: Final = "REJECT_ODD_BIT_TYPE"
REJECT_SINK_VALUE: Final = "REJECT_SINK_VALUE"
REJECT_SINK_BALANCE: Final = "REJECT_SINK_BALANCE"
REJECT_MALFORMED_CANONICAL_LEAF: Final = "REJECT_MALFORMED_CANONICAL_LEAF"
REJECT_REGISTRY_INDEX_OUT_OF_RANGE: Final = "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"
REJECT_REMOVED_AGGREGATE_MAP: Final = "REJECT_REMOVED_AGGREGATE_MAP"
REJECT_REMOVED_RATIONAL_PARAMETER: Final = "REJECT_REMOVED_RATIONAL_PARAMETER"
REJECT_NEW_SYMBOL_IN_OLD_DSL: Final = "REJECT_NEW_SYMBOL_IN_OLD_DSL"


class InputAdapterError(ValueError):
    """Stable fail-closed rejection from the Q0 observation boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _reject(code: str, detail: str) -> "None":
    raise InputAdapterError(code, detail)


class BottomValueV1(str, Enum):
    """The single non-observable result for absent or undefined data."""

    BOTTOM = "⊥"


BOTTOM: Final = BottomValueV1.BOTTOM
BOTTOM_V1: Final = BOTTOM

BitValue: TypeAlias = int | BottomValueV1
RationalValue: TypeAlias = Fraction | BottomValueV1
BoundedIntValue: TypeAlias = int | BottomValueV1
BoolValue: TypeAlias = bool | BottomValueV1
LeafValue: TypeAlias = Fraction | int | bool | BottomValueV1


@dataclass(frozen=True, slots=True)
class Entity:
    """One immutable entity projection exposed to old-DSL leaves."""

    slot_index: int
    bit: BitValue
    role_id: int | BottomValueV1
    orientation: int | BottomValueV1
    quantities: tuple[RationalValue, RationalValue]
    scope_membership: tuple[BoolValue, BoolValue, BoolValue, BoolValue]

    def quantity(self, quantity_id: int) -> RationalValue:
        _require_registry_id(quantity_id, QUANTITY_IDS, "QuantityId")
        return self.quantities[quantity_id]

    def membership(self, scope_id: int) -> BoolValue:
        _require_registry_id(scope_id, SCOPE_IDS, "ScopeId")
        return self.scope_membership[scope_id]


@dataclass(frozen=True, slots=True)
class ObservationEnvironment:
    """Immutable observations for one canonical typed input object."""

    input_signature_id: int
    input_object_tag: int
    canonical_input_object: tuple[object, ...]
    set_size: int
    entities: tuple[Entity, ...]
    context_flags: tuple[BoolValue, BoolValue, BoolValue, BoolValue]
    task_flags: tuple[BoolValue, BoolValue]

    def bit_at(self, slot_index: int) -> BitValue:
        _require_registry_id(slot_index, tuple(range(8)), "EntitySlotId")
        if slot_index >= self.set_size:
            return BOTTOM
        return self.entities[slot_index].bit

    def context_flag(self, context_id: int) -> BoolValue:
        _require_registry_id(context_id, CONTEXT_IDS, "ContextId")
        return self.context_flags[context_id]

    def task_flag(self, task_id: int) -> BoolValue:
        _require_registry_id(task_id, TASK_IDS, "TaskId")
        return self.task_flags[task_id]


# Versioned aliases make the intended downstream API explicit without making
# the class names cumbersome at every evaluator call site.
EntityObservationV1 = Entity
ObservationEnvironmentV1 = ObservationEnvironment


def _as_array(value: object, *, code: str, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        _reject(code, f"{name} must be a CBOR array")
    return tuple(value)


def _require_registry_id(value: object, registry: tuple[int, ...], name: str) -> int:
    if type(value) is not int or value not in registry:
        _reject(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            f"{name} {value!r} is outside its frozen registry",
        )
    return value


def parse_odd_input_object_v1(value: object) -> tuple[object, ...]:
    """Validate and freeze one decoded ``OddInputV1`` object."""

    array = _as_array(value, code=REJECT_TYPED_INPUT_PREFIX, name="OddInputV1")
    if (
        len(array) != 5
        or type(array[0]) is not int
        or array[0] != 1
        or type(array[1]) is not int
        or array[1] != ODD_INPUT_TAG
        or type(array[2]) is not bytes
        or array[2] != ODD_INPUT_SCHEMA_ID
    ):
        _reject(REJECT_TYPED_INPUT_PREFIX, "OddInputV1 prefix or arity mismatch")
    set_size = array[3]
    if type(set_size) is not int or set_size not in (5, 6, 7, 8):
        _reject(REJECT_ODD_SET_SIZE, "odd set_size must be one of 5, 6, 7, 8")
    bits = _as_array(array[4], code=REJECT_ODD_BIT_COUNT, name="odd bits")
    if len(bits) != set_size:
        _reject(REJECT_ODD_BIT_COUNT, "odd bit count must equal set_size")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        _reject(REJECT_ODD_BIT_TYPE, "odd bits must be CBOR uint 0 or 1")
    return (1, ODD_INPUT_TAG, ODD_INPUT_SCHEMA_ID, set_size, bits)


def parse_sink_input_object_v1(value: object) -> tuple[object, ...]:
    """Validate and freeze one decoded ``SinkInputV1`` object."""

    array = _as_array(value, code=REJECT_TYPED_INPUT_PREFIX, name="SinkInputV1")
    if (
        len(array) != 7
        or type(array[0]) is not int
        or array[0] != 1
        or type(array[1]) is not int
        or array[1] != SINK_INPUT_TAG
        or type(array[2]) is not bytes
        or array[2] != SINK_INPUT_SCHEMA_ID
    ):
        _reject(REJECT_TYPED_INPUT_PREFIX, "SinkInputV1 prefix or arity mismatch")
    values = array[3:]
    if any(type(item) is not int or not 0 <= item <= 4 for item in values):
        _reject(REJECT_SINK_VALUE, "sink a, b, c, d must be CBOR uint in [0, 4]")
    a, b, c, d = values
    if d != a + b - c:
        _reject(REJECT_SINK_BALANCE, "sink input must satisfy d = a + b - c")
    return (1, SINK_INPUT_TAG, SINK_INPUT_SCHEMA_ID, a, b, c, d)


def _odd_environment(value: object) -> ObservationEnvironment:
    canonical = parse_odd_input_object_v1(value)
    set_size = canonical[3]
    bits = canonical[4]
    assert type(set_size) is int and isinstance(bits, tuple)
    entities = tuple(
        Entity(
            slot_index=index,
            bit=bit,
            role_id=BOTTOM,
            orientation=BOTTOM,
            quantities=(BOTTOM, BOTTOM),
            scope_membership=(BOTTOM, BOTTOM, BOTTOM, BOTTOM),
        )
        for index, bit in enumerate(bits)
    )
    return ObservationEnvironment(
        input_signature_id=ODD_INPUT_SIGNATURE_ID,
        input_object_tag=ODD_INPUT_TAG,
        canonical_input_object=canonical,
        set_size=set_size,
        entities=entities,
        context_flags=(BOTTOM, BOTTOM, BOTTOM, BOTTOM),
        task_flags=(BOTTOM, BOTTOM),
    )


def _sink_environment(value: object) -> ObservationEnvironment:
    canonical = parse_sink_input_object_v1(value)
    measurements = canonical[3:]
    orientations = (1, 1, -1, -1)
    entities = tuple(
        Entity(
            slot_index=index,
            bit=BOTTOM,
            role_id=index,
            orientation=orientations[index],
            quantities=(Fraction(measurement, 1), BOTTOM),
            scope_membership=(BOTTOM, BOTTOM, BOTTOM, True),
        )
        for index, measurement in enumerate(measurements)
    )
    return ObservationEnvironment(
        input_signature_id=SINK_INPUT_SIGNATURE_ID,
        input_object_tag=SINK_INPUT_TAG,
        canonical_input_object=canonical,
        set_size=4,
        entities=entities,
        context_flags=(BOTTOM, BOTTOM, BOTTOM, BOTTOM),
        task_flags=(BOTTOM, BOTTOM),
    )


def observation_environment_from_object_v1(value: object) -> ObservationEnvironment:
    """Dispatch one validated typed input to its observation environment."""

    array = _as_array(value, code=REJECT_TYPED_INPUT_PREFIX, name="typed input")
    if len(array) < 2 or type(array[1]) is not int:
        _reject(REJECT_TYPED_INPUT_PREFIX, "typed input has no numeric tag")
    if array[1] == ODD_INPUT_TAG:
        return _odd_environment(array)
    if array[1] == SINK_INPUT_TAG:
        return _sink_environment(array)
    _reject(REJECT_TYPED_INPUT_PREFIX, "unknown typed input tag")


def decode_observation_environment_v1(payload: bytes) -> ObservationEnvironment:
    """Decode strict canonical CBOR and construct an immutable environment."""

    return observation_environment_from_object_v1(canonical_cbor_decode(payload))


def _active_aggregate_map_id(map_id: object) -> int:
    if type(map_id) is not int or map_id < 0 or map_id > 5:
        _reject(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            f"AggregateMapId {map_id!r} is outside allocated IDs 0..5",
        )
    if map_id in TOMBSTONED_AGGREGATE_MAP_IDS:
        _reject(
            REJECT_REMOVED_AGGREGATE_MAP,
            f"AggregateMapId {map_id} is tombstoned in {DSL_VERSION}",
        )
    assert map_id in ACTIVE_AGGREGATE_MAP_IDS
    return map_id


def _active_rational_parameter_id(parameter_id: object) -> int:
    if type(parameter_id) is not int or parameter_id < 0 or parameter_id > 6:
        _reject(
            REJECT_REGISTRY_INDEX_OUT_OF_RANGE,
            f"RationalParameterId {parameter_id!r} is outside allocated IDs 0..6",
        )
    if parameter_id in TOMBSTONED_RATIONAL_PARAMETER_IDS:
        _reject(
            REJECT_REMOVED_RATIONAL_PARAMETER,
            f"RationalParameterId {parameter_id} is tombstoned in {DSL_VERSION}",
        )
    assert parameter_id in ACTIVE_RATIONAL_PARAMETER_IDS
    return parameter_id


def _validate_scope_extension(value: object) -> tuple[tuple[int, bool], ...]:
    extension = _as_array(
        value,
        code=REJECT_MALFORMED_CANONICAL_LEAF,
        name="scope extension",
    )
    clauses: list[tuple[int, bool]] = []
    for raw_clause in extension:
        clause = _as_array(
            raw_clause,
            code=REJECT_MALFORMED_CANONICAL_LEAF,
            name="scope clause",
        )
        if len(clause) != 2:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "scope clause arity must be two")
        context_id = _require_registry_id(clause[0], CONTEXT_IDS, "ContextId")
        if type(clause[1]) is not bool:
            _reject(
                REJECT_MALFORMED_CANONICAL_LEAF,
                "scope expectation must be a CBOR boolean",
            )
        clauses.append((context_id, clause[1]))
    normalized = tuple(clauses)
    if (
        len(normalized) > 2
        or normalized != tuple(sorted(normalized))
        or len({context_id for context_id, _ in normalized}) != len(normalized)
    ):
        _reject(
            REJECT_MALFORMED_CANONICAL_LEAF,
            "scope clauses must be unique, sorted, and have arity at most two",
        )
    return normalized


def evaluate_aggregate_values_v1(
    map_id: int,
    values: Sequence[RationalValue],
    *,
    orientations: Sequence[int | BottomValueV1] | None = None,
) -> RationalValue | BoundedIntValue:
    """Evaluate one active map with exact arithmetic and strict bottom."""

    active_map = _active_aggregate_map_id(map_id)
    if not isinstance(values, (tuple, list)):
        raise TypeError("aggregate values must be an array")
    frozen_values = tuple(values)
    if any(value is BOTTOM for value in frozen_values):
        return BOTTOM
    if any(type(value) is not Fraction for value in frozen_values):
        raise TypeError("aggregate values must be exact Fractions or bottom")
    exact_values = frozen_values
    if any(value not in RATIONAL_VALUE_GRID for value in exact_values):
        return BOTTOM

    if active_map == AGGREGATE_COUNT_NONZERO_ID:
        count = sum(value != 0 for value in exact_values)
        return count if -8 <= count <= 8 else BOTTOM
    if active_map == AGGREGATE_SUM_ID:
        result = sum(exact_values, Fraction(0, 1))
        return result if result in RATIONAL_VALUE_GRID else BOTTOM

    if orientations is None or not isinstance(orientations, (tuple, list)):
        return BOTTOM
    frozen_orientations = tuple(orientations)
    if len(frozen_orientations) != len(exact_values):
        return BOTTOM
    if any(orientation is BOTTOM for orientation in frozen_orientations):
        return BOTTOM
    if any(
        type(orientation) is not int or orientation not in (-1, 1)
        for orientation in frozen_orientations
    ):
        raise TypeError("signed-balance orientations must be -1, +1, or bottom")
    result = sum(
        (
            orientation * value
            for orientation, value in zip(frozen_orientations, exact_values)
        ),
        Fraction(0, 1),
    )
    return result if result in RATIONAL_VALUE_GRID else BOTTOM


def evaluate_environment_aggregate_v1(
    environment: ObservationEnvironment,
    map_id: int,
    scope_id: int,
    quantity_id: int,
    scope_extension: object,
) -> RationalValue | BoundedIntValue:
    """Evaluate an aggregate leaf without inventing unavailable metadata."""

    _active_aggregate_map_id(map_id)
    _require_registry_id(scope_id, SCOPE_IDS, "ScopeId")
    _require_registry_id(quantity_id, QUANTITY_IDS, "QuantityId")
    extension = _validate_scope_extension(scope_extension)

    # Odd inputs publish no aggregate metadata.  Sink inputs publish only q0
    # over control_volume_all_observed with an empty extension.
    if environment.input_signature_id != SINK_INPUT_SIGNATURE_ID:
        return BOTTOM
    if (
        quantity_id != QUANTITY_Q0_ID
        or scope_id != SCOPE_CONTROL_VOLUME_ALL_OBSERVED_ID
        or extension
    ):
        return BOTTOM

    values: list[RationalValue] = []
    orientations: list[int | BottomValueV1] = []
    for entity in environment.entities:
        membership = entity.membership(scope_id)
        if membership is BOTTOM:
            return BOTTOM
        if type(membership) is not bool:
            raise TypeError("scope membership must be bool or bottom")
        if membership:
            values.append(entity.quantity(quantity_id))
            orientations.append(entity.orientation)
    return evaluate_aggregate_values_v1(
        map_id,
        tuple(values),
        orientations=tuple(orientations),
    )


def _leaf_parameters(value: object) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        _reject(REJECT_MALFORMED_CANONICAL_LEAF, "leaf parameters must be an array")
    return tuple(value)


def evaluate_leaf_v1(
    environment: ObservationEnvironment,
    leaf_operator_id: int,
    parameters: Sequence[object] = (),
) -> LeafValue:
    """Evaluate one numeric-tag old-DSL leaf against an observation environment."""

    if not isinstance(environment, ObservationEnvironment):
        raise TypeError("environment must be an ObservationEnvironment")
    if type(leaf_operator_id) is not int or not 0 <= leaf_operator_id <= 6:
        _reject(REJECT_REGISTRY_INDEX_OUT_OF_RANGE, "unknown old-DSL leaf ID")
    parts = _leaf_parameters(parameters)

    if leaf_operator_id == LEAF_SCALAR_CONST_ID:
        if len(parts) != 1:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "scalar_const arity must be one")
        parameter_id = _active_rational_parameter_id(parts[0])
        return RATIONAL_PARAMETER_VALUES[parameter_id]
    if leaf_operator_id == LEAF_BIT_AT_ID:
        if len(parts) != 1:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "bit_at arity must be one")
        return environment.bit_at(parts[0])
    if leaf_operator_id == LEAF_SET_SIZE_ID:
        if parts:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "set_size arity must be zero")
        return environment.set_size
    if leaf_operator_id == LEAF_AGGREGATE_ID:
        if len(parts) != 4:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "aggregate arity must be four")
        map_id, scope_id, quantity_id, extension = parts
        return evaluate_environment_aggregate_v1(
            environment,
            map_id,  # type: ignore[arg-type]
            scope_id,  # type: ignore[arg-type]
            quantity_id,  # type: ignore[arg-type]
            extension,
        )
    if leaf_operator_id == LEAF_CONTEXT_FLAG_ID:
        if len(parts) != 1:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "context_flag arity must be one")
        return environment.context_flag(parts[0])
    if leaf_operator_id == LEAF_TASK_FLAG_ID:
        if len(parts) != 1:
            _reject(REJECT_MALFORMED_CANONICAL_LEAF, "task_flag arity must be one")
        return environment.task_flag(parts[0])
    _reject(REJECT_NEW_SYMBOL_IN_OLD_DSL, "new symbols are not old-DSL leaves")


def evaluate_canonical_leaf_v1(
    environment: ObservationEnvironment,
    canonical_leaf: object,
) -> LeafValue:
    """Evaluate an exact canonical leaf node ``(0, LeafId, ...parameters)``."""

    if not isinstance(canonical_leaf, tuple):
        _reject(REJECT_MALFORMED_CANONICAL_LEAF, "canonical leaf must be a CBOR array")
    if len(canonical_leaf) < 2 or type(canonical_leaf[0]) is not int or canonical_leaf[0] != 0:
        _reject(REJECT_MALFORMED_CANONICAL_LEAF, "canonical leaf tag must be zero")
    return evaluate_leaf_v1(environment, canonical_leaf[1], canonical_leaf[2:])


__all__ = [
    "ACTIVE_AGGREGATE_MAP_IDS",
    "ACTIVE_RATIONAL_PARAMETER_IDS",
    "ADAPTER_SCHEMA_ID",
    "AGGREGATE_COUNT_NONZERO_ID",
    "AGGREGATE_SIGNED_BALANCE_ID",
    "AGGREGATE_SUM_ID",
    "BOTTOM",
    "BOTTOM_V1",
    "BottomValueV1",
    "DSL_VERSION",
    "Entity",
    "EntityObservationV1",
    "InputAdapterError",
    "LEAF_AGGREGATE_ID",
    "LEAF_BIT_AT_ID",
    "LEAF_CONTEXT_FLAG_ID",
    "LEAF_SCALAR_CONST_ID",
    "LEAF_SET_SIZE_ID",
    "LEAF_TASK_FLAG_ID",
    "ObservationEnvironment",
    "ObservationEnvironmentV1",
    "ODD_INPUT_SCHEMA_ID",
    "ODD_INPUT_SIGNATURE_ID",
    "ODD_INPUT_TAG",
    "QUANTITY_Q0_ID",
    "QUANTITY_Q1_ID",
    "RATIONAL_VALUE_GRID",
    "SCOPE_ALL_OBSERVED_ID",
    "SCOPE_BOUNDARY_ONLY_ID",
    "SCOPE_CONTROL_VOLUME_ALL_OBSERVED_ID",
    "SCOPE_PRIMARY_ONLY_ID",
    "SINK_INPUT_SCHEMA_ID",
    "SINK_INPUT_SIGNATURE_ID",
    "SINK_INPUT_TAG",
    "decode_observation_environment_v1",
    "evaluate_aggregate_values_v1",
    "evaluate_canonical_leaf_v1",
    "evaluate_environment_aggregate_v1",
    "evaluate_leaf_v1",
    "observation_environment_from_object_v1",
    "parse_odd_input_object_v1",
    "parse_sink_input_object_v1",
]

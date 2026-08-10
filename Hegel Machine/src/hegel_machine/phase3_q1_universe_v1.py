"""Target-blind production universes for Phase-3A-Q1.

Q1 closes the old-DSL quotient separately for each frozen input signature.
This module reconstructs only the canonical input rows.  It deliberately has
no target rule, truth-row constructor, split assignment, role matcher, or
dependency on the historical module that generated inputs and truth together.

The two RFC6962 roots are historical payload identities.  Reproducing them
does not authorize reading the corresponding truth roots in Q1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from .phase3_q0_input_adapter_v1 import (
    InputAdapterError,
    ODD_INPUT_SCHEMA_ID,
    ODD_INPUT_SIGNATURE_ID,
    ODD_INPUT_TAG,
    SINK_INPUT_SCHEMA_ID,
    SINK_INPUT_SIGNATURE_ID,
    SINK_INPUT_TAG,
    observation_environment_from_object_v1,
    parse_odd_input_object_v1,
    parse_sink_input_object_v1,
)
from .strict_cbor_v1 import canonical_cbor_encode, rfc6962_root


Q1_UNIVERSE_SCHEMA_ID: Final = "hegel-phase3a-q1-production-universes/1"
UNIVERSE_ROW_TAG: Final = 0x3201
UNIVERSE_ROW_SCHEMA_ID: Final = b"hegel-bounded-universe-row/1"

ODD_UNIVERSE_ROW_COUNT: Final = 480
SINK_UNIVERSE_ROW_COUNT: Final = 85
ODD_UNIVERSE_ROOT: Final = bytes.fromhex(
    "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05"
)
SINK_UNIVERSE_ROOT: Final = bytes.fromhex(
    "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5"
)

REJECT_Q1_UNIVERSE_SIGNATURE: Final = "REJECT_Q1_UNIVERSE_SIGNATURE"
FAIL_Q1_UNIVERSE_ROW_COUNT: Final = "FAIL_Q1_UNIVERSE_ROW_COUNT"
FAIL_Q1_UNIVERSE_ROW_ORDER: Final = "FAIL_Q1_UNIVERSE_ROW_ORDER"
FAIL_Q1_UNIVERSE_ROOT: Final = "FAIL_Q1_UNIVERSE_ROOT"


class Q1UniverseError(ValueError):
    """Stable fail-closed error at the target-blind Q1 universe boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> "None":
    raise Q1UniverseError(code, detail)


def _odd_input_objects_v1() -> Iterable[tuple[object, ...]]:
    for set_size in range(5, 9):
        for numeric_value in range(1 << set_size):
            bits = tuple(
                (numeric_value >> (set_size - 1 - offset)) & 1
                for offset in range(set_size)
            )
            yield parse_odd_input_object_v1(
                (1, ODD_INPUT_TAG, ODD_INPUT_SCHEMA_ID, set_size, bits)
            )


def _sink_input_objects_v1() -> Iterable[tuple[object, ...]]:
    for inflow_a in range(5):
        for inflow_b in range(5):
            for primary_outflow in range(5):
                auxiliary_outflow = inflow_a + inflow_b - primary_outflow
                if 0 <= auxiliary_outflow <= 4:
                    yield parse_sink_input_object_v1(
                        (
                            1,
                            SINK_INPUT_TAG,
                            SINK_INPUT_SCHEMA_ID,
                            inflow_a,
                            inflow_b,
                            primary_outflow,
                            auxiliary_outflow,
                        )
                    )


def _universe_row_v1(
    universe_index: int,
    input_signature_id: int,
    canonical_input_object: tuple[object, ...],
) -> tuple[object, ...]:
    if type(universe_index) is not int or universe_index < 0:
        _fail(FAIL_Q1_UNIVERSE_ROW_ORDER, "universe index must be a uint")
    if type(input_signature_id) is not int:
        _fail(
            REJECT_Q1_UNIVERSE_SIGNATURE,
            "input signature ID must be a type-exact integer",
        )
    if type(canonical_input_object) is not tuple:
        _fail(
            FAIL_Q1_UNIVERSE_ROW_ORDER,
            "canonical input object must be a type-exact tuple",
        )
    # Constructing the observation environment supplies a second independent
    # totality/type check without learning any target output.
    try:
        environment = observation_environment_from_object_v1(canonical_input_object)
    except InputAdapterError as error:
        _fail(FAIL_Q1_UNIVERSE_ROW_ORDER, f"input object is invalid: {error}")
    if environment.input_signature_id != input_signature_id:
        _fail(
            REJECT_Q1_UNIVERSE_SIGNATURE,
            "typed input and row signature differ",
        )
    return (
        1,
        UNIVERSE_ROW_TAG,
        UNIVERSE_ROW_SCHEMA_ID,
        universe_index,
        input_signature_id,
        canonical_input_object,
    )


@dataclass(frozen=True, slots=True)
class ProductionUniverseV1:
    """One ordered, target-blind input-signature universe."""

    input_signature_id: int
    rows: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int:
            _fail(
                REJECT_Q1_UNIVERSE_SIGNATURE,
                "input signature ID must be a type-exact integer",
            )
        if type(self.rows) is not tuple:
            _fail(
                FAIL_Q1_UNIVERSE_ROW_ORDER,
                "universe rows must be an immutable type-exact tuple",
            )
        expected = {
            ODD_INPUT_SIGNATURE_ID: (ODD_UNIVERSE_ROW_COUNT, ODD_UNIVERSE_ROOT),
            SINK_INPUT_SIGNATURE_ID: (SINK_UNIVERSE_ROW_COUNT, SINK_UNIVERSE_ROOT),
        }.get(self.input_signature_id)
        if expected is None:
            _fail(
                REJECT_Q1_UNIVERSE_SIGNATURE,
                "Q1 admits only frozen input signatures 1 and 2",
            )
        expected_count, expected_root = expected
        if len(self.rows) != expected_count:
            _fail(
                FAIL_Q1_UNIVERSE_ROW_COUNT,
                f"signature {self.input_signature_id} requires {expected_count} rows",
            )
        for index, row in enumerate(self.rows):
            if (
                type(row) is not tuple
                or len(row) != 6
                or type(row[0]) is not int
                or row[0] != 1
                or type(row[1]) is not int
                or row[1] != UNIVERSE_ROW_TAG
                or type(row[2]) is not bytes
                or row[2] != UNIVERSE_ROW_SCHEMA_ID
                or type(row[3]) is not int
                or row[3] != index
                or type(row[4]) is not int
                or row[4] != self.input_signature_id
                or type(row[5]) is not tuple
            ):
                _fail(
                    FAIL_Q1_UNIVERSE_ROW_ORDER,
                    f"row {index} has a noncanonical prefix or order",
                )
            try:
                environment = observation_environment_from_object_v1(row[5])
            except InputAdapterError as error:
                _fail(
                    FAIL_Q1_UNIVERSE_ROW_ORDER,
                    f"row {index} input object is invalid: {error}",
                )
            if environment.input_signature_id != self.input_signature_id:
                _fail(
                    REJECT_Q1_UNIVERSE_SIGNATURE,
                    f"row {index} input signature differs",
                )
        actual_root = rfc6962_root(list(self.rows))
        if actual_root != expected_root:
            _fail(
                FAIL_Q1_UNIVERSE_ROOT,
                f"signature {self.input_signature_id} root differs",
            )

    @property
    def universe_root(self) -> bytes:
        return rfc6962_root(list(self.rows))

    @property
    def canonical_input_objects(self) -> tuple[tuple[object, ...], ...]:
        return tuple(row[5] for row in self.rows)  # type: ignore[misc]

    @property
    def canonical_row_bytes(self) -> tuple[bytes, ...]:
        return tuple(canonical_cbor_encode(row) for row in self.rows)

    def observation_environments(self) -> tuple[object, ...]:
        return tuple(
            observation_environment_from_object_v1(value)
            for value in self.canonical_input_objects
        )


def production_universe_v1(input_signature_id: int) -> ProductionUniverseV1:
    """Build one exact production universe without constructing truth rows."""

    if type(input_signature_id) is not int:
        _fail(
            REJECT_Q1_UNIVERSE_SIGNATURE,
            "input signature ID must be a type-exact integer",
        )
    if input_signature_id == ODD_INPUT_SIGNATURE_ID:
        inputs = _odd_input_objects_v1()
    elif input_signature_id == SINK_INPUT_SIGNATURE_ID:
        inputs = _sink_input_objects_v1()
    else:
        _fail(
            REJECT_Q1_UNIVERSE_SIGNATURE,
            "Q1 admits only frozen input signatures 1 and 2",
        )
    rows = tuple(
        _universe_row_v1(index, input_signature_id, value)
        for index, value in enumerate(inputs)
    )
    return ProductionUniverseV1(input_signature_id, rows)


def all_production_universes_v1() -> tuple[ProductionUniverseV1, ...]:
    """Return the two closures' universes in numeric signature order."""

    return tuple(
        production_universe_v1(input_signature_id)
        for input_signature_id in (
            ODD_INPUT_SIGNATURE_ID,
            SINK_INPUT_SIGNATURE_ID,
        )
    )


__all__ = [
    "FAIL_Q1_UNIVERSE_ROOT",
    "FAIL_Q1_UNIVERSE_ROW_COUNT",
    "FAIL_Q1_UNIVERSE_ROW_ORDER",
    "ODD_UNIVERSE_ROOT",
    "ODD_UNIVERSE_ROW_COUNT",
    "ProductionUniverseV1",
    "Q1UniverseError",
    "Q1_UNIVERSE_SCHEMA_ID",
    "REJECT_Q1_UNIVERSE_SIGNATURE",
    "SINK_UNIVERSE_ROOT",
    "SINK_UNIVERSE_ROW_COUNT",
    "UNIVERSE_ROW_SCHEMA_ID",
    "UNIVERSE_ROW_TAG",
    "all_production_universes_v1",
    "production_universe_v1",
]

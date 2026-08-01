"""Typed odd/sink formal rows for the Phase-3A M2.5 v1.1.2 wire.

This module is deliberately separate from :mod:`phase3_m25_wire_v1`.  The
older module remains the v1.1.1 synthetic foundation; this file implements the
newly frozen ``IdDigestV1``, ``OddInputV1`` and ``SinkInputV1`` boundary and
the deterministic 480/85-row generators.  It performs no seed generation,
signing, authoritative publication, or M3 state transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Final, Iterable, Sequence

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_leaf_hash,
    rfc6962_root,
)


MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"

ID_DIGEST_PREFIX: Final = b"HEGEL/ID_DIGEST/V1\x00"
CANONICAL_INPUT_DOMAIN: Final = "HEGEL/CANONICAL_INPUT/V1"

ODD_INPUT_TAG: Final = 0x3401
SINK_INPUT_TAG: Final = 0x3402
UNIVERSE_ROW_TAG: Final = 0x3201
TRUTH_ROW_TAG: Final = 0x3202

ODD_INPUT_SCHEMA_ID: Final = b"hegel-odd-input/1"
SINK_INPUT_SCHEMA_ID: Final = b"hegel-sink-input/1"
UNIVERSE_ROW_SCHEMA_ID: Final = b"hegel-bounded-universe-row/1"
TRUTH_ROW_SCHEMA_ID: Final = b"hegel-target-truth-row/1"

ODD_INPUT_SIGNATURE_ID: Final = 1
SINK_INPUT_SIGNATURE_ID: Final = 2

REJECT_MACHINE_ID_NON_ASCII: Final = "REJECT_MACHINE_ID_NON_ASCII"
REJECT_MACHINE_ID_SYNTAX: Final = "REJECT_MACHINE_ID_SYNTAX"
REJECT_MACHINE_ID_LENGTH: Final = "REJECT_MACHINE_ID_LENGTH"
REJECT_TYPED_INPUT_PREFIX: Final = "REJECT_TYPED_INPUT_PREFIX"
REJECT_ODD_SET_SIZE: Final = "REJECT_ODD_SET_SIZE"
REJECT_ODD_BIT_COUNT: Final = "REJECT_ODD_BIT_COUNT"
REJECT_ODD_BIT_TYPE: Final = "REJECT_ODD_BIT_TYPE"
REJECT_SINK_VALUE: Final = "REJECT_SINK_VALUE"
REJECT_SINK_BALANCE: Final = "REJECT_SINK_BALANCE"
REJECT_UNIVERSE_ROW_SCHEMA: Final = "REJECT_UNIVERSE_ROW_SCHEMA"
REJECT_TRUTH_ROW_SCHEMA: Final = "REJECT_TRUTH_ROW_SCHEMA"
FAIL_UNIVERSE_INDEX_DUPLICATE: Final = "FAIL_UNIVERSE_INDEX_DUPLICATE"
FAIL_UNIVERSE_INDEX_GAP: Final = "FAIL_UNIVERSE_INDEX_GAP"
FAIL_CANONICAL_INPUT_HASH_MISMATCH: Final = (
    "FAIL_CANONICAL_INPUT_HASH_MISMATCH"
)
FAIL_TARGET_OUTPUT_TYPE: Final = "FAIL_TARGET_OUTPUT_TYPE"
FAIL_INPUT_SIGNATURE_MISMATCH: Final = "FAIL_INPUT_SIGNATURE_MISMATCH"
FAIL_ROW_ORDERING: Final = "FAIL_ROW_ORDERING"

_MACHINE_ID_RE: Final = re.compile(rb"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


class M25TypedRowError(ValueError):
    """Stable fail-closed error for the v1.1.2 typed-row boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> "None":
    raise M25TypedRowError(code, detail)


def id_digest_preimage_v1(machine_id: str) -> bytes:
    """Return the exact ``IdDigestV1`` preimage for one machine ID."""

    if not isinstance(machine_id, str):
        raise TypeError("machine_id must be text")
    try:
        encoded = machine_id.encode("ascii")
    except UnicodeEncodeError:
        _fail(REJECT_MACHINE_ID_NON_ASCII, "machine ID must contain ASCII only")
    if len(encoded) > 256:
        _fail(REJECT_MACHINE_ID_LENGTH, "machine ID exceeds 256 ASCII bytes")
    if not encoded or _MACHINE_ID_RE.fullmatch(encoded) is None:
        _fail(REJECT_MACHINE_ID_SYNTAX, "machine ID violates the frozen syntax")
    return ID_DIGEST_PREFIX + encoded


def id_digest_v1(machine_id: str) -> bytes:
    """Return ``SHA-256(IdDigestV1 preimage)``."""

    return sha256(id_digest_preimage_v1(machine_id)).digest()


def _as_array(value: object, *, code: str, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        _fail(code, f"{name} must be a CBOR array")
    return tuple(value)


def validate_odd_input_object(value: object) -> tuple[object, ...]:
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
        _fail(REJECT_TYPED_INPUT_PREFIX, "OddInputV1 prefix or arity mismatch")
    set_size = array[3]
    if type(set_size) is not int or set_size not in (5, 6, 7, 8):
        _fail(REJECT_ODD_SET_SIZE, "odd set_size must be one of 5, 6, 7, 8")
    bits = _as_array(array[4], code=REJECT_ODD_BIT_COUNT, name="odd bits")
    if len(bits) != set_size:
        _fail(REJECT_ODD_BIT_COUNT, "odd bit count must equal set_size")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        _fail(REJECT_ODD_BIT_TYPE, "odd bits must be CBOR uint 0 or 1")
    return (1, ODD_INPUT_TAG, ODD_INPUT_SCHEMA_ID, set_size, bits)


def odd_input_v1(set_size: int, bits: Sequence[int]) -> tuple[object, ...]:
    """Construct one typed odd input without admitting Python bools."""

    if not isinstance(bits, (tuple, list)):
        _fail(REJECT_ODD_BIT_COUNT, "odd bits must be an array")
    return validate_odd_input_object(
        (1, ODD_INPUT_TAG, ODD_INPUT_SCHEMA_ID, set_size, tuple(bits))
    )


def decode_odd_input_v1(payload: bytes) -> tuple[object, ...]:
    return validate_odd_input_object(canonical_cbor_decode(payload))


def validate_sink_input_object(value: object) -> tuple[object, ...]:
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
        _fail(REJECT_TYPED_INPUT_PREFIX, "SinkInputV1 prefix or arity mismatch")
    values = array[3:]
    if any(type(item) is not int or not 0 <= item <= 4 for item in values):
        _fail(REJECT_SINK_VALUE, "sink a, b, c, d must be CBOR uint in [0, 4]")
    a, b, c, d = values
    if d != a + b - c:
        _fail(REJECT_SINK_BALANCE, "sink input must satisfy d = a + b - c")
    return (1, SINK_INPUT_TAG, SINK_INPUT_SCHEMA_ID, a, b, c, d)


def sink_input_v1(a: int, b: int, c: int, d: int) -> tuple[object, ...]:
    return validate_sink_input_object(
        (1, SINK_INPUT_TAG, SINK_INPUT_SCHEMA_ID, a, b, c, d)
    )


def decode_sink_input_v1(payload: bytes) -> tuple[object, ...]:
    return validate_sink_input_object(canonical_cbor_decode(payload))


def typed_input_signature_id(value: object) -> int:
    """Validate a typed input and return its frozen InputSignatureId."""

    array = _as_array(value, code=REJECT_TYPED_INPUT_PREFIX, name="typed input")
    if len(array) < 2 or type(array[1]) is not int:
        _fail(REJECT_TYPED_INPUT_PREFIX, "typed input has no numeric tag")
    if array[1] == ODD_INPUT_TAG:
        validate_odd_input_object(array)
        return ODD_INPUT_SIGNATURE_ID
    if array[1] == SINK_INPUT_TAG:
        validate_sink_input_object(array)
        return SINK_INPUT_SIGNATURE_ID
    _fail(REJECT_TYPED_INPUT_PREFIX, "unknown typed input tag")


def canonical_input_hash_v1(value: object) -> bytes:
    typed_input_signature_id(value)
    return content_hash(CANONICAL_INPUT_DOMAIN, value)


def bounded_universe_row_v1(
    universe_index: int,
    input_signature_id: int,
    canonical_input_object: object,
) -> tuple[object, ...]:
    if type(universe_index) is not int or universe_index < 0:
        _fail(FAIL_ROW_ORDERING, "universe_index must be a nonnegative uint")
    actual_signature = typed_input_signature_id(canonical_input_object)
    if type(input_signature_id) is not int or input_signature_id != actual_signature:
        _fail(
            FAIL_INPUT_SIGNATURE_MISMATCH,
            "row InputSignatureId does not match the canonical input tag",
        )
    return (
        1,
        UNIVERSE_ROW_TAG,
        UNIVERSE_ROW_SCHEMA_ID,
        universe_index,
        input_signature_id,
        canonical_input_object,
    )


def target_truth_row_v1(
    universe_index: int,
    canonical_input_hash: bytes,
    target_output: int,
) -> tuple[object, ...]:
    if type(universe_index) is not int or universe_index < 0:
        _fail(FAIL_ROW_ORDERING, "universe_index must be a nonnegative uint")
    if type(canonical_input_hash) is not bytes or len(canonical_input_hash) != 32:
        _fail(
            FAIL_CANONICAL_INPUT_HASH_MISMATCH,
            "canonical_input_hash must be exactly 32 bytes",
        )
    if type(target_output) is not int or target_output not in (0, 1):
        _fail(FAIL_TARGET_OUTPUT_TYPE, "target_output must be CBOR uint Bit 0 or 1")
    return (
        1,
        TRUTH_ROW_TAG,
        TRUTH_ROW_SCHEMA_ID,
        universe_index,
        canonical_input_hash,
        target_output,
    )


def _parse_universe_row(row: object) -> tuple[int, int, object]:
    array = _as_array(row, code=REJECT_UNIVERSE_ROW_SCHEMA, name="universe row")
    if (
        len(array) != 6
        or array[:3] != (1, UNIVERSE_ROW_TAG, UNIVERSE_ROW_SCHEMA_ID)
        or type(array[3]) is not int
        or array[3] < 0
        or type(array[4]) is not int
    ):
        _fail(REJECT_UNIVERSE_ROW_SCHEMA, "universe row schema mismatch")
    actual_signature = typed_input_signature_id(array[5])
    if array[4] != actual_signature:
        _fail(FAIL_INPUT_SIGNATURE_MISMATCH, "universe row signature mismatch")
    return array[3], array[4], array[5]


def _parse_truth_row(row: object) -> tuple[int, bytes, int]:
    array = _as_array(row, code=REJECT_TRUTH_ROW_SCHEMA, name="truth row")
    if (
        len(array) != 6
        or array[:3] != (1, TRUTH_ROW_TAG, TRUTH_ROW_SCHEMA_ID)
        or type(array[3]) is not int
        or array[3] < 0
    ):
        _fail(REJECT_TRUTH_ROW_SCHEMA, "truth row schema mismatch")
    input_hash = array[4]
    if type(input_hash) is not bytes or len(input_hash) != 32:
        _fail(
            FAIL_CANONICAL_INPUT_HASH_MISMATCH,
            "truth-row canonical input hash must be 32 bytes",
        )
    output = array[5]
    if type(output) is not int or output not in (0, 1):
        _fail(FAIL_TARGET_OUTPUT_TYPE, "truth output must be CBOR uint Bit 0 or 1")
    return array[3], input_hash, output


def _validate_indices(indices: Sequence[int]) -> None:
    if len(set(indices)) != len(indices):
        _fail(FAIL_UNIVERSE_INDEX_DUPLICATE, "universe indices must be unique")
    if any(left >= right for left, right in zip(indices, indices[1:])):
        _fail(FAIL_ROW_ORDERING, "rows must be ordered by ascending universe_index")
    if tuple(indices) != tuple(range(len(indices))):
        _fail(FAIL_UNIVERSE_INDEX_GAP, "universe indices must be contiguous from zero")


def validate_typed_role_rows(
    universe_rows: Sequence[object],
    truth_rows: Sequence[object],
    *,
    expected_input_signature_id: int,
) -> None:
    """Validate index, signature, hash, and Bit-output binding for one role."""

    parsed_universe = tuple(_parse_universe_row(row) for row in universe_rows)
    parsed_truth = tuple(_parse_truth_row(row) for row in truth_rows)
    universe_indices = tuple(item[0] for item in parsed_universe)
    truth_indices = tuple(item[0] for item in parsed_truth)
    _validate_indices(universe_indices)
    _validate_indices(truth_indices)
    if universe_indices != truth_indices:
        _fail(FAIL_UNIVERSE_INDEX_GAP, "universe and truth indices differ")
    for universe, truth in zip(parsed_universe, parsed_truth, strict=True):
        _, signature_id, input_object = universe
        _, input_hash, _ = truth
        if signature_id != expected_input_signature_id:
            _fail(FAIL_INPUT_SIGNATURE_MISMATCH, "role InputSignatureId mismatch")
        if canonical_input_hash_v1(input_object) != input_hash:
            _fail(
                FAIL_CANONICAL_INPUT_HASH_MISMATCH,
                "truth row does not bind its canonical input",
            )


@dataclass(frozen=True, slots=True)
class TypedRoleRows:
    role_name: str
    input_signature_id: int
    universe_rows: tuple[tuple[object, ...], ...]
    truth_rows: tuple[tuple[object, ...], ...]

    def validate(self) -> None:
        validate_typed_role_rows(
            self.universe_rows,
            self.truth_rows,
            expected_input_signature_id=self.input_signature_id,
        )

    @property
    def universe_root(self) -> bytes:
        self.validate()
        return rfc6962_root(list(self.universe_rows))

    @property
    def truth_root(self) -> bytes:
        self.validate()
        return rfc6962_root(list(self.truth_rows))


def _build_role_rows(
    role_name: str,
    input_signature_id: int,
    inputs_and_outputs: Iterable[tuple[tuple[object, ...], int]],
) -> TypedRoleRows:
    universe_rows: list[tuple[object, ...]] = []
    truth_rows: list[tuple[object, ...]] = []
    for index, (input_object, output) in enumerate(inputs_and_outputs):
        input_hash = canonical_input_hash_v1(input_object)
        universe_rows.append(
            bounded_universe_row_v1(index, input_signature_id, input_object)
        )
        truth_rows.append(target_truth_row_v1(index, input_hash, output))
    result = TypedRoleRows(
        role_name=role_name,
        input_signature_id=input_signature_id,
        universe_rows=tuple(universe_rows),
        truth_rows=tuple(truth_rows),
    )
    result.validate()
    return result


def generate_odd_role_rows_v1() -> TypedRoleRows:
    def inputs() -> Iterable[tuple[tuple[object, ...], int]]:
        for set_size in range(5, 9):
            for numeric_value in range(1 << set_size):
                bits = tuple(
                    (numeric_value >> (set_size - 1 - offset)) & 1
                    for offset in range(set_size)
                )
                yield odd_input_v1(set_size, bits), sum(bits) % 2

    result = _build_role_rows("odd", ODD_INPUT_SIGNATURE_ID, inputs())
    if len(result.universe_rows) != 480:
        raise AssertionError("odd generator must emit exactly 480 rows")
    return result


def generate_sink_role_rows_v1() -> TypedRoleRows:
    def inputs() -> Iterable[tuple[tuple[object, ...], int]]:
        for a in range(5):
            for b in range(5):
                for c in range(5):
                    for d in range(5):
                        if d == a + b - c:
                            yield sink_input_v1(a, b, c, d), 1

    result = _build_role_rows("sink", SINK_INPUT_SIGNATURE_ID, inputs())
    if len(result.universe_rows) != 85:
        raise AssertionError("sink generator must emit exactly 85 rows")
    return result


def typed_role_report_v1(rows: TypedRoleRows) -> dict[str, object]:
    """Render a deterministic diagnostic report for shared golden testing."""

    rows.validate()
    samples: list[dict[str, object]] = []
    for input_row, truth_row in zip(
        rows.universe_rows[:2], rows.truth_rows[:2], strict=True
    ):
        input_object = input_row[5]
        input_cbor = canonical_cbor_encode(input_object)
        universe_cbor = canonical_cbor_encode(input_row)
        truth_cbor = canonical_cbor_encode(truth_row)
        samples.append(
            {
                "universe_index": input_row[3],
                "input_cbor_hex": input_cbor.hex(),
                "canonical_input_hash_hex": canonical_input_hash_v1(
                    input_object
                ).hex(),
                "universe_row_cbor_hex": universe_cbor.hex(),
                "universe_leaf_hash_hex": rfc6962_leaf_hash(input_row).hex(),
                "truth_row_cbor_hex": truth_cbor.hex(),
                "truth_leaf_hash_hex": rfc6962_leaf_hash(truth_row).hex(),
            }
        )
    return {
        "role_name": rows.role_name,
        "input_signature_id": rows.input_signature_id,
        "row_count": len(rows.universe_rows),
        "samples": samples,
        "universe_two_row_root_hex": rfc6962_root(
            list(rows.universe_rows[:2])
        ).hex(),
        "truth_two_row_root_hex": rfc6962_root(list(rows.truth_rows[:2])).hex(),
        "universe_root_hex": rows.universe_root.hex(),
        "truth_root_hex": rows.truth_root.hex(),
    }


def complete_typed_rows_report_v1() -> dict[str, object]:
    machine_id = "hegel-old-dsl-v1.1.0"
    return {
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "id_digest": {
            "machine_id": machine_id,
            "preimage_hex": id_digest_preimage_v1(machine_id).hex(),
            "digest_hex": id_digest_v1(machine_id).hex(),
        },
        "roles": [
            typed_role_report_v1(generate_odd_role_rows_v1()),
            typed_role_report_v1(generate_sink_role_rows_v1()),
        ],
    }


__all__ = [
    "CANONICAL_INPUT_DOMAIN",
    "FAIL_CANONICAL_INPUT_HASH_MISMATCH",
    "FAIL_INPUT_SIGNATURE_MISMATCH",
    "FAIL_ROW_ORDERING",
    "FAIL_TARGET_OUTPUT_TYPE",
    "FAIL_UNIVERSE_INDEX_DUPLICATE",
    "FAIL_UNIVERSE_INDEX_GAP",
    "ID_DIGEST_PREFIX",
    "M25TypedRowError",
    "MACHINE_FREEZE_ID",
    "ODD_INPUT_SIGNATURE_ID",
    "ODD_INPUT_TAG",
    "REJECT_MACHINE_ID_LENGTH",
    "REJECT_MACHINE_ID_NON_ASCII",
    "REJECT_MACHINE_ID_SYNTAX",
    "REJECT_ODD_BIT_COUNT",
    "REJECT_ODD_BIT_TYPE",
    "REJECT_ODD_SET_SIZE",
    "REJECT_SINK_BALANCE",
    "REJECT_SINK_VALUE",
    "REJECT_TYPED_INPUT_PREFIX",
    "SINK_INPUT_SIGNATURE_ID",
    "SINK_INPUT_TAG",
    "TypedRoleRows",
    "bounded_universe_row_v1",
    "canonical_input_hash_v1",
    "complete_typed_rows_report_v1",
    "decode_odd_input_v1",
    "decode_sink_input_v1",
    "generate_odd_role_rows_v1",
    "generate_sink_role_rows_v1",
    "id_digest_preimage_v1",
    "id_digest_v1",
    "odd_input_v1",
    "sink_input_v1",
    "target_truth_row_v1",
    "typed_input_signature_id",
    "typed_role_report_v1",
    "validate_odd_input_object",
    "validate_sink_input_object",
    "validate_typed_role_rows",
]

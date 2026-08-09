"""Sealed target-free strict-admission vectors for shrink step 3.

The ordered manifest binds exact source JSON bytes or formal CBOR bytes plus
the expected disposition.  Python and Rust may implement admission
independently, but a dual qualification compares this manifest root and the
root of every observed outcome rather than trusting equal category counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final, Mapping

from .strict_cbor_v1 import canonical_cbor_encode


GOLDEN_MANIFEST_DOMAIN: Final = b"HEGEL/SHRINK3/STRICT_GOLDEN_MANIFEST/V1"
GOLDEN_OUTCOME_DOMAIN: Final = b"HEGEL/SHRINK3/STRICT_GOLDEN_OUTCOME/V1"
ACCEPT_PARENT_IDENTITY: Final = "ACCEPT_PARENT_IDENTITY"


@dataclass(frozen=True, slots=True)
class StrictGoldenVectorV1:
    vector_id: str
    category: str
    boundary: str
    input_value: object
    expected_disposition: str

    def __post_init__(self) -> None:
        if not self.vector_id or not self.vector_id.isascii():
            raise ValueError("golden vector ID must be nonempty ASCII")
        if self.boundary not in {"SOURCE_JSON", "FORMAL_CBOR"}:
            raise ValueError("unknown golden vector boundary")
        if not self.category.endswith("_checks"):
            raise ValueError("golden vector category must be a report count field")

    @property
    def input_wire(self) -> bytes:
        if self.boundary == "SOURCE_JSON":
            return json.dumps(
                self.input_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        return canonical_cbor_encode(self.input_value)


def _source(
    vector_id: str, category: str, value: object, expected: str
) -> StrictGoldenVectorV1:
    return StrictGoldenVectorV1(
        vector_id, category, "SOURCE_JSON", value, expected
    )


def _formal(
    vector_id: str, category: str, value: object, expected: str
) -> StrictGoldenVectorV1:
    return StrictGoldenVectorV1(
        vector_id, category, "FORMAL_CBOR", value, expected
    )


def _nonconstant_binary(name: str) -> list[object]:
    return [
        name,
        ["bit_to_scalar", ["bit_at", 0]],
        ["scalar_const", 1, 1],
    ]


STRICT_GOLDEN_VECTORS_V1: Final = (
    _source("S01", "surviving_identity_checks", ["scalar_const", 1], ACCEPT_PARENT_IDENTITY),
    _source("S02", "surviving_identity_checks", ["scalar_const", 3], ACCEPT_PARENT_IDENTITY),
    _source("S03", "surviving_identity_checks", ["scalar_const", 5], ACCEPT_PARENT_IDENTITY),
    _source("S04", "surviving_identity_checks", _nonconstant_binary("difference"), ACCEPT_PARENT_IDENTITY),
    _source(
        "S05",
        "surviving_identity_checks",
        ["difference", ["scalar_const", 1], ["scalar_const", 5]],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "S06",
        "surviving_identity_checks",
        [
            "greater_equal",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1],
        ],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "S07",
        "surviving_identity_checks",
        [
            "aggregate",
            "signed_balance_v1",
            "scope_all_observed_v1",
            "q0",
            [],
        ],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "S08",
        "surviving_identity_checks",
        [
            "equal_exact",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1],
        ],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source("A01", "source_add_rejection_checks", _nonconstant_binary("add"), "REJECT_REMOVED_BINARY_OPERATOR"),
    _source(
        "A02",
        "source_add_rejection_checks",
        ["add", ["scalar_const", 1], ["scalar_const", 5]],
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _source(
        "A03",
        "source_add_rejection_checks",
        ["add", ["scalar_const", 5], ["scalar_const", 5]],
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _source(
        "A04",
        "source_add_rejection_checks",
        ["difference", _nonconstant_binary("add"), ["scalar_const", 3]],
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _source(
        "P01",
        "source_priority_checks",
        ["add", ["scalar_const", 1]],
        "REJECT_MALFORMED_SOURCE_AST",
    ),
    _source(
        "P02",
        "source_priority_checks",
        ["add", ["bit_at", 0], ["scalar_const", 1]],
        "REJECT_IMPLICIT_COERCION",
    ),
    _source(
        "P03",
        "source_priority_checks",
        [
            "add",
            ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
            ["scalar_const", -2, 1],
        ],
        "REJECT_REMOVED_AGGREGATE_MAP",
    ),
    _source(
        "P04",
        "source_priority_checks",
        ["add", ["scalar_const", -2, 1], ["scalar_const", 1]],
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    ),
    _source(
        "P05",
        "source_priority_checks",
        ["unknown_outer", _nonconstant_binary("add")],
        "REJECT_UNKNOWN_EXPRESSION",
    ),
    _source(
        "P06",
        "source_priority_checks",
        [7, ["scalar_const", 1], ["scalar_const", 5]],
        "REJECT_MALFORMED_SOURCE_AST",
    ),
    _formal(
        "F01",
        "formal_add_rejection_checks",
        (1, (2, 0, (0, 0, 5), (1, 0, (0, 1, 0)))),
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _formal(
        "F02",
        "formal_add_rejection_checks",
        (1, (2, 0, (0, 0, 1), (0, 0, 5))),
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _formal(
        "F03",
        "formal_add_rejection_checks",
        (1, (2, 1, (2, 0, (1, 0, (0, 1, 0)), (0, 0, 5)), (0, 0, 3))),
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _formal("Q01", "formal_priority_checks", (1, (2, 0, (0, 0, 1))), "REJECT_NONCANONICAL_AST"),
    _formal(
        "Q02",
        "formal_priority_checks",
        (1, (2, 0, (0, 1, 0), (0, 0, 1))),
        "REJECT_TYPE_MISMATCH",
    ),
    _formal(
        "Q03",
        "formal_priority_checks",
        (1, (2, 0, (0, 3, 2, 0, 0, ()), (0, 0, 0))),
        "REJECT_REMOVED_AGGREGATE_MAP",
    ),
    _formal(
        "Q04",
        "formal_priority_checks",
        (1, (2, 0, (0, 0, 0), (0, 0, 1))),
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    ),
    _formal(
        "Q05",
        "formal_priority_checks",
        (1, (2, 4, (2, 0, (1, 0, (0, 1, 0)), (0, 0, 1)), (0, 0, 1))),
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _formal(
        "Q06",
        "formal_priority_checks",
        (1, (0, 0, 1, (2, 0, (0, 0, 1), (0, 0, 5)))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal("H01", "formal_shape_priority_checks", (1, (4, ())), "REJECT_NONCANONICAL_AST"),
    _formal(
        "H02",
        "formal_shape_priority_checks",
        (1, (4, ((2, 0, (0, 0, 1), (0, 0, 5)),))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "H03",
        "formal_shape_priority_checks",
        (1, (4, ((0, 4, 0), (0, 4, 1), (0, 4, 2), (2, 0, (0, 0, 1), (0, 0, 5))))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "H04",
        "formal_shape_priority_checks",
        (1, (2, 0, (0, 3, 2, 0, 0, ((0, False), (1, False), (2, False))), (0, 0, 0))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "H05",
        "formal_shape_priority_checks",
        (1, (2, 0, (0, 3, 2, 0, 0, ((1, False), (0, False))), (0, 0, 0))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "H06",
        "formal_shape_priority_checks",
        (1, (2, 0, (0, 3, 2, 0, 0, ((0, False), (0, True))), (0, 0, 0))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "R01",
        "formal_alias_or_reserved_checks",
        (1, (2, 4, (0, 0, 1), (0, 0, 5))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "R02",
        "formal_alias_or_reserved_checks",
        (1, (2, 7, (0, 0, 1), (0, 0, 5))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "R03",
        "formal_alias_or_reserved_checks",
        (1, (2, 8, (0, 0, 1), (0, 0, 5))),
        "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
    ),
)


def _framed_hash(domain: bytes, rows: tuple[tuple[bytes, ...], ...]) -> str:
    digest = sha256()
    digest.update(domain)
    digest.update(b"\x00")
    for row in rows:
        for field in row:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def strict_golden_manifest_root_v1() -> str:
    rows = tuple(
        (
            vector.vector_id.encode("ascii"),
            vector.category.encode("ascii"),
            vector.boundary.encode("ascii"),
            vector.input_wire,
            vector.expected_disposition.encode("ascii"),
        )
        for vector in STRICT_GOLDEN_VECTORS_V1
    )
    return _framed_hash(GOLDEN_MANIFEST_DOMAIN, rows)


def accepted_outcome_bytes(cbor_bytes: bytes, digest: bytes) -> bytes:
    if type(cbor_bytes) is not bytes or type(digest) is not bytes or len(digest) != 32:
        raise TypeError("accepted golden outcome requires bytes and a 32-byte hash")
    return b"ACCEPT\x00" + len(cbor_bytes).to_bytes(8, "big") + cbor_bytes + digest


def rejected_outcome_bytes(error_code: str) -> bytes:
    return b"REJECT\x00" + error_code.encode("ascii")


def strict_golden_outcome_root_v1(outcomes: Mapping[str, bytes]) -> str:
    expected_ids = tuple(vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1)
    if tuple(outcomes) != expected_ids:
        raise ValueError("golden outcomes must use exact manifest order")
    rows = tuple(
        (vector_id.encode("ascii"), outcomes[vector_id])
        for vector_id in expected_ids
    )
    return _framed_hash(GOLDEN_OUTCOME_DOMAIN, rows)


if len(STRICT_GOLDEN_VECTORS_V1) != 36 or len(
    {vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1}
) != 36:
    raise AssertionError("shrink-3 strict golden vector manifest drift")


__all__ = [
    "ACCEPT_PARENT_IDENTITY",
    "GOLDEN_MANIFEST_DOMAIN",
    "GOLDEN_OUTCOME_DOMAIN",
    "STRICT_GOLDEN_VECTORS_V1",
    "StrictGoldenVectorV1",
    "accepted_outcome_bytes",
    "rejected_outcome_bytes",
    "strict_golden_manifest_root_v1",
    "strict_golden_outcome_root_v1",
]

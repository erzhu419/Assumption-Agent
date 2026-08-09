"""Sealed target-free strict-admission vectors for shrink step 4."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final, Mapping

from .strict_cbor_v1 import canonical_cbor_encode


GOLDEN_MANIFEST_DOMAIN: Final = b"HEGEL/SHRINK4/STRICT_GOLDEN_MANIFEST/V1"
GOLDEN_OUTCOME_DOMAIN: Final = b"HEGEL/SHRINK4/STRICT_GOLDEN_OUTCOME/V1"
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


_SOURCE_A: Final = ["context_flag", "c0"]
_SOURCE_B: Final = ["context_flag", "c1"]
_SOURCE_C: Final = ["context_flag", "c2"]
_SOURCE_D: Final = ["context_flag", "c3"]
_FORMAL_A: Final = (0, 4, 0)
_FORMAL_B: Final = (0, 4, 1)
_FORMAL_C: Final = (0, 4, 2)
_FORMAL_D: Final = (0, 4, 3)
_FORMAL_ADD_BOOL: Final = (
    2,
    2,
    (2, 0, (0, 0, 1), (0, 0, 5)),
    (0, 0, 3),
)
_FORMAL_REMOVED_AGGREGATE_BOOL: Final = (
    2,
    2,
    (0, 3, 2, 0, 0, ()),
    (0, 0, 1),
)
_FORMAL_REMOVED_RATIONAL_BOOL: Final = (
    2,
    2,
    (0, 0, 0),
    (0, 0, 1),
)
_FORMAL_ALIAS_BOOL: Final = (2, 4, (0, 0, 1), (0, 0, 5))
_FORMAL_RESERVED_BOOL: Final = (2, 7, (0, 0, 1), (0, 0, 5))


STRICT_GOLDEN_VECTORS_V1: Final = (
    _source(
        "S01",
        "surviving_identity_checks",
        ["scalar_const", 1],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "S02",
        "surviving_identity_checks",
        ["difference", ["scalar_const", 1], ["scalar_const", 5]],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "S03",
        "surviving_identity_checks",
        ["top_level_AND", _SOURCE_A, _SOURCE_B],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "N01",
        "source_normalization_before_limit_checks",
        ["top_level_AND", _SOURCE_A],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "N02",
        "source_normalization_before_limit_checks",
        ["top_level_AND", _SOURCE_A, _SOURCE_A, _SOURCE_B],
        ACCEPT_PARENT_IDENTITY,
    ),
    _source(
        "L01",
        "source_structural_limit_checks",
        ["top_level_AND", _SOURCE_A, _SOURCE_B, _SOURCE_C],
        "REJECT_STRUCTURAL_LIMIT",
    ),
    _source(
        "L02",
        "source_structural_limit_checks",
        [
            "top_level_AND",
            _SOURCE_A,
            ["top_level_AND", _SOURCE_B, _SOURCE_C],
        ],
        "REJECT_STRUCTURAL_LIMIT",
    ),
    _source(
        "P01",
        "source_priority_checks",
        [
            "top_level_AND",
            _SOURCE_A,
            _SOURCE_B,
            [
                "equal_exact",
                [
                    "aggregate",
                    "mean_v1",
                    "scope_all_observed_v1",
                    "q0",
                    [],
                ],
                ["scalar_const", 1],
            ],
        ],
        "REJECT_REMOVED_AGGREGATE_MAP",
    ),
    _source(
        "P02",
        "source_priority_checks",
        [
            "top_level_AND",
            _SOURCE_A,
            _SOURCE_B,
            [
                "equal_exact",
                ["scalar_const", -2, 1],
                ["scalar_const", 1],
            ],
        ],
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    ),
    _source(
        "P03",
        "source_priority_checks",
        [
            "top_level_AND",
            _SOURCE_A,
            _SOURCE_B,
            [
                "equal_exact",
                ["add", ["scalar_const", 1], ["scalar_const", 5]],
                ["scalar_const", 3],
            ],
        ],
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _source(
        "P04",
        "source_priority_checks",
        ["top_level_AND", _SOURCE_A, _SOURCE_B, ["scalar_const", 1]],
        "REJECT_TYPE_MISMATCH",
    ),
    _source(
        "P05",
        "source_priority_checks",
        [
            "top_level_AND",
            _SOURCE_A,
            _SOURCE_B,
            ["add", ["scalar_const", 1]],
        ],
        "REJECT_MALFORMED_SOURCE_AST",
    ),
    _formal(
        "F01",
        "formal_surviving_identity_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B))),
        ACCEPT_PARENT_IDENTITY,
    ),
    _formal(
        "F02",
        "formal_structural_limit_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_C))),
        "REJECT_STRUCTURAL_LIMIT",
    ),
    _formal(
        "F03",
        "formal_priority_checks",
        (1, (4, (_FORMAL_C, _FORMAL_B, _FORMAL_A))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "F04",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_REMOVED_AGGREGATE_BOOL))),
        "REJECT_REMOVED_AGGREGATE_MAP",
    ),
    _formal(
        "F05",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_REMOVED_RATIONAL_BOOL))),
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    ),
    _formal(
        "F06",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_ADD_BOOL))),
        "REJECT_REMOVED_BINARY_OPERATOR",
    ),
    _formal(
        "F07",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_ALIAS_BOOL))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "F08",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_RESERVED_BOOL))),
        "REJECT_NONCANONICAL_AST",
    ),
    _formal(
        "F09",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, (0, 0, 1)))),
        "REJECT_TYPE_MISMATCH",
    ),
    _formal(
        "F10",
        "formal_priority_checks",
        (1, (4, (_FORMAL_A, _FORMAL_B, _FORMAL_C, _FORMAL_D))),
        "REJECT_NONCANONICAL_AST",
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


if len(STRICT_GOLDEN_VECTORS_V1) != 22 or len(
    {vector.vector_id for vector in STRICT_GOLDEN_VECTORS_V1}
) != 22:
    raise AssertionError("shrink-4 strict golden vector manifest drift")


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

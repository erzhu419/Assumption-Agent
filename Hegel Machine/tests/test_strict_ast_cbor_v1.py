from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from hegel_machine.strict_ast_v1 import (
    StrictAstError,
    canonicalize_source_ast,
    decode_canonical_ast,
    migrate_legacy_scope_alias,
)
from hegel_machine.strict_cbor_v1 import (
    StrictCborError,
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "golden_vectors" / "strict_ast_cbor_v1.json"


def _vectors() -> dict[str, object]:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def test_shared_golden_vector_header_is_bound_to_v102() -> None:
    vectors = _vectors()
    assert vectors["schema_version"] == "hegel-strict-golden-v1"
    assert vectors["freeze_version"] == "hegel-freeze-p2b-p3-v1.0.2"
    assert vectors["cbor_profile_id"] == "hegel-cbor-det-v1"
    assert vectors["ast_schema_id"] == "hegel-canonical-ast-v1"


def test_strict_cbor_encode_and_round_trip_golden_vectors() -> None:
    for vector in _vectors()["cbor_encode_vectors"]:
        assert isinstance(vector, dict)
        if "byte_string_hex" in vector:
            value = bytes.fromhex(vector["byte_string_hex"])
        else:
            value = vector["value"]
        encoded = canonical_cbor_encode(value)
        assert encoded.hex() == vector["expected_cbor_hex"], vector["name"]
        decoded = canonical_cbor_decode(encoded)
        if isinstance(value, list):
            assert decoded == tuple(value)
        else:
            assert decoded == value


def test_strict_cbor_rejection_golden_vectors() -> None:
    for vector in _vectors()["cbor_reject_vectors"]:
        assert isinstance(vector, dict)
        with pytest.raises(StrictCborError) as rejected:
            canonical_cbor_decode(bytes.fromhex(vector["encoded_hex"]))
        assert rejected.value.code == vector["error_code"], vector["name"]


def test_strict_cbor_decoder_rejects_nesting_beyond_shared_limit() -> None:
    accepted = (b"\x81" * 64) + b"\x00"
    rejected = (b"\x81" * 65) + b"\x00"
    canonical_cbor_decode(accepted)
    with pytest.raises(StrictCborError) as error:
        canonical_cbor_decode(rejected)
    assert error.value.code == "REJECT_CBOR_NESTING"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("text", "REJECT_CBOR_TEXT"),
        ({"map": 1}, "REJECT_CBOR_MAP"),
        (1.0, "REJECT_CBOR_FLOAT"),
        (1 << 64, "REJECT_CBOR_INTEGER_RANGE"),
        (-(1 << 64) - 1, "REJECT_CBOR_INTEGER_RANGE"),
    ],
)
def test_strict_cbor_encoder_rejects_values_outside_profile(
    value: object, code: str
) -> None:
    with pytest.raises(StrictCborError) as rejected:
        canonical_cbor_encode(value)
    assert rejected.value.code == code


def test_ast_acceptance_golden_vectors_and_strict_decode() -> None:
    for vector in _vectors()["ast_accept_vectors"]:
        assert isinstance(vector, dict)
        accepted = canonicalize_source_ast(vector["source_ast"])
        assert accepted.cbor_bytes.hex() == vector["canonical_cbor_hex"], vector["name"]
        assert accepted.hash_id == vector["canonical_ast_hash"], vector["name"]
        assert accepted.root_operator_id == vector["root_operator_id"], vector["name"]
        assert accepted.metrics.output_sort == vector["output_sort"], vector["name"]
        assert accepted.metrics.depth == vector["depth"], vector["name"]
        assert accepted.metrics.node_count == vector["node_count"], vector["name"]
        assert decode_canonical_ast(accepted.cbor_bytes) == accepted


def test_ast_rejection_golden_vectors() -> None:
    for vector in _vectors()["ast_reject_vectors"]:
        assert isinstance(vector, dict)
        with pytest.raises(StrictAstError) as rejected:
            canonicalize_source_ast(vector["source_ast"])
        assert rejected.value.code == vector["error_code"], vector["name"]


def test_scope_alias_only_exists_in_explicit_migration_adapter() -> None:
    alias = [
        "aggregate",
        "sum_v1",
        "control_volume_primary_only_v1",
        "q0",
        [],
    ]
    canonical = ["aggregate", "sum_v1", "scope_primary_only_v1", "q0", []]
    assert migrate_legacy_scope_alias(alias) == canonicalize_source_ast(canonical)


def test_named_source_transport_does_not_invent_optional_or_text_slot_forms() -> None:
    with pytest.raises(StrictAstError, match="REJECT_REGISTRY_INDEX_OUT_OF_RANGE"):
        canonicalize_source_ast(["bit_at", "e0"])
    with pytest.raises(StrictAstError, match="REJECT_MALFORMED_SOURCE_AST"):
        canonicalize_source_ast(
            ["aggregate", "sum_v1", "scope_primary_only_v1", "q0"]
        )


def test_only_frozen_local_algebraic_rewrites_are_applied() -> None:
    zero = ["scalar_const", 0, 1]
    half = ["scalar_const", 1, 2]
    one = ["scalar_const", 1, 1]
    negative_one = ["scalar_const", -1, 1]
    bit = ["bit_to_scalar", ["bit_at", 0]]

    assert canonicalize_source_ast(["scalar_const", 2, 2]) == canonicalize_source_ast(one)

    assert canonicalize_source_ast(["add", bit, zero]) == canonicalize_source_ast(bit)
    assert canonicalize_source_ast(["difference", bit, zero]) == canonicalize_source_ast(bit)
    assert canonicalize_source_ast(["difference", bit, bit]) == canonicalize_source_ast(zero)
    assert canonicalize_source_ast(["absolute", ["absolute", bit]]) == canonicalize_source_ast(
        ["absolute", bit]
    )
    assert canonicalize_source_ast(["add", half, half]) == canonicalize_source_ast(one)
    assert canonicalize_source_ast(["difference", zero, one]) == canonicalize_source_ast(
        negative_one
    )
    assert canonicalize_source_ast(["absolute", negative_one]) == canonicalize_source_ast(one)

    # Two constants whose exact sum is outside the parameter grid remain an add.
    assert canonicalize_source_ast(
        ["add", ["scalar_const", 2, 1], ["scalar_const", 1, 1]]
    ).metrics.node_count == 3


def test_decoder_rejects_rewrite_required_ast_even_when_cbor_is_deterministic() -> None:
    zero_index = 3
    one_index = 5
    rewrite_required = (
        1,
        (2, 0, (0, 0, zero_index), (0, 0, one_index)),
    )
    with pytest.raises(StrictAstError, match="REJECT_NONCANONICAL_AST"):
        decode_canonical_ast(canonical_cbor_encode(rewrite_required))

    greater_equal = (1, (2, 4, (0, 0, one_index), (0, 0, zero_index)))
    with pytest.raises(StrictAstError, match="REJECT_NONCANONICAL_AST"):
        decode_canonical_ast(canonical_cbor_encode(greater_equal))

    approx_zero = (1, (3, 0, (0, 0, zero_index), (0, 0, one_index), 0))
    with pytest.raises(StrictAstError, match="REJECT_NONCANONICAL_AST"):
        decode_canonical_ast(canonical_cbor_encode(approx_zero))


def test_decoder_rejects_wrong_envelope_and_trailing_bytes() -> None:
    with pytest.raises(StrictAstError, match="REJECT_UNKNOWN_AST_SCHEMA"):
        decode_canonical_ast(canonical_cbor_encode((2, (0, 2))))
    accepted = canonicalize_source_ast(["set_size"])
    with pytest.raises(StrictCborError, match="REJECT_TRAILING_CBOR"):
        decode_canonical_ast(accepted.cbor_bytes + b"\x00")


def test_tolerance_and_aggregate_metadata_do_not_count_as_children() -> None:
    aggregate = canonicalize_source_ast(
        [
            "aggregate",
            "sum_v1",
            "scope_primary_only_v1",
            "q0",
            [["c0", True], ["c1", False]],
        ]
    )
    assert aggregate.metrics.node_count == 1
    assert aggregate.metrics.depth == 0
    assert aggregate.metrics.scope_clause_count == 2

    approximate = canonicalize_source_ast(
        [
            "approx_equal",
            ["scalar_const", -1, 1],
            ["scalar_const", 1, 1],
            1,
        ]
    )
    assert approximate.metrics.node_count == 3
    assert approximate.metrics.depth == 1

    normalized_tolerance = canonicalize_source_ast(
        [
            "approx_equal",
            ["scalar_const", -1, 1],
            ["scalar_const", 1, 1],
            2,
            8,
        ]
    )
    assert normalized_tolerance == approximate

    for invalid in ((1, 0), (-1, 4)):
        with pytest.raises(StrictAstError, match="REJECT_REGISTRY_INDEX_OUT_OF_RANGE"):
            canonicalize_source_ast(
                [
                    "approx_equal",
                    ["scalar_const", -1, 1],
                    ["scalar_const", 1, 1],
                    *invalid,
                ]
            )


def test_rfc6962_golden_vectors_and_no_duplicate_last() -> None:
    for vector in _vectors()["rfc6962_vectors"]:
        records = [(1, index) for index in range(vector["leaf_count"])]
        assert rfc6962_root(records).hex() == vector["expected_root"], vector["name"]

    records = [(1, index) for index in range(3)]
    leaves = [sha256(b"\x00" + canonical_cbor_encode(item)).digest() for item in records]
    duplicate_last_root = sha256(
        b"\x01"
        + sha256(b"\x01" + leaves[0] + leaves[1]).digest()
        + sha256(b"\x01" + leaves[2] + leaves[2]).digest()
    ).digest()
    assert rfc6962_root(records) != duplicate_last_root


def test_content_hash_has_domain_separator_and_canonical_ast_uses_it() -> None:
    value = (1, (0, 2))
    expected = sha256(b"HEGEL/AST/V1\x00" + canonical_cbor_encode(value)).digest()
    assert content_hash("HEGEL/AST/V1", value) == expected
    assert canonicalize_source_ast(["set_size"]).digest == expected

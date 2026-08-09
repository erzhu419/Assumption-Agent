from __future__ import annotations

import pytest

from hegel_machine.phase3_m3_shrink3_core_v1 import (
    RESERVED_FORMAL_BINARY_OPERATOR_ERROR,
    UNALLOCATED_BINARY_OPERATOR_REGISTRY_ERROR,
    UNKNOWN_SOURCE_OPERATOR_NAME_ERROR,
)
from hegel_machine.phase3_shrink3_registry_v1 import (
    BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID,
    OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID,
    REJECTION_PRIORITY_DIAGNOSTIC_ID,
    SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID,
    binary_operator_registry_object,
)
from hegel_machine.phase3_shrink2_capacity_v1 import (
    EXPECTED_SHRINK2_SOURCE_COUNT,
    iter_shrink2_capacity_candidate_asts,
)
from hegel_machine.strict_ast_shrink2_v1 import (
    canonicalize_shrink2_source_ast,
    decode_shrink2_canonical_ast,
)
from hegel_machine.strict_ast_shrink3_v1 import (
    canonicalize_shrink3_source_ast,
    decode_shrink3_canonical_ast,
    read_legacy_parent_program,
)
from hegel_machine.strict_ast_v1 import StrictAstError
from hegel_machine.strict_cbor_v1 import StrictCborError, canonical_cbor_encode


REMOVED_BINARY = "REJECT_REMOVED_BINARY_OPERATOR"


def test_diagnostic_identity_constants_are_literal_pinned() -> None:
    assert BINARY_OPERATOR_REGISTRY_DIAGNOSTIC_ID == (
        "binary_operator_registry_"
        "0877efd49c27eadbd9d558ed2475889eb54ed75377c986092e1d701e9b162724"
    )
    assert REJECTION_PRIORITY_DIAGNOSTIC_ID == (
        "rejection_priority_"
        "50d2d00a766638968b124f8884d6c1529959b11c3cb862d27828f433f7d574a4"
    )
    assert OPERATOR_ADMISSION_SEMANTICS_DIAGNOSTIC_ID == (
        "operator_admission_semantics_"
        "8689788b1c78739ed04fd41c20dac10ae4dd6bbe08fe029efbe955c8d048d5d9"
    )
    assert SHRUNK_DSL_SURFACE_DIAGNOSTIC_ID == (
        "dsl_spec_"
        "af0ea33f9542f06a6988addeee673dfa42df0dfe2bf99210221c10653348d3d0"
    )


def _assert_error(callable_: object, expected: str) -> None:
    with pytest.raises((StrictAstError, StrictCborError)) as raised:
        callable_()  # type: ignore[operator]
    assert raised.value.code == expected


def _nonconstant_binary_source(name: str) -> list[object]:
    return [
        name,
        ["bit_to_scalar", ["bit_at", 0]],
        ["scalar_const", 1, 1],
    ]


def test_reserved_binary_id_has_boundary_specific_inherited_errors() -> None:
    registry = binary_operator_registry_object()
    assert registry["unallocated_registry_id_error"] == (
        UNALLOCATED_BINARY_OPERATOR_REGISTRY_ERROR
    ) == "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"
    assert registry["source_numeric_operator_id_accepted"] is False
    assert registry["unknown_source_operator_name_error"] == (
        UNKNOWN_SOURCE_OPERATOR_NAME_ERROR
    ) == "REJECT_UNKNOWN_EXPRESSION"
    assert registry["formal_reserved_id_error"] == (
        RESERVED_FORMAL_BINARY_OPERATOR_ERROR
    ) == "REJECT_NONCANONICAL_AST"

    _assert_error(
        lambda: canonicalize_shrink3_source_ast(
            [7, ["scalar_const", 1], ["scalar_const", 5]]
        ),
        "REJECT_MALFORMED_SOURCE_AST",
    )
    _assert_error(
        lambda: canonicalize_shrink3_source_ast(
            ["not_an_old_dsl_operator", ["scalar_const", 1]]
        ),
        "REJECT_UNKNOWN_EXPRESSION",
    )
    _assert_error(
        lambda: decode_shrink3_canonical_ast(
            canonical_cbor_encode((1, (2, 7, (0, 0, 1), (0, 0, 5))))
        ),
        "REJECT_NONCANONICAL_AST",
    )


def test_add_source_name_and_formal_id_zero_are_both_tombstoned() -> None:
    source = _nonconstant_binary_source("add")
    parent = canonicalize_shrink2_source_ast(source)
    assert parent.value[1][0:2] == (2, 0)  # type: ignore[index]

    _assert_error(lambda: canonicalize_shrink3_source_ast(source), REMOVED_BINARY)
    _assert_error(
        lambda: decode_shrink3_canonical_ast(parent.cbor_bytes),
        REMOVED_BINARY,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (["add", ["scalar_const", 1]], "REJECT_MALFORMED_SOURCE_AST"),
        (
            ["add", ["bit_at", 0], ["scalar_const", 1]],
            "REJECT_IMPLICIT_COERCION",
        ),
        (
            [
                "add",
                ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []],
                ["scalar_const", -2, 1],
            ],
            "REJECT_REMOVED_AGGREGATE_MAP",
        ),
        (
            ["add", ["scalar_const", -2, 1], ["scalar_const", 1, 1]],
            "REJECT_REMOVED_RATIONAL_PARAMETER",
        ),
        (_nonconstant_binary_source("add"), REMOVED_BINARY),
    ],
)
def test_source_parent_validation_and_tombstone_priority_is_exact(
    source: object, expected: str
) -> None:
    _assert_error(lambda: canonicalize_shrink3_source_ast(source), expected)


def test_nested_add_tombstone_precedes_parent_structural_limit_accounting() -> None:
    nested_add = _nonconstant_binary_source("add")
    oversized = [
        "difference",
        ["difference", nested_add, ["scalar_const", -1, 1]],
        [
            "difference",
            ["bit_to_scalar", ["bit_at", 1]],
            ["scalar_const", 1, 1],
        ],
    ]
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(oversized),
        "REJECT_STRUCTURAL_LIMIT",
    )
    _assert_error(lambda: canonicalize_shrink3_source_ast(oversized), REMOVED_BINARY)


@pytest.mark.parametrize(
    ("formal_value", "expected"),
    [
        (
            (1, (2, 0, (0, 0, 1))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (2, 0, (0, 1, 0), (0, 0, 1))),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            (1, (2, 0, (0, 3, 2, 0, 0, ()), (0, 0, 0))),
            "REJECT_REMOVED_AGGREGATE_MAP",
        ),
        (
            (1, (2, 0, (0, 0, 0), (0, 0, 1))),
            "REJECT_REMOVED_RATIONAL_PARAMETER",
        ),
        (
            (1, (2, 0, (0, 0, 1), (0, 0, 5))),
            REMOVED_BINARY,
        ),
        (
            (1, (2, 1, (1, 0, (0, 1, 0)), (0, 0, 3))),
            "REJECT_NONCANONICAL_AST",
        ),
    ],
)
def test_formal_structural_tombstone_and_normalization_priority_is_exact(
    formal_value: object, expected: str
) -> None:
    payload = canonical_cbor_encode(formal_value)
    _assert_error(lambda: decode_shrink3_canonical_ast(payload), expected)


@pytest.mark.parametrize(
    "formal_value",
    [
        (1, (4, ())),
        (1, (4, ((2, 0, (0, 0, 1), (0, 0, 5)),))),
        (
            1,
            (
                4,
                (
                    (0, 4, 0),
                    (0, 4, 1),
                    (0, 4, 2),
                    (2, 0, (0, 0, 1), (0, 0, 5)),
                ),
            ),
        ),
        (
            1,
            (
                2,
                0,
                (0, 3, 2, 0, 0, ((0, False), (1, False), (2, False))),
                (0, 0, 0),
            ),
        ),
        (
            1,
            (
                2,
                0,
                (0, 3, 2, 0, 0, ((1, False), (0, False))),
                (0, 0, 0),
            ),
        ),
        (
            1,
            (
                2,
                0,
                (0, 3, 2, 0, 0, ((0, False), (0, True))),
                (0, 0, 0),
            ),
        ),
    ],
)
def test_formal_shape_errors_precede_every_tombstone_scan(
    formal_value: object,
) -> None:
    _assert_error(
        lambda: decode_shrink3_canonical_ast(canonical_cbor_encode(formal_value)),
        "REJECT_NONCANONICAL_AST",
    )


@pytest.mark.parametrize(
    ("formal_value", "expected"),
    [
        (
            (1, (2, 4, (0, 3, 2, 0, 0, ()), (0, 0, 0))),
            "REJECT_REMOVED_AGGREGATE_MAP",
        ),
        (
            (1, (2, 4, (0, 0, 0), (0, 0, 1))),
            "REJECT_REMOVED_RATIONAL_PARAMETER",
        ),
        (
            (
                1,
                (
                    2,
                    4,
                    (2, 0, (1, 0, (0, 1, 0)), (0, 0, 1)),
                    (0, 0, 1),
                ),
            ),
            REMOVED_BINARY,
        ),
        (
            (1, (2, 4, (0, 0, 1), (0, 0, 5))),
            "REJECT_NONCANONICAL_AST",
        ),
    ],
)
def test_source_only_formal_alias_cannot_hide_a_higher_priority_tombstone(
    formal_value: object, expected: str
) -> None:
    _assert_error(
        lambda: decode_shrink3_canonical_ast(canonical_cbor_encode(formal_value)),
        expected,
    )


@pytest.mark.parametrize(
    "source",
    [
        ["scalar_const", -1, 1],
        ["absolute", ["scalar_const", -1, 1]],
        _nonconstant_binary_source("difference"),
        ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []],
        [
            "equal_exact",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1, 1],
        ],
    ],
)
def test_surviving_ast_cbor_bytes_and_hash_are_parent_stable(source: object) -> None:
    parent = canonicalize_shrink2_source_ast(source)
    child = canonicalize_shrink3_source_ast(source)
    assert child.cbor_bytes == parent.cbor_bytes
    assert child.hash_id == parent.hash_id
    assert decode_shrink3_canonical_ast(parent.cbor_bytes) == parent


def test_preregistered_shrink2_survivor_subset_is_byte_stable() -> None:
    sources = tuple(iter_shrink2_capacity_candidate_asts())
    assert len(sources) == EXPECTED_SHRINK2_SOURCE_COUNT == 2_160
    for source in sources:
        parent = canonicalize_shrink2_source_ast(source)
        child = canonicalize_shrink3_source_ast(source)
        assert child.cbor_bytes == parent.cbor_bytes
        assert child.hash_id == parent.hash_id
        assert decode_shrink3_canonical_ast(parent.cbor_bytes) == parent


def test_difference_retains_immutable_source_name_and_formal_id_one() -> None:
    source = _nonconstant_binary_source("difference")
    parent = canonicalize_shrink2_source_ast(source)
    child = canonicalize_shrink3_source_ast(source)
    assert child.value[1][0:2] == (2, 1)  # type: ignore[index]
    assert child == parent
    assert decode_shrink3_canonical_ast(child.cbor_bytes) == child


def test_legacy_reader_reports_add_without_rewrite_or_migration() -> None:
    removed = canonicalize_shrink2_source_ast(_nonconstant_binary_source("add"))
    report = read_legacy_parent_program(removed.cbor_bytes)
    assert report == {
        "legacy_program_status": "VALID_UNDER_PARENT_DSL_ONLY",
        "parent_dsl_version": "hegel-old-dsl-v1.2.0",
        "parent_effective_freeze_version": "hegel-freeze-p2b-p3-v1.2.0",
        "current_dsl_version": "hegel-old-dsl-v1.3.0",
        "canonical_ast_hash": removed.hash_id,
        "admitted_under_current_dsl": False,
        "current_dsl_error_code": REMOVED_BINARY,
        "automatic_operator_migration_performed": False,
    }

    retained = canonicalize_shrink2_source_ast(
        _nonconstant_binary_source("difference")
    )
    retained_report = read_legacy_parent_program(retained.cbor_bytes)
    assert retained_report["legacy_program_status"] == (
        "VALID_UNDER_PARENT_AND_CHILD_DSL"
    )
    assert retained_report["admitted_under_current_dsl"] is True
    assert retained_report["current_dsl_error_code"] is None
    assert retained_report["automatic_operator_migration_performed"] is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            ["unknown_outer", _nonconstant_binary_source("add")],
            "REJECT_UNKNOWN_EXPRESSION",
        ),
        (
            ["scalar_const", 1, _nonconstant_binary_source("add")],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
        (
            [
                "aggregate",
                0,
                0,
                0,
                [[0, _nonconstant_binary_source("add")]],
            ],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
    ],
)
def test_malformed_source_payload_is_not_blindly_scanned_for_add(
    source: object, expected: str
) -> None:
    with pytest.raises(StrictAstError) as parent_raised:
        canonicalize_shrink2_source_ast(source)
    with pytest.raises(StrictAstError) as child_raised:
        canonicalize_shrink3_source_ast(source)
    assert parent_raised.value.code == expected
    assert child_raised.value.code == parent_raised.value.code


@pytest.mark.parametrize(
    "formal_value",
    [
        (1, (0, 0, 1, (2, 0, (0, 0, 1), (0, 0, 5)))),
        (
            1,
            (
                0,
                3,
                0,
                0,
                0,
                ((0, (2, 0, (0, 0, 1), (0, 0, 5))),),
            ),
        ),
    ],
)
def test_malformed_formal_payload_is_not_blindly_scanned_for_add(
    formal_value: object,
) -> None:
    payload = canonical_cbor_encode(formal_value)
    with pytest.raises(StrictAstError) as parent_raised:
        decode_shrink2_canonical_ast(payload)
    with pytest.raises(StrictAstError) as child_raised:
        decode_shrink3_canonical_ast(payload)
    assert child_raised.value.code == parent_raised.value.code
    assert child_raised.value.code != REMOVED_BINARY

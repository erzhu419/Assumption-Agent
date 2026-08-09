from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hegel_machine.phase3_shrink2_capacity_v1 import (
    EXPECTED_SHRINK2_SOURCE_COUNT,
    iter_shrink2_capacity_candidate_asts,
    shrink2_constant_atoms_v1,
    shrink2_mixed_atoms_v1,
    shrink2_rational_aggregate_leaves_v1,
)
from hegel_machine.phase3_shrink2_registry_v1 import (
    ACTIVE_AGGREGATE_IDS,
    ACTIVE_RATIONAL_PARAMETER_IDS,
    DSL_VERSION,
    FREEZE_VERSION,
    HUMAN_AMENDMENT_ID,
    NEXT_ALLOCATABLE_RATIONAL_PARAMETER_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    RATIONAL_PARAMETER_CODE_SPACE_SIZE,
    RATIONAL_PARAMETER_CODE_WIDTH_BITS,
    RATIONAL_PARAMETER_REGISTRY,
    RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID,
    RESERVED_RATIONAL_PARAMETER_IDS,
    SHRINK_STEP_ID,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
    operator_admission_semantics_object,
    rational_parameter_registry_object,
    shrunk_dsl_surface_object,
)
from hegel_machine.strict_ast_shrink1_v1 import (
    canonicalize_shrink1_source_ast,
    decode_shrink1_canonical_ast,
)
from hegel_machine.strict_ast_shrink2_v1 import (
    canonicalize_shrink2_source_ast,
    decode_shrink2_canonical_ast,
    rational_parameter_id_is_active,
    read_legacy_parent_program,
)
from hegel_machine.strict_ast_v1 import StrictAstError, decode_canonical_ast
from hegel_machine.strict_cbor_v1 import StrictCborError, canonical_cbor_encode


def _assert_error(callable_: object, expected: str) -> None:
    with pytest.raises((StrictAstError, StrictCborError)) as raised:
        callable_()  # type: ignore[operator]
    assert raised.value.code == expected


def _rational_leaf_ids(value: tuple[object, ...]) -> tuple[int, ...]:
    found: list[int] = []

    def visit(node: object) -> None:
        assert isinstance(node, tuple)
        tag = node[0]
        if tag == 0:
            if node[1] == 0:
                found.append(node[2])  # type: ignore[arg-type]
            return
        if tag == 1:
            visit(node[2])
        elif tag in {2, 3}:
            visit(node[2])
            visit(node[3])
        else:
            assert tag == 4
            for child in node[1]:  # type: ignore[union-attr]
                visit(child)

    assert value[0] == 1
    visit(value[1])
    return tuple(found)


def test_shrink2_versions_and_sparse_rational_registry_are_exact() -> None:
    assert PARENT_DSL_VERSION == "hegel-old-dsl-v1.1.0"
    assert PARENT_FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.1.2"
    assert DSL_VERSION == "hegel-old-dsl-v1.2.0"
    assert FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.2.0"
    assert HUMAN_AMENDMENT_ID == "hegel-freeze-p2b-p3-v1.2.0-shrink-step2"
    assert SHRINK_STEP_ID == (
        "SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1"
    )
    assert RATIONAL_PARAMETER_CODE_WIDTH_BITS == 3
    assert RATIONAL_PARAMETER_CODE_SPACE_SIZE == 8
    assert ACTIVE_RATIONAL_PARAMETER_IDS == (1, 3, 5)
    assert TOMBSTONED_RATIONAL_PARAMETER_IDS == (0, 2, 4, 6)
    assert RESERVED_RATIONAL_PARAMETER_IDS == (7,)
    assert NEXT_ALLOCATABLE_RATIONAL_PARAMETER_ID is None
    assert tuple(entry.numeric_id for entry in RATIONAL_PARAMETER_REGISTRY) == tuple(
        range(7)
    )
    assert tuple(entry.state for entry in RATIONAL_PARAMETER_REGISTRY) == (
        "TOMBSTONE",
        "ACTIVE",
        "TOMBSTONE",
        "ACTIVE",
        "TOMBSTONE",
        "ACTIVE",
        "TOMBSTONE",
    )
    assert RATIONAL_PARAMETER_REGISTRY_DIAGNOSTIC_ID.startswith(
        "rational_parameter_registry_"
    )


def test_shrink1_aggregate_tombstones_and_all_other_freezes_are_inherited() -> None:
    assert ACTIVE_AGGREGATE_IDS == (0, 1, 5)
    assert TOMBSTONED_AGGREGATE_IDS == (2, 3, 4)
    registry = rational_parameter_registry_object()
    assert registry["id_compaction_allowed"] is False
    assert registry["id_reuse_allowed"] is False
    assert registry["reserved_out_of_range_ids"] == [7]
    semantics = operator_admission_semantics_object()
    assert semantics["removed_parameter_disposition"] == (
        "REJECT_REMOVED_RATIONAL_PARAMETER"
    )
    assert semantics["constant_fold_result_must_be_active"] is True
    assert semantics["inactive_fold_result_disposition"] == "RETAIN_OPERATOR_AST"
    for unchanged in (
        "typing_changed",
        "bottom_semantics_changed",
        "rewrite_rules_changed",
        "closure_budget_changed",
        "structural_limits_changed",
        "scope_catalog_changed",
        "target_or_control_semantics_changed",
        "mdl_code_table_changed",
    ):
        assert semantics[unchanged] is False
    surface = shrunk_dsl_surface_object()
    assert surface["pre_registered_delta_only"] == (
        "reduce RationalParameter to {-1,0,1}"
    )
    assert surface["execution_state"] == "NOT_RUN"
    assert surface["complete_closure_enumerated"] is False
    assert surface["formal_roots"] is None


@pytest.mark.parametrize("numeric_id", [1, 3, 5])
def test_active_rational_parameter_ids_remain_admitted(numeric_id: int) -> None:
    assert rational_parameter_id_is_active(numeric_id) is True


@pytest.mark.parametrize("numeric_id", [0, 2, 4, 6])
def test_tombstoned_rational_parameter_ids_use_exact_error(numeric_id: int) -> None:
    _assert_error(
        lambda: rational_parameter_id_is_active(numeric_id),
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    )


@pytest.mark.parametrize("numeric_id", [-1, 7, 8, True, "3"])
def test_reserved_or_unknown_rational_parameter_ids_stay_out_of_range(
    numeric_id: object,
) -> None:
    _assert_error(
        lambda: rational_parameter_id_is_active(numeric_id),  # type: ignore[arg-type]
        "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
    )


@pytest.mark.parametrize(
    "source",
    [
        ["scalar_const", 0],
        ["scalar_const", 2],
        ["scalar_const", 4],
        ["scalar_const", 6],
        ["scalar_const", -2, 1],
        ["scalar_const", -2, 4],
        ["scalar_const", 1, 2],
        ["scalar_const", 4, 2],
    ],
)
def test_removed_source_parameters_use_exact_error(source: object) -> None:
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(source),
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    )


@pytest.mark.parametrize(
    "source",
    [
        ["scalar_const", -1],
        ["scalar_const", -1, -1],
        ["scalar_const", -2, -1],
        ["scalar_const", 10**100],
        ["scalar_const", 10**100, 1],
        ["bit_at", -1],
        ["context_flag", -1],
        ["task_flag", -1],
        ["scalar_const", "bad-index"],
        ["bit_at", True],
        ["aggregate", False, "scope_all_observed_v1", "q0", []],
        ["context_flag", []],
        ["aggregate", -1, "scope_all_observed_v1", "q0", []],
        ["aggregate", "sum_v1", -1, "q0", []],
        ["aggregate", "sum_v1", "scope_all_observed_v1", -1, []],
        [
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            -1,
        ],
        [
            "approx_equal",
            ["scalar_const", 1],
            ["scalar_const", 5],
            -1,
            -4,
        ],
    ],
)
def test_source_numeric_domain_uses_exact_out_of_range_code(source: object) -> None:
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(source),
        "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
    )


def test_positive_denominator_alias_still_resolves_exactly() -> None:
    child = canonicalize_shrink2_source_ast(["scalar_const", -2, 2])
    assert child.value == (1, (0, 0, 1))
    wide = 10**100
    wide_child = canonicalize_shrink2_source_ast(["scalar_const", wide, wide])
    assert wide_child.value == (1, (0, 0, 5))


def test_noninteger_tolerance_shorthand_remains_malformed() -> None:
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(
            [
                "approx_equal",
                ["scalar_const", 1],
                ["scalar_const", 5],
                "not-an-index",
            ]
        ),
        "REJECT_MALFORMED_SOURCE_AST",
    )


def test_aggregate_tombstone_has_global_priority_after_valid_source_parse() -> None:
    source = [
        "less_equal",
        ["scalar_const", 0],
        [
            "aggregate",
            "mean_v1",
            "scope_all_observed_v1",
            "q0",
            [],
        ],
    ]
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(source),
        "REJECT_REMOVED_AGGREGATE_MAP",
    )


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (["scalar_const", 7], "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"),
        (
            ["scalar_const", 0, ["scalar_const", -2, 1]],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
        (
            ["unknown_outer", ["scalar_const", -2, 1]],
            "REJECT_UNKNOWN_EXPRESSION",
        ),
        (
            ["top_level_AND", [[], ["scalar_const", -2, 1]]],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
    ],
)
def test_source_precheck_never_scans_arbitrary_payloads(
    source: object, expected_code: str
) -> None:
    with pytest.raises(StrictAstError) as parent_raised:
        canonicalize_shrink1_source_ast(source)
    with pytest.raises(StrictAstError) as child_raised:
        canonicalize_shrink2_source_ast(source)
    assert parent_raised.value.code == expected_code
    assert child_raised.value.code == parent_raised.value.code


@pytest.mark.parametrize(
    "source",
    [
        ["scalar_const", -1, 1],
        ["scalar_const", 0, 1],
        ["scalar_const", 1, 1],
        ["absolute", ["scalar_const", -1, 1]],
        [
            "add",
            ["bit_to_scalar", ["bit_at", 0]],
            ["scalar_const", 1, 1],
        ],
        ["aggregate", "sum_v1", "scope_all_observed_v1", "q0", []],
    ],
)
def test_surviving_parent_ast_bytes_and_hashes_are_stable(source: object) -> None:
    parent = canonicalize_shrink1_source_ast(source)
    child = canonicalize_shrink2_source_ast(source)
    assert child.cbor_bytes == parent.cbor_bytes
    assert child.hash_id == parent.hash_id
    assert decode_shrink2_canonical_ast(parent.cbor_bytes) == parent


@pytest.mark.parametrize(
    "source",
    [
        ["add", ["scalar_const", 1, 1], ["scalar_const", 1, 1]],
        ["add", ["scalar_const", -1, 1], ["scalar_const", -1, 1]],
        ["difference", ["scalar_const", -1, 1], ["scalar_const", 1, 1]],
        ["difference", ["scalar_const", 1, 1], ["scalar_const", -1, 1]],
    ],
)
def test_active_input_folds_never_reintroduce_tombstones(source: object) -> None:
    parent = canonicalize_shrink1_source_ast(source)
    assert _rational_leaf_ids(parent.value) in {(0,), (6,)}
    child = canonicalize_shrink2_source_ast(source)
    assert child.value[1][0] == 2  # type: ignore[index]
    assert set(_rational_leaf_ids(child.value)) <= set(ACTIVE_RATIONAL_PARAMETER_IDS)
    assert decode_shrink2_canonical_ast(child.cbor_bytes) == child


def test_active_fold_to_active_zero_is_still_performed() -> None:
    source = ["add", ["scalar_const", 1, 1], ["scalar_const", -1, 1]]
    parent = canonicalize_shrink1_source_ast(source)
    child = canonicalize_shrink2_source_ast(source)
    assert child == parent
    assert child.value == (1, (0, 0, 3))


def test_all_active_constant_binary_pairs_are_closed_under_child_identity() -> None:
    sources = {
        1: ["scalar_const", -1, 1],
        3: ["scalar_const", 0, 1],
        5: ["scalar_const", 1, 1],
    }
    for operator in ("add", "difference"):
        for left_id, left in sources.items():
            for right_id, right in sources.items():
                source = [operator, left, right]
                child = canonicalize_shrink2_source_ast(source)
                assert decode_shrink2_canonical_ast(child.cbor_bytes) == child
                assert set(_rational_leaf_ids(child.value)) <= set(
                    ACTIVE_RATIONAL_PARAMETER_IDS
                )
                parent = canonicalize_shrink1_source_ast(source)
                if set(_rational_leaf_ids(parent.value)) <= set(
                    ACTIVE_RATIONAL_PARAMETER_IDS
                ):
                    assert child.cbor_bytes == parent.cbor_bytes
                    assert child.hash_id == parent.hash_id
                else:
                    assert child.value[1][0] == 2  # type: ignore[index]
                    assert set(_rational_leaf_ids(child.value)) == {
                        left_id,
                        right_id,
                    }


def test_nested_add_is_commutative_deterministic_without_tombstone_folds() -> None:
    negative = ["scalar_const", -1, 1]
    positive = ["scalar_const", 1, 1]
    first = canonicalize_shrink2_source_ast(
        ["add", ["add", positive, positive], negative]
    )
    second = canonicalize_shrink2_source_ast(
        ["add", negative, ["add", positive, positive]]
    )
    assert first.cbor_bytes == second.cbor_bytes
    assert first.hash_id == second.hash_id
    assert set(_rational_leaf_ids(first.value)) <= set(ACTIVE_RATIONAL_PARAMETER_IDS)
    assert decode_shrink2_canonical_ast(first.cbor_bytes) == first


@pytest.mark.parametrize("numeric_id", [0, 2, 4, 6])
def test_formal_parent_programs_with_removed_parameters_are_rejected(
    numeric_id: int,
) -> None:
    parent = canonicalize_shrink1_source_ast(["scalar_const", numeric_id])
    assert decode_shrink1_canonical_ast(parent.cbor_bytes) == parent
    _assert_error(
        lambda: decode_shrink2_canonical_ast(parent.cbor_bytes),
        "REJECT_REMOVED_RATIONAL_PARAMETER",
    )
    legacy = read_legacy_parent_program(parent.cbor_bytes)
    assert legacy["legacy_program_status"] == "VALID_UNDER_PARENT_DSL_ONLY"
    assert legacy["current_dsl_error_code"] == "REJECT_REMOVED_RATIONAL_PARAMETER"
    assert legacy["automatic_parameter_migration_performed"] is False


@pytest.mark.parametrize(
    "formal_value",
    [
        (2, (0, 0, 0)),
        (1, (0, 0, 0, 99)),
        (1, (2, 4, (0, 0, 0), (0, 0, 3))),
    ],
)
def test_formal_precheck_requires_a_structurally_legal_ast_leaf(
    formal_value: object,
) -> None:
    payload = canonical_cbor_encode(formal_value)
    with pytest.raises(StrictAstError) as parent_raised:
        decode_canonical_ast(payload)
    with pytest.raises(StrictAstError) as child_raised:
        decode_shrink2_canonical_ast(payload)
    assert child_raised.value.code == parent_raised.value.code
    assert child_raised.value.code != "REJECT_REMOVED_RATIONAL_PARAMETER"


@pytest.mark.parametrize(
    ("formal_value", "expected_code"),
    [
        ((1, (-1,)), "REJECT_UNKNOWN_EXPRESSION"),
        ((1, (0, -1)), "REJECT_UNKNOWN_EXPRESSION"),
        ((1, (0, 0, -1)), "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"),
        ((1, (1, 4, (0, 0, 3))), "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"),
        (
            (1, (2, 7, (0, 0, 3), (0, 0, 3))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (3, 1, (0, 0, 3), (0, 0, 3), 0)),
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
        (
            (1, (4, ((0, 4, 0), (0, 4, 1), (0, 4, 2), (0, 4, 3)))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (0, 3, 0, 0, 0, ((0, False), (0, True)))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (0, 3, 0, 0, 0, ((0, False), (1, False), (2, False)))),
            "REJECT_NONCANONICAL_AST",
        ),
        (
            (1, (0, 3, 99, 0, 0, ((0,),))),
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
        (
            (1, (2, 2, (0, 0, 0), (0, 0, 3, 99))),
            "REJECT_NONCANONICAL_AST",
        ),
    ],
)
def test_formal_failure_code_and_tombstone_priority_matrix_is_exact(
    formal_value: object, expected_code: str
) -> None:
    _assert_error(
        lambda: decode_shrink2_canonical_ast(canonical_cbor_encode(formal_value)),
        expected_code,
    )


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (["top_level_AND"], "REJECT_EMPTY_CONJUNCTION"),
        (["sign", ["bit_at", 0]], "REJECT_TYPE_MISMATCH"),
        (
            ["add", ["set_size"], ["bit_at", 0]],
            "REJECT_IMPLICIT_COERCION",
        ),
        (
            ["approx_equal", ["scalar_const", 1], ["bit_at", 0], 1],
            "REJECT_TYPE_MISMATCH",
        ),
        (
            ["add", ["set_size"], ["not_in_old_dsl"]],
            "REJECT_UNKNOWN_EXPRESSION",
        ),
        (
            ["add", ["sign", ["bit_at", 0]], ["not_in_old_dsl"]],
            "REJECT_TYPE_MISMATCH",
        ),
        (
            ["aggregate", 99, 0, 0, [[0]]],
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
        ),
        (
            ["aggregate", 0, 0, 0, [[99, "not-a-bool"]]],
            "REJECT_MALFORMED_SOURCE_AST",
        ),
        (
            ["new_symbol_call", ["scalar_const", 99]],
            "REJECT_NEW_SYMBOL_IN_OLD_DSL",
        ),
    ],
)
def test_source_left_to_right_failure_priority_is_exact(
    source: object, expected_code: str
) -> None:
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(source),
        expected_code,
    )


@pytest.mark.parametrize(
    ("payload_hex", "expected_code"),
    [
        ("", "REJECT_TRUNCATED_CBOR"),
        ("8201", "REJECT_TRUNCATED_CBOR"),
        ("1901", "REJECT_TRUNCATED_CBOR"),
        ("1c", "REJECT_RESERVED_CBOR"),
        ("f7", "REJECT_CBOR_UNDEFINED"),
        ("e0", "REJECT_CBOR_SIMPLE"),
    ],
)
def test_formal_cbor_failure_taxonomy_is_exact(
    payload_hex: str, expected_code: str
) -> None:
    _assert_error(
        lambda: decode_shrink2_canonical_ast(bytes.fromhex(payload_hex)),
        expected_code,
    )


@pytest.mark.parametrize(
    ("formal_value", "expected_code"),
    [
        ((1, (0, 6)), "REJECT_NEW_SYMBOL_IN_OLD_DSL"),
        ((1, (0, 6, -10)), "REJECT_NEW_SYMBOL_IN_OLD_DSL"),
        ((1, (1, 3, (0, 1, 0))), "REJECT_TYPE_MISMATCH"),
        (
            (1, (2, 0, (1, 3, (0, 1, 0)), (0, 99))),
            "REJECT_TYPE_MISMATCH",
        ),
        (
            (1, (3, 0, (0, 0, 1), (0, 1, 0), 99)),
            "REJECT_TYPE_MISMATCH",
        ),
    ],
)
def test_formal_subtree_failure_priority_precedes_later_fields(
    formal_value: object, expected_code: str
) -> None:
    _assert_error(
        lambda: decode_shrink2_canonical_ast(canonical_cbor_encode(formal_value)),
        expected_code,
    )


def test_inherited_aggregate_tombstones_remain_rejected_on_both_boundaries() -> None:
    source = ["aggregate", "mean_v1", "scope_all_observed_v1", "q0", []]
    _assert_error(
        lambda: canonicalize_shrink2_source_ast(source),
        "REJECT_REMOVED_AGGREGATE_MAP",
    )
    parent_bytes = decode_canonical_ast(
        canonical_cbor_encode((1, (0, 3, 2, 0, 0, ())))
    ).cbor_bytes
    _assert_error(
        lambda: decode_shrink2_canonical_ast(parent_bytes),
        "REJECT_REMOVED_AGGREGATE_MAP",
    )


def test_preregistered_shrink2_constructive_subset_is_exact_and_roundtrips() -> None:
    assert len(shrink2_constant_atoms_v1()) == 15
    assert len(shrink2_rational_aggregate_leaves_v1()) == 16
    assert len(shrink2_mixed_atoms_v1()) == 144
    sources = tuple(iter_shrink2_capacity_candidate_asts())
    assert len(sources) == EXPECTED_SHRINK2_SOURCE_COUNT == 2_160

    programs = tuple(canonicalize_shrink2_source_ast(source) for source in sources)
    assert len({program.cbor_bytes for program in programs}) == 2_160
    assert all(
        decode_shrink2_canonical_ast(program.cbor_bytes) == program
        for program in programs
    )
    assert all(
        set(_rational_leaf_ids(program.value)) <= set(ACTIVE_RATIONAL_PARAMETER_IDS)
        for program in programs
    )


def test_capacity_import_does_not_instantiate_target_or_split_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    entrypoint = (
        project_root
        / "src/hegel_machine/phase3_shrink2_capacity_entrypoint_v1.py"
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(entrypoint),
            "--capacity-replay",
        ],
        cwd=project_root,
        env={"PATH": os.environ.get("PATH", "")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    report = json.loads(probe.stdout)
    assert report["target_or_split_modules_loaded"] is False
    assert report["human_amendment_id"] == HUMAN_AMENDMENT_ID
    assert report["shrink_step_id"] == SHRINK_STEP_ID
    assert "hegel_machine.phase3_dsl_v1" not in report["loaded_hegel_modules"]
    assert (
        "hegel_machine.phase3_shrink1_registry_v1"
        not in report["loaded_hegel_modules"]
    )

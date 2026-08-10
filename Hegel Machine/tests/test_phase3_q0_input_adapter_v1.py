from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path

import pytest

from hegel_machine import phase3_q0_input_adapter_v1 as adapter
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


def _odd(bits: tuple[int, ...]) -> tuple[object, ...]:
    return (
        1,
        adapter.ODD_INPUT_TAG,
        adapter.ODD_INPUT_SCHEMA_ID,
        len(bits),
        bits,
    )


def _sink(a: int, b: int, c: int, d: int) -> tuple[object, ...]:
    return (
        1,
        adapter.SINK_INPUT_TAG,
        adapter.SINK_INPUT_SCHEMA_ID,
        a,
        b,
        c,
        d,
    )


def _error_code(callable_, *args: object) -> str:
    with pytest.raises(adapter.InputAdapterError) as caught:
        callable_(*args)
    return caught.value.code


def test_odd_cbor_decodes_to_frozen_observations_only() -> None:
    raw = _odd((1, 0, 1, 1, 0))
    environment = adapter.decode_observation_environment_v1(
        canonical_cbor_encode(raw)
    )

    assert environment.input_signature_id == adapter.ODD_INPUT_SIGNATURE_ID
    assert environment.canonical_input_object == raw
    assert environment.set_size == 5
    assert tuple(entity.bit for entity in environment.entities) == (1, 0, 1, 1, 0)
    assert adapter.evaluate_canonical_leaf_v1(environment, (0, 1, 0)) == 1
    assert adapter.evaluate_canonical_leaf_v1(environment, (0, 2)) == 5

    with pytest.raises(FrozenInstanceError):
        environment.set_size = 6  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        environment.entities[0].bit = 0  # type: ignore[misc]


def test_odd_out_of_range_and_all_unpublished_channels_are_bottom() -> None:
    environment = adapter.observation_environment_from_object_v1(
        _odd((1, 0, 1, 0, 1))
    )

    assert environment.bit_at(5) is adapter.BOTTOM
    assert environment.bit_at(7) is adapter.BOTTOM
    assert environment.context_flag(0) is adapter.BOTTOM
    assert environment.task_flag(0) is adapter.BOTTOM
    assert environment.entities[0].quantity(0) is adapter.BOTTOM
    assert environment.entities[0].membership(0) is adapter.BOTTOM
    assert environment.entities[0].role_id is adapter.BOTTOM
    assert environment.entities[0].orientation is adapter.BOTTOM
    assert adapter.evaluate_environment_aggregate_v1(environment, 0, 3, 0, ()) is adapter.BOTTOM
    assert adapter.evaluate_environment_aggregate_v1(environment, 1, 0, 1, ()) is adapter.BOTTOM

    # Missing observations are a distinct singleton, never a guessed scalar or
    # boolean value.
    assert type(adapter.BOTTOM) is adapter.BottomValueV1
    assert adapter.BOTTOM != 0
    assert adapter.BOTTOM is not False


def test_sink_environment_freezes_q0_roles_scope_and_orientations() -> None:
    environment = adapter.decode_observation_environment_v1(
        canonical_cbor_encode(_sink(1, 4, 2, 3))
    )

    assert environment.input_signature_id == adapter.SINK_INPUT_SIGNATURE_ID
    assert environment.set_size == 4
    assert tuple(entity.quantity(0) for entity in environment.entities) == (
        Fraction(1),
        Fraction(4),
        Fraction(2),
        Fraction(3),
    )
    assert tuple(entity.role_id for entity in environment.entities) == (0, 1, 2, 3)
    assert tuple(entity.orientation for entity in environment.entities) == (1, 1, -1, -1)
    assert tuple(entity.membership(3) for entity in environment.entities) == (
        True,
        True,
        True,
        True,
    )

    for entity in environment.entities:
        assert entity.bit is adapter.BOTTOM
        assert entity.quantity(1) is adapter.BOTTOM
        assert entity.membership(0) is adapter.BOTTOM
        assert entity.membership(1) is adapter.BOTTOM
        assert entity.membership(2) is adapter.BOTTOM
    assert environment.bit_at(0) is adapter.BOTTOM
    assert environment.context_flag(3) is adapter.BOTTOM
    assert environment.task_flag(1) is adapter.BOTTOM


def test_sink_designated_signed_balance_is_exactly_zero() -> None:
    environment = adapter.observation_environment_from_object_v1(
        _sink(1, 4, 2, 3)
    )

    assert adapter.evaluate_environment_aggregate_v1(environment, 0, 3, 0, ()) == Fraction(10)
    assert adapter.evaluate_environment_aggregate_v1(environment, 1, 3, 0, ()) == 4
    assert adapter.evaluate_environment_aggregate_v1(environment, 5, 3, 0, ()) == Fraction(0)
    assert adapter.evaluate_canonical_leaf_v1(
        environment,
        (0, 3, 5, 3, 0, ()),
    ) == Fraction(0)


def test_sink_unpublished_quantity_scope_or_extension_is_bottom() -> None:
    environment = adapter.observation_environment_from_object_v1(
        _sink(0, 1, 0, 1)
    )

    assert adapter.evaluate_environment_aggregate_v1(environment, 5, 3, 1, ()) is adapter.BOTTOM
    for scope_id in (0, 1, 2):
        assert (
            adapter.evaluate_environment_aggregate_v1(
                environment,
                5,
                scope_id,
                0,
                (),
            )
            is adapter.BOTTOM
        )
    assert (
        adapter.evaluate_environment_aggregate_v1(
            environment,
            5,
            3,
            0,
            ((0, True),),
        )
        is adapter.BOTTOM
    )


def test_aggregate_maps_are_bottom_strict_and_rational_grid_bounded() -> None:
    assert adapter.evaluate_aggregate_values_v1(
        0,
        (Fraction(1), adapter.BOTTOM),
    ) is adapter.BOTTOM
    assert adapter.evaluate_aggregate_values_v1(
        5,
        (Fraction(1),),
    ) is adapter.BOTTOM
    assert adapter.evaluate_aggregate_values_v1(
        0,
        (Fraction(64), Fraction(1)),
    ) is adapter.BOTTOM
    assert adapter.evaluate_aggregate_values_v1(
        0,
        (Fraction(1, 9),),
    ) is adapter.BOTTOM
    assert adapter.evaluate_aggregate_values_v1(
        1,
        tuple(Fraction(1) for _ in range(9)),
    ) is adapter.BOTTOM

    assert _error_code(adapter.evaluate_aggregate_values_v1, 2, ()) == (
        adapter.REJECT_REMOVED_AGGREGATE_MAP
    )


def test_leaf_dispatch_uses_sparse_parameter_ids_and_strict_arity() -> None:
    environment = adapter.observation_environment_from_object_v1(
        _odd((0, 0, 0, 0, 0))
    )

    assert adapter.evaluate_leaf_v1(environment, 0, (1,)) == Fraction(-1)
    assert adapter.evaluate_leaf_v1(environment, 0, (3,)) == Fraction(0)
    assert adapter.evaluate_leaf_v1(environment, 0, (5,)) == Fraction(1)
    assert adapter.evaluate_leaf_v1(environment, 4, (0,)) is adapter.BOTTOM
    assert adapter.evaluate_leaf_v1(environment, 5, (0,)) is adapter.BOTTOM
    assert _error_code(adapter.evaluate_leaf_v1, environment, 0, (0,)) == (
        adapter.REJECT_REMOVED_RATIONAL_PARAMETER
    )
    assert _error_code(adapter.evaluate_leaf_v1, environment, 6, ()) == (
        adapter.REJECT_NEW_SYMBOL_IN_OLD_DSL
    )
    assert _error_code(adapter.evaluate_leaf_v1, environment, 2, (0,)) == (
        adapter.REJECT_MALFORMED_CANONICAL_LEAF
    )


@pytest.mark.parametrize(
    ("value", "expected_code"),
    (
        ((1, 0x9999, adapter.ODD_INPUT_SCHEMA_ID, 5, (0, 0, 0, 0, 0)), adapter.REJECT_TYPED_INPUT_PREFIX),
        ((1, adapter.ODD_INPUT_TAG, b"wrong", 5, (0, 0, 0, 0, 0)), adapter.REJECT_TYPED_INPUT_PREFIX),
        ((1, adapter.ODD_INPUT_TAG, adapter.ODD_INPUT_SCHEMA_ID, 4, (0, 0, 0, 0)), adapter.REJECT_ODD_SET_SIZE),
        ((1, adapter.ODD_INPUT_TAG, adapter.ODD_INPUT_SCHEMA_ID, 5, (0, 1)), adapter.REJECT_ODD_BIT_COUNT),
        ((1, adapter.ODD_INPUT_TAG, adapter.ODD_INPUT_SCHEMA_ID, 5, (0, 0, True, 0, 0)), adapter.REJECT_ODD_BIT_TYPE),
        ((1, adapter.SINK_INPUT_TAG, adapter.SINK_INPUT_SCHEMA_ID, 0, 0, 0, 1), adapter.REJECT_SINK_BALANCE),
        ((1, adapter.SINK_INPUT_TAG, adapter.SINK_INPUT_SCHEMA_ID, 0, 0, 0, True), adapter.REJECT_SINK_VALUE),
    ),
)
def test_typed_input_parser_fails_closed(value: object, expected_code: str) -> None:
    assert _error_code(adapter.observation_environment_from_object_v1, value) == expected_code


def test_adapter_import_surface_has_no_fixture_or_answer_dependency() -> None:
    source_path = Path(adapter.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    forbidden_fragments = (
        "phase3_dsl_v1",
        "phase3_m25_rows_v1",
        "static_basis",
        "truth",
        "split",
        "target",
    )
    assert not {
        module_name
        for module_name in imported
        if any(fragment in module_name for fragment in forbidden_fragments)
    }
    assert imported <= {
        "__future__",
        "dataclasses",
        "enum",
        "fractions",
        "typing",
        "strict_cbor_v1",
    }

"""Target-blind scalar evaluator for the Phase-3A-Q0 old-DSL projection.

This module intentionally sits below the quotient wire contract.  It consumes
only an admitted strict AST and an observation environment, and returns one
typed exact value or the adapter's unique bottom sentinel.  It has no access
to target truth, split assignments, role labels, or quotient classes, which
lets the contract independently replay every archived representative without
introducing a contract/oracle import cycle.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Final, NoReturn

from . import phase3_q0_input_adapter_v1 as _adapter
from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast
from .strict_ast_v1 import CanonicalAst


_UNARY_NAMES: Final = (
    "bit_to_scalar",
    "int_to_scalar",
    "absolute",
    "sign",
)
_BINARY_NAMES: Final = {
    1: "difference",
    2: "equal_exact",
    3: "less_equal",
    5: "same_sign",
    6: "opposite_sign",
}
_TOLERANCES: Final = (Fraction(0), Fraction(1, 4), Fraction(1, 2))


class Q0EvaluatorError(ValueError):
    """Stable fail-closed error from target-blind Q0 scalar replay."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q0EvaluatorError(code, detail)


def _bottom(value: object) -> bool:
    return value is _adapter.BOTTOM


def _require_fraction(value: object, operator: str) -> Fraction:
    if type(value) is not Fraction or value not in _adapter.RATIONAL_VALUE_GRID:
        _fail("FAIL_Q0_EVALUATOR_SORT", f"{operator} requires RationalValue")
    return value


def _bounded_rational(value: Fraction) -> object:
    return value if value in _adapter.RATIONAL_VALUE_GRID else _adapter.BOTTOM


def _evaluate_node_v1(
    node: tuple[object, ...],
    environment: _adapter.ObservationEnvironment,
) -> object:
    tag = node[0]
    if tag == 0:
        return _adapter.evaluate_canonical_leaf_v1(environment, node)
    if tag == 1:
        operator = node[1]
        child = _evaluate_node_v1(node[2], environment)  # type: ignore[arg-type]
        if _bottom(child):
            return _adapter.BOTTOM
        if operator == 0:
            if type(child) is not int or child not in (0, 1):
                _fail("FAIL_Q0_EVALUATOR_SORT", "bit_to_scalar requires Bit")
            return Fraction(child, 1)
        if operator == 1:
            if type(child) is not int or not -8 <= child <= 8:
                _fail(
                    "FAIL_Q0_EVALUATOR_SORT",
                    "int_to_scalar requires BoundedInt",
                )
            return Fraction(child, 1)
        if type(operator) is not int or operator not in (2, 3):
            _fail("FAIL_Q0_EVALUATOR_OPERATOR", f"unknown unary ID {operator!r}")
        value = _require_fraction(child, _UNARY_NAMES[operator])
        if operator == 2:
            return _bounded_rational(abs(value))
        return -1 if value < 0 else 1 if value > 0 else 0
    if tag == 2:
        operator = node[1]
        left = _evaluate_node_v1(node[2], environment)  # type: ignore[arg-type]
        right = _evaluate_node_v1(node[3], environment)  # type: ignore[arg-type]
        if _bottom(left) or _bottom(right):
            return _adapter.BOTTOM
        if operator in (1, 2, 3):
            exact_left = _require_fraction(left, _BINARY_NAMES[operator])
            exact_right = _require_fraction(right, _BINARY_NAMES[operator])
            if operator == 1:
                return _bounded_rational(exact_left - exact_right)
            if operator == 2:
                return exact_left == exact_right
            return exact_left <= exact_right
        if operator in (5, 6):
            if (
                type(left) is not int
                or type(right) is not int
                or left not in (-1, 0, 1)
                or right not in (-1, 0, 1)
            ):
                _fail(
                    "FAIL_Q0_EVALUATOR_SORT",
                    f"{_BINARY_NAMES[operator]} requires Sign",
                )
            if operator == 5:
                return left == right
            return left == -right and left != 0
        _fail("FAIL_Q0_EVALUATOR_OPERATOR", f"unknown binary ID {operator!r}")
    if tag == 3:
        left = _evaluate_node_v1(node[2], environment)  # type: ignore[arg-type]
        right = _evaluate_node_v1(node[3], environment)  # type: ignore[arg-type]
        if _bottom(left) or _bottom(right):
            return _adapter.BOTTOM
        exact_left = _require_fraction(left, "approx_equal")
        exact_right = _require_fraction(right, "approx_equal")
        tolerance_id = node[4]
        if type(tolerance_id) is not int or tolerance_id not in (1, 2):
            _fail(
                "FAIL_Q0_EVALUATOR_OPERATOR",
                "Q0 admits tolerance IDs 1 and 2 only",
            )
        return abs(exact_left - exact_right) <= _TOLERANCES[tolerance_id]
    if tag == 4:
        raw_children = node[1]
        if not isinstance(raw_children, tuple):
            _fail("FAIL_Q0_EVALUATOR_OPERATOR", "AND children must be an array")
        values = tuple(
            _evaluate_node_v1(child, environment) for child in raw_children
        )
        if any(_bottom(value) for value in values):
            return _adapter.BOTTOM
        if len(values) != 2 or any(type(value) is not bool for value in values):
            _fail("FAIL_Q0_EVALUATOR_SORT", "Q0 AND2 requires two Bool children")
        return values[0] and values[1]
    _fail("FAIL_Q0_EVALUATOR_OPERATOR", f"unknown canonical tag {tag!r}")


def evaluate_canonical_ast_raw_v1(
    ast: CanonicalAst,
    environment: _adapter.ObservationEnvironment,
) -> object:
    """Replay one admitted AST against one target-blind observation row."""

    if not isinstance(ast, CanonicalAst):
        raise TypeError("ast must be CanonicalAst")
    if not isinstance(environment, _adapter.ObservationEnvironment):
        raise TypeError("environment must be ObservationEnvironment")
    replay = decode_shrink6_canonical_ast(ast.cbor_bytes)
    if replay.digest != ast.digest or replay.cbor_bytes != ast.cbor_bytes:
        _fail("FAIL_Q0_AST_IDENTITY", "strict AST replay identity differs")
    node = replay.value[1]
    if not isinstance(node, tuple):  # pragma: no cover - strict decoder closes it
        _fail("FAIL_Q0_AST_IDENTITY", "strict AST node is not an array")
    return _evaluate_node_v1(node, environment)


def evaluate_canonical_ast_on_environments_v1(
    ast: CanonicalAst,
    environments: tuple[_adapter.ObservationEnvironment, ...],
) -> tuple[object, ...]:
    """Return the ordered raw output vector for an exact environment tuple."""

    if type(environments) is not tuple or any(
        not isinstance(environment, _adapter.ObservationEnvironment)
        for environment in environments
    ):
        raise TypeError("environments must be a tuple of ObservationEnvironment")
    return tuple(
        evaluate_canonical_ast_raw_v1(ast, environment)
        for environment in environments
    )


__all__ = [
    "Q0EvaluatorError",
    "evaluate_canonical_ast_on_environments_v1",
    "evaluate_canonical_ast_raw_v1",
]

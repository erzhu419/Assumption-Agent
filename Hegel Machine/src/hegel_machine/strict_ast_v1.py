"""Strict canonical AST and normalize-before-count implementation.

This module implements the v1.0.2 numeric-tag AST wire independently of the
older diagnostic tuple representation.  Source ASTs may use readable names,
but program identity is *only* ``CanonicalAstV1`` deterministic-CBOR bytes.

The canonicalizer applies the finite rewrite list before enforcing the frozen
depth/node/resource limits.  It never inserts a coercion and never performs an
extensional, SMT-derived, or target-aware simplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import Final, Iterable, Sequence

from .phase3_dsl_v1 import (
    AGGREGATE_MAP_IDS,
    CONTEXT_IDS,
    QUANTITY_IDS,
    RATIONAL_PARAMETER_GRID,
    SCOPE_IDS,
    STRUCTURAL_LIMITS,
    TASK_IDS,
)
from .strict_cbor_v1 import (
    StrictCborError,
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
)


AST_SCHEMA_ID: Final = "hegel-canonical-ast-v1"
AST_SCHEMA_VERSION: Final = 1
AST_HASH_DOMAIN: Final = "HEGEL/AST/V1"
DEPRECATED_SCOPE_ALIAS: Final = "control_volume_primary_only_v1"
CANONICAL_PRIMARY_SCOPE: Final = "scope_primary_only_v1"

LEAF_IDS: Final = {
    "scalar_const": 0,
    "bit_at": 1,
    "set_size": 2,
    "aggregate": 3,
    "context_flag": 4,
    "task_flag": 5,
    "new_symbol_call": 6,
}
UNARY_IDS: Final = {
    "bit_to_scalar": 0,
    "int_to_scalar": 1,
    "absolute": 2,
    "sign": 3,
}
BINARY_IDS: Final = {
    "add": 0,
    "difference": 1,
    "equal_exact": 2,
    "less_equal": 3,
    "greater_equal": 4,
    "same_sign": 5,
    "opposite_sign": 6,
}
TERNARY_IDS: Final = {"approx_equal": 0}

_PARAMETER_INDEX: Final = {
    (atom.numerator, atom.denominator): index
    for index, atom in enumerate(RATIONAL_PARAMETER_GRID)
}
_PARAMETER_VALUE: Final = tuple(
    Fraction(atom.numerator, atom.denominator) for atom in RATIONAL_PARAMETER_GRID
)
_TOLERANCE_VALUE: Final = (Fraction(0), Fraction(1, 4), Fraction(1, 2))


class StrictAstError(ValueError):
    """Stable strict-AST rejection with a machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _reject(code: str, detail: str) -> "None":
    raise StrictAstError(code, detail)


@dataclass(frozen=True, slots=True)
class _Expr:
    tag: int
    operator_id: int
    sort: str
    children: tuple["_Expr", ...] = ()
    parameters: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class AstMetrics:
    output_sort: str
    depth: int
    node_count: int
    scalar_parameter_occurrences: int
    aggregate_leaf_count: int
    distinct_bit_slots: frozenset[int]
    scope_clause_count: int
    top_level_clause_count: int


@dataclass(frozen=True, slots=True)
class CanonicalAst:
    """An immutable accepted old-DSL program identity."""

    value: tuple[object, ...]
    cbor_bytes: bytes
    digest: bytes
    root_operator_id: int
    metrics: AstMetrics

    @property
    def hash_id(self) -> str:
        return "sha256:" + self.digest.hex()


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        _reject("REJECT_MALFORMED_SOURCE_AST", f"{name} must be an array")
    return tuple(value)


def _uint(value: object, name: str, upper_exclusive: int) -> int:
    if type(value) is not int:
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", f"{name} must be uint")
    if value < 0 or value >= upper_exclusive:
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", f"{name} is out of range")
    return value


def _registry_index(value: object, registry: Sequence[str], name: str) -> int:
    if type(value) is int:
        return _uint(value, name, len(registry))
    if not isinstance(value, str) or value not in registry:
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", f"unknown {name}: {value!r}")
    return registry.index(value)


def _parameter_index(parts: tuple[object, ...]) -> int:
    if len(parts) == 1:
        return _uint(parts[0], "rational_parameter_index", len(_PARAMETER_VALUE))
    if len(parts) != 2 or any(type(item) is not int for item in parts):
        _reject(
            "REJECT_MALFORMED_SOURCE_AST",
            "scalar_const needs an index or integer numerator/denominator",
        )
    numerator, denominator = parts
    assert type(numerator) is int and type(denominator) is int
    if denominator <= 0:
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", "denominator must be positive")
    reduced = Fraction(numerator, denominator)
    index = _PARAMETER_INDEX.get((reduced.numerator, reduced.denominator))
    if index is None:
        _reject(
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
            "scalar literal is outside the frozen RationalParameter grid",
        )
    return index


def _tolerance_index(value: object) -> int:
    if type(value) is int:
        return _uint(value, "tolerance_index", len(_TOLERANCE_VALUE))
    pair = _sequence(value, "tolerance")
    if len(pair) != 2 or any(type(item) is not int for item in pair):
        _reject("REJECT_MALFORMED_SOURCE_AST", "tolerance must be an index or pair")
    if pair[1] <= 0:  # type: ignore[operator]
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", "tolerance denominator must be positive")
    exact = Fraction(pair[0], pair[1])  # type: ignore[arg-type]
    if exact not in _TOLERANCE_VALUE:
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", "tolerance left frozen grid")
    return _TOLERANCE_VALUE.index(exact)


def _expect_children(
    children: tuple[_Expr, ...], expected: tuple[str, ...], operator: str
) -> None:
    actual = tuple(child.sort for child in children)
    if actual == expected:
        return
    if operator in {"add", "difference", "equal_exact", "less_equal", "greater_equal"} and any(
        sort == "Bit" for sort in actual
    ):
        _reject(
            "REJECT_IMPLICIT_COERCION",
            f"{operator} received Bit; explicit bit_to_scalar is required",
        )
    _reject(
        "REJECT_TYPE_MISMATCH",
        f"{operator} expects {expected}, received {actual}",
    )


def _parse_source(value: object, *, migrate_scope_alias: bool) -> _Expr:
    node = _sequence(value, "source AST node")
    if not node or not isinstance(node[0], str):
        _reject("REJECT_MALFORMED_SOURCE_AST", "source node needs a text operator")
    name = node[0]
    args = node[1:]

    if name == "scalar_const":
        return _Expr(0, 0, "RationalValue", parameters=(_parameter_index(args),))
    if name == "bit_at":
        if len(args) != 1:
            _reject("REJECT_MALFORMED_SOURCE_AST", "bit_at arity must be one")
        return _Expr(0, 1, "Bit", parameters=(_uint(args[0], "entity_slot", 8),))
    if name == "set_size":
        if args:
            _reject("REJECT_MALFORMED_SOURCE_AST", "set_size has no arguments")
        return _Expr(0, 2, "BoundedInt")
    if name == "aggregate":
        if len(args) == 4:
            map_value, scope_value, quantity_value, extension_value = args
        else:
            _reject(
                "REJECT_MALFORMED_SOURCE_AST",
                "aggregate requires map, scope, quantity, and extension",
            )
        if scope_value == DEPRECATED_SCOPE_ALIAS:
            if not migrate_scope_alias:
                _reject(
                    "REJECT_NONCANONICAL_SCOPE_ALIAS",
                    "source alias is forbidden at the formal canonicalizer boundary",
                )
            scope_value = CANONICAL_PRIMARY_SCOPE
        map_index = _registry_index(map_value, AGGREGATE_MAP_IDS, "aggregate_map")
        scope_index = _registry_index(scope_value, SCOPE_IDS, "scope")
        quantity_index = _registry_index(quantity_value, QUANTITY_IDS, "quantity")
        extension = _sequence(extension_value, "scope extension")
        if len(extension) > STRUCTURAL_LIMITS.max_scope_clauses:
            _reject("REJECT_STRUCTURAL_LIMIT", "scope extension exceeds two clauses")
        clauses: list[tuple[int, bool]] = []
        for raw_clause in extension:
            clause = _sequence(raw_clause, "scope clause")
            if len(clause) != 2 or type(clause[1]) is not bool:
                _reject(
                    "REJECT_MALFORMED_SOURCE_AST",
                    "scope clause must be [context, bool]",
                )
            clauses.append(
                (
                    _registry_index(clause[0], CONTEXT_IDS, "context"),
                    clause[1],
                )
            )
        if len({context for context, _ in clauses}) != len(clauses):
            _reject("REJECT_DUPLICATE_SCOPE_CONTEXT", "scope contexts must be unique")
        clauses.sort()
        output_sort = "BoundedInt" if AGGREGATE_MAP_IDS[map_index] == "count_nonzero_v1" else "RationalValue"
        return _Expr(
            0,
            3,
            output_sort,
            parameters=(map_index, scope_index, quantity_index, tuple(clauses)),
        )
    if name == "context_flag":
        if len(args) != 1:
            _reject("REJECT_MALFORMED_SOURCE_AST", "context_flag arity must be one")
        return _Expr(
            0,
            4,
            "Bool",
            parameters=(_registry_index(args[0], CONTEXT_IDS, "context"),),
        )
    if name == "task_flag":
        if len(args) != 1:
            _reject("REJECT_MALFORMED_SOURCE_AST", "task_flag arity must be one")
        return _Expr(
            0,
            5,
            "Bool",
            parameters=(_registry_index(args[0], TASK_IDS, "task"),),
        )
    if name == "new_symbol_call":
        _reject("REJECT_NEW_SYMBOL_IN_OLD_DSL", "new symbols are Phase-3B only")

    if name in UNARY_IDS:
        if len(args) != 1:
            _reject("REJECT_MALFORMED_SOURCE_AST", f"{name} arity must be one")
        child = _parse_source(args[0], migrate_scope_alias=migrate_scope_alias)
        expected = {
            "bit_to_scalar": "Bit",
            "int_to_scalar": "BoundedInt",
            "absolute": "RationalValue",
            "sign": "RationalValue",
        }[name]
        _expect_children((child,), (expected,), name)
        output = {
            "bit_to_scalar": "RationalValue",
            "int_to_scalar": "RationalValue",
            "absolute": "RationalValue",
            "sign": "Sign",
        }[name]
        return _Expr(1, UNARY_IDS[name], output, (child,))

    if name in BINARY_IDS:
        if len(args) != 2:
            _reject("REJECT_MALFORMED_SOURCE_AST", f"{name} arity must be two")
        children = tuple(
            _parse_source(arg, migrate_scope_alias=migrate_scope_alias) for arg in args
        )
        if name in {"same_sign", "opposite_sign"}:
            expected = ("Sign", "Sign")
        else:
            expected = ("RationalValue", "RationalValue")
        _expect_children(children, expected, name)
        output = "RationalValue" if name in {"add", "difference"} else "Bool"
        return _Expr(2, BINARY_IDS[name], output, children)

    if name == "approx_equal":
        if len(args) not in {3, 4}:
            _reject(
                "REJECT_MALFORMED_SOURCE_AST",
                "approx_equal needs two children plus an index or numerator/denominator",
            )
        children = tuple(
            _parse_source(arg, migrate_scope_alias=migrate_scope_alias)
            for arg in args[:2]
        )
        _expect_children(children, ("RationalValue", "RationalValue"), name)
        if len(args) == 3:
            if type(args[2]) is not int:
                _reject(
                    "REJECT_MALFORMED_SOURCE_AST",
                    "tolerance rational must use separate numerator/denominator fields",
                )
            tolerance = _tolerance_index(args[2])
        else:
            tolerance = _tolerance_index(args[2:])
        return _Expr(3, 0, "Bool", children, (tolerance,))

    if name == "top_level_AND":
        raw_children: tuple[object, ...]
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            possible = tuple(args[0])
            if possible and all(
                isinstance(item, (list, tuple)) and item for item in possible
            ):
                raw_children = possible
            else:
                raw_children = args
        else:
            raw_children = args
        if not raw_children:
            _reject("REJECT_EMPTY_CONJUNCTION", "AND0 has no canonical true node")
        children = tuple(
            _parse_source(arg, migrate_scope_alias=migrate_scope_alias)
            for arg in raw_children
        )
        _expect_children(children, ("Bool",) * len(children), name)
        return _Expr(4, 0, "Bool", children)

    _reject("REJECT_UNKNOWN_EXPRESSION", f"unknown old-DSL expression: {name!r}")


def _expr_value(expr: _Expr) -> tuple[object, ...]:
    if expr.tag == 0:
        if expr.operator_id == 0:
            return (0, 0, expr.parameters[0])
        if expr.operator_id == 1:
            return (0, 1, expr.parameters[0])
        if expr.operator_id == 2:
            return (0, 2)
        if expr.operator_id == 3:
            return (0, 3, *expr.parameters)
        if expr.operator_id in {4, 5, 6}:
            return (0, expr.operator_id, expr.parameters[0])
    if expr.tag == 1:
        return (1, expr.operator_id, _expr_value(expr.children[0]))
    if expr.tag == 2:
        return (
            2,
            expr.operator_id,
            _expr_value(expr.children[0]),
            _expr_value(expr.children[1]),
        )
    if expr.tag == 3:
        return (
            3,
            expr.operator_id,
            _expr_value(expr.children[0]),
            _expr_value(expr.children[1]),
            expr.parameters[0],
        )
    if expr.tag == 4:
        return (4, tuple(_expr_value(child) for child in expr.children))
    raise AssertionError("internal expression schema drift")


def _node_bytes(expr: _Expr) -> bytes:
    return canonical_cbor_encode(_expr_value(expr))


def _commutative_key(expr: _Expr) -> tuple[bytes, bytes]:
    encoded = _node_bytes(expr)
    return sha256(encoded).digest(), encoded


def _constant_index(expr: _Expr) -> int | None:
    if expr.tag == 0 and expr.operator_id == 0:
        return int(expr.parameters[0])
    return None


def _constant(value: Fraction) -> _Expr | None:
    index = _PARAMETER_INDEX.get((value.numerator, value.denominator))
    return None if index is None else _Expr(0, 0, "RationalValue", parameters=(index,))


def _zero(expr: _Expr) -> bool:
    index = _constant_index(expr)
    return index is not None and _PARAMETER_VALUE[index] == 0


def _flatten_add(expr: _Expr) -> list[_Expr]:
    if expr.tag == 2 and expr.operator_id == 0:
        return _flatten_add(expr.children[0]) + _flatten_add(expr.children[1])
    return [expr]


def _build_add(operands: list[_Expr]) -> tuple[_Expr, bool]:
    result = operands[-1]
    folded = False
    for operand in reversed(operands[:-1]):
        left_index = _constant_index(operand)
        right_index = _constant_index(result)
        if left_index is not None and right_index is not None:
            candidate = _constant(
                _PARAMETER_VALUE[left_index] + _PARAMETER_VALUE[right_index]
            )
            if candidate is not None:
                result = candidate
                folded = True
                continue
        result = _Expr(2, 0, "RationalValue", (operand, result))
    return result, folded


def _normalize_add(children: tuple[_Expr, _Expr]) -> _Expr:
    operands = [
        operand
        for child in children
        for operand in _flatten_add(child)
        if not _zero(operand)
    ]
    if not operands:
        zero = _constant(Fraction(0))
        assert zero is not None
        return zero
    if len(operands) == 1:
        return operands[0]
    while True:
        operands.sort(key=_commutative_key)
        rebuilt, folded = _build_add(operands)
        if not folded:
            return rebuilt
        operands = [item for item in _flatten_add(rebuilt) if not _zero(item)]
        if not operands:
            zero = _constant(Fraction(0))
            assert zero is not None
            return zero
        if len(operands) == 1:
            return operands[0]


def _normalize(expr: _Expr) -> _Expr:
    if expr.tag == 0:
        return expr
    children = tuple(_normalize(child) for child in expr.children)
    if expr.tag == 1:
        child = children[0]
        if expr.operator_id == 2:
            if child.tag == 1 and child.operator_id == 2:
                return child
            index = _constant_index(child)
            if index is not None:
                folded = _constant(abs(_PARAMETER_VALUE[index]))
                if folded is not None:
                    return folded
        return _Expr(1, expr.operator_id, expr.sort, (child,))
    if expr.tag == 2:
        if expr.operator_id == 0:
            return _normalize_add((children[0], children[1]))
        if expr.operator_id == 1:
            left, right = children
            if _zero(right):
                return left
            if _node_bytes(left) == _node_bytes(right):
                zero = _constant(Fraction(0))
                assert zero is not None
                return zero
            left_index, right_index = _constant_index(left), _constant_index(right)
            if left_index is not None and right_index is not None:
                folded = _constant(
                    _PARAMETER_VALUE[left_index] - _PARAMETER_VALUE[right_index]
                )
                if folded is not None:
                    return folded
            return _Expr(2, 1, expr.sort, (left, right))
        if expr.operator_id == 4:
            return _Expr(2, 3, "Bool", (children[1], children[0]))
        if expr.operator_id in {2, 5, 6}:
            ordered = tuple(sorted(children, key=_commutative_key))
            return _Expr(2, expr.operator_id, expr.sort, ordered)
        return _Expr(2, expr.operator_id, expr.sort, children)
    if expr.tag == 3:
        ordered = tuple(sorted(children, key=_commutative_key))
        if expr.parameters[0] == 0:
            return _Expr(2, 2, "Bool", ordered)
        return _Expr(3, 0, "Bool", ordered, expr.parameters)
    if expr.tag == 4:
        flattened: list[_Expr] = []
        for child in children:
            if child.tag == 4:
                flattened.extend(child.children)
            else:
                flattened.append(child)
        unique = {_node_bytes(child): child for child in flattened}
        clauses = tuple(unique[key] for key in sorted(unique))
        if not clauses:
            _reject("REJECT_EMPTY_CONJUNCTION", "AND0 has no canonical true node")
        if len(clauses) == 1:
            return clauses[0]
        return _Expr(4, 0, "Bool", clauses)
    raise AssertionError("internal expression tag drift")


def _merge_metrics(expr: _Expr) -> AstMetrics:
    if expr.tag == 0:
        scalar_count = int(expr.operator_id == 0)
        aggregate_count = int(expr.operator_id == 3)
        bit_slots = (
            frozenset((int(expr.parameters[0]),))
            if expr.operator_id == 1
            else frozenset()
        )
        scope_count = (
            len(expr.parameters[3]) if expr.operator_id == 3 else 0
        )
        return AstMetrics(
            expr.sort,
            0,
            1,
            scalar_count,
            aggregate_count,
            bit_slots,
            scope_count,
            0,
        )
    child_metrics = tuple(_merge_metrics(child) for child in expr.children)
    return AstMetrics(
        output_sort=expr.sort,
        depth=1 + max(metric.depth for metric in child_metrics),
        node_count=1 + sum(metric.node_count for metric in child_metrics),
        scalar_parameter_occurrences=sum(
            metric.scalar_parameter_occurrences for metric in child_metrics
        ),
        aggregate_leaf_count=sum(
            metric.aggregate_leaf_count for metric in child_metrics
        ),
        distinct_bit_slots=frozenset().union(
            *(metric.distinct_bit_slots for metric in child_metrics)
        ),
        scope_clause_count=sum(metric.scope_clause_count for metric in child_metrics),
        top_level_clause_count=len(expr.children) if expr.tag == 4 else 0,
    )


def _enforce_limits(expr: _Expr) -> AstMetrics:
    metrics = _merge_metrics(expr)
    limits = STRUCTURAL_LIMITS
    failures = []
    if metrics.depth > limits.max_total_ast_depth:
        failures.append("depth")
    if metrics.node_count > limits.max_total_node_count:
        failures.append("node_count")
    if metrics.top_level_clause_count > limits.max_top_level_clauses:
        failures.append("top_level_clauses")
    if len(metrics.distinct_bit_slots) > limits.max_distinct_bit_slots:
        failures.append("distinct_bit_slots")
    if metrics.aggregate_leaf_count > limits.max_aggregate_leaves:
        failures.append("aggregate_leaves")
    if metrics.scope_clause_count > limits.max_scope_clauses:
        failures.append("scope_clauses")
    if metrics.scalar_parameter_occurrences > limits.max_fitted_scalar_parameters:
        failures.append("scalar_parameters")
    if failures:
        _reject(
            "REJECT_STRUCTURAL_LIMIT",
            "canonical AST exceeds " + ", ".join(failures),
        )
    return metrics


def _root_operator_id(expr: _Expr) -> int:
    if expr.tag == 0:
        return expr.operator_id
    if expr.tag == 1:
        return 0x0100 + expr.operator_id
    if expr.tag == 2:
        return 0x0200 + expr.operator_id
    if expr.tag == 3:
        return 0x0300 + expr.operator_id
    if expr.tag == 4:
        return 0x0400
    raise AssertionError("root tag drift")


def _accepted(expr: _Expr) -> CanonicalAst:
    metrics = _enforce_limits(expr)
    value = (AST_SCHEMA_VERSION, _expr_value(expr))
    encoded = canonical_cbor_encode(value)
    return CanonicalAst(
        value=value,
        cbor_bytes=encoded,
        digest=content_hash(AST_HASH_DOMAIN, value),
        root_operator_id=_root_operator_id(expr),
        metrics=metrics,
    )


def canonicalize_source_ast(
    source_ast: object, *, migrate_legacy_scope_alias: bool = False
) -> CanonicalAst:
    """Type-check and normalize a readable source AST into program identity.

    The migration flag is deliberately explicit and defaults to false.  Formal
    callers therefore reject the deprecated documentation alias rather than
    silently creating a fifth scope token.
    """

    parsed = _parse_source(
        source_ast,
        migrate_scope_alias=migrate_legacy_scope_alias,
    )
    return _accepted(_normalize(parsed))


def _canonical_array(value: object, name: str, length: int | None = None) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        _reject("REJECT_NONCANONICAL_AST", f"{name} must be a CBOR array")
    if length is not None and len(value) != length:
        _reject("REJECT_NONCANONICAL_AST", f"{name} has the wrong length")
    return value


def _canonical_uint(value: object, name: str, upper_exclusive: int) -> int:
    if type(value) is not int or value < 0 or value >= upper_exclusive:
        _reject("REJECT_REGISTRY_INDEX_OUT_OF_RANGE", f"{name} is out of range")
    return value


def _parse_canonical_node(value: object) -> _Expr:
    node = _canonical_array(value, "canonical node")
    if not node or type(node[0]) is not int:
        _reject("REJECT_NONCANONICAL_AST", "canonical node needs a numeric tag")
    tag = node[0]
    if tag == 0:
        if len(node) < 2 or type(node[1]) is not int:
            _reject("REJECT_NONCANONICAL_AST", "leaf needs a numeric ID")
        leaf = node[1]
        if leaf == 0:
            _canonical_array(node, "scalar_const", 3)
            index = _canonical_uint(node[2], "rational_parameter_index", 7)
            return _Expr(0, 0, "RationalValue", parameters=(index,))
        if leaf == 1:
            _canonical_array(node, "bit_at", 3)
            return _Expr(
                0,
                1,
                "Bit",
                parameters=(_canonical_uint(node[2], "entity_slot", 8),),
            )
        if leaf == 2:
            _canonical_array(node, "set_size", 2)
            return _Expr(0, 2, "BoundedInt")
        if leaf == 3:
            _canonical_array(node, "aggregate", 6)
            map_index = _canonical_uint(node[2], "aggregate_map", len(AGGREGATE_MAP_IDS))
            scope_index = _canonical_uint(node[3], "scope", len(SCOPE_IDS))
            quantity_index = _canonical_uint(node[4], "quantity", len(QUANTITY_IDS))
            extension = _canonical_array(node[5], "scope extension")
            clauses: list[tuple[int, bool]] = []
            for raw_clause in extension:
                clause = _canonical_array(raw_clause, "scope clause", 2)
                context = _canonical_uint(clause[0], "context", len(CONTEXT_IDS))
                if type(clause[1]) is not bool:
                    _reject("REJECT_NONCANONICAL_AST", "scope expectation must be bool")
                clauses.append((context, clause[1]))
            if len(clauses) > 2 or tuple(clauses) != tuple(sorted(clauses)):
                _reject("REJECT_NONCANONICAL_AST", "scope clauses are not sorted")
            if len({context for context, _ in clauses}) != len(clauses):
                _reject("REJECT_NONCANONICAL_AST", "duplicate scope context")
            output = "BoundedInt" if AGGREGATE_MAP_IDS[map_index] == "count_nonzero_v1" else "RationalValue"
            return _Expr(
                0,
                3,
                output,
                parameters=(map_index, scope_index, quantity_index, tuple(clauses)),
            )
        if leaf == 4:
            _canonical_array(node, "context_flag", 3)
            return _Expr(
                0,
                4,
                "Bool",
                parameters=(_canonical_uint(node[2], "context", len(CONTEXT_IDS)),),
            )
        if leaf == 5:
            _canonical_array(node, "task_flag", 3)
            return _Expr(
                0,
                5,
                "Bool",
                parameters=(_canonical_uint(node[2], "task", len(TASK_IDS)),),
            )
        if leaf == 6:
            _reject("REJECT_NEW_SYMBOL_IN_OLD_DSL", "new symbols are Phase-3B only")
        _reject("REJECT_UNKNOWN_EXPRESSION", "unknown leaf ID")
    if tag == 1:
        _canonical_array(node, "unary", 3)
        operator = _canonical_uint(node[1], "unary_operator", 4)
        child = _parse_canonical_node(node[2])
        expected = ("Bit", "BoundedInt", "RationalValue", "RationalValue")[operator]
        _expect_children((child,), (expected,), "canonical unary")
        output = ("RationalValue", "RationalValue", "RationalValue", "Sign")[operator]
        return _Expr(1, operator, output, (child,))
    if tag == 2:
        _canonical_array(node, "binary", 4)
        operator = _canonical_uint(node[1], "binary_operator", 8)
        if operator in {4, 7}:
            _reject("REJECT_NONCANONICAL_AST", "source-only or reserved binary ID")
        children = (_parse_canonical_node(node[2]), _parse_canonical_node(node[3]))
        expected = ("Sign", "Sign") if operator in {5, 6} else (
            "RationalValue",
            "RationalValue",
        )
        _expect_children(children, expected, "canonical binary")
        output = "RationalValue" if operator in {0, 1} else "Bool"
        return _Expr(2, operator, output, children)
    if tag == 3:
        _canonical_array(node, "ternary", 5)
        operator = _canonical_uint(node[1], "ternary_operator", 1)
        children = (_parse_canonical_node(node[2]), _parse_canonical_node(node[3]))
        _expect_children(children, ("RationalValue", "RationalValue"), "approx_equal")
        tolerance = _canonical_uint(node[4], "tolerance", 3)
        return _Expr(3, operator, "Bool", children, (tolerance,))
    if tag == 4:
        _canonical_array(node, "conjunction", 2)
        raw_children = _canonical_array(node[1], "conjunction clauses")
        if len(raw_children) not in {2, 3}:
            _reject("REJECT_NONCANONICAL_AST", "canonical AND needs two or three atoms")
        children = tuple(_parse_canonical_node(item) for item in raw_children)
        _expect_children(children, ("Bool",) * len(children), "top_level_AND")
        return _Expr(4, 0, "Bool", children)
    _reject("REJECT_UNKNOWN_EXPRESSION", "unknown canonical node tag")


def decode_canonical_ast(payload: bytes) -> CanonicalAst:
    """Strictly decode a canonical AST without repairing its byte identity."""

    try:
        value = canonical_cbor_decode(payload)
    except StrictCborError:
        raise
    envelope = _canonical_array(value, "CanonicalAstV1", 2)
    if envelope[0] != AST_SCHEMA_VERSION or type(envelope[0]) is not int:
        _reject("REJECT_UNKNOWN_AST_SCHEMA", "unknown CanonicalAst schema version")
    parsed = _parse_canonical_node(envelope[1])
    normalized = _normalize(parsed)
    if _expr_value(parsed) != _expr_value(normalized):
        _reject(
            "REJECT_NONCANONICAL_AST",
            "AST still requires a frozen normalization rewrite",
        )
    accepted = _accepted(parsed)
    if accepted.cbor_bytes != payload:
        _reject("REJECT_NONCANONICAL_AST", "AST re-encoding differs")
    return accepted


def migrate_legacy_scope_alias(source_ast: object) -> CanonicalAst:
    """Explicit diagnostic migration adapter for the one frozen source alias."""

    return canonicalize_source_ast(source_ast, migrate_legacy_scope_alias=True)


__all__ = [
    "AST_HASH_DOMAIN",
    "AST_SCHEMA_ID",
    "AST_SCHEMA_VERSION",
    "AstMetrics",
    "CanonicalAst",
    "StrictAstError",
    "canonicalize_source_ast",
    "decode_canonical_ast",
    "migrate_legacy_scope_alias",
]

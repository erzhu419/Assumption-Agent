"""Strict AST admission for ``hegel-old-dsl-v1.2.0``.

Shrink step 2 inherits the v1 numeric AST/CBOR identity and the shrink-1
aggregate-map tombstones.  It additionally rejects RationalParameterId
0/2/4/6 while retaining their numeric positions forever.

Normalization is child-aware.  A rewrite may fold constants only when its
result is one of active IDs 1/3/5.  Thus active-input ``1 + 1`` remains an
``add`` AST instead of reintroducing tombstoned ID 6, while a fold such as
``1 + -1`` still produces active zero ID 3.
"""

from __future__ import annotations

from fractions import Fraction

from . import strict_ast_v1 as _parent
from .phase3_m3_shrink2_core_v1 import (
    ACTIVE_RATIONAL_PARAMETER_IDS,
    DSL_VERSION,
    RATIONAL_PARAMETER_ALLOCATED_ID_COUNT,
    REMOVED_RATIONAL_PARAMETER_ERROR,
    TOMBSTONED_RATIONAL_PARAMETER_IDS,
    UNKNOWN_RATIONAL_PARAMETER_ERROR,
)
from .strict_ast_shrink1_v1 import (
    PROGRAM_SEMANTIC_IDENTITY_DOMAIN,
    ProgramAdmissionIdentityV1,
    ProgramSemanticIdentityV1,
    _precheck_source_tombstones as _precheck_inherited_source_aggregate_tombstones,
    _precheck_formal_tombstones as _precheck_inherited_aggregate_tombstones,
    decode_shrink1_canonical_ast,
)
from .strict_ast_v1 import CanonicalAst, StrictAstError
from .strict_cbor_v1 import canonical_cbor_decode


_RATIONAL_PARAMETER_INDEX_BY_VALUE = {
    (-2, 1): 0,
    (-1, 1): 1,
    (-1, 2): 2,
    (0, 1): 3,
    (1, 2): 4,
    (1, 1): 5,
    (2, 1): 6,
}


def _reject_removed(detail: str) -> "None":
    raise StrictAstError(REMOVED_RATIONAL_PARAMETER_ERROR, detail)


def rational_parameter_id_is_active(numeric_id: int) -> bool:
    """Validate one allocated RationalParameterId/v1 under shrink step 2."""

    if (
        type(numeric_id) is not int
        or numeric_id < 0
        or numeric_id >= RATIONAL_PARAMETER_ALLOCATED_ID_COUNT
    ):
        raise StrictAstError(
            UNKNOWN_RATIONAL_PARAMETER_ERROR,
            f"RationalParameterId {numeric_id!r} is outside allocated IDs 0..6",
        )
    if numeric_id in TOMBSTONED_RATIONAL_PARAMETER_IDS:
        _reject_removed(
            f"RationalParameterId {numeric_id} is tombstoned in {DSL_VERSION}"
        )
    return numeric_id in ACTIVE_RATIONAL_PARAMETER_IDS


def _source_parameter_id(node: tuple[object, ...]) -> int:
    """Resolve a parent-validated scalar_const source leaf to its immutable ID."""

    if len(node) == 2:
        numeric_id = node[1]
        assert type(numeric_id) is int
        return numeric_id
    assert len(node) == 3
    numerator, denominator = node[1], node[2]
    assert type(numerator) is int and type(denominator) is int and denominator > 0
    reduced = Fraction(numerator, denominator)
    return _RATIONAL_PARAMETER_INDEX_BY_VALUE[
        (reduced.numerator, reduced.denominator)
    ]


def _precheck_validated_source_parameter_tombstones(value: object) -> None:
    """Inspect only AST child positions after the parent accepted the source."""

    assert isinstance(value, (list, tuple))
    node = tuple(value)
    assert node and isinstance(node[0], str)
    name = node[0]

    if name == "scalar_const":
        numeric_id = _source_parameter_id(node)
        if numeric_id in TOMBSTONED_RATIONAL_PARAMETER_IDS:
            _reject_removed(
                f"source RationalParameterId {numeric_id} is tombstoned in {DSL_VERSION}"
            )
        return
    if name in {
        "bit_at",
        "set_size",
        "aggregate",
        "context_flag",
        "task_flag",
    }:
        return
    if name in {"bit_to_scalar", "int_to_scalar", "absolute", "sign"}:
        _precheck_validated_source_parameter_tombstones(node[1])
        return
    if name in {
        "add",
        "difference",
        "equal_exact",
        "less_equal",
        "greater_equal",
        "same_sign",
        "opposite_sign",
    }:
        _precheck_validated_source_parameter_tombstones(node[1])
        _precheck_validated_source_parameter_tombstones(node[2])
        return
    if name == "approx_equal":
        _precheck_validated_source_parameter_tombstones(node[1])
        _precheck_validated_source_parameter_tombstones(node[2])
        return
    assert name == "top_level_AND"
    args = node[1:]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        possible = tuple(args[0])
        if possible and all(
            isinstance(item, (list, tuple)) and item for item in possible
        ):
            children = possible
        else:
            children = args
    else:
        children = args
    for child in children:
        _precheck_validated_source_parameter_tombstones(child)


def _child_constant(value: Fraction) -> _parent._Expr | None:
    """Create a constant only when the fold result remains child-admissible."""

    index = _RATIONAL_PARAMETER_INDEX_BY_VALUE.get(
        (value.numerator, value.denominator)
    )
    if index not in ACTIVE_RATIONAL_PARAMETER_IDS:
        return None
    return _parent._Expr(0, 0, "RationalValue", parameters=(index,))


def _child_zero(expr: _parent._Expr) -> bool:
    index = _parent._constant_index(expr)
    return index == 3


def _flatten_add(expr: _parent._Expr) -> list[_parent._Expr]:
    if expr.tag == 2 and expr.operator_id == 0:
        return _flatten_add(expr.children[0]) + _flatten_add(expr.children[1])
    return [expr]


def _build_add(
    operands: list[_parent._Expr],
) -> tuple[_parent._Expr, bool]:
    result = operands[-1]
    folded = False
    for operand in reversed(operands[:-1]):
        left_index = _parent._constant_index(operand)
        right_index = _parent._constant_index(result)
        if left_index is not None and right_index is not None:
            candidate = _child_constant(
                _parent._PARAMETER_VALUE[left_index]
                + _parent._PARAMETER_VALUE[right_index]
            )
            if candidate is not None:
                result = candidate
                folded = True
                continue
        result = _parent._Expr(2, 0, "RationalValue", (operand, result))
    return result, folded


def _normalize_add(
    children: tuple[_parent._Expr, _parent._Expr],
) -> _parent._Expr:
    operands = [
        operand
        for child in children
        for operand in _flatten_add(child)
        if not _child_zero(operand)
    ]
    if not operands:
        zero = _child_constant(Fraction(0))
        assert zero is not None
        return zero
    if len(operands) == 1:
        return operands[0]
    while True:
        operands.sort(key=_parent._commutative_key)
        rebuilt, folded = _build_add(operands)
        if not folded:
            return rebuilt
        operands = [item for item in _flatten_add(rebuilt) if not _child_zero(item)]
        if not operands:
            zero = _child_constant(Fraction(0))
            assert zero is not None
            return zero
        if len(operands) == 1:
            return operands[0]


def _normalize_child(expr: _parent._Expr) -> _parent._Expr:
    """Apply the frozen rewrite list without creating a removed constant."""

    if expr.tag == 0:
        return expr
    children = tuple(_normalize_child(child) for child in expr.children)
    if expr.tag == 1:
        child = children[0]
        if expr.operator_id == 2:
            if child.tag == 1 and child.operator_id == 2:
                return child
            index = _parent._constant_index(child)
            if index is not None:
                folded = _child_constant(abs(_parent._PARAMETER_VALUE[index]))
                if folded is not None:
                    return folded
        return _parent._Expr(1, expr.operator_id, expr.sort, (child,))
    if expr.tag == 2:
        if expr.operator_id == 0:
            return _normalize_add((children[0], children[1]))
        if expr.operator_id == 1:
            left, right = children
            if _child_zero(right):
                return left
            if _parent._node_bytes(left) == _parent._node_bytes(right):
                zero = _child_constant(Fraction(0))
                assert zero is not None
                return zero
            left_index = _parent._constant_index(left)
            right_index = _parent._constant_index(right)
            if left_index is not None and right_index is not None:
                folded = _child_constant(
                    _parent._PARAMETER_VALUE[left_index]
                    - _parent._PARAMETER_VALUE[right_index]
                )
                if folded is not None:
                    return folded
            return _parent._Expr(2, 1, expr.sort, (left, right))
        if expr.operator_id == 4:
            return _parent._Expr(2, 3, "Bool", (children[1], children[0]))
        if expr.operator_id in {2, 5, 6}:
            ordered = tuple(sorted(children, key=_parent._commutative_key))
            return _parent._Expr(2, expr.operator_id, expr.sort, ordered)
        return _parent._Expr(2, expr.operator_id, expr.sort, children)
    if expr.tag == 3:
        ordered = tuple(sorted(children, key=_parent._commutative_key))
        if expr.parameters[0] == 0:
            return _parent._Expr(2, 2, "Bool", ordered)
        return _parent._Expr(3, 0, "Bool", ordered, expr.parameters)
    if expr.tag == 4:
        flattened: list[_parent._Expr] = []
        for child in children:
            if child.tag == 4:
                flattened.extend(child.children)
            else:
                flattened.append(child)
        unique = {_parent._node_bytes(child): child for child in flattened}
        clauses = tuple(unique[key] for key in sorted(unique))
        if not clauses:
            raise StrictAstError(
                "REJECT_EMPTY_CONJUNCTION", "AND0 has no canonical true node"
            )
        if len(clauses) == 1:
            return clauses[0]
        return _parent._Expr(4, 0, "Bool", clauses)
    raise AssertionError("internal expression tag drift")


def canonicalize_shrink2_source_ast(
    source_ast: object, *, migrate_legacy_scope_alias: bool = False
) -> CanonicalAst:
    """Canonicalize a source AST under the shrink-2 sparse registries.

    The parent is invoked first solely to preserve all malformed/type/registry
    error priorities and shrink-1 aggregate tombstones.  A second parse then
    applies the child-aware normalizer.
    """

    _parent.canonicalize_source_ast(
        source_ast,
        migrate_legacy_scope_alias=migrate_legacy_scope_alias,
    )
    _precheck_inherited_source_aggregate_tombstones(source_ast)
    _precheck_validated_source_parameter_tombstones(source_ast)
    parsed = _parent._parse_source(
        source_ast,
        migrate_scope_alias=migrate_legacy_scope_alias,
    )
    return _parent._accepted(_normalize_child(parsed))


def _precheck_formal_parameter_node(value: object) -> None:
    node = value
    assert isinstance(node, tuple) and node and type(node[0]) is int
    tag = node[0]
    if tag == 0:
        if node[1] == 0:
            numeric_id = node[2]
            assert type(numeric_id) is int
            if numeric_id in TOMBSTONED_RATIONAL_PARAMETER_IDS:
                _reject_removed(
                    f"formal RationalParameterId {numeric_id} is tombstoned in {DSL_VERSION}"
                )
        return
    if tag == 1:
        _precheck_formal_parameter_node(node[2])
        return
    if tag == 2:
        _precheck_formal_parameter_node(node[2])
        _precheck_formal_parameter_node(node[3])
        return
    if tag == 3:
        _precheck_formal_parameter_node(node[2])
        _precheck_formal_parameter_node(node[3])
        return
    assert tag == 4
    for child in node[1]:
        _precheck_formal_parameter_node(child)


def decode_shrink2_canonical_ast(payload: bytes) -> CanonicalAst:
    """Decode canonical child bytes without repairing their program identity."""

    value = canonical_cbor_decode(payload)
    envelope = _parent._canonical_array(value, "CanonicalAstV1", 2)
    if envelope[0] != _parent.AST_SCHEMA_VERSION or type(envelope[0]) is not int:
        raise StrictAstError(
            "REJECT_UNKNOWN_AST_SCHEMA", "unknown CanonicalAst schema version"
        )
    parsed = _parent._parse_canonical_node(envelope[1])

    # Structural validation above ensures these scans can see only legitimate
    # AST leaves, never arbitrary CBOR payloads.  Inherited aggregate rejection
    # keeps priority over the new RationalParameter rejection.
    _precheck_inherited_aggregate_tombstones(value)
    _precheck_formal_parameter_node(envelope[1])

    normalized = _normalize_child(parsed)
    if _parent._expr_value(parsed) != _parent._expr_value(normalized):
        raise StrictAstError(
            "REJECT_NONCANONICAL_AST",
            "AST still requires a frozen child normalization rewrite",
        )
    accepted = _parent._accepted(parsed)
    if accepted.cbor_bytes != payload:
        raise StrictAstError("REJECT_NONCANONICAL_AST", "AST re-encoding differs")
    return accepted


def read_legacy_parent_program(payload: bytes) -> dict[str, object]:
    """Describe a v1.1 parent program at the shrink-2 admission boundary."""

    parent = decode_shrink1_canonical_ast(payload)
    try:
        decode_shrink2_canonical_ast(payload)
    except StrictAstError as error:
        child_status = "REJECTED"
        child_error = error.code
    else:
        child_status = "ADMITTED"
        child_error = None
    return {
        "legacy_program_status": "VALID_UNDER_PARENT_DSL_ONLY"
        if child_status == "REJECTED"
        else "VALID_UNDER_PARENT_AND_CHILD_DSL",
        "parent_dsl_version": "hegel-old-dsl-v1.1.0",
        "parent_effective_freeze_version": "hegel-freeze-p2b-p3-v1.1.2",
        "current_dsl_version": DSL_VERSION,
        "canonical_ast_hash": parent.hash_id,
        "admitted_under_current_dsl": child_status == "ADMITTED",
        "current_dsl_error_code": child_error,
        "automatic_parameter_migration_performed": False,
    }


__all__ = [
    "PROGRAM_SEMANTIC_IDENTITY_DOMAIN",
    "ProgramAdmissionIdentityV1",
    "ProgramSemanticIdentityV1",
    "canonicalize_shrink2_source_ast",
    "decode_shrink2_canonical_ast",
    "rational_parameter_id_is_active",
    "read_legacy_parent_program",
]

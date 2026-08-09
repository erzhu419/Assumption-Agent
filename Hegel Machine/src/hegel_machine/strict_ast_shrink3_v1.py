"""Strict AST admission for ``hegel-old-dsl-v1.3.0``.

Shrink step 3 inherits the shrink-2 numeric AST/CBOR identity, aggregate-map
and RationalParameter tombstones, and normalization rules.  Its only language
delta is admission: source operator ``add`` and formal ``BinaryOperatorId 0``
are tombstoned.  ``difference`` keeps immutable ID 1.

The parent language is always structurally and type validated before a child
tombstone scan.  For a structurally legal formal tree, rejection priority is
global and frozen as aggregate-map tombstone, RationalParameter tombstone,
binary-operator tombstone, then noncanonical normalization.  In particular,
source-only formal ID 4 is validated using its unchanged binary type contract
before its late noncanonical rejection, so it cannot hide a child tombstone.
"""

from __future__ import annotations

from . import strict_ast_shrink2_v1 as _parent
from . import strict_ast_v1 as _base
from .phase3_m3_shrink3_core_v1 import (
    DSL_VERSION,
    REMOVED_BINARY_OPERATOR_ERROR,
)
from .strict_ast_shrink2_v1 import (
    PROGRAM_SEMANTIC_IDENTITY_DOMAIN,
    ProgramAdmissionIdentityV1,
    ProgramSemanticIdentityV1,
)
from .strict_ast_v1 import CanonicalAst, StrictAstError
from .strict_cbor_v1 import canonical_cbor_decode


def _reject_removed_add(detail: str) -> "None":
    raise StrictAstError(REMOVED_BINARY_OPERATOR_ERROR, detail)


def _precheck_validated_source_add_tombstone(value: object) -> None:
    """Inspect only child positions of a parent-validated source AST."""

    assert isinstance(value, (list, tuple))
    node = tuple(value)
    assert node and isinstance(node[0], str)
    name = node[0]

    if name == "add":
        _reject_removed_add(f"source operator add is tombstoned in {DSL_VERSION}")
    if name in {
        "scalar_const",
        "bit_at",
        "set_size",
        "aggregate",
        "context_flag",
        "task_flag",
    }:
        return
    if name in {"bit_to_scalar", "int_to_scalar", "absolute", "sign"}:
        _precheck_validated_source_add_tombstone(node[1])
        return
    if name in {
        "difference",
        "equal_exact",
        "less_equal",
        "greater_equal",
        "same_sign",
        "opposite_sign",
    }:
        _precheck_validated_source_add_tombstone(node[1])
        _precheck_validated_source_add_tombstone(node[2])
        return
    if name == "approx_equal":
        _precheck_validated_source_add_tombstone(node[1])
        _precheck_validated_source_add_tombstone(node[2])
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
        _precheck_validated_source_add_tombstone(child)


def canonicalize_shrink3_source_ast(
    source_ast: object, *, migrate_legacy_scope_alias: bool = False
) -> CanonicalAst:
    """Admit one source AST without rewriting a removed binary operator.

    Full parent parsing, typing, and registry validation are deliberately
    first.  The three global tombstone scans then precede normalization and
    structural-limit accounting, so an otherwise legal oversized source
    cannot hide a removed operator behind a later limit failure.
    """

    parsed = _base._parse_source(
        source_ast,
        migrate_scope_alias=migrate_legacy_scope_alias,
    )
    _parent._precheck_inherited_source_aggregate_tombstones(source_ast)
    _parent._precheck_validated_source_parameter_tombstones(source_ast)
    _precheck_validated_source_add_tombstone(source_ast)
    return _base._accepted(_parent._normalize_child(parsed))


def _formal_structural_proxy(value: object) -> object:
    """Replace legal-position formal ID 4 solely for structural validation.

    The parent parser rejects source-only binary ID 4 before visiting its
    children.  ID 4 and canonical ``less_equal`` ID 3 have the same input and
    output sorts, so substituting 3 permits complete structural/type checking.
    This routine follows only schema child positions; malformed leaf payloads
    and other arbitrary fields are never searched or rewritten.
    """

    if not isinstance(value, tuple) or not value or type(value[0]) is not int:
        return value
    tag = value[0]
    if tag == 0:
        return value
    if tag == 1 and len(value) == 3:
        return (value[0], value[1], _formal_structural_proxy(value[2]))
    if tag == 2 and len(value) == 4:
        operator = 3 if type(value[1]) is int and value[1] == 4 else value[1]
        return (
            value[0],
            operator,
            _formal_structural_proxy(value[2]),
            _formal_structural_proxy(value[3]),
        )
    if tag == 3 and len(value) == 5:
        return (
            value[0],
            value[1],
            _formal_structural_proxy(value[2]),
            _formal_structural_proxy(value[3]),
            value[4],
        )
    if tag == 4 and len(value) == 2 and isinstance(value[1], tuple):
        return (value[0], tuple(_formal_structural_proxy(item) for item in value[1]))
    return value


def _precheck_validated_formal_add_tombstone(value: object) -> None:
    """Reject formal ID 0 after the complete structural validation pass."""

    assert isinstance(value, tuple) and value and type(value[0]) is int
    tag = value[0]
    if tag == 0:
        return
    if tag == 1:
        _precheck_validated_formal_add_tombstone(value[2])
        return
    if tag == 2:
        if value[1] == 0:
            _reject_removed_add(
                f"formal BinaryOperatorId 0 is tombstoned in {DSL_VERSION}"
            )
        _precheck_validated_formal_add_tombstone(value[2])
        _precheck_validated_formal_add_tombstone(value[3])
        return
    if tag == 3:
        _precheck_validated_formal_add_tombstone(value[2])
        _precheck_validated_formal_add_tombstone(value[3])
        return
    assert tag == 4
    for child in value[1]:
        _precheck_validated_formal_add_tombstone(child)


def _validated_formal_contains_source_alias(value: object) -> bool:
    """Return whether a validated AST contains source-only formal ID 4."""

    assert isinstance(value, tuple) and value and type(value[0]) is int
    tag = value[0]
    if tag == 0:
        return False
    if tag == 1:
        return _validated_formal_contains_source_alias(value[2])
    if tag in {2, 3}:
        if tag == 2 and value[1] == 4:
            return True
        return _validated_formal_contains_source_alias(
            value[2]
        ) or _validated_formal_contains_source_alias(value[3])
    assert tag == 4
    return any(_validated_formal_contains_source_alias(child) for child in value[1])


def decode_shrink3_canonical_ast(payload: bytes) -> CanonicalAst:
    """Decode canonical shrink-3 bytes without repair or operator migration."""

    value = canonical_cbor_decode(payload)
    envelope = _base._canonical_array(value, "CanonicalAstV1", 2)
    if envelope[0] != _base.AST_SCHEMA_VERSION or type(envelope[0]) is not int:
        raise StrictAstError(
            "REJECT_UNKNOWN_AST_SCHEMA", "unknown CanonicalAst schema version"
        )

    # This completes structure, registry, and type validation while allowing
    # source-only ID 4 to reach the late noncanonical check below.
    structural_value = _formal_structural_proxy(envelope[1])
    parsed = _base._parse_canonical_node(structural_value)

    # These are global passes over validated AST positions.  Their order is a
    # frozen observable part of the shrink-3 admission contract.
    _parent._precheck_inherited_aggregate_tombstones(value)
    _parent._precheck_formal_parameter_node(envelope[1])
    _precheck_validated_formal_add_tombstone(envelope[1])

    if _validated_formal_contains_source_alias(envelope[1]):
        raise StrictAstError(
            "REJECT_NONCANONICAL_AST",
            "source-only BinaryOperatorId 4 requires frozen normalization",
        )

    normalized = _parent._normalize_child(parsed)
    if _base._expr_value(parsed) != _base._expr_value(normalized):
        raise StrictAstError(
            "REJECT_NONCANONICAL_AST",
            "AST still requires a frozen child normalization rewrite",
        )
    accepted = _base._accepted(parsed)
    if accepted.cbor_bytes != payload:
        raise StrictAstError("REJECT_NONCANONICAL_AST", "AST re-encoding differs")
    return accepted


def read_legacy_parent_program(payload: bytes) -> dict[str, object]:
    """Describe a v1.2 program at the shrink-3 admission boundary."""

    parent = _parent.decode_shrink2_canonical_ast(payload)
    try:
        decode_shrink3_canonical_ast(payload)
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
        "parent_dsl_version": "hegel-old-dsl-v1.2.0",
        "parent_effective_freeze_version": "hegel-freeze-p2b-p3-v1.2.0",
        "current_dsl_version": DSL_VERSION,
        "canonical_ast_hash": parent.hash_id,
        "admitted_under_current_dsl": child_status == "ADMITTED",
        "current_dsl_error_code": child_error,
        "automatic_operator_migration_performed": False,
    }


__all__ = [
    "PROGRAM_SEMANTIC_IDENTITY_DOMAIN",
    "ProgramAdmissionIdentityV1",
    "ProgramSemanticIdentityV1",
    "canonicalize_shrink3_source_ast",
    "decode_shrink3_canonical_ast",
    "read_legacy_parent_program",
]

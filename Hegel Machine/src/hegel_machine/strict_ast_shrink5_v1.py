"""Strict AST admission for ``hegel-old-dsl-v1.5.0``.

Shrink step 5 delegates the complete parse, typing, registry/tombstone,
normalization, canonical-order, CBOR, and parent-limit checks to the qualified
shrink-4 boundary.  It then enforces its sole child delta on the normalized
tree: a surviving canonical AST may contain no more than six nodes.

This ordering is intentional.  Source AND1 elimination, nested-AND flattening,
canonical sorting, deduplication, and the inherited two-clause check happen
before the new limit.  A canonical seven-node parent program is therefore
rejected as ``REJECT_STRUCTURAL_LIMIT``, while a raw source that normalizes to
at most six nodes remains a byte-identical parent survivor.  Inherited
malformed/type/registry/tombstone and noncanonical priorities are unchanged.
"""

from __future__ import annotations

from . import strict_ast_shrink4_v1 as _parent
from .phase3_m3_shrink5_core_v1 import (
    DSL_VERSION,
    MAX_TOTAL_NODE_COUNT,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    STRUCTURAL_LIMIT_ERROR,
)
from .strict_ast_shrink4_v1 import (
    PROGRAM_SEMANTIC_IDENTITY_DOMAIN,
    ProgramAdmissionIdentityV1,
    ProgramSemanticIdentityV1,
)
from .strict_ast_v1 import CanonicalAst, StrictAstError


def _enforce_shrink5_node_limit(program: CanonicalAst) -> CanonicalAst:
    if program.metrics.node_count > MAX_TOTAL_NODE_COUNT:
        raise StrictAstError(
            STRUCTURAL_LIMIT_ERROR,
            "canonical AST exceeds max_total_node_count",
        )
    return program


def canonicalize_shrink5_source_ast(
    source_ast: object, *, migrate_legacy_scope_alias: bool = False
) -> CanonicalAst:
    """Normalize under the parent, then enforce the six-node child limit."""

    parent = _parent.canonicalize_shrink4_source_ast(
        source_ast,
        migrate_legacy_scope_alias=migrate_legacy_scope_alias,
    )
    return _enforce_shrink5_node_limit(parent)


def decode_shrink5_canonical_ast(payload: bytes) -> CanonicalAst:
    """Decode exact parent bytes, then reject a seven-node child."""

    parent = _parent.decode_shrink4_canonical_ast(payload)
    return _enforce_shrink5_node_limit(parent)


def read_legacy_parent_program(payload: bytes) -> dict[str, object]:
    """Describe one valid v1.4 program at the shrink-5 boundary."""

    parent = _parent.decode_shrink4_canonical_ast(payload)
    try:
        decode_shrink5_canonical_ast(payload)
    except StrictAstError as error:
        child_status = "REJECTED"
        child_error = error.code
    else:
        child_status = "ADMITTED"
        child_error = None
    return {
        "legacy_program_status": (
            "VALID_UNDER_PARENT_DSL_ONLY"
            if child_status == "REJECTED"
            else "VALID_UNDER_PARENT_AND_CHILD_DSL"
        ),
        "parent_dsl_version": PARENT_DSL_VERSION,
        "parent_effective_freeze_version": PARENT_FREEZE_VERSION,
        "current_dsl_version": DSL_VERSION,
        "canonical_ast_hash": parent.hash_id,
        "admitted_under_current_dsl": child_status == "ADMITTED",
        "current_dsl_error_code": child_error,
        "automatic_rewrite_or_migration_performed": False,
    }


__all__ = [
    "PROGRAM_SEMANTIC_IDENTITY_DOMAIN",
    "ProgramAdmissionIdentityV1",
    "ProgramSemanticIdentityV1",
    "canonicalize_shrink5_source_ast",
    "decode_shrink5_canonical_ast",
    "read_legacy_parent_program",
]

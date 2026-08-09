"""Strict AST admission for ``hegel-old-dsl-v1.6.0``.

Shrink step 6 delegates parsing, typing, registries, normalization, canonical
ordering, CBOR, and all shrink-5 limits to the parent boundary.  It then
rejects a normalized canonical AST whose depth exceeds three.  Parent
survivors retain byte, hash, and MDL identity; no rewrite or migration is
performed at the new boundary.
"""

from __future__ import annotations

from . import strict_ast_shrink5_v1 as _parent
from .phase3_m3_shrink6_core_v1 import (
    DSL_VERSION,
    MAX_TOTAL_AST_DEPTH,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    STRUCTURAL_LIMIT_ERROR,
)
from .strict_ast_shrink5_v1 import (
    PROGRAM_SEMANTIC_IDENTITY_DOMAIN,
    ProgramAdmissionIdentityV1,
    ProgramSemanticIdentityV1,
)
from .strict_ast_v1 import CanonicalAst, StrictAstError


def _enforce_shrink6_depth_limit(program: CanonicalAst) -> CanonicalAst:
    if program.metrics.depth > MAX_TOTAL_AST_DEPTH:
        raise StrictAstError(
            STRUCTURAL_LIMIT_ERROR,
            "canonical AST exceeds max_total_ast_depth",
        )
    return program


def canonicalize_shrink6_source_ast(
    source_ast: object, *, migrate_legacy_scope_alias: bool = False
) -> CanonicalAst:
    """Normalize under shrink-5, then enforce the depth-three child limit."""

    parent = _parent.canonicalize_shrink5_source_ast(
        source_ast,
        migrate_legacy_scope_alias=migrate_legacy_scope_alias,
    )
    return _enforce_shrink6_depth_limit(parent)


def decode_shrink6_canonical_ast(payload: bytes) -> CanonicalAst:
    """Decode exact shrink-5 bytes, then reject a depth-four child."""

    parent = _parent.decode_shrink5_canonical_ast(payload)
    return _enforce_shrink6_depth_limit(parent)


def read_legacy_parent_program(payload: bytes) -> dict[str, object]:
    """Describe one valid v1.5 program at the shrink-6 boundary."""

    parent = _parent.decode_shrink5_canonical_ast(payload)
    try:
        decode_shrink6_canonical_ast(payload)
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
    "canonicalize_shrink6_source_ast",
    "decode_shrink6_canonical_ast",
    "read_legacy_parent_program",
]

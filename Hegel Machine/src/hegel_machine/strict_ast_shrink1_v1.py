"""Strict AST admission profile for ``hegel-old-dsl-v1.1.0``.

The numeric AST schema, deterministic CBOR profile, normalization rules, and
AST hash domain are inherited byte-for-byte from the parent implementation.
The only child-language delta is an early admission rejection for aggregate
map tombstones 2, 3, and 4.  The precheck runs before the parent type checker or
canonicalizer and therefore removed maps are never executed or counted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .phase3_m3_shrink1_core_v1 import (
    ACTIVE_AGGREGATE_IDS,
    DSL_VERSION,
    REGISTRY_WIDTH,
    REMOVED_AGGREGATE_ERROR,
    TOMBSTONED_AGGREGATE_IDS,
    TOMBSTONED_AGGREGATE_NAMES,
)
from .strict_ast_v1 import (
    CanonicalAst,
    StrictAstError,
    canonicalize_source_ast as canonicalize_parent_source_ast,
    decode_canonical_ast as decode_parent_canonical_ast,
)
from .strict_cbor_v1 import canonical_cbor_decode, content_hash_id


PROGRAM_SEMANTIC_IDENTITY_DOMAIN = "HEGEL/PROGRAM_SEMANTIC_IDENTITY/V1"


def _reject_removed(detail: str) -> "None":
    raise StrictAstError(REMOVED_AGGREGATE_ERROR, detail)


def _precheck_source_tombstones(value: object) -> None:
    if not isinstance(value, (list, tuple)):
        return
    node = tuple(value)
    if not node or not isinstance(node[0], str):
        return
    name = node[0]
    if name == "aggregate" and len(node) == 5:
        map_value = node[1]
        if map_value in TOMBSTONED_AGGREGATE_NAMES or (
            type(map_value) is int and map_value in TOMBSTONED_AGGREGATE_IDS
        ):
            _reject_removed(f"aggregate map {map_value!r} is tombstoned in {DSL_VERSION}")
        return
    if name in {"bit_to_scalar", "int_to_scalar", "absolute", "sign"}:
        if len(node) == 2:
            _precheck_source_tombstones(node[1])
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
        if len(node) == 3:
            _precheck_source_tombstones(node[1])
            _precheck_source_tombstones(node[2])
        return
    if name == "approx_equal":
        if len(node) in {4, 5}:
            _precheck_source_tombstones(node[1])
            _precheck_source_tombstones(node[2])
        return
    if name != "top_level_AND" or len(node) < 2:
        return
    raw_children: tuple[object, ...]
    if len(node) == 2 and isinstance(node[1], (list, tuple)):
        possible = tuple(node[1])
        if possible and all(
            isinstance(item, (list, tuple)) and item for item in possible
        ):
            raw_children = possible
        else:
            raw_children = node[1:]
    else:
        raw_children = node[1:]
    for child in raw_children:
        _precheck_source_tombstones(child)


def _precheck_formal_node_tombstones(value: object) -> None:
    if not isinstance(value, tuple) or not value or type(value[0]) is not int:
        return
    tag = value[0]
    if tag == 0 and len(value) == 6 and type(value[1]) is int and value[1] == 3:
        map_value = value[2]
        if type(map_value) is int and map_value in TOMBSTONED_AGGREGATE_IDS:
            _reject_removed(f"formal AggregateMapId {map_value} is tombstoned in {DSL_VERSION}")
        return
    if tag == 1 and len(value) == 3:
        _precheck_formal_node_tombstones(value[2])
        return
    if tag == 2 and len(value) == 4:
        _precheck_formal_node_tombstones(value[2])
        _precheck_formal_node_tombstones(value[3])
        return
    if tag == 3 and len(value) == 5:
        _precheck_formal_node_tombstones(value[2])
        _precheck_formal_node_tombstones(value[3])
        return
    if tag == 4 and len(value) == 2 and isinstance(value[1], tuple):
        for child in value[1]:
            _precheck_formal_node_tombstones(child)


def _precheck_formal_tombstones(value: object) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or value[0] != 1
    ):
        return
    _precheck_formal_node_tombstones(value[1])


def canonicalize_shrink1_source_ast(
    source_ast: object, *, migrate_legacy_scope_alias: bool = False
) -> CanonicalAst:
    """Accept a source AST under the child sparse registry.

    Alias migration remains an explicit diagnostic-only switch inherited from
    the parent.  No removed aggregate is automatically migrated.
    """

    _precheck_source_tombstones(source_ast)
    return canonicalize_parent_source_ast(
        source_ast,
        migrate_legacy_scope_alias=migrate_legacy_scope_alias,
    )


def decode_shrink1_canonical_ast(payload: bytes) -> CanonicalAst:
    """Decode formal AST bytes after an early generic-CBOR tombstone check."""

    value = canonical_cbor_decode(payload)
    _precheck_formal_tombstones(value)
    return decode_parent_canonical_ast(payload)


def read_legacy_parent_program(payload: bytes) -> dict[str, object]:
    """Validate historical bytes under the parent DSL without child admission."""

    parent = decode_parent_canonical_ast(payload)
    try:
        decode_shrink1_canonical_ast(payload)
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
        "parent_dsl_version": "hegel-old-dsl-v1.0.0",
        "current_dsl_version": DSL_VERSION,
        "canonical_ast_hash": parent.hash_id,
        "admitted_under_current_dsl": child_status == "ADMITTED",
        "current_dsl_error_code": child_error,
        "automatic_map_migration_performed": False,
    }


def _root_digest(root: str, name: str) -> bytes:
    if not isinstance(root, str) or not root.startswith("sha256:"):
        raise ValueError(f"{name} must be a sha256 root")
    raw = root.removeprefix("sha256:")
    if len(raw) != 64:
        raise ValueError(f"{name} must carry exactly 32 digest bytes")
    try:
        return bytes.fromhex(raw)
    except ValueError as error:
        raise ValueError(f"{name} is not hexadecimal") from error


@dataclass(frozen=True, slots=True)
class ProgramAdmissionIdentityV1:
    canonical_ast_hash: str
    dsl_spec_root: str
    identifier_registry_root: str

    @property
    def canonical_value(self) -> tuple[bytes, bytes, bytes]:
        """Return the exact three-item identity frozen by the amendment."""

        return (
            _root_digest(self.canonical_ast_hash, "canonical_ast_hash"),
            _root_digest(self.dsl_spec_root, "dsl_spec_root"),
            _root_digest(self.identifier_registry_root, "identifier_registry_root"),
        )


@dataclass(frozen=True, slots=True)
class ProgramSemanticIdentityV1:
    canonical_ast_hash: str
    dsl_spec_root: str
    operator_semantics_root: str
    identifier_registry_root: str

    @property
    def content_id(self) -> str:
        value = (
            _root_digest(self.canonical_ast_hash, "canonical_ast_hash"),
            _root_digest(self.dsl_spec_root, "dsl_spec_root"),
            _root_digest(self.operator_semantics_root, "operator_semantics_root"),
            _root_digest(self.identifier_registry_root, "identifier_registry_root"),
        )
        return content_hash_id(PROGRAM_SEMANTIC_IDENTITY_DOMAIN, value)


def aggregate_id_is_active(numeric_id: int) -> bool:
    if type(numeric_id) is not int or not 0 <= numeric_id < REGISTRY_WIDTH:
        raise StrictAstError(
            "REJECT_REGISTRY_INDEX_OUT_OF_RANGE",
            f"AggregateMapId {numeric_id!r} is outside 0..{REGISTRY_WIDTH}",
        )
    if numeric_id in TOMBSTONED_AGGREGATE_IDS:
        _reject_removed(f"AggregateMapId {numeric_id} is tombstoned in {DSL_VERSION}")
    return numeric_id in ACTIVE_AGGREGATE_IDS


__all__ = [
    "PROGRAM_SEMANTIC_IDENTITY_DOMAIN",
    "ProgramAdmissionIdentityV1",
    "ProgramSemanticIdentityV1",
    "aggregate_id_is_active",
    "canonicalize_shrink1_source_ast",
    "decode_shrink1_canonical_ast",
    "read_legacy_parent_program",
]

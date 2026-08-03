"""Target-free formal record encoder used by the M3 program enumerator."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping, NoReturn


_SCHEMAS: Final = MappingProxyType(
    {
        "CanonicalProgramRecordV2": (
            0x3207,
            b"hegel-canonical-program-record/2",
            (
                "program_index",
                "canonical_ast_cbor_bytes",
                "canonical_ast_hash",
                "output_sort_id",
                "ast_depth",
                "ast_node_count",
                "distinct_bit_slot_count",
                "program_mdl_length_q32",
                "child_dsl_spec_root",
                "operator_semantics_root",
                "identifier_registry_root",
            ),
        ),
        "ProgramChunkManifestV2": (
            0x3209,
            b"hegel-program-chunk-manifest/2",
            (
                "chunk_index",
                "first_program_index",
                "last_program_index",
                "record_count",
                "canonical_program_record_subtree_root",
                "compressed_program_blob_hash",
                "uncompressed_program_byte_length",
            ),
        ),
        "BucketAccountingRecordV1": (
            0x320C,
            b"hegel-bucket-accounting-record/1",
            (
                "bucket_index",
                "output_sort_id",
                "ast_depth",
                "ast_node_count",
                "raw_operator_applications",
                "accepted_canonical_programs",
                "syntactic_duplicates",
                "type_rejections",
                "structural_limit_rejections",
                "rewrite_collapses",
                "first_program_index_or_null",
                "last_program_index_or_null",
            ),
        ),
    }
)


class M3RecordWireError(ValueError):
    pass


def _reject(detail: str) -> NoReturn:
    raise M3RecordWireError(detail)


def build_m3_record_object_v1(
    name: str, fields: Mapping[str, object]
) -> tuple[object, ...]:
    """Build one exact numeric-tag array without importing target schemas."""

    try:
        tag, schema_id, order = _SCHEMAS[name]
    except KeyError:
        _reject(f"unsupported target-free M3 record schema {name!r}")
    if not isinstance(fields, Mapping) or set(fields) != set(order):
        _reject(f"{name} field set differs")
    values = tuple(fields[field] for field in order)
    for field, value in zip(order, values, strict=True):
        if field.endswith(("_root", "_hash")):
            if type(value) is not bytes or len(value) != 32:
                _reject(f"{name}.{field} must be 32 bytes")
        elif field == "canonical_ast_cbor_bytes":
            if type(value) is not bytes or not value:
                _reject(f"{name}.{field} must be nonempty bytes")
        elif field.endswith("_or_null"):
            if value is not None and (type(value) is not int or value < 0):
                _reject(f"{name}.{field} must be uint or null")
        elif type(value) is not int or value < 0:
            _reject(f"{name}.{field} must be uint")
    return (1, tag, schema_id, *values)


__all__ = ["M3RecordWireError", "build_m3_record_object_v1"]

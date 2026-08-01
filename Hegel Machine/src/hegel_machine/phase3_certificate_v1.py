"""Exact, fail-closed contracts for Phase-3 closure and MDL certificates.

This module implements the byte-independent parts of the frozen
``hegel-freeze-p2b-p3-v1.0.1`` implementation-audit amendment:

* strict record schemas;
* RFC-6962 Merkle tree hashing (including non-power-of-two trees);
* 4,096-record chunk geometry;
* Python/Rust replay-agreement checks;
* Ed25519 role and threshold verification when ``cryptography`` is present;
* the frozen prefix-code and unsigned-Q32 MDL arithmetic helpers.

It deliberately does not issue a formal certificate.  The workspace has no
canonical-CBOR dependency, Rust replay implementation, output-archive replay,
trusted latest-key-status resolver, or complete MDL AST scorer.  JSON is never
substituted for canonical CBOR, and caller-supplied code lengths are never
accepted as formal scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from enum import Enum
from hashlib import sha256
from math import comb
import re
from types import MappingProxyType
from typing import Any, ClassVar, Final

try:  # Optional by design; pyproject.toml intentionally has no dependency.
    import cbor2 as _cbor2
except ImportError:  # pragma: no cover - availability is environment-specific.
    _cbor2 = None

try:  # Optional verifier backend; formal issuance has additional blockers.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - availability is environment-specific.
    _InvalidSignature = None
    _Ed25519PublicKey = None


FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.1"
DSL_VERSION: Final = "hegel-old-dsl-v1.0.0"
MDL_CODE_TABLE_ID: Final = "hegel-mdl-prefix-v1.0.0"
FIXED_POINT_PRECISION_ID: Final = "unsigned-q32-2^-32-bit"
EXACT_EXTENSIONAL_EQUIVALENCE: Final = "exact_extensional"

PROGRAM_RECORD_SCHEMA: Final = "closure-program-record-v1"
REPLAY_SUMMARY_SCHEMA: Final = "closure-replay-summary-v1"
OUTSIDE_CLAIM_SCHEMA: Final = "outside-frozen-closure-claim-v1"
OUTSIDE_BODY_SCHEMA: Final = "outside-certificate-body-v1"
KEY_EPOCH_SCHEMA: Final = "phase3-key-epoch-v1"
KEY_REVOCATION_SCHEMA: Final = "phase3-key-revocation-v1"
MDL_BINDINGS_SCHEMA: Final = "phase3-mdl-certificate-bindings-v1"
MDL_REQUEST_SCHEMA: Final = "phase3-mdl-replay-request-v1"

RECORDS_PER_CHUNK: Final = 4_096
MAX_CANONICAL_PROGRAM_COUNT: Final = 50_000
MAX_RAW_OPERATOR_APPLICATIONS: Final = 5_000_000
Q32_SCALE: Final = 1 << 32
MDL_DECIMAL_PRECISION: Final = 80

CANONICAL_CBOR_ENCODER_IMPLEMENTED: Final = _cbor2 is not None
ED25519_VERIFIER_IMPLEMENTED: Final = _Ed25519PublicKey is not None
CANONICAL_AST_SCHEMA_IMPLEMENTED: Final = False
PROGRAM_OUTPUT_ARCHIVE_REPLAY_IMPLEMENTED: Final = False
PYTHON_CLOSURE_REPLAY_IMPLEMENTED: Final = False
RUST_CLOSURE_REPLAY_IMPLEMENTED: Final = False
LATEST_KEY_STATUS_RESOLVER_IMPLEMENTED: Final = False
FORMAL_OUTSIDE_CERTIFICATE_ISSUANCE_IMPLEMENTED: Final = False
FORMAL_MDL_AST_SCORER_IMPLEMENTED: Final = False
PYTHON_MDL_REPLAY_IMPLEMENTED: Final = False
RUST_MDL_REPLAY_IMPLEMENTED: Final = False


SPECIFICATION_RESOLUTION_BLOCKERS: Final = (
    "canonical_cbor_backend_not_declared_as_project_dependency",
    "canonical_ast_strict_schema_and_root_operator_extraction_not_frozen",
    "program_output_blob_archive_record_and_root_schema_not_frozen",
    "canonical_program_archive_root_and_chunk_manifest_root_relation_not_frozen",
    "canonical_program_archive_root_vs_program_archive_root_alias_not_frozen",
    "diagnostic_json_id_vs_cbor_rfc6962_root_preimage_and_algorithm_bridge_not_frozen",
    "match_program_hash_identity_not_frozen",
    "exhaustion_receipt_root_preimage_exclusion_rule_not_frozen",
    "final_certificate_envelope_and_timestamp_schema_not_frozen",
    "latest_key_status_manifest_discovery_and_trust_anchor_not_frozen",
    "ed25519_public_key_and_signature_wire_encoding_not_frozen",
    "key_revocation_manifest_exact_fields_not_frozen",
    "mdl_ast_and_new_symbol_canonical_wire_schema_not_frozen",
    "new_reducer_v1_16_bit_header_value_not_frozen",
    "mdl_dual_replay_receipt_and_certificate_envelope_schema_not_frozen",
    "cross_language_q32_log2_reference_algorithm_not_frozen",
    "repository_commit_sha_hash_algorithm_and_wire_format_not_frozen",
)


class CapabilityUnavailable(RuntimeError):
    """Raised instead of silently substituting a weaker implementation."""


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        boundary = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {boundary}")
    return value


def _require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must use sha256:<lowercase-hex>")
    return value


def _require_hex(value: object, name: str, byte_length: int) -> str:
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{byte_length * 2}}}", value
    ) is None:
        raise ValueError(f"{name} must be {byte_length} lowercase-hex bytes")
    return value


def _require_tuple(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    return value


def _strict_mapping(
    value: object,
    expected_keys: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    keys = set(value)
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        unknown = sorted(keys - expected_keys)
        raise ValueError(f"{name} schema mismatch; missing={missing}, unknown={unknown}")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def sha256_id(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("SHA-256 payload must be bytes")
    return "sha256:" + sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenCborMap:
    """Immutable, deterministic map-shaped value for strict record payloads."""

    entries: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        _require_tuple(self.entries, "frozen CBOR map entries")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            for item in self.entries
        ):
            raise TypeError("frozen CBOR map entries must be nonempty string pairs")
        keys = tuple(key for key, _ in self.entries)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("frozen CBOR map keys must be unique and sorted")
        for _, value in self.entries:
            _validate_frozen_cbor_value(value)

    @classmethod
    def from_mapping(cls, value: object) -> "FrozenCborMap":
        if not isinstance(value, Mapping):
            raise TypeError("canonical CBOR object must be a mapping")
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("canonical CBOR map keys must be nonempty strings")
        return cls(
            tuple(
                (key, _freeze_cbor_value(item))
                for key, item in sorted(value.items())
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {key: _thaw_cbor_value(value) for key, value in self.entries}


def _freeze_cbor_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str, bytes}:
        return value
    if isinstance(value, Mapping):
        return FrozenCborMap.from_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cbor_value(item) for item in value)
    if isinstance(value, float):
        raise TypeError("binary floating point is forbidden in certificate CBOR")
    raise TypeError(f"unsupported certificate CBOR value: {type(value).__name__}")


def _validate_frozen_cbor_value(value: object) -> None:
    if value is None or type(value) in {bool, int, str, bytes}:
        return
    if isinstance(value, FrozenCborMap):
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_frozen_cbor_value(item)
        return
    raise TypeError(f"invalid frozen CBOR value: {type(value).__name__}")


def _thaw_cbor_value(value: object) -> object:
    if isinstance(value, FrozenCborMap):
        return value.to_mapping()
    if isinstance(value, tuple):
        return [_thaw_cbor_value(item) for item in value]
    return value


def canonical_cbor_bytes(value: Mapping[str, object] | FrozenCborMap) -> bytes:
    """Return canonical CBOR, never a JSON or ad-hoc fallback.

    ``cbor2.dumps(..., canonical=True)`` is used only when the optional backend
    exists.  The strict value walker rejects floats and non-string map keys.
    """

    if not CANONICAL_CBOR_ENCODER_IMPLEMENTED:
        raise CapabilityUnavailable(
            "canonical CBOR encoder unavailable: install/freeze a cbor2 backend"
        )
    frozen = value if isinstance(value, FrozenCborMap) else FrozenCborMap.from_mapping(value)
    assert _cbor2 is not None
    return _cbor2.dumps(frozen.to_mapping(), canonical=True)


def canonical_cbor_sha256(value: Mapping[str, object] | FrozenCborMap) -> str:
    return sha256_id(canonical_cbor_bytes(value))


def rfc6962_leaf_hash(canonical_record_bytes: bytes) -> bytes:
    if not isinstance(canonical_record_bytes, bytes):
        raise TypeError("RFC-6962 leaf payload must be bytes")
    return sha256(b"\x00" + canonical_record_bytes).digest()


def rfc6962_node_hash(left: bytes, right: bytes) -> bytes:
    if not isinstance(left, bytes) or not isinstance(right, bytes):
        raise TypeError("RFC-6962 child hashes must be bytes")
    if len(left) != 32 or len(right) != 32:
        raise ValueError("RFC-6962 child hashes must be 32 bytes")
    return sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_less_than(value: int) -> int:
    if value <= 1:
        raise ValueError("split size needs at least two leaves")
    return 1 << ((value - 1).bit_length() - 1)


def rfc6962_merkle_root(encoded_leaves: tuple[bytes, ...]) -> bytes:
    """Compute the RFC-6962 Merkle Tree Hash without duplicating tail leaves."""

    _require_tuple(encoded_leaves, "RFC-6962 leaves")
    if any(not isinstance(item, bytes) for item in encoded_leaves):
        raise TypeError("RFC-6962 leaves must contain bytes")
    if not encoded_leaves:
        return sha256(b"").digest()
    if len(encoded_leaves) == 1:
        return rfc6962_leaf_hash(encoded_leaves[0])
    split = _largest_power_of_two_less_than(len(encoded_leaves))
    return rfc6962_node_hash(
        rfc6962_merkle_root(encoded_leaves[:split]),
        rfc6962_merkle_root(encoded_leaves[split:]),
    )


def rfc6962_merkle_root_id(encoded_leaves: tuple[bytes, ...]) -> str:
    return "sha256:" + rfc6962_merkle_root(encoded_leaves).hex()


@dataclass(frozen=True, slots=True)
class ProgramRecord:
    schema_version: str
    program_index: int
    canonical_ast: FrozenCborMap
    canonical_ast_hash: str
    output_sort: str
    depth: int
    node_count: int
    distinct_entity_slot_count: int
    program_code_length_q32: int
    undefined_row_bitmap_hash: str
    output_vector_hash: str
    extensional_class_hash: str
    first_extensional_representative_index: int
    dsl_spec_root: str
    bounded_universe_root: str

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "program_index",
            "canonical_ast",
            "canonical_ast_hash",
            "output_sort",
            "depth",
            "node_count",
            "distinct_entity_slot_count",
            "program_code_length_q32",
            "undefined_row_bitmap_hash",
            "output_vector_hash",
            "extensional_class_hash",
            "first_extensional_representative_index",
            "dsl_spec_root",
            "bounded_universe_root",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != PROGRAM_RECORD_SCHEMA:
            raise ValueError("unsupported program-record schema")
        _require_int(self.program_index, "program_index")
        if not isinstance(self.canonical_ast, FrozenCborMap):
            raise TypeError("canonical_ast must be a FrozenCborMap")
        _require_sha256(self.canonical_ast_hash, "canonical_ast_hash")
        _require_nonempty_text(self.output_sort, "output_sort")
        _require_int(self.depth, "depth", maximum=4)
        _require_int(self.node_count, "node_count", maximum=7)
        _require_int(
            self.distinct_entity_slot_count,
            "distinct_entity_slot_count",
            maximum=4,
        )
        _require_int(self.program_code_length_q32, "program_code_length_q32")
        for name in (
            "undefined_row_bitmap_hash",
            "output_vector_hash",
            "extensional_class_hash",
            "dsl_spec_root",
            "bounded_universe_root",
        ):
            _require_sha256(getattr(self, name), name)
        _require_int(
            self.first_extensional_representative_index,
            "first_extensional_representative_index",
            maximum=self.program_index,
        )

    @classmethod
    def from_mapping(cls, value: object) -> "ProgramRecord":
        data = _strict_mapping(value, cls._KEYS, "program record")
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            program_index=data["program_index"],  # type: ignore[arg-type]
            canonical_ast=FrozenCborMap.from_mapping(data["canonical_ast"]),
            canonical_ast_hash=data["canonical_ast_hash"],  # type: ignore[arg-type]
            output_sort=data["output_sort"],  # type: ignore[arg-type]
            depth=data["depth"],  # type: ignore[arg-type]
            node_count=data["node_count"],  # type: ignore[arg-type]
            distinct_entity_slot_count=data["distinct_entity_slot_count"],  # type: ignore[arg-type]
            program_code_length_q32=data["program_code_length_q32"],  # type: ignore[arg-type]
            undefined_row_bitmap_hash=data["undefined_row_bitmap_hash"],  # type: ignore[arg-type]
            output_vector_hash=data["output_vector_hash"],  # type: ignore[arg-type]
            extensional_class_hash=data["extensional_class_hash"],  # type: ignore[arg-type]
            first_extensional_representative_index=data[
                "first_extensional_representative_index"
            ],  # type: ignore[arg-type]
            dsl_spec_root=data["dsl_spec_root"],  # type: ignore[arg-type]
            bounded_universe_root=data["bounded_universe_root"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "program_index": self.program_index,
            "canonical_ast": self.canonical_ast.to_mapping(),
            "canonical_ast_hash": self.canonical_ast_hash,
            "output_sort": self.output_sort,
            "depth": self.depth,
            "node_count": self.node_count,
            "distinct_entity_slot_count": self.distinct_entity_slot_count,
            "program_code_length_q32": self.program_code_length_q32,
            "undefined_row_bitmap_hash": self.undefined_row_bitmap_hash,
            "output_vector_hash": self.output_vector_hash,
            "extensional_class_hash": self.extensional_class_hash,
            "first_extensional_representative_index": (
                self.first_extensional_representative_index
            ),
            "dsl_spec_root": self.dsl_spec_root,
            "bounded_universe_root": self.bounded_universe_root,
        }

    def canonical_ast_hash_matches(self) -> bool:
        return canonical_cbor_sha256(self.canonical_ast) == self.canonical_ast_hash


@dataclass(frozen=True, slots=True)
class UniverseRow:
    universe_index: int
    input_signature_id: str
    canonical_input: FrozenCborMap
    canonical_input_hash: str

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "universe_index",
            "input_signature_id",
            "canonical_input",
            "canonical_input_hash",
        }
    )

    def __post_init__(self) -> None:
        _require_int(self.universe_index, "universe_index")
        _require_nonempty_text(self.input_signature_id, "input_signature_id")
        if not isinstance(self.canonical_input, FrozenCborMap):
            raise TypeError("canonical_input must be a FrozenCborMap")
        _require_sha256(self.canonical_input_hash, "canonical_input_hash")

    @classmethod
    def from_mapping(cls, value: object) -> "UniverseRow":
        data = _strict_mapping(value, cls._KEYS, "bounded-universe row")
        return cls(
            universe_index=data["universe_index"],  # type: ignore[arg-type]
            input_signature_id=data["input_signature_id"],  # type: ignore[arg-type]
            canonical_input=FrozenCborMap.from_mapping(data["canonical_input"]),
            canonical_input_hash=data["canonical_input_hash"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "universe_index": self.universe_index,
            "input_signature_id": self.input_signature_id,
            "canonical_input": self.canonical_input.to_mapping(),
            "canonical_input_hash": self.canonical_input_hash,
        }

    def canonical_input_hash_matches(self) -> bool:
        return canonical_cbor_sha256(self.canonical_input) == self.canonical_input_hash


@dataclass(frozen=True, slots=True)
class TargetTruthRow:
    universe_index: int
    canonical_input_hash: str
    target_output: int

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "universe_index",
            "canonical_input_hash",
            "target_output",
        }
    )

    def __post_init__(self) -> None:
        _require_int(self.universe_index, "universe_index")
        _require_sha256(self.canonical_input_hash, "canonical_input_hash")
        if type(self.target_output) is not int or self.target_output not in (0, 1):
            raise ValueError("target_output must be an integer bit")

    @classmethod
    def from_mapping(cls, value: object) -> "TargetTruthRow":
        data = _strict_mapping(value, cls._KEYS, "target-truth row")
        return cls(
            universe_index=data["universe_index"],  # type: ignore[arg-type]
            canonical_input_hash=data["canonical_input_hash"],  # type: ignore[arg-type]
            target_output=data["target_output"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "universe_index": self.universe_index,
            "canonical_input_hash": self.canonical_input_hash,
            "target_output": self.target_output,
        }


def validate_universe_and_target_rows(
    universe_rows: tuple[UniverseRow, ...],
    target_rows: tuple[TargetTruthRow, ...],
) -> None:
    _require_tuple(universe_rows, "bounded-universe rows")
    _require_tuple(target_rows, "target-truth rows")
    if not universe_rows or len(universe_rows) != len(target_rows):
        raise ValueError("universe and target rows must be nonempty and aligned")
    for index, (universe, target) in enumerate(zip(universe_rows, target_rows, strict=True)):
        if universe.universe_index != index or target.universe_index != index:
            raise ValueError("universe and target indices must be contiguous and aligned")
        if universe.canonical_input_hash != target.canonical_input_hash:
            raise ValueError("target row does not bind its bounded-universe input")


def bounded_universe_root(universe_rows: tuple[UniverseRow, ...]) -> str:
    _require_tuple(universe_rows, "bounded-universe rows")
    if any(row.universe_index != index for index, row in enumerate(universe_rows)):
        raise ValueError("bounded-universe rows must use contiguous canonical indices")
    if any(not row.canonical_input_hash_matches() for row in universe_rows):
        raise ValueError("bounded-universe row canonical_input_hash mismatch")
    return rfc6962_merkle_root_id(
        tuple(canonical_cbor_bytes(row.to_mapping()) for row in universe_rows)
    )


def target_truth_table_root(target_rows: tuple[TargetTruthRow, ...]) -> str:
    _require_tuple(target_rows, "target-truth rows")
    if any(row.universe_index != index for index, row in enumerate(target_rows)):
        raise ValueError("target rows must use contiguous canonical indices")
    return rfc6962_merkle_root_id(
        tuple(canonical_cbor_bytes(row.to_mapping()) for row in target_rows)
    )


@dataclass(frozen=True, slots=True)
class ChunkManifest:
    chunk_index: int
    first_program_index: int
    last_program_index: int
    record_count: int
    record_merkle_root: str
    compressed_blob_sha256: str
    uncompressed_byte_length: int

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "chunk_index",
            "first_program_index",
            "last_program_index",
            "record_count",
            "record_merkle_root",
            "compressed_blob_sha256",
            "uncompressed_byte_length",
        }
    )

    def __post_init__(self) -> None:
        _require_int(self.chunk_index, "chunk_index")
        _require_int(self.first_program_index, "first_program_index")
        _require_int(self.last_program_index, "last_program_index")
        _require_int(self.record_count, "record_count", minimum=1, maximum=RECORDS_PER_CHUNK)
        if self.first_program_index != self.chunk_index * RECORDS_PER_CHUNK:
            raise ValueError("chunk first index does not match 4,096-record geometry")
        if self.last_program_index != self.first_program_index + self.record_count - 1:
            raise ValueError("chunk last index disagrees with record_count")
        _require_sha256(self.record_merkle_root, "record_merkle_root")
        _require_sha256(self.compressed_blob_sha256, "compressed_blob_sha256")
        _require_int(self.uncompressed_byte_length, "uncompressed_byte_length")

    @classmethod
    def from_mapping(cls, value: object) -> "ChunkManifest":
        data = _strict_mapping(value, cls._KEYS, "chunk manifest")
        return cls(**data)  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        return {
            "chunk_index": self.chunk_index,
            "first_program_index": self.first_program_index,
            "last_program_index": self.last_program_index,
            "record_count": self.record_count,
            "record_merkle_root": self.record_merkle_root,
            "compressed_blob_sha256": self.compressed_blob_sha256,
            "uncompressed_byte_length": self.uncompressed_byte_length,
        }


def validate_chunk_manifests(
    chunks: tuple[ChunkManifest, ...],
    *,
    canonical_program_count: int,
) -> None:
    _require_tuple(chunks, "chunk manifests")
    _require_int(canonical_program_count, "canonical_program_count")
    expected_chunk_count = (
        0
        if canonical_program_count == 0
        else (canonical_program_count + RECORDS_PER_CHUNK - 1) // RECORDS_PER_CHUNK
    )
    if len(chunks) != expected_chunk_count:
        raise ValueError("chunk count does not cover the canonical program archive")
    for index, chunk in enumerate(chunks):
        if chunk.chunk_index != index:
            raise ValueError("chunk indices must be contiguous")
        expected_count = min(
            RECORDS_PER_CHUNK,
            canonical_program_count - index * RECORDS_PER_CHUNK,
        )
        if chunk.record_count != expected_count:
            raise ValueError("non-final chunks must contain exactly 4,096 records")


def chunk_manifest_root(chunks: tuple[ChunkManifest, ...]) -> str:
    _require_tuple(chunks, "chunk manifests")
    if any(chunk.chunk_index != index for index, chunk in enumerate(chunks)):
        raise ValueError("chunk manifests must use canonical chunk order")
    return rfc6962_merkle_root_id(
        tuple(canonical_cbor_bytes(chunk.to_mapping()) for chunk in chunks)
    )


@dataclass(frozen=True, slots=True)
class BucketCount:
    """One dynamic-programming bucket in an exhaustion receipt."""

    output_sort: str
    depth: int
    node_count: int
    raw_operator_applications: int
    accepted_canonical_programs: int
    canonical_duplicates: int
    type_rejections: int
    limit_rejections: int

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "output_sort",
            "depth",
            "node_count",
            "raw_operator_applications",
            "accepted_canonical_programs",
            "canonical_duplicates",
            "type_rejections",
            "limit_rejections",
        }
    )

    def __post_init__(self) -> None:
        _require_nonempty_text(self.output_sort, "output_sort")
        _require_int(self.depth, "depth", maximum=4)
        _require_int(self.node_count, "node_count", maximum=7)
        for field_name in (
            "raw_operator_applications",
            "accepted_canonical_programs",
            "canonical_duplicates",
            "type_rejections",
            "limit_rejections",
        ):
            _require_int(getattr(self, field_name), field_name)

    @property
    def canonical_key(self) -> tuple[str, int, int]:
        return (self.output_sort, self.depth, self.node_count)

    @classmethod
    def from_mapping(cls, value: object) -> "BucketCount":
        data = _strict_mapping(value, cls._KEYS, "exhaustion bucket count")
        return cls(**data)  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        return {
            "output_sort": self.output_sort,
            "depth": self.depth,
            "node_count": self.node_count,
            "raw_operator_applications": self.raw_operator_applications,
            "accepted_canonical_programs": self.accepted_canonical_programs,
            "canonical_duplicates": self.canonical_duplicates,
            "type_rejections": self.type_rejections,
            "limit_rejections": self.limit_rejections,
        }


@dataclass(frozen=True, slots=True)
class ExhaustionReceipt:
    """The exact receipt fields frozen in section 9.8.

    ``exhaustion_receipt_root`` is validated as a digest but is intentionally
    not recomputed: the freeze includes the root in the record without defining
    whether that field is omitted, nulled, or domain-separated in its preimage.
    """

    implementation_id: str
    dsl_spec_root: str
    bucket_counts: tuple[BucketCount, ...]
    raw_operator_application_count: int
    canonical_program_count: int
    frontier_exhausted: bool
    program_archive_root: str
    output_archive_root: str
    exhaustion_receipt_root: str

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "implementation_id",
            "dsl_spec_root",
            "bucket_counts",
            "raw_operator_application_count",
            "canonical_program_count",
            "frontier_exhausted",
            "program_archive_root",
            "output_archive_root",
            "exhaustion_receipt_root",
        }
    )

    def __post_init__(self) -> None:
        _require_nonempty_text(self.implementation_id, "implementation_id")
        _require_sha256(self.dsl_spec_root, "dsl_spec_root")
        _require_tuple(self.bucket_counts, "bucket_counts")
        if not self.bucket_counts or any(
            not isinstance(bucket, BucketCount) for bucket in self.bucket_counts
        ):
            raise TypeError("bucket_counts must be a nonempty tuple of BucketCount")
        keys = tuple(bucket.canonical_key for bucket in self.bucket_counts)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("bucket_counts must be unique and canonically sorted")
        _require_int(
            self.raw_operator_application_count,
            "raw_operator_application_count",
        )
        _require_int(
            self.canonical_program_count,
            "canonical_program_count",
        )
        _require_bool(self.frontier_exhausted, "frontier_exhausted")
        if sum(item.raw_operator_applications for item in self.bucket_counts) != (
            self.raw_operator_application_count
        ):
            raise ValueError("bucket raw counts do not sum to the receipt total")
        if sum(item.accepted_canonical_programs for item in self.bucket_counts) != (
            self.canonical_program_count
        ):
            raise ValueError("bucket accepted counts do not sum to the receipt total")
        for field_name in (
            "program_archive_root",
            "output_archive_root",
            "exhaustion_receipt_root",
        ):
            _require_sha256(getattr(self, field_name), field_name)

    @classmethod
    def from_mapping(cls, value: object) -> "ExhaustionReceipt":
        data = _strict_mapping(value, cls._KEYS, "exhaustion receipt")
        raw_buckets = data["bucket_counts"]
        if not isinstance(raw_buckets, list):
            raise TypeError("exhaustion receipt bucket_counts must be a list")
        return cls(
            implementation_id=data["implementation_id"],  # type: ignore[arg-type]
            dsl_spec_root=data["dsl_spec_root"],  # type: ignore[arg-type]
            bucket_counts=tuple(BucketCount.from_mapping(item) for item in raw_buckets),
            raw_operator_application_count=data[
                "raw_operator_application_count"
            ],  # type: ignore[arg-type]
            canonical_program_count=data["canonical_program_count"],  # type: ignore[arg-type]
            frontier_exhausted=data["frontier_exhausted"],  # type: ignore[arg-type]
            program_archive_root=data["program_archive_root"],  # type: ignore[arg-type]
            output_archive_root=data["output_archive_root"],  # type: ignore[arg-type]
            exhaustion_receipt_root=data[
                "exhaustion_receipt_root"
            ],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "implementation_id": self.implementation_id,
            "dsl_spec_root": self.dsl_spec_root,
            "bucket_counts": [item.to_mapping() for item in self.bucket_counts],
            "raw_operator_application_count": self.raw_operator_application_count,
            "canonical_program_count": self.canonical_program_count,
            "frontier_exhausted": self.frontier_exhausted,
            "program_archive_root": self.program_archive_root,
            "output_archive_root": self.output_archive_root,
            "exhaustion_receipt_root": self.exhaustion_receipt_root,
        }


class ReplayLanguage(str, Enum):
    PYTHON = "python"
    RUST = "rust"


class ReplayStatus(str, Enum):
    COMPLETE = "COMPLETE"
    INCONCLUSIVE_BUDGET = "INCONCLUSIVE_BUDGET"
    DSL_TOO_LARGE = "DSL_TOO_LARGE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Source-bound replay result; counts are taken only from its receipt."""

    language: ReplayLanguage
    status: ReplayStatus
    receipt: ExhaustionReceipt
    operator_semantics_root: str
    identifier_registry_root: str
    canonicalizer_source_root: str
    enumerator_source_root: str
    evaluator_source_root: str
    bounded_universe_root: str
    target_truth_table_root: str
    chunk_manifest_root: str
    match_program_hashes: tuple[str, ...]
    undefined_target_row_count: int
    raw_expansion_limit_hit: bool
    wall_clock_abort_hit: bool
    all_type_buckets_closed: bool
    schema_version: str = REPLAY_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SUMMARY_SCHEMA:
            raise ValueError("unsupported replay-summary schema")
        if not isinstance(self.language, ReplayLanguage):
            raise TypeError("language must be ReplayLanguage")
        if not isinstance(self.status, ReplayStatus):
            raise TypeError("status must be ReplayStatus")
        if not isinstance(self.receipt, ExhaustionReceipt):
            raise TypeError("receipt must be ExhaustionReceipt")
        expected_prefix = self.language.value
        if not self.receipt.implementation_id.lower().startswith(expected_prefix):
            raise ValueError("receipt implementation_id does not identify replay language")
        for field_name in (
            "operator_semantics_root",
            "identifier_registry_root",
            "canonicalizer_source_root",
            "enumerator_source_root",
            "evaluator_source_root",
            "bounded_universe_root",
            "target_truth_table_root",
            "chunk_manifest_root",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        _require_tuple(self.match_program_hashes, "match_program_hashes")
        for item in self.match_program_hashes:
            _require_sha256(item, "match_program_hash")
        if self.match_program_hashes != tuple(sorted(set(self.match_program_hashes))):
            raise ValueError("match_program_hashes must be sorted and unique")
        _require_int(self.undefined_target_row_count, "undefined_target_row_count")
        _require_bool(self.raw_expansion_limit_hit, "raw_expansion_limit_hit")
        _require_bool(self.wall_clock_abort_hit, "wall_clock_abort_hit")
        _require_bool(self.all_type_buckets_closed, "all_type_buckets_closed")

    @property
    def implementation_id(self) -> str:
        return self.receipt.implementation_id

    @property
    def canonical_program_count(self) -> int:
        return self.receipt.canonical_program_count

    @property
    def match_set_count(self) -> int:
        return len(self.match_program_hashes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "language": self.language.value,
            "status": self.status.value,
            "receipt": self.receipt.to_mapping(),
            "operator_semantics_root": self.operator_semantics_root,
            "identifier_registry_root": self.identifier_registry_root,
            "canonicalizer_source_root": self.canonicalizer_source_root,
            "enumerator_source_root": self.enumerator_source_root,
            "evaluator_source_root": self.evaluator_source_root,
            "bounded_universe_root": self.bounded_universe_root,
            "target_truth_table_root": self.target_truth_table_root,
            "chunk_manifest_root": self.chunk_manifest_root,
            "match_program_hashes": list(self.match_program_hashes),
            "undefined_target_row_count": self.undefined_target_row_count,
            "raw_expansion_limit_hit": self.raw_expansion_limit_hit,
            "wall_clock_abort_hit": self.wall_clock_abort_hit,
            "all_type_buckets_closed": self.all_type_buckets_closed,
        }


@dataclass(frozen=True, slots=True)
class ReplayAgreement:
    python: ReplaySummary
    rust: ReplaySummary

    def __post_init__(self) -> None:
        if not isinstance(self.python, ReplaySummary) or not isinstance(
            self.rust, ReplaySummary
        ):
            raise TypeError("replay agreement requires two ReplaySummary objects")
        if self.python.language is not ReplayLanguage.PYTHON:
            raise ValueError("python replay has the wrong implementation language")
        if self.rust.language is not ReplayLanguage.RUST:
            raise ValueError("rust replay has the wrong implementation language")

    def agreement_failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.python.implementation_id == self.rust.implementation_id:
            failures.append("replay_implementation_ids_not_independent")
        for field_name in (
            "canonicalizer_source_root",
            "enumerator_source_root",
            "evaluator_source_root",
        ):
            if getattr(self.python, field_name) == getattr(self.rust, field_name):
                failures.append(f"shared_{field_name}")
        equal_fields = {
            "dsl_spec_root": (
                self.python.receipt.dsl_spec_root,
                self.rust.receipt.dsl_spec_root,
            ),
            "operator_semantics_root": (
                self.python.operator_semantics_root,
                self.rust.operator_semantics_root,
            ),
            "identifier_registry_root": (
                self.python.identifier_registry_root,
                self.rust.identifier_registry_root,
            ),
            "canonical_program_count": (
                self.python.canonical_program_count,
                self.rust.canonical_program_count,
            ),
            "program_archive_root": (
                self.python.receipt.program_archive_root,
                self.rust.receipt.program_archive_root,
            ),
            "output_archive_root": (
                self.python.receipt.output_archive_root,
                self.rust.receipt.output_archive_root,
            ),
            "match_set_count": (
                self.python.match_set_count,
                self.rust.match_set_count,
            ),
            "match_program_hashes": (
                self.python.match_program_hashes,
                self.rust.match_program_hashes,
            ),
            "bounded_universe_root": (
                self.python.bounded_universe_root,
                self.rust.bounded_universe_root,
            ),
            "target_truth_table_root": (
                self.python.target_truth_table_root,
                self.rust.target_truth_table_root,
            ),
            "chunk_manifest_root": (
                self.python.chunk_manifest_root,
                self.rust.chunk_manifest_root,
            ),
        }
        failures.extend(
            f"replay_{name}_mismatch"
            for name, pair in equal_fields.items()
            if pair[0] != pair[1]
        )
        return tuple(failures)

    def outside_condition_failures(self) -> tuple[str, ...]:
        failures = list(self.agreement_failures())
        for replay in (self.python, self.rust):
            prefix = replay.language.value
            if replay.status is not ReplayStatus.COMPLETE:
                failures.append(f"{prefix}_replay_not_complete")
            if not replay.receipt.frontier_exhausted:
                failures.append(f"{prefix}_frontier_not_exhausted")
            if not replay.all_type_buckets_closed:
                failures.append(f"{prefix}_type_buckets_not_closed")
            if replay.raw_expansion_limit_hit:
                failures.append(f"{prefix}_raw_expansion_limit_hit")
            if replay.wall_clock_abort_hit:
                failures.append(f"{prefix}_wall_clock_abort_hit")
            if replay.match_set_count != 0:
                failures.append(f"{prefix}_match_set_not_empty")
            if replay.undefined_target_row_count != 0:
                failures.append(f"{prefix}_undefined_target_rows")
            if replay.canonical_program_count > MAX_CANONICAL_PROGRAM_COUNT:
                failures.append(f"{prefix}_canonical_program_limit_exceeded")
            if (
                replay.receipt.raw_operator_application_count
                > MAX_RAW_OPERATOR_APPLICATIONS
            ):
                failures.append(f"{prefix}_raw_operator_application_limit_exceeded")
        return tuple(failures)

    def to_mapping(self) -> dict[str, object]:
        return {
            "python": self.python.to_mapping(),
            "rust": self.rust.to_mapping(),
        }


class DslSpecStatus(str, Enum):
    FROZEN = "FROZEN"
    DRAFT = "DRAFT"


@dataclass(frozen=True, slots=True)
class OutsideFrozenClosureClaim:
    """The only bounded language-exclusion claim authorized by the freeze."""

    dsl_version: str
    bounded_universe_root: str
    target_truth_table_root: str
    equivalence: str = EXACT_EXTENSIONAL_EQUIVALENCE
    claim_kind: str = "OUTSIDE_FROZEN_CLOSURE"
    schema_version: str = OUTSIDE_CLAIM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OUTSIDE_CLAIM_SCHEMA:
            raise ValueError("unsupported outside-claim schema")
        if self.claim_kind != "OUTSIDE_FROZEN_CLOSURE":
            raise ValueError("the only authorized claim is OUTSIDE_FROZEN_CLOSURE")
        if self.dsl_version != DSL_VERSION:
            raise ValueError("claim must bind the frozen old-DSL version")
        _require_sha256(self.bounded_universe_root, "bounded_universe_root")
        _require_sha256(self.target_truth_table_root, "target_truth_table_root")
        if self.equivalence != EXACT_EXTENSIONAL_EQUIVALENCE:
            raise ValueError("claim equivalence must be exact_extensional")

    def render(self) -> str:
        return (
            "OUTSIDE_FROZEN_CLOSURE("
            f"{self.dsl_version},"
            f"{self.bounded_universe_root},"
            f"{self.target_truth_table_root},"
            "equivalence=exact_extensional)"
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "claim_kind": self.claim_kind,
            "dsl_version": self.dsl_version,
            "bounded_universe_root": self.bounded_universe_root,
            "target_truth_table_root": self.target_truth_table_root,
            "equivalence": self.equivalence,
        }


def _require_repository_commit(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be a 40-character lowercase Git commit")
    return value


@dataclass(frozen=True, slots=True)
class ReplayEnvironmentBinding:
    """Signed provenance for one language-specific replay environment."""

    language: ReplayLanguage
    replay_implementation_id: str
    repository_commit_sha: str
    container_image_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.language, ReplayLanguage):
            raise TypeError("language must be ReplayLanguage")
        _require_nonempty_text(
            self.replay_implementation_id,
            "replay_implementation_id",
        )
        expected_prefix = self.language.value
        if not self.replay_implementation_id.lower().startswith(expected_prefix):
            raise ValueError(
                "replay_implementation_id does not identify binding language"
            )
        _require_repository_commit(
            self.repository_commit_sha,
            "repository_commit_sha",
        )
        _require_sha256(self.container_image_digest, "container_image_digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "language": self.language.value,
            "replay_implementation_id": self.replay_implementation_id,
            "repository_commit_sha": self.repository_commit_sha,
            "container_image_digest": self.container_image_digest,
        }


@dataclass(frozen=True, slots=True)
class OutsideCertificateBody:
    """Versioned internal body pending a formally frozen envelope schema."""

    claim: OutsideFrozenClosureClaim
    dsl_spec_status: DslSpecStatus
    target_commitment_precedes_synthesis: bool
    replay_agreement: ReplayAgreement
    covert_channel_audit_pass: bool
    key_epoch: int
    issued_at: str
    python_replay_environment: ReplayEnvironmentBinding
    rust_replay_environment: ReplayEnvironmentBinding
    schema_version: str = OUTSIDE_BODY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != OUTSIDE_BODY_SCHEMA:
            raise ValueError("unsupported outside-certificate body schema")
        if not isinstance(self.claim, OutsideFrozenClosureClaim):
            raise TypeError("claim must be OutsideFrozenClosureClaim")
        if not isinstance(self.dsl_spec_status, DslSpecStatus):
            raise TypeError("dsl_spec_status must be DslSpecStatus")
        _require_bool(
            self.target_commitment_precedes_synthesis,
            "target_commitment_precedes_synthesis",
        )
        if not isinstance(self.replay_agreement, ReplayAgreement):
            raise TypeError("replay_agreement must be ReplayAgreement")
        _require_bool(self.covert_channel_audit_pass, "covert_channel_audit_pass")
        _require_int(self.key_epoch, "key_epoch", minimum=1)
        _require_nonempty_text(self.issued_at, "issued_at")
        for field_name, expected_language, replay in (
            (
                "python_replay_environment",
                ReplayLanguage.PYTHON,
                self.replay_agreement.python,
            ),
            (
                "rust_replay_environment",
                ReplayLanguage.RUST,
                self.replay_agreement.rust,
            ),
        ):
            environment = getattr(self, field_name)
            if not isinstance(environment, ReplayEnvironmentBinding):
                raise TypeError(f"{field_name} must be ReplayEnvironmentBinding")
            if environment.language is not expected_language:
                raise ValueError(f"{field_name} has the wrong replay language")
            if environment.replay_implementation_id != replay.implementation_id:
                raise ValueError(
                    f"{field_name} does not bind its replay implementation"
                )

    def machine_condition_failures(self) -> tuple[str, ...]:
        failures = list(self.replay_agreement.outside_condition_failures())
        if self.dsl_spec_status is not DslSpecStatus.FROZEN:
            failures.append("dsl_spec_not_frozen")
        if not self.target_commitment_precedes_synthesis:
            failures.append("target_commitment_did_not_precede_synthesis")
        if not self.covert_channel_audit_pass:
            failures.append("covert_channel_audit_failed")
        python = self.replay_agreement.python
        if self.claim.bounded_universe_root != python.bounded_universe_root:
            failures.append("claim_bounded_universe_root_mismatch")
        if self.claim.target_truth_table_root != python.target_truth_table_root:
            failures.append("claim_target_truth_table_root_mismatch")
        return tuple(failures)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "claim": self.claim.to_mapping(),
            "dsl_spec_status": self.dsl_spec_status.value,
            "target_commitment_precedes_synthesis": (
                self.target_commitment_precedes_synthesis
            ),
            "replay_agreement": self.replay_agreement.to_mapping(),
            "covert_channel_audit_pass": self.covert_channel_audit_pass,
            "key_epoch": self.key_epoch,
            "issued_at": self.issued_at,
            "python_replay_environment": self.python_replay_environment.to_mapping(),
            "rust_replay_environment": self.rust_replay_environment.to_mapping(),
        }


class KeyRole(str, Enum):
    CUSTODIAN = "K_custodian"
    REPLAY_PYTHON = "K_replay_python"
    REPLAY_RUST = "K_replay_rust"


FORMAL_CERTIFICATE_ROLES: Final = (
    KeyRole.CUSTODIAN,
    KeyRole.REPLAY_PYTHON,
    KeyRole.REPLAY_RUST,
)


@dataclass(frozen=True, slots=True)
class Ed25519PublicKeyRecord:
    """A provisional raw-byte key encoding; formal wire encoding is unresolved."""

    role: KeyRole
    key_id: str
    key_epoch: int
    public_key_hex: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, KeyRole):
            raise TypeError("role must be KeyRole")
        _require_nonempty_text(self.key_id, "key_id")
        _require_int(self.key_epoch, "key_epoch", minimum=1)
        _require_hex(self.public_key_hex, "public_key_hex", 32)

    def to_mapping(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "key_id": self.key_id,
            "key_epoch": self.key_epoch,
            "public_key_hex": self.public_key_hex,
        }


@dataclass(frozen=True, slots=True)
class DetachedSignature:
    role: KeyRole
    key_id: str
    key_epoch: int
    signature_hex: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, KeyRole):
            raise TypeError("role must be KeyRole")
        _require_nonempty_text(self.key_id, "key_id")
        _require_int(self.key_epoch, "key_epoch", minimum=1)
        _require_hex(self.signature_hex, "signature_hex", 64)

    def to_mapping(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "key_id": self.key_id,
            "key_epoch": self.key_epoch,
            "signature_hex": self.signature_hex,
        }


@dataclass(frozen=True, slots=True)
class SignatureVerification:
    threshold: int
    valid_roles: tuple[KeyRole, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return len(self.valid_roles) >= self.threshold and not self.failures


def verify_ed25519_digest(
    digest: bytes,
    signatures: tuple[DetachedSignature, ...],
    public_keys: tuple[Ed25519PublicKeyRecord, ...],
    *,
    threshold: int,
    required_roles: tuple[KeyRole, ...] | None = None,
    key_epoch: int | None = None,
) -> SignatureVerification:
    """Verify role-distinct signatures over an already computed SHA-256 digest."""

    if not isinstance(digest, bytes) or len(digest) != 32:
        raise ValueError("signed digest must be exactly 32 bytes")
    _require_tuple(signatures, "detached signatures")
    _require_tuple(public_keys, "public keys")
    _require_int(threshold, "threshold", minimum=1, maximum=3)
    if required_roles is not None:
        _require_tuple(required_roles, "required roles")
        if len(required_roles) != len(set(required_roles)):
            raise ValueError("required roles must be unique")
        if any(not isinstance(role, KeyRole) for role in required_roles):
            raise TypeError("required roles must contain KeyRole values")

    failures: list[str] = []
    if not ED25519_VERIFIER_IMPLEMENTED:
        return SignatureVerification(threshold, (), ("ed25519_backend_unavailable",))

    key_by_id: dict[str, Ed25519PublicKeyRecord] = {}
    key_roles: set[KeyRole] = set()
    for key in public_keys:
        if not isinstance(key, Ed25519PublicKeyRecord):
            raise TypeError("public_keys must contain Ed25519PublicKeyRecord")
        if key.key_id in key_by_id:
            failures.append("duplicate_public_key_id")
        if key.role in key_roles:
            failures.append("duplicate_public_key_role")
        key_by_id[key.key_id] = key
        key_roles.add(key.role)

    seen_signature_ids: set[str] = set()
    seen_signature_roles: set[KeyRole] = set()
    valid_roles: list[KeyRole] = []
    for signature in signatures:
        if not isinstance(signature, DetachedSignature):
            raise TypeError("signatures must contain DetachedSignature")
        if signature.key_id in seen_signature_ids:
            failures.append("duplicate_signature_key_id")
            continue
        if signature.role in seen_signature_roles:
            failures.append("duplicate_signature_role")
            continue
        seen_signature_ids.add(signature.key_id)
        seen_signature_roles.add(signature.role)
        key = key_by_id.get(signature.key_id)
        if key is None:
            failures.append(f"unknown_signature_key:{signature.key_id}")
            continue
        if key.role is not signature.role:
            failures.append(f"signature_role_mismatch:{signature.key_id}")
            continue
        if key.key_epoch != signature.key_epoch:
            failures.append(f"signature_key_epoch_mismatch:{signature.key_id}")
            continue
        if key_epoch is not None and signature.key_epoch != key_epoch:
            failures.append(f"signature_not_in_required_epoch:{signature.key_id}")
            continue
        assert _Ed25519PublicKey is not None
        assert _InvalidSignature is not None
        try:
            verifier = _Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(key.public_key_hex)
            )
            verifier.verify(bytes.fromhex(signature.signature_hex), digest)
        except (_InvalidSignature, ValueError):
            failures.append(f"invalid_signature:{signature.key_id}")
            continue
        valid_roles.append(signature.role)

    if required_roles is not None:
        missing = set(required_roles) - set(valid_roles)
        failures.extend(
            f"missing_required_signature:{role.value}"
            for role in sorted(missing, key=lambda item: item.value)
        )
    if len(valid_roles) < threshold:
        failures.append("signature_threshold_not_met")
    return SignatureVerification(
        threshold=threshold,
        valid_roles=tuple(sorted(valid_roles, key=lambda item: item.value)),
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class KeyEpochManifest:
    key_epoch: int
    previous_key_epoch: int
    new_public_keys: tuple[Ed25519PublicKeyRecord, ...]
    effective_at: str
    reason: str
    invalidate_certificates_before: str | None
    schema_version: str = KEY_EPOCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != KEY_EPOCH_SCHEMA:
            raise ValueError("unsupported key-epoch schema")
        _require_int(self.key_epoch, "key_epoch", minimum=2)
        _require_int(self.previous_key_epoch, "previous_key_epoch", minimum=1)
        if self.previous_key_epoch != self.key_epoch - 1:
            raise ValueError("key epochs must form an adjacent chain")
        _require_tuple(self.new_public_keys, "new_public_keys")
        if len(self.new_public_keys) != 3:
            raise ValueError("key epoch must install exactly three role keys")
        if {key.role for key in self.new_public_keys} != set(FORMAL_CERTIFICATE_ROLES):
            raise ValueError("key epoch must install all three formal roles")
        if any(key.key_epoch != self.key_epoch for key in self.new_public_keys):
            raise ValueError("new key records must bind the new key epoch")
        if len({key.key_id for key in self.new_public_keys}) != 3:
            raise ValueError("new key ids must be unique")
        _require_nonempty_text(self.effective_at, "effective_at")
        _require_nonempty_text(self.reason, "reason")
        if self.invalidate_certificates_before is not None:
            _require_nonempty_text(
                self.invalidate_certificates_before,
                "invalidate_certificates_before",
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "key_epoch": self.key_epoch,
            "previous_key_epoch": self.previous_key_epoch,
            "new_public_keys": [key.to_mapping() for key in self.new_public_keys],
            "effective_at": self.effective_at,
            "reason": self.reason,
            "invalidate_certificates_before": self.invalidate_certificates_before,
        }


@dataclass(frozen=True, slots=True)
class KeyRevocationManifest:
    key_epoch: int
    revoked_key_ids: tuple[str, ...]
    effective_at: str
    reason: str
    invalidate_certificates_before: str | None
    invalidate_certificates_after: str | None
    schema_version: str = KEY_REVOCATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != KEY_REVOCATION_SCHEMA:
            raise ValueError("unsupported key-revocation schema")
        _require_int(self.key_epoch, "key_epoch", minimum=1)
        _require_tuple(self.revoked_key_ids, "revoked_key_ids")
        if not self.revoked_key_ids or any(
            not isinstance(item, str) or not item for item in self.revoked_key_ids
        ):
            raise TypeError("revoked_key_ids must be a nonempty tuple of key ids")
        if self.revoked_key_ids != tuple(sorted(set(self.revoked_key_ids))):
            raise ValueError("revoked_key_ids must be sorted and unique")
        _require_nonempty_text(self.effective_at, "effective_at")
        _require_nonempty_text(self.reason, "reason")
        for field_name in (
            "invalidate_certificates_before",
            "invalidate_certificates_after",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_nonempty_text(value, field_name)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "key_epoch": self.key_epoch,
            "revoked_key_ids": list(self.revoked_key_ids),
            "effective_at": self.effective_at,
            "reason": self.reason,
            "invalidate_certificates_before": self.invalidate_certificates_before,
            "invalidate_certificates_after": self.invalidate_certificates_after,
        }


def verify_rotation_or_revocation_signatures(
    manifest_without_signatures: Mapping[str, object] | FrozenCborMap,
    signatures: tuple[DetachedSignature, ...],
    old_epoch_keys: tuple[Ed25519PublicKeyRecord, ...],
    *,
    old_key_epoch: int,
) -> SignatureVerification:
    """Apply the frozen old-epoch 2-of-3 rule to a key-status manifest."""

    _require_tuple(signatures, "detached signatures")
    _require_tuple(old_epoch_keys, "old epoch trust store")
    _require_int(old_key_epoch, "old_key_epoch", minimum=1)
    if any(not isinstance(key, Ed25519PublicKeyRecord) for key in old_epoch_keys):
        raise TypeError(
            "old epoch trust store must contain Ed25519PublicKeyRecord objects"
        )

    trust_store_failures: list[str] = []
    if len(old_epoch_keys) != len(FORMAL_CERTIFICATE_ROLES):
        trust_store_failures.append("old_epoch_trust_store_size_not_three")
    roles = tuple(key.role for key in old_epoch_keys)
    if len(roles) != len(set(roles)):
        trust_store_failures.append("old_epoch_trust_store_duplicate_role")
    missing_roles = set(FORMAL_CERTIFICATE_ROLES) - set(roles)
    trust_store_failures.extend(
        f"old_epoch_trust_store_missing_role:{role.value}"
        for role in sorted(missing_roles, key=lambda item: item.value)
    )
    key_ids = tuple(key.key_id for key in old_epoch_keys)
    if len(key_ids) != len(set(key_ids)):
        trust_store_failures.append("old_epoch_trust_store_duplicate_key_id")
    trust_store_failures.extend(
        f"old_epoch_trust_store_key_epoch_mismatch:{key.key_id}"
        for key in old_epoch_keys
        if key.key_epoch != old_key_epoch
    )
    if trust_store_failures:
        return SignatureVerification(
            threshold=2,
            valid_roles=(),
            failures=tuple(trust_store_failures),
        )

    digest = sha256(canonical_cbor_bytes(manifest_without_signatures)).digest()
    return verify_ed25519_digest(
        digest,
        signatures,
        old_epoch_keys,
        threshold=2,
        key_epoch=old_key_epoch,
    )


@dataclass(frozen=True, slots=True)
class OutsideCertificateEnvelope:
    body: OutsideCertificateBody
    signatures: tuple[DetachedSignature, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.body, OutsideCertificateBody):
            raise TypeError("body must be OutsideCertificateBody")
        _require_tuple(self.signatures, "certificate signatures")
        if any(not isinstance(item, DetachedSignature) for item in self.signatures):
            raise TypeError("certificate signatures must be DetachedSignature records")


@dataclass(frozen=True, slots=True)
class OutsideCertificateAssessment:
    issued: bool
    claim: str | None
    verified_roles: tuple[KeyRole, ...]
    failures: tuple[str, ...]


def outside_certificate_capability_failures() -> tuple[str, ...]:
    checks = (
        (
            FORMAL_OUTSIDE_CERTIFICATE_ISSUANCE_IMPLEMENTED,
            "formal_outside_certificate_issuance_unimplemented",
        ),
        (CANONICAL_CBOR_ENCODER_IMPLEMENTED, "canonical_cbor_unavailable"),
        (CANONICAL_AST_SCHEMA_IMPLEMENTED, "canonical_ast_schema_unimplemented"),
        (
            PROGRAM_OUTPUT_ARCHIVE_REPLAY_IMPLEMENTED,
            "program_output_archive_replay_unimplemented",
        ),
        (PYTHON_CLOSURE_REPLAY_IMPLEMENTED, "python_closure_replay_unimplemented"),
        (RUST_CLOSURE_REPLAY_IMPLEMENTED, "rust_closure_replay_unimplemented"),
        (
            LATEST_KEY_STATUS_RESOLVER_IMPLEMENTED,
            "latest_key_status_resolver_unimplemented",
        ),
        (ED25519_VERIFIER_IMPLEMENTED, "ed25519_verifier_unavailable"),
    )
    return tuple(reason for implemented, reason in checks if not implemented)


def verify_outside_certificate(
    envelope: OutsideCertificateEnvelope,
    public_keys: tuple[Ed25519PublicKeyRecord, ...],
    *,
    latest_key_epoch: int | None,
) -> OutsideCertificateAssessment:
    """Fail closed unless every frozen condition and local capability exists."""

    if not isinstance(envelope, OutsideCertificateEnvelope):
        raise TypeError("envelope must be OutsideCertificateEnvelope")
    _require_tuple(public_keys, "public_keys")
    failures = list(outside_certificate_capability_failures())
    failures.extend(envelope.body.machine_condition_failures())
    verified_roles: tuple[KeyRole, ...] = ()

    if latest_key_epoch is None:
        failures.append("latest_key_epoch_unresolved")
    elif type(latest_key_epoch) is not int or latest_key_epoch < 1:
        raise ValueError("latest_key_epoch must be a positive integer or None")
    elif latest_key_epoch != envelope.body.key_epoch:
        failures.append("certificate_not_in_latest_key_epoch")

    if CANONICAL_CBOR_ENCODER_IMPLEMENTED and ED25519_VERIFIER_IMPLEMENTED:
        digest = sha256(canonical_cbor_bytes(envelope.body.to_mapping())).digest()
        signature_result = verify_ed25519_digest(
            digest,
            envelope.signatures,
            public_keys,
            threshold=3,
            required_roles=FORMAL_CERTIFICATE_ROLES,
            key_epoch=envelope.body.key_epoch,
        )
        verified_roles = signature_result.valid_roles
        failures.extend(signature_result.failures)
    elif len(envelope.signatures) != 3:
        failures.append("formal_certificate_requires_exactly_three_signatures")

    failures = list(dict.fromkeys(failures))
    issued = not failures
    return OutsideCertificateAssessment(
        issued=issued,
        claim=envelope.body.claim.render() if issued else None,
        verified_roles=verified_roles,
        failures=tuple(failures),
    )


AST_SHAPE_PREFIXES: Final = MappingProxyType(
    {
        "leaf": "00",
        "unary": "01",
        "binary": "10",
        "ternary": "110",
        "top_level_and_1": "1110",
        "top_level_and_2": "11110",
        "top_level_and_3": "111110",
        "reserved": "111111",
    }
)
LEAF_CLASS_CODES: Final = MappingProxyType(
    {
        "scalar_const": "000",
        "bit_at": "001",
        "set_size": "010",
        "aggregate": "011",
        "context_flag": "100",
        "task_flag": "101",
        "new_symbol_call": "110",
        "reserved": "111",
    }
)
UNARY_TOKEN_CODES: Final = MappingProxyType(
    {
        "bit_to_scalar": "00",
        "int_to_scalar": "01",
        "absolute": "10",
        "sign": "11",
    }
)
BINARY_TOKEN_CODES: Final = MappingProxyType(
    {
        "add": "000",
        "difference": "001",
        "equal_exact": "010",
        "less_equal": "011",
        "greater_equal": "100",
        "same_sign": "101",
        "opposite_sign": "110",
        "reserved": "111",
    }
)
TERNARY_TOKEN_CODES: Final = MappingProxyType(
    {"approx_equal": "0", "reserved": "1"}
)
RATIONAL_PARAMETER_CODES: Final = MappingProxyType(
    {
        "-2": "000",
        "-1": "001",
        "-1/2": "010",
        "0": "011",
        "1/2": "100",
        "1": "101",
        "2": "110",
        "reserved": "111",
    }
)
TOLERANCE_CODES: Final = MappingProxyType(
    {"0": "00", "1/4": "01", "1/2": "10", "reserved": "11"}
)
SCOPE_CLAUSE_COUNT_CODES: Final = MappingProxyType({0: "0", 1: "10", 2: "11"})


def prefix_code_is_prefix_free(codes: tuple[str, ...]) -> bool:
    _require_tuple(codes, "prefix codes")
    if any(
        not isinstance(code, str)
        or not code
        or any(bit not in "01" for bit in code)
        for code in codes
    ):
        raise ValueError("prefix codes must be nonempty binary strings")
    return len(codes) == len(set(codes)) and not any(
        right.startswith(left)
        for left in codes
        for right in codes
        if left != right
    )


def elias_delta_bit_length(registry_index: int) -> int:
    """Return the exact frozen Elias-delta length for a one-based index."""

    _require_int(registry_index, "registry_index", minimum=1)
    floor_log2_n = registry_index.bit_length() - 1
    return (
        floor_log2_n
        + 2 * ((floor_log2_n + 1).bit_length() - 1)
        + 1
    )


def scope_extension_code_length_bits(clause_count: int) -> int:
    """Code the clause count and each (ContextId, expected Bool) pair."""

    _require_int(clause_count, "clause_count", maximum=2)
    return len(SCOPE_CLAUSE_COUNT_CODES[clause_count]) + clause_count * 3


def aggregate_leaf_code_length_bits(clause_count: int) -> int:
    """Length of the frozen aggregate leaf, including its scope extension."""

    return (
        len(AST_SHAPE_PREFIXES["leaf"])
        + len(LEAF_CLASS_CODES["aggregate"])
        + 3  # AggregateMapId
        + 2  # ScopeId
        + 1  # QuantityId
        + scope_extension_code_length_bits(clause_count)
    )


def new_reducer_fixed_code_length_bits(*, arity: int, clause_count: int) -> int:
    """Reducer definition length excluding the ordinary combiner-AST length."""

    _require_int(arity, "arity", minimum=1)
    return (
        16  # NEW_REDUCER_V1 header
        + elias_delta_bit_length(arity)
        + arity * 4  # input sort IDs
        + 4  # output sort ID
        + 1  # reduction scheme
        + 3  # identity RationalParameter
        + 4  # maximum supported set size
        + scope_extension_code_length_bits(clause_count)
        + 256  # verifier specification hash reference
    )


def ceil_log2_q32_integer(value: int) -> int:
    """Ceil(log2(value) * 2^32) without binary floating point."""

    _require_int(value, "log2 value", minimum=1)
    if value & (value - 1) == 0:
        return (value.bit_length() - 1) * Q32_SCALE
    with localcontext() as context:
        context.prec = MDL_DECIMAL_PRECISION
        scaled = (Decimal(value).ln() / Decimal(2).ln()) * Decimal(Q32_SCALE)
        return int(scaled.to_integral_value(rounding=ROUND_CEILING))


def binary_enumerative_data_code_length_q32(row_count: int, error_count: int) -> int:
    """Frozen binary enumerative error length in unsigned Q32 units."""

    _require_int(row_count, "row_count")
    _require_int(error_count, "error_count", maximum=row_count)
    exact_log_argument = (row_count + 1) * comb(row_count, error_count)
    return ceil_log2_q32_integer(exact_log_argument)


def mdl_required_gain_q32(old_data_code_length_q32: int) -> int:
    _require_int(old_data_code_length_q32, "old_data_code_length_q32")
    five_percent_ceiling = (old_data_code_length_q32 + 19) // 20
    return max(32 * Q32_SCALE, five_percent_ceiling)


@dataclass(frozen=True, slots=True)
class MdlCertificateBindings:
    mdl_code_table_root: str
    dsl_spec_root: str
    identifier_registry_root: str
    discovery_partition_root: str
    validation_partition_root: str
    sealed_partition_root: str
    target_truth_table_root: str
    old_program_ast_hash: str
    new_symbol_definition_hash: str
    new_call_program_ast_hash: str
    old_prediction_vector_root: str
    new_prediction_vector_root: str
    validation_prediction_root: str
    sealed_prediction_root: str
    fixed_point_precision_id: str
    mdl_algorithm_id: str
    repository_commit_sha: str
    container_image_digest: str
    schema_version: str = MDL_BINDINGS_SCHEMA

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "mdl_code_table_root",
            "dsl_spec_root",
            "identifier_registry_root",
            "discovery_partition_root",
            "validation_partition_root",
            "sealed_partition_root",
            "target_truth_table_root",
            "old_program_ast_hash",
            "new_symbol_definition_hash",
            "new_call_program_ast_hash",
            "old_prediction_vector_root",
            "new_prediction_vector_root",
            "validation_prediction_root",
            "sealed_prediction_root",
            "fixed_point_precision_id",
            "mdl_algorithm_id",
            "repository_commit_sha",
            "container_image_digest",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != MDL_BINDINGS_SCHEMA:
            raise ValueError("unsupported MDL-binding schema")
        for field_name in (
            "mdl_code_table_root",
            "dsl_spec_root",
            "identifier_registry_root",
            "discovery_partition_root",
            "validation_partition_root",
            "sealed_partition_root",
            "target_truth_table_root",
            "old_program_ast_hash",
            "new_symbol_definition_hash",
            "new_call_program_ast_hash",
            "old_prediction_vector_root",
            "new_prediction_vector_root",
            "validation_prediction_root",
            "sealed_prediction_root",
            "container_image_digest",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.fixed_point_precision_id != FIXED_POINT_PRECISION_ID:
            raise ValueError("MDL binding must use the frozen unsigned-Q32 precision")
        _require_nonempty_text(self.mdl_algorithm_id, "mdl_algorithm_id")
        _require_repository_commit(self.repository_commit_sha, "repository_commit_sha")

    @classmethod
    def from_mapping(cls, value: object) -> "MdlCertificateBindings":
        data = _strict_mapping(value, cls._KEYS, "MDL certificate bindings")
        return cls(**data)  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mdl_code_table_root": self.mdl_code_table_root,
            "dsl_spec_root": self.dsl_spec_root,
            "identifier_registry_root": self.identifier_registry_root,
            "discovery_partition_root": self.discovery_partition_root,
            "validation_partition_root": self.validation_partition_root,
            "sealed_partition_root": self.sealed_partition_root,
            "target_truth_table_root": self.target_truth_table_root,
            "old_program_ast_hash": self.old_program_ast_hash,
            "new_symbol_definition_hash": self.new_symbol_definition_hash,
            "new_call_program_ast_hash": self.new_call_program_ast_hash,
            "old_prediction_vector_root": self.old_prediction_vector_root,
            "new_prediction_vector_root": self.new_prediction_vector_root,
            "validation_prediction_root": self.validation_prediction_root,
            "sealed_prediction_root": self.sealed_prediction_root,
            "fixed_point_precision_id": self.fixed_point_precision_id,
            "mdl_algorithm_id": self.mdl_algorithm_id,
            "repository_commit_sha": self.repository_commit_sha,
            "container_image_digest": self.container_image_digest,
        }


MDL_IGNORED_CALLER_FIELDS: Final = frozenset(
    {
        "length",
        "Fraction",
        "fraction",
        "delta_L",
        "threshold_pass",
        "L_old_program",
        "L_train_given_old",
        "L_new_symbol_definition",
        "L_new_call_program",
        "L_train_given_new",
        "required_delta_L",
    }
)


def _freeze_bit_vector(
    value: object,
    name: str,
    *,
    allow_undefined: bool,
) -> tuple[int | None, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must not be empty")
    allowed = {0, 1, None} if allow_undefined else {0, 1}
    for item in result:
        if item not in allowed or isinstance(item, bool):
            label = "bits or None" if allow_undefined else "integer bits"
            raise ValueError(f"{name} must contain {label}")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MdlReplayRequest:
    bindings: MdlCertificateBindings
    code_table_id: str
    old_program_ast: FrozenCborMap
    new_symbol_definition: FrozenCborMap
    new_call_program_ast: FrozenCborMap
    discovery_target_labels: tuple[int | None, ...]
    old_discovery_predictions: tuple[int | None, ...]
    new_discovery_predictions: tuple[int | None, ...]
    ignored_caller_fields: tuple[str, ...] = ()
    schema_version: str = MDL_REQUEST_SCHEMA

    _REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "bindings",
            "code_table_id",
            "old_program_ast",
            "new_symbol_definition",
            "new_call_program_ast",
            "discovery_target_labels",
            "old_discovery_predictions",
            "new_discovery_predictions",
        }
    )

    def __post_init__(self) -> None:
        if self.schema_version != MDL_REQUEST_SCHEMA:
            raise ValueError("unsupported MDL replay-request schema")
        if not isinstance(self.bindings, MdlCertificateBindings):
            raise TypeError("bindings must be MdlCertificateBindings")
        if self.code_table_id != MDL_CODE_TABLE_ID:
            raise ValueError("MDL replay must use the frozen prefix-code table")
        for field_name in (
            "old_program_ast",
            "new_symbol_definition",
            "new_call_program_ast",
        ):
            if not isinstance(getattr(self, field_name), FrozenCborMap):
                raise TypeError(f"{field_name} must be FrozenCborMap")
        target = _freeze_bit_vector(
            self.discovery_target_labels,
            "discovery_target_labels",
            allow_undefined=False,
        )
        old = _freeze_bit_vector(
            self.old_discovery_predictions,
            "old_discovery_predictions",
            allow_undefined=True,
        )
        new = _freeze_bit_vector(
            self.new_discovery_predictions,
            "new_discovery_predictions",
            allow_undefined=True,
        )
        if len(target) != len(old) or len(target) != len(new):
            raise ValueError("MDL labels and prediction vectors must be aligned")
        if self.ignored_caller_fields != tuple(sorted(set(self.ignored_caller_fields))):
            raise ValueError("ignored_caller_fields must be sorted and unique")
        if any(item not in MDL_IGNORED_CALLER_FIELDS for item in self.ignored_caller_fields):
            raise ValueError("unknown ignored caller field")

    @classmethod
    def from_mapping(cls, value: object) -> "MdlReplayRequest":
        if not isinstance(value, Mapping):
            raise TypeError("MDL replay request must be a mapping")
        ignored = tuple(sorted(set(value) & MDL_IGNORED_CALLER_FIELDS))
        filtered = {key: item for key, item in value.items() if key not in ignored}
        data = _strict_mapping(filtered, cls._REQUIRED_KEYS, "MDL replay request")
        return cls(
            schema_version=data["schema_version"],  # type: ignore[arg-type]
            bindings=MdlCertificateBindings.from_mapping(data["bindings"]),
            code_table_id=data["code_table_id"],  # type: ignore[arg-type]
            old_program_ast=FrozenCborMap.from_mapping(data["old_program_ast"]),
            new_symbol_definition=FrozenCborMap.from_mapping(
                data["new_symbol_definition"]
            ),
            new_call_program_ast=FrozenCborMap.from_mapping(
                data["new_call_program_ast"]
            ),
            discovery_target_labels=_freeze_bit_vector(
                data["discovery_target_labels"],
                "discovery_target_labels",
                allow_undefined=False,
            ),
            old_discovery_predictions=_freeze_bit_vector(
                data["old_discovery_predictions"],
                "old_discovery_predictions",
                allow_undefined=True,
            ),
            new_discovery_predictions=_freeze_bit_vector(
                data["new_discovery_predictions"],
                "new_discovery_predictions",
                allow_undefined=True,
            ),
            ignored_caller_fields=ignored,
        )


class MdlScorerStatus(str, Enum):
    HARD_DISABLED = "HARD_DISABLED"


@dataclass(frozen=True, slots=True)
class FormalMdlAssessment:
    status: MdlScorerStatus
    formal_gate_pass: bool
    row_count: int
    old_error_count: int
    new_error_count: int
    train_given_old_q32: int
    train_given_new_q32: int
    required_delta_l_q32: int
    diagnostic_data_delta_q32: int
    old_program_length_q32: None
    new_symbol_definition_length_q32: None
    new_call_program_length_q32: None
    delta_l_q32: None
    ignored_caller_fields: tuple[str, ...]
    blockers: tuple[str, ...]


def formal_mdl_capability_failures() -> tuple[str, ...]:
    checks = (
        (CANONICAL_CBOR_ENCODER_IMPLEMENTED, "canonical_cbor_unavailable"),
        (CANONICAL_AST_SCHEMA_IMPLEMENTED, "canonical_ast_schema_unimplemented"),
        (FORMAL_MDL_AST_SCORER_IMPLEMENTED, "formal_mdl_ast_scorer_unimplemented"),
        (PYTHON_MDL_REPLAY_IMPLEMENTED, "python_mdl_replay_unimplemented"),
        (RUST_MDL_REPLAY_IMPLEMENTED, "rust_mdl_replay_unimplemented"),
    )
    failures = [reason for implemented, reason in checks if not implemented]
    failures.extend(
        (
            "mdl_ast_and_new_symbol_wire_schema_unfrozen",
            "cross_language_q32_log2_reference_algorithm_unfrozen",
        )
    )
    return tuple(dict.fromkeys(failures))


def score_mdl_formally(request: MdlReplayRequest) -> FormalMdlAssessment:
    """Recompute safe diagnostics, but never claim a formal MDL gate yet."""

    if not isinstance(request, MdlReplayRequest):
        raise TypeError("request must be MdlReplayRequest")
    target = request.discovery_target_labels
    old_errors = sum(
        prediction is None or prediction != label
        for prediction, label in zip(
            request.old_discovery_predictions,
            target,
            strict=True,
        )
    )
    new_errors = sum(
        prediction is None or prediction != label
        for prediction, label in zip(
            request.new_discovery_predictions,
            target,
            strict=True,
        )
    )
    old_data = binary_enumerative_data_code_length_q32(len(target), old_errors)
    new_data = binary_enumerative_data_code_length_q32(len(target), new_errors)
    return FormalMdlAssessment(
        status=MdlScorerStatus.HARD_DISABLED,
        formal_gate_pass=False,
        row_count=len(target),
        old_error_count=old_errors,
        new_error_count=new_errors,
        train_given_old_q32=old_data,
        train_given_new_q32=new_data,
        required_delta_l_q32=mdl_required_gain_q32(old_data),
        diagnostic_data_delta_q32=old_data - new_data,
        old_program_length_q32=None,
        new_symbol_definition_length_q32=None,
        new_call_program_length_q32=None,
        delta_l_q32=None,
        ignored_caller_fields=request.ignored_caller_fields,
        blockers=formal_mdl_capability_failures(),
    )


__all__ = [
    "AST_SHAPE_PREFIXES",
    "BINARY_TOKEN_CODES",
    "BucketCount",
    "CANONICAL_CBOR_ENCODER_IMPLEMENTED",
    "CapabilityUnavailable",
    "ChunkManifest",
    "DetachedSignature",
    "DslSpecStatus",
    "Ed25519PublicKeyRecord",
    "ExhaustionReceipt",
    "FORMAL_CERTIFICATE_ROLES",
    "FORMAL_OUTSIDE_CERTIFICATE_ISSUANCE_IMPLEMENTED",
    "FIXED_POINT_PRECISION_ID",
    "FormalMdlAssessment",
    "FrozenCborMap",
    "KeyEpochManifest",
    "KeyRevocationManifest",
    "KeyRole",
    "LEAF_CLASS_CODES",
    "MAX_CANONICAL_PROGRAM_COUNT",
    "MAX_RAW_OPERATOR_APPLICATIONS",
    "MDL_CODE_TABLE_ID",
    "MDL_IGNORED_CALLER_FIELDS",
    "MdlCertificateBindings",
    "MdlReplayRequest",
    "MdlScorerStatus",
    "OutsideCertificateAssessment",
    "OutsideCertificateBody",
    "OutsideCertificateEnvelope",
    "OutsideFrozenClosureClaim",
    "ProgramRecord",
    "Q32_SCALE",
    "RATIONAL_PARAMETER_CODES",
    "RECORDS_PER_CHUNK",
    "ReplayAgreement",
    "ReplayEnvironmentBinding",
    "ReplayLanguage",
    "ReplayStatus",
    "ReplaySummary",
    "SPECIFICATION_RESOLUTION_BLOCKERS",
    "SignatureVerification",
    "TOLERANCE_CODES",
    "TERNARY_TOKEN_CODES",
    "TargetTruthRow",
    "UNARY_TOKEN_CODES",
    "UniverseRow",
    "aggregate_leaf_code_length_bits",
    "binary_enumerative_data_code_length_q32",
    "bounded_universe_root",
    "canonical_cbor_bytes",
    "canonical_cbor_sha256",
    "ceil_log2_q32_integer",
    "chunk_manifest_root",
    "elias_delta_bit_length",
    "formal_mdl_capability_failures",
    "mdl_required_gain_q32",
    "new_reducer_fixed_code_length_bits",
    "outside_certificate_capability_failures",
    "prefix_code_is_prefix_free",
    "rfc6962_leaf_hash",
    "rfc6962_merkle_root",
    "rfc6962_merkle_root_id",
    "rfc6962_node_hash",
    "scope_extension_code_length_bits",
    "score_mdl_formally",
    "sha256_id",
    "target_truth_table_root",
    "validate_chunk_manifests",
    "validate_universe_and_target_rows",
    "verify_ed25519_digest",
    "verify_outside_certificate",
    "verify_rotation_or_revocation_signatures",
]

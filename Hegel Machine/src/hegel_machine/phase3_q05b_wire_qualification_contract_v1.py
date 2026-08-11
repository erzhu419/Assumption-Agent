"""Qualification-only exact wire for Phase-3A Q0.5b.

The objects in this module are deliberately outside the formal Q1 tag range.
They bind a bounded node-three diagnostic, its complete
810-leaf seed manifest, deterministic sidecars, and a two-stage *non-Q1*
qualification receipt.  They cannot create a Q1 fixed-point record, populate
one of the eight formal output slots, increment the Q1 gate count/mask, or
transition Q1 away from ``NOT_RUN``.

The wire module does not itself execute the implemented supervisor, actor, and
host-replay path.  At Commit A no actual attempt has executed, so evidence for
qualification predicates 1 through 20 remains pending.  Predicate 14 has a
frozen source-level dual-encoder capability, but it is not an actual
qualification pass until isolated actor/host evidence is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
import json
from typing import Final, NoReturn, Sequence

from . import phase3_q0_quotient_contract_v1 as _q0
from . import phase3_q1_archive_projection_v1 as _projection
from . import phase3_q1_capacity_preflight_v1 as _capacity
from . import phase3_q1_external_sort_profile_v1 as _external_sort
from . import phase3_q1_formal_archive_contract_v1 as _formal
from . import phase3_q1_semantic_coverage_v1 as _coverage
from .phase3_q1_partition_snapshot_v1 import Q1PartitionSnapshotV1
from .phase3_q1_universe_v1 import production_universe_v1
from .strict_ast_shrink6_v1 import decode_shrink6_canonical_ast
from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


QUALIFICATION_WIRE_VERSION: Final = "hegel-q05b-wire-qualification-v1.0.0"
QUALIFICATION_SCOPE_ID: Final = "BOUNDED_NODE3_SOURCE_AND_WIRE_QUALIFICATION"
QUALIFICATION_ENGINEERING_STATUS: Final = (
    "ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED"
)

# This registry is qualification-only.  None of these values may be added to
# the separately frozen formal Q1 tag registry.
Q05B_FULL_LEAF_MANIFEST_ROW_TAG: Final = 0x3A00
Q05B_FULL_LEAF_MANIFEST_TAG: Final = 0x3A01
Q05B_NODE3_PARTITION_EVIDENCE_TAG: Final = 0x3A02
Q05B_SIDECAR_MANIFEST_TAG: Final = 0x3A03
Q05B_NODE3_GOLDEN_MANIFEST_TAG: Final = 0x3A04
Q05B_QUALIFICATION_CANDIDATE_RECEIPT_TAG: Final = 0x3A05
Q05B_QUALIFICATION_RECEIPT_TAG: Final = 0x3A06
Q05B_BOUNDED_NODE3_STATE_TAG: Final = 0x3A07

Q05B_QUALIFICATION_TAG_REGISTRY: Final = (
    (Q05B_FULL_LEAF_MANIFEST_ROW_TAG, b"Q05B_FULL_LEAF_MANIFEST_ROW"),
    (Q05B_FULL_LEAF_MANIFEST_TAG, b"Q05B_FULL_LEAF_MANIFEST"),
    (Q05B_NODE3_PARTITION_EVIDENCE_TAG, b"Q05B_NODE3_PARTITION_EVIDENCE"),
    (Q05B_SIDECAR_MANIFEST_TAG, b"Q05B_SIDECAR_MANIFEST"),
    (Q05B_NODE3_GOLDEN_MANIFEST_TAG, b"Q05B_NODE3_GOLDEN_MANIFEST"),
    (
        Q05B_QUALIFICATION_CANDIDATE_RECEIPT_TAG,
        b"Q05B_QUALIFICATION_CANDIDATE_RECEIPT",
    ),
    (Q05B_QUALIFICATION_RECEIPT_TAG, b"Q05B_QUALIFICATION_RECEIPT"),
    (Q05B_BOUNDED_NODE3_STATE_TAG, b"Q05B_BOUNDED_NODE3_STATE"),
)

FULL_LEAF_MANIFEST_ROW_SCHEMA_ID: Final = b"hegel-q05b-full-leaf-row/1"
FULL_LEAF_MANIFEST_SCHEMA_ID: Final = b"hegel-q05b-full-leaf-manifest/1"
NODE3_PARTITION_EVIDENCE_SCHEMA_ID: Final = (
    b"hegel-q05b-node3-partition-evidence/1"
)
BOUNDED_NODE3_STATE_SCHEMA_ID: Final = b"hegel-q05b-bounded-node3-state/1"
SIDECAR_MANIFEST_SCHEMA_ID: Final = b"hegel-q05b-sidecar-manifest/1"
NODE3_GOLDEN_MANIFEST_SCHEMA_ID: Final = b"hegel-q05b-node3-golden-manifest/1"
QUALIFICATION_CANDIDATE_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q05b-qualification-candidate-receipt/1"
)
QUALIFICATION_RECEIPT_SCHEMA_ID: Final = b"hegel-q05b-qualification-receipt/1"

FULL_LEAF_MANIFEST_SIDECAR_CONTENT_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/FULL_V16_LEAF_MANIFEST_SIDECAR/V1"
)
NODE3_PARTITION_EVIDENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/NODE3/PARTITION_EVIDENCE/V1"
)
SIDECAR_MANIFEST_ROOT_DOMAIN: Final = "HEGEL/Q05B/NODE3/SIDECAR_MANIFEST/V1"
NODE3_GOLDEN_MANIFEST_ROOT_DOMAIN: Final = "HEGEL/Q05B/NODE3/GOLDEN_MANIFEST/V1"
QUALIFICATION_CANDIDATE_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/CANDIDATE_RECEIPT/V1"
)
QUALIFICATION_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/FINAL_RECEIPT/V1"
)
BOUNDED_NODE3_STATE_ROOT_DOMAIN: Final = "HEGEL/Q05B/NODE3/BOUNDED_STATE/V1"
QUALIFICATION_PREDICATE_REGISTRY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/PREDICATE_REGISTRY/V1"
)
QUALIFICATION_PRE_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/PRE_RECEIPT/V1"
)
PRE_RECEIPT_EVIDENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/PRE_RECEIPT_EVIDENCE/V1"
)
PREDICATE20_EVIDENCE_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/PREDICATE20_EVIDENCE/V1"
)
QUALIFICATION_TAG_REGISTRY_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/TAG_REGISTRY/V1"
)
QUALIFICATION_WIRE_PROFILE_ROOT_DOMAIN: Final = (
    "HEGEL/Q05B/QUALIFICATION/WIRE_PROFILE/V1"
)
SIDECAR_CONTENT_ROOT_DOMAINS: Final = (
    FULL_LEAF_MANIFEST_SIDECAR_CONTENT_ROOT_DOMAIN,
    NODE3_PARTITION_EVIDENCE_ROOT_DOMAIN,
    NODE3_PARTITION_EVIDENCE_ROOT_DOMAIN,
)

FULL_LEAF_MANIFEST_RELATIVE_PATH: Final = (
    b"preimages/000-full-v16-leaf-manifest-v1.cbor"
)
ODD_PARTITION_EVIDENCE_RELATIVE_PATH: Final = (
    b"preimages/001-odd-node3-partition-evidence-v1.cbor"
)
SINK_PARTITION_EVIDENCE_RELATIVE_PATH: Final = (
    b"preimages/002-sink-node3-partition-evidence-v1.cbor"
)
SIDECAR_MANIFEST_RELATIVE_PATH: Final = (
    b"neutral/q05b-node3-sidecar-manifest-v1.cbor"
)
NODE3_GOLDEN_MANIFEST_RELATIVE_PATH: Final = (
    b"neutral/q05b-node3-golden-manifest-v1.cbor"
)
ORDERED_PREIMAGE_RELATIVE_PATHS: Final = (
    FULL_LEAF_MANIFEST_RELATIVE_PATH,
    ODD_PARTITION_EVIDENCE_RELATIVE_PATH,
    SINK_PARTITION_EVIDENCE_RELATIVE_PATH,
)
ORDERED_OUTPUT_RELATIVE_PATHS: Final = ORDERED_PREIMAGE_RELATIVE_PATHS + (
    SIDECAR_MANIFEST_RELATIVE_PATH,
    NODE3_GOLDEN_MANIFEST_RELATIVE_PATH,
)
OUTPUT_FILE_MODE: Final = 0o444
ORDERED_OUTPUT_PATH_MODE_ROWS: Final = tuple(
    (path, OUTPUT_FILE_MODE) for path in ORDERED_OUTPUT_RELATIVE_PATHS
)

FULL_V16_LEAF_COUNT: Final = 810
NODE3_MAXIMUM_AST_DEPTH: Final = 3
NODE3_MAXIMUM_AST_NODE_COUNT: Final = 3
NODE3_STRUCTURAL_BOUNDARY_DEPTH: Final = 4
Q1_NOT_RUN_STATE_ID: Final = 0
Q2_NOT_RUN_STATE_ID: Final = 0
Q1_GATE_COUNT: Final = 0
Q1_GATE_MASK: Final = 0
Q1_GATE_TOTAL: Final = 20
Q1_FORMAL_FIXED_POINT_TAG_OR_NULL: Final = None
MAX_CHUNK_FRAMED_BYTES: Final = 16_777_216
MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES: Final = 16_777_207

Q0_SEMANTIC_BINDING_ROOT: Final = _q0.q0_semantic_binding_root_v1()
Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION: Final = bytes.fromhex(
    "ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2"
)
SEMANTIC_SOURCE_ROOTS: Final = (
    _q0.Q0_CHILD_DSL_SPEC_ROOT,
    _q0.Q0_OPERATOR_SEMANTICS_ROOT,
    _q0.Q0_IDENTIFIER_REGISTRY_ROOT,
    _q0.Q0_CANONICAL_AST_SCHEMA_ROOT,
    _q0.Q0_CANONICAL_CBOR_PROFILE_ROOT,
    Q0_SEMANTIC_BINDING_ROOT,
    Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION,
)
Q1_PREREGISTRATION_DOCUMENT_SHA256: Final = bytes.fromhex(
    "2fbbba865abf0589c0d48ead9a170fae0b81f1cc1d440ddbc9c5d93909615f42"
)
POST_SHRINK6_NORMATIVE_DOCUMENT_SHA256: Final = bytes.fromhex(
    "1df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b95"
)
Q1_MDL_PROFILE_ID: Final = b"hegel-mdl-prefix-v1.0.0"
VERSION_BINDING_ROWS: Final = (
    (1, b"dsl_version", _formal.DSL_VERSION.encode("ascii")),
    (2, b"dsl_freeze_version", _formal.DSL_FREEZE_VERSION.encode("ascii")),
    (
        3,
        b"closure_semantics_version",
        _formal.CLOSURE_SEMANTICS_VERSION.encode("ascii"),
    ),
    (4, b"archive_wire_version", _formal.ARCHIVE_WIRE_VERSION.encode("ascii")),
    (
        5,
        b"projection_freeze_version",
        _formal.PROJECTION_FREEZE_VERSION.encode("ascii"),
    ),
    (6, b"qualification_wire_version", QUALIFICATION_WIRE_VERSION.encode("ascii")),
)

Q1_NULL_OUTPUT_SLOTS: Final = tuple(
    (index, name, None)
    for index, name in enumerate(_formal.Q1_OUTPUT_SLOT_NAMES, start=1)
)

QUALIFICATION_PREDICATE_REGISTRY: Final = (
    (1, b"QUALIFICATION_TAG_NAMESPACE_SEPARATE_FROM_FORMAL_Q1"),
    (2, b"STRICT_CANONICAL_CBOR_DUAL_REPLAY"),
    (3, b"FULL_810_LEAF_MANIFEST_EXACT_REPLAY"),
    (4, b"Q0_AND_Q1_PREREG_SEMANTIC_SOURCE_ROOTS_BOUND"),
    (5, b"BOUNDED_NODE3_SCOPE_NO_FORMAL_FIXED_POINT_ALIAS"),
    (6, b"NEUTRAL_GOLDEN_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL"),
    (7, b"SIDECAR_MANIFEST_PYTHON_RUST_HOST_BYTE_EQUAL"),
    (8, b"SIDECAR_RAW_SHA_LENGTH_CONTENT_ROOT_REPLAY"),
    (9, b"PYTHON_ACTOR_SOURCE_RUNTIME_IDENTITY_QUALIFIED"),
    (10, b"RUST_ACTOR_SOURCE_RUNTIME_IDENTITY_QUALIFIED"),
    (11, b"TRUSTED_HOST_READ_ONLY_REPLAY_QUALIFIED"),
    (12, b"STRICT_PARTITION_MANIFEST_BUNDLE_ASSEMBLER_REPLAY"),
    (13, b"CHUNK_FRAMING_BOUNDARY_AND_TAMPER_VECTORS_PASS"),
    (14, b"COUNTING_DISCARD_AND_MATERIALIZED_ENCODER_EQUAL"),
    (15, b"EXTERNAL_SORT_RUN_AND_MERGE_REPLAY_PASS"),
    (16, b"THREE_ACTOR_SCRATCH_LEDGER_REPLAY_PASS"),
    (17, b"OUTPUT_AND_METADATA_FORMULA_REPLAY_PASS"),
    (18, b"COLLISION_DUPLICATE_AND_TAMPER_FAIL_CLOSED"),
    (19, b"OFFLINE_SOURCE_RUNTIME_AND_FILESYSTEM_ISOLATION_PASS"),
    (20, b"CANDIDATE_RECEIPT_VALIDATED_WHILE_Q1_REMAINS_NOT_RUN"),
)
QUALIFICATION_PREDICATE_REGISTRY_ROOT: Final = content_hash(
    QUALIFICATION_PREDICATE_REGISTRY_ROOT_DOMAIN,
    QUALIFICATION_PREDICATE_REGISTRY,
)
QUALIFICATION_TAG_REGISTRY_ROOT: Final = content_hash(
    QUALIFICATION_TAG_REGISTRY_ROOT_DOMAIN,
    Q05B_QUALIFICATION_TAG_REGISTRY,
)
IMPLEMENTATION_BLOCKED_PREDICATE_IDS: Final = ()
PENDING_ACTUAL_EVIDENCE_PREDICATE_IDS: Final = tuple(range(1, 21))
COMMIT_A_ACTUAL_PRECONDITIONS_V1: Final = {
    "actual_entrypoint_implemented": True,
    "source_freeze_execution_status": "NOT_EXECUTED_AT_COMMIT_A",
    "execution_admission_policy": "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION",
    "implementation_blocked_predicate_ids": list(
        IMPLEMENTATION_BLOCKED_PREDICATE_IDS
    ),
    "pending_actual_evidence_predicate_ids": list(
        PENDING_ACTUAL_EVIDENCE_PREDICATE_IDS
    ),
    "clean_full40_commit_required": True,
    "head_must_equal_requested_commit": True,
    "commit_a_config_blob_must_equal_runtime_config": True,
    "pinned_local_images_required": True,
    "sealed_source_cargo_runtime_required": True,
    "attempt_unique_docker_execution_authority_required": True,
    "initial_and_precreate_name_absence_required": True,
    "docker_cleanup_owned_cid_only_required": True,
    "foreign_or_unknown_docker_state_zero_mutation_required": True,
    "artifact_target_must_be_absent": True,
    "all_runtime_preconditions_required": True,
    "all_20_predicates_required_before_artifact": True,
    "toctou_revalidation_required": True,
    "atomic_noreplace_artifact_required": True,
}
PREDICATE14_SOURCE_CAPABILITY_FROZEN: Final = True

COUNTING_DISCARD_SCHEMA_FIELDS: Final = (
    b"version",
    b"schema_id",
    b"input_signature_id",
    b"universe_root",
    b"stream_kind_id",
    b"record_count",
    b"canonical_record_payload_bytes",
    b"framed_stream_bytes",
    b"chunk_count",
    b"descriptor_object",
    b"chunk_manifest_objects",
    b"external_sort_projection_object",
    b"diagnostic_commitment",
    b"retained_framed_blob_count",
    b"retained_framed_blob_bytes",
)
COUNTING_DISCARD_EQUALITY_RULES: Final = (
    b"MATERIALIZED_FRAMED_BLOBS_STRICT_REPLAY_AND_REENCODE_EXACT",
    b"RECORD_COUNT_EQUALS_MATERIALIZED_DESCRIPTOR_SLOT_3",
    b"FRAMED_STREAM_BYTES_EQUALS_MATERIALIZED_DESCRIPTOR_SLOT_5",
    b"CHUNK_COUNT_EQUALS_MATERIALIZED_DESCRIPTOR_SLOT_6",
    b"DESCRIPTOR_OBJECT_EQUALS_MATERIALIZED_PROJECTED_SLOT_5",
    b"CHUNK_MANIFEST_OBJECTS_EQUAL_MATERIALIZED_PROJECTED_SLOT_6",
    b"EXTERNAL_SORT_PROJECTION_EQUALS_MATERIALIZED_PROJECTED_SLOT_7",
    b"DIAGNOSTIC_COMMITMENT_EQUALS_MATERIALIZED_PROJECTED_SLOT_8",
    b"COUNTING_SINK_REENCODES_ORDERED_FORMAL_RECORDS_INDEPENDENTLY",
    b"RETAINED_FRAMED_BLOB_COUNT_AND_BYTES_EQUAL_ZERO_ZERO",
)

FROZEN_NODE3_PRIMARY_COUNTS: Final = (
    # input_signature_id, rows, raw, strict, rewrites, classes, cohorts, bank, frontier
    (1, 480, 1048, 1048, 22, 40, 86, 110, 59),
    (2, 85, 1101, 1101, 26, 28, 112, 144, 84),
)


class SidecarContentKindId(IntEnum):
    FULL_V16_LEAF_MANIFEST = 1
    ODD_NODE3_PARTITION_EVIDENCE = 2
    SINK_NODE3_PARTITION_EVIDENCE = 3


class Q05BWireQualificationError(ValueError):
    """Stable fail-closed error from the Q0.5b qualification wire."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q05BWireQualificationError(code, detail)


def _uint(value: object, name: str, maximum: int = (1 << 64) - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail("REJECT_Q05B_UINT", f"{name} is outside uint range")
    return value


def _bytes(value: object, name: str, length: int | None = None) -> bytes:
    if type(value) is not bytes or (length is not None and len(value) != length):
        suffix = "bytes" if length is None else f"exactly {length} bytes"
        _fail("REJECT_Q05B_BYTES", f"{name} must be {suffix}")
    return value


def _root32(value: object, name: str) -> bytes:
    return _bytes(value, name, 32)


def _tuple(value: object, name: str, length: int | None = None) -> tuple[object, ...]:
    if type(value) is not tuple or (length is not None and len(value) != length):
        suffix = "tuple" if length is None else f"{length}-item tuple"
        _fail("REJECT_Q05B_ARRAY", f"{name} must be exact {suffix}")
    return value


def _strict_cbor_object(payload: bytes, name: str) -> tuple[object, ...]:
    _bytes(payload, name)
    try:
        value = canonical_cbor_decode(payload)
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q05B_CBOR", f"{name}: {error}")
    if type(value) is not tuple or canonical_cbor_encode(value) != payload:
        _fail("REJECT_Q05B_CBOR", f"{name} must be one canonical CBOR array")
    return value


def _q1_authority_object_v1() -> tuple[object, ...]:
    return (
        Q1_NOT_RUN_STATE_ID,
        Q1_GATE_COUNT,
        Q1_GATE_MASK,
        Q1_GATE_TOTAL,
        len(Q1_NULL_OUTPUT_SLOTS),
        Q1_NULL_OUTPUT_SLOTS,
        None,  # q1 receipt
        Q2_NOT_RUN_STATE_ID,
        None,  # M3 formal roots
        False,  # formal fixed point claimed
        Q1_FORMAL_FIXED_POINT_TAG_OR_NULL,
        False,  # target truth accessed
        False,  # split accessed
        False,  # role evaluation performed
        False,  # outside certificate issued
        False,  # ACTIVE transition allowed
    )


def validate_q1_authority_closed_v1(value: object) -> None:
    if canonical_cbor_encode(value) != canonical_cbor_encode(
        _q1_authority_object_v1()
    ):
        _fail(
            "REJECT_Q05B_Q1_AUTHORITY",
            (
                "Q1 must remain NOT_RUN/0/0 of 20 with eight null roots; "
                "Q2/M3/certificate/ACTIVE must remain closed"
            ),
        )


@dataclass(frozen=True, slots=True)
class Q05BFullLeafManifestRowV1:
    leaf_index: int
    output_sort_id: int
    root_operator_id: int
    canonical_ast_cbor: bytes
    canonical_ast_hash: bytes

    def __post_init__(self) -> None:
        _uint(self.leaf_index, "leaf_index", FULL_V16_LEAF_COUNT - 1)
        _uint(self.output_sort_id, "output_sort_id", 5)
        if self.output_sort_id == 0:
            _fail("REJECT_Q05B_LEAF_ROW", "output sort ID zero is unregistered")
        _uint(self.root_operator_id, "root_operator_id", 0xFFFF)
        _bytes(self.canonical_ast_cbor, "canonical_ast_cbor")
        _root32(self.canonical_ast_hash, "canonical_ast_hash")
        try:
            ast = decode_shrink6_canonical_ast(self.canonical_ast_cbor)
        except (TypeError, ValueError) as error:
            _fail("REJECT_Q05B_LEAF_AST", str(error))
        expected_sort = _capacity.OUTPUT_SORT_IDS[ast.metrics.output_sort]
        if (
            ast.metrics.depth != 0
            or ast.metrics.node_count != 1
            or ast.root_operator_id != self.root_operator_id
            or expected_sort != self.output_sort_id
            or ast.digest != self.canonical_ast_hash
        ):
            _fail("REJECT_Q05B_LEAF_AST", "leaf AST metrics or digest differ")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q05B_FULL_LEAF_MANIFEST_ROW_TAG,
            FULL_LEAF_MANIFEST_ROW_SCHEMA_ID,
            self.leaf_index,
            self.output_sort_id,
            self.root_operator_id,
            self.canonical_ast_cbor,
            self.canonical_ast_hash,
        )


@dataclass(frozen=True, slots=True)
class Q05BFullLeafManifestV1:
    rows: tuple[Q05BFullLeafManifestRowV1, ...]

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or any(
            type(row) is not Q05BFullLeafManifestRowV1 for row in self.rows
        ):
            _fail("REJECT_Q05B_LEAF_MANIFEST", "rows have wrong exact type")
        if len(self.rows) != FULL_V16_LEAF_COUNT:
            _fail("REJECT_Q05B_LEAF_MANIFEST", "leaf count is not exactly 810")
        if tuple(row.leaf_index for row in self.rows) != tuple(
            range(FULL_V16_LEAF_COUNT)
        ):
            _fail("REJECT_Q05B_LEAF_ORDER", "leaf indices are not contiguous")
        order = tuple(
            (row.output_sort_id, row.root_operator_id, row.canonical_ast_cbor)
            for row in self.rows
        )
        if order != tuple(sorted(order)) or len(
            {row.canonical_ast_cbor for row in self.rows}
        ) != FULL_V16_LEAF_COUNT:
            _fail("REJECT_Q05B_LEAF_ORDER", "leaf order or uniqueness differs")

    def canonical_object(self) -> tuple[object, ...]:
        row_objects = tuple(row.canonical_object() for row in self.rows)
        return (
            1,
            Q05B_FULL_LEAF_MANIFEST_TAG,
            FULL_LEAF_MANIFEST_SCHEMA_ID,
            _q0.DSL_VERSION.encode("ascii"),
            _q0.DSL_FREEZE_VERSION.encode("ascii"),
            FULL_V16_LEAF_COUNT,
            rfc6962_root(row_objects),
            row_objects,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def manifest_root(self) -> bytes:
        # The formal Q1 semantic binding consumes this exact root.  It binds only
        # the 810 ordered leaf identities, not Q0 qualification provenance.
        return rfc6962_root(tuple(row.canonical_object() for row in self.rows))


def full_v16_leaf_manifest_v1() -> Q05BFullLeafManifestV1:
    """Replay the authoritative Q1 preregistered 810-leaf source manifest."""

    asts = _capacity._frozen_leaf_asts_v1(raw_cap=FULL_V16_LEAF_COUNT)  # noqa: SLF001
    rows = tuple(
        Q05BFullLeafManifestRowV1(
            leaf_index=index,
            output_sort_id=_capacity.OUTPUT_SORT_IDS[ast.metrics.output_sort],
            root_operator_id=ast.root_operator_id,
            canonical_ast_cbor=ast.cbor_bytes,
            canonical_ast_hash=ast.digest,
        )
        for index, ast in enumerate(asts)
    )
    return Q05BFullLeafManifestV1(rows)


def decode_full_v16_leaf_manifest_v1(payload: bytes) -> Q05BFullLeafManifestV1:
    value = _strict_cbor_object(payload, "full leaf manifest")
    if (
        len(value) != 8
        or value[:3]
        != (1, Q05B_FULL_LEAF_MANIFEST_TAG, FULL_LEAF_MANIFEST_SCHEMA_ID)
        or value[3:5]
        != (
            _q0.DSL_VERSION.encode("ascii"),
            _q0.DSL_FREEZE_VERSION.encode("ascii"),
        )
        or value[5] != FULL_V16_LEAF_COUNT
        or type(value[7]) is not tuple
        or value[6] != rfc6962_root(value[7])
    ):
        _fail("REJECT_Q05B_LEAF_MANIFEST", "manifest header differs")
    rows: list[Q05BFullLeafManifestRowV1] = []
    for item in value[7]:
        row = _tuple(item, "leaf row", 8)
        if row[:3] != (
            1,
            Q05B_FULL_LEAF_MANIFEST_ROW_TAG,
            FULL_LEAF_MANIFEST_ROW_SCHEMA_ID,
        ):
            _fail("REJECT_Q05B_LEAF_ROW", "leaf row header differs")
        rows.append(Q05BFullLeafManifestRowV1(*row[3:]))
    manifest = Q05BFullLeafManifestV1(tuple(rows))
    expected = full_v16_leaf_manifest_v1()
    if manifest.canonical_bytes != expected.canonical_bytes:
        _fail(
            "REJECT_Q05B_LEAF_MANIFEST",
            "leaf rows differ from independent authoritative regeneration",
        )
    return manifest


def q1_semantic_binding_manifest_v1(
    full_leaf_manifest: Q05BFullLeafManifestV1,
) -> _formal.Q1SemanticBindingManifestV1:
    """Build the exact formal Q1 semantic *input binding*, not an output root.

    This is source qualification material only.  The resulting semantic root
    and projection-profile root are inputs to a future admitted Q1 run; they
    do not populate any of the eight formal output slots.
    """

    if type(full_leaf_manifest) is not Q05BFullLeafManifestV1:
        raise TypeError("full_leaf_manifest must be Q05BFullLeafManifestV1")
    return _q1_semantic_binding_manifest_from_leaf_root_v1(
        full_leaf_manifest.manifest_root
    )


def _q1_semantic_binding_manifest_from_leaf_root_v1(
    full_leaf_manifest_root: bytes,
) -> _formal.Q1SemanticBindingManifestV1:
    _root32(full_leaf_manifest_root, "full_leaf_manifest_root")
    return _formal.Q1SemanticBindingManifestV1(
        child_dsl_root=_q0.Q0_CHILD_DSL_SPEC_ROOT,
        operator_semantics_root=_q0.Q0_OPERATOR_SEMANTICS_ROOT,
        identifier_registry_root=_q0.Q0_IDENTIFIER_REGISTRY_ROOT,
        canonical_ast_root=_q0.Q0_CANONICAL_AST_SCHEMA_ROOT,
        canonical_cbor_root=_q0.Q0_CANONICAL_CBOR_PROFILE_ROOT,
        mdl_profile_id=Q1_MDL_PROFILE_ID,
        q0_receipt_root=Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION,
        full_v16_leaf_manifest_root=full_leaf_manifest_root,
        preregistration_document_sha256=Q1_PREREGISTRATION_DOCUMENT_SHA256,
        post_shrink6_document_sha256=POST_SHRINK6_NORMATIVE_DOCUMENT_SHA256,
    )


def q1_semantic_and_projection_roots_v1(
    full_leaf_manifest: Q05BFullLeafManifestV1,
) -> tuple[bytes, bytes]:
    semantic_manifest = q1_semantic_binding_manifest_v1(full_leaf_manifest)
    semantic_root = semantic_manifest.manifest_root
    projection_root = _formal.projection_profile_root_v1(
        semantic_binding_root=semantic_root,
        coverage_registry_root=rfc6962_root(
            _formal.expected_coverage_registry_v1()
        ),
        resource_guard_registry=_formal.Q1_RESOURCE_GUARD_REGISTRY,
    )
    return semantic_root, projection_root


@dataclass(frozen=True, slots=True)
class Q05BNode3PartitionEvidenceV1:
    input_signature_id: int
    universe_root: bytes
    record_set_object: tuple[object, ...]
    coverage_rows: tuple[tuple[object, ...], ...]
    stream_rows: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (
            1,
            2,
        ):
            _fail("REJECT_Q05B_PARTITION", "input signature must be exact 1 or 2")
        expected_root = production_universe_v1(self.input_signature_id).universe_root
        if type(self.universe_root) is not bytes or self.universe_root != expected_root:
            _fail("REJECT_Q05B_PARTITION", "universe root differs")
        record_set = _tuple(self.record_set_object, "record_set_object", 7)
        if (
            record_set[:4]
            != (
                1,
                _projection.SNAPSHOT_RECORD_SET_SCHEMA_ID,
                self.input_signature_id,
                self.universe_root,
            )
            or any(type(row) is not tuple for row in record_set[4:])
        ):
            _fail("REJECT_Q05B_PARTITION", "record-set object differs")
        if type(self.coverage_rows) is not tuple or len(self.coverage_rows) != 846:
            _fail("REJECT_Q05B_PARTITION", "coverage row count is not 846")
        expected_registry = _formal.expected_coverage_registry_v1()
        for index, item in enumerate(self.coverage_rows):
            row = _tuple(item, "coverage evidence row", 4)
            record = _tuple(row[0], "coverage record", 15)
            for name, preimages in zip(
                ("eligible", "processed", "strict"), row[1:], strict=True
            ):
                _tuple(preimages, f"{name} preimages")
            if (
                record[:5]
                != (
                    1,
                    _formal.Q1_SEMANTIC_COVERAGE_TAG,
                    _formal.SEMANTIC_COVERAGE_SCHEMA_ID,
                    self.input_signature_id,
                    self.universe_root,
                )
                or (record[5], record[6]) != expected_registry[index]
                or record[7] != len(row[1])
                or record[9] != len(row[2])
                or record[11] != len(row[3])
                or record[8] != rfc6962_root(row[1])
                or record[10] != rfc6962_root(row[2])
                or record[12] != rfc6962_root(row[3])
            ):
                _fail("REJECT_Q05B_PARTITION", "coverage preimage replay differs")
        if type(self.stream_rows) is not tuple or len(self.stream_rows) != 4:
            _fail("REJECT_Q05B_PARTITION", "exactly four stream rows required")
        for expected_kind, item in enumerate(self.stream_rows, start=1):
            row = _tuple(item, "stream evidence row", 5)
            projected = _tuple(row[1], "projected stream object", 9)
            descriptor = _tuple(projected[5], "materialized descriptor", 8)
            manifests = _tuple(projected[6], "materialized chunk manifests")
            external_projection = _tuple(
                projected[7], "materialized external-sort projection", 15
            )
            framed_blobs = _tuple(row[2], "framed blobs")
            trace = _tuple(row[3], "external-sort trace", 6)
            trace_projection = _tuple(trace[2], "trace projection", 15)
            _tuple(trace[3], "trace ordered rows")
            _tuple(trace[4], "trace run manifests")
            _tuple(trace[5], "trace scratch events")
            counting = _tuple(row[4], "counting/discard stream", 15)
            counting_descriptor = _tuple(
                counting[9], "counting/discard descriptor", 8
            )
            counting_manifests = _tuple(
                counting[10], "counting/discard chunk manifests"
            )
            counting_external_projection = _tuple(
                counting[11], "counting/discard external-sort projection", 15
            )
            try:
                replayed_records = tuple(
                    record
                    for blob in framed_blobs
                    for record in _formal.replay_framed_records_v1(blob)
                )
            except _formal.Q1ArchiveContractError as error:
                _fail(
                    "REJECT_Q05B_PARTITION",
                    f"materialized framed-record replay failed: {error.code}",
                )
            replayed_payload_bytes = sum(
                len(canonical_cbor_encode(record)) for record in replayed_records
            )
            replayed_framed_bytes = sum(len(blob) for blob in framed_blobs)
            if (
                type(row[0]) is not int
                or row[0] != expected_kind
                or type(projected[0]) is not int
                or type(projected[2]) is not int
                or type(projected[4]) is not int
                or projected[:5]
                != (
                    1,
                    _projection.PROJECTED_STREAM_SCHEMA_ID,
                    self.input_signature_id,
                    self.universe_root,
                    expected_kind,
                )
                or any(type(blob) is not bytes for blob in framed_blobs)
                or any(
                    type(descriptor[index]) is not int
                    for index in (0, 2, 3, 5, 6)
                )
                or type(projected[8]) is not bytes
                or len(projected[8]) != 32
                or descriptor[3] != len(replayed_records)
                or descriptor[5] != replayed_framed_bytes
                or descriptor[6] != len(framed_blobs)
                or len(manifests) != len(framed_blobs)
                or type(trace[0]) is not int
                or trace[:2] != (1, _external_sort.EXTERNAL_SORT_TRACE_SCHEMA_ID)
                or trace_projection != external_projection
                or counting[:5]
                != (
                    1,
                    _projection.COUNTING_DISCARD_STREAM_SCHEMA_ID,
                    self.input_signature_id,
                    self.universe_root,
                    expected_kind,
                )
                or any(
                    type(counting[index]) is not int
                    for index in (0, 2, 4, 5, 6, 7, 8, 13, 14)
                )
                or counting[5] != descriptor[3]
                or counting[6] != replayed_payload_bytes
                or counting[7] != descriptor[5]
                or counting[7] != replayed_framed_bytes
                or counting[8] != descriptor[6]
                or counting_descriptor != descriptor
                or counting_manifests != manifests
                or counting_external_projection != external_projection
                or counting[12] != projected[8]
                or counting[13:] != (0, 0)
            ):
                _fail("REJECT_Q05B_PARTITION", "stream evidence binding differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q05B_NODE3_PARTITION_EVIDENCE_TAG,
            NODE3_PARTITION_EVIDENCE_SCHEMA_ID,
            self.input_signature_id,
            self.universe_root,
            self.record_set_object,
            846,
            self.coverage_rows,
            4,
            self.stream_rows,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def evidence_root(self) -> bytes:
        return content_hash(
            NODE3_PARTITION_EVIDENCE_ROOT_DOMAIN,
            self.canonical_object(),
        )


def node3_partition_evidence_v1(
    snapshot: Q1PartitionSnapshotV1,
    record_set: _projection.Q1SnapshotRecordSetV1,
    coverage_archive: _coverage.Q1SemanticCoverageArchiveV1,
) -> Q05BNode3PartitionEvidenceV1:
    """Materialize the exact neutral sidecar preimage for one node3 partition."""

    if type(snapshot) is not Q1PartitionSnapshotV1:
        raise TypeError("snapshot must be Q1PartitionSnapshotV1")
    if (
        snapshot.limits.maximum_ast_depth != NODE3_MAXIMUM_AST_DEPTH
        or snapshot.limits.maximum_ast_node_count != NODE3_MAXIMUM_AST_NODE_COUNT
        or snapshot.q1_state != "NOT_RUN"
        or snapshot.q1_gate_count != 0
        or snapshot.q1_gate_mask != 0
        or snapshot.q1_formal_roots is not None
    ):
        _fail("REJECT_Q05B_NODE3_SCOPE", "snapshot is not bounded node3/NOT_RUN")
    if (
        record_set.input_signature_id != snapshot.input_signature_id
        or coverage_archive.input_signature_id != snapshot.input_signature_id
        or record_set.universe_root != snapshot.universe_root
        or coverage_archive.universe_root != snapshot.universe_root
    ):
        _fail("REJECT_Q05B_PARTITION", "snapshot consumer bindings differ")
    coverage_rows = tuple(
        (
            record.canonical_object(),
            preimage.eligible_application_keys,
            preimage.processed_application_keys,
            preimage.strict_admission_preimages,
        )
        for record, preimage in zip(
            coverage_archive.coverage_records,
            coverage_archive.coverage_preimages,
            strict=True,
        )
    )
    stream_records = (
        record_set.program_records,
        record_set.cohort_records,
        record_set.class_records,
        coverage_archive.coverage_records,
    )
    stream_rows: list[tuple[object, ...]] = []
    for kind, records in zip(_formal.ArchiveStreamKindId, stream_records, strict=True):
        projected = _projection.project_record_stream_v1(
            records,
            input_signature_id=snapshot.input_signature_id,
            universe_root=snapshot.universe_root,
            stream_kind_id=kind,
        )
        trace = _external_sort.project_external_sort_trace_v1(
            tuple(
                (
                    _projection._stream_sort_key(record, kind),  # noqa: SLF001
                    canonical_cbor_encode(record.canonical_object()),
                )
                for record in records
            ),
            input_signature_id=snapshot.input_signature_id,
            stream_kind_id=kind,
        )
        if trace.projection != projected.external_sort_projection:
            _fail("REJECT_Q05B_PARTITION", "stream/trace sort projection differs")
        counting = _projection.counting_discard_record_stream_v1(
            records,
            input_signature_id=snapshot.input_signature_id,
            universe_root=snapshot.universe_root,
            stream_kind_id=kind,
        )
        _projection.validate_counting_discard_matches_materialized_v1(
            counting,
            projected,
        )
        stream_rows.append(
            (
                int(kind),
                projected.canonical_diagnostic_object(),
                projected.chunks.framed_blobs,
                trace.canonical_object(),
                counting.canonical_object(),
            )
        )
    return Q05BNode3PartitionEvidenceV1(
        snapshot.input_signature_id,
        snapshot.universe_root,
        record_set.canonical_diagnostic_object(),
        coverage_rows,
        tuple(stream_rows),
    )


def decode_node3_partition_evidence_v1(
    payload: bytes,
) -> Q05BNode3PartitionEvidenceV1:
    value = _strict_cbor_object(payload, "node3 partition evidence")
    if (
        len(value) != 10
        or value[:3]
        != (
            1,
            Q05B_NODE3_PARTITION_EVIDENCE_TAG,
            NODE3_PARTITION_EVIDENCE_SCHEMA_ID,
        )
        or value[6] != 846
        or value[8] != 4
    ):
        _fail("REJECT_Q05B_PARTITION", "partition evidence header differs")
    return Q05BNode3PartitionEvidenceV1(
        input_signature_id=value[3],
        universe_root=value[4],
        record_set_object=value[5],
        coverage_rows=value[7],
        stream_rows=value[9],
    )


@dataclass(frozen=True, slots=True)
class Q05BBoundedNode3StateV1:
    input_signature_id: int
    universe_root: bytes
    universe_row_count: int
    terminal_status: bytes
    primary_counts: tuple[int, ...]
    maximum_bank_points_per_class: int
    maximum_frontier_points_per_class: int
    peak_work_queue_points: int
    peak_saturation_round_count: int
    coverage_record_root: bytes
    partition_evidence_root: bytes

    def __post_init__(self) -> None:
        if type(self.input_signature_id) is not int or self.input_signature_id not in (
            1,
            2,
        ):
            _fail("REJECT_Q05B_BOUNDED_STATE", "input signature differs")
        if self.universe_root != production_universe_v1(
            self.input_signature_id
        ).universe_root:
            _fail("REJECT_Q05B_BOUNDED_STATE", "universe root differs")
        expected = FROZEN_NODE3_PRIMARY_COUNTS[self.input_signature_id - 1]
        if (
            self.universe_row_count != expected[1]
            or type(self.primary_counts) is not tuple
            or self.primary_counts != expected[2:]
            or self.terminal_status
            != _capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED.encode("ascii")
        ):
            _fail("REJECT_Q05B_BOUNDED_STATE", "frozen node3 golden differs")
        for name in (
            "maximum_bank_points_per_class",
            "maximum_frontier_points_per_class",
            "peak_work_queue_points",
            "peak_saturation_round_count",
        ):
            _uint(getattr(self, name), name)
        raw, strict, rewrites, classes, cohorts, bank, frontier = self.primary_counts
        if not (
            raw == strict
            and rewrites <= strict
            and 1 <= classes <= cohorts <= bank
            and classes <= frontier <= bank
            and 1
            <= self.maximum_frontier_points_per_class
            <= self.maximum_bank_points_per_class
            <= bank
            and self.peak_work_queue_points >= bank
            and self.peak_saturation_round_count == 5
        ):
            _fail("REJECT_Q05B_BOUNDED_STATE", "node3 count relation differs")
        _root32(self.coverage_record_root, "coverage_record_root")
        _root32(self.partition_evidence_root, "partition_evidence_root")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q05B_BOUNDED_NODE3_STATE_TAG,
            BOUNDED_NODE3_STATE_SCHEMA_ID,
            QUALIFICATION_SCOPE_ID.encode("ascii"),
            self.input_signature_id,
            self.universe_root,
            self.universe_row_count,
            NODE3_MAXIMUM_AST_DEPTH,
            NODE3_MAXIMUM_AST_NODE_COUNT,
            NODE3_STRUCTURAL_BOUNDARY_DEPTH,
            self.terminal_status,
            True,  # work queue empty at bounded terminal
            True,  # depth-three delta is zero
            True,  # structural-boundary delta is zero
            True,  # all 846 bounded coverage rows replayed
            True,  # eligible and processed sets/counts are equal
            self.primary_counts,
            self.maximum_bank_points_per_class,
            self.maximum_frontier_points_per_class,
            self.peak_work_queue_points,
            self.peak_saturation_round_count,
            self.coverage_record_root,
            self.partition_evidence_root,
            False,  # never a formal Q1 fixed-point record
            None,
            _q1_authority_object_v1(),
        )

    @property
    def state_root(self) -> bytes:
        return content_hash(BOUNDED_NODE3_STATE_ROOT_DOMAIN, self.canonical_object())


def bounded_node3_state_from_object_v1(
    value: object,
) -> Q05BBoundedNode3StateV1:
    state = _tuple(value, "bounded node3 state", 26)
    if (
        state[:4]
        != (
            1,
            Q05B_BOUNDED_NODE3_STATE_TAG,
            BOUNDED_NODE3_STATE_SCHEMA_ID,
            QUALIFICATION_SCOPE_ID.encode("ascii"),
        )
        or type(state[0]) is not int
        or type(state[1]) is not int
        or type(state[2]) is not bytes
        or type(state[3]) is not bytes
        or state[7:10]
        != (
            NODE3_MAXIMUM_AST_DEPTH,
            NODE3_MAXIMUM_AST_NODE_COUNT,
            NODE3_STRUCTURAL_BOUNDARY_DEPTH,
        )
        or any(type(item) is not int for item in state[7:10])
        or any(item is not True for item in state[11:16])
        or state[23] is not False
        or state[24] is not None
    ):
        _fail("REJECT_Q05B_BOUNDED_STATE", "bounded state wire differs")
    validate_q1_authority_closed_v1(state[25])
    replay = Q05BBoundedNode3StateV1(
        input_signature_id=state[4],
        universe_root=state[5],
        universe_row_count=state[6],
        terminal_status=state[10],
        primary_counts=state[16],
        maximum_bank_points_per_class=state[17],
        maximum_frontier_points_per_class=state[18],
        peak_work_queue_points=state[19],
        peak_saturation_round_count=state[20],
        coverage_record_root=state[21],
        partition_evidence_root=state[22],
    )
    if canonical_cbor_encode(replay.canonical_object()) != canonical_cbor_encode(state):
        _fail("REJECT_Q05B_BOUNDED_STATE", "bounded state replay differs")
    return replay


def bounded_node3_state_v1(
    snapshot: Q1PartitionSnapshotV1,
    evidence: Q05BNode3PartitionEvidenceV1,
) -> Q05BBoundedNode3StateV1:
    if (
        type(snapshot) is not Q1PartitionSnapshotV1
        or type(evidence) is not Q05BNode3PartitionEvidenceV1
        or snapshot.input_signature_id != evidence.input_signature_id
        or snapshot.universe_root != evidence.universe_root
        or snapshot.limits.maximum_ast_depth != NODE3_MAXIMUM_AST_DEPTH
        or snapshot.limits.maximum_ast_node_count != NODE3_MAXIMUM_AST_NODE_COUNT
        or snapshot.terminal_status
        != _capacity.LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
    ):
        _fail("REJECT_Q05B_BOUNDED_STATE", "snapshot/evidence scope differs")
    barriers = snapshot.depth_barriers
    if tuple(row.depth for row in barriers) != (0, 1, 2, 3, 4) or tuple(
        row.barrier_kind for row in barriers
    ) != (
        "LEAF_SEED",
        "CONSTRUCTION_DEPTH",
        "CONSTRUCTION_DEPTH",
        "CONSTRUCTION_DEPTH",
        "STRUCTURAL_BOUNDARY",
    ):
        _fail("REJECT_Q05B_BOUNDED_STATE", "depth barrier registry differs")
    for barrier in barriers[-2:]:
        if any(
            getattr(barrier, name) != 0
            for name in (
                "eligible_raw_application_count",
                "strict_admitted_application_count",
                "rewrite_collapse_count",
                "new_behavior_class_count",
                "new_signature_cohort_count",
                "continuation_bank_mutation_count",
            )
        ):
            _fail("REJECT_Q05B_BOUNDED_STATE", "terminal delta is not zero")
    coverage_objects = tuple(row[0] for row in evidence.coverage_rows)
    if len(coverage_objects) != 846 or any(row[1] != row[2] for row in evidence.coverage_rows):
        _fail("REJECT_Q05B_BOUNDED_STATE", "bounded coverage is incomplete")
    return Q05BBoundedNode3StateV1(
        snapshot.input_signature_id,
        snapshot.universe_root,
        snapshot.universe_row_count,
        snapshot.terminal_status.encode("ascii"),
        (
            snapshot.raw_operator_application_count,
            snapshot.strict_admitted_application_count,
            snapshot.rewrite_collapse_count,
            snapshot.behavior_class_count,
            snapshot.signature_cohort_count,
            snapshot.continuation_bank_point_count,
            snapshot.visible_frontier_point_count,
        ),
        snapshot.maximum_bank_points_per_class,
        snapshot.maximum_frontier_points_per_class,
        snapshot.peak_work_queue_points,
        snapshot.peak_saturation_round_count,
        rfc6962_root(coverage_objects),
        evidence.evidence_root,
    )


def _sidecar_file_row_v1(
    *,
    file_index: int,
    relative_path: bytes,
    content_kind_id: SidecarContentKindId,
    payload: bytes,
    expected_domain: str,
) -> tuple[object, ...]:
    _uint(file_index, "file_index", 2)
    if relative_path != ORDERED_PREIMAGE_RELATIVE_PATHS[file_index]:
        _fail("REJECT_Q05B_SIDECAR_PATH", "sidecar relative path/order differs")
    if int(content_kind_id) != file_index + 1:
        _fail("REJECT_Q05B_SIDECAR_KIND", "content kind/order differs")
    value = _strict_cbor_object(payload, "sidecar payload")
    root = content_hash(expected_domain, value)
    return (
        file_index,
        relative_path,
        int(content_kind_id),
        OUTPUT_FILE_MODE,
        len(payload),
        sha256(payload).digest(),
        expected_domain.encode("ascii"),
        root,
    )


@dataclass(frozen=True, slots=True)
class Q05BSidecarManifestV1:
    file_rows: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        if type(self.file_rows) is not tuple or len(self.file_rows) != 3:
            _fail("REJECT_Q05B_SIDECAR", "exactly three preimage files required")
        for index, item in enumerate(self.file_rows):
            row = _tuple(item, "sidecar file row", 8)
            if (
                type(row[0]) is not int
                or row[0] != index
                or type(row[1]) is not bytes
                or row[1] != ORDERED_PREIMAGE_RELATIVE_PATHS[index]
                or type(row[2]) is not int
                or row[2] != index + 1
                or type(row[3]) is not int
                or row[3] != OUTPUT_FILE_MODE
                or type(row[4]) is not int
                or row[4] < 1
                or type(row[5]) is not bytes
                or len(row[5]) != 32
                or row[6] != SIDECAR_CONTENT_ROOT_DOMAINS[index].encode("ascii")
                or type(row[7]) is not bytes
                or len(row[7]) != 32
            ):
                _fail("REJECT_Q05B_SIDECAR", "sidecar file row differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q05B_SIDECAR_MANIFEST_TAG,
            SIDECAR_MANIFEST_SCHEMA_ID,
            len(self.file_rows),
            self.file_rows,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def manifest_root(self) -> bytes:
        return content_hash(SIDECAR_MANIFEST_ROOT_DOMAIN, self.canonical_object())


def sidecar_manifest_v1(
    full_leaf_manifest: Q05BFullLeafManifestV1,
    odd_evidence: Q05BNode3PartitionEvidenceV1,
    sink_evidence: Q05BNode3PartitionEvidenceV1,
) -> Q05BSidecarManifestV1:
    if (
        type(full_leaf_manifest) is not Q05BFullLeafManifestV1
        or type(odd_evidence) is not Q05BNode3PartitionEvidenceV1
        or type(sink_evidence) is not Q05BNode3PartitionEvidenceV1
        or odd_evidence.input_signature_id != 1
        or sink_evidence.input_signature_id != 2
    ):
        _fail("REJECT_Q05B_SIDECAR", "sidecar source object types/order differ")
    payloads = (
        full_leaf_manifest.canonical_bytes,
        odd_evidence.canonical_bytes,
        sink_evidence.canonical_bytes,
    )
    rows = tuple(
        _sidecar_file_row_v1(
            file_index=index,
            relative_path=ORDERED_PREIMAGE_RELATIVE_PATHS[index],
            content_kind_id=SidecarContentKindId(index + 1),
            payload=payload,
            expected_domain=SIDECAR_CONTENT_ROOT_DOMAINS[index],
        )
        for index, payload in enumerate(payloads)
    )
    return Q05BSidecarManifestV1(rows)


def replay_sidecar_manifest_v1(
    manifest_payload: bytes,
    preimage_payloads: Sequence[bytes],
) -> Q05BSidecarManifestV1:
    value = _strict_cbor_object(manifest_payload, "sidecar manifest")
    if (
        len(value) != 5
        or value[:3]
        != (1, Q05B_SIDECAR_MANIFEST_TAG, SIDECAR_MANIFEST_SCHEMA_ID)
        or value[3] != 3
        or type(value[4]) is not tuple
        or len(preimage_payloads) != 3
        or any(type(payload) is not bytes for payload in preimage_payloads)
    ):
        _fail("REJECT_Q05B_SIDECAR", "sidecar manifest wire differs")
    manifest = Q05BSidecarManifestV1(value[4])
    for index, (row, payload) in enumerate(
        zip(manifest.file_rows, preimage_payloads, strict=True)
    ):
        decoded = _strict_cbor_object(payload, f"sidecar[{index}]")
        if (
            row[4] != len(payload)
            or row[5] != sha256(payload).digest()
            or row[7] != content_hash(SIDECAR_CONTENT_ROOT_DOMAINS[index], decoded)
        ):
            _fail("REJECT_Q05B_SIDECAR_REPLAY", "sidecar bytes/root differ")
    decode_full_v16_leaf_manifest_v1(preimage_payloads[0])
    odd = decode_node3_partition_evidence_v1(preimage_payloads[1])
    sink = decode_node3_partition_evidence_v1(preimage_payloads[2])
    if (odd.input_signature_id, sink.input_signature_id) != (1, 2):
        _fail("REJECT_Q05B_SIDECAR_REPLAY", "partition order differs")
    return manifest


def _partition_summary_row_v1(
    snapshot: Q1PartitionSnapshotV1,
    evidence: Q05BNode3PartitionEvidenceV1,
) -> tuple[object, ...]:
    if snapshot.input_signature_id != evidence.input_signature_id:
        _fail("REJECT_Q05B_GOLDEN", "snapshot/evidence signature differs")
    stream_summaries: list[tuple[object, ...]] = []
    for item in evidence.stream_rows:
        projected = item[1]
        descriptor = projected[5]
        external_projection = projected[7]
        stream_summaries.append(
            (
                item[0],
                descriptor[3],
                descriptor[4],
                descriptor[5],
                descriptor[6],
                descriptor[7],
                projected[8],
                content_hash(
                    _external_sort.EXTERNAL_SORT_PROJECTION_ROOT_DOMAIN,
                    external_projection,
                ),
                external_projection[10],
            )
        )
    coverage_objects = tuple(row[0] for row in evidence.coverage_rows)
    payload = evidence.canonical_bytes
    return (
        snapshot.input_signature_id,
        snapshot.universe_root,
        snapshot.universe_row_count,
        NODE3_MAXIMUM_AST_DEPTH,
        NODE3_MAXIMUM_AST_NODE_COUNT,
        NODE3_STRUCTURAL_BOUNDARY_DEPTH,
        snapshot.terminal_status.encode("ascii"),
        snapshot.raw_operator_application_count,
        snapshot.strict_admitted_application_count,
        snapshot.rewrite_collapse_count,
        snapshot.behavior_class_count,
        snapshot.signature_cohort_count,
        snapshot.continuation_bank_point_count,
        snapshot.visible_frontier_point_count,
        snapshot.maximum_bank_points_per_class,
        snapshot.maximum_frontier_points_per_class,
        snapshot.peak_work_queue_points,
        content_hash(
            _projection.SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
            evidence.record_set_object,
        ),
        len(coverage_objects),
        rfc6962_root(coverage_objects),
        len(stream_summaries),
        tuple(stream_summaries),
        len(payload),
        sha256(payload).digest(),
        evidence.evidence_root,
    )


@dataclass(frozen=True, slots=True)
class Q05BNode3GoldenManifestV1:
    q1_semantic_binding_root: bytes
    q1_projection_profile_root: bytes
    full_leaf_manifest_root: bytes
    sidecar_manifest_root: bytes
    bounded_state_rows: tuple[tuple[object, ...], ...]
    partition_summaries: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        _root32(self.q1_semantic_binding_root, "q1_semantic_binding_root")
        _root32(self.q1_projection_profile_root, "q1_projection_profile_root")
        _root32(self.full_leaf_manifest_root, "full_leaf_manifest_root")
        _root32(self.sidecar_manifest_root, "sidecar_manifest_root")
        expected_semantic = _q1_semantic_binding_manifest_from_leaf_root_v1(
            self.full_leaf_manifest_root
        ).manifest_root
        expected_projection = _formal.projection_profile_root_v1(
            semantic_binding_root=expected_semantic,
            coverage_registry_root=rfc6962_root(
                _formal.expected_coverage_registry_v1()
            ),
            resource_guard_registry=_formal.Q1_RESOURCE_GUARD_REGISTRY,
        )
        if (
            self.q1_semantic_binding_root != expected_semantic
            or self.q1_projection_profile_root != expected_projection
        ):
            _fail(
                "REJECT_Q05B_GOLDEN",
                "Q1 semantic-binding/projection roots do not replay from leaf/static inputs",
            )
        if type(self.bounded_state_rows) is not tuple or len(
            self.bounded_state_rows
        ) != 2:
            _fail("REJECT_Q05B_GOLDEN", "two bounded state rows required")
        for expected_id, item in zip((1, 2), self.bounded_state_rows, strict=True):
            row = _tuple(item, "bounded state row", 3)
            state = _tuple(row[1], "bounded state object", 26)
            replayed_state = bounded_node3_state_from_object_v1(state)
            if (
                type(row[0]) is not int
                or row[0] != expected_id
                or state[:5]
                != (
                    1,
                    Q05B_BOUNDED_NODE3_STATE_TAG,
                    BOUNDED_NODE3_STATE_SCHEMA_ID,
                    QUALIFICATION_SCOPE_ID.encode("ascii"),
                    expected_id,
                )
                or row[2]
                != content_hash(BOUNDED_NODE3_STATE_ROOT_DOMAIN, state)
                or replayed_state.state_root != row[2]
            ):
                _fail("REJECT_Q05B_GOLDEN", "bounded state row differs")
        if type(self.partition_summaries) is not tuple or len(
            self.partition_summaries
        ) != 2:
            _fail("REJECT_Q05B_GOLDEN", "two partition summaries required")
        for expected_id, item in zip((1, 2), self.partition_summaries, strict=True):
            row = _tuple(item, "partition summary", 25)
            expected_counts = FROZEN_NODE3_PRIMARY_COUNTS[expected_id - 1]
            state = self.bounded_state_rows[expected_id - 1][1]
            integer_slots = (
                0,
                2,
                3,
                4,
                5,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                18,
                20,
                22,
            )
            if any(type(row[index]) is not int for index in integer_slots):
                _fail("REJECT_Q05B_GOLDEN", "summary uint slot aliases bool")
            if type(row[6]) is not bytes:
                _fail("REJECT_Q05B_GOLDEN", "summary status must be bytes")
            stream_summaries = _tuple(row[21], "stream summaries", 4)
            for expected_kind, stream in enumerate(stream_summaries, start=1):
                stream_row = _tuple(stream, "stream summary", 9)
                if (
                    type(stream_row[0]) is not int
                    or stream_row[0] != expected_kind
                    or any(type(stream_row[index]) is not int for index in (1, 3, 4, 8))
                    or any(
                        type(stream_row[index]) is not bytes
                        or len(stream_row[index]) != 32
                        for index in (2, 5, 6, 7)
                    )
                ):
                    _fail("REJECT_Q05B_GOLDEN", "stream summary wire differs")
            if (
                row[0] != expected_id
                or row[1] != production_universe_v1(expected_id).universe_root
                or row[2] != len(production_universe_v1(expected_id).rows)
                or (row[0], row[2], *row[7:14]) != expected_counts
                or row[3:6]
                != (
                    NODE3_MAXIMUM_AST_DEPTH,
                    NODE3_MAXIMUM_AST_NODE_COUNT,
                    NODE3_STRUCTURAL_BOUNDARY_DEPTH,
                )
                or row[18] != 846
                or row[20] != 4
                or type(row[21]) is not tuple
                or len(row[21]) != 4
                or any(type(root) is not bytes or len(root) != 32 for root in row[17:20:2])
                or type(row[22]) is not int
                or row[22] < 1
                or type(row[23]) is not bytes
                or len(row[23]) != 32
                or type(row[24]) is not bytes
                or len(row[24]) != 32
                or row[6] != state[10]
                or row[7:14] != state[16]
                or row[14:17] != state[17:20]
                or row[19] != state[21]
                or row[24] != state[22]
            ):
                _fail("REJECT_Q05B_GOLDEN", "partition summary differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q05B_NODE3_GOLDEN_MANIFEST_TAG,
            NODE3_GOLDEN_MANIFEST_SCHEMA_ID,
            QUALIFICATION_WIRE_VERSION.encode("ascii"),
            QUALIFICATION_SCOPE_ID.encode("ascii"),
            NODE3_MAXIMUM_AST_DEPTH,
            NODE3_MAXIMUM_AST_NODE_COUNT,
            NODE3_STRUCTURAL_BOUNDARY_DEPTH,
            VERSION_BINDING_ROWS,
            QUALIFICATION_TAG_REGISTRY_ROOT,
            qualification_wire_profile_root_v1(),
            SEMANTIC_SOURCE_ROOTS,
            self.q1_semantic_binding_root,
            self.q1_projection_profile_root,
            self.full_leaf_manifest_root,
            self.sidecar_manifest_root,
            len(self.bounded_state_rows),
            self.bounded_state_rows,
            len(self.partition_summaries),
            self.partition_summaries,
            _q1_authority_object_v1(),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def manifest_root(self) -> bytes:
        return content_hash(
            NODE3_GOLDEN_MANIFEST_ROOT_DOMAIN,
            self.canonical_object(),
        )


def node3_golden_manifest_v1(
    full_leaf_manifest: Q05BFullLeafManifestV1,
    odd_snapshot: Q1PartitionSnapshotV1,
    odd_evidence: Q05BNode3PartitionEvidenceV1,
    sink_snapshot: Q1PartitionSnapshotV1,
    sink_evidence: Q05BNode3PartitionEvidenceV1,
    sidecar_manifest: Q05BSidecarManifestV1,
) -> Q05BNode3GoldenManifestV1:
    if (
        odd_snapshot.input_signature_id != 1
        or sink_snapshot.input_signature_id != 2
        or odd_evidence.input_signature_id != 1
        or sink_evidence.input_signature_id != 2
    ):
        _fail("REJECT_Q05B_GOLDEN", "odd/sink order differs")
    expected_sidecar = sidecar_manifest_v1(
        full_leaf_manifest,
        odd_evidence,
        sink_evidence,
    )
    if sidecar_manifest.canonical_bytes != expected_sidecar.canonical_bytes:
        _fail("REJECT_Q05B_GOLDEN", "sidecar manifest replay differs")
    semantic_root, projection_root = q1_semantic_and_projection_roots_v1(
        full_leaf_manifest
    )
    odd_state = bounded_node3_state_v1(odd_snapshot, odd_evidence)
    sink_state = bounded_node3_state_v1(sink_snapshot, sink_evidence)
    return Q05BNode3GoldenManifestV1(
        semantic_root,
        projection_root,
        full_leaf_manifest.manifest_root,
        sidecar_manifest.manifest_root,
        (
            (1, odd_state.canonical_object(), odd_state.state_root),
            (2, sink_state.canonical_object(), sink_state.state_root),
        ),
        (
            _partition_summary_row_v1(odd_snapshot, odd_evidence),
            _partition_summary_row_v1(sink_snapshot, sink_evidence),
        ),
    )


def decode_node3_golden_manifest_v1(payload: bytes) -> Q05BNode3GoldenManifestV1:
    value = _strict_cbor_object(payload, "node3 golden manifest")
    if (
        len(value) != 21
        or value[:3]
        != (1, Q05B_NODE3_GOLDEN_MANIFEST_TAG, NODE3_GOLDEN_MANIFEST_SCHEMA_ID)
        or value[3:12]
        != (
            QUALIFICATION_WIRE_VERSION.encode("ascii"),
            QUALIFICATION_SCOPE_ID.encode("ascii"),
            NODE3_MAXIMUM_AST_DEPTH,
            NODE3_MAXIMUM_AST_NODE_COUNT,
            NODE3_STRUCTURAL_BOUNDARY_DEPTH,
            VERSION_BINDING_ROWS,
            QUALIFICATION_TAG_REGISTRY_ROOT,
            qualification_wire_profile_root_v1(),
            SEMANTIC_SOURCE_ROOTS,
        )
        or value[16] != 2
        or value[18] != 2
    ):
        _fail("REJECT_Q05B_GOLDEN", "golden manifest header differs")
    validate_q1_authority_closed_v1(value[20])
    manifest = Q05BNode3GoldenManifestV1(
        value[12],
        value[13],
        value[14],
        value[15],
        value[17],
        value[19],
    )
    if manifest.canonical_bytes != payload:
        _fail("REJECT_Q05B_GOLDEN", "golden manifest replay differs")
    return manifest


def pre_receipt_evidence_root_v1(
    source_commit: bytes,
    predicate_rows_1_through_19: tuple[tuple[object, ...], ...],
) -> bytes:
    _bytes(source_commit, "source_commit", 20)
    if type(predicate_rows_1_through_19) is not tuple or len(
        predicate_rows_1_through_19
    ) != 19:
        _fail("REJECT_Q05B_PREDICATE", "pre-receipt requires 19 rows")
    for expected, row in zip(
        QUALIFICATION_PREDICATE_REGISTRY[:19],
        predicate_rows_1_through_19,
        strict=True,
    ):
        _validate_predicate_row_v1(row, expected)
    return content_hash(
        PRE_RECEIPT_EVIDENCE_ROOT_DOMAIN,
        (
            (1, b"git-sha1-raw20", source_commit),
            19,
            0x7FFFF,
            predicate_rows_1_through_19,
        ),
    )


def _validate_predicate_row_v1(
    value: object,
    expected_registry_row: tuple[int, bytes],
) -> tuple[object, ...]:
    row = _tuple(value, "qualification predicate row", 4)
    if (
        type(row[0]) is not int
        or row[0] != expected_registry_row[0]
        or type(row[1]) is not bytes
        or row[1] != expected_registry_row[1]
        or type(row[2]) is not bool
        or row[2] is not True
        or type(row[3]) is not bytes
        or len(row[3]) != 32
    ):
        _fail(
            "REJECT_Q05B_PREDICATE",
            "predicate id/name/pass/evidence slots differ type-exactly",
        )
    return row


def predicate20_evidence_root_v1(candidate_receipt_root: bytes) -> bytes:
    _root32(candidate_receipt_root, "candidate_receipt_root")
    return content_hash(
        PREDICATE20_EVIDENCE_ROOT_DOMAIN,
        (
            candidate_receipt_root,
            True,
            _q1_authority_object_v1(),
        ),
    )


@dataclass(frozen=True, slots=True)
class Q05BQualificationCandidateReceiptV1:
    source_commit: bytes
    q1_semantic_binding_root: bytes
    q1_projection_profile_root: bytes
    q0_receipt_root: bytes
    full_leaf_manifest_root: bytes
    implementation_roots: tuple[bytes, ...]
    neutral_manifest_roots: tuple[bytes, ...]
    bounded_state_roots: tuple[bytes, ...]
    bundle_evidence_root: bytes
    isolation_evidence_root: bytes
    resource_evidence_root: bytes
    pre_receipt_evidence_root: bytes
    predicate_rows_1_through_19: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        _bytes(self.source_commit, "source_commit", 20)
        for name in (
            "q1_semantic_binding_root",
            "q1_projection_profile_root",
            "q0_receipt_root",
            "full_leaf_manifest_root",
            "bundle_evidence_root",
            "isolation_evidence_root",
            "resource_evidence_root",
            "pre_receipt_evidence_root",
        ):
            _root32(getattr(self, name), name)
        for name, roots, expected_count in (
            ("implementation_roots", self.implementation_roots, 3),
            ("neutral_manifest_roots", self.neutral_manifest_roots, 3),
            ("bounded_state_roots", self.bounded_state_roots, 2),
        ):
            if type(roots) is not tuple or len(roots) != expected_count:
                _fail("REJECT_Q05B_RECEIPT_BINDING", f"{name} length differs")
            for index, root in enumerate(roots):
                _root32(root, f"{name}[{index}]")
        if len(set(self.implementation_roots)) != 3:
            _fail(
                "REJECT_Q05B_RECEIPT_BINDING",
                "three actor-specific implementation roots must be distinct",
            )
        if len(set(self.neutral_manifest_roots)) != 1:
            _fail(
                "REJECT_Q05B_RECEIPT_BINDING",
                "Python/Rust/host neutral manifest roots must be exactly equal",
            )
        if self.q0_receipt_root != Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION:
            _fail("REJECT_Q05B_RECEIPT_BINDING", "Q0 receipt root differs")
        expected_semantic = _q1_semantic_binding_manifest_from_leaf_root_v1(
            self.full_leaf_manifest_root
        ).manifest_root
        expected_projection = _formal.projection_profile_root_v1(
            semantic_binding_root=expected_semantic,
            coverage_registry_root=rfc6962_root(
                _formal.expected_coverage_registry_v1()
            ),
            resource_guard_registry=_formal.Q1_RESOURCE_GUARD_REGISTRY,
        )
        if (
            self.q1_semantic_binding_root != expected_semantic
            or self.q1_projection_profile_root != expected_projection
        ):
            _fail(
                "REJECT_Q05B_RECEIPT_BINDING",
                "semantic/projection roots do not replay from full leaf root",
            )
        if type(self.predicate_rows_1_through_19) is not tuple or len(
            self.predicate_rows_1_through_19
        ) != 19:
            _fail("REJECT_Q05B_PREDICATE", "candidate requires exactly 19 rows")
        for expected, item in zip(
            QUALIFICATION_PREDICATE_REGISTRY[:19],
            self.predicate_rows_1_through_19,
            strict=True,
        ):
            _validate_predicate_row_v1(item, expected)
        expected_pre_receipt_evidence_root = pre_receipt_evidence_root_v1(
            self.source_commit,
            self.predicate_rows_1_through_19,
        )
        if self.pre_receipt_evidence_root != expected_pre_receipt_evidence_root:
            _fail(
                "REJECT_Q05B_RECEIPT_BINDING",
                "pre-receipt evidence root differs from source/predicate replay",
            )

    @property
    def source_commit_object(self) -> tuple[object, ...]:
        return (1, b"git-sha1-raw20", self.source_commit)

    @property
    def implementation_root_rows(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (actor_id, root)
            for actor_id, root in zip(
                (b"python", b"rust", b"host"),
                self.implementation_roots,
                strict=True,
            )
        )

    @property
    def neutral_manifest_root_rows(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (actor_id, root)
            for actor_id, root in zip(
                (b"python", b"rust", b"host"),
                self.neutral_manifest_roots,
                strict=True,
            )
        )

    @property
    def bounded_state_root_rows(self) -> tuple[tuple[object, ...], ...]:
        return ((1, self.bounded_state_roots[0]), (2, self.bounded_state_roots[1]))

    def _pre_receipt_object(self) -> tuple[object, ...]:
        return (
            self.source_commit_object,
            VERSION_BINDING_ROWS,
            QUALIFICATION_TAG_REGISTRY_ROOT,
            qualification_wire_profile_root_v1(),
            QUALIFICATION_PREDICATE_REGISTRY_ROOT,
            self.q1_semantic_binding_root,
            self.q1_projection_profile_root,
            self.q0_receipt_root,
            self.full_leaf_manifest_root,
            self.implementation_root_rows,
            self.neutral_manifest_root_rows,
            self.bounded_state_root_rows,
            self.bundle_evidence_root,
            self.isolation_evidence_root,
            self.resource_evidence_root,
            self.pre_receipt_evidence_root,
            _q1_authority_object_v1(),
        )

    @property
    def pre_receipt_root(self) -> bytes:
        return content_hash(
            QUALIFICATION_PRE_RECEIPT_ROOT_DOMAIN,
            self._pre_receipt_object(),
        )

    @property
    def predicate_mask(self) -> int:
        return 0x7FFFF

    @property
    def passed_predicate_count(self) -> int:
        return 19

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q05B_QUALIFICATION_CANDIDATE_RECEIPT_TAG,
            QUALIFICATION_CANDIDATE_RECEIPT_SCHEMA_ID,
            self.source_commit_object,
            VERSION_BINDING_ROWS,
            QUALIFICATION_TAG_REGISTRY_ROOT,
            qualification_wire_profile_root_v1(),
            QUALIFICATION_PREDICATE_REGISTRY_ROOT,
            self.q1_semantic_binding_root,
            self.q1_projection_profile_root,
            self.q0_receipt_root,
            self.full_leaf_manifest_root,
            self.implementation_root_rows,
            self.neutral_manifest_root_rows,
            self.bounded_state_root_rows,
            self.bundle_evidence_root,
            self.isolation_evidence_root,
            self.resource_evidence_root,
            self.pre_receipt_evidence_root,
            self.pre_receipt_root,
            19,
            self.predicate_rows_1_through_19,
            self.passed_predicate_count,
            self.predicate_mask,
            _q1_authority_object_v1(),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(
            QUALIFICATION_CANDIDATE_RECEIPT_ROOT_DOMAIN,
            self.canonical_object(),
        )


def decode_qualification_candidate_receipt_v1(
    payload: bytes,
) -> Q05BQualificationCandidateReceiptV1:
    value = _strict_cbor_object(payload, "qualification candidate receipt")
    if (
        len(value) != 25
        or value[:3]
        != (
            1,
            Q05B_QUALIFICATION_CANDIDATE_RECEIPT_TAG,
            QUALIFICATION_CANDIDATE_RECEIPT_SCHEMA_ID,
        )
        or value[4] != VERSION_BINDING_ROWS
        or value[5] != QUALIFICATION_TAG_REGISTRY_ROOT
        or value[6] != qualification_wire_profile_root_v1()
        or value[7] != QUALIFICATION_PREDICATE_REGISTRY_ROOT
        or value[20] != 19
        or value[22:24] != (19, 0x7FFFF)
    ):
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "candidate wire differs")
    validate_q1_authority_closed_v1(value[24])
    source_commit_object = _tuple(value[3], "source commit", 3)
    if source_commit_object[:2] != (1, b"git-sha1-raw20"):
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "source commit wire differs")
    implementation_rows = _tuple(value[12], "implementation roots", 3)
    neutral_rows = _tuple(value[13], "neutral roots", 3)
    bounded_rows = _tuple(value[14], "bounded state roots", 2)
    actor_ids = (b"python", b"rust", b"host")
    if tuple(_tuple(row, "implementation root row", 2)[0] for row in implementation_rows) != actor_ids:
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "implementation actor IDs differ")
    if tuple(_tuple(row, "neutral root row", 2)[0] for row in neutral_rows) != actor_ids:
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "neutral actor IDs differ")
    if tuple(_tuple(row, "bounded root row", 2)[0] for row in bounded_rows) != (1, 2):
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "bounded role IDs differ")
    receipt = Q05BQualificationCandidateReceiptV1(
        source_commit=source_commit_object[2],
        q1_semantic_binding_root=value[8],
        q1_projection_profile_root=value[9],
        q0_receipt_root=value[10],
        full_leaf_manifest_root=value[11],
        implementation_roots=tuple(_tuple(row, "implementation root row", 2)[1] for row in implementation_rows),
        neutral_manifest_roots=tuple(_tuple(row, "neutral root row", 2)[1] for row in neutral_rows),
        bounded_state_roots=tuple(_tuple(row, "bounded root row", 2)[1] for row in bounded_rows),
        bundle_evidence_root=value[15],
        isolation_evidence_root=value[16],
        resource_evidence_root=value[17],
        pre_receipt_evidence_root=value[18],
        predicate_rows_1_through_19=value[21],
    )
    if value[19] != receipt.pre_receipt_root:
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "pre-receipt root differs")
    if canonical_cbor_encode(receipt.canonical_object()) != payload:
        _fail("REJECT_Q05B_CANDIDATE_RECEIPT", "candidate replay differs")
    return receipt


@dataclass(frozen=True, slots=True)
class Q05BQualificationReceiptV1:
    candidate_receipt: Q05BQualificationCandidateReceiptV1

    def __post_init__(self) -> None:
        if type(self.candidate_receipt) is not Q05BQualificationCandidateReceiptV1:
            _fail("REJECT_Q05B_RECEIPT", "candidate has wrong exact type")
        validate_q1_authority_closed_v1(_q1_authority_object_v1())

    def canonical_object(self) -> tuple[object, ...]:
        predicate20 = (
            20,
            QUALIFICATION_PREDICATE_REGISTRY[19][1],
            True,
            predicate20_evidence_root_v1(self.candidate_receipt.receipt_root),
        )
        rows = self.candidate_receipt.predicate_rows_1_through_19 + (predicate20,)
        return (
            1,
            Q05B_QUALIFICATION_RECEIPT_TAG,
            QUALIFICATION_RECEIPT_SCHEMA_ID,
            self.candidate_receipt.receipt_root,
            self.candidate_receipt.canonical_object(),
            20,
            QUALIFICATION_PREDICATE_REGISTRY,
            rows,
            20,
            (1 << 20) - 1,
            True,
            _q1_authority_object_v1(),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(QUALIFICATION_RECEIPT_ROOT_DOMAIN, self.canonical_object())


def decode_qualification_receipt_v1(payload: bytes) -> Q05BQualificationReceiptV1:
    value = _strict_cbor_object(payload, "qualification receipt")
    if (
        len(value) != 12
        or value[:3]
        != (
            1,
            Q05B_QUALIFICATION_RECEIPT_TAG,
            QUALIFICATION_RECEIPT_SCHEMA_ID,
        )
        or value[5:7] != (20, QUALIFICATION_PREDICATE_REGISTRY)
        or value[8:11] != (20, (1 << 20) - 1, True)
    ):
        _fail("REJECT_Q05B_RECEIPT", "final receipt wire differs")
    validate_q1_authority_closed_v1(value[11])
    rows = _tuple(value[7], "final predicate rows", 20)
    for expected, row in zip(
        QUALIFICATION_PREDICATE_REGISTRY,
        rows,
        strict=True,
    ):
        _validate_predicate_row_v1(row, expected)
    candidate_payload = canonical_cbor_encode(value[4])
    candidate = decode_qualification_candidate_receipt_v1(candidate_payload)
    if value[3] != candidate.receipt_root:
        _fail("REJECT_Q05B_RECEIPT", "candidate receipt root differs")
    expected_rows = candidate.predicate_rows_1_through_19 + (
        (
            20,
            QUALIFICATION_PREDICATE_REGISTRY[19][1],
            True,
            predicate20_evidence_root_v1(candidate.receipt_root),
        ),
    )
    if value[7] != expected_rows:
        _fail("REJECT_Q05B_RECEIPT", "predicate 20 vector differs")
    receipt = Q05BQualificationReceiptV1(candidate)
    if receipt.canonical_bytes != payload:
        _fail("REJECT_Q05B_RECEIPT", "final receipt replay differs")
    return receipt


def cbor_bstr_encoded_length_v1(raw_payload_length: int) -> int:
    """Return exact strict-CBOR byte-string length without allocating payload."""

    _uint(raw_payload_length, "raw_payload_length")
    if raw_payload_length <= 23:
        header = 1
    elif raw_payload_length <= 0xFF:
        header = 2
    elif raw_payload_length <= 0xFFFF:
        header = 3
    elif raw_payload_length <= 0xFFFFFFFF:
        header = 5
    else:
        header = 9
    return header + raw_payload_length


def framed_bstr_record_length_v1(raw_payload_length: int) -> int:
    return 4 + cbor_bstr_encoded_length_v1(raw_payload_length)


def bstr_record_fits_frozen_chunk_v1(raw_payload_length: int) -> bool:
    return framed_bstr_record_length_v1(raw_payload_length) <= MAX_CHUNK_FRAMED_BYTES


ACTOR_STDOUT_REQUIRED_FIELDS: Final = (
    "action_id",
    "actor_id",
    "file_count",
    "implementation_id",
    "neutral_manifest_length",
    "neutral_manifest_raw_sha256",
    "neutral_manifest_relative_path",
    "neutral_manifest_root",
    "q1_formal_roots",
    "q1_gate_count",
    "q1_gate_mask",
    "q1_output_slots",
    "q1_state",
    "runtime_identity_sha256",
    "sidecar_manifest_length",
    "sidecar_manifest_raw_sha256",
    "sidecar_manifest_relative_path",
    "sidecar_manifest_root",
    "source_identity_sha256",
    "schema_version",
    "status",
)
ACTOR_ENVELOPE_SCHEMA_VERSION: Final = "hegel-q05b-actor-envelope/1"
ACTOR_ACTION_ID: Final = "bounded-node3-golden-v1"
ACTOR_CANDIDATE_STATUS: Final = (
    "BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED"
)
ACTOR_IMPLEMENTATION_ID_REGISTRY: Final = (
    (
        "PYTHON_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
    ),
    (
        "RUST_ENDPOINT",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1",
    ),
    (
        "TRUSTED_HOST_REPLAY",
        "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1",
    ),
)


def validate_actor_stdout_envelope_v1(payload: bytes) -> dict[str, object]:
    """Validate one actor-specific one-line JSON envelope plus final LF."""

    _bytes(payload, "actor stdout")
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        _fail("REJECT_Q05B_ACTOR_STDOUT", "stdout must be one JSON line plus LF")

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail("REJECT_Q05B_ACTOR_STDOUT", "duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: _fail(
                "REJECT_Q05B_ACTOR_STDOUT", f"nonfinite token {token}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail("REJECT_Q05B_ACTOR_STDOUT", str(error))
    if type(value) is not dict or tuple(sorted(value)) != tuple(
        sorted(ACTOR_STDOUT_REQUIRED_FIELDS)
    ):
        _fail("REJECT_Q05B_ACTOR_STDOUT", "stdout field set differs")
    canonical = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        _fail("REJECT_Q05B_ACTOR_STDOUT", "stdout JSON is not canonical")
    if (
        (value["actor_id"], value["implementation_id"])
        not in ACTOR_IMPLEMENTATION_ID_REGISTRY
        or value["schema_version"] != ACTOR_ENVELOPE_SCHEMA_VERSION
        or value["status"] != ACTOR_CANDIDATE_STATUS
        or value["action_id"] != ACTOR_ACTION_ID
        or type(value["file_count"]) is not int
        or value["file_count"] != len(ORDERED_OUTPUT_RELATIVE_PATHS)
        or type(value["q1_gate_count"]) is not int
        or type(value["q1_gate_mask"]) is not int
        or value["neutral_manifest_relative_path"]
        != NODE3_GOLDEN_MANIFEST_RELATIVE_PATH.decode("ascii")
        or value["sidecar_manifest_relative_path"]
        != SIDECAR_MANIFEST_RELATIVE_PATH.decode("ascii")
        or value["q1_state"] != "NOT_RUN"
        or value["q1_gate_count"] != 0
        or value["q1_gate_mask"] != 0
        or value["q1_formal_roots"] is not None
        or value["q1_output_slots"] != [None] * 8
    ):
        _fail("REJECT_Q05B_ACTOR_STDOUT", "stdout authority/path binding differs")
    for name in (
        "neutral_manifest_length",
        "sidecar_manifest_length",
    ):
        if type(value[name]) is not int or value[name] < 1:
            _fail("REJECT_Q05B_ACTOR_STDOUT", f"{name} differs")
    for name in (
        "neutral_manifest_raw_sha256",
        "neutral_manifest_root",
        "runtime_identity_sha256",
        "sidecar_manifest_raw_sha256",
        "sidecar_manifest_root",
        "source_identity_sha256",
    ):
        item = value[name]
        if type(item) is not str or len(item) != 64:
            _fail("REJECT_Q05B_ACTOR_STDOUT", f"{name} must be 64 lowercase hex")
        try:
            decoded = bytes.fromhex(item)
        except ValueError:
            _fail("REJECT_Q05B_ACTOR_STDOUT", f"{name} is not hex")
        if decoded.hex() != item:
            _fail("REJECT_Q05B_ACTOR_STDOUT", f"{name} is not lowercase hex")
    return value


FAILURE_CODE_REGISTRY: Final = (
    b"REJECT_Q05B_UINT",
    b"REJECT_Q05B_BYTES",
    b"REJECT_Q05B_ARRAY",
    b"REJECT_Q05B_CBOR",
    b"REJECT_Q05B_Q1_AUTHORITY",
    b"REJECT_Q05B_LEAF_ROW",
    b"REJECT_Q05B_LEAF_AST",
    b"REJECT_Q05B_LEAF_MANIFEST",
    b"REJECT_Q05B_LEAF_ORDER",
    b"REJECT_Q05B_NODE3_SCOPE",
    b"REJECT_Q05B_PARTITION",
    b"REJECT_Q05B_BOUNDED_STATE",
    b"REJECT_Q05B_SIDECAR_PATH",
    b"REJECT_Q05B_SIDECAR_KIND",
    b"REJECT_Q05B_SIDECAR",
    b"REJECT_Q05B_SIDECAR_REPLAY",
    b"REJECT_Q05B_GOLDEN",
    b"REJECT_Q05B_PREDICATE",
    b"REJECT_Q05B_CANDIDATE_RECEIPT",
    b"REJECT_Q05B_RECEIPT",
    b"REJECT_Q05B_RECEIPT_BINDING",
    b"REJECT_Q05B_ACTOR_STDOUT",
    b"FAIL_SHA256_PREIMAGE_COLLISION",
    b"INCONCLUSIVE_Q05B_OUTPUT_LIMIT",
    b"INCONCLUSIVE_Q05B_SCRATCH_LIMIT",
)

QUALIFICATION_SCHEMA_REGISTRY: Final = (
    (Q05B_FULL_LEAF_MANIFEST_ROW_TAG, FULL_LEAF_MANIFEST_ROW_SCHEMA_ID, 8),
    (Q05B_FULL_LEAF_MANIFEST_TAG, FULL_LEAF_MANIFEST_SCHEMA_ID, 8),
    (
        Q05B_NODE3_PARTITION_EVIDENCE_TAG,
        NODE3_PARTITION_EVIDENCE_SCHEMA_ID,
        10,
    ),
    (Q05B_SIDECAR_MANIFEST_TAG, SIDECAR_MANIFEST_SCHEMA_ID, 5),
    (Q05B_NODE3_GOLDEN_MANIFEST_TAG, NODE3_GOLDEN_MANIFEST_SCHEMA_ID, 21),
    (
        Q05B_QUALIFICATION_CANDIDATE_RECEIPT_TAG,
        QUALIFICATION_CANDIDATE_RECEIPT_SCHEMA_ID,
        25,
    ),
    (Q05B_QUALIFICATION_RECEIPT_TAG, QUALIFICATION_RECEIPT_SCHEMA_ID, 12),
    (Q05B_BOUNDED_NODE3_STATE_TAG, BOUNDED_NODE3_STATE_SCHEMA_ID, 26),
)
QUALIFICATION_HASH_DOMAIN_REGISTRY: Final = (
    (1, b"NODE3_PARTITION_EVIDENCE", NODE3_PARTITION_EVIDENCE_ROOT_DOMAIN.encode("ascii")),
    (2, b"SIDECAR_MANIFEST", SIDECAR_MANIFEST_ROOT_DOMAIN.encode("ascii")),
    (3, b"NODE3_GOLDEN_MANIFEST", NODE3_GOLDEN_MANIFEST_ROOT_DOMAIN.encode("ascii")),
    (4, b"BOUNDED_NODE3_STATE", BOUNDED_NODE3_STATE_ROOT_DOMAIN.encode("ascii")),
    (5, b"PREDICATE_REGISTRY", QUALIFICATION_PREDICATE_REGISTRY_ROOT_DOMAIN.encode("ascii")),
    (6, b"TAG_REGISTRY", QUALIFICATION_TAG_REGISTRY_ROOT_DOMAIN.encode("ascii")),
    (7, b"PRE_RECEIPT_EVIDENCE", PRE_RECEIPT_EVIDENCE_ROOT_DOMAIN.encode("ascii")),
    (8, b"PRE_RECEIPT", QUALIFICATION_PRE_RECEIPT_ROOT_DOMAIN.encode("ascii")),
    (9, b"CANDIDATE_RECEIPT", QUALIFICATION_CANDIDATE_RECEIPT_ROOT_DOMAIN.encode("ascii")),
    (10, b"PREDICATE20_EVIDENCE", PREDICATE20_EVIDENCE_ROOT_DOMAIN.encode("ascii")),
    (11, b"FINAL_RECEIPT", QUALIFICATION_RECEIPT_ROOT_DOMAIN.encode("ascii")),
    (12, b"WIRE_PROFILE", QUALIFICATION_WIRE_PROFILE_ROOT_DOMAIN.encode("ascii")),
)
Q1_AUTHORITY_SCHEMA_FIELDS: Final = (
    b"q1_state_id",
    b"q1_gate_count",
    b"q1_gate_mask",
    b"q1_gate_total",
    b"q1_output_slot_count",
    b"q1_output_slots",
    b"q1_receipt_or_null",
    b"q2_state_id",
    b"m3_formal_roots_or_null",
    b"formal_fixed_point_claimed",
    b"formal_fixed_point_tag_or_null",
    b"target_truth_accessed",
    b"split_accessed",
    b"role_evaluation_performed",
    b"outside_certificate_issued",
    b"active_transition_allowed",
)


def qualification_wire_profile_object_v1() -> tuple[object, ...]:
    return (
        1,
        b"hegel-q05b-qualification-wire-profile/1",
        QUALIFICATION_WIRE_VERSION.encode("ascii"),
        QUALIFICATION_TAG_REGISTRY_ROOT,
        Q05B_QUALIFICATION_TAG_REGISTRY,
        QUALIFICATION_SCHEMA_REGISTRY,
        QUALIFICATION_HASH_DOMAIN_REGISTRY,
        FAILURE_CODE_REGISTRY,
        ORDERED_OUTPUT_PATH_MODE_ROWS,
        (
            NODE3_MAXIMUM_AST_DEPTH,
            NODE3_MAXIMUM_AST_NODE_COUNT,
            NODE3_STRUCTURAL_BOUNDARY_DEPTH,
        ),
        (
            MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES,
            cbor_bstr_encoded_length_v1(MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES),
            MAX_CHUNK_FRAMED_BYTES,
            MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES + 1,
            framed_bstr_record_length_v1(
                MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES + 1
            ),
        ),
        Q1_AUTHORITY_SCHEMA_FIELDS,
        _q1_authority_object_v1(),
        QUALIFICATION_PREDICATE_REGISTRY_ROOT,
        QUALIFICATION_PREDICATE_REGISTRY,
        (
            _external_sort.EXTERNAL_SORT_TRACE_SCHEMA_ID,
            6,
            (
                b"version",
                b"schema_id",
                b"projection_object",
                b"ordered_rows",
                b"run_manifests",
                b"scratch_events",
            ),
        ),
        (
            _projection.COUNTING_DISCARD_STREAM_SCHEMA_ID,
            len(COUNTING_DISCARD_SCHEMA_FIELDS),
            COUNTING_DISCARD_SCHEMA_FIELDS,
            COUNTING_DISCARD_EQUALITY_RULES,
            PREDICATE14_SOURCE_CAPABILITY_FROZEN,
        ),
        (
            ACTOR_ENVELOPE_SCHEMA_VERSION.encode("ascii"),
            ACTOR_ACTION_ID.encode("ascii"),
            ACTOR_CANDIDATE_STATUS.encode("ascii"),
            tuple(
                (actor.encode("ascii"), implementation.encode("ascii"))
                for actor, implementation in ACTOR_IMPLEMENTATION_ID_REGISTRY
            ),
            tuple(field.encode("ascii") for field in ACTOR_STDOUT_REQUIRED_FIELDS),
        ),
    )


def qualification_wire_profile_root_v1() -> bytes:
    return content_hash(
        QUALIFICATION_WIRE_PROFILE_ROOT_DOMAIN,
        qualification_wire_profile_object_v1(),
    )


__all__ = [
    "ACTOR_ACTION_ID",
    "ACTOR_CANDIDATE_STATUS",
    "ACTOR_ENVELOPE_SCHEMA_VERSION",
    "ACTOR_IMPLEMENTATION_ID_REGISTRY",
    "ACTOR_STDOUT_REQUIRED_FIELDS",
    "COUNTING_DISCARD_EQUALITY_RULES",
    "COUNTING_DISCARD_SCHEMA_FIELDS",
    "COMMIT_A_ACTUAL_PRECONDITIONS_V1",
    "FAILURE_CODE_REGISTRY",
    "FULL_LEAF_MANIFEST_RELATIVE_PATH",
    "BOUNDED_NODE3_STATE_ROOT_DOMAIN",
    "FROZEN_NODE3_PRIMARY_COUNTS",
    "FULL_LEAF_MANIFEST_SIDECAR_CONTENT_ROOT_DOMAIN",
    "FULL_V16_LEAF_COUNT",
    "MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES",
    "MAX_CHUNK_FRAMED_BYTES",
    "IMPLEMENTATION_BLOCKED_PREDICATE_IDS",
    "NODE3_GOLDEN_MANIFEST_RELATIVE_PATH",
    "NODE3_GOLDEN_MANIFEST_ROOT_DOMAIN",
    "NODE3_PARTITION_EVIDENCE_ROOT_DOMAIN",
    "ODD_PARTITION_EVIDENCE_RELATIVE_PATH",
    "ORDERED_OUTPUT_RELATIVE_PATHS",
    "ORDERED_OUTPUT_PATH_MODE_ROWS",
    "ORDERED_PREIMAGE_RELATIVE_PATHS",
    "Q05BFullLeafManifestRowV1",
    "Q05BFullLeafManifestV1",
    "Q05BBoundedNode3StateV1",
    "Q05BNode3GoldenManifestV1",
    "Q05BNode3PartitionEvidenceV1",
    "Q05BQualificationCandidateReceiptV1",
    "Q05BQualificationReceiptV1",
    "Q05BSidecarManifestV1",
    "Q05BWireQualificationError",
    "Q05B_QUALIFICATION_TAG_REGISTRY",
    "Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION",
    "Q1_NULL_OUTPUT_SLOTS",
    "PREDICATE14_SOURCE_CAPABILITY_FROZEN",
    "PENDING_ACTUAL_EVIDENCE_PREDICATE_IDS",
    "QUALIFICATION_ENGINEERING_STATUS",
    "QUALIFICATION_PREDICATE_REGISTRY",
    "QUALIFICATION_PREDICATE_REGISTRY_ROOT",
    "QUALIFICATION_TAG_REGISTRY_ROOT",
    "OUTPUT_FILE_MODE",
    "SEMANTIC_SOURCE_ROOTS",
    "SIDECAR_MANIFEST_RELATIVE_PATH",
    "SIDECAR_MANIFEST_ROOT_DOMAIN",
    "SINK_PARTITION_EVIDENCE_RELATIVE_PATH",
    "VERSION_BINDING_ROWS",
    "bounded_node3_state_v1",
    "bounded_node3_state_from_object_v1",
    "bstr_record_fits_frozen_chunk_v1",
    "cbor_bstr_encoded_length_v1",
    "decode_full_v16_leaf_manifest_v1",
    "decode_node3_golden_manifest_v1",
    "decode_node3_partition_evidence_v1",
    "decode_qualification_candidate_receipt_v1",
    "decode_qualification_receipt_v1",
    "framed_bstr_record_length_v1",
    "full_v16_leaf_manifest_v1",
    "node3_golden_manifest_v1",
    "node3_partition_evidence_v1",
    "q1_semantic_and_projection_roots_v1",
    "q1_semantic_binding_manifest_v1",
    "pre_receipt_evidence_root_v1",
    "predicate20_evidence_root_v1",
    "qualification_wire_profile_object_v1",
    "qualification_wire_profile_root_v1",
    "replay_sidecar_manifest_v1",
    "sidecar_manifest_v1",
    "validate_actor_stdout_envelope_v1",
    "validate_q1_authority_closed_v1",
]

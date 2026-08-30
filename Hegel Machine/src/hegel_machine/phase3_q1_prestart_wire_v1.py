"""Strict Q1 prestart control-plane wire.

This module defines the candidate owner-authorizable wire for admitting the
first fourteen Q1 gates while Q1 remains ``NOT_RUN``.  It is deliberately
separate from the frozen formal-archive tag registry.  Constructing or
replaying these objects cannot start Q1, invoke an endpoint, or populate any
formal output slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NoReturn

from .phase3_q1_formal_archive_contract_v1 import Q1_OUTPUT_SLOT_NAMES
from .strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode, content_hash


Q1_RUN_GENESIS_TAG: Final = 0x3B00
Q1_PRESTART_ADMISSION_RECEIPT_TAG: Final = 0x3B01

RUN_GENESIS_SCHEMA_ID: Final = b"hegel-q1-run-genesis/1"
PRESTART_RECEIPT_SCHEMA_ID: Final = b"hegel-q1-prestart-admission-receipt/1"
TAG_REGISTRY_SCHEMA_ID: Final = b"hegel-q1-prestart-tag-registry/1"
GATE_REGISTRY_SCHEMA_ID: Final = b"hegel-q1-gate-registry/1"
SOURCE_IDENTITY_SCHEMA_ID: Final = b"git-sha1-commit-tree-sha256-closure/1"
FORMAL_RESOURCE_ENVELOPE_SCHEMA_ID: Final = b"hegel-q1-formal-resource-envelope/1"

TAG_REGISTRY_ROOT_DOMAIN: Final = "HEGEL/Q1/PRESTART_TAG_REGISTRY/V1"
GATE_REGISTRY_ROOT_DOMAIN: Final = "HEGEL/Q1/GATE_REGISTRY/V1"
RUN_GENESIS_ROOT_DOMAIN: Final = "HEGEL/Q1/RUN_GENESIS/V1"
GATE_EVIDENCE_ROOT_DOMAIN: Final = "HEGEL/Q1/PRESTART_GATE_EVIDENCE/V1"
PRESTART_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/PRESTART_ADMISSION_RECEIPT/V1"
)
FORMAL_OUTPUT_LAYOUT_ROOT_DOMAIN: Final = "HEGEL/Q1/FORMAL_OUTPUT_LAYOUT/V1"
FORMAL_ACTOR_IMPLEMENTATION_ROOT_DOMAIN: Final = (
    "HEGEL/Q1/FORMAL_ACTOR_IMPLEMENTATION/V1"
)

PRESTART_TAG_REGISTRY: Final = (
    (Q1_RUN_GENESIS_TAG, b"Q1_RUN_GENESIS"),
    (Q1_PRESTART_ADMISSION_RECEIPT_TAG, b"Q1_PRESTART_ADMISSION_RECEIPT"),
)

VERSION_BINDING_ROWS: Final = (
    (1, b"dsl_version", b"hegel-old-dsl-v1.6.0"),
    (2, b"dsl_freeze_version", b"hegel-freeze-p2b-p3-v1.6.0"),
    (3, b"closure_semantics_version", b"hegel-quotient-closure-v1.0.1"),
    (4, b"archive_wire_version", b"hegel-q1-archive-wire-v1.0.0"),
    (
        5,
        b"projection_freeze_version",
        b"hegel-freeze-p3a-q05a-q1-projection-v1.0.0",
    ),
    (6, b"prestart_wire_version", b"hegel-q1-prestart-wire-v1.0.0"),
)

FORMAL_OUTPUT_LAYOUT_ROWS: Final = (
    (1, b"signature_ids", (1, 2)),
    (2, b"stream_ids", (1, 2, 3, 4)),
    (3, b"chunk_path_template", b"signature-%02u/stream-%02u/chunk-%06u.framed"),
    (4, b"chunk_manifest_path_template", b"signature-%02u/chunk-manifests.framed"),
    (5, b"fixed_point_path_template", b"signature-%02u/fixed-point.framed"),
    (6, b"signature_manifest_path_template", b"signature-%02u/signature-manifest.framed"),
    (7, b"closure_bundle_path", b"closure-bundle.framed"),
    (8, b"endpoint_result_path", b"endpoint-result.json"),
    (9, b"frame_length_prefix_bytes", 4),
    (10, b"compression_id", 0),
)
FORMAL_OUTPUT_LAYOUT_ROOT: Final = content_hash(
    FORMAL_OUTPUT_LAYOUT_ROOT_DOMAIN,
    (1, b"hegel-q1-formal-output-layout/1", FORMAL_OUTPUT_LAYOUT_ROWS),
)

FORMAL_ACTOR_REGISTRY: Final = (
    (
        b"python",
        b"hegel-python-q1-formal-archive-materializer-v1",
        b"materialize-full-node6-formal-archive-v1",
    ),
    (
        b"rust",
        b"hegel-rust-q1-formal-archive-materializer-v1",
        b"materialize-full-node6-formal-archive-v1",
    ),
    (
        b"host",
        b"hegel-host-q1-formal-archive-replay-v1",
        b"replay-full-node6-formal-archive-v1",
    ),
)

Q1_GATE_REGISTRY: Final = (
    (1, b"POST_SHRINK6_NORMATIVE_AND_Q1_FREEZE_BYTES_BOUND"),
    (2, b"Q0_14_OF_14_RECEIPT_AND_SOURCE_REPLAYED"),
    (3, b"V16_FULL_DSL_STRICT_ROOT_FAMILY_REQUALIFIED"),
    (4, b"PRODUCTION_UNIVERSE_ROWS_AND_ADAPTERS_DUAL_QUALIFIED"),
    (5, b"Q1_BEHAVIOR_BOTTOM_AND_COLLISION_CODEC_QUALIFIED"),
    (6, b"Q1_CONSTRUCTION_SIGNATURE_AND_MULTIPLICITY_POLICY_QUALIFIED"),
    (7, b"Q1_PARETO_DOMINANCE_AND_CLASS_MDL_QUALIFIED"),
    (8, b"FULL_OPERATOR_CONGRUENCE_AND_REWRITE_PROFILES_QUALIFIED"),
    (9, b"STRUCTURAL_INDUCTION_AND_FIXED_POINT_CONTRACT_QUALIFIED"),
    (10, b"Q1_ARCHIVE_WIRE_ROOT_DAG_RESOURCE_PROJECTION_AND_GOLDENS_PASS"),
    (11, b"TARGET_BLIND_CAPACITY_PREFLIGHT_AND_RESOURCE_ENVELOPE_FROZEN"),
    (12, b"PYTHON_RUST_IMPLEMENTATION_AND_OFFLINE_ISOLATION_BOUND"),
    (13, b"TARGET_TRUTH_SPLIT_ROLE_INPUTS_ABSENT"),
    (14, b"Q1_RUN_GENESIS_NON_NULL_AND_EIGHT_OUTPUT_SLOTS_NULL"),
    (15, b"BOTH_SIGNATURE_FIXED_POINTS_COMPLETE_NO_GUARD_HIT"),
    (16, b"PYTHON_RUST_SIGNATURE_ARCHIVES_AND_STATE_PREIMAGES_BYTE_EQUAL"),
    (17, b"HOST_REPLAY_DUAL_SIGNATURE_CLOSURES_PASS"),
    (18, b"TARGET_BLIND_ACCESS_LEDGER_AND_ISOLATION_POSTCHECK_PASS"),
    (19, b"Q1_CLOSURE_BUNDLE_AND_COMPLETION_RECEIPT_SEALED"),
    (20, b"Q1_COMPLETE_Q2_M3_ROLE_CERTIFICATE_OUTPUTS_NULL_NOT_RUN"),
)

# Each tuple fixes the exact positional subject wire for one prestart gate.
# Kinds are enforced by :func:`_validate_gate_subjects_v1`.
Q1_PRESTART_GATE_SUBJECT_REGISTRY: Final = (
    (1, ((b"post_shrink6_normative_raw_sha256", b"root32"),
         (b"q1_freeze_raw_sha256", b"root32"),
         (b"prestart_wire_contract_raw_sha256", b"root32"))),
    (2, ((b"q0_artifact_raw_sha256", b"root32"),
         (b"q0_receipt_root", b"root32"),
         (b"q0_source_commit", b"bytes20"))),
    (3, ((b"q0_receipt_root", b"root32"),
         (b"full_v16_leaf_manifest_root", b"root32"),
         (b"q1_semantic_binding_root", b"root32"))),
    (4, ((b"q0_receipt_root", b"root32"),
         (b"odd_universe_root", b"root32"),
         (b"sink_universe_root", b"root32"),
         (b"adapter_binding_root", b"root32"))),
    (5, ((b"q0_receipt_root", b"root32"),
         (b"q05b_receipt_root", b"root32"),
         (b"behavior_codec_root", b"root32"),
         (b"dual_semantic_agreement_root", b"root32"))),
    (6, ((b"q0_receipt_root", b"root32"),
         (b"q05b_receipt_root", b"root32"),
         (b"construction_multiplicity_policy_root", b"root32"))),
    (7, ((b"q0_receipt_root", b"root32"),
         (b"q05b_receipt_root", b"root32"),
         (b"pareto_class_mdl_policy_root", b"root32"))),
    (8, ((b"q0_receipt_root", b"root32"),
         (b"q05b_receipt_root", b"root32"),
         (b"operator_congruence_rewrite_root", b"root32"))),
    (9, ((b"q0_receipt_root", b"root32"),
         (b"v6_artifact_raw_sha256", b"root32"),
         (b"fixed_point_contract_root", b"root32"))),
    (10, ((b"q05b_artifact_raw_sha256", b"root32"),
          (b"q05b_receipt_root", b"root32"),
          (b"final_archive_wire_source_closure_root", b"root32"))),
    (11, ((b"v6_capsule_digest", b"root32"),
          (b"v6_terminal_record_digest", b"root32"),
          (b"v6_artifact_raw_sha256", b"root32"),
          (b"v6_outcome_digest", b"root32"),
          (b"formal_resource_envelope", b"formal_resource_envelope"))),
    (12, ((b"final_implementation_root_rows", b"implementation_rows"),
          (b"final_actor_source_closure_root", b"root32"),
          (b"offline_runtime_isolation_root", b"root32"),
          (b"formal_output_layout_root", b"root32"))),
    (13, ((b"final_actor_source_closure_root", b"root32"),
          (b"mount_manifest_root", b"root32"),
          (b"input_key_registry_absence_root", b"root32"),
          (b"target_truth_accessed", b"false"),
          (b"split_accessed", b"false"),
          (b"role_evaluation_performed", b"false"))),
    (14, ((b"run_genesis_root", b"root32"),
          (b"run_genesis_object", b"run_genesis_object"),
          (b"q1_output_slots", b"null_output_slots"),
          (b"q1_start_authorized", b"false"))),
)

FORMAL_RESOURCE_ENVELOPE: Final = (
    1,
    FORMAL_RESOURCE_ENVELOPE_SCHEMA_ID,
    (
        (1, ((1, 32768), (2, 1024), (3, 4096), (4, 32),
             (5, 8192), (6, 4096), (7, 16384), (8, 7))),
        (2, ((1, 32768), (2, 256), (3, 1024), (4, 256),
             (5, 8192), (6, 4096), (7, 16384), (8, 7))),
    ),
    (
        (1, b"python", ((9, 16777216), (10, 4194304),
                         (11, 2147483648), (12, 3600))),
        (2, b"rust", ((9, 16777216), (10, 4194304),
                       (11, 2147483648), (12, 3600))),
        (3, b"host", ((9, 2097152), (10, 1),
                       (11, 67108864), (12, 3600))),
    ),
)

Q1_GATE_TOTAL: Final = 20
Q1_PRESTART_GATE_COUNT: Final = 14
Q1_PRESTART_GATE_MASK: Final = 0x003FFF


class Q1PrestartWireError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q1PrestartWireError(code, detail)


def _exact_bytes(value: object, length: int, label: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_Q1_PRESTART_BYTES", f"{label} must be exactly {length} bytes")
    return value


def _uint(value: object, label: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 64) - 1:
        _fail("REJECT_Q1_PRESTART_UINT", f"{label} must be uint64")
    return value


def _strict_object(payload: bytes, label: str) -> tuple[object, ...]:
    if type(payload) is not bytes:
        _fail("REJECT_Q1_PRESTART_CBOR", f"{label} payload must be bytes")
    try:
        value = canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q1_PRESTART_CBOR", f"{label}: {error}")
    if type(value) is not tuple or canonical_cbor_encode(value) != payload:
        _fail("REJECT_Q1_PRESTART_CBOR", f"{label} is not strict canonical array CBOR")
    return value


def _implementation_rows(value: object) -> tuple[tuple[bytes, bytes], ...]:
    if type(value) is not tuple or len(value) != 3:
        _fail("REJECT_Q1_IMPLEMENTATION_ROWS", "three final actor rows are required")
    expected = (b"python", b"rust", b"host")
    rows: list[tuple[bytes, bytes]] = []
    for actor, row in zip(expected, value, strict=True):
        if type(row) is not tuple or len(row) != 2 or row[0] != actor:
            _fail("REJECT_Q1_IMPLEMENTATION_ROWS", "actor order or row shape differs")
        rows.append((actor, _exact_bytes(row[1], 32, f"{actor!r} implementation root")))
    return tuple(rows)


def q1_null_output_slots_v1() -> tuple[tuple[object, ...], ...]:
    return tuple((index, name, None) for index, name in enumerate(Q1_OUTPUT_SLOT_NAMES, 1))


def q1_prestart_authority_object_v1() -> tuple[object, ...]:
    return (
        0, Q1_PRESTART_GATE_COUNT, Q1_PRESTART_GATE_MASK, Q1_GATE_TOTAL,
        len(Q1_OUTPUT_SLOT_NAMES), q1_null_output_slots_v1(), None,
        0, None, False, None, False, False, False, False, False, False, False,
    )


PRESTART_TAG_REGISTRY_ROOT: Final = content_hash(
    TAG_REGISTRY_ROOT_DOMAIN,
    (1, TAG_REGISTRY_SCHEMA_ID, PRESTART_TAG_REGISTRY),
)
Q1_GATE_REGISTRY_ROOT: Final = content_hash(
    GATE_REGISTRY_ROOT_DOMAIN,
    (1, GATE_REGISTRY_SCHEMA_ID, Q1_GATE_REGISTRY, Q1_PRESTART_GATE_SUBJECT_REGISTRY),
)


@dataclass(frozen=True, slots=True)
class Q1SourceIdentityV1:
    source_commit: bytes
    source_tree: bytes
    source_closure_sha256: bytes

    def __post_init__(self) -> None:
        _exact_bytes(self.source_commit, 20, "source commit")
        _exact_bytes(self.source_tree, 20, "source tree")
        _exact_bytes(self.source_closure_sha256, 32, "source closure")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1, SOURCE_IDENTITY_SCHEMA_ID, self.source_commit,
            self.source_tree, self.source_closure_sha256,
        )


@dataclass(frozen=True, slots=True)
class Q1FormalActorIdentityV1:
    actor_id: bytes
    implementation_id: bytes
    action_id: bytes
    source_manifest_root: bytes
    entrypoint_raw_sha256: bytes
    dependency_closure_root: bytes
    runtime_identity_root: bytes
    offline_isolation_root: bytes

    def __post_init__(self) -> None:
        matching = tuple(row for row in FORMAL_ACTOR_REGISTRY if row[0] == self.actor_id)
        if len(matching) != 1 or matching[0][1:] != (
            self.implementation_id, self.action_id,
        ):
            _fail("REJECT_Q1_FORMAL_ACTOR_IDENTITY", "actor identity/action differs")
        for label, root in (
            ("source manifest", self.source_manifest_root),
            ("entrypoint", self.entrypoint_raw_sha256),
            ("dependency closure", self.dependency_closure_root),
            ("runtime identity", self.runtime_identity_root),
            ("offline isolation", self.offline_isolation_root),
        ):
            _exact_bytes(root, 32, label)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            b"hegel-q1-formal-actor-implementation/1",
            self.actor_id,
            self.implementation_id,
            self.action_id,
            self.source_manifest_root,
            self.entrypoint_raw_sha256,
            self.dependency_closure_root,
            self.runtime_identity_root,
            self.offline_isolation_root,
            FORMAL_OUTPUT_LAYOUT_ROOT,
            (),
            False,
            False,
            False,
        )

    @property
    def implementation_root(self) -> bytes:
        return content_hash(
            FORMAL_ACTOR_IMPLEMENTATION_ROOT_DOMAIN,
            self.canonical_object(),
        )


@dataclass(frozen=True, slots=True)
class Q1RunGenesisV1:
    run_id: bytes
    source_identity: Q1SourceIdentityV1
    q1_semantic_binding_root: bytes
    q1_projection_profile_root: bytes
    q0_14_of_14_receipt_root: bytes
    q05b_qualification_receipt_root: bytes
    full_v16_leaf_manifest_root: bytes
    final_implementation_root_rows: tuple[tuple[bytes, bytes], ...]
    created_at_unix_seconds: int

    def __post_init__(self) -> None:
        _exact_bytes(self.run_id, 16, "run ID")
        if type(self.source_identity) is not Q1SourceIdentityV1:
            _fail("REJECT_Q1_RUN_GENESIS", "source identity exact type differs")
        for label, root in (
            ("semantic binding", self.q1_semantic_binding_root),
            ("projection profile", self.q1_projection_profile_root),
            ("Q0 receipt", self.q0_14_of_14_receipt_root),
            ("Q0.5b receipt", self.q05b_qualification_receipt_root),
            ("full v1.6 leaf manifest", self.full_v16_leaf_manifest_root),
        ):
            _exact_bytes(root, 32, label)
        _implementation_rows(self.final_implementation_root_rows)
        _uint(self.created_at_unix_seconds, "created_at_unix_seconds")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_RUN_GENESIS_TAG,
            RUN_GENESIS_SCHEMA_ID,
            self.run_id,
            self.source_identity.canonical_object(),
            VERSION_BINDING_ROWS,
            PRESTART_TAG_REGISTRY_ROOT,
            Q1_GATE_REGISTRY_ROOT,
            self.q1_semantic_binding_root,
            self.q1_projection_profile_root,
            self.q0_14_of_14_receipt_root,
            self.q05b_qualification_receipt_root,
            self.full_v16_leaf_manifest_root,
            FORMAL_RESOURCE_ENVELOPE,
            self.final_implementation_root_rows,
            q1_prestart_authority_object_v1(),
            self.created_at_unix_seconds,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def genesis_root(self) -> bytes:
        return content_hash(RUN_GENESIS_ROOT_DOMAIN, self.canonical_object())


def decode_run_genesis_v1(payload: bytes) -> Q1RunGenesisV1:
    value = _strict_object(payload, "run genesis")
    if (
        len(value) != 17
        or value[:3] != (1, Q1_RUN_GENESIS_TAG, RUN_GENESIS_SCHEMA_ID)
        or value[5] != VERSION_BINDING_ROWS
        or value[6] != PRESTART_TAG_REGISTRY_ROOT
        or value[7] != Q1_GATE_REGISTRY_ROOT
        or value[13] != FORMAL_RESOURCE_ENVELOPE
        or value[15] != q1_prestart_authority_object_v1()
    ):
        _fail("REJECT_Q1_RUN_GENESIS", "run genesis fixed wire differs")
    source = value[4]
    if type(source) is not tuple or len(source) != 5 or source[:2] != (
        1, SOURCE_IDENTITY_SCHEMA_ID,
    ):
        _fail("REJECT_Q1_RUN_GENESIS", "source identity wire differs")
    genesis = Q1RunGenesisV1(
        run_id=value[3],
        source_identity=Q1SourceIdentityV1(source[2], source[3], source[4]),
        q1_semantic_binding_root=value[8],
        q1_projection_profile_root=value[9],
        q0_14_of_14_receipt_root=value[10],
        q05b_qualification_receipt_root=value[11],
        full_v16_leaf_manifest_root=value[12],
        final_implementation_root_rows=value[14],
        created_at_unix_seconds=value[16],
    )
    if genesis.canonical_bytes != payload:
        _fail("REJECT_Q1_RUN_GENESIS", "run genesis replay differs")
    return genesis


def _validate_gate_subjects_v1(
    gate_id: int,
    subjects: object,
    genesis: Q1RunGenesisV1,
) -> tuple[object, ...]:
    if type(gate_id) is not int or not 1 <= gate_id <= Q1_PRESTART_GATE_COUNT:
        _fail("REJECT_Q1_GATE_EVIDENCE", "gate ID is outside prestart range")
    if type(subjects) is not tuple:
        _fail("REJECT_Q1_GATE_EVIDENCE", "gate subjects must be an exact tuple")
    specs = Q1_PRESTART_GATE_SUBJECT_REGISTRY[gate_id - 1][1]
    if len(subjects) != len(specs):
        _fail("REJECT_Q1_GATE_EVIDENCE", "gate subject cardinality differs")
    for value, (_, kind) in zip(subjects, specs, strict=True):
        if kind == b"root32":
            _exact_bytes(value, 32, "gate subject root")
        elif kind == b"bytes20":
            _exact_bytes(value, 20, "gate subject commit")
        elif kind == b"false":
            if value is not False:
                _fail("REJECT_Q1_GATE_EVIDENCE", "required false subject differs")
        elif kind == b"formal_resource_envelope":
            if value != FORMAL_RESOURCE_ENVELOPE:
                _fail("REJECT_Q1_GATE_EVIDENCE", "resource envelope differs")
        elif kind == b"implementation_rows":
            if _implementation_rows(value) != genesis.final_implementation_root_rows:
                _fail("REJECT_Q1_GATE_EVIDENCE", "implementation subjects differ")
        elif kind == b"run_genesis_object":
            if value != genesis.canonical_object():
                _fail("REJECT_Q1_GATE_EVIDENCE", "embedded genesis differs")
        elif kind == b"null_output_slots":
            if value != q1_null_output_slots_v1():
                _fail("REJECT_Q1_GATE_EVIDENCE", "null output slots differ")
        else:  # pragma: no cover - frozen registry construction guard
            raise AssertionError(f"unknown gate subject kind {kind!r}")
    if gate_id == 14 and subjects[0] != genesis.genesis_root:
        _fail("REJECT_Q1_GATE_EVIDENCE", "Gate 14 genesis root differs")
    return subjects


def gate_evidence_root_v1(
    source_identity: Q1SourceIdentityV1,
    genesis: Q1RunGenesisV1,
    gate_id: int,
    subjects: tuple[object, ...],
) -> bytes:
    if type(source_identity) is not Q1SourceIdentityV1:
        _fail("REJECT_Q1_GATE_EVIDENCE", "source identity exact type differs")
    if type(genesis) is not Q1RunGenesisV1:
        _fail("REJECT_Q1_GATE_EVIDENCE", "genesis exact type differs")
    if source_identity != genesis.source_identity:
        _fail("REJECT_Q1_GATE_EVIDENCE", "source identity/genesis binding differs")
    checked = _validate_gate_subjects_v1(gate_id, subjects, genesis)
    registry_row = Q1_GATE_REGISTRY[gate_id - 1]
    return content_hash(
        GATE_EVIDENCE_ROOT_DOMAIN,
        (
            source_identity.canonical_object(), Q1_GATE_REGISTRY_ROOT,
            gate_id, registry_row[1], checked,
        ),
    )


def gate_row_v1(
    source_identity: Q1SourceIdentityV1,
    genesis: Q1RunGenesisV1,
    gate_id: int,
    subjects: tuple[object, ...],
) -> tuple[object, ...]:
    return (
        gate_id,
        Q1_GATE_REGISTRY[gate_id - 1][1],
        True,
        gate_evidence_root_v1(source_identity, genesis, gate_id, subjects),
    )


def verify_prestart_gate_evidence_v1(
    receipt: "Q1PrestartAdmissionReceiptV1",
    gate_subject_rows_1_through_14: tuple[tuple[object, ...], ...],
) -> None:
    """Recompute all fourteen evidence roots from the exact subject rows.

    The compact receipt carries evidence roots, while an enclosing capsule or
    authoritative artifact carries the replayed subject rows.  Admission must
    call this function before accepting the receipt.
    """

    if type(receipt) is not Q1PrestartAdmissionReceiptV1:
        _fail("REJECT_Q1_GATE_EVIDENCE", "receipt exact type differs")
    if type(gate_subject_rows_1_through_14) is not tuple or len(
        gate_subject_rows_1_through_14
    ) != Q1_PRESTART_GATE_COUNT:
        _fail("REJECT_Q1_GATE_EVIDENCE", "fourteen subject rows are required")
    expected_rows = tuple(
        gate_row_v1(
            receipt.genesis.source_identity,
            receipt.genesis,
            gate_id,
            subjects,
        )
        for gate_id, subjects in enumerate(gate_subject_rows_1_through_14, 1)
    )
    if expected_rows != receipt.gate_rows_1_through_14:
        _fail("REJECT_Q1_GATE_EVIDENCE", "receipt evidence roots differ from replay")


@dataclass(frozen=True, slots=True)
class Q1PrestartAdmissionReceiptV1:
    genesis: Q1RunGenesisV1
    gate_rows_1_through_14: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        if type(self.genesis) is not Q1RunGenesisV1:
            _fail("REJECT_Q1_PRESTART_RECEIPT", "genesis exact type differs")
        if type(self.gate_rows_1_through_14) is not tuple or len(
            self.gate_rows_1_through_14
        ) != Q1_PRESTART_GATE_COUNT:
            _fail("REJECT_Q1_PRESTART_RECEIPT", "exactly fourteen gate rows required")
        for expected, row in zip(
            Q1_GATE_REGISTRY[:Q1_PRESTART_GATE_COUNT],
            self.gate_rows_1_through_14,
            strict=True,
        ):
            if (
                type(row) is not tuple
                or len(row) != 4
                or row[:3] != (expected[0], expected[1], True)
                or type(row[3]) is not bytes
                or len(row[3]) != 32
            ):
                _fail("REJECT_Q1_PRESTART_RECEIPT", "gate row wire differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q1_PRESTART_ADMISSION_RECEIPT_TAG,
            PRESTART_RECEIPT_SCHEMA_ID,
            self.genesis.source_identity.canonical_object(),
            VERSION_BINDING_ROWS,
            PRESTART_TAG_REGISTRY_ROOT,
            Q1_GATE_REGISTRY_ROOT,
            Q1_GATE_REGISTRY,
            self.genesis.genesis_root,
            self.genesis.canonical_object(),
            Q1_PRESTART_GATE_COUNT,
            self.gate_rows_1_through_14,
            Q1_PRESTART_GATE_COUNT,
            Q1_PRESTART_GATE_MASK,
            True,
            q1_prestart_authority_object_v1(),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(PRESTART_RECEIPT_ROOT_DOMAIN, self.canonical_object())


def decode_prestart_admission_receipt_v1(
    payload: bytes,
) -> Q1PrestartAdmissionReceiptV1:
    value = _strict_object(payload, "prestart admission receipt")
    if (
        len(value) != 16
        or value[:3] != (
            1, Q1_PRESTART_ADMISSION_RECEIPT_TAG, PRESTART_RECEIPT_SCHEMA_ID,
        )
        or value[4] != VERSION_BINDING_ROWS
        or value[5] != PRESTART_TAG_REGISTRY_ROOT
        or value[6] != Q1_GATE_REGISTRY_ROOT
        or value[7] != Q1_GATE_REGISTRY
        or value[10] != Q1_PRESTART_GATE_COUNT
        or value[12:15] != (
            Q1_PRESTART_GATE_COUNT, Q1_PRESTART_GATE_MASK, True,
        )
        or value[15] != q1_prestart_authority_object_v1()
    ):
        _fail("REJECT_Q1_PRESTART_RECEIPT", "receipt fixed wire differs")
    genesis_payload = canonical_cbor_encode(value[9])
    genesis = decode_run_genesis_v1(genesis_payload)
    if value[3] != genesis.source_identity.canonical_object() or value[8] != genesis.genesis_root:
        _fail("REJECT_Q1_PRESTART_RECEIPT", "receipt/genesis binding differs")
    rows = value[11]
    receipt = Q1PrestartAdmissionReceiptV1(genesis, rows)
    if receipt.canonical_bytes != payload:
        _fail("REJECT_Q1_PRESTART_RECEIPT", "receipt replay differs")
    return receipt


__all__ = [
    "FORMAL_ACTOR_REGISTRY",
    "FORMAL_OUTPUT_LAYOUT_ROOT",
    "FORMAL_OUTPUT_LAYOUT_ROWS",
    "FORMAL_RESOURCE_ENVELOPE",
    "PRESTART_TAG_REGISTRY",
    "PRESTART_TAG_REGISTRY_ROOT",
    "Q1_GATE_REGISTRY",
    "Q1_GATE_REGISTRY_ROOT",
    "Q1_PRESTART_GATE_COUNT",
    "Q1_PRESTART_GATE_MASK",
    "Q1_PRESTART_GATE_SUBJECT_REGISTRY",
    "Q1PrestartAdmissionReceiptV1",
    "Q1PrestartWireError",
    "Q1FormalActorIdentityV1",
    "Q1RunGenesisV1",
    "Q1SourceIdentityV1",
    "VERSION_BINDING_ROWS",
    "decode_prestart_admission_receipt_v1",
    "decode_run_genesis_v1",
    "gate_evidence_root_v1",
    "gate_row_v1",
    "q1_null_output_slots_v1",
    "q1_prestart_authority_object_v1",
    "verify_prestart_gate_evidence_v1",
]

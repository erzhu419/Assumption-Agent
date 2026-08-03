"""Strict wire and state primitives for the internal-shadow M3 track.

This module intentionally has no authority over the formal M2.5/M3 wire.  It
reuses only the deterministic CBOR codec and allocates a disjoint tag range,
hash domain, state machine, and 12-gate admission registry.  In particular, no
object produced here is a formal root, certificate, external attestation, or
authorization to run ``phase3-m3-start``.

The module contains no CSPRNG, key generation, signing, process launching, or
filesystem mutation.  Those effects belong to the purpose-separated runtime;
this file only validates the bytes that the runtime is allowed to exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
import re
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .strict_cbor_v1 import (
    StrictCborValue,
    canonical_cbor_decode,
    canonical_cbor_encode,
)


SHADOW_TRACK_ID: Final = "hegel-internal-shadow-v1"
FORMAL_MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
FORMAL_CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
SHADOW_ARTIFACT_KIND: Final = "INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE"
SHADOW_ARTIFACT_KIND_ID: Final = 1
SHADOW_ISOLATION_PROFILE_ID: Final = 1
SHADOW_GATE_COUNT: Final = 12
SHADOW_ALL_GATES_BITSET: Final = 0x0FFF
SHADOW_ISOLATION_INVARIANT_BITSET: Final = 0x3FFFF
SHADOW_SIGNATURE_PREFIX: Final = (
    b"HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/SIGNATURE/V1"
)
_SHADOW_DIGEST_PREFIX: Final = b"HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/"
_SHADOW_TREE_PREFIX: Final = b"HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/TREE/"
_TREE_DOMAIN_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_UINT64_MAX: Final = (1 << 64) - 1


class ShadowWireError(ValueError):
    """Stable fail-closed rejection from the shadow-only wire."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> "None":
    raise ShadowWireError(code, detail)


class ShadowArtifactKindId(IntEnum):
    INVALID = 0
    INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE = 1


class ShadowPurposeId(IntEnum):
    INVALID = 0
    SHADOW_CUSTODIAN_AND_SPLIT_COORDINATOR = 1
    SHADOW_PYTHON_REPLAY_WORKER = 2
    SHADOW_RUST_REPLAY_WORKER = 3
    SHADOW_POLICY_AUDIT_WORKER = 4


class ShadowStateId(IntEnum):
    NOT_ADMITTED = 0
    ADMITTED_NOT_STARTED = 1
    RUNNING_CANONICAL_ENUMERATION = 2
    RUNNING_ROLE_EVALUATION = 3
    COMPLETE_CANDIDATE = 4
    DSL_TOO_LARGE_CANDIDATE = 5
    INCONCLUSIVE_BUDGET = 6
    INCONCLUSIVE_SEMANTICS = 7
    INCONCLUSIVE_EXECUTION = 8
    ABORTED_POLICY_VIOLATION = 9
    ABORTED_OPERATOR = 10


class ShadowTransitionReasonId(IntEnum):
    INVALID = 0
    SHADOW_ADMISSION_GATES_12_OF_12 = 1
    EXPLICIT_SHADOW_START = 2
    CANONICAL_FRONTIER_CLOSED = 3
    SYNTACTIC_CAPACITY_WITNESS_50001 = 4
    SEARCH_BUDGET_HIT = 5
    SEMANTICS_OR_DUAL_REPLAY_MISMATCH = 6
    EXECUTION_FAILURE = 7
    ROLE_EVALUATION_COMPLETE = 8
    POLICY_VIOLATION = 9
    EXPLICIT_OPERATOR_ABORT = 10


class ShadowOutcomeId(IntEnum):
    NOT_RUN = 0
    COMPLETE_CANDIDATE = 1
    DSL_TOO_LARGE_CANDIDATE = 2
    INCONCLUSIVE_BUDGET = 3
    INCONCLUSIVE_SEMANTICS = 4
    INCONCLUSIVE_EXECUTION = 5
    ABORTED_POLICY_VIOLATION = 6
    ABORTED_OPERATOR = 7


class ShadowDisclosureEventTypeId(IntEnum):
    INVALID = 0
    LEDGER_GENESIS = 1
    SEALED_ASSIGNMENT_DELIVERED_TO_ROLE_EVALUATOR = 2
    TERMINAL_ROLE_SUMMARY_REVEALED_TO_ORCHESTRATOR = 3
    FORBIDDEN_ARTIFACT_EXPOSED_TO_SYNTHESIS = 4
    PUBLIC_SHADOW_ARTIFACT_PUBLISHED = 5


class ShadowAdmissionGateId(IntEnum):
    SHADOW_OWNER_POLICY_BOUND = 1
    SHADOW_BASIS_COMMIT_PINNED = 2
    SHADOW_READ_ONLY_SNAPSHOT_VERIFIED = 3
    SHADOW_DETERMINISTIC_DUAL_BASELINE_PASS = 4
    FORMAL_TRACK_INVARIANTS_UNCHANGED = 5
    SHADOW_TAG_AND_DOMAIN_SEPARATION_PASS = 6
    FOUR_PURPOSE_LAUNCH_PLAN_EXACT = 7
    LOCAL_NAMESPACE_AND_SECCOMP_ISOLATION_AVAILABLE = 8
    SHADOW_FD_POLICY_VERIFIED = 9
    SHADOW_SECRET_NONPERSISTENCE_PLAN_VERIFIED = 10
    SYNTHESIS_BLINDNESS_ROUTE_VERIFIED = 11
    SHADOW_OUTPUT_AND_CLAIM_LINTER_PASS = 12


class ShadowProbePhaseId(IntEnum):
    ADMISSION_PROBE = 1
    START_RUNTIME_PROBE = 2


class ShadowLandlockStatusId(IntEnum):
    NOT_PROBED = 0
    ENFORCED = 1
    UNAVAILABLE = 2
    PARTIAL = 3


class ShadowAttackSyscallId(IntEnum):
    SOCKET_AF_INET_STREAM = 1
    SOCKET_AF_INET6_STREAM = 2
    MOUNT = 3
    PTRACE_TRACEME = 4
    BPF_MAP_CREATE = 5
    PERF_EVENT_OPEN = 6


class ShadowImplementationId(IntEnum):
    PYTHON = 1
    RUST = 2


class ShadowTargetRoleId(IntEnum):
    ODD_OUTSIDE_TARGET = 1
    SINK_NULL_CONTROL = 2


SHADOW_PURPOSE_IDS: Final = tuple(ShadowPurposeId(value) for value in range(1, 5))
SHADOW_TERMINAL_STATES: Final = frozenset(
    ShadowStateId(value) for value in range(4, 11)
)

SHADOW_GATE_FAILURES: Final = MappingProxyType(
    {
        ShadowAdmissionGateId.SHADOW_OWNER_POLICY_BOUND: "FAIL_SHADOW_POLICY_NOT_BOUND",
        ShadowAdmissionGateId.SHADOW_BASIS_COMMIT_PINNED: "FAIL_SHADOW_BASIS_COMMIT_MISMATCH",
        ShadowAdmissionGateId.SHADOW_READ_ONLY_SNAPSHOT_VERIFIED: "FAIL_SHADOW_SNAPSHOT_NOT_READ_ONLY",
        ShadowAdmissionGateId.SHADOW_DETERMINISTIC_DUAL_BASELINE_PASS: "FAIL_SHADOW_BASELINE_DUAL_MISMATCH",
        ShadowAdmissionGateId.FORMAL_TRACK_INVARIANTS_UNCHANGED: "FAIL_SHADOW_FORMAL_STATE_MUTATION",
        ShadowAdmissionGateId.SHADOW_TAG_AND_DOMAIN_SEPARATION_PASS: "FAIL_SHADOW_DOMAIN_COLLISION",
        ShadowAdmissionGateId.FOUR_PURPOSE_LAUNCH_PLAN_EXACT: "FAIL_SHADOW_PURPOSE_SET",
        ShadowAdmissionGateId.LOCAL_NAMESPACE_AND_SECCOMP_ISOLATION_AVAILABLE: "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
        ShadowAdmissionGateId.SHADOW_FD_POLICY_VERIFIED: "FAIL_SHADOW_SECRET_CHANNEL_POLICY",
        ShadowAdmissionGateId.SHADOW_SECRET_NONPERSISTENCE_PLAN_VERIFIED: "FAIL_SHADOW_SECRET_PERSISTENCE_POLICY",
        ShadowAdmissionGateId.SYNTHESIS_BLINDNESS_ROUTE_VERIFIED: "FAIL_SHADOW_SYNTHESIS_BLINDNESS",
        ShadowAdmissionGateId.SHADOW_OUTPUT_AND_CLAIM_LINTER_PASS: "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED",
    }
)


FORMAL_TRACK_SNAPSHOT: Final = MappingProxyType(
    {
        "freeze_id": FORMAL_MACHINE_FREEZE_ID,
        "child_dsl_id": FORMAL_CHILD_DSL_ID,
        "gates_satisfied": 14,
        "gates_total": 24,
        "m3_state_id": 0,
        "m3_state_name": "NOT_RUN",
        "m3_entry_qualified": False,
        "m3_entry_allowed": False,
        "m3_run_started": False,
        "formal_roots": None,
        "formal_seed_first_instantiated": False,
        "external_actor_evidence": False,
        "outside_certificate_allowed": False,
        "mdl_certificate_allowed": False,
        "active_promotion_allowed": False,
    }
)


SHADOW_HASH_DOMAINS: Final = MappingProxyType(
    {
        "POLICY_BINDING": "ShadowPolicyBindingV1",
        "PURPOSE_WORKER": "ShadowPurposeWorkerManifestV1",
        "ISOLATION_MANIFEST": "ShadowIsolationManifestV1",
        "ADMISSION_RECEIPT": "ShadowAdmissionReceiptV1",
        "ENVELOPE": "ShadowEnvelopeV1",
        "RUN_GENESIS": "ShadowRunGenesisV1",
        "STATE_RECORD": "ShadowStateRecordV1",
        "ENUMERATION_RECEIPT": "ShadowEnumerationReceiptV1",
        "ROLE_EVALUATION_RECEIPT": "ShadowRoleEvaluationReceiptV1",
        "DUAL_REPLAY_AGREEMENT": "ShadowDualReplayAgreementV1",
        "DISCLOSURE_LEDGER_RECORD": "ShadowDisclosureLedgerRecordV1",
        "EXECUTION_BUNDLE": "ShadowExecutionBundleV1",
        "ISOLATION_PLAN": "ShadowIsolationPlanV1",
        "SECURITY_PROBE_RECEIPT": "ShadowSecurityProbeReceiptV1",
    }
)


SHADOW_OBJECT_TAGS: Final = MappingProxyType(
    {
        "ShadowPolicyBindingV1": 0x7A00,
        "ShadowPurposeWorkerManifestV1": 0x7A01,
        "ShadowIsolationManifestV1": 0x7A02,
        "ShadowAdmissionReceiptV1": 0x7A03,
        "ShadowEnvelopeV1": 0x7A04,
        "ShadowRunGenesisV1": 0x7A05,
        "ShadowStateRecordV1": 0x7A06,
        "ShadowEnumerationReceiptV1": 0x7A07,
        "ShadowRoleEvaluationReceiptV1": 0x7A08,
        "ShadowDualReplayAgreementV1": 0x7A09,
        "ShadowDisclosureLedgerRecordV1": 0x7A0A,
        "ShadowExecutionBundleV1": 0x7A0B,
        "ShadowIsolationPlanV1": 0x7A0C,
        "ShadowSecurityProbeReceiptV1": 0x7A0D,
    }
)


@dataclass(frozen=True)
class ShadowSchema:
    name: str
    tag: int
    schema_id: bytes
    fields: tuple[str, ...]
    digest_domain: str

    @property
    def prefix(self) -> tuple[int, int, bytes]:
        return (1, self.tag, self.schema_id)


def _schema(
    name: str,
    tag: int,
    schema_id: str,
    fields: tuple[str, ...],
    digest_domain: str,
) -> ShadowSchema:
    return ShadowSchema(name, tag, schema_id.encode("ascii"), fields, digest_domain)


_GENESIS_OUTPUT_FIELDS: Final = (
    "canonical_program_archive_digest_or_null",
    "program_chunk_manifest_digest_or_null",
    "bucket_accounting_digest_or_null",
    "odd_output_archive_digest_or_null",
    "odd_match_set_digest_or_null",
    "odd_role_receipt_digest_or_null",
    "sink_output_archive_digest_or_null",
    "sink_match_set_digest_or_null",
    "sink_role_receipt_digest_or_null",
    "dual_replay_agreement_digest_or_null",
)


_SCHEMAS: Final = (
    _schema(
        "ShadowPolicyBindingV1",
        0x7A00,
        "hegel-internal-shadow-policy-binding/1",
        (
            "artifact_kind_id",
            "shadow_track_id",
            "formal_machine_freeze_id",
            "formal_child_dsl_id",
            "amendment_git_blob_sha256",
            "basis_commit_id",
        ),
        "POLICY_BINDING",
    ),
    _schema(
        "ShadowPurposeWorkerManifestV1",
        0x7A01,
        "hegel-internal-shadow-purpose-worker/1",
        (
            "artifact_kind_id", "shadow_run_id", "shadow_purpose_id",
            "worker_instance_id", "isolation_profile_id", "basis_commit_id",
            "snapshot_manifest_digest", "executable_manifest_digest",
            "environment_manifest_digest", "namespace_manifest_digest",
            "ephemeral_key_id", "ephemeral_public_key", "key_epoch",
            "external_independence_claim",
        ),
        "PURPOSE_WORKER",
    ),
    _schema(
        "ShadowIsolationManifestV1",
        0x7A02,
        "hegel-internal-shadow-isolation-manifest/1",
        (
            "artifact_kind_id", "shadow_run_id", "basis_commit_id",
            "snapshot_manifest_digest", "purpose_worker_digests",
            "isolation_invariant_bitset", "required_security_probe_digest",
            "fd_policy_digest", "output_allowlist_digest",
            "secret_lint_policy_digest", "created_at_unix_seconds",
            "external_independence_claim",
        ),
        "ISOLATION_MANIFEST",
    ),
    _schema(
        "ShadowAdmissionReceiptV1",
        0x7A03,
        "hegel-internal-shadow-admission-receipt/1",
        (
            "artifact_kind_id", "shadow_run_id", "policy_binding_digest",
            "isolation_plan_digest", "basis_commit_id", "shadow_gate_bitset",
            "shadow_gate_count", "formal_gates_satisfied", "formal_gates_total",
            "formal_m3_state_id", "formal_roots_all_null",
            "external_actor_evidence", "admitted_at_unix_seconds",
        ),
        "ADMISSION_RECEIPT",
    ),
    _schema(
        "ShadowEnvelopeV1",
        0x7A04,
        "hegel-internal-shadow-envelope/1",
        (
            "artifact_kind_id", "shadow_run_id", "enclosed_shadow_object_digest",
            "signer_purpose_id", "signer_key_id", "key_epoch",
            "signature_64_bytes", "external_independence_claim",
        ),
        "ENVELOPE",
    ),
    _schema(
        "ShadowRunGenesisV1",
        0x7A05,
        "hegel-internal-shadow-run-genesis/1",
        (
            "artifact_kind_id", "shadow_run_id", "policy_binding_digest",
            "admission_receipt_digest", "isolation_manifest_digest",
            "basis_commit_id", "initial_shadow_state_id",
        ) + _GENESIS_OUTPUT_FIELDS + (
            "created_at_unix_seconds", "formal_run_genesis_claim",
        ),
        "RUN_GENESIS",
    ),
    _schema(
        "ShadowStateRecordV1",
        0x7A06,
        "hegel-internal-shadow-state-record/1",
        (
            "artifact_kind_id", "shadow_run_id", "transition_index",
            "previous_state_record_digest_or_null", "from_shadow_state_id",
            "to_shadow_state_id", "transition_reason_id",
            "triggering_shadow_receipt_digest_or_null", "recorded_at_unix_seconds",
            "formal_gates_satisfied", "formal_gates_total", "formal_m3_state_id",
        ),
        "STATE_RECORD",
    ),
    _schema(
        "ShadowIsolationPlanV1",
        0x7A0C,
        "hegel-internal-shadow-isolation-plan/1",
        (
            "artifact_kind_id", "shadow_run_id", "basis_commit_id",
            "snapshot_manifest_digest", "purpose_ids", "isolation_profile_id",
            "worker_launch_plan_digest", "required_security_probe_digest",
            "fd_policy_digest", "output_allowlist_digest",
            "secret_lint_policy_digest", "external_independence_claim",
        ),
        "ISOLATION_PLAN",
    ),
    _schema(
        "ShadowSecurityProbeReceiptV1",
        0x7A0D,
        "hegel-internal-shadow-security-probe-receipt/1",
        (
            "artifact_kind_id", "shadow_run_id", "shadow_purpose_id",
            "probe_phase_id", "worker_instance_id", "basis_commit_id",
            "proc_status_seccomp_value", "proc_status_no_new_privs_value",
            "attack_syscall_errno_rows", "landlock_status_id",
            "landlock_nonblocking_gap_disclosed",
            "transient_capability_probe_incident_count",
            "transient_capability_probe_incident_digest_or_null",
            "observed_at_unix_seconds", "external_security_attestation_claim",
        ),
        "SECURITY_PROBE_RECEIPT",
    ),
)


SHADOW_SCHEMA_REGISTRY: Final = MappingProxyType(
    {schema.name: schema for schema in _SCHEMAS}
)
_SCHEMA_BY_IDENTITY: Final = MappingProxyType(
    {(schema.tag, schema.schema_id): schema for schema in _SCHEMAS}
)


LEGAL_SHADOW_TRANSITIONS: Final = frozenset(
    {
        (ShadowStateId(0), ShadowStateId(1), ShadowTransitionReasonId(1)),
        (ShadowStateId(1), ShadowStateId(2), ShadowTransitionReasonId(2)),
        (ShadowStateId(1), ShadowStateId(8), ShadowTransitionReasonId(7)),
        (ShadowStateId(1), ShadowStateId(9), ShadowTransitionReasonId(9)),
        (ShadowStateId(1), ShadowStateId(10), ShadowTransitionReasonId(10)),
        (ShadowStateId(2), ShadowStateId(3), ShadowTransitionReasonId(3)),
        (ShadowStateId(2), ShadowStateId(5), ShadowTransitionReasonId(4)),
        (ShadowStateId(2), ShadowStateId(6), ShadowTransitionReasonId(5)),
        (ShadowStateId(2), ShadowStateId(7), ShadowTransitionReasonId(6)),
        (ShadowStateId(2), ShadowStateId(8), ShadowTransitionReasonId(7)),
        (ShadowStateId(2), ShadowStateId(9), ShadowTransitionReasonId(9)),
        (ShadowStateId(2), ShadowStateId(10), ShadowTransitionReasonId(10)),
        (ShadowStateId(3), ShadowStateId(4), ShadowTransitionReasonId(8)),
        (ShadowStateId(3), ShadowStateId(7), ShadowTransitionReasonId(6)),
        (ShadowStateId(3), ShadowStateId(8), ShadowTransitionReasonId(7)),
        (ShadowStateId(3), ShadowStateId(9), ShadowTransitionReasonId(9)),
        (ShadowStateId(3), ShadowStateId(10), ShadowTransitionReasonId(10)),
    }
)


def _require_uint(value: object, field: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        _fail("REJECT_SHADOW_FIELD_TYPE", f"{field} must be a CBOR uint64")
    return value


def _require_bytes(value: object, length: int, field: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_SHADOW_FIELD_TYPE", f"{field} must be exactly {length} bytes")
    return value


def _require_opaque_id(value: object, field: str) -> bytes:
    result = _require_bytes(value, 16, field)
    if result == bytes(16):
        _fail("REJECT_SHADOW_FIELD_VALUE", f"{field} may not be all zero")
    return result


def git_sha1_commit_id(raw_digest: bytes) -> tuple[int, bytes]:
    """Return the shadow wire's explicit Git SHA-1 commit identity."""

    _require_bytes(raw_digest, 20, "basis_commit_id")
    return (1, raw_digest)


def _validate_git_commit(value: object, field: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        _fail("REJECT_SHADOW_GIT_COMMIT_ID", f"{field} must be [1, sha1_20_bytes]")
    if type(value[0]) is not int or value[0] != 1:
        _fail("REJECT_SHADOW_GIT_COMMIT_ID", f"{field} algorithm must be SHA-1 ID 1")
    _require_bytes(value[1], 20, field)


def _enum_value(enum_type: type[IntEnum], value: object, field: str) -> IntEnum:
    if type(value) is not int:
        _fail("REJECT_SHADOW_FIELD_TYPE", f"{field} must be a numeric enum")
    try:
        return enum_type(value)
    except ValueError:
        _fail("REJECT_SHADOW_ENUM_VALUE", f"{field} uses an unknown or reserved value")


def _validate_artifact_kind(value: object) -> None:
    if type(value) is not int or value != SHADOW_ARTIFACT_KIND_ID:
        _fail("FAIL_SHADOW_ARTIFACT_KIND", "shadow artifact kind must be numeric ID 1")


def _validate_false_claim(value: object, field: str) -> None:
    if value is not False:
        _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", f"{field} must be strict false")


def _validate_field(field: str, value: object) -> None:
    if field == "artifact_kind_id":
        _validate_artifact_kind(value)
        return
    if field in {"shadow_track_id", "formal_machine_freeze_id", "formal_child_dsl_id"}:
        if type(value) is not bytes:
            _fail("REJECT_SHADOW_FIELD_TYPE", f"{field} must be ASCII bytes")
        return
    if field == "basis_commit_id":
        _validate_git_commit(value, field)
        return
    if field in {"shadow_run_id", "worker_instance_id", "ephemeral_key_id", "signer_key_id"}:
        _require_opaque_id(value, field)
        return
    if field == "ephemeral_public_key":
        _require_bytes(value, 32, field)
        return
    if field == "signature_64_bytes":
        _require_bytes(value, 64, field)
        return
    if field == "purpose_worker_digests":
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            _fail("FAIL_SHADOW_PURPOSE_SET", "purpose_worker_digests must have four entries")
        normalized = tuple(_require_bytes(item, 32, field) for item in value)
        if len(set(normalized)) != 4:
            _fail("FAIL_SHADOW_PROCESS_REUSE", "purpose worker digests must be distinct")
        return
    if field == "purpose_ids":
        if (
            not isinstance(value, (tuple, list))
            or any(type(item) is not int for item in value)
            or tuple(value) != (1, 2, 3, 4)
        ):
            _fail("FAIL_SHADOW_PURPOSE_SET", "purpose_ids must be exactly [1,2,3,4]")
        return
    if field == "attack_syscall_errno_rows":
        expected = tuple((value, 1) for value in range(1, 7))
        if not isinstance(value, (tuple, list)):
            _fail("FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE", "syscall probe rows must be an array")
        normalized = tuple(
            tuple(row)
            if (
                isinstance(row, (tuple, list))
                and len(row) == 2
                and all(type(item) is int for item in row)
            )
            else ()
            for row in value
        )
        if normalized != expected:
            _fail(
                "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
                "all six ordered syscall probes must return EPERM",
            )
        return
    if field.endswith("_digest") or field == "amendment_git_blob_sha256":
        _require_bytes(value, 32, field)
        return
    if field.endswith("_digest_or_null"):
        if value is not None:
            _require_bytes(value, 32, field)
        return
    if field in {
        "external_independence_claim", "external_actor_evidence",
        "formal_run_genesis_claim", "external_security_attestation_claim",
    }:
        _validate_false_claim(value, field)
        return
    if field in {"formal_roots_all_null", "landlock_nonblocking_gap_disclosed"}:
        if type(value) is not bool:
            _fail("REJECT_SHADOW_FIELD_TYPE", f"{field} must be a strict boolean")
        return
    enum_fields: Mapping[str, type[IntEnum]] = {
        "shadow_purpose_id": ShadowPurposeId,
        "signer_purpose_id": ShadowPurposeId,
        "from_shadow_state_id": ShadowStateId,
        "to_shadow_state_id": ShadowStateId,
        "initial_shadow_state_id": ShadowStateId,
        "transition_reason_id": ShadowTransitionReasonId,
        "probe_phase_id": ShadowProbePhaseId,
        "landlock_status_id": ShadowLandlockStatusId,
    }
    enum_type = enum_fields.get(field)
    if enum_type is not None:
        normalized = _enum_value(enum_type, value, field)
        if normalized.value == 0 and field not in {"from_shadow_state_id", "landlock_status_id"}:
            _fail("REJECT_SHADOW_ENUM_VALUE", f"{field} may not use INVALID/NOT_ADMITTED")
        return
    if field in {
        "isolation_profile_id", "key_epoch", "isolation_invariant_bitset",
        "shadow_gate_bitset", "shadow_gate_count", "formal_gates_satisfied",
        "formal_gates_total", "formal_m3_state_id", "transition_index",
        "created_at_unix_seconds", "admitted_at_unix_seconds",
        "recorded_at_unix_seconds", "observed_at_unix_seconds",
        "proc_status_seccomp_value", "proc_status_no_new_privs_value",
        "transient_capability_probe_incident_count",
    }:
        _require_uint(value, field)
        return
    _fail("REJECT_SHADOW_FIELD_TYPE", f"no exact wire type is registered for {field}")


def _validate_formal_wire_snapshot_fields(fields: Mapping[str, object]) -> None:
    if (
        fields.get("formal_gates_satisfied") != 14
        or fields.get("formal_gates_total") != 24
        or fields.get("formal_m3_state_id") != 0
    ):
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "formal state must remain 14/24 / NOT_RUN")


def _validate_cross_field_guards(name: str, fields: Mapping[str, object]) -> None:
    if fields["artifact_kind_id"] != SHADOW_ARTIFACT_KIND_ID:
        _fail("FAIL_SHADOW_ARTIFACT_KIND", "artifact kind is not the frozen shadow kind")

    if name == "ShadowPolicyBindingV1":
        exact = {
            "shadow_track_id": SHADOW_TRACK_ID.encode("ascii"),
            "formal_machine_freeze_id": FORMAL_MACHINE_FREEZE_ID.encode("ascii"),
            "formal_child_dsl_id": FORMAL_CHILD_DSL_ID.encode("ascii"),
        }
        if any(fields[field] != value for field, value in exact.items()):
            _fail("FAIL_SHADOW_POLICY_NOT_BOUND", "policy machine IDs differ from the amendment")
    elif name == "ShadowPurposeWorkerManifestV1":
        if fields["isolation_profile_id"] != SHADOW_ISOLATION_PROFILE_ID:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "worker isolation profile is not v1")
        if fields["key_epoch"] != 0:
            _fail("FAIL_SHADOW_KEY_REUSE", "ephemeral shadow key epoch must be zero")
    elif name == "ShadowIsolationManifestV1":
        if fields["isolation_invariant_bitset"] != SHADOW_ISOLATION_INVARIANT_BITSET:
            _fail(
                "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
                "all and only the 18 frozen isolation predicates must pass",
            )
    elif name == "ShadowAdmissionReceiptV1":
        validate_shadow_admission_bitset(
            fields["shadow_gate_bitset"], fields["shadow_gate_count"]
        )
        _validate_formal_wire_snapshot_fields(fields)
        if fields["formal_roots_all_null"] is not True:
            _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "formal roots must all remain null")
        if fields["external_actor_evidence"] is not False:
            _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", "shadow admission is not external evidence")
    elif name == "ShadowEnvelopeV1":
        if fields["key_epoch"] != 0:
            _fail("FAIL_SHADOW_KEY_REUSE", "ephemeral shadow key epoch must be zero")
    elif name == "ShadowRunGenesisV1":
        if fields["initial_shadow_state_id"] != ShadowStateId.ADMITTED_NOT_STARTED:
            _fail("FAIL_SHADOW_INVALID_TRANSITION", "genesis initial state must be 1")
        if any(fields[field] is not None for field in _GENESIS_OUTPUT_FIELDS):
            _fail("FAIL_SHADOW_FORBIDDEN_CLAIM", "all ten genesis output slots must be null")
    elif name == "ShadowStateRecordV1":
        _validate_formal_wire_snapshot_fields(fields)
        normalized = validate_shadow_state_transition(
            fields["from_shadow_state_id"],
            fields["to_shadow_state_id"],
            fields["transition_reason_id"],
        )
        transition_index = fields["transition_index"]
        previous = fields["previous_state_record_digest_or_null"]
        if transition_index == 0:
            exact = (
                ShadowStateId.NOT_ADMITTED,
                ShadowStateId.ADMITTED_NOT_STARTED,
                ShadowTransitionReasonId.SHADOW_ADMISSION_GATES_12_OF_12,
            )
            if normalized != exact or previous is not None:
                _fail("FAIL_SHADOW_INVALID_TRANSITION", "transition index 0 must be exact admission")
        elif previous is None:
            _fail("FAIL_SHADOW_INVALID_TRANSITION", "noninitial state record requires previous digest")
    elif name == "ShadowIsolationPlanV1":
        if fields["isolation_profile_id"] != SHADOW_ISOLATION_PROFILE_ID:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", "isolation plan profile is not v1")
    elif name == "ShadowSecurityProbeReceiptV1":
        if (
            fields["proc_status_seccomp_value"] != 2
            or fields["proc_status_no_new_privs_value"] != 1
        ):
            _fail(
                "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
                "security probe requires Seccomp=2 and NoNewPrivs=1",
            )
        landlock_enforced = fields["landlock_status_id"] == ShadowLandlockStatusId.ENFORCED
        if fields["landlock_nonblocking_gap_disclosed"] is landlock_enforced:
            _fail(
                "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED",
                "Landlock nonblocking gap disclosure does not match enforcement status",
            )
        count = fields["transient_capability_probe_incident_count"]
        incident_digest = fields["transient_capability_probe_incident_digest_or_null"]
        if (count == 0) != (incident_digest is None):
            _fail(
                "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED",
                "incident count zero iff incident digest is null",
            )


def build_shadow_object(name: str, fields: Mapping[str, object]) -> tuple[object, ...]:
    """Build a strict shadow numeric-array object from named construction fields."""

    schema = SHADOW_SCHEMA_REGISTRY.get(name)
    if schema is None:
        _fail("REJECT_UNKNOWN_SHADOW_SCHEMA", f"unknown shadow object {name!r}")
    if not isinstance(fields, Mapping):
        raise TypeError("shadow object fields must be a mapping")
    expected = set(schema.fields)
    actual = set(fields)
    if expected != actual:
        _fail(
            "REJECT_SHADOW_FIELD_SET",
            f"{name} field mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}",
        )
    ordered = tuple(fields[field] for field in schema.fields)
    for field, value in zip(schema.fields, ordered, strict=True):
        _validate_field(field, value)
    _validate_cross_field_guards(name, fields)
    result = schema.prefix + ordered
    canonical_cbor_encode(result)
    return result


def encode_shadow_object(name: str, fields: Mapping[str, object]) -> bytes:
    return canonical_cbor_encode(build_shadow_object(name, fields))


@dataclass(frozen=True)
class DecodedShadowObject:
    schema: ShadowSchema
    fields: Mapping[str, StrictCborValue]
    value: tuple[StrictCborValue, ...]


def decode_shadow_object(
    payload: bytes,
    *,
    expected_name: str | None = None,
) -> DecodedShadowObject:
    """Strictly decode one shadow object and reject the formal tag namespace."""

    value = canonical_cbor_decode(payload)
    if not isinstance(value, tuple) or len(value) < 3:
        _fail("REJECT_SHADOW_OBJECT_PREFIX", "shadow object must be a numeric array")
    if type(value[0]) is not int or value[0] != 1:
        _fail("REJECT_SHADOW_OBJECT_PREFIX", "shadow schema version must equal one")
    if type(value[1]) is not int or type(value[2]) is not bytes:
        _fail("REJECT_SHADOW_OBJECT_PREFIX", "shadow tag/schema ID types are invalid")
    if 0x3000 <= value[1] <= 0x34FF:
        _fail(
            "REJECT_SHADOW_FORMAL_TAG_NAMESPACE",
            "formal M2.5/M3 tags are never accepted by the shadow decoder",
        )
    schema = _SCHEMA_BY_IDENTITY.get((value[1], value[2]))
    if schema is None:
        _fail("REJECT_UNKNOWN_SHADOW_SCHEMA", "unknown shadow tag/schema-ID pair")
    if expected_name is not None and schema.name != expected_name:
        _fail(
            "REJECT_SHADOW_SCHEMA_MISMATCH",
            f"expected {expected_name}, decoded {schema.name}",
        )
    if len(value) != 3 + len(schema.fields):
        _fail("REJECT_SHADOW_FIELD_SET", f"wrong array length for {schema.name}")
    decoded_fields = dict(zip(schema.fields, value[3:], strict=True))
    for field, field_value in decoded_fields.items():
        _validate_field(field, field_value)
    _validate_cross_field_guards(schema.name, decoded_fields)
    return DecodedShadowObject(
        schema=schema,
        fields=MappingProxyType(decoded_fields),
        value=value,
    )


def shadow_digest_v1(domain: str, value: object) -> bytes:
    """Compute a closed-registry ``ShadowDigestV1`` identity."""

    if type(domain) is not str or domain not in SHADOW_HASH_DOMAINS:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "unknown, non-ASCII, or nonexact shadow domain")
    try:
        domain_bytes = domain.encode("ascii")
    except UnicodeEncodeError:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "shadow domain must be exact ASCII")
    return sha256(
        _SHADOW_DIGEST_PREFIX
        + domain_bytes
        + b"/V1\x00"
        + canonical_cbor_encode(value)
    ).digest()


def shadow_object_digest(name: str, fields: Mapping[str, object]) -> bytes:
    schema = SHADOW_SCHEMA_REGISTRY.get(name)
    if schema is None:
        _fail("REJECT_UNKNOWN_SHADOW_SCHEMA", f"unknown shadow object {name!r}")
    return shadow_digest_v1(schema.digest_domain, build_shadow_object(name, fields))


def shadow_purpose_worker_digest_set_v1(
    manifests: Sequence[Mapping[str, object]],
) -> tuple[bytes, bytes, bytes, bytes]:
    """Validate the ordered four-purpose worker set and return its digests."""

    if not isinstance(manifests, (tuple, list)) or len(manifests) != 4:
        _fail("FAIL_SHADOW_PURPOSE_SET", "worker manifest set must contain four rows")
    built = [
        build_shadow_object("ShadowPurposeWorkerManifestV1", fields)
        for fields in manifests
    ]
    purposes = tuple(row[5] for row in built)
    if purposes != (1, 2, 3, 4):
        _fail("FAIL_SHADOW_PURPOSE_SET", "worker manifests must be ordered by purpose 1..4")
    # Prefix positions are fixed by the numeric-array schema.
    for position, label in ((4, "shadow run"), (8, "basis commit"), (9, "snapshot")):
        if len({canonical_cbor_encode(row[position]) for row in built}) != 1:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", f"workers disagree on {label}")
    for position, label in (
        (6, "worker instance"),
        (13, "ephemeral key ID"),
        (14, "ephemeral public key"),
    ):
        if len({row[position] for row in built}) != 4:
            code = "FAIL_SHADOW_KEY_REUSE" if position in {13, 14} else "FAIL_SHADOW_PROCESS_REUSE"
            _fail(code, f"workers reuse a {label}")
    result = tuple(
        shadow_digest_v1("PURPOSE_WORKER", row) for row in built
    )
    assert len(result) == 4
    return (result[0], result[1], result[2], result[3])


def shadow_security_probe_set_digest_v1(
    receipts: Sequence[Mapping[str, object]],
    *,
    expected_phase: int | ShadowProbePhaseId,
) -> bytes:
    """Validate and hash the ordered four-purpose mandatory live-probe set."""

    raw_phase = expected_phase.value if isinstance(expected_phase, ShadowProbePhaseId) else expected_phase
    phase = _enum_value(ShadowProbePhaseId, raw_phase, "expected_phase")
    if not isinstance(receipts, (tuple, list)) or len(receipts) != 4:
        _fail(
            "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
            "security probe set must contain four purpose rows",
        )
    built = [
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
        for fields in receipts
    ]
    if tuple(row[5] for row in built) != (1, 2, 3, 4):
        _fail("FAIL_SHADOW_PURPOSE_SET", "security probes must be ordered by purpose 1..4")
    if any(row[6] != phase.value for row in built):
        _fail(
            "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE",
            "security probe phase differs from the required ceremony phase",
        )
    for position, label in ((4, "shadow run"), (8, "basis commit")):
        if len({canonical_cbor_encode(row[position]) for row in built}) != 1:
            _fail("FAIL_SHADOW_RUNTIME_PLAN_MISMATCH", f"security probes disagree on {label}")
    if len({row[7] for row in built}) != 4:
        _fail("FAIL_SHADOW_PROCESS_REUSE", "security probes reuse a worker instance")
    incident_bindings = {
        (row[14], row[15])
        for row in built
    }
    if len(incident_bindings) != 1:
        _fail(
            "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED",
            "security probes disagree on the diagnostic incident collection",
        )
    return shadow_tree_digest_v1("SECURITY_PROBE_SET", built)


def _tree_prefix(domain: str) -> bytes:
    if type(domain) is not str or _TREE_DOMAIN_RE.fullmatch(domain) is None:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "tree domain must be an exact uppercase role token")
    if domain in SHADOW_HASH_DOMAINS or "FORMAL" in domain or "RFC6962" in domain:
        _fail("FAIL_SHADOW_DOMAIN_COLLISION", "tree domain collides with a non-tree identity class")
    return _SHADOW_TREE_PREFIX + domain.encode("ascii") + b"/V1\x00"


def shadow_tree_digest_v1(domain: str, records: Sequence[object]) -> bytes:
    """Hash an ordered shadow collection with its non-RFC6962 preimages."""

    if not isinstance(records, (tuple, list)):
        raise TypeError("shadow tree records must be a list or tuple")
    prefix = _tree_prefix(domain)
    normalized = tuple(records)
    if not normalized:
        return sha256(prefix + b"\x02").digest()
    if len(normalized) == 1:
        return sha256(prefix + b"\x00" + canonical_cbor_encode(normalized[0])).digest()
    split = 1 << ((len(normalized) - 1).bit_length() - 1)
    left = shadow_tree_digest_v1(domain, normalized[:split])
    right = shadow_tree_digest_v1(domain, normalized[split:])
    return sha256(prefix + b"\x01" + left + right).digest()


def shadow_signature_preimage_v1(
    shadow_object_digest_32_bytes: bytes,
    shadow_purpose_id: int | ShadowPurposeId,
    key_epoch: int,
    shadow_run_id: bytes,
) -> bytes:
    """Build the exact temporary-signature preimage without signing it."""

    digest = _require_bytes(
        shadow_object_digest_32_bytes, 32, "shadow_object_digest_32_bytes"
    )
    raw_purpose = (
        shadow_purpose_id.value
        if isinstance(shadow_purpose_id, ShadowPurposeId)
        else shadow_purpose_id
    )
    purpose = _enum_value(ShadowPurposeId, raw_purpose, "shadow_purpose_id")
    if purpose is ShadowPurposeId.INVALID:
        _fail("REJECT_SHADOW_ENUM_VALUE", "INVALID purpose cannot sign")
    if type(key_epoch) is not int or key_epoch != 0:
        _fail("FAIL_SHADOW_KEY_REUSE", "shadow key epoch must be zero")
    run_id = _require_opaque_id(shadow_run_id, "shadow_run_id")
    return (
        SHADOW_SIGNATURE_PREFIX
        + b"\x00"
        + digest
        + purpose.value.to_bytes(2, "big")
        + key_epoch.to_bytes(8, "big")
        + run_id
    )


def validate_shadow_artifact_header(
    *,
    artifact_kind_id: object,
    artifact_kind: object,
    external_independence_claim: object,
    formal_evidence_claim: object,
) -> None:
    """Validate the mandatory public JSON identity/claim quartet."""

    _validate_artifact_kind(artifact_kind_id)
    if artifact_kind != SHADOW_ARTIFACT_KIND:
        _fail("FAIL_SHADOW_ARTIFACT_KIND", "artifact kind string/ID mismatch")
    _validate_false_claim(external_independence_claim, "external_independence_claim")
    _validate_false_claim(formal_evidence_claim, "formal_evidence_claim")


def validate_formal_track_snapshot(snapshot: Mapping[str, object]) -> None:
    """Require the complete, exact formal ``14/24 / NOT_RUN`` snapshot."""

    if not isinstance(snapshot, Mapping):
        _fail("FAIL_SHADOW_FORMAL_STATUS_OMITTED", "formal track snapshot is absent")
    missing = set(FORMAL_TRACK_SNAPSHOT) - set(snapshot)
    if missing:
        _fail(
            "FAIL_SHADOW_FORMAL_STATUS_OMITTED",
            f"formal snapshot omits {sorted(missing)}",
        )
    extra = set(snapshot) - set(FORMAL_TRACK_SNAPSHOT)
    if extra or any(snapshot[key] != value for key, value in FORMAL_TRACK_SNAPSHOT.items()):
        _fail("FAIL_SHADOW_FORMAL_STATE_MUTATION", "formal snapshot differs from the frozen invariant")


def shadow_gate_bitset(
    gate_results: Mapping[ShadowAdmissionGateId | int | str, object],
) -> int:
    """Encode exactly twelve named gate booleans, without requiring all pass."""

    if not isinstance(gate_results, Mapping):
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "gate results must be a mapping")
    normalized: dict[ShadowAdmissionGateId, bool] = {}
    for raw_gate, passed in gate_results.items():
        try:
            if type(raw_gate) is str:
                gate = ShadowAdmissionGateId[raw_gate]
            elif isinstance(raw_gate, ShadowAdmissionGateId):
                gate = raw_gate
            elif type(raw_gate) is int:
                gate = ShadowAdmissionGateId(raw_gate)
            else:
                raise TypeError("gate ID must be exact int, enum, or name")
        except (KeyError, TypeError, ValueError):
            _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", f"unknown shadow gate {raw_gate!r}")
        if type(passed) is not bool:
            _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", f"gate {gate.name} is not strict boolean")
        if gate in normalized:
            _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", f"duplicate shadow gate {gate.name}")
        normalized[gate] = passed
    expected = set(ShadowAdmissionGateId)
    if set(normalized) != expected:
        missing = sorted(gate.name for gate in expected - set(normalized))
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", f"gate set is incomplete: {missing}")
    result = 0
    for gate, passed in normalized.items():
        if passed:
            result |= 1 << (gate.value - 1)
    return result


def require_shadow_admission(
    gate_results: Mapping[ShadowAdmissionGateId | int | str, object],
) -> int:
    """Require 12/12 and raise the frozen first failing gate code."""

    bitset = shadow_gate_bitset(gate_results)
    if bitset != SHADOW_ALL_GATES_BITSET:
        for gate in ShadowAdmissionGateId:
            if not bitset & (1 << (gate.value - 1)):
                _fail(SHADOW_GATE_FAILURES[gate], f"shadow gate {gate.value} {gate.name} failed")
    return bitset


def validate_shadow_admission_bitset(bitset: object, count: object) -> None:
    if type(bitset) is not int or type(count) is not int:
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "admission bitset/count must be integers")
    if bitset != SHADOW_ALL_GATES_BITSET or count != SHADOW_GATE_COUNT:
        _fail("FAIL_SHADOW_ADMISSION_INCOMPLETE", "shadow admission requires exact 0x0FFF and 12")


def validate_shadow_state_transition(
    from_state: int | ShadowStateId,
    to_state: int | ShadowStateId,
    reason: int | ShadowTransitionReasonId,
) -> tuple[ShadowStateId, ShadowStateId, ShadowTransitionReasonId]:
    """Validate the disjoint shadow state graph."""

    raw_values = (from_state, to_state, reason)
    if any(type(value) is bool for value in raw_values):
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "boolean is not a shadow numeric enum")
    try:
        normalized = (
            ShadowStateId(from_state),
            ShadowStateId(to_state),
            ShadowTransitionReasonId(reason),
        )
    except (TypeError, ValueError) as exc:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", f"unknown shadow state/reason: {exc}")
    if normalized[0] in SHADOW_TERMINAL_STATES:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "terminal shadow state has no outgoing edge")
    if normalized not in LEGAL_SHADOW_TRANSITIONS:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", f"illegal shadow transition {normalized!r}")
    return normalized


def validate_shadow_state_chain_link(
    previous_state_record_digest: bytes | None,
    expected_previous_state_record_digest: bytes | None,
) -> None:
    for field, value in (
        ("previous_state_record_digest", previous_state_record_digest),
        ("expected_previous_state_record_digest", expected_previous_state_record_digest),
    ):
        if value is not None:
            _require_bytes(value, 32, field)
    if previous_state_record_digest != expected_previous_state_record_digest:
        _fail("FAIL_SHADOW_INVALID_TRANSITION", "shadow state chain digest mismatch")


__all__ = [
    "DecodedShadowObject",
    "FORMAL_CHILD_DSL_ID",
    "FORMAL_MACHINE_FREEZE_ID",
    "FORMAL_TRACK_SNAPSHOT",
    "LEGAL_SHADOW_TRANSITIONS",
    "SHADOW_ALL_GATES_BITSET",
    "SHADOW_ARTIFACT_KIND",
    "SHADOW_ARTIFACT_KIND_ID",
    "SHADOW_GATE_COUNT",
    "SHADOW_GATE_FAILURES",
    "SHADOW_HASH_DOMAINS",
    "SHADOW_ISOLATION_INVARIANT_BITSET",
    "SHADOW_ISOLATION_PROFILE_ID",
    "SHADOW_OBJECT_TAGS",
    "SHADOW_PURPOSE_IDS",
    "SHADOW_SCHEMA_REGISTRY",
    "SHADOW_SIGNATURE_PREFIX",
    "SHADOW_TERMINAL_STATES",
    "SHADOW_TRACK_ID",
    "ShadowAdmissionGateId",
    "ShadowArtifactKindId",
    "ShadowAttackSyscallId",
    "ShadowDisclosureEventTypeId",
    "ShadowImplementationId",
    "ShadowLandlockStatusId",
    "ShadowOutcomeId",
    "ShadowProbePhaseId",
    "ShadowPurposeId",
    "ShadowSchema",
    "ShadowStateId",
    "ShadowTargetRoleId",
    "ShadowTransitionReasonId",
    "ShadowWireError",
    "build_shadow_object",
    "decode_shadow_object",
    "encode_shadow_object",
    "git_sha1_commit_id",
    "require_shadow_admission",
    "shadow_digest_v1",
    "shadow_gate_bitset",
    "shadow_object_digest",
    "shadow_purpose_worker_digest_set_v1",
    "shadow_security_probe_set_digest_v1",
    "shadow_signature_preimage_v1",
    "shadow_tree_digest_v1",
    "validate_formal_track_snapshot",
    "validate_shadow_admission_bitset",
    "validate_shadow_artifact_header",
    "validate_shadow_state_chain_link",
    "validate_shadow_state_transition",
]

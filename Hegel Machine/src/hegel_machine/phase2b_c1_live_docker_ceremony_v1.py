"""Owner-accepted local Docker bootstrap for the Phase-2B C1 six-role graph.

The single public executor in this module provisions six purpose-separated
containers and their private Docker volumes, proves that all six actors are
simultaneously live and locally qualified, generates one fresh Ed25519 key in
each actor, and materializes the registry/STORE/TIME genesis graph verified by
``phase2b_c1_six_role_wire_v1``.  Private keys never leave their actor volume.

This is deliberately same-admin, local-Docker mechanics.  It is not external
organizational authority, malicious-host protection, an actual-960 run, a
recognizer operation, scoring, a formal gate, effect evidence, or C1 exit
evidence.  This slice also does not execute the reserved ``/cas`` persistence
surface or verify crash recovery.  Successful calls retain opaque
purpose-private volume locators so the later bounded ceremony can use the same
keys and durable STORE/TIME state; containers are always removed and
independently proved absent first.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Final, Iterator, Mapping, NoReturn, Sequence

from .phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
    load_phase2b_c1_container_actor_profile_v2,
)
from .phase2b_c1_six_role_wire_v1 import (
    PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
    PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
    Phase2BC1CeremonyContextBodyV1,
    Phase2BC1CeremonyContextV1,
    Phase2BC1DetachedSignatureV1,
    Phase2BC1ExecutionRegistryBodyV1,
    Phase2BC1ExecutionRegistryCandidateV1,
    Phase2BC1LogicalGenesisTokenBodyV1,
    Phase2BC1LogicalGenesisTokenCandidateV1,
    Phase2BC1LogicalStageV1,
    Phase2BC1RoleKeyV1,
    Phase2BC1SignaturePurposeV1,
    Phase2BC1SixRoleV1,
    Phase2BC1SixRoleWireDispositionV1,
    Phase2BC1SixRoleWireVerificationV1,
    Phase2BC1StoreIfAbsentReceiptBodyV1,
    Phase2BC1StoreIfAbsentReceiptCandidateV1,
    canonical_phase2b_c1_execution_registry_bytes_v1,
    phase2b_c1_attempt_id_v1,
    phase2b_c1_ceremony_context_id_v1,
    phase2b_c1_execution_registry_id_v1,
    phase2b_c1_governance_registry_signature_preimage_v1,
    phase2b_c1_logical_genesis_signature_preimage_v1,
    phase2b_c1_logical_genesis_token_id_v1,
    phase2b_c1_role_key_id_v1,
    phase2b_c1_store_receipt_id_v1,
    phase2b_c1_store_receipt_signature_preimage_v1,
    verify_phase2b_c1_six_role_wire_v1,
)
from .phase2b_recognizer_runtime_cli_v2 import (
    MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
    decode_execution_freeze_manifest_json_v2,
)
from .phase3_local_runtime_v1 import (
    DEFAULT_DOCKER_EXECUTABLE,
    DEFAULT_DOCKER_SOCKET,
    LOCAL_DOCKER_HOST,
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    prepare_local_docker_control_plane_v1,
)


PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_VERSION: Final = (
    "hegel-machine-phase2b-c1-live-docker-ceremony/1"
)
PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_CLAIM_LEVEL: Final = (
    "OWNER_ACCEPTED_SAME_ADMIN_LOCAL_DOCKER_BOOTSTRAP_MECHANICS_ONLY"
)
PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_COMPLETE_REASON: Final = (
    "phase2b_c1_owner_accepted_local_docker_bootstrap_complete_not_actual_960"
)
RECOGNIZER_RUNTIME_IMAGE_DIGEST_V1: Final = (
    "sha256:d27247e9b4b6fcbeca38aea81359b29a855c05f7a7a6e9f2c2276d360160054f"
)
RECOGNIZER_SOURCE_ROOT_SHA256_V1: Final = (
    "d71f96926156b78bbd42d016b04c40aaa3e0abc5dad2510adc05c23e8dd54f54"
)
SCORER_RUNTIME_IMAGE_DIGEST_V1: Final = (
    "sha256:1f89c913b94d2688080a65225fae9564f9c0925215d9ed13060eadc64918a3b9"
)
SCORER_SOURCE_ROOT_SHA256_V1: Final = (
    "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
SECCOMP_PROFILE_PATH_V1: Final = (
    PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
)
SECCOMP_PROFILE_SHA256_V1: Final = (
    "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
)
OPENSSL_EXECUTABLE_V1: Final = "/usr/bin/openssl"
OPENSSL_SHA256_V1: Final = (
    "38633817061f9d161c8b14c1ff72c9f223e8208d977a4c15d843010611a5dded"
)
OPENSSL_VERSION_V1: Final = (
    "OpenSSL 3.5.6 7 Apr 2026 (Library: OpenSSL 3.5.6 7 Apr 2026)"
)
OPENSSL_MODE_V1: Final = 0o100755
OPENSSL_UID_GID_V1: Final = (0, 0)
OPENSSL_NLINK_V1: Final = 1
OPENSSL_SIZE_V1: Final = 1_103_304
_RESOLVED_DOCKER_SOCKET_V1: Final = Path("/run/docker.sock")

FAIL_PREFLIGHT: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_PREFLIGHT"
FAIL_SECCOMP: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_SECCOMP"
FAIL_DOCKER_CONTROL_PLANE: Final = (
    "FAIL_PHASE2B_C1_LIVE_CEREMONY_DOCKER_CONTROL_PLANE"
)
FAIL_IMAGE_BINDING: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_IMAGE_BINDING"
FAIL_VOLUME_PROVISIONING: Final = (
    "FAIL_PHASE2B_C1_LIVE_CEREMONY_VOLUME_PROVISIONING"
)
FAIL_CONTAINER_LIFECYCLE: Final = (
    "FAIL_PHASE2B_C1_LIVE_CEREMONY_CONTAINER_LIFECYCLE"
)
FAIL_QUALIFICATION: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_QUALIFICATION"
FAIL_KEYGEN: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_KEYGEN"
FAIL_SIGNATURE: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_SIGNATURE"
FAIL_STORE: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_STORE"
FAIL_TIME: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_TIME"
FAIL_WIRE: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_WIRE"
FAIL_CLEANUP: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_CLEANUP"
FAIL_DOCKER_TIMEOUT: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_DOCKER_TIMEOUT"
FAIL_DOCKER_CAPTURE: Final = "FAIL_PHASE2B_C1_LIVE_CEREMONY_DOCKER_CAPTURE"

_CAMPAIGN_PREFIX: Final = "phase2b_actual_unsealed_960_campaign_v2_"
_ATTEMPT_PREFIX: Final = "phase2b_actual_unsealed_960_attempt_v2_"
_EXECUTION_FREEZE_PREFIX: Final = "phase2b_execution_freeze_"
_IMAGE_DIGEST_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_CONTAINER_ID_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME_PATTERN: Final = re.compile(r"hegel-p2b-c1-[0-9a-f]{24}-p[1-6]")
_INITIALIZER_NAME_PATTERN: Final = re.compile(r"hegel-p2b-c1-init-[0-9a-f]{24}")
_VOLUME_NAME_PATTERN: Final = re.compile(r"hp2bc1v-[0-9a-f]{48}")
_NAMESPACE_VALUE_PATTERN: Final = re.compile(r"(?:pid|mnt|net|ipc|uts):\[[0-9]+\]")
_MAXIMUM_DOCKER_STDOUT_BYTES: Final = 1_048_576
_MAXIMUM_DOCKER_STDERR_BYTES: Final = 16_384
_MAXIMUM_PUBLIC_STDIN_BYTES: Final = 262_144
_DOCKER_TIMEOUT_SECONDS: Final = 60
_CEREMONY_TOTAL_DEADLINE_SECONDS: Final = 900
_CEREMONY_CLEANUP_RESERVE_SECONDS: Final = 300
# A killed Docker CLI does not cancel a request already accepted by Engine.
# After an ambiguous create, hold the exact name under observation for one full
# ordinary Docker-command timeout before accepting stable absence.  The hard
# ceremony deadline still bounds every probe and the complete cleanup pass.
_AMBIGUOUS_RECONCILIATION_OBSERVATIONS: Final = 61
_AMBIGUOUS_RECONCILIATION_INTERVAL_SECONDS: Final = 1.0
_MAXIMUM_SECCOMP_PROFILE_BYTES: Final = 65_536
_EMPTY_SHA256: Final = hashlib.sha256(b"").hexdigest()

_TARGET_ENGINE_MASKED_PATHS_V1: Final = (
    "/proc/acpi", "/proc/asound", "/proc/interrupts", "/proc/kcore",
    "/proc/keys", "/proc/latency_stats", "/proc/sched_debug", "/proc/scsi",
    "/proc/timer_list", "/proc/timer_stats", "/sys/devices/virtual/powercap",
    "/sys/firmware",
)
_TARGET_ENGINE_READONLY_PATHS_V1: Final = (
    "/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys", "/proc/sysrq-trigger",
)

_FINAL_IMAGE_ENVIRONMENT_V1: Final = (
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG=C.UTF-8",
    "GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D",
    "PYTHON_VERSION=3.11.15",
    "PYTHON_SHA256=272179ddd9a2e41a0fc8e42e33dfbdca0b3711aa5abf372d3f2d51543d09b625",
    "HOME=/nonexistent",
    "PYTHONHASHSEED=0",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=/opt/hegel/src",
)
_CONTAINER_ENVIRONMENT_V1: Final = (
    "PYTHONHASHSEED=0",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=/opt/hegel/src",
    "HOME=/nonexistent",
    "NO_PROXY=*",
    *_FINAL_IMAGE_ENVIRONMENT_V1[:5],
)
_CEREMONY_LABEL: Final = "org.hegel-machine.phase2b.ceremony"
_OWNER_ACCEPTED_LABEL: Final = (
    "org.hegel-machine.phase2b.local-owner-accepted"
)
_OBJECT_KIND_LABEL: Final = "org.hegel-machine.phase2b.object-kind"
_OBJECT_ID_LABEL: Final = "org.hegel-machine.phase2b.object-id-sha256"
_PURPOSE_LABEL: Final = "org.hegel-machine.phase2b.purpose-id"
_ROLE_LABEL: Final = "org.hegel-machine.phase2b.role"
_VOLUME_LABEL: Final = "org.hegel-machine.phase2b.volume-name-sha256"

_CEREMONY_DEADLINES_V1: ContextVar[tuple[float, float] | None] = ContextVar(
    "phase2b_c1_live_ceremony_deadlines", default=None
)
_CEREMONY_CLEANUP_PHASE_V1: ContextVar[bool] = ContextVar(
    "phase2b_c1_live_ceremony_cleanup_phase", default=False
)

_ROLE_MOUNTS: Final = (
    (1, Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR, ("/state",)),
    (2, Phase2BC1SixRoleV1.CUSTODIAN, ("/state", "/custody")),
    (3, Phase2BC1SixRoleV1.STORE_RECEIPT, ("/state", "/ledger", "/cas")),
    (4, Phase2BC1SixRoleV1.TIME_ATTESTOR, ("/state",)),
    (5, Phase2BC1SixRoleV1.RECOGNIZER, ("/state", "/prediction")),
    (6, Phase2BC1SixRoleV1.SCORER_EVALUATOR, ("/state", "/score")),
)
_NAMESPACE_ORDER: Final = ("pid", "mnt", "net", "ipc", "uts")


class Phase2BC1LiveDockerCeremonyErrorV1(RuntimeError):
    """Stable fail-closed error from the live local ceremony boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BC1LiveDockerCeremonyErrorV1(code, detail)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be exact lowercase SHA-256")
    return value


def _require_identifier(value: object, prefix: str, label: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError(f"{label} prefix drift")
    _require_sha(value[len(prefix):], f"{label} suffix")
    return value


def _require_image(value: object, label: str) -> str:
    if type(value) is not str or _IMAGE_DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be exact sha256 image ID")
    return value


@dataclass(frozen=True, slots=True)
class Phase2BC1LiveDockerCeremonyRequestV1:
    """Complete caller-supplied public identity for one bootstrap attempt."""

    campaign_id: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    execution_freeze_manifest_sha256: str
    execution_freeze_manifest_bytes: bytes
    attempt_nonce_sha256: str
    attempt_id: str
    recognizer_image_digest: str
    recognizer_source_root_sha256: str
    scorer_image_digest: str
    scorer_source_root_sha256: str

    def __post_init__(self) -> None:
        _snapshot_request_v1(self)


def _snapshot_request_v1(value: object) -> dict[str, str | bytes]:
    if type(value) is not Phase2BC1LiveDockerCeremonyRequestV1:
        raise TypeError("live ceremony request exact type drift")
    raw = {item.name: getattr(value, item.name) for item in fields(type(value))}
    if any(
        type(raw[name]) is not str
        for name in raw
        if name != "execution_freeze_manifest_bytes"
    ) or type(raw["execution_freeze_manifest_bytes"]) is not bytes:
        raise TypeError("live ceremony request primitive type drift")
    manifest_bytes = bytes(raw["execution_freeze_manifest_bytes"])
    if not 1 <= len(manifest_bytes) <= MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2:
        raise ValueError("execution-freeze manifest byte bound drift")
    closed = {
        "campaign_id": _require_identifier(raw["campaign_id"], _CAMPAIGN_PREFIX, "campaign ID"),
        "protocol_id": _require_identifier(raw["protocol_id"], "phase2b_protocol_", "protocol ID"),
        "exact_freeze_id": _require_identifier(raw["exact_freeze_id"], "phase2b_exact_freeze_", "exact-freeze ID"),
        "execution_freeze_manifest_id": _require_identifier(raw["execution_freeze_manifest_id"], _EXECUTION_FREEZE_PREFIX, "execution-freeze ID"),
        "execution_freeze_manifest_sha256": _require_sha(raw["execution_freeze_manifest_sha256"], "execution-freeze SHA"),
        "execution_freeze_manifest_bytes": manifest_bytes,
        "attempt_nonce_sha256": _require_sha(raw["attempt_nonce_sha256"], "attempt nonce SHA"),
        "attempt_id": _require_identifier(raw["attempt_id"], _ATTEMPT_PREFIX, "attempt ID"),
        "recognizer_image_digest": _require_image(raw["recognizer_image_digest"], "recognizer image"),
        "recognizer_source_root_sha256": _require_sha(raw["recognizer_source_root_sha256"], "recognizer source root"),
        "scorer_image_digest": _require_image(raw["scorer_image_digest"], "scorer image"),
        "scorer_source_root_sha256": _require_sha(raw["scorer_source_root_sha256"], "scorer source root"),
    }
    if raw != closed:
        raise ValueError("live ceremony request normalization drift")
    return dict(closed)


@dataclass(frozen=True, slots=True)
class Phase2BC1LiveOperationBindingV1:
    purpose_id: int
    role: Phase2BC1SixRoleV1
    operation: str
    command_sha256: str
    stdin_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    started_monotonic_ns: int
    finished_monotonic_ns: int

    def __post_init__(self) -> None:
        if (
            type(self.purpose_id) is not int
            or not 1 <= self.purpose_id <= 6
            or type(self.role) is not Phase2BC1SixRoleV1
            or self.role is not PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER[self.purpose_id - 1]
            or type(self.operation) is not str
            or not self.operation
        ):
            raise ValueError("operation binding purpose/role drift")
        for name in ("command_sha256", "stdin_sha256", "stdout_sha256", "stderr_sha256"):
            _require_sha(getattr(self, name), name)
        if (
            type(self.started_monotonic_ns) is not int
            or type(self.finished_monotonic_ns) is not int
            or self.started_monotonic_ns < 0
            or self.finished_monotonic_ns < self.started_monotonic_ns
        ):
            raise ValueError("operation monotonic observation drift")


@dataclass(frozen=True, slots=True)
class Phase2BC1LiveRoleRuntimeBindingV1:
    purpose_id: int
    role: Phase2BC1SixRoleV1
    image_digest: str
    container_name: str
    container_id: str
    purpose_private_volumes: tuple[tuple[str, str], ...]
    qualification_sha256: str
    namespace_identities: tuple[tuple[str, str], ...]
    openssl_sha256: str
    openssl_version: str

    def __post_init__(self) -> None:
        expected = _ROLE_MOUNTS[self.purpose_id - 1] if type(self.purpose_id) is int and 1 <= self.purpose_id <= 6 else None
        if (
            expected is None
            or self.role is not expected[1]
            or _CONTAINER_NAME_PATTERN.fullmatch(self.container_name) is None
            or _CONTAINER_ID_PATTERN.fullmatch(self.container_id) is None
        ):
            raise ValueError("runtime role/container binding drift")
        _require_image(self.image_digest, "runtime image")
        if (
            type(self.purpose_private_volumes) is not tuple
            or tuple(path for path, _ in self.purpose_private_volumes) != expected[2]
            or any(
                type(row) is not tuple or len(row) != 2
                or _VOLUME_NAME_PATTERN.fullmatch(row[1]) is None
                for row in self.purpose_private_volumes
            )
            or len({name for _, name in self.purpose_private_volumes})
            != len(self.purpose_private_volumes)
        ):
            raise ValueError("purpose-private volume binding drift")
        _require_sha(self.qualification_sha256, "qualification SHA")
        if (
            self.namespace_identities != tuple(
                (name, dict(self.namespace_identities).get(name, ""))
                for name in _NAMESPACE_ORDER
            )
            or any(
                _NAMESPACE_VALUE_PATTERN.fullmatch(value) is None
                for _, value in self.namespace_identities
            )
            or self.openssl_sha256 != OPENSSL_SHA256_V1
            or self.openssl_version != OPENSSL_VERSION_V1
        ):
            raise ValueError("live qualification binding drift")


_TRUE_RESULT_CLAIMS: Final = (
    "profile_v2_preflight_verified",
    "owner_accepted_authority_requirement_bound",
    "docker_invoked",
    "six_containers_simultaneously_live_verified",
    "actor_qualification_executed",
    "technical_role_independence_live_verified",
    "fresh_keys_generated",
    "signatures_generated",
    "owner_accepted_registry_signature_verified",
    "store_if_absent_zero_to_one_executed",
    "durable_store_file_and_directory_fsync_verified",
    "logical_genesis_persisted_monotonically",
    "six_role_wire_graph_verified",
    "containers_removed",
    "container_absence_proved",
    "purpose_private_volumes_retained",
)
_FALSE_RESULT_CLAIMS: Final = (
    "external_organizational_authority_verified",
    "independent_human_actors_verified",
    "malicious_host_root_resistance",
    "malicious_docker_daemon_resistance",
    "hostile_administrator_tamper_resistance_verified",
    "owner_deletion_resistance_verified",
    "owner_rollback_resistance_verified",
    "cas_persistence_executed",
    "crash_recovery_verified",
    "cross_host_global_one_shot_verified",
    "authoritative_utc_time_verified",
    "runtime_attestation_authority_verified",
    "actual_960_case_run_verified",
    "recognizer_executed",
    "recognizer_operation_signature_verified",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)
_RESULT_ISSUANCE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1LiveDockerCeremonyReceiptV1:
    ceremony_receipt_id: str
    version: str
    claim_level: str
    reason: str
    authority_class: str
    request: Phase2BC1LiveDockerCeremonyRequestV1
    context: Phase2BC1CeremonyContextV1
    registry: Phase2BC1ExecutionRegistryCandidateV1
    store_receipt: Phase2BC1StoreIfAbsentReceiptCandidateV1
    logical_genesis_token: Phase2BC1LogicalGenesisTokenCandidateV1
    wire_verification: Phase2BC1SixRoleWireVerificationV1
    role_runtime_bindings: tuple[Phase2BC1LiveRoleRuntimeBindingV1, ...]
    operation_bindings: tuple[Phase2BC1LiveOperationBindingV1, ...]
    docker_control_plane_binding_sha256: str
    recognizer_image_inspect_sha256: str
    scorer_image_inspect_sha256: str
    openssl_sha256: str
    openssl_version: str
    stored_registry_sha256: str
    logical_genesis_record_sha256: str
    profile_v2_preflight_verified: bool
    owner_accepted_authority_requirement_bound: bool
    docker_invoked: bool
    six_containers_simultaneously_live_verified: bool
    actor_qualification_executed: bool
    technical_role_independence_live_verified: bool
    fresh_keys_generated: bool
    signatures_generated: bool
    owner_accepted_registry_signature_verified: bool
    store_if_absent_zero_to_one_executed: bool
    durable_store_file_and_directory_fsync_verified: bool
    logical_genesis_persisted_monotonically: bool
    six_role_wire_graph_verified: bool
    containers_removed: bool
    container_absence_proved: bool
    purpose_private_volumes_retained: bool
    external_organizational_authority_verified: bool
    independent_human_actors_verified: bool
    malicious_host_root_resistance: bool
    malicious_docker_daemon_resistance: bool
    hostile_administrator_tamper_resistance_verified: bool
    owner_deletion_resistance_verified: bool
    owner_rollback_resistance_verified: bool
    cas_persistence_executed: bool
    crash_recovery_verified: bool
    cross_host_global_one_shot_verified: bool
    authoritative_utc_time_verified: bool
    runtime_attestation_authority_verified: bool
    actual_960_case_run_verified: bool
    recognizer_executed: bool
    recognizer_operation_signature_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __new__(cls, token: object = None) -> "Phase2BC1LiveDockerCeremonyReceiptV1":
        if token is not _RESULT_ISSUANCE_TOKEN:
            raise TypeError("live ceremony receipts are privately issued")
        return object.__new__(cls)

    def __post_init__(self) -> None:
        if (
            _require_identifier(self.ceremony_receipt_id, "phase2b_c1_live_docker_ceremony_receipt_v1_", "ceremony receipt ID") != self.ceremony_receipt_id
            or self.version != PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_VERSION
            or self.claim_level != PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_CLAIM_LEVEL
            or self.reason != PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_COMPLETE_REASON
            or self.authority_class != PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS
            or type(self.request) is not Phase2BC1LiveDockerCeremonyRequestV1
            or type(self.role_runtime_bindings) is not tuple
            or len(self.role_runtime_bindings) != 6
            or tuple(row.purpose_id for row in self.role_runtime_bindings) != tuple(range(1, 7))
            or type(self.operation_bindings) is not tuple
            or len(self.operation_bindings) != 15
            or self.openssl_sha256 != OPENSSL_SHA256_V1
            or self.openssl_version != OPENSSL_VERSION_V1
        ):
            raise ValueError("live ceremony receipt identity/shape drift")
        for name in (
            "docker_control_plane_binding_sha256", "recognizer_image_inspect_sha256",
            "scorer_image_inspect_sha256", "stored_registry_sha256",
            "logical_genesis_record_sha256",
        ):
            _require_sha(getattr(self, name), name)
        if any(getattr(self, name) is not True for name in _TRUE_RESULT_CLAIMS):
            raise ValueError("live ceremony true-claim drift")
        if any(getattr(self, name) is not False for name in _FALSE_RESULT_CLAIMS):
            raise ValueError("live ceremony false-claim drift")


@dataclass(frozen=True, slots=True)
class _BoundedProcessV1:
    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    started_monotonic_ns: int
    finished_monotonic_ns: int


@dataclass(frozen=True, slots=True)
class _SeccompBindingV1:
    path: Path
    canonical_bytes: bytes
    value: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ImageBindingV1:
    digest: str
    inspect_sha256: str
    labels: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _ContainerTargetV1:
    name: str
    image_digest: str
    ownership_labels: tuple[tuple[str, str], ...]
    expected_labels: tuple[tuple[str, str], ...]
    object_kind: str
    user: str
    args: tuple[str, ...]
    environment: tuple[str, ...]
    working_directory: str
    mounts: tuple[tuple[str, str], ...]
    memory_bytes: int
    pids_limit: int
    cap_add: tuple[str, ...]
    tmpfs: tuple[tuple[str, str], ...]
    nofile_limit: int


@dataclass(frozen=True, slots=True)
class _OwnedContainerV1:
    target: _ContainerTargetV1
    container_id: str


@dataclass(frozen=True, slots=True)
class _OwnedVolumeV1:
    name: str
    labels: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _RoleStateV1:
    purpose_id: int
    role: Phase2BC1SixRoleV1
    image_digest: str
    container_name: str
    volumes: tuple[tuple[str, str], ...]
    image_labels: tuple[tuple[str, str], ...]
    container_labels: tuple[tuple[str, str], ...]
    container_id: str | None = None
    qualification: Mapping[str, object] | None = None


_VOLUME_INITIALIZER_CODE_V1: Final = r"""
import os, stat, sys
path = "/bootstrap"
meta = os.lstat(path)
if not stat.S_ISDIR(meta.st_mode) or os.listdir(path):
    raise SystemExit(91)
# With every capability dropped except CHOWN, chmod must happen while uid 0
# still owns the new volume root.  Chowning first would require CAP_FOWNER.
os.chmod(path, 0o700)
os.chown(path, 65534, 65534)
descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
meta = os.lstat(path)
if meta.st_uid != 65534 or meta.st_gid != 65534 or stat.S_IMODE(meta.st_mode) != 0o700:
    raise SystemExit(92)
""".strip()


_ACTOR_SUPERVISOR_CODE_V1: Final = r"""
import base64, json, os, stat, sys, time
purpose = int(sys.argv[1])
role = sys.argv[2]
paths = json.loads(base64.b64decode(sys.argv[3], validate=True).decode("ascii"))
if purpose not in range(1, 7) or not isinstance(role, str) or not isinstance(paths, list):
    raise SystemExit(91)
for path in paths:
    meta = os.lstat(path)
    if (not stat.S_ISDIR(meta.st_mode) or meta.st_uid != 65534 or meta.st_gid != 65534
            or stat.S_IMODE(meta.st_mode) != 0o700):
        raise SystemExit(92)
while True:
    time.sleep(3600)
""".strip()


_ACTOR_WORKER_CODE_V1: Final = r"""
import base64, errno, fcntl, hashlib, json, os, re, stat, subprocess, sys

OPENSSL = "/usr/bin/openssl"
OPENSSL_SHA = "38633817061f9d161c8b14c1ff72c9f223e8208d977a4c15d843010611a5dded"
OPENSSL_VERSION = "OpenSSL 3.5.6 7 Apr 2026 (Library: OpenSSL 3.5.6 7 Apr 2026)"
PRIVATE_NAME = "ed25519-key.der"
PRIVATE_PREFIX = bytes.fromhex("302e020100300506032b657004220420")
PUBLIC_PREFIX = bytes.fromhex("302a300506032b6570032100")
MAXIMUM_INPUT = 262144
ENV = {"HOME":"/nonexistent","LANG":"C","LC_ALL":"C","PATH":"/usr/bin:/bin"}
ROLES = ("GOVERNANCE_ANCHOR","CUSTODIAN","STORE_RECEIPT","TIME_ATTESTOR","RECOGNIZER","SCORER_EVALUATOR")
MOUNTS = (("/state",),("/state","/custody"),("/state","/ledger","/cas"),("/state",),("/state","/prediction"),("/state","/score"))

def die(code):
    sys.stderr.write("phase2b_c1_actor_worker_failed\n")
    raise SystemExit(code)

def emit(value):
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("ascii") + b"\n"
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()

def read_input():
    payload = sys.stdin.buffer.read(MAXIMUM_INPUT + 1)
    if len(payload) > MAXIMUM_INPUT:
        die(70)
    return payload

def write_all(fd, payload):
    offset = 0
    while offset < len(payload):
        count = os.write(fd, payload[offset:])
        if count <= 0:
            die(71)
        offset += count

def directory_fd(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)

def exact_directory(path):
    meta = os.lstat(path)
    if (not stat.S_ISDIR(meta.st_mode) or stat.S_IMODE(meta.st_mode) != 0o700
            or meta.st_uid != os.geteuid() or meta.st_gid != os.getegid()):
        die(72)
    return meta

def exact_private_key(state):
    directory = directory_fd(state)
    try:
        fd = os.open(PRIVATE_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            before = os.fstat(fd)
            payload = os.read(fd, 49)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    fingerprint = lambda row: (row.st_dev,row.st_ino,row.st_mode,row.st_uid,row.st_gid,row.st_nlink,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
    if (fingerprint(before) != fingerprint(after) or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600 or before.st_uid != os.geteuid()
            or before.st_gid != os.getegid() or before.st_nlink != 1 or before.st_size != 48
            or len(payload) != 48 or not payload.startswith(PRIVATE_PREFIX)):
        die(73)
    return payload

def openssl(arguments, payload=b""):
    completed = subprocess.run([OPENSSL, *arguments], input=payload, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=ENV, close_fds=True, timeout=30, check=False)
    if completed.returncode != 0 or completed.stderr != b"":
        die(74)
    return completed.stdout

def openssl_identity():
    meta = os.lstat(OPENSSL)
    if (not stat.S_ISREG(meta.st_mode) or meta.st_mode != 0o100755 or meta.st_uid != 0
            or meta.st_gid != 0 or meta.st_nlink != 1 or meta.st_size != 1103304):
        die(75)
    fd = os.open(OPENSSL, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1048576)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    version = openssl(["version"]).decode("ascii").rstrip("\n")
    if digest.hexdigest() != OPENSSL_SHA or version != OPENSSL_VERSION:
        die(76)
    return digest.hexdigest(), version

def qualify(payload, purpose, role):
    try:
        value = json.loads(payload.decode("ascii"))
    except Exception:
        die(77)
    if (not isinstance(value, dict) or set(value) != {"mount_paths","nonce_sha256"}
            or not isinstance(value["mount_paths"], list)
            or tuple(value["mount_paths"]) != MOUNTS[purpose - 1]
            or not isinstance(value["nonce_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["nonce_sha256"]) is None):
        die(78)
    mount_rows = []
    for index, path in enumerate(value["mount_paths"]):
        if not isinstance(path, str) or not path.startswith("/") or path == "/":
            die(79)
        meta = exact_directory(path)
        if os.listdir(path):
            die(80)
        token = hashlib.sha256((value["nonce_sha256"] + "\x00" + path).encode("ascii")).digest()
        pending = ".qualification-pending"
        published = ".qualification-published"
        renamed = ".qualification-renamed"
        directory = directory_fd(path)
        try:
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
            try:
                write_all(fd, token)
                os.fchmod(fd, 0o600)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.link(pending, published, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            os.fsync(directory)
            try:
                os.link(pending, published, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                die(81)
            except FileExistsError:
                pass
            os.rename(pending, renamed, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
            os.unlink(published, dir_fd=directory)
            os.unlink(renamed, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)
        if os.listdir(path):
            die(82)
        mount_rows.append({"path":path,"device":meta.st_dev,"mode_octal":"0700","owner_uid":meta.st_uid,"owner_gid":meta.st_gid,"fsync":True,"rename":True,"hardlink_no_replace":True})
    openssl_sha, version = openssl_identity()
    namespaces = {}
    for name in ("pid","mnt","net","ipc","uts"):
        identity = os.readlink("/proc/self/ns/" + name)
        if re.fullmatch(name + r":\[[0-9]+\]", identity) is None:
            die(83)
        namespaces[name] = identity
    emit({"operation":"QUALIFIED","purpose_id":purpose,"role":role,"uid":os.geteuid(),"gid":os.getegid(),
        "nonce_sha256":value["nonce_sha256"],"mounts":mount_rows,"namespaces":namespaces,
        "openssl_sha256":openssl_sha,"openssl_version":version,"key_material_created":False})

def keygen(payload, purpose, role, state):
    if payload != b"":
        die(84)
    exact_directory(state)
    openssl_identity()
    directory = directory_fd(state)
    pending = ".ed25519-key.pending"
    try:
        if os.listdir(directory):
            die(85)
        private = openssl(["genpkey","-algorithm","ED25519","-outform","DER"])
        if len(private) != 48 or not private.startswith(PRIVATE_PREFIX):
            die(86)
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            write_all(fd, private)
            os.fchmod(fd, 0o600)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(pending, PRIVATE_NAME, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
        os.unlink(pending, dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)
    private = exact_private_key(state)
    public_der = openssl(["pkey","-inform","DER","-pubout","-outform","DER"], private)
    if len(public_der) != 44 or not public_der.startswith(PUBLIC_PREFIX):
        die(87)
    public = public_der[len(PUBLIC_PREFIX):]
    emit({"operation":"KEY_GENERATED","purpose_id":purpose,"role":role,"algorithm":"Ed25519",
        "public_key_hex":public.hex(),"public_spki_der_sha256":hashlib.sha256(public_der).hexdigest(),
        "private_key_exported":False,"private_key_file_mode_octal":"0600","file_fsync":True,"directory_fsync":True,"hardlink_no_replace":True})

def sign_message(payload, purpose, role, state, operation="SIGNED"):
    if not 1 <= len(payload) <= MAXIMUM_INPUT:
        die(88)
    private = exact_private_key(state)
    signature = openssl(["pkeyutl","-sign","-rawin","-inkey",state + "/" + PRIVATE_NAME,"-keyform","DER"], payload)
    if len(signature) != 64:
        die(89)
    emit({"operation":operation,"purpose_id":purpose,"role":role,"preimage_sha256":hashlib.sha256(payload).hexdigest(),"signature_hex":signature.hex()})

def lock_file(path, name):
    directory = directory_fd(path)
    try:
        created = False
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
            created = True
        except FileExistsError:
            fd = os.open(name, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        meta = os.fstat(fd)
        if (not stat.S_ISREG(meta.st_mode) or stat.S_IMODE(meta.st_mode) != 0o600
                or meta.st_uid != os.geteuid() or meta.st_gid != os.getegid()
                or meta.st_nlink != 1 or meta.st_size != 0):
            die(90)
        if created:
            os.fsync(fd)
            os.fsync(directory)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return directory, fd
    except BaseException:
        os.close(directory)
        raise

def durable_publish(path, target, content, allowed_existing):
    directory, lock = lock_file(path, allowed_existing)
    try:
        names = set(os.listdir(directory))
        if names != {allowed_existing} or target in names:
            die(91)
        pending = "." + target + ".pending-" + os.urandom(16).hex()
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            write_all(fd, content)
            os.fchmod(fd, 0o600)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(pending, target, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
        os.unlink(pending, dir_fd=directory)
        os.fsync(directory)
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            meta = os.fstat(fd)
            observed = b""
            while len(observed) <= len(content):
                chunk = os.read(fd, len(content) + 1 - len(observed))
                if not chunk:
                    break
                observed += chunk
        finally:
            os.close(fd)
        if (observed != content or not stat.S_ISREG(meta.st_mode) or stat.S_IMODE(meta.st_mode) != 0o600
                or meta.st_uid != os.geteuid() or meta.st_gid != os.getegid() or meta.st_nlink != 1):
            die(92)
        return hashlib.sha256(observed).hexdigest()
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)
        os.close(directory)

def store_and_sign(payload, purpose, role, state, ledger):
    try:
        value = json.loads(payload.decode("ascii"))
    except Exception:
        die(93)
    if not isinstance(value, dict) or set(value) != {"content_b64","object_key","preimage_b64","stored_content_sha256"}:
        die(94)
    try:
        content = base64.b64decode(value["content_b64"], validate=True)
        preimage = base64.b64decode(value["preimage_b64"], validate=True)
    except Exception:
        die(95)
    if (not isinstance(value["object_key"], str) or not isinstance(value["stored_content_sha256"], str)
            or hashlib.sha256(content).hexdigest() != value["stored_content_sha256"]):
        die(96)
    exact_private_key(state)
    exact_directory(ledger)
    target = "registry-" + hashlib.sha256(value["object_key"].encode("ascii")).hexdigest() + ".generation-0000000001"
    record_sha = durable_publish(ledger, target, content, ".registry.lock")
    signature = openssl(["pkeyutl","-sign","-rawin","-inkey",state + "/" + PRIVATE_NAME,"-keyform","DER"], preimage)
    if len(signature) != 64:
        die(97)
    emit({"operation":"STORE_IF_ABSENT_SIGNED","purpose_id":purpose,"role":role,"condition":"IF_ABSENT",
        "previous_generation":0,"new_generation":1,"stored_content_sha256":record_sha,
        "record_name_sha256":hashlib.sha256(target.encode("ascii")).hexdigest(),
        "preimage_sha256":hashlib.sha256(preimage).hexdigest(),"signature_hex":signature.hex(),
        "file_fsync":True,"directory_fsync":True,"hardlink_no_replace":True})

def time_and_sign(payload, purpose, role, state):
    try:
        value = json.loads(payload.decode("ascii"))
    except Exception:
        die(98)
    if (not isinstance(value, dict) or set(value) != {"logical_sequence","preimage_b64","previous_logical_sequence_or_null"}
            or value["previous_logical_sequence_or_null"] is not None or value["logical_sequence"] != 0):
        die(99)
    try:
        preimage = base64.b64decode(value["preimage_b64"], validate=True)
    except Exception:
        die(100)
    exact_private_key(state)
    directory, lock = lock_file(state, ".logical-sequence.lock")
    try:
        pattern = re.compile(r"logical-sequence-([0-9]{20})")
        observed = sorted(int(match.group(1)) for name in os.listdir(directory) if (match := pattern.fullmatch(name)))
        if observed:
            die(101)
        target = "logical-sequence-00000000000000000000"
        record = json.dumps({"logical_sequence":0,"preimage_sha256":hashlib.sha256(preimage).hexdigest(),"previous_logical_sequence_or_null":None}, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
        names = set(os.listdir(directory))
        if names != {PRIVATE_NAME, ".logical-sequence.lock"}:
            die(102)
        pending = ".logical-sequence.pending-" + os.urandom(16).hex()
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            write_all(fd, record); os.fchmod(fd, 0o600); os.fsync(fd)
        finally:
            os.close(fd)
        os.link(pending, target, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory); os.unlink(pending, dir_fd=directory); os.fsync(directory)
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            meta = os.fstat(fd)
            observed_record = os.read(fd, len(record) + 1)
        finally:
            os.close(fd)
        if (observed_record != record or not stat.S_ISREG(meta.st_mode)
                or stat.S_IMODE(meta.st_mode) != 0o600 or meta.st_uid != os.geteuid()
                or meta.st_gid != os.getegid() or meta.st_nlink != 1
                or meta.st_size != len(record)):
            die(103)
        signature = openssl(["pkeyutl","-sign","-rawin","-inkey",state + "/" + PRIVATE_NAME,"-keyform","DER"], preimage)
        if len(signature) != 64:
            die(104)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); os.close(lock); os.close(directory)
    emit({"operation":"TIME_GENESIS_SIGNED","purpose_id":purpose,"role":role,
        "previous_logical_sequence_or_null":None,"logical_sequence":0,"record_sha256":hashlib.sha256(record).hexdigest(),
        "preimage_sha256":hashlib.sha256(preimage).hexdigest(),"signature_hex":signature.hex(),
        "file_fsync":True,"directory_fsync":True,"hardlink_no_replace":True})

if len(sys.argv) != 5:
    die(105)
operation, purpose_text, role, state = sys.argv[1:]
try:
    purpose = int(purpose_text)
except ValueError:
    die(106)
if (purpose not in range(1, 7) or role != ROLES[purpose - 1] or state != "/state"
        or os.geteuid() != 65534 or os.getegid() != 65534):
    die(107)
payload = read_input()
if operation == "qualify":
    qualify(payload, purpose, role)
elif operation == "keygen":
    keygen(payload, purpose, role, state)
elif operation == "sign":
    sign_message(payload, purpose, role, state)
elif operation == "store-and-sign":
    store_and_sign(payload, purpose, role, state, "/ledger")
elif operation == "time-genesis-and-sign":
    time_and_sign(payload, purpose, role, state)
else:
    die(108)
""".strip()


@contextmanager
def _ceremony_deadline_scope_v1() -> Iterator[None]:
    # Re-entrant so the public-entry scope and the lifecycle scope share one
    # deadline instead of silently granting a second budget after preflight.
    if _CEREMONY_DEADLINES_V1.get() is not None:
        yield
        return
    started = time.monotonic()
    deadlines = (
        started
        + _CEREMONY_TOTAL_DEADLINE_SECONDS
        - _CEREMONY_CLEANUP_RESERVE_SECONDS,
        started + _CEREMONY_TOTAL_DEADLINE_SECONDS,
    )
    token = _CEREMONY_DEADLINES_V1.set(deadlines)
    try:
        yield
    finally:
        _CEREMONY_DEADLINES_V1.reset(token)


@contextmanager
def _ceremony_cleanup_phase_v1() -> Iterator[None]:
    token = _CEREMONY_CLEANUP_PHASE_V1.set(True)
    try:
        yield
    finally:
        _CEREMONY_CLEANUP_PHASE_V1.reset(token)


def _bounded_by_ceremony_deadline_v1(timeout_seconds: int) -> float:
    deadlines = _CEREMONY_DEADLINES_V1.get()
    if deadlines is None:
        return float(timeout_seconds)
    deadline = deadlines[1 if _CEREMONY_CLEANUP_PHASE_V1.get() else 0]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _fail(FAIL_DOCKER_TIMEOUT, "ceremony total deadline exhausted")
    return min(float(timeout_seconds), remaining)


def _kill_process_v1(process: subprocess.Popen[bytes]) -> None:
    def wait_bound() -> float:
        deadlines = _CEREMONY_DEADLINES_V1.get()
        if deadlines is None:
            return 0.5
        return max(0.001, min(0.5, deadlines[1] - time.monotonic()))

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=wait_bound())
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=wait_bound())
        except (OSError, subprocess.TimeoutExpired):
            pass


def _run_docker_command_v1(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: Path,
    stdin_payload: bytes | None = None,
    timeout_seconds: int = _DOCKER_TIMEOUT_SECONDS,
    maximum_stdout_bytes: int = _MAXIMUM_DOCKER_STDOUT_BYTES,
    maximum_stderr_bytes: int = _MAXIMUM_DOCKER_STDERR_BYTES,
) -> _BoundedProcessV1:
    """Run one exact absolute Docker command with bounded public input/output."""

    if (
        type(command) not in {tuple, list}
        or len(command) < 3
        or any(type(item) is not str or not item or "\x00" in item for item in command)
        or command[0] != DEFAULT_DOCKER_EXECUTABLE.as_posix()
        or command[1] != f"--host={LOCAL_DOCKER_HOST}"
        or type(working_directory) is not type(Path("/"))
        or not working_directory.is_absolute()
        or type(timeout_seconds) is not int
        or timeout_seconds < 1
        or type(maximum_stdout_bytes) is not int
        or maximum_stdout_bytes < 1
        or type(maximum_stderr_bytes) is not int
        or maximum_stderr_bytes < 1
    ):
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker command/limit vector drift")
    if stdin_payload is not None and (
        type(stdin_payload) is not bytes or len(stdin_payload) > _MAXIMUM_PUBLIC_STDIN_BYTES
    ):
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker public stdin shape/cap drift")

    effective_timeout = _bounded_by_ceremony_deadline_v1(timeout_seconds)
    process: subprocess.Popen[bytes] | None = None
    stdin_file = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    started = time.monotonic_ns()
    deadline = time.monotonic() + effective_timeout
    try:
        if stdin_payload is None:
            child_stdin: object = subprocess.DEVNULL
        else:
            stdin_file = tempfile.TemporaryFile(mode="w+b", dir=working_directory)
            stdin_file.write(stdin_payload)
            stdin_file.flush()
            os.fsync(stdin_file.fileno())
            stdin_file.seek(0)
            child_stdin = stdin_file
        process = subprocess.Popen(
            list(command), cwd=working_directory.as_posix(), stdin=child_stdin,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(environment),
            close_fds=True, start_new_session=True, bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker client pipes unavailable")
        streams = {
            process.stdout.fileno(): (process.stdout, stdout, maximum_stdout_bytes, "stdout"),
            process.stderr.fileno(): (process.stderr, stderr, maximum_stderr_bytes, "stderr"),
        }
        for descriptor, (stream, _, _, _) in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(stream, selectors.EVENT_READ, data=descriptor)
        open_descriptors = set(streams)
        while open_descriptors:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_v1(process)
                _fail(FAIL_DOCKER_TIMEOUT, f"Docker command exceeded {timeout_seconds}s")
            events = selector.select(timeout=min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = tuple(
                    (
                        selectors.SelectorKey(streams[fd][0], fd, selectors.EVENT_READ, fd),
                        selectors.EVENT_READ,
                    )
                    for fd in tuple(open_descriptors)
                )
            for key, _ in events:
                descriptor = key.data
                if descriptor not in open_descriptors:
                    continue
                stream, buffer, cap, label = streams[descriptor]
                try:
                    chunk = os.read(descriptor, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    open_descriptors.remove(descriptor)
                    continue
                if len(buffer) + len(chunk) > cap:
                    _kill_process_v1(process)
                    _fail(FAIL_DOCKER_CAPTURE, f"Docker {label} exceeded byte cap")
                buffer.extend(chunk)
        try:
            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_process_v1(process)
            _fail(FAIL_DOCKER_TIMEOUT, f"Docker command exceeded {timeout_seconds}s")
        return _BoundedProcessV1(
            command=tuple(command), returncode=returncode,
            stdout=bytes(stdout), stderr=bytes(stderr),
            started_monotonic_ns=started, finished_monotonic_ns=time.monotonic_ns(),
        )
    except Phase2BC1LiveDockerCeremonyErrorV1:
        if process is not None:
            _kill_process_v1(process)
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        if process is not None:
            _kill_process_v1(process)
        _fail(FAIL_DOCKER_CONTROL_PLANE, f"Docker client failed locally: {type(exc).__name__}")
    except BaseException:
        if process is not None:
            _kill_process_v1(process)
        raise
    finally:
        selector.close()
        if stdin_file is not None:
            stdin_file.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _decode_json_v1(payload: bytes, *, code: str, label: str) -> object:
    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError("duplicate JSON key")
            value[name] = item
        return value

    try:
        return json.loads(
            payload.decode("ascii"), object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _fail(code, f"{label} JSON rejected: {type(exc).__name__}")


def _expect_success_v1(result: _BoundedProcessV1, *, code: str, label: str) -> None:
    if result.returncode != 0 or result.stderr != b"":
        _fail(code, f"{label} failed or emitted stderr")


@contextmanager
def _prepared_runtime_v1(runtime_parent: Path) -> Iterator[tuple[Path, LocalDockerControlPlaneV1]]:
    try:
        owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-phase2b-c1-ceremony-",
            repository_root=REPOSITORY_ROOT,
            parent=runtime_parent,
        )
        with owner as name:
            root = Path(name)
            control = prepare_local_docker_control_plane_v1(
                root, repository_root=REPOSITORY_ROOT
            )
            yield root, control
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_DOCKER_CONTROL_PLANE, f"local runtime rejected: {exc.code}")


def _validate_control_plane_v1(control: LocalDockerControlPlaneV1, root: Path) -> None:
    expected_environment = {
        "DOCKER_CONFIG", "DOCKER_HOST", "HOME", "LANG", "LC_ALL", "PATH"
    }
    if (
        type(control) is not LocalDockerControlPlaneV1
        or control.executable != DEFAULT_DOCKER_EXECUTABLE
        or control.socket_path != _RESOLVED_DOCKER_SOCKET_V1
        or control.config_directory.parent != root
        or set(control.environment) != expected_environment
        or control.environment.get("DOCKER_HOST") != LOCAL_DOCKER_HOST
        or control.environment.get("PATH") != "/usr/bin:/bin"
        or any("proxy" in name.casefold() for name in control.environment)
        or control.command("version")[:2]
        != [DEFAULT_DOCKER_EXECUTABLE.as_posix(), f"--host={LOCAL_DOCKER_HOST}"]
    ):
        _fail(FAIL_DOCKER_CONTROL_PLANE, "local Docker control plane drift")


def _load_seccomp_v1() -> _SeccompBindingV1:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            SECCOMP_PROFILE_PATH_V1,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        )
        before = os.fstat(descriptor)
        resolved_open_file = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        payload_buffer = bytearray()
        while len(payload_buffer) <= _MAXIMUM_SECCOMP_PROFILE_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    16_384,
                    _MAXIMUM_SECCOMP_PROFILE_BYTES + 1 - len(payload_buffer),
                ),
            )
            if not chunk:
                break
            payload_buffer.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        _fail(FAIL_SECCOMP, f"seccomp source unavailable: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    payload = bytes(payload_buffer)
    fingerprint = lambda row: (
        row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
        row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
    )
    if (
        fingerprint(before) != fingerprint(after)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != len(payload)
        or not 1 <= len(payload) <= _MAXIMUM_SECCOMP_PROFILE_BYTES
        or resolved_open_file != SECCOMP_PROFILE_PATH_V1
        or _sha256(payload) != SECCOMP_PROFILE_SHA256_V1
    ):
        _fail(FAIL_SECCOMP, "seccomp stable-open identity differs from frozen profile")
    value = _decode_json_v1(payload, code=FAIL_SECCOMP, label="seccomp")
    if (
        type(value) is not dict
        or value.get("defaultAction") != "SCMP_ACT_ALLOW"
        or type(value.get("syscalls")) is not list
        or not value["syscalls"]
    ):
        _fail(FAIL_SECCOMP, "seccomp semantic shape drift")
    canonical = _canonical_json_bytes(value)
    return _SeccompBindingV1(
        path=SECCOMP_PROFILE_PATH_V1,
        canonical_bytes=canonical,
        value=MappingProxyType(dict(value)),
    )


def _stage_seccomp_v1(
    source: _SeccompBindingV1, runtime_root: Path
) -> _SeccompBindingV1:
    filename = "phase2b-c1-actor-seccomp.json"
    directory: int | None = None
    descriptor: int | None = None
    try:
        directory = os.open(
            runtime_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=directory,
        )
        offset = 0
        while offset < len(source.canonical_bytes):
            written = os.write(descriptor, source.canonical_bytes[offset:])
            if written <= 0:
                _fail(FAIL_SECCOMP, "staged seccomp write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(directory)
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=directory,
        )
        observed = bytearray()
        while len(observed) <= _MAXIMUM_SECCOMP_PROFILE_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    16_384,
                    _MAXIMUM_SECCOMP_PROFILE_BYTES + 1 - len(observed),
                ),
            )
            if not chunk:
                break
            observed.extend(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        _fail(FAIL_SECCOMP, f"private seccomp staging failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)
    fingerprint = lambda row: (
        row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
        row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
    )
    if (
        fingerprint(before) != fingerprint(after)
        or not stat.S_ISREG(after.st_mode)
        or stat.S_IMODE(after.st_mode) != 0o400
        or after.st_nlink != 1
        or bytes(observed) != source.canonical_bytes
    ):
        _fail(FAIL_SECCOMP, "private seccomp snapshot identity drift")
    return _SeccompBindingV1(
        path=runtime_root / filename,
        canonical_bytes=source.canonical_bytes,
        value=source.value,
    )


def _preflight_request_v1(
    request: Phase2BC1LiveDockerCeremonyRequestV1,
) -> Phase2BC1LiveDockerCeremonyRequestV1:
    try:
        closed = _snapshot_request_v1(request)
        snapshot = load_phase2b_c1_container_actor_profile_v2()
        execution_freeze = decode_execution_freeze_manifest_json_v2(
            closed["execution_freeze_manifest_bytes"]
        )
    except Phase2BC1LiveDockerCeremonyErrorV1:
        raise
    except Exception as exc:
        _fail(FAIL_PREFLIGHT, f"request/profile snapshot rejected: {type(exc).__name__}")
    if (
        snapshot.authority_class != PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V2
        or snapshot.authority_class != PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS
        or snapshot.profile_content_id != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
        or snapshot.profile_content_id != PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
        or snapshot.canonical_profile_sha256 != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        or snapshot.canonical_profile_sha256 != PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
        or snapshot.bound_source_sha256_verified is not True
    ):
        _fail(FAIL_PREFLIGHT, "exact profile-v2/authority identity drift")
    profile = snapshot.profile
    roles = profile.get("actor_roles")
    requirement = profile.get("owner_accepted_authority_requirement")
    docker_policy = profile.get("docker_control_policy")
    if (
        type(roles) is not tuple
        or len(roles) != 6
        or tuple((row["purpose_id"], row["role"], row["exclusive_writable_mounts"]) for row in roles)
        != tuple((purpose, role.value, mounts) for purpose, role, mounts in _ROLE_MOUNTS)
        or not isinstance(requirement, Mapping)
        or requirement.get("requirement_id") != PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
        or not isinstance(docker_policy, Mapping)
        or docker_policy.get("executable") != DEFAULT_DOCKER_EXECUTABLE.as_posix()
        or docker_policy.get("socket") != DEFAULT_DOCKER_SOCKET.as_posix()
        or docker_policy.get("six_live_containers_concurrent_for_separation_probe") is not True
        or docker_policy.get("purpose_private_state_volume_required") is not True
    ):
        _fail(FAIL_PREFLIGHT, "effective live actor policy drift")
    if (
        _sha256(closed["execution_freeze_manifest_bytes"])
        != closed["execution_freeze_manifest_sha256"]
        or execution_freeze.manifest_id != closed["execution_freeze_manifest_id"]
        or execution_freeze.protocol_id != closed["protocol_id"]
        or execution_freeze.exact_freeze_id != closed["exact_freeze_id"]
        or execution_freeze.recognizer_image_digest
        != closed["recognizer_image_digest"]
    ):
        _fail(
            FAIL_PREFLIGHT,
            "canonical execution-freeze material/identity binding drift",
        )
    if (
        closed["protocol_id"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_PROTOCOL_ID
        or closed["exact_freeze_id"] != PHASE2B_C1_SIX_ROLE_WIRE_V1_FROZEN_EXACT_FREEZE_ID
        or closed["recognizer_image_digest"] != RECOGNIZER_RUNTIME_IMAGE_DIGEST_V1
        or closed["recognizer_source_root_sha256"] != RECOGNIZER_SOURCE_ROOT_SHA256_V1
        or closed["scorer_image_digest"] != SCORER_RUNTIME_IMAGE_DIGEST_V1
        or closed["scorer_source_root_sha256"] != SCORER_SOURCE_ROOT_SHA256_V1
        or SCORER_SOURCE_ROOT_SHA256_V1
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_SCORER_SOURCE_ROOT_SHA256
    ):
        _fail(FAIL_PREFLIGHT, "frozen protocol/image/source identity drift")
    expected_attempt = phase2b_c1_attempt_id_v1(
        authority_requirement_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
        actor_profile_content_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
        actor_profile_canonical_sha256=PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
        campaign_id=closed["campaign_id"], protocol_id=closed["protocol_id"],
        exact_freeze_id=closed["exact_freeze_id"],
        execution_freeze_manifest_id=closed["execution_freeze_manifest_id"],
        execution_freeze_manifest_sha256=closed["execution_freeze_manifest_sha256"],
        attempt_nonce_sha256=closed["attempt_nonce_sha256"],
    )
    if closed["attempt_id"] != expected_attempt:
        _fail(FAIL_PREFLIGHT, "attempt ID is not the exact complete preflight address")
    return Phase2BC1LiveDockerCeremonyRequestV1(**closed)


def _inspect_image_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    *,
    image_digest: str,
    source_root_sha256: str,
    role_kind: str,
) -> _ImageBindingV1:
    result = _run_docker_command_v1(
        control.command("image", "inspect", image_digest),
        environment=control.environment, working_directory=root,
    )
    _expect_success_v1(result, code=FAIL_IMAGE_BINDING, label=f"{role_kind} image inspect")
    decoded = _decode_json_v1(result.stdout, code=FAIL_IMAGE_BINDING, label=f"{role_kind} image inspect")
    if type(decoded) is not list or len(decoded) != 1 or type(decoded[0]) is not dict:
        _fail(FAIL_IMAGE_BINDING, f"{role_kind} image inspect root drift")
    value = decoded[0]
    config = value.get("Config")
    rootfs = value.get("RootFS")
    if (
        value.get("Id") != image_digest
        or value.get("Os") != "linux"
        or value.get("Architecture") not in {"amd64", "x86_64"}
        or type(config) is not dict
        or set(config) != {"ArgsEscaped", "Cmd", "Entrypoint", "Env", "Labels", "User", "WorkingDir"}
        or config.get("ArgsEscaped") is not True
        or config.get("User") != "65534:65534"
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Env") != list(_FINAL_IMAGE_ENVIRONMENT_V1)
        or type(config.get("Labels")) is not dict
        or any(
            type(name) is not str or type(item) is not str
            for name, item in config["Labels"].items()
        )
        or type(rootfs) is not dict
        or rootfs.get("Type") != "layers"
        or type(rootfs.get("Layers")) is not list
        or not rootfs["Layers"]
        or any(
            type(item) is not str or _IMAGE_DIGEST_PATTERN.fullmatch(item) is None
            for item in rootfs["Layers"]
        )
    ):
        _fail(FAIL_IMAGE_BINDING, f"{role_kind} image identity/config drift")
    labels = config["Labels"]
    if (
        labels.get("org.hegel-machine.phase2b.actor-profile.content-id")
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
        or labels.get("org.hegel-machine.phase2b.actor-profile.canonical-sha256")
        != PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
    ):
        _fail(FAIL_IMAGE_BINDING, f"{role_kind} image profile labels drift")
    if role_kind == "recognizer":
        if (
            config.get("Entrypoint") != ["/opt/hegel/bin/phase2b-recognizer"]
            or config.get("Cmd")
            != ["--input", "/phase2b/input", "--output", "/phase2b/output"]
            or
            labels.get("org.hegel-machine.phase2b.recognizer-source-root.sha256")
            != source_root_sha256
            or labels.get("org.hegel-machine.phase2b.owner-accepted-authority-requirement.id")
            != PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
        ):
            _fail(FAIL_IMAGE_BINDING, "recognizer image source/authority labels drift")
    elif role_kind == "scorer":
        if (
            config.get("Entrypoint") != ["/opt/hegel/bin/phase2b-scorer"]
            or config.get("Cmd")
            != [
                "--input",
                "/phase2b/input/phase2b_scorer_bundle_v1.bin",
                "--output",
                "/phase2b/output",
            ]
            or labels.get("org.hegel-machine.phase2b.scorer-source-root.sha256")
            != source_root_sha256
        ):
            _fail(FAIL_IMAGE_BINDING, "scorer image source label drift")
    else:
        _fail(FAIL_IMAGE_BINDING, "image role kind drift")
    return _ImageBindingV1(
        digest=image_digest,
        inspect_sha256=_sha256(result.stdout),
        labels=tuple(sorted(labels.items())),
    )


def _ceremony_digest_v1(run_token: str) -> str:
    if type(run_token) is not str or re.fullmatch(r"[0-9a-f]{24}", run_token) is None:
        raise RuntimeError("ceremony run-token shape drift")
    return _sha256(run_token.encode("ascii"))


def _ownership_labels_v1(
    run_token: str,
    *,
    object_kind: str,
    object_name: str,
    purpose_id: int | None = None,
    role: Phase2BC1SixRoleV1 | None = None,
    volume_name: str | None = None,
) -> tuple[tuple[str, str], ...]:
    labels = {
        _CEREMONY_LABEL: _ceremony_digest_v1(run_token),
        _OWNER_ACCEPTED_LABEL: "true",
        _OBJECT_KIND_LABEL: object_kind,
        _OBJECT_ID_LABEL: _sha256(object_name.encode("ascii")),
    }
    if purpose_id is not None:
        if type(purpose_id) is not int or not 1 <= purpose_id <= 6:
            raise RuntimeError("ownership purpose drift")
        labels[_PURPOSE_LABEL] = str(purpose_id)
    if role is not None:
        if type(role) is not Phase2BC1SixRoleV1:
            raise RuntimeError("ownership role drift")
        labels[_ROLE_LABEL] = role.value
    if volume_name is not None:
        if _VOLUME_NAME_PATTERN.fullmatch(volume_name) is None:
            raise RuntimeError("ownership volume locator drift")
        labels[_VOLUME_LABEL] = _sha256(volume_name.encode("ascii"))
    return tuple(sorted(labels.items()))


def _merged_container_labels_v1(
    image_labels: tuple[tuple[str, str], ...],
    ownership_labels: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    merged = dict(image_labels)
    if any(name in merged for name, _ in ownership_labels):
        raise RuntimeError("image/ceremony label namespace collision")
    merged.update(ownership_labels)
    return tuple(sorted(merged.items()))


def _opaque_volume_name_v1(
    run_token: str,
    purpose_id: int,
    mount_path: str,
    *,
    attempt_id: str,
) -> str:
    if purpose_id == 3 and mount_path == "/ledger":
        # A deterministic, still-opaque ledger locator makes Docker's retained
        # local volume a cooperating-process one-shot reservation for this
        # exact attempt.  A per-run label checked after `volume create` closes
        # the inspect/create race because create is otherwise idempotent.
        material = (
            b"HEGEL/PHASE2B/C1/LIVE_STORE_LEDGER_VOLUME/V1\x00"
            + attempt_id.encode("ascii")
        )
    else:
        material = (
            b"HEGEL/PHASE2B/C1/LIVE_VOLUME/V1\x00"
            + run_token.encode("ascii") + b"\x00"
            + str(purpose_id).encode("ascii") + b"\x00"
            + mount_path.encode("ascii")
        )
    digest = hashlib.sha256(material).hexdigest()[:48]
    name = "hp2bc1v-" + digest
    if _VOLUME_NAME_PATTERN.fullmatch(name) is None:
        raise RuntimeError("opaque volume name construction drift")
    return name


def _make_role_states_v1(
    request: Phase2BC1LiveDockerCeremonyRequestV1,
    run_token: str,
    *,
    recognizer_image: _ImageBindingV1,
    scorer_image: _ImageBindingV1,
) -> tuple[_RoleStateV1, ...]:
    rows = []
    for purpose, role, mount_paths in _ROLE_MOUNTS:
        image = recognizer_image if purpose == 5 else scorer_image
        container_name = f"hegel-p2b-c1-{run_token}-p{purpose}"
        ownership_labels = _ownership_labels_v1(
            run_token,
            object_kind="actor",
            object_name=container_name,
            purpose_id=purpose,
            role=role,
        )
        rows.append(
            _RoleStateV1(
                purpose_id=purpose,
                role=role,
                image_digest=image.digest,
                container_name=container_name,
                volumes=tuple(
                    (
                        path,
                        _opaque_volume_name_v1(
                            run_token, purpose, path, attempt_id=request.attempt_id
                        ),
                    )
                    for path in mount_paths
                ),
                image_labels=image.labels,
                container_labels=_merged_container_labels_v1(
                    image.labels, ownership_labels
                ),
            )
        )
    if (
        len({name for row in rows for _, name in row.volumes}) != 11
        or len({row.container_name for row in rows}) != 6
    ):
        raise RuntimeError("role runtime identity collision")
    return tuple(rows)


def _actor_target_v1(role: _RoleStateV1, run_token: str) -> _ContainerTargetV1:
    ownership = _ownership_labels_v1(
        run_token,
        object_kind="actor",
        object_name=role.container_name,
        purpose_id=role.purpose_id,
        role=role.role,
    )
    if role.container_labels != _merged_container_labels_v1(
        role.image_labels, ownership
    ):
        raise RuntimeError("actor target label construction drift")
    paths_wire = base64.b64encode(
        _canonical_json_bytes([path for path, _ in role.volumes])
    ).decode("ascii")
    arguments = (
        "-c", _ACTOR_SUPERVISOR_CODE_V1, str(role.purpose_id),
        role.role.value, paths_wire,
    )
    environment = (
        f"HEGEL_ACTOR_PROFILE_CONTENT_ID={PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID}",
        f"HEGEL_AUTHORITY_REQUIREMENT_ID={PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID}",
        f"HEGEL_PURPOSE_ID={role.purpose_id}",
        f"HEGEL_ROLE={role.role.value}",
        *_CONTAINER_ENVIRONMENT_V1,
    )
    memory_bytes = 536_870_912 if role.purpose_id <= 4 else 2_147_483_648
    pids_limit = 64 if role.purpose_id <= 4 else 128
    return _ContainerTargetV1(
        name=role.container_name,
        image_digest=role.image_digest,
        ownership_labels=ownership,
        expected_labels=role.container_labels,
        object_kind="actor",
        user="65534:65534",
        args=arguments,
        environment=environment,
        working_directory="/opt/hegel",
        mounts=role.volumes,
        memory_bytes=memory_bytes,
        pids_limit=pids_limit,
        cap_add=(),
        tmpfs=((
            "/tmp",
            "rw,noexec,nosuid,nodev,size=16777216,uid=65534,gid=65534,mode=0700",
        ),),
        nofile_limit=128,
    )


def _initializer_target_v1(
    *,
    volume_name: str,
    image: _ImageBindingV1,
    run_token: str,
) -> _ContainerTargetV1:
    name = "hegel-p2b-c1-init-" + _sha256(
        b"HEGEL/PHASE2B/C1/VOLUME_INITIALIZER/V1\x00"
        + run_token.encode("ascii")
        + b"\x00"
        + volume_name.encode("ascii")
    )[:24]
    if _INITIALIZER_NAME_PATTERN.fullmatch(name) is None:
        raise RuntimeError("initializer name construction drift")
    ownership = _ownership_labels_v1(
        run_token,
        object_kind="volume-initializer",
        object_name=name,
        volume_name=volume_name,
    )
    return _ContainerTargetV1(
        name=name,
        image_digest=image.digest,
        ownership_labels=ownership,
        expected_labels=_merged_container_labels_v1(image.labels, ownership),
        object_kind="volume-initializer",
        user="0:0",
        args=("-c", _VOLUME_INITIALIZER_CODE_V1),
        environment=_CONTAINER_ENVIRONMENT_V1,
        working_directory="/opt/hegel",
        mounts=(("/bootstrap", volume_name),),
        memory_bytes=134_217_728,
        pids_limit=32,
        cap_add=("CHOWN",),
        tmpfs=(),
        nofile_limit=64,
    )


def _container_absence_result_v1(
    result: _BoundedProcessV1, container_name: str
) -> bool:
    expected_stderr = (
        f"Error response from daemon: No such container: {container_name}\n"
    ).encode("ascii")
    return (
        result.returncode == 1
        and result.stdout == b"[]\n"
        and result.stderr == expected_stderr
    )


def _volume_absence_result_v1(
    result: _BoundedProcessV1, volume_name: str
) -> bool:
    expected_stderr = (
        f"Error response from daemon: get {volume_name}: no such volume\n"
    ).encode("ascii")
    return (
        result.returncode == 1
        and result.stdout == b"[]\n"
        and result.stderr == expected_stderr
    )


def _phase_command_timeout_v1() -> int:
    return 5 if _CEREMONY_CLEANUP_PHASE_V1.get() else _DOCKER_TIMEOUT_SECONDS


def _inspect_container_reference_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    reference: str,
    *,
    code: str,
    label: str,
) -> dict[str, object] | None:
    result = _run_docker_command_v1(
        control.command("container", "inspect", reference),
        environment=control.environment,
        working_directory=root,
        timeout_seconds=_phase_command_timeout_v1(),
    )
    if _container_absence_result_v1(result, reference):
        return None
    _expect_success_v1(result, code=code, label=label)
    value = _decode_json_v1(result.stdout, code=code, label=label)
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(code, f"{label} root shape drift")
    return value[0]


def _inspect_volume_reference_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    volume_name: str,
    *,
    code: str,
    label: str,
) -> dict[str, object] | None:
    result = _run_docker_command_v1(
        control.command("volume", "inspect", volume_name),
        environment=control.environment,
        working_directory=root,
        timeout_seconds=_phase_command_timeout_v1(),
    )
    if _volume_absence_result_v1(result, volume_name):
        return None
    _expect_success_v1(result, code=code, label=label)
    value = _decode_json_v1(result.stdout, code=code, label=label)
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(code, f"{label} root shape drift")
    return value[0]


def _validate_security_options_v1(
    value: object,
    seccomp: _SeccompBindingV1,
    *,
    code: str,
    label: str,
) -> None:
    expected_seccomp = "seccomp=" + seccomp.canonical_bytes.decode("ascii")
    if value != ["no-new-privileges:true", expected_seccomp]:
        _fail(code, f"{label} exact security-option vector drift")
    # Parse the observation itself after the exact vector check.  Never parse
    # the expected bytes and mistake that for an observation of Engine state.
    embedded = value[1].removeprefix("seccomp=").encode("ascii")
    decoded = _decode_json_v1(embedded, code=code, label=f"{label} seccomp")
    if (
        embedded != _canonical_json_bytes(decoded)
        or decoded != dict(seccomp.value)
    ):
        _fail(code, f"{label} embedded seccomp object drift")


def _validate_container_mounts_v1(
    mounts: object,
    target: _ContainerTargetV1,
    *,
    code: str,
) -> None:
    if type(mounts) is not list or len(mounts) != len(target.mounts):
        _fail(code, "container exact mount count drift")
    seen: set[tuple[str, str]] = set()
    expected = set(target.mounts)
    for row in mounts:
        if type(row) is not dict:
            _fail(code, "container mount row type drift")
        destination = row.get("Destination")
        name = row.get("Name")
        source = row.get("Source")
        pair = (destination, name)
        if (
            set(row) != {
                "Type", "Name", "Source", "Destination", "Driver", "Mode",
                "RW", "Propagation",
            }
            or pair not in expected
            or pair in seen
            or row.get("Type") != "volume"
            or row.get("Driver") != "local"
            or row.get("Mode") != "z"
            or row.get("RW") is not True
            or row.get("Propagation") != ""
            or source != f"/var/lib/docker/volumes/{name}/_data"
        ):
            _fail(code, "container exact writable-volume mount drift")
        seen.add(pair)
    if seen != expected:
        _fail(code, "container mount destination/name set drift")


def _validate_container_row_v1(
    row: Mapping[str, object],
    target: _ContainerTargetV1,
    seccomp: _SeccompBindingV1,
    *,
    code: str,
    expected_state: str | None,
) -> str:
    config = row.get("Config")
    host = row.get("HostConfig")
    state = row.get("State")
    if not all(type(item) is dict for item in (config, host, state)):
        _fail(code, "container inspect config/host/state section drift")
    container_id = row.get("Id")
    if type(container_id) is not str or _CONTAINER_ID_PATTERN.fullmatch(container_id) is None:
        _fail(code, "container inspect full ID drift")
    _validate_security_options_v1(
        host.get("SecurityOpt"), seccomp, code=code, label=target.object_kind
    )
    _validate_container_mounts_v1(row.get("Mounts"), target, code=code)
    expected_cap_add = list(target.cap_add) if target.cap_add else None
    expected_tmpfs = dict(target.tmpfs) if target.tmpfs else None
    expected_ulimits = [
        {
            "Name": "nofile",
            "Hard": target.nofile_limit,
            "Soft": target.nofile_limit,
        }
    ]
    if (
        row.get("Name") != "/" + target.name
        or row.get("Image") != target.image_digest
        or row.get("Path") != "/usr/local/bin/python3"
        or row.get("Args") != list(target.args)
        or config.get("Hostname") != target.name
        or config.get("User") != target.user
        or config.get("Image") != target.image_digest
        or config.get("Entrypoint") != ["/usr/local/bin/python3"]
        or config.get("Cmd") != list(target.args)
        or config.get("WorkingDir") != target.working_directory
        or config.get("Env") != list(target.environment)
        or config.get("Labels") != dict(target.expected_labels)
        or config.get("Healthcheck") != {"Test": ["NONE"]}
        or config.get("Volumes") is not None
        or config.get("StopTimeout") != 1
        or host.get("NetworkMode") != "none"
        or host.get("IpcMode") != "private"
        or host.get("ReadonlyRootfs") is not True
        or host.get("Privileged") is not False
        or host.get("AutoRemove") is not False
        or host.get("CapDrop") != ["ALL"]
        or host.get("CapAdd") != expected_cap_add
        or host.get("Memory") != target.memory_bytes
        or host.get("MemorySwap") != target.memory_bytes
        or host.get("PidsLimit") != target.pids_limit
        or host.get("NanoCpus") != 0
        or host.get("CpuShares") != 0
        or host.get("CpuPeriod") != 0
        or host.get("CpuQuota") != 0
        or host.get("CpuRealtimePeriod") != 0
        or host.get("CpuRealtimeRuntime") != 0
        or host.get("CpusetCpus") != ""
        or host.get("CpusetMems") != ""
        or host.get("BlkioWeight") != 0
        or host.get("BlkioWeightDevice") != []
        or host.get("BlkioDeviceReadBps") != []
        or host.get("BlkioDeviceWriteBps") != []
        or host.get("BlkioDeviceReadIOps") != []
        or host.get("BlkioDeviceWriteIOps") != []
        or host.get("MemoryReservation") != 0
        or host.get("MemorySwappiness") is not None
        or host.get("CpuCount") != 0
        or host.get("CpuPercent") != 0
        or host.get("IOMaximumIOps") != 0
        or host.get("IOMaximumBandwidth") != 0
        or host.get("Ulimits") != expected_ulimits
        or host.get("Tmpfs") != expected_tmpfs
        or host.get("Runtime") != "runc"
        or host.get("Isolation") != ""
        or host.get("Cgroup") != ""
        or host.get("CgroupParent") != ""
        or host.get("CgroupnsMode") != "private"
        or host.get("PidMode") != ""
        or host.get("UsernsMode") != ""
        or host.get("UTSMode") != ""
        or host.get("Devices") != []
        or host.get("DeviceCgroupRules") is not None
        or host.get("DeviceRequests") is not None
        or host.get("Binds") is not None
        or host.get("VolumesFrom") is not None
        or host.get("VolumeDriver") != ""
        or host.get("GroupAdd") is not None
        or host.get("Links") is not None
        or host.get("PortBindings") != {}
        or host.get("PublishAllPorts") is not False
        or host.get("Dns") is not None
        or host.get("DnsOptions") != []
        or host.get("DnsSearch") != []
        or host.get("ExtraHosts") is not None
        or host.get("RestartPolicy") != {"Name": "no", "MaximumRetryCount": 0}
        or host.get("ContainerIDFile") != ""
        or host.get("ConsoleSize") != [0, 0]
        or host.get("ShmSize") != 67_108_864
        or host.get("OomScoreAdj") != 0
        or host.get("OomKillDisable") is not None
        or host.get("MaskedPaths") != list(_TARGET_ENGINE_MASKED_PATHS_V1)
        or host.get("ReadonlyPaths") != list(_TARGET_ENGINE_READONLY_PATHS_V1)
    ):
        _fail(code, f"{target.object_kind} exact container policy drift")
    if expected_state is not None:
        running = expected_state == "running"
        if (
            state.get("Status") != expected_state
            or state.get("Running") is not running
            or state.get("Paused") is not False
            or state.get("Restarting") is not False
            or state.get("OOMKilled") is not False
            or state.get("Dead") is not False
            or state.get("ExitCode") != 0
            or (
                (type(state.get("Pid")) is not int or state["Pid"] <= 0)
                if running
                else state.get("Pid") != 0
            )
        ):
            _fail(code, f"{target.object_kind} container state drift")
    return container_id


def _register_owned_container_v1(
    owned: dict[str, _OwnedContainerV1],
    target: _ContainerTargetV1,
    container_id: str,
) -> _OwnedContainerV1:
    candidate = _OwnedContainerV1(target=target, container_id=container_id)
    previous = owned.get(target.name)
    if previous is not None and previous != candidate:
        _fail(FAIL_CONTAINER_LIFECYCLE, "owned container identity changed")
    owned[target.name] = candidate
    return candidate


def _reconcile_container_target_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    target: _ContainerTargetV1,
    owned: dict[str, _OwnedContainerV1],
    *,
    code: str,
    expected_state: str | None,
) -> tuple[_OwnedContainerV1, dict[str, object]] | None:
    row = _inspect_container_reference_v1(
        control, root, target.name, code=code,
        label=f"{target.object_kind} reconciliation inspect",
    )
    if row is None:
        return None
    container_id = _validate_container_row_v1(
        row, target, seccomp, code=code, expected_state=expected_state
    )
    return _register_owned_container_v1(owned, target, container_id), row


def _volume_labels_v1(run_token: str, volume_name: str) -> tuple[tuple[str, str], ...]:
    return _ownership_labels_v1(
        run_token,
        object_kind="purpose-private-volume",
        object_name=volume_name,
        volume_name=volume_name,
    )


def _validate_owned_volume_row_v1(
    row: Mapping[str, object],
    *,
    volume_name: str,
    labels: tuple[tuple[str, str], ...],
    code: str,
) -> None:
    if (
        row.get("Name") != volume_name
        or row.get("Driver") != "local"
        or row.get("Scope") != "local"
        or row.get("Options") not in (None, {})
        or row.get("Labels") != dict(labels)
    ):
        _fail(code, "volume exact ownership/driver identity drift")


def _reconcile_volume_target_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    *,
    volume_name: str,
    labels: tuple[tuple[str, str], ...],
    owned: dict[str, _OwnedVolumeV1],
    code: str,
) -> _OwnedVolumeV1 | None:
    row = _inspect_volume_reference_v1(
        control, root, volume_name, code=code,
        label="purpose-private volume reconciliation inspect",
    )
    if row is None:
        return None
    _validate_owned_volume_row_v1(
        row, volume_name=volume_name, labels=labels, code=code
    )
    candidate = _OwnedVolumeV1(name=volume_name, labels=labels)
    previous = owned.get(volume_name)
    if previous is not None and previous != candidate:
        _fail(code, "owned volume identity changed")
    owned[volume_name] = candidate
    return candidate


def _container_create_command_v1(
    control: LocalDockerControlPlaneV1,
    seccomp: _SeccompBindingV1,
    target: _ContainerTargetV1,
) -> list[str]:
    arguments = [
        "container", "create", f"--name={target.name}",
        f"--hostname={target.name}", "--platform=linux/amd64",
        "--no-healthcheck", "--pull=never", "--network=none",
        "--ipc=private", "--read-only", f"--user={target.user}",
        "--workdir=/opt/hegel", "--restart=no", "--stop-timeout=1",
        "--runtime=runc", "--cgroupns=private", "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={seccomp.path.as_posix()}",
        f"--pids-limit={target.pids_limit}",
        f"--memory={target.memory_bytes}",
        f"--memory-swap={target.memory_bytes}",
        f"--ulimit=nofile={target.nofile_limit}:{target.nofile_limit}",
    ]
    arguments.extend(f"--cap-add={name}" for name in target.cap_add)
    arguments.extend(f"--tmpfs={path}:{options}" for path, options in target.tmpfs)
    arguments.extend(f"--env={item}" for item in target.environment)
    arguments.extend(f"--label={name}={value}" for name, value in target.ownership_labels)
    arguments.extend(
        f"--mount=type=volume,src={name},dst={path},volume-nocopy"
        for path, name in target.mounts
    )
    arguments.extend(
        ["--entrypoint=/usr/local/bin/python3", target.image_digest, *target.args]
    )
    return control.command(*arguments)


def _create_owned_container_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    target: _ContainerTargetV1,
    owned: dict[str, _OwnedContainerV1],
    ambiguous: set[str],
    *,
    code: str,
) -> _OwnedContainerV1:
    preexisting = _inspect_container_reference_v1(
        control, root, target.name, code=code,
        label=f"{target.object_kind} pre-create absence inspect",
    )
    if preexisting is not None:
        _fail(code, f"{target.object_kind} exact-name collision preserved")
    create_result: _BoundedProcessV1 | None = None
    create_error: BaseException | None = None
    ambiguous.add(target.name)
    try:
        create_result = _run_docker_command_v1(
            _container_create_command_v1(control, seccomp, target),
            environment=control.environment, working_directory=root,
        )
    except BaseException as exc:
        create_error = exc
    reconciled: tuple[_OwnedContainerV1, dict[str, object]] | None = None
    reconcile_error: BaseException | None = None
    try:
        reconciled = _reconcile_container_target_v1(
            control, root, seccomp, target, owned,
            code=code, expected_state="created",
        )
    except BaseException as exc:
        reconcile_error = exc
    if reconciled is not None:
        ambiguous.discard(target.name)
    if create_error is not None:
        raise create_error
    if reconcile_error is not None:
        raise reconcile_error
    if create_result is None or reconciled is None:
        _fail(code, f"{target.object_kind} create lacked exact-owned side effect")
    container, _ = reconciled
    if (
        create_result.returncode != 0
        or create_result.stderr != b""
        or create_result.stdout != (container.container_id + "\n").encode("ascii")
    ):
        _fail(code, f"{target.object_kind} create output/inspect ID drift")
    return container


def _start_actor_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    role: _RoleStateV1,
) -> None:
    if role.container_id is None:
        _fail(FAIL_CONTAINER_LIFECYCLE, "actor start lacks owned full ID")
    result = _run_docker_command_v1(
        control.command("container", "start", role.container_id),
        environment=control.environment, working_directory=root,
    )
    _expect_success_v1(result, code=FAIL_CONTAINER_LIFECYCLE, label="actor start")
    if result.stdout != (role.container_id + "\n").encode("ascii"):
        _fail(FAIL_CONTAINER_LIFECYCLE, "actor start full-ID output drift")


def _inspect_actor_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    role: _RoleStateV1,
    run_token: str,
) -> None:
    if role.container_id is None:
        _fail(FAIL_CONTAINER_LIFECYCLE, "actor inspect lacks owned full ID")
    row = _inspect_container_reference_v1(
        control, root, role.container_id, code=FAIL_CONTAINER_LIFECYCLE,
        label="actor full-ID inspect",
    )
    if row is None:
        _fail(FAIL_CONTAINER_LIFECYCLE, "owned actor unexpectedly absent")
    observed_id = _validate_container_row_v1(
        row, _actor_target_v1(role, run_token), seccomp,
        code=FAIL_CONTAINER_LIFECYCLE, expected_state="running",
    )
    if observed_id != role.container_id:
        _fail(FAIL_CONTAINER_LIFECYCLE, "actor full-ID inspect identity drift")


def _initialize_volume_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    target: _ContainerTargetV1,
    owned_containers: dict[str, _OwnedContainerV1],
    ambiguous_containers: set[str],
) -> None:
    container = _create_owned_container_v1(
        control, root, seccomp, target, owned_containers,
        ambiguous_containers,
        code=FAIL_VOLUME_PROVISIONING,
    )
    result = _run_docker_command_v1(
        control.command("container", "start", "--attach", container.container_id),
        environment=control.environment, working_directory=root,
    )
    _expect_success_v1(
        result, code=FAIL_VOLUME_PROVISIONING, label="volume initializer start/attach"
    )
    if result.stdout != b"":
        _fail(FAIL_VOLUME_PROVISIONING, "volume initializer emitted stdout")
    row = _inspect_container_reference_v1(
        control, root, container.container_id, code=FAIL_VOLUME_PROVISIONING,
        label="volume initializer exited inspect",
    )
    if row is None:
        _fail(FAIL_VOLUME_PROVISIONING, "volume initializer disappeared before cleanup")
    observed_id = _validate_container_row_v1(
        row, target, seccomp, code=FAIL_VOLUME_PROVISIONING,
        expected_state="exited",
    )
    if observed_id != container.container_id:
        _fail(FAIL_VOLUME_PROVISIONING, "initializer full-ID identity drift")


def _provision_volume_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    *,
    volume_name: str,
    initializer_target: _ContainerTargetV1,
    run_token: str,
    owned_volumes: dict[str, _OwnedVolumeV1],
    owned_containers: dict[str, _OwnedContainerV1],
    ambiguous_volumes: set[str],
    ambiguous_containers: set[str],
) -> None:
    labels = _volume_labels_v1(run_token, volume_name)
    preexisting = _inspect_volume_reference_v1(
        control, root, volume_name, code=FAIL_VOLUME_PROVISIONING,
        label="purpose-private volume pre-create absence inspect",
    )
    if preexisting is not None:
        _fail(FAIL_VOLUME_PROVISIONING, "volume exact-name collision preserved")
    create_result: _BoundedProcessV1 | None = None
    create_error: BaseException | None = None
    ambiguous_volumes.add(volume_name)
    try:
        create_result = _run_docker_command_v1(
            control.command(
                "volume", "create", "--driver=local",
                *(f"--label={name}={value}" for name, value in labels),
                volume_name,
            ),
            environment=control.environment, working_directory=root,
        )
    except BaseException as exc:
        create_error = exc
    reconciled: _OwnedVolumeV1 | None = None
    reconcile_error: BaseException | None = None
    try:
        reconciled = _reconcile_volume_target_v1(
            control, root, volume_name=volume_name, labels=labels,
            owned=owned_volumes, code=FAIL_VOLUME_PROVISIONING,
        )
    except BaseException as exc:
        reconcile_error = exc
    if reconciled is not None:
        ambiguous_volumes.discard(volume_name)
    if create_error is not None:
        raise create_error
    if reconcile_error is not None:
        raise reconcile_error
    if create_result is None or reconciled is None:
        _fail(FAIL_VOLUME_PROVISIONING, "volume create lacked exact-owned side effect")
    if (
        create_result.returncode != 0
        or create_result.stderr != b""
        or create_result.stdout != (volume_name + "\n").encode("ascii")
    ):
        _fail(FAIL_VOLUME_PROVISIONING, "volume create output drift")
    _initialize_volume_v1(
        control, root, seccomp, initializer_target, owned_containers,
        ambiguous_containers,
    )
    _cleanup_containers_best_effort_v1(
        control, root, seccomp, (initializer_target,), owned_containers,
        ambiguous_containers,
    )


def _create_actor_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    role: _RoleStateV1,
    run_token: str,
    owned: dict[str, _OwnedContainerV1],
    ambiguous: set[str],
) -> _RoleStateV1:
    container = _create_owned_container_v1(
        control, root, seccomp, _actor_target_v1(role, run_token), owned,
        ambiguous,
        code=FAIL_CONTAINER_LIFECYCLE,
    )
    return replace(role, container_id=container.container_id)


def _record_actor_exec_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    role: _RoleStateV1,
    *,
    operation_name: str,
    worker_operation: str,
    stdin_payload: bytes,
    bindings: list[Phase2BC1LiveOperationBindingV1],
    failure_code: str,
) -> Mapping[str, object]:
    if role.container_id is None:
        _fail(failure_code, f"{operation_name} lacks owned actor full ID")
    command = control.command(
        "container", "exec", "--interactive", role.container_id,
        "/usr/local/bin/python3", "-c", _ACTOR_WORKER_CODE_V1,
        worker_operation, str(role.purpose_id), role.role.value, "/state",
    )
    result = _run_docker_command_v1(
        command, environment=control.environment, working_directory=root,
        stdin_payload=stdin_payload,
    )
    _expect_success_v1(result, code=failure_code, label=operation_name)
    if not result.stdout.endswith(b"\n") or result.stdout.count(b"\n") != 1:
        _fail(failure_code, f"{operation_name} public output framing drift")
    decoded = _decode_json_v1(result.stdout, code=failure_code, label=operation_name)
    if type(decoded) is not dict:
        _fail(failure_code, f"{operation_name} public output root drift")
    bindings.append(
        Phase2BC1LiveOperationBindingV1(
            purpose_id=role.purpose_id, role=role.role, operation=operation_name,
            command_sha256=_sha256(_canonical_json_bytes(tuple(command))),
            stdin_sha256=_sha256(stdin_payload), stdout_sha256=_sha256(result.stdout),
            stderr_sha256=_sha256(result.stderr),
            started_monotonic_ns=result.started_monotonic_ns,
            finished_monotonic_ns=result.finished_monotonic_ns,
        )
    )
    return MappingProxyType(dict(decoded))


def _qualify_all_roles_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    roles: tuple[_RoleStateV1, ...],
    operations: list[Phase2BC1LiveOperationBindingV1],
    run_token: str,
) -> tuple[tuple[_RoleStateV1, ...], tuple[Phase2BC1LiveRoleRuntimeBindingV1, ...]]:
    qualified: list[_RoleStateV1] = []
    runtime_bindings: list[Phase2BC1LiveRoleRuntimeBindingV1] = []
    for role in roles:
        nonce_sha = _sha256(secrets.token_bytes(32))
        payload = _canonical_json_bytes(
            {"mount_paths": [path for path, _ in role.volumes], "nonce_sha256": nonce_sha}
        )
        value = _record_actor_exec_v1(
            control, root, role, operation_name="QUALIFY",
            worker_operation="qualify", stdin_payload=payload,
            bindings=operations, failure_code=FAIL_QUALIFICATION,
        )
        expected_fields = {
            "operation", "purpose_id", "role", "uid", "gid", "nonce_sha256",
            "mounts", "namespaces", "openssl_sha256", "openssl_version",
            "key_material_created",
        }
        namespaces = value.get("namespaces")
        mounts = value.get("mounts")
        if (
            set(value) != expected_fields
            or value.get("operation") != "QUALIFIED"
            or value.get("purpose_id") != role.purpose_id
            or value.get("role") != role.role.value
            or value.get("uid") != 65534
            or value.get("gid") != 65534
            or value.get("nonce_sha256") != nonce_sha
            or value.get("key_material_created") is not False
            or value.get("openssl_sha256") != OPENSSL_SHA256_V1
            or value.get("openssl_version") != OPENSSL_VERSION_V1
            or type(namespaces) is not dict
            or set(namespaces) != set(_NAMESPACE_ORDER)
            or any(
                type(namespaces[name]) is not str
                or _NAMESPACE_VALUE_PATTERN.fullmatch(namespaces[name]) is None
                for name in _NAMESPACE_ORDER
            )
            or type(mounts) is not list
            or len(mounts) != len(role.volumes)
        ):
            _fail(FAIL_QUALIFICATION, "actor qualification identity/output drift")
        for expected_path, observed in zip(
            (path for path, _ in role.volumes), mounts, strict=True
        ):
            if (
                type(observed) is not dict
                or set(observed) != {
                    "path", "device", "mode_octal", "owner_uid", "owner_gid",
                    "fsync", "rename", "hardlink_no_replace",
                }
                or observed.get("path") != expected_path
                or type(observed.get("device")) is not int
                or observed.get("mode_octal") != "0700"
                or observed.get("owner_uid") != 65534
                or observed.get("owner_gid") != 65534
                or observed.get("fsync") is not True
                or observed.get("rename") is not True
                or observed.get("hardlink_no_replace") is not True
            ):
                _fail(FAIL_QUALIFICATION, "actor private-mount qualification drift")
        frozen = MappingProxyType(dict(value))
        updated = replace(role, qualification=frozen)
        qualified.append(updated)
        if updated.container_id is None:
            _fail(FAIL_QUALIFICATION, "qualified actor lacks a container ID")
        runtime_bindings.append(
            Phase2BC1LiveRoleRuntimeBindingV1(
                purpose_id=updated.purpose_id, role=updated.role,
                image_digest=updated.image_digest, container_name=updated.container_name,
                container_id=updated.container_id,
                purpose_private_volumes=updated.volumes,
                qualification_sha256=_sha256(_canonical_json_bytes(dict(value))),
                namespace_identities=tuple((name, namespaces[name]) for name in _NAMESPACE_ORDER),
                openssl_sha256=OPENSSL_SHA256_V1,
                openssl_version=OPENSSL_VERSION_V1,
            )
        )

    # Every probe completed while the other five supervisors remained live;
    # this final six-way replay occurs before the first key operation.
    qualified_tuple = tuple(qualified)
    for role in qualified_tuple:
        _inspect_actor_v1(control, root, seccomp, role, run_token)
    for namespace in _NAMESPACE_ORDER:
        identities = {
            dict(binding.namespace_identities)[namespace]
            for binding in runtime_bindings
        }
        if len(identities) != 6:
            _fail(FAIL_QUALIFICATION, f"{namespace} namespace is not pairwise distinct")
    if len({name for row in runtime_bindings for _, name in row.purpose_private_volumes}) != 11:
        _fail(FAIL_QUALIFICATION, "purpose-private volume locators are not globally distinct")
    return qualified_tuple, tuple(runtime_bindings)


def _generate_role_keys_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    roles: tuple[_RoleStateV1, ...],
    operations: list[Phase2BC1LiveOperationBindingV1],
) -> tuple[Phase2BC1RoleKeyV1, ...]:
    keys: list[Phase2BC1RoleKeyV1] = []
    expected_fields = {
        "operation", "purpose_id", "role", "algorithm", "public_key_hex",
        "public_spki_der_sha256", "private_key_exported",
        "private_key_file_mode_octal", "file_fsync", "directory_fsync",
        "hardlink_no_replace",
    }
    for role in roles:
        value = _record_actor_exec_v1(
            control, root, role, operation_name="KEYGEN",
            worker_operation="keygen", stdin_payload=b"",
            bindings=operations, failure_code=FAIL_KEYGEN,
        )
        if (
            set(value) != expected_fields
            or value.get("operation") != "KEY_GENERATED"
            or value.get("purpose_id") != role.purpose_id
            or value.get("role") != role.role.value
            or value.get("algorithm") != "Ed25519"
            or value.get("private_key_exported") is not False
            or value.get("private_key_file_mode_octal") != "0600"
            or value.get("file_fsync") is not True
            or value.get("directory_fsync") is not True
            or value.get("hardlink_no_replace") is not True
            or type(value.get("public_spki_der_sha256")) is not str
            or _SHA256_PATTERN.fullmatch(value["public_spki_der_sha256"]) is None
            or type(value.get("public_key_hex")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value["public_key_hex"]) is None
        ):
            _fail(FAIL_KEYGEN, "fresh role key public output drift")
        public_key = bytes.fromhex(value["public_key_hex"])
        if _sha256(bytes.fromhex("302a300506032b6570032100") + public_key) != value["public_spki_der_sha256"]:
            _fail(FAIL_KEYGEN, "Ed25519 SPKI DER binding drift")
        keys.append(
            Phase2BC1RoleKeyV1(
                purpose_id=role.purpose_id, role=role.role, algorithm="Ed25519",
                key_epoch=PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
                purpose_private=True, public_key=public_key,
                key_id=phase2b_c1_role_key_id_v1(
                    purpose_id=role.purpose_id, role=role.role, public_key=public_key
                ),
            )
        )
    if len({key.public_key for key in keys}) != 6 or len({key.key_id for key in keys}) != 6:
        _fail(FAIL_KEYGEN, "fresh role keys are not pairwise distinct")
    return tuple(keys)


def _build_context_v1(
    request: Phase2BC1LiveDockerCeremonyRequestV1,
) -> Phase2BC1CeremonyContextV1:
    body = Phase2BC1CeremonyContextBodyV1(
        version=PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        authority_class=PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS,
        authority_requirement_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID,
        actor_profile_content_id=PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID,
        actor_profile_canonical_sha256=PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256,
        campaign_id=request.campaign_id, protocol_id=request.protocol_id,
        exact_freeze_id=request.exact_freeze_id,
        execution_freeze_manifest_id=request.execution_freeze_manifest_id,
        execution_freeze_manifest_sha256=request.execution_freeze_manifest_sha256,
        attempt_nonce_sha256=request.attempt_nonce_sha256,
        attempt_id=request.attempt_id,
        recognizer_image_digest=request.recognizer_image_digest,
        recognizer_source_root_sha256=request.recognizer_source_root_sha256,
        scorer_image_digest=request.scorer_image_digest,
        scorer_source_root_sha256=request.scorer_source_root_sha256,
    )
    return Phase2BC1CeremonyContextV1(
        body=body, context_id=phase2b_c1_ceremony_context_id_v1(body)
    )


def _build_live_graph_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    request: Phase2BC1LiveDockerCeremonyRequestV1,
    roles: tuple[_RoleStateV1, ...],
    keys: tuple[Phase2BC1RoleKeyV1, ...],
    operations: list[Phase2BC1LiveOperationBindingV1],
) -> tuple[
    Phase2BC1CeremonyContextV1,
    Phase2BC1ExecutionRegistryCandidateV1,
    Phase2BC1StoreIfAbsentReceiptCandidateV1,
    Phase2BC1LogicalGenesisTokenCandidateV1,
    Phase2BC1SixRoleWireVerificationV1,
    str,
    str,
]:
    context = _build_context_v1(request)
    registry_body = Phase2BC1ExecutionRegistryBodyV1(
        version=PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        context_id=context.context_id,
        authority_class=context.body.authority_class,
        authority_requirement_id=context.body.authority_requirement_id,
        actor_profile_content_id=context.body.actor_profile_content_id,
        actor_profile_canonical_sha256=context.body.actor_profile_canonical_sha256,
        campaign_id=context.body.campaign_id, protocol_id=context.body.protocol_id,
        exact_freeze_id=context.body.exact_freeze_id,
        execution_freeze_manifest_id=context.body.execution_freeze_manifest_id,
        execution_freeze_manifest_sha256=context.body.execution_freeze_manifest_sha256,
        attempt_nonce_sha256=context.body.attempt_nonce_sha256,
        attempt_id=context.body.attempt_id,
        recognizer_image_digest=context.body.recognizer_image_digest,
        recognizer_source_root_sha256=context.body.recognizer_source_root_sha256,
        scorer_image_digest=context.body.scorer_image_digest,
        scorer_source_root_sha256=context.body.scorer_source_root_sha256,
        role_keys=keys,
    )
    governance_preimage = phase2b_c1_governance_registry_signature_preimage_v1(
        registry_body, signer_key_id=keys[0].key_id
    )
    governance_output = _record_actor_exec_v1(
        control, root, roles[0], operation_name="GOVERNANCE_SIGN",
        worker_operation="sign", stdin_payload=governance_preimage,
        bindings=operations, failure_code=FAIL_SIGNATURE,
    )
    if (
        set(governance_output) != {"operation", "purpose_id", "role", "preimage_sha256", "signature_hex"}
        or governance_output.get("operation") != "SIGNED"
        or governance_output.get("purpose_id") != 1
        or governance_output.get("role") != roles[0].role.value
        or governance_output.get("preimage_sha256") != _sha256(governance_preimage)
        or type(governance_output.get("signature_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{128}", governance_output["signature_hex"]) is None
    ):
        _fail(FAIL_SIGNATURE, "governance signature output drift")
    governance_signature = Phase2BC1DetachedSignatureV1(
        purpose_id=1, role=Phase2BC1SixRoleV1.GOVERNANCE_ANCHOR,
        signature_purpose=Phase2BC1SignaturePurposeV1.GOVERNANCE_CHECKPOINT,
        algorithm="Ed25519", key_epoch=PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
        signer_key_id=keys[0].key_id,
        signature=bytes.fromhex(governance_output["signature_hex"]),
    )
    registry = Phase2BC1ExecutionRegistryCandidateV1(
        body=registry_body, governance_signature=governance_signature,
        execution_registry_id=phase2b_c1_execution_registry_id_v1(
            registry_body, governance_signature
        ),
    )
    registry_bytes = canonical_phase2b_c1_execution_registry_bytes_v1(registry)
    registry_sha = _sha256(registry_bytes)
    object_key = (
        "phase2b/c1/actual-unsealed-960/"
        f"{request.campaign_id}/attempt-0/execution-registry"
    )
    store_body = Phase2BC1StoreIfAbsentReceiptBodyV1(
        version=PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        context_id=context.context_id,
        authority_requirement_id=context.body.authority_requirement_id,
        actor_profile_content_id=context.body.actor_profile_content_id,
        campaign_id=request.campaign_id, attempt_id=request.attempt_id,
        execution_registry_id=registry.execution_registry_id,
        object_key=object_key, previous_generation=0, new_generation=1,
        condition="IF_ABSENT", previous_content_sha256_or_null=None,
        stored_content_sha256=registry_sha,
    )
    store_preimage = phase2b_c1_store_receipt_signature_preimage_v1(
        store_body, signer_key_id=keys[2].key_id
    )
    store_payload = _canonical_json_bytes(
        {
            "content_b64": base64.b64encode(registry_bytes).decode("ascii"),
            "object_key": object_key,
            "preimage_b64": base64.b64encode(store_preimage).decode("ascii"),
            "stored_content_sha256": registry_sha,
        }
    )
    store_output = _record_actor_exec_v1(
        control, root, roles[2], operation_name="STORE_IF_ABSENT_AND_SIGN",
        worker_operation="store-and-sign", stdin_payload=store_payload,
        bindings=operations, failure_code=FAIL_STORE,
    )
    expected_store_fields = {
        "operation", "purpose_id", "role", "condition", "previous_generation",
        "new_generation", "stored_content_sha256", "record_name_sha256",
        "preimage_sha256", "signature_hex", "file_fsync", "directory_fsync",
        "hardlink_no_replace",
    }
    expected_store_target = (
        "registry-" + _sha256(object_key.encode("ascii"))
        + ".generation-0000000001"
    )
    if (
        set(store_output) != expected_store_fields
        or store_output.get("operation") != "STORE_IF_ABSENT_SIGNED"
        or store_output.get("purpose_id") != 3
        or store_output.get("role") != roles[2].role.value
        or store_output.get("condition") != "IF_ABSENT"
        or store_output.get("previous_generation") != 0
        or store_output.get("new_generation") != 1
        or store_output.get("stored_content_sha256") != registry_sha
        or store_output.get("preimage_sha256") != _sha256(store_preimage)
        or store_output.get("record_name_sha256")
        != _sha256(expected_store_target.encode("ascii"))
        or type(store_output.get("signature_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{128}", store_output["signature_hex"]) is None
        or any(store_output.get(name) is not True for name in ("file_fsync", "directory_fsync", "hardlink_no_replace"))
    ):
        _fail(FAIL_STORE, "STORE durable IF_ABSENT output drift")
    store_signature = Phase2BC1DetachedSignatureV1(
        purpose_id=3, role=Phase2BC1SixRoleV1.STORE_RECEIPT,
        signature_purpose=Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT,
        algorithm="Ed25519", key_epoch=PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
        signer_key_id=keys[2].key_id,
        signature=bytes.fromhex(store_output["signature_hex"]),
    )
    store_receipt = Phase2BC1StoreIfAbsentReceiptCandidateV1(
        body=store_body, store_signature=store_signature,
        store_receipt_id=phase2b_c1_store_receipt_id_v1(store_body, store_signature),
    )
    time_body = Phase2BC1LogicalGenesisTokenBodyV1(
        version=PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        context_id=context.context_id,
        authority_requirement_id=context.body.authority_requirement_id,
        actor_profile_content_id=context.body.actor_profile_content_id,
        campaign_id=request.campaign_id, attempt_id=request.attempt_id,
        execution_registry_id=registry.execution_registry_id,
        store_receipt_id=store_receipt.store_receipt_id,
        time_semantics=PHASE2B_C1_SIX_ROLE_WIRE_V1_TIME_SEMANTICS,
        previous_logical_sequence_or_null=None, logical_sequence=0,
        stage=Phase2BC1LogicalStageV1.PREREGISTERED, stage_index=0,
    )
    time_preimage = phase2b_c1_logical_genesis_signature_preimage_v1(
        time_body, signer_key_id=keys[3].key_id
    )
    time_payload = _canonical_json_bytes(
        {
            "logical_sequence": 0,
            "preimage_b64": base64.b64encode(time_preimage).decode("ascii"),
            "previous_logical_sequence_or_null": None,
        }
    )
    time_output = _record_actor_exec_v1(
        control, root, roles[3], operation_name="TIME_GENESIS_AND_SIGN",
        worker_operation="time-genesis-and-sign", stdin_payload=time_payload,
        bindings=operations, failure_code=FAIL_TIME,
    )
    expected_time_fields = {
        "operation", "purpose_id", "role", "previous_logical_sequence_or_null",
        "logical_sequence", "record_sha256", "preimage_sha256", "signature_hex",
        "file_fsync", "directory_fsync", "hardlink_no_replace",
    }
    expected_time_record = _canonical_json_bytes(
        {
            "logical_sequence": 0,
            "preimage_sha256": _sha256(time_preimage),
            "previous_logical_sequence_or_null": None,
        }
    ) + b"\n"
    if (
        set(time_output) != expected_time_fields
        or time_output.get("operation") != "TIME_GENESIS_SIGNED"
        or time_output.get("purpose_id") != 4
        or time_output.get("role") != roles[3].role.value
        or time_output.get("previous_logical_sequence_or_null") is not None
        or time_output.get("logical_sequence") != 0
        or time_output.get("preimage_sha256") != _sha256(time_preimage)
        or time_output.get("record_sha256") != _sha256(expected_time_record)
        or type(time_output.get("signature_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{128}", time_output["signature_hex"]) is None
        or any(time_output.get(name) is not True for name in ("file_fsync", "directory_fsync", "hardlink_no_replace"))
    ):
        _fail(FAIL_TIME, "TIME durable logical-genesis output drift")
    time_signature = Phase2BC1DetachedSignatureV1(
        purpose_id=4, role=Phase2BC1SixRoleV1.TIME_ATTESTOR,
        signature_purpose=Phase2BC1SignaturePurposeV1.DECLARED_TIME_TOKEN,
        algorithm="Ed25519", key_epoch=PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH,
        signer_key_id=keys[3].key_id,
        signature=bytes.fromhex(time_output["signature_hex"]),
    )
    time_token = Phase2BC1LogicalGenesisTokenCandidateV1(
        body=time_body, time_signature=time_signature,
        logical_genesis_token_id=phase2b_c1_logical_genesis_token_id_v1(
            time_body, time_signature
        ),
    )
    wire = verify_phase2b_c1_six_role_wire_v1(
        context=context, registry=registry, store_receipt=store_receipt,
        logical_genesis_token=time_token,
    )
    if (
        type(wire) is not Phase2BC1SixRoleWireVerificationV1
        or wire.disposition
        is not Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY
        or wire.result_id is None
        or wire.partial_output_published is not False
    ):
        _fail(FAIL_WIRE, "existing six-role wire rejected the live graph")
    return (
        context, registry, store_receipt, time_token, wire,
        registry_sha, time_output["record_sha256"],
    )


def _cleanup_containers_best_effort_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    seccomp: _SeccompBindingV1,
    targets: Sequence[_ContainerTargetV1],
    owned: dict[str, _OwnedContainerV1],
    ambiguous: set[str],
) -> None:
    failures: list[str] = []
    with _ceremony_cleanup_phase_v1():
        for target in reversed(tuple(targets)):
            candidate = owned.get(target.name)
            managed_by_this_call = candidate is not None or target.name in ambiguous
            if candidate is None and target.name in ambiguous:
                # Repeated exact absence observations across one complete
                # Docker-command timeout close the bounded late-daemon window.
                # A materialized exact-owned object is registered;
                # foreign/malformed collisions are preserved.
                for attempt in range(_AMBIGUOUS_RECONCILIATION_OBSERVATIONS):
                    try:
                        reconciled = _reconcile_container_target_v1(
                            control, root, seccomp, target, owned,
                            code=FAIL_CLEANUP, expected_state=None,
                        )
                        if reconciled is not None:
                            candidate = reconciled[0]
                            ambiguous.discard(target.name)
                            break
                    except BaseException as exc:
                        failures.append(
                            f"container-reconcile:{target.name}:{type(exc).__name__}"
                        )
                        break
                    if attempt + 1 < _AMBIGUOUS_RECONCILIATION_OBSERVATIONS:
                        try:
                            pause = min(
                                _AMBIGUOUS_RECONCILIATION_INTERVAL_SECONDS,
                                _bounded_by_ceremony_deadline_v1(1),
                            )
                            time.sleep(pause)
                        except BaseException as exc:
                            failures.append(
                                f"container-reconcile-delay:{target.name}:"
                                f"{type(exc).__name__}"
                            )
                            break
            if candidate is not None:
                safe_to_remove = False
                try:
                    row = _inspect_container_reference_v1(
                        control, root, candidate.container_id,
                        code=FAIL_CLEANUP,
                        label="owned container pre-delete full-ID inspect",
                    )
                    if row is not None:
                        observed_id = _validate_container_row_v1(
                            row, candidate.target, seccomp,
                            code=FAIL_CLEANUP, expected_state=None,
                        )
                        safe_to_remove = observed_id == candidate.container_id
                    else:
                        owned.pop(target.name, None)
                except BaseException as exc:
                    failures.append(
                        f"container-predelete:{target.name}:{type(exc).__name__}"
                    )
                if safe_to_remove:
                    try:
                        removed = _run_docker_command_v1(
                            control.command(
                                "container", "rm", "--force", candidate.container_id
                            ),
                            environment=control.environment,
                            working_directory=root,
                            timeout_seconds=5,
                        )
                        if not (
                            removed.returncode == 0
                            and removed.stdout
                            == (candidate.container_id + "\n").encode("ascii")
                            and removed.stderr == b""
                        ):
                            failures.append(f"container-remove:{target.name}:output")
                    except BaseException as exc:
                        failures.append(
                            f"container-remove:{target.name}:{type(exc).__name__}"
                        )
                try:
                    by_id = _inspect_container_reference_v1(
                        control, root, candidate.container_id,
                        code=FAIL_CLEANUP,
                        label="owned container post-delete full-ID absence",
                    )
                    if by_id is not None:
                        failures.append(f"container-id-remains:{target.name}")
                    else:
                        owned.pop(target.name, None)
                except BaseException as exc:
                    failures.append(
                        f"container-id-absence:{target.name}:{type(exc).__name__}"
                    )
            try:
                by_name = _inspect_container_reference_v1(
                    control, root, target.name, code=FAIL_CLEANUP,
                    label="container post-delete exact-name absence",
                )
                if by_name is not None:
                    # Validate only to distinguish an exact-owned survivor in
                    # diagnostics.  Either exact-owned or foreign is retained;
                    # never issue a name-based delete.
                    try:
                        _validate_container_row_v1(
                            by_name, target, seccomp,
                            code=FAIL_CLEANUP, expected_state=None,
                        )
                        suffix = "owned-remains"
                    except BaseException:
                        suffix = "foreign-preserved"
                    if managed_by_this_call:
                        failures.append(f"container-name:{target.name}:{suffix}")
                else:
                    ambiguous.discard(target.name)
            except BaseException as exc:
                failures.append(
                    f"container-name-absence:{target.name}:{type(exc).__name__}"
                )
    if failures:
        _fail(
            FAIL_CLEANUP,
            f"container cleanup collected {len(failures)} failures: "
            + ",".join(failures),
        )


def _cleanup_volumes_best_effort_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    volume_targets: Sequence[tuple[str, tuple[tuple[str, str], ...]]],
    owned: dict[str, _OwnedVolumeV1],
    ambiguous: set[str],
) -> None:
    failures: list[str] = []
    with _ceremony_cleanup_phase_v1():
        for name, labels in reversed(tuple(volume_targets)):
            candidate = owned.get(name)
            managed_by_this_call = candidate is not None or name in ambiguous
            if candidate is None and name in ambiguous:
                for attempt in range(_AMBIGUOUS_RECONCILIATION_OBSERVATIONS):
                    try:
                        candidate = _reconcile_volume_target_v1(
                            control, root, volume_name=name, labels=labels,
                            owned=owned, code=FAIL_CLEANUP,
                        )
                        if candidate is not None:
                            ambiguous.discard(name)
                            break
                    except BaseException as exc:
                        failures.append(
                            f"volume-reconcile:{name}:{type(exc).__name__}"
                        )
                        break
                    if attempt + 1 < _AMBIGUOUS_RECONCILIATION_OBSERVATIONS:
                        try:
                            pause = min(
                                _AMBIGUOUS_RECONCILIATION_INTERVAL_SECONDS,
                                _bounded_by_ceremony_deadline_v1(1),
                            )
                            time.sleep(pause)
                        except BaseException as exc:
                            failures.append(
                                f"volume-reconcile-delay:{name}:"
                                f"{type(exc).__name__}"
                            )
                            break
            if candidate is not None:
                safe_to_remove = False
                try:
                    # Revalidate immediately before name-based volume removal;
                    # a same-name foreign replacement is never deleted.
                    row = _inspect_volume_reference_v1(
                        control, root, name, code=FAIL_CLEANUP,
                        label="owned volume pre-delete reinspection",
                    )
                    if row is not None:
                        _validate_owned_volume_row_v1(
                            row, volume_name=name,
                            labels=candidate.labels, code=FAIL_CLEANUP,
                        )
                        safe_to_remove = True
                    else:
                        owned.pop(name, None)
                except BaseException as exc:
                    failures.append(
                        f"volume-predelete:{name}:{type(exc).__name__}"
                    )
                if safe_to_remove:
                    try:
                        removed = _run_docker_command_v1(
                            control.command("volume", "rm", "--force", name),
                            environment=control.environment,
                            working_directory=root,
                            timeout_seconds=5,
                        )
                        if not (
                            removed.returncode == 0
                            and removed.stdout == (name + "\n").encode("ascii")
                            and removed.stderr == b""
                        ):
                            failures.append(f"volume-remove:{name}:output")
                    except BaseException as exc:
                        failures.append(
                            f"volume-remove:{name}:{type(exc).__name__}"
                        )
            try:
                row = _inspect_volume_reference_v1(
                    control, root, name, code=FAIL_CLEANUP,
                    label="volume post-delete exact-name absence",
                )
                if row is not None:
                    try:
                        _validate_owned_volume_row_v1(
                            row, volume_name=name, labels=labels,
                            code=FAIL_CLEANUP,
                        )
                        suffix = "owned-remains"
                    except BaseException:
                        suffix = "foreign-preserved"
                    if managed_by_this_call:
                        failures.append(f"volume-name:{name}:{suffix}")
                else:
                    owned.pop(name, None)
                    ambiguous.discard(name)
            except BaseException as exc:
                failures.append(
                    f"volume-absence:{name}:{type(exc).__name__}"
                )
    if failures:
        _fail(
            FAIL_CLEANUP,
            f"volume cleanup collected {len(failures)} failures: "
            + ",".join(failures),
        )


def _prove_volumes_retained_v1(
    control: LocalDockerControlPlaneV1,
    root: Path,
    volume_targets: Sequence[tuple[str, tuple[tuple[str, str], ...]]],
    owned: Mapping[str, _OwnedVolumeV1],
) -> None:
    if (
        len(volume_targets) != 11
        or len({name for name, _ in volume_targets}) != 11
        or set(owned) != {name for name, _ in volume_targets}
    ):
        _fail(FAIL_CLEANUP, "retained private-volume locator set drift")
    for name, labels in volume_targets:
        row = _inspect_volume_reference_v1(
            control, root, name, code=FAIL_CLEANUP,
            label="retained private-volume inspect",
        )
        if row is None:
            _fail(FAIL_CLEANUP, "retained private volume became absent")
        _validate_owned_volume_row_v1(
            row, volume_name=name, labels=labels, code=FAIL_CLEANUP
        )


def _issue_receipt_v1(
    *,
    request: Phase2BC1LiveDockerCeremonyRequestV1,
    context: Phase2BC1CeremonyContextV1,
    registry: Phase2BC1ExecutionRegistryCandidateV1,
    store_receipt: Phase2BC1StoreIfAbsentReceiptCandidateV1,
    time_token: Phase2BC1LogicalGenesisTokenCandidateV1,
    wire: Phase2BC1SixRoleWireVerificationV1,
    runtime_bindings: tuple[Phase2BC1LiveRoleRuntimeBindingV1, ...],
    operation_bindings: tuple[Phase2BC1LiveOperationBindingV1, ...],
    control_plane_sha256: str,
    recognizer_image_inspect_sha256: str,
    scorer_image_inspect_sha256: str,
    stored_registry_sha256: str,
    logical_genesis_record_sha256: str,
) -> Phase2BC1LiveDockerCeremonyReceiptV1:
    identity_payload = {
        "version": PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_VERSION,
        "context_id": context.context_id,
        "attempt_id": request.attempt_id,
        "execution_registry_id": registry.execution_registry_id,
        "store_receipt_id": store_receipt.store_receipt_id,
        "logical_genesis_token_id": time_token.logical_genesis_token_id,
        "wire_result_id": wire.result_id,
        "runtime_bindings": [
            {
                "purpose_id": row.purpose_id,
                "role": row.role.value,
                "image_digest": row.image_digest,
                "container_name": row.container_name,
                "container_id": row.container_id,
                "purpose_private_volumes": row.purpose_private_volumes,
                "qualification_sha256": row.qualification_sha256,
                "namespace_identities": row.namespace_identities,
                "openssl_sha256": row.openssl_sha256,
                "openssl_version": row.openssl_version,
            }
            for row in runtime_bindings
        ],
        "operation_bindings": [
            {
                "purpose_id": row.purpose_id,
                "role": row.role.value,
                "operation": row.operation,
                "command_sha256": row.command_sha256,
                "stdin_sha256": row.stdin_sha256,
                "stdout_sha256": row.stdout_sha256,
                "stderr_sha256": row.stderr_sha256,
                "started_monotonic_ns": row.started_monotonic_ns,
                "finished_monotonic_ns": row.finished_monotonic_ns,
            }
            for row in operation_bindings
        ],
        "control_plane_sha256": control_plane_sha256,
        "recognizer_image_inspect_sha256": recognizer_image_inspect_sha256,
        "scorer_image_inspect_sha256": scorer_image_inspect_sha256,
        "stored_registry_sha256": stored_registry_sha256,
        "logical_genesis_record_sha256": logical_genesis_record_sha256,
    }
    receipt_id = (
        "phase2b_c1_live_docker_ceremony_receipt_v1_"
        + _sha256(
            b"HEGEL/PHASE2B/C1/LIVE_DOCKER_CEREMONY_RECEIPT/V1\x00"
            + _canonical_json_bytes(identity_payload)
        )
    )
    values: list[tuple[str, object]] = [
        ("ceremony_receipt_id", receipt_id),
        ("version", PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_VERSION),
        ("claim_level", PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_CLAIM_LEVEL),
        ("reason", PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_COMPLETE_REASON),
        ("authority_class", PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS),
        ("request", request), ("context", context), ("registry", registry),
        ("store_receipt", store_receipt), ("logical_genesis_token", time_token),
        ("wire_verification", wire), ("role_runtime_bindings", runtime_bindings),
        ("operation_bindings", operation_bindings),
        ("docker_control_plane_binding_sha256", control_plane_sha256),
        ("recognizer_image_inspect_sha256", recognizer_image_inspect_sha256),
        ("scorer_image_inspect_sha256", scorer_image_inspect_sha256),
        ("openssl_sha256", OPENSSL_SHA256_V1),
        ("openssl_version", OPENSSL_VERSION_V1),
        ("stored_registry_sha256", stored_registry_sha256),
        ("logical_genesis_record_sha256", logical_genesis_record_sha256),
    ]
    values.extend((name, True) for name in _TRUE_RESULT_CLAIMS)
    values.extend((name, False) for name in _FALSE_RESULT_CLAIMS)
    expected_fields = tuple(item.name for item in fields(Phase2BC1LiveDockerCeremonyReceiptV1))
    if tuple(name for name, _ in values) != expected_fields:
        raise RuntimeError("live ceremony receipt issuance field drift")
    result = Phase2BC1LiveDockerCeremonyReceiptV1(_RESULT_ISSUANCE_TOKEN)
    for name, value in values:
        object.__setattr__(result, name, value)
    result.__post_init__()
    return result


def execute_phase2b_c1_live_docker_ceremony_v1(
    request: Phase2BC1LiveDockerCeremonyRequestV1,
    *,
    runtime_parent: Path = Path("/tmp"),
) -> Phase2BC1LiveDockerCeremonyReceiptV1:
    """Execute and close one owner-accepted six-role Docker bootstrap.

    No external qualification receipt is accepted.  Every qualification fact
    is observed from the six simultaneously live containers in this call.
    Successful calls retain only the opaque named volumes and public result;
    all six containers are removed and proved absent before issuance.
    """

    if type(request) is not Phase2BC1LiveDockerCeremonyRequestV1:
        _fail(FAIL_PREFLIGHT, "executor requires exact live ceremony request")
    if type(runtime_parent) is not type(Path("/")) or not runtime_parent.is_absolute():
        _fail(FAIL_PREFLIGHT, "runtime parent must be exact absolute Path")
    if _CEREMONY_DEADLINES_V1.get() is None:
        # Enter before request/profile/seccomp preflight.  The recursive call
        # sees the active context and the lifecycle's nested scope is re-entrant,
        # so one wall-clock budget owns the complete public call.
        with _ceremony_deadline_scope_v1():
            return execute_phase2b_c1_live_docker_ceremony_v1(
                request, runtime_parent=runtime_parent
            )
    request = _preflight_request_v1(request)
    source_seccomp = _load_seccomp_v1()

    with _ceremony_deadline_scope_v1():
        with _prepared_runtime_v1(runtime_parent) as (runtime_root, control):
            _validate_control_plane_v1(control, runtime_root)
            seccomp = _stage_seccomp_v1(source_seccomp, runtime_root)
            control_sha = _sha256(_canonical_json_bytes(dict(control.binding)))
            recognizer_image = _inspect_image_v1(
                control, runtime_root,
                image_digest=request.recognizer_image_digest,
                source_root_sha256=request.recognizer_source_root_sha256,
                role_kind="recognizer",
            )
            scorer_image = _inspect_image_v1(
                control, runtime_root,
                image_digest=request.scorer_image_digest,
                source_root_sha256=request.scorer_source_root_sha256,
                role_kind="scorer",
            )
            run_token = secrets.token_hex(12)
            roles = _make_role_states_v1(
                request, run_token,
                recognizer_image=recognizer_image,
                scorer_image=scorer_image,
            )
            raw_volume_plan = sorted(
                (
                    (role.purpose_id, mount_path, volume_name)
                    for role in roles
                    for mount_path, volume_name in role.volumes
                ),
                key=lambda row: (
                    0 if row[:2] == (3, "/ledger") else 1,
                    row[0], row[1],
                ),
            )
            volume_targets = tuple(
                (name, _volume_labels_v1(run_token, name))
                for _, _, name in raw_volume_plan
            )
            initializer_targets = tuple(
                _initializer_target_v1(
                    volume_name=name, image=scorer_image, run_token=run_token
                )
                for _, _, name in raw_volume_plan
            )
            actor_targets = tuple(_actor_target_v1(role, run_token) for role in roles)
            all_container_targets = (*initializer_targets, *actor_targets)
            if len({target.name for target in all_container_targets}) != 17:
                raise RuntimeError("ceremony container target identity collision")

            owned_volumes: dict[str, _OwnedVolumeV1] = {}
            owned_containers: dict[str, _OwnedContainerV1] = {}
            ambiguous_volumes: set[str] = set()
            ambiguous_containers: set[str] = set()
            primary_error: BaseException | None = None
            completed: tuple[object, ...] | None = None
            operations: list[Phase2BC1LiveOperationBindingV1] = []
            try:
                for (_, _, volume_name), initializer_target in zip(
                    raw_volume_plan, initializer_targets, strict=True
                ):
                    _provision_volume_v1(
                        control, runtime_root, seccomp,
                        volume_name=volume_name,
                        initializer_target=initializer_target,
                        run_token=run_token,
                        owned_volumes=owned_volumes,
                        owned_containers=owned_containers,
                        ambiguous_volumes=ambiguous_volumes,
                        ambiguous_containers=ambiguous_containers,
                    )
                created_roles: list[_RoleStateV1] = []
                for role in roles:
                    created_roles.append(
                        _create_actor_v1(
                            control, runtime_root, seccomp, role, run_token,
                            owned_containers, ambiguous_containers,
                        )
                    )
                roles = tuple(created_roles)
                if len({role.container_id for role in roles}) != 6:
                    _fail(
                        FAIL_CONTAINER_LIFECYCLE,
                        "actor container full IDs are not distinct",
                    )
                for role in roles:
                    _start_actor_v1(control, runtime_root, role)
                for role in roles:
                    _inspect_actor_v1(
                        control, runtime_root, seccomp, role, run_token
                    )
                roles, runtime_bindings = _qualify_all_roles_v1(
                    control, runtime_root, seccomp, roles, operations, run_token
                )
                keys = _generate_role_keys_v1(
                    control, runtime_root, roles, operations
                )
                graph = _build_live_graph_v1(
                    control, runtime_root, request, roles, keys, operations
                )
                if len(operations) != 15:
                    raise RuntimeError("live ceremony operation count drift")
                completed = (*graph, runtime_bindings, tuple(operations))
            except BaseException as exc:
                primary_error = exc

            container_cleanup_error: BaseException | None = None
            try:
                _cleanup_containers_best_effort_v1(
                    control, runtime_root, seccomp,
                    all_container_targets, owned_containers,
                    ambiguous_containers,
                )
            except BaseException as exc:
                container_cleanup_error = exc

            retention_error: BaseException | None = None
            if primary_error is None and container_cleanup_error is None:
                try:
                    _prove_volumes_retained_v1(
                        control, runtime_root, volume_targets, owned_volumes
                    )
                except BaseException as exc:
                    retention_error = exc

            if (
                primary_error is not None
                or container_cleanup_error is not None
                or retention_error is not None
            ):
                volume_cleanup_error: BaseException | None = None
                try:
                    _cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as exc:
                    volume_cleanup_error = exc
                cleanup_causes = tuple(
                    exc for exc in (
                        container_cleanup_error, retention_error,
                        volume_cleanup_error,
                    ) if exc is not None
                )
                if cleanup_causes:
                    _fail(
                        FAIL_CLEANUP,
                        "ceremony cleanup failed after exhaustive pass: "
                        + ",".join(type(exc).__name__ for exc in cleanup_causes),
                    )
                if primary_error is None:
                    raise RuntimeError("primary ceremony failure issuance drift")
                raise primary_error

            if completed is None or len(owned_volumes) != 11:
                try:
                    _cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as exc:
                    _fail(
                        FAIL_CLEANUP,
                        f"incomplete success cleanup failed: {type(exc).__name__}",
                    )
                _fail(FAIL_CLEANUP, "success graph/volume handoff incomplete")
            (
                context, registry, store_receipt, time_token, wire,
                stored_registry_sha, logical_record_sha, runtime_bindings,
                operation_bindings,
            ) = completed
            if (
                type(context) is not Phase2BC1CeremonyContextV1
                or type(registry) is not Phase2BC1ExecutionRegistryCandidateV1
                or type(store_receipt) is not Phase2BC1StoreIfAbsentReceiptCandidateV1
                or type(time_token) is not Phase2BC1LogicalGenesisTokenCandidateV1
                or type(wire) is not Phase2BC1SixRoleWireVerificationV1
                or type(runtime_bindings) is not tuple
                or type(operation_bindings) is not tuple
                or type(stored_registry_sha) is not str
                or type(logical_record_sha) is not str
            ):
                try:
                    _cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as exc:
                    _fail(
                        FAIL_CLEANUP,
                        f"shape-drift cleanup failed: {type(exc).__name__}",
                    )
                _fail(FAIL_CLEANUP, "closed success graph primitive shape drift")
            try:
                return _issue_receipt_v1(
                    request=request, context=context, registry=registry,
                    store_receipt=store_receipt, time_token=time_token, wire=wire,
                    runtime_bindings=runtime_bindings,
                    operation_bindings=operation_bindings,
                    control_plane_sha256=control_sha,
                    recognizer_image_inspect_sha256=(
                        recognizer_image.inspect_sha256
                    ),
                    scorer_image_inspect_sha256=scorer_image.inspect_sha256,
                    stored_registry_sha256=stored_registry_sha,
                    logical_genesis_record_sha256=logical_record_sha,
                )
            except BaseException as receipt_error:
                try:
                    _cleanup_volumes_best_effort_v1(
                        control, runtime_root, volume_targets, owned_volumes,
                        ambiguous_volumes,
                    )
                except BaseException as cleanup_error:
                    _fail(
                        FAIL_CLEANUP,
                        "receipt issuance cleanup failed after exhaustive pass: "
                        + type(cleanup_error).__name__,
                    )
                raise receipt_error


if tuple(item[0] for item in _ROLE_MOUNTS) != tuple(range(1, 7)):
    raise RuntimeError("live ceremony purpose order drift")
if tuple(item[1] for item in _ROLE_MOUNTS) != PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER:
    raise RuntimeError("live ceremony role order drift")
# Paths intentionally repeat only /state; each role receives a distinct opaque
# volume despite the common in-container destination.
if {path for _, _, paths in _ROLE_MOUNTS for path in paths} != {
    "/state", "/custody", "/ledger", "/cas", "/prediction", "/score"
}:
    raise RuntimeError("live ceremony mount policy drift")
if set(_TRUE_RESULT_CLAIMS) & set(_FALSE_RESULT_CLAIMS):
    raise RuntimeError("live ceremony claim polarity overlap")


__all__ = (
    "FAIL_CLEANUP", "FAIL_CONTAINER_LIFECYCLE", "FAIL_DOCKER_CAPTURE",
    "FAIL_DOCKER_CONTROL_PLANE", "FAIL_DOCKER_TIMEOUT", "FAIL_IMAGE_BINDING",
    "FAIL_KEYGEN", "FAIL_PREFLIGHT", "FAIL_QUALIFICATION", "FAIL_SECCOMP",
    "FAIL_SIGNATURE", "FAIL_STORE", "FAIL_TIME", "FAIL_VOLUME_PROVISIONING",
    "FAIL_WIRE", "OPENSSL_EXECUTABLE_V1", "OPENSSL_SHA256_V1",
    "OPENSSL_VERSION_V1", "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_CLAIM_LEVEL",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_COMPLETE_REASON",
    "PHASE2B_C1_LIVE_DOCKER_CEREMONY_V1_VERSION",
    "RECOGNIZER_RUNTIME_IMAGE_DIGEST_V1", "RECOGNIZER_SOURCE_ROOT_SHA256_V1",
    "SCORER_RUNTIME_IMAGE_DIGEST_V1", "SCORER_SOURCE_ROOT_SHA256_V1",
    "Phase2BC1LiveDockerCeremonyErrorV1",
    "Phase2BC1LiveDockerCeremonyReceiptV1",
    "Phase2BC1LiveDockerCeremonyRequestV1",
    "Phase2BC1LiveOperationBindingV1", "Phase2BC1LiveRoleRuntimeBindingV1",
    "execute_phase2b_c1_live_docker_ceremony_v1",
)

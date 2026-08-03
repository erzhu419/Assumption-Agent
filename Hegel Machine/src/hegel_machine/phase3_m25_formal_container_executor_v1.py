"""Owner-authorized, fail-closed M2.5 formal container ceremony executor.

This module is the executable boundary between ``14/24 / NOT_RUN`` and a
public, replayable ``24/24 / NOT_RUN`` bundle.  Merely importing it, running a
dry-run, or running its synthetic backend cannot generate a key or seed and
cannot promote a formal gate. The real first-run path is available only
through the explicit ``execute`` API and only after Commit-A evidence binds
*runnable M3 closure enumerators*. Static wire/hash replayers are intentionally
rejected as execution implementation bindings. A separately named recovery
API can reopen an already durable PENDING transaction; it never runs from
ordinary execute and permits a first CSPRNG call only when all seed-intent
artifacts are provably absent.

The four actor containers stay alive together for the whole ceremony.  Each
has a purpose-private, run-labelled ``/state`` volume for its Ed25519 private
key and explicit crash recovery. Only purpose 1 receives the narrow,
repository-external ``/custody`` bind holding
the O_EXCL transaction metadata, marker, durable generation intent/completion
receipt and one 0600 raw seed.  Actor workers are purpose-scoped replayers,
not generic signing oracles.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from dataclasses import dataclass, fields as dataclass_fields
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn, Sequence

from .phase3_m25_container_ceremony_v1 import (
    AUTHORITY_CLASS,
    TECHNICAL_ACTOR_DISCLOSURE_V1,
    GateEvidenceInputsV1,
    M25ContainerCeremonyError,
    build_actor_key_manifest_fields_v1,
    build_single_signature_envelope_fields_v1,
    complete_marker_v1,
    create_pending_marker_v1,
    evaluate_gates_15_24_v1,
    parse_ed25519_spki_der_v1,
    promote_gate_evidence_v1,
    read_marker_snapshot_v1,
    require_full_split_response_agreement_v2,
    validate_ceremony_admission_v1,
)
from .phase3_m25_external_v1 import MarkerSnapshot, assert_public_payload_contains_no_secret_fields
from .phase3_m25_formal_static_basis_v1 import (
    FORMAL_GIT_EXECUTABLE,
    FormalStaticBasisV1,
    build_formal_static_basis_v1,
    build_python_static_replay_receipt_v1,
    formal_git_environment_v1,
    run_rust_static_replay_receipt_v1,
    validate_dual_static_replay_receipts_v1,
)
from .phase3_m25_parent_absence_audit_v1 import (
    ParentAbsenceAuditEvidence,
    build_parent_absence_attestation_fields_v2,
    generate_parent_absence_audit_v1,
    replay_parent_absence_audit_v1,
)
from .phase3_m25_purpose4_detached_audit_v1 import (
    DetachedParentSnapshotV1,
    Purpose4DetachedAuditError,
    _runtime_inventory as _purpose4_runtime_inventory_v1,
    _runtime_source_bindings_v1 as _purpose4_runtime_source_bindings_v1,
    _set_snapshot_read_only as _set_purpose4_snapshot_read_only_v1,
    prepare_detached_parent_snapshot_v1,
    validate_detached_parent_snapshot_v1,
)
from .phase3_m25_purpose4_keybearing_detached_v1 import (
    MAX_RESPONSE_BYTES as PURPOSE4_KEYBEARING_MAX_RESPONSE_BYTES,
    Purpose4KeyBearingError,
    build_purpose4_keybearing_request_v1,
    canonical_json_v1 as purpose4_canonical_json_v1,
    validate_purpose4_keybearing_response_v1,
)
from .phase3_m25_bridge_full_dag_replay_v1 import (
    BridgeDagReplayError,
    make_openssl_ed25519_verifier_v1,
    replay_bridge_dag_package_v1,
    validate_bridge_actor_replay_receipt_v1,
)
from .phase3_m25_bridge_dag_node_builder_v1 import (
    BridgeDagNodeBuildError,
    BridgeDagNodeBuildInputsV1,
    BridgeDagPackageBuildInputsV1,
    M3ExecutionBindingContractFieldsV1,
    build_bridge_dag_replay_package_from_inputs_v1,
)
from .phase3_m25_bridge_dag_binary_qualification_v1 import (
    BridgeDagBinaryQualificationError,
    DEFAULT_RUST_BRIDGE_DAG_BINARY,
    DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
    load_qualified_rust_bridge_dag_binary_binding_v1,
)
from .phase3_m25_rows_v1 import generate_odd_role_rows_v1, generate_sink_role_rows_v1
from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    build_local_docker_daemon_identity_receipt_v1,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
    validate_linux_local_durable_custody_location_v1,
    validate_linux_local_durable_custody_v1,
)
from .phase3_m3_implementation_qualification_v1 import (
    M3ImplementationQualificationError,
    build_qualified_formal_static_basis_v1,
    validate_m3_execution_implementation_bindings_v1,
)
from .phase3_m25_wire_v1 import (
    LEGACY_PARENT_SOURCE_IDS,
    M3_RUN_OUTPUT_ROOTS,
    OBJECT_TAGS,
    bridge_attestation_signature_preimage_v1,
    candidate_content_root,
    candidate_record_tree_root,
    encode_formal_object,
    external_signature_preimage_v1,
    git_sha1_commit_id,
    id_digest_v1,
)
from .strict_cbor_v1 import rfc6962_root


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = PROJECT_ROOT.parent
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase3_container_actor_profile_v1.json"
SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_internal_actor_seccomp_v1.json"
BUILD_SECCOMP_PATH: Final = PROJECT_ROOT / "config/phase3_m3_offline_build_seccomp_v1.json"
PYTHON_LIVE_PROBE_PATH: Final = PROJECT_ROOT / "tools/phase3_container_actor_probe_v1.py"
RUST_LIVE_PROBE_PATH: Final = PROJECT_ROOT / "tools/phase3_container_actor_probe_v1.rs"
PYTHON_WORKER_PATH: Final = PROJECT_ROOT / "tools/phase3_m25_formal_actor_worker_v1.py"
PYTHON_BRIDGE_WORKER_PATH: Final = (
    PROJECT_ROOT / "tools/phase3_m25_python_bridge_actor_worker_v1.py"
)
PARENT_AUDITOR_WORKER_PATH: Final = (
    PROJECT_ROOT / "tools/phase3_m25_parent_auditor_actor_worker_v1.py"
)
RUST_WORKER_PATH: Final = PROJECT_ROOT / "tools/phase3_m25_formal_rust_actor_worker_v1.sh"
PYTHON_SPLIT_PATH: Final = PROJECT_ROOT / "tools/phase3_split_partition_calculator_fd3_v1.py"
SEED_CUSTODY_VERIFIER_PATH: Final = (
    PROJECT_ROOT / "tools/phase3_m25_seed_custody_verifier_v1.py"
)
RUST_SPLIT_SOURCE_PATH: Final = PROJECT_ROOT / "tools/phase3_split_partition_calculator_fd3_v1.rs"
LIVE_ACTOR_PROTOCOL_QUALIFICATION_REPORT_PATH: Final = (
    PROJECT_ROOT
    / "artifacts/phase3_m25_external/"
    "phase3_m25_live_actor_protocol_qualification_v1.json"
)

PURPOSE4_RUNTIME_SOURCE_SPECS: Final = (
    (
        "Hegel Machine/src/hegel_machine/phase3_m25_purpose4_keybearing_detached_v1.py",
        "hegel_machine/phase3_m25_purpose4_keybearing_detached_v1.py",
    ),
    (
        "Hegel Machine/src/hegel_machine/phase3_m25_purpose4_detached_audit_v1.py",
        "hegel_machine/phase3_m25_purpose4_detached_audit_v1.py",
    ),
    (
        "Hegel Machine/src/hegel_machine/phase3_m25_parent_absence_audit_v1.py",
        "hegel_machine/phase3_m25_parent_absence_audit_v1.py",
    ),
    (
        "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
        "hegel_machine/phase3_m25_wire_v1.py",
    ),
    (
        "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
        "hegel_machine/strict_cbor_v1.py",
    ),
    (
        "Hegel Machine/src/hegel_machine/phase3_local_runtime_v1.py",
        "hegel_machine/phase3_local_runtime_v1.py",
    ),
    (
        "Hegel Machine/tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py",
        "tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py",
    ),
    (
        "Hegel Machine/tools/phase3_m25_actor_operation_probe_v1.py",
        "tools/phase3_m25_actor_operation_probe_v1.py",
    ),
    (
        "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
        "tools/phase3_container_actor_probe_v1.py",
    ),
    (
        "Hegel Machine/config/phase3_container_actor_profile_v1.json",
        "control/phase3_container_actor_profile_v1.json",
    ),
    (
        "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json",
        "control/phase3_internal_actor_seccomp_v1.json",
    ),
)

ACTOR_SNAPSHOT_PATHS_BY_PURPOSE: Final = MappingProxyType({
    1: (
        "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_bridge_full_dag_replay_v1.py",
        "Hegel Machine/tools/phase3_m25_formal_actor_worker_v1.py",
        "Hegel Machine/tools/phase3_split_partition_calculator_fd3_v1.py",
        "Hegel Machine/tools/phase3_split_partition_calculator_fd3_v1.rs",
        "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
        "Hegel Machine/tools/phase3_m25_actor_operation_probe_v1.py",
    ),
    2: (
        "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_bridge_full_dag_replay_v1.py",
        "Hegel Machine/tools/phase3_m25_python_bridge_actor_worker_v1.py",
        "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
        "Hegel Machine/tools/phase3_m25_actor_operation_probe_v1.py",
    ),
    3: (
        "Hegel Machine/tools/phase3_m25_formal_rust_actor_worker_v1.sh",
        "Hegel Machine/tools/phase3_container_actor_probe_v1.rs",
    ),
    4: (
        "Hegel Machine/src/hegel_machine/strict_cbor_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_wire_v1.py",
        "Hegel Machine/src/hegel_machine/phase3_m25_parent_absence_audit_v1.py",
        "Hegel Machine/tools/phase3_m25_parent_auditor_actor_worker_v1.py",
        "Hegel Machine/tools/phase3_container_actor_probe_v1.py",
        "Hegel Machine/tools/phase3_m25_actor_operation_probe_v1.py",
    ),
})

PUBLIC_REPLAY_SCHEMA: Final = "hegel-phase3-m25-public-gate-evidence-replay/1"
READINESS_SCHEMA: Final = "hegel-phase3-m25-formal-container-readiness/2"
SYNTHETIC_SCHEMA: Final = "hegel-phase3-m25-synthetic-container-ceremony/1"
TRANSACTION_JOURNAL_SCHEMA: Final = "hegel-phase3-m25-ceremony-transaction-journal/1"
TRANSACTION_LOCK_SCHEMA: Final = "hegel-phase3-m25-persistent-ceremony-lock/4"
OUTPUT_RESERVATION_SCHEMA: Final = "hegel-phase3-m25-public-output-reservation/2"
PUBLICATION_RECEIPT_SCHEMA: Final = "hegel-phase3-m25-publication-receipt/1"
PRESTAGE_INTENT_SCHEMA: Final = "hegel-phase3-m25-prestage-intent/1"
ACTOR_TRUST_CHECKPOINT_SCHEMA: Final = (
    "hegel-phase3-m25-actor-trust-checkpoint/1"
)
OPERATION_PROBE_SCHEMA: Final = "hegel-phase3-m25-operation-bound-live-probe/1"
RUST_OPERATION_PARENT_SCHEMA: Final = (
    "hegel-phase3-m25-rust-operation-parent-binding/1"
)
HOST_OPERATION_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-m25-host-verified-operation-bound-live-probe/2"
)
SEED_CUSTODY_VERIFICATION_SCHEMA: Final = (
    "hegel-phase3-m25-keyless-seed-custody-verification/1"
)
SEED_CUSTODY_INNER_VERIFICATION_SCHEMA: Final = (
    "hegel-phase3-m25-keyless-seed-custody-inner-verification/1"
)
COMMIT_B_EVIDENCE_BASENAME: Final = "phase3_m25_formal_gate_evidence_v1.json"
COMMIT_B_PROMOTION_BASENAME: Final = "phase3_m25_gate_promotion_v1.json"
SPLIT_VERSION_DIGEST: Final = id_digest_v1("hegel-split-contract-p3a-v1.1.2")

_ACTOR_BASE_ENVIRONMENT_KEYS: Final = frozenset({
    "HEGEL_ACTOR_IMAGE_REF",
    "HEGEL_ACTOR_PROFILE_ID",
    "HEGEL_BASIS_COMMIT",
    "HEGEL_DAEMON_RECEIPT_SHA256",
    "HEGEL_HOST_REPOSITORY_PATH_SHA256",
    "HEGEL_PROFILE_SHA256",
    "HEGEL_PURPOSE_ID",
    "HEGEL_RUN_ID",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
})
_ACTOR_PRIVATE_ENVIRONMENT_KEYS: Final = frozenset({
    "HEGEL_HOST_REPOSITORY_PATH",
})
_ACTOR_OPERATION_ENVIRONMENT_KEYS: Final = frozenset({
    "HEGEL_OPERATION_ID",
    "HEGEL_OPERATION_NONCE",
    "HEGEL_OPERATION_REQUEST_SHA256",
    "HEGEL_OPERATION_SEQUENCE",
    "HEGEL_PROBE_INPUT_WRITE_PATH",
})
_ACTOR_OPERATIONS_BY_PURPOSE: Final = MappingProxyType({
    1: frozenset({
        "qualify-only",
        "keygen",
        "keygen-resume",
        "purpose1-authorized-sign",
        "bridge-replay-sign-python",
        "seed-split-real",
        "seed-split-resume",
        "seed-split-synthetic",
        "complete-marker",
    }),
    2: frozenset({
        "qualify-only", "keygen", "keygen-resume", "bridge-replay-sign-python"
    }),
    3: frozenset({
        "qualify-only", "keygen", "keygen-resume", "bridge-replay-sign-rust"
    }),
    4: frozenset({
        "qualify-only", "keygen", "keygen-resume", "purpose4-parent-sign"
    }),
})

FAIL_EXECUTION_BINDINGS = "FAIL_M25_M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY"
FAIL_PREFLIGHT = "FAIL_M25_FORMAL_CONTAINER_PREFLIGHT"
FAIL_CONTAINER = "FAIL_M25_FORMAL_CONTAINER_RUNTIME"
FAIL_CUSTODY = "FAIL_M25_FORMAL_CUSTODY_STATE"
FAIL_PUBLICATION = "FAIL_M25_FORMAL_PUBLICATION"
FAIL_SYNTHETIC_PROMOTION = "FAIL_M25_SYNTHETIC_FORMAL_PROMOTION_FORBIDDEN"
FAIL_TRANSACTION_LOCK = "FAIL_M25_FORMAL_CEREMONY_LOCKED_OR_RESERVED"
FAIL_PURPOSE4_REPLAY_UNRESOLVED = "FAIL_M25_PURPOSE4_DETACHED_GIT_REPLAY_NOT_IMPLEMENTED"
FAIL_BRIDGE_REPLAY_UNRESOLVED = "FAIL_M25_BRIDGE_FULL_DAG_REPLAY_NOT_IMPLEMENTED"
FAIL_ACTOR_LIVE_PROBE_UNRESOLVED = "FAIL_M25_SIGNER_ACTOR_LIVE_PROBE_NOT_BOUND"
FAIL_ACTOR_PROTOCOL_QUALIFICATION_UNRESOLVED = (
    "FAIL_M25_LIVE_ACTOR_PROTOCOL_QUALIFICATION_NOT_READY"
)
FAIL_POST_STAGE_RECOVERY_UNRESOLVED = (
    "FAIL_M25_POST_STAGE_TRANSACTION_RECOVERY_NOT_FROZEN"
)
FAIL_PRESTAGE_RECOVERY_UNRESOLVED = (
    "FAIL_M25_PRESTAGE_TRANSACTION_RECOVERY_NOT_FROZEN"
)

# Source-level controls are implemented.  The Docker backend still advertises
# ``FAIL_BRIDGE_REPLAY_UNRESOLVED`` dynamically until an exact, post-Commit-A
# offline Rust qualification report and its persisted binary have both been
# validated.  Keeping that distinction here permits the frozen Commit-A code
# to become eligible through evidence without changing its source afterward.
UNRESOLVED_AUTHORITATIVE_BLOCKERS: Final[tuple[str, ...]] = ()
# The intent/checkpoint continuation, host recovery-anchor/UID-65534 reclaim,
# atomic-stage, raw-seed verifier and exact-abort fault matrices now pass.  This
# admits a fresh execution attempt; it does not claim external genesis, create
# formal evidence, or change the observed 14/24 / NOT_RUN state by itself.
_PRESTAGE_RECOVERY_IMPLEMENTED: Final = True

_TRANSACTION_STATES: Final = (
    "RESERVED",
    "STAGED_PROSPECTIVE_REPLAY_PASSED",
    "SEED_CUSTODY_VERIFIED",
    "MARKER_COMPLETE",
    "ACTORS_ABSENT",
    "PUBLISHED",
)
_STAGED_FILENAMES: Final = MappingProxyType({
    "evidence": "public-evidence.json",
    "promotion": "promotion.json",
    "receipt": "publication-receipt.json",
})
_PRESTAGE_INTENT_FILENAME: Final = "prestage-intent.json"
_ACTOR_TRUST_CHECKPOINT_FILENAME: Final = "actor-trust-checkpoint.json"
_LIVE_QUALIFICATION_BUNDLE_FILENAME: Final = "live-qualification-bundle.json"
_SEED_CUSTODY_VERIFICATION_FILENAME: Final = "seed-custody-verification.json"
_RECOVERY_ANCHOR_FILENAME: Final = "recovery-anchor.json"
_RECOVERY_ANCHOR_READY_FILENAME: Final = "recovery-anchor.ready.json"
_RECOVERY_ANCHOR_SCHEMA: Final = "hegel-phase3-m25-host-recovery-anchor/1"
_RECOVERY_ANCHOR_READY_SCHEMA: Final = (
    "hegel-phase3-m25-host-recovery-anchor-ready/1"
)
_PRESEED_ABORT_PLAN_FILENAME: Final = "phase3_m25_preseed_abort_plan.json"
_PRESEED_ABORT_PLAN_SCHEMA: Final = "hegel-phase3-m25-preseed-abort-plan/1"
_PRESEED_ABORT_ABSENCE_FILENAME: Final = (
    "phase3_m25_preseed_actor_absence.json"
)
_PRESEED_ABORT_ABSENCE_SCHEMA: Final = (
    "hegel-phase3-m25-preseed-actor-absence-receipt/1"
)
_PRESEED_ABORT_TOMBSTONE_SCHEMA: Final = (
    "hegel-phase3-m25-preseed-abort-terminal-tombstone/1"
)
_PRESEED_ABORT_RETIREMENT_SCHEMA: Final = (
    "hegel-phase3-m25-preseed-abort-output-retirement/1"
)
_RESERVATION_BOOTSTRAP_ORDER: Final = (
    "opaque_run_reservation",
    "opaque_ledger_reservation",
    "public_evidence_reservation",
    "public_promotion_reservation",
    "publication_receipt_reservation",
    "stage_directory",
    "live_qualification_bundle",
    "reserved_journal",
    "prestage_intent",
)

# ``python -m hegel_machine...`` executes the package ``__init__`` before the
# CLI module.  Consequently the honest runtime closure is the complete
# top-level Python package, not just this executor's direct imports.  Binding
# every file is deliberately conservative and prevents an uncommitted helper
# from entering an otherwise committed ceremony.
_RUNTIME_PACKAGE_INPUTS: Final = tuple(
    sorted((PROJECT_ROOT / "src/hegel_machine").glob("*.py"))
)
_RUNTIME_TOOL_INPUTS: Final = tuple(
    PROJECT_ROOT / relative
    for relative in (
        "tools/phase3_m25_formal_actor_worker_v1.py",
        "tools/phase3_m25_python_bridge_actor_worker_v1.py",
        "tools/phase3_m25_parent_auditor_actor_worker_v1.py",
        "tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py",
        "tools/phase3_m25_formal_rust_actor_worker_v1.sh",
        "tools/phase3_split_partition_calculator_fd3_v1.py",
        "tools/phase3_split_partition_calculator_fd3_v1.rs",
        "tools/phase3_container_actor_probe_v1.py",
        "tools/phase3_container_actor_probe_v1.rs",
        "tools/phase3_m25_actor_operation_probe_v1.py",
        "tools/phase3_m25_bridge_dag_binary_qualification_v1.py",
        "tools/phase3_m25_seed_custody_verifier_v1.py",
    )
)
_RUST_STATIC_REPLAYER_SOURCE_INPUTS: Final = tuple(
    PROJECT_ROOT / relative
    for relative in (
        "rust/formal_bridge_m25/Cargo.toml",
        "rust/formal_bridge_m25/Cargo.lock",
        "rust/formal_bridge_m25/src/lib.rs",
        "rust/formal_bridge_m25/src/main.rs",
    )
)
_RUST_BRIDGE_DAG_REPLAYER_SOURCE_INPUTS: Final = tuple(
    PROJECT_ROOT / relative
    for relative in (
        "rust/m25_bridge_dag_replay/Cargo.toml",
        "rust/m25_bridge_dag_replay/Cargo.lock",
        "rust/m25_bridge_dag_replay/src/lib.rs",
        "rust/m25_bridge_dag_replay/src/main.rs",
    )
)
REQUIRED_COMMIT_A_INPUTS: Final = (
    PROFILE_PATH,
    SECCOMP_PATH,
    BUILD_SECCOMP_PATH,
    *_RUNTIME_PACKAGE_INPUTS,
    *_RUNTIME_TOOL_INPUTS,
    *_RUST_STATIC_REPLAYER_SOURCE_INPUTS,
    *_RUST_BRIDGE_DAG_REPLAYER_SOURCE_INPUTS,
)


class FormalContainerExecutorError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise FormalContainerExecutorError(code, detail)


def _run(
    command: Sequence[str],
    *,
    stdin: bytes | None = None,
    timeout: int = 180,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    if command and command[0] == "docker":
        _fail(FAIL_CONTAINER, "unbound Docker CLI invocation is forbidden")
    cwd: Path | None = None
    if working_directory is not None:
        cwd = Path(os.path.abspath(os.fspath(working_directory)))
        try:
            metadata = cwd.lstat()
        except OSError as exc:
            _fail(FAIL_CONTAINER, f"command working directory is absent: {exc}")
        if cwd.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            _fail(FAIL_CONTAINER, "command working directory inode differs")
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=(
                {"LC_ALL": "C.UTF-8", "LANG": "C", "PATH": "/usr/bin:/bin"}
                if environment is None
                else dict(environment)
            ),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_CONTAINER, f"command could not complete: {type(exc).__name__}")
    if check and completed.returncode != 0:
        # Actor stderr is constrained to a stable code, but keep the host
        # report bounded and never include arbitrary stdout.
        detail = completed.stderr.decode("ascii", "replace")[-256:].strip()
        _fail(FAIL_CONTAINER, f"container command exited {completed.returncode}: {detail}")
    return completed


def _commit(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        _fail(FAIL_PREFLIGHT, "basis commit must be lowercase SHA-1 hex")
    return value


def _reject_caller_symlink_chain_v1(path: Path, label: str) -> None:
    """Reject a caller-selected path if any existing component is a symlink."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            if cursor.is_symlink():
                _fail(FAIL_PREFLIGHT, f"{label} contains a symlink component")
        except OSError as exc:
            _fail(FAIL_PREFLIGHT, f"{label} path chain cannot be inspected: {exc}")


def validate_commit_b_output_names_v1(
    public_evidence_path: Path, public_promotion_path: Path
) -> None:
    if public_evidence_path.name != COMMIT_B_EVIDENCE_BASENAME:
        _fail(FAIL_PUBLICATION, "formal evidence output basename is not Commit-B allowlisted")
    if public_promotion_path.name != COMMIT_B_PROMOTION_BASENAME:
        _fail(FAIL_PUBLICATION, "formal promotion output basename is not Commit-B allowlisted")


def _transport(value: object) -> object:
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, MarkerSnapshot):
        return {
            "state": value.state,
            "split_version_digest": _transport(value.split_version_digest),
            "seed_commitment_manifest_root": _transport(value.seed_commitment_manifest_root),
            "custodian_key_id": _transport(value.custodian_key_id),
            "created_at_unix_seconds": value.created_at_unix_seconds,
        }
    if isinstance(value, Mapping):
        return {str(key): _transport(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_transport(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    raise TypeError(f"unsupported public transport value {type(value).__name__}")


def _restore(value: object) -> object:
    if type(value) is dict:
        if set(value) == {"bytes_hex"} and type(value["bytes_hex"]) is str:
            return bytes.fromhex(value["bytes_hex"])
        return {str(key): _restore(item) for key, item in value.items()}
    if type(value) is list:
        return tuple(_restore(item) for item in value)
    return value


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(_transport(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _validate_prestage_runtime_bindings_v1(
    fields: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "m3_execution_implementation_binding_roots",
        "formal_rust_replay_binary_path",
        "formal_rust_replay_binary_sha256",
        "rust_bridge_dag_replay_binary_path",
        "rust_bridge_dag_replay_binary_sha256",
        "rust_bridge_dag_qualification_report_sha256",
        "actor_profile_sha256",
    }
    if type(fields) is not dict or set(fields) != expected_keys:
        _fail(FAIL_PREFLIGHT, "prestage runtime binding field set differs")
    roots = fields.get("m3_execution_implementation_binding_roots")
    if (
        type(roots) is not dict
        or set(roots)
        != {
            "python_implementation_binding_root",
            "rust_implementation_binding_root",
        }
        or any(type(value) is not bytes or len(value) != 32 for value in roots.values())
        or any(
            type(fields.get(name)) is not bytes or len(fields[name]) != 32
            for name in (
                "formal_rust_replay_binary_sha256",
                "rust_bridge_dag_replay_binary_sha256",
                "rust_bridge_dag_qualification_report_sha256",
                "actor_profile_sha256",
            )
        )
        or any(
            type(fields.get(name)) is not str
            or not str(fields[name]).startswith("/")
            for name in (
                "formal_rust_replay_binary_path",
                "rust_bridge_dag_replay_binary_path",
            )
        )
    ):
        _fail(FAIL_PREFLIGHT, "prestage runtime binding identity differs")
    return dict(fields)


def build_prestage_intent_fields_v1(
    *,
    basis_commit: str,
    run_id: bytes,
    ledger_id: bytes,
    created_at_unix_seconds: int,
    trust_genesis_id: bytes,
    actor_qualification_report: Mapping[str, object],
    errata_qualification_report: Mapping[str, object],
    rust_bridge_dag_qualification_report_sha256: bytes,
    live_actor_protocol_qualification_bundle_content_id: bytes,
    qualification_only_key_ids: Mapping[int, bytes],
    live_actor_protocol_qualification_bundle: Mapping[str, object],
    live_actor_protocol_qualification_canonical_bundle_bytes: bytes,
    live_actor_protocol_daemon_receipt_binding: bytes,
    runtime_binding_fields: Mapping[str, object],
) -> dict[str, object]:
    """Freeze every selectable public input before any role key or seed exists.

    The embedded reports are intentional.  A recovery must replay the exact
    admitted copies, not let an operator select another report with the same
    Commit-A identity after seeing a split.  This diagnostic JSON is outside
    the frozen formal CBOR wire.
    """

    commit = _commit(basis_commit)
    if type(run_id) is not bytes or len(run_id) != 16:
        _fail(FAIL_PREFLIGHT, "prestage run ID must be 16 bytes")
    if type(ledger_id) is not bytes or len(ledger_id) != 16 or ledger_id == run_id:
        _fail(FAIL_PREFLIGHT, "prestage ledger ID must be a distinct 16-byte value")
    if type(created_at_unix_seconds) is not int or created_at_unix_seconds < 0:
        _fail(FAIL_PREFLIGHT, "prestage timestamp must be a non-negative integer")
    if type(trust_genesis_id) is not bytes or len(trust_genesis_id) != 16:
        _fail(FAIL_PREFLIGHT, "prestage trust-genesis ID must be 16 bytes")
    if trust_genesis_id in {run_id, ledger_id}:
        _fail(
            FAIL_PREFLIGHT,
            "run, ledger and trust-genesis IDs must be pairwise distinct",
        )
    if type(rust_bridge_dag_qualification_report_sha256) is not bytes or len(
        rust_bridge_dag_qualification_report_sha256
    ) != 32:
        _fail(FAIL_PREFLIGHT, "prestage Rust bridge qualification ID must be 32 bytes")
    if type(live_actor_protocol_qualification_bundle_content_id) is not bytes or len(
        live_actor_protocol_qualification_bundle_content_id
    ) != 32:
        _fail(FAIL_PREFLIGHT, "prestage live actor protocol bundle ID must be 32 bytes")
    if (
        type(qualification_only_key_ids) is not dict
        or set(qualification_only_key_ids) != {1, 2, 3, 4}
        or any(
            type(value) is not bytes or len(value) != 16
            for value in qualification_only_key_ids.values()
        )
        or len(set(qualification_only_key_ids.values())) != 4
    ):
        _fail(
            FAIL_PREFLIGHT,
            "prestage qualification-only key ID set must be four distinct 16-byte IDs",
        )
    if not isinstance(live_actor_protocol_qualification_bundle, Mapping):
        _fail(FAIL_PREFLIGHT, "live actor protocol bundle must be a mapping")
    live_bundle = dict(live_actor_protocol_qualification_bundle)
    if (
        type(live_actor_protocol_qualification_canonical_bundle_bytes) is not bytes
        or _canonical_json(live_bundle)
        != live_actor_protocol_qualification_canonical_bundle_bytes
    ):
        _fail(
            FAIL_PREFLIGHT,
            "live actor protocol bundle differs from consumed canonical bytes",
        )
    live_bundle_sha256 = hashlib.sha256(
        live_actor_protocol_qualification_canonical_bundle_bytes
    ).digest()
    if (
        type(live_actor_protocol_daemon_receipt_binding) is not bytes
        or len(live_actor_protocol_daemon_receipt_binding) != 32
    ):
        _fail(FAIL_PREFLIGHT, "live actor protocol daemon binding must be 32 bytes")
    if not isinstance(actor_qualification_report, Mapping) or not isinstance(
        errata_qualification_report, Mapping
    ):
        _fail(FAIL_PREFLIGHT, "prestage qualification reports must be mappings")
    if not isinstance(runtime_binding_fields, Mapping):
        _fail(FAIL_PREFLIGHT, "prestage runtime bindings must be a mapping")
    runtime_bindings = _validate_prestage_runtime_bindings_v1(
        dict(runtime_binding_fields)
    )
    actor_report = dict(actor_qualification_report)
    errata_report = dict(errata_qualification_report)
    fields: dict[str, object] = {
        "schema": PRESTAGE_INTENT_SCHEMA,
        "basis_commit": commit,
        "run_id_hex": run_id.hex(),
        "ledger_id_hex": ledger_id.hex(),
        "created_at_unix_seconds": created_at_unix_seconds,
        "trust_genesis_id_hex": trust_genesis_id.hex(),
        "actor_qualification_report": actor_report,
        "actor_qualification_report_sha256": hashlib.sha256(
            _canonical_json(actor_report)
        ).hexdigest(),
        "errata_qualification_report": errata_report,
        "errata_qualification_report_sha256": hashlib.sha256(
            _canonical_json(errata_report)
        ).hexdigest(),
        "rust_bridge_dag_qualification_report_sha256": (
            rust_bridge_dag_qualification_report_sha256.hex()
        ),
        "live_actor_protocol_qualification_bundle_content_id": (
            live_actor_protocol_qualification_bundle_content_id
        ),
        "live_actor_protocol_qualification_bundle": live_bundle,
        "live_actor_protocol_qualification_bundle_sha256": live_bundle_sha256,
        "live_actor_protocol_daemon_receipt_binding": (
            live_actor_protocol_daemon_receipt_binding
        ),
        "qualification_only_key_id_rows": tuple(
            {
                "purpose_id": purpose,
                "qualification_only_key_id_16_bytes": qualification_only_key_ids[purpose],
            }
            for purpose in (1, 2, 3, 4)
        ),
        "runtime_binding_fields": runtime_bindings,
        "selection_frozen_before_actor_keygen": True,
        "selection_frozen_before_seed_instantiation": True,
        "formal_cbor_wire_changed": False,
    }
    assert_public_payload_contains_no_secret_fields(fields)
    return fields


def validate_prestage_intent_fields_v1(
    fields: Mapping[str, object],
    *,
    basis_commit: str,
    run_id: bytes,
    ledger_id: bytes,
) -> dict[str, object]:
    """Validate and return one exact, transport-restored prestage intent."""

    expected_keys = {
        "schema",
        "basis_commit",
        "run_id_hex",
        "ledger_id_hex",
        "created_at_unix_seconds",
        "trust_genesis_id_hex",
        "actor_qualification_report",
        "actor_qualification_report_sha256",
        "errata_qualification_report",
        "errata_qualification_report_sha256",
        "rust_bridge_dag_qualification_report_sha256",
        "live_actor_protocol_qualification_bundle_content_id",
        "live_actor_protocol_qualification_bundle",
        "live_actor_protocol_qualification_bundle_sha256",
        "live_actor_protocol_daemon_receipt_binding",
        "qualification_only_key_id_rows",
        "runtime_binding_fields",
        "selection_frozen_before_actor_keygen",
        "selection_frozen_before_seed_instantiation",
        "formal_cbor_wire_changed",
    }
    if type(fields) is not dict or set(fields) != expected_keys:
        _fail(FAIL_TRANSACTION_LOCK, "prestage intent field set differs")
    restored = _restore(dict(fields))
    if type(restored) is not dict:
        _fail(FAIL_TRANSACTION_LOCK, "prestage intent transport is invalid")
    actor_report = restored.get("actor_qualification_report")
    errata_report = restored.get("errata_qualification_report")
    trust_hex = restored.get("trust_genesis_id_hex")
    bridge_hex = restored.get("rust_bridge_dag_qualification_report_sha256")
    protocol_bundle_id = restored.get(
        "live_actor_protocol_qualification_bundle_content_id"
    )
    protocol_bundle = restored.get("live_actor_protocol_qualification_bundle")
    protocol_bundle_sha256 = restored.get(
        "live_actor_protocol_qualification_bundle_sha256"
    )
    protocol_daemon_binding = restored.get(
        "live_actor_protocol_daemon_receipt_binding"
    )
    qualification_rows = restored.get("qualification_only_key_id_rows")
    runtime_bindings = restored.get("runtime_binding_fields")
    if (
        restored.get("schema") != PRESTAGE_INTENT_SCHEMA
        or restored.get("basis_commit") != _commit(basis_commit)
        or restored.get("run_id_hex") != run_id.hex()
        or restored.get("ledger_id_hex") != ledger_id.hex()
        or type(restored.get("created_at_unix_seconds")) is not int
        or restored["created_at_unix_seconds"] < 0
        or type(trust_hex) is not str
        or re.fullmatch(r"[0-9a-f]{32}", trust_hex) is None
        or trust_hex in {run_id.hex(), ledger_id.hex()}
        or type(bridge_hex) is not str
        or re.fullmatch(r"[0-9a-f]{64}", bridge_hex) is None
        or type(protocol_bundle_id) is not bytes
        or len(protocol_bundle_id) != 32
        or type(protocol_bundle) is not dict
        or type(protocol_bundle_sha256) is not bytes
        or len(protocol_bundle_sha256) != 32
        or protocol_bundle_sha256
        != hashlib.sha256(_canonical_json(protocol_bundle)).digest()
        or type(protocol_daemon_binding) is not bytes
        or len(protocol_daemon_binding) != 32
        or type(qualification_rows) is not tuple
        or len(qualification_rows) != 4
        or type(runtime_bindings) is not dict
        or type(actor_report) is not dict
        or type(errata_report) is not dict
        or restored.get("actor_qualification_report_sha256")
        != hashlib.sha256(_canonical_json(actor_report)).hexdigest()
        or restored.get("errata_qualification_report_sha256")
        != hashlib.sha256(_canonical_json(errata_report)).hexdigest()
        or restored.get("selection_frozen_before_actor_keygen") is not True
        or restored.get("selection_frozen_before_seed_instantiation") is not True
        or restored.get("formal_cbor_wire_changed") is not False
    ):
        _fail(FAIL_TRANSACTION_LOCK, "prestage intent identity or digest differs")
    try:
        qualification_key_ids = {
            int(row["purpose_id"]): row["qualification_only_key_id_16_bytes"]
            for row in qualification_rows
        }
    except (KeyError, TypeError, ValueError) as exc:
        _fail(
            FAIL_TRANSACTION_LOCK,
            f"prestage qualification-only key rows are invalid: {exc}",
        )
    if (
        any(type(row) is not dict or set(row) != {
            "purpose_id", "qualification_only_key_id_16_bytes"
        } for row in qualification_rows)
        or tuple(row["purpose_id"] for row in qualification_rows) != (1, 2, 3, 4)
        or set(qualification_key_ids) != {1, 2, 3, 4}
        or any(
            type(value) is not bytes or len(value) != 16
            for value in qualification_key_ids.values()
        )
        or len(set(qualification_key_ids.values())) != 4
    ):
        _fail(
            FAIL_TRANSACTION_LOCK,
            "prestage qualification-only key rows differ from the exact four-role set",
        )
    try:
        _validate_prestage_runtime_bindings_v1(runtime_bindings)
    except FormalContainerExecutorError as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"prestage runtime binding differs: {exc.detail}")
    if runtime_bindings["rust_bridge_dag_qualification_report_sha256"] != bytes.fromhex(
        bridge_hex
    ):
        _fail(FAIL_TRANSACTION_LOCK, "prestage bridge report identities disagree")
    assert_public_payload_contains_no_secret_fields(restored)
    return restored


def _qualification_only_key_ids_from_intent_v1(
    intent: Mapping[str, object],
) -> Mapping[int, bytes]:
    rows = intent.get("qualification_only_key_id_rows")
    if type(rows) is not tuple or len(rows) != 4 or any(
        type(row) is not dict
        or set(row) != {"purpose_id", "qualification_only_key_id_16_bytes"}
        for row in rows
    ):
        _fail(FAIL_TRANSACTION_LOCK, "bound qualification-only key rows are invalid")
    values = {
        row["purpose_id"]: row["qualification_only_key_id_16_bytes"]
        for row in rows
    }
    if (
        tuple(values) != (1, 2, 3, 4)
        or any(type(value) is not bytes or len(value) != 16 for value in values.values())
        or len(set(values.values())) != 4
    ):
        _fail(FAIL_TRANSACTION_LOCK, "bound qualification-only key identity differs")
    return MappingProxyType(values)  # type: ignore[arg-type]


def _require_formal_key_ids_disjoint_from_qualification_v1(
    formal_key_ids: Mapping[int, bytes],
    qualification_key_ids: Mapping[int, bytes],
) -> None:
    if set(formal_key_ids) != {1, 2, 3, 4} or set(qualification_key_ids) != {
        1, 2, 3, 4
    }:
        _fail(FAIL_PREFLIGHT, "formal/qualification key ID role set differs")
    overlap = set(formal_key_ids.values()) & set(qualification_key_ids.values())
    if overlap:
        _fail(
            FAIL_PREFLIGHT,
            "formal ceremony key IDs overlap qualification-only key IDs",
        )


def _publish_exclusive(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                _fail(FAIL_PUBLICATION, "short O_EXCL public-evidence write")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class CeremonyReadinessV1:
    basis_commit: str
    ready: bool
    blockers: tuple[str, ...]
    formal_gates_before: int = 14
    formal_gates_after: int = 14
    child_state: str = "NOT_RUN"
    qualification_side_effects_performed: bool = True
    ceremony_actor_key_seed_marker_side_effects_performed: bool = False

    def public_report(self) -> dict[str, object]:
        return {
            "schema": READINESS_SCHEMA,
            "basis_commit": self.basis_commit,
            "ready_for_explicit_execute": self.ready,
            "blockers": list(self.blockers),
            "formal_gates_before": self.formal_gates_before,
            "formal_gates_after": self.formal_gates_after,
            "child_state": self.child_state,
            "m3_run_started": False,
            "qualification_side_effects_performed": (
                self.qualification_side_effects_performed
            ),
            "qualification_network_mode": "none",
            "qualification_persistent_rust_binary_verified_or_written": True,
            "qualification_non_authoritative_roots_computed": True,
            "ceremony_actor_key_seed_marker_side_effects_performed": (
                self.ceremony_actor_key_seed_marker_side_effects_performed
            ),
            "formal_authority_or_gate_effect": "NONE",
            "static_replay_roots_are_execution_bindings": False,
        }


@dataclass(frozen=True, slots=True)
class ArchivedActorProtocolQualificationBindingV1:
    """Strict replay identity with no live-admission capability."""

    basis_commit: str
    bundle_content_id: bytes
    qualification_key_ids: Mapping[int, bytes]
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class LiveActorProtocolAdmissionBindingV1:
    """Identity extracted only after one same-process sealed live admission."""

    basis_commit: str
    bundle_content_id: bytes
    qualification_key_ids: Mapping[int, bytes]
    daemon_receipt_binding: bytes
    canonical_bundle_bytes: bytes
    report: Mapping[str, object]


def _validate_actor_protocol_binding_object_v1(
    value: object,
    *,
    basis_commit: str,
    live: bool,
) -> ArchivedActorProtocolQualificationBindingV1 | LiveActorProtocolAdmissionBindingV1:
    report_commit = getattr(value, "basis_commit")
    identifier = getattr(value, "bundle_content_id")
    key_ids_raw = getattr(value, "qualification_key_ids")
    daemon_binding = getattr(value, "daemon_receipt_binding", None)
    canonical_bundle_bytes = getattr(value, "canonical_bundle_bytes", None)
    if live:
        if type(canonical_bundle_bytes) is not bytes:
            raise ValueError("live qualification canonical bundle bytes are absent")
        try:
            report_raw = json.loads(canonical_bundle_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("live qualification canonical bundle is invalid") from exc
        if type(report_raw) is not dict or _canonical_json(report_raw) != canonical_bundle_bytes:
            raise ValueError("live qualification bundle bytes are not strict canonical JSON")
    else:
        report_raw = getattr(value, "report")
    if (
        report_commit != _commit(basis_commit)
        or type(identifier) is not bytes
        or len(identifier) != 32
        or not isinstance(key_ids_raw, Mapping)
        or not isinstance(report_raw, Mapping)
    ):
        raise ValueError("signed qualification bundle identity is malformed")
    key_ids = dict(key_ids_raw)
    report = dict(report_raw)
    if (
        set(key_ids) != {1, 2, 3, 4}
        or any(type(item) is not bytes or len(item) != 16 for item in key_ids.values())
        or len(set(key_ids.values())) != 4
    ):
        raise ValueError("signed qualification-only key ID set is malformed")
    if live:
        if type(daemon_binding) is not bytes or len(daemon_binding) != 32:
            raise ValueError("live qualification daemon binding is malformed")
        return LiveActorProtocolAdmissionBindingV1(
            basis_commit=report_commit,
            bundle_content_id=identifier,
            qualification_key_ids=MappingProxyType(key_ids),
            daemon_receipt_binding=daemon_binding,
            canonical_bundle_bytes=canonical_bundle_bytes,
            report=MappingProxyType(report),
        )
    if daemon_binding is not None:
        raise ValueError("archive replay unexpectedly carries a live daemon capability")
    return ArchivedActorProtocolQualificationBindingV1(
        basis_commit=report_commit,
        bundle_content_id=identifier,
        qualification_key_ids=MappingProxyType(key_ids),
        report=MappingProxyType(report),
    )


def load_actor_protocol_archive_qualification_v1(
    basis_commit: str,
) -> ArchivedActorProtocolQualificationBindingV1:
    """Strictly replay the stable archive; never grant live admission.

    The qualifier imports this executor to exercise its actor protocol, so a
    top-level import here would be circular.  Execute/recovery call this only
    after module initialization and before any formal role key or seed action.
    """

    try:
        from .phase3_m25_actor_protocol_qualification_v1 import (
            load_actor_protocol_qualification_report_v1,
        )

        report = load_actor_protocol_qualification_report_v1(
            LIVE_ACTOR_PROTOCOL_QUALIFICATION_REPORT_PATH,
            expected_basis_commit=_commit(basis_commit),
        )
        validated = _validate_actor_protocol_binding_object_v1(
            report,
            basis_commit=basis_commit,
            live=False,
        )
        assert isinstance(validated, ArchivedActorProtocolQualificationBindingV1)
        return validated
    except Exception as exc:
        _fail(
            FAIL_ACTOR_PROTOCOL_QUALIFICATION_UNRESOLVED,
            f"fixed live actor protocol qualification is invalid: {type(exc).__name__}",
        )


def qualify_live_actor_protocol_admission_v1(
    *, basis_commit: str, custody_directory: Path
) -> LiveActorProtocolAdmissionBindingV1:
    """Run the same-process no-seed protocol and consume its sealed token."""

    try:
        from .phase3_m25_actor_protocol_qualification_v1 import (
            ConsumedLiveActorProtocolAdmissionV1,
            consume_live_actor_protocol_admission_v1,
            qualify_live_actor_protocol_v1,
        )
        admission = qualify_live_actor_protocol_v1(
            basis_commit=_commit(basis_commit),
            custody_directory=custody_directory,
        )
        consumed = consume_live_actor_protocol_admission_v1(
            admission,
            expected_basis_commit=_commit(basis_commit),
        )
        if type(consumed) is not ConsumedLiveActorProtocolAdmissionV1:
            raise ValueError("live qualifier consumer returned the wrong type")
        validated = _validate_actor_protocol_binding_object_v1(
            consumed,
            basis_commit=basis_commit,
            live=True,
        )
        assert isinstance(validated, LiveActorProtocolAdmissionBindingV1)
        return validated
    except Exception as exc:
        _fail(
            FAIL_ACTOR_PROTOCOL_QUALIFICATION_UNRESOLVED,
            f"same-process live actor protocol admission failed: {type(exc).__name__}",
        )


def replay_transaction_local_actor_protocol_bundle_v1(
    *, basis_commit: str, bundle: Mapping[str, object]
) -> ArchivedActorProtocolQualificationBindingV1:
    """Strictly replay the transaction-local signed bundle without capability."""

    try:
        from .phase3_m25_actor_protocol_qualification_v1 import (
            ReplayedActorProtocolQualificationV1,
            validate_actor_protocol_qualification_report_v1,
        )

        replayed = validate_actor_protocol_qualification_report_v1(
            dict(bundle),
            expected_basis_commit=_commit(basis_commit),
        )
        if type(replayed) is not ReplayedActorProtocolQualificationV1:
            raise ValueError("transaction-local replay returned the wrong type")
        validated = _validate_actor_protocol_binding_object_v1(
            replayed,
            basis_commit=basis_commit,
            live=False,
        )
        assert isinstance(validated, ArchivedActorProtocolQualificationBindingV1)
        return validated
    except Exception as exc:
        _fail(
            FAIL_ACTOR_PROTOCOL_QUALIFICATION_UNRESOLVED,
            f"transaction-local actor protocol bundle is invalid: {type(exc).__name__}",
        )


def load_live_actor_protocol_qualification_id_v1(basis_commit: str) -> bytes:
    """Compatibility accessor for diagnostics; formal code uses the binding."""

    return load_actor_protocol_archive_qualification_v1(basis_commit).bundle_content_id


def inspect_formal_ceremony_readiness_v1(basis_commit: str) -> CeremonyReadinessV1:
    """Qualify the basis and stop before actor/key/seed/marker side effects.

    Offline implementation qualification is intentionally part of this guard:
    a plain static basis always carries the frozen NOT_READY sentinel and can
    never authorize the ceremony.  The qualifier may use temporary, network-
    disabled Docker build/replay containers and persist the commit-bound Rust
    enumerator.  It computes non-authoritative implementation-binding and
    qualification-receipt roots, but it cannot create actor keys, a split
    seed, markers, formal M3 output roots, formal authority, a gate transition,
    or an M3 state transition.
    """

    commit = _commit(basis_commit)
    basis = build_qualified_formal_static_basis_v1(commit)
    blockers = [*basis.blocking_gaps, *UNRESOLVED_AUTHORITATIVE_BLOCKERS]
    if not _PRESTAGE_RECOVERY_IMPLEMENTED:
        blockers.append(FAIL_PRESTAGE_RECOVERY_UNRESOLVED)
    bindings_ready = basis.implementation_inputs.get(
        "m3_execution_implementation_bindings_ready"
    )
    bindings = basis.implementation_inputs.get(
        "m3_execution_implementation_binding_roots"
    )
    if bindings_ready is not True or not isinstance(bindings, Mapping):
        if "M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY" not in blockers:
            blockers.append("M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY")
    else:
        if set(bindings) != {
            "python_implementation_binding_root",
            "rust_implementation_binding_root",
        } or any(type(value) is not bytes or len(value) != 32 for value in bindings.values()):
            blockers.append("M3_EXECUTION_IMPLEMENTATION_BINDING_ROOT_SET_INVALID")
    if not any(
        blocker.startswith("M3_EXECUTION_IMPLEMENTATION_") for blocker in blockers
    ):
        try:
            load_qualified_rust_bridge_dag_binary_binding_v1(
                expected_basis_commit=commit,
                report_path=DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
            )
        except (OSError, BridgeDagBinaryQualificationError):
            blockers.append(FAIL_BRIDGE_REPLAY_UNRESOLVED)
    try:
        load_actor_protocol_archive_qualification_v1(commit)
    except FormalContainerExecutorError as exc:
        if exc.code == FAIL_ACTOR_PROTOCOL_QUALIFICATION_UNRESOLVED:
            blockers.append(exc.code)
        else:
            raise
    return CeremonyReadinessV1(commit, not blockers, tuple(sorted(set(blockers))))


def require_formal_ceremony_ready_v1(basis: FormalStaticBasisV1) -> Mapping[str, bytes]:
    """Reject static replay bindings before any irreversible seed action."""

    try:
        roots = validate_m3_execution_implementation_bindings_v1(basis)
    except M3ImplementationQualificationError as error:
        _fail(
            FAIL_EXECUTION_BINDINGS,
            "runnable, commit-bound Python/Rust closure enumerators are absent or "
            f"substituted ({error.code}: {error.detail}); static replayers cannot substitute",
        )
    if basis.blocking_gaps:
        _fail(FAIL_PREFLIGHT, "static basis still reports: " + ",".join(basis.blocking_gaps))
    return MappingProxyType(dict(roots))


def serialize_gate_evidence_inputs_v1(inputs: GateEvidenceInputsV1) -> dict[str, object]:
    fields = {field.name: getattr(inputs, field.name) for field in dataclass_fields(inputs)}
    assert_public_payload_contains_no_secret_fields(fields)
    payload: dict[str, object] = {
        "schema": PUBLIC_REPLAY_SCHEMA,
        "artifact_kind": "FORMAL_GATE_EVIDENCE_INPUTS_PUBLIC_REPLAY",
        "gate_evidence_inputs": _transport(fields),
        "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
        "contains_private_key": False,
        "contains_raw_split_seed": False,
        "contains_split_assignment_rows": False,
    }
    payload["payload_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def load_gate_evidence_inputs_v1(payload: Mapping[str, object]) -> GateEvidenceInputsV1:
    """Load the complete public preimage bundle used by a later M3 start."""

    expected_top_level = {
        "schema",
        "artifact_kind",
        "gate_evidence_inputs",
        "authority_disclosure",
        "contains_private_key",
        "contains_raw_split_seed",
        "contains_split_assignment_rows",
        "payload_sha256",
    }
    if type(payload) is not dict or set(payload) != expected_top_level:
        _fail(FAIL_PUBLICATION, "public replay top-level field set differs")
    if payload.get("schema") != PUBLIC_REPLAY_SCHEMA:
        _fail(FAIL_PUBLICATION, "public replay schema differs")
    if payload.get("artifact_kind") != "FORMAL_GATE_EVIDENCE_INPUTS_PUBLIC_REPLAY":
        _fail(FAIL_PUBLICATION, "public replay artifact kind differs")
    if payload.get("authority_disclosure") != dict(TECHNICAL_ACTOR_DISCLOSURE_V1):
        _fail(FAIL_PUBLICATION, "public replay technical-actor disclosure differs")
    for flag in (
        "contains_private_key",
        "contains_raw_split_seed",
        "contains_split_assignment_rows",
    ):
        if payload.get(flag) is not False:
            _fail(FAIL_PUBLICATION, f"public replay secret flag is not strict false: {flag}")
    assert_public_payload_contains_no_secret_fields(payload)
    expected_hash = payload.get("payload_sha256")
    body = dict(payload)
    body.pop("payload_sha256", None)
    if type(expected_hash) is not str or hashlib.sha256(_canonical_json(body)).hexdigest() != expected_hash:
        _fail(FAIL_PUBLICATION, "public replay bundle digest differs")
    try:
        raw = _restore(payload.get("gate_evidence_inputs"))
    except (TypeError, ValueError) as exc:
        _fail(FAIL_PUBLICATION, f"public replay transport encoding is invalid: {exc}")
    if not isinstance(raw, Mapping):
        _fail(FAIL_PUBLICATION, "gate-evidence input body is absent")
    names = {field.name for field in dataclass_fields(GateEvidenceInputsV1)}
    if set(raw) != names:
        _fail(FAIL_PUBLICATION, "GateEvidenceInputsV1 public field set differs")
    marker_raw = raw["marker_snapshot"]
    marker_fields = {
        "state",
        "split_version_digest",
        "seed_commitment_manifest_root",
        "custodian_key_id",
        "created_at_unix_seconds",
    }
    if type(marker_raw) is not dict or set(marker_raw) != marker_fields:
        _fail(FAIL_PUBLICATION, "marker snapshot is invalid")
    values = dict(raw)
    try:
        values["marker_snapshot"] = MarkerSnapshot(**marker_raw)
        result = GateEvidenceInputsV1(**values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        _fail(FAIL_PUBLICATION, f"public replay typed reconstruction failed: {exc}")
    assert_public_payload_contains_no_secret_fields(
        {field.name: getattr(result, field.name) for field in dataclass_fields(result)}
    )
    return result


def replay_public_gate_evidence_v1(payload: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct all GateEvidenceInputs and independently replay 24/24."""

    inputs = load_gate_evidence_inputs_v1(payload)
    return promote_gate_evidence_v1(evaluate_gates_15_24_v1(inputs))


def _fsync_directory_v1(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_durable_v1(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"exclusive reservation failed for {path.name}: {exc}")
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                _fail(FAIL_PUBLICATION, f"short durable write for {path.name}")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_v1(path.parent)


def _install_bootstrap_file_no_replace_v1(
    path: Path,
    payload: bytes,
    mode: int,
    *,
    fault: Callable[[str], None] | None,
    label: str,
) -> None:
    """Crash-resumable no-overwrite install for immutable bootstrap files.

    The visible target is created only by a hard-link after the deterministic
    ``.next`` inode is fully written and fsynced.  Empty/partial ``.next``
    inodes are provably precommit and may be repaired while the transaction
    lock (or, for the initial lock, the ``.next`` inode lock) is exclusively
    held.  A target plus ``.next`` is accepted only when both names identify
    the same exact inode.
    """

    temporary = path.with_name(path.name + ".next")

    def inject(point: str) -> None:
        if fault is not None:
            fault(f"{point}_{label}")

    def exact_file(candidate: Path) -> bool:
        if candidate.is_symlink():
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap {candidate.name} is a symlink")
        try:
            metadata = candidate.stat()
            observed = candidate.read_bytes()
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap {candidate.name} cannot be read: {exc}")
        return (
            stat.S_ISREG(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == mode
            and observed == payload
        )

    if path.exists() or path.is_symlink():
        if not exact_file(path):
            _fail(FAIL_TRANSACTION_LOCK, f"committed bootstrap file differs: {path.name}")
        if temporary.exists() or temporary.is_symlink():
            if not exact_file(temporary):
                _fail(
                    FAIL_TRANSACTION_LOCK,
                    f"committed bootstrap file retains a differing next inode: {path.name}",
                )
            try:
                if not os.path.samefile(path, temporary):
                    _fail(
                        FAIL_TRANSACTION_LOCK,
                        f"bootstrap target/next are not the same committed inode: {path.name}",
                    )
                temporary.unlink()
                inject("after_bootstrap_next_unlink")
                _fsync_directory_v1(path.parent)
            except OSError as exc:
                _fail(FAIL_TRANSACTION_LOCK, f"bootstrap next cleanup failed: {exc}")
        return

    descriptor: int | None = None
    try:
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink():
                _fail(FAIL_TRANSACTION_LOCK, f"bootstrap next is a symlink: {temporary.name}")
            descriptor = os.open(
                temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                _fail(FAIL_TRANSACTION_LOCK, f"bootstrap next inode is active: {exc}")
            metadata = os.fstat(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = bytearray()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed.extend(chunk)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != mode
                or bytes(observed) != payload
            ):
                # No target exists: this inode never crossed the hard-link
                # commit point and is safe to discard under the held lock.
                temporary.unlink()
                _fsync_directory_v1(path.parent)
                os.close(descriptor)
                descriptor = None
            else:
                # A crash may have occurred during the prior fsync. Re-fsync
                # the exact recovered inode and its directory before commit.
                os.fsync(descriptor)
                inject("after_bootstrap_next_fsync")
                _fsync_directory_v1(path.parent)
        if descriptor is None:
            descriptor = os.open(
                temporary,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fchmod(descriptor, mode)
            inject("after_bootstrap_next_inode")
            first_count = min(len(payload), 16)
            if first_count:
                written = os.write(descriptor, payload[:first_count])
                if written != first_count:
                    _fail(FAIL_TRANSACTION_LOCK, "bootstrap initial write was short")
                inject("after_bootstrap_partial_write")
            offset = first_count
            while offset < len(payload):
                count = os.write(descriptor, payload[offset:])
                if count <= 0:
                    _fail(FAIL_TRANSACTION_LOCK, "bootstrap durable write was short")
                offset += count
            os.fsync(descriptor)
            inject("after_bootstrap_next_fsync")
            _fsync_directory_v1(path.parent)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap target raced into existence: {path.name}")
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap no-replace link failed: {exc}")
        inject("after_bootstrap_link")
        temporary.unlink()
        inject("after_bootstrap_next_unlink")
        _fsync_directory_v1(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_atomic_resumable_v1(
    path: Path,
    payload: bytes,
    mode: int,
    *,
    fault: Callable[[str], None] | None = None,
    fault_label: str,
) -> None:
    """Install one immutable file via an exact, resumable ``.next`` inode.

    A complete ``.next`` file can be promoted after a crash.  A partial or
    differing inode is never overwritten because doing so would erase evidence
    of an interrupted durable write.
    """

    def inject(suffix: str) -> None:
        if fault is not None:
            fault(f"{suffix}_{fault_label}")

    def require_exact(candidate: Path, label: str) -> None:
        if candidate.is_symlink():
            _fail(FAIL_PUBLICATION, f"{label} may not be a symlink")
        try:
            metadata = candidate.stat()
            observed = candidate.read_bytes()
        except OSError as exc:
            _fail(FAIL_PUBLICATION, f"{label} cannot be read: {exc}")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
            or observed != payload
        ):
            _fail(FAIL_PUBLICATION, f"{label} differs from the exact resumable payload")

    temporary = path.with_name(path.name + ".next")
    if path.exists() or path.is_symlink():
        require_exact(path, path.name)
        if temporary.exists() or temporary.is_symlink():
            _fail(FAIL_PUBLICATION, f"completed {path.name} retains a next inode")
        return
    if temporary.exists() or temporary.is_symlink():
        require_exact(temporary, temporary.name)
    else:
        inject("before_stage_next_write")
        _write_exclusive_durable_v1(temporary, payload, mode)
        inject("after_stage_next_fsync")
    os.replace(temporary, path)
    inject("after_stage_rename_before_dir_fsync")
    _fsync_directory_v1(path.parent)
    inject("after_stage_dir_fsync")


def _validate_seed_custody_verification_receipt_v1(
    receipt: Mapping[str, object],
    *,
    expected_commitment: bytes,
    expected_daemon_binding: bytes,
) -> bytes:
    """Validate one bounded outer receipt without reading retained seed bytes."""

    required = {
        "schema",
        "verified",
        "seed_commitment_hex",
        "seed_length_bytes",
        "seed_intent_sha256",
        "completion_receipt_sha256",
        "raw_seed_read_inside_keyless_verifier",
        "raw_seed_exported",
        "private_key_mount_present",
        "state_mount_present",
        "verifier_numeric_uid",
        "verifier_numeric_gid",
        "custody_artifacts_owned_by_verifier_identity",
        "inner_receipt_sha256",
        "verifier_tool_sha256",
        "docker_command_argv_sha256",
        "docker_command_policy_sha256",
        "docker_image_ref",
        "docker_seccomp_sha256",
        "docker_daemon_receipt_sha256",
        "docker_control_plane_binding_sha256",
        "docker_network_mode",
        "docker_ipc_mode",
        "docker_read_only_rootfs",
        "docker_stdout_limit_bytes",
        "docker_timeout_seconds",
        "custody_owner_policy_id",
        "incarnation_fields_nonidentity",
        "receipt_sha256",
    }
    if type(receipt) is not dict and not isinstance(receipt, MappingProxyType):
        _fail(FAIL_CUSTODY, "keyless seed-custody verifier receipt is not a mapping")
    body = dict(receipt)
    claimed_hash = body.pop("receipt_sha256", None)
    digest_fields = (
        "seed_intent_sha256",
        "completion_receipt_sha256",
        "inner_receipt_sha256",
        "verifier_tool_sha256",
        "docker_command_argv_sha256",
        "docker_command_policy_sha256",
        "docker_seccomp_sha256",
        "docker_daemon_receipt_sha256",
        "docker_control_plane_binding_sha256",
    )
    if (
        type(expected_commitment) is not bytes
        or len(expected_commitment) != 32
        or type(expected_daemon_binding) is not bytes
        or len(expected_daemon_binding) != 32
        or set(receipt) != required
        or receipt.get("schema") != SEED_CUSTODY_VERIFICATION_SCHEMA
        or receipt.get("verified") is not True
        or receipt.get("seed_commitment_hex") != expected_commitment.hex()
        or receipt.get("seed_length_bytes") != 32
        or any(
            type(receipt.get(field)) is not str
            or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field))) is None
            for field in digest_fields
        )
        or receipt.get("docker_daemon_receipt_sha256")
        != expected_daemon_binding.hex()
        or type(receipt.get("docker_image_ref")) is not str
        or re.fullmatch(
            r"[^\s@]+@sha256:[0-9a-f]{64}", str(receipt.get("docker_image_ref"))
        )
        is None
        or receipt.get("raw_seed_read_inside_keyless_verifier") is not True
        or receipt.get("raw_seed_exported") is not False
        or receipt.get("private_key_mount_present") is not False
        or receipt.get("state_mount_present") is not False
        or type(receipt.get("verifier_numeric_uid")) is not int
        or int(receipt.get("verifier_numeric_uid", -1)) < 0
        or type(receipt.get("verifier_numeric_gid")) is not int
        or int(receipt.get("verifier_numeric_gid", -1)) < 0
        or receipt.get("custody_artifacts_owned_by_verifier_identity") is not True
        or receipt.get("docker_network_mode") != "none"
        or receipt.get("docker_ipc_mode") != "private"
        or receipt.get("docker_read_only_rootfs") is not True
        or receipt.get("docker_stdout_limit_bytes") != 8192
        or receipt.get("docker_timeout_seconds") != 120
        or receipt.get("custody_owner_policy_id")
        != "EXACT_UNIFORM_CURRENT_OWNER_HOST_OR_65534_V1"
        or receipt.get("incarnation_fields_nonidentity") is not True
        or claimed_hash != hashlib.sha256(_canonical_json(body)).hexdigest()
    ):
        _fail(FAIL_CUSTODY, "keyless seed-custody verifier receipt differs")
    payload = _canonical_json(dict(receipt))
    assert_public_payload_contains_no_secret_fields(receipt)
    return payload


def _seed_custody_receipt_stable_identity_v1(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Remove only explicitly non-identity, per-incarnation execution fields."""

    projected = dict(receipt)
    for field in (
        "receipt_sha256",
        "inner_receipt_sha256",
        "docker_command_argv_sha256",
        "verifier_numeric_uid",
        "verifier_numeric_gid",
    ):
        projected.pop(field, None)
    return projected


def _replace_atomic_exact_payload_v1(
    path: Path,
    *,
    expected_old_payload: bytes,
    new_payload: bytes,
    mode: int,
    fault: Callable[[str], None] | None,
    fault_label: str,
) -> None:
    """Resume one exact old-to-new replacement and reject every third value."""

    temporary = path.with_name(path.name + ".next")

    def read_exact(candidate: Path, label: str) -> bytes:
        if candidate.is_symlink():
            _fail(FAIL_PUBLICATION, f"{label} may not be a symlink")
        try:
            metadata = candidate.stat()
            payload = candidate.read_bytes()
        except OSError as exc:
            _fail(FAIL_PUBLICATION, f"{label} cannot be read: {exc}")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
        ):
            _fail(FAIL_PUBLICATION, f"{label} inode differs")
        return payload

    current = read_exact(path, path.name)
    if current == new_payload:
        if temporary.exists() or temporary.is_symlink():
            _fail(FAIL_PUBLICATION, f"completed {path.name} retains a next inode")
        return
    if current != expected_old_payload:
        _fail(FAIL_PUBLICATION, f"{path.name} is not the exact old transition value")
    if temporary.exists() or temporary.is_symlink():
        if read_exact(temporary, temporary.name) != new_payload:
            _fail(FAIL_PUBLICATION, f"{temporary.name} differs from exact transition")
    else:
        if fault is not None:
            fault(f"before_transition_next_write_{fault_label}")
        _write_exclusive_durable_v1(temporary, new_payload, mode)
        if fault is not None:
            fault(f"after_transition_next_fsync_{fault_label}")
    os.replace(temporary, path)
    if fault is not None:
        fault(f"after_transition_rename_{fault_label}")
    _fsync_directory_v1(path.parent)
    if fault is not None:
        fault(f"after_transition_dir_fsync_{fault_label}")


class FormalCeremonyTransactionV1:
    """Persistent, fail-closed transaction around one ceremony publication.

    The lock, opaque-ID reservations, output reservations, staging directory,
    and journal are created with exclusive filesystem operations.  A failed
    transaction is intentionally not auto-deleted. The separately named
    post-stage rehydration procedure must validate and reuse its exact durable
    identity; ordinary execution may never redraw the split seed or replace an
    opaque ID. The staging tree contains public material only.
    """

    def __init__(
        self,
        *,
        basis_commit: str,
        custody_directory: Path,
        public_evidence_path: Path,
        public_promotion_path: Path,
        run_id: bytes,
        ledger_id: bytes,
        prestage_intent_fields: Mapping[str, object] | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.basis_commit = _commit(basis_commit)
        _reject_caller_symlink_chain_v1(custody_directory, "custody path")
        _reject_caller_symlink_chain_v1(public_evidence_path, "evidence path")
        _reject_caller_symlink_chain_v1(public_promotion_path, "promotion path")
        self.custody_directory = custody_directory.resolve()
        self.public_evidence_path = public_evidence_path.resolve()
        self.public_promotion_path = public_promotion_path.resolve()
        if type(run_id) is not bytes or len(run_id) != 16:
            _fail(FAIL_PREFLIGHT, "run ID must be 16 bytes")
        if type(ledger_id) is not bytes or len(ledger_id) != 16:
            _fail(FAIL_PREFLIGHT, "ledger ID must be 16 bytes")
        if run_id == ledger_id:
            _fail(FAIL_PREFLIGHT, "run and ledger IDs must differ")
        self.run_id = run_id
        self.ledger_id = ledger_id
        self._prestage_intent_fields = (
            None
            if prestage_intent_fields is None
            else validate_prestage_intent_fields_v1(
                dict(prestage_intent_fields),
                basis_commit=self.basis_commit,
                run_id=self.run_id,
                ledger_id=self.ledger_id,
            )
        )
        self._prestage_intent_bytes = (
            None
            if self._prestage_intent_fields is None
            else _canonical_json(self._prestage_intent_fields)
        )
        self.publication_receipt_path = self.public_promotion_path.with_name(
            self.public_promotion_path.name + ".publication-receipt.json"
        )
        self._output_reservation_paths = {
            path: path.with_name(f".{path.name}.hegel-reserved")
            for path in (
                self.public_evidence_path,
                self.public_promotion_path,
                self.publication_receipt_path,
            )
        }
        self._fault_injector = fault_injector
        self._directory_descriptor: int | None = None
        self._anchor_descriptor: int | None = None
        self._lock_descriptor: int | None = None
        self._stage_directory: Path | None = None
        self._journal_path: Path | None = None
        self._state = "NEW"
        self._staged_payloads: dict[str, bytes] = {}
        self._recovery_marker_snapshot: MarkerSnapshot | None = None
        self._recovery_phase: str | None = None
        self._staged_seed_commitment: bytes | None = None
        self._verified_seed_commitment: bytes | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def journal_path(self) -> Path | None:
        return self._journal_path

    @property
    def recovery_phase(self) -> str | None:
        return self._recovery_phase

    @property
    def recovery_marker_snapshot(self) -> MarkerSnapshot | None:
        return self._recovery_marker_snapshot

    def _lock_fields(self) -> dict[str, object]:
        if self._prestage_intent_fields is None or self._prestage_intent_bytes is None:
            prestage_transport: object = None
            artifact_specs: object = None
        else:
            prestage_transport = _transport(self._prestage_intent_fields)
            artifact_specs = _transport(self._reservation_artifact_specs_v1())
        return {
            "schema": TRANSACTION_LOCK_SCHEMA,
            "reservation_bootstrap_state": "RESERVING_EXACT_PREFIX",
            "basis_commit": self.basis_commit,
            "run_id_hex": self.run_id.hex(),
            "ledger_id_hex": self.ledger_id.hex(),
            "custody_directory": str(self.custody_directory),
            "public_output_parent": str(self.public_evidence_path.parent),
            "public_evidence_path": str(self.public_evidence_path),
            "public_promotion_path": str(self.public_promotion_path),
            "publication_receipt_path": str(self.publication_receipt_path),
            "stage_directory_name": ".hegel-m25-stage-" + self.run_id.hex(),
            "prestage_intent_sha256_or_null": (
                None
                if self._prestage_intent_bytes is None
                else hashlib.sha256(self._prestage_intent_bytes).hexdigest()
            ),
            "prestage_intent_transport_or_null": prestage_transport,
            "ordered_reservation_artifact_specs_or_null": artifact_specs,
            "recovery_required_if_incomplete": True,
            "bootstrap_completion_requires_exact_full_prefix": True,
        }

    def _opaque_reservation_fields(self, kind: str, value: bytes) -> dict[str, object]:
        return {
            "schema": "hegel-phase3-m25-opaque-id-reservation/1",
            "kind": kind,
            "opaque_id_hex": value.hex(),
            "basis_commit": self.basis_commit,
        }

    def _reservation_artifact_specs_v1(self) -> tuple[dict[str, object], ...]:
        """Return the immutable post-lock creation plan with full public bodies."""

        if self._prestage_intent_fields is None or self._prestage_intent_bytes is None:
            _fail(FAIL_TRANSACTION_LOCK, "reservation plan requires a prestage intent")
        stage = self.public_evidence_path.parent / (
            ".hegel-m25-stage-" + self.run_id.hex()
        )
        file_rows: tuple[tuple[str, Path, int, bytes], ...] = (
            (
                "opaque_run_reservation",
                self.custody_directory
                / f"opaque-run-{self.run_id.hex()}.reserved",
                0o600,
                _canonical_json(self._opaque_reservation_fields("run", self.run_id)),
            ),
            (
                "opaque_ledger_reservation",
                self.custody_directory
                / f"opaque-ledger-{self.ledger_id.hex()}.reserved",
                0o600,
                _canonical_json(
                    self._opaque_reservation_fields("ledger", self.ledger_id)
                ),
            ),
            (
                "public_evidence_reservation",
                self._output_reservation_paths[self.public_evidence_path],
                0o600,
                _canonical_json(
                    self._output_reservation_fields(self.public_evidence_path)
                ),
            ),
            (
                "public_promotion_reservation",
                self._output_reservation_paths[self.public_promotion_path],
                0o600,
                _canonical_json(
                    self._output_reservation_fields(self.public_promotion_path)
                ),
            ),
            (
                "publication_receipt_reservation",
                self._output_reservation_paths[self.publication_receipt_path],
                0o600,
                _canonical_json(
                    self._output_reservation_fields(self.publication_receipt_path)
                ),
            ),
        )
        specs: list[dict[str, object]] = [
            {
                "step": step,
                "inode_kind": "regular_file",
                "absolute_path": str(path),
                "mode_octal": "0600",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_transport": json.loads(payload),
            }
            for step, path, _mode, payload in file_rows
        ]
        specs.append({
            "step": "stage_directory",
            "inode_kind": "directory",
            "absolute_path": str(stage),
            "mode_octal": "0700",
            "payload_sha256": None,
            "payload_transport": None,
        })
        live_bundle = self._prestage_intent_fields.get(
            "live_actor_protocol_qualification_bundle"
        )
        if type(live_bundle) is not dict:
            _fail(
                FAIL_TRANSACTION_LOCK,
                "reservation plan lacks its transaction-local live qualification bundle",
            )
        live_bundle_payload = _canonical_json(live_bundle)
        for step, path, payload in (
            (
                "live_qualification_bundle",
                stage / _LIVE_QUALIFICATION_BUNDLE_FILENAME,
                live_bundle_payload,
            ),
            (
                "reserved_journal",
                stage / "transaction-journal.json",
                _canonical_json(self._journal_fields("RESERVED")),
            ),
            (
                "prestage_intent",
                stage / _PRESTAGE_INTENT_FILENAME,
                self._prestage_intent_bytes,
            ),
        ):
            specs.append({
                "step": step,
                "inode_kind": "regular_file",
                "absolute_path": str(path),
                "mode_octal": "0600",
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_transport": json.loads(payload),
            })
        if tuple(row["step"] for row in specs) != _RESERVATION_BOOTSTRAP_ORDER:
            raise AssertionError("reservation bootstrap order drifted")
        return tuple(specs)

    def _output_reservation_fields(self, path: Path) -> dict[str, object]:
        label_by_path = {
            self.public_evidence_path: "evidence",
            self.public_promotion_path: "promotion",
            self.publication_receipt_path: "receipt",
        }
        return {
            "schema": OUTPUT_RESERVATION_SCHEMA,
            "basis_commit": self.basis_commit,
            "run_id_hex": self.run_id.hex(),
            "output_kind": label_by_path[path],
            "output_path": str(path),
            "state": "RESERVED_NOT_PUBLIC",
        }

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)

    def _preflight_paths(self) -> None:
        outputs = (
            self.public_evidence_path,
            self.public_promotion_path,
            self.publication_receipt_path,
        )
        if len(set(outputs)) != 3:
            _fail(FAIL_PUBLICATION, "public output paths must be pairwise distinct")
        parents = {path.parent.resolve() for path in outputs}
        if len(parents) != 1:
            _fail(FAIL_PUBLICATION, "all public outputs must share one filesystem directory")
        parent = next(iter(parents))
        if not parent.is_dir() or parent.is_symlink():
            _fail(FAIL_PUBLICATION, "public output parent must be an existing real directory")
        for path in outputs:
            if path.exists() or path.is_symlink():
                _fail(FAIL_TRANSACTION_LOCK, f"public output is already occupied: {path.name}")
            retired_marker = _preseed_abort_retirement_marker_path_v1(path)
            if retired_marker.exists() or retired_marker.is_symlink():
                _fail(
                    FAIL_TRANSACTION_LOCK,
                    f"public output path is permanently retired: {path.name}",
                )
        retired_tombstone = self.public_evidence_path.with_name(
            f".{self.public_evidence_path.name}.hegel-preseed-abort-terminal.json"
        )
        if retired_tombstone.exists() or retired_tombstone.is_symlink():
            _fail(
                FAIL_TRANSACTION_LOCK,
                "public evidence path is permanently retired by a preseed abort tombstone",
            )
        try:
            self.custody_directory.relative_to(parent)
        except ValueError:
            pass
        else:
            _fail(FAIL_PUBLICATION, "public output directory may not contain custody state")

    def _reservation_step_materials_v1(
        self,
    ) -> tuple[tuple[str, str, Path, int, bytes | None], ...]:
        rows: list[tuple[str, str, Path, int, bytes | None]] = []
        for spec in self._reservation_artifact_specs_v1():
            step = str(spec["step"])
            kind = str(spec["inode_kind"])
            path = Path(str(spec["absolute_path"]))
            mode = int(str(spec["mode_octal"]), 8)
            payload = (
                None
                if kind == "directory"
                else _canonical_json(spec["payload_transport"])
            )
            if (
                (payload is None) != (spec["payload_sha256"] is None)
                or (
                    payload is not None
                    and hashlib.sha256(payload).hexdigest()
                    != spec["payload_sha256"]
                )
            ):
                raise AssertionError("reservation artifact plan digest drifted")
            rows.append((step, kind, path, mode, payload))
        return tuple(rows)

    @staticmethod
    def _require_exact_bootstrap_inode_v1(
        *, kind: str, path: Path, mode: int, payload: bytes | None
    ) -> None:
        if path.is_symlink():
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap artifact is a symlink: {path.name}")
        try:
            metadata = path.stat()
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap artifact cannot be stated: {exc}")
        if kind == "directory":
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
                _fail(FAIL_TRANSACTION_LOCK, f"bootstrap directory differs: {path.name}")
            return
        try:
            observed = path.read_bytes()
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap file cannot be read: {exc}")
        if (
            kind != "regular_file"
            or payload is None
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
            or observed != payload
        ):
            _fail(FAIL_TRANSACTION_LOCK, f"bootstrap file differs: {path.name}")

    def _validate_reservation_exact_prefix_v1(self) -> int:
        """Return the durable creation-prefix length; reject every gap/extra."""

        steps = self._reservation_step_materials_v1()
        present: list[bool] = []
        for _step, kind, path, mode, payload in steps:
            exists = path.exists() or path.is_symlink()
            present.append(exists)
            if exists:
                self._require_exact_bootstrap_inode_v1(
                    kind=kind, path=path, mode=mode, payload=payload
                )
        prefix_length = 0
        while prefix_length < len(present) and present[prefix_length]:
            prefix_length += 1
        if any(present[prefix_length:]):
            _fail(FAIL_TRANSACTION_LOCK, "reservation artifacts are not an exact prefix")

        expected_custody = {
            "phase3_m25_ceremony.lock",
            f"opaque-run-{self.run_id.hex()}.reserved",
            f"opaque-ledger-{self.ledger_id.hex()}.reserved",
        }
        actual_custody = {path.name for path in self.custody_directory.iterdir()}
        if not actual_custody <= expected_custody:
            _fail(FAIL_CUSTODY, "reservation bootstrap custody contains unknown state")
        stage = self.public_evidence_path.parent / (
            ".hegel-m25-stage-" + self.run_id.hex()
        )
        if stage.exists() and not stage.is_symlink():
            actual_stage = {path.name for path in stage.iterdir()}
            anchor_entries = {
                _RECOVERY_ANCHOR_FILENAME,
                _RECOVERY_ANCHOR_FILENAME + ".next",
                _RECOVERY_ANCHOR_READY_FILENAME,
                _RECOVERY_ANCHOR_READY_FILENAME + ".next",
            }
            if (
                bool(actual_stage & anchor_entries)
                and prefix_length != len(steps)
            ):
                _fail(
                    FAIL_TRANSACTION_LOCK,
                    "recovery anchor precedes complete reservation bootstrap",
                )
            if not actual_stage <= {
                _LIVE_QUALIFICATION_BUNDLE_FILENAME,
                "transaction-journal.json",
                _PRESTAGE_INTENT_FILENAME,
                *anchor_entries,
            }:
                _fail(FAIL_TRANSACTION_LOCK, "reservation stage contains unknown state")
        return prefix_length

    def _recover_bootstrap_next_prefix_v1(self) -> None:
        """Resolve only consecutive current-step ``.next`` crash states."""

        gap_seen = False
        for step, kind, path, mode, payload in self._reservation_step_materials_v1():
            next_path = path.with_name(path.name + ".next")
            target_present = path.exists() or path.is_symlink()
            next_present = next_path.exists() or next_path.is_symlink()
            if gap_seen:
                if target_present or next_present:
                    _fail(
                        FAIL_TRANSACTION_LOCK,
                        "bootstrap next/target artifacts are not an exact prefix",
                    )
                continue
            if kind == "directory":
                if next_present:
                    _fail(FAIL_TRANSACTION_LOCK, "bootstrap directory has an illegal next inode")
                if not target_present:
                    gap_seen = True
                continue
            assert payload is not None
            if target_present or next_present:
                _install_bootstrap_file_no_replace_v1(
                    path,
                    payload,
                    mode,
                    fault=self._fault,
                    label=step,
                )
            else:
                gap_seen = True

    def _build_recovery_anchor_fields_v1(self) -> dict[str, object]:
        if self._prestage_intent_fields is None or self._prestage_intent_bytes is None:
            _fail(FAIL_TRANSACTION_LOCK, "recovery anchor lacks prestage identity")
        stage = self.public_evidence_path.parent / (
            ".hegel-m25-stage-" + self.run_id.hex()
        )
        lock_path = self.custody_directory / "phase3_m25_ceremony.lock"
        evidence_reservation = self._output_reservation_paths[
            self.public_evidence_path
        ]
        try:
            custody_metadata = self.custody_directory.lstat()
            stage_metadata = stage.lstat()
            lock_metadata = lock_path.lstat()
            lock_payload = lock_path.read_bytes()
            evidence_reservation_payload = evidence_reservation.read_bytes()
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"recovery anchor input is absent: {exc}")
        runtime_bindings = self._prestage_intent_fields.get(
            "runtime_binding_fields"
        )
        daemon_binding = self._prestage_intent_fields.get(
            "live_actor_protocol_daemon_receipt_binding"
        )
        profile_digest = (
            None
            if not isinstance(runtime_bindings, Mapping)
            else runtime_bindings.get("actor_profile_sha256")
        )
        if (
            not stat.S_ISDIR(custody_metadata.st_mode)
            or stat.S_IMODE(custody_metadata.st_mode) != 0o700
            or (custody_metadata.st_uid, custody_metadata.st_gid)
            != (os.geteuid(), os.getegid())
            or not stat.S_ISDIR(stage_metadata.st_mode)
            or stat.S_IMODE(stage_metadata.st_mode) != 0o700
            or not stat.S_ISREG(lock_metadata.st_mode)
            or stat.S_IMODE(lock_metadata.st_mode) != 0o600
            or type(daemon_binding) is not bytes
            or len(daemon_binding) != 32
            or type(profile_digest) is not bytes
            or len(profile_digest) != 32
        ):
            _fail(FAIL_TRANSACTION_LOCK, "recovery anchor input identity differs")
        return {
            "schema": _RECOVERY_ANCHOR_SCHEMA,
            "basis_commit": self.basis_commit,
            "run_id_hex": self.run_id.hex(),
            "ledger_id_hex": self.ledger_id.hex(),
            "custody_absolute_path": str(self.custody_directory),
            "custody_st_dev": custody_metadata.st_dev,
            "custody_st_ino": custody_metadata.st_ino,
            "custody_mode_octal": "0700",
            "custody_owner_before_uid": custody_metadata.st_uid,
            "custody_owner_before_gid": custody_metadata.st_gid,
            "custody_owner_handoff_uid": 65534,
            "custody_owner_handoff_gid": 65534,
            "stage_absolute_path": str(stage),
            "stage_st_dev": stage_metadata.st_dev,
            "stage_st_ino": stage_metadata.st_ino,
            "internal_lock_st_dev": lock_metadata.st_dev,
            "internal_lock_st_ino": lock_metadata.st_ino,
            "internal_lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
            "prestage_intent_sha256": hashlib.sha256(
                self._prestage_intent_bytes
            ).hexdigest(),
            "public_evidence_path": str(self.public_evidence_path),
            "public_promotion_path": str(self.public_promotion_path),
            "publication_receipt_path": str(self.publication_receipt_path),
            "public_evidence_reservation_sha256": hashlib.sha256(
                evidence_reservation_payload
            ).hexdigest(),
            "actor_profile_sha256": profile_digest.hex(),
            "docker_daemon_receipt_sha256": daemon_binding.hex(),
            "owner_reclaim_policy_id": "PINNED_OFFLINE_CAP_CHOWN_ONLY_V1",
            "anchor_flock_held_before_ready": True,
            "original_execution_holds_anchor_until_close": True,
            "raw_seed_bytes_read_by_anchor_protocol": False,
        }

    def _validate_recovery_anchor_fields_v1(
        self,
        fields: Mapping[str, object],
        *,
        require_internal_lock_bytes: bool,
    ) -> None:
        required = {
            "schema", "basis_commit", "run_id_hex", "ledger_id_hex",
            "custody_absolute_path", "custody_st_dev", "custody_st_ino",
            "custody_mode_octal", "custody_owner_before_uid",
            "custody_owner_before_gid", "custody_owner_handoff_uid",
            "custody_owner_handoff_gid", "stage_absolute_path", "stage_st_dev",
            "stage_st_ino", "internal_lock_st_dev", "internal_lock_st_ino",
            "internal_lock_sha256", "prestage_intent_sha256",
            "public_evidence_path", "public_promotion_path",
            "publication_receipt_path", "public_evidence_reservation_sha256",
            "actor_profile_sha256", "docker_daemon_receipt_sha256",
            "owner_reclaim_policy_id", "anchor_flock_held_before_ready",
            "original_execution_holds_anchor_until_close",
            "raw_seed_bytes_read_by_anchor_protocol",
        }
        stage = self.public_evidence_path.parent / (
            ".hegel-m25-stage-" + self.run_id.hex()
        )
        try:
            custody_metadata = self.custody_directory.lstat()
            stage_metadata = stage.lstat()
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"recovery anchor inode is absent: {exc}")
        if (
            set(fields) != required
            or fields.get("schema") != _RECOVERY_ANCHOR_SCHEMA
            or fields.get("basis_commit") != self.basis_commit
            or fields.get("run_id_hex") != self.run_id.hex()
            or fields.get("ledger_id_hex") != self.ledger_id.hex()
            or fields.get("custody_absolute_path") != str(self.custody_directory)
            or fields.get("custody_st_dev") != custody_metadata.st_dev
            or fields.get("custody_st_ino") != custody_metadata.st_ino
            or fields.get("custody_mode_octal") != "0700"
            or (custody_metadata.st_uid, custody_metadata.st_gid)
            not in {
                (
                    fields.get("custody_owner_before_uid"),
                    fields.get("custody_owner_before_gid"),
                ),
                (
                    fields.get("custody_owner_handoff_uid"),
                    fields.get("custody_owner_handoff_gid"),
                ),
            }
            or fields.get("custody_owner_handoff_uid") != 65534
            or fields.get("custody_owner_handoff_gid") != 65534
            or fields.get("stage_absolute_path") != str(stage)
            or fields.get("stage_st_dev") != stage_metadata.st_dev
            or fields.get("stage_st_ino") != stage_metadata.st_ino
            or fields.get("public_evidence_path") != str(self.public_evidence_path)
            or fields.get("public_promotion_path") != str(self.public_promotion_path)
            or fields.get("publication_receipt_path") != str(
                self.publication_receipt_path
            )
            or fields.get("owner_reclaim_policy_id")
            != "PINNED_OFFLINE_CAP_CHOWN_ONLY_V1"
            or fields.get("anchor_flock_held_before_ready") is not True
            or fields.get("original_execution_holds_anchor_until_close") is not True
            or fields.get("raw_seed_bytes_read_by_anchor_protocol") is not False
            or any(
                type(fields.get(name)) is not str
                or re.fullmatch(r"[0-9a-f]{64}", str(fields.get(name))) is None
                for name in (
                    "internal_lock_sha256",
                    "prestage_intent_sha256",
                    "public_evidence_reservation_sha256",
                    "actor_profile_sha256",
                    "docker_daemon_receipt_sha256",
                )
            )
        ):
            _fail(FAIL_TRANSACTION_LOCK, "host recovery anchor fields differ")
        evidence_reservation = self._output_reservation_paths[
            self.public_evidence_path
        ]
        try:
            reservation_payload = evidence_reservation.read_bytes()
        except OSError as exc:
            _fail(
                FAIL_TRANSACTION_LOCK,
                f"anchor-bound evidence reservation is absent: {exc}",
            )
        if fields.get("public_evidence_reservation_sha256") != hashlib.sha256(
            reservation_payload
        ).hexdigest():
            _fail(FAIL_TRANSACTION_LOCK, "anchor-bound evidence reservation differs")
        if require_internal_lock_bytes:
            lock_path = self.custody_directory / "phase3_m25_ceremony.lock"
            try:
                lock_metadata = lock_path.lstat()
                lock_payload = lock_path.read_bytes()
            except OSError as exc:
                _fail(FAIL_TRANSACTION_LOCK, f"anchor-bound internal state is absent: {exc}")
            if (
                fields.get("internal_lock_st_dev") != lock_metadata.st_dev
                or fields.get("internal_lock_st_ino") != lock_metadata.st_ino
                or fields.get("internal_lock_sha256")
                != hashlib.sha256(lock_payload).hexdigest()
                or self._prestage_intent_bytes is None
                or fields.get("prestage_intent_sha256")
                != hashlib.sha256(self._prestage_intent_bytes).hexdigest()
            ):
                _fail(FAIL_TRANSACTION_LOCK, "anchor-bound internal bytes differ")

    def _ensure_recovery_anchor_locked_v1(self) -> None:
        if self._anchor_descriptor is not None:
            return
        if self._stage_directory is None:
            self._stage_directory = self.public_evidence_path.parent / (
                ".hegel-m25-stage-" + self.run_id.hex()
            )
        anchor_path = self._stage_directory / _RECOVERY_ANCHOR_FILENAME
        ready_path = self._stage_directory / _RECOVERY_ANCHOR_READY_FILENAME
        if anchor_path.exists() or anchor_path.is_symlink():
            anchor_payload, anchor_fields = self._read_canonical_regular_file(
                anchor_path,
                mode=0o600,
                code=FAIL_TRANSACTION_LOCK,
                label="host recovery anchor",
            )
            self._validate_recovery_anchor_fields_v1(
                anchor_fields, require_internal_lock_bytes=True
            )
        else:
            anchor_fields = self._build_recovery_anchor_fields_v1()
            anchor_payload = _canonical_json(anchor_fields)
            _write_atomic_resumable_v1(
                anchor_path,
                anchor_payload,
                0o600,
                fault=self._fault,
                fault_label="recovery_anchor",
            )
        try:
            descriptor = os.open(
                anchor_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            anchor_metadata = os.fstat(descriptor)
            path_metadata = anchor_path.lstat()
            if (
                not stat.S_ISREG(anchor_metadata.st_mode)
                or (anchor_metadata.st_dev, anchor_metadata.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                _fail(FAIL_TRANSACTION_LOCK, "recovery anchor lock inode differs")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if "descriptor" in locals():
                os.close(descriptor)
            _fail(FAIL_TRANSACTION_LOCK, f"recovery anchor is live-locked: {exc}")
        except BaseException:
            if "descriptor" in locals():
                os.close(descriptor)
            raise
        self._anchor_descriptor = descriptor
        self._fault("after_recovery_anchor_flock")
        ready_fields = {
            "schema": _RECOVERY_ANCHOR_READY_SCHEMA,
            "anchor_filename": _RECOVERY_ANCHOR_FILENAME,
            "anchor_sha256": hashlib.sha256(anchor_payload).hexdigest(),
            "state": "ANCHOR_FLOCK_ACQUIRED_BEFORE_READY",
            "raw_seed_bytes_read": False,
        }
        ready_payload = _canonical_json(ready_fields)
        _write_atomic_resumable_v1(
            ready_path,
            ready_payload,
            0o600,
            fault=self._fault,
            fault_label="recovery_anchor_ready",
        )
        self._fault("after_recovery_anchor_ready")

    def _acquire_custody_directory_lock_v1(self) -> None:
        """Serialize the pre-anchor bootstrap on the existing custody inode."""

        if self._directory_descriptor is not None:
            return
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self.custody_directory, flags)
            metadata = os.fstat(descriptor)
            path_metadata = self.custody_directory.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or (metadata.st_dev, metadata.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                _fail(FAIL_CUSTODY, "custody directory lock inode differs")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if "descriptor" in locals():
                os.close(descriptor)
            _fail(
                FAIL_TRANSACTION_LOCK,
                f"cannot acquire custody directory liveness lock: {exc}",
            )
        except BaseException:
            if "descriptor" in locals():
                os.close(descriptor)
            raise
        self._directory_descriptor = descriptor

    def reserve(self) -> None:
        self._acquire_custody_directory_lock_v1()
        try:
            self._reserve_with_directory_lock_held_v1()
        except BaseException:
            self.close_lock()
            raise

    def _reserve_with_directory_lock_held_v1(self) -> None:
        if self._state not in {"NEW", "RESERVING", "RESERVED"}:
            _fail(FAIL_TRANSACTION_LOCK, "transaction cannot enter reservation bootstrap")
        if self._prestage_intent_bytes is None:
            _fail(
                FAIL_TRANSACTION_LOCK,
                "formal transaction requires a bound prestage intent",
            )
        self._preflight_paths()
        if not self.custody_directory.is_dir():
            _fail(FAIL_CUSTODY, "transaction requires an existing custody directory")
        if (self.custody_directory.stat().st_mode & 0o777) != 0o700:
            _fail(FAIL_CUSTODY, "transaction custody directory must be mode 0700")

        lock_path = self.custody_directory / "phase3_m25_ceremony.lock"
        lock_payload = _canonical_json(self._lock_fields())
        lock_next_path = lock_path.with_name(lock_path.name + ".next")
        if not (lock_path.exists() or lock_path.is_symlink()):
            actual_names = {path.name for path in self.custody_directory.iterdir()}
            if not actual_names <= {lock_next_path.name}:
                _fail(
                    FAIL_CUSTODY,
                    "new reservation bootstrap contains state beyond its lock.next",
                )
            self._fault("before_reservation_persistent_lock")
            _install_bootstrap_file_no_replace_v1(
                lock_path,
                lock_payload,
                0o600,
                fault=self._fault,
                label="persistent_lock",
            )
            self._fault("after_reservation_persistent_lock")
        else:
            _install_bootstrap_file_no_replace_v1(
                lock_path,
                lock_payload,
                0o600,
                fault=self._fault,
                label="persistent_lock",
            )
        try:
            self._lock_descriptor = os.open(
                lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"cannot open persistent ceremony lock: {exc}")
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
            _fail(FAIL_TRANSACTION_LOCK, f"cannot acquire persistent ceremony lock: {exc}")

        try:
            self._state = "RESERVING"
            self._recover_bootstrap_next_prefix_v1()
            prefix_length = self._validate_reservation_exact_prefix_v1()
            steps = self._reservation_step_materials_v1()
            for step, kind, path, mode, payload in steps[prefix_length:]:
                self._fault(f"before_reservation_{step}")
                if kind == "directory":
                    try:
                        path.mkdir(mode=mode)
                    except OSError as exc:
                        _fail(
                            FAIL_TRANSACTION_LOCK,
                            f"cannot create reservation stage: {exc}",
                        )
                    _fsync_directory_v1(path.parent)
                else:
                    assert payload is not None
                    _install_bootstrap_file_no_replace_v1(
                        path,
                        payload,
                        mode,
                        fault=self._fault,
                        label=step,
                    )
                self._fault(f"after_reservation_{step}")
            if self._validate_reservation_exact_prefix_v1() != len(steps):
                _fail(FAIL_TRANSACTION_LOCK, "reservation bootstrap did not complete")
            stage = self.public_evidence_path.parent / (
                ".hegel-m25-stage-" + self.run_id.hex()
            )
            self._stage_directory = stage
            self._journal_path = stage / "transaction-journal.json"
            self._ensure_recovery_anchor_locked_v1()
            self._state = "RESERVED"
            self._fault("after_reservations")
        except BaseException:
            self.close_lock()
            raise

    def persist_actor_trust_checkpoint_v1(
        self, actor_trust: "ActorPublicKeysV1"
    ) -> dict[str, object]:
        """Durably freeze all four public key identities before seed genesis."""

        if (
            self._state != "RESERVED"
            or self._stage_directory is None
            or self._prestage_intent_bytes is None
        ):
            _fail(
                FAIL_TRANSACTION_LOCK,
                "actor-trust checkpoint requires one bound RESERVED prestage intent",
            )
        intent_hash = hashlib.sha256(self._prestage_intent_bytes).hexdigest()
        fields = build_actor_trust_checkpoint_fields_v1(
            actor_trust=actor_trust,
            basis_commit=self.basis_commit,
            run_id=self.run_id,
            ledger_id=self.ledger_id,
            prestage_intent_sha256=intent_hash,
        )
        payload = _canonical_json(fields)
        path = self._stage_directory / _ACTOR_TRUST_CHECKPOINT_FILENAME
        _write_atomic_resumable_v1(
            path,
            payload,
            0o600,
            fault=self._fault,
            fault_label="actor_trust_checkpoint",
        )
        self._fault("after_actor_trust_checkpoint_durable")
        return fields

    def load_actor_trust_checkpoint_v1(self) -> dict[str, object]:
        if self._stage_directory is None or self._prestage_intent_bytes is None:
            _fail(FAIL_TRANSACTION_LOCK, "bound actor-trust checkpoint context is absent")
        _payload, fields = self._read_canonical_regular_file(
            self._stage_directory / _ACTOR_TRUST_CHECKPOINT_FILENAME,
            mode=0o600,
            code=FAIL_CUSTODY,
            label="actor-trust checkpoint",
        )
        return fields

    def _journal_fields(self, state: str | None = None) -> dict[str, object]:
        journal_state = self._state if state is None else state
        return {
            "schema": TRANSACTION_JOURNAL_SCHEMA,
            "basis_commit": self.basis_commit,
            "run_id_hex": self.run_id.hex(),
            "ledger_id_hex": self.ledger_id.hex(),
            "state": journal_state,
            "marker_complete": journal_state in {
                "MARKER_COMPLETE", "ACTORS_ABSENT", "PUBLISHED"
            },
            "actors_absent": journal_state in {"ACTORS_ABSENT", "PUBLISHED"},
            "public_outputs_complete": journal_state == "PUBLISHED",
            "pending_recovery_protocol_frozen": True,
        }

    def _write_journal(self) -> None:
        if self._journal_path is None:
            raise AssertionError("transaction journal path is absent")
        payload = _canonical_json(self._journal_fields())
        if not self._journal_path.exists():
            _write_exclusive_durable_v1(self._journal_path, payload, 0o600)
            return
        temporary = self._journal_path.with_name("transaction-journal.next")
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_symlink():
                _fail(FAIL_TRANSACTION_LOCK, "next transaction journal is a symlink")
            try:
                metadata = temporary.stat()
                observed = temporary.read_bytes()
            except OSError as exc:
                _fail(FAIL_TRANSACTION_LOCK, f"next journal cannot be read: {exc}")
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or observed != payload
            ):
                _fail(FAIL_TRANSACTION_LOCK, "next journal differs from requested transition")
        else:
            self._fault("before_journal_next_write")
            _write_exclusive_durable_v1(temporary, payload, 0o600)
            self._fault("after_journal_next_fsync")
        os.replace(temporary, self._journal_path)
        _fsync_directory_v1(self._journal_path.parent)

    @staticmethod
    def _read_canonical_regular_file(
        path: Path, *, mode: int, code: str, label: str
    ) -> tuple[bytes, dict[str, object]]:
        if path.is_symlink():
            _fail(code, f"{label} may not be a symlink")
        try:
            metadata = path.stat()
            payload = path.read_bytes()
            value = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(code, f"{label} cannot be loaded: {exc}")
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != mode
            or type(value) is not dict
            or payload != _canonical_json(value)
        ):
            _fail(code, f"{label} is not canonical regular mode {mode:04o}")
        return payload, value

    @classmethod
    def _load_bound_prestage_intent(
        cls,
        *,
        stage_directory: Path,
        expected_sha256_or_null: object,
        basis_commit: str,
        run_id: bytes,
        ledger_id: bytes,
        required: bool,
    ) -> dict[str, object] | None:
        path = stage_directory / _PRESTAGE_INTENT_FILENAME
        if expected_sha256_or_null is None:
            if required or path.exists() or path.is_symlink():
                _fail(FAIL_TRANSACTION_LOCK, "bound prestage intent is absent")
            return None
        if (
            type(expected_sha256_or_null) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256_or_null) is None
        ):
            _fail(FAIL_TRANSACTION_LOCK, "prestage intent lock digest is invalid")
        payload, transported = cls._read_canonical_regular_file(
            path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="prestage intent",
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256_or_null:
            _fail(FAIL_TRANSACTION_LOCK, "prestage intent differs from persistent lock")
        return validate_prestage_intent_fields_v1(
            transported,
            basis_commit=basis_commit,
            run_id=run_id,
            ledger_id=ledger_id,
        )

    @classmethod
    def _load_bound_live_qualification_bundle_v1(
        cls,
        *,
        stage_directory: Path,
        prestage_intent: Mapping[str, object],
    ) -> dict[str, object]:
        payload, bundle = cls._read_canonical_regular_file(
            stage_directory / _LIVE_QUALIFICATION_BUNDLE_FILENAME,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="transaction-local live qualification bundle",
        )
        expected_bundle = prestage_intent.get(
            "live_actor_protocol_qualification_bundle"
        )
        expected_sha256 = prestage_intent.get(
            "live_actor_protocol_qualification_bundle_sha256"
        )
        if (
            type(expected_bundle) is not dict
            or type(expected_sha256) is not bytes
            or len(expected_sha256) != 32
            or bundle != expected_bundle
            or payload != _canonical_json(expected_bundle)
            or hashlib.sha256(payload).digest() != expected_sha256
        ):
            _fail(
                FAIL_TRANSACTION_LOCK,
                "transaction-local live qualification bundle differs from intent",
            )
        return bundle

    @classmethod
    def rehydrate_reservation_bootstrap_v1(
        cls,
        *,
        custody_directory: Path,
        public_evidence_path: Path,
        public_promotion_path: Path,
        fault_injector: Callable[[str], None] | None = None,
    ) -> "FormalCeremonyTransactionV1":
        """Complete one exact pre-actor creation prefix without new entropy.

        The immutable persistent lock is the full plan and contains the exact
        prestage intent selected before the first artifact was created.  This
        path accepts only the frozen creation prefix and delegates completion
        to ``reserve``; it cannot choose a new ID, report, path, or body.
        """

        custody = custody_directory.resolve()
        lock_path = custody / "phase3_m25_ceremony.lock"
        _raw, lock_fields = cls._read_canonical_regular_file(
            lock_path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="reservation bootstrap lock",
        )
        try:
            run_id = bytes.fromhex(str(lock_fields["run_id_hex"]))
            ledger_id = bytes.fromhex(str(lock_fields["ledger_id_hex"]))
            restored_intent = _restore(
                lock_fields["prestage_intent_transport_or_null"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            _fail(
                FAIL_TRANSACTION_LOCK,
                f"reservation bootstrap identity cannot be restored: {exc}",
            )
        if type(restored_intent) is not dict:
            _fail(FAIL_TRANSACTION_LOCK, "reservation bootstrap intent is absent")
        transaction = cls(
            basis_commit=lock_fields.get("basis_commit"),  # type: ignore[arg-type]
            custody_directory=custody,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            run_id=run_id,
            ledger_id=ledger_id,
            prestage_intent_fields=restored_intent,
            fault_injector=fault_injector,
        )
        if lock_fields != transaction._lock_fields():
            _fail(
                FAIL_TRANSACTION_LOCK,
                "reservation bootstrap caller/plan differs from persistent lock",
            )
        transaction.reserve()
        return transaction

    def _validate_staged_transaction_identity(
        self, inputs: GateEvidenceInputsV1, receipt: Mapping[str, object]
    ) -> None:
        if inputs.basis_commit != self.basis_commit:
            _fail(FAIL_TRANSACTION_LOCK, "staged basis commit differs from persistent lock")
        run_maps = (
            inputs.execution_candidate_fields,
            inputs.bridge_statement_fields,
            inputs.execution_manifest_fields,
            inputs.run_genesis_fields,
        )
        if any(fields.get("run_id") != self.run_id for fields in run_maps):
            _fail(FAIL_TRANSACTION_LOCK, "staged run ID differs from persistent lock")
        if inputs.ledger_genesis_fields.get("ledger_id") != self.ledger_id:
            _fail(FAIL_TRANSACTION_LOCK, "staged ledger ID differs from persistent lock")
        expected_ids = {1: self.run_id, 2: self.ledger_id}
        for label, rows in (
            ("intent", inputs.opaque_registration_intents),
            ("record", inputs.opaque_registry_records),
        ):
            observed: dict[int, bytes] = {}
            for row in rows:
                kind = row.get("opaque_id_kind_id")
                value = row.get("opaque_id_16_bytes")
                if type(kind) is not int or type(value) is not bytes or kind in observed:
                    _fail(FAIL_TRANSACTION_LOCK, f"staged opaque {label} set is not unique")
                observed[kind] = value
            if observed != expected_ids:
                _fail(
                    FAIL_TRANSACTION_LOCK,
                    f"staged opaque {label} IDs differ from the two reservations",
                )
        if (
            receipt.get("basis_commit") != self.basis_commit
            or receipt.get("run_id_hex") != self.run_id.hex()
            or receipt.get("ledger_id_hex") != self.ledger_id.hex()
        ):
            _fail(FAIL_TRANSACTION_LOCK, "staged publication receipt identity differs")

    def _load_and_verify_stage(
        self,
        *,
        replay: Callable[[Mapping[str, object]], Mapping[str, object]],
    ) -> tuple[GateEvidenceInputsV1, MarkerSnapshot]:
        if self._stage_directory is None:
            _fail(FAIL_TRANSACTION_LOCK, "post-stage recovery stage path is absent")
        staged: dict[str, bytes] = {}
        decoded: dict[str, dict[str, object]] = {}
        for label, filename in _STAGED_FILENAMES.items():
            payload, value = self._read_canonical_regular_file(
                self._stage_directory / filename,
                mode=0o600,
                code=FAIL_PUBLICATION,
                label=f"staged {label}",
            )
            staged[label] = payload
            decoded[label] = value
        inputs = load_gate_evidence_inputs_v1(decoded["evidence"])
        if self._prestage_intent_fields is None or self._prestage_intent_bytes is None:
            _fail(FAIL_TRANSACTION_LOCK, "post-stage replay lacks prestage identity")
        else:
            try:
                public_keys = {
                    int(manifest["purpose_id"]): manifest["public_key_32_bytes"]
                    for manifest in inputs.actor_key_manifests
                }
                trust_id = bytes.fromhex(
                    str(self._prestage_intent_fields["trust_genesis_id_hex"])
                )
                timestamp = int(
                    self._prestage_intent_fields["created_at_unix_seconds"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                _fail(FAIL_CUSTODY, f"staged actor-trust identity is invalid: {exc}")
            actor_trust = build_actor_trust_v1(
                public_keys=public_keys,  # type: ignore[arg-type]
                timestamp=timestamp,
                basis_commit=self.basis_commit,
                trust_genesis_id=trust_id,
            )
            if (
                inputs.actor_key_manifests != actor_trust.manifests
                or inputs.replacement_policy_fields
                != actor_trust.replacement_policy_fields
                or inputs.trust_genesis_fields != actor_trust.trust_genesis_fields
            ):
                _fail(FAIL_CUSTODY, "staged formal actor trust differs from prestage identity")
            validate_actor_trust_checkpoint_fields_v1(
                self.load_actor_trust_checkpoint_v1(),
                expected_actor_trust=actor_trust,
                basis_commit=self.basis_commit,
                run_id=self.run_id,
                ledger_id=self.ledger_id,
                prestage_intent_sha256=hashlib.sha256(
                    self._prestage_intent_bytes
                ).hexdigest(),
            )
        replayed = replay(decoded["evidence"])
        if _canonical_json(replayed) != staged["promotion"]:
            _fail(FAIL_PUBLICATION, "post-stage recovery replay differs from staged promotion")
        receipt = decoded["receipt"]
        expected_receipt_fields = {
            "schema",
            "basis_commit",
            "run_id_hex",
            "ledger_id_hex",
            "public_evidence_sha256",
            "public_promotion_sha256",
            "seed_custody_verification_receipt_sha256_or_null",
            "prospective_public_replay_passed",
            "marker_was_complete_during_staging",
            "actor_cleanup_required_before_publication",
            "authority_disclosure",
            "contains_private_key",
            "contains_raw_split_seed",
            "contains_split_assignment_rows",
        }
        if (
            set(receipt) != expected_receipt_fields
            or receipt.get("schema") != PUBLICATION_RECEIPT_SCHEMA
            or receipt.get("public_evidence_sha256")
            != hashlib.sha256(staged["evidence"]).hexdigest()
            or receipt.get("public_promotion_sha256")
            != hashlib.sha256(staged["promotion"]).hexdigest()
            or (
                receipt.get(
                    "seed_custody_verification_receipt_sha256_or_null"
                )
                is not None
                and (
                    type(
                        receipt.get(
                            "seed_custody_verification_receipt_sha256_or_null"
                        )
                    )
                    is not str
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(
                            receipt.get(
                                "seed_custody_verification_receipt_sha256_or_null"
                            )
                        ),
                    )
                    is None
                )
            )
            or receipt.get("prospective_public_replay_passed") is not True
            or receipt.get("marker_was_complete_during_staging") is not False
            or receipt.get("actor_cleanup_required_before_publication") is not True
            or receipt.get("authority_disclosure")
            != dict(TECHNICAL_ACTOR_DISCLOSURE_V1)
            or any(
                receipt.get(flag) is not False
                for flag in (
                    "contains_private_key",
                    "contains_raw_split_seed",
                    "contains_split_assignment_rows",
                )
            )
        ):
            _fail(FAIL_PUBLICATION, "staged publication receipt fields differ")
        assert_public_payload_contains_no_secret_fields(receipt)
        self._validate_staged_transaction_identity(inputs, receipt)
        expected_marker = inputs.marker_snapshot
        staged_seed_commitment = inputs.split_seed_commitment_fields.get(
            "split_seed_commitment_digest"
        )
        if (
            expected_marker.state != "COMPLETE"
            or expected_marker.seed_commitment_manifest_root is None
            or type(staged_seed_commitment) is not bytes
            or len(staged_seed_commitment) != 32
        ):
            _fail(FAIL_CUSTODY, "staged marker snapshot is not exact COMPLETE")
        daemon_binding = self._prestage_intent_fields.get(
            "live_actor_protocol_daemon_receipt_binding"
        )
        if type(daemon_binding) is not bytes or len(daemon_binding) != 32:
            _fail(FAIL_CUSTODY, "staged seed verifier daemon binding is absent")
        verification_path = (
            self._stage_directory / _SEED_CUSTODY_VERIFICATION_FILENAME
        )
        verification_digest: str | None = None
        if verification_path.exists() or verification_path.is_symlink():
            verification_payload, verification_fields = (
                self._read_canonical_regular_file(
                    verification_path,
                    mode=0o600,
                    code=FAIL_CUSTODY,
                    label="durable seed-custody verification",
                )
            )
            _validate_seed_custody_verification_receipt_v1(
                verification_fields,
                expected_commitment=staged_seed_commitment,
                expected_daemon_binding=daemon_binding,
            )
            verification_digest = hashlib.sha256(verification_payload).hexdigest()
        receipt_verification_digest = receipt.get(
            "seed_custody_verification_receipt_sha256_or_null"
        )
        if receipt_verification_digest not in {None, verification_digest}:
            _fail(FAIL_CUSTODY, "staged seed-verifier receipt binding differs")
        if self._state in {
            "SEED_CUSTODY_VERIFIED",
            "MARKER_COMPLETE",
            "ACTORS_ABSENT",
            "PUBLISHED",
        } and (
            verification_digest is None
            or receipt_verification_digest != verification_digest
        ):
            _fail(FAIL_CUSTODY, "verified transaction lacks its durable seed receipt")
        self._staged_payloads = staged
        self._staged_seed_commitment = staged_seed_commitment
        return inputs, expected_marker

    @classmethod
    def rehydrate_post_stage_v1(
        cls,
        *,
        custody_directory: Path,
        public_evidence_path: Path,
        public_promotion_path: Path,
        fault_injector: Callable[[str], None] | None = None,
        replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
        actors: "CeremonyActorsV1 | None" = None,
    ) -> "FormalCeremonyTransactionV1":
        """Acquire the host recovery anchor, reclaim if needed, then rehydrate.

        Before actor cleanup the custody inode may still be owned by the
        sealed nobody actor and therefore intentionally untraversable by the
        host.  The public evidence reservation locates the host-owned recovery
        anchor without traversing custody.  Its nonblocking flock is acquired
        before an exact Docker reclaim and is retained by the returned
        transaction.  Once publication has removed that reservation, custody
        must already be host-owned and the persistent internal lock remains
        the liveness barrier.
        """

        evidence = Path(os.path.abspath(os.fspath(public_evidence_path)))
        reservation = evidence.with_name(f".{evidence.name}.hegel-reserved")
        anchor_identity: FormalCeremonyTransactionV1 | None = None
        if reservation.exists() or reservation.is_symlink():
            anchor_identity, _anchor = _acquire_host_recovery_anchor_v1(
                custody_directory=custody_directory,
                public_evidence_path=public_evidence_path,
                public_promotion_path=public_promotion_path,
                actors=actors,
            )
        try:
            transaction = cls._rehydrate_post_stage_host_owned_v1(
                custody_directory=custody_directory,
                public_evidence_path=public_evidence_path,
                public_promotion_path=public_promotion_path,
                fault_injector=fault_injector,
                replay=replay,
            )
            if anchor_identity is not None:
                if (
                    transaction.basis_commit != anchor_identity.basis_commit
                    or transaction.run_id != anchor_identity.run_id
                    or transaction.ledger_id != anchor_identity.ledger_id
                    or transaction._anchor_descriptor is not None
                    or transaction._directory_descriptor is not None
                ):
                    transaction.close_lock()
                    _fail(
                        FAIL_TRANSACTION_LOCK,
                        "post-stage recovery anchor identity differs",
                    )
                transaction._anchor_descriptor = anchor_identity._anchor_descriptor
                anchor_identity._anchor_descriptor = None
                transaction._directory_descriptor = anchor_identity._directory_descriptor
                anchor_identity._directory_descriptor = None
            return transaction
        finally:
            if anchor_identity is not None:
                anchor_identity.close_lock()

    @classmethod
    def _rehydrate_post_stage_host_owned_v1(
        cls,
        *,
        custody_directory: Path,
        public_evidence_path: Path,
        public_promotion_path: Path,
        fault_injector: Callable[[str], None] | None = None,
        replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
    ) -> "FormalCeremonyTransactionV1":
        """Reopen one exact durable post-stage transaction without new entropy.

        The persistent lock is the sole source of the Commit-A/run/ledger and
        output-path identities.  Recovery never calls ``secrets``, never
        creates a reservation, and rejects every artifact that is not the
        exact canonical byte sequence expected for its derived crash phase.
        """

        custody = custody_directory.resolve()
        if (
            custody.is_symlink()
            or not custody.is_dir()
            or stat.S_IMODE(custody.stat().st_mode) != 0o700
        ):
            _fail(FAIL_CUSTODY, "post-stage recovery custody directory is invalid")
        lock_path = custody / "phase3_m25_ceremony.lock"
        raw_lock, lock_fields = cls._read_canonical_regular_file(
            lock_path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="persistent ceremony lock",
        )
        del raw_lock
        if (
            set(lock_fields)
            != {
                "schema",
                "reservation_bootstrap_state",
                "basis_commit",
                "run_id_hex",
                "ledger_id_hex",
                "custody_directory",
                "public_output_parent",
                "public_evidence_path",
                "public_promotion_path",
                "publication_receipt_path",
                "stage_directory_name",
                "prestage_intent_sha256_or_null",
                "prestage_intent_transport_or_null",
                "ordered_reservation_artifact_specs_or_null",
                "recovery_required_if_incomplete",
                "bootstrap_completion_requires_exact_full_prefix",
            }
            or lock_fields.get("schema") != TRANSACTION_LOCK_SCHEMA
            or lock_fields.get("reservation_bootstrap_state")
            != "RESERVING_EXACT_PREFIX"
            or lock_fields.get("recovery_required_if_incomplete") is not True
            or lock_fields.get("bootstrap_completion_requires_exact_full_prefix")
            is not True
        ):
            _fail(FAIL_TRANSACTION_LOCK, "persistent ceremony lock fields differ")
        try:
            run_id = bytes.fromhex(lock_fields["run_id_hex"])
            ledger_id = bytes.fromhex(lock_fields["ledger_id_hex"])
        except (KeyError, TypeError, ValueError) as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"persistent opaque ID is invalid: {exc}")
        expected_stage_name = ".hegel-m25-stage-" + run_id.hex()
        expected_receipt_path = public_promotion_path.resolve().with_name(
            public_promotion_path.resolve().name + ".publication-receipt.json"
        )
        if (
            lock_fields.get("stage_directory_name") != expected_stage_name
            or lock_fields.get("public_evidence_path")
            != str(public_evidence_path.resolve())
            or lock_fields.get("public_promotion_path")
            != str(public_promotion_path.resolve())
            or lock_fields.get("publication_receipt_path")
            != str(expected_receipt_path)
        ):
            _fail(FAIL_TRANSACTION_LOCK, "caller paths or stage identity differ from lock")
        prestage_stage = public_evidence_path.resolve().parent / expected_stage_name
        if (
            prestage_stage.is_symlink()
            or not prestage_stage.is_dir()
            or stat.S_IMODE(prestage_stage.stat().st_mode) != 0o700
        ):
            _fail(FAIL_TRANSACTION_LOCK, "post-stage directory is invalid")
        prestage_intent = cls._load_bound_prestage_intent(
            stage_directory=prestage_stage,
            expected_sha256_or_null=lock_fields.get(
                "prestage_intent_sha256_or_null"
            ),
            basis_commit=lock_fields.get("basis_commit"),  # type: ignore[arg-type]
            run_id=run_id,
            ledger_id=ledger_id,
            required=True,
        )
        assert prestage_intent is not None
        cls._load_bound_live_qualification_bundle_v1(
            stage_directory=prestage_stage,
            prestage_intent=prestage_intent,
        )
        transaction = cls(
            basis_commit=lock_fields.get("basis_commit"),  # type: ignore[arg-type]
            custody_directory=custody,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
            run_id=run_id,
            ledger_id=ledger_id,
            prestage_intent_fields=prestage_intent,
            fault_injector=fault_injector,
        )
        if lock_fields != transaction._lock_fields():
            _fail(FAIL_TRANSACTION_LOCK, "caller paths or transaction identity differ from lock")
        outputs = (
            transaction.public_evidence_path,
            transaction.public_promotion_path,
            transaction.publication_receipt_path,
        )
        if len(set(outputs)) != 3 or len({path.parent for path in outputs}) != 1:
            _fail(FAIL_PUBLICATION, "recovery output paths are not one exact directory")
        if outputs[0].parent.is_symlink() or not outputs[0].parent.is_dir():
            _fail(FAIL_PUBLICATION, "recovery output parent is invalid")
        descriptor = os.open(lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            _fail(FAIL_TRANSACTION_LOCK, f"another ceremony/recovery holds the lock: {exc}")
        transaction._lock_descriptor = descriptor
        try:
            for kind, value in (("run", run_id), ("ledger", ledger_id)):
                path = custody / f"opaque-{kind}-{value.hex()}.reserved"
                raw, fields = cls._read_canonical_regular_file(
                    path,
                    mode=0o600,
                    code=FAIL_TRANSACTION_LOCK,
                    label=f"opaque {kind} reservation",
                )
                expected = {
                    "schema": "hegel-phase3-m25-opaque-id-reservation/1",
                    "kind": kind,
                    "opaque_id_hex": value.hex(),
                    "basis_commit": transaction.basis_commit,
                }
                if fields != expected or raw != _canonical_json(expected):
                    _fail(FAIL_TRANSACTION_LOCK, f"opaque {kind} reservation differs")

            stage = outputs[0].parent / str(lock_fields["stage_directory_name"])
            if (
                stage.is_symlink()
                or not stage.is_dir()
                or stat.S_IMODE(stage.stat().st_mode) != 0o700
            ):
                _fail(FAIL_TRANSACTION_LOCK, "post-stage directory is invalid")
            transaction._stage_directory = stage
            transaction._journal_path = stage / "transaction-journal.json"
            _journal_bytes, journal = cls._read_canonical_regular_file(
                transaction._journal_path,
                mode=0o600,
                code=FAIL_TRANSACTION_LOCK,
                label="transaction journal",
            )
            main_state = journal.get("state")
            if main_state not in _TRANSACTION_STATES:
                _fail(FAIL_TRANSACTION_LOCK, "transaction journal state is unknown")
            if journal != transaction._journal_fields(str(main_state)):
                _fail(FAIL_TRANSACTION_LOCK, "transaction journal fields differ")
            entries = {path.name for path in stage.iterdir()}
            expected_entries_without_seed_verification = {
                "transaction-journal.json",
                *_STAGED_FILENAMES.values(),
            }
            expected_entries_without_seed_verification.update({
                _PRESTAGE_INTENT_FILENAME,
                _LIVE_QUALIFICATION_BUNDLE_FILENAME,
                _ACTOR_TRUST_CHECKPOINT_FILENAME,
                _RECOVERY_ANCHOR_FILENAME,
                _RECOVERY_ANCHOR_READY_FILENAME,
            })
            expected_entries = (
                expected_entries_without_seed_verification
                | {_SEED_CUSTODY_VERIFICATION_FILENAME}
            )
            next_path = stage / "transaction-journal.next"
            promote_next_journal = False
            if frozenset(entries) in {
                frozenset(
                    expected_entries_without_seed_verification
                    | {"transaction-journal.next"}
                ),
                frozenset(expected_entries | {"transaction-journal.next"}),
            }:
                _next_bytes, next_journal = cls._read_canonical_regular_file(
                    next_path,
                    mode=0o600,
                    code=FAIL_TRANSACTION_LOCK,
                    label="next transaction journal",
                )
                next_state = next_journal.get("state")
                if (
                    next_state not in _TRANSACTION_STATES
                    or next_journal != transaction._journal_fields(str(next_state))
                    or _TRANSACTION_STATES.index(str(next_state))
                    != _TRANSACTION_STATES.index(str(main_state)) + 1
                ):
                    _fail(FAIL_TRANSACTION_LOCK, "next transaction journal is not one exact step")
                journal = next_journal
                state = next_state
                promote_next_journal = True
            elif frozenset(entries) in {
                frozenset(expected_entries_without_seed_verification),
                frozenset(expected_entries),
            }:
                state = main_state
            else:
                _fail(FAIL_TRANSACTION_LOCK, "post-stage directory field set differs")
            promote_reserved_stage = state == "RESERVED"
            if promote_reserved_stage:
                # All three stage files were already fsynced and are replayed
                # below. A crash immediately before the STAGED journal write
                # may therefore advance exactly this one journal edge.
                state = "STAGED_PROSPECTIVE_REPLAY_PASSED"
            elif state not in _TRANSACTION_STATES[1:]:
                _fail(FAIL_TRANSACTION_LOCK, "transaction is not at a post-stage state")
            transaction._state = str(state)
            inputs, expected_marker = transaction._load_and_verify_stage(replay=replay)

            marker_path = custody / "split_seed_instantiation.marker"
            marker = read_marker_snapshot_v1(marker_path)
            marker_next_path = marker_path.with_name(marker_path.name + ".complete.tmp")
            promote_complete_marker = False
            if marker_next_path.exists() or marker_next_path.is_symlink():
                if marker_next_path.is_symlink():
                    _fail(FAIL_CUSTODY, "next COMPLETE marker may not be a symlink")
                marker_next = read_marker_snapshot_v1(marker_next_path)
                if (
                    marker.state != "PENDING"
                    or marker_next != expected_marker
                    or state != "SEED_CUSTODY_VERIFIED"
                ):
                    _fail(FAIL_CUSTODY, "next COMPLETE marker is not one exact transition")
                marker = marker_next
                promote_complete_marker = True
            if marker.state == "PENDING":
                pending_expected = MarkerSnapshot(
                    "PENDING",
                    expected_marker.split_version_digest,
                    None,
                    expected_marker.custodian_key_id,
                    expected_marker.created_at_unix_seconds,
                )
                if marker != pending_expected or state not in {
                    "STAGED_PROSPECTIVE_REPLAY_PASSED",
                    "SEED_CUSTODY_VERIFIED",
                }:
                    _fail(FAIL_CUSTODY, "PENDING marker/journal differs from staged snapshot")
            elif marker.state == "COMPLETE":
                if marker != expected_marker:
                    _fail(FAIL_CUSTODY, "COMPLETE marker differs from staged snapshot")
            else:
                _fail(FAIL_CUSTODY, "post-stage marker state is invalid")

            seed_paths = {
                "split_seed_generation.intent": None,
                "split_seed_generation.complete": None,
                "split_master_seed.bin": None,
            }
            expected_custody_names = {
                "phase3_m25_ceremony.lock",
                f"opaque-run-{run_id.hex()}.reserved",
                f"opaque-ledger-{ledger_id.hex()}.reserved",
                "split_seed_instantiation.marker",
                *seed_paths,
            }
            if promote_complete_marker:
                expected_custody_names.add(marker_next_path.name)
            if {path.name for path in custody.iterdir()} != expected_custody_names:
                _fail(FAIL_CUSTODY, "post-stage custody path set differs")
            intent_path = custody / "split_seed_generation.intent"
            completion_path = custody / "split_seed_generation.complete"
            seed_path = custody / "split_master_seed.bin"
            intent_payload, intent = cls._read_canonical_regular_file(
                intent_path,
                mode=0o600,
                code=FAIL_CUSTODY,
                label="seed generation intent",
            )
            if intent != {
                "schema": "hegel-phase3-m25-seed-generation-intent/1",
                "state": "CSPRNG_CALL_COMMITTED_NO_REDRAW",
            }:
                _fail(FAIL_CUSTODY, "seed generation intent differs")
            _completion_payload, completion = cls._read_canonical_regular_file(
                completion_path,
                mode=0o600,
                code=FAIL_CUSTODY,
                label="seed generation completion receipt",
            )
            seed_commitment = inputs.split_seed_commitment_fields.get(
                "split_seed_commitment_digest"
            )
            if (
                set(completion)
                != {
                    "attempt",
                    "intent_sha256",
                    "schema",
                    "seed_commitment_hex",
                    "seed_length_bytes",
                }
                or completion.get("attempt") != 1
                or completion.get("intent_sha256")
                != hashlib.sha256(intent_payload).hexdigest()
                or completion.get("schema")
                != "hegel-phase3-m25-seed-generation-complete/1"
                or type(seed_commitment) is not bytes
                or completion.get("seed_commitment_hex") != seed_commitment.hex()
                or completion.get("seed_length_bytes") != 32
            ):
                _fail(FAIL_CUSTODY, "seed completion receipt differs from staged commitment")
            if seed_path.is_symlink():
                _fail(FAIL_CUSTODY, "raw seed path may not be a symlink")
            try:
                seed_metadata = seed_path.stat()
            except OSError as exc:
                _fail(FAIL_CUSTODY, f"raw seed metadata cannot be read: {exc}")
            if (
                not stat.S_ISREG(seed_metadata.st_mode)
                or stat.S_IMODE(seed_metadata.st_mode) != 0o600
                or seed_metadata.st_size != 32
            ):
                _fail(FAIL_CUSTODY, "raw seed inode differs; bytes were not read")

            present_labels: list[str] = []
            for label, output in zip(("evidence", "promotion", "receipt"), outputs, strict=True):
                reservation = transaction._output_reservation_paths[output]
                if output.exists() or output.is_symlink():
                    if output.is_symlink():
                        _fail(FAIL_PUBLICATION, f"published {label} may not be a symlink")
                    metadata = output.stat()
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or stat.S_IMODE(metadata.st_mode) != 0o644
                        or output.read_bytes() != transaction._staged_payloads[label]
                    ):
                        _fail(FAIL_PUBLICATION, f"published {label} differs from stage")
                    present_labels.append(label)
                if reservation.exists() or reservation.is_symlink():
                    raw, fields = cls._read_canonical_regular_file(
                        reservation,
                        mode=0o600,
                        code=FAIL_TRANSACTION_LOCK,
                        label=f"{label} output reservation",
                    )
                    expected = transaction._output_reservation_fields(output)
                    if fields != expected or raw != _canonical_json(expected):
                        _fail(FAIL_TRANSACTION_LOCK, f"{label} output reservation differs")
                elif not output.exists():
                    _fail(FAIL_TRANSACTION_LOCK, f"absent {label} lacks its reservation")
            prefix = ("evidence", "promotion", "receipt")[: len(present_labels)]
            if tuple(present_labels) != prefix:
                _fail(FAIL_PUBLICATION, "published output subset is not the write-order prefix")
            if len(present_labels) < 3 and any(
                not reservation.exists()
                for reservation in transaction._output_reservation_paths.values()
            ):
                _fail(FAIL_TRANSACTION_LOCK, "partial publication lost an output reservation")

            if marker.state == "PENDING":
                phase = "STAGED_PENDING"
            elif present_labels:
                if state not in {"ACTORS_ABSENT", "PUBLISHED"}:
                    _fail(FAIL_TRANSACTION_LOCK, "publication precedes ACTORS_ABSENT journal")
                if len(present_labels) < 3:
                    phase = "PARTIAL_PUBLICATION"
                elif state == "PUBLISHED":
                    if any(
                        reservation.exists()
                        for reservation in transaction._output_reservation_paths.values()
                    ):
                        _fail(FAIL_TRANSACTION_LOCK, "PUBLISHED retains an output reservation")
                    phase = "PUBLISHED"
                else:
                    phase = "ALL_PUBLIC_OUTPUTS_UNJOURNALED"
            elif state == "PUBLISHED":
                _fail(FAIL_PUBLICATION, "PUBLISHED journal lacks final outputs")
            elif state == "ACTORS_ABSENT":
                phase = "ACTORS_ABSENT"
            else:
                phase = "MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN"
            transaction._recovery_marker_snapshot = marker
            transaction._recovery_phase = phase
            if promote_complete_marker:
                os.replace(marker_next_path, marker_path)
                _fsync_directory_v1(custody)
            if promote_next_journal:
                os.replace(next_path, transaction._journal_path)
                _fsync_directory_v1(stage)
            elif promote_reserved_stage:
                transaction._write_journal()
            return transaction
        except BaseException:
            transaction.close_lock()
            raise

    def stage_and_prospectively_replay(
        self,
        replay_payload: Mapping[str, object],
        prospective_promotion: Mapping[str, object],
        *,
        replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
    ) -> None:
        if self._state != "RESERVED" or self._stage_directory is None:
            _fail(FAIL_PUBLICATION, "transaction is not ready for staging")
        # Presence and canonical framing are required here; semantic equality
        # to the recovered role keys is checked before this method is entered
        # by both first-run and recovery coordinators.
        self.load_actor_trust_checkpoint_v1()
        evidence_bytes = _canonical_json(dict(replay_payload))
        promotion_bytes = _canonical_json(dict(prospective_promotion))
        try:
            loaded_payload = json.loads(evidence_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(FAIL_PUBLICATION, f"staged evidence JSON is invalid: {exc}")
        loaded_inputs = load_gate_evidence_inputs_v1(loaded_payload)
        staged_seed_commitment = loaded_inputs.split_seed_commitment_fields.get(
            "split_seed_commitment_digest"
        )
        if type(staged_seed_commitment) is not bytes or len(staged_seed_commitment) != 32:
            _fail(FAIL_CUSTODY, "staged split seed commitment is absent")
        replayed = replay(loaded_payload)
        if _canonical_json(replayed) != promotion_bytes:
            _fail(FAIL_PUBLICATION, "prospective replay differs from staged promotion")

        receipt = {
            "schema": PUBLICATION_RECEIPT_SCHEMA,
            "basis_commit": self.basis_commit,
            "run_id_hex": self.run_id.hex(),
            "ledger_id_hex": self.ledger_id.hex(),
            "public_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "public_promotion_sha256": hashlib.sha256(promotion_bytes).hexdigest(),
            "seed_custody_verification_receipt_sha256_or_null": None,
            "prospective_public_replay_passed": True,
            "marker_was_complete_during_staging": False,
            "actor_cleanup_required_before_publication": True,
            "authority_disclosure": dict(TECHNICAL_ACTOR_DISCLOSURE_V1),
            "contains_private_key": False,
            "contains_raw_split_seed": False,
            "contains_split_assignment_rows": False,
        }
        receipt_bytes = _canonical_json(receipt)
        assert_public_payload_contains_no_secret_fields(receipt)
        staged = {
            "evidence": (self._stage_directory / "public-evidence.json", evidence_bytes),
            "promotion": (self._stage_directory / "promotion.json", promotion_bytes),
            "receipt": (self._stage_directory / "publication-receipt.json", receipt_bytes),
        }
        for name, (path, payload) in staged.items():
            _write_atomic_resumable_v1(
                path,
                payload,
                0o600,
                fault=self._fault,
                fault_label=name,
            )
        _fsync_directory_v1(self._stage_directory)
        # Reload from disk, not the caller's objects, before permitting marker
        # completion.  This is the key transaction-ordering guard.
        disk_payload = json.loads(staged["evidence"][0].read_bytes())
        disk_promotion = replay(disk_payload)
        if _canonical_json(disk_promotion) != staged["promotion"][0].read_bytes():
            _fail(FAIL_PUBLICATION, "durable staged replay differs")
        self._staged_payloads = {name: payload for name, (_path, payload) in staged.items()}
        self._staged_seed_commitment = staged_seed_commitment
        self._state = "STAGED_PROSPECTIVE_REPLAY_PASSED"
        self._write_journal()
        self._fault("after_durable_staging")

    def record_seed_custody_verification_v1(
        self,
        receipt: Mapping[str, object],
    ) -> None:
        """Durably bind the first verifier receipt and live-check every recovery."""

        if self._state not in {
            "STAGED_PROSPECTIVE_REPLAY_PASSED",
            "SEED_CUSTODY_VERIFIED",
            "MARKER_COMPLETE",
            "ACTORS_ABSENT",
            "PUBLISHED",
        }:
            _fail(FAIL_CUSTODY, "seed custody cannot be verified in this state")
        expected = self._staged_seed_commitment
        daemon_binding = (
            None
            if self._prestage_intent_fields is None
            else self._prestage_intent_fields.get(
                "live_actor_protocol_daemon_receipt_binding"
            )
        )
        if (
            type(expected) is not bytes
            or len(expected) != 32
            or type(daemon_binding) is not bytes
            or len(daemon_binding) != 32
            or self._stage_directory is None
        ):
            _fail(FAIL_CUSTODY, "seed verifier transaction binding is absent")
        live_payload = _validate_seed_custody_verification_receipt_v1(
            receipt,
            expected_commitment=expected,
            expected_daemon_binding=daemon_binding,
        )
        verification_path = (
            self._stage_directory / _SEED_CUSTODY_VERIFICATION_FILENAME
        )
        if verification_path.exists() or verification_path.is_symlink():
            durable_payload, durable_receipt = self._read_canonical_regular_file(
                verification_path,
                mode=0o600,
                code=FAIL_CUSTODY,
                label="durable seed-custody verification",
            )
            _validate_seed_custody_verification_receipt_v1(
                durable_receipt,
                expected_commitment=expected,
                expected_daemon_binding=daemon_binding,
            )
            if _seed_custody_receipt_stable_identity_v1(
                durable_receipt
            ) != _seed_custody_receipt_stable_identity_v1(receipt):
                _fail(
                    FAIL_CUSTODY,
                    "live seed verifier differs from the durable stable execution binding",
                )
        else:
            _write_atomic_resumable_v1(
                verification_path,
                live_payload,
                0o600,
                fault=self._fault,
                fault_label="seed_custody_verification",
            )
            durable_payload = live_payload
        self._fault("after_seed_custody_verification_receipt_durable")

        receipt_path = self._stage_directory / _STAGED_FILENAMES["receipt"]
        _current_payload, current_fields = self._read_canonical_regular_file(
            receipt_path,
            mode=0o600,
            code=FAIL_PUBLICATION,
            label="staged publication receipt",
        )
        binding_field = "seed_custody_verification_receipt_sha256_or_null"
        durable_digest = hashlib.sha256(durable_payload).hexdigest()
        observed_binding = current_fields.get(binding_field)
        if observed_binding not in {None, durable_digest}:
            _fail(FAIL_PUBLICATION, "publication receipt seed-verifier binding differs")
        old_fields = dict(current_fields)
        old_fields[binding_field] = None
        new_fields = dict(current_fields)
        new_fields[binding_field] = durable_digest
        new_receipt_payload = _canonical_json(new_fields)
        _replace_atomic_exact_payload_v1(
            receipt_path,
            expected_old_payload=_canonical_json(old_fields),
            new_payload=new_receipt_payload,
            mode=0o600,
            fault=self._fault,
            fault_label="publication_seed_verifier_binding",
        )
        self._staged_payloads["receipt"] = new_receipt_payload
        self._verified_seed_commitment = expected
        if self._state == "STAGED_PROSPECTIVE_REPLAY_PASSED":
            self._state = "SEED_CUSTODY_VERIFIED"
        if self._state == "SEED_CUSTODY_VERIFIED":
            self._write_journal()
        self._fault("after_seed_custody_verification")

    def record_marker_complete(self, actual: MarkerSnapshot, expected: MarkerSnapshot) -> None:
        if actual != expected or actual.state != "COMPLETE":
            _fail(FAIL_CUSTODY, "actual COMPLETE marker differs from staged evidence")
        if self._state in {"MARKER_COMPLETE", "ACTORS_ABSENT", "PUBLISHED"}:
            return
        if self._state != "SEED_CUSTODY_VERIFIED":
            _fail(
                FAIL_CUSTODY,
                "marker completion is forbidden before durable keyless seed verification",
            )
        if (
            self._staged_seed_commitment is None
            or self._verified_seed_commitment != self._staged_seed_commitment
        ):
            _fail(
                FAIL_CUSTODY,
                "marker completion requires keyless verification of the staged seed commitment",
            )
        self._state = "MARKER_COMPLETE"
        self._write_journal()
        self._fault("after_marker_complete")

    def record_actors_absent(self) -> None:
        if self._state in {"ACTORS_ABSENT", "PUBLISHED"}:
            return
        if self._state != "MARKER_COMPLETE":
            _fail(FAIL_CONTAINER, "actors may be finalized only after marker completion")
        self._state = "ACTORS_ABSENT"
        self._write_journal()
        self._fault("after_actor_cleanup")

    def _replace_reserved_output(self, path: Path, payload: bytes, label: str) -> None:
        reservation = self._output_reservation_paths[path]
        reservation_valid = False
        if reservation.exists() or reservation.is_symlink():
            if reservation.is_symlink():
                _fail(FAIL_TRANSACTION_LOCK, f"output reservation is a symlink: {label}")
            try:
                reservation_metadata = reservation.stat()
                reservation_payload = reservation.read_bytes()
            except OSError as exc:
                _fail(FAIL_TRANSACTION_LOCK, f"output reservation cannot be read: {label}: {exc}")
            reservation_valid = (
                stat.S_ISREG(reservation_metadata.st_mode)
                and stat.S_IMODE(reservation_metadata.st_mode) == 0o600
                and reservation_payload
                == _canonical_json(self._output_reservation_fields(path))
            )
            if not reservation_valid:
                _fail(FAIL_TRANSACTION_LOCK, f"output reservation changed: {label}")
        if path.exists() or path.is_symlink():
            if path.is_symlink():
                _fail(FAIL_PUBLICATION, f"existing public output is a symlink: {label}")
            metadata = path.stat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o644
                or path.read_bytes() != payload
            ):
                _fail(FAIL_PUBLICATION, f"existing public output differs: {label}")
            return
        if not reservation_valid:
            _fail(FAIL_TRANSACTION_LOCK, f"output reservation disappeared: {label}")
        _write_exclusive_durable_v1(path, payload, 0o644)

    def publish(
        self,
        *,
        replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
    ) -> None:
        if self._state == "PUBLISHED":
            return
        if self._state != "ACTORS_ABSENT":
            _fail(FAIL_PUBLICATION, "publication requires verified actor absence")
        if set(self._staged_payloads) != {"evidence", "promotion", "receipt"}:
            _fail(FAIL_PUBLICATION, "durable staged payload set is incomplete")
        if self._stage_directory is None:
            raise AssertionError("stage directory is absent")
        try:
            disk_evidence_bytes = (self._stage_directory / "public-evidence.json").read_bytes()
            disk_promotion_bytes = (self._stage_directory / "promotion.json").read_bytes()
            disk_receipt_bytes = (
                self._stage_directory / "publication-receipt.json"
            ).read_bytes()
            disk_receipt = json.loads(disk_receipt_bytes)
            disk_evidence = json.loads(disk_evidence_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(FAIL_PUBLICATION, f"durable staging cannot be reloaded: {exc}")
        if _canonical_json(replay(disk_evidence)) != disk_promotion_bytes:
            _fail(FAIL_PUBLICATION, "post-cleanup public replay differs from staged promotion")
        verification_payload, verification_fields = self._read_canonical_regular_file(
            self._stage_directory / _SEED_CUSTODY_VERIFICATION_FILENAME,
            mode=0o600,
            code=FAIL_CUSTODY,
            label="durable seed-custody verification",
        )
        daemon_binding = (
            None
            if self._prestage_intent_fields is None
            else self._prestage_intent_fields.get(
                "live_actor_protocol_daemon_receipt_binding"
            )
        )
        if (
            type(self._staged_seed_commitment) is not bytes
            or type(daemon_binding) is not bytes
            or len(daemon_binding) != 32
        ):
            _fail(FAIL_CUSTODY, "publication seed-verifier identity is absent")
        _validate_seed_custody_verification_receipt_v1(
            verification_fields,
            expected_commitment=self._staged_seed_commitment,
            expected_daemon_binding=daemon_binding,
        )
        if (
            type(disk_receipt) is not dict
            or disk_receipt.get("schema") != PUBLICATION_RECEIPT_SCHEMA
            or disk_receipt.get("public_evidence_sha256")
            != hashlib.sha256(disk_evidence_bytes).hexdigest()
            or disk_receipt.get("public_promotion_sha256")
            != hashlib.sha256(disk_promotion_bytes).hexdigest()
            or disk_receipt.get("authority_disclosure")
            != dict(TECHNICAL_ACTOR_DISCLOSURE_V1)
            or disk_receipt.get(
                "seed_custody_verification_receipt_sha256_or_null"
            )
            != hashlib.sha256(verification_payload).hexdigest()
        ):
            _fail(FAIL_PUBLICATION, "staged publication receipt does not bind payload bytes")
        self._staged_payloads = {
            "evidence": disk_evidence_bytes,
            "promotion": disk_promotion_bytes,
            "receipt": disk_receipt_bytes,
        }
        self._fault("before_publication")
        self._replace_reserved_output(
            self.public_evidence_path, self._staged_payloads["evidence"], "evidence"
        )
        self._fault("after_evidence_publication")
        self._replace_reserved_output(
            self.public_promotion_path, self._staged_payloads["promotion"], "promotion"
        )
        self._fault("after_promotion_publication")
        self._replace_reserved_output(
            self.publication_receipt_path, self._staged_payloads["receipt"], "receipt"
        )
        self._fault("after_receipt_publication")
        for index, reservation in enumerate(self._output_reservation_paths.values(), start=1):
            if reservation.exists() or reservation.is_symlink():
                try:
                    reservation.unlink()
                except OSError as exc:
                    _fail(FAIL_PUBLICATION, f"cannot remove output reservation: {exc}")
            self._fault(f"after_output_reservation_{index}_cleanup")
        self._fault("after_output_reservation_cleanup")
        _fsync_directory_v1(self.public_evidence_path.parent)
        self._state = "PUBLISHED"
        self._write_journal()

    def close_lock(self) -> None:
        if self._lock_descriptor is not None:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
        if self._directory_descriptor is not None:
            fcntl.flock(self._directory_descriptor, fcntl.LOCK_UN)
            os.close(self._directory_descriptor)
            self._directory_descriptor = None
        if self._anchor_descriptor is not None:
            fcntl.flock(self._anchor_descriptor, fcntl.LOCK_UN)
            os.close(self._anchor_descriptor)
            self._anchor_descriptor = None

    def __enter__(self) -> "FormalCeremonyTransactionV1":
        if self._lock_descriptor is None:
            _fail(FAIL_TRANSACTION_LOCK, "transaction context requires an acquired lock")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close_lock()
        return False


@dataclass(slots=True)
class PendingCeremonyRecoveryV1(AbstractContextManager["PendingCeremonyRecoveryV1"]):
    """Acquired, read-only identity for an explicit PENDING recovery action."""

    basis_commit: str
    run_id: bytes
    ledger_id: bytes
    marker_snapshot: MarkerSnapshot
    journal_state: str
    stage_directory: Path
    custody_directory: Path
    public_evidence_path: Path
    public_promotion_path: Path
    prestage_intent_fields: Mapping[str, object]
    prestage_intent_sha256: str
    actor_trust_checkpoint_fields: Mapping[str, object]
    lock_descriptor: int
    anchor_descriptor: int = -1
    directory_descriptor: int = -1

    def close(self) -> None:
        if self.lock_descriptor >= 0:
            fcntl.flock(self.lock_descriptor, fcntl.LOCK_UN)
            os.close(self.lock_descriptor)
            self.lock_descriptor = -1
        if self.directory_descriptor >= 0:
            fcntl.flock(self.directory_descriptor, fcntl.LOCK_UN)
            os.close(self.directory_descriptor)
            self.directory_descriptor = -1
        if self.anchor_descriptor >= 0:
            fcntl.flock(self.anchor_descriptor, fcntl.LOCK_UN)
            os.close(self.anchor_descriptor)
            self.anchor_descriptor = -1

    def __enter__(self) -> "PendingCeremonyRecoveryV1":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _acquire_host_recovery_anchor_v1(
    *,
    custody_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
    actors: "CeremonyActorsV1 | None",
) -> tuple[FormalCeremonyTransactionV1, dict[str, object]]:
    """Acquire the host-side liveness lock before opening a handed-off tree."""

    _reject_caller_symlink_chain_v1(custody_directory, "recovery custody path")
    _reject_caller_symlink_chain_v1(public_evidence_path, "recovery evidence path")
    _reject_caller_symlink_chain_v1(public_promotion_path, "recovery promotion path")
    custody = Path(os.path.abspath(os.fspath(custody_directory)))
    evidence = Path(os.path.abspath(os.fspath(public_evidence_path)))
    promotion = Path(os.path.abspath(os.fspath(public_promotion_path)))
    reservation_path = evidence.with_name(f".{evidence.name}.hegel-reserved")
    reservation_payload, reservation = (
        FormalCeremonyTransactionV1._read_canonical_regular_file(
            reservation_path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="host recovery evidence reservation",
        )
    )
    if (
        set(reservation)
        != {"schema", "basis_commit", "run_id_hex", "output_kind", "output_path", "state"}
        or reservation.get("schema") != OUTPUT_RESERVATION_SCHEMA
        or reservation.get("output_kind") != "evidence"
        or reservation.get("output_path") != str(evidence)
        or reservation.get("state") != "RESERVED_NOT_PUBLIC"
        or type(reservation.get("run_id_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{32}", str(reservation.get("run_id_hex"))) is None
    ):
        _fail(FAIL_TRANSACTION_LOCK, "host recovery evidence reservation differs")
    basis_commit = _commit(reservation.get("basis_commit"))  # type: ignore[arg-type]
    run_id = bytes.fromhex(str(reservation["run_id_hex"]))
    stage = evidence.parent / (".hegel-m25-stage-" + run_id.hex())
    anchor_path = stage / _RECOVERY_ANCHOR_FILENAME
    ready_path = stage / _RECOVERY_ANCHOR_READY_FILENAME
    descriptor = -1
    identity: FormalCeremonyTransactionV1 | None = None
    try:
        descriptor = os.open(
            anchor_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail(FAIL_TRANSACTION_LOCK, "host recovery anchor is live-locked")
        metadata = os.fstat(descriptor)
        path_metadata = anchor_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (metadata.st_dev, metadata.st_ino)
            != (path_metadata.st_dev, path_metadata.st_ino)
            or metadata.st_size <= 1
            or metadata.st_size > 64 * 1024
        ):
            _fail(FAIL_TRANSACTION_LOCK, "host recovery anchor inode differs")
        anchor_payload = os.read(descriptor, metadata.st_size + 1)
        anchor = json.loads(anchor_payload)
        if type(anchor) is not dict or _canonical_json(anchor) != anchor_payload:
            _fail(FAIL_TRANSACTION_LOCK, "host recovery anchor is not canonical")
        ledger_hex = anchor.get("ledger_id_hex")
        if type(ledger_hex) is not str or re.fullmatch(r"[0-9a-f]{32}", ledger_hex) is None:
            _fail(FAIL_TRANSACTION_LOCK, "host recovery anchor ledger ID differs")
        ledger_id = bytes.fromhex(ledger_hex)
        identity = FormalCeremonyTransactionV1(
            basis_commit=basis_commit,
            custody_directory=custody,
            public_evidence_path=evidence,
            public_promotion_path=promotion,
            run_id=run_id,
            ledger_id=ledger_id,
        )
        identity._anchor_descriptor = descriptor
        descriptor = -1
        identity._stage_directory = stage
        identity._validate_recovery_anchor_fields_v1(
            anchor, require_internal_lock_bytes=False
        )
        ready_payload, ready = FormalCeremonyTransactionV1._read_canonical_regular_file(
            ready_path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="host recovery anchor ready record",
        )
        expected_ready = {
            "schema": _RECOVERY_ANCHOR_READY_SCHEMA,
            "anchor_filename": _RECOVERY_ANCHOR_FILENAME,
            "anchor_sha256": hashlib.sha256(anchor_payload).hexdigest(),
            "state": "ANCHOR_FLOCK_ACQUIRED_BEFORE_READY",
            "raw_seed_bytes_read": False,
        }
        if ready != expected_ready or ready_payload != _canonical_json(expected_ready):
            _fail(FAIL_TRANSACTION_LOCK, "host recovery anchor ready record differs")
        custody_metadata = custody.lstat()
        before_owner = (
            anchor.get("custody_owner_before_uid"),
            anchor.get("custody_owner_before_gid"),
        )
        handoff_owner = (
            anchor.get("custody_owner_handoff_uid"),
            anchor.get("custody_owner_handoff_gid"),
        )
        current_owner = (custody_metadata.st_uid, custody_metadata.st_gid)
        if current_owner == handoff_owner:
            if type(actors) is not DockerCeremonyActorsV1 or not actors.authoritative:
                _fail(
                    FAIL_CUSTODY,
                    "65534-owned recovery requires the sealed Docker reclaim actor",
                )
            actors.reclaim_pending_custody_from_anchor_v1(anchor)
        elif current_owner != before_owner:
            _fail(FAIL_CUSTODY, "host recovery custody owner differs from anchor")
        identity._acquire_custody_directory_lock_v1()
        # Now that traversal is restored, bind the exact internal lock and
        # prestage bytes after the caller loads them below.
        return identity, anchor
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if identity is not None:
            identity.close_lock()
        elif descriptor >= 0:
            os.close(descriptor)
        _fail(FAIL_TRANSACTION_LOCK, f"host recovery anchor cannot be acquired: {exc}")
    except BaseException:
        if identity is not None:
            identity.close_lock()
        elif descriptor >= 0:
            os.close(descriptor)
        raise


def acquire_pending_ceremony_recovery_v1(
    *,
    custody_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
    actors: "CeremonyActorsV1 | None" = None,
) -> PendingCeremonyRecoveryV1:
    anchor_identity, anchor_fields = _acquire_host_recovery_anchor_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
        actors=actors,
    )
    try:
        return _acquire_pending_ceremony_recovery_core_v1(
            anchor_identity=anchor_identity,
            anchor_fields=anchor_fields,
            public_evidence_path=public_evidence_path,
            public_promotion_path=public_promotion_path,
        )
    except BaseException:
        anchor_identity.close_lock()
        raise


def _acquire_pending_ceremony_recovery_core_v1(
    *,
    anchor_identity: FormalCeremonyTransactionV1,
    anchor_fields: Mapping[str, object],
    public_evidence_path: Path,
    public_promotion_path: Path,
) -> PendingCeremonyRecoveryV1:
    """Acquire an existing transaction without creating or redrawing state.

    This API is deliberately separate from ordinary ``execute``.  It validates
    the persistent lock, opaque-ID reservations, output reservations, journal,
    and exact PENDING marker, then holds an advisory lock for the caller.  It
    may invoke only the sealed metadata-only Docker reclaim path when the
    anchor proves the tree is still owned by UID/GID 65534, and never calls a
    random source.
    """

    custody = anchor_identity.custody_directory
    if not custody.is_dir() or (custody.stat().st_mode & 0o777) != 0o700:
        anchor_identity.close_lock()
        _fail(FAIL_CUSTODY, "pending recovery custody directory is invalid")
    lock_path = custody / "phase3_m25_ceremony.lock"
    if lock_path.is_symlink():
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery lock may not be a symlink")
    try:
        metadata = lock_path.stat()
        raw_lock = lock_path.read_bytes()
    except OSError as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"pending recovery lock cannot be read: {exc}")
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery lock is not regular mode 0600")
    try:
        lock_fields = json.loads(raw_lock)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"pending recovery lock JSON is invalid: {exc}")
    expected_lock_fields = {
        "schema", "reservation_bootstrap_state", "basis_commit", "run_id_hex", "ledger_id_hex",
        "custody_directory", "public_output_parent",
        "public_evidence_path", "public_promotion_path",
        "publication_receipt_path", "stage_directory_name",
        "prestage_intent_sha256_or_null",
        "prestage_intent_transport_or_null",
        "ordered_reservation_artifact_specs_or_null",
        "recovery_required_if_incomplete",
        "bootstrap_completion_requires_exact_full_prefix",
    }
    if (
        type(lock_fields) is not dict
        or set(lock_fields) != expected_lock_fields
        or lock_fields.get("schema") != TRANSACTION_LOCK_SCHEMA
        or lock_fields.get("reservation_bootstrap_state")
        != "RESERVING_EXACT_PREFIX"
        or lock_fields.get("recovery_required_if_incomplete") is not True
        or lock_fields.get("bootstrap_completion_requires_exact_full_prefix")
        is not True
        or _canonical_json(lock_fields) != raw_lock
    ):
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery lock fields differ")
    basis_commit = _commit(lock_fields["basis_commit"])
    try:
        run_id = bytes.fromhex(lock_fields["run_id_hex"])
        ledger_id = bytes.fromhex(lock_fields["ledger_id_hex"])
    except (TypeError, ValueError) as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"pending recovery opaque ID is invalid: {exc}")
    if len(run_id) != 16 or len(ledger_id) != 16 or run_id == ledger_id:
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery opaque IDs differ from schema")
    if (
        basis_commit != anchor_identity.basis_commit
        or run_id != anchor_identity.run_id
        or ledger_id != anchor_identity.ledger_id
    ):
        anchor_identity.close_lock()
        _fail(FAIL_TRANSACTION_LOCK, "internal lock differs from host recovery anchor")
    evidence = public_evidence_path.resolve()
    promotion = public_promotion_path.resolve()
    receipt = promotion.with_name(promotion.name + ".publication-receipt.json")
    expected_stage_name = ".hegel-m25-stage-" + run_id.hex()
    if (
        lock_fields.get("public_evidence_path") != str(evidence)
        or lock_fields.get("public_promotion_path") != str(promotion)
        or lock_fields.get("publication_receipt_path") != str(receipt)
        or lock_fields.get("stage_directory_name") != expected_stage_name
    ):
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery paths differ from persistent lock")
    stage = evidence.parent / expected_stage_name
    if (
        stage.is_symlink()
        or not stage.is_dir()
        or stat.S_IMODE(stage.stat().st_mode) != 0o700
    ):
        _fail(FAIL_TRANSACTION_LOCK, "PENDING recovery stage directory differs")
    prestage_intent = FormalCeremonyTransactionV1._load_bound_prestage_intent(
        stage_directory=stage,
        expected_sha256_or_null=lock_fields.get(
            "prestage_intent_sha256_or_null"
        ),
        basis_commit=basis_commit,
        run_id=run_id,
        ledger_id=ledger_id,
        required=True,
    )
    assert prestage_intent is not None
    FormalCeremonyTransactionV1._load_bound_live_qualification_bundle_v1(
        stage_directory=stage,
        prestage_intent=prestage_intent,
    )
    identity = anchor_identity
    identity._prestage_intent_fields = MappingProxyType(dict(prestage_intent))
    identity._prestage_intent_bytes = _canonical_json(prestage_intent)
    if lock_fields != identity._lock_fields():
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery paths differ from persistent lock")
    identity._validate_recovery_anchor_fields_v1(
        anchor_fields, require_internal_lock_bytes=True
    )

    descriptor = os.open(lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        _fail(FAIL_TRANSACTION_LOCK, f"another ceremony/recovery holds the lock: {exc}")
    try:
        marker = read_marker_snapshot_v1(custody / "split_seed_instantiation.marker")
        if marker.state != "PENDING":
            _fail(FAIL_CUSTODY, "explicit recovery requires a PENDING marker")
        expected_custody_names = {
            "phase3_m25_ceremony.lock",
            f"opaque-run-{run_id.hex()}.reserved",
            f"opaque-ledger-{ledger_id.hex()}.reserved",
            "split_seed_instantiation.marker",
        }
        optional_seed_names = {
            "split_seed_generation.intent",
            "split_seed_generation.complete",
            "split_master_seed.bin",
        }
        actual_custody_names = {path.name for path in custody.iterdir()}
        if (
            not expected_custody_names <= actual_custody_names
            or not actual_custody_names <= expected_custody_names | optional_seed_names
        ):
            _fail(FAIL_CUSTODY, "pending recovery custody path set differs")
        seed_prefix = (
            "split_seed_generation.intent",
            "split_master_seed.bin",
            "split_seed_generation.complete",
        )
        present_seed_prefix = tuple(
            name for name in seed_prefix if name in actual_custody_names
        )
        if present_seed_prefix != seed_prefix[: len(present_seed_prefix)]:
            _fail(FAIL_CUSTODY, "pending seed artifacts are not an exact prefix")
        if present_seed_prefix:
            expected_intent = {
                "schema": "hegel-phase3-m25-seed-generation-intent/1",
                "state": "CSPRNG_CALL_COMMITTED_NO_REDRAW",
            }
            intent_payload, intent_fields = (
                FormalCeremonyTransactionV1._read_canonical_regular_file(
                    custody / seed_prefix[0],
                    mode=0o600,
                    code=FAIL_CUSTODY,
                    label="pending seed-generation intent",
                )
            )
            if intent_fields != expected_intent:
                _fail(FAIL_CUSTODY, "pending seed-generation intent differs")
        else:
            intent_payload = b""
        if len(present_seed_prefix) >= 2:
            seed_path = custody / seed_prefix[1]
            if seed_path.is_symlink():
                _fail(FAIL_CUSTODY, "pending raw seed path may not be a symlink")
            try:
                seed_metadata = seed_path.stat()
            except OSError as exc:
                _fail(FAIL_CUSTODY, f"pending raw seed metadata cannot be read: {exc}")
            if (
                not stat.S_ISREG(seed_metadata.st_mode)
                or stat.S_IMODE(seed_metadata.st_mode) != 0o600
                or seed_metadata.st_size != 32
            ):
                _fail(FAIL_CUSTODY, "pending raw seed inode differs; bytes were not read")
        if len(present_seed_prefix) == 3:
            _completion_payload, completion_fields = (
                FormalCeremonyTransactionV1._read_canonical_regular_file(
                    custody / seed_prefix[2],
                    mode=0o600,
                    code=FAIL_CUSTODY,
                    label="pending seed-generation completion",
                )
            )
            commitment_hex = completion_fields.get("seed_commitment_hex")
            if (
                set(completion_fields)
                != {
                    "attempt",
                    "intent_sha256",
                    "schema",
                    "seed_commitment_hex",
                    "seed_length_bytes",
                }
                or completion_fields.get("attempt") != 1
                or completion_fields.get("intent_sha256")
                != hashlib.sha256(intent_payload).hexdigest()
                or completion_fields.get("schema")
                != "hegel-phase3-m25-seed-generation-complete/1"
                or type(commitment_hex) is not str
                or re.fullmatch(r"[0-9a-f]{64}", commitment_hex) is None
                or completion_fields.get("seed_length_bytes") != 32
            ):
                _fail(FAIL_CUSTODY, "pending seed-generation completion differs")
        for kind, value in (("run", run_id), ("ledger", ledger_id)):
            reservation = custody / f"opaque-{kind}-{value.hex()}.reserved"
            try:
                reservation_metadata = reservation.stat()
                reservation_bytes = reservation.read_bytes()
                reservation_fields = json.loads(reservation_bytes)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                _fail(FAIL_TRANSACTION_LOCK, f"opaque {kind} reservation is invalid: {exc}")
            expected_reservation = {
                "schema": "hegel-phase3-m25-opaque-id-reservation/1",
                "kind": kind,
                "opaque_id_hex": value.hex(),
                "basis_commit": basis_commit,
            }
            if (
                reservation.is_symlink()
                or not stat.S_ISREG(reservation_metadata.st_mode)
                or stat.S_IMODE(reservation_metadata.st_mode) != 0o600
                or reservation_fields != expected_reservation
                or reservation_bytes != _canonical_json(expected_reservation)
            ):
                _fail(FAIL_TRANSACTION_LOCK, f"opaque {kind} reservation fields differ")
        for output in (evidence, promotion, receipt):
            if output.exists() or output.is_symlink():
                _fail(FAIL_PUBLICATION, "PENDING recovery found a final public output")
            reservation = output.with_name(f".{output.name}.hegel-reserved")
            try:
                reservation_metadata = reservation.stat()
                reservation_bytes = reservation.read_bytes()
            except OSError as exc:
                _fail(
                    FAIL_TRANSACTION_LOCK,
                    f"PENDING recovery output reservation cannot be read: {exc}",
                )
            if (
                reservation.is_symlink()
                or not stat.S_ISREG(reservation_metadata.st_mode)
                or stat.S_IMODE(reservation_metadata.st_mode) != 0o600
                or reservation_bytes
                != _canonical_json(identity._output_reservation_fields(output))
            ):
                _fail(FAIL_TRANSACTION_LOCK, "PENDING recovery output reservation is absent")
        checkpoint_payload, checkpoint_fields = (
            FormalCeremonyTransactionV1._read_canonical_regular_file(
                stage / _ACTOR_TRUST_CHECKPOINT_FILENAME,
                mode=0o600,
                code=FAIL_CUSTODY,
                label="actor-trust checkpoint",
            )
        )
        del checkpoint_payload
        restored_checkpoint = _restore(dict(checkpoint_fields))
        try:
            purpose_rows = restored_checkpoint["purpose_key_rows"]
            checkpoint_public_keys = {
                int(row["purpose_id"]): row["public_key_32_bytes"]
                for row in purpose_rows
            }
            checkpoint_trust = build_actor_trust_v1(
                public_keys=checkpoint_public_keys,
                timestamp=int(prestage_intent["created_at_unix_seconds"]),
                basis_commit=basis_commit,
                trust_genesis_id=bytes.fromhex(
                    str(prestage_intent["trust_genesis_id_hex"])
                ),
            )
            validate_actor_trust_checkpoint_fields_v1(
                checkpoint_fields,
                expected_actor_trust=checkpoint_trust,
                basis_commit=basis_commit,
                run_id=run_id,
                ledger_id=ledger_id,
                prestage_intent_sha256=hashlib.sha256(
                    _canonical_json(prestage_intent)
                ).hexdigest(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            _fail(FAIL_CUSTODY, f"actor-trust checkpoint cannot be reconstructed: {exc}")
        expected_pending = MarkerSnapshot(
            "PENDING",
            SPLIT_VERSION_DIGEST,
            None,
            checkpoint_trust.key_ids[1],
            int(prestage_intent["created_at_unix_seconds"]),
        )
        if marker != expected_pending:
            _fail(FAIL_CUSTODY, "PENDING marker differs from actor-trust checkpoint")
        journal_path = stage / "transaction-journal.json"
        try:
            journal_metadata = journal_path.stat()
            journal_bytes = journal_path.read_bytes()
            journal = json.loads(journal_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"PENDING recovery journal is invalid: {exc}")
        expected_journal = {
            "schema": TRANSACTION_JOURNAL_SCHEMA,
            "basis_commit": basis_commit,
            "run_id_hex": run_id.hex(),
            "ledger_id_hex": ledger_id.hex(),
            "state": journal.get("state") if type(journal) is dict else None,
            "marker_complete": False,
            "actors_absent": False,
            "public_outputs_complete": False,
            "pending_recovery_protocol_frozen": True,
        }
        if (
            type(journal) is not dict
            or journal_path.is_symlink()
            or not stat.S_ISREG(journal_metadata.st_mode)
            or stat.S_IMODE(journal_metadata.st_mode) != 0o600
            or journal_bytes != _canonical_json(journal)
            or journal.get("state")
            not in {
                "RESERVED",
                "STAGED_PROSPECTIVE_REPLAY_PASSED",
                "SEED_CUSTODY_VERIFIED",
            }
            or journal != expected_journal
        ):
            _fail(FAIL_TRANSACTION_LOCK, "PENDING recovery journal state differs")
        allowed_stage_entries = {
            "transaction-journal.json",
            _PRESTAGE_INTENT_FILENAME,
            _LIVE_QUALIFICATION_BUNDLE_FILENAME,
            _ACTOR_TRUST_CHECKPOINT_FILENAME,
            *_STAGED_FILENAMES.values(),
            *(filename + ".next" for filename in _STAGED_FILENAMES.values()),
            _SEED_CUSTODY_VERIFICATION_FILENAME,
            _SEED_CUSTODY_VERIFICATION_FILENAME + ".next",
            _RECOVERY_ANCHOR_FILENAME,
            _RECOVERY_ANCHOR_READY_FILENAME,
            "transaction-journal.next",
        }
        actual_stage_entries = {path.name for path in stage.iterdir()}
        required_stage_entries = {
            "transaction-journal.json",
            _PRESTAGE_INTENT_FILENAME,
            _LIVE_QUALIFICATION_BUNDLE_FILENAME,
            _ACTOR_TRUST_CHECKPOINT_FILENAME,
        }
        if (
            not required_stage_entries <= actual_stage_entries
            or not actual_stage_entries <= allowed_stage_entries
        ):
            _fail(FAIL_TRANSACTION_LOCK, "PENDING recovery stage path set differs")
        recovery = PendingCeremonyRecoveryV1(
            basis_commit=basis_commit,
            run_id=run_id,
            ledger_id=ledger_id,
            marker_snapshot=marker,
            journal_state=journal["state"],
            stage_directory=stage,
            custody_directory=custody,
            public_evidence_path=evidence,
            public_promotion_path=promotion,
            prestage_intent_fields=MappingProxyType(dict(prestage_intent)),
            prestage_intent_sha256=hashlib.sha256(
                _canonical_json(prestage_intent)
            ).hexdigest(),
            actor_trust_checkpoint_fields=MappingProxyType(dict(checkpoint_fields)),
            lock_descriptor=descriptor,
            anchor_descriptor=(
                -1
                if identity._anchor_descriptor is None
                else identity._anchor_descriptor
            ),
            directory_descriptor=(
                -1
                if identity._directory_descriptor is None
                else identity._directory_descriptor
            ),
        )
        identity._anchor_descriptor = None
        identity._directory_descriptor = None
        return recovery
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def _validate_preseed_abort_actor_absence_receipt_v1(
    receipt: Mapping[str, object], *, basis_commit: str, run_id: bytes,
    expected_daemon_binding: bytes,
) -> bytes:
    required = {
        "schema", "basis_commit", "run_id_hex", "exact_run_label_checked",
        "actor_containers_absent", "actor_key_volumes_absent",
        "seed_continuity_state_absent", "docker_daemon_receipt_sha256",
        "receipt_sha256",
    }
    body = dict(receipt)
    claimed = body.pop("receipt_sha256", None)
    if (
        set(receipt) != required
        or receipt.get("schema") != _PRESEED_ABORT_ABSENCE_SCHEMA
        or receipt.get("basis_commit") != basis_commit
        or receipt.get("run_id_hex") != run_id.hex()
        or any(
            receipt.get(field) is not True
            for field in (
                "exact_run_label_checked", "actor_containers_absent",
                "actor_key_volumes_absent", "seed_continuity_state_absent",
            )
        )
        or type(receipt.get("docker_daemon_receipt_sha256")) is not str
        or type(expected_daemon_binding) is not bytes
        or len(expected_daemon_binding) != 32
        or receipt.get("docker_daemon_receipt_sha256")
        != expected_daemon_binding.hex()
        or re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("docker_daemon_receipt_sha256"))
        ) is None
        or claimed != hashlib.sha256(_canonical_json(body)).hexdigest()
    ):
        _fail(FAIL_CONTAINER, "preseed actor-absence receipt differs")
    assert_public_payload_contains_no_secret_fields(receipt)
    return _canonical_json(dict(receipt))


def _preseed_abort_retirement_marker_path_v1(output_path: Path) -> Path:
    return output_path.with_name(
        f".{output_path.name}.hegel-preseed-abort-retired.json"
    )


def _validate_preseed_abort_terminal_tombstone_v1(
    payload: bytes,
    fields: Mapping[str, object],
    *,
    custody: Path,
    evidence: Path,
    promotion: Path,
    receipt_output: Path,
    expected_basis_commit: str | None = None,
    expected_run_id: bytes | None = None,
    expected_ledger_id: bytes | None = None,
    expected_daemon_binding: bytes | None = None,
    expected_absence_sha256: str | None = None,
) -> tuple[str, bytes, bytes, bytes, Path]:
    required = {
        "schema", "basis_commit", "run_id_hex", "ledger_id_hex",
        "custody_absolute_path", "custody_st_dev", "custody_st_ino",
        "custody_parent_absolute_path", "custody_parent_st_dev",
        "custody_parent_st_ino", "public_parent_absolute_path",
        "public_parent_st_dev", "public_parent_st_ino", "stage_absolute_path",
        "public_evidence_path", "public_promotion_path",
        "publication_receipt_path", "abort_plan_sha256",
        "actor_absence_receipt_sha256", "docker_daemon_receipt_sha256",
        "retirement_marker_rows",
        "all_plan_targets_must_be_absent_before_success",
        "formal_outputs_published", "raw_seed_bytes_read",
    }
    basis = fields.get("basis_commit")
    run_hex = fields.get("run_id_hex")
    ledger_hex = fields.get("ledger_id_hex")
    daemon_hex = fields.get("docker_daemon_receipt_sha256")
    retirement_rows = fields.get("retirement_marker_rows")
    if (
        set(fields) != required
        or fields.get("schema") != _PRESEED_ABORT_TOMBSTONE_SCHEMA
        or type(basis) is not str
        or re.fullmatch(r"[0-9a-f]{40}", basis) is None
        or type(run_hex) is not str
        or re.fullmatch(r"[0-9a-f]{32}", run_hex) is None
        or type(ledger_hex) is not str
        or re.fullmatch(r"[0-9a-f]{32}", ledger_hex) is None
        or run_hex == ledger_hex
        or type(daemon_hex) is not str
        or re.fullmatch(r"[0-9a-f]{64}", daemon_hex) is None
        or type(retirement_rows) not in {list, tuple}
        or type(fields.get("abort_plan_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", str(fields.get("abort_plan_sha256"))) is None
        or type(fields.get("actor_absence_receipt_sha256")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", str(fields.get("actor_absence_receipt_sha256"))
        ) is None
        or fields.get("all_plan_targets_must_be_absent_before_success") is not True
        or fields.get("formal_outputs_published") is not False
        or fields.get("raw_seed_bytes_read") is not False
        or payload != _canonical_json(dict(fields))
    ):
        _fail(FAIL_TRANSACTION_LOCK, "preseed abort terminal tombstone differs")
    run_id = bytes.fromhex(run_hex)
    ledger_id = bytes.fromhex(ledger_hex)
    daemon_binding = bytes.fromhex(daemon_hex)
    derived_stage = evidence.parent / (".hegel-m25-stage-" + run_hex)
    expected_retirement_rows = tuple(
        {
            "original_output_role": role,
            "retired_output_path": str(output),
            "retirement_marker_path": str(
                _preseed_abort_retirement_marker_path_v1(output)
            ),
        }
        for role, output in (
            ("evidence", evidence),
            ("promotion", promotion),
            ("publication_receipt", receipt_output),
        )
    )
    try:
        normalized_retirement_rows = tuple(dict(row) for row in retirement_rows)
    except (TypeError, ValueError):
        _fail(FAIL_TRANSACTION_LOCK, "terminal tombstone retirement rows differ")
    try:
        custody_metadata = custody.lstat()
        custody_parent_metadata = custody.parent.lstat()
        public_parent_metadata = evidence.parent.lstat()
    except OSError as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"terminal tombstone parent inode is absent: {exc}")
    if (
        fields.get("custody_absolute_path") != str(custody)
        or fields.get("custody_st_dev") != custody_metadata.st_dev
        or fields.get("custody_st_ino") != custody_metadata.st_ino
        or fields.get("custody_parent_absolute_path") != str(custody.parent)
        or fields.get("custody_parent_st_dev") != custody_parent_metadata.st_dev
        or fields.get("custody_parent_st_ino") != custody_parent_metadata.st_ino
        or fields.get("public_parent_absolute_path") != str(evidence.parent)
        or fields.get("public_parent_st_dev") != public_parent_metadata.st_dev
        or fields.get("public_parent_st_ino") != public_parent_metadata.st_ino
        or fields.get("stage_absolute_path") != str(derived_stage)
        or fields.get("public_evidence_path") != str(evidence)
        or fields.get("public_promotion_path") != str(promotion)
        or fields.get("publication_receipt_path") != str(receipt_output)
        or normalized_retirement_rows != expected_retirement_rows
        or promotion.parent != evidence.parent
        or receipt_output.parent != evidence.parent
        or (expected_basis_commit is not None and basis != expected_basis_commit)
        or (expected_run_id is not None and run_id != expected_run_id)
        or (expected_ledger_id is not None and ledger_id != expected_ledger_id)
        or (
            expected_daemon_binding is not None
            and daemon_binding != expected_daemon_binding
        )
        or (
            expected_absence_sha256 is not None
            and fields.get("actor_absence_receipt_sha256")
            != expected_absence_sha256
        )
    ):
        _fail(FAIL_TRANSACTION_LOCK, "terminal tombstone derived identity differs")
    tombstone_sha256 = hashlib.sha256(payload).hexdigest()
    marker_required = {
        "schema", "basis_commit", "run_id_hex", "ledger_id_hex",
        "original_output_role", "retired_output_path",
        "terminal_tombstone_path", "terminal_tombstone_sha256",
        "original_public_evidence_path", "original_public_promotion_path",
        "original_publication_receipt_path", "path_permanently_retired",
        "formal_gate_artifact",
    }
    for row in expected_retirement_rows:
        marker_path = Path(row["retirement_marker_path"])
        marker_payload, marker = FormalCeremonyTransactionV1._read_canonical_regular_file(
            marker_path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="preseed abort output-retirement marker",
        )
        expected_marker = {
            "schema": _PRESEED_ABORT_RETIREMENT_SCHEMA,
            "basis_commit": basis,
            "run_id_hex": run_hex,
            "ledger_id_hex": ledger_hex,
            "original_output_role": row["original_output_role"],
            "retired_output_path": row["retired_output_path"],
            "terminal_tombstone_path": str(
                evidence.with_name(
                    f".{evidence.name}.hegel-preseed-abort-terminal.json"
                )
            ),
            "terminal_tombstone_sha256": tombstone_sha256,
            "original_public_evidence_path": str(evidence),
            "original_public_promotion_path": str(promotion),
            "original_publication_receipt_path": str(receipt_output),
            "path_permanently_retired": True,
            "formal_gate_artifact": False,
        }
        if (
            set(marker) != marker_required
            or marker != expected_marker
            or marker_payload != _canonical_json(expected_marker)
        ):
            _fail(FAIL_TRANSACTION_LOCK, "output-retirement marker differs")
    return basis, run_id, ledger_id, daemon_binding, derived_stage


def _preseed_abort_record_v1(
    path: Path, *, inode_kind: str, label: str
) -> dict[str, object]:
    if path.is_symlink():
        _fail(FAIL_TRANSACTION_LOCK, f"preseed abort {label} is a symlink")
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail(FAIL_TRANSACTION_LOCK, f"preseed abort {label} is absent: {exc}")
    if inode_kind == "directory":
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort {label} directory differs")
        payload_digest: str | None = None
        mode = "0700"
    else:
        if (
            inode_kind != "regular_file"
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort {label} file differs")
        try:
            payload_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort {label} cannot be read: {exc}")
        mode = "0600"
    return {
        "absolute_path": str(path),
        "inode_kind": inode_kind,
        "mode_octal": mode,
        "st_dev": metadata.st_dev,
        "st_ino": metadata.st_ino,
        "payload_sha256_or_null": payload_digest,
    }


def _abort_preseed_reserved_transaction_core_v1(
    *,
    custody_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
    actors: "CeremonyActorsV1",
    fault_injector: Callable[[str], None] | None = None,
) -> None:
    """Crash-resumable exact abort for one RESERVED, pre-seed transaction.

    Actor absence is re-established on every invocation.  An immutable plan
    binds every deletable inode and byte digest before the first unlink.
    Progress is inferred solely from one absent prefix followed by one exact
    present suffix.  The plan is deleted penultimately and the live-locked
    persistent ceremony lock is deleted last; no recursive deletion exists.
    """

    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot authorize preseed abort")

    def fault(point: str) -> None:
        if fault_injector is not None:
            fault_injector(point)

    _reject_caller_symlink_chain_v1(custody_directory, "preseed abort custody path")
    _reject_caller_symlink_chain_v1(public_evidence_path, "preseed abort evidence path")
    _reject_caller_symlink_chain_v1(public_promotion_path, "preseed abort promotion path")
    custody = Path(os.path.abspath(os.fspath(custody_directory)))
    evidence = Path(os.path.abspath(os.fspath(public_evidence_path)))
    promotion = Path(os.path.abspath(os.fspath(public_promotion_path)))
    receipt_output = promotion.with_name(promotion.name + ".publication-receipt.json")
    outputs = (evidence, promotion, receipt_output)
    tombstone_path = evidence.with_name(
        f".{evidence.name}.hegel-preseed-abort-terminal.json"
    )
    if custody.is_symlink():
        _fail(FAIL_CUSTODY, "preseed abort custody directory is a symlink")
    evidence_reservation_probe = evidence.with_name(
        f".{evidence.name}.hegel-reserved"
    )
    anchor_guard: FormalCeremonyTransactionV1 | None = None
    directory_descriptor = -1
    public_parent_descriptor = -1
    if evidence_reservation_probe.exists() or evidence_reservation_probe.is_symlink():
        anchor_guard, _anchor_fields = _acquire_host_recovery_anchor_v1(
            custody_directory=custody,
            public_evidence_path=evidence,
            public_promotion_path=promotion,
            actors=actors,
        )
    else:
        try:
            directory_descriptor = os.open(
                custody,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            directory_metadata = os.fstat(directory_descriptor)
            path_metadata = custody.lstat()
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or stat.S_IMODE(directory_metadata.st_mode) != 0o700
                or (directory_metadata.st_dev, directory_metadata.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                _fail(FAIL_CUSTODY, "preseed abort custody directory differs")
            fcntl.flock(directory_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
                directory_descriptor = -1
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort directory is live-locked: {exc}")
    try:
        public_parent_descriptor = os.open(
            evidence.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        public_parent_metadata = os.fstat(public_parent_descriptor)
        public_parent_path_metadata = evidence.parent.lstat()
        if (
            not stat.S_ISDIR(public_parent_metadata.st_mode)
            or (public_parent_metadata.st_dev, public_parent_metadata.st_ino)
            != (
                public_parent_path_metadata.st_dev,
                public_parent_path_metadata.st_ino,
            )
        ):
            _fail(FAIL_PUBLICATION, "preseed abort public parent inode differs")
        fcntl.flock(public_parent_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if public_parent_descriptor >= 0:
            os.close(public_parent_descriptor)
        if anchor_guard is not None:
            anchor_guard.close_lock()
        elif directory_descriptor >= 0:
            fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            os.close(directory_descriptor)
        _fail(FAIL_TRANSACTION_LOCK, f"preseed abort public parent is live-locked: {exc}")
    except BaseException:
        if public_parent_descriptor >= 0:
            os.close(public_parent_descriptor)
        if anchor_guard is not None:
            anchor_guard.close_lock()
        elif directory_descriptor >= 0:
            fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            os.close(directory_descriptor)
        raise
    lock_descriptor = -1
    try:
        lock_path = custody / "phase3_m25_ceremony.lock"
        if not (lock_path.exists() or lock_path.is_symlink()):
            reservations = tuple(
                output.with_name(f".{output.name}.hegel-reserved")
                for output in outputs
            )
            tombstone_payload, tombstone = (
                FormalCeremonyTransactionV1._read_canonical_regular_file(
                    tombstone_path,
                    mode=0o600,
                    code=FAIL_TRANSACTION_LOCK,
                    label="preseed abort terminal tombstone",
                )
            )
            basis_commit, run_id, _ledger_id, expected_daemon, stage_path = (
                _validate_preseed_abort_terminal_tombstone_v1(
                    tombstone_payload,
                    tombstone,
                    custody=custody,
                    evidence=evidence,
                    promotion=promotion,
                    receipt_output=receipt_output,
                )
            )
            live_absence = actors.recover_preseed_private_state_and_verify_absent(run_id)
            live_payload = _validate_preseed_abort_actor_absence_receipt_v1(
                live_absence,
                basis_commit=basis_commit,
                run_id=run_id,
                expected_daemon_binding=expected_daemon,
            )
            if tombstone.get("actor_absence_receipt_sha256") != hashlib.sha256(
                live_payload
            ).hexdigest():
                _fail(FAIL_CONTAINER, "terminal abort actor absence binding differs")
            if (
                any(custody.iterdir())
                or stage_path.exists()
                or stage_path.is_symlink()
                or any(path.exists() or path.is_symlink() for path in (*outputs, *reservations))
            ):
                _fail(FAIL_TRANSACTION_LOCK, "completed preseed abort has unexpected residue")
            _fsync_directory_v1(custody)
            _fsync_directory_v1(evidence.parent)
            return
        lock_payload, lock_fields = FormalCeremonyTransactionV1._read_canonical_regular_file(
            lock_path,
            mode=0o600,
            code=FAIL_TRANSACTION_LOCK,
            label="preseed abort persistent lock",
        )
        try:
            run_id = bytes.fromhex(str(lock_fields["run_id_hex"]))
            ledger_id = bytes.fromhex(str(lock_fields["ledger_id_hex"]))
        except (KeyError, TypeError, ValueError) as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort identity is invalid: {exc}")
        transported_intent = lock_fields.get("prestage_intent_transport_or_null")
        try:
            restored_intent = _restore(transported_intent)
        except (TypeError, ValueError) as exc:
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort intent transport is invalid: {exc}")
        if type(restored_intent) is not dict:
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort intent transport is absent")
        identity = FormalCeremonyTransactionV1(
            basis_commit=lock_fields.get("basis_commit"),  # type: ignore[arg-type]
            custody_directory=custody,
            public_evidence_path=evidence,
            public_promotion_path=promotion,
            run_id=run_id,
            ledger_id=ledger_id,
            prestage_intent_fields=restored_intent,
        )
        if (
            len(run_id) != 16
            or len(ledger_id) != 16
            or run_id == ledger_id
            or lock_fields != identity._lock_fields()
            or lock_payload != _canonical_json(identity._lock_fields())
        ):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort lock fields differ")
        try:
            lock_descriptor = os.open(
                lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
                lock_descriptor = -1
            _fail(FAIL_TRANSACTION_LOCK, f"preseed abort lock is live-locked: {exc}")

        forbidden = {
            "split_seed_instantiation.marker", "split_seed_generation.intent",
            "split_seed_generation.complete", "split_master_seed.bin",
        }
        if any(
            (custody / name).exists() or (custody / name).is_symlink()
            for name in forbidden
        ):
            _fail(FAIL_CUSTODY, "preseed abort found marker or seed continuity state")
        if any(output.exists() or output.is_symlink() for output in outputs):
            _fail(FAIL_PUBLICATION, "preseed abort found a final public output")

        stage = evidence.parent / (".hegel-m25-stage-" + run_id.hex())
        output_reservations = tuple(identity._output_reservation_paths[output] for output in outputs)
        run_reservation = custody / f"opaque-run-{run_id.hex()}.reserved"
        ledger_reservation = custody / f"opaque-ledger-{ledger_id.hex()}.reserved"
        absence_path = custody / _PRESEED_ABORT_ABSENCE_FILENAME
        plan_path = custody / _PRESEED_ABORT_PLAN_FILENAME
        absence_next = absence_path.with_name(absence_path.name + ".next")
        plan_next = plan_path.with_name(plan_path.name + ".next")
        daemon_binding = identity._prestage_intent_fields.get(
            "live_actor_protocol_daemon_receipt_binding"
        )
        runtime_bindings = identity._prestage_intent_fields.get(
            "runtime_binding_fields"
        )
        actor_report_sha256 = identity._prestage_intent_fields.get(
            "actor_qualification_report_sha256"
        )
        if (
            type(daemon_binding) is not bytes
            or len(daemon_binding) != 32
            or not isinstance(runtime_bindings, Mapping)
            or type(runtime_bindings.get("actor_profile_sha256")) is not bytes
            or len(runtime_bindings["actor_profile_sha256"]) != 32
            or type(actor_report_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", actor_report_sha256) is None
        ):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort actor binding is absent")
        expected_absence_fields: dict[str, object] = {
            "schema": _PRESEED_ABORT_ABSENCE_SCHEMA,
            "basis_commit": identity.basis_commit,
            "run_id_hex": run_id.hex(),
            "exact_run_label_checked": True,
            "actor_containers_absent": True,
            "actor_key_volumes_absent": True,
            "seed_continuity_state_absent": True,
            "docker_daemon_receipt_sha256": daemon_binding.hex(),
        }
        expected_absence_fields["receipt_sha256"] = hashlib.sha256(
            _canonical_json(expected_absence_fields)
        ).hexdigest()
        expected_absence_payload = _canonical_json(expected_absence_fields)

        plan_present = plan_path.exists() or plan_path.is_symlink()
        plan_next_present = plan_next.exists() or plan_next.is_symlink()
        if not plan_present:
            # Once the immutable plan has been deleted, the only legal crash
            # point leaves the locked persistent lock as the exact final
            # suffix.  Re-establish actor absence, then finish that last step.
            if (
                {path.name for path in custody.iterdir()} == {lock_path.name}
                and not (stage.exists() or stage.is_symlink())
                and not any(
                    path.exists() or path.is_symlink() for path in output_reservations
                )
            ):
                terminal_payload, terminal = (
                    FormalCeremonyTransactionV1._read_canonical_regular_file(
                        tombstone_path,
                        mode=0o600,
                        code=FAIL_TRANSACTION_LOCK,
                        label="preseed abort terminal tombstone",
                    )
                )
                _validate_preseed_abort_terminal_tombstone_v1(
                    terminal_payload,
                    terminal,
                    custody=custody,
                    evidence=evidence,
                    promotion=promotion,
                    receipt_output=receipt_output,
                    expected_basis_commit=identity.basis_commit,
                    expected_run_id=run_id,
                    expected_ledger_id=ledger_id,
                    expected_daemon_binding=daemon_binding,
                    expected_absence_sha256=hashlib.sha256(
                        expected_absence_payload
                    ).hexdigest(),
                )
                live_absence = actors.recover_preseed_private_state_and_verify_absent(run_id)
                live_absence_payload = _validate_preseed_abort_actor_absence_receipt_v1(
                    live_absence,
                    basis_commit=identity.basis_commit,
                    run_id=run_id,
                    expected_daemon_binding=daemon_binding,
                )
                if live_absence_payload != expected_absence_payload:
                    _fail(FAIL_CONTAINER, "terminal abort actor absence differs")
                # Re-fsync every parent whose earlier unlink may have been the
                # crash boundary before authorizing the final lock removal.
                _fsync_directory_v1(evidence.parent)
                _fsync_directory_v1(custody)
                fault("before_preseed_abort_delete_terminal_lock")
                lock_path.unlink()
                fault("after_preseed_abort_delete_terminal_lock_before_parent_fsync")
                _fsync_directory_v1(custody)
                fault("after_preseed_abort_delete_terminal_lock_parent_fsync")
                return

            expected_custody_before_plan = {
                lock_path.name, run_reservation.name, ledger_reservation.name,
            }
            optional_abort_names = {plan_next.name}
            actual_custody = {path.name for path in custody.iterdir()}
            if (
                not expected_custody_before_plan <= actual_custody
                or not actual_custody <= expected_custody_before_plan | optional_abort_names
            ):
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort pre-plan custody differs")

            for kind, value, reservation in (
                ("run", run_id, run_reservation),
                ("ledger", ledger_id, ledger_reservation),
            ):
                payload, fields = FormalCeremonyTransactionV1._read_canonical_regular_file(
                    reservation,
                    mode=0o600,
                    code=FAIL_TRANSACTION_LOCK,
                    label=f"preseed opaque {kind} reservation",
                )
                expected = {
                    "schema": "hegel-phase3-m25-opaque-id-reservation/1",
                    "kind": kind,
                    "opaque_id_hex": value.hex(),
                    "basis_commit": identity.basis_commit,
                }
                if fields != expected or payload != _canonical_json(expected):
                    _fail(FAIL_TRANSACTION_LOCK, f"preseed opaque {kind} reservation differs")
            for output, reservation in zip(outputs, output_reservations, strict=True):
                payload, fields = FormalCeremonyTransactionV1._read_canonical_regular_file(
                    reservation,
                    mode=0o600,
                    code=FAIL_TRANSACTION_LOCK,
                    label="preseed output reservation",
                )
                expected = identity._output_reservation_fields(output)
                if fields != expected or payload != _canonical_json(expected):
                    _fail(FAIL_TRANSACTION_LOCK, "preseed output reservation differs")
            if (
                stage.is_symlink()
                or not stage.is_dir()
                or stat.S_IMODE(stage.stat().st_mode) != 0o700
            ):
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort stage directory differs")
            required_stage = {
                "transaction-journal.json", _PRESTAGE_INTENT_FILENAME,
                _LIVE_QUALIFICATION_BUNDLE_FILENAME, _RECOVERY_ANCHOR_FILENAME,
                _RECOVERY_ANCHOR_READY_FILENAME,
            }
            checkpoint_variants = {
                _ACTOR_TRUST_CHECKPOINT_FILENAME,
                _ACTOR_TRUST_CHECKPOINT_FILENAME + ".next",
            }
            stage_entries = {path.name for path in stage.iterdir()}
            if (
                not required_stage <= stage_entries
                or len(stage_entries & checkpoint_variants) != 1
                or stage_entries != required_stage | (stage_entries & checkpoint_variants)
            ):
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort stage path set differs")
            _journal_payload, journal = FormalCeremonyTransactionV1._read_canonical_regular_file(
                stage / "transaction-journal.json",
                mode=0o600,
                code=FAIL_TRANSACTION_LOCK,
                label="preseed abort transaction journal",
            )
            if journal != identity._journal_fields("RESERVED"):
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort journal is not RESERVED")
            intent_payload, _intent_fields = FormalCeremonyTransactionV1._read_canonical_regular_file(
                stage / _PRESTAGE_INTENT_FILENAME,
                mode=0o600,
                code=FAIL_TRANSACTION_LOCK,
                label="preseed abort intent",
            )
            if intent_payload != identity._prestage_intent_bytes:
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort intent differs from lock")

            rows: list[dict[str, object]] = []
            for reservation in output_reservations:
                rows.append(_preseed_abort_record_v1(
                    reservation, inode_kind="regular_file", label=reservation.name
                ))
            for name in sorted(stage_entries):
                rows.append(_preseed_abort_record_v1(
                    stage / name, inode_kind="regular_file", label=name
                ))
            rows.append(_preseed_abort_record_v1(
                stage, inode_kind="directory", label="stage directory"
            ))
            for reservation in (run_reservation, ledger_reservation):
                rows.append(_preseed_abort_record_v1(
                    reservation, inode_kind="regular_file", label=reservation.name
                ))
            rows.append({
                "absolute_path": str(absence_path),
                "inode_kind": "regular_file",
                "mode_octal": "0600",
                "st_dev": None,
                "st_ino": None,
                "payload_sha256_or_null": hashlib.sha256(
                    expected_absence_payload
                ).hexdigest(),
            })
            rows.append({
                "absolute_path": str(plan_path),
                "inode_kind": "regular_file",
                "mode_octal": "0600",
                "st_dev": None,
                "st_ino": None,
                "payload_sha256_or_null": None,
            })
            rows.append(_preseed_abort_record_v1(
                lock_path, inode_kind="regular_file", label=lock_path.name
            ))
            ordered_rows = tuple({"order": index, **row} for index, row in enumerate(rows))
            custody_metadata = custody.lstat()
            custody_parent_metadata = custody.parent.lstat()
            public_parent_metadata = evidence.parent.lstat()
            plan_fields: dict[str, object] = {
                "schema": _PRESEED_ABORT_PLAN_SCHEMA,
                "basis_commit": identity.basis_commit,
                "run_id_hex": run_id.hex(),
                "ledger_id_hex": ledger_id.hex(),
                "custody_absolute_path": str(custody),
                "stage_absolute_path": str(stage),
                "public_evidence_path": str(evidence),
                "public_promotion_path": str(promotion),
                "publication_receipt_path": str(receipt_output),
                "actor_absence_receipt_sha256": hashlib.sha256(
                    expected_absence_payload
                ).hexdigest(),
                "docker_daemon_receipt_sha256": daemon_binding.hex(),
                "actor_profile_sha256": runtime_bindings[
                    "actor_profile_sha256"
                ].hex(),
                "actor_qualification_report_sha256": actor_report_sha256,
                "terminal_tombstone_path": str(tombstone_path),
                "forbidden_seed_state_names": tuple(sorted(forbidden)),
                "custody_st_dev": custody_metadata.st_dev,
                "custody_st_ino": custody_metadata.st_ino,
                "custody_parent_st_dev": custody_parent_metadata.st_dev,
                "custody_parent_st_ino": custody_parent_metadata.st_ino,
                "public_parent_st_dev": public_parent_metadata.st_dev,
                "public_parent_st_ino": public_parent_metadata.st_ino,
                "ordered_deletion_rows": ordered_rows,
                "progress_is_exact_absent_prefix_only": True,
                "recursive_delete_used": False,
                "raw_seed_bytes_read": False,
                "plan_deleted_penultimately_lock_deleted_last": True,
            }
            plan_payload = _canonical_json(plan_fields)
            _write_atomic_resumable_v1(
                plan_path,
                plan_payload,
                0o600,
                fault=fault_injector,
                fault_label="preseed_abort_plan",
            )
            fault("after_preseed_abort_plan_durable")
        else:
            if plan_next_present:
                _fail(FAIL_TRANSACTION_LOCK, "durable preseed abort plan retains next")
            plan_payload, plan_fields = FormalCeremonyTransactionV1._read_canonical_regular_file(
                plan_path,
                mode=0o600,
                code=FAIL_TRANSACTION_LOCK,
                label="preseed abort plan",
            )

        # Validate the immutable plan's complete authority surface even after
        # earlier targets have disappeared.
        expected_plan_keys = {
            "schema", "basis_commit", "run_id_hex", "ledger_id_hex",
            "custody_absolute_path", "stage_absolute_path", "public_evidence_path",
            "public_promotion_path", "publication_receipt_path",
            "actor_absence_receipt_sha256", "docker_daemon_receipt_sha256",
            "actor_profile_sha256", "actor_qualification_report_sha256",
            "terminal_tombstone_path", "forbidden_seed_state_names",
            "custody_st_dev", "custody_st_ino", "custody_parent_st_dev",
            "custody_parent_st_ino", "public_parent_st_dev",
            "public_parent_st_ino",
            "ordered_deletion_rows",
            "progress_is_exact_absent_prefix_only", "recursive_delete_used",
            "raw_seed_bytes_read", "plan_deleted_penultimately_lock_deleted_last",
        }
        raw_rows = plan_fields.get("ordered_deletion_rows")
        if (
            set(plan_fields) != expected_plan_keys
            or plan_fields.get("schema") != _PRESEED_ABORT_PLAN_SCHEMA
            or plan_fields.get("basis_commit") != identity.basis_commit
            or plan_fields.get("run_id_hex") != run_id.hex()
            or plan_fields.get("ledger_id_hex") != ledger_id.hex()
            or plan_fields.get("custody_absolute_path") != str(custody)
            or plan_fields.get("stage_absolute_path") != str(stage)
            or plan_fields.get("public_evidence_path") != str(evidence)
            or plan_fields.get("public_promotion_path") != str(promotion)
            or plan_fields.get("publication_receipt_path") != str(receipt_output)
            or plan_fields.get("actor_absence_receipt_sha256")
            != hashlib.sha256(expected_absence_payload).hexdigest()
            or plan_fields.get("docker_daemon_receipt_sha256")
            != daemon_binding.hex()
            or plan_fields.get("actor_profile_sha256")
            != runtime_bindings["actor_profile_sha256"].hex()
            or plan_fields.get("actor_qualification_report_sha256")
            != actor_report_sha256
            or plan_fields.get("terminal_tombstone_path") != str(tombstone_path)
            or tuple(plan_fields.get("forbidden_seed_state_names", ()))
            != tuple(sorted(forbidden))
            or plan_fields.get("custody_st_dev") != custody.lstat().st_dev
            or plan_fields.get("custody_st_ino") != custody.lstat().st_ino
            or plan_fields.get("custody_parent_st_dev")
            != custody.parent.lstat().st_dev
            or plan_fields.get("custody_parent_st_ino")
            != custody.parent.lstat().st_ino
            or plan_fields.get("public_parent_st_dev")
            != evidence.parent.lstat().st_dev
            or plan_fields.get("public_parent_st_ino")
            != evidence.parent.lstat().st_ino
            or plan_fields.get("progress_is_exact_absent_prefix_only") is not True
            or plan_fields.get("recursive_delete_used") is not False
            or plan_fields.get("raw_seed_bytes_read") is not False
            or plan_fields.get("plan_deleted_penultimately_lock_deleted_last") is not True
            or type(raw_rows) not in {list, tuple}
        ):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort plan fields differ")
        rows = list(raw_rows)  # type: ignore[arg-type]
        row_keys = {
            "order", "absolute_path", "inode_kind", "mode_octal", "st_dev",
            "st_ino", "payload_sha256_or_null",
        }
        if not rows or any(
            type(row) is not dict
            or set(row) != row_keys
            or row.get("order") != index
            for index, row in enumerate(rows)
        ):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort deletion rows differ")
        row_paths = tuple(Path(str(row["absolute_path"])) for row in rows)
        try:
            stage_index = row_paths.index(stage)
        except ValueError:
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort plan lacks the stage directory")
        stage_names = {path.name for path in row_paths[3:stage_index]}
        required_stage = {
            "transaction-journal.json", _PRESTAGE_INTENT_FILENAME,
            _LIVE_QUALIFICATION_BUNDLE_FILENAME, _RECOVERY_ANCHOR_FILENAME,
            _RECOVERY_ANCHOR_READY_FILENAME,
        }
        checkpoint_variants = {
            _ACTOR_TRUST_CHECKPOINT_FILENAME,
            _ACTOR_TRUST_CHECKPOINT_FILENAME + ".next",
        }
        expected_tail = (
            stage, run_reservation, ledger_reservation, absence_path, plan_path, lock_path
        )
        if (
            row_paths[:3] != output_reservations
            or any(path.parent != stage for path in row_paths[3:stage_index])
            or not required_stage <= stage_names
            or len(stage_names & checkpoint_variants) != 1
            or stage_names != required_stage | (stage_names & checkpoint_variants)
            or row_paths[stage_index:] != expected_tail
            or len(row_paths) != len(set(row_paths))
        ):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort ordered path authority differs")

        # The plan is now durable.  Only now may the exact run-labelled Docker
        # cleanup occur.  Every re-entry repeats the live absence check before
        # examining or advancing the deletion prefix.
        fault("before_preseed_abort_actor_absence")
        live_absence = actors.recover_preseed_private_state_and_verify_absent(run_id)
        live_absence_payload = _validate_preseed_abort_actor_absence_receipt_v1(
            live_absence,
            basis_commit=identity.basis_commit,
            run_id=run_id,
            expected_daemon_binding=daemon_binding,
        )
        fault("after_preseed_abort_actor_absence")
        if live_absence_payload != expected_absence_payload:
            _fail(FAIL_CONTAINER, "preseed actor absence differs from frozen plan")

        absence_index = row_paths.index(absence_path)
        deletion_already_started = any(
            not (path.exists() or path.is_symlink())
            for path in row_paths[:absence_index]
        )
        if absence_path.exists() or absence_path.is_symlink() or not deletion_already_started:
            _write_atomic_resumable_v1(
                absence_path,
                expected_absence_payload,
                0o600,
                fault=fault_injector,
                fault_label="preseed_abort_actor_absence",
            )
            fault("after_preseed_abort_actor_absence_receipt_durable")
        if absence_next.exists() or absence_next.is_symlink():
            _fail(FAIL_TRANSACTION_LOCK, "abort absence receipt retains a next inode")

        retirement_marker_rows = tuple(
            {
                "original_output_role": role,
                "retired_output_path": str(output),
                "retirement_marker_path": str(
                    _preseed_abort_retirement_marker_path_v1(output)
                ),
            }
            for role, output in (
                ("evidence", evidence),
                ("promotion", promotion),
                ("publication_receipt", receipt_output),
            )
        )
        tombstone_fields: dict[str, object] = {
            "schema": _PRESEED_ABORT_TOMBSTONE_SCHEMA,
            "basis_commit": identity.basis_commit,
            "run_id_hex": run_id.hex(),
            "ledger_id_hex": ledger_id.hex(),
            "custody_absolute_path": str(custody),
            "custody_st_dev": plan_fields["custody_st_dev"],
            "custody_st_ino": plan_fields["custody_st_ino"],
            "custody_parent_absolute_path": str(custody.parent),
            "custody_parent_st_dev": plan_fields["custody_parent_st_dev"],
            "custody_parent_st_ino": plan_fields["custody_parent_st_ino"],
            "public_parent_absolute_path": str(evidence.parent),
            "public_parent_st_dev": plan_fields["public_parent_st_dev"],
            "public_parent_st_ino": plan_fields["public_parent_st_ino"],
            "stage_absolute_path": str(stage),
            "public_evidence_path": str(evidence),
            "public_promotion_path": str(promotion),
            "publication_receipt_path": str(receipt_output),
            "abort_plan_sha256": hashlib.sha256(plan_payload).hexdigest(),
            "actor_absence_receipt_sha256": hashlib.sha256(
                expected_absence_payload
            ).hexdigest(),
            "docker_daemon_receipt_sha256": daemon_binding.hex(),
            "retirement_marker_rows": retirement_marker_rows,
            "all_plan_targets_must_be_absent_before_success": True,
            "formal_outputs_published": False,
            "raw_seed_bytes_read": False,
        }
        tombstone_payload = _canonical_json(tombstone_fields)
        tombstone_next = tombstone_path.with_name(tombstone_path.name + ".next")
        if deletion_already_started and not (
            tombstone_path.exists()
            or tombstone_path.is_symlink()
            or tombstone_next.exists()
            or tombstone_next.is_symlink()
        ):
            _fail(FAIL_TRANSACTION_LOCK, "started abort lost its terminal tombstone")
        _write_atomic_resumable_v1(
            tombstone_path,
            tombstone_payload,
            0o600,
            fault=fault_injector,
            fault_label="preseed_abort_terminal_tombstone",
        )
        fault("after_preseed_abort_terminal_tombstone_durable")
        tombstone_sha256 = hashlib.sha256(tombstone_payload).hexdigest()
        for row in retirement_marker_rows:
            marker_path = Path(row["retirement_marker_path"])
            marker_next = marker_path.with_name(marker_path.name + ".next")
            if deletion_already_started and not (
                marker_path.exists()
                or marker_path.is_symlink()
                or marker_next.exists()
                or marker_next.is_symlink()
            ):
                _fail(
                    FAIL_TRANSACTION_LOCK,
                    "started abort lost an output-retirement marker",
                )
            marker_fields = {
                "schema": _PRESEED_ABORT_RETIREMENT_SCHEMA,
                "basis_commit": identity.basis_commit,
                "run_id_hex": run_id.hex(),
                "ledger_id_hex": ledger_id.hex(),
                "original_output_role": row["original_output_role"],
                "retired_output_path": row["retired_output_path"],
                "terminal_tombstone_path": str(tombstone_path),
                "terminal_tombstone_sha256": tombstone_sha256,
                "original_public_evidence_path": str(evidence),
                "original_public_promotion_path": str(promotion),
                "original_publication_receipt_path": str(receipt_output),
                "path_permanently_retired": True,
                "formal_gate_artifact": False,
            }
            _write_atomic_resumable_v1(
                marker_path,
                _canonical_json(marker_fields),
                0o600,
                fault=fault_injector,
                fault_label=(
                    "preseed_abort_retirement_"
                    + str(row["original_output_role"])
                ),
            )
            fault(
                "after_preseed_abort_retirement_marker_durable_"
                + str(row["original_output_role"])
            )

        present = tuple(path.exists() or path.is_symlink() for path in row_paths)
        first_present = next((index for index, value in enumerate(present) if value), len(present))
        if present != (False,) * first_present + (True,) * (len(present) - first_present):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort progress is not an exact prefix")
        if first_present >= len(rows) or row_paths[-1] != lock_path:
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort lost its terminal lock")
        # A prior process may have died after unlink/rmdir but before the
        # corresponding parent fsync.  Re-fsync every still-addressable parent
        # of the inferred absent prefix before advancing it.
        prefix_parents = {
            path.parent
            for path in row_paths[:first_present]
            if path.parent.exists() and not path.parent.is_symlink()
        }
        prefix_parents.update({custody, evidence.parent})
        for parent in sorted(prefix_parents, key=lambda value: str(value)):
            _fsync_directory_v1(parent)

        expected_custody_names = {
            path.name
            for path, exists in zip(row_paths, present, strict=True)
            if exists and path.parent == custody
        }
        if {path.name for path in custody.iterdir()} != expected_custody_names:
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort custody suffix contains unknown state")
        if stage.exists() or stage.is_symlink():
            expected_stage_names = {
                path.name
                for path, exists in zip(row_paths, present, strict=True)
                if exists and path.parent == stage
            }
            if stage.is_symlink() or {path.name for path in stage.iterdir()} != expected_stage_names:
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort stage suffix contains unknown state")

        for index in range(first_present, len(rows)):
            row = rows[index]
            path = row_paths[index]
            if path.is_symlink():
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort suffix contains a symlink")
            metadata = path.lstat()
            kind = row["inode_kind"]
            mode = int(str(row["mode_octal"]), 8)
            if (
                (kind == "regular_file" and not stat.S_ISREG(metadata.st_mode))
                or (kind == "directory" and not stat.S_ISDIR(metadata.st_mode))
                or kind not in {"regular_file", "directory"}
                or stat.S_IMODE(metadata.st_mode) != mode
                or (
                    path not in {plan_path, absence_path}
                    and (metadata.st_dev, metadata.st_ino)
                    != (row["st_dev"], row["st_ino"])
                )
            ):
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort suffix inode differs")
            digest = row["payload_sha256_or_null"]
            if kind == "regular_file" and path != plan_path:
                if (
                    type(digest) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    _fail(FAIL_TRANSACTION_LOCK, "preseed abort suffix bytes differ")
            elif digest is not None:
                _fail(FAIL_TRANSACTION_LOCK, "preseed abort non-payload digest differs")

        for index in range(first_present, len(rows)):
            path = row_paths[index]
            fault(f"before_preseed_abort_delete_{index}")
            if rows[index]["inode_kind"] == "directory":
                path.rmdir()
            else:
                path.unlink()
            fault(f"after_preseed_abort_delete_{index}_before_parent_fsync")
            _fsync_directory_v1(path.parent)
            fault(f"after_preseed_abort_delete_{index}_parent_fsync")
        if any(custody.iterdir()):
            _fail(FAIL_TRANSACTION_LOCK, "preseed abort left custody residue")
    finally:
        if lock_descriptor >= 0:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        if public_parent_descriptor >= 0:
            fcntl.flock(public_parent_descriptor, fcntl.LOCK_UN)
            os.close(public_parent_descriptor)
        if anchor_guard is not None:
            anchor_guard.close_lock()
        elif directory_descriptor >= 0:
            fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
            os.close(directory_descriptor)


def abort_preseed_reserved_transaction_v1(
    *,
    custody_directory: Path,
    public_evidence_path: Path,
    public_promotion_path: Path,
    actors: "CeremonyActorsV1",
) -> None:
    """Public exact-abort boundary; only the sealed Docker backend may erase."""

    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot authorize preseed abort")
    if type(actors) is not DockerCeremonyActorsV1:
        _fail(
            FAIL_PREFLIGHT,
            "formal preseed abort requires the sealed DockerCeremonyActorsV1 backend",
        )
    _abort_preseed_reserved_transaction_core_v1(
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
        actors=actors,
    )


def resume_pending_split_calculators_v1(
    *,
    recovery: PendingCeremonyRecoveryV1,
    custody_directory: Path,
    rust_formal_replay_binary: Path,
) -> tuple[bytes, bytes]:
    """Explicitly replay split calculators from a durable PENDING seed.

    The ordinary execute path never calls this function.  The recovered
    purpose-1 key must already exist in its run-labelled volume. The seed
    worker either accepts exact intent + 32-byte seed + durable completion
    receipt without a new CSPRNG call, or accepts the narrowly proven
    PENDING-with-no-intent edge and performs first genesis after first writing
    durable intent and an empty inode. Any intent-bearing incomplete state and
    every missing recovery key are terminal; neither may be replaced.
    """

    if not isinstance(recovery, PendingCeremonyRecoveryV1) or recovery.lock_descriptor < 0:
        _fail(FAIL_TRANSACTION_LOCK, "pending recovery lock is not held")
    basis = build_formal_static_basis_v1(recovery.basis_commit)
    backend = DockerCeremonyActorsV1(
        basis_commit=recovery.basis_commit,
        custody_directory=custody_directory,
        rust_formal_replay_binary=rust_formal_replay_binary,
        timestamp=recovery.marker_snapshot.created_at_unix_seconds,
    )
    backend.validate_rust_replay_binding(basis)
    backend.validate_rust_bridge_dag_binding()
    backend.prepare_pending_recovery(recovery.run_id)
    started = False
    try:
        backend.start()
        started = True
        backend.keygen(1)  # keygen-resume: missing key is terminal, never regenerated
        frames = backend.resume_pending_seed_split()
        require_full_split_response_agreement_v2(*frames)
        return frames
    finally:
        if started:
            # Recovery never authorizes private-volume destruction or marker
            # completion.  It only reconstructs public calculator frames.
            backend.stop_for_recovery_and_verify_absent()


def _continue_pre_stage_pending_recovery_core_v1(
    *,
    recovery: PendingCeremonyRecoveryV1,
    actors: "CeremonyActorsV1",
    replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
    fault_injector: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Finish a PENDING transaction by exact same-key/same-preimage replay.

    This is the only API allowed to sign before durable formal staging during
    recovery.  It obtains every formal identity from the bound prestage intent,
    the four retained role volumes and the retained seed.  It never draws a
    run, ledger, trust or key identity.  The purpose-1 seed worker independently
    enforces the one narrow first-genesis edge when PENDING exists but no seed
    intent exists; every intent-bearing incomplete state is terminal.
    """

    if (
        not isinstance(recovery, PendingCeremonyRecoveryV1)
        or recovery.lock_descriptor < 0
    ):
        _fail(FAIL_TRANSACTION_LOCK, "pending prestage recovery lock is not held")
    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot recover formal evidence")
    if not _PRESTAGE_RECOVERY_IMPLEMENTED:
        _fail(
            FAIL_PRESTAGE_RECOVERY_UNRESOLVED,
            "prestage recovery implementation has not passed its fault matrix",
        )

    intent = dict(recovery.prestage_intent_fields)
    timestamp = intent.get("created_at_unix_seconds")
    trust_hex = intent.get("trust_genesis_id_hex")
    actor_report = intent.get("actor_qualification_report")
    errata_report = intent.get("errata_qualification_report")
    bridge_report_hex = intent.get("rust_bridge_dag_qualification_report_sha256")
    protocol_bundle_id = intent.get(
        "live_actor_protocol_qualification_bundle_content_id"
    )
    transaction_local_bundle = intent.get(
        "live_actor_protocol_qualification_bundle"
    )
    frozen_daemon_binding = intent.get(
        "live_actor_protocol_daemon_receipt_binding"
    )
    if (
        type(timestamp) is not int
        or type(trust_hex) is not str
        or type(actor_report) is not dict
        or type(errata_report) is not dict
        or type(bridge_report_hex) is not str
        or type(protocol_bundle_id) is not bytes
        or len(protocol_bundle_id) != 32
        or type(transaction_local_bundle) is not dict
        or type(frozen_daemon_binding) is not bytes
        or len(frozen_daemon_binding) != 32
    ):
        _fail(FAIL_TRANSACTION_LOCK, "pending prestage intent cannot be reconstructed")
    trust_id = bytes.fromhex(trust_hex)
    expected_bridge_report_id = bytes.fromhex(bridge_report_hex)
    expected_qualification_key_ids = _qualification_only_key_ids_from_intent_v1(
        intent
    )

    basis = build_qualified_formal_static_basis_v1(recovery.basis_commit)
    implementation_roots = require_formal_ceremony_ready_v1(basis)
    validate_ceremony_admission_v1(
        actor_qualification_report=actor_report,
        errata_qualification_report=errata_report,
        basis_commit=recovery.basis_commit,
        committed_input_paths=REQUIRED_COMMIT_A_INPUTS,
    )
    live_protocol = replay_transaction_local_actor_protocol_bundle_v1(
        basis_commit=recovery.basis_commit,
        bundle=transaction_local_bundle,
    )
    if (
        live_protocol.bundle_content_id != protocol_bundle_id
        or dict(live_protocol.qualification_key_ids)
        != dict(expected_qualification_key_ids)
    ):
        _fail(
            FAIL_PREFLIGHT,
            "recovery signed actor-protocol bundle/key identity differs",
        )
    if isinstance(actors, DockerCeremonyActorsV1):
        actors.validate_rust_replay_binding(basis)
        actors.validate_rust_bridge_dag_binding()
    actor_blockers = tuple(actors.unresolved_formal_blockers())
    if actor_blockers:
        _fail(
            FAIL_PREFLIGHT,
            "authoritative recovery remains fail-closed: " + ",".join(actor_blockers),
        )
    if actors.bridge_qualification_report_id_v1() != expected_bridge_report_id:
        _fail(FAIL_PREFLIGHT, "recovery bridge qualification report identity differs")
    recovered_runtime_bindings = actors.prestage_runtime_binding_fields_v1(
        implementation_roots
    )
    frozen_runtime_bindings = intent.get("runtime_binding_fields")
    if (
        not isinstance(recovered_runtime_bindings, Mapping)
        or type(frozen_runtime_bindings) is not dict
        or dict(recovered_runtime_bindings) != frozen_runtime_bindings
        or _canonical_json(dict(recovered_runtime_bindings))
        != _canonical_json(frozen_runtime_bindings)
    ):
        _fail(FAIL_PREFLIGHT, "recovery runtime/implementation binding differs")

    transaction = FormalCeremonyTransactionV1(
        basis_commit=recovery.basis_commit,
        custody_directory=recovery.custody_directory,
        public_evidence_path=recovery.public_evidence_path,
        public_promotion_path=recovery.public_promotion_path,
        run_id=recovery.run_id,
        ledger_id=recovery.ledger_id,
        prestage_intent_fields=intent,
        fault_injector=fault_injector,
    )
    transaction._lock_descriptor = recovery.lock_descriptor
    recovery.lock_descriptor = -1
    transaction._anchor_descriptor = (
        None if recovery.anchor_descriptor < 0 else recovery.anchor_descriptor
    )
    recovery.anchor_descriptor = -1
    transaction._directory_descriptor = (
        None if recovery.directory_descriptor < 0 else recovery.directory_descriptor
    )
    recovery.directory_descriptor = -1
    transaction._stage_directory = recovery.stage_directory
    transaction._journal_path = recovery.stage_directory / "transaction-journal.json"
    transaction._state = recovery.journal_state
    actors_started = False
    actors_absent = False
    destruction_authorized = False
    try:
        actors.prepare_pending_recovery(recovery.run_id)
        actors.start()
        actors_started = True
        actors.validate_frozen_daemon_receipt_binding_v1(
            frozen_daemon_binding
        )
        public_keys: dict[int, bytes] = {}
        for purpose in (1, 2, 3, 4):
            public_keys[purpose] = actors.keygen(purpose)
            transaction._fault(f"after_recovery_actor_keygen_{purpose}")
        actor_trust = build_actor_trust_v1(
            public_keys=public_keys,
            timestamp=timestamp,
            basis_commit=recovery.basis_commit,
            trust_genesis_id=trust_id,
        )
        _require_formal_key_ids_disjoint_from_qualification_v1(
            actor_trust.key_ids,
            expected_qualification_key_ids,
        )
        validate_actor_trust_checkpoint_fields_v1(
            recovery.actor_trust_checkpoint_fields,
            expected_actor_trust=actor_trust,
            basis_commit=recovery.basis_commit,
            run_id=recovery.run_id,
            ledger_id=recovery.ledger_id,
            prestage_intent_sha256=recovery.prestage_intent_sha256,
        )
        expected_pending = MarkerSnapshot(
            "PENDING",
            SPLIT_VERSION_DIGEST,
            None,
            actor_trust.key_ids[1],
            timestamp,
        )
        if recovery.marker_snapshot != expected_pending:
            _fail(FAIL_CUSTODY, "PENDING marker differs from frozen trust checkpoint")

        python_frame, rust_frame = actors.resume_pending_seed_split()
        require_full_split_response_agreement_v2(python_frame, rust_frame)
        transaction._fault("after_recovery_seed_split_frames")

        if transaction._state == "RESERVED":
            python_receipt = build_python_static_replay_receipt_v1(basis)
            static_control_plane, static_daemon_binding = (
                actors.static_replay_control_plane_v1()
            )
            rust_receipt = run_rust_static_replay_receipt_v1(
                basis,
                control_plane=static_control_plane,
                daemon_receipt_binding=static_daemon_binding,
            )
            parent = generate_parent_absence_audit_v1(REPOSITORY_ROOT)
            replay_parent_absence_audit_v1(parent, repository=REPOSITORY_ROOT)
            inputs = _build_gate_inputs_and_sign_v1(
                basis=basis,
                parent=parent,
                actor_report=actor_report,
                errata_report=errata_report,
                python_static_receipt=python_receipt,
                rust_static_receipt=rust_receipt,
                execution_binding_roots=implementation_roots,
                actors=actors,
                timestamp=timestamp,
                run_id=recovery.run_id,
                ledger_id=recovery.ledger_id,
                trust_id=trust_id,
                frozen_actor_trust=actor_trust,
                frozen_split_frames=(python_frame, rust_frame),
                fault_injector=transaction._fault,
            )
            replay_payload = serialize_gate_evidence_inputs_v1(inputs)
            prospective_promotion = promote_gate_evidence_v1(
                evaluate_gates_15_24_v1(inputs)
            )
            transaction.stage_and_prospectively_replay(
                replay_payload, prospective_promotion, replay=replay
            )
        elif transaction._state in {
            "STAGED_PROSPECTIVE_REPLAY_PASSED",
            "SEED_CUSTODY_VERIFIED",
        }:
            inputs, _expected_marker = transaction._load_and_verify_stage(replay=replay)
            if (
                inputs.python_split_frame != python_frame
                or inputs.rust_split_frame != rust_frame
            ):
                _fail(FAIL_CUSTODY, "durable stage differs from recovered split frames")
        else:
            _fail(FAIL_TRANSACTION_LOCK, "prestage recovery journal state is not recoverable")

        staged_seed_commitment = transaction._staged_seed_commitment
        if type(staged_seed_commitment) is not bytes or len(staged_seed_commitment) != 32:
            _fail(FAIL_CUSTODY, "durable staged seed commitment is absent")
        seed_verification_receipt = actors.verify_seed_custody_commitment_v1(
            staged_seed_commitment
        )
        transaction.record_seed_custody_verification_v1(
            seed_verification_receipt
        )
        seed_root = candidate_content_root(
            "SplitSeedCommitmentManifestV1", inputs.split_seed_commitment_fields
        )
        actual_marker = actors.complete_marker(seed_root)
        transaction.record_marker_complete(actual_marker, inputs.marker_snapshot)
        actors.authorize_actor_key_volume_destruction(actual_marker)
        destruction_authorized = True
        actors.destroy_actor_key_volumes_and_verify_absent()
        actors_absent = True
        transaction.record_actors_absent()
        transaction.publish(replay=replay)
        published_payload = json.loads(transaction.public_evidence_path.read_bytes())
        published_promotion = replay(published_payload)
        if _canonical_json(published_promotion) != transaction.public_promotion_path.read_bytes():
            _fail(FAIL_PUBLICATION, "recovered evidence/promotion replay differs")
        return published_payload, dict(published_promotion)
    except BaseException as original:
        if actors_started and not actors_absent:
            try:
                if destruction_authorized:
                    actors.destroy_actor_key_volumes_and_verify_absent()
                else:
                    actors.stop_for_recovery_and_verify_absent()
            except BaseException as cleanup:
                if isinstance(cleanup, FormalContainerExecutorError):
                    raise cleanup from original
                _fail(FAIL_CONTAINER, f"prestage recovery cleanup raised {type(cleanup).__name__}")
        raise
    finally:
        transaction.close_lock()


def continue_pre_stage_pending_recovery_v1(
    *,
    recovery: PendingCeremonyRecoveryV1,
    actors: "CeremonyActorsV1",
    replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
    fault_injector: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Public formal recovery boundary; only the sealed Docker backend enters."""

    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot recover formal evidence")
    if type(actors) is not DockerCeremonyActorsV1:
        _fail(
            FAIL_PREFLIGHT,
            "formal prestage recovery requires the sealed DockerCeremonyActorsV1 backend",
        )
    return _continue_pre_stage_pending_recovery_core_v1(
        recovery=recovery,
        actors=actors,
        replay=replay,
        fault_injector=fault_injector,
    )


def _continue_post_stage_transaction_recovery_core_v1(
    *,
    transaction: FormalCeremonyTransactionV1,
    actors: "CeremonyActorsV1",
    replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
) -> tuple[dict[str, object], dict[str, object]]:
    """Idempotently continue an exact rehydrated transaction to publication.

    No opaque ID, key, signature, seed, or formal evidence is created here.
    A PENDING post-stage marker may only reopen the existing purpose volumes,
    replay the already-completed seed calculators, and compare their frames to
    the durable public stage before performing the already-staged marker
    transition.  COMPLETE recovery only removes exact run-labelled private
    state (if any) and republishes exact staged bytes.
    """

    if (
        not isinstance(transaction, FormalCeremonyTransactionV1)
        or transaction.recovery_phase is None
        or transaction._lock_descriptor is None
    ):
        _fail(FAIL_TRANSACTION_LOCK, "post-stage transaction is not rehydrated and locked")
    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot recover formal publication")
    inputs, expected_marker = transaction._load_and_verify_stage(replay=replay)
    frozen_daemon_binding = (
        None
        if transaction._prestage_intent_fields is None
        else transaction._prestage_intent_fields.get(
            "live_actor_protocol_daemon_receipt_binding"
        )
    )
    if type(frozen_daemon_binding) is not bytes or len(frozen_daemon_binding) != 32:
        _fail(FAIL_TRANSACTION_LOCK, "post-stage frozen daemon binding is absent")
    marker = read_marker_snapshot_v1(
        transaction.custody_directory / "split_seed_instantiation.marker"
    )
    if marker.state == "PENDING":
        if transaction.recovery_phase != "STAGED_PENDING":
            _fail(FAIL_TRANSACTION_LOCK, "PENDING marker recovery phase differs")
        actors.prepare_post_stage_pending_recovery(transaction.run_id)
        if isinstance(actors, DockerCeremonyActorsV1):
            actors.validate_rust_replay_binding(
                build_formal_static_basis_v1(transaction.basis_commit)
            )
        started = False
        destruction_authorized = False
        actors_absent = False
        try:
            actors.start()
            started = True
            actors.validate_frozen_daemon_receipt_binding_v1(
                frozen_daemon_binding
            )
            public_key = actors.keygen(1)
            if (
                type(public_key) is not bytes
                or len(public_key) != 32
                or hashlib.sha256(public_key).digest()[:16] != marker.custodian_key_id
            ):
                _fail(FAIL_CUSTODY, "recovered purpose-1 key differs from staged marker")
            python_frame, rust_frame = actors.resume_post_stage_seed_split()
            require_full_split_response_agreement_v2(python_frame, rust_frame)
            if (
                python_frame != inputs.python_split_frame
                or rust_frame != inputs.rust_split_frame
            ):
                _fail(FAIL_CUSTODY, "recovered split frames differ from durable stage")
            staged_seed_commitment = transaction._staged_seed_commitment
            if (
                type(staged_seed_commitment) is not bytes
                or len(staged_seed_commitment) != 32
            ):
                _fail(FAIL_CUSTODY, "durable staged seed commitment is absent")
            seed_verification_receipt = actors.verify_seed_custody_commitment_v1(
                staged_seed_commitment
            )
            transaction.record_seed_custody_verification_v1(
                seed_verification_receipt
            )
            seed_root = expected_marker.seed_commitment_manifest_root
            assert seed_root is not None
            actual_marker = actors.complete_marker(seed_root)
            transaction.record_marker_complete(actual_marker, expected_marker)
            actors.authorize_actor_key_volume_destruction(actual_marker)
            destruction_authorized = True
            actors.destroy_actor_key_volumes_and_verify_absent()
            actors_absent = True
            transaction.record_actors_absent()
        except BaseException as original:
            if started and not actors_absent:
                try:
                    if destruction_authorized:
                        actors.destroy_actor_key_volumes_and_verify_absent()
                    else:
                        actors.stop_for_recovery_and_verify_absent()
                except BaseException as cleanup:
                    if isinstance(cleanup, FormalContainerExecutorError):
                        raise cleanup from original
                    _fail(FAIL_CONTAINER, f"post-stage actor cleanup raised {type(cleanup).__name__}")
            raise
    else:
        if marker != expected_marker:
            _fail(FAIL_CUSTODY, "recovery COMPLETE marker differs from durable stage")
        actors.validate_frozen_daemon_receipt_binding_v1(
            frozen_daemon_binding
        )
        staged_seed_commitment = transaction._staged_seed_commitment
        if type(staged_seed_commitment) is not bytes or len(staged_seed_commitment) != 32:
            _fail(FAIL_CUSTODY, "durable staged seed commitment is absent")
        seed_verification_receipt = actors.verify_seed_custody_commitment_v1(
            staged_seed_commitment
        )
        transaction.record_seed_custody_verification_v1(
            seed_verification_receipt
        )
        transaction.record_marker_complete(marker, expected_marker)
        actors.recover_complete_actor_key_volumes_and_verify_absent(
            transaction.run_id, marker
        )
        transaction.record_actors_absent()

    transaction.publish(replay=replay)
    published_payload = json.loads(transaction.public_evidence_path.read_bytes())
    published_promotion = replay(published_payload)
    if (
        _canonical_json(published_promotion)
        != transaction.public_promotion_path.read_bytes()
        or transaction.public_evidence_path.read_bytes()
        != transaction._staged_payloads["evidence"]
        or transaction.publication_receipt_path.read_bytes()
        != transaction._staged_payloads["receipt"]
    ):
        _fail(FAIL_PUBLICATION, "recovered final publication differs from durable stage")
    return published_payload, dict(published_promotion)


def continue_post_stage_transaction_recovery_v1(
    *,
    transaction: FormalCeremonyTransactionV1,
    actors: "CeremonyActorsV1",
    replay: Callable[[Mapping[str, object]], Mapping[str, object]] = replay_public_gate_evidence_v1,
) -> tuple[dict[str, object], dict[str, object]]:
    """Public post-stage boundary; signing remains forbidden and Docker-sealed."""

    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot recover formal publication")
    if type(actors) is not DockerCeremonyActorsV1:
        _fail(
            FAIL_PREFLIGHT,
            "formal post-stage recovery requires the sealed DockerCeremonyActorsV1 backend",
        )
    return _continue_post_stage_transaction_recovery_core_v1(
        transaction=transaction,
        actors=actors,
        replay=replay,
    )


def _bundle_rows(envelopes: Sequence[tuple[int, Mapping[str, object]]]) -> dict[str, object]:
    rows = sorted(
        (
            purpose,
            envelope["enclosed_manifest_root"],
            candidate_content_root("SignedManifestEnvelopeV1", envelope),
        )
        for purpose, envelope in envelopes
    )
    return {"attestations": tuple(rows)}


def _opaque_registry(
    *, run_id: bytes, ledger_id: bytes, seed_root: bytes, trust_root: bytes,
    timestamp: int, commit_wire: tuple[int, bytes]
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    intents = (
        {
            "opaque_id_kind_id": 1,
            "opaque_id_16_bytes": run_id,
            "registration_context_root": trust_root,
            "created_at_unix_seconds": timestamp,
            "repository_commit_id": commit_wire,
        },
        {
            "opaque_id_kind_id": 2,
            "opaque_id_16_bytes": ledger_id,
            "registration_context_root": seed_root,
            "created_at_unix_seconds": timestamp,
            "repository_commit_id": commit_wire,
        },
    )
    records = tuple(
        {
            "registry_sequence_number": index,
            "opaque_id_kind_id": intent["opaque_id_kind_id"],
            "opaque_id_16_bytes": intent["opaque_id_16_bytes"],
            "first_seen_object_root": candidate_content_root("OpaqueIdRegistrationIntentV1", intent),
            "first_seen_repository_commit_id": commit_wire,
            "created_at_unix_seconds": timestamp,
        }
        for index, intent in enumerate(intents)
    )
    first_tree = candidate_record_tree_root("OpaqueIdRegistryRecordV1", records[:1])
    first = {
        "previous_snapshot_root_or_null": None,
        "registry_tree_root": first_tree,
        "record_count": 1,
        "added_record_root": first_tree,
        "repository_commit_id": commit_wire,
    }
    second_tree = candidate_record_tree_root("OpaqueIdRegistryRecordV1", records)
    last_single = rfc6962_root([encode_formal_object("OpaqueIdRegistryRecordV1", records[1])])
    second = {
        "previous_snapshot_root_or_null": candidate_content_root("OpaqueIdRegistrySnapshotV1", first),
        "registry_tree_root": second_tree,
        "record_count": 2,
        "added_record_root": last_single,
        "repository_commit_id": commit_wire,
    }
    return intents, records, (first, second)


@dataclass(frozen=True, slots=True)
class ActorPublicKeysV1:
    public_keys: Mapping[int, bytes]
    key_ids: Mapping[int, bytes]
    manifests: tuple[Mapping[str, object], ...]
    replacement_policy_fields: Mapping[str, object]
    trust_genesis_fields: Mapping[str, object]
    trust_genesis_root: bytes


def build_actor_trust_v1(
    *, public_keys: Mapping[int, bytes], timestamp: int, basis_commit: str,
    trust_genesis_id: bytes,
) -> ActorPublicKeysV1:
    if set(public_keys) != {1, 2, 3, 4} or len(set(public_keys.values())) != 4:
        _fail(FAIL_PREFLIGHT, "four distinct actor public keys are required")
    manifests = tuple(
        build_actor_key_manifest_fields_v1(
            purpose_id=purpose,
            public_key=public_keys[purpose],
            created_at_unix_seconds=timestamp,
            basis_commit=basis_commit,
        )
        for purpose in (1, 2, 3, 4)
    )
    replacement = {
        "key_rotation_threshold": 2,
        "key_revocation_threshold": 2,
        "custodian_replacement_requires_new_seed_version": True,
        "actor_key_reuse_across_purposes_allowed": False,
        "secret_material_export_allowed": False,
    }
    replacement_root = candidate_content_root("ReplacementPolicyV1", replacement)
    trust = {
        "trust_genesis_id_16_bytes": trust_genesis_id,
        "purpose_key_entries": tuple(
            (purpose, candidate_content_root("ActorKeyManifestV1", manifest))
            for purpose, manifest in zip((1, 2, 3, 4), manifests, strict=True)
        ),
        "purpose_key_policy_root": replacement_root,
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": git_sha1_commit_id(bytes.fromhex(basis_commit)),
    }
    trust_root = candidate_content_root("ActorTrustGenesisV1", trust)
    return ActorPublicKeysV1(
        MappingProxyType(dict(public_keys)),
        MappingProxyType({purpose: manifest["key_id"] for purpose, manifest in zip((1,2,3,4), manifests, strict=True)}),
        manifests,
        MappingProxyType(replacement),
        MappingProxyType(trust),
        trust_root,
    )


def build_actor_trust_checkpoint_fields_v1(
    *,
    actor_trust: ActorPublicKeysV1,
    basis_commit: str,
    run_id: bytes,
    ledger_id: bytes,
    prestage_intent_sha256: str,
) -> dict[str, object]:
    """Build the public, crash-stable identity of all four role keys."""

    if (
        re.fullmatch(r"[0-9a-f]{64}", prestage_intent_sha256) is None
        or set(actor_trust.public_keys) != {1, 2, 3, 4}
        or set(actor_trust.key_ids) != {1, 2, 3, 4}
    ):
        _fail(FAIL_TRANSACTION_LOCK, "actor-trust checkpoint inputs are invalid")
    purpose_rows = tuple(
        {
            "purpose_id": purpose,
            "public_key_32_bytes": actor_trust.public_keys[purpose],
            "key_id_16_bytes": actor_trust.key_ids[purpose],
        }
        for purpose in (1, 2, 3, 4)
    )
    fields: dict[str, object] = {
        "schema": ACTOR_TRUST_CHECKPOINT_SCHEMA,
        "basis_commit": _commit(basis_commit),
        "run_id_hex": run_id.hex(),
        "ledger_id_hex": ledger_id.hex(),
        "prestage_intent_sha256": prestage_intent_sha256,
        "purpose_key_rows": purpose_rows,
        "actor_key_manifests": actor_trust.manifests,
        "replacement_policy_fields": actor_trust.replacement_policy_fields,
        "trust_genesis_fields": actor_trust.trust_genesis_fields,
        "actor_trust_genesis_root": actor_trust.trust_genesis_root,
        "all_four_key_identities_frozen_before_seed": True,
        "contains_private_key": False,
    }
    assert_public_payload_contains_no_secret_fields(fields)
    return fields


def validate_actor_trust_checkpoint_fields_v1(
    fields: Mapping[str, object],
    *,
    expected_actor_trust: ActorPublicKeysV1,
    basis_commit: str,
    run_id: bytes,
    ledger_id: bytes,
    prestage_intent_sha256: str,
) -> dict[str, object]:
    """Require byte-equivalence to the trust identity derived from role volumes."""

    restored = _restore(dict(fields))
    if type(restored) is not dict:
        _fail(FAIL_CUSTODY, "actor-trust checkpoint transport is invalid")
    expected = build_actor_trust_checkpoint_fields_v1(
        actor_trust=expected_actor_trust,
        basis_commit=basis_commit,
        run_id=run_id,
        ledger_id=ledger_id,
        prestage_intent_sha256=prestage_intent_sha256,
    )
    if restored != expected or _canonical_json(restored) != _canonical_json(expected):
        _fail(FAIL_CUSTODY, "actor-trust checkpoint differs from recovered role keys")
    return restored


class CeremonyActorsV1:
    """Purpose actor interface; real implementation is container-only."""

    authoritative: bool = False

    def unresolved_formal_blockers(self) -> tuple[str, ...]:
        return UNRESOLVED_AUTHORITATIVE_BLOCKERS

    def bridge_qualification_report_id_v1(self) -> bytes | None:
        """Return the exact 32-byte public qualification report identity."""

        return None

    def prestage_runtime_binding_fields_v1(
        self, execution_binding_roots: Mapping[str, bytes]
    ) -> Mapping[str, object] | None:
        return None

    def validate_frozen_daemon_receipt_binding_v1(
        self, expected: bytes
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def static_replay_control_plane_v1(
        self,
    ) -> tuple[LocalDockerControlPlaneV1, bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare_pending_recovery(self, run_id: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def reclaim_pending_custody_from_anchor_v1(
        self, anchor_fields: Mapping[str, object]
    ) -> Mapping[str, object]:  # pragma: no cover
        raise NotImplementedError

    def recover_preseed_private_state_and_verify_absent(
        self, run_id: bytes
    ) -> Mapping[str, object]:  # pragma: no cover
        raise NotImplementedError

    def start(self) -> "CeremonyActorsV1":  # pragma: no cover - interface
        return self

    def keygen(self, purpose: int) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def seed_split(self) -> tuple[bytes, bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    def verify_seed_custody_commitment_v1(
        self, expected_commitment: bytes
    ) -> Mapping[str, object]:  # pragma: no cover - interface
        raise NotImplementedError

    def resume_pending_seed_split(self) -> tuple[bytes, bytes]:  # pragma: no cover
        raise NotImplementedError

    def prepare_post_stage_pending_recovery(self, run_id: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    def resume_post_stage_seed_split(self) -> tuple[bytes, bytes]:  # pragma: no cover
        raise NotImplementedError

    def recover_complete_private_state_and_verify_absent(
        self, run_id: bytes, marker: MarkerSnapshot
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def recover_complete_actor_key_volumes_and_verify_absent(
        self, run_id: bytes, marker: MarkerSnapshot
    ) -> None:  # pragma: no cover
        # Compatibility bridge for test/future backends that implemented the
        # old over-broad name.  The canonical API names exactly what is erased.
        self.recover_complete_private_state_and_verify_absent(run_id, marker)

    def prospective_complete_marker(
        self, seed_manifest_root: bytes
    ) -> MarkerSnapshot:  # pragma: no cover - interface
        raise NotImplementedError

    def complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:  # pragma: no cover
        raise NotImplementedError

    def close_and_verify_absent(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop_for_recovery_and_verify_absent(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def destroy_private_state_and_verify_absent(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def destroy_actor_key_volumes_and_verify_absent(self) -> None:  # pragma: no cover
        self.destroy_private_state_and_verify_absent()

    def authorize_private_state_destruction(
        self, marker: MarkerSnapshot
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def authorize_actor_key_volume_destruction(
        self, marker: MarkerSnapshot
    ) -> None:  # pragma: no cover - interface
        self.authorize_private_state_destruction(marker)

    def sign_object(self, name: str, fields: Mapping[str, object]) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def sign_parent(self, evidence: ParentAbsenceAuditEvidence, fields: Mapping[str, object]) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def sign_bridge(
        self,
        purpose: int,
        fields: Mapping[str, object],
        replay_package: bytes,
    ) -> bytes:  # pragma: no cover
        raise NotImplementedError


def _renameat2_noreplace_v1(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    """Linux ``renameat2(RENAME_NOREPLACE)`` with no weaker fallback."""

    if (
        os.name != "posix"
        or not hasattr(os, "uname")
        or os.uname().sysname != "Linux"
        or type(source_parent_fd) is not int
        or type(destination_parent_fd) is not int
        or type(source_name) is not str
        or type(destination_name) is not str
        or not source_name
        or not destination_name
        or "/" in source_name
        or "/" in destination_name
        or "\0" in source_name
        or "\0" in destination_name
    ):
        raise OSError(
            errno.ENOSYS,
            "purpose-4 atomic no-replace adoption requires Linux renameat2",
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(
            errno.ENOSYS,
            "purpose-4 atomic no-replace adoption has no renameat2 runtime",
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE from linux/fs.h.
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(
            error_number,
            "purpose-4 atomic no-replace snapshot adoption failed: "
            + os.strerror(error_number),
        )


def _record_cleanup_error_v1(
    primary: BaseException,
    label: str,
    cleanup: BaseException,
) -> None:
    """Attach cleanup diagnostics without replacing the primary exception."""

    existing = getattr(primary, "_hegel_cleanup_error_chain", ())
    setattr(
        primary,
        "_hegel_cleanup_error_chain",
        (*existing, (label, type(cleanup).__name__, str(cleanup))),
    )


def _require_exact_owned_directory_identity_v1(
    path: Path,
    expected_identity: tuple[int, int],
) -> os.stat_result:
    """Return nofollow metadata only for the exact caller-owned directory."""

    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise OSError("purpose-4 cleanup directory identity differs")
    return metadata


def _remove_exact_owned_purpose4_tree_v1(
    path: Path,
    expected_identity: tuple[int, int],
    *,
    record_quarantine_path: Callable[[Path], None] | None = None,
) -> None:
    """Atomically claim and descriptor-purge only the exact directory inode."""

    def purge_directory_fd(directory_fd: int) -> None:
        opened = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
        ):
            raise OSError("purpose-4 cleanup opened a non-owned directory")
        os.fchmod(directory_fd, 0o700)
        for name in sorted(os.listdir(directory_fd)):
            if not name or name in {".", ".."} or "/" in name or "\0" in name:
                raise OSError("purpose-4 cleanup entry name is invalid")
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            identity = entry.st_dev, entry.st_ino
            if stat.S_ISDIR(entry.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_fd,
                )
                try:
                    child_open = os.fstat(child_fd)
                    if (
                        (child_open.st_dev, child_open.st_ino) != identity
                        or not stat.S_ISDIR(child_open.st_mode)
                    ):
                        raise OSError(
                            "purpose-4 cleanup child directory identity changed"
                        )
                    purge_directory_fd(child_fd)
                    child_path = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        (child_path.st_dev, child_path.st_ino) != identity
                        or not stat.S_ISDIR(child_path.st_mode)
                    ):
                        raise OSError(
                            "purpose-4 cleanup child path identity changed"
                        )
                    os.rmdir(name, dir_fd=directory_fd)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry.st_mode):
                current = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    (current.st_dev, current.st_ino) != identity
                    or not stat.S_ISREG(current.st_mode)
                ):
                    raise OSError("purpose-4 cleanup file identity changed")
                os.unlink(name, dir_fd=directory_fd)
            else:
                # In particular, never chmod, open through, or unlink a
                # symlink introduced into the cleanup quarantine.
                raise OSError("purpose-4 cleanup refuses a non-regular entry")
        os.fsync(directory_fd)

    parent = path.parent
    if (
        not path.name
        or path.name in {".", ".."}
        or parent.resolve(strict=True) != parent
    ):
        raise OSError("purpose-4 cleanup path is not canonical")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    primary_error: BaseException | None = None
    removal_completed = False
    try:
        parent_fd = os.open(parent, flags)
        descriptors.append(parent_fd)
        root_fd = os.open(path.name, flags, dir_fd=parent_fd)
        descriptors.append(root_fd)
        parent_path = parent.lstat()
        parent_open = os.fstat(parent_fd)
        root_path = path.lstat()
        root_open = os.fstat(root_fd)
        if (
            stat.S_ISLNK(parent_path.st_mode)
            or not stat.S_ISDIR(parent_path.st_mode)
            or not stat.S_ISDIR(parent_open.st_mode)
            or stat.S_ISLNK(root_path.st_mode)
            or not stat.S_ISDIR(root_path.st_mode)
            or not stat.S_ISDIR(root_open.st_mode)
            or parent_open.st_uid != os.geteuid()
            or root_open.st_uid != os.geteuid()
            or (parent_path.st_dev, parent_path.st_ino)
            != (parent_open.st_dev, parent_open.st_ino)
            or (root_path.st_dev, root_path.st_ino) != expected_identity
            or (root_open.st_dev, root_open.st_ino) != expected_identity
            or root_open.st_dev != parent_open.st_dev
        ):
            raise OSError("purpose-4 cleanup root identity differs")

        quarantine_name = ".hegel-purpose4-cleanup-" + secrets.token_hex(16)
        _renameat2_noreplace_v1(
            parent_fd,
            path.name,
            parent_fd,
            quarantine_name,
        )
        quarantined = os.stat(
            quarantine_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            (quarantined.st_dev, quarantined.st_ino) != expected_identity
            or not stat.S_ISDIR(quarantined.st_mode)
        ):
            restore_error: BaseException | None = None
            try:
                _renameat2_noreplace_v1(
                    parent_fd,
                    quarantine_name,
                    parent_fd,
                    path.name,
                )
            except BaseException as exc:
                restore_error = exc
            mismatch = OSError(
                "purpose-4 cleanup quarantine captured a replacement inode"
            )
            if restore_error is not None:
                _record_cleanup_error_v1(
                    mismatch,
                    "replacement-restore",
                    restore_error,
                )
                raise mismatch from restore_error
            raise mismatch

        quarantine = parent / quarantine_name
        if record_quarantine_path is not None:
            record_quarantine_path(quarantine)
        purge_directory_fd(root_fd)
        final = os.stat(
            quarantine_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            (final.st_dev, final.st_ino) != expected_identity
            or not stat.S_ISDIR(final.st_mode)
        ):
            raise OSError("purpose-4 cleanup quarantine identity changed")
        os.rmdir(quarantine_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        removal_completed = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_failures: list[OSError] = []
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError as exc:
                close_failures.append(exc)
        if close_failures:
            detail = ",".join(type(exc).__name__ for exc in close_failures)
            if primary_error is not None:
                for close_failure in close_failures:
                    _record_cleanup_error_v1(
                        primary_error,
                        "cleanup-descriptor-close",
                        close_failure,
                    )
            elif not removal_completed:
                raise OSError(
                    "purpose-4 cleanup descriptor close failed: " + detail
                ) from close_failures[0]


def _adopt_read_only_purpose4_snapshot_v1(
    snapshot: DetachedParentSnapshotV1,
    destination: Path,
    *,
    expected_basis_commit: str,
    record_adopted_identity: Callable[[tuple[int, int]], None] | None = None,
) -> None:
    """Atomically adopt one frozen snapshot without copying its object store.

    Linux requires write permission on a directory moved between different
    parents because its ``..`` entry changes.  The detached builder freezes
    the snapshot root at 0555, so the adoption grants owner-write to that one
    held inode only for ``renameat2(RENAME_NOREPLACE)``.  All descendants
    remain read-only.  Held directory descriptors bind every identity check,
    the source and destination must be on one filesystem, and the complete
    detached manifest is replayed after the root has been restored to 0555.
    """

    source = snapshot.root
    source_parent = source.parent
    destination_parent = destination.parent
    if (
        not source.name
        or not destination.name
        or source.name in {".", ".."}
        or destination.name in {".", ".."}
        or source_parent.resolve(strict=True) != source_parent
        or destination_parent.resolve(strict=True) != destination_parent
    ):
        raise OSError("purpose-4 snapshot adoption path is not canonical")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    primary_error: BaseException | None = None
    try:
        source_parent_fd = os.open(source_parent, flags)
        descriptors.append(source_parent_fd)
        destination_parent_fd = os.open(destination_parent, flags)
        descriptors.append(destination_parent_fd)
        source_fd = os.open(source.name, flags, dir_fd=source_parent_fd)
        descriptors.append(source_fd)

        source_parent_path = source_parent.lstat()
        destination_parent_path = destination_parent.lstat()
        source_path = source.lstat()
        source_parent_open = os.fstat(source_parent_fd)
        destination_parent_open = os.fstat(destination_parent_fd)
        source_open = os.fstat(source_fd)
        if any(
            not stat.S_ISDIR(item.st_mode)
            for item in (
                source_parent_path,
                destination_parent_path,
                source_path,
                source_parent_open,
                destination_parent_open,
                source_open,
            )
        ):
            raise OSError("purpose-4 snapshot adoption paths must be directories")
        if (
            (source_parent_path.st_dev, source_parent_path.st_ino)
            != (source_parent_open.st_dev, source_parent_open.st_ino)
            or (destination_parent_path.st_dev, destination_parent_path.st_ino)
            != (destination_parent_open.st_dev, destination_parent_open.st_ino)
            or (source_path.st_dev, source_path.st_ino)
            != (source_open.st_dev, source_open.st_ino)
        ):
            raise OSError("purpose-4 snapshot adoption path identity changed")
        if (
            source_open.st_uid != os.geteuid()
            or source_parent_open.st_uid != os.geteuid()
            or destination_parent_open.st_uid != os.geteuid()
            or stat.S_IMODE(source_open.st_mode) != 0o555
            or stat.S_IMODE(source_parent_open.st_mode) & 0o300 != 0o300
            or stat.S_IMODE(destination_parent_open.st_mode) & 0o300 != 0o300
        ):
            raise OSError(
                "purpose-4 snapshot adoption requires owned writable parents "
                "and an exact mode-0555 source root"
            )
        if len({
            source_open.st_dev,
            source_parent_open.st_dev,
            destination_parent_open.st_dev,
        }) != 1:
            raise OSError("purpose-4 snapshot adoption crossed a filesystem")
        try:
            os.stat(
                destination.name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise OSError("purpose-4 snapshot adoption destination already exists")

        os.fchmod(source_fd, 0o755)
        adoption_error: BaseException | None = None
        try:
            if stat.S_IMODE(os.fstat(source_fd).st_mode) != 0o755:
                raise OSError("purpose-4 snapshot root did not enter adoption mode")
            _renameat2_noreplace_v1(
                source_parent_fd,
                source.name,
                destination_parent_fd,
                destination.name,
            )
            # From this instruction onward every failure path must clean the
            # adopted destination, not the now-absent temporary source path.
            snapshot.root = destination
            if record_adopted_identity is not None:
                record_adopted_identity((source_open.st_dev, source_open.st_ino))
        except BaseException as exc:
            adoption_error = exc
        try:
            os.fchmod(source_fd, 0o555)
        except BaseException as restore_error:
            if adoption_error is not None:
                _record_cleanup_error_v1(
                    adoption_error,
                    "snapshot-root-mode-restore",
                    restore_error,
                )
                raise adoption_error from restore_error
            raise
        if adoption_error is not None:
            raise adoption_error

        adopted_open = os.fstat(source_fd)
        adopted_path = os.stat(
            destination.name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        try:
            os.stat(source.name, dir_fd=source_parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OSError("purpose-4 snapshot source name survived adoption")
        if (
            (adopted_open.st_dev, adopted_open.st_ino)
            != (source_open.st_dev, source_open.st_ino)
            or (adopted_path.st_dev, adopted_path.st_ino)
            != (source_open.st_dev, source_open.st_ino)
            or not stat.S_ISDIR(adopted_path.st_mode)
            or stat.S_IMODE(adopted_open.st_mode) != 0o555
            or stat.S_IMODE(adopted_path.st_mode) != 0o555
        ):
            raise OSError("purpose-4 snapshot adoption identity or mode differs")
        os.fsync(source_fd)
        os.fsync(source_parent_fd)
        if (
            destination_parent_open.st_dev,
            destination_parent_open.st_ino,
        ) != (
            source_parent_open.st_dev,
            source_parent_open.st_ino,
        ):
            os.fsync(destination_parent_fd)

        replayed = validate_detached_parent_snapshot_v1(
            destination,
            snapshot.manifest,
            git_executable=snapshot.git_executable,
            require_frozen_parent=True,
            expected_basis_commit=expected_basis_commit,
        )
        if dict(replayed) != dict(snapshot.manifest):
            raise OSError("purpose-4 adopted snapshot manifest differs")
        final_path = os.stat(
            destination.name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        final_open = os.fstat(source_fd)
        if (
            (final_path.st_dev, final_path.st_ino)
            != (source_open.st_dev, source_open.st_ino)
            or (final_open.st_dev, final_open.st_ino)
            != (source_open.st_dev, source_open.st_ino)
            or not stat.S_ISDIR(final_path.st_mode)
            or stat.S_IMODE(final_path.st_mode) != 0o555
            or stat.S_IMODE(final_open.st_mode) != 0o555
        ):
            raise OSError("purpose-4 adopted snapshot changed during replay")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close_failure: OSError | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError as exc:
                if close_failure is None:
                    close_failure = exc
        if close_failure is not None:
            if primary_error is not None:
                _record_cleanup_error_v1(
                    primary_error,
                    "snapshot-descriptor-close",
                    close_failure,
                )
            else:
                raise close_failure


class DockerCeremonyActorsV1(CeremonyActorsV1, AbstractContextManager["DockerCeremonyActorsV1"]):
    """Four simultaneously-live, offline, digest-pinned purpose containers."""

    authoritative = True

    def __init__(
        self,
        *,
        basis_commit: str,
        custody_directory: Path,
        rust_formal_replay_binary: Path,
        timestamp: int,
        rust_bridge_dag_replay_binary: Path = DEFAULT_RUST_BRIDGE_DAG_BINARY,
        rust_bridge_dag_qualification_report: Path = (
            DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT
        ),
    ) -> None:
        self.basis_commit = _commit(basis_commit)
        self._custody_requested = Path(custody_directory)
        self.custody_directory = custody_directory.resolve()
        self.rust_formal_replay_binary = rust_formal_replay_binary.resolve()
        self.rust_bridge_dag_replay_binary = Path(
            rust_bridge_dag_replay_binary
        ).resolve()
        self.rust_bridge_dag_qualification_report = Path(
            rust_bridge_dag_qualification_report
        ).resolve()
        self.timestamp = timestamp
        self._temporary: LinuxLocalTemporaryDirectoryV1 | None = None
        self._root: Path | None = None
        self._docker_control_plane: LocalDockerControlPlaneV1 | None = None
        self._docker_daemon_receipt: Mapping[str, object] | None = None
        self._docker_daemon_binding: bytes | None = None
        self._docker_root_directory: Path | None = None
        self._runtime_seccomp_path: Path | None = None
        self._build_seccomp_path: Path | None = None
        self._containers: dict[int, str] = {}
        self._container_names: dict[int, str] = {}
        self._public_keys: dict[int, bytes] = {}
        self._key_ids: dict[int, bytes] = {}
        self._profile: Mapping[str, object] = {}
        self._pending_marker: MarkerSnapshot | None = None
        self._ceremony_token: str | None = None
        self._custody_handed_off = False
        self._bound_rust_replay_digest: bytes | None = None
        self._bound_rust_bridge_dag_digest: bytes | None = None
        self._bound_rust_bridge_dag_report_sha256: bytes | None = None
        self._transaction_run_id: bytes | None = None
        self._state_volumes: dict[int, str] = {}
        self._marker_completed_after_staging = False
        self._profile_digest: bytes | None = None
        self._recovery_mode = False
        self._volume_initialization_receipts: dict[int, Mapping[str, object]] = {}
        self._live_actor_probe_receipts: dict[int, Mapping[str, object]] = {}
        self._operation_probe_receipts: list[Mapping[str, object]] = []
        self._operation_sequences: dict[int, int] = {purpose: 0 for purpose in (1, 2, 3, 4)}
        self._live_actor_set_qualified = False
        self._custody_durability_receipt: Mapping[str, object] | None = None
        self._custody_retention_receipt: Mapping[str, object] | None = None
        self._purpose4_snapshot_manifest: Mapping[str, object] | None = None
        self._purpose4_runtime_bundle: Mapping[str, object] | None = None
        self._purpose4_snapshot_path: Path | None = None
        self._purpose4_snapshot_identity: tuple[int, int] | None = None
        self._purpose4_snapshot_owner: DetachedParentSnapshotV1 | None = None
        self._purpose4_snapshot_tree_removed = False
        self._purpose4_runtime_path: Path | None = None
        self._purpose4_runtime_identity: tuple[int, int] | None = None
        self._purpose4_foreign_entries: dict[
            Path, tuple[int, int] | None
        ] = {}
        self._purpose4_vacated_paths: set[Path] = set()

    def _refresh_purpose4_foreign_entries_v1(self) -> None:
        """Record every object that reappears at a vacated cleanup name."""

        for path in self._purpose4_vacated_paths:
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                identity = None
            else:
                identity = metadata.st_dev, metadata.st_ino
            self._purpose4_foreign_entries[path] = identity

    def unresolved_formal_blockers(self) -> tuple[str, ...]:
        blockers = list(UNRESOLVED_AUTHORITATIVE_BLOCKERS)
        if (
            self._bound_rust_bridge_dag_digest is None
            or self._bound_rust_bridge_dag_report_sha256 is None
        ):
            blockers.append(FAIL_BRIDGE_REPLAY_UNRESOLVED)
        return tuple(blockers)

    def bridge_qualification_report_id_v1(self) -> bytes | None:
        return self._bound_rust_bridge_dag_report_sha256

    def prestage_runtime_binding_fields_v1(
        self, execution_binding_roots: Mapping[str, bytes]
    ) -> Mapping[str, object] | None:
        if (
            self._bound_rust_replay_digest is None
            or self._bound_rust_bridge_dag_digest is None
            or self._bound_rust_bridge_dag_report_sha256 is None
        ):
            return None
        try:
            formal_path = self.rust_formal_replay_binary.resolve(strict=True)
            bridge_path = self.rust_bridge_dag_replay_binary.resolve(strict=True)
            profile_digest = hashlib.sha256(
                self._git_blob(
                    "Hegel Machine/config/phase3_container_actor_profile_v1.json"
                )
            ).digest()
        except OSError:
            return None
        return MappingProxyType(
            _validate_prestage_runtime_bindings_v1({
                "m3_execution_implementation_binding_roots": dict(
                    execution_binding_roots
                ),
                "formal_rust_replay_binary_path": formal_path.as_posix(),
                "formal_rust_replay_binary_sha256": self._bound_rust_replay_digest,
                "rust_bridge_dag_replay_binary_path": bridge_path.as_posix(),
                "rust_bridge_dag_replay_binary_sha256": (
                    self._bound_rust_bridge_dag_digest
                ),
                "rust_bridge_dag_qualification_report_sha256": (
                    self._bound_rust_bridge_dag_report_sha256
                ),
                "actor_profile_sha256": profile_digest,
            })
        )

    def validate_frozen_daemon_receipt_binding_v1(self, expected: bytes) -> None:
        if type(expected) is not bytes or len(expected) != 32:
            _fail(FAIL_PREFLIGHT, "frozen live-qualification daemon binding is invalid")
        self._ensure_local_runtime()
        if self._docker_daemon_binding != expected:
            _fail(
                FAIL_PREFLIGHT,
                "formal actor daemon differs from live-qualification daemon",
            )

    def static_replay_control_plane_v1(
        self,
    ) -> tuple[LocalDockerControlPlaneV1, bytes]:
        self._ensure_local_runtime()
        if self._docker_control_plane is None or self._docker_daemon_binding is None:
            _fail(FAIL_PREFLIGHT, "formal static replay control plane is absent")
        return self._docker_control_plane, self._docker_daemon_binding

    @property
    def operation_sequence_scope(self) -> str:
        """Sequences are monotonic only within one live-container incarnation.

        Recovery intentionally creates a new diagnostic epoch and may restart
        sequence numbers and nonces.  Formal identities never depend on these
        host-operation diagnostics.
        """

        return "LIVE_CONTAINER_INCARNATION_ONLY"

    def bind_transaction_identity(self, run_id: bytes) -> None:
        if type(run_id) is not bytes or len(run_id) != 16:
            _fail(FAIL_PREFLIGHT, "Docker actor transaction run ID must be 16 bytes")
        if self._transaction_run_id is not None and self._transaction_run_id != run_id:
            _fail(FAIL_PREFLIGHT, "Docker actor transaction identity cannot be rebound")
        self._transaction_run_id = run_id

    def prepare_pending_recovery(self, run_id: bytes) -> None:
        self.bind_transaction_identity(run_id)
        self._recovery_mode = True

    def reclaim_pending_custody_from_anchor_v1(
        self, anchor_fields: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Reclaim one anchor-bound 65534-owned tree without reading seed bytes."""

        if (
            anchor_fields.get("schema") != _RECOVERY_ANCHOR_SCHEMA
            or anchor_fields.get("basis_commit") != self.basis_commit
            or anchor_fields.get("custody_absolute_path")
            != str(self.custody_directory)
            or anchor_fields.get("custody_owner_handoff_uid") != 65534
            or anchor_fields.get("custody_owner_handoff_gid") != 65534
            or anchor_fields.get("owner_reclaim_policy_id")
            != "PINNED_OFFLINE_CAP_CHOWN_ONLY_V1"
        ):
            _fail(FAIL_CUSTODY, "pending custody reclaim anchor differs")
        self._ensure_local_runtime()
        self._load_committed_profile()
        if self._profile_digest is None:
            _fail(FAIL_CONTAINER, "pending custody reclaim profile digest is absent")
        if (
            anchor_fields.get("actor_profile_sha256")
            != self._profile_digest.hex()
            or anchor_fields.get("docker_daemon_receipt_sha256")
            != self._docker_daemon_binding.hex()
        ):
            _fail(FAIL_CONTAINER, "pending custody reclaim runtime binding differs")
        assert self._root is not None
        if self._runtime_seccomp_path is None:
            control = self._root / "control"
            control.mkdir(mode=0o700, exist_ok=True)
            self._runtime_seccomp_path = control / SECCOMP_PATH.name
            self._write_snapshot(
                self._runtime_seccomp_path,
                self._git_blob(
                    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"
                ),
                0o444,
            )
        try:
            before = self.custody_directory.lstat()
        except OSError as exc:
            _fail(FAIL_CUSTODY, f"pending custody reclaim inode is absent: {exc}")
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o700
            or before.st_uid != 65534
            or before.st_gid != 65534
            or anchor_fields.get("custody_st_dev") != before.st_dev
            or anchor_fields.get("custody_st_ino") != before.st_ino
        ):
            _fail(FAIL_CUSTODY, "pending custody reclaim source inode differs")
        self._custody_handed_off = True
        self._reclaim_custody_from_actor()
        after = self.custody_directory.lstat()
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (after.st_uid, after.st_gid) != (os.geteuid(), os.getegid())
            or stat.S_IMODE(after.st_mode) != 0o700
        ):
            _fail(FAIL_CUSTODY, "pending custody reclaim result differs")
        receipt: dict[str, object] = {
            "schema": "hegel-phase3-m25-pending-custody-reclaim/1",
            "basis_commit": self.basis_commit,
            "run_id_hex": anchor_fields.get("run_id_hex"),
            "custody_st_dev": after.st_dev,
            "custody_st_ino": after.st_ino,
            "source_owner_uid": 65534,
            "source_owner_gid": 65534,
            "destination_owner_uid": after.st_uid,
            "destination_owner_gid": after.st_gid,
            "docker_daemon_receipt_sha256": self._docker_daemon_binding.hex(),
            "actor_profile_sha256": self._profile_digest.hex(),
            "network_mode_none": True,
            "ipc_private": True,
            "cap_drop_all_then_add_chown_only": True,
            "private_key_volume_mounted": False,
            "raw_seed_bytes_read": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json(receipt)
        ).hexdigest()
        return MappingProxyType(receipt)

    def prepare_post_stage_pending_recovery(self, run_id: bytes) -> None:
        self.prepare_pending_recovery(run_id)

    @property
    def volume_initialization_receipts(self) -> Mapping[int, Mapping[str, object]]:
        return MappingProxyType(dict(self._volume_initialization_receipts))

    @property
    def docker_daemon_receipt(self) -> Mapping[str, object] | None:
        return (
            None
            if self._docker_daemon_receipt is None
            else MappingProxyType(dict(self._docker_daemon_receipt))
        )

    @property
    def live_actor_probe_receipts(self) -> Mapping[int, Mapping[str, object]]:
        return MappingProxyType(dict(self._live_actor_probe_receipts))

    @property
    def operation_probe_receipts(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self._operation_probe_receipts)

    @property
    def custody_durability_receipt(self) -> Mapping[str, object] | None:
        return self._custody_durability_receipt

    @property
    def custody_retention_receipt(self) -> Mapping[str, object] | None:
        return self._custody_retention_receipt

    def _ensure_local_runtime(self) -> None:
        if self._docker_control_plane is not None:
            return
        try:
            temporary = LinuxLocalTemporaryDirectoryV1(
                prefix="hegel-m25-formal-",
                repository_root=REPOSITORY_ROOT,
            )
            root = Path(temporary.name)
            control_plane = prepare_local_docker_control_plane_v1(
                root,
                repository_root=REPOSITORY_ROOT,
            )
            version = _run(
                control_plane.command("version", "--format", "{{json .}}"),
                environment=control_plane.environment,
            )
            info = _run(
                control_plane.command("info", "--format", "{{json .}}"),
                environment=control_plane.environment,
            )
            version_payload = json.loads(version.stdout)
            info_payload = json.loads(info.stdout)
            receipt = build_local_docker_daemon_identity_receipt_v1(
                control_plane,
                version_payload=version_payload,
                info_payload=info_payload,
                repository_root=REPOSITORY_ROOT,
            )
            binding = local_docker_daemon_receipt_binding_v1(receipt)
            docker_root = Path(str(info_payload["DockerRootDir"])).resolve(strict=True)
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            Phase3LocalRuntimeError,
            FormalContainerExecutorError,
        ) as exc:
            if "temporary" in locals():
                temporary.cleanup()
            if isinstance(exc, FormalContainerExecutorError):
                raise
            _fail(FAIL_CONTAINER, f"local Docker control plane is invalid: {exc}")
        self._temporary = temporary
        self._root = root
        self._docker_control_plane = control_plane
        self._docker_daemon_receipt = MappingProxyType(receipt)
        self._docker_daemon_binding = binding
        self._docker_root_directory = docker_root

    def _docker_command(self, *arguments: str) -> list[str]:
        if self._docker_control_plane is None:
            _fail(FAIL_CONTAINER, "local Docker control plane is not initialized")
        return self._docker_control_plane.command(*arguments)

    def _docker(
        self,
        *arguments: str,
        timeout: int = 180,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        if self._docker_control_plane is None:
            _fail(FAIL_CONTAINER, "local Docker control plane is not initialized")
        return _run(
            self._docker_control_plane.command(*arguments),
            timeout=timeout,
            check=check,
            environment=self._docker_control_plane.environment,
        )

    def _load_committed_profile(self) -> None:
        raw_profile = self._git_blob(
            "Hegel Machine/config/phase3_container_actor_profile_v1.json"
        )
        try:
            profile = json.loads(raw_profile)
        except json.JSONDecodeError as exc:
            _fail(FAIL_CONTAINER, f"committed actor profile is invalid: {exc}")
        if type(profile) is not dict:
            _fail(FAIL_CONTAINER, "committed actor profile is not an object")
        if profile.get("authority_disclosure") != dict(
            TECHNICAL_ACTOR_DISCLOSURE_V1
        ):
            _fail(
                FAIL_CONTAINER,
                "committed actor profile authority disclosure differs from the "
                "centralized exact seven-field disclosure",
            )
        self._profile = profile
        self._profile_digest = hashlib.sha256(raw_profile).digest()

    def validate_rust_replay_binding(self, basis: FormalStaticBasisV1) -> bytes:
        expected_path = basis.implementation_inputs.get("rust_binary_path")
        expected_digest = basis.implementation_inputs.get("rust_binary_sha256")
        if type(expected_path) is not str or type(expected_digest) is not bytes or len(expected_digest) != 32:
            _fail(FAIL_PREFLIGHT, "static basis lacks an exact Rust replay binary binding")
        try:
            actual_path = self.rust_formal_replay_binary.resolve(strict=True)
            expected_resolved = Path(expected_path).resolve(strict=True)
        except OSError as exc:
            _fail(FAIL_PREFLIGHT, f"Rust replay binary path cannot be resolved: {exc}")
        if actual_path != expected_resolved:
            _fail(FAIL_PREFLIGHT, "Rust replay binary path differs from Commit-A binding")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(actual_path, flags)
        except OSError as exc:
            _fail(FAIL_PREFLIGHT, f"Rust replay binary cannot be opened: {exc}")
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or not (metadata.st_mode & 0o111):
                _fail(FAIL_PREFLIGHT, "Rust replay binding is not a regular executable")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            actual_digest = digest.digest()
        finally:
            os.close(descriptor)
        if actual_digest != expected_digest:
            _fail(FAIL_PREFLIGHT, "Rust replay binary digest differs from Commit-A binding")
        self._bound_rust_replay_digest = actual_digest
        return actual_digest

    def validate_rust_bridge_dag_binding(self) -> bytes:
        """Bind the post-Commit-A qualified Rust bridge replayer exactly.

        The qualification loader independently rechecks the report, exact Git
        source bindings, offline build evidence, and stable persisted binary.
        This method additionally forbids substituting another caller-selected
        executable path before the actor-private snapshot is created.
        """

        try:
            actual_path = self.rust_bridge_dag_replay_binary.resolve(strict=True)
            stable_path = DEFAULT_RUST_BRIDGE_DAG_BINARY.resolve(strict=True)
            if actual_path != stable_path:
                _fail(
                    FAIL_PREFLIGHT,
                    "Rust bridge DAG binary path differs from the stable qualification path",
                )
            report, digest_text = load_qualified_rust_bridge_dag_binary_binding_v1(
                expected_basis_commit=self.basis_commit,
                report_path=self.rust_bridge_dag_qualification_report,
            )
        except (OSError, BridgeDagBinaryQualificationError) as exc:
            _fail(FAIL_PREFLIGHT, f"Rust bridge DAG qualification is invalid: {exc}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest_text):
            _fail(FAIL_PREFLIGHT, "Rust bridge DAG qualification digest is malformed")
        report_identifier = report.get("diagnostic_report_sha256")
        if (
            type(report_identifier) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", report_identifier) is None
        ):
            _fail(FAIL_PREFLIGHT, "Rust bridge DAG qualification report ID is malformed")
        digest = bytes.fromhex(digest_text.removeprefix("sha256:"))
        self._bound_rust_bridge_dag_digest = digest
        self._bound_rust_bridge_dag_report_sha256 = bytes.fromhex(
            report_identifier.removeprefix("sha256:")
        )
        return digest

    def _git_blob(self, relative: str) -> bytes:
        repository_root = REPOSITORY_ROOT.resolve(strict=True)
        if (
            type(relative) is not str
            or not relative
            or "\0" in relative
            or relative.startswith("-")
            or FORMAL_GIT_EXECUTABLE.resolve(strict=True) != FORMAL_GIT_EXECUTABLE
            or repository_root != REPOSITORY_ROOT
            or repository_root.is_symlink()
        ):
            _fail(FAIL_PREFLIGHT, "formal Git blob path or executable differs")
        completed = _run(
            [
                str(FORMAL_GIT_EXECUTABLE),
                "show",
                f"{self.basis_commit}:{relative}",
            ],
            timeout=60,
            environment=formal_git_environment_v1(),
            working_directory=repository_root,
        )
        return completed.stdout

    def _write_snapshot(self, path: Path, payload: bytes, mode: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)

    def _prepare_inputs(self) -> None:
        assert self._root is not None
        self._load_committed_profile()
        control = self._root / "control"
        control.mkdir(mode=0o700)
        self._runtime_seccomp_path = control / SECCOMP_PATH.name
        self._build_seccomp_path = control / BUILD_SECCOMP_PATH.name
        self._write_snapshot(
            self._runtime_seccomp_path,
            self._git_blob("Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"),
            0o444,
        )
        self._write_snapshot(
            self._build_seccomp_path,
            self._git_blob("Hegel Machine/config/phase3_m3_offline_build_seccomp_v1.json"),
            0o444,
        )
        for purpose in (1, 2, 3, 4):
            actor = self._root / f"purpose-{purpose}"
            input_dir = actor / "input"
            output_dir = actor / "output"
            output_dir.mkdir(parents=True, mode=0o777)
            output_dir.chmod(0o777)
            for relative in ACTOR_SNAPSHOT_PATHS_BY_PURPOSE[purpose]:
                if "/src/hegel_machine/" in relative:
                    destination = input_dir / "src/hegel_machine" / Path(relative).name
                    mode = 0o444
                else:
                    destination = input_dir / "tools" / Path(relative).name
                    mode = 0o555 if relative.endswith((".py", ".sh")) else 0o444
                self._write_snapshot(destination, self._git_blob(relative), mode)
        if self._bound_rust_replay_digest is None:
            _fail(FAIL_CONTAINER, "Rust replay path/digest binding was not validated")
        if not self.rust_formal_replay_binary.is_file():
            _fail(FAIL_CONTAINER, "bound Rust formal replay binary is absent")
        rust_copy = self._root / "purpose-3/input/rust-formal-replay"
        shutil.copyfile(self.rust_formal_replay_binary, rust_copy)
        if hashlib.sha256(rust_copy.read_bytes()).digest() != self._bound_rust_replay_digest:
            _fail(FAIL_CONTAINER, "Rust replay snapshot digest changed during copy")
        rust_copy.chmod(0o555)
        if (
            self._bound_rust_bridge_dag_digest is None
            or self._bound_rust_bridge_dag_report_sha256 is None
        ):
            _fail(
                FAIL_CONTAINER,
                "Rust bridge DAG qualification binding was not validated",
            )
        if not self.rust_bridge_dag_replay_binary.is_file():
            _fail(FAIL_CONTAINER, "bound Rust bridge DAG replay binary is absent")
        bridge_copy = self._root / "purpose-3/input/rust-bridge-dag-replay"
        shutil.copyfile(self.rust_bridge_dag_replay_binary, bridge_copy)
        if (
            hashlib.sha256(bridge_copy.read_bytes()).digest()
            != self._bound_rust_bridge_dag_digest
        ):
            _fail(
                FAIL_CONTAINER,
                "Rust bridge DAG replay snapshot digest changed during copy",
            )
        bridge_copy.chmod(0o555)
        self._compile_rust_split()
        self._prepare_purpose4_detached_inputs()

    def _prepare_purpose4_detached_inputs(self) -> None:
        """Freeze the Commit-A-only runtime and parent history for purpose 4.

        The long-lived actor receives a read-only, self-contained Git object
        store and committed runtime bytes.  It does not receive host-generated
        audit rows, an attestation, or a signing preimage.
        """

        assert self._root is not None
        input_directory = self._root / "purpose-4/input"
        snapshot_destination = input_directory / "detached-parent-snapshot"
        runtime_root = input_directory / "runtime"
        existing_destinations = tuple(
            path
            for path in (snapshot_destination, runtime_root)
            if path.exists() or path.is_symlink()
        )
        for path in existing_destinations:
            try:
                metadata = path.lstat()
            except OSError:
                identity = None
            else:
                identity = metadata.st_dev, metadata.st_ino
            self._purpose4_foreign_entries[path] = identity
        if existing_destinations:
            _fail(FAIL_CONTAINER, "purpose-4 detached input destination already exists")

        snapshot = None
        try:
            snapshot = prepare_detached_parent_snapshot_v1(
                REPOSITORY_ROOT,
                basis_commit=self.basis_commit,
                temporary_parent=self._root,
            )
            self._purpose4_snapshot_owner = snapshot
            source_metadata = snapshot.root.lstat()
            if (
                stat.S_ISLNK(source_metadata.st_mode)
                or not stat.S_ISDIR(source_metadata.st_mode)
                or source_metadata.st_uid != os.geteuid()
            ):
                raise OSError("purpose-4 prepared snapshot identity is invalid")
            self._purpose4_snapshot_identity = (
                source_metadata.st_dev,
                source_metadata.st_ino,
            )
            prepared_source_path = snapshot.root
            temporary = snapshot._temporary

            def record_adopted_identity(identity: tuple[int, int]) -> None:
                if identity != self._purpose4_snapshot_identity:
                    raise OSError("purpose-4 adopted snapshot identity changed")
                self._purpose4_snapshot_path = snapshot_destination
                self._purpose4_vacated_paths.add(prepared_source_path)

            # The helper uses held inodes and renameat2(RENAME_NOREPLACE)
            # only; it cannot degrade into a copy or replace a raced target.
            _adopt_read_only_purpose4_snapshot_v1(
                snapshot,
                snapshot_destination,
                expected_basis_commit=self.basis_commit,
                record_adopted_identity=record_adopted_identity,
            )
            self._purpose4_snapshot_manifest = MappingProxyType(
                dict(snapshot.manifest)
            )
            self._refresh_purpose4_foreign_entries_v1()
            if self._purpose4_foreign_entries:
                raise OSError(
                    "purpose-4 adoption left a foreign object at a vacated path"
                )
            if temporary is not None:
                temporary.cleanup()
            snapshot._temporary = None
            self._purpose4_snapshot_owner = None

            runtime_root.mkdir(parents=True, mode=0o700)
            self._purpose4_runtime_path = runtime_root
            runtime_metadata = runtime_root.lstat()
            self._purpose4_runtime_identity = (
                runtime_metadata.st_dev,
                runtime_metadata.st_ino,
            )
            source_specs = tuple(
                (REPOSITORY_ROOT / repository_path, runtime_path)
                for repository_path, runtime_path in PURPOSE4_RUNTIME_SOURCE_SPECS
            )
            source_bindings = _purpose4_runtime_source_bindings_v1(
                source_specs,
                basis_commit=self.basis_commit,
                git_executable=snapshot.git_executable,
                repository=REPOSITORY_ROOT,
            )
            for repository_path, runtime_path in PURPOSE4_RUNTIME_SOURCE_SPECS:
                destination = runtime_root / runtime_path
                self._write_snapshot(
                    destination,
                    self._git_blob(repository_path),
                    0o444,
                )

            git_destination = runtime_root / "bin/git"
            git_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot.git_executable, git_destination)
            git_destination.chmod(0o555)
            git_binding = snapshot.manifest.get("git_runtime_binding")
            if not isinstance(git_binding, Mapping) or (
                git_binding.get("container_path") != "/runtime/bin/git"
                or git_binding.get("byte_length") != git_destination.stat().st_size
                or git_binding.get("sha256")
                != hashlib.sha256(git_destination.read_bytes()).hexdigest()
            ):
                _fail(FAIL_CONTAINER, "purpose-4 Git runtime copy differs from snapshot binding")

            for directory in sorted(
                (path for path in runtime_root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            runtime_root.chmod(0o555)
            runtime_inventory = _purpose4_runtime_inventory_v1(runtime_root)
            self._purpose4_runtime_bundle = MappingProxyType({
                "runtime_inventory": runtime_inventory,
                "runtime_source_bindings": source_bindings,
            })
        except (OSError, Purpose4DetachedAuditError) as exc:
            unwind_failures: list[str] = []
            if self._purpose4_snapshot_path is None:
                try:
                    foreign_metadata = snapshot_destination.lstat()
                except FileNotFoundError:
                    foreign_present = False
                    foreign_identity = None
                except OSError:
                    foreign_present = True
                    foreign_identity = None
                else:
                    foreign_present = True
                    foreign_identity = (
                        foreign_metadata.st_dev,
                        foreign_metadata.st_ino,
                    )
                if foreign_present:
                    self._purpose4_foreign_entries[
                        snapshot_destination
                    ] = foreign_identity
                    unwind_failures.append("snapshot-foreign-entry-retained")
            owner = self._purpose4_snapshot_owner
            if owner is not None:
                if not self._purpose4_snapshot_tree_removed:
                    owner_path_was_adopted = self._purpose4_snapshot_path is not None
                    owner_path = (
                        self._purpose4_snapshot_path
                        if owner_path_was_adopted
                        else owner.root
                    )

                    def record_owner_quarantine(quarantine: Path) -> None:
                        owner.root = quarantine
                        self._purpose4_vacated_paths.add(owner_path)
                        if owner_path_was_adopted:
                            self._purpose4_snapshot_path = quarantine

                    try:
                        if self._purpose4_snapshot_identity is None:
                            raise OSError(
                                "purpose-4 snapshot cleanup identity is absent"
                            )
                        _remove_exact_owned_purpose4_tree_v1(
                            owner_path,
                            self._purpose4_snapshot_identity,
                            record_quarantine_path=record_owner_quarantine,
                        )
                    except Exception as cleanup:
                        unwind_failures.append(
                            "snapshot-owner-exact-remove:"
                            + type(cleanup).__name__
                        )
                    else:
                        self._purpose4_snapshot_tree_removed = True
                        if owner_path.exists() or owner_path.is_symlink():
                            foreign = owner_path.lstat()
                            self._purpose4_foreign_entries[owner_path] = (
                                foreign.st_dev,
                                foreign.st_ino,
                            )
                            unwind_failures.append(
                                "snapshot-owner-post-quarantine-foreign-"
                                "entry-retained"
                            )

                self._refresh_purpose4_foreign_entries_v1()
                if (
                    self._purpose4_snapshot_tree_removed
                    and not self._purpose4_foreign_entries
                ):
                    try:
                        detached_temporary = owner._temporary
                        if detached_temporary is not None:
                            detached_temporary.cleanup()
                        owner._temporary = None
                    except Exception as cleanup:
                        unwind_failures.append(
                            "snapshot-owner-close:" + type(cleanup).__name__
                        )
                    else:
                        self._purpose4_snapshot_owner = None
                        self._purpose4_snapshot_path = None
                        self._purpose4_snapshot_identity = None
                        self._purpose4_snapshot_manifest = None
                        self._purpose4_snapshot_tree_removed = False
                elif (
                    self._purpose4_snapshot_tree_removed
                    and self._purpose4_foreign_entries
                    and not any(
                        item.startswith("snapshot-foreign-entry-retained")
                        for item in unwind_failures
                    )
                ):
                    unwind_failures.append(
                        "snapshot-owner-foreign-entry-retained"
                    )

            snapshot_removed = self._purpose4_snapshot_owner is None
            if (
                self._purpose4_snapshot_owner is None
                and self._purpose4_snapshot_path is not None
            ):
                original_snapshot_path = self._purpose4_snapshot_path

                def record_snapshot_quarantine(quarantine: Path) -> None:
                    self._purpose4_vacated_paths.add(original_snapshot_path)
                    self._purpose4_snapshot_path = quarantine

                try:
                    if self._purpose4_snapshot_identity is None:
                        raise OSError("purpose-4 snapshot cleanup identity is absent")
                    _remove_exact_owned_purpose4_tree_v1(
                        self._purpose4_snapshot_path,
                        self._purpose4_snapshot_identity,
                        record_quarantine_path=record_snapshot_quarantine,
                    )
                except Exception as cleanup:
                    unwind_failures.append(
                        "snapshot-exact-remove:" + type(cleanup).__name__
                    )
                else:
                    snapshot_removed = True
                    self._purpose4_snapshot_path = None
                    self._purpose4_snapshot_identity = None
                    self._purpose4_snapshot_manifest = None
                    if (
                        original_snapshot_path.exists()
                        or original_snapshot_path.is_symlink()
                    ):
                        foreign = original_snapshot_path.lstat()
                        self._purpose4_foreign_entries[
                            original_snapshot_path
                        ] = (foreign.st_dev, foreign.st_ino)
                        unwind_failures.append(
                            "snapshot-post-quarantine-foreign-entry-retained"
                        )
            elif (
                self._purpose4_snapshot_path is None
                and self._purpose4_snapshot_owner is None
            ):
                self._purpose4_snapshot_identity = None

            runtime_removed = self._purpose4_runtime_path is None
            if self._purpose4_runtime_path is not None:
                original_runtime_path = self._purpose4_runtime_path

                def record_runtime_quarantine(quarantine: Path) -> None:
                    self._purpose4_vacated_paths.add(original_runtime_path)
                    self._purpose4_runtime_path = quarantine

                try:
                    if self._purpose4_runtime_identity is None:
                        raise OSError("purpose-4 runtime cleanup identity is absent")
                    _remove_exact_owned_purpose4_tree_v1(
                        self._purpose4_runtime_path,
                        self._purpose4_runtime_identity,
                        record_quarantine_path=record_runtime_quarantine,
                    )
                except Exception as cleanup:
                    unwind_failures.append("runtime:" + type(cleanup).__name__)
                else:
                    runtime_removed = True
                    self._purpose4_runtime_path = None
                    self._purpose4_runtime_identity = None
                    self._purpose4_runtime_bundle = None
                    if (
                        original_runtime_path.exists()
                        or original_runtime_path.is_symlink()
                    ):
                        foreign = original_runtime_path.lstat()
                        self._purpose4_foreign_entries[
                            original_runtime_path
                        ] = (foreign.st_dev, foreign.st_ino)
                        unwind_failures.append(
                            "runtime-post-quarantine-foreign-entry-retained"
                        )
            self._refresh_purpose4_foreign_entries_v1()
            if self._purpose4_foreign_entries and not any(
                "foreign-entry-retained" in item for item in unwind_failures
            ):
                unwind_failures.append("vacated-path-foreign-entry-retained")
            if not unwind_failures and snapshot_removed and runtime_removed:
                self._purpose4_snapshot_manifest = None
                self._purpose4_runtime_bundle = None
                self._purpose4_snapshot_path = None
                self._purpose4_snapshot_identity = None
                self._purpose4_snapshot_owner = None
                self._purpose4_runtime_path = None
                self._purpose4_runtime_identity = None
            suffix = (
                ""
                if not unwind_failures
                else "; unwind failed: " + ",".join(unwind_failures)
            )
            _fail(
                FAIL_CONTAINER,
                f"purpose-4 detached input preparation failed: {exc}{suffix}",
            )

    def _compile_rust_split(self) -> None:
        assert self._root is not None
        if self._build_seccomp_path is None:
            _fail(FAIL_CONTAINER, "frozen offline-build seccomp snapshot is absent")
        images = self._profile.get("images")
        if not isinstance(images, Mapping) or type(images.get("rust_attester")) is not str:
            _fail(FAIL_CONTAINER, "Rust actor image is absent")
        source = self._root / "purpose-1/input/tools/phase3_split_partition_calculator_fd3_v1.rs"
        probe_source = self._root / "purpose-3/input/tools/phase3_container_actor_probe_v1.rs"
        build = self._root / "rust-split-build"
        build.mkdir(mode=0o777)
        build.chmod(0o777)
        command = self._docker_command(
            "run", "--rm", "--pull=never", "--network=none",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._build_seccomp_path}",
            "--user=65534:65534", "--pids-limit=64", "--memory=512m",
            "--memory-swap=512m",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
            f"--mount=type=bind,src={source},dst=/source.rs,readonly,bind-propagation=rprivate",
            f"--mount=type=bind,src={probe_source},dst=/probe.rs,readonly,bind-propagation=rprivate",
            f"--mount=type=bind,src={build},dst=/build,bind-propagation=rprivate",
            "--entrypoint=/usr/bin/env", str(images["rust_attester"]),
            "-i", "PATH=/usr/local/cargo/bin:/usr/bin:/bin",
            "RUSTUP_HOME=/usr/local/rustup", "CARGO_HOME=/usr/local/cargo",
            "TMPDIR=/build", "/bin/sh", "-c",
            "/usr/local/cargo/bin/rustc --edition=2021 -C debuginfo=0 "
            "-C strip=symbols /source.rs -o /build/rust-split-calculator && "
            "/usr/local/cargo/bin/rustc --edition=2021 -C debuginfo=0 "
            "-C strip=symbols /probe.rs -o /build/rust-live-probe",
        )
        assert self._docker_control_plane is not None
        _run(
            command,
            timeout=240,
            environment=self._docker_control_plane.environment,
        )
        binary = build / "rust-split-calculator"
        probe_binary = build / "rust-live-probe"
        if not binary.is_file() or not probe_binary.is_file():
            _fail(FAIL_CONTAINER, "offline Rust actor build produced an incomplete binary set")
        destination = self._root / "purpose-1/input/rust-split-calculator"
        shutil.copyfile(binary, destination)
        destination.chmod(0o555)
        probe_destination = self._root / "purpose-3/input/rust-live-probe"
        shutil.copyfile(probe_binary, probe_destination)
        probe_destination.chmod(0o555)

    def _base_container_command(self, purpose: int, name: str) -> list[str]:
        assert self._root is not None
        if self._runtime_seccomp_path is None:
            _fail(FAIL_CONTAINER, "frozen runtime seccomp snapshot is absent")
        images = self._profile["images"]
        assert isinstance(images, Mapping)
        image_key = {1: "custodian", 2: "python_attester", 3: "rust_attester", 4: "policy_auditor"}[purpose]
        image = images[image_key]
        if type(image) is not str or "@sha256:" not in image:
            _fail(FAIL_CONTAINER, "actor image is not digest pinned")
        actor = self._root / f"purpose-{purpose}"
        command = self._docker_command(
            "run", "--detach", f"--name={name}", "--pull=never",
            f"--label=hegel.m25.ceremony={self._ceremony_token}",
            f"--label=hegel.m25.purpose={purpose}",
            f"--label=hegel.m25.run={self._transaction_run_id.hex()}",
            f"--label=hegel.m25.basis={self.basis_commit}",
            f"--label=hegel.m25.profile_sha256={self._profile_digest.hex()}",
            f"--label=hegel.m25.daemon_receipt_sha256={self._docker_daemon_binding.hex()}",
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._runtime_seccomp_path}",
            "--user=65534:65534", "--pids-limit=64", "--memory=512m",
            "--memory-swap=512m", "--ulimit=nofile=64:64", "--ipc=private",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=0700",
            f"--mount=type=volume,src={self._state_volumes[purpose]},dst=/state,volume-nocopy",
            f"--mount=type=bind,src={actor / 'input'},dst=/input,readonly,bind-propagation=rprivate",
            f"--mount=type=bind,src={actor / 'output'},dst=/output,bind-propagation=rprivate",
        )
        if purpose == 1:
            command.append(
                f"--mount=type=bind,src={self.custody_directory},dst=/custody,bind-propagation=rprivate"
            )
        command.extend(["--entrypoint=/usr/bin/env", image, "-i"])
        command.extend(
            f"{key}={value}"
            for key, value in self._actor_environment(purpose).items()
        )
        command.extend(["/bin/sleep", "2147483647"])
        return command

    def _actor_environment(
        self,
        purpose: int,
        *,
        operation: str | None = None,
        operation_sequence: int | None = None,
        operation_nonce: bytes | None = None,
        operation_request_digest: bytes | None = None,
    ) -> dict[str, str]:
        if (
            self._transaction_run_id is None
            or self._profile_digest is None
            or self._docker_daemon_binding is None
        ):
            _fail(FAIL_CONTAINER, "actor environment lacks run/profile/daemon binding")
        images = self._profile.get("images")
        if not isinstance(images, Mapping):
            _fail(FAIL_CONTAINER, "actor image registry is absent")
        image_key = {
            1: "custodian",
            2: "python_attester",
            3: "rust_attester",
            4: "policy_auditor",
        }[purpose]
        image = images.get(image_key)
        if type(image) is not str:
            _fail(FAIL_CONTAINER, "actor image binding is absent")
        environment = {
            "HEGEL_ACTOR_IMAGE_REF": image,
            "HEGEL_ACTOR_PROFILE_ID": "hegel-owner-accepted-container-technical-actors-v1",
            "HEGEL_BASIS_COMMIT": self.basis_commit,
            "HEGEL_DAEMON_RECEIPT_SHA256": self._docker_daemon_binding.hex(),
            "HEGEL_HOST_REPOSITORY_PATH_SHA256": hashlib.sha256(
                REPOSITORY_ROOT.resolve().as_posix().encode("utf-8")
            ).hexdigest(),
            "HEGEL_PROFILE_SHA256": self._profile_digest.hex(),
            "HEGEL_PURPOSE_ID": str(purpose),
            "HEGEL_RUN_ID": self._transaction_run_id.hex(),
            "LANG": "C",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        if operation is not None:
            if (
                type(operation_sequence) is not int
                or operation_sequence <= 0
                or type(operation_nonce) is not bytes
                or len(operation_nonce) != 16
                or type(operation_request_digest) is not bytes
                or len(operation_request_digest) != 32
            ):
                _fail(FAIL_CONTAINER, "operation environment binding is incomplete")
            environment["HEGEL_OPERATION_ID"] = operation
            environment["HEGEL_OPERATION_NONCE"] = operation_nonce.hex()
            environment["HEGEL_OPERATION_REQUEST_SHA256"] = (
                operation_request_digest.hex()
            )
            environment["HEGEL_OPERATION_SEQUENCE"] = str(operation_sequence)
            environment["HEGEL_PROBE_INPUT_WRITE_PATH"] = "/input/.hegel-write-probe"
        return environment

    def _actor_launch_environment(
        self,
        purpose: int,
        *,
        operation: str | None = None,
        operation_sequence: int | None = None,
        operation_nonce: bytes | None = None,
        operation_request_digest: bytes | None = None,
    ) -> dict[str, str]:
        """Add the raw clone path only to the private process environment."""

        environment = self._actor_environment(
            purpose,
            operation=operation,
            operation_sequence=operation_sequence,
            operation_nonce=operation_nonce,
            operation_request_digest=operation_request_digest,
        )
        environment["HEGEL_HOST_REPOSITORY_PATH"] = (
            REPOSITORY_ROOT.resolve().as_posix()
        )
        return environment

    def _state_volume_name(self, purpose: int) -> str:
        if self._transaction_run_id is None:
            _fail(FAIL_CONTAINER, "actor state volume lacks a run binding")
        return f"hegel-m25-state-{self._transaction_run_id.hex()}-p{purpose}"

    def _state_volume_labels(self, purpose: int) -> dict[str, str]:
        if self._profile_digest is None:
            _fail(FAIL_CONTAINER, "actor profile digest is absent")
        images = self._profile.get("images")
        if not isinstance(images, Mapping):
            _fail(FAIL_CONTAINER, "actor image registry is absent")
        image_key = {
            1: "custodian", 2: "python_attester", 3: "rust_attester", 4: "policy_auditor"
        }[purpose]
        image = images.get(image_key)
        if type(image) is not str or "@sha256:" not in image:
            _fail(FAIL_CONTAINER, "actor image is not digest pinned")
        assert self._transaction_run_id is not None
        return {
            "hegel.m25.state": "true",
            "hegel.m25.run": self._transaction_run_id.hex(),
            "hegel.m25.purpose": str(purpose),
            "hegel.m25.basis": self.basis_commit,
            "hegel.m25.profile_sha256": self._profile_digest.hex(),
            "hegel.m25.image_sha256": image.rsplit(":", 1)[-1],
            "hegel.m25.daemon_receipt_sha256": self._docker_daemon_binding.hex(),
        }

    def _initialize_new_state_volume(self, purpose: int, volume_name: str) -> None:
        if self._transaction_run_id is None or self._profile_digest is None:
            _fail(FAIL_CONTAINER, "private volume initializer lacks transaction/profile binding")
        images = self._profile["images"]
        assert isinstance(images, Mapping)
        image_key = {
            1: "custodian", 2: "python_attester", 3: "rust_attester", 4: "policy_auditor"
        }[purpose]
        image = images[image_key]
        assert isinstance(image, str)
        if self._runtime_seccomp_path is None:
            _fail(FAIL_CONTAINER, "frozen runtime seccomp snapshot is absent")
        common = [
            "--rm", "--pull=never", "--network=none", "--read-only",
            "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._runtime_seccomp_path}",
            "--pids-limit=16", "--memory=64m", "--memory-swap=64m",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=4m,mode=0700",
            f"--mount=type=volume,src={volume_name},dst=/state,volume-nocopy",
        ]
        initializer = self._docker_command(
            "run", *common, "--cap-drop=ALL", "--cap-add=CHOWN",
            "--user=0:0", "--entrypoint=/usr/bin/env", image, "-i",
            "PATH=/usr/local/bin:/usr/bin:/bin", "/bin/sh", "-c",
            "chmod 0700 /state && chown 65534:65534 /state",
        )
        assert self._docker_control_plane is not None
        _run(
            initializer,
            timeout=60,
            environment=self._docker_control_plane.environment,
        )
        probe = self._docker_command(
            "run", *common, "--cap-drop=ALL", "--user=65534:65534",
            "--entrypoint=/usr/bin/env", image, "-i",
            "PATH=/usr/local/bin:/usr/bin:/bin", "/bin/sh", "-c",
            "test \"$(stat -c %u:%g:%a /state)\" = 65534:65534:700 && "
            ": > /state/.hegel-write-probe && test -f /state/.hegel-write-probe && "
            "rm /state/.hegel-write-probe",
        )
        _run(
            probe,
            timeout=60,
            environment=self._docker_control_plane.environment,
        )
        receipt: dict[str, object] = {
            "schema": "hegel-phase3-m25-private-volume-initialization-receipt/1",
            "basis_commit": self.basis_commit,
            "run_id_hex": self._transaction_run_id.hex(),
            "purpose_id": purpose,
            "volume_name_sha256": hashlib.sha256(volume_name.encode("ascii")).hexdigest(),
            "image_sha256": image.rsplit(":", 1)[-1],
            "profile_sha256": self._profile_digest.hex(),
            "initializer_network_none": True,
            "initializer_capabilities": ["CHOWN"],
            "nonroot_live_write_stat_probe_passed": True,
            "resulting_uid": 65534,
            "resulting_gid": 65534,
            "resulting_mode_octal": "0700",
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
        self._volume_initialization_receipts[purpose] = MappingProxyType(receipt)

    def _validate_state_volume_inspection(
        self,
        *,
        purpose: int,
        name: str,
        row: Mapping[str, object],
        labels: Mapping[str, str],
    ) -> dict[str, object]:
        if self._docker_root_directory is None:
            _fail(FAIL_CONTAINER, "Docker root identity is absent")
        options = row.get("Options")
        mountpoint = Path(str(row.get("Mountpoint", "")))
        expected_mountpoint = self._docker_root_directory / "volumes" / name / "_data"
        checks = {
            "name_exact": row.get("Name") == name,
            "driver_local": row.get("Driver") == "local",
            "scope_local": row.get("Scope") == "local",
            "options_empty": options is None or options == {},
            "labels_exact": row.get("Labels") == dict(labels),
            "daemon_managed_mountpoint_exact": mountpoint == expected_mountpoint,
            "not_bind_nfs_or_plugin": (
                row.get("Driver") == "local"
                and (options is None or options == {})
                and mountpoint == expected_mountpoint
            ),
        }
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            _fail(
                FAIL_CONTAINER,
                f"purpose-{purpose} state volume identity failed: {failed}",
            )
        return {
            "driver": "local",
            "scope": "local",
            "options_empty": True,
            "daemon_managed_mountpoint_sha256": hashlib.sha256(
                mountpoint.as_posix().encode("utf-8")
            ).hexdigest(),
            "checks": checks,
        }

    def _inspect_and_validate_state_volume(
        self,
        purpose: int,
        name: str,
        labels: Mapping[str, str],
    ) -> dict[str, object]:
        inspected = self._docker("volume", "inspect", name, check=False)
        if inspected.returncode != 0:
            _fail(FAIL_CONTAINER, "actor state volume is absent")
        try:
            rows = json.loads(inspected.stdout)
            if type(rows) is not list or len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise TypeError
            row = rows[0]
        except (TypeError, json.JSONDecodeError) as exc:
            _fail(FAIL_CONTAINER, f"actor state volume inspection is invalid: {exc}")
        return self._validate_state_volume_inspection(
            purpose=purpose,
            name=name,
            row=row,
            labels=labels,
        )

    def _create_or_validate_state_volumes(self) -> None:
        for purpose in (1, 2, 3, 4):
            name = self._state_volume_name(purpose)
            labels = self._state_volume_labels(purpose)
            if self._recovery_mode:
                self._inspect_and_validate_state_volume(purpose, name, labels)
                self._state_volumes[purpose] = name
                continue
            probe = self._docker("volume", "inspect", name, check=False)
            if probe.returncode == 0:
                _fail(FAIL_CONTAINER, "first-run actor state volume already exists")
            command = self._docker_command("volume", "create")
            for key, value in sorted(labels.items()):
                command.append(f"--label={key}={value}")
            command.append(name)
            assert self._docker_control_plane is not None
            created = _run(command, environment=self._docker_control_plane.environment)
            # Record the explicit requested name before validating Docker's
            # response so exception cleanup can still destroy the volume.
            self._state_volumes[purpose] = name
            if created.stdout.decode("ascii", "strict").strip() != name:
                _fail(FAIL_CONTAINER, "Docker returned another actor state volume name")
            volume_identity = self._inspect_and_validate_state_volume(
                purpose, name, labels
            )
            self._initialize_new_state_volume(purpose, name)
            receipt = dict(self._volume_initialization_receipts[purpose])
            receipt["volume_identity"] = volume_identity
            receipt["daemon_receipt_sha256"] = self._docker_daemon_binding.hex()
            receipt.pop("receipt_sha256", None)
            receipt["receipt_sha256"] = hashlib.sha256(
                _canonical_json(receipt)
            ).hexdigest()
            self._volume_initialization_receipts[purpose] = MappingProxyType(receipt)

    def _recover_remove_exact_actor_containers(self) -> None:
        """Remove only actor containers exactly bound to this transaction."""

        if self._transaction_run_id is None or self._profile_digest is None:
            _fail(FAIL_CONTAINER, "actor recovery lacks run/profile binding")
        expected_by_volume = {
            self._state_volume_name(purpose): purpose for purpose in (1, 2, 3, 4)
        }
        discovered: set[str] = set()
        for volume_name in expected_by_volume:
            listed = self._docker(
                "ps", "-aq", "--filter", f"volume={volume_name}",
                timeout=30,
                check=False,
            )
            if listed.returncode != 0:
                _fail(FAIL_CONTAINER, "cannot enumerate recovery actor containers")
            discovered.update(
                row for row in listed.stdout.decode("ascii", "strict").splitlines() if row
            )
        labelled = self._docker(
            "ps", "-aq", "--filter",
            f"label=hegel.m25.run={self._transaction_run_id.hex()}",
            timeout=30,
            check=False,
        )
        if labelled.returncode != 0:
            _fail(FAIL_CONTAINER, "cannot enumerate run-labelled recovery actors")
        labelled_ids = {
            row for row in labelled.stdout.decode("ascii", "strict").splitlines() if row
        }
        if labelled_ids != discovered:
            _fail(FAIL_CONTAINER, "run-labelled and state-mounted actor sets differ")
        if not discovered:
            return
        inspection = self._docker("inspect", *sorted(discovered), timeout=60)
        try:
            rows = json.loads(inspection.stdout)
        except json.JSONDecodeError as exc:
            _fail(FAIL_CONTAINER, f"recovery actor inspection is invalid: {exc}")
        if type(rows) is not list or len(rows) != len(discovered):
            _fail(FAIL_CONTAINER, "recovery actor inspection cardinality differs")
        seen_purposes: set[int] = set()
        images = self._profile.get("images")
        if not isinstance(images, Mapping):
            _fail(FAIL_CONTAINER, "recovery image registry is absent")
        image_key = {
            1: "custodian", 2: "python_attester", 3: "rust_attester", 4: "policy_auditor"
        }
        for row in rows:
            try:
                labels = row["Config"]["Labels"]
                purpose = int(labels["hegel.m25.purpose"])
                mounts = row["Mounts"]
                host = row["HostConfig"]
                config = row["Config"]
            except (KeyError, TypeError, ValueError) as exc:
                _fail(FAIL_CONTAINER, f"recovery actor metadata is invalid: {exc}")
            expected_labels = {
                "hegel.m25.purpose": str(purpose),
                "hegel.m25.run": self._transaction_run_id.hex(),
                "hegel.m25.basis": self.basis_commit,
                "hegel.m25.profile_sha256": self._profile_digest.hex(),
            }
            state_mounts = [
                mount
                for mount in mounts
                if mount.get("Type") == "volume" and mount.get("Destination") == "/state"
            ]
            transaction_private_mounts = [
                mount
                for mount in mounts
                if mount.get("Type") == "volume"
                and mount.get("Name") in expected_by_volume
            ]
            expected_mount_profile = {
                "/state": ("volume", True),
                "/input": ("bind", False),
                "/output": ("bind", True),
                **({"/custody": ("bind", True)} if purpose == 1 else {}),
            }
            actual_mount_profile = {
                mount.get("Destination"): (mount.get("Type"), mount.get("RW"))
                for mount in mounts
            }
            if (
                purpose not in {1, 2, 3, 4}
                or purpose in seen_purposes
                or not isinstance(labels, Mapping)
                or any(labels.get(key) != value for key, value in expected_labels.items())
                or type(labels.get("hegel.m25.ceremony")) is not str
                or re.fullmatch(r"[0-9a-f]{16}", labels["hegel.m25.ceremony"]) is None
                or len(state_mounts) != 1
                or state_mounts[0].get("Name") != self._state_volume_name(purpose)
                or len(transaction_private_mounts) != 1
                or actual_mount_profile != expected_mount_profile
                or host.get("NetworkMode") != "none"
                or host.get("ReadonlyRootfs") is not True
                or host.get("Privileged") is not False
                or "ALL" not in (host.get("CapDrop") or ())
                or config.get("User") != "65534:65534"
                or config.get("Image") != images.get(image_key[purpose])
            ):
                _fail(FAIL_CONTAINER, "recovery actor does not match the frozen live profile")
            seen_purposes.add(purpose)
        for container_id in sorted(discovered):
            removed = self._docker(
                "rm", "--force", container_id, timeout=30, check=False
            )
            if removed.returncode != 0:
                _fail(FAIL_CONTAINER, "recovery actor removal failed")
        survivor = self._docker(
            "ps", "-aq", "--filter",
            f"label=hegel.m25.run={self._transaction_run_id.hex()}",
            timeout=30,
            check=False,
        )
        if survivor.returncode != 0 or survivor.stdout.strip():
            _fail(FAIL_CONTAINER, "a run-labelled recovery actor remains")

    def _load_complete_recovery_profile(self, run_id: bytes) -> None:
        self.bind_transaction_identity(run_id)
        self._recovery_mode = True
        self._ensure_local_runtime()
        self._load_committed_profile()
        assert self._root is not None
        control = self._root / "control"
        control.mkdir(mode=0o700, exist_ok=True)
        self._runtime_seccomp_path = control / SECCOMP_PATH.name
        self._write_snapshot(
            self._runtime_seccomp_path,
            self._git_blob("Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"),
            0o444,
        )
        try:
            location = validate_linux_local_durable_custody_location_v1(
                self._custody_requested,
                repository_root=REPOSITORY_ROOT,
                allowed_owner_uids=frozenset({os.geteuid(), 65534}),
            )
            self._custody_durability_receipt = MappingProxyType({
                **location,
                "durability_probe_deferred_until_after_actor_reclaim": True,
            })
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_CUSTODY, f"durable custody location failed: {exc.code}: {exc.detail}")

    def _reclaim_and_fully_validate_complete_custody(self) -> None:
        if self._custody_durability_receipt is None:
            _fail(FAIL_CUSTODY, "complete recovery custody location was not qualified")
        owner_uid = self._custody_durability_receipt.get("owner_uid")
        if owner_uid == 65534:
            self._custody_handed_off = True
            self._reclaim_custody_from_actor()
        elif owner_uid != os.geteuid():
            _fail(FAIL_CUSTODY, "complete recovery custody owner differs")
        try:
            self._custody_durability_receipt = MappingProxyType(
                validate_linux_local_durable_custody_v1(
                    self._custody_requested,
                    repository_root=REPOSITORY_ROOT,
                )
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_CUSTODY, f"durable custody failed: {exc.code}: {exc.detail}")

    def recover_complete_private_state_and_verify_absent(
        self, run_id: bytes, marker: MarkerSnapshot
    ) -> None:
        if marker.state != "COMPLETE" or marker.seed_commitment_manifest_root is None:
            _fail(FAIL_CUSTODY, "complete-state recovery requires an exact COMPLETE marker")
        try:
            self._load_complete_recovery_profile(run_id)
            self._recover_remove_exact_actor_containers()
            self._reclaim_and_fully_validate_complete_custody()
            for purpose in (1, 2, 3, 4):
                volume_name = self._state_volume_name(purpose)
                inspected = self._docker("volume", "inspect", volume_name, check=False)
                if inspected.returncode != 0:
                    listed_exact = self._docker(
                        "volume", "ls", "-q", "--filter",
                        f"name=^{volume_name}$",
                        timeout=30,
                        check=False,
                    )
                    if listed_exact.returncode != 0:
                        _fail(FAIL_CONTAINER, "cannot establish recovery volume absence")
                    listed_names = listed_exact.stdout.decode("ascii", "strict").splitlines()
                    if listed_names:
                        _fail(FAIL_CONTAINER, "recovery volume is listed but cannot be inspected")
                    continue
                try:
                    rows = json.loads(inspected.stdout)
                    if type(rows) is not list or len(rows) != 1 or not isinstance(rows[0], Mapping):
                        raise TypeError
                    row = rows[0]
                except (TypeError, json.JSONDecodeError) as exc:
                    _fail(FAIL_CONTAINER, f"recovery volume inspection is invalid: {exc}")
                self._validate_state_volume_inspection(
                    purpose=purpose,
                    name=volume_name,
                    row=row,
                    labels=self._state_volume_labels(purpose),
                )
                removed = self._docker(
                    "volume", "rm", volume_name, timeout=30, check=False
                )
                if removed.returncode != 0:
                    _fail(FAIL_CONTAINER, "recovery actor-key-volume removal failed")
            remaining = self._docker(
                "volume", "ls", "-q", "--filter",
                f"label=hegel.m25.run={run_id.hex()}",
                timeout=30,
                check=False,
            )
            if remaining.returncode != 0 or remaining.stdout.strip():
                _fail(FAIL_CONTAINER, "recovery actor key volume remains")
            self._marker_completed_after_staging = True
            self._verify_complete_custody_retained()
        finally:
            if self._docker_control_plane is not None:
                self._cleanup_local_runtime()

    def recover_preseed_private_state_and_verify_absent(
        self, run_id: bytes
    ) -> Mapping[str, object]:
        """Remove only exact run-labelled role state when no seed marker exists."""

        forbidden = (
            "split_seed_instantiation.marker",
            "split_seed_generation.intent",
            "split_seed_generation.complete",
            "split_master_seed.bin",
        )
        if any(
            (self.custody_directory / name).exists()
            or (self.custody_directory / name).is_symlink()
            for name in forbidden
        ):
            _fail(FAIL_CUSTODY, "preseed actor-state cleanup found seed continuity state")
        try:
            self._load_complete_recovery_profile(run_id)
            self._recover_remove_exact_actor_containers()
            self._reclaim_and_fully_validate_complete_custody()
            for purpose in (1, 2, 3, 4):
                volume_name = self._state_volume_name(purpose)
                inspected = self._docker("volume", "inspect", volume_name, check=False)
                if inspected.returncode != 0:
                    listed = self._docker(
                        "volume", "ls", "-q", "--filter",
                        f"name=^{volume_name}$", timeout=30, check=False,
                    )
                    if listed.returncode != 0 or listed.stdout.strip():
                        _fail(FAIL_CONTAINER, "preseed volume absence cannot be established")
                    continue
                try:
                    rows = json.loads(inspected.stdout)
                    if (
                        type(rows) is not list
                        or len(rows) != 1
                        or not isinstance(rows[0], Mapping)
                    ):
                        raise TypeError
                except (TypeError, json.JSONDecodeError) as exc:
                    _fail(FAIL_CONTAINER, f"preseed volume inspection is invalid: {exc}")
                self._validate_state_volume_inspection(
                    purpose=purpose,
                    name=volume_name,
                    row=rows[0],
                    labels=self._state_volume_labels(purpose),
                )
                removed = self._docker(
                    "volume", "rm", volume_name, timeout=30, check=False
                )
                if removed.returncode != 0:
                    _fail(FAIL_CONTAINER, "preseed actor-key-volume removal failed")
            remaining_containers = self._docker(
                "ps", "-aq", "--filter", f"label=hegel.m25.run={run_id.hex()}",
                timeout=30, check=False,
            )
            remaining_volumes = self._docker(
                "volume", "ls", "-q", "--filter",
                f"label=hegel.m25.run={run_id.hex()}", timeout=30, check=False,
            )
            if (
                remaining_containers.returncode != 0
                or remaining_containers.stdout.strip()
                or remaining_volumes.returncode != 0
                or remaining_volumes.stdout.strip()
            ):
                _fail(FAIL_CONTAINER, "preseed run-labelled private state remains")
            receipt: dict[str, object] = {
                "schema": "hegel-phase3-m25-preseed-actor-absence-receipt/1",
                "basis_commit": self.basis_commit,
                "run_id_hex": run_id.hex(),
                "exact_run_label_checked": True,
                "actor_containers_absent": True,
                "actor_key_volumes_absent": True,
                "seed_continuity_state_absent": True,
                "docker_daemon_receipt_sha256": self._docker_daemon_binding.hex(),
            }
            receipt["receipt_sha256"] = hashlib.sha256(
                _canonical_json(receipt)
            ).hexdigest()
            return MappingProxyType(receipt)
        finally:
            if self._docker_control_plane is not None:
                self._cleanup_local_runtime()

    def recover_complete_actor_key_volumes_and_verify_absent(
        self, run_id: bytes, marker: MarkerSnapshot
    ) -> None:
        self.recover_complete_private_state_and_verify_absent(run_id, marker)

    def start(self) -> "DockerCeremonyActorsV1":
        if self._containers or self._temporary is not None:
            _fail(FAIL_CONTAINER, "Docker ceremony actors may be started only once")
        self._ensure_local_runtime()
        try:
            return self._start_with_local_runtime()
        except BaseException as original:
            if self._temporary is not None or self._containers:
                try:
                    self.close()
                except BaseException as cleanup:
                    raise cleanup from original
            raise

    def _start_with_local_runtime(self) -> "DockerCeremonyActorsV1":
        if self._recovery_mode:
            self._load_committed_profile()
            # A host power loss may leave the 0700 custody tree owned by the
            # container UID.  This administrative step handles only metadata
            # ownership and never reads the seed.
            self._custody_handed_off = True
            self._reclaim_custody_from_actor()
        try:
            self._custody_durability_receipt = MappingProxyType(
                validate_linux_local_durable_custody_v1(
                    self._custody_requested,
                    repository_root=REPOSITORY_ROOT,
                )
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_CUSTODY, f"durable custody failed: {exc.code}: {exc.detail}")
        if not self.custody_directory.is_dir() or any(self.custody_directory.iterdir()):
            # A transaction lock plus exact run/ledger reservations are the
            # only allowed pre-start custody entries.
            allowed = {
                "phase3_m25_ceremony.lock",
                *(
                    path.name
                    for path in self.custody_directory.glob("opaque-*-*.reserved")
                ),
            }
            actual = {path.name for path in self.custody_directory.iterdir()}
            reservation_names = {
                name
                for name in actual
                if re.fullmatch(r"opaque-(run|ledger)-[0-9a-f]{32}\.reserved", name)
            }
            if self._recovery_mode:
                allowed.update({
                    "split_seed_instantiation.marker",
                    "split_seed_generation.intent",
                    "split_seed_generation.complete",
                    "split_master_seed.bin",
                })
            kinds = {
                match.group(1)
                for name in reservation_names
                if (
                    match := re.fullmatch(
                        r"opaque-(run|ledger)-[0-9a-f]{32}\.reserved", name
                    )
                )
            }
            if (
                not actual
                or (not self._recovery_mode and actual != allowed)
                or (self._recovery_mode and (
                    not actual <= allowed
                    or "split_seed_instantiation.marker" not in actual
                ))
                or len(reservation_names) != 2
                or kinds != {"run", "ledger"}
            ):
                _fail(FAIL_CUSTODY, "Docker ceremony custody reservation set is invalid")
        if (self.custody_directory.stat().st_mode & 0o777) != 0o700:
            _fail(FAIL_CUSTODY, "custody directory must be 0700")
        token = secrets.token_hex(8)
        try:
            self._prepare_inputs()
            if self._transaction_run_id is None or self._profile_digest is None:
                _fail(
                    FAIL_CONTAINER,
                    "transaction-bound actor state volume identity is absent",
                )
            self._ceremony_token = token
            if self._recovery_mode:
                self._recover_remove_exact_actor_containers()
            self._create_or_validate_state_volumes()
            for purpose in (1, 2, 3, 4):
                name = f"hegel-m25-formal-p{purpose}-{token}"
                assert self._docker_control_plane is not None
                result = _run(
                    self._base_container_command(purpose, name),
                    environment=self._docker_control_plane.environment,
                )
                container_id = result.stdout.decode("ascii", "strict").strip()
                if not re.fullmatch(r"[0-9a-f]{64}", container_id):
                    _fail(FAIL_CONTAINER, "Docker returned an invalid container ID")
                self._containers[purpose] = container_id
                self._container_names[purpose] = name
            inspection = self._docker(
                "inspect", *self._containers.values(), timeout=60
            )
            rows = json.loads(inspection.stdout)
            if len(rows) != 4 or any(
                row["State"]["Running"] is not True
                or row["HostConfig"]["NetworkMode"] != "none"
                or row["HostConfig"]["ReadonlyRootfs"] is not True
                for row in rows
            ):
                _fail(FAIL_CONTAINER, "the four purpose containers are not simultaneously isolated")
            for purpose in (1, 2, 3, 4):
                self._exec(purpose, "qualify-only")
            if set(self._live_actor_probe_receipts) != {1, 2, 3, 4}:
                _fail(FAIL_CONTAINER, "the four live actors did not all produce a fresh receipt")
            host_pids = {
                int(receipt["host_inspection"]["host_pid"])
                for receipt in self._live_actor_probe_receipts.values()
            }
            container_ids = {
                str(receipt["container_id"])
                for receipt in self._live_actor_probe_receipts.values()
            }
            if len(host_pids) != 4 or len(container_ids) != 4:
                _fail(FAIL_CONTAINER, "live actor process/container identities are not distinct")
            for namespace in ("ipc", "mnt", "net", "pid", "uts"):
                identities = {
                    str(receipt["namespaces"][namespace])
                    for receipt in self._live_actor_probe_receipts.values()
                }
                if len(identities) != 4:
                    _fail(
                        FAIL_CONTAINER,
                        f"live actor {namespace} namespace identities are not distinct",
                    )
            self._live_actor_set_qualified = True
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                raise cleanup from original
            raise
        return self

    def __enter__(self) -> "DockerCeremonyActorsV1":
        return self.start()

    def _cleanup_local_runtime(self) -> None:
        temporary = self._temporary
        failures: list[str] = []
        cleanup_blocked = False
        if self._purpose4_foreign_entries:
            for path, expected_identity in self._purpose4_foreign_entries.items():
                try:
                    metadata = path.lstat()
                except OSError as exc:
                    observed = "absent-or-unreadable:" + type(exc).__name__
                else:
                    identity = metadata.st_dev, metadata.st_ino
                    observed = (
                        "same-identity"
                        if expected_identity is not None
                        and identity == expected_identity
                        else "identity-changed"
                    )
                failures.append(
                    "purpose4-foreign-entry-retained:"
                    + path.name
                    + ":"
                    + observed
                )
            cleanup_blocked = True
        owner = self._purpose4_snapshot_owner
        if owner is not None:
            owner_foreign = False
            if not self._purpose4_snapshot_tree_removed:
                owner_path_was_adopted = self._purpose4_snapshot_path is not None
                owner_path = (
                    self._purpose4_snapshot_path
                    if owner_path_was_adopted
                    else owner.root
                )

                def record_owner_quarantine(quarantine: Path) -> None:
                    owner.root = quarantine
                    self._purpose4_vacated_paths.add(owner_path)
                    if owner_path_was_adopted:
                        self._purpose4_snapshot_path = quarantine

                try:
                    if self._purpose4_snapshot_identity is None:
                        raise OSError(
                            "purpose-4 snapshot cleanup identity is absent"
                        )
                    _remove_exact_owned_purpose4_tree_v1(
                        owner_path,
                        self._purpose4_snapshot_identity,
                        record_quarantine_path=record_owner_quarantine,
                    )
                except Exception as exc:
                    failures.append(
                        "purpose4-detached-owner-exact-remove:"
                        + type(exc).__name__
                    )
                    cleanup_blocked = True
                else:
                    self._purpose4_snapshot_tree_removed = True
                    if owner_path.exists() or owner_path.is_symlink():
                        foreign = owner_path.lstat()
                        self._purpose4_foreign_entries[owner_path] = (
                            foreign.st_dev,
                            foreign.st_ino,
                        )
                        failures.append(
                            "purpose4-detached-owner-post-quarantine-foreign-"
                            "entry-retained"
                        )
                        cleanup_blocked = True
                        owner_foreign = True
            self._refresh_purpose4_foreign_entries_v1()
            if self._purpose4_foreign_entries:
                cleanup_blocked = True
            if (
                self._purpose4_snapshot_tree_removed
                and not owner_foreign
                and not cleanup_blocked
            ):
                try:
                    detached_temporary = owner._temporary
                    if detached_temporary is not None:
                        detached_temporary.cleanup()
                    owner._temporary = None
                except Exception as exc:
                    failures.append(
                        "purpose4-detached-owner-close:" + type(exc).__name__
                    )
                    cleanup_blocked = True
                else:
                    self._purpose4_snapshot_owner = None
                    self._purpose4_snapshot_path = None
                    self._purpose4_snapshot_identity = None
                    self._purpose4_snapshot_manifest = None
                    self._purpose4_snapshot_tree_removed = False

        if (
            self._purpose4_snapshot_owner is None
            and self._purpose4_snapshot_path is not None
        ):
            original_snapshot_path = self._purpose4_snapshot_path

            def record_snapshot_quarantine(quarantine: Path) -> None:
                self._purpose4_vacated_paths.add(original_snapshot_path)
                self._purpose4_snapshot_path = quarantine

            try:
                if self._purpose4_snapshot_identity is None:
                    raise OSError("purpose-4 snapshot cleanup identity is absent")
                _remove_exact_owned_purpose4_tree_v1(
                    self._purpose4_snapshot_path,
                    self._purpose4_snapshot_identity,
                    record_quarantine_path=record_snapshot_quarantine,
                )
            except Exception as exc:
                failures.append(
                    "purpose4-detached-snapshot-exact-remove:"
                    + type(exc).__name__
                )
                cleanup_blocked = True
            else:
                self._purpose4_snapshot_path = None
                self._purpose4_snapshot_identity = None
                self._purpose4_snapshot_manifest = None
                if (
                    original_snapshot_path.exists()
                    or original_snapshot_path.is_symlink()
                ):
                    foreign = original_snapshot_path.lstat()
                    self._purpose4_foreign_entries[
                        original_snapshot_path
                    ] = (foreign.st_dev, foreign.st_ino)
                    failures.append(
                        "purpose4-detached-snapshot-post-quarantine-foreign-"
                        "entry-retained"
                    )
                    cleanup_blocked = True

        if self._purpose4_runtime_path is not None:
            original_runtime_path = self._purpose4_runtime_path

            def record_runtime_quarantine(quarantine: Path) -> None:
                self._purpose4_vacated_paths.add(original_runtime_path)
                self._purpose4_runtime_path = quarantine

            try:
                if self._purpose4_runtime_identity is None:
                    raise OSError("purpose-4 runtime cleanup identity is absent")
                _remove_exact_owned_purpose4_tree_v1(
                    self._purpose4_runtime_path,
                    self._purpose4_runtime_identity,
                    record_quarantine_path=record_runtime_quarantine,
                )
            except Exception as exc:
                failures.append(
                    "purpose4-committed-runtime-exact-remove:"
                    + type(exc).__name__
                )
                cleanup_blocked = True
            else:
                self._purpose4_runtime_path = None
                self._purpose4_runtime_identity = None
                self._purpose4_runtime_bundle = None
                if original_runtime_path.exists() or original_runtime_path.is_symlink():
                    foreign = original_runtime_path.lstat()
                    self._purpose4_foreign_entries[
                        original_runtime_path
                    ] = (foreign.st_dev, foreign.st_ino)
                    failures.append(
                        "purpose4-committed-runtime-post-quarantine-foreign-"
                        "entry-retained"
                    )
                    cleanup_blocked = True

        self._refresh_purpose4_foreign_entries_v1()
        if self._purpose4_foreign_entries:
            cleanup_blocked = True
            if not any(
                item.startswith("purpose4-foreign-entry-retained:")
                for item in failures
            ):
                failures.append("purpose4-vacated-path-foreign-entry-retained")
        cleanup_completed = temporary is None and not cleanup_blocked
        if temporary is not None and not cleanup_blocked:
            try:
                temporary.cleanup()
                cleanup_completed = True
            except Exception as exc:
                failures.append(f"actor-temporary-cleanup:{type(exc).__name__}")
                cleanup_blocked = True
        if cleanup_completed:
            self._purpose4_snapshot_manifest = None
            self._purpose4_runtime_bundle = None
            self._purpose4_snapshot_path = None
            self._purpose4_snapshot_identity = None
            self._purpose4_snapshot_owner = None
            self._purpose4_snapshot_tree_removed = False
            self._purpose4_runtime_path = None
            self._purpose4_runtime_identity = None
            self._purpose4_foreign_entries.clear()
            self._purpose4_vacated_paths.clear()
            self._docker_control_plane = None
            self._runtime_seccomp_path = None
            self._build_seccomp_path = None
            self._docker_root_directory = None
            self._root = None
            self._temporary = None
        if failures:
            _fail(
                FAIL_CONTAINER,
                "actor local-runtime cleanup failed: " + ",".join(failures),
            )

    def _verify_complete_custody_retained(self) -> Mapping[str, object]:
        """Verify metadata-only retention of the raw seed continuity state.

        The supervisor deliberately never opens or hashes the raw seed.  The
        receipt proves only its exact inode type, owner, mode and 32-byte size,
        while the four actor key volumes are handled by a separate lifecycle.
        """

        marker_path = self.custody_directory / "split_seed_instantiation.marker"
        marker = read_marker_snapshot_v1(marker_path)
        if marker.state != "COMPLETE" or marker.seed_commitment_manifest_root is None:
            _fail(FAIL_CUSTODY, "custody retention verification requires COMPLETE")
        expected = {
            "split_seed_instantiation.marker": None,
            "split_seed_generation.intent": None,
            "split_seed_generation.complete": None,
            "split_master_seed.bin": 32,
        }
        rows: list[dict[str, object]] = []
        for name, exact_size in expected.items():
            path = self.custody_directory / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                _fail(FAIL_CUSTODY, f"retained custody artifact is absent: {name}: {exc}")
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (exact_size is not None and metadata.st_size != exact_size)
            ):
                _fail(FAIL_CUSTODY, f"retained custody artifact metadata differs: {name}")
            rows.append({
                "name": name,
                "mode_octal": "0600",
                "owner_uid": metadata.st_uid,
                "size_or_null": metadata.st_size if exact_size is not None else None,
            })
        receipt: dict[str, object] = {
            "schema": "hegel-phase3-m25-complete-custody-retention/1",
            "actor_key_volume_count_expected_absent": 4,
            "custody_artifact_rows": rows,
            "raw_seed_bytes_read_by_supervisor": False,
            "raw_seed_retained_for_frozen_continuity_policy": True,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json(receipt)
        ).hexdigest()
        self._custody_retention_receipt = MappingProxyType(receipt)
        return self._custody_retention_receipt

    def close(self) -> None:
        if self._docker_control_plane is None:
            if self._containers or self._state_volumes:
                _fail(
                    FAIL_CONTAINER,
                    "actor identities remain without their bound Docker control plane",
                )
            return
        failures: list[str] = []
        container_ids = tuple(self._containers.values())
        for container_id in container_ids:
            removed = self._docker(
                "rm", "--force", container_id, timeout=30, check=False
            )
            if removed.returncode != 0:
                failures.append("docker-rm-nonzero")
        if container_ids:
            inspection = self._docker(
                "inspect", *container_ids, timeout=30, check=False
            )
            if inspection.returncode == 0:
                failures.append("actor-container-still-inspectable")
        if self._ceremony_token is not None:
            listed = self._docker(
                "ps", "-aq", "--filter",
                f"label=hegel.m25.ceremony={self._ceremony_token}",
                timeout=30,
                check=False,
            )
            if listed.returncode != 0 or listed.stdout.strip():
                failures.append("actor-or-labelled-descendant-remains")
        try:
            self._reclaim_custody_from_actor()
        except FormalContainerExecutorError:
            failures.append("custody-reclaim-failed")
        marker_path = self.custody_directory / "split_seed_instantiation.marker"
        marker_error: str | None = None
        marker_state = "ABSENT"
        if marker_path.exists() or marker_path.is_symlink():
            try:
                marker_state = read_marker_snapshot_v1(marker_path).state
            except Exception as exc:
                # Malformed/unreadable marker state can conceal an irreversible
                # seed choice.  Retain the private volumes and make cleanup
                # fatal; never infer ABSENT from a parse or permission error.
                marker_state = "INVALID_RETAIN"
                marker_error = type(exc).__name__

        destroy_volumes = marker_state == "ABSENT" or (
            marker_state == "COMPLETE" and self._marker_completed_after_staging
        )
        if destroy_volumes:
            volume_failures_before = len(failures)
            for purpose, volume_name in tuple(self._state_volumes.items()):
                try:
                    self._inspect_and_validate_state_volume(
                        purpose,
                        volume_name,
                        self._state_volume_labels(purpose),
                    )
                except FormalContainerExecutorError:
                    failures.append(f"purpose-{purpose}-state-volume-identity-failed")
                    continue
                removed = self._docker(
                    "volume", "rm", volume_name, timeout=30, check=False
                )
                if removed.returncode != 0:
                    failures.append(f"purpose-{purpose}-state-volume-remove-failed")
                    continue
                present = self._docker(
                    "volume", "inspect", volume_name, timeout=30, check=False
                )
                if present.returncode == 0:
                    failures.append(f"purpose-{purpose}-state-volume-still-inspectable")
                listed_exact = self._docker(
                    "volume", "ls", "-q", "--filter",
                    f"name=^{volume_name}$",
                    timeout=30,
                    check=False,
                )
                if listed_exact.returncode != 0 or listed_exact.stdout.strip():
                    failures.append(f"purpose-{purpose}-state-volume-still-listed")
            if self._transaction_run_id is not None:
                listed_volumes = self._docker(
                    "volume", "ls", "-q", "--filter",
                    f"label=hegel.m25.run={self._transaction_run_id.hex()}",
                    timeout=30,
                    check=False,
                )
                if listed_volumes.returncode != 0 or listed_volumes.stdout.strip():
                    failures.append("purpose-state-volume-remains-after-destruction")
            if len(failures) == volume_failures_before:
                self._state_volumes.clear()
        elif self._state_volumes:
            # PENDING, unauthorized COMPLETE, and invalid marker states retain
            # all role-private volumes.  Marker absence is the only pre-seed
            # state and is handled by the destruction branch above.
            for purpose, volume_name in self._state_volumes.items():
                try:
                    self._inspect_and_validate_state_volume(
                        purpose,
                        volume_name,
                        self._state_volume_labels(purpose),
                    )
                except FormalContainerExecutorError:
                    failures.append("pending-purpose-state-volume-invalid-or-lost")
        if marker_error is not None:
            failures.append("invalid-marker-private-state-retained:" + marker_error)
        if (
            marker_state == "COMPLETE"
            and self._marker_completed_after_staging
            and not self._state_volumes
        ):
            try:
                self._verify_complete_custody_retained()
            except FormalContainerExecutorError:
                failures.append("complete-custody-retention-verification-failed")
        if failures:
            _fail(FAIL_CONTAINER, "container cleanup verification failed: " + ",".join(failures))
        self._containers.clear()
        self._container_names.clear()
        self._live_actor_set_qualified = False
        self._cleanup_local_runtime()
        self._ceremony_token = None

    def close_and_verify_absent(self) -> None:
        self.close()

    def stop_for_recovery_and_verify_absent(self) -> None:
        if self._marker_completed_after_staging:
            _fail(FAIL_CONTAINER, "recovery stop cannot follow destruction authorization")
        self.close()

    def destroy_private_state_and_verify_absent(self) -> None:
        if not self._marker_completed_after_staging:
            _fail(FAIL_CONTAINER, "private state destruction is not authorized")
        self.close()

    def destroy_actor_key_volumes_and_verify_absent(self) -> None:
        self.destroy_private_state_and_verify_absent()

    def authorize_private_state_destruction(self, marker: MarkerSnapshot) -> None:
        if marker.state != "COMPLETE" or marker.seed_commitment_manifest_root is None:
            _fail(FAIL_CUSTODY, "private actor state destruction requires COMPLETE marker")
        self._marker_completed_after_staging = True

    def authorize_actor_key_volume_destruction(
        self, marker: MarkerSnapshot
    ) -> None:
        self.authorize_private_state_destruction(marker)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def _actor_dirs(self, purpose: int) -> tuple[Path, Path]:
        if self._root is None or purpose not in self._containers:
            _fail(FAIL_CONTAINER, "actor set is not live")
        return (
            self._root / f"purpose-{purpose}/input",
            self._root / f"purpose-{purpose}/output",
        )

    def _clear_output(self, purpose: int, name: str) -> None:
        _input, output = self._actor_dirs(purpose)
        path = output / name
        if path.exists() or path.is_symlink():
            path.unlink()

    @staticmethod
    def _read_single_json_line(path: Path, *, maximum_bytes: int = 512 * 1024) -> tuple[dict[str, object], bytes]:
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise OSError("receipt is not a regular non-symlink file")
            if metadata.st_size <= 1 or metadata.st_size > maximum_bytes:
                raise OSError("receipt size is outside the frozen bound")
            payload = path.read_bytes()
            if len(payload) != metadata.st_size or not payload.endswith(b"\n"):
                raise OSError("receipt framing differs")
            if payload.count(b"\n") != 1 or b"\x00" in payload:
                raise OSError("receipt is not exactly one JSON line")
            value = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(FAIL_CONTAINER, f"operation probe receipt is invalid: {exc}")
        if type(value) is not dict:
            _fail(FAIL_CONTAINER, "operation probe receipt is not a JSON object")
        if _canonical_json(value) != payload:
            _fail(FAIL_CONTAINER, "operation probe receipt is not canonical JSON")
        return value, payload

    def _operation_request_digest(
        self,
        *,
        purpose: int,
        operation: str,
        sequence: int,
        nonce: bytes,
    ) -> bytes:
        input_directory, _output = self._actor_dirs(purpose)
        rows: list[dict[str, object]] = []
        for root_text, directory_names, file_names in os.walk(
            input_directory, topdown=True, followlinks=False
        ):
            root = Path(root_text)
            root_metadata = root.lstat()
            if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
                _fail(FAIL_CONTAINER, "operation input tree contains an invalid directory")
            directory_names.sort()
            file_names.sort()
            for directory_name in directory_names:
                directory = root / directory_name
                metadata = directory.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    _fail(FAIL_CONTAINER, "operation input tree contains a symlink")
            for file_name in file_names:
                path = root / file_name
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    _fail(FAIL_CONTAINER, "operation input tree contains a non-regular file")
                digest = hashlib.sha256()
                descriptor = os.open(
                    path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    opened = os.fstat(descriptor)
                    if (
                        opened.st_dev != metadata.st_dev
                        or opened.st_ino != metadata.st_ino
                        or opened.st_size != metadata.st_size
                    ):
                        _fail(FAIL_CONTAINER, "operation input changed while it was bound")
                    while True:
                        chunk = os.read(descriptor, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                finally:
                    os.close(descriptor)
                rows.append({
                    "path": path.relative_to(input_directory).as_posix(),
                    "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
                    "size": metadata.st_size,
                    "sha256": digest.hexdigest(),
                })
        if (
            self._transaction_run_id is None
            or self._profile_digest is None
            or self._docker_daemon_binding is None
            or purpose not in self._containers
        ):
            _fail(FAIL_CONTAINER, "operation request identity is incomplete")
        body = {
            "schema": "hegel-phase3-m25-operation-request-binding/1",
            "basis_commit": self.basis_commit,
            "container_id": self._containers[purpose],
            "daemon_receipt_sha256": self._docker_daemon_binding.hex(),
            "input_rows": rows,
            "operation_id": operation,
            "operation_nonce_hex": nonce.hex(),
            "operation_sequence": sequence,
            "profile_sha256": self._profile_digest.hex(),
            "purpose_id": purpose,
            "run_id_hex": self._transaction_run_id.hex(),
        }
        return hashlib.sha256(
            b"HEGEL/M25/OPERATION_REQUEST_BINDING/V1\x00" + _canonical_json(body)
        ).digest()

    @staticmethod
    def _environment_from_rows(rows: object) -> dict[str, str]:
        if type(rows) is not list or any(type(row) is not str for row in rows):
            _fail(FAIL_CONTAINER, "container environment inspection is invalid")
        result: dict[str, str] = {}
        for row in rows:
            key, separator, value = row.partition("=")
            if not separator or not key or key in result:
                _fail(FAIL_CONTAINER, "container environment inspection is ambiguous")
            result[key] = value
        return result

    def _inspect_live_actor(self, purpose: int) -> dict[str, object]:
        if purpose not in self._containers or purpose not in self._container_names:
            _fail(FAIL_CONTAINER, "actor container identity is absent")
        inspected = self._docker("inspect", self._containers[purpose], timeout=60)
        try:
            rows = json.loads(inspected.stdout)
            if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
                raise TypeError
            row = rows[0]
            config = row["Config"]
            host = row["HostConfig"]
            state = row["State"]
            mounts = row["Mounts"]
            if not all(isinstance(value, Mapping) for value in (config, host, state)):
                raise TypeError
            if type(mounts) is not list:
                raise TypeError
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            _fail(FAIL_CONTAINER, f"live actor inspection is invalid: {exc}")
        assert self._root is not None
        assert self._profile_digest is not None
        assert self._transaction_run_id is not None
        assert self._docker_daemon_binding is not None
        images = self._profile.get("images")
        if not isinstance(images, Mapping):
            _fail(FAIL_CONTAINER, "actor image registry is absent")
        image_key = {
            1: "custodian", 2: "python_attester", 3: "rust_attester", 4: "policy_auditor"
        }[purpose]
        expected_image = images.get(image_key)
        if type(expected_image) is not str:
            _fail(FAIL_CONTAINER, "actor image is absent")
        expected_labels = {
            "hegel.m25.ceremony": self._ceremony_token,
            "hegel.m25.purpose": str(purpose),
            "hegel.m25.run": self._transaction_run_id.hex(),
            "hegel.m25.basis": self.basis_commit,
            "hegel.m25.profile_sha256": self._profile_digest.hex(),
            "hegel.m25.daemon_receipt_sha256": self._docker_daemon_binding.hex(),
        }
        labels = config.get("Labels")
        if not isinstance(labels, Mapping) or any(
            labels.get(key) != value for key, value in expected_labels.items()
        ) or {
            key for key in labels if str(key).startswith("hegel.m25.")
        } != set(expected_labels):
            _fail(FAIL_CONTAINER, "live actor labels differ from the transaction")
        # PID 1 is long lived and may spawn later key-bearing operations.  It
        # therefore receives only the report-safe environment.  The raw clone
        # path is injected into the short-lived operation probe process below,
        # where it is removed before any cryptographic or replay child runs.
        expected_base_environment = self._actor_environment(purpose)
        expected_pid1_command = [
            "-i",
            *(f"{key}={value}" for key, value in expected_base_environment.items()),
            "/bin/sleep",
            "2147483647",
        ]
        actor = self._root / f"purpose-{purpose}"
        expected_mounts: dict[str, tuple[str, str | None, bool]] = {
            "/state": ("volume", self._state_volumes[purpose], True),
            "/input": ("bind", (actor / "input").resolve(strict=True).as_posix(), False),
            "/output": ("bind", (actor / "output").resolve(strict=True).as_posix(), True),
        }
        if purpose == 1:
            expected_mounts["/custody"] = (
                "bind", self.custody_directory.resolve(strict=True).as_posix(), True
            )
        actual_mounts: dict[str, tuple[str, str | None, bool]] = {}
        for mount in mounts:
            if not isinstance(mount, Mapping):
                _fail(FAIL_CONTAINER, "live actor mount row is invalid")
            destination = mount.get("Destination")
            if type(destination) is not str or destination in actual_mounts:
                _fail(FAIL_CONTAINER, "live actor mount destination is ambiguous")
            mount_type = mount.get("Type")
            source = mount.get("Name") if mount_type == "volume" else mount.get("Source")
            actual_mounts[destination] = (str(mount_type), source, mount.get("RW") is True)
        tmpfs = host.get("Tmpfs")
        security = set(host.get("SecurityOpt") or ())
        seccomp_rows = [
            value for value in security
            if type(value) is str and value.startswith("seccomp=")
        ]
        try:
            committed_seccomp = (
                None
                if self._runtime_seccomp_path is None
                else json.loads(self._runtime_seccomp_path.read_bytes())
            )
            inspected_seccomp = (
                json.loads(seccomp_rows[0].partition("=")[2])
                if len(seccomp_rows) == 1
                else None
            )
        except (OSError, json.JSONDecodeError):
            committed_seccomp = None
            inspected_seccomp = None
        ulimits = host.get("Ulimits") or ()
        expected_image_digest = expected_image.rsplit(":", 1)[-1]
        checks = {
            "container_id_exact": row.get("Id") == self._containers[purpose],
            "container_name_exact": row.get("Name") == "/" + self._container_names[purpose],
            "running": state.get("Running") is True and int(state.get("Pid", 0)) > 0,
            "image_reference_exact": config.get("Image") == expected_image,
            "image_id_digest_exact": str(row.get("Image", "")).endswith(expected_image_digest),
            "user_nonroot_exact": config.get("User") == "65534:65534",
            "pid1_env_i_command_exact": config.get("Cmd") == expected_pid1_command,
            "entrypoint_exact": config.get("Entrypoint") == ["/usr/bin/env"],
            "network_none": host.get("NetworkMode") == "none",
            "read_only_root": host.get("ReadonlyRootfs") is True,
            "not_privileged": host.get("Privileged") is False,
            "capabilities_exact": (host.get("CapDrop") or ()) == ["ALL"] and not (host.get("CapAdd") or ()),
            "security_options_exact": len(security) == 2
            and "no-new-privileges" in security
            and len(seccomp_rows) == 1,
            "runtime_seccomp_exact": committed_seccomp is not None
            and inspected_seccomp == committed_seccomp,
            "resource_limits_exact": host.get("PidsLimit") == 64
            and host.get("Memory") == 512 * 1024 * 1024
            and host.get("MemorySwap") == 512 * 1024 * 1024,
            "nofile_exact": any(
                isinstance(row_value, Mapping)
                and row_value.get("Name") == "nofile"
                and row_value.get("Soft") == 64
                and row_value.get("Hard") == 64
                for row_value in ulimits
            ),
            "ipc_private": host.get("IpcMode") == "private",
            "tmpfs_private_exact": isinstance(tmpfs, Mapping)
            and set(tmpfs) == {"/tmp"}
            and all(
                token in str(tmpfs["/tmp"])
                for token in ("rw", "noexec", "nosuid", "nodev", "uid=65534", "gid=65534")
            ),
            "mount_set_exact": actual_mounts == expected_mounts,
        }
        self._inspect_and_validate_state_volume(
            purpose, self._state_volumes[purpose], self._state_volume_labels(purpose)
        )
        if not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            _fail(FAIL_CONTAINER, f"live actor host inspection failed: {failed}")
        evidence = {
            "checks": checks,
            "container_id": self._containers[purpose],
            "container_name": self._container_names[purpose],
            "host_pid": int(state["Pid"]),
            "image_ref": expected_image,
            "mount_destinations": sorted(actual_mounts),
            "purpose_id": purpose,
        }
        evidence["inspection_sha256"] = hashlib.sha256(
            _canonical_json(evidence)
        ).hexdigest()
        return evidence

    @staticmethod
    def _validate_namespace_rows(rows: object) -> dict[str, str]:
        if type(rows) is not dict or set(rows) != {"ipc", "mnt", "net", "pid", "uts"}:
            _fail(FAIL_CONTAINER, "operation probe namespace set differs")
        if any(
            type(value) is not str
            or re.fullmatch(r"[a-z]+:\[[1-9][0-9]*\]", value) is None
            for value in rows.values()
        ):
            _fail(FAIL_CONTAINER, "operation probe namespace identity is invalid")
        return dict(rows)

    @staticmethod
    def _validate_common_probe_fields(
        receipt: Mapping[str, object],
        *,
        expected_environment: Mapping[str, str],
        purpose: int,
    ) -> dict[str, str]:
        if receipt.get("environment") != dict(expected_environment):
            _fail(FAIL_CONTAINER, "Rust operation environment receipt differs")
        identity = receipt.get("identity")
        status = receipt.get("proc_status")
        filesystem = receipt.get("filesystem_probes")
        syscall_rows = receipt.get("syscall_probes")
        if not all(isinstance(row, Mapping) for row in (identity, status, filesystem)):
            _fail(FAIL_CONTAINER, "operation probe structural fields are invalid")
        assert isinstance(identity, Mapping)
        assert isinstance(status, Mapping)
        assert isinstance(filesystem, Mapping)
        expected_syscalls = {
            "socket(AF_INET, SOCK_STREAM)",
            "socket(AF_INET6, SOCK_STREAM)",
            "mount",
            "ptrace(PTRACE_TRACEME)",
            "bpf(BPF_MAP_CREATE)",
            "perf_event_open",
        }
        if (
            identity.get("uid") != 65534
            or identity.get("gid") != 65534
            or type(identity.get("pid")) is not int
            or identity["pid"] <= 1
            or any(status.get(name) != "0000000000000000" for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"))
            or status.get("NoNewPrivs") != 1
            or status.get("Seccomp") != 2
            or receipt.get("network_interfaces") != ["lo"]
            or type(syscall_rows) is not list
            or {row.get("probe_id") for row in syscall_rows if isinstance(row, Mapping)} != expected_syscalls
            or any(
                not isinstance(row, Mapping)
                or row.get("return_value") != -1
                or row.get("errno") != 1
                for row in syscall_rows
            )
            or filesystem.get("forbidden_paths_present") != []
            or filesystem.get("cross_purpose_paths_present") != []
            or not isinstance(filesystem.get("root_write"), Mapping)
            or filesystem["root_write"].get("denied") is not True
            or filesystem["root_write"].get("errno") not in {1, 13, 30}
            or not isinstance(filesystem.get("input_write"), Mapping)
            or filesystem["input_write"].get("denied") is not True
            or filesystem["input_write"].get("errno") not in {1, 13, 30}
            or receipt.get("open_fds") != [0, 1, 2]
            or receipt.get("purpose_id") != purpose
        ):
            _fail(FAIL_CONTAINER, "operation probe isolation evidence differs")
        return DockerCeremonyActorsV1._validate_namespace_rows(
            receipt.get("namespaces")
        )

    def _validate_python_operation_receipt(
        self,
        receipt: dict[str, object],
        payload: bytes,
        *,
        purpose: int,
        operation: str,
        sequence: int,
        nonce: bytes,
        request_digest: bytes,
        expected_environment: Mapping[str, str],
    ) -> dict[str, str]:
        expected_keys = {
            "schema", "implementation", "operation_id", "operation_sequence",
            "operation_nonce_hex", "operation_request_sha256", "purpose_id",
            "identity", "proc_status", "namespaces", "network_interfaces",
            "syscall_probes", "filesystem_probes", "operation_environment",
            "pid1_environment", "worker_open_fds", "pid1_open_fds",
            "cgroup_limits", "required_checks", "all_required_checks_passed",
            "receipt_sha256",
        }
        body = dict(receipt)
        claimed_hash = body.pop("receipt_sha256", None)
        if (
            set(receipt) != expected_keys
            or receipt.get("schema") != OPERATION_PROBE_SCHEMA
            or receipt.get("implementation") != "python-ctypes-in-process-v1"
            or receipt.get("operation_id") != operation
            or receipt.get("operation_sequence") != sequence
            or receipt.get("operation_nonce_hex") != nonce.hex()
            or receipt.get("operation_request_sha256") != request_digest.hex()
            or receipt.get("purpose_id") != purpose
            or receipt.get("operation_environment") != dict(expected_environment)
            or receipt.get("pid1_environment")
            != {key: expected_environment[key] for key in _ACTOR_BASE_ENVIRONMENT_KEYS}
            or receipt.get("worker_open_fds") != [0, 1, 2]
            or receipt.get("pid1_open_fds") != [0, 1, 2]
            or receipt.get("all_required_checks_passed") is not True
            or type(receipt.get("required_checks")) is not dict
            or not receipt["required_checks"]
            or not all(value is True for value in receipt["required_checks"].values())
            or type(claimed_hash) is not str
            or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed_hash
            or hashlib.sha256(payload).hexdigest() == "0" * 64
        ):
            _fail(FAIL_CONTAINER, "Python operation-bound receipt differs")
        # Independently replay the material fields instead of trusting the
        # worker's required_checks aggregate.
        common = {
            "environment": receipt["operation_environment"],
            "identity": receipt["identity"],
            "proc_status": receipt["proc_status"],
            "namespaces": receipt["namespaces"],
            "network_interfaces": receipt["network_interfaces"],
            "syscall_probes": receipt["syscall_probes"],
            "filesystem_probes": receipt["filesystem_probes"],
            "open_fds": receipt["worker_open_fds"],
            "purpose_id": purpose,
        }
        namespaces = self._validate_common_probe_fields(
            common, expected_environment=expected_environment, purpose=purpose
        )
        filesystem = receipt["filesystem_probes"]
        cgroup = receipt["cgroup_limits"]
        assert isinstance(filesystem, Mapping)
        custody_write_required = purpose == 1 and operation in {
            "seed-split-real", "seed-split-resume", "seed-split-synthetic", "complete-marker"
        }
        expected_mounts = sorted(
            ["/input", "/output", "/state", "/tmp"]
            + (["/custody"] if purpose == 1 else [])
        )
        custody_probe = filesystem.get("custody_write_or_null")
        if (
            filesystem.get("output_write") != {"succeeded": True, "errno": 0}
            or filesystem.get("state_write") != {"succeeded": True, "errno": 0}
            or filesystem.get("mount_destinations") != expected_mounts
            or filesystem.get("custody_present") is not (purpose == 1)
            or (custody_write_required and custody_probe != {"succeeded": True, "errno": 0})
            or (not custody_write_required and custody_probe is not None)
            or not isinstance(cgroup, Mapping)
            or cgroup.get("memory_max") != str(512 * 1024 * 1024)
            or cgroup.get("memory_swap_max") != "0"
            or cgroup.get("pids_max") != "64"
        ):
            _fail(FAIL_CONTAINER, "Python operation probe filesystem/cgroup differs")
        return namespaces

    def _validate_rust_operation_receipt(
        self,
        parent: dict[str, object],
        raw: dict[str, object],
        raw_payload: bytes,
        *,
        purpose: int,
        operation: str,
        sequence: int,
        nonce: bytes,
        request_digest: bytes,
        expected_environment: Mapping[str, str],
    ) -> dict[str, str]:
        expected_parent_keys = {
            "schema", "purpose_id", "operation_id", "operation_sequence",
            "operation_nonce_hex", "operation_request_sha256",
            "parent_environment_count", "parent_pid", "rust_probe_sha256",
        }
        expected_raw_keys = {
            "environment", "filesystem_probes", "identity", "implementation",
            "namespaces", "network_interfaces", "open_fds", "proc_status",
            "profile_id", "purpose_id", "schema", "syscall_probes",
        }
        if (
            set(parent) != expected_parent_keys
            or parent.get("schema") != RUST_OPERATION_PARENT_SCHEMA
            or parent.get("purpose_id") != purpose
            or parent.get("operation_id") != operation
            or parent.get("operation_sequence") != sequence
            or parent.get("operation_nonce_hex") != nonce.hex()
            or parent.get("operation_request_sha256") != request_digest.hex()
            or parent.get("parent_environment_count")
            != len(expected_environment) + len(_ACTOR_PRIVATE_ENVIRONMENT_KEYS)
            or type(parent.get("parent_pid")) is not int
            or parent["parent_pid"] <= 1
            or parent.get("rust_probe_sha256") != hashlib.sha256(raw_payload).hexdigest()
            or set(raw) != expected_raw_keys
            or raw.get("schema") != "hegel-container-actor-live-probe/1"
            or raw.get("implementation") != "rust-ffi-v1"
            or raw.get("profile_id")
            != "hegel-owner-accepted-container-technical-actors-v1"
        ):
            _fail(FAIL_CONTAINER, "Rust operation parent/probe receipt differs")
        return self._validate_common_probe_fields(
            raw, expected_environment=expected_environment, purpose=purpose
        )

    def _exec(
        self,
        purpose: int,
        operation: str,
        *,
        operation_request_digest_override: bytes | None = None,
        timeout_seconds: int = 240,
    ) -> Mapping[str, object]:
        if purpose not in _ACTOR_OPERATIONS_BY_PURPOSE or operation not in _ACTOR_OPERATIONS_BY_PURPOSE[purpose]:
            _fail(FAIL_CONTAINER, "purpose operation is not allowlisted")
        if operation != "qualify-only" and not self._live_actor_set_qualified:
            _fail(FAIL_CONTAINER, "all four live actors must qualify before sensitive operations")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 1800:
            _fail(FAIL_CONTAINER, "actor operation timeout is outside the allowed range")
        if operation_request_digest_override is not None and (
            purpose != 4
            or operation != "purpose4-parent-sign"
            or type(operation_request_digest_override) is not bytes
            or len(operation_request_digest_override) != 32
        ):
            _fail(FAIL_CONTAINER, "operation request digest override is not authorized")
        before = self._inspect_live_actor(purpose)
        sequence = self._operation_sequences[purpose] + 1
        nonce = secrets.token_bytes(16)
        request_digest = (
            operation_request_digest_override
            if operation_request_digest_override is not None
            else self._operation_request_digest(
                purpose=purpose,
                operation=operation,
                sequence=sequence,
                nonce=nonce,
            )
        )
        self._operation_sequences[purpose] = sequence
        _input, output = self._actor_dirs(purpose)
        receipt_path = output / f"operation-probe-{operation}.json"
        raw_rust_path = output / f"operation-rust-probe-{operation}.json"
        for path in (receipt_path, raw_rust_path):
            if path.exists() or path.is_symlink():
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    _fail(FAIL_CONTAINER, "prior operation receipt is not removable regular state")
                path.unlink()
        worker_by_purpose = {
            1: "/input/tools/phase3_m25_formal_actor_worker_v1.py",
            2: "/input/tools/phase3_m25_python_bridge_actor_worker_v1.py",
            4: "/input/tools/phase3_m25_parent_auditor_actor_worker_v1.py",
        }
        if purpose == 3:
            worker = [
                "/bin/sh",
                "/input/tools/phase3_m25_formal_rust_actor_worker_v1.sh",
                operation,
            ]
        elif purpose == 4 and operation == "purpose4-parent-sign":
            worker = [
                "/usr/local/bin/python3",
                "-I",
                "-B",
                "/input/runtime/tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py",
                operation,
            ]
        else:
            worker = ["/usr/local/bin/python3", worker_by_purpose[purpose], operation]
        environment = self._actor_environment(
            purpose,
            operation=operation,
            operation_sequence=sequence,
            operation_nonce=nonce,
            operation_request_digest=request_digest,
        )
        launch_environment = self._actor_launch_environment(
            purpose,
            operation=operation,
            operation_sequence=sequence,
            operation_nonce=nonce,
            operation_request_digest=request_digest,
        )
        command = self._docker_command(
            "exec",
            "--user=65534:65534",
            self._containers[purpose],
            "/usr/bin/env",
            "-i",
            *(f"{key}={value}" for key, value in launch_environment.items()),
            *worker,
        )
        assert self._docker_control_plane is not None
        _run(
            command,
            timeout=timeout_seconds,
            environment=self._docker_control_plane.environment,
        )
        receipt, receipt_payload = self._read_single_json_line(receipt_path)
        if purpose == 3:
            raw, raw_payload = self._read_single_json_line(raw_rust_path)
            namespaces = self._validate_rust_operation_receipt(
                receipt,
                raw,
                raw_payload,
                purpose=purpose,
                operation=operation,
                sequence=sequence,
                nonce=nonce,
                request_digest=request_digest,
                expected_environment=environment,
            )
            live_probe_sha256 = hashlib.sha256(raw_payload).hexdigest()
        else:
            namespaces = self._validate_python_operation_receipt(
                receipt,
                receipt_payload,
                purpose=purpose,
                operation=operation,
                sequence=sequence,
                nonce=nonce,
                request_digest=request_digest,
                expected_environment=environment,
            )
            live_probe_sha256 = hashlib.sha256(receipt_payload).hexdigest()
        after = self._inspect_live_actor(purpose)
        if after != before:
            _fail(FAIL_CONTAINER, "actor host identity changed across one operation")
        host_receipt: dict[str, object] = {
            "schema": HOST_OPERATION_RECEIPT_SCHEMA,
            "basis_commit": self.basis_commit,
            "container_id": self._containers[purpose],
            "daemon_receipt_sha256": self._docker_daemon_binding.hex(),
            "host_inspection": before,
            "live_probe_receipt_sha256": live_probe_sha256,
            "namespaces": namespaces,
            "operation_id": operation,
            "operation_nonce_hex": nonce.hex(),
            "operation_request_sha256": request_digest.hex(),
            "operation_sequence": sequence,
            "operation_sequence_scope": self.operation_sequence_scope,
            "purpose_id": purpose,
            "same_live_container_before_after": True,
        }
        host_receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json(host_receipt)
        ).hexdigest()
        frozen = MappingProxyType(host_receipt)
        self._operation_probe_receipts.append(frozen)
        if operation == "qualify-only":
            self._live_actor_probe_receipts[purpose] = frozen
        return MappingProxyType(dict(receipt))

    def keygen(self, purpose: int) -> bytes:
        if purpose in self._public_keys:
            return self._public_keys[purpose]
        self._clear_output(purpose, "ed25519-public.der")
        self._exec(purpose, "keygen-resume" if self._recovery_mode else "keygen")
        _input, output = self._actor_dirs(purpose)
        public, key_id = parse_ed25519_spki_der_v1((output / "ed25519-public.der").read_bytes())
        self._public_keys[purpose] = public
        self._key_ids[purpose] = key_id
        return public

    def _set_custody_owner(self, uid: int, gid: int) -> None:
        images = self._profile["images"]
        assert isinstance(images, Mapping)
        image = images["custodian"]
        if type(image) is not str or "@sha256:" not in image:
            _fail(FAIL_CONTAINER, "custodian image is not digest pinned")
        if self._runtime_seccomp_path is None:
            _fail(FAIL_CONTAINER, "frozen runtime seccomp snapshot is absent")
        assert self._docker_control_plane is not None
        try:
            source_metadata = self.custody_directory.lstat()
        except OSError as exc:
            _fail(FAIL_CONTAINER, f"custody ownership source is absent: {exc}")
        if (
            not stat.S_ISDIR(source_metadata.st_mode)
            or stat.S_IMODE(source_metadata.st_mode) != 0o700
        ):
            _fail(FAIL_CONTAINER, "custody ownership source inode differs")
        source_uid, source_gid = source_metadata.st_uid, source_metadata.st_gid
        metadata_program = (
            "import json,os,stat;rows=[];"
            "entries=sorted(os.scandir('/custody'),key=lambda x:x.name);"
            "[(lambda s,e:rows.append({'name':e.name,'kind':'regular' if "
            "stat.S_ISREG(s.st_mode) else 'other','mode_octal':format(stat.S_IMODE(s.st_mode),'04o'),"
            "'uid':s.st_uid,'gid':s.st_gid}))(e.stat(follow_symlinks=False),e) for e in entries];"
            "print(json.dumps(rows,sort_keys=True,separators=(',',':')))"
        )

        def list_metadata(list_uid: int, list_gid: int) -> list[dict[str, object]]:
            command = self._docker_command(
                "run", "--rm", "--pull=never", "--network=none",
                "--ipc=private", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                f"--security-opt=seccomp={self._runtime_seccomp_path}",
                f"--user={list_uid}:{list_gid}", "--pids-limit=16",
                "--memory=64m", "--memory-swap=64m", "--ulimit=nofile=32:32",
                (
                    "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=4m,"
                    f"uid={list_uid},gid={list_gid},mode=0700"
                ),
                (
                    f"--mount=type=bind,src={self.custody_directory},"
                    "dst=/custody,readonly,bind-propagation=rprivate"
                ),
                "--entrypoint=/usr/local/bin/python3", str(image),
                "-c", metadata_program,
            )
            completed = _run(
                command,
                environment=self._docker_control_plane.environment,
                check=False,
                timeout=60,
            )
            if (
                completed.returncode != 0
                or completed.stderr
                or len(completed.stdout) > 16 * 1024
            ):
                _fail(FAIL_CONTAINER, "custody metadata-only lister failed")
            try:
                rows = json.loads(completed.stdout)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _fail(FAIL_CONTAINER, f"custody metadata listing is invalid: {exc}")
            if type(rows) is not list:
                _fail(FAIL_CONTAINER, "custody metadata listing is not an array")
            return rows

        rows = list_metadata(source_uid, source_gid)
        allowed_names = {
            "phase3_m25_ceremony.lock",
            "split_seed_instantiation.marker",
            "split_seed_instantiation.marker.complete.tmp",
            "split_seed_generation.intent",
            "split_seed_generation.complete",
            "split_master_seed.bin",
        }
        names: list[str] = []
        for row in rows:
            name = None if not isinstance(row, Mapping) else row.get("name")
            if (
                type(row) is not dict
                or set(row) != {"name", "kind", "mode_octal", "uid", "gid"}
                or type(name) is not str
                or (
                    name not in allowed_names
                    and re.fullmatch(
                        r"opaque-(run|ledger)-[0-9a-f]{32}\.reserved", name
                    )
                    is None
                )
                or row.get("kind") != "regular"
                or row.get("mode_octal") != "0600"
                or row.get("uid") != source_uid
                or row.get("gid") != source_gid
            ):
                _fail(FAIL_CONTAINER, "custody metadata row differs before ownership transfer")
            names.append(name)
        if (
            names != sorted(names)
            or len(names) != len(set(names))
            or "phase3_m25_ceremony.lock" not in names
            or len([name for name in names if name.startswith("opaque-run-")]) != 1
            or len([name for name in names if name.startswith("opaque-ledger-")]) != 1
        ):
            _fail(FAIL_CONTAINER, "custody ownership transfer path set differs")

        command_parts: list[str] = [
            "run", "--rm", "--pull=never", "--network=none", "--ipc=private",
            "--read-only", "--cap-drop=ALL", "--cap-add=CHOWN",
            "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._runtime_seccomp_path}",
            "--user=0:0", "--pids-limit=16", "--memory=64m", "--memory-swap=64m",
            "--ulimit=nofile=32:32",
        ]
        target_paths: list[str] = []
        for index, name in enumerate(names):
            target = f"/targets/item-{index}"
            command_parts.append(
                f"--mount=type=bind,src={self.custody_directory / name},dst={target},bind-propagation=rprivate"
            )
            target_paths.append(target)
        command_parts.append(
            f"--mount=type=bind,src={self.custody_directory},dst=/target-root,bind-propagation=rprivate"
        )
        command_parts.extend((
            "--entrypoint=/usr/bin/env", str(image), "-i",
            "PATH=/usr/local/bin:/usr/bin:/bin", "/bin/chown",
            f"{uid}:{gid}", *target_paths, "/target-root",
        ))
        _run(
            self._docker_command(*command_parts),
            environment=self._docker_control_plane.environment,
        )
        metadata = self.custody_directory.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != uid
            or metadata.st_gid != gid
            or (metadata.st_dev, metadata.st_ino)
            != (source_metadata.st_dev, source_metadata.st_ino)
        ):
            _fail(FAIL_CONTAINER, "custody ownership transfer did not take effect")
        after_rows = list_metadata(uid, gid)
        if (
            [row.get("name") for row in after_rows if isinstance(row, Mapping)]
            != names
            or any(
                type(row) is not dict
                or row.get("uid") != uid
                or row.get("gid") != gid
                or row.get("kind") != "regular"
                or row.get("mode_octal") != "0600"
                for row in after_rows
            )
        ):
            _fail(FAIL_CONTAINER, "custody ownership transfer metadata differs")

    def _handoff_custody_to_actor(self) -> None:
        self._set_custody_owner(65534, 65534)
        self._custody_handed_off = True

    def _reclaim_custody_from_actor(self) -> None:
        if not self._custody_handed_off:
            return
        self._set_custody_owner(os.getuid(), os.getgid())
        self._custody_handed_off = False

    def seed_split(self) -> tuple[bytes, bytes]:
        if 1 not in self._key_ids:
            _fail(FAIL_CUSTODY, "purpose-1 keygen must precede the marker")
        _marker_path, pending = create_pending_marker_v1(
            secret_state_directory=self.custody_directory,
            split_version_digest=SPLIT_VERSION_DIGEST,
            custodian_key_id=self._key_ids[1],
            created_at_unix_seconds=self.timestamp,
        )
        self._pending_marker = pending
        self._handoff_custody_to_actor()
        for name in ("python-split-frame.bin", "rust-split-frame.bin", "split-mode.txt"):
            self._clear_output(1, name)
        self._exec(1, "seed-split-real")
        _input, output = self._actor_dirs(1)
        if (output / "split-mode.txt").read_text(encoding="ascii") != "REAL_FIRST_GENESIS\n":
            _fail(FAIL_CUSTODY, "purpose 1 did not report real first genesis")
        return (
            (output / "python-split-frame.bin").read_bytes(),
            (output / "rust-split-frame.bin").read_bytes(),
        )

    def resume_pending_seed_split(self) -> tuple[bytes, bytes]:
        if not self._recovery_mode:
            _fail(FAIL_CUSTODY, "pending seed recovery requires explicit recovery mode")
        if 1 not in self._key_ids:
            _fail(FAIL_CUSTODY, "purpose-1 key recovery must precede seed recovery")
        marker_path = self.custody_directory / "split_seed_instantiation.marker"
        marker = read_marker_snapshot_v1(marker_path)
        if marker.state != "PENDING" or marker.custodian_key_id != self._key_ids[1]:
            _fail(FAIL_CUSTODY, "PENDING marker does not bind the recovered purpose-1 key")
        self._pending_marker = marker
        self._handoff_custody_to_actor()
        for name in ("python-split-frame.bin", "rust-split-frame.bin", "split-mode.txt"):
            self._clear_output(1, name)
        self._exec(1, "seed-split-resume")
        _input, output = self._actor_dirs(1)
        mode = (output / "split-mode.txt").read_text(encoding="ascii")
        if mode not in {
            "REAL_PENDING_RESUME\n",
            "REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT\n",
        }:
            _fail(FAIL_CUSTODY, "purpose 1 did not report explicit pending recovery")
        return (
            (output / "python-split-frame.bin").read_bytes(),
            (output / "rust-split-frame.bin").read_bytes(),
        )

    def resume_post_stage_seed_split(self) -> tuple[bytes, bytes]:
        """Resume only a fully durable seed; this path can never call CSPRNG."""

        paths = (
            self.custody_directory / "split_seed_generation.intent",
            self.custody_directory / "split_master_seed.bin",
            self.custody_directory / "split_seed_generation.complete",
        )
        if any(path.is_symlink() or not path.is_file() for path in paths):
            _fail(FAIL_CUSTODY, "post-stage seed state is incomplete; redraw is forbidden")
        frames = self.resume_pending_seed_split()
        assert self._root is not None
        mode_path = self._root / "purpose-1/output/split-mode.txt"
        if mode_path.read_bytes() != b"REAL_PENDING_RESUME\n":
            _fail(FAIL_CUSTODY, "post-stage recovery attempted a non-resume seed mode")
        return frames

    def verify_seed_custody_commitment_v1(
        self, expected_commitment: bytes
    ) -> Mapping[str, object]:
        """Run a one-shot keyless, read-only verifier over retained custody."""

        if type(expected_commitment) is not bytes or len(expected_commitment) != 32:
            _fail(FAIL_CUSTODY, "expected seed commitment must be 32 bytes")
        self._ensure_local_runtime()
        if not self._profile:
            self._load_committed_profile()
        assert self._root is not None
        if self._runtime_seccomp_path is None:
            control = self._root / "control"
            control.mkdir(mode=0o700, exist_ok=True)
            self._runtime_seccomp_path = control / SECCOMP_PATH.name
            self._write_snapshot(
                self._runtime_seccomp_path,
                self._git_blob(
                    "Hegel Machine/config/phase3_internal_actor_seccomp_v1.json"
                ),
                0o444,
            )
        verifier = self._root / "seed-custody-verifier/verify.py"
        self._write_snapshot(
            verifier,
            self._git_blob(
                "Hegel Machine/tools/phase3_m25_seed_custody_verifier_v1.py"
            ),
            0o444,
        )
        images = self._profile.get("images")
        image = None if not isinstance(images, Mapping) else images.get("custodian")
        if type(image) is not str or "@sha256:" not in image:
            _fail(FAIL_CONTAINER, "keyless verifier image is not digest pinned")
        assert self._runtime_seccomp_path is not None
        custody_rows = (
            (self.custody_directory, 0o700, stat.S_ISDIR),
            (
                self.custody_directory / "split_seed_generation.intent",
                0o600,
                stat.S_ISREG,
            ),
            (
                self.custody_directory / "split_master_seed.bin",
                0o600,
                stat.S_ISREG,
            ),
            (
                self.custody_directory / "split_seed_generation.complete",
                0o600,
                stat.S_ISREG,
            ),
        )
        owner: tuple[int, int] | None = None
        for path, mode, kind in custody_rows:
            try:
                metadata = path.lstat()
            except OSError as exc:
                _fail(FAIL_CUSTODY, f"keyless verifier custody metadata is absent: {exc}")
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not kind(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != mode
            ):
                _fail(FAIL_CUSTODY, "keyless verifier custody inode policy differs")
            row_owner = (metadata.st_uid, metadata.st_gid)
            if owner is None:
                owner = row_owner
            elif row_owner != owner:
                _fail(FAIL_CUSTODY, "keyless verifier custody ownership is not uniform")
        assert owner is not None
        verifier_uid, verifier_gid = owner
        if owner not in {
            (os.geteuid(), os.getegid()),
            (65534, 65534),
        }:
            _fail(
                FAIL_CUSTODY,
                "keyless verifier custody owner is outside the frozen host/nobody policy",
            )
        seccomp_digest = hashlib.sha256(
            self._runtime_seccomp_path.read_bytes()
        ).hexdigest()
        verifier_digest = hashlib.sha256(verifier.read_bytes()).hexdigest()
        assert self._docker_control_plane is not None
        owner_policy_id = "EXACT_UNIFORM_CURRENT_OWNER_HOST_OR_65534_V1"
        command_policy = {
            "schema": "hegel-phase3-m25-keyless-verifier-command-policy/1",
            "docker_control_plane_binding": dict(
                self._docker_control_plane.binding
            ),
            "image_ref": image,
            "seccomp_sha256": seccomp_digest,
            "verifier_tool_sha256": verifier_digest,
            "network_mode": "none",
            "ipc_mode": "private",
            "read_only_rootfs": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "pids_limit": 16,
            "memory_bytes": 64 * 1024 * 1024,
            "memory_swap_bytes": 64 * 1024 * 1024,
            "nofile_soft_hard": "32:32",
            "custody_mount_read_only": True,
            "verifier_mount_read_only": True,
            "state_mount_present": False,
            "private_key_mount_present": False,
            "owner_policy_id": owner_policy_id,
            "entrypoint": "/usr/local/bin/python3",
            "stdout_limit_bytes": 8192,
            "timeout_seconds": 120,
        }
        command_policy_digest = hashlib.sha256(
            _canonical_json(command_policy)
        ).hexdigest()
        command = self._docker_command(
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--ipc=private",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--security-opt=seccomp={self._runtime_seccomp_path}",
            f"--user={verifier_uid}:{verifier_gid}",
            "--pids-limit=16",
            "--memory=64m",
            "--memory-swap=64m",
            "--ulimit=nofile=32:32",
            (
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=4m,"
                f"uid={verifier_uid},gid={verifier_gid},mode=0700"
            ),
            f"--mount=type=bind,src={self.custody_directory},dst=/custody,readonly,bind-propagation=rprivate",
            f"--mount=type=bind,src={verifier},dst=/input/verify.py,readonly,bind-propagation=rprivate",
            "--env",
            f"HEGEL_EXPECTED_SEED_COMMITMENT_HEX={expected_commitment.hex()}",
            "--env",
            f"HEGEL_VERIFIER_NUMERIC_UID={verifier_uid}",
            "--env",
            f"HEGEL_VERIFIER_NUMERIC_GID={verifier_gid}",
            "--entrypoint=/usr/local/bin/python3",
            image,
            "/input/verify.py",
        )
        assert self._docker_control_plane is not None
        command_digest = hashlib.sha256(
            _canonical_json({"argv": tuple(command)})
        ).hexdigest()
        completed = _run(
            command,
            environment=self._docker_control_plane.environment,
            check=False,
            timeout=120,
        )
        if (
            completed.returncode != 0
            or completed.stderr
            or len(completed.stdout) > 8192
        ):
            _fail(FAIL_CUSTODY, "keyless seed-custody verifier failed")
        try:
            inner_receipt = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail(FAIL_CUSTODY, f"keyless verifier receipt is invalid: {exc}")
        inner_required = {
            "schema",
            "verified",
            "seed_commitment_hex",
            "seed_length_bytes",
            "seed_intent_sha256",
            "completion_receipt_sha256",
            "raw_seed_read_inside_keyless_verifier",
            "raw_seed_exported",
            "private_key_mount_present",
            "state_mount_present",
            "verifier_numeric_uid",
            "verifier_numeric_gid",
            "custody_artifacts_owned_by_verifier_identity",
            "receipt_sha256",
        }
        if type(inner_receipt) is not dict:
            _fail(FAIL_CUSTODY, "keyless verifier receipt is not an object")
        inner_body = dict(inner_receipt)
        inner_claimed_hash = inner_body.pop("receipt_sha256", None)
        if (
            set(inner_receipt) != inner_required
            or _canonical_json(inner_receipt) != completed.stdout
            or inner_receipt.get("schema")
            != SEED_CUSTODY_INNER_VERIFICATION_SCHEMA
            or inner_receipt.get("verified") is not True
            or inner_receipt.get("seed_commitment_hex")
            != expected_commitment.hex()
            or inner_receipt.get("seed_length_bytes") != 32
            or type(inner_receipt.get("seed_intent_sha256")) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}", str(inner_receipt.get("seed_intent_sha256"))
            )
            is None
            or type(inner_receipt.get("completion_receipt_sha256")) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(inner_receipt.get("completion_receipt_sha256")),
            )
            is None
            or inner_receipt.get("raw_seed_read_inside_keyless_verifier") is not True
            or inner_receipt.get("raw_seed_exported") is not False
            or inner_receipt.get("private_key_mount_present") is not False
            or inner_receipt.get("state_mount_present") is not False
            or inner_receipt.get("verifier_numeric_uid") != verifier_uid
            or inner_receipt.get("verifier_numeric_gid") != verifier_gid
            or inner_receipt.get(
                "custody_artifacts_owned_by_verifier_identity"
            )
            is not True
            or inner_claimed_hash
            != hashlib.sha256(_canonical_json(inner_body)).hexdigest()
        ):
            _fail(FAIL_CUSTODY, "keyless verifier returned a differing commitment")
        receipt: dict[str, object] = {
            **{
                key: value
                for key, value in inner_receipt.items()
                if key not in {"schema", "receipt_sha256"}
            },
            "schema": SEED_CUSTODY_VERIFICATION_SCHEMA,
            "inner_receipt_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "verifier_tool_sha256": verifier_digest,
            "docker_command_argv_sha256": command_digest,
            "docker_command_policy_sha256": command_policy_digest,
            "docker_image_ref": image,
            "docker_seccomp_sha256": seccomp_digest,
            "docker_daemon_receipt_sha256": self._docker_daemon_binding.hex(),
            "docker_control_plane_binding_sha256": hashlib.sha256(
                _canonical_json(dict(self._docker_control_plane.binding))
            ).hexdigest(),
            "docker_network_mode": "none",
            "docker_ipc_mode": "private",
            "docker_read_only_rootfs": True,
            "docker_stdout_limit_bytes": 8192,
            "docker_timeout_seconds": 120,
            "custody_owner_policy_id": owner_policy_id,
            "incarnation_fields_nonidentity": True,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json(receipt)
        ).hexdigest()
        assert_public_payload_contains_no_secret_fields(receipt)
        return MappingProxyType(receipt)

    def prospective_complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        pending = self._pending_marker
        if pending is None or pending.state != "PENDING":
            _fail(FAIL_CUSTODY, "PENDING marker snapshot is absent")
        if type(seed_manifest_root) is not bytes or len(seed_manifest_root) != 32:
            _fail(FAIL_CUSTODY, "seed manifest root must be 32 bytes")
        return MarkerSnapshot(
            "COMPLETE",
            pending.split_version_digest,
            seed_manifest_root,
            pending.custodian_key_id,
            pending.created_at_unix_seconds,
        )

    def complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        input_dir, output = self._actor_dirs(1)
        (input_dir / "seed-manifest-root.bin").write_bytes(seed_manifest_root)
        self._clear_output(1, "complete-marker.json")
        self._exec(1, "complete-marker")
        value = json.loads((output / "complete-marker.json").read_bytes())
        snapshot = MarkerSnapshot(
            state=value["state"],
            split_version_digest=bytes.fromhex(value["split_version_digest_hex"]),
            seed_commitment_manifest_root=bytes.fromhex(value["seed_commitment_manifest_root_hex_or_null"]),
            custodian_key_id=bytes.fromhex(value["custodian_key_id_hex"]),
            created_at_unix_seconds=value["created_at_unix_seconds"],
        )
        return snapshot

    def _signature(self, purpose: int) -> bytes:
        _input, output = self._actor_dirs(purpose)
        signature = (output / "ed25519-signature.bin").read_bytes()
        if len(signature) != 64:
            _fail(FAIL_CONTAINER, "actor signature is not 64 bytes")
        return signature

    def sign_object(self, name: str, fields: Mapping[str, object]) -> bytes:
        input_dir, _output = self._actor_dirs(1)
        root = candidate_content_root(name, fields)
        request = {
            "schema_name": name,
            "formal_cbor_hex": encode_formal_object(name, fields).hex(),
            "expected_root_hex": root.hex(),
        }
        (input_dir / "authorized-object.json").write_bytes(_canonical_json(request))
        signing_preimage = input_dir / "signing-preimage.bin"
        signing_preimage.write_bytes(
            external_signature_preimage_v1(OBJECT_TAGS[name], root, 1, 0)
        )
        self._clear_output(1, "ed25519-signature.bin")
        try:
            self._exec(1, "purpose1-authorized-sign")
            return self._signature(1)
        finally:
            try:
                signing_preimage.unlink()
            except FileNotFoundError:
                pass

    def sign_parent(self, evidence: ParentAbsenceAuditEvidence, fields: Mapping[str, object]) -> bytes:
        input_dir, output = self._actor_dirs(4)
        expected_root = candidate_content_root(
            "ParentManifestAbsenceAttestationV2", fields
        )
        if 4 not in self._key_ids or 4 not in self._public_keys:
            _fail(FAIL_CONTAINER, "purpose-4 keygen must precede detached replay")
        if (
            self._purpose4_snapshot_manifest is None
            or self._purpose4_runtime_bundle is None
            or self._purpose4_snapshot_path is None
        ):
            _fail(FAIL_CONTAINER, "purpose-4 detached runtime is absent")
        images = self._profile.get("images")
        if not isinstance(images, Mapping) or type(images.get("policy_auditor")) is not str:
            _fail(FAIL_CONTAINER, "purpose-4 actor image binding is absent")
        runtime_inventory = self._purpose4_runtime_bundle.get("runtime_inventory")
        source_bindings = self._purpose4_runtime_bundle.get(
            "runtime_source_bindings"
        )
        if not isinstance(runtime_inventory, Mapping) or not isinstance(
            source_bindings, Mapping
        ):
            _fail(FAIL_CONTAINER, "purpose-4 runtime binding is incomplete")
        try:
            request = build_purpose4_keybearing_request_v1(
                basis_commit=self.basis_commit,
                actor_image_ref=str(images["policy_auditor"]),
                snapshot_manifest=self._purpose4_snapshot_manifest,
                runtime_inventory=runtime_inventory,
                runtime_source_bindings=source_bindings,
                audited_at_unix_seconds=self.timestamp,
                expected_local_key_id=self._key_ids[4],
            )
        except Purpose4KeyBearingError as exc:
            _fail(FAIL_CONTAINER, f"purpose-4 detached request is invalid: {exc}")
        request_path = input_dir / "purpose4-keybearing-request.json"
        response_path = output / "purpose4-keybearing-detached-response.json"
        forbidden_host_oracles = (
            input_dir / "parent-audit-replay.json",
            input_dir / "signing-preimage.bin",
        )
        if any(path.exists() or path.is_symlink() for path in forbidden_host_oracles):
            _fail(FAIL_CONTAINER, "legacy purpose-4 host signing oracle is present")
        if request_path.exists() or request_path.is_symlink():
            _fail(FAIL_CONTAINER, "purpose-4 detached request already exists")
        request_payload = purpose4_canonical_json_v1(request)
        self._write_snapshot(request_path, request_payload, 0o444)
        self._clear_output(4, response_path.name)
        exact_operation_probe = self._exec(
            4,
            "purpose4-parent-sign",
            operation_request_digest_override=bytes.fromhex(
                str(request["request_sha256"])
            ),
            timeout_seconds=1800,
        )
        response, response_payload = self._read_single_json_line(
            response_path,
            maximum_bytes=PURPOSE4_KEYBEARING_MAX_RESPONSE_BYTES,
        )
        if purpose4_canonical_json_v1(response) != response_payload:
            _fail(FAIL_CONTAINER, "purpose-4 response is not canonical JSON")
        if response.get("operation_probe_receipt") != dict(exact_operation_probe):
            _fail(
                FAIL_CONTAINER,
                "purpose-4 response embeds a different operation probe receipt",
            )
        assert self._root is not None
        verifier_directory = self._root / "purpose-4-host-verifier"
        try:
            verifier_directory.mkdir(mode=0o700)
            verifier = make_openssl_ed25519_verifier_v1(verifier_directory)
            result = validate_purpose4_keybearing_response_v1(
                response,
                request=request,
                signature_verifier=verifier,
            )
        except (OSError, BridgeDagReplayError, Purpose4KeyBearingError) as exc:
            _fail(FAIL_CONTAINER, f"purpose-4 detached response is invalid: {exc}")
        if (
            result.attestation_root != expected_root
            or result.audit_bundle_root != evidence.audit_bundle_root
            or result.signer_public_key != self._public_keys[4]
            or result.signer_key_id != self._key_ids[4]
        ):
            _fail(FAIL_CONTAINER, "purpose-4 detached result differs from host replay")
        return result.signature

    def sign_bridge(
        self,
        purpose: int,
        fields: Mapping[str, object],
        replay_package: bytes,
    ) -> bytes:
        input_dir, _output = self._actor_dirs(purpose)
        if purpose not in (1, 2, 3):
            _fail(FAIL_CONTAINER, "bridge purpose must be 1,2,3")
        if type(replay_package) is not bytes or not replay_package:
            _fail(FAIL_CONTAINER, "bridge full-DAG replay package is absent")
        root = candidate_content_root("BridgeReplayStatementV1", fields)
        forbidden_host_oracles = tuple(
            input_dir / name
            for name in (
                "bridge-statement.cbor",
                "expected-root.bin",
                "signing-preimage.bin",
                "rust-decode-request.json",
                "rust-decode-response.json",
                "rust-content-hash-request.json",
                "rust-content-hash-response.json",
            )
        )
        if any(path.exists() or path.is_symlink() for path in forbidden_host_oracles):
            _fail(FAIL_CONTAINER, "legacy bridge host signing oracle is present")
        package_path = input_dir / "bridge-dag-package.cbor"
        if package_path.exists() or package_path.is_symlink():
            _fail(FAIL_CONTAINER, "bridge replay package already exists")
        assert self._root is not None
        verifier_directory = self._root / f"bridge-host-verifier-p{purpose}"
        try:
            verifier_directory.mkdir(mode=0o700)
            verifier = make_openssl_ed25519_verifier_v1(verifier_directory)
            expected_result = replay_bridge_dag_package_v1(
                replay_package,
                allow_authoritative=True,
                signature_verifier=verifier,
            )
        except (OSError, BridgeDagReplayError) as exc:
            _fail(FAIL_CONTAINER, f"host bridge full-DAG replay failed: {exc}")
        if (
            expected_result.purpose_id != purpose
            or expected_result.authoritative is not True
            or expected_result.eligible_to_sign_bridge_statement is not True
            or expected_result.purpose1_signature_verified is not (purpose != 1)
            or expected_result.split_membership_recomputed is not False
            or expected_result.bridge_statement_root != root
        ):
            _fail(FAIL_CONTAINER, "bridge package replay result is not signer-eligible")
        self._write_snapshot(package_path, replay_package, 0o444)
        self._clear_output(purpose, "ed25519-signature.bin")
        self._clear_output(purpose, "bridge-dag-replay-receipt.json")
        if purpose in {1, 2}:
            self._exec(purpose, "bridge-replay-sign-python")
        else:
            rust_replayer = input_dir / "rust-bridge-dag-replay"
            if (
                rust_replayer.is_symlink()
                or not rust_replayer.is_file()
                or not (rust_replayer.stat().st_mode & 0o111)
            ):
                _fail(FAIL_CONTAINER, "bound Rust bridge DAG replayer is absent")
            self._exec(3, "bridge-replay-sign-rust")
        _input, output = self._actor_dirs(purpose)
        receipt_path = output / "bridge-dag-replay-receipt.json"
        try:
            receipt_metadata = receipt_path.lstat()
            receipt_payload = receipt_path.read_bytes()
        except OSError as exc:
            _fail(FAIL_CONTAINER, f"bridge actor replay receipt is absent: {exc}")
        if (
            stat.S_ISLNK(receipt_metadata.st_mode)
            or not stat.S_ISREG(receipt_metadata.st_mode)
            or len(receipt_payload) != receipt_metadata.st_size
        ):
            _fail(FAIL_CONTAINER, "bridge actor replay receipt inode differs")
        try:
            validate_bridge_actor_replay_receipt_v1(
                receipt_payload,
                expected_result=expected_result,
                expected_implementation=(
                    "rust-full-dag-replay-v1"
                    if purpose == 3
                    else "python-full-dag-replay-v1"
                ),
                require_authoritative=True,
            )
            signature = self._signature(purpose)
            verifier(
                self._public_keys[purpose],
                signature,
                bridge_attestation_signature_preimage_v1(root, purpose, 0),
            )
        except (KeyError, BridgeDagReplayError) as exc:
            _fail(FAIL_CONTAINER, f"bridge actor result validation failed: {exc}")
        return signature


@dataclass(slots=True)
class SyntheticCeremonyActorsV1(CeremonyActorsV1):
    """Deterministic, in-process negative control that can never be promoted."""

    private_keys: Mapping[int, object]
    split_frame: bytes
    marker: MarkerSnapshot
    authoritative: bool = False

    @classmethod
    def create(cls, split_frame: bytes, timestamp: int) -> "SyntheticCeremonyActorsV1":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        keys = {purpose: Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"hegel-m25-synthetic-key-{purpose}".encode()).digest()) for purpose in (1,2,3,4)}
        return cls(
            MappingProxyType(keys),
            split_frame,
            MarkerSnapshot("PENDING", SPLIT_VERSION_DIGEST, None, bytes(16), timestamp),
        )

    def keygen(self, purpose: int) -> bytes:
        from cryptography.hazmat.primitives import serialization

        key = self.private_keys[purpose]
        return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)  # type: ignore[union-attr]

    def seed_split(self) -> tuple[bytes, bytes]:
        return self.split_frame, self.split_frame

    def resume_pending_seed_split(self) -> tuple[bytes, bytes]:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actor has no pending recovery authority")

    def prepare_post_stage_pending_recovery(self, run_id: bytes) -> None:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actor has no post-stage recovery authority")

    def resume_post_stage_seed_split(self) -> tuple[bytes, bytes]:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actor has no post-stage recovery authority")

    def recover_complete_private_state_and_verify_absent(
        self, run_id: bytes, marker: MarkerSnapshot
    ) -> None:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actor has no post-stage recovery authority")

    def prospective_complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        return MarkerSnapshot(
            "COMPLETE",
            self.marker.split_version_digest,
            seed_manifest_root,
            self.marker.custodian_key_id,
            self.marker.created_at_unix_seconds,
        )

    def complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        self.marker = self.prospective_complete_marker(seed_manifest_root)
        return self.marker

    def close_and_verify_absent(self) -> None:
        return None

    def stop_for_recovery_and_verify_absent(self) -> None:
        return None

    def destroy_private_state_and_verify_absent(self) -> None:
        return None

    def authorize_private_state_destruction(self, marker: MarkerSnapshot) -> None:
        if marker.state != "COMPLETE":
            _fail(FAIL_CUSTODY, "synthetic marker is not COMPLETE")

    def sign_object(self, name: str, fields: Mapping[str, object]) -> bytes:
        root = candidate_content_root(name, fields)
        tag = OBJECT_TAGS[name]
        return self.private_keys[1].sign(external_signature_preimage_v1(tag, root, 1, 0))  # type: ignore[union-attr]

    def sign_parent(self, evidence: ParentAbsenceAuditEvidence, fields: Mapping[str, object]) -> bytes:
        replay_parent_absence_audit_v1(evidence)
        root = candidate_content_root("ParentManifestAbsenceAttestationV2", fields)
        return self.private_keys[4].sign(external_signature_preimage_v1(OBJECT_TAGS["ParentManifestAbsenceAttestationV2"], root, 4, 0))  # type: ignore[union-attr]

    def sign_bridge(
        self,
        purpose: int,
        fields: Mapping[str, object],
        replay_package: bytes,
    ) -> bytes:
        root = candidate_content_root("BridgeReplayStatementV1", fields)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        def verify(candidate: bytes, signature: bytes, message: bytes) -> None:
            Ed25519PublicKey.from_public_bytes(candidate).verify(signature, message)

        replayed = replay_bridge_dag_package_v1(
            replay_package,
            allow_authoritative=False,
            signature_verifier=verify,
        )
        if replayed.purpose_id != purpose or replayed.bridge_statement_root != root:
            raise ValueError("synthetic bridge replay package differs")
        return self.private_keys[purpose].sign(bridge_attestation_signature_preimage_v1(root, purpose, 0))  # type: ignore[union-attr]


def _role_binding_fields(
    basis: FormalStaticBasisV1, *, role_id: int, split_root: bytes,
    custodian_root: bytes, continuity_root: bytes, parent_root: bytes,
    timestamp: int,
) -> dict[str, object]:
    static_name = (
        "DslRoleBindingManifestV1/OUTSIDE_TARGET"
        if role_id == 1
        else "DslRoleBindingManifestV1/IN_LANGUAGE_NULL"
    )
    fields = {
        **dict(basis.preseed_manifest_static_fields[static_name]),
        "split_binding_manifest_root": split_root,
        "custodian_binding_manifest_root": custodian_root,
        "seed_continuity_manifest_root": continuity_root,
        "parent_manifest_absence_attestation_root_or_null": parent_root,
        "created_at_unix_seconds": timestamp,
    }
    encode_formal_object("DslRoleBindingManifestV1", fields)
    return fields


def _build_gate_inputs_and_sign_v1(
    *, basis: FormalStaticBasisV1, parent: ParentAbsenceAuditEvidence,
    actor_report: Mapping[str, object], errata_report: Mapping[str, object],
    python_static_receipt: Mapping[str, object], rust_static_receipt: Mapping[str, object],
    execution_binding_roots: Mapping[str, bytes], actors: CeremonyActorsV1,
    timestamp: int, run_id: bytes, ledger_id: bytes, trust_id: bytes,
    frozen_actor_trust: ActorPublicKeysV1 | None = None,
    frozen_split_frames: tuple[bytes, bytes] | None = None,
    fault_injector: Callable[[str], None] | None = None,
) -> GateEvidenceInputsV1:
    """Construct the frozen DAG and obtain only purpose-authorized signatures."""

    static_roots = validate_dual_static_replay_receipts_v1(
        basis, python_static_receipt, rust_static_receipt
    )
    if frozen_actor_trust is None:
        public_keys = {purpose: actors.keygen(purpose) for purpose in (1, 2, 3, 4)}
        actor_trust = build_actor_trust_v1(
            public_keys=public_keys,
            timestamp=timestamp,
            basis_commit=basis.basis_commit,
            trust_genesis_id=trust_id,
        )
    else:
        actor_trust = build_actor_trust_v1(
            public_keys=frozen_actor_trust.public_keys,
            timestamp=timestamp,
            basis_commit=basis.basis_commit,
            trust_genesis_id=trust_id,
        )
        if actor_trust != frozen_actor_trust:
            _fail(FAIL_CUSTODY, "frozen actor trust differs before formal DAG construction")
    if frozen_split_frames is None:
        python_frame, rust_frame = actors.seed_split()
    else:
        if (
            type(frozen_split_frames) is not tuple
            or len(frozen_split_frames) != 2
            or any(type(frame) is not bytes for frame in frozen_split_frames)
        ):
            _fail(FAIL_CUSTODY, "frozen split-frame pair is invalid")
        python_frame, rust_frame = frozen_split_frames
    if fault_injector is not None:
        fault_injector("after_seed_split_frames")
    split = require_full_split_response_agreement_v2(python_frame, rust_frame)
    commit_wire = git_sha1_commit_id(bytes.fromhex(basis.basis_commit))

    parent_fields = build_parent_absence_attestation_fields_v2(
        parent, auditor_key_id=actor_trust.key_ids[4], audited_at_unix_seconds=timestamp
    )
    parent_root = candidate_content_root("ParentManifestAbsenceAttestationV2", parent_fields)
    parent_sig = actors.sign_parent(parent, parent_fields)
    if fault_injector is not None:
        fault_injector("after_parent_attestation_signature")

    seed_fields = {
        "split_contract_root": basis.roots["split_contract_root"],
        "target_bundle_root": basis.roots["target_bundle_root"],
        "split_seed_commitment_digest": split.seed_commitment,
        "seed_length_bytes": 32,
        "rng_profile_id_digest": id_digest_v1("hegel-os-csprng-v1"),
        "kdf_profile_id_digest": id_digest_v1("hegel-hkdf-sha256-split-v1"),
        "commitment_profile_id_digest": id_digest_v1("hegel-split-seed-commitment-v1"),
        "custodian_key_id": actor_trust.key_ids[1],
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    seed_root = candidate_content_root("SplitSeedCommitmentManifestV1", seed_fields)
    # The evidence is built against the exact prospective COMPLETE snapshot,
    # but no marker transition is performed here.  The executor may call the
    # mutating operation only after this whole DAG has been serialized,
    # durably staged, reloaded, and prospectively replayed.
    marker = actors.prospective_complete_marker(seed_root)
    intents, registry_records, snapshots = _opaque_registry(
        run_id=run_id, ledger_id=ledger_id, seed_root=seed_root,
        trust_root=actor_trust.trust_genesis_root, timestamp=timestamp,
        commit_wire=commit_wire,
    )
    ledger_fields = {
        "ledger_id": ledger_id,
        "sequence_number": 0,
        "previous_record_root_or_null": None,
        "event_type_id": 1,
        "actor_key_id": actor_trust.key_ids[1],
        "subject_manifest_root": seed_root,
        "revealed_artifact_root_or_null": None,
        "authorization_root_or_null": None,
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    ledger_root = candidate_content_root("HiddenAccessLedgerRecordV1", ledger_fields)
    core_fields = {
        "custodian_key_id": actor_trust.key_ids[1],
        "custodian_public_key_32_bytes": actor_trust.public_keys[1],
        "custodian_key_epoch": 0,
        "responsibility_bitmask": 0b011111,
        "valid_from_unix_seconds": timestamp,
        "valid_until_unix_seconds_or_null": None,
        "replacement_policy_root": candidate_content_root("ReplacementPolicyV1", actor_trust.replacement_policy_fields),
        "repository_commit_id": commit_wire,
    }
    core_root = candidate_content_root("CustodianBindingCoreV1", core_fields)
    continuity_fields = dict(basis.preseed_manifest_static_fields["SeedContinuityManifestV1"])
    continuity_fields.update({
        "current_seed_commitment_manifest_root": seed_root,
        "parent_manifest_absence_attestation_root": parent_root,
        "hidden_access_ledger_genesis_root": ledger_root,
        "custodian_binding_core_root": core_root,
        "instantiated_at_unix_seconds": timestamp,
    })
    continuity_root = candidate_content_root("SeedContinuityManifestV1", continuity_fields)
    custodian_fields = {
        "custodian_key_id": actor_trust.key_ids[1],
        "custodian_public_key_32_bytes": actor_trust.public_keys[1],
        "custodian_key_epoch": 0,
        "responsibility_bitmask": 0b011111,
        "split_seed_commitment_manifest_root": seed_root,
        "hidden_access_ledger_genesis_root": ledger_root,
        "seed_continuity_manifest_root": continuity_root,
        "valid_from_unix_seconds": timestamp,
        "valid_until_unix_seconds_or_null": None,
        "replacement_policy_root": core_fields["replacement_policy_root"],
        "repository_commit_id": commit_wire,
    }
    custodian_root = candidate_content_root("CustodianBindingManifestV1", custodian_fields)
    split_binding_fields = {
        **dict(basis.preseed_manifest_static_fields["SplitBindingManifestV1"]),
        "split_seed_commitment_manifest_root": seed_root,
        "seed_continuity_manifest_root": continuity_root,
        "outside_target_discovery_root": split.roots["outside_discovery_split_root"],
        "outside_target_validation_root": split.roots["outside_validation_split_root"],
        "outside_target_sealed_root": split.roots["outside_sealed_split_root"],
        "null_control_discovery_root": split.roots["null_discovery_split_root"],
        "null_control_validation_root": split.roots["null_validation_split_root"],
        "null_control_sealed_root": split.roots["null_sealed_split_root"],
        "hidden_access_ledger_genesis_root": ledger_root,
        "hidden_access_ledger_head_root": ledger_root,
        "created_at_unix_seconds": timestamp,
    }
    split_binding_root = candidate_content_root("SplitBindingManifestV1", split_binding_fields)
    outside_fields = _role_binding_fields(
        basis, role_id=1, split_root=split_binding_root, custodian_root=custodian_root,
        continuity_root=continuity_root, parent_root=parent_root, timestamp=timestamp,
    )
    null_fields = _role_binding_fields(
        basis, role_id=2, split_root=split_binding_root, custodian_root=custodian_root,
        continuity_root=continuity_root, parent_root=parent_root, timestamp=timestamp,
    )
    outside_root = candidate_content_root("DslRoleBindingManifestV1", outside_fields)
    null_root = candidate_content_root("DslRoleBindingManifestV1", null_fields)
    shrink_fields = dict(basis.preseed_manifest_static_fields["DslShrinkTransitionFormalV1"])
    shrink_fields.update({
        "outside_target_binding_manifest_root": outside_root,
        "null_control_binding_manifest_root": null_root,
        "split_binding_manifest_root": split_binding_root,
        "custodian_binding_manifest_root": custodian_root,
        "seed_continuity_manifest_root": continuity_root,
        "created_at_unix_seconds": timestamp,
    })
    shrink_root = candidate_content_root("DslShrinkTransitionFormalV1", shrink_fields)

    signed_objects = (
        ("SplitSeedCommitmentManifestV1", seed_fields),
        ("CustodianBindingManifestV1", custodian_fields),
        ("SeedContinuityManifestV1", continuity_fields),
        ("HiddenAccessLedgerRecordV1", ledger_fields),
    )
    external_envelopes: list[tuple[int, Mapping[str, object]]] = []
    for signature_index, (name, fields) in enumerate(signed_objects, start=1):
        root = candidate_content_root(name, fields)
        signature = actors.sign_object(name, fields)
        external_envelopes.append((1, build_single_signature_envelope_fields_v1(
            enclosed_object_tag=OBJECT_TAGS[name], enclosed_manifest_root=root,
            created_at_unix_seconds=timestamp, signer_key_id=actor_trust.key_ids[1],
            signature=signature,
        )))
        if fault_injector is not None:
            fault_injector(f"after_purpose1_object_signature_{signature_index}")
    external_envelopes.append((4, build_single_signature_envelope_fields_v1(
        enclosed_object_tag=OBJECT_TAGS["ParentManifestAbsenceAttestationV2"],
        enclosed_manifest_root=parent_root, created_at_unix_seconds=timestamp,
        signer_key_id=actor_trust.key_ids[4], signature=parent_sig,
    )))
    external_bundle_fields = _bundle_rows(external_envelopes)
    external_bundle_root = candidate_content_root("AttestationBundleV1", external_bundle_fields)
    final_snapshot_root = candidate_content_root("OpaqueIdRegistrySnapshotV1", snapshots[1])
    candidate_fields = {
        "run_id": run_id,
        **dict(basis.m3_candidate_static_fields),
        "approval_manifest_root": basis.roots["approval_manifest_root"],
        "shrink_transition_root": shrink_root,
        "outside_target_binding_manifest_root": outside_root,
        "null_control_binding_manifest_root": null_root,
        "split_binding_manifest_root": split_binding_root,
        "custodian_binding_manifest_root": custodian_root,
        "seed_continuity_manifest_root": continuity_root,
        "custodian_attestation_bundle_root": external_bundle_root,
        "parent_absence_attestation_root": parent_root,
        "hidden_access_ledger_genesis_root": ledger_root,
        "hidden_access_ledger_head_root": ledger_root,
        "opaque_id_registry_snapshot_root": final_snapshot_root,
        "actor_trust_genesis_root": actor_trust.trust_genesis_root,
        **dict(split.roots),
        **dict(execution_binding_roots),
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    candidate_root = candidate_content_root("M3ExecutionCandidateV1", candidate_fields)
    bridge_fields = {
        "run_id": run_id,
        "diagnostic_formal_bridge_root": basis.roots["diagnostic_formal_bridge_root"],
        "m3_execution_candidate_root": candidate_root,
        "child_dsl_spec_root": basis.roots["child_dsl_spec_root"],
        "child_freeze_root": basis.roots["child_freeze_root"],
        "actor_trust_genesis_root": actor_trust.trust_genesis_root,
        "opaque_id_registry_snapshot_root": final_snapshot_root,
    }
    bridge_root = candidate_content_root("BridgeReplayStatementV1", bridge_fields)
    bridge_node_inputs = BridgeDagNodeBuildInputsV1(
        basis=basis,
        candidate_fields=candidate_fields,
        dynamic_object_fields=MappingProxyType({
            "shrink_transition_root": shrink_fields,
            "outside_target_binding_manifest_root": outside_fields,
            "null_control_binding_manifest_root": null_fields,
            "split_binding_manifest_root": split_binding_fields,
            "custodian_binding_manifest_root": custodian_fields,
            "seed_continuity_manifest_root": continuity_fields,
            "hidden_access_ledger_genesis_root": ledger_fields,
            "hidden_access_ledger_head_root": ledger_fields,
        }),
        external_attestation_bundle_fields=external_bundle_fields,
        parent_attestation_fields=parent_fields,
        final_opaque_snapshot_fields=snapshots[1],
        actor_trust_fields=actor_trust.trust_genesis_fields,
        outside_typed_rows=generate_odd_role_rows_v1(),
        null_typed_rows=generate_sink_role_rows_v1(),
        sealed_split_roots=MappingProxyType({
            name: split.roots[name]
            for name in (
                "outside_discovery_split_root",
                "outside_validation_split_root",
                "outside_sealed_split_root",
                "null_discovery_split_root",
                "null_validation_split_root",
                "null_sealed_split_root",
            )
        }),
        m3_execution_fields=M3ExecutionBindingContractFieldsV1(
            python_implementation_binding_fields=basis.objects[
                "python_m3_implementation_binding"
            ],
            rust_implementation_binding_fields=basis.objects[
                "rust_m3_implementation_binding"
            ],
            traversal_contract_fields=basis.objects["traversal_contract"],
            bucket_accounting_contract_fields=basis.objects[
                "bucket_accounting_contract"
            ],
            program_archive_contract_fields=basis.objects[
                "program_archive_contract"
            ],
            output_archive_contract_fields=basis.objects[
                "output_archive_contract"
            ],
            state_machine_contract_fields=basis.objects[
                "state_machine_contract"
            ],
        ),
    )
    bridge_signatures: dict[int, bytes] = {}
    try:
        purpose1_package = build_bridge_dag_replay_package_from_inputs_v1(
            BridgeDagPackageBuildInputsV1(
                node_inputs=bridge_node_inputs,
                purpose_id=1,
                bridge_statement_fields=bridge_fields,
                purpose1_actor_key_manifest_fields=actor_trust.manifests[0],
                purpose1_bridge_signature=None,
                authority=actors.authoritative,
            )
        )
        bridge_signatures[1] = actors.sign_bridge(
            1, bridge_fields, purpose1_package
        )
        if fault_injector is not None:
            fault_injector("after_bridge_signature_p1")
        for purpose in (2, 3):
            package = build_bridge_dag_replay_package_from_inputs_v1(
                BridgeDagPackageBuildInputsV1(
                    node_inputs=bridge_node_inputs,
                    purpose_id=purpose,
                    bridge_statement_fields=bridge_fields,
                    purpose1_actor_key_manifest_fields=actor_trust.manifests[0],
                    purpose1_bridge_signature=bridge_signatures[1],
                    authority=actors.authoritative,
                )
            )
            bridge_signatures[purpose] = actors.sign_bridge(
                purpose, bridge_fields, package
            )
            if fault_injector is not None:
                fault_injector(f"after_bridge_signature_p{purpose}")
    except BridgeDagNodeBuildError as exc:
        _fail(FAIL_PREFLIGHT, f"complete bridge DAG construction failed: {exc}")
    bridge_envelopes = tuple(
        (
            purpose,
            build_single_signature_envelope_fields_v1(
                enclosed_object_tag=OBJECT_TAGS["BridgeReplayStatementV1"],
                enclosed_manifest_root=bridge_root,
                created_at_unix_seconds=timestamp,
                signer_key_id=actor_trust.key_ids[purpose],
                signature=bridge_signatures[purpose],
            ),
        )
        for purpose in (1, 2, 3)
    )
    bridge_bundle_fields = _bundle_rows(bridge_envelopes)
    bridge_bundle_root = candidate_content_root("AttestationBundleV1", bridge_bundle_fields)
    execution_fields = {
        "run_id": run_id,
        "m3_execution_candidate_root": candidate_root,
        "bridge_replay_statement_root": bridge_root,
        "bridge_attestation_bundle_root": bridge_bundle_root,
        "actor_trust_genesis_root": actor_trust.trust_genesis_root,
        "opaque_id_registry_snapshot_root": final_snapshot_root,
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    execution_root = candidate_content_root("M3ExecutionManifestV2", execution_fields)
    run_fields = {
        "run_id": run_id,
        "execution_manifest_root": execution_root,
        "initial_state_id": 0,
        **{f"{name}_or_null": None for name in M3_RUN_OUTPUT_ROOTS},
        "created_at_unix_seconds": timestamp,
        "repository_commit_id": commit_wire,
    }
    encode_formal_object("M3RunGenesisV1", run_fields)
    typed_roots = {
        name: basis.roots[name] for name in (
            "outside_target_universe_root", "outside_target_truth_root",
            "null_control_universe_root", "null_control_truth_root",
        )
    }
    canonical_bindings = (
        ("NormativeApprovalManifestV1", basis.preseed_manifest_static_fields["NormativeApprovalManifestV1"]),
        ("SplitBindingManifestV1", split_binding_fields),
        ("CustodianBindingManifestV1", custodian_fields),
        ("SeedContinuityManifestV1", continuity_fields),
        ("DslShrinkTransitionFormalV1", shrink_fields),
        ("DslRoleBindingManifestV1", outside_fields),
        ("DslRoleBindingManifestV1", null_fields),
    )
    return GateEvidenceInputsV1(
        basis_commit=basis.basis_commit,
        actor_qualification_report=actor_report,
        errata_qualification_report=errata_report,
        marker_snapshot=marker,
        actor_key_manifests=actor_trust.manifests,
        replacement_policy_fields=actor_trust.replacement_policy_fields,
        trust_genesis_fields=actor_trust.trust_genesis_fields,
        split_seed_commitment_fields=seed_fields,
        ledger_genesis_fields=ledger_fields,
        parent_top_level_path_rows=parent.top_level_path_rows,
        parent_history_rows=parent.history_rows,
        parent_touched_rows=parent.touched_path_rows_by_history_row,
        parent_legacy_rows=parent.legacy_source_rows,
        parent_audit_bundle_fields=parent.audit_bundle_fields,
        parent_attestation_fields=parent_fields,
        external_envelopes=tuple(external_envelopes),
        external_bundle_fields=external_bundle_fields,
        canonical_binding_objects=canonical_bindings,
        python_static_roots=static_roots,
        rust_static_roots=static_roots,
        python_typed_roots=typed_roots,
        rust_typed_roots=typed_roots,
        python_split_frame=python_frame,
        rust_split_frame=rust_frame,
        opaque_registration_intents=intents,
        opaque_registry_records=registry_records,
        opaque_registry_snapshots=snapshots,
        execution_candidate_fields=candidate_fields,
        bridge_statement_fields=bridge_fields,
        bridge_envelopes=bridge_envelopes,
        bridge_bundle_fields=bridge_bundle_fields,
        execution_manifest_fields=execution_fields,
        run_genesis_fields=run_fields,
    )


def execute_formal_container_ceremony_v1(
    *, basis_commit: str, actor_qualification_report: Mapping[str, object],
    errata_qualification_report: Mapping[str, object], custody_directory: Path,
    public_evidence_path: Path, public_promotion_path: Path,
    actors: CeremonyActorsV1,
    qualification_custody_directory: Path | None = None,
    fault_injector: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute after preflight; an authoritative run requires Docker actors.

    ``DockerCeremonyActorsV1`` is intentionally injected so fault tests can
    stop at every boundary.  The default CLI is the only place that constructs
    the real backend, and does so only for its explicit ``execute`` subcommand.
    """

    commit = _commit(basis_commit)
    basis = build_qualified_formal_static_basis_v1(commit)
    implementation_roots = require_formal_ceremony_ready_v1(basis)
    validate_ceremony_admission_v1(
        actor_qualification_report=actor_qualification_report,
        errata_qualification_report=errata_qualification_report,
        basis_commit=commit,
        committed_input_paths=REQUIRED_COMMIT_A_INPUTS,
    )
    if not actors.authoritative:
        _fail(FAIL_SYNTHETIC_PROMOTION, "synthetic actors cannot enter formal execution")
    if type(actors) is not DockerCeremonyActorsV1:
        _fail(
            FAIL_PREFLIGHT,
            "formal execute requires the sealed DockerCeremonyActorsV1 backend",
        )
    if not _PRESTAGE_RECOVERY_IMPLEMENTED:
        _fail(
            FAIL_PRESTAGE_RECOVERY_UNRESOLVED,
            "prestage recovery implementation has not passed its fault matrix",
        )
    if isinstance(actors, DockerCeremonyActorsV1):
        actors.validate_rust_replay_binding(basis)
        actors.validate_rust_bridge_dag_binding()
    actor_blockers = tuple(actors.unresolved_formal_blockers())
    if actor_blockers:
        _fail(
            FAIL_PREFLIGHT,
            "authoritative ceremony remains fail-closed: " + ",".join(actor_blockers),
        )
    bridge_report_id = actors.bridge_qualification_report_id_v1()
    if type(bridge_report_id) is not bytes or len(bridge_report_id) != 32:
        _fail(FAIL_PREFLIGHT, "authoritative bridge qualification report ID is absent")
    runtime_binding_fields = actors.prestage_runtime_binding_fields_v1(
        implementation_roots
    )
    if not isinstance(runtime_binding_fields, Mapping):
        _fail(FAIL_PREFLIGHT, "authoritative prestage runtime binding is absent")
    runtime_binding_fields = _validate_prestage_runtime_bindings_v1(
        dict(runtime_binding_fields)
    )
    if qualification_custody_directory is None:
        _fail(
            FAIL_PREFLIGHT,
            "formal execute requires a separate live-qualification custody directory",
        )
    _reject_caller_symlink_chain_v1(
        qualification_custody_directory,
        "live-qualification custody path",
    )
    qualification_custody = qualification_custody_directory.resolve()
    formal_custody = custody_directory.resolve()
    if qualification_custody == formal_custody:
        _fail(FAIL_PREFLIGHT, "qualification and formal custody must be distinct")
    try:
        qualification_custody.relative_to(formal_custody)
    except ValueError:
        pass
    else:
        _fail(FAIL_PREFLIGHT, "qualification custody may not be nested in formal custody")
    try:
        formal_custody.relative_to(qualification_custody)
    except ValueError:
        pass
    else:
        _fail(FAIL_PREFLIGHT, "formal custody may not be nested in qualification custody")
    if (
        not qualification_custody.is_dir()
        or qualification_custody.is_symlink()
        or stat.S_IMODE(qualification_custody.stat().st_mode) != 0o700
        or any(qualification_custody.iterdir())
    ):
        _fail(
            FAIL_CUSTODY,
            "live-qualification custody must be existing, empty and mode 0700",
        )
    if not custody_directory.is_dir() or any(custody_directory.iterdir()):
        _fail(FAIL_CUSTODY, "real execute requires an existing empty external custody directory")
    if (custody_directory.stat().st_mode & 0o777) != 0o700:
        _fail(FAIL_CUSTODY, "external custody directory must be mode 0700")
    live_actor_protocol = qualify_live_actor_protocol_admission_v1(
        basis_commit=commit,
        custody_directory=qualification_custody,
    )
    if any(qualification_custody.iterdir()):
        _fail(
            FAIL_CUSTODY,
            "live actor protocol admission did not restore empty qualification custody",
        )
    validate_commit_b_output_names_v1(public_evidence_path, public_promotion_path)
    python_receipt = build_python_static_replay_receipt_v1(basis)
    static_control_plane, static_daemon_binding = (
        actors.static_replay_control_plane_v1()
    )
    rust_receipt = run_rust_static_replay_receipt_v1(
        basis,
        control_plane=static_control_plane,
        daemon_receipt_binding=static_daemon_binding,
    )
    parent = generate_parent_absence_audit_v1(REPOSITORY_ROOT)
    replay_parent_absence_audit_v1(parent, repository=REPOSITORY_ROOT)
    timestamp_value = getattr(actors, "timestamp", None)
    timestamp = timestamp_value if type(timestamp_value) is int else int(time.time())
    run_id = secrets.token_bytes(16)
    ledger_id = secrets.token_bytes(16)
    trust_id = secrets.token_bytes(16)
    prestage_intent = build_prestage_intent_fields_v1(
        basis_commit=commit,
        run_id=run_id,
        ledger_id=ledger_id,
        created_at_unix_seconds=timestamp,
        trust_genesis_id=trust_id,
        actor_qualification_report=actor_qualification_report,
        errata_qualification_report=errata_qualification_report,
        rust_bridge_dag_qualification_report_sha256=bridge_report_id,
        live_actor_protocol_qualification_bundle_content_id=(
            live_actor_protocol.bundle_content_id
        ),
        qualification_only_key_ids=dict(
            live_actor_protocol.qualification_key_ids
        ),
        live_actor_protocol_qualification_bundle=dict(
            live_actor_protocol.report
        ),
        live_actor_protocol_qualification_canonical_bundle_bytes=(
            live_actor_protocol.canonical_bundle_bytes
        ),
        live_actor_protocol_daemon_receipt_binding=(
            live_actor_protocol.daemon_receipt_binding
        ),
        runtime_binding_fields=runtime_binding_fields,
    )
    if isinstance(actors, DockerCeremonyActorsV1):
        actors.bind_transaction_identity(run_id)
    transaction = FormalCeremonyTransactionV1(
        basis_commit=commit,
        custody_directory=custody_directory,
        public_evidence_path=public_evidence_path,
        public_promotion_path=public_promotion_path,
        run_id=run_id,
        ledger_id=ledger_id,
        prestage_intent_fields=prestage_intent,
        fault_injector=fault_injector,
    )
    actors_started = False
    actors_absent = False
    private_state_destruction_authorized = False
    try:
        transaction.reserve()
        actors.start()
        actors_started = True
        actors.validate_frozen_daemon_receipt_binding_v1(
            live_actor_protocol.daemon_receipt_binding
        )
        public_keys: dict[int, bytes] = {}
        for purpose in (1, 2, 3, 4):
            public_keys[purpose] = actors.keygen(purpose)
            transaction._fault(f"after_actor_keygen_{purpose}")
        actor_trust = build_actor_trust_v1(
            public_keys=public_keys,
            timestamp=timestamp,
            basis_commit=commit,
            trust_genesis_id=trust_id,
        )
        _require_formal_key_ids_disjoint_from_qualification_v1(
            actor_trust.key_ids,
            live_actor_protocol.qualification_key_ids,
        )
        checkpoint = transaction.persist_actor_trust_checkpoint_v1(actor_trust)
        validate_actor_trust_checkpoint_fields_v1(
            checkpoint,
            expected_actor_trust=actor_trust,
            basis_commit=commit,
            run_id=run_id,
            ledger_id=ledger_id,
            prestage_intent_sha256=hashlib.sha256(
                _canonical_json(prestage_intent)
            ).hexdigest(),
        )
        inputs = _build_gate_inputs_and_sign_v1(
            basis=basis, parent=parent, actor_report=actor_qualification_report,
            errata_report=errata_qualification_report,
            python_static_receipt=python_receipt, rust_static_receipt=rust_receipt,
            execution_binding_roots=implementation_roots, actors=actors,
            timestamp=timestamp, run_id=run_id,
            ledger_id=ledger_id, trust_id=trust_id,
            frozen_actor_trust=actor_trust,
            fault_injector=transaction._fault,
        )
        replay_payload = serialize_gate_evidence_inputs_v1(inputs)
        prospective_promotion = promote_gate_evidence_v1(
            evaluate_gates_15_24_v1(inputs)
        )
        transaction.stage_and_prospectively_replay(
            replay_payload, prospective_promotion
        )
        staged_seed_commitment = transaction._staged_seed_commitment
        if type(staged_seed_commitment) is not bytes or len(staged_seed_commitment) != 32:
            _fail(FAIL_CUSTODY, "durable staged seed commitment is absent")
        seed_verification_receipt = actors.verify_seed_custody_commitment_v1(
            staged_seed_commitment
        )
        transaction.record_seed_custody_verification_v1(
            seed_verification_receipt
        )
        seed_root = candidate_content_root(
            "SplitSeedCommitmentManifestV1", inputs.split_seed_commitment_fields
        )
        actual_marker = actors.complete_marker(seed_root)
        transaction.record_marker_complete(actual_marker, inputs.marker_snapshot)
        actors.authorize_actor_key_volume_destruction(actual_marker)
        private_state_destruction_authorized = True
        actors.destroy_actor_key_volumes_and_verify_absent()
        actors_absent = True
        transaction.record_actors_absent()
        transaction.publish()
        # Return only values reconstructed from the published bytes.  This is
        # intentionally after actor absence and publication journal completion.
        published_payload = json.loads(public_evidence_path.read_bytes())
        published_promotion = replay_public_gate_evidence_v1(published_payload)
        if _canonical_json(published_promotion) != public_promotion_path.read_bytes():
            _fail(FAIL_PUBLICATION, "published evidence/promotion replay differs")
        return published_payload, published_promotion
    except BaseException as original:
        if actors_started and not actors_absent:
            try:
                if private_state_destruction_authorized:
                    actors.destroy_actor_key_volumes_and_verify_absent()
                else:
                    actors.stop_for_recovery_and_verify_absent()
            except BaseException as cleanup:
                if isinstance(cleanup, FormalContainerExecutorError):
                    raise cleanup from original
                _fail(FAIL_CONTAINER, f"actor cleanup raised {type(cleanup).__name__}")
        raise
    finally:
        transaction.close_lock()


__all__ = [
    "ActorPublicKeysV1",
    "CeremonyActorsV1",
    "CeremonyReadinessV1",
    "DockerCeremonyActorsV1",
    "FormalCeremonyTransactionV1",
    "PendingCeremonyRecoveryV1",
    "FAIL_EXECUTION_BINDINGS",
    "FAIL_POST_STAGE_RECOVERY_UNRESOLVED",
    "FAIL_SYNTHETIC_PROMOTION",
    "FormalContainerExecutorError",
    "PUBLIC_REPLAY_SCHEMA",
    "SYNTHETIC_SCHEMA",
    "SyntheticCeremonyActorsV1",
    "build_actor_trust_v1",
    "acquire_pending_ceremony_recovery_v1",
    "abort_preseed_reserved_transaction_v1",
    "continue_pre_stage_pending_recovery_v1",
    "continue_post_stage_transaction_recovery_v1",
    "execute_formal_container_ceremony_v1",
    "inspect_formal_ceremony_readiness_v1",
    "load_gate_evidence_inputs_v1",
    "replay_public_gate_evidence_v1",
    "resume_pending_split_calculators_v1",
    "require_formal_ceremony_ready_v1",
    "serialize_gate_evidence_inputs_v1",
]

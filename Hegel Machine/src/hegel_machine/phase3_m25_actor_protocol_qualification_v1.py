"""Commit-A-bound live qualification of the isolated actor protocol.

This is deliberately *not* external genesis.  Four real, long-lived Docker
actors generate ephemeral purpose-private keys and execute the complete
purpose-4 detached-history path plus the purpose-1 -> purpose-2/purpose-3
bridge-DAG protocol.  The split response and prospective marker are fixed,
public synthetic fixtures supplied by this supervisor.  The Docker custodian
is never asked to instantiate, resume, read, or complete a seed.

The public diagnostic archive intentionally carries the complete replayable
qualification evidence: ephemeral public keys and signatures, operation and
bridge receipt bodies, the fixed public synthetic split fixture, exact actor
identities, a pre-cleanup destruction plan and a post-cleanup absence receipt.
Private keys, real seed material, authoritative formal roots and formal gate
evidence never leave their prohibited boundaries.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import threading
from types import MappingProxyType
from typing import Final, Mapping, NoReturn, Sequence

from .phase3_local_runtime_v1 import (
    LinuxLocalTemporaryDirectoryV1,
    Phase3LocalRuntimeError,
    build_local_docker_daemon_identity_receipt_v1,
    local_docker_daemon_receipt_binding_v1,
    prepare_local_docker_control_plane_v1,
    validate_linux_local_durable_custody_location_v1,
)
from .phase3_m25_bridge_dag_binary_qualification_v1 import (
    DEFAULT_RUST_BRIDGE_DAG_BINARY,
    DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
    BridgeDagBinaryQualificationError,
    load_qualified_rust_bridge_dag_binary_binding_v1,
)
from .phase3_m25_bridge_full_dag_replay_v1 import (
    BridgeDagReplayError,
    replay_bridge_dag_package_v1,
    validate_bridge_actor_replay_receipt_v1,
)
from .phase3_m25_container_ceremony_v1 import (
    SPLIT_RESPONSE_ROWS,
    SplitCalculatorPublicResponseV2,
    SplitRootCommitment,
    TECHNICAL_ACTOR_DISCLOSURE_V1,
    encode_split_calculator_public_frame_v2,
    require_full_split_response_agreement_v2,
)
from .phase3_m25_external_v1 import MarkerSnapshot
from .phase3_m25_purpose4_keybearing_detached_v1 import (
    Purpose4KeyBearingError,
    canonical_json_v1 as purpose4_canonical_json_v1,
    validate_purpose4_keybearing_response_v1,
)
from .phase3_m25_wire_v1 import bridge_attestation_signature_preimage_v1
from .phase3_m25_formal_container_executor_v1 import (
    REPOSITORY_ROOT,
    REQUIRED_COMMIT_A_INPUTS,
    HOST_OPERATION_RECEIPT_SCHEMA,
    SPLIT_VERSION_DIGEST,
    CeremonyActorsV1,
    DockerCeremonyActorsV1,
    FormalContainerExecutorError,
    _build_gate_inputs_and_sign_v1,
    build_python_static_replay_receipt_v1,
    generate_parent_absence_audit_v1,
    replay_parent_absence_audit_v1,
    require_formal_ceremony_ready_v1,
    run_rust_static_replay_receipt_v1,
)
from .phase3_m3_implementation_qualification_v1 import (
    M3ImplementationQualificationError,
    build_qualified_formal_static_basis_v1,
    load_committed_dual_golden_v1,
    validate_qualification_receipt_v1,
)
from .phase3_m25_formal_static_basis_v1 import DEFAULT_RUST_BINARY


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
SCHEMA_VERSION: Final = "hegel-phase3-m25-live-actor-protocol-qualification/2"
ARTIFACT_KIND: Final = "NON_FORMAL_PUBLIC_SYNTHETIC_PROTOCOL_QUALIFICATION"
STATUS: Final = "OFFLINE_COMMIT_A_LIVE_ACTOR_PROTOCOL_PASS"
CLAIM_LEVEL: Final = "TECHNICAL_ACTOR_PROTOCOL_QUALIFICATION_ONLY"
EVIDENCE_SCHEMA: Final = "hegel-phase3-m25-live-actor-protocol-evidence/2"
KEY_MANIFEST_SCHEMA: Final = "ProtocolQualificationActorKeyManifestV1"
STATEMENT_SCHEMA: Final = "hegel-phase3-m25-protocol-qualification-statement/1"
SIGNATURE_ENVELOPE_SCHEMA: Final = (
    "hegel-phase3-m25-protocol-qualification-signature-envelope/1"
)
CLEANUP_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-m25-protocol-qualification-cleanup-absence/1"
)
DESTRUCTION_PLAN_SCHEMA: Final = (
    "hegel-phase3-m25-protocol-qualification-destruction-plan/1"
)
FROZEN_QUALIFICATION_TIMESTAMP: Final = 1_735_689_600
REPORT_HASH_DOMAIN: Final = (
    b"HEGEL/M25/LIVE_ACTOR_PROTOCOL_QUALIFICATION_REPORT/V1\x00"
)
SOURCE_SET_HASH_DOMAIN: Final = (
    b"HEGEL/M25/LIVE_ACTOR_PROTOCOL_QUALIFICATION_SOURCE_SET/V1\x00"
)
EVIDENCE_HASH_DOMAIN: Final = (
    b"HEGEL/M25/LIVE_ACTOR_PROTOCOL_QUALIFICATION_EVIDENCE/V2\x00"
)
KEY_MANIFEST_HASH_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_ACTOR_KEY_MANIFEST/V1\x00"
)
STATEMENT_REQUEST_HASH_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_FINALIZE_REQUEST/V1\x00"
)
STATEMENT_HASH_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_STATEMENT/V1\x00"
)
STATEMENT_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_FINALIZE_SIGNATURE/V1\x00"
)
CLEANUP_RECEIPT_HASH_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_CLEANUP_ABSENCE/V1\x00"
)
DESTRUCTION_PLAN_HASH_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_DESTRUCTION_PLAN/V1\x00"
)
BUNDLE_AUTHORITY_HASH_DOMAIN: Final = (
    b"HEGEL/M25/PROTOCOL_QUALIFICATION_VALIDATED_BUNDLE/V1\x00"
)
LIVE_ADMISSION_TOKEN_MAC_DOMAIN: Final = (
    b"HEGEL/M25/LIVE_ACTOR_PROTOCOL_ADMISSION_TOKEN_MAC/V1\x00"
)
OPERATION_REQUEST_HASH_DOMAIN: Final = (
    b"HEGEL/M25/OPERATION_REQUEST_BINDING/V1\x00"
)
PUBLIC_SPLIT_DOMAIN: Final = b"HEGEL/M25/PUBLIC_SYNTHETIC_PROTOCOL_SPLIT/V1\x00"
MAX_BRIDGE_PACKAGE_BYTES: Final = 64 * 1024 * 1024

MODULE_REPOSITORY_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m25_actor_protocol_qualification_v1.py"
)
TOOL_REPOSITORY_PATH: Final = (
    "Hegel Machine/tools/phase3_m25_actor_protocol_qualification_v1.py"
)
FINALIZE_WORKER_REPOSITORY_PATH: Final = (
    "Hegel Machine/tools/phase3_m25_protocol_qualification_finalize_worker_v1.sh"
)
TEST_REPOSITORY_PATH: Final = (
    "Hegel Machine/tests/test_phase3_m25_actor_protocol_qualification_v1.py"
)
DOCUMENT_REPOSITORY_PATH: Final = (
    "Hegel Machine/docs/"
    "Hegel_Machine_Phase3A_M25_Live_Actor_Protocol_Qualification_v1.md"
)
DEFAULT_REPORT_PATH: Final = (
    PROJECT_ROOT
    / "artifacts/phase3_m25_external/"
    "phase3_m25_live_actor_protocol_qualification_v1.json"
)

FAIL_COMMIT: Final = "FAIL_M25_ACTOR_PROTOCOL_QUALIFICATION_COMMIT"
FAIL_CUSTODY: Final = "FAIL_M25_ACTOR_PROTOCOL_QUALIFICATION_CUSTODY"
FAIL_PROTOCOL: Final = "FAIL_M25_ACTOR_PROTOCOL_QUALIFICATION_PROTOCOL"
FAIL_CLEANUP: Final = "FAIL_M25_ACTOR_PROTOCOL_QUALIFICATION_CLEANUP"
FAIL_REPORT: Final = "FAIL_M25_ACTOR_PROTOCOL_QUALIFICATION_REPORT"

INDEPENDENCE_DISCLOSURE: Final = TECHNICAL_ACTOR_DISCLOSURE_V1

AUTHORITY_BOUNDARY: Final = MappingProxyType({
    "authority_class": ARTIFACT_KIND,
    "authoritative_formal_evidence": False,
    "real_seed_generated": False,
    "real_seed_accessed": False,
    "docker_seed_split_called": False,
    "docker_complete_marker_called": False,
    "custody_marker_created": False,
    "ephemeral_actor_private_keys_generated": True,
    "ephemeral_actor_signatures_generated": True,
    "ephemeral_private_keys_published": False,
    "ephemeral_public_keys_published": True,
    "ephemeral_signatures_published": True,
    "gate_evidence_published": False,
    "authoritative_formal_roots_generated": False,
    "synthetic_formal_shaped_roots_computed_in_memory": True,
    "formal_roots_published": False,
    "synthetic_protocol_objects_constructed_in_memory": True,
    "synthetic_protocol_objects_published": True,
    "m3_gates_before": 14,
    "m3_gates_after": 14,
    "m3_gate_delta": 0,
    "m3_state": "NOT_RUN",
    "m3_run_started": False,
})

EXPECTED_OPERATION_SEQUENCE: Final = (
    (1, 1, "qualify-only"),
    (2, 1, "qualify-only"),
    (3, 1, "qualify-only"),
    (4, 1, "qualify-only"),
    (1, 2, "keygen"),
    (2, 2, "keygen"),
    (3, 2, "keygen"),
    (4, 2, "keygen"),
    (4, 3, "purpose4-parent-sign"),
    (1, 3, "purpose1-authorized-sign"),
    (1, 4, "purpose1-authorized-sign"),
    (1, 5, "purpose1-authorized-sign"),
    (1, 6, "purpose1-authorized-sign"),
    (1, 7, "bridge-replay-sign-python"),
    (2, 3, "bridge-replay-sign-python"),
    (3, 3, "bridge-replay-sign-rust"),
)


class ActorProtocolQualificationError(RuntimeError):
    """Stable fail-closed error for this non-formal qualification."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise ActorProtocolQualificationError(code, detail)


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(FAIL_REPORT, f"diagnostic JSON is not canonical: {exc}")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _content_id(domain: bytes, value: object) -> str:
    return _sha256(domain + _canonical_json(value))


def _raw_base64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _decode_raw_base64(value: object, *, label: str, maximum: int) -> bytes:
    if type(value) is not str or len(value) > ((maximum + 2) // 3) * 4:
        _fail(FAIL_REPORT, f"{label} base64 size is outside the bound")
    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        _fail(FAIL_REPORT, f"{label} is not strict base64: {exc}")
    if len(payload) > maximum or _raw_base64(payload) != value:
        _fail(FAIL_REPORT, f"{label} base64 is non-canonical or oversized")
    return payload


def _restore_transport(value: object) -> object:
    if type(value) is dict:
        if set(value) == {"bytes_hex"}:
            encoded = value["bytes_hex"]
            if type(encoded) is not str or re.fullmatch(r"(?:[0-9a-f]{2})*", encoded) is None:
                _fail(FAIL_REPORT, "transport bytes_hex is malformed")
            return bytes.fromhex(encoded)
        return {str(key): _restore_transport(item) for key, item in value.items()}
    if type(value) is list:
        return tuple(_restore_transport(item) for item in value)
    if value is None or type(value) in {bool, int, str}:
        return value
    _fail(FAIL_REPORT, f"unsupported public transport value {type(value).__name__}")


def _verify_ed25519(public_key: bytes, signature: bytes, preimage: bytes) -> None:
    if len(public_key) != 32 or len(signature) != 64:
        _fail(FAIL_REPORT, "qualification Ed25519 key/signature length differs")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, preimage)
    except Exception as exc:
        _fail(FAIL_REPORT, f"qualification Ed25519 signature is invalid: {exc}")


def _signature_verifier(public_key: bytes, signature: bytes, preimage: bytes) -> None:
    _verify_ed25519(public_key, signature, preimage)


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        _fail(FAIL_REPORT, f"{label} is not an exact SHA-256")
    return value


def _require_commit(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        _fail(FAIL_COMMIT, "basis commit must be lowercase 40-hex")
    return value


def _public_split_component(label: str) -> bytes:
    return hashlib.sha256(PUBLIC_SPLIT_DOMAIN + label.encode("ascii")).digest()


PUBLIC_SYNTHETIC_SPLIT_FRAME: Final = encode_split_calculator_public_frame_v2(
    SplitCalculatorPublicResponseV2(
        seed_commitment=_public_split_component("commitment"),
        partitions=tuple(
            SplitRootCommitment(
                role_id,
                partition_id,
                row_count,
                _public_split_component(f"{role_id}/{partition_id}/{row_count}"),
            )
            for role_id, partition_id, row_count in SPLIT_RESPONSE_ROWS
        ),
    )
)
PUBLIC_SYNTHETIC_SPLIT_FRAME_SHA256: Final = _sha256(
    PUBLIC_SYNTHETIC_SPLIT_FRAME
)


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def reject_float(_value: str) -> NoReturn:
        _fail(FAIL_REPORT, "report contains a non-integer number")

    def reject_constant(_value: str) -> NoReturn:
        _fail(FAIL_REPORT, "report contains a non-JSON constant")

    def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(FAIL_REPORT, f"report contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii", "strict"),
            parse_float=reject_float,
            parse_constant=reject_constant,
            object_pairs_hook=exact_object,
        )
    except ActorProtocolQualificationError:
        raise
    except Exception as exc:
        _fail(FAIL_REPORT, f"report is not strict JSON: {exc}")
    if type(value) is not dict or _canonical_json(value) != payload:
        _fail(FAIL_REPORT, "report is not exactly one canonical JSON line")
    return value


def _git(arguments: Sequence[str], *, timeout: int = 120) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env={
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_PROTOCOL_FROM_USER": "0",
                "GIT_SSH_COMMAND": "false",
                "GIT_TERMINAL_PROMPT": "0",
                "HOME": "/nonexistent",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_COMMIT, f"Git operation failed: {type(exc).__name__}")
    if completed.returncode != 0:
        _fail(
            FAIL_COMMIT,
            "Git operation failed: "
            + completed.stderr.decode("utf-8", "replace")[-1000:],
        )
    return completed.stdout


def _repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except (OSError, ValueError) as exc:
        _fail(FAIL_COMMIT, f"Commit-A input escapes repository: {path}: {exc}")


def _source_set_paths_v1() -> tuple[str, ...]:
    paths = {
        _repository_relative(Path(path)) for path in REQUIRED_COMMIT_A_INPUTS
    }
    paths.update({
        MODULE_REPOSITORY_PATH,
        TOOL_REPOSITORY_PATH,
        FINALIZE_WORKER_REPOSITORY_PATH,
        TEST_REPOSITORY_PATH,
        DOCUMENT_REPOSITORY_PATH,
    })
    return tuple(sorted(paths))


def _commit_source_set_digest_from_git_v1(commit: str) -> tuple[str, int]:
    commit = _require_commit(commit)
    paths = _source_set_paths_v1()
    digest = hashlib.sha256()
    digest.update(SOURCE_SET_HASH_DOMAIN)
    for relative in paths:
        committed = _git(("show", f"{commit}:{relative}"))
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(hashlib.sha256(committed).digest())
    return "sha256:" + digest.hexdigest(), len(paths)


def _commit_source_set_digest_v1(commit: str) -> tuple[str, int]:
    commit = _require_commit(commit)
    head = _git(("rev-parse", "HEAD"), timeout=30).decode("ascii").strip()
    if head != commit:
        _fail(FAIL_COMMIT, "qualification basis must equal the exact current HEAD")
    paths = _source_set_paths_v1()
    committed_digest, committed_count = _commit_source_set_digest_from_git_v1(commit)
    for relative in paths:
        committed = _git(("show", f"{commit}:{relative}"))
        worktree_path = REPOSITORY_ROOT / relative
        try:
            metadata = worktree_path.lstat()
            worktree = worktree_path.read_bytes()
        except OSError as exc:
            _fail(FAIL_COMMIT, f"Commit-A input is absent: {relative}: {exc}")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(FAIL_COMMIT, f"Commit-A input is not a regular file: {relative}")
        if worktree != committed:
            _fail(FAIL_COMMIT, f"worktree input differs from Commit A: {relative}")
    return committed_digest, committed_count


def _hash_regular_file(path: Path, *, code: str = FAIL_PROTOCOL) -> str:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail(code, f"expected regular non-symlink file: {path.name}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or opened.st_size != metadata.st_size
            ):
                _fail(code, f"file identity changed while hashing: {path.name}")
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
    except OSError as exc:
        _fail(code, f"cannot hash {path.name}: {exc}")
    return "sha256:" + digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_custody_directory_lock_v1(path: Path, *, code: str) -> int:
    """Acquire the pre-existing directory mutex before observing its entries."""

    descriptor: int | None = None
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(code, "qualification custody is a symlink/non-directory")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            _fail(code, "qualification custody changed while opened")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except ActorProtocolQualificationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except (BlockingIOError, OSError) as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(code, f"qualification custody is active or cannot be locked: {exc}")


def _release_custody_directory_lock_v1(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _exclusive_file(path: Path, payload: bytes) -> tuple[int, int, str]:
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short reservation write")
                view = view[written:]
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
            descriptor = None
        _fsync_directory(path.parent)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        # A failed write/fsync must not leave a custody wedge.  Remove only the
        # exact inode created by this call; never follow a replacement or
        # delete an unrecognized path.
        if created_identity is not None:
            try:
                observed = path.lstat()
                if (
                    stat.S_ISREG(observed.st_mode)
                    and not stat.S_ISLNK(observed.st_mode)
                    and (observed.st_dev, observed.st_ino) == created_identity
                ):
                    path.unlink()
                    try:
                        _fsync_directory(path.parent)
                    except OSError:
                        pass
            except FileNotFoundError:
                pass
            except OSError:
                # Preserve the original failure.  The exact-inode checks above
                # ensure this path never broadens into destructive cleanup.
                pass
        _fail(FAIL_CUSTODY, f"cannot create exact reservation {path.name}: {exc}")
    return metadata.st_dev, metadata.st_ino, _sha256(payload)


@dataclass(slots=True)
class QualificationCustodyReservationV1:
    """Three exact files; never removes a directory or an unrecognized path."""

    custody_directory: Path
    basis_commit: str
    run_id: bytes
    ledger_id: bytes
    paths: tuple[Path, ...] = field(init=False)
    fingerprints: dict[Path, tuple[int, int, str]] = field(default_factory=dict)
    custody_lock_descriptor: int | None = None
    custody_lock_held: bool = False
    lock_descriptor: int | None = None
    lock_held: bool = False
    actors_started: bool = False
    actors_verified_absent: bool = False
    requested_custody_directory: Path = field(init=False)

    def __post_init__(self) -> None:
        self.requested_custody_directory = Path(
            os.path.abspath(os.fspath(self.custody_directory))
        )
        try:
            requested_metadata = self.requested_custody_directory.lstat()
        except OSError as exc:
            _fail(FAIL_CUSTODY, f"qualification custody path is absent: {exc}")
        if stat.S_ISLNK(requested_metadata.st_mode):
            _fail(FAIL_CUSTODY, "qualification custody path cannot be a symlink alias")
        self.custody_directory = self.requested_custody_directory
        self.paths = (
            self.custody_directory / "phase3_m25_ceremony.lock",
            self.custody_directory / f"opaque-run-{self.run_id.hex()}.reserved",
            self.custody_directory / f"opaque-ledger-{self.ledger_id.hex()}.reserved",
        )

    def reserve(self) -> None:
        try:
            location = validate_linux_local_durable_custody_location_v1(
                self.requested_custody_directory,
                repository_root=REPOSITORY_ROOT,
                allowed_owner_uids=frozenset({os.geteuid()}),
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_CUSTODY, f"qualification custody is invalid: {exc}")
        try:
            self.custody_lock_descriptor = _acquire_custody_directory_lock_v1(
                self.custody_directory, code=FAIL_CUSTODY
            )
            self.custody_lock_held = True
            if any(self.custody_directory.iterdir()):
                _fail(FAIL_CUSTODY, "qualification custody must be initially empty")
            common = {
                "schema": "hegel-phase3-m25-actor-protocol-qualification-reservation/1",
                "artifact_kind": ARTIFACT_KIND,
                "basis_commit": self.basis_commit,
                "custody_location_receipt_sha256": _sha256(_canonical_json(location)),
            }
            payloads = (
                {
                    **common,
                    "reservation_kind": "lock",
                    "run_id_hex": self.run_id.hex(),
                    "ledger_id_hex": self.ledger_id.hex(),
                },
                {
                    **common,
                    "reservation_kind": "run",
                    "opaque_id_hex": self.run_id.hex(),
                },
                {
                    **common,
                    "reservation_kind": "ledger",
                    "opaque_id_hex": self.ledger_id.hex(),
                },
            )
            self.fingerprints[self.paths[0]] = _exclusive_file(
                self.paths[0], _canonical_json(payloads[0])
            )
            self.lock_descriptor = os.open(
                self.paths[0], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            opened_lock = os.fstat(self.lock_descriptor)
            if (opened_lock.st_dev, opened_lock.st_ino) != self.fingerprints[
                self.paths[0]
            ][:2]:
                _fail(FAIL_CUSTODY, "qualification lock changed while opened")
            fcntl.flock(self.lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_held = True
            for path, value in zip(self.paths[1:], payloads[1:], strict=True):
                self.fingerprints[path] = _exclusive_file(path, _canonical_json(value))
        except BaseException:
            if not self.actors_started and self.fingerprints:
                self._remove_created_exact_files(pre_actor_cleanup=True)
            else:
                self.release_lock_without_cleanup()
            raise

    def mark_actors_started(self) -> None:
        self.actors_started = True

    def mark_actors_verified_absent(self) -> None:
        self.actors_verified_absent = True

    def _validate_fingerprint(self, path: Path) -> None:
        expected = self.fingerprints.get(path)
        if expected is None:
            _fail(FAIL_CLEANUP, f"reservation fingerprint is absent: {path.name}")
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (metadata.st_dev, metadata.st_ino) != expected[:2]
            or _hash_regular_file(path, code=FAIL_CLEANUP) != expected[2]
        ):
            _fail(FAIL_CLEANUP, f"reservation identity changed: {path.name}")

    def _remove_created_exact_files(self, *, pre_actor_cleanup: bool) -> None:
        try:
            if self.custody_lock_descriptor is None or not self.custody_lock_held:
                _fail(FAIL_CLEANUP, "custody directory lock is not held by reservation")
            custody_metadata = self.custody_directory.lstat()
            opened_custody = os.fstat(self.custody_lock_descriptor)
            if (
                stat.S_ISLNK(custody_metadata.st_mode)
                or not stat.S_ISDIR(custody_metadata.st_mode)
                or (opened_custody.st_dev, opened_custody.st_ino)
                != (custody_metadata.st_dev, custody_metadata.st_ino)
            ):
                _fail(FAIL_CLEANUP, "held custody directory identity differs")
            lock_path = self.paths[0]
            if lock_path in self.fingerprints:
                if self.lock_descriptor is None or not self.lock_held:
                    _fail(FAIL_CLEANUP, "reservation file lock is not held")
                opened_lock = os.fstat(self.lock_descriptor)
                if (opened_lock.st_dev, opened_lock.st_ino) != self.fingerprints[
                    lock_path
                ][:2]:
                    _fail(FAIL_CLEANUP, "held reservation lock identity differs")
            if self.actors_started and not self.actors_verified_absent:
                _fail(FAIL_CLEANUP, "reservations cannot be removed before actor absence")
            expected_existing = {path.name for path in self.fingerprints}
            actual = {path.name for path in self.custody_directory.iterdir()}
            if not actual <= expected_existing:
                _fail(FAIL_CLEANUP, "custody contains an unrecognized path; cleanup refused")
            if not pre_actor_cleanup and actual != expected_existing:
                _fail(FAIL_CLEANUP, "qualification reservation set is incomplete")
            for path in reversed(self.paths[1:]):
                if path not in self.fingerprints:
                    continue
                self._validate_fingerprint(path)
                path.unlink()
                self.fingerprints.pop(path)
            # Make removal of every opaque reservation durable while the valid
            # lock inode is still present and held.  Only after this barrier may
            # the lock become the final unlink, preventing crash reordering from
            # resurrecting an opaque file without its recovery identity.
            _fsync_directory(self.custody_directory)
            if lock_path in self.fingerprints:
                self._validate_fingerprint(lock_path)
                lock_path.unlink()
                self.fingerprints.pop(lock_path)
            _fsync_directory(self.custody_directory)
            if any(self.custody_directory.iterdir()):
                _fail(FAIL_CLEANUP, "qualification custody is not empty after exact cleanup")
        finally:
            # Locks remain held through the final durable unlink on success,
            # but every failure path releases its raw descriptors while
            # preserving whatever on-disk recovery evidence remains.
            self.release_lock_without_cleanup()

    def cleanup_after_actor_absence(self) -> None:
        if not self.actors_verified_absent:
            try:
                _fail(FAIL_CLEANUP, "actor absence was not established")
            finally:
                self.release_lock_without_cleanup()
        self._remove_created_exact_files(pre_actor_cleanup=False)

    def cleanup_before_actor_start(self) -> None:
        if self.actors_started:
            try:
                _fail(FAIL_CLEANUP, "pre-actor cleanup called after actors started")
            finally:
                self.release_lock_without_cleanup()
        self._remove_created_exact_files(pre_actor_cleanup=True)

    def release_lock_without_cleanup(self) -> None:
        """Release an adopted orphan lock while preserving evidence for retry."""

        if self.lock_descriptor is not None:
            descriptor = self.lock_descriptor
            self.lock_descriptor = None
            self.lock_held = False
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass
        if self.custody_lock_descriptor is not None:
            descriptor = self.custody_lock_descriptor
            self.custody_lock_descriptor = None
            self.custody_lock_held = False
            try:
                _release_custody_directory_lock_v1(descriptor)
            except OSError:
                # close(2) is attempted by the helper even if LOCK_UN fails.
                pass
        self.lock_held = False
        self.custody_lock_held = False


def _recover_orphaned_qualification_reservation_v1(
    *,
    custody_directory: Path,
    basis_commit: str,
    rust_formal_replay_binary: Path,
) -> bool:
    """Recover an exact no-seed qualifier orphan left by process termination.

    A partial reservation proves actors could not yet have started.  A complete
    reservation is conservatively treated as post-start: the formal backend's
    pre-seed recovery validates and removes only exact run-labelled containers
    and purpose-private volumes before the three reservation inodes are
    unlinked.  Unknown paths, a live flock, another commit, or any seed file
    fail closed.
    """

    custody = Path(os.path.abspath(os.fspath(custody_directory)))
    try:
        custody_metadata = custody.lstat()
        if (
            stat.S_ISLNK(custody_metadata.st_mode)
            or not stat.S_ISDIR(custody_metadata.st_mode)
        ):
            _fail(
                FAIL_CUSTODY,
                "orphan qualification custody cannot be a symlink/non-directory",
            )
        entry_location = validate_linux_local_durable_custody_location_v1(
            custody,
            repository_root=REPOSITORY_ROOT,
            allowed_owner_uids=frozenset({os.geteuid()}),
        )
    except ActorProtocolQualificationError:
        raise
    except (OSError, Phase3LocalRuntimeError) as exc:
        _fail(FAIL_CUSTODY, f"orphan qualification custody is invalid: {exc}")
    custody_lock_descriptor = _acquire_custody_directory_lock_v1(
        custody, code=FAIL_CUSTODY
    )
    custody_lock_owner: list[int | None] = [custody_lock_descriptor]
    try:
        return _recover_orphaned_qualification_reservation_under_lock_v1(
            custody=custody,
            basis_commit=basis_commit,
            rust_formal_replay_binary=rust_formal_replay_binary,
            entry_location=entry_location,
            custody_lock_owner=custody_lock_owner,
        )
    finally:
        if custody_lock_owner[0] is not None:
            _release_custody_directory_lock_v1(custody_lock_owner[0])


def _recover_orphaned_qualification_reservation_under_lock_v1(
    *,
    custody: Path,
    basis_commit: str,
    rust_formal_replay_binary: Path,
    entry_location: Mapping[str, object],
    custody_lock_owner: list[int | None],
) -> bool:
    """Recover while the caller holds the custody-directory mutex."""

    actual_names = {path.name for path in custody.iterdir()}
    if not actual_names:
        return False
    lock_name = "phase3_m25_ceremony.lock"
    if lock_name not in actual_names:
        _fail(FAIL_CUSTODY, "orphan qualification has no exact lock reservation")
    lock_path = custody / lock_name
    lock_descriptor: int | None = None
    try:
        lock_metadata = lock_path.lstat()
        if (
            stat.S_ISLNK(lock_metadata.st_mode)
            or not stat.S_ISREG(lock_metadata.st_mode)
            or stat.S_IMODE(lock_metadata.st_mode) != 0o600
        ):
            _fail(FAIL_CUSTODY, "orphan qualification lock identity differs")
        lock_descriptor = os.open(
            lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        opened = os.fstat(lock_descriptor)
        if (opened.st_dev, opened.st_ino) != (
            lock_metadata.st_dev,
            lock_metadata.st_ino,
        ):
            _fail(FAIL_CUSTODY, "orphan qualification lock changed while opened")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            lock_body = _strict_json_object(lock_path.read_bytes())
        except ActorProtocolQualificationError:
            if actual_names == {lock_name}:
                # The lock is the first reservation written.  With no run or
                # ledger inode, reserve() could not have returned and actor
                # start was unreachable.  This is the only safe SIGKILL
                # mid-write case: remove the exact opened inode, durably, and
                # never invoke Docker cleanup without a parseable run label.
                observed = lock_path.lstat()
                if (
                    stat.S_ISREG(observed.st_mode)
                    and not stat.S_ISLNK(observed.st_mode)
                    and (observed.st_dev, observed.st_ino)
                    == (opened.st_dev, opened.st_ino)
                ):
                    lock_path.unlink()
                    _fsync_directory(custody)
                    return True
            raise
        run_hex = lock_body.get("run_id_hex")
        ledger_hex = lock_body.get("ledger_id_hex")
        if (
            type(run_hex) is not str
            or re.fullmatch(r"[0-9a-f]{32}", run_hex) is None
            or type(ledger_hex) is not str
            or re.fullmatch(r"[0-9a-f]{32}", ledger_hex) is None
        ):
            _fail(FAIL_CUSTODY, "orphan qualification opaque identity differs")
        run_id = bytes.fromhex(run_hex)
        ledger_id = bytes.fromhex(ledger_hex)
        reservation = QualificationCustodyReservationV1(
            custody, basis_commit, run_id, ledger_id
        )
        if custody_lock_owner[0] is None:
            _fail(FAIL_CUSTODY, "orphan custody directory lock ownership was lost")
        reservation.custody_lock_descriptor = custody_lock_owner[0]
        reservation.custody_lock_held = True
        custody_lock_owner[0] = None
        reservation.lock_descriptor = lock_descriptor
        reservation.lock_held = True
        lock_descriptor = None
    except ActorProtocolQualificationError:
        raise
    except (BlockingIOError, OSError) as exc:
        _fail(FAIL_CUSTODY, f"cannot inspect or lock orphan qualification: {exc}")
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_descriptor)
            except OSError:
                pass
    expected_names = {path.name for path in reservation.paths}
    try:
        if not actual_names <= expected_names:
            _fail(FAIL_CUSTODY, "orphan qualification custody has an unknown path")
        # Qualification never delegates seed operations to the formal Docker
        # backend, so a legitimate orphan remains owned by the supervisor UID.
        # Any ownership transition is outside this protocol and fails closed.
        location = entry_location
        common = {
            "schema": "hegel-phase3-m25-actor-protocol-qualification-reservation/1",
            "artifact_kind": ARTIFACT_KIND,
            "basis_commit": basis_commit,
            "custody_location_receipt_sha256": _sha256(_canonical_json(location)),
        }
        expected_payloads = {
            reservation.paths[0]: {
                **common,
                "reservation_kind": "lock",
                "run_id_hex": run_hex,
                "ledger_id_hex": ledger_hex,
            },
            reservation.paths[1]: {
                **common,
                "reservation_kind": "run",
                "opaque_id_hex": run_hex,
            },
            reservation.paths[2]: {
                **common,
                "reservation_kind": "ledger",
                "opaque_id_hex": ledger_hex,
            },
        }
        for path, expected in expected_payloads.items():
            if path.name not in actual_names:
                continue
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                _fail(FAIL_CUSTODY, f"orphan reservation differs: {path.name}")
            try:
                observed_payload = _strict_json_object(path.read_bytes())
            except ActorProtocolQualificationError:
                observed_payload = None
            if path == reservation.paths[0] and observed_payload != expected:
                _fail(FAIL_CUSTODY, "orphan lock reservation binding differs")
            if (
                path != reservation.paths[0]
                and observed_payload is not None
                and observed_payload != expected
            ):
                _fail(FAIL_CUSTODY, f"orphan opaque reservation binding differs: {path.name}")
            # A run/ledger file can be interrupted mid-write.  The already
            # validated lock binds both exact IDs; regardless of whether an
            # opaque body is complete or externally damaged, Docker pre-seed
            # absence recovery runs before this exact inode is removed.
            reservation.fingerprints[path] = (
                metadata.st_dev,
                metadata.st_ino,
                _hash_regular_file(path, code=FAIL_CUSTODY),
            )
        # Even a strict prefix is passed through Docker absence recovery.  This
        # remains safe if a reservation file was externally lost after actor
        # start; the valid lock body still supplies the exact run label.
        reservation.mark_actors_started()
        recovery_backend = DockerCeremonyActorsV1(
            basis_commit=basis_commit,
            custody_directory=custody,
            rust_formal_replay_binary=rust_formal_replay_binary,
            rust_bridge_dag_replay_binary=DEFAULT_RUST_BRIDGE_DAG_BINARY,
            rust_bridge_dag_qualification_report=(
                DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT
            ),
            timestamp=FROZEN_QUALIFICATION_TIMESTAMP,
        )
        receipt = recovery_backend.recover_preseed_private_state_and_verify_absent(
            run_id
        )
        if (
            receipt.get("run_id_hex") != run_hex
            or receipt.get("actor_containers_absent") is not True
            or receipt.get("actor_key_volumes_absent") is not True
            or receipt.get("seed_continuity_state_absent") is not True
        ):
            _fail(FAIL_CLEANUP, "orphan pre-seed recovery receipt differs")
        reservation.mark_actors_verified_absent()
        reservation.cleanup_after_actor_absence()
        return True
    except BaseException:
        reservation.release_lock_without_cleanup()
        raise


def _operation_input_rows_v1(input_directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_text, directory_names, file_names in os.walk(
        input_directory, topdown=True, followlinks=False
    ):
        root = Path(root_text)
        metadata = root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            _fail(FAIL_PROTOCOL, "operation input contains an invalid directory")
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            child = root / directory_name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISDIR(
                child_metadata.st_mode
            ):
                _fail(FAIL_PROTOCOL, "operation input contains a symlink directory")
        for file_name in file_names:
            path = root / file_name
            file_metadata = path.lstat()
            if stat.S_ISLNK(file_metadata.st_mode) or not stat.S_ISREG(
                file_metadata.st_mode
            ):
                _fail(FAIL_PROTOCOL, "operation input contains a non-regular file")
            rows.append({
                "path": path.relative_to(input_directory).as_posix(),
                "mode_octal": f"{stat.S_IMODE(file_metadata.st_mode):04o}",
                "size": file_metadata.st_size,
                "sha256": _hash_regular_file(path).removeprefix("sha256:"),
            })
    return rows


def _generic_operation_request_body_v1(
    backend: DockerCeremonyActorsV1,
    host_receipt: Mapping[str, object],
) -> dict[str, object]:
    purpose = host_receipt["purpose_id"]
    assert type(purpose) is int
    input_directory, _output = backend._actor_dirs(purpose)
    return {
        "schema": "hegel-phase3-m25-operation-request-binding/1",
        "basis_commit": backend.basis_commit,
        "container_id": host_receipt["container_id"],
        "daemon_receipt_sha256": host_receipt["daemon_receipt_sha256"],
        "input_rows": _operation_input_rows_v1(input_directory),
        "operation_id": host_receipt["operation_id"],
        "operation_nonce_hex": host_receipt["operation_nonce_hex"],
        "operation_sequence": host_receipt["operation_sequence"],
        "profile_sha256": backend._profile_digest.hex(),  # type: ignore[union-attr]
        "purpose_id": purpose,
        "run_id_hex": backend._transaction_run_id.hex(),  # type: ignore[union-attr]
    }


@dataclass(slots=True)
class OperationEvidenceRecorderV1:
    """Capture every leaf immediately after the backend validates it."""

    backend: DockerCeremonyActorsV1
    rows: list[dict[str, object]] = field(default_factory=list)
    _original_exec: object | None = None

    def install(self) -> None:
        if self._original_exec is not None:
            _fail(FAIL_PROTOCOL, "operation evidence recorder installed twice")
        original = self.backend._exec
        self._original_exec = original

        def captured_exec(
            purpose: int,
            operation: str,
            *,
            operation_request_digest_override: bytes | None = None,
            timeout_seconds: int = 240,
        ) -> Mapping[str, object]:
            result = original(
                purpose,
                operation,
                operation_request_digest_override=operation_request_digest_override,
                timeout_seconds=timeout_seconds,
            )
            if not self.backend._operation_probe_receipts:
                _fail(FAIL_PROTOCOL, "backend omitted host operation receipt")
            host = dict(self.backend._operation_probe_receipts[-1])
            if (
                host.get("purpose_id") != purpose
                or host.get("operation_id") != operation
            ):
                _fail(FAIL_PROTOCOL, "captured host operation receipt differs")
            input_directory, output_directory = self.backend._actor_dirs(purpose)
            rust_raw: dict[str, object] | None = None
            if purpose == 3:
                rust_raw, _payload = self.backend._read_single_json_line(
                    output_directory / f"operation-rust-probe-{operation}.json"
                )
            if operation_request_digest_override is None:
                request_binding: dict[str, object] = {
                    "kind": "GENERIC_INPUT_TREE_V1",
                    "body": _generic_operation_request_body_v1(self.backend, host),
                }
                request_digest = hashlib.sha256(
                    OPERATION_REQUEST_HASH_DOMAIN
                    + _canonical_json(request_binding["body"])
                ).hexdigest()
            else:
                request_path = input_directory / "purpose4-keybearing-request.json"
                try:
                    request = json.loads(request_path.read_bytes())
                except (OSError, json.JSONDecodeError) as exc:
                    _fail(FAIL_PROTOCOL, f"purpose-4 request capture failed: {exc}")
                if type(request) is not dict:
                    _fail(FAIL_PROTOCOL, "purpose-4 request is not an object")
                request_binding = {
                    "kind": "PURPOSE4_REQUEST_SHA256_V1",
                    "body": request,
                }
                request_digest = operation_request_digest_override.hex()
            if request_digest != host.get("operation_request_sha256"):
                _fail(FAIL_PROTOCOL, "captured operation request preimage differs")
            self.rows.append({
                "purpose_id": purpose,
                "operation_id": operation,
                "operation_sequence": host["operation_sequence"],
                "operation_nonce_hex": host["operation_nonce_hex"],
                "request_binding": request_binding,
                "actor_receipt": dict(result),
                "rust_raw_probe_receipt_or_null": rust_raw,
                "host_receipt": host,
            })
            return result

        self.backend._exec = captured_exec  # type: ignore[method-assign]


@dataclass(slots=True)
class PublicSyntheticProtocolActorV1(CeremonyActorsV1):
    """Synthetic split/marker shim around the real Docker signing actors."""

    delegate: DockerCeremonyActorsV1
    split_frame: bytes = PUBLIC_SYNTHETIC_SPLIT_FRAME
    authoritative: bool = True
    keygen_purposes: list[int] = field(default_factory=list)
    signed_object_names: list[str] = field(default_factory=list)
    bridge_purposes: list[int] = field(default_factory=list)
    bridge_replay_receipt_sha256: dict[int, str] = field(default_factory=dict)
    bridge_evidence_rows: dict[int, dict[str, object]] = field(default_factory=dict)
    purpose4_response_sha256: str | None = None
    purpose4_request: dict[str, object] | None = None
    purpose4_response: dict[str, object] | None = None
    parent_sign_count: int = 0
    synthetic_split_delivery_count: int = 0
    prospective_marker_count: int = 0
    docker_seed_method_call_count: int = 0

    def keygen(self, purpose: int) -> bytes:
        if purpose in self.keygen_purposes:
            _fail(FAIL_PROTOCOL, f"qualification keygen repeated for purpose {purpose}")
        public_key = self.delegate.keygen(purpose)
        self.keygen_purposes.append(purpose)
        return public_key

    def seed_split(self) -> tuple[bytes, bytes]:
        self.synthetic_split_delivery_count += 1
        if self.synthetic_split_delivery_count != 1:
            _fail(FAIL_PROTOCOL, "public synthetic split frame delivered more than once")
        return self.split_frame, self.split_frame

    def prospective_complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        self.prospective_marker_count += 1
        if self.prospective_marker_count != 1:
            _fail(FAIL_PROTOCOL, "synthetic prospective marker requested more than once")
        return MarkerSnapshot(
            state="COMPLETE",
            split_version_digest=SPLIT_VERSION_DIGEST,
            seed_commitment_manifest_root=seed_manifest_root,
            custodian_key_id=_public_split_component("marker-custodian-id")[:16],
            created_at_unix_seconds=FROZEN_QUALIFICATION_TIMESTAMP,
        )

    def sign_object(self, name: str, fields: Mapping[str, object]) -> bytes:
        signature = self.delegate.sign_object(name, fields)
        self.signed_object_names.append(name)
        return signature

    def sign_parent(self, evidence, fields: Mapping[str, object]) -> bytes:
        if self.parent_sign_count:
            _fail(FAIL_PROTOCOL, "purpose-4 detached parent operation repeated")
        signature = self.delegate.sign_parent(evidence, fields)
        self.parent_sign_count = 1
        _input, output = self.delegate._actor_dirs(4)
        self.purpose4_response_sha256 = _hash_regular_file(
            output / "purpose4-keybearing-detached-response.json"
        )
        try:
            request = json.loads(
                (_input / "purpose4-keybearing-request.json").read_bytes()
            )
            response = json.loads(
                (output / "purpose4-keybearing-detached-response.json").read_bytes()
            )
        except (OSError, json.JSONDecodeError) as exc:
            _fail(FAIL_PROTOCOL, f"purpose-4 evidence capture failed: {exc}")
        if type(request) is not dict or type(response) is not dict:
            _fail(FAIL_PROTOCOL, "purpose-4 evidence is not object-shaped")
        self.purpose4_request = request
        self.purpose4_response = response
        return signature

    def sign_bridge(
        self,
        purpose: int,
        fields: Mapping[str, object],
        replay_package: bytes,
    ) -> bytes:
        if purpose in self.bridge_purposes:
            _fail(FAIL_PROTOCOL, f"bridge replay repeated for purpose {purpose}")
        signature = self.delegate.sign_bridge(purpose, fields, replay_package)
        _input, output = self.delegate._actor_dirs(purpose)
        self.bridge_replay_receipt_sha256[purpose] = _hash_regular_file(
            output / "bridge-dag-replay-receipt.json"
        )
        try:
            replay_receipt = json.loads(
                (output / "bridge-dag-replay-receipt.json").read_bytes()
            )
        except (OSError, json.JSONDecodeError) as exc:
            _fail(FAIL_PROTOCOL, f"bridge replay evidence capture failed: {exc}")
        if type(replay_receipt) is not dict:
            _fail(FAIL_PROTOCOL, "bridge replay receipt is not an object")
        if len(replay_package) > MAX_BRIDGE_PACKAGE_BYTES:
            _fail(FAIL_PROTOCOL, "bridge replay package exceeds public archive bound")
        self.bridge_evidence_rows[purpose] = {
            "purpose_id": purpose,
            "package_base64": _raw_base64(replay_package),
            "package_size": len(replay_package),
            "package_sha256": _sha256(replay_package),
            "replay_receipt": replay_receipt,
            "bridge_signature_hex": signature.hex(),
        }
        self.bridge_purposes.append(purpose)
        return signature

    # Every state-changing Docker-seed method is an explicit fail-closed trap.
    def complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        self.docker_seed_method_call_count += 1
        _fail(FAIL_PROTOCOL, "qualification cannot complete a custody marker")

    def resume_pending_seed_split(self) -> tuple[bytes, bytes]:
        self.docker_seed_method_call_count += 1
        _fail(FAIL_PROTOCOL, "qualification cannot resume a seed split")

    def prepare_post_stage_pending_recovery(self, run_id: bytes) -> None:
        self.docker_seed_method_call_count += 1
        _fail(FAIL_PROTOCOL, "qualification cannot enter seed recovery")

    def resume_post_stage_seed_split(self) -> tuple[bytes, bytes]:
        self.docker_seed_method_call_count += 1
        _fail(FAIL_PROTOCOL, "qualification cannot enter post-stage seed recovery")


def _operation_receipt_summary_v1(
    backend: DockerCeremonyActorsV1,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    observed: list[tuple[int, int, str]] = []
    for receipt in backend._operation_probe_receipts:
        purpose = receipt.get("purpose_id")
        sequence = receipt.get("operation_sequence")
        operation = receipt.get("operation_id")
        digest = receipt.get("receipt_sha256")
        if (
            type(purpose) is not int
            or type(sequence) is not int
            or type(operation) is not str
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            _fail(FAIL_PROTOCOL, "host operation receipt summary is malformed")
        observed.append((purpose, sequence, operation))
        rows.append({
            "purpose_id": purpose,
            "operation_sequence": sequence,
            "operation_id": operation,
            "host_verified_receipt_sha256": "sha256:" + digest,
        })
    if tuple(observed) != EXPECTED_OPERATION_SEQUENCE:
        _fail(FAIL_PROTOCOL, "live actor operation sequence differs from qualification contract")
    return tuple(rows)


def _wrapper_summary_v1(
    wrapper: PublicSyntheticProtocolActorV1,
) -> dict[str, object]:
    expected_objects = [
        "SplitSeedCommitmentManifestV1",
        "CustodianBindingManifestV1",
        "SeedContinuityManifestV1",
        "HiddenAccessLedgerRecordV1",
    ]
    if (
        wrapper.keygen_purposes != [1, 2, 3, 4]
        or wrapper.signed_object_names != expected_objects
        or wrapper.parent_sign_count != 1
        or wrapper.bridge_purposes != [1, 2, 3]
        or set(wrapper.bridge_replay_receipt_sha256) != {1, 2, 3}
        or set(wrapper.bridge_evidence_rows) != {1, 2, 3}
        or wrapper.purpose4_response_sha256 is None
        or wrapper.purpose4_request is None
        or wrapper.purpose4_response is None
        or wrapper.synthetic_split_delivery_count != 1
        or wrapper.prospective_marker_count != 1
        or wrapper.docker_seed_method_call_count != 0
    ):
        _fail(FAIL_PROTOCOL, "live actor protocol call graph is incomplete")
    return {
        "four_distinct_ephemeral_actor_keys_generated": True,
        "purpose1_authorized_object_sign_count": 4,
        "purpose4_detached_full_history_sign_count": 1,
        "bridge_replay_order": [1, 2, 3],
        "synthetic_split_frame_delivery_count": 1,
        "synthetic_prospective_marker_count": 1,
        "docker_seed_or_marker_method_call_count": 0,
        "purpose4_detached_response_sha256": wrapper.purpose4_response_sha256,
        "bridge_actor_replay_receipts": [
            {
                "purpose_id": purpose,
                "implementation": (
                    "rust-full-dag-replay-v1"
                    if purpose == 3
                    else "python-full-dag-replay-v1"
                ),
                "validated_receipt_sha256": wrapper.bridge_replay_receipt_sha256[purpose],
            }
            for purpose in (1, 2, 3)
        ],
    }


def _qualification_key_manifests_v1(
    backend: DockerCeremonyActorsV1,
) -> list[dict[str, object]]:
    images = backend._profile.get("images")
    if not isinstance(images, Mapping):
        _fail(FAIL_PROTOCOL, "qualification image registry is absent")
    image_keys = {
        1: "custodian",
        2: "python_attester",
        3: "rust_attester",
        4: "policy_auditor",
    }
    rows: list[dict[str, object]] = []
    for purpose in (1, 2, 3, 4):
        public_key = backend._public_keys.get(purpose)
        key_id = backend._key_ids.get(purpose)
        image = images.get(image_keys[purpose])
        container_id = backend._containers.get(purpose)
        if (
            type(public_key) is not bytes
            or len(public_key) != 32
            or type(key_id) is not bytes
            or len(key_id) != 16
            or hashlib.sha256(public_key).digest()[:16] != key_id
            or type(image) is not str
            or re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image) is None
            or type(container_id) is not str
            or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
        ):
            _fail(FAIL_PROTOCOL, f"qualification actor identity {purpose} is incomplete")
        manifest = {
            "schema": KEY_MANIFEST_SCHEMA,
            "purpose_id": purpose,
            "usage": "LIVE_PROTOCOL_QUALIFICATION_ONLY",
            "public_key_32_hex": public_key.hex(),
            "key_id_16_hex": key_id.hex(),
            "basis_commit": backend.basis_commit,
            "profile_sha256": backend._profile_digest.hex(),  # type: ignore[union-attr]
            "image_ref": image,
            "daemon_receipt_sha256": backend._docker_daemon_binding.hex(),  # type: ignore[union-attr]
            "container_id": container_id,
            "created_at_unix_seconds": FROZEN_QUALIFICATION_TIMESTAMP,
            "ephemeral_qualification_identity": True,
            "eligible_for_formal_actor_trust": False,
            "formal_genesis_reuse_forbidden": True,
            "must_destroy": True,
        }
        manifest["manifest_content_id"] = _content_id(
            KEY_MANIFEST_HASH_DOMAIN, manifest
        )
        rows.append(manifest)
    if len({row["public_key_32_hex"] for row in rows}) != 4:
        _fail(FAIL_PROTOCOL, "qualification keys are not purpose-distinct")
    return rows


def _destruction_plan_v1(
    backend: DockerCeremonyActorsV1,
) -> dict[str, object]:
    rows = []
    for purpose in (1, 2, 3, 4):
        container_id = backend._containers.get(purpose)
        volume_name = backend._state_volumes.get(purpose)
        if type(container_id) is not str or type(volume_name) is not str:
            _fail(FAIL_PROTOCOL, "cleanup identity set is incomplete")
        rows.append({
            "purpose_id": purpose,
            "container_id": container_id,
            "actor_key_volume_name": volume_name,
            "must_remove_container": True,
            "must_remove_actor_key_volume": True,
        })
    plan: dict[str, object] = {
        "schema": DESTRUCTION_PLAN_SCHEMA,
        "basis_commit": backend.basis_commit,
        "qualification_run_id_hex": backend._transaction_run_id.hex(),  # type: ignore[union-attr]
        "daemon_receipt_sha256": backend._docker_daemon_binding.hex(),  # type: ignore[union-attr]
        "actor_rows": rows,
        "required_cleanup_order": [
            "CONTAINERS_REMOVED_AND_VERIFIED_ABSENT",
            "KEY_VOLUMES_REMOVED_AND_VERIFIED_ABSENT",
            "EXACT_RESERVATIONS_REMOVED",
        ],
        "seed_or_marker_artifacts_must_remain_absent": True,
        "qualification_keys_must_be_destroyed": True,
        "formal_genesis_reuse_forbidden": True,
    }
    plan["destruction_plan_content_id"] = _content_id(
        DESTRUCTION_PLAN_HASH_DOMAIN, plan
    )
    return plan


def _controlled_docker_call_v1(control_plane, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            control_plane.command(*arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
            env=dict(control_plane.environment),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(FAIL_CLEANUP, f"controlled Docker absence check failed: {type(exc).__name__}")


def _live_daemon_and_absence_rows_v1(
    plan: Mapping[str, object],
) -> tuple[str, list[dict[str, object]]]:
    try:
        temporary = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-m25-protocol-absence-", repository_root=REPOSITORY_ROOT
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_CLEANUP, f"absence control plane temp failed: {exc}")
    with temporary as temporary_name:
        root = Path(temporary_name)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                root, repository_root=REPOSITORY_ROOT
            )
            version_call = _controlled_docker_call_v1(
                control_plane, "version", "--format={{json .}}"
            )
            info_call = _controlled_docker_call_v1(
                control_plane, "info", "--format={{json .}}"
            )
            if version_call.returncode != 0 or info_call.returncode != 0:
                _fail(FAIL_CLEANUP, "local Docker daemon identity call failed")
            daemon_receipt = build_local_docker_daemon_identity_receipt_v1(
                control_plane,
                version_payload=json.loads(version_call.stdout),
                info_payload=json.loads(info_call.stdout),
                repository_root=REPOSITORY_ROOT,
            )
            daemon_binding = local_docker_daemon_receipt_binding_v1(
                daemon_receipt
            ).hex()
        except (Phase3LocalRuntimeError, json.JSONDecodeError) as exc:
            _fail(FAIL_CLEANUP, f"local Docker daemon qualification failed: {exc}")
        rows: list[dict[str, object]] = []
        actor_rows = plan.get("actor_rows")
        assert isinstance(actor_rows, list)
        for planned in actor_rows:
            assert isinstance(planned, Mapping)
            purpose = int(planned["purpose_id"])
            container_id = str(planned["container_id"])
            volume_name = str(planned["actor_key_volume_name"])
            container_inspect = _controlled_docker_call_v1(
                control_plane, "inspect", container_id
            )
            container_list = _controlled_docker_call_v1(
                control_plane, "ps", "-aq", "--no-trunc", "--filter",
                f"id={container_id}",
            )
            volume_inspect = _controlled_docker_call_v1(
                control_plane, "volume", "inspect", volume_name
            )
            volume_list = _controlled_docker_call_v1(
                control_plane, "volume", "ls", "-q", "--filter",
                f"name=^{volume_name}$",
            )
            if (
                container_inspect.returncode == 0
                or container_list.returncode != 0
                or container_list.stdout
                or volume_inspect.returncode == 0
                or volume_list.returncode != 0
                or volume_list.stdout
            ):
                _fail(FAIL_CLEANUP, f"purpose-{purpose} actor state remains")
            rows.append({
                "purpose_id": purpose,
                "container_id": container_id,
                "actor_key_volume_name": volume_name,
                "container_inspect_returncode": container_inspect.returncode,
                "container_inspect_stdout_sha256": _sha256(container_inspect.stdout),
                "container_list_returncode": container_list.returncode,
                "container_list_stdout_sha256": _sha256(container_list.stdout),
                "volume_inspect_returncode": volume_inspect.returncode,
                "volume_inspect_stdout_sha256": _sha256(volume_inspect.stdout),
                "volume_list_returncode": volume_list.returncode,
                "volume_list_stdout_sha256": _sha256(volume_list.stdout),
                "container_verified_absent": True,
                "actor_key_volume_verified_absent": True,
            })
        return daemon_binding, rows


def _post_cleanup_absence_receipt_v1(
    *,
    plan: Mapping[str, object],
    daemon_binding: str,
    actor_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": CLEANUP_RECEIPT_SCHEMA,
        "basis_commit": plan["basis_commit"],
        "qualification_run_id_hex": plan["qualification_run_id_hex"],
        "daemon_receipt_sha256": daemon_binding,
        "destruction_plan_content_id": plan["destruction_plan_content_id"],
        "actor_rows": [dict(row) for row in actor_rows],
        "seed_marker_absent": True,
        "seed_intent_absent": True,
        "seed_completion_absent": True,
        "raw_seed_absent": True,
        "exact_reservations_removed": True,
        "custody_directory_empty": True,
        "custody_directory_removed": False,
    }
    receipt["cleanup_receipt_content_id"] = _content_id(
        CLEANUP_RECEIPT_HASH_DOMAIN, receipt
    )
    return receipt


def _qualification_statement_preimage_v1(
    purpose: int, statement: Mapping[str, object]
) -> bytes:
    if purpose not in (1, 2, 3, 4):
        _fail(FAIL_REPORT, "qualification signature purpose is invalid")
    statement_id = bytes.fromhex(
        _content_id(STATEMENT_HASH_DOMAIN, dict(statement)).removeprefix("sha256:")
    )
    return STATEMENT_SIGNATURE_DOMAIN + bytes([purpose]) + statement_id


def _install_finalize_workers_v1(backend: DockerCeremonyActorsV1) -> None:
    payload = _git(("show", f"{backend.basis_commit}:{FINALIZE_WORKER_REPOSITORY_PATH}"))
    for purpose in (1, 2, 3, 4):
        input_directory, _output = backend._actor_dirs(purpose)
        destination = input_directory / "tools" / Path(
            FINALIZE_WORKER_REPOSITORY_PATH
        ).name
        if destination.exists() or destination.is_symlink():
            _fail(FAIL_PROTOCOL, "qualification finalize worker destination occupied")
        destination.write_bytes(payload)
        destination.chmod(0o555)


def _finalize_qualification_statements_v1(
    *,
    backend: DockerCeremonyActorsV1,
    evidence_content_id: str,
    key_manifests: Sequence[Mapping[str, object]],
    destruction_plan_content_id: str,
) -> list[dict[str, object]]:
    by_purpose = {int(row["purpose_id"]): row for row in key_manifests}
    envelopes: list[dict[str, object]] = []
    for purpose in (1, 2, 3, 4):
        manifest = by_purpose[purpose]
        sequence = backend._operation_sequences[purpose] + 1
        nonce = secrets.token_bytes(16)
        before = backend._inspect_live_actor(purpose)
        base_statement: dict[str, object] = {
            "schema": STATEMENT_SCHEMA,
            "operation_id": "qualification-finalize",
            "purpose_id": purpose,
            "basis_commit": backend.basis_commit,
            "qualification_run_id_hex": backend._transaction_run_id.hex(),  # type: ignore[union-attr]
            "qualification_evidence_content_id": evidence_content_id,
            "destruction_plan_content_id": destruction_plan_content_id,
            "qualification_key_manifest_content_id": manifest[
                "manifest_content_id"
            ],
            "profile_sha256": backend._profile_digest.hex(),  # type: ignore[union-attr]
            "image_ref": manifest["image_ref"],
            "daemon_receipt_sha256": backend._docker_daemon_binding.hex(),  # type: ignore[union-attr]
            "container_id": backend._containers[purpose],
            "key_id_16_hex": manifest["key_id_16_hex"],
            "operation_sequence": sequence,
            "operation_nonce_hex": nonce.hex(),
            "container_inspection_sha256": _sha256(_canonical_json(before)),
            "formal_authority": False,
            "formal_gates_before": 14,
            "formal_gates_after": 14,
            "m3_state": "NOT_RUN",
            "m3_started": False,
            "real_seed_generated": False,
            "real_seed_accessed": False,
            "formal_output_published": False,
            "qualification_identity_usage": "LIVE_PROTOCOL_QUALIFICATION_ONLY",
            "eligible_for_formal_actor_trust": False,
            "formal_genesis_reuse_forbidden": True,
            "must_destroy_after_qualification": True,
            "independence_disclosure": dict(INDEPENDENCE_DISCLOSURE),
        }
        request_digest = _content_id(
            STATEMENT_REQUEST_HASH_DOMAIN, base_statement
        )
        statement = {
            **base_statement,
            "qualification_finalize_request_sha256": request_digest,
        }
        preimage = _qualification_statement_preimage_v1(purpose, statement)
        input_directory, output_directory = backend._actor_dirs(purpose)
        input_path = input_directory / "qualification-finalize-preimage.bin"
        statement_path = input_directory / "qualification-finalize-statement.json"
        request_path = input_directory / "qualification-finalize-request.json"
        output_path = output_directory / "qualification-finalize-signature.bin"
        probe_path = output_directory / "qualification-finalize-probe.json"
        live_probe_path = output_directory / "qualification-finalize-live-probe.json"
        if any(path.exists() or path.is_symlink() for path in (
            input_path, statement_path, request_path, output_path, probe_path,
            live_probe_path,
        )):
            _fail(FAIL_PROTOCOL, "qualification finalize path is already occupied")
        request = {
            "schema": "hegel-phase3-m25-protocol-qualification-finalize-request/1",
            "purpose_id": purpose,
            "statement": statement,
            "preimage_sha256": _sha256(preimage),
        }
        request_payload = _canonical_json(request)
        operation_request_digest = hashlib.sha256(request_payload).digest()
        input_path.write_bytes(preimage)
        input_path.chmod(0o444)
        statement_path.write_bytes(_canonical_json(statement))
        statement_path.chmod(0o444)
        request_path.write_bytes(request_payload)
        request_path.chmod(0o444)
        environment = backend._actor_environment(
            purpose,
            operation="qualification-finalize",
            operation_sequence=sequence,
            operation_nonce=nonce,
            operation_request_digest=operation_request_digest,
        )
        environment["HEGEL_QUALIFICATION_PREIMAGE_SHA256"] = hashlib.sha256(
            preimage
        ).hexdigest()
        launch_environment = backend._actor_launch_environment(
            purpose,
            operation="qualification-finalize",
            operation_sequence=sequence,
            operation_nonce=nonce,
            operation_request_digest=operation_request_digest,
        )
        launch_environment["HEGEL_QUALIFICATION_PREIMAGE_SHA256"] = (
            environment["HEGEL_QUALIFICATION_PREIMAGE_SHA256"]
        )
        try:
            backend._docker(
                "exec",
                "--user=65534:65534",
                backend._containers[purpose],
                "/usr/bin/env",
                "-i",
                *(f"{key}={value}" for key, value in launch_environment.items()),
                "/bin/sh",
                "/input/tools/phase3_m25_protocol_qualification_finalize_worker_v1.sh",
                "qualification-finalize",
                timeout=120,
            )
            after = backend._inspect_live_actor(purpose)
            if after != before:
                _fail(FAIL_PROTOCOL, "qualification finalize changed actor identity")
            signature = output_path.read_bytes()
            finalize_probe = json.loads(probe_path.read_bytes())
            finalize_live_probe = json.loads(live_probe_path.read_bytes())
            if type(finalize_probe) is not dict or type(finalize_live_probe) is not dict:
                _fail(FAIL_PROTOCOL, "qualification finalize probe is not object-shaped")
            try:
                DockerCeremonyActorsV1._validate_common_probe_fields(
                    finalize_live_probe,
                    expected_environment=environment,
                    purpose=purpose,
                )
            except FormalContainerExecutorError as exc:
                _fail(
                    FAIL_PROTOCOL,
                    f"qualification finalize live probe failed: {exc.code}",
                )
            if finalize_probe != {
                "live_probe_sha256": hashlib.sha256(
                    _canonical_json(finalize_live_probe)
                ).hexdigest(),
                "operation_id": "qualification-finalize",
                "operation_nonce_hex": nonce.hex(),
                "operation_request_sha256": operation_request_digest.hex(),
                "operation_sequence": sequence,
                "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
                "purpose_id": purpose,
                "schema": "hegel-phase3-m25-protocol-qualification-finalize-probe/1",
                "signature_sha256": hashlib.sha256(signature).hexdigest(),
            }:
                _fail(FAIL_PROTOCOL, "qualification finalize probe binding differs")
            public_key = bytes.fromhex(str(manifest["public_key_32_hex"]))
            _verify_ed25519(public_key, signature, preimage)
        finally:
            if input_path.exists():
                input_path.unlink()
            if statement_path.exists():
                statement_path.unlink()
            if request_path.exists():
                request_path.unlink()
        backend._operation_sequences[purpose] = sequence
        envelopes.append({
            "schema": SIGNATURE_ENVELOPE_SCHEMA,
            "purpose_id": purpose,
            "statement": statement,
            "container_inspection": before,
            "finalize_request": request,
            "finalize_probe_receipt": finalize_probe,
            "finalize_live_probe_receipt": finalize_live_probe,
            "signature_hex": signature.hex(),
            "signature_verified_before_actor_destruction": True,
        })
    return envelopes


def _report_hash(body: Mapping[str, object]) -> str:
    return _sha256(REPORT_HASH_DOMAIN + _canonical_json(dict(body)))


@dataclass(frozen=True, slots=True)
class ReplayedActorProtocolQualificationV1:
    basis_commit: str
    bundle_content_id: bytes
    qualification_key_ids: Mapping[int, bytes]
    report: Mapping[str, object]


_LIVE_ADMISSION_CONSTRUCTOR_SEAL = object()
_PROCESS_ADMISSION_SECRET: bytes | None = None
_CONSUMED_ADMISSION_MACS: set[bytes] = set()
_CONSUMED_ADMISSION_LOCK = threading.Lock()
_ADMISSION_ISSUE_LOCK = threading.Lock()


def _reset_live_admission_state_after_fork_v1() -> None:
    """A forked child is never the issuer process of a parent's token."""

    global _PROCESS_ADMISSION_SECRET
    global _CONSUMED_ADMISSION_LOCK
    global _ADMISSION_ISSUE_LOCK
    _PROCESS_ADMISSION_SECRET = None
    _CONSUMED_ADMISSION_MACS.clear()
    # A lock held by a vanished thread cannot safely be reused after fork.
    _CONSUMED_ADMISSION_LOCK = threading.Lock()
    _ADMISSION_ISSUE_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_live_admission_state_after_fork_v1)


def _admission_mac_preimage_v1(
    *,
    basis_commit: str,
    bundle_content_id: bytes,
    qualification_key_ids: Mapping[int, bytes],
    daemon_receipt_binding: bytes,
    canonical_bundle_bytes: bytes,
    live_run_nonce: bytes,
    issuer_pid: int,
) -> bytes:
    if (
        type(bundle_content_id) is not bytes
        or len(bundle_content_id) != 32
        or type(daemon_receipt_binding) is not bytes
        or len(daemon_receipt_binding) != 32
        or type(canonical_bundle_bytes) is not bytes
        or not canonical_bundle_bytes
        or type(live_run_nonce) is not bytes
        or len(live_run_nonce) != 16
        or type(issuer_pid) is not int
        or issuer_pid <= 1
    ):
        raise TypeError("live admission token component is malformed")
    key_ids = dict(qualification_key_ids)
    if (
        set(key_ids) != {1, 2, 3, 4}
        or any(type(value) is not bytes or len(value) != 16 for value in key_ids.values())
        or len(set(key_ids.values())) != 4
    ):
        raise TypeError("live admission qualification key set is malformed")
    body = {
        "basis_commit": _require_commit(basis_commit),
        "bundle_content_id_hex": bundle_content_id.hex(),
        "canonical_bundle_sha256": hashlib.sha256(canonical_bundle_bytes).hexdigest(),
        "daemon_receipt_binding_hex": daemon_receipt_binding.hex(),
        "live_run_nonce_hex": live_run_nonce.hex(),
        "issuer_pid": issuer_pid,
        "qualification_key_ids": {
            str(purpose): key_ids[purpose].hex() for purpose in (1, 2, 3, 4)
        },
    }
    return LIVE_ADMISSION_TOKEN_MAC_DOMAIN + _canonical_json(body)


class LiveActorProtocolAdmissionV1:
    """Opaque, process-local, one-shot admission produced only by a live run.

    The process-random HMAC is an engineering capability boundary inside the
    disclosed same-admin threat model.  It is deliberately neither a hardware
    trust anchor nor protection against arbitrary monkeypatching in this
    Python process.
    """

    __slots__ = (
        "_basis_commit",
        "_bundle_content_id",
        "_qualification_key_ids",
        "_daemon_receipt_binding",
        "_canonical_bundle_bytes",
        "_live_run_nonce",
        "_issuer_pid",
        "_token_mac",
    )

    def __init__(
        self,
        *,
        basis_commit: str,
        bundle_content_id: bytes,
        qualification_key_ids: Mapping[int, bytes],
        daemon_receipt_binding: bytes,
        canonical_bundle_bytes: bytes,
        live_run_nonce: bytes,
        issuer_pid: int,
        token_mac: bytes,
        _seal: object,
    ) -> None:
        if _seal is not _LIVE_ADMISSION_CONSTRUCTOR_SEAL:
            raise TypeError("LiveActorProtocolAdmissionV1 is live-run-only")
        if type(token_mac) is not bytes or len(token_mac) != 32:
            raise TypeError("live admission token MAC is malformed")
        object.__setattr__(self, "_basis_commit", basis_commit)
        object.__setattr__(self, "_bundle_content_id", bundle_content_id)
        object.__setattr__(
            self,
            "_qualification_key_ids",
            MappingProxyType(dict(qualification_key_ids)),
        )
        object.__setattr__(self, "_daemon_receipt_binding", daemon_receipt_binding)
        object.__setattr__(self, "_canonical_bundle_bytes", canonical_bundle_bytes)
        object.__setattr__(self, "_live_run_nonce", live_run_nonce)
        object.__setattr__(self, "_issuer_pid", issuer_pid)
        object.__setattr__(self, "_token_mac", token_mac)

    def __repr__(self) -> str:
        return "LiveActorProtocolAdmissionV1(<opaque one-shot capability>)"

    def __reduce__(self) -> NoReturn:
        raise TypeError("live actor protocol admission tokens are not serializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live actor protocol admission tokens are not copyable")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("live actor protocol admission tokens are not copyable")


@dataclass(frozen=True, slots=True)
class ConsumedLiveActorProtocolAdmissionV1:
    """Immutable values released by one successful token consumption."""

    basis_commit: str
    bundle_content_id: bytes
    qualification_key_ids: Mapping[int, bytes]
    daemon_receipt_binding: bytes
    canonical_bundle_bytes: bytes


def consume_live_actor_protocol_admission_v1(
    token: LiveActorProtocolAdmissionV1,
    *,
    expected_basis_commit: str,
    expected_bundle_content_id: bytes | None = None,
    expected_daemon_receipt_binding: bytes | None = None,
) -> ConsumedLiveActorProtocolAdmissionV1:
    """Authenticate and consume exactly one process-local live admission."""

    if type(token) is not LiveActorProtocolAdmissionV1:
        _fail(FAIL_PROTOCOL, "live actor protocol admission token type differs")
    if token._issuer_pid != os.getpid():
        _fail(FAIL_PROTOCOL, "live actor protocol admission crossed a process boundary")
    commit = _require_commit(expected_basis_commit)
    if token._basis_commit != commit:
        _fail(FAIL_PROTOCOL, "live actor protocol admission binds another commit")
    if (
        expected_bundle_content_id is not None
        and token._bundle_content_id != expected_bundle_content_id
    ):
        _fail(FAIL_PROTOCOL, "live actor protocol admission bundle identity differs")
    if (
        expected_daemon_receipt_binding is not None
        and token._daemon_receipt_binding != expected_daemon_receipt_binding
    ):
        _fail(FAIL_PROTOCOL, "live actor protocol admission daemon identity differs")
    try:
        preimage = _admission_mac_preimage_v1(
            basis_commit=token._basis_commit,
            bundle_content_id=token._bundle_content_id,
            qualification_key_ids=token._qualification_key_ids,
            daemon_receipt_binding=token._daemon_receipt_binding,
            canonical_bundle_bytes=token._canonical_bundle_bytes,
            live_run_nonce=token._live_run_nonce,
            issuer_pid=token._issuer_pid,
        )
    except (TypeError, ActorProtocolQualificationError) as exc:
        _fail(FAIL_PROTOCOL, f"live actor protocol admission is malformed: {exc}")
    secret = _PROCESS_ADMISSION_SECRET
    if secret is None:
        _fail(FAIL_PROTOCOL, "live actor protocol admission issuer secret is absent")
    expected_mac = hmac.digest(secret, preimage, "sha256")
    if not hmac.compare_digest(token._token_mac, expected_mac):
        _fail(FAIL_PROTOCOL, "live actor protocol admission HMAC differs")
    # Replaying the immutable bytes here catches accidental corruption and
    # ensures the capability releases exactly the archive already validated by
    # the live qualifier, rather than a parallel mutable object graph.
    replayed = validate_actor_protocol_qualification_report_v1(
        _strict_json_object(token._canonical_bundle_bytes),
        expected_basis_commit=commit,
    )
    if (
        replayed.bundle_content_id != token._bundle_content_id
        or dict(replayed.qualification_key_ids) != dict(token._qualification_key_ids)
    ):
        _fail(FAIL_PROTOCOL, "live actor protocol admission replay identity differs")
    with _CONSUMED_ADMISSION_LOCK:
        if token._token_mac in _CONSUMED_ADMISSION_MACS:
            _fail(FAIL_PROTOCOL, "live actor protocol admission was already consumed")
        _CONSUMED_ADMISSION_MACS.add(token._token_mac)
    return ConsumedLiveActorProtocolAdmissionV1(
        basis_commit=token._basis_commit,
        bundle_content_id=token._bundle_content_id,
        qualification_key_ids=MappingProxyType(dict(token._qualification_key_ids)),
        daemon_receipt_binding=token._daemon_receipt_binding,
        canonical_bundle_bytes=token._canonical_bundle_bytes,
    )


def _validate_key_manifests_v1(
    rows: object, *, basis_commit: str, evidence: Mapping[str, object]
) -> dict[int, tuple[bytes, bytes, Mapping[str, object]]]:
    if type(rows) is not list or len(rows) != 4:
        _fail(FAIL_REPORT, "qualification key manifest set differs")
    profile = evidence["profile"]
    assert isinstance(profile, Mapping)
    result: dict[int, tuple[bytes, bytes, Mapping[str, object]]] = {}
    for expected_purpose, row in zip((1, 2, 3, 4), rows, strict=True):
        if type(row) is not dict:
            _fail(FAIL_REPORT, "qualification key manifest is not an object")
        body = dict(row)
        content_id = body.pop("manifest_content_id", None)
        if (
            set(body) != {
                "schema", "purpose_id", "usage", "public_key_32_hex",
                "key_id_16_hex", "basis_commit", "profile_sha256", "image_ref",
                "daemon_receipt_sha256", "container_id",
                "created_at_unix_seconds", "ephemeral_qualification_identity",
                "eligible_for_formal_actor_trust", "formal_genesis_reuse_forbidden",
                "must_destroy",
            }
            or body["schema"] != KEY_MANIFEST_SCHEMA
            or body["purpose_id"] != expected_purpose
            or body["usage"] != "LIVE_PROTOCOL_QUALIFICATION_ONLY"
            or body["basis_commit"] != basis_commit
            or body["profile_sha256"] != profile["profile_sha256"]
            or body["image_ref"] != profile["images"][str(expected_purpose)]
            or body["daemon_receipt_sha256"] != profile["daemon_receipt_sha256"]
            or body["container_id"] != evidence["actor_runtime_rows"][expected_purpose - 1]["container_id"]
            or body["created_at_unix_seconds"] != FROZEN_QUALIFICATION_TIMESTAMP
            or body["ephemeral_qualification_identity"] is not True
            or body["eligible_for_formal_actor_trust"] is not False
            or body["formal_genesis_reuse_forbidden"] is not True
            or body["must_destroy"] is not True
            or content_id != _content_id(KEY_MANIFEST_HASH_DOMAIN, body)
        ):
            _fail(FAIL_REPORT, "qualification key manifest policy/binding differs")
        try:
            public_key = bytes.fromhex(str(body["public_key_32_hex"]))
            key_id = bytes.fromhex(str(body["key_id_16_hex"]))
        except ValueError as exc:
            _fail(FAIL_REPORT, f"qualification key encoding is invalid: {exc}")
        if (
            len(public_key) != 32
            or len(key_id) != 16
            or hashlib.sha256(public_key).digest()[:16] != key_id
        ):
            _fail(FAIL_REPORT, "qualification key ID differs")
        result[expected_purpose] = (public_key, key_id, row)
    if len({item[0] for item in result.values()}) != 4:
        _fail(FAIL_REPORT, "one qualification key was reused across purposes")
    return result


def _expected_operation_environment_v1(
    *,
    evidence: Mapping[str, object],
    purpose: int,
    operation: str,
    sequence: int,
    nonce_hex: str,
    request_digest_hex: str,
) -> dict[str, str]:
    profile = evidence["profile"]
    assert isinstance(profile, Mapping)
    return {
        "HEGEL_ACTOR_IMAGE_REF": str(profile["images"][str(purpose)]),
        "HEGEL_ACTOR_PROFILE_ID": "hegel-owner-accepted-container-technical-actors-v1",
        "HEGEL_BASIS_COMMIT": str(evidence["basis_commit"]),
        "HEGEL_DAEMON_RECEIPT_SHA256": str(profile["daemon_receipt_sha256"]),
        "HEGEL_HOST_REPOSITORY_PATH_SHA256": str(
            profile["host_repository_path_sha256"]
        ),
        "HEGEL_PROFILE_SHA256": str(profile["profile_sha256"]),
        "HEGEL_PURPOSE_ID": str(purpose),
        "HEGEL_RUN_ID": str(evidence["qualification_run_id_hex"]),
        "LANG": "C",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HEGEL_OPERATION_ID": operation,
        "HEGEL_OPERATION_NONCE": nonce_hex,
        "HEGEL_OPERATION_REQUEST_SHA256": request_digest_hex,
        "HEGEL_OPERATION_SEQUENCE": str(sequence),
        "HEGEL_PROBE_INPUT_WRITE_PATH": "/input/.hegel-write-probe",
    }


def _validate_operation_evidence_v1(evidence: Mapping[str, object]) -> None:
    rows = evidence.get("operation_rows")
    if type(rows) is not list or len(rows) != len(EXPECTED_OPERATION_SEQUENCE):
        _fail(FAIL_REPORT, "operation evidence row count differs")
    nonces: set[str] = set()
    dummy = object.__new__(DockerCeremonyActorsV1)
    for expected, row in zip(EXPECTED_OPERATION_SEQUENCE, rows, strict=True):
        if type(row) is not dict or set(row) != {
            "purpose_id", "operation_id", "operation_sequence",
            "operation_nonce_hex", "request_binding", "actor_receipt",
            "rust_raw_probe_receipt_or_null", "host_receipt",
        }:
            _fail(FAIL_REPORT, "operation evidence row shape differs")
        purpose, sequence, operation = expected
        nonce_hex = row["operation_nonce_hex"]
        if (
            (row["purpose_id"], row["operation_sequence"], row["operation_id"])
            != expected
            or type(nonce_hex) is not str
            or re.fullmatch(r"[0-9a-f]{32}", nonce_hex) is None
            or nonce_hex in nonces
            or "seed" in operation
            or operation == "complete-marker"
        ):
            _fail(FAIL_REPORT, "operation order/nonce/no-seed policy differs")
        nonces.add(nonce_hex)
        binding = row["request_binding"]
        host = row["host_receipt"]
        actor = row["actor_receipt"]
        raw = row["rust_raw_probe_receipt_or_null"]
        if type(binding) is not dict or type(host) is not dict or type(actor) is not dict:
            _fail(FAIL_REPORT, "operation receipt/request body is absent")
        if set(binding) != {"kind", "body"}:
            _fail(FAIL_REPORT, "operation request binding field set differs")
        if binding.get("kind") == "GENERIC_INPUT_TREE_V1":
            request_body = binding.get("body")
            if type(request_body) is not dict:
                _fail(FAIL_REPORT, "generic request body is absent")
            input_rows = request_body.get("input_rows")
            request_digest = hashlib.sha256(
                OPERATION_REQUEST_HASH_DOMAIN + _canonical_json(request_body)
            ).hexdigest()
            if (
                set(request_body) != {
                    "schema", "basis_commit", "container_id",
                    "daemon_receipt_sha256", "input_rows", "operation_id",
                    "operation_nonce_hex", "operation_sequence",
                    "profile_sha256", "purpose_id", "run_id_hex",
                }
                or request_body.get("schema")
                != "hegel-phase3-m25-operation-request-binding/1"
                or request_body.get("basis_commit") != evidence["basis_commit"]
                or request_body.get("container_id") != host.get("container_id")
                or request_body.get("daemon_receipt_sha256")
                != evidence["profile"]["daemon_receipt_sha256"]
                or request_body.get("profile_sha256")
                != evidence["profile"]["profile_sha256"]
                or request_body.get("operation_id") != operation
                or request_body.get("operation_sequence") != sequence
                or request_body.get("operation_nonce_hex") != nonce_hex
                or request_body.get("purpose_id") != purpose
                or request_body.get("run_id_hex") != evidence["qualification_run_id_hex"]
                or type(input_rows) is not list
            ):
                _fail(FAIL_REPORT, "generic request preimage binding differs")
            observed_paths: list[str] = []
            for input_row in input_rows:
                if (
                    type(input_row) is not dict
                    or set(input_row) != {"path", "mode_octal", "size", "sha256"}
                    or type(input_row.get("path")) is not str
                    or not input_row["path"]
                    or input_row["path"].startswith("/")
                    or ".." in Path(input_row["path"]).parts
                    or type(input_row.get("mode_octal")) is not str
                    or re.fullmatch(r"0[0-7]{3}", input_row["mode_octal"]) is None
                    or type(input_row.get("size")) is not int
                    or input_row["size"] < 0
                    or type(input_row.get("sha256")) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", input_row["sha256"]) is None
                    or any(
                        marker in input_row["path"]
                        for marker in (
                            "split_master_seed", "split_seed_generation",
                            "split_seed_instantiation.marker",
                        )
                    )
                ):
                    _fail(FAIL_REPORT, "generic request input row differs")
                observed_paths.append(input_row["path"])
            if observed_paths != sorted(set(observed_paths)):
                _fail(FAIL_REPORT, "generic request input tree is not canonical")
        elif binding.get("kind") == "PURPOSE4_REQUEST_SHA256_V1" and purpose == 4 and operation == "purpose4-parent-sign":
            request_body = binding.get("body")
            if type(request_body) is not dict or type(request_body.get("request_sha256")) is not str:
                _fail(FAIL_REPORT, "purpose-4 request binding is absent")
            request_digest = str(request_body["request_sha256"])
        else:
            _fail(FAIL_REPORT, "operation request binding kind differs")
        host_body = dict(host)
        claimed_host_hash = host_body.pop("receipt_sha256", None)
        if (
            set(host) != {
                "schema", "basis_commit", "container_id",
                "daemon_receipt_sha256", "host_inspection",
                "live_probe_receipt_sha256", "namespaces", "operation_id",
                "operation_nonce_hex", "operation_request_sha256",
                "operation_sequence", "operation_sequence_scope",
                "purpose_id", "same_live_container_before_after",
                "receipt_sha256",
            }
            or host.get("schema") != HOST_OPERATION_RECEIPT_SCHEMA
            or claimed_host_hash != hashlib.sha256(_canonical_json(host_body)).hexdigest()
            or host.get("basis_commit") != evidence["basis_commit"]
            or host.get("purpose_id") != purpose
            or host.get("operation_id") != operation
            or host.get("operation_sequence") != sequence
            or host.get("operation_nonce_hex") != nonce_hex
            or host.get("operation_request_sha256") != request_digest
            or host.get("container_id") != evidence["actor_runtime_rows"][purpose - 1]["container_id"]
            or host.get("daemon_receipt_sha256")
            != evidence["profile"]["daemon_receipt_sha256"]
            or host.get("host_inspection")
            != evidence["actor_runtime_rows"][purpose - 1]["container_inspection"]
            or host.get("operation_sequence_scope")
            != "LIVE_CONTAINER_INCARNATION_ONLY"
            or host.get("same_live_container_before_after") is not True
        ):
            _fail(FAIL_REPORT, "host operation receipt binding differs")
        try:
            DockerCeremonyActorsV1._validate_namespace_rows(host.get("namespaces"))
        except FormalContainerExecutorError as exc:
            _fail(FAIL_REPORT, f"host operation namespace replay failed: {exc.code}")
        expected_environment = _expected_operation_environment_v1(
            evidence=evidence, purpose=purpose, operation=operation,
            sequence=sequence, nonce_hex=nonce_hex, request_digest_hex=request_digest,
        )
        try:
            if purpose == 3:
                if type(raw) is not dict:
                    _fail(FAIL_REPORT, "Rust raw operation receipt is absent")
                DockerCeremonyActorsV1._validate_rust_operation_receipt(
                    dummy, actor, raw, _canonical_json(raw), purpose=purpose,
                    operation=operation, sequence=sequence,
                    nonce=bytes.fromhex(nonce_hex),
                    request_digest=bytes.fromhex(request_digest),
                    expected_environment=expected_environment,
                )
                expected_live_sha = hashlib.sha256(_canonical_json(raw)).hexdigest()
            else:
                if raw is not None:
                    _fail(FAIL_REPORT, "non-Rust operation has a Rust raw receipt")
                DockerCeremonyActorsV1._validate_python_operation_receipt(
                    dummy, actor, _canonical_json(actor), purpose=purpose,
                    operation=operation, sequence=sequence,
                    nonce=bytes.fromhex(nonce_hex),
                    request_digest=bytes.fromhex(request_digest),
                    expected_environment=expected_environment,
                )
                expected_live_sha = hashlib.sha256(_canonical_json(actor)).hexdigest()
        except FormalContainerExecutorError as exc:
            _fail(FAIL_REPORT, f"operation receipt replay failed: {exc.code}")
        if host.get("live_probe_receipt_sha256") != expected_live_sha:
            _fail(FAIL_REPORT, "host/actor operation receipt response binding differs")


def _validate_purpose4_and_bridge_v1(
    evidence: Mapping[str, object],
    keys: Mapping[int, tuple[bytes, bytes, Mapping[str, object]]],
) -> None:
    p4 = evidence.get("purpose4_evidence")
    if type(p4) is not dict or set(p4) != {"request", "response"}:
        _fail(FAIL_REPORT, "purpose-4 replay evidence differs")
    request, response = p4["request"], p4["response"]
    if type(request) is not dict or type(response) is not dict:
        _fail(FAIL_REPORT, "purpose-4 request/response body is absent")
    try:
        result = validate_purpose4_keybearing_response_v1(
            response, request=request, signature_verifier=_signature_verifier
        )
    except Purpose4KeyBearingError as exc:
        _fail(FAIL_REPORT, f"purpose-4 strict replay failed: {exc}")
    if result.signer_public_key != keys[4][0] or result.signer_key_id != keys[4][1]:
        _fail(FAIL_REPORT, "purpose-4 response used another qualification actor")
    p4_operation = next(
        row for row in evidence["operation_rows"]
        if row["purpose_id"] == 4 and row["operation_id"] == "purpose4-parent-sign"
    )
    if (
        response.get("operation_probe_receipt") != p4_operation["actor_receipt"]
        or request != p4_operation["request_binding"]["body"]
    ):
        _fail(FAIL_REPORT, "purpose-4 response/request operation binding differs")

    bridge_rows = evidence.get("bridge_evidence_rows")
    if type(bridge_rows) is not list or len(bridge_rows) != 3:
        _fail(FAIL_REPORT, "bridge evidence set differs")
    for purpose, row in zip((1, 2, 3), bridge_rows, strict=True):
        if type(row) is not dict or set(row) != {
            "purpose_id", "package_base64", "package_size", "package_sha256",
            "replay_receipt", "bridge_signature_hex",
        } or row["purpose_id"] != purpose:
            _fail(FAIL_REPORT, "bridge evidence row shape/role differs")
        package = _decode_raw_base64(
            row["package_base64"], label=f"purpose-{purpose} bridge package",
            maximum=MAX_BRIDGE_PACKAGE_BYTES,
        )
        if row["package_size"] != len(package) or row["package_sha256"] != _sha256(package):
            _fail(FAIL_REPORT, "bridge package size/digest differs")
        signature_hex = row["bridge_signature_hex"]
        if (
            type(signature_hex) is not str
            or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None
        ):
            _fail(FAIL_REPORT, "bridge signature encoding is not exact lowercase hex")
        try:
            replay = replay_bridge_dag_package_v1(
                package, allow_authoritative=True,
                signature_verifier=_signature_verifier,
            )
            if (
                replay.purpose_id != purpose
                or replay.authoritative is not True
                or replay.eligible_to_sign_bridge_statement is not True
                or replay.purpose1_signature_verified is not (purpose != 1)
            ):
                _fail(FAIL_REPORT, "bridge replay authority/P1 verification differs")
            validate_bridge_actor_replay_receipt_v1(
                _canonical_json(row["replay_receipt"]),
                expected_result=replay,
                expected_implementation=(
                    "rust-full-dag-replay-v1" if purpose == 3
                    else "python-full-dag-replay-v1"
                ),
                require_authoritative=True,
            )
            signature = bytes.fromhex(signature_hex)
            _verify_ed25519(
                keys[purpose][0], signature,
                bridge_attestation_signature_preimage_v1(
                    replay.bridge_statement_root, purpose, 0
                ),
            )
        except (BridgeDagReplayError, ValueError) as exc:
            _fail(FAIL_REPORT, f"bridge strict replay/signature failed: {exc}")


def _validate_destruction_plan_v1(
    plan: object, *, evidence: Mapping[str, object]
) -> Mapping[str, object]:
    if type(plan) is not dict:
        _fail(FAIL_REPORT, "signed destruction plan is absent")
    body = dict(plan)
    content_id = body.pop("destruction_plan_content_id", None)
    if (
        set(body) != {
            "schema", "basis_commit", "qualification_run_id_hex",
            "daemon_receipt_sha256", "actor_rows", "required_cleanup_order",
            "seed_or_marker_artifacts_must_remain_absent",
            "qualification_keys_must_be_destroyed",
            "formal_genesis_reuse_forbidden",
        }
        or body["schema"] != DESTRUCTION_PLAN_SCHEMA
        or body["basis_commit"] != evidence["basis_commit"]
        or body["qualification_run_id_hex"] != evidence["qualification_run_id_hex"]
        or body["daemon_receipt_sha256"] != evidence["profile"]["daemon_receipt_sha256"]
        or body["required_cleanup_order"] != [
            "CONTAINERS_REMOVED_AND_VERIFIED_ABSENT",
            "KEY_VOLUMES_REMOVED_AND_VERIFIED_ABSENT",
            "EXACT_RESERVATIONS_REMOVED",
        ]
        or body["seed_or_marker_artifacts_must_remain_absent"] is not True
        or body["qualification_keys_must_be_destroyed"] is not True
        or body["formal_genesis_reuse_forbidden"] is not True
        or content_id != _content_id(DESTRUCTION_PLAN_HASH_DOMAIN, body)
    ):
        _fail(FAIL_REPORT, "destruction plan semantics/hash differs")
    actor_rows = body["actor_rows"]
    if type(actor_rows) is not list or len(actor_rows) != 4:
        _fail(FAIL_REPORT, "destruction plan actor rows differ")
    for purpose, row in zip((1, 2, 3, 4), actor_rows, strict=True):
        expected_volume_name = (
            f"hegel-m25-state-{body['qualification_run_id_hex']}-p{purpose}"
        )
        if type(row) is not dict or set(row) != {
            "purpose_id", "container_id", "actor_key_volume_name",
            "must_remove_container", "must_remove_actor_key_volume",
        } or row["purpose_id"] != purpose or row["container_id"] != evidence["actor_runtime_rows"][purpose - 1]["container_id"] or row["actor_key_volume_name"] != expected_volume_name or row["must_remove_container"] is not True or row["must_remove_actor_key_volume"] is not True:
            _fail(FAIL_REPORT, "destruction plan actor identity differs")
    return plan


def _validate_cleanup_receipt_v1(
    receipt: object, *, evidence: Mapping[str, object], plan: Mapping[str, object]
) -> Mapping[str, object]:
    if type(receipt) is not dict:
        _fail(FAIL_REPORT, "post-cleanup absence receipt is absent")
    body = dict(receipt)
    content_id = body.pop("cleanup_receipt_content_id", None)
    if (
        set(body) != {
            "schema", "basis_commit", "qualification_run_id_hex",
            "daemon_receipt_sha256", "destruction_plan_content_id", "actor_rows",
            "seed_marker_absent", "seed_intent_absent", "seed_completion_absent",
            "raw_seed_absent", "exact_reservations_removed",
            "custody_directory_empty", "custody_directory_removed",
        }
        or body["schema"] != CLEANUP_RECEIPT_SCHEMA
        or body["basis_commit"] != evidence["basis_commit"]
        or body["qualification_run_id_hex"] != evidence["qualification_run_id_hex"]
        or body["daemon_receipt_sha256"] != evidence["profile"]["daemon_receipt_sha256"]
        or body["destruction_plan_content_id"] != plan["destruction_plan_content_id"]
        or any(body[name] is not True for name in (
            "seed_marker_absent", "seed_intent_absent", "seed_completion_absent",
            "raw_seed_absent", "exact_reservations_removed", "custody_directory_empty",
        ))
        or body["custody_directory_removed"] is not False
        or content_id != _content_id(CLEANUP_RECEIPT_HASH_DOMAIN, body)
    ):
        _fail(FAIL_REPORT, "post-cleanup absence receipt semantics/hash differs")
    actor_rows = body["actor_rows"]
    plan_rows = plan["actor_rows"]
    if type(actor_rows) is not list or len(actor_rows) != 4:
        _fail(FAIL_REPORT, "post-cleanup actor rows differ")
    for purpose, row, planned in zip((1, 2, 3, 4), actor_rows, plan_rows, strict=True):
        returncode_names = (
            "container_inspect_returncode",
            "container_list_returncode",
            "volume_inspect_returncode",
            "volume_list_returncode",
        )
        if type(row) is not dict or set(row) != {
            "purpose_id", "container_id", "actor_key_volume_name",
            "container_inspect_returncode", "container_inspect_stdout_sha256",
            "container_list_returncode", "container_list_stdout_sha256",
            "volume_inspect_returncode", "volume_inspect_stdout_sha256",
            "volume_list_returncode", "volume_list_stdout_sha256",
            "container_verified_absent", "actor_key_volume_verified_absent",
        } or any(type(row[name]) is not int for name in returncode_names) or row["purpose_id"] != purpose or row["container_id"] != planned["container_id"] or row["actor_key_volume_name"] != planned["actor_key_volume_name"] or row["container_inspect_returncode"] == 0 or row["volume_inspect_returncode"] == 0 or row["container_list_returncode"] != 0 or row["volume_list_returncode"] != 0 or row["container_list_stdout_sha256"] != _sha256(b"") or row["volume_list_stdout_sha256"] != _sha256(b"") or row["container_verified_absent"] is not True or row["actor_key_volume_verified_absent"] is not True:
            _fail(FAIL_REPORT, "post-cleanup exact absence row differs")
        for key in (
            "container_inspect_stdout_sha256", "container_list_stdout_sha256",
            "volume_inspect_stdout_sha256", "volume_list_stdout_sha256",
        ):
            _require_sha256(row[key], key)
    return receipt


def _validate_statements_v1(
    rows: object,
    *,
    evidence: Mapping[str, object],
    evidence_content_id: str,
    destruction_plan_content_id: str,
    keys: Mapping[int, tuple[bytes, bytes, Mapping[str, object]]],
) -> None:
    if type(rows) is not list or len(rows) != 4:
        _fail(FAIL_REPORT, "qualification requires exactly four statements")
    prior_sequences = {purpose: 0 for purpose in (1, 2, 3, 4)}
    for purpose, sequence, _operation in EXPECTED_OPERATION_SEQUENCE:
        prior_sequences[purpose] = max(prior_sequences[purpose], sequence)
    operation_nonces = {row["operation_nonce_hex"] for row in evidence["operation_rows"]}
    statement_nonces: set[str] = set()
    for purpose, envelope in zip((1, 2, 3, 4), rows, strict=True):
        if type(envelope) is not dict or set(envelope) != {
            "schema", "purpose_id", "statement", "container_inspection",
            "finalize_request", "finalize_probe_receipt",
            "finalize_live_probe_receipt",
            "signature_hex", "signature_verified_before_actor_destruction",
        } or envelope["schema"] != SIGNATURE_ENVELOPE_SCHEMA or envelope["purpose_id"] != purpose or envelope["signature_verified_before_actor_destruction"] is not True:
            _fail(FAIL_REPORT, "qualification signature envelope differs")
        statement = envelope["statement"]
        inspection = envelope["container_inspection"]
        finalize_request = envelope["finalize_request"]
        finalize_probe = envelope["finalize_probe_receipt"]
        finalize_live_probe = envelope["finalize_live_probe_receipt"]
        if any(
            type(value) is not dict
            for value in (
                statement,
                inspection,
                finalize_request,
                finalize_probe,
                finalize_live_probe,
            )
        ):
            _fail(FAIL_REPORT, "qualification statement/inspection body is absent")
        base = dict(statement)
        request_digest = base.pop("qualification_finalize_request_sha256", None)
        nonce = statement.get("operation_nonce_hex")
        if (
            set(base) != {
                "schema", "operation_id", "purpose_id", "basis_commit",
                "qualification_run_id_hex", "qualification_evidence_content_id",
                "destruction_plan_content_id",
                "qualification_key_manifest_content_id", "profile_sha256",
                "image_ref", "daemon_receipt_sha256", "container_id",
                "key_id_16_hex", "operation_sequence", "operation_nonce_hex",
                "container_inspection_sha256", "formal_authority",
                "formal_gates_before", "formal_gates_after", "m3_state", "m3_started",
                "real_seed_generated", "real_seed_accessed", "formal_output_published",
                "qualification_identity_usage", "eligible_for_formal_actor_trust",
                "formal_genesis_reuse_forbidden", "must_destroy_after_qualification",
                "independence_disclosure",
            }
            or statement["schema"] != STATEMENT_SCHEMA
            or statement["operation_id"] != "qualification-finalize"
            or statement["purpose_id"] != purpose
            or statement["basis_commit"] != evidence["basis_commit"]
            or statement["qualification_run_id_hex"] != evidence["qualification_run_id_hex"]
            or statement["qualification_evidence_content_id"] != evidence_content_id
            or statement["destruction_plan_content_id"] != destruction_plan_content_id
            or statement["qualification_key_manifest_content_id"] != keys[purpose][2]["manifest_content_id"]
            or statement["profile_sha256"] != evidence["profile"]["profile_sha256"]
            or statement["image_ref"] != evidence["profile"]["images"][str(purpose)]
            or statement["daemon_receipt_sha256"] != evidence["profile"]["daemon_receipt_sha256"]
            or statement["container_id"] != evidence["actor_runtime_rows"][purpose - 1]["container_id"]
            or statement["key_id_16_hex"] != keys[purpose][1].hex()
            or statement["operation_sequence"] != prior_sequences[purpose] + 1
            or type(nonce) is not str
            or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
            or nonce in operation_nonces
            or nonce in statement_nonces
            or inspection != evidence["actor_runtime_rows"][purpose - 1]["container_inspection"]
            or statement["container_inspection_sha256"] != _sha256(_canonical_json(inspection))
            or statement["formal_authority"] is not False
            or statement["formal_gates_before"] != 14
            or statement["formal_gates_after"] != 14
            or statement["m3_state"] != "NOT_RUN"
            or statement["m3_started"] is not False
            or statement["real_seed_generated"] is not False
            or statement["real_seed_accessed"] is not False
            or statement["formal_output_published"] is not False
            or statement["qualification_identity_usage"] != "LIVE_PROTOCOL_QUALIFICATION_ONLY"
            or statement["eligible_for_formal_actor_trust"] is not False
            or statement["formal_genesis_reuse_forbidden"] is not True
            or statement["must_destroy_after_qualification"] is not True
            or statement["independence_disclosure"] != dict(INDEPENDENCE_DISCLOSURE)
            or request_digest != _content_id(STATEMENT_REQUEST_HASH_DOMAIN, base)
        ):
            _fail(FAIL_REPORT, "qualification statement binding/claim differs")
        statement_nonces.add(nonce)
        preimage = _qualification_statement_preimage_v1(purpose, statement)
        if finalize_request != {
            "schema": "hegel-phase3-m25-protocol-qualification-finalize-request/1",
            "purpose_id": purpose,
            "statement": statement,
            "preimage_sha256": _sha256(preimage),
        }:
            _fail(FAIL_REPORT, "qualification finalize request/preimage differs")
        operation_request_digest = hashlib.sha256(
            _canonical_json(finalize_request)
        ).hexdigest()
        signature_hex = envelope["signature_hex"]
        if (
            type(signature_hex) is not str
            or re.fullmatch(r"[0-9a-f]{128}", signature_hex) is None
        ):
            _fail(FAIL_REPORT, "qualification signature encoding is not exact lowercase hex")
        try:
            signature = bytes.fromhex(signature_hex)
        except ValueError as exc:
            _fail(FAIL_REPORT, f"qualification signature encoding is invalid: {exc}")
        expected_finalize_probe = {
            "live_probe_sha256": hashlib.sha256(
                _canonical_json(finalize_live_probe)
            ).hexdigest(),
            "operation_id": "qualification-finalize",
            "operation_nonce_hex": nonce,
            "operation_request_sha256": operation_request_digest,
            "operation_sequence": prior_sequences[purpose] + 1,
            "preimage_sha256": hashlib.sha256(preimage).hexdigest(),
            "purpose_id": purpose,
            "schema": "hegel-phase3-m25-protocol-qualification-finalize-probe/1",
            "signature_sha256": hashlib.sha256(signature).hexdigest(),
        }
        if finalize_probe != expected_finalize_probe:
            _fail(FAIL_REPORT, "qualification finalize worker receipt differs")
        expected_environment = _expected_operation_environment_v1(
            evidence=evidence,
            purpose=purpose,
            operation="qualification-finalize",
            sequence=prior_sequences[purpose] + 1,
            nonce_hex=nonce,
            request_digest_hex=operation_request_digest,
        )
        expected_environment["HEGEL_QUALIFICATION_PREIMAGE_SHA256"] = (
            hashlib.sha256(preimage).hexdigest()
        )
        expected_live_keys = {
            "environment", "filesystem_probes", "identity", "implementation",
            "namespaces", "network_interfaces", "open_fds", "proc_status",
            "profile_id", "purpose_id", "schema", "syscall_probes",
        }
        if (
            set(finalize_live_probe) != expected_live_keys
            or finalize_live_probe.get("schema")
            != "hegel-container-actor-live-probe/1"
            or finalize_live_probe.get("implementation")
            != ("rust-ffi-v1" if purpose == 3 else "python-ctypes-v1")
            or finalize_live_probe.get("profile_id")
            != "hegel-owner-accepted-container-technical-actors-v1"
        ):
            _fail(FAIL_REPORT, "qualification finalize live probe shape differs")
        try:
            DockerCeremonyActorsV1._validate_common_probe_fields(
                finalize_live_probe,
                expected_environment=expected_environment,
                purpose=purpose,
            )
        except FormalContainerExecutorError as exc:
            _fail(FAIL_REPORT, f"qualification finalize live probe failed: {exc.code}")
        _verify_ed25519(
            keys[purpose][0], signature,
            preimage,
        )


def _validate_evidence_v1(evidence: object, *, basis_commit: str) -> str:
    if type(evidence) is not dict or set(evidence) != {
        "schema", "basis_commit", "qualification_run_id_hex", "profile",
        "actor_runtime_rows", "operation_rows", "purpose4_evidence",
        "bridge_evidence_rows", "public_synthetic_fixture",
        "protocol_call_graph", "formal_track_claims",
    }:
        _fail(FAIL_REPORT, "qualification evidence field set differs")
    if (
        evidence["schema"] != EVIDENCE_SCHEMA
        or evidence["basis_commit"] != basis_commit
        or type(evidence["qualification_run_id_hex"]) is not str
        or re.fullmatch(r"[0-9a-f]{32}", evidence["qualification_run_id_hex"]) is None
    ):
        _fail(FAIL_REPORT, "qualification evidence identity differs")
    profile = evidence["profile"]
    runtime_rows = evidence["actor_runtime_rows"]
    fixture = evidence["public_synthetic_fixture"]
    call_graph = evidence["protocol_call_graph"]
    claims = evidence["formal_track_claims"]
    if type(profile) is not dict or set(profile) != {
        "profile_sha256", "images", "daemon_receipt", "daemon_receipt_sha256",
        "host_repository_path_sha256",
    } or type(profile["images"]) is not dict or set(profile["images"]) != {"1", "2", "3", "4"}:
        _fail(FAIL_REPORT, "qualification profile binding differs")
    _require_sha256("sha256:" + str(profile["profile_sha256"]), "profile")
    _require_sha256("sha256:" + str(profile["daemon_receipt_sha256"]), "daemon")
    _require_sha256(
        "sha256:" + str(profile["host_repository_path_sha256"]),
        "host repository path",
    )
    try:
        if (
            not isinstance(profile["daemon_receipt"], Mapping)
            or local_docker_daemon_receipt_binding_v1(
                profile["daemon_receipt"]
            ).hex()
            != profile["daemon_receipt_sha256"]
        ):
            _fail(FAIL_REPORT, "daemon receipt body/binding differs")
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_REPORT, f"daemon receipt replay failed: {exc}")
    profile_blob = _git((
        "show",
        f"{basis_commit}:Hegel Machine/config/phase3_container_actor_profile_v1.json",
    ))
    try:
        committed_profile = json.loads(profile_blob)
        committed_images = committed_profile["images"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        _fail(FAIL_REPORT, f"committed actor profile cannot be replayed: {exc}")
    image_names = {
        1: "custodian",
        2: "python_attester",
        3: "rust_attester",
        4: "policy_auditor",
    }
    if (
        type(committed_profile) is not dict
        or type(committed_images) is not dict
        or profile["profile_sha256"] != hashlib.sha256(profile_blob).hexdigest()
        or profile["images"]
        != {
            str(purpose): committed_images[image_names[purpose]]
            for purpose in (1, 2, 3, 4)
        }
    ):
        _fail(FAIL_REPORT, "actor profile/image registry differs from Commit A")
    for purpose, image in profile["images"].items():
        if (
            purpose not in {"1", "2", "3", "4"}
            or type(image) is not str
            or re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image) is None
        ):
            _fail(FAIL_REPORT, "actor image binding is not digest-pinned")
    if type(runtime_rows) is not list or len(runtime_rows) != 4:
        _fail(FAIL_REPORT, "actor runtime row set differs")
    for purpose, row in zip((1, 2, 3, 4), runtime_rows, strict=True):
        if type(row) is not dict or set(row) != {
            "purpose_id", "container_id", "container_inspection",
            "volume_initialization_receipt",
        } or row["purpose_id"] != purpose or type(row["container_id"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["container_id"]) is None or type(row["container_inspection"]) is not dict or type(row["volume_initialization_receipt"]) is not dict:
            _fail(FAIL_REPORT, "actor runtime identity row differs")
        inspection = row["container_inspection"]
        volume = row["volume_initialization_receipt"]
        assert isinstance(inspection, dict) and isinstance(volume, dict)
        inspection_body = dict(inspection)
        inspection_hash = inspection_body.pop("inspection_sha256", None)
        volume_body = dict(volume)
        volume_hash = volume_body.pop("receipt_sha256", None)
        checks = inspection.get("checks")
        expected_inspection_checks = {
            "container_id_exact", "container_name_exact", "running",
            "image_reference_exact", "image_id_digest_exact",
            "user_nonroot_exact", "pid1_env_i_command_exact",
            "entrypoint_exact", "network_none", "read_only_root",
            "not_privileged", "capabilities_exact", "security_options_exact",
            "runtime_seccomp_exact", "resource_limits_exact", "nofile_exact",
            "ipc_private", "tmpfs_private_exact", "mount_set_exact",
        }
        expected_volume_checks = {
            "name_exact", "driver_local", "scope_local", "options_empty",
            "labels_exact", "daemon_managed_mountpoint_exact",
            "not_bind_nfs_or_plugin",
        }
        volume_identity = volume.get("volume_identity")
        volume_checks = (
            volume_identity.get("checks")
            if isinstance(volume_identity, Mapping)
            else None
        )
        expected_mounts = sorted(
            ["/input", "/output", "/state"]
            + (["/custody"] if purpose == 1 else [])
        )
        volume_name = (
            f"hegel-m25-state-{evidence['qualification_run_id_hex']}-p{purpose}"
        )
        if (
            set(inspection) != {
                "checks", "container_id", "container_name", "host_pid",
                "image_ref", "mount_destinations", "purpose_id",
                "inspection_sha256",
            }
            or inspection_hash != hashlib.sha256(_canonical_json(inspection_body)).hexdigest()
            or inspection.get("container_id") != row["container_id"]
            or inspection.get("image_ref") != profile["images"][str(purpose)]
            or inspection.get("purpose_id") != purpose
            or type(inspection.get("container_name")) is not str
            or re.fullmatch(
                rf"hegel-m25-formal-p{purpose}-[0-9a-f]{{16}}",
                str(inspection.get("container_name")),
            ) is None
            or type(inspection.get("host_pid")) is not int
            or inspection["host_pid"] <= 1
            or inspection.get("mount_destinations") != expected_mounts
            or type(checks) is not dict
            or set(checks) != expected_inspection_checks
            or not all(item is True for item in checks.values())
            or set(volume) != {
                "schema", "basis_commit", "run_id_hex", "purpose_id",
                "volume_name_sha256", "image_sha256", "profile_sha256",
                "initializer_network_none", "initializer_capabilities",
                "nonroot_live_write_stat_probe_passed", "resulting_uid",
                "resulting_gid", "resulting_mode_octal", "volume_identity",
                "daemon_receipt_sha256", "receipt_sha256",
            }
            or volume_hash != hashlib.sha256(_canonical_json(volume_body)).hexdigest()
            or volume.get("schema")
            != "hegel-phase3-m25-private-volume-initialization-receipt/1"
            or volume.get("purpose_id") != purpose
            or volume.get("basis_commit") != basis_commit
            or volume.get("run_id_hex") != evidence["qualification_run_id_hex"]
            or volume.get("volume_name_sha256")
            != hashlib.sha256(volume_name.encode("ascii")).hexdigest()
            or volume.get("image_sha256")
            != str(profile["images"][str(purpose)]).rsplit(":", 1)[-1]
            or volume.get("profile_sha256") != profile["profile_sha256"]
            or volume.get("daemon_receipt_sha256")
            != profile["daemon_receipt_sha256"]
            or volume.get("initializer_network_none") is not True
            or volume.get("initializer_capabilities") != ["CHOWN"]
            or volume.get("nonroot_live_write_stat_probe_passed") is not True
            or volume.get("resulting_uid") != 65534
            or volume.get("resulting_gid") != 65534
            or volume.get("resulting_mode_octal") != "0700"
            or not isinstance(volume_identity, Mapping)
            or set(volume_identity) != {
                "driver", "scope", "options_empty",
                "daemon_managed_mountpoint_sha256", "checks",
            }
            or volume_identity.get("driver") != "local"
            or volume_identity.get("scope") != "local"
            or volume_identity.get("options_empty") is not True
            or type(volume_identity.get("daemon_managed_mountpoint_sha256"))
            is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(volume_identity.get("daemon_managed_mountpoint_sha256")),
            ) is None
            or type(volume_checks) is not dict
            or set(volume_checks) != expected_volume_checks
            or not all(item is True for item in volume_checks.values())
        ):
            _fail(FAIL_REPORT, "actor inspection/volume receipt replay differs")
    if type(fixture) is not dict or set(fixture) != {
        "split_frame_base64", "split_frame_size", "split_frame_sha256",
        "contains_assignments", "contains_real_seed",
    }:
        _fail(FAIL_REPORT, "public synthetic fixture shape differs")
    frame = _decode_raw_base64(
        fixture["split_frame_base64"], label="public synthetic split frame", maximum=4096
    )
    if frame != PUBLIC_SYNTHETIC_SPLIT_FRAME or fixture != {
        "split_frame_base64": _raw_base64(PUBLIC_SYNTHETIC_SPLIT_FRAME),
        "split_frame_size": len(PUBLIC_SYNTHETIC_SPLIT_FRAME),
        "split_frame_sha256": PUBLIC_SYNTHETIC_SPLIT_FRAME_SHA256,
        "contains_assignments": False,
        "contains_real_seed": False,
    }:
        _fail(FAIL_REPORT, "public synthetic split fixture changed")
    require_full_split_response_agreement_v2(frame, frame)
    if call_graph != {
        "keygen_order": [1, 2, 3, 4],
        "purpose1_authorized_object_names": [
            "SplitSeedCommitmentManifestV1", "CustodianBindingManifestV1",
            "SeedContinuityManifestV1", "HiddenAccessLedgerRecordV1",
        ],
        "purpose4_detached_sign_count": 1,
        "bridge_replay_order": [1, 2, 3],
        "docker_seed_or_marker_method_call_count": 0,
    } or claims != {
        "formal_authority": False,
        "authoritative_formal_roots_generated": False,
        "synthetic_formal_shaped_roots_computed_in_memory": True,
        "formal_roots_published": False,
        "gate_evidence_published": False,
        "formal_gates_before": 14,
        "formal_gates_after": 14,
        "m3_state": "NOT_RUN",
        "m3_started": False,
        "real_seed_generated": False,
        "real_seed_accessed": False,
    }:
        _fail(FAIL_REPORT, "protocol call graph/formal claims differ")
    _validate_operation_evidence_v1(evidence)
    return _content_id(EVIDENCE_HASH_DOMAIN, evidence)


def _validate_implementation_bindings_v1(
    bindings: object, *, basis_commit: str
) -> None:
    if type(bindings) is not dict or set(bindings) != {
        "formal_rust_replay_binary_sha256",
        "bridge_rust_replay_binary_sha256",
        "bridge_rust_qualification_report_sha256",
        "m3_implementation_qualification_receipt_sha256",
        "m3_implementation_qualification_receipt",
    }:
        _fail(FAIL_REPORT, "implementation binding field set differs")
    for key in (
        "formal_rust_replay_binary_sha256",
        "bridge_rust_replay_binary_sha256",
        "bridge_rust_qualification_report_sha256",
        "m3_implementation_qualification_receipt_sha256",
    ):
        _require_sha256(bindings[key], key)
    receipt = bindings["m3_implementation_qualification_receipt"]
    if type(receipt) is not dict:
        _fail(FAIL_REPORT, "M3 implementation qualification receipt body is absent")
    if bindings["m3_implementation_qualification_receipt_sha256"] != _sha256(
        _canonical_json(receipt)
    ):
        _fail(FAIL_REPORT, "M3 implementation qualification receipt digest differs")
    try:
        golden, _golden_preimage, _golden_root = load_committed_dual_golden_v1(
            REPOSITORY_ROOT, basis_commit
        )
        validate_qualification_receipt_v1(
            receipt, golden=golden, basis_commit=basis_commit
        )
        bridge_report, bridge_digest = (
            load_qualified_rust_bridge_dag_binary_binding_v1(
                expected_basis_commit=basis_commit,
                report_path=DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
            )
        )
    except (
        M3ImplementationQualificationError,
        BridgeDagBinaryQualificationError,
        OSError,
    ) as exc:
        _fail(FAIL_REPORT, f"implementation binding replay failed: {type(exc).__name__}")
    if (
        bindings["formal_rust_replay_binary_sha256"]
        != _hash_regular_file(DEFAULT_RUST_BINARY, code=FAIL_REPORT)
        or bindings["bridge_rust_replay_binary_sha256"] != bridge_digest
        or bindings["bridge_rust_qualification_report_sha256"]
        != str(bridge_report["diagnostic_report_sha256"])
    ):
        _fail(FAIL_REPORT, "implementation executable/report binding differs")


def validate_actor_protocol_qualification_report_v1(
    report: Mapping[str, object],
    *,
    expected_basis_commit: str | None = None,
    verify_commit_sources: bool = True,
    verify_local_implementation_bindings: bool = True,
) -> ReplayedActorProtocolQualificationV1:
    """Replay a qualification archive.

    The two verification switches exist for the Commit-B public-tree verifier,
    which independently checks the same source and implementation bindings
    against Git blobs supplied by its caller.  Ordinary qualification callers
    retain the stronger defaults and therefore still require the live
    worktree, persisted Rust binaries, and local qualification report.
    """

    if type(verify_commit_sources) is not bool or type(
        verify_local_implementation_bindings
    ) is not bool:
        _fail(FAIL_REPORT, "qualification replay verification switches are not booleans")
    if not isinstance(report, Mapping):
        _fail(FAIL_REPORT, "diagnostic archive is not an object")
    value = dict(report)
    if set(value) != {
        "schema_version", "artifact_kind", "status", "claim_level",
        "basis_commit", "commit_a_source_set_sha256", "commit_a_source_file_count",
        "authority_boundary", "independence_disclosure", "implementation_bindings",
        "qualification_key_manifests", "evidence_bundle",
        "destruction_plan", "cleanup_absence_receipt", "qualification_statements",
        "bundle_content_id", "diagnostic_report_sha256",
    }:
        _fail(FAIL_REPORT, "qualification archive field set differs")
    basis_commit = _require_commit(value["basis_commit"])
    if expected_basis_commit is not None and basis_commit != _require_commit(expected_basis_commit):
        _fail(FAIL_REPORT, "qualification archive binds another basis commit")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["artifact_kind"] != ARTIFACT_KIND
        or value["status"] != STATUS
        or value["claim_level"] != CLAIM_LEVEL
        or value["authority_boundary"] != dict(AUTHORITY_BOUNDARY)
        or value["independence_disclosure"] != dict(INDEPENDENCE_DISCLOSURE)
        or type(value["commit_a_source_file_count"]) is not int
        or value["commit_a_source_file_count"] < 1
    ):
        _fail(FAIL_REPORT, "qualification archive status/authority differs")
    _require_sha256(value["commit_a_source_set_sha256"], "source set")
    if verify_commit_sources:
        expected_source_digest, expected_source_count = (
            _commit_source_set_digest_from_git_v1(basis_commit)
        )
        if (
            value["commit_a_source_set_sha256"] != expected_source_digest
            or value["commit_a_source_file_count"] != expected_source_count
        ):
            _fail(FAIL_REPORT, "qualification source set differs from Commit A")
    if verify_local_implementation_bindings:
        _validate_implementation_bindings_v1(
            value["implementation_bindings"], basis_commit=basis_commit
        )
    evidence = value["evidence_bundle"]
    evidence_content_id = _validate_evidence_v1(evidence, basis_commit=basis_commit)
    assert isinstance(evidence, Mapping)
    keys = _validate_key_manifests_v1(
        value["qualification_key_manifests"], basis_commit=basis_commit, evidence=evidence
    )
    _validate_purpose4_and_bridge_v1(evidence, keys)
    plan = _validate_destruction_plan_v1(
        value["destruction_plan"], evidence=evidence
    )
    _validate_statements_v1(
        value["qualification_statements"], evidence=evidence,
        evidence_content_id=evidence_content_id,
        destruction_plan_content_id=str(plan["destruction_plan_content_id"]), keys=keys,
    )
    cleanup = _validate_cleanup_receipt_v1(
        value["cleanup_absence_receipt"], evidence=evidence, plan=plan
    )
    authority_preimage = {
        "basis_commit": basis_commit,
        "evidence_content_id": evidence_content_id,
        "destruction_plan_content_id": plan["destruction_plan_content_id"],
        "cleanup_receipt_content_id": cleanup["cleanup_receipt_content_id"],
        "qualification_key_manifest_content_ids": [
            keys[purpose][2]["manifest_content_id"] for purpose in (1, 2, 3, 4)
        ],
        "qualification_statement_sha256": [
            _sha256(_canonical_json(row)) for row in value["qualification_statements"]
        ],
    }
    expected_bundle_content_id = _content_id(
        BUNDLE_AUTHORITY_HASH_DOMAIN, authority_preimage
    )
    if value["bundle_content_id"] != expected_bundle_content_id:
        _fail(FAIL_REPORT, "validated bundle authority content ID differs")
    claimed = value["diagnostic_report_sha256"]
    body = dict(value)
    body.pop("diagnostic_report_sha256")
    if claimed != _report_hash(body):
        _fail(FAIL_REPORT, "qualification archive content ID differs")
    bundle_id = bytes.fromhex(
        _require_sha256(value["bundle_content_id"], "bundle authority ID").removeprefix("sha256:")
    )
    return ReplayedActorProtocolQualificationV1(
        basis_commit=basis_commit,
        bundle_content_id=bundle_id,
        qualification_key_ids=MappingProxyType({
            purpose: row[1] for purpose, row in keys.items()
        }),
        report=MappingProxyType(value),
    )


def canonical_actor_protocol_qualification_report_bytes_v1(
    report: Mapping[str, object],
) -> bytes:
    validated = validate_actor_protocol_qualification_report_v1(report)
    return _canonical_json(dict(validated.report))


def load_actor_protocol_qualification_report_v1(
    path: Path = DEFAULT_REPORT_PATH,
    *,
    expected_basis_commit: str | None = None,
) -> ReplayedActorProtocolQualificationV1:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(FAIL_REPORT, f"cannot read qualification archive: {exc}")
    return validate_actor_protocol_qualification_report_v1(
        _strict_json_object(payload), expected_basis_commit=expected_basis_commit
    )


def qualify_live_actor_protocol_v1(
    *,
    basis_commit: str,
    custody_directory: Path,
) -> LiveActorProtocolAdmissionV1:
    """Run the live actor protocol without any Docker seed/marker operation."""

    global _PROCESS_ADMISSION_SECRET
    commit = _require_commit(basis_commit)
    source_set_sha256, source_file_count = _commit_source_set_digest_v1(commit)
    try:
        basis = build_qualified_formal_static_basis_v1(commit)
        implementation_roots = require_formal_ceremony_ready_v1(basis)
        python_receipt = build_python_static_replay_receipt_v1(basis)
        rust_binary = Path(str(basis.implementation_inputs["rust_binary_path"]))
        bridge_report, bridge_digest = load_qualified_rust_bridge_dag_binary_binding_v1(
            expected_basis_commit=commit,
            report_path=DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
        )
    except (
        FormalContainerExecutorError,
        M3ImplementationQualificationError,
        BridgeDagBinaryQualificationError,
        OSError,
    ) as exc:
        _fail(FAIL_PROTOCOL, f"Commit-A implementation binding failed: {exc}")
    if bridge_digest != _hash_regular_file(DEFAULT_RUST_BRIDGE_DAG_BINARY):
        _fail(FAIL_PROTOCOL, "qualified bridge binary digest changed")

    _recover_orphaned_qualification_reservation_v1(
        custody_directory=Path(custody_directory),
        basis_commit=commit,
        rust_formal_replay_binary=rust_binary,
    )

    run_id = secrets.token_bytes(16)
    ledger_id = secrets.token_bytes(16)
    trust_id = secrets.token_bytes(16)
    if len({run_id, ledger_id, trust_id}) != 3:
        _fail(FAIL_PROTOCOL, "qualification opaque identities collided")
    reservation = QualificationCustodyReservationV1(
        custody_directory=Path(custody_directory),
        basis_commit=commit,
        run_id=run_id,
        ledger_id=ledger_id,
    )
    reservation.reserve()
    backend = DockerCeremonyActorsV1(
        basis_commit=commit,
        custody_directory=reservation.custody_directory,
        rust_formal_replay_binary=rust_binary,
        rust_bridge_dag_replay_binary=DEFAULT_RUST_BRIDGE_DAG_BINARY,
        rust_bridge_dag_qualification_report=DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
        timestamp=FROZEN_QUALIFICATION_TIMESTAMP,
    )
    wrapper = PublicSyntheticProtocolActorV1(backend)
    operation_recorder = OperationEvidenceRecorderV1(backend)
    operation_recorder.install()
    actors_started = False
    actors_absent = False
    evidence_bundle: dict[str, object] | None = None
    key_manifests: list[dict[str, object]] | None = None
    destruction_plan: dict[str, object] | None = None
    qualification_statements: list[dict[str, object]] | None = None
    cleanup_actor_rows: list[dict[str, object]] | None = None
    cleanup_daemon_binding: str | None = None
    rust_receipt: dict[str, object] | None = None
    try:
        backend.validate_rust_replay_binding(basis)
        backend.validate_rust_bridge_dag_binding()
        blockers = tuple(backend.unresolved_formal_blockers())
        if blockers:
            _fail(FAIL_PROTOCOL, "Docker backend remains blocked: " + ",".join(blockers))
        backend.bind_transaction_identity(run_id)
        backend.start()
        actors_started = True
        reservation.mark_actors_started()
        _install_finalize_workers_v1(backend)
        static_control_plane, static_daemon_binding = (
            backend.static_replay_control_plane_v1()
        )
        try:
            rust_receipt = run_rust_static_replay_receipt_v1(
                basis,
                control_plane=static_control_plane,
                daemon_receipt_binding=static_daemon_binding,
                rust_binary=rust_binary,
            )
        except Exception as exc:
            _fail(
                FAIL_PROTOCOL,
                f"controlled Rust static replay failed: {type(exc).__name__}",
            )
        # The returned object necessarily contains ephemeral public protocol
        # values.  It remains process-local, is never evaluated/promoted or
        # serialized, and is dropped immediately after call-graph checks.
        ephemeral_inputs = _build_gate_inputs_and_sign_v1(
            basis=basis,
            parent=(parent := generate_parent_absence_audit_v1(REPOSITORY_ROOT)),
            actor_report=MappingProxyType({"qualification_only": True}),
            errata_report=MappingProxyType({"qualification_only": True}),
            python_static_receipt=python_receipt,
            rust_static_receipt=rust_receipt,
            execution_binding_roots=implementation_roots,
            actors=wrapper,
            timestamp=FROZEN_QUALIFICATION_TIMESTAMP,
            run_id=run_id,
            ledger_id=ledger_id,
            trust_id=trust_id,
        )
        replay_parent_absence_audit_v1(parent, repository=REPOSITORY_ROOT)
        _wrapper_summary_v1(wrapper)
        _operation_receipt_summary_v1(backend)
        daemon_binding = backend._docker_daemon_binding
        if type(daemon_binding) is not bytes or len(daemon_binding) != 32:
            _fail(FAIL_PROTOCOL, "Docker daemon identity binding is absent")
        images = backend._profile.get("images")
        if not isinstance(images, Mapping):
            _fail(FAIL_PROTOCOL, "Docker actor image profile is absent")
        image_key = {
            1: "custodian", 2: "python_attester",
            3: "rust_attester", 4: "policy_auditor",
        }
        runtime_rows = [
            {
                "purpose_id": purpose,
                "container_id": backend._containers[purpose],
                "container_inspection": dict(backend._inspect_live_actor(purpose)),
                "volume_initialization_receipt": dict(
                    backend._volume_initialization_receipts[purpose]
                ),
            }
            for purpose in (1, 2, 3, 4)
        ]
        evidence_bundle = {
            "schema": EVIDENCE_SCHEMA,
            "basis_commit": commit,
            "qualification_run_id_hex": run_id.hex(),
            "profile": {
                "profile_sha256": backend._profile_digest.hex(),  # type: ignore[union-attr]
                "images": {
                    str(purpose): images[image_key[purpose]]
                    for purpose in (1, 2, 3, 4)
                },
                "daemon_receipt": dict(backend._docker_daemon_receipt),  # type: ignore[arg-type]
                "daemon_receipt_sha256": daemon_binding.hex(),
                "host_repository_path_sha256": hashlib.sha256(
                    REPOSITORY_ROOT.resolve().as_posix().encode("utf-8")
                ).hexdigest(),
            },
            "actor_runtime_rows": runtime_rows,
            "operation_rows": [dict(row) for row in operation_recorder.rows],
            "purpose4_evidence": {
                "request": wrapper.purpose4_request,
                "response": wrapper.purpose4_response,
            },
            "bridge_evidence_rows": [
                wrapper.bridge_evidence_rows[purpose]
                for purpose in (1, 2, 3)
            ],
            "public_synthetic_fixture": {
                "split_frame_base64": _raw_base64(PUBLIC_SYNTHETIC_SPLIT_FRAME),
                "split_frame_size": len(PUBLIC_SYNTHETIC_SPLIT_FRAME),
                "split_frame_sha256": PUBLIC_SYNTHETIC_SPLIT_FRAME_SHA256,
                "contains_assignments": False,
                "contains_real_seed": False,
            },
            "protocol_call_graph": {
                "keygen_order": list(wrapper.keygen_purposes),
                "purpose1_authorized_object_names": list(wrapper.signed_object_names),
                "purpose4_detached_sign_count": wrapper.parent_sign_count,
                "bridge_replay_order": list(wrapper.bridge_purposes),
                "docker_seed_or_marker_method_call_count": (
                    wrapper.docker_seed_method_call_count
                ),
            },
            "formal_track_claims": {
                "formal_authority": False,
                "authoritative_formal_roots_generated": False,
                "synthetic_formal_shaped_roots_computed_in_memory": True,
                "formal_roots_published": False,
                "gate_evidence_published": False,
                "formal_gates_before": 14,
                "formal_gates_after": 14,
                "m3_state": "NOT_RUN",
                "m3_started": False,
                "real_seed_generated": False,
                "real_seed_accessed": False,
            },
        }
        key_manifests = _qualification_key_manifests_v1(backend)
        destruction_plan = _destruction_plan_v1(backend)
        evidence_content_id = _content_id(
            EVIDENCE_HASH_DOMAIN, evidence_bundle
        )
        qualification_statements = _finalize_qualification_statements_v1(
            backend=backend,
            evidence_content_id=evidence_content_id,
            key_manifests=key_manifests,
            destruction_plan_content_id=str(
                destruction_plan["destruction_plan_content_id"]
            ),
        )
        # Explicitly discard every in-memory key/signature/root-bearing object.
        del ephemeral_inputs
        del parent
    finally:
        try:
            if actors_started:
                backend.close_and_verify_absent()
                actors_absent = (
                    not backend._containers
                    and not backend._state_volumes
                    and backend._docker_control_plane is None
                    and backend._temporary is None
                )
                if not actors_absent:
                    _fail(FAIL_CLEANUP, "Docker actors or key volumes remain")
                if destruction_plan is not None:
                    cleanup_daemon_binding, cleanup_actor_rows = (
                        _live_daemon_and_absence_rows_v1(destruction_plan)
                    )
                    if cleanup_daemon_binding != destruction_plan[
                        "daemon_receipt_sha256"
                    ]:
                        _fail(FAIL_CLEANUP, "post-cleanup Docker daemon changed")
                reservation.mark_actors_verified_absent()
                reservation.cleanup_after_actor_absence()
            else:
                # start() is itself fail-closed and verifies cleanup.  Require its
                # observable postcondition before removing only our three files.
                actors_absent = (
                    not backend._containers
                    and not backend._state_volumes
                    and backend._docker_control_plane is None
                    and backend._temporary is None
                )
                if actors_absent:
                    reservation.cleanup_before_actor_start()
        finally:
            # Any backend or reservation-cleanup exception preserves disk
            # evidence but cannot leave a process-lifetime advisory-lock wedge.
            reservation.release_lock_without_cleanup()

    if (
        not actors_absent
        or evidence_bundle is None
        or key_manifests is None
        or destruction_plan is None
        or qualification_statements is None
        or cleanup_actor_rows is None
        or cleanup_daemon_binding is None
    ):
        _fail(FAIL_PROTOCOL, "live actor qualification did not reach its complete boundary")
    seed_names = {
        "split_seed_instantiation.marker",
        "split_seed_generation.intent",
        "split_seed_generation.complete",
        "split_master_seed.bin",
    }
    if any((reservation.custody_directory / name).exists() for name in seed_names):
        _fail(FAIL_CLEANUP, "a forbidden seed or marker artifact exists")
    if any(reservation.custody_directory.iterdir()):
        _fail(FAIL_CLEANUP, "qualification custody is not empty")

    cleanup_receipt = _post_cleanup_absence_receipt_v1(
        plan=destruction_plan,
        daemon_binding=cleanup_daemon_binding,
        actor_rows=cleanup_actor_rows,
    )

    m3_receipt = basis.implementation_inputs["m3_implementation_qualification_receipt"]
    report_body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": STATUS,
        "claim_level": CLAIM_LEVEL,
        "basis_commit": commit,
        "commit_a_source_set_sha256": source_set_sha256,
        "commit_a_source_file_count": source_file_count,
        "authority_boundary": dict(AUTHORITY_BOUNDARY),
        "independence_disclosure": dict(INDEPENDENCE_DISCLOSURE),
        "implementation_bindings": {
            "formal_rust_replay_binary_sha256": _hash_regular_file(rust_binary),
            "bridge_rust_replay_binary_sha256": bridge_digest,
            "bridge_rust_qualification_report_sha256": str(
                bridge_report["diagnostic_report_sha256"]
            ),
            "m3_implementation_qualification_receipt_sha256": _sha256(
                _canonical_json(_json_transport(m3_receipt))
            ),
            "m3_implementation_qualification_receipt": _json_transport(
                m3_receipt
            ),
        },
        "qualification_key_manifests": key_manifests,
        "evidence_bundle": evidence_bundle,
        "destruction_plan": destruction_plan,
        "cleanup_absence_receipt": cleanup_receipt,
        "qualification_statements": qualification_statements,
    }
    authority_preimage = {
        "basis_commit": commit,
        "evidence_content_id": _content_id(EVIDENCE_HASH_DOMAIN, evidence_bundle),
        "destruction_plan_content_id": destruction_plan[
            "destruction_plan_content_id"
        ],
        "cleanup_receipt_content_id": cleanup_receipt[
            "cleanup_receipt_content_id"
        ],
        "qualification_key_manifest_content_ids": [
            row["manifest_content_id"] for row in key_manifests
        ],
        "qualification_statement_sha256": [
            _sha256(_canonical_json(row)) for row in qualification_statements
        ],
    }
    report_body["bundle_content_id"] = _content_id(
        BUNDLE_AUTHORITY_HASH_DOMAIN, authority_preimage
    )
    report_body["diagnostic_report_sha256"] = _report_hash(report_body)
    replayed = validate_actor_protocol_qualification_report_v1(report_body)
    canonical_bundle_bytes = _canonical_json(dict(replayed.report))
    if _strict_json_object(canonical_bundle_bytes) != dict(replayed.report):
        _fail(FAIL_PROTOCOL, "validated live archive canonicalization changed value")
    daemon_receipt_binding = bytes.fromhex(cleanup_daemon_binding)
    live_run_nonce = secrets.token_bytes(16)
    issuer_pid = os.getpid()
    preimage = _admission_mac_preimage_v1(
        basis_commit=replayed.basis_commit,
        bundle_content_id=replayed.bundle_content_id,
        qualification_key_ids=replayed.qualification_key_ids,
        daemon_receipt_binding=daemon_receipt_binding,
        canonical_bundle_bytes=canonical_bundle_bytes,
        live_run_nonce=live_run_nonce,
        issuer_pid=issuer_pid,
    )
    # This is intentionally in the live qualifier's post-cleanup terminal
    # branch.  No archive/replay callable accepts a replay object and mints a
    # capability; loading diagnostic evidence remains entropy-free.
    with _ADMISSION_ISSUE_LOCK:
        if _PROCESS_ADMISSION_SECRET is None:
            _PROCESS_ADMISSION_SECRET = secrets.token_bytes(32)
        secret = _PROCESS_ADMISSION_SECRET
    return LiveActorProtocolAdmissionV1(
        basis_commit=replayed.basis_commit,
        bundle_content_id=replayed.bundle_content_id,
        qualification_key_ids=replayed.qualification_key_ids,
        daemon_receipt_binding=daemon_receipt_binding,
        canonical_bundle_bytes=canonical_bundle_bytes,
        live_run_nonce=live_run_nonce,
        issuer_pid=issuer_pid,
        token_mac=hmac.digest(secret, preimage, "sha256"),
        _seal=_LIVE_ADMISSION_CONSTRUCTOR_SEAL,
    )


def _json_transport(value: object) -> object:
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _json_transport(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_transport(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    _fail(FAIL_REPORT, f"unsupported diagnostic transport type {type(value).__name__}")


__all__ = [
    "ARTIFACT_KIND",
    "ActorProtocolQualificationError",
    "AUTHORITY_BOUNDARY",
    "DEFAULT_REPORT_PATH",
    "INDEPENDENCE_DISCLOSURE",
    "ConsumedLiveActorProtocolAdmissionV1",
    "LiveActorProtocolAdmissionV1",
    "PUBLIC_SYNTHETIC_SPLIT_FRAME_SHA256",
    "PublicSyntheticProtocolActorV1",
    "QualificationCustodyReservationV1",
    "ReplayedActorProtocolQualificationV1",
    "canonical_actor_protocol_qualification_report_bytes_v1",
    "consume_live_actor_protocol_admission_v1",
    "load_actor_protocol_qualification_report_v1",
    "qualify_live_actor_protocol_v1",
    "validate_actor_protocol_qualification_report_v1",
]

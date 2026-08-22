"""Mechanics-only CAS store for prospective Phase-2B Step-5 retention.

The module supplies a strict Unix ``SOCK_SEQPACKET``/``SCM_RIGHTS`` transport,
content-addressed durable storage, Ed25519-signed store receipts, and an
append-only checkpoint chain.  It deliberately does not start a listener or a
service.  Importing or using it in the producer process, under the same UID, or
on a merely local filesystem is not authority separation.  A qualifying
deployment still needs a separately controlled UID/control domain, pinned
startup/runtime closure, protected key custody, and independently durable
storage.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import fcntl
import hashlib
import os
import re
import socket
import stat
import struct
import threading
from typing import Final, Mapping

from . import phase2b_formal_wire_covert_step5_authority_v1 as authority
from .rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # The module remains importable but every signing operation fails closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - depends on the deployment image.
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = "hegel-machine-phase2b-formal-wire-covert-step5-store/1"
STORE_REQUEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-scm-request/1"
)
STORE_RECEIPT_BODY_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-receipt-body/1"
)
SIGNED_STORE_RECEIPT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-cas-store-receipt/1"
)
STORE_CHECKPOINT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-checkpoint/1"
)
STORE_HEAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-head/1"
)
STORE_RESPONSE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-response/1"
)

CAS_STORE_PURPOSE: Final = "CAS_STORE"
CAS_STORE_RECEIPT_PURPOSE: Final = "SIGN_CAS_IF_ABSENT_RECEIPT"
CAS_STORE_CHECKPOINT_PURPOSE: Final = "CAS_STORE_APPEND_ONLY_CHECKPOINT"
CAS_STORE_HEAD_PURPOSE: Final = "CAS_STORE_ATOMIC_CHECKPOINT_HEAD"
CAS_STORE_RESPONSE_PURPOSE: Final = "CAS_STORE_RETENTION_RESPONSE"
CAS_STORE_SIGNATURE_DOMAIN: Final = (
    "HEGEL_MACHINE_PHASE2B_FORMAL_WIRE_COVERT_STEP5_CAS_STORE_RECEIPT_V1"
)
_CAS_STORE_SIGNATURE_PREFIX: Final = (
    CAS_STORE_SIGNATURE_DOMAIN.encode("ascii") + b"\x00"
)

AUTHORITY_CLAIM_BOUNDARY: Final = (
    "CAS_STORE_MECHANICS_ONLY_SAME_PROCESS_OR_SAME_UID_OR_LOCAL_FILESYSTEM_"
    "IS_NOT_AUTHORITY_ACTUAL_AUTHORITY_REQUIRES_INDEPENDENT_UID_CONTROL_DOMAIN_"
    "KEY_CUSTODY_DURABILITY_AND_LIVE_RUNTIME_CLOSURE"
)

MAXIMUM_REQUEST_BYTES: Final = 64 * 1_024
MAXIMUM_RESPONSE_BYTES: Final = 512 * 1_024
MAXIMUM_CHECKPOINT_BYTES: Final = 512 * 1_024
MAXIMUM_HEAD_BYTES: Final = 16 * 1_024
MAXIMUM_RECEIPT_BYTES: Final = 256 * 1_024
GENESIS_CHECKPOINT_SHA256: Final = "sha256:" + "0" * 64
_MAXIMUM_SAFE_JSON_INTEGER: Final = (1 << 53) - 1

# Declarative producer-contract pins.  The server never imports these producer
# modules; an independently pinned live closure must compare these bytes before
# deployment.  The literal request/ACK contract below is the only server input.
AUDITOR_SOURCE_SHA256_PIN: Final = (
    "5d77403dc41c661cb8a8d7deb90c9b2644de7a27fda9369c37f76eb82474a9be"
)
QV2_SOURCE_SHA256_PIN: Final = (
    "921bbb771d27c339acaf2c5e1e48f35a899c56552929065c02894e590d32e2ed"
)
RUNTIME_SOURCE_SHA256_PIN: Final = (
    "35315054bf360a6c7148e79b018f7657a16b19d5f833563693461830422b4979"
)
AUTHORITY_SOURCE_SHA256_PIN: Final = (
    "382679c51352a015fb77b806a5a066d0fc6c998f13ccb30d9bf8385c83b56762"
)
RETENTION_ACK_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-retention-ack/1"
)
EXPECTED_RETAINED_OBJECTS: Final = 12
EXPECTED_CLEANUP_CALLS: Final = 4
MODE_ORDER: Final = ("KNOWN_NULL", "KNOWN_BINDING_LEAK")
INVARIANT_BASENAME: Final = "invariant-result.json"
STATISTICS_BASENAMES: Final = tuple(f"target-{index}.json" for index in range(5))
EXPECTED_OBJECT_ORDER: Final = tuple(
    (mode, phase, target_index)
    for mode in MODE_ORDER
    for phase, target_index in (
        (("INVARIANCE", None),)
        + tuple(("STATISTICS", index) for index in range(5))
    )
)
INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND: Final = 1_048_576
MAXIMUM_STATISTICS_ARTIFACT_BYTES: Final = 17_825_792
KEY_EPOCH: Final = 1

_IDENTIFIER_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{7,511}")
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_HEX_32_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_HEX_64_PATTERN: Final = re.compile(r"[0-9a-f]{128}")
_CHECKPOINT_NAME_PATTERN: Final = re.compile(r"[0-9]{20}\.json")
_OBJECT_NAME_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_AUTHORITY_PROFILE_ID_PATTERN: Final = re.compile(
    r"phase2b_step5_authority_profile_v1_[0-9a-f]{64}"
)
_SIGNER_REGISTRY_ID_PATTERN: Final = re.compile(
    r"phase2b_step5_signer_registry_v1_[0-9a-f]{64}"
)
_ROLE_KEY_ID_PATTERN: Final = re.compile(
    r"phase2b_step5_role_key_v1_[0-9a-f]{64}"
)

_REQUEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "attempt_context_id",
        "retention_session_id",
        "request_id",
        "ordinal",
        "mode",
        "phase",
        "target_index_or_null",
        "source_basename",
        "auditor_reported_sha256",
        "auditor_reported_size_bytes",
        "fd_count",
    }
)
_RECEIPT_BODY_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "signature_domain",
        "chain_id",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "signer_key_id",
        "attempt_context_id",
        "retention_session_id",
        "request_id",
        "ordinal",
        "mode",
        "phase",
        "target_index_or_null",
        "source_basename",
        "peer_pid",
        "peer_uid",
        "peer_gid",
        "content_address",
        "sha256",
        "size_bytes",
        "object_disposition",
        "checkpoint_index",
        "previous_checkpoint_sha256",
        "durable_before_return",
        "readback_verified",
        "claim_boundary",
    }
)
_SIGNED_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "signature_domain",
        "signed_body",
        "signed_body_sha256",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "signer_public_key_hex",
        "signer_key_id",
        "ed25519_signature_hex",
    }
)
_CHECKPOINT_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "chain_id",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "checkpoint_index",
        "previous_checkpoint_sha256",
        "request_id",
        "content_address",
        "size_bytes",
        "signed_receipt_sha256",
        "signed_receipt",
    }
)
_HEAD_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "chain_id",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "checkpoint_count",
        "last_checkpoint_sha256",
    }
)
_RESPONSE_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "request_id",
        "checkpoint_sha256",
        "checkpoint_record",
        "acknowledgement",
        "claim_boundary",
    }
)
_ACK_KEYS: Final = frozenset(
    {
        "schema_version",
        "attempt_context_id",
        "retention_session_id",
        "request_id",
        "ordinal",
        "mode",
        "phase",
        "source_basename",
        "content_address",
        "sha256",
        "size_bytes",
        "durable_before_return",
        "readback_verified",
        "store_receipt_ref",
    }
)


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 payload must be exact bytes")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is not one bounded exact identifier")
    return value


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is not one canonical SHA-256 address")
    return value


def _require_wire_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
) -> int:
    if type(value) is int:
        result = value
    elif type(value) is float and value.is_integer():
        result = int(value)
    else:
        raise TypeError(f"{label} is not one exact JSON integer")
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside its frozen bound")
    return result


def _require_exact_mapping(
    value: object,
    keys: frozenset[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} field set drift")
    return value


def _require_no_follow_support() -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("CAS store requires O_NOFOLLOW")
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("CAS store requires Linux SO_PEERCRED")


def _require_frozen_contract_v1() -> None:
    if (
        STORE_REQUEST_SCHEMA_VERSION
        != "hegel-machine-phase2b-formal-wire-covert-step5-scm-request/1"
        or RETENTION_ACK_SCHEMA_VERSION
        != "hegel-machine-phase2b-formal-wire-covert-step5-retention-ack/1"
        or EXPECTED_RETAINED_OBJECTS != 12
        or EXPECTED_CLEANUP_CALLS != 4
        or len(EXPECTED_OBJECT_ORDER) != 12
        or MODE_ORDER != ("KNOWN_NULL", "KNOWN_BINDING_LEAK")
        or INVARIANT_BASENAME != "invariant-result.json"
        or STATISTICS_BASENAMES
        != tuple(f"target-{index}.json" for index in range(5))
        or INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND != 1_048_576
        or MAXIMUM_STATISTICS_ARTIFACT_BYTES != 17_825_792
        or any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in (
                AUDITOR_SOURCE_SHA256_PIN,
                QV2_SOURCE_SHA256_PIN,
                RUNTIME_SOURCE_SHA256_PIN,
                AUTHORITY_SOURCE_SHA256_PIN,
            )
        )
    ):
        raise RuntimeError("CAS store frozen producer/authority contract drift")


def _require_authority_profile_id(value: object) -> str:
    if type(value) is not str or _AUTHORITY_PROFILE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("authority profile ID syntax drift")
    return value


def _require_signer_registry_id(value: object) -> str:
    if type(value) is not str or _SIGNER_REGISTRY_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("signer registry ID syntax drift")
    return value


def _require_role_key_id(value: object) -> str:
    if type(value) is not str or _ROLE_KEY_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("CAS_STORE role key ID syntax drift")
    return value


@dataclass(frozen=True, slots=True)
class _CasStoreAuthorityBindingV1:
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    signer_key_id: str
    public_key_bytes: bytes


def _cas_store_role_key_id_for_public_key_v1(public_key: bytes) -> str:
    if type(public_key) is not bytes or len(public_key) != 32:
        raise ValueError("CAS_STORE registry public key must be raw 32 bytes")
    row = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": "CAS_STORE",
        "control_domain": authority.ROLE_CONTROL_DOMAINS_V1["CAS_STORE"],
        "authorized_signature_purposes": list(
            authority.ROLE_SIGNATURE_PURPOSES_V1["CAS_STORE"]
        ),
        "algorithm": "Ed25519",
        "key_epoch": KEY_EPOCH,
        "public_key_hex": public_key.hex(),
    }
    return authority.phase2b_step5_role_key_id_v1(row)


def _bind_verified_cas_store_authority_v1(
    trust_lock: authority.Phase2BStep5VerifiedArtifactV1,
    authority_profile: authority.Phase2BStep5VerifiedArtifactV1,
    signer_registry: authority.Phase2BStep5VerifiedArtifactV1,
) -> _CasStoreAuthorityBindingV1:
    lock, profile, registry = authority._require_locked_profile_registry_graph_v1(
        trust_lock,
        authority_profile,
        signer_registry,
    )
    profile_id = _require_authority_profile_id(profile.artifact_id)
    registry_id = _require_signer_registry_id(registry.artifact_id)
    role_policy = profile.body.get("role_policy")
    if not isinstance(role_policy, Mapping) or (
        tuple(role_policy.get("roles_in_exact_order", ())) != authority.ROLE_ORDER_V1
        or role_policy.get("role_count") != 8
        or role_policy.get("distinct_public_keys") is not True
        or role_policy.get("distinct_key_ids") is not True
        or role_policy.get("distinct_control_domains") is not True
        or role_policy.get("key_epoch") != KEY_EPOCH
    ):
        raise ValueError("authority profile exact role policy drift")
    if (
        lock.body.get("authority_profile_id") != profile_id
        or lock.body.get("signer_registry_id") != registry_id
        or registry.body.get("authority_profile_id") != profile_id
    ):
        raise ValueError("CAS_STORE authority graph lineage drift")
    rows = registry.body.get("role_keys_in_exact_order")
    if type(rows) is not tuple or len(rows) != len(authority.ROLE_ORDER_V1):
        raise ValueError("signer registry role-key cardinality drift")
    cas_index = authority.ROLE_ORDER_V1.index("CAS_STORE")
    raw_row = rows[cas_index]
    if not isinstance(raw_row, Mapping):
        raise ValueError("CAS_STORE registry row type drift")
    row = {
        "schema_version": raw_row.get("schema_version"),
        "role": raw_row.get("role"),
        "control_domain": raw_row.get("control_domain"),
        "authorized_signature_purposes": list(
            raw_row.get("authorized_signature_purposes", ())
        ),
        "algorithm": raw_row.get("algorithm"),
        "key_epoch": raw_row.get("key_epoch"),
        "public_key_hex": raw_row.get("public_key_hex"),
        "key_id": raw_row.get("key_id"),
    }
    if (
        row["schema_version"] != "hegel-machine-phase2b-step5-role-key/1"
        or row["role"] != "CAS_STORE"
        or row["control_domain"]
        != authority.ROLE_CONTROL_DOMAINS_V1["CAS_STORE"]
        or row["authorized_signature_purposes"]
        != list(authority.ROLE_SIGNATURE_PURPOSES_V1["CAS_STORE"])
        or row["algorithm"] != "Ed25519"
        or row["key_epoch"] != KEY_EPOCH
        or type(row["public_key_hex"]) is not str
        or _HEX_32_PATTERN.fullmatch(row["public_key_hex"]) is None
    ):
        raise ValueError("CAS_STORE role policy/public key drift")
    public_key_bytes = bytes.fromhex(row["public_key_hex"])
    expected_key_id = _cas_store_role_key_id_for_public_key_v1(public_key_bytes)
    key_id = _require_role_key_id(row["key_id"])
    if key_id != expected_key_id:
        raise ValueError("CAS_STORE authority role key content ID drift")
    return _CasStoreAuthorityBindingV1(
        authority_profile_id=profile_id,
        signer_registry_id=registry_id,
        key_epoch=KEY_EPOCH,
        signer_key_id=key_id,
        public_key_bytes=public_key_bytes,
    )


def _load_and_bind_cas_store_authority_v1(
    *,
    stage_b_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    out_of_band_expected_lock_id: str,
    out_of_band_expected_lock_payload_sha256: str,
) -> _CasStoreAuthorityBindingV1:
    """Load the complete authority graph from exact bytes inside CAS_STORE.

    ``Phase2BStep5VerifiedArtifactV1`` instances are reusable pure-verifier
    results, not transferable capabilities.  In particular, accepting such an
    instance from bootstrap code would let a caller reach the authority
    module's private test/construction helpers and bypass the bytes-only
    Stage-B trust-anchor loaders.  The service boundary therefore accepts only
    exact artifact bytes plus the independently supplied Stage-B pins and
    constructs every verified token locally.
    """

    if (
        type(stage_b_lock_payload) is not bytes
        or type(authority_profile_payload) is not bytes
        or type(signer_registry_payload) is not bytes
    ):
        raise TypeError("CAS_STORE authority artifacts must be exact bytes")
    lock = authority.load_step5_stage_b_trust_lock_v1(
        stage_b_lock_payload,
        out_of_band_expected_lock_id=out_of_band_expected_lock_id,
        out_of_band_expected_payload_sha256=(
            out_of_band_expected_lock_payload_sha256
        ),
    )
    profile = authority.load_step5_authority_profile_v1(
        authority_profile_payload,
        lock,
    )
    registry = authority.load_step5_signer_registry_v1(
        signer_registry_payload,
        lock,
        profile,
    )
    return _bind_verified_cas_store_authority_v1(lock, profile, registry)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("durable write made no progress")
        offset += written


def _read_regular_at(
    directory_fd: int,
    basename: str,
    maximum_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(basename, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise ValueError("persisted file is not one bounded regular link")
        blocks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(65_536, before.st_size - offset), offset)
            if not block:
                raise ValueError("persisted file ended before its recorded size")
            blocks.append(block)
            offset += len(block)
        if os.pread(fd, 1, offset):
            raise ValueError("persisted file grew during bounded readback")
        after = os.fstat(fd)
        if not _same_file_identity(before, after):
            raise ValueError("persisted file changed during bounded readback")
        return b"".join(blocks)
    finally:
        os.close(fd)


def _atomic_replace_file_at(
    directory_fd: int,
    basename: str,
    payload: bytes,
    *,
    maximum_bytes: int,
    nonce: int,
) -> None:
    if not 0 < len(payload) <= maximum_bytes:
        raise ValueError("atomic state payload is outside its frozen byte bound")
    temporary = f".{basename}.pending-{os.getpid()}-{nonce}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(
            temporary,
            basename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        if _read_regular_at(directory_fd, basename, maximum_bytes) != payload:
            raise ValueError("atomic state readback differs from committed bytes")
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _checkpoint_name(index: int) -> str:
    if type(index) is not int or not 0 <= index <= _MAXIMUM_SAFE_JSON_INTEGER:
        raise ValueError("checkpoint index is outside its frozen bound")
    name = f"{index:020d}.json"
    if _CHECKPOINT_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("checkpoint filename overflow")
    return name


@dataclass(frozen=True, slots=True)
class PeerCredentialsV1:
    pid: int
    uid: int
    gid: int

    def __post_init__(self) -> None:
        if type(self) is not PeerCredentialsV1:
            raise TypeError("peer credentials must use the exact DTO")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("peer PID must be one positive exact integer")
        if type(self.uid) is not int or self.uid < 0:
            raise ValueError("peer UID must be one nonnegative exact integer")
        if type(self.gid) is not int or self.gid < 0:
            raise ValueError("peer GID must be one nonnegative exact integer")


@dataclass(frozen=True, slots=True)
class StoreRequestV1:
    schema_version: str
    attempt_context_id: str
    retention_session_id: str
    request_id: str
    ordinal: int
    mode: str
    phase: str
    target_index_or_null: int | None
    source_basename: str
    auditor_reported_sha256: str
    auditor_reported_size_bytes: int
    fd_count: int

    def __post_init__(self) -> None:
        if type(self) is not StoreRequestV1:
            raise TypeError("store request must use the exact DTO")
        _require_frozen_contract_v1()
        if self.schema_version != STORE_REQUEST_SCHEMA_VERSION:
            raise ValueError("store request schema is not runtime request/1")
        _require_identifier(self.attempt_context_id, "request attempt context ID")
        _require_identifier(self.retention_session_id, "request retention session ID")
        _require_sha256(self.request_id, "request ID")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < (
            EXPECTED_RETAINED_OBJECTS
        ):
            raise ValueError("request ordinal drift")
        expected_mode, expected_phase, expected_target = EXPECTED_OBJECT_ORDER[
            self.ordinal
        ]
        if (
            self.mode != expected_mode
            or self.phase != expected_phase
            or self.target_index_or_null != expected_target
            or (
                expected_target is not None
                and type(self.target_index_or_null) is not int
            )
        ):
            raise ValueError("request ordinal/mode/phase/target binding drift")
        expected_basename = (
            INVARIANT_BASENAME
            if expected_target is None
            else STATISTICS_BASENAMES[expected_target]
        )
        if self.source_basename != expected_basename:
            raise ValueError("request source basename binding drift")
        _require_sha256(
            self.auditor_reported_sha256,
            "request auditor-reported SHA-256",
        )
        maximum = (
            INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
            if self.phase == "INVARIANCE"
            else MAXIMUM_STATISTICS_ARTIFACT_BYTES
        )
        if (
            type(self.auditor_reported_size_bytes) is not int
            or not 0 < self.auditor_reported_size_bytes <= maximum
        ):
            raise ValueError("request auditor-reported size drift")
        if self.fd_count != 1 or type(self.fd_count) is not int:
            raise ValueError("store request must declare exactly one descriptor")
        if self.request_id != _sha256(
            canonicalize_rfc8785_v1(self._mapping_without_request_id())
        ):
            raise ValueError("store request ID does not bind its exact runtime fields")

    def _mapping_without_request_id(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "ordinal": self.ordinal,
            "mode": self.mode,
            "phase": self.phase,
            "target_index_or_null": self.target_index_or_null,
            "source_basename": self.source_basename,
            "auditor_reported_sha256": self.auditor_reported_sha256,
            "auditor_reported_size_bytes": self.auditor_reported_size_bytes,
            "fd_count": self.fd_count,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._mapping_without_request_id(), "request_id": self.request_id}

    def canonical_bytes(self) -> bytes:
        return canonicalize_rfc8785_v1(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: object) -> "StoreRequestV1":
        body = _require_exact_mapping(value, _REQUEST_KEYS, "store request")
        target_value = body["target_index_or_null"]
        target = (
            None
            if target_value is None
            else _require_wire_integer(
                target_value,
                "request target index",
                maximum=len(STATISTICS_BASENAMES) - 1,
            )
        )
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
            ordinal=_require_wire_integer(
                body["ordinal"],
                "request ordinal",
                maximum=EXPECTED_RETAINED_OBJECTS - 1,
            ),
            mode=body["mode"],  # type: ignore[arg-type]
            phase=body["phase"],  # type: ignore[arg-type]
            target_index_or_null=target,
            source_basename=body["source_basename"],  # type: ignore[arg-type]
            auditor_reported_sha256=body["auditor_reported_sha256"],  # type: ignore[arg-type]
            auditor_reported_size_bytes=_require_wire_integer(
                body["auditor_reported_size_bytes"],
                "request auditor-reported size",
                minimum=1,
                maximum=MAXIMUM_STATISTICS_ARTIFACT_BYTES,
            ),
            fd_count=_require_wire_integer(
                body["fd_count"], "request FD count", minimum=1, maximum=1
            ),
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "StoreRequestV1":
        if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_REQUEST_BYTES:
            raise ValueError("store request wire bytes are outside the frozen bound")
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
        request = cls.from_mapping(decoded)
        if request.canonical_bytes() != payload:
            raise ValueError("store request DTO/canonical byte binding drift")
        return request


@dataclass(frozen=True, slots=True)
class StoreReceiptBodyV1:
    schema_version: str
    purpose: str
    signature_domain: str
    chain_id: str
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    signer_key_id: str
    attempt_context_id: str
    retention_session_id: str
    request_id: str
    ordinal: int
    mode: str
    phase: str
    target_index_or_null: int | None
    source_basename: str
    peer_pid: int
    peer_uid: int
    peer_gid: int
    content_address: str
    sha256: str
    size_bytes: int
    object_disposition: str
    checkpoint_index: int
    previous_checkpoint_sha256: str
    durable_before_return: bool
    readback_verified: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if type(self) is not StoreReceiptBodyV1:
            raise TypeError("store receipt body must use the exact DTO")
        if (
            self.schema_version != STORE_RECEIPT_BODY_SCHEMA_VERSION
            or self.purpose != CAS_STORE_RECEIPT_PURPOSE
            or self.signature_domain != CAS_STORE_SIGNATURE_DOMAIN
            or self.claim_boundary != AUTHORITY_CLAIM_BOUNDARY
        ):
            raise ValueError("store receipt schema/purpose/domain/claim drift")
        _require_identifier(self.chain_id, "store chain ID")
        _require_authority_profile_id(self.authority_profile_id)
        _require_signer_registry_id(self.signer_registry_id)
        if type(self.key_epoch) is not int or self.key_epoch != KEY_EPOCH:
            raise ValueError("store receipt key epoch drift")
        _require_role_key_id(self.signer_key_id)
        _require_identifier(self.attempt_context_id, "receipt attempt context ID")
        _require_identifier(self.retention_session_id, "receipt retention session ID")
        _require_sha256(self.request_id, "receipt request ID")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < (
            EXPECTED_RETAINED_OBJECTS
        ):
            raise ValueError("receipt ordinal drift")
        expected_mode, expected_phase, expected_target = EXPECTED_OBJECT_ORDER[
            self.ordinal
        ]
        if (
            self.mode != expected_mode
            or self.phase != expected_phase
            or self.target_index_or_null != expected_target
            or (
                expected_target is not None
                and type(self.target_index_or_null) is not int
            )
        ):
            raise ValueError("receipt ordinal/mode/phase/target binding drift")
        expected_basename = (
            INVARIANT_BASENAME
            if expected_target is None
            else STATISTICS_BASENAMES[expected_target]
        )
        if self.source_basename != expected_basename:
            raise ValueError("receipt basename binding drift")
        PeerCredentialsV1(self.peer_pid, self.peer_uid, self.peer_gid)
        _require_sha256(self.content_address, "receipt content address")
        _require_sha256(self.sha256, "receipt SHA-256")
        if self.content_address != self.sha256:
            raise ValueError("receipt content address/SHA-256 drift")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("receipt object size drift")
        if self.object_disposition not in {"CREATED", "PRESENT_VERIFIED"}:
            raise ValueError("receipt object disposition drift")
        if (
            type(self.checkpoint_index) is not int
            or not 0 <= self.checkpoint_index <= _MAXIMUM_SAFE_JSON_INTEGER
            or self.checkpoint_index != self.ordinal
        ):
            raise ValueError("receipt checkpoint index drift")
        _require_sha256(
            self.previous_checkpoint_sha256,
            "receipt previous checkpoint",
        )
        if self.durable_before_return is not True or self.readback_verified is not True:
            raise ValueError("receipt does not assert durable verified readback")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "signature_domain": self.signature_domain,
            "chain_id": self.chain_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "key_epoch": self.key_epoch,
            "signer_key_id": self.signer_key_id,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "request_id": self.request_id,
            "ordinal": self.ordinal,
            "mode": self.mode,
            "phase": self.phase,
            "target_index_or_null": self.target_index_or_null,
            "source_basename": self.source_basename,
            "peer_pid": self.peer_pid,
            "peer_uid": self.peer_uid,
            "peer_gid": self.peer_gid,
            "content_address": self.content_address,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "object_disposition": self.object_disposition,
            "checkpoint_index": self.checkpoint_index,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "durable_before_return": self.durable_before_return,
            "readback_verified": self.readback_verified,
            "claim_boundary": self.claim_boundary,
        }

    def canonical_bytes(self) -> bytes:
        return canonicalize_rfc8785_v1(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: object) -> "StoreReceiptBodyV1":
        body = _require_exact_mapping(value, _RECEIPT_BODY_KEYS, "receipt body")
        target_value = body["target_index_or_null"]
        target = (
            None
            if target_value is None
            else _require_wire_integer(
                target_value,
                "receipt target index",
                maximum=len(STATISTICS_BASENAMES) - 1,
            )
        )
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            signature_domain=body["signature_domain"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            key_epoch=_require_wire_integer(
                body["key_epoch"], "receipt key epoch", minimum=1, maximum=1
            ),
            signer_key_id=body["signer_key_id"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
            ordinal=_require_wire_integer(
                body["ordinal"],
                "receipt ordinal",
                maximum=EXPECTED_RETAINED_OBJECTS - 1,
            ),
            mode=body["mode"],  # type: ignore[arg-type]
            phase=body["phase"],  # type: ignore[arg-type]
            target_index_or_null=target,
            source_basename=body["source_basename"],  # type: ignore[arg-type]
            peer_pid=_require_wire_integer(
                body["peer_pid"], "receipt peer PID", minimum=1
            ),
            peer_uid=_require_wire_integer(body["peer_uid"], "receipt peer UID"),
            peer_gid=_require_wire_integer(body["peer_gid"], "receipt peer GID"),
            content_address=body["content_address"],  # type: ignore[arg-type]
            sha256=body["sha256"],  # type: ignore[arg-type]
            size_bytes=_require_wire_integer(
                body["size_bytes"], "receipt size", minimum=1
            ),
            object_disposition=body["object_disposition"],  # type: ignore[arg-type]
            checkpoint_index=_require_wire_integer(
                body["checkpoint_index"], "receipt checkpoint index"
            ),
            previous_checkpoint_sha256=body["previous_checkpoint_sha256"],  # type: ignore[arg-type]
            durable_before_return=body["durable_before_return"],  # type: ignore[arg-type]
            readback_verified=body["readback_verified"],  # type: ignore[arg-type]
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SignedStoreReceiptV1:
    schema_version: str
    purpose: str
    signature_domain: str
    signed_body: StoreReceiptBodyV1
    signed_body_sha256: str
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    signer_public_key_hex: str
    signer_key_id: str
    ed25519_signature_hex: str

    def __post_init__(self) -> None:
        if type(self) is not SignedStoreReceiptV1:
            raise TypeError("signed store receipt must use the exact DTO")
        if (
            self.schema_version != SIGNED_STORE_RECEIPT_SCHEMA_VERSION
            or self.purpose != CAS_STORE_RECEIPT_PURPOSE
            or self.signature_domain != CAS_STORE_SIGNATURE_DOMAIN
        ):
            raise ValueError("signed receipt schema/purpose/domain drift")
        self.signed_body.__post_init__()
        if self.signed_body_sha256 != _sha256(self.signed_body.canonical_bytes()):
            raise ValueError("signed receipt body hash drift")
        _require_authority_profile_id(self.authority_profile_id)
        _require_signer_registry_id(self.signer_registry_id)
        if type(self.key_epoch) is not int or self.key_epoch != KEY_EPOCH:
            raise ValueError("signed receipt key epoch drift")
        if (
            type(self.signer_public_key_hex) is not str
            or _HEX_32_PATTERN.fullmatch(self.signer_public_key_hex) is None
        ):
            raise ValueError("signed receipt public key wire encoding drift")
        if (
            self.signed_body.authority_profile_id != self.authority_profile_id
            or self.signed_body.signer_registry_id != self.signer_registry_id
            or self.signed_body.key_epoch != self.key_epoch
            or self.signed_body.signer_key_id != self.signer_key_id
        ):
            raise ValueError("signed receipt authority/signer binding drift")
        _require_role_key_id(self.signer_key_id)
        if (
            type(self.ed25519_signature_hex) is not str
            or _HEX_64_PATTERN.fullmatch(self.ed25519_signature_hex) is None
        ):
            raise ValueError("signed receipt Ed25519 signature encoding drift")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "signature_domain": self.signature_domain,
            "signed_body": self.signed_body.to_mapping(),
            "signed_body_sha256": self.signed_body_sha256,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "key_epoch": self.key_epoch,
            "signer_public_key_hex": self.signer_public_key_hex,
            "signer_key_id": self.signer_key_id,
            "ed25519_signature_hex": self.ed25519_signature_hex,
        }

    def canonical_bytes(self) -> bytes:
        payload = canonicalize_rfc8785_v1(self.to_mapping())
        if len(payload) > MAXIMUM_RECEIPT_BYTES:
            raise ValueError("signed receipt exceeds its frozen wire bound")
        return payload

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_mapping(cls, value: object) -> "SignedStoreReceiptV1":
        body = _require_exact_mapping(value, _SIGNED_RECEIPT_KEYS, "signed receipt")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            signature_domain=body["signature_domain"],  # type: ignore[arg-type]
            signed_body=StoreReceiptBodyV1.from_mapping(body["signed_body"]),
            signed_body_sha256=body["signed_body_sha256"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            key_epoch=_require_wire_integer(
                body["key_epoch"], "signed receipt key epoch", minimum=1, maximum=1
            ),
            signer_public_key_hex=body["signer_public_key_hex"],  # type: ignore[arg-type]
            signer_key_id=body["signer_key_id"],  # type: ignore[arg-type]
            ed25519_signature_hex=body["ed25519_signature_hex"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "SignedStoreReceiptV1":
        if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_RECEIPT_BYTES:
            raise ValueError("signed receipt wire bytes are outside the frozen bound")
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
        receipt = cls.from_mapping(decoded)
        if receipt.canonical_bytes() != payload:
            raise ValueError("signed receipt DTO/canonical byte binding drift")
        return receipt


def _signature_preimage(body: StoreReceiptBodyV1) -> bytes:
    body.__post_init__()
    return _CAS_STORE_SIGNATURE_PREFIX + body.canonical_bytes()


def verify_signed_store_receipt_v1(
    receipt: SignedStoreReceiptV1,
    *,
    expected_public_key: bytes,
    expected_signer_key_id: str,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
) -> StoreReceiptBodyV1:
    """Verify with the fixed Ed25519 backend; no callback verifier is accepted."""

    if type(receipt) is not SignedStoreReceiptV1:
        raise TypeError("receipt verification requires the exact signed receipt DTO")
    receipt.__post_init__()
    if type(expected_public_key) is not bytes or len(expected_public_key) != 32:
        raise ValueError("expected Ed25519 public key must be exact raw 32 bytes")
    _require_role_key_id(expected_signer_key_id)
    _require_authority_profile_id(expected_authority_profile_id)
    _require_signer_registry_id(expected_signer_registry_id)
    if expected_signer_key_id != _cas_store_role_key_id_for_public_key_v1(
        expected_public_key
    ):
        raise ValueError("expected CAS_STORE key ID/public key authority binding drift")
    if (
        receipt.signer_public_key_hex != expected_public_key.hex()
        or receipt.signer_key_id != expected_signer_key_id
        or receipt.authority_profile_id != expected_authority_profile_id
        or receipt.signer_registry_id != expected_signer_registry_id
        or receipt.key_epoch != KEY_EPOCH
    ):
        raise ValueError("signed receipt differs from the pinned authority binding")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        raise RuntimeError("cryptography Ed25519 verification backend is unavailable")
    try:
        public_key = _Ed25519PublicKey.from_public_bytes(expected_public_key)
        public_key.verify(
            bytes.fromhex(receipt.ed25519_signature_hex),
            _signature_preimage(receipt.signed_body),
        )
    except _InvalidSignature as exc:
        raise ValueError("CAS_STORE Ed25519 receipt signature is invalid") from exc
    return receipt.signed_body


class Ed25519CasStoreSignerV1:
    """Concrete in-process signer; possession here is mechanics, not authority."""

    __slots__ = ("_private_key", "public_key_bytes", "key_id")

    def __init__(
        self,
        private_key_seed: bytes,
        *,
        signer_key_id: str,
        expected_public_key: bytes,
    ) -> None:
        if type(private_key_seed) is not bytes or len(private_key_seed) != 32:
            raise ValueError("CAS_STORE Ed25519 private key seed must be exact 32 bytes")
        if (
            _Ed25519PrivateKey is None
            or _serialization is None
            or _Ed25519PublicKey is None
        ):
            raise RuntimeError("cryptography Ed25519 signing backend is unavailable")
        self._private_key = _Ed25519PrivateKey.from_private_bytes(private_key_seed)
        self.public_key_bytes = self._private_key.public_key().public_bytes(
            encoding=_serialization.Encoding.Raw,
            format=_serialization.PublicFormat.Raw,
        )
        if (
            type(expected_public_key) is not bytes
            or len(expected_public_key) != 32
            or self.public_key_bytes != expected_public_key
        ):
            raise ValueError("CAS_STORE private seed/public registry key mismatch")
        self.key_id = _require_role_key_id(signer_key_id)
        if self.key_id != _cas_store_role_key_id_for_public_key_v1(
            self.public_key_bytes
        ):
            raise ValueError("CAS_STORE signer key ID/public key authority binding drift")

    def sign(self, body: StoreReceiptBodyV1) -> SignedStoreReceiptV1:
        body.__post_init__()
        if body.signer_key_id != self.key_id:
            raise ValueError("receipt body is not bound to this CAS_STORE signer")
        signature = self._private_key.sign(_signature_preimage(body))
        receipt = SignedStoreReceiptV1(
            schema_version=SIGNED_STORE_RECEIPT_SCHEMA_VERSION,
            purpose=CAS_STORE_RECEIPT_PURPOSE,
            signature_domain=CAS_STORE_SIGNATURE_DOMAIN,
            signed_body=body,
            signed_body_sha256=_sha256(body.canonical_bytes()),
            authority_profile_id=body.authority_profile_id,
            signer_registry_id=body.signer_registry_id,
            key_epoch=body.key_epoch,
            signer_public_key_hex=self.public_key_bytes.hex(),
            signer_key_id=self.key_id,
            ed25519_signature_hex=signature.hex(),
        )
        verify_signed_store_receipt_v1(
            receipt,
            expected_public_key=self.public_key_bytes,
            expected_signer_key_id=self.key_id,
            expected_authority_profile_id=body.authority_profile_id,
            expected_signer_registry_id=body.signer_registry_id,
        )
        return receipt


@dataclass(frozen=True, slots=True)
class StoreCheckpointV1:
    schema_version: str
    purpose: str
    chain_id: str
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    checkpoint_index: int
    previous_checkpoint_sha256: str
    request_id: str
    content_address: str
    size_bytes: int
    signed_receipt_sha256: str
    signed_receipt: SignedStoreReceiptV1

    def __post_init__(self) -> None:
        if type(self) is not StoreCheckpointV1:
            raise TypeError("store checkpoint must use the exact DTO")
        if (
            self.schema_version != STORE_CHECKPOINT_SCHEMA_VERSION
            or self.purpose != CAS_STORE_CHECKPOINT_PURPOSE
        ):
            raise ValueError("checkpoint schema/purpose drift")
        _require_identifier(self.chain_id, "checkpoint chain ID")
        _require_authority_profile_id(self.authority_profile_id)
        _require_signer_registry_id(self.signer_registry_id)
        if type(self.key_epoch) is not int or self.key_epoch != KEY_EPOCH:
            raise ValueError("checkpoint key epoch drift")
        if (
            type(self.checkpoint_index) is not int
            or not 0 <= self.checkpoint_index < EXPECTED_RETAINED_OBJECTS
        ):
            raise ValueError("checkpoint index drift")
        _require_sha256(self.previous_checkpoint_sha256, "previous checkpoint")
        _require_sha256(self.request_id, "checkpoint request ID")
        _require_sha256(self.content_address, "checkpoint content address")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("checkpoint object size drift")
        self.signed_receipt.__post_init__()
        if self.signed_receipt_sha256 != self.signed_receipt.receipt_sha256:
            raise ValueError("checkpoint signed receipt hash drift")
        receipt = self.signed_receipt.signed_body
        if (
            receipt.chain_id != self.chain_id
            or receipt.authority_profile_id != self.authority_profile_id
            or receipt.signer_registry_id != self.signer_registry_id
            or receipt.key_epoch != self.key_epoch
            or receipt.checkpoint_index != self.checkpoint_index
            or receipt.previous_checkpoint_sha256
            != self.previous_checkpoint_sha256
            or receipt.request_id != self.request_id
            or receipt.content_address != self.content_address
            or receipt.size_bytes != self.size_bytes
        ):
            raise ValueError("checkpoint/signed receipt linkage drift")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "chain_id": self.chain_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "key_epoch": self.key_epoch,
            "checkpoint_index": self.checkpoint_index,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "request_id": self.request_id,
            "content_address": self.content_address,
            "size_bytes": self.size_bytes,
            "signed_receipt_sha256": self.signed_receipt_sha256,
            "signed_receipt": self.signed_receipt.to_mapping(),
        }

    def canonical_bytes(self) -> bytes:
        payload = canonicalize_rfc8785_v1(self.to_mapping())
        if len(payload) > MAXIMUM_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds its frozen wire bound")
        return payload

    @property
    def checkpoint_sha256(self) -> str:
        return _sha256(self.canonical_bytes())

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "StoreCheckpointV1":
        if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_CHECKPOINT_BYTES:
            raise ValueError("checkpoint bytes are outside the frozen bound")
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
        body = _require_exact_mapping(decoded, _CHECKPOINT_KEYS, "checkpoint")
        value = cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            key_epoch=_require_wire_integer(
                body["key_epoch"], "checkpoint key epoch", minimum=1, maximum=1
            ),
            checkpoint_index=_require_wire_integer(
                body["checkpoint_index"],
                "checkpoint index",
                maximum=EXPECTED_RETAINED_OBJECTS - 1,
            ),
            previous_checkpoint_sha256=body["previous_checkpoint_sha256"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
            content_address=body["content_address"],  # type: ignore[arg-type]
            size_bytes=_require_wire_integer(
                body["size_bytes"], "checkpoint object size", minimum=1
            ),
            signed_receipt_sha256=body["signed_receipt_sha256"],  # type: ignore[arg-type]
            signed_receipt=SignedStoreReceiptV1.from_mapping(
                body["signed_receipt"]
            ),
        )
        if value.canonical_bytes() != payload:
            raise ValueError("checkpoint DTO/canonical byte binding drift")
        return value


@dataclass(frozen=True, slots=True)
class StoreHeadV1:
    schema_version: str
    purpose: str
    chain_id: str
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    checkpoint_count: int
    last_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not StoreHeadV1:
            raise TypeError("store head must use the exact DTO")
        if (
            self.schema_version != STORE_HEAD_SCHEMA_VERSION
            or self.purpose != CAS_STORE_HEAD_PURPOSE
        ):
            raise ValueError("store head schema/purpose drift")
        _require_identifier(self.chain_id, "head chain ID")
        _require_authority_profile_id(self.authority_profile_id)
        _require_signer_registry_id(self.signer_registry_id)
        if type(self.key_epoch) is not int or self.key_epoch != KEY_EPOCH:
            raise ValueError("store head key epoch drift")
        if (
            type(self.checkpoint_count) is not int
            or not 0 <= self.checkpoint_count <= EXPECTED_RETAINED_OBJECTS
        ):
            raise ValueError("head checkpoint count drift")
        _require_sha256(self.last_checkpoint_sha256, "head checkpoint SHA-256")
        if (
            self.checkpoint_count == 0
            and self.last_checkpoint_sha256 != GENESIS_CHECKPOINT_SHA256
        ):
            raise ValueError("empty store head does not use the genesis link")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "chain_id": self.chain_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "key_epoch": self.key_epoch,
            "checkpoint_count": self.checkpoint_count,
            "last_checkpoint_sha256": self.last_checkpoint_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonicalize_rfc8785_v1(self.to_mapping())

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "StoreHeadV1":
        if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_HEAD_BYTES:
            raise ValueError("store head bytes are outside the frozen bound")
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
        body = _require_exact_mapping(decoded, _HEAD_KEYS, "store head")
        value = cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            key_epoch=_require_wire_integer(
                body["key_epoch"], "head key epoch", minimum=1, maximum=1
            ),
            checkpoint_count=_require_wire_integer(
                body["checkpoint_count"],
                "head checkpoint count",
                maximum=EXPECTED_RETAINED_OBJECTS,
            ),
            last_checkpoint_sha256=body["last_checkpoint_sha256"],  # type: ignore[arg-type]
        )
        if value.canonical_bytes() != payload:
            raise ValueError("store head DTO/canonical byte binding drift")
        return value


@dataclass(slots=True)
class ReceivedStoreRequestV1:
    request: StoreRequestV1
    fd: int
    peer: PeerCredentialsV1
    _closed: bool = False

    def __post_init__(self) -> None:
        if type(self) is not ReceivedStoreRequestV1:
            raise TypeError("received store request must use the exact DTO")
        self.request.__post_init__()
        self.peer.__post_init__()
        if type(self.fd) is not int or self.fd < 0:
            raise ValueError("received descriptor is invalid")

    def close(self) -> None:
        if not self._closed:
            os.close(self.fd)
            self._closed = True

    def __enter__(self) -> "ReceivedStoreRequestV1":
        if self._closed:
            raise RuntimeError("received request descriptor is already closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


def _peer_credentials(connection: socket.socket) -> PeerCredentialsV1:
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    if type(raw) is not bytes or len(raw) != 12:
        raise RuntimeError("SO_PEERCRED returned a non-canonical credential record")
    pid, uid, gid = struct.unpack("3i", raw)
    return PeerCredentialsV1(pid=pid, uid=uid, gid=gid)


def _require_unix_seqpacket(connection: socket.socket) -> None:
    if type(connection) is not socket.socket:
        raise TypeError("SCM transport requires the exact socket type")
    if connection.family != socket.AF_UNIX:
        raise ValueError("SCM transport requires AF_UNIX")
    socket_type = connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
    if socket_type != socket.SOCK_SEQPACKET:
        raise ValueError("SCM transport requires SOCK_SEQPACKET message boundaries")


def recv_store_request_v1(
    connection: socket.socket,
    *,
    allowed_peer_uids: frozenset[int],
) -> ReceivedStoreRequestV1:
    """Receive exactly one canonical runtime request and one SCM_RIGHTS FD."""

    _require_no_follow_support()
    _require_unix_seqpacket(connection)
    if (
        type(allowed_peer_uids) is not frozenset
        or not allowed_peer_uids
        or any(type(item) is not int or item < 0 for item in allowed_peer_uids)
    ):
        raise ValueError("allowed peer UIDs must be one nonempty exact frozenset")
    peer = _peer_credentials(connection)
    if peer.uid not in allowed_peer_uids:
        raise PermissionError("SO_PEERCRED UID is not authorized for CAS_STORE")
    descriptor_size = array("i").itemsize
    flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
    payload, ancillary, message_flags, address = connection.recvmsg(
        MAXIMUM_REQUEST_BYTES + 1,
        socket.CMSG_SPACE(descriptor_size * 2),
        flags,
    )
    received_fds: list[int] = []
    try:
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                if len(data) % descriptor_size:
                    raise ValueError("SCM_RIGHTS control data alignment drift")
                values = array("i")
                values.frombytes(data)
                received_fds.extend(int(item) for item in values)
            else:
                raise ValueError("SCM transport carried an unexpected control record")
        if message_flags & (
            getattr(socket, "MSG_TRUNC", 0) | getattr(socket, "MSG_CTRUNC", 0)
        ):
            raise ValueError("SCM request or control record was truncated")
        if address not in {None, "", b""}:
            raise ValueError("connected SCM request carried an unexpected source address")
        if len(received_fds) != 1:
            raise ValueError("SCM request must carry exactly one descriptor")
        if not 0 < len(payload) <= MAXIMUM_REQUEST_BYTES:
            raise ValueError("SCM request payload is outside the frozen byte bound")
        received_fd = received_fds[0]
        os.set_inheritable(received_fd, False)
        request = StoreRequestV1.from_canonical_bytes(payload)
        result = ReceivedStoreRequestV1(request=request, fd=received_fd, peer=peer)
        received_fds.clear()
        return result
    finally:
        for descriptor in received_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


def send_store_request_fd_v1(
    connection: socket.socket,
    request: StoreRequestV1,
    fd: int,
) -> None:
    """Send one exact request/FD packet; this does not verify a server reply."""

    _require_unix_seqpacket(connection)
    if type(request) is not StoreRequestV1:
        raise TypeError("SCM sender requires the exact store request DTO")
    request.__post_init__()
    if type(fd) is not int or fd < 0:
        raise ValueError("SCM sender descriptor is invalid")
    payload = request.canonical_bytes()
    descriptors = array("i", [fd])
    sent = connection.sendmsg(
        [payload],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())],
    )
    if sent != len(payload):
        raise OSError("SCM request send was not one complete packet")


@dataclass(frozen=True, slots=True)
class RetentionAcknowledgementV1:
    schema_version: str
    attempt_context_id: str
    retention_session_id: str
    request_id: str
    ordinal: int
    mode: str
    phase: str
    source_basename: str
    content_address: str
    sha256: str
    size_bytes: int
    durable_before_return: bool
    readback_verified: bool
    store_receipt_ref: str

    def __post_init__(self) -> None:
        if type(self) is not RetentionAcknowledgementV1:
            raise TypeError("retention acknowledgement must use the exact store DTO")
        if self.schema_version != RETENTION_ACK_SCHEMA_VERSION:
            raise ValueError("retention acknowledgement schema drift")
        _require_identifier(self.attempt_context_id, "ack attempt context ID")
        _require_identifier(self.retention_session_id, "ack retention session ID")
        _require_sha256(self.request_id, "ack request ID")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < (
            EXPECTED_RETAINED_OBJECTS
        ):
            raise ValueError("retention acknowledgement ordinal drift")
        expected_mode, expected_phase, expected_target = EXPECTED_OBJECT_ORDER[
            self.ordinal
        ]
        expected_basename = (
            INVARIANT_BASENAME
            if expected_target is None
            else STATISTICS_BASENAMES[expected_target]
        )
        if (
            self.mode != expected_mode
            or self.phase != expected_phase
            or self.source_basename != expected_basename
        ):
            raise ValueError("retention acknowledgement worker identity drift")
        _require_sha256(self.content_address, "ack content address")
        _require_sha256(self.sha256, "ack SHA-256")
        if self.content_address != self.sha256:
            raise ValueError("retention acknowledgement content address drift")
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("retention acknowledgement size drift")
        if self.durable_before_return is not True or self.readback_verified is not True:
            raise ValueError("retention acknowledgement durability/readback drift")
        _require_identifier(self.store_receipt_ref, "ack store receipt reference")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "request_id": self.request_id,
            "ordinal": self.ordinal,
            "mode": self.mode,
            "phase": self.phase,
            "source_basename": self.source_basename,
            "content_address": self.content_address,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "durable_before_return": self.durable_before_return,
            "readback_verified": self.readback_verified,
            "store_receipt_ref": self.store_receipt_ref,
        }

    def to_runtime_ack_v1(self) -> object:
        # Producer-only conversion is intentionally lazy.  Server import and
        # server request handling never load auditor, QV2, or runtime modules.
        from . import phase2b_formal_wire_covert_step5_runtime_v1 as runtime_module

        return runtime_module.RetentionIngestAckV1(**self.to_mapping())


def _ack_from_mapping(value: object) -> RetentionAcknowledgementV1:
    body = _require_exact_mapping(value, _ACK_KEYS, "retention acknowledgement")
    return RetentionAcknowledgementV1(
        schema_version=body["schema_version"],  # type: ignore[arg-type]
        attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
        retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
        request_id=body["request_id"],  # type: ignore[arg-type]
        ordinal=_require_wire_integer(
            body["ordinal"],
            "ack ordinal",
            maximum=EXPECTED_RETAINED_OBJECTS - 1,
        ),
        mode=body["mode"],  # type: ignore[arg-type]
        phase=body["phase"],  # type: ignore[arg-type]
        source_basename=body["source_basename"],  # type: ignore[arg-type]
        content_address=body["content_address"],  # type: ignore[arg-type]
        sha256=body["sha256"],  # type: ignore[arg-type]
        size_bytes=_require_wire_integer(body["size_bytes"], "ack size", minimum=1),
        durable_before_return=body["durable_before_return"],  # type: ignore[arg-type]
        readback_verified=body["readback_verified"],  # type: ignore[arg-type]
        store_receipt_ref=body["store_receipt_ref"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class StoreResponseV1:
    schema_version: str
    purpose: str
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    request_id: str
    checkpoint_sha256: str
    checkpoint_record: StoreCheckpointV1
    acknowledgement: RetentionAcknowledgementV1
    claim_boundary: str

    def __post_init__(self) -> None:
        if type(self) is not StoreResponseV1:
            raise TypeError("store response must use the exact DTO")
        if (
            self.schema_version != STORE_RESPONSE_SCHEMA_VERSION
            or self.purpose != CAS_STORE_RESPONSE_PURPOSE
            or self.claim_boundary != AUTHORITY_CLAIM_BOUNDARY
        ):
            raise ValueError("store response schema/purpose/claim drift")
        _require_authority_profile_id(self.authority_profile_id)
        _require_signer_registry_id(self.signer_registry_id)
        if type(self.key_epoch) is not int or self.key_epoch != KEY_EPOCH:
            raise ValueError("store response key epoch drift")
        _require_sha256(self.request_id, "response request ID")
        self.checkpoint_record.__post_init__()
        if (
            self.checkpoint_sha256 != self.checkpoint_record.checkpoint_sha256
            or self.checkpoint_record.authority_profile_id
            != self.authority_profile_id
            or self.checkpoint_record.signer_registry_id
            != self.signer_registry_id
            or self.checkpoint_record.key_epoch != self.key_epoch
        ):
            raise ValueError("response checkpoint hash drift")
        self.acknowledgement.__post_init__()
        receipt = self.checkpoint_record.signed_receipt
        body = receipt.signed_body
        expected_ref = "cas-store-receipt-sha256-" + receipt.receipt_sha256[7:]
        if (
            self.request_id != body.request_id
            or self.acknowledgement.request_id != body.request_id
            or self.acknowledgement.attempt_context_id
            != body.attempt_context_id
            or self.acknowledgement.retention_session_id
            != body.retention_session_id
            or self.acknowledgement.ordinal != body.ordinal
            or self.acknowledgement.mode != body.mode
            or self.acknowledgement.phase != body.phase
            or self.acknowledgement.source_basename != body.source_basename
            or self.acknowledgement.content_address != body.content_address
            or self.acknowledgement.sha256 != body.sha256
            or self.acknowledgement.size_bytes != body.size_bytes
            or self.acknowledgement.store_receipt_ref != expected_ref
        ):
            raise ValueError("response acknowledgement/signed receipt linkage drift")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "key_epoch": self.key_epoch,
            "request_id": self.request_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_record": self.checkpoint_record.to_mapping(),
            "acknowledgement": self.acknowledgement.to_mapping(),
            "claim_boundary": self.claim_boundary,
        }

    def canonical_bytes(self) -> bytes:
        payload = canonicalize_rfc8785_v1(self.to_mapping())
        if len(payload) > MAXIMUM_RESPONSE_BYTES:
            raise ValueError("store response exceeds its frozen wire bound")
        return payload

    @classmethod
    def from_canonical_bytes(
        cls,
        payload: bytes,
        *,
        expected_public_key: bytes,
        expected_signer_key_id: str,
        expected_authority_profile_id: str,
        expected_signer_registry_id: str,
    ) -> "StoreResponseV1":
        if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_RESPONSE_BYTES:
            raise ValueError("store response bytes are outside the frozen bound")
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
        body = _require_exact_mapping(decoded, _RESPONSE_KEYS, "store response")
        checkpoint_payload = canonicalize_rfc8785_v1(body["checkpoint_record"])
        checkpoint = StoreCheckpointV1.from_canonical_bytes(checkpoint_payload)
        verify_signed_store_receipt_v1(
            checkpoint.signed_receipt,
            expected_public_key=expected_public_key,
            expected_signer_key_id=expected_signer_key_id,
            expected_authority_profile_id=expected_authority_profile_id,
            expected_signer_registry_id=expected_signer_registry_id,
        )
        value = cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            key_epoch=_require_wire_integer(
                body["key_epoch"], "response key epoch", minimum=1, maximum=1
            ),
            request_id=body["request_id"],  # type: ignore[arg-type]
            checkpoint_sha256=body["checkpoint_sha256"],  # type: ignore[arg-type]
            checkpoint_record=checkpoint,
            acknowledgement=_ack_from_mapping(body["acknowledgement"]),
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
        )
        if value.canonical_bytes() != payload:
            raise ValueError("store response DTO/canonical byte binding drift")
        return value


class CasStoreV1:
    """Durable local mechanics; caller must supply actual deployment isolation."""

    __slots__ = (
        "root_directory",
        "chain_id",
        "attempt_context_id",
        "retention_session_id",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "allowed_peer_uids",
        "signer_public_key",
        "signer_key_id",
        "_signer",
        "_root_fd",
        "_objects_fd",
        "_staging_fd",
        "_checkpoints_fd",
        "_state_fd",
        "_lock_fd",
        "_thread_lock",
        "_nonce",
        "_closed",
    )

    def __init__(
        self,
        *,
        root_directory: str,
        chain_id: str,
        attempt_context_id: str,
        retention_session_id: str,
        private_key_seed: bytes,
        allowed_peer_uids: frozenset[int],
        stage_b_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id: str,
        out_of_band_expected_lock_payload_sha256: str,
        create: bool = False,
    ) -> None:
        _require_no_follow_support()
        _require_frozen_contract_v1()
        if type(root_directory) is not str or not os.path.isabs(root_directory):
            raise ValueError("CAS store root must be one absolute path")
        normalized = os.path.abspath(root_directory)
        if normalized != root_directory:
            raise ValueError("CAS store root path is not lexically canonical")
        if (
            type(allowed_peer_uids) is not frozenset
            or not allowed_peer_uids
            or any(type(item) is not int or item < 0 for item in allowed_peer_uids)
        ):
            raise ValueError("allowed peer UIDs must be one nonempty exact frozenset")
        self.root_directory = root_directory
        self.chain_id = _require_identifier(chain_id, "CAS store chain ID")
        self.attempt_context_id = _require_identifier(
            attempt_context_id,
            "CAS store attempt context ID",
        )
        self.retention_session_id = _require_identifier(
            retention_session_id,
            "CAS store retention session ID",
        )
        self.allowed_peer_uids = allowed_peer_uids
        binding = _load_and_bind_cas_store_authority_v1(
            stage_b_lock_payload=stage_b_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id=out_of_band_expected_lock_id,
            out_of_band_expected_lock_payload_sha256=(
                out_of_band_expected_lock_payload_sha256
            ),
        )
        self.authority_profile_id = binding.authority_profile_id
        self.signer_registry_id = binding.signer_registry_id
        self.key_epoch = binding.key_epoch
        self._signer = Ed25519CasStoreSignerV1(
            private_key_seed,
            signer_key_id=binding.signer_key_id,
            expected_public_key=binding.public_key_bytes,
        )
        self.signer_public_key = self._signer.public_key_bytes
        self.signer_key_id = self._signer.key_id
        self._root_fd = -1
        self._objects_fd = -1
        self._staging_fd = -1
        self._checkpoints_fd = -1
        self._state_fd = -1
        self._lock_fd = -1
        self._thread_lock = threading.Lock()
        self._nonce = 0
        self._closed = False
        try:
            if create:
                try:
                    os.mkdir(root_directory, 0o700)
                except FileExistsError:
                    pass
            if os.path.realpath(root_directory) != root_directory:
                raise ValueError("CAS store root resolves through a symlink")
            self._root_fd = os.open(
                root_directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            self._require_private_directory(self._root_fd, "CAS store root")
            self._objects_fd = self._open_or_create_directory("objects", create)
            self._staging_fd = self._open_or_create_directory("staging", create)
            self._checkpoints_fd = self._open_or_create_directory(
                "checkpoints", create
            )
            self._state_fd = self._open_or_create_directory("state", create)
            lock_flags = (
                os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_CREAT
                if create
                else os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            self._lock_fd = os.open(
                "store.lock",
                lock_flags,
                0o600,
                dir_fd=self._root_fd,
            )
            lock_stat = os.fstat(self._lock_fd)
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or lock_stat.st_nlink != 1
                or lock_stat.st_uid != os.geteuid()
                or lock_stat.st_mode & 0o077
                or lock_stat.st_size != 0
            ):
                raise PermissionError("CAS store lock file ownership/mode drift")
            os.fsync(self._lock_fd)
            os.fsync(self._root_fd)
            if set(os.listdir(self._root_fd)) != {
                "objects",
                "staging",
                "checkpoints",
                "state",
                "store.lock",
            }:
                raise ValueError("CAS store root entry set drift")
            with self._exclusive_store_lock():
                self._require_empty_staging()
                records = self._load_chain_and_repair_head()
                self._validate_object_namespace(records)
        except BaseException:
            self.close()
            raise

    def _require_private_directory(self, fd: int, label: str) -> None:
        observed = os.fstat(fd)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_mode & 0o077
        ):
            raise PermissionError(f"{label} is not an owner-private directory")

    def _open_or_create_directory(self, basename: str, create: bool) -> int:
        if create:
            try:
                os.mkdir(basename, 0o700, dir_fd=self._root_fd)
                os.fsync(self._root_fd)
            except FileExistsError:
                pass
        fd = os.open(
            basename,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._root_fd,
        )
        try:
            self._require_private_directory(fd, f"CAS store {basename} directory")
            return fd
        except BaseException:
            os.close(fd)
            raise

    class _StoreLockContext:
        def __init__(self, owner: "CasStoreV1") -> None:
            self.owner = owner

        def __enter__(self) -> None:
            if self.owner._closed or self.owner._lock_fd < 0:
                raise RuntimeError("CAS store is closed")
            self.owner._thread_lock.acquire()
            try:
                fcntl.flock(self.owner._lock_fd, fcntl.LOCK_EX)
            except BaseException:
                self.owner._thread_lock.release()
                raise

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
            try:
                fcntl.flock(self.owner._lock_fd, fcntl.LOCK_UN)
            finally:
                self.owner._thread_lock.release()
            return False

    def _exclusive_store_lock(self) -> "CasStoreV1._StoreLockContext":
        return self._StoreLockContext(self)

    def _require_empty_staging(self) -> None:
        names = tuple(os.listdir(self._staging_fd))
        if names:
            raise RuntimeError(
                "CAS staging is nonempty; crash recovery requires explicit operator review"
            )

    def _head_for_records(
        self, records: tuple[StoreCheckpointV1, ...]
    ) -> StoreHeadV1:
        return StoreHeadV1(
            schema_version=STORE_HEAD_SCHEMA_VERSION,
            purpose=CAS_STORE_HEAD_PURPOSE,
            chain_id=self.chain_id,
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            key_epoch=self.key_epoch,
            checkpoint_count=len(records),
            last_checkpoint_sha256=(
                GENESIS_CHECKPOINT_SHA256
                if not records
                else records[-1].checkpoint_sha256
            ),
        )

    def _write_head(self, head: StoreHeadV1) -> None:
        self._nonce += 1
        _atomic_replace_file_at(
            self._state_fd,
            "head.json",
            head.canonical_bytes(),
            maximum_bytes=MAXIMUM_HEAD_BYTES,
            nonce=self._nonce,
        )

    def _load_chain_and_repair_head(self) -> tuple[StoreCheckpointV1, ...]:
        names = tuple(sorted(os.listdir(self._checkpoints_fd)))
        if any(_CHECKPOINT_NAME_PATTERN.fullmatch(name) is None for name in names):
            raise ValueError("checkpoint directory contains an unknown entry")
        records: list[StoreCheckpointV1] = []
        previous = GENESIS_CHECKPOINT_SHA256
        request_ids: set[str] = set()
        hashes: list[str] = []
        for index, name in enumerate(names):
            if name != _checkpoint_name(index):
                raise ValueError("checkpoint filenames are not one contiguous chain")
            payload = _read_regular_at(
                self._checkpoints_fd, name, MAXIMUM_CHECKPOINT_BYTES
            )
            record = StoreCheckpointV1.from_canonical_bytes(payload)
            verify_signed_store_receipt_v1(
                record.signed_receipt,
                expected_public_key=self.signer_public_key,
                expected_signer_key_id=self.signer_key_id,
                expected_authority_profile_id=self.authority_profile_id,
                expected_signer_registry_id=self.signer_registry_id,
            )
            receipt_body = record.signed_receipt.signed_body
            if (
                record.chain_id != self.chain_id
                or record.authority_profile_id != self.authority_profile_id
                or record.signer_registry_id != self.signer_registry_id
                or record.key_epoch != self.key_epoch
                or record.checkpoint_index != index
                or receipt_body.ordinal != index
                or receipt_body.attempt_context_id != self.attempt_context_id
                or receipt_body.retention_session_id
                != self.retention_session_id
                or record.previous_checkpoint_sha256 != previous
                or record.request_id in request_ids
            ):
                raise ValueError("checkpoint chain continuity/request uniqueness drift")
            request_ids.add(record.request_id)
            previous = record.checkpoint_sha256
            hashes.append(previous)
            records.append(record)
        expected = self._head_for_records(tuple(records))
        state_names = tuple(os.listdir(self._state_fd))
        if state_names not in {(), ("head.json",)}:
            raise ValueError("CAS state directory contains an unknown entry")
        if not state_names:
            self._write_head(expected)
            return tuple(records)
        observed = StoreHeadV1.from_canonical_bytes(
            _read_regular_at(self._state_fd, "head.json", MAXIMUM_HEAD_BYTES)
        )
        if (
            observed.chain_id != self.chain_id
            or observed.authority_profile_id != self.authority_profile_id
            or observed.signer_registry_id != self.signer_registry_id
            or observed.key_epoch != self.key_epoch
        ):
            raise ValueError("CAS head chain/authority binding drift")
        if observed == expected:
            return tuple(records)
        if not 0 <= observed.checkpoint_count <= len(records):
            raise ValueError("CAS head is ahead of its append-only checkpoint files")
        prefix_hash = (
            GENESIS_CHECKPOINT_SHA256
            if observed.checkpoint_count == 0
            else hashes[observed.checkpoint_count - 1]
        )
        if observed.last_checkpoint_sha256 != prefix_hash:
            raise ValueError("CAS head is not an exact prefix of the checkpoint chain")
        self._write_head(expected)
        return tuple(records)

    def _validate_object_namespace(
        self,
        records: tuple[StoreCheckpointV1, ...],
    ) -> None:
        names = tuple(os.listdir(self._objects_fd))
        if any(_OBJECT_NAME_PATTERN.fullmatch(name) is None for name in names):
            raise ValueError("CAS object directory contains an unknown entry")
        maximum_object_bytes = max(
            INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND,
            MAXIMUM_STATISTICS_ARTIFACT_BYTES,
        )
        for name in names:
            payload = _read_regular_at(
                self._objects_fd,
                name,
                maximum_object_bytes,
            )
            if _sha256(payload) != "sha256:" + name:
                raise ValueError("CAS object namespace/content-address binding drift")
        for record in records:
            basename = record.content_address[7:]
            payload = _read_regular_at(
                self._objects_fd,
                basename,
                max(record.size_bytes, 1),
            )
            if len(payload) != record.size_bytes or _sha256(payload) != (
                record.content_address
            ):
                raise ValueError("checkpoint-referenced CAS object readback drift")

    def _hash_received_fd(
        self,
        request: StoreRequestV1,
        fd: int,
    ) -> tuple[os.stat_result, str, int]:
        if type(fd) is not int or fd < 0:
            raise ValueError("CAS ingest requires one valid descriptor")
        access_mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        if access_mode != os.O_RDONLY:
            raise ValueError("SCM_RIGHTS artifact descriptor is not read-only")
        before = os.fstat(fd)
        maximum = (
            INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND
            if request.phase == "INVARIANCE"
            else MAXIMUM_STATISTICS_ARTIFACT_BYTES
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ValueError("SCM_RIGHTS artifact is not one bounded regular link")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block:
                raise ValueError("SCM_RIGHTS artifact ended during independent hash")
            digest.update(block)
            offset += len(block)
        if os.pread(fd, 1, offset):
            raise ValueError("SCM_RIGHTS artifact grew during independent hash")
        after = os.fstat(fd)
        if not _same_file_identity(before, after):
            raise ValueError("SCM_RIGHTS artifact changed during independent hash")
        observed_sha256 = "sha256:" + digest.hexdigest()
        if (
            observed_sha256 != request.auditor_reported_sha256
            or offset != request.auditor_reported_size_bytes
        ):
            raise ValueError("CAS_STORE hash/size differs from runtime request/1")
        return after, observed_sha256, offset

    def _persist_object_if_absent(
        self,
        source_fd: int,
        source_snapshot: os.stat_result,
        sha256: str,
        size_bytes: int,
    ) -> str:
        basename = sha256[7:]
        if _OBJECT_NAME_PATTERN.fullmatch(basename) is None:
            raise ValueError("CAS object content address basename drift")
        self._require_empty_staging()
        self._nonce += 1
        temporary = f"pending-{os.getpid()}-{self._nonce}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        temporary_fd: int | None = None
        disposition = "CREATED"
        try:
            temporary_fd = os.open(
                temporary,
                flags,
                0o600,
                dir_fd=self._staging_fd,
            )
            digest = hashlib.sha256()
            offset = 0
            while offset < size_bytes:
                block = os.pread(source_fd, min(1_048_576, size_bytes - offset), offset)
                if not block:
                    raise ValueError("source artifact ended during CAS copy")
                digest.update(block)
                _write_all(temporary_fd, block)
                offset += len(block)
            if os.pread(source_fd, 1, offset):
                raise ValueError("source artifact grew during CAS copy")
            after_copy = os.fstat(source_fd)
            if (
                not _same_file_identity(source_snapshot, after_copy)
                or offset != size_bytes
                or "sha256:" + digest.hexdigest() != sha256
            ):
                raise ValueError("source artifact changed during CAS copy")
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            try:
                os.link(
                    temporary,
                    basename,
                    src_dir_fd=self._staging_fd,
                    dst_dir_fd=self._objects_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                disposition = "PRESENT_VERIFIED"
            os.unlink(temporary, dir_fd=self._staging_fd)
            os.fsync(self._staging_fd)
            os.fsync(self._objects_fd)
            stored = _read_regular_at(self._objects_fd, basename, size_bytes)
            if len(stored) != size_bytes or _sha256(stored) != sha256:
                raise ValueError("CAS object reopen/readback verification drift")
            object_fd = os.open(
                basename,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self._objects_fd,
            )
            try:
                os.fsync(object_fd)
            finally:
                os.close(object_fd)
            return disposition
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            try:
                os.unlink(temporary, dir_fd=self._staging_fd)
                os.fsync(self._staging_fd)
            except FileNotFoundError:
                pass

    def _append_checkpoint(self, checkpoint: StoreCheckpointV1) -> None:
        payload = checkpoint.canonical_bytes()
        name = _checkpoint_name(checkpoint.checkpoint_index)
        self._nonce += 1
        temporary = f".{name}.pending-{os.getpid()}-{self._nonce}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        fd: int | None = None
        try:
            fd = os.open(temporary, flags, 0o600, dir_fd=self._checkpoints_fd)
            _write_all(fd, payload)
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.link(
                temporary,
                name,
                src_dir_fd=self._checkpoints_fd,
                dst_dir_fd=self._checkpoints_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary, dir_fd=self._checkpoints_fd)
            os.fsync(self._checkpoints_fd)
            if _read_regular_at(
                self._checkpoints_fd, name, MAXIMUM_CHECKPOINT_BYTES
            ) != payload:
                raise ValueError(
                    "append-only checkpoint readback differs from commit"
                )
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=self._checkpoints_fd)
                os.fsync(self._checkpoints_fd)
            except FileNotFoundError:
                pass

    def _response_for_checkpoint(
        self,
        checkpoint: StoreCheckpointV1,
    ) -> StoreResponseV1:
        body = checkpoint.signed_receipt.signed_body
        acknowledgement = RetentionAcknowledgementV1(
            schema_version=RETENTION_ACK_SCHEMA_VERSION,
            attempt_context_id=body.attempt_context_id,
            retention_session_id=body.retention_session_id,
            request_id=body.request_id,
            ordinal=body.ordinal,
            mode=body.mode,
            phase=body.phase,
            source_basename=body.source_basename,
            content_address=body.content_address,
            sha256=body.sha256,
            size_bytes=body.size_bytes,
            durable_before_return=True,
            readback_verified=True,
            store_receipt_ref=(
                "cas-store-receipt-sha256-"
                + checkpoint.signed_receipt.receipt_sha256[7:]
            ),
        )
        response = StoreResponseV1(
            schema_version=STORE_RESPONSE_SCHEMA_VERSION,
            purpose=CAS_STORE_RESPONSE_PURPOSE,
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            key_epoch=self.key_epoch,
            request_id=body.request_id,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
            checkpoint_record=checkpoint,
            acknowledgement=acknowledgement,
            claim_boundary=AUTHORITY_CLAIM_BOUNDARY,
        )
        StoreResponseV1.from_canonical_bytes(
            response.canonical_bytes(),
            expected_public_key=self.signer_public_key,
            expected_signer_key_id=self.signer_key_id,
            expected_authority_profile_id=self.authority_profile_id,
            expected_signer_registry_id=self.signer_registry_id,
        )
        return response

    def _require_exact_replay(
        self,
        request: StoreRequestV1,
        fd: int,
        peer: PeerCredentialsV1,
        checkpoint: StoreCheckpointV1,
    ) -> StoreResponseV1:
        body = checkpoint.signed_receipt.signed_body
        if (
            request.attempt_context_id != body.attempt_context_id
            or request.retention_session_id != body.retention_session_id
            or request.request_id != body.request_id
            or request.ordinal != body.ordinal
            or request.mode != body.mode
            or request.phase != body.phase
            or request.target_index_or_null != body.target_index_or_null
            or request.source_basename != body.source_basename
            or request.auditor_reported_sha256 != body.sha256
            or request.auditor_reported_size_bytes != body.size_bytes
            or peer.pid != body.peer_pid
            or peer.uid != body.peer_uid
            or peer.gid != body.peer_gid
        ):
            raise ValueError("duplicate request replay context/peer drift")
        snapshot, sha256, size_bytes = self._hash_received_fd(request, fd)
        stored = _read_regular_at(
            self._objects_fd,
            body.content_address[7:],
            body.size_bytes,
        )
        if (
            sha256 != body.sha256
            or size_bytes != body.size_bytes
            or _sha256(stored) != body.content_address
            or len(stored) != body.size_bytes
            or not _same_file_identity(snapshot, os.fstat(fd))
        ):
            raise ValueError("duplicate request FD/object replay drift")
        return self._response_for_checkpoint(checkpoint)

    def ingest_open_fd(
        self,
        request: StoreRequestV1,
        fd: int,
        peer: PeerCredentialsV1,
    ) -> StoreResponseV1:
        """Ingest a borrowed FD; success means object, checkpoint, and head are durable."""

        if type(request) is not StoreRequestV1:
            raise TypeError("CAS ingest requires the exact store request DTO")
        if type(peer) is not PeerCredentialsV1:
            raise TypeError("CAS ingest requires exact peer credentials")
        request.__post_init__()
        peer.__post_init__()
        if peer.uid not in self.allowed_peer_uids:
            raise PermissionError("peer UID is not authorized for CAS_STORE ingest")
        if (
            request.attempt_context_id != self.attempt_context_id
            or request.retention_session_id != self.retention_session_id
        ):
            raise ValueError("request differs from the frozen CAS store session")
        with self._exclusive_store_lock():
            self._require_empty_staging()
            records = self._load_chain_and_repair_head()
            duplicate = tuple(
                item for item in records if item.request_id == request.request_id
            )
            if duplicate:
                if len(duplicate) != 1:
                    raise ValueError("duplicate request checkpoint cardinality drift")
                return self._require_exact_replay(request, fd, peer, duplicate[0])
            if (
                len(records) >= EXPECTED_RETAINED_OBJECTS
                or request.ordinal != len(records)
            ):
                raise ValueError("CAS_STORE request/checkpoint ordinal order drift")
            snapshot, sha256, size_bytes = self._hash_received_fd(request, fd)
            disposition = self._persist_object_if_absent(
                fd,
                snapshot,
                sha256,
                size_bytes,
            )
            after_store = os.fstat(fd)
            if not _same_file_identity(snapshot, after_store):
                raise ValueError("source artifact changed after CAS persistence")
            index = len(records)
            previous = (
                GENESIS_CHECKPOINT_SHA256
                if not records
                else records[-1].checkpoint_sha256
            )
            receipt_body = StoreReceiptBodyV1(
                schema_version=STORE_RECEIPT_BODY_SCHEMA_VERSION,
                purpose=CAS_STORE_RECEIPT_PURPOSE,
                signature_domain=CAS_STORE_SIGNATURE_DOMAIN,
                chain_id=self.chain_id,
                authority_profile_id=self.authority_profile_id,
                signer_registry_id=self.signer_registry_id,
                key_epoch=self.key_epoch,
                signer_key_id=self.signer_key_id,
                attempt_context_id=request.attempt_context_id,
                retention_session_id=request.retention_session_id,
                request_id=request.request_id,
                ordinal=request.ordinal,
                mode=request.mode,
                phase=request.phase,
                target_index_or_null=request.target_index_or_null,
                source_basename=request.source_basename,
                peer_pid=peer.pid,
                peer_uid=peer.uid,
                peer_gid=peer.gid,
                content_address=sha256,
                sha256=sha256,
                size_bytes=size_bytes,
                object_disposition=disposition,
                checkpoint_index=index,
                previous_checkpoint_sha256=previous,
                durable_before_return=True,
                readback_verified=True,
                claim_boundary=AUTHORITY_CLAIM_BOUNDARY,
            )
            signed_receipt = self._signer.sign(receipt_body)
            checkpoint = StoreCheckpointV1(
                schema_version=STORE_CHECKPOINT_SCHEMA_VERSION,
                purpose=CAS_STORE_CHECKPOINT_PURPOSE,
                chain_id=self.chain_id,
                authority_profile_id=self.authority_profile_id,
                signer_registry_id=self.signer_registry_id,
                key_epoch=self.key_epoch,
                checkpoint_index=index,
                previous_checkpoint_sha256=previous,
                request_id=request.request_id,
                content_address=sha256,
                size_bytes=size_bytes,
                signed_receipt_sha256=signed_receipt.receipt_sha256,
                signed_receipt=signed_receipt,
            )
            self._append_checkpoint(checkpoint)
            committed_records = (*records, checkpoint)
            self._write_head(self._head_for_records(committed_records))
            return self._response_for_checkpoint(checkpoint)

    def receive_and_ingest_once(self, connection: socket.socket) -> StoreResponseV1:
        received = recv_store_request_v1(
            connection,
            allowed_peer_uids=self.allowed_peer_uids,
        )
        with received:
            return self.ingest_open_fd(received.request, received.fd, received.peer)

    def handle_one_connection(self, connection: socket.socket) -> StoreResponseV1:
        """Receive, durably commit, then send one canonical success response."""

        response = self.receive_and_ingest_once(connection)
        payload = response.canonical_bytes()
        sent = connection.send(payload)
        if sent != len(payload):
            raise OSError("CAS_STORE response send was not one complete packet")
        return response

    def audit_chain(self) -> tuple[StoreCheckpointV1, ...]:
        with self._exclusive_store_lock():
            records = self._load_chain_and_repair_head()
            self._validate_object_namespace(records)
            return records

    def audit_complete_session(self) -> tuple[StoreCheckpointV1, ...]:
        records = self.audit_chain()
        observed = tuple(
            (
                record.signed_receipt.signed_body.ordinal,
                record.signed_receipt.signed_body.mode,
                record.signed_receipt.signed_body.phase,
                record.signed_receipt.signed_body.target_index_or_null,
                record.signed_receipt.signed_body.attempt_context_id,
                record.signed_receipt.signed_body.retention_session_id,
            )
            for record in records
        )
        expected = tuple(
            (
                ordinal,
                mode,
                phase,
                target,
                self.attempt_context_id,
                self.retention_session_id,
            )
            for ordinal, (mode, phase, target) in enumerate(EXPECTED_OBJECT_ORDER)
        )
        if len(records) != EXPECTED_RETAINED_OBJECTS or observed != expected:
            raise RuntimeError("CAS_STORE session is not exact complete 12-object order")
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in (
            "_lock_fd",
            "_state_fd",
            "_checkpoints_fd",
            "_staging_fd",
            "_objects_fd",
            "_root_fd",
        ):
            descriptor = getattr(self, name, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                finally:
                    setattr(self, name, -1)

    def __enter__(self) -> "CasStoreV1":
        if self._closed:
            raise RuntimeError("CAS store is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


class ScmRightsRetentionSinkV1:
    """Runtime sink client with fixed response parsing and Ed25519 verification."""

    __slots__ = (
        "_connection",
        "_expected_server_uid",
        "_expected_public_key",
        "_expected_signer_key_id",
        "_expected_authority_profile_id",
        "_expected_signer_registry_id",
        "_lock",
        "_active",
    )

    def __init__(
        self,
        *,
        connection: socket.socket,
        expected_server_uid: int,
        expected_public_key: bytes,
        expected_signer_key_id: str,
        expected_authority_profile_id: str,
        expected_signer_registry_id: str,
    ) -> None:
        _require_unix_seqpacket(connection)
        if type(expected_server_uid) is not int or expected_server_uid < 0:
            raise ValueError("expected CAS_STORE UID must be a nonnegative integer")
        if type(expected_public_key) is not bytes or len(expected_public_key) != 32:
            raise ValueError("expected CAS_STORE public key must be raw 32 bytes")
        peer = _peer_credentials(connection)
        if peer.uid != expected_server_uid:
            raise PermissionError("connected CAS_STORE SO_PEERCRED UID drift")
        self._connection = connection
        self._expected_server_uid = expected_server_uid
        self._expected_public_key = expected_public_key
        self._expected_signer_key_id = _require_role_key_id(
            expected_signer_key_id
        )
        self._expected_authority_profile_id = _require_authority_profile_id(
            expected_authority_profile_id
        )
        self._expected_signer_registry_id = _require_signer_registry_id(
            expected_signer_registry_id
        )
        self._lock = threading.Lock()
        self._active = False

    def ingest_open_fd(
        self,
        metadata: Mapping[str, object],
        fd: int,
        sha256: str,
        size_bytes: int,
    ) -> object:
        if type(metadata) is not dict:
            raise TypeError("SCM retention metadata must be one exact dict")
        request = StoreRequestV1.from_mapping(dict(metadata))
        if (
            sha256 != request.auditor_reported_sha256
            or size_bytes != request.auditor_reported_size_bytes
            or type(sha256) is not str
            or type(size_bytes) is not int
        ):
            raise ValueError("runtime sink arguments/request metadata linkage drift")
        with self._lock:
            if self._active:
                raise RuntimeError("SCM retention sink reentrancy drift")
            self._active = True
            try:
                peer = _peer_credentials(self._connection)
                if peer.uid != self._expected_server_uid:
                    raise PermissionError("CAS_STORE peer UID changed")
                send_store_request_fd_v1(self._connection, request, fd)
                payload, ancillary, flags, address = self._connection.recvmsg(
                    MAXIMUM_RESPONSE_BYTES + 1,
                    socket.CMSG_SPACE(array("i").itemsize * 2),
                )
                received_fds: list[int] = []
                try:
                    descriptor_size = array("i").itemsize
                    for level, kind, data in ancillary:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            if len(data) % descriptor_size:
                                raise ValueError(
                                    "CAS_STORE response descriptor alignment drift"
                                )
                            values = array("i")
                            values.frombytes(data)
                            received_fds.extend(int(item) for item in values)
                    if ancillary:
                        raise ValueError(
                            "CAS_STORE response carried unexpected control records"
                        )
                    if flags & (
                        getattr(socket, "MSG_TRUNC", 0)
                        | getattr(socket, "MSG_CTRUNC", 0)
                    ):
                        raise ValueError("CAS_STORE response was truncated")
                    if address not in {None, "", b""}:
                        raise ValueError(
                            "CAS_STORE response carried an unexpected address"
                        )
                    response = StoreResponseV1.from_canonical_bytes(
                        payload,
                        expected_public_key=self._expected_public_key,
                        expected_signer_key_id=self._expected_signer_key_id,
                        expected_authority_profile_id=(
                            self._expected_authority_profile_id
                        ),
                        expected_signer_registry_id=(
                            self._expected_signer_registry_id
                        ),
                    )
                    if response.request_id != request.request_id:
                        raise ValueError("CAS_STORE response/request ID linkage drift")
                    return response.acknowledgement.to_runtime_ack_v1()
                finally:
                    for descriptor in received_fds:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
            finally:
                self._active = False


__all__ = [
    "SCHEMA_VERSION",
    "STORE_REQUEST_SCHEMA_VERSION",
    "STORE_RECEIPT_BODY_SCHEMA_VERSION",
    "SIGNED_STORE_RECEIPT_SCHEMA_VERSION",
    "STORE_CHECKPOINT_SCHEMA_VERSION",
    "STORE_HEAD_SCHEMA_VERSION",
    "STORE_RESPONSE_SCHEMA_VERSION",
    "CAS_STORE_PURPOSE",
    "CAS_STORE_RECEIPT_PURPOSE",
    "CAS_STORE_CHECKPOINT_PURPOSE",
    "CAS_STORE_HEAD_PURPOSE",
    "CAS_STORE_RESPONSE_PURPOSE",
    "CAS_STORE_SIGNATURE_DOMAIN",
    "AUTHORITY_CLAIM_BOUNDARY",
    "AUDITOR_SOURCE_SHA256_PIN",
    "QV2_SOURCE_SHA256_PIN",
    "RUNTIME_SOURCE_SHA256_PIN",
    "AUTHORITY_SOURCE_SHA256_PIN",
    "RETENTION_ACK_SCHEMA_VERSION",
    "EXPECTED_RETAINED_OBJECTS",
    "EXPECTED_CLEANUP_CALLS",
    "MODE_ORDER",
    "INVARIANT_BASENAME",
    "STATISTICS_BASENAMES",
    "EXPECTED_OBJECT_ORDER",
    "INVARIANCE_WORKER_ARTIFACT_BYTES_BOUND",
    "MAXIMUM_STATISTICS_ARTIFACT_BYTES",
    "KEY_EPOCH",
    "MAXIMUM_REQUEST_BYTES",
    "MAXIMUM_RESPONSE_BYTES",
    "GENESIS_CHECKPOINT_SHA256",
    "PeerCredentialsV1",
    "StoreRequestV1",
    "StoreReceiptBodyV1",
    "SignedStoreReceiptV1",
    "StoreCheckpointV1",
    "StoreHeadV1",
    "RetentionAcknowledgementV1",
    "StoreResponseV1",
    "ReceivedStoreRequestV1",
    "Ed25519CasStoreSignerV1",
    "CasStoreV1",
    "ScmRightsRetentionSinkV1",
    "recv_store_request_v1",
    "send_store_request_fd_v1",
    "verify_signed_store_receipt_v1",
]

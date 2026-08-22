"""Defensive, mechanics-only logical time for prospective Phase-2B Step 5.

This module freezes four RFC-8785 byte schemas: an issuance request, an
Ed25519-signed logical-time token, an immutable ledger checkpoint, and an
immutable versioned ledger head.  A local service primitive serializes
issuance under an exclusive file lock and publishes a contiguous append-only
chain.  It deliberately does not start a service, open a socket, inspect a
container, invoke systemd, or use a wall clock.

Successful parsing, signature verification, or local ledger append establishes
only cryptographic, canonical-wire, and local durability mechanics.  It is not
Step-5 authority evidence by itself.  Before any such record can contribute to
authority, an independent verifier still has to establish at least the pinned
service image and runtime closure, the expected service UID, the distinct
``STEP5_LOGICAL_TIME_CONTROL`` control domain, protected private-key custody,
the out-of-band authority-profile and signer-registry pins, freshness of the
Step-5 attempt context, the event-order policy (including terminal closure),
and an externally durable CAS checkpoint that commits the complete ledger.
Local ownership, ``flock``, fsync, and readback do not establish those facts.

The bytes-only replay entry point captures its verification call graph at
import time, so ordinary rebinding of module globals cannot turn signature or
chain checks into permissive callbacks.  This is a mutation-isolation KAT
property, not a claim of resistance to mutation of function defaults, closure
cells, arbitrary process memory, or the loaded code itself.  Same-process code
integrity remains part of the external image/runtime-closure boundary.

This still-uncommitted and unfrozen logical-time candidate now depends only on
the Step-5 authority-profile/2 and signer-registry/2 graph.  Authority/profile
V1 IDs and V1/V2 hybrids are rejected.  The logical-time wire schemas remain
``/1`` because their exact field sets, domains, role-key/1 projection, and
signature purpose did not change; this is a pre-freeze dependency replacement,
not a compatibility promise for any V1 authority graph.

The signing private key is never accepted by a parser, verifier, constructor,
or ledger initializer.  One explicitly service-side issuance method accepts a
``cryptography`` Ed25519 private-key *object* (never seed bytes), uses it for
one call, and does not retain it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import os
import re
import stat
import threading
from typing import Final

from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Importability is preserved; all crypto operations still fail closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - depends on the pinned service image.
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-logical-time/1"
)
LOGICAL_TIME_REQUEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-logical-time-request/1"
)
SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-logical-time-token/1"
)
LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-logical-time-ledger-head/1"
)
LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-logical-time-ledger-checkpoint/1"
)

LOGICAL_TIME_ROLE_V1: Final = "LOGICAL_TIME"
LOGICAL_TIME_CONTROL_DOMAIN_V1: Final = "STEP5_LOGICAL_TIME_CONTROL"
SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1: Final = "SIGN_LOGICAL_TIME_TOKEN"
LOGICAL_TIME_KEY_EPOCH_V1: Final = 1
ED25519_ALGORITHM_V1: Final = "Ed25519"

LEDGER_HEAD_PURPOSE_V1: Final = "STEP5_LOGICAL_TIME_APPEND_ONLY_LEDGER_HEAD"
LEDGER_CHECKPOINT_PURPOSE_V1: Final = (
    "STEP5_LOGICAL_TIME_APPEND_ONLY_LEDGER_CHECKPOINT"
)

AUTHORITY_PROFILE_PREFIX_V2: Final = "phase2b_step5_authority_profile_v2_"
SIGNER_REGISTRY_PREFIX_V2: Final = "phase2b_step5_signer_registry_v2_"
ROLE_KEY_PREFIX_V1: Final = "phase2b_step5_role_key_v1_"
REQUEST_ID_PREFIX_V1: Final = "phase2b_step5_logical_time_request_v1_"
TOKEN_ID_PREFIX_V1: Final = "phase2b_step5_logical_time_token_v1_"
LEDGER_ID_PREFIX_V1: Final = "phase2b_step5_logical_time_ledger_v1_"
LEDGER_HEAD_ID_PREFIX_V1: Final = "phase2b_step5_logical_time_head_v1_"
LEDGER_CHECKPOINT_ID_PREFIX_V1: Final = (
    "phase2b_step5_logical_time_checkpoint_v1_"
)

GENESIS_TOKEN_ID_V1: Final = TOKEN_ID_PREFIX_V1 + ("0" * 64)
GENESIS_CHECKPOINT_ID_V1: Final = LEDGER_CHECKPOINT_ID_PREFIX_V1 + ("0" * 64)

DEPENDENCY_REPLACEMENT_V1: Final = (
    "UNCOMMITTED_UNFROZEN_CANDIDATE_REPLACES_AUTHORITY_PROFILE_V1_AND_"
    "SIGNER_REGISTRY_V1_WITH_AUTHORITY_PROFILE_V2_AND_SIGNER_REGISTRY_V2_"
    "ONLY_LOGICAL_TIME_SCHEMA_V1_FIELD_SHAPE_DOMAINS_ROLE_KEY_AND_PURPOSE_"
    "UNCHANGED_NO_V1_OR_HYBRID_COMPATIBILITY"
)

AUTHORITY_CLAIM_BOUNDARY_V1: Final = (
    "LOGICAL_TIME_CANONICAL_SIGNATURE_AND_LOCAL_APPEND_ONLY_MECHANICS_ONLY_"
    "NOT_AUTHORITY_NOT_TRUSTED_UTC_NOT_EXTERNAL_NOTARIZATION_NOT_RUNTIME_"
    "ATTESTATION_NOT_EVENT_ORDER_POLICY_NOT_TERMINAL_CLOSURE_NOT_CAS_CHECKPOINT"
)
REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1: Final = (
    "OUT_OF_BAND_AUTHORITY_PROFILE_PIN_VERIFIED",
    "OUT_OF_BAND_SIGNER_REGISTRY_PIN_AND_LOGICAL_TIME_ROLE_POLICY_VERIFIED",
    "FRESH_STEP5_ATTEMPT_CONTEXT_VERIFIED",
    "PINNED_LOGICAL_TIME_SERVICE_IMAGE_AND_RUNTIME_CLOSURE_VERIFIED",
    "EXPECTED_DISTINCT_SERVICE_UID_VERIFIED",
    "STEP5_LOGICAL_TIME_CONTROL_DOMAIN_VERIFIED",
    "PRIVATE_KEY_CUSTODY_VERIFIED",
    "FROZEN_EVENT_ORDER_AND_TERMINAL_POLICY_VERIFIED",
    "EXTERNAL_DURABLE_CAS_CHECKPOINT_OF_COMPLETE_LEDGER_VERIFIED",
)

MAXIMUM_REQUEST_BYTES: Final = 64 * 1024
MAXIMUM_TOKEN_BYTES: Final = 128 * 1024
MAXIMUM_HEAD_BYTES: Final = 64 * 1024
MAXIMUM_CHECKPOINT_BYTES: Final = 256 * 1024
_MAXIMUM_SAFE_JSON_INTEGER: Final = (1 << 53) - 1
_MAXIMUM_IDENTIFIER_BYTES: Final = 512
_MAXIMUM_PURPOSE_BYTES: Final = 128

_LOWER_HEX_32_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_LOWER_HEX_64_PATTERN: Final = re.compile(r"[0-9a-f]{128}")
_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_IDENTIFIER_PATTERN: Final = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}"
)
_PURPOSE_PATTERN: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}")
_VERSIONED_FILENAME_PATTERN: Final = re.compile(r"[0-9]{20}\.json")

_ROLE_KEY_ID_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/ROLE_KEY/V1\x00"
_REQUEST_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/REQUEST_ID/V1\x00"
)
_TOKEN_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/SIGN/LOGICAL_TIME_TOKEN/V1\x00"
)
_TOKEN_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/TOKEN_ID/V1\x00"
)
_LEDGER_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_ID/V1\x00"
)
_LEDGER_HEAD_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_HEAD_ID/V1\x00"
)
_LEDGER_CHECKPOINT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_CHECKPOINT_ID/V1\x00"
)

_REQUEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "request_nonce",
        "signature_purpose",
        "purpose",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "prior_token_id",
        "next_sequence",
        "request_id",
    }
)
_TOKEN_KEYS: Final = frozenset(
    {
        "schema_version",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "request_id",
        "request_nonce",
        "signature_purpose",
        "purpose",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "prior_token_id",
        "sequence",
        "algorithm",
        "signature_hex",
        "token_id",
    }
)
_HEAD_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "ledger_id",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "checkpoint_count",
        "next_sequence",
        "last_token_id",
        "last_checkpoint_id",
        "head_id",
    }
)
_CHECKPOINT_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "ledger_id",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "checkpoint_index",
        "sequence",
        "previous_head_id",
        "previous_checkpoint_id",
        "request_id",
        "request_nonce",
        "token_id",
        "signed_token_sha256",
        "signed_token",
        "checkpoint_id",
    }
)


FAIL_INPUT_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_SCHEMA"
FAIL_BINDING_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_BINDING"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_CONTENT_ID"
FAIL_SIGNATURE_BACKEND_V1: Final = (
    "FAIL_STEP5_LOGICAL_TIME_V1_SIGNATURE_BACKEND"
)
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_SIGNATURE"
FAIL_LEDGER_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_LEDGER"
FAIL_FILESYSTEM_V1: Final = "FAIL_STEP5_LOGICAL_TIME_V1_FILESYSTEM"


class Phase2BStep5LogicalTimeErrorV1(ValueError):
    """Stable fail-closed error for Step-5 logical-time mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> None:
    raise Phase2BStep5LogicalTimeErrorV1(code, detail)


def _canonical(value: object) -> bytes:
    try:
        return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5LogicalTimeErrorV1(
            FAIL_CANONICAL_V1, str(exc)
        ) from exc


def _restore_integral_numbers(value: object) -> object:
    if type(value) is float:
        if not value.is_integer() or not abs(value) <= _MAXIMUM_SAFE_JSON_INTEGER:
            _fail(FAIL_SCHEMA_V1, "logical-time schemas do not admit fractions")
        return int(value)
    if type(value) is list:
        return [_restore_integral_numbers(item) for item in value]
    if type(value) is dict:
        return {
            key: _restore_integral_numbers(item) for key, item in value.items()
        }
    return value


def _decode(payload: bytes, maximum_bytes: int, label: str) -> dict[str, object]:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, f"{label} must be exact bytes")
    if not 1 <= len(payload) <= maximum_bytes or payload.endswith(b"\n"):
        _fail(FAIL_INPUT_V1, f"{label} is outside its exact byte boundary")
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
    except (RFC8785Error, TypeError, ValueError, RecursionError) as exc:
        raise Phase2BStep5LogicalTimeErrorV1(
            FAIL_CANONICAL_V1, f"{label}: {exc}"
        ) from exc
    restored = _restore_integral_numbers(decoded)
    if type(restored) is not dict or _canonical(restored) != payload:
        _fail(FAIL_CANONICAL_V1, f"{label} canonical readback differs")
    return restored


def _exact_object(
    value: object, keys: frozenset[str], label: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        _fail(FAIL_SCHEMA_V1, f"{label} exact field set differs")
    return value


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be one bounded exact identifier")
    if len(value.encode("ascii")) > _MAXIMUM_IDENTIFIER_BYTES:
        _fail(FAIL_SCHEMA_V1, f"{label} exceeds its ASCII byte bound")
    return value


def _prefixed_id(value: object, prefix: str, label: str) -> str:
    result = _identifier(value, label)
    suffix = result[len(prefix) :] if result.startswith(prefix) else ""
    if not result.startswith(prefix) or _LOWER_HEX_32_PATTERN.fullmatch(suffix) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} prefix or digest differs")
    return result


def _sha256_address(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be sha256:<lowerhex64>")
    return value


def _nonce(value: object, label: str) -> str:
    if type(value) is not str or _LOWER_HEX_32_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lowercase 32-byte hex")
    return value


def _purpose(value: object, label: str) -> str:
    if type(value) is not str or _PURPOSE_PATTERN.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be one frozen uppercase token")
    if len(value.encode("ascii")) > _MAXIMUM_PURPOSE_BYTES:
        _fail(FAIL_SCHEMA_V1, f"{label} exceeds its ASCII byte bound")
    return value


def _wire_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(FAIL_SCHEMA_V1, f"{label} exact integer differs")
    return value


def _sha256_bytes(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "SHA-256 input must be exact bytes")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _content_id(prefix: str, domain: bytes, body: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(body)).hexdigest()


def derive_step5_logical_time_role_key_id_v1(
    logical_time_public_key: bytes,
    *,
    key_epoch: int = LOGICAL_TIME_KEY_EPOCH_V1,
) -> str:
    """Recompute the exact authority-registry LOGICAL_TIME role-key ID.

    This is the Step-5 authority role-key content address, not
    ``sha256(public_key)``.  Its policy projection exactly binds role, control
    domain, authorized signature purpose, algorithm, epoch, and raw public key.
    """

    if type(logical_time_public_key) is not bytes or len(logical_time_public_key) != 32:
        _fail(FAIL_INPUT_V1, "logical-time public key must be exact raw 32 bytes")
    epoch = _wire_integer(key_epoch, "logical-time key epoch", minimum=1)
    if epoch != LOGICAL_TIME_KEY_EPOCH_V1:
        _fail(FAIL_BINDING_V1, "role-key/1 requires logical-time key epoch 1")
    role_key_subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": LOGICAL_TIME_ROLE_V1,
        "control_domain": LOGICAL_TIME_CONTROL_DOMAIN_V1,
        "authorized_signature_purposes": [
            SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1
        ],
        "algorithm": ED25519_ALGORITHM_V1,
        "key_epoch": epoch,
        "public_key_hex": logical_time_public_key.hex(),
    }
    return _content_id(
        ROLE_KEY_PREFIX_V1, _ROLE_KEY_ID_DOMAIN, role_key_subject
    )


def _validate_common_bindings(
    *,
    authority_profile_id: object,
    signer_registry_id: object,
    logical_time_key_id: object,
    logical_time_key_epoch: object,
    attempt_context_id: object,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_identifier=_identifier,
    _fail_exact=_fail,
    _authority_profile_prefix: str = AUTHORITY_PROFILE_PREFIX_V2,
    _signer_registry_prefix: str = SIGNER_REGISTRY_PREFIX_V2,
    _role_key_prefix: str = ROLE_KEY_PREFIX_V1,
    _required_key_epoch: int = LOGICAL_TIME_KEY_EPOCH_V1,
) -> tuple[str, str, str, int, str]:
    profile = _validate_prefixed_id(
        authority_profile_id,
        _authority_profile_prefix,
        "authority profile ID",
    )
    registry = _validate_prefixed_id(
        signer_registry_id,
        _signer_registry_prefix,
        "signer registry ID",
    )
    key_id = _validate_prefixed_id(
        logical_time_key_id, _role_key_prefix, "logical-time role-key ID"
    )
    epoch = _validate_integer(
        logical_time_key_epoch, "logical-time key epoch", minimum=1
    )
    if epoch != _required_key_epoch:
        _fail_exact(
            FAIL_BINDING_V1, "logical-time schema/1 requires key epoch 1"
        )
    attempt = _validate_identifier(
        attempt_context_id, "fresh Step-5 attempt context ID"
    )
    return profile, registry, key_id, epoch, attempt


@dataclass(frozen=True, slots=True)
class LogicalTimeRequestV1:
    schema_version: str
    authority_profile_id: str
    signer_registry_id: str
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    request_nonce: str
    signature_purpose: str
    purpose: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    prior_token_id: str
    next_sequence: int
    request_id: str

    def __post_init__(
        self,
        _validate_common=_validate_common_bindings,
        _validate_nonce=_nonce,
        _validate_purpose=_purpose,
        _validate_identifier=_identifier,
        _validate_sha256=_sha256_address,
        _validate_prefixed_id=_prefixed_id,
        _validate_integer=_wire_integer,
        _content_address=_content_id,
        _fail_exact=_fail,
        _schema_version: str = LOGICAL_TIME_REQUEST_SCHEMA_VERSION,
        _signature_purpose: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
        _token_id_prefix: str = TOKEN_ID_PREFIX_V1,
        _request_id_prefix: str = REQUEST_ID_PREFIX_V1,
        _request_id_domain: bytes = _REQUEST_ID_DOMAIN,
    ) -> None:
        if type(self) is not LogicalTimeRequestV1:
            _fail(FAIL_INPUT_V1, "logical-time request must use the exact DTO")
        if self.schema_version != _schema_version:
            _fail_exact(FAIL_SCHEMA_V1, "logical-time request schema differs")
        _validate_common(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            logical_time_key_id=self.logical_time_key_id,
            logical_time_key_epoch=self.logical_time_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        _validate_nonce(self.request_nonce, "request nonce")
        if self.signature_purpose != _signature_purpose:
            _fail_exact(
                FAIL_BINDING_V1,
                "logical-time request signature purpose differs",
            )
        _validate_purpose(self.purpose, "logical-time event purpose")
        _validate_identifier(self.bound_artifact_id, "bound artifact ID")
        _validate_sha256(self.bound_artifact_sha256, "bound artifact SHA-256")
        _validate_prefixed_id(
            self.prior_token_id, _token_id_prefix, "prior token ID"
        )
        _validate_integer(self.next_sequence, "requested next sequence")
        expected = _content_address(
            _request_id_prefix, _request_id_domain, self._id_subject()
        )
        if self.request_id != expected:
            _fail_exact(FAIL_CONTENT_ID_V1, "logical-time request ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "logical_time_key_id": self.logical_time_key_id,
            "logical_time_key_epoch": self.logical_time_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "request_nonce": self.request_nonce,
            "signature_purpose": self.signature_purpose,
            "purpose": self.purpose,
            "bound_artifact_id": self.bound_artifact_id,
            "bound_artifact_sha256": self.bound_artifact_sha256,
            "prior_token_id": self.prior_token_id,
            "next_sequence": self.next_sequence,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "request_id": self.request_id}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_mapping())

    @classmethod
    def from_canonical_bytes(
        cls,
        payload: bytes,
        _decode_exact=_decode,
        _exact_mapping=_exact_object,
        _maximum_bytes: int = MAXIMUM_REQUEST_BYTES,
        _request_keys: frozenset[str] = _REQUEST_KEYS,
        _fail_exact=_fail,
    ) -> "LogicalTimeRequestV1":
        body = _exact_mapping(
            _decode_exact(payload, _maximum_bytes, "logical-time request"),
            _request_keys,
            "logical-time request",
        )
        value = cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            logical_time_key_id=body["logical_time_key_id"],  # type: ignore[arg-type]
            logical_time_key_epoch=body["logical_time_key_epoch"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            request_nonce=body["request_nonce"],  # type: ignore[arg-type]
            signature_purpose=body["signature_purpose"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            bound_artifact_id=body["bound_artifact_id"],  # type: ignore[arg-type]
            bound_artifact_sha256=body["bound_artifact_sha256"],  # type: ignore[arg-type]
            prior_token_id=body["prior_token_id"],  # type: ignore[arg-type]
            next_sequence=body["next_sequence"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
        )
        if value.canonical_bytes() != payload:
            _fail_exact(
                FAIL_CANONICAL_V1,
                "logical-time request DTO readback differs",
            )
        return value


def build_logical_time_request_v1(
    *,
    authority_profile_id: str,
    signer_registry_id: str,
    logical_time_key_id: str,
    logical_time_key_epoch: int,
    attempt_context_id: str,
    request_nonce: str,
    purpose: str,
    bound_artifact_id: str,
    bound_artifact_sha256: str,
    prior_token_id: str,
    next_sequence: int,
) -> bytes:
    """Build and reparse one exact request/1 payload; no signing occurs."""

    subject = {
        "schema_version": LOGICAL_TIME_REQUEST_SCHEMA_VERSION,
        "authority_profile_id": authority_profile_id,
        "signer_registry_id": signer_registry_id,
        "logical_time_key_id": logical_time_key_id,
        "logical_time_key_epoch": logical_time_key_epoch,
        "attempt_context_id": attempt_context_id,
        "request_nonce": request_nonce,
        "signature_purpose": SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
        "purpose": purpose,
        "bound_artifact_id": bound_artifact_id,
        "bound_artifact_sha256": bound_artifact_sha256,
        "prior_token_id": prior_token_id,
        "next_sequence": next_sequence,
    }
    request_id = _content_id(
        REQUEST_ID_PREFIX_V1, _REQUEST_ID_DOMAIN, subject
    )
    payload = _canonical({**subject, "request_id": request_id})
    return LogicalTimeRequestV1.from_canonical_bytes(payload).canonical_bytes()


@dataclass(frozen=True, slots=True)
class SignedLogicalTimeTokenV1:
    schema_version: str
    authority_profile_id: str
    signer_registry_id: str
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    request_id: str
    request_nonce: str
    signature_purpose: str
    purpose: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    prior_token_id: str
    sequence: int
    algorithm: str
    signature_hex: str
    token_id: str

    def __post_init__(
        self,
        _validate_common=_validate_common_bindings,
        _validate_prefixed_id=_prefixed_id,
        _validate_nonce=_nonce,
        _validate_purpose=_purpose,
        _validate_identifier=_identifier,
        _validate_sha256=_sha256_address,
        _validate_integer=_wire_integer,
        _content_address=_content_id,
        _fail_exact=_fail,
        _schema_version: str = SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION,
        _signature_purpose: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
        _algorithm: str = ED25519_ALGORITHM_V1,
        _request_id_prefix: str = REQUEST_ID_PREFIX_V1,
        _token_id_prefix: str = TOKEN_ID_PREFIX_V1,
        _token_id_domain: bytes = _TOKEN_ID_DOMAIN,
    ) -> None:
        if type(self) is not SignedLogicalTimeTokenV1:
            _fail(FAIL_INPUT_V1, "logical-time token must use the exact DTO")
        if self.schema_version != _schema_version:
            _fail_exact(FAIL_SCHEMA_V1, "signed logical-time token schema differs")
        _validate_common(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            logical_time_key_id=self.logical_time_key_id,
            logical_time_key_epoch=self.logical_time_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        _validate_prefixed_id(
            self.request_id, _request_id_prefix, "token request ID"
        )
        _validate_nonce(self.request_nonce, "token request nonce")
        if self.signature_purpose != _signature_purpose:
            _fail_exact(
                FAIL_BINDING_V1, "logical-time token signature purpose differs"
            )
        _validate_purpose(self.purpose, "logical-time token event purpose")
        _validate_identifier(self.bound_artifact_id, "token bound artifact ID")
        _validate_sha256(
            self.bound_artifact_sha256, "token bound artifact SHA-256"
        )
        _validate_prefixed_id(
            self.prior_token_id, _token_id_prefix, "token prior ID"
        )
        _validate_integer(self.sequence, "logical-time token sequence")
        if self.algorithm != _algorithm:
            _fail_exact(FAIL_SCHEMA_V1, "logical-time token algorithm differs")
        if (
            type(self.signature_hex) is not str
            or _LOWER_HEX_64_PATTERN.fullmatch(self.signature_hex) is None
        ):
            _fail_exact(
                FAIL_SCHEMA_V1,
                "logical-time token signature encoding differs",
            )
        expected = _content_address(
            _token_id_prefix, _token_id_domain, self._id_subject()
        )
        if self.token_id != expected:
            _fail_exact(FAIL_CONTENT_ID_V1, "logical-time token ID differs")

    def _unsigned_body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "logical_time_key_id": self.logical_time_key_id,
            "logical_time_key_epoch": self.logical_time_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "request_id": self.request_id,
            "request_nonce": self.request_nonce,
            "signature_purpose": self.signature_purpose,
            "purpose": self.purpose,
            "bound_artifact_id": self.bound_artifact_id,
            "bound_artifact_sha256": self.bound_artifact_sha256,
            "prior_token_id": self.prior_token_id,
            "sequence": self.sequence,
            "algorithm": self.algorithm,
        }

    def _id_subject(self) -> dict[str, object]:
        return {**self._unsigned_body(), "signature_hex": self.signature_hex}

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "token_id": self.token_id}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_mapping())

    def signature_preimage(self) -> bytes:
        wrapper = {
            "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
            "artifact_kind": "logical_time",
            "role": LOGICAL_TIME_ROLE_V1,
            "purpose": SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
            "key_epoch": self.logical_time_key_epoch,
            "signer_key_id": self.logical_time_key_id,
            "body": self._unsigned_body(),
        }
        return _TOKEN_SIGNATURE_DOMAIN + _canonical(wrapper)

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "SignedLogicalTimeTokenV1":
        body = _exact_object(
            _decode(payload, MAXIMUM_TOKEN_BYTES, "signed logical-time token"),
            _TOKEN_KEYS,
            "signed logical-time token",
        )
        value = _token_from_mapping(body)
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL_V1, "logical-time token DTO readback differs")
        return value


def _token_from_mapping(
    value: object,
    _token_type=SignedLogicalTimeTokenV1,
    _exact_mapping=_exact_object,
    _token_keys: frozenset[str] = _TOKEN_KEYS,
) -> SignedLogicalTimeTokenV1:
    body = _exact_mapping(value, _token_keys, "signed logical-time token")
    return _token_type(
        schema_version=body["schema_version"],  # type: ignore[arg-type]
        authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
        signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
        logical_time_key_id=body["logical_time_key_id"],  # type: ignore[arg-type]
        logical_time_key_epoch=body["logical_time_key_epoch"],  # type: ignore[arg-type]
        attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
        request_id=body["request_id"],  # type: ignore[arg-type]
        request_nonce=body["request_nonce"],  # type: ignore[arg-type]
        signature_purpose=body["signature_purpose"],  # type: ignore[arg-type]
        purpose=body["purpose"],  # type: ignore[arg-type]
        bound_artifact_id=body["bound_artifact_id"],  # type: ignore[arg-type]
        bound_artifact_sha256=body["bound_artifact_sha256"],  # type: ignore[arg-type]
        prior_token_id=body["prior_token_id"],  # type: ignore[arg-type]
        sequence=body["sequence"],  # type: ignore[arg-type]
        algorithm=body["algorithm"],  # type: ignore[arg-type]
        signature_hex=body["signature_hex"],  # type: ignore[arg-type]
        token_id=body["token_id"],  # type: ignore[arg-type]
    )


def _request_from_token(
    token: SignedLogicalTimeTokenV1,
    _request_type: type[LogicalTimeRequestV1] = LogicalTimeRequestV1,
    _request_schema_version: str = LOGICAL_TIME_REQUEST_SCHEMA_VERSION,
) -> LogicalTimeRequestV1:
    return _request_type(
        schema_version=_request_schema_version,
        authority_profile_id=token.authority_profile_id,
        signer_registry_id=token.signer_registry_id,
        logical_time_key_id=token.logical_time_key_id,
        logical_time_key_epoch=token.logical_time_key_epoch,
        attempt_context_id=token.attempt_context_id,
        request_nonce=token.request_nonce,
        signature_purpose=token.signature_purpose,
        purpose=token.purpose,
        bound_artifact_id=token.bound_artifact_id,
        bound_artifact_sha256=token.bound_artifact_sha256,
        prior_token_id=token.prior_token_id,
        next_sequence=token.sequence,
        request_id=token.request_id,
    )


def _require_public_key_binding(
    logical_time_public_key: bytes,
    logical_time_key_id: str,
    logical_time_key_epoch: int,
    _derive_role_key_id=derive_step5_logical_time_role_key_id_v1,
    _fail_exact=_fail,
) -> None:
    expected = _derive_role_key_id(
        logical_time_public_key, key_epoch=logical_time_key_epoch
    )
    if logical_time_key_id != expected:
        _fail_exact(
            FAIL_BINDING_V1,
            "logical-time key ID is not the frozen registry role-key content ID",
        )


def _verify_token_signature(
    token: SignedLogicalTimeTokenV1,
    logical_time_public_key: bytes,
    _require_key_binding=_require_public_key_binding,
    _public_key_type=_Ed25519PublicKey,
    _invalid_signature_type=_InvalidSignature,
    _fail_exact=_fail,
) -> None:
    _require_key_binding(
        logical_time_public_key,
        token.logical_time_key_id,
        token.logical_time_key_epoch,
    )
    if _public_key_type is None or _invalid_signature_type is None:
        _fail_exact(
            FAIL_SIGNATURE_BACKEND_V1,
            "cryptography Ed25519 verification backend is unavailable",
        )
    try:
        public_key = _public_key_type.from_public_bytes(logical_time_public_key)
        public_key.verify(
            bytes.fromhex(token.signature_hex), token.signature_preimage()
        )
    except _invalid_signature_type as exc:
        raise Phase2BStep5LogicalTimeErrorV1(
            FAIL_SIGNATURE_V1, "logical-time Ed25519 signature rejected"
        ) from exc
    except ValueError as exc:
        raise Phase2BStep5LogicalTimeErrorV1(
            FAIL_SIGNATURE_V1, "logical-time Ed25519 public key rejected"
        ) from exc


def verify_signed_logical_time_token_v1(
    token_payload: bytes,
    *,
    request_payload: bytes,
    logical_time_public_key: bytes,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
    expected_logical_time_key_id: str,
    expected_logical_time_key_epoch: int,
    expected_attempt_context_id: str,
) -> SignedLogicalTimeTokenV1:
    """Verify one exact request/token pair with ``cryptography`` directly.

    The caller remains responsible for deciding whether the request's event
    purpose and bound artifact are authorized by the frozen Step-5 graph.
    """

    request = LogicalTimeRequestV1.from_canonical_bytes(request_payload)
    token = SignedLogicalTimeTokenV1.from_canonical_bytes(token_payload)
    expected_common = _validate_common_bindings(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        logical_time_key_id=expected_logical_time_key_id,
        logical_time_key_epoch=expected_logical_time_key_epoch,
        attempt_context_id=expected_attempt_context_id,
    )
    observed_common = (
        request.authority_profile_id,
        request.signer_registry_id,
        request.logical_time_key_id,
        request.logical_time_key_epoch,
        request.attempt_context_id,
    )
    if observed_common != expected_common:
        _fail(FAIL_BINDING_V1, "logical-time request differs from expected pins")
    token_projection = _request_from_token(token)
    if token_projection != request:
        _fail(FAIL_BINDING_V1, "logical-time token does not exact-project request")
    _require_public_key_binding(
        logical_time_public_key,
        expected_logical_time_key_id,
        expected_logical_time_key_epoch,
    )
    _verify_token_signature(token, logical_time_public_key)
    return token


@dataclass(frozen=True, slots=True)
class LogicalTimeLedgerHeadV1:
    schema_version: str
    purpose: str
    ledger_id: str
    authority_profile_id: str
    signer_registry_id: str
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    checkpoint_count: int
    next_sequence: int
    last_token_id: str
    last_checkpoint_id: str
    head_id: str

    def __post_init__(
        self,
        _validate_prefixed_id=_prefixed_id,
        _validate_common=_validate_common_bindings,
        _validate_integer=_wire_integer,
        _content_address=_content_id,
        _fail_exact=_fail,
        _schema_version: str = LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION,
        _head_purpose: str = LEDGER_HEAD_PURPOSE_V1,
        _ledger_id_prefix: str = LEDGER_ID_PREFIX_V1,
        _token_id_prefix: str = TOKEN_ID_PREFIX_V1,
        _checkpoint_id_prefix: str = LEDGER_CHECKPOINT_ID_PREFIX_V1,
        _head_id_prefix: str = LEDGER_HEAD_ID_PREFIX_V1,
        _head_id_domain: bytes = _LEDGER_HEAD_ID_DOMAIN,
        _genesis_token_id: str = GENESIS_TOKEN_ID_V1,
        _genesis_checkpoint_id: str = GENESIS_CHECKPOINT_ID_V1,
    ) -> None:
        if type(self) is not LogicalTimeLedgerHeadV1:
            _fail(FAIL_INPUT_V1, "logical-time head must use the exact DTO")
        if (
            self.schema_version != _schema_version
            or self.purpose != _head_purpose
        ):
            _fail_exact(
                FAIL_SCHEMA_V1, "logical-time head schema or purpose differs"
            )
        _validate_prefixed_id(
            self.ledger_id, _ledger_id_prefix, "logical-time ledger ID"
        )
        _validate_common(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            logical_time_key_id=self.logical_time_key_id,
            logical_time_key_epoch=self.logical_time_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        count = _validate_integer(self.checkpoint_count, "head checkpoint count")
        next_sequence = _validate_integer(
            self.next_sequence, "head next sequence"
        )
        if count != next_sequence:
            _fail_exact(FAIL_LEDGER_V1, "head count and next sequence differ")
        _validate_prefixed_id(
            self.last_token_id, _token_id_prefix, "head last token ID"
        )
        _validate_prefixed_id(
            self.last_checkpoint_id,
            _checkpoint_id_prefix,
            "head last checkpoint ID",
        )
        if count == 0 and (
            self.last_token_id != _genesis_token_id
            or self.last_checkpoint_id != _genesis_checkpoint_id
        ):
            _fail_exact(FAIL_LEDGER_V1, "genesis head sentinel linkage differs")
        expected = _content_address(
            _head_id_prefix, _head_id_domain, self._id_subject()
        )
        if self.head_id != expected:
            _fail_exact(FAIL_CONTENT_ID_V1, "logical-time head ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "ledger_id": self.ledger_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "logical_time_key_id": self.logical_time_key_id,
            "logical_time_key_epoch": self.logical_time_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "checkpoint_count": self.checkpoint_count,
            "next_sequence": self.next_sequence,
            "last_token_id": self.last_token_id,
            "last_checkpoint_id": self.last_checkpoint_id,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "head_id": self.head_id}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_mapping())

    @classmethod
    def from_canonical_bytes(
        cls,
        payload: bytes,
        _decode_exact=_decode,
        _exact_mapping=_exact_object,
        _maximum_bytes: int = MAXIMUM_HEAD_BYTES,
        _head_keys: frozenset[str] = _HEAD_KEYS,
        _fail_exact=_fail,
    ) -> "LogicalTimeLedgerHeadV1":
        body = _exact_mapping(
            _decode_exact(payload, _maximum_bytes, "logical-time ledger head"),
            _head_keys,
            "logical-time ledger head",
        )
        value = cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            ledger_id=body["ledger_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            logical_time_key_id=body["logical_time_key_id"],  # type: ignore[arg-type]
            logical_time_key_epoch=body["logical_time_key_epoch"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            checkpoint_count=body["checkpoint_count"],  # type: ignore[arg-type]
            next_sequence=body["next_sequence"],  # type: ignore[arg-type]
            last_token_id=body["last_token_id"],  # type: ignore[arg-type]
            last_checkpoint_id=body["last_checkpoint_id"],  # type: ignore[arg-type]
            head_id=body["head_id"],  # type: ignore[arg-type]
        )
        if value.canonical_bytes() != payload:
            _fail_exact(
                FAIL_CANONICAL_V1, "logical-time head DTO readback differs"
            )
        return value


@dataclass(frozen=True, slots=True)
class LogicalTimeLedgerCheckpointV1:
    schema_version: str
    purpose: str
    ledger_id: str
    authority_profile_id: str
    signer_registry_id: str
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    checkpoint_index: int
    sequence: int
    previous_head_id: str
    previous_checkpoint_id: str
    request_id: str
    request_nonce: str
    token_id: str
    signed_token_sha256: str
    signed_token: SignedLogicalTimeTokenV1
    checkpoint_id: str

    def __post_init__(
        self,
        _validate_prefixed_id=_prefixed_id,
        _validate_common=_validate_common_bindings,
        _validate_integer=_wire_integer,
        _validate_nonce=_nonce,
        _validate_sha256=_sha256_address,
        _token_type=SignedLogicalTimeTokenV1,
        _validate_token=SignedLogicalTimeTokenV1.__post_init__,
        _sha256_exact=_sha256_bytes,
        _content_address=_content_id,
        _fail_exact=_fail,
        _schema_version: str = LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION,
        _checkpoint_purpose: str = LEDGER_CHECKPOINT_PURPOSE_V1,
        _ledger_id_prefix: str = LEDGER_ID_PREFIX_V1,
        _head_id_prefix: str = LEDGER_HEAD_ID_PREFIX_V1,
        _checkpoint_id_prefix: str = LEDGER_CHECKPOINT_ID_PREFIX_V1,
        _request_id_prefix: str = REQUEST_ID_PREFIX_V1,
        _token_id_prefix: str = TOKEN_ID_PREFIX_V1,
        _checkpoint_id_domain: bytes = _LEDGER_CHECKPOINT_ID_DOMAIN,
    ) -> None:
        if type(self) is not LogicalTimeLedgerCheckpointV1:
            _fail(FAIL_INPUT_V1, "logical-time checkpoint must use the exact DTO")
        if (
            self.schema_version != _schema_version
            or self.purpose != _checkpoint_purpose
        ):
            _fail_exact(
                FAIL_SCHEMA_V1,
                "logical-time checkpoint schema or purpose differs",
            )
        _validate_prefixed_id(
            self.ledger_id, _ledger_id_prefix, "checkpoint ledger ID"
        )
        _validate_common(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            logical_time_key_id=self.logical_time_key_id,
            logical_time_key_epoch=self.logical_time_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        index = _validate_integer(self.checkpoint_index, "checkpoint index")
        sequence = _validate_integer(self.sequence, "checkpoint sequence")
        if index != sequence:
            _fail_exact(
                FAIL_LEDGER_V1,
                "checkpoint index and token sequence differ",
            )
        _validate_prefixed_id(
            self.previous_head_id,
            _head_id_prefix,
            "checkpoint previous head ID",
        )
        _validate_prefixed_id(
            self.previous_checkpoint_id,
            _checkpoint_id_prefix,
            "checkpoint previous checkpoint ID",
        )
        _validate_prefixed_id(
            self.request_id, _request_id_prefix, "checkpoint request ID"
        )
        _validate_nonce(self.request_nonce, "checkpoint request nonce")
        _validate_prefixed_id(
            self.token_id, _token_id_prefix, "checkpoint token ID"
        )
        _validate_sha256(
            self.signed_token_sha256, "checkpoint token SHA-256"
        )
        if type(self.signed_token) is not _token_type:
            _fail_exact(
                FAIL_INPUT_V1, "checkpoint signed token must use exact DTO"
            )
        _validate_token(self.signed_token)
        if (
            self.signed_token.token_id != self.token_id
            or self.signed_token.request_id != self.request_id
            or self.signed_token.request_nonce != self.request_nonce
            or self.signed_token.sequence != self.sequence
            or _sha256_exact(self.signed_token.canonical_bytes())
            != self.signed_token_sha256
        ):
            _fail_exact(
                FAIL_BINDING_V1, "checkpoint signed-token linkage differs"
            )
        expected = _content_address(
            _checkpoint_id_prefix,
            _checkpoint_id_domain,
            self._id_subject(),
        )
        if self.checkpoint_id != expected:
            _fail_exact(FAIL_CONTENT_ID_V1, "logical-time checkpoint ID differs")

    def _id_subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "ledger_id": self.ledger_id,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "logical_time_key_id": self.logical_time_key_id,
            "logical_time_key_epoch": self.logical_time_key_epoch,
            "attempt_context_id": self.attempt_context_id,
            "checkpoint_index": self.checkpoint_index,
            "sequence": self.sequence,
            "previous_head_id": self.previous_head_id,
            "previous_checkpoint_id": self.previous_checkpoint_id,
            "request_id": self.request_id,
            "request_nonce": self.request_nonce,
            "token_id": self.token_id,
            "signed_token_sha256": self.signed_token_sha256,
            "signed_token": self.signed_token.to_mapping(),
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._id_subject(), "checkpoint_id": self.checkpoint_id}

    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_mapping())

    @classmethod
    def from_canonical_bytes(
        cls,
        payload: bytes,
        _decode_exact=_decode,
        _exact_mapping=_exact_object,
        _token_projection=_token_from_mapping,
        _maximum_bytes: int = MAXIMUM_CHECKPOINT_BYTES,
        _checkpoint_keys: frozenset[str] = _CHECKPOINT_KEYS,
        _token_keys: frozenset[str] = _TOKEN_KEYS,
        _fail_exact=_fail,
    ) -> "LogicalTimeLedgerCheckpointV1":
        body = _exact_mapping(
            _decode_exact(
                payload,
                _maximum_bytes,
                "logical-time ledger checkpoint",
            ),
            _checkpoint_keys,
            "logical-time ledger checkpoint",
        )
        token_mapping = _exact_mapping(
            body["signed_token"], _token_keys, "checkpoint signed token"
        )
        value = cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            ledger_id=body["ledger_id"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            logical_time_key_id=body["logical_time_key_id"],  # type: ignore[arg-type]
            logical_time_key_epoch=body["logical_time_key_epoch"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            checkpoint_index=body["checkpoint_index"],  # type: ignore[arg-type]
            sequence=body["sequence"],  # type: ignore[arg-type]
            previous_head_id=body["previous_head_id"],  # type: ignore[arg-type]
            previous_checkpoint_id=body["previous_checkpoint_id"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
            request_nonce=body["request_nonce"],  # type: ignore[arg-type]
            token_id=body["token_id"],  # type: ignore[arg-type]
            signed_token_sha256=body["signed_token_sha256"],  # type: ignore[arg-type]
            signed_token=_token_projection(token_mapping),
            checkpoint_id=body["checkpoint_id"],  # type: ignore[arg-type]
        )
        if value.canonical_bytes() != payload:
            _fail_exact(
                FAIL_CANONICAL_V1,
                "logical-time checkpoint DTO readback differs",
            )
        return value


def _ledger_id(
    *,
    authority_profile_id: str,
    signer_registry_id: str,
    logical_time_key_id: str,
    logical_time_key_epoch: int,
    attempt_context_id: str,
) -> str:
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-logical-time-ledger-identity/1",
        "authority_profile_id": authority_profile_id,
        "signer_registry_id": signer_registry_id,
        "logical_time_key_id": logical_time_key_id,
        "logical_time_key_epoch": logical_time_key_epoch,
        "attempt_context_id": attempt_context_id,
    }
    return _content_id(LEDGER_ID_PREFIX_V1, _LEDGER_ID_DOMAIN, subject)


def _head(
    *,
    ledger_id: str,
    authority_profile_id: str,
    signer_registry_id: str,
    logical_time_key_id: str,
    logical_time_key_epoch: int,
    attempt_context_id: str,
    checkpoint_count: int,
    last_token_id: str,
    last_checkpoint_id: str,
) -> LogicalTimeLedgerHeadV1:
    subject = {
        "schema_version": LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION,
        "purpose": LEDGER_HEAD_PURPOSE_V1,
        "ledger_id": ledger_id,
        "authority_profile_id": authority_profile_id,
        "signer_registry_id": signer_registry_id,
        "logical_time_key_id": logical_time_key_id,
        "logical_time_key_epoch": logical_time_key_epoch,
        "attempt_context_id": attempt_context_id,
        "checkpoint_count": checkpoint_count,
        "next_sequence": checkpoint_count,
        "last_token_id": last_token_id,
        "last_checkpoint_id": last_checkpoint_id,
    }
    return LogicalTimeLedgerHeadV1(
        **subject,
        head_id=_content_id(
            LEDGER_HEAD_ID_PREFIX_V1, _LEDGER_HEAD_ID_DOMAIN, subject
        ),
    )


def _checkpoint(
    *,
    previous_head: LogicalTimeLedgerHeadV1,
    token: SignedLogicalTimeTokenV1,
) -> LogicalTimeLedgerCheckpointV1:
    subject = {
        "schema_version": LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION,
        "purpose": LEDGER_CHECKPOINT_PURPOSE_V1,
        "ledger_id": previous_head.ledger_id,
        "authority_profile_id": previous_head.authority_profile_id,
        "signer_registry_id": previous_head.signer_registry_id,
        "logical_time_key_id": previous_head.logical_time_key_id,
        "logical_time_key_epoch": previous_head.logical_time_key_epoch,
        "attempt_context_id": previous_head.attempt_context_id,
        "checkpoint_index": token.sequence,
        "sequence": token.sequence,
        "previous_head_id": previous_head.head_id,
        "previous_checkpoint_id": previous_head.last_checkpoint_id,
        "request_id": token.request_id,
        "request_nonce": token.request_nonce,
        "token_id": token.token_id,
        "signed_token_sha256": _sha256_bytes(token.canonical_bytes()),
        "signed_token": token,
    }
    id_subject = {
        **subject,
        "signed_token": token.to_mapping(),
    }
    return LogicalTimeLedgerCheckpointV1(
        **subject,
        checkpoint_id=_content_id(
            LEDGER_CHECKPOINT_ID_PREFIX_V1,
            _LEDGER_CHECKPOINT_ID_DOMAIN,
            id_subject,
        ),
    )


def _common_tuple(value: object) -> tuple[str, str, str, int, str]:
    return (
        object.__getattribute__(value, "authority_profile_id"),
        object.__getattribute__(value, "signer_registry_id"),
        object.__getattribute__(value, "logical_time_key_id"),
        object.__getattribute__(value, "logical_time_key_epoch"),
        object.__getattribute__(value, "attempt_context_id"),
    )


def _verify_transition(
    previous_head: LogicalTimeLedgerHeadV1,
    checkpoint: LogicalTimeLedgerCheckpointV1,
    next_head: LogicalTimeLedgerHeadV1,
    logical_time_public_key: bytes,
    _verify_signature=_verify_token_signature,
    _project_request=_request_from_token,
    _common_projection=_common_tuple,
    _fail_exact=_fail,
) -> None:
    previous_head.__post_init__()
    checkpoint.__post_init__()
    next_head.__post_init__()
    token = checkpoint.signed_token
    _verify_signature(token, logical_time_public_key)
    _project_request(token).__post_init__()
    if not (
        previous_head.ledger_id
        == checkpoint.ledger_id
        == next_head.ledger_id
    ):
        _fail_exact(FAIL_LEDGER_V1, "ledger ID changes across one transition")
    if not (
        _common_projection(previous_head)
        == _common_projection(checkpoint)
        == _common_projection(token)
        == _common_projection(next_head)
    ):
        _fail_exact(
            FAIL_BINDING_V1,
            "ledger authority/registry/attempt/key binding fork",
        )
    expected_index = previous_head.checkpoint_count
    if (
        checkpoint.checkpoint_index != expected_index
        or checkpoint.sequence != previous_head.next_sequence
        or token.sequence != previous_head.next_sequence
        or token.prior_token_id != previous_head.last_token_id
        or checkpoint.previous_head_id != previous_head.head_id
        or checkpoint.previous_checkpoint_id
        != previous_head.last_checkpoint_id
    ):
        _fail_exact(
            FAIL_LEDGER_V1, "checkpoint is not the exact next chain element"
        )
    if (
        next_head.checkpoint_count != expected_index + 1
        or next_head.next_sequence != expected_index + 1
        or next_head.last_token_id != token.token_id
        or next_head.last_checkpoint_id != checkpoint.checkpoint_id
    ):
        _fail_exact(
            FAIL_LEDGER_V1, "next head does not exact-commit the checkpoint"
        )


def _verify_logical_time_ledger_transition_implementation_v1(
    previous_head_payload: bytes,
    checkpoint_payload: bytes,
    next_head_payload: bytes,
    *,
    logical_time_public_key: bytes,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
    expected_logical_time_key_id: str,
    expected_logical_time_key_epoch: int,
    expected_attempt_context_id: str,
    _load_head=LogicalTimeLedgerHeadV1.from_canonical_bytes,
    _load_checkpoint=LogicalTimeLedgerCheckpointV1.from_canonical_bytes,
    _validate_common=_validate_common_bindings,
    _common_projection=_common_tuple,
    _require_key_binding=_require_public_key_binding,
    _verify_transition_exact=_verify_transition,
    _fail_exact=_fail,
) -> LogicalTimeLedgerHeadV1:
    """Verify one bytes-only checkpoint transition without filesystem trust."""

    previous_head = _load_head(previous_head_payload)
    checkpoint = _load_checkpoint(checkpoint_payload)
    next_head = _load_head(next_head_payload)
    expected = _validate_common(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        logical_time_key_id=expected_logical_time_key_id,
        logical_time_key_epoch=expected_logical_time_key_epoch,
        attempt_context_id=expected_attempt_context_id,
    )
    if _common_projection(previous_head) != expected:
        _fail_exact(
            FAIL_BINDING_V1, "ledger transition differs from expected pins"
        )
    _require_key_binding(
        logical_time_public_key,
        expected_logical_time_key_id,
        expected_logical_time_key_epoch,
    )
    _verify_transition_exact(
        previous_head, checkpoint, next_head, logical_time_public_key
    )
    return next_head


def _build_logical_time_transition_entry_v1(implementation):
    """Hide captured verifier internals behind the exact public call surface."""

    def entry(
        previous_head_payload: bytes,
        checkpoint_payload: bytes,
        next_head_payload: bytes,
        *,
        logical_time_public_key: bytes,
        expected_authority_profile_id: str,
        expected_signer_registry_id: str,
        expected_logical_time_key_id: str,
        expected_logical_time_key_epoch: int,
        expected_attempt_context_id: str,
    ) -> LogicalTimeLedgerHeadV1:
        return implementation(
            previous_head_payload,
            checkpoint_payload,
            next_head_payload,
            logical_time_public_key=logical_time_public_key,
            expected_authority_profile_id=expected_authority_profile_id,
            expected_signer_registry_id=expected_signer_registry_id,
            expected_logical_time_key_id=expected_logical_time_key_id,
            expected_logical_time_key_epoch=expected_logical_time_key_epoch,
            expected_attempt_context_id=expected_attempt_context_id,
        )

    entry.__name__ = "verify_logical_time_ledger_transition_v1"
    entry.__qualname__ = "verify_logical_time_ledger_transition_v1"
    return entry


verify_logical_time_ledger_transition_v1 = _build_logical_time_transition_entry_v1(
    _verify_logical_time_ledger_transition_implementation_v1
)
del _verify_logical_time_ledger_transition_implementation_v1
del _build_logical_time_transition_entry_v1


@dataclass(frozen=True, slots=True)
class ExpectedLogicalEventV1:
    """Exact caller-supplied event plan row for whole-chain replay.

    This DTO describes the event sequence that a higher-level frozen protocol
    expects.  Matching it proves exact equality only; this module does not
    decide that the plan itself was authorized by governance.
    """

    purpose: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    request_nonce: str

    def __post_init__(self) -> None:
        if type(self) is not ExpectedLogicalEventV1:
            _fail(FAIL_INPUT_V1, "expected logical event must use the exact DTO")
        _purpose(self.purpose, "expected logical event purpose")
        _identifier(self.bound_artifact_id, "expected bound artifact ID")
        _sha256_address(
            self.bound_artifact_sha256, "expected bound artifact SHA-256"
        )
        _nonce(self.request_nonce, "expected logical event request nonce")

    def as_tuple(self) -> tuple[str, str, str, str]:
        self.__post_init__()
        return (
            self.purpose,
            self.bound_artifact_id,
            self.bound_artifact_sha256,
            self.request_nonce,
        )


def _expected_logical_event_tuple_v1(
    value: object,
    _event_type=ExpectedLogicalEventV1,
    _validate_purpose=_purpose,
    _validate_identifier=_identifier,
    _validate_sha256=_sha256_address,
    _validate_nonce=_nonce,
    _fail_exact=_fail,
) -> tuple[str, str, str, str]:
    """Project one event DTO without dynamically resolving its class methods."""

    if type(value) is not _event_type:
        _fail_exact(FAIL_INPUT_V1, "expected logical event must use exact DTO")
    return (
        _validate_purpose(value.purpose, "expected logical event purpose"),
        _validate_identifier(
            value.bound_artifact_id, "expected bound artifact ID"
        ),
        _validate_sha256(
            value.bound_artifact_sha256, "expected bound artifact SHA-256"
        ),
        _validate_nonce(
            value.request_nonce, "expected logical event request nonce"
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedLogicalTimeChainV1:
    """Privately issued whole-chain mechanics result, never authority evidence."""

    authority_profile_id: str
    signer_registry_id: str
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    ledger_id: str
    verified_event_count: int
    verified_signature_count: int
    verified_event_plan: tuple[tuple[str, str, str, str], ...]
    request_ids: tuple[str, ...]
    token_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    head_ids: tuple[str, ...]
    final_head_id: str
    final_head_payload_sha256: str
    zero_event_genesis_only: bool
    mechanics_only: bool
    authority_evidence: bool
    external_anti_fork_verified: bool
    event_authorization_verified: bool
    attempt_context_freshness_verified: bool
    service_image_verified: bool
    service_uid_verified: bool
    control_domain_verified: bool
    key_custody_verified: bool
    external_cas_checkpoint_verified: bool
    trusted_utc_verified: bool
    _issuance_sentinel: object = field(init=False, repr=False, compare=False)

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiedLogicalTimeChainV1":
        raise TypeError("verified logical-time chains are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified logical-time chains are privately issued")


def _build_verified_logical_time_chain_helpers_v1():
    """Create one hidden result lineage for this imported module instance."""

    sentinel = object()
    result_type = VerifiedLogicalTimeChainV1
    head_type = LogicalTimeLedgerHeadV1
    validate_common = _validate_common_bindings
    ledger_identity = _ledger_id
    genesis_head = _head
    load_head = LogicalTimeLedgerHeadV1.from_canonical_bytes
    validate_head = LogicalTimeLedgerHeadV1.__post_init__
    common_projection = _common_tuple
    validate_integer = _wire_integer
    validate_prefixed_id = _prefixed_id
    validate_sha256 = _sha256_address
    validate_purpose = _purpose
    validate_identifier = _identifier
    validate_nonce = _nonce
    sha256_bytes = _sha256_bytes
    fail_exact = _fail
    request_id_prefix = REQUEST_ID_PREFIX_V1
    token_id_prefix = TOKEN_ID_PREFIX_V1
    checkpoint_id_prefix = LEDGER_CHECKPOINT_ID_PREFIX_V1
    head_id_prefix = LEDGER_HEAD_ID_PREFIX_V1
    genesis_token_id = GENESIS_TOKEN_ID_V1
    genesis_checkpoint_id = GENESIS_CHECKPOINT_ID_V1

    def require(value: object) -> VerifiedLogicalTimeChainV1:
        if type(value) is not result_type:
            fail_exact(
                FAIL_INPUT_V1,
                "verified logical-time chain must use the exact private DTO",
            )
        try:
            observed_sentinel = object.__getattribute__(
                value, "_issuance_sentinel"
            )
        except AttributeError:
            fail_exact(
                FAIL_INPUT_V1,
                "verified logical-time chain lacks its private lineage",
            )
        if observed_sentinel is not sentinel:
            fail_exact(
                FAIL_INPUT_V1,
                "verified logical-time chain private lineage differs",
            )

        common = validate_common(
            authority_profile_id=value.authority_profile_id,
            signer_registry_id=value.signer_registry_id,
            logical_time_key_id=value.logical_time_key_id,
            logical_time_key_epoch=value.logical_time_key_epoch,
            attempt_context_id=value.attempt_context_id,
        )
        expected_ledger_id = ledger_identity(
            authority_profile_id=common[0],
            signer_registry_id=common[1],
            logical_time_key_id=common[2],
            logical_time_key_epoch=common[3],
            attempt_context_id=common[4],
        )
        if value.ledger_id != expected_ledger_id:
            fail_exact(FAIL_BINDING_V1, "verified result ledger ID differs")

        count = validate_integer(
            value.verified_event_count, "verified event count"
        )
        signature_count = validate_integer(
            value.verified_signature_count, "verified signature count"
        )
        if signature_count != count:
            fail_exact(
                FAIL_LEDGER_V1,
                "verified result event and signature counts differ",
            )
        if type(value.verified_event_plan) is not tuple:
            fail_exact(FAIL_INPUT_V1, "verified event plan must be exact tuple")
        rows: set[tuple[str, str, str, str]] = set()
        nonces: set[str] = set()
        event_bindings: set[tuple[str, str, str]] = set()
        for row in value.verified_event_plan:
            if type(row) is not tuple or len(row) != 4:
                fail_exact(
                    FAIL_INPUT_V1,
                    "verified event-plan row must be one exact four-tuple",
                )
            purpose = validate_purpose(row[0], "verified event purpose")
            artifact_id = validate_identifier(
                row[1], "verified event artifact ID"
            )
            artifact_sha256 = validate_sha256(
                row[2], "verified event artifact SHA-256"
            )
            nonce = validate_nonce(row[3], "verified event request nonce")
            exact_row = (purpose, artifact_id, artifact_sha256, nonce)
            event_binding = (purpose, artifact_id, artifact_sha256)
            if (
                exact_row in rows
                or nonce in nonces
                or event_binding in event_bindings
            ):
                fail_exact(
                    FAIL_LEDGER_V1,
                    "verified result event plan contains a duplicate",
                )
            rows.add(exact_row)
            nonces.add(nonce)
            event_bindings.add(event_binding)

        def exact_id_tuple(
            value_tuple: object,
            prefix: str,
            label: str,
            required_count: int,
        ) -> tuple[str, ...]:
            if type(value_tuple) is not tuple or len(value_tuple) != required_count:
                fail_exact(
                    FAIL_INPUT_V1, f"verified result {label} tuple differs"
                )
            validated = tuple(
                validate_prefixed_id(item, prefix, label) for item in value_tuple
            )
            if len(set(validated)) != len(validated):
                fail_exact(
                    FAIL_LEDGER_V1,
                    f"verified result {label} tuple contains a duplicate",
                )
            return validated

        request_ids = exact_id_tuple(
            value.request_ids, request_id_prefix, "request IDs", count
        )
        token_ids = exact_id_tuple(
            value.token_ids, token_id_prefix, "token IDs", count
        )
        checkpoint_ids = exact_id_tuple(
            value.checkpoint_ids,
            checkpoint_id_prefix,
            "checkpoint IDs",
            count,
        )
        head_ids = exact_id_tuple(
            value.head_ids,
            head_id_prefix,
            "head IDs",
            count + 1,
        )
        if not (
            len(value.verified_event_plan) == count
            and len(request_ids) == len(token_ids) == len(checkpoint_ids) == count
            and value.final_head_id == head_ids[-1]
        ):
            fail_exact(FAIL_LEDGER_V1, "verified result cardinality differs")
        validate_prefixed_id(
            value.final_head_id,
            head_id_prefix,
            "verified final head ID",
        )
        validate_sha256(
            value.final_head_payload_sha256,
            "verified final head payload SHA-256",
        )
        expected_genesis = genesis_head(
            ledger_id=expected_ledger_id,
            authority_profile_id=common[0],
            signer_registry_id=common[1],
            logical_time_key_id=common[2],
            logical_time_key_epoch=common[3],
            attempt_context_id=common[4],
            checkpoint_count=0,
            last_token_id=genesis_token_id,
            last_checkpoint_id=genesis_checkpoint_id,
        )
        if head_ids[0] != expected_genesis.head_id:
            fail_exact(FAIL_LEDGER_V1, "verified result genesis head differs")
        if (
            value.zero_event_genesis_only is not (count == 0)
            or value.mechanics_only is not True
            or value.authority_evidence is not False
            or value.external_anti_fork_verified is not False
            or value.event_authorization_verified is not False
            or value.attempt_context_freshness_verified is not False
            or value.service_image_verified is not False
            or value.service_uid_verified is not False
            or value.control_domain_verified is not False
            or value.key_custody_verified is not False
            or value.external_cas_checkpoint_verified is not False
            or value.trusted_utc_verified is not False
        ):
            fail_exact(
                FAIL_BINDING_V1,
                "verified result mechanics/authority boundary differs",
            )
        return value

    def issue(
        *,
        common: tuple[str, str, str, int, str],
        ledger_id: str,
        event_plan: tuple[tuple[str, str, str, str], ...],
        request_ids: tuple[str, ...],
        token_ids: tuple[str, ...],
        checkpoint_ids: tuple[str, ...],
        head_ids: tuple[str, ...],
        final_head: LogicalTimeLedgerHeadV1,
        final_head_payload: bytes,
    ) -> VerifiedLogicalTimeChainV1:
        if type(common) is not tuple or len(common) != 5:
            fail_exact(FAIL_INPUT_V1, "private result common tuple differs")
        validated_common = validate_common(
            authority_profile_id=common[0],
            signer_registry_id=common[1],
            logical_time_key_id=common[2],
            logical_time_key_epoch=common[3],
            attempt_context_id=common[4],
        )
        if type(final_head) is not head_type:
            fail_exact(FAIL_INPUT_V1, "private result final head DTO differs")
        validate_head(final_head)
        if type(final_head_payload) is not bytes:
            fail_exact(FAIL_INPUT_V1, "private result final head must be bytes")
        reparsed_final_head = load_head(final_head_payload)
        if reparsed_final_head != final_head:
            fail_exact(FAIL_BINDING_V1, "private result final head bytes differ")
        count = len(event_plan) if type(event_plan) is tuple else -1
        if not (
            common_projection(final_head) == validated_common
            and final_head.ledger_id == ledger_id
            and final_head.checkpoint_count == count
            and final_head.next_sequence == count
            and type(head_ids) is tuple
            and len(head_ids) == count + 1
            and head_ids[-1] == final_head.head_id
        ):
            fail_exact(FAIL_LEDGER_V1, "private result final-head linkage differs")
        result = object.__new__(result_type)
        object.__setattr__(result, "_issuance_sentinel", sentinel)
        object.__setattr__(result, "authority_profile_id", validated_common[0])
        object.__setattr__(result, "signer_registry_id", validated_common[1])
        object.__setattr__(result, "logical_time_key_id", validated_common[2])
        object.__setattr__(result, "logical_time_key_epoch", validated_common[3])
        object.__setattr__(result, "attempt_context_id", validated_common[4])
        object.__setattr__(result, "ledger_id", ledger_id)
        object.__setattr__(result, "verified_event_count", count)
        object.__setattr__(result, "verified_signature_count", count)
        object.__setattr__(result, "verified_event_plan", event_plan)
        object.__setattr__(result, "request_ids", request_ids)
        object.__setattr__(result, "token_ids", token_ids)
        object.__setattr__(result, "checkpoint_ids", checkpoint_ids)
        object.__setattr__(result, "head_ids", head_ids)
        object.__setattr__(result, "final_head_id", final_head.head_id)
        object.__setattr__(
            result,
            "final_head_payload_sha256",
            sha256_bytes(final_head_payload),
        )
        object.__setattr__(result, "zero_event_genesis_only", count == 0)
        object.__setattr__(result, "mechanics_only", True)
        object.__setattr__(result, "authority_evidence", False)
        object.__setattr__(result, "external_anti_fork_verified", False)
        object.__setattr__(result, "event_authorization_verified", False)
        object.__setattr__(result, "attempt_context_freshness_verified", False)
        object.__setattr__(result, "service_image_verified", False)
        object.__setattr__(result, "service_uid_verified", False)
        object.__setattr__(result, "control_domain_verified", False)
        object.__setattr__(result, "key_custody_verified", False)
        object.__setattr__(result, "external_cas_checkpoint_verified", False)
        object.__setattr__(result, "trusted_utc_verified", False)
        return require(result)

    return issue, require


(
    _hidden_issue_verified_logical_time_chain_v1,
    require_verified_logical_time_chain_v1,
) = _build_verified_logical_time_chain_helpers_v1()
del _build_verified_logical_time_chain_helpers_v1


def _verify_logical_time_ledger_replay_implementation_v1(
    genesis_head_payload: bytes,
    checkpoint_payloads: tuple[bytes, ...],
    subsequent_head_payloads: tuple[bytes, ...],
    *,
    logical_time_public_key: bytes,
    expected_authority_profile_id: str,
    expected_signer_registry_id: str,
    expected_logical_time_key_id: str,
    expected_logical_time_key_epoch: int,
    expected_attempt_context_id: str,
    expected_event_plan: tuple[ExpectedLogicalEventV1, ...],
    _event_type=ExpectedLogicalEventV1,
    _event_tuple=_expected_logical_event_tuple_v1,
    _validate_common=_validate_common_bindings,
    _require_key_binding=_require_public_key_binding,
    _load_head=LogicalTimeLedgerHeadV1.from_canonical_bytes,
    _load_checkpoint=LogicalTimeLedgerCheckpointV1.from_canonical_bytes,
    _ledger_identity=_ledger_id,
    _common_projection=_common_tuple,
    _verify_transition_exact=_verify_transition,
    _project_request=_request_from_token,
    _issue_result=_hidden_issue_verified_logical_time_chain_v1,
    _fail_exact=_fail,
    _genesis_token_id: str = GENESIS_TOKEN_ID_V1,
    _genesis_checkpoint_id: str = GENESIS_CHECKPOINT_ID_V1,
) -> VerifiedLogicalTimeChainV1:
    """Replay a complete caller-supplied canonical logical-time chain.

    All wire records enter as exact bytes.  No decoded-object verifier or
    callback is accepted.  ``subsequent_head_payloads[i]`` must be the exact
    head produced by ``checkpoint_payloads[i]``.  The supplied event plan is
    exact-compared in order against every signed token; matching the plan does
    not establish that governance authorized the plan.

    The zero-event case is accepted only when all three supplied tuples are
    exact empty tuples and ``genesis_head_payload`` is the exact configured
    genesis.  Its private result says ``zero_event_genesis_only=True``, has
    zero verified signatures, and remains mechanics-only with every external
    authority predicate false.
    """

    if type(checkpoint_payloads) is not tuple:
        _fail_exact(FAIL_INPUT_V1, "checkpoint payloads must be an exact tuple")
    if type(subsequent_head_payloads) is not tuple:
        _fail_exact(
            FAIL_INPUT_V1, "subsequent head payloads must be an exact tuple"
        )
    if type(expected_event_plan) is not tuple:
        _fail_exact(FAIL_INPUT_V1, "expected event plan must be an exact tuple")
    if any(type(item) is not bytes for item in checkpoint_payloads):
        _fail_exact(FAIL_INPUT_V1, "every checkpoint payload must be exact bytes")
    if any(type(item) is not bytes for item in subsequent_head_payloads):
        _fail_exact(
            FAIL_INPUT_V1, "every subsequent head payload must be exact bytes"
        )
    if any(
        type(item) is not _event_type for item in expected_event_plan
    ):
        _fail_exact(FAIL_INPUT_V1, "every expected event must use the exact DTO")
    count = len(expected_event_plan)
    if len(checkpoint_payloads) != count or len(subsequent_head_payloads) != count:
        _fail_exact(
            FAIL_LEDGER_V1,
            "event plan, checkpoint tuple, and subsequent-head tuple lengths differ",
        )

    expected_common = _validate_common(
        authority_profile_id=expected_authority_profile_id,
        signer_registry_id=expected_signer_registry_id,
        logical_time_key_id=expected_logical_time_key_id,
        logical_time_key_epoch=expected_logical_time_key_epoch,
        attempt_context_id=expected_attempt_context_id,
    )
    _require_key_binding(
        logical_time_public_key,
        expected_logical_time_key_id,
        expected_logical_time_key_epoch,
    )
    genesis = _load_head(genesis_head_payload)
    expected_ledger_id = _ledger_identity(
        authority_profile_id=expected_common[0],
        signer_registry_id=expected_common[1],
        logical_time_key_id=expected_common[2],
        logical_time_key_epoch=expected_common[3],
        attempt_context_id=expected_common[4],
    )
    if (
        _common_projection(genesis) != expected_common
        or genesis.ledger_id != expected_ledger_id
        or genesis.checkpoint_count != 0
        or genesis.next_sequence != 0
        or genesis.last_token_id != _genesis_token_id
        or genesis.last_checkpoint_id != _genesis_checkpoint_id
    ):
        _fail_exact(FAIL_LEDGER_V1, "whole-chain replay genesis differs")

    plan_rows: list[tuple[str, str, str, str]] = []
    plan_nonces: set[str] = set()
    plan_event_bindings: set[tuple[str, str, str]] = set()
    for item in expected_event_plan:
        row = _event_tuple(item)
        event_binding = (row[0], row[1], row[2])
        if (
            row in plan_rows
            or item.request_nonce in plan_nonces
            or event_binding in plan_event_bindings
        ):
            _fail_exact(
                FAIL_LEDGER_V1, "expected event plan contains a duplicate"
            )
        plan_rows.append(row)
        plan_nonces.add(item.request_nonce)
        plan_event_bindings.add(event_binding)

    current_head = genesis
    current_head_payload = genesis_head_payload
    request_ids: list[str] = []
    token_ids: list[str] = []
    checkpoint_ids: list[str] = []
    head_ids: list[str] = [genesis.head_id]
    observed_request_ids: set[str] = set()
    observed_request_nonces: set[str] = set()
    observed_token_ids: set[str] = set()
    observed_checkpoint_ids: set[str] = set()
    observed_head_ids: set[str] = {genesis.head_id}

    for index, (checkpoint_payload, next_head_payload, expected_event) in enumerate(
        zip(
            checkpoint_payloads,
            subsequent_head_payloads,
            expected_event_plan,
            strict=True,
        )
    ):
        checkpoint = _load_checkpoint(checkpoint_payload)
        next_head = _load_head(next_head_payload)
        _verify_transition_exact(
            current_head, checkpoint, next_head, logical_time_public_key
        )
        token = checkpoint.signed_token
        request = _project_request(token)
        if (
            checkpoint.checkpoint_index != index
            or token.sequence != index
            or request.next_sequence != index
        ):
            _fail_exact(
                FAIL_LEDGER_V1, "whole-chain replay sequence is not contiguous"
            )
        observed_event = (
            token.purpose,
            token.bound_artifact_id,
            token.bound_artifact_sha256,
            token.request_nonce,
        )
        if observed_event != _event_tuple(expected_event):
            _fail_exact(
                FAIL_BINDING_V1,
                "signed logical-time event differs from the exact event plan",
            )
        if (
            request.request_id in observed_request_ids
            or request.request_nonce in observed_request_nonces
            or token.token_id in observed_token_ids
            or checkpoint.checkpoint_id in observed_checkpoint_ids
            or next_head.head_id in observed_head_ids
        ):
            _fail_exact(
                FAIL_LEDGER_V1, "whole-chain replay duplicate or fork detected"
            )
        observed_request_ids.add(request.request_id)
        observed_request_nonces.add(request.request_nonce)
        observed_token_ids.add(token.token_id)
        observed_checkpoint_ids.add(checkpoint.checkpoint_id)
        observed_head_ids.add(next_head.head_id)
        request_ids.append(request.request_id)
        token_ids.append(token.token_id)
        checkpoint_ids.append(checkpoint.checkpoint_id)
        head_ids.append(next_head.head_id)
        current_head = next_head
        current_head_payload = next_head_payload

    if (
        current_head.checkpoint_count != count
        or current_head.next_sequence != count
        or len(request_ids) != count
        or len(head_ids) != count + 1
    ):
        _fail_exact(FAIL_LEDGER_V1, "whole-chain replay final head count differs")
    if count == 0 and (
        checkpoint_payloads != ()
        or subsequent_head_payloads != ()
        or expected_event_plan != ()
        or current_head != genesis
        or current_head_payload != genesis_head_payload
    ):
        _fail_exact(
            FAIL_LEDGER_V1, "zero-event replay is not exact genesis-only input"
        )

    return _issue_result(
        common=expected_common,
        ledger_id=expected_ledger_id,
        event_plan=tuple(plan_rows),
        request_ids=tuple(request_ids),
        token_ids=tuple(token_ids),
        checkpoint_ids=tuple(checkpoint_ids),
        head_ids=tuple(head_ids),
        final_head=current_head,
        final_head_payload=current_head_payload,
    )


def _build_logical_time_replay_entry_v1(implementation):
    """Hide the import-time replay call graph behind the public signature."""

    def entry(
        genesis_head_payload: bytes,
        checkpoint_payloads: tuple[bytes, ...],
        subsequent_head_payloads: tuple[bytes, ...],
        *,
        logical_time_public_key: bytes,
        expected_authority_profile_id: str,
        expected_signer_registry_id: str,
        expected_logical_time_key_id: str,
        expected_logical_time_key_epoch: int,
        expected_attempt_context_id: str,
        expected_event_plan: tuple[ExpectedLogicalEventV1, ...],
    ) -> VerifiedLogicalTimeChainV1:
        return implementation(
            genesis_head_payload,
            checkpoint_payloads,
            subsequent_head_payloads,
            logical_time_public_key=logical_time_public_key,
            expected_authority_profile_id=expected_authority_profile_id,
            expected_signer_registry_id=expected_signer_registry_id,
            expected_logical_time_key_id=expected_logical_time_key_id,
            expected_logical_time_key_epoch=expected_logical_time_key_epoch,
            expected_attempt_context_id=expected_attempt_context_id,
            expected_event_plan=expected_event_plan,
        )

    entry.__name__ = "verify_logical_time_ledger_replay_v1"
    entry.__qualname__ = "verify_logical_time_ledger_replay_v1"
    return entry


verify_logical_time_ledger_replay_v1 = _build_logical_time_replay_entry_v1(
    _verify_logical_time_ledger_replay_implementation_v1
)
del _verify_logical_time_ledger_replay_implementation_v1
del _build_logical_time_replay_entry_v1
del _hidden_issue_verified_logical_time_chain_v1


@dataclass(frozen=True, slots=True)
class LogicalTimeServiceIssuanceV1:
    """Bytes returned after one durable service-side append."""

    request_id: str
    token_id: str
    checkpoint_id: str
    head_id: str
    signed_token_payload: bytes
    checkpoint_payload: bytes
    head_payload: bytes
    mechanics_only: bool = True
    authority_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not LogicalTimeServiceIssuanceV1:
            _fail(FAIL_INPUT_V1, "issuance result must use the exact DTO")
        _prefixed_id(self.request_id, REQUEST_ID_PREFIX_V1, "issuance request ID")
        _prefixed_id(self.token_id, TOKEN_ID_PREFIX_V1, "issuance token ID")
        _prefixed_id(
            self.checkpoint_id,
            LEDGER_CHECKPOINT_ID_PREFIX_V1,
            "issuance checkpoint ID",
        )
        _prefixed_id(self.head_id, LEDGER_HEAD_ID_PREFIX_V1, "issuance head ID")
        token = SignedLogicalTimeTokenV1.from_canonical_bytes(
            self.signed_token_payload
        )
        checkpoint = LogicalTimeLedgerCheckpointV1.from_canonical_bytes(
            self.checkpoint_payload
        )
        head = LogicalTimeLedgerHeadV1.from_canonical_bytes(self.head_payload)
        if (
            token.request_id != self.request_id
            or token.token_id != self.token_id
            or checkpoint.checkpoint_id != self.checkpoint_id
            or head.head_id != self.head_id
            or checkpoint.signed_token != token
            or head.last_token_id != token.token_id
            or head.last_checkpoint_id != checkpoint.checkpoint_id
            or self.mechanics_only is not True
            or self.authority_evidence is not False
        ):
            _fail(FAIL_BINDING_V1, "service issuance result linkage differs")


def _require_filesystem_support() -> None:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        _fail(
            FAIL_FILESYSTEM_V1,
            "logical-time ledger requires Linux no-follow directory operations",
        )


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_nlink == right.st_nlink
        and left.st_uid == right.st_uid
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            _fail(FAIL_FILESYSTEM_V1, "ledger durable write made no progress")
        offset += written


def _read_regular_at(
    directory_fd: int,
    name: str,
    maximum_bytes: int,
) -> bytes:
    fd = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not 1 <= before.st_size <= maximum_bytes
        ):
            _fail(FAIL_FILESYSTEM_V1, "ledger record metadata differs")
        blocks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(65_536, before.st_size - offset), offset)
            if not block:
                _fail(FAIL_FILESYSTEM_V1, "ledger record ended during readback")
            blocks.append(block)
            offset += len(block)
        if os.pread(fd, 1, offset):
            _fail(FAIL_FILESYSTEM_V1, "ledger record grew during readback")
        after = os.fstat(fd)
        if not _same_file(before, after):
            _fail(FAIL_FILESYSTEM_V1, "ledger record changed during readback")
        return b"".join(blocks)
    finally:
        os.close(fd)


def _versioned_filename(index: int) -> str:
    _wire_integer(index, "ledger filename index")
    name = f"{index:020d}.json"
    if _VERSIONED_FILENAME_PATTERN.fullmatch(name) is None:
        _fail(FAIL_LEDGER_V1, "ledger filename index overflows exact format")
    return name


class LogicalTimeLedgerV1:
    """Service-side, local append-only mechanics under an exclusive lock.

    Heads are immutable versions ``0..N`` and checkpoints are immutable
    versions ``0..N-1``.  No mutable head pointer exists.  Every append exact-
    compares the caller's prior-head bytes with the reopened current head.
    Publication uses ``O_EXCL`` for both a staged file and the final name,
    fsyncs each file and its directory, reparses the final canonical bytes, and
    then audits the entire chain.  A crash after one half of a transition is
    intentionally not repaired: the gap/extra leaves the ledger failed closed
    for explicit operator review and external-checkpoint reconciliation.
    """

    __slots__ = (
        "root_directory",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "logical_time_public_key",
        "ledger_id",
        "_root_fd",
        "_heads_fd",
        "_checkpoints_fd",
        "_lock_fd",
        "_thread_lock",
        "_publication_nonce",
        "_closed",
    )

    def __init__(
        self,
        *,
        root_directory: str,
        authority_profile_id: str,
        signer_registry_id: str,
        logical_time_key_id: str,
        logical_time_key_epoch: int,
        attempt_context_id: str,
        logical_time_public_key: bytes,
        create: bool = False,
    ) -> None:
        _require_filesystem_support()
        if type(root_directory) is not str or not os.path.isabs(root_directory):
            _fail(FAIL_INPUT_V1, "ledger root must be one absolute path")
        if os.path.abspath(root_directory) != root_directory:
            _fail(FAIL_INPUT_V1, "ledger root path is not lexically canonical")
        common = _validate_common_bindings(
            authority_profile_id=authority_profile_id,
            signer_registry_id=signer_registry_id,
            logical_time_key_id=logical_time_key_id,
            logical_time_key_epoch=logical_time_key_epoch,
            attempt_context_id=attempt_context_id,
        )
        _require_public_key_binding(
            logical_time_public_key,
            logical_time_key_id,
            logical_time_key_epoch,
        )
        self.root_directory = root_directory
        (
            self.authority_profile_id,
            self.signer_registry_id,
            self.logical_time_key_id,
            self.logical_time_key_epoch,
            self.attempt_context_id,
        ) = common
        self.logical_time_public_key = bytes(logical_time_public_key)
        self.ledger_id = _ledger_id(
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            logical_time_key_id=self.logical_time_key_id,
            logical_time_key_epoch=self.logical_time_key_epoch,
            attempt_context_id=self.attempt_context_id,
        )
        self._root_fd = -1
        self._heads_fd = -1
        self._checkpoints_fd = -1
        self._lock_fd = -1
        self._thread_lock = threading.Lock()
        self._publication_nonce = 0
        self._closed = False
        created_root = False
        try:
            if create:
                try:
                    os.mkdir(root_directory, 0o700)
                    created_root = True
                    parent_directory = os.path.dirname(root_directory)
                    parent_fd = os.open(
                        parent_directory,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                    )
                    try:
                        os.fsync(parent_fd)
                    finally:
                        os.close(parent_fd)
                except FileExistsError:
                    pass
            if not os.path.isdir(root_directory):
                _fail(FAIL_FILESYSTEM_V1, "ledger root does not exist as a directory")
            if os.path.realpath(root_directory) != root_directory:
                _fail(FAIL_FILESYSTEM_V1, "ledger root resolves through a symlink")
            self._root_fd = os.open(
                root_directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            self._require_private_directory(self._root_fd, "ledger root")
            if created_root:
                os.mkdir("heads", 0o700, dir_fd=self._root_fd)
                os.mkdir("checkpoints", 0o700, dir_fd=self._root_fd)
                self._lock_fd = os.open(
                    "ledger.lock",
                    os.O_RDWR
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._root_fd,
                )
                os.fsync(self._lock_fd)
                os.fsync(self._root_fd)
            else:
                if set(os.listdir(self._root_fd)) != {
                    "heads",
                    "checkpoints",
                    "ledger.lock",
                }:
                    _fail(FAIL_LEDGER_V1, "ledger root exact entry set differs")
                self._lock_fd = os.open(
                    "ledger.lock",
                    os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=self._root_fd,
                )
            self._validate_lock_file()
            self._heads_fd = self._open_private_directory("heads")
            self._checkpoints_fd = self._open_private_directory("checkpoints")
            with self._exclusive_lock():
                if created_root:
                    if os.listdir(self._heads_fd) or os.listdir(self._checkpoints_fd):
                        _fail(FAIL_LEDGER_V1, "new ledger directories are not empty")
                    genesis = _head(
                        ledger_id=self.ledger_id,
                        authority_profile_id=self.authority_profile_id,
                        signer_registry_id=self.signer_registry_id,
                        logical_time_key_id=self.logical_time_key_id,
                        logical_time_key_epoch=self.logical_time_key_epoch,
                        attempt_context_id=self.attempt_context_id,
                        checkpoint_count=0,
                        last_token_id=GENESIS_TOKEN_ID_V1,
                        last_checkpoint_id=GENESIS_CHECKPOINT_ID_V1,
                    )
                    self._publish_exclusive(
                        self._heads_fd,
                        _versioned_filename(0),
                        genesis.canonical_bytes(),
                        MAXIMUM_HEAD_BYTES,
                        LogicalTimeLedgerHeadV1.from_canonical_bytes,
                    )
                self._load_chain()
        except BaseException:
            self.close()
            raise

    def _require_private_directory(self, fd: int, label: str) -> None:
        observed = os.fstat(fd)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or observed.st_uid != os.geteuid()
        ):
            _fail(FAIL_FILESYSTEM_V1, f"{label} ownership or mode differs")

    def _validate_lock_file(self) -> None:
        observed = os.fstat(self._lock_fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
            or observed.st_size != 0
        ):
            _fail(FAIL_FILESYSTEM_V1, "ledger lock metadata differs")

    def _open_private_directory(self, name: str) -> int:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._root_fd,
        )
        try:
            self._require_private_directory(fd, f"ledger {name} directory")
            return fd
        except BaseException:
            os.close(fd)
            raise

    class _ExclusiveLock:
        def __init__(self, owner: "LogicalTimeLedgerV1") -> None:
            self.owner = owner

        def __enter__(self) -> None:
            if self.owner._closed or self.owner._lock_fd < 0:
                _fail(FAIL_LEDGER_V1, "logical-time ledger is closed")
            self.owner._thread_lock.acquire()
            try:
                fcntl.flock(self.owner._lock_fd, fcntl.LOCK_EX)
            except BaseException:
                self.owner._thread_lock.release()
                raise

        def __exit__(
            self, exc_type: object, exc: object, traceback: object
        ) -> bool:
            try:
                fcntl.flock(self.owner._lock_fd, fcntl.LOCK_UN)
            finally:
                self.owner._thread_lock.release()
            return False

    def _exclusive_lock(self) -> "LogicalTimeLedgerV1._ExclusiveLock":
        return self._ExclusiveLock(self)

    def _expected_common(self) -> tuple[str, str, str, int, str]:
        return (
            self.authority_profile_id,
            self.signer_registry_id,
            self.logical_time_key_id,
            self.logical_time_key_epoch,
            self.attempt_context_id,
        )

    def _exact_names(self, directory_fd: int, count: int, label: str) -> tuple[str, ...]:
        names = tuple(sorted(os.listdir(directory_fd)))
        expected = tuple(_versioned_filename(index) for index in range(count))
        if names != expected:
            _fail(
                FAIL_LEDGER_V1,
                f"{label} filenames are not the exact contiguous 0..N set",
            )
        return names

    def _load_chain(
        self,
    ) -> tuple[
        tuple[LogicalTimeLedgerHeadV1, ...],
        tuple[LogicalTimeLedgerCheckpointV1, ...],
        tuple[bytes, ...],
    ]:
        self._require_private_directory(self._root_fd, "ledger root")
        self._require_private_directory(self._heads_fd, "ledger heads directory")
        self._require_private_directory(
            self._checkpoints_fd, "ledger checkpoints directory"
        )
        self._validate_lock_file()
        if set(os.listdir(self._root_fd)) != {
            "heads",
            "checkpoints",
            "ledger.lock",
        }:
            _fail(FAIL_LEDGER_V1, "ledger root exact entry set differs")
        head_names = tuple(sorted(os.listdir(self._heads_fd)))
        checkpoint_names = tuple(sorted(os.listdir(self._checkpoints_fd)))
        if not head_names:
            _fail(FAIL_LEDGER_V1, "logical-time ledger has no genesis head")
        self._exact_names(self._heads_fd, len(head_names), "ledger head")
        self._exact_names(
            self._checkpoints_fd,
            len(checkpoint_names),
            "ledger checkpoint",
        )
        if len(head_names) != len(checkpoint_names) + 1:
            _fail(FAIL_LEDGER_V1, "ledger head/checkpoint cardinality differs")

        head_payloads = tuple(
            _read_regular_at(self._heads_fd, name, MAXIMUM_HEAD_BYTES)
            for name in head_names
        )
        heads = tuple(
            LogicalTimeLedgerHeadV1.from_canonical_bytes(payload)
            for payload in head_payloads
        )
        checkpoints = tuple(
            LogicalTimeLedgerCheckpointV1.from_canonical_bytes(
                _read_regular_at(
                    self._checkpoints_fd, name, MAXIMUM_CHECKPOINT_BYTES
                )
            )
            for name in checkpoint_names
        )

        genesis = heads[0]
        if (
            genesis.ledger_id != self.ledger_id
            or _common_tuple(genesis) != self._expected_common()
            or genesis.checkpoint_count != 0
            or genesis.next_sequence != 0
            or genesis.last_token_id != GENESIS_TOKEN_ID_V1
            or genesis.last_checkpoint_id != GENESIS_CHECKPOINT_ID_V1
        ):
            _fail(FAIL_LEDGER_V1, "genesis head differs from ledger configuration")

        request_ids: set[str] = set()
        request_nonces: set[str] = set()
        for index, checkpoint in enumerate(checkpoints):
            if (
                heads[index].checkpoint_count != index
                or checkpoint.checkpoint_index != index
                or heads[index + 1].checkpoint_count != index + 1
            ):
                _fail(FAIL_LEDGER_V1, "ledger filename/content index differs")
            _verify_transition(
                heads[index],
                checkpoint,
                heads[index + 1],
                self.logical_time_public_key,
            )
            if (
                checkpoint.request_id in request_ids
                or checkpoint.request_nonce in request_nonces
            ):
                _fail(FAIL_LEDGER_V1, "ledger request ID or nonce replay detected")
            request_ids.add(checkpoint.request_id)
            request_nonces.add(checkpoint.request_nonce)
        return heads, checkpoints, head_payloads

    def _publish_exclusive(
        self,
        directory_fd: int,
        final_name: str,
        payload: bytes,
        maximum_bytes: int,
        parser: object,
    ) -> None:
        if type(payload) is not bytes or not 1 <= len(payload) <= maximum_bytes:
            _fail(FAIL_INPUT_V1, "ledger publication payload is outside its bound")
        self._publication_nonce += 1
        temporary_name = (
            f".pending-{os.getpid()}-{self._publication_nonce}-{final_name}"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
        )
        temporary_fd: int | None = None
        final_fd: int | None = None
        final_created = False
        try:
            temporary_fd = os.open(
                temporary_name, flags, 0o600, dir_fd=directory_fd
            )
            _write_all(temporary_fd, payload)
            os.fsync(temporary_fd)
            os.close(temporary_fd)
            temporary_fd = None
            if _read_regular_at(
                directory_fd, temporary_name, maximum_bytes
            ) != payload:
                _fail(FAIL_FILESYSTEM_V1, "staged ledger readback differs")

            # The final name is also opened with O_EXCL.  A crash may expose a
            # partial final record; that is intentional fail-closed poison, not
            # a record that this module silently repairs or overwrites.
            final_fd = os.open(final_name, flags, 0o600, dir_fd=directory_fd)
            final_created = True
            _write_all(final_fd, payload)
            os.fsync(final_fd)
            os.close(final_fd)
            final_fd = None
            os.fsync(directory_fd)
            observed = _read_regular_at(directory_fd, final_name, maximum_bytes)
            if observed != payload:
                _fail(FAIL_FILESYSTEM_V1, "final ledger readback differs")
            if not callable(parser):
                _fail(FAIL_INPUT_V1, "internal ledger parser is not callable")
            parser(observed)  # type: ignore[operator]
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            if _read_regular_at(directory_fd, final_name, maximum_bytes) != payload:
                _fail(FAIL_FILESYSTEM_V1, "post-fsync ledger readback differs")
        except FileExistsError as exc:
            raise Phase2BStep5LogicalTimeErrorV1(
                FAIL_LEDGER_V1,
                f"exclusive ledger publication collision for {final_name}",
            ) from exc
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            if final_fd is not None:
                os.close(final_fd)
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            # Never unlink a created final record on error.  Append-only
            # ambiguity must remain visible and make subsequent audits fail.
            _ = final_created

    def current_head_bytes(self) -> bytes:
        """Return reopened canonical bytes for the currently audited head."""

        with self._exclusive_lock():
            _heads, _checkpoints, head_payloads = self._load_chain()
            return bytes(head_payloads[-1])

    def issue_service_side_v1(
        self,
        *,
        request_payload: bytes,
        expected_previous_head_payload: bytes,
        private_key: object,
    ) -> LogicalTimeServiceIssuanceV1:
        """Sign and durably append exactly one request inside the service.

        ``private_key`` must be a concrete ``cryptography`` Ed25519 private-key
        object.  Seed bytes, PEM bytes, callbacks, signer protocols, and
        injected verifier functions are rejected.  The object is used only in
        this call and is never assigned to the ledger instance.
        """

        if _Ed25519PrivateKey is None or _serialization is None:
            _fail(
                FAIL_SIGNATURE_BACKEND_V1,
                "cryptography Ed25519 signing backend is unavailable",
            )
        if not isinstance(private_key, _Ed25519PrivateKey):
            _fail(
                FAIL_INPUT_V1,
                "service issuance requires an Ed25519 private-key object",
            )
        request = LogicalTimeRequestV1.from_canonical_bytes(request_payload)
        expected_head = LogicalTimeLedgerHeadV1.from_canonical_bytes(
            expected_previous_head_payload
        )
        if _common_tuple(request) != self._expected_common():
            _fail(FAIL_BINDING_V1, "issuance request differs from service pins")

        public_bytes = private_key.public_key().public_bytes(
            encoding=_serialization.Encoding.Raw,
            format=_serialization.PublicFormat.Raw,
        )
        if public_bytes != self.logical_time_public_key:
            _fail(FAIL_BINDING_V1, "service private key differs from pinned role key")

        with self._exclusive_lock():
            heads, checkpoints, head_payloads = self._load_chain()
            previous_head = heads[-1]
            current_payload = head_payloads[-1]
            if (
                expected_head != previous_head
                or expected_previous_head_payload != current_payload
            ):
                _fail(
                    FAIL_LEDGER_V1,
                    "caller previous-head bytes are stale or compare unequal",
                )
            if request.prior_token_id != previous_head.last_token_id:
                _fail(FAIL_LEDGER_V1, "request prior token is not the current head")
            if request.next_sequence != previous_head.next_sequence:
                _fail(FAIL_LEDGER_V1, "request does not ask for the exact next sequence")
            if previous_head.next_sequence == _MAXIMUM_SAFE_JSON_INTEGER:
                _fail(FAIL_LEDGER_V1, "logical-time sequence space is exhausted")
            if any(
                item.request_id == request.request_id
                or item.request_nonce == request.request_nonce
                for item in checkpoints
            ):
                _fail(FAIL_LEDGER_V1, "request ID or request nonce replay detected")

            unsigned_body = {
                "schema_version": SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION,
                "authority_profile_id": request.authority_profile_id,
                "signer_registry_id": request.signer_registry_id,
                "logical_time_key_id": request.logical_time_key_id,
                "logical_time_key_epoch": request.logical_time_key_epoch,
                "attempt_context_id": request.attempt_context_id,
                "request_id": request.request_id,
                "request_nonce": request.request_nonce,
                "signature_purpose": request.signature_purpose,
                "purpose": request.purpose,
                "bound_artifact_id": request.bound_artifact_id,
                "bound_artifact_sha256": request.bound_artifact_sha256,
                "prior_token_id": request.prior_token_id,
                "sequence": request.next_sequence,
                "algorithm": ED25519_ALGORITHM_V1,
            }
            preimage_wrapper = {
                "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
                "artifact_kind": "logical_time",
                "role": LOGICAL_TIME_ROLE_V1,
                "purpose": SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
                "key_epoch": request.logical_time_key_epoch,
                "signer_key_id": request.logical_time_key_id,
                "body": unsigned_body,
            }
            signature = private_key.sign(
                _TOKEN_SIGNATURE_DOMAIN + _canonical(preimage_wrapper)
            )
            if type(signature) is not bytes or len(signature) != 64:
                _fail(FAIL_SIGNATURE_V1, "Ed25519 signer returned invalid bytes")
            token_id_subject = {
                **unsigned_body,
                "signature_hex": signature.hex(),
            }
            token = SignedLogicalTimeTokenV1(
                **unsigned_body,
                signature_hex=signature.hex(),
                token_id=_content_id(
                    TOKEN_ID_PREFIX_V1, _TOKEN_ID_DOMAIN, token_id_subject
                ),
            )
            _verify_token_signature(token, self.logical_time_public_key)
            if _request_from_token(token) != request:
                _fail(FAIL_BINDING_V1, "issued token request projection differs")

            checkpoint = _checkpoint(previous_head=previous_head, token=token)
            next_head = _head(
                ledger_id=self.ledger_id,
                authority_profile_id=self.authority_profile_id,
                signer_registry_id=self.signer_registry_id,
                logical_time_key_id=self.logical_time_key_id,
                logical_time_key_epoch=self.logical_time_key_epoch,
                attempt_context_id=self.attempt_context_id,
                checkpoint_count=previous_head.checkpoint_count + 1,
                last_token_id=token.token_id,
                last_checkpoint_id=checkpoint.checkpoint_id,
            )
            _verify_transition(
                previous_head,
                checkpoint,
                next_head,
                self.logical_time_public_key,
            )
            checkpoint_payload = checkpoint.canonical_bytes()
            next_head_payload = next_head.canonical_bytes()
            self._publish_exclusive(
                self._checkpoints_fd,
                _versioned_filename(checkpoint.checkpoint_index),
                checkpoint_payload,
                MAXIMUM_CHECKPOINT_BYTES,
                LogicalTimeLedgerCheckpointV1.from_canonical_bytes,
            )
            self._publish_exclusive(
                self._heads_fd,
                _versioned_filename(next_head.checkpoint_count),
                next_head_payload,
                MAXIMUM_HEAD_BYTES,
                LogicalTimeLedgerHeadV1.from_canonical_bytes,
            )
            audited_heads, audited_checkpoints, audited_head_payloads = (
                self._load_chain()
            )
            if (
                audited_heads[-1] != next_head
                or audited_checkpoints[-1] != checkpoint
                or audited_head_payloads[-1] != next_head_payload
            ):
                _fail(FAIL_LEDGER_V1, "post-append full-chain readback differs")

            result = LogicalTimeServiceIssuanceV1(
                request_id=request.request_id,
                token_id=token.token_id,
                checkpoint_id=checkpoint.checkpoint_id,
                head_id=next_head.head_id,
                signed_token_payload=token.canonical_bytes(),
                checkpoint_payload=checkpoint_payload,
                head_payload=next_head_payload,
            )
            result.__post_init__()
            return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in ("_checkpoints_fd", "_heads_fd", "_lock_fd", "_root_fd"):
            fd = object.__getattribute__(self, name)
            if fd >= 0:
                os.close(fd)
                object.__setattr__(self, name, -1)

    def __enter__(self) -> "LogicalTimeLedgerV1":
        if self._closed:
            _fail(FAIL_LEDGER_V1, "logical-time ledger is closed")
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> bool:
        self.close()
        return False


def load_logical_time_request_v1(payload: bytes) -> LogicalTimeRequestV1:
    """Load one canonical bytes-only request/1."""

    return LogicalTimeRequestV1.from_canonical_bytes(payload)


def load_signed_logical_time_token_v1(
    payload: bytes,
) -> SignedLogicalTimeTokenV1:
    """Structurally load token/1 bytes; use the verifier for authenticity."""

    return SignedLogicalTimeTokenV1.from_canonical_bytes(payload)


def load_logical_time_ledger_head_v1(
    payload: bytes,
) -> LogicalTimeLedgerHeadV1:
    """Load one canonical bytes-only immutable ledger head/1."""

    return LogicalTimeLedgerHeadV1.from_canonical_bytes(payload)


def load_logical_time_ledger_checkpoint_v1(
    payload: bytes,
) -> LogicalTimeLedgerCheckpointV1:
    """Load one canonical bytes-only immutable ledger checkpoint/1."""

    return LogicalTimeLedgerCheckpointV1.from_canonical_bytes(payload)


def phase2b_step5_logical_time_request_id_preimage_v1(payload: bytes) -> bytes:
    """Return the domain-separated content-ID preimage for request/1 bytes."""

    request = LogicalTimeRequestV1.from_canonical_bytes(payload)
    return _REQUEST_ID_DOMAIN + _canonical(request._id_subject())


def phase2b_step5_logical_time_token_signature_preimage_v1(
    payload: bytes,
) -> bytes:
    """Return the exact purpose-separated Ed25519 preimage for token/1."""

    return SignedLogicalTimeTokenV1.from_canonical_bytes(
        payload
    ).signature_preimage()


def phase2b_step5_logical_time_token_id_preimage_v1(payload: bytes) -> bytes:
    """Return the domain-separated content-ID preimage for token/1 bytes."""

    token = SignedLogicalTimeTokenV1.from_canonical_bytes(payload)
    return _TOKEN_ID_DOMAIN + _canonical(token._id_subject())


def phase2b_step5_logical_time_head_id_preimage_v1(payload: bytes) -> bytes:
    """Return the domain-separated content-ID preimage for head/1 bytes."""

    head = LogicalTimeLedgerHeadV1.from_canonical_bytes(payload)
    return _LEDGER_HEAD_ID_DOMAIN + _canonical(head._id_subject())


def phase2b_step5_logical_time_checkpoint_id_preimage_v1(
    payload: bytes,
) -> bytes:
    """Return the domain-separated content-ID preimage for checkpoint/1."""

    checkpoint = LogicalTimeLedgerCheckpointV1.from_canonical_bytes(payload)
    return _LEDGER_CHECKPOINT_ID_DOMAIN + _canonical(checkpoint._id_subject())


__all__ = [
    "AUTHORITY_CLAIM_BOUNDARY_V1",
    "AUTHORITY_PROFILE_PREFIX_V2",
    "DEPENDENCY_REPLACEMENT_V1",
    "ED25519_ALGORITHM_V1",
    "ExpectedLogicalEventV1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_FILESYSTEM_V1",
    "FAIL_INPUT_V1",
    "FAIL_LEDGER_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_SIGNATURE_BACKEND_V1",
    "FAIL_SIGNATURE_V1",
    "GENESIS_CHECKPOINT_ID_V1",
    "GENESIS_TOKEN_ID_V1",
    "LEDGER_CHECKPOINT_ID_PREFIX_V1",
    "LEDGER_CHECKPOINT_PURPOSE_V1",
    "LEDGER_HEAD_ID_PREFIX_V1",
    "LEDGER_HEAD_PURPOSE_V1",
    "LEDGER_ID_PREFIX_V1",
    "LOGICAL_TIME_CONTROL_DOMAIN_V1",
    "LOGICAL_TIME_KEY_EPOCH_V1",
    "LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION",
    "LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION",
    "LOGICAL_TIME_REQUEST_SCHEMA_VERSION",
    "LOGICAL_TIME_ROLE_V1",
    "LogicalTimeLedgerCheckpointV1",
    "LogicalTimeLedgerHeadV1",
    "LogicalTimeLedgerV1",
    "LogicalTimeRequestV1",
    "LogicalTimeServiceIssuanceV1",
    "MAXIMUM_CHECKPOINT_BYTES",
    "MAXIMUM_HEAD_BYTES",
    "MAXIMUM_REQUEST_BYTES",
    "MAXIMUM_TOKEN_BYTES",
    "Phase2BStep5LogicalTimeErrorV1",
    "REQUEST_ID_PREFIX_V1",
    "REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1",
    "ROLE_KEY_PREFIX_V1",
    "SCHEMA_VERSION",
    "SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION",
    "SIGNER_REGISTRY_PREFIX_V2",
    "SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1",
    "SignedLogicalTimeTokenV1",
    "TOKEN_ID_PREFIX_V1",
    "VerifiedLogicalTimeChainV1",
    "build_logical_time_request_v1",
    "derive_step5_logical_time_role_key_id_v1",
    "load_logical_time_ledger_checkpoint_v1",
    "load_logical_time_ledger_head_v1",
    "load_logical_time_request_v1",
    "load_signed_logical_time_token_v1",
    "phase2b_step5_logical_time_checkpoint_id_preimage_v1",
    "phase2b_step5_logical_time_head_id_preimage_v1",
    "phase2b_step5_logical_time_request_id_preimage_v1",
    "phase2b_step5_logical_time_token_id_preimage_v1",
    "phase2b_step5_logical_time_token_signature_preimage_v1",
    "require_verified_logical_time_chain_v1",
    "verify_logical_time_ledger_transition_v1",
    "verify_logical_time_ledger_replay_v1",
    "verify_signed_logical_time_token_v1",
]

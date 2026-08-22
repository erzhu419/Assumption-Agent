"""Generic planned CAS mechanics for prospective Phase-2B Step-5 authority.

This module is deliberately mechanics-only.  It implements an exact
retention-plan wire contract, a generic one-FD ``SCM_RIGHTS`` request, durable
content-addressed IF_ABSENT storage, Ed25519 CAS_STORE receipts, immutable
checkpoint/head/response records, and producer-free pure replay of a complete
presented session.  It does not start a listener, allocate an attempt, prove a
UID/control-domain boundary, protect a private key, pin a running image, or
provide an external anti-rollback anchor.

This still-uncommitted store/2 candidate keeps its existing wire field shapes
while replacing its provisional authority/profile V1 dependency with the
frozen authority/profile V2 lock graph.  That dependency replacement adds no
authority claim and does not make any static V2 service binding live evidence.

The service-side import graph intentionally contains no auditor, QV2,
qualification-runtime, owner-status, or V1-store module.  Authority bootstrap
starts from raw Stage-B/profile/registry bytes and independently supplied
Stage-B pins on every construction or public verification path.
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
import weakref
from typing import Final, Mapping, NoReturn

from . import phase2b_formal_wire_covert_step5_authority_v2 as authority
from .rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)

try:  # Import remains possible; every crypto operation still fails closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - deployment-image dependent.
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = "hegel-machine-phase2b-formal-wire-covert-step5-store/2"
RETENTION_PLAN_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-retention-plan/1"
)
GENERIC_STORE_REQUEST_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-generic-scm-request/1"
)
TRANSITION_BINDING_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-transition-binding/1"
)
STORE_RECEIPT_BODY_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-receipt-body/2"
)
SIGNED_STORE_RECEIPT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-signed-cas-store-receipt/2"
)
STORE_CHECKPOINT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-checkpoint/2"
)
STORE_HEAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-head/2"
)
STORE_RESPONSE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-cas-store-response/2"
)

CAS_STORE_ROLE: Final = "CAS_STORE"
CAS_STORE_CONTROL_DOMAIN: Final = "STEP5_CAS_STORE_CONTROL"
CAS_STORE_RECEIPT_PURPOSE: Final = "SIGN_CAS_IF_ABSENT_RECEIPT"
CAS_STORE_SIGNATURE_DOMAIN: Final = (
    "HEGEL_MACHINE_PHASE2B_FORMAL_WIRE_COVERT_STEP5_CAS_STORE_RECEIPT_V1"
)
CAS_STORE_CHECKPOINT_PURPOSE: Final = "CAS_STORE_APPEND_ONLY_CHECKPOINT_V2"
CAS_STORE_HEAD_PURPOSE: Final = "CAS_STORE_CONTENT_DERIVED_HEAD_V2"
CAS_STORE_RESPONSE_PURPOSE: Final = "CAS_STORE_DURABLE_RETENTION_RESPONSE_V2"
KEY_EPOCH: Final = 1

AUTHORITY_CLAIM_BOUNDARY: Final = (
    "CAS_STORE_V2_MECHANICS_ONLY_SAME_PROCESS_SAME_UID_LOCAL_FILESYSTEM_"
    "DECLARATIVE_SOURCE_PINS_AND_STATIC_AUTHORITY_V2_SERVICE_BINDINGS_NOT_LIVE_"
    "AND_PRESENTED_COMPLETE_REPLAY_ARE_NOT_AUTHORITY_"
    "NOT_IMAGE_OR_RUNTIME_CLOSURE_NOT_KEY_CUSTODY_NOT_ANTI_ROLLBACK_OR_ANTI_CLONE"
)
SOURCE_PIN_CLAIM_BOUNDARY: Final = (
    "SOURCE_SHA256_PINS_RECORD_INSPECTED_INPUTS_ONLY_AND_DO_NOT_AUTHENTICATE_"
    "THE_IMPORTED_OR_RUNNING_SOURCE_TREE"
)
FAIL_CLOSED_ROLLBACK_DISCLOSURE: Final = (
    "GAPS_FORKS_STALE_CURRENT_HEAD_AND_PARTIAL_APPENDS_FAIL_CLOSED_BUT_A_"
    "CONSISTENT_WHOLE_TREE_ROLLBACK_OR_CLONE_REQUIRES_AN_EXTERNAL_ANCHOR_TO_DETECT"
)
REQUIRED_EXTERNAL_CONTROLS: Final = (
    "PINNED_CAS_STORE_SOURCE_IMAGE_AND_RUNTIME_CLOSURE",
    "DISTINCT_LIVE_CAS_STORE_UID_AND_CONTROL_DOMAIN",
    "CAS_STORE_PRIVATE_KEY_CUSTODY_OUTSIDE_PRODUCER",
    "INDEPENDENT_DURABLE_STORAGE_AND_EXTERNAL_COMPLETE_HEAD_ANCHOR",
    "FRESH_ATTEMPT_CONTEXT_AND_RETENTION_PLAN_AUTHORIZATION",
    "PRODUCER_FREE_PURE_REPLAY_IN_A_PINNED_DISTINCT_RUNTIME",
)

# These are documentary inspection pins, never an import-time trust oracle.
INSPECTED_STORE_V1_SOURCE_SHA256: Final = (
    "d22f003771cb7fa378299b53382e556a4d7b723a678659962db1577b2517f5f0"
)
INSPECTED_AUTHORITY_V2_SOURCE_SHA256: Final = (
    "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86"
)
INSPECTED_OWNER_STATUS_V1_SOURCE_SHA256: Final = (
    "bba6736a855154442167bb89c4d6042708bf8dbb230c05802983d897e14d416d"
)

MAXIMUM_PLAN_BYTES: Final = 4 * 1_024 * 1_024
MAXIMUM_REQUEST_BYTES: Final = 128 * 1_024
MAXIMUM_RECEIPT_BYTES: Final = 1 * 1_024 * 1_024
MAXIMUM_CHECKPOINT_BYTES: Final = 2 * 1_024 * 1_024
MAXIMUM_HEAD_BYTES: Final = 1 * 1_024 * 1_024
MAXIMUM_RESPONSE_BYTES: Final = 5 * 1_024 * 1_024
MAXIMUM_PLAN_OBJECTS: Final = 65_536
MAXIMUM_PLANNED_OBJECT_BYTES: Final = 1 << 40
_MAXIMUM_SAFE_JSON_INTEGER: Final = (1 << 53) - 1
GENESIS_CHECKPOINT_SHA256: Final = "sha256:" + ("0" * 64)

PLAN_ID_PREFIX: Final = "phase2b_step5_retention_plan_v1_"
REQUEST_ID_PREFIX: Final = "phase2b_step5_cas_request_v1_"
HEAD_STATE_ID_PREFIX: Final = "phase2b_step5_cas_head_state_v2_"
_PLAN_ID_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/CAS/RETENTION_PLAN/V1\x00"
_REQUEST_ID_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/CAS/GENERIC_REQUEST/V1\x00"
_HEAD_STATE_ID_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/CAS/HEAD_STATE/V2\x00"
_SIGNATURE_PREFIX: Final = CAS_STORE_SIGNATURE_DOMAIN.encode("ascii") + b"\x00"

_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{7,511}")
_ARTIFACT_ID_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{7,1023}")
_SHA256_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_HEX32_RE: Final = re.compile(r"[0-9a-f]{64}")
_HEX64_RE: Final = re.compile(r"[0-9a-f]{128}")
_PROFILE_ID_RE: Final = re.compile(r"phase2b_step5_authority_profile_v2_[0-9a-f]{64}")
_REGISTRY_ID_RE: Final = re.compile(r"phase2b_step5_signer_registry_v2_[0-9a-f]{64}")
_ROLE_KEY_ID_RE: Final = re.compile(r"phase2b_step5_role_key_v1_[0-9a-f]{64}")
_PLAN_ID_RE: Final = re.compile(r"phase2b_step5_retention_plan_v1_[0-9a-f]{64}")
_REQUEST_ID_RE: Final = re.compile(r"phase2b_step5_cas_request_v1_[0-9a-f]{64}")
_HEAD_STATE_ID_RE: Final = re.compile(r"phase2b_step5_cas_head_state_v2_[0-9a-f]{64}")
_RECORD_NAME_RE: Final = re.compile(r"[0-9]{20}\.json")
_OBJECT_NAME_RE: Final = re.compile(r"[0-9a-f]{64}")

FAIL_INPUT: Final = "FAIL_STEP5_CAS_STORE_V2_INPUT"
FAIL_CANONICAL: Final = "FAIL_STEP5_CAS_STORE_V2_CANONICAL"
FAIL_SCHEMA: Final = "FAIL_STEP5_CAS_STORE_V2_SCHEMA"
FAIL_BINDING: Final = "FAIL_STEP5_CAS_STORE_V2_BINDING"
FAIL_AUTHORITY: Final = "FAIL_STEP5_CAS_STORE_V2_AUTHORITY"
FAIL_SIGNATURE: Final = "FAIL_STEP5_CAS_STORE_V2_SIGNATURE"
FAIL_FILESYSTEM: Final = "FAIL_STEP5_CAS_STORE_V2_FILESYSTEM"
FAIL_CHAIN: Final = "FAIL_STEP5_CAS_STORE_V2_CHAIN"
FAIL_INCOMPLETE: Final = "FAIL_STEP5_CAS_STORE_V2_INCOMPLETE"


class CasStoreV2Error(ValueError):
    """Stable fail-closed error for V2 CAS mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise CasStoreV2Error(code, detail)


def _canonical(value: object) -> bytes:
    try:
        return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise CasStoreV2Error(FAIL_CANONICAL, "value is outside exact RFC8785") from exc


def _decode(payload: bytes, maximum: int, label: str) -> object:
    if type(payload) is not bytes or not 0 < len(payload) <= maximum:
        _fail(FAIL_INPUT, f"{label} bytes are outside the frozen bound")
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(payload)
    except Exception as exc:
        raise CasStoreV2Error(FAIL_CANONICAL, f"{label} is not exact RFC8785 bytes") from exc
    restored = _restore_wire_integral_numbers(decoded)
    if _canonical(restored) != payload:
        _fail(FAIL_CANONICAL, f"{label} differs after exact integer restoration")
    return restored


def _restore_wire_integral_numbers(value: object) -> object:
    """Restore canonical binary64 JSON integers to exact Python ``int`` DTOs."""

    if type(value) is float and value.is_integer() and abs(value) <= _MAXIMUM_SAFE_JSON_INTEGER:
        return int(value)
    if type(value) is list:
        return [_restore_wire_integral_numbers(item) for item in value]
    if type(value) is dict:
        return {
            key: _restore_wire_integral_numbers(item)
            for key, item in value.items()
        }
    return value


def _exact_object(value: object, keys: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != keys:
        _fail(FAIL_SCHEMA, f"{label} exact field set differs")
    return value


def _wire_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
) -> int:
    # The RFC8785 decoder restores canonical JSON integer spellings as exact
    # ``int`` objects before DTO construction.  Direct DTO callers must meet
    # that same type contract; accepting ``1.0`` here would create a second,
    # non-wire-equivalent construction path and ``bool`` is deliberately not
    # an integer for this protocol.
    if type(value) is not int:
        _fail(FAIL_SCHEMA, f"{label} is not one exact JSON integer")
    result = value
    if not minimum <= result <= maximum:
        _fail(FAIL_SCHEMA, f"{label} is outside its frozen bound")
    return result


def _literal(value: object, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        _fail(FAIL_SCHEMA, f"{label} differs")
    return value


def _pattern(value: object, expression: re.Pattern[str], label: str) -> str:
    if type(value) is not str or expression.fullmatch(value) is None:
        _fail(FAIL_SCHEMA, f"{label} syntax differs")
    return value


def _identifier(value: object, label: str) -> str:
    return _pattern(value, _IDENTIFIER_RE, label)


def _artifact_id(value: object, label: str) -> str:
    return _pattern(value, _ARTIFACT_ID_RE, label)


def _sha256(value: object, label: str) -> str:
    return _pattern(value, _SHA256_RE, label)


def _sha256_bytes(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _content_id(prefix: str, domain: bytes, subject: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(subject)).hexdigest()


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(FAIL_SCHEMA, f"{label} must be an exact boolean")
    return value


_PLAN_OBJECT_KEYS: Final = frozenset({"ordinal", "artifact_kind", "maximum_bytes"})
_PLAN_KEYS: Final = frozenset(
    {
        "schema_version",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "plan_kind",
        "expected_object_count",
        "objects_in_exact_order",
        "complete_head_required",
        "plan_id",
    }
)
_REQUEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "plan_id",
        "request_id",
        "ordinal",
        "artifact_kind",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "reported_size_bytes",
        "fd_count",
    }
)


@dataclass(frozen=True, slots=True)
class PlannedObjectV1:
    ordinal: int
    artifact_kind: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        if type(self) is not PlannedObjectV1:
            _fail(FAIL_INPUT, "planned object must use exact DTO")
        _wire_integer(self.ordinal, "planned object ordinal", maximum=MAXIMUM_PLAN_OBJECTS - 1)
        _identifier(self.artifact_kind, "planned artifact kind")
        _wire_integer(
            self.maximum_bytes,
            "planned object maximum bytes",
            minimum=1,
            maximum=MAXIMUM_PLANNED_OBJECT_BYTES,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "artifact_kind": self.artifact_kind,
            "maximum_bytes": self.maximum_bytes,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "PlannedObjectV1":
        body = _exact_object(value, _PLAN_OBJECT_KEYS, "planned object")
        return cls(
            ordinal=_wire_integer(
                body["ordinal"], "planned object ordinal", maximum=MAXIMUM_PLAN_OBJECTS - 1
            ),
            artifact_kind=body["artifact_kind"],  # type: ignore[arg-type]
            maximum_bytes=_wire_integer(
                body["maximum_bytes"],
                "planned object maximum bytes",
                minimum=1,
                maximum=MAXIMUM_PLANNED_OBJECT_BYTES,
            ),
        )


@dataclass(frozen=True, slots=True)
class RetentionPlanV1:
    schema_version: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    retention_session_id: str
    chain_id: str
    plan_kind: str
    expected_object_count: int
    objects_in_exact_order: tuple[PlannedObjectV1, ...]
    complete_head_required: bool
    plan_id: str

    def __post_init__(self) -> None:
        if type(self) is not RetentionPlanV1:
            _fail(FAIL_INPUT, "retention plan must use exact DTO")
        _literal(self.schema_version, RETENTION_PLAN_SCHEMA_VERSION, "retention plan schema")
        _pattern(self.authority_profile_id, _PROFILE_ID_RE, "authority profile ID")
        _pattern(self.signer_registry_id, _REGISTRY_ID_RE, "signer registry ID")
        _identifier(self.attempt_context_id, "attempt context ID")
        _identifier(self.retention_session_id, "retention session ID")
        _identifier(self.chain_id, "CAS chain ID")
        _identifier(self.plan_kind, "retention plan kind")
        count = _wire_integer(
            self.expected_object_count,
            "expected object count",
            minimum=1,
            maximum=MAXIMUM_PLAN_OBJECTS,
        )
        if type(self.objects_in_exact_order) is not tuple or len(self.objects_in_exact_order) != count:
            _fail(FAIL_BINDING, "retention plan object cardinality differs")
        for index, item in enumerate(self.objects_in_exact_order):
            if type(item) is not PlannedObjectV1:
                _fail(FAIL_INPUT, "retention plan object must use exact DTO")
            item.__post_init__()
            if item.ordinal != index:
                _fail(FAIL_BINDING, "retention plan ordinals are not exact contiguous order")
        if _require_bool(self.complete_head_required, "complete-head requirement") is not True:
            _fail(FAIL_BINDING, "retention plan/1 requires an exact complete head")
        _pattern(self.plan_id, _PLAN_ID_RE, "retention plan ID")
        if self.plan_id != _content_id(PLAN_ID_PREFIX, _PLAN_ID_DOMAIN, self._subject()):
            _fail(FAIL_BINDING, "retention plan ID does not bind exact content")

    def _subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "chain_id": self.chain_id,
            "plan_kind": self.plan_kind,
            "expected_object_count": self.expected_object_count,
            "objects_in_exact_order": [item.to_mapping() for item in self.objects_in_exact_order],
            "complete_head_required": self.complete_head_required,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._subject(), "plan_id": self.plan_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_PLAN_BYTES:
            _fail(FAIL_INPUT, "retention plan exceeds byte bound")
        return payload

    def entry(self, ordinal: int) -> PlannedObjectV1:
        if type(ordinal) is not int or not 0 <= ordinal < self.expected_object_count:
            _fail(FAIL_BINDING, "ordinal is outside retention plan")
        return self.objects_in_exact_order[ordinal]

    @classmethod
    def from_mapping(cls, value: object) -> "RetentionPlanV1":
        body = _exact_object(value, _PLAN_KEYS, "retention plan")
        raw_objects = body["objects_in_exact_order"]
        if type(raw_objects) is not list:
            _fail(FAIL_SCHEMA, "retention plan objects must be one exact array")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            plan_kind=body["plan_kind"],  # type: ignore[arg-type]
            expected_object_count=_wire_integer(
                body["expected_object_count"],
                "expected object count",
                minimum=1,
                maximum=MAXIMUM_PLAN_OBJECTS,
            ),
            objects_in_exact_order=tuple(PlannedObjectV1.from_mapping(item) for item in raw_objects),
            complete_head_required=_require_bool(
                body["complete_head_required"], "complete-head requirement"
            ),
            plan_id=body["plan_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "RetentionPlanV1":
        value = cls.from_mapping(_decode(payload, MAXIMUM_PLAN_BYTES, "retention plan"))
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL, "retention plan DTO readback differs")
        return value


def build_retention_plan_v1(
    *,
    authority_profile_id: str,
    signer_registry_id: str,
    attempt_context_id: str,
    retention_session_id: str,
    chain_id: str,
    plan_kind: str,
    objects_in_exact_order: tuple[tuple[str, int], ...],
    complete_head_required: bool = True,
) -> bytes:
    """Construct canonical plan/1 bytes; this builder grants no authority."""

    if type(objects_in_exact_order) is not tuple or not objects_in_exact_order:
        _fail(FAIL_INPUT, "plan builder requires one nonempty exact tuple")
    planned: list[PlannedObjectV1] = []
    for ordinal, raw in enumerate(objects_in_exact_order):
        if type(raw) is not tuple or len(raw) != 2:
            _fail(FAIL_INPUT, "plan builder rows must be exact (kind, maximum) tuples")
        planned.append(PlannedObjectV1(ordinal, raw[0], raw[1]))
    subject = {
        "schema_version": RETENTION_PLAN_SCHEMA_VERSION,
        "authority_profile_id": authority_profile_id,
        "signer_registry_id": signer_registry_id,
        "attempt_context_id": attempt_context_id,
        "retention_session_id": retention_session_id,
        "chain_id": chain_id,
        "plan_kind": plan_kind,
        "expected_object_count": len(planned),
        "objects_in_exact_order": [item.to_mapping() for item in planned],
        "complete_head_required": complete_head_required,
    }
    plan = RetentionPlanV1(
        schema_version=RETENTION_PLAN_SCHEMA_VERSION,
        authority_profile_id=authority_profile_id,
        signer_registry_id=signer_registry_id,
        attempt_context_id=attempt_context_id,
        retention_session_id=retention_session_id,
        chain_id=chain_id,
        plan_kind=plan_kind,
        expected_object_count=len(planned),
        objects_in_exact_order=tuple(planned),
        complete_head_required=complete_head_required,
        plan_id=_content_id(PLAN_ID_PREFIX, _PLAN_ID_DOMAIN, subject),
    )
    return RetentionPlanV1.from_canonical_bytes(plan.canonical_bytes()).canonical_bytes()


def load_retention_plan_v1(payload: bytes) -> RetentionPlanV1:
    return RetentionPlanV1.from_canonical_bytes(payload)


@dataclass(frozen=True, slots=True)
class GenericStoreRequestV1:
    schema_version: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str
    retention_session_id: str
    chain_id: str
    plan_id: str
    request_id: str
    ordinal: int
    artifact_kind: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    reported_size_bytes: int
    fd_count: int

    def __post_init__(self) -> None:
        if type(self) is not GenericStoreRequestV1:
            _fail(FAIL_INPUT, "generic store request must use exact DTO")
        _literal(self.schema_version, GENERIC_STORE_REQUEST_SCHEMA_VERSION, "request schema")
        _pattern(self.authority_profile_id, _PROFILE_ID_RE, "request profile ID")
        _pattern(self.signer_registry_id, _REGISTRY_ID_RE, "request registry ID")
        _identifier(self.attempt_context_id, "request attempt context ID")
        _identifier(self.retention_session_id, "request retention session ID")
        _identifier(self.chain_id, "request chain ID")
        _pattern(self.plan_id, _PLAN_ID_RE, "request plan ID")
        _pattern(self.request_id, _REQUEST_ID_RE, "request ID")
        _wire_integer(self.ordinal, "request ordinal", maximum=MAXIMUM_PLAN_OBJECTS - 1)
        _identifier(self.artifact_kind, "request artifact kind")
        _artifact_id(self.bound_artifact_id, "bound artifact ID")
        _sha256(self.bound_artifact_sha256, "bound artifact SHA-256")
        _wire_integer(
            self.reported_size_bytes,
            "reported object size",
            minimum=1,
            maximum=MAXIMUM_PLANNED_OBJECT_BYTES,
        )
        if type(self.fd_count) is not int or self.fd_count != 1:
            _fail(FAIL_BINDING, "generic request must declare exactly one FD")
        if self.request_id != _content_id(REQUEST_ID_PREFIX, _REQUEST_ID_DOMAIN, self._subject()):
            _fail(FAIL_BINDING, "request ID does not bind exact request fields")

    def _subject(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "chain_id": self.chain_id,
            "plan_id": self.plan_id,
            "ordinal": self.ordinal,
            "artifact_kind": self.artifact_kind,
            "bound_artifact_id": self.bound_artifact_id,
            "bound_artifact_sha256": self.bound_artifact_sha256,
            "reported_size_bytes": self.reported_size_bytes,
            "fd_count": self.fd_count,
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self._subject(), "request_id": self.request_id}

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_REQUEST_BYTES:
            _fail(FAIL_INPUT, "generic request exceeds byte bound")
        return payload

    def require_plan(self, plan: RetentionPlanV1) -> None:
        if type(plan) is not RetentionPlanV1:
            _fail(FAIL_INPUT, "request validation requires exact plan DTO")
        plan.__post_init__()
        entry = plan.entry(self.ordinal)
        if (
            self.authority_profile_id != plan.authority_profile_id
            or self.signer_registry_id != plan.signer_registry_id
            or self.attempt_context_id != plan.attempt_context_id
            or self.retention_session_id != plan.retention_session_id
            or self.chain_id != plan.chain_id
            or self.plan_id != plan.plan_id
            or self.artifact_kind != entry.artifact_kind
            or self.reported_size_bytes > entry.maximum_bytes
        ):
            _fail(FAIL_BINDING, "generic request differs from exact retention plan")

    @classmethod
    def from_mapping(cls, value: object) -> "GenericStoreRequestV1":
        body = _exact_object(value, _REQUEST_KEYS, "generic store request")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            plan_id=body["plan_id"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
            ordinal=_wire_integer(body["ordinal"], "request ordinal", maximum=MAXIMUM_PLAN_OBJECTS - 1),
            artifact_kind=body["artifact_kind"],  # type: ignore[arg-type]
            bound_artifact_id=body["bound_artifact_id"],  # type: ignore[arg-type]
            bound_artifact_sha256=body["bound_artifact_sha256"],  # type: ignore[arg-type]
            reported_size_bytes=_wire_integer(
                body["reported_size_bytes"],
                "reported size",
                minimum=1,
                maximum=MAXIMUM_PLANNED_OBJECT_BYTES,
            ),
            fd_count=_wire_integer(body["fd_count"], "FD count", minimum=1, maximum=1),
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "GenericStoreRequestV1":
        value = cls.from_mapping(_decode(payload, MAXIMUM_REQUEST_BYTES, "generic request"))
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL, "generic request DTO readback differs")
        return value


def build_generic_store_request_v1(
    *,
    retention_plan_payload: bytes,
    ordinal: int,
    bound_artifact_id: str,
    bound_artifact_sha256: str,
    reported_size_bytes: int,
) -> bytes:
    """Construct canonical generic request/1 bytes from raw plan bytes."""

    plan = RetentionPlanV1.from_canonical_bytes(retention_plan_payload)
    entry = plan.entry(ordinal)
    subject = {
        "schema_version": GENERIC_STORE_REQUEST_SCHEMA_VERSION,
        "authority_profile_id": plan.authority_profile_id,
        "signer_registry_id": plan.signer_registry_id,
        "attempt_context_id": plan.attempt_context_id,
        "retention_session_id": plan.retention_session_id,
        "chain_id": plan.chain_id,
        "plan_id": plan.plan_id,
        "ordinal": ordinal,
        "artifact_kind": entry.artifact_kind,
        "bound_artifact_id": bound_artifact_id,
        "bound_artifact_sha256": bound_artifact_sha256,
        "reported_size_bytes": reported_size_bytes,
        "fd_count": 1,
    }
    request = GenericStoreRequestV1(
        **subject,
        request_id=_content_id(REQUEST_ID_PREFIX, _REQUEST_ID_DOMAIN, subject),
    )
    request.require_plan(plan)
    return GenericStoreRequestV1.from_canonical_bytes(request.canonical_bytes()).canonical_bytes()


def load_generic_store_request_v1(payload: bytes) -> GenericStoreRequestV1:
    return GenericStoreRequestV1.from_canonical_bytes(payload)


@dataclass(frozen=True, slots=True)
class _CasStoreAuthorityBindingV2:
    authority_profile_id: str
    signer_registry_id: str
    signer_role: str
    control_domain: str
    signature_purpose: str
    key_epoch: int
    signer_key_id: str
    public_key_bytes: bytes


def _cas_store_role_key_id(
    public_key: bytes,
    _derive_role_key_id=authority.phase2b_step5_role_key_id_v2,
) -> str:
    if type(public_key) is not bytes or len(public_key) != 32:
        _fail(FAIL_AUTHORITY, "CAS_STORE public key must be exact raw 32 bytes")
    row = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": CAS_STORE_ROLE,
        "control_domain": CAS_STORE_CONTROL_DOMAIN,
        "authorized_signature_purposes": [CAS_STORE_RECEIPT_PURPOSE],
        "algorithm": "Ed25519",
        "key_epoch": KEY_EPOCH,
        "public_key_hex": public_key.hex(),
    }
    try:
        return _derive_role_key_id(row)
    except Exception as exc:
        raise CasStoreV2Error(FAIL_AUTHORITY, "CAS_STORE role-key derivation failed") from exc


def _load_authority_binding_v2(
    *,
    stage_b_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    _load_lock=authority.load_step5_stage_b_trust_lock_v2,
    _load_profile=authority.load_step5_authority_profile_v2,
    _load_registry=authority.load_step5_signer_registry_v2,
    _require_graph=authority.require_locked_step5_authority_graph_v2,
    _role_order=authority.ROLE_ORDER_V2,
    _role_purposes=authority.ROLE_SIGNATURE_PURPOSES_V2,
    _service_order=authority.SERVICE_ORDER_V2,
) -> _CasStoreAuthorityBindingV2:
    """Bootstrap only through the authority module's public bytes-only loaders."""

    if (
        type(stage_b_lock_payload) is not bytes
        or type(authority_profile_payload) is not bytes
        or type(signer_registry_payload) is not bytes
    ):
        _fail(FAIL_INPUT, "authority bootstrap artifacts must be exact bytes")
    try:
        lock = _load_lock(
            stage_b_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        profile = _load_profile(
            authority_profile_payload, lock
        )
        registry = _load_registry(
            signer_registry_payload, lock, profile
        )
        _require_graph(lock, profile, registry)
    except Exception as exc:
        raise CasStoreV2Error(FAIL_AUTHORITY, "Stage-B/profile/registry bytes rejected") from exc

    profile_id = _pattern(profile.artifact_id, _PROFILE_ID_RE, "authority profile ID")
    registry_id = _pattern(registry.artifact_id, _REGISTRY_ID_RE, "signer registry ID")
    if (
        lock.body.get("authority_profile_id") != profile_id
        or lock.body.get("signer_registry_id") != registry_id
        or registry.body.get("authority_profile_id") != profile_id
    ):
        _fail(FAIL_AUTHORITY, "authority graph lineage is not closed")
    policy = profile.body.get("role_policy")
    if not isinstance(policy, Mapping) or (
        tuple(policy.get("roles_in_exact_order", ())) != _role_order
        or policy.get("role_count") != 8
        or policy.get("distinct_public_keys") is not True
        or policy.get("distinct_key_ids") is not True
        or policy.get("distinct_control_domains") is not True
        or policy.get("key_epoch") != KEY_EPOCH
        or policy.get("purpose_policy_version") != 2
        or tuple(policy.get("services_in_exact_order", ()))
        != _service_order
        or policy.get("service_binding_count") != 5
        or policy.get("distinct_service_uids_gids") is not True
        or policy.get("static_service_identity_only") is not True
    ):
        _fail(FAIL_AUTHORITY, "authority profile exact role policy differs")
    rows = registry.body.get("role_keys_in_exact_order")
    if type(rows) is not tuple or len(rows) != len(_role_order):
        _fail(FAIL_AUTHORITY, "signer registry exact role cardinality differs")
    for cascade_role in (
        "GOVERNANCE_ANCHOR",
        "SOURCE_RUNTIME_CUSTODIAN",
        "PURE_REPLAYER",
    ):
        cascade_row = rows[_role_order.index(cascade_role)]
        if (
            not isinstance(cascade_row, Mapping)
            or tuple(cascade_row.get("authorized_signature_purposes", ()))
            != _role_purposes[cascade_role]
        ):
            _fail(FAIL_AUTHORITY, "authority V2 purpose cascade differs")
    raw = rows[_role_order.index(CAS_STORE_ROLE)]
    if not isinstance(raw, Mapping):
        _fail(FAIL_AUTHORITY, "CAS_STORE registry row type differs")
    purposes = raw.get("authorized_signature_purposes")
    if type(purposes) is not tuple:
        _fail(FAIL_AUTHORITY, "CAS_STORE purpose array is not frozen exact sequence")
    public_hex = raw.get("public_key_hex")
    if (
        raw.get("schema_version") != "hegel-machine-phase2b-step5-role-key/1"
        or raw.get("role") != CAS_STORE_ROLE
        or raw.get("control_domain") != CAS_STORE_CONTROL_DOMAIN
        or purposes != (CAS_STORE_RECEIPT_PURPOSE,)
        or raw.get("algorithm") != "Ed25519"
        or raw.get("key_epoch") != KEY_EPOCH
        or type(public_hex) is not str
        or _HEX32_RE.fullmatch(public_hex) is None
    ):
        _fail(FAIL_AUTHORITY, "CAS_STORE registry policy/public key differs")
    public_key = bytes.fromhex(public_hex)
    key_id = _pattern(raw.get("key_id"), _ROLE_KEY_ID_RE, "CAS_STORE role-key ID")
    if key_id != _cas_store_role_key_id(public_key):
        _fail(FAIL_AUTHORITY, "CAS_STORE registry key ID/content differs")
    return _CasStoreAuthorityBindingV2(
        authority_profile_id=profile_id,
        signer_registry_id=registry_id,
        signer_role=CAS_STORE_ROLE,
        control_domain=CAS_STORE_CONTROL_DOMAIN,
        signature_purpose=CAS_STORE_RECEIPT_PURPOSE,
        key_epoch=KEY_EPOCH,
        signer_key_id=key_id,
        public_key_bytes=public_key,
    )


_TRANSITION_KEYS: Final = frozenset(
    {
        "schema_version",
        "signer_role",
        "signer_control_domain",
        "signature_purpose",
        "authority_profile_id",
        "signer_registry_id",
        "key_epoch",
        "signer_key_id",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "plan_id",
        "plan_kind",
        "request_id",
        "ordinal",
        "artifact_kind",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "object_id",
        "object_sha256",
        "object_size_bytes",
        "previous_checkpoint_sha256",
        "expected_object_count",
        "observed_object_count",
        "complete",
        "complete_head_required",
        "head_state_id",
    }
)


@dataclass(frozen=True, slots=True)
class StoreTransitionBindingV2:
    schema_version: str
    signer_role: str
    signer_control_domain: str
    signature_purpose: str
    authority_profile_id: str
    signer_registry_id: str
    key_epoch: int
    signer_key_id: str
    attempt_context_id: str
    retention_session_id: str
    chain_id: str
    plan_id: str
    plan_kind: str
    request_id: str
    ordinal: int
    artifact_kind: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    object_id: str
    object_sha256: str
    object_size_bytes: int
    previous_checkpoint_sha256: str
    expected_object_count: int
    observed_object_count: int
    complete: bool
    complete_head_required: bool
    head_state_id: str

    def __post_init__(self) -> None:
        if type(self) is not StoreTransitionBindingV2:
            _fail(FAIL_INPUT, "transition binding must use exact DTO")
        _literal(self.schema_version, TRANSITION_BINDING_SCHEMA_VERSION, "transition schema")
        _literal(self.signer_role, CAS_STORE_ROLE, "transition signer role")
        _literal(
            self.signer_control_domain,
            CAS_STORE_CONTROL_DOMAIN,
            "transition signer control domain",
        )
        _literal(self.signature_purpose, CAS_STORE_RECEIPT_PURPOSE, "transition purpose")
        _pattern(self.authority_profile_id, _PROFILE_ID_RE, "transition profile ID")
        _pattern(self.signer_registry_id, _REGISTRY_ID_RE, "transition registry ID")
        if type(self.key_epoch) is not int or self.key_epoch != KEY_EPOCH:
            _fail(FAIL_BINDING, "transition key epoch differs")
        _pattern(self.signer_key_id, _ROLE_KEY_ID_RE, "transition signer key ID")
        _identifier(self.attempt_context_id, "transition attempt context ID")
        _identifier(self.retention_session_id, "transition retention session ID")
        _identifier(self.chain_id, "transition chain ID")
        _pattern(self.plan_id, _PLAN_ID_RE, "transition plan ID")
        _identifier(self.plan_kind, "transition plan kind")
        _pattern(self.request_id, _REQUEST_ID_RE, "transition request ID")
        ordinal = _wire_integer(
            self.ordinal, "transition ordinal", maximum=MAXIMUM_PLAN_OBJECTS - 1
        )
        _identifier(self.artifact_kind, "transition artifact kind")
        _artifact_id(self.bound_artifact_id, "transition bound artifact ID")
        _sha256(self.bound_artifact_sha256, "transition bound artifact SHA-256")
        _sha256(self.object_id, "transition object ID")
        _sha256(self.object_sha256, "transition object SHA-256")
        _wire_integer(
            self.object_size_bytes,
            "transition object size",
            minimum=1,
            maximum=MAXIMUM_PLANNED_OBJECT_BYTES,
        )
        _sha256(self.previous_checkpoint_sha256, "transition predecessor checkpoint")
        expected = _wire_integer(
            self.expected_object_count,
            "transition expected object count",
            minimum=1,
            maximum=MAXIMUM_PLAN_OBJECTS,
        )
        observed = _wire_integer(
            self.observed_object_count,
            "transition observed object count",
            minimum=1,
            maximum=MAXIMUM_PLAN_OBJECTS,
        )
        complete = _require_bool(self.complete, "transition complete flag")
        if _require_bool(self.complete_head_required, "transition complete-head flag") is not True:
            _fail(FAIL_BINDING, "transition does not require an exact complete head")
        if (
            self.object_id != self.object_sha256
            or self.object_sha256 != self.bound_artifact_sha256
        ):
            _fail(FAIL_BINDING, "actual object ID/SHA differs from bound artifact SHA")
        if observed != ordinal + 1 or observed > expected or complete is not (observed == expected):
            _fail(FAIL_BINDING, "transition ordinal/count/complete linkage differs")
        if ordinal == 0:
            if self.previous_checkpoint_sha256 != GENESIS_CHECKPOINT_SHA256:
                _fail(FAIL_CHAIN, "first transition predecessor is not genesis")
        elif self.previous_checkpoint_sha256 == GENESIS_CHECKPOINT_SHA256:
            _fail(FAIL_CHAIN, "non-genesis transition uses genesis predecessor")
        _pattern(self.head_state_id, _HEAD_STATE_ID_RE, "content-derived head state ID")
        if self.head_state_id != _content_id(
            HEAD_STATE_ID_PREFIX, _HEAD_STATE_ID_DOMAIN, self._head_subject()
        ):
            _fail(FAIL_BINDING, "head state ID does not bind exact transition content")

    def _head_subject(self) -> dict[str, object]:
        result = self.to_mapping()
        del result["head_state_id"]
        return result

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "signer_role": self.signer_role,
            "signer_control_domain": self.signer_control_domain,
            "signature_purpose": self.signature_purpose,
            "authority_profile_id": self.authority_profile_id,
            "signer_registry_id": self.signer_registry_id,
            "key_epoch": self.key_epoch,
            "signer_key_id": self.signer_key_id,
            "attempt_context_id": self.attempt_context_id,
            "retention_session_id": self.retention_session_id,
            "chain_id": self.chain_id,
            "plan_id": self.plan_id,
            "plan_kind": self.plan_kind,
            "request_id": self.request_id,
            "ordinal": self.ordinal,
            "artifact_kind": self.artifact_kind,
            "bound_artifact_id": self.bound_artifact_id,
            "bound_artifact_sha256": self.bound_artifact_sha256,
            "object_id": self.object_id,
            "object_sha256": self.object_sha256,
            "object_size_bytes": self.object_size_bytes,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "expected_object_count": self.expected_object_count,
            "observed_object_count": self.observed_object_count,
            "complete": self.complete,
            "complete_head_required": self.complete_head_required,
            "head_state_id": self.head_state_id,
        }

    def require_authority(self, binding: _CasStoreAuthorityBindingV2) -> None:
        if type(binding) is not _CasStoreAuthorityBindingV2 or (
            self.signer_role != binding.signer_role
            or self.signer_control_domain != binding.control_domain
            or self.signature_purpose != binding.signature_purpose
            or self.authority_profile_id != binding.authority_profile_id
            or self.signer_registry_id != binding.signer_registry_id
            or self.key_epoch != binding.key_epoch
            or self.signer_key_id != binding.signer_key_id
        ):
            _fail(FAIL_AUTHORITY, "transition differs from pinned CAS_STORE authority")

    def to_request(self) -> GenericStoreRequestV1:
        return GenericStoreRequestV1(
            schema_version=GENERIC_STORE_REQUEST_SCHEMA_VERSION,
            authority_profile_id=self.authority_profile_id,
            signer_registry_id=self.signer_registry_id,
            attempt_context_id=self.attempt_context_id,
            retention_session_id=self.retention_session_id,
            chain_id=self.chain_id,
            plan_id=self.plan_id,
            request_id=self.request_id,
            ordinal=self.ordinal,
            artifact_kind=self.artifact_kind,
            bound_artifact_id=self.bound_artifact_id,
            bound_artifact_sha256=self.bound_artifact_sha256,
            reported_size_bytes=self.object_size_bytes,
            fd_count=1,
        )

    def require_plan(self, plan: RetentionPlanV1) -> None:
        request = self.to_request()
        request.require_plan(plan)
        if (
            self.plan_kind != plan.plan_kind
            or self.expected_object_count != plan.expected_object_count
            or self.complete_head_required != plan.complete_head_required
        ):
            _fail(FAIL_BINDING, "transition plan summary differs")

    @classmethod
    def from_mapping(cls, value: object) -> "StoreTransitionBindingV2":
        body = _exact_object(value, _TRANSITION_KEYS, "transition binding")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            signer_role=body["signer_role"],  # type: ignore[arg-type]
            signer_control_domain=body["signer_control_domain"],  # type: ignore[arg-type]
            signature_purpose=body["signature_purpose"],  # type: ignore[arg-type]
            authority_profile_id=body["authority_profile_id"],  # type: ignore[arg-type]
            signer_registry_id=body["signer_registry_id"],  # type: ignore[arg-type]
            key_epoch=_wire_integer(body["key_epoch"], "transition key epoch", minimum=1, maximum=1),
            signer_key_id=body["signer_key_id"],  # type: ignore[arg-type]
            attempt_context_id=body["attempt_context_id"],  # type: ignore[arg-type]
            retention_session_id=body["retention_session_id"],  # type: ignore[arg-type]
            chain_id=body["chain_id"],  # type: ignore[arg-type]
            plan_id=body["plan_id"],  # type: ignore[arg-type]
            plan_kind=body["plan_kind"],  # type: ignore[arg-type]
            request_id=body["request_id"],  # type: ignore[arg-type]
            ordinal=_wire_integer(body["ordinal"], "transition ordinal", maximum=MAXIMUM_PLAN_OBJECTS - 1),
            artifact_kind=body["artifact_kind"],  # type: ignore[arg-type]
            bound_artifact_id=body["bound_artifact_id"],  # type: ignore[arg-type]
            bound_artifact_sha256=body["bound_artifact_sha256"],  # type: ignore[arg-type]
            object_id=body["object_id"],  # type: ignore[arg-type]
            object_sha256=body["object_sha256"],  # type: ignore[arg-type]
            object_size_bytes=_wire_integer(
                body["object_size_bytes"],
                "transition object size",
                minimum=1,
                maximum=MAXIMUM_PLANNED_OBJECT_BYTES,
            ),
            previous_checkpoint_sha256=body["previous_checkpoint_sha256"],  # type: ignore[arg-type]
            expected_object_count=_wire_integer(
                body["expected_object_count"],
                "transition expected count",
                minimum=1,
                maximum=MAXIMUM_PLAN_OBJECTS,
            ),
            observed_object_count=_wire_integer(
                body["observed_object_count"],
                "transition observed count",
                minimum=1,
                maximum=MAXIMUM_PLAN_OBJECTS,
            ),
            complete=_require_bool(body["complete"], "transition complete flag"),
            complete_head_required=_require_bool(
                body["complete_head_required"], "transition complete-head flag"
            ),
            head_state_id=body["head_state_id"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_request(
        cls,
        request: GenericStoreRequestV1,
        *,
        plan: RetentionPlanV1,
        authority_binding: _CasStoreAuthorityBindingV2,
        actual_sha256: str,
        actual_size_bytes: int,
        previous_checkpoint_sha256: str,
    ) -> "StoreTransitionBindingV2":
        request.require_plan(plan)
        if request.bound_artifact_sha256 != actual_sha256 or request.reported_size_bytes != actual_size_bytes:
            _fail(FAIL_BINDING, "actual object differs from request binding")
        subject = {
            "schema_version": TRANSITION_BINDING_SCHEMA_VERSION,
            "signer_role": authority_binding.signer_role,
            "signer_control_domain": authority_binding.control_domain,
            "signature_purpose": authority_binding.signature_purpose,
            "authority_profile_id": authority_binding.authority_profile_id,
            "signer_registry_id": authority_binding.signer_registry_id,
            "key_epoch": authority_binding.key_epoch,
            "signer_key_id": authority_binding.signer_key_id,
            "attempt_context_id": request.attempt_context_id,
            "retention_session_id": request.retention_session_id,
            "chain_id": request.chain_id,
            "plan_id": request.plan_id,
            "plan_kind": plan.plan_kind,
            "request_id": request.request_id,
            "ordinal": request.ordinal,
            "artifact_kind": request.artifact_kind,
            "bound_artifact_id": request.bound_artifact_id,
            "bound_artifact_sha256": request.bound_artifact_sha256,
            "object_id": actual_sha256,
            "object_sha256": actual_sha256,
            "object_size_bytes": actual_size_bytes,
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
            "expected_object_count": plan.expected_object_count,
            "observed_object_count": request.ordinal + 1,
            "complete": request.ordinal + 1 == plan.expected_object_count,
            "complete_head_required": plan.complete_head_required,
        }
        value = cls(
            **subject,
            head_state_id=_content_id(HEAD_STATE_ID_PREFIX, _HEAD_STATE_ID_DOMAIN, subject),
        )
        value.require_authority(authority_binding)
        value.require_plan(plan)
        return value


_RECEIPT_BODY_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "signature_domain",
        "binding",
        "peer_pid",
        "peer_uid",
        "peer_gid",
        "object_disposition",
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
        "binding",
        "signed_body",
        "signed_body_sha256",
        "signer_public_key_hex",
        "ed25519_signature_hex",
    }
)


@dataclass(frozen=True, slots=True)
class StoreReceiptBodyV2:
    schema_version: str
    purpose: str
    signature_domain: str
    binding: StoreTransitionBindingV2
    peer_pid: int
    peer_uid: int
    peer_gid: int
    object_disposition: str
    durable_before_return: bool
    readback_verified: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if type(self) is not StoreReceiptBodyV2:
            _fail(FAIL_INPUT, "receipt body must use exact DTO")
        _literal(self.schema_version, STORE_RECEIPT_BODY_SCHEMA_VERSION, "receipt body schema")
        _literal(self.purpose, CAS_STORE_RECEIPT_PURPOSE, "receipt body purpose")
        _literal(self.signature_domain, CAS_STORE_SIGNATURE_DOMAIN, "receipt signature domain")
        if type(self.binding) is not StoreTransitionBindingV2:
            _fail(FAIL_INPUT, "receipt body binding must use exact DTO")
        self.binding.__post_init__()
        _wire_integer(self.peer_pid, "receipt peer PID", minimum=1)
        _wire_integer(self.peer_uid, "receipt peer UID")
        _wire_integer(self.peer_gid, "receipt peer GID")
        if type(self.object_disposition) is not str or self.object_disposition not in {
            "CREATED_IF_ABSENT",
            "PRESENT_VERIFIED",
        }:
            _fail(FAIL_BINDING, "receipt object disposition differs")
        if (
            _require_bool(self.durable_before_return, "receipt durability") is not True
            or _require_bool(self.readback_verified, "receipt readback") is not True
        ):
            _fail(FAIL_BINDING, "receipt lacks durable verified readback")
        _literal(self.claim_boundary, AUTHORITY_CLAIM_BOUNDARY, "receipt claim boundary")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "signature_domain": self.signature_domain,
            "binding": self.binding.to_mapping(),
            "peer_pid": self.peer_pid,
            "peer_uid": self.peer_uid,
            "peer_gid": self.peer_gid,
            "object_disposition": self.object_disposition,
            "durable_before_return": self.durable_before_return,
            "readback_verified": self.readback_verified,
            "claim_boundary": self.claim_boundary,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_RECEIPT_BYTES:
            _fail(FAIL_INPUT, "receipt body exceeds byte bound")
        return payload

    @classmethod
    def from_mapping(cls, value: object) -> "StoreReceiptBodyV2":
        body = _exact_object(value, _RECEIPT_BODY_KEYS, "receipt body")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            signature_domain=body["signature_domain"],  # type: ignore[arg-type]
            binding=StoreTransitionBindingV2.from_mapping(body["binding"]),
            peer_pid=_wire_integer(body["peer_pid"], "receipt peer PID", minimum=1),
            peer_uid=_wire_integer(body["peer_uid"], "receipt peer UID"),
            peer_gid=_wire_integer(body["peer_gid"], "receipt peer GID"),
            object_disposition=body["object_disposition"],  # type: ignore[arg-type]
            durable_before_return=_require_bool(body["durable_before_return"], "receipt durability"),
            readback_verified=_require_bool(body["readback_verified"], "receipt readback"),
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SignedStoreReceiptV2:
    schema_version: str
    purpose: str
    signature_domain: str
    binding: StoreTransitionBindingV2
    signed_body: StoreReceiptBodyV2
    signed_body_sha256: str
    signer_public_key_hex: str
    ed25519_signature_hex: str

    def __post_init__(self) -> None:
        if type(self) is not SignedStoreReceiptV2:
            _fail(FAIL_INPUT, "signed receipt must use exact DTO")
        _literal(self.schema_version, SIGNED_STORE_RECEIPT_SCHEMA_VERSION, "signed receipt schema")
        _literal(self.purpose, CAS_STORE_RECEIPT_PURPOSE, "signed receipt purpose")
        _literal(self.signature_domain, CAS_STORE_SIGNATURE_DOMAIN, "signed receipt domain")
        if type(self.binding) is not StoreTransitionBindingV2 or type(self.signed_body) is not StoreReceiptBodyV2:
            _fail(FAIL_INPUT, "signed receipt nested records must use exact DTOs")
        self.binding.__post_init__()
        self.signed_body.__post_init__()
        if self.binding != self.signed_body.binding:
            _fail(FAIL_BINDING, "signed receipt/body transition bindings differ")
        if self.signed_body_sha256 != _sha256_bytes(self.signed_body.canonical_bytes()):
            _fail(FAIL_BINDING, "signed receipt body hash differs")
        _pattern(self.signer_public_key_hex, _HEX32_RE, "receipt signer public key")
        _pattern(self.ed25519_signature_hex, _HEX64_RE, "receipt Ed25519 signature")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "signature_domain": self.signature_domain,
            "binding": self.binding.to_mapping(),
            "signed_body": self.signed_body.to_mapping(),
            "signed_body_sha256": self.signed_body_sha256,
            "signer_public_key_hex": self.signer_public_key_hex,
            "ed25519_signature_hex": self.ed25519_signature_hex,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_RECEIPT_BYTES:
            _fail(FAIL_INPUT, "signed receipt exceeds byte bound")
        return payload

    @property
    def receipt_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_mapping(cls, value: object) -> "SignedStoreReceiptV2":
        body = _exact_object(value, _SIGNED_RECEIPT_KEYS, "signed receipt")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            signature_domain=body["signature_domain"],  # type: ignore[arg-type]
            binding=StoreTransitionBindingV2.from_mapping(body["binding"]),
            signed_body=StoreReceiptBodyV2.from_mapping(body["signed_body"]),
            signed_body_sha256=body["signed_body_sha256"],  # type: ignore[arg-type]
            signer_public_key_hex=body["signer_public_key_hex"],  # type: ignore[arg-type]
            ed25519_signature_hex=body["ed25519_signature_hex"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "SignedStoreReceiptV2":
        value = cls.from_mapping(_decode(payload, MAXIMUM_RECEIPT_BYTES, "signed receipt"))
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL, "signed receipt DTO readback differs")
        return value


def _receipt_signature_preimage(body: StoreReceiptBodyV2) -> bytes:
    if type(body) is not StoreReceiptBodyV2:
        _fail(FAIL_INPUT, "receipt signature requires exact body DTO")
    body.__post_init__()
    return _SIGNATURE_PREFIX + body.canonical_bytes()


def _build_fixed_ed25519_receipt_verifier():
    """Capture the concrete verification backend before module globals can drift."""

    backend_loader = (
        None
        if _Ed25519PublicKey is None
        else _Ed25519PublicKey.from_public_bytes
    )
    invalid_signature_type = _InvalidSignature
    signature_preimage = _receipt_signature_preimage
    fail = _fail

    def verify_signed_receipt(
        receipt: SignedStoreReceiptV2,
        authority_binding: _CasStoreAuthorityBindingV2,
    ) -> StoreReceiptBodyV2:
        if type(receipt) is not SignedStoreReceiptV2:
            fail(FAIL_INPUT, "receipt verification requires exact DTO")
        receipt.__post_init__()
        receipt.binding.require_authority(authority_binding)
        if receipt.signer_public_key_hex != authority_binding.public_key_bytes.hex():
            fail(FAIL_AUTHORITY, "receipt public key differs from pinned registry")
        if backend_loader is None or invalid_signature_type is None:
            fail(FAIL_SIGNATURE, "cryptography Ed25519 verifier is unavailable")
        try:
            backend_loader(authority_binding.public_key_bytes).verify(
                bytes.fromhex(receipt.ed25519_signature_hex),
                signature_preimage(receipt.signed_body),
            )
        except invalid_signature_type as exc:
            raise CasStoreV2Error(
                FAIL_SIGNATURE, "CAS_STORE Ed25519 receipt rejected"
            ) from exc
        return receipt.signed_body

    return verify_signed_receipt


_verify_signed_receipt = _build_fixed_ed25519_receipt_verifier()
del _build_fixed_ed25519_receipt_verifier


def verify_signed_store_receipt_v2(
    *,
    signed_receipt_payload: bytes,
    stage_b_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
) -> StoreReceiptBodyV2:
    """Bytes-only fixed-backend receipt verification; no callback is accepted."""

    binding = _load_authority_binding_v2(
        stage_b_lock_payload=stage_b_lock_payload,
        authority_profile_payload=authority_profile_payload,
        signer_registry_payload=signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_lock_payload_sha256_digest=(
            out_of_band_expected_lock_payload_sha256_digest
        ),
    )
    receipt = SignedStoreReceiptV2.from_canonical_bytes(signed_receipt_payload)
    return _verify_signed_receipt(receipt, binding)


class _Ed25519CasStoreSignerV2:
    __slots__ = ("_private_key", "_authority_binding")

    def __init__(self, private_key_seed: bytes, binding: _CasStoreAuthorityBindingV2) -> None:
        if type(private_key_seed) is not bytes or len(private_key_seed) != 32:
            _fail(FAIL_INPUT, "CAS_STORE private seed must be exact 32 bytes")
        if _Ed25519PrivateKey is None or _serialization is None:
            _fail(FAIL_SIGNATURE, "cryptography Ed25519 signer is unavailable")
        private_key = _Ed25519PrivateKey.from_private_bytes(private_key_seed)
        public_key = private_key.public_key().public_bytes(
            encoding=_serialization.Encoding.Raw,
            format=_serialization.PublicFormat.Raw,
        )
        if public_key != binding.public_key_bytes or _cas_store_role_key_id(public_key) != binding.signer_key_id:
            _fail(FAIL_AUTHORITY, "CAS_STORE private seed differs from registry key")
        self._private_key = private_key
        self._authority_binding = binding

    def sign(self, body: StoreReceiptBodyV2) -> SignedStoreReceiptV2:
        body.binding.require_authority(self._authority_binding)
        receipt = SignedStoreReceiptV2(
            schema_version=SIGNED_STORE_RECEIPT_SCHEMA_VERSION,
            purpose=CAS_STORE_RECEIPT_PURPOSE,
            signature_domain=CAS_STORE_SIGNATURE_DOMAIN,
            binding=body.binding,
            signed_body=body,
            signed_body_sha256=_sha256_bytes(body.canonical_bytes()),
            signer_public_key_hex=self._authority_binding.public_key_bytes.hex(),
            ed25519_signature_hex=self._private_key.sign(_receipt_signature_preimage(body)).hex(),
        )
        _verify_signed_receipt(receipt, self._authority_binding)
        return receipt


_CHECKPOINT_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "binding",
        "signed_receipt_sha256",
        "signed_receipt",
        "claim_boundary",
    }
)
_HEAD_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "binding",
        "checkpoint_sha256",
        "signed_receipt_sha256",
        "claim_boundary",
        "rollback_disclosure",
    }
)
_RESPONSE_KEYS: Final = frozenset(
    {
        "schema_version",
        "purpose",
        "binding",
        "checkpoint_sha256",
        "checkpoint_record",
        "head_sha256",
        "head_record",
        "durable_before_return",
        "readback_verified",
        "claim_boundary",
    }
)


@dataclass(frozen=True, slots=True)
class StoreCheckpointV2:
    schema_version: str
    purpose: str
    binding: StoreTransitionBindingV2
    signed_receipt_sha256: str
    signed_receipt: SignedStoreReceiptV2
    claim_boundary: str

    def __post_init__(self) -> None:
        if type(self) is not StoreCheckpointV2:
            _fail(FAIL_INPUT, "checkpoint must use exact DTO")
        _literal(self.schema_version, STORE_CHECKPOINT_SCHEMA_VERSION, "checkpoint schema")
        _literal(self.purpose, CAS_STORE_CHECKPOINT_PURPOSE, "checkpoint purpose")
        if type(self.binding) is not StoreTransitionBindingV2 or type(self.signed_receipt) is not SignedStoreReceiptV2:
            _fail(FAIL_INPUT, "checkpoint nested records must use exact DTOs")
        self.binding.__post_init__()
        self.signed_receipt.__post_init__()
        _sha256(self.signed_receipt_sha256, "checkpoint receipt SHA-256")
        if (
            self.binding != self.signed_receipt.binding
            or self.signed_receipt_sha256 != self.signed_receipt.receipt_sha256
        ):
            _fail(FAIL_BINDING, "checkpoint/signed-receipt linkage differs")
        _literal(self.claim_boundary, AUTHORITY_CLAIM_BOUNDARY, "checkpoint claim boundary")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "binding": self.binding.to_mapping(),
            "signed_receipt_sha256": self.signed_receipt_sha256,
            "signed_receipt": self.signed_receipt.to_mapping(),
            "claim_boundary": self.claim_boundary,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_CHECKPOINT_BYTES:
            _fail(FAIL_INPUT, "checkpoint exceeds byte bound")
        return payload

    @property
    def checkpoint_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_mapping(cls, value: object) -> "StoreCheckpointV2":
        body = _exact_object(value, _CHECKPOINT_KEYS, "checkpoint")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            binding=StoreTransitionBindingV2.from_mapping(body["binding"]),
            signed_receipt_sha256=body["signed_receipt_sha256"],  # type: ignore[arg-type]
            signed_receipt=SignedStoreReceiptV2.from_mapping(body["signed_receipt"]),
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "StoreCheckpointV2":
        value = cls.from_mapping(_decode(payload, MAXIMUM_CHECKPOINT_BYTES, "checkpoint"))
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL, "checkpoint DTO readback differs")
        return value


@dataclass(frozen=True, slots=True)
class StoreHeadV2:
    schema_version: str
    purpose: str
    binding: StoreTransitionBindingV2
    checkpoint_sha256: str
    signed_receipt_sha256: str
    claim_boundary: str
    rollback_disclosure: str

    def __post_init__(self) -> None:
        if type(self) is not StoreHeadV2:
            _fail(FAIL_INPUT, "head must use exact DTO")
        _literal(self.schema_version, STORE_HEAD_SCHEMA_VERSION, "head schema")
        _literal(self.purpose, CAS_STORE_HEAD_PURPOSE, "head purpose")
        if type(self.binding) is not StoreTransitionBindingV2:
            _fail(FAIL_INPUT, "head binding must use exact DTO")
        self.binding.__post_init__()
        _sha256(self.checkpoint_sha256, "head checkpoint SHA-256")
        _sha256(self.signed_receipt_sha256, "head receipt SHA-256")
        _literal(self.claim_boundary, AUTHORITY_CLAIM_BOUNDARY, "head claim boundary")
        _literal(
            self.rollback_disclosure,
            FAIL_CLOSED_ROLLBACK_DISCLOSURE,
            "head rollback disclosure",
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "binding": self.binding.to_mapping(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "signed_receipt_sha256": self.signed_receipt_sha256,
            "claim_boundary": self.claim_boundary,
            "rollback_disclosure": self.rollback_disclosure,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_HEAD_BYTES:
            _fail(FAIL_INPUT, "head exceeds byte bound")
        return payload

    @property
    def head_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_mapping(cls, value: object) -> "StoreHeadV2":
        body = _exact_object(value, _HEAD_KEYS, "head")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            binding=StoreTransitionBindingV2.from_mapping(body["binding"]),
            checkpoint_sha256=body["checkpoint_sha256"],  # type: ignore[arg-type]
            signed_receipt_sha256=body["signed_receipt_sha256"],  # type: ignore[arg-type]
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
            rollback_disclosure=body["rollback_disclosure"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "StoreHeadV2":
        value = cls.from_mapping(_decode(payload, MAXIMUM_HEAD_BYTES, "head"))
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL, "head DTO readback differs")
        return value


@dataclass(frozen=True, slots=True)
class StoreResponseV2:
    schema_version: str
    purpose: str
    binding: StoreTransitionBindingV2
    checkpoint_sha256: str
    checkpoint_record: StoreCheckpointV2
    head_sha256: str
    head_record: StoreHeadV2
    durable_before_return: bool
    readback_verified: bool
    claim_boundary: str

    def __post_init__(self) -> None:
        if type(self) is not StoreResponseV2:
            _fail(FAIL_INPUT, "response must use exact DTO")
        _literal(self.schema_version, STORE_RESPONSE_SCHEMA_VERSION, "response schema")
        _literal(self.purpose, CAS_STORE_RESPONSE_PURPOSE, "response purpose")
        if (
            type(self.binding) is not StoreTransitionBindingV2
            or type(self.checkpoint_record) is not StoreCheckpointV2
            or type(self.head_record) is not StoreHeadV2
        ):
            _fail(FAIL_INPUT, "response nested records must use exact DTOs")
        self.binding.__post_init__()
        self.checkpoint_record.__post_init__()
        self.head_record.__post_init__()
        _sha256(self.checkpoint_sha256, "response checkpoint SHA-256")
        _sha256(self.head_sha256, "response head SHA-256")
        if (
            self.binding != self.checkpoint_record.binding
            or self.binding != self.head_record.binding
            or self.checkpoint_sha256 != self.checkpoint_record.checkpoint_sha256
            or self.head_sha256 != self.head_record.head_sha256
            or self.head_record.checkpoint_sha256 != self.checkpoint_sha256
            or self.head_record.signed_receipt_sha256
            != self.checkpoint_record.signed_receipt_sha256
        ):
            _fail(FAIL_BINDING, "response/checkpoint/head linkage differs")
        if (
            _require_bool(self.durable_before_return, "response durability") is not True
            or _require_bool(self.readback_verified, "response readback") is not True
        ):
            _fail(FAIL_BINDING, "response lacks durable verified readback")
        _literal(self.claim_boundary, AUTHORITY_CLAIM_BOUNDARY, "response claim boundary")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "binding": self.binding.to_mapping(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_record": self.checkpoint_record.to_mapping(),
            "head_sha256": self.head_sha256,
            "head_record": self.head_record.to_mapping(),
            "durable_before_return": self.durable_before_return,
            "readback_verified": self.readback_verified,
            "claim_boundary": self.claim_boundary,
        }

    def canonical_bytes(self) -> bytes:
        payload = _canonical(self.to_mapping())
        if len(payload) > MAXIMUM_RESPONSE_BYTES:
            _fail(FAIL_INPUT, "response exceeds byte bound")
        return payload

    @property
    def response_sha256(self) -> str:
        return _sha256_bytes(self.canonical_bytes())

    @classmethod
    def from_mapping(cls, value: object) -> "StoreResponseV2":
        body = _exact_object(value, _RESPONSE_KEYS, "response")
        return cls(
            schema_version=body["schema_version"],  # type: ignore[arg-type]
            purpose=body["purpose"],  # type: ignore[arg-type]
            binding=StoreTransitionBindingV2.from_mapping(body["binding"]),
            checkpoint_sha256=body["checkpoint_sha256"],  # type: ignore[arg-type]
            checkpoint_record=StoreCheckpointV2.from_mapping(body["checkpoint_record"]),
            head_sha256=body["head_sha256"],  # type: ignore[arg-type]
            head_record=StoreHeadV2.from_mapping(body["head_record"]),
            durable_before_return=_require_bool(body["durable_before_return"], "response durability"),
            readback_verified=_require_bool(body["readback_verified"], "response readback"),
            claim_boundary=body["claim_boundary"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> "StoreResponseV2":
        value = cls.from_mapping(_decode(payload, MAXIMUM_RESPONSE_BYTES, "response"))
        if value.canonical_bytes() != payload:
            _fail(FAIL_CANONICAL, "response DTO readback differs")
        return value


def load_store_checkpoint_v2(payload: bytes) -> StoreCheckpointV2:
    return StoreCheckpointV2.from_canonical_bytes(payload)


def load_store_head_v2(payload: bytes) -> StoreHeadV2:
    return StoreHeadV2.from_canonical_bytes(payload)


def load_store_response_v2(payload: bytes) -> StoreResponseV2:
    return StoreResponseV2.from_canonical_bytes(payload)


def _verify_record_triplet(
    *,
    checkpoint: StoreCheckpointV2,
    head: StoreHeadV2,
    response: StoreResponseV2,
    plan: RetentionPlanV1,
    authority_binding: _CasStoreAuthorityBindingV2,
) -> StoreTransitionBindingV2:
    body = _verify_signed_receipt(checkpoint.signed_receipt, authority_binding)
    binding = checkpoint.binding
    binding.require_authority(authority_binding)
    binding.require_plan(plan)
    if (
        body.binding != binding
        or head.binding != binding
        or response.binding != binding
        or response.checkpoint_record != checkpoint
        or response.head_record != head
        or head.checkpoint_sha256 != checkpoint.checkpoint_sha256
        or head.signed_receipt_sha256 != checkpoint.signed_receipt_sha256
        or response.checkpoint_sha256 != checkpoint.checkpoint_sha256
        or response.head_sha256 != head.head_sha256
    ):
        _fail(FAIL_BINDING, "record triplet exact linkage differs")
    return binding


@dataclass(frozen=True, slots=True)
class PeerCredentialsV2:
    pid: int
    uid: int
    gid: int

    def __post_init__(self) -> None:
        if type(self) is not PeerCredentialsV2:
            _fail(FAIL_INPUT, "peer credentials must use exact DTO")
        _wire_integer(self.pid, "peer PID", minimum=1)
        _wire_integer(self.uid, "peer UID")
        _wire_integer(self.gid, "peer GID")


@dataclass(slots=True)
class ReceivedStoreRequestV2:
    request: GenericStoreRequestV1
    fd: int
    peer: PeerCredentialsV2
    _closed: bool = False

    def __post_init__(self) -> None:
        if type(self) is not ReceivedStoreRequestV2:
            _fail(FAIL_INPUT, "received request must use exact DTO")
        if type(self.request) is not GenericStoreRequestV1 or type(self.peer) is not PeerCredentialsV2:
            _fail(FAIL_INPUT, "received request nested DTO differs")
        self.request.__post_init__()
        self.peer.__post_init__()
        if type(self.fd) is not int or self.fd < 0:
            _fail(FAIL_INPUT, "received descriptor is invalid")

    def close(self) -> None:
        if not self._closed:
            os.close(self.fd)
            self._closed = True

    def __enter__(self) -> "ReceivedStoreRequestV2":
        if self._closed:
            _fail(FAIL_INPUT, "received descriptor is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


def _require_linux_mechanics() -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(socket, "SO_PEERCRED"):
        _fail(FAIL_INPUT, "CAS_STORE V2 requires Linux O_NOFOLLOW and SO_PEERCRED")


def _require_seqpacket(connection: socket.socket) -> None:
    if type(connection) is not socket.socket:
        _fail(FAIL_INPUT, "SCM transport requires exact socket type")
    if connection.family != socket.AF_UNIX:
        _fail(FAIL_INPUT, "SCM transport requires AF_UNIX")
    if connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET:
        _fail(FAIL_INPUT, "SCM transport requires SOCK_SEQPACKET")


def _peer_credentials(connection: socket.socket) -> PeerCredentialsV2:
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    if type(raw) is not bytes or len(raw) != 12:
        _fail(FAIL_INPUT, "SO_PEERCRED record differs")
    pid, uid, gid = struct.unpack("3i", raw)
    return PeerCredentialsV2(pid=pid, uid=uid, gid=gid)


def _require_uid_set(value: object) -> frozenset[int]:
    if (
        type(value) is not frozenset
        or not value
        or any(type(item) is not int or item < 0 for item in value)
    ):
        _fail(FAIL_INPUT, "allowed peer UIDs must be one nonempty exact frozenset")
    return value


def recv_store_request_v2(
    connection: socket.socket,
    *,
    allowed_peer_uids: frozenset[int],
) -> ReceivedStoreRequestV2:
    """Receive exactly one request/1 packet and exactly one SCM_RIGHTS FD."""

    _require_linux_mechanics()
    _require_seqpacket(connection)
    allowed = _require_uid_set(allowed_peer_uids)
    peer = _peer_credentials(connection)
    if peer.uid not in allowed:
        raise PermissionError("SO_PEERCRED UID is not authorized for CAS_STORE V2")
    descriptor_size = array("i").itemsize
    payload, ancillary, flags, address = connection.recvmsg(
        MAXIMUM_REQUEST_BYTES + 1,
        socket.CMSG_SPACE(descriptor_size * 2),
        getattr(socket, "MSG_CMSG_CLOEXEC", 0),
    )
    received_fds: list[int] = []
    try:
        for level, kind, data in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                _fail(FAIL_INPUT, "SCM packet contains unexpected control record")
            if len(data) % descriptor_size:
                _fail(FAIL_INPUT, "SCM_RIGHTS descriptor alignment differs")
            values = array("i")
            values.frombytes(data)
            received_fds.extend(int(item) for item in values)
        if flags & (getattr(socket, "MSG_TRUNC", 0) | getattr(socket, "MSG_CTRUNC", 0)):
            _fail(FAIL_INPUT, "SCM request or control record was truncated")
        if address not in {None, "", b""}:
            _fail(FAIL_INPUT, "connected SCM packet contains source address")
        if len(received_fds) != 1:
            _fail(FAIL_INPUT, "SCM request must carry exactly one descriptor")
        request = GenericStoreRequestV1.from_canonical_bytes(payload)
        descriptor = received_fds.pop()
        os.set_inheritable(descriptor, False)
        return ReceivedStoreRequestV2(request=request, fd=descriptor, peer=peer)
    finally:
        for descriptor in received_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass


def send_store_request_fd_v2(
    connection: socket.socket,
    request: GenericStoreRequestV1,
    fd: int,
) -> None:
    _require_seqpacket(connection)
    if type(request) is not GenericStoreRequestV1:
        _fail(FAIL_INPUT, "SCM sender requires exact generic request DTO")
    request.__post_init__()
    if type(fd) is not int or fd < 0:
        _fail(FAIL_INPUT, "SCM sender descriptor is invalid")
    payload = request.canonical_bytes()
    descriptors = array("i", [fd])
    sent = connection.sendmsg(
        [payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())]
    )
    if sent != len(payload):
        raise OSError("SCM request send was not one complete packet")


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


def _require_private_regular(snapshot: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISREG(snapshot.st_mode)
        or snapshot.st_nlink != 1
        or snapshot.st_uid != os.geteuid()
        or snapshot.st_mode & 0o077
    ):
        _fail(FAIL_FILESYSTEM, f"{label} is not one owner-private regular link")


def _read_regular_at(directory_fd: int, basename: str, maximum: int) -> bytes:
    fd = os.open(
        basename,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(fd)
        _require_private_regular(before, basename)
        if not 0 < before.st_size <= maximum:
            _fail(FAIL_FILESYSTEM, f"{basename} size is outside bound")
        blocks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block:
                _fail(FAIL_FILESYSTEM, f"{basename} ended during readback")
            blocks.append(block)
            offset += len(block)
        if os.pread(fd, 1, offset):
            _fail(FAIL_FILESYSTEM, f"{basename} grew during readback")
        if not _same_file_identity(before, os.fstat(fd)):
            _fail(FAIL_FILESYSTEM, f"{basename} changed during readback")
        return b"".join(blocks)
    finally:
        os.close(fd)


def _hash_regular_at(
    directory_fd: int,
    basename: str,
    maximum: int,
) -> tuple[str, int]:
    fd = os.open(
        basename,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(fd)
        _require_private_regular(before, basename)
        if not 0 < before.st_size <= maximum:
            _fail(FAIL_FILESYSTEM, f"{basename} object size is outside plan")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block:
                _fail(FAIL_FILESYSTEM, f"{basename} object ended during hash")
            digest.update(block)
            offset += len(block)
        if os.pread(fd, 1, offset) or not _same_file_identity(before, os.fstat(fd)):
            _fail(FAIL_FILESYSTEM, f"{basename} object changed during hash")
        return "sha256:" + digest.hexdigest(), offset
    finally:
        os.close(fd)


def _write_immutable_at(
    directory_fd: int,
    basename: str,
    payload: bytes,
    maximum: int,
) -> None:
    if type(payload) is not bytes or not 0 < len(payload) <= maximum:
        _fail(FAIL_INPUT, "immutable record bytes are outside bound")
    fd = os.open(
        basename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(directory_fd)
    if _read_regular_at(directory_fd, basename, maximum) != payload:
        _fail(FAIL_FILESYSTEM, "immutable record readback differs")


def _atomic_replace_at(
    directory_fd: int,
    basename: str,
    payload: bytes,
    maximum: int,
    nonce: int,
) -> None:
    temporary = f".{basename}.pending-{os.getpid()}-{nonce}"
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary, basename, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        if _read_regular_at(directory_fd, basename, maximum) != payload:
            _fail(FAIL_FILESYSTEM, "atomic current-head readback differs")
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _record_name(index: int) -> str:
    _wire_integer(index, "record index", maximum=MAXIMUM_PLAN_OBJECTS - 1)
    name = f"{index:020d}.json"
    if _RECORD_NAME_RE.fullmatch(name) is None:
        _fail(FAIL_SCHEMA, "record filename overflow")
    return name


@dataclass(frozen=True, slots=True)
class _SessionRecordV2:
    checkpoint: StoreCheckpointV2
    head: StoreHeadV2
    response: StoreResponseV2


class CasStoreV2:
    """Generic planned durable CAS mechanics; no listener is started here."""

    __slots__ = (
        "root_directory",
        "retention_plan",
        "allowed_peer_uids",
        "authority_profile_id",
        "signer_registry_id",
        "signer_key_id",
        "signer_public_key",
        "key_epoch",
        "_authority_binding",
        "_signer",
        "_root_fd",
        "_objects_fd",
        "_staging_fd",
        "_checkpoints_fd",
        "_heads_fd",
        "_responses_fd",
        "_state_fd",
        "_lock_fd",
        "_thread_lock",
        "_nonce",
        "_closed",
    )

    _DIRECTORIES: Final = (
        "objects",
        "staging",
        "checkpoints",
        "heads",
        "responses",
        "state",
    )
    _ROOT_ENTRIES: Final = frozenset((*_DIRECTORIES, "store.lock", "plan.json"))

    def __init__(
        self,
        *,
        root_directory: str,
        retention_plan_payload: bytes,
        private_key_seed: bytes,
        allowed_peer_uids: frozenset[int],
        stage_b_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
        create: bool = False,
    ) -> None:
        _require_linux_mechanics()
        if type(root_directory) is not str or not os.path.isabs(root_directory):
            _fail(FAIL_INPUT, "CAS_STORE root must be one absolute path")
        if os.path.abspath(root_directory) != root_directory:
            _fail(FAIL_INPUT, "CAS_STORE root path is not lexically canonical")
        if type(create) is not bool:
            _fail(FAIL_INPUT, "create flag must be exact boolean")
        plan = RetentionPlanV1.from_canonical_bytes(retention_plan_payload)
        authority_binding = _load_authority_binding_v2(
            stage_b_lock_payload=stage_b_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        if (
            plan.authority_profile_id != authority_binding.authority_profile_id
            or plan.signer_registry_id != authority_binding.signer_registry_id
        ):
            _fail(FAIL_AUTHORITY, "retention plan differs from bootstrapped authority graph")
        self.root_directory = root_directory
        self.retention_plan = plan
        self.allowed_peer_uids = _require_uid_set(allowed_peer_uids)
        self.authority_profile_id = authority_binding.authority_profile_id
        self.signer_registry_id = authority_binding.signer_registry_id
        self.signer_key_id = authority_binding.signer_key_id
        self.signer_public_key = authority_binding.public_key_bytes
        self.key_epoch = authority_binding.key_epoch
        self._authority_binding = authority_binding
        self._signer = _Ed25519CasStoreSignerV2(private_key_seed, authority_binding)
        self._root_fd = -1
        self._objects_fd = -1
        self._staging_fd = -1
        self._checkpoints_fd = -1
        self._heads_fd = -1
        self._responses_fd = -1
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
                _fail(FAIL_FILESYSTEM, "CAS_STORE root resolves through a symlink")
            self._root_fd = os.open(
                root_directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            self._require_private_directory(self._root_fd, "CAS_STORE root")
            initial = frozenset(os.listdir(self._root_fd))
            initialize = not initial
            if initialize and not create:
                _fail(FAIL_FILESYSTEM, "uninitialized CAS_STORE root requires create=True")
            if not initialize and initial != self._ROOT_ENTRIES:
                _fail(FAIL_FILESYSTEM, "CAS_STORE root exact entry set differs")
            if initialize:
                for basename in self._DIRECTORIES:
                    os.mkdir(basename, 0o700, dir_fd=self._root_fd)
                self._lock_fd = os.open(
                    "store.lock",
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._root_fd,
                )
                _write_immutable_at(
                    self._root_fd,
                    "plan.json",
                    retention_plan_payload,
                    MAXIMUM_PLAN_BYTES,
                )
                os.fsync(self._root_fd)
            else:
                self._lock_fd = os.open(
                    "store.lock",
                    os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=self._root_fd,
                )
            self._require_lock_file()
            self._objects_fd = self._open_directory("objects")
            self._staging_fd = self._open_directory("staging")
            self._checkpoints_fd = self._open_directory("checkpoints")
            self._heads_fd = self._open_directory("heads")
            self._responses_fd = self._open_directory("responses")
            self._state_fd = self._open_directory("state")
            if frozenset(os.listdir(self._root_fd)) != self._ROOT_ENTRIES:
                _fail(FAIL_FILESYSTEM, "CAS_STORE root changed during bootstrap")
            if _read_regular_at(self._root_fd, "plan.json", MAXIMUM_PLAN_BYTES) != retention_plan_payload:
                _fail(FAIL_BINDING, "persisted retention plan exact bytes differ")
            with self._exclusive_lock():
                self._audit_chain_locked()
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
            _fail(FAIL_FILESYSTEM, f"{label} is not owner-private directory")

    def _open_directory(self, basename: str) -> int:
        fd = os.open(
            basename,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._root_fd,
        )
        try:
            self._require_private_directory(fd, f"CAS_STORE {basename}")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _require_lock_file(self) -> None:
        observed = os.fstat(self._lock_fd)
        _require_private_regular(observed, "CAS_STORE lock")
        if observed.st_size != 0:
            _fail(FAIL_FILESYSTEM, "CAS_STORE lock file is not empty")
        os.fsync(self._lock_fd)

    class _ExclusiveLock:
        def __init__(self, owner: "CasStoreV2") -> None:
            self.owner = owner

        def __enter__(self) -> None:
            if self.owner._closed or self.owner._lock_fd < 0:
                _fail(FAIL_FILESYSTEM, "CAS_STORE is closed")
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

    def _exclusive_lock(self) -> "CasStoreV2._ExclusiveLock":
        return self._ExclusiveLock(self)

    def _require_empty_staging(self) -> None:
        if tuple(os.listdir(self._staging_fd)):
            _fail(
                FAIL_FILESYSTEM,
                "staging is nonempty; partial append requires explicit operator review",
            )

    def _record_names(self, fd: int, label: str) -> tuple[str, ...]:
        names = tuple(sorted(os.listdir(fd)))
        for index, name in enumerate(names):
            if _RECORD_NAME_RE.fullmatch(name) is None or name != _record_name(index):
                _fail(FAIL_CHAIN, f"{label} filenames contain a gap, fork, or unknown entry")
        return names

    def _audit_chain_locked(self) -> tuple[_SessionRecordV2, ...]:
        self._require_empty_staging()
        checkpoint_names = self._record_names(self._checkpoints_fd, "checkpoint")
        head_names = self._record_names(self._heads_fd, "head")
        response_names = self._record_names(self._responses_fd, "response")
        if not (len(checkpoint_names) == len(head_names) == len(response_names)):
            _fail(FAIL_CHAIN, "checkpoint/head/response append cardinalities differ")
        if len(checkpoint_names) > self.retention_plan.expected_object_count:
            _fail(FAIL_CHAIN, "record count exceeds exact retention plan")
        state_names = frozenset(os.listdir(self._state_fd))
        expected_state = frozenset() if not checkpoint_names else frozenset({"current-head.json"})
        if state_names != expected_state:
            _fail(FAIL_CHAIN, "current-head state is missing, stale-shaped, or forked")

        records: list[_SessionRecordV2] = []
        previous = GENESIS_CHECKPOINT_SHA256
        request_ids: set[str] = set()
        expected_objects: dict[str, int] = {}
        for index, (checkpoint_name, head_name, response_name) in enumerate(
            zip(checkpoint_names, head_names, response_names, strict=True)
        ):
            checkpoint = StoreCheckpointV2.from_canonical_bytes(
                _read_regular_at(self._checkpoints_fd, checkpoint_name, MAXIMUM_CHECKPOINT_BYTES)
            )
            head = StoreHeadV2.from_canonical_bytes(
                _read_regular_at(self._heads_fd, head_name, MAXIMUM_HEAD_BYTES)
            )
            response = StoreResponseV2.from_canonical_bytes(
                _read_regular_at(self._responses_fd, response_name, MAXIMUM_RESPONSE_BYTES)
            )
            binding = _verify_record_triplet(
                checkpoint=checkpoint,
                head=head,
                response=response,
                plan=self.retention_plan,
                authority_binding=self._authority_binding,
            )
            if (
                binding.ordinal != index
                or binding.previous_checkpoint_sha256 != previous
                or binding.request_id in request_ids
            ):
                _fail(FAIL_CHAIN, "record order, predecessor, or request uniqueness differs")
            request_ids.add(binding.request_id)
            previous = checkpoint.checkpoint_sha256
            basename = binding.object_id[7:]
            expected_objects[basename] = max(
                expected_objects.get(basename, 0), self.retention_plan.entry(index).maximum_bytes
            )
            records.append(_SessionRecordV2(checkpoint, head, response))

        object_names = frozenset(os.listdir(self._objects_fd))
        if any(_OBJECT_NAME_RE.fullmatch(name) is None for name in object_names):
            _fail(FAIL_FILESYSTEM, "object namespace contains unknown entry")
        if object_names != frozenset(expected_objects):
            _fail(FAIL_CHAIN, "object namespace is not exact referenced content set")
        verified_objects: dict[str, tuple[str, int]] = {}
        for basename, maximum in expected_objects.items():
            observed = _hash_regular_at(self._objects_fd, basename, maximum)
            if observed[0] != "sha256:" + basename:
                _fail(FAIL_FILESYSTEM, "object namespace/content address differs")
            verified_objects[basename] = observed
        for record in records:
            binding = record.checkpoint.binding
            if verified_objects[binding.object_id[7:]] != (
                binding.object_sha256,
                binding.object_size_bytes,
            ):
                _fail(FAIL_FILESYSTEM, "record-referenced object readback differs")
        if records:
            current = _read_regular_at(self._state_fd, "current-head.json", MAXIMUM_HEAD_BYTES)
            if current != records[-1].head.canonical_bytes():
                _fail(FAIL_CHAIN, "current head is not exact latest immutable head")
        return tuple(records)

    def _hash_received_fd(
        self,
        request: GenericStoreRequestV1,
        fd: int,
    ) -> tuple[os.stat_result, str, int]:
        request.require_plan(self.retention_plan)
        if type(fd) is not int or fd < 0:
            _fail(FAIL_INPUT, "CAS ingest requires one valid descriptor")
        if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
            _fail(FAIL_INPUT, "SCM_RIGHTS artifact descriptor is not read-only")
        before = os.fstat(fd)
        maximum = self.retention_plan.entry(request.ordinal).maximum_bytes
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            _fail(FAIL_INPUT, "SCM_RIGHTS artifact is not one bounded regular link")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(fd, min(1_048_576, before.st_size - offset), offset)
            if not block:
                _fail(FAIL_INPUT, "SCM_RIGHTS artifact ended during independent hash")
            digest.update(block)
            offset += len(block)
        if os.pread(fd, 1, offset) or not _same_file_identity(before, os.fstat(fd)):
            _fail(FAIL_INPUT, "SCM_RIGHTS artifact changed during independent hash")
        sha = "sha256:" + digest.hexdigest()
        if (
            sha != request.bound_artifact_sha256
            or offset != request.reported_size_bytes
        ):
            _fail(FAIL_BINDING, "actual FD hash/size differs from generic request")
        return before, sha, offset

    def _persist_object_if_absent(
        self,
        source_fd: int,
        snapshot: os.stat_result,
        sha256: str,
        size_bytes: int,
    ) -> str:
        basename = sha256[7:]
        if _OBJECT_NAME_RE.fullmatch(basename) is None:
            _fail(FAIL_BINDING, "object content-address basename differs")
        self._require_empty_staging()
        self._nonce += 1
        temporary = f"pending-{os.getpid()}-{self._nonce}"
        temporary_fd: int | None = None
        disposition = "CREATED_IF_ABSENT"
        try:
            temporary_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._staging_fd,
            )
            digest = hashlib.sha256()
            offset = 0
            while offset < size_bytes:
                block = os.pread(source_fd, min(1_048_576, size_bytes - offset), offset)
                if not block:
                    _fail(FAIL_FILESYSTEM, "source artifact ended during CAS copy")
                digest.update(block)
                _write_all(temporary_fd, block)
                offset += len(block)
            if (
                os.pread(source_fd, 1, offset)
                or not _same_file_identity(snapshot, os.fstat(source_fd))
                or offset != size_bytes
                or "sha256:" + digest.hexdigest() != sha256
            ):
                _fail(FAIL_FILESYSTEM, "source artifact changed during CAS copy")
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
            observed = _hash_regular_at(self._objects_fd, basename, size_bytes)
            if observed != (sha256, size_bytes):
                _fail(FAIL_FILESYSTEM, "CAS object reopen/readback differs")
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

    def _exact_replay(
        self,
        request: GenericStoreRequestV1,
        fd: int,
        peer: PeerCredentialsV2,
        record: _SessionRecordV2,
    ) -> StoreResponseV2:
        body = record.checkpoint.signed_receipt.signed_body
        if request != record.checkpoint.binding.to_request() or peer != PeerCredentialsV2(
            body.peer_pid, body.peer_uid, body.peer_gid
        ):
            _fail(FAIL_CHAIN, "duplicate request replay metadata or peer differs")
        snapshot, sha, size = self._hash_received_fd(request, fd)
        stored_sha, stored_size = _hash_regular_at(
            self._objects_fd,
            record.checkpoint.binding.object_id[7:],
            self.retention_plan.entry(request.ordinal).maximum_bytes,
        )
        if (
            (sha, size) != (stored_sha, stored_size)
            or not _same_file_identity(snapshot, os.fstat(fd))
        ):
            _fail(FAIL_CHAIN, "duplicate request FD/object exact replay differs")
        return record.response

    def ingest_open_fd(
        self,
        request: GenericStoreRequestV1,
        fd: int,
        peer: PeerCredentialsV2,
    ) -> StoreResponseV2:
        if type(request) is not GenericStoreRequestV1 or type(peer) is not PeerCredentialsV2:
            _fail(FAIL_INPUT, "CAS ingest requires exact request and peer DTOs")
        request.__post_init__()
        request.require_plan(self.retention_plan)
        peer.__post_init__()
        if peer.uid not in self.allowed_peer_uids:
            raise PermissionError("peer UID is not authorized for CAS_STORE V2")
        with self._exclusive_lock():
            records = self._audit_chain_locked()
            duplicates = tuple(
                record for record in records if record.checkpoint.binding.request_id == request.request_id
            )
            if duplicates:
                if len(duplicates) != 1:
                    _fail(FAIL_CHAIN, "duplicate request cardinality differs")
                return self._exact_replay(request, fd, peer, duplicates[0])
            if request.ordinal != len(records):
                _fail(FAIL_CHAIN, "request ordinal is a gap, fork, or completed-session append")
            snapshot, actual_sha, actual_size = self._hash_received_fd(request, fd)
            disposition = self._persist_object_if_absent(
                fd, snapshot, actual_sha, actual_size
            )
            if not _same_file_identity(snapshot, os.fstat(fd)):
                _fail(FAIL_FILESYSTEM, "source FD changed after CAS persistence")
            previous = (
                GENESIS_CHECKPOINT_SHA256
                if not records
                else records[-1].checkpoint.checkpoint_sha256
            )
            binding = StoreTransitionBindingV2.from_request(
                request,
                plan=self.retention_plan,
                authority_binding=self._authority_binding,
                actual_sha256=actual_sha,
                actual_size_bytes=actual_size,
                previous_checkpoint_sha256=previous,
            )
            receipt_body = StoreReceiptBodyV2(
                schema_version=STORE_RECEIPT_BODY_SCHEMA_VERSION,
                purpose=CAS_STORE_RECEIPT_PURPOSE,
                signature_domain=CAS_STORE_SIGNATURE_DOMAIN,
                binding=binding,
                peer_pid=peer.pid,
                peer_uid=peer.uid,
                peer_gid=peer.gid,
                object_disposition=disposition,
                durable_before_return=True,
                readback_verified=True,
                claim_boundary=AUTHORITY_CLAIM_BOUNDARY,
            )
            receipt = self._signer.sign(receipt_body)
            checkpoint = StoreCheckpointV2(
                schema_version=STORE_CHECKPOINT_SCHEMA_VERSION,
                purpose=CAS_STORE_CHECKPOINT_PURPOSE,
                binding=binding,
                signed_receipt_sha256=receipt.receipt_sha256,
                signed_receipt=receipt,
                claim_boundary=AUTHORITY_CLAIM_BOUNDARY,
            )
            head = StoreHeadV2(
                schema_version=STORE_HEAD_SCHEMA_VERSION,
                purpose=CAS_STORE_HEAD_PURPOSE,
                binding=binding,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                signed_receipt_sha256=receipt.receipt_sha256,
                claim_boundary=AUTHORITY_CLAIM_BOUNDARY,
                rollback_disclosure=FAIL_CLOSED_ROLLBACK_DISCLOSURE,
            )
            response = StoreResponseV2(
                schema_version=STORE_RESPONSE_SCHEMA_VERSION,
                purpose=CAS_STORE_RESPONSE_PURPOSE,
                binding=binding,
                checkpoint_sha256=checkpoint.checkpoint_sha256,
                checkpoint_record=checkpoint,
                head_sha256=head.head_sha256,
                head_record=head,
                durable_before_return=True,
                readback_verified=True,
                claim_boundary=AUTHORITY_CLAIM_BOUNDARY,
            )
            name = _record_name(request.ordinal)
            _write_immutable_at(
                self._checkpoints_fd, name, checkpoint.canonical_bytes(), MAXIMUM_CHECKPOINT_BYTES
            )
            _write_immutable_at(self._heads_fd, name, head.canonical_bytes(), MAXIMUM_HEAD_BYTES)
            _write_immutable_at(
                self._responses_fd, name, response.canonical_bytes(), MAXIMUM_RESPONSE_BYTES
            )
            self._nonce += 1
            _atomic_replace_at(
                self._state_fd,
                "current-head.json",
                head.canonical_bytes(),
                MAXIMUM_HEAD_BYTES,
                self._nonce,
            )
            committed = self._audit_chain_locked()
            if len(committed) != request.ordinal + 1 or committed[-1].response != response:
                _fail(FAIL_CHAIN, "post-append exact audit differs")
            return response

    def receive_and_ingest_once(self, connection: socket.socket) -> StoreResponseV2:
        received = recv_store_request_v2(
            connection, allowed_peer_uids=self.allowed_peer_uids
        )
        with received:
            return self.ingest_open_fd(received.request, received.fd, received.peer)

    def handle_one_connection(self, connection: socket.socket) -> StoreResponseV2:
        response = self.receive_and_ingest_once(connection)
        payload = response.canonical_bytes()
        sent = connection.send(payload)
        if sent != len(payload):
            raise OSError("CAS_STORE response send was not one complete packet")
        return response

    def audit_chain(self) -> tuple[_SessionRecordV2, ...]:
        with self._exclusive_lock():
            return self._audit_chain_locked()

    def audit_complete_session(self) -> tuple[_SessionRecordV2, ...]:
        records = self.audit_chain()
        if (
            len(records) != self.retention_plan.expected_object_count
            or not records
            or records[-1].head.binding.complete is not True
            or records[-1].head.binding.observed_object_count
            != self.retention_plan.expected_object_count
        ):
            _fail(FAIL_INCOMPLETE, "CAS_STORE session lacks exact complete head")
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in (
            "_lock_fd",
            "_state_fd",
            "_responses_fd",
            "_heads_fd",
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

    def __enter__(self) -> "CasStoreV2":
        if self._closed:
            _fail(FAIL_FILESYSTEM, "CAS_STORE is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False


def _build_private_complete_session_mechanics() -> tuple[
    type[object], object, object
]:
    """Build the public result type and verifier around an unexported token.

    The capability and issuer live only in this closure.  In particular there
    is no module-global token or issuer helper that a same-process caller can
    simply import or rebind.  This is a KAT-strength construction boundary, not
    a claim of resistance to arbitrary Python runtime introspection or code
    mutation (which remains outside ``AUTHORITY_CLAIM_BOUNDARY``).
    """

    issuance_token = object()
    issued_lock = threading.Lock()
    issued_results: dict[
        int, tuple[weakref.ReferenceType[object], tuple[object, ...]]
    ] = {}
    immutable_field_names = (
        "verification_status",
        "authority_profile_id",
        "signer_registry_id",
        "signer_role",
        "signer_key_id",
        "key_epoch",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "plan_id",
        "plan_kind",
        "expected_object_count",
        "complete_head_state_id",
        "complete_head_sha256",
        "complete_checkpoint_sha256",
        "complete_response_sha256",
        "object_bindings_in_exact_order",
        "claim_boundary",
        "rollback_disclosure",
    )

    class _WeakReferenceableIssuedResult:
        __slots__ = ("__weakref__",)

    @dataclass(frozen=True, slots=True, init=False)
    class VerifiedCasStoreCompleteSessionV2(_WeakReferenceableIssuedResult):
        """Privately issued result of exact bytes-only complete-session replay."""

        verification_status: str
        authority_profile_id: str
        signer_registry_id: str
        signer_role: str
        signer_key_id: str
        key_epoch: int
        attempt_context_id: str
        retention_session_id: str
        chain_id: str
        plan_id: str
        plan_kind: str
        expected_object_count: int
        complete_head_state_id: str
        complete_head_sha256: str
        complete_checkpoint_sha256: str
        complete_response_sha256: str
        object_bindings_in_exact_order: tuple[tuple[object, ...], ...]
        claim_boundary: str
        rollback_disclosure: str

        def __new__(cls, token: object = None) -> "VerifiedCasStoreCompleteSessionV2":
            if token is not issuance_token:
                raise TypeError(
                    "verified CAS_STORE complete-session tokens are privately issued"
                )
            return object.__new__(cls)

        def __copy__(self) -> "VerifiedCasStoreCompleteSessionV2":
            raise TypeError("verified CAS_STORE results cannot be copied")

        def __deepcopy__(
            self, memo: object
        ) -> "VerifiedCasStoreCompleteSessionV2":
            raise TypeError("verified CAS_STORE results cannot be deep-copied")

    # Capture all public byte loaders and critical replay helpers at module
    # construction time.  Adding or rebinding similarly named module globals
    # later cannot turn the public verifier into a caller-supplied callback
    # surface or replace its private issuer.
    load_plan = RetentionPlanV1.from_canonical_bytes
    load_request = GenericStoreRequestV1.from_canonical_bytes
    load_receipt = SignedStoreReceiptV2.from_canonical_bytes
    load_checkpoint = StoreCheckpointV2.from_canonical_bytes
    load_head = StoreHeadV2.from_canonical_bytes
    load_response = StoreResponseV2.from_canonical_bytes
    load_authority = _load_authority_binding_v2
    verify_triplet = _verify_record_triplet
    verify_receipt = _verify_signed_receipt
    sha256_bytes = _sha256_bytes
    fail = _fail

    def immutable_equal(current: object, issued: object) -> bool:
        if type(current) is not type(issued):
            return False
        if type(current) is tuple:
            return len(current) == len(issued) and all(  # type: ignore[arg-type]
                immutable_equal(left, right)
                for left, right in zip(current, issued, strict=True)  # type: ignore[arg-type]
            )
        return current == issued

    def snapshot_result(
        result: VerifiedCasStoreCompleteSessionV2,
    ) -> tuple[object, ...]:
        try:
            return tuple(getattr(result, name) for name in immutable_field_names)
        except (AttributeError, TypeError) as exc:
            raise CasStoreV2Error(
                FAIL_BINDING, "verified complete-session result fields are missing"
            ) from exc

    def register_result(result: VerifiedCasStoreCompleteSessionV2) -> None:
        identity = id(result)

        def forget(reference: weakref.ReferenceType[object]) -> None:
            with issued_lock:
                current = issued_results.get(identity)
                if current is not None and current[0] is reference:
                    del issued_results[identity]

        reference = weakref.ref(result, forget)
        snapshot = snapshot_result(result)
        with issued_lock:
            if identity in issued_results:
                fail(FAIL_BINDING, "verified result identity unexpectedly reused")
            issued_results[identity] = (reference, snapshot)

    def require_verified_cas_store_complete_session_v2(
        candidate: object,
    ) -> VerifiedCasStoreCompleteSessionV2:
        """Require exact issued identity and every immutable result field."""

        if type(candidate) is not VerifiedCasStoreCompleteSessionV2:
            fail(FAIL_INPUT, "complete-session result has the wrong exact type")
        identity = id(candidate)
        with issued_lock:
            issued = issued_results.get(identity)
            if issued is None or issued[0]() is not candidate:
                fail(FAIL_INPUT, "complete-session result was not privately issued")
            frozen_snapshot = issued[1]
        current_snapshot = snapshot_result(candidate)
        if not immutable_equal(current_snapshot, frozen_snapshot):
            fail(FAIL_BINDING, "issued complete-session result was mutated")
        return candidate

    def issue_complete_result(
        *,
        plan: RetentionPlanV1,
        authority_binding: _CasStoreAuthorityBindingV2,
        records: tuple[
            tuple[StoreCheckpointV2, StoreHeadV2, StoreResponseV2], ...
        ],
    ) -> VerifiedCasStoreCompleteSessionV2:
        last_checkpoint, last_head, last_response = records[-1]
        result = VerifiedCasStoreCompleteSessionV2(issuance_token)
        values: dict[str, object] = {
            "verification_status": (
                "PRESENTED_COMPLETE_CAS_SESSION_CRYPTOGRAPHIC_AND_CONTENT_REPLAY_"
                "VERIFIED_MECHANICS_ONLY_NOT_AUTHORITY"
            ),
            "authority_profile_id": plan.authority_profile_id,
            "signer_registry_id": plan.signer_registry_id,
            "signer_role": authority_binding.signer_role,
            "signer_key_id": authority_binding.signer_key_id,
            "key_epoch": authority_binding.key_epoch,
            "attempt_context_id": plan.attempt_context_id,
            "retention_session_id": plan.retention_session_id,
            "chain_id": plan.chain_id,
            "plan_id": plan.plan_id,
            "plan_kind": plan.plan_kind,
            "expected_object_count": plan.expected_object_count,
            "complete_head_state_id": last_head.binding.head_state_id,
            "complete_head_sha256": last_head.head_sha256,
            "complete_checkpoint_sha256": last_checkpoint.checkpoint_sha256,
            "complete_response_sha256": last_response.response_sha256,
            "object_bindings_in_exact_order": tuple(
                (
                    checkpoint.binding.ordinal,
                    checkpoint.binding.artifact_kind,
                    checkpoint.binding.bound_artifact_id,
                    checkpoint.binding.bound_artifact_sha256,
                    checkpoint.binding.object_id,
                    checkpoint.binding.object_size_bytes,
                    checkpoint.signed_receipt_sha256,
                    checkpoint.checkpoint_sha256,
                    head.head_sha256,
                    response.response_sha256,
                )
                for checkpoint, head, response in records
            ),
            "claim_boundary": AUTHORITY_CLAIM_BOUNDARY,
            "rollback_disclosure": FAIL_CLOSED_ROLLBACK_DISCLOSURE,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        register_result(result)
        return require_verified_cas_store_complete_session_v2(result)

    def verify_cas_store_complete_session_v2(
        *,
        retention_plan_payload: bytes,
        request_payloads_in_exact_order: tuple[bytes, ...],
        object_payloads_in_exact_order: tuple[bytes, ...],
        signed_receipt_payloads_in_exact_order: tuple[bytes, ...],
        checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        head_payloads_in_exact_order: tuple[bytes, ...],
        response_payloads_in_exact_order: tuple[bytes, ...],
        stage_b_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_lock_payload_sha256_digest: bytes,
    ) -> VerifiedCasStoreCompleteSessionV2:
        """Purely replay one complete presented CAS session from exact bytes.

        No decoded artifact or verifier callback is accepted.  Every original
        request, object, signed receipt, checkpoint, head, and response byte
        string is independently parsed and matched to the embedded signed
        graph.  Success remains mechanics-only and makes no live-isolation,
        key-custody, freshness, or anti-rollback/clone claim.
        """

        plan = load_plan(retention_plan_payload)
        authority_binding = load_authority(
            stage_b_lock_payload=stage_b_lock_payload,
            authority_profile_payload=authority_profile_payload,
            signer_registry_payload=signer_registry_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_lock_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        if (
            plan.authority_profile_id != authority_binding.authority_profile_id
            or plan.signer_registry_id != authority_binding.signer_registry_id
        ):
            fail(FAIL_AUTHORITY, "complete-session plan differs from authority graph")
        sequences = (
            request_payloads_in_exact_order,
            object_payloads_in_exact_order,
            signed_receipt_payloads_in_exact_order,
            checkpoint_payloads_in_exact_order,
            head_payloads_in_exact_order,
            response_payloads_in_exact_order,
        )
        if any(type(sequence) is not tuple for sequence in sequences):
            fail(FAIL_INPUT, "complete-session artifacts must be exact tuples of bytes")
        expected = plan.expected_object_count
        if any(len(sequence) != expected for sequence in sequences):
            fail(FAIL_INCOMPLETE, "complete-session artifact cardinality differs from plan")
        if any(
            type(payload) is not bytes
            for sequence in sequences
            for payload in sequence
        ):
            fail(FAIL_INPUT, "complete-session artifacts must contain exact bytes only")

        previous = GENESIS_CHECKPOINT_SHA256
        request_ids: set[str] = set()
        records: list[tuple[StoreCheckpointV2, StoreHeadV2, StoreResponseV2]] = []
        for index in range(expected):
            request_payload = request_payloads_in_exact_order[index]
            object_payload = object_payloads_in_exact_order[index]
            receipt_payload = signed_receipt_payloads_in_exact_order[index]
            checkpoint_payload = checkpoint_payloads_in_exact_order[index]
            head_payload = head_payloads_in_exact_order[index]
            response_payload = response_payloads_in_exact_order[index]
            request = load_request(request_payload)
            receipt = load_receipt(receipt_payload)
            checkpoint = load_checkpoint(checkpoint_payload)
            head = load_head(head_payload)
            response = load_response(response_payload)
            binding = verify_triplet(
                checkpoint=checkpoint,
                head=head,
                response=response,
                plan=plan,
                authority_binding=authority_binding,
            )
            verify_receipt(receipt, authority_binding)
            if (
                binding.ordinal != index
                or binding.previous_checkpoint_sha256 != previous
                or binding.request_id in request_ids
            ):
                fail(FAIL_CHAIN, "pure replay found a gap, fork, or duplicate request")
            if (
                request != binding.to_request()
                or request.canonical_bytes() != request_payload
                or receipt != checkpoint.signed_receipt
                or receipt.canonical_bytes() != receipt_payload
                or response.checkpoint_record.canonical_bytes() != checkpoint_payload
                or response.head_record.canonical_bytes() != head_payload
            ):
                fail(
                    FAIL_BINDING,
                    "presented request/receipt/checkpoint/head bytes differ from signed graph",
                )
            entry = plan.entry(index)
            if not 0 < len(object_payload) <= entry.maximum_bytes:
                fail(FAIL_BINDING, "presented object bytes are outside planned maximum")
            object_sha = sha256_bytes(object_payload)
            if (
                object_sha != binding.object_id
                or object_sha != request.bound_artifact_sha256
                or len(object_payload) != binding.object_size_bytes
                or len(object_payload) != request.reported_size_bytes
            ):
                fail(FAIL_BINDING, "presented object bytes differ from signed request/object")
            request_ids.add(binding.request_id)
            previous = checkpoint.checkpoint_sha256
            records.append((checkpoint, head, response))

        if (
            not records
            or records[-1][1].binding.complete is not True
            or records[-1][1].binding.observed_object_count != expected
            or any(head.binding.complete is True for _, head, _ in records[:-1])
        ):
            fail(FAIL_INCOMPLETE, "presented session lacks one exact terminal complete head")
        return require_verified_cas_store_complete_session_v2(
            issue_complete_result(
                plan=plan,
                authority_binding=authority_binding,
                records=tuple(records),
            )
        )

    return (
        VerifiedCasStoreCompleteSessionV2,
        verify_cas_store_complete_session_v2,
        require_verified_cas_store_complete_session_v2,
    )


(
    VerifiedCasStoreCompleteSessionV2,
    verify_cas_store_complete_session_v2,
    require_verified_cas_store_complete_session_v2,
) = _build_private_complete_session_mechanics()
del _build_private_complete_session_mechanics


# Descriptive alias: the V2 service's request schema itself is /1.
STORE_REQUEST_SCHEMA_VERSION: Final = GENERIC_STORE_REQUEST_SCHEMA_VERSION
StoreRequestV2 = GenericStoreRequestV1


__all__ = [
    "SCHEMA_VERSION",
    "RETENTION_PLAN_SCHEMA_VERSION",
    "GENERIC_STORE_REQUEST_SCHEMA_VERSION",
    "STORE_REQUEST_SCHEMA_VERSION",
    "TRANSITION_BINDING_SCHEMA_VERSION",
    "STORE_RECEIPT_BODY_SCHEMA_VERSION",
    "SIGNED_STORE_RECEIPT_SCHEMA_VERSION",
    "STORE_CHECKPOINT_SCHEMA_VERSION",
    "STORE_HEAD_SCHEMA_VERSION",
    "STORE_RESPONSE_SCHEMA_VERSION",
    "CAS_STORE_ROLE",
    "CAS_STORE_CONTROL_DOMAIN",
    "CAS_STORE_RECEIPT_PURPOSE",
    "CAS_STORE_SIGNATURE_DOMAIN",
    "CAS_STORE_CHECKPOINT_PURPOSE",
    "CAS_STORE_HEAD_PURPOSE",
    "CAS_STORE_RESPONSE_PURPOSE",
    "KEY_EPOCH",
    "AUTHORITY_CLAIM_BOUNDARY",
    "SOURCE_PIN_CLAIM_BOUNDARY",
    "FAIL_CLOSED_ROLLBACK_DISCLOSURE",
    "REQUIRED_EXTERNAL_CONTROLS",
    "INSPECTED_STORE_V1_SOURCE_SHA256",
    "INSPECTED_AUTHORITY_V2_SOURCE_SHA256",
    "INSPECTED_OWNER_STATUS_V1_SOURCE_SHA256",
    "MAXIMUM_PLAN_BYTES",
    "MAXIMUM_REQUEST_BYTES",
    "MAXIMUM_RECEIPT_BYTES",
    "MAXIMUM_CHECKPOINT_BYTES",
    "MAXIMUM_HEAD_BYTES",
    "MAXIMUM_RESPONSE_BYTES",
    "MAXIMUM_PLAN_OBJECTS",
    "MAXIMUM_PLANNED_OBJECT_BYTES",
    "GENESIS_CHECKPOINT_SHA256",
    "PLAN_ID_PREFIX",
    "REQUEST_ID_PREFIX",
    "HEAD_STATE_ID_PREFIX",
    "CasStoreV2Error",
    "PlannedObjectV1",
    "RetentionPlanV1",
    "GenericStoreRequestV1",
    "StoreRequestV2",
    "StoreTransitionBindingV2",
    "StoreReceiptBodyV2",
    "SignedStoreReceiptV2",
    "StoreCheckpointV2",
    "StoreHeadV2",
    "StoreResponseV2",
    "PeerCredentialsV2",
    "ReceivedStoreRequestV2",
    "CasStoreV2",
    "VerifiedCasStoreCompleteSessionV2",
    "build_retention_plan_v1",
    "load_retention_plan_v1",
    "build_generic_store_request_v1",
    "load_generic_store_request_v1",
    "load_store_checkpoint_v2",
    "load_store_head_v2",
    "load_store_response_v2",
    "recv_store_request_v2",
    "send_store_request_fd_v2",
    "verify_signed_store_receipt_v2",
    "verify_cas_store_complete_session_v2",
    "require_verified_cas_store_complete_session_v2",
]

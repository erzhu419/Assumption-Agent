"""Local mechanics-only actor for the Step-5 admission ``A`` operation.

This module implements the local, initial-slot mechanics behind the frozen
external-witness admission contract::

    Q1 request + Q1 response -> signed A compare-and-append request
        -> A checkpoint + A receipt -> A head (commit marker)

The actor is constructed from three exact canonical raw policy payloads and
independent raw ID/SHA-256 pins.  Construction closes the bootstrap -> online
-> durable policy graph, verifies the bootstrap and A policy signatures,
matches the supplied Ed25519 private key to the pinned A public key, and opens
only an existing absolute owner-private ``0700`` directory.  The private key
is accepted only by actor construction and is never exposed by an issued
result.

One directory is one frozen slot, keyed by ``(attempt_context_id,
anchor_sequence=1)``.  An exclusive, fsynced slot intent is the compare fence.
For a first request the actor verifies the complete raw Q1/Q1-response/A
signature and mapping closure, signs the contract-exact checkpoint, head, and
receipt, and publishes request/checkpoint/receipt records with atomic
no-replace hard links.  The head is published last and is the sole commit
marker.  Every record is fsynced, reopened, metadata checked, and read back;
the directory is fsynced at each publication boundary.  A complete exact retry
returns the original three payloads without signing again.  A different
request for the slot, a partial crash prefix, a gap, corruption, rollbacked
high-water policy, symlink, hardlink, or extra namespace permanently fails
closed; this module performs no recovery or repair.

These mechanics do *not* establish external durability, rollback resistance,
non-equivocation, liveness, currentness, key custody, authority, R01,
readiness, or execution authorization.  The directory and private key are
local construction inputs only.  There is no listener, daemon, systemd unit,
network endpoint, decoded mapping input, UID override, caller-supplied outcome,
or real-admission entry point.  Real admission remains ``NOT_IMPLEMENTED``.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
import re
import stat
from types import FunctionType, MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1 import (
    ALGORITHM_V1 as _ALGORITHM,
    A_COMPARE_AND_APPEND_RECEIPT_BODY_FIELDS_V1 as _A_RECEIPT_BODY_FIELDS,
    A_COMPARE_AND_APPEND_RECEIPT_FIELDS_V1 as _A_RECEIPT_FIELDS,
    A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1 as _A_RECEIPT_SCHEMA,
    A_COMPARE_AND_APPEND_REQUEST_BODY_FIELDS_V1 as _A_REQUEST_BODY_FIELDS,
    A_COMPARE_AND_APPEND_REQUEST_FIELDS_V1 as _A_REQUEST_FIELDS,
    A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1 as _A_REQUEST_SCHEMA,
    A_DURABLE_CHECKPOINT_BODY_FIELDS_V1 as _A_CHECKPOINT_BODY_FIELDS,
    A_DURABLE_CHECKPOINT_FIELDS_V1 as _A_CHECKPOINT_FIELDS,
    A_DURABLE_CHECKPOINT_SCHEMA_V1 as _A_CHECKPOINT_SCHEMA,
    A_DURABLE_HEAD_BODY_FIELDS_V1 as _A_HEAD_BODY_FIELDS,
    A_DURABLE_HEAD_FIELDS_V1 as _A_HEAD_FIELDS,
    A_DURABLE_HEAD_SCHEMA_V1 as _A_HEAD_SCHEMA,
    A_PURPOSES_IN_EXACT_ORDER_V1 as _A_PURPOSES,
    BOOTSTRAP_EXTERNAL_WITNESS_POLICY_FIELDS_V1 as _BOOTSTRAP_POLICY_FIELDS,
    BOOTSTRAP_EXTERNAL_WITNESS_POLICY_SCHEMA_V1 as _BOOTSTRAP_POLICY_SCHEMA,
    BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1 as _BOOTSTRAP_PURPOSES,
    CANDIDATE_CLAIM_LEVEL_V1 as _ADMISSION_CLAIM_LEVEL,
    CANDIDATE_STATUS_V1 as _ADMISSION_STATUS,
    CAS_OUTCOME_V1 as _CAS_OUTCOME,
    DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1 as _DURABLE_POLICY_BODY_FIELDS,
    DURABLE_ANCHOR_POLICY_FIELDS_V1 as _DURABLE_POLICY_FIELDS,
    DURABLE_ANCHOR_POLICY_SCHEMA_V1 as _DURABLE_POLICY_SCHEMA,
    DURABLE_MAPPING_FIELDS_V1 as _DURABLE_MAPPING_FIELDS,
    ONLINE_PURPOSES_IN_EXACT_ORDER_V1 as _ONLINE_PURPOSES,
    ONLINE_WITNESS_POLICY_BODY_FIELDS_V1 as _ONLINE_POLICY_BODY_FIELDS,
    ONLINE_WITNESS_POLICY_FIELDS_V1 as _ONLINE_POLICY_FIELDS,
    ONLINE_WITNESS_POLICY_SCHEMA_V1 as _ONLINE_POLICY_SCHEMA,
    PREPARE_OUTCOME_V1 as _PREPARE_OUTCOME,
    Q1_PREPARE_REQUEST_BODY_FIELDS_V1 as _Q1_REQUEST_BODY_FIELDS,
    Q1_PREPARE_REQUEST_FIELDS_V1 as _Q1_REQUEST_FIELDS,
    Q1_PREPARE_REQUEST_SCHEMA_V1 as _Q1_REQUEST_SCHEMA,
    Q1_PREPARE_RESPONSE_BODY_FIELDS_V1 as _Q1_RESPONSE_BODY_FIELDS,
    Q1_PREPARE_RESPONSE_FIELDS_V1 as _Q1_RESPONSE_FIELDS,
    Q1_PREPARE_RESPONSE_SCHEMA_V1 as _Q1_RESPONSE_SCHEMA,
    RUNTIME_CUSTODY_BINDING_FIELDS_V1 as _RUNTIME_CUSTODY_FIELDS,
    SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1 as _SIGN_A_RECEIPT,
    SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1 as _SIGN_A_REQUEST,
    SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1 as _SIGN_A_CHECKPOINT,
    SIGN_A_DURABLE_HEAD_PURPOSE_V1 as _SIGN_A_HEAD,
    SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1 as _SIGN_DURABLE_POLICY,
    SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1 as _SIGN_ONLINE_POLICY,
    SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1 as _SIGN_Q1_REQUEST,
    SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1 as _SIGN_Q1_RESPONSE,
    SIGNATURE_FIELDS_V1 as _SIGNATURE_FIELDS,
    canonical_external_witness_admission_json_bytes_v1 as _contract_canonical,
    phase2b_step5_external_witness_admission_content_id_v1 as _contract_content_id,
    phase2b_step5_external_witness_admission_signature_preimage_v1 as _contract_signature_preimage,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical,
)

try:  # Clean import remains possible; use fails closed without the backend.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - deployment image dependent.
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


ADMISSION_CONTRACT_SOURCE_SHA256_V1: Final = (
    "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285"
)

LOCAL_DURABLE_ANCHOR_SLOT_INTENT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-"
    "external-admission-a-local-slot-intent/1"
)
LOCAL_DURABLE_ANCHOR_SLOT_KEY_PROFILE_V1: Final = (
    "UNIQUE_BY_ATTEMPT_CONTEXT_ID_AND_ANCHOR_SEQUENCE"
)
LOCAL_COMPARE_AND_APPEND_SEMANTICS_V1: Final = (
    "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
)
LOCAL_MECHANICS_CLAIM_BOUNDARY_V1: Final = (
    "LOCAL_A_EXACT_GENESIS_COMPARE_AND_APPEND_EXCLUSIVE_SLOT_FSYNC_READBACK_"
    "MECHANICS_ONLY_NOT_EXTERNAL_DURABILITY_NOT_HOST_ROLLBACK_RESISTANCE_"
    "NOT_NON_EQUIVOCATION_NOT_LIVE_NOT_CURRENT_NOT_KEY_CUSTODY_NOT_AUTHORITY_"
    "NOT_R01_NOT_READINESS_NOT_EXECUTION"
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_CONTENT_ID"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_BINDING"
FAIL_TRUST_ANCHOR_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_TRUST_ANCHOR"
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_SIGNATURE"
FAIL_KEY_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_KEY"
FAIL_FILESYSTEM_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_FILESYSTEM"
FAIL_CONFLICT_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_SLOT_CONFLICT"
FAIL_POISONED_V1: Final = "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_POISONED_SLOT"
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_CLAIM_BOUNDARY"
)
FAIL_NOT_IMPLEMENTED_V1: Final = (
    "FAIL_STEP5_R01_DURABLE_ANCHOR_V1_NOT_IMPLEMENTED"
)

_BOOTSTRAP_STATUS: Final = "CANDIDATE_NONADMITTED"
_BOOTSTRAP_CLAIM_LEVEL: Final = (
    "SIGNED_WIRE_MECHANICS_CANDIDATE_NOT_LIVE_NOT_CURRENT_NOT_DURABLE_"
    "NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
_SIGNATURE_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-admission-signature/1"
)
_BOOTSTRAP_POLICY_PREFIX: Final = "phase2b_step5_external_witness_policy_v1_"
_ONLINE_POLICY_PREFIX: Final = "phase2b_step5_online_witness_policy_v1_"
_DURABLE_POLICY_PREFIX: Final = (
    "phase2b_step5_external_durable_anchor_policy_v1_"
)
_Q1_REQUEST_PREFIX: Final = "phase2b_step5_external_admission_q1_request_v1_"
_Q1_RESPONSE_PREFIX: Final = "phase2b_step5_external_admission_q1_response_v1_"
_A_REQUEST_PREFIX: Final = "phase2b_step5_external_admission_a_request_v1_"
_A_CHECKPOINT_PREFIX: Final = (
    "phase2b_step5_external_admission_a_checkpoint_v1_"
)
_A_HEAD_PREFIX: Final = "phase2b_step5_external_admission_a_head_v1_"
_A_RECEIPT_PREFIX: Final = (
    "phase2b_step5_external_admission_a_receipt_v1_"
)
_A_GENESIS_CHECKPOINT_ID: Final = _A_CHECKPOINT_PREFIX + ("0" * 64)
_A_GENESIS_HEAD_ID: Final = _A_HEAD_PREFIX + ("0" * 64)
_BOOTSTRAP_POLICY_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
)
_CURRENT_MAP_LEAF_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/CURRENT_MAP_LEAF/V1\0"
)
_LOCAL_SLOT_KEY_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/A/LOCAL_SLOT_KEY/V1\0"
)
_LOCAL_SLOT_INTENT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/A/LOCAL_SLOT_INTENT/ID/V1\0"
)
_LOCAL_SLOT_KEY_PREFIX: Final = "phase2b_step5_external_admission_a_slot_v1_"
_LOCAL_SLOT_INTENT_PREFIX: Final = (
    "phase2b_step5_external_admission_a_slot_intent_v1_"
)

_MAXIMUM_CANONICAL_BYTES: Final = 16 * 1024 * 1024
_MAXIMUM_SPOOL_RECORD_BYTES: Final = 16 * 1024 * 1024
_MAXIMUM_SAFE_INTEGER: Final = (1 << 53) - 1
_MAXIMUM_PATH_BYTES: Final = 4096
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX128 = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_SAFE_BASENAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,254}")

_SLOT_INTENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "mechanics_claim_boundary",
        "slot_key",
        "attempt_context_id",
        "anchor_sequence",
        "bootstrap_external_witness_policy_id",
        "bootstrap_external_witness_policy_payload_sha256",
        "online_witness_policy_id",
        "online_witness_policy_payload_sha256",
        "durable_anchor_policy_id",
        "durable_anchor_policy_payload_sha256",
        "q1_request_id",
        "q1_request_payload_sha256",
        "q1_response_id",
        "q1_response_payload_sha256",
        "a_request_id",
        "a_request_payload_sha256",
        "a_checkpoint_id",
        "a_checkpoint_payload_sha256",
        "a_head_id",
        "a_head_payload_sha256",
        "a_receipt_id",
        "a_receipt_payload_sha256",
        "slot_intent_id",
        "q1_request_file_basename",
        "q1_response_file_basename",
        "a_request_file_basename",
        "a_checkpoint_file_basename",
        "a_receipt_file_basename",
        "a_head_file_basename",
        "publication_order_in_exact_order",
        "head_is_sole_commit_marker",
    }
)


class Phase2BStep5R01DurableAnchorErrorV1(ValueError):
    """Stable fail-closed error for local A-actor mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01DurableAnchorErrorV1(code, detail)


def _canonical(value: object, _implementation=_canonicalize) -> bytes:
    try:
        return _implementation(value)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_CANONICAL_V1, "RFC-8785 canonicalization failed"
        ) from exc


def _restore_wire_integers(value: object) -> object:
    if type(value) is float and math.isfinite(value) and value.is_integer():
        if abs(value) > _MAXIMUM_SAFE_INTEGER:
            _fail(FAIL_SCHEMA_V1, "decoded integer exceeds I-JSON safe range")
        return int(value)
    if type(value) is list:
        return [_restore_wire_integers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_wire_integers(item) for key, item in value.items()}
    return value


def _decode(payload: bytes, label: str, _decoder=_decode_canonical) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAXIMUM_CANONICAL_BYTES
        or payload.endswith(b"\n")
    ):
        _fail(FAIL_INPUT_V1, f"{label} must be bounded exact bytes without newline")
    try:
        decoded = _restore_wire_integers(_decoder(payload))
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_CANONICAL_V1, f"{label} is not exact canonical RFC-8785"
        ) from exc
    if type(decoded) is not dict or _canonical(decoded) != payload:
        _fail(FAIL_CANONICAL_V1, f"{label} canonical readback differs")
    return decoded


def _object(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(FAIL_SCHEMA_V1, f"{label} keyset differs")
    return value


def _text(value: object, label: str, maximum: int = 4096) -> str:
    if type(value) is not str or not 1 <= len(value.encode("utf-8")) <= maximum:
        _fail(FAIL_SCHEMA_V1, f"{label} must be bounded nonempty text")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAXIMUM_SAFE_INTEGER:
        _fail(FAIL_SCHEMA_V1, f"{label} must be a safe integer >= {minimum}")
    return value


def _hex64(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lowercase SHA-256 hex")
    return value


def _hex128(value: object, label: str) -> str:
    if type(value) is not str or _HEX128.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be exact lowercase Ed25519 hex")
    return value


def _identifier(value: object, label: str, prefix: str | None = None) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be an exact identifier")
    if prefix is not None and not value.startswith(prefix):
        _fail(FAIL_SCHEMA_V1, f"{label} prefix differs")
    return value


def _nonce(value: object, label: str) -> str:
    nonce = _hex64(value, label)
    if nonce == "0" * 64:
        _fail(FAIL_SCHEMA_V1, f"{label} must be nonzero")
    return nonce


def _payload_sha(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "payload SHA input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _without(value: Mapping[str, object], *names: str) -> dict[str, object]:
    return {key: item for key, item in value.items() if key not in names}


def _decode_oob_id(value: bytes, label: str, prefix: str) -> str:
    if type(value) is not bytes or not 1 <= len(value) <= 512:
        _fail(FAIL_INPUT_V1, f"{label} OOB ID must be bounded exact bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_INPUT_V1, f"{label} OOB ID must be exact ASCII"
        ) from exc
    return _identifier(decoded, label, prefix)


def _load_oob_payload(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    prefix: str,
    label: str,
) -> tuple[dict[str, object], str]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAXIMUM_CANONICAL_BYTES
        or payload.endswith(b"\n")
        or type(expected_payload_sha256_digest) is not bytes
        or len(expected_payload_sha256_digest) != 32
        or expected_payload_sha256_digest == b"\x00" * 32
    ):
        _fail(FAIL_INPUT_V1, f"{label} payload and OOB digest must be exact bytes")
    expected_id = _decode_oob_id(expected_id_bytes, label, prefix)
    if hashlib.sha256(payload).digest() != expected_payload_sha256_digest:
        _fail(FAIL_TRUST_ANCHOR_V1, f"{label} raw SHA-256 pin differs")
    return _decode(payload, label), expected_id


def _verify_ed25519(
    *, public_key_hex: str, signature_hex: str, preimage: bytes, label: str
) -> None:
    _hex64(public_key_hex, f"{label} public key")
    _hex128(signature_hex, f"{label} signature")
    if type(preimage) is not bytes or not preimage:
        _fail(FAIL_SIGNATURE_V1, f"{label} preimage must be exact bytes")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        public_key = _Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(bytes.fromhex(signature_hex), preimage)
    except Exception as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_SIGNATURE_V1, f"{label} Ed25519 signature rejected"
        ) from exc


def _signature(
    value: object,
    *,
    expected_key_id: str,
    expected_key_epoch: int,
    expected_purpose: str,
    public_key_hex: str,
    preimage: bytes,
    label: str,
) -> dict[str, object]:
    signature = _object(value, frozenset(_SIGNATURE_FIELDS), f"{label} signature")
    if (
        signature["schema_version"] != _SIGNATURE_SCHEMA
        or signature["algorithm"] != _ALGORITHM
        or signature["signer_key_id"] != expected_key_id
        or type(signature["key_epoch"]) is not int
        or signature["key_epoch"] != expected_key_epoch
        or signature["purpose"] != expected_purpose
    ):
        _fail(FAIL_SIGNATURE_V1, f"{label} signature metadata differs")
    _verify_ed25519(
        public_key_hex=public_key_hex,
        signature_hex=_hex128(signature["signature_hex"], f"{label} signature"),
        preimage=preimage,
        label=label,
    )
    return signature


def _contract_id(artifact_kind: str, body: object) -> str:
    try:
        value = _contract_content_id(artifact_kind, body)
    except Exception as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_CONTENT_ID_V1, "admission-contract content-ID derivation failed"
        ) from exc
    return _identifier(value, f"{artifact_kind} content ID")


def _contract_preimage(
    *,
    artifact_kind: str,
    body: object,
    artifact_id: str,
    signer_key_id: str,
    key_epoch: int,
    purpose: str,
) -> bytes:
    try:
        value = _contract_signature_preimage(
            artifact_kind=artifact_kind,
            body=body,
            artifact_id=artifact_id,
            signer_key_id=signer_key_id,
            key_epoch=key_epoch,
            purpose=purpose,
        )
    except Exception as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_SIGNATURE_V1, "admission-contract signature preimage failed"
        ) from exc
    if type(value) is not bytes or not value:
        _fail(FAIL_SIGNATURE_V1, "admission-contract signature preimage differs")
    return value


def _load_bootstrap_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
) -> dict[str, object]:
    decoded, expected_id = _load_oob_payload(
        payload,
        expected_id_bytes=expected_id_bytes,
        expected_payload_sha256_digest=expected_payload_sha256_digest,
        prefix=_BOOTSTRAP_POLICY_PREFIX,
        label="bootstrap external-witness policy",
    )
    policy = _object(
        decoded,
        frozenset(_BOOTSTRAP_POLICY_FIELDS),
        "bootstrap external-witness policy",
    )
    if (
        policy["schema_version"] != _BOOTSTRAP_POLICY_SCHEMA
        or policy["status"] != _BOOTSTRAP_STATUS
        or policy["claim_level"] != _BOOTSTRAP_CLAIM_LEVEL
        or policy["algorithm"] != _ALGORITHM
        or type(policy["witness_key_epoch"]) is not int
        or policy["witness_key_epoch"] != 1
    ):
        _fail(FAIL_SCHEMA_V1, "bootstrap policy header differs")
    _text(policy["witness_namespace"], "bootstrap witness namespace", 256)
    _identifier(policy["witness_control_domain_id"], "bootstrap control domain")
    _identifier(policy["witness_key_id"], "bootstrap key ID")
    _hex64(policy["public_key_hex"], "bootstrap public key")
    expected_computed = _BOOTSTRAP_POLICY_PREFIX + hashlib.sha256(
        _BOOTSTRAP_POLICY_DOMAIN
        + _canonical(_without(policy, "external_witness_policy_id"))
    ).hexdigest()
    if (
        policy["external_witness_policy_id"] != expected_computed
        or expected_computed != expected_id
    ):
        _fail(FAIL_CONTENT_ID_V1, "bootstrap policy content ID differs")
    return policy


def _load_signed_root(
    payload: bytes,
    *,
    label: str,
    schema: str,
    fields: frozenset[str],
    body_fields: frozenset[str],
    artifact_kind: str,
    id_field: str,
    signature_field: str,
    expected_key_id: str,
    expected_key_epoch: int,
    expected_purpose: str,
    public_key_hex: str,
) -> tuple[dict[str, object], dict[str, object]]:
    root = _object(_decode(payload, label), fields, label)
    body = _object(root["body"], body_fields, f"{label} body")
    if root["schema_version"] != schema or body["schema_version"] != schema:
        _fail(FAIL_SCHEMA_V1, f"{label} schema differs")
    artifact_id = _contract_id(artifact_kind, body)
    if root[id_field] != artifact_id:
        _fail(FAIL_CONTENT_ID_V1, f"{label} content ID differs")
    preimage = _contract_preimage(
        artifact_kind=artifact_kind,
        body=body,
        artifact_id=artifact_id,
        signer_key_id=expected_key_id,
        key_epoch=expected_key_epoch,
        purpose=expected_purpose,
    )
    _signature(
        root[signature_field],
        expected_key_id=expected_key_id,
        expected_key_epoch=expected_key_epoch,
        expected_purpose=expected_purpose,
        public_key_hex=public_key_hex,
        preimage=preimage,
        label=label,
    )
    root["body"] = body
    return root, body


def _load_online_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
) -> dict[str, object]:
    decoded, expected_id = _load_oob_payload(
        payload,
        expected_id_bytes=expected_id_bytes,
        expected_payload_sha256_digest=expected_payload_sha256_digest,
        prefix=_ONLINE_POLICY_PREFIX,
        label="online-witness policy",
    )
    root = _object(decoded, frozenset(_ONLINE_POLICY_FIELDS), "online policy")
    body = _object(
        root["body"], frozenset(_ONLINE_POLICY_BODY_FIELDS), "online policy body"
    )
    if (
        root["schema_version"] != _ONLINE_POLICY_SCHEMA
        or body["schema_version"] != _ONLINE_POLICY_SCHEMA
        or body["status"] != _ADMISSION_STATUS
        or body["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or type(body["policy_epoch"]) is not int
        or body["policy_epoch"] != 1
        or body["algorithm"] != _ALGORITHM
        or type(body["bootstrap_key_epoch"]) is not int
        or body["bootstrap_key_epoch"] != 1
        or type(body["online_key_epoch"]) is not int
        or body["online_key_epoch"] != 1
        or type(body["bootstrap_authorized_signature_purposes_in_exact_order"])
        is not list
        or tuple(body["bootstrap_authorized_signature_purposes_in_exact_order"])
        != tuple(_BOOTSTRAP_PURPOSES)
        or type(body["online_authorized_signature_purposes_in_exact_order"])
        is not list
        or tuple(body["online_authorized_signature_purposes_in_exact_order"])
        != tuple(_ONLINE_PURPOSES)
        or body["online_key_nonexportable_required"] is not True
        or body["runtime_identity_profile"]
        != "LINUX_AF_UNIX_SEQPACKET_PROCFS_ENTRY_EXIT_V1"
    ):
        _fail(FAIL_SCHEMA_V1, "online policy header/purpose order differs")
    for field in (
        "bootstrap_key_id",
        "online_key_id",
        "witness_control_domain_id",
        "service_binding_id",
        "runtime_closure_policy_id",
        "custody_attestation_policy_id",
        "custody_attestor_key_id",
    ):
        _identifier(body[field], f"online policy {field}")
    for field in (
        "service_uid",
        "service_gid",
        "service_user_namespace_inode",
        "custody_attestor_key_epoch",
    ):
        _integer(body[field], f"online policy {field}", 1)
    for field in (
        "runtime_closure_policy_payload_sha256",
        "custody_attestation_policy_payload_sha256",
    ):
        _hex64(body[field], f"online policy {field}")
    _text(body["witness_namespace"], "online witness namespace", 256)
    bootstrap_public = _hex64(
        body["bootstrap_public_key_hex"], "online policy bootstrap public key"
    )
    online_public = _hex64(
        body["online_public_key_hex"], "online policy online public key"
    )
    custody_public = _hex64(
        body["custody_attestor_public_key_hex"],
        "online policy custody-attestor public key",
    )
    if (
        body["bootstrap_external_witness_policy_id"]
        != bootstrap_policy["external_witness_policy_id"]
        or body["bootstrap_external_witness_policy_payload_sha256"]
        != _payload_sha(bootstrap_policy_payload)
        or body["bootstrap_key_id"] != bootstrap_policy["witness_key_id"]
        or body["bootstrap_key_epoch"] != bootstrap_policy["witness_key_epoch"]
        or bootstrap_public != bootstrap_policy["public_key_hex"]
        or body["witness_namespace"] != bootstrap_policy["witness_namespace"]
        or body["witness_control_domain_id"]
        == bootstrap_policy["witness_control_domain_id"]
    ):
        _fail(FAIL_BINDING_V1, "online/bootstrap policy linkage differs")
    if (
        len({bootstrap_public, online_public, custody_public}) != 3
        or len(
            {
                body["bootstrap_key_id"],
                body["online_key_id"],
                body["custody_attestor_key_id"],
            }
        )
        != 3
        or body["service_uid"] == body["service_gid"]
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "online policy keys/identity are not disjoint")
    policy_id = _contract_id("online_witness_policy", body)
    if root["online_witness_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "online policy content ID differs")
    preimage = _contract_preimage(
        artifact_kind="online_witness_policy",
        body=body,
        artifact_id=policy_id,
        signer_key_id=body["bootstrap_key_id"],
        key_epoch=body["bootstrap_key_epoch"],
        purpose=_SIGN_ONLINE_POLICY,
    )
    _signature(
        root["bootstrap_signature"],
        expected_key_id=body["bootstrap_key_id"],
        expected_key_epoch=body["bootstrap_key_epoch"],
        expected_purpose=_SIGN_ONLINE_POLICY,
        public_key_hex=bootstrap_public,
        preimage=preimage,
        label="online policy bootstrap",
    )
    root["body"] = body
    return root


def _genesis_mapping(attempt_context_id: str) -> dict[str, object]:
    _identifier(attempt_context_id, "attempt-context ID")
    return {
        "external_anchor_id": None,
        "external_anchor_payload_sha256": None,
        "external_receipt_id": None,
        "external_receipt_payload_sha256": None,
        "external_ledger_id": None,
        "external_ledger_payload_sha256": None,
        "external_checkpoint_id": None,
        "external_checkpoint_payload_sha256": None,
        "external_head_id": None,
        "external_head_payload_sha256": None,
        "witness_sequence": 0,
        "tree_size": 0,
        "signed_tree_root_sha256": None,
        "attempt_context_id": attempt_context_id,
        "anchor_sequence": 0,
        "effective_state": "ABSENT",
        "superseded": False,
        "revoked": False,
        "map_leaf_index": None,
        "map_tree_size": 0,
        "map_root_sha256": None,
        "map_inclusion_path_sha256_in_exact_order": [],
        "presented_to_live_consistency_path_sha256_in_exact_order": [],
    }


def _current_map_leaf_root(mapping: Mapping[str, object]) -> str:
    subject = _without(
        mapping,
        "map_leaf_index",
        "map_tree_size",
        "map_root_sha256",
        "map_inclusion_path_sha256_in_exact_order",
        "presented_to_live_consistency_path_sha256_in_exact_order",
    )
    return hashlib.sha256(_CURRENT_MAP_LEAF_DOMAIN + _canonical(subject)).hexdigest()


def _mapping(
    value: object, label: str, expected_attempt_context_id: str
) -> dict[str, object]:
    mapping = _object(value, frozenset(_DURABLE_MAPPING_FIELDS), label)
    sequence = _integer(mapping["witness_sequence"], f"{label} witness sequence")
    tree_size = _integer(mapping["tree_size"], f"{label} tree size")
    anchor_sequence = _integer(
        mapping["anchor_sequence"], f"{label} anchor sequence"
    )
    map_tree_size = _integer(mapping["map_tree_size"], f"{label} map tree size")
    if mapping["attempt_context_id"] != expected_attempt_context_id:
        _fail(FAIL_BINDING_V1, f"{label} attempt context differs")
    if mapping["superseded"] is not False or mapping["revoked"] is not False:
        _fail(FAIL_BINDING_V1, f"{label} state booleans differ")
    for field in (
        "map_inclusion_path_sha256_in_exact_order",
        "presented_to_live_consistency_path_sha256_in_exact_order",
    ):
        path = mapping[field]
        if type(path) is not list or len(path) > 64:
            _fail(FAIL_SCHEMA_V1, f"{label} {field} shape differs")
        for index, digest in enumerate(path):
            _hex64(digest, f"{label} {field} {index}")
    if sequence == 0 or tree_size == 0:
        if mapping != _genesis_mapping(expected_attempt_context_id):
            _fail(FAIL_BINDING_V1, f"{label} partial genesis mapping differs")
        return mapping
    if sequence != 1 or tree_size != 1 or anchor_sequence != 1:
        _fail(FAIL_BINDING_V1, f"{label} only initial mapping is frozen")
    for field, prefix in (
        ("external_anchor_id", "phase2b_step5_signed_owner_status_anti_fork_anchor_v1_"),
        ("external_receipt_id", "phase2b_step5_external_owner_status_anchor_receipt_v1_"),
        ("external_ledger_id", "phase2b_step5_external_witness_ledger_v1_"),
        ("external_checkpoint_id", "phase2b_step5_external_witness_checkpoint_v1_"),
        ("external_head_id", "phase2b_step5_external_witness_head_v1_"),
    ):
        _identifier(mapping[field], f"{label} {field}", prefix)
    for field in (
        "external_anchor_payload_sha256",
        "external_receipt_payload_sha256",
        "external_ledger_payload_sha256",
        "external_checkpoint_payload_sha256",
        "external_head_payload_sha256",
        "signed_tree_root_sha256",
    ):
        _hex64(mapping[field], f"{label} {field}")
    if (
        mapping["effective_state"] != "CURRENT_UNSUPERSEDED"
        or mapping["map_leaf_index"] != 0
        or type(mapping["map_leaf_index"]) is not int
        or map_tree_size != 1
        or mapping["map_inclusion_path_sha256_in_exact_order"] != []
        or mapping["presented_to_live_consistency_path_sha256_in_exact_order"]
        != []
        or mapping["map_root_sha256"] != _current_map_leaf_root(mapping)
    ):
        _fail(FAIL_BINDING_V1, f"{label} authenticated current mapping differs")
    return mapping


def _load_durable_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_sha256_digest: bytes,
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
    online_policy: Mapping[str, object],
) -> dict[str, object]:
    decoded, expected_id = _load_oob_payload(
        payload,
        expected_id_bytes=expected_id_bytes,
        expected_payload_sha256_digest=expected_payload_sha256_digest,
        prefix=_DURABLE_POLICY_PREFIX,
        label="durable-anchor policy",
    )
    root = _object(decoded, frozenset(_DURABLE_POLICY_FIELDS), "durable policy")
    body = _object(
        root["body"],
        frozenset(_DURABLE_POLICY_BODY_FIELDS),
        "durable policy body",
    )
    if (
        root["schema_version"] != _DURABLE_POLICY_SCHEMA
        or body["schema_version"] != _DURABLE_POLICY_SCHEMA
        or body["status"] != _ADMISSION_STATUS
        or body["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or type(body["policy_epoch"]) is not int
        or body["policy_epoch"] != 1
        or body["algorithm"] != _ALGORITHM
        or type(body["anchor_key_epoch"]) is not int
        or body["anchor_key_epoch"] != 1
        or type(body["authorized_signature_purposes_in_exact_order"]) is not list
        or tuple(body["authorized_signature_purposes_in_exact_order"])
        != tuple(_A_PURPOSES)
        or body["durable_before_return_required"] is not True
        or body["admission_slot_key_profile"]
        != LOCAL_DURABLE_ANCHOR_SLOT_KEY_PROFILE_V1
        or body["compare_and_append_semantics"]
        != LOCAL_COMPARE_AND_APPEND_SEMANTICS_V1
    ):
        _fail(FAIL_SCHEMA_V1, "durable policy header differs")
    _text(body["anchor_namespace"], "durable anchor namespace", 256)
    _identifier(body["anchor_control_domain_id"], "durable control domain")
    _identifier(body["anchor_key_id"], "durable A key ID")
    a_public = _hex64(body["anchor_public_key_hex"], "durable A public key")
    for field in (
        "bootstrap_external_witness_policy_payload_sha256",
        "genesis_tree_root_sha256",
        "oob_high_water_checkpoint_payload_sha256",
        "oob_high_water_head_payload_sha256",
        "oob_high_water_tree_root_sha256",
    ):
        _hex64(body[field], f"durable policy {field}")
    for field in (
        "oob_high_water_checkpoint_sequence",
        "oob_high_water_head_sequence",
    ):
        _integer(body[field], f"durable policy {field}")
    _identifier(
        body["genesis_checkpoint_id"],
        "durable genesis checkpoint ID",
        _A_CHECKPOINT_PREFIX,
    )
    _identifier(
        body["genesis_head_id"], "durable genesis head ID", _A_HEAD_PREFIX
    )
    _identifier(
        body["oob_high_water_checkpoint_id"],
        "durable OOB checkpoint ID",
        _A_CHECKPOINT_PREFIX,
    )
    _identifier(
        body["oob_high_water_head_id"],
        "durable OOB head ID",
        _A_HEAD_PREFIX,
    )
    candidate_high_water = body["oob_high_water_mapping"]
    if type(candidate_high_water) is not dict:
        _fail(FAIL_SCHEMA_V1, "durable OOB high-water mapping must be an object")
    attempt_context_id = _identifier(
        candidate_high_water.get("attempt_context_id"),
        "durable OOB attempt-context ID",
    )
    high_water = _mapping(
        candidate_high_water,
        "durable policy OOB high-water mapping",
        attempt_context_id,
    )
    if (
        body["bootstrap_external_witness_policy_id"]
        != bootstrap_policy["external_witness_policy_id"]
        or body["bootstrap_external_witness_policy_payload_sha256"]
        != _payload_sha(bootstrap_policy_payload)
        or body["online_witness_policy_id"]
        != online_policy["online_witness_policy_id"]
        or body["genesis_checkpoint_id"] != _A_GENESIS_CHECKPOINT_ID
        or body["genesis_head_id"] != _A_GENESIS_HEAD_ID
        or body["genesis_tree_root_sha256"] != "0" * 64
        or body["oob_high_water_checkpoint_id"] != _A_GENESIS_CHECKPOINT_ID
        or body["oob_high_water_checkpoint_payload_sha256"] != "0" * 64
        or body["oob_high_water_checkpoint_sequence"] != 0
        or body["oob_high_water_head_id"] != _A_GENESIS_HEAD_ID
        or body["oob_high_water_head_payload_sha256"] != "0" * 64
        or body["oob_high_water_head_sequence"] != 0
        or body["oob_high_water_tree_root_sha256"] != "0" * 64
        or high_water != _genesis_mapping(attempt_context_id)
    ):
        _fail(
            FAIL_BINDING_V1,
            "durable policy must be exact genesis/OOB high-water",
        )
    online_body = online_policy["body"]
    if (
        a_public
        in {
            bootstrap_policy["public_key_hex"],
            online_body["online_public_key_hex"],
            online_body["custody_attestor_public_key_hex"],
        }
        or body["anchor_key_id"]
        in {
            bootstrap_policy["witness_key_id"],
            online_body["online_key_id"],
            online_body["custody_attestor_key_id"],
        }
        or body["anchor_control_domain_id"]
        in {
            bootstrap_policy["witness_control_domain_id"],
            online_body["witness_control_domain_id"],
        }
        or body["anchor_namespace"]
        in {
            bootstrap_policy["witness_namespace"],
            online_body["witness_namespace"],
        }
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "A key/control/namespace is not disjoint")
    policy_id = _contract_id("durable_anchor_policy", body)
    if root["durable_anchor_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "durable policy content ID differs")
    preimage = _contract_preimage(
        artifact_kind="durable_anchor_policy",
        body=body,
        artifact_id=policy_id,
        signer_key_id=body["anchor_key_id"],
        key_epoch=body["anchor_key_epoch"],
        purpose=_SIGN_DURABLE_POLICY,
    )
    _signature(
        root["signature"],
        expected_key_id=body["anchor_key_id"],
        expected_key_epoch=body["anchor_key_epoch"],
        expected_purpose=_SIGN_DURABLE_POLICY,
        public_key_hex=a_public,
        preimage=preimage,
        label="durable-anchor policy",
    )
    root["body"] = body
    return root


def _load_policy_closure(
    bootstrap_external_witness_policy_payload: bytes,
    online_witness_policy_payload: bytes,
    durable_anchor_policy_payload: bytes,
    *,
    out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
    out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_online_witness_policy_id_bytes: bytes,
    out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
    out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    bootstrap = _load_bootstrap_policy(
        bootstrap_external_witness_policy_payload,
        expected_id_bytes=(
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes
        ),
        expected_payload_sha256_digest=(
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest
        ),
    )
    online = _load_online_policy(
        online_witness_policy_payload,
        expected_id_bytes=out_of_band_expected_online_witness_policy_id_bytes,
        expected_payload_sha256_digest=(
            out_of_band_expected_online_witness_policy_payload_sha256_digest
        ),
        bootstrap_policy_payload=bootstrap_external_witness_policy_payload,
        bootstrap_policy=bootstrap,
    )
    durable = _load_durable_policy(
        durable_anchor_policy_payload,
        expected_id_bytes=out_of_band_expected_durable_anchor_policy_id_bytes,
        expected_payload_sha256_digest=(
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest
        ),
        bootstrap_policy_payload=bootstrap_external_witness_policy_payload,
        bootstrap_policy=bootstrap,
        online_policy=online,
    )
    return bootstrap, online, durable


def _require_matching_a_private_key(
    durable_anchor_private_key: object, durable_policy: Mapping[str, object]
) -> object:
    if (
        _Ed25519PrivateKey is None
        or _serialization is None
        or not isinstance(durable_anchor_private_key, _Ed25519PrivateKey)
    ):
        _fail(FAIL_KEY_V1, "A actor construction requires Ed25519PrivateKey")
    try:
        public_hex = durable_anchor_private_key.public_key().public_bytes(
            _serialization.Encoding.Raw,
            _serialization.PublicFormat.Raw,
        ).hex()
    except Exception as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_KEY_V1, "A private-key public projection failed"
        ) from exc
    if public_hex != durable_policy["body"]["anchor_public_key_hex"]:
        _fail(FAIL_KEY_V1, "A private key differs from raw-pinned durable policy")
    return durable_anchor_private_key


def _slot_key(attempt_context_id: str, anchor_sequence: int = 1) -> str:
    _identifier(attempt_context_id, "slot attempt-context ID")
    _integer(anchor_sequence, "slot anchor sequence", 1)
    subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-external-admission-a-slot-key/1"
        ),
        "attempt_context_id": attempt_context_id,
        "anchor_sequence": anchor_sequence,
    }
    return _LOCAL_SLOT_KEY_PREFIX + hashlib.sha256(
        _LOCAL_SLOT_KEY_DOMAIN + _canonical(subject)
    ).hexdigest()


def _common_body(
    body: Mapping[str, object],
    *,
    attempt_context_id: str,
    bootstrap_policy_id: str,
    online_policy_id: str,
    durable_policy_id: str,
    label: str,
) -> None:
    for field in (
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
    ):
        _identifier(body[field], f"{label} {field}")
    if (
        body["status"] != _ADMISSION_STATUS
        or body["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or body["attempt_context_id"] != attempt_context_id
        or body["bootstrap_external_witness_policy_id"] != bootstrap_policy_id
        or body["online_witness_policy_id"] != online_policy_id
        or body["durable_anchor_policy_id"] != durable_policy_id
    ):
        _fail(FAIL_BINDING_V1, f"{label} common attempt/policy binding differs")


def _common_projection(body: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        body[field]
        for field in (
            "status",
            "claim_level",
            "stage_b_trust_lock_id",
            "authority_profile_id",
            "signer_registry_id",
            "attempt_context_id",
            "bootstrap_external_witness_policy_id",
            "online_witness_policy_id",
            "durable_anchor_policy_id",
        )
    )


def _runtime_custody_binding(
    value: object, online_body: Mapping[str, object], label: str
) -> dict[str, object]:
    binding = _object(value, frozenset(_RUNTIME_CUSTODY_FIELDS), label)
    for field in (
        "service_binding_id",
        "runtime_instance_id",
        "control_domain_id",
        "runtime_closure_policy_id",
        "runtime_closure_report_id",
        "custody_attestation_policy_id",
        "custody_attestation_id",
    ):
        _identifier(binding[field], f"{label} {field}")
    for field in (
        "peer_pid",
        "peer_uid",
        "peer_gid",
        "user_namespace_inode",
        "process_start_ticks",
    ):
        _integer(binding[field], f"{label} {field}", 1)
    for field in (
        "runtime_closure_report_payload_sha256",
        "custody_attestation_payload_sha256",
    ):
        _hex64(binding[field], f"{label} {field}")
    if (
        binding["service_binding_id"] != online_body["service_binding_id"]
        or binding["peer_uid"] != online_body["service_uid"]
        or binding["peer_gid"] != online_body["service_gid"]
        or binding["user_namespace_inode"]
        != online_body["service_user_namespace_inode"]
        or binding["control_domain_id"]
        != online_body["witness_control_domain_id"]
        or binding["runtime_closure_policy_id"]
        != online_body["runtime_closure_policy_id"]
        or binding["custody_attestation_policy_id"]
        != online_body["custody_attestation_policy_id"]
        or binding["online_key_nonexportable"] is not True
    ):
        _fail(FAIL_BINDING_V1, f"{label} policy projection differs")
    return binding


def _load_request_closure(
    q1_prepare_request_payload: bytes,
    q1_prepare_response_payload: bytes,
    a_compare_and_append_request_payload: bytes,
    *,
    bootstrap_policy: Mapping[str, object],
    online_policy: Mapping[str, object],
    durable_policy: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    online_body = online_policy["body"]
    durable_body = durable_policy["body"]
    attempt_context_id = durable_body["oob_high_water_mapping"][
        "attempt_context_id"
    ]
    common = {
        "attempt_context_id": attempt_context_id,
        "bootstrap_policy_id": bootstrap_policy["external_witness_policy_id"],
        "online_policy_id": online_policy["online_witness_policy_id"],
        "durable_policy_id": durable_policy["durable_anchor_policy_id"],
    }
    q1_request, q1_request_body = _load_signed_root(
        q1_prepare_request_payload,
        label="Q1 PREPARE request",
        schema=_Q1_REQUEST_SCHEMA,
        fields=frozenset(_Q1_REQUEST_FIELDS),
        body_fields=frozenset(_Q1_REQUEST_BODY_FIELDS),
        artifact_kind="q1_prepare_request",
        id_field="q1_request_id",
        signature_field="signature",
        expected_key_id=bootstrap_policy["witness_key_id"],
        expected_key_epoch=bootstrap_policy["witness_key_epoch"],
        expected_purpose=_SIGN_Q1_REQUEST,
        public_key_hex=bootstrap_policy["public_key_hex"],
    )
    _common_body(q1_request_body, label="Q1 request", **common)
    prepare_nonce = _nonce(q1_request_body["prepare_nonce"], "Q1 prepare nonce")
    candidate_mapping = _mapping(
        q1_request_body["candidate_mapping"],
        "Q1 candidate mapping",
        attempt_context_id,
    )
    if (
        candidate_mapping["anchor_sequence"] != 1
        or q1_request_body["expected_previous_durable_checkpoint_id"]
        != durable_body["oob_high_water_checkpoint_id"]
        or q1_request_body["expected_previous_durable_head_id"]
        != durable_body["oob_high_water_head_id"]
        or q1_request_body[
            "expected_previous_durable_checkpoint_payload_sha256"
        ]
        != durable_body["oob_high_water_checkpoint_payload_sha256"]
        or q1_request_body["expected_previous_durable_head_payload_sha256"]
        != durable_body["oob_high_water_head_payload_sha256"]
        or q1_request_body["expected_previous_durable_checkpoint_sequence"]
        != durable_body["oob_high_water_checkpoint_sequence"]
        or q1_request_body["expected_previous_durable_head_sequence"]
        != durable_body["oob_high_water_head_sequence"]
        or q1_request_body["expected_previous_durable_tree_root_sha256"]
        != durable_body["oob_high_water_tree_root_sha256"]
    ):
        _fail(FAIL_BINDING_V1, "Q1 request candidate/high-water differs")

    q1_response, q1_response_body = _load_signed_root(
        q1_prepare_response_payload,
        label="Q1 PREPARE response",
        schema=_Q1_RESPONSE_SCHEMA,
        fields=frozenset(_Q1_RESPONSE_FIELDS),
        body_fields=frozenset(_Q1_RESPONSE_BODY_FIELDS),
        artifact_kind="q1_prepare_response",
        id_field="q1_response_id",
        signature_field="signature",
        expected_key_id=online_body["online_key_id"],
        expected_key_epoch=online_body["online_key_epoch"],
        expected_purpose=_SIGN_Q1_RESPONSE,
        public_key_hex=online_body["online_public_key_hex"],
    )
    _common_body(q1_response_body, label="Q1 response", **common)
    if _common_projection(q1_response_body) != _common_projection(q1_request_body):
        _fail(FAIL_BINDING_V1, "Q1 response common authority triple differs")
    confirmation_nonce = _nonce(
        q1_response_body["confirmation_nonce"], "Q1 confirmation nonce"
    )
    if confirmation_nonce == prepare_nonce:
        _fail(FAIL_BINDING_V1, "prepare and confirmation nonces must be distinct")
    _runtime_custody_binding(
        q1_response_body["runtime_custody_binding"],
        online_body,
        "Q1 runtime/custody binding",
    )
    if (
        q1_response_body["q1_request_id"] != q1_request["q1_request_id"]
        or q1_response_body["q1_request_payload_sha256"]
        != _payload_sha(q1_prepare_request_payload)
        or q1_response_body["prepare_nonce"] != prepare_nonce
        or _mapping(
            q1_response_body["observed_current_mapping"],
            "Q1 observed mapping",
            attempt_context_id,
        )
        != candidate_mapping
        or _mapping(
            q1_response_body["prepared_candidate_mapping"],
            "Q1 prepared candidate mapping",
            attempt_context_id,
        )
        != candidate_mapping
        or q1_response_body["prepare_outcome"] != _PREPARE_OUTCOME
    ):
        _fail(FAIL_BINDING_V1, "Q1 response linkage/outcome differs")

    a_request, a_request_body = _load_signed_root(
        a_compare_and_append_request_payload,
        label="A compare-and-append request",
        schema=_A_REQUEST_SCHEMA,
        fields=frozenset(_A_REQUEST_FIELDS),
        body_fields=frozenset(_A_REQUEST_BODY_FIELDS),
        artifact_kind="a_compare_and_append_request",
        id_field="a_request_id",
        signature_field="signature",
        expected_key_id=online_body["online_key_id"],
        expected_key_epoch=online_body["online_key_epoch"],
        expected_purpose=_SIGN_A_REQUEST,
        public_key_hex=online_body["online_public_key_hex"],
    )
    _common_body(a_request_body, label="A request", **common)
    if _common_projection(a_request_body) != _common_projection(q1_request_body):
        _fail(FAIL_BINDING_V1, "A request common authority triple differs")
    expected_mapping = _mapping(
        a_request_body["expected_current_mapping"],
        "A expected current mapping",
        attempt_context_id,
    )
    proposed_mapping = _mapping(
        a_request_body["proposed_current_mapping"],
        "A proposed current mapping",
        attempt_context_id,
    )
    if (
        a_request_body["q1_request_id"] != q1_request["q1_request_id"]
        or a_request_body["q1_request_payload_sha256"]
        != _payload_sha(q1_prepare_request_payload)
        or a_request_body["q1_response_id"] != q1_response["q1_response_id"]
        or a_request_body["q1_response_payload_sha256"]
        != _payload_sha(q1_prepare_response_payload)
        or a_request_body["prepare_nonce"] != prepare_nonce
        or a_request_body["confirmation_nonce"] != confirmation_nonce
        or expected_mapping != durable_body["oob_high_water_mapping"]
        or expected_mapping != _genesis_mapping(attempt_context_id)
        or proposed_mapping != candidate_mapping
        or a_request_body["expected_previous_durable_checkpoint_id"]
        != durable_body["oob_high_water_checkpoint_id"]
        or a_request_body[
            "expected_previous_durable_checkpoint_payload_sha256"
        ]
        != durable_body["oob_high_water_checkpoint_payload_sha256"]
        or a_request_body["expected_previous_durable_head_id"]
        != durable_body["oob_high_water_head_id"]
        or a_request_body["expected_previous_durable_head_payload_sha256"]
        != durable_body["oob_high_water_head_payload_sha256"]
        or a_request_body["expected_previous_durable_checkpoint_sequence"]
        != durable_body["oob_high_water_checkpoint_sequence"]
        or a_request_body["expected_previous_durable_head_sequence"]
        != durable_body["oob_high_water_head_sequence"]
        or a_request_body["expected_previous_durable_tree_root_sha256"]
        != durable_body["oob_high_water_tree_root_sha256"]
        or a_request_body["compare_and_append_mode"]
        != LOCAL_COMPARE_AND_APPEND_SEMANTICS_V1
    ):
        _fail(FAIL_BINDING_V1, "A request Q1/genesis/CAS binding differs")
    return (
        q1_request,
        q1_request_body,
        q1_response,
        q1_response_body,
        a_request,
        a_request_body,
    )


def _protocol_common_from_a_request(
    a_request_body: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": a_request_body["status"],
        "claim_level": a_request_body["claim_level"],
        "stage_b_trust_lock_id": a_request_body["stage_b_trust_lock_id"],
        "authority_profile_id": a_request_body["authority_profile_id"],
        "signer_registry_id": a_request_body["signer_registry_id"],
        "attempt_context_id": a_request_body["attempt_context_id"],
        "bootstrap_external_witness_policy_id": a_request_body[
            "bootstrap_external_witness_policy_id"
        ],
        "online_witness_policy_id": a_request_body["online_witness_policy_id"],
        "durable_anchor_policy_id": a_request_body["durable_anchor_policy_id"],
    }


def _expected_a_output_bodies(
    *,
    a_request_payload: bytes,
    a_request: Mapping[str, object],
    a_request_body: Mapping[str, object],
    durable_policy: Mapping[str, object],
    checkpoint_identity: tuple[str, str] | None = None,
    head_identity: tuple[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    durable_body = durable_policy["body"]
    common = _protocol_common_from_a_request(a_request_body)
    checkpoint_body = {
        **common,
        "schema_version": _A_CHECKPOINT_SCHEMA,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _payload_sha(a_request_payload),
        "prepare_nonce": a_request_body["prepare_nonce"],
        "confirmation_nonce": a_request_body["confirmation_nonce"],
        "previous_mapping": a_request_body["expected_current_mapping"],
        "committed_mapping": a_request_body["proposed_current_mapping"],
        "cas_outcome": _CAS_OUTCOME,
        "checkpoint_sequence": (
            durable_body["oob_high_water_checkpoint_sequence"] + 1
        ),
        "previous_a_checkpoint_id": durable_body[
            "oob_high_water_checkpoint_id"
        ],
    }
    if checkpoint_identity is None:
        checkpoint_id = _contract_id("a_durable_checkpoint", checkpoint_body)
        checkpoint_sha = ""
    else:
        checkpoint_id, checkpoint_sha = checkpoint_identity
    head_body = {
        **common,
        "schema_version": _A_HEAD_SCHEMA,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _payload_sha(a_request_payload),
        "a_checkpoint_id": checkpoint_id,
        "a_checkpoint_payload_sha256": checkpoint_sha,
        "prepare_nonce": a_request_body["prepare_nonce"],
        "confirmation_nonce": a_request_body["confirmation_nonce"],
        "committed_mapping": a_request_body["proposed_current_mapping"],
        "cas_outcome": _CAS_OUTCOME,
        "head_sequence": durable_body["oob_high_water_head_sequence"] + 1,
        "previous_a_head_id": durable_body["oob_high_water_head_id"],
    }
    if head_identity is None:
        head_id = _contract_id("a_durable_head", head_body)
        head_sha = ""
    else:
        head_id, head_sha = head_identity
    receipt_body = {
        **common,
        "schema_version": _A_RECEIPT_SCHEMA,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _payload_sha(a_request_payload),
        "a_checkpoint_id": checkpoint_id,
        "a_checkpoint_payload_sha256": checkpoint_sha,
        "a_head_id": head_id,
        "a_head_payload_sha256": head_sha,
        "prepare_nonce": a_request_body["prepare_nonce"],
        "confirmation_nonce": a_request_body["confirmation_nonce"],
        "previous_mapping": a_request_body["expected_current_mapping"],
        "committed_mapping": a_request_body["proposed_current_mapping"],
        "cas_outcome": _CAS_OUTCOME,
        "compare_matched": True,
        "append_applied": True,
        "oob_high_water_checkpoint_id": durable_body[
            "oob_high_water_checkpoint_id"
        ],
        "oob_high_water_head_id": durable_body["oob_high_water_head_id"],
        # This statement becomes returnable only after all records, including
        # the head commit marker, pass fsync/reopen/readback validation.
        "durable_before_return": True,
    }
    return checkpoint_body, head_body, receipt_body


def _sign_protocol_object(
    *,
    artifact_kind: str,
    schema: str,
    id_field: str,
    body: Mapping[str, object],
    signer_key_id: str,
    key_epoch: int,
    purpose: str,
    private_key: object,
    public_key_hex: str,
) -> tuple[bytes, dict[str, object]]:
    exact_body = dict(body)
    artifact_id = _contract_id(artifact_kind, exact_body)
    preimage = _contract_preimage(
        artifact_kind=artifact_kind,
        body=exact_body,
        artifact_id=artifact_id,
        signer_key_id=signer_key_id,
        key_epoch=key_epoch,
        purpose=purpose,
    )
    try:
        signature_bytes = private_key.sign(preimage)
    except Exception as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_KEY_V1, f"A signing failed for {artifact_kind}"
        ) from exc
    if type(signature_bytes) is not bytes or len(signature_bytes) != 64:
        _fail(FAIL_SIGNATURE_V1, f"A signature length differs for {artifact_kind}")
    signature = {
        "schema_version": _SIGNATURE_SCHEMA,
        "algorithm": _ALGORITHM,
        "signer_key_id": signer_key_id,
        "key_epoch": key_epoch,
        "purpose": purpose,
        "signature_hex": signature_bytes.hex(),
    }
    _signature(
        signature,
        expected_key_id=signer_key_id,
        expected_key_epoch=key_epoch,
        expected_purpose=purpose,
        public_key_hex=public_key_hex,
        preimage=preimage,
        label=artifact_kind,
    )
    root = {
        "schema_version": schema,
        "body": exact_body,
        id_field: artifact_id,
        "signature": signature,
    }
    try:
        payload = _contract_canonical(root)
    except Exception as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_CANONICAL_V1, f"contract canonicalization failed for {artifact_kind}"
        ) from exc
    if type(payload) is not bytes or _decode(payload, artifact_kind) != root:
        _fail(FAIL_CANONICAL_V1, f"signed {artifact_kind} readback differs")
    return payload, root


def _compose_a_outputs(
    *,
    a_request_payload: bytes,
    a_request: Mapping[str, object],
    a_request_body: Mapping[str, object],
    durable_policy: Mapping[str, object],
    private_key: object,
) -> tuple[
    bytes,
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    dict[str, object],
]:
    durable_body = durable_policy["body"]
    checkpoint_body, _, _ = _expected_a_output_bodies(
        a_request_payload=a_request_payload,
        a_request=a_request,
        a_request_body=a_request_body,
        durable_policy=durable_policy,
    )
    checkpoint_payload, checkpoint = _sign_protocol_object(
        artifact_kind="a_durable_checkpoint",
        schema=_A_CHECKPOINT_SCHEMA,
        id_field="a_checkpoint_id",
        body=checkpoint_body,
        signer_key_id=durable_body["anchor_key_id"],
        key_epoch=durable_body["anchor_key_epoch"],
        purpose=_SIGN_A_CHECKPOINT,
        private_key=private_key,
        public_key_hex=durable_body["anchor_public_key_hex"],
    )
    _, head_body, _ = _expected_a_output_bodies(
        a_request_payload=a_request_payload,
        a_request=a_request,
        a_request_body=a_request_body,
        durable_policy=durable_policy,
        checkpoint_identity=(
            checkpoint["a_checkpoint_id"],
            _payload_sha(checkpoint_payload),
        ),
    )
    head_payload, head = _sign_protocol_object(
        artifact_kind="a_durable_head",
        schema=_A_HEAD_SCHEMA,
        id_field="a_head_id",
        body=head_body,
        signer_key_id=durable_body["anchor_key_id"],
        key_epoch=durable_body["anchor_key_epoch"],
        purpose=_SIGN_A_HEAD,
        private_key=private_key,
        public_key_hex=durable_body["anchor_public_key_hex"],
    )
    _, _, receipt_body = _expected_a_output_bodies(
        a_request_payload=a_request_payload,
        a_request=a_request,
        a_request_body=a_request_body,
        durable_policy=durable_policy,
        checkpoint_identity=(
            checkpoint["a_checkpoint_id"],
            _payload_sha(checkpoint_payload),
        ),
        head_identity=(head["a_head_id"], _payload_sha(head_payload)),
    )
    receipt_payload, receipt = _sign_protocol_object(
        artifact_kind="a_compare_and_append_receipt",
        schema=_A_RECEIPT_SCHEMA,
        id_field="a_receipt_id",
        body=receipt_body,
        signer_key_id=durable_body["anchor_key_id"],
        key_epoch=durable_body["anchor_key_epoch"],
        purpose=_SIGN_A_RECEIPT,
        private_key=private_key,
        public_key_hex=durable_body["anchor_public_key_hex"],
    )
    return (
        checkpoint_payload,
        checkpoint,
        head_payload,
        head,
        receipt_payload,
        receipt,
    )


def _verify_stored_a_outputs(
    *,
    checkpoint_payload: bytes,
    head_payload: bytes,
    receipt_payload: bytes,
    a_request_payload: bytes,
    a_request: Mapping[str, object],
    a_request_body: Mapping[str, object],
    durable_policy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    durable_body = durable_policy["body"]
    checkpoint, checkpoint_body = _load_signed_root(
        checkpoint_payload,
        label="stored A checkpoint",
        schema=_A_CHECKPOINT_SCHEMA,
        fields=frozenset(_A_CHECKPOINT_FIELDS),
        body_fields=frozenset(_A_CHECKPOINT_BODY_FIELDS),
        artifact_kind="a_durable_checkpoint",
        id_field="a_checkpoint_id",
        signature_field="signature",
        expected_key_id=durable_body["anchor_key_id"],
        expected_key_epoch=durable_body["anchor_key_epoch"],
        expected_purpose=_SIGN_A_CHECKPOINT,
        public_key_hex=durable_body["anchor_public_key_hex"],
    )
    head, head_body = _load_signed_root(
        head_payload,
        label="stored A head",
        schema=_A_HEAD_SCHEMA,
        fields=frozenset(_A_HEAD_FIELDS),
        body_fields=frozenset(_A_HEAD_BODY_FIELDS),
        artifact_kind="a_durable_head",
        id_field="a_head_id",
        signature_field="signature",
        expected_key_id=durable_body["anchor_key_id"],
        expected_key_epoch=durable_body["anchor_key_epoch"],
        expected_purpose=_SIGN_A_HEAD,
        public_key_hex=durable_body["anchor_public_key_hex"],
    )
    receipt, receipt_body = _load_signed_root(
        receipt_payload,
        label="stored A receipt",
        schema=_A_RECEIPT_SCHEMA,
        fields=frozenset(_A_RECEIPT_FIELDS),
        body_fields=frozenset(_A_RECEIPT_BODY_FIELDS),
        artifact_kind="a_compare_and_append_receipt",
        id_field="a_receipt_id",
        signature_field="signature",
        expected_key_id=durable_body["anchor_key_id"],
        expected_key_epoch=durable_body["anchor_key_epoch"],
        expected_purpose=_SIGN_A_RECEIPT,
        public_key_hex=durable_body["anchor_public_key_hex"],
    )
    expected_checkpoint_body, expected_head_body, expected_receipt_body = (
        _expected_a_output_bodies(
            a_request_payload=a_request_payload,
            a_request=a_request,
            a_request_body=a_request_body,
            durable_policy=durable_policy,
            checkpoint_identity=(
                checkpoint["a_checkpoint_id"],
                _payload_sha(checkpoint_payload),
            ),
            head_identity=(head["a_head_id"], _payload_sha(head_payload)),
        )
    )
    if (
        checkpoint_body != expected_checkpoint_body
        or head_body != expected_head_body
        or receipt_body != expected_receipt_body
        or head_body["a_checkpoint_id"] != checkpoint["a_checkpoint_id"]
        or head_body["a_checkpoint_payload_sha256"]
        != _payload_sha(checkpoint_payload)
        or receipt_body["a_head_id"] != head["a_head_id"]
        or receipt_body["a_head_payload_sha256"] != _payload_sha(head_payload)
        or receipt_body["durable_before_return"] is not True
    ):
        _fail(FAIL_POISONED_V1, "stored A checkpoint/head/receipt lineage differs")
    return checkpoint, head, receipt


def _require_linux_open_flags() -> None:
    for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"):
        if not hasattr(os, name):
            _fail(FAIL_FILESYSTEM_V1, f"required Linux open flag {name} unavailable")


def _validate_private_directory_stat(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        _fail(
            FAIL_FILESYSTEM_V1,
            "A spool must be an existing owner-owned exact-0700 directory",
        )


def _open_owner_private_directory(path: str) -> int:
    _require_linux_open_flags()
    if (
        type(path) is not str
        or not path.startswith("/")
        or "\x00" in path
        or not 1 <= len(path.encode("utf-8")) <= _MAXIMUM_PATH_BYTES
        or not os.path.isabs(path)
    ):
        _fail(FAIL_INPUT_V1, "A spool path must be bounded exact absolute text")
    try:
        before = os.stat(path, follow_symlinks=False)
        _validate_private_directory_stat(before)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        directory_fd = os.open(path, flags)
        after = os.fstat(directory_fd)
        _validate_private_directory_stat(after)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            os.close(directory_fd)
            _fail(FAIL_FILESYSTEM_V1, "A spool path changed while opening")
        return directory_fd
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, "A spool directory open failed closed"
        ) from exc


def _basename(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _SAFE_BASENAME.fullmatch(value) is None
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
    ):
        _fail(FAIL_SCHEMA_V1, f"{label} must be a safe fixed basename")
    return value


def _validate_regular_record_stat(
    metadata: os.stat_result, *, expected_size: int | None, label: str
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_size < 1
        or metadata.st_size > _MAXIMUM_SPOOL_RECORD_BYTES
        or (expected_size is not None and metadata.st_size != expected_size)
    ):
        _fail(FAIL_POISONED_V1, f"{label} metadata/link-count differs")


def _read_bounded_record(directory_fd: int, basename: str) -> bytes:
    name = _basename(basename, "spool record basename")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        record_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_POISONED_V1, f"spool record {name} cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(record_fd)
        _validate_regular_record_stat(before, expected_size=None, label=name)
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(record_fd, min(remaining, 1024 * 1024))
            if type(chunk) is not bytes or not chunk:
                _fail(FAIL_POISONED_V1, f"spool record {name} read was short")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(record_fd, 1) != b"":
            _fail(FAIL_POISONED_V1, f"spool record {name} grew during read")
        after = os.fstat(record_fd)
        _validate_regular_record_stat(
            after, expected_size=before.st_size, label=name
        )
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(FAIL_POISONED_V1, f"spool record {name} changed during read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            _fail(FAIL_POISONED_V1, f"spool record {name} length differs")
        return payload
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_POISONED_V1, f"spool record {name} read failed closed"
        ) from exc
    finally:
        os.close(record_fd)


def _read_exact_record(directory_fd: int, basename: str, expected: bytes) -> bytes:
    if (
        type(expected) is not bytes
        or not 1 <= len(expected) <= _MAXIMUM_SPOOL_RECORD_BYTES
    ):
        _fail(FAIL_INPUT_V1, "expected spool record must be bounded exact bytes")
    payload = _read_bounded_record(directory_fd, basename)
    if payload != expected:
        _fail(FAIL_POISONED_V1, f"spool record {basename} byte readback differs")
    return payload


def _write_all(record_fd: int, payload: bytes, label: str) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(record_fd, view[offset:])
        except OSError as exc:
            raise Phase2BStep5R01DurableAnchorErrorV1(
                FAIL_FILESYSTEM_V1, f"{label} write failed closed"
            ) from exc
        if type(written) is not int or written <= 0 or written > len(view) - offset:
            _fail(FAIL_FILESYSTEM_V1, f"{label} write made invalid progress")
        offset += written


def _write_exclusive_record(
    directory_fd: int, basename: str, payload: bytes
) -> None:
    name = _basename(basename, "exclusive spool basename")
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAXIMUM_SPOOL_RECORD_BYTES
    ):
        _fail(FAIL_INPUT_V1, "exclusive spool payload must be bounded exact bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        record_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError:
        _fail(FAIL_CONFLICT_V1, f"exclusive spool record {name} already exists")
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, f"exclusive spool record {name} create failed"
        ) from exc
    try:
        os.fchmod(record_fd, 0o600)
        created = os.fstat(record_fd)
        if (
            not stat.S_ISREG(created.st_mode)
            or stat.S_IMODE(created.st_mode) != 0o600
            or created.st_uid != os.geteuid()
            or created.st_nlink != 1
            or created.st_size != 0
        ):
            _fail(FAIL_FILESYSTEM_V1, f"new spool record {name} metadata differs")
        _write_all(record_fd, payload, name)
        os.fsync(record_fd)
        final = os.fstat(record_fd)
        _validate_regular_record_stat(
            final, expected_size=len(payload), label=name
        )
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, f"exclusive spool record {name} fsync failed"
        ) from exc
    finally:
        os.close(record_fd)
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, f"exclusive spool record {name} directory fsync failed"
        ) from exc
    _read_exact_record(directory_fd, name, payload)


def _staging_basename(final_basename: str) -> str:
    final_name = _basename(final_basename, "final publication basename")
    return "stage-v1-" + hashlib.sha256(final_name.encode("ascii")).hexdigest()


def _publish_staged_no_replace(
    directory_fd: int, final_basename: str, payload: bytes
) -> None:
    final_name = _basename(final_basename, "final publication basename")
    stage_name = _staging_basename(final_name)
    _write_exclusive_record(directory_fd, stage_name, payload)
    try:
        os.link(
            stage_name,
            final_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
        linked = os.stat(final_name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(linked.st_mode)
            or stat.S_IMODE(linked.st_mode) != 0o600
            or linked.st_uid != os.geteuid()
            or linked.st_nlink != 2
            or linked.st_size != len(payload)
        ):
            _fail(FAIL_FILESYSTEM_V1, f"published link {final_name} differs")
        os.unlink(stage_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1,
            f"atomic no-replace publication for {final_name} failed closed",
        ) from exc
    _read_exact_record(directory_fd, final_name, payload)


def _directory_entries(directory_fd: int) -> tuple[str, ...]:
    try:
        entries = os.listdir(directory_fd)
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, "A spool namespace listing failed closed"
        ) from exc
    if type(entries) is not list or any(type(name) is not str for name in entries):
        _fail(FAIL_FILESYSTEM_V1, "A spool namespace listing shape differs")
    if len(entries) != len(set(entries)):
        _fail(FAIL_POISONED_V1, "A spool namespace contains duplicate names")
    return tuple(entries)


def _lock_directory_exclusive(directory_fd: int) -> tuple[int, int]:
    try:
        before = os.fstat(directory_fd)
        _validate_private_directory_stat(before)
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        after = os.fstat(directory_fd)
        _validate_private_directory_stat(after)
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except (OSError, ValueError) as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, "exclusive A spool directory lock failed closed"
        ) from exc
    identity = (after.st_dev, after.st_ino)
    if identity != (before.st_dev, before.st_ino):
        _fail(FAIL_FILESYSTEM_V1, "A spool directory changed while locking")
    return identity


def _artifact_basename(label: str, artifact_id: str) -> str:
    _text(label, "artifact basename label", 64)
    _identifier(artifact_id, f"{label} artifact ID")
    digest = artifact_id.rsplit("_", 1)[-1]
    _hex64(digest, f"{label} artifact-ID digest")
    return _basename(f"{label}-v1-{digest}.json", f"{label} basename")


def _intent_basename(slot_key: str) -> str:
    _identifier(slot_key, "local A slot key", _LOCAL_SLOT_KEY_PREFIX)
    digest = slot_key[len(_LOCAL_SLOT_KEY_PREFIX) :]
    _hex64(digest, "local A slot-key digest")
    return _basename(f"a-slot-intent-v1-{digest}.json", "slot intent basename")


def _slot_intent_id(intent_without_id: Mapping[str, object]) -> str:
    return _LOCAL_SLOT_INTENT_PREFIX + hashlib.sha256(
        _LOCAL_SLOT_INTENT_ID_DOMAIN + _canonical(intent_without_id)
    ).hexdigest()


def _compose_slot_intent(
    *,
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
    online_policy_payload: bytes,
    online_policy: Mapping[str, object],
    durable_policy_payload: bytes,
    durable_policy: Mapping[str, object],
    q1_request_payload: bytes,
    q1_request: Mapping[str, object],
    q1_response_payload: bytes,
    q1_response: Mapping[str, object],
    a_request_payload: bytes,
    a_request: Mapping[str, object],
    checkpoint_payload: bytes,
    checkpoint: Mapping[str, object],
    head_payload: bytes,
    head: Mapping[str, object],
    receipt_payload: bytes,
    receipt: Mapping[str, object],
) -> tuple[bytes, dict[str, object], str, tuple[str, ...]]:
    attempt_context_id = durable_policy["body"]["oob_high_water_mapping"][
        "attempt_context_id"
    ]
    slot_key = _slot_key(attempt_context_id, 1)
    q1_request_basename = _artifact_basename(
        "q1-request", q1_request["q1_request_id"]
    )
    q1_response_basename = _artifact_basename(
        "q1-response", q1_response["q1_response_id"]
    )
    a_request_basename = _artifact_basename(
        "a-request", a_request["a_request_id"]
    )
    checkpoint_basename = _artifact_basename(
        "a-checkpoint", checkpoint["a_checkpoint_id"]
    )
    receipt_basename = _artifact_basename("a-receipt", receipt["a_receipt_id"])
    head_basename = _artifact_basename("a-head", head["a_head_id"])
    publication_order = (
        q1_request_basename,
        q1_response_basename,
        a_request_basename,
        checkpoint_basename,
        receipt_basename,
        head_basename,
    )
    intent: dict[str, object] = {
        "schema_version": LOCAL_DURABLE_ANCHOR_SLOT_INTENT_SCHEMA_V1,
        "status": _ADMISSION_STATUS,
        "claim_level": _ADMISSION_CLAIM_LEVEL,
        "mechanics_claim_boundary": LOCAL_MECHANICS_CLAIM_BOUNDARY_V1,
        "slot_key": slot_key,
        "attempt_context_id": attempt_context_id,
        "anchor_sequence": 1,
        "bootstrap_external_witness_policy_id": bootstrap_policy[
            "external_witness_policy_id"
        ],
        "bootstrap_external_witness_policy_payload_sha256": _payload_sha(
            bootstrap_policy_payload
        ),
        "online_witness_policy_id": online_policy["online_witness_policy_id"],
        "online_witness_policy_payload_sha256": _payload_sha(online_policy_payload),
        "durable_anchor_policy_id": durable_policy["durable_anchor_policy_id"],
        "durable_anchor_policy_payload_sha256": _payload_sha(
            durable_policy_payload
        ),
        "q1_request_id": q1_request["q1_request_id"],
        "q1_request_payload_sha256": _payload_sha(q1_request_payload),
        "q1_response_id": q1_response["q1_response_id"],
        "q1_response_payload_sha256": _payload_sha(q1_response_payload),
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _payload_sha(a_request_payload),
        "a_checkpoint_id": checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _payload_sha(checkpoint_payload),
        "a_head_id": head["a_head_id"],
        "a_head_payload_sha256": _payload_sha(head_payload),
        "a_receipt_id": receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _payload_sha(receipt_payload),
        "q1_request_file_basename": q1_request_basename,
        "q1_response_file_basename": q1_response_basename,
        "a_request_file_basename": a_request_basename,
        "a_checkpoint_file_basename": checkpoint_basename,
        "a_receipt_file_basename": receipt_basename,
        "a_head_file_basename": head_basename,
        "publication_order_in_exact_order": list(publication_order),
        "head_is_sole_commit_marker": True,
    }
    intent["slot_intent_id"] = _slot_intent_id(intent)
    if frozenset(intent) != _SLOT_INTENT_FIELDS:
        _fail(FAIL_SCHEMA_V1, "composed local slot-intent keyset differs")
    payload = _canonical(intent)
    if _decode(payload, "composed local slot intent") != intent:
        _fail(FAIL_CANONICAL_V1, "composed local slot-intent readback differs")
    return payload, intent, _intent_basename(slot_key), publication_order


def _load_slot_intent(
    payload: bytes, *, expected_slot_key: str
) -> dict[str, object]:
    intent = _object(
        _decode(payload, "local A slot intent"),
        _SLOT_INTENT_FIELDS,
        "local A slot intent",
    )
    if (
        intent["schema_version"] != LOCAL_DURABLE_ANCHOR_SLOT_INTENT_SCHEMA_V1
        or intent["status"] != _ADMISSION_STATUS
        or intent["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or intent["mechanics_claim_boundary"]
        != LOCAL_MECHANICS_CLAIM_BOUNDARY_V1
        or intent["slot_key"] != expected_slot_key
        or type(intent["anchor_sequence"]) is not int
        or intent["anchor_sequence"] != 1
        or intent["head_is_sole_commit_marker"] is not True
    ):
        _fail(FAIL_POISONED_V1, "local A slot-intent header differs")
    _identifier(intent["attempt_context_id"], "slot-intent attempt-context ID")
    for field, prefix in (
        (
            "bootstrap_external_witness_policy_id",
            _BOOTSTRAP_POLICY_PREFIX,
        ),
        ("online_witness_policy_id", _ONLINE_POLICY_PREFIX),
        ("durable_anchor_policy_id", _DURABLE_POLICY_PREFIX),
        ("q1_request_id", _Q1_REQUEST_PREFIX),
        ("q1_response_id", _Q1_RESPONSE_PREFIX),
        ("a_request_id", _A_REQUEST_PREFIX),
        ("a_checkpoint_id", _A_CHECKPOINT_PREFIX),
        ("a_head_id", _A_HEAD_PREFIX),
        ("a_receipt_id", _A_RECEIPT_PREFIX),
        ("slot_intent_id", _LOCAL_SLOT_INTENT_PREFIX),
    ):
        _identifier(intent[field], f"slot intent {field}", prefix)
    for field in (
        "bootstrap_external_witness_policy_payload_sha256",
        "online_witness_policy_payload_sha256",
        "durable_anchor_policy_payload_sha256",
        "q1_request_payload_sha256",
        "q1_response_payload_sha256",
        "a_request_payload_sha256",
        "a_checkpoint_payload_sha256",
        "a_head_payload_sha256",
        "a_receipt_payload_sha256",
    ):
        _hex64(intent[field], f"slot intent {field}")
    basename_fields = (
        "q1_request_file_basename",
        "q1_response_file_basename",
        "a_request_file_basename",
        "a_checkpoint_file_basename",
        "a_receipt_file_basename",
        "a_head_file_basename",
    )
    basenames = tuple(
        _basename(intent[field], f"slot intent {field}") for field in basename_fields
    )
    if (
        type(intent["publication_order_in_exact_order"]) is not list
        or tuple(intent["publication_order_in_exact_order"]) != basenames
        or len(set(basenames)) != len(basenames)
        or basenames
        != (
            _artifact_basename("q1-request", intent["q1_request_id"]),
            _artifact_basename("q1-response", intent["q1_response_id"]),
            _artifact_basename("a-request", intent["a_request_id"]),
            _artifact_basename("a-checkpoint", intent["a_checkpoint_id"]),
            _artifact_basename("a-receipt", intent["a_receipt_id"]),
            _artifact_basename("a-head", intent["a_head_id"]),
        )
    ):
        _fail(FAIL_POISONED_V1, "local A slot-intent publication order differs")
    expected_intent_id = _slot_intent_id(_without(intent, "slot_intent_id"))
    if intent["slot_intent_id"] != expected_intent_id:
        _fail(FAIL_POISONED_V1, "local A slot-intent content ID differs")
    return intent


def _load_existing_complete_slot(
    directory_fd: int,
    *,
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
    online_policy_payload: bytes,
    online_policy: Mapping[str, object],
    durable_policy_payload: bytes,
    durable_policy: Mapping[str, object],
    q1_request_payload: bytes,
    q1_request: Mapping[str, object],
    q1_response_payload: bytes,
    q1_response: Mapping[str, object],
    a_request_payload: bytes,
    a_request: Mapping[str, object],
    a_request_body: Mapping[str, object],
) -> tuple[bytes, bytes, bytes, tuple[str, ...]] | None:
    attempt_context_id = durable_policy["body"]["oob_high_water_mapping"][
        "attempt_context_id"
    ]
    slot_key = _slot_key(attempt_context_id, 1)
    intent_basename = _intent_basename(slot_key)
    entries = _directory_entries(directory_fd)
    if not entries:
        return None
    if intent_basename not in entries:
        _fail(
            FAIL_POISONED_V1,
            "nonempty A spool lacks the exact durable slot intent",
        )
    intent_payload = _read_bounded_record(directory_fd, intent_basename)
    intent = _load_slot_intent(intent_payload, expected_slot_key=slot_key)
    publication_order = tuple(intent["publication_order_in_exact_order"])
    expected_entries = {intent_basename, *publication_order}
    if set(entries) != expected_entries or len(entries) != len(expected_entries):
        _fail(
            FAIL_POISONED_V1,
            "A slot is partial, gapped, staged, hardlinked, or has extra namespace",
        )
    policy_bindings = (
        intent["attempt_context_id"] == attempt_context_id,
        intent["bootstrap_external_witness_policy_id"]
        == bootstrap_policy["external_witness_policy_id"],
        intent["bootstrap_external_witness_policy_payload_sha256"]
        == _payload_sha(bootstrap_policy_payload),
        intent["online_witness_policy_id"]
        == online_policy["online_witness_policy_id"],
        intent["online_witness_policy_payload_sha256"]
        == _payload_sha(online_policy_payload),
        intent["durable_anchor_policy_id"]
        == durable_policy["durable_anchor_policy_id"],
        intent["durable_anchor_policy_payload_sha256"]
        == _payload_sha(durable_policy_payload),
    )
    if not all(policy_bindings):
        _fail(FAIL_CONFLICT_V1, "existing A slot is bound to different raw policies")
    request_bindings = (
        intent["q1_request_id"] == q1_request["q1_request_id"],
        intent["q1_request_payload_sha256"] == _payload_sha(q1_request_payload),
        intent["q1_response_id"] == q1_response["q1_response_id"],
        intent["q1_response_payload_sha256"] == _payload_sha(q1_response_payload),
        intent["a_request_id"] == a_request["a_request_id"],
        intent["a_request_payload_sha256"] == _payload_sha(a_request_payload),
    )
    if not all(request_bindings):
        _fail(FAIL_CONFLICT_V1, "same A slot already committed a different request")
    _read_exact_record(
        directory_fd, intent["q1_request_file_basename"], q1_request_payload
    )
    _read_exact_record(
        directory_fd, intent["q1_response_file_basename"], q1_response_payload
    )
    _read_exact_record(
        directory_fd, intent["a_request_file_basename"], a_request_payload
    )
    checkpoint_payload = _read_bounded_record(
        directory_fd, intent["a_checkpoint_file_basename"]
    )
    receipt_payload = _read_bounded_record(
        directory_fd, intent["a_receipt_file_basename"]
    )
    # The head is read last because its no-replace directory entry is the sole
    # local commit marker for this mechanics-only slot.
    head_payload = _read_bounded_record(
        directory_fd, intent["a_head_file_basename"]
    )
    checkpoint, head, receipt = _verify_stored_a_outputs(
        checkpoint_payload=checkpoint_payload,
        head_payload=head_payload,
        receipt_payload=receipt_payload,
        a_request_payload=a_request_payload,
        a_request=a_request,
        a_request_body=a_request_body,
        durable_policy=durable_policy,
    )
    output_bindings = (
        intent["a_checkpoint_id"] == checkpoint["a_checkpoint_id"],
        intent["a_checkpoint_payload_sha256"] == _payload_sha(checkpoint_payload),
        intent["a_head_id"] == head["a_head_id"],
        intent["a_head_payload_sha256"] == _payload_sha(head_payload),
        intent["a_receipt_id"] == receipt["a_receipt_id"],
        intent["a_receipt_payload_sha256"] == _payload_sha(receipt_payload),
    )
    if not all(output_bindings):
        _fail(FAIL_POISONED_V1, "A slot-intent output ID/SHA binding differs")
    final_entries = _directory_entries(directory_fd)
    if set(final_entries) != expected_entries or len(final_entries) != len(
        expected_entries
    ):
        _fail(FAIL_POISONED_V1, "A spool namespace changed during replay")
    return (
        checkpoint_payload,
        head_payload,
        receipt_payload,
        (intent_basename, *publication_order),
    )


def _publish_new_slot(
    directory_fd: int,
    *,
    bootstrap_policy_payload: bytes,
    bootstrap_policy: Mapping[str, object],
    online_policy_payload: bytes,
    online_policy: Mapping[str, object],
    durable_policy_payload: bytes,
    durable_policy: Mapping[str, object],
    q1_request_payload: bytes,
    q1_request: Mapping[str, object],
    q1_response_payload: bytes,
    q1_response: Mapping[str, object],
    a_request_payload: bytes,
    a_request: Mapping[str, object],
    a_request_body: Mapping[str, object],
    private_key: object,
) -> tuple[bytes, bytes, bytes, tuple[str, ...]]:
    if _directory_entries(directory_fd):
        _fail(FAIL_CONFLICT_V1, "new A publication requires an empty slot directory")
    (
        checkpoint_payload,
        checkpoint,
        head_payload,
        head,
        receipt_payload,
        receipt,
    ) = _compose_a_outputs(
        a_request_payload=a_request_payload,
        a_request=a_request,
        a_request_body=a_request_body,
        durable_policy=durable_policy,
        private_key=private_key,
    )
    intent_payload, _intent, intent_basename, publication_order = (
        _compose_slot_intent(
            bootstrap_policy_payload=bootstrap_policy_payload,
            bootstrap_policy=bootstrap_policy,
            online_policy_payload=online_policy_payload,
            online_policy=online_policy,
            durable_policy_payload=durable_policy_payload,
            durable_policy=durable_policy,
            q1_request_payload=q1_request_payload,
            q1_request=q1_request,
            q1_response_payload=q1_response_payload,
            q1_response=q1_response,
            a_request_payload=a_request_payload,
            a_request=a_request,
            checkpoint_payload=checkpoint_payload,
            checkpoint=checkpoint,
            head_payload=head_payload,
            head=head,
            receipt_payload=receipt_payload,
            receipt=receipt,
        )
    )
    _write_exclusive_record(directory_fd, intent_basename, intent_payload)
    publications = (
        (publication_order[0], q1_request_payload),
        (publication_order[1], q1_response_payload),
        (publication_order[2], a_request_payload),
        (publication_order[3], checkpoint_payload),
        (publication_order[4], receipt_payload),
        # Sole commit marker: no final artifact may be published after this.
        (publication_order[5], head_payload),
    )
    for basename, payload in publications:
        _publish_staged_no_replace(directory_fd, basename, payload)
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, "final A slot directory fsync failed closed"
        ) from exc
    loaded = _load_existing_complete_slot(
        directory_fd,
        bootstrap_policy_payload=bootstrap_policy_payload,
        bootstrap_policy=bootstrap_policy,
        online_policy_payload=online_policy_payload,
        online_policy=online_policy,
        durable_policy_payload=durable_policy_payload,
        durable_policy=durable_policy,
        q1_request_payload=q1_request_payload,
        q1_request=q1_request,
        q1_response_payload=q1_response_payload,
        q1_response=q1_response,
        a_request_payload=a_request_payload,
        a_request=a_request,
        a_request_body=a_request_body,
    )
    if loaded is None:
        _fail(FAIL_POISONED_V1, "new A slot disappeared after publication")
    if loaded[:3] != (checkpoint_payload, head_payload, receipt_payload):
        _fail(FAIL_POISONED_V1, "new A slot final byte readback differs")
    return loaded


@dataclass(frozen=True, slots=True)
class _ActorStateV1:
    bootstrap_policy_payload: bytes
    online_policy_payload: bytes
    durable_policy_payload: bytes
    bootstrap_policy_id_bytes: bytes
    bootstrap_policy_digest: bytes
    online_policy_id_bytes: bytes
    online_policy_digest: bytes
    durable_policy_id_bytes: bytes
    durable_policy_digest: bytes
    private_key: object
    spool_path: str
    spool_device: int
    spool_inode: int


@dataclass(frozen=True, slots=True, init=False)
class LocalDurableAnchorActorV1:
    """Opaque, privately constructed local A actor; it exposes no key."""

    bootstrap_external_witness_policy_id: str
    online_witness_policy_id: str
    durable_anchor_policy_id: str
    attempt_context_id: str
    slot_key: str
    mechanics_claim_boundary: str
    raw_three_policy_closure_verified: bool
    exact_genesis_oob_high_water_verified: bool
    matching_a_key_mechanics_verified: bool
    owner_private_spool_binding_mechanics_verified: bool
    key_custody_verified: bool
    external_durability_verified: bool
    host_rollback_resistance_verified: bool
    non_equivocation_verified: bool
    live_service_verified: bool
    external_currentness_verified: bool
    logical_time_ancestry_verified: bool
    authority_established: bool
    r01_satisfied: bool
    readiness_satisfied: bool
    execution_authorized: bool

    def __new__(cls, *args: object, **kwargs: object) -> "LocalDurableAnchorActorV1":
        raise TypeError("local durable-anchor actors are privately constructed")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("local durable-anchor actors are privately constructed")


@dataclass(frozen=True, slots=True, init=False)
class IssuedLocalDurableAnchorAppendCandidateV1:
    """Opaque checkpoint/head/receipt result; never admission evidence."""

    a_durable_checkpoint_payload: bytes
    a_durable_head_payload: bytes
    a_compare_and_append_receipt_payload: bytes
    payloads_in_contract_order: tuple[bytes, bytes, bytes]
    artifact_ids: Mapping[str, str]
    payload_sha256s: Mapping[str, str]
    spool_file_basenames_in_commit_order: tuple[str, ...]
    slot_key: str
    idempotent_replay: bool
    mechanics_claim_boundary: str
    raw_policy_and_request_signature_mechanics_verified: bool
    exact_genesis_oob_high_water_verified: bool
    local_same_slot_compare_and_append_mechanics_verified: bool
    concurrent_same_slot_serialization_mechanics_verified: bool
    a_ed25519_signature_mechanics_verified: bool
    local_fsync_reopen_readback_mechanics_verified: bool
    head_last_commit_marker_mechanics_verified: bool
    key_custody_verified: bool
    external_durability_verified: bool
    host_rollback_resistance_verified: bool
    non_equivocation_verified: bool
    live_service_verified: bool
    external_currentness_verified: bool
    global_anti_fork_verified: bool
    logical_time_ancestry_verified: bool
    authority_established: bool
    r01_satisfied: bool
    readiness_satisfied: bool
    execution_authorized: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "IssuedLocalDurableAnchorAppendCandidateV1":
        raise TypeError("local durable-anchor append results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("local durable-anchor append results are privately issued")


def _build_actor_lineage():
    actor_type = LocalDurableAnchorActorV1
    error_type = Phase2BStep5R01DurableAnchorErrorV1
    failure_input = FAIL_INPUT_V1
    boundary = LOCAL_MECHANICS_CLAIM_BOUNDARY_V1
    issued: dict[
        int,
        tuple[
            object,
            _ActorStateV1,
            str,
            str,
            tuple[str, str, str],
        ],
    ] = {}

    def reject(detail: str) -> NoReturn:
        raise error_type(failure_input, detail)

    def issue(
        state: _ActorStateV1,
        bootstrap_policy: Mapping[str, object],
        online_policy: Mapping[str, object],
        durable_policy: Mapping[str, object],
    ) -> LocalDurableAnchorActorV1:
        value = object.__new__(actor_type)
        attempt_context_id = durable_policy["body"]["oob_high_water_mapping"][
            "attempt_context_id"
        ]
        object.__setattr__(
            value,
            "bootstrap_external_witness_policy_id",
            bootstrap_policy["external_witness_policy_id"],
        )
        object.__setattr__(
            value, "online_witness_policy_id", online_policy["online_witness_policy_id"]
        )
        object.__setattr__(
            value,
            "durable_anchor_policy_id",
            durable_policy["durable_anchor_policy_id"],
        )
        slot_key = _slot_key(attempt_context_id, 1)
        object.__setattr__(value, "attempt_context_id", attempt_context_id)
        object.__setattr__(value, "slot_key", slot_key)
        object.__setattr__(value, "mechanics_claim_boundary", boundary)
        for field in (
            "raw_three_policy_closure_verified",
            "exact_genesis_oob_high_water_verified",
            "matching_a_key_mechanics_verified",
            "owner_private_spool_binding_mechanics_verified",
        ):
            object.__setattr__(value, field, True)
        for field in (
            "key_custody_verified",
            "external_durability_verified",
            "host_rollback_resistance_verified",
            "non_equivocation_verified",
            "live_service_verified",
            "external_currentness_verified",
            "logical_time_ancestry_verified",
            "authority_established",
            "r01_satisfied",
            "readiness_satisfied",
            "execution_authorized",
        ):
            object.__setattr__(value, field, False)
        policy_ids = (
            bootstrap_policy["external_witness_policy_id"],
            online_policy["online_witness_policy_id"],
            durable_policy["durable_anchor_policy_id"],
        )
        issued[id(value)] = (
            value,
            state,
            attempt_context_id,
            slot_key,
            policy_ids,
        )
        return value

    def require(value: object) -> LocalDurableAnchorActorV1:
        if type(value) is not actor_type:
            reject("constructed local durable-anchor actor required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            reject("local durable-anchor actor lacks lineage")
        state = record[1]
        expected_attempt = record[2]
        expected_slot = record[3]
        expected_ids = record[4]
        if (
            type(value.bootstrap_external_witness_policy_id) is not str
            or type(value.online_witness_policy_id) is not str
            or type(value.durable_anchor_policy_id) is not str
            or type(value.attempt_context_id) is not str
            or type(value.slot_key) is not str
            or type(value.mechanics_claim_boundary) is not str
            or value.bootstrap_external_witness_policy_id != expected_ids[0]
            or value.online_witness_policy_id != expected_ids[1]
            or value.durable_anchor_policy_id != expected_ids[2]
            or value.attempt_context_id != expected_attempt
            or value.slot_key != expected_slot
            or value.mechanics_claim_boundary != boundary
            or value.raw_three_policy_closure_verified is not True
            or value.exact_genesis_oob_high_water_verified is not True
            or value.matching_a_key_mechanics_verified is not True
            or value.owner_private_spool_binding_mechanics_verified is not True
            or value.key_custody_verified is not False
            or value.external_durability_verified is not False
            or value.host_rollback_resistance_verified is not False
            or value.non_equivocation_verified is not False
            or value.live_service_verified is not False
            or value.external_currentness_verified is not False
            or value.logical_time_ancestry_verified is not False
            or value.authority_established is not False
            or value.r01_satisfied is not False
            or value.readiness_satisfied is not False
            or value.execution_authorized is not False
        ):
            reject("local durable-anchor actor lineage was mutated")
        return value

    def state(value: object) -> _ActorStateV1:
        actor = require(value)
        return issued[id(actor)][1]

    return issue, require, state


_issue_actor, _require_actor, _actor_state = _build_actor_lineage()
del _build_actor_lineage


def _build_result_lineage():
    result_type = IssuedLocalDurableAnchorAppendCandidateV1
    error_type = Phase2BStep5R01DurableAnchorErrorV1
    failure_input = FAIL_INPUT_V1
    boundary = LOCAL_MECHANICS_CLAIM_BOUNDARY_V1
    proxy = MappingProxyType
    issued: dict[int, tuple[object, ...]] = {}

    def reject(detail: str) -> NoReturn:
        raise error_type(failure_input, detail)

    def issue(
        *,
        checkpoint_payload: bytes,
        head_payload: bytes,
        receipt_payload: bytes,
        artifact_ids: Mapping[str, str],
        payload_sha256s: Mapping[str, str],
        spool_names: tuple[str, ...],
        slot_key: str,
        idempotent_replay: bool,
    ) -> IssuedLocalDurableAnchorAppendCandidateV1:
        if type(idempotent_replay) is not bool:
            reject("idempotent replay flag must be exact bool")
        value = object.__new__(result_type)
        checkpoint_copy = bytes(checkpoint_payload)
        head_copy = bytes(head_payload)
        receipt_copy = bytes(receipt_payload)
        contract_order = (checkpoint_copy, head_copy, receipt_copy)
        frozen_ids = proxy(dict(artifact_ids))
        frozen_shas = proxy(dict(payload_sha256s))
        names_copy = tuple(spool_names)
        object.__setattr__(value, "a_durable_checkpoint_payload", checkpoint_copy)
        object.__setattr__(value, "a_durable_head_payload", head_copy)
        object.__setattr__(
            value, "a_compare_and_append_receipt_payload", receipt_copy
        )
        object.__setattr__(value, "payloads_in_contract_order", contract_order)
        object.__setattr__(value, "artifact_ids", frozen_ids)
        object.__setattr__(value, "payload_sha256s", frozen_shas)
        object.__setattr__(
            value, "spool_file_basenames_in_commit_order", names_copy
        )
        object.__setattr__(value, "slot_key", slot_key)
        object.__setattr__(value, "idempotent_replay", idempotent_replay)
        object.__setattr__(value, "mechanics_claim_boundary", boundary)
        for field in (
            "raw_policy_and_request_signature_mechanics_verified",
            "exact_genesis_oob_high_water_verified",
            "local_same_slot_compare_and_append_mechanics_verified",
            "concurrent_same_slot_serialization_mechanics_verified",
            "a_ed25519_signature_mechanics_verified",
            "local_fsync_reopen_readback_mechanics_verified",
            "head_last_commit_marker_mechanics_verified",
        ):
            object.__setattr__(value, field, True)
        for field in (
            "key_custody_verified",
            "external_durability_verified",
            "host_rollback_resistance_verified",
            "non_equivocation_verified",
            "live_service_verified",
            "external_currentness_verified",
            "global_anti_fork_verified",
            "logical_time_ancestry_verified",
            "authority_established",
            "r01_satisfied",
            "readiness_satisfied",
            "execution_authorized",
        ):
            object.__setattr__(value, field, False)
        issued[id(value)] = (
            value,
            checkpoint_copy,
            head_copy,
            receipt_copy,
            contract_order,
            frozen_ids,
            frozen_shas,
            names_copy,
            slot_key,
            idempotent_replay,
        )
        return value

    def require(value: object) -> IssuedLocalDurableAnchorAppendCandidateV1:
        if type(value) is not result_type:
            reject("issued local durable-anchor result required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            reject("local durable-anchor result lacks lineage")
        if (
            value.a_durable_checkpoint_payload is not record[1]
            or value.a_durable_head_payload is not record[2]
            or value.a_compare_and_append_receipt_payload is not record[3]
            or value.payloads_in_contract_order is not record[4]
            or value.artifact_ids is not record[5]
            or value.payload_sha256s is not record[6]
            or value.spool_file_basenames_in_commit_order is not record[7]
            or type(value.slot_key) is not str
            or type(value.mechanics_claim_boundary) is not str
            or type(value.idempotent_replay) is not bool
            or value.slot_key != record[8]
            or value.idempotent_replay is not record[9]
            or value.mechanics_claim_boundary != boundary
            or value.raw_policy_and_request_signature_mechanics_verified is not True
            or value.exact_genesis_oob_high_water_verified is not True
            or value.local_same_slot_compare_and_append_mechanics_verified is not True
            or value.concurrent_same_slot_serialization_mechanics_verified is not True
            or value.a_ed25519_signature_mechanics_verified is not True
            or value.local_fsync_reopen_readback_mechanics_verified is not True
            or value.head_last_commit_marker_mechanics_verified is not True
            or value.key_custody_verified is not False
            or value.external_durability_verified is not False
            or value.host_rollback_resistance_verified is not False
            or value.non_equivocation_verified is not False
            or value.live_service_verified is not False
            or value.external_currentness_verified is not False
            or value.global_anti_fork_verified is not False
            or value.logical_time_ancestry_verified is not False
            or value.authority_established is not False
            or value.r01_satisfied is not False
            or value.readiness_satisfied is not False
            or value.execution_authorized is not False
        ):
            reject("local durable-anchor result lineage was mutated")
        return value

    return issue, require


_issue_result, _require_result = _build_result_lineage()
del _build_result_lineage


def _construct_local_durable_anchor_actor_implementation_v1(
    bootstrap_external_witness_policy_payload: bytes,
    online_witness_policy_payload: bytes,
    durable_anchor_policy_payload: bytes,
    *,
    out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
    out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_online_witness_policy_id_bytes: bytes,
    out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
    out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
    out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
    durable_anchor_private_key: object,
    owner_private_spool_directory_path: str,
) -> LocalDurableAnchorActorV1:
    """Construct one local A actor from exact raw policy pins and one A key.

    The private key is accepted only here.  Construction validates a local
    service binding; it is not evidence of non-exportability or key custody.
    """

    bootstrap, online, durable = _load_policy_closure(
        bootstrap_external_witness_policy_payload,
        online_witness_policy_payload,
        durable_anchor_policy_payload,
        out_of_band_expected_bootstrap_external_witness_policy_id_bytes=(
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes
        ),
        out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=(
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest
        ),
        out_of_band_expected_online_witness_policy_id_bytes=(
            out_of_band_expected_online_witness_policy_id_bytes
        ),
        out_of_band_expected_online_witness_policy_payload_sha256_digest=(
            out_of_band_expected_online_witness_policy_payload_sha256_digest
        ),
        out_of_band_expected_durable_anchor_policy_id_bytes=(
            out_of_band_expected_durable_anchor_policy_id_bytes
        ),
        out_of_band_expected_durable_anchor_policy_payload_sha256_digest=(
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest
        ),
    )
    private_key = _require_matching_a_private_key(
        durable_anchor_private_key, durable
    )
    directory_fd = _open_owner_private_directory(
        owner_private_spool_directory_path
    )
    try:
        construction_identity = _lock_directory_exclusive(directory_fd)
        os.fsync(directory_fd)
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, "A spool construction barrier failed closed"
        ) from exc
    finally:
        os.close(directory_fd)
    state = _ActorStateV1(
        bootstrap_policy_payload=bytes(bootstrap_external_witness_policy_payload),
        online_policy_payload=bytes(online_witness_policy_payload),
        durable_policy_payload=bytes(durable_anchor_policy_payload),
        bootstrap_policy_id_bytes=bytes(
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes
        ),
        bootstrap_policy_digest=bytes(
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest
        ),
        online_policy_id_bytes=bytes(
            out_of_band_expected_online_witness_policy_id_bytes
        ),
        online_policy_digest=bytes(
            out_of_band_expected_online_witness_policy_payload_sha256_digest
        ),
        durable_policy_id_bytes=bytes(
            out_of_band_expected_durable_anchor_policy_id_bytes
        ),
        durable_policy_digest=bytes(
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest
        ),
        private_key=private_key,
        spool_path=str(owner_private_spool_directory_path),
        spool_device=construction_identity[0],
        spool_inode=construction_identity[1],
    )
    return _issue_actor(state, bootstrap, online, durable)


def _compare_and_append_local_durable_anchor_implementation_v1(
    actor: object,
    q1_prepare_request_payload: bytes,
    q1_prepare_response_payload: bytes,
    a_compare_and_append_request_payload: bytes,
) -> IssuedLocalDurableAnchorAppendCandidateV1:
    """Serialize, compare, and append one initial A slot from raw bytes only."""

    state = _actor_state(actor)
    bootstrap, online, durable = _load_policy_closure(
        state.bootstrap_policy_payload,
        state.online_policy_payload,
        state.durable_policy_payload,
        out_of_band_expected_bootstrap_external_witness_policy_id_bytes=(
            state.bootstrap_policy_id_bytes
        ),
        out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest=(
            state.bootstrap_policy_digest
        ),
        out_of_band_expected_online_witness_policy_id_bytes=(
            state.online_policy_id_bytes
        ),
        out_of_band_expected_online_witness_policy_payload_sha256_digest=(
            state.online_policy_digest
        ),
        out_of_band_expected_durable_anchor_policy_id_bytes=(
            state.durable_policy_id_bytes
        ),
        out_of_band_expected_durable_anchor_policy_payload_sha256_digest=(
            state.durable_policy_digest
        ),
    )
    private_key = _require_matching_a_private_key(state.private_key, durable)
    (
        q1_request,
        _q1_request_body,
        q1_response,
        _q1_response_body,
        a_request,
        a_request_body,
    ) = _load_request_closure(
        q1_prepare_request_payload,
        q1_prepare_response_payload,
        a_compare_and_append_request_payload,
        bootstrap_policy=bootstrap,
        online_policy=online,
        durable_policy=durable,
    )
    directory_fd = _open_owner_private_directory(state.spool_path)
    completed: tuple[bytes, bytes, bytes, tuple[str, ...]]
    idempotent_replay: bool
    try:
        locked_identity = _lock_directory_exclusive(directory_fd)
        if locked_identity != (state.spool_device, state.spool_inode):
            _fail(
                FAIL_FILESYSTEM_V1,
                "A spool identity differs from actor construction",
            )
        existing = _load_existing_complete_slot(
            directory_fd,
            bootstrap_policy_payload=state.bootstrap_policy_payload,
            bootstrap_policy=bootstrap,
            online_policy_payload=state.online_policy_payload,
            online_policy=online,
            durable_policy_payload=state.durable_policy_payload,
            durable_policy=durable,
            q1_request_payload=q1_prepare_request_payload,
            q1_request=q1_request,
            q1_response_payload=q1_prepare_response_payload,
            q1_response=q1_response,
            a_request_payload=a_compare_and_append_request_payload,
            a_request=a_request,
            a_request_body=a_request_body,
        )
        if existing is None:
            completed = _publish_new_slot(
                directory_fd,
                bootstrap_policy_payload=state.bootstrap_policy_payload,
                bootstrap_policy=bootstrap,
                online_policy_payload=state.online_policy_payload,
                online_policy=online,
                durable_policy_payload=state.durable_policy_payload,
                durable_policy=durable,
                q1_request_payload=q1_prepare_request_payload,
                q1_request=q1_request,
                q1_response_payload=q1_prepare_response_payload,
                q1_response=q1_response,
                a_request_payload=a_compare_and_append_request_payload,
                a_request=a_request,
                a_request_body=a_request_body,
                private_key=private_key,
            )
            idempotent_replay = False
        else:
            completed = existing
            idempotent_replay = True
        after = os.fstat(directory_fd)
        _validate_private_directory_stat(after)
        if (after.st_dev, after.st_ino) != locked_identity:
            _fail(FAIL_FILESYSTEM_V1, "locked A spool identity changed")
        if (after.st_dev, after.st_ino) != (
            state.spool_device,
            state.spool_inode,
        ):
            _fail(
                FAIL_FILESYSTEM_V1,
                "A spool identity differs from actor construction after append",
            )
        path_after = os.stat(state.spool_path, follow_symlinks=False)
        _validate_private_directory_stat(path_after)
        if (path_after.st_dev, path_after.st_ino) != locked_identity:
            _fail(FAIL_FILESYSTEM_V1, "A spool path no longer names locked directory")
    except Phase2BStep5R01DurableAnchorErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01DurableAnchorErrorV1(
            FAIL_FILESYSTEM_V1, "local A compare-and-append failed closed"
        ) from exc
    finally:
        os.close(directory_fd)
    checkpoint_payload, head_payload, receipt_payload, spool_names = completed
    checkpoint = _object(
        _decode(checkpoint_payload, "returned A checkpoint"),
        frozenset(_A_CHECKPOINT_FIELDS),
        "returned A checkpoint",
    )
    head = _object(
        _decode(head_payload, "returned A head"),
        frozenset(_A_HEAD_FIELDS),
        "returned A head",
    )
    receipt = _object(
        _decode(receipt_payload, "returned A receipt"),
        frozenset(_A_RECEIPT_FIELDS),
        "returned A receipt",
    )
    artifact_ids = {
        "bootstrap_external_witness_policy_id": bootstrap[
            "external_witness_policy_id"
        ],
        "online_witness_policy_id": online["online_witness_policy_id"],
        "durable_anchor_policy_id": durable["durable_anchor_policy_id"],
        "q1_request_id": q1_request["q1_request_id"],
        "q1_response_id": q1_response["q1_response_id"],
        "a_request_id": a_request["a_request_id"],
        "a_checkpoint_id": checkpoint["a_checkpoint_id"],
        "a_head_id": head["a_head_id"],
        "a_receipt_id": receipt["a_receipt_id"],
    }
    payload_sha256s = {
        "bootstrap_external_witness_policy_payload_sha256": _payload_sha(
            state.bootstrap_policy_payload
        ),
        "online_witness_policy_payload_sha256": _payload_sha(
            state.online_policy_payload
        ),
        "durable_anchor_policy_payload_sha256": _payload_sha(
            state.durable_policy_payload
        ),
        "q1_request_payload_sha256": _payload_sha(q1_prepare_request_payload),
        "q1_response_payload_sha256": _payload_sha(q1_prepare_response_payload),
        "a_request_payload_sha256": _payload_sha(
            a_compare_and_append_request_payload
        ),
        "a_checkpoint_payload_sha256": _payload_sha(checkpoint_payload),
        "a_head_payload_sha256": _payload_sha(head_payload),
        "a_receipt_payload_sha256": _payload_sha(receipt_payload),
    }
    return _issue_result(
        checkpoint_payload=checkpoint_payload,
        head_payload=head_payload,
        receipt_payload=receipt_payload,
        artifact_ids=artifact_ids,
        payload_sha256s=payload_sha256s,
        spool_names=spool_names,
        slot_key=_slot_key(a_request_body["attempt_context_id"], 1),
        idempotent_replay=idempotent_replay,
    )


def require_constructed_local_durable_anchor_actor_v1(
    value: object,
) -> LocalDurableAnchorActorV1:
    return _require_actor(value)


def require_issued_local_durable_anchor_append_candidate_v1(
    value: object,
) -> IssuedLocalDurableAnchorAppendCandidateV1:
    return _require_result(value)


def verify_real_external_witness_admission_v1(
    *args: object, **kwargs: object
) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "real external durable/non-equivocating/live/current admission is NOT_IMPLEMENTED",
    )


def _definition_captured_function_graph_v1() -> dict[str, object]:
    """Clone local functions and function-valued defaults onto a private graph."""

    module_namespace = globals()
    private_namespace = dict(module_namespace)
    originals: dict[str, FunctionType] = {
        name: value
        for name, value in module_namespace.items()
        if type(value) is FunctionType
        and value.__module__ == __name__
        and name != "_definition_captured_function_graph_v1"
    }
    provisional: dict[str, FunctionType] = {}
    provisional_by_identity: dict[int, FunctionType] = {}
    for name, implementation in originals.items():
        clone = FunctionType(
            implementation.__code__,
            private_namespace,
            implementation.__name__,
            implementation.__defaults__,
            implementation.__closure__,
        )
        provisional[name] = clone
        provisional_by_identity[id(implementation)] = clone
    private_namespace.update(provisional)

    def provisional_default(value: object) -> object:
        return provisional_by_identity.get(id(value), value)

    captured: dict[str, FunctionType] = {}
    for name, implementation in originals.items():
        defaults = implementation.__defaults__
        clone = FunctionType(
            implementation.__code__,
            private_namespace,
            implementation.__name__,
            (
                None
                if defaults is None
                else tuple(provisional_default(value) for value in defaults)
            ),
            implementation.__closure__,
        )
        clone.__kwdefaults__ = (
            None
            if implementation.__kwdefaults__ is None
            else {
                key: provisional_default(value)
                for key, value in implementation.__kwdefaults__.items()
            }
        )
        clone.__annotations__ = dict(implementation.__annotations__)
        clone.__doc__ = implementation.__doc__
        clone.__module__ = implementation.__module__
        clone.__qualname__ = implementation.__qualname__
        captured[name] = clone
    final_by_identity = {
        id(originals[name]): clone for name, clone in captured.items()
    }

    def final_default(value: object) -> object:
        return final_by_identity.get(id(value), value)

    for name, implementation in originals.items():
        clone = captured[name]
        defaults = implementation.__defaults__
        clone.__defaults__ = (
            None
            if defaults is None
            else tuple(final_default(value) for value in defaults)
        )
        clone.__kwdefaults__ = (
            None
            if implementation.__kwdefaults__ is None
            else {
                key: final_default(value)
                for key, value in implementation.__kwdefaults__.items()
            }
        )
    private_namespace.update(captured)
    return captured


_CAPTURED = _definition_captured_function_graph_v1()
construct_local_durable_anchor_actor_v1 = _CAPTURED[
    "_construct_local_durable_anchor_actor_implementation_v1"
]
compare_and_append_local_durable_anchor_v1 = _CAPTURED[
    "_compare_and_append_local_durable_anchor_implementation_v1"
]
require_constructed_local_durable_anchor_actor_v1 = _CAPTURED[
    "require_constructed_local_durable_anchor_actor_v1"
]
require_issued_local_durable_anchor_append_candidate_v1 = _CAPTURED[
    "require_issued_local_durable_anchor_append_candidate_v1"
]
verify_real_external_witness_admission_v1 = _CAPTURED[
    "verify_real_external_witness_admission_v1"
]
del _CAPTURED
del _definition_captured_function_graph_v1
del _construct_local_durable_anchor_actor_implementation_v1
del _compare_and_append_local_durable_anchor_implementation_v1
del _issue_actor
del _require_actor
del _actor_state
del _issue_result
del _require_result


__all__ = [
    "ADMISSION_CONTRACT_SOURCE_SHA256_V1",
    "LOCAL_DURABLE_ANCHOR_SLOT_INTENT_SCHEMA_V1",
    "LOCAL_DURABLE_ANCHOR_SLOT_KEY_PROFILE_V1",
    "LOCAL_COMPARE_AND_APPEND_SEMANTICS_V1",
    "LOCAL_MECHANICS_CLAIM_BOUNDARY_V1",
    "FAIL_INPUT_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_BINDING_V1",
    "FAIL_TRUST_ANCHOR_V1",
    "FAIL_SIGNATURE_V1",
    "FAIL_KEY_V1",
    "FAIL_FILESYSTEM_V1",
    "FAIL_CONFLICT_V1",
    "FAIL_POISONED_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "Phase2BStep5R01DurableAnchorErrorV1",
    "LocalDurableAnchorActorV1",
    "IssuedLocalDurableAnchorAppendCandidateV1",
    "construct_local_durable_anchor_actor_v1",
    "compare_and_append_local_durable_anchor_v1",
    "require_constructed_local_durable_anchor_actor_v1",
    "require_issued_local_durable_anchor_append_candidate_v1",
    "verify_real_external_witness_admission_v1",
]

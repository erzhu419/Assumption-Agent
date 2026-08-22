"""Local in-process online-witness mechanics for the prospective R01 admission.

This module implements only the online ``W`` actor portion of the frozen
external-witness admission transcript.  It consumes exact raw authority,
policy, initial external-candidate, Q1, A, and Q2 bytes.  The actor derives the
candidate current-map entry by replaying the five raw candidate artifacts,
creates its confirmation nonce internally, and signs the exact admission
contract Q1 response, A request, and Q2 response preimages.

One process-global fence is held for each witness namespace.  Exact retries
return the originally signed bytes; a different request for an occupied slot
fails closed.  This fence is deliberately in-process only: it is neither a
durable intent log nor restart recovery.  A process restart loses the fence and
therefore cannot continue a prior session.  A concrete Linux ``/proc`` reader
binds both phases to the same local PID/start-time/UID/GID/user-namespace
observation.  There is no caller callback, decoded mapping, UID override, or
caller nonce input.

The initial anchor contains only IDs and digests for its current cut.  This
module does not receive or replay the complete logical-time ancestry.  It also
does not verify a runtime-closure report, custody attestation, durable A
implementation, external currentness, non-equivocation, or a live transport.
All such claims, plus authority, R01, readiness, and execution, remain false.
The full bytes-only admission verifier must close the transcript and logical
ancestry later; real admission is intentionally NOT_IMPLEMENTED here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re
import secrets
import threading
from types import FunctionType, MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    DETACHED_SIGNATURE_FIELDS_V2 as _AUTH_SIGNATURE_FIELDS,
    DETACHED_SIGNATURE_SCHEMA_V2 as _AUTH_SIGNATURE_SCHEMA,
    ROLE_ORDER_V2 as _AUTH_ROLE_ORDER,
    Phase2BStep5AuthorityErrorV2 as _AuthorityError,
    load_step5_authority_profile_v2 as _load_authority_profile,
    load_step5_signer_registry_v2 as _load_signer_registry,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_lock,
    phase2b_step5_signature_preimage_v2 as _authority_signature_preimage,
    require_locked_step5_authority_graph_v2 as _require_authority_graph,
)
from .phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 import (
    CANDIDATE_CLAIM_LEVEL_V1 as _EVIDENCE_CLAIM_LEVEL,
    CANDIDATE_STATUS_V1 as _EVIDENCE_STATUS,
    EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1 as _EXTERNAL_RECEIPT_FIELDS,
    EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1 as _EXTERNAL_RECEIPT_SCHEMA,
    EXTERNAL_WITNESS_POLICY_FIELDS_V1 as _BOOTSTRAP_POLICY_FIELDS,
    EXTERNAL_WITNESS_POLICY_SCHEMA_V1 as _BOOTSTRAP_POLICY_SCHEMA,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1 as _ANCHOR_BODY_FIELDS,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_FIELDS_V1 as _ANCHOR_FIELDS,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1 as _ANCHOR_SCHEMA,
    phase2b_step5_external_anchor_receipt_signature_preimage_v1 as _external_receipt_preimage,
    phase2b_step5_r01_evidence_content_id_v1 as _evidence_content_id,
)
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
    BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1 as _BOOTSTRAP_PURPOSES,
    CANDIDATE_CLAIM_LEVEL_V1 as _ADMISSION_CLAIM_LEVEL,
    CANDIDATE_STATUS_V1 as _ADMISSION_STATUS,
    CAS_OUTCOME_V1 as _CAS_OUTCOME,
    CONFIRM_OUTCOME_V1 as _CONFIRM_OUTCOME,
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
    Q2_CONFIRM_REQUEST_BODY_FIELDS_V1 as _Q2_REQUEST_BODY_FIELDS,
    Q2_CONFIRM_REQUEST_FIELDS_V1 as _Q2_REQUEST_FIELDS,
    Q2_CONFIRM_REQUEST_SCHEMA_V1 as _Q2_REQUEST_SCHEMA,
    Q2_CONFIRM_RESPONSE_BODY_FIELDS_V1 as _Q2_RESPONSE_BODY_FIELDS,
    Q2_CONFIRM_RESPONSE_FIELDS_V1 as _Q2_RESPONSE_FIELDS,
    Q2_CONFIRM_RESPONSE_SCHEMA_V1 as _Q2_RESPONSE_SCHEMA,
    RUNTIME_CUSTODY_BINDING_FIELDS_V1 as _RUNTIME_BINDING_FIELDS,
    SIGNATURE_FIELDS_V1 as _SIGNATURE_FIELDS,
    SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1 as _SIGN_A_RECEIPT,
    SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1 as _SIGN_A_REQUEST,
    SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1 as _SIGN_A_CHECKPOINT,
    SIGN_A_DURABLE_HEAD_PURPOSE_V1 as _SIGN_A_HEAD,
    SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1 as _SIGN_DURABLE_POLICY,
    SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1 as _SIGN_ONLINE_POLICY,
    SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1 as _SIGN_Q1_REQUEST,
    SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1 as _SIGN_Q1_RESPONSE,
    SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1 as _SIGN_Q2_REQUEST,
    SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1 as _SIGN_Q2_RESPONSE,
    canonical_external_witness_admission_json_bytes_v1 as _admission_canonical,
    phase2b_step5_external_witness_admission_content_id_v1 as _admission_content_id,
    phase2b_step5_external_witness_admission_signature_preimage_v1 as _admission_preimage,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical,
)

try:
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey as _Ed25519PrivateKey,
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - deployment-image dependency.
    _InvalidSignature = None
    _serialization = None
    _Ed25519PrivateKey = None
    _Ed25519PublicKey = None


ADMISSION_CONTRACT_SOURCE_SHA256_V1: Final = (
    "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285"
)
EXTERNAL_WITNESS_SOURCE_SHA256_V1: Final = (
    "b0622d5e50e2393e116211a64622e6d3d36d76d476afdcd6e5fd98a777061a88"
)
EVIDENCE_CONTRACT_SOURCE_SHA256_V1: Final = (
    "574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d"
)

ONLINE_WITNESS_MECHANICS_RESULT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-r01-online-witness-mechanics-result/1"
)
ONLINE_WITNESS_MECHANICS_CLAIM_BOUNDARY_V1: Final = (
    "IN_PROCESS_NAMESPACE_FENCE_AND_LOCAL_PROCFS_OBSERVATION_AND_EXACT_"
    "Q1_A_Q2_SIGNATURE_COMPOSITION_MECHANICS_ONLY_NOT_DURABLE_SESSION_"
    "NOT_RESTART_RECOVERY_NOT_LIVE_TRANSPORT_NOT_RUNTIME_CLOSURE_NOT_KEY_"
    "CUSTODY_NOT_LOGICAL_TIME_ANCESTRY_NOT_EXTERNAL_CURRENTNESS_NOT_"
    "NON_EQUIVOCATION_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_CONTENT_ID"
FAIL_TRUST_ANCHOR_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_TRUST_ANCHOR"
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_SIGNATURE"
FAIL_KEY_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_KEY"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_BINDING"
FAIL_RUNTIME_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_RUNTIME"
FAIL_FENCE_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_FENCE"
FAIL_POISONED_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_POISONED"
FAIL_RESULT_V1: Final = "FAIL_STEP5_R01_ONLINE_WITNESS_V1_RESULT"
FAIL_NOT_IMPLEMENTED_V1: Final = (
    "FAIL_STEP5_R01_ONLINE_WITNESS_V1_NOT_IMPLEMENTED"
)

_MAX_CANONICAL_BYTES: Final = 16 * 1024 * 1024
_MAX_SAFE_INTEGER: Final = (1 << 53) - 1
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX128 = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_CONTENT_ID = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")
_BOOTSTRAP_PREFIX: Final = "phase2b_step5_external_witness_policy_v1_"
_BOOTSTRAP_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
_LEDGER_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-ledger/1"
)
_CHECKPOINT_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-checkpoint/1"
)
_HEAD_SCHEMA: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-head/1"
)
_EXTERNAL_MECHANICS_BOUNDARY: Final = (
    "INITIAL_ONE_LEAF_EXTERNAL_WITNESS_SIGNATURE_AND_EXCLUSIVE_LOCAL_SPOOL_"
    "MECHANICS_ONLY_NOT_DISTINCT_UID_NOT_INDEPENDENT_OR_LOCAL_FILESYSTEM_"
    "NOT_HOST_ROLLBACK_RESISTANCE_NOT_KEY_CUSTODY_NOT_EXTERNAL_CONTROL_"
    "NOT_EXTERNAL_CURRENTNESS_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)
_LEDGER_FIELDS: Final = frozenset({
    "schema_version", "status", "claim_level", "mechanics_claim_boundary",
    "stage_b_trust_lock_id", "authority_profile_id", "signer_registry_id",
    "attempt_context_id", "external_witness_policy_id", "witness_namespace",
    "witness_control_domain_id", "witness_key_id", "witness_key_epoch",
    "initial_anchor_id", "initial_anchor_payload_sha256", "initial_receipt_id",
    "initial_receipt_payload_sha256", "witness_sequence", "tree_size",
    "signed_tree_root_sha256", "ledger_id",
})
_CHECKPOINT_FIELDS: Final = frozenset({
    "schema_version", "status", "claim_level", "mechanics_claim_boundary",
    "external_witness_policy_id", "ledger_id", "initial_receipt_id",
    "initial_receipt_payload_sha256", "witness_sequence", "tree_size",
    "signed_tree_root_sha256", "checkpoint_sequence", "previous_checkpoint_id",
    "checkpoint_id",
})
_HEAD_FIELDS: Final = frozenset({
    "schema_version", "status", "claim_level", "mechanics_claim_boundary",
    "external_witness_policy_id", "ledger_id", "checkpoint_id",
    "checkpoint_payload_sha256", "initial_receipt_id",
    "initial_receipt_payload_sha256", "witness_sequence", "tree_size",
    "signed_tree_root_sha256", "previous_head_id", "head_id",
})
_LEDGER_PREFIX: Final = "phase2b_step5_external_witness_ledger_v1_"
_CHECKPOINT_PREFIX: Final = "phase2b_step5_external_witness_checkpoint_v1_"
_HEAD_PREFIX: Final = "phase2b_step5_external_witness_head_v1_"
_LEDGER_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/LEDGER/V1\0"
_CHECKPOINT_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/CHECKPOINT/V1\0"
_HEAD_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/HEAD/V1\0"
_MERKLE_LEAF_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/MERKLE_LEAF/V1\0"
_CURRENT_MAP_LEAF_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_ADMISSION/CURRENT_MAP_LEAF/V1\0"
)
_A_GENESIS_CHECKPOINT_ID: Final = (
    "phase2b_step5_external_admission_a_checkpoint_v1_" + "0" * 64
)
_A_GENESIS_HEAD_ID: Final = (
    "phase2b_step5_external_admission_a_head_v1_" + "0" * 64
)
_SIGNATURE_SCHEMA: Final = (
    "hegel-machine-phase2b-step5-external-admission-signature/1"
)
_TOKEN_BYTES: Final = secrets.token_bytes
_SHA256: Final = hashlib.sha256
_ISFINITE: Final = math.isfinite
_OS_OPEN: Final = os.open
_OS_READ: Final = os.read
_OS_CLOSE: Final = os.close
_OS_GETPID: Final = os.getpid
_OS_GETUID: Final = os.getuid
_OS_GETGID: Final = os.getgid
_OS_STAT: Final = os.stat
_OS_READ_FLAGS: Final = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


class Phase2BStep5R01OnlineWitnessErrorV1(ValueError):
    """Stable fail-closed error for local W-actor mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01OnlineWitnessErrorV1(code, detail)


def _canonical(value: object, _implementation=_canonicalize) -> bytes:
    try:
        return _implementation(value)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_CANONICAL_V1, "RFC-8785 canonicalization failed"
        ) from exc


def _restore_wire_integers(value: object) -> object:
    if type(value) is float and _ISFINITE(value) and value.is_integer():
        if abs(value) > _MAX_SAFE_INTEGER:
            _fail(FAIL_SCHEMA_V1, "decoded integer exceeds I-JSON safe range")
        return int(value)
    if type(value) is list:
        return [_restore_wire_integers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_wire_integers(item) for key, item in value.items()}
    return value


def _decode(payload: bytes, label: str, maximum: int = _MAX_CANONICAL_BYTES,
            _decoder=_decode_canonical) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= maximum
        or payload.endswith(b"\n")
    ):
        _fail(FAIL_INPUT_V1, f"{label} must be bounded exact bytes without LF")
    try:
        value = _restore_wire_integers(_decoder(payload))
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_CANONICAL_V1, f"{label} is not exact canonical RFC-8785"
        ) from exc
    if type(value) is not dict:
        _fail(FAIL_SCHEMA_V1, f"{label} must be one exact object")
    return value


def _object(value: object, fields: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(FAIL_SCHEMA_V1, f"{label} exact keyset differs")
    return value


def _text(value: object, label: str, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        _fail(FAIL_SCHEMA_V1, f"{label} must be bounded nonempty text")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_SAFE_INTEGER:
        _fail(FAIL_SCHEMA_V1, f"{label} must be an exact bounded integer")
    return value


def _hex64(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be lowercase SHA-256 hex")
    return value


def _hex128(value: object, label: str) -> str:
    if type(value) is not str or _HEX128.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be lowercase Ed25519 signature hex")
    return value


def _identifier(value: object, label: str, prefix: str | None = None) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} identifier syntax differs")
    if prefix is not None and not value.startswith(prefix):
        _fail(FAIL_SCHEMA_V1, f"{label} prefix differs")
    return value


def _content_identifier(value: object, label: str, prefix: str) -> str:
    if type(value) is not str or _CONTENT_ID.fullmatch(value) is None or not value.startswith(prefix):
        _fail(FAIL_SCHEMA_V1, f"{label} content-ID syntax differs")
    return value


def _nonce(value: object, label: str) -> str:
    nonce = _hex64(value, label)
    if nonce == "0" * 64:
        _fail(FAIL_BINDING_V1, f"{label} must be nonzero")
    return nonce


def _without(mapping: Mapping[str, object], *names: str) -> dict[str, object]:
    return {key: value for key, value in mapping.items() if key not in names}


def _payload_sha(payload: bytes) -> str:
    if type(payload) is not bytes:
        _fail(FAIL_INPUT_V1, "payload SHA input must be exact bytes")
    return _SHA256(payload).hexdigest()


def _decode_oob_id(value: bytes, label: str, prefix: str) -> str:
    if type(value) is not bytes or not 1 <= len(value) <= 512:
        _fail(FAIL_INPUT_V1, f"{label} OOB ID must be bounded exact bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_INPUT_V1, f"{label} OOB ID must be exact ASCII"
        ) from exc
    return _content_identifier(decoded, label, prefix)


def _decode_expected_identifier(value: bytes, label: str) -> str:
    if type(value) is not bytes or not 1 <= len(value) <= 512:
        _fail(FAIL_INPUT_V1, f"{label} must be bounded exact bytes")
    try:
        decoded = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_INPUT_V1, f"{label} must be exact ASCII"
        ) from exc
    return _identifier(decoded, label)


def _verify_ed25519(*, public_key_hex: str, signature_hex: str,
                    preimage: bytes, label: str) -> None:
    _hex64(public_key_hex, f"{label} public key")
    _hex128(signature_hex, f"{label} signature")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        key = _Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        key.verify(bytes.fromhex(signature_hex), preimage)
    except Exception as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_SIGNATURE_V1, f"{label} Ed25519 signature rejected"
        ) from exc


def _signature(value: object, *, expected_key_id: str, expected_key_epoch: int,
               expected_purpose: str, public_key_hex: str, preimage: bytes,
               label: str) -> dict[str, object]:
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


def _load_authority_context(
    lock_payload: bytes,
    profile_payload: bytes,
    registry_payload: bytes,
    *,
    expected_lock_id_bytes: bytes,
    expected_lock_digest: bytes,
) -> tuple[object, object, object, tuple[Mapping[str, object], ...]]:
    if (
        any(type(item) is not bytes for item in (lock_payload, profile_payload, registry_payload,
                                                  expected_lock_id_bytes, expected_lock_digest))
        or len(expected_lock_digest) != 32
        or expected_lock_digest == b"\x00" * 32
    ):
        _fail(FAIL_INPUT_V1, "authority graph and Stage-B pins must be exact raw bytes")
    try:
        lock = _load_stage_b_lock(
            lock_payload,
            out_of_band_expected_lock_id_bytes=expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=expected_lock_digest,
        )
        profile = _load_authority_profile(profile_payload, lock)
        registry = _load_signer_registry(registry_payload, lock, profile)
        lock, profile, registry = _require_authority_graph(lock, profile, registry)
    except _AuthorityError as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_TRUST_ANCHOR_V1, "raw authority-v2 graph rejected"
        ) from exc
    try:
        rows = registry.body["role_keys_in_exact_order"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_TRUST_ANCHOR_V1, "verified authority role keys unavailable"
        ) from exc
    if type(rows) is not tuple or len(rows) != len(_AUTH_ROLE_ORDER):
        _fail(FAIL_TRUST_ANCHOR_V1, "verified authority role set differs")
    for role, row in zip(_AUTH_ROLE_ORDER, rows, strict=True):
        if type(row) is not MappingProxyType or row.get("role") != role:
            _fail(FAIL_TRUST_ANCHOR_V1, "verified authority role order differs")
    return lock, profile, registry, rows


def _load_oob_payload(payload: bytes, *, expected_id_bytes: bytes,
                      expected_digest: bytes, prefix: str,
                      label: str) -> tuple[dict[str, object], str]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAX_CANONICAL_BYTES
        or type(expected_digest) is not bytes
        or len(expected_digest) != 32
        or expected_digest == b"\x00" * 32
    ):
        _fail(FAIL_INPUT_V1, f"{label} payload and OOB pins must be exact bytes")
    if _SHA256(payload).digest() != expected_digest:
        _fail(FAIL_TRUST_ANCHOR_V1, f"{label} raw SHA-256 pin differs")
    expected_id = _decode_oob_id(expected_id_bytes, label, prefix)
    return _decode(payload, label), expected_id


def _load_bootstrap_policy(payload: bytes, *, expected_id_bytes: bytes,
                           expected_digest: bytes,
                           role_keys: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    decoded, expected_id = _load_oob_payload(
        payload, expected_id_bytes=expected_id_bytes,
        expected_digest=expected_digest, prefix=_BOOTSTRAP_PREFIX,
        label="bootstrap external-witness policy",
    )
    policy = _object(decoded, frozenset(_BOOTSTRAP_POLICY_FIELDS), "bootstrap policy")
    if (
        policy["schema_version"] != _BOOTSTRAP_POLICY_SCHEMA
        or policy["status"] != _EVIDENCE_STATUS
        or policy["claim_level"] != _EVIDENCE_CLAIM_LEVEL
        or policy["algorithm"] != _ALGORITHM
        or type(policy["witness_key_epoch"]) is not int
        or policy["witness_key_epoch"] != 1
    ):
        _fail(FAIL_SCHEMA_V1, "bootstrap policy header differs")
    _text(policy["witness_namespace"], "bootstrap namespace", 256)
    _identifier(policy["witness_control_domain_id"], "bootstrap control domain")
    _identifier(policy["witness_key_id"], "bootstrap key ID")
    public_key = _hex64(policy["public_key_hex"], "bootstrap public key")
    policy_id = _BOOTSTRAP_PREFIX + _SHA256(
        _BOOTSTRAP_DOMAIN + _canonical(_without(policy, "external_witness_policy_id"))
    ).hexdigest()
    if policy["external_witness_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "bootstrap policy content ID differs")
    if (
        public_key in {row["public_key_hex"] for row in role_keys}
        or policy["witness_key_id"] in {row["key_id"] for row in role_keys}
        or policy["witness_control_domain_id"] in {row["control_domain"] for row in role_keys}
        or policy["witness_namespace"] in set(_AUTH_ROLE_ORDER)
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "bootstrap witness is not authority-disjoint")
    return policy


def _load_online_policy(payload: bytes, *, expected_id_bytes: bytes,
                        expected_digest: bytes,
                        role_keys: tuple[Mapping[str, object], ...],
                        authority_profile: object,
                        bootstrap_payload: bytes,
                        bootstrap: Mapping[str, object]) -> dict[str, object]:
    decoded, expected_id = _load_oob_payload(
        payload, expected_id_bytes=expected_id_bytes,
        expected_digest=expected_digest,
        prefix="phase2b_step5_online_witness_policy_v1_",
        label="online-witness policy",
    )
    root = _object(decoded, frozenset(_ONLINE_POLICY_FIELDS), "online policy")
    body = _object(root["body"], frozenset(_ONLINE_POLICY_BODY_FIELDS), "online policy body")
    if (
        root["schema_version"] != _ONLINE_POLICY_SCHEMA
        or body["schema_version"] != _ONLINE_POLICY_SCHEMA
        or body["status"] != _ADMISSION_STATUS
        or body["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or type(body["policy_epoch"]) is not int or body["policy_epoch"] != 1
        or body["algorithm"] != _ALGORITHM
        or type(body["bootstrap_key_epoch"]) is not int or body["bootstrap_key_epoch"] != 1
        or type(body["online_key_epoch"]) is not int or body["online_key_epoch"] != 1
        or type(body["bootstrap_authorized_signature_purposes_in_exact_order"]) is not list
        or tuple(body["bootstrap_authorized_signature_purposes_in_exact_order"]) != _BOOTSTRAP_PURPOSES
        or type(body["online_authorized_signature_purposes_in_exact_order"]) is not list
        or tuple(body["online_authorized_signature_purposes_in_exact_order"]) != _ONLINE_PURPOSES
        or body["online_key_nonexportable_required"] is not True
        or body["runtime_identity_profile"] != "LINUX_AF_UNIX_SEQPACKET_PROCFS_ENTRY_EXIT_V1"
    ):
        _fail(FAIL_SCHEMA_V1, "online policy header or purpose order differs")
    for name in ("bootstrap_key_id", "online_key_id", "witness_control_domain_id",
                 "service_binding_id", "runtime_closure_policy_id",
                 "custody_attestation_policy_id", "custody_attestor_key_id"):
        _identifier(body[name], f"online policy {name}")
    for name in ("service_uid", "service_gid", "service_user_namespace_inode",
                 "custody_attestor_key_epoch"):
        _integer(body[name], f"online policy {name}", 1)
    for name in ("runtime_closure_policy_payload_sha256",
                 "custody_attestation_policy_payload_sha256"):
        _hex64(body[name], f"online policy {name}")
    _text(body["witness_namespace"], "online namespace", 256)
    bootstrap_public = _hex64(body["bootstrap_public_key_hex"], "bootstrap public key")
    online_public = _hex64(body["online_public_key_hex"], "online public key")
    custody_public = _hex64(body["custody_attestor_public_key_hex"], "custody public key")
    if (
        body["bootstrap_external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or body["bootstrap_external_witness_policy_payload_sha256"] != _payload_sha(bootstrap_payload)
        or body["bootstrap_key_id"] != bootstrap["witness_key_id"]
        or body["bootstrap_key_epoch"] != bootstrap["witness_key_epoch"]
        or bootstrap_public != bootstrap["public_key_hex"]
        or body["witness_namespace"] != bootstrap["witness_namespace"]
        or body["witness_control_domain_id"] == bootstrap["witness_control_domain_id"]
    ):
        _fail(FAIL_BINDING_V1, "online/bootstrap policy linkage differs")
    policy_id = _admission_content_id("online_witness_policy", body)
    if root["online_witness_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "online policy content ID differs")
    preimage = _admission_preimage(
        artifact_kind="online_witness_policy", body=body, artifact_id=policy_id,
        signer_key_id=body["bootstrap_key_id"], key_epoch=body["bootstrap_key_epoch"],
        purpose=_SIGN_ONLINE_POLICY,
    )
    _signature(root["bootstrap_signature"], expected_key_id=body["bootstrap_key_id"],
               expected_key_epoch=body["bootstrap_key_epoch"],
               expected_purpose=_SIGN_ONLINE_POLICY,
               public_key_hex=bootstrap_public, preimage=preimage,
               label="online policy bootstrap")
    authority_public = {row["public_key_hex"] for row in role_keys}
    authority_ids = {row["key_id"] for row in role_keys}
    authority_controls = {row["control_domain"] for row in role_keys}
    try:
        services = authority_profile.body["service_static_bindings_in_exact_order"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_TRUST_ANCHOR_V1, "authority service bindings unavailable"
        ) from exc
    if type(services) is not tuple or len(services) != 5:
        _fail(FAIL_TRUST_ANCHOR_V1, "authority service set differs")
    authority_uids_gids = {item for row in services for item in (row["uid"], row["gid"])}
    if (
        len({bootstrap_public, online_public, custody_public}) != 3
        or len({body["bootstrap_key_id"], body["online_key_id"], body["custody_attestor_key_id"]}) != 3
        or any(key in authority_public for key in (bootstrap_public, online_public, custody_public))
        or any(key in authority_ids for key in (body["bootstrap_key_id"], body["online_key_id"],
                                                body["custody_attestor_key_id"]))
        or body["witness_control_domain_id"] in authority_controls
        or body["service_binding_id"] in {row["binding_id"] for row in services}
        or body["service_uid"] == body["service_gid"]
        or body["service_uid"] in authority_uids_gids
        or body["service_gid"] in authority_uids_gids
        or body["runtime_closure_policy_payload_sha256"]
            in {row["static_runtime_closure_policy_sha256"] for row in services}
        or body["witness_namespace"] in set(_AUTH_ROLE_ORDER)
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "online key/control/UID-GID is not authority-disjoint")
    root["body"] = body
    return root


def _load_durable_policy(payload: bytes, *, expected_id_bytes: bytes,
                         expected_digest: bytes,
                         role_keys: tuple[Mapping[str, object], ...],
                         bootstrap_payload: bytes,
                         bootstrap: Mapping[str, object],
                         online: Mapping[str, object]) -> dict[str, object]:
    decoded, expected_id = _load_oob_payload(
        payload, expected_id_bytes=expected_id_bytes,
        expected_digest=expected_digest,
        prefix="phase2b_step5_external_durable_anchor_policy_v1_",
        label="durable-anchor policy",
    )
    root = _object(decoded, frozenset(_DURABLE_POLICY_FIELDS), "durable policy")
    body = _object(root["body"], frozenset(_DURABLE_POLICY_BODY_FIELDS), "durable policy body")
    if (
        root["schema_version"] != _DURABLE_POLICY_SCHEMA
        or body["schema_version"] != _DURABLE_POLICY_SCHEMA
        or body["status"] != _ADMISSION_STATUS
        or body["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or type(body["policy_epoch"]) is not int or body["policy_epoch"] != 1
        or body["algorithm"] != _ALGORITHM
        or type(body["anchor_key_epoch"]) is not int or body["anchor_key_epoch"] != 1
        or type(body["authorized_signature_purposes_in_exact_order"]) is not list
        or tuple(body["authorized_signature_purposes_in_exact_order"]) != _A_PURPOSES
        or body["durable_before_return_required"] is not True
        or body["admission_slot_key_profile"] != "UNIQUE_BY_ATTEMPT_CONTEXT_ID_AND_ANCHOR_SEQUENCE"
        or body["compare_and_append_semantics"] != "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
    ):
        _fail(FAIL_SCHEMA_V1, "durable policy header differs")
    _text(body["anchor_namespace"], "A namespace", 256)
    _identifier(body["anchor_control_domain_id"], "A control domain")
    _identifier(body["anchor_key_id"], "A key ID")
    durable_public = _hex64(body["anchor_public_key_hex"], "A public key")
    for name in ("bootstrap_external_witness_policy_payload_sha256", "genesis_tree_root_sha256",
                 "oob_high_water_checkpoint_payload_sha256", "oob_high_water_head_payload_sha256",
                 "oob_high_water_tree_root_sha256"):
        _hex64(body[name], f"durable policy {name}")
    for name in ("oob_high_water_checkpoint_sequence", "oob_high_water_head_sequence"):
        _integer(body[name], f"durable policy {name}")
    _content_identifier(body["oob_high_water_checkpoint_id"], "A high-water checkpoint",
                        "phase2b_step5_external_admission_a_checkpoint_v1_")
    _content_identifier(body["oob_high_water_head_id"], "A high-water head",
                        "phase2b_step5_external_admission_a_head_v1_")
    if (
        body["bootstrap_external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or body["bootstrap_external_witness_policy_payload_sha256"] != _payload_sha(bootstrap_payload)
        or body["online_witness_policy_id"] != online["online_witness_policy_id"]
        or body["genesis_checkpoint_id"] != _A_GENESIS_CHECKPOINT_ID
        or body["genesis_head_id"] != _A_GENESIS_HEAD_ID
        or body["genesis_tree_root_sha256"] != "0" * 64
    ):
        _fail(FAIL_BINDING_V1, "durable policy bootstrap/online/genesis linkage differs")
    policy_id = _admission_content_id("durable_anchor_policy", body)
    if root["durable_anchor_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_CONTENT_ID_V1, "durable policy content ID differs")
    online_body = online["body"]
    if (
        durable_public in {row["public_key_hex"] for row in role_keys}
        or body["anchor_key_id"] in {row["key_id"] for row in role_keys}
        or body["anchor_control_domain_id"] in {row["control_domain"] for row in role_keys}
        or durable_public in {bootstrap["public_key_hex"], online_body["online_public_key_hex"],
                              online_body["custody_attestor_public_key_hex"]}
        or body["anchor_key_id"] in {bootstrap["witness_key_id"], online_body["online_key_id"],
                                     online_body["custody_attestor_key_id"]}
        or body["anchor_control_domain_id"] in {bootstrap["witness_control_domain_id"],
                                                online_body["witness_control_domain_id"]}
        or body["anchor_namespace"] in {bootstrap["witness_namespace"],
                                        online_body["witness_namespace"], *_AUTH_ROLE_ORDER}
    ):
        _fail(FAIL_TRUST_ANCHOR_V1, "A key/control/namespace is not fully disjoint")
    preimage = _admission_preimage(
        artifact_kind="durable_anchor_policy", body=body, artifact_id=policy_id,
        signer_key_id=body["anchor_key_id"], key_epoch=body["anchor_key_epoch"],
        purpose=_SIGN_DURABLE_POLICY,
    )
    _signature(root["signature"], expected_key_id=body["anchor_key_id"],
               expected_key_epoch=body["anchor_key_epoch"],
               expected_purpose=_SIGN_DURABLE_POLICY, public_key_hex=durable_public,
               preimage=preimage, label="durable policy")
    root["body"] = body
    return root


def _verify_pure_anchor(root: Mapping[str, object], body: Mapping[str, object],
                        role_keys: tuple[Mapping[str, object], ...]) -> None:
    pure = role_keys[_AUTH_ROLE_ORDER.index("PURE_REPLAYER")]
    signature = _object(root["signature"], frozenset(_AUTH_SIGNATURE_FIELDS),
                        "PURE anchor signature")
    if (
        signature["schema_version"] != _AUTH_SIGNATURE_SCHEMA
        or signature["algorithm"] != _ALGORITHM
        or signature["role"] != "PURE_REPLAYER"
        or signature["purpose"] != "SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR"
        or signature["signer_key_id"] != pure["key_id"]
        or type(signature["key_epoch"]) is not int
        or signature["key_epoch"] != pure["key_epoch"]
        or signature["purpose"] not in pure["authorized_signature_purposes"]
    ):
        _fail(FAIL_SIGNATURE_V1, "PURE anchor signature metadata differs")
    try:
        preimage = _authority_signature_preimage(
            artifact_kind="owner_status_anti_fork_anchor", body=body,
            role="PURE_REPLAYER", purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
            signer_key_id=pure["key_id"], key_epoch=pure["key_epoch"],
        )
    except _AuthorityError as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_SIGNATURE_V1, "PURE anchor preimage rejected"
        ) from exc
    _verify_ed25519(public_key_hex=pure["public_key_hex"],
                    signature_hex=_hex128(signature["signature_hex"], "PURE signature"),
                    preimage=preimage, label="PURE anchor")


def _current_map_leaf_root(mapping: Mapping[str, object]) -> str:
    subject = _without(mapping, "map_leaf_index", "map_tree_size", "map_root_sha256",
                       "map_inclusion_path_sha256_in_exact_order",
                       "presented_to_live_consistency_path_sha256_in_exact_order")
    return _SHA256(_CURRENT_MAP_LEAF_DOMAIN + _canonical(subject)).hexdigest()


def _replay_candidate(anchor_payload: bytes, receipt_payload: bytes,
                      ledger_payload: bytes, checkpoint_payload: bytes,
                      head_payload: bytes, *, lock: object, profile: object,
                      registry: object, role_keys: tuple[Mapping[str, object], ...],
                      bootstrap: Mapping[str, object], attempt_id: str) -> dict[str, object]:
    anchor = _object(_decode(anchor_payload, "candidate anchor"),
                     frozenset(_ANCHOR_FIELDS), "candidate anchor")
    body = _object(anchor["body"], frozenset(_ANCHOR_BODY_FIELDS), "candidate anchor body")
    if (
        anchor["schema_version"] != _ANCHOR_SCHEMA or body["schema_version"] != _ANCHOR_SCHEMA
        or body["status"] != _EVIDENCE_STATUS or body["claim_level"] != _EVIDENCE_CLAIM_LEVEL
        or body["stage_b_trust_lock_id"] != lock.artifact_id
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["attempt_context_id"] != attempt_id
        or body["external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or type(body["anchor_sequence"]) is not int or body["anchor_sequence"] != 1
        or body["previous_external_anchor_receipt_id"] is not None
    ):
        _fail(FAIL_BINDING_V1, "candidate anchor lineage differs")
    for name, prefix in (
        ("current_cut_statement_id", "phase2b_step5_current_cut_statement_v1_"),
        ("current_cut_token_id", "phase2b_step5_logical_time_token_v1_"),
        ("current_cut_checkpoint_id", "phase2b_step5_logical_time_checkpoint_v1_"),
        ("current_cut_head_id", "phase2b_step5_logical_time_head_v1_"),
        ("owner_status_ledger_id", "phase2b_step5_owner_status_ledger_v1_"),
        ("owner_current_head_id", "phase2b_step5_owner_status_head_v1_"),
        ("projection_id", "phase2b_step5_owner_status_projection_v1_"),
    ):
        _content_identifier(body[name], f"anchor {name}", prefix)
    for name in ("current_cut_statement_payload_sha256",):
        _hex64(body[name], f"anchor {name}")
    for name in ("current_cut_token_payload_sha256", "current_cut_checkpoint_payload_sha256",
                 "current_cut_head_payload_sha256", "complete_event_set_root"):
        value = body[name]
        if type(value) is not str or not value.startswith("sha256:") or _HEX64.fullmatch(value[7:]) is None:
            _fail(FAIL_SCHEMA_V1, f"anchor {name} SHA address differs")
    _text(body["anchor_nonce"], "anchor nonce", 256)
    if anchor["anchor_id"] != _evidence_content_id("signed_owner_status_anti_fork_anchor", body):
        _fail(FAIL_CONTENT_ID_V1, "candidate anchor ID differs")
    _verify_pure_anchor(anchor, body, role_keys)

    receipt = _object(_decode(receipt_payload, "candidate receipt"),
                      frozenset(_EXTERNAL_RECEIPT_FIELDS), "candidate receipt")
    tree_root = _SHA256(_MERKLE_LEAF_DOMAIN + anchor_payload).hexdigest()
    if (
        receipt["schema_version"] != _EXTERNAL_RECEIPT_SCHEMA
        or receipt["external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or receipt["witness_namespace"] != bootstrap["witness_namespace"]
        or receipt["witness_control_domain_id"] != bootstrap["witness_control_domain_id"]
        or receipt["witness_key_id"] != bootstrap["witness_key_id"]
        or type(receipt["witness_key_epoch"]) is not int
        or receipt["witness_key_epoch"] != bootstrap["witness_key_epoch"]
        or receipt["anchor_id"] != anchor["anchor_id"]
        or receipt["anchor_payload_sha256"] != _payload_sha(anchor_payload)
        or type(receipt["witness_sequence"]) is not int or receipt["witness_sequence"] != 1
        or receipt["previous_external_receipt_id"] is not None
        or type(receipt["leaf_index"]) is not int or receipt["leaf_index"] != 0
        or type(receipt["tree_size"]) is not int or receipt["tree_size"] != 1
        or receipt["signed_tree_root_sha256"] != tree_root
        or type(receipt["inclusion_path_sha256_in_exact_order"]) is not list
        or receipt["inclusion_path_sha256_in_exact_order"] != []
        or receipt["algorithm"] != _ALGORITHM
    ):
        _fail(FAIL_BINDING_V1, "candidate receipt lineage differs")
    receipt_id = _evidence_content_id(
        "external_owner_status_anchor_receipt",
        _without(receipt, "signature_hex", "receipt_id"),
    )
    if receipt["receipt_id"] != receipt_id:
        _fail(FAIL_CONTENT_ID_V1, "candidate receipt ID differs")
    try:
        receipt_preimage = _external_receipt_preimage(dict(receipt))
    except Exception as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_SIGNATURE_V1, "candidate receipt preimage rejected"
        ) from exc
    _verify_ed25519(public_key_hex=bootstrap["public_key_hex"],
                    signature_hex=_hex128(receipt["signature_hex"], "candidate receipt signature"),
                    preimage=receipt_preimage, label="candidate receipt")

    ledger = _object(_decode(ledger_payload, "candidate ledger"), _LEDGER_FIELDS, "candidate ledger")
    checkpoint = _object(_decode(checkpoint_payload, "candidate checkpoint"),
                         _CHECKPOINT_FIELDS, "candidate checkpoint")
    head = _object(_decode(head_payload, "candidate head"), _HEAD_FIELDS, "candidate head")
    expected_ledger_id = _LEDGER_PREFIX + _SHA256(
        _LEDGER_DOMAIN + _canonical(_without(ledger, "ledger_id"))
    ).hexdigest()
    expected_checkpoint_id = _CHECKPOINT_PREFIX + _SHA256(
        _CHECKPOINT_DOMAIN + _canonical(_without(checkpoint, "checkpoint_id"))
    ).hexdigest()
    expected_head_id = _HEAD_PREFIX + _SHA256(
        _HEAD_DOMAIN + _canonical(_without(head, "head_id"))
    ).hexdigest()
    if not (
        ledger["schema_version"] == _LEDGER_SCHEMA
        and checkpoint["schema_version"] == _CHECKPOINT_SCHEMA
        and head["schema_version"] == _HEAD_SCHEMA
        and ledger["status"] == checkpoint["status"] == head["status"] == _EVIDENCE_STATUS
        and ledger["claim_level"] == checkpoint["claim_level"] == head["claim_level"] == _EVIDENCE_CLAIM_LEVEL
        and ledger["mechanics_claim_boundary"] == checkpoint["mechanics_claim_boundary"]
            == head["mechanics_claim_boundary"] == _EXTERNAL_MECHANICS_BOUNDARY
    ):
        _fail(FAIL_BINDING_V1, "candidate spool claim boundary differs")
    if (
        ledger["ledger_id"] != expected_ledger_id
        or checkpoint["checkpoint_id"] != expected_checkpoint_id
        or head["head_id"] != expected_head_id
        or ledger["stage_b_trust_lock_id"] != lock.artifact_id
        or ledger["authority_profile_id"] != profile.artifact_id
        or ledger["signer_registry_id"] != registry.artifact_id
        or ledger["attempt_context_id"] != attempt_id
        or ledger["external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or ledger["witness_namespace"] != bootstrap["witness_namespace"]
        or ledger["witness_control_domain_id"] != bootstrap["witness_control_domain_id"]
        or ledger["witness_key_id"] != bootstrap["witness_key_id"]
        or type(ledger["witness_key_epoch"]) is not int or ledger["witness_key_epoch"] != 1
        or ledger["initial_anchor_id"] != anchor["anchor_id"]
        or ledger["initial_anchor_payload_sha256"] != _payload_sha(anchor_payload)
        or ledger["initial_receipt_id"] != receipt["receipt_id"]
        or ledger["initial_receipt_payload_sha256"] != _payload_sha(receipt_payload)
        or type(ledger["witness_sequence"]) is not int or ledger["witness_sequence"] != 1
        or type(ledger["tree_size"]) is not int or ledger["tree_size"] != 1
        or ledger["signed_tree_root_sha256"] != tree_root
        or checkpoint["external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or checkpoint["ledger_id"] != ledger["ledger_id"]
        or checkpoint["initial_receipt_id"] != receipt["receipt_id"]
        or checkpoint["initial_receipt_payload_sha256"] != _payload_sha(receipt_payload)
        or type(checkpoint["witness_sequence"]) is not int or checkpoint["witness_sequence"] != 1
        or type(checkpoint["tree_size"]) is not int or checkpoint["tree_size"] != 1
        or checkpoint["signed_tree_root_sha256"] != tree_root
        or type(checkpoint["checkpoint_sequence"]) is not int or checkpoint["checkpoint_sequence"] != 1
        or checkpoint["previous_checkpoint_id"] is not None
        or head["external_witness_policy_id"] != bootstrap["external_witness_policy_id"]
        or head["ledger_id"] != ledger["ledger_id"]
        or head["checkpoint_id"] != checkpoint["checkpoint_id"]
        or head["checkpoint_payload_sha256"] != _payload_sha(checkpoint_payload)
        or head["initial_receipt_id"] != receipt["receipt_id"]
        or head["initial_receipt_payload_sha256"] != _payload_sha(receipt_payload)
        or type(head["witness_sequence"]) is not int or head["witness_sequence"] != 1
        or type(head["tree_size"]) is not int or head["tree_size"] != 1
        or head["signed_tree_root_sha256"] != tree_root
        or head["previous_head_id"] is not None
    ):
        _fail(FAIL_BINDING_V1, "candidate spool lineage differs")
    mapping: dict[str, object] = {
        "external_anchor_id": anchor["anchor_id"],
        "external_anchor_payload_sha256": _payload_sha(anchor_payload),
        "external_receipt_id": receipt["receipt_id"],
        "external_receipt_payload_sha256": _payload_sha(receipt_payload),
        "external_ledger_id": ledger["ledger_id"],
        "external_ledger_payload_sha256": _payload_sha(ledger_payload),
        "external_checkpoint_id": checkpoint["checkpoint_id"],
        "external_checkpoint_payload_sha256": _payload_sha(checkpoint_payload),
        "external_head_id": head["head_id"],
        "external_head_payload_sha256": _payload_sha(head_payload),
        "witness_sequence": 1, "tree_size": 1,
        "signed_tree_root_sha256": tree_root,
        "attempt_context_id": attempt_id, "anchor_sequence": 1,
        "effective_state": "CURRENT_UNSUPERSEDED", "superseded": False,
        "revoked": False, "map_leaf_index": 0, "map_tree_size": 1,
        "map_root_sha256": None,
        "map_inclusion_path_sha256_in_exact_order": [],
        "presented_to_live_consistency_path_sha256_in_exact_order": [],
    }
    mapping["map_root_sha256"] = _current_map_leaf_root(mapping)
    return _mapping(mapping, "derived candidate mapping", attempt_id)


def _genesis_mapping(attempt_id: str) -> dict[str, object]:
    return {
        "external_anchor_id": None, "external_anchor_payload_sha256": None,
        "external_receipt_id": None, "external_receipt_payload_sha256": None,
        "external_ledger_id": None, "external_ledger_payload_sha256": None,
        "external_checkpoint_id": None, "external_checkpoint_payload_sha256": None,
        "external_head_id": None, "external_head_payload_sha256": None,
        "witness_sequence": 0, "tree_size": 0, "signed_tree_root_sha256": None,
        "attempt_context_id": attempt_id, "anchor_sequence": 0,
        "effective_state": "ABSENT", "superseded": False, "revoked": False,
        "map_leaf_index": None, "map_tree_size": 0, "map_root_sha256": None,
        "map_inclusion_path_sha256_in_exact_order": [],
        "presented_to_live_consistency_path_sha256_in_exact_order": [],
    }


def _mapping(value: object, label: str, attempt_id: str) -> dict[str, object]:
    mapping = _object(value, frozenset(_DURABLE_MAPPING_FIELDS), label)
    sequence = _integer(mapping["witness_sequence"], f"{label} witness sequence")
    tree_size = _integer(mapping["tree_size"], f"{label} tree size")
    anchor_sequence = _integer(mapping["anchor_sequence"], f"{label} anchor sequence")
    map_tree_size = _integer(mapping["map_tree_size"], f"{label} map tree size")
    if mapping["attempt_context_id"] != attempt_id:
        _fail(FAIL_BINDING_V1, f"{label} attempt differs")
    if mapping["superseded"] is not False or mapping["revoked"] is not False:
        _fail(FAIL_BINDING_V1, f"{label} state booleans differ")
    for name in ("map_inclusion_path_sha256_in_exact_order",
                 "presented_to_live_consistency_path_sha256_in_exact_order"):
        path = mapping[name]
        if type(path) is not list or len(path) > 64:
            _fail(FAIL_SCHEMA_V1, f"{label} {name} shape differs")
        for digest in path:
            _hex64(digest, f"{label} {name} digest")
    if sequence == 0 or tree_size == 0:
        if mapping != _genesis_mapping(attempt_id):
            _fail(FAIL_BINDING_V1, f"{label} partial genesis differs")
        return mapping
    for name in ("external_anchor_id", "external_receipt_id", "external_ledger_id",
                 "external_checkpoint_id", "external_head_id"):
        _identifier(mapping[name], f"{label} {name}")
    for name in ("external_anchor_payload_sha256", "external_receipt_payload_sha256",
                 "external_ledger_payload_sha256", "external_checkpoint_payload_sha256",
                 "external_head_payload_sha256", "signed_tree_root_sha256"):
        _hex64(mapping[name], f"{label} {name}")
    if (
        sequence != 1 or tree_size != 1 or anchor_sequence != 1
        or mapping["effective_state"] != "CURRENT_UNSUPERSEDED"
        or type(mapping["map_leaf_index"]) is not int or mapping["map_leaf_index"] != 0
        or map_tree_size != 1
        or mapping["map_inclusion_path_sha256_in_exact_order"] != []
        or mapping["presented_to_live_consistency_path_sha256_in_exact_order"] != []
        or mapping["map_root_sha256"] != _current_map_leaf_root(mapping)
    ):
        _fail(FAIL_BINDING_V1, f"{label} authenticated current mapping differs")
    return mapping


def _common(body: Mapping[str, object], context: Mapping[str, object], label: str) -> None:
    if (
        body["status"] != _ADMISSION_STATUS
        or body["claim_level"] != _ADMISSION_CLAIM_LEVEL
        or body["stage_b_trust_lock_id"] != context["lock"].artifact_id
        or body["authority_profile_id"] != context["profile"].artifact_id
        or body["signer_registry_id"] != context["registry"].artifact_id
        or body["attempt_context_id"] != context["attempt_id"]
        or body["bootstrap_external_witness_policy_id"]
            != context["bootstrap"]["external_witness_policy_id"]
        or body["online_witness_policy_id"] != context["online"]["online_witness_policy_id"]
        or body["durable_anchor_policy_id"] != context["durable"]["durable_anchor_policy_id"]
    ):
        _fail(FAIL_BINDING_V1, f"{label} common authority/attempt/policy binding differs")


def _load_protocol(payload: bytes, *, label: str, schema: str,
                   fields: frozenset[str], body_fields: frozenset[str],
                   artifact_kind: str, id_field: str,
                   expected_key_id: str, expected_key_epoch: int,
                   expected_purpose: str, public_key_hex: str) -> tuple[dict[str, object], dict[str, object]]:
    root = _object(_decode(payload, label), fields, label)
    body = _object(root["body"], body_fields, f"{label} body")
    if root["schema_version"] != schema or body["schema_version"] != schema:
        _fail(FAIL_SCHEMA_V1, f"{label} schema differs")
    expected_id = _admission_content_id(artifact_kind, body)
    if root[id_field] != expected_id:
        _fail(FAIL_CONTENT_ID_V1, f"{label} ID differs")
    preimage = _admission_preimage(
        artifact_kind=artifact_kind, body=body, artifact_id=expected_id,
        signer_key_id=expected_key_id, key_epoch=expected_key_epoch,
        purpose=expected_purpose,
    )
    _signature(root["signature"], expected_key_id=expected_key_id,
               expected_key_epoch=expected_key_epoch,
               expected_purpose=expected_purpose, public_key_hex=public_key_hex,
               preimage=preimage, label=label)
    root["body"] = body
    return root, body


def _sign_protocol(*, artifact_kind: str, schema: str, id_field: str,
                   body: Mapping[str, object], private_key: object,
                   signer_key_id: str, key_epoch: int,
                   purpose: str) -> bytes:
    exact_body = dict(body)
    artifact_id = _admission_content_id(artifact_kind, exact_body)
    preimage = _admission_preimage(
        artifact_kind=artifact_kind, body=exact_body, artifact_id=artifact_id,
        signer_key_id=signer_key_id, key_epoch=key_epoch, purpose=purpose,
    )
    try:
        signature_hex = private_key.sign(preimage).hex()
    except Exception as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_KEY_V1, "online private-key signing failed"
        ) from exc
    root = {
        "schema_version": schema,
        "body": exact_body,
        id_field: artifact_id,
        "signature": {
            "schema_version": _SIGNATURE_SCHEMA, "algorithm": _ALGORITHM,
            "signer_key_id": signer_key_id, "key_epoch": key_epoch,
            "purpose": purpose, "signature_hex": signature_hex,
        },
    }
    return _admission_canonical(root)


def _read_bounded_proc(path: str, maximum: int = 64 * 1024) -> bytes:
    try:
        descriptor = _OS_OPEN(path, _OS_READ_FLAGS)
    except OSError as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_RUNTIME_V1, f"cannot open concrete procfs record {path}"
        ) from exc
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = _OS_READ(descriptor, min(4096, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                _fail(FAIL_RUNTIME_V1, f"procfs record {path} exceeds bound")
        return b"".join(chunks)
    finally:
        _OS_CLOSE(descriptor)


def _observe_process() -> dict[str, object]:
    pid = _OS_GETPID()
    uid = _OS_GETUID()
    gid = _OS_GETGID()
    raw_stat = _read_bounded_proc(f"/proc/{pid}/stat")
    close = raw_stat.rfind(b")")
    if close < 1:
        _fail(FAIL_RUNTIME_V1, "procfs stat comm boundary differs")
    fields = raw_stat[close + 2:].split()
    if len(fields) < 20:
        _fail(FAIL_RUNTIME_V1, "procfs stat field count differs")
    try:
        start_ticks = int(fields[19])
        user_ns_inode = _OS_STAT(f"/proc/{pid}/ns/user", follow_symlinks=True).st_ino
    except (OSError, TypeError, ValueError) as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_RUNTIME_V1, "concrete process identity observation failed"
        ) from exc
    for value, label in ((pid, "PID"), (uid, "UID"), (gid, "GID"),
                         (start_ticks, "start ticks"), (user_ns_inode, "user namespace inode")):
        _integer(value, f"local process {label}", 1 if label not in {"UID", "GID"} else 0)
    return {
        "pid": pid, "uid": uid, "gid": gid,
        "process_start_ticks": start_ticks,
        "user_namespace_inode": user_ns_inode,
    }


def _runtime_binding(context: Mapping[str, object], observation: Mapping[str, object]) -> dict[str, object]:
    body = context["online"]["body"]
    if (
        observation["uid"] != body["service_uid"]
        or observation["gid"] != body["service_gid"]
        or observation["user_namespace_inode"] != body["service_user_namespace_inode"]
    ):
        _fail(FAIL_RUNTIME_V1, "concrete process UID/GID/user namespace differs from pinned online policy")
    runtime_subject = {
        "schema_version": "hegel-machine-phase2b-step5-online-witness-local-runtime-observation/1",
        "service_binding_id": body["service_binding_id"],
        "runtime_closure_policy_id": body["runtime_closure_policy_id"],
        "pid": observation["pid"], "uid": observation["uid"], "gid": observation["gid"],
        "process_start_ticks": observation["process_start_ticks"],
        "user_namespace_inode": observation["user_namespace_inode"],
        "control_domain_id": body["witness_control_domain_id"],
    }
    runtime_bytes = _canonical(runtime_subject)
    runtime_digest = _SHA256(runtime_bytes).hexdigest()
    custody_subject = {
        "schema_version": "hegel-machine-phase2b-step5-online-witness-local-key-observation/1",
        "custody_attestation_policy_id": body["custody_attestation_policy_id"],
        "online_key_id": body["online_key_id"],
        "online_public_key_hex": body["online_public_key_hex"],
        "process_start_ticks": observation["process_start_ticks"],
    }
    custody_bytes = _canonical(custody_subject)
    custody_digest = _SHA256(custody_bytes).hexdigest()
    return {
        "service_binding_id": body["service_binding_id"],
        "runtime_instance_id": "phase2b_step5_online_witness_local_runtime_v1_" + runtime_digest,
        "peer_pid": observation["pid"], "peer_uid": observation["uid"],
        "peer_gid": observation["gid"],
        "user_namespace_inode": observation["user_namespace_inode"],
        "process_start_ticks": observation["process_start_ticks"],
        "control_domain_id": body["witness_control_domain_id"],
        "runtime_closure_policy_id": body["runtime_closure_policy_id"],
        "runtime_closure_report_id": "phase2b_step5_online_witness_local_runtime_report_v1_" + runtime_digest,
        "runtime_closure_report_payload_sha256": runtime_digest,
        "custody_attestation_policy_id": body["custody_attestation_policy_id"],
        "custody_attestation_id": "phase2b_step5_online_witness_local_custody_observation_v1_" + custody_digest,
        "custody_attestation_payload_sha256": custody_digest,
        "online_key_nonexportable": True,
    }


def _validate_runtime_binding(binding: object, context: Mapping[str, object],
                              label: str) -> dict[str, object]:
    value = _object(binding, frozenset(_RUNTIME_BINDING_FIELDS), label)
    policy = context["online"]["body"]
    for name in ("service_binding_id", "runtime_instance_id", "control_domain_id",
                 "runtime_closure_policy_id", "runtime_closure_report_id",
                 "custody_attestation_policy_id", "custody_attestation_id"):
        _identifier(value[name], f"{label} {name}")
    for name in ("peer_pid", "peer_uid", "peer_gid", "user_namespace_inode",
                 "process_start_ticks"):
        _integer(value[name], f"{label} {name}", 1)
    _hex64(value["runtime_closure_report_payload_sha256"], f"{label} runtime SHA")
    _hex64(value["custody_attestation_payload_sha256"], f"{label} custody SHA")
    if (
        value["service_binding_id"] != policy["service_binding_id"]
        or value["peer_uid"] != policy["service_uid"]
        or value["peer_gid"] != policy["service_gid"]
        or value["user_namespace_inode"] != policy["service_user_namespace_inode"]
        or value["control_domain_id"] != policy["witness_control_domain_id"]
        or value["runtime_closure_policy_id"] != policy["runtime_closure_policy_id"]
        or value["custody_attestation_policy_id"] != policy["custody_attestation_policy_id"]
        or value["online_key_nonexportable"] is not True
    ):
        _fail(FAIL_BINDING_V1, f"{label} policy binding differs")
    return value


def _require_private_key(private_key: object, online: Mapping[str, object]) -> object:
    if _Ed25519PrivateKey is None or _serialization is None:
        _fail(FAIL_KEY_V1, "cryptography Ed25519 backend unavailable")
    if not isinstance(private_key, _Ed25519PrivateKey):
        _fail(FAIL_KEY_V1, "fixed Ed25519 online private key required")
    try:
        public_hex = private_key.public_key().public_bytes(
            _serialization.Encoding.Raw, _serialization.PublicFormat.Raw
        ).hex()
    except Exception as exc:
        raise Phase2BStep5R01OnlineWitnessErrorV1(
            FAIL_KEY_V1, "cannot derive online public key"
        ) from exc
    if public_hex != online["body"]["online_public_key_hex"]:
        _fail(FAIL_KEY_V1, "online private key differs from raw-pinned policy")
    return private_key


def _context_fingerprint(context: Mapping[str, object]) -> str:
    return _SHA256(
        b"HEGEL/PHASE2B/STEP5/R01/ONLINE_WITNESS/CONTEXT/V1\0"
        + b"".join(_SHA256(payload).digest() for payload in context["raw_payloads"])
        + context["attempt_id"].encode("ascii")
    ).hexdigest()


def _load_context(*args: object, **kwargs: object) -> dict[str, object]:
    names = (
        "stage_b_trust_lock_payload", "authority_profile_payload", "signer_registry_payload",
        "bootstrap_external_witness_policy_payload", "online_witness_policy_payload",
        "durable_anchor_policy_payload", "candidate_anchor_payload", "candidate_receipt_payload",
        "candidate_ledger_payload", "candidate_checkpoint_payload", "candidate_head_payload",
    )
    if len(args) != len(names):
        _fail(FAIL_INPUT_V1, "actor constructor positional raw-payload arity differs")
    raw = dict(zip(names, args, strict=True))
    if any(type(value) is not bytes for value in raw.values()):
        _fail(FAIL_INPUT_V1, "actor constructor accepts exact raw payload bytes only")
    required = frozenset({
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "out_of_band_expected_bootstrap_external_witness_policy_id_bytes",
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest",
        "out_of_band_expected_online_witness_policy_id_bytes",
        "out_of_band_expected_online_witness_policy_payload_sha256_digest",
        "out_of_band_expected_durable_anchor_policy_id_bytes",
        "out_of_band_expected_durable_anchor_policy_payload_sha256_digest",
        "expected_attempt_context_id_bytes", "online_witness_private_key",
    })
    if frozenset(kwargs) != required:
        _fail(FAIL_INPUT_V1, "actor constructor exact keyword set differs")
    attempt_id = _decode_expected_identifier(
        kwargs["expected_attempt_context_id_bytes"], "expected attempt context ID"
    )
    lock, profile, registry, role_keys = _load_authority_context(
        raw["stage_b_trust_lock_payload"], raw["authority_profile_payload"],
        raw["signer_registry_payload"],
        expected_lock_id_bytes=kwargs["out_of_band_expected_lock_id_bytes"],
        expected_lock_digest=kwargs["out_of_band_expected_lock_payload_sha256_digest"],
    )
    bootstrap = _load_bootstrap_policy(
        raw["bootstrap_external_witness_policy_payload"],
        expected_id_bytes=kwargs["out_of_band_expected_bootstrap_external_witness_policy_id_bytes"],
        expected_digest=kwargs["out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest"],
        role_keys=role_keys,
    )
    online = _load_online_policy(
        raw["online_witness_policy_payload"],
        expected_id_bytes=kwargs["out_of_band_expected_online_witness_policy_id_bytes"],
        expected_digest=kwargs["out_of_band_expected_online_witness_policy_payload_sha256_digest"],
        role_keys=role_keys, authority_profile=profile,
        bootstrap_payload=raw["bootstrap_external_witness_policy_payload"],
        bootstrap=bootstrap,
    )
    durable = _load_durable_policy(
        raw["durable_anchor_policy_payload"],
        expected_id_bytes=kwargs["out_of_band_expected_durable_anchor_policy_id_bytes"],
        expected_digest=kwargs["out_of_band_expected_durable_anchor_policy_payload_sha256_digest"],
        role_keys=role_keys, bootstrap_payload=raw["bootstrap_external_witness_policy_payload"],
        bootstrap=bootstrap, online=online,
    )
    candidate_raw = tuple(raw[name] for name in names[6:])
    mapping = _replay_candidate(
        *candidate_raw, lock=lock, profile=profile, registry=registry,
        role_keys=role_keys, bootstrap=bootstrap, attempt_id=attempt_id,
    )
    genesis = _mapping(durable["body"]["oob_high_water_mapping"],
                       "durable OOB high-water mapping", attempt_id)
    if (
        genesis != _genesis_mapping(attempt_id)
        or durable["body"]["oob_high_water_checkpoint_sequence"] != 0
        or durable["body"]["oob_high_water_head_sequence"] != 0
        or durable["body"]["oob_high_water_checkpoint_id"] != _A_GENESIS_CHECKPOINT_ID
        or durable["body"]["oob_high_water_head_id"] != _A_GENESIS_HEAD_ID
        or durable["body"]["oob_high_water_checkpoint_payload_sha256"] != "0" * 64
        or durable["body"]["oob_high_water_head_payload_sha256"] != "0" * 64
        or durable["body"]["oob_high_water_tree_root_sha256"] != "0" * 64
    ):
        _fail(FAIL_BINDING_V1, "only exact durable genesis high-water is frozen")
    private_key = _require_private_key(kwargs["online_witness_private_key"], online)
    context: dict[str, object] = {
        "lock": lock, "profile": profile, "registry": registry,
        "role_keys": role_keys, "attempt_id": attempt_id,
        "bootstrap": bootstrap, "online": online, "durable": durable,
        "candidate_mapping": mapping, "genesis_mapping": genesis,
        "candidate_raw": candidate_raw,
        "raw_payloads": tuple(raw[name] for name in names),
        "private_key": private_key,
    }
    context["fingerprint"] = _context_fingerprint(context)
    return context


@dataclass(frozen=True, slots=True)
class _FenceSessionV1:
    state: str
    context_fingerprint: str
    q1_request_payload: bytes
    q1_response_payload: bytes
    a_request_payload: bytes
    confirmation_nonce: str
    runtime_observation: Mapping[str, object]
    runtime_binding: Mapping[str, object]
    a_checkpoint_payload: bytes | None = None
    a_head_payload: bytes | None = None
    a_receipt_payload: bytes | None = None
    q2_request_payload: bytes | None = None
    q2_response_payload: bytes | None = None


_FENCE_LOCK = threading.RLock()
_FENCE_SESSIONS: dict[str, _FenceSessionV1] = {}


@dataclass(frozen=True, slots=True, init=False)
class PreparedR01OnlineWitnessMechanicsV1:
    schema_version: str
    mechanics_claim_boundary: str
    mechanics_only: bool
    witness_namespace: str
    attempt_context_id: str
    q1_request_id: str
    q1_response_id: str
    a_request_id: str
    confirmation_nonce_sha256: str
    q1_response_payload: bytes
    a_compare_and_append_request_payload: bytes
    replayed_existing_session: bool
    in_process_namespace_fence_verified: bool
    local_process_observation_captured_at_q1: bool
    runtime_binding_self_report_only: bool
    distinct_live_uid_gid_verified: bool
    live_transport_verified: bool
    durability_verified: bool
    non_equivocation_verified: bool
    global_anti_fork_verified: bool
    logical_time_ancestry_verified: bool
    restart_recovery_verified: bool
    durable_session_verified: bool
    live_external_currentness_verified: bool
    runtime_closure_verified: bool
    key_custody_verified: bool
    authority_evidence: bool
    r01_satisfied: bool
    readiness_evidence: bool
    execution_authorized: bool

    def __new__(cls, *args: object, **kwargs: object) -> "PreparedR01OnlineWitnessMechanicsV1":
        raise TypeError("online-witness mechanics results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("online-witness mechanics results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class CompletedR01OnlineWitnessMechanicsV1:
    schema_version: str
    mechanics_claim_boundary: str
    mechanics_only: bool
    witness_namespace: str
    attempt_context_id: str
    q2_request_id: str
    q2_response_id: str
    q2_response_payload: bytes
    replayed_terminal_result: bool
    in_process_namespace_fence_verified: bool
    same_local_process_observation_across_q1_q2: bool
    candidate_raw_replay_equal_across_q1_q2: bool
    exact_a_triple_verified: bool
    exact_bootstrap_q2_request_verified: bool
    runtime_binding_self_report_only: bool
    distinct_live_uid_gid_verified: bool
    logical_time_ancestry_verified: bool
    restart_recovery_verified: bool
    durable_session_verified: bool
    live_transport_verified: bool
    live_external_currentness_verified: bool
    durability_verified: bool
    runtime_closure_verified: bool
    key_custody_verified: bool
    non_equivocation_verified: bool
    global_anti_fork_verified: bool
    authority_evidence: bool
    r01_satisfied: bool
    readiness_evidence: bool
    execution_authorized: bool

    def __new__(cls, *args: object, **kwargs: object) -> "CompletedR01OnlineWitnessMechanicsV1":
        raise TypeError("online-witness mechanics results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("online-witness mechanics results are privately issued")


def _build_result_lineage():
    prepared_type = PreparedR01OnlineWitnessMechanicsV1
    completed_type = CompletedR01OnlineWitnessMechanicsV1
    error_type = Phase2BStep5R01OnlineWitnessErrorV1
    failure_code = FAIL_RESULT_V1
    prepared_issued: dict[int, tuple[object, tuple[object, ...]]] = {}
    completed_issued: dict[int, tuple[object, tuple[object, ...]]] = {}

    def reject(detail: str) -> NoReturn:
        raise error_type(failure_code, detail)

    def issue(result_type: type, registry: dict[int, tuple[object, tuple[object, ...]]],
              values: Mapping[str, object]) -> object:
        result = object.__new__(result_type)
        fields = tuple(result_type.__dataclass_fields__)
        if frozenset(values) != frozenset(fields):
            reject("private result issuance keyset differs")
        fingerprint = tuple(values[name] for name in fields)
        for name, value in zip(fields, fingerprint, strict=True):
            object.__setattr__(result, name, value)
        registry[id(result)] = (result, fingerprint)
        return result

    def require(value: object, result_type: type,
                registry: dict[int, tuple[object, tuple[object, ...]]]) -> object:
        if type(value) is not result_type:
            reject("privately issued online-witness result required")
        record = registry.get(id(value))
        fields = tuple(result_type.__dataclass_fields__)
        try:
            observed = tuple(getattr(value, name) for name in fields)
        except (AttributeError, TypeError):
            reject("online-witness result fields are incomplete")
        if (
            record is None
            or record[0] is not value
            or len(observed) != len(record[1])
            or any(
                type(item) is not type(expected) or item != expected
                for item, expected in zip(observed, record[1], strict=True)
            )
        ):
            reject("online-witness result lacks private lineage")
        return value

    def issue_prepared(**values: object) -> PreparedR01OnlineWitnessMechanicsV1:
        return issue(prepared_type, prepared_issued, values)  # type: ignore[return-value]

    def issue_completed(**values: object) -> CompletedR01OnlineWitnessMechanicsV1:
        return issue(completed_type, completed_issued, values)  # type: ignore[return-value]

    def require_prepared(value: object) -> PreparedR01OnlineWitnessMechanicsV1:
        return require(value, prepared_type, prepared_issued)  # type: ignore[return-value]

    def require_completed(value: object) -> CompletedR01OnlineWitnessMechanicsV1:
        return require(value, completed_type, completed_issued)  # type: ignore[return-value]

    return issue_prepared, issue_completed, require_prepared, require_completed


(_issue_prepared, _issue_completed,
 require_prepared_r01_online_witness_mechanics_v1,
 require_completed_r01_online_witness_mechanics_v1) = _build_result_lineage()
del _build_result_lineage


def _result_common_false() -> dict[str, object]:
    return {
        "logical_time_ancestry_verified": False,
        "restart_recovery_verified": False,
        "durable_session_verified": False,
        "live_external_currentness_verified": False,
        "runtime_closure_verified": False,
        "key_custody_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
    }


def _prepared_result(context: Mapping[str, object], session: _FenceSessionV1,
                     replayed: bool) -> PreparedR01OnlineWitnessMechanicsV1:
    q1 = _decode(session.q1_request_payload, "stored Q1 request")
    q1_response = _decode(session.q1_response_payload, "stored Q1 response")
    a_request = _decode(session.a_request_payload, "stored A request")
    return _issue_prepared(
        schema_version=ONLINE_WITNESS_MECHANICS_RESULT_SCHEMA_V1,
        mechanics_claim_boundary=ONLINE_WITNESS_MECHANICS_CLAIM_BOUNDARY_V1,
        mechanics_only=True,
        witness_namespace=context["online"]["body"]["witness_namespace"],
        attempt_context_id=context["attempt_id"],
        q1_request_id=q1["q1_request_id"], q1_response_id=q1_response["q1_response_id"],
        a_request_id=a_request["a_request_id"],
        confirmation_nonce_sha256=_SHA256(bytes.fromhex(session.confirmation_nonce)).hexdigest(),
        q1_response_payload=session.q1_response_payload,
        a_compare_and_append_request_payload=session.a_request_payload,
        replayed_existing_session=replayed,
        in_process_namespace_fence_verified=True,
        local_process_observation_captured_at_q1=True,
        runtime_binding_self_report_only=True,
        distinct_live_uid_gid_verified=False,
        live_transport_verified=False,
        durability_verified=False,
        non_equivocation_verified=False,
        global_anti_fork_verified=False,
        **_result_common_false(),
    )


def _completed_result(context: Mapping[str, object], session: _FenceSessionV1,
                      replayed: bool) -> CompletedR01OnlineWitnessMechanicsV1:
    q2 = _decode(session.q2_request_payload, "stored Q2 request")
    response = _decode(session.q2_response_payload, "stored Q2 response")
    values = _result_common_false()
    values.update({
        "live_transport_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
    })
    return _issue_completed(
        schema_version=ONLINE_WITNESS_MECHANICS_RESULT_SCHEMA_V1,
        mechanics_claim_boundary=ONLINE_WITNESS_MECHANICS_CLAIM_BOUNDARY_V1,
        mechanics_only=True,
        witness_namespace=context["online"]["body"]["witness_namespace"],
        attempt_context_id=context["attempt_id"],
        q2_request_id=q2["q2_request_id"], q2_response_id=response["q2_response_id"],
        q2_response_payload=session.q2_response_payload,
        replayed_terminal_result=replayed,
        in_process_namespace_fence_verified=True,
        same_local_process_observation_across_q1_q2=True,
        candidate_raw_replay_equal_across_q1_q2=True,
        exact_a_triple_verified=True,
        exact_bootstrap_q2_request_verified=True,
        runtime_binding_self_report_only=True,
        distinct_live_uid_gid_verified=False,
        **values,
    )


def _begin_prepare(context: Mapping[str, object], q1_payload: bytes) -> PreparedR01OnlineWitnessMechanicsV1:
    replayed_mapping = _replay_candidate(
        *context["candidate_raw"], lock=context["lock"], profile=context["profile"],
        registry=context["registry"], role_keys=context["role_keys"],
        bootstrap=context["bootstrap"], attempt_id=context["attempt_id"],
    )
    if replayed_mapping != context["candidate_mapping"]:
        _fail(FAIL_BINDING_V1, "candidate raw replay drifted before Q1")
    q1, body = _load_protocol(
        q1_payload, label="Q1 PREPARE request", schema=_Q1_REQUEST_SCHEMA,
        fields=frozenset(_Q1_REQUEST_FIELDS), body_fields=frozenset(_Q1_REQUEST_BODY_FIELDS),
        artifact_kind="q1_prepare_request", id_field="q1_request_id",
        expected_key_id=context["online"]["body"]["bootstrap_key_id"],
        expected_key_epoch=context["online"]["body"]["bootstrap_key_epoch"],
        expected_purpose=_SIGN_Q1_REQUEST,
        public_key_hex=context["online"]["body"]["bootstrap_public_key_hex"],
    )
    _common(body, context, "Q1 request")
    prepare_nonce = _nonce(body["prepare_nonce"], "Q1 prepare nonce")
    durable = context["durable"]["body"]
    if (
        _mapping(body["candidate_mapping"], "Q1 candidate mapping", context["attempt_id"])
            != context["candidate_mapping"]
        or body["expected_previous_durable_checkpoint_id"] != durable["oob_high_water_checkpoint_id"]
        or body["expected_previous_durable_head_id"] != durable["oob_high_water_head_id"]
        or body["expected_previous_durable_checkpoint_payload_sha256"]
            != durable["oob_high_water_checkpoint_payload_sha256"]
        or body["expected_previous_durable_head_payload_sha256"]
            != durable["oob_high_water_head_payload_sha256"]
        or body["expected_previous_durable_checkpoint_sequence"]
            != durable["oob_high_water_checkpoint_sequence"]
        or body["expected_previous_durable_head_sequence"] != durable["oob_high_water_head_sequence"]
        or body["expected_previous_durable_tree_root_sha256"] != durable["oob_high_water_tree_root_sha256"]
    ):
        _fail(FAIL_BINDING_V1, "Q1 candidate or durable high-water differs")
    namespace = context["online"]["body"]["witness_namespace"]
    with _FENCE_LOCK:
        existing = _FENCE_SESSIONS.get(namespace)
        if existing is not None:
            if existing.state == "POISONED":
                _fail(FAIL_POISONED_V1, "namespace session is permanently poisoned")
            if existing.context_fingerprint != context["fingerprint"] or existing.q1_request_payload != q1_payload:
                if existing.state == "PREPARED":
                    _FENCE_SESSIONS[namespace] = _FenceSessionV1(
                        state="POISONED",
                        context_fingerprint=existing.context_fingerprint,
                        q1_request_payload=existing.q1_request_payload,
                        q1_response_payload=existing.q1_response_payload,
                        a_request_payload=existing.a_request_payload,
                        confirmation_nonce=existing.confirmation_nonce,
                        runtime_observation=existing.runtime_observation,
                        runtime_binding=existing.runtime_binding,
                    )
                _fail(FAIL_FENCE_V1, "namespace slot already holds a different Q1/context")
            return _prepared_result(context, existing, True)
        try:
            observation = _observe_process()
            runtime_binding = _validate_runtime_binding(
                _runtime_binding(context, observation), context, "Q1 local runtime binding"
            )
            confirmation_nonce = _TOKEN_BYTES(32).hex()
            if confirmation_nonce == prepare_nonce or confirmation_nonce == "0" * 64:
                _fail(FAIL_RUNTIME_V1, "generated confirmation nonce collision")
            common = {
                "status": _ADMISSION_STATUS, "claim_level": _ADMISSION_CLAIM_LEVEL,
                "stage_b_trust_lock_id": context["lock"].artifact_id,
                "authority_profile_id": context["profile"].artifact_id,
                "signer_registry_id": context["registry"].artifact_id,
                "attempt_context_id": context["attempt_id"],
                "bootstrap_external_witness_policy_id": context["bootstrap"]["external_witness_policy_id"],
                "online_witness_policy_id": context["online"]["online_witness_policy_id"],
                "durable_anchor_policy_id": context["durable"]["durable_anchor_policy_id"],
            }
            response_body = {
                **common, "schema_version": _Q1_RESPONSE_SCHEMA,
                "q1_request_id": q1["q1_request_id"],
                "q1_request_payload_sha256": _payload_sha(q1_payload),
                "prepare_nonce": prepare_nonce, "confirmation_nonce": confirmation_nonce,
                "observed_current_mapping": context["candidate_mapping"],
                "prepared_candidate_mapping": context["candidate_mapping"],
                "runtime_custody_binding": runtime_binding,
                "prepare_outcome": _PREPARE_OUTCOME,
            }
            q1_response_payload = _sign_protocol(
                artifact_kind="q1_prepare_response", schema=_Q1_RESPONSE_SCHEMA,
                id_field="q1_response_id", body=response_body,
                private_key=context["private_key"],
                signer_key_id=context["online"]["body"]["online_key_id"],
                key_epoch=context["online"]["body"]["online_key_epoch"],
                purpose=_SIGN_Q1_RESPONSE,
            )
            q1_response, _ = _load_protocol(
                q1_response_payload, label="composed Q1 response", schema=_Q1_RESPONSE_SCHEMA,
                fields=frozenset(_Q1_RESPONSE_FIELDS), body_fields=frozenset(_Q1_RESPONSE_BODY_FIELDS),
                artifact_kind="q1_prepare_response", id_field="q1_response_id",
                expected_key_id=context["online"]["body"]["online_key_id"],
                expected_key_epoch=context["online"]["body"]["online_key_epoch"],
                expected_purpose=_SIGN_Q1_RESPONSE,
                public_key_hex=context["online"]["body"]["online_public_key_hex"],
            )
            a_body = {
                **common, "schema_version": _A_REQUEST_SCHEMA,
                "q1_request_id": q1["q1_request_id"],
                "q1_request_payload_sha256": _payload_sha(q1_payload),
                "q1_response_id": q1_response["q1_response_id"],
                "q1_response_payload_sha256": _payload_sha(q1_response_payload),
                "prepare_nonce": prepare_nonce, "confirmation_nonce": confirmation_nonce,
                "expected_current_mapping": context["genesis_mapping"],
                "proposed_current_mapping": context["candidate_mapping"],
                "expected_previous_durable_checkpoint_id": durable["oob_high_water_checkpoint_id"],
                "expected_previous_durable_checkpoint_payload_sha256": durable["oob_high_water_checkpoint_payload_sha256"],
                "expected_previous_durable_head_id": durable["oob_high_water_head_id"],
                "expected_previous_durable_head_payload_sha256": durable["oob_high_water_head_payload_sha256"],
                "expected_previous_durable_checkpoint_sequence": durable["oob_high_water_checkpoint_sequence"],
                "expected_previous_durable_head_sequence": durable["oob_high_water_head_sequence"],
                "expected_previous_durable_tree_root_sha256": durable["oob_high_water_tree_root_sha256"],
                "compare_and_append_mode": "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE",
            }
            a_request_payload = _sign_protocol(
                artifact_kind="a_compare_and_append_request", schema=_A_REQUEST_SCHEMA,
                id_field="a_request_id", body=a_body, private_key=context["private_key"],
                signer_key_id=context["online"]["body"]["online_key_id"],
                key_epoch=context["online"]["body"]["online_key_epoch"],
                purpose=_SIGN_A_REQUEST,
            )
            _load_protocol(
                a_request_payload, label="composed A request", schema=_A_REQUEST_SCHEMA,
                fields=frozenset(_A_REQUEST_FIELDS), body_fields=frozenset(_A_REQUEST_BODY_FIELDS),
                artifact_kind="a_compare_and_append_request", id_field="a_request_id",
                expected_key_id=context["online"]["body"]["online_key_id"],
                expected_key_epoch=context["online"]["body"]["online_key_epoch"],
                expected_purpose=_SIGN_A_REQUEST,
                public_key_hex=context["online"]["body"]["online_public_key_hex"],
            )
            session = _FenceSessionV1(
                state="PREPARED", context_fingerprint=context["fingerprint"],
                q1_request_payload=q1_payload, q1_response_payload=q1_response_payload,
                a_request_payload=a_request_payload, confirmation_nonce=confirmation_nonce,
                runtime_observation=MappingProxyType(dict(observation)),
                runtime_binding=MappingProxyType(dict(runtime_binding)),
            )
            _FENCE_SESSIONS[namespace] = session
            return _prepared_result(context, session, False)
        except Exception:
            if namespace not in _FENCE_SESSIONS:
                _FENCE_SESSIONS[namespace] = _FenceSessionV1(
                    state="POISONED", context_fingerprint=context["fingerprint"],
                    q1_request_payload=q1_payload, q1_response_payload=b"",
                    a_request_payload=b"", confirmation_nonce="0" * 64,
                    runtime_observation=MappingProxyType({}),
                    runtime_binding=MappingProxyType({}),
                )
            raise


def _confirm(context: Mapping[str, object], a_checkpoint_payload: bytes,
             a_head_payload: bytes, a_receipt_payload: bytes,
             q2_payload: bytes) -> CompletedR01OnlineWitnessMechanicsV1:
    if any(type(item) is not bytes for item in (
        a_checkpoint_payload, a_head_payload, a_receipt_payload, q2_payload
    )):
        _fail(FAIL_INPUT_V1, "confirm accepts exact raw A/Q2 bytes only")
    namespace = context["online"]["body"]["witness_namespace"]
    with _FENCE_LOCK:
        session = _FENCE_SESSIONS.get(namespace)
        if session is None:
            _fail(FAIL_FENCE_V1, "no in-process prepared namespace session")
        if session.state == "POISONED":
            _fail(FAIL_POISONED_V1, "namespace session is permanently poisoned")
        exact_terminal = (
            session.state == "COMPLETE"
            and session.context_fingerprint == context["fingerprint"]
            and session.a_checkpoint_payload == a_checkpoint_payload
            and session.a_head_payload == a_head_payload
            and session.a_receipt_payload == a_receipt_payload
            and session.q2_request_payload == q2_payload
        )
        if exact_terminal:
            return _completed_result(context, session, True)
        if session.state != "PREPARED" or session.context_fingerprint != context["fingerprint"]:
            _fail(FAIL_FENCE_V1, "namespace session state/context differs")
        durable = context["durable"]["body"]
        online = context["online"]["body"]
        try:
            observation = _observe_process()
            if observation != dict(session.runtime_observation):
                _fail(FAIL_RUNTIME_V1, "local PID/start-time/UID-GID/user namespace drifted across Q1/Q2")
            binding = _validate_runtime_binding(
                _runtime_binding(context, observation), context, "Q2 local runtime binding"
            )
            if binding != dict(session.runtime_binding):
                _fail(FAIL_RUNTIME_V1, "local runtime binding drifted across Q1/Q2")
            replayed_mapping = _replay_candidate(
                *context["candidate_raw"], lock=context["lock"], profile=context["profile"],
                registry=context["registry"], role_keys=context["role_keys"],
                bootstrap=context["bootstrap"], attempt_id=context["attempt_id"],
            )
            if replayed_mapping != context["candidate_mapping"]:
                _fail(FAIL_BINDING_V1, "candidate raw replay drifted across Q1/Q2")
            q1 = _decode(session.q1_request_payload, "stored Q1 request")
            q1_response = _decode(session.q1_response_payload, "stored Q1 response")
            a_request = _decode(session.a_request_payload, "stored A request")
            prepare_nonce = q1["body"]["prepare_nonce"]
            confirmation_nonce = session.confirmation_nonce
            common = {
                "status": _ADMISSION_STATUS, "claim_level": _ADMISSION_CLAIM_LEVEL,
                "stage_b_trust_lock_id": context["lock"].artifact_id,
                "authority_profile_id": context["profile"].artifact_id,
                "signer_registry_id": context["registry"].artifact_id,
                "attempt_context_id": context["attempt_id"],
                "bootstrap_external_witness_policy_id": context["bootstrap"]["external_witness_policy_id"],
                "online_witness_policy_id": context["online"]["online_witness_policy_id"],
                "durable_anchor_policy_id": context["durable"]["durable_anchor_policy_id"],
            }
            a_checkpoint, checkpoint_body = _load_protocol(
                a_checkpoint_payload, label="A durable checkpoint", schema=_A_CHECKPOINT_SCHEMA,
                fields=frozenset(_A_CHECKPOINT_FIELDS), body_fields=frozenset(_A_CHECKPOINT_BODY_FIELDS),
                artifact_kind="a_durable_checkpoint", id_field="a_checkpoint_id",
                expected_key_id=durable["anchor_key_id"], expected_key_epoch=durable["anchor_key_epoch"],
                expected_purpose=_SIGN_A_CHECKPOINT, public_key_hex=durable["anchor_public_key_hex"],
            )
            _common(checkpoint_body, context, "A checkpoint")
            if (
                checkpoint_body["a_request_id"] != a_request["a_request_id"]
                or checkpoint_body["a_request_payload_sha256"] != _payload_sha(session.a_request_payload)
                or checkpoint_body["prepare_nonce"] != prepare_nonce
                or checkpoint_body["confirmation_nonce"] != confirmation_nonce
                or _mapping(checkpoint_body["previous_mapping"], "A previous mapping", context["attempt_id"])
                    != context["genesis_mapping"]
                or _mapping(checkpoint_body["committed_mapping"], "A committed mapping", context["attempt_id"])
                    != context["candidate_mapping"]
                or checkpoint_body["cas_outcome"] != _CAS_OUTCOME
                or type(checkpoint_body["checkpoint_sequence"]) is not int
                or checkpoint_body["checkpoint_sequence"] != 1
                or checkpoint_body["previous_a_checkpoint_id"] != durable["oob_high_water_checkpoint_id"]
            ):
                _fail(FAIL_BINDING_V1, "A checkpoint transition differs")
            a_head, head_body = _load_protocol(
                a_head_payload, label="A durable head", schema=_A_HEAD_SCHEMA,
                fields=frozenset(_A_HEAD_FIELDS), body_fields=frozenset(_A_HEAD_BODY_FIELDS),
                artifact_kind="a_durable_head", id_field="a_head_id",
                expected_key_id=durable["anchor_key_id"], expected_key_epoch=durable["anchor_key_epoch"],
                expected_purpose=_SIGN_A_HEAD, public_key_hex=durable["anchor_public_key_hex"],
            )
            _common(head_body, context, "A head")
            if (
                head_body["a_request_id"] != a_request["a_request_id"]
                or head_body["a_request_payload_sha256"] != _payload_sha(session.a_request_payload)
                or head_body["a_checkpoint_id"] != a_checkpoint["a_checkpoint_id"]
                or head_body["a_checkpoint_payload_sha256"] != _payload_sha(a_checkpoint_payload)
                or head_body["prepare_nonce"] != prepare_nonce
                or head_body["confirmation_nonce"] != confirmation_nonce
                or _mapping(head_body["committed_mapping"], "A head mapping", context["attempt_id"])
                    != context["candidate_mapping"]
                or head_body["cas_outcome"] != _CAS_OUTCOME
                or type(head_body["head_sequence"]) is not int or head_body["head_sequence"] != 1
                or head_body["previous_a_head_id"] != durable["oob_high_water_head_id"]
            ):
                _fail(FAIL_BINDING_V1, "A head transition differs")
            a_receipt, receipt_body = _load_protocol(
                a_receipt_payload, label="A compare-and-append receipt", schema=_A_RECEIPT_SCHEMA,
                fields=frozenset(_A_RECEIPT_FIELDS), body_fields=frozenset(_A_RECEIPT_BODY_FIELDS),
                artifact_kind="a_compare_and_append_receipt", id_field="a_receipt_id",
                expected_key_id=durable["anchor_key_id"], expected_key_epoch=durable["anchor_key_epoch"],
                expected_purpose=_SIGN_A_RECEIPT, public_key_hex=durable["anchor_public_key_hex"],
            )
            _common(receipt_body, context, "A receipt")
            if (
                receipt_body["a_request_id"] != a_request["a_request_id"]
                or receipt_body["a_request_payload_sha256"] != _payload_sha(session.a_request_payload)
                or receipt_body["a_checkpoint_id"] != a_checkpoint["a_checkpoint_id"]
                or receipt_body["a_checkpoint_payload_sha256"] != _payload_sha(a_checkpoint_payload)
                or receipt_body["a_head_id"] != a_head["a_head_id"]
                or receipt_body["a_head_payload_sha256"] != _payload_sha(a_head_payload)
                or receipt_body["prepare_nonce"] != prepare_nonce
                or receipt_body["confirmation_nonce"] != confirmation_nonce
                or _mapping(receipt_body["previous_mapping"], "A receipt previous mapping", context["attempt_id"])
                    != context["genesis_mapping"]
                or _mapping(receipt_body["committed_mapping"], "A receipt committed mapping", context["attempt_id"])
                    != context["candidate_mapping"]
                or receipt_body["cas_outcome"] != _CAS_OUTCOME
                or receipt_body["compare_matched"] is not True
                or receipt_body["append_applied"] is not True
                or receipt_body["oob_high_water_checkpoint_id"] != durable["oob_high_water_checkpoint_id"]
                or receipt_body["oob_high_water_head_id"] != durable["oob_high_water_head_id"]
                or receipt_body["durable_before_return"] is not True
            ):
                _fail(FAIL_BINDING_V1, "A receipt CAS binding differs")
            q2, q2_body = _load_protocol(
                q2_payload, label="Q2 CONFIRM request", schema=_Q2_REQUEST_SCHEMA,
                fields=frozenset(_Q2_REQUEST_FIELDS), body_fields=frozenset(_Q2_REQUEST_BODY_FIELDS),
                artifact_kind="q2_confirm_request", id_field="q2_request_id",
                expected_key_id=online["bootstrap_key_id"],
                expected_key_epoch=online["bootstrap_key_epoch"],
                expected_purpose=_SIGN_Q2_REQUEST, public_key_hex=online["bootstrap_public_key_hex"],
            )
            _common(q2_body, context, "Q2 request")
            if (
                q2_body["q1_response_id"] != q1_response["q1_response_id"]
                or q2_body["q1_response_payload_sha256"] != _payload_sha(session.q1_response_payload)
                or q2_body["a_request_id"] != a_request["a_request_id"]
                or q2_body["a_request_payload_sha256"] != _payload_sha(session.a_request_payload)
                or q2_body["a_receipt_id"] != a_receipt["a_receipt_id"]
                or q2_body["a_receipt_payload_sha256"] != _payload_sha(a_receipt_payload)
                or q2_body["a_checkpoint_id"] != a_checkpoint["a_checkpoint_id"]
                or q2_body["a_checkpoint_payload_sha256"] != _payload_sha(a_checkpoint_payload)
                or q2_body["a_head_id"] != a_head["a_head_id"]
                or q2_body["a_head_payload_sha256"] != _payload_sha(a_head_payload)
                or q2_body["prepare_nonce"] != prepare_nonce
                or q2_body["confirmation_nonce"] != confirmation_nonce
                or _mapping(q2_body["expected_current_mapping"], "Q2 expected mapping", context["attempt_id"])
                    != context["candidate_mapping"]
                or q2_body["expected_cas_outcome"] != _CAS_OUTCOME
            ):
                _fail(FAIL_BINDING_V1, "Q2 request A/current binding differs")
            response_body = {
                **common, "schema_version": _Q2_RESPONSE_SCHEMA,
                "q2_request_id": q2["q2_request_id"],
                "q2_request_payload_sha256": _payload_sha(q2_payload),
                "a_receipt_id": a_receipt["a_receipt_id"],
                "a_receipt_payload_sha256": _payload_sha(a_receipt_payload),
                "prepare_nonce": prepare_nonce, "confirmation_nonce": confirmation_nonce,
                "observed_current_mapping": context["candidate_mapping"],
                "runtime_custody_binding": binding,
                "observed_cas_outcome": _CAS_OUTCOME,
                "confirm_outcome": _CONFIRM_OUTCOME,
            }
            q2_response_payload = _sign_protocol(
                artifact_kind="q2_confirm_response", schema=_Q2_RESPONSE_SCHEMA,
                id_field="q2_response_id", body=response_body,
                private_key=context["private_key"], signer_key_id=online["online_key_id"],
                key_epoch=online["online_key_epoch"], purpose=_SIGN_Q2_RESPONSE,
            )
            _load_protocol(
                q2_response_payload, label="composed Q2 response", schema=_Q2_RESPONSE_SCHEMA,
                fields=frozenset(_Q2_RESPONSE_FIELDS), body_fields=frozenset(_Q2_RESPONSE_BODY_FIELDS),
                artifact_kind="q2_confirm_response", id_field="q2_response_id",
                expected_key_id=online["online_key_id"], expected_key_epoch=online["online_key_epoch"],
                expected_purpose=_SIGN_Q2_RESPONSE, public_key_hex=online["online_public_key_hex"],
            )
            completed = _FenceSessionV1(
                state="COMPLETE", context_fingerprint=session.context_fingerprint,
                q1_request_payload=session.q1_request_payload,
                q1_response_payload=session.q1_response_payload,
                a_request_payload=session.a_request_payload,
                confirmation_nonce=session.confirmation_nonce,
                runtime_observation=session.runtime_observation,
                runtime_binding=session.runtime_binding,
                a_checkpoint_payload=a_checkpoint_payload, a_head_payload=a_head_payload,
                a_receipt_payload=a_receipt_payload, q2_request_payload=q2_payload,
                q2_response_payload=q2_response_payload,
            )
            _FENCE_SESSIONS[namespace] = completed
            return _completed_result(context, completed, False)
        except Exception:
            _FENCE_SESSIONS[namespace] = _FenceSessionV1(
                state="POISONED", context_fingerprint=session.context_fingerprint,
                q1_request_payload=session.q1_request_payload,
                q1_response_payload=session.q1_response_payload,
                a_request_payload=session.a_request_payload,
                confirmation_nonce=session.confirmation_nonce,
                runtime_observation=session.runtime_observation,
                runtime_binding=session.runtime_binding,
            )
            raise


def verify_real_external_witness_admission_v1(*args: object, **kwargs: object) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "real admission requires durable restart-safe W/A services, live transport, custody, currentness, and full b480 logical ancestry verification",
    )


def _definition_captured_function_graph_v1() -> dict[str, object]:
    namespace = dict(globals())
    local_functions = {
        name: value for name, value in namespace.items()
        if type(value) is FunctionType and value.__module__ == __name__
    }
    private_namespace = dict(namespace)
    provisional: dict[str, object] = {}
    for name, function in local_functions.items():
        clone = FunctionType(function.__code__, private_namespace, function.__name__,
                             function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        provisional[name] = clone
    private_namespace.update(provisional)
    captured: dict[str, object] = {}
    for name, function in provisional.items():
        clone = FunctionType(function.__code__, private_namespace, function.__name__,
                             function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        captured[name] = clone
    private_namespace.update(captured)
    return captured


_CAPTURED = _definition_captured_function_graph_v1()


def _build_actor_type_v1(load_context, begin_prepare, confirm,
                         error_type, failure_code):
    states: dict[int, tuple[object, Mapping[str, object]]] = {}

    def reject(detail: str) -> NoReturn:
        raise error_type(failure_code, detail)

    def state(actor: object) -> Mapping[str, object]:
        record = states.get(id(actor))
        if (
            type(actor) is not R01OnlineWitnessActorV1
            or record is None
            or record[0] is not actor
        ):
            reject("online-witness actor lacks private state lineage")
        return record[1]

    class R01OnlineWitnessActorV1:
        """One fixed-key, raw-pinned, in-process W mechanics actor."""

        __slots__ = ()

        def __init_subclass__(cls, **kwargs: object) -> NoReturn:
            reject("online-witness actor cannot be subclassed")

        def __init__(
            self,
            stage_b_trust_lock_payload: bytes,
            authority_profile_payload: bytes,
            signer_registry_payload: bytes,
            bootstrap_external_witness_policy_payload: bytes,
            online_witness_policy_payload: bytes,
            durable_anchor_policy_payload: bytes,
            candidate_anchor_payload: bytes,
            candidate_receipt_payload: bytes,
            candidate_ledger_payload: bytes,
            candidate_checkpoint_payload: bytes,
            candidate_head_payload: bytes,
            *,
            out_of_band_expected_lock_id_bytes: bytes,
            out_of_band_expected_lock_payload_sha256_digest: bytes,
            out_of_band_expected_bootstrap_external_witness_policy_id_bytes: bytes,
            out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest: bytes,
            out_of_band_expected_online_witness_policy_id_bytes: bytes,
            out_of_band_expected_online_witness_policy_payload_sha256_digest: bytes,
            out_of_band_expected_durable_anchor_policy_id_bytes: bytes,
            out_of_band_expected_durable_anchor_policy_payload_sha256_digest: bytes,
            expected_attempt_context_id_bytes: bytes,
            online_witness_private_key: object,
        ) -> None:
            context = load_context(
                stage_b_trust_lock_payload,
                authority_profile_payload,
                signer_registry_payload,
                bootstrap_external_witness_policy_payload,
                online_witness_policy_payload,
                durable_anchor_policy_payload,
                candidate_anchor_payload,
                candidate_receipt_payload,
                candidate_ledger_payload,
                candidate_checkpoint_payload,
                candidate_head_payload,
                out_of_band_expected_lock_id_bytes=(
                    out_of_band_expected_lock_id_bytes
                ),
                out_of_band_expected_lock_payload_sha256_digest=(
                    out_of_band_expected_lock_payload_sha256_digest
                ),
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
                expected_attempt_context_id_bytes=expected_attempt_context_id_bytes,
                online_witness_private_key=online_witness_private_key,
            )
            if id(self) in states:
                reject("online-witness actor identity collision")
            states[id(self)] = (self, context)

        def begin_prepare(self, q1_prepare_request_payload: bytes) -> PreparedR01OnlineWitnessMechanicsV1:
            return begin_prepare(state(self), q1_prepare_request_payload)

        def confirm(self, a_durable_checkpoint_payload: bytes,
                    a_durable_head_payload: bytes,
                    a_compare_and_append_receipt_payload: bytes,
                    q2_confirm_request_payload: bytes) -> CompletedR01OnlineWitnessMechanicsV1:
            return confirm(state(self), a_durable_checkpoint_payload,
                           a_durable_head_payload, a_compare_and_append_receipt_payload,
                           q2_confirm_request_payload)

        @property
        def witness_namespace(self) -> str:
            return state(self)["online"]["body"]["witness_namespace"]

        @property
        def attempt_context_id(self) -> str:
            return state(self)["attempt_id"]

        def __copy__(self) -> NoReturn:
            reject("online-witness actor cannot be copied")

        def __deepcopy__(self, memo: object) -> NoReturn:
            reject("online-witness actor cannot be deep-copied")

        def __reduce__(self) -> NoReturn:
            reject("online-witness actor cannot be serialized")

    R01OnlineWitnessActorV1.__name__ = "R01OnlineWitnessActorV1"
    R01OnlineWitnessActorV1.__qualname__ = "R01OnlineWitnessActorV1"
    R01OnlineWitnessActorV1.__module__ = __name__
    return R01OnlineWitnessActorV1


R01OnlineWitnessActorV1 = _build_actor_type_v1(
    _CAPTURED["_load_context"], _CAPTURED["_begin_prepare"], _CAPTURED["_confirm"],
    Phase2BStep5R01OnlineWitnessErrorV1, FAIL_RESULT_V1,
)

require_prepared_r01_online_witness_mechanics_v1 = _CAPTURED[
    "require_prepared_r01_online_witness_mechanics_v1"
]
require_completed_r01_online_witness_mechanics_v1 = _CAPTURED[
    "require_completed_r01_online_witness_mechanics_v1"
]
verify_real_external_witness_admission_v1 = _CAPTURED[
    "verify_real_external_witness_admission_v1"
]

del _build_actor_type_v1
del _CAPTURED
del _definition_captured_function_graph_v1
del _load_context
del _begin_prepare
del _confirm
del _issue_prepared
del _issue_completed
del _FENCE_LOCK
del _FENCE_SESSIONS


__all__ = [
    "ADMISSION_CONTRACT_SOURCE_SHA256_V1",
    "EXTERNAL_WITNESS_SOURCE_SHA256_V1",
    "EVIDENCE_CONTRACT_SOURCE_SHA256_V1",
    "ONLINE_WITNESS_MECHANICS_RESULT_SCHEMA_V1",
    "ONLINE_WITNESS_MECHANICS_CLAIM_BOUNDARY_V1",
    "FAIL_INPUT_V1", "FAIL_CANONICAL_V1", "FAIL_SCHEMA_V1",
    "FAIL_CONTENT_ID_V1", "FAIL_TRUST_ANCHOR_V1", "FAIL_SIGNATURE_V1",
    "FAIL_KEY_V1", "FAIL_BINDING_V1", "FAIL_RUNTIME_V1", "FAIL_FENCE_V1",
    "FAIL_POISONED_V1", "FAIL_RESULT_V1", "FAIL_NOT_IMPLEMENTED_V1",
    "Phase2BStep5R01OnlineWitnessErrorV1",
    "PreparedR01OnlineWitnessMechanicsV1",
    "CompletedR01OnlineWitnessMechanicsV1",
    "R01OnlineWitnessActorV1",
    "require_prepared_r01_online_witness_mechanics_v1",
    "require_completed_r01_online_witness_mechanics_v1",
    "verify_real_external_witness_admission_v1",
]

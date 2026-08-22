"""Pure bytes-only logical-time replay core for Phase-2B Step 5.

This module deliberately does not import the logical-time service module.  It
independently re-freezes the logical-time request/token/checkpoint/head ``/1``
wire schemas and replays a complete caller-supplied chain from canonical bytes.
The only project dependencies are the pure authority/profile V2 verifier and
the RFC-8785 implementation.  Ed25519 verification uses ``cryptography``
directly.

Successful replay establishes all of the following, and only the following:

* the raw Stage-B lock, authority profile, and signer registry close under the
  two caller-supplied out-of-band Stage-B pins;
* the registry's ``LOGICAL_TIME`` role key, exact singleton signing purpose,
  and static logical-time service binding are internally consistent;
* every request, token, checkpoint, and head is canonical and content-addressed
  according to the re-frozen ``/1`` domains;
* every token has a valid Ed25519 signature, exactly projects its independently
  supplied request, and belongs to the exact caller-supplied event plan;
* the supplied heads form one contiguous genesis-to-final chain, and the last
  supplied head exactly commits the last supplied checkpoint.

The last point is *supplied-chain finality*, not external currentness.  This
core has no wall clock, trusted UTC source, filesystem, CAS client, anti-fork
anchor, runtime attestation, service issuer, private key, or readiness issuer.
Its private result therefore keeps every external-currentness, UTC, CAS,
runtime, authority, and readiness predicate false.

The public verifier captures its complete call graph at import time.  The
private result lineage retains strong object identities and exact immutable
field snapshots, detecting ``object.__new__`` forgeries and post-issuance
``object.__setattr__`` mutation.  These are Python caller/KAT properties, not
claims of resistance to arbitrary closure-cell, process-memory, interpreter,
or loaded-code mutation; those remain outside this verifier-only core.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    Phase2BStep5AuthorityErrorV2 as _AuthorityErrorV2,
    load_step5_authority_profile_v2 as _load_authority_profile_v2,
    load_step5_signer_registry_v2 as _load_signer_registry_v2,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_trust_lock_v2,
    require_locked_step5_authority_graph_v2 as _require_authority_graph_v2,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical_rfc8785_v1,
)

try:  # Importability is preserved; verification still fails closed.
    from cryptography.exceptions import InvalidSignature as _InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - depends on the pinned verifier image.
    _InvalidSignature = None
    _Ed25519PublicKey = None


SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-logical-time-replay-core/1"
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

LOGICAL_TIME_ROLE_V1: Final = "LOGICAL_TIME"
LOGICAL_TIME_CONTROL_DOMAIN_V1: Final = "STEP5_LOGICAL_TIME_CONTROL"
LOGICAL_TIME_SERVICE_NAME_V2: Final = "logical_time"
SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1: Final = "SIGN_LOGICAL_TIME_TOKEN"
LOGICAL_TIME_KEY_EPOCH_V1: Final = 1
ED25519_ALGORITHM_V1: Final = "Ed25519"
LEDGER_HEAD_PURPOSE_V1: Final = "STEP5_LOGICAL_TIME_APPEND_ONLY_LEDGER_HEAD"
LEDGER_CHECKPOINT_PURPOSE_V1: Final = (
    "STEP5_LOGICAL_TIME_APPEND_ONLY_LEDGER_CHECKPOINT"
)

GENESIS_TOKEN_ID_V1: Final = TOKEN_ID_PREFIX_V1 + ("0" * 64)
GENESIS_CHECKPOINT_ID_V1: Final = LEDGER_CHECKPOINT_ID_PREFIX_V1 + ("0" * 64)

AUTHORITY_CLAIM_BOUNDARY_V1: Final = (
    "VERIFIER_ONLY_CANONICAL_SIGNATURE_AND_SUPPLIED_CHAIN_MECHANICS_NOT_"
    "EXTERNAL_CURRENTNESS_NOT_TRUSTED_UTC_NOT_CAS_NOT_RUNTIME_ATTESTATION_"
    "NOT_AUTHORITY_NOT_READINESS_NOT_EXECUTION_AUTHORIZATION"
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

FAIL_INPUT_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_SCHEMA"
FAIL_AUTHORITY_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_AUTHORITY"
FAIL_BINDING_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_BINDING"
FAIL_CONTENT_ID_V1: Final = (
    "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_CONTENT_ID"
)
FAIL_SIGNATURE_BACKEND_V1: Final = (
    "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_SIGNATURE_BACKEND"
)
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_SIGNATURE"
FAIL_CHAIN_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_CHAIN"
FAIL_RESULT_V1: Final = "FAIL_STEP5_LOGICAL_TIME_REPLAY_CORE_V1_RESULT"


class Phase2BStep5LogicalTimeReplayCoreErrorV1(ValueError):
    """Stable fail-closed error for the verifier-only replay core."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5LogicalTimeReplayCoreErrorV1(code, detail)


def _canonical(
    value: object,
    _canonicalize=_canonicalize_rfc8785_v1,
    _rfc_error_type=_RFC8785Error,
    _error_type=Phase2BStep5LogicalTimeReplayCoreErrorV1,
    _fail_code: str = FAIL_CANONICAL_V1,
) -> bytes:
    try:
        return _canonicalize(value)  # type: ignore[arg-type]
    except (_rfc_error_type, TypeError, ValueError, RecursionError) as exc:
        raise _error_type(_fail_code, str(exc)) from exc


def _restore_integral_numbers(
    value: object,
    _fail_exact=_fail,
    _maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
) -> object:
    # Keep recursion inside this invocation so rebinding the module helper after
    # import cannot replace nested-number validation.
    def restore(item: object) -> object:
        if type(item) is float:
            if not item.is_integer() or not abs(item) <= _maximum:
                _fail_exact(
                    FAIL_SCHEMA_V1,
                    "logical-time /1 wire schemas admit exact safe integers only",
                )
            return int(item)
        if type(item) is list:
            return [restore(child) for child in item]
        if type(item) is dict:
            return {key: restore(child) for key, child in item.items()}
        return item

    return restore(value)


def _decode(
    payload: bytes,
    maximum_bytes: int,
    label: str,
    _decode_exact=_decode_canonical_rfc8785_v1,
    _restore=_restore_integral_numbers,
    _canonical_exact=_canonical,
    _fail_exact=_fail,
    _rfc_error_type=_RFC8785Error,
    _error_type=Phase2BStep5LogicalTimeReplayCoreErrorV1,
) -> dict[str, object]:
    if type(payload) is not bytes:
        _fail_exact(FAIL_INPUT_V1, f"{label} must be exact bytes")
    if not 1 <= len(payload) <= maximum_bytes or payload.endswith(b"\n"):
        _fail_exact(
            FAIL_INPUT_V1, f"{label} is outside its exact byte boundary"
        )
    try:
        decoded = _decode_exact(payload)
    except (_rfc_error_type, TypeError, ValueError, RecursionError) as exc:
        raise _error_type(
            FAIL_CANONICAL_V1, f"{label}: {exc}"
        ) from exc
    restored = _restore(decoded)
    if type(restored) is not dict or _canonical_exact(restored) != payload:
        _fail_exact(FAIL_CANONICAL_V1, f"{label} canonical readback differs")
    return restored


def _exact_object(
    value: object,
    fields: frozenset[str],
    label: str,
    _fail_exact=_fail,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(fields):
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exact field set differs")
    return value


def _identifier(
    value: object,
    label: str,
    _pattern=_IDENTIFIER_PATTERN,
    _maximum_bytes: int = _MAXIMUM_IDENTIFIER_BYTES,
    _fail_exact=_fail,
    _error_type=Phase2BStep5LogicalTimeReplayCoreErrorV1,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be one bounded identifier")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _error_type(
            FAIL_SCHEMA_V1, f"{label} must be exact ASCII"
        ) from exc
    if len(encoded) > _maximum_bytes:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exceeds its ASCII byte bound")
    return value


def _prefixed_id(
    value: object,
    prefix: str,
    label: str,
    _identifier_exact=_identifier,
    _digest_pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
) -> str:
    result = _identifier_exact(value, label)
    suffix = result[len(prefix) :] if result.startswith(prefix) else ""
    if not result.startswith(prefix) or _digest_pattern.fullmatch(suffix) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} prefix or digest differs")
    return result


def _sha256_address(
    value: object,
    label: str,
    _pattern=_SHA256_PATTERN,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be sha256:<lowerhex64>")
    return value


def _nonce(
    value: object,
    label: str,
    _pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(
            FAIL_SCHEMA_V1,
            f"{label} must be exact lowercase 32-byte hexadecimal",
        )
    return value


def _purpose(
    value: object,
    label: str,
    _pattern=_PURPOSE_PATTERN,
    _maximum_bytes: int = _MAXIMUM_PURPOSE_BYTES,
    _fail_exact=_fail,
) -> str:
    if type(value) is not str or _pattern.fullmatch(value) is None:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} must be one uppercase token")
    if len(value.encode("ascii")) > _maximum_bytes:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exceeds its ASCII byte bound")
    return value


def _wire_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _MAXIMUM_SAFE_JSON_INTEGER,
    _fail_exact=_fail,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail_exact(FAIL_SCHEMA_V1, f"{label} exact integer differs")
    return value


def _sha256_bytes(
    payload: bytes,
    _sha256=hashlib.sha256,
    _fail_exact=_fail,
) -> str:
    if type(payload) is not bytes:
        _fail_exact(FAIL_INPUT_V1, "SHA-256 input must be exact bytes")
    return "sha256:" + _sha256(payload).hexdigest()


def _content_id(
    prefix: str,
    domain: bytes,
    body: object,
    _canonical_exact=_canonical,
    _sha256=hashlib.sha256,
) -> str:
    return prefix + _sha256(domain + _canonical_exact(body)).hexdigest()


def _common_tuple(
    value: Mapping[str, object],
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_identifier=_identifier,
    _authority_prefix: str = AUTHORITY_PROFILE_PREFIX_V2,
    _registry_prefix: str = SIGNER_REGISTRY_PREFIX_V2,
    _role_key_prefix: str = ROLE_KEY_PREFIX_V1,
    _required_epoch: int = LOGICAL_TIME_KEY_EPOCH_V1,
    _fail_exact=_fail,
) -> tuple[str, str, str, int, str]:
    profile_id = _validate_prefixed_id(
        value["authority_profile_id"],
        _authority_prefix,
        "authority profile ID",
    )
    registry_id = _validate_prefixed_id(
        value["signer_registry_id"],
        _registry_prefix,
        "signer registry ID",
    )
    key_id = _validate_prefixed_id(
        value["logical_time_key_id"],
        _role_key_prefix,
        "logical-time role-key ID",
    )
    epoch = _validate_integer(
        value["logical_time_key_epoch"],
        "logical-time key epoch",
        minimum=1,
    )
    if epoch != _required_epoch:
        _fail_exact(
            FAIL_BINDING_V1, "logical-time /1 requires exact key epoch 1"
        )
    attempt_context_id = _validate_identifier(
        value["attempt_context_id"], "Step-5 attempt context ID"
    )
    return profile_id, registry_id, key_id, epoch, attempt_context_id


def _require_expected_common(
    observed: tuple[str, str, str, int, str],
    expected: tuple[str, str, str, int, str],
    label: str,
    _fail_exact=_fail,
) -> None:
    if type(observed) is not tuple or observed != expected:
        _fail_exact(FAIL_BINDING_V1, f"{label} authority/key/attempt pins differ")


def _request_id_subject(
    body: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "logical_time_key_id": body["logical_time_key_id"],
        "logical_time_key_epoch": body["logical_time_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "request_nonce": body["request_nonce"],
        "signature_purpose": body["signature_purpose"],
        "purpose": body["purpose"],
        "bound_artifact_id": body["bound_artifact_id"],
        "bound_artifact_sha256": body["bound_artifact_sha256"],
        "prior_token_id": body["prior_token_id"],
        "next_sequence": body["next_sequence"],
    }


def _validate_request_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common=_require_expected_common,
    _validate_nonce=_nonce,
    _validate_purpose=_purpose,
    _validate_identifier=_identifier,
    _validate_sha256=_sha256_address,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _id_subject=_request_id_subject,
    _content_address=_content_id,
    _request_fields: frozenset[str] = _REQUEST_KEYS,
    _schema: str = LOGICAL_TIME_REQUEST_SCHEMA_VERSION,
    _signature_purpose: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
    _token_prefix: str = TOKEN_ID_PREFIX_V1,
    _request_prefix: str = REQUEST_ID_PREFIX_V1,
    _request_domain: bytes = _REQUEST_ID_DOMAIN,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _request_fields, "logical-time request")
    if value["schema_version"] != _schema:
        _fail_exact(FAIL_SCHEMA_V1, "logical-time request schema differs")
    _require_common(
        _common_projection(value), expected_common, "logical-time request"
    )
    _validate_nonce(value["request_nonce"], "request nonce")
    if value["signature_purpose"] != _signature_purpose:
        _fail_exact(FAIL_BINDING_V1, "request signature purpose differs")
    _validate_purpose(value["purpose"], "request event purpose")
    _validate_identifier(value["bound_artifact_id"], "request artifact ID")
    _validate_sha256(
        value["bound_artifact_sha256"], "request artifact SHA-256"
    )
    _validate_prefixed_id(
        value["prior_token_id"], _token_prefix, "request prior token ID"
    )
    _validate_integer(value["next_sequence"], "request next sequence")
    expected_id = _content_address(
        _request_prefix, _request_domain, _id_subject(value)
    )
    if value["request_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "logical-time request ID differs")
    return value


def _parse_request(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    _decode_exact=_decode,
    _validate_mapping=_validate_request_mapping,
    _maximum: int = MAXIMUM_REQUEST_BYTES,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, _maximum, "logical-time request"),
        expected_common,
    )


def _token_unsigned_body(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "logical_time_key_id": body["logical_time_key_id"],
        "logical_time_key_epoch": body["logical_time_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "request_id": body["request_id"],
        "request_nonce": body["request_nonce"],
        "signature_purpose": body["signature_purpose"],
        "purpose": body["purpose"],
        "bound_artifact_id": body["bound_artifact_id"],
        "bound_artifact_sha256": body["bound_artifact_sha256"],
        "prior_token_id": body["prior_token_id"],
        "sequence": body["sequence"],
        "algorithm": body["algorithm"],
    }


def _token_id_subject(
    body: Mapping[str, object],
    _unsigned=_token_unsigned_body,
) -> dict[str, object]:
    return {**_unsigned(body), "signature_hex": body["signature_hex"]}


def _validate_token_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common=_require_expected_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_nonce=_nonce,
    _validate_purpose=_purpose,
    _validate_identifier=_identifier,
    _validate_sha256=_sha256_address,
    _validate_integer=_wire_integer,
    _id_subject=_token_id_subject,
    _content_address=_content_id,
    _token_fields: frozenset[str] = _TOKEN_KEYS,
    _schema: str = SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION,
    _signature_purpose: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
    _algorithm: str = ED25519_ALGORITHM_V1,
    _request_prefix: str = REQUEST_ID_PREFIX_V1,
    _token_prefix: str = TOKEN_ID_PREFIX_V1,
    _signature_pattern=_LOWER_HEX_64_PATTERN,
    _token_domain: bytes = _TOKEN_ID_DOMAIN,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _token_fields, "signed logical-time token")
    if value["schema_version"] != _schema:
        _fail_exact(FAIL_SCHEMA_V1, "signed logical-time token schema differs")
    _require_common(
        _common_projection(value), expected_common, "signed logical-time token"
    )
    _validate_prefixed_id(
        value["request_id"], _request_prefix, "token request ID"
    )
    _validate_nonce(value["request_nonce"], "token request nonce")
    if value["signature_purpose"] != _signature_purpose:
        _fail_exact(FAIL_BINDING_V1, "token signature purpose differs")
    _validate_purpose(value["purpose"], "token event purpose")
    _validate_identifier(value["bound_artifact_id"], "token artifact ID")
    _validate_sha256(value["bound_artifact_sha256"], "token artifact SHA-256")
    _validate_prefixed_id(
        value["prior_token_id"], _token_prefix, "token prior token ID"
    )
    _validate_integer(value["sequence"], "logical-time token sequence")
    if value["algorithm"] != _algorithm:
        _fail_exact(FAIL_SCHEMA_V1, "logical-time token algorithm differs")
    if (
        type(value["signature_hex"]) is not str
        or _signature_pattern.fullmatch(value["signature_hex"]) is None
    ):
        _fail_exact(FAIL_SCHEMA_V1, "token signature encoding differs")
    expected_id = _content_address(
        _token_prefix, _token_domain, _id_subject(value)
    )
    if value["token_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "logical-time token ID differs")
    return value


def _parse_token(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    _decode_exact=_decode,
    _validate_mapping=_validate_token_mapping,
    _maximum: int = MAXIMUM_TOKEN_BYTES,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, _maximum, "signed logical-time token"),
        expected_common,
    )


def _request_projection_from_token(
    token: Mapping[str, object],
    _schema: str = LOGICAL_TIME_REQUEST_SCHEMA_VERSION,
) -> dict[str, object]:
    return {
        "schema_version": _schema,
        "authority_profile_id": token["authority_profile_id"],
        "signer_registry_id": token["signer_registry_id"],
        "logical_time_key_id": token["logical_time_key_id"],
        "logical_time_key_epoch": token["logical_time_key_epoch"],
        "attempt_context_id": token["attempt_context_id"],
        "request_nonce": token["request_nonce"],
        "signature_purpose": token["signature_purpose"],
        "purpose": token["purpose"],
        "bound_artifact_id": token["bound_artifact_id"],
        "bound_artifact_sha256": token["bound_artifact_sha256"],
        "prior_token_id": token["prior_token_id"],
        "next_sequence": token["sequence"],
        "request_id": token["request_id"],
    }


def _token_signature_preimage(
    token: Mapping[str, object],
    _unsigned=_token_unsigned_body,
    _canonical_exact=_canonical,
    _domain: bytes = _TOKEN_SIGNATURE_DOMAIN,
    _role: str = LOGICAL_TIME_ROLE_V1,
    _purpose_exact: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
) -> bytes:
    wrapper = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
        "artifact_kind": "logical_time",
        "role": _role,
        "purpose": _purpose_exact,
        "key_epoch": token["logical_time_key_epoch"],
        "signer_key_id": token["logical_time_key_id"],
        "body": _unsigned(token),
    }
    return _domain + _canonical_exact(wrapper)


def _derive_logical_time_role_key_id(
    public_key: bytes,
    *,
    key_epoch: int,
    _validate_integer=_wire_integer,
    _content_address=_content_id,
    _role_key_prefix: str = ROLE_KEY_PREFIX_V1,
    _role_key_domain: bytes = _ROLE_KEY_ID_DOMAIN,
    _role: str = LOGICAL_TIME_ROLE_V1,
    _control_domain: str = LOGICAL_TIME_CONTROL_DOMAIN_V1,
    _signature_purpose: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
    _algorithm: str = ED25519_ALGORITHM_V1,
    _required_epoch: int = LOGICAL_TIME_KEY_EPOCH_V1,
    _fail_exact=_fail,
) -> str:
    if type(public_key) is not bytes or len(public_key) != 32:
        _fail_exact(FAIL_AUTHORITY_V1, "logical-time public key is not raw 32 bytes")
    epoch = _validate_integer(key_epoch, "logical-time role key epoch", minimum=1)
    if epoch != _required_epoch:
        _fail_exact(FAIL_AUTHORITY_V1, "logical-time role key epoch differs")
    subject = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": _role,
        "control_domain": _control_domain,
        "authorized_signature_purposes": [_signature_purpose],
        "algorithm": _algorithm,
        "key_epoch": epoch,
        "public_key_hex": public_key.hex(),
    }
    return _content_address(_role_key_prefix, _role_key_domain, subject)


def _verify_token_signature(
    token: Mapping[str, object],
    public_key: bytes,
    *,
    _derive_key_id=_derive_logical_time_role_key_id,
    _preimage=_token_signature_preimage,
    _public_key_type=_Ed25519PublicKey,
    _invalid_signature_type=_InvalidSignature,
    _fail_exact=_fail,
    _error_type=Phase2BStep5LogicalTimeReplayCoreErrorV1,
) -> None:
    derived_key_id = _derive_key_id(
        public_key, key_epoch=token["logical_time_key_epoch"]  # type: ignore[arg-type]
    )
    if token["logical_time_key_id"] != derived_key_id:
        _fail_exact(FAIL_BINDING_V1, "token role-key content address differs")
    if _public_key_type is None or _invalid_signature_type is None:
        _fail_exact(
            FAIL_SIGNATURE_BACKEND_V1,
            "cryptography Ed25519 verification backend is unavailable",
        )
    try:
        verifier = _public_key_type.from_public_bytes(public_key)
        verifier.verify(
            bytes.fromhex(token["signature_hex"]),  # type: ignore[arg-type]
            _preimage(token),
        )
    except _invalid_signature_type as exc:
        raise _error_type(
            FAIL_SIGNATURE_V1, "logical-time Ed25519 signature rejected"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _error_type(
            FAIL_SIGNATURE_V1, "logical-time Ed25519 public key rejected"
        ) from exc


def _head_id_subject(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "purpose": body["purpose"],
        "ledger_id": body["ledger_id"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "logical_time_key_id": body["logical_time_key_id"],
        "logical_time_key_epoch": body["logical_time_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "checkpoint_count": body["checkpoint_count"],
        "next_sequence": body["next_sequence"],
        "last_token_id": body["last_token_id"],
        "last_checkpoint_id": body["last_checkpoint_id"],
    }


def _validate_head_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common=_require_expected_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _id_subject=_head_id_subject,
    _content_address=_content_id,
    _head_fields: frozenset[str] = _HEAD_KEYS,
    _schema: str = LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION,
    _purpose_exact: str = LEDGER_HEAD_PURPOSE_V1,
    _ledger_prefix: str = LEDGER_ID_PREFIX_V1,
    _token_prefix: str = TOKEN_ID_PREFIX_V1,
    _checkpoint_prefix: str = LEDGER_CHECKPOINT_ID_PREFIX_V1,
    _head_prefix: str = LEDGER_HEAD_ID_PREFIX_V1,
    _head_domain: bytes = _LEDGER_HEAD_ID_DOMAIN,
    _genesis_token: str = GENESIS_TOKEN_ID_V1,
    _genesis_checkpoint: str = GENESIS_CHECKPOINT_ID_V1,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _head_fields, "logical-time ledger head")
    if value["schema_version"] != _schema or value["purpose"] != _purpose_exact:
        _fail_exact(FAIL_SCHEMA_V1, "logical-time head schema or purpose differs")
    _validate_prefixed_id(value["ledger_id"], _ledger_prefix, "head ledger ID")
    _require_common(_common_projection(value), expected_common, "ledger head")
    count = _validate_integer(value["checkpoint_count"], "head checkpoint count")
    next_sequence = _validate_integer(value["next_sequence"], "head next sequence")
    if count != next_sequence:
        _fail_exact(FAIL_CHAIN_V1, "head count and next sequence differ")
    _validate_prefixed_id(value["last_token_id"], _token_prefix, "head token ID")
    _validate_prefixed_id(
        value["last_checkpoint_id"], _checkpoint_prefix, "head checkpoint ID"
    )
    if count == 0 and (
        value["last_token_id"] != _genesis_token
        or value["last_checkpoint_id"] != _genesis_checkpoint
    ):
        _fail_exact(FAIL_CHAIN_V1, "genesis head sentinel linkage differs")
    expected_id = _content_address(_head_prefix, _head_domain, _id_subject(value))
    if value["head_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "logical-time head ID differs")
    return value


def _parse_head(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    _decode_exact=_decode,
    _validate_mapping=_validate_head_mapping,
    _maximum: int = MAXIMUM_HEAD_BYTES,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, _maximum, "logical-time ledger head"),
        expected_common,
    )


def _checkpoint_id_subject(body: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": body["schema_version"],
        "purpose": body["purpose"],
        "ledger_id": body["ledger_id"],
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "logical_time_key_id": body["logical_time_key_id"],
        "logical_time_key_epoch": body["logical_time_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "checkpoint_index": body["checkpoint_index"],
        "sequence": body["sequence"],
        "previous_head_id": body["previous_head_id"],
        "previous_checkpoint_id": body["previous_checkpoint_id"],
        "request_id": body["request_id"],
        "request_nonce": body["request_nonce"],
        "token_id": body["token_id"],
        "signed_token_sha256": body["signed_token_sha256"],
        "signed_token": body["signed_token"],
    }


def _validate_checkpoint_mapping(
    body: object,
    expected_common: tuple[str, str, str, int, str],
    *,
    _exact_mapping=_exact_object,
    _common_projection=_common_tuple,
    _require_common=_require_expected_common,
    _validate_prefixed_id=_prefixed_id,
    _validate_integer=_wire_integer,
    _validate_nonce=_nonce,
    _validate_sha256=_sha256_address,
    _validate_token=_validate_token_mapping,
    _canonical_exact=_canonical,
    _sha256_exact=_sha256_bytes,
    _id_subject=_checkpoint_id_subject,
    _content_address=_content_id,
    _checkpoint_fields: frozenset[str] = _CHECKPOINT_KEYS,
    _schema: str = LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION,
    _purpose_exact: str = LEDGER_CHECKPOINT_PURPOSE_V1,
    _ledger_prefix: str = LEDGER_ID_PREFIX_V1,
    _head_prefix: str = LEDGER_HEAD_ID_PREFIX_V1,
    _checkpoint_prefix: str = LEDGER_CHECKPOINT_ID_PREFIX_V1,
    _request_prefix: str = REQUEST_ID_PREFIX_V1,
    _token_prefix: str = TOKEN_ID_PREFIX_V1,
    _checkpoint_domain: bytes = _LEDGER_CHECKPOINT_ID_DOMAIN,
    _fail_exact=_fail,
) -> dict[str, object]:
    value = _exact_mapping(body, _checkpoint_fields, "logical-time checkpoint")
    if value["schema_version"] != _schema or value["purpose"] != _purpose_exact:
        _fail_exact(
            FAIL_SCHEMA_V1, "logical-time checkpoint schema or purpose differs"
        )
    _validate_prefixed_id(
        value["ledger_id"], _ledger_prefix, "checkpoint ledger ID"
    )
    _require_common(
        _common_projection(value), expected_common, "logical-time checkpoint"
    )
    index = _validate_integer(value["checkpoint_index"], "checkpoint index")
    sequence = _validate_integer(value["sequence"], "checkpoint sequence")
    if index != sequence:
        _fail_exact(FAIL_CHAIN_V1, "checkpoint index and sequence differ")
    _validate_prefixed_id(
        value["previous_head_id"], _head_prefix, "checkpoint previous head ID"
    )
    _validate_prefixed_id(
        value["previous_checkpoint_id"],
        _checkpoint_prefix,
        "checkpoint previous checkpoint ID",
    )
    _validate_prefixed_id(
        value["request_id"], _request_prefix, "checkpoint request ID"
    )
    _validate_nonce(value["request_nonce"], "checkpoint request nonce")
    _validate_prefixed_id(value["token_id"], _token_prefix, "checkpoint token ID")
    _validate_sha256(value["signed_token_sha256"], "checkpoint token SHA-256")
    embedded_token = _validate_token(value["signed_token"], expected_common)
    embedded_token_bytes = _canonical_exact(embedded_token)
    if (
        embedded_token["token_id"] != value["token_id"]
        or embedded_token["request_id"] != value["request_id"]
        or embedded_token["request_nonce"] != value["request_nonce"]
        or embedded_token["sequence"] != value["sequence"]
        or _sha256_exact(embedded_token_bytes) != value["signed_token_sha256"]
    ):
        _fail_exact(FAIL_BINDING_V1, "checkpoint signed-token linkage differs")
    expected_id = _content_address(
        _checkpoint_prefix, _checkpoint_domain, _id_subject(value)
    )
    if value["checkpoint_id"] != expected_id:
        _fail_exact(FAIL_CONTENT_ID_V1, "logical-time checkpoint ID differs")
    return value


def _parse_checkpoint(
    payload: bytes,
    expected_common: tuple[str, str, str, int, str],
    _decode_exact=_decode,
    _validate_mapping=_validate_checkpoint_mapping,
    _maximum: int = MAXIMUM_CHECKPOINT_BYTES,
) -> dict[str, object]:
    return _validate_mapping(
        _decode_exact(payload, _maximum, "logical-time ledger checkpoint"),
        expected_common,
    )


def _ledger_id(
    common: tuple[str, str, str, int, str],
    _content_address=_content_id,
    _prefix: str = LEDGER_ID_PREFIX_V1,
    _domain: bytes = _LEDGER_ID_DOMAIN,
) -> str:
    subject = {
        "schema_version": (
            "hegel-machine-phase2b-step5-logical-time-ledger-identity/1"
        ),
        "authority_profile_id": common[0],
        "signer_registry_id": common[1],
        "logical_time_key_id": common[2],
        "logical_time_key_epoch": common[3],
        "attempt_context_id": common[4],
    }
    return _content_address(_prefix, _domain, subject)


def _expected_head_mapping(
    *,
    common: tuple[str, str, str, int, str],
    ledger_id: str,
    checkpoint_count: int,
    last_token_id: str,
    last_checkpoint_id: str,
    _content_address=_content_id,
    _schema: str = LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION,
    _purpose_exact: str = LEDGER_HEAD_PURPOSE_V1,
    _prefix: str = LEDGER_HEAD_ID_PREFIX_V1,
    _domain: bytes = _LEDGER_HEAD_ID_DOMAIN,
) -> dict[str, object]:
    subject = {
        "schema_version": _schema,
        "purpose": _purpose_exact,
        "ledger_id": ledger_id,
        "authority_profile_id": common[0],
        "signer_registry_id": common[1],
        "logical_time_key_id": common[2],
        "logical_time_key_epoch": common[3],
        "attempt_context_id": common[4],
        "checkpoint_count": checkpoint_count,
        "next_sequence": checkpoint_count,
        "last_token_id": last_token_id,
        "last_checkpoint_id": last_checkpoint_id,
    }
    return {
        **subject,
        "head_id": _content_address(_prefix, _domain, subject),
    }


def _exact_mapping_equal(
    left: Mapping[str, object],
    right: Mapping[str, object],
    _canonical_exact=_canonical,
) -> bool:
    return _canonical_exact(left) == _canonical_exact(right)


@dataclass(frozen=True, slots=True)
class ExpectedLogicalEventCoreV1:
    """One exact event-plan row; tuple order fixes the logical sequence."""

    purpose: str
    bound_artifact_id: str
    bound_artifact_sha256: str
    request_nonce: str

    def __post_init__(
        self,
        _validate_purpose=_purpose,
        _validate_identifier=_identifier,
        _validate_sha256=_sha256_address,
        _validate_nonce=_nonce,
    ) -> None:
        _validate_purpose(self.purpose, "expected event purpose")
        _validate_identifier(self.bound_artifact_id, "expected artifact ID")
        _validate_sha256(
            self.bound_artifact_sha256, "expected artifact SHA-256"
        )
        _validate_nonce(self.request_nonce, "expected request nonce")


def _event_row(
    value: object,
    _event_type=ExpectedLogicalEventCoreV1,
    _validate_purpose=_purpose,
    _validate_identifier=_identifier,
    _validate_sha256=_sha256_address,
    _validate_nonce=_nonce,
    _fail_exact=_fail,
) -> tuple[str, str, str, str]:
    if type(value) is not _event_type:
        _fail_exact(FAIL_INPUT_V1, "expected event row must use the exact DTO")
    return (
        _validate_purpose(value.purpose, "expected event purpose"),
        _validate_identifier(value.bound_artifact_id, "expected artifact ID"),
        _validate_sha256(
            value.bound_artifact_sha256, "expected artifact SHA-256"
        ),
        _validate_nonce(value.request_nonce, "expected request nonce"),
    )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedLogicalTimeReplayCoreV1:
    """Privately issued replay mechanics result; never authority evidence."""

    schema_version: str
    stage_b_trust_lock_id: str
    authority_profile_id: str
    signer_registry_id: str
    logical_time_key_id: str
    logical_time_key_epoch: int
    attempt_context_id: str
    ledger_id: str
    verified_event_count: int
    verified_request_count: int
    verified_signature_count: int
    verified_checkpoint_count: int
    verified_head_count: int
    verified_event_plan: tuple[tuple[str, str, str, str], ...]
    request_ids: tuple[str, ...]
    token_ids: tuple[str, ...]
    checkpoint_ids: tuple[str, ...]
    head_ids: tuple[str, ...]
    final_head_id: str
    final_head_payload_sha256: str
    authority_graph_signature_backend: str
    supplied_chain_final_head_verified: bool
    zero_event_genesis_only: bool
    verifier_only: bool
    private_key_present: bool
    issuer_path_present: bool
    filesystem_observed: bool
    service_observed: bool
    external_currentness_verified: bool
    external_anti_fork_verified: bool
    trusted_utc_verified: bool
    external_cas_checkpoint_verified: bool
    runtime_closure_verified: bool
    authority_evidence: bool
    readiness_evidence: bool
    r01_satisfied: bool
    execution_authorized: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "VerifiedLogicalTimeReplayCoreV1":
        raise TypeError("verified logical-time replay results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("verified logical-time replay results are privately issued")


def _build_verified_result_lineage_v1():
    """Close issuance and exact mutation snapshots over one module instance."""

    result_type = VerifiedLogicalTimeReplayCoreV1
    fail_exact = _fail
    validate_prefixed_id = _prefixed_id
    validate_identifier = _identifier
    validate_integer = _wire_integer
    validate_sha256 = _sha256_address
    schema = SCHEMA_VERSION
    lock_prefix = "phase2b_step5_stage_b_trust_lock_v2_"
    profile_prefix = AUTHORITY_PROFILE_PREFIX_V2
    registry_prefix = SIGNER_REGISTRY_PREFIX_V2
    key_prefix = ROLE_KEY_PREFIX_V1
    ledger_prefix = LEDGER_ID_PREFIX_V1
    request_prefix = REQUEST_ID_PREFIX_V1
    token_prefix = TOKEN_ID_PREFIX_V1
    checkpoint_prefix = LEDGER_CHECKPOINT_ID_PREFIX_V1
    head_prefix = LEDGER_HEAD_ID_PREFIX_V1
    backend = "CRYPTOGRAPHY_ED25519"
    issued: dict[
        int,
        tuple[
            VerifiedLogicalTimeReplayCoreV1,
            tuple[object, ...],
            tuple[tuple[str, str, str, str], ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ],
    ] = {}

    scalar_fields = (
        "schema_version",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "ledger_id",
        "verified_event_count",
        "verified_request_count",
        "verified_signature_count",
        "verified_checkpoint_count",
        "verified_head_count",
        "final_head_id",
        "final_head_payload_sha256",
        "authority_graph_signature_backend",
        "supplied_chain_final_head_verified",
        "zero_event_genesis_only",
        "verifier_only",
        "private_key_present",
        "issuer_path_present",
        "filesystem_observed",
        "service_observed",
        "external_currentness_verified",
        "external_anti_fork_verified",
        "trusted_utc_verified",
        "external_cas_checkpoint_verified",
        "runtime_closure_verified",
        "authority_evidence",
        "readiness_evidence",
        "r01_satisfied",
        "execution_authorized",
    )

    def require(value: object) -> VerifiedLogicalTimeReplayCoreV1:
        if type(value) is not result_type:
            fail_exact(FAIL_RESULT_V1, "verified replay result exact type differs")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            fail_exact(
                FAIL_RESULT_V1,
                "verified replay result lacks private issuance lineage",
            )
        (
            _,
            expected_scalars,
            expected_plan,
            expected_requests,
            expected_tokens,
            expected_checkpoints,
            expected_heads,
        ) = record
        try:
            observed_scalars = tuple(
                object.__getattribute__(value, name) for name in scalar_fields
            )
            intact = (
                observed_scalars == expected_scalars
                and value.verified_event_plan is expected_plan
                and value.request_ids is expected_requests
                and value.token_ids is expected_tokens
                and value.checkpoint_ids is expected_checkpoints
                and value.head_ids is expected_heads
            )
        except (AttributeError, TypeError):
            intact = False
        if not intact:
            fail_exact(FAIL_RESULT_V1, "verified replay result was mutated")

        count = validate_integer(value.verified_event_count, "verified event count")
        if not (
            value.schema_version == schema
            and validate_prefixed_id(
                value.stage_b_trust_lock_id, lock_prefix, "Stage-B lock ID"
            )
            == value.stage_b_trust_lock_id
            and validate_prefixed_id(
                value.authority_profile_id, profile_prefix, "authority profile ID"
            )
            == value.authority_profile_id
            and validate_prefixed_id(
                value.signer_registry_id, registry_prefix, "signer registry ID"
            )
            == value.signer_registry_id
            and validate_prefixed_id(
                value.logical_time_key_id, key_prefix, "logical-time key ID"
            )
            == value.logical_time_key_id
            and value.logical_time_key_epoch == 1
            and validate_identifier(
                value.attempt_context_id, "attempt context ID"
            )
            == value.attempt_context_id
            and validate_prefixed_id(value.ledger_id, ledger_prefix, "ledger ID")
            == value.ledger_id
            and value.verified_request_count == count
            and value.verified_signature_count == count
            and value.verified_checkpoint_count == count
            and value.verified_head_count == count + 1
            and type(value.verified_event_plan) is tuple
            and len(value.verified_event_plan) == count
            and type(value.request_ids) is tuple
            and len(value.request_ids) == count
            and type(value.token_ids) is tuple
            and len(value.token_ids) == count
            and type(value.checkpoint_ids) is tuple
            and len(value.checkpoint_ids) == count
            and type(value.head_ids) is tuple
            and len(value.head_ids) == count + 1
            and value.final_head_id == value.head_ids[-1]
            and validate_sha256(
                value.final_head_payload_sha256, "final head payload SHA-256"
            )
            == value.final_head_payload_sha256
            and value.authority_graph_signature_backend == backend
            and value.supplied_chain_final_head_verified is True
            and value.zero_event_genesis_only is (count == 0)
            and value.verifier_only is True
            and value.private_key_present is False
            and value.issuer_path_present is False
            and value.filesystem_observed is False
            and value.service_observed is False
            and value.external_currentness_verified is False
            and value.external_anti_fork_verified is False
            and value.trusted_utc_verified is False
            and value.external_cas_checkpoint_verified is False
            and value.runtime_closure_verified is False
            and value.authority_evidence is False
            and value.readiness_evidence is False
            and value.r01_satisfied is False
            and value.execution_authorized is False
        ):
            fail_exact(FAIL_RESULT_V1, "verified replay result semantics differ")
        for item in value.request_ids:
            validate_prefixed_id(item, request_prefix, "verified request ID")
        for item in value.token_ids:
            validate_prefixed_id(item, token_prefix, "verified token ID")
        for item in value.checkpoint_ids:
            validate_prefixed_id(item, checkpoint_prefix, "verified checkpoint ID")
        for item in value.head_ids:
            validate_prefixed_id(item, head_prefix, "verified head ID")
        return value

    def issue(
        *,
        stage_b_trust_lock_id: str,
        common: tuple[str, str, str, int, str],
        ledger_id: str,
        event_plan: tuple[tuple[str, str, str, str], ...],
        request_ids: tuple[str, ...],
        token_ids: tuple[str, ...],
        checkpoint_ids: tuple[str, ...],
        head_ids: tuple[str, ...],
        final_head_payload_sha256: str,
    ) -> VerifiedLogicalTimeReplayCoreV1:
        if (
            type(common) is not tuple
            or len(common) != 5
            or type(event_plan) is not tuple
            or type(request_ids) is not tuple
            or type(token_ids) is not tuple
            or type(checkpoint_ids) is not tuple
            or type(head_ids) is not tuple
        ):
            fail_exact(FAIL_RESULT_V1, "private replay issuance shape differs")
        count = len(event_plan)
        if not (
            len(request_ids) == len(token_ids) == len(checkpoint_ids) == count
            and len(head_ids) == count + 1
        ):
            fail_exact(FAIL_RESULT_V1, "private replay issuance counts differ")
        result = object.__new__(result_type)
        values: dict[str, object] = {
            "schema_version": schema,
            "stage_b_trust_lock_id": stage_b_trust_lock_id,
            "authority_profile_id": common[0],
            "signer_registry_id": common[1],
            "logical_time_key_id": common[2],
            "logical_time_key_epoch": common[3],
            "attempt_context_id": common[4],
            "ledger_id": ledger_id,
            "verified_event_count": count,
            "verified_request_count": count,
            "verified_signature_count": count,
            "verified_checkpoint_count": count,
            "verified_head_count": count + 1,
            "verified_event_plan": event_plan,
            "request_ids": request_ids,
            "token_ids": token_ids,
            "checkpoint_ids": checkpoint_ids,
            "head_ids": head_ids,
            "final_head_id": head_ids[-1],
            "final_head_payload_sha256": final_head_payload_sha256,
            "authority_graph_signature_backend": backend,
            "supplied_chain_final_head_verified": True,
            "zero_event_genesis_only": count == 0,
            "verifier_only": True,
            "private_key_present": False,
            "issuer_path_present": False,
            "filesystem_observed": False,
            "service_observed": False,
            "external_currentness_verified": False,
            "external_anti_fork_verified": False,
            "trusted_utc_verified": False,
            "external_cas_checkpoint_verified": False,
            "runtime_closure_verified": False,
            "authority_evidence": False,
            "readiness_evidence": False,
            "r01_satisfied": False,
            "execution_authorized": False,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        scalar_snapshot = tuple(values[name] for name in scalar_fields)
        issued[id(result)] = (
            result,
            scalar_snapshot,
            event_plan,
            request_ids,
            token_ids,
            checkpoint_ids,
            head_ids,
        )
        return require(result)

    return issue, require


(
    _hidden_issue_verified_logical_time_replay_core_v1,
    require_verified_logical_time_replay_core_v1,
) = _build_verified_result_lineage_v1()
del _build_verified_result_lineage_v1


def _extract_logical_time_authority(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    _load_lock=_load_stage_b_trust_lock_v2,
    _load_profile=_load_authority_profile_v2,
    _load_registry=_load_signer_registry_v2,
    _require_graph=_require_authority_graph_v2,
    _derive_key_id=_derive_logical_time_role_key_id,
    _canonical_exact=_canonical,
    _role: str = LOGICAL_TIME_ROLE_V1,
    _control_domain: str = LOGICAL_TIME_CONTROL_DOMAIN_V1,
    _service_name: str = LOGICAL_TIME_SERVICE_NAME_V2,
    _signature_purpose: str = SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
    _algorithm: str = ED25519_ALGORITHM_V1,
    _required_epoch: int = LOGICAL_TIME_KEY_EPOCH_V1,
    _public_key_pattern=_LOWER_HEX_32_PATTERN,
    _fail_exact=_fail,
    _authority_error_type=_AuthorityErrorV2,
    _error_type=Phase2BStep5LogicalTimeReplayCoreErrorV1,
) -> tuple[str, str, str, int, bytes]:
    """Verify the raw V2 authority graph and extract its logical-time key."""

    if (
        type(stage_b_trust_lock_payload) is not bytes
        or type(authority_profile_payload) is not bytes
        or type(signer_registry_payload) is not bytes
        or type(out_of_band_expected_lock_id_bytes) is not bytes
        or type(out_of_band_expected_payload_sha256_digest) is not bytes
    ):
        _fail_exact(FAIL_INPUT_V1, "authority graph and OOB pins must be exact bytes")
    try:
        lock = _load_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
        )
        profile = _load_profile(authority_profile_payload, lock)
        registry = _load_registry(
            signer_registry_payload,
            lock,
            profile,
        )
        lock, profile, registry = _require_graph(lock, profile, registry)
    except _authority_error_type as exc:
        raise _error_type(
            FAIL_AUTHORITY_V1, f"authority/profile V2 graph rejected: {exc}"
        ) from exc

    if registry.signature_backends != ("CRYPTOGRAPHY_ED25519",):
        _fail_exact(FAIL_AUTHORITY_V1, "registry signature backend lineage differs")
    rows = registry.body["role_keys_in_exact_order"]
    if type(rows) is not tuple or len(rows) != 8:
        _fail_exact(FAIL_AUTHORITY_V1, "authority registry is not exact-eight-role")
    logical_rows = tuple(row for row in rows if row["role"] == _role)
    if len(logical_rows) != 1:
        _fail_exact(FAIL_AUTHORITY_V1, "registry LOGICAL_TIME row cardinality differs")
    key = logical_rows[0]
    if not (
        key["schema_version"] == "hegel-machine-phase2b-step5-role-key/1"
        and key["role"] == _role
        and key["control_domain"] == _control_domain
        and key["authorized_signature_purposes"] == (_signature_purpose,)
        and type(key["authorized_signature_purposes"]) is tuple
        and key["algorithm"] == _algorithm
        and type(key["key_epoch"]) is int
        and key["key_epoch"] == _required_epoch
        and type(key["public_key_hex"]) is str
        and _public_key_pattern.fullmatch(key["public_key_hex"]) is not None
        and type(key["key_id"]) is str
    ):
        _fail_exact(FAIL_AUTHORITY_V1, "registry LOGICAL_TIME policy differs")
    public_key = bytes.fromhex(key["public_key_hex"])
    expected_key_id = _derive_key_id(public_key, key_epoch=key["key_epoch"])
    if key["key_id"] != expected_key_id:
        _fail_exact(FAIL_AUTHORITY_V1, "registry LOGICAL_TIME key ID differs")

    services = profile.body["service_static_bindings_in_exact_order"]
    if type(services) is not tuple or len(services) != 5:
        _fail_exact(FAIL_AUTHORITY_V1, "profile static-service cardinality differs")
    logical_services = tuple(
        service for service in services if service["service_name"] == _service_name
    )
    if len(logical_services) != 1:
        _fail_exact(FAIL_AUTHORITY_V1, "logical-time static service cardinality differs")
    service = logical_services[0]
    if not (
        service["role"] == _role
        and service["control_domain"] == _control_domain
        and service["role_key_id"] == expected_key_id
        and service["key_epoch"] == _required_epoch
        and type(service["authorized_signature_purposes"]) is tuple
        and service["authorized_signature_purposes"] == (_signature_purpose,)
    ):
        _fail_exact(FAIL_AUTHORITY_V1, "logical-time service/key purpose link differs")

    # Re-canonicalizing these immutable projections makes accidental extended
    # Python values fail here even though authority V2 already validated them.
    _canonical_exact(
        {
            "role": key["role"],
            "control_domain": key["control_domain"],
            "authorized_signature_purposes": list(
                key["authorized_signature_purposes"]
            ),
            "algorithm": key["algorithm"],
            "key_epoch": key["key_epoch"],
            "public_key_hex": key["public_key_hex"],
            "key_id": key["key_id"],
        }
    )
    return (
        profile.artifact_id,
        registry.artifact_id,
        expected_key_id,
        _required_epoch,
        public_key,
    )


def _verify_logical_time_replay_core_implementation_v1(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    request_payloads_in_exact_order: tuple[bytes, ...],
    signed_token_payloads_in_exact_order: tuple[bytes, ...],
    checkpoint_payloads_in_exact_order: tuple[bytes, ...],
    head_payloads_in_exact_order: tuple[bytes, ...],
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_payload_sha256_digest: bytes,
    expected_attempt_context_id: str,
    expected_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
    _extract_authority=_extract_logical_time_authority,
    _validate_identifier=_identifier,
    _event_projection=_event_row,
    _event_type=ExpectedLogicalEventCoreV1,
    _parse_request_exact=_parse_request,
    _parse_token_exact=_parse_token,
    _parse_checkpoint_exact=_parse_checkpoint,
    _parse_head_exact=_parse_head,
    _request_projection=_request_projection_from_token,
    _verify_signature_exact=_verify_token_signature,
    _ledger_identity=_ledger_id,
    _expected_head=_expected_head_mapping,
    _mapping_equal=_exact_mapping_equal,
    _canonical_exact=_canonical,
    _sha256_exact=_sha256_bytes,
    _issue_result=_hidden_issue_verified_logical_time_replay_core_v1,
    _genesis_token: str = GENESIS_TOKEN_ID_V1,
    _genesis_checkpoint: str = GENESIS_CHECKPOINT_ID_V1,
    _fail_exact=_fail,
) -> VerifiedLogicalTimeReplayCoreV1:
    """Verify one complete raw authority-bound logical-time chain."""

    tuple_inputs = (
        (
            request_payloads_in_exact_order,
            "request payloads",
        ),
        (
            signed_token_payloads_in_exact_order,
            "signed-token payloads",
        ),
        (
            checkpoint_payloads_in_exact_order,
            "checkpoint payloads",
        ),
        (head_payloads_in_exact_order, "head payloads"),
    )
    for value, label in tuple_inputs:
        if type(value) is not tuple:
            _fail_exact(FAIL_INPUT_V1, f"{label} must be an exact tuple")
        if any(type(item) is not bytes for item in value):
            _fail_exact(FAIL_INPUT_V1, f"every {label} item must be exact bytes")
    if type(expected_event_plan) is not tuple:
        _fail_exact(FAIL_INPUT_V1, "expected event plan must be an exact tuple")
    if any(type(item) is not _event_type for item in expected_event_plan):
        _fail_exact(FAIL_INPUT_V1, "every expected event must use the exact DTO")
    attempt_context_id = _validate_identifier(
        expected_attempt_context_id, "expected Step-5 attempt context ID"
    )
    count = len(expected_event_plan)
    if not (
        len(request_payloads_in_exact_order) == count
        and len(signed_token_payloads_in_exact_order) == count
        and len(checkpoint_payloads_in_exact_order) == count
        and len(head_payloads_in_exact_order) == count + 1
    ):
        _fail_exact(
            FAIL_CHAIN_V1,
            "plan/request/token/checkpoint/head cardinalities differ",
        )

    (
        authority_profile_id,
        signer_registry_id,
        logical_time_key_id,
        logical_time_key_epoch,
        logical_time_public_key,
    ) = _extract_authority(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=out_of_band_expected_lock_id_bytes,
        out_of_band_expected_payload_sha256_digest=(
            out_of_band_expected_payload_sha256_digest
        ),
    )
    expected_common = (
        authority_profile_id,
        signer_registry_id,
        logical_time_key_id,
        logical_time_key_epoch,
        attempt_context_id,
    )
    expected_ledger_id = _ledger_identity(expected_common)

    plan_rows: list[tuple[str, str, str, str]] = []
    plan_nonces: set[str] = set()
    plan_event_bindings: set[tuple[str, str, str]] = set()
    for event in expected_event_plan:
        row = _event_projection(event)
        binding = (row[0], row[1], row[2])
        if row in plan_rows or row[3] in plan_nonces or binding in plan_event_bindings:
            _fail_exact(FAIL_CHAIN_V1, "expected event plan contains a duplicate")
        plan_rows.append(row)
        plan_nonces.add(row[3])
        plan_event_bindings.add(binding)

    heads = tuple(
        _parse_head_exact(payload, expected_common)
        for payload in head_payloads_in_exact_order
    )
    expected_genesis = _expected_head(
        common=expected_common,
        ledger_id=expected_ledger_id,
        checkpoint_count=0,
        last_token_id=_genesis_token,
        last_checkpoint_id=_genesis_checkpoint,
    )
    if not _mapping_equal(heads[0], expected_genesis):
        _fail_exact(FAIL_CHAIN_V1, "supplied genesis head differs")

    request_ids: list[str] = []
    token_ids: list[str] = []
    checkpoint_ids: list[str] = []
    head_ids: list[str] = [heads[0]["head_id"]]  # type: ignore[list-item]
    seen_request_ids: set[str] = set()
    seen_nonces: set[str] = set()
    seen_token_ids: set[str] = set()
    seen_checkpoint_ids: set[str] = set()
    seen_head_ids: set[str] = {heads[0]["head_id"]}  # type: ignore[arg-type]

    previous_head = heads[0]
    for index in range(count):
        request_payload = request_payloads_in_exact_order[index]
        token_payload = signed_token_payloads_in_exact_order[index]
        checkpoint_payload = checkpoint_payloads_in_exact_order[index]
        request = _parse_request_exact(request_payload, expected_common)
        token = _parse_token_exact(token_payload, expected_common)
        checkpoint = _parse_checkpoint_exact(checkpoint_payload, expected_common)
        next_head = heads[index + 1]
        expected_event = plan_rows[index]

        if not _mapping_equal(request, _request_projection(token)):
            _fail_exact(
                FAIL_BINDING_V1,
                "raw signed token does not exact-project raw request",
            )
        _verify_signature_exact(token, logical_time_public_key)
        if not _mapping_equal(checkpoint["signed_token"], token):  # type: ignore[arg-type]
            _fail_exact(
                FAIL_BINDING_V1,
                "checkpoint embedded token differs from raw token payload",
            )
        if _canonical_exact(checkpoint["signed_token"]) != token_payload:
            _fail_exact(
                FAIL_BINDING_V1,
                "checkpoint embedded token bytes differ from raw token payload",
            )
        observed_event = (
            request["purpose"],
            request["bound_artifact_id"],
            request["bound_artifact_sha256"],
            request["request_nonce"],
        )
        if observed_event != expected_event:
            _fail_exact(
                FAIL_BINDING_V1,
                "raw request/token event differs from exact event plan",
            )
        if not (
            request["next_sequence"] == index
            and token["sequence"] == index
            and checkpoint["checkpoint_index"] == index
            and checkpoint["sequence"] == index
            and request["prior_token_id"] == previous_head["last_token_id"]
            and token["prior_token_id"] == previous_head["last_token_id"]
            and checkpoint["ledger_id"] == expected_ledger_id
            and checkpoint["previous_head_id"] == previous_head["head_id"]
            and checkpoint["previous_checkpoint_id"]
            == previous_head["last_checkpoint_id"]
            and checkpoint["request_id"] == request["request_id"]
            and checkpoint["request_nonce"] == request["request_nonce"]
            and checkpoint["token_id"] == token["token_id"]
            and checkpoint["signed_token_sha256"] == _sha256_exact(token_payload)
        ):
            _fail_exact(FAIL_CHAIN_V1, "logical-time transition linkage differs")

        expected_next_head = _expected_head(
            common=expected_common,
            ledger_id=expected_ledger_id,
            checkpoint_count=index + 1,
            last_token_id=token["token_id"],  # type: ignore[arg-type]
            last_checkpoint_id=checkpoint["checkpoint_id"],  # type: ignore[arg-type]
        )
        if not _mapping_equal(next_head, expected_next_head):
            _fail_exact(
                FAIL_CHAIN_V1,
                "next head does not exact-commit the contiguous checkpoint",
            )

        request_id = request["request_id"]
        request_nonce = request["request_nonce"]
        token_id = token["token_id"]
        checkpoint_id = checkpoint["checkpoint_id"]
        next_head_id = next_head["head_id"]
        if (
            request_id in seen_request_ids
            or request_nonce in seen_nonces
            or token_id in seen_token_ids
            or checkpoint_id in seen_checkpoint_ids
            or next_head_id in seen_head_ids
        ):
            _fail_exact(FAIL_CHAIN_V1, "duplicate or forked chain identity detected")
        seen_request_ids.add(request_id)  # type: ignore[arg-type]
        seen_nonces.add(request_nonce)  # type: ignore[arg-type]
        seen_token_ids.add(token_id)  # type: ignore[arg-type]
        seen_checkpoint_ids.add(checkpoint_id)  # type: ignore[arg-type]
        seen_head_ids.add(next_head_id)  # type: ignore[arg-type]
        request_ids.append(request_id)  # type: ignore[arg-type]
        token_ids.append(token_id)  # type: ignore[arg-type]
        checkpoint_ids.append(checkpoint_id)  # type: ignore[arg-type]
        head_ids.append(next_head_id)  # type: ignore[arg-type]
        previous_head = next_head

    final_head = heads[-1]
    if not (
        final_head["checkpoint_count"] == count
        and final_head["next_sequence"] == count
        and len(request_ids) == len(token_ids) == len(checkpoint_ids) == count
        and len(head_ids) == count + 1
        and head_ids[-1] == final_head["head_id"]
    ):
        _fail_exact(FAIL_CHAIN_V1, "supplied final head count or linkage differs")
    if count == 0 and not (
        request_payloads_in_exact_order == ()
        and signed_token_payloads_in_exact_order == ()
        and checkpoint_payloads_in_exact_order == ()
        and expected_event_plan == ()
        and head_payloads_in_exact_order == (head_payloads_in_exact_order[0],)
    ):
        _fail_exact(FAIL_CHAIN_V1, "zero-event replay is not exact genesis-only")

    return _issue_result(
        stage_b_trust_lock_id=out_of_band_expected_lock_id_bytes.decode("ascii"),
        common=expected_common,
        ledger_id=expected_ledger_id,
        event_plan=tuple(plan_rows),
        request_ids=tuple(request_ids),
        token_ids=tuple(token_ids),
        checkpoint_ids=tuple(checkpoint_ids),
        head_ids=tuple(head_ids),
        final_head_payload_sha256=_sha256_exact(head_payloads_in_exact_order[-1]),
    )


def _build_public_replay_entry_v1(implementation):
    """Expose only the captured verifier; no decoded-object callback surface."""

    def verify_logical_time_replay_core_v1(
        stage_b_trust_lock_payload: bytes,
        authority_profile_payload: bytes,
        signer_registry_payload: bytes,
        request_payloads_in_exact_order: tuple[bytes, ...],
        signed_token_payloads_in_exact_order: tuple[bytes, ...],
        checkpoint_payloads_in_exact_order: tuple[bytes, ...],
        head_payloads_in_exact_order: tuple[bytes, ...],
        *,
        out_of_band_expected_lock_id_bytes: bytes,
        out_of_band_expected_payload_sha256_digest: bytes,
        expected_attempt_context_id: str,
        expected_event_plan: tuple[ExpectedLogicalEventCoreV1, ...],
    ) -> VerifiedLogicalTimeReplayCoreV1:
        return implementation(
            stage_b_trust_lock_payload,
            authority_profile_payload,
            signer_registry_payload,
            request_payloads_in_exact_order,
            signed_token_payloads_in_exact_order,
            checkpoint_payloads_in_exact_order,
            head_payloads_in_exact_order,
            out_of_band_expected_lock_id_bytes=(
                out_of_band_expected_lock_id_bytes
            ),
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_payload_sha256_digest
            ),
            expected_attempt_context_id=expected_attempt_context_id,
            expected_event_plan=expected_event_plan,
        )

    return verify_logical_time_replay_core_v1


verify_logical_time_replay_core_v1 = _build_public_replay_entry_v1(
    _verify_logical_time_replay_core_implementation_v1
)
del _build_public_replay_entry_v1
del _verify_logical_time_replay_core_implementation_v1
del _hidden_issue_verified_logical_time_replay_core_v1


def verify_external_currentness_v1(*args: object, **kwargs: object) -> NoReturn:
    """Fail closed: supplied-chain finality is not external currentness."""

    _fail(
        FAIL_CHAIN_V1,
        "external currentness and anti-fork anchoring are outside this core",
    )


__all__ = [
    "AUTHORITY_CLAIM_BOUNDARY_V1",
    "ED25519_ALGORITHM_V1",
    "ExpectedLogicalEventCoreV1",
    "FAIL_AUTHORITY_V1",
    "FAIL_BINDING_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_CHAIN_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_INPUT_V1",
    "FAIL_RESULT_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_SIGNATURE_BACKEND_V1",
    "FAIL_SIGNATURE_V1",
    "GENESIS_CHECKPOINT_ID_V1",
    "GENESIS_TOKEN_ID_V1",
    "LEDGER_CHECKPOINT_ID_PREFIX_V1",
    "LEDGER_CHECKPOINT_PURPOSE_V1",
    "LEDGER_HEAD_ID_PREFIX_V1",
    "LEDGER_HEAD_PURPOSE_V1",
    "LOGICAL_TIME_CONTROL_DOMAIN_V1",
    "LOGICAL_TIME_KEY_EPOCH_V1",
    "LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION",
    "LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION",
    "LOGICAL_TIME_REQUEST_SCHEMA_VERSION",
    "LOGICAL_TIME_ROLE_V1",
    "Phase2BStep5LogicalTimeReplayCoreErrorV1",
    "REQUEST_ID_PREFIX_V1",
    "ROLE_KEY_PREFIX_V1",
    "SCHEMA_VERSION",
    "SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION",
    "SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1",
    "TOKEN_ID_PREFIX_V1",
    "VerifiedLogicalTimeReplayCoreV1",
    "require_verified_logical_time_replay_core_v1",
    "verify_external_currentness_v1",
    "verify_logical_time_replay_core_v1",
]

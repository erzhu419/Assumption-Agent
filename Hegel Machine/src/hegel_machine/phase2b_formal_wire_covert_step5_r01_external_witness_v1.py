"""Mechanics-only initial external-witness spool for prospective Step-5 R01.

The sole accepting production entry consumes raw canonical authority-v2,
external-policy, and PURE-signed anti-fork-anchor bytes.  It reloads the
authority graph from the raw Stage-B lock/profile/registry under independent
out-of-band lock pins, checks separately supplied expected authority and
attempt IDs, verifies the PURE Ed25519 signature without a callback, checks a
fixed Ed25519 private key against the raw-pinned external policy, and signs the
exact evidence-contract ``EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT /1`` preimage.

Only the initial one-leaf Merkle case is implemented.  An existing absolute,
owner-private directory receives a fixed initialization lock followed by exact
receipt, ledger, checkpoint, and head files.  Every file is created with
``O_NOFOLLOW|O_EXCL``, fsynced, and reopened for exact bounded readback; the
directory is fsynced after every publication.  A crash can leave an immutable
partial spool which intentionally blocks reuse rather than being deleted.

These local mechanics do not establish a distinct UID, independent/local
filesystem status, host rollback resistance, key custody, external control,
external currentness, authority, R01, readiness, or execution authorization.
There is no listener, systemd unit, process observation, or live admission
path in this module.  Real external-witness admission remains NOT_IMPLEMENTED.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import re
import stat
from types import FunctionType, MappingProxyType
from typing import Final, Mapping, NoReturn

from .phase2b_formal_wire_covert_step5_authority_v2 import (
    AUTHORITY_PROFILE_PREFIX_V2 as _AUTHORITY_PROFILE_PREFIX,
    DETACHED_SIGNATURE_FIELDS_V2 as _AUTH_SIGNATURE_FIELDS,
    DETACHED_SIGNATURE_SCHEMA_V2 as _AUTH_SIGNATURE_SCHEMA,
    ROLE_ORDER_V2 as _AUTH_ROLE_ORDER,
    SIGNER_REGISTRY_PREFIX_V2 as _SIGNER_REGISTRY_PREFIX,
    STAGE_B_TRUST_LOCK_PREFIX_V2 as _STAGE_B_TRUST_LOCK_PREFIX,
    Phase2BStep5AuthorityErrorV2 as _AuthorityError,
    load_step5_authority_profile_v2 as _load_authority_profile,
    load_step5_signer_registry_v2 as _load_signer_registry,
    load_step5_stage_b_trust_lock_v2 as _load_stage_b_lock,
    phase2b_step5_signature_preimage_v2 as _authority_signature_preimage,
    require_locked_step5_authority_graph_v2 as _require_authority_graph,
)
from .phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 import (
    CANDIDATE_CLAIM_LEVEL_V1 as _CANDIDATE_CLAIM_LEVEL,
    CANDIDATE_STATUS_V1 as _CANDIDATE_STATUS,
    EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_FIELDS_V1 as _EXTERNAL_RECEIPT_FIELDS,
    EXTERNAL_OWNER_STATUS_ANCHOR_RECEIPT_SCHEMA_V1 as _EXTERNAL_RECEIPT_SCHEMA,
    EXTERNAL_WITNESS_POLICY_FIELDS_V1 as _EXTERNAL_POLICY_FIELDS,
    EXTERNAL_WITNESS_POLICY_SCHEMA_V1 as _EXTERNAL_POLICY_SCHEMA,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_BODY_FIELDS_V1 as _ANCHOR_BODY_FIELDS,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_FIELDS_V1 as _ANCHOR_FIELDS,
    SIGNED_OWNER_STATUS_ANTI_FORK_ANCHOR_SCHEMA_V1 as _ANCHOR_SCHEMA,
    phase2b_step5_external_anchor_receipt_signature_preimage_v1 as _external_signature_preimage,
    phase2b_step5_r01_evidence_content_id_v1 as _evidence_content_id,
)
from .rfc8785_v1 import (
    RFC8785Error as _RFC8785Error,
    canonicalize_rfc8785_v1 as _canonicalize,
    decode_and_require_canonical_rfc8785_v1 as _decode_canonical,
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


EVIDENCE_CONTRACT_SOURCE_SHA256_V1: Final = (
    "574e07e7ce0f19075c29fa59cc94e4e988a34c301e9e09005a8aed805d69e96d"
)
EXTERNAL_WITNESS_LEDGER_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-ledger/1"
)
EXTERNAL_WITNESS_CHECKPOINT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-checkpoint/1"
)
EXTERNAL_WITNESS_HEAD_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-external-witness-head/1"
)
EXTERNAL_WITNESS_INITIALIZATION_LOCK_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-formal-wire-covert-step5-"
    "external-witness-initialization-lock/1"
)

MECHANICS_CLAIM_BOUNDARY_V1: Final = (
    "INITIAL_ONE_LEAF_EXTERNAL_WITNESS_SIGNATURE_AND_EXCLUSIVE_LOCAL_SPOOL_"
    "MECHANICS_ONLY_NOT_DISTINCT_UID_NOT_INDEPENDENT_OR_LOCAL_FILESYSTEM_"
    "NOT_HOST_ROLLBACK_RESISTANCE_NOT_KEY_CUSTODY_NOT_EXTERNAL_CONTROL_"
    "NOT_EXTERNAL_CURRENTNESS_NOT_AUTHORITY_NOT_R01_NOT_READINESS_NOT_EXECUTION"
)

FAIL_INPUT_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_INPUT"
FAIL_CANONICAL_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_CANONICAL"
FAIL_SCHEMA_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_SCHEMA"
FAIL_CONTENT_ID_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_CONTENT_ID"
FAIL_BINDING_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_BINDING"
FAIL_TRUST_ANCHOR_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_TRUST_ANCHOR"
FAIL_SIGNATURE_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_SIGNATURE"
FAIL_KEY_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_KEY"
FAIL_FILESYSTEM_V1: Final = "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_FILESYSTEM"
FAIL_CLAIM_BOUNDARY_V1: Final = (
    "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_CLAIM_BOUNDARY"
)
FAIL_NOT_IMPLEMENTED_V1: Final = (
    "FAIL_STEP5_R01_EXTERNAL_WITNESS_V1_NOT_IMPLEMENTED"
)

_MAXIMUM_CANONICAL_BYTES: Final = 8 * 1024 * 1024
_MAXIMUM_SPOOL_RECORD_BYTES: Final = 16 * 1024 * 1024
_MAXIMUM_SAFE_INTEGER: Final = (1 << 53) - 1
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX128 = re.compile(r"[0-9a-f]{128}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_./:-]{0,255}_[0-9a-f]{64}")

_EXTERNAL_POLICY_PREFIX: Final = "phase2b_step5_external_witness_policy_v1_"
_EXTERNAL_POLICY_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
)
_EXTERNAL_POLICY_DOMAIN_SHA256_KAT: Final = (
    "8b8710e61858548d939526a99a12ccacd3c671657a08f65dd131a8ce414142de"
)
_MERKLE_LEAF_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/MERKLE_LEAF/V1\0"
)
_MERKLE_LEAF_DOMAIN_SHA256_KAT: Final = (
    "6fd2d607ce8b6e94d35d24e7606d1af84c09488eb460b6f3a200faadc3ece3a6"
)
_LEDGER_PREFIX: Final = "phase2b_step5_external_witness_ledger_v1_"
_LEDGER_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/LEDGER/V1\0"
)
_CHECKPOINT_PREFIX: Final = "phase2b_step5_external_witness_checkpoint_v1_"
_CHECKPOINT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/CHECKPOINT/V1\0"
)
_HEAD_PREFIX: Final = "phase2b_step5_external_witness_head_v1_"
_HEAD_DOMAIN: Final = b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS/HEAD/V1\0"

_CURRENT_CUT_STATEMENT_PREFIX: Final = "phase2b_step5_current_cut_statement_v1_"
_LOGICAL_TIME_TOKEN_PREFIX: Final = "phase2b_step5_logical_time_token_v1_"
_LOGICAL_TIME_CHECKPOINT_PREFIX: Final = (
    "phase2b_step5_logical_time_checkpoint_v1_"
)
_LOGICAL_TIME_HEAD_PREFIX: Final = "phase2b_step5_logical_time_head_v1_"
_OWNER_LEDGER_PREFIX: Final = "phase2b_step5_owner_status_ledger_v1_"
_OWNER_HEAD_PREFIX: Final = "phase2b_step5_owner_status_head_v1_"
_OWNER_PROJECTION_PREFIX: Final = "phase2b_step5_owner_status_projection_v1_"

_INITIALIZATION_LOCK_BASENAME: Final = "external-witness-initialization-v1.lock"

_INITIALIZATION_LOCK_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "mechanics_claim_boundary",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "external_witness_policy_id",
        "external_witness_policy_payload_sha256",
        "anchor_id",
        "anchor_payload_sha256",
        "receipt_id",
        "receipt_payload_sha256",
        "receipt_file_basename",
        "ledger_id",
        "ledger_payload_sha256",
        "ledger_file_basename",
        "checkpoint_id",
        "checkpoint_payload_sha256",
        "checkpoint_file_basename",
        "head_id",
        "head_payload_sha256",
        "head_file_basename",
    }
)

_LEDGER_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "mechanics_claim_boundary",
        "stage_b_trust_lock_id",
        "authority_profile_id",
        "signer_registry_id",
        "attempt_context_id",
        "external_witness_policy_id",
        "witness_namespace",
        "witness_control_domain_id",
        "witness_key_id",
        "witness_key_epoch",
        "initial_anchor_id",
        "initial_anchor_payload_sha256",
        "initial_receipt_id",
        "initial_receipt_payload_sha256",
        "witness_sequence",
        "tree_size",
        "signed_tree_root_sha256",
        "ledger_id",
    }
)
_CHECKPOINT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "mechanics_claim_boundary",
        "external_witness_policy_id",
        "ledger_id",
        "initial_receipt_id",
        "initial_receipt_payload_sha256",
        "witness_sequence",
        "tree_size",
        "signed_tree_root_sha256",
        "checkpoint_sequence",
        "previous_checkpoint_id",
        "checkpoint_id",
    }
)
_HEAD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "status",
        "claim_level",
        "mechanics_claim_boundary",
        "external_witness_policy_id",
        "ledger_id",
        "checkpoint_id",
        "checkpoint_payload_sha256",
        "initial_receipt_id",
        "initial_receipt_payload_sha256",
        "witness_sequence",
        "tree_size",
        "signed_tree_root_sha256",
        "previous_head_id",
        "head_id",
    }
)


class Phase2BStep5R01ExternalWitnessErrorV1(ValueError):
    """Stable fail-closed error for initial external-witness mechanics."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BStep5R01ExternalWitnessErrorV1(code, detail)


def _canonical(value: object, _implementation=_canonicalize) -> bytes:
    try:
        return _implementation(value)
    except (TypeError, ValueError, _RFC8785Error) as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_CANONICAL_V1, "RFC-8785 canonicalization failed"
        ) from exc


def _restore_wire_integers(value: object) -> object:
    if type(value) is float and math.isfinite(value) and value.is_integer():
        if abs(value) > _MAXIMUM_SAFE_INTEGER:
            _fail(FAIL_SCHEMA_V1, "decoded integer exceeds the I-JSON safe range")
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
        raise Phase2BStep5R01ExternalWitnessErrorV1(
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
    if (
        type(value) is not int
        or value < minimum
        or value > _MAXIMUM_SAFE_INTEGER
    ):
        _fail(FAIL_SCHEMA_V1, f"{label} must be a safe integer >= {minimum}")
    return value


def _hex64(value: object, label: str) -> str:
    text = _text(value, label, 64)
    if _HEX64.fullmatch(text) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be lowercase SHA-256 hex")
    return text


def _sha_address(value: object, label: str) -> str:
    text = _text(value, label, 71)
    if not text.startswith("sha256:") or _HEX64.fullmatch(text[7:]) is None:
        _fail(FAIL_SCHEMA_V1, f"{label} must be sha256:<lowercase SHA-256 hex>")
    return text


def _identifier(value: object, label: str, prefix: str | None = None) -> str:
    text = _text(value, label, 320)
    if _IDENTIFIER.fullmatch(text) is None or (
        prefix is not None and not text.startswith(prefix)
    ):
        _fail(FAIL_SCHEMA_V1, f"{label} identifier differs")
    return text


def _without(mapping: Mapping[str, object], *names: str) -> dict[str, object]:
    return {key: value for key, value in mapping.items() if key not in names}


def _payload_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_id(prefix: str, domain: bytes, subject: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(subject)).hexdigest()


def _require_frozen_domain_kats() -> None:
    if (
        hashlib.sha256(_EXTERNAL_POLICY_DOMAIN).hexdigest()
        != _EXTERNAL_POLICY_DOMAIN_SHA256_KAT
        or hashlib.sha256(_MERKLE_LEAF_DOMAIN).hexdigest()
        != _MERKLE_LEAF_DOMAIN_SHA256_KAT
    ):
        _fail(
            FAIL_TRUST_ANCHOR_V1,
            "external-policy or one-leaf Merkle domain KAT differs",
        )


def _decode_expected_identifier(
    raw: bytes, label: str, prefix: str | None = None
) -> str:
    if type(raw) is not bytes or not 1 <= len(raw) <= 320:
        _fail(FAIL_INPUT_V1, f"{label} must be bounded raw ASCII bytes")
    try:
        decoded = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_INPUT_V1, f"{label} is not ASCII"
        ) from exc
    return _identifier(decoded, label, prefix)


def _authority_role_keys(registry: object) -> tuple[Mapping[str, object], ...]:
    try:
        rows = registry.body["role_keys_in_exact_order"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_TRUST_ANCHOR_V1, "verified registry projection unavailable"
        ) from exc
    if type(rows) is not tuple or len(rows) != len(_AUTH_ROLE_ORDER):
        _fail(FAIL_TRUST_ANCHOR_V1, "verified registry role-key order differs")
    for expected_role, row in zip(_AUTH_ROLE_ORDER, rows, strict=True):
        if type(row) is not MappingProxyType or row.get("role") != expected_role:
            _fail(FAIL_TRUST_ANCHOR_V1, "verified registry role-key shape differs")
    return rows


def _pure_role_key(registry: object) -> Mapping[str, object]:
    rows = _authority_role_keys(registry)
    return rows[_AUTH_ROLE_ORDER.index("PURE_REPLAYER")]


def _load_raw_authority_graph(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    expected_stage_b_trust_lock_id_bytes: bytes,
    expected_authority_profile_id_bytes: bytes,
    expected_signer_registry_id_bytes: bytes,
) -> tuple[object, object, object]:
    expected_lock_id = _decode_expected_identifier(
        expected_stage_b_trust_lock_id_bytes,
        "expected Stage-B trust-lock ID",
        _STAGE_B_TRUST_LOCK_PREFIX,
    )
    expected_profile_id = _decode_expected_identifier(
        expected_authority_profile_id_bytes,
        "expected authority-profile ID",
        _AUTHORITY_PROFILE_PREFIX,
    )
    expected_registry_id = _decode_expected_identifier(
        expected_signer_registry_id_bytes,
        "expected signer-registry ID",
        _SIGNER_REGISTRY_PREFIX,
    )
    if (
        type(out_of_band_expected_lock_id_bytes) is not bytes
        or expected_stage_b_trust_lock_id_bytes
        != out_of_band_expected_lock_id_bytes
    ):
        _fail(
            FAIL_TRUST_ANCHOR_V1,
            "expected trust-lock ID and independent OOB lock ID differ",
        )
    try:
        lock = _load_stage_b_lock(
            stage_b_trust_lock_payload,
            out_of_band_expected_lock_id_bytes=(
                out_of_band_expected_lock_id_bytes
            ),
            out_of_band_expected_payload_sha256_digest=(
                out_of_band_expected_lock_payload_sha256_digest
            ),
        )
        profile = _load_authority_profile(authority_profile_payload, lock)
        registry = _load_signer_registry(
            signer_registry_payload, lock, profile
        )
        lock, profile, registry = _require_authority_graph(
            lock, profile, registry
        )
    except _AuthorityError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_TRUST_ANCHOR_V1, "raw authority-v2 graph rejected"
        ) from exc
    if (
        lock.artifact_id != expected_lock_id
        or profile.artifact_id != expected_profile_id
        or registry.artifact_id != expected_registry_id
    ):
        _fail(
            FAIL_TRUST_ANCHOR_V1,
            "raw authority-v2 graph differs from expected authority IDs",
        )
    return lock, profile, registry


def _load_external_policy(
    payload: bytes,
    *,
    expected_id_bytes: bytes,
    expected_payload_digest: bytes,
    registry: object,
) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 1 <= len(payload) <= _MAXIMUM_CANONICAL_BYTES
        or payload.endswith(b"\n")
    ):
        _fail(
            FAIL_INPUT_V1,
            "external policy must be bounded exact bytes without newline",
        )
    if (
        type(expected_id_bytes) is not bytes
        or type(expected_payload_digest) is not bytes
        or len(expected_payload_digest) != 32
    ):
        _fail(FAIL_INPUT_V1, "external policy OOB pins must be exact raw bytes")
    if hashlib.sha256(payload).digest() != expected_payload_digest:
        _fail(FAIL_TRUST_ANCHOR_V1, "external policy raw SHA-256 pin differs")
    expected_id = _decode_expected_identifier(
        expected_id_bytes,
        "expected external-witness policy ID",
        _EXTERNAL_POLICY_PREFIX,
    )
    policy = _object(
        _decode(payload, "external-witness policy"),
        frozenset(_EXTERNAL_POLICY_FIELDS),
        "external-witness policy",
    )
    if (
        policy["schema_version"] != _EXTERNAL_POLICY_SCHEMA
        or policy["status"] != _CANDIDATE_STATUS
        or policy["claim_level"] != _CANDIDATE_CLAIM_LEVEL
        or policy["algorithm"] != "Ed25519"
        or type(policy["witness_key_epoch"]) is not int
        or policy["witness_key_epoch"] != 1
    ):
        _fail(FAIL_SCHEMA_V1, "external-witness policy header differs")
    public_key_hex = _hex64(
        policy["public_key_hex"], "external-witness public key"
    )
    _identifier(policy["witness_key_id"], "external-witness key ID")
    _identifier(
        policy["witness_control_domain_id"],
        "external-witness control-domain ID",
    )
    _text(policy["witness_namespace"], "external-witness namespace", 256)
    subject = _without(policy, "external_witness_policy_id")
    policy_id = _content_id(
        _EXTERNAL_POLICY_PREFIX, _EXTERNAL_POLICY_DOMAIN, subject
    )
    if policy["external_witness_policy_id"] != policy_id or policy_id != expected_id:
        _fail(FAIL_TRUST_ANCHOR_V1, "external-witness policy content ID differs")
    role_keys = _authority_role_keys(registry)
    if (
        public_key_hex in {row["public_key_hex"] for row in role_keys}
        or policy["witness_key_id"] in {row["key_id"] for row in role_keys}
        or policy["witness_control_domain_id"]
        in {row["control_domain"] for row in role_keys}
        or policy["witness_namespace"] in set(_AUTH_ROLE_ORDER)
        or policy["witness_control_domain_id"] in set(_AUTH_ROLE_ORDER)
    ):
        _fail(
            FAIL_TRUST_ANCHOR_V1,
            "external witness is not disjoint from eight authority roles",
        )
    return policy


def _validate_anchor_body(
    value: object,
    *,
    lock: object,
    profile: object,
    registry: object,
    expected_attempt_context_id: str,
    external_policy_id: str,
) -> dict[str, object]:
    body = _object(value, frozenset(_ANCHOR_BODY_FIELDS), "anti-fork anchor body")
    if (
        body["schema_version"] != _ANCHOR_SCHEMA
        or body["status"] != _CANDIDATE_STATUS
        or body["claim_level"] != _CANDIDATE_CLAIM_LEVEL
    ):
        _fail(FAIL_CLAIM_BOUNDARY_V1, "anti-fork anchor candidate header differs")
    if (
        body["stage_b_trust_lock_id"] != lock.artifact_id
        or body["authority_profile_id"] != profile.artifact_id
        or body["signer_registry_id"] != registry.artifact_id
        or body["attempt_context_id"] != expected_attempt_context_id
        or body["external_witness_policy_id"] != external_policy_id
    ):
        _fail(FAIL_BINDING_V1, "anti-fork anchor authority/attempt/policy differs")
    _identifier(
        body["current_cut_statement_id"],
        "current-cut statement ID",
        _CURRENT_CUT_STATEMENT_PREFIX,
    )
    _hex64(
        body["current_cut_statement_payload_sha256"],
        "current-cut statement payload SHA",
    )
    _identifier(
        body["current_cut_token_id"],
        "current-cut logical-time token ID",
        _LOGICAL_TIME_TOKEN_PREFIX,
    )
    _identifier(
        body["current_cut_checkpoint_id"],
        "current-cut logical-time checkpoint ID",
        _LOGICAL_TIME_CHECKPOINT_PREFIX,
    )
    _identifier(
        body["current_cut_head_id"],
        "current-cut logical-time head ID",
        _LOGICAL_TIME_HEAD_PREFIX,
    )
    _identifier(
        body["owner_status_ledger_id"],
        "owner-status ledger ID",
        _OWNER_LEDGER_PREFIX,
    )
    _identifier(
        body["owner_current_head_id"],
        "owner-status current-head ID",
        _OWNER_HEAD_PREFIX,
    )
    _identifier(
        body["projection_id"],
        "owner-status projection ID",
        _OWNER_PROJECTION_PREFIX,
    )
    for field in (
        "current_cut_token_payload_sha256",
        "current_cut_checkpoint_payload_sha256",
        "current_cut_head_payload_sha256",
        "complete_event_set_root",
    ):
        _sha_address(body[field], f"anti-fork anchor {field}")
    if _integer(body["anchor_sequence"], "anchor sequence", 1) != 1:
        _fail(FAIL_BINDING_V1, "only the initial witness sequence is implemented")
    if body["previous_external_anchor_receipt_id"] is not None:
        _fail(FAIL_BINDING_V1, "initial anchor must not name a predecessor")
    _text(body["anchor_nonce"], "anti-fork anchor nonce", 256)
    return body


def _verify_pure_anchor_signature(
    root: Mapping[str, object], *, body: Mapping[str, object], registry: object
) -> None:
    signature = _object(
        root["signature"],
        frozenset(_AUTH_SIGNATURE_FIELDS),
        "PURE detached signature",
    )
    key = _pure_role_key(registry)
    if (
        signature["schema_version"] != _AUTH_SIGNATURE_SCHEMA
        or signature["algorithm"] != "Ed25519"
        or signature["role"] != "PURE_REPLAYER"
        or signature["purpose"] != "SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR"
        or type(signature["key_epoch"]) is not int
        or signature["key_epoch"] != key["key_epoch"]
        or signature["signer_key_id"] != key["key_id"]
        or signature["purpose"] not in key["authorized_signature_purposes"]
    ):
        _fail(FAIL_SIGNATURE_V1, "PURE anchor signature metadata differs")
    signature_hex = _text(signature["signature_hex"], "PURE signature", 128)
    if _HEX128.fullmatch(signature_hex) is None:
        _fail(FAIL_SIGNATURE_V1, "PURE Ed25519 signature encoding differs")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        preimage = _authority_signature_preimage(
            artifact_kind="owner_status_anti_fork_anchor",
            body=body,
            role="PURE_REPLAYER",
            purpose="SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
            signer_key_id=key["key_id"],
            key_epoch=key["key_epoch"],
        )
        public_key = _Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(key["public_key_hex"])
        )
        public_key.verify(bytes.fromhex(signature_hex), preimage)
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_SIGNATURE_V1, "PURE anti-fork-anchor signature rejected"
        ) from exc


def _load_signed_anchor(
    payload: bytes,
    *,
    lock: object,
    profile: object,
    registry: object,
    expected_attempt_context_id: str,
    external_policy_id: str,
) -> dict[str, object]:
    root = _object(
        _decode(payload, "PURE-signed owner-status anti-fork anchor"),
        frozenset(_ANCHOR_FIELDS),
        "PURE-signed owner-status anti-fork anchor",
    )
    if root["schema_version"] != _ANCHOR_SCHEMA:
        _fail(FAIL_SCHEMA_V1, "anti-fork anchor wrapper schema differs")
    body = _validate_anchor_body(
        root["body"],
        lock=lock,
        profile=profile,
        registry=registry,
        expected_attempt_context_id=expected_attempt_context_id,
        external_policy_id=external_policy_id,
    )
    expected_anchor_id = _evidence_content_id(
        "signed_owner_status_anti_fork_anchor", body
    )
    if root["anchor_id"] != expected_anchor_id:
        _fail(FAIL_CONTENT_ID_V1, "anti-fork anchor ID differs")
    _verify_pure_anchor_signature(root, body=body, registry=registry)
    return root


def _require_matching_private_key(
    external_witness_private_key: object,
    policy: Mapping[str, object],
) -> object:
    if (
        _Ed25519PrivateKey is None
        or _Ed25519PublicKey is None
        or _serialization is None
        or _InvalidSignature is None
    ):
        _fail(FAIL_KEY_V1, "cryptography Ed25519 backend unavailable")
    if not isinstance(external_witness_private_key, _Ed25519PrivateKey):
        _fail(FAIL_KEY_V1, "fixed external-witness Ed25519 private key required")
    try:
        public_bytes = external_witness_private_key.public_key().public_bytes(
            encoding=_serialization.Encoding.Raw,
            format=_serialization.PublicFormat.Raw,
        )
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_KEY_V1, "external-witness public-key derivation failed"
        ) from exc
    if (
        type(public_bytes) is not bytes
        or len(public_bytes) != 32
        or public_bytes.hex() != policy["public_key_hex"]
    ):
        _fail(FAIL_KEY_V1, "fixed private key differs from raw-pinned policy")
    return external_witness_private_key


def _compose_and_sign_initial_receipt(
    *,
    anchor_payload: bytes,
    anchor: Mapping[str, object],
    policy: Mapping[str, object],
    private_key: object,
) -> tuple[dict[str, object], bytes]:
    tree_root = hashlib.sha256(_MERKLE_LEAF_DOMAIN + anchor_payload).hexdigest()
    unsigned_id_subject: dict[str, object] = {
        "schema_version": _EXTERNAL_RECEIPT_SCHEMA,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "witness_namespace": policy["witness_namespace"],
        "witness_control_domain_id": policy["witness_control_domain_id"],
        "witness_key_id": policy["witness_key_id"],
        "witness_key_epoch": policy["witness_key_epoch"],
        "anchor_id": anchor["anchor_id"],
        "anchor_payload_sha256": _payload_sha(anchor_payload),
        "witness_sequence": 1,
        "previous_external_receipt_id": None,
        "leaf_index": 0,
        "tree_size": 1,
        "signed_tree_root_sha256": tree_root,
        "inclusion_path_sha256_in_exact_order": [],
        "algorithm": "Ed25519",
    }
    receipt_id = _evidence_content_id(
        "external_owner_status_anchor_receipt", unsigned_id_subject
    )
    unsigned_for_signature = dict(unsigned_id_subject)
    unsigned_for_signature["receipt_id"] = receipt_id
    try:
        preimage = _external_signature_preimage(unsigned_for_signature)
        signature = private_key.sign(preimage)
        if type(signature) is not bytes or len(signature) != 64:
            _fail(FAIL_SIGNATURE_V1, "external Ed25519 signer returned invalid bytes")
        public_key = _Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(policy["public_key_hex"])
        )
        public_key.verify(signature, preimage)
    except Phase2BStep5R01ExternalWitnessErrorV1:
        raise
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_SIGNATURE_V1, "external-witness receipt signing failed"
        ) from exc
    receipt = dict(unsigned_for_signature)
    receipt["signature_hex"] = signature.hex()
    if frozenset(receipt) != frozenset(_EXTERNAL_RECEIPT_FIELDS):
        _fail(FAIL_SCHEMA_V1, "composed external receipt keyset differs")
    payload = _canonical(receipt)
    decoded = _object(
        _decode(payload, "composed external anchor receipt"),
        frozenset(_EXTERNAL_RECEIPT_FIELDS),
        "composed external anchor receipt",
    )
    if decoded != receipt:
        _fail(FAIL_CANONICAL_V1, "composed external receipt readback differs")
    return receipt, payload


def _load_existing_initial_receipt(
    payload: bytes,
    *,
    anchor_payload: bytes,
    anchor: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    receipt = _object(
        _decode(payload, "persisted external anchor receipt"),
        frozenset(_EXTERNAL_RECEIPT_FIELDS),
        "persisted external anchor receipt",
    )
    _integer(receipt["witness_key_epoch"], "persisted witness key epoch", 1)
    _integer(receipt["witness_sequence"], "persisted witness sequence", 1)
    _integer(receipt["leaf_index"], "persisted witness leaf index")
    _integer(receipt["tree_size"], "persisted witness tree size", 1)
    if (
        receipt["schema_version"] != _EXTERNAL_RECEIPT_SCHEMA
        or receipt["external_witness_policy_id"]
        != policy["external_witness_policy_id"]
        or receipt["witness_namespace"] != policy["witness_namespace"]
        or receipt["witness_control_domain_id"]
        != policy["witness_control_domain_id"]
        or receipt["witness_key_id"] != policy["witness_key_id"]
        or receipt["witness_key_epoch"] != policy["witness_key_epoch"]
        or receipt["algorithm"] != "Ed25519"
        or receipt["anchor_id"] != anchor["anchor_id"]
        or receipt["anchor_payload_sha256"] != _payload_sha(anchor_payload)
        or receipt["witness_sequence"] != 1
        or receipt["previous_external_receipt_id"] is not None
        or receipt["leaf_index"] != 0
        or receipt["tree_size"] != 1
        or type(receipt["inclusion_path_sha256_in_exact_order"]) is not list
        or receipt["inclusion_path_sha256_in_exact_order"]
    ):
        _fail(FAIL_BINDING_V1, "persisted initial receipt lineage differs")
    expected_root = hashlib.sha256(_MERKLE_LEAF_DOMAIN + anchor_payload).hexdigest()
    if receipt["signed_tree_root_sha256"] != expected_root:
        _fail(FAIL_BINDING_V1, "persisted one-leaf Merkle root differs")
    signature_hex = _text(
        receipt["signature_hex"], "persisted external signature", 128
    )
    if _HEX128.fullmatch(signature_hex) is None:
        _fail(FAIL_SIGNATURE_V1, "persisted external signature encoding differs")
    expected_receipt_id = _evidence_content_id(
        "external_owner_status_anchor_receipt",
        _without(receipt, "signature_hex", "receipt_id"),
    )
    if receipt["receipt_id"] != expected_receipt_id:
        _fail(FAIL_CONTENT_ID_V1, "persisted external receipt ID differs")
    if _Ed25519PublicKey is None or _InvalidSignature is None:
        _fail(FAIL_SIGNATURE_V1, "cryptography Ed25519 backend unavailable")
    try:
        public_key = _Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(policy["public_key_hex"])
        )
        public_key.verify(
            bytes.fromhex(signature_hex),
            _external_signature_preimage(dict(receipt)),
        )
    except Exception as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_SIGNATURE_V1, "persisted external receipt signature rejected"
        ) from exc
    return receipt


def _compose_spool_artifacts(
    *,
    lock: object,
    profile: object,
    registry: object,
    expected_attempt_context_id: str,
    policy: Mapping[str, object],
    anchor_payload: bytes,
    anchor: Mapping[str, object],
    receipt_payload: bytes,
    receipt: Mapping[str, object],
) -> tuple[bytes, dict[str, object], bytes, dict[str, object], bytes, dict[str, object]]:
    ledger: dict[str, object] = {
        "schema_version": EXTERNAL_WITNESS_LEDGER_SCHEMA_V1,
        "status": _CANDIDATE_STATUS,
        "claim_level": _CANDIDATE_CLAIM_LEVEL,
        "mechanics_claim_boundary": MECHANICS_CLAIM_BOUNDARY_V1,
        "stage_b_trust_lock_id": lock.artifact_id,
        "authority_profile_id": profile.artifact_id,
        "signer_registry_id": registry.artifact_id,
        "attempt_context_id": expected_attempt_context_id,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "witness_namespace": policy["witness_namespace"],
        "witness_control_domain_id": policy["witness_control_domain_id"],
        "witness_key_id": policy["witness_key_id"],
        "witness_key_epoch": policy["witness_key_epoch"],
        "initial_anchor_id": anchor["anchor_id"],
        "initial_anchor_payload_sha256": _payload_sha(anchor_payload),
        "initial_receipt_id": receipt["receipt_id"],
        "initial_receipt_payload_sha256": _payload_sha(receipt_payload),
        "witness_sequence": 1,
        "tree_size": 1,
        "signed_tree_root_sha256": receipt["signed_tree_root_sha256"],
    }
    ledger["ledger_id"] = _content_id(_LEDGER_PREFIX, _LEDGER_DOMAIN, ledger)
    if frozenset(ledger) != _LEDGER_FIELDS:
        _fail(FAIL_SCHEMA_V1, "external-witness ledger keyset differs")
    ledger_payload = _canonical(ledger)

    checkpoint: dict[str, object] = {
        "schema_version": EXTERNAL_WITNESS_CHECKPOINT_SCHEMA_V1,
        "status": _CANDIDATE_STATUS,
        "claim_level": _CANDIDATE_CLAIM_LEVEL,
        "mechanics_claim_boundary": MECHANICS_CLAIM_BOUNDARY_V1,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "ledger_id": ledger["ledger_id"],
        "initial_receipt_id": receipt["receipt_id"],
        "initial_receipt_payload_sha256": _payload_sha(receipt_payload),
        "witness_sequence": 1,
        "tree_size": 1,
        "signed_tree_root_sha256": receipt["signed_tree_root_sha256"],
        "checkpoint_sequence": 1,
        "previous_checkpoint_id": None,
    }
    checkpoint["checkpoint_id"] = _content_id(
        _CHECKPOINT_PREFIX, _CHECKPOINT_DOMAIN, checkpoint
    )
    if frozenset(checkpoint) != _CHECKPOINT_FIELDS:
        _fail(FAIL_SCHEMA_V1, "external-witness checkpoint keyset differs")
    checkpoint_payload = _canonical(checkpoint)

    head: dict[str, object] = {
        "schema_version": EXTERNAL_WITNESS_HEAD_SCHEMA_V1,
        "status": _CANDIDATE_STATUS,
        "claim_level": _CANDIDATE_CLAIM_LEVEL,
        "mechanics_claim_boundary": MECHANICS_CLAIM_BOUNDARY_V1,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "ledger_id": ledger["ledger_id"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_payload_sha256": _payload_sha(checkpoint_payload),
        "initial_receipt_id": receipt["receipt_id"],
        "initial_receipt_payload_sha256": _payload_sha(receipt_payload),
        "witness_sequence": 1,
        "tree_size": 1,
        "signed_tree_root_sha256": receipt["signed_tree_root_sha256"],
        "previous_head_id": None,
    }
    head["head_id"] = _content_id(_HEAD_PREFIX, _HEAD_DOMAIN, head)
    if frozenset(head) != _HEAD_FIELDS:
        _fail(FAIL_SCHEMA_V1, "external-witness head keyset differs")
    head_payload = _canonical(head)
    for payload, fields, label in (
        (ledger_payload, _LEDGER_FIELDS, "external-witness ledger"),
        (checkpoint_payload, _CHECKPOINT_FIELDS, "external-witness checkpoint"),
        (head_payload, _HEAD_FIELDS, "external-witness head"),
    ):
        _object(_decode(payload, label), fields, label)
    return (
        ledger_payload,
        ledger,
        checkpoint_payload,
        checkpoint,
        head_payload,
        head,
    )


def _require_linux_open_flags() -> None:
    for name in ("O_NOFOLLOW", "O_EXCL", "O_DIRECTORY", "O_CLOEXEC"):
        if not hasattr(os, name):
            _fail(FAIL_FILESYSTEM_V1, f"Linux {name} support is required")


def _validate_private_directory_stat(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        _fail(FAIL_FILESYSTEM_V1, "spool directory is not owner-private")


def _open_owner_private_directory(path: str) -> int:
    _require_linux_open_flags()
    if (
        type(path) is not str
        or not 1 <= len(path.encode("utf-8")) <= 4096
        or "\x00" in path
        or not os.path.isabs(path)
    ):
        _fail(FAIL_INPUT_V1, "spool directory must be a bounded absolute path")
    try:
        directory_fd = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1, "owner-private spool directory open failed"
        ) from exc
    try:
        _validate_private_directory_stat(os.fstat(directory_fd))
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


def _validate_regular_record_stat(
    metadata: os.stat_result, *, expected_size: int, label: str
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size != expected_size
    ):
        _fail(FAIL_FILESYSTEM_V1, f"{label} is not one owner-private exact file")


def _read_bounded_record(directory_fd: int, basename: str) -> bytes:
    if (
        type(basename) is not str
        or not 1 <= len(basename.encode("utf-8")) <= 512
        or "/" in basename
        or basename in (".", "..")
    ):
        _fail(FAIL_FILESYSTEM_V1, "persisted spool basename is unsafe")
    try:
        fd = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1, "persisted spool record open failed"
        ) from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
            or not 1 <= metadata.st_size <= _MAXIMUM_SPOOL_RECORD_BYTES
        ):
            _fail(
                FAIL_FILESYSTEM_V1,
                "persisted spool record is not one bounded owner-private file",
            )
        os.fsync(fd)
        expected_size = metadata.st_size
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(fd, min(remaining, 65_536))
            if not chunk:
                _fail(FAIL_FILESYSTEM_V1, "persisted spool record ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            _fail(FAIL_FILESYSTEM_V1, "persisted spool record grew during read")
        payload = b"".join(chunks)
        if len(payload) != expected_size:
            _fail(FAIL_FILESYSTEM_V1, "persisted spool record size drifted")
        return payload
    finally:
        os.close(fd)


def _read_exact_record(directory_fd: int, basename: str, expected: bytes) -> bytes:
    if type(expected) is not bytes or len(expected) > _MAXIMUM_SPOOL_RECORD_BYTES:
        _fail(FAIL_INPUT_V1, "spool record exceeds the frozen bound")
    try:
        fd = os.open(
            basename,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1, "spool record reopen failed"
        ) from exc
    try:
        _validate_regular_record_stat(
            os.fstat(fd), expected_size=len(expected), label=basename
        )
        chunks: list[bytes] = []
        remaining = len(expected)
        while remaining:
            chunk = os.read(fd, min(remaining, 65_536))
            if not chunk:
                _fail(FAIL_FILESYSTEM_V1, f"{basename} ended during readback")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            _fail(FAIL_FILESYSTEM_V1, f"{basename} grew during readback")
        actual = b"".join(chunks)
        if actual != expected:
            _fail(FAIL_FILESYSTEM_V1, f"{basename} changed during readback")
        return actual
    finally:
        os.close(fd)


def _write_exclusive_record(directory_fd: int, basename: str, payload: bytes) -> None:
    if (
        type(basename) is not str
        or "/" in basename
        or basename in ("", ".", "..")
        or type(payload) is not bytes
        or not 1 <= len(payload) <= _MAXIMUM_SPOOL_RECORD_BYTES
    ):
        _fail(FAIL_INPUT_V1, "exclusive spool record arguments differ")
    try:
        fd = os.open(
            basename,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1,
            "exclusive spool publication failed; reuse is forbidden",
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                _fail(FAIL_FILESYSTEM_V1, "spool record write made no progress")
            offset += written
        os.fsync(fd)
        _validate_regular_record_stat(
            os.fstat(fd), expected_size=len(payload), label=basename
        )
    finally:
        os.close(fd)
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1, "spool directory fsync failed"
        ) from exc
    _read_exact_record(directory_fd, basename, payload)


def _record_basenames(
    *,
    receipt_payload: bytes,
    ledger_payload: bytes,
    checkpoint_payload: bytes,
    head_payload: bytes,
) -> tuple[str, str, str, str]:
    receipt_basename = (
        "00000000000000000001.external-anchor-receipt."
        + _payload_sha(receipt_payload)
        + ".json"
    )
    ledger_basename = (
        "00000000000000000001.external-witness-ledger."
        + _payload_sha(ledger_payload)
        + ".json"
    )
    checkpoint_basename = (
        "00000000000000000001.external-witness-checkpoint."
        + _payload_sha(checkpoint_payload)
        + ".json"
    )
    head_basename = (
        "00000000000000000001.external-witness-head."
        + _payload_sha(head_payload)
        + ".json"
    )
    return (
        receipt_basename,
        ledger_basename,
        checkpoint_basename,
        head_basename,
    )


def _compose_initialization_lock(
    *,
    lock: object,
    profile: object,
    registry: object,
    expected_attempt_context_id: str,
    policy_payload: bytes,
    policy: Mapping[str, object],
    anchor_payload: bytes,
    anchor: Mapping[str, object],
    receipt_payload: bytes,
    receipt: Mapping[str, object],
    ledger_payload: bytes,
    ledger: Mapping[str, object],
    checkpoint_payload: bytes,
    checkpoint: Mapping[str, object],
    head_payload: bytes,
    head: Mapping[str, object],
) -> tuple[bytes, tuple[str, str, str, str]]:
    basenames = _record_basenames(
        receipt_payload=receipt_payload,
        ledger_payload=ledger_payload,
        checkpoint_payload=checkpoint_payload,
        head_payload=head_payload,
    )
    receipt_basename, ledger_basename, checkpoint_basename, head_basename = (
        basenames
    )
    intent = {
        "schema_version": EXTERNAL_WITNESS_INITIALIZATION_LOCK_SCHEMA_V1,
        "status": _CANDIDATE_STATUS,
        "claim_level": _CANDIDATE_CLAIM_LEVEL,
        "mechanics_claim_boundary": MECHANICS_CLAIM_BOUNDARY_V1,
        "stage_b_trust_lock_id": lock.artifact_id,
        "authority_profile_id": profile.artifact_id,
        "signer_registry_id": registry.artifact_id,
        "attempt_context_id": expected_attempt_context_id,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "external_witness_policy_payload_sha256": _payload_sha(policy_payload),
        "anchor_id": anchor["anchor_id"],
        "anchor_payload_sha256": _payload_sha(anchor_payload),
        "receipt_id": receipt["receipt_id"],
        "receipt_payload_sha256": _payload_sha(receipt_payload),
        "receipt_file_basename": receipt_basename,
        "ledger_id": ledger["ledger_id"],
        "ledger_payload_sha256": _payload_sha(ledger_payload),
        "ledger_file_basename": ledger_basename,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_payload_sha256": _payload_sha(checkpoint_payload),
        "checkpoint_file_basename": checkpoint_basename,
        "head_id": head["head_id"],
        "head_payload_sha256": _payload_sha(head_payload),
        "head_file_basename": head_basename,
    }
    if frozenset(intent) != _INITIALIZATION_LOCK_FIELDS:
        _fail(FAIL_SCHEMA_V1, "initialization-lock keyset differs")
    return _canonical(intent), basenames


def _load_existing_committed_initial_leaf(
    owner_private_spool_directory_path: str,
    *,
    lock: object,
    profile: object,
    registry: object,
    expected_attempt_context_id: str,
    policy_payload: bytes,
    policy: Mapping[str, object],
    anchor_payload: bytes,
    anchor: Mapping[str, object],
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    tuple[str, ...],
] | None:
    directory_fd = _open_owner_private_directory(
        owner_private_spool_directory_path
    )
    try:
        try:
            entries = os.listdir(directory_fd)
        except OSError as exc:
            raise Phase2BStep5R01ExternalWitnessErrorV1(
                FAIL_FILESYSTEM_V1, "spool directory enumeration failed"
            ) from exc
        if type(entries) is not list or any(type(name) is not str for name in entries):
            _fail(FAIL_FILESYSTEM_V1, "spool directory enumeration shape differs")
        if not entries:
            return None
        if _INITIALIZATION_LOCK_BASENAME not in entries:
            _fail(
                FAIL_FILESYSTEM_V1,
                "nonempty spool lacks the immutable initialization lock",
            )
        lock_payload = _read_bounded_record(
            directory_fd, _INITIALIZATION_LOCK_BASENAME
        )
        intent = _object(
            _decode(lock_payload, "external-witness initialization lock"),
            _INITIALIZATION_LOCK_FIELDS,
            "external-witness initialization lock",
        )
        if (
            intent["schema_version"]
            != EXTERNAL_WITNESS_INITIALIZATION_LOCK_SCHEMA_V1
            or intent["status"] != _CANDIDATE_STATUS
            or intent["claim_level"] != _CANDIDATE_CLAIM_LEVEL
            or intent["mechanics_claim_boundary"]
            != MECHANICS_CLAIM_BOUNDARY_V1
            or intent["stage_b_trust_lock_id"] != lock.artifact_id
            or intent["authority_profile_id"] != profile.artifact_id
            or intent["signer_registry_id"] != registry.artifact_id
            or intent["attempt_context_id"] != expected_attempt_context_id
            or intent["external_witness_policy_id"]
            != policy["external_witness_policy_id"]
            or intent["external_witness_policy_payload_sha256"]
            != _payload_sha(policy_payload)
            or intent["anchor_id"] != anchor["anchor_id"]
            or intent["anchor_payload_sha256"] != _payload_sha(anchor_payload)
        ):
            _fail(
                FAIL_BINDING_V1,
                "committed initial leaf differs from presented raw authority/policy/anchor",
            )
        basenames = tuple(
            intent[field]
            for field in (
                "receipt_file_basename",
                "ledger_file_basename",
                "checkpoint_file_basename",
                "head_file_basename",
            )
        )
        if any(
            type(name) is not str
            or not 1 <= len(name.encode("utf-8")) <= 512
            or "/" in name
            or name in (".", "..")
            for name in basenames
        ) or len(set(basenames)) != 4:
            _fail(FAIL_FILESYSTEM_V1, "initialization-lock basenames differ")
        expected_entries = {_INITIALIZATION_LOCK_BASENAME, *basenames}
        if set(entries) != expected_entries or len(entries) != 5:
            _fail(
                FAIL_FILESYSTEM_V1,
                "partial, gapped, duplicate, or extra initial spool is poisoned",
            )
        receipt_payload = _read_bounded_record(directory_fd, basenames[0])
        ledger_payload = _read_bounded_record(directory_fd, basenames[1])
        checkpoint_payload = _read_bounded_record(directory_fd, basenames[2])
        head_payload = _read_bounded_record(directory_fd, basenames[3])
        payloads = (
            receipt_payload,
            ledger_payload,
            checkpoint_payload,
            head_payload,
        )
        for payload, field in zip(
            payloads,
            (
                "receipt_payload_sha256",
                "ledger_payload_sha256",
                "checkpoint_payload_sha256",
                "head_payload_sha256",
            ),
            strict=True,
        ):
            if _payload_sha(payload) != intent[field]:
                _fail(FAIL_FILESYSTEM_V1, "initial spool payload SHA differs")
        receipt = _load_existing_initial_receipt(
            receipt_payload,
            anchor_payload=anchor_payload,
            anchor=anchor,
            policy=policy,
        )
        (
            expected_ledger_payload,
            ledger,
            expected_checkpoint_payload,
            checkpoint,
            expected_head_payload,
            head,
        ) = _compose_spool_artifacts(
            lock=lock,
            profile=profile,
            registry=registry,
            expected_attempt_context_id=expected_attempt_context_id,
            policy=policy,
            anchor_payload=anchor_payload,
            anchor=anchor,
            receipt_payload=receipt_payload,
            receipt=receipt,
        )
        if (
            ledger_payload != expected_ledger_payload
            or checkpoint_payload != expected_checkpoint_payload
            or head_payload != expected_head_payload
            or intent["receipt_id"] != receipt["receipt_id"]
            or intent["ledger_id"] != ledger["ledger_id"]
            or intent["checkpoint_id"] != checkpoint["checkpoint_id"]
            or intent["head_id"] != head["head_id"]
        ):
            _fail(FAIL_BINDING_V1, "committed initial spool lineage differs")
        expected_lock_payload, expected_basenames = _compose_initialization_lock(
            lock=lock,
            profile=profile,
            registry=registry,
            expected_attempt_context_id=expected_attempt_context_id,
            policy_payload=policy_payload,
            policy=policy,
            anchor_payload=anchor_payload,
            anchor=anchor,
            receipt_payload=receipt_payload,
            receipt=receipt,
            ledger_payload=ledger_payload,
            ledger=ledger,
            checkpoint_payload=checkpoint_payload,
            checkpoint=checkpoint,
            head_payload=head_payload,
            head=head,
        )
        if lock_payload != expected_lock_payload or basenames != expected_basenames:
            _fail(FAIL_BINDING_V1, "initialization-lock exact replay differs")
        os.fsync(directory_fd)
        final_entries = os.listdir(directory_fd)
        if set(final_entries) != expected_entries or len(final_entries) != 5:
            _fail(FAIL_FILESYSTEM_V1, "initial spool changed during retry readback")
        names = (_INITIALIZATION_LOCK_BASENAME, *basenames)
        return (
            receipt,
            receipt_payload,
            ledger,
            ledger_payload,
            checkpoint,
            checkpoint_payload,
            head,
            head_payload,
            names,
        )
    except Phase2BStep5R01ExternalWitnessErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1,
            "initial external-witness spool publication failed closed",
        ) from exc
    finally:
        os.close(directory_fd)


def _spool_initial_leaf(
    owner_private_spool_directory_path: str,
    *,
    initialization_lock_payload: bytes,
    receipt_payload: bytes,
    ledger_payload: bytes,
    checkpoint_payload: bytes,
    head_payload: bytes,
    basenames: tuple[str, str, str, str],
) -> tuple[str, ...]:
    receipt_basename, ledger_basename, checkpoint_basename, head_basename = (
        basenames
    )
    directory_fd = _open_owner_private_directory(
        owner_private_spool_directory_path
    )
    try:
        if os.listdir(directory_fd):
            _fail(
                FAIL_FILESYSTEM_V1,
                "initial publication requires an empty dedicated spool",
            )
        publications = (
            (_INITIALIZATION_LOCK_BASENAME, initialization_lock_payload),
            (receipt_basename, receipt_payload),
            (ledger_basename, ledger_payload),
            (checkpoint_basename, checkpoint_payload),
            (head_basename, head_payload),
        )
        for basename, payload in publications:
            _write_exclusive_record(directory_fd, basename, payload)
        os.fsync(directory_fd)
        for basename, payload in publications:
            _read_exact_record(directory_fd, basename, payload)
        entries = os.listdir(directory_fd)
        if set(entries) != {name for name, _payload in publications} or len(entries) != 5:
            _fail(FAIL_FILESYSTEM_V1, "initial spool gained an extra or missing file")
        return tuple(basename for basename, _payload in publications)
    except Phase2BStep5R01ExternalWitnessErrorV1:
        raise
    except OSError as exc:
        raise Phase2BStep5R01ExternalWitnessErrorV1(
            FAIL_FILESYSTEM_V1,
            "initial external-witness spool publication failed closed",
        ) from exc
    finally:
        os.close(directory_fd)


@dataclass(frozen=True, slots=True, init=False)
class IssuedInitialExternalWitnessCandidateV1:
    receipt_payload: bytes
    artifact_ids: Mapping[str, str]
    payload_sha256s: Mapping[str, str]
    spool_file_basenames_in_commit_order: tuple[str, ...]
    mechanics_claim_boundary: str
    raw_authority_signature_mechanics_verified: bool
    receipt_signature_mechanics_verified: bool
    local_fsync_readback_mechanics_verified: bool
    same_uid_isolation_verified: bool
    independent_or_local_filesystem_verified: bool
    host_rollback_resistance_verified: bool
    key_custody_verified: bool
    attempt_freshness_verified: bool
    external_live_service_verified: bool
    external_control_verified: bool
    external_currentness_verified: bool
    external_anti_fork_anchor_verified: bool
    durability_established: bool
    authority_established: bool
    r01_satisfied: bool
    readiness_satisfied: bool
    execution_authorized: bool

    def __new__(
        cls, *args: object, **kwargs: object
    ) -> "IssuedInitialExternalWitnessCandidateV1":
        raise TypeError("external-witness candidate results are privately issued")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("external-witness candidate results are privately issued")


def _build_result_lineage():
    result_type = IssuedInitialExternalWitnessCandidateV1
    failure = _fail
    failure_input = FAIL_INPUT_V1
    claim_boundary = MECHANICS_CLAIM_BOUNDARY_V1
    mapping_proxy = MappingProxyType
    issued: dict[int, tuple[object, ...]] = {}

    def issue(
        receipt_payload: bytes,
        artifact_ids: Mapping[str, str],
        payload_sha256s: Mapping[str, str],
        spool_names: tuple[str, ...],
    ) -> IssuedInitialExternalWitnessCandidateV1:
        value = object.__new__(result_type)
        frozen_ids = mapping_proxy(dict(artifact_ids))
        frozen_shas = mapping_proxy(dict(payload_sha256s))
        receipt_copy = bytes(receipt_payload)
        names_copy = tuple(spool_names)
        object.__setattr__(value, "receipt_payload", receipt_copy)
        object.__setattr__(value, "artifact_ids", frozen_ids)
        object.__setattr__(value, "payload_sha256s", frozen_shas)
        object.__setattr__(
            value, "spool_file_basenames_in_commit_order", names_copy
        )
        object.__setattr__(
            value, "mechanics_claim_boundary", claim_boundary
        )
        object.__setattr__(
            value, "raw_authority_signature_mechanics_verified", True
        )
        object.__setattr__(value, "receipt_signature_mechanics_verified", True)
        object.__setattr__(
            value, "local_fsync_readback_mechanics_verified", True
        )
        for field in (
            "same_uid_isolation_verified",
            "independent_or_local_filesystem_verified",
            "host_rollback_resistance_verified",
            "key_custody_verified",
            "attempt_freshness_verified",
            "external_live_service_verified",
            "external_control_verified",
            "external_currentness_verified",
            "external_anti_fork_anchor_verified",
            "durability_established",
            "authority_established",
            "r01_satisfied",
            "readiness_satisfied",
            "execution_authorized",
        ):
            object.__setattr__(value, field, False)
        issued[id(value)] = (
            value,
            receipt_copy,
            frozen_ids,
            frozen_shas,
            names_copy,
        )
        return value

    def require(value: object) -> IssuedInitialExternalWitnessCandidateV1:
        if type(value) is not result_type:
            failure(failure_input, "issued external-witness candidate required")
        record = issued.get(id(value))
        if record is None or record[0] is not value:
            failure(failure_input, "external-witness candidate lacks lineage")
        if (
            value.receipt_payload is not record[1]
            or value.artifact_ids is not record[2]
            or value.payload_sha256s is not record[3]
            or value.spool_file_basenames_in_commit_order is not record[4]
            or value.mechanics_claim_boundary != claim_boundary
            or value.raw_authority_signature_mechanics_verified is not True
            or value.receipt_signature_mechanics_verified is not True
            or value.local_fsync_readback_mechanics_verified is not True
            or value.same_uid_isolation_verified is not False
            or value.independent_or_local_filesystem_verified is not False
            or value.host_rollback_resistance_verified is not False
            or value.key_custody_verified is not False
            or value.attempt_freshness_verified is not False
            or value.external_live_service_verified is not False
            or value.external_control_verified is not False
            or value.external_currentness_verified is not False
            or value.external_anti_fork_anchor_verified is not False
            or value.durability_established is not False
            or value.authority_established is not False
            or value.r01_satisfied is not False
            or value.readiness_satisfied is not False
            or value.execution_authorized is not False
        ):
            failure(failure_input, "external-witness candidate lineage was mutated")
        return value

    return issue, require


_issue_result, _require_result = _build_result_lineage()
del _build_result_lineage


def _issue_initial_external_owner_status_anchor_receipt_implementation_v1(
    stage_b_trust_lock_payload: bytes,
    authority_profile_payload: bytes,
    signer_registry_payload: bytes,
    external_witness_policy_payload: bytes,
    signed_owner_status_anti_fork_anchor_payload: bytes,
    *,
    out_of_band_expected_lock_id_bytes: bytes,
    out_of_band_expected_lock_payload_sha256_digest: bytes,
    out_of_band_expected_external_witness_policy_id_bytes: bytes,
    out_of_band_expected_external_witness_policy_payload_sha256_digest: bytes,
    expected_stage_b_trust_lock_id_bytes: bytes,
    expected_authority_profile_id_bytes: bytes,
    expected_signer_registry_id_bytes: bytes,
    expected_attempt_context_id_bytes: bytes,
    external_witness_private_key: object,
    owner_private_spool_directory_path: str,
) -> IssuedInitialExternalWitnessCandidateV1:
    """Sign and exclusively spool an initial mechanics-only witness receipt.

    All protocol artifacts and expected authority/attempt identifiers enter as
    raw bytes.  No parsed mapping, verified token, or verifier callback is
    accepted.  The private key and spool directory are fixed service-binding
    inputs; neither is evidence of custody, independence, or external control.
    """
    _require_frozen_domain_kats()
    expected_attempt_context_id = _decode_expected_identifier(
        expected_attempt_context_id_bytes, "expected attempt-context ID"
    )
    lock, profile, registry = _load_raw_authority_graph(
        stage_b_trust_lock_payload,
        authority_profile_payload,
        signer_registry_payload,
        out_of_band_expected_lock_id_bytes=(
            out_of_band_expected_lock_id_bytes
        ),
        out_of_band_expected_lock_payload_sha256_digest=(
            out_of_band_expected_lock_payload_sha256_digest
        ),
        expected_stage_b_trust_lock_id_bytes=(
            expected_stage_b_trust_lock_id_bytes
        ),
        expected_authority_profile_id_bytes=(
            expected_authority_profile_id_bytes
        ),
        expected_signer_registry_id_bytes=expected_signer_registry_id_bytes,
    )
    policy = _load_external_policy(
        external_witness_policy_payload,
        expected_id_bytes=(
            out_of_band_expected_external_witness_policy_id_bytes
        ),
        expected_payload_digest=(
            out_of_band_expected_external_witness_policy_payload_sha256_digest
        ),
        registry=registry,
    )
    anchor = _load_signed_anchor(
        signed_owner_status_anti_fork_anchor_payload,
        lock=lock,
        profile=profile,
        registry=registry,
        expected_attempt_context_id=expected_attempt_context_id,
        external_policy_id=policy["external_witness_policy_id"],
    )
    private_key = _require_matching_private_key(
        external_witness_private_key, policy
    )
    existing = _load_existing_committed_initial_leaf(
        owner_private_spool_directory_path,
        lock=lock,
        profile=profile,
        registry=registry,
        expected_attempt_context_id=expected_attempt_context_id,
        policy_payload=external_witness_policy_payload,
        policy=policy,
        anchor_payload=signed_owner_status_anti_fork_anchor_payload,
        anchor=anchor,
    )
    if existing is None:
        receipt, receipt_payload = _compose_and_sign_initial_receipt(
            anchor_payload=signed_owner_status_anti_fork_anchor_payload,
            anchor=anchor,
            policy=policy,
            private_key=private_key,
        )
        (
            ledger_payload,
            ledger,
            checkpoint_payload,
            checkpoint,
            head_payload,
            head,
        ) = _compose_spool_artifacts(
            lock=lock,
            profile=profile,
            registry=registry,
            expected_attempt_context_id=expected_attempt_context_id,
            policy=policy,
            anchor_payload=signed_owner_status_anti_fork_anchor_payload,
            anchor=anchor,
            receipt_payload=receipt_payload,
            receipt=receipt,
        )
        initialization_lock_payload, basenames = _compose_initialization_lock(
            lock=lock,
            profile=profile,
            registry=registry,
            expected_attempt_context_id=expected_attempt_context_id,
            policy_payload=external_witness_policy_payload,
            policy=policy,
            anchor_payload=signed_owner_status_anti_fork_anchor_payload,
            anchor=anchor,
            receipt_payload=receipt_payload,
            receipt=receipt,
            ledger_payload=ledger_payload,
            ledger=ledger,
            checkpoint_payload=checkpoint_payload,
            checkpoint=checkpoint,
            head_payload=head_payload,
            head=head,
        )
        spool_names = _spool_initial_leaf(
            owner_private_spool_directory_path,
            initialization_lock_payload=initialization_lock_payload,
            receipt_payload=receipt_payload,
            ledger_payload=ledger_payload,
            checkpoint_payload=checkpoint_payload,
            head_payload=head_payload,
            basenames=basenames,
        )
    else:
        (
            receipt,
            receipt_payload,
            ledger,
            ledger_payload,
            checkpoint,
            checkpoint_payload,
            head,
            head_payload,
            spool_names,
        ) = existing
    artifact_ids = {
        "stage_b_trust_lock_id": lock.artifact_id,
        "authority_profile_id": profile.artifact_id,
        "signer_registry_id": registry.artifact_id,
        "external_witness_policy_id": policy["external_witness_policy_id"],
        "signed_owner_status_anti_fork_anchor_id": anchor["anchor_id"],
        "external_owner_status_anchor_receipt_id": receipt["receipt_id"],
        "external_witness_ledger_id": ledger["ledger_id"],
        "external_witness_checkpoint_id": checkpoint["checkpoint_id"],
        "external_witness_head_id": head["head_id"],
    }
    payload_sha256s = {
        "stage_b_trust_lock_payload_sha256": _payload_sha(
            stage_b_trust_lock_payload
        ),
        "authority_profile_payload_sha256": _payload_sha(
            authority_profile_payload
        ),
        "signer_registry_payload_sha256": _payload_sha(
            signer_registry_payload
        ),
        "external_witness_policy_payload_sha256": _payload_sha(
            external_witness_policy_payload
        ),
        "signed_owner_status_anti_fork_anchor_payload_sha256": _payload_sha(
            signed_owner_status_anti_fork_anchor_payload
        ),
        "external_owner_status_anchor_receipt_payload_sha256": _payload_sha(
            receipt_payload
        ),
        "external_witness_ledger_payload_sha256": _payload_sha(ledger_payload),
        "external_witness_checkpoint_payload_sha256": _payload_sha(
            checkpoint_payload
        ),
        "external_witness_head_payload_sha256": _payload_sha(head_payload),
    }
    return _issue_result(
        receipt_payload, artifact_ids, payload_sha256s, spool_names
    )


def require_issued_initial_external_witness_candidate_v1(
    value: object,
) -> IssuedInitialExternalWitnessCandidateV1:
    return _require_result(value)


def verify_real_external_witness_admission_v1(
    *args: object, **kwargs: object
) -> NoReturn:
    _fail(
        FAIL_NOT_IMPLEMENTED_V1,
        "real external control/currentness/rollback-resistant admission is not implemented",
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
issue_initial_external_owner_status_anchor_receipt_v1 = _CAPTURED[
    "_issue_initial_external_owner_status_anchor_receipt_implementation_v1"
]
require_issued_initial_external_witness_candidate_v1 = _CAPTURED[
    "require_issued_initial_external_witness_candidate_v1"
]
verify_real_external_witness_admission_v1 = _CAPTURED[
    "verify_real_external_witness_admission_v1"
]
del _CAPTURED
del _definition_captured_function_graph_v1
del _issue_initial_external_owner_status_anchor_receipt_implementation_v1
del _issue_result


__all__ = [
    "EVIDENCE_CONTRACT_SOURCE_SHA256_V1",
    "EXTERNAL_WITNESS_LEDGER_SCHEMA_V1",
    "EXTERNAL_WITNESS_CHECKPOINT_SCHEMA_V1",
    "EXTERNAL_WITNESS_HEAD_SCHEMA_V1",
    "MECHANICS_CLAIM_BOUNDARY_V1",
    "FAIL_INPUT_V1",
    "FAIL_CANONICAL_V1",
    "FAIL_SCHEMA_V1",
    "FAIL_CONTENT_ID_V1",
    "FAIL_BINDING_V1",
    "FAIL_TRUST_ANCHOR_V1",
    "FAIL_SIGNATURE_V1",
    "FAIL_KEY_V1",
    "FAIL_FILESYSTEM_V1",
    "FAIL_CLAIM_BOUNDARY_V1",
    "FAIL_NOT_IMPLEMENTED_V1",
    "Phase2BStep5R01ExternalWitnessErrorV1",
    "IssuedInitialExternalWitnessCandidateV1",
    "issue_initial_external_owner_status_anchor_receipt_v1",
    "require_issued_initial_external_witness_candidate_v1",
    "verify_real_external_witness_admission_v1",
]

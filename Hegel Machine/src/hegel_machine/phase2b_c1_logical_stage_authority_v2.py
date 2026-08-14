"""Pure owner-accepted authorization for the committed Phase-2B C1 stage wire.

This module is a conservative overlay.  It never changes the frozen six-role
wire, actor profile V2, logical-stage V1 DTOs, or their content-address domains.
It verifies a caller-supplied V1 genesis graph, an exact V2 context projecting
to that graph, one six-role unanimous authorization amendment, and a separate
STORE ``IF_ABSENT`` receipt for the V2 registry.  The wrapper then independently
re-verifies the committed logical-stage V1 prefix.

Successful verification establishes cryptographic/content-address mechanics
only.  It does not observe owner intent, execution, Docker, filesystems, time,
an actual campaign, scoring, a gate, effect, or C1 completion.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import json
import re
from typing import Final

from hegel_machine import phase2b_c1_logical_stage_wire_v1 as logical
from hegel_machine import phase2b_c1_six_role_wire_v1 as six

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey as _Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised by backend monkeypatch tests.
    _Ed25519PublicKey = None


PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION: Final = (
    "hegel-machine-phase2b-c1-logical-stage-authority/2"
)
PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_CLAIM_LEVEL: Final = (
    "CALLER_SUPPLIED_OWNER_ACCEPTED_POLICY_CRYPTOGRAPHIC_MECHANICS_ONLY"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1: Final = (
    "phase2b_c1_logical_stage_wire_schema_v1_"
    "1a625fbc369b057b9fc295fec02391d0421fc071ca41905b15901d4365833b5d"
)
PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1: Final = (
    "phase2b_c1_stage_event_contract_manifest_v1_"
    "c0929d05bdae56205e08700ab760e88a4668c686e73fe4be7a00ce6d69f9baf7"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2: Final = (
    "phase2b_c1_logical_stage_wire_policy_v2_"
    "c88653f2caa92eae1dfb7456217c4ca05c339e8f23e0540a06e435552dc59573"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_COMMIT_V1: Final = (
    "81ab887787c10882392738f70396909120b9faf6"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_BLOB_V1: Final = (
    "9b0364fee6e9f351c85e4ca2feded5c11e14ba32"
)
PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_SHA256_V1: Final = (
    "a79873f88e8907030a042b0f4b2f4da6fcfc96448df112bcf21caf182dd2e407"
)
PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SEMANTICS_V2: Final = (
    "OWNER_ACCEPTED_SIX_ROLE_UNANIMOUS_FORWARD_AUTHORIZATION_NOT_EXECUTION"
)
PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2: Final = (
    "LOGICAL_STAGE_AUTHORIZATION_AMENDMENT_ACCEPTANCE"
)

_MAXIMUM_ASCII_TEXT_BYTES: Final = 4_096
_RAW_SIGNATURE_BYTES: Final = 64
_CONTEXT_PREFIX: Final = "phase2b_c1_logical_stage_authority_context_v2_"
_AMENDMENT_PREFIX: Final = (
    "phase2b_c1_logical_stage_authorization_amendment_v2_"
)
_REGISTRY_PREFIX: Final = "phase2b_c1_owner_accepted_execution_registry_v2_"
_STORE_RECEIPT_PREFIX: Final = (
    "phase2b_c1_logical_stage_authorization_if_absent_receipt_v2_"
)
_RESULT_PREFIX: Final = "phase2b_c1_logical_stage_authority_verification_v2_"
_PREFIX_RESULT_PREFIX: Final = (
    "phase2b_c1_authorized_logical_stage_prefix_verification_v2_"
)
_CONTEXT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORITY/CONTEXT/V2\x00"
)
_AMENDMENT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/AMENDMENT_ID/V2\x00"
)
_ROLE_ACCEPTANCE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/ROLE_ACCEPTANCE/V2\x00"
)
_REGISTRY_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/REGISTRY_ID/V2\x00"
)
_STORE_SIGNATURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/STORE_IF_ABSENT/V2\x00"
)
_STORE_RECEIPT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/STORE_RECEIPT_ID/V2\x00"
)
_RESULT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/RESULT/V2\x00"
)
_PREFIX_RESULT_ID_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/LOGICAL_STAGE_AUTHORIZATION/PREFIX_RESULT/V2\x00"
)

_AUTHORIZED_STAGES: Final = (
    six.Phase2BC1LogicalStageV1.RAW_INPUT_COMMITTED,
    six.Phase2BC1LogicalStageV1.PACKAGE_COMMITMENTS_FROZEN,
    six.Phase2BC1LogicalStageV1.RUN_STARTED,
    six.Phase2BC1LogicalStageV1.RUN_FINISHED,
)
_AUTHORIZED_STAGE_INDICES: Final = (1, 2, 3, 4)


class Phase2BC1LogicalStageAuthorityDispositionV2(str, Enum):
    CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION = (
        "CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION"
    )
    REJECTED = "REJECTED"


class Phase2BC1LogicalStageAuthorityReasonV2(str, Enum):
    V2_AUTHORITY_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED = (
        "V2_AUTHORITY_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    GRAPH_SNAPSHOT_INVALID = "GRAPH_SNAPSHOT_INVALID"
    LEGACY_GRAPH_REJECTED = "LEGACY_GRAPH_REJECTED"
    CONTEXT_INVALID = "CONTEXT_INVALID"
    AMENDMENT_INVALID = "AMENDMENT_INVALID"
    REGISTRY_INVALID = "REGISTRY_INVALID"
    STORE_RECEIPT_INVALID = "STORE_RECEIPT_INVALID"
    GRAPH_BINDING_INVALID = "GRAPH_BINDING_INVALID"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Phase2BC1AuthorizedLogicalStagePrefixDispositionV2(str, Enum):
    OWNER_ACCEPTED_V2_POLICY_AND_CALLER_SUPPLIED_V1_PREFIX_VERIFIED_NOT_EXECUTION = (
        "OWNER_ACCEPTED_V2_POLICY_AND_CALLER_SUPPLIED_V1_PREFIX_VERIFIED_NOT_EXECUTION"
    )
    REJECTED = "REJECTED"


class Phase2BC1AuthorizedLogicalStagePrefixReasonV2(str, Enum):
    V2_POLICY_AND_V1_PREFIX_CRYPTOGRAPHIC_MECHANICS_VERIFIED = (
        "V2_POLICY_AND_V1_PREFIX_CRYPTOGRAPHIC_MECHANICS_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    GRAPH_SNAPSHOT_INVALID = "GRAPH_SNAPSHOT_INVALID"
    AUTHORITY_GRAPH_REJECTED = "AUTHORITY_GRAPH_REJECTED"
    LOGICAL_STAGE_PREFIX_REJECTED = "LOGICAL_STAGE_PREFIX_REJECTED"
    SIGNATURE_BACKEND_UNAVAILABLE = "SIGNATURE_BACKEND_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def _ascii_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be ASCII") from exc
    if not 1 <= len(encoded) <= _MAXIMUM_ASCII_TEXT_BYTES:
        raise ValueError(f"{name} ASCII size drift")
    return value


def _sha256(value: object, name: str) -> str:
    text = _ascii_text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _identifier(value: object, prefix: str, name: str) -> str:
    text = _ascii_text(value, name)
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    _sha256(text[len(prefix):], f"{name} suffix")
    return text


def _image_digest(value: object, name: str) -> str:
    text = _ascii_text(value, name)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be exact image digest")
    return text


def _exact_integer(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} integer drift")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _raw_bytes(value: object, length: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        raise TypeError(f"{name} must be exact {length}-byte bytes")
    return bytes(value)


def _canonical_named_bytes(pairs: tuple[tuple[str, object], ...]) -> bytes:
    return json.dumps(
        pairs, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + _canonical_named_bytes(pairs)).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageAuthorityContextBodyV2:
    version: str
    authority_class: str
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    execution_freeze_manifest_sha256: str
    attempt_nonce_sha256: str
    attempt_id: str
    recognizer_image_digest: str
    recognizer_source_root_sha256: str
    scorer_image_digest: str
    scorer_source_root_sha256: str
    legacy_context_id: str
    legacy_execution_registry_id: str
    logical_stage_wire_schema_id: str
    logical_stage_wire_policy_id: str
    stage_event_contract_manifest_id: str

    def __post_init__(self) -> None:
        _snapshot_context_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageAuthorityContextV2:
    body: Phase2BC1LogicalStageAuthorityContextBodyV2
    authority_context_id: str

    def __post_init__(self) -> None:
        _snapshot_context(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageAuthorizationAmendmentBodyV2:
    version: str
    authority_context_id: str
    legacy_context_id: str
    legacy_execution_registry_id: str
    authority_class: str
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    attempt_id: str
    logical_stage_wire_schema_id: str
    logical_stage_wire_policy_id: str
    stage_event_contract_manifest_id: str
    authorized_stage_order: tuple[six.Phase2BC1LogicalStageV1, ...]
    authorized_stage_indices: tuple[int, ...]
    required_signer_roles: tuple[six.Phase2BC1SixRoleV1, ...]
    required_signer_key_ids: tuple[str, ...]
    authorization_semantics: str
    legacy_token_authorization_field_must_remain_unchanged: bool
    legacy_actual_campaign_eligible_must_remain_false: bool
    actual_execution_claimed: bool

    def __post_init__(self) -> None:
        _snapshot_amendment_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageAuthorizationSignatureV2:
    purpose_id: int
    role: six.Phase2BC1SixRoleV1
    signature_purpose: str
    algorithm: str
    key_epoch: int
    signer_key_id: str
    signature: bytes

    def __post_init__(self) -> None:
        _snapshot_authorization_signature(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1OwnerAcceptedExecutionRegistryV2:
    body: Phase2BC1LogicalStageAuthorizationAmendmentBodyV2
    authorization_amendment_id: str
    role_acceptance_signatures: tuple[
        Phase2BC1LogicalStageAuthorizationSignatureV2, ...
    ]
    execution_registry_id: str

    def __post_init__(self) -> None:
        _snapshot_registry(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2:
    version: str
    authority_context_id: str
    legacy_context_id: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    object_key: str
    previous_generation: int
    new_generation: int
    condition: str
    previous_content_sha256_or_null: str | None
    stored_content_sha256: str

    def __post_init__(self) -> None:
        _snapshot_store_body(self)


@dataclass(frozen=True, slots=True)
class Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2:
    body: Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2
    store_signature: six.Phase2BC1DetachedSignatureV1
    store_receipt_id: str

    def __post_init__(self) -> None:
        _snapshot_store_receipt(self)


def _snapshot_context_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageAuthorityContextBodyV2:
        raise TypeError("authority context body exact type drift")
    raw = {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(type(value))
    }
    return {
        "version": _ascii_text(raw["version"], "context version"),
        "authority_class": _ascii_text(raw["authority_class"], "authority class"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"],
            "phase2b_c1_container_actor_profile_v2_",
            "actor profile content ID",
        ),
        "actor_profile_canonical_sha256": _sha256(
            raw["actor_profile_canonical_sha256"], "actor profile canonical SHA"
        ),
        "campaign_id": _identifier(
            raw["campaign_id"], "phase2b_actual_unsealed_960_campaign_v2_", "campaign ID"
        ),
        "protocol_id": _identifier(raw["protocol_id"], "phase2b_protocol_", "protocol ID"),
        "exact_freeze_id": _identifier(
            raw["exact_freeze_id"], "phase2b_exact_freeze_", "exact-freeze ID"
        ),
        "execution_freeze_manifest_id": _identifier(
            raw["execution_freeze_manifest_id"],
            "phase2b_execution_freeze_",
            "execution-freeze ID",
        ),
        "execution_freeze_manifest_sha256": _sha256(
            raw["execution_freeze_manifest_sha256"], "execution-freeze SHA"
        ),
        "attempt_nonce_sha256": _sha256(raw["attempt_nonce_sha256"], "attempt nonce SHA"),
        "attempt_id": _identifier(
            raw["attempt_id"], "phase2b_actual_unsealed_960_attempt_v2_", "attempt ID"
        ),
        "recognizer_image_digest": _image_digest(
            raw["recognizer_image_digest"], "recognizer image digest"
        ),
        "recognizer_source_root_sha256": _sha256(
            raw["recognizer_source_root_sha256"], "recognizer source root"
        ),
        "scorer_image_digest": _image_digest(raw["scorer_image_digest"], "scorer image digest"),
        "scorer_source_root_sha256": _sha256(
            raw["scorer_source_root_sha256"], "scorer source root"
        ),
        "legacy_context_id": _identifier(
            raw["legacy_context_id"],
            "phase2b_c1_six_role_ceremony_context_v1_",
            "legacy context ID",
        ),
        "legacy_execution_registry_id": _identifier(
            raw["legacy_execution_registry_id"],
            "phase2b_c1_owner_accepted_execution_registry_v1_",
            "legacy registry ID",
        ),
        "logical_stage_wire_schema_id": _identifier(
            raw["logical_stage_wire_schema_id"],
            "phase2b_c1_logical_stage_wire_schema_v1_",
            "logical-stage schema ID",
        ),
        "logical_stage_wire_policy_id": _identifier(
            raw["logical_stage_wire_policy_id"],
            "phase2b_c1_logical_stage_wire_policy_v2_",
            "logical-stage policy ID",
        ),
        "stage_event_contract_manifest_id": _identifier(
            raw["stage_event_contract_manifest_id"],
            "phase2b_c1_stage_event_contract_manifest_v1_",
            "stage-event manifest ID",
        ),
    }


def _snapshot_context(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageAuthorityContextV2:
        raise TypeError("authority context exact type drift")
    return {
        "body": _snapshot_context_body(object.__getattribute__(value, "body")),
        "authority_context_id": _identifier(
            object.__getattribute__(value, "authority_context_id"),
            _CONTEXT_PREFIX,
            "authority context ID",
        ),
    }


def _snapshot_amendment_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageAuthorizationAmendmentBodyV2:
        raise TypeError("authorization amendment body exact type drift")
    raw = {
        item.name: object.__getattribute__(value, item.name)
        for item in fields(type(value))
    }
    if type(raw["authorized_stage_order"]) is not tuple or any(
        type(item) is not six.Phase2BC1LogicalStageV1
        for item in raw["authorized_stage_order"]
    ):
        raise TypeError("authorized stage order exact tuple/enum drift")
    if type(raw["authorized_stage_indices"]) is not tuple:
        raise TypeError("authorized stage indices exact tuple drift")
    if type(raw["required_signer_roles"]) is not tuple or any(
        type(item) is not six.Phase2BC1SixRoleV1
        for item in raw["required_signer_roles"]
    ):
        raise TypeError("required signer roles exact tuple/enum drift")
    if type(raw["required_signer_key_ids"]) is not tuple:
        raise TypeError("required signer key IDs exact tuple drift")
    return {
        "version": _ascii_text(raw["version"], "amendment version"),
        "authority_context_id": _identifier(
            raw["authority_context_id"], _CONTEXT_PREFIX, "amendment context ID"
        ),
        "legacy_context_id": _identifier(
            raw["legacy_context_id"],
            "phase2b_c1_six_role_ceremony_context_v1_",
            "amendment legacy context ID",
        ),
        "legacy_execution_registry_id": _identifier(
            raw["legacy_execution_registry_id"],
            "phase2b_c1_owner_accepted_execution_registry_v1_",
            "amendment legacy registry ID",
        ),
        "authority_class": _ascii_text(raw["authority_class"], "amendment authority class"),
        "authority_requirement_id": _identifier(
            raw["authority_requirement_id"],
            "phase2b_c1_owner_accepted_container_authority_requirement_v1_",
            "amendment authority requirement ID",
        ),
        "actor_profile_content_id": _identifier(
            raw["actor_profile_content_id"],
            "phase2b_c1_container_actor_profile_v2_",
            "amendment actor profile content ID",
        ),
        "actor_profile_canonical_sha256": _sha256(
            raw["actor_profile_canonical_sha256"], "amendment profile SHA"
        ),
        "campaign_id": _identifier(
            raw["campaign_id"], "phase2b_actual_unsealed_960_campaign_v2_", "amendment campaign ID"
        ),
        "attempt_id": _identifier(
            raw["attempt_id"], "phase2b_actual_unsealed_960_attempt_v2_", "amendment attempt ID"
        ),
        "logical_stage_wire_schema_id": _identifier(
            raw["logical_stage_wire_schema_id"],
            "phase2b_c1_logical_stage_wire_schema_v1_",
            "amendment schema ID",
        ),
        "logical_stage_wire_policy_id": _identifier(
            raw["logical_stage_wire_policy_id"],
            "phase2b_c1_logical_stage_wire_policy_v2_",
            "amendment policy ID",
        ),
        "stage_event_contract_manifest_id": _identifier(
            raw["stage_event_contract_manifest_id"],
            "phase2b_c1_stage_event_contract_manifest_v1_",
            "amendment manifest ID",
        ),
        "authorized_stage_order": tuple(raw["authorized_stage_order"]),
        "authorized_stage_indices": tuple(
            _exact_integer(item, "authorized stage index", maximum=9)
            for item in raw["authorized_stage_indices"]
        ),
        "required_signer_roles": tuple(raw["required_signer_roles"]),
        "required_signer_key_ids": tuple(
            _identifier(item, "phase2b_c1_six_role_key_v1_", "required signer key ID")
            for item in raw["required_signer_key_ids"]
        ),
        "authorization_semantics": _ascii_text(
            raw["authorization_semantics"], "authorization semantics"
        ),
        "legacy_token_authorization_field_must_remain_unchanged": _exact_bool(
            raw["legacy_token_authorization_field_must_remain_unchanged"],
            "legacy token authorization preservation",
        ),
        "legacy_actual_campaign_eligible_must_remain_false": _exact_bool(
            raw["legacy_actual_campaign_eligible_must_remain_false"],
            "legacy actual eligibility preservation",
        ),
        "actual_execution_claimed": _exact_bool(
            raw["actual_execution_claimed"], "actual execution claim"
        ),
    }


def _snapshot_authorization_signature(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageAuthorizationSignatureV2:
        raise TypeError("authorization signature exact type drift")
    if type(value.role) is not six.Phase2BC1SixRoleV1:
        raise TypeError("authorization signature role exact enum drift")
    return {
        "purpose_id": _exact_integer(value.purpose_id, "signature purpose ID", maximum=6),
        "role": value.role,
        "signature_purpose": _ascii_text(value.signature_purpose, "signature purpose"),
        "algorithm": _ascii_text(value.algorithm, "signature algorithm"),
        "key_epoch": _exact_integer(value.key_epoch, "signature key epoch", maximum=2**31 - 1),
        "signer_key_id": _identifier(
            value.signer_key_id, "phase2b_c1_six_role_key_v1_", "signature key ID"
        ),
        "signature": _raw_bytes(value.signature, _RAW_SIGNATURE_BYTES, "authorization signature"),
    }


def _snapshot_registry(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1OwnerAcceptedExecutionRegistryV2:
        raise TypeError("V2 execution registry exact type drift")
    signatures = object.__getattribute__(value, "role_acceptance_signatures")
    if type(signatures) is not tuple:
        raise TypeError("registry signatures must be an exact tuple")
    return {
        "body": _snapshot_amendment_body(object.__getattribute__(value, "body")),
        "authorization_amendment_id": _identifier(
            object.__getattribute__(value, "authorization_amendment_id"),
            _AMENDMENT_PREFIX,
            "authorization amendment ID",
        ),
        "role_acceptance_signatures": tuple(
            _snapshot_authorization_signature(item) for item in signatures
        ),
        "execution_registry_id": _identifier(
            object.__getattribute__(value, "execution_registry_id"),
            _REGISTRY_PREFIX,
            "V2 execution registry ID",
        ),
    }


def _snapshot_store_body(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2:
        raise TypeError("V2 STORE receipt body exact type drift")
    previous = value.previous_content_sha256_or_null
    if previous is not None:
        previous = _sha256(previous, "previous stored content SHA")
    return {
        "version": _ascii_text(value.version, "STORE version"),
        "authority_context_id": _identifier(value.authority_context_id, _CONTEXT_PREFIX, "STORE context ID"),
        "legacy_context_id": _identifier(
            value.legacy_context_id,
            "phase2b_c1_six_role_ceremony_context_v1_",
            "STORE legacy context ID",
        ),
        "campaign_id": _identifier(
            value.campaign_id, "phase2b_actual_unsealed_960_campaign_v2_", "STORE campaign ID"
        ),
        "attempt_id": _identifier(
            value.attempt_id, "phase2b_actual_unsealed_960_attempt_v2_", "STORE attempt ID"
        ),
        "execution_registry_id": _identifier(
            value.execution_registry_id, _REGISTRY_PREFIX, "STORE registry ID"
        ),
        "object_key": _ascii_text(value.object_key, "STORE object key"),
        "previous_generation": _exact_integer(
            value.previous_generation, "previous generation", maximum=2**63 - 1
        ),
        "new_generation": _exact_integer(
            value.new_generation, "new generation", maximum=2**63 - 1
        ),
        "condition": _ascii_text(value.condition, "STORE condition"),
        "previous_content_sha256_or_null": previous,
        "stored_content_sha256": _sha256(value.stored_content_sha256, "stored content SHA"),
    }


def _snapshot_legacy_signature(value: object) -> dict[str, object]:
    if type(value) is not six.Phase2BC1DetachedSignatureV1:
        raise TypeError("legacy detached signature exact type drift")
    if type(value.role) is not six.Phase2BC1SixRoleV1 or type(
        value.signature_purpose
    ) is not six.Phase2BC1SignaturePurposeV1:
        raise TypeError("legacy detached signature enum drift")
    return {
        "purpose_id": _exact_integer(value.purpose_id, "STORE signature purpose ID", maximum=6),
        "role": value.role,
        "signature_purpose": value.signature_purpose,
        "algorithm": _ascii_text(value.algorithm, "STORE signature algorithm"),
        "key_epoch": _exact_integer(value.key_epoch, "STORE signature key epoch", maximum=2**31 - 1),
        "signer_key_id": _identifier(
            value.signer_key_id, "phase2b_c1_six_role_key_v1_", "STORE signer key ID"
        ),
        "signature": _raw_bytes(value.signature, _RAW_SIGNATURE_BYTES, "STORE signature"),
    }


def _snapshot_store_receipt(value: object) -> dict[str, object]:
    if type(value) is not Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2:
        raise TypeError("V2 STORE receipt candidate exact type drift")
    return {
        "body": _snapshot_store_body(object.__getattribute__(value, "body")),
        "store_signature": _snapshot_legacy_signature(
            object.__getattribute__(value, "store_signature")
        ),
        "store_receipt_id": _identifier(
            object.__getattribute__(value, "store_receipt_id"),
            _STORE_RECEIPT_PREFIX,
            "V2 STORE receipt ID",
        ),
    }


def _enum_pairs(values: dict[str, object]) -> tuple[tuple[str, object], ...]:
    pairs: list[tuple[str, object]] = []
    for name, value in values.items():
        if type(value) is tuple and value and isinstance(value[0], Enum):
            pairs.append((name, tuple(item.value for item in value)))
        elif isinstance(value, Enum):
            pairs.append((name, value.value))
        else:
            pairs.append((name, value))
    return tuple(pairs)


def _authorization_signature_pairs(value: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return (
        ("purpose_id", value["purpose_id"]),
        ("role", value["role"].value),
        ("signature_purpose", value["signature_purpose"]),
        ("algorithm", value["algorithm"]),
        ("key_epoch", value["key_epoch"]),
        ("signer_key_id", value["signer_key_id"]),
        ("signature_hex", value["signature"].hex()),
    )


def _legacy_signature_pairs(value: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return (
        ("purpose_id", value["purpose_id"]),
        ("role", value["role"].value),
        ("signature_purpose", value["signature_purpose"].value),
        ("algorithm", value["algorithm"]),
        ("key_epoch", value["key_epoch"]),
        ("signer_key_id", value["signer_key_id"]),
        ("signature_hex", value["signature"].hex()),
    )


def phase2b_c1_logical_stage_authority_context_id_v2(
    body: Phase2BC1LogicalStageAuthorityContextBodyV2,
) -> str:
    closed = _snapshot_context_body(body)
    return _CONTEXT_PREFIX + _digest(_CONTEXT_DOMAIN, tuple(closed.items()))


def project_phase2b_c1_logical_stage_authority_context_to_v1(
    context: Phase2BC1LogicalStageAuthorityContextV2,
) -> six.Phase2BC1CeremonyContextV1:
    closed = _snapshot_context(context)
    body = closed["body"]
    projected_values = {name: body[name] for name in (
            "authority_class", "authority_requirement_id",
            "actor_profile_content_id", "actor_profile_canonical_sha256",
            "campaign_id", "protocol_id", "exact_freeze_id",
            "execution_freeze_manifest_id", "execution_freeze_manifest_sha256",
            "attempt_nonce_sha256", "attempt_id", "recognizer_image_digest",
            "recognizer_source_root_sha256", "scorer_image_digest",
            "scorer_source_root_sha256",
        )}
    projected_body = six.Phase2BC1CeremonyContextBodyV1(
        version=six.PHASE2B_C1_SIX_ROLE_WIRE_V1_VERSION,
        **projected_values,
    )
    return six.Phase2BC1CeremonyContextV1(
        body=projected_body,
        context_id=six.phase2b_c1_ceremony_context_id_v1(projected_body),
    )


def phase2b_c1_logical_stage_authorization_amendment_id_v2(
    body: Phase2BC1LogicalStageAuthorizationAmendmentBodyV2,
) -> str:
    closed = _snapshot_amendment_body(body)
    return _AMENDMENT_PREFIX + _digest(_AMENDMENT_ID_DOMAIN, _enum_pairs(closed))


def phase2b_c1_logical_stage_authorization_signature_preimage_v2(
    body: Phase2BC1LogicalStageAuthorizationAmendmentBodyV2,
    *,
    purpose_id: int,
    role: six.Phase2BC1SixRoleV1,
    signer_key_id: str,
) -> bytes:
    closed = _snapshot_amendment_body(body)
    purpose = _exact_integer(purpose_id, "authorization purpose ID", maximum=6)
    if type(role) is not six.Phase2BC1SixRoleV1:
        raise TypeError("authorization role exact enum drift")
    signer = _identifier(
        signer_key_id, "phase2b_c1_six_role_key_v1_", "authorization signer key ID"
    )
    header = (
        ("purpose_id", purpose),
        ("role", role.value),
        ("signature_purpose", PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2),
        ("algorithm", "Ed25519"),
        ("key_epoch", six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH),
        ("signer_key_id", signer),
    )
    return _ROLE_ACCEPTANCE_SIGNATURE_DOMAIN + _canonical_named_bytes((
        ("signature_header", header),
        ("body", _enum_pairs(closed)),
    ))


def phase2b_c1_owner_accepted_execution_registry_id_v2(
    body: Phase2BC1LogicalStageAuthorizationAmendmentBodyV2,
    authorization_amendment_id: str,
    role_acceptance_signatures: tuple[
        Phase2BC1LogicalStageAuthorizationSignatureV2, ...
    ],
) -> str:
    closed_body = _snapshot_amendment_body(body)
    amendment_id = _identifier(
        authorization_amendment_id, _AMENDMENT_PREFIX, "authorization amendment ID"
    )
    if type(role_acceptance_signatures) is not tuple:
        raise TypeError("role acceptance signatures must be an exact tuple")
    signatures = tuple(
        _snapshot_authorization_signature(item) for item in role_acceptance_signatures
    )
    return _REGISTRY_PREFIX + _digest(_REGISTRY_ID_DOMAIN, (
        ("body", _enum_pairs(closed_body)),
        ("authorization_amendment_id", amendment_id),
        ("role_acceptance_signatures", tuple(
            _authorization_signature_pairs(item) for item in signatures
        )),
    ))


def canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2(
    registry: Phase2BC1OwnerAcceptedExecutionRegistryV2,
) -> bytes:
    closed = _snapshot_registry(registry)
    return _canonical_named_bytes((
        ("body", _enum_pairs(closed["body"])),
        ("authorization_amendment_id", closed["authorization_amendment_id"]),
        ("role_acceptance_signatures", tuple(
            _authorization_signature_pairs(item)
            for item in closed["role_acceptance_signatures"]
        )),
        ("execution_registry_id", closed["execution_registry_id"]),
    ))


def phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
    body: Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2,
    *,
    signer_key_id: str,
) -> bytes:
    closed = _snapshot_store_body(body)
    signer = _identifier(
        signer_key_id, "phase2b_c1_six_role_key_v1_", "STORE signer key ID"
    )
    header = (
        ("purpose_id", 3),
        ("role", six.Phase2BC1SixRoleV1.STORE_RECEIPT.value),
        ("signature_purpose", six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT.value),
        ("algorithm", "Ed25519"),
        ("key_epoch", six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH),
        ("signer_key_id", signer),
    )
    return _STORE_SIGNATURE_DOMAIN + _canonical_named_bytes((
        ("signature_header", header),
        ("body", tuple(closed.items())),
    ))


def phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
    body: Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2,
    store_signature: six.Phase2BC1DetachedSignatureV1,
) -> str:
    closed_body = _snapshot_store_body(body)
    closed_signature = _snapshot_legacy_signature(store_signature)
    return _STORE_RECEIPT_PREFIX + _digest(_STORE_RECEIPT_ID_DOMAIN, (
        ("body", tuple(closed_body.items())),
        ("store_signature", _legacy_signature_pairs(closed_signature)),
    ))


class _BackendUnavailable(Exception):
    pass


class _SignatureInvalid(Exception):
    pass


def _verify_ed25519(public_key: bytes, signature: bytes, message: bytes) -> None:
    if _Ed25519PublicKey is None:
        raise _BackendUnavailable
    try:
        _Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except Exception as exc:
        raise _SignatureInvalid from exc


_AUTHORITY_TRUE_CLAIMS: Final = (
    "complete_graph_snapshotted_before_verification",
    "legacy_six_role_graph_reverified",
    "authority_context_content_address_verified",
    "authority_context_conservative_projection_verified",
    "exact_contract_identity_triple_verified",
    "authorization_amendment_content_address_verified",
    "six_unanimous_role_acceptance_signatures_verified",
    "v2_execution_registry_content_address_verified",
    "v2_store_if_absent_transition_verified",
    "v2_store_receipt_signature_verified",
    "v2_store_receipt_content_address_verified",
    "campaign_attempt_profile_registry_bindings_verified",
    "v2_policy_admission_verified",
    "verification_result_privately_issued",
)
_FALSE_CLAIMS: Final = (
    "legacy_registry_policy_authorized",
    "actual_campaign_eligible",
    "actual_execution_claimed",
    "execution_authorized",
    "owner_acceptance_observed",
    "authorization_precedes_stage1_globally_verified",
    "docker_invoked",
    "filesystem_read",
    "filesystem_written",
    "keys_generated",
    "signatures_generated",
    "time_operation_executed",
    "timeline_observed",
    "run_start_observed",
    "run_finish_observed",
    "recognizer_executed",
    "runtime_executed",
    "scoring_performed",
    "prediction_scored",
    "formal_gate_evaluation_performed",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
    "cross_host_global_one_shot_verified",
)
_AUTHORITY_RESULT_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1LogicalStageAuthorityVerificationV2:
    disposition: Phase2BC1LogicalStageAuthorityDispositionV2
    reason: Phase2BC1LogicalStageAuthorityReasonV2
    version: str
    claim_level: str
    result_id: str | None
    authority_context_id: str | None
    legacy_context_id: str | None
    legacy_execution_registry_id: str | None
    execution_registry_id: str | None
    authorization_amendment_id: str | None
    store_receipt_id: str | None
    logical_stage_wire_schema_id: str | None
    logical_stage_wire_policy_id: str | None
    stage_event_contract_manifest_id: str | None
    role_key_ids: tuple[str, ...]
    partial_output_published: bool
    complete_graph_snapshotted_before_verification: bool
    legacy_six_role_graph_reverified: bool
    authority_context_content_address_verified: bool
    authority_context_conservative_projection_verified: bool
    exact_contract_identity_triple_verified: bool
    authorization_amendment_content_address_verified: bool
    six_unanimous_role_acceptance_signatures_verified: bool
    v2_execution_registry_content_address_verified: bool
    v2_store_if_absent_transition_verified: bool
    v2_store_receipt_signature_verified: bool
    v2_store_receipt_content_address_verified: bool
    campaign_attempt_profile_registry_bindings_verified: bool
    v2_policy_admission_verified: bool
    verification_result_privately_issued: bool
    legacy_registry_policy_authorized: bool
    actual_campaign_eligible: bool
    actual_execution_claimed: bool
    execution_authorized: bool
    owner_acceptance_observed: bool
    authorization_precedes_stage1_globally_verified: bool
    docker_invoked: bool
    filesystem_read: bool
    filesystem_written: bool
    keys_generated: bool
    signatures_generated: bool
    time_operation_executed: bool
    timeline_observed: bool
    run_start_observed: bool
    run_finish_observed: bool
    recognizer_executed: bool
    runtime_executed: bool
    scoring_performed: bool
    prediction_scored: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool
    cross_host_global_one_shot_verified: bool

    def __new__(cls, token: object = None) -> "Phase2BC1LogicalStageAuthorityVerificationV2":
        if token is not _AUTHORITY_RESULT_TOKEN:
            raise TypeError("authority verification results are privately issued")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _AUTHORITY_RESULT_TOKEN:
            raise TypeError("authority verification results are privately issued")


def _issue_authority_result(
    *,
    disposition: Phase2BC1LogicalStageAuthorityDispositionV2,
    reason: Phase2BC1LogicalStageAuthorityReasonV2,
    context: Phase2BC1LogicalStageAuthorityContextV2 | None = None,
    registry: Phase2BC1OwnerAcceptedExecutionRegistryV2 | None = None,
    store: Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2 | None = None,
) -> Phase2BC1LogicalStageAuthorityVerificationV2:
    success = disposition is Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION
    metadata: list[tuple[str, object]] = [
        ("disposition", disposition),
        ("reason", reason),
        ("version", PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION),
        ("claim_level", PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_CLAIM_LEVEL),
    ]
    if success:
        if context is None or registry is None or store is None:
            raise RuntimeError("authority success lacks graph")
        body = context.body
        role_ids = registry.body.required_signer_key_ids
        result_id = _RESULT_PREFIX + _digest(_RESULT_ID_DOMAIN, (
            ("version", PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION),
            ("authority_context_id", context.authority_context_id),
            ("legacy_context_id", body.legacy_context_id),
            ("legacy_execution_registry_id", body.legacy_execution_registry_id),
            ("execution_registry_id", registry.execution_registry_id),
            ("authorization_amendment_id", registry.authorization_amendment_id),
            ("store_receipt_id", store.store_receipt_id),
        ))
        values = (
            result_id, context.authority_context_id, body.legacy_context_id,
            body.legacy_execution_registry_id, registry.execution_registry_id,
            registry.authorization_amendment_id, store.store_receipt_id,
            body.logical_stage_wire_schema_id, body.logical_stage_wire_policy_id,
            body.stage_event_contract_manifest_id, role_ids,
        )
    else:
        values = (None,) * 10 + ((),)
    metadata.extend(zip((
        "result_id", "authority_context_id", "legacy_context_id",
        "legacy_execution_registry_id", "execution_registry_id",
        "authorization_amendment_id", "store_receipt_id",
        "logical_stage_wire_schema_id", "logical_stage_wire_policy_id",
        "stage_event_contract_manifest_id", "role_key_ids",
    ), values, strict=True))
    metadata.append(("partial_output_published", False))
    metadata.extend((name, success) for name in _AUTHORITY_TRUE_CLAIMS)
    metadata.extend((name, False) for name in _FALSE_CLAIMS)
    result = Phase2BC1LogicalStageAuthorityVerificationV2(_AUTHORITY_RESULT_TOKEN)
    if tuple(name for name, _ in metadata) != tuple(
        item.name for item in fields(type(result))
    ):
        raise RuntimeError("authority result field drift")
    for name, value in metadata:
        object.__setattr__(result, name, value)
    return result


def _reject_authority(
    reason: Phase2BC1LogicalStageAuthorityReasonV2,
) -> Phase2BC1LogicalStageAuthorityVerificationV2:
    return _issue_authority_result(
        disposition=Phase2BC1LogicalStageAuthorityDispositionV2.REJECTED,
        reason=reason,
    )


def _validate_authority_bindings(
    *,
    legacy_context: six.Phase2BC1CeremonyContextV1,
    legacy_registry: six.Phase2BC1ExecutionRegistryCandidateV1,
    authority_context: Phase2BC1LogicalStageAuthorityContextV2,
    registry: Phase2BC1OwnerAcceptedExecutionRegistryV2,
    store: Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2,
) -> None:
    closed_context = _snapshot_context(authority_context)
    body = closed_context["body"]
    if (
        body["version"] != PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION
        or body["authority_class"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_CLASS
        or body["authority_requirement_id"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_AUTHORITY_REQUIREMENT_ID
        or body["actor_profile_content_id"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CONTENT_ID
        or body["actor_profile_canonical_sha256"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ACTOR_PROFILE_CANONICAL_SHA256
        or body["logical_stage_wire_schema_id"] != PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1
        or body["logical_stage_wire_policy_id"] != PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2
        or body["stage_event_contract_manifest_id"] != PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1
        or body["legacy_context_id"] != legacy_context.context_id
        or body["legacy_execution_registry_id"] != legacy_registry.execution_registry_id
        or authority_context.authority_context_id != phase2b_c1_logical_stage_authority_context_id_v2(authority_context.body)
    ):
        raise ValueError("authority context frozen identity drift")
    projected = project_phase2b_c1_logical_stage_authority_context_to_v1(authority_context)
    if projected != legacy_context:
        raise ValueError("authority context conservative projection drift")

    closed_registry = _snapshot_registry(registry)
    amendment = closed_registry["body"]
    expected_amendment = {
        "version": PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION,
        "authority_context_id": authority_context.authority_context_id,
        "legacy_context_id": legacy_context.context_id,
        "legacy_execution_registry_id": legacy_registry.execution_registry_id,
        "authority_class": body["authority_class"],
        "authority_requirement_id": body["authority_requirement_id"],
        "actor_profile_content_id": body["actor_profile_content_id"],
        "actor_profile_canonical_sha256": body["actor_profile_canonical_sha256"],
        "campaign_id": body["campaign_id"],
        "attempt_id": body["attempt_id"],
        "logical_stage_wire_schema_id": PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1,
        "logical_stage_wire_policy_id": PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2,
        "stage_event_contract_manifest_id": PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1,
        "authorized_stage_order": _AUTHORIZED_STAGES,
        "authorized_stage_indices": _AUTHORIZED_STAGE_INDICES,
        "required_signer_roles": six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER,
        "required_signer_key_ids": tuple(key.key_id for key in legacy_registry.body.role_keys),
        "authorization_semantics": PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SEMANTICS_V2,
        "legacy_token_authorization_field_must_remain_unchanged": True,
        "legacy_actual_campaign_eligible_must_remain_false": True,
        "actual_execution_claimed": False,
    }
    if amendment != expected_amendment:
        raise ValueError("authorization amendment binding drift")
    expected_amendment_id = phase2b_c1_logical_stage_authorization_amendment_id_v2(registry.body)
    if registry.authorization_amendment_id != expected_amendment_id:
        raise ValueError("authorization amendment content address drift")
    signatures = closed_registry["role_acceptance_signatures"]
    if len(signatures) != 6:
        raise ValueError("authorization registry must contain exactly six signatures")
    for purpose, (role, key, signature) in enumerate(zip(
        six.PHASE2B_C1_SIX_ROLE_WIRE_V1_ROLE_ORDER,
        legacy_registry.body.role_keys,
        signatures,
        strict=True,
    ), start=1):
        if (
            signature["purpose_id"] != purpose
            or signature["role"] is not role
            or signature["signature_purpose"] != PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2
            or signature["algorithm"] != "Ed25519"
            or signature["key_epoch"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH
            or signature["signer_key_id"] != key.key_id
        ):
            raise ValueError("ordered unanimous role signature metadata drift")
    expected_registry_id = phase2b_c1_owner_accepted_execution_registry_id_v2(
        registry.body,
        registry.authorization_amendment_id,
        registry.role_acceptance_signatures,
    )
    if registry.execution_registry_id != expected_registry_id:
        raise ValueError("V2 execution registry content address drift")

    closed_store = _snapshot_store_receipt(store)
    store_body = closed_store["body"]
    expected_object_key = (
        "phase2b/c1/actual-unsealed-960/"
        f"{body['campaign_id']}/attempt-0/logical-stage-authorization-registry-v2"
    )
    expected_stored_sha = hashlib.sha256(
        canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2(registry)
    ).hexdigest()
    if (
        store_body["version"] != PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION
        or store_body["authority_context_id"] != authority_context.authority_context_id
        or store_body["legacy_context_id"] != legacy_context.context_id
        or store_body["campaign_id"] != body["campaign_id"]
        or store_body["attempt_id"] != body["attempt_id"]
        or store_body["execution_registry_id"] != registry.execution_registry_id
        or store_body["object_key"] != expected_object_key
        or store_body["previous_generation"] != 0
        or store_body["new_generation"] != 1
        or store_body["condition"] != "IF_ABSENT"
        or store_body["previous_content_sha256_or_null"] is not None
        or store_body["stored_content_sha256"] != expected_stored_sha
    ):
        raise ValueError("V2 STORE receipt graph or transition drift")
    signature = closed_store["store_signature"]
    store_key = legacy_registry.body.role_keys[2]
    if (
        signature["purpose_id"] != 3
        or signature["role"] is not six.Phase2BC1SixRoleV1.STORE_RECEIPT
        or signature["signature_purpose"] is not six.Phase2BC1SignaturePurposeV1.STORE_IF_ABSENT_RECEIPT
        or signature["algorithm"] != "Ed25519"
        or signature["key_epoch"] != six.PHASE2B_C1_SIX_ROLE_WIRE_V1_KEY_EPOCH
        or signature["signer_key_id"] != store_key.key_id
    ):
        raise ValueError("V2 STORE signature metadata drift")
    if store.store_receipt_id != phase2b_c1_logical_stage_authorization_store_receipt_id_v2(
        store.body, store.store_signature
    ):
        raise ValueError("V2 STORE receipt content address drift")


_ORDINARY_INPUT_ERRORS: Final = (
    AttributeError, TypeError, ValueError, OverflowError, UnicodeError,
)


def verify_phase2b_c1_logical_stage_authority_graph_v2(
    *,
    legacy_context: six.Phase2BC1CeremonyContextV1,
    legacy_registry: six.Phase2BC1ExecutionRegistryCandidateV1,
    legacy_store_receipt: six.Phase2BC1StoreIfAbsentReceiptCandidateV1,
    legacy_logical_genesis_token: six.Phase2BC1LogicalGenesisTokenCandidateV1,
    authority_context: Phase2BC1LogicalStageAuthorityContextV2,
    execution_registry: Phase2BC1OwnerAcceptedExecutionRegistryV2,
    authorization_store_receipt: Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2,
) -> Phase2BC1LogicalStageAuthorityVerificationV2:
    if (
        type(legacy_context) is not six.Phase2BC1CeremonyContextV1
        or type(legacy_registry) is not six.Phase2BC1ExecutionRegistryCandidateV1
        or type(legacy_store_receipt) is not six.Phase2BC1StoreIfAbsentReceiptCandidateV1
        or type(legacy_logical_genesis_token) is not six.Phase2BC1LogicalGenesisTokenCandidateV1
        or type(authority_context) is not Phase2BC1LogicalStageAuthorityContextV2
        or type(execution_registry) is not Phase2BC1OwnerAcceptedExecutionRegistryV2
        or type(authorization_store_receipt) is not Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2
    ):
        return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.WRONG_INPUT_TYPE)
    try:
        try:
            closed = copy.deepcopy((
                legacy_context, legacy_registry, legacy_store_receipt,
                legacy_logical_genesis_token, authority_context,
                execution_registry, authorization_store_receipt,
            ))
        except Exception:
            return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.GRAPH_SNAPSHOT_INVALID)
        (
            legacy_context, legacy_registry, legacy_store_receipt,
            legacy_logical_genesis_token, authority_context,
            execution_registry, authorization_store_receipt,
        ) = closed
        legacy = six.verify_phase2b_c1_six_role_wire_v1(
            context=legacy_context,
            registry=legacy_registry,
            store_receipt=legacy_store_receipt,
            logical_genesis_token=legacy_logical_genesis_token,
        )
        if legacy.disposition is not six.Phase2BC1SixRoleWireDispositionV1.CALLER_SUPPLIED_SIX_ROLE_WIRE_GRAPH_VERIFIED_NOT_AUTHORITY:
            if legacy.reason is six.Phase2BC1SixRoleWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE:
                return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
            return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.LEGACY_GRAPH_REJECTED)
        try:
            _validate_authority_bindings(
                legacy_context=legacy_context,
                legacy_registry=legacy_registry,
                authority_context=authority_context,
                registry=execution_registry,
                store=authorization_store_receipt,
            )
        except _ORDINARY_INPUT_ERRORS:
            return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.GRAPH_BINDING_INVALID)
        try:
            for key, signature in zip(
                legacy_registry.body.role_keys,
                execution_registry.role_acceptance_signatures,
                strict=True,
            ):
                _verify_ed25519(
                    key.public_key,
                    signature.signature,
                    phase2b_c1_logical_stage_authorization_signature_preimage_v2(
                        execution_registry.body,
                        purpose_id=key.purpose_id,
                        role=key.role,
                        signer_key_id=key.key_id,
                    ),
                )
            store_key = legacy_registry.body.role_keys[2]
            _verify_ed25519(
                store_key.public_key,
                authorization_store_receipt.store_signature.signature,
                phase2b_c1_logical_stage_authorization_store_signature_preimage_v2(
                    authorization_store_receipt.body,
                    signer_key_id=store_key.key_id,
                ),
            )
        except _BackendUnavailable:
            return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
        except _SignatureInvalid:
            return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_INVALID)
        return _issue_authority_result(
            disposition=Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION,
            reason=Phase2BC1LogicalStageAuthorityReasonV2.V2_AUTHORITY_GRAPH_SIGNATURE_AND_CONTENT_MECHANICS_VERIFIED,
            context=authority_context,
            registry=execution_registry,
            store=authorization_store_receipt,
        )
    except _ORDINARY_INPUT_ERRORS:
        return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.GRAPH_BINDING_INVALID)
    except Exception:
        return _reject_authority(Phase2BC1LogicalStageAuthorityReasonV2.INTERNAL_ERROR)


_PREFIX_RESULT_TOKEN: Final = object()
_PREFIX_TRUE_CLAIMS: Final = (
    "complete_graph_snapshotted_before_verification",
    "authority_graph_reverified",
    "logical_stage_wire_v1_reverified",
    "v2_policy_admission_verified",
    "v1_legacy_false_fields_preserved",
    "verification_result_privately_issued",
)


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1AuthorizedLogicalStagePrefixVerificationV2:
    disposition: Phase2BC1AuthorizedLogicalStagePrefixDispositionV2
    reason: Phase2BC1AuthorizedLogicalStagePrefixReasonV2
    version: str
    claim_level: str
    result_id: str | None
    authority_result_id: str | None
    logical_stage_result_id: str | None
    authority_context_id: str | None
    execution_registry_id: str | None
    legacy_context_id: str | None
    legacy_execution_registry_id: str | None
    terminal_stage: six.Phase2BC1LogicalStageV1 | None
    terminal_logical_sequence: int | None
    stage_token_ids: tuple[str, ...]
    event_projection_ids: tuple[str, ...]
    partial_output_published: bool
    complete_graph_snapshotted_before_verification: bool
    authority_graph_reverified: bool
    logical_stage_wire_v1_reverified: bool
    v2_policy_admission_verified: bool
    v1_legacy_false_fields_preserved: bool
    verification_result_privately_issued: bool
    legacy_registry_policy_authorized: bool
    actual_campaign_eligible: bool
    actual_execution_claimed: bool
    execution_authorized: bool
    owner_acceptance_observed: bool
    authorization_precedes_stage1_globally_verified: bool
    docker_invoked: bool
    filesystem_read: bool
    filesystem_written: bool
    keys_generated: bool
    signatures_generated: bool
    time_operation_executed: bool
    timeline_observed: bool
    run_start_observed: bool
    run_finish_observed: bool
    recognizer_executed: bool
    runtime_executed: bool
    scoring_performed: bool
    prediction_scored: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool
    cross_host_global_one_shot_verified: bool

    def __new__(cls, token: object = None) -> "Phase2BC1AuthorizedLogicalStagePrefixVerificationV2":
        if token is not _PREFIX_RESULT_TOKEN:
            raise TypeError("authorized prefix results are privately issued")
        return object.__new__(cls)

    def __init__(self, token: object = None) -> None:
        if token is not _PREFIX_RESULT_TOKEN:
            raise TypeError("authorized prefix results are privately issued")


def _issue_prefix_result(
    *,
    disposition: Phase2BC1AuthorizedLogicalStagePrefixDispositionV2,
    reason: Phase2BC1AuthorizedLogicalStagePrefixReasonV2,
    authority: Phase2BC1LogicalStageAuthorityVerificationV2 | None = None,
    stage: logical.Phase2BC1LogicalStageWireVerificationV1 | None = None,
) -> Phase2BC1AuthorizedLogicalStagePrefixVerificationV2:
    success = disposition is Phase2BC1AuthorizedLogicalStagePrefixDispositionV2.OWNER_ACCEPTED_V2_POLICY_AND_CALLER_SUPPLIED_V1_PREFIX_VERIFIED_NOT_EXECUTION
    values: list[tuple[str, object]] = [
        ("disposition", disposition), ("reason", reason),
        ("version", PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION),
        ("claim_level", PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_CLAIM_LEVEL),
    ]
    if success:
        if authority is None or stage is None or authority.result_id is None or stage.result_id is None:
            raise RuntimeError("prefix success lacks verifier results")
        result_id = _PREFIX_RESULT_PREFIX + _digest(_PREFIX_RESULT_ID_DOMAIN, (
            ("authority_result_id", authority.result_id),
            ("logical_stage_result_id", stage.result_id),
            ("stage_token_ids", stage.stage_token_ids),
            ("event_projection_ids", stage.event_projection_ids),
        ))
        metadata = (
            result_id, authority.result_id, stage.result_id,
            authority.authority_context_id, authority.execution_registry_id,
            authority.legacy_context_id, authority.legacy_execution_registry_id,
            stage.terminal_stage, stage.terminal_logical_sequence,
            stage.stage_token_ids, stage.event_projection_ids,
        )
    else:
        metadata = (None,) * 9 + ((), ())
    values.extend(zip((
        "result_id", "authority_result_id", "logical_stage_result_id",
        "authority_context_id", "execution_registry_id", "legacy_context_id",
        "legacy_execution_registry_id", "terminal_stage",
        "terminal_logical_sequence", "stage_token_ids", "event_projection_ids",
    ), metadata, strict=True))
    values.append(("partial_output_published", False))
    values.extend((name, success) for name in _PREFIX_TRUE_CLAIMS)
    values.extend((name, False) for name in _FALSE_CLAIMS)
    result = Phase2BC1AuthorizedLogicalStagePrefixVerificationV2(_PREFIX_RESULT_TOKEN)
    if tuple(name for name, _ in values) != tuple(item.name for item in fields(type(result))):
        raise RuntimeError("authorized prefix result field drift")
    for name, value in values:
        object.__setattr__(result, name, value)
    return result


def _reject_prefix(
    reason: Phase2BC1AuthorizedLogicalStagePrefixReasonV2,
) -> Phase2BC1AuthorizedLogicalStagePrefixVerificationV2:
    return _issue_prefix_result(
        disposition=Phase2BC1AuthorizedLogicalStagePrefixDispositionV2.REJECTED,
        reason=reason,
    )


def verify_authorized_phase2b_c1_logical_stage_prefix_v2(
    *,
    legacy_context: six.Phase2BC1CeremonyContextV1,
    legacy_registry: six.Phase2BC1ExecutionRegistryCandidateV1,
    legacy_store_receipt: six.Phase2BC1StoreIfAbsentReceiptCandidateV1,
    legacy_logical_genesis_token: six.Phase2BC1LogicalGenesisTokenCandidateV1,
    authority_context: Phase2BC1LogicalStageAuthorityContextV2,
    execution_registry: Phase2BC1OwnerAcceptedExecutionRegistryV2,
    authorization_store_receipt: Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2,
    event_projections: tuple[logical.Phase2BC1LogicalStageEventProjectionV1, ...],
    stage_tokens: tuple[logical.Phase2BC1LogicalStageTokenCandidateV1, ...],
) -> Phase2BC1AuthorizedLogicalStagePrefixVerificationV2:
    if type(event_projections) is not tuple or type(stage_tokens) is not tuple:
        return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.WRONG_INPUT_TYPE)
    try:
        try:
            closed = copy.deepcopy((
                legacy_context, legacy_registry, legacy_store_receipt,
                legacy_logical_genesis_token, authority_context,
                execution_registry, authorization_store_receipt,
                event_projections, stage_tokens,
            ))
        except Exception:
            return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.GRAPH_SNAPSHOT_INVALID)
        (
            legacy_context, legacy_registry, legacy_store_receipt,
            legacy_logical_genesis_token, authority_context,
            execution_registry, authorization_store_receipt,
            event_projections, stage_tokens,
        ) = closed
        authority = verify_phase2b_c1_logical_stage_authority_graph_v2(
            legacy_context=legacy_context,
            legacy_registry=legacy_registry,
            legacy_store_receipt=legacy_store_receipt,
            legacy_logical_genesis_token=legacy_logical_genesis_token,
            authority_context=authority_context,
            execution_registry=execution_registry,
            authorization_store_receipt=authorization_store_receipt,
        )
        if authority.disposition is not Phase2BC1LogicalStageAuthorityDispositionV2.CALLER_SUPPLIED_OWNER_ACCEPTED_V2_AUTHORITY_GRAPH_VERIFIED_NOT_EXECUTION:
            if authority.reason is Phase2BC1LogicalStageAuthorityReasonV2.SIGNATURE_BACKEND_UNAVAILABLE:
                return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
            return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.AUTHORITY_GRAPH_REJECTED)
        stage = logical.verify_phase2b_c1_logical_stage_prefix_v1(
            context=legacy_context,
            registry=legacy_registry,
            store_receipt=legacy_store_receipt,
            logical_genesis_token=legacy_logical_genesis_token,
            event_projections=event_projections,
            stage_tokens=stage_tokens,
        )
        if stage.disposition is not logical.Phase2BC1LogicalStageWireDispositionV1.CALLER_SUPPLIED_CONTIGUOUS_PREFIX_VERIFIED_NOT_AUTHORITY:
            if stage.reason is logical.Phase2BC1LogicalStageWireReasonV1.SIGNATURE_BACKEND_UNAVAILABLE:
                return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.SIGNATURE_BACKEND_UNAVAILABLE)
            return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.LOGICAL_STAGE_PREFIX_REJECTED)
        if (
            stage.legacy_registry_policy_authorized is not False
            or stage.actual_campaign_eligible is not False
            or stage.actual_execution_claimed is not False
            or stage.execution_authorized is not False
        ):
            return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.LOGICAL_STAGE_PREFIX_REJECTED)
        return _issue_prefix_result(
            disposition=Phase2BC1AuthorizedLogicalStagePrefixDispositionV2.OWNER_ACCEPTED_V2_POLICY_AND_CALLER_SUPPLIED_V1_PREFIX_VERIFIED_NOT_EXECUTION,
            reason=Phase2BC1AuthorizedLogicalStagePrefixReasonV2.V2_POLICY_AND_V1_PREFIX_CRYPTOGRAPHIC_MECHANICS_VERIFIED,
            authority=authority,
            stage=stage,
        )
    except _ORDINARY_INPUT_ERRORS:
        return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.LOGICAL_STAGE_PREFIX_REJECTED)
    except Exception:
        return _reject_prefix(Phase2BC1AuthorizedLogicalStagePrefixReasonV2.INTERNAL_ERROR)


_ALL_DOMAINS: Final = (
    _CONTEXT_DOMAIN, _AMENDMENT_ID_DOMAIN, _ROLE_ACCEPTANCE_SIGNATURE_DOMAIN,
    _REGISTRY_ID_DOMAIN, _STORE_SIGNATURE_DOMAIN, _STORE_RECEIPT_ID_DOMAIN,
    _RESULT_ID_DOMAIN, _PREFIX_RESULT_ID_DOMAIN,
)
if len(set(_ALL_DOMAINS)) != len(_ALL_DOMAINS) or any(
    not item.endswith(b"\x00") for item in _ALL_DOMAINS
):
    raise RuntimeError("logical-stage authority domain separation drift")
if set(_AUTHORITY_TRUE_CLAIMS) & set(_FALSE_CLAIMS):
    raise RuntimeError("authority claim polarity overlap")
if set(_PREFIX_TRUE_CLAIMS) & set(_FALSE_CLAIMS):
    raise RuntimeError("prefix claim polarity overlap")


__all__ = (
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_VERSION",
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORITY_V2_CLAIM_LEVEL",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SCHEMA_ID_V1",
    "PHASE2B_C1_STAGE_EVENT_CONTRACT_MANIFEST_ID_V1",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_POLICY_ID_V2",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_COMMIT_V1",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_BLOB_V1",
    "PHASE2B_C1_LOGICAL_STAGE_WIRE_SOURCE_SHA256_V1",
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SEMANTICS_V2",
    "PHASE2B_C1_LOGICAL_STAGE_AUTHORIZATION_SIGNATURE_PURPOSE_V2",
    "Phase2BC1LogicalStageAuthorityDispositionV2",
    "Phase2BC1LogicalStageAuthorityReasonV2",
    "Phase2BC1AuthorizedLogicalStagePrefixDispositionV2",
    "Phase2BC1AuthorizedLogicalStagePrefixReasonV2",
    "Phase2BC1LogicalStageAuthorityContextBodyV2",
    "Phase2BC1LogicalStageAuthorityContextV2",
    "Phase2BC1LogicalStageAuthorizationAmendmentBodyV2",
    "Phase2BC1LogicalStageAuthorizationSignatureV2",
    "Phase2BC1OwnerAcceptedExecutionRegistryV2",
    "Phase2BC1LogicalStageAuthorizationStoreReceiptBodyV2",
    "Phase2BC1LogicalStageAuthorizationStoreReceiptCandidateV2",
    "Phase2BC1LogicalStageAuthorityVerificationV2",
    "Phase2BC1AuthorizedLogicalStagePrefixVerificationV2",
    "phase2b_c1_logical_stage_authority_context_id_v2",
    "project_phase2b_c1_logical_stage_authority_context_to_v1",
    "phase2b_c1_logical_stage_authorization_amendment_id_v2",
    "phase2b_c1_logical_stage_authorization_signature_preimage_v2",
    "phase2b_c1_owner_accepted_execution_registry_id_v2",
    "canonical_phase2b_c1_owner_accepted_execution_registry_bytes_v2",
    "phase2b_c1_logical_stage_authorization_store_signature_preimage_v2",
    "phase2b_c1_logical_stage_authorization_store_receipt_id_v2",
    "verify_phase2b_c1_logical_stage_authority_graph_v2",
    "verify_authorized_phase2b_c1_logical_stage_prefix_v2",
)

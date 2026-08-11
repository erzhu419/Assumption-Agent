"""Secret-key batch mechanics for the Phase-2B trusted-wire profile.

This stage consumes only validated public transform authorities and exact
custodian inputs.  It derives purpose-separated keys, shuffles the complete
batch, assigns UUIDv4 values after the shuffle, rebuilds public-only wire
provenance, and emits fixed 65,536-byte envelopes with secret HMAC padding.

The implementation is mechanics evidence.  It neither authenticates a
custodian nor establishes independence, one-shot execution, a formal covert-
channel audit, or Phase-2B/C1 exit.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import hmac
import math
import re
import struct
from typing import Final

from .phase2b_exact_transform_semantics_v1 import (
    EXACT_TRANSFORM_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION,
    ExactTransformAtom,
    ExactTransformCompilation,
    PublicTransformEvidenceBundleV2,
    TransformCompilationDisposition,
    compile_exact_transform_provenance_v1,
    run_exact_transform_semantics,
)
from .phase2b_trusted_wire_typed_authority_v1 import (
    TYPED_AUTHORITY_CODEC_POLICY_ID,
    TYPED_AUTHORITY_CODEC_VERSION,
    TYPED_AUTHORITY_SCHEMA_ID,
    decode_typed_transform_authority_profile_v1,
    encode_typed_transform_authority_profile_v1,
)
from .phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    ENVELOPE_HEADER_BYTES,
    ENVELOPE_MAGIC,
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    MAXIMUM_PAYLOAD_BYTES,
    MAXIMUM_ARRAY_ENTRIES,
    MAXIMUM_ASCII_STRING_BYTES,
    MAXIMUM_PROFILE_DEPTH,
    MAXIMUM_PROFILE_NODES,
    MAXIMUM_RATIONAL_BIT_LENGTH,
    MAXIMUM_SAFE_INTEGER,
    MAXIMUM_UNIQUE_UUIDS,
    MAXIMUM_UUID_OCCURRENCES,
    MINIMUM_PADDING_BYTES,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    TRUSTED_WIRE_ENVELOPE_VERSION,
    NamespaceFieldAuditV1,
    ProfileDisposition,
    TrustedWireProfileCompilationV1,
    audit_namespace_paths_v1,
    compile_transform_authority_profile_mechanics_v1,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)


TRUSTED_WIRE_BATCH_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-batch-mechanics/2"
)
TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-batch-payload/2"
)
TRUSTED_WIRE_KEY_SCHEDULE_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-hkdf-sha256/1"
)
TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-public-provenance/2"
)
TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS: Final = (
    "authority",
    "field_manifest_id",
    "jcs_profile_id",
    "public_provenance_version",
    "schema_version",
    "typed_authority_schema_id",
)
EXACT_TRANSFORM_VALIDATOR_POLICY_ID: Final = EXACT_TRANSFORM_POLICY_ID

MAXIMUM_BATCH_AUTHORITIES: Final = 1_024
RUN_ID_BYTES: Final = 32
IKM_BYTES: Final = 32
MAXIMUM_UUID_COLLISION_RETRIES: Final = 10
MAXIMUM_SHUFFLE_REJECTION_DRAWS: Final = 32
MAXIMUM_OBSERVATIONS_PER_AUTHORITY: Final = 2_048
MAXIMUM_ENTITIES_PER_AUTHORITY: Final = 1_024
MAXIMUM_CONTRACTS_PER_AUTHORITY: Final = 256
MAXIMUM_METADATA_ROWS_PER_AUTHORITY: Final = 2_048
TOTAL_STRING_BUDGET_MULTIPLIER: Final = 256

_RUN_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/RUN/V1"
_HKDF_INFO_SHUFFLE: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/KEY/SHUFFLE/V1"
_HKDF_INFO_ID: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/KEY/ID/V1"
_HKDF_INFO_PADDING: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/KEY/PADDING/V1"
_SHUFFLE_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/SHUFFLE/V1"
_UUID_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/UUID/V1"
_PADDING_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/PADDING/V1"
_BASE_PROVENANCE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/TRUSTED_WIRE/BASE_PROVENANCE/V1\x00"
)
_BATCH_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/BATCH/V1\x00"
_POLICY_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/POLICY/V1\x00"
_RUN_COMMITMENT_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/RUN_COMMITMENT/V1\x00"
_ENVELOPE_ID_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/ENVELOPE_ID/V1\x00"
_SECRET_REPLAY_RECEIPT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/TRUSTED_WIRE/SECRET_REPLAY_RECEIPT/V1\x00"
)
_HEADER: Final = struct.Struct(">8sHHI32s32s")
_UUID4_TEXT: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_NAMESPACES: Final = (
    "aggregate_map",
    "bundle",
    "clock",
    "component",
    "context",
    "entity",
    "frame",
    "observation",
    "quantity",
    "quotient_class",
    "role_candidate",
    "scale",
    "source_channel",
    "task",
    "transform",
    "unit",
)
_ALGORITHM_PROFILE: Final = (
    "salt=sha256(run_domain||run_id)",
    "keys=three_pairwise_distinct_32_byte_ikm_hkdf_sha256_purpose_domains",
    "shuffle=hmac_sha256_u64_fisher_yates_rejection_max_32_draws",
    "uuid=hmac_sha256(domain||run_id||u16be_namespace_length||namespace||u64be_counter||u8_retry)",
    "uuid_schedule=post_shuffle_case_local_map_namespace_then_old_uuid_sort_batch_global_counters",
    "padding=hmac_sha256(domain||run_id||payload_sha256||u32be_block)",
    "header=8s_u16be_u16be_u32be_sha256_sha256_80_bytes",
    "provenance=public_base_then_exact_validator_native_derived_compiler_before_framing",
    "typed_authority=closed_profile_decode_reencode_then_direct_exact_transform_complete",
)
_BASE_PROVENANCE_FORMULA: Final = (
    "sha256",
    "base_provenance_domain_then_accepted_jcs_observation_without_provenance",
    "lowercase_hex_no_prefix",
)
_SECRET_REPLAY_RECEIPT_FIELDS: Final = (
    "authority_count",
    "batch_id",
    "policy_id",
    "run_id_commitment",
    "source_authority_content_ids",
)


class TrustedWireBatchDisposition(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


def _require_digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise TypeError(f"{name} must use the frozen content-ID prefix")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError(f"{name} must end in lowercase SHA-256")
    return value


def _require_exact_bool_claims(*values: object) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError("trusted-wire claim fields must use exact bool values")


@dataclass(frozen=True, slots=True)
class TrustedWireKeySourcesV1:
    shuffle_ikm: bytes = field(repr=False)
    id_ikm: bytes = field(repr=False)
    padding_ikm: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireKeySourcesV1:
            raise TypeError("trusted-wire key sources must use the exact type")
        values = (self.shuffle_ikm, self.id_ikm, self.padding_ikm)
        if any(type(value) is not bytes for value in values):
            raise TypeError("trusted-wire IKM values must use exact bytes")
        if any(len(value) != IKM_BYTES for value in values):
            raise ValueError("trusted-wire IKM values must be exactly 32 bytes")
        if len(set(values)) != 3:
            raise ValueError(
                "shuffle, ID, and padding IKM must be pairwise distinct"
            )


@dataclass(frozen=True, slots=True)
class TrustedWireBatchPolicyV1:
    schema_version: str = TRUSTED_WIRE_BATCH_SCHEMA_VERSION
    payload_schema_version: str = TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION
    key_schedule_version: str = TRUSTED_WIRE_KEY_SCHEDULE_VERSION
    public_provenance_version: str = TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
    typed_authority_schema_id: str = TYPED_AUTHORITY_SCHEMA_ID
    typed_authority_codec_version: str = TYPED_AUTHORITY_CODEC_VERSION
    typed_authority_codec_policy_id: str = TYPED_AUTHORITY_CODEC_POLICY_ID
    exact_transform_validator_policy_id: str = EXACT_TRANSFORM_VALIDATOR_POLICY_ID
    exact_transform_provenance_compiler_version: str = (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION
    )
    exact_transform_provenance_compiler_policy_id: str = (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
    )
    maximum_authorities: int = MAXIMUM_BATCH_AUTHORITIES
    run_id_bytes: int = RUN_ID_BYTES
    ikm_bytes: int = IKM_BYTES
    maximum_uuid_collision_retries: int = MAXIMUM_UUID_COLLISION_RETRIES
    maximum_shuffle_rejection_draws: int = MAXIMUM_SHUFFLE_REJECTION_DRAWS
    envelope_bytes: int = ENVELOPE_BYTES
    header_bytes: int = ENVELOPE_HEADER_BYTES
    minimum_padding_bytes: int = MINIMUM_PADDING_BYTES
    maximum_payload_bytes: int = MAXIMUM_PAYLOAD_BYTES
    maximum_profile_depth: int = MAXIMUM_PROFILE_DEPTH
    maximum_profile_nodes: int = MAXIMUM_PROFILE_NODES
    maximum_array_entries: int = MAXIMUM_ARRAY_ENTRIES
    maximum_ascii_string_bytes: int = MAXIMUM_ASCII_STRING_BYTES
    maximum_uuid_occurrences: int = MAXIMUM_UUID_OCCURRENCES
    maximum_unique_uuids: int = MAXIMUM_UNIQUE_UUIDS
    maximum_safe_integer: int = MAXIMUM_SAFE_INTEGER
    maximum_rational_bit_length: int = MAXIMUM_RATIONAL_BIT_LENGTH
    maximum_observations_per_authority: int = MAXIMUM_OBSERVATIONS_PER_AUTHORITY
    maximum_entities_per_authority: int = MAXIMUM_ENTITIES_PER_AUTHORITY
    maximum_contracts_per_authority: int = MAXIMUM_CONTRACTS_PER_AUTHORITY
    maximum_metadata_rows_per_authority: int = MAXIMUM_METADATA_ROWS_PER_AUTHORITY
    total_string_budget_multiplier: int = TOTAL_STRING_BUDGET_MULTIPLIER
    namespaces: tuple[str, ...] = _NAMESPACES

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireBatchPolicyV1:
            raise TypeError("trusted-wire batch policy must use the exact type")
        if (
            self.schema_version,
            self.payload_schema_version,
            self.key_schedule_version,
            self.public_provenance_version,
            self.typed_authority_schema_id,
            self.typed_authority_codec_version,
            self.typed_authority_codec_policy_id,
            self.exact_transform_validator_policy_id,
            self.exact_transform_provenance_compiler_version,
            self.exact_transform_provenance_compiler_policy_id,
            self.maximum_authorities,
            self.run_id_bytes,
            self.ikm_bytes,
            self.maximum_uuid_collision_retries,
            self.maximum_shuffle_rejection_draws,
            self.envelope_bytes,
            self.header_bytes,
            self.minimum_padding_bytes,
            self.maximum_payload_bytes,
            self.maximum_profile_depth,
            self.maximum_profile_nodes,
            self.maximum_array_entries,
            self.maximum_ascii_string_bytes,
            self.maximum_uuid_occurrences,
            self.maximum_unique_uuids,
            self.maximum_safe_integer,
            self.maximum_rational_bit_length,
            self.maximum_observations_per_authority,
            self.maximum_entities_per_authority,
            self.maximum_contracts_per_authority,
            self.maximum_metadata_rows_per_authority,
            self.total_string_budget_multiplier,
            self.namespaces,
        ) != (
            TRUSTED_WIRE_BATCH_SCHEMA_VERSION,
            TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION,
            TRUSTED_WIRE_KEY_SCHEDULE_VERSION,
            TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
            TYPED_AUTHORITY_SCHEMA_ID,
            TYPED_AUTHORITY_CODEC_VERSION,
            TYPED_AUTHORITY_CODEC_POLICY_ID,
            EXACT_TRANSFORM_VALIDATOR_POLICY_ID,
            EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION,
            EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
            MAXIMUM_BATCH_AUTHORITIES,
            RUN_ID_BYTES,
            IKM_BYTES,
            MAXIMUM_UUID_COLLISION_RETRIES,
            MAXIMUM_SHUFFLE_REJECTION_DRAWS,
            ENVELOPE_BYTES,
            ENVELOPE_HEADER_BYTES,
            MINIMUM_PADDING_BYTES,
            MAXIMUM_PAYLOAD_BYTES,
            MAXIMUM_PROFILE_DEPTH,
            MAXIMUM_PROFILE_NODES,
            MAXIMUM_ARRAY_ENTRIES,
            MAXIMUM_ASCII_STRING_BYTES,
            MAXIMUM_UUID_OCCURRENCES,
            MAXIMUM_UNIQUE_UUIDS,
            MAXIMUM_SAFE_INTEGER,
            MAXIMUM_RATIONAL_BIT_LENGTH,
            MAXIMUM_OBSERVATIONS_PER_AUTHORITY,
            MAXIMUM_ENTITIES_PER_AUTHORITY,
            MAXIMUM_CONTRACTS_PER_AUTHORITY,
            MAXIMUM_METADATA_ROWS_PER_AUTHORITY,
            TOTAL_STRING_BUDGET_MULTIPLIER,
            _NAMESPACES,
        ):
            raise ValueError("trusted-wire batch policy drift")

    @property
    def policy_id(self) -> str:
        value = {
            "algorithm_profile": list(_ALGORITHM_PROFILE),
            "base_provenance_formula": list(_BASE_PROVENANCE_FORMULA),
            "claim_boolean_contract": "all_receipt_claim_fields_exact_bool",
            "domain_hex": {
                "base_provenance": _BASE_PROVENANCE_DOMAIN.hex(),
                "batch": _BATCH_DOMAIN.hex(),
                "envelope_id": _ENVELOPE_ID_DOMAIN.hex(),
                "hkdf_info_id": _HKDF_INFO_ID.hex(),
                "hkdf_info_padding": _HKDF_INFO_PADDING.hex(),
                "hkdf_info_shuffle": _HKDF_INFO_SHUFFLE.hex(),
                "padding": _PADDING_DOMAIN.hex(),
                "run": _RUN_DOMAIN.hex(),
                "run_commitment": _RUN_COMMITMENT_DOMAIN.hex(),
                "secret_replay_receipt": _SECRET_REPLAY_RECEIPT_DOMAIN.hex(),
                "shuffle": _SHUFFLE_DOMAIN.hex(),
                "uuid": _UUID_DOMAIN.hex(),
            },
            "envelope_bytes": self.envelope_bytes,
            "envelope_magic_hex": ENVELOPE_MAGIC.hex(),
            "envelope_version": TRUSTED_WIRE_ENVELOPE_VERSION,
            "exact_transform_validator_policy_id": self.exact_transform_validator_policy_id,
            "exact_transform_provenance_compiler_policy_id": self.exact_transform_provenance_compiler_policy_id,
            "exact_transform_provenance_compiler_version": self.exact_transform_provenance_compiler_version,
            "field_manifest_id": FIELD_MANIFEST_ID,
            "header_struct_format": _HEADER.format,
            "header_bytes": self.header_bytes,
            "ikm_bytes": self.ikm_bytes,
            "key_schedule_version": self.key_schedule_version,
            "jcs_profile_id": JCS_PROFILE_ID,
            "maximum_authorities": self.maximum_authorities,
            "maximum_array_entries": self.maximum_array_entries,
            "maximum_ascii_string_bytes": self.maximum_ascii_string_bytes,
            "maximum_contracts_per_authority": self.maximum_contracts_per_authority,
            "maximum_entities_per_authority": self.maximum_entities_per_authority,
            "maximum_metadata_rows_per_authority": self.maximum_metadata_rows_per_authority,
            "maximum_observations_per_authority": self.maximum_observations_per_authority,
            "maximum_payload_bytes": self.maximum_payload_bytes,
            "maximum_profile_depth": self.maximum_profile_depth,
            "maximum_profile_nodes": self.maximum_profile_nodes,
            "maximum_rational_bit_length": self.maximum_rational_bit_length,
            "maximum_safe_integer": self.maximum_safe_integer,
            "maximum_shuffle_rejection_draws": self.maximum_shuffle_rejection_draws,
            "maximum_unique_uuids": self.maximum_unique_uuids,
            "maximum_uuid_occurrences": self.maximum_uuid_occurrences,
            "maximum_uuid_collision_retries": self.maximum_uuid_collision_retries,
            "minimum_padding_bytes": self.minimum_padding_bytes,
            "namespaces": list(self.namespaces),
            "public_provenance_version": self.public_provenance_version,
            "secret_replay_receipt_contract": {
                "fields": list(_SECRET_REPLAY_RECEIPT_FIELDS),
                "ordered_source_authority_roots": True,
                "stored_id_recomputed_on_validation": True,
            },
            "payload_schema_version": self.payload_schema_version,
            "payload_top_level_fields": list(
                TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS
            ),
            "run_id_bytes": self.run_id_bytes,
            "schema_version": self.schema_version,
            "total_string_budget_multiplier": self.total_string_budget_multiplier,
            "typed_authority_codec_policy_id": self.typed_authority_codec_policy_id,
            "typed_authority_codec_version": self.typed_authority_codec_version,
            "typed_authority_schema_id": self.typed_authority_schema_id,
        }
        return "phase2b_trusted_wire_batch_policy_" + hashlib.sha256(
            _POLICY_DOMAIN + encode_phase2b_jcs_profile_v1(value)
        ).hexdigest()


DEFAULT_TRUSTED_WIRE_BATCH_POLICY: Final = TrustedWireBatchPolicyV1()
TRUSTED_WIRE_BATCH_POLICY_ID: Final = DEFAULT_TRUSTED_WIRE_BATCH_POLICY.policy_id


def _secret_replay_receipt_id(
    *,
    batch_id: str,
    run_id_commitment: str,
    authority_count: int,
    source_authority_content_ids: tuple[str, ...],
) -> str:
    payload = encode_phase2b_jcs_profile_v1(
        {
            "authority_count": authority_count,
            "batch_id": batch_id,
            "policy_id": TRUSTED_WIRE_BATCH_POLICY_ID,
            "run_id_commitment": run_id_commitment,
            "source_authority_content_ids": list(source_authority_content_ids),
        }
    )
    return "phase2b_trusted_wire_secret_replay_" + hashlib.sha256(
        _SECRET_REPLAY_RECEIPT_DOMAIN + payload
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustedWireBatchPreflightV1:
    disposition: TrustedWireBatchDisposition
    reason: str
    authority_count: int
    schema_version: str = TRUSTED_WIRE_BATCH_SCHEMA_VERSION
    policy_id: str = TRUSTED_WIRE_BATCH_POLICY_ID
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireBatchPreflightV1:
            raise TypeError("trusted-wire preflight must use the exact type")
        if self.disposition is not TrustedWireBatchDisposition.ABSTAIN:
            raise ValueError("trusted-wire preflight must abstain")
        if type(self.reason) is not str or not self.reason.isascii() or not self.reason:
            raise ValueError("trusted-wire preflight reason must be nonempty ASCII")
        if type(self.authority_count) is not int or self.authority_count < 0:
            raise ValueError("trusted-wire preflight count is invalid")
        if self.schema_version != TRUSTED_WIRE_BATCH_SCHEMA_VERSION:
            raise ValueError("trusted-wire preflight schema drift")
        if self.policy_id != TRUSTED_WIRE_BATCH_POLICY_ID:
            raise ValueError("trusted-wire preflight policy drift")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL:
            raise ValueError("trusted-wire preflight cannot issue formal evidence")


@dataclass(frozen=True, slots=True)
class TrustedWireEnvelopeV1:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit_id: str
    payload_schema_version: str
    public_provenance_version: str
    typed_authority_schema_id: str
    public_provenance_verified: bool = True
    structural_hashes_verified: bool = True
    secret_padding_replay_verified: bool = False
    typed_authority_decode_replay_implemented: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireEnvelopeV1:
            raise TypeError("trusted envelope must use the exact type")
        decoded = decode_and_audit_trusted_envelope_v1(self.envelope)
        if (
            self.envelope_id != decoded.envelope_id
            or self.payload_sha256 != decoded.payload_sha256
            or self.padding_sha256 != decoded.padding_sha256
            or self.payload_bytes != decoded.payload_bytes
            or self.padding_bytes != decoded.padding_bytes
            or self.namespace_audit_id != decoded.namespace_audit.audit_id
            or self.payload_schema_version != decoded.payload_schema_version
            or self.public_provenance_version
            != decoded.public_provenance_version
            or self.typed_authority_schema_id
            != decoded.typed_authority_schema_id
        ):
            raise ValueError("trusted envelope receipt drifts from its bytes")
        _require_exact_bool_claims(
            self.public_provenance_verified,
            self.structural_hashes_verified,
            self.secret_padding_replay_verified,
            self.typed_authority_decode_replay_implemented,
            self.origin_authenticated,
            self.formal_covert_audit,
        )
        if not self.public_provenance_verified or not self.structural_hashes_verified:
            raise ValueError("trusted envelope must pass public structural replay")
        if any(
            (
                self.secret_padding_replay_verified,
                self.typed_authority_decode_replay_implemented,
                self.origin_authenticated,
                self.formal_covert_audit,
            )
        ):
            raise ValueError("public envelope receipt cannot claim secret authority")


@dataclass(frozen=True, slots=True)
class DecodedTrustedEnvelopeV1:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit: NamespaceFieldAuditV1
    payload_schema_version: str
    public_provenance_version: str
    typed_authority_schema_id: str
    public_provenance_verified: bool = True
    structural_hashes_verified: bool = True
    secret_padding_replay_verified: bool = False
    typed_authority_decode_replay_implemented: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False

    def __post_init__(self) -> None:
        if type(self) is not DecodedTrustedEnvelopeV1:
            raise TypeError("decoded trusted envelope must use the exact type")
        payload, padding, _, audit = _decode_structural_envelope(self.envelope)
        _require_digest(self.envelope_id, "phase2b_trusted_envelope_", "envelope ID")
        for value, name in (
            (self.payload_sha256, "payload SHA-256"),
            (self.padding_sha256, "padding SHA-256"),
        ):
            if type(value) is not str or len(value) != 64 or any(
                item not in "0123456789abcdef" for item in value
            ):
                raise ValueError(f"{name} is invalid")
        if self.payload_bytes + self.padding_bytes + ENVELOPE_HEADER_BYTES != ENVELOPE_BYTES:
            raise ValueError("decoded trusted envelope lengths drift")
        if self.padding_bytes < MINIMUM_PADDING_BYTES:
            raise ValueError("decoded trusted envelope padding is too short")
        if type(self.namespace_audit) is not NamespaceFieldAuditV1:
            raise TypeError("decoded trusted envelope needs exact namespace audit")
        if (
            self.payload_schema_version
            != TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION
            or self.public_provenance_version
            != TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
            or self.typed_authority_schema_id != TYPED_AUTHORITY_SCHEMA_ID
        ):
            raise ValueError("decoded trusted envelope typed identity drift")
        expected_id = "phase2b_trusted_envelope_" + hashlib.sha256(
            _ENVELOPE_ID_DOMAIN + self.envelope
        ).hexdigest()
        if (
            self.envelope_id != expected_id
            or self.payload_sha256 != hashlib.sha256(payload).hexdigest()
            or self.padding_sha256 != hashlib.sha256(padding).hexdigest()
            or self.payload_bytes != len(payload)
            or self.padding_bytes != len(padding)
            or self.namespace_audit != audit
        ):
            raise ValueError("decoded trusted envelope receipt drifts from its bytes")
        _require_exact_bool_claims(
            self.public_provenance_verified,
            self.structural_hashes_verified,
            self.secret_padding_replay_verified,
            self.typed_authority_decode_replay_implemented,
            self.origin_authenticated,
            self.formal_covert_audit,
        )
        if not self.public_provenance_verified or not self.structural_hashes_verified:
            raise ValueError("decoded trusted envelope lacks structural evidence")
        if any(
            (
                self.secret_padding_replay_verified,
                self.typed_authority_decode_replay_implemented,
                self.origin_authenticated,
                self.formal_covert_audit,
            )
        ):
            raise ValueError("public decoder cannot claim secret authority")


_BATCH_ISSUE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class TrustedWireBatchV1:
    disposition: TrustedWireBatchDisposition
    schema_version: str
    payload_schema_version: str
    key_schedule_version: str
    public_provenance_version: str
    typed_authority_schema_id: str
    typed_authority_codec_policy_id: str
    exact_transform_provenance_compiler_policy_id: str
    jcs_profile_id: str
    field_manifest_id: str
    policy_id: str
    run_id_commitment: str
    envelopes: tuple[TrustedWireEnvelopeV1, ...]
    batch_id: str
    uuid_collision_retry_count: int
    uuid_collision_warning: bool
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    whole_batch_shuffle_applied: bool = True
    purpose_separated_keys_applied: bool = True
    post_shuffle_hmac_uuidv4_applied: bool = True
    secret_hmac_padding_applied: bool = True
    atomic_batch_emission: bool = True
    typed_authority_decode_replay_implemented: bool = False
    formal_uuid_audit: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False
    c1_exit_evidence: bool = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("trusted-wire batches are issued only by the exact builder")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        run_id_commitment: str,
        envelopes: tuple[TrustedWireEnvelopeV1, ...],
        batch_id: str,
        uuid_collision_retry_count: int,
        uuid_collision_warning: bool,
    ) -> "TrustedWireBatchV1":
        if token is not _BATCH_ISSUE_TOKEN:
            raise TypeError("trusted-wire batch issuer token mismatch")
        value = object.__new__(cls)
        frozen_values: tuple[tuple[str, object], ...] = (
            ("disposition", TrustedWireBatchDisposition.COMPLETE),
            ("schema_version", TRUSTED_WIRE_BATCH_SCHEMA_VERSION),
            (
                "payload_schema_version",
                TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION,
            ),
            ("key_schedule_version", TRUSTED_WIRE_KEY_SCHEDULE_VERSION),
            (
                "public_provenance_version",
                TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
            ),
            ("typed_authority_schema_id", TYPED_AUTHORITY_SCHEMA_ID),
            (
                "typed_authority_codec_policy_id",
                TYPED_AUTHORITY_CODEC_POLICY_ID,
            ),
            (
                "exact_transform_provenance_compiler_policy_id",
                EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
            ),
            ("jcs_profile_id", JCS_PROFILE_ID),
            ("field_manifest_id", FIELD_MANIFEST_ID),
            ("policy_id", TRUSTED_WIRE_BATCH_POLICY_ID),
            ("run_id_commitment", run_id_commitment),
            ("envelopes", envelopes),
            ("batch_id", batch_id),
            ("uuid_collision_retry_count", uuid_collision_retry_count),
            ("uuid_collision_warning", uuid_collision_warning),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("whole_batch_shuffle_applied", True),
            ("purpose_separated_keys_applied", True),
            ("post_shuffle_hmac_uuidv4_applied", True),
            ("secret_hmac_padding_applied", True),
            ("atomic_batch_emission", True),
            ("typed_authority_decode_replay_implemented", False),
            ("formal_uuid_audit", False),
            ("formal_covert_audit", False),
            ("sealed_holdout_eligible", False),
            ("c1_exit_evidence", False),
        )
        for name, item in frozen_values:
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not TrustedWireBatchV1:
            raise TypeError("trusted-wire batch must use the exact type")
        if self.disposition is not TrustedWireBatchDisposition.COMPLETE:
            raise ValueError("trusted-wire batch must be complete")
        if (
            self.schema_version != TRUSTED_WIRE_BATCH_SCHEMA_VERSION
            or self.payload_schema_version
            != TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION
            or self.key_schedule_version != TRUSTED_WIRE_KEY_SCHEDULE_VERSION
            or self.public_provenance_version
            != TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
            or self.typed_authority_schema_id != TYPED_AUTHORITY_SCHEMA_ID
            or self.typed_authority_codec_policy_id
            != TYPED_AUTHORITY_CODEC_POLICY_ID
            or self.exact_transform_provenance_compiler_policy_id
            != EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
            or self.jcs_profile_id != JCS_PROFILE_ID
            or self.field_manifest_id != FIELD_MANIFEST_ID
            or self.policy_id != TRUSTED_WIRE_BATCH_POLICY_ID
        ):
            raise ValueError("trusted-wire batch identity drift")
        _require_digest(
            self.run_id_commitment,
            "phase2b_trusted_wire_run_",
            "run ID commitment",
        )
        if type(self.envelopes) is not tuple or not self.envelopes or any(
            type(value) is not TrustedWireEnvelopeV1 for value in self.envelopes
        ):
            raise TypeError("trusted-wire batch needs exact immutable envelopes")
        for envelope in self.envelopes:
            envelope.__post_init__()
        if len(self.envelopes) > MAXIMUM_BATCH_AUTHORITIES:
            raise ValueError("trusted-wire batch exceeds the authority cap")
        if type(self.uuid_collision_retry_count) is not int or (
            self.uuid_collision_retry_count < 0
        ):
            raise ValueError("trusted-wire collision retry count is invalid")
        if type(self.uuid_collision_warning) is not bool or (
            self.uuid_collision_warning
            is not (self.uuid_collision_retry_count > 0)
        ):
            raise ValueError("trusted-wire collision warning disagrees with retries")
        expected_batch_id = _batch_id(
            self.run_id_commitment,
            self.envelopes,
            self.uuid_collision_retry_count,
        )
        if self.batch_id != expected_batch_id:
            raise ValueError("trusted-wire batch root drift")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL:
            raise ValueError("trusted-wire batch cannot issue formal evidence")
        _require_exact_bool_claims(
            self.whole_batch_shuffle_applied,
            self.purpose_separated_keys_applied,
            self.post_shuffle_hmac_uuidv4_applied,
            self.secret_hmac_padding_applied,
            self.atomic_batch_emission,
            self.typed_authority_decode_replay_implemented,
            self.formal_uuid_audit,
            self.formal_covert_audit,
            self.sealed_holdout_eligible,
            self.c1_exit_evidence,
        )
        if not all(
            (
                self.whole_batch_shuffle_applied,
                self.purpose_separated_keys_applied,
                self.post_shuffle_hmac_uuidv4_applied,
                self.secret_hmac_padding_applied,
                self.atomic_batch_emission,
            )
        ) or any(
            (
                self.formal_uuid_audit,
                self.formal_covert_audit,
                self.sealed_holdout_eligible,
                self.c1_exit_evidence,
                self.typed_authority_decode_replay_implemented,
            )
        ):
            raise ValueError("trusted-wire batch claim boundary drift")


_REPLAY_ISSUE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class TrustedWireReplayReceiptV1:
    batch_id: str
    run_id_commitment: str
    authority_count: int
    source_authority_content_ids: tuple[str, ...]
    policy_id: str
    replay_receipt_id: str
    replay_verified: bool = True
    secret_key_schedule_replayed: bool = True
    secret_padding_replayed: bool = True
    shuffle_and_uuid_assignment_replayed: bool = True
    typed_authority_decode_replay_implemented: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "trusted-wire secret replay receipts are issued only by exact replay"
        )

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        batch_id: str,
        run_id_commitment: str,
        authority_count: int,
        source_authority_content_ids: tuple[str, ...],
    ) -> "TrustedWireReplayReceiptV1":
        if token is not _REPLAY_ISSUE_TOKEN:
            raise TypeError("trusted-wire replay issuer token mismatch")
        value = object.__new__(cls)
        object.__setattr__(value, "batch_id", batch_id)
        object.__setattr__(value, "run_id_commitment", run_id_commitment)
        object.__setattr__(value, "authority_count", authority_count)
        object.__setattr__(
            value,
            "source_authority_content_ids",
            source_authority_content_ids,
        )
        object.__setattr__(value, "policy_id", TRUSTED_WIRE_BATCH_POLICY_ID)
        object.__setattr__(
            value,
            "replay_receipt_id",
            _secret_replay_receipt_id(
                batch_id=batch_id,
                run_id_commitment=run_id_commitment,
                authority_count=authority_count,
                source_authority_content_ids=source_authority_content_ids,
            ),
        )
        object.__setattr__(value, "replay_verified", True)
        object.__setattr__(value, "secret_key_schedule_replayed", True)
        object.__setattr__(value, "secret_padding_replayed", True)
        object.__setattr__(value, "shuffle_and_uuid_assignment_replayed", True)
        object.__setattr__(value, "typed_authority_decode_replay_implemented", False)
        object.__setattr__(value, "origin_authenticated", False)
        object.__setattr__(value, "formal_covert_audit", False)
        object.__setattr__(value, "sealed_holdout_eligible", False)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not TrustedWireReplayReceiptV1:
            raise TypeError("trusted-wire replay receipt must use the exact type")
        _require_digest(self.batch_id, "phase2b_trusted_wire_batch_", "batch ID")
        _require_digest(
            self.run_id_commitment,
            "phase2b_trusted_wire_run_",
            "run ID commitment",
        )
        _require_digest(
            self.policy_id,
            "phase2b_trusted_wire_batch_policy_",
            "trusted-wire replay policy ID",
        )
        if self.policy_id != TRUSTED_WIRE_BATCH_POLICY_ID:
            raise ValueError("trusted-wire replay policy drift")
        if type(self.authority_count) is not int or not (
            1 <= self.authority_count <= MAXIMUM_BATCH_AUTHORITIES
        ):
            raise ValueError("trusted-wire replay authority count is invalid")
        if (
            type(self.source_authority_content_ids) is not tuple
            or len(self.source_authority_content_ids) != self.authority_count
        ):
            raise TypeError("trusted-wire replay needs every ordered source root")
        for item in self.source_authority_content_ids:
            _require_digest(
                item,
                "phase2b_public_transform_evidence_",
                "source authority content ID",
            )
        _require_digest(
            self.replay_receipt_id,
            "phase2b_trusted_wire_secret_replay_",
            "secret replay receipt ID",
        )
        if self.replay_receipt_id != _secret_replay_receipt_id(
            batch_id=self.batch_id,
            run_id_commitment=self.run_id_commitment,
            authority_count=self.authority_count,
            source_authority_content_ids=self.source_authority_content_ids,
        ):
            raise ValueError("trusted-wire secret replay receipt root drift")
        _require_exact_bool_claims(
            self.replay_verified,
            self.secret_key_schedule_replayed,
            self.secret_padding_replayed,
            self.shuffle_and_uuid_assignment_replayed,
            self.typed_authority_decode_replay_implemented,
            self.origin_authenticated,
            self.formal_covert_audit,
            self.sealed_holdout_eligible,
        )
        if not all(
            (
                self.replay_verified,
                self.secret_key_schedule_replayed,
                self.secret_padding_replayed,
                self.shuffle_and_uuid_assignment_replayed,
            )
        ) or any(
            (
                self.origin_authenticated,
                self.formal_covert_audit,
                self.sealed_holdout_eligible,
                self.typed_authority_decode_replay_implemented,
            )
        ):
            raise ValueError("trusted-wire replay claim boundary drift")


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.digest(salt, ikm, "sha256")


def _hkdf_expand(prk: bytes, info: bytes, length: int = 32) -> bytes:
    if type(prk) is not bytes or len(prk) != 32:
        raise TypeError("HKDF PRK must be exact 32-byte input")
    if type(info) is not bytes or not 0 <= length <= 255 * 32:
        raise ValueError("HKDF expand parameters are invalid")
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.digest(prk, previous + info + bytes((counter,)), "sha256")
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def _derive_keys(run_id: bytes, sources: TrustedWireKeySourcesV1) -> tuple[bytes, bytes, bytes]:
    salt = hashlib.sha256(_RUN_DOMAIN + run_id).digest()
    infos = (_HKDF_INFO_SHUFFLE, _HKDF_INFO_ID, _HKDF_INFO_PADDING)
    ikms = (sources.shuffle_ikm, sources.id_ikm, sources.padding_ikm)
    keys = tuple(
        _hkdf_expand(_hkdf_extract(salt, ikm), info)
        for ikm, info in zip(ikms, infos, strict=True)
    )
    if len(set(keys)) != 3:
        raise ValueError("derived trusted-wire purpose keys collided")
    return keys  # type: ignore[return-value]


class _HmacU64Stream:
    __slots__ = ("_key", "_run_id", "_counter")

    def __init__(self, key: bytes, run_id: bytes) -> None:
        self._key = key
        self._run_id = run_id
        self._counter = 0

    def next_u64(self) -> int:
        value = hmac.digest(
            self._key,
            _SHUFFLE_DOMAIN + self._run_id + self._counter.to_bytes(8, "big"),
            "sha256",
        )
        self._counter += 1
        return int.from_bytes(value[:8], "big")


def _uniform_below(stream: _HmacU64Stream, bound: int) -> int:
    if type(bound) is not int or not 1 <= bound <= (1 << 64):
        raise ValueError("shuffle bound is outside the frozen range")
    limit = (1 << 64) - ((1 << 64) % bound)
    for _ in range(MAXIMUM_SHUFFLE_REJECTION_DRAWS):
        candidate = stream.next_u64()
        if candidate < limit:
            return candidate % bound
    raise ValueError("shuffle rejection-sampling draw budget exhausted")


def _shuffle_indices(count: int, key: bytes, run_id: bytes) -> tuple[int, ...]:
    values = list(range(count))
    stream = _HmacU64Stream(key, run_id)
    for index in range(count - 1, 0, -1):
        other = _uniform_below(stream, index + 1)
        values[index], values[other] = values[other], values[index]
    return tuple(values)


def _uuid4_candidate(
    key: bytes,
    run_id: bytes,
    namespace: str,
    counter: int,
    retry: int,
) -> str:
    message = (
        _UUID_DOMAIN
        + run_id
        + len(namespace.encode("ascii")).to_bytes(2, "big")
        + namespace.encode("ascii")
        + counter.to_bytes(8, "big")
        + retry.to_bytes(1, "big")
    )
    raw = bytearray(hmac.digest(key, message, "sha256")[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    text = raw.hex()
    return f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"


def _pointer_parts(pointer: str) -> tuple[str, ...]:
    parts = tuple(item for item in pointer.split("/") if item)
    if not parts or parts[0] != "authority":
        raise ValueError("namespace pointer is not authority-rooted")
    return parts[1:]


def _set_pointer(root: dict[str, object], pointer: str, value: str) -> None:
    parts = _pointer_parts(pointer)
    current: object = root
    for part in parts[:-1]:
        if type(current) is dict:
            current = current[part]
        elif type(current) is list:
            current = current[int(part)]
        else:
            raise ValueError("namespace pointer leaves the authority tree")
    final = parts[-1]
    if type(current) is dict:
        current[final] = value
    elif type(current) is list:
        current[int(final)] = value
    else:
        raise ValueError("namespace pointer has an invalid terminal parent")


def _rename_authority_ids(
    authority_mapping: dict[str, object],
    audit: NamespaceFieldAuditV1,
    id_key: bytes,
    run_id: bytes,
    counters: dict[str, int],
    allocated: set[str],
    renamings: dict[tuple[str, str], str],
    collision_retries: list[int],
) -> dict[str, object]:
    result = deepcopy(authority_mapping)
    unique_keys = tuple(
        sorted(
            {
                (occurrence.namespace, occurrence.public_uuid)
                for occurrence in audit.occurrences
            }
        )
    )
    for map_key in unique_keys:
        if map_key in renamings:
            continue
        namespace, _ = map_key
        counter = counters[namespace]
        replacement: str | None = None
        for retry in range(MAXIMUM_UUID_COLLISION_RETRIES + 1):
            candidate = _uuid4_candidate(
                id_key,
                run_id,
                namespace,
                counter,
                retry,
            )
            if candidate not in allocated:
                replacement = candidate
                collision_retries[0] += retry
                break
        if replacement is None:
            raise ValueError("HMAC UUIDv4 collision retry budget exhausted")
        counters[namespace] = counter + 1
        allocated.add(replacement)
        renamings[map_key] = replacement
    for occurrence in audit.occurrences:
        replacement = renamings[(occurrence.namespace, occurrence.public_uuid)]
        _set_pointer(result, occurrence.json_pointer, replacement)
    return result


def _without_provenance(value: dict[str, object]) -> dict[str, object]:
    result = deepcopy(value)
    result.pop("provenance_sha256", None)
    return result


def _ref_key(value: object) -> tuple[object, ...]:
    if type(value) is not dict:
        raise ValueError("component reference is malformed")
    required = ("scale_id", "observation_id", "ordinal", "component_id")
    if any(name not in value for name in required):
        raise ValueError("component reference is incomplete")
    return tuple(value[name] for name in required)


def _sort_ref_array(value: object) -> None:
    if type(value) is not list:
        raise ValueError("component-reference array is malformed")
    value.sort(key=_ref_key)


def _canonicalize_sparse_row(row: object) -> None:
    if type(row) is not dict or type(row.get("terms")) is not list:
        raise ValueError("sparse affine row is malformed")
    row["terms"].sort(key=lambda item: _ref_key(item["input_ref"]))


def _group_key(group: object) -> tuple[object, ...]:
    if type(group) is not dict:
        raise ValueError("certificate partition group is malformed")
    inputs = group.get("input_refs")
    outputs = group.get("output_refs")
    if type(inputs) is not list or type(outputs) is not list:
        raise ValueError("certificate partition refs are malformed")
    return (
        tuple(_ref_key(item) for item in inputs),
        tuple(_ref_key(item) for item in outputs),
    )


def _canonicalize_certificate(
    certificate: object,
    operation: object,
) -> None:
    if type(certificate) is not dict:
        raise ValueError("transform certificate is malformed")
    for name in (
        "inverse_rows",
        "source_commutation_rows",
        "target_commutation_rows",
    ):
        rows = certificate.get(name)
        if rows is None:
            continue
        if type(rows) is not list:
            raise ValueError("certificate sparse rows are malformed")
        for row in rows:
            _canonicalize_sparse_row(row)
        rows.sort(key=lambda item: _ref_key(item["output_ref"]))
    groups = certificate.get("groups")
    if groups is not None:
        if type(groups) is not list:
            raise ValueError("certificate partition groups are malformed")
        for group in groups:
            if type(group) is not dict:
                raise ValueError("certificate partition group is malformed")
            _sort_ref_array(group.get("input_refs"))
            _sort_ref_array(group.get("output_refs"))
        if operation == "coarse_graining":
            quotient_ids = certificate.get("quotient_class_ids")
            if type(quotient_ids) is not list or len(quotient_ids) != len(groups):
                raise ValueError("coarse group/quotient pairing is malformed")
            paired = sorted(
                zip(groups, quotient_ids, strict=True),
                key=lambda item: _group_key(item[0]),
            )
            groups[:] = [item[0] for item in paired]
            quotient_ids[:] = [item[1] for item in paired]
        else:
            groups.sort(key=_group_key)
    selected = certificate.get("selected_inputs")
    grid_points = certificate.get("grid_points")
    if selected is not None or grid_points is not None:
        if type(selected) is not list or type(grid_points) is not list or len(
            selected
        ) != len(grid_points):
            raise ValueError("sampling selected-input/grid pairing is malformed")
        if len({_ref_key(input_ref) for input_ref in selected}) != len(selected):
            raise ValueError("sampling selected inputs repeat")
    discarded = certificate.get("discarded_inputs")
    if discarded is not None:
        _sort_ref_array(discarded)


def _canonicalize_public_authority(authority: dict[str, object]) -> None:
    """Restore every constructor-defined set order after random ID assignment."""

    base = authority.get("base_bundle")
    metadata = authority.get("observation_metadata")
    contracts = authority.get("transform_contracts")
    if type(base) is not dict or type(metadata) is not list or type(contracts) is not list:
        raise ValueError("authority canonicalization containers are malformed")

    entities = base.get("entity_candidates")
    observations = base.get("observations")
    graph = base.get("aggregation_graph")
    transforms = base.get("transform_catalog")
    task = base.get("task_target")
    if (
        type(entities) is not list
        or type(observations) is not list
        or type(graph) is not dict
        or type(transforms) is not list
        or type(task) is not dict
    ):
        raise ValueError("base authority canonicalization shape is malformed")
    entities.sort(key=lambda item: item["entity_id"])
    for entity in entities:
        entity["role_candidate_ids"].sort()
    base["role_ids"].sort()
    base["quantity_ids"].sort()
    observations.sort(key=lambda item: item["observation_id"])
    for observation in observations:
        observation["entity_ids"].sort()
        observation["role_candidate_ids"].sort()
    task["entity_ids"].sort()
    task["quantity_ids"].sort()
    graph["scale_ids"].sort()
    graph["root_scale_ids"].sort()
    graph["edges"].sort(
        key=lambda item: (
            item["source_scale_id"],
            item["target_scale_id"],
            item["transform_id"],
        )
    )
    transforms.sort(key=lambda item: item["transform_id"])
    base["missingness_mask"].sort()
    metadata.sort(key=lambda item: item["observation_id"])

    contracts.sort(key=lambda item: item["transform_id"])
    for contract in contracts:
        _sort_ref_array(contract.get("input_components"))
        outputs = contract.get("output_components")
        output_observations = contract.get("output_observations")
        kernel_rows = contract.get("kernel_rows")
        discrete = contract.get("discrete_mappings")
        if (
            type(outputs) is not list
            or type(output_observations) is not list
            or type(kernel_rows) is not list
            or type(discrete) is not list
        ):
            raise ValueError("transform contract canonical arrays are malformed")
        outputs.sort(key=lambda item: _ref_key(item["ref"]))
        output_observations.sort(
            key=lambda item: (item["scale_id"], item["observation_id"])
        )
        for output in output_observations:
            output["entity_ids"].sort()
            output["role_candidate_ids"].sort()
            output["source_observation_ids"].sort()
            output["component_refs"].sort(key=lambda item: item["ordinal"])
        for row in kernel_rows:
            _canonicalize_sparse_row(row)
        kernel_rows.sort(key=lambda item: _ref_key(item["output_ref"]))
        discrete.sort(
            key=lambda item: (
                _ref_key(item["output_ref"]),
                _ref_key(item["input_ref"]),
            )
        )
        _canonicalize_certificate(
            contract.get("certificate"),
            contract.get("operation"),
        )


def _public_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + encode_phase2b_jcs_profile_v1(value)).hexdigest()


def _compile_native_public_provenance(
    authority: dict[str, object],
) -> PublicTransformEvidenceBundleV2:
    base = authority.get("base_bundle")
    if type(base) is not dict:
        raise ValueError("authority profile has malformed provenance containers")
    observations = base.get("observations")
    if type(observations) is not list:
        raise ValueError("base observation table is malformed")
    seen_observations: set[str] = set()
    for observation in observations:
        if type(observation) is not dict:
            raise ValueError("base observation is malformed")
        observation_id = observation.get("observation_id")
        if type(observation_id) is not str:
            raise ValueError("base observation ID is malformed")
        if observation_id in seen_observations:
            raise ValueError("base observation ID repeats during provenance replay")
        seen_observations.add(observation_id)
        provenance = _public_digest(
            _BASE_PROVENANCE_DOMAIN,
            _without_provenance(observation),
        )
        observation["provenance_sha256"] = provenance
    typed = decode_typed_transform_authority_profile_v1(authority)
    compiled = compile_exact_transform_provenance_v1(typed)
    compiled_profile = encode_typed_transform_authority_profile_v1(compiled)
    authority.clear()
    authority.update(compiled_profile)
    decoded = decode_typed_transform_authority_profile_v1(authority)
    if decoded != compiled:
        raise ValueError("native provenance typed roundtrip drift")
    result = run_exact_transform_semantics(decoded)
    if (
        type(result) is not ExactTransformCompilation
        or result.disposition is not TransformCompilationDisposition.COMPLETE
    ):
        reason = getattr(result, "reason", "transform_not_complete")
        raise ValueError("native provenance direct transform replay failed:" + reason)
    return decoded


def _audit_public_provenance(authority: dict[str, object]) -> None:
    replay = deepcopy(authority)
    _canonicalize_public_authority(replay)
    if replay != authority:
        raise ValueError("renamed authority set-like arrays are not canonical")
    _compile_native_public_provenance(replay)
    base_actual = authority["base_bundle"]
    base_replay = replay["base_bundle"]
    contracts_actual = authority["transform_contracts"]
    contracts_replay = replay["transform_contracts"]
    if type(base_actual) is not dict or type(base_replay) is not dict:
        raise ValueError("authority base bundle is malformed")
    if base_actual.get("observations") != base_replay.get("observations"):
        raise ValueError("base public provenance replay mismatch")
    if contracts_actual != contracts_replay:
        raise ValueError("derived public provenance replay mismatch")


def _secret_padding(key: bytes, run_id: bytes, payload_sha256: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    remaining = length
    counter = 0
    while remaining:
        block = hmac.digest(
            key,
            _PADDING_DOMAIN
            + run_id
            + payload_sha256
            + counter.to_bytes(4, "big"),
            "sha256",
        )
        blocks.append(block[:remaining])
        remaining -= min(remaining, len(block))
        counter += 1
    return b"".join(blocks)


def _frame_secret_payload(payload: bytes, padding_key: bytes, run_id: bytes) -> bytes:
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("trusted payload exceeds fixed-envelope capacity")
    payload_digest = hashlib.sha256(payload).digest()
    padding_length = ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES - len(payload)
    padding = _secret_padding(padding_key, run_id, payload_digest, padding_length)
    padding_digest = hashlib.sha256(padding).digest()
    header = _HEADER.pack(
        ENVELOPE_MAGIC,
        TRUSTED_WIRE_ENVELOPE_VERSION,
        ENVELOPE_HEADER_BYTES,
        len(payload),
        payload_digest,
        padding_digest,
    )
    envelope = header + payload + padding
    if len(envelope) != ENVELOPE_BYTES:
        raise RuntimeError("trusted envelope framing length drift")
    return envelope


def _decode_structural_envelope(
    envelope: bytes,
) -> tuple[bytes, bytes, dict[str, object], NamespaceFieldAuditV1]:
    if type(envelope) is not bytes:
        raise TypeError("trusted envelope must use exact bytes")
    if len(envelope) != ENVELOPE_BYTES:
        raise ValueError("trusted envelope must be exactly 65,536 bytes")
    magic, version, header_length, payload_length, payload_hash, padding_hash = (
        _HEADER.unpack(envelope[:ENVELOPE_HEADER_BYTES])
    )
    if magic != ENVELOPE_MAGIC or version != TRUSTED_WIRE_ENVELOPE_VERSION:
        raise ValueError("trusted envelope magic or version drift")
    if header_length != ENVELOPE_HEADER_BYTES or payload_length > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("trusted envelope header or payload length drift")
    payload = envelope[ENVELOPE_HEADER_BYTES : ENVELOPE_HEADER_BYTES + payload_length]
    padding = envelope[ENVELOPE_HEADER_BYTES + payload_length :]
    if len(padding) < MINIMUM_PADDING_BYTES:
        raise ValueError("trusted envelope padding is too short")
    if hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("trusted envelope payload hash drift")
    if hashlib.sha256(padding).digest() != padding_hash:
        raise ValueError("trusted envelope padding hash drift")
    decoded = decode_phase2b_jcs_profile_v1(payload)
    if type(decoded) is not dict or frozenset(decoded) != frozenset(
        TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS
    ):
        raise ValueError("trusted envelope payload schema drift")
    if (
        decoded["schema_version"] != TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION
        or decoded["jcs_profile_id"] != JCS_PROFILE_ID
        or decoded["field_manifest_id"] != FIELD_MANIFEST_ID
        or decoded["public_provenance_version"]
        != TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
        or decoded["typed_authority_schema_id"] != TYPED_AUTHORITY_SCHEMA_ID
        or type(decoded["authority"]) is not dict
    ):
        raise ValueError("trusted envelope payload identity drift")
    audit = audit_namespace_paths_v1(decoded["authority"])
    _audit_public_provenance(decoded["authority"])
    return payload, padding, decoded, audit


def decode_and_audit_trusted_envelope_v1(envelope: bytes) -> DecodedTrustedEnvelopeV1:
    """Replay public framing, hashes, accepted-JCS, paths, and provenance only."""

    payload, padding, decoded, audit = _decode_structural_envelope(envelope)
    return DecodedTrustedEnvelopeV1(
        envelope=envelope,
        envelope_id="phase2b_trusted_envelope_"
        + hashlib.sha256(_ENVELOPE_ID_DOMAIN + envelope).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        padding_sha256=hashlib.sha256(padding).hexdigest(),
        payload_bytes=len(payload),
        padding_bytes=len(padding),
        namespace_audit=audit,
        payload_schema_version=decoded["schema_version"],  # type: ignore[arg-type]
        public_provenance_version=decoded["public_provenance_version"],  # type: ignore[arg-type]
        typed_authority_schema_id=decoded["typed_authority_schema_id"],  # type: ignore[arg-type]
    )


def _batch_id(
    run_id_commitment: str,
    envelopes: tuple[TrustedWireEnvelopeV1, ...],
    uuid_collision_retry_count: int,
) -> str:
    value = {
        "envelope_ids": [item.envelope_id for item in envelopes],
        "field_manifest_id": FIELD_MANIFEST_ID,
        "jcs_profile_id": JCS_PROFILE_ID,
        "policy_id": TRUSTED_WIRE_BATCH_POLICY_ID,
        "payload_schema_version": TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION,
        "public_provenance_version": TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
        "run_id_commitment": run_id_commitment,
        "schema_version": TRUSTED_WIRE_BATCH_SCHEMA_VERSION,
        "typed_authority_schema_id": TYPED_AUTHORITY_SCHEMA_ID,
        "uuid_collision_retry_count": uuid_collision_retry_count,
    }
    return "phase2b_trusted_wire_batch_" + hashlib.sha256(
        _BATCH_DOMAIN + encode_phase2b_jcs_profile_v1(value)
    ).hexdigest()


def _run_commitment(run_id: bytes) -> str:
    return "phase2b_trusted_wire_run_" + hashlib.sha256(
        _RUN_COMMITMENT_DOMAIN + run_id
    ).hexdigest()


def _hash_free_authority_preflight(
    authority: PublicTransformEvidenceBundleV2,
) -> str | None:
    """Mirror the accepted-profile caps without hashing or serializing."""

    nodes = 0
    entries = 0
    total_string_bytes = 0
    uuid_occurrences = 0
    unique_uuids: set[str] = set()
    stack: list[tuple[object, int]] = [(authority, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAXIMUM_PROFILE_NODES:
            return "node_budget_exceeded"
        if depth > MAXIMUM_PROFILE_DEPTH:
            return "depth_budget_exceeded"
        exact_type = type(value)
        if exact_type is ExactTransformAtom:
            if (
                value.numerator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
                or value.denominator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
            ):
                return "rational_bit_length_exceeded"
            # The accepted mapping represents each rational as a two-field
            # object with canonical decimal strings.
            entries += 2
            stack.append((str(value.numerator), depth + 1))
            stack.append((str(value.denominator), depth + 1))
        elif exact_type is str:
            try:
                encoded = value.encode("ascii", errors="strict")
            except UnicodeEncodeError:
                return "non_ascii_string"
            if len(encoded) > MAXIMUM_ASCII_STRING_BYTES:
                return "string_budget_exceeded"
            total_string_bytes += len(encoded)
            if (
                total_string_bytes
                > MAXIMUM_PROFILE_NODES * TOTAL_STRING_BUDGET_MULTIPLIER
            ):
                return "total_string_budget_exceeded"
            if _UUID4_TEXT.fullmatch(value) is not None:
                uuid_occurrences += 1
                unique_uuids.add(value)
                if uuid_occurrences > MAXIMUM_UUID_OCCURRENCES:
                    return "uuid_occurrence_budget_exceeded"
                if len(unique_uuids) > MAXIMUM_UNIQUE_UUIDS:
                    return "unique_uuid_budget_exceeded"
        elif exact_type is int:
            if abs(value) > MAXIMUM_SAFE_INTEGER:
                return "safe_integer_budget_exceeded"
        elif exact_type is float:
            if not math.isfinite(value):
                return "nonfinite_binary64"
        elif exact_type in (bool, type(None)):
            pass
        elif isinstance(value, Enum):
            stack.append((value.value, depth + 1))
        elif exact_type is tuple:
            if len(value) > MAXIMUM_ARRAY_ENTRIES:
                return "array_entry_budget_exceeded"
            entries += len(value)
            if entries > MAXIMUM_PROFILE_NODES:
                return "total_entry_budget_exceeded"
            stack.extend((item, depth + 1) for item in value)
        elif is_dataclass(value):
            rows = fields(value)
            if len(rows) > MAXIMUM_ARRAY_ENTRIES:
                return "object_entry_budget_exceeded"
            entries += len(rows)
            if entries > MAXIMUM_PROFILE_NODES:
                return "total_entry_budget_exceeded"
            stack.extend(
                (getattr(value, item.name), depth + 1) for item in rows
            )
        else:
            return "unsupported_authority_type"
    return None


def _preflight(
    authorities: object,
    run_id: object,
    key_sources: object,
) -> TrustedWireBatchPreflightV1 | None:
    if type(authorities) is not tuple:
        raise TypeError("trusted-wire authorities must use an exact tuple")
    count = len(authorities)
    if not 1 <= count <= MAXIMUM_BATCH_AUTHORITIES:
        return TrustedWireBatchPreflightV1(
            TrustedWireBatchDisposition.ABSTAIN,
            "authority_count_outside_1_through_1024",
            count,
        )
    if type(run_id) is not bytes:
        raise TypeError("trusted-wire run ID must use exact bytes")
    if len(run_id) != RUN_ID_BYTES:
        return TrustedWireBatchPreflightV1(
            TrustedWireBatchDisposition.ABSTAIN,
            "run_id_must_be_exactly_32_bytes",
            count,
        )
    if type(key_sources) is not TrustedWireKeySourcesV1:
        raise TypeError("trusted-wire key sources must use the exact frozen type")
    key_sources.__post_init__()
    if any(type(value) is not PublicTransformEvidenceBundleV2 for value in authorities):
        raise TypeError("trusted-wire batch contains a non-V2 authority")
    # Cheap top-level guards precede the full hash-free object-tree pass.
    if any(
        len(value.base_bundle.observations) > MAXIMUM_OBSERVATIONS_PER_AUTHORITY
        or len(value.base_bundle.entity_candidates) > MAXIMUM_ENTITIES_PER_AUTHORITY
        or len(value.transform_contracts) > MAXIMUM_CONTRACTS_PER_AUTHORITY
        or len(value.observation_metadata) > MAXIMUM_METADATA_ROWS_PER_AUTHORITY
        for value in authorities
    ):
        return TrustedWireBatchPreflightV1(
            TrustedWireBatchDisposition.ABSTAIN,
            "hash_free_authority_shape_budget_exceeded",
            count,
        )
    for authority in authorities:
        reason = _hash_free_authority_preflight(authority)
        if reason is not None:
            return TrustedWireBatchPreflightV1(
                TrustedWireBatchDisposition.ABSTAIN,
                "hash_free_profile_preflight:" + reason,
                count,
            )
    return None


def build_trusted_wire_batch_v1(
    *,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV1,
) -> TrustedWireBatchV1 | TrustedWireBatchPreflightV1:
    """Build one all-or-nothing secret-padded, shuffled trusted-wire batch."""

    rejection = _preflight(authorities, run_id, key_sources)
    if rejection is not None:
        return rejection
    compilations: list[TrustedWireProfileCompilationV1] = []
    for authority in authorities:
        compilation = compile_transform_authority_profile_mechanics_v1(authority)
        if (
            type(compilation) is not TrustedWireProfileCompilationV1
            or compilation.disposition is not ProfileDisposition.COMPLETE
        ):
            reason = getattr(compilation, "reason", "profile_not_complete")
            return TrustedWireBatchPreflightV1(
                TrustedWireBatchDisposition.ABSTAIN,
                "authority_profile_rejected:" + str(reason),
                len(authorities),
            )
        compilations.append(compilation)

    try:
        shuffle_key, id_key, padding_key = _derive_keys(run_id, key_sources)
        order = _shuffle_indices(len(compilations), shuffle_key, run_id)
    except (OverflowError, TypeError, ValueError):
        return TrustedWireBatchPreflightV1(
            TrustedWireBatchDisposition.ABSTAIN,
            "key_schedule_or_shuffle_failed",
            len(authorities),
        )
    counters = {namespace: 0 for namespace in _NAMESPACES}
    allocated: set[str] = set()
    collision_retries = [0]
    envelope_rows: list[TrustedWireEnvelopeV1] = []
    for original_index in order:
        compilation = compilations[original_index]
        decoded = decode_phase2b_jcs_profile_v1(compilation.payload)
        if type(decoded) is not dict or type(decoded.get("authority")) is not dict:
            return TrustedWireBatchPreflightV1(
                TrustedWireBatchDisposition.ABSTAIN,
                "authority_profile_payload_drift",
                len(authorities),
            )
        try:
            authority_mapping = _rename_authority_ids(
                decoded["authority"],
                compilation.namespace_audit,
                id_key,
                run_id,
                counters,
                allocated,
                {},
                collision_retries,
            )
            _canonicalize_public_authority(authority_mapping)
            typed_authority = _compile_native_public_provenance(
                authority_mapping
            )
            if (
                encode_typed_transform_authority_profile_v1(typed_authority)
                != authority_mapping
            ):
                raise ValueError("native typed authority profile roundtrip drift")
            audit = audit_namespace_paths_v1(authority_mapping)
            payload_value = {
                "authority": authority_mapping,
                "field_manifest_id": FIELD_MANIFEST_ID,
                "jcs_profile_id": JCS_PROFILE_ID,
                "public_provenance_version": (
                    TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
                ),
                "schema_version": TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION,
                "typed_authority_schema_id": TYPED_AUTHORITY_SCHEMA_ID,
            }
            if tuple(sorted(payload_value)) != TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS:
                raise RuntimeError("trusted-wire payload field manifest drift")
            payload = encode_phase2b_jcs_profile_v1(payload_value)
            envelope = _frame_secret_payload(payload, padding_key, run_id)
            public_receipt = decode_and_audit_trusted_envelope_v1(envelope)
        except (AttributeError, KeyError, TypeError, ValueError):
            return TrustedWireBatchPreflightV1(
                TrustedWireBatchDisposition.ABSTAIN,
                "renamed_authority_or_provenance_replay_failed",
                len(authorities),
            )
        if public_receipt.namespace_audit != audit:
            return TrustedWireBatchPreflightV1(
                TrustedWireBatchDisposition.ABSTAIN,
                "renamed_namespace_audit_drift",
                len(authorities),
            )
        envelope_rows.append(
            TrustedWireEnvelopeV1(
                envelope=envelope,
                envelope_id=public_receipt.envelope_id,
                payload_sha256=public_receipt.payload_sha256,
                padding_sha256=public_receipt.padding_sha256,
                payload_bytes=public_receipt.payload_bytes,
                padding_bytes=public_receipt.padding_bytes,
                namespace_audit_id=public_receipt.namespace_audit.audit_id,
                payload_schema_version=public_receipt.payload_schema_version,
                public_provenance_version=(
                    public_receipt.public_provenance_version
                ),
                typed_authority_schema_id=(
                    public_receipt.typed_authority_schema_id
                ),
            )
        )
    rows = tuple(envelope_rows)
    commitment = _run_commitment(run_id)
    return TrustedWireBatchV1._issue(
        _BATCH_ISSUE_TOKEN,
        run_id_commitment=commitment,
        envelopes=rows,
        batch_id=_batch_id(commitment, rows, collision_retries[0]),
        uuid_collision_retry_count=collision_retries[0],
        uuid_collision_warning=collision_retries[0] > 0,
    )


def verify_trusted_wire_batch_replay_v1(
    *,
    batch: TrustedWireBatchV1,
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV1,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
) -> TrustedWireReplayReceiptV1:
    """Custodian-side exact secret replay; no keys or permutations are returned."""

    if type(batch) is not TrustedWireBatchV1:
        raise TypeError("trusted-wire replay requires the exact batch type")
    batch._validate()
    rebuilt = build_trusted_wire_batch_v1(
        authorities=authorities,
        run_id=run_id,
        key_sources=key_sources,
    )
    if type(rebuilt) is not TrustedWireBatchV1 or rebuilt != batch:
        raise ValueError("trusted-wire secret replay mismatch")
    return TrustedWireReplayReceiptV1._issue(
        _REPLAY_ISSUE_TOKEN,
        batch_id=batch.batch_id,
        run_id_commitment=batch.run_id_commitment,
        authority_count=len(batch.envelopes),
        source_authority_content_ids=tuple(
            authority.content_id for authority in authorities
        ),
    )


__all__ = (
    "DEFAULT_TRUSTED_WIRE_BATCH_POLICY",
    "DecodedTrustedEnvelopeV1",
    "IKM_BYTES",
    "EXACT_TRANSFORM_VALIDATOR_POLICY_ID",
    "MAXIMUM_BATCH_AUTHORITIES",
    "MAXIMUM_CONTRACTS_PER_AUTHORITY",
    "MAXIMUM_ENTITIES_PER_AUTHORITY",
    "MAXIMUM_METADATA_ROWS_PER_AUTHORITY",
    "MAXIMUM_OBSERVATIONS_PER_AUTHORITY",
    "MAXIMUM_SHUFFLE_REJECTION_DRAWS",
    "MAXIMUM_UUID_COLLISION_RETRIES",
    "RUN_ID_BYTES",
    "TRUSTED_WIRE_BATCH_SCHEMA_VERSION",
    "TRUSTED_WIRE_KEY_SCHEDULE_VERSION",
    "TRUSTED_WIRE_BATCH_POLICY_ID",
    "TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS",
    "TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION",
    "TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION",
    "TrustedWireBatchDisposition",
    "TrustedWireBatchPolicyV1",
    "TrustedWireBatchPreflightV1",
    "TrustedWireBatchV1",
    "TrustedWireEnvelopeV1",
    "TrustedWireKeySourcesV1",
    "TrustedWireReplayReceiptV1",
    "build_trusted_wire_batch_v1",
    "decode_and_audit_trusted_envelope_v1",
    "verify_trusted_wire_batch_replay_v1",
)

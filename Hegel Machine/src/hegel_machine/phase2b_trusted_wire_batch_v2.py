"""Compact-authority V2 trusted-wire batch mechanics.

This module keeps the frozen V1 shuffle and post-shuffle UUID allocation
schedule, but owns a distinct payload schema, envelope frame, content-ID
domains, padding domain, policy, and exact public types.  It emits no secret
replay, origin, formal-audit, sealed-holdout, recognizer, or C1 evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
import hashlib
import hmac
import json
import math
import struct
from types import MappingProxyType
from typing import Callable, Final, Mapping

from .phase2b_exact_transform_semantics_v1 import (
    EXACT_TRANSFORM_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION,
    ExactTransformCompilation,
    PublicTransformEvidenceBundleV2,
    TransformCompilationDisposition,
    run_exact_transform_semantics,
)
from .phase2b_trusted_wire_typed_authority_v1 import (
    encode_typed_transform_authority_profile_v1,
)
from .phase2b_trusted_wire_typed_authority_v2 import (
    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
    COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
    decode_typed_transform_authority_profile_v2,
    encode_typed_transform_authority_profile_v2,
)
from .phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    ENVELOPE_HEADER_BYTES,
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    MAXIMUM_ARRAY_ENTRIES,
    MAXIMUM_ASCII_STRING_BYTES,
    MAXIMUM_PAYLOAD_BYTES,
    MAXIMUM_PROFILE_DEPTH,
    MAXIMUM_PROFILE_NODES,
    MAXIMUM_RATIONAL_BIT_LENGTH,
    MAXIMUM_SAFE_INTEGER,
    MAXIMUM_UNIQUE_UUIDS,
    MAXIMUM_UUID_OCCURRENCES,
    MINIMUM_PADDING_BYTES,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    NamespaceFieldAuditV1,
    NamespaceOccurrenceV1,
    audit_namespace_paths_v1,
    encode_phase2b_jcs_profile_v1,
)
from . import phase2b_trusted_wire_batch_v1 as _batch_v1
from . import phase2b_trusted_wire_typed_authority_v1 as _typed_v1
from . import phase2b_trusted_wire_typed_authority_v2 as _typed_v2


TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-batch-mechanics/3"
)
TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-batch-payload/3"
)
TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS: Final = (
    "authority",
    "field_manifest_id",
    "jcs_profile_id",
    "public_provenance_version",
    "schema_version",
    "typed_authority_codec_policy_id",
    "typed_authority_codec_version",
    "typed_authority_schema_id",
)
TRUSTED_WIRE_ENVELOPE_V2_MAGIC: Final = b"HGP2BW2\x00"
TRUSTED_WIRE_ENVELOPE_V2_VERSION: Final = 2

MAXIMUM_BATCH_V2_AUTHORITIES: Final = _batch_v1.MAXIMUM_BATCH_AUTHORITIES
RUN_ID_BYTES: Final = _batch_v1.RUN_ID_BYTES
IKM_BYTES: Final = _batch_v1.IKM_BYTES
MAXIMUM_UUID_COLLISION_RETRIES: Final = (
    _batch_v1.MAXIMUM_UUID_COLLISION_RETRIES
)
MAXIMUM_SHUFFLE_REJECTION_DRAWS: Final = (
    _batch_v1.MAXIMUM_SHUFFLE_REJECTION_DRAWS
)
MAXIMUM_BATCH_V2_COLLISION_RETRY_COUNT: Final = (
    MAXIMUM_BATCH_V2_AUTHORITIES
    * MAXIMUM_UUID_OCCURRENCES
    * MAXIMUM_UUID_COLLISION_RETRIES
)
_MAXIMUM_BATCH_V2_UUID_SIDECAR_ENTRIES: Final = (
    MAXIMUM_BATCH_V2_AUTHORITIES * MAXIMUM_UUID_OCCURRENCES
)

_HEADER_V2: Final = struct.Struct(">8sHHI32s32s")
if _HEADER_V2.size != ENVELOPE_HEADER_BYTES:
    raise RuntimeError("V2 trusted-wire header size drift")

_PADDING_DOMAIN_V2: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/PADDING/V2"
_POLICY_DOMAIN_V2: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/POLICY/V2\x00"
_BATCH_DOMAIN_V2: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE/BATCH/V2\x00"
_ENVELOPE_ID_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/TRUSTED_WIRE/ENVELOPE_ID/V2\x00"
)
_RUN_COMMITMENT_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/TRUSTED_WIRE/RUN_COMMITMENT/V2\x00"
)
_NAMESPACE_AUDIT_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/TRUSTED_WIRE/NAMESPACE_AUDIT/V2\x00"
)

_SHARED_V1_HELPER_CONTRACT: Final = (
    (
        "_hash_free_authority_preflight",
        "bounded_exact_authority_tree_before_logical_profile_materialization",
    ),
    (
        "_derive_keys",
        "V1_HKDF_SHA256_run_salt_and_three_purpose_info_domains",
    ),
    (
        "_shuffle_indices",
        "V1_HMAC_u64_Fisher_Yates_rejection_schedule",
    ),
    (
        "_rename_authority_ids",
        "single_live_post_shuffle_case_local_map_and_batch_global_counters",
    ),
    (
        "_canonicalize_public_authority",
        "frozen_set_like_array_and_certificate_canonicalization",
    ),
    (
        "_compile_native_public_provenance",
        "V1_base_provenance_then_exact_native_derived_provenance",
    ),
    (
        "_audit_public_provenance",
        "independent_public_provenance_replay_on_expanded_logical_authority",
    ),
)
_V2_ALGORITHM_PROFILE: Final = (
    "source=exact_V2_authority_then_V1_logical_encode_without_verbose_payload",
    "source_gate=expanded_FIELD_namespace_audit_and_direct_exact_transform_COMPLETE",
    "keys_shuffle_uuid=exact_V1_batch_policy_and_shared_helper_contract",
    "allocation=one_live_post_shuffle_rename_pass_no_post_hoc_rerun",
    "provenance=V1_native_public_provenance_compiler_before_compact_encoding",
    "authority=compact_V2_encode_decode_reencode_exact_mapping_equality",
    "payload=closed_8_field_accepted_JCS_full_wrapper_max_65424_bytes",
    "frame=V2_magic_u16_version_u16_header_u32_payload_two_sha256",
    "padding=V2_HMAC_SHA256_domain_run_payload_digest_u32_block",
    "unlinkability=all_source_audit_UUIDs_disjoint_from_all_fresh_decoded_public_audit_UUIDs",
    "emission=whole_batch_atomic_no_partial_envelope_or_root_tuples",
)

_KEY_SOURCE_FIELDS_V2: Final = ("shuffle_ikm", "id_ikm", "padding_ikm")
_POLICY_FIELDS_V2: Final = (
    "schema_version", "payload_schema_version", "v1_batch_policy_id",
    "key_schedule_version", "public_provenance_version",
    "typed_authority_schema_id", "typed_authority_codec_version",
    "typed_authority_codec_policy_id", "exact_transform_policy_id",
    "exact_provenance_compiler_version",
    "exact_provenance_compiler_policy_id", "maximum_authorities",
    "envelope_bytes", "header_bytes", "maximum_payload_bytes",
    "minimum_padding_bytes",
)
_REJECTION_FIELDS_V2: Final = (
    "disposition", "reason", "authority_count", "schema_version", "policy_id",
    "envelopes", "envelope_ids", "authority_content_ids",
    "transform_result_ids", "claim_level", "recognizer_capacity_evidence",
    "origin_authenticated", "formal_uuid_audit", "formal_covert_audit",
    "sealed_holdout_eligible", "c1_exit_evidence",
)
_DECODED_FIELDS_V2: Final = (
    "envelope", "envelope_id", "payload_sha256", "padding_sha256",
    "payload_bytes", "padding_bytes", "namespace_audit", "namespace_audit_id",
    "authority", "authority_content_id", "transform_result_id",
    "payload_schema_version", "public_provenance_version",
    "typed_authority_schema_id", "typed_authority_codec_version",
    "typed_authority_codec_policy_id", "structural_hashes_verified",
    "public_provenance_verified", "typed_authority_decode_replay_verified",
    "direct_exact_transform_replay_verified", "secret_padding_replay_verified",
    "origin_authenticated", "formal_uuid_audit", "formal_covert_audit",
    "sealed_holdout_eligible", "c1_exit_evidence",
)
_ENVELOPE_FIELDS_V2: Final = (
    "envelope", "envelope_id", "payload_sha256", "padding_sha256",
    "payload_bytes", "padding_bytes", "namespace_audit_id",
    "authority_content_id", "transform_result_id", "typed_authority_schema_id",
    "typed_authority_codec_version", "typed_authority_codec_policy_id",
    "structural_hashes_verified", "public_provenance_verified",
    "typed_authority_decode_replay_verified",
    "direct_exact_transform_replay_verified", "secret_padding_replay_verified",
    "origin_authenticated", "formal_uuid_audit", "formal_covert_audit",
    "sealed_holdout_eligible", "c1_exit_evidence",
)
_BATCH_FIELDS_V2: Final = (
    "disposition", "schema_version", "payload_schema_version",
    "key_schedule_version", "public_provenance_version",
    "typed_authority_schema_id", "typed_authority_codec_version",
    "typed_authority_codec_policy_id",
    "exact_transform_provenance_compiler_policy_id", "jcs_profile_id",
    "field_manifest_id", "policy_id", "run_id_commitment", "envelopes",
    "envelope_ids", "authority_content_ids", "transform_result_ids", "batch_id",
    "uuid_collision_retry_count", "uuid_collision_warning", "claim_level",
    "whole_batch_shuffle_publicly_verified",
    "purpose_separated_keys_publicly_verified",
    "post_shuffle_hmac_uuidv4_publicly_verified",
    "secret_hmac_padding_publicly_verified",
    "atomic_complete_batch_structure_verified",
    "typed_authority_decode_replay_verified",
    "direct_exact_transform_replay_verified", "recognizer_capacity_evidence",
    "origin_authenticated", "formal_uuid_audit", "formal_covert_audit",
    "sealed_holdout_eligible", "c1_exit_evidence",
)
_ENVELOPE_TRUE_CLAIMS_V2: Final = (
    "structural_hashes_verified", "public_provenance_verified",
    "typed_authority_decode_replay_verified",
    "direct_exact_transform_replay_verified",
)
_ENVELOPE_FALSE_CLAIMS_V2: Final = (
    "secret_padding_replay_verified", "origin_authenticated",
    "formal_uuid_audit", "formal_covert_audit", "sealed_holdout_eligible",
    "c1_exit_evidence",
)
_REJECTION_FALSE_CLAIMS_V2: Final = (
    "recognizer_capacity_evidence", "origin_authenticated",
    "formal_uuid_audit", "formal_covert_audit", "sealed_holdout_eligible",
    "c1_exit_evidence",
)
_BATCH_TRUE_CLAIMS_V2: Final = (
    "atomic_complete_batch_structure_verified",
    "typed_authority_decode_replay_verified",
    "direct_exact_transform_replay_verified",
)
_BATCH_FALSE_CLAIMS_V2: Final = (
    "whole_batch_shuffle_publicly_verified",
    "purpose_separated_keys_publicly_verified",
    "post_shuffle_hmac_uuidv4_publicly_verified",
    "secret_hmac_padding_publicly_verified", "recognizer_capacity_evidence",
    "origin_authenticated", "formal_uuid_audit", "formal_covert_audit",
    "sealed_holdout_eligible", "c1_exit_evidence",
)


class TrustedWireBatchDispositionV2(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class TrustedWireKeySourcesV2:
    shuffle_ikm: bytes = field(repr=False)
    id_ikm: bytes = field(repr=False)
    padding_ikm: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireKeySourcesV2:
            raise TypeError("V2 trusted-wire key sources must use the exact type")
        values = (self.shuffle_ikm, self.id_ikm, self.padding_ikm)
        if any(type(value) is not bytes for value in values):
            raise TypeError("V2 trusted-wire IKM values must use exact bytes")
        if any(len(value) != IKM_BYTES for value in values):
            raise ValueError("V2 trusted-wire IKM values must be exactly 32 bytes")
        if len(set(values)) != 3:
            raise ValueError("V2 shuffle, ID, and padding IKM must be distinct")


def _ascii(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be an exact nonempty string")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use ASCII") from exc
    return value


def _json_ascii(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use ASCII") from exc
    return value


def _bounded_ascii_v2(value: object, name: str) -> str:
    """Validate exact ASCII with an O(1) character cap before encoding."""

    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if len(value) > MAXIMUM_ASCII_STRING_BYTES:
        raise ValueError(f"{name} string cap exceeded")
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use ASCII") from exc
    return value


def _digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if len(value) != len(prefix) + 64:
        raise ValueError(f"{name} must have the exact digest length")
    text = value
    if not text.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    suffix = text[len(prefix) :]
    if any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError(f"{name} must end in lowercase SHA-256")
    return text


def _hex64(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if len(value) != 64:
        raise ValueError(f"{name} must have the exact SHA-256 length")
    text = value
    if any(item not in "0123456789abcdef" for item in text):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return text


def _exact_bool(*values: object) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError("V2 trusted-wire claims must use exact booleans")


def _reject_json_float(_: str) -> object:
    raise ValueError("V2 payload raw JSON floats are forbidden")


def _reject_json_constant(_: str) -> object:
    raise ValueError("V2 payload nonfinite JSON constants are forbidden")


def _parse_json_integer(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > 16:
        raise ValueError("V2 payload integer exceeds the safe range")
    result = int(value, 10)
    if abs(result) > MAXIMUM_SAFE_INTEGER:
        raise ValueError("V2 payload integer exceeds the safe range")
    return result


def _pairs_to_exact_dict_v2(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("V2 payload object repeats a key")
        result[key] = value
    return result


def _exact_json_tree_v2(value: object) -> None:
    """Check exact JSON types; authority-specific caps belong to compact V2."""

    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        exact_type = type(current)
        if exact_type is dict:
            for key, item in current.items():
                _json_ascii(key, "V2 payload object key")
                stack.append(item)
        elif exact_type is list:
            stack.extend(current)
        elif exact_type is str:
            _json_ascii(current, "V2 payload string")
        elif exact_type is int:
            if abs(current) > MAXIMUM_SAFE_INTEGER:
                raise ValueError("V2 payload integer exceeds the safe range")
        elif exact_type in (bool, type(None)):
            pass
        else:
            raise TypeError("V2 payload contains a non-exact JSON node")


def _validate_payload_mapping_v2(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("V2 payload must be an exact object")
    if len(value) != len(TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS):
        raise ValueError("V2 payload schema closed-field drift")
    keys = tuple(value.keys())
    for key in keys:
        _bounded_ascii_v2(key, "V2 payload root key")
    if frozenset(keys) != frozenset(TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS):
        raise ValueError("V2 payload schema closed-field drift")
    expected = (
        FIELD_MANIFEST_ID,
        JCS_PROFILE_ID,
        _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
        TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION,
        COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
    )
    actual = (
        value["field_manifest_id"],
        value["jcs_profile_id"],
        value["public_provenance_version"],
        value["schema_version"],
        value["typed_authority_codec_policy_id"],
        value["typed_authority_codec_version"],
        value["typed_authority_schema_id"],
    )
    for item in actual:
        _bounded_ascii_v2(item, "V2 payload identity")
    if actual != expected:
        raise ValueError("V2 payload identity drift")
    if type(value["authority"]) is not dict:
        raise TypeError("V2 payload authority must be an exact object")
    compact, strings = _typed_v2._validate_compact_profile(value["authority"])
    expanded = _typed_v2._expand_compact_value(compact["body"], strings)
    if type(expanded) is not dict:
        raise ValueError("V2 payload compact authority root drift")
    authority = _typed_v1.decode_typed_transform_authority_profile_v1(expanded)
    logical = _typed_v1.encode_typed_transform_authority_profile_v1(authority)
    if logical != expanded or _typed_v2._compact_logical_profile(logical) != compact:
        raise ValueError("V2 payload compact authority is noncanonical")
    _exact_json_tree_v2(value)
    return value


def _encode_payload_jcs_v2(value: object) -> bytes:
    root = _validate_payload_mapping_v2(value)
    encoded = json.dumps(
        root,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")
    if len(encoded) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("V2 payload exceeds the fixed-envelope byte capacity")
    return encoded


def _decode_payload_jcs_v2(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("V2 payload must use exact bytes")
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("V2 payload exceeds the fixed-envelope byte capacity")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("V2 payload must use ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_to_exact_dict_v2,
            parse_int=_parse_json_integer,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("V2 payload is invalid JSON") from exc
    root = _validate_payload_mapping_v2(value)
    if _encode_payload_jcs_v2(root) != payload:
        raise ValueError("V2 payload is not canonical accepted JCS")
    return root


@dataclass(frozen=True, slots=True)
class TrustedWireBatchPolicyV2:
    schema_version: str = TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION
    payload_schema_version: str = TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION
    v1_batch_policy_id: str = _batch_v1.TRUSTED_WIRE_BATCH_POLICY_ID
    key_schedule_version: str = _batch_v1.TRUSTED_WIRE_KEY_SCHEDULE_VERSION
    public_provenance_version: str = (
        _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
    )
    typed_authority_schema_id: str = COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    typed_authority_codec_version: str = COMPACT_TYPED_AUTHORITY_CODEC_VERSION
    typed_authority_codec_policy_id: str = (
        COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
    )
    exact_transform_policy_id: str = EXACT_TRANSFORM_POLICY_ID
    exact_provenance_compiler_version: str = (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION
    )
    exact_provenance_compiler_policy_id: str = (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
    )
    maximum_authorities: int = MAXIMUM_BATCH_V2_AUTHORITIES
    envelope_bytes: int = ENVELOPE_BYTES
    header_bytes: int = ENVELOPE_HEADER_BYTES
    maximum_payload_bytes: int = MAXIMUM_PAYLOAD_BYTES
    minimum_padding_bytes: int = MINIMUM_PADDING_BYTES

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireBatchPolicyV2:
            raise TypeError("V2 trusted-wire policy must use the exact type")
        string_values = (
            self.schema_version, self.payload_schema_version,
            self.v1_batch_policy_id, self.key_schedule_version,
            self.public_provenance_version, self.typed_authority_schema_id,
            self.typed_authority_codec_version,
            self.typed_authority_codec_policy_id,
            self.exact_transform_policy_id,
            self.exact_provenance_compiler_version,
            self.exact_provenance_compiler_policy_id,
        )
        integer_values = (
            self.maximum_authorities, self.envelope_bytes, self.header_bytes,
            self.maximum_payload_bytes, self.minimum_padding_bytes,
        )
        if any(type(item) is not str for item in string_values):
            raise TypeError("V2 trusted-wire policy strings must be exact")
        if any(type(item) is not int for item in integer_values):
            raise TypeError("V2 trusted-wire policy integers must be exact")
        if (
            self.schema_version,
            self.payload_schema_version,
            self.v1_batch_policy_id,
            self.key_schedule_version,
            self.public_provenance_version,
            self.typed_authority_schema_id,
            self.typed_authority_codec_version,
            self.typed_authority_codec_policy_id,
            self.exact_transform_policy_id,
            self.exact_provenance_compiler_version,
            self.exact_provenance_compiler_policy_id,
            self.maximum_authorities,
            self.envelope_bytes,
            self.header_bytes,
            self.maximum_payload_bytes,
            self.minimum_padding_bytes,
        ) != (
            TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
            TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION,
            _batch_v1.TRUSTED_WIRE_BATCH_POLICY_ID,
            _batch_v1.TRUSTED_WIRE_KEY_SCHEDULE_VERSION,
            _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
            COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
            COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
            COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
            EXACT_TRANSFORM_POLICY_ID,
            EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION,
            EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
            MAXIMUM_BATCH_V2_AUTHORITIES,
            ENVELOPE_BYTES,
            ENVELOPE_HEADER_BYTES,
            MAXIMUM_PAYLOAD_BYTES,
            MINIMUM_PADDING_BYTES,
        ):
            raise ValueError("V2 trusted-wire policy drift")

    @property
    def policy_id(self) -> str:
        self.__post_init__()
        value = {
            "algorithm_profile": list(_V2_ALGORITHM_PROFILE),
            "claim_contract": {
                "envelope_true": list(_ENVELOPE_TRUE_CLAIMS_V2),
                "envelope_false": list(_ENVELOPE_FALSE_CLAIMS_V2),
                "envelope_claims_apply_to": [
                    "DecodedTrustedEnvelopeV2",
                    "TrustedWireEnvelopeV2",
                ],
                "rejection_false": list(_REJECTION_FALSE_CLAIMS_V2),
                "batch_true": list(_BATCH_TRUE_CLAIMS_V2),
                "batch_false": list(_BATCH_FALSE_CLAIMS_V2),
                "claim_level_literal": NON_AUTHORITATIVE_CLAIM_LEVEL,
                "exact_bool": True,
                "custodian_secret_replay_implemented": False,
                "secret_replay_boundary": "future_exact_V2_rebuild_API",
            },
            "compact_typed_authority": {
                "codec_policy_id": self.typed_authority_codec_policy_id,
                "codec_version": self.typed_authority_codec_version,
                "schema_id": self.typed_authority_schema_id,
            },
            "domains_hex": {
                "batch": _BATCH_DOMAIN_V2.hex(),
                "envelope_id": _ENVELOPE_ID_DOMAIN_V2.hex(),
                "namespace_audit": _NAMESPACE_AUDIT_DOMAIN_V2.hex(),
                "padding": _PADDING_DOMAIN_V2.hex(),
                "policy": _POLICY_DOMAIN_V2.hex(),
                "run_commitment": _RUN_COMMITMENT_DOMAIN_V2.hex(),
            },
            "exact_transform": {
                "provenance_compiler_policy_id": (
                    self.exact_provenance_compiler_policy_id
                ),
                "provenance_compiler_version": (
                    self.exact_provenance_compiler_version
                ),
                "validator_policy_id": self.exact_transform_policy_id,
            },
            "fixed_envelope": {
                "bytes": self.envelope_bytes,
                "header_bytes": self.header_bytes,
                "header_format": _HEADER_V2.format,
                "magic_hex": TRUSTED_WIRE_ENVELOPE_V2_MAGIC.hex(),
                "maximum_payload_bytes": self.maximum_payload_bytes,
                "minimum_padding_bytes": self.minimum_padding_bytes,
                "version": TRUSTED_WIRE_ENVELOPE_V2_VERSION,
            },
            "identity": {
                "field_manifest_id": FIELD_MANIFEST_ID,
                "jcs_profile_id": JCS_PROFILE_ID,
                "key_schedule_version": self.key_schedule_version,
                "payload_schema_version": self.payload_schema_version,
                "public_provenance_version": self.public_provenance_version,
                "schema_version": self.schema_version,
            },
            "payload_top_level_fields": list(
                TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS
            ),
            "public_dataclass_field_manifests": {
                "key_sources": list(_KEY_SOURCE_FIELDS_V2),
                "policy": list(_POLICY_FIELDS_V2),
                "rejection": list(_REJECTION_FIELDS_V2),
                "decoded_envelope": list(_DECODED_FIELDS_V2),
                "issued_envelope": list(_ENVELOPE_FIELDS_V2),
                "batch": list(_BATCH_FIELDS_V2),
            },
            "public_root_formulas": {
                "namespace_audit_ordered_preimage": [
                    "manifest_id:u16be_length_then_ascii",
                    "claim_level:u16be_length_then_ascii",
                    "formal_uuid_namespace_field_audit:u8_exact_bool",
                    "frozen_minimum_count:u16be",
                    "frozen_minimum_rows:u16be_length_then_ascii_each",
                    "schema_registry_count:u16be",
                    "schema_registry_rows:u16be_length_then_ascii_each",
                    "zero_occurrence_count:u16be",
                    "zero_occurrence_rows:u16be_length_then_ascii_each",
                    "occurrence_count:u32be",
                    "each_occurrence_namespace_path_uuid_rule:u16be_length_then_ascii_each",
                ],
                "batch_ordered_preimage": [
                    "row_count:u32be",
                    "run_commitment:u16be_length_then_ascii",
                    "policy_id:u16be_length_then_ascii",
                    "typed_schema_id:u16be_length_then_ascii",
                    "collision_retries:u64be",
                    "each_ordered_envelope_authority_result_id:u16be_length_then_ascii_each",
                ],
                "envelope_id": "sha256(domain||exact_65536_envelope)",
                "run_commitment": "sha256(domain||exact_32_byte_run_id)",
                "frame_hash_fields": "sha256(payload_bytes),sha256(secret_padding_bytes)",
                "secret_padding_block": (
                    "hmac_sha256(key,domain||run_id||payload_sha256||u32be_counter)"
                ),
                "validate_before_hash_scope": {
                    "namespace_audit": "all_ordered_preimage_fields_and_occurrence_caps",
                    "batch": "exact_bounded_rows_stored_and_projected_ID_columns_unique_retry_bound",
                    "envelope_id": "exact_65536_byte_envelope",
                    "run_commitment": "exact_32_byte_run_id",
                    "frame_hashes": "exact_payload_bytes_payload_cap_exact_32_byte_padding_key_and_run_id",
                    "padding_blocks": "exact_32_byte_key_run_payload_digest_and_bounded_length",
                },
            },
            "cross_rejection": [
                "V1_stage_A_4_field_verbose_payload_is_not_V2_8_field_payload",
                "V1_batch_payload_6_fields_and_magic_version1_are_not_V2_payload_or_frame",
                "V1_logical_typed_authority_root_is_not_compact_V2_authority_root",
                "V1_and_V2_key_envelope_batch_and_rejection_dataclass_types_are_distinct",
            ],
            "expanded_logical_and_source_caps": {
                "authorities": self.maximum_authorities,
                "array_entries": MAXIMUM_ARRAY_ENTRIES,
                "ascii_string_bytes": MAXIMUM_ASCII_STRING_BYTES,
                "contracts_per_authority": _batch_v1.MAXIMUM_CONTRACTS_PER_AUTHORITY,
                "entities_per_authority": _batch_v1.MAXIMUM_ENTITIES_PER_AUTHORITY,
                "ikm_bytes": IKM_BYTES,
                "metadata_rows_per_authority": _batch_v1.MAXIMUM_METADATA_ROWS_PER_AUTHORITY,
                "observations_per_authority": _batch_v1.MAXIMUM_OBSERVATIONS_PER_AUTHORITY,
                "profile_depth": MAXIMUM_PROFILE_DEPTH,
                "profile_nodes": MAXIMUM_PROFILE_NODES,
                "rational_bit_length": MAXIMUM_RATIONAL_BIT_LENGTH,
                "run_id_bytes": RUN_ID_BYTES,
                "safe_integer": MAXIMUM_SAFE_INTEGER,
                "shuffle_rejection_draws": MAXIMUM_SHUFFLE_REJECTION_DRAWS,
                "unique_uuids": MAXIMUM_UNIQUE_UUIDS,
                "uuid_collision_retries": MAXIMUM_UUID_COLLISION_RETRIES,
                "batch_collision_retry_count": (
                    MAXIMUM_BATCH_V2_COLLISION_RETRY_COUNT
                ),
                "private_source_and_public_uuid_sidecar_entries": (
                    _MAXIMUM_BATCH_V2_UUID_SIDECAR_ENTRIES
                ),
                "uuid_occurrences": MAXIMUM_UUID_OCCURRENCES,
            },
            "source_exact_tree_preflight": {
                "accepted_dataclass_types": sorted(
                    item.__module__ + "." + item.__qualname__
                    for item in _typed_v1._PROFILE_DATACLASS_TYPES
                ),
                "accepted_enum_types": sorted(
                    item.__module__ + "." + item.__qualname__
                    for item in _typed_v1._PROFILE_ENUM_TYPES
                ),
                "branch_order": [
                    "ExactTransformAtom_exact_integer_fields_and_4096_bit_cap_then_two_decimal_string_nodes",
                    "exact_str_len_at_most_2048_then_ASCII_total_bytes_at_most_nodes_times_256_then_bounded_UUID_counts",
                    "exact_int_safe_integer",
                    "exact_float_finite",
                    "exact_bool_or_None",
                    "exact_frozen_enum_type_then_value",
                    "exact_tuple_len_at_most_4096_then_children",
                    "exact_frozen_dataclass_type_then_fields_of_exact_type_and_safe_slot_reads",
                    "all_other_types_rejected",
                ],
                "global_caps": {
                    "depth": MAXIMUM_PROFILE_DEPTH,
                    "entries": MAXIMUM_PROFILE_NODES,
                    "nodes": MAXIMUM_PROFILE_NODES,
                    "total_string_bytes": MAXIMUM_PROFILE_NODES * 256,
                    "unique_UUIDs": MAXIMUM_UNIQUE_UUIDS,
                    "UUID_occurrences": MAXIMUM_UUID_OCCURRENCES,
                },
                "precedes_specialized_nested_lengths_and_all_V1_helpers": True,
                "reused_before_decoded_authority_codec_reencode": True,
            },
            "payload_physical_contract": {
                "authority_caps_owned_by": self.typed_authority_codec_policy_id,
                "canonical_ascii_json_sorted_keys_no_float_duplicate_keys_rejected": True,
                "decoder_shallow_order": [
                    "exact_dict",
                    "exact_8_entries",
                    "each_key_exact_str_length_at_most_2048_then_ASCII",
                    "closed_key_set_without_prevalidation_sort",
                    "seven_identity_values_exact_str_length_at_most_2048_then_ASCII_and_literal_equal",
                    "authority_exact_dict",
                    "compact_authority_resource_decode_and_canonical_reencode",
                ],
                "exact_closed_top_level_field_count": len(
                    TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS
                ),
                "frame_custody_order": (
                    "exact_32_byte_padding_key_and_run_id_before_payload_SHA256"
                ),
                "full_wrapper_byte_cap": self.maximum_payload_bytes,
                "jcs_profile_id_is_syntax_binding_not_V1_physical_tree_caps": JCS_PROFILE_ID,
                "materialization_boundary": (
                    "bounded_compact_profile_tree_and_seven_small_metadata_values_"
                    "are_materialized_by_canonical_json_encoder_then_the_full_"
                    "encoded_wrapper_must_be_at_most_65424_bytes;this_is_not_a_"
                    "pre_materialization_byte_planner"
                ),
                "safe_integer": MAXIMUM_SAFE_INTEGER,
            },
            "shared_v1_mechanics": {
                "batch_policy_id": self.v1_batch_policy_id,
                "helper_contract": [list(item) for item in _SHARED_V1_HELPER_CONTRACT],
                "key_sources_exact_type": "TrustedWireKeySourcesV2",
                "V1_derive_keys_uses_read_only_three_bytes_field_contract": True,
                "global_source_UUIDs_do_not_seed_allocated_or_change_schedule": True,
                "no_source_file_or_source_sha_runtime_dependency": True,
                "not_shared": [
                    "V1_profile_compiler",
                    "V1_batch_core",
                "V1_payload_JCS_physical_resource_checker_frame_or_decoder",
                    "V1_batch_envelope_run_or_padding_ID_domains",
                ],
            },
            "validation_order": [
                "exact_top_level_types_and_shallow_counts",
                "source_iterative_exact_V1_logical_dataclass_enum_tuple_scalar_schema_closure_and_string_length_cap_before_nested_shape_access_or_V1_helpers",
                "V1_hash_free_authority_caps",
                "V1_logical_typed_encode_namespace_audit_direct_transform_COMPLETE",
                "collect_bounded_all_source_audit_UUID_set_before_key_schedule",
                "V1_key_derivation_shuffle_single_live_rename_native_provenance",
                "compact_V2_encode_full_8_field_accepted_JCS_capacity",
                "V2_frame_then_public_self_decode_exact_authority_result_parity",
                "collect_fresh_decoded_public_audit_UUIDs_and_reject_any_global_source_intersection_before_row_append",
                "repeat_global_source_public_disjoint_assertion_before_batch_issue",
                "append_row_only_after_complete_case_then_atomic_batch_issue",
            ],
            "decoded_envelope_validation_order": [
                "exact_self_claims_envelope_authority_top_types",
                "exact_digest_scalar_length_relationship_and_identity_literals",
                "exact_source_tree_resource_preflight",
                "typed_authority_codec_canonical_reencode",
                "stored_namespace_audit_exact_replay_root",
                "full_envelope_public_replay_or_private_fresh_parts",
            ],
            "namespace_audit_validation_order": [
                "exact_four_tuple_fields_before_any_nested_sort_or_hash",
                "fixed_small_namespace_tuple_lengths_10_16_2",
                "occurrence_count_at_most_2048_before_V1_audit_validation",
                "manifest_claim_exact_str_length_at_most_2048_then_ASCII_and_formal_exact_bool",
                "fixed_tuple_elements_exact_str_length_at_most_2048_then_ASCII",
                "each_occurrence_exact_type_and_four_exact_string_fields",
                "each_occurrence_string_length_at_most_2048_then_ASCII_before_post_init_and_sort",
                "each_occurrence_V1_deep_validation_before_audit_canonical_sort",
                "V1_audit_canonical_validation",
                "ordered_streaming_hash",
            ],
            "batch_root_validation_predicates": [
                "exact_nonempty_envelope_tuple_count_at_most_1024",
                "each_row_exact_TrustedWireEnvelopeV2",
                "stored_envelope_authority_result_ID_tuples_exact_and_equal_ordered_row_projections",
                "each_ID_exact_digest_and_each_ID_column_unique",
                "collision_retry_exact_int_within_derived_bound",
                "all_predicates_complete_before_batch_hash",
            ],
            "uuid_unlinkability_gate": {
                "source": "all_UUIDs_from_all_source_FIELD_namespace_audits",
                "public": "all_UUIDs_from_each_fresh_decoded_public_FIELD_namespace_audit",
                "global_disjoint_required": True,
                "failure": "whole_batch_ABSTAIN_no_partial_public_roots",
                "private_bounded_sidecar_only": True,
                "sidecar_entry_cap_each": _MAXIMUM_BATCH_V2_UUID_SIDECAR_ENTRIES,
                "sidecars_not_hashed_serialized_or_exposed": True,
                "allocator_seed_or_schedule_change": False,
            },
            "validated_context_reuse": (
                "module_private_API_optimization_identity_tokens_only;not_an_"
                "adversarial_same_process_security_boundary;public_zero_argument_"
                "validation_always_replays_envelope_rows_and_batch_roots;all_"
                "provided_context_objects_are_exact_type_count_and_row_identity_"
                "checked_before_copy_projection_equality_or_deep_replay"
            ),
        }
        return "phase2b_trusted_wire_batch_v2_policy_" + hashlib.sha256(
            _POLICY_DOMAIN_V2 + encode_phase2b_jcs_profile_v1(value)
        ).hexdigest()


DEFAULT_TRUSTED_WIRE_BATCH_V2_POLICY: Final = TrustedWireBatchPolicyV2()
TRUSTED_WIRE_BATCH_V2_POLICY_ID: Final = (
    DEFAULT_TRUSTED_WIRE_BATCH_V2_POLICY.policy_id
)


def _secret_padding_v2(
    key: bytes,
    run_id: bytes,
    payload_sha256: bytes,
    length: int,
) -> bytes:
    if type(key) is not bytes or len(key) != 32:
        raise TypeError("V2 padding key must be exact 32 bytes")
    if type(run_id) is not bytes or len(run_id) != RUN_ID_BYTES:
        raise TypeError("V2 padding run ID must be exact 32 bytes")
    if type(payload_sha256) is not bytes or len(payload_sha256) != 32:
        raise TypeError("V2 padding payload digest must be exact 32 bytes")
    if type(length) is not int or not (
        0 <= length <= ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES
    ):
        raise ValueError("V2 padding length is invalid")
    blocks: list[bytes] = []
    remaining = length
    counter = 0
    while remaining:
        block = hmac.digest(
            key,
            _PADDING_DOMAIN_V2
            + run_id
            + payload_sha256
            + counter.to_bytes(4, "big"),
            "sha256",
        )
        blocks.append(block[:remaining])
        remaining -= min(remaining, len(block))
        counter += 1
    return b"".join(blocks)


def _frame_secret_payload_v2(
    payload: bytes,
    padding_key: bytes,
    run_id: bytes,
) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("V2 trusted payload must use exact bytes")
    if type(padding_key) is not bytes or len(padding_key) != 32:
        raise TypeError("V2 padding key must be exact 32 bytes")
    if type(run_id) is not bytes or len(run_id) != RUN_ID_BYTES:
        raise TypeError("V2 padding run ID must be exact 32 bytes")
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("V2 trusted payload exceeds fixed-envelope capacity")
    payload_digest = hashlib.sha256(payload).digest()
    padding_length = ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES - len(payload)
    if padding_length < MINIMUM_PADDING_BYTES:
        raise ValueError("V2 trusted payload leaves insufficient secret padding")
    padding = _secret_padding_v2(
        padding_key,
        run_id,
        payload_digest,
        padding_length,
    )
    header = _HEADER_V2.pack(
        TRUSTED_WIRE_ENVELOPE_V2_MAGIC,
        TRUSTED_WIRE_ENVELOPE_V2_VERSION,
        ENVELOPE_HEADER_BYTES,
        len(payload),
        payload_digest,
        hashlib.sha256(padding).digest(),
    )
    envelope = header + payload + padding
    if len(envelope) != ENVELOPE_BYTES:
        raise RuntimeError("V2 trusted envelope framing length drift")
    return envelope


@dataclass(frozen=True, slots=True)
class TrustedWireBatchRejectionV2:
    disposition: TrustedWireBatchDispositionV2
    reason: str
    authority_count: int
    schema_version: str = TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION
    policy_id: str = TRUSTED_WIRE_BATCH_V2_POLICY_ID
    envelopes: tuple[()] = ()
    envelope_ids: tuple[()] = ()
    authority_content_ids: tuple[()] = ()
    transform_result_ids: tuple[()] = ()
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    recognizer_capacity_evidence: bool = False
    origin_authenticated: bool = False
    formal_uuid_audit: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False
    c1_exit_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireBatchRejectionV2:
            raise TypeError("V2 trusted-wire rejection must use the exact type")
        if self.disposition is not TrustedWireBatchDispositionV2.ABSTAIN:
            raise ValueError("V2 trusted-wire rejection must abstain")
        _bounded_ascii_v2(self.reason, "V2 trusted-wire rejection reason")
        if not self.reason:
            raise TypeError("V2 trusted-wire rejection reason must be nonempty")
        if type(self.authority_count) is not int or self.authority_count < 0:
            raise ValueError("V2 trusted-wire rejection count is invalid")
        if (
            type(self.schema_version) is not str
            or self.schema_version != TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION
            or type(self.policy_id) is not str
            or self.policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("V2 trusted-wire rejection identity drift")
        if any(
            type(value) is not tuple or value != ()
            for value in (
                self.envelopes,
                self.envelope_ids,
                self.authority_content_ids,
                self.transform_result_ids,
            )
        ):
            raise ValueError("V2 trusted-wire rejection must expose no partial rows")
        _exact_bool(*(getattr(self, name) for name in _REJECTION_FALSE_CLAIMS_V2))
        if any(getattr(self, name) for name in _REJECTION_FALSE_CLAIMS_V2):
            raise ValueError("V2 trusted-wire rejection cannot make broad claims")


@dataclass(frozen=True, slots=True)
class _DecodedEnvelopePartsV2:
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit: NamespaceFieldAuditV1
    namespace_audit_id: str
    authority: PublicTransformEvidenceBundleV2
    authority_content_id: str
    transform_result_id: str
    payload_schema_version: str
    public_provenance_version: str
    typed_authority_schema_id: str
    typed_authority_codec_version: str
    typed_authority_codec_policy_id: str


_DECODED_CONTEXT_TOKEN_V2: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class DecodedTrustedEnvelopeV2:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit: NamespaceFieldAuditV1
    namespace_audit_id: str
    authority: PublicTransformEvidenceBundleV2
    authority_content_id: str
    transform_result_id: str
    payload_schema_version: str
    public_provenance_version: str
    typed_authority_schema_id: str
    typed_authority_codec_version: str
    typed_authority_codec_policy_id: str
    structural_hashes_verified: bool = True
    public_provenance_verified: bool = True
    typed_authority_decode_replay_verified: bool = True
    direct_exact_transform_replay_verified: bool = True
    secret_padding_replay_verified: bool = False
    origin_authenticated: bool = False
    formal_uuid_audit: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False
    c1_exit_evidence: bool = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("decoded V2 envelopes are issued only by the exact decoder")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        envelope: bytes,
        parts: _DecodedEnvelopePartsV2,
    ) -> "DecodedTrustedEnvelopeV2":
        if token is not _DECODED_CONTEXT_TOKEN_V2:
            raise TypeError("decoded V2 envelope issuer token mismatch")
        if type(parts) is not _DecodedEnvelopePartsV2:
            raise TypeError("decoded V2 envelope parts must use the exact type")
        if type(envelope) is not bytes or len(envelope) != ENVELOPE_BYTES:
            raise TypeError("decoded V2 envelope issuer needs exact fixed bytes")
        value = object.__new__(cls)
        object.__setattr__(value, "envelope", envelope)
        for item in fields(_DecodedEnvelopePartsV2):
            object.__setattr__(value, item.name, getattr(parts, item.name))
        for name in _ENVELOPE_TRUE_CLAIMS_V2:
            object.__setattr__(value, name, True)
        for name in _ENVELOPE_FALSE_CLAIMS_V2:
            object.__setattr__(value, name, False)
        value._validate(parts=parts, context_token=_DECODED_CONTEXT_TOKEN_V2)
        return value

    def _validate(
        self,
        *,
        parts: _DecodedEnvelopePartsV2 | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not DecodedTrustedEnvelopeV2:
            raise TypeError("decoded V2 trusted envelope must use the exact type")
        if (parts is None) is not (context_token is None):
            raise TypeError("decoded V2 envelope context token mismatch")
        if parts is not None and context_token is not _DECODED_CONTEXT_TOKEN_V2:
            raise TypeError("decoded V2 envelope context is private")
        if parts is not None and type(parts) is not _DecodedEnvelopePartsV2:
            raise TypeError("decoded V2 envelope parts must use the exact type")
        _exact_bool(*(getattr(self, name) for name in (
            *_ENVELOPE_TRUE_CLAIMS_V2, *_ENVELOPE_FALSE_CLAIMS_V2
        )))
        if not all(getattr(self, name) for name in _ENVELOPE_TRUE_CLAIMS_V2) or any(
            getattr(self, name) for name in _ENVELOPE_FALSE_CLAIMS_V2
        ):
            raise ValueError("decoded V2 envelope claim boundary drift")
        if type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_BYTES:
            raise TypeError("decoded V2 envelope needs exact fixed bytes")
        if type(self.authority) is not PublicTransformEvidenceBundleV2:
            raise TypeError("decoded V2 envelope authority must use the exact type")
        for value, prefix, name in (
            (self.envelope_id, "phase2b_trusted_envelope_v2_", "envelope ID"),
            (
                self.authority_content_id,
                "phase2b_public_transform_evidence_",
                "authority content ID",
            ),
            (
                self.transform_result_id,
                "phase2b_exact_transform_result_",
                "transform result ID",
            ),
            (
                self.namespace_audit_id,
                "phase2b_namespace_audit_v2_",
                "namespace audit ID",
            ),
        ):
            _digest(value, prefix, name)
        _hex64(self.payload_sha256, "payload SHA-256")
        _hex64(self.padding_sha256, "padding SHA-256")
        if (
            type(self.payload_bytes) is not int
            or type(self.padding_bytes) is not int
            or type(self.namespace_audit) is not NamespaceFieldAuditV1
        ):
            raise TypeError("decoded V2 envelope scalar or audit type drift")
        if (
            not 0 <= self.payload_bytes <= MAXIMUM_PAYLOAD_BYTES
            or self.padding_bytes < MINIMUM_PADDING_BYTES
            or ENVELOPE_HEADER_BYTES + self.payload_bytes + self.padding_bytes
            != ENVELOPE_BYTES
        ):
            raise ValueError("decoded V2 envelope length relationship drift")
        for value, expected, name in (
            (self.payload_schema_version, TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION, "payload schema"),
            (self.public_provenance_version, _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION, "provenance version"),
            (self.typed_authority_schema_id, COMPACT_TYPED_AUTHORITY_SCHEMA_ID, "typed schema"),
            (self.typed_authority_codec_version, COMPACT_TYPED_AUTHORITY_CODEC_VERSION, "typed codec version"),
            (self.typed_authority_codec_policy_id, COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID, "typed codec policy"),
        ):
            if type(value) is not str or value != expected:
                raise ValueError(f"decoded V2 envelope {name} drift")
        authority_reason = _source_string_cap_preflight_v2(self.authority)
        if authority_reason is not None:
            raise ValueError(
                "decoded V2 envelope authority shallow validation failed:"
                + authority_reason
            )
        encode_typed_transform_authority_profile_v2(self.authority)
        if _namespace_audit_id_v2(self.namespace_audit) != self.namespace_audit_id:
            raise ValueError("decoded V2 envelope stored namespace audit drift")
        replay = (
            _decode_structural_envelope_v2(self.envelope)
            if parts is None
            else parts
        )
        if type(replay) is not _DecodedEnvelopePartsV2:
            raise TypeError("decoded V2 envelope replay parts drift")
        if (
            self.envelope_id != replay.envelope_id
            or self.payload_sha256 != replay.payload_sha256
            or self.padding_sha256 != replay.padding_sha256
            or self.payload_bytes != replay.payload_bytes
            or self.padding_bytes != replay.padding_bytes
            or self.namespace_audit != replay.namespace_audit
            or self.namespace_audit_id != replay.namespace_audit_id
            or self.authority != replay.authority
            or self.authority_content_id != replay.authority_content_id
            or self.transform_result_id != replay.transform_result_id
            or self.payload_schema_version != replay.payload_schema_version
            or self.public_provenance_version != replay.public_provenance_version
            or self.typed_authority_schema_id
            != replay.typed_authority_schema_id
            or self.typed_authority_codec_version
            != replay.typed_authority_codec_version
            or self.typed_authority_codec_policy_id
            != replay.typed_authority_codec_policy_id
        ):
            raise ValueError("decoded V2 envelope receipt drifts from its bytes")


_ENVELOPE_ISSUE_TOKEN_V2: Final = object()
_ENVELOPE_CONTEXT_TOKEN_V2: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class TrustedWireEnvelopeV2:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit_id: str
    authority_content_id: str
    transform_result_id: str
    typed_authority_schema_id: str
    typed_authority_codec_version: str
    typed_authority_codec_policy_id: str
    structural_hashes_verified: bool = True
    public_provenance_verified: bool = True
    typed_authority_decode_replay_verified: bool = True
    direct_exact_transform_replay_verified: bool = True
    secret_padding_replay_verified: bool = False
    origin_authenticated: bool = False
    formal_uuid_audit: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False
    c1_exit_evidence: bool = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 trusted envelopes are issued only by the exact builder")

    @classmethod
    def _issue(
        cls,
        token: object,
        decoded: DecodedTrustedEnvelopeV2,
    ) -> "TrustedWireEnvelopeV2":
        if token is not _ENVELOPE_ISSUE_TOKEN_V2:
            raise TypeError("V2 trusted envelope issuer token mismatch")
        if type(decoded) is not DecodedTrustedEnvelopeV2:
            raise TypeError("V2 trusted envelope issuer needs exact decoded input")
        value = object.__new__(cls)
        frozen = (
            ("envelope", decoded.envelope),
            ("envelope_id", decoded.envelope_id),
            ("payload_sha256", decoded.payload_sha256),
            ("padding_sha256", decoded.padding_sha256),
            ("payload_bytes", decoded.payload_bytes),
            ("padding_bytes", decoded.padding_bytes),
            ("namespace_audit_id", decoded.namespace_audit_id),
            ("authority_content_id", decoded.authority_content_id),
            ("transform_result_id", decoded.transform_result_id),
            ("typed_authority_schema_id", decoded.typed_authority_schema_id),
            (
                "typed_authority_codec_version",
                decoded.typed_authority_codec_version,
            ),
            (
                "typed_authority_codec_policy_id",
                decoded.typed_authority_codec_policy_id,
            ),
            ("structural_hashes_verified", True),
            ("public_provenance_verified", True),
            ("typed_authority_decode_replay_verified", True),
            ("direct_exact_transform_replay_verified", True),
            ("secret_padding_replay_verified", False),
            ("origin_authenticated", False),
            ("formal_uuid_audit", False),
            ("formal_covert_audit", False),
            ("sealed_holdout_eligible", False),
            ("c1_exit_evidence", False),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        value._validate(
            decoded=decoded,
            context_token=_ENVELOPE_CONTEXT_TOKEN_V2,
        )
        return value

    def _validate(
        self,
        *,
        decoded: DecodedTrustedEnvelopeV2 | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not TrustedWireEnvelopeV2:
            raise TypeError("V2 trusted envelope must use the exact type")
        if (decoded is None) is not (context_token is None):
            raise TypeError("V2 trusted envelope context token mismatch")
        if decoded is not None and context_token is not _ENVELOPE_CONTEXT_TOKEN_V2:
            raise TypeError("V2 trusted envelope context is private")
        if decoded is not None and type(decoded) is not DecodedTrustedEnvelopeV2:
            raise TypeError("V2 trusted envelope decoded context must be exact")
        _exact_bool(*(getattr(self, name) for name in (
            *_ENVELOPE_TRUE_CLAIMS_V2, *_ENVELOPE_FALSE_CLAIMS_V2
        )))
        if not all(getattr(self, name) for name in _ENVELOPE_TRUE_CLAIMS_V2) or any(
            getattr(self, name) for name in _ENVELOPE_FALSE_CLAIMS_V2
        ):
            raise ValueError("V2 trusted envelope claim boundary drift")
        if type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_BYTES:
            raise TypeError("V2 trusted envelope needs exact fixed bytes")
        _digest(self.envelope_id, "phase2b_trusted_envelope_v2_", "envelope ID")
        _hex64(self.payload_sha256, "payload SHA-256")
        _hex64(self.padding_sha256, "padding SHA-256")
        if type(self.payload_bytes) is not int or type(self.padding_bytes) is not int:
            raise TypeError("V2 trusted envelope lengths must be exact integers")
        if (
            not 0 <= self.payload_bytes <= MAXIMUM_PAYLOAD_BYTES
            or self.padding_bytes < MINIMUM_PADDING_BYTES
            or ENVELOPE_HEADER_BYTES + self.payload_bytes + self.padding_bytes
            != ENVELOPE_BYTES
        ):
            raise ValueError("V2 trusted envelope length relationship drift")
        _digest(self.namespace_audit_id, "phase2b_namespace_audit_v2_", "namespace audit ID")
        _digest(self.authority_content_id, "phase2b_public_transform_evidence_", "authority content ID")
        _digest(self.transform_result_id, "phase2b_exact_transform_result_", "transform result ID")
        for value, expected, name in (
            (self.typed_authority_schema_id, COMPACT_TYPED_AUTHORITY_SCHEMA_ID, "typed schema"),
            (self.typed_authority_codec_version, COMPACT_TYPED_AUTHORITY_CODEC_VERSION, "typed codec version"),
            (self.typed_authority_codec_policy_id, COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID, "typed codec policy"),
        ):
            if type(value) is not str or value != expected:
                raise ValueError(f"V2 trusted envelope {name} drift")
        replay = (
            decode_and_audit_trusted_envelope_v2(self.envelope)
            if decoded is None
            else decoded
        )
        if type(replay) is not DecodedTrustedEnvelopeV2:
            raise TypeError("V2 trusted envelope decoded context drift")
        if (
            self.envelope_id != replay.envelope_id
            or self.payload_sha256 != replay.payload_sha256
            or self.padding_sha256 != replay.padding_sha256
            or self.payload_bytes != replay.payload_bytes
            or self.padding_bytes != replay.padding_bytes
            or self.namespace_audit_id != replay.namespace_audit_id
            or self.authority_content_id != replay.authority_content_id
            or self.transform_result_id != replay.transform_result_id
            or self.typed_authority_schema_id
            != replay.typed_authority_schema_id
            or self.typed_authority_codec_version
            != replay.typed_authority_codec_version
            or self.typed_authority_codec_policy_id
            != replay.typed_authority_codec_policy_id
        ):
            raise ValueError("V2 trusted envelope receipt drifts from bytes")


def _decode_structural_envelope_v2(
    envelope: bytes,
) -> _DecodedEnvelopePartsV2:
    if type(envelope) is not bytes:
        raise TypeError("V2 trusted envelope must use exact bytes")
    if len(envelope) != ENVELOPE_BYTES:
        raise ValueError("V2 trusted envelope must be exactly 65,536 bytes")
    magic, version, header_length, payload_length, payload_hash, padding_hash = (
        _HEADER_V2.unpack(envelope[:ENVELOPE_HEADER_BYTES])
    )
    if (
        magic != TRUSTED_WIRE_ENVELOPE_V2_MAGIC
        or version != TRUSTED_WIRE_ENVELOPE_V2_VERSION
    ):
        raise ValueError("V2 trusted envelope magic or version drift")
    if (
        header_length != ENVELOPE_HEADER_BYTES
        or payload_length > MAXIMUM_PAYLOAD_BYTES
    ):
        raise ValueError("V2 trusted envelope header or payload length drift")
    payload = envelope[
        ENVELOPE_HEADER_BYTES : ENVELOPE_HEADER_BYTES + payload_length
    ]
    padding = envelope[ENVELOPE_HEADER_BYTES + payload_length :]
    if len(padding) < MINIMUM_PADDING_BYTES:
        raise ValueError("V2 trusted envelope padding is too short")
    if hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("V2 trusted envelope payload hash drift")
    if hashlib.sha256(padding).digest() != padding_hash:
        raise ValueError("V2 trusted envelope padding hash drift")
    decoded = _decode_payload_jcs_v2(payload)
    authority = decode_typed_transform_authority_profile_v2(decoded["authority"])
    if encode_typed_transform_authority_profile_v2(authority) != decoded["authority"]:
        raise ValueError("V2 compact authority canonical replay drift")
    logical = encode_typed_transform_authority_profile_v1(authority)
    audit = audit_namespace_paths_v1(logical)
    _batch_v1._audit_public_provenance(logical)
    audit_id = _namespace_audit_id_v2(audit)
    transform = run_exact_transform_semantics(authority)
    if (
        type(transform) is not ExactTransformCompilation
        or transform.disposition is not TransformCompilationDisposition.COMPLETE
    ):
        reason = getattr(transform, "reason", "transform_not_complete")
        raise ValueError("V2 direct exact transform replay failed:" + str(reason))
    return _DecodedEnvelopePartsV2(
        envelope_id="phase2b_trusted_envelope_v2_"
        + hashlib.sha256(_ENVELOPE_ID_DOMAIN_V2 + envelope).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        padding_sha256=hashlib.sha256(padding).hexdigest(),
        payload_bytes=len(payload),
        padding_bytes=len(padding),
        namespace_audit=audit,
        namespace_audit_id=audit_id,
        authority=authority,
        authority_content_id=authority.content_id,
        transform_result_id=transform.result_id,
        payload_schema_version=decoded["schema_version"],
        public_provenance_version=decoded["public_provenance_version"],
        typed_authority_schema_id=decoded["typed_authority_schema_id"],
        typed_authority_codec_version=decoded["typed_authority_codec_version"],
        typed_authority_codec_policy_id=decoded["typed_authority_codec_policy_id"],
    )


def decode_and_audit_trusted_envelope_v2(
    envelope: bytes,
) -> DecodedTrustedEnvelopeV2:
    """Replay V2 public frame, compact authority, provenance, and transform."""

    parts = _decode_structural_envelope_v2(envelope)
    return DecodedTrustedEnvelopeV2._issue(
        _DECODED_CONTEXT_TOKEN_V2,
        envelope=envelope,
        parts=parts,
    )


def _run_commitment_v2(run_id: bytes) -> str:
    if type(run_id) is not bytes or len(run_id) != RUN_ID_BYTES:
        raise TypeError("V2 run commitment needs exact 32-byte run ID")
    return "phase2b_trusted_wire_run_v2_" + hashlib.sha256(
        _RUN_COMMITMENT_DOMAIN_V2 + run_id
    ).hexdigest()


def _framed_ascii(value: object, name: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if len(value) > 65_535:
        raise ValueError(f"{name} exceeds the framed-string cap")
    try:
        text = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use ASCII") from exc
    return len(text).to_bytes(2, "big") + text


def _namespace_audit_id_v2(audit: NamespaceFieldAuditV1) -> str:
    if type(audit) is not NamespaceFieldAuditV1:
        raise TypeError("V2 namespace root needs exact namespace audit")
    try:
        manifest_id = audit.manifest_id
        frozen = audit.frozen_minimum_namespaces
        registry = audit.schema_registry_namespaces
        zero = audit.zero_occurrence_namespaces
        occurrences = audit.occurrences
        claim_level = audit.claim_level
        formal = audit.formal_uuid_namespace_field_audit
    except AttributeError as exc:
        raise TypeError("V2 namespace audit is missing a frozen field") from exc
    _bounded_ascii_v2(manifest_id, "V2 namespace manifest ID")
    _bounded_ascii_v2(claim_level, "V2 namespace claim level")
    if type(formal) is not bool:
        raise TypeError("V2 namespace audit formal claim must be exact bool")
    for value, name in (
        (frozen, "frozen namespaces"),
        (registry, "registry namespaces"),
        (zero, "zero namespaces"),
        (occurrences, "occurrences"),
    ):
        if type(value) is not tuple:
            raise TypeError(f"V2 namespace audit {name} must use exact tuple")
    if (len(frozen), len(registry), len(zero)) != (10, 16, 2):
        raise ValueError("V2 namespace audit fixed tuple lengths drift")
    if len(occurrences) > MAXIMUM_UUID_OCCURRENCES:
        raise ValueError("V2 namespace audit occurrence cap exceeded")
    for values, name in (
        (frozen, "V2 frozen namespace"),
        (registry, "V2 registry namespace"),
        (zero, "V2 zero-occurrence namespace"),
    ):
        for item in values:
            _bounded_ascii_v2(item, name)
    for occurrence in occurrences:
        if type(occurrence) is not NamespaceOccurrenceV1:
            raise TypeError("V2 namespace occurrence must use exact type")
        try:
            occurrence_fields = (
                occurrence.namespace,
                occurrence.json_pointer,
                occurrence.public_uuid,
                occurrence.rule_id,
            )
        except AttributeError as exc:
            raise TypeError("V2 namespace occurrence is missing a field") from exc
        if any(type(item) is not str for item in occurrence_fields):
            raise TypeError("V2 namespace occurrence strings must be exact")
        for item in occurrence_fields:
            _bounded_ascii_v2(item, "V2 namespace occurrence")
        occurrence.__post_init__()
    audit.__post_init__()
    digest = hashlib.sha256()
    digest.update(_NAMESPACE_AUDIT_DOMAIN_V2)
    digest.update(_framed_ascii(manifest_id, "namespace manifest ID"))
    digest.update(_framed_ascii(claim_level, "namespace claim level"))
    digest.update(bytes((int(formal),)))
    digest.update(len(frozen).to_bytes(2, "big"))
    for item in frozen:
        digest.update(_framed_ascii(item, "frozen namespace"))
    digest.update(len(registry).to_bytes(2, "big"))
    for item in registry:
        digest.update(_framed_ascii(item, "registry namespace"))
    digest.update(len(zero).to_bytes(2, "big"))
    for item in zero:
        digest.update(_framed_ascii(item, "zero-occurrence namespace"))
    digest.update(len(occurrences).to_bytes(4, "big"))
    for occurrence in occurrences:
        digest.update(_framed_ascii(occurrence.namespace, "occurrence namespace"))
        digest.update(_framed_ascii(occurrence.json_pointer, "occurrence path"))
        digest.update(_framed_ascii(occurrence.public_uuid, "occurrence UUID"))
        digest.update(_framed_ascii(occurrence.rule_id, "occurrence rule ID"))
    return "phase2b_namespace_audit_v2_" + digest.hexdigest()


def _batch_id_v2(
    run_id_commitment: str,
    envelopes: tuple[TrustedWireEnvelopeV2, ...],
    collision_retries: int,
) -> str:
    _digest(
        run_id_commitment,
        "phase2b_trusted_wire_run_v2_",
        "V2 run commitment",
    )
    if type(envelopes) is not tuple or not envelopes:
        raise TypeError("V2 batch root needs exact nonempty envelope tuple")
    if type(collision_retries) is not int or not (
        0 <= collision_retries <= MAXIMUM_BATCH_V2_COLLISION_RETRY_COUNT
    ):
        raise ValueError("V2 batch root collision count is invalid")
    if len(envelopes) > MAXIMUM_BATCH_V2_AUTHORITIES:
        raise ValueError("V2 batch root exceeds the row cap")
    for item in envelopes:
        if type(item) is not TrustedWireEnvelopeV2:
            raise TypeError("V2 batch root needs exact envelope rows")
        _digest(item.envelope_id, "phase2b_trusted_envelope_v2_", "envelope ID")
        _digest(
            item.authority_content_id,
            "phase2b_public_transform_evidence_",
            "authority content ID",
        )
        _digest(
            item.transform_result_id,
            "phase2b_exact_transform_result_",
            "transform result ID",
        )
    for values in (
        tuple(item.envelope_id for item in envelopes),
        tuple(item.authority_content_id for item in envelopes),
        tuple(item.transform_result_id for item in envelopes),
    ):
        if len(set(values)) != len(values):
            raise ValueError("V2 batch root public IDs repeat")
    digest = hashlib.sha256()
    digest.update(_BATCH_DOMAIN_V2)
    digest.update(len(envelopes).to_bytes(4, "big"))
    digest.update(_framed_ascii(run_id_commitment, "V2 run commitment"))
    digest.update(_framed_ascii(TRUSTED_WIRE_BATCH_V2_POLICY_ID, "V2 policy ID"))
    digest.update(
        _framed_ascii(COMPACT_TYPED_AUTHORITY_SCHEMA_ID, "compact schema ID")
    )
    digest.update(collision_retries.to_bytes(8, "big"))
    for item in envelopes:
        digest.update(_framed_ascii(item.envelope_id, "envelope ID"))
        digest.update(_framed_ascii(item.authority_content_id, "authority content ID"))
        digest.update(_framed_ascii(item.transform_result_id, "transform result ID"))
    return "phase2b_trusted_wire_batch_v2_" + digest.hexdigest()


_BATCH_ISSUE_TOKEN_V2: Final = object()
_BATCH_CONTEXT_TOKEN_V2: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class TrustedWireBatchV2:
    disposition: TrustedWireBatchDispositionV2
    schema_version: str
    payload_schema_version: str
    key_schedule_version: str
    public_provenance_version: str
    typed_authority_schema_id: str
    typed_authority_codec_version: str
    typed_authority_codec_policy_id: str
    exact_transform_provenance_compiler_policy_id: str
    jcs_profile_id: str
    field_manifest_id: str
    policy_id: str
    run_id_commitment: str
    envelopes: tuple[TrustedWireEnvelopeV2, ...]
    envelope_ids: tuple[str, ...]
    authority_content_ids: tuple[str, ...]
    transform_result_ids: tuple[str, ...]
    batch_id: str
    uuid_collision_retry_count: int
    uuid_collision_warning: bool
    claim_level: str
    whole_batch_shuffle_publicly_verified: bool
    purpose_separated_keys_publicly_verified: bool
    post_shuffle_hmac_uuidv4_publicly_verified: bool
    secret_hmac_padding_publicly_verified: bool
    atomic_complete_batch_structure_verified: bool
    typed_authority_decode_replay_verified: bool
    direct_exact_transform_replay_verified: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 trusted-wire batches are issued only by the exact builder")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        run_id_commitment: str,
        envelopes: tuple[TrustedWireEnvelopeV2, ...],
        uuid_collision_retry_count: int,
    ) -> "TrustedWireBatchV2":
        if token is not _BATCH_ISSUE_TOKEN_V2:
            raise TypeError("V2 trusted-wire batch issuer token mismatch")
        if (
            type(envelopes) is not tuple
            or not 1 <= len(envelopes) <= MAXIMUM_BATCH_V2_AUTHORITIES
            or any(type(item) is not TrustedWireEnvelopeV2 for item in envelopes)
        ):
            raise TypeError("V2 batch issuer needs exact bounded envelope rows")
        value = object.__new__(cls)
        envelope_ids = tuple(item.envelope_id for item in envelopes)
        authority_ids = tuple(item.authority_content_id for item in envelopes)
        result_ids = tuple(item.transform_result_id for item in envelopes)
        frozen = (
            ("disposition", TrustedWireBatchDispositionV2.COMPLETE),
            ("schema_version", TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION),
            ("payload_schema_version", TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION),
            ("key_schedule_version", _batch_v1.TRUSTED_WIRE_KEY_SCHEDULE_VERSION),
            ("public_provenance_version", _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION),
            ("typed_authority_schema_id", COMPACT_TYPED_AUTHORITY_SCHEMA_ID),
            ("typed_authority_codec_version", COMPACT_TYPED_AUTHORITY_CODEC_VERSION),
            ("typed_authority_codec_policy_id", COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID),
            ("exact_transform_provenance_compiler_policy_id", EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID),
            ("jcs_profile_id", JCS_PROFILE_ID),
            ("field_manifest_id", FIELD_MANIFEST_ID),
            ("policy_id", TRUSTED_WIRE_BATCH_V2_POLICY_ID),
            ("run_id_commitment", run_id_commitment),
            ("envelopes", envelopes),
            ("envelope_ids", envelope_ids),
            ("authority_content_ids", authority_ids),
            ("transform_result_ids", result_ids),
            ("batch_id", _batch_id_v2(run_id_commitment, envelopes, uuid_collision_retry_count)),
            ("uuid_collision_retry_count", uuid_collision_retry_count),
            ("uuid_collision_warning", uuid_collision_retry_count > 0),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("whole_batch_shuffle_publicly_verified", False),
            ("purpose_separated_keys_publicly_verified", False),
            ("post_shuffle_hmac_uuidv4_publicly_verified", False),
            ("secret_hmac_padding_publicly_verified", False),
            ("atomic_complete_batch_structure_verified", True),
            ("typed_authority_decode_replay_verified", True),
            ("direct_exact_transform_replay_verified", True),
            ("recognizer_capacity_evidence", False),
            ("origin_authenticated", False),
            ("formal_uuid_audit", False),
            ("formal_covert_audit", False),
            ("sealed_holdout_eligible", False),
            ("c1_exit_evidence", False),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        value._validate(
            validated_rows=envelopes,
            context_token=_BATCH_CONTEXT_TOKEN_V2,
        )
        return value

    def _validate(
        self,
        *,
        validated_rows: tuple[TrustedWireEnvelopeV2, ...] | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not TrustedWireBatchV2:
            raise TypeError("V2 trusted-wire batch must use the exact type")
        if (validated_rows is None) is not (context_token is None):
            raise TypeError("V2 trusted-wire batch context token mismatch")
        if validated_rows is not None and context_token is not _BATCH_CONTEXT_TOKEN_V2:
            raise TypeError("V2 trusted-wire batch context is private")
        if validated_rows is not None and (
            type(validated_rows) is not tuple
            or not 1 <= len(validated_rows) <= MAXIMUM_BATCH_V2_AUTHORITIES
            or any(
                type(item) is not TrustedWireEnvelopeV2
                for item in validated_rows
            )
        ):
            raise TypeError("V2 trusted-wire validated rows must be exact and bounded")
        if self.disposition is not TrustedWireBatchDispositionV2.COMPLETE:
            raise ValueError("V2 trusted-wire batch must be complete")
        _exact_bool(*(getattr(self, name) for name in (
            *_BATCH_TRUE_CLAIMS_V2, *_BATCH_FALSE_CLAIMS_V2,
            "uuid_collision_warning",
        )))
        if not all(getattr(self, name) for name in _BATCH_TRUE_CLAIMS_V2) or any(
            getattr(self, name) for name in _BATCH_FALSE_CLAIMS_V2
        ):
            raise ValueError("V2 trusted-wire batch claim boundary drift")
        if (
            type(self.schema_version) is not str
            or self.schema_version != TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION
            or type(self.payload_schema_version) is not str
            or self.payload_schema_version != TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION
            or type(self.key_schedule_version) is not str
            or self.key_schedule_version != _batch_v1.TRUSTED_WIRE_KEY_SCHEDULE_VERSION
            or type(self.public_provenance_version) is not str
            or self.public_provenance_version != _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
            or type(self.typed_authority_schema_id) is not str
            or self.typed_authority_schema_id != COMPACT_TYPED_AUTHORITY_SCHEMA_ID
            or type(self.typed_authority_codec_version) is not str
            or self.typed_authority_codec_version != COMPACT_TYPED_AUTHORITY_CODEC_VERSION
            or type(self.typed_authority_codec_policy_id) is not str
            or self.typed_authority_codec_policy_id != COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
            or type(self.exact_transform_provenance_compiler_policy_id) is not str
            or self.exact_transform_provenance_compiler_policy_id != EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
            or type(self.jcs_profile_id) is not str
            or self.jcs_profile_id != JCS_PROFILE_ID
            or type(self.field_manifest_id) is not str
            or self.field_manifest_id != FIELD_MANIFEST_ID
            or type(self.policy_id) is not str
            or self.policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("V2 trusted-wire batch identity drift")
        _digest(self.run_id_commitment, "phase2b_trusted_wire_run_v2_", "V2 run commitment")
        if (
            type(self.envelopes) is not tuple
            or not 1 <= len(self.envelopes) <= MAXIMUM_BATCH_V2_AUTHORITIES
            or any(type(item) is not TrustedWireEnvelopeV2 for item in self.envelopes)
        ):
            raise TypeError("V2 trusted-wire batch needs exact envelope rows")
        stored_id_columns = (
            (
                self.envelope_ids,
                "phase2b_trusted_envelope_v2_",
                "envelope IDs",
            ),
            (
                self.authority_content_ids,
                "phase2b_public_transform_evidence_",
                "authority IDs",
            ),
            (
                self.transform_result_ids,
                "phase2b_exact_transform_result_",
                "transform result IDs",
            ),
        )
        for actual, prefix, name in stored_id_columns:
            if (
                type(actual) is not tuple
                or len(actual) != len(self.envelopes)
                or len(actual) > MAXIMUM_BATCH_V2_AUTHORITIES
            ):
                raise ValueError(f"V2 trusted-wire stored {name} shape drift")
            for item in actual:
                _digest(item, prefix, name)
            if len(set(actual)) != len(actual):
                raise ValueError(f"V2 trusted-wire stored {name} repeat")
        for row in self.envelopes:
            _digest(
                row.envelope_id,
                "phase2b_trusted_envelope_v2_",
                "row envelope ID",
            )
            _digest(
                row.authority_content_id,
                "phase2b_public_transform_evidence_",
                "row authority content ID",
            )
            _digest(
                row.transform_result_id,
                "phase2b_exact_transform_result_",
                "row transform result ID",
            )
        expected_columns = (
            tuple(item.envelope_id for item in self.envelopes),
            tuple(item.authority_content_id for item in self.envelopes),
            tuple(item.transform_result_id for item in self.envelopes),
        )
        for (actual, _, name), expected in zip(
            stored_id_columns,
            expected_columns,
        ):
            if not all(
                actual_item == expected_item
                for actual_item, expected_item in zip(actual, expected)
            ):
                raise ValueError(f"V2 trusted-wire {name} drift")
        if type(self.uuid_collision_retry_count) is not int or not (
            0 <= self.uuid_collision_retry_count
            <= MAXIMUM_BATCH_V2_COLLISION_RETRY_COUNT
        ):
            raise ValueError("V2 collision retry count is invalid")
        if type(self.uuid_collision_warning) is not bool or self.uuid_collision_warning is not (self.uuid_collision_retry_count > 0):
            raise ValueError("V2 collision warning drift")
        _digest(self.batch_id, "phase2b_trusted_wire_batch_v2_", "V2 batch ID")
        if validated_rows is None:
            for item in self.envelopes:
                item._validate()
        elif len(validated_rows) != len(self.envelopes) or not all(
            validated is stored
            for validated, stored in zip(validated_rows, self.envelopes)
        ):
            raise ValueError("V2 trusted-wire batch validated rows drift")
        if self.batch_id != _batch_id_v2(self.run_id_commitment, self.envelopes, self.uuid_collision_retry_count):
            raise ValueError("V2 trusted-wire batch root drift")


@dataclass(frozen=True, slots=True)
class _SourceAuthorityV2:
    logical_profile: dict[str, object]
    namespace_audit: NamespaceFieldAuditV1


class _SourcePublicUUIDCollisionV2(ValueError):
    pass


def _reject_v2(reason: str, count: int) -> TrustedWireBatchRejectionV2:
    return TrustedWireBatchRejectionV2(
        disposition=TrustedWireBatchDispositionV2.ABSTAIN,
        reason=reason,
        authority_count=count,
    )


def _source_string_cap_preflight_v2(
    authority: PublicTransformEvidenceBundleV2,
) -> str | None:
    """Mirror V1 logical resources with an exact closed-schema traversal."""

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
        if exact_type is _typed_v1.ExactTransformAtom:
            try:
                numerator = value.numerator
                denominator = value.denominator
            except AttributeError:
                return "exact_atom_field_missing"
            if type(numerator) is not int or type(denominator) is not int:
                return "exact_atom_integer_type_drift"
            if (
                numerator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
                or denominator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
            ):
                return "rational_bit_length_exceeded"
            entries += 2
            stack.append((str(numerator), depth + 1))
            stack.append((str(denominator), depth + 1))
        elif exact_type is str:
            if len(value) > MAXIMUM_ASCII_STRING_BYTES:
                return "string_budget_exceeded"
            try:
                encoded = value.encode("ascii", errors="strict")
            except UnicodeEncodeError:
                return "non_ascii_string"
            total_string_bytes += len(encoded)
            if total_string_bytes > MAXIMUM_PROFILE_NODES * 256:
                return "total_string_budget_exceeded"
            if _typed_v1._UUID4.fullmatch(value) is not None:
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
        elif exact_type in _typed_v1._PROFILE_ENUM_TYPES:
            stack.append((value.value, depth + 1))
        elif exact_type is tuple:
            if len(value) > MAXIMUM_ARRAY_ENTRIES:
                return "array_budget_exceeded"
            entries += len(value)
            stack.extend((item, depth + 1) for item in value)
        elif exact_type in _typed_v1._PROFILE_DATACLASS_TYPES:
            rows = fields(exact_type)
            if len(rows) > MAXIMUM_ARRAY_ENTRIES:
                return "object_budget_exceeded"
            entries += len(rows)
            try:
                stack.extend(
                    (getattr(value, item.name), depth + 1) for item in rows
                )
            except AttributeError:
                return "dataclass_field_missing"
        else:
            return "unsupported_exact_type"
        if entries > MAXIMUM_PROFILE_NODES:
            return "total_entry_budget_exceeded"
    return None


def _source_preflight_v2(
    authorities: object,
    run_id: object,
    key_sources: object,
) -> tuple[
    TrustedWireBatchRejectionV2 | None,
    tuple[_SourceAuthorityV2, ...],
    set[str],
]:
    if type(authorities) is not tuple:
        raise TypeError("V2 trusted-wire authorities must use an exact tuple")
    count = len(authorities)
    if not 1 <= count <= MAXIMUM_BATCH_V2_AUTHORITIES:
        return (
            _reject_v2("authority_count_outside_1_through_1024", count),
            (),
            set(),
        )
    if type(run_id) is not bytes:
        raise TypeError("V2 trusted-wire run ID must use exact bytes")
    if len(run_id) != RUN_ID_BYTES:
        return _reject_v2("run_id_must_be_exactly_32_bytes", count), (), set()
    if type(key_sources) is not TrustedWireKeySourcesV2:
        raise TypeError("V2 trusted-wire key sources must use the exact V2 type")
    key_sources.__post_init__()
    if any(
        type(authority) is not PublicTransformEvidenceBundleV2
        for authority in authorities
    ):
        raise TypeError("V2 trusted-wire batch contains a non-exact authority")

    compiled: list[_SourceAuthorityV2] = []
    source_uuids: set[str] = set()
    for authority in authorities:
        try:
            reason = _source_string_cap_preflight_v2(authority)
            if reason is not None:
                return _reject_v2(
                    "source_schema_and_string_preflight:" + reason,
                    count,
                ), (), set()
            if (
                len(authority.base_bundle.observations)
                > _batch_v1.MAXIMUM_OBSERVATIONS_PER_AUTHORITY
                or len(authority.base_bundle.entity_candidates)
                > _batch_v1.MAXIMUM_ENTITIES_PER_AUTHORITY
                or len(authority.transform_contracts)
                > _batch_v1.MAXIMUM_CONTRACTS_PER_AUTHORITY
                or len(authority.observation_metadata)
                > _batch_v1.MAXIMUM_METADATA_ROWS_PER_AUTHORITY
            ):
                return _reject_v2(
                    "hash_free_authority_shape_budget_exceeded",
                    count,
                ), (), set()
            reason = _batch_v1._hash_free_authority_preflight(authority)
            if reason is not None:
                return _reject_v2(
                    "hash_free_profile_preflight:" + reason,
                    count,
                ), (), set()
            logical = encode_typed_transform_authority_profile_v1(authority)
            audit = audit_namespace_paths_v1(logical)
            source_uuids.update(
                occurrence.public_uuid for occurrence in audit.occurrences
            )
            if len(source_uuids) > _MAXIMUM_BATCH_V2_UUID_SIDECAR_ENTRIES:
                return _reject_v2(
                    "source_uuid_sidecar_budget_exceeded",
                    count,
                ), (), set()
            transform = run_exact_transform_semantics(authority)
            if (
                type(transform) is not ExactTransformCompilation
                or transform.disposition
                is not TransformCompilationDisposition.COMPLETE
            ):
                reason = getattr(transform, "reason", "transform_not_complete")
                return _reject_v2(
                    "source_transform_not_complete:" + str(reason),
                    count,
                ), (), set()
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
            return (
                _reject_v2("source_authority_validation_failed", count),
                (),
                set(),
            )
        compiled.append(_SourceAuthorityV2(logical, audit))
    return None, tuple(compiled), source_uuids


def _build_trusted_wire_batch_core_v2(
    *,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV2,
    per_case_projection_compiler: Callable[
        [
            int,
            Mapping[tuple[str, str], str],
            PublicTransformEvidenceBundleV2,
        ],
        object,
    ]
    | None,
) -> tuple[TrustedWireBatchV2 | TrustedWireBatchRejectionV2, tuple[object, ...]]:
    """Build V2 once; an optional private projector observes live renamings."""

    rejection, sources, source_uuids = _source_preflight_v2(
        authorities,
        run_id,
        key_sources,
    )
    if rejection is not None:
        return rejection, ()
    try:
        shuffle_key, id_key, padding_key = _batch_v1._derive_keys(
            run_id,
            key_sources,
        )
        order = _batch_v1._shuffle_indices(len(sources), shuffle_key, run_id)
    except (AttributeError, OverflowError, TypeError, ValueError):
        return _reject_v2("key_schedule_or_shuffle_failed", len(sources)), ()

    counters = {namespace: 0 for namespace in _batch_v1._NAMESPACES}
    allocated: set[str] = set()
    collision_retries = [0]
    rows: list[TrustedWireEnvelopeV2] = []
    projections: list[object] = []
    public_uuids: set[str] = set()
    for source_index in order:
        source = sources[source_index]
        renamings: dict[tuple[str, str], str] = {}
        try:
            authority_mapping = _batch_v1._rename_authority_ids(
                source.logical_profile,
                source.namespace_audit,
                id_key,
                run_id,
                counters,
                allocated,
                renamings,
                collision_retries,
            )
            _batch_v1._canonicalize_public_authority(authority_mapping)
            typed_authority = _batch_v1._compile_native_public_provenance(
                authority_mapping
            )
            if (
                encode_typed_transform_authority_profile_v1(typed_authority)
                != authority_mapping
            ):
                raise ValueError("V2 native authority logical roundtrip drift")

            projection: object | None = None
            if per_case_projection_compiler is not None:
                state_before = (
                    tuple(sorted(counters.items())),
                    len(allocated),
                    tuple(sorted(renamings.items())),
                    collision_retries[0],
                )
                profile_before = encode_typed_transform_authority_profile_v1(
                    typed_authority
                )
                projection = per_case_projection_compiler(
                    source_index,
                    MappingProxyType(renamings),
                    typed_authority,
                )
                state_after = (
                    tuple(sorted(counters.items())),
                    len(allocated),
                    tuple(sorted(renamings.items())),
                    collision_retries[0],
                )
                if (
                    projection is None
                    or state_after != state_before
                    or encode_typed_transform_authority_profile_v1(
                        typed_authority
                    )
                    != profile_before
                ):
                    raise ValueError("V2 private projection mutated allocation")

            compact = encode_typed_transform_authority_profile_v2(typed_authority)
            payload_value = {
                "authority": compact,
                "field_manifest_id": FIELD_MANIFEST_ID,
                "jcs_profile_id": JCS_PROFILE_ID,
                "public_provenance_version": (
                    _batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
                ),
                "schema_version": TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION,
                "typed_authority_codec_policy_id": (
                    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
                ),
                "typed_authority_codec_version": (
                    COMPACT_TYPED_AUTHORITY_CODEC_VERSION
                ),
                "typed_authority_schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
            }
            payload = _encode_payload_jcs_v2(payload_value)
            envelope = _frame_secret_payload_v2(payload, padding_key, run_id)
            decoded = decode_and_audit_trusted_envelope_v2(envelope)
            row_public_uuids = {
                occurrence.public_uuid
                for occurrence in decoded.namespace_audit.occurrences
            }
            if not source_uuids.isdisjoint(row_public_uuids):
                raise _SourcePublicUUIDCollisionV2(
                    "source and public UUID namespaces intersect"
                )
            public_uuids.update(row_public_uuids)
            if len(public_uuids) > _MAXIMUM_BATCH_V2_UUID_SIDECAR_ENTRIES:
                raise ValueError("V2 public UUID sidecar budget exceeded")
            expected_transform = run_exact_transform_semantics(typed_authority)
            if (
                type(expected_transform) is not ExactTransformCompilation
                or expected_transform.disposition
                is not TransformCompilationDisposition.COMPLETE
                or decoded.authority != typed_authority
                or decoded.authority_content_id != typed_authority.content_id
                or decoded.transform_result_id != expected_transform.result_id
                or decoded.namespace_audit
                != audit_namespace_paths_v1(authority_mapping)
            ):
                raise ValueError("V2 envelope typed self-replay drift")
            row = TrustedWireEnvelopeV2._issue(
                _ENVELOPE_ISSUE_TOKEN_V2,
                decoded,
            )
        except _SourcePublicUUIDCollisionV2:
            return _reject_v2(
                "source_public_uuid_collision",
                len(sources),
            ), ()
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return _reject_v2(
                "renamed_authority_or_compact_envelope_replay_failed",
                len(sources),
            ), ()
        rows.append(row)
        if per_case_projection_compiler is not None:
            projections.append(projection)

    if not source_uuids.isdisjoint(public_uuids):
        return _reject_v2("source_public_uuid_collision", len(sources)), ()
    batch = TrustedWireBatchV2._issue(
        _BATCH_ISSUE_TOKEN_V2,
        run_id_commitment=_run_commitment_v2(run_id),
        envelopes=tuple(rows),
        uuid_collision_retry_count=collision_retries[0],
    )
    return batch, tuple(projections)


def build_trusted_wire_batch_v2(
    *,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV2,
) -> TrustedWireBatchV2 | TrustedWireBatchRejectionV2:
    """Build one atomic compact-authority V2 secret-padded batch."""

    result, projections = _build_trusted_wire_batch_core_v2(
        authorities=authorities,
        run_id=run_id,
        key_sources=key_sources,
        per_case_projection_compiler=None,
    )
    if projections:
        raise RuntimeError("public V2 batch builder retained private projections")
    return result


def _assert_public_field_manifests_v2() -> None:
    for cls, manifest in (
        (TrustedWireKeySourcesV2, _KEY_SOURCE_FIELDS_V2),
        (TrustedWireBatchPolicyV2, _POLICY_FIELDS_V2),
        (TrustedWireBatchRejectionV2, _REJECTION_FIELDS_V2),
        (DecodedTrustedEnvelopeV2, _DECODED_FIELDS_V2),
        (TrustedWireEnvelopeV2, _ENVELOPE_FIELDS_V2),
        (TrustedWireBatchV2, _BATCH_FIELDS_V2),
    ):
        if tuple(item.name for item in fields(cls)) != manifest:
            raise RuntimeError(f"V2 trusted-wire {cls.__name__} field manifest drift")


_assert_public_field_manifests_v2()


__all__ = (
    "DEFAULT_TRUSTED_WIRE_BATCH_V2_POLICY",
    "DecodedTrustedEnvelopeV2",
    "IKM_BYTES",
    "MAXIMUM_BATCH_V2_AUTHORITIES",
    "MAXIMUM_BATCH_V2_COLLISION_RETRY_COUNT",
    "MAXIMUM_SHUFFLE_REJECTION_DRAWS",
    "MAXIMUM_UUID_COLLISION_RETRIES",
    "RUN_ID_BYTES",
    "TRUSTED_WIRE_BATCH_V2_PAYLOAD_FIELDS",
    "TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION",
    "TRUSTED_WIRE_BATCH_V2_POLICY_ID",
    "TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION",
    "TRUSTED_WIRE_ENVELOPE_V2_MAGIC",
    "TRUSTED_WIRE_ENVELOPE_V2_VERSION",
    "TrustedWireBatchDispositionV2",
    "TrustedWireBatchPolicyV2",
    "TrustedWireBatchRejectionV2",
    "TrustedWireBatchV2",
    "TrustedWireEnvelopeV2",
    "TrustedWireKeySourcesV2",
    "build_trusted_wire_batch_v2",
    "decode_and_audit_trusted_envelope_v2",
)

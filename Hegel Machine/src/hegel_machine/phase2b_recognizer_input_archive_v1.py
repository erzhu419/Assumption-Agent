"""Trusted recognizer-input archive mechanics for Phase-2B.

The private issuer rebuilds the Stage-B batch once while compiling a public
adapter registry from each case's live HMAC renamings.  Public archive rows do
not retain source indices, source roots, old identifiers, permutations, maps,
or keys.  The raw decoder proves only bounded structural/typed replay; it does
not claim batch membership, source-registry projection, secret custody, origin,
formal covert audit, sealed eligibility, recognizer execution, or C1 exit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
import struct
from typing import Final, Mapping
from uuid import UUID

from .bootstrap import initial_theory
from .hashing import stable_hash
from .phase2b_adapter import (
    LawWireBinding,
    ObservableChannelBinding,
    Phase2BAdapterRegistry,
)
from .phase2b_exact_transform_semantics_v1 import (
    PublicTransformEvidenceBundleV2,
)
from .phase2b_freeze_v1 import (
    CanonicalFamilyId,
    frozen_phase2b_exact_freeze,
)
from .phase2b_trusted_wire_batch_v1 import (
    MAXIMUM_BATCH_AUTHORITIES,
    TRUSTED_WIRE_BATCH_POLICY_ID,
    TrustedWireBatchPreflightV1,
    TrustedWireBatchV1,
    TrustedWireKeySourcesV1,
)
from . import phase2b_trusted_wire_batch_v1 as _trusted_wire_batch
from .phase2b_trusted_wire_typed_authority_v1 import (
    TYPED_AUTHORITY_SCHEMA_ID,
    encode_typed_transform_authority_profile_v1,
)
from .phase2b_trusted_wire_typed_replay_v1 import (
    TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID,
    TypedTrustedEnvelopeReplayV1,
    TypedTrustedWireBatchReplayV1,
    TypedTrustedWireReplayDisposition,
    decode_and_replay_typed_trusted_envelope_v1,
)
from .phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    MAXIMUM_ARRAY_ENTRIES,
    MAXIMUM_ASCII_STRING_BYTES,
    MAXIMUM_SAFE_INTEGER,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)
from .schema import LawKind


PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-recognizer-registry/1"
)
TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION: Final = (
    "hegel-machine-phase2b-trusted-recognizer-input-archive/1"
)
ARCHIVE_MAGIC: Final = b"HGRIAV1\x00"
ARCHIVE_WIRE_VERSION: Final = 1
_ARCHIVE_HEADER: Final = struct.Struct(">8sHHII32s")
ARCHIVE_HEADER_BYTES: Final = _ARCHIVE_HEADER.size
MAXIMUM_ARCHIVE_METADATA_BYTES: Final = 16_384
MAXIMUM_PUBLIC_REGISTRY_BYTES: Final = 65_536
MAXIMUM_PUBLIC_REGISTRY_LAWS: Final = len(LawKind)
MAXIMUM_PUBLIC_REGISTRY_CHANNELS: Final = MAXIMUM_ARRAY_ENTRIES
MAXIMUM_ROLES_PER_LAW: Final = 64
MAXIMUM_OBSERVABLES_PER_LAW: Final = 64
MAXIMUM_REGISTRY_TEXT_BYTES: Final = 65_536
MAXIMUM_GLOBAL_SOURCE_UUIDS: Final = 262_144
MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES: Final = (
    ARCHIVE_HEADER_BYTES
    + MAXIMUM_ARCHIVE_METADATA_BYTES
    + MAXIMUM_BATCH_AUTHORITIES
    * (4 + MAXIMUM_PUBLIC_REGISTRY_BYTES + ENVELOPE_BYTES)
)

_ARCHIVE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/TRUSTED_RECOGNIZER_INPUT_ARCHIVE/V1\x00"
)
_ROW_DOMAIN: Final = b"HEGEL/PHASE2B/RECOGNIZER_INPUT_ROW/V1\x00"
_REGISTRY_DOMAIN: Final = b"HEGEL/PHASE2B/PUBLIC_RECOGNIZER_REGISTRY/V1\x00"
_REGISTRY_ISSUE_TOKEN: Final = object()
_ROW_ISSUE_TOKEN: Final = object()
_ROW_CONTEXT_TOKEN: Final = object()
_DECODE_ISSUE_TOKEN: Final = object()
_PARSED_CONTEXT_TOKEN: Final = object()
_ARCHIVE_ENCODE_TOKEN: Final = object()

FROZEN_BRIDGE_FAMILY_UUID_ALIASES: Final = (
    (LawKind.SYMMETRY, "58351910-f1ea-4613-b5b2-47d9cc2f1652"),
    (LawKind.MONOTONICITY, "16ba12ce-f178-4226-ac97-2120adb62073"),
    (LawKind.CONSERVATION, "773faef6-c762-4ca6-b389-f2a593cb1f99"),
    (LawKind.COMPLEMENTARITY, "431cb872-0237-4751-a3f8-e5fc2a2a3b38"),
    (LawKind.NEGATIVE_FEEDBACK, "1d9fd5a5-ac24-4dd0-9b70-e257391585e5"),
    (LawKind.LOCALITY, "c4a5cad4-444f-4e54-a341-c21ffe29d2c5"),
)
_BRIDGE_FAMILY_BY_KIND: Final = dict(FROZEN_BRIDGE_FAMILY_UUID_ALIASES)
_CANONICAL_FAMILY_BY_KIND: Final = dict(
    frozen_phase2b_exact_freeze().family_mapping
)
_FROZEN_THEORY: Final = initial_theory()
_FROZEN_THEORY_ROLE_COUNT: Final = 15
_FROZEN_THEORY_OBSERVABLE_COUNT: Final = 35
_FROZEN_CANDIDATE_COUNT: Final = 50_000

_LAW_FIELDS: Final = (
    "bridge_family_id",
    "canonical_family_id",
    "law_id",
    "law_kind",
    "required_observable_ids",
    "role_ids",
)
_CHANNEL_FIELDS: Final = ("observable_id", "quantity_id")
_REGISTRY_FIELDS: Final = (
    "family_alias_policy_id",
    "law_bindings",
    "maximum_candidate_count",
    "observable_channels",
    "schema_version",
    "theory_version_id",
)
_METADATA_FIELDS: Final = (
    "archive_policy_id",
    "archive_version",
    "batch_id",
    "batch_policy_id",
    "execution_commitment",
    "row_count",
    "secret_replay_receipt_id",
    "source_registry_id",
    "typed_authority_schema_id",
    "typed_replay_receipt_id",
)

_ROW_FIELDS: Final = (
    "authority_content_id",
    "envelope",
    "envelope_id",
    "payload_sha256",
    "public_registry",
    "public_registry_id",
    "row_id",
    "transform_result_id",
)
_SOURCE_CASE_FIELD_MANIFEST: Final = ("authority", "adapter_registry")
_REJECTION_FIELD_MANIFEST: Final = (
    "disposition",
    "reason",
    "case_count",
    "batch_id",
    "archive",
    "rows",
    "public_registry_ids",
)
_DECODED_ARCHIVE_FIELD_MANIFEST: Final = (
    "disposition",
    "archive",
    "archive_id",
    "schema_version",
    "policy_id",
    "batch_id",
    "batch_policy_id",
    "run_id_commitment",
    "typed_replay_receipt_id",
    "secret_replay_receipt_id",
    "source_registry_id",
    "rows",
    "row_ids",
    "envelope_ids",
    "public_registry_ids",
    "authority_content_ids",
    "transform_result_ids",
    "claim_level",
    "structural_archive_verified",
    "row_bijection_verified",
    "registry_schema_verified",
    "direct_payload_transform_replay_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "secret_custodian_replay_verified",
    "origin_authenticated",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "recognizer_executed",
    "prediction_archive_evaluated",
    "c1_exit_evidence",
)

PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID: Final = stable_hash(
    {
        "aliases": tuple(
            (
                law_kind.value,
                bridge_id,
                _CANONICAL_FAMILY_BY_KIND[law_kind].value,
            )
            for law_kind, bridge_id in FROZEN_BRIDGE_FAMILY_UUID_ALIASES
        ),
        "contract": (
            "bridge_family_id_is_fixed_public_uuidv4_not_hmac_renamed",
            "canonical_family_id_is_exact_phase2b_freeze_mapping",
            "source_registry_family_uuid_is_not_emitted",
        ),
        "version": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION,
    },
    prefix="phase2b_public_recognizer_family_alias_policy_",
)

PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID: Final = stable_hash(
    {
        "accepted_jcs": True,
        "caps": {
            "channels": MAXIMUM_PUBLIC_REGISTRY_CHANNELS,
            "laws": MAXIMUM_PUBLIC_REGISTRY_LAWS,
            "observables_per_law": MAXIMUM_OBSERVABLES_PER_LAW,
            "registry_bytes": MAXIMUM_PUBLIC_REGISTRY_BYTES,
            "roles_per_law": MAXIMUM_ROLES_PER_LAW,
            "text_bytes": MAXIMUM_REGISTRY_TEXT_BYTES,
        },
        "channel_fields": _CHANNEL_FIELDS,
        "family_alias_policy_id": PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID,
        "law_fields": _LAW_FIELDS,
        "registry_fields": _REGISTRY_FIELDS,
        "wire_grammar": (
            "exact_closed_mapping_and_arrays",
            "law_kind_enum_and_canonical_family_enum_strings",
            "bridge_role_quantity_ids_are_canonical_uuidv4",
            "one_law_per_six_frozen_law_kinds",
            "semantic_strings_are_bounded_ascii",
        ),
        "frozen_theory": {
            "candidate_count": _FROZEN_CANDIDATE_COUNT,
            "laws": tuple(
                (
                    law.law_id,
                    law.kind.value,
                    law.roles,
                    law.required_observables,
                )
                for law in _FROZEN_THEORY.relation_laws
            ),
            "observable_count": _FROZEN_THEORY_OBSERVABLE_COUNT,
            "role_count": _FROZEN_THEORY_ROLE_COUNT,
            "version_id": _FROZEN_THEORY.version_id,
        },
    },
    prefix="phase2b_public_recognizer_registry_schema_",
)

RECOGNIZER_INPUT_ARCHIVE_POLICY_ID: Final = stable_hash(
    {
        "archive_contract": (
            "fixed_header_then_closed_jcs_metadata",
            "each_row_is_u32_registry_length_then_registry_jcs_then_exact_envelope",
            "row_adjacency_is_registry_envelope_bijection",
            "row_registry_scope_exactly_matches_typed_authority_task_scope",
            "all_output_authority_uuid_namespaces_are_cross_row_disjoint",
            "issuer_uses_one_stage_b_live_allocation_with_read_only_projection",
            "no_source_index_root_old_id_map_permutation_or_key_in_public_output",
            "one_global_source_registry_commitment_after_hash_free_frozen_theory_validation",
            "fixed_family_aliases_disjoint_from_all_source_and_output_uuids",
            "exact_top_level_type_misuse_raises_other_drift_abstains_atomically",
            "custodian_gate_is_issuance_only_not_a_durable_public_secret_claim",
            "validated_context_reuse_is_private_token_gated",
        ),
        "archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION,
        "binary": {
            "header_format": _ARCHIVE_HEADER.format,
            "header_bytes": ARCHIVE_HEADER_BYTES,
            "magic_hex": ARCHIVE_MAGIC.hex(),
            "wire_version": ARCHIVE_WIRE_VERSION,
        },
        "caps": {
            "archive_bytes": MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES,
            "authorities": MAXIMUM_BATCH_AUTHORITIES,
            "global_public_uuids": MAXIMUM_GLOBAL_SOURCE_UUIDS,
            "global_source_uuids": MAXIMUM_GLOBAL_SOURCE_UUIDS,
            "metadata_bytes": MAXIMUM_ARCHIVE_METADATA_BYTES,
            "registry_bytes": MAXIMUM_PUBLIC_REGISTRY_BYTES,
        },
        "claims": {
            "exact_bool": True,
            "raw_true": (
                "structural_archive_verified",
                "row_bijection_verified",
                "registry_schema_verified",
                "direct_payload_transform_replay_verified",
            ),
            "raw_false": (
                "batch_policy_membership_verified",
                "source_registry_projection_verified",
                "secret_custodian_replay_verified",
                "origin_authenticated",
                "formal_covert_audit",
                "sealed_holdout_eligible",
                "recognizer_executed",
                "prediction_archive_evaluated",
                "c1_exit_evidence",
            ),
            "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
        },
        "metadata_fields": _METADATA_FIELDS,
        "receipt_fields": {
            "decoded": _DECODED_ARCHIVE_FIELD_MANIFEST,
            "rejection": _REJECTION_FIELD_MANIFEST,
            "source_case": _SOURCE_CASE_FIELD_MANIFEST,
        },
        "row_fields": _ROW_FIELDS,
        "public_registry_schema_id": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID,
        "roots": (
            "row_id=sha256(row_domain||accepted_jcs_public_row_roots)",
            "archive_id=sha256(archive_domain||all_archive_bytes)",
            "public_registry_id=sha256(registry_domain||accepted_jcs_registry)",
            "source_registry_id=stable_hash(exact_validated_frozen_source_registry)",
            "all_caps_and_exact_semantics_validate_before_public_hashes",
        ),
        "stage_b_policy_id": TRUSTED_WIRE_BATCH_POLICY_ID,
        "stage_c_replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID,
        "typed_authority_schema_id": TYPED_AUTHORITY_SCHEMA_ID,
    },
    prefix="phase2b_recognizer_input_archive_policy_",
)

for _manifest_name, _manifest in (
    ("law", _LAW_FIELDS),
    ("channel", _CHANNEL_FIELDS),
    ("registry", _REGISTRY_FIELDS),
    ("metadata", _METADATA_FIELDS),
    ("row", _ROW_FIELDS),
):
    if _manifest != tuple(sorted(_manifest)):
        raise RuntimeError(f"{_manifest_name} field manifest is not canonical")

_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _uuid4(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 36
        or _UUID4.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be an exact canonical lowercase UUIDv4")
    if UUID(value).version != 4:
        raise ValueError(f"{name} must be a canonical lowercase UUIDv4")
    return value


def _ascii(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAXIMUM_ASCII_STRING_BYTES
        or not value.isascii()
    ):
        raise ValueError(f"{name} must use exact bounded nonempty ASCII")
    return value


def _digest(value: object, prefix: str, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != len(prefix) + 64
        or not value.startswith(prefix)
    ):
        raise ValueError(f"{name} must be an exact prefixed SHA-256")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError(f"{name} must end in lowercase SHA-256")
    return value


def _sha256_text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _exact_bool_values(*values: object) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError("recognizer-input claim fields require exact bool values")


def _closed_mapping(
    value: object,
    fields_manifest: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must use an exact mapping")
    if len(value) != len(fields_manifest):
        raise ValueError(f"{name} field count drift")
    if tuple(sorted(value)) != fields_manifest:
        raise ValueError(f"{name} closed schema drift")
    return value


@dataclass(frozen=True, slots=True)
class TrustedRecognizerSourceCaseV1:
    authority: PublicTransformEvidenceBundleV2
    adapter_registry: Phase2BAdapterRegistry

    def __post_init__(self) -> None:
        if type(self) is not TrustedRecognizerSourceCaseV1:
            raise TypeError("source case must use the exact trusted type")
        if type(self.authority) is not PublicTransformEvidenceBundleV2:
            raise TypeError("source case needs an exact V2 authority")
        if type(self.adapter_registry) is not Phase2BAdapterRegistry:
            raise TypeError("source case needs an exact adapter registry")
        encode_typed_transform_authority_profile_v1(self.authority)
        _validate_source_registry(self.adapter_registry)


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerLawBindingV1:
    law_id: str
    law_kind: LawKind
    canonical_family_id: CanonicalFamilyId
    bridge_family_id: str
    role_ids: tuple[tuple[str, str], ...]
    required_observable_ids: tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("public recognizer law bindings are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        law_id: str,
        law_kind: LawKind,
        role_ids: tuple[tuple[str, str], ...],
        required_observable_ids: tuple[str, ...],
    ) -> "PublicRecognizerLawBindingV1":
        if token is not _REGISTRY_ISSUE_TOKEN:
            raise TypeError("public recognizer law issuer token mismatch")
        value = object.__new__(cls)
        object.__setattr__(value, "law_id", law_id)
        object.__setattr__(value, "law_kind", law_kind)
        object.__setattr__(
            value,
            "canonical_family_id",
            _CANONICAL_FAMILY_BY_KIND[law_kind],
        )
        object.__setattr__(
            value,
            "bridge_family_id",
            _BRIDGE_FAMILY_BY_KIND[law_kind],
        )
        object.__setattr__(value, "role_ids", role_ids)
        object.__setattr__(
            value,
            "required_observable_ids",
            required_observable_ids,
        )
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerLawBindingV1:
            raise TypeError("public recognizer law must use the exact type")
        _ascii(self.law_id, "public recognizer law ID")
        if type(self.law_kind) is not LawKind:
            raise TypeError("public recognizer law kind must use the exact enum")
        if (
            type(self.canonical_family_id) is not CanonicalFamilyId
            or self.canonical_family_id
            is not _CANONICAL_FAMILY_BY_KIND[self.law_kind]
            or self.bridge_family_id != _BRIDGE_FAMILY_BY_KIND[self.law_kind]
        ):
            raise ValueError("public recognizer family mapping drift")
        _uuid4(self.bridge_family_id, "public bridge family ID")
        if (
            type(self.role_ids) is not tuple
            or not self.role_ids
            or len(self.role_ids) > MAXIMUM_ROLES_PER_LAW
            or type(self.required_observable_ids) is not tuple
            or not self.required_observable_ids
            or len(self.required_observable_ids) > MAXIMUM_OBSERVABLES_PER_LAW
        ):
            raise TypeError("public recognizer law arrays are invalid")
        if self.role_ids != tuple(sorted(self.role_ids)):
            raise ValueError("public recognizer roles are not canonical")
        semantic_roles: list[str] = []
        wire_roles: list[str] = []
        for item in self.role_ids:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("public recognizer role binding must be an exact pair")
            semantic_roles.append(_ascii(item[0], "public semantic role"))
            wire_roles.append(_uuid4(item[1], "public role ID"))
        if len(set(semantic_roles)) != len(semantic_roles) or len(
            set(wire_roles)
        ) != len(wire_roles):
            raise ValueError("public recognizer law repeats a role")
        if self.required_observable_ids != tuple(
            sorted(self.required_observable_ids)
        ):
            raise ValueError("public recognizer observables are not canonical")
        for item in self.required_observable_ids:
            _ascii(item, "public required observable")


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerObservableChannelV1:
    quantity_id: str
    observable_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("public recognizer channels are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        quantity_id: str,
        observable_id: str,
    ) -> "PublicRecognizerObservableChannelV1":
        if token is not _REGISTRY_ISSUE_TOKEN:
            raise TypeError("public recognizer channel issuer token mismatch")
        value = object.__new__(cls)
        object.__setattr__(value, "quantity_id", quantity_id)
        object.__setattr__(value, "observable_id", observable_id)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerObservableChannelV1:
            raise TypeError("public recognizer channel must use the exact type")
        _uuid4(self.quantity_id, "public quantity ID")
        _ascii(self.observable_id, "public observable ID")


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerRegistryV1:
    schema_version: str
    theory_version_id: str
    law_bindings: tuple[PublicRecognizerLawBindingV1, ...]
    observable_channels: tuple[PublicRecognizerObservableChannelV1, ...]
    maximum_candidate_count: int
    family_alias_policy_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("public recognizer registries are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        theory_version_id: str,
        law_bindings: tuple[PublicRecognizerLawBindingV1, ...],
        observable_channels: tuple[PublicRecognizerObservableChannelV1, ...],
        maximum_candidate_count: int,
    ) -> "PublicRecognizerRegistryV1":
        if token is not _REGISTRY_ISSUE_TOKEN:
            raise TypeError("public recognizer registry issuer token mismatch")
        value = object.__new__(cls)
        object.__setattr__(
            value,
            "schema_version",
            PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION,
        )
        object.__setattr__(value, "theory_version_id", theory_version_id)
        object.__setattr__(value, "law_bindings", law_bindings)
        object.__setattr__(value, "observable_channels", observable_channels)
        object.__setattr__(
            value,
            "maximum_candidate_count",
            maximum_candidate_count,
        )
        object.__setattr__(
            value,
            "family_alias_policy_id",
            PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID,
        )
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerRegistryV1:
            raise TypeError("public recognizer registry must use the exact type")
        _ascii(self.schema_version, "public recognizer registry schema")
        _digest(
            self.family_alias_policy_id,
            "phase2b_public_recognizer_family_alias_policy_",
            "public recognizer family alias policy ID",
        )
        if self.schema_version != PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION:
            raise ValueError("public recognizer registry schema drift")
        if self.family_alias_policy_id != PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID:
            raise ValueError("public recognizer family alias policy drift")
        _digest(self.theory_version_id, "theory_", "theory version ID")
        if (
            type(self.law_bindings) is not tuple
            or len(self.law_bindings) != MAXIMUM_PUBLIC_REGISTRY_LAWS
            or any(
                type(item) is not PublicRecognizerLawBindingV1
                for item in self.law_bindings
            )
        ):
            raise TypeError("public recognizer registry needs exactly six laws")
        for item in self.law_bindings:
            item._validate()
        if tuple(item.law_kind for item in self.law_bindings) != tuple(LawKind):
            raise ValueError("public recognizer laws are not in frozen family order")
        if len({item.law_id for item in self.law_bindings}) != len(
            self.law_bindings
        ):
            raise ValueError("public recognizer registry repeats a law")
        if (
            type(self.observable_channels) is not tuple
            or not self.observable_channels
            or len(self.observable_channels) > MAXIMUM_PUBLIC_REGISTRY_CHANNELS
            or any(
                type(item) is not PublicRecognizerObservableChannelV1
                for item in self.observable_channels
            )
        ):
            raise TypeError("public recognizer registry channels are invalid")
        for item in self.observable_channels:
            item._validate()
        if self.observable_channels != tuple(
            sorted(
                self.observable_channels,
                key=lambda item: item.observable_id,
            )
        ):
            raise ValueError("public recognizer channels are not canonical")
        if len({item.quantity_id for item in self.observable_channels}) != len(
            self.observable_channels
        ) or len({item.observable_id for item in self.observable_channels}) != len(
            self.observable_channels
        ):
            raise ValueError("public recognizer registry repeats a channel")
        if (
            type(self.maximum_candidate_count) is not int
            or not 1 <= self.maximum_candidate_count <= MAXIMUM_SAFE_INTEGER
        ):
            raise ValueError("public recognizer candidate cap is invalid")
        registered = {item.observable_id for item in self.observable_channels}
        if any(
            not set(item.required_observable_ids).issubset(registered)
            for item in self.law_bindings
        ):
            raise ValueError("public recognizer law cites an unknown observable")
        adapter = _public_registry_adapter_unchecked(self)
        role_ids = {
            (law.law_id, semantic_role): wire_id
            for law in adapter.law_bindings
            for semantic_role, wire_id in law.role_ids
        }
        quantity_ids = {
            item.observable_id: item.quantity_id
            for item in adapter.observable_channels
        }
        rebuilt = Phase2BAdapterRegistry.from_theory(
            _FROZEN_THEORY,
            family_ids={
                law_kind: _BRIDGE_FAMILY_BY_KIND[law_kind]
                for law_kind in LawKind
            },
            role_ids=role_ids,
            quantity_ids=quantity_ids,
            maximum_candidate_count=_FROZEN_CANDIDATE_COUNT,
        )
        if rebuilt != adapter:
            raise ValueError("public registry does not exactly rebuild frozen theory")
        encoded = _encode_public_registry(self)
        if len(encoded) > MAXIMUM_PUBLIC_REGISTRY_BYTES:
            raise ValueError("public recognizer registry exceeds its byte cap")

    @property
    def registry_id(self) -> str:
        self._validate()
        return "phase2b_public_recognizer_registry_" + hashlib.sha256(
            _REGISTRY_DOMAIN + _encode_public_registry(self)
        ).hexdigest()

    def to_adapter_registry(self) -> Phase2BAdapterRegistry:
        self._validate()
        return _public_registry_adapter_unchecked(self)


def _public_registry_adapter_unchecked(
    value: PublicRecognizerRegistryV1,
) -> Phase2BAdapterRegistry:
    return Phase2BAdapterRegistry(
        theory_version_id=value.theory_version_id,
        law_bindings=tuple(
            sorted(
                (
                    LawWireBinding(
                        law_id=item.law_id,
                        law_kind=item.law_kind,
                        family_id=item.bridge_family_id,
                        role_ids=item.role_ids,
                        required_observable_ids=item.required_observable_ids,
                    )
                    for item in value.law_bindings
                ),
                key=lambda item: item.law_id,
            )
        ),
        observable_channels=tuple(
            ObservableChannelBinding(
                quantity_id=item.quantity_id,
                observable_id=item.observable_id,
            )
            for item in value.observable_channels
        ),
        maximum_candidate_count=value.maximum_candidate_count,
    )


def _validate_source_registry(registry: Phase2BAdapterRegistry) -> None:
    if type(registry) is not Phase2BAdapterRegistry:
        raise TypeError("source registry must use the exact adapter type")
    if type(registry.theory_version_id) is not str:
        raise TypeError("source registry theory version must be exact text")
    _digest(registry.theory_version_id, "theory_", "source theory version ID")
    if registry.theory_version_id != _FROZEN_THEORY.version_id:
        raise ValueError("source registry does not bind the frozen theory")
    if (
        type(registry.law_bindings) is not tuple
        or len(registry.law_bindings) != MAXIMUM_PUBLIC_REGISTRY_LAWS
        or any(type(item) is not LawWireBinding for item in registry.law_bindings)
    ):
        raise TypeError("source registry must contain exactly six exact laws")
    for item in registry.law_bindings:
        if type(item.law_kind) is not LawKind:
            raise TypeError("source registry law kind must use the exact enum")
        if (
            type(item.role_ids) is not tuple
            or not item.role_ids
            or len(item.role_ids) > MAXIMUM_ROLES_PER_LAW
            or type(item.required_observable_ids) is not tuple
            or not item.required_observable_ids
            or len(item.required_observable_ids) > MAXIMUM_OBSERVABLES_PER_LAW
        ):
            raise ValueError("source registry law arrays exceed frozen caps")
        _ascii(item.law_id, "source registry law ID")
        _ascii(item.family_id, "source registry family ID")
        for role in item.role_ids:
            if type(role) is not tuple or len(role) != 2:
                raise TypeError("source registry role must use an exact pair")
            _ascii(role[0], "source registry semantic role")
            _ascii(role[1], "source registry role ID")
        for observable_id in item.required_observable_ids:
            _ascii(observable_id, "source registry required observable")
        item.__post_init__()
    if {item.law_kind for item in registry.law_bindings} != set(LawKind):
        raise ValueError("source registry must contain one law per family")
    if len({item.law_kind for item in registry.law_bindings}) != len(LawKind):
        raise ValueError("source registry repeats a law family")
    if (
        type(registry.observable_channels) is not tuple
        or not registry.observable_channels
        or len(registry.observable_channels) > MAXIMUM_PUBLIC_REGISTRY_CHANNELS
        or any(
            type(item) is not ObservableChannelBinding
            for item in registry.observable_channels
        )
    ):
        raise TypeError("source registry observable channels are invalid")
    for item in registry.observable_channels:
        _ascii(item.quantity_id, "source registry quantity ID")
        _ascii(item.observable_id, "source registry observable ID")
        item.__post_init__()
    if (
        type(registry.maximum_candidate_count) is not int
        or registry.maximum_candidate_count != _FROZEN_CANDIDATE_COUNT
    ):
        raise ValueError("source registry candidate cap is not frozen")
    text_items = [registry.theory_version_id]
    for law in registry.law_bindings:
        text_items.extend(
            (
                law.law_id,
                law.law_kind.value,
                law.family_id,
                *law.required_observable_ids,
            )
        )
        for role, wire_id in law.role_ids:
            text_items.extend((role, wire_id))
    for channel in registry.observable_channels:
        text_items.extend((channel.observable_id, channel.quantity_id))
    for item in text_items:
        _ascii(item, "source registry text")
    if sum(len(item.encode("ascii")) for item in text_items) > MAXIMUM_REGISTRY_TEXT_BYTES:
        raise ValueError("source registry text budget exceeded")
    family_ids = {
        item.law_kind: item.family_id for item in registry.law_bindings
    }
    role_ids = {
        (item.law_id, semantic_role): wire_id
        for item in registry.law_bindings
        for semantic_role, wire_id in item.role_ids
    }
    quantity_ids = {
        item.observable_id: item.quantity_id
        for item in registry.observable_channels
    }
    if (
        len(role_ids) != _FROZEN_THEORY_ROLE_COUNT
        or len(quantity_ids) != _FROZEN_THEORY_OBSERVABLE_COUNT
    ):
        raise ValueError("source registry frozen vocabulary cardinality drift")
    rebuilt = Phase2BAdapterRegistry.from_theory(
        _FROZEN_THEORY,
        family_ids=family_ids,
        role_ids=role_ids,
        quantity_ids=quantity_ids,
        maximum_candidate_count=_FROZEN_CANDIDATE_COUNT,
    )
    if rebuilt != registry:
        raise ValueError("source registry does not exactly rebuild frozen theory")


def _law_mapping(value: PublicRecognizerLawBindingV1) -> dict[str, object]:
    value._validate()
    return {
        "bridge_family_id": value.bridge_family_id,
        "canonical_family_id": value.canonical_family_id.value,
        "law_id": value.law_id,
        "law_kind": value.law_kind.value,
        "required_observable_ids": list(value.required_observable_ids),
        "role_ids": [list(item) for item in value.role_ids],
    }


def _registry_mapping(value: PublicRecognizerRegistryV1) -> dict[str, object]:
    return {
        "family_alias_policy_id": value.family_alias_policy_id,
        "law_bindings": [_law_mapping(item) for item in value.law_bindings],
        "maximum_candidate_count": value.maximum_candidate_count,
        "observable_channels": [
            {
                "observable_id": item.observable_id,
                "quantity_id": item.quantity_id,
            }
            for item in value.observable_channels
        ],
        "schema_version": value.schema_version,
        "theory_version_id": value.theory_version_id,
    }


def _encode_public_registry(value: PublicRecognizerRegistryV1) -> bytes:
    return encode_phase2b_jcs_profile_v1(_registry_mapping(value))


def _decode_public_registry(payload: bytes) -> PublicRecognizerRegistryV1:
    if type(payload) is not bytes or not payload:
        raise TypeError("public recognizer registry payload must use exact bytes")
    if len(payload) > MAXIMUM_PUBLIC_REGISTRY_BYTES:
        raise ValueError("public recognizer registry payload exceeds its cap")
    root = _closed_mapping(
        decode_phase2b_jcs_profile_v1(payload),
        _REGISTRY_FIELDS,
        "public recognizer registry",
    )
    if root["schema_version"] != PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION:
        raise ValueError("public recognizer registry wire schema drift")
    if root["family_alias_policy_id"] != PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID:
        raise ValueError("public recognizer family alias wire drift")
    law_rows = root["law_bindings"]
    channel_rows = root["observable_channels"]
    if type(law_rows) is not list or len(law_rows) != len(LawKind):
        raise TypeError("public recognizer law wire must contain six rows")
    if (
        type(channel_rows) is not list
        or not channel_rows
        or len(channel_rows) > MAXIMUM_PUBLIC_REGISTRY_CHANNELS
    ):
        raise TypeError("public recognizer channel wire is invalid")
    laws: list[PublicRecognizerLawBindingV1] = []
    for raw in law_rows:
        row = _closed_mapping(raw, _LAW_FIELDS, "public recognizer law")
        try:
            law_kind = LawKind(row["law_kind"])
            canonical_family = CanonicalFamilyId(row["canonical_family_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("public recognizer family discriminator drift") from exc
        if canonical_family is not _CANONICAL_FAMILY_BY_KIND[law_kind]:
            raise ValueError("public recognizer canonical family drift")
        raw_roles = row["role_ids"]
        raw_observables = row["required_observable_ids"]
        if type(raw_roles) is not list or type(raw_observables) is not list:
            raise TypeError("public recognizer law arrays must be exact arrays")
        roles: list[tuple[str, str]] = []
        for raw_role in raw_roles:
            if type(raw_role) is not list or len(raw_role) != 2:
                raise TypeError("public recognizer role wire must be a pair")
            if any(type(item) is not str for item in raw_role):
                raise TypeError("public recognizer role wire needs exact text")
            roles.append((raw_role[0], raw_role[1]))
        if any(type(item) is not str for item in raw_observables):
            raise TypeError("public recognizer observable wire needs exact text")
        law = PublicRecognizerLawBindingV1._issue(
            _REGISTRY_ISSUE_TOKEN,
            law_id=row["law_id"],
            law_kind=law_kind,
            role_ids=tuple(roles),
            required_observable_ids=tuple(raw_observables),
        )
        if law.bridge_family_id != row["bridge_family_id"]:
            raise ValueError("public bridge family alias wire drift")
        laws.append(law)
    channels: list[PublicRecognizerObservableChannelV1] = []
    for raw in channel_rows:
        row = _closed_mapping(raw, _CHANNEL_FIELDS, "public recognizer channel")
        channels.append(
            PublicRecognizerObservableChannelV1._issue(
                _REGISTRY_ISSUE_TOKEN,
                quantity_id=row["quantity_id"],
                observable_id=row["observable_id"],
            )
        )
    candidate_count = root["maximum_candidate_count"]
    if type(candidate_count) is not int:
        raise TypeError("public recognizer candidate cap must be an exact integer")
    registry = PublicRecognizerRegistryV1._issue(
        _REGISTRY_ISSUE_TOKEN,
        theory_version_id=root["theory_version_id"],
        law_bindings=tuple(laws),
        observable_channels=tuple(channels),
        maximum_candidate_count=candidate_count,
    )
    if _encode_public_registry(registry) != payload:
        raise ValueError("public recognizer registry is not canonical")
    return registry


def _profile_uuid4_values(
    root: object,
    *,
    values: set[str] | None = None,
) -> set[str]:
    """Collect canonical UUIDv4 strings after the typed codec's own preflight."""

    collected = set() if values is None else values
    stack = [root]
    while stack:
        value = stack.pop()
        if type(value) is str:
            if (
                len(value) == 36
                and _UUID4.fullmatch(value) is not None
                and UUID(value).version == 4
            ):
                collected.add(value)
                if len(collected) > MAXIMUM_GLOBAL_SOURCE_UUIDS:
                    raise ValueError("global UUID collection exceeds its cap")
        elif type(value) is dict:
            stack.extend(value.keys())
            stack.extend(value.values())
        elif type(value) in (list, tuple):
            stack.extend(value)
    return collected


def _source_uuid_set(
    source_cases: tuple[TrustedRecognizerSourceCaseV1, ...],
) -> frozenset[str]:
    values: set[str] = set()
    registry = source_cases[0].adapter_registry
    for law in registry.law_bindings:
        values.add(law.family_id)
        values.update(wire_id for _, wire_id in law.role_ids)
    values.update(item.quantity_id for item in registry.observable_channels)
    if len(values) > MAXIMUM_GLOBAL_SOURCE_UUIDS:
        raise ValueError("global source UUID collection exceeds its cap")
    for source_case in source_cases:
        _profile_uuid4_values(
            encode_typed_transform_authority_profile_v1(source_case.authority),
            values=values,
        )
    return frozenset(values)


def _registry_scope_ids(
    registry: PublicRecognizerRegistryV1,
) -> tuple[frozenset[str], frozenset[str]]:
    adapter = registry.to_adapter_registry()
    return (
        frozenset(
            wire_id
            for law in adapter.law_bindings
            for _, wire_id in law.role_ids
        ),
        frozenset(item.quantity_id for item in adapter.observable_channels),
    )


def _compile_public_registry_from_live_renamings(
    *,
    source_registry: Phase2BAdapterRegistry,
    renamings: Mapping[tuple[str, str], str],
    typed_authority: PublicTransformEvidenceBundleV2,
) -> PublicRecognizerRegistryV1:
    _validate_source_registry(source_registry)
    aliases = set(_BRIDGE_FAMILY_BY_KIND.values())
    source_registry_uuids = {
        law.family_id for law in source_registry.law_bindings
    } | {
        wire_id
        for law in source_registry.law_bindings
        for _, wire_id in law.role_ids
    } | {
        item.quantity_id for item in source_registry.observable_channels
    }
    source_authority_uuids = {old_uuid for _, old_uuid in renamings}
    output_authority_uuids = set(renamings.values())
    if aliases & (
        source_registry_uuids
        | source_authority_uuids
        | output_authority_uuids
    ):
        raise ValueError("fixed bridge family alias collides with case UUIDs")
    expected = {
        ("role_candidate", wire_id)
        for law in source_registry.law_bindings
        for _, wire_id in law.role_ids
    } | {
        ("quantity", item.quantity_id)
        for item in source_registry.observable_channels
    }
    actual = {
        key
        for key in renamings
        if key[0] in ("role_candidate", "quantity")
    }
    if actual != expected:
        raise ValueError("source registry live namespace coverage drift")
    if any(key not in renamings for key in expected):
        raise ValueError("source registry live renaming is incomplete")
    law_by_kind = {item.law_kind: item for item in source_registry.law_bindings}
    laws = tuple(
        PublicRecognizerLawBindingV1._issue(
            _REGISTRY_ISSUE_TOKEN,
            law_id=law_by_kind[law_kind].law_id,
            law_kind=law_kind,
            role_ids=tuple(
                (
                    semantic_role,
                    renamings[("role_candidate", old_wire_id)],
                )
                for semantic_role, old_wire_id in law_by_kind[
                    law_kind
                ].role_ids
            ),
            required_observable_ids=(
                law_by_kind[law_kind].required_observable_ids
            ),
        )
        for law_kind in LawKind
    )
    channels = tuple(
        PublicRecognizerObservableChannelV1._issue(
            _REGISTRY_ISSUE_TOKEN,
            quantity_id=renamings[("quantity", item.quantity_id)],
            observable_id=item.observable_id,
        )
        for item in sorted(
            source_registry.observable_channels,
            key=lambda value: value.observable_id,
        )
    )
    registry = PublicRecognizerRegistryV1._issue(
        _REGISTRY_ISSUE_TOKEN,
        theory_version_id=source_registry.theory_version_id,
        law_bindings=laws,
        observable_channels=channels,
        maximum_candidate_count=source_registry.maximum_candidate_count,
    )
    adapter = registry.to_adapter_registry()
    if {wire_id for law in adapter.law_bindings for _, wire_id in law.role_ids} != set(
        typed_authority.base_bundle.role_ids
    ) or {item.quantity_id for item in adapter.observable_channels} != set(
        typed_authority.base_bundle.quantity_ids
    ):
        raise ValueError("public registry does not cover the renamed authority")
    return registry


class RecognizerInputArchiveDisposition(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True, init=False)
class TrustedRecognizerInputRowV1:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    authority_content_id: str
    transform_result_id: str
    public_registry: PublicRecognizerRegistryV1
    public_registry_id: str
    row_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("recognizer-input rows are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        envelope: bytes,
        public_registry: PublicRecognizerRegistryV1,
        typed_replay: TypedTrustedEnvelopeReplayV1 | None = None,
    ) -> "TrustedRecognizerInputRowV1":
        if token is not _ROW_ISSUE_TOKEN:
            raise TypeError("recognizer-input row issuer token mismatch")
        typed = (
            decode_and_replay_typed_trusted_envelope_v1(envelope)
            if typed_replay is None
            else typed_replay
        )
        if (
            type(typed) is not TypedTrustedEnvelopeReplayV1
            or typed.envelope != envelope
        ):
            raise ValueError("recognizer-input row typed replay context drift")
        value = object.__new__(cls)
        object.__setattr__(value, "envelope", envelope)
        object.__setattr__(value, "envelope_id", typed.envelope_id)
        object.__setattr__(value, "payload_sha256", typed.payload_sha256)
        object.__setattr__(
            value,
            "authority_content_id",
            typed.authority_content_id,
        )
        object.__setattr__(
            value,
            "transform_result_id",
            typed.transform_result_id,
        )
        object.__setattr__(value, "public_registry", public_registry)
        object.__setattr__(
            value,
            "public_registry_id",
            public_registry.registry_id,
        )
        object.__setattr__(
            value,
            "row_id",
            _recognizer_input_row_id(
                envelope_id=typed.envelope_id,
                payload_sha256=typed.payload_sha256,
                authority_content_id=typed.authority_content_id,
                transform_result_id=typed.transform_result_id,
                public_registry_id=public_registry.registry_id,
            ),
        )
        value._validate(
            typed_replay=typed,
            context_token=_ROW_CONTEXT_TOKEN,
        )
        return value

    def _validate(
        self,
        *,
        typed_replay: TypedTrustedEnvelopeReplayV1 | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not TrustedRecognizerInputRowV1:
            raise TypeError("recognizer-input row must use the exact type")
        if (typed_replay is None) is not (context_token is None):
            raise TypeError("recognizer-input row replay context token mismatch")
        if (
            typed_replay is not None
            and context_token is not _ROW_CONTEXT_TOKEN
        ):
            raise TypeError("recognizer-input row replay context is private")
        if type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_BYTES:
            raise TypeError("recognizer-input row needs one exact envelope")
        _digest(
            self.envelope_id,
            "phase2b_trusted_envelope_",
            "recognizer-input row envelope ID",
        )
        _sha256_text(
            self.payload_sha256,
            "recognizer-input row payload SHA-256",
        )
        _digest(
            self.authority_content_id,
            "phase2b_public_transform_evidence_",
            "recognizer-input row authority content ID",
        )
        _digest(
            self.transform_result_id,
            "phase2b_exact_transform_result_",
            "recognizer-input row transform result ID",
        )
        _digest(
            self.public_registry_id,
            "phase2b_public_recognizer_registry_",
            "recognizer-input row public registry ID",
        )
        _digest(
            self.row_id,
            "phase2b_recognizer_input_row_",
            "recognizer-input row ID",
        )
        if type(self.public_registry) is not PublicRecognizerRegistryV1:
            raise TypeError("recognizer-input row needs an exact public registry")
        self.public_registry._validate()
        typed = (
            decode_and_replay_typed_trusted_envelope_v1(self.envelope)
            if typed_replay is None
            else typed_replay
        )
        if (
            type(typed) is not TypedTrustedEnvelopeReplayV1
            or typed.envelope != self.envelope
        ):
            raise ValueError("recognizer-input row typed replay context drift")
        registry_roles, registry_quantities = _registry_scope_ids(
            self.public_registry
        )
        if (
            registry_roles != frozenset(typed.authority.base_bundle.role_ids)
            or registry_quantities
            != frozenset(typed.authority.base_bundle.quantity_ids)
            or registry_quantities
            != frozenset(
                typed.authority.base_bundle.task_target.quantity_ids
            )
            or frozenset(
                typed.authority.base_bundle.task_target.entity_ids
            )
            != frozenset(
                item.entity_id
                for item in typed.authority.base_bundle.entity_candidates
            )
            or any(
                not item.role_candidate_ids
                or not frozenset(item.role_candidate_ids).issubset(
                    registry_roles
                )
                for item in typed.authority.base_bundle.entity_candidates
            )
        ):
            raise ValueError(
                "public registry and envelope authority scope disagree"
            )
        authority_uuids = _profile_uuid4_values(
            encode_typed_transform_authority_profile_v1(typed.authority)
        )
        if set(_BRIDGE_FAMILY_BY_KIND.values()) & authority_uuids:
            raise ValueError("public family alias collides with output authority")
        if (
            self.envelope_id != typed.envelope_id
            or self.payload_sha256 != typed.payload_sha256
            or self.authority_content_id != typed.authority_content_id
            or self.transform_result_id != typed.transform_result_id
            or self.public_registry_id != self.public_registry.registry_id
        ):
            raise ValueError("recognizer-input row content roots drift")
        expected = _recognizer_input_row_id(
            envelope_id=self.envelope_id,
            payload_sha256=self.payload_sha256,
            authority_content_id=self.authority_content_id,
            transform_result_id=self.transform_result_id,
            public_registry_id=self.public_registry_id,
        )
        if self.row_id != expected:
            raise ValueError("recognizer-input row ID drift")


def _recognizer_input_row_id(
    *,
    envelope_id: str,
    payload_sha256: str,
    authority_content_id: str,
    transform_result_id: str,
    public_registry_id: str,
) -> str:
    _digest(envelope_id, "phase2b_trusted_envelope_", "envelope ID")
    _digest(
        authority_content_id,
        "phase2b_public_transform_evidence_",
        "authority content ID",
    )
    _digest(
        transform_result_id,
        "phase2b_exact_transform_result_",
        "transform result ID",
    )
    _digest(
        public_registry_id,
        "phase2b_public_recognizer_registry_",
        "public registry ID",
    )
    if (
        type(payload_sha256) is not str
        or len(payload_sha256) != 64
        or any(item not in "0123456789abcdef" for item in payload_sha256)
    ):
        raise ValueError("recognizer-input payload SHA-256 is invalid")
    payload = encode_phase2b_jcs_profile_v1(
        {
            "authority_content_id": authority_content_id,
            "envelope_id": envelope_id,
            "payload_sha256": payload_sha256,
            "public_registry_id": public_registry_id,
            "transform_result_id": transform_result_id,
        }
    )
    return "phase2b_recognizer_input_row_" + hashlib.sha256(
        _ROW_DOMAIN + payload
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustedRecognizerInputArchiveRejectionV1:
    disposition: RecognizerInputArchiveDisposition
    reason: str
    case_count: int
    batch_id: str | None
    archive: None = None
    rows: tuple[()] = ()
    public_registry_ids: tuple[()] = ()

    def __post_init__(self) -> None:
        if type(self) is not TrustedRecognizerInputArchiveRejectionV1:
            raise TypeError("recognizer-input rejection must use the exact type")
        if self.disposition is not RecognizerInputArchiveDisposition.ABSTAIN:
            raise ValueError("recognizer-input rejection must abstain")
        _ascii(self.reason, "recognizer-input rejection reason")
        if type(self.case_count) is not int or self.case_count < 0:
            raise ValueError("recognizer-input rejection count is invalid")
        if self.batch_id is not None:
            _digest(
                self.batch_id,
                "phase2b_trusted_wire_batch_",
                "recognizer-input rejection batch ID",
            )
        if (
            self.archive is not None
            or type(self.rows) is not tuple
            or self.rows != ()
            or type(self.public_registry_ids) is not tuple
            or self.public_registry_ids != ()
        ):
            raise ValueError("recognizer-input rejection leaked partial output")


@dataclass(frozen=True, slots=True)
class _ParsedRecognizerInputArchiveV1:
    archive_id: str
    metadata: dict[str, object]
    rows: tuple[TrustedRecognizerInputRowV1, ...]


def _validate_archive_metadata(
    metadata: dict[str, object],
    *,
    row_count: int,
) -> None:
    _closed_mapping(metadata, _METADATA_FIELDS, "recognizer-input metadata")
    if (
        metadata["archive_version"]
        != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION
        or metadata["archive_policy_id"]
        != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID
        or metadata["batch_policy_id"] != TRUSTED_WIRE_BATCH_POLICY_ID
        or metadata["typed_authority_schema_id"] != TYPED_AUTHORITY_SCHEMA_ID
    ):
        raise ValueError("recognizer-input metadata policy drift")
    if type(metadata["row_count"]) is not int or metadata["row_count"] != row_count:
        raise ValueError("recognizer-input metadata row count drift")
    _digest(
        metadata["archive_policy_id"],
        "phase2b_recognizer_input_archive_policy_",
        "recognizer-input archive policy ID",
    )
    _digest(
        metadata["batch_id"],
        "phase2b_trusted_wire_batch_",
        "recognizer-input batch ID",
    )
    _digest(
        metadata["batch_policy_id"],
        "phase2b_trusted_wire_batch_policy_",
        "recognizer-input batch policy ID",
    )
    _digest(
        metadata["execution_commitment"],
        "phase2b_trusted_wire_run_",
        "recognizer-input run commitment",
    )
    _digest(
        metadata["typed_replay_receipt_id"],
        "phase2b_typed_trusted_wire_batch_replay_",
        "recognizer-input typed replay receipt ID",
    )
    _digest(
        metadata["secret_replay_receipt_id"],
        "phase2b_trusted_wire_secret_replay_",
        "recognizer-input secret replay receipt ID",
    )
    _digest(
        metadata["source_registry_id"],
        "phase2b_adapter_registry_",
        "recognizer-input source registry ID",
    )
    _digest(
        metadata["typed_authority_schema_id"],
        "phase2b_trusted_wire_typed_authority_schema_",
        "recognizer-input typed authority schema ID",
    )


def _archive_metadata(
    *,
    batch: TrustedWireBatchV1,
    typed_replay_receipt_id: str,
    secret_replay_receipt_id: str,
    source_registry_id: str,
    row_count: int,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID,
        "archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION,
        "batch_id": batch.batch_id,
        "batch_policy_id": TRUSTED_WIRE_BATCH_POLICY_ID,
        "row_count": row_count,
        "execution_commitment": batch.run_id_commitment,
        "secret_replay_receipt_id": secret_replay_receipt_id,
        "source_registry_id": source_registry_id,
        "typed_authority_schema_id": TYPED_AUTHORITY_SCHEMA_ID,
        "typed_replay_receipt_id": typed_replay_receipt_id,
    }
    _validate_archive_metadata(metadata, row_count=row_count)
    return metadata


def _recognizer_archive_id(archive: bytes) -> str:
    if type(archive) is not bytes:
        raise TypeError("recognizer-input archive ID requires exact bytes")
    if not archive or len(archive) > MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES:
        raise ValueError("recognizer-input archive byte budget drift")
    return "phase2b_recognizer_input_archive_" + hashlib.sha256(
        _ARCHIVE_DOMAIN + archive
    ).hexdigest()


def _encode_recognizer_input_archive(
    *,
    metadata: dict[str, object],
    rows: tuple[TrustedRecognizerInputRowV1, ...],
    validated_rows_token: object | None = None,
) -> bytes:
    if (
        type(rows) is not tuple
        or not rows
        or len(rows) > MAXIMUM_BATCH_AUTHORITIES
        or any(type(item) is not TrustedRecognizerInputRowV1 for item in rows)
    ):
        raise TypeError("recognizer-input archive needs exact bounded rows")
    _validate_archive_metadata(metadata, row_count=len(rows))
    metadata_payload = encode_phase2b_jcs_profile_v1(metadata)
    if not metadata_payload or len(metadata_payload) > MAXIMUM_ARCHIVE_METADATA_BYTES:
        raise ValueError("recognizer-input metadata exceeds its byte cap")
    encoded_rows: list[tuple[bytes, bytes]] = []
    total_bytes = ARCHIVE_HEADER_BYTES + len(metadata_payload)
    for row in rows:
        if validated_rows_token is None:
            row._validate()
        elif validated_rows_token is not _ARCHIVE_ENCODE_TOKEN:
            raise TypeError("recognizer-input archive encoder token mismatch")
        registry_payload = _encode_public_registry(row.public_registry)
        if not registry_payload or len(registry_payload) > MAXIMUM_PUBLIC_REGISTRY_BYTES:
            raise ValueError("public registry exceeds its archive row cap")
        total_bytes += 4 + len(registry_payload) + ENVELOPE_BYTES
        if total_bytes > MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES:
            raise ValueError("recognizer-input archive exceeds its byte cap")
        encoded_rows.append((registry_payload, row.envelope))
    header = _ARCHIVE_HEADER.pack(
        ARCHIVE_MAGIC,
        ARCHIVE_WIRE_VERSION,
        0,
        len(rows),
        len(metadata_payload),
        hashlib.sha256(metadata_payload).digest(),
    )
    parts = [header, metadata_payload]
    for registry_payload, envelope in encoded_rows:
        parts.extend(
            (
                struct.pack(">I", len(registry_payload)),
                registry_payload,
                envelope,
            )
        )
    archive = b"".join(parts)
    if len(archive) != total_bytes:
        raise RuntimeError("recognizer-input archive length accounting drift")
    return archive


def _parse_public_archive(archive: bytes) -> _ParsedRecognizerInputArchiveV1:
    if type(archive) is not bytes:
        raise TypeError("recognizer-input archive input must use exact bytes")
    if (
        len(archive) < ARCHIVE_HEADER_BYTES
        or len(archive) > MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES
    ):
        raise ValueError("recognizer-input archive byte budget drift")
    magic, wire_version, flags, row_count, metadata_bytes, metadata_sha = (
        _ARCHIVE_HEADER.unpack_from(archive, 0)
    )
    if (
        magic != ARCHIVE_MAGIC
        or wire_version != ARCHIVE_WIRE_VERSION
        or flags != 0
        or not 1 <= row_count <= MAXIMUM_BATCH_AUTHORITIES
        or not 1 <= metadata_bytes <= MAXIMUM_ARCHIVE_METADATA_BYTES
    ):
        raise ValueError("recognizer-input archive header drift")
    offset = ARCHIVE_HEADER_BYTES
    metadata_end = offset + metadata_bytes
    minimum_rows_bytes = row_count * (4 + 1 + ENVELOPE_BYTES)
    if metadata_end + minimum_rows_bytes > len(archive):
        raise ValueError("recognizer-input archive is truncated")
    metadata_payload = archive[offset:metadata_end]
    if hashlib.sha256(metadata_payload).digest() != metadata_sha:
        raise ValueError("recognizer-input metadata digest drift")
    decoded_metadata = decode_phase2b_jcs_profile_v1(metadata_payload)
    metadata = _closed_mapping(
        decoded_metadata,
        _METADATA_FIELDS,
        "recognizer-input metadata",
    )
    if encode_phase2b_jcs_profile_v1(metadata) != metadata_payload:
        raise ValueError("recognizer-input metadata is not canonical")
    _validate_archive_metadata(metadata, row_count=row_count)
    offset = metadata_end
    rows: list[TrustedRecognizerInputRowV1] = []
    opaque_ids_seen: set[str] = set()
    authority_uuids_seen: set[str] = set()
    for _ in range(row_count):
        if offset + 4 > len(archive):
            raise ValueError("recognizer-input registry length is truncated")
        (registry_bytes,) = struct.unpack_from(">I", archive, offset)
        offset += 4
        if not 1 <= registry_bytes <= MAXIMUM_PUBLIC_REGISTRY_BYTES:
            raise ValueError("recognizer-input registry length exceeds its cap")
        envelope_end = offset + registry_bytes + ENVELOPE_BYTES
        if envelope_end > len(archive):
            raise ValueError("recognizer-input row is truncated")
        registry_payload = archive[offset : offset + registry_bytes]
        offset += registry_bytes
        envelope = archive[offset : offset + ENVELOPE_BYTES]
        offset += ENVELOPE_BYTES
        registry = _decode_public_registry(registry_payload)
        typed = decode_and_replay_typed_trusted_envelope_v1(envelope)
        row = TrustedRecognizerInputRowV1._issue(
            _ROW_ISSUE_TOKEN,
            envelope=envelope,
            public_registry=registry,
            typed_replay=typed,
        )
        role_ids, quantity_ids = _registry_scope_ids(registry)
        case_opaque_ids = set(role_ids | quantity_ids)
        if case_opaque_ids & opaque_ids_seen:
            raise ValueError("recognizer-input rows repeat opaque registry IDs")
        opaque_ids_seen.update(case_opaque_ids)
        case_authority_uuids = _profile_uuid4_values(
            encode_typed_transform_authority_profile_v1(typed.authority)
        )
        if case_authority_uuids & authority_uuids_seen:
            raise ValueError("recognizer-input rows repeat an authority UUID")
        authority_uuids_seen.update(case_authority_uuids)
        if len(authority_uuids_seen) > MAXIMUM_GLOBAL_SOURCE_UUIDS:
            raise ValueError("global public UUID collection exceeds its cap")
        rows.append(row)
    if offset != len(archive):
        raise ValueError("recognizer-input archive has trailing bytes")
    frozen_rows = tuple(rows)
    for roots, name in (
        (tuple(item.row_id for item in frozen_rows), "row"),
        (tuple(item.envelope_id for item in frozen_rows), "envelope"),
        (tuple(item.public_registry_id for item in frozen_rows), "registry"),
    ):
        if len(set(roots)) != len(roots):
            raise ValueError(f"recognizer-input archive repeats a {name} root")
    return _ParsedRecognizerInputArchiveV1(
        archive_id=_recognizer_archive_id(archive),
        metadata=metadata,
        rows=frozen_rows,
    )


@dataclass(frozen=True, slots=True, init=False)
class DecodedRecognizerInputArchiveV1:
    disposition: RecognizerInputArchiveDisposition
    archive: bytes
    archive_id: str
    schema_version: str
    policy_id: str
    batch_id: str
    batch_policy_id: str
    run_id_commitment: str
    typed_replay_receipt_id: str
    secret_replay_receipt_id: str
    source_registry_id: str
    rows: tuple[TrustedRecognizerInputRowV1, ...]
    row_ids: tuple[str, ...]
    envelope_ids: tuple[str, ...]
    public_registry_ids: tuple[str, ...]
    authority_content_ids: tuple[str, ...]
    transform_result_ids: tuple[str, ...]
    claim_level: str
    structural_archive_verified: bool
    row_bijection_verified: bool
    registry_schema_verified: bool
    direct_payload_transform_replay_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    secret_custodian_replay_verified: bool
    origin_authenticated: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    recognizer_executed: bool
    prediction_archive_evaluated: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("decoded recognizer-input archives are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        archive: bytes,
        parsed: _ParsedRecognizerInputArchiveV1,
    ) -> "DecodedRecognizerInputArchiveV1":
        if token is not _DECODE_ISSUE_TOKEN:
            raise TypeError("decoded recognizer-input issuer token mismatch")
        metadata = parsed.metadata
        rows = parsed.rows
        value = object.__new__(cls)
        frozen: tuple[tuple[str, object], ...] = (
            ("disposition", RecognizerInputArchiveDisposition.COMPLETE),
            ("archive", archive),
            ("archive_id", parsed.archive_id),
            ("schema_version", TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION),
            ("policy_id", RECOGNIZER_INPUT_ARCHIVE_POLICY_ID),
            ("batch_id", metadata["batch_id"]),
            ("batch_policy_id", metadata["batch_policy_id"]),
            ("run_id_commitment", metadata["execution_commitment"]),
            ("typed_replay_receipt_id", metadata["typed_replay_receipt_id"]),
            ("secret_replay_receipt_id", metadata["secret_replay_receipt_id"]),
            ("source_registry_id", metadata["source_registry_id"]),
            ("rows", rows),
            ("row_ids", tuple(item.row_id for item in rows)),
            ("envelope_ids", tuple(item.envelope_id for item in rows)),
            (
                "public_registry_ids",
                tuple(item.public_registry_id for item in rows),
            ),
            (
                "authority_content_ids",
                tuple(item.authority_content_id for item in rows),
            ),
            (
                "transform_result_ids",
                tuple(item.transform_result_id for item in rows),
            ),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("structural_archive_verified", True),
            ("row_bijection_verified", True),
            ("registry_schema_verified", True),
            ("direct_payload_transform_replay_verified", True),
            ("batch_policy_membership_verified", False),
            ("source_registry_projection_verified", False),
            ("secret_custodian_replay_verified", False),
            ("origin_authenticated", False),
            ("formal_covert_audit", False),
            ("sealed_holdout_eligible", False),
            ("recognizer_executed", False),
            ("prediction_archive_evaluated", False),
            ("c1_exit_evidence", False),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        value._validate(
            parsed=parsed,
            context_token=_PARSED_CONTEXT_TOKEN,
        )
        return value

    def _validate(
        self,
        *,
        parsed: _ParsedRecognizerInputArchiveV1 | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not DecodedRecognizerInputArchiveV1:
            raise TypeError("decoded recognizer-input archive needs the exact type")
        if (parsed is None) is not (context_token is None):
            raise TypeError("decoded recognizer-input parsed context token mismatch")
        if parsed is not None and context_token is not _PARSED_CONTEXT_TOKEN:
            raise TypeError("decoded recognizer-input parsed context is private")
        if type(self.archive) is not bytes:
            raise TypeError("decoded recognizer-input archive needs exact bytes")
        if self.disposition is not RecognizerInputArchiveDisposition.COMPLETE:
            raise ValueError("decoded recognizer-input disposition drift")
        _exact_bool_values(
            self.structural_archive_verified,
            self.row_bijection_verified,
            self.registry_schema_verified,
            self.direct_payload_transform_replay_verified,
            self.batch_policy_membership_verified,
            self.source_registry_projection_verified,
            self.secret_custodian_replay_verified,
            self.origin_authenticated,
            self.formal_covert_audit,
            self.sealed_holdout_eligible,
            self.recognizer_executed,
            self.prediction_archive_evaluated,
            self.c1_exit_evidence,
        )
        if not all(
            (
                self.structural_archive_verified,
                self.row_bijection_verified,
                self.registry_schema_verified,
                self.direct_payload_transform_replay_verified,
            )
        ) or any(
            (
                self.batch_policy_membership_verified,
                self.source_registry_projection_verified,
                self.secret_custodian_replay_verified,
                self.origin_authenticated,
                self.formal_covert_audit,
                self.sealed_holdout_eligible,
                self.recognizer_executed,
                self.prediction_archive_evaluated,
                self.c1_exit_evidence,
            )
        ):
            raise ValueError("decoded recognizer-input claim boundary drift")
        _digest(
            self.archive_id,
            "phase2b_recognizer_input_archive_",
            "decoded recognizer-input archive ID",
        )
        _ascii(self.schema_version, "decoded recognizer-input schema version")
        _digest(
            self.policy_id,
            "phase2b_recognizer_input_archive_policy_",
            "decoded recognizer-input policy ID",
        )
        _digest(
            self.batch_id,
            "phase2b_trusted_wire_batch_",
            "decoded recognizer-input batch ID",
        )
        _digest(
            self.batch_policy_id,
            "phase2b_trusted_wire_batch_policy_",
            "decoded recognizer-input batch policy ID",
        )
        _digest(
            self.run_id_commitment,
            "phase2b_trusted_wire_run_",
            "decoded recognizer-input execution commitment",
        )
        _digest(
            self.typed_replay_receipt_id,
            "phase2b_typed_trusted_wire_batch_replay_",
            "decoded recognizer-input typed replay receipt ID",
        )
        _digest(
            self.secret_replay_receipt_id,
            "phase2b_trusted_wire_secret_replay_",
            "decoded recognizer-input secret replay receipt ID",
        )
        _digest(
            self.source_registry_id,
            "phase2b_adapter_registry_",
            "decoded recognizer-input source registry ID",
        )
        _ascii(self.claim_level, "decoded recognizer-input claim level")
        if (
            type(self.rows) is not tuple
            or not 1 <= len(self.rows) <= MAXIMUM_BATCH_AUTHORITIES
            or any(type(item) is not TrustedRecognizerInputRowV1 for item in self.rows)
        ):
            raise TypeError("decoded recognizer-input rows need an exact bounded tuple")
        if parsed is None:
            for item in self.rows:
                item._validate()
        root_tuples = (
            self.row_ids,
            self.envelope_ids,
            self.public_registry_ids,
            self.authority_content_ids,
            self.transform_result_ids,
        )
        if any(
            type(items) is not tuple or len(items) != len(self.rows)
            for items in root_tuples
        ):
            raise TypeError("decoded recognizer-input roots need exact full tuples")
        for item in self.row_ids:
            _digest(
                item,
                "phase2b_recognizer_input_row_",
                "decoded recognizer-input row ID",
            )
        for item in self.envelope_ids:
            _digest(
                item,
                "phase2b_trusted_envelope_",
                "decoded recognizer-input envelope ID",
            )
        for item in self.public_registry_ids:
            _digest(
                item,
                "phase2b_public_recognizer_registry_",
                "decoded recognizer-input public registry ID",
            )
        for item in self.authority_content_ids:
            _digest(
                item,
                "phase2b_public_transform_evidence_",
                "decoded recognizer-input authority content ID",
            )
        for item in self.transform_result_ids:
            _digest(
                item,
                "phase2b_exact_transform_result_",
                "decoded recognizer-input transform result ID",
            )
        current = _parse_public_archive(self.archive) if parsed is None else parsed
        metadata = current.metadata
        if (
            self.disposition is not RecognizerInputArchiveDisposition.COMPLETE
            or self.archive_id != current.archive_id
            or self.schema_version != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION
            or self.policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID
            or self.batch_id != metadata["batch_id"]
            or self.batch_policy_id != metadata["batch_policy_id"]
            or self.run_id_commitment != metadata["execution_commitment"]
            or self.typed_replay_receipt_id
            != metadata["typed_replay_receipt_id"]
            or self.secret_replay_receipt_id
            != metadata["secret_replay_receipt_id"]
            or self.source_registry_id != metadata["source_registry_id"]
            or self.rows != current.rows
            or self.row_ids != tuple(item.row_id for item in current.rows)
            or self.envelope_ids
            != tuple(item.envelope_id for item in current.rows)
            or self.public_registry_ids
            != tuple(item.public_registry_id for item in current.rows)
            or self.authority_content_ids
            != tuple(item.authority_content_id for item in current.rows)
            or self.transform_result_ids
            != tuple(item.transform_result_id for item in current.rows)
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("decoded recognizer-input receipt drift")


def _decode_public_archive(
    archive: bytes,
) -> DecodedRecognizerInputArchiveV1:
    parsed = _parse_public_archive(archive)
    return DecodedRecognizerInputArchiveV1._issue(
        _DECODE_ISSUE_TOKEN,
        archive=archive,
        parsed=parsed,
    )


def _safe_rejection_batch_id(value: object) -> str | None:
    prefix = "phase2b_trusted_wire_batch_"
    if type(value) is not str or len(value) != len(prefix) + 64:
        return None
    suffix = value[len(prefix) :] if value.startswith(prefix) else ""
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        return None
    return value


def _archive_rejection(
    *,
    reason: str,
    case_count: int,
    batch_id: str | None,
) -> TrustedRecognizerInputArchiveRejectionV1:
    return TrustedRecognizerInputArchiveRejectionV1(
        disposition=RecognizerInputArchiveDisposition.ABSTAIN,
        reason=reason,
        case_count=case_count,
        batch_id=batch_id,
    )


def _issue_trusted_archive(
    *,
    typed_replay: TypedTrustedWireBatchReplayV1,
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV1,
    source_cases: tuple[TrustedRecognizerSourceCaseV1, ...],
) -> DecodedRecognizerInputArchiveV1 | TrustedRecognizerInputArchiveRejectionV1:
    if type(typed_replay) is not TypedTrustedWireBatchReplayV1:
        raise TypeError("recognizer-input issuer requires the exact Stage-C receipt")
    if type(run_id) is not bytes:
        raise TypeError("recognizer-input issuer run ID must use exact bytes")
    if type(key_sources) is not TrustedWireKeySourcesV1:
        raise TypeError("recognizer-input issuer requires exact key sources")
    if type(source_cases) is not tuple:
        raise TypeError("recognizer-input issuer source cases must use an exact tuple")
    case_count = len(source_cases)
    batch_id = _safe_rejection_batch_id(getattr(typed_replay, "batch_id", None))
    if not 1 <= case_count <= MAXIMUM_BATCH_AUTHORITIES:
        return _archive_rejection(
            reason="source_case_count_out_of_bounds",
            case_count=case_count,
            batch_id=batch_id,
        )
    if any(type(item) is not TrustedRecognizerSourceCaseV1 for item in source_cases):
        raise TypeError("recognizer-input issuer requires exact source case rows")
    try:
        key_sources.__post_init__()
        for source_case in source_cases:
            source_case.__post_init__()
        global_registry = source_cases[0].adapter_registry
        if any(
            item.adapter_registry != global_registry for item in source_cases
        ):
            raise ValueError("source cases do not share one exact global registry")
        _validate_source_registry(global_registry)
        global_source_uuids = _source_uuid_set(source_cases)
        fixed_aliases = frozenset(_BRIDGE_FAMILY_BY_KIND.values())
        if fixed_aliases & global_source_uuids:
            raise ValueError("fixed public family aliases collide with source UUIDs")
        typed_replay._validate()
        if (
            typed_replay.disposition
            is not TypedTrustedWireReplayDisposition.COMPLETE
            or len(typed_replay.rows) != case_count
        ):
            raise ValueError("Stage-C receipt is not a complete equal-sized batch")
        if tuple(item.authority for item in source_cases) != (
            typed_replay.source_authorities
        ):
            raise ValueError("source case order disagrees with Stage-C authority order")
        source_registry_id = global_registry.registry_id
        typed_replay_receipt_id = typed_replay.receipt_id
        public_authority_uuids_seen: set[str] = set()

        def compile_projection(
            source_index: int,
            renamings: Mapping[tuple[str, str], str],
            typed_authority: PublicTransformEvidenceBundleV2,
        ) -> object:
            if (
                type(source_index) is not int
                or not 0 <= source_index < case_count
            ):
                raise ValueError("live source index is outside the private case tuple")
            registry = _compile_public_registry_from_live_renamings(
                source_registry=source_cases[source_index].adapter_registry,
                renamings=renamings,
                typed_authority=typed_authority,
            )
            output_authority_uuids = _profile_uuid4_values(
                encode_typed_transform_authority_profile_v1(typed_authority)
            )
            public_roles, public_quantities = _registry_scope_ids(registry)
            case_public_uuids = (
                output_authority_uuids
                | set(public_roles)
                | set(public_quantities)
                | set(fixed_aliases)
            )
            if case_public_uuids & global_source_uuids:
                raise ValueError("public UUID collides with the global source set")
            unlinkable = case_public_uuids - set(fixed_aliases)
            if unlinkable & public_authority_uuids_seen:
                raise ValueError("public case UUID namespaces overlap")
            public_authority_uuids_seen.update(unlinkable)
            if len(public_authority_uuids_seen) > MAXIMUM_GLOBAL_SOURCE_UUIDS:
                raise ValueError("global public UUID collection exceeds its cap")
            return registry

        rebuilt, projections = (
            _trusted_wire_batch._build_trusted_wire_batch_core_v1(
                authorities=tuple(item.authority for item in source_cases),
                run_id=run_id,
                key_sources=key_sources,
                per_case_projection_compiler=compile_projection,
            )
        )
        if type(rebuilt) is not TrustedWireBatchV1:
            if type(rebuilt) is not TrustedWireBatchPreflightV1:
                raise TypeError("trusted-wire core returned an unknown type")
            return _archive_rejection(
                reason="trusted_wire_rebuild_or_projection_abstained",
                case_count=case_count,
                batch_id=batch_id,
            )
        rebuilt._validate()
        if rebuilt != typed_replay.batch:
            raise ValueError("one-pass rebuilt batch disagrees with Stage-C receipt")
        if (
            type(projections) is not tuple
            or len(projections) != case_count
            or any(
                type(item) is not PublicRecognizerRegistryV1
                for item in projections
            )
        ):
            raise ValueError("one-pass registry projections are incomplete")
        rows = tuple(
            TrustedRecognizerInputRowV1._issue(
                _ROW_ISSUE_TOKEN,
                envelope=envelope.envelope,
                public_registry=registry,
                typed_replay=typed_row,
            )
            for envelope, registry, typed_row in zip(
                rebuilt.envelopes,
                projections,
                typed_replay.rows,
                strict=True,
            )
        )
        if (
            tuple(item.envelope_id for item in rows)
            != tuple(item.envelope_id for item in typed_replay.rows)
            or tuple(item.authority_content_id for item in rows)
            != typed_replay.authority_content_ids
            or tuple(item.transform_result_id for item in rows)
            != typed_replay.transform_result_ids
        ):
            raise ValueError("archive rows disagree with Stage-C replay roots")
        metadata = _archive_metadata(
            batch=rebuilt,
            typed_replay_receipt_id=typed_replay_receipt_id,
            secret_replay_receipt_id=typed_replay.secret_replay_receipt_id,
            source_registry_id=source_registry_id,
            row_count=len(rows),
        )
        archive = _encode_recognizer_input_archive(
            metadata=metadata,
            rows=rows,
            validated_rows_token=_ARCHIVE_ENCODE_TOKEN,
        )
        decoded = _decode_public_archive(archive)
        if (
            decoded.batch_id != rebuilt.batch_id
            or decoded.run_id_commitment != rebuilt.run_id_commitment
            or decoded.typed_replay_receipt_id != typed_replay_receipt_id
            or decoded.secret_replay_receipt_id
            != typed_replay.secret_replay_receipt_id
            or decoded.source_registry_id != source_registry_id
            or decoded.rows != rows
        ):
            raise ValueError("archive public self-replay roots drift")
        return decoded
    except (AttributeError, KeyError, OverflowError, RuntimeError, TypeError, ValueError):
        return _archive_rejection(
            reason="trusted_recognizer_input_replay_failed",
            case_count=case_count,
            batch_id=batch_id,
        )


def _assert_public_field_manifests() -> None:
    mapping_actual = (
        tuple(sorted(item.name for item in fields(PublicRecognizerLawBindingV1))),
        tuple(
            sorted(
                item.name for item in fields(PublicRecognizerObservableChannelV1)
            )
        ),
        tuple(sorted(item.name for item in fields(PublicRecognizerRegistryV1))),
        tuple(sorted(item.name for item in fields(TrustedRecognizerInputRowV1))),
    )
    mapping_expected = (
        _LAW_FIELDS,
        _CHANNEL_FIELDS,
        _REGISTRY_FIELDS,
        _ROW_FIELDS,
    )
    receipt_actual = (
        tuple(item.name for item in fields(TrustedRecognizerSourceCaseV1)),
        tuple(
            item.name
            for item in fields(TrustedRecognizerInputArchiveRejectionV1)
        ),
        tuple(item.name for item in fields(DecodedRecognizerInputArchiveV1)),
    )
    receipt_expected = (
        _SOURCE_CASE_FIELD_MANIFEST,
        _REJECTION_FIELD_MANIFEST,
        _DECODED_ARCHIVE_FIELD_MANIFEST,
    )
    if mapping_actual != mapping_expected or receipt_actual != receipt_expected:
        raise RuntimeError("recognizer-input archive field manifest drift")


_assert_public_field_manifests()


def decode_public_recognizer_input_archive_v1(
    archive: bytes,
) -> DecodedRecognizerInputArchiveV1:
    """Decode one bounded public archive without secret/source claims."""

    return _decode_public_archive(archive)


def issue_trusted_recognizer_input_archive_v1(
    *,
    typed_replay: TypedTrustedWireBatchReplayV1,
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV1,
    source_cases: tuple[TrustedRecognizerSourceCaseV1, ...],
) -> DecodedRecognizerInputArchiveV1 | TrustedRecognizerInputArchiveRejectionV1:
    """Rebuild once and atomically issue the public recognizer-input archive."""

    return _issue_trusted_archive(
        typed_replay=typed_replay,
        run_id=run_id,
        key_sources=key_sources,
        source_cases=source_cases,
    )


__all__ = (
    "ARCHIVE_HEADER_BYTES",
    "ARCHIVE_MAGIC",
    "ARCHIVE_WIRE_VERSION",
    "FROZEN_BRIDGE_FAMILY_UUID_ALIASES",
    "MAXIMUM_ARCHIVE_METADATA_BYTES",
    "MAXIMUM_GLOBAL_SOURCE_UUIDS",
    "MAXIMUM_PUBLIC_REGISTRY_BYTES",
    "MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES",
    "PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID",
    "PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID",
    "PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_VERSION",
    "RECOGNIZER_INPUT_ARCHIVE_POLICY_ID",
    "TRUSTED_RECOGNIZER_INPUT_ARCHIVE_VERSION",
    "DecodedRecognizerInputArchiveV1",
    "PublicRecognizerLawBindingV1",
    "PublicRecognizerObservableChannelV1",
    "PublicRecognizerRegistryV1",
    "RecognizerInputArchiveDisposition",
    "TrustedRecognizerInputArchiveRejectionV1",
    "TrustedRecognizerInputRowV1",
    "TrustedRecognizerSourceCaseV1",
    "decode_public_recognizer_input_archive_v1",
    "issue_trusted_recognizer_input_archive_v1",
)

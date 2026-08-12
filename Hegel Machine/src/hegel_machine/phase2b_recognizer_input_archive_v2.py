"""Independent compact-V2 recognizer-input archive.

The private issuer observes the one live V2 allocation pass through the typed
replay core.  The public decoder verifies only bounded bytes, canonical closed
metadata and registries, row adjacency/scope, compact typed replay, and public
cross-row UUID separation.  No source or secret commitment is public.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
import struct
from types import MappingProxyType
from typing import Final, Mapping
from uuid import UUID

from .bootstrap import initial_theory
from .hashing import stable_hash
from .phase2b_adapter import LawWireBinding, ObservableChannelBinding, Phase2BAdapterRegistry
from .phase2b_exact_transform_semantics_v1 import PublicTransformEvidenceBundleV2
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from .phase2b_trusted_wire_batch_v2 import (
    DecodedTrustedEnvelopeV2,
    MAXIMUM_BATCH_V2_AUTHORITIES,
    TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    TrustedWireBatchV2,
    TrustedWireKeySourcesV2,
)
from . import phase2b_trusted_wire_batch_v2 as _batch_v2
from . import phase2b_trusted_wire_batch_v1 as _batch_v1
from .phase2b_trusted_wire_typed_authority_v1 import encode_typed_transform_authority_profile_v1
from .phase2b_trusted_wire_typed_authority_v2 import (
    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
    COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
)
from .phase2b_trusted_wire_typed_replay_v2 import (
    TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
    TypedTrustedWireBatchReplayV2,
    _replay_typed_trusted_wire_batch_core_v2,
    decode_and_replay_typed_trusted_envelope_v2,
)
from .phase2b_trusted_wire_v1 import (
    ENVELOPE_BYTES,
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    MAXIMUM_ARRAY_ENTRIES,
    MAXIMUM_ASCII_STRING_BYTES,
    MAXIMUM_UUID_OCCURRENCES,
    MAXIMUM_SAFE_INTEGER,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)
from .schema import LawKind


PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION: Final = "hegel-machine-phase2b-public-recognizer-registry/2"
TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION: Final = "hegel-machine-phase2b-trusted-recognizer-input-archive/2"
ARCHIVE_MAGIC_V2: Final = b"HGRIAV2\x00"
ARCHIVE_WIRE_VERSION_V2: Final = 2
_ARCHIVE_HEADER_V2: Final = struct.Struct(">8sHHII32s")
ARCHIVE_HEADER_BYTES_V2: Final = _ARCHIVE_HEADER_V2.size
MAXIMUM_ARCHIVE_METADATA_BYTES_V2: Final = 16_384
MAXIMUM_PUBLIC_REGISTRY_BYTES_V2: Final = 65_536
MAXIMUM_GLOBAL_SOURCE_UUIDS_V2: Final = (
    MAXIMUM_BATCH_V2_AUTHORITIES * MAXIMUM_UUID_OCCURRENCES
)
MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2: Final = (
    ARCHIVE_HEADER_BYTES_V2 + MAXIMUM_ARCHIVE_METADATA_BYTES_V2
    + MAXIMUM_BATCH_V2_AUTHORITIES * (4 + MAXIMUM_PUBLIC_REGISTRY_BYTES_V2 + ENVELOPE_BYTES)
)

_ARCHIVE_DOMAIN_V2: Final = b"HEGEL/PHASE2B/TRUSTED_RECOGNIZER_INPUT_ARCHIVE/V2\x00"
_ROW_DOMAIN_V2: Final = b"HEGEL/PHASE2B/RECOGNIZER_INPUT_ROW/V2\x00"
_REGISTRY_DOMAIN_V2: Final = b"HEGEL/PHASE2B/PUBLIC_RECOGNIZER_REGISTRY/V2\x00"
_REGISTRY_TOKEN_V2: Final = object()
_ROW_TOKEN_V2: Final = object()
_DECODE_TOKEN_V2: Final = object()

FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2: Final = (
    (LawKind.SYMMETRY, "58351910-f1ea-4613-b5b2-47d9cc2f1652"),
    (LawKind.MONOTONICITY, "16ba12ce-f178-4226-ac97-2120adb62073"),
    (LawKind.CONSERVATION, "773faef6-c762-4ca6-b389-f2a593cb1f99"),
    (LawKind.COMPLEMENTARITY, "431cb872-0237-4751-a3f8-e5fc2a2a3b38"),
    (LawKind.NEGATIVE_FEEDBACK, "1d9fd5a5-ac24-4dd0-9b70-e257391585e5"),
    (LawKind.LOCALITY, "c4a5cad4-444f-4e54-a341-c21ffe29d2c5"),
)
_BRIDGE_BY_KIND_V2: Final = dict(FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2)
_CANONICAL_BY_KIND_V2: Final = dict(frozen_phase2b_exact_freeze().family_mapping)
_FROZEN_THEORY_V2: Final = initial_theory()

_LAW_FIELDS_V2: Final = ("bridge_family_id", "canonical_family_id", "law_id", "law_kind", "required_observable_ids", "role_ids")
_CHANNEL_FIELDS_V2: Final = ("observable_id", "quantity_id")
_REGISTRY_FIELDS_V2: Final = ("family_alias_policy_id", "law_bindings", "maximum_candidate_count", "observable_channels", "schema_version", "theory_version_id")
_METADATA_FIELDS_V2: Final = (
    "archive_policy_id", "archive_version", "batch_id", "batch_policy_id", "row_count",
    "typed_replay_policy_id", "typed_authority_schema_id", "typed_authority_codec_version",
    "typed_authority_codec_policy_id", "public_registry_schema_id",
)
_ARCHIVE_METADATA_FIELDS_V2: Final = _METADATA_FIELDS_V2
_SOURCE_FIELDS_V2: Final = ("authority", "adapter_registry")
_ROW_FIELDS_V2: Final = (
    "authority_content_id", "envelope", "envelope_id", "namespace_audit_id", "padding_sha256",
    "payload_sha256", "public_registry", "public_registry_id", "row_id", "transform_result_id",
)
_REJECTION_FIELDS_V2: Final = ("disposition", "reason", "case_count", "batch_id", "archive", "rows", "row_ids", "envelope_ids", "public_registry_ids", "authority_content_ids", "transform_result_ids")
_TRUE_CLAIMS_V2: Final = ("structural_archive_verified", "row_bijection_verified", "registry_schema_verified", "registry_authority_exact_scope_verified", "compact_typed_replay_verified", "direct_payload_transform_replay_verified", "cross_row_public_uuid_disjoint_verified")
_FALSE_CLAIMS_V2: Final = ("batch_policy_membership_verified", "source_registry_projection_verified", "source_public_disjoint_verified", "single_live_allocation_verified", "secret_custodian_replay_verified", "origin_authenticated", "formal_uuid_audit", "formal_covert_audit", "sealed_holdout_eligible", "recognizer_executed", "prediction_archive_evaluated", "recognizer_capacity_evidence", "c1_exit_evidence")
_DECODED_FIELDS_V2: Final = (
    "disposition", "archive", "archive_id", "archive_version", "policy_id", "batch_id", "batch_policy_id",
    "typed_replay_policy_id", "typed_authority_schema_id", "typed_authority_codec_version",
    "typed_authority_codec_policy_id", "public_registry_schema_id", "rows", "row_ids", "envelope_ids",
    "public_registry_ids", "authority_content_ids", "transform_result_ids", "claim_level",
    *_TRUE_CLAIMS_V2, *_FALSE_CLAIMS_V2,
)

PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2: Final = stable_hash(
    {"aliases": tuple((kind.value, alias, _CANONICAL_BY_KIND_V2[kind].value) for kind, alias in FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2),
     "contract": ("fixed_public_uuidv4_alias", "source_family_uuid_not_public", "alias_may_repeat_only_across_rows"),
     "version": PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION},
    prefix="phase2b_public_recognizer_family_alias_policy_v2_",
)
PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2: Final = stable_hash(
    {"accepted_jcs": "phase2b_jcs_profile_v1_exact_reencode", "fields": _REGISTRY_FIELDS_V2,
     "law_fields": _LAW_FIELDS_V2, "channel_fields": _CHANNEL_FIELDS_V2,
     "family_alias_policy_id": PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
     "version": PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION},
    prefix="phase2b_public_recognizer_registry_schema_v2_",
)
_ARCHIVE_POLICY_VALUE_V2: Final = {
    "archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
    "wire": {
        "header_bytes": ARCHIVE_HEADER_BYTES_V2, "header_format": _ARCHIVE_HEADER_V2.format,
        "magic_hex": ARCHIVE_MAGIC_V2.hex(), "wire_version": ARCHIVE_WIRE_VERSION_V2,
        "header_fields": ("magic", "wire_version", "header_bytes", "metadata_bytes", "row_count", "metadata_sha256"),
        "row_grammar": "u32be_registry_length||canonical_registry_JCS||exact_65536_byte_V2_envelope",
    },
    "caps": {
        "archive_bytes": MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
        "authorities": MAXIMUM_BATCH_V2_AUTHORITIES, "envelope_bytes": ENVELOPE_BYTES,
        "global_source_uuids": MAXIMUM_GLOBAL_SOURCE_UUIDS_V2,
        "global_public_uuids": MAXIMUM_GLOBAL_SOURCE_UUIDS_V2,
        "global_uuid_sidecar_formula": "MAXIMUM_BATCH_V2_AUTHORITIES*MAXIMUM_UUID_OCCURRENCES_per_source_or_public_sidecar",
        "metadata_bytes": MAXIMUM_ARCHIVE_METADATA_BYTES_V2,
        "registry_bytes": MAXIMUM_PUBLIC_REGISTRY_BYTES_V2,
        "registry_channels": MAXIMUM_ARRAY_ENTRIES, "registry_laws": len(LawKind),
        "registry_text_bytes": 65_536, "roles_per_law": 64, "observables_per_law": 64,
    },
    "canonical_json": {"accepted_profile_id": JCS_PROFILE_ID, "field_manifest_id": FIELD_MANIFEST_ID, "exact_decode_reencode": True},
    "field_manifests": {
        "law": _LAW_FIELDS_V2, "channel": _CHANNEL_FIELDS_V2, "registry": _REGISTRY_FIELDS_V2,
        "metadata": _METADATA_FIELDS_V2, "source_case": _SOURCE_FIELDS_V2, "row": _ROW_FIELDS_V2,
        "rejection": _REJECTION_FIELDS_V2, "decoded": _DECODED_FIELDS_V2,
    },
    "metadata_fields": _METADATA_FIELDS_V2,
    "registry_schema_id": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
    "family_alias_policy_id": PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
    "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    "typed_replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
    "typed_authority_schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
    "typed_authority_codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
    "typed_authority_codec_policy_id": COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    "typed_authority": {"schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID, "codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION, "codec_policy_id": COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID},
    "claims": {"exact_builtin_bool": True, "public_true": _TRUE_CLAIMS_V2, "public_false": _FALSE_CLAIMS_V2, "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL},
    "roots": {
        "row": "sha256(V2_row_domain||accepted_JCS_of_all_seven_public_roots)",
        "registry": "sha256(V2_registry_domain||canonical_registry_JCS)",
        "archive": "sha256(V2_archive_domain||all_bounded_archive_bytes)",
        "validate_before_hash": True,
    },
    "private_issuance": {
        "typed_private_core_call_count": 1, "expected_batch_exact": True,
        "live_mapping_proxy_projection": True, "projection_order": "shuffled_envelope_order",
        "atomic_rejection": "archive_rows_and_all_ID_tuples_empty_batch_id_none",
        "source_sidecar": "all_source_authority_and_registry_family_role_quantity_UUIDs_never_allocator_seed",
        "source_public_disjoint_gate_is_not_durable_public_claim": True,
        "live_renaming_closure": (
            "exact_MappingProxyType_before_iteration",
            "entry_count_at_most_MAXIMUM_UUID_OCCURRENCES",
            "exact_2_tuple_key_of_allowed_namespace_and_canonical_old_UUIDv4",
            "canonical_new_UUIDv4_value_and_all_values_unique",
            "full_entry_closure_before_set_index_or_registry_projection",
        ),
    },
    "uuid_contract": {
        "fixed_aliases_may_repeat_across_rows": True,
        "fixed_aliases_collide_with_no_source_or_unlinkable_public_UUID": True,
        "cross_row_unlinkable": "authority_UUIDs_plus_registry_role_quantity_UUIDs_minus_fixed_aliases_are_disjoint",
    },
    "registry_authority_exact_scope": (
        "registry_roles_equals_authority_roles",
        "registry_quantities_equals_authority_quantities_equals_task_target_quantities",
        "task_target_entities_equals_entity_candidate_entities",
        "every_entity_has_nonempty_role_candidates_subset_of_registry_roles",
    ),
    "resource_order": "archive_cap_header_count_before_parse_hash;registry_envelope_caps_before_decode;exact_types_caps_before_set_sort_hash_decode",
    "public_exclusions": ("run_commitment", "collision_count", "typed_receipt", "secret_receipt", "source_registry", "source_roots", "source_or_secret_commitments"),
    "cross_rejection": ("archive_v1_magic", "batch_v1_envelope_magic", "registry_v1_schema", "metadata_v1_schema", "typed_authority_v1_schema"),
}
RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2: Final = stable_hash(
    _ARCHIVE_POLICY_VALUE_V2,
    prefix="phase2b_recognizer_input_archive_policy_v2_",
)

_UUID4_V2 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _ascii_v2(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > MAXIMUM_ASCII_STRING_BYTES or not value.isascii():
        raise ValueError(f"{name} must use exact bounded nonempty ASCII")
    return value


def _uuid4_v2(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 36 or _UUID4_V2.fullmatch(value) is None or UUID(value).version != 4:
        raise ValueError(f"{name} must be canonical lowercase UUIDv4")
    return value


def _digest_v2(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or len(value) != len(prefix) + 64 or not value.startswith(prefix):
        raise ValueError(f"{name} must be an exact prefixed SHA-256")
    if any(item not in "0123456789abcdef" for item in value[len(prefix):]):
        raise ValueError(f"{name} must end in lowercase SHA-256")
    return value


def _hex64_v2(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ValueError(f"{name} must be exact lowercase SHA-256")
    return value


def _closed_v2(value: object, manifest: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must use an exact mapping")
    if len(value) != len(manifest) or any(type(key) is not str for key in value) or set(value) != set(manifest):
        raise ValueError(f"{name} closed schema drift")
    return value


def _profile_uuid4_values_v2(root: object, *, values: set[str] | None = None) -> set[str]:
    collected = set() if values is None else values
    stack = [root]
    while stack:
        value = stack.pop()
        if type(value) is str:
            if len(value) == 36 and _UUID4_V2.fullmatch(value) is not None and UUID(value).version == 4:
                collected.add(value)
                if len(collected) > MAXIMUM_GLOBAL_SOURCE_UUIDS_V2:
                    raise ValueError("V2 global UUID sidecar cap exceeded")
        elif type(value) is dict:
            stack.extend(value.keys())
            stack.extend(value.values())
        elif type(value) in (tuple, list):
            stack.extend(value)
    return collected


def _validate_source_registry_v2(registry: Phase2BAdapterRegistry) -> None:
    if type(registry) is not Phase2BAdapterRegistry:
        raise TypeError("V2 source registry exact type drift")
    _digest_v2(registry.theory_version_id, "theory_", "V2 source theory")
    if registry.theory_version_id != _FROZEN_THEORY_V2.version_id:
        raise ValueError("V2 source registry theory drift")
    if type(registry.law_bindings) is not tuple or len(registry.law_bindings) != len(LawKind) or any(type(item) is not LawWireBinding for item in registry.law_bindings):
        raise TypeError("V2 source registry law shape drift")
    text_items = [registry.theory_version_id]
    for law in registry.law_bindings:
        if type(law.law_id) is not str or type(law.law_kind) is not LawKind or type(law.family_id) is not str:
            raise TypeError("V2 source law scalar type drift")
        if type(law.role_ids) is not tuple or not 1 <= len(law.role_ids) <= 64 or type(law.required_observable_ids) is not tuple or not 1 <= len(law.required_observable_ids) <= 64:
            raise TypeError("V2 source law array type or cap drift")
        for pair in law.role_ids:
            if type(pair) is not tuple or len(pair) != 2 or any(type(item) is not str for item in pair): raise TypeError("V2 source role pair drift")
        if any(type(item) is not str for item in law.required_observable_ids): raise TypeError("V2 source required observable drift")
        _ascii_v2(law.law_id, "V2 source law ID")
        _uuid4_v2(law.family_id, "V2 source family ID")
        for semantic, wire in law.role_ids:
            _ascii_v2(semantic, "V2 source semantic role")
            _uuid4_v2(wire, "V2 source role ID")
            text_items.extend((semantic, wire))
        for observable in law.required_observable_ids: text_items.append(_ascii_v2(observable, "V2 source required observable"))
        text_items.extend((law.law_id, law.law_kind.value, law.family_id))
        law.__post_init__()
    kinds = tuple(item.law_kind for item in registry.law_bindings)
    if set(kinds) != set(LawKind) or len(set(kinds)) != len(LawKind): raise ValueError("V2 source registry family coverage drift")
    if type(registry.observable_channels) is not tuple or not registry.observable_channels or len(registry.observable_channels) > MAXIMUM_ARRAY_ENTRIES or any(type(item) is not ObservableChannelBinding for item in registry.observable_channels):
        raise TypeError("V2 source registry channel shape drift")
    for channel in registry.observable_channels:
        if type(channel.quantity_id) is not str or type(channel.observable_id) is not str: raise TypeError("V2 source channel scalar drift")
        _uuid4_v2(channel.quantity_id, "V2 source quantity ID")
        _ascii_v2(channel.observable_id, "V2 source observable ID")
        text_items.extend((channel.quantity_id, channel.observable_id))
        channel.__post_init__()
    if sum(len(item.encode("ascii")) for item in text_items) > 65_536: raise ValueError("V2 source registry text cap exceeded")
    if type(registry.maximum_candidate_count) is not int or registry.maximum_candidate_count != 50_000:
        raise ValueError("V2 source candidate cap drift")
    families = {law.law_kind: law.family_id for law in registry.law_bindings}
    roles = {(law.law_id, semantic): wire for law in registry.law_bindings for semantic, wire in law.role_ids}
    quantities = {item.observable_id: item.quantity_id for item in registry.observable_channels}
    if len(roles) != 15 or len(quantities) != 35:
        raise ValueError("V2 source frozen vocabulary cardinality drift")
    rebuilt = Phase2BAdapterRegistry.from_theory(_FROZEN_THEORY_V2, family_ids=families, role_ids=roles, quantity_ids=quantities, maximum_candidate_count=50_000)
    if rebuilt != registry:
        raise ValueError("V2 source registry does not exactly rebuild theory")


class RecognizerInputArchiveDispositionV2(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True, slots=True)
class TrustedRecognizerSourceCaseV2:
    authority: PublicTransformEvidenceBundleV2
    adapter_registry: Phase2BAdapterRegistry

    def __post_init__(self) -> None:
        if type(self) is not TrustedRecognizerSourceCaseV2 or type(self.authority) is not PublicTransformEvidenceBundleV2 or type(self.adapter_registry) is not Phase2BAdapterRegistry:
            raise TypeError("V2 source case exact type drift")
        reason = _batch_v2._source_string_cap_preflight_v2(self.authority)
        if reason is not None: raise ValueError("V2 source authority preflight:" + reason)
        _validate_source_registry_v2(self.adapter_registry)
        encode_typed_transform_authority_profile_v1(self.authority)


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerLawBindingV2:
    law_id: str
    law_kind: LawKind
    canonical_family_id: CanonicalFamilyId
    bridge_family_id: str
    role_ids: tuple[tuple[str, str], ...]
    required_observable_ids: tuple[str, ...]
    def __init__(self, *args: object, **kwargs: object) -> None: raise TypeError("V2 law bindings are privately issued")

    @classmethod
    def _issue(cls, token: object, *, law_id: str, law_kind: LawKind, role_ids: tuple[tuple[str, str], ...], required_observable_ids: tuple[str, ...]) -> "PublicRecognizerLawBindingV2":
        if token is not _REGISTRY_TOKEN_V2: raise TypeError("V2 law issuer token drift")
        value = object.__new__(cls)
        for name, item in (("law_id", law_id), ("law_kind", law_kind), ("canonical_family_id", _CANONICAL_BY_KIND_V2[law_kind]), ("bridge_family_id", _BRIDGE_BY_KIND_V2[law_kind]), ("role_ids", role_ids), ("required_observable_ids", required_observable_ids)):
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerLawBindingV2 or type(self.law_kind) is not LawKind: raise TypeError("V2 law exact type drift")
        if type(self.law_id) is not str or type(self.canonical_family_id) is not CanonicalFamilyId or type(self.bridge_family_id) is not str or type(self.role_ids) is not tuple or type(self.required_observable_ids) is not tuple:
            raise TypeError("V2 law scalar or array exact type drift")
        _ascii_v2(self.law_id, "V2 public law ID")
        if type(self.canonical_family_id) is not CanonicalFamilyId or self.canonical_family_id is not _CANONICAL_BY_KIND_V2[self.law_kind] or self.bridge_family_id != _BRIDGE_BY_KIND_V2[self.law_kind]: raise ValueError("V2 family alias drift")
        _uuid4_v2(self.bridge_family_id, "V2 bridge family")
        if not self.role_ids or len(self.role_ids) > 64: raise ValueError("V2 public role array drift")
        semantics, wires = [], []
        for item in self.role_ids:
            if type(item) is not tuple or len(item) != 2: raise TypeError("V2 public role binding drift")
            if type(item[0]) is not str or type(item[1]) is not str: raise TypeError("V2 public role scalar drift")
            semantics.append(_ascii_v2(item[0], "V2 semantic role")); wires.append(_uuid4_v2(item[1], "V2 role ID"))
        if self.role_ids != tuple(sorted(self.role_ids)): raise ValueError("V2 public roles are not canonical")
        if len(set(semantics)) != len(semantics) or len(set(wires)) != len(wires): raise ValueError("V2 public role repeat")
        if not self.required_observable_ids or len(self.required_observable_ids) > 64 or any(type(item) is not str for item in self.required_observable_ids): raise ValueError("V2 observable array drift")
        for item in self.required_observable_ids: _ascii_v2(item, "V2 required observable")
        if self.required_observable_ids != tuple(sorted(self.required_observable_ids)): raise ValueError("V2 observables are not canonical")


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerObservableChannelV2:
    quantity_id: str
    observable_id: str
    def __init__(self, *args: object, **kwargs: object) -> None: raise TypeError("V2 channels are privately issued")

    @classmethod
    def _issue(cls, token: object, *, quantity_id: str, observable_id: str) -> "PublicRecognizerObservableChannelV2":
        if token is not _REGISTRY_TOKEN_V2: raise TypeError("V2 channel issuer token drift")
        value = object.__new__(cls); object.__setattr__(value, "quantity_id", quantity_id); object.__setattr__(value, "observable_id", observable_id); value._validate(); return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerObservableChannelV2: raise TypeError("V2 channel exact type drift")
        if type(self.quantity_id) is not str or type(self.observable_id) is not str: raise TypeError("V2 channel scalar type drift")
        _uuid4_v2(self.quantity_id, "V2 quantity ID"); _ascii_v2(self.observable_id, "V2 observable ID")


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerRegistryV2:
    schema_version: str
    theory_version_id: str
    law_bindings: tuple[PublicRecognizerLawBindingV2, ...]
    observable_channels: tuple[PublicRecognizerObservableChannelV2, ...]
    maximum_candidate_count: int
    family_alias_policy_id: str
    def __init__(self, *args: object, **kwargs: object) -> None: raise TypeError("V2 registries are privately issued")

    @classmethod
    def _issue(cls, token: object, *, theory_version_id: str, law_bindings: tuple[PublicRecognizerLawBindingV2, ...], observable_channels: tuple[PublicRecognizerObservableChannelV2, ...], maximum_candidate_count: int) -> "PublicRecognizerRegistryV2":
        if token is not _REGISTRY_TOKEN_V2: raise TypeError("V2 registry issuer token drift")
        value = object.__new__(cls)
        for name, item in (("schema_version", PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION), ("theory_version_id", theory_version_id), ("law_bindings", law_bindings), ("observable_channels", observable_channels), ("maximum_candidate_count", maximum_candidate_count), ("family_alias_policy_id", PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2)):
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerRegistryV2: raise TypeError("V2 registry exact type drift")
        if any(type(item) is not str for item in (self.schema_version, self.theory_version_id, self.family_alias_policy_id)):
            raise TypeError("V2 registry identity scalar type drift")
        if self.schema_version != PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION or self.family_alias_policy_id != PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2: raise ValueError("V2 registry identity drift")
        if self.theory_version_id != _FROZEN_THEORY_V2.version_id: raise ValueError("V2 registry theory drift")
        if type(self.law_bindings) is not tuple or len(self.law_bindings) != len(LawKind) or any(type(item) is not PublicRecognizerLawBindingV2 for item in self.law_bindings): raise TypeError("V2 registry law shape drift")
        for item in self.law_bindings: item._validate()
        if tuple(item.law_kind for item in self.law_bindings) != tuple(LawKind): raise ValueError("V2 registry family order drift")
        if len({item.law_id for item in self.law_bindings}) != len(self.law_bindings): raise ValueError("V2 registry repeats a law ID")
        if type(self.observable_channels) is not tuple or not self.observable_channels or len(self.observable_channels) > MAXIMUM_ARRAY_ENTRIES or any(type(item) is not PublicRecognizerObservableChannelV2 for item in self.observable_channels): raise TypeError("V2 registry channel shape drift")
        for item in self.observable_channels: item._validate()
        if self.observable_channels != tuple(sorted(self.observable_channels, key=lambda item: item.observable_id)): raise ValueError("V2 registry channel order drift")
        if len({item.quantity_id for item in self.observable_channels}) != len(self.observable_channels) or len({item.observable_id for item in self.observable_channels}) != len(self.observable_channels): raise ValueError("V2 registry channel repeat")
        if type(self.maximum_candidate_count) is not int or self.maximum_candidate_count != 50_000: raise ValueError("V2 registry candidate cap drift")
        if any(not set(law.required_observable_ids).issubset({item.observable_id for item in self.observable_channels}) for law in self.law_bindings): raise ValueError("V2 registry unknown observable")
        text_items = [self.schema_version, self.theory_version_id, self.family_alias_policy_id]
        for law in self.law_bindings:
            text_items.extend((law.law_id, law.law_kind.value, law.canonical_family_id.value, law.bridge_family_id, *law.required_observable_ids))
            for semantic, wire in law.role_ids: text_items.extend((semantic, wire))
        for channel in self.observable_channels: text_items.extend((channel.observable_id, channel.quantity_id))
        if sum(len(item.encode("ascii")) for item in text_items) > 65_536: raise ValueError("V2 public registry text cap exceeded")
        adapter = _registry_adapter_unchecked_v2(self)
        rebuilt = Phase2BAdapterRegistry.from_theory(
            _FROZEN_THEORY_V2,
            family_ids={kind: _BRIDGE_BY_KIND_V2[kind] for kind in LawKind},
            role_ids={(law.law_id, semantic): wire for law in adapter.law_bindings for semantic, wire in law.role_ids},
            quantity_ids={item.observable_id: item.quantity_id for item in adapter.observable_channels},
            maximum_candidate_count=50_000,
        )
        if rebuilt != adapter: raise ValueError("V2 public registry does not exactly rebuild frozen theory")
        if len(_encode_registry_v2(self)) > MAXIMUM_PUBLIC_REGISTRY_BYTES_V2: raise ValueError("V2 registry byte cap exceeded")

    @property
    def registry_id(self) -> str:
        self._validate()
        return "phase2b_public_recognizer_registry_v2_" + hashlib.sha256(_REGISTRY_DOMAIN_V2 + _encode_registry_v2(self)).hexdigest()

    def to_adapter_registry(self) -> Phase2BAdapterRegistry:
        self._validate()
        return _registry_adapter_unchecked_v2(self)


def _registry_adapter_unchecked_v2(value: PublicRecognizerRegistryV2) -> Phase2BAdapterRegistry:
    return Phase2BAdapterRegistry(
        theory_version_id=value.theory_version_id,
        law_bindings=tuple(sorted((LawWireBinding(law_id=law.law_id, law_kind=law.law_kind, family_id=law.bridge_family_id, role_ids=law.role_ids, required_observable_ids=law.required_observable_ids) for law in value.law_bindings), key=lambda item: item.law_id)),
        observable_channels=tuple(ObservableChannelBinding(quantity_id=item.quantity_id, observable_id=item.observable_id) for item in value.observable_channels),
        maximum_candidate_count=value.maximum_candidate_count,
    )


def _registry_mapping_v2(value: PublicRecognizerRegistryV2) -> dict[str, object]:
    return {
        "family_alias_policy_id": value.family_alias_policy_id,
        "law_bindings": [{
            "bridge_family_id": law.bridge_family_id,
            "canonical_family_id": law.canonical_family_id.value,
            "law_id": law.law_id,
            "law_kind": law.law_kind.value,
            "required_observable_ids": list(law.required_observable_ids),
            "role_ids": [list(item) for item in law.role_ids],
        } for law in value.law_bindings],
        "maximum_candidate_count": value.maximum_candidate_count,
        "observable_channels": [{"observable_id": item.observable_id, "quantity_id": item.quantity_id} for item in value.observable_channels],
        "schema_version": value.schema_version,
        "theory_version_id": value.theory_version_id,
    }


def _encode_registry_v2(value: PublicRecognizerRegistryV2) -> bytes:
    if type(value) is not PublicRecognizerRegistryV2: raise TypeError("V2 registry encoder exact type drift")
    return encode_phase2b_jcs_profile_v1(_registry_mapping_v2(value))


def _decode_registry_v2(payload: bytes) -> PublicRecognizerRegistryV2:
    if type(payload) is not bytes or not 1 <= len(payload) <= MAXIMUM_PUBLIC_REGISTRY_BYTES_V2: raise ValueError("V2 registry byte cap drift")
    root = _closed_v2(decode_phase2b_jcs_profile_v1(payload), _REGISTRY_FIELDS_V2, "V2 public registry")
    if root["schema_version"] != PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION or root["family_alias_policy_id"] != PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2: raise ValueError("V2 registry wire discriminator drift")
    raw_laws, raw_channels = root["law_bindings"], root["observable_channels"]
    if type(raw_laws) is not list or len(raw_laws) != len(LawKind): raise TypeError("V2 registry laws wire drift")
    if type(raw_channels) is not list or not 1 <= len(raw_channels) <= MAXIMUM_ARRAY_ENTRIES: raise TypeError("V2 registry channels wire drift")
    laws = []
    for raw in raw_laws:
        row = _closed_v2(raw, _LAW_FIELDS_V2, "V2 public law")
        try: kind = LawKind(row["law_kind"]); canonical = CanonicalFamilyId(row["canonical_family_id"])
        except (TypeError, ValueError) as exc: raise ValueError("V2 public law discriminator drift") from exc
        if canonical is not _CANONICAL_BY_KIND_V2[kind] or row["bridge_family_id"] != _BRIDGE_BY_KIND_V2[kind]: raise ValueError("V2 public family wire drift")
        raw_roles, raw_observables = row["role_ids"], row["required_observable_ids"]
        if type(raw_roles) is not list or not 1 <= len(raw_roles) <= 64 or type(raw_observables) is not list or not 1 <= len(raw_observables) <= 64: raise TypeError("V2 public law arrays wire drift")
        roles = []
        for pair in raw_roles:
            if type(pair) is not list or len(pair) != 2 or any(type(item) is not str for item in pair): raise TypeError("V2 public role wire drift")
            roles.append((pair[0], pair[1]))
        if any(type(item) is not str for item in raw_observables): raise TypeError("V2 observable wire drift")
        laws.append(PublicRecognizerLawBindingV2._issue(_REGISTRY_TOKEN_V2, law_id=row["law_id"], law_kind=kind, role_ids=tuple(roles), required_observable_ids=tuple(raw_observables)))
    channels = []
    for raw in raw_channels:
        row = _closed_v2(raw, _CHANNEL_FIELDS_V2, "V2 public channel")
        channels.append(PublicRecognizerObservableChannelV2._issue(_REGISTRY_TOKEN_V2, quantity_id=row["quantity_id"], observable_id=row["observable_id"]))
    if type(root["maximum_candidate_count"]) is not int: raise TypeError("V2 registry candidate cap type drift")
    registry = PublicRecognizerRegistryV2._issue(_REGISTRY_TOKEN_V2, theory_version_id=root["theory_version_id"], law_bindings=tuple(laws), observable_channels=tuple(channels), maximum_candidate_count=root["maximum_candidate_count"])
    if _encode_registry_v2(registry) != payload: raise ValueError("V2 registry is not canonical accepted JCS")
    return registry


def _registry_scope_v2(registry: PublicRecognizerRegistryV2) -> tuple[frozenset[str], frozenset[str]]:
    registry._validate()
    return (
        frozenset(wire for law in registry.law_bindings for _, wire in law.role_ids),
        frozenset(item.quantity_id for item in registry.observable_channels),
    )


def _validate_registry_authority_scope_v2(registry: PublicRecognizerRegistryV2, authority: PublicTransformEvidenceBundleV2) -> None:
    if type(registry) is not PublicRecognizerRegistryV2 or type(authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError("V2 registry-authority scope exact type drift")
    roles, quantities = _registry_scope_v2(registry); base = authority.base_bundle
    if (
        roles != frozenset(base.role_ids)
        or quantities != frozenset(base.quantity_ids)
        or quantities != frozenset(base.task_target.quantity_ids)
        or frozenset(base.task_target.entity_ids)
        != frozenset(item.entity_id for item in base.entity_candidates)
        or any(
            not item.role_candidate_ids
            or not frozenset(item.role_candidate_ids).issubset(roles)
            for item in base.entity_candidates
        )
    ):
        raise ValueError("V2 registry and authority exact role quantity entity scope disagree")


def _compile_registry_v2(*, source_registry: Phase2BAdapterRegistry, renamings: Mapping[tuple[str, str], str], typed_authority: PublicTransformEvidenceBundleV2) -> PublicRecognizerRegistryV2:
    if type(renamings) is not MappingProxyType: raise TypeError("V2 live renamings must use exact MappingProxyType")
    if len(renamings) > MAXIMUM_UUID_OCCURRENCES: raise ValueError("V2 live renaming entry cap exceeded")
    new_values: set[str] = set()
    for key, new_uuid in renamings.items():
        if type(key) is not tuple or len(key) != 2: raise TypeError("V2 live renaming key must use an exact pair")
        namespace, old_uuid = key
        if type(namespace) is not str or type(old_uuid) is not str or type(new_uuid) is not str:
            raise TypeError("V2 live renaming scalar type drift")
        if namespace not in _batch_v1._NAMESPACES: raise ValueError("V2 live renaming namespace drift")
        _uuid4_v2(old_uuid, "V2 live old UUID")
        _uuid4_v2(new_uuid, "V2 live new UUID")
        if new_uuid in new_values: raise ValueError("V2 live renaming repeats a new UUID")
        new_values.add(new_uuid)
    if type(typed_authority) is not PublicTransformEvidenceBundleV2: raise TypeError("V2 projected authority exact type drift")
    _validate_source_registry_v2(source_registry)
    aliases = set(_BRIDGE_BY_KIND_V2.values())
    source_registry_ids = ({law.family_id for law in source_registry.law_bindings} | {wire for law in source_registry.law_bindings for _, wire in law.role_ids} | {item.quantity_id for item in source_registry.observable_channels})
    source_authority_ids = {old for _, old in renamings}
    output_authority_ids = set(renamings.values())
    if aliases & (source_registry_ids | source_authority_ids | output_authority_ids): raise ValueError("V2 fixed alias collision")
    expected = ({("role_candidate", wire) for law in source_registry.law_bindings for _, wire in law.role_ids} | {("quantity", item.quantity_id) for item in source_registry.observable_channels})
    actual = {key for key in renamings if key[0] in ("role_candidate", "quantity")}
    if actual != expected: raise ValueError("V2 live registry renaming coverage drift")
    law_by_kind = {item.law_kind: item for item in source_registry.law_bindings}
    laws = tuple(PublicRecognizerLawBindingV2._issue(
        _REGISTRY_TOKEN_V2, law_id=law_by_kind[kind].law_id, law_kind=kind,
        role_ids=tuple((semantic, renamings[("role_candidate", wire)]) for semantic, wire in law_by_kind[kind].role_ids),
        required_observable_ids=law_by_kind[kind].required_observable_ids,
    ) for kind in LawKind)
    channels = tuple(PublicRecognizerObservableChannelV2._issue(_REGISTRY_TOKEN_V2, quantity_id=renamings[("quantity", item.quantity_id)], observable_id=item.observable_id) for item in sorted(source_registry.observable_channels, key=lambda item: item.observable_id))
    registry = PublicRecognizerRegistryV2._issue(_REGISTRY_TOKEN_V2, theory_version_id=source_registry.theory_version_id, law_bindings=laws, observable_channels=channels, maximum_candidate_count=source_registry.maximum_candidate_count)
    _validate_registry_authority_scope_v2(registry, typed_authority)
    return registry


@dataclass(frozen=True, slots=True, init=False)
class TrustedRecognizerInputRowV2:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    namespace_audit_id: str
    authority_content_id: str
    transform_result_id: str
    public_registry: PublicRecognizerRegistryV2
    public_registry_id: str
    row_id: str
    def __init__(self, *args: object, **kwargs: object) -> None: raise TypeError("V2 archive rows are privately issued")

    @classmethod
    def _issue(cls, token: object, *, envelope: bytes, public_registry: PublicRecognizerRegistryV2, typed_row: DecodedTrustedEnvelopeV2 | None = None) -> "TrustedRecognizerInputRowV2":
        if token is not _ROW_TOKEN_V2: raise TypeError("V2 row issuer token drift")
        if type(envelope) is not bytes or len(envelope) != ENVELOPE_BYTES: raise TypeError("V2 row issuer envelope exact length drift")
        if type(public_registry) is not PublicRecognizerRegistryV2: raise TypeError("V2 row registry exact type drift")
        public_registry._validate()
        if typed_row is not None and type(typed_row) is not DecodedTrustedEnvelopeV2: raise TypeError("V2 row supplied typed context exact type drift")
        typed = decode_and_replay_typed_trusted_envelope_v2(envelope) if typed_row is None else typed_row
        if typed_row is not None: typed_row._validate()
        if type(typed) is not DecodedTrustedEnvelopeV2 or typed.envelope != envelope: raise ValueError("V2 row typed context drift")
        value = object.__new__(cls)
        frozen = (
            ("envelope", envelope), ("envelope_id", typed.envelope_id),
            ("payload_sha256", typed.payload_sha256), ("padding_sha256", typed.padding_sha256),
            ("namespace_audit_id", typed.namespace_audit_id), ("authority_content_id", typed.authority_content_id),
            ("transform_result_id", typed.transform_result_id), ("public_registry", public_registry),
            ("public_registry_id", public_registry.registry_id),
        )
        for name, item in frozen: object.__setattr__(value, name, item)
        object.__setattr__(value, "row_id", _row_id_v2(value))
        value._validate(typed_row=typed, token=_ROW_TOKEN_V2)
        return value

    def _validate(self, *, typed_row: DecodedTrustedEnvelopeV2 | None = None, token: object | None = None) -> None:
        if type(self) is not TrustedRecognizerInputRowV2: raise TypeError("V2 row exact type drift")
        if (typed_row is None) is not (token is None) or (typed_row is not None and token is not _ROW_TOKEN_V2): raise TypeError("V2 row context token drift")
        if type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_BYTES: raise TypeError("V2 row envelope exact length drift")
        _digest_v2(self.envelope_id, "phase2b_trusted_envelope_v2_", "V2 row envelope ID")
        _hex64_v2(self.payload_sha256, "V2 row payload SHA"); _hex64_v2(self.padding_sha256, "V2 row padding SHA")
        _digest_v2(self.namespace_audit_id, "phase2b_namespace_audit_v2_", "V2 row namespace root")
        _digest_v2(self.authority_content_id, "phase2b_public_transform_evidence_", "V2 row authority root")
        _digest_v2(self.transform_result_id, "phase2b_exact_transform_result_", "V2 row result root")
        _digest_v2(self.public_registry_id, "phase2b_public_recognizer_registry_v2_", "V2 row registry root")
        _digest_v2(self.row_id, "phase2b_recognizer_input_row_v2_", "V2 row ID")
        if type(self.public_registry) is not PublicRecognizerRegistryV2: raise TypeError("V2 row registry type drift")
        self.public_registry._validate()
        if typed_row is not None and type(typed_row) is not DecodedTrustedEnvelopeV2: raise TypeError("V2 row supplied replay type drift")
        typed = decode_and_replay_typed_trusted_envelope_v2(self.envelope) if typed_row is None else typed_row
        if typed_row is not None: typed_row._validate()
        if type(typed) is not DecodedTrustedEnvelopeV2 or typed.envelope != self.envelope: raise ValueError("V2 row typed replay drift")
        roles, quantities = _registry_scope_v2(self.public_registry)
        _validate_registry_authority_scope_v2(self.public_registry, typed.authority)
        authority_uuids = _profile_uuid4_values_v2(encode_typed_transform_authority_profile_v1(typed.authority))
        aliases = set(_BRIDGE_BY_KIND_V2.values())
        if aliases & (authority_uuids | set(roles) | set(quantities)): raise ValueError("V2 fixed alias collides with unlinkable public UUID")
        if (self.envelope_id, self.payload_sha256, self.padding_sha256, self.namespace_audit_id, self.authority_content_id, self.transform_result_id, self.public_registry_id) != (typed.envelope_id, typed.payload_sha256, typed.padding_sha256, typed.namespace_audit_id, typed.authority_content_id, typed.transform_result_id, self.public_registry.registry_id): raise ValueError("V2 row stored roots drift")
        if self.row_id != _row_id_v2(self): raise ValueError("V2 row root drift")


def _row_id_v2(row: TrustedRecognizerInputRowV2) -> str:
    if type(row) is not TrustedRecognizerInputRowV2: raise TypeError("V2 row root exact type drift")
    _digest_v2(row.authority_content_id, "phase2b_public_transform_evidence_", "V2 row root authority ID")
    _digest_v2(row.envelope_id, "phase2b_trusted_envelope_v2_", "V2 row root envelope ID")
    _digest_v2(row.namespace_audit_id, "phase2b_namespace_audit_v2_", "V2 row root namespace ID")
    _hex64_v2(row.padding_sha256, "V2 row root padding SHA")
    _hex64_v2(row.payload_sha256, "V2 row root payload SHA")
    _digest_v2(row.public_registry_id, "phase2b_public_recognizer_registry_v2_", "V2 row root registry ID")
    _digest_v2(row.transform_result_id, "phase2b_exact_transform_result_", "V2 row root transform ID")
    mapping = {name: getattr(row, name) for name in ("authority_content_id", "envelope_id", "namespace_audit_id", "padding_sha256", "payload_sha256", "public_registry_id", "transform_result_id")}
    payload = encode_phase2b_jcs_profile_v1(mapping)
    return "phase2b_recognizer_input_row_v2_" + hashlib.sha256(_ROW_DOMAIN_V2 + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustedRecognizerInputArchiveRejectionV2:
    disposition: RecognizerInputArchiveDispositionV2
    reason: str
    case_count: int
    batch_id: str | None
    archive: None = None
    rows: tuple[()] = ()
    row_ids: tuple[()] = ()
    envelope_ids: tuple[()] = ()
    public_registry_ids: tuple[()] = ()
    authority_content_ids: tuple[()] = ()
    transform_result_ids: tuple[()] = ()

    def __post_init__(self) -> None:
        if type(self) is not TrustedRecognizerInputArchiveRejectionV2 or self.disposition is not RecognizerInputArchiveDispositionV2.ABSTAIN: raise TypeError("V2 rejection identity drift")
        _ascii_v2(self.reason, "V2 rejection reason")
        if type(self.case_count) is not int or self.case_count < 0: raise ValueError("V2 rejection count drift")
        if self.batch_id is not None: raise ValueError("V2 rejection must expose no batch root")
        if self.archive is not None or any(type(item) is not tuple or item != () for item in (self.rows, self.row_ids, self.envelope_ids, self.public_registry_ids, self.authority_content_ids, self.transform_result_ids)): raise ValueError("V2 rejection leaked partial output")


@dataclass(frozen=True, slots=True)
class _ParsedArchiveV2:
    archive_id: str
    metadata: dict[str, object]
    rows: tuple[TrustedRecognizerInputRowV2, ...]


def _metadata_v2(*, batch_id: str, row_count: int) -> dict[str, object]:
    value: dict[str, object] = {
        "archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "batch_id": batch_id,
        "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "row_count": row_count,
        "typed_replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "typed_authority_schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "typed_authority_codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "typed_authority_codec_policy_id": COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "public_registry_schema_id": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
    }
    _validate_metadata_v2(value, row_count=row_count)
    return value


def _validate_metadata_v2(value: object, *, row_count: int) -> dict[str, object]:
    metadata = _closed_v2(value, _METADATA_FIELDS_V2, "V2 archive metadata")
    if any(type(metadata[name]) is not str for name in _METADATA_FIELDS_V2 if name != "row_count"):
        raise TypeError("V2 archive metadata identity values must use exact strings")
    if type(metadata["row_count"]) is not int:
        raise TypeError("V2 archive metadata row count must use exact int")
    _digest_v2(metadata["archive_policy_id"], "phase2b_recognizer_input_archive_policy_v2_", "V2 archive policy ID")
    _digest_v2(metadata["batch_id"], "phase2b_trusted_wire_batch_v2_", "V2 archive batch ID")
    _digest_v2(metadata["batch_policy_id"], "phase2b_trusted_wire_batch_v2_policy_", "V2 archive batch policy ID")
    _digest_v2(metadata["typed_replay_policy_id"], "phase2b_typed_trusted_wire_replay_policy_v2_", "V2 archive typed replay policy ID")
    if metadata != _metadata_identity_v2(metadata["batch_id"], row_count): raise ValueError("V2 archive metadata identity drift")
    return metadata


def _metadata_identity_v2(batch_id: object, row_count: object) -> dict[str, object]:
    _digest_v2(batch_id, "phase2b_trusted_wire_batch_v2_", "V2 archive batch ID")
    if type(row_count) is not int or not 1 <= row_count <= MAXIMUM_BATCH_V2_AUTHORITIES: raise ValueError("V2 metadata row count drift")
    return {
        "archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "batch_id": batch_id, "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "row_count": row_count, "typed_replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "typed_authority_schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "typed_authority_codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "typed_authority_codec_policy_id": COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "public_registry_schema_id": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
    }


def _archive_id_v2(archive: bytes) -> str:
    if type(archive) is not bytes or not ARCHIVE_HEADER_BYTES_V2 <= len(archive) <= MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2: raise ValueError("V2 archive ID byte cap drift")
    return "phase2b_recognizer_input_archive_v2_" + hashlib.sha256(_ARCHIVE_DOMAIN_V2 + archive).hexdigest()


def _encode_archive_v2(*, metadata: dict[str, object], rows: tuple[TrustedRecognizerInputRowV2, ...]) -> bytes:
    if type(rows) is not tuple or not 1 <= len(rows) <= MAXIMUM_BATCH_V2_AUTHORITIES or any(type(row) is not TrustedRecognizerInputRowV2 for row in rows): raise TypeError("V2 archive row shape drift")
    _validate_metadata_v2(metadata, row_count=len(rows))
    metadata_bytes = encode_phase2b_jcs_profile_v1(metadata)
    if not 1 <= len(metadata_bytes) <= MAXIMUM_ARCHIVE_METADATA_BYTES_V2: raise ValueError("V2 metadata byte cap drift")
    total = ARCHIVE_HEADER_BYTES_V2 + len(metadata_bytes); encoded = []
    for row in rows:
        registry = _encode_registry_v2(row.public_registry)
        if not 1 <= len(registry) <= MAXIMUM_PUBLIC_REGISTRY_BYTES_V2: raise ValueError("V2 registry byte cap drift")
        total += 4 + len(registry) + ENVELOPE_BYTES
        if total > MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2: raise ValueError("V2 archive byte cap exceeded")
        encoded.append((registry, row.envelope))
    header = _ARCHIVE_HEADER_V2.pack(ARCHIVE_MAGIC_V2, ARCHIVE_WIRE_VERSION_V2, ARCHIVE_HEADER_BYTES_V2, len(metadata_bytes), len(rows), hashlib.sha256(metadata_bytes).digest())
    parts = [header, metadata_bytes]
    for registry, envelope in encoded: parts.extend((struct.pack(">I", len(registry)), registry, envelope))
    archive = b"".join(parts)
    if len(archive) != total: raise RuntimeError("V2 archive length accounting drift")
    return archive


def _parse_archive_v2(archive: bytes) -> _ParsedArchiveV2:
    if type(archive) is not bytes: raise TypeError("V2 archive input must use exact bytes")
    if not ARCHIVE_HEADER_BYTES_V2 <= len(archive) <= MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2: raise ValueError("V2 archive byte cap drift")
    magic, version, header_bytes, metadata_length, row_count, metadata_digest = _ARCHIVE_HEADER_V2.unpack_from(archive, 0)
    if magic != ARCHIVE_MAGIC_V2 or version != ARCHIVE_WIRE_VERSION_V2 or header_bytes != ARCHIVE_HEADER_BYTES_V2: raise ValueError("V2 archive header discriminator drift")
    if not 1 <= row_count <= MAXIMUM_BATCH_V2_AUTHORITIES or not 1 <= metadata_length <= MAXIMUM_ARCHIVE_METADATA_BYTES_V2: raise ValueError("V2 archive header count cap drift")
    offset = ARCHIVE_HEADER_BYTES_V2; metadata_end = offset + metadata_length
    if metadata_end + row_count * (4 + 1 + ENVELOPE_BYTES) > len(archive): raise ValueError("V2 archive truncated before bounded rows")
    metadata_bytes = archive[offset:metadata_end]
    if hashlib.sha256(metadata_bytes).digest() != metadata_digest: raise ValueError("V2 metadata digest drift")
    metadata = _closed_v2(decode_phase2b_jcs_profile_v1(metadata_bytes), _METADATA_FIELDS_V2, "V2 archive metadata")
    if encode_phase2b_jcs_profile_v1(metadata) != metadata_bytes: raise ValueError("V2 metadata is not canonical accepted JCS")
    _validate_metadata_v2(metadata, row_count=row_count)
    offset = metadata_end; rows = []; unlinkable_seen: set[str] = set(); aliases = set(_BRIDGE_BY_KIND_V2.values())
    for _ in range(row_count):
        if offset + 4 > len(archive): raise ValueError("V2 registry length truncated")
        (registry_length,) = struct.unpack_from(">I", archive, offset); offset += 4
        if not 1 <= registry_length <= MAXIMUM_PUBLIC_REGISTRY_BYTES_V2: raise ValueError("V2 registry length cap drift")
        row_end = offset + registry_length + ENVELOPE_BYTES
        if row_end > len(archive): raise ValueError("V2 archive row truncated")
        registry_bytes = archive[offset:offset + registry_length]; offset += registry_length
        envelope = archive[offset:offset + ENVELOPE_BYTES]; offset += ENVELOPE_BYTES
        registry = _decode_registry_v2(registry_bytes)
        typed = decode_and_replay_typed_trusted_envelope_v2(envelope)
        row = TrustedRecognizerInputRowV2._issue(_ROW_TOKEN_V2, envelope=envelope, public_registry=registry, typed_row=typed)
        roles, quantities = _registry_scope_v2(registry)
        authority_ids = _profile_uuid4_values_v2(encode_typed_transform_authority_profile_v1(typed.authority))
        case_unlinkable = (authority_ids | set(roles) | set(quantities)) - aliases
        if aliases & (authority_ids | set(roles) | set(quantities)): raise ValueError("V2 aliases collide with public unlinkable UUIDs")
        if case_unlinkable & unlinkable_seen: raise ValueError("V2 archive repeats cross-row public UUID")
        unlinkable_seen.update(case_unlinkable)
        if len(unlinkable_seen) > MAXIMUM_GLOBAL_SOURCE_UUIDS_V2: raise ValueError("V2 global public UUID cap exceeded")
        rows.append(row)
    if offset != len(archive): raise ValueError("V2 archive trailing bytes")
    frozen = tuple(rows)
    for roots in (tuple(item.row_id for item in frozen), tuple(item.envelope_id for item in frozen), tuple(item.public_registry_id for item in frozen), tuple(item.authority_content_id for item in frozen), tuple(item.transform_result_id for item in frozen)):
        if len(set(roots)) != len(roots): raise ValueError("V2 archive repeats a public root")
    return _ParsedArchiveV2(archive_id=_archive_id_v2(archive), metadata=metadata, rows=frozen)


@dataclass(frozen=True, slots=True, init=False)
class DecodedRecognizerInputArchiveV2:
    disposition: RecognizerInputArchiveDispositionV2
    archive: bytes
    archive_id: str
    archive_version: str
    policy_id: str
    batch_id: str
    batch_policy_id: str
    typed_replay_policy_id: str
    typed_authority_schema_id: str
    typed_authority_codec_version: str
    typed_authority_codec_policy_id: str
    public_registry_schema_id: str
    rows: tuple[TrustedRecognizerInputRowV2, ...]
    row_ids: tuple[str, ...]
    envelope_ids: tuple[str, ...]
    public_registry_ids: tuple[str, ...]
    authority_content_ids: tuple[str, ...]
    transform_result_ids: tuple[str, ...]
    claim_level: str
    structural_archive_verified: bool
    row_bijection_verified: bool
    registry_schema_verified: bool
    registry_authority_exact_scope_verified: bool
    compact_typed_replay_verified: bool
    direct_payload_transform_replay_verified: bool
    cross_row_public_uuid_disjoint_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    recognizer_executed: bool
    prediction_archive_evaluated: bool
    recognizer_capacity_evidence: bool
    c1_exit_evidence: bool
    def __init__(self, *args: object, **kwargs: object) -> None: raise TypeError("V2 decoded archives are privately issued")

    @classmethod
    def _issue(cls, token: object, *, archive: bytes, parsed: _ParsedArchiveV2) -> "DecodedRecognizerInputArchiveV2":
        if token is not _DECODE_TOKEN_V2 or type(parsed) is not _ParsedArchiveV2: raise TypeError("V2 decoded issuer context drift")
        current = _parse_archive_v2(archive)
        metadata, rows = current.metadata, current.rows; value = object.__new__(cls)
        frozen = (
            ("disposition", RecognizerInputArchiveDispositionV2.COMPLETE), ("archive", archive), ("archive_id", current.archive_id),
            ("archive_version", metadata["archive_version"]), ("policy_id", metadata["archive_policy_id"]),
            ("batch_id", metadata["batch_id"]), ("batch_policy_id", metadata["batch_policy_id"]),
            ("typed_replay_policy_id", metadata["typed_replay_policy_id"]), ("typed_authority_schema_id", metadata["typed_authority_schema_id"]),
            ("typed_authority_codec_version", metadata["typed_authority_codec_version"]), ("typed_authority_codec_policy_id", metadata["typed_authority_codec_policy_id"]),
            ("public_registry_schema_id", metadata["public_registry_schema_id"]), ("rows", rows),
            ("row_ids", tuple(item.row_id for item in rows)), ("envelope_ids", tuple(item.envelope_id for item in rows)),
            ("public_registry_ids", tuple(item.public_registry_id for item in rows)), ("authority_content_ids", tuple(item.authority_content_id for item in rows)),
            ("transform_result_ids", tuple(item.transform_result_id for item in rows)), ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
        )
        for name, item in frozen: object.__setattr__(value, name, item)
        for name in _TRUE_CLAIMS_V2: object.__setattr__(value, name, True)
        for name in _FALSE_CLAIMS_V2: object.__setattr__(value, name, False)
        value._validate(parsed=current, token=_DECODE_TOKEN_V2)
        return value

    def _validate(self, *, parsed: _ParsedArchiveV2 | None = None, token: object | None = None) -> None:
        if type(self) is not DecodedRecognizerInputArchiveV2: raise TypeError("V2 decoded exact type drift")
        if (parsed is None) is not (token is None) or (parsed is not None and token is not _DECODE_TOKEN_V2): raise TypeError("V2 decoded context token drift")
        if parsed is not None:
            if type(parsed) is not _ParsedArchiveV2 or type(parsed.archive_id) is not str or type(parsed.metadata) is not dict or type(parsed.rows) is not tuple:
                raise TypeError("V2 decoded parsed context shape drift")
        if type(self.archive) is not bytes: raise TypeError("V2 decoded archive bytes type drift")
        if not ARCHIVE_HEADER_BYTES_V2 <= len(self.archive) <= MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2:
            raise ValueError("V2 decoded archive byte cap drift")
        if self.disposition is not RecognizerInputArchiveDispositionV2.COMPLETE: raise ValueError("V2 decoded disposition drift")
        claims = tuple(getattr(self, name) for name in (*_TRUE_CLAIMS_V2, *_FALSE_CLAIMS_V2))
        if any(type(item) is not bool for item in claims) or not all(getattr(self, name) for name in _TRUE_CLAIMS_V2) or any(getattr(self, name) for name in _FALSE_CLAIMS_V2): raise ValueError("V2 decoded claim boundary drift")
        _digest_v2(self.archive_id, "phase2b_recognizer_input_archive_v2_", "V2 decoded archive ID")
        _digest_v2(self.batch_id, "phase2b_trusted_wire_batch_v2_", "V2 decoded batch ID")
        identity = (self.archive_version, self.policy_id, self.batch_policy_id, self.typed_replay_policy_id, self.typed_authority_schema_id, self.typed_authority_codec_version, self.typed_authority_codec_policy_id, self.public_registry_schema_id, self.claim_level)
        if any(type(item) is not str for item in identity): raise TypeError("V2 decoded identity exact string drift")
        expected_identity = (TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION, RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2, TRUSTED_WIRE_BATCH_V2_POLICY_ID, TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID, COMPACT_TYPED_AUTHORITY_SCHEMA_ID, COMPACT_TYPED_AUTHORITY_CODEC_VERSION, COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID, PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2, NON_AUTHORITATIVE_CLAIM_LEVEL)
        if identity != expected_identity: raise ValueError("V2 decoded policy identity drift")
        if type(self.rows) is not tuple or not 1 <= len(self.rows) <= MAXIMUM_BATCH_V2_AUTHORITIES or any(type(item) is not TrustedRecognizerInputRowV2 for item in self.rows): raise TypeError("V2 decoded row shape drift")
        columns = (self.row_ids, self.envelope_ids, self.public_registry_ids, self.authority_content_ids, self.transform_result_ids)
        column_specs = (
            (self.row_ids, "phase2b_recognizer_input_row_v2_"),
            (self.envelope_ids, "phase2b_trusted_envelope_v2_"),
            (self.public_registry_ids, "phase2b_public_recognizer_registry_v2_"),
            (self.authority_content_ids, "phase2b_public_transform_evidence_"),
            (self.transform_result_ids, "phase2b_exact_transform_result_"),
        )
        for column, prefix in column_specs:
            if type(column) is not tuple or len(column) != len(self.rows): raise TypeError("V2 decoded root column shape drift")
            for item in column: _digest_v2(item, prefix, "V2 decoded root")
        for row in self.rows: row._validate()
        if parsed is not None:
            _validate_metadata_v2(parsed.metadata, row_count=len(parsed.rows))
            if (
                _archive_id_v2(self.archive) != parsed.archive_id
                or any(type(item) is not TrustedRecognizerInputRowV2 for item in parsed.rows)
                or len(parsed.rows) != len(self.rows)
                or not all(actual is expected for actual, expected in zip(parsed.rows, self.rows))
            ):
                raise ValueError("V2 decoded parsed context does not bind archive bytes")
        current = _parse_archive_v2(self.archive) if parsed is None else parsed
        expected = (tuple(item.row_id for item in current.rows), tuple(item.envelope_id for item in current.rows), tuple(item.public_registry_id for item in current.rows), tuple(item.authority_content_id for item in current.rows), tuple(item.transform_result_id for item in current.rows))
        rows_drift = self.rows != current.rows if parsed is None else False
        if self.archive_id != current.archive_id or rows_drift or columns != expected or self.batch_id != current.metadata["batch_id"]: raise ValueError("V2 decoded public self-replay drift")


def issue_trusted_recognizer_input_archive_v2(*, batch: TrustedWireBatchV2, run_id: bytes, key_sources: TrustedWireKeySourcesV2, source_cases: tuple[TrustedRecognizerSourceCaseV2, ...]) -> DecodedRecognizerInputArchiveV2 | TrustedRecognizerInputArchiveRejectionV2:
    """Issue atomically after one exact V2 private-core replay/build."""
    if type(batch) is not TrustedWireBatchV2: raise TypeError("V2 issuer batch exact type drift")
    if type(run_id) is not bytes: raise TypeError("V2 issuer run ID exact type drift")
    if type(key_sources) is not TrustedWireKeySourcesV2: raise TypeError("V2 issuer key source exact type drift")
    if type(source_cases) is not tuple: raise TypeError("V2 issuer source cases exact tuple drift")
    count = len(source_cases)
    if not 1 <= count <= MAXIMUM_BATCH_V2_AUTHORITIES:
        return _reject_archive_v2("source_case_count_out_of_bounds", count)
    if any(type(item) is not TrustedRecognizerSourceCaseV2 for item in source_cases): raise TypeError("V2 issuer source row exact type drift")
    try:
        key_sources.__post_init__()
        source_uuids: set[str] = set()
        for source_case in source_cases:
            source_case.__post_init__()
            registry = source_case.adapter_registry
            source_uuids.update(law.family_id for law in registry.law_bindings)
            source_uuids.update(wire for law in registry.law_bindings for _, wire in law.role_ids)
            source_uuids.update(item.quantity_id for item in registry.observable_channels)
            _profile_uuid4_values_v2(encode_typed_transform_authority_profile_v1(source_case.authority), values=source_uuids)
        if len(source_uuids) > MAXIMUM_GLOBAL_SOURCE_UUIDS_V2: raise ValueError("V2 global source UUID cap exceeded")
        aliases = set(_BRIDGE_BY_KIND_V2.values())
        if aliases & source_uuids: raise ValueError("V2 aliases collide with global source UUIDs")
        unlinkable_seen: set[str] = set()

        def compile_projection(source_index: int, renamings: Mapping[tuple[str, str], str], typed_authority: PublicTransformEvidenceBundleV2) -> object:
            if type(source_index) is not int or not 0 <= source_index < count: raise ValueError("V2 live source index drift")
            if type(renamings) is not MappingProxyType: raise TypeError("V2 live renamings must be MappingProxyType")
            if type(typed_authority) is not PublicTransformEvidenceBundleV2: raise TypeError("V2 live authority exact type drift")
            registry = _compile_registry_v2(source_registry=source_cases[source_index].adapter_registry, renamings=renamings, typed_authority=typed_authority)
            roles, quantities = _registry_scope_v2(registry)
            authority_ids = _profile_uuid4_values_v2(encode_typed_transform_authority_profile_v1(typed_authority))
            case_public = authority_ids | set(roles) | set(quantities) | aliases
            if case_public & source_uuids: raise ValueError("V2 public UUID collides with global source sidecar")
            unlinkable = case_public - aliases
            if unlinkable & unlinkable_seen: raise ValueError("V2 cross-row unlinkable UUID collision")
            unlinkable_seen.update(unlinkable)
            if len(unlinkable_seen) > MAXIMUM_GLOBAL_SOURCE_UUIDS_V2: raise ValueError("V2 global public UUID cap exceeded")
            return registry

        replay, projections = _replay_typed_trusted_wire_batch_core_v2(
            authorities=tuple(item.authority for item in source_cases), run_id=run_id,
            key_sources=key_sources, expected_batch=batch,
            per_case_projection_compiler=compile_projection,
        )
        if type(replay) is not TypedTrustedWireBatchReplayV2: return _reject_archive_v2("trusted_v2_rebuild_or_projection_abstained", count)
        if type(projections) is not tuple or len(projections) != count or any(type(item) is not PublicRecognizerRegistryV2 for item in projections): raise ValueError("V2 aligned registry projections incomplete")
        if replay.batch is not batch and replay.batch != batch: raise ValueError("V2 replay expected batch identity drift")
        if len(replay.rows) != count: raise ValueError("V2 replay row count drift")
        rows = tuple(TrustedRecognizerInputRowV2._issue(_ROW_TOKEN_V2, envelope=typed.envelope, public_registry=registry, typed_row=typed) for typed, registry in zip(replay.rows, projections, strict=True))
        if tuple(item.envelope_id for item in rows) != batch.envelope_ids or tuple(item.authority_content_id for item in rows) != batch.authority_content_ids or tuple(item.transform_result_id for item in rows) != batch.transform_result_ids: raise ValueError("V2 archive rows drift from expected public batch roots")
        archive = _encode_archive_v2(metadata=_metadata_v2(batch_id=batch.batch_id, row_count=count), rows=rows)
        decoded = decode_public_recognizer_input_archive_v2(archive)
        if decoded.batch_id != batch.batch_id or decoded.rows != rows: raise ValueError("V2 archive public self-revalidation drift")
        return decoded
    except (AttributeError, KeyError, OverflowError, RecursionError, RuntimeError, TypeError, ValueError):
        return _reject_archive_v2("trusted_recognizer_input_archive_v2_failed", count)


def decode_public_recognizer_input_archive_v2(archive: bytes) -> DecodedRecognizerInputArchiveV2:
    """Revalidate one bounded public V2 archive from public bytes only."""
    parsed = _parse_archive_v2(archive)
    return DecodedRecognizerInputArchiveV2._issue(_DECODE_TOKEN_V2, archive=archive, parsed=parsed)


def _reject_archive_v2(reason: str, count: int) -> TrustedRecognizerInputArchiveRejectionV2:
    return TrustedRecognizerInputArchiveRejectionV2(disposition=RecognizerInputArchiveDispositionV2.ABSTAIN, reason=reason, case_count=count, batch_id=None)


def _assert_field_manifests_v2() -> None:
    mapping_actual = (
        tuple(sorted(item.name for item in fields(PublicRecognizerLawBindingV2))),
        tuple(sorted(item.name for item in fields(PublicRecognizerObservableChannelV2))),
        tuple(sorted(item.name for item in fields(PublicRecognizerRegistryV2))),
        tuple(sorted(item.name for item in fields(TrustedRecognizerInputRowV2))),
    )
    if mapping_actual != (_LAW_FIELDS_V2, _CHANNEL_FIELDS_V2, _REGISTRY_FIELDS_V2, _ROW_FIELDS_V2):
        raise RuntimeError("V2 archive mapping field manifest drift")
    receipt_actual = (
        tuple(item.name for item in fields(TrustedRecognizerSourceCaseV2)),
        tuple(item.name for item in fields(TrustedRecognizerInputArchiveRejectionV2)),
        tuple(item.name for item in fields(DecodedRecognizerInputArchiveV2)),
    )
    if receipt_actual != (_SOURCE_FIELDS_V2, _REJECTION_FIELDS_V2, _DECODED_FIELDS_V2):
        raise RuntimeError("V2 archive receipt field manifest drift")


_assert_field_manifests_v2()


__all__ = (
    "ARCHIVE_HEADER_BYTES_V2", "ARCHIVE_MAGIC_V2", "ARCHIVE_WIRE_VERSION_V2",
    "FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2", "MAXIMUM_ARCHIVE_METADATA_BYTES_V2",
    "MAXIMUM_GLOBAL_SOURCE_UUIDS_V2", "MAXIMUM_PUBLIC_REGISTRY_BYTES_V2",
    "MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2", "PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2",
    "PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2", "PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION",
    "RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2", "TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION",
    "DecodedRecognizerInputArchiveV2", "PublicRecognizerLawBindingV2",
    "PublicRecognizerObservableChannelV2", "PublicRecognizerRegistryV2",
    "RecognizerInputArchiveDispositionV2", "TrustedRecognizerInputArchiveRejectionV2",
    "TrustedRecognizerInputRowV2", "TrustedRecognizerSourceCaseV2",
    "decode_public_recognizer_input_archive_v2", "issue_trusted_recognizer_input_archive_v2",
)

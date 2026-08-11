"""Non-authoritative Phase-2B trusted-wire profile mechanics, stage A.

This module deliberately stops before the security-sensitive batch builder.  It
defines a schema-closed accepted-JCS subset, an explicit UUID namespace/path
manifest, and fixed-envelope structural/hash replay.  It does not shuffle cases,
assign HMAC UUIDs, re-sign renamed provenance, use secret padding, or issue a
formal covert-channel audit receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import re
import struct
from typing import Final

from .phase2b_exact_transform_semantics_v1 import (
    EXACT_TRANSFORM_POLICY_ID,
    ExactTransformAtom,
    ExactTransformCompilation,
    PublicTransformEvidenceBundleV2,
    TransformCompilationDisposition,
    run_exact_transform_semantics,
)


TRUSTED_WIRE_PROFILE_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-profile-mechanics/1"
)
TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-payload/1"
)
TRUSTED_WIRE_MANIFEST_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-field-manifest/1"
)
TRUSTED_WIRE_ENVELOPE_VERSION: Final = 1
NON_AUTHORITATIVE_CLAIM_LEVEL: Final = "NON_AUTHORITATIVE_MECHANICS_ONLY"

_ID_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE_PROFILE_ID/V1\x00"
_MANIFEST_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE_MANIFEST_ID/V1\x00"
_PAYLOAD_DOMAIN: Final = b"HEGEL/PHASE2B/TRUSTED_WIRE_PAYLOAD_ID/V1\x00"
_ENVELOPE_DOMAIN: Final = b"HEGEL/PHASE2B/FIXED_ENVELOPE_ID/V1\x00"
_PUBLIC_PADDING_DOMAIN: Final = b"HEGEL/PHASE2B/PUBLIC_TEST_PADDING/V1\x00"

ENVELOPE_MAGIC: Final = b"HGP2BW1\x00"
ENVELOPE_BYTES: Final = 65_536
ENVELOPE_HEADER_BYTES: Final = 80
MINIMUM_PADDING_BYTES: Final = 32
MAXIMUM_PAYLOAD_BYTES: Final = (
    ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES - MINIMUM_PADDING_BYTES
)
_HEADER: Final = struct.Struct(">8sHHI32s32s")

MAXIMUM_PROFILE_DEPTH: Final = 64
MAXIMUM_PROFILE_NODES: Final = 16_384
MAXIMUM_ARRAY_ENTRIES: Final = 4_096
MAXIMUM_ASCII_STRING_BYTES: Final = 2_048
MAXIMUM_UUID_OCCURRENCES: Final = 2_048
MAXIMUM_UNIQUE_UUIDS: Final = 1_024
MAXIMUM_SAFE_INTEGER: Final = (1 << 53) - 1
MAXIMUM_RATIONAL_BIT_LENGTH: Final = 4_096

_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UUID_SHAPED = re.compile(
    r"^(?:urn:uuid:)?(?:"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|\{[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}\})$",
    flags=re.IGNORECASE,
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProfileDisposition(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


def _sha_id(domain: bytes, payload: bytes, prefix: str) -> str:
    return prefix + hashlib.sha256(domain + payload).hexdigest()


def _require_ascii(value: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use the ASCII profile") from exc
    if len(encoded) > MAXIMUM_ASCII_STRING_BYTES:
        raise ValueError(f"{name} exceeds the frozen string budget")
    return value


def _encode_string(value: str) -> str:
    _require_ascii(value, "JCS string")
    pieces = ['"']
    short = {
        '"': r'\"',
        "\\": r"\\",
        "\b": r"\b",
        "\t": r"\t",
        "\n": r"\n",
        "\f": r"\f",
        "\r": r"\r",
    }
    for character in value:
        replacement = short.get(character)
        if replacement is not None:
            pieces.append(replacement)
        elif ord(character) < 0x20:
            pieces.append(f"\\u{ord(character):04x}")
        else:
            pieces.append(character)
    pieces.append('"')
    return "".join(pieces)


def _encode_profile_value(value: object, depth: int = 0) -> str:
    if depth > MAXIMUM_PROFILE_DEPTH:
        raise ValueError("accepted-JCS value exceeds the depth budget")
    exact_type = type(value)
    if exact_type is type(None):
        return "null"
    if exact_type is bool:
        return "true" if value else "false"
    if exact_type is int:
        if abs(value) > MAXIMUM_SAFE_INTEGER:
            raise ValueError("accepted-JCS integer exceeds the safe range")
        return str(value)
    if exact_type is float:
        raise TypeError("raw JSON floats are outside the accepted-JCS profile")
    if exact_type is str:
        return _encode_string(value)
    if exact_type in (tuple, list):
        if len(value) > MAXIMUM_ARRAY_ENTRIES:
            raise ValueError("accepted-JCS array exceeds the entry budget")
        return "[" + ",".join(
            _encode_profile_value(item, depth + 1) for item in value
        ) + "]"
    if exact_type is dict:
        if len(value) > MAXIMUM_ARRAY_ENTRIES:
            raise ValueError("accepted-JCS object exceeds the entry budget")
        for key in value:
            _require_ascii(key, "JCS object key")
        return "{" + ",".join(
            _encode_string(key)
            + ":"
            + _encode_profile_value(value[key], depth + 1)
            for key in sorted(value)
        ) + "}"
    raise TypeError("value is outside the schema-closed accepted-JCS profile")


def _profile_resource_check(value: object) -> None:
    nodes = 0
    array_entries = 0
    string_bytes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAXIMUM_PROFILE_NODES:
            raise ValueError("accepted-JCS value exceeds the node budget")
        if depth > MAXIMUM_PROFILE_DEPTH:
            raise ValueError("accepted-JCS value exceeds the depth budget")
        exact_type = type(current)
        if exact_type is str:
            _require_ascii(current, "accepted-JCS string")
            string_bytes += len(current)
            if string_bytes > MAXIMUM_PROFILE_NODES * 256:
                raise ValueError("accepted-JCS strings exceed the total budget")
        elif exact_type in (tuple, list):
            array_entries += len(current)
            if (
                len(current) > MAXIMUM_ARRAY_ENTRIES
                or array_entries > MAXIMUM_PROFILE_NODES
            ):
                raise ValueError("accepted-JCS arrays exceed the entry budget")
            stack.extend((item, depth + 1) for item in current)
        elif exact_type is dict:
            array_entries += len(current)
            if (
                len(current) > MAXIMUM_ARRAY_ENTRIES
                or array_entries > MAXIMUM_PROFILE_NODES
            ):
                raise ValueError("accepted-JCS objects exceed the entry budget")
            for key, item in current.items():
                _require_ascii(key, "accepted-JCS object key")
                stack.append((item, depth + 1))
        elif exact_type is int:
            if abs(current) > MAXIMUM_SAFE_INTEGER:
                raise ValueError("accepted-JCS integer exceeds the safe range")
        elif exact_type in (type(None), bool):
            pass
        elif exact_type is float:
            raise TypeError("raw JSON floats are outside the accepted-JCS profile")
        else:
            raise TypeError("accepted-JCS value contains an unsupported type")


def encode_phase2b_jcs_profile_v1(value: object) -> bytes:
    """Encode the schema-closed ASCII/safe-integer accepted-JCS subset."""

    _profile_resource_check(value)
    encoded = _encode_profile_value(value).encode("ascii", errors="strict")
    if len(encoded) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("accepted-JCS payload exceeds the fixed-envelope budget")
    return encoded


def _reject_float(_: str) -> object:
    raise ValueError("raw JSON floats are outside the accepted-JCS profile")


def _reject_constant(_: str) -> object:
    raise ValueError("nonfinite JSON constants are forbidden")


def _pairs_to_exact_dict(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("accepted-JCS object repeats a key")
        result[key] = value
    return result


def decode_phase2b_jcs_profile_v1(payload: bytes) -> object:
    """Decode and require byte-for-byte canonical accepted-JCS input."""

    if type(payload) is not bytes:
        raise TypeError("accepted-JCS payload must be exact bytes")
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("accepted-JCS payload exceeds the fixed-envelope budget")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("accepted-JCS payload must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_to_exact_dict,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("accepted-JCS payload is invalid JSON") from exc
    _profile_resource_check(value)
    if encode_phase2b_jcs_profile_v1(value) != payload:
        raise ValueError("accepted-JCS payload is not canonical")
    return value


JCS_PROFILE_ID: Final = _sha_id(
    _ID_DOMAIN,
    encode_phase2b_jcs_profile_v1(
        {
            "ascii_strings_only": True,
            "binary64_representation": "f64be:16-lowercase-hex",
            "integer_range": "abs<=2^53-1",
            "rational_representation": "decimal-string-pair",
            "version": TRUSTED_WIRE_PROFILE_VERSION,
        }
    ),
    "phase2b_jcs_profile_",
)


@dataclass(frozen=True, slots=True)
class NamespacePathRuleV1:
    namespace: str
    json_pointer_pattern: str
    frozen_required_projection: bool
    expected_zero_occurrences: bool = False
    nullable: bool = False

    def __post_init__(self) -> None:
        if type(self) is not NamespacePathRuleV1:
            raise TypeError("namespace path rule must use the exact type")
        _require_ascii(self.namespace, "namespace")
        _require_ascii(self.json_pointer_pattern, "namespace path")
        if not self.json_pointer_pattern.startswith("/authority/"):
            raise ValueError("namespace path must be rooted at /authority")
        if "**" in self.json_pointer_pattern.split("/"):
            raise ValueError("namespace paths cannot use recursive wildcards")
        if type(self.frozen_required_projection) is not bool or type(
            self.expected_zero_occurrences
        ) is not bool or type(self.nullable) is not bool:
            raise TypeError("namespace rule flags must be exact booleans")

    @property
    def rule_id(self) -> str:
        payload = encode_phase2b_jcs_profile_v1(
            {
                "expected_zero_occurrences": self.expected_zero_occurrences,
                "frozen_required_projection": self.frozen_required_projection,
                "json_pointer_pattern": self.json_pointer_pattern,
                "namespace": self.namespace,
                "nullable": self.nullable,
            }
        )
        return _sha_id(_MANIFEST_DOMAIN, payload, "phase2b_namespace_rule_")


_FROZEN_MINIMUM_NAMESPACES: Final = frozenset(
    {
        "bundle",
        "observation",
        "entity",
        "role_candidate",
        "quantity",
        "context",
        "task",
        "scale",
        "aggregate_map",
        "transform",
    }
)
_SCHEMA_EXTENSION_NAMESPACES: Final = frozenset(
    {"source_channel", "clock", "frame", "unit", "component", "quotient_class"}
)


def _rule(
    namespace: str,
    path: str,
    *,
    zero: bool = False,
    nullable: bool = False,
) -> NamespacePathRuleV1:
    return NamespacePathRuleV1(
        namespace=namespace,
        json_pointer_pattern=path,
        frozen_required_projection=namespace in _FROZEN_MINIMUM_NAMESPACES,
        expected_zero_occurrences=zero,
        nullable=nullable,
    )


def _component_ref_rules(prefix: str) -> tuple[NamespacePathRuleV1, ...]:
    return (
        _rule("scale", prefix + "/scale_id"),
        _rule("observation", prefix + "/observation_id"),
        _rule("component", prefix + "/component_id"),
    )


def _sparse_row_rules(prefix: str) -> tuple[NamespacePathRuleV1, ...]:
    return (
        *_component_ref_rules(prefix + "/output_ref"),
        *_component_ref_rules(prefix + "/terms/*/input_ref"),
    )


FIELD_NAMESPACE_RULES: Final = tuple(
    sorted(
        (
            _rule("bundle", "/authority/base_bundle/bundle_id"),
            _rule("entity", "/authority/base_bundle/entity_candidates/*/entity_id"),
            _rule("role_candidate", "/authority/base_bundle/entity_candidates/*/role_candidate_ids/*"),
            _rule("role_candidate", "/authority/base_bundle/role_ids/*"),
            _rule("quantity", "/authority/base_bundle/quantity_ids/*"),
            _rule("observation", "/authority/base_bundle/observations/*/observation_id"),
            _rule("source_channel", "/authority/base_bundle/observations/*/source_channel_id"),
            _rule("entity", "/authority/base_bundle/observations/*/entity_ids/*"),
            _rule("role_candidate", "/authority/base_bundle/observations/*/role_candidate_ids/*"),
            _rule("quantity", "/authority/base_bundle/observations/*/quantity_id"),
            _rule("clock", "/authority/base_bundle/observations/*/temporal_support/clock_id"),
            _rule("frame", "/authority/base_bundle/observations/*/spatial_support/frame_id"),
            _rule("task", "/authority/base_bundle/task_target/task_id"),
            _rule("entity", "/authority/base_bundle/task_target/entity_ids/*"),
            _rule("quantity", "/authority/base_bundle/task_target/quantity_ids/*"),
            _rule("scale", "/authority/base_bundle/aggregation_graph/scale_ids/*"),
            _rule("scale", "/authority/base_bundle/aggregation_graph/root_scale_ids/*"),
            _rule("scale", "/authority/base_bundle/aggregation_graph/edges/*/source_scale_id"),
            _rule("scale", "/authority/base_bundle/aggregation_graph/edges/*/target_scale_id"),
            _rule("transform", "/authority/base_bundle/aggregation_graph/edges/*/transform_id"),
            _rule("transform", "/authority/base_bundle/transform_catalog/*/transform_id"),
            _rule("observation", "/authority/base_bundle/missingness_mask/*"),
            _rule("observation", "/authority/observation_metadata/*/observation_id"),
            _rule("scale", "/authority/observation_metadata/*/scale_id"),
            _rule("component", "/authority/observation_metadata/*/component_ids/*"),
            _rule("unit", "/authority/observation_metadata/*/unit_id", nullable=True),
            _rule("frame", "/authority/observation_metadata/*/coordinate_frame_id", nullable=True),
            _rule("transform", "/authority/transform_contracts/*/transform_id"),
            _rule("scale", "/authority/transform_contracts/*/source_scale_id"),
            _rule("scale", "/authority/transform_contracts/*/target_scale_id"),
            *_component_ref_rules(
                "/authority/transform_contracts/*/input_components/*"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/output_components/*/ref"
            ),
            _rule(
                "unit",
                "/authority/transform_contracts/*/output_components/*/unit_id",
                nullable=True,
            ),
            _rule(
                "frame",
                "/authority/transform_contracts/*/output_components/*/coordinate_frame_id",
                nullable=True,
            ),
            _rule(
                "clock",
                "/authority/transform_contracts/*/output_components/*/temporal_support/clock_id",
            ),
            _rule(
                "frame",
                "/authority/transform_contracts/*/output_components/*/spatial_support/frame_id",
            ),
            _rule(
                "scale",
                "/authority/transform_contracts/*/output_observations/*/scale_id",
            ),
            _rule(
                "observation",
                "/authority/transform_contracts/*/output_observations/*/observation_id",
            ),
            _rule(
                "source_channel",
                "/authority/transform_contracts/*/output_observations/*/source_channel_id",
            ),
            _rule(
                "entity",
                "/authority/transform_contracts/*/output_observations/*/entity_ids/*",
            ),
            _rule(
                "role_candidate",
                "/authority/transform_contracts/*/output_observations/*/role_candidate_ids/*",
            ),
            _rule(
                "quantity",
                "/authority/transform_contracts/*/output_observations/*/quantity_id",
            ),
            _rule(
                "unit",
                "/authority/transform_contracts/*/output_observations/*/unit_id",
                nullable=True,
            ),
            _rule(
                "clock",
                "/authority/transform_contracts/*/output_observations/*/temporal_support/clock_id",
            ),
            _rule(
                "frame",
                "/authority/transform_contracts/*/output_observations/*/spatial_support/frame_id",
            ),
            _rule(
                "observation",
                "/authority/transform_contracts/*/output_observations/*/source_observation_ids/*",
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/output_observations/*/component_refs/*"
            ),
            *_sparse_row_rules(
                "/authority/transform_contracts/*/kernel_rows/*"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/discrete_mappings/*/input_ref"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/discrete_mappings/*/output_ref"
            ),
            _rule(
                "unit",
                "/authority/transform_contracts/*/certificate/source_unit_id",
            ),
            _rule(
                "unit",
                "/authority/transform_contracts/*/certificate/target_unit_id",
            ),
            _rule(
                "frame",
                "/authority/transform_contracts/*/certificate/source_frame_id",
            ),
            _rule(
                "frame",
                "/authority/transform_contracts/*/certificate/target_frame_id",
            ),
            _rule(
                "frame",
                "/authority/transform_contracts/*/certificate/grid_frame_id",
                nullable=True,
            ),
            _rule(
                "quotient_class",
                "/authority/transform_contracts/*/certificate/quotient_class_ids/*",
            ),
            *_sparse_row_rules(
                "/authority/transform_contracts/*/certificate/inverse_rows/*"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/certificate/groups/*/input_refs/*"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/certificate/groups/*/output_refs/*"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/certificate/selected_inputs/*"
            ),
            *_component_ref_rules(
                "/authority/transform_contracts/*/certificate/discarded_inputs/*"
            ),
            *_sparse_row_rules(
                "/authority/transform_contracts/*/certificate/source_commutation_rows/*"
            ),
            *_sparse_row_rules(
                "/authority/transform_contracts/*/certificate/target_commutation_rows/*"
            ),
            _rule("context", "/authority/__absent_context_id", zero=True),
            _rule("aggregate_map", "/authority/__absent_aggregate_map_id", zero=True),
        ),
        key=lambda item: (item.namespace, item.json_pointer_pattern),
    )
)


def _rules_payload() -> object:
    return {
        "rules": [
            {
                "expected_zero_occurrences": item.expected_zero_occurrences,
                "frozen_required_projection": item.frozen_required_projection,
                "json_pointer_pattern": item.json_pointer_pattern,
                "namespace": item.namespace,
                "nullable": item.nullable,
                "rule_id": item.rule_id,
            }
            for item in FIELD_NAMESPACE_RULES
        ],
        "schema_extension_namespaces": sorted(_SCHEMA_EXTENSION_NAMESPACES),
        "version": TRUSTED_WIRE_MANIFEST_VERSION,
    }


FIELD_MANIFEST_ID: Final = _sha_id(
    _MANIFEST_DOMAIN,
    encode_phase2b_jcs_profile_v1(_rules_payload()),
    "phase2b_field_manifest_",
)


def _pattern_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    def visit(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern):
            return path_index == len(path)
        token = pattern[pattern_index]
        if token == "**":
            return visit(pattern_index + 1, path_index) or (
                path_index < len(path) and visit(pattern_index, path_index + 1)
            )
        if path_index == len(path):
            return False
        return (token == "*" or token == path[path_index]) and visit(
            pattern_index + 1,
            path_index + 1,
        )

    return visit(0, 0)


def _rule_pattern(rule: NamespacePathRuleV1) -> tuple[str, ...]:
    return tuple(item for item in rule.json_pointer_pattern.split("/") if item)


@dataclass(frozen=True, slots=True)
class NamespaceOccurrenceV1:
    namespace: str
    json_pointer: str
    public_uuid: str
    rule_id: str

    def __post_init__(self) -> None:
        if type(self) is not NamespaceOccurrenceV1:
            raise TypeError("namespace occurrence must use the exact type")
        _require_ascii(self.namespace, "occurrence namespace")
        _require_ascii(self.json_pointer, "occurrence path")
        if _UUID4.fullmatch(self.public_uuid) is None:
            raise ValueError("namespace occurrence needs a lowercase UUIDv4")
        matching_rules = tuple(
            rule for rule in FIELD_NAMESPACE_RULES if rule.rule_id == self.rule_id
        )
        if len(matching_rules) != 1:
            raise ValueError("namespace occurrence has an unknown rule root")
        rule = matching_rules[0]
        path = tuple(item for item in self.json_pointer.split("/") if item)
        if rule.namespace != self.namespace or not _pattern_matches(
            _rule_pattern(rule), path
        ):
            raise ValueError("namespace occurrence does not match its rule")


@dataclass(frozen=True, slots=True)
class NamespaceFieldAuditV1:
    manifest_id: str
    frozen_minimum_namespaces: tuple[str, ...]
    schema_registry_namespaces: tuple[str, ...]
    zero_occurrence_namespaces: tuple[str, ...]
    occurrences: tuple[NamespaceOccurrenceV1, ...]
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    formal_uuid_namespace_field_audit: bool = False

    def __post_init__(self) -> None:
        if type(self) is not NamespaceFieldAuditV1:
            raise TypeError("namespace audit must use the exact type")
        if self.manifest_id != FIELD_MANIFEST_ID:
            raise ValueError("namespace audit manifest root drift")
        expected_frozen = tuple(sorted(_FROZEN_MINIMUM_NAMESPACES))
        expected_registry = tuple(
            sorted(_FROZEN_MINIMUM_NAMESPACES | _SCHEMA_EXTENSION_NAMESPACES)
        )
        if self.frozen_minimum_namespaces != expected_frozen:
            raise ValueError("frozen minimum namespace set drift")
        if self.schema_registry_namespaces != expected_registry:
            raise ValueError("schema namespace registry drift")
        if self.zero_occurrence_namespaces != ("aggregate_map", "context"):
            raise ValueError("zero-occurrence namespace disclosure drift")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL or type(
            self.formal_uuid_namespace_field_audit
        ) is not bool or self.formal_uuid_namespace_field_audit:
            raise ValueError("namespace mechanics cannot issue formal evidence")
        if type(self.occurrences) is not tuple or any(
            type(item) is not NamespaceOccurrenceV1 for item in self.occurrences
        ):
            raise TypeError("namespace occurrences must use exact immutable rows")
        if self.occurrences != tuple(
            sorted(
                self.occurrences,
                key=lambda item: (
                    item.namespace,
                    item.json_pointer,
                    item.public_uuid,
                ),
            )
        ):
            raise ValueError("namespace occurrences are not canonical")

    @property
    def audit_id(self) -> str:
        payload = encode_phase2b_jcs_profile_v1(
            {
                "claim_level": self.claim_level,
                "formal": self.formal_uuid_namespace_field_audit,
                "manifest_id": self.manifest_id,
                "occurrences": [
                    {
                        "namespace": item.namespace,
                        "path": item.json_pointer,
                        "public_uuid": item.public_uuid,
                        "rule_id": item.rule_id,
                    }
                    for item in self.occurrences
                ],
                "registry": list(self.schema_registry_namespaces),
                "zero": list(self.zero_occurrence_namespaces),
            }
        )
        return _sha_id(_MANIFEST_DOMAIN, payload, "phase2b_namespace_audit_")


def audit_namespace_paths_v1(authority_profile: object) -> NamespaceFieldAuditV1:
    """Audit every UUID-shaped leaf against the explicit path manifest."""

    if type(authority_profile) is not dict:
        raise TypeError("namespace audit requires an exact authority mapping")
    _profile_resource_check(authority_profile)
    occurrences: list[NamespaceOccurrenceV1] = []
    namespace_by_uuid: dict[str, str] = {}
    stack: list[tuple[object, tuple[str, ...]]] = [
        (authority_profile, ("authority",))
    ]
    while stack:
        value, path = stack.pop()
        exact_type = type(value)
        if exact_type is dict:
            for key, item in value.items():
                stack.append((item, (*path, key)))
        elif exact_type is list:
            for index in range(len(value) - 1, -1, -1):
                stack.append((value[index], (*path, str(index))))
        else:
            matches = tuple(
                rule
                for rule in FIELD_NAMESPACE_RULES
                if _pattern_matches(_rule_pattern(rule), path)
            )
            if len(matches) > 1:
                raise ValueError("UUID path is ambiguous in the manifest")
            if matches and value is None:
                if not matches[0].nullable:
                    raise ValueError("required manifested UUID path is null")
                continue
            if matches and (
                exact_type is not str or _UUID4.fullmatch(value) is None
            ):
                raise ValueError("manifested UUID path does not contain UUIDv4")
            if not matches:
                if exact_type is str and _UUID_SHAPED.fullmatch(value) is not None:
                    raise ValueError("UUID path is absent from the manifest")
                continue
            if len(matches) != 1:
                raise ValueError("UUID path is absent from or ambiguous in the manifest")
            rule = matches[0]
            prior = namespace_by_uuid.setdefault(value, rule.namespace)
            if prior != rule.namespace:
                raise ValueError("one UUID cannot alias two namespaces")
            occurrences.append(
                NamespaceOccurrenceV1(
                    namespace=rule.namespace,
                    json_pointer="/" + "/".join(path),
                    public_uuid=value,
                    rule_id=rule.rule_id,
                )
            )
            if len(occurrences) > MAXIMUM_UUID_OCCURRENCES:
                raise ValueError("namespace occurrence budget exceeded")
            if len(namespace_by_uuid) > MAXIMUM_UNIQUE_UUIDS:
                raise ValueError("namespace unique-UUID budget exceeded")

    present = {item.namespace for item in occurrences}
    if present & {"aggregate_map", "context"}:
        raise ValueError("zero-occurrence frozen namespaces unexpectedly appeared")
    return NamespaceFieldAuditV1(
        manifest_id=FIELD_MANIFEST_ID,
        frozen_minimum_namespaces=tuple(sorted(_FROZEN_MINIMUM_NAMESPACES)),
        schema_registry_namespaces=tuple(
            sorted(_FROZEN_MINIMUM_NAMESPACES | _SCHEMA_EXTENSION_NAMESPACES)
        ),
        zero_occurrence_namespaces=("aggregate_map", "context"),
        occurrences=tuple(
            sorted(
                occurrences,
                key=lambda item: (
                    item.namespace,
                    item.json_pointer,
                    item.public_uuid,
                ),
            )
        ),
    )


def _authority_profile_value(value: object) -> object:
    exact_type = type(value)
    if exact_type is ExactTransformAtom:
        if (
            value.numerator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
            or value.denominator.bit_length() > MAXIMUM_RATIONAL_BIT_LENGTH
        ):
            raise ValueError("exact rational exceeds the frozen bit-length budget")
        return {
            "denominator_decimal": str(value.denominator),
            "numerator_decimal": str(value.numerator),
        }
    if exact_type is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite binary64 is outside the wire profile")
        return "f64be:" + struct.pack(">d", value).hex()
    if exact_type in (str, int, bool, type(None)):
        return value
    if isinstance(value, Enum):
        return value.value
    if exact_type is tuple:
        return [_authority_profile_value(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _authority_profile_value(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError("authority contains a value outside the closed profile schema")


@dataclass(frozen=True, slots=True)
class TrustedWireProfilePreflightRejectionV1:
    disposition: ProfileDisposition
    reason: str
    bundle_id: str
    authority_schema_version: str
    profile_version: str = TRUSTED_WIRE_PROFILE_VERSION
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireProfilePreflightRejectionV1:
            raise TypeError("profile rejection must use the exact type")
        if self.disposition is not ProfileDisposition.ABSTAIN:
            raise ValueError("profile rejection must abstain")
        _require_ascii(self.reason, "profile rejection reason")
        _require_ascii(
            self.authority_schema_version,
            "profile rejection authority schema version",
        )
        if _UUID4.fullmatch(self.bundle_id) is None:
            raise ValueError("profile rejection needs the public bundle ID")
        if self.profile_version != TRUSTED_WIRE_PROFILE_VERSION:
            raise ValueError("profile rejection version drift")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL:
            raise ValueError("profile rejection cannot issue authority")


@dataclass(frozen=True, slots=True)
class TrustedWireProfileCompilationV1:
    disposition: ProfileDisposition
    profile_version: str
    jcs_profile_id: str
    field_manifest_id: str
    payload_schema_version: str
    transform_policy_id: str
    payload: bytes
    payload_sha256: str
    payload_id: str
    namespace_audit: NamespaceFieldAuditV1
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    public_id_renaming_applied: bool = False
    global_shuffle_applied: bool = False
    hmac_uuid_assignment_applied: bool = False
    provenance_rebound_to_public_payload: bool = False
    typed_authority_decode_replay_implemented: bool = False
    batch_atomic_builder_implemented: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    trusted_wire_builder_implemented: bool = False
    secret_padding_replay_implemented: bool = False

    def __post_init__(self) -> None:
        if type(self) is not TrustedWireProfileCompilationV1:
            raise TypeError("profile compilation must use the exact type")
        if self.disposition is not ProfileDisposition.COMPLETE:
            raise ValueError("profile compilation must be complete")
        if (
            self.profile_version != TRUSTED_WIRE_PROFILE_VERSION
            or self.jcs_profile_id != JCS_PROFILE_ID
            or self.field_manifest_id != FIELD_MANIFEST_ID
            or self.payload_schema_version != TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION
            or self.transform_policy_id != EXACT_TRANSFORM_POLICY_ID
        ):
            raise ValueError("profile compilation identity drift")
        if type(self.payload) is not bytes:
            raise TypeError("profile payload must use exact bytes")
        decoded = decode_phase2b_jcs_profile_v1(self.payload)
        if type(decoded) is not dict or set(decoded) != {
            "authority",
            "field_manifest_id",
            "jcs_profile_id",
            "schema_version",
        }:
            raise ValueError("profile payload has the wrong closed schema")
        if (
            decoded["schema_version"] != TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION
            or decoded["jcs_profile_id"] != JCS_PROFILE_ID
            or decoded["field_manifest_id"] != FIELD_MANIFEST_ID
        ):
            raise ValueError("profile payload identity drift")
        digest = hashlib.sha256(self.payload).hexdigest()
        if self.payload_sha256 != digest or _HEX64.fullmatch(digest) is None:
            raise ValueError("profile payload SHA-256 drift")
        expected_id = _sha_id(_PAYLOAD_DOMAIN, self.payload, "phase2b_wire_payload_")
        if self.payload_id != expected_id:
            raise ValueError("profile payload ID drift")
        if type(self.namespace_audit) is not NamespaceFieldAuditV1:
            raise TypeError("profile compilation needs exact namespace audit")
        expected_audit = audit_namespace_paths_v1(decoded["authority"])
        if self.namespace_audit != expected_audit:
            raise ValueError("profile namespace audit does not match the payload")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL or any(
            (
                self.public_id_renaming_applied,
                self.global_shuffle_applied,
                self.hmac_uuid_assignment_applied,
                self.provenance_rebound_to_public_payload,
                self.typed_authority_decode_replay_implemented,
                self.batch_atomic_builder_implemented,
                self.origin_authenticated,
                self.formal_covert_audit,
                self.trusted_wire_builder_implemented,
                self.secret_padding_replay_implemented,
            )
        ):
            raise ValueError("stage-A mechanics cannot issue trusted-wire authority")


def compile_transform_authority_profile_mechanics_v1(
    authority: PublicTransformEvidenceBundleV2,
) -> TrustedWireProfileCompilationV1 | TrustedWireProfilePreflightRejectionV1:
    """Compile one unshuffled authority into the non-authoritative wire profile."""

    if type(authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError("wire profile compiler requires the exact V2 authority")
    bundle_id = authority.base_bundle.bundle_id
    try:
        transform_result = run_exact_transform_semantics(authority)
    except (AttributeError, TypeError, ValueError):
        return TrustedWireProfilePreflightRejectionV1(
            ProfileDisposition.ABSTAIN,
            "transform_authority_validation_failed",
            bundle_id,
            authority.schema_version,
        )
    if (
        type(transform_result) is not ExactTransformCompilation
        or transform_result.disposition is not TransformCompilationDisposition.COMPLETE
    ):
        reason = getattr(transform_result, "reason", "transform_not_complete")
        return TrustedWireProfilePreflightRejectionV1(
            ProfileDisposition.ABSTAIN,
            "transform_" + reason,
            bundle_id,
            authority.schema_version,
        )
    try:
        authority_profile = _authority_profile_value(authority)
        namespace_audit = audit_namespace_paths_v1(authority_profile)
        payload_value = {
            "authority": authority_profile,
            "field_manifest_id": FIELD_MANIFEST_ID,
            "jcs_profile_id": JCS_PROFILE_ID,
            "schema_version": TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION,
        }
        payload = encode_phase2b_jcs_profile_v1(payload_value)
    except (AttributeError, TypeError, ValueError) as exc:
        return TrustedWireProfilePreflightRejectionV1(
            ProfileDisposition.ABSTAIN,
            "profile_compilation_failed:" + str(exc),
            bundle_id,
            authority.schema_version,
        )
    return TrustedWireProfileCompilationV1(
        disposition=ProfileDisposition.COMPLETE,
        profile_version=TRUSTED_WIRE_PROFILE_VERSION,
        jcs_profile_id=JCS_PROFILE_ID,
        field_manifest_id=FIELD_MANIFEST_ID,
        payload_schema_version=TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION,
        transform_policy_id=EXACT_TRANSFORM_POLICY_ID,
        payload=payload,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_id=_sha_id(_PAYLOAD_DOMAIN, payload, "phase2b_wire_payload_"),
        namespace_audit=namespace_audit,
    )


def _public_test_padding(payload_sha256: bytes, length: int) -> bytes:
    pieces: list[bytes] = []
    counter = 0
    remaining = length
    while remaining:
        block = hashlib.sha256(
            _PUBLIC_PADDING_DOMAIN
            + payload_sha256
            + counter.to_bytes(4, byteorder="big", signed=False)
        ).digest()
        pieces.append(block[:remaining])
        remaining -= min(remaining, len(block))
        counter += 1
    return b"".join(pieces)


def _validated_envelope_parts(
    envelope: bytes,
) -> tuple[
    bytes,
    bytes,
    dict[str, object],
    NamespaceFieldAuditV1,
    str,
    str,
]:
    if type(envelope) is not bytes:
        raise TypeError("fixed envelope must use exact bytes")
    if len(envelope) != ENVELOPE_BYTES:
        raise ValueError("fixed envelope must be exactly 65,536 bytes")
    magic, version, header_bytes, payload_length, payload_hash, padding_hash = (
        _HEADER.unpack(envelope[:ENVELOPE_HEADER_BYTES])
    )
    if magic != ENVELOPE_MAGIC or version != TRUSTED_WIRE_ENVELOPE_VERSION:
        raise ValueError("fixed envelope magic or version drift")
    if header_bytes != ENVELOPE_HEADER_BYTES:
        raise ValueError("fixed envelope header length drift")
    if payload_length > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("fixed envelope payload length exceeds capacity")
    padding_length = ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES - payload_length
    if padding_length < MINIMUM_PADDING_BYTES:
        raise ValueError("fixed envelope padding is too short")
    payload = envelope[
        ENVELOPE_HEADER_BYTES : ENVELOPE_HEADER_BYTES + payload_length
    ]
    padding = envelope[ENVELOPE_HEADER_BYTES + payload_length :]
    if hashlib.sha256(payload).digest() != payload_hash:
        raise ValueError("fixed envelope payload hash drift")
    if hashlib.sha256(padding).digest() != padding_hash:
        raise ValueError("fixed envelope padding hash drift")
    if padding != _public_test_padding(payload_hash, padding_length):
        raise ValueError("fixed envelope public test padding drift")
    decoded = decode_phase2b_jcs_profile_v1(payload)
    if type(decoded) is not dict or set(decoded) != {
        "authority",
        "field_manifest_id",
        "jcs_profile_id",
        "schema_version",
    }:
        raise ValueError("fixed envelope payload schema drift")
    if (
        decoded["schema_version"] != TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION
        or decoded["jcs_profile_id"] != JCS_PROFILE_ID
        or decoded["field_manifest_id"] != FIELD_MANIFEST_ID
    ):
        raise ValueError("fixed envelope payload identity drift")
    namespace_audit = audit_namespace_paths_v1(decoded["authority"])
    envelope_id = _sha_id(_ENVELOPE_DOMAIN, envelope, "phase2b_fixed_envelope_")
    payload_id = _sha_id(_PAYLOAD_DOMAIN, payload, "phase2b_wire_payload_")
    return payload, padding, decoded, namespace_audit, envelope_id, payload_id


@dataclass(frozen=True, slots=True)
class FixedEnvelopeMechanicsV1:
    envelope: bytes
    envelope_id: str
    payload_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit_id: str
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    public_test_padding_used: bool = True
    secret_padding_replay_verified: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    trusted_wire_builder_implemented: bool = False

    def __post_init__(self) -> None:
        if type(self) is not FixedEnvelopeMechanicsV1:
            raise TypeError("fixed envelope mechanics must use the exact type")
        if type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_BYTES:
            raise ValueError("fixed envelope must be exactly 65,536 bytes")
        if self.payload_bytes + self.padding_bytes + ENVELOPE_HEADER_BYTES != (
            ENVELOPE_BYTES
        ) or self.padding_bytes < MINIMUM_PADDING_BYTES:
            raise ValueError("fixed envelope lengths drift")
        payload, padding, _, audit, envelope_id, payload_id = (
            _validated_envelope_parts(self.envelope)
        )
        if (
            self.envelope_id != envelope_id
            or self.payload_id != payload_id
            or self.payload_sha256 != hashlib.sha256(payload).hexdigest()
            or self.padding_sha256 != hashlib.sha256(padding).hexdigest()
            or self.payload_bytes != len(payload)
            or self.padding_bytes != len(padding)
            or self.namespace_audit_id != audit.audit_id
        ):
            raise ValueError("fixed envelope receipt does not replay its bytes")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL or (
            not self.public_test_padding_used
            or self.secret_padding_replay_verified
            or self.origin_authenticated
            or self.formal_covert_audit
            or self.trusted_wire_builder_implemented
        ):
            raise ValueError("fixed-envelope mechanics cannot claim secret authority")


def frame_fixed_envelope_mechanics_v1(
    compilation: TrustedWireProfileCompilationV1,
) -> FixedEnvelopeMechanicsV1:
    """Frame a profile payload using explicitly public test padding."""

    if type(compilation) is not TrustedWireProfileCompilationV1:
        raise TypeError("fixed-envelope framing requires an exact compilation")
    compilation.__post_init__()
    payload = compilation.payload
    if len(payload) > MAXIMUM_PAYLOAD_BYTES:
        raise ValueError("profile payload exceeds fixed-envelope capacity")
    payload_digest = hashlib.sha256(payload).digest()
    padding_length = ENVELOPE_BYTES - ENVELOPE_HEADER_BYTES - len(payload)
    padding = _public_test_padding(payload_digest, padding_length)
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
    envelope_id = _sha_id(_ENVELOPE_DOMAIN, envelope, "phase2b_fixed_envelope_")
    return FixedEnvelopeMechanicsV1(
        envelope=envelope,
        envelope_id=envelope_id,
        payload_id=compilation.payload_id,
        payload_sha256=payload_digest.hex(),
        padding_sha256=padding_digest.hex(),
        payload_bytes=len(payload),
        padding_bytes=len(padding),
        namespace_audit_id=compilation.namespace_audit.audit_id,
    )


@dataclass(frozen=True, slots=True)
class DecodedFixedEnvelopeMechanicsV1:
    envelope: bytes
    envelope_id: str
    payload_id: str
    payload_sha256: str
    padding_sha256: str
    payload_bytes: int
    padding_bytes: int
    namespace_audit: NamespaceFieldAuditV1
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL
    structural_hashes_verified: bool = True
    canonical_profile_verified: bool = True
    public_test_padding_verified: bool = True
    secret_padding_replay_verified: bool = False
    typed_authority_decode_replay_implemented: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    trusted_wire_builder_implemented: bool = False

    def __post_init__(self) -> None:
        if type(self) is not DecodedFixedEnvelopeMechanicsV1:
            raise TypeError("decoded envelope must use the exact type")
        if type(self.namespace_audit) is not NamespaceFieldAuditV1:
            raise TypeError("decoded envelope needs exact namespace audit")
        payload, padding, _, audit, envelope_id, payload_id = (
            _validated_envelope_parts(self.envelope)
        )
        if (
            self.envelope_id != envelope_id
            or self.payload_id != payload_id
            or self.payload_sha256 != hashlib.sha256(payload).hexdigest()
            or self.padding_sha256 != hashlib.sha256(padding).hexdigest()
            or self.payload_bytes != len(payload)
            or self.padding_bytes != len(padding)
            or self.namespace_audit != audit
        ):
            raise ValueError("decoded envelope receipt does not replay its bytes")
        if self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL or not all(
            (
                self.structural_hashes_verified,
                self.canonical_profile_verified,
                self.public_test_padding_verified,
            )
        ) or any(
            (
                self.secret_padding_replay_verified,
                self.typed_authority_decode_replay_implemented,
                self.origin_authenticated,
                self.formal_covert_audit,
                self.trusted_wire_builder_implemented,
            )
        ):
            raise ValueError("decoded mechanics cannot claim secret authority")

    @property
    def payload_mapping(self) -> dict[str, object]:
        payload, _, decoded, _, _, _ = _validated_envelope_parts(self.envelope)
        if not payload:
            raise ValueError("decoded envelope payload cannot be empty")
        return decoded


def decode_and_audit_fixed_envelope_mechanics_v1(
    envelope: bytes,
) -> DecodedFixedEnvelopeMechanicsV1:
    """Verify framing, hashes, canonical profile, and explicit namespace paths."""

    payload, padding, _, namespace_audit, envelope_id, payload_id = (
        _validated_envelope_parts(envelope)
    )
    return DecodedFixedEnvelopeMechanicsV1(
        envelope=envelope,
        envelope_id=envelope_id,
        payload_id=payload_id,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        padding_sha256=hashlib.sha256(padding).hexdigest(),
        payload_bytes=len(payload),
        padding_bytes=len(padding),
        namespace_audit=namespace_audit,
    )


__all__ = (
    "DecodedFixedEnvelopeMechanicsV1",
    "ENVELOPE_BYTES",
    "ENVELOPE_HEADER_BYTES",
    "ENVELOPE_MAGIC",
    "FIELD_MANIFEST_ID",
    "FIELD_NAMESPACE_RULES",
    "FixedEnvelopeMechanicsV1",
    "JCS_PROFILE_ID",
    "MAXIMUM_PAYLOAD_BYTES",
    "NON_AUTHORITATIVE_CLAIM_LEVEL",
    "NamespaceFieldAuditV1",
    "NamespaceOccurrenceV1",
    "NamespacePathRuleV1",
    "ProfileDisposition",
    "TRUSTED_WIRE_ENVELOPE_VERSION",
    "TRUSTED_WIRE_MANIFEST_VERSION",
    "TRUSTED_WIRE_PAYLOAD_SCHEMA_VERSION",
    "TRUSTED_WIRE_PROFILE_VERSION",
    "TrustedWireProfileCompilationV1",
    "TrustedWireProfilePreflightRejectionV1",
    "audit_namespace_paths_v1",
    "compile_transform_authority_profile_mechanics_v1",
    "decode_and_audit_fixed_envelope_mechanics_v1",
    "decode_phase2b_jcs_profile_v1",
    "encode_phase2b_jcs_profile_v1",
    "frame_fixed_envelope_mechanics_v1",
)

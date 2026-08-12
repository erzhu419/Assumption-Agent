"""Lossless compact V2 codec for one exact transform authority.

The compact profile is only a representation of the V1 logical typed-authority
profile.  It has no batch, secret, replay, recognizer, origin, or formal claim.
No caller-selected table, schema, policy, compression algorithm, or seed is
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

from .hashing import stable_hash
from .phase2b_exact_transform_semantics_v1 import (
    PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
    PublicTransformEvidenceBundleV2,
)
from .phase2b_trusted_wire_v1 import (
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    audit_namespace_paths_v1,
)
from .phase2b_wire import PUBLIC_EVIDENCE_SCHEMA_VERSION
from . import phase2b_trusted_wire_v1 as _wire_v1
from . import phase2b_trusted_wire_typed_authority_v1 as _typed_v1


COMPACT_TYPED_AUTHORITY_CODEC_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-typed-authority-codec/2"
)

MAXIMUM_EXPANDED_PROFILE_NODES: Final = _wire_v1.MAXIMUM_PROFILE_NODES
MAXIMUM_EXPANDED_PROFILE_ENTRIES: Final = _wire_v1.MAXIMUM_PROFILE_NODES
MAXIMUM_EXPANDED_PROFILE_DEPTH: Final = _wire_v1.MAXIMUM_PROFILE_DEPTH
MAXIMUM_EXPANDED_CONTAINER_ENTRIES: Final = _wire_v1.MAXIMUM_ARRAY_ENTRIES
MAXIMUM_EXPANDED_ASCII_STRING_BYTES: Final = (
    _wire_v1.MAXIMUM_ASCII_STRING_BYTES
)
MAXIMUM_EXPANDED_TOTAL_STRING_BYTES: Final = (
    _wire_v1.MAXIMUM_PROFILE_NODES * 256
)
MAXIMUM_EXPANDED_UUID_OCCURRENCES: Final = (
    _wire_v1.MAXIMUM_UUID_OCCURRENCES
)
MAXIMUM_EXPANDED_UNIQUE_UUIDS: Final = _wire_v1.MAXIMUM_UNIQUE_UUIDS
MAXIMUM_EXPANDED_RATIONAL_BIT_LENGTH: Final = (
    _wire_v1.MAXIMUM_RATIONAL_BIT_LENGTH
)
MAXIMUM_SAFE_INTEGER: Final = _wire_v1.MAXIMUM_SAFE_INTEGER

MAXIMUM_COMPACT_STRING_TABLE_ENTRIES: Final = (
    MAXIMUM_EXPANDED_PROFILE_NODES + MAXIMUM_EXPANDED_PROFILE_ENTRIES
)
MAXIMUM_COMPACT_PROFILE_NODES: Final = (
    3
    * (MAXIMUM_EXPANDED_PROFILE_NODES + MAXIMUM_EXPANDED_PROFILE_ENTRIES)
    + 4
)
MAXIMUM_COMPACT_PROFILE_ENTRIES: Final = MAXIMUM_COMPACT_PROFILE_NODES
MAXIMUM_COMPACT_PROFILE_DEPTH: Final = MAXIMUM_EXPANDED_PROFILE_DEPTH + 2
MAXIMUM_COMPACT_CONTAINER_ENTRIES: Final = max(
    1 + 2 * MAXIMUM_EXPANDED_CONTAINER_ENTRIES,
    MAXIMUM_COMPACT_STRING_TABLE_ENTRIES,
)
MAXIMUM_COMPACT_STRING_TABLE_BYTES: Final = (
    MAXIMUM_EXPANDED_TOTAL_STRING_BYTES
)
MAXIMUM_COMPACT_TOTAL_STRING_BYTES: Final = (
    MAXIMUM_COMPACT_STRING_TABLE_BYTES
    + 6 * MAXIMUM_EXPANDED_ASCII_STRING_BYTES
)

_EXPECTED_V1_CAPS: Final = (
    16_384,
    64,
    4_096,
    2_048,
    2_048,
    1_024,
    4_096,
    (1 << 53) - 1,
)
if (
    MAXIMUM_EXPANDED_PROFILE_NODES,
    MAXIMUM_EXPANDED_PROFILE_DEPTH,
    MAXIMUM_EXPANDED_CONTAINER_ENTRIES,
    MAXIMUM_EXPANDED_ASCII_STRING_BYTES,
    MAXIMUM_EXPANDED_UUID_OCCURRENCES,
    MAXIMUM_EXPANDED_UNIQUE_UUIDS,
    MAXIMUM_EXPANDED_RATIONAL_BIT_LENGTH,
    MAXIMUM_SAFE_INTEGER,
) != _EXPECTED_V1_CAPS:
    raise RuntimeError("V1 logical or fixed-envelope caps require a V2 revision")

_ROOT_FIELDS: Final = ("body", "codec_version", "schema_id", "strings")
_STRING_TAG: Final = "bare_nonnegative_exact_integer_table_index"
_INTEGER_TAG: Final = 0
_ARRAY_TAG: Final = 1
_OBJECT_TAG: Final = 2
_TAG_GRAMMAR: Final = (
    ("string", _STRING_TAG),
    ("integer", "[0,exact_safe_integer]"),
    ("array", "[1,...compact_values]"),
    (
        "object",
        "[2,key_table_index,value,...] with strictly increasing key indices",
    ),
    ("null", "exact_native_null"),
    ("boolean", "exact_native_boolean"),
)
_UUID4: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_COMPACT_CAP_MANIFEST: Final = (
    ("container_entries", MAXIMUM_COMPACT_CONTAINER_ENTRIES),
    ("depth", MAXIMUM_COMPACT_PROFILE_DEPTH),
    ("nodes", MAXIMUM_COMPACT_PROFILE_NODES),
    (
        "resource_formula",
        (
            "N=expanded_nodes<=16384;E=expanded_entries<=16384;"
            "C=logical_container_nodes;I=logical_integer_leaves;"
            "K=logical_object_entries;S=distinct_string_values_and_keys;"
            "C+I<=N;K<=E;S<=N+E;"
            "body_physical_nodes=N+C+2I+K<=3N+E;"
            "body_physical_entries=E+C+2I+K<=2N+2E;"
            "full_root_adds_S+4_nodes_and_entries;"
            "raw_nodes<=4N+2E+4;raw_entries<=3N+3E+4;"
            "unified_raw_cap=3*(Nmax+Emax)+4=98308;"
            "string_table_cap=Nmax+Emax=32768;"
            "container_cap=max(1+2*expanded_container,string_table_cap)"
            "=32768;depth=expanded_depth+2=66;"
            "raw_string_bytes=table_bytes+6*expanded_ascii_string_bytes"
        ),
    ),
    ("scope", "in_memory_codec_preflight_not_envelope_capacity"),
    ("string_table_bytes", MAXIMUM_COMPACT_STRING_TABLE_BYTES),
    ("string_table_entries", MAXIMUM_COMPACT_STRING_TABLE_ENTRIES),
    ("total_entries", MAXIMUM_COMPACT_PROFILE_ENTRIES),
    ("total_string_bytes", MAXIMUM_COMPACT_TOTAL_STRING_BYTES),
)
_EXPANDED_CAP_MANIFEST: Final = (
    ("ascii_string_bytes", MAXIMUM_EXPANDED_ASCII_STRING_BYTES),
    ("container_entries", MAXIMUM_EXPANDED_CONTAINER_ENTRIES),
    ("depth", MAXIMUM_EXPANDED_PROFILE_DEPTH),
    ("nodes", MAXIMUM_EXPANDED_PROFILE_NODES),
    ("rational_bit_length_v1_oracle", MAXIMUM_EXPANDED_RATIONAL_BIT_LENGTH),
    ("safe_integer", MAXIMUM_SAFE_INTEGER),
    ("total_entries", MAXIMUM_EXPANDED_PROFILE_ENTRIES),
    ("total_string_bytes_per_reference", MAXIMUM_EXPANDED_TOTAL_STRING_BYTES),
    ("unique_uuids", MAXIMUM_EXPANDED_UNIQUE_UUIDS),
    ("uuid_occurrences_per_reference", MAXIMUM_EXPANDED_UUID_OCCURRENCES),
)
_CAP_MANIFEST: Final = (
    ("compact", _COMPACT_CAP_MANIFEST),
    ("expanded", _EXPANDED_CAP_MANIFEST),
)

COMPACT_TYPED_AUTHORITY_SCHEMA_ID: Final = stable_hash(
    {
        "caps": _CAP_MANIFEST,
        "codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "cross_rejection": (
            "v1_root_has_logical_authority_fields_not_compact_root_fields",
            "v2_root_has_compact_root_fields_not_v1_logical_authority_fields",
            "codec_version_and_schema_id_are_exact_discriminators",
        ),
        "logical_oracle": {
            "codec_policy_id": _typed_v1.TYPED_AUTHORITY_CODEC_POLICY_ID,
            "codec_version": _typed_v1.TYPED_AUTHORITY_CODEC_VERSION,
            "public_evidence_schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
            "public_transform_evidence_schema_version": (
                PUBLIC_TRANSFORM_EVIDENCE_SCHEMA_VERSION
            ),
            "schema_id": _typed_v1.TYPED_AUTHORITY_SCHEMA_ID,
        },
        "namespace_and_jcs_bindings": {
            "accepted_jcs_profile_id": JCS_PROFILE_ID,
            "expanded_uuid_field_manifest_id": FIELD_MANIFEST_ID,
        },
        "root": {
            "body": "one_compact_tagged_value_rooted_at_object_tag",
            "closed_fields": _ROOT_FIELDS,
            "codec_version": "exact_current_literal",
            "schema_id": "exact_current_schema_id",
            "strings": (
                "exact_ascii_sorted_unique_all_used_table_no_inline_body_string"
            ),
        },
        "tag_grammar": _TAG_GRAMMAR,
        "type_grammar": (
            "all_dict_list_str_int_bool_null_nodes_use_exact_builtin_types",
            "bare_body_integer_is_only_a_nonnegative_string_table_reference",
            "logical_integer_is_only_the_exact_two_item_tag_zero_form",
            "object_key_references_are_strictly_increasing",
        ),
    },
    prefix="phase2b_trusted_wire_compact_typed_authority_schema_",
)

COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID: Final = stable_hash(
    {
        "canonicality": "decode_then_frozen_encoder_exact_mapping_equality",
        "caps": _CAP_MANIFEST,
        "expanded_namespace_audit": (
            "charge_every_table_dereference_string_bytes_but_charge_uuid_only_"
            "for_logical_value_refs_to_match_v1_then_run_FIELD_manifest_audit_"
            "before_v1_logical_decoder_encoder_oracle"
        ),
        "identity": {
            "accepted_jcs_profile_id": JCS_PROFILE_ID,
            "expanded_uuid_field_manifest_id": FIELD_MANIFEST_ID,
            "future_wire_boundary": (
                "encoded_bytes_and_envelope_capacity_belong_to_payload_schema_v3"
            ),
            "schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
            "version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        },
        "no_caller_knobs": (
            "no_compression_algorithm",
            "no_dictionary_or_table_input",
            "no_seed",
            "no_schema_or_policy_override",
        ),
        "validation_order": (
            "exact_shallow_root_and_compact_raw_caps",
            "closed_root_and_sorted_unique_ascii_table",
            "iterative_compact_grammar_and_expanded_budget_plan",
            "all_table_entries_used",
            "expanded_profile_materialization",
            "expanded_FIELD_manifest_namespace_audit",
            "v1_logical_decode",
            "v1_logical_canonical_reencode_equality",
            "v2_canonical_reencode_exact_mapping_equality",
            "no_content_hash_or_claim_receipt",
        ),
        "v1_logical_codec_policy_id": _typed_v1.TYPED_AUTHORITY_CODEC_POLICY_ID,
        "v1_logical_codec_version": _typed_v1.TYPED_AUTHORITY_CODEC_VERSION,
        "v1_logical_schema_id": _typed_v1.TYPED_AUTHORITY_SCHEMA_ID,
    },
    prefix="phase2b_trusted_wire_compact_typed_authority_codec_policy_",
)


def _ascii(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must use ASCII") from exc
    if len(encoded) > MAXIMUM_EXPANDED_ASCII_STRING_BYTES:
        raise ValueError(f"{name} exceeds the string cap")
    return value


def _safe_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if abs(value) > MAXIMUM_SAFE_INTEGER:
        raise ValueError(f"{name} exceeds the safe-integer cap")
    return value


def _raw_compact_resource_check(root: object) -> None:
    """Bound the physical compact tree before root set/sort operations."""

    nodes = 0
    entries = 0
    string_bytes = 0
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > MAXIMUM_COMPACT_PROFILE_NODES:
            raise ValueError("compact typed authority exceeds the raw node cap")
        if depth > MAXIMUM_COMPACT_PROFILE_DEPTH:
            raise ValueError("compact typed authority exceeds the raw depth cap")
        exact_type = type(value)
        if exact_type is dict:
            if len(value) > len(_ROOT_FIELDS):
                raise ValueError("compact typed authority object exceeds the root cap")
            entries += len(value)
            if entries > MAXIMUM_COMPACT_PROFILE_ENTRIES:
                raise ValueError("compact typed authority exceeds the raw entry cap")
            for key, item in value.items():
                text = _ascii(key, "compact typed authority object key")
                string_bytes += len(text)
                stack.append((item, depth + 1))
        elif exact_type is list:
            if len(value) > MAXIMUM_COMPACT_CONTAINER_ENTRIES:
                raise ValueError("compact typed authority container exceeds its cap")
            entries += len(value)
            if entries > MAXIMUM_COMPACT_PROFILE_ENTRIES:
                raise ValueError("compact typed authority exceeds the raw entry cap")
            stack.extend((item, depth + 1) for item in value)
        elif exact_type is str:
            text = _ascii(value, "compact typed authority string")
            string_bytes += len(text)
        elif exact_type is int:
            _safe_integer(value, "compact typed authority integer")
        elif exact_type in (bool, type(None)):
            pass
        else:
            raise TypeError("compact typed authority contains a non-JCS node")
        if string_bytes > MAXIMUM_COMPACT_TOTAL_STRING_BYTES:
            raise ValueError("compact typed authority exceeds the raw string cap")


def _closed_root(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("compact typed authority root must be an exact object")
    if len(value) != len(_ROOT_FIELDS) or tuple(sorted(value)) != _ROOT_FIELDS:
        raise ValueError("compact typed authority root closed schema mismatch")
    if (
        type(value["codec_version"]) is not str
        or value["codec_version"] != COMPACT_TYPED_AUTHORITY_CODEC_VERSION
    ):
        raise ValueError("compact typed authority codec version drift")
    _ascii(value["codec_version"], "compact typed authority codec version")
    if (
        type(value["schema_id"]) is not str
        or value["schema_id"] != COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    ):
        raise ValueError("compact typed authority schema ID drift")
    _ascii(value["schema_id"], "compact typed authority schema ID")
    return value


def _shallow_closed_root(value: object) -> dict[str, object]:
    """Reject top-level drift before walking any caller-controlled body."""

    if type(value) is not dict:
        raise TypeError("compact typed authority root must be an exact object")
    if len(value) != len(_ROOT_FIELDS):
        raise ValueError("compact typed authority root closed schema mismatch")
    for key in value:
        _ascii(key, "compact typed authority root key")
    if tuple(sorted(value)) != _ROOT_FIELDS:
        raise ValueError("compact typed authority root closed schema mismatch")
    return _closed_root(value)


def _string_table(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError("compact typed authority strings must be an exact array")
    if not value or len(value) > MAXIMUM_COMPACT_STRING_TABLE_ENTRIES:
        raise ValueError("compact typed authority string table count is out of bounds")
    result: list[str] = []
    total_bytes = 0
    prior: str | None = None
    for index, item in enumerate(value):
        text = _ascii(item, f"compact typed authority strings[{index}]")
        if prior is not None and text <= prior:
            raise ValueError(
                "compact typed authority strings must be sorted and unique"
            )
        total_bytes += len(text)
        if total_bytes > MAXIMUM_COMPACT_STRING_TABLE_BYTES:
            raise ValueError("compact typed authority string table exceeds its cap")
        result.append(text)
        prior = text
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _ExpandedBudgetPlan:
    used_table_indices: tuple[bool, ...]
    expanded_nodes: int
    expanded_entries: int
    expanded_string_bytes: int
    expanded_uuid_occurrences: int
    expanded_unique_uuids: int


def _table_reference(
    value: object,
    strings: tuple[str, ...],
    used: list[bool],
    *,
    name: str,
) -> tuple[int, str]:
    index = _safe_integer(value, name)
    if index < 0 or index >= len(strings):
        raise ValueError(f"{name} is outside the string table")
    used[index] = True
    return index, strings[index]


def _plan_expanded_profile(
    body: object,
    strings: tuple[str, ...],
) -> _ExpandedBudgetPlan:
    """Validate compact grammar and simulate V1 resource use before expansion."""

    if (
        type(body) is not list
        or not body
        or type(body[0]) is not int
        or body[0] != _OBJECT_TAG
    ):
        raise ValueError("compact typed authority body must be one tagged object")

    nodes = 0
    entries = 0
    string_bytes = 0
    uuid_occurrences = 0
    unique_uuids: set[str] = set()
    used = [False] * len(strings)
    stack: list[tuple[object, int, str]] = [(body, 0, "body")]
    while stack:
        value, depth, name = stack.pop()
        nodes += 1
        if nodes > MAXIMUM_EXPANDED_PROFILE_NODES:
            raise ValueError("compact typed authority expanded node cap exceeded")
        if depth > MAXIMUM_EXPANDED_PROFILE_DEPTH:
            raise ValueError("compact typed authority expanded depth cap exceeded")

        exact_type = type(value)
        if exact_type in (bool, type(None)):
            continue
        if exact_type is int:
            _, text = _table_reference(value, strings, used, name=name)
            string_bytes += len(text)
            if _UUID4.fullmatch(text) is not None:
                uuid_occurrences += 1
                unique_uuids.add(text)
                if uuid_occurrences > MAXIMUM_EXPANDED_UUID_OCCURRENCES:
                    raise ValueError(
                        "compact typed authority expanded UUID occurrence cap exceeded"
                    )
                if len(unique_uuids) > MAXIMUM_EXPANDED_UNIQUE_UUIDS:
                    raise ValueError(
                        "compact typed authority expanded unique UUID cap exceeded"
                    )
        elif exact_type is list:
            if not value or type(value[0]) is not int:
                raise ValueError(f"{name} has no exact compact tag")
            tag = value[0]
            if tag == _INTEGER_TAG:
                if len(value) != 2:
                    raise ValueError(f"{name} integer tag arity drift")
                _safe_integer(value[1], f"{name} integer")
            elif tag == _ARRAY_TAG:
                count = len(value) - 1
                if count > MAXIMUM_EXPANDED_CONTAINER_ENTRIES:
                    raise ValueError(f"{name} expanded array exceeds its cap")
                entries += count
                for index in range(len(value) - 1, 0, -1):
                    stack.append(
                        (value[index], depth + 1, f"{name}[{index - 1}]")
                    )
            elif tag == _OBJECT_TAG:
                if (len(value) - 1) % 2:
                    raise ValueError(f"{name} object tag arity drift")
                count = (len(value) - 1) // 2
                if count > MAXIMUM_EXPANDED_CONTAINER_ENTRIES:
                    raise ValueError(f"{name} expanded object exceeds its cap")
                entries += count
                prior_index = -1
                for offset in range(1, len(value), 2):
                    key_index, key = _table_reference(
                        value[offset],
                        strings,
                        used,
                        name=f"{name} key index",
                    )
                    if key_index <= prior_index:
                        raise ValueError(
                            f"{name} object key indices must be strictly increasing"
                        )
                    prior_index = key_index
                    # Match the V1 raw-profile resource semantics exactly: object
                    # keys charge repeated bytes but are not nodes or UUID leaves.
                    string_bytes += len(key)
                for offset in range(len(value) - 1, 1, -2):
                    key = strings[value[offset - 1]]
                    stack.append((value[offset], depth + 1, f"{name}.{key}"))
            else:
                raise ValueError(f"{name} uses an unknown compact tag")
        else:
            raise TypeError(
                f"{name} contains an inline string, object, float, or non-exact node"
            )

        if entries > MAXIMUM_EXPANDED_PROFILE_ENTRIES:
            raise ValueError("compact typed authority expanded entry cap exceeded")
        if string_bytes > MAXIMUM_EXPANDED_TOTAL_STRING_BYTES:
            raise ValueError("compact typed authority expanded string cap exceeded")

    if not all(used):
        raise ValueError("compact typed authority string table contains unused rows")
    return _ExpandedBudgetPlan(
        used_table_indices=tuple(used),
        expanded_nodes=nodes,
        expanded_entries=entries,
        expanded_string_bytes=string_bytes,
        expanded_uuid_occurrences=uuid_occurrences,
        expanded_unique_uuids=len(unique_uuids),
    )


def _expand_compact_value(value: object, strings: tuple[str, ...]) -> object:
    exact_type = type(value)
    if exact_type in (bool, type(None)):
        return value
    if exact_type is int:
        return strings[value]
    if exact_type is not list:
        raise RuntimeError("validated compact value type drift")
    tag = value[0]
    if tag == _INTEGER_TAG:
        return value[1]
    if tag == _ARRAY_TAG:
        return [_expand_compact_value(item, strings) for item in value[1:]]
    if tag == _OBJECT_TAG:
        return {
            strings[value[index]]: _expand_compact_value(
                value[index + 1], strings
            )
            for index in range(1, len(value), 2)
        }
    raise RuntimeError("validated compact tag drift")


def _logical_strings(root: object) -> tuple[str, ...]:
    values: set[str] = set()
    stack: list[object] = [root]
    while stack:
        value = stack.pop()
        exact_type = type(value)
        if exact_type is dict:
            for key, item in value.items():
                values.add(_ascii(key, "V1 logical profile object key"))
                stack.append(item)
        elif exact_type is list:
            stack.extend(value)
        elif exact_type is str:
            values.add(_ascii(value, "V1 logical profile string"))
        elif exact_type is int:
            _safe_integer(value, "V1 logical profile integer")
        elif exact_type in (bool, type(None)):
            pass
        else:
            raise TypeError("V1 logical profile contains a non-profile node")
    result = tuple(sorted(values))
    if not result or len(result) > MAXIMUM_COMPACT_STRING_TABLE_ENTRIES:
        raise ValueError("V1 logical profile string table count is out of bounds")
    if sum(len(item) for item in result) > MAXIMUM_COMPACT_STRING_TABLE_BYTES:
        raise ValueError("V1 logical profile string table exceeds its byte cap")
    return result


def _compact_logical_value(value: object, table: dict[str, int]) -> object:
    exact_type = type(value)
    if exact_type in (bool, type(None)):
        return value
    if exact_type is str:
        return table[value]
    if exact_type is int:
        return [_INTEGER_TAG, value]
    if exact_type is list:
        return [_ARRAY_TAG, *(_compact_logical_value(item, table) for item in value)]
    if exact_type is dict:
        pairs = sorted((table[key], item) for key, item in value.items())
        result: list[object] = [_OBJECT_TAG]
        for key_index, item in pairs:
            result.append(key_index)
            result.append(_compact_logical_value(item, table))
        return result
    raise RuntimeError("validated V1 logical profile type drift")


def _compact_logical_profile(logical: dict[str, object]) -> dict[str, object]:
    strings = _logical_strings(logical)
    table = {value: index for index, value in enumerate(strings)}
    return {
        "body": _compact_logical_value(logical, table),
        "codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "strings": list(strings),
    }


def _validate_compact_profile(
    compact_profile: object,
) -> tuple[dict[str, object], tuple[str, ...]]:
    root = _shallow_closed_root(compact_profile)
    _raw_compact_resource_check(root)
    strings = _string_table(root["strings"])
    _plan_expanded_profile(root["body"], strings)
    return root, strings


def encode_typed_transform_authority_profile_v2(
    authority: PublicTransformEvidenceBundleV2,
) -> dict[str, object]:
    """Encode one exact authority into the frozen compact V2 profile."""

    if type(authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError("compact typed authority encoder requires the exact V2 type")
    logical = _typed_v1.encode_typed_transform_authority_profile_v1(authority)
    audit_namespace_paths_v1(logical)
    compact = _compact_logical_profile(logical)
    root, strings = _validate_compact_profile(compact)
    expanded = _expand_compact_value(root["body"], strings)
    if type(expanded) is not dict or expanded != logical:
        raise RuntimeError("compact typed authority encoder lost logical profile data")
    return compact


def decode_typed_transform_authority_profile_v2(
    compact_profile: object,
) -> PublicTransformEvidenceBundleV2:
    """Decode one exact compact V2 profile into the exact authority."""

    root, strings = _validate_compact_profile(compact_profile)
    expanded = _expand_compact_value(root["body"], strings)
    if type(expanded) is not dict:
        raise ValueError("compact typed authority did not expand to one object")
    audit_namespace_paths_v1(expanded)
    authority = _typed_v1.decode_typed_transform_authority_profile_v1(expanded)
    canonical_logical = _typed_v1.encode_typed_transform_authority_profile_v1(
        authority
    )
    if canonical_logical != expanded:
        raise ValueError("compact typed authority logical profile is noncanonical")
    canonical_compact = _compact_logical_profile(canonical_logical)
    _validate_compact_profile(canonical_compact)
    if canonical_compact != root:
        raise ValueError("compact typed authority profile is not canonical")
    return authority


__all__ = (
    "COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID",
    "COMPACT_TYPED_AUTHORITY_CODEC_VERSION",
    "COMPACT_TYPED_AUTHORITY_SCHEMA_ID",
    "MAXIMUM_COMPACT_CONTAINER_ENTRIES",
    "MAXIMUM_COMPACT_PROFILE_DEPTH",
    "MAXIMUM_COMPACT_PROFILE_ENTRIES",
    "MAXIMUM_COMPACT_PROFILE_NODES",
    "MAXIMUM_COMPACT_STRING_TABLE_BYTES",
    "MAXIMUM_COMPACT_STRING_TABLE_ENTRIES",
    "MAXIMUM_EXPANDED_ASCII_STRING_BYTES",
    "MAXIMUM_EXPANDED_CONTAINER_ENTRIES",
    "MAXIMUM_EXPANDED_PROFILE_DEPTH",
    "MAXIMUM_EXPANDED_PROFILE_ENTRIES",
    "MAXIMUM_EXPANDED_PROFILE_NODES",
    "MAXIMUM_EXPANDED_TOTAL_STRING_BYTES",
    "MAXIMUM_EXPANDED_UNIQUE_UUIDS",
    "MAXIMUM_EXPANDED_UUID_OCCURRENCES",
    "MAXIMUM_SAFE_INTEGER",
    "decode_typed_transform_authority_profile_v2",
    "encode_typed_transform_authority_profile_v2",
)

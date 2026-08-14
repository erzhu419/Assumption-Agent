"""Rich exact-10 Phase-2B formal trusted-wire V3.

V3 is a generator-facing formal schema, not a lossless wrapper around the
older V2 authority.  In particular it eliminates the V2 extension UUID
namespaces: clock/frame semantics live in ``context_id`` records, unit
semantics are inline SI exponents owned by ``quantity_id``, components use
structured bounded references, and quotient/aggregation semantics are owned by
``aggregate_map_id``.  The only UUID-shaped public leaves belong to the ten
namespaces frozen by the Phase-2B exact protocol.

The builder validates the complete semantic graph, erases source tuple/order
artifacts, globally shuffles cases, and only then assigns HMAC UUIDv4 IDs.  It
emits RFC-8785 canonical JSON in fixed 65,536-byte envelopes.  A separately
returned, non-serialized private alignment binds source rows to shuffled public
envelopes for a trusted pre-release auditor.  No function here generates an
actual corpus, runs a recognizer, scores answers, proves key-source origin, or
claims a covert-channel pass.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import hmac
import math
import re
import struct
import unicodedata
from types import MappingProxyType
from typing import Final

from . import phase2b_exact_transform_core_v1 as _exact_core_v1
from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)


FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-trusted-wire/3"
)
FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-trusted-wire-payload/3"
)
FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-case-graph/3"
)
FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-public-case/3"
)
FORMAL_OPERATION_SEMANTICS_VERSION_V3: Final = (
    "hegel-machine-phase2b-exact-transform-semantics/1-projection-v3"
)
EXACT_TRANSFORM_CORE_SOURCE_SHA256_V3: Final = (
    "sha256:e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f"
)
KEY_SOURCE_TRANSCRIPT_V3_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-key-source-transcript/3"
)
FORMAL_GLOBAL_TRANSFORM_CATALOG_V3_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-global-transform-catalog/3"
)
FORMAL_GLOBAL_TRANSFORM_CATALOG_BINDING_V3_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-global-transform-catalog-binding/3"
)
FORMAL_TRUSTED_WIRE_V3_CLAIM_LEVEL: Final = (
    "FORMAL_WIRE_PUBLIC_STRUCTURE_ONLY_NOT_KEY_TRANSCRIPT_NOT_COVERT_AUDIT_NOT_EXECUTION"
)

ENVELOPE_MAGIC_V3: Final = b"HGP2BW3\x00"
ENVELOPE_VERSION_V3: Final = 3
ENVELOPE_BYTES_V3: Final = 65_536
ENVELOPE_HEADER_BYTES_V3: Final = 80
MINIMUM_PADDING_BYTES_V3: Final = 32
MAXIMUM_PAYLOAD_BYTES_V3: Final = (
    ENVELOPE_BYTES_V3 - ENVELOPE_HEADER_BYTES_V3 - MINIMUM_PADDING_BYTES_V3
)
MAXIMUM_BATCH_CASES_V3: Final = 1_024
MAXIMUM_RECORDS_PER_KIND_V3: Final = 4_096
MAXIMUM_COMPONENTS_PER_OBSERVATION_V3: Final = 256
MAXIMUM_UUID_COLLISION_RETRIES_V3: Final = 10
MAXIMUM_JSON_NESTING_DEPTH_V3: Final = 64
MAXIMUM_JSON_CONTAINER_TOKENS_V3: Final = 16_384
MAXIMUM_SHUFFLE_REJECTION_DRAWS_V3: Final = 4_096
MAXIMUM_TEXT_UTF8_BYTES_V3: Final = 8_192
MAXIMUM_SEMANTIC_DESCRIPTOR_SEGMENTS_V3: Final = 16
MAXIMUM_SEMANTIC_DESCRIPTOR_SEGMENT_UTF8_BYTES_V3: Final = 1_024
MAXIMUM_SEMANTIC_DESCRIPTOR_TOTAL_UTF8_BYTES_V3: Final = 4_096
MAXIMUM_RATIONAL_BITS_V3: Final = 4_096
MAXIMUM_CANDIDATE_COUNT_V3: Final = 50_000
RUN_ID_BYTES_V3: Final = 32
IKM_BYTES_V3: Final = 32

NAMESPACE_REGISTRY_V3: Final = (
    "bundle_id",
    "observation_id",
    "entity_id",
    "role_candidate_id",
    "quantity_id",
    "context_id",
    "task_id",
    "scale_id",
    "aggregate_map_id",
    "transform_id",
)
MAXIMUM_PRIVATE_UUID_MAPPINGS_PER_CASE_V3: Final = (
    len(NAMESPACE_REGISTRY_V3) * MAXIMUM_RECORDS_PER_KIND_V3
)
PUBLIC_OBSERVATION_ARRAY_PATHS_V3: Final = (
    "/public_case/graph/observations",
)
PUBLIC_PAYLOAD_MUST_BE_ABSENT_FIELDS_V3: Final = (
    "answer",
    "case_sequence_index",
    "challenge_label",
    "filename",
    "latent",
    "local_id",
    "order_label",
    "overlap_stratum",
    "private_answer",
    "private_truth",
    "public_order",
    "row_order",
    "source_case_id",
    "source_id",
    "source_index",
    "source_order",
    "transform_catalog",
    "unused_transform_ids",
)

_UUID4_RE: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UUID_SHAPED_RE: Final = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UUID_TOKEN_RE: Final = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_SHA256_TOKEN_RE: Final = re.compile(r"sha256:[0-9a-fA-F]{64}")
_HEX_DIGEST_TOKEN_RE: Final = re.compile(
    r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])"
)
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_DECIMAL_RE: Final = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_UNSIGNED_DECIMAL_RE: Final = re.compile(r"^(?:0|[1-9][0-9]*)$")
_ENVELOPE_HEADER_V3: Final = struct.Struct(">8sHHI32s32s")
if _ENVELOPE_HEADER_V3.size != ENVELOPE_HEADER_BYTES_V3:
    raise RuntimeError("formal trusted-wire V3 header size drift")

_RUN_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/RUN/V3\x00"
_SHUFFLE_INFO_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/HKDF/SHUFFLE/V3\x00"
_ID_INFO_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/HKDF/ID/V3\x00"
_PADDING_INFO_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/HKDF/PADDING/V3\x00"
_SHUFFLE_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/SHUFFLE/V3\x00"
ID_FRAME_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/UUID/V3\x00"
_PADDING_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/PADDING/V3\x00"
_OBS_PROVENANCE_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/OBSERVATION/V3\x00"
_CASE_PROVENANCE_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/CASE/V3\x00"
_SOURCE_CASE_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/SOURCE_CASE/V3\x00"
_ENVELOPE_ID_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/ENVELOPE/V3\x00"
_BATCH_ID_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/BATCH/V3\x00"
_KEY_RECEIPT_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/KEY_RECEIPT/V3\x00"
_ALIGNMENT_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/ALIGNMENT/V3\x00"
_ALIGNMENT_LINK_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/ALIGNMENT_LINK/V3\x00"
_SOURCE_TRANSFORM_USE_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/SOURCE_TRANSFORM_USE/V3\x00"
_TRANSFORM_CATALOG_ROOT_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/TRANSFORM_CATALOG_ROOT/V3\x00"
_TRANSFORM_CATALOG_BINDING_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/TRANSFORM_CATALOG_BINDING/V3\x00"
_VARIANT_CONTEXT_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/VARIANT_CONTEXT/V3\x00"
_PRIVATE_ROW_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/PRIVATE_ROW/V3\x00"
_BUILD_SUCCESS_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/BUILD_SUCCESS/V3\x00"
_POLICY_DOMAIN_V3: Final = b"HEGEL/PHASE2B/FORMAL_TRUSTED_WIRE/POLICY/V3\x00"

# Frozen Unicode Default_Ignorable_Code_Point ranges used by the semantic
# descriptor profile.  This deliberately includes variation selectors (Mn),
# Hangul fillers (Lo), shorthand-format controls, tags, and supplementary
# variation selectors rather than relying only on general category Cf.
_DEFAULT_IGNORABLE_RANGES_V3: Final = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C),
    (0x115F, 0x1160), (0x17B4, 0x17B5), (0x180B, 0x180E),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F),
    (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
)
UNICODE_DATA_VERSION_V3: Final = "13.0.0"
if unicodedata.unidata_version != UNICODE_DATA_VERSION_V3:
    raise RuntimeError("formal semantic descriptor Unicode database version drift")
UCD13_DERIVED_CORE_PROPERTIES_SHA256_V3: Final = (
    "sha256:a5d45f59b39deaab3c72ce8c1a2e212a5e086dff11b1f9d5bb0e352642e82248"
)
UCD13_PROP_LIST_SHA256_V3: Final = (
    "sha256:485b5a3ed25dbf1f94dfa5a9b69d8b4550ffd0c33045ccc55ccfd7c80b2a40cf"
)
DEFAULT_IGNORABLE_CODEPOINT_COUNT_V3: Final = 4_173
if sum(upper - lower + 1 for lower, upper in _DEFAULT_IGNORABLE_RANGES_V3) != DEFAULT_IGNORABLE_CODEPOINT_COUNT_V3:
    raise RuntimeError("formal semantic descriptor Default_Ignorable cardinality drift")
DEFAULT_IGNORABLE_RANGE_TABLE_SHA256_V3: Final = (
    "sha256:629f6c5f9f20e8322371ff59b61ce34246a9c4afb8fd299756540058dc8a553e"
)
if "sha256:" + hashlib.sha256(
    canonicalize_rfc8785_v1([list(value) for value in _DEFAULT_IGNORABLE_RANGES_V3])
).hexdigest() != DEFAULT_IGNORABLE_RANGE_TABLE_SHA256_V3:
    raise RuntimeError("formal semantic descriptor Default_Ignorable table digest drift")


def _is_default_ignorable_v3(character: str) -> bool:
    point = ord(character)
    return any(lower <= point <= upper for lower, upper in _DEFAULT_IGNORABLE_RANGES_V3)


def _is_unicode_noncharacter_v3(character: str) -> bool:
    point = ord(character)
    return 0xFDD0 <= point <= 0xFDEF or point & 0xFFFF in (0xFFFE, 0xFFFF)


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise ValueError(f"{name} must be an exact {'possibly empty' if empty else 'nonempty'} string")
    try:
        encoded = value.encode("utf-8", errors="strict")
        canonicalize_rfc8785_v1(value)
    except (UnicodeError, RFC8785Error) as exc:
        raise ValueError(f"{name} is outside the RFC-8785 string domain") from exc
    if len(encoded) > MAXIMUM_TEXT_UTF8_BYTES_V3:
        raise ValueError(f"{name} exceeds the UTF-8 byte budget")
    return value


def _public_semantic_text_v3(
    value: object,
    name: str,
    *,
    empty: bool = False,
) -> str:
    """Validate one non-ID public string under the frozen UCD-13 profile.

    Typed UUID, digest, decimal, schema, and content-ID fields never enter this
    validator; their exact field-specific validators own those representations.
    Every free semantic/token string does enter it, so an identifier cannot be
    duplicated inside a longer evidence or metadata string.
    """
    result = _text(value, name, empty=empty)
    if unicodedata.normalize("NFC", result) != result:
        raise ValueError(f"{name} must already be NFC")
    if any(
        unicodedata.category(character)
        in {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
        for character in result
    ):
        raise ValueError(
            f"{name} contains an invisible/private/undefined code point"
        )
    if any(_is_default_ignorable_v3(character) for character in result):
        raise ValueError(f"{name} contains a frozen default-ignorable code point")
    if any(_is_unicode_noncharacter_v3(character) for character in result):
        raise ValueError(f"{name} contains a Unicode noncharacter")
    if (
        _UUID_TOKEN_RE.search(result)
        or _SHA256_TOKEN_RE.search(result)
        or _HEX_DIGEST_TOKEN_RE.search(result)
    ):
        raise ValueError(f"{name} contains an identifier/digest token")
    return result


def _uuid(value: object, name: str) -> str:
    if type(value) is not str or _UUID4_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase UUIDv4")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be sha256:<64-lower-hex>")
    return value


def _prefixed_digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    suffix = value[len(prefix):]
    if len(suffix) != 64 or any(ch not in "0123456789abcdef" for ch in suffix):
        raise ValueError(f"{name} digest drift")
    return value


def _tuple(value: object, name: str, *, nonempty: bool = False) -> tuple[object, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise ValueError(f"{name} must be an exact {'nonempty ' if nonempty else ''}tuple")
    if len(value) > MAXIMUM_RECORDS_PER_KIND_V3:
        raise ValueError(f"{name} exceeds the record cap")
    return value


def _uuid_tuple(value: object, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    rows = _tuple(value, name, nonempty=nonempty)
    if any(type(item) is not str for item in rows):
        raise TypeError(f"{name} contains a non-string ID")
    result = tuple(_uuid(item, name) for item in rows)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be sorted unique")
    return result


@dataclass(frozen=True, order=True, slots=True)
class RationalValueV3:
    numerator: str
    denominator: str

    def __post_init__(self) -> None:
        if type(self.numerator) is not str or _DECIMAL_RE.fullmatch(self.numerator) is None:
            raise ValueError("rational numerator must be canonical decimal")
        if type(self.denominator) is not str or _DECIMAL_RE.fullmatch(self.denominator) is None:
            raise ValueError("rational denominator must be canonical decimal")
        numerator, denominator = int(self.numerator), int(self.denominator)
        if denominator <= 0 or math.gcd(numerator, denominator) != 1:
            raise ValueError("rational must be reduced with positive denominator")
        if max(numerator.bit_length(), denominator.bit_length()) > MAXIMUM_RATIONAL_BITS_V3:
            raise ValueError("rational exceeds the bit budget")


def _fraction_v3(value: RationalValueV3) -> Fraction:
    if type(value) is not RationalValueV3:
        raise TypeError("exact rational conversion type drift")
    value.__post_init__()
    return Fraction(int(value.numerator), int(value.denominator))


_BINDING_TOKEN_RE_V3: Final = re.compile(r"^(?:none|[a-z][a-z0-9_.-]{0,63})$")


def _binding_token_v3(value: object, name: str) -> str:
    _public_semantic_text_v3(value, name)
    if type(value) is not str or _BINDING_TOKEN_RE_V3.fullmatch(value) is None:
        raise ValueError(f"{name} must be a frozen non-UUID ASCII binding token")
    return value


@dataclass(frozen=True, slots=True)
class NullableTextV3:
    available: bool
    value: str | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("nullable text available must be exact bool")
        if self.available:
            _public_semantic_text_v3(self.value, "nullable text", empty=True)
        elif self.value is not None:
            raise ValueError("unavailable nullable text must have value null")


@dataclass(frozen=True, slots=True)
class NullableBooleanV3:
    available: bool
    value: bool | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("nullable Boolean available must be exact bool")
        if self.available and type(self.value) is not bool:
            raise TypeError("available nullable Boolean needs exact bool")
        if not self.available and self.value is not None:
            raise ValueError("unavailable nullable Boolean must have value null")


@dataclass(frozen=True, slots=True)
class NullableRationalVectorV3:
    available: bool
    value: tuple[RationalValueV3, ...] | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("nullable rational-vector available must be exact bool")
        if self.available:
            if type(self.value) is not tuple or not self.value:
                raise ValueError("available rational vector must be nonempty")
            if len(self.value) > MAXIMUM_COMPONENTS_PER_OBSERVATION_V3:
                raise ValueError("rational vector exceeds component cap")
            for item in self.value:
                if type(item) is not RationalValueV3:
                    raise TypeError("rational vector contains wrong type")
                item.__post_init__()
        elif self.value is not None:
            raise ValueError("unavailable rational vector must have value null")


@dataclass(frozen=True, slots=True)
class ExactTemporalSupportV3:
    start: RationalValueV3
    end: RationalValueV3

    def __post_init__(self) -> None:
        if type(self.start) is not RationalValueV3 or type(self.end) is not RationalValueV3:
            raise TypeError("temporal endpoints need exact rationals")
        self.start.__post_init__(); self.end.__post_init__()
        if int(self.start.numerator) * int(self.end.denominator) > int(self.end.numerator) * int(self.start.denominator):
            raise ValueError("temporal start exceeds end")


@dataclass(frozen=True, slots=True)
class NullableTemporalSupportV3:
    available: bool
    value: ExactTemporalSupportV3 | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("nullable temporal available must be bool")
        if self.available:
            if type(self.value) is not ExactTemporalSupportV3:
                raise TypeError("available temporal support needs exact value")
            self.value.__post_init__()
        elif self.value is not None:
            raise ValueError("unavailable temporal support must have value null")


@dataclass(frozen=True, slots=True)
class ExactSpatialSupportV3:
    lower: tuple[RationalValueV3, ...]
    upper: tuple[RationalValueV3, ...]

    def __post_init__(self) -> None:
        if type(self.lower) is not tuple or type(self.upper) is not tuple or not 1 <= len(self.lower) == len(self.upper) <= 4:
            raise ValueError("spatial support needs one to four paired dimensions")
        for lower, upper in zip(self.lower, self.upper, strict=True):
            if type(lower) is not RationalValueV3 or type(upper) is not RationalValueV3:
                raise TypeError("spatial endpoints need exact rationals")
            lower.__post_init__(); upper.__post_init__()
            if int(lower.numerator) * int(upper.denominator) > int(upper.numerator) * int(lower.denominator):
                raise ValueError("spatial lower exceeds upper")


@dataclass(frozen=True, slots=True)
class NullableSpatialSupportV3:
    available: bool
    value: ExactSpatialSupportV3 | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("nullable spatial available must be bool")
        if self.available:
            if type(self.value) is not ExactSpatialSupportV3:
                raise TypeError("available spatial support needs exact value")
            self.value.__post_init__()
        elif self.value is not None:
            raise ValueError("unavailable spatial support must have value null")


@dataclass(frozen=True, slots=True)
class FormalUnitBindingV3:
    semantic_key: str
    scale_to_si: RationalValueV3
    offset_to_si: RationalValueV3

    def __post_init__(self) -> None:
        _binding_token_v3(self.semantic_key, "unit semantic binding")
        if type(self.scale_to_si) is not RationalValueV3 or type(
            self.offset_to_si
        ) is not RationalValueV3:
            raise TypeError("unit binding needs exact rational scale and offset")
        self.scale_to_si.__post_init__()
        self.offset_to_si.__post_init__()
        if _fraction_v3(self.scale_to_si) <= 0:
            raise ValueError("unit binding scale_to_si must be strictly positive")


@dataclass(frozen=True, slots=True)
class FormalEntityRecordV3:
    entity_id: str
    role_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormalQuantityRecordV3:
    quantity_id: str
    semantic_key: str
    si_exponents: tuple[int, int, int, int, int, int, int]
    unit_binding: FormalUnitBindingV3
    value_kind: str


@dataclass(frozen=True, slots=True)
class FormalContextRecordV3:
    context_id: str
    temporal_reference_kind: str
    clock_binding: str
    spatial_reference_kind: str
    frame_binding: str
    coordinate_labels: tuple[str, ...]
    temporal_support: NullableTemporalSupportV3
    spatial_support: NullableSpatialSupportV3
    orientation: NullableTextV3


@dataclass(frozen=True, slots=True)
class FormalObservationRecordV3:
    observation_id: str
    entity_ids: tuple[str, ...]
    role_candidate_ids: tuple[str, ...]
    quantity_id: str
    context_id: str
    scale_id: str
    components: tuple[FormalComponentDescriptorV3, ...]
    numeric_value: NullableRationalVectorV3
    boolean_value: NullableBooleanV3
    uncertainty_model: str
    uncertainty_radius: NullableRationalVectorV3
    public_provenance_commitment: str


@dataclass(frozen=True, slots=True)
class FormalTaskTargetV3:
    task_id: str
    entity_ids: tuple[str, ...]
    quantity_ids: tuple[str, ...]


SEMANTIC_DESCRIPTOR_KINDS_V3: Final = (
    "natural_language",
    "symbolic",
    "mixed",
)


@dataclass(frozen=True, slots=True)
class FormalSemanticDescriptorV3:
    kind: str
    segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in SEMANTIC_DESCRIPTOR_KINDS_V3:
            raise ValueError("semantic descriptor kind leaves its evidence-carrier enum")
        if type(self.segments) is not tuple or not 1 <= len(
            self.segments
        ) <= MAXIMUM_SEMANTIC_DESCRIPTOR_SEGMENTS_V3:
            raise ValueError("semantic descriptor segment count is outside 1 through 16")
        total = 0
        for segment in self.segments:
            _public_semantic_text_v3(segment, "semantic descriptor segment")
            encoded = segment.encode("utf-8")
            if len(encoded) > MAXIMUM_SEMANTIC_DESCRIPTOR_SEGMENT_UTF8_BYTES_V3:
                raise ValueError("semantic descriptor segment exceeds its UTF-8 byte bound")
            total += len(encoded)
        if total > MAXIMUM_SEMANTIC_DESCRIPTOR_TOTAL_UTF8_BYTES_V3:
            raise ValueError("semantic descriptor exceeds its total UTF-8 byte bound")


@dataclass(frozen=True, slots=True)
class FormalScaleRecordV3:
    scale_id: str
    semantic_level: str
    is_root: bool


@dataclass(frozen=True, order=True, slots=True)
class FormalComponentDescriptorV3:
    label: str
    semantic_axis: str
    value_role: str


@dataclass(frozen=True, order=True, slots=True)
class FormalComponentRefV3:
    observation_id: str
    quantity_id: str
    context_id: str
    scale_id: str
    component_axis: str
    component_index: int


@dataclass(frozen=True, slots=True)
class FormalExactTermV3:
    input_ref: FormalComponentRefV3
    coefficient: RationalValueV3


@dataclass(frozen=True, slots=True)
class FormalExactRowV3:
    output_ref: FormalComponentRefV3
    terms: tuple[FormalExactTermV3, ...]
    offset: RationalValueV3


@dataclass(frozen=True, order=True, slots=True)
class FormalSamplingGridPointV3:
    input_ref: FormalComponentRefV3
    coordinates: tuple[RationalValueV3, ...]


@dataclass(frozen=True, order=True, slots=True)
class FormalDiscreteMappingV3:
    input_ref: FormalComponentRefV3
    output_ref: FormalComponentRefV3


@dataclass(frozen=True, slots=True)
class FormalAggregateGroupV3:
    input_refs: tuple[FormalComponentRefV3, ...]
    output_refs: tuple[FormalComponentRefV3, ...]
    quotient_label: str


@dataclass(frozen=True, slots=True)
class FormalAggregateMapRecordV3:
    aggregate_map_id: str
    source_scale_id: str
    target_scale_id: str
    transform_id: str
    operation: str
    reducer: str
    groups: tuple[FormalAggregateGroupV3, ...]


TRANSFORM_OPERATIONS_V3: Final = (
    "identity", "unit_conversion", "coordinate_affine",
    "temporal_aggregation", "spatial_aggregation", "sampling_resolution",
    "equivalent_split_merge", "coarse_graining",
)
FORMAL_WEIGHTED_MEAN_BASIS_V3: Final = (
    "caller_declared_exact_coefficients_not_duration_or_volume_inference"
)

@dataclass(frozen=True, slots=True)
class FormalIdentityCertificateV3:
    semantics_version: str
    missing_policy: str
    inverse_contract: str


@dataclass(frozen=True, slots=True)
class FormalUnitConversionCertificateV3:
    semantics_version: str
    factor: RationalValueV3
    inverse_factor: RationalValueV3
    missing_policy: str
    orientation: str
    commutation_contract: str


@dataclass(frozen=True, slots=True)
class FormalCoordinateAffineCertificateV3:
    semantics_version: str
    source_frame_binding: str
    target_frame_binding: str
    dimension: int
    inverse_rows: tuple[FormalExactRowV3, ...]
    missing_policy: str
    support_contract: str
    inverse_contract: str


@dataclass(frozen=True, slots=True)
class FormalTemporalAggregationCertificateV3:
    semantics_version: str
    aggregate_map_id: str
    reducer: str
    missing_policy: str
    boundary_policy: str
    support_contract: str
    weighted_mean_basis: str


@dataclass(frozen=True, slots=True)
class FormalSpatialAggregationCertificateV3:
    semantics_version: str
    aggregate_map_id: str
    reducer: str
    missing_policy: str
    boundary_policy: str
    support_contract: str
    weighted_mean_basis: str


@dataclass(frozen=True, slots=True)
class FormalSamplingResolutionCertificateV3:
    semantics_version: str
    axis: str
    selected_inputs: tuple[FormalComponentRefV3, ...]
    discarded_inputs: tuple[FormalComponentRefV3, ...]
    grid_points: tuple[FormalSamplingGridPointV3, ...]
    grid_dimension: int
    grid_frame_binding: str
    missing_policy: str
    boundary_policy: str
    kernel_contract: str


@dataclass(frozen=True, slots=True)
class FormalEquivalentSplitMergeCertificateV3:
    semantics_version: str
    aggregate_map_id: str
    direction: str
    inverse_rows: tuple[FormalExactRowV3, ...]
    missing_policy: str
    equivalence_contract: str


@dataclass(frozen=True, slots=True)
class FormalCoarseGrainingCertificateV3:
    semantics_version: str
    aggregate_map_id: str
    reducer: str
    source_commutation_rows: tuple[FormalExactRowV3, ...]
    target_commutation_rows: tuple[FormalExactRowV3, ...]
    quotient_aggregate_map_id: str
    quotient_labels: tuple[str, ...]
    missing_policy: str
    boundary_policy: str
    commutation_contract: str
    weighted_mean_basis: str


FormalOperationCertificateV3 = (
    FormalIdentityCertificateV3
    | FormalUnitConversionCertificateV3
    | FormalCoordinateAffineCertificateV3
    | FormalTemporalAggregationCertificateV3
    | FormalSpatialAggregationCertificateV3
    | FormalSamplingResolutionCertificateV3
    | FormalEquivalentSplitMergeCertificateV3
    | FormalCoarseGrainingCertificateV3
)

OPERATION_CERTIFICATE_TYPE_V3: Final = {
    "identity": FormalIdentityCertificateV3,
    "unit_conversion": FormalUnitConversionCertificateV3,
    "coordinate_affine": FormalCoordinateAffineCertificateV3,
    "temporal_aggregation": FormalTemporalAggregationCertificateV3,
    "spatial_aggregation": FormalSpatialAggregationCertificateV3,
    "sampling_resolution": FormalSamplingResolutionCertificateV3,
    "equivalent_split_merge": FormalEquivalentSplitMergeCertificateV3,
    "coarse_graining": FormalCoarseGrainingCertificateV3,
}

# Kept as a stable public manifest name for downstream profile binding.  The
# values now name exact typed certificate fields, not free-form clauses.
OPERATION_CLAUSE_MANIFEST_V3: Final = {
    operation: frozenset(item.name for item in fields(certificate_type))
    for operation, certificate_type in OPERATION_CERTIFICATE_TYPE_V3.items()
}
SEMANTIC_CLAUSE_NAMES_V3: Final = tuple(
    sorted({name for names in OPERATION_CLAUSE_MANIFEST_V3.values() for name in names})
)


@dataclass(frozen=True, slots=True)
class FormalTransformContractV3:
    transform_id: str
    operation: str
    source_scale_id: str
    target_scale_id: str
    input_refs: tuple[FormalComponentRefV3, ...]
    output_refs: tuple[FormalComponentRefV3, ...]
    kernel_rows: tuple[FormalExactRowV3, ...]
    discrete_mappings: tuple[FormalDiscreteMappingV3, ...]
    certificate: FormalOperationCertificateV3


@dataclass(frozen=True, slots=True)
class FormalSourceTransformUseV3:
    aggregate_map: FormalAggregateMapRecordV3
    transform: FormalTransformContractV3

    def __post_init__(self) -> None:
        if type(self.aggregate_map) is not FormalAggregateMapRecordV3 or type(
            self.transform
        ) is not FormalTransformContractV3:
            raise TypeError("source transform use needs exact map/contract types")
        if (
            self.aggregate_map.transform_id != self.transform.transform_id
            or self.aggregate_map.operation != self.transform.operation
            or self.aggregate_map.source_scale_id != self.transform.source_scale_id
            or self.aggregate_map.target_scale_id != self.transform.target_scale_id
        ):
            raise ValueError("source transform use map/contract identity mismatch")


@dataclass(frozen=True, slots=True)
class FormalFamilyAliasV3:
    alias: str
    canonical_family: str


@dataclass(frozen=True, slots=True)
class FormalLawBindingV3:
    law_name: str
    family: str
    law_kind: str
    required_observation_ids: tuple[str, ...]
    required_role_candidate_ids: tuple[str, ...]
    required_quantity_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormalObservableChannelV3:
    observation_id: str
    quantity_id: str


@dataclass(frozen=True, slots=True)
class FormalRecognizerRegistryV3:
    theory_policy: str
    family_aliases: tuple[FormalFamilyAliasV3, ...]
    law_bindings: tuple[FormalLawBindingV3, ...]
    observable_channels: tuple[FormalObservableChannelV3, ...]
    max_candidate_count: int


@dataclass(frozen=True, slots=True)
class FormalEligibilityMaskV3:
    observation_ids: tuple[str, ...]
    role_candidate_ids: tuple[str, ...]
    scale_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormalCaseGraphV3:
    schema_version: str
    bundle_id: str
    entities: tuple[FormalEntityRecordV3, ...]
    role_candidate_ids: tuple[str, ...]
    quantities: tuple[FormalQuantityRecordV3, ...]
    contexts: tuple[FormalContextRecordV3, ...]
    observations: tuple[FormalObservationRecordV3, ...]
    task_target: FormalTaskTargetV3
    semantic_descriptor: FormalSemanticDescriptorV3
    scales: tuple[FormalScaleRecordV3, ...]
    aggregate_maps: tuple[FormalAggregateMapRecordV3, ...]
    transforms: tuple[FormalTransformContractV3, ...]
    recognizer_registry: FormalRecognizerRegistryV3
    eligibility_mask: FormalEligibilityMaskV3

    def __post_init__(self) -> None:
        _validate_case_graph_v3(self)


@dataclass(frozen=True, slots=True)
class FormalSourceCaseV3:
    graph: FormalCaseGraphV3 = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.graph) is not FormalCaseGraphV3:
            raise TypeError("formal source case needs exact FormalCaseGraphV3")
        self.graph.__post_init__()


@dataclass(frozen=True, slots=True)
class FormalPublicCaseV3:
    schema_version: str
    graph: FormalCaseGraphV3
    public_provenance_commitment: str

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION:
            raise ValueError("formal public case schema drift")
        if type(self.graph) is not FormalCaseGraphV3:
            raise TypeError("formal public case graph type drift")
        self.graph.__post_init__()
        _digest(self.public_provenance_commitment, "public case provenance")
        _validate_routed_public_strings_v3(self)


@dataclass(frozen=True, order=True, slots=True)
class FormalGlobalTransformCatalogEntryV3:
    operation: str
    exact_source_transform_use_sha256: str

    def __post_init__(self) -> None:
        if self.operation not in TRANSFORM_OPERATIONS_V3:
            raise ValueError("global transform catalog entry operation is unknown")
        _digest(self.exact_source_transform_use_sha256, "global exact source transform-use digest")


@dataclass(frozen=True, slots=True)
class FormalGlobalTransformCatalogV3:
    schema_version: str
    entries: tuple[FormalGlobalTransformCatalogEntryV3, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_GLOBAL_TRANSFORM_CATALOG_V3_SCHEMA_VERSION:
            raise ValueError("global transform catalog schema drift")
        if type(self.entries) is not tuple or not self.entries or any(
            type(entry) is not FormalGlobalTransformCatalogEntryV3 for entry in self.entries
        ):
            raise TypeError("global transform catalog needs exact nonempty entries")
        for entry in self.entries:
            entry.__post_init__()
        if self.entries != tuple(sorted(self.entries)) or len(set(self.entries)) != len(self.entries):
            raise ValueError("global transform catalog entries must be sorted unique")


@dataclass(frozen=True, slots=True)
class FormalGlobalTransformCatalogBindingReceiptV3:
    schema_version: str
    transform_catalog_root: str
    catalog_entry_commitments: tuple[str, ...]
    receipt_id: str

    def __post_init__(self) -> None:
        if self.schema_version != FORMAL_GLOBAL_TRANSFORM_CATALOG_BINDING_V3_SCHEMA_VERSION:
            raise ValueError("global transform catalog binding schema drift")
        _digest(self.transform_catalog_root, "global transform catalog binding root")
        if (
            type(self.catalog_entry_commitments) is not tuple
            or not self.catalog_entry_commitments
            or self.catalog_entry_commitments != tuple(sorted(self.catalog_entry_commitments))
            or len(set(self.catalog_entry_commitments)) != len(self.catalog_entry_commitments)
        ):
            raise ValueError("global transform catalog binding entries must be sorted unique")
        for value in self.catalog_entry_commitments:
            _digest(value, "global transform catalog entry commitment")
        _prefixed_digest(
            self.receipt_id,
            "phase2b_formal_global_transform_catalog_binding_v3_",
            "global transform catalog binding receipt ID",
        )


def _record_ids(records: tuple[object, ...], field_name: str, name: str) -> tuple[str, ...]:
    values = tuple(getattr(item, field_name) for item in records)
    if len(set(values)) != len(values):
        raise ValueError(f"formal graph repeats {name}")
    return values


def _validate_component_ref_v3(
    ref: FormalComponentRefV3,
    observations: dict[str, FormalObservationRecordV3],
) -> None:
    if type(ref) is not FormalComponentRefV3:
        raise TypeError("component reference exact type drift")
    observation = observations.get(_uuid(ref.observation_id, "component observation_id"))
    if observation is None:
        raise ValueError("component reference observation is unresolved")
    if (
        ref.quantity_id != observation.quantity_id
        or ref.context_id != observation.context_id
        or ref.scale_id != observation.scale_id
    ):
        raise ValueError("component reference semantic owner mismatch")
    _public_semantic_text_v3(ref.component_axis, "component axis")
    labels = tuple(component.label for component in observation.components)
    if ref.component_axis not in labels:
        raise ValueError("component reference axis is unresolved")
    if type(ref.component_index) is not int or not 0 <= ref.component_index < len(labels):
        raise ValueError("component reference index is outside observation")
    if labels[ref.component_index] != ref.component_axis:
        raise ValueError("component reference axis/index mismatch")


def _validate_exact_rows_v3(
    rows: tuple[FormalExactRowV3, ...],
    *,
    input_domain: set[FormalComponentRefV3],
    output_domain: set[FormalComponentRefV3],
    observations: dict[str, FormalObservationRecordV3],
    name: str,
) -> None:
    if type(rows) is not tuple:
        raise TypeError(f"{name} rows must use exact tuple")
    outputs: list[FormalComponentRefV3] = []
    for row in rows:
        if type(row) is not FormalExactRowV3 or row.output_ref not in output_domain:
            raise ValueError(f"{name} row output leaves its declared side")
        _validate_component_ref_v3(row.output_ref, observations)
        if type(row.terms) is not tuple or not row.terms:
            raise ValueError(f"{name} row needs nonempty exact terms")
        row.offset.__post_init__()
        inputs: list[FormalComponentRefV3] = []
        for term in row.terms:
            if type(term) is not FormalExactTermV3 or term.input_ref not in input_domain:
                raise ValueError(f"{name} row term leaves its declared side")
            _validate_component_ref_v3(term.input_ref, observations)
            if _fraction_v3(term.coefficient) == 0:
                raise ValueError(f"{name} sparse row cannot encode a zero coefficient")
            inputs.append(term.input_ref)
        if tuple(inputs) != tuple(sorted(inputs)) or len(set(inputs)) != len(inputs):
            raise ValueError(f"{name} row terms must be canonical sorted unique")
        outputs.append(row.output_ref)
    if tuple(outputs) != tuple(sorted(outputs)) or len(set(outputs)) != len(outputs):
        raise ValueError(f"{name} row outputs must be canonical sorted unique")


def _project_transform_to_exact_core_v1(
    *,
    contract: FormalTransformContractV3,
    aggregate: FormalAggregateMapRecordV3,
    observations: dict[str, FormalObservationRecordV3],
    quantities: dict[str, FormalQuantityRecordV3],
    contexts: dict[str, FormalContextRecordV3],
) -> tuple[
    _exact_core_v1.ExactTransformCoreProfileV1,
    tuple[_exact_core_v1.CoreComponentValueV1, ...],
]:
    """Losslessly project one exact-ten V3 edge into the shared pure core."""
    scale_cache: dict[str, _exact_core_v1.CoreScaleRefV1] = {}
    observation_cache: dict[str, _exact_core_v1.CoreObservationRefV1] = {}

    def core_ref(ref: FormalComponentRefV3) -> _exact_core_v1.CoreComponentRefV1:
        return _exact_core_v1.CoreComponentRefV1(
            scale_cache.setdefault(
                ref.scale_id, _exact_core_v1.CoreScaleRefV1(ref.scale_id)
            ),
            observation_cache.setdefault(
                ref.observation_id,
                _exact_core_v1.CoreObservationRefV1(ref.observation_id),
            ),
            ref.component_index,
        )

    def orientation(context: FormalContextRecordV3) -> str | None:
        return context.orientation.value if context.orientation.available else None

    def clock_anchor(context: FormalContextRecordV3) -> str:
        body = {
            "clock_binding": context.clock_binding,
            "temporal_reference_kind": context.temporal_reference_kind,
        }
        return "v3-clock:" + hashlib.sha256(
            canonicalize_rfc8785_v1(body)
        ).hexdigest()

    def frame_anchor(context: FormalContextRecordV3) -> str:
        body = {
            "coordinate_labels": list(context.coordinate_labels),
            "frame_binding": context.frame_binding,
            "spatial_reference_kind": context.spatial_reference_kind,
        }
        return "v3-frame:" + hashlib.sha256(
            canonicalize_rfc8785_v1(body)
        ).hexdigest()

    def core_metadata(
        ref: FormalComponentRefV3,
    ) -> _exact_core_v1.CoreComponentMetadataV1:
        observation = observations[ref.observation_id]
        descriptor = observation.components[ref.component_index]
        quantity = quantities[ref.quantity_id]
        context = contexts[ref.context_id]
        unit = None
        if quantity.value_kind == "numeric":
            unit = _exact_core_v1.CoreUnitBindingV1(
                quantity.unit_binding.semantic_key,
                _fraction_v3(quantity.unit_binding.scale_to_si),
                _fraction_v3(quantity.unit_binding.offset_to_si),
            )
        temporal = None
        if context.temporal_support.available:
            support = context.temporal_support.value
            if type(support) is not ExactTemporalSupportV3:
                raise TypeError("temporal support projection cache drift")
            temporal = _exact_core_v1.CoreTemporalSupportV1(
                _exact_core_v1.CoreClockBindingV1(clock_anchor(context)),
                _fraction_v3(support.start),
                _fraction_v3(support.end),
            )
        spatial = None
        if context.spatial_support.available:
            support = context.spatial_support.value
            if type(support) is not ExactSpatialSupportV3:
                raise TypeError("spatial support projection cache drift")
            spatial = _exact_core_v1.CoreSpatialSupportV1(
                _exact_core_v1.CoreFrameBindingV1(
                    frame_anchor(context), orientation(context)
                ),
                tuple(_fraction_v3(value) for value in support.lower),
                tuple(_fraction_v3(value) for value in support.upper),
            )
        return _exact_core_v1.CoreComponentMetadataV1(
            ref=core_ref(ref),
            component_label=descriptor.label,
            entity_keys=observation.entity_ids,
            role_keys=observation.role_candidate_ids,
            quantity=_exact_core_v1.CoreQuantityMetadataV1(
                quantity.semantic_key, quantity.si_exponents, unit
            ),
            axis=_exact_core_v1.CoreComponentAxisV1(descriptor.semantic_axis),
            value_role=_exact_core_v1.CoreComponentValueRoleV1(
                descriptor.value_role
            ),
            temporal_support=temporal,
            spatial_support=spatial,
        )

    def core_value(
        ref: FormalComponentRefV3,
    ) -> _exact_core_v1.CoreComponentValueV1:
        observation = observations[ref.observation_id]
        metadata = core_metadata(ref)
        if observation.numeric_value.available:
            values = observation.numeric_value.value
            radii = observation.uncertainty_radius.value
            if type(values) is not tuple or type(radii) is not tuple:
                raise TypeError("numeric value/radius projection cache drift")
            return _exact_core_v1.CoreComponentValueV1(
                metadata=metadata,
                value_kind=_exact_core_v1.CoreComponentValueKindV1.NUMERIC,
                numeric_value=_fraction_v3(values[ref.component_index]),
                uncertainty_model=_exact_core_v1.CoreUncertaintyModelV1(
                    observation.uncertainty_model
                ),
                absolute_radius=_fraction_v3(radii[ref.component_index]),
                boolean_value=None,
            )
        if observation.boolean_value.available:
            return _exact_core_v1.CoreComponentValueV1(
                metadata=metadata,
                value_kind=_exact_core_v1.CoreComponentValueKindV1.BOOLEAN,
                numeric_value=None,
                uncertainty_model=_exact_core_v1.CoreUncertaintyModelV1.NOT_APPLICABLE,
                absolute_radius=None,
                boolean_value=observation.boolean_value.value,
            )
        return _exact_core_v1.CoreComponentValueV1(
            metadata=metadata,
            value_kind=_exact_core_v1.CoreComponentValueKindV1.MISSING,
            numeric_value=None,
            uncertainty_model=_exact_core_v1.CoreUncertaintyModelV1.NOT_APPLICABLE,
            absolute_radius=None,
            boolean_value=None,
        )

    def core_row(row: FormalExactRowV3) -> _exact_core_v1.CoreExactRowV1:
        return _exact_core_v1.CoreExactRowV1(
            core_ref(row.output_ref),
            tuple(
                _exact_core_v1.CoreExactTermV1(
                    core_ref(term.input_ref), _fraction_v3(term.coefficient)
                )
                for term in row.terms
            ),
            _fraction_v3(row.offset),
        )

    groups = tuple(
        _exact_core_v1.CoreAggregateGroupV1(
            tuple(core_ref(ref) for ref in group.input_refs),
            tuple(core_ref(ref) for ref in group.output_refs),
            group.quotient_label,
        )
        for group in aggregate.groups
    )

    def output_label_bindings() -> tuple[
        _exact_core_v1.CoreOutputLabelBindingV1, ...
    ]:
        return tuple(
            sorted(
                (
                    _exact_core_v1.CoreOutputLabelBindingV1(
                        tuple(core_ref(ref) for ref in group.input_refs),
                        core_ref(output_ref),
                        observations[output_ref.observation_id]
                        .components[output_ref.component_index]
                        .label,
                        group.quotient_label,
                    )
                    for group in aggregate.groups
                    for output_ref in group.output_refs
                ),
                key=lambda item: item.output_ref,
            )
        )

    certificate = contract.certificate
    missing_policy = _exact_core_v1.CoreMissingPolicyV1(certificate.missing_policy)
    core_certificate: _exact_core_v1.CoreOperationCertificateV1
    if type(certificate) is FormalIdentityCertificateV3:
        if certificate.inverse_contract != "exact_one_to_one_identity":
            raise ValueError("identity inverse-contract literal drift")
        core_certificate = _exact_core_v1.CoreIdentityCertificateV1(missing_policy)
    elif type(certificate) is FormalUnitConversionCertificateV3:
        if certificate.orientation != "target_value_equals_source_value_times_factor" or certificate.commutation_contract != "si_dimension_and_support_preserved":
            raise ValueError("unit-conversion semantic literal drift")
        core_certificate = _exact_core_v1.CoreUnitConversionCertificateV1(
            _fraction_v3(certificate.factor),
            _fraction_v3(certificate.inverse_factor),
            missing_policy,
        )
    elif type(certificate) is FormalCoordinateAffineCertificateV3:
        if certificate.support_contract != "exact_affine_box_enclosure" or certificate.inverse_contract != "two_sided_exact_sparse_affine_inverse":
            raise ValueError("coordinate-affine semantic literal drift")
        source_context = contexts[contract.input_refs[0].context_id]
        target_context = contexts[contract.output_refs[0].context_id]
        if (
            certificate.source_frame_binding != source_context.frame_binding
            or certificate.target_frame_binding != target_context.frame_binding
        ):
            raise ValueError("coordinate-affine certificate frame binding drift")
        core_certificate = _exact_core_v1.CoreCoordinateAffineCertificateV1(
            _exact_core_v1.CoreFrameBindingV1(
                frame_anchor(source_context), orientation(source_context)
            ),
            _exact_core_v1.CoreFrameBindingV1(
                frame_anchor(target_context), orientation(target_context)
            ),
            certificate.dimension,
            tuple(core_row(row) for row in certificate.inverse_rows),
            missing_policy,
        )
    elif type(certificate) is FormalTemporalAggregationCertificateV3:
        if (
            certificate.boundary_policy != "exact_partition"
            or certificate.support_contract
            != "same_clock_half_open_partition_modulo_final_endpoint"
            or certificate.weighted_mean_basis != FORMAL_WEIGHTED_MEAN_BASIS_V3
        ):
            raise ValueError("temporal aggregation semantic literal drift")
        core_certificate = _exact_core_v1.CoreTemporalAggregationCertificateV1(
            _exact_core_v1.CoreReducerV1(certificate.reducer),
            groups,
            missing_policy,
            _exact_core_v1.CoreBoundaryPolicyV1(certificate.boundary_policy),
            _exact_core_v1.CoreWeightedMeanBasisV1(
                certificate.weighted_mean_basis
            ),
        )
    elif type(certificate) is FormalSpatialAggregationCertificateV3:
        if (
            certificate.boundary_policy != "exact_partition"
            or certificate.support_contract
            != "same_frame_interior_disjoint_positive_volume_partition"
            or certificate.weighted_mean_basis != FORMAL_WEIGHTED_MEAN_BASIS_V3
        ):
            raise ValueError("spatial aggregation semantic literal drift")
        core_certificate = _exact_core_v1.CoreSpatialAggregationCertificateV1(
            _exact_core_v1.CoreReducerV1(certificate.reducer),
            groups,
            missing_policy,
            _exact_core_v1.CoreBoundaryPolicyV1(certificate.boundary_policy),
            _exact_core_v1.CoreWeightedMeanBasisV1(
                certificate.weighted_mean_basis
            ),
        )
    elif type(certificate) is FormalSamplingResolutionCertificateV3:
        if certificate.boundary_policy != "valid_only" or certificate.kernel_contract != "single_series_ordered_exact_subselection":
            raise ValueError("sampling semantic literal drift")
        frame_binding = None
        if certificate.grid_frame_binding != "none":
            source_context = contexts[certificate.selected_inputs[0].context_id]
            if certificate.grid_frame_binding != source_context.frame_binding:
                raise ValueError("sampling certificate grid frame binding drift")
            frame_binding = _exact_core_v1.CoreFrameBindingV1(
                frame_anchor(source_context), orientation(source_context)
            )
        core_certificate = _exact_core_v1.CoreSamplingResolutionCertificateV1(
            _exact_core_v1.CoreComponentAxisV1(certificate.axis),
            tuple(core_ref(ref) for ref in certificate.selected_inputs),
            tuple(core_ref(ref) for ref in certificate.discarded_inputs),
            tuple(
                _exact_core_v1.CoreSamplingGridPointV1(
                    core_ref(point.input_ref),
                    tuple(_fraction_v3(value) for value in point.coordinates),
                )
                for point in certificate.grid_points
            ),
            certificate.grid_dimension,
            frame_binding,
            missing_policy,
            _exact_core_v1.CoreBoundaryPolicyV1(certificate.boundary_policy),
        )
    elif type(certificate) is FormalEquivalentSplitMergeCertificateV3:
        if certificate.equivalence_contract != "extensive_sum_one_sided_exact_inverse":
            raise ValueError("split/merge semantic literal drift")
        core_certificate = _exact_core_v1.CoreEquivalentSplitMergeCertificateV1(
            _exact_core_v1.CoreSplitMergeDirectionV1(certificate.direction),
            groups,
            tuple(core_row(row) for row in certificate.inverse_rows),
            output_label_bindings(),
            missing_policy,
        )
    elif type(certificate) is FormalCoarseGrainingCertificateV3:
        if (
            certificate.boundary_policy != "exact_partition"
            or certificate.commutation_contract
            != "transform_after_source_equals_target_after_transform"
            or certificate.weighted_mean_basis != FORMAL_WEIGHTED_MEAN_BASIS_V3
        ):
            raise ValueError("coarse-graining semantic literal drift")
        core_certificate = _exact_core_v1.CoreCoarseGrainingCertificateV1(
            _exact_core_v1.CoreReducerV1(certificate.reducer),
            groups,
            certificate.quotient_labels,
            tuple(core_row(row) for row in certificate.source_commutation_rows),
            tuple(core_row(row) for row in certificate.target_commutation_rows),
            output_label_bindings(),
            missing_policy,
            _exact_core_v1.CoreBoundaryPolicyV1(certificate.boundary_policy),
            _exact_core_v1.CoreWeightedMeanBasisV1(
                certificate.weighted_mean_basis
            ),
        )
    else:
        raise TypeError("unreachable V3 typed certificate")

    profile = _exact_core_v1.ExactTransformCoreProfileV1(
        source_components=tuple(
            sorted((core_value(ref) for ref in contract.input_refs), key=lambda item: item.ref)
        ),
        contract=_exact_core_v1.CoreExactTransformContractV1(
            operation=_exact_core_v1.CoreTransformOperationV1(contract.operation),
            source_scale_ref=scale_cache.setdefault(
                contract.source_scale_id,
                _exact_core_v1.CoreScaleRefV1(contract.source_scale_id),
            ),
            target_scale_ref=scale_cache.setdefault(
                contract.target_scale_id,
                _exact_core_v1.CoreScaleRefV1(contract.target_scale_id),
            ),
            input_refs=tuple(core_ref(ref) for ref in contract.input_refs),
            output_metadata=tuple(
                sorted(
                    (core_metadata(ref) for ref in contract.output_refs),
                    key=lambda item: item.ref,
                )
            ),
            kernel_rows=tuple(core_row(row) for row in contract.kernel_rows),
            discrete_mappings=tuple(
                _exact_core_v1.CoreDiscreteMappingV1(
                    core_ref(mapping.input_ref), core_ref(mapping.output_ref)
                )
                for mapping in contract.discrete_mappings
            ),
            certificate=core_certificate,
        ),
    )
    claimed = tuple(
        sorted((core_value(ref) for ref in contract.output_refs), key=lambda item: item.ref)
    )
    return profile, claimed


def _verify_transform_contract_against_values_v3(
    *,
    contract: FormalTransformContractV3,
    aggregate: FormalAggregateMapRecordV3,
    observations: dict[str, FormalObservationRecordV3],
    quantities: dict[str, FormalQuantityRecordV3],
    contexts: dict[str, FormalContextRecordV3],
) -> None:
    def context_semantics(
        ref: FormalComponentRefV3,
        *,
        omit_temporal_support: bool = False,
        omit_spatial_support: bool = False,
    ) -> tuple[object, ...]:
        context = contexts[ref.context_id]
        return (
            context.temporal_reference_kind,
            context.clock_binding,
            None if omit_temporal_support else context.temporal_support,
            context.spatial_reference_kind,
            context.frame_binding,
            context.coordinate_labels,
            None if omit_spatial_support else context.spatial_support,
            context.orientation,
        )

    def group_contexts_match(
        *,
        omit_temporal_support: bool = False,
        omit_spatial_support: bool = False,
    ) -> bool:
        return all(
            len(
                {
                    context_semantics(
                        ref,
                        omit_temporal_support=omit_temporal_support,
                        omit_spatial_support=omit_spatial_support,
                    )
                    for ref in (*group.input_refs, *group.output_refs)
                }
            )
            == 1
            for group in aggregate.groups
        )

    if contract.operation in {
        "identity",
        "unit_conversion",
        "equivalent_split_merge",
        "coarse_graining",
    } and not group_contexts_match():
        raise ValueError("transform operation does not preserve full context semantics")
    if contract.operation == "temporal_aggregation" and not group_contexts_match(
        omit_temporal_support=True
    ):
        raise ValueError("temporal aggregation changes a non-temporal context basis")
    if contract.operation == "spatial_aggregation" and not group_contexts_match(
        omit_spatial_support=True
    ):
        raise ValueError("spatial aggregation changes a non-spatial context basis")
    if contract.operation == "sampling_resolution":
        certificate = contract.certificate
        if type(certificate) is not FormalSamplingResolutionCertificateV3:
            raise TypeError("sampling certificate type drift before exact projection")
        if not group_contexts_match(
            omit_temporal_support=certificate.axis == "temporal",
            omit_spatial_support=certificate.axis == "spatial",
        ):
            raise ValueError("sampling changes a non-sampled context basis")
    if contract.operation == "coordinate_affine":
        source_contexts = {context_semantics(ref) for ref in contract.input_refs}
        target_contexts = {context_semantics(ref) for ref in contract.output_refs}
        if len(source_contexts) != 1 or len(target_contexts) != 1:
            raise ValueError("coordinate-affine sides do not have unique contexts")
        source_temporal = {
            (
                contexts[ref.context_id].temporal_reference_kind,
                contexts[ref.context_id].clock_binding,
                contexts[ref.context_id].temporal_support,
            )
            for ref in contract.input_refs
        }
        target_temporal = {
            (
                contexts[ref.context_id].temporal_reference_kind,
                contexts[ref.context_id].clock_binding,
                contexts[ref.context_id].temporal_support,
            )
            for ref in contract.output_refs
        }
        if source_temporal != target_temporal:
            raise ValueError("coordinate-affine changes temporal context semantics")
    if any(
        observations[ref.observation_id].uncertainty_model == "standard_error"
        for ref in (*contract.input_refs, *contract.output_refs)
    ):
        # Source construction may carry this sentinel only so the batch
        # builder can perform its required pre-root, whole-batch rejection.
        return
    profile, claimed = _project_transform_to_exact_core_v1(
        contract=contract,
        aggregate=aggregate,
        observations=observations,
        quantities=quantities,
        contexts=contexts,
    )
    result = _exact_core_v1.verify_exact_transform_core_outputs_v1(
        profile, claimed
    )
    if result.disposition is not _exact_core_v1.CoreTransformDispositionV1.COMPLETE:
        failure = result.failure
        detail = "unknown" if failure is None else failure.code.value + ":" + failure.detail
        raise ValueError("exact transform core rejected V3 projection: " + detail)
    return


def _validate_operation_certificate_literals_v3(
    *,
    contract: FormalTransformContractV3,
    aggregate: FormalAggregateMapRecordV3,
    contexts: dict[str, FormalContextRecordV3],
) -> None:
    """Validate every operation-tagged string before STANDARD_ERROR abstention."""
    certificate = contract.certificate
    expected_missing_policy = (
        "explicit_preserve"
        if contract.operation in {"identity", "unit_conversion", "coordinate_affine"}
        else "reject"
    )
    if certificate.missing_policy != expected_missing_policy:
        raise ValueError("formal transform missing-policy literal drift")
    non_reducers = {
        "identity",
        "unit_conversion",
        "coordinate_affine",
        "sampling_resolution",
        "equivalent_split_merge",
    }
    if (
        contract.operation in non_reducers
        and aggregate.reducer != "not_applicable"
    ) or (
        contract.operation in {"temporal_aggregation", "spatial_aggregation", "coarse_graining"}
        and aggregate.reducer not in {"sum", "weighted_mean"}
    ):
        raise ValueError("operation/reducer applicability drift")
    if type(certificate) is FormalIdentityCertificateV3:
        if certificate.inverse_contract != "exact_one_to_one_identity":
            raise ValueError("identity inverse-contract literal drift")
    elif type(certificate) is FormalUnitConversionCertificateV3:
        if (
            certificate.orientation
            != "target_value_equals_source_value_times_factor"
            or certificate.commutation_contract
            != "si_dimension_and_support_preserved"
        ):
            raise ValueError("unit-conversion semantic literal drift")
    elif type(certificate) is FormalCoordinateAffineCertificateV3:
        if (
            certificate.support_contract != "exact_affine_box_enclosure"
            or certificate.inverse_contract
            != "two_sided_exact_sparse_affine_inverse"
        ):
            raise ValueError("coordinate-affine semantic literal drift")
        source_context = contexts[contract.input_refs[0].context_id]
        target_context = contexts[contract.output_refs[0].context_id]
        if (
            certificate.source_frame_binding != source_context.frame_binding
            or certificate.target_frame_binding != target_context.frame_binding
        ):
            raise ValueError("coordinate-affine certificate frame binding drift")
    elif type(certificate) is FormalTemporalAggregationCertificateV3:
        if (
            certificate.boundary_policy != "exact_partition"
            or certificate.support_contract
            != "same_clock_half_open_partition_modulo_final_endpoint"
            or certificate.weighted_mean_basis != FORMAL_WEIGHTED_MEAN_BASIS_V3
        ):
            raise ValueError("temporal aggregation semantic literal drift")
    elif type(certificate) is FormalSpatialAggregationCertificateV3:
        if (
            certificate.boundary_policy != "exact_partition"
            or certificate.support_contract
            != "same_frame_interior_disjoint_positive_volume_partition"
            or certificate.weighted_mean_basis != FORMAL_WEIGHTED_MEAN_BASIS_V3
        ):
            raise ValueError("spatial aggregation semantic literal drift")
    elif type(certificate) is FormalSamplingResolutionCertificateV3:
        if (
            certificate.axis not in {"temporal", "spatial"}
            or certificate.boundary_policy != "valid_only"
            or certificate.kernel_contract
            != "single_series_ordered_exact_subselection"
        ):
            raise ValueError("sampling semantic literal drift")
        if certificate.grid_frame_binding != "none":
            source_context = contexts[certificate.selected_inputs[0].context_id]
            if certificate.grid_frame_binding != source_context.frame_binding:
                raise ValueError("sampling certificate grid frame binding drift")
    elif type(certificate) is FormalEquivalentSplitMergeCertificateV3:
        if certificate.direction not in {"split", "merge"} or (
            certificate.equivalence_contract
            != "extensive_sum_one_sided_exact_inverse"
        ):
            raise ValueError("split/merge semantic literal drift")
    elif type(certificate) is FormalCoarseGrainingCertificateV3:
        if (
            certificate.boundary_policy != "exact_partition"
            or certificate.commutation_contract
            != "transform_after_source_equals_target_after_transform"
            or certificate.weighted_mean_basis != FORMAL_WEIGHTED_MEAN_BASIS_V3
        ):
            raise ValueError("coarse-graining semantic literal drift")
    else:
        raise TypeError("unreachable typed certificate literal validator")

def _validate_case_graph_v3(graph: FormalCaseGraphV3) -> None:
    """Validate the closed rich graph before exact-core projection.

    Operation equations and claimed target values are checked by
    ``_verify_transform_contract_against_values_v3``.  This outer validator
    owns UUID/reference/canonical-order closure and rejects any profile that
    cannot be projected losslessly into that exact core.
    """
    if graph.schema_version != FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION:
        raise ValueError("formal case graph schema drift")
    _validate_routed_public_strings_v3(graph)
    _uuid(graph.bundle_id, "bundle_id")
    for name in (
        "entities", "quantities", "contexts", "observations", "scales",
        "aggregate_maps", "transforms",
    ):
        _tuple(getattr(graph, name), name, nonempty=True)
    roles = set(_uuid_tuple(graph.role_candidate_ids, "role_candidate_ids"))
    for item in graph.entities:
        if type(item) is not FormalEntityRecordV3:
            raise TypeError("entity record exact type drift")
        _uuid(item.entity_id, "entity_id")
        if not set(_uuid_tuple(item.role_candidate_ids, "entity roles")).issubset(roles):
            raise ValueError("entity role reference leaves registry")
    entities = set(_record_ids(graph.entities, "entity_id", "entity_id"))

    quantities_by_id: dict[str, FormalQuantityRecordV3] = {}
    for item in graph.quantities:
        if type(item) is not FormalQuantityRecordV3:
            raise TypeError("quantity record exact type drift")
        _uuid(item.quantity_id, "quantity_id")
        if item.quantity_id in quantities_by_id:
            raise ValueError("formal graph repeats quantity_id")
        _binding_token_v3(item.semantic_key, "quantity semantic binding")
        if type(item.si_exponents) is not tuple or len(item.si_exponents) != 7 or any(
            type(value) is not int or abs(value) > 16 for value in item.si_exponents
        ):
            raise ValueError("quantity SI exponents are invalid")
        if type(item.unit_binding) is not FormalUnitBindingV3:
            raise TypeError("quantity unit binding exact type drift")
        item.unit_binding.__post_init__()
        if item.value_kind not in ("numeric", "boolean"):
            raise ValueError("quantity value_kind is unknown")
        if item.value_kind == "boolean":
            if (
                item.si_exponents != (0, 0, 0, 0, 0, 0, 0)
                or item.unit_binding.semantic_key != "none"
                or _fraction_v3(item.unit_binding.scale_to_si) != 1
                or _fraction_v3(item.unit_binding.offset_to_si) != 0
            ):
                raise ValueError(
                    "Boolean/missing-compatible quantity must use the exact unitless sentinel"
                )
        elif item.unit_binding.semantic_key == "none":
            raise ValueError("numeric quantity cannot use the unitless discrete sentinel")
        quantities_by_id[item.quantity_id] = item

    contexts_by_id: dict[str, FormalContextRecordV3] = {}
    for item in graph.contexts:
        if type(item) is not FormalContextRecordV3:
            raise TypeError("context record exact type drift")
        _uuid(item.context_id, "context_id")
        if item.context_id in contexts_by_id:
            raise ValueError("formal graph repeats context_id")
        if item.temporal_reference_kind not in ("none", "relative_monotonic", "utc_unix", "domain_epoch") or item.spatial_reference_kind not in ("none", "cartesian", "geodetic", "topological"):
            raise ValueError("context reference kind is unknown")
        _binding_token_v3(item.clock_binding, "context clock binding")
        _binding_token_v3(item.frame_binding, "context frame binding")
        if type(item.coordinate_labels) is not tuple or len(item.coordinate_labels) > 4:
            raise ValueError("context coordinate labels are invalid")
        for label in item.coordinate_labels:
            _public_semantic_text_v3(label, "context coordinate label")
        if len(set(item.coordinate_labels)) != len(item.coordinate_labels):
            raise ValueError("context coordinate labels repeat")
        if type(item.temporal_support) is not NullableTemporalSupportV3 or type(item.spatial_support) is not NullableSpatialSupportV3 or type(item.orientation) is not NullableTextV3:
            raise TypeError("context nullable field type drift")
        item.temporal_support.__post_init__(); item.spatial_support.__post_init__(); item.orientation.__post_init__()
        temporal_none = item.temporal_reference_kind == "none"
        if temporal_none is item.temporal_support.available or temporal_none != (item.clock_binding == "none"):
            raise ValueError("context temporal reference/support/binding mismatch")
        metric = item.spatial_reference_kind in ("cartesian", "geodetic")
        if metric:
            support = item.spatial_support.value
            if item.frame_binding == "none" or not item.coordinate_labels or type(support) is not ExactSpatialSupportV3 or len(item.coordinate_labels) != len(support.lower):
                raise ValueError("metric spatial context binding/support dimension mismatch")
        elif item.coordinate_labels or item.spatial_support.available or item.orientation.available:
            raise ValueError("nonmetric spatial context cannot carry metric fields")
        if (item.spatial_reference_kind == "none") != (item.frame_binding == "none"):
            raise ValueError("context spatial reference/frame binding mismatch")
        contexts_by_id[item.context_id] = item

    scales: set[str] = set()
    for item in graph.scales:
        if type(item) is not FormalScaleRecordV3:
            raise TypeError("scale record exact type drift")
        _uuid(item.scale_id, "scale_id"); _public_semantic_text_v3(item.semantic_level, "scale semantic level")
        if type(item.is_root) is not bool or item.scale_id in scales:
            raise ValueError("scale record bool/identity drift")
        scales.add(item.scale_id)
    if not any(item.is_root for item in graph.scales):
        raise ValueError("formal graph needs a root scale")

    semantic_axes = {"scalar", "temporal", "spatial", "channel", "entity", "coordinate", "coarse", "control"}
    value_roles = {"intensive", "extensive", "coordinate", "boolean_control", "missing"}
    observations: dict[str, FormalObservationRecordV3] = {}
    for item in graph.observations:
        if type(item) is not FormalObservationRecordV3:
            raise TypeError("observation record exact type drift")
        _uuid(item.observation_id, "observation_id")
        if item.observation_id in observations:
            raise ValueError("formal graph repeats observation_id")
        if not set(_uuid_tuple(item.entity_ids, "observation entities")).issubset(entities) or not set(_uuid_tuple(item.role_candidate_ids, "observation roles")).issubset(roles):
            raise ValueError("observation entity/role reference leaves registry")
        if item.quantity_id not in quantities_by_id or item.context_id not in contexts_by_id or item.scale_id not in scales:
            raise ValueError("observation semantic owner reference unresolved")
        if type(item.components) is not tuple or not 1 <= len(item.components) <= MAXIMUM_COMPONENTS_PER_OBSERVATION_V3:
            raise ValueError("observation component descriptor count invalid")
        labels: list[str] = []
        for component in item.components:
            if type(component) is not FormalComponentDescriptorV3:
                raise TypeError("component descriptor exact type drift")
            _public_semantic_text_v3(component.label, "component label")
            if component.semantic_axis not in semantic_axes or component.value_role not in value_roles:
                raise ValueError("component semantic axis/value role leaves frozen enum")
            labels.append(component.label)
        if len(set(labels)) != len(labels):
            raise ValueError("observation repeats component label")
        item.numeric_value.__post_init__(); item.boolean_value.__post_init__(); item.uncertainty_radius.__post_init__()
        missing = not item.numeric_value.available and not item.boolean_value.available
        kind = quantities_by_id[item.quantity_id].value_kind
        if missing and kind != "boolean":
            raise ValueError(
                "missing observation requires an explicit unitless discrete quantity"
            )
        if not missing and kind == "numeric":
            if not item.numeric_value.available or item.boolean_value.available or len(item.numeric_value.value or ()) != len(item.components):
                raise ValueError("numeric observation value/component mismatch")
            if any(component.value_role in {"boolean_control", "missing"} for component in item.components):
                raise ValueError("present numeric observation has discrete component role")
        elif not missing and (not item.boolean_value.available or item.numeric_value.available or len(item.components) != 1):
            raise ValueError("boolean observation value/component mismatch")
        elif not missing and (
            item.components[0].semantic_axis != "control"
            or item.components[0].value_role != "boolean_control"
        ):
            raise ValueError("Boolean observation component semantics drift")
        if missing and any(component.value_role != "missing" for component in item.components):
            raise ValueError("missing observation needs explicit missing component roles")
        if item.uncertainty_model not in ("absolute_bound", "not_applicable", "standard_error"):
            raise ValueError("observation uncertainty model unknown")
        if kind == "numeric" and not missing:
            radius = item.uncertainty_radius.value or ()
            if item.uncertainty_model == "not_applicable" or not item.uncertainty_radius.available or len(radius) != len(item.components) or any(_fraction_v3(value) < 0 for value in radius):
                raise ValueError("numeric uncertainty radius is not exact component-complete nonnegative")
        elif item.uncertainty_model != "not_applicable" or item.uncertainty_radius.available:
            raise ValueError("boolean/missing uncertainty must be not_applicable")
        _digest(item.public_provenance_commitment, "observation provenance")
        observations[item.observation_id] = item

    task = graph.task_target
    if type(task) is not FormalTaskTargetV3:
        raise TypeError("task target exact type drift")
    _uuid(task.task_id, "task_id")
    if not set(_uuid_tuple(task.entity_ids, "task entities")).issubset(entities) or not set(_uuid_tuple(task.quantity_ids, "task quantities")).issubset(quantities_by_id):
        raise ValueError("task target reference leaves registry")
    if type(graph.semantic_descriptor) is not FormalSemanticDescriptorV3:
        raise TypeError("formal semantic descriptor type drift")
    graph.semantic_descriptor.__post_init__()

    aggregate_by_transform: dict[str, FormalAggregateMapRecordV3] = {}
    aggregate_ids: set[str] = set()
    for item in graph.aggregate_maps:
        if type(item) is not FormalAggregateMapRecordV3:
            raise TypeError("aggregate-map record exact type drift")
        _uuid(item.aggregate_map_id, "aggregate_map_id"); _uuid(item.transform_id, "transform_id")
        if item.aggregate_map_id in aggregate_ids or item.transform_id in aggregate_by_transform:
            raise ValueError("aggregate maps/transforms are not one-to-one")
        if item.operation not in TRANSFORM_OPERATIONS_V3 or item.source_scale_id not in scales or item.target_scale_id not in scales or item.reducer not in ("not_applicable", "sum", "weighted_mean"):
            raise ValueError("aggregate-map operation/scale/reducer drift")
        _tuple(item.groups, "aggregate groups", nonempty=True)
        if item.groups != tuple(
            sorted(
                item.groups,
                key=lambda group: (
                    group.input_refs,
                    group.output_refs,
                    group.quotient_label,
                ),
            )
        ):
            raise ValueError(
                "aggregate groups must be canonical before quotient-label binding"
            )
        labels: set[str] = set()
        for group in item.groups:
            if type(group) is not FormalAggregateGroupV3 or not group.input_refs or not group.output_refs:
                raise ValueError("aggregate group exact nonempty shape drift")
            if group.input_refs != tuple(sorted(group.input_refs)) or group.output_refs != tuple(sorted(group.output_refs)):
                raise ValueError("aggregate refs must be canonical")
            _public_semantic_text_v3(group.quotient_label, "quotient label")
            if group.quotient_label in labels:
                raise ValueError("aggregate quotient label repeated")
            labels.add(group.quotient_label)
            for ref in (*group.input_refs, *group.output_refs):
                _validate_component_ref_v3(ref, observations)
        aggregate_ids.add(item.aggregate_map_id)
        aggregate_by_transform[item.transform_id] = item

    transforms: dict[str, FormalTransformContractV3] = {}
    for contract in graph.transforms:
        if type(contract) is not FormalTransformContractV3:
            raise TypeError("transform contract exact type drift")
        _uuid(contract.transform_id, "transform_id")
        if contract.transform_id in transforms or contract.transform_id not in aggregate_by_transform:
            raise ValueError("transform is repeated or unused")
        aggregate = aggregate_by_transform[contract.transform_id]
        if (contract.operation, contract.source_scale_id, contract.target_scale_id) != (aggregate.operation, aggregate.source_scale_id, aggregate.target_scale_id):
            raise ValueError("aggregate-map/transform identity mismatch")
        if type(contract.certificate) is not OPERATION_CERTIFICATE_TYPE_V3[contract.operation] or getattr(contract.certificate, "semantics_version", None) != FORMAL_OPERATION_SEMANTICS_VERSION_V3:
            raise ValueError("transform operation-tagged certificate mismatch")
        if type(contract.input_refs) is not tuple or type(contract.output_refs) is not tuple or not contract.input_refs or not contract.output_refs:
            raise ValueError("transform declared refs must be exact nonempty tuples")
        if contract.input_refs != tuple(sorted(contract.input_refs)) or contract.output_refs != tuple(sorted(contract.output_refs)) or len(set(contract.input_refs)) != len(contract.input_refs) or len(set(contract.output_refs)) != len(contract.output_refs):
            raise ValueError("transform declared refs must be canonical unique")
        for ref in (*contract.input_refs, *contract.output_refs):
            _validate_component_ref_v3(ref, observations)
        if any(ref.scale_id != contract.source_scale_id for ref in contract.input_refs) or any(ref.scale_id != contract.target_scale_id for ref in contract.output_refs):
            raise ValueError("transform ref leaves declared source/target scale")
        group_inputs = tuple(ref for group in aggregate.groups for ref in group.input_refs)
        group_outputs = tuple(ref for group in aggregate.groups for ref in group.output_refs)
        if len(set(group_inputs)) != len(group_inputs) or len(set(group_outputs)) != len(group_outputs) or set(group_inputs) != set(contract.input_refs) or set(group_outputs) != set(contract.output_refs):
            raise ValueError("aggregate groups do not exactly partition transform refs")
        _validate_exact_rows_v3(contract.kernel_rows, input_domain=set(contract.input_refs), output_domain=set(contract.output_refs), observations=observations, name="kernel")
        if type(contract.discrete_mappings) is not tuple:
            raise TypeError("discrete mappings must use exact tuple")
        for mapping in contract.discrete_mappings:
            if type(mapping) is not FormalDiscreteMappingV3 or mapping.input_ref not in contract.input_refs or mapping.output_ref not in contract.output_refs:
                raise ValueError("discrete mapping leaves declared refs")
        if contract.discrete_mappings != tuple(sorted(contract.discrete_mappings)) or len({row.input_ref for row in contract.discrete_mappings}) != len(contract.discrete_mappings) or len({row.output_ref for row in contract.discrete_mappings}) != len(contract.discrete_mappings):
            raise ValueError("discrete mappings must be canonical one-to-one")
        for row in contract.kernel_rows:
            matching = tuple(
                group for group in aggregate.groups if row.output_ref in group.output_refs
            )
            if len(matching) != 1 or any(
                term.input_ref not in matching[0].input_refs for term in row.terms
            ):
                raise ValueError("kernel row crosses an aggregate-group boundary")
        for mapping in contract.discrete_mappings:
            matching = tuple(
                group
                for group in aggregate.groups
                if mapping.output_ref in group.output_refs
            )
            if len(matching) != 1 or mapping.input_ref not in matching[0].input_refs:
                raise ValueError("discrete mapping crosses an aggregate-group boundary")
        kernel_outputs = {row.output_ref for row in contract.kernel_rows}
        discrete_outputs = {row.output_ref for row in contract.discrete_mappings}
        if kernel_outputs & discrete_outputs or kernel_outputs | discrete_outputs != set(contract.output_refs):
            raise ValueError("kernel and discrete mappings do not exactly cover outputs")
        certificate = contract.certificate
        for binding_name in ("aggregate_map_id", "quotient_aggregate_map_id"):
            if hasattr(certificate, binding_name) and getattr(certificate, binding_name) != aggregate.aggregate_map_id:
                raise ValueError("certificate aggregate-map binding drift")
        if hasattr(certificate, "reducer") and getattr(certificate, "reducer") != aggregate.reducer:
            raise ValueError("certificate/aggregate reducer drift")
        if hasattr(certificate, "weighted_mean_basis") and getattr(
            certificate, "weighted_mean_basis"
        ) != FORMAL_WEIGHTED_MEAN_BASIS_V3:
            raise ValueError("certificate weighted-mean basis literal drift")
        if type(certificate) is FormalCoarseGrainingCertificateV3 and certificate.quotient_labels != tuple(group.quotient_label for group in aggregate.groups):
            raise ValueError("coarse quotient labels do not positionally bind groups")
        for row_name, input_domain, output_domain in (
            ("inverse_rows", set(contract.output_refs), set(contract.input_refs)),
            ("source_commutation_rows", set(contract.input_refs), set(contract.input_refs)),
            ("target_commutation_rows", set(contract.output_refs), set(contract.output_refs)),
        ):
            if hasattr(certificate, row_name):
                auxiliary_rows = getattr(certificate, row_name)
                _validate_exact_rows_v3(auxiliary_rows, input_domain=input_domain, output_domain=output_domain, observations=observations, name=row_name)
                for row in auxiliary_rows:
                    if row_name == "inverse_rows":
                        matching = tuple(
                            group
                            for group in aggregate.groups
                            if row.output_ref in group.input_refs
                        )
                        allowed_inputs = () if len(matching) != 1 else matching[0].output_refs
                    elif row_name == "source_commutation_rows":
                        matching = tuple(
                            group
                            for group in aggregate.groups
                            if row.output_ref in group.input_refs
                        )
                        allowed_inputs = () if len(matching) != 1 else matching[0].input_refs
                    else:
                        matching = tuple(
                            group
                            for group in aggregate.groups
                            if row.output_ref in group.output_refs
                        )
                        allowed_inputs = () if len(matching) != 1 else matching[0].output_refs
                    if len(matching) != 1 or any(
                        term.input_ref not in allowed_inputs for term in row.terms
                    ):
                        raise ValueError(
                            f"{row_name} row crosses an aggregate-group boundary"
                        )
        if type(certificate) is FormalSamplingResolutionCertificateV3:
            if certificate.axis not in ("temporal", "spatial") or type(certificate.grid_dimension) is not int or not 1 <= certificate.grid_dimension <= 4:
                raise ValueError("sampling axis/grid dimension drift")
            if not certificate.selected_inputs or len(set(certificate.selected_inputs)) != len(certificate.selected_inputs) or certificate.discarded_inputs != tuple(sorted(certificate.discarded_inputs)) or len(set(certificate.discarded_inputs)) != len(certificate.discarded_inputs):
                raise ValueError("sampling selected/discarded canonicality drift")
            if set(certificate.selected_inputs) & set(certificate.discarded_inputs) or set(certificate.selected_inputs) | set(certificate.discarded_inputs) != set(contract.input_refs):
                raise ValueError("sampling selected/discarded coverage drift")
            if len(certificate.grid_points) != len(certificate.selected_inputs):
                raise ValueError("sampling grid/selected count drift")
            for selected, point in zip(certificate.selected_inputs, certificate.grid_points, strict=True):
                if type(point) is not FormalSamplingGridPointV3 or point.input_ref != selected or len(point.coordinates) != certificate.grid_dimension:
                    raise ValueError("sampling grid point/input positional binding drift")
                for atom in point.coordinates:
                    atom.__post_init__()
            _binding_token_v3(certificate.grid_frame_binding, "sampling grid frame binding")
            if (certificate.axis == "temporal") != (certificate.grid_frame_binding == "none"):
                raise ValueError("sampling axis/grid frame applicability drift")
        _validate_operation_certificate_literals_v3(
            contract=contract,
            aggregate=aggregate,
            contexts=contexts_by_id,
        )
        _verify_transform_contract_against_values_v3(contract=contract, aggregate=aggregate, observations=observations, quantities=quantities_by_id, contexts=contexts_by_id)
        transforms[contract.transform_id] = contract
    if set(transforms) != set(aggregate_by_transform):
        raise ValueError("formal graph contains an unused aggregate map")

    registry = graph.recognizer_registry
    if type(registry) is not FormalRecognizerRegistryV3 or type(registry.max_candidate_count) is not int or not 1 <= registry.max_candidate_count <= MAXIMUM_CANDIDATE_COUNT_V3:
        raise ValueError("recognizer registry type/count drift")
    _public_semantic_text_v3(registry.theory_policy, "theory policy")
    aliases: set[str] = set(); families: set[str] = set()
    for alias in registry.family_aliases:
        if type(alias) is not FormalFamilyAliasV3:
            raise TypeError("family alias type drift")
        _public_semantic_text_v3(alias.alias, "family alias"); _public_semantic_text_v3(alias.canonical_family, "canonical family")
        if alias.alias in aliases:
            raise ValueError("family alias repeats")
        aliases.add(alias.alias); families.add(alias.canonical_family)
    law_names: set[str] = set()
    for law in registry.law_bindings:
        if type(law) is not FormalLawBindingV3:
            raise TypeError("law binding type drift")
        _public_semantic_text_v3(law.law_name, "law name"); _public_semantic_text_v3(law.family, "law family"); _public_semantic_text_v3(law.law_kind, "law kind")
        if law.law_name in law_names or law.family not in aliases | families or not set(_uuid_tuple(law.required_observation_ids, "law observations")).issubset(observations) or not set(_uuid_tuple(law.required_role_candidate_ids, "law roles")).issubset(roles) or not set(_uuid_tuple(law.required_quantity_ids, "law quantities")).issubset(quantities_by_id):
            raise ValueError("law binding closure drift")
        law_names.add(law.law_name)
    channels: dict[str, str] = {}
    for channel in registry.observable_channels:
        if type(channel) is not FormalObservableChannelV3 or channel.observation_id in channels or channel.observation_id not in observations or channel.quantity_id != observations[channel.observation_id].quantity_id:
            raise ValueError("observable-channel closure drift")
        channels[channel.observation_id] = channel.quantity_id
    if set(channels) != set(observations):
        raise ValueError("observable channels must cover every observation")
    mask = graph.eligibility_mask
    if type(mask) is not FormalEligibilityMaskV3 or not set(_uuid_tuple(mask.observation_ids, "eligibility observations")).issubset(observations) or not set(_uuid_tuple(mask.role_candidate_ids, "eligibility roles")).issubset(roles) or not set(_uuid_tuple(mask.scale_ids, "eligibility scales")).issubset(scales):
        raise ValueError("eligibility-mask closure drift")

    adjacency = {scale: set() for scale in scales}; indegree = {scale: 0 for scale in scales}
    for aggregate in graph.aggregate_maps:
        edge = (aggregate.source_scale_id, aggregate.target_scale_id)
        if aggregate.target_scale_id in adjacency[aggregate.source_scale_id]:
            raise ValueError("aggregate graph repeats an edge")
        adjacency[aggregate.source_scale_id].add(aggregate.target_scale_id); indegree[aggregate.target_scale_id] += 1
    visiting: set[str] = set(); visited: set[str] = set()
    def visit(scale: str) -> None:
        if scale in visiting:
            raise ValueError("aggregate graph is cyclic")
        if scale in visited:
            return
        visiting.add(scale)
        for target in adjacency[scale]: visit(target)
        visiting.remove(scale); visited.add(scale)
    for scale in scales: visit(scale)
    roots = {item.scale_id for item in graph.scales if item.is_root}
    if roots != {scale for scale, degree in indegree.items() if degree == 0}:
        raise ValueError("scale root flags do not equal indegree-zero scales")
    reachable: set[str] = set(); frontier = list(roots)
    while frontier:
        scale = frontier.pop()
        if scale not in reachable:
            reachable.add(scale); frontier.extend(adjacency[scale])
    if reachable != scales:
        raise ValueError("not every scale is root-reachable")


def _profile_value_v3(value: object) -> object:
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is tuple:
        return [_profile_value_v3(item) for item in value]
    if is_dataclass(value):
        return {item.name: _profile_value_v3(getattr(value, item.name)) for item in fields(value)}
    raise TypeError("formal graph contains a value outside the closed profile")


_FORMAL_PUBLIC_GRAPH_DTO_TYPES_V3: Final = (
    RationalValueV3,
    NullableTextV3,
    NullableBooleanV3,
    NullableRationalVectorV3,
    ExactTemporalSupportV3,
    NullableTemporalSupportV3,
    ExactSpatialSupportV3,
    NullableSpatialSupportV3,
    FormalUnitBindingV3,
    FormalEntityRecordV3,
    FormalQuantityRecordV3,
    FormalContextRecordV3,
    FormalObservationRecordV3,
    FormalTaskTargetV3,
    FormalSemanticDescriptorV3,
    FormalScaleRecordV3,
    FormalComponentDescriptorV3,
    FormalComponentRefV3,
    FormalExactTermV3,
    FormalExactRowV3,
    FormalSamplingGridPointV3,
    FormalDiscreteMappingV3,
    FormalAggregateGroupV3,
    FormalAggregateMapRecordV3,
    FormalIdentityCertificateV3,
    FormalUnitConversionCertificateV3,
    FormalCoordinateAffineCertificateV3,
    FormalTemporalAggregationCertificateV3,
    FormalSpatialAggregationCertificateV3,
    FormalSamplingResolutionCertificateV3,
    FormalEquivalentSplitMergeCertificateV3,
    FormalCoarseGrainingCertificateV3,
    FormalTransformContractV3,
    FormalFamilyAliasV3,
    FormalLawBindingV3,
    FormalObservableChannelV3,
    FormalRecognizerRegistryV3,
    FormalEligibilityMaskV3,
    FormalCaseGraphV3,
    FormalPublicCaseV3,
)
FORMAL_PUBLIC_DTO_FIELD_MANIFEST_V3: Final = {
    **{
        data_type.__name__: tuple(item.name for item in fields(data_type))
        for data_type in _FORMAL_PUBLIC_GRAPH_DTO_TYPES_V3
    },
    "FormalBatchContextV3": (
        "case_count_decimal",
        "transform_catalog_root",
        "id_collision_retry_count_decimal",
        "run_id_commitment",
    ),
    "FormalPayloadV3": (
        "batch_context",
        "field_classification_manifest_id",
        "graph_completeness_policy_id",
        "public_case",
        "schema_version",
    ),
}

# Exact routing for every public DTO field whose direct value is ``str``,
# ``str | None``, or ``tuple[str, ...]``.  Nested DTO fields are routed by the
# nested type.  This is intentionally independent of the leaf classifier:
# semantic text must never be confused with typed UUID/hash/decimal fields.
_FORMAL_PUBLIC_STRING_FIELD_POLICY_ROWS_V3: Final = {
    "RationalValueV3": {"numerator": "CANONICAL_DECIMAL", "denominator": "CANONICAL_DECIMAL"},
    "NullableTextV3": {"value": "SEMANTIC_TEXT_OR_NULL"},
    "FormalUnitBindingV3": {"semantic_key": "BINDING_TOKEN"},
    "FormalEntityRecordV3": {"entity_id": "UUID", "role_candidate_ids": "UUID_TUPLE"},
    "FormalQuantityRecordV3": {"quantity_id": "UUID", "semantic_key": "BINDING_TOKEN", "value_kind": "ENUM_LITERAL"},
    "FormalContextRecordV3": {
        "context_id": "UUID", "temporal_reference_kind": "ENUM_LITERAL",
        "clock_binding": "BINDING_TOKEN", "spatial_reference_kind": "ENUM_LITERAL",
        "frame_binding": "BINDING_TOKEN", "coordinate_labels": "SEMANTIC_TEXT_TUPLE",
    },
    "FormalObservationRecordV3": {
        "observation_id": "UUID", "entity_ids": "UUID_TUPLE",
        "role_candidate_ids": "UUID_TUPLE", "quantity_id": "UUID",
        "context_id": "UUID", "scale_id": "UUID", "uncertainty_model": "ENUM_LITERAL",
        "public_provenance_commitment": "DIGEST",
    },
    "FormalTaskTargetV3": {"task_id": "UUID", "entity_ids": "UUID_TUPLE", "quantity_ids": "UUID_TUPLE"},
    "FormalSemanticDescriptorV3": {"kind": "ENUM_LITERAL", "segments": "SEMANTIC_DESCRIPTOR_TEXT_TUPLE"},
    "FormalScaleRecordV3": {"scale_id": "UUID", "semantic_level": "SEMANTIC_TEXT"},
    "FormalComponentDescriptorV3": {"label": "SEMANTIC_TEXT", "semantic_axis": "ENUM_LITERAL", "value_role": "ENUM_LITERAL"},
    "FormalComponentRefV3": {
        "observation_id": "UUID", "quantity_id": "UUID", "context_id": "UUID",
        "scale_id": "UUID", "component_axis": "SEMANTIC_TEXT",
    },
    "FormalAggregateGroupV3": {"quotient_label": "SEMANTIC_TEXT"},
    "FormalAggregateMapRecordV3": {
        "aggregate_map_id": "UUID", "source_scale_id": "UUID", "target_scale_id": "UUID",
        "transform_id": "UUID", "operation": "ENUM_LITERAL", "reducer": "ENUM_LITERAL",
    },
    "FormalIdentityCertificateV3": {"semantics_version": "EXACT_LITERAL", "missing_policy": "ENUM_LITERAL", "inverse_contract": "EXACT_LITERAL"},
    "FormalUnitConversionCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "missing_policy": "ENUM_LITERAL",
        "orientation": "EXACT_LITERAL", "commutation_contract": "EXACT_LITERAL",
    },
    "FormalCoordinateAffineCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "source_frame_binding": "BINDING_TOKEN",
        "target_frame_binding": "BINDING_TOKEN", "missing_policy": "ENUM_LITERAL",
        "support_contract": "EXACT_LITERAL", "inverse_contract": "EXACT_LITERAL",
    },
    "FormalTemporalAggregationCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "aggregate_map_id": "UUID",
        "reducer": "ENUM_LITERAL", "missing_policy": "ENUM_LITERAL",
        "boundary_policy": "EXACT_LITERAL", "support_contract": "EXACT_LITERAL",
        "weighted_mean_basis": "EXACT_LITERAL",
    },
    "FormalSpatialAggregationCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "aggregate_map_id": "UUID",
        "reducer": "ENUM_LITERAL", "missing_policy": "ENUM_LITERAL",
        "boundary_policy": "EXACT_LITERAL", "support_contract": "EXACT_LITERAL",
        "weighted_mean_basis": "EXACT_LITERAL",
    },
    "FormalSamplingResolutionCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "axis": "ENUM_LITERAL",
        "grid_frame_binding": "BINDING_TOKEN", "missing_policy": "ENUM_LITERAL",
        "boundary_policy": "EXACT_LITERAL", "kernel_contract": "EXACT_LITERAL",
    },
    "FormalEquivalentSplitMergeCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "aggregate_map_id": "UUID",
        "direction": "ENUM_LITERAL", "missing_policy": "ENUM_LITERAL",
        "equivalence_contract": "EXACT_LITERAL",
    },
    "FormalCoarseGrainingCertificateV3": {
        "semantics_version": "EXACT_LITERAL", "aggregate_map_id": "UUID",
        "reducer": "ENUM_LITERAL", "quotient_aggregate_map_id": "UUID",
        "quotient_labels": "SEMANTIC_TEXT_TUPLE", "missing_policy": "ENUM_LITERAL",
        "boundary_policy": "EXACT_LITERAL", "commutation_contract": "EXACT_LITERAL",
        "weighted_mean_basis": "EXACT_LITERAL",
    },
    "FormalTransformContractV3": {
        "transform_id": "UUID", "operation": "ENUM_LITERAL",
        "source_scale_id": "UUID", "target_scale_id": "UUID",
    },
    "FormalFamilyAliasV3": {"alias": "SEMANTIC_TEXT", "canonical_family": "SEMANTIC_TEXT"},
    "FormalLawBindingV3": {
        "law_name": "SEMANTIC_TEXT", "family": "SEMANTIC_TEXT", "law_kind": "SEMANTIC_TEXT",
        "required_observation_ids": "UUID_TUPLE", "required_role_candidate_ids": "UUID_TUPLE",
        "required_quantity_ids": "UUID_TUPLE",
    },
    "FormalObservableChannelV3": {"observation_id": "UUID", "quantity_id": "UUID"},
    "FormalRecognizerRegistryV3": {"theory_policy": "SEMANTIC_TEXT"},
    "FormalEligibilityMaskV3": {"observation_ids": "UUID_TUPLE", "role_candidate_ids": "UUID_TUPLE", "scale_ids": "UUID_TUPLE"},
    "FormalCaseGraphV3": {
        "schema_version": "EXACT_LITERAL", "bundle_id": "UUID",
        "role_candidate_ids": "UUID_TUPLE",
    },
    "FormalPublicCaseV3": {"schema_version": "EXACT_LITERAL", "public_provenance_commitment": "DIGEST"},
    "FormalBatchContextV3": {
        "case_count_decimal": "UNSIGNED_DECIMAL", "transform_catalog_root": "DIGEST",
        "id_collision_retry_count_decimal": "UNSIGNED_DECIMAL", "run_id_commitment": "DIGEST",
    },
    "FormalPayloadV3": {
        "field_classification_manifest_id": "PREFIXED_DIGEST_ID",
        "graph_completeness_policy_id": "PREFIXED_DIGEST_ID", "schema_version": "EXACT_LITERAL",
    },
}
FORMAL_PUBLIC_STRING_FIELD_POLICY_V3: Final = MappingProxyType(
    {
        type_name: MappingProxyType(dict(field_routes))
        for type_name, field_routes in _FORMAL_PUBLIC_STRING_FIELD_POLICY_ROWS_V3.items()
    }
)


def _validate_routed_public_strings_v3(value: object) -> None:
    """Fail closed on any concrete public string leaf missing exact routing."""
    stack = [value]
    while stack:
        current = stack.pop()
        if is_dataclass(current):
            type_name = type(current).__name__
            policy = FORMAL_PUBLIC_STRING_FIELD_POLICY_V3.get(type_name, {})
            for item in fields(current):
                child = getattr(current, item.name)
                route = policy.get(item.name)
                tuple_strings = type(child) is tuple and (
                    (bool(child) and all(type(row) is str for row in child))
                    or route
                    in {
                        "UUID_TUPLE",
                        "SEMANTIC_TEXT_TUPLE",
                        "SEMANTIC_DESCRIPTOR_TEXT_TUPLE",
                    }
                )
                direct_strings = (
                    type(child) is str
                    or child is None and item.name in policy
                    or tuple_strings
                )
                if direct_strings:
                    if route is None:
                        raise ValueError(
                            f"public string field is not routed: {type_name}.{item.name}"
                        )
                    if route in {"SEMANTIC_TEXT", "SEMANTIC_TEXT_OR_NULL"}:
                        if child is not None:
                            _public_semantic_text_v3(
                                child, f"{type_name}.{item.name}",
                                empty=route == "SEMANTIC_TEXT_OR_NULL",
                            )
                    elif route in {"SEMANTIC_TEXT_TUPLE", "SEMANTIC_DESCRIPTOR_TEXT_TUPLE"}:
                        if type(child) is not tuple:
                            raise TypeError("public semantic string collection type drift")
                        for row in child:
                            _public_semantic_text_v3(
                                row, f"{type_name}.{item.name}[]"
                            )
                    elif route == "BINDING_TOKEN":
                        _binding_token_v3(child, f"{type_name}.{item.name}")
                    elif route == "UUID":
                        _uuid(child, f"{type_name}.{item.name}")
                    elif route == "UUID_TUPLE":
                        _uuid_tuple(child, f"{type_name}.{item.name}", nonempty=False)
                    elif route == "DIGEST":
                        _digest(child, f"{type_name}.{item.name}")
                    elif route == "CANONICAL_DECIMAL":
                        if type(child) is not str or _DECIMAL_RE.fullmatch(child) is None:
                            raise ValueError("routed canonical decimal drift")
                    elif route == "UNSIGNED_DECIMAL":
                        if type(child) is not str or _UNSIGNED_DECIMAL_RE.fullmatch(child) is None:
                            raise ValueError("routed unsigned decimal drift")
                    elif route == "PREFIXED_DIGEST_ID":
                        if type(child) is not str or len(child.rsplit("_", 1)[-1]) != 64:
                            raise ValueError("routed prefixed digest ID drift")
                    elif route not in {"ENUM_LITERAL", "EXACT_LITERAL"}:
                        raise RuntimeError("unknown public string-field route")
                else:
                    stack.append(child)
        elif type(current) is tuple:
            stack.extend(current)


FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3: Final = (
    "phase2b_formal_case_graph_completeness_policy_v3_"
    + hashlib.sha256(
        _POLICY_DOMAIN_V3
        + canonicalize_rfc8785_v1(
            {
                "exact_uuid_namespaces": list(NAMESPACE_REGISTRY_V3),
                "extension_uuid_policy": "MUST_BE_ABSENT",
                "graph_fields": [item.name for item in fields(FormalCaseGraphV3)],
                "operations": list(TRANSFORM_OPERATIONS_V3),
                "operation_typed_certificate_manifest": {key: sorted(value) for key, value in OPERATION_CLAUSE_MANIFEST_V3.items()},
                "public_dto_exact_field_manifest": {
                    key: list(value) for key, value in sorted(FORMAL_PUBLIC_DTO_FIELD_MANIFEST_V3.items())
                },
                "public_string_field_policy": {
                    key: dict(sorted(value.items()))
                    for key, value in sorted(FORMAL_PUBLIC_STRING_FIELD_POLICY_V3.items())
                },
                "cross_closure": (
                    "roots_indegree_reachability_unique_edges_scale_matched_refs_"
                    "aggregate_group_exact_partitions_group_boundary_closure_unique_quotient_labels_"
                    "typed_sparse_rows_discrete_maps_weighted_mean_declared_basis_output_label_bindings_"
                    "operation_certificate_equations_target_value_absolute_uncertainty_registry_scope_"
                    "context_support_dimension_uncertainty_component_nonnegative_"
                    "semantic_descriptor_closed_kind_nfc_utf8_bounds_safe_unicode_default_ignorable_no_identifier_digest_tokens"
                ),
                "exact_transform_core_source_sha256": EXACT_TRANSFORM_CORE_SOURCE_SHA256_V3,
                "exact_transform_core_version": _exact_core_v1.EXACT_TRANSFORM_CORE_VERSION_V1,
                "operation_semantics_version": FORMAL_OPERATION_SEMANTICS_VERSION_V3,
                "semantic_descriptor_unicode_data_version": UNICODE_DATA_VERSION_V3,
                "semantic_descriptor_ucd_derived_core_properties_sha256": UCD13_DERIVED_CORE_PROPERTIES_SHA256_V3,
                "semantic_descriptor_ucd_prop_list_sha256": UCD13_PROP_LIST_SHA256_V3,
                "semantic_descriptor_default_ignorable_count": DEFAULT_IGNORABLE_CODEPOINT_COUNT_V3,
                "semantic_descriptor_default_ignorable_range_table_sha256": DEFAULT_IGNORABLE_RANGE_TABLE_SHA256_V3,
                "semantic_descriptor_noncharacter_policy": "REJECT_ALL_UNICODE_NONCHARACTERS",
                "source_basis": "FRESH_GENERATOR_FACING_V3_NOT_LOSSLESS_V2",
                "version": FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION,
            }
        )
    ).hexdigest()
)


def _mapping(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} closed field set drift")
    return value


def _array(value: object, name: str, *, nonempty: bool = False) -> list[object]:
    if type(value) is not list or (nonempty and not value) or len(value) > MAXIMUM_RECORDS_PER_KIND_V3:
        raise ValueError(f"{name} array shape drift")
    return value


def _rational_from_mapping_v3(value: object) -> RationalValueV3:
    row = _mapping(value, {"numerator", "denominator"}, "rational")
    return RationalValueV3(row["numerator"], row["denominator"])  # type: ignore[arg-type]


def _nullable_text_from_mapping_v3(value: object) -> NullableTextV3:
    row = _mapping(value, {"available", "value"}, "nullable text")
    return NullableTextV3(row["available"], row["value"])  # type: ignore[arg-type]


def _nullable_bool_from_mapping_v3(value: object) -> NullableBooleanV3:
    row = _mapping(value, {"available", "value"}, "nullable Boolean")
    return NullableBooleanV3(row["available"], row["value"])  # type: ignore[arg-type]


def _nullable_rationals_from_mapping_v3(value: object) -> NullableRationalVectorV3:
    row = _mapping(value, {"available", "value"}, "nullable rationals")
    raw = row["value"]
    parsed = None if raw is None else tuple(_rational_from_mapping_v3(item) for item in _array(raw, "rational vector", nonempty=True))
    return NullableRationalVectorV3(row["available"], parsed)  # type: ignore[arg-type]


def _nullable_temporal_from_mapping_v3(value: object) -> NullableTemporalSupportV3:
    row = _mapping(value, {"available", "value"}, "nullable temporal")
    raw = row["value"]
    parsed = None
    if raw is not None:
        support = _mapping(raw, {"start", "end"}, "temporal support")
        parsed = ExactTemporalSupportV3(_rational_from_mapping_v3(support["start"]), _rational_from_mapping_v3(support["end"]))
    return NullableTemporalSupportV3(row["available"], parsed)  # type: ignore[arg-type]


def _nullable_spatial_from_mapping_v3(value: object) -> NullableSpatialSupportV3:
    row = _mapping(value, {"available", "value"}, "nullable spatial")
    raw = row["value"]
    parsed = None
    if raw is not None:
        support = _mapping(raw, {"lower", "upper"}, "spatial support")
        parsed = ExactSpatialSupportV3(
            tuple(_rational_from_mapping_v3(item) for item in _array(support["lower"], "spatial lower", nonempty=True)),
            tuple(_rational_from_mapping_v3(item) for item in _array(support["upper"], "spatial upper", nonempty=True)),
        )
    return NullableSpatialSupportV3(row["available"], parsed)  # type: ignore[arg-type]


def _component_ref_from_mapping_v3(value: object) -> FormalComponentRefV3:
    row = _mapping(value, {"observation_id", "quantity_id", "context_id", "scale_id", "component_axis", "component_index"}, "component ref")
    return FormalComponentRefV3(**row)  # type: ignore[arg-type]


def _exact_term_from_mapping_v3(value: object) -> FormalExactTermV3:
    row = _mapping(value, {"input_ref", "coefficient"}, "exact term")
    return FormalExactTermV3(_component_ref_from_mapping_v3(row["input_ref"]), _rational_from_mapping_v3(row["coefficient"]))


def _exact_row_from_mapping_v3(value: object) -> FormalExactRowV3:
    row = _mapping(value, {"output_ref", "terms", "offset"}, "exact row")
    return FormalExactRowV3(
        _component_ref_from_mapping_v3(row["output_ref"]),
        tuple(_exact_term_from_mapping_v3(item) for item in _array(row["terms"], "exact terms", nonempty=True)),
        _rational_from_mapping_v3(row["offset"]),
    )


def _aggregate_group_from_mapping_v3(value: object) -> FormalAggregateGroupV3:
    row = _mapping(value, {"input_refs", "output_refs", "quotient_label"}, "aggregate group")
    return FormalAggregateGroupV3(
        tuple(_component_ref_from_mapping_v3(item) for item in _array(row["input_refs"], "aggregate input refs", nonempty=True)),
        tuple(_component_ref_from_mapping_v3(item) for item in _array(row["output_refs"], "aggregate output refs", nonempty=True)),
        row["quotient_label"],  # type: ignore[arg-type]
    )


def _certificate_from_mapping_v3(operation: object, value: object) -> FormalOperationCertificateV3:
    if type(operation) is not str or operation not in OPERATION_CERTIFICATE_TYPE_V3:
        raise ValueError("typed transform certificate operation is unknown")
    certificate_type = OPERATION_CERTIFICATE_TYPE_V3[operation]
    row = _mapping(
        value,
        {item.name for item in fields(certificate_type)},
        f"{operation} certificate",
    )
    converted = dict(row)
    for name in ("factor", "inverse_factor"):
        if name in converted:
            converted[name] = _rational_from_mapping_v3(converted[name])
    for name in (
        "inverse_rows",
        "source_commutation_rows",
        "target_commutation_rows",
    ):
        if name in converted:
            converted[name] = tuple(
                _exact_row_from_mapping_v3(item)
                for item in _array(converted[name], name)
            )
    for name in ("selected_inputs", "discarded_inputs"):
        if name in converted:
            converted[name] = tuple(
                _component_ref_from_mapping_v3(item)
                for item in _array(converted[name], name, nonempty=name == "selected_inputs")
            )
    if "grid_points" in converted:
        converted["grid_points"] = tuple(
            FormalSamplingGridPointV3(
                _component_ref_from_mapping_v3(
                    _mapping(item, {"input_ref", "coordinates"}, "sampling grid point")["input_ref"]
                ),
                tuple(
                    _rational_from_mapping_v3(atom)
                    for atom in _array(
                        _mapping(item, {"input_ref", "coordinates"}, "sampling grid point")["coordinates"],
                        "sampling grid coordinates",
                        nonempty=True,
                    )
                ),
            )
            for item in _array(converted["grid_points"], "sampling grid points", nonempty=True)
        )
    if "quotient_labels" in converted:
        converted["quotient_labels"] = tuple(converted["quotient_labels"])
    return certificate_type(**converted)  # type: ignore[return-value,arg-type]


def _graph_from_mapping_v3(value: object) -> FormalCaseGraphV3:
    keys = {item.name for item in fields(FormalCaseGraphV3)}
    row = _mapping(value, keys, "formal case graph")
    entities = tuple(
        FormalEntityRecordV3(value["entity_id"], tuple(value["role_candidate_ids"]))  # type: ignore[arg-type]
        for value in (
            _mapping(item, {"entity_id", "role_candidate_ids"}, "entity")
            for item in _array(row["entities"], "entities", nonempty=True)
        )
    )
    quantities = tuple(
        FormalQuantityRecordV3(
            value["quantity_id"],
            value["semantic_key"],
            tuple(value["si_exponents"]),
            FormalUnitBindingV3(
                _mapping(value["unit_binding"], {"semantic_key", "scale_to_si", "offset_to_si"}, "unit binding")["semantic_key"],  # type: ignore[arg-type]
                _rational_from_mapping_v3(
                    _mapping(value["unit_binding"], {"semantic_key", "scale_to_si", "offset_to_si"}, "unit binding")["scale_to_si"]
                ),
                _rational_from_mapping_v3(
                    _mapping(value["unit_binding"], {"semantic_key", "scale_to_si", "offset_to_si"}, "unit binding")["offset_to_si"]
                ),
            ),
            value["value_kind"],
        )  # type: ignore[arg-type]
        for value in (
            _mapping(item, {"quantity_id", "semantic_key", "si_exponents", "unit_binding", "value_kind"}, "quantity")
            for item in _array(row["quantities"], "quantities", nonempty=True)
        )
    )
    contexts = []
    for item in _array(row["contexts"], "contexts", nonempty=True):
        value = _mapping(item, {"context_id", "temporal_reference_kind", "clock_binding", "spatial_reference_kind", "frame_binding", "coordinate_labels", "temporal_support", "spatial_support", "orientation"}, "context")
        contexts.append(FormalContextRecordV3(
            context_id=value["context_id"],  # type: ignore[arg-type]
            temporal_reference_kind=value["temporal_reference_kind"],  # type: ignore[arg-type]
            clock_binding=value["clock_binding"],  # type: ignore[arg-type]
            spatial_reference_kind=value["spatial_reference_kind"],  # type: ignore[arg-type]
            frame_binding=value["frame_binding"],  # type: ignore[arg-type]
            coordinate_labels=tuple(value["coordinate_labels"]),  # type: ignore[arg-type]
            temporal_support=_nullable_temporal_from_mapping_v3(value["temporal_support"]),
            spatial_support=_nullable_spatial_from_mapping_v3(value["spatial_support"]),
            orientation=_nullable_text_from_mapping_v3(value["orientation"]),
        ))
    observations = []
    observation_keys = {item.name for item in fields(FormalObservationRecordV3)}
    for item in _array(row["observations"], "observations", nonempty=True):
        value = _mapping(item, observation_keys, "observation")
        observations.append(FormalObservationRecordV3(
            observation_id=value["observation_id"],  # type: ignore[arg-type]
            entity_ids=tuple(value["entity_ids"]),  # type: ignore[arg-type]
            role_candidate_ids=tuple(value["role_candidate_ids"]),  # type: ignore[arg-type]
            quantity_id=value["quantity_id"],  # type: ignore[arg-type]
            context_id=value["context_id"],  # type: ignore[arg-type]
            scale_id=value["scale_id"],  # type: ignore[arg-type]
            components=tuple(
                FormalComponentDescriptorV3(
                    **_mapping(
                        component,
                        {"label", "semantic_axis", "value_role"},
                        "component descriptor",
                    )
                )
                for component in _array(value["components"], "component descriptors", nonempty=True)
            ),  # type: ignore[arg-type]
            numeric_value=_nullable_rationals_from_mapping_v3(value["numeric_value"]),
            boolean_value=_nullable_bool_from_mapping_v3(value["boolean_value"]),
            uncertainty_model=value["uncertainty_model"],  # type: ignore[arg-type]
            uncertainty_radius=_nullable_rationals_from_mapping_v3(value["uncertainty_radius"]),
            public_provenance_commitment=value["public_provenance_commitment"],  # type: ignore[arg-type]
        ))
    task_value = _mapping(row["task_target"], {"task_id", "entity_ids", "quantity_ids"}, "task target")
    task = FormalTaskTargetV3(task_value["task_id"], tuple(task_value["entity_ids"]), tuple(task_value["quantity_ids"]))  # type: ignore[arg-type]
    descriptor_value = _mapping(
        row["semantic_descriptor"],
        {"kind", "segments"},
        "semantic descriptor",
    )
    descriptor = FormalSemanticDescriptorV3(
        descriptor_value["kind"],  # type: ignore[arg-type]
        tuple(
            _array(
                descriptor_value["segments"],
                "semantic descriptor segments",
                nonempty=True,
            )
        ),  # type: ignore[arg-type]
    )
    scales = tuple(FormalScaleRecordV3(**_mapping(item, {"scale_id", "semantic_level", "is_root"}, "scale")) for item in _array(row["scales"], "scales", nonempty=True))  # type: ignore[arg-type]
    aggregates = []
    for item in _array(row["aggregate_maps"], "aggregate maps", nonempty=True):
        value = _mapping(item, {"aggregate_map_id", "source_scale_id", "target_scale_id", "transform_id", "operation", "reducer", "groups"}, "aggregate map")
        aggregates.append(FormalAggregateMapRecordV3(
            value["aggregate_map_id"], value["source_scale_id"], value["target_scale_id"], value["transform_id"], value["operation"], value["reducer"],  # type: ignore[arg-type]
            tuple(_aggregate_group_from_mapping_v3(group) for group in _array(value["groups"], "aggregate groups", nonempty=True)),
        ))
    transforms = []
    for item in _array(row["transforms"], "transforms", nonempty=True):
        value = _mapping(item, {"transform_id", "operation", "source_scale_id", "target_scale_id", "input_refs", "output_refs", "kernel_rows", "discrete_mappings", "certificate"}, "transform")
        transforms.append(FormalTransformContractV3(
            value["transform_id"], value["operation"], value["source_scale_id"], value["target_scale_id"],  # type: ignore[arg-type]
            tuple(_component_ref_from_mapping_v3(ref) for ref in _array(value["input_refs"], "transform input refs", nonempty=True)),
            tuple(_component_ref_from_mapping_v3(ref) for ref in _array(value["output_refs"], "transform output refs", nonempty=True)),
            tuple(_exact_row_from_mapping_v3(exact_row) for exact_row in _array(value["kernel_rows"], "kernel rows")),
            tuple(
                FormalDiscreteMappingV3(
                    _component_ref_from_mapping_v3(
                        _mapping(mapping, {"input_ref", "output_ref"}, "discrete mapping")["input_ref"]
                    ),
                    _component_ref_from_mapping_v3(
                        _mapping(mapping, {"input_ref", "output_ref"}, "discrete mapping")["output_ref"]
                    ),
                )
                for mapping in _array(value["discrete_mappings"], "discrete mappings")
            ),
            _certificate_from_mapping_v3(value["operation"], value["certificate"]),
        ))
    registry_value = _mapping(row["recognizer_registry"], {"theory_policy", "family_aliases", "law_bindings", "observable_channels", "max_candidate_count"}, "recognizer registry")
    registry = FormalRecognizerRegistryV3(
        registry_value["theory_policy"],  # type: ignore[arg-type]
        tuple(FormalFamilyAliasV3(**_mapping(item, {"alias", "canonical_family"}, "family alias")) for item in _array(registry_value["family_aliases"], "family aliases", nonempty=True)),  # type: ignore[arg-type]
        tuple(FormalLawBindingV3(
            **{
                **_mapping(item, {"law_name", "family", "law_kind", "required_observation_ids", "required_role_candidate_ids", "required_quantity_ids"}, "law binding"),
                "required_observation_ids": tuple(item["required_observation_ids"]),  # type: ignore[index]
                "required_role_candidate_ids": tuple(item["required_role_candidate_ids"]),  # type: ignore[index]
                "required_quantity_ids": tuple(item["required_quantity_ids"]),  # type: ignore[index]
            }
        ) for item in _array(registry_value["law_bindings"], "law bindings", nonempty=True)),
        tuple(FormalObservableChannelV3(**_mapping(item, {"observation_id", "quantity_id"}, "observable channel")) for item in _array(registry_value["observable_channels"], "observable channels", nonempty=True)),  # type: ignore[arg-type]
        registry_value["max_candidate_count"],  # type: ignore[arg-type]
    )
    mask_value = _mapping(row["eligibility_mask"], {"observation_ids", "role_candidate_ids", "scale_ids"}, "eligibility mask")
    return FormalCaseGraphV3(
        schema_version=row["schema_version"],  # type: ignore[arg-type]
        bundle_id=row["bundle_id"],  # type: ignore[arg-type]
        entities=entities,
        role_candidate_ids=tuple(row["role_candidate_ids"]),  # type: ignore[arg-type]
        quantities=quantities,
        contexts=tuple(contexts),
        observations=tuple(observations),
        task_target=task,
        semantic_descriptor=descriptor,
        scales=scales,
        aggregate_maps=tuple(aggregates),
        transforms=tuple(transforms),
        recognizer_registry=registry,
        eligibility_mask=FormalEligibilityMaskV3(tuple(mask_value["observation_ids"]), tuple(mask_value["role_candidate_ids"]), tuple(mask_value["scale_ids"])),  # type: ignore[arg-type]
    )


def _canonicalize_graph_mapping_v3(value: dict[str, object], *, preserve_observation_order: bool) -> dict[str, object]:
    _mapping(
        value,
        {item.name for item in fields(FormalCaseGraphV3)},
        "formal graph canonicalization input",
    )
    result = deepcopy(value)
    def ref_key(ref: dict[str, object]) -> tuple[object, ...]:
        return (
            ref["scale_id"], ref["observation_id"], ref["quantity_id"],
            ref["context_id"], ref["component_axis"], ref["component_index"],
        )
    sort_fields = {
        "entities": "entity_id", "quantities": "quantity_id", "contexts": "context_id",
        "scales": "scale_id", "aggregate_maps": "aggregate_map_id", "transforms": "transform_id",
    }
    for name, key in sort_fields.items():
        result[name].sort(key=lambda item: item[key])  # type: ignore[union-attr,index]
    if not preserve_observation_order:
        result["observations"].sort(key=lambda item: item["observation_id"])  # type: ignore[union-attr,index]
    result["role_candidate_ids"].sort()  # type: ignore[union-attr]
    for entity in result["entities"]: entity["role_candidate_ids"].sort()  # type: ignore[union-attr,index]
    for observation in result["observations"]:  # type: ignore[union-attr]
        observation["entity_ids"].sort(); observation["role_candidate_ids"].sort()
    result["task_target"]["entity_ids"].sort(); result["task_target"]["quantity_ids"].sort()  # type: ignore[index,union-attr]
    for aggregate in result["aggregate_maps"]:  # type: ignore[union-attr]
        transform = next(
            (
                item
                for item in result["transforms"]  # type: ignore[union-attr]
                if item["transform_id"] == aggregate["transform_id"]
            ),
            None,
        )
        if transform is None:
            raise ValueError("aggregate-map transform is absent during canonicalization")
        certificate = transform["certificate"]
        if transform["operation"] == "coarse_graining":
            # Preserve the explicit group/output-label association before UUID
            # relabeling changes lexical reference order.  A mismatched input
            # is rejected; canonicalization never repairs a forged positional
            # certificate silently.
            expected_labels = tuple(
                group["quotient_label"] for group in aggregate["groups"]
            )
            if tuple(certificate["quotient_labels"]) != expected_labels:
                raise ValueError(
                    "coarse quotient labels do not bind the presented groups"
                )
        for group in aggregate["groups"]:
            group["input_refs"].sort(key=ref_key); group["output_refs"].sort(key=ref_key)
        aggregate["groups"].sort(key=lambda group: (tuple(ref_key(ref) for ref in group["input_refs"]), tuple(ref_key(ref) for ref in group["output_refs"]), group["quotient_label"]))
        if transform["operation"] == "coarse_graining":
            certificate["quotient_labels"] = [
                group["quotient_label"] for group in aggregate["groups"]
            ]
    for transform in result["transforms"]:  # type: ignore[union-attr]
        transform["input_refs"].sort(key=ref_key); transform["output_refs"].sort(key=ref_key)
        for exact_row in transform["kernel_rows"]:
            exact_row["terms"].sort(key=lambda term: ref_key(term["input_ref"]))
        transform["kernel_rows"].sort(key=lambda row: ref_key(row["output_ref"]))
        transform["discrete_mappings"].sort(
            key=lambda row: (ref_key(row["input_ref"]), ref_key(row["output_ref"]))
        )
        certificate = transform["certificate"]
        for name in (
            "inverse_rows", "source_commutation_rows", "target_commutation_rows"
        ):
            for exact_row in certificate.get(name, []):
                exact_row["terms"].sort(key=lambda term: ref_key(term["input_ref"]))
            certificate.get(name, []).sort(key=lambda row: ref_key(row["output_ref"]))
        if "discarded_inputs" in certificate:
            certificate["discarded_inputs"].sort(key=ref_key)
    registry = result["recognizer_registry"]
    registry["family_aliases"].sort(key=lambda item: item["alias"])  # type: ignore[index,union-attr]
    registry["law_bindings"].sort(key=lambda item: item["law_name"])  # type: ignore[index,union-attr]
    registry["observable_channels"].sort(key=lambda item: item["observation_id"])  # type: ignore[index,union-attr]
    for law in registry["law_bindings"]:  # type: ignore[index]
        law["required_observation_ids"].sort(); law["required_role_candidate_ids"].sort(); law["required_quantity_ids"].sort()
    mask = result["eligibility_mask"]
    mask["observation_ids"].sort(); mask["role_candidate_ids"].sort(); mask["scale_ids"].sort()  # type: ignore[index]
    return result


def _refresh_graph_provenance_v3(value: dict[str, object], global_root: str) -> None:
    observations = value["observations"]
    for observation in observations:  # type: ignore[union-attr]
        body = deepcopy(observation)
        body.pop("public_provenance_commitment", None)
        observation["public_provenance_commitment"] = "sha256:" + hashlib.sha256(
            _OBS_PROVENANCE_DOMAIN_V3 + canonicalize_rfc8785_v1({"transform_catalog_root": global_root, "observation": body})
        ).hexdigest()


_ID_FIELD_NAMESPACE_V3: Final = {
    "bundle_id": "bundle_id",
    "observation_id": "observation_id", "observation_ids": "observation_id", "required_observation_ids": "observation_id",
    "entity_id": "entity_id", "entity_ids": "entity_id",
    "role_candidate_id": "role_candidate_id", "role_candidate_ids": "role_candidate_id", "required_role_candidate_ids": "role_candidate_id",
    "quantity_id": "quantity_id", "quantity_ids": "quantity_id", "required_quantity_ids": "quantity_id",
    "context_id": "context_id", "context_ids": "context_id",
    "task_id": "task_id", "task_ids": "task_id",
    "scale_id": "scale_id", "scale_ids": "scale_id", "source_scale_id": "scale_id", "target_scale_id": "scale_id",
    "aggregate_map_id": "aggregate_map_id", "aggregate_map_ids": "aggregate_map_id",
    "quotient_aggregate_map_id": "aggregate_map_id",
    "transform_id": "transform_id", "transform_ids": "transform_id",
}


def _walk_uuid_leaves_v3(value: object, *, field_name: str | None = None, pointer: str = "") -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    namespace = _ID_FIELD_NAMESPACE_V3.get(field_name or "")
    if type(value) is dict:
        for key, item in value.items():
            rows.extend(_walk_uuid_leaves_v3(item, field_name=key, pointer=pointer + "/" + key))
    elif type(value) is list:
        for index, item in enumerate(value):
            rows.extend(_walk_uuid_leaves_v3(item, field_name=field_name, pointer=pointer + "/" + str(index)))
    elif type(value) is str and _UUID_SHAPED_RE.fullmatch(value):
        if namespace is None or _UUID4_RE.fullmatch(value) is None:
            raise ValueError("UUID-shaped leaf is outside the frozen exact-10 namespace closure")
        rows.append((pointer, namespace, value))
    return tuple(rows)


def _rename_graph_ids_v3(value: dict[str, object], renamings: dict[tuple[str, str], str]) -> dict[str, object]:
    result = deepcopy(value)
    def visit(current: object, field_name: str | None = None) -> object:
        namespace = _ID_FIELD_NAMESPACE_V3.get(field_name or "")
        if type(current) is dict:
            return {key: visit(item, key) for key, item in current.items()}
        if type(current) is list:
            return [visit(item, field_name) for item in current]
        if type(current) is str and _UUID_SHAPED_RE.fullmatch(current):
            if namespace is None:
                raise ValueError("unclassified UUID-shaped formal graph leaf")
            return renamings[(namespace, current)]
        return current
    renamed = visit(result)
    if type(renamed) is not dict:
        raise TypeError("renamed formal graph root type drift")
    return renamed


class FormalTrustedWireDispositionV3(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


class PublicLeafClassV3(str, Enum):
    EVIDENCE_BEARING = "EVIDENCE_BEARING"
    CHANNEL_ONLY = "CHANNEL_ONLY"


@dataclass(frozen=True, slots=True)
class ClassifiedPublicLeafV3:
    json_pointer: str
    classification: PublicLeafClassV3
    value: object


def _walk_leaves_v3(value: object, pointer: str = "") -> tuple[tuple[str, object], ...]:
    rows: list[tuple[str, object]] = []
    if type(value) is dict:
        for key in sorted(value):
            rows.extend(_walk_leaves_v3(value[key], pointer + "/" + key))
    elif type(value) is list:
        for index, item in enumerate(value):
            rows.extend(_walk_leaves_v3(item, pointer + "/" + str(index)))
    else:
        rows.append((pointer, value))
    return tuple(rows)


_CHANNEL_LAST_SEGMENTS_V3: Final = frozenset(
    {
        "available", "case_count_decimal", "component_index",
        "field_classification_manifest_id", "graph_completeness_policy_id",
        "id_collision_retry_count_decimal", "max_candidate_count",
        "public_provenance_commitment", "run_id_commitment", "schema_version",
    }
)


def classify_public_payload_leaves_v3(payload: dict[str, object]) -> tuple[ClassifiedPublicLeafV3, ...]:
    if type(payload) is not dict:
        raise TypeError("formal public payload must be exact dict")
    result: list[ClassifiedPublicLeafV3] = []
    for pointer, value in _walk_leaves_v3(payload):
        last = pointer.rsplit("/", 1)[-1]
        channel = (
            not pointer.startswith("/public_case/graph/")
            or "/eligibility_mask/" in pointer
            or last in _CHANNEL_LAST_SEGMENTS_V3
            or value is None
            or (type(value) is str and _UUID4_RE.fullmatch(value) is not None)
        )
        result.append(
            ClassifiedPublicLeafV3(
                pointer,
                PublicLeafClassV3.CHANNEL_ONLY if channel else PublicLeafClassV3.EVIDENCE_BEARING,
                value,
            )
        )
    if len({row.json_pointer for row in result}) != len(result):
        raise RuntimeError("concrete public leaf classification repeated a pointer")
    return tuple(result)


FIELD_CLASSIFICATION_MANIFEST_ID_V3: Final = (
    "phase2b_formal_wire_field_manifest_v3_"
    + hashlib.sha256(
        _POLICY_DOMAIN_V3
        + canonicalize_rfc8785_v1(
            {
                "channel_last_segments": sorted(_CHANNEL_LAST_SEGMENTS_V3),
                "eligibility_mask": "CHANNEL_ONLY",
                "missing_nullable_null": "CHANNEL_ONLY",
                "namespace_field_manifest": dict(sorted(_ID_FIELD_NAMESPACE_V3.items())),
                "namespace_uuid_leaves": "CHANNEL_ONLY",
                "other_closed_graph_leaves": "EVIDENCE_BEARING",
                "outside_graph": "CHANNEL_ONLY",
                "path_pattern_precedence": [
                    "outside_public_case_graph",
                    "eligibility_mask_subtree",
                    "channel_last_segment",
                    "null_leaf",
                    "uuid_leaf",
                    "closed_graph_evidence_default",
                ],
                "public_dto_exact_field_manifest": {
                    key: list(value) for key, value in sorted(FORMAL_PUBLIC_DTO_FIELD_MANIFEST_V3.items())
                },
                "version": FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,
            }
        )
    ).hexdigest()
)


@dataclass(frozen=True, slots=True)
class NamespaceUuidChannelV3:
    json_pointer: str
    namespace: str
    public_uuid: str
    uuid_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.json_pointer) is not str or not self.json_pointer.startswith("/public_case/graph/"):
            raise ValueError("namespace channel needs an absolute public-graph JSON pointer")
        if self.namespace not in NAMESPACE_REGISTRY_V3:
            raise ValueError("namespace channel is outside the frozen exact-10 registry")
        _uuid(self.public_uuid, "namespace channel public UUID")
        expected = bytes.fromhex(self.public_uuid.replace("-", ""))
        if type(self.uuid_bytes) is not bytes or len(self.uuid_bytes) != 16 or self.uuid_bytes != expected:
            raise ValueError("namespace channel UUID bytes/text mismatch")


@dataclass(frozen=True, slots=True)
class MissingnessChannelV3:
    json_pointer: str
    missing: bool

    def __post_init__(self) -> None:
        if type(self.json_pointer) is not str or not self.json_pointer.startswith("/public_case/graph/"):
            raise ValueError("missingness channel needs an absolute public-graph JSON pointer")
        if type(self.missing) is not bool:
            raise TypeError("missingness channel value must be exact bool")


@dataclass(frozen=True, slots=True)
class SemanticDescriptorChannelV3:
    kind: str
    segment_count: int
    segment_utf8_lengths: tuple[int, ...]
    segment_trailing_u0020_counts: tuple[int, ...]
    segment_boundary_codepoints: tuple[tuple[int, int], ...]
    segment_lexical_token_counts: tuple[int, ...]
    segment_non_alnum_codepoint_sequences: tuple[tuple[int, ...], ...]
    segment_category_run_signatures: tuple[tuple[str, ...], ...]
    segment_order_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in SEMANTIC_DESCRIPTOR_KINDS_V3:
            raise ValueError("semantic descriptor channel kind drift")
        if type(self.segment_count) is not int or not 1 <= self.segment_count <= MAXIMUM_SEMANTIC_DESCRIPTOR_SEGMENTS_V3:
            raise ValueError("semantic descriptor channel count drift")
        for name in (
            "segment_utf8_lengths",
            "segment_trailing_u0020_counts",
            "segment_boundary_codepoints",
            "segment_lexical_token_counts",
            "segment_non_alnum_codepoint_sequences",
            "segment_category_run_signatures",
        ):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != self.segment_count:
                raise ValueError(f"semantic descriptor channel {name} length drift")
        if any(type(value) is not int or value < 0 for value in self.segment_utf8_lengths):
            raise TypeError("semantic descriptor UTF-8 length channel drift")
        if any(type(value) is not int or value < 0 for value in self.segment_trailing_u0020_counts):
            raise TypeError("semantic descriptor trailing-space channel drift")
        if any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not int or not 0 <= value <= 0x10FFFF for value in pair)
            for pair in self.segment_boundary_codepoints
        ):
            raise TypeError("semantic descriptor boundary channel drift")
        if any(type(value) is not int or value < 0 for value in self.segment_lexical_token_counts):
            raise TypeError("semantic descriptor lexical-token channel drift")
        if any(
            type(row) is not tuple
            or any(type(value) is not int or not 0 <= value <= 0x10FFFF for value in row)
            for row in self.segment_non_alnum_codepoint_sequences
        ):
            raise TypeError("semantic descriptor format-scalar channel drift")
        if any(
            type(row) is not tuple
            or any(type(value) is not str or not value for value in row)
            for row in self.segment_category_run_signatures
        ):
            raise TypeError("semantic descriptor category-run channel drift")
        _digest(self.segment_order_sha256, "semantic descriptor order channel digest")


@dataclass(frozen=True, slots=True)
class ChannelProjectionV3:
    payload_length: int
    padding_length: int
    padding_sha256: str
    padding_prefix_32: bytes
    padding_suffix_32: bytes
    namespace_uuid_channels: tuple[NamespaceUuidChannelV3, ...]
    missingness_channels: tuple[MissingnessChannelV3, ...]
    missingness_bits: tuple[bool, ...]
    observation_array_paths: tuple[str, ...]
    record_order_tokens: tuple[str, ...]
    component_slot_order_tokens: tuple[str, ...]
    eligibility_mask_tokens: tuple[str, ...]
    semantic_descriptor_channel: SemanticDescriptorChannelV3
    public_provenance_commitment: str

    def __post_init__(self) -> None:
        if type(self.payload_length) is not int or not 0 < self.payload_length <= MAXIMUM_PAYLOAD_BYTES_V3:
            raise ValueError("channel payload length is outside the fixed envelope")
        if type(self.padding_length) is not int or self.padding_length < MINIMUM_PADDING_BYTES_V3:
            raise ValueError("channel padding length is outside the fixed envelope")
        if self.payload_length + self.padding_length + ENVELOPE_HEADER_BYTES_V3 != ENVELOPE_BYTES_V3:
            raise ValueError("channel payload/padding lengths do not close the fixed envelope")
        _digest(self.padding_sha256, "channel padding digest")
        if (
            type(self.padding_prefix_32) is not bytes
            or type(self.padding_suffix_32) is not bytes
            or len(self.padding_prefix_32) != 32
            or len(self.padding_suffix_32) != 32
        ):
            raise ValueError("channel padding prefix/suffix must be exact 32-byte values")
        if type(self.namespace_uuid_channels) is not tuple or any(
            type(row) is not NamespaceUuidChannelV3 for row in self.namespace_uuid_channels
        ):
            raise TypeError("namespace UUID channels must use exact immutable rows")
        for row in self.namespace_uuid_channels:
            row.__post_init__()
        if len({row.json_pointer for row in self.namespace_uuid_channels}) != len(self.namespace_uuid_channels):
            raise ValueError("namespace UUID channel repeats a JSON pointer")
        if type(self.missingness_channels) is not tuple or any(
            type(row) is not MissingnessChannelV3 for row in self.missingness_channels
        ):
            raise TypeError("missingness channels must use exact immutable rows")
        for row in self.missingness_channels:
            row.__post_init__()
        if len({row.json_pointer for row in self.missingness_channels}) != len(self.missingness_channels):
            raise ValueError("missingness channel repeats a JSON pointer")
        if type(self.missingness_bits) is not tuple or self.missingness_bits != tuple(
            row.missing for row in self.missingness_channels
        ):
            raise ValueError("missingness bit compatibility view does not match pointer channels")
        if self.observation_array_paths != PUBLIC_OBSERVATION_ARRAY_PATHS_V3:
            raise ValueError("observation array path manifest drift")
        for name in (
            "record_order_tokens", "component_slot_order_tokens", "eligibility_mask_tokens"
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(type(value) is not str or not value for value in values):
                raise TypeError(f"{name} must be an exact tuple of nonempty strings")
        if type(self.semantic_descriptor_channel) is not SemanticDescriptorChannelV3:
            raise TypeError("semantic descriptor channel projection type drift")
        self.semantic_descriptor_channel.__post_init__()
        _digest(self.public_provenance_commitment, "channel public provenance commitment")


@dataclass(frozen=True, slots=True)
class FormalWireKeySourcesV3:
    shuffle_ikm: bytes = field(repr=False)
    id_ikm: bytes = field(repr=False)
    padding_ikm: bytes = field(repr=False)

    def __post_init__(self) -> None:
        values = (self.shuffle_ikm, self.id_ikm, self.padding_ikm)
        if any(type(value) is not bytes or len(value) != IKM_BYTES_V3 for value in values):
            raise ValueError("formal wire key sources must be exact 32-byte values")
        if len(set(values)) != 3:
            raise ValueError("shuffle, ID, and padding IKM values must differ")


@dataclass(frozen=True, slots=True)
class KeySourceTranscriptReceiptV3:
    schema_version: str
    shuffle_seed_commitment: str
    id_seed_commitment: str
    padding_seed_commitment: str
    latent_seed_commitments: tuple[str, ...]
    answer_seed_commitments: tuple[str, ...]
    receipt_id: str

    def __post_init__(self) -> None:
        if self.schema_version != KEY_SOURCE_TRANSCRIPT_V3_SCHEMA_VERSION:
            raise ValueError("key-source transcript schema drift")
        for value in (self.shuffle_seed_commitment, self.id_seed_commitment, self.padding_seed_commitment): _digest(value, "key seed commitment")
        for name in ("latent_seed_commitments", "answer_seed_commitments"):
            values = getattr(self, name)
            if type(values) is not tuple or not values or values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise ValueError(f"{name} must be sorted nonempty unique")
            for value in values: _digest(value, name)
        _prefixed_digest(self.receipt_id, "phase2b_key_source_transcript_v3_", "key receipt ID")


def seed_commitment_v3(seed: bytes) -> str:
    if type(seed) is not bytes or not seed:
        raise TypeError("seed commitment needs exact nonempty bytes")
    return "sha256:" + hashlib.sha256(seed).hexdigest()


def _key_receipt_body_v3(keys: FormalWireKeySourcesV3, latent: tuple[str, ...], answer: tuple[str, ...]) -> dict[str, object]:
    return {
        "answer_seed_commitments": list(answer),
        "id_seed_commitment": seed_commitment_v3(keys.id_ikm),
        "latent_seed_commitments": list(latent),
        "padding_seed_commitment": seed_commitment_v3(keys.padding_ikm),
        "schema_version": KEY_SOURCE_TRANSCRIPT_V3_SCHEMA_VERSION,
        "shuffle_seed_commitment": seed_commitment_v3(keys.shuffle_ikm),
    }


def issue_key_source_transcript_receipt_v3(*, key_sources: FormalWireKeySourcesV3, latent_seed_commitments: tuple[str, ...], answer_seed_commitments: tuple[str, ...]) -> KeySourceTranscriptReceiptV3:
    if type(key_sources) is not FormalWireKeySourcesV3: raise TypeError("key source exact type drift")
    key_sources.__post_init__()
    latent, answer = tuple(sorted(latent_seed_commitments)), tuple(sorted(answer_seed_commitments))
    for values in (latent, answer):
        if not values or len(set(values)) != len(values): raise ValueError("seed commitment classes must be nonempty unique")
        for value in values: _digest(value, "seed commitment")
    body = _key_receipt_body_v3(key_sources, latent, answer)
    receipt_id = "phase2b_key_source_transcript_v3_" + hashlib.sha256(_KEY_RECEIPT_DOMAIN_V3 + canonicalize_rfc8785_v1(body)).hexdigest()
    return KeySourceTranscriptReceiptV3(
        KEY_SOURCE_TRANSCRIPT_V3_SCHEMA_VERSION,
        body["shuffle_seed_commitment"], body["id_seed_commitment"], body["padding_seed_commitment"],  # type: ignore[arg-type]
        latent, answer, receipt_id,
    )


def _verify_key_receipt_v3(keys: FormalWireKeySourcesV3, receipt: KeySourceTranscriptReceiptV3) -> None:
    keys.__post_init__(); receipt.__post_init__()
    body = _key_receipt_body_v3(keys, receipt.latent_seed_commitments, receipt.answer_seed_commitments)
    expected = "phase2b_key_source_transcript_v3_" + hashlib.sha256(_KEY_RECEIPT_DOMAIN_V3 + canonicalize_rfc8785_v1(body)).hexdigest()
    if receipt.receipt_id != expected or (
        receipt.shuffle_seed_commitment,
        receipt.id_seed_commitment,
        receipt.padding_seed_commitment,
    ) != (
        body["shuffle_seed_commitment"], body["id_seed_commitment"], body["padding_seed_commitment"]
    ):
        raise ValueError("key-source transcript does not bind supplied keys")
    all_values = (
        receipt.shuffle_seed_commitment, receipt.id_seed_commitment,
        receipt.padding_seed_commitment, *receipt.latent_seed_commitments,
        *receipt.answer_seed_commitments,
    )
    if len(set(all_values)) != len(all_values):
        raise ValueError("key/latent/answer commitments overlap")


@dataclass(frozen=True, slots=True)
class FormalBatchContextV3:
    case_count_decimal: str
    transform_catalog_root: str
    id_collision_retry_count_decimal: str
    run_id_commitment: str

    def __post_init__(self) -> None:
        for name in ("case_count_decimal", "id_collision_retry_count_decimal"):
            value = getattr(self, name)
            if type(value) is not str or _UNSIGNED_DECIMAL_RE.fullmatch(value) is None: raise ValueError(f"{name} must be canonical ASCII unsigned decimal string")
        if not 1 <= int(self.case_count_decimal) <= MAXIMUM_BATCH_CASES_V3: raise ValueError("batch case count out of range")
        maximum_retries = (
            int(self.case_count_decimal)
            * len(NAMESPACE_REGISTRY_V3)
            * MAXIMUM_RECORDS_PER_KIND_V3
            * MAXIMUM_UUID_COLLISION_RETRIES_V3
        )
        if int(self.id_collision_retry_count_decimal) > maximum_retries:
            raise ValueError("batch collision retry count exceeds the realizable schema bound")
        _digest(self.transform_catalog_root, "global transform catalog root")
        _digest(self.run_id_commitment, "run ID commitment")
        _validate_routed_public_strings_v3(self)


def _context_mapping_v3(value: FormalBatchContextV3) -> dict[str, object]:
    value.__post_init__()
    return {item.name: getattr(value, item.name) for item in fields(type(value))}


def _context_from_mapping_v3(value: object) -> FormalBatchContextV3:
    row = _mapping(value, {item.name for item in fields(FormalBatchContextV3)}, "batch context")
    return FormalBatchContextV3(**row)  # type: ignore[arg-type]


def _public_case_mapping_v3(case: FormalPublicCaseV3) -> dict[str, object]:
    return {
        "graph": _profile_value_v3(case.graph),
        "public_provenance_commitment": case.public_provenance_commitment,
        "schema_version": case.schema_version,
    }


def _payload_mapping_v3(context: FormalBatchContextV3, case: FormalPublicCaseV3) -> dict[str, object]:
    return {
        "batch_context": _context_mapping_v3(context),
        "field_classification_manifest_id": FIELD_CLASSIFICATION_MANIFEST_ID_V3,
        "graph_completeness_policy_id": FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3,
        "public_case": _public_case_mapping_v3(case),
        "schema_version": FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,
    }


def _reject_forbidden_v3(value: object) -> None:
    stack = [value]; forbidden = set(PUBLIC_PAYLOAD_MUST_BE_ABSENT_FIELDS_V3)
    while stack:
        current = stack.pop()
        if type(current) is dict:
            if any(key in forbidden for key in current): raise ValueError("public payload contains MUST_BE_ABSENT field")
            stack.extend(current.values())
        elif type(current) is list: stack.extend(current)


def _restore_json_integers_v3(value: object) -> object:
    if type(value) is float:
        if not value.is_integer(): raise ValueError("formal V3 payload contains a raw binary64")
        return int(value)
    if type(value) is list: return [_restore_json_integers_v3(item) for item in value]
    if type(value) is dict: return {key: _restore_json_integers_v3(item) for key, item in value.items()}
    return value


def _bounded_json_wire_preflight_v3(payload: bytes) -> None:
    """Bound nesting iteratively before any recursive JSON implementation runs."""
    if type(payload) is not bytes or not payload:
        raise ValueError("formal JSON payload must be exact nonempty bytes")
    depth = 0
    containers = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            containers += 1
            if (
                depth > MAXIMUM_JSON_NESTING_DEPTH_V3
                or containers > MAXIMUM_JSON_CONTAINER_TOKENS_V3
            ):
                raise ValueError("formal JSON nesting/container budget exceeded")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise ValueError("formal JSON container balance drift")
    if in_string or escaped or depth != 0:
        raise ValueError("formal JSON lexical/container closure drift")


def _normalize_formal_payload_structure_v3(payload: dict[str, object]) -> dict[str, object]:
    """Internal structural normalizer; it does not prove catalog membership."""
    if type(payload) is not dict: raise TypeError("variant payload must be exact dict")
    value = deepcopy(payload)
    expected = {"batch_context", "field_classification_manifest_id", "graph_completeness_policy_id", "public_case", "schema_version"}
    _mapping(value, expected, "formal payload"); _reject_forbidden_v3(value)
    if value["schema_version"] != FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION or value["field_classification_manifest_id"] != FIELD_CLASSIFICATION_MANIFEST_ID_V3 or value["graph_completeness_policy_id"] != FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3:
        raise ValueError("formal payload identity drift")
    context = _context_from_mapping_v3(value["batch_context"])
    public = _mapping(value["public_case"], {"graph", "public_provenance_commitment", "schema_version"}, "public case")
    if public["schema_version"] != FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION: raise ValueError("public case schema drift")
    if type(public["graph"]) is not dict: raise TypeError("public graph must be exact dict")
    graph_mapping = _canonicalize_graph_mapping_v3(public["graph"], preserve_observation_order=True)
    uuid_namespace: dict[str, str] = {}
    for _, namespace, public_uuid in _walk_uuid_leaves_v3(graph_mapping):
        previous = uuid_namespace.setdefault(public_uuid, namespace)
        if previous != namespace:
            raise ValueError("one public UUID is reused across namespaces in one envelope")
    _refresh_graph_provenance_v3(graph_mapping, context.transform_catalog_root)
    graph = _graph_from_mapping_v3(graph_mapping)
    if any(
        observation.uncertainty_model == "standard_error"
        for observation in graph.observations
    ):
        raise ValueError("STANDARD_ERROR cannot enter a formal public payload")
    normalized_graph = _profile_value_v3(graph)
    case_commitment = "sha256:" + hashlib.sha256(
        _CASE_PROVENANCE_DOMAIN_V3
        + canonicalize_rfc8785_v1({"transform_catalog_root": context.transform_catalog_root, "graph": normalized_graph})
    ).hexdigest()
    result = {
        "batch_context": _context_mapping_v3(context),
        "field_classification_manifest_id": FIELD_CLASSIFICATION_MANIFEST_ID_V3,
        "graph_completeness_policy_id": FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3,
        "public_case": {
            "graph": normalized_graph,
            "public_provenance_commitment": case_commitment,
            "schema_version": FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION,
        },
        "schema_version": FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,
    }
    classify_public_payload_leaves_v3(result)
    return result


@dataclass(frozen=True, slots=True)
class DecodedFormalEnvelopeV3:
    envelope: bytes
    envelope_id: str
    payload: dict[str, object]
    payload_bytes: bytes
    padding_bytes: bytes
    payload_length: int
    padding_length: int
    payload_sha256: str
    padding_sha256: str
    batch_context: FormalBatchContextV3
    public_case: FormalPublicCaseV3
    classified_leaves: tuple[ClassifiedPublicLeafV3, ...]
    channel_projection: ChannelProjectionV3
    structural_replay_verified: bool
    catalog_membership_verified: bool
    formal_covert_audit: bool
    actual_execution_claimed: bool

    def __post_init__(self) -> None:
        if type(self.envelope) is not bytes or len(self.envelope) != ENVELOPE_BYTES_V3:
            raise ValueError("decoded envelope cache has wrong bytes/length")
        if type(self.payload) is not dict or type(self.payload_bytes) is not bytes or type(self.padding_bytes) is not bytes:
            raise TypeError("decoded envelope cache has mutable-type drift")
        if (
            type(self.payload_length) is not int
            or type(self.padding_length) is not int
            or self.payload_length != len(self.payload_bytes)
            or self.padding_length != len(self.padding_bytes)
            or self.payload_length + self.padding_length + ENVELOPE_HEADER_BYTES_V3 != ENVELOPE_BYTES_V3
        ):
            raise ValueError("decoded envelope cached lengths disagree")
        if self.envelope != self.envelope[:ENVELOPE_HEADER_BYTES_V3] + self.payload_bytes + self.padding_bytes:
            raise ValueError("decoded envelope cached byte slices disagree")
        magic, version, header_bytes, header_payload_length, payload_digest, padding_digest = _ENVELOPE_HEADER_V3.unpack(
            self.envelope[:ENVELOPE_HEADER_BYTES_V3]
        )
        if (
            (magic, version, header_bytes)
            != (ENVELOPE_MAGIC_V3, ENVELOPE_VERSION_V3, ENVELOPE_HEADER_BYTES_V3)
            or header_payload_length != self.payload_length
            or payload_digest != hashlib.sha256(self.payload_bytes).digest()
            or padding_digest != hashlib.sha256(self.padding_bytes).digest()
        ):
            raise ValueError("decoded envelope cached header does not freshly replay")
        _bounded_json_wire_preflight_v3(self.payload_bytes)
        try:
            reparsed = _restore_json_integers_v3(
                decode_and_require_canonical_rfc8785_v1(self.payload_bytes)
            )
        except (RecursionError, MemoryError):
            raise ValueError("bounded decoded-envelope cache replay rejected") from None
        if reparsed != self.payload:
            raise ValueError("decoded envelope cached payload does not freshly decode")
        try:
            normalized = _normalize_formal_payload_structure_v3(deepcopy(reparsed))
            reconstructed_context = _context_from_mapping_v3(
                normalized["batch_context"]
            )
            reconstructed_public_mapping = _mapping(
                normalized["public_case"],
                {"graph", "public_provenance_commitment", "schema_version"},
                "decoded cached public case",
            )
            reconstructed_graph = _graph_from_mapping_v3(
                reconstructed_public_mapping["graph"]
            )
            reconstructed_public_case = FormalPublicCaseV3(
                reconstructed_public_mapping["schema_version"],
                reconstructed_graph,
                reconstructed_public_mapping["public_provenance_commitment"],
            )
        except (RecursionError, MemoryError):
            raise ValueError("bounded decoded-envelope closed replay rejected") from None
        if normalized != reparsed:
            raise ValueError("decoded envelope cached payload is not normalized")
        if canonicalize_rfc8785_v1(self.payload) != self.payload_bytes:
            raise ValueError("decoded envelope cached payload is not its canonical wire bytes")
        expected_payload_sha = "sha256:" + hashlib.sha256(self.payload_bytes).hexdigest()
        expected_padding_sha = "sha256:" + hashlib.sha256(self.padding_bytes).hexdigest()
        if self.payload_sha256 != expected_payload_sha or self.padding_sha256 != expected_padding_sha:
            raise ValueError("decoded envelope cached digest mismatch")
        expected_id = "phase2b_formal_trusted_envelope_v3_" + hashlib.sha256(
            _ENVELOPE_ID_DOMAIN_V3 + self.envelope
        ).hexdigest()
        if self.envelope_id != expected_id:
            raise ValueError("decoded envelope cached ID mismatch")
        if type(self.batch_context) is not FormalBatchContextV3 or type(self.public_case) is not FormalPublicCaseV3:
            raise TypeError("decoded envelope cached formal types drift")
        if (
            self.batch_context != reconstructed_context
            or self.public_case != reconstructed_public_case
        ):
            raise ValueError("decoded envelope cached context/public case is payload-spliced")
        expected_classified = classify_public_payload_leaves_v3(normalized)
        if type(self.classified_leaves) is not tuple or self.classified_leaves != expected_classified:
            raise ValueError("decoded envelope cached leaf classification mismatch")
        if type(self.channel_projection) is not ChannelProjectionV3:
            raise TypeError("decoded envelope cached channel projection type drift")
        expected_projection = _channel_projection_v3(
            reconstructed_public_case, self.payload_length, self.padding_bytes
        )
        if self.channel_projection != expected_projection:
            raise ValueError("decoded envelope cached channel projection mismatch")
        if (
            self.structural_replay_verified is not True
            or self.catalog_membership_verified is not False
            or self.formal_covert_audit is not False
            or self.actual_execution_claimed is not False
        ):
            raise ValueError("decoded envelope claim boundary drift")


@dataclass(frozen=True, slots=True)
class FormalTrustedWireEnvelopeV3:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    padding_sha256: str
    payload_length: int
    padding_length: int
    public_case: FormalPublicCaseV3
    channel_projection: ChannelProjectionV3
    structural_replay_verified: bool
    formal_covert_audit: bool
    actual_execution_claimed: bool

    def __post_init__(self) -> None:
        decoded = decode_and_audit_formal_envelope_v3(self.envelope)
        expected = (
            decoded.envelope_id, decoded.payload_sha256, decoded.padding_sha256,
            decoded.payload_length, decoded.padding_length, decoded.public_case,
            decoded.channel_projection,
        )
        actual = (
            self.envelope_id, self.payload_sha256, self.padding_sha256,
            self.payload_length, self.padding_length, self.public_case,
            self.channel_projection,
        )
        if actual != expected:
            raise ValueError("trusted envelope cached public fields do not replay")
        if (
            self.structural_replay_verified is not True
            or self.formal_covert_audit is not False
            or self.actual_execution_claimed is not False
        ):
            raise ValueError("trusted envelope claim boundary drift")


def _verify_trusted_envelope_cache_v3(
    row: FormalTrustedWireEnvelopeV3,
    decoded: DecodedFormalEnvelopeV3,
) -> None:
    expected = (
        decoded.envelope_id,
        decoded.payload_sha256,
        decoded.padding_sha256,
        decoded.payload_length,
        decoded.padding_length,
        decoded.public_case,
        decoded.channel_projection,
    )
    actual = (
        row.envelope_id,
        row.payload_sha256,
        row.padding_sha256,
        row.payload_length,
        row.padding_length,
        row.public_case,
        row.channel_projection,
    )
    if actual != expected:
        raise ValueError("trusted envelope cached public fields do not replay")
    if (
        row.structural_replay_verified is not True
        or row.formal_covert_audit is not False
        or row.actual_execution_claimed is not False
    ):
        raise ValueError("trusted envelope claim boundary drift")


def _verify_decoded_batch_closure_v3(
    decoded: tuple[DecodedFormalEnvelopeV3, ...],
    expected_context: FormalBatchContextV3,
) -> None:
    """Fresh context, global UUID ownership, and retry closure for COMPLETE DTOs."""
    if type(decoded) is not tuple or not decoded or any(
        type(row) is not DecodedFormalEnvelopeV3 for row in decoded
    ):
        raise TypeError("fresh batch closure needs exact decoded envelopes")
    expected_context.__post_init__()
    if (
        len(decoded) != int(expected_context.case_count_decimal)
        or any(row.batch_context != expected_context for row in decoded)
    ):
        raise ValueError("fresh batch envelope contexts disagree")
    uuid_owner: dict[str, tuple[str, int]] = {}
    for public_row_index, row in enumerate(decoded):
        for channel in row.channel_projection.namespace_uuid_channels:
            previous = uuid_owner.setdefault(
                channel.public_uuid, (channel.namespace, public_row_index)
            )
            if previous[0] != channel.namespace:
                raise ValueError("fresh batch public UUID crosses namespaces")
            if previous[1] != public_row_index:
                raise ValueError("fresh batch public UUID is reused across envelopes")
    if int(expected_context.id_collision_retry_count_decimal) > (
        len(uuid_owner) * MAXIMUM_UUID_COLLISION_RETRIES_V3
    ):
        raise ValueError("fresh batch retry count exceeds allocated UUID capacity")


@dataclass(frozen=True, slots=True)
class FormalPrivateUuidMappingV3:
    namespace: str
    source_uuid: str
    public_uuid: str

    def __post_init__(self) -> None:
        if self.namespace not in NAMESPACE_REGISTRY_V3:
            raise ValueError("private UUID mapping namespace is outside exact-10")
        _uuid(self.source_uuid, "private UUID mapping source")
        _uuid(self.public_uuid, "private UUID mapping public target")


@dataclass(frozen=True, slots=True)
class FormalPrivateAuditAlignmentLinkV3:
    public_envelope_sha256: str
    source_row_index: int
    source_case_commitment: str
    source_to_public_uuid_rows: tuple[FormalPrivateUuidMappingV3, ...] = field(
        repr=False
    )
    label_binding_id: str

    def __post_init__(self) -> None:
        _digest(self.public_envelope_sha256, "alignment link envelope digest")
        if type(self.source_row_index) is not int or self.source_row_index < 0:
            raise ValueError("alignment link source row index must be nonnegative integer")
        _digest(self.source_case_commitment, "alignment link source commitment")
        if type(self.source_to_public_uuid_rows) is not tuple or not self.source_to_public_uuid_rows or any(
            type(row) is not FormalPrivateUuidMappingV3
            for row in self.source_to_public_uuid_rows
        ):
            raise TypeError("alignment link needs exact private UUID mapping rows")
        if (
            len(self.source_to_public_uuid_rows)
            > MAXIMUM_PRIVATE_UUID_MAPPINGS_PER_CASE_V3
        ):
            raise ValueError("alignment private UUID mapping exceeds the case cap")
        for row in self.source_to_public_uuid_rows:
            row.__post_init__()
        if self.source_to_public_uuid_rows != tuple(
            sorted(
                self.source_to_public_uuid_rows,
                key=lambda row: (row.namespace, row.source_uuid),
            )
        ):
            raise ValueError("alignment private UUID mapping rows are not canonical")
        if (
            len({row.source_uuid for row in self.source_to_public_uuid_rows})
            != len(self.source_to_public_uuid_rows)
            or len({row.public_uuid for row in self.source_to_public_uuid_rows})
            != len(self.source_to_public_uuid_rows)
            or {row.namespace for row in self.source_to_public_uuid_rows}
            != set(NAMESPACE_REGISTRY_V3)
        ):
            raise ValueError("alignment private UUID mapping is not exact-10 and bijective")
        _digest(self.label_binding_id, "alignment link label binding ID")


@dataclass(frozen=True, slots=True)
class FormalPrivateAuditAlignmentV3:
    batch_id: str
    run_id_commitment: str
    key_source_receipt_id: str
    source_case_count: int
    links_in_public_order: tuple[FormalPrivateAuditAlignmentLinkV3, ...]
    alignment_id: str

    def __post_init__(self) -> None:
        self._validate_issued_shape()

    def _validate_issued_shape(self) -> None:
        """Replay cooperative content linkage; this authenticates no issuer."""
        _prefixed_digest(self.batch_id, "phase2b_formal_trusted_wire_batch_v3_", "alignment batch ID")
        _digest(self.run_id_commitment, "alignment run commitment")
        _prefixed_digest(self.key_source_receipt_id, "phase2b_key_source_transcript_v3_", "alignment key receipt ID")
        count = self.source_case_count
        if type(count) is not int or not 1 <= count <= MAXIMUM_BATCH_CASES_V3:
            raise ValueError("alignment source case count is outside 1 through 1024")
        if type(self.links_in_public_order) is not tuple or len(
            self.links_in_public_order
        ) != count or any(
            type(link) is not FormalPrivateAuditAlignmentLinkV3
            for link in self.links_in_public_order
        ):
            raise ValueError("alignment links do not have exact coverage")
        for link in self.links_in_public_order:
            link.__post_init__()
            if link.label_binding_id != _label_binding_id_v3(
                batch_id=self.batch_id,
                public_envelope_sha256=link.public_envelope_sha256,
                source_row_index=link.source_row_index,
                source_case_commitment=link.source_case_commitment,
                source_to_public_uuid_rows=link.source_to_public_uuid_rows,
            ):
                raise ValueError("alignment per-row label binding ID mismatch")
        if tuple(sorted(link.source_row_index for link in self.links_in_public_order)) != tuple(range(count)):
            raise ValueError("alignment source row mapping is not an exact bijection")
        if len({link.public_envelope_sha256 for link in self.links_in_public_order}) != count:
            raise ValueError("alignment envelope links must be unique")
        if len({link.source_case_commitment for link in self.links_in_public_order}) != count:
            raise ValueError("alignment source case commitments must be unique")
        if len({link.label_binding_id for link in self.links_in_public_order}) != count:
            raise ValueError("alignment label binding IDs must be unique")
        all_sources = tuple(
            row.source_uuid
            for link in self.links_in_public_order
            for row in link.source_to_public_uuid_rows
        )
        all_public = tuple(
            row.public_uuid
            for link in self.links_in_public_order
            for row in link.source_to_public_uuid_rows
        )
        if (
            len(set(all_sources)) != len(all_sources)
            or len(set(all_public)) != len(all_public)
            or set(all_sources) & set(all_public)
        ):
            raise ValueError("alignment UUID mapping is not globally injective across source rows")
        _prefixed_digest(
            self.alignment_id,
            "phase2b_formal_private_audit_alignment_v3_",
            "private alignment ID",
        )
        body = _alignment_body_v3(
            batch_id=self.batch_id,
            run_id_commitment=self.run_id_commitment,
            key_source_receipt_id=self.key_source_receipt_id,
            source_case_count=count,
            links=self.links_in_public_order,
        )
        if self.alignment_id != _alignment_id_v3(body):
            raise ValueError("private alignment content ID does not match its issued body")

    @property
    def public_envelope_sha256_in_order(self) -> tuple[str, ...]:
        return tuple(link.public_envelope_sha256 for link in self.links_in_public_order)

    @property
    def source_row_index_by_public_row(self) -> tuple[int, ...]:
        return tuple(link.source_row_index for link in self.links_in_public_order)

    @property
    def source_case_commitments_in_source_order(self) -> tuple[str, ...]:
        return tuple(
            link.source_case_commitment
            for link in sorted(
                self.links_in_public_order,
                key=lambda item: item.source_row_index,
            )
        )


@dataclass(frozen=True, order=True, slots=True)
class FormalUuidRenamingV3:
    namespace: str
    source_uuid: str
    target_uuid: str

    def __post_init__(self) -> None:
        if self.namespace not in NAMESPACE_REGISTRY_V3:
            raise ValueError("variant UUID renaming namespace is outside exact-10")
        _uuid(self.source_uuid, "variant source UUID")
        _uuid(self.target_uuid, "variant target UUID")


@dataclass(frozen=True, order=True, slots=True)
class FormalObservationOrderV3:
    envelope_id: str
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _prefixed_digest(
            self.envelope_id,
            "phase2b_formal_trusted_envelope_v3_",
            "observation-order envelope ID",
        )
        if (
            type(self.observation_ids) is not tuple
            or not self.observation_ids
            or len(self.observation_ids) > MAXIMUM_RECORDS_PER_KIND_V3
        ):
            raise ValueError("observation order needs an exact nonempty tuple")
        for value in self.observation_ids:
            _uuid(value, "observation-order UUID")
        if len(set(self.observation_ids)) != len(self.observation_ids):
            raise ValueError("observation order repeats an observation UUID")


@dataclass(frozen=True, slots=True)
class FormalVariantScheduleV3:
    case_order_envelope_ids: tuple[str, ...]
    uuid_renamings: tuple[FormalUuidRenamingV3, ...]
    observation_orders: tuple[FormalObservationOrderV3, ...]

    def __post_init__(self) -> None:
        if (
            type(self.case_order_envelope_ids) is not tuple
            or not self.case_order_envelope_ids
            or len(self.case_order_envelope_ids) > MAXIMUM_BATCH_CASES_V3
        ):
            raise ValueError("variant case order needs an exact nonempty tuple")
        for value in self.case_order_envelope_ids:
            _prefixed_digest(
                value,
                "phase2b_formal_trusted_envelope_v3_",
                "variant case-order envelope ID",
            )
        if len(set(self.case_order_envelope_ids)) != len(self.case_order_envelope_ids):
            raise ValueError("variant case order repeats an envelope")
        if type(self.uuid_renamings) is not tuple or any(
            type(row) is not FormalUuidRenamingV3 for row in self.uuid_renamings
        ):
            raise TypeError("variant UUID renamings need exact immutable rows")
        if len(self.uuid_renamings) > (
            MAXIMUM_BATCH_CASES_V3 * MAXIMUM_PRIVATE_UUID_MAPPINGS_PER_CASE_V3
        ):
            raise ValueError("variant UUID renaming collection exceeds the batch cap")
        for row in self.uuid_renamings:
            row.__post_init__()
        if type(self.observation_orders) is not tuple or any(
            type(row) is not FormalObservationOrderV3 for row in self.observation_orders
        ):
            raise TypeError("variant observation orders need exact immutable rows")
        if len(self.observation_orders) > MAXIMUM_BATCH_CASES_V3:
            raise ValueError("variant observation-order collection exceeds the batch cap")
        for row in self.observation_orders:
            row.__post_init__()


@dataclass(frozen=True, slots=True)
class FormalVariantPayloadV3:
    envelope_id: str
    payload: dict[str, object]
    padding_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _prefixed_digest(
            self.envelope_id,
            "phase2b_formal_trusted_envelope_v3_",
            "variant payload envelope ID",
        )
        if type(self.payload) is not dict or type(self.padding_bytes) is not bytes:
            raise TypeError("variant payload row type drift")


@dataclass(frozen=True, slots=True)
class _FormalVariantBaseEnvelopeV3:
    envelope_id: str
    canonical_payload_bytes: bytes = field(repr=False)
    padding_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _prefixed_digest(
            self.envelope_id,
            "phase2b_formal_trusted_envelope_v3_",
            "variant base envelope ID",
        )
        if type(self.canonical_payload_bytes) is not bytes or type(
            self.padding_bytes
        ) is not bytes:
            raise TypeError("variant base envelope payload/padding bytes type drift")
        decoded = decode_and_require_canonical_rfc8785_v1(
            self.canonical_payload_bytes
        )
        if type(decoded) is not dict:
            raise ValueError("variant base canonical payload root is not an object")
        if (
            len(self.canonical_payload_bytes)
            + len(self.padding_bytes)
            + ENVELOPE_HEADER_BYTES_V3
            != ENVELOPE_BYTES_V3
            or len(self.padding_bytes) < MINIMUM_PADDING_BYTES_V3
        ):
            raise ValueError("variant base bytes do not close a fixed envelope")


@dataclass(frozen=True, slots=True)
class _FormalVariantContextV3:
    batch_id: str
    private_alignment_id: str
    transform_catalog_root: str
    catalog_binding_receipt_id: str
    global_transform_catalog: FormalGlobalTransformCatalogV3 = field(
        repr=False, compare=False
    )
    transform_catalog_binding_receipt: FormalGlobalTransformCatalogBindingReceiptV3 = field(
        repr=False, compare=False
    )
    base_envelopes: tuple[_FormalVariantBaseEnvelopeV3, ...] = field(
        repr=False
    )
    context_id: str
    origin_authenticated: bool

    def __post_init__(self) -> None:
        self._validate_issued_shape()

    def _validate_issued_shape(self) -> None:
        """Replay private structural context linkage; this authenticates no issuer."""
        _prefixed_digest(
            self.batch_id,
            "phase2b_formal_trusted_wire_batch_v3_",
            "variant context batch ID",
        )
        _prefixed_digest(
            self.private_alignment_id,
            "phase2b_formal_private_audit_alignment_v3_",
            "variant context private alignment ID",
        )
        _digest(self.transform_catalog_root, "variant context catalog root")
        _prefixed_digest(
            self.catalog_binding_receipt_id,
            "phase2b_formal_global_transform_catalog_binding_v3_",
            "variant context catalog binding receipt ID",
        )
        if type(self.global_transform_catalog) is not FormalGlobalTransformCatalogV3 or type(
            self.transform_catalog_binding_receipt
        ) is not FormalGlobalTransformCatalogBindingReceiptV3:
            raise TypeError("variant context catalog proof type drift")
        verify_formal_global_transform_catalog_binding_v3(
            catalog=self.global_transform_catalog,
            binding_receipt=self.transform_catalog_binding_receipt,
        )
        if (
            self.transform_catalog_binding_receipt.transform_catalog_root
            != self.transform_catalog_root
            or self.transform_catalog_binding_receipt.receipt_id
            != self.catalog_binding_receipt_id
        ):
            raise ValueError("variant context catalog proof/root mismatch")
        if type(self.base_envelopes) is not tuple or not self.base_envelopes or any(
            type(row) is not _FormalVariantBaseEnvelopeV3 for row in self.base_envelopes
        ):
            raise TypeError("variant context base envelopes are not exact")
        for row in self.base_envelopes:
            row.__post_init__()
        if len({row.envelope_id for row in self.base_envelopes}) != len(self.base_envelopes):
            raise ValueError("variant context repeats a base envelope")
        _prefixed_digest(
            self.context_id,
            "phase2b_formal_variant_context_v3_",
            "variant context ID",
        )
        if self.context_id != _variant_context_id_v3(
            batch_id=self.batch_id,
            private_alignment_id=self.private_alignment_id,
            transform_catalog_root=self.transform_catalog_root,
            catalog_binding_receipt_id=self.catalog_binding_receipt_id,
            base_envelopes=self.base_envelopes,
        ):
            raise ValueError("variant context content ID mismatch")
        if self.origin_authenticated is not False:
            raise ValueError("variant context claim boundary drift")


@dataclass(frozen=True, slots=True)
class FormalPrivateAuditRowV3:
    source_case_commitment: str
    label_binding_id: str
    private_payload_bytes: bytes = field(repr=False, compare=False)
    row_binding: str
    semantic_truth_verified: bool
    origin_authenticated: bool

    def __post_init__(self) -> None:
        self._validate_issued_shape()

    def _validate_issued_shape(self) -> None:
        """Replay cooperative byte/source linkage; this authenticates no issuer."""
        _digest(self.source_case_commitment, "private audit row source commitment")
        _digest(self.label_binding_id, "private audit row label binding ID")
        if type(self.private_payload_bytes) is not bytes:
            raise TypeError("private audit row payload must be exact bytes")
        expected = "sha256:" + hashlib.sha256(
            _PRIVATE_ROW_DOMAIN_V3
            + self.source_case_commitment.encode("ascii")
            + self.label_binding_id.encode("ascii")
            + len(self.private_payload_bytes).to_bytes(8, "big")
            + self.private_payload_bytes
        ).hexdigest()
        if self.row_binding != expected:
            raise ValueError("private audit row content/source binding mismatch")
        if self.semantic_truth_verified is not False or self.origin_authenticated is not False:
            raise ValueError("private audit row truth/origin claim boundary drift")


@dataclass(frozen=True, slots=True)
class FormalTrustedWireBatchRejectionV3:
    disposition: FormalTrustedWireDispositionV3
    reason: str
    source_case_count: int
    envelopes: tuple[FormalTrustedWireEnvelopeV3, ...] = ()
    envelope_ids: tuple[str, ...] = ()
    batch_id: None = None
    key_source_transcript_verified: bool = False
    key_source_independence_verified: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    actual_execution_claimed: bool = False

    def __post_init__(self) -> None:
        if self.disposition is not FormalTrustedWireDispositionV3.ABSTAIN:
            raise ValueError("wire rejection disposition must be ABSTAIN")
        _text(self.reason, "wire rejection reason")
        if type(self.source_case_count) is not int or self.source_case_count < 0:
            raise ValueError("wire rejection source count must be an exact nonnegative integer")
        if self.envelopes != () or self.envelope_ids != () or self.batch_id is not None:
            raise ValueError("wire rejection must be atomic empty")
        if any(
            value is not False
            for value in (
                self.key_source_transcript_verified,
                self.key_source_independence_verified,
                self.origin_authenticated,
                self.formal_covert_audit,
                self.actual_execution_claimed,
            )
        ):
            raise ValueError("wire rejection cannot carry positive claims")


@dataclass(frozen=True, slots=True)
class FormalTrustedWireBatchV3:
    disposition: FormalTrustedWireDispositionV3
    schema_version: str
    claim_level: str
    batch_context: FormalBatchContextV3
    key_source_receipt_id: str
    envelopes: tuple[FormalTrustedWireEnvelopeV3, ...]
    envelope_ids: tuple[str, ...]
    batch_id: str
    key_source_transcript_verified: bool = False
    key_source_independence_verified: bool = False
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    capacity_1024_qualified: bool = False
    actual_execution_claimed: bool = False

    def __post_init__(self) -> None:
        if self.disposition is not FormalTrustedWireDispositionV3.COMPLETE:
            raise ValueError("successful wire batch disposition must be COMPLETE")
        if self.schema_version != FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION or self.claim_level != FORMAL_TRUSTED_WIRE_V3_CLAIM_LEVEL:
            raise ValueError("successful wire batch schema/claim identity drift")
        if type(self.batch_context) is not FormalBatchContextV3:
            raise TypeError("successful wire batch context type drift")
        self.batch_context.__post_init__()
        _prefixed_digest(self.key_source_receipt_id, "phase2b_key_source_transcript_v3_", "wire key receipt ID")
        if type(self.envelopes) is not tuple or not self.envelopes or any(
            type(row) is not FormalTrustedWireEnvelopeV3 for row in self.envelopes
        ):
            raise TypeError("successful wire batch envelopes must be exact nonempty tuple")
        fresh_decoded = tuple(
            decode_and_audit_formal_envelope_v3(row.envelope)
            for row in self.envelopes
        )
        for row, decoded in zip(self.envelopes, fresh_decoded, strict=True):
            _verify_trusted_envelope_cache_v3(row, decoded)
        _verify_decoded_batch_closure_v3(fresh_decoded, self.batch_context)
        count = len(self.envelopes)
        if count != int(self.batch_context.case_count_decimal):
            raise ValueError("successful wire batch case count/context mismatch")
        if type(self.envelope_ids) is not tuple or self.envelope_ids != tuple(row.envelope_id for row in self.envelopes):
            raise ValueError("successful wire batch envelope ID cache mismatch")
        if len(set(self.envelope_ids)) != count:
            raise ValueError("successful wire batch repeats an envelope ID")
        if self.batch_id != _batch_id_v3(
            self.batch_context, self.key_source_receipt_id, self.envelope_ids
        ):
            raise ValueError("successful wire batch content ID mismatch")
        if any(
            value is not False
            for value in (
                self.key_source_transcript_verified,
                self.key_source_independence_verified,
                self.origin_authenticated,
                self.formal_covert_audit,
                self.capacity_1024_qualified,
                self.actual_execution_claimed,
            )
        ):
            raise ValueError("successful wire batch claim boundary drift")


@dataclass(frozen=True, slots=True)
class FormalTrustedWireBatchReplayV3:
    disposition: FormalTrustedWireDispositionV3
    batch_context: FormalBatchContextV3
    key_source_receipt_id: str
    decoded_envelopes: tuple[DecodedFormalEnvelopeV3, ...]
    envelope_ids: tuple[str, ...]
    batch_id: str
    structural_replay_verified: bool
    catalog_membership_verified: bool = False
    key_source_transcript_verified: bool = False
    secret_shuffle_replayed: bool = False
    secret_id_assignment_replayed: bool = False
    secret_padding_replayed: bool = False
    formal_covert_audit: bool = False
    actual_execution_claimed: bool = False

    def __post_init__(self) -> None:
        if self.disposition is not FormalTrustedWireDispositionV3.COMPLETE:
            raise ValueError("successful public replay disposition must be COMPLETE")
        if type(self.batch_context) is not FormalBatchContextV3:
            raise TypeError("successful public replay context type drift")
        self.batch_context.__post_init__()
        _prefixed_digest(
            self.key_source_receipt_id,
            "phase2b_key_source_transcript_v3_",
            "public replay key receipt ID",
        )
        if type(self.decoded_envelopes) is not tuple or not self.decoded_envelopes or any(
            type(row) is not DecodedFormalEnvelopeV3 for row in self.decoded_envelopes
        ):
            raise TypeError("successful public replay needs exact decoded envelopes")
        count = len(self.decoded_envelopes)
        if count != int(self.batch_context.case_count_decimal):
            raise ValueError("successful public replay count/context mismatch")
        if type(self.envelope_ids) is not tuple or self.envelope_ids != tuple(
            row.envelope_id for row in self.decoded_envelopes
        ):
            raise ValueError("successful public replay envelope ID cache mismatch")
        if len(set(self.envelope_ids)) != count or self.batch_id != _batch_id_v3(
            self.batch_context, self.key_source_receipt_id, self.envelope_ids
        ):
            raise ValueError("successful public replay batch identity mismatch")
        fresh = tuple(
            decode_and_audit_formal_envelope_v3(row.envelope)
            for row in self.decoded_envelopes
        )
        if fresh != self.decoded_envelopes:
            raise ValueError("successful public replay decoded caches are not fresh")
        _verify_decoded_batch_closure_v3(fresh, self.batch_context)
        if self.structural_replay_verified is not True or any(
            value is not False
            for value in (
                self.key_source_transcript_verified,
                self.catalog_membership_verified,
                self.secret_shuffle_replayed,
                self.secret_id_assignment_replayed,
                self.secret_padding_replayed,
                self.formal_covert_audit,
                self.actual_execution_claimed,
            )
        ):
            raise ValueError("successful public replay claim boundary drift")


@dataclass(frozen=True, slots=True)
class FormalTrustedWireBuildSuccessV3:
    public_batch: FormalTrustedWireBatchV3
    key_source_receipt: KeySourceTranscriptReceiptV3 = field(repr=False)
    private_audit_alignment: FormalPrivateAuditAlignmentV3 = field(repr=False)
    variant_context: _FormalVariantContextV3 = field(repr=False)
    success_id: str
    builder_input_transcript_consistency_checked: bool
    builder_input_catalog_membership_checked: bool
    private_alignment_linkage_checked: bool
    origin_authenticated: bool

    def __post_init__(self) -> None:
        self._validate_issued_shape()

    def _validate_issued_shape(self) -> None:
        """Replay content linkage only; caller-held secrets are required below."""
        if type(self.public_batch) is not FormalTrustedWireBatchV3:
            raise TypeError("wire build success public batch type drift")
        if type(self.key_source_receipt) is not KeySourceTranscriptReceiptV3:
            raise TypeError("wire build success key receipt type drift")
        self.key_source_receipt.__post_init__()
        if type(self.private_audit_alignment) is not FormalPrivateAuditAlignmentV3:
            raise TypeError("wire build success private alignment type drift")
        if type(self.variant_context) is not _FormalVariantContextV3:
            raise TypeError("wire build success variant context type drift")
        self.variant_context._validate_issued_shape()
        if not verify_private_audit_alignment_v3(
            batch=self.public_batch,
            alignment=self.private_audit_alignment,
        ):
            raise ValueError("wire build success private alignment did not fully replay")
        if (
            self.key_source_receipt.receipt_id != self.public_batch.key_source_receipt_id
            or self.key_source_receipt.receipt_id
            != self.private_audit_alignment.key_source_receipt_id
            or
            self.variant_context.batch_id != self.public_batch.batch_id
            or self.variant_context.private_alignment_id
            != self.private_audit_alignment.alignment_id
            or self.variant_context.transform_catalog_root
            != self.public_batch.batch_context.transform_catalog_root
            or tuple(row.envelope_id for row in self.variant_context.base_envelopes)
            != self.public_batch.envelope_ids
        ):
            raise ValueError("wire build success variant context is batch/catalog stale")
        for public_envelope, base in zip(
            self.public_batch.envelopes,
            self.variant_context.base_envelopes,
            strict=True,
        ):
            base_payload_bytes = base.canonical_payload_bytes
            payload_start = ENVELOPE_HEADER_BYTES_V3
            payload_end = payload_start + len(base_payload_bytes)
            if (
                base.envelope_id != public_envelope.envelope_id
                or public_envelope.envelope[payload_start:payload_end]
                != base_payload_bytes
                or public_envelope.envelope[payload_end:] != base.padding_bytes
            ):
                raise ValueError("wire build success variant base/public envelope splice")
        _verify_variant_context_catalog_membership_v3(
            public_batch=self.public_batch,
            private_alignment=self.private_audit_alignment,
            variant_context=self.variant_context,
        )
        _prefixed_digest(
            self.success_id,
            "phase2b_formal_trusted_wire_build_success_v3_",
            "wire build success ID",
        )
        if self.success_id != _build_success_id_v3(
            public_batch=self.public_batch,
            key_source_receipt=self.key_source_receipt,
            private_alignment=self.private_audit_alignment,
            variant_context=self.variant_context,
        ):
            raise ValueError("wire build success content ID mismatch")
        if (
            self.builder_input_transcript_consistency_checked is not True
            or self.builder_input_catalog_membership_checked is not True
            or self.private_alignment_linkage_checked is not True
            or self.origin_authenticated is not False
        ):
            raise ValueError("wire build success claim boundary drift")


def _reject_v3(reason: str, count: int) -> FormalTrustedWireBatchRejectionV3:
    return FormalTrustedWireBatchRejectionV3(FormalTrustedWireDispositionV3.ABSTAIN, reason, count)


def _frame_payload_with_padding_v3(payload: bytes, padding: bytes) -> bytes:
    if type(payload) is not bytes or type(padding) is not bytes: raise TypeError("payload/padding must use exact bytes")
    if not 0 < len(payload) <= MAXIMUM_PAYLOAD_BYTES_V3: raise ValueError("payload exceeds formal envelope budget")
    required = ENVELOPE_BYTES_V3 - ENVELOPE_HEADER_BYTES_V3 - len(payload)
    if len(padding) != required or len(padding) < MINIMUM_PADDING_BYTES_V3: raise ValueError("padding does not close fixed envelope")
    header = _ENVELOPE_HEADER_V3.pack(
        ENVELOPE_MAGIC_V3, ENVELOPE_VERSION_V3, ENVELOPE_HEADER_BYTES_V3,
        len(payload), hashlib.sha256(payload).digest(), hashlib.sha256(padding).digest(),
    )
    result = header + payload + padding
    if len(result) != ENVELOPE_BYTES_V3: raise RuntimeError("fixed envelope length drift")
    return result


def _channel_projection_v3(public_case: FormalPublicCaseV3, payload_length: int, padding: bytes) -> ChannelProjectionV3:
    graph_mapping = _profile_value_v3(public_case.graph)
    uuid_rows = tuple(
        NamespaceUuidChannelV3(
            "/public_case/graph" + pointer,
            namespace,
            value,
            bytes.fromhex(value.replace("-", "")),
        )
        for pointer, namespace, value in _walk_uuid_leaves_v3(graph_mapping)
    )
    missingness: list[MissingnessChannelV3] = []

    def visit_missingness(current: object, pointer: str) -> None:
        if type(current) is dict:
            if set(current) == {"available", "value"} and type(current["available"]) is bool:
                missingness.append(MissingnessChannelV3(pointer, not current["available"]))
            for key in sorted(current):
                visit_missingness(current[key], pointer + "/" + key)
        elif type(current) is list:
            for index, item in enumerate(current):
                visit_missingness(item, pointer + "/" + str(index))

    visit_missingness(graph_mapping, "/public_case/graph")
    record_tokens: list[str] = []
    for name, id_field in (
        ("entities", "entity_id"), ("quantities", "quantity_id"),
        ("contexts", "context_id"), ("observations", "observation_id"),
        ("scales", "scale_id"), ("aggregate_maps", "aggregate_map_id"),
        ("transforms", "transform_id"),
    ):
        for item in graph_mapping[name]: record_tokens.append(name + ":" + item[id_field])  # type: ignore[index,union-attr]
    component_tokens = tuple(
        observation.observation_id
        + ":"
        + ",".join(
            component.label
            + "/"
            + component.semantic_axis
            + "/"
            + component.value_role
            for component in observation.components
        )
        for observation in public_case.graph.observations
    )
    mask = public_case.graph.eligibility_mask
    eligibility = tuple(
        ["observation_id:" + value for value in mask.observation_ids]
        + ["role_candidate_id:" + value for value in mask.role_candidate_ids]
        + ["scale_id:" + value for value in mask.scale_ids]
    )
    descriptor = public_case.graph.semantic_descriptor

    def lexical_token_count(segment: str) -> int:
        in_token = False
        count = 0
        for character in segment:
            is_token = unicodedata.category(character)[0] in {"L", "M", "N"}
            if is_token and not in_token:
                count += 1
            in_token = is_token
        return count

    def category_runs(segment: str) -> tuple[str, ...]:
        result: list[str] = []
        previous: str | None = None
        count = 0
        for character in segment:
            current = unicodedata.category(character)
            if current == previous:
                count += 1
                continue
            if previous is not None:
                result.append(previous + ":" + str(count))
            previous = current
            count = 1
        if previous is not None:
            result.append(previous + ":" + str(count))
        return tuple(result)

    descriptor_channel = SemanticDescriptorChannelV3(
        kind=descriptor.kind,
        segment_count=len(descriptor.segments),
        segment_utf8_lengths=tuple(len(value.encode("utf-8")) for value in descriptor.segments),
        segment_trailing_u0020_counts=tuple(
            len(value) - len(value.rstrip(" ")) for value in descriptor.segments
        ),
        segment_boundary_codepoints=tuple(
            (ord(value[0]), ord(value[-1])) for value in descriptor.segments
        ),
        segment_lexical_token_counts=tuple(
            lexical_token_count(value) for value in descriptor.segments
        ),
        segment_non_alnum_codepoint_sequences=tuple(
            tuple(ord(character) for character in value if not character.isalnum())
            for value in descriptor.segments
        ),
        segment_category_run_signatures=tuple(
            category_runs(value) for value in descriptor.segments
        ),
        segment_order_sha256="sha256:"
        + hashlib.sha256(
            canonicalize_rfc8785_v1(list(descriptor.segments))
        ).hexdigest(),
    )
    return ChannelProjectionV3(
        payload_length=payload_length,
        padding_length=len(padding),
        padding_sha256="sha256:" + hashlib.sha256(padding).hexdigest(),
        padding_prefix_32=padding[:32],
        padding_suffix_32=padding[-32:],
        namespace_uuid_channels=uuid_rows,
        missingness_channels=tuple(missingness),
        missingness_bits=tuple(row.missing for row in missingness),
        observation_array_paths=PUBLIC_OBSERVATION_ARRAY_PATHS_V3,
        record_order_tokens=tuple(record_tokens),
        component_slot_order_tokens=component_tokens,
        eligibility_mask_tokens=eligibility,
        semantic_descriptor_channel=descriptor_channel,
        public_provenance_commitment=public_case.public_provenance_commitment,
    )


def _decode_and_audit_formal_envelope_impl_v3(
    envelope: bytes,
) -> DecodedFormalEnvelopeV3:
    if type(envelope) is not bytes or len(envelope) != ENVELOPE_BYTES_V3: raise ValueError("formal envelope must be exact 65536 bytes")
    magic, version, header_bytes, payload_length, payload_digest, padding_digest = _ENVELOPE_HEADER_V3.unpack(envelope[:ENVELOPE_HEADER_BYTES_V3])
    if (magic, version, header_bytes) != (ENVELOPE_MAGIC_V3, ENVELOPE_VERSION_V3, ENVELOPE_HEADER_BYTES_V3) or not 0 < payload_length <= MAXIMUM_PAYLOAD_BYTES_V3:
        raise ValueError("formal envelope header drift")
    payload_bytes = envelope[header_bytes:header_bytes + payload_length]
    padding = envelope[header_bytes + payload_length:]
    if len(padding) < MINIMUM_PADDING_BYTES_V3 or hashlib.sha256(payload_bytes).digest() != payload_digest or hashlib.sha256(padding).digest() != padding_digest:
        raise ValueError("formal envelope structural digest/length failure")
    _bounded_json_wire_preflight_v3(payload_bytes)
    decoded = decode_and_require_canonical_rfc8785_v1(payload_bytes)
    decoded = _restore_json_integers_v3(decoded)
    if type(decoded) is not dict: raise ValueError("formal envelope payload root must be object")
    normalized = _normalize_formal_payload_structure_v3(decoded)
    if normalized != decoded: raise ValueError("formal envelope payload is not normalized")
    context = _context_from_mapping_v3(decoded["batch_context"])
    public_mapping = decoded["public_case"]
    graph = _graph_from_mapping_v3(public_mapping["graph"])  # type: ignore[index]
    public_case = FormalPublicCaseV3(public_mapping["schema_version"], graph, public_mapping["public_provenance_commitment"])  # type: ignore[index,arg-type]
    projection = _channel_projection_v3(public_case, payload_length, padding)
    envelope_id = "phase2b_formal_trusted_envelope_v3_" + hashlib.sha256(_ENVELOPE_ID_DOMAIN_V3 + envelope).hexdigest()
    return DecodedFormalEnvelopeV3(
        envelope, envelope_id, decoded, payload_bytes, padding, payload_length,
        len(padding), "sha256:" + payload_digest.hex(), "sha256:" + padding_digest.hex(),
        context, public_case, classify_public_payload_leaves_v3(decoded), projection,
        True, False, False, False,
    )


def decode_and_audit_formal_envelope_v3(
    envelope: bytes,
) -> DecodedFormalEnvelopeV3:
    try:
        return _decode_and_audit_formal_envelope_impl_v3(envelope)
    except (RecursionError, MemoryError):
        raise ValueError("bounded formal envelope decode rejected") from None


@dataclass(frozen=True, slots=True)
class _FormalValidatedVariantSnapshotV3:
    public_batch: FormalTrustedWireBatchV3
    key_source_receipt: KeySourceTranscriptReceiptV3
    private_alignment: FormalPrivateAuditAlignmentV3
    variant_context: _FormalVariantContextV3
    success_id: str


def _trusted_envelope_from_primitive_bytes_v3(
    envelope: bytes,
) -> FormalTrustedWireEnvelopeV3:
    if type(envelope) is not bytes:
        raise TypeError("variant snapshot envelope must be exact bytes")
    decoded = decode_and_audit_formal_envelope_v3(envelope)
    return FormalTrustedWireEnvelopeV3(
        envelope=bytes(envelope),
        envelope_id=decoded.envelope_id,
        payload_sha256=decoded.payload_sha256,
        padding_sha256=decoded.padding_sha256,
        payload_length=decoded.payload_length,
        padding_length=decoded.padding_length,
        public_case=decoded.public_case,
        channel_projection=decoded.channel_projection,
        structural_replay_verified=True,
        formal_covert_audit=False,
        actual_execution_claimed=False,
    )


def _verify_variant_context_catalog_membership_v3(
    *,
    public_batch: FormalTrustedWireBatchV3,
    private_alignment: FormalPrivateAuditAlignmentV3,
    variant_context: _FormalVariantContextV3,
) -> bool:
    """Replay exact base bytes, inverse UUID maps, and source catalog entries."""
    public_batch.__post_init__()
    verify_private_audit_alignment_v3(
        batch=public_batch, alignment=private_alignment
    )
    variant_context._validate_issued_shape()
    if (
        variant_context.batch_id != public_batch.batch_id
        or variant_context.private_alignment_id != private_alignment.alignment_id
        or variant_context.transform_catalog_root
        != public_batch.batch_context.transform_catalog_root
        or tuple(row.envelope_id for row in variant_context.base_envelopes)
        != public_batch.envelope_ids
    ):
        raise ValueError("variant context is stale for its full build result")
    allowed = {
        (entry.operation, entry.exact_source_transform_use_sha256)
        for entry in variant_context.global_transform_catalog.entries
    }
    all_source_uuids: set[str] = set()
    all_public_uuids: set[str] = set()
    for public_row, link, base in zip(
        public_batch.envelopes,
        private_alignment.links_in_public_order,
        variant_context.base_envelopes,
        strict=True,
    ):
        decoded = decode_and_audit_formal_envelope_v3(public_row.envelope)
        if (
            base.envelope_id != decoded.envelope_id
            or base.canonical_payload_bytes != decoded.payload_bytes
            or base.padding_bytes != decoded.padding_bytes
        ):
            raise ValueError("variant base bytes are stale or public-envelope spliced")
        inverse: dict[tuple[str, str], str] = {}
        for row in link.source_to_public_uuid_rows:
            inverse[(row.namespace, row.public_uuid)] = row.source_uuid
            all_source_uuids.add(row.source_uuid)
            all_public_uuids.add(row.public_uuid)
        public_graph = decoded.payload["public_case"]["graph"]  # type: ignore[index]
        source_graph = _rename_graph_ids_v3(deepcopy(public_graph), inverse)  # type: ignore[arg-type]
        source_graph = _canonicalize_graph_mapping_v3(
            source_graph, preserve_observation_order=False
        )
        transforms = {
            transform["transform_id"]: transform
            for transform in source_graph["transforms"]  # type: ignore[index,union-attr]
        }
        for aggregate_map in source_graph["aggregate_maps"]:  # type: ignore[index,union-attr]
            transform = transforms[aggregate_map["transform_id"]]  # type: ignore[index]
            membership = (
                transform["operation"],  # type: ignore[index]
                _source_transform_use_hash_from_mappings_v3(
                    aggregate_map, transform  # type: ignore[arg-type]
                ),
            )
            if membership not in allowed:
                raise ValueError(
                    "variant inverse source transform use is absent from catalog"
                )
    if all_source_uuids & all_public_uuids:
        raise ValueError("source and public UUID domains are not disjoint")
    return True


def _snapshot_and_validate_build_success_for_variants_v3(
    build_success: FormalTrustedWireBuildSuccessV3,
) -> _FormalValidatedVariantSnapshotV3:
    """Copy primitive state once, reconstruct it, validate it, then stop reading input."""
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("formal variant API needs exact full build success")
    # Snapshot only immutable primitive values before validating anything.
    # No code below this block reads ``build_success`` or any of its child
    # objects again, so replacing a frozen field after validation cannot alter
    # the bytes subsequently normalized/re-encoded.
    original_batch = build_success.public_batch
    original_receipt = build_success.key_source_receipt
    original_alignment = build_success.private_audit_alignment
    original_context = build_success.variant_context
    batch_snapshot = (
        original_batch.disposition.value,
        original_batch.schema_version,
        original_batch.claim_level,
        (
            original_batch.batch_context.case_count_decimal,
            original_batch.batch_context.transform_catalog_root,
            original_batch.batch_context.id_collision_retry_count_decimal,
            original_batch.batch_context.run_id_commitment,
        ),
        original_batch.key_source_receipt_id,
        tuple(bytes(row.envelope) for row in original_batch.envelopes),
        tuple(original_batch.envelope_ids),
        original_batch.batch_id,
        original_batch.key_source_transcript_verified,
        original_batch.key_source_independence_verified,
        original_batch.origin_authenticated,
        original_batch.formal_covert_audit,
        original_batch.capacity_1024_qualified,
        original_batch.actual_execution_claimed,
    )
    receipt_snapshot = (
        original_receipt.schema_version,
        original_receipt.shuffle_seed_commitment,
        original_receipt.id_seed_commitment,
        original_receipt.padding_seed_commitment,
        tuple(original_receipt.latent_seed_commitments),
        tuple(original_receipt.answer_seed_commitments),
        original_receipt.receipt_id,
    )
    alignment_snapshot = (
        original_alignment.batch_id,
        original_alignment.run_id_commitment,
        original_alignment.key_source_receipt_id,
        original_alignment.source_case_count,
        tuple(
            (
                link.public_envelope_sha256,
                link.source_row_index,
                link.source_case_commitment,
                tuple(
                    (row.namespace, row.source_uuid, row.public_uuid)
                    for row in link.source_to_public_uuid_rows
                ),
                link.label_binding_id,
            )
            for link in original_alignment.links_in_public_order
        ),
        original_alignment.alignment_id,
    )
    context_snapshot = (
        original_context.batch_id,
        original_context.private_alignment_id,
        original_context.transform_catalog_root,
        original_context.catalog_binding_receipt_id,
        (
            original_context.global_transform_catalog.schema_version,
            tuple(
                (entry.operation, entry.exact_source_transform_use_sha256)
                for entry in original_context.global_transform_catalog.entries
            ),
        ),
        (
            original_context.transform_catalog_binding_receipt.schema_version,
            original_context.transform_catalog_binding_receipt.transform_catalog_root,
            tuple(
                original_context.transform_catalog_binding_receipt.catalog_entry_commitments
            ),
            original_context.transform_catalog_binding_receipt.receipt_id,
        ),
        tuple(
            (
                row.envelope_id,
                bytes(row.canonical_payload_bytes),
                bytes(row.padding_bytes),
            )
            for row in original_context.base_envelopes
        ),
        original_context.context_id,
        original_context.origin_authenticated,
    )
    success_snapshot = (
        build_success.success_id,
        build_success.builder_input_transcript_consistency_checked,
        build_success.builder_input_catalog_membership_checked,
        build_success.private_alignment_linkage_checked,
        build_success.origin_authenticated,
    )
    del original_batch, original_receipt, original_alignment, original_context

    batch_context_snapshot = batch_snapshot[3]
    batch_context = FormalBatchContextV3(
        batch_context_snapshot[0],
        batch_context_snapshot[1],
        batch_context_snapshot[2],
        batch_context_snapshot[3],
    )
    trusted_rows = tuple(
        _trusted_envelope_from_primitive_bytes_v3(row) for row in batch_snapshot[5]
    )
    batch = FormalTrustedWireBatchV3(
        disposition=FormalTrustedWireDispositionV3(batch_snapshot[0]),
        schema_version=batch_snapshot[1],
        claim_level=batch_snapshot[2],
        batch_context=batch_context,
        key_source_receipt_id=batch_snapshot[4],
        envelopes=trusted_rows,
        envelope_ids=batch_snapshot[6],
        batch_id=batch_snapshot[7],
        key_source_transcript_verified=batch_snapshot[8],
        key_source_independence_verified=batch_snapshot[9],
        origin_authenticated=batch_snapshot[10],
        formal_covert_audit=batch_snapshot[11],
        capacity_1024_qualified=batch_snapshot[12],
        actual_execution_claimed=batch_snapshot[13],
    )
    receipt = KeySourceTranscriptReceiptV3(*receipt_snapshot)
    links = tuple(
        FormalPrivateAuditAlignmentLinkV3(
            link[0],
            link[1],
            link[2],
            tuple(
                FormalPrivateUuidMappingV3(*row) for row in link[3]
            ),
            link[4],
        )
        for link in alignment_snapshot[4]
    )
    alignment = FormalPrivateAuditAlignmentV3(
        alignment_snapshot[0],
        alignment_snapshot[1],
        alignment_snapshot[2],
        alignment_snapshot[3],
        links,
        alignment_snapshot[5],
    )
    catalog_snapshot = context_snapshot[4]
    catalog = FormalGlobalTransformCatalogV3(
        catalog_snapshot[0],
        tuple(
            FormalGlobalTransformCatalogEntryV3(*entry)
            for entry in catalog_snapshot[1]
        ),
    )
    binding_snapshot = context_snapshot[5]
    binding = FormalGlobalTransformCatalogBindingReceiptV3(
        binding_snapshot[0],
        binding_snapshot[1],
        binding_snapshot[2],
        binding_snapshot[3],
    )
    bases = tuple(
        _FormalVariantBaseEnvelopeV3(*row) for row in context_snapshot[6]
    )
    variant_context = _FormalVariantContextV3(
        context_snapshot[0],
        context_snapshot[1],
        context_snapshot[2],
        context_snapshot[3],
        catalog,
        binding,
        bases,
        context_snapshot[7],
        context_snapshot[8],
    )
    if (
        receipt.receipt_id != batch.key_source_receipt_id
        or receipt.receipt_id != alignment.key_source_receipt_id
        or success_snapshot[1] is not True
        or success_snapshot[2] is not True
        or success_snapshot[3] is not True
        or success_snapshot[4] is not False
    ):
        raise ValueError("variant build success claim/content linkage drift")
    if success_snapshot[0] != _build_success_id_v3(
        public_batch=batch,
        key_source_receipt=receipt,
        private_alignment=alignment,
        variant_context=variant_context,
    ):
        raise ValueError("variant full build success content ID mismatch")
    _verify_variant_context_catalog_membership_v3(
        public_batch=batch,
        private_alignment=alignment,
        variant_context=variant_context,
    )
    return _FormalValidatedVariantSnapshotV3(
        batch, receipt, alignment, variant_context, success_snapshot[0]
    )


def _snapshot_variant_schedule_v3(
    schedule: FormalVariantScheduleV3,
) -> FormalVariantScheduleV3:
    """Copy primitive schedule state once and freshly validate the copy."""
    if type(schedule) is not FormalVariantScheduleV3:
        raise TypeError("formal variant issuance needs exact schedule type")
    raw = (
        tuple(schedule.case_order_envelope_ids),
        tuple(
            (row.namespace, row.source_uuid, row.target_uuid)
            for row in schedule.uuid_renamings
        ),
        tuple(
            (row.envelope_id, tuple(row.observation_ids))
            for row in schedule.observation_orders
        ),
    )
    return FormalVariantScheduleV3(
        raw[0],
        tuple(FormalUuidRenamingV3(*row) for row in raw[1]),
        tuple(FormalObservationOrderV3(*row) for row in raw[2]),
    )


def _validated_variant_payload_rows_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    schedule: FormalVariantScheduleV3,
) -> tuple[FormalVariantPayloadV3, ...]:
    schedule = _snapshot_variant_schedule_v3(schedule)
    snapshot = _snapshot_and_validate_build_success_for_variants_v3(build_success)
    bases = {
        row.envelope_id: row for row in snapshot.variant_context.base_envelopes
    }
    if set(schedule.case_order_envelope_ids) != set(bases):
        raise ValueError("variant case-order schedule is not an exact batch permutation")

    base_payloads: dict[str, dict[str, object]] = {}
    for envelope_id, base in bases.items():
        decoded = _restore_json_integers_v3(
            decode_and_require_canonical_rfc8785_v1(base.canonical_payload_bytes)
        )
        if type(decoded) is not dict:
            raise ValueError("variant base payload root is not an object")
        normalized = _normalize_formal_payload_structure_v3(decoded)
        if normalized != decoded:
            raise ValueError("variant base payload bytes are not normalized")
        base_payloads[envelope_id] = decoded

    uuid_domain: dict[str, str] = {}
    for envelope_id in tuple(bases):
        graph = base_payloads[envelope_id]["public_case"]["graph"]  # type: ignore[index]
        for _, namespace, value in _walk_uuid_leaves_v3(graph):
            previous = uuid_domain.setdefault(value, namespace)
            if previous != namespace:
                raise ValueError("variant base has cross-namespace UUID reuse")
    private_source_pairs = {
        (row.namespace, row.source_uuid)
        for link in snapshot.private_alignment.links_in_public_order
        for row in link.source_to_public_uuid_rows
    }
    private_source_uuids = {
        source_uuid for _, source_uuid in private_source_pairs
    }
    if set(uuid_domain) & private_source_uuids:
        raise ValueError("variant base public UUID domain intersects private source UUIDs")
    if schedule.uuid_renamings:
        if schedule.uuid_renamings != tuple(
            sorted(schedule.uuid_renamings, key=lambda row: (row.namespace, row.source_uuid))
        ):
            raise ValueError("variant global UUID bijection rows are not canonical")
        if {row.source_uuid for row in schedule.uuid_renamings} != set(uuid_domain):
            raise ValueError("variant global UUID bijection does not exactly cover the batch domain")
        if len({row.target_uuid for row in schedule.uuid_renamings}) != len(uuid_domain):
            raise ValueError("variant global UUID targets are not injective")
        if any(uuid_domain[row.source_uuid] != row.namespace for row in schedule.uuid_renamings):
            raise ValueError("variant global UUID bijection changes namespace")
        if any(
            row.target_uuid in private_source_uuids
            or (row.namespace, row.target_uuid) in private_source_pairs
            for row in schedule.uuid_renamings
        ):
            raise ValueError(
                "variant target UUID domain intersects private source UUIDs"
            )
        renamings = {
            (row.namespace, row.source_uuid): row.target_uuid
            for row in schedule.uuid_renamings
        }
    else:
        renamings = {
            (namespace, value): value for value, namespace in uuid_domain.items()
        }

    if schedule.observation_orders != tuple(
        sorted(schedule.observation_orders, key=lambda row: row.envelope_id)
    ) or len(schedule.observation_orders) != len(bases) or {
        row.envelope_id for row in schedule.observation_orders
    } != set(bases):
        raise ValueError("variant observation schedules must canonically cover every envelope")
    observation_orders = {
        row.envelope_id: row.observation_ids for row in schedule.observation_orders
    }

    rows_by_id: dict[str, FormalVariantPayloadV3] = {}
    for envelope_id, base in bases.items():
        base_payload = base_payloads[envelope_id]
        base_graph = base_payload["public_case"]["graph"]  # type: ignore[index]
        base_observations = base_graph["observations"]  # type: ignore[index]
        base_observation_ids = tuple(
            observation["observation_id"] for observation in base_observations
        )
        requested_order = observation_orders[envelope_id]
        if set(requested_order) != set(base_observation_ids) or len(
            requested_order
        ) != len(base_observation_ids):
            raise ValueError("variant observation order is not an exact case permutation")
        by_id = {
            observation["observation_id"]: observation
            for observation in base_observations
        }
        graph = deepcopy(base_graph)
        graph["observations"] = [deepcopy(by_id[value]) for value in requested_order]
        renamed_graph = _rename_graph_ids_v3(graph, renamings)
        proposed = deepcopy(base_payload)
        proposed["public_case"]["graph"] = renamed_graph  # type: ignore[index]
        proposed["public_case"]["public_provenance_commitment"] = "sha256:" + "0" * 64  # type: ignore[index]
        normalized = _normalize_formal_payload_structure_v3(proposed)

        restored = deepcopy(normalized)
        restored_graph = _rename_graph_ids_v3(
            restored["public_case"]["graph"],  # type: ignore[index,arg-type]
            {
                (namespace, target): source
                for (namespace, source), target in renamings.items()
            },
        )
        restored_by_id = {
            observation["observation_id"]: observation
            for observation in restored_graph["observations"]  # type: ignore[index]
        }
        restored_graph["observations"] = [
            restored_by_id[value] for value in base_observation_ids
        ]
        restored["public_case"]["graph"] = restored_graph  # type: ignore[index]
        restored["public_case"]["public_provenance_commitment"] = "sha256:" + "0" * 64  # type: ignore[index]
        if _normalize_formal_payload_structure_v3(restored) != base_payload:
            raise ValueError("variant inverse-map replay changed base public semantics")
        rows_by_id[envelope_id] = FormalVariantPayloadV3(
            envelope_id, normalized, base.padding_bytes
        )
    return tuple(rows_by_id[value] for value in schedule.case_order_envelope_ids)


def make_formal_variant_payloads_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    schedule: FormalVariantScheduleV3,
) -> tuple[FormalVariantPayloadV3, ...]:
    try:
        return _validated_variant_payload_rows_v3(
            build_success=build_success,
            schedule=schedule,
        )
    except (RecursionError, MemoryError):
        raise ValueError("bounded formal variant issuance rejected") from None


def _snapshot_variant_payload_mapping_v3(
    payload: dict[str, object],
) -> dict[str, object]:
    if type(payload) is not dict:
        raise TypeError("formal variant payload must be exact dict")
    snapshot_bytes = canonicalize_rfc8785_v1(payload)
    snapshot = _restore_json_integers_v3(
        decode_and_require_canonical_rfc8785_v1(snapshot_bytes)
    )
    if type(snapshot) is not dict:
        raise TypeError("formal variant payload snapshot root type drift")
    return snapshot


def _normalize_variant_payload_against_expected_v3(
    *,
    payload_snapshot: dict[str, object],
    expected: dict[str, FormalVariantPayloadV3],
    envelope_id: str,
) -> dict[str, object]:
    if envelope_id not in expected:
        raise ValueError("formal variant envelope ID is outside the full build batch")
    normalized = _normalize_formal_payload_structure_v3(payload_snapshot)
    if normalized != expected[envelope_id].payload:
        raise ValueError(
            "formal variant contains a semantic/root change outside its explicit schedule"
        )
    return deepcopy(expected[envelope_id].payload)


def normalize_formal_variant_payload_v3(
    payload: dict[str, object],
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    schedule: FormalVariantScheduleV3,
    envelope_id: str,
) -> dict[str, object]:
    try:
        payload_snapshot = _snapshot_variant_payload_mapping_v3(payload)
        expected = {
            row.envelope_id: row
            for row in _validated_variant_payload_rows_v3(
                build_success=build_success,
                schedule=schedule,
            )
        }
        return _normalize_variant_payload_against_expected_v3(
            payload_snapshot=payload_snapshot,
            expected=expected,
            envelope_id=envelope_id,
        )
    except (RecursionError, MemoryError):
        raise ValueError("bounded formal variant normalization rejected") from None


def reencode_formal_variant_payload_envelope_v3(
    *,
    payload: dict[str, object],
    padding_bytes: bytes,
    build_success: FormalTrustedWireBuildSuccessV3,
    schedule: FormalVariantScheduleV3,
    envelope_id: str,
) -> bytes:
    try:
        if type(padding_bytes) is not bytes:
            raise TypeError("formal variant padding must be exact bytes")
        padding_snapshot = bytes(padding_bytes)
        payload_snapshot = _snapshot_variant_payload_mapping_v3(payload)
        expected = {
            row.envelope_id: row
            for row in _validated_variant_payload_rows_v3(
                build_success=build_success,
                schedule=schedule,
            )
        }
        if (
            envelope_id not in expected
            or padding_snapshot != expected[envelope_id].padding_bytes
        ):
            raise ValueError(
                "formal variant padding/envelope is outside the full build batch"
            )
        normalized = _normalize_variant_payload_against_expected_v3(
            payload_snapshot=payload_snapshot,
            expected=expected,
            envelope_id=envelope_id,
        )
        envelope = _frame_payload_with_padding_v3(
            canonicalize_rfc8785_v1(normalized), padding_snapshot
        )
        decode_and_audit_formal_envelope_v3(envelope)
        return envelope
    except (RecursionError, MemoryError):
        raise ValueError("bounded formal variant reencode rejected") from None


def reencode_formal_variant_batch_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    schedule: FormalVariantScheduleV3,
) -> tuple[bytes, ...]:
    try:
        rows = _validated_variant_payload_rows_v3(
            build_success=build_success,
            schedule=schedule,
        )
        envelopes = tuple(
            _frame_payload_with_padding_v3(
                canonicalize_rfc8785_v1(row.payload), row.padding_bytes
            )
            for row in rows
        )
        for envelope in envelopes:
            decode_and_audit_formal_envelope_v3(envelope)
        return envelopes
    except (RecursionError, MemoryError):
        raise ValueError("bounded formal variant batch reencode rejected") from None


def _derive_keys_v3(run_id: bytes, keys: FormalWireKeySourcesV3) -> tuple[bytes, bytes, bytes]:
    salt = hashlib.sha256(_RUN_DOMAIN_V3 + run_id).digest()
    result = tuple(
        hmac.digest(hmac.digest(salt, ikm, "sha256"), info + b"\x01", "sha256")
        for ikm, info in zip((keys.shuffle_ikm, keys.id_ikm, keys.padding_ikm), (_SHUFFLE_INFO_V3, _ID_INFO_V3, _PADDING_INFO_V3), strict=True)
    )
    if len(set(result)) != 3: raise ValueError("derived purpose keys collided")
    return result  # type: ignore[return-value]


class _ShuffleStreamV3:
    def __init__(self, key: bytes, run_id: bytes) -> None:
        self.key, self.run_id, self.counter = key, run_id, 0
    def next_u64(self) -> int:
        value = hmac.digest(self.key, _SHUFFLE_DOMAIN_V3 + self.run_id + self.counter.to_bytes(8, "big"), "sha256")
        self.counter += 1
        return int.from_bytes(value[:8], "big")


def _uniform_below_v3(stream: _ShuffleStreamV3, bound: int) -> int:
    limit = (1 << 64) - ((1 << 64) % bound)
    for _ in range(MAXIMUM_SHUFFLE_REJECTION_DRAWS_V3):
        value = stream.next_u64()
        if value < limit: return value % bound
    raise ValueError("shuffle rejection draw budget exhausted")


def _shuffle_indices_v3(count: int, key: bytes, run_id: bytes) -> tuple[int, ...]:
    values = list(range(count)); stream = _ShuffleStreamV3(key, run_id)
    for index in range(count - 1, 0, -1):
        other = _uniform_below_v3(stream, index + 1)
        values[index], values[other] = values[other], values[index]
    return tuple(values)


def id_frame_v3(run_id: bytes, namespace: str, counter: int, retry: int) -> bytes:
    if type(run_id) is not bytes or len(run_id) != RUN_ID_BYTES_V3: raise ValueError("ID frame run ID length drift")
    if namespace not in NAMESPACE_REGISTRY_V3: raise ValueError("ID frame namespace outside exact-10")
    if type(counter) is not int or not 0 <= counter < (1 << 64) or type(retry) is not int or not 0 <= retry <= MAXIMUM_UUID_COLLISION_RETRIES_V3:
        raise ValueError("ID frame counter/retry out of range")
    encoded = namespace.encode("ascii")
    return ID_FRAME_DOMAIN_V3 + run_id + len(encoded).to_bytes(2, "big") + encoded + counter.to_bytes(8, "big") + retry.to_bytes(1, "big")


def _uuid_candidate_v3(key: bytes, run_id: bytes, namespace: str, counter: int, retry: int) -> str:
    raw = bytearray(hmac.digest(key, id_frame_v3(run_id, namespace, counter, retry), "sha256")[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40; raw[8] = (raw[8] & 0x3F) | 0x80
    text = raw.hex(); return f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"


def _padding_v3(key: bytes, run_id: bytes, payload_digest: bytes, length: int) -> bytes:
    result = bytearray(); counter = 0
    while len(result) < length:
        result.extend(hmac.digest(key, _PADDING_DOMAIN_V3 + run_id + payload_digest + counter.to_bytes(4, "big"), "sha256")); counter += 1
    return bytes(result[:length])


def _run_commitment_v3(run_id: bytes) -> str:
    return "sha256:" + hashlib.sha256(_RUN_DOMAIN_V3 + run_id).hexdigest()


def _source_transform_use_hash_from_mappings_v3(
    aggregate_map: dict[str, object],
    transform: dict[str, object],
) -> str:
    _mapping(
        aggregate_map,
        {item.name for item in fields(FormalAggregateMapRecordV3)},
        "exact source aggregate-map catalog profile",
    )
    _mapping(
        transform,
        {item.name for item in fields(FormalTransformContractV3)},
        "exact source transform catalog profile",
    )
    body = {
        "aggregate_map": deepcopy(aggregate_map),
        "transform": deepcopy(transform),
    }
    return "sha256:" + hashlib.sha256(
        _SOURCE_TRANSFORM_USE_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def formal_source_transform_use_hash_v3(
    source_use: FormalSourceTransformUseV3,
) -> str:
    """Hash one exact canonical source aggregate-map/transform use, UUIDs included."""
    if type(source_use) is not FormalSourceTransformUseV3:
        raise TypeError("source transform-use hash needs exact FormalSourceTransformUseV3")
    source_use.__post_init__()
    return _source_transform_use_hash_from_mappings_v3(
        _profile_value_v3(source_use.aggregate_map),  # type: ignore[arg-type]
        _profile_value_v3(source_use.transform),  # type: ignore[arg-type]
    )


def _catalog_entry_commitment_v3(
    entry: FormalGlobalTransformCatalogEntryV3,
) -> str:
    entry.__post_init__()
    return "sha256:" + hashlib.sha256(
        _TRANSFORM_CATALOG_BINDING_DOMAIN_V3
        + canonicalize_rfc8785_v1(_profile_value_v3(entry))
    ).hexdigest()


def formal_global_transform_catalog_root_v3(
    catalog: FormalGlobalTransformCatalogV3,
) -> str:
    if type(catalog) is not FormalGlobalTransformCatalogV3:
        raise TypeError("global transform catalog exact type drift")
    catalog.__post_init__()
    body = {
        "entries": [_profile_value_v3(entry) for entry in catalog.entries],
        "schema_version": catalog.schema_version,
    }
    return "sha256:" + hashlib.sha256(
        _TRANSFORM_CATALOG_ROOT_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def _catalog_binding_body_v3(
    *,
    root: str,
    entry_commitments: tuple[str, ...],
) -> dict[str, object]:
    return {
        "catalog_entry_commitments": list(entry_commitments),
        "schema_version": FORMAL_GLOBAL_TRANSFORM_CATALOG_BINDING_V3_SCHEMA_VERSION,
        "transform_catalog_root": root,
    }


def issue_formal_global_transform_catalog_v3(
    *,
    source_cases: tuple[FormalSourceCaseV3, ...],
) -> tuple[
    FormalGlobalTransformCatalogV3,
    FormalGlobalTransformCatalogBindingReceiptV3,
]:
    """Issue a private closed catalog and its content-addressed binding receipt."""
    if type(source_cases) is not tuple or not source_cases:
        raise TypeError("catalog issuance needs an exact nonempty source-case tuple")
    entries: set[FormalGlobalTransformCatalogEntryV3] = set()
    for source_case in source_cases:
        if type(source_case) is not FormalSourceCaseV3:
            raise TypeError("catalog source case exact type drift")
        source_case.__post_init__()
        mapping = _canonicalize_graph_mapping_v3(
            _profile_value_v3(source_case.graph),  # type: ignore[arg-type]
            preserve_observation_order=False,
        )
        transforms = {
            transform["transform_id"]: transform
            for transform in mapping["transforms"]  # type: ignore[index,union-attr]
        }
        for aggregate_map in mapping["aggregate_maps"]:  # type: ignore[index,union-attr]
            transform = transforms[aggregate_map["transform_id"]]
            entries.add(
                FormalGlobalTransformCatalogEntryV3(
                    transform["operation"],  # type: ignore[index,arg-type]
                    _source_transform_use_hash_from_mappings_v3(
                        aggregate_map, transform  # type: ignore[arg-type]
                    ),
                )
            )
    catalog = FormalGlobalTransformCatalogV3(
        FORMAL_GLOBAL_TRANSFORM_CATALOG_V3_SCHEMA_VERSION,
        tuple(sorted(entries)),
    )
    root = formal_global_transform_catalog_root_v3(catalog)
    entry_commitments = tuple(sorted(_catalog_entry_commitment_v3(entry) for entry in catalog.entries))
    body = _catalog_binding_body_v3(root=root, entry_commitments=entry_commitments)
    receipt = FormalGlobalTransformCatalogBindingReceiptV3(
        FORMAL_GLOBAL_TRANSFORM_CATALOG_BINDING_V3_SCHEMA_VERSION,
        root,
        entry_commitments,
        "phase2b_formal_global_transform_catalog_binding_v3_"
        + hashlib.sha256(
            _TRANSFORM_CATALOG_BINDING_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
        ).hexdigest(),
    )
    verify_formal_global_transform_catalog_binding_v3(
        catalog=catalog,
        binding_receipt=receipt,
    )
    return catalog, receipt


def verify_formal_global_transform_catalog_binding_v3(
    *,
    catalog: FormalGlobalTransformCatalogV3,
    binding_receipt: FormalGlobalTransformCatalogBindingReceiptV3,
) -> bool:
    if type(catalog) is not FormalGlobalTransformCatalogV3 or type(
        binding_receipt
    ) is not FormalGlobalTransformCatalogBindingReceiptV3:
        raise TypeError("catalog binding verifier exact type drift")
    catalog.__post_init__(); binding_receipt.__post_init__()
    root = formal_global_transform_catalog_root_v3(catalog)
    entry_commitments = tuple(sorted(_catalog_entry_commitment_v3(entry) for entry in catalog.entries))
    body = _catalog_binding_body_v3(root=root, entry_commitments=entry_commitments)
    expected_id = "phase2b_formal_global_transform_catalog_binding_v3_" + hashlib.sha256(
        _TRANSFORM_CATALOG_BINDING_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
    ).hexdigest()
    if (
        binding_receipt.transform_catalog_root != root
        or binding_receipt.catalog_entry_commitments != entry_commitments
        or binding_receipt.receipt_id != expected_id
    ):
        raise ValueError("global transform catalog binding receipt mismatch")
    return True


def source_case_commitment_v3(source_case: FormalSourceCaseV3, *, transform_catalog_root: str) -> str:
    if type(source_case) is not FormalSourceCaseV3: raise TypeError("source case exact type drift")
    _digest(transform_catalog_root, "global transform catalog root")
    source_case.__post_init__()
    mapping = _canonicalize_graph_mapping_v3(_profile_value_v3(source_case.graph), preserve_observation_order=False)
    _refresh_graph_provenance_v3(mapping, transform_catalog_root)
    _graph_from_mapping_v3(mapping)
    return "sha256:" + hashlib.sha256(_SOURCE_CASE_DOMAIN_V3 + canonicalize_rfc8785_v1(mapping)).hexdigest()


def _batch_id_v3(
    context: FormalBatchContextV3,
    key_source_receipt_id: str,
    envelope_ids: tuple[str, ...],
) -> str:
    _prefixed_digest(
        key_source_receipt_id,
        "phase2b_key_source_transcript_v3_",
        "batch ID key receipt ID",
    )
    body = {
        "batch_context": _context_mapping_v3(context),
        "envelope_ids_in_public_order": list(envelope_ids),
        "key_source_receipt_id": key_source_receipt_id,
        "schema_version": FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,
    }
    return "phase2b_formal_trusted_wire_batch_v3_" + hashlib.sha256(_BATCH_ID_DOMAIN_V3 + canonicalize_rfc8785_v1(body)).hexdigest()


def _alignment_id_v3(alignment_body: dict[str, object]) -> str:
    return "phase2b_formal_private_audit_alignment_v3_" + hashlib.sha256(_ALIGNMENT_DOMAIN_V3 + canonicalize_rfc8785_v1(alignment_body)).hexdigest()


def _alignment_body_v3(
    *,
    batch_id: str,
    run_id_commitment: str,
    key_source_receipt_id: str,
    source_case_count: int,
    links: tuple[FormalPrivateAuditAlignmentLinkV3, ...],
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "key_source_receipt_id": key_source_receipt_id,
        "links_in_public_order": [_profile_value_v3(link) for link in links],
        "run_id_commitment": run_id_commitment,
        "source_case_count": source_case_count,
    }


def _label_binding_id_v3(
    *,
    batch_id: str,
    public_envelope_sha256: str,
    source_row_index: int,
    source_case_commitment: str,
    source_to_public_uuid_rows: tuple[FormalPrivateUuidMappingV3, ...],
) -> str:
    body = {
        "batch_id": batch_id,
        "public_envelope_sha256": public_envelope_sha256,
        "source_case_commitment": source_case_commitment,
        "source_row_index": source_row_index,
        "source_to_public_uuid_rows": [
            _profile_value_v3(row) for row in source_to_public_uuid_rows
        ],
    }
    return "sha256:" + hashlib.sha256(
        _ALIGNMENT_LINK_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def _variant_context_id_v3(
    *,
    batch_id: str,
    private_alignment_id: str,
    transform_catalog_root: str,
    catalog_binding_receipt_id: str,
    base_envelopes: tuple[_FormalVariantBaseEnvelopeV3, ...],
) -> str:
    body = {
        "base_envelopes": [
            {
                "envelope_id": row.envelope_id,
                "padding_sha256": "sha256:"
                + hashlib.sha256(row.padding_bytes).hexdigest(),
                "payload_sha256": "sha256:"
                + hashlib.sha256(row.canonical_payload_bytes).hexdigest(),
            }
            for row in base_envelopes
        ],
        "batch_id": batch_id,
        "catalog_binding_receipt_id": catalog_binding_receipt_id,
        "private_alignment_id": private_alignment_id,
        "transform_catalog_root": transform_catalog_root,
    }
    return "phase2b_formal_variant_context_v3_" + hashlib.sha256(
        _VARIANT_CONTEXT_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def _build_success_id_v3(
    *,
    public_batch: FormalTrustedWireBatchV3,
    key_source_receipt: KeySourceTranscriptReceiptV3,
    private_alignment: FormalPrivateAuditAlignmentV3,
    variant_context: _FormalVariantContextV3,
) -> str:
    body = {
        "batch_id": public_batch.batch_id,
        "key_source_receipt": _profile_value_v3(key_source_receipt),
        "private_alignment_id": private_alignment.alignment_id,
        "variant_context_id": variant_context.context_id,
        "schema_version": FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,
    }
    return "phase2b_formal_trusted_wire_build_success_v3_" + hashlib.sha256(
        _BUILD_SUCCESS_DOMAIN_V3 + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def _build_success_snapshot_bytes_v3(
    build_success: FormalTrustedWireBuildSuccessV3,
) -> bytes:
    """Serialize a complete content-bound projection, including hidden bytes."""
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("build success snapshot needs exact type")
    batch = build_success.public_batch
    context = build_success.variant_context
    body = {
        "batch": {
            "batch_context": _context_mapping_v3(batch.batch_context),
            "batch_id": batch.batch_id,
            "claim_level": batch.claim_level,
            "envelopes": [
                {
                    "bytes": len(row.envelope),
                    "envelope_id": row.envelope_id,
                    "sha256": "sha256:" + hashlib.sha256(row.envelope).hexdigest(),
                }
                for row in batch.envelopes
            ],
            "key_source_receipt_id": batch.key_source_receipt_id,
            "schema_version": batch.schema_version,
        },
        "build_claims": {
            "builder_input_catalog_membership_checked": build_success.builder_input_catalog_membership_checked,
            "builder_input_transcript_consistency_checked": build_success.builder_input_transcript_consistency_checked,
            "origin_authenticated": build_success.origin_authenticated,
            "private_alignment_linkage_checked": build_success.private_alignment_linkage_checked,
        },
        "key_source_receipt": _profile_value_v3(build_success.key_source_receipt),
        "private_alignment": _profile_value_v3(build_success.private_audit_alignment),
        "success_id": build_success.success_id,
        "variant_context": {
            "base_envelopes": [
                {
                    "envelope_id": row.envelope_id,
                    "padding_bytes": len(row.padding_bytes),
                    "padding_sha256": "sha256:" + hashlib.sha256(row.padding_bytes).hexdigest(),
                    "payload_bytes": len(row.canonical_payload_bytes),
                    "payload_sha256": "sha256:" + hashlib.sha256(row.canonical_payload_bytes).hexdigest(),
                }
                for row in context.base_envelopes
            ],
            "batch_id": context.batch_id,
            "catalog": _profile_value_v3(context.global_transform_catalog),
            "catalog_binding_receipt": _profile_value_v3(
                context.transform_catalog_binding_receipt
            ),
            "catalog_binding_receipt_id": context.catalog_binding_receipt_id,
            "context_id": context.context_id,
            "origin_authenticated": context.origin_authenticated,
            "private_alignment_id": context.private_alignment_id,
            "transform_catalog_root": context.transform_catalog_root,
        },
    }
    return canonicalize_rfc8785_v1(body)


def verify_formal_trusted_wire_build_success_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    run_id: bytes,
    key_sources: FormalWireKeySourcesV3,
    source_cases: tuple[FormalSourceCaseV3, ...],
    global_transform_catalog: FormalGlobalTransformCatalogV3,
    transform_catalog_binding_receipt: FormalGlobalTransformCatalogBindingReceiptV3,
) -> bool:
    """Freshly verify a cooperative build result against caller-held evidence.

    The verifier deterministically rebuilds shuffle, ID allocation, padding,
    public envelopes, private alignment, and private variant context from the
    caller-held run ID, key sources, source cases, and catalog.  The returned
    Boolean establishes exact content/key/catalog consistency for these
    supplied objects.  It does not authenticate who generated the keys,
    catalog, cases, labels, or build result.
    """
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("fresh build verifier needs exact build success")
    supplied_snapshot = _snapshot_and_validate_build_success_for_variants_v3(
        build_success
    )
    supplied = FormalTrustedWireBuildSuccessV3(
        supplied_snapshot.public_batch,
        supplied_snapshot.key_source_receipt,
        supplied_snapshot.private_alignment,
        supplied_snapshot.variant_context,
        supplied_snapshot.success_id,
        True,
        True,
        True,
        False,
    )
    if type(run_id) is not bytes or len(run_id) != RUN_ID_BYTES_V3:
        raise ValueError("fresh build verifier run ID must be exact 32 bytes")
    if supplied.public_batch.batch_context.run_id_commitment != _run_commitment_v3(
        run_id
    ):
        raise ValueError("fresh build verifier run ID is stale or spliced")
    if type(key_sources) is not FormalWireKeySourcesV3:
        raise TypeError("fresh build verifier needs exact caller-held key sources")
    _verify_key_receipt_v3(key_sources, supplied.key_source_receipt)
    if type(global_transform_catalog) is not FormalGlobalTransformCatalogV3 or type(
        transform_catalog_binding_receipt
    ) is not FormalGlobalTransformCatalogBindingReceiptV3:
        raise TypeError("fresh build verifier needs exact catalog evidence")
    verify_formal_global_transform_catalog_binding_v3(
        catalog=global_transform_catalog,
        binding_receipt=transform_catalog_binding_receipt,
    )
    variant_context = supplied.variant_context
    if (
        global_transform_catalog != variant_context.global_transform_catalog
        or transform_catalog_binding_receipt
        != variant_context.transform_catalog_binding_receipt
    ):
        raise ValueError("fresh build verifier catalog evidence is stale or spliced")
    if type(source_cases) is not tuple or len(source_cases) != len(
        supplied.public_batch.envelopes
    ) or any(type(case) is not FormalSourceCaseV3 for case in source_cases):
        raise ValueError("fresh build verifier source cases do not exactly cover batch")
    expected_commitments = tuple(
        source_case_commitment_v3(
            case,
            transform_catalog_root=transform_catalog_binding_receipt.transform_catalog_root,
        )
        for case in source_cases
    )
    alignment = supplied.private_audit_alignment
    if expected_commitments != alignment.source_case_commitments_in_source_order:
        raise ValueError("fresh build verifier source commitments are stale or reordered")
    allowed = {
        (entry.operation, entry.exact_source_transform_use_sha256)
        for entry in global_transform_catalog.entries
    }
    links_by_source = {
        link.source_row_index: link for link in alignment.links_in_public_order
    }
    for source_index, case in enumerate(source_cases):
        mapping = _canonicalize_graph_mapping_v3(
            _profile_value_v3(case.graph), preserve_observation_order=False
        )
        source_uuid_rows = {
            (namespace, source_uuid)
            for _, namespace, source_uuid in _walk_uuid_leaves_v3(mapping)
        }
        link_rows = {
            (row.namespace, row.source_uuid)
            for row in links_by_source[source_index].source_to_public_uuid_rows
        }
        if source_uuid_rows != link_rows:
            raise ValueError("fresh build verifier private exact-10 UUID mapping is incomplete")
        transforms = {
            transform["transform_id"]: transform
            for transform in mapping["transforms"]  # type: ignore[index,union-attr]
        }
        for aggregate_map in mapping["aggregate_maps"]:  # type: ignore[index,union-attr]
            transform = transforms[aggregate_map["transform_id"]]  # type: ignore[index]
            member = (
                transform["operation"],  # type: ignore[index]
                _source_transform_use_hash_from_mappings_v3(
                    aggregate_map, transform  # type: ignore[arg-type]
                ),
            )
            if member not in allowed:
                raise ValueError("fresh build verifier source transform use leaves catalog")
    fresh = build_formal_trusted_wire_batch_v3(
        source_cases=source_cases,
        global_transform_catalog=global_transform_catalog,
        transform_catalog_binding_receipt=transform_catalog_binding_receipt,
        run_id=run_id,
        key_sources=key_sources,
        key_source_receipt=supplied.key_source_receipt,
    )
    if type(fresh) is not FormalTrustedWireBuildSuccessV3:
        raise ValueError("fresh build verifier deterministic replay rejected")
    fresh._validate_issued_shape()
    if _build_success_snapshot_bytes_v3(fresh) != _build_success_snapshot_bytes_v3(
        supplied
    ):
        raise ValueError("fresh build verifier replay differs from supplied success")
    return True


def verify_private_audit_alignment_v3(
    *,
    batch: FormalTrustedWireBatchV3,
    alignment: FormalPrivateAuditAlignmentV3,
) -> bool:
    if type(batch) is not FormalTrustedWireBatchV3 or type(alignment) is not FormalPrivateAuditAlignmentV3:
        raise TypeError("alignment verifier requires exact wire types")
    batch.__post_init__()
    alignment._validate_issued_shape()
    count = len(batch.envelopes)
    if (
        alignment.batch_id != batch.batch_id
        or alignment.run_id_commitment != batch.batch_context.run_id_commitment
        or alignment.key_source_receipt_id != batch.key_source_receipt_id
        or alignment.source_case_count != count
        or len(alignment.links_in_public_order) != count
        or tuple(sorted(alignment.source_row_index_by_public_row)) != tuple(range(count))
        or len(set(alignment.source_case_commitments_in_source_order)) != count
    ):
        raise ValueError("private audit alignment coverage/bijection/batch binding failed")
    actual_sha = tuple("sha256:" + hashlib.sha256(row.envelope).hexdigest() for row in batch.envelopes)
    if alignment.public_envelope_sha256_in_order != actual_sha:
        raise ValueError("private audit alignment envelope splice detected")
    for public_index, (envelope, link) in enumerate(
        zip(batch.envelopes, alignment.links_in_public_order, strict=True)
    ):
        public_uuid_rows = {
            (namespace, value)
            for _, namespace, value in _walk_uuid_leaves_v3(
                _profile_value_v3(envelope.public_case.graph)  # type: ignore[arg-type]
            )
        }
        aligned_public_rows = {
            (row.namespace, row.public_uuid)
            for row in link.source_to_public_uuid_rows
        }
        if public_uuid_rows != aligned_public_rows:
            raise ValueError(
                f"private UUID mapping/public envelope coverage mismatch at row {public_index}"
            )
    body = _alignment_body_v3(
        batch_id=alignment.batch_id,
        run_id_commitment=alignment.run_id_commitment,
        key_source_receipt_id=alignment.key_source_receipt_id,
        source_case_count=alignment.source_case_count,
        links=alignment.links_in_public_order,
    )
    if alignment.alignment_id != _alignment_id_v3(body):
        raise ValueError("private audit alignment content ID mismatch")
    return True


def issue_formal_private_audit_rows_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    private_payloads_in_source_order: tuple[bytes, ...],
) -> tuple[FormalPrivateAuditRowV3, ...]:
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("private audit row issuance needs exact wire build success")
    build_success._validate_issued_shape()
    alignment = build_success.private_audit_alignment
    if type(private_payloads_in_source_order) is not tuple or len(
        private_payloads_in_source_order
    ) != alignment.source_case_count or any(
        type(value) is not bytes for value in private_payloads_in_source_order
    ):
        raise ValueError("private payload bytes do not exactly cover source rows")
    links_by_source = {
        link.source_row_index: link for link in alignment.links_in_public_order
    }
    rows: list[FormalPrivateAuditRowV3] = []
    for source_index, private_payload in enumerate(private_payloads_in_source_order):
        link = links_by_source[source_index]
        binding = "sha256:" + hashlib.sha256(
            _PRIVATE_ROW_DOMAIN_V3
            + link.source_case_commitment.encode("ascii")
            + link.label_binding_id.encode("ascii")
            + len(private_payload).to_bytes(8, "big")
            + private_payload
        ).hexdigest()
        rows.append(
            FormalPrivateAuditRowV3(
                source_case_commitment=link.source_case_commitment,
                label_binding_id=link.label_binding_id,
                private_payload_bytes=private_payload,
                row_binding=binding,
                semantic_truth_verified=False,
                origin_authenticated=False,
            )
        )
    return tuple(rows)


def align_private_rows_to_public_v3(
    *,
    batch: FormalTrustedWireBatchV3,
    alignment: FormalPrivateAuditAlignmentV3,
    private_rows_in_source_order: tuple[FormalPrivateAuditRowV3, ...],
) -> tuple[FormalPrivateAuditRowV3, ...]:
    verify_private_audit_alignment_v3(batch=batch, alignment=alignment)
    if type(private_rows_in_source_order) is not tuple or len(
        private_rows_in_source_order
    ) != alignment.source_case_count or any(
        type(row) is not FormalPrivateAuditRowV3
        for row in private_rows_in_source_order
    ):
        raise TypeError("private aligned rows must be exact issued DTOs with full coverage")
    links_by_source = {
        link.source_row_index: link for link in alignment.links_in_public_order
    }
    for source_index, row in enumerate(private_rows_in_source_order):
        row._validate_issued_shape()
        link = links_by_source[source_index]
        if (
            row.source_case_commitment != link.source_case_commitment
            or row.label_binding_id != link.label_binding_id
        ):
            raise ValueError("private row is swapped, stale, or source/link-spliced")
    return tuple(
        private_rows_in_source_order[link.source_row_index]
        for link in alignment.links_in_public_order
    )


def _project_source_label_with_link_v3(
    *,
    link: FormalPrivateAuditAlignmentLinkV3,
    canonical_source_label_bytes: bytes,
) -> bytes:
    if type(canonical_source_label_bytes) is not bytes:
        raise TypeError("private source label must be canonical RFC8785 bytes")
    decoded = _restore_json_integers_v3(
        decode_and_require_canonical_rfc8785_v1(canonical_source_label_bytes)
    )
    if type(decoded) is not dict:
        raise ValueError("private source label root must be a closed JSON object")
    mappings = {
        row.source_uuid: (row.namespace, row.public_uuid)
        for row in link.source_to_public_uuid_rows
    }

    def project(value: object, field_name: str | None = None) -> object:
        if type(value) is dict:
            result: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str or (
                    _UUID_TOKEN_RE.search(key)
                    or _SHA256_TOKEN_RE.search(key)
                    or _HEX_DIGEST_TOKEN_RE.search(key)
                ):
                    raise ValueError("private label field name contains an embedded identifier/digest")
                result[key] = project(item, key)
            return result
        if type(value) is list:
            return [project(item, field_name) for item in value]
        if type(value) is str:
            if _UUID_SHAPED_RE.fullmatch(value):
                expected_namespace = _ID_FIELD_NAMESPACE_V3.get(field_name or "")
                if value not in mappings:
                    raise ValueError("private label contains a foreign/unmapped UUID")
                actual_namespace, public_uuid = mappings[value]
                if expected_namespace is None or expected_namespace != actual_namespace:
                    raise ValueError("private label UUID field crosses its frozen namespace")
                return public_uuid
            if (
                _UUID_TOKEN_RE.search(value)
                or _SHA256_TOKEN_RE.search(value)
                or _HEX_DIGEST_TOKEN_RE.search(value)
            ):
                raise ValueError("private label string contains an embedded identifier/digest")
        return value

    projected = project(decoded)
    if type(projected) is not dict:
        raise TypeError("projected private label root type drift")
    return canonicalize_rfc8785_v1(projected)


def project_private_source_label_to_public_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    source_row_index: int,
    canonical_source_label_bytes: bytes,
) -> bytes:
    """Project one canonical private label through an issued row mapping.

    This proves only exact in-process source/public identifier linkage.  It does
    not authenticate the label origin or establish that its non-ID semantics are
    a true answer.
    """
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("private label projection needs exact wire build success")
    build_success._validate_issued_shape()
    if type(source_row_index) is not int or not 0 <= source_row_index < len(
        build_success.private_audit_alignment.links_in_public_order
    ):
        raise ValueError("private label source row index is outside the issued batch")
    links = {
        link.source_row_index: link
        for link in build_success.private_audit_alignment.links_in_public_order
    }
    return _project_source_label_with_link_v3(
        link=links[source_row_index],
        canonical_source_label_bytes=canonical_source_label_bytes,
    )


def project_private_source_labels_to_public_order_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    canonical_source_label_bytes_in_source_order: tuple[bytes, ...],
) -> tuple[bytes, ...]:
    """Project and shuffle a complete canonical private-label batch once."""
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("private label batch projection needs exact wire build success")
    build_success._validate_issued_shape()
    alignment = build_success.private_audit_alignment
    if type(canonical_source_label_bytes_in_source_order) is not tuple or len(
        canonical_source_label_bytes_in_source_order
    ) != alignment.source_case_count or any(
        type(value) is not bytes
        for value in canonical_source_label_bytes_in_source_order
    ):
        raise ValueError("private label bytes do not exactly cover source rows")
    return tuple(
        _project_source_label_with_link_v3(
            link=link,
            canonical_source_label_bytes=canonical_source_label_bytes_in_source_order[
                link.source_row_index
            ],
        )
        for link in alignment.links_in_public_order
    )


def issue_projected_private_audit_rows_v3(
    *,
    build_success: FormalTrustedWireBuildSuccessV3,
    canonical_source_label_bytes_in_source_order: tuple[bytes, ...],
) -> tuple[FormalPrivateAuditRowV3, ...]:
    """Issue public-order private rows whose bytes contain projected public IDs."""
    if type(build_success) is not FormalTrustedWireBuildSuccessV3:
        raise TypeError("projected private-row issuance needs exact wire build success")
    build_success._validate_issued_shape()
    alignment = build_success.private_audit_alignment
    if type(canonical_source_label_bytes_in_source_order) is not tuple or len(
        canonical_source_label_bytes_in_source_order
    ) != alignment.source_case_count or any(
        type(value) is not bytes
        for value in canonical_source_label_bytes_in_source_order
    ):
        raise ValueError("private label bytes do not exactly cover source rows")
    links_by_source = {
        link.source_row_index: link for link in alignment.links_in_public_order
    }
    projected_in_source_order = tuple(
        _project_source_label_with_link_v3(
            link=links_by_source[source_index],
            canonical_source_label_bytes=canonical_source_label_bytes,
        )
        for source_index, canonical_source_label_bytes in enumerate(
            canonical_source_label_bytes_in_source_order
        )
    )
    issued = issue_formal_private_audit_rows_v3(
        build_success=build_success,
        private_payloads_in_source_order=projected_in_source_order,
    )
    return align_private_rows_to_public_v3(
        batch=build_success.public_batch,
        alignment=alignment,
        private_rows_in_source_order=issued,
    )


def replay_formal_trusted_wire_batch_v3(
    envelopes: tuple[bytes, ...],
    *,
    key_source_receipt_id: str,
) -> FormalTrustedWireBatchReplayV3 | FormalTrustedWireBatchRejectionV3:
    if type(envelopes) is not tuple: raise TypeError("formal replay envelopes must use exact tuple")
    count = len(envelopes)
    if not 1 <= count <= MAXIMUM_BATCH_CASES_V3: return _reject_v3("envelope_count_outside_1_through_1024", count)
    try:
        _prefixed_digest(
            key_source_receipt_id,
            "phase2b_key_source_transcript_v3_",
            "public replay key receipt ID",
        )
        decoded = tuple(decode_and_audit_formal_envelope_v3(value) for value in envelopes)
        context = decoded[0].batch_context
        if any(row.batch_context != context for row in decoded) or int(context.case_count_decimal) != count:
            raise ValueError("public replay batch contexts disagree")
        envelope_ids = tuple(row.envelope_id for row in decoded)
        if len(set(envelope_ids)) != count: raise ValueError("public replay repeats an envelope")
        uuid_owner: dict[str, tuple[str, int]] = {}
        for public_row_index, row in enumerate(decoded):
            for channel in row.channel_projection.namespace_uuid_channels:
                previous = uuid_owner.setdefault(
                    channel.public_uuid, (channel.namespace, public_row_index)
                )
                if previous[0] != channel.namespace:
                    raise ValueError("public UUID crosses namespaces")
                if previous[1] != public_row_index:
                    raise ValueError("public UUID is reused across formal envelopes")
        if int(context.id_collision_retry_count_decimal) > len(uuid_owner) * MAXIMUM_UUID_COLLISION_RETRIES_V3:
            raise ValueError("public retry count exceeds allocated UUID retry capacity")
        batch_id = _batch_id_v3(context, key_source_receipt_id, envelope_ids)
    except (AttributeError, KeyError, OverflowError, RecursionError, RFC8785Error, TypeError, ValueError):
        return _reject_v3("bounded_public_batch_replay_failed", count)
    except MemoryError:
        return _reject_v3("bounded_public_batch_replay_failed", count)
    return FormalTrustedWireBatchReplayV3(
        FormalTrustedWireDispositionV3.COMPLETE, context, key_source_receipt_id,
        decoded, envelope_ids,
        batch_id, True,
    )


def build_formal_trusted_wire_batch_v3(
    *,
    source_cases: tuple[FormalSourceCaseV3, ...],
    global_transform_catalog: FormalGlobalTransformCatalogV3,
    transform_catalog_binding_receipt: FormalGlobalTransformCatalogBindingReceiptV3,
    run_id: bytes,
    key_sources: FormalWireKeySourcesV3,
    key_source_receipt: KeySourceTranscriptReceiptV3,
) -> FormalTrustedWireBuildSuccessV3 | FormalTrustedWireBatchRejectionV3:
    if type(source_cases) is not tuple: raise TypeError("formal source cases must use exact tuple")
    count = len(source_cases)
    if not 1 <= count <= MAXIMUM_BATCH_CASES_V3: return _reject_v3("source_case_count_outside_1_through_1024", count)
    if type(run_id) is not bytes or len(run_id) != RUN_ID_BYTES_V3: return _reject_v3("run_id_must_be_exactly_32_bytes", count)
    try:
        verify_formal_global_transform_catalog_binding_v3(
            catalog=global_transform_catalog,
            binding_receipt=transform_catalog_binding_receipt,
        )
        transform_catalog_root = transform_catalog_binding_receipt.transform_catalog_root
        allowed_source_transform_uses = {
            (entry.operation, entry.exact_source_transform_use_sha256)
            for entry in global_transform_catalog.entries
        }
        if any(type(case) is not FormalSourceCaseV3 for case in source_cases): raise TypeError("source case exact type drift")
        # Whole-batch standard_error preflight precedes receipt IDs, source
        # roots, shuffle, UUID allocation, envelopes, and batch roots.
        for case in source_cases:
            case.__post_init__()
            if any(observation.uncertainty_model == "standard_error" for observation in case.graph.observations):
                return _reject_v3("STANDARD_ERROR_UNSUPPORTED", count)
        if type(key_sources) is not FormalWireKeySourcesV3 or type(key_source_receipt) is not KeySourceTranscriptReceiptV3:
            raise TypeError("key inputs exact type drift")
        _verify_key_receipt_v3(key_sources, key_source_receipt)

        private_rows: list[tuple[bytes, int, dict[str, object]]] = []
        source_uuid_owner: dict[tuple[str, str], int] = {}
        source_uuid_namespace: dict[str, str] = {}
        for source_index, case in enumerate(source_cases):
            mapping = _canonicalize_graph_mapping_v3(_profile_value_v3(case.graph), preserve_observation_order=False)
            transforms = {
                transform["transform_id"]: transform
                for transform in mapping["transforms"]  # type: ignore[index,union-attr]
            }
            for aggregate_map in mapping["aggregate_maps"]:  # type: ignore[index,union-attr]
                transform = transforms[aggregate_map["transform_id"]]  # type: ignore[index]
                membership = (
                    transform["operation"],  # type: ignore[index]
                    _source_transform_use_hash_from_mappings_v3(
                        aggregate_map, transform  # type: ignore[arg-type]
                    ),
                )
                if membership not in allowed_source_transform_uses:
                    raise ValueError("exact source aggregate-map/transform use is absent from the bound global catalog")
            _refresh_graph_provenance_v3(mapping, transform_catalog_root)
            _graph_from_mapping_v3(mapping)
            for _, namespace, source_uuid in _walk_uuid_leaves_v3(mapping):
                previous_namespace = source_uuid_namespace.setdefault(
                    source_uuid, namespace
                )
                if previous_namespace != namespace:
                    raise ValueError(
                        "one private source UUID is reused across exact-ten namespaces"
                    )
                previous_owner = source_uuid_owner.setdefault(
                    (namespace, source_uuid), source_index
                )
                if previous_owner != source_index:
                    raise ValueError("source UUID is reused across private source cases")
            private_rows.append((canonicalize_rfc8785_v1(mapping), source_index, mapping))
        if len({row[0] for row in private_rows}) != count: return _reject_v3("duplicate_private_source_case", count)
        original_commitments = tuple(
            source_case_commitment_v3(
                case,
                transform_catalog_root=transform_catalog_root,
            )
            for case in source_cases
        )
        stable = sorted(private_rows, key=lambda row: row[0])
        shuffle_key, id_key, padding_key = _derive_keys_v3(run_id, key_sources)
        shuffled = tuple(stable[index] for index in _shuffle_indices_v3(count, shuffle_key, run_id))

        counters = {namespace: 0 for namespace in NAMESPACE_REGISTRY_V3}
        allocated: set[str] = set(); renamings: dict[tuple[str, str], str] = {}; collision_count = 0
        public_graphs: list[
            tuple[
                int,
                dict[str, object],
                tuple[FormalPrivateUuidMappingV3, ...],
            ]
        ] = []
        for _, source_index, mapping in shuffled:
            unique = sorted({(namespace, value) for _, namespace, value in _walk_uuid_leaves_v3(mapping)})
            for namespace, source_id in unique:
                key = (namespace, source_id)
                if key in renamings: continue
                counter = counters[namespace]; replacement: str | None = None
                for retry in range(MAXIMUM_UUID_COLLISION_RETRIES_V3 + 1):
                    candidate = _uuid_candidate_v3(id_key, run_id, namespace, counter, retry)
                    if candidate not in allocated and candidate not in source_uuid_namespace:
                        replacement = candidate; collision_count += retry; break
                if replacement is None: raise ValueError("UUID collision retry budget exhausted")
                counters[namespace] += 1; allocated.add(replacement); renamings[key] = replacement
            renamed = _rename_graph_ids_v3(mapping, renamings)
            renamed = _canonicalize_graph_mapping_v3(renamed, preserve_observation_order=False)
            _refresh_graph_provenance_v3(renamed, transform_catalog_root)
            _graph_from_mapping_v3(renamed)
            source_to_public = tuple(
                sorted(
                    (
                        FormalPrivateUuidMappingV3(
                            namespace,
                            source_id,
                            renamings[(namespace, source_id)],
                        )
                        for namespace, source_id in unique
                    ),
                    key=lambda row: (row.namespace, row.source_uuid),
                )
            )
            public_graphs.append((source_index, renamed, source_to_public))

        context = FormalBatchContextV3(str(count), transform_catalog_root, str(collision_count), _run_commitment_v3(run_id))
        rows: list[FormalTrustedWireEnvelopeV3] = []
        variant_bases: list[_FormalVariantBaseEnvelopeV3] = []
        source_indices: list[int] = []
        private_uuid_rows_in_public_order: list[
            tuple[FormalPrivateUuidMappingV3, ...]
        ] = []
        for source_index, graph_mapping, source_to_public in public_graphs:
            raw = {
                "batch_context": _context_mapping_v3(context),
                "field_classification_manifest_id": FIELD_CLASSIFICATION_MANIFEST_ID_V3,
                "graph_completeness_policy_id": FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3,
                "public_case": {"graph": graph_mapping, "public_provenance_commitment": "sha256:" + "0" * 64, "schema_version": FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION},
                "schema_version": FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION,
            }
            payload = _normalize_formal_payload_structure_v3(raw)
            payload_bytes = canonicalize_rfc8785_v1(payload)
            if len(payload_bytes) > MAXIMUM_PAYLOAD_BYTES_V3: raise ValueError("formal public graph exceeds envelope payload capacity")
            padding = _padding_v3(padding_key, run_id, hashlib.sha256(payload_bytes).digest(), ENVELOPE_BYTES_V3 - ENVELOPE_HEADER_BYTES_V3 - len(payload_bytes))
            envelope = _frame_payload_with_padding_v3(payload_bytes, padding)
            decoded = decode_and_audit_formal_envelope_v3(envelope)
            rows.append(FormalTrustedWireEnvelopeV3(
                envelope, decoded.envelope_id, decoded.payload_sha256,
                decoded.padding_sha256, decoded.payload_length, decoded.padding_length,
                decoded.public_case, decoded.channel_projection, True, False, False,
            ))
            variant_bases.append(
                _FormalVariantBaseEnvelopeV3(
                    decoded.envelope_id,
                    bytes(decoded.payload_bytes),
                    decoded.padding_bytes,
                )
            )
            source_indices.append(source_index)
            private_uuid_rows_in_public_order.append(source_to_public)
        envelope_ids = tuple(row.envelope_id for row in rows); batch_id = _batch_id_v3(
            context, key_source_receipt.receipt_id, envelope_ids
        )
        envelope_sha = tuple("sha256:" + hashlib.sha256(row.envelope).hexdigest() for row in rows)
        links = tuple(
            FormalPrivateAuditAlignmentLinkV3(
                public_envelope_sha256=envelope_sha[public_index],
                source_row_index=source_index,
                source_case_commitment=original_commitments[source_index],
                source_to_public_uuid_rows=private_uuid_rows_in_public_order[public_index],
                label_binding_id=_label_binding_id_v3(
                    batch_id=batch_id,
                    public_envelope_sha256=envelope_sha[public_index],
                    source_row_index=source_index,
                    source_case_commitment=original_commitments[source_index],
                    source_to_public_uuid_rows=private_uuid_rows_in_public_order[public_index],
                ),
            )
            for public_index, source_index in enumerate(source_indices)
        )
        alignment_body = _alignment_body_v3(
            batch_id=batch_id, run_id_commitment=context.run_id_commitment,
            key_source_receipt_id=key_source_receipt.receipt_id,
            source_case_count=count, links=links,
        )
        alignment = FormalPrivateAuditAlignmentV3(
            batch_id=batch_id,
            run_id_commitment=context.run_id_commitment,
            key_source_receipt_id=key_source_receipt.receipt_id,
            source_case_count=count,
            links_in_public_order=links,
            alignment_id=_alignment_id_v3(alignment_body),
        )
        batch = FormalTrustedWireBatchV3(
            disposition=FormalTrustedWireDispositionV3.COMPLETE,
            schema_version=FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,
            claim_level=FORMAL_TRUSTED_WIRE_V3_CLAIM_LEVEL,
            batch_context=context,
            key_source_receipt_id=key_source_receipt.receipt_id,
            envelopes=tuple(rows),
            envelope_ids=envelope_ids,
            batch_id=batch_id,
        )
        base_envelopes = tuple(variant_bases)
        variant_context_id = _variant_context_id_v3(
            batch_id=batch_id,
            private_alignment_id=alignment.alignment_id,
            transform_catalog_root=transform_catalog_root,
            catalog_binding_receipt_id=transform_catalog_binding_receipt.receipt_id,
            base_envelopes=base_envelopes,
        )
        variant_context = _FormalVariantContextV3(
            batch_id=batch_id,
            private_alignment_id=alignment.alignment_id,
            transform_catalog_root=transform_catalog_root,
            catalog_binding_receipt_id=transform_catalog_binding_receipt.receipt_id,
            global_transform_catalog=global_transform_catalog,
            transform_catalog_binding_receipt=transform_catalog_binding_receipt,
            base_envelopes=base_envelopes,
            context_id=variant_context_id,
            origin_authenticated=False,
        )
        provisional_success_id = _build_success_id_v3(
            public_batch=batch,
            key_source_receipt=key_source_receipt,
            private_alignment=alignment,
            variant_context=variant_context,
        )
        return FormalTrustedWireBuildSuccessV3(
            public_batch=batch,
            key_source_receipt=key_source_receipt,
            private_audit_alignment=alignment,
            variant_context=variant_context,
            success_id=provisional_success_id,
            builder_input_transcript_consistency_checked=True,
            builder_input_catalog_membership_checked=True,
            private_alignment_linkage_checked=True,
            origin_authenticated=False,
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RFC8785Error,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _reject_v3("formal_wire_preflight_or_atomic_build_failed", count)
    except MemoryError:
        return _reject_v3("formal_wire_preflight_or_atomic_build_failed", count)


__all__ = tuple(
    name for name in globals()
    if (
        name.startswith("Formal")
        or name.startswith("Nullable")
        or name.startswith("Exact")
        or name in {
            "ChannelProjectionV3", "ClassifiedPublicLeafV3",
            "DecodedFormalEnvelopeV3", "ENVELOPE_BYTES_V3",
            "ENVELOPE_HEADER_BYTES_V3", "ENVELOPE_MAGIC_V3",
            "ENVELOPE_VERSION_V3", "FIELD_CLASSIFICATION_MANIFEST_ID_V3",
            "DEFAULT_IGNORABLE_CODEPOINT_COUNT_V3",
            "DEFAULT_IGNORABLE_RANGE_TABLE_SHA256_V3",
            "EXACT_TRANSFORM_CORE_SOURCE_SHA256_V3",
            "FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3",
            "FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION",
            "FORMAL_GLOBAL_TRANSFORM_CATALOG_BINDING_V3_SCHEMA_VERSION",
            "FORMAL_GLOBAL_TRANSFORM_CATALOG_V3_SCHEMA_VERSION",
            "FORMAL_PUBLIC_DTO_FIELD_MANIFEST_V3",
            "FORMAL_PUBLIC_STRING_FIELD_POLICY_V3",
            "FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION",
            "FORMAL_OPERATION_SEMANTICS_VERSION_V3",
            "FORMAL_WEIGHTED_MEAN_BASIS_V3",
            "FORMAL_TRUSTED_WIRE_V3_CLAIM_LEVEL",
            "FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION",
            "FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION", "ID_FRAME_DOMAIN_V3",
            "KEY_SOURCE_TRANSCRIPT_V3_SCHEMA_VERSION",
            "MAXIMUM_BATCH_CASES_V3", "MAXIMUM_PAYLOAD_BYTES_V3",
            "MAXIMUM_PRIVATE_UUID_MAPPINGS_PER_CASE_V3",
            "MAXIMUM_UUID_COLLISION_RETRIES_V3", "NAMESPACE_REGISTRY_V3",
            "MissingnessChannelV3", "NamespaceUuidChannelV3",
            "OPERATION_CLAUSE_MANIFEST_V3", "PUBLIC_OBSERVATION_ARRAY_PATHS_V3",
            "PUBLIC_PAYLOAD_MUST_BE_ABSENT_FIELDS_V3", "PublicLeafClassV3",
            "RationalValueV3", "SEMANTIC_CLAUSE_NAMES_V3",
            "SemanticDescriptorChannelV3", "UNICODE_DATA_VERSION_V3",
            "UCD13_DERIVED_CORE_PROPERTIES_SHA256_V3",
            "UCD13_PROP_LIST_SHA256_V3",
            "TRANSFORM_OPERATIONS_V3", "align_private_rows_to_public_v3",
            "build_formal_trusted_wire_batch_v3",
            "classify_public_payload_leaves_v3",
            "decode_and_audit_formal_envelope_v3", "id_frame_v3",
            "formal_source_transform_use_hash_v3",
            "formal_global_transform_catalog_root_v3",
            "issue_formal_global_transform_catalog_v3",
            "issue_formal_private_audit_rows_v3",
            "issue_projected_private_audit_rows_v3",
            "issue_key_source_transcript_receipt_v3",
            "make_formal_variant_payloads_v3",
            "normalize_formal_variant_payload_v3",
            "project_private_source_label_to_public_v3",
            "project_private_source_labels_to_public_order_v3",
            "reencode_formal_variant_batch_v3",
            "reencode_formal_variant_payload_envelope_v3",
            "replay_formal_trusted_wire_batch_v3", "seed_commitment_v3",
            "source_case_commitment_v3", "verify_private_audit_alignment_v3",
            "verify_formal_trusted_wire_build_success_v3",
            "verify_formal_global_transform_catalog_binding_v3",
        }
    )
)

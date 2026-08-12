"""Compact-V2 public recognizer-prediction archive mechanics.

The private builder maps exactly the frozen 960 public recognizer-input rows
through the committed single-row V2 mapping and then emits a bounded,
canonical, publicly replayable archive.  The public decoder verifies only the
bytes, closed schemas, framing, content roots, and ordered row coverage.  It
cannot establish that the supplied input archive was trusted, that a frozen
execution environment actually ran, that the row mapper was used to create the
bytes, or that any prediction is correct.  This module exposes public
structural codec mechanics only; synthetic fixtures do not evidence an actual
run, capacity, scoring, effect, or C1 exit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
import struct
from typing import Final
from uuid import UUID

from .bootstrap import initial_theory
from .hashing import stable_hash
from .phase2b_exact_derived_witness_bridge_v1 import (
    EXACT_DERIVED_BRIDGE_POLICY_ID,
    EXACT_DERIVED_MATCHER_SEMANTICS_ID,
    EXACT_DERIVED_SELECTION_POLICY_ID,
)
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from . import phase2b_recognizer_input_archive_v2 as _input_v2
from .phase2b_recognizer_input_archive_v2 import (
    DecodedRecognizerInputArchiveV2,
    PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
    PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
    PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
    PublicRecognizerRegistryV2,
    FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2,
    RecognizerInputArchiveDispositionV2,
    TrustedRecognizerInputRowV2,
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
)
from . import phase2b_recognizer_prediction_v2 as _prediction_v2
from .phase2b_recognizer_prediction_v2 import (
    PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID,
    PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION,
    RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2,
    PredictionDecisionV2,
    PublicRecognizerPredictionOutcomeV2,
)
from .phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
    frozen_phase2b_protocol,
)
from .phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT
from .phase2b_trusted_wire_batch_v2 import (
    TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
)
from .phase2b_trusted_wire_typed_authority_v2 import (
    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
    COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
)
from .phase2b_trusted_wire_typed_replay_v2 import (
    TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
    TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
)
from .phase2b_trusted_wire_v1 import (
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    MAXIMUM_ASCII_STRING_BYTES,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)
from .phase2b_wire import (
    PREDICTION_SCHEMA_VERSION,
    PredictionBundle,
    PredictionDisposition,
    PredictionReason,
    RoleBinding,
)


PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-prediction-run-context/2"
)
PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-recognizer-prediction-record/2"
)
RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION: Final = (
    "hegel-machine-phase2b-recognizer-prediction-archive/2"
)

PREDICTION_ARCHIVE_MAGIC_V2: Final = b"HGP2PA2\x00"
PREDICTION_ARCHIVE_WIRE_VERSION_V2: Final = 2
_PREDICTION_ARCHIVE_HEADER_V2: Final = struct.Struct(">8sHHII32s")
PREDICTION_ARCHIVE_HEADER_BYTES_V2: Final = _PREDICTION_ARCHIVE_HEADER_V2.size
MAXIMUM_PREDICTION_MANIFEST_BYTES_V2: Final = 16_384
MAXIMUM_PREDICTION_RECORD_BYTES_V2: Final = 32_768
MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2: Final = (
    PREDICTION_ARCHIVE_HEADER_BYTES_V2
    + MAXIMUM_PREDICTION_MANIFEST_BYTES_V2
    + TOTAL_RECOGNIZER_CASE_COUNT * (4 + MAXIMUM_PREDICTION_RECORD_BYTES_V2)
)

_ARCHIVE_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/RECOGNIZER_PREDICTION_ARCHIVE/V2\x00"
)
_RECORD_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/RECOGNIZER_PREDICTION_RECORD/V2\x00"
)
_INPUT_ROWS_DOMAIN_V2: Final = b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00"
_RECORD_IDS_DOMAIN_V2: Final = b"HEGEL/PHASE2B/PREDICTION_RECORD_IDS/V2\x00"
_INPUT_ROW_DOMAIN_V2: Final = b"HEGEL/PHASE2B/RECOGNIZER_INPUT_ROW/V2\x00"

_CONTEXT_ISSUE_TOKEN_V2: Final = object()
_RECORD_ISSUE_TOKEN_V2: Final = object()
_RECORD_CONTEXT_TOKEN_V2: Final = object()
_DECODE_ISSUE_TOKEN_V2: Final = object()
_PARSED_CONTEXT_TOKEN_V2: Final = object()

_FROZEN_THEORY: Final = initial_theory()
_FROZEN_PROTOCOL: Final = frozen_phase2b_protocol()
_FROZEN_EXACT_FREEZE: Final = frozen_phase2b_exact_freeze()

_CONTEXT_FIELDS_V2: Final = (
    "batch_id",
    "batch_policy_id",
    "claim_level",
    "context_id",
    "execution_freeze_manifest_id",
    "expected_prediction_count",
    "input_archive_id",
    "input_archive_policy_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_row_ids_root",
    "protocol_id",
    "schema_version",
)
_CONTEXT_PREIMAGE_FIELDS_V2: Final = tuple(
    name for name in _CONTEXT_FIELDS_V2 if name != "context_id"
)
_RECORD_FIELDS_V2: Final = (
    "bridge_compilation_id",
    "bridge_decision_id",
    "bridge_outcome_id",
    "canonical_family_id",
    "decision",
    "input_authority_content_id",
    "input_envelope_id",
    "input_namespace_audit_id",
    "input_padding_sha256",
    "input_payload_sha256",
    "input_public_registry_id",
    "input_row_id",
    "input_transform_result_id",
    "prediction",
    "prediction_content_id",
    "record_id",
    "run_context_id",
    "schema_version",
)
_RECORD_PREIMAGE_FIELDS_V2: Final = tuple(
    name for name in _RECORD_FIELDS_V2 if name != "record_id"
)
_MANIFEST_FIELDS_V2: Final = (
    "archive_policy_id",
    "archive_schema_version",
    "batch_id",
    "batch_policy_id",
    "claim_level",
    "execution_freeze_manifest_id",
    "expected_prediction_count",
    "input_archive_id",
    "input_archive_policy_id",
    "input_archive_sha256",
    "input_archive_version",
    "input_row_ids_root",
    "prediction_record_ids_root",
    "protocol_id",
    "record_count",
    "run_context_id",
)
_TRUE_DECODED_CLAIMS_V2: Final = (
    "structural_archive_verified",
    "canonical_record_framing_verified",
    "record_schema_verified",
    "row_root_coverage_verified",
)
_FALSE_DECODED_CLAIMS_V2: Final = (
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)
_DECODED_FIELDS_V2: Final = (
    "disposition",
    "archive",
    "archive_id",
    "schema_version",
    "policy_id",
    "context",
    "records",
    "input_row_ids",
    "prediction_record_ids",
    "prediction_content_ids",
    "claim_level",
    *_TRUE_DECODED_CLAIMS_V2,
    *_FALSE_DECODED_CLAIMS_V2,
)
_REJECTION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "input_count",
    "input_archive_id",
    "archive",
    "records",
    "prediction_record_ids",
    "recognizer_capacity_evidence",
)
_PREDICTION_BUNDLE_FIELDS: Final = (
    "admissible_scale_ids",
    "binding",
    "bundle_id",
    "disposition",
    "family_id",
    "freeze_manifest_sha256",
    "input_root_sha256",
    "protocol_sha256",
    "reason",
    "schema_version",
)
_ROLE_BINDING_FIELDS: Final = ("entity_id", "role_id")
_FORBIDDEN_RECOGNIZER_FIELD_TOKENS: Final = (
    "answer",
    "case_index",
    "case_position",
    "challenge",
    "gold",
    "index",
    "main",
    "ordinal",
    "partition",
    "score",
    "metric",
)
_CANONICAL_FAMILY_BY_KIND: Final = dict(_FROZEN_EXACT_FREEZE.family_mapping)
_BRIDGE_FAMILY_BY_KIND: Final = dict(FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2)
_KIND_BY_CANONICAL_FAMILY: Final = {
    family: kind for kind, family in _FROZEN_EXACT_FREEZE.family_mapping
}
_ALLOWED_ABSTAIN_REASONS_V2: Final = frozenset(
    {
        PredictionReason.NO_PASSING_CANDIDATE,
        PredictionReason.MULTIPLE_STRUCTURAL_MATCHES,
        PredictionReason.NONIDENTIFIABLE_SCALE,
        PredictionReason.INSUFFICIENT_EVIDENCE,
        PredictionReason.INSUFFICIENT_MARGIN,
        PredictionReason.INCOMPLETE_CANDIDATE_COVERAGE,
        PredictionReason.VERIFIER_ERROR,
        PredictionReason.RESOURCE_LIMIT,
    }
)
_UUID4_V2: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class PredictionArchiveDispositionV2(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID: Final = stable_hash(
    {
        "schema_version": PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION,
        "fields": _CONTEXT_FIELDS_V2,
        "preimage_fields": _CONTEXT_PREIMAGE_FIELDS_V2,
        "prediction_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "input_row_order": "exact_V2_input_archive_wire_order",
        "input_row_root": (
            "V2_domain||u32_count||repeated(u16_ascii_length||row_id)"
        ),
        "preimage_validation_order": (
            "exact_closed_mapping",
            "exact_string_field_types",
            "exact_builtin_int_prediction_count_equals_960",
            "frozen_identity_comparisons",
            "content_root_hash",
        ),
        "root": {
            "formula": "stable_hash(exact_closed_context_preimage)",
            "prefix": "phase2b_public_prediction_run_context_v2_",
            "validate_before_hash": True,
        },
    },
    prefix="phase2b_public_prediction_run_context_schema_v2_",
)

PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID: Final = stable_hash(
    {
        "schema_version": PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION,
        "fields": _RECORD_FIELDS_V2,
        "preimage_fields": _RECORD_PREIMAGE_FIELDS_V2,
        "prediction_bundle_schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_outcome_schema_version": (
            PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION
        ),
        "prediction_outcome_schema_id": (
            PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID
        ),
        "row_policy_id": RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2,
        "input_roots": (
            "authority_content_id",
            "envelope_id",
            "namespace_audit_id",
            "padding_sha256",
            "payload_sha256",
            "public_registry_id",
            "transform_result_id",
        ),
        "input_row_root_recalculation": (
            "independent_same_module_sha256_of_V2_input_row_domain_and_exact_"
            "canonical_seven_root_mapping"
        ),
        "record_issue_input_row_validation_order": (
            "exact_TrustedRecognizerInputRowV2_type",
            "local_exact_seven_root_prefix_and_two_SHA_closure",
            "independent_input_row_id_recalculation",
            "outcome_to_input_row_comparison",
            "no_input_row_zero_argument_validation_or_replay",
        ),
        "outcome_roots": (
            "bridge_outcome_id",
            "bridge_compilation_id",
            "bridge_decision_id",
            "prediction_content_id",
        ),
        "bridge_outcome_stage_shape": (
            "preflight_v2_requires_ABSTAIN_RESOURCE_LIMIT_and_null_compilation_"
            "decision_roots",
            "derived_run_requires_nonnull_compilation_and_decision_roots",
        ),
        "nested_prediction_bundle_validation": (
            "exact_object_nested_types_and_caps_before_manual_mapping_or_content_hash"
        ),
        "forbidden_public_semantic_tokens": _FORBIDDEN_RECOGNIZER_FIELD_TOKENS,
        "root": {
            "domain_hex": _RECORD_DOMAIN_V2.hex(),
            "formula": "sha256(domain||canonical_closed_record_preimage)",
            "prefix": "phase2b_recognizer_prediction_record_v2_",
            "validate_before_hash": True,
        },
    },
    prefix="phase2b_public_recognizer_prediction_record_schema_v2_",
)

_ARCHIVE_POLICY_VALUE_V2: Final = {
    "archive_version": RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
    "wire": {
        "magic_hex": PREDICTION_ARCHIVE_MAGIC_V2.hex(),
        "wire_version": PREDICTION_ARCHIVE_WIRE_VERSION_V2,
        "header_format": _PREDICTION_ARCHIVE_HEADER_V2.format,
        "header_bytes": PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        "header_fields": (
            "magic",
            "wire_version",
            "header_bytes",
            "manifest_bytes",
            "record_count",
            "body_sha256",
        ),
        "body": "canonical_manifest||960*(u32be_length||canonical_record)",
    },
    "caps": {
        "archive_bytes": MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
        "manifest_bytes": MAXIMUM_PREDICTION_MANIFEST_BYTES_V2,
        "record_bytes": MAXIMUM_PREDICTION_RECORD_BYTES_V2,
        "prediction_count": TOTAL_RECOGNIZER_CASE_COUNT,
    },
    "field_manifests": {
        "context": _CONTEXT_FIELDS_V2,
        "context_preimage": _CONTEXT_PREIMAGE_FIELDS_V2,
        "record": _RECORD_FIELDS_V2,
        "record_preimage": _RECORD_PREIMAGE_FIELDS_V2,
        "manifest": _MANIFEST_FIELDS_V2,
        "decoded": _DECODED_FIELDS_V2,
        "rejection": _REJECTION_FIELDS_V2,
    },
    "manifest_validation_order": (
        "exact_closed_mapping",
        "exact_string_field_types",
        "exact_builtin_int_expected_and_record_counts_equal_960",
        "frozen_identity_comparisons",
        "content_root_validation",
    ),
    "forbidden_public_semantic_tokens": _FORBIDDEN_RECOGNIZER_FIELD_TOKENS,
    "schema_ids": {
        "context": PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID,
        "record": PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID,
    },
    "dependency_bindings": {
        "prediction_row_policy_transitively_binds": (
            "registry_alias_typed_replay_compact_codec_batch_prediction_bundle_"
            "freeze_theory_and_exact_derived_bridge"
        ),
        "protocol_id": _FROZEN_PROTOCOL.protocol_id,
        "frozen_theory_version_id": _FROZEN_THEORY.version_id,
        "exact_freeze_id": _FROZEN_EXACT_FREEZE.freeze_id,
        "exact_freeze_family_mapping": tuple(
            (kind.value, family.value)
            for kind, family in _FROZEN_EXACT_FREEZE.family_mapping
        ),
        "input_archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "input_archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "public_registry_version": PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
        "public_registry_schema_id": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
        "public_family_alias_policy_id": PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
        "typed_replay_version": TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
        "typed_replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "compact_typed_authority_schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "compact_typed_authority_codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "compact_typed_authority_codec_policy_id": COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "batch_schema_version": TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
        "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "prediction_bundle_schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_outcome_schema_id": (
            PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID
        ),
        "prediction_row_policy_id": RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2,
        "derived_bridge_policy_id": EXACT_DERIVED_BRIDGE_POLICY_ID,
        "derived_matcher_semantics_id": EXACT_DERIVED_MATCHER_SEMANTICS_ID,
        "derived_selection_policy_id": EXACT_DERIVED_SELECTION_POLICY_ID,
    },
    "canonical_json": {
        "accepted_profile_id": JCS_PROFILE_ID,
        "field_manifest_id": FIELD_MANIFEST_ID,
        "exact_decode_reencode": True,
    },
    "private_builder": {
        "exact_input_archive_type": "DecodedRecognizerInputArchiveV2",
        "exact_execution_freeze_type": "ExecutionFreezeManifest",
        "execution_freeze_validation_order": (
            "exact_manifest_type",
            "local_protocol_exact_freeze_theory_prefixed_SHA256_closure",
            "local_git_image_four_implementation_SHA_and_isolation_closure",
            "local_exact_baseline_tuple_count_items_enums_strings_SHA_and_bool_closure",
            "baseline_registration_post_init",
            "execution_manifest_post_init",
            "current_frozen_identity_comparisons",
            "manifest_content_root",
        ),
        "shallow_exact_960_before_public_input_archive_decode_or_sha256": True,
        "public_input_archive_decode_API_operation_count": 1,
        "upstream_internal_structural_parse_count": (
            "bounded_performance_property_not_a_public_claim"
        ),
        "preserve_input_wire_order": True,
        "single_row_mapper_call_count_per_row": 1,
        "atomic_rejection_without_partial_records": True,
    },
    "public_decoder": {
        "public_bytes_only": True,
        "decoded_issue_reparses_archive_and_requires_supplied_parsed_exact_parity": True,
        "cheap_claim_and_stored_column_closure_before_deep_record_validation": True,
        "structural_true": _TRUE_DECODED_CLAIMS_V2,
        "non_evidence_false": _FALSE_DECODED_CLAIMS_V2,
        "exact_builtin_bool": True,
        "prediction_content_uniqueness_required": False,
    },
    "roots": {
        "archive_domain_hex": _ARCHIVE_DOMAIN_V2.hex(),
        "record_domain_hex": _RECORD_DOMAIN_V2.hex(),
        "input_rows_domain_hex": _INPUT_ROWS_DOMAIN_V2.hex(),
        "record_ids_domain_hex": _RECORD_IDS_DOMAIN_V2.hex(),
        "input_and_record_sequence_order": "preserved_input_wire_order",
        "validate_before_hash": True,
    },
    "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
    "scope": (
        "public_structural_codec_mechanics_only",
        "synthetic_fixture_not_actual_run",
        "no_runtime_capacity_scoring_effect_or_c1_claim",
    ),
    "cross_version_rejection": (
        "V1_prediction_archive_magic",
        "V1_input_archive_type",
        "V1_prediction_record_type",
    ),
}
RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2: Final = stable_hash(
    _ARCHIVE_POLICY_VALUE_V2,
    prefix="phase2b_recognizer_prediction_archive_policy_v2_",
)


def _ascii_v2(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAXIMUM_ASCII_STRING_BYTES
        or not value.isascii()
    ):
        raise ValueError(f"{name} must use exact bounded nonempty ASCII")
    return value


def _hex64_v2(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ValueError(f"{name} must use exact lowercase SHA-256")
    return value


def _digest_v2(value: object, prefix: str, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != len(prefix) + 64
        or not value.startswith(prefix)
        or any(item not in "0123456789abcdef" for item in value[len(prefix) :])
    ):
        raise ValueError(f"{name} must use an exact prefixed SHA-256")
    return value


def _uuid4_v2(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 36
        or _UUID4_V2.fullmatch(value) is None
        or UUID(value).version != 4
    ):
        raise ValueError(f"{name} must use canonical lowercase UUIDv4")
    return value


def _closed_v2(
    value: object,
    manifest: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must use an exact mapping")
    if (
        len(value) != len(manifest)
        or any(type(key) is not str for key in value)
        or set(value) != set(manifest)
    ):
        raise ValueError(f"{name} closed schema drift")
    return value


def _reject_forbidden_public_semantics_v2(value: object) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            for key, item in current.items():
                if type(key) is not str:
                    raise TypeError("V2 public prediction wire key requires exact text")
                folded = key.casefold()
                if any(token in folded for token in _FORBIDDEN_RECOGNIZER_FIELD_TOKENS):
                    raise ValueError("V2 public prediction wire contains forbidden field")
                stack.append(item)
        elif type(current) in (list, tuple):
            stack.extend(current)
        elif type(current) is str:
            folded = current.casefold()
            if any(token in folded for token in _FORBIDDEN_RECOGNIZER_FIELD_TOKENS):
                raise ValueError("V2 public prediction wire contains forbidden value")


def _sequence_root_v2(
    values: tuple[str, ...],
    *,
    item_prefix: str,
    domain: bytes,
    output_prefix: str,
    name: str,
) -> str:
    if type(values) is not tuple or len(values) != TOTAL_RECOGNIZER_CASE_COUNT:
        raise TypeError(f"{name} sequence must contain exact frozen count")
    encoded_values: list[bytes] = []
    for value in values:
        _digest_v2(value, item_prefix, name)
        encoded = value.encode("ascii")
        if len(encoded) > 65_535:
            raise ValueError(f"{name} encoded length drift")
        encoded_values.append(encoded)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(TOTAL_RECOGNIZER_CASE_COUNT.to_bytes(4, "big"))
    for encoded in encoded_values:
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return output_prefix + digest.hexdigest()


def _input_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _sequence_root_v2(
        values,
        item_prefix="phase2b_recognizer_input_row_v2_",
        domain=_INPUT_ROWS_DOMAIN_V2,
        output_prefix="phase2b_prediction_input_rows_v2_",
        name="V2 prediction input row ID",
    )


def _prediction_record_ids_root_v2(values: tuple[str, ...]) -> str:
    return _sequence_root_v2(
        values,
        item_prefix="phase2b_recognizer_prediction_record_v2_",
        domain=_RECORD_IDS_DOMAIN_V2,
        output_prefix="phase2b_prediction_records_v2_",
        name="V2 prediction record ID",
    )


def _context_mapping_without_id_v2(
    value: "PublicPredictionRunContextV2",
) -> dict[str, object]:
    return {
        "batch_id": value.batch_id,
        "batch_policy_id": value.batch_policy_id,
        "claim_level": value.claim_level,
        "execution_freeze_manifest_id": value.execution_freeze_manifest_id,
        "expected_prediction_count": value.expected_prediction_count,
        "input_archive_id": value.input_archive_id,
        "input_archive_policy_id": value.input_archive_policy_id,
        "input_archive_sha256": value.input_archive_sha256,
        "input_archive_version": value.input_archive_version,
        "input_row_ids_root": value.input_row_ids_root,
        "protocol_id": value.protocol_id,
        "schema_version": value.schema_version,
    }


def _context_mapping_v2(
    value: "PublicPredictionRunContextV2",
) -> dict[str, object]:
    mapping = _context_mapping_without_id_v2(value)
    mapping["context_id"] = value.context_id
    return mapping


def _validate_context_preimage_v2(value: object) -> dict[str, object]:
    mapping = _closed_v2(
        value,
        _CONTEXT_PREIMAGE_FIELDS_V2,
        "V2 public prediction context preimage",
    )
    string_fields = tuple(
        name for name in _CONTEXT_PREIMAGE_FIELDS_V2
        if name != "expected_prediction_count"
    )
    if any(type(mapping[name]) is not str for name in string_fields):
        raise TypeError("V2 context identity fields require exact strings")
    if (
        type(mapping["expected_prediction_count"]) is not int
        or mapping["expected_prediction_count"] != TOTAL_RECOGNIZER_CASE_COUNT
    ):
        raise ValueError("V2 context prediction count drift")
    if (
        mapping["schema_version"]
        != PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION
        or mapping["protocol_id"] != _FROZEN_PROTOCOL.protocol_id
        or mapping["input_archive_version"]
        != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or mapping["input_archive_policy_id"]
        != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or mapping["batch_policy_id"] != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or mapping["claim_level"] != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("V2 context policy identity drift")
    _ascii_v2(mapping["schema_version"], "V2 context schema")
    _digest_v2(mapping["protocol_id"], "phase2b_protocol_", "V2 context protocol")
    _digest_v2(
        mapping["execution_freeze_manifest_id"],
        "phase2b_execution_freeze_",
        "V2 context execution freeze",
    )
    _digest_v2(
        mapping["input_archive_id"],
        "phase2b_recognizer_input_archive_v2_",
        "V2 context input archive ID",
    )
    _digest_v2(
        mapping["input_archive_policy_id"],
        "phase2b_recognizer_input_archive_policy_v2_",
        "V2 context input archive policy",
    )
    _hex64_v2(mapping["input_archive_sha256"], "V2 context input archive SHA-256")
    _digest_v2(
        mapping["input_row_ids_root"],
        "phase2b_prediction_input_rows_v2_",
        "V2 context input row root",
    )
    _digest_v2(mapping["batch_id"], "phase2b_trusted_wire_batch_v2_", "V2 context batch ID")
    _digest_v2(
        mapping["batch_policy_id"],
        "phase2b_trusted_wire_batch_v2_policy_",
        "V2 context batch policy",
    )
    return mapping


def _context_id_v2(mapping_without_id: dict[str, object]) -> str:
    validated = _validate_context_preimage_v2(mapping_without_id)
    return stable_hash(
        validated,
        prefix="phase2b_public_prediction_run_context_v2_",
    )


@dataclass(frozen=True, slots=True, init=False)
class PublicPredictionRunContextV2:
    batch_id: str
    batch_policy_id: str
    claim_level: str
    context_id: str
    execution_freeze_manifest_id: str
    expected_prediction_count: int
    input_archive_id: str
    input_archive_policy_id: str
    input_archive_sha256: str
    input_archive_version: str
    input_row_ids_root: str
    protocol_id: str
    schema_version: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 public prediction contexts are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        batch_id: str,
        input_archive_id: str,
        input_archive_sha256: str,
        input_row_ids_root: str,
        execution_freeze_manifest_id: str,
    ) -> "PublicPredictionRunContextV2":
        if token is not _CONTEXT_ISSUE_TOKEN_V2:
            raise TypeError("V2 context issuer token drift")
        value = object.__new__(cls)
        frozen = (
            ("batch_id", batch_id),
            ("batch_policy_id", TRUSTED_WIRE_BATCH_V2_POLICY_ID),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("execution_freeze_manifest_id", execution_freeze_manifest_id),
            ("expected_prediction_count", TOTAL_RECOGNIZER_CASE_COUNT),
            ("input_archive_id", input_archive_id),
            ("input_archive_policy_id", RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2),
            ("input_archive_sha256", input_archive_sha256),
            ("input_archive_version", TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION),
            ("input_row_ids_root", input_row_ids_root),
            ("protocol_id", _FROZEN_PROTOCOL.protocol_id),
            ("schema_version", PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        object.__setattr__(
            value,
            "context_id",
            _context_id_v2(_context_mapping_without_id_v2(value)),
        )
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicPredictionRunContextV2:
            raise TypeError("V2 context exact type drift")
        _validate_context_preimage_v2(_context_mapping_without_id_v2(self))
        _digest_v2(
            self.context_id,
            "phase2b_public_prediction_run_context_v2_",
            "V2 context ID",
        )
        if self.context_id != _context_id_v2(_context_mapping_without_id_v2(self)):
            raise ValueError("V2 context root drift")


def _decode_context_v2(mapping: object) -> PublicPredictionRunContextV2:
    value = _closed_v2(mapping, _CONTEXT_FIELDS_V2, "V2 public prediction context")
    context = PublicPredictionRunContextV2._issue(
        _CONTEXT_ISSUE_TOKEN_V2,
        batch_id=value["batch_id"],  # type: ignore[arg-type]
        input_archive_id=value["input_archive_id"],  # type: ignore[arg-type]
        input_archive_sha256=value["input_archive_sha256"],  # type: ignore[arg-type]
        input_row_ids_root=value["input_row_ids_root"],  # type: ignore[arg-type]
        execution_freeze_manifest_id=value["execution_freeze_manifest_id"],  # type: ignore[arg-type]
    )
    if _context_mapping_v2(context) != value:
        raise ValueError("V2 context canonical roundtrip drift")
    return context


def _validate_execution_freeze_manifest_v2(
    manifest: ExecutionFreezeManifest,
) -> None:
    """Validate exact current freeze identities before input archive replay."""

    if type(manifest) is not ExecutionFreezeManifest:
        raise TypeError("V2 prediction archive requires exact freeze manifest")
    _digest_v2(
        manifest.protocol_id,
        "phase2b_protocol_",
        "V2 execution manifest protocol ID",
    )
    _digest_v2(
        manifest.exact_freeze_id,
        "phase2b_exact_freeze_",
        "V2 execution manifest exact freeze ID",
    )
    _digest_v2(
        manifest.theory_version_id,
        "theory_",
        "V2 execution manifest theory version ID",
    )
    if (
        type(manifest.git_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", manifest.git_commit) is None
    ):
        raise ValueError("V2 execution manifest Git commit format drift")
    if (
        type(manifest.recognizer_image_digest) is not str
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            manifest.recognizer_image_digest,
        ) is None
    ):
        raise ValueError("V2 execution manifest recognizer image digest drift")
    for name in (
        "configuration_sha256",
        "adapter_implementation_sha256",
        "selector_implementation_sha256",
        "verifier_registry_sha256",
    ):
        _hex64_v2(getattr(manifest, name), f"V2 execution manifest {name}")
    _ascii_v2(
        manifest.isolation_profile_id,
        "V2 execution manifest isolation profile ID",
    )
    if (
        type(manifest.baseline_registrations) is not tuple
        or len(manifest.baseline_registrations) != len(BaselineKind)
    ):
        raise TypeError("V2 execution manifest baseline tuple drift")
    for registration in manifest.baseline_registrations:
        if type(registration) is not BaselineRegistration:
            raise TypeError("V2 execution baseline exact type drift")
        if type(registration.kind) is not BaselineKind:
            raise TypeError("V2 execution baseline kind enum drift")
        _ascii_v2(registration.baseline_spec_id, "V2 baseline specification ID")
        _ascii_v2(registration.implementation_id, "V2 baseline implementation ID")
        _hex64_v2(registration.artifact_sha256, "V2 baseline artifact SHA-256")
        if type(registration.frozen_before_holdout_generation) is not bool:
            raise TypeError("V2 baseline frozen flag exact bool drift")
    if {item.kind for item in manifest.baseline_registrations} != set(BaselineKind):
        raise ValueError("V2 execution baseline kind coverage drift")
    for registration in manifest.baseline_registrations:
        registration.__post_init__()
    manifest.__post_init__()
    if (
        manifest.protocol_id != _FROZEN_PROTOCOL.protocol_id
        or manifest.exact_freeze_id != _FROZEN_EXACT_FREEZE.freeze_id
        or manifest.theory_version_id != _FROZEN_THEORY.version_id
    ):
        raise ValueError("V2 execution manifest current authority drift")
    _digest_v2(
        manifest.manifest_id,
        "phase2b_execution_freeze_",
        "V2 execution freeze manifest ID",
    )


class _ShallowPredictionInputError(ValueError):
    def __init__(self, reason: str, input_count: int, input_archive_id: str | None):
        super().__init__(reason)
        self.reason = reason
        self.input_count = input_count
        self.input_archive_id = input_archive_id


def _shallow_prediction_input_v2(
    input_archive: DecodedRecognizerInputArchiveV2,
) -> tuple[tuple[TrustedRecognizerInputRowV2, ...], tuple[str, ...]]:
    """Close exact 960 stored roots before decode or archive SHA-256."""

    if type(input_archive) is not DecodedRecognizerInputArchiveV2:
        raise TypeError("V2 prediction builder requires exact input archive type")
    archive_id: str | None = None
    if type(input_archive.archive_id) is str:
        try:
            archive_id = _digest_v2(
                input_archive.archive_id,
                "phase2b_recognizer_input_archive_v2_",
                "V2 shallow input archive ID",
            )
        except ValueError:
            archive_id = None
    rows = input_archive.rows
    count = len(rows) if type(rows) is tuple else 0
    if type(rows) is not tuple or count != TOTAL_RECOGNIZER_CASE_COUNT:
        raise _ShallowPredictionInputError(
            "input_row_count_not_exact_960",
            count,
            archive_id,
        )
    if type(input_archive.archive) is not bytes or not (
        _input_v2.ARCHIVE_HEADER_BYTES_V2
        <= len(input_archive.archive)
        <= _input_v2.MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2
    ):
        raise _ShallowPredictionInputError(
            "input_archive_byte_shape_invalid",
            count,
            archive_id,
        )
    if (
        input_archive.disposition is not RecognizerInputArchiveDispositionV2.COMPLETE
        or type(input_archive.row_ids) is not tuple
        or len(input_archive.row_ids) != count
    ):
        raise _ShallowPredictionInputError(
            "input_archive_stored_row_column_invalid",
            count,
            archive_id,
        )
    stored_columns = (
        input_archive.envelope_ids,
        input_archive.public_registry_ids,
        input_archive.authority_content_ids,
        input_archive.transform_result_ids,
    )
    if any(type(column) is not tuple or len(column) != count for column in stored_columns):
        raise _ShallowPredictionInputError(
            "input_archive_stored_root_column_invalid",
            count,
            archive_id,
        )
    stored_specs = (
        (input_archive.row_ids, "phase2b_recognizer_input_row_v2_"),
        (input_archive.envelope_ids, "phase2b_trusted_envelope_v2_"),
        (
            input_archive.public_registry_ids,
            "phase2b_public_recognizer_registry_v2_",
        ),
        (input_archive.authority_content_ids, "phase2b_public_transform_evidence_"),
        (input_archive.transform_result_ids, "phase2b_exact_transform_result_"),
    )
    for column, prefix in stored_specs:
        for item in column:
            _digest_v2(item, prefix, "V2 shallow stored root")
    for index, row in enumerate(rows):
        if type(row) is not TrustedRecognizerInputRowV2:
            raise TypeError("V2 shallow input row exact type drift")
        if type(row.envelope) is not bytes or len(row.envelope) != _input_v2.ENVELOPE_BYTES:
            raise TypeError("V2 shallow input envelope shape drift")
        _digest_v2(row.row_id, "phase2b_recognizer_input_row_v2_", "V2 shallow row ID")
        _digest_v2(row.envelope_id, "phase2b_trusted_envelope_v2_", "V2 shallow envelope ID")
        _digest_v2(row.namespace_audit_id, "phase2b_namespace_audit_v2_", "V2 shallow namespace ID")
        _digest_v2(row.authority_content_id, "phase2b_public_transform_evidence_", "V2 shallow authority ID")
        _digest_v2(row.transform_result_id, "phase2b_exact_transform_result_", "V2 shallow transform ID")
        _digest_v2(row.public_registry_id, "phase2b_public_recognizer_registry_v2_", "V2 shallow registry ID")
        _hex64_v2(row.payload_sha256, "V2 shallow payload SHA-256")
        _hex64_v2(row.padding_sha256, "V2 shallow padding SHA-256")
        if type(row.public_registry) is not PublicRecognizerRegistryV2:
            raise TypeError("V2 shallow public registry exact type drift")
        expected = (
            input_archive.row_ids[index],
            input_archive.envelope_ids[index],
            input_archive.public_registry_ids[index],
            input_archive.authority_content_ids[index],
            input_archive.transform_result_ids[index],
        )
        actual = (
            row.row_id,
            row.envelope_id,
            row.public_registry_id,
            row.authority_content_id,
            row.transform_result_id,
        )
        if actual != expected:
            raise _ShallowPredictionInputError(
                "input_archive_stored_root_parity_drift",
                count,
                archive_id,
            )
    row_ids = tuple(row.row_id for row in rows)
    if len(set(row_ids)) != count:
        raise _ShallowPredictionInputError(
            "input_archive_repeats_row_id",
            count,
            archive_id,
        )
    return rows, row_ids


def _prediction_mapping_unchecked_v2(value: PredictionBundle) -> dict[str, object]:
    _validate_prediction_bundle_object_v2(value)
    # Manual mapping is intentional: no object method, enum ``.value`` access,
    # nested iteration, list allocation, or content hash occurs until the
    # object-side exact-type and resource-cap closure above has succeeded.
    return {
        "admissible_scale_ids": list(value.admissible_scale_ids),
        "binding": [
            {"entity_id": item.entity_id, "role_id": item.role_id}
            for item in value.binding
        ],
        "bundle_id": value.bundle_id,
        "disposition": value.disposition.value,
        "family_id": value.family_id,
        "freeze_manifest_sha256": value.freeze_manifest_sha256,
        "input_root_sha256": value.input_root_sha256,
        "protocol_sha256": value.protocol_sha256,
        "reason": value.reason.value,
        "schema_version": value.schema_version,
    }


def _validate_prediction_bundle_object_v2(value: PredictionBundle) -> None:
    """Close exact nested fields and caps before mapping or content hashing."""

    if type(value) is not PredictionBundle:
        raise TypeError("V2 archive prediction requires exact PredictionBundle")
    if (
        type(value.schema_version) is not str
        or value.schema_version != PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("V2 archive prediction schema drift")
    _uuid4_v2(value.bundle_id, "V2 archive prediction bundle ID")
    for name in ("input_root_sha256", "protocol_sha256", "freeze_manifest_sha256"):
        _hex64_v2(getattr(value, name), f"V2 archive prediction {name}")
    if type(value.disposition) is not PredictionDisposition:
        raise TypeError("V2 archive prediction disposition exact enum drift")
    if type(value.reason) is not PredictionReason:
        raise TypeError("V2 archive prediction reason exact enum drift")
    if value.family_id is not None:
        _uuid4_v2(value.family_id, "V2 archive prediction family ID")
    if type(value.binding) is not tuple or len(value.binding) > 64:
        raise TypeError("V2 archive prediction binding exact bounded tuple drift")
    role_ids: list[str] = []
    entity_ids: list[str] = []
    for item in value.binding:
        if type(item) is not RoleBinding:
            raise TypeError("V2 archive prediction binding exact item drift")
        role_ids.append(_uuid4_v2(item.role_id, "V2 archive prediction role ID"))
        entity_ids.append(_uuid4_v2(item.entity_id, "V2 archive prediction entity ID"))
    if (
        value.binding != tuple(sorted(value.binding, key=lambda item: item.role_id))
        or len(set(role_ids)) != len(role_ids)
        or len(set(entity_ids)) != len(entity_ids)
    ):
        raise ValueError("V2 archive prediction binding canonicality drift")
    if (
        type(value.admissible_scale_ids) is not tuple
        or len(value.admissible_scale_ids) > 4_096
    ):
        raise TypeError("V2 archive prediction scales exact bounded tuple drift")
    for scale_id in value.admissible_scale_ids:
        _uuid4_v2(scale_id, "V2 archive prediction scale ID")
    if value.admissible_scale_ids != tuple(sorted(set(value.admissible_scale_ids))):
        raise ValueError("V2 archive prediction scale canonicality drift")
    if value.disposition is PredictionDisposition.UNIQUE_MATCH:
        if (
            value.family_id is None
            or not value.binding
            or not value.admissible_scale_ids
            or value.reason is not PredictionReason.UNIQUE_STRUCTURAL_MATCH
        ):
            raise ValueError("V2 archive positive prediction shape drift")
    elif (
        value.family_id is not None
        or value.binding
        or value.admissible_scale_ids
        or value.reason is PredictionReason.UNIQUE_STRUCTURAL_MATCH
    ):
        raise ValueError("V2 archive abstention prediction shape drift")


def _decode_prediction_bundle_v2(mapping: object) -> PredictionBundle:
    root = _closed_v2(mapping, _PREDICTION_BUNDLE_FIELDS, "V2 prediction bundle")
    if root["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError("V2 prediction bundle schema drift")
    if type(root["schema_version"]) is not str:
        raise TypeError("V2 prediction schema requires exact text")
    _uuid4_v2(root["bundle_id"], "V2 prediction bundle ID")
    for name in ("input_root_sha256", "protocol_sha256", "freeze_manifest_sha256"):
        _hex64_v2(root[name], f"V2 prediction {name}")
    if (
        type(root["disposition"]) is not str
        or root["disposition"] not in {item.value for item in PredictionDisposition}
    ):
        raise ValueError("V2 prediction disposition drift")
    if (
        type(root["reason"]) is not str
        or root["reason"] not in {item.value for item in PredictionReason}
    ):
        raise ValueError("V2 prediction reason drift")
    if root["family_id"] is not None:
        _uuid4_v2(root["family_id"], "V2 prediction family ID")
    raw_bindings = root["binding"]
    if type(raw_bindings) is not list or len(raw_bindings) > 64:
        raise TypeError("V2 prediction binding wire drift")
    for item in raw_bindings:
        binding = _closed_v2(item, _ROLE_BINDING_FIELDS, "V2 prediction role binding")
        _uuid4_v2(binding["role_id"], "V2 prediction role ID")
        _uuid4_v2(binding["entity_id"], "V2 prediction entity ID")
    raw_scales = root["admissible_scale_ids"]
    if type(raw_scales) is not list or len(raw_scales) > 4_096:
        raise TypeError("V2 prediction scale wire drift")
    for scale in raw_scales:
        _uuid4_v2(scale, "V2 prediction scale ID")
    prediction = PredictionBundle.from_mapping(root)
    if type(prediction) is not PredictionBundle:
        raise TypeError("V2 prediction decoder returned nonexact bundle")
    if _prediction_mapping_unchecked_v2(prediction) != root:
        raise ValueError("V2 prediction bundle canonical roundtrip drift")
    return prediction


def _prediction_mapping_v2(value: PredictionBundle) -> dict[str, object]:
    mapping = _prediction_mapping_unchecked_v2(value)
    if _decode_prediction_bundle_v2(mapping) != value:
        raise ValueError("V2 prediction bundle structural pollution")
    return mapping


def _bridge_outcome_id_v2(value: object) -> str:
    if type(value) is not str:
        raise TypeError("V2 bridge outcome ID requires exact text")
    for prefix in (
        "phase2b_exact_derived_run_",
        "phase2b_exact_derived_preflight_v2_",
    ):
        if value.startswith(prefix):
            return _digest_v2(value, prefix, "V2 bridge outcome ID")
    raise ValueError("V2 bridge outcome ID prefix drift")


def _input_row_id_from_record_roots_v2(mapping: dict[str, object]) -> str:
    root_mapping = {
        "authority_content_id": mapping["input_authority_content_id"],
        "envelope_id": mapping["input_envelope_id"],
        "namespace_audit_id": mapping["input_namespace_audit_id"],
        "padding_sha256": mapping["input_padding_sha256"],
        "payload_sha256": mapping["input_payload_sha256"],
        "public_registry_id": mapping["input_public_registry_id"],
        "transform_result_id": mapping["input_transform_result_id"],
    }
    payload = encode_phase2b_jcs_profile_v1(root_mapping)
    return "phase2b_recognizer_input_row_v2_" + hashlib.sha256(
        _INPUT_ROW_DOMAIN_V2 + payload
    ).hexdigest()


def _validate_record_issue_input_row_v2(
    input_row: TrustedRecognizerInputRowV2,
) -> None:
    """Close stored public row roots locally without replay or row validation."""

    if type(input_row) is not TrustedRecognizerInputRowV2:
        raise TypeError("V2 prediction record input row exact type drift")
    _digest_v2(
        input_row.row_id,
        "phase2b_recognizer_input_row_v2_",
        "V2 prediction record input row ID",
    )
    _digest_v2(
        input_row.envelope_id,
        "phase2b_trusted_envelope_v2_",
        "V2 prediction record input envelope ID",
    )
    _digest_v2(
        input_row.namespace_audit_id,
        "phase2b_namespace_audit_v2_",
        "V2 prediction record input namespace ID",
    )
    _digest_v2(
        input_row.authority_content_id,
        "phase2b_public_transform_evidence_",
        "V2 prediction record input authority ID",
    )
    _digest_v2(
        input_row.transform_result_id,
        "phase2b_exact_transform_result_",
        "V2 prediction record input transform ID",
    )
    _digest_v2(
        input_row.public_registry_id,
        "phase2b_public_recognizer_registry_v2_",
        "V2 prediction record input registry ID",
    )
    _hex64_v2(
        input_row.payload_sha256,
        "V2 prediction record input payload SHA-256",
    )
    _hex64_v2(
        input_row.padding_sha256,
        "V2 prediction record input padding SHA-256",
    )
    roots: dict[str, object] = {
        "input_authority_content_id": input_row.authority_content_id,
        "input_envelope_id": input_row.envelope_id,
        "input_namespace_audit_id": input_row.namespace_audit_id,
        "input_padding_sha256": input_row.padding_sha256,
        "input_payload_sha256": input_row.payload_sha256,
        "input_public_registry_id": input_row.public_registry_id,
        "input_row_id": input_row.row_id,
        "input_transform_result_id": input_row.transform_result_id,
    }
    if input_row.row_id != _input_row_id_from_record_roots_v2(roots):
        raise ValueError("V2 prediction record input row root drift")


def _validate_decision_mapping_v2(
    *,
    decision: PredictionDecisionV2,
    canonical_family_id: CanonicalFamilyId | None,
    prediction: PredictionBundle,
) -> None:
    if type(decision) is not PredictionDecisionV2:
        raise TypeError("V2 record decision exact enum drift")
    if type(prediction) is not PredictionBundle:
        raise TypeError("V2 record prediction exact type drift")
    if decision is PredictionDecisionV2.ABSTAIN:
        if (
            canonical_family_id is not None
            or prediction.disposition is not PredictionDisposition.ABSTAIN
            or type(prediction.reason) is not PredictionReason
            or prediction.reason not in _ALLOWED_ABSTAIN_REASONS_V2
            or prediction.family_id is not None
            or prediction.binding != ()
            or prediction.admissible_scale_ids != ()
        ):
            raise ValueError("V2 abstention carries positive or unknown fields")
        return
    if type(canonical_family_id) is not CanonicalFamilyId:
        raise TypeError("V2 positive record requires exact canonical family")
    law_kind = _KIND_BY_CANONICAL_FAMILY.get(canonical_family_id)
    if law_kind is None or prediction.family_id != _BRIDGE_FAMILY_BY_KIND[law_kind]:
        raise ValueError("V2 canonical and bridge family mapping drift")
    if (
        prediction.disposition is not PredictionDisposition.UNIQUE_MATCH
        or prediction.reason is not PredictionReason.UNIQUE_STRUCTURAL_MATCH
        or type(prediction.binding) is not tuple
        or not prediction.binding
        or any(type(item) is not RoleBinding for item in prediction.binding)
    ):
        raise ValueError("V2 positive prediction disposition or binding drift")
    scale_count = len(prediction.admissible_scale_ids)
    if decision is PredictionDecisionV2.ANSWER and scale_count != 1:
        raise ValueError("V2 ANSWER requires exactly one admissible scale")
    if decision is PredictionDecisionV2.ANSWER_SET and scale_count <= 1:
        raise ValueError("V2 ANSWER_SET requires multiple admissible scales")


def _record_mapping_without_id_v2(
    value: "PublicRecognizerPredictionRecordV2",
) -> dict[str, object]:
    if type(value) is not PublicRecognizerPredictionRecordV2:
        raise TypeError("V2 prediction record mapping exact type drift")
    if type(value.decision) is not PredictionDecisionV2:
        raise TypeError("V2 prediction record mapping decision exact enum drift")
    if (
        value.canonical_family_id is not None
        and type(value.canonical_family_id) is not CanonicalFamilyId
    ):
        raise TypeError("V2 prediction record mapping canonical family type drift")
    _validate_prediction_bundle_object_v2(value.prediction)
    return {
        "bridge_compilation_id": value.bridge_compilation_id,
        "bridge_decision_id": value.bridge_decision_id,
        "bridge_outcome_id": value.bridge_outcome_id,
        "canonical_family_id": (
            None if value.canonical_family_id is None else value.canonical_family_id.value
        ),
        "decision": value.decision.value,
        "input_authority_content_id": value.input_authority_content_id,
        "input_envelope_id": value.input_envelope_id,
        "input_namespace_audit_id": value.input_namespace_audit_id,
        "input_padding_sha256": value.input_padding_sha256,
        "input_payload_sha256": value.input_payload_sha256,
        "input_public_registry_id": value.input_public_registry_id,
        "input_row_id": value.input_row_id,
        "input_transform_result_id": value.input_transform_result_id,
        "prediction": _prediction_mapping_v2(value.prediction),
        "prediction_content_id": value.prediction_content_id,
        "run_context_id": value.run_context_id,
        "schema_version": value.schema_version,
    }


def _record_mapping_v2(
    value: "PublicRecognizerPredictionRecordV2",
) -> dict[str, object]:
    mapping = _record_mapping_without_id_v2(value)
    mapping["record_id"] = value.record_id
    return mapping


def _validate_record_preimage_v2(value: object) -> dict[str, object]:
    mapping = _closed_v2(
        value,
        _RECORD_PREIMAGE_FIELDS_V2,
        "V2 public prediction record preimage",
    )
    if (
        type(mapping["schema_version"]) is not str
        or mapping["schema_version"]
        != PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION
    ):
        raise ValueError("V2 prediction record schema drift")
    _digest_v2(
        mapping["run_context_id"],
        "phase2b_public_prediction_run_context_v2_",
        "V2 prediction record context ID",
    )
    _digest_v2(
        mapping["input_row_id"],
        "phase2b_recognizer_input_row_v2_",
        "V2 prediction input row ID",
    )
    _digest_v2(
        mapping["input_envelope_id"],
        "phase2b_trusted_envelope_v2_",
        "V2 prediction input envelope ID",
    )
    _digest_v2(
        mapping["input_namespace_audit_id"],
        "phase2b_namespace_audit_v2_",
        "V2 prediction namespace audit ID",
    )
    _digest_v2(
        mapping["input_authority_content_id"],
        "phase2b_public_transform_evidence_",
        "V2 prediction authority ID",
    )
    _digest_v2(
        mapping["input_transform_result_id"],
        "phase2b_exact_transform_result_",
        "V2 prediction transform ID",
    )
    _digest_v2(
        mapping["input_public_registry_id"],
        "phase2b_public_recognizer_registry_v2_",
        "V2 prediction registry ID",
    )
    _hex64_v2(mapping["input_payload_sha256"], "V2 prediction payload SHA-256")
    _hex64_v2(mapping["input_padding_sha256"], "V2 prediction padding SHA-256")
    if mapping["input_row_id"] != _input_row_id_from_record_roots_v2(mapping):
        raise ValueError("V2 prediction input row root recalculation drift")
    _bridge_outcome_id_v2(mapping["bridge_outcome_id"])
    if (mapping["bridge_compilation_id"] is None) != (
        mapping["bridge_decision_id"] is None
    ):
        raise ValueError("V2 prediction bridge roots are partial")
    if mapping["bridge_compilation_id"] is not None:
        _digest_v2(
            mapping["bridge_compilation_id"],
            "phase2b_exact_derived_bridge_result_",
            "V2 prediction bridge compilation ID",
        )
        _digest_v2(
            mapping["bridge_decision_id"],
            "phase2b_exact_derived_decision_",
            "V2 prediction bridge decision ID",
        )
    if type(mapping["decision"]) is not str:
        raise TypeError("V2 prediction decision wire requires exact text")
    try:
        decision = PredictionDecisionV2(mapping["decision"])
    except ValueError as exc:
        raise ValueError("V2 prediction decision wire drift") from exc
    is_preflight = mapping["bridge_outcome_id"].startswith(
        "phase2b_exact_derived_preflight_v2_"
    )
    if is_preflight:
        if (
            decision is not PredictionDecisionV2.ABSTAIN
            or mapping["bridge_compilation_id"] is not None
            or mapping["bridge_decision_id"] is not None
        ):
            raise ValueError("V2 preflight outcome stage shape drift")
    elif (
        mapping["bridge_compilation_id"] is None
        or mapping["bridge_decision_id"] is None
    ):
        raise ValueError("V2 derived-run outcome stage shape drift")
    raw_family = mapping["canonical_family_id"]
    if raw_family is None:
        canonical_family_id = None
    else:
        if type(raw_family) is not str:
            raise TypeError("V2 canonical family wire requires exact text or null")
        try:
            canonical_family_id = CanonicalFamilyId(raw_family)
        except ValueError as exc:
            raise ValueError("V2 canonical family wire drift") from exc
    prediction = _decode_prediction_bundle_v2(mapping["prediction"])
    if is_preflight and prediction.reason is not PredictionReason.RESOURCE_LIMIT:
        raise ValueError("V2 preflight outcome public reason drift")
    _digest_v2(
        mapping["prediction_content_id"],
        "phase2b_prediction_",
        "V2 prediction content ID",
    )
    if (
        prediction.content_id != mapping["prediction_content_id"]
        or prediction.input_root_sha256 != mapping["input_payload_sha256"]
    ):
        raise ValueError("V2 prediction bundle content or input root drift")
    _validate_decision_mapping_v2(
        decision=decision,
        canonical_family_id=canonical_family_id,
        prediction=prediction,
    )
    _reject_forbidden_public_semantics_v2(mapping)
    return mapping


def _record_id_v2(mapping_without_id: dict[str, object]) -> str:
    validated = _validate_record_preimage_v2(mapping_without_id)
    payload = encode_phase2b_jcs_profile_v1(validated)
    if not 1 <= len(payload) <= MAXIMUM_PREDICTION_RECORD_BYTES_V2:
        raise ValueError("V2 prediction record byte cap drift")
    return "phase2b_recognizer_prediction_record_v2_" + hashlib.sha256(
        _RECORD_DOMAIN_V2 + payload
    ).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerPredictionRecordV2:
    bridge_compilation_id: str | None
    bridge_decision_id: str | None
    bridge_outcome_id: str
    canonical_family_id: CanonicalFamilyId | None
    decision: PredictionDecisionV2
    input_authority_content_id: str
    input_envelope_id: str
    input_namespace_audit_id: str
    input_padding_sha256: str
    input_payload_sha256: str
    input_public_registry_id: str
    input_row_id: str
    input_transform_result_id: str
    prediction: PredictionBundle
    prediction_content_id: str
    record_id: str
    run_context_id: str
    schema_version: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 public prediction records are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        context: PublicPredictionRunContextV2,
        input_row: TrustedRecognizerInputRowV2,
        outcome: PublicRecognizerPredictionOutcomeV2,
    ) -> "PublicRecognizerPredictionRecordV2":
        if token is not _RECORD_ISSUE_TOKEN_V2:
            raise TypeError("V2 prediction record issuer token drift")
        if type(context) is not PublicPredictionRunContextV2:
            raise TypeError("V2 prediction record context exact type drift")
        if type(input_row) is not TrustedRecognizerInputRowV2:
            raise TypeError("V2 prediction record input row exact type drift")
        if type(outcome) is not PublicRecognizerPredictionOutcomeV2:
            raise TypeError("V2 prediction record outcome exact type drift")
        context._validate()
        outcome._validate()
        _validate_record_issue_input_row_v2(input_row)
        if (
            outcome.input_row_id != input_row.row_id
            or outcome.input_payload_sha256 != input_row.payload_sha256
            or outcome.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
            or outcome.prediction.protocol_sha256
            != context.protocol_id.rsplit("_", 1)[1]
            or outcome.prediction.freeze_manifest_sha256
            != context.execution_freeze_manifest_id.rsplit("_", 1)[1]
        ):
            raise ValueError("V2 prediction outcome row or context binding drift")
        _bridge_outcome_id_v2(outcome.bridge_outcome_id)
        canonical_prediction = _decode_prediction_bundle_v2(
            _prediction_mapping_unchecked_v2(outcome.prediction)
        )
        _validate_decision_mapping_v2(
            decision=outcome.decision,
            canonical_family_id=outcome.canonical_family_id,
            prediction=canonical_prediction,
        )
        value = object.__new__(cls)
        frozen = (
            ("bridge_compilation_id", outcome.bridge_compilation_id),
            ("bridge_decision_id", outcome.bridge_decision_id),
            ("bridge_outcome_id", outcome.bridge_outcome_id),
            ("canonical_family_id", outcome.canonical_family_id),
            ("decision", outcome.decision),
            ("input_authority_content_id", input_row.authority_content_id),
            ("input_envelope_id", input_row.envelope_id),
            ("input_namespace_audit_id", input_row.namespace_audit_id),
            ("input_padding_sha256", input_row.padding_sha256),
            ("input_payload_sha256", input_row.payload_sha256),
            ("input_public_registry_id", input_row.public_registry_id),
            ("input_row_id", input_row.row_id),
            ("input_transform_result_id", input_row.transform_result_id),
            ("prediction", canonical_prediction),
            ("prediction_content_id", canonical_prediction.content_id),
            ("run_context_id", context.context_id),
            ("schema_version", PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        object.__setattr__(
            value,
            "record_id",
            _record_id_v2(_record_mapping_without_id_v2(value)),
        )
        value._validate(context=context, token=_RECORD_CONTEXT_TOKEN_V2)
        return value

    def _validate(
        self,
        *,
        context: PublicPredictionRunContextV2 | None = None,
        token: object | None = None,
    ) -> None:
        if type(self) is not PublicRecognizerPredictionRecordV2:
            raise TypeError("V2 prediction record exact type drift")
        if (context is None) is not (token is None):
            raise TypeError("V2 prediction record context token shape drift")
        if context is not None and token is not _RECORD_CONTEXT_TOKEN_V2:
            raise TypeError("V2 prediction record context is private")
        validated = _validate_record_preimage_v2(_record_mapping_without_id_v2(self))
        _digest_v2(
            self.record_id,
            "phase2b_recognizer_prediction_record_v2_",
            "V2 prediction record ID",
        )
        if self.record_id != _record_id_v2(validated):
            raise ValueError("V2 prediction record root drift")
        if context is not None:
            context._validate()
            if (
                self.run_context_id != context.context_id
                or self.prediction.protocol_sha256
                != context.protocol_id.rsplit("_", 1)[1]
                or self.prediction.freeze_manifest_sha256
                != context.execution_freeze_manifest_id.rsplit("_", 1)[1]
            ):
                raise ValueError("V2 prediction record context binding drift")


def _decode_record_v2(
    mapping: object,
    *,
    context: PublicPredictionRunContextV2,
) -> PublicRecognizerPredictionRecordV2:
    value = _closed_v2(mapping, _RECORD_FIELDS_V2, "V2 public prediction record")
    preimage = {name: value[name] for name in _RECORD_PREIMAGE_FIELDS_V2}
    _validate_record_preimage_v2(preimage)
    if type(value["decision"]) is not str:
        raise TypeError("V2 prediction decision wire exact text drift")
    try:
        decision = PredictionDecisionV2(value["decision"])
    except ValueError as exc:
        raise ValueError("V2 prediction decision wire value is unknown") from exc
    raw_family = value["canonical_family_id"]
    if raw_family is None:
        canonical_family_id = None
    else:
        if type(raw_family) is not str:
            raise TypeError("V2 canonical family wire exact text drift")
        try:
            canonical_family_id = CanonicalFamilyId(raw_family)
        except ValueError as exc:
            raise ValueError("V2 canonical family wire value is unknown") from exc
    record = object.__new__(PublicRecognizerPredictionRecordV2)
    decoded = (
        ("bridge_compilation_id", value["bridge_compilation_id"]),
        ("bridge_decision_id", value["bridge_decision_id"]),
        ("bridge_outcome_id", value["bridge_outcome_id"]),
        ("canonical_family_id", canonical_family_id),
        ("decision", decision),
        ("input_authority_content_id", value["input_authority_content_id"]),
        ("input_envelope_id", value["input_envelope_id"]),
        ("input_namespace_audit_id", value["input_namespace_audit_id"]),
        ("input_padding_sha256", value["input_padding_sha256"]),
        ("input_payload_sha256", value["input_payload_sha256"]),
        ("input_public_registry_id", value["input_public_registry_id"]),
        ("input_row_id", value["input_row_id"]),
        ("input_transform_result_id", value["input_transform_result_id"]),
        ("prediction", _decode_prediction_bundle_v2(value["prediction"])),
        ("prediction_content_id", value["prediction_content_id"]),
        ("record_id", value["record_id"]),
        ("run_context_id", value["run_context_id"]),
        ("schema_version", value["schema_version"]),
    )
    for name, item in decoded:
        object.__setattr__(record, name, item)
    record._validate(context=context, token=_RECORD_CONTEXT_TOKEN_V2)
    _reject_forbidden_public_semantics_v2(value)
    if _record_mapping_v2(record) != value:
        raise ValueError("V2 prediction record canonical roundtrip drift")
    return record


def _manifest_mapping_v2(
    *,
    context: PublicPredictionRunContextV2,
    prediction_record_ids_root: str,
) -> dict[str, object]:
    context._validate()
    _digest_v2(
        prediction_record_ids_root,
        "phase2b_prediction_records_v2_",
        "V2 prediction record sequence root",
    )
    mapping: dict[str, object] = {
        "archive_policy_id": RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
        "archive_schema_version": RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
        "batch_id": context.batch_id,
        "batch_policy_id": context.batch_policy_id,
        "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
        "execution_freeze_manifest_id": context.execution_freeze_manifest_id,
        "expected_prediction_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "input_archive_id": context.input_archive_id,
        "input_archive_policy_id": context.input_archive_policy_id,
        "input_archive_sha256": context.input_archive_sha256,
        "input_archive_version": context.input_archive_version,
        "input_row_ids_root": context.input_row_ids_root,
        "prediction_record_ids_root": prediction_record_ids_root,
        "protocol_id": context.protocol_id,
        "record_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "run_context_id": context.context_id,
    }
    _reject_forbidden_public_semantics_v2(mapping)
    return mapping


def _validate_manifest_v2(
    value: object,
) -> tuple[dict[str, object], PublicPredictionRunContextV2]:
    mapping = _closed_v2(value, _MANIFEST_FIELDS_V2, "V2 prediction manifest")
    string_fields = tuple(
        name for name in _MANIFEST_FIELDS_V2
        if name not in {"expected_prediction_count", "record_count"}
    )
    if any(type(mapping[name]) is not str for name in string_fields):
        raise TypeError("V2 prediction manifest identity fields require exact strings")
    if any(
        type(mapping[name]) is not int
        or mapping[name] != TOTAL_RECOGNIZER_CASE_COUNT
        for name in ("expected_prediction_count", "record_count")
    ):
        raise ValueError("V2 prediction manifest count drift")
    if (
        mapping["archive_policy_id"] != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or mapping["archive_schema_version"]
        != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or mapping["batch_policy_id"] != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or mapping["input_archive_policy_id"]
        != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or mapping["input_archive_version"]
        != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        or mapping["protocol_id"] != _FROZEN_PROTOCOL.protocol_id
        or mapping["claim_level"] != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("V2 prediction manifest policy identity drift")
    _digest_v2(
        mapping["archive_policy_id"],
        "phase2b_recognizer_prediction_archive_policy_v2_",
        "V2 prediction archive policy ID",
    )
    _digest_v2(
        mapping["batch_id"],
        "phase2b_trusted_wire_batch_v2_",
        "V2 prediction manifest batch ID",
    )
    _digest_v2(
        mapping["batch_policy_id"],
        "phase2b_trusted_wire_batch_v2_policy_",
        "V2 prediction manifest batch policy",
    )
    _digest_v2(
        mapping["input_archive_id"],
        "phase2b_recognizer_input_archive_v2_",
        "V2 prediction manifest input archive ID",
    )
    _digest_v2(
        mapping["input_archive_policy_id"],
        "phase2b_recognizer_input_archive_policy_v2_",
        "V2 prediction manifest input archive policy",
    )
    _hex64_v2(mapping["input_archive_sha256"], "V2 prediction input archive SHA-256")
    _digest_v2(
        mapping["input_row_ids_root"],
        "phase2b_prediction_input_rows_v2_",
        "V2 prediction input row root",
    )
    _digest_v2(
        mapping["prediction_record_ids_root"],
        "phase2b_prediction_records_v2_",
        "V2 prediction record root",
    )
    _digest_v2(mapping["protocol_id"], "phase2b_protocol_", "V2 prediction protocol")
    _digest_v2(
        mapping["execution_freeze_manifest_id"],
        "phase2b_execution_freeze_",
        "V2 prediction execution freeze",
    )
    _digest_v2(
        mapping["run_context_id"],
        "phase2b_public_prediction_run_context_v2_",
        "V2 prediction run context ID",
    )
    context = PublicPredictionRunContextV2._issue(
        _CONTEXT_ISSUE_TOKEN_V2,
        batch_id=mapping["batch_id"],  # type: ignore[arg-type]
        input_archive_id=mapping["input_archive_id"],  # type: ignore[arg-type]
        input_archive_sha256=mapping["input_archive_sha256"],  # type: ignore[arg-type]
        input_row_ids_root=mapping["input_row_ids_root"],  # type: ignore[arg-type]
        execution_freeze_manifest_id=mapping["execution_freeze_manifest_id"],  # type: ignore[arg-type]
    )
    if context.context_id != mapping["run_context_id"]:
        raise ValueError("V2 prediction manifest context root drift")
    _reject_forbidden_public_semantics_v2(mapping)
    return mapping, context


def _encode_prediction_archive_v2(
    *,
    context: PublicPredictionRunContextV2,
    records: tuple[PublicRecognizerPredictionRecordV2, ...],
) -> bytes:
    if (
        type(records) is not tuple
        or len(records) != TOTAL_RECOGNIZER_CASE_COUNT
        or any(type(item) is not PublicRecognizerPredictionRecordV2 for item in records)
    ):
        raise TypeError("V2 prediction archive requires exactly 960 exact records")
    context._validate()
    for record in records:
        record._validate(context=context, token=_RECORD_CONTEXT_TOKEN_V2)
    input_row_ids = tuple(record.input_row_id for record in records)
    record_ids = tuple(record.record_id for record in records)
    if (
        len(set(input_row_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
        or len(set(record_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
    ):
        raise ValueError("V2 prediction archive repeats a row or record root")
    if _input_row_ids_root_v2(input_row_ids) != context.input_row_ids_root:
        raise ValueError("V2 prediction archive changed input wire order")
    manifest = _manifest_mapping_v2(
        context=context,
        prediction_record_ids_root=_prediction_record_ids_root_v2(record_ids),
    )
    manifest_payload = encode_phase2b_jcs_profile_v1(manifest)
    if not 1 <= len(manifest_payload) <= MAXIMUM_PREDICTION_MANIFEST_BYTES_V2:
        raise ValueError("V2 prediction manifest byte cap drift")
    body_parts = [manifest_payload]
    total = PREDICTION_ARCHIVE_HEADER_BYTES_V2 + len(manifest_payload)
    for record in records:
        payload = encode_phase2b_jcs_profile_v1(_record_mapping_v2(record))
        if not 1 <= len(payload) <= MAXIMUM_PREDICTION_RECORD_BYTES_V2:
            raise ValueError("V2 prediction record byte cap drift")
        framed = struct.pack(">I", len(payload)) + payload
        total += len(framed)
        if total > MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2:
            raise ValueError("V2 prediction archive byte cap exceeded")
        body_parts.append(framed)
    body = b"".join(body_parts)
    header = _PREDICTION_ARCHIVE_HEADER_V2.pack(
        PREDICTION_ARCHIVE_MAGIC_V2,
        PREDICTION_ARCHIVE_WIRE_VERSION_V2,
        PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        len(manifest_payload),
        TOTAL_RECOGNIZER_CASE_COUNT,
        hashlib.sha256(body).digest(),
    )
    archive = header + body
    if len(archive) != total:
        raise RuntimeError("V2 prediction archive byte accounting drift")
    return archive


def _archive_id_v2(archive: bytes) -> str:
    if type(archive) is not bytes or not (
        PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= len(archive)
        <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
    ):
        raise ValueError("V2 prediction archive ID byte cap drift")
    return "phase2b_recognizer_prediction_archive_v2_" + hashlib.sha256(
        _ARCHIVE_DOMAIN_V2 + archive
    ).hexdigest()


def _read_record_frame_v2(
    archive: bytes,
    offset: int,
    *,
    record_index: int,
) -> tuple[bytes, int]:
    if type(offset) is not int or type(record_index) is not int:
        raise TypeError("V2 prediction frame offsets require exact ints")
    if offset + 4 > len(archive):
        raise ValueError(f"V2 prediction record {record_index} length truncated")
    (length,) = struct.unpack_from(">I", archive, offset)
    offset += 4
    if not 1 <= length <= MAXIMUM_PREDICTION_RECORD_BYTES_V2:
        raise ValueError(f"V2 prediction record {record_index} length cap drift")
    end = offset + length
    if end > len(archive):
        raise ValueError(f"V2 prediction record {record_index} truncated")
    return archive[offset:end], end


@dataclass(frozen=True, slots=True)
class _ParsedPredictionArchiveV2:
    archive_id: str
    context: PublicPredictionRunContextV2
    records: tuple[PublicRecognizerPredictionRecordV2, ...]


def _parse_prediction_archive_v2(archive: bytes) -> _ParsedPredictionArchiveV2:
    if type(archive) is not bytes:
        raise TypeError("V2 prediction archive input requires exact bytes")
    minimum = (
        PREDICTION_ARCHIVE_HEADER_BYTES_V2
        + 1
        + TOTAL_RECOGNIZER_CASE_COUNT * (4 + 1)
    )
    if not minimum <= len(archive) <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2:
        raise ValueError("V2 prediction archive byte cap drift")
    (
        magic,
        wire_version,
        header_bytes,
        manifest_length,
        record_count,
        body_sha256,
    ) = _PREDICTION_ARCHIVE_HEADER_V2.unpack_from(archive, 0)
    if (
        magic != PREDICTION_ARCHIVE_MAGIC_V2
        or wire_version != PREDICTION_ARCHIVE_WIRE_VERSION_V2
        or header_bytes != PREDICTION_ARCHIVE_HEADER_BYTES_V2
    ):
        raise ValueError("V2 prediction archive header discriminator drift")
    if record_count != TOTAL_RECOGNIZER_CASE_COUNT:
        raise ValueError("V2 prediction archive record count is not exact 960")
    if not 1 <= manifest_length <= MAXIMUM_PREDICTION_MANIFEST_BYTES_V2:
        raise ValueError("V2 prediction archive manifest length cap drift")
    body = archive[PREDICTION_ARCHIVE_HEADER_BYTES_V2 :]
    if (
        manifest_length + record_count * (4 + 1) > len(body)
        or hashlib.sha256(body).digest() != body_sha256
    ):
        raise ValueError("V2 prediction archive body shape or digest drift")
    manifest_end = PREDICTION_ARCHIVE_HEADER_BYTES_V2 + manifest_length
    manifest_payload = archive[PREDICTION_ARCHIVE_HEADER_BYTES_V2 : manifest_end]
    manifest_object = decode_phase2b_jcs_profile_v1(manifest_payload)
    manifest, context = _validate_manifest_v2(manifest_object)
    if encode_phase2b_jcs_profile_v1(manifest) != manifest_payload:
        raise ValueError("V2 prediction manifest is not canonical accepted JCS")
    offset = manifest_end
    records: list[PublicRecognizerPredictionRecordV2] = []
    for index in range(TOTAL_RECOGNIZER_CASE_COUNT):
        payload, offset = _read_record_frame_v2(
            archive,
            offset,
            record_index=index,
        )
        record_object = decode_phase2b_jcs_profile_v1(payload)
        record = _decode_record_v2(record_object, context=context)
        if encode_phase2b_jcs_profile_v1(_record_mapping_v2(record)) != payload:
            raise ValueError(f"V2 prediction record {index} is not canonical JCS")
        records.append(record)
    if offset != len(archive):
        raise ValueError("V2 prediction archive has trailing bytes")
    frozen_records = tuple(records)
    input_row_ids = tuple(record.input_row_id for record in frozen_records)
    record_ids = tuple(record.record_id for record in frozen_records)
    if (
        len(set(input_row_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
        or len(set(record_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
    ):
        raise ValueError("V2 prediction archive repeats a row or record root")
    if (
        _input_row_ids_root_v2(input_row_ids) != manifest["input_row_ids_root"]
        or _prediction_record_ids_root_v2(record_ids)
        != manifest["prediction_record_ids_root"]
        or context.input_row_ids_root != manifest["input_row_ids_root"]
    ):
        raise ValueError("V2 prediction archive ordered coverage root drift")
    return _ParsedPredictionArchiveV2(
        archive_id=_archive_id_v2(archive),
        context=context,
        records=frozen_records,
    )


@dataclass(frozen=True, slots=True, init=False)
class DecodedRecognizerPredictionArchiveV2:
    disposition: PredictionArchiveDispositionV2
    archive: bytes
    archive_id: str
    schema_version: str
    policy_id: str
    context: PublicPredictionRunContextV2
    records: tuple[PublicRecognizerPredictionRecordV2, ...]
    input_row_ids: tuple[str, ...]
    prediction_record_ids: tuple[str, ...]
    prediction_content_ids: tuple[str, ...]
    claim_level: str
    structural_archive_verified: bool
    canonical_record_framing_verified: bool
    record_schema_verified: bool
    row_root_coverage_verified: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    execution_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    prediction_scored: bool
    effect_evidence: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 decoded prediction archives are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        archive: bytes,
        parsed: _ParsedPredictionArchiveV2,
    ) -> "DecodedRecognizerPredictionArchiveV2":
        if token is not _DECODE_ISSUE_TOKEN_V2:
            raise TypeError("V2 decoded prediction issuer token drift")
        if type(parsed) is not _ParsedPredictionArchiveV2:
            raise TypeError("V2 decoded prediction parsed context type drift")
        current = _parse_prediction_archive_v2(archive)
        if (
            type(parsed.archive_id) is not str
            or type(parsed.context) is not PublicPredictionRunContextV2
            or type(parsed.records) is not tuple
            or len(parsed.records) != TOTAL_RECOGNIZER_CASE_COUNT
            or any(
                type(item) is not PublicRecognizerPredictionRecordV2
                for item in parsed.records
            )
        ):
            raise TypeError("V2 decoded prediction supplied parsed shape drift")
        parsed.context._validate()
        for record in parsed.records:
            record._validate(
                context=parsed.context,
                token=_RECORD_CONTEXT_TOKEN_V2,
            )
        if (
            parsed.archive_id != current.archive_id
            or parsed.context.context_id != current.context.context_id
            or tuple(item.record_id for item in parsed.records)
            != tuple(item.record_id for item in current.records)
        ):
            raise ValueError("V2 decoded prediction supplied parsed context drift")
        value = object.__new__(cls)
        records = current.records
        frozen = (
            ("disposition", PredictionArchiveDispositionV2.COMPLETE),
            ("archive", archive),
            ("archive_id", current.archive_id),
            ("schema_version", RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION),
            ("policy_id", RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2),
            ("context", current.context),
            ("records", records),
            ("input_row_ids", tuple(item.input_row_id for item in records)),
            ("prediction_record_ids", tuple(item.record_id for item in records)),
            (
                "prediction_content_ids",
                tuple(item.prediction_content_id for item in records),
            ),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        for name in _TRUE_DECODED_CLAIMS_V2:
            object.__setattr__(value, name, True)
        for name in _FALSE_DECODED_CLAIMS_V2:
            object.__setattr__(value, name, False)
        value._validate(parsed=current, token=_PARSED_CONTEXT_TOKEN_V2)
        return value

    def _validate(
        self,
        *,
        parsed: _ParsedPredictionArchiveV2 | None = None,
        token: object | None = None,
    ) -> None:
        if type(self) is not DecodedRecognizerPredictionArchiveV2:
            raise TypeError("V2 decoded prediction archive exact type drift")
        if (parsed is None) is not (token is None):
            raise TypeError("V2 decoded prediction parsed context token shape drift")
        if parsed is not None and token is not _PARSED_CONTEXT_TOKEN_V2:
            raise TypeError("V2 decoded prediction parsed context is private")
        if type(self.archive) is not bytes or not (
            PREDICTION_ARCHIVE_HEADER_BYTES_V2
            <= len(self.archive)
            <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
        ):
            raise TypeError("V2 decoded prediction archive bytes shape drift")
        if self.disposition is not PredictionArchiveDispositionV2.COMPLETE:
            raise ValueError("V2 decoded prediction disposition drift")
        claims = tuple(
            getattr(self, name)
            for name in (*_TRUE_DECODED_CLAIMS_V2, *_FALSE_DECODED_CLAIMS_V2)
        )
        if (
            any(type(item) is not bool for item in claims)
            or not all(getattr(self, name) for name in _TRUE_DECODED_CLAIMS_V2)
            or any(getattr(self, name) for name in _FALSE_DECODED_CLAIMS_V2)
        ):
            raise ValueError("V2 decoded prediction claim boundary drift")
        _digest_v2(
            self.archive_id,
            "phase2b_recognizer_prediction_archive_v2_",
            "V2 decoded prediction archive ID",
        )
        if (
            type(self.schema_version) is not str
            or self.schema_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
            or type(self.policy_id) is not str
            or self.policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("V2 decoded prediction policy identity drift")
        _digest_v2(
            self.policy_id,
            "phase2b_recognizer_prediction_archive_policy_v2_",
            "V2 decoded prediction policy ID",
        )
        if type(self.context) is not PublicPredictionRunContextV2:
            raise TypeError("V2 decoded prediction context exact type drift")
        self.context._validate()
        if (
            type(self.records) is not tuple
            or len(self.records) != TOTAL_RECOGNIZER_CASE_COUNT
            or any(type(item) is not PublicRecognizerPredictionRecordV2 for item in self.records)
        ):
            raise TypeError("V2 decoded prediction record tuple drift")
        column_specs = (
            (
                self.input_row_ids,
                "phase2b_recognizer_input_row_v2_",
                "V2 decoded prediction input row ID",
            ),
            (
                self.prediction_record_ids,
                "phase2b_recognizer_prediction_record_v2_",
                "V2 decoded prediction record ID",
            ),
            (
                self.prediction_content_ids,
                "phase2b_prediction_",
                "V2 decoded prediction content ID",
            ),
        )
        for column, prefix, name in column_specs:
            if type(column) is not tuple or len(column) != TOTAL_RECOGNIZER_CASE_COUNT:
                raise TypeError("V2 decoded prediction root column shape drift")
            for item in column:
                _digest_v2(item, prefix, name)
        if (
            len(set(self.input_row_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
            or len(set(self.prediction_record_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            raise ValueError("V2 decoded prediction repeats row or record root")
        # Prediction content IDs deliberately may repeat for identical public
        # PredictionBundles; they are content roots, not row identifiers.
        for record in self.records:
            _digest_v2(
                record.input_row_id,
                "phase2b_recognizer_input_row_v2_",
                "V2 decoded record input row ID",
            )
            _digest_v2(
                record.record_id,
                "phase2b_recognizer_prediction_record_v2_",
                "V2 decoded record ID",
            )
            _digest_v2(
                record.prediction_content_id,
                "phase2b_prediction_",
                "V2 decoded record prediction content ID",
            )
        expected_columns = (
            tuple(item.input_row_id for item in self.records),
            tuple(item.record_id for item in self.records),
            tuple(item.prediction_content_id for item in self.records),
        )
        if (
            (self.input_row_ids, self.prediction_record_ids, self.prediction_content_ids)
            != expected_columns
        ):
            raise ValueError("V2 decoded prediction stored root column drift")
        if parsed is not None:
            if (
                type(parsed) is not _ParsedPredictionArchiveV2
                or type(parsed.archive_id) is not str
                or type(parsed.context) is not PublicPredictionRunContextV2
                or type(parsed.records) is not tuple
                or parsed.context is not self.context
                or parsed.records is not self.records
                or _archive_id_v2(self.archive) != parsed.archive_id
                or self.archive_id != parsed.archive_id
            ):
                raise ValueError("V2 decoded prediction parsed bytes or identity drift")
        for record in self.records:
            record._validate(context=self.context, token=_RECORD_CONTEXT_TOKEN_V2)
        actual = _parse_prediction_archive_v2(self.archive) if parsed is None else parsed
        if type(actual) is not _ParsedPredictionArchiveV2:
            raise TypeError("V2 decoded prediction parsed context exact type drift")
        if (
            self.archive_id != actual.archive_id
            or self.context != actual.context
            or self.records != actual.records
            or self.input_row_ids != tuple(item.input_row_id for item in actual.records)
            or self.prediction_record_ids != tuple(item.record_id for item in actual.records)
            or self.prediction_content_ids
            != tuple(item.prediction_content_id for item in actual.records)
            or _input_row_ids_root_v2(self.input_row_ids)
            != self.context.input_row_ids_root
        ):
            raise ValueError("V2 decoded prediction public replay drift")


@dataclass(frozen=True, slots=True)
class RecognizerPredictionArchiveRejectionV2:
    disposition: PredictionArchiveDispositionV2
    reason: str
    input_count: int
    input_archive_id: str | None
    archive: None = None
    records: tuple[()] = ()
    prediction_record_ids: tuple[()] = ()
    recognizer_capacity_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not RecognizerPredictionArchiveRejectionV2:
            raise TypeError("V2 prediction rejection exact type drift")
        if self.disposition is not PredictionArchiveDispositionV2.ABSTAIN:
            raise ValueError("V2 prediction rejection must abstain")
        _ascii_v2(self.reason, "V2 prediction rejection reason")
        if type(self.input_count) is not int or self.input_count < 0:
            raise ValueError("V2 prediction rejection count drift")
        if self.input_archive_id is not None:
            _digest_v2(
                self.input_archive_id,
                "phase2b_recognizer_input_archive_v2_",
                "V2 prediction rejection input archive ID",
            )
        if (
            self.archive is not None
            or type(self.records) is not tuple
            or self.records != ()
            or type(self.prediction_record_ids) is not tuple
            or self.prediction_record_ids != ()
            or type(self.recognizer_capacity_evidence) is not bool
            or self.recognizer_capacity_evidence
        ):
            raise ValueError("V2 prediction rejection leaked partial output")


def _rejection_v2(
    reason: str,
    *,
    input_count: int,
    input_archive_id: str | None,
) -> RecognizerPredictionArchiveRejectionV2:
    return RecognizerPredictionArchiveRejectionV2(
        disposition=PredictionArchiveDispositionV2.ABSTAIN,
        reason=reason,
        input_count=input_count,
        input_archive_id=input_archive_id,
    )


def _safe_shallow_rejection_context_v2(
    input_archive: DecodedRecognizerInputArchiveV2,
) -> tuple[int, str | None]:
    count = 0
    archive_id: str | None = None
    try:
        rows = object.__getattribute__(input_archive, "rows")
        if type(rows) is tuple:
            count = len(rows)
    except (AttributeError, TypeError):
        pass
    try:
        candidate = object.__getattribute__(input_archive, "archive_id")
        archive_id = _digest_v2(
            candidate,
            "phase2b_recognizer_input_archive_v2_",
            "V2 rejection input archive ID",
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return count, archive_id


def decode_public_recognizer_prediction_archive_v2(
    archive: bytes,
) -> DecodedRecognizerPredictionArchiveV2:
    """Decode bounded V2 bytes as structural mechanics and nothing more."""

    parsed = _parse_prediction_archive_v2(archive)
    return DecodedRecognizerPredictionArchiveV2._issue(
        _DECODE_ISSUE_TOKEN_V2,
        archive=archive,
        parsed=parsed,
    )


def build_recognizer_prediction_archive_v2(
    *,
    input_archive: DecodedRecognizerInputArchiveV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> DecodedRecognizerPredictionArchiveV2 | RecognizerPredictionArchiveRejectionV2:
    """Build atomically from exact V2 input bytes and the current freeze."""

    if type(input_archive) is not DecodedRecognizerInputArchiveV2:
        raise TypeError("V2 prediction builder requires exact input archive")
    if type(execution_freeze_manifest) is not ExecutionFreezeManifest:
        raise TypeError("V2 prediction builder requires exact execution freeze")
    try:
        _, shallow_row_ids = _shallow_prediction_input_v2(input_archive)
    except _ShallowPredictionInputError as exc:
        return _rejection_v2(
            exc.reason,
            input_count=exc.input_count,
            input_archive_id=exc.input_archive_id,
        )
    except (AttributeError, KeyError, OverflowError, RecursionError, RuntimeError, TypeError, ValueError):
        count, archive_id = _safe_shallow_rejection_context_v2(input_archive)
        return _rejection_v2(
            "prediction_input_shallow_validation_failed",
            input_count=count,
            input_archive_id=archive_id,
        )
    input_count = len(shallow_row_ids)
    _, shallow_archive_id = _safe_shallow_rejection_context_v2(input_archive)
    try:
        _validate_execution_freeze_manifest_v2(execution_freeze_manifest)
        # This is the sole public V2 input-archive decode in a successful build.
        canonical_input = _input_v2.decode_public_recognizer_input_archive_v2(
            input_archive.archive
        )
        if type(canonical_input) is not DecodedRecognizerInputArchiveV2:
            raise TypeError("V2 public input decoder returned nonexact archive")
        if (
            canonical_input.archive != input_archive.archive
            or canonical_input.archive_id != shallow_archive_id
            or canonical_input.row_ids != shallow_row_ids
            or canonical_input.disposition is not RecognizerInputArchiveDispositionV2.COMPLETE
            or canonical_input.policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
            or canonical_input.archive_version
            != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
            or canonical_input.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        ):
            raise ValueError("V2 input archive public self-decode parity drift")
        canonical_rows = canonical_input.rows
        if (
            type(canonical_rows) is not tuple
            or len(canonical_rows) != TOTAL_RECOGNIZER_CASE_COUNT
            or any(type(row) is not TrustedRecognizerInputRowV2 for row in canonical_rows)
        ):
            raise TypeError("V2 canonical input row tuple drift")
        # The input archive digest is intentionally computed only after the
        # bounded public self-decode above has accepted all bytes.
        context = PublicPredictionRunContextV2._issue(
            _CONTEXT_ISSUE_TOKEN_V2,
            batch_id=canonical_input.batch_id,
            input_archive_id=canonical_input.archive_id,
            input_archive_sha256=hashlib.sha256(canonical_input.archive).hexdigest(),
            input_row_ids_root=_input_row_ids_root_v2(canonical_input.row_ids),
            execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
        )
        records: list[PublicRecognizerPredictionRecordV2] = []
        for row in canonical_rows:
            # The module alias is used only for this public API call.  Raw
            # archive decoding never consumes the mapper's private helpers.
            outcome = _prediction_v2.recognize_public_input_row_v2(
                input_row=row,
                execution_freeze_manifest=execution_freeze_manifest,
            )
            if type(outcome) is not PublicRecognizerPredictionOutcomeV2:
                raise TypeError("V2 row mapper returned nonexact outcome")
            records.append(
                PublicRecognizerPredictionRecordV2._issue(
                    _RECORD_ISSUE_TOKEN_V2,
                    context=context,
                    input_row=row,
                    outcome=outcome,
                )
            )
        frozen_records = tuple(records)
        archive = _encode_prediction_archive_v2(
            context=context,
            records=frozen_records,
        )
        decoded = decode_public_recognizer_prediction_archive_v2(archive)
        if (
            decoded.context != context
            or decoded.records != frozen_records
            or decoded.input_row_ids != canonical_input.row_ids
        ):
            raise ValueError("V2 prediction archive public self-decode parity drift")
        return decoded
    except (AttributeError, KeyError, OverflowError, RecursionError, RuntimeError, TypeError, ValueError):
        return _rejection_v2(
            "recognizer_prediction_archive_v2_failed",
            input_count=input_count,
            input_archive_id=shallow_archive_id,
        )


def _assert_public_field_manifests_v2() -> None:
    for name, manifest in (
        ("context", _CONTEXT_FIELDS_V2),
        ("context preimage", _CONTEXT_PREIMAGE_FIELDS_V2),
        ("record", _RECORD_FIELDS_V2),
        ("record preimage", _RECORD_PREIMAGE_FIELDS_V2),
        ("manifest", _MANIFEST_FIELDS_V2),
        ("prediction", _PREDICTION_BUNDLE_FIELDS),
    ):
        if manifest != tuple(sorted(manifest)):
            raise RuntimeError(f"V2 prediction {name} field manifest is not canonical")
    actual = (
        tuple(item.name for item in fields(PublicPredictionRunContextV2)),
        tuple(item.name for item in fields(PublicRecognizerPredictionRecordV2)),
        tuple(item.name for item in fields(DecodedRecognizerPredictionArchiveV2)),
        tuple(item.name for item in fields(RecognizerPredictionArchiveRejectionV2)),
    )
    expected = (
        _CONTEXT_FIELDS_V2,
        _RECORD_FIELDS_V2,
        _DECODED_FIELDS_V2,
        _REJECTION_FIELDS_V2,
    )
    if actual != expected:
        raise RuntimeError("V2 prediction public dataclass field manifest drift")
    if TOTAL_RECOGNIZER_CASE_COUNT != 960:
        raise RuntimeError("V2 prediction frozen case count drift")


_assert_public_field_manifests_v2()


__all__ = (
    "MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2",
    "MAXIMUM_PREDICTION_MANIFEST_BYTES_V2",
    "MAXIMUM_PREDICTION_RECORD_BYTES_V2",
    "PREDICTION_ARCHIVE_HEADER_BYTES_V2",
    "PREDICTION_ARCHIVE_MAGIC_V2",
    "PREDICTION_ARCHIVE_WIRE_VERSION_V2",
    "PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID",
    "PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION",
    "PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_ID",
    "PUBLIC_RECOGNIZER_PREDICTION_RECORD_V2_SCHEMA_VERSION",
    "RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2",
    "RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION",
    "DecodedRecognizerPredictionArchiveV2",
    "PredictionArchiveDispositionV2",
    "PublicPredictionRunContextV2",
    "PublicRecognizerPredictionRecordV2",
    "RecognizerPredictionArchiveRejectionV2",
    "build_recognizer_prediction_archive_v2",
    "decode_public_recognizer_prediction_archive_v2",
)

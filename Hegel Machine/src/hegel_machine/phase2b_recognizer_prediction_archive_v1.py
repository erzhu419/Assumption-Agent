"""Split-blind, non-authoritative Phase-2B prediction archive mechanics.

The builder replays each public recognizer-input row through the frozen
derived-witness bridge as an issuance gate.  The durable result is deliberately
only a public, structurally replayable archive: it does not prove that an
external recognizer ran, that an execution manifest was independently issued,
that the input rows belong to a trusted holdout, or that any prediction is
correct.  No answer key or 720/240 cohort label enters this module's public
wire format.
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
from .phase2b_exact_bridge_v1 import (
    ExactBridgeDisposition,
    ExactCandidateStatus,
    ExactSelectionDisposition,
)
from .phase2b_exact_derived_witness_bridge_v1 import (
    EXACT_DERIVED_BRIDGE_POLICY_ID,
    EXACT_DERIVED_MATCHER_SEMANTICS_ID,
    EXACT_DERIVED_SELECTION_POLICY_ID,
    ExactDerivedBridgePreflightRejection,
    ExactDerivedBridgeRun,
    run_exact_derived_witness_bridge,
)
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from .phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
    frozen_phase2b_protocol,
)
from .phase2b_recognizer_input_archive_v1 import (
    FROZEN_BRIDGE_FAMILY_UUID_ALIASES,
    PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID,
    PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID,
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID,
    DecodedRecognizerInputArchiveV1,
    PublicRecognizerRegistryV1,
    TrustedRecognizerInputRowV1,
)
from .phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT
from .phase2b_trusted_wire_typed_replay_v1 import (
    TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID,
    TypedTrustedEnvelopeReplayV1,
    decode_and_replay_typed_trusted_envelope_v1,
)
from .phase2b_trusted_wire_v1 import (
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
from .schema import LawKind


PUBLIC_RUN_CONTEXT_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-run-context/1"
)
PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-recognizer-prediction-record/1"
)
RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-recognizer-prediction-archive/1"
)

PREDICTION_ARCHIVE_MAGIC: Final = b"HGP2PA1\x00"
PREDICTION_ARCHIVE_WIRE_VERSION: Final = 1
_ARCHIVE_HEADER: Final = struct.Struct(">8sHHI32s")
PREDICTION_ARCHIVE_HEADER_BYTES: Final = _ARCHIVE_HEADER.size
MAXIMUM_PREDICTION_MANIFEST_BYTES: Final = 16_384
MAXIMUM_PREDICTION_RECORD_BYTES: Final = 32_768
MAXIMUM_PREDICTION_ARCHIVE_BYTES: Final = (
    PREDICTION_ARCHIVE_HEADER_BYTES
    + 4
    + MAXIMUM_PREDICTION_MANIFEST_BYTES
    + TOTAL_RECOGNIZER_CASE_COUNT * (4 + MAXIMUM_PREDICTION_RECORD_BYTES)
)

_ARCHIVE_DOMAIN: Final = b"HEGEL/PHASE2B/RECOGNIZER_PREDICTION_ARCHIVE/V1\x00"
_RECORD_DOMAIN: Final = b"HEGEL/PHASE2B/RECOGNIZER_PREDICTION_RECORD/V1\x00"
_PREFLIGHT_DOMAIN: Final = b"HEGEL/PHASE2B/DERIVED_PREFLIGHT_OUTCOME/V1\x00"
_INPUT_ROWS_DOMAIN: Final = b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V1\x00"
_RECORD_IDS_DOMAIN: Final = b"HEGEL/PHASE2B/PREDICTION_RECORD_IDS/V1\x00"
_CONTEXT_ISSUE_TOKEN: Final = object()
_RECORD_ISSUE_TOKEN: Final = object()
_RECORD_CONTEXT_TOKEN: Final = object()
_DECODE_ISSUE_TOKEN: Final = object()
_PARSED_CONTEXT_TOKEN: Final = object()

_FROZEN_THEORY: Final = initial_theory()
_FROZEN_PROTOCOL: Final = frozen_phase2b_protocol()
_FROZEN_EXACT_FREEZE: Final = frozen_phase2b_exact_freeze()
_BRIDGE_FAMILY_BY_KIND: Final = dict(FROZEN_BRIDGE_FAMILY_UUID_ALIASES)
_CANONICAL_FAMILY_BY_KIND: Final = dict(_FROZEN_EXACT_FREEZE.family_mapping)
_KIND_BY_CANONICAL_FAMILY: Final = {
    family_id: law_kind
    for law_kind, family_id in _FROZEN_EXACT_FREEZE.family_mapping
}

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
)

_CONTEXT_FIELDS: Final = (
    "claim_level",
    "context_id",
    "execution_freeze_manifest_id",
    "expected_prediction_count",
    "input_archive_id",
    "input_archive_sha256",
    "input_row_ids_root",
    "protocol_id",
    "schema_version",
)
_CONTEXT_PREIMAGE_FIELDS: Final = tuple(
    name for name in _CONTEXT_FIELDS if name != "context_id"
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
_PREFLIGHT_OUTCOME_FIELDS: Final = (
    "disposition",
    "reason",
    "bundle_id",
    "wrapper_schema_version",
    "theory_schema_version",
    "registry_theory_version_id",
    "bridge_policy_id",
    "matcher_semantics_id",
)
_RECORD_FIELDS: Final = (
    "bridge_compilation_id",
    "bridge_decision_id",
    "bridge_outcome_id",
    "canonical_family_id",
    "decision",
    "input_authority_content_id",
    "input_payload_sha256",
    "input_row_id",
    "input_transform_result_id",
    "prediction",
    "prediction_content_id",
    "public_registry_id",
    "record_id",
    "run_context_id",
    "schema_version",
)
_RECORD_PREIMAGE_FIELDS: Final = tuple(
    name for name in _RECORD_FIELDS if name != "record_id"
)
_MANIFEST_FIELDS: Final = (
    "archive_policy_id",
    "archive_schema_version",
    "claim_level",
    "execution_freeze_manifest_id",
    "expected_prediction_count",
    "input_archive_id",
    "input_archive_sha256",
    "input_row_ids_root",
    "prediction_record_ids_root",
    "protocol_id",
    "record_count",
    "run_context_id",
)
_OUTCOME_FIELD_MANIFEST: Final = (
    "input_row_id",
    "input_payload_sha256",
    "decision",
    "canonical_family_id",
    "prediction",
    "bridge_outcome_id",
    "bridge_compilation_id",
    "bridge_decision_id",
    "claim_level",
)
_DECODED_FIELD_MANIFEST: Final = (
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
    "structural_archive_verified",
    "canonical_record_framing_verified",
    "record_schema_verified",
    "row_root_coverage_verified",
    "input_archive_membership_verified",
    "execution_manifest_authority_verified",
    "derived_bridge_mapping_verified",
    "runtime_executed",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "sealed_holdout_eligible",
    "formal_covert_audit",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)
_REJECTION_FIELD_MANIFEST: Final = (
    "disposition",
    "reason",
    "input_count",
    "input_archive_id",
    "archive",
    "records",
    "prediction_record_ids",
    "recognizer_capacity_evidence",
)


class PredictionDecisionV1(str, Enum):
    ANSWER = "unique_identification"
    ANSWER_SET = "admissible_scale_set"
    ABSTAIN = "abstain"


class RecognizerPredictionArchiveDisposition(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


# This is an exact, closed conversion table.  A new derived-bridge reason must
# deliberately revise the archive policy before it can cross this boundary.
_ABSTAIN_REASON_PAIRS: Final = (
    ("no_passing_structure", PredictionReason.NO_PASSING_CANDIDATE),
    ("multiple_passing_structures", PredictionReason.MULTIPLE_STRUCTURAL_MATCHES),
    ("nonidentifiable_interval_overlap", PredictionReason.NONIDENTIFIABLE_SCALE),
    ("selected_structure_has_inconclusive_scale", PredictionReason.NONIDENTIFIABLE_SCALE),
    ("inconclusive_structural_competitor", PredictionReason.NONIDENTIFIABLE_SCALE),
    ("missing_scale_competitor", PredictionReason.NONIDENTIFIABLE_SCALE),
    ("missing_binding_competitor", PredictionReason.INSUFFICIENT_EVIDENCE),
    ("missing_structural_competitor", PredictionReason.INSUFFICIENT_EVIDENCE),
    ("bridge_explicit_support_required", PredictionReason.INSUFFICIENT_EVIDENCE),
    (
        "bridge_no_injective_role_binding",
        PredictionReason.INSUFFICIENT_EVIDENCE,
    ),
    ("insufficient_structural_margin", PredictionReason.INSUFFICIENT_MARGIN),
    (
        "bridge_incomplete_role_candidate_coverage",
        PredictionReason.INCOMPLETE_CANDIDATE_COVERAGE,
    ),
    (
        "bridge_inventory_quantity_scope_mismatch",
        PredictionReason.INCOMPLETE_CANDIDATE_COVERAGE,
    ),
    (
        "bridge_inventory_entity_scope_mismatch",
        PredictionReason.INCOMPLETE_CANDIDATE_COVERAGE,
    ),
    (
        "bridge_unused_or_ambiguously_consumed_derived_observation",
        PredictionReason.INCOMPLETE_CANDIDATE_COVERAGE,
    ),
    ("candidate_evaluation_error", PredictionReason.VERIFIER_ERROR),
    ("RESOURCE_LIMIT:selection_margin_bit_length", PredictionReason.RESOURCE_LIMIT),
    ("RESOURCE_LIMIT:raw_role_binding_product", PredictionReason.RESOURCE_LIMIT),
    (
        "RESOURCE_LIMIT:raw_role_binding_scale_product",
        PredictionReason.RESOURCE_LIMIT,
    ),
    ("RESOURCE_LIMIT:projected_candidate_count", PredictionReason.RESOURCE_LIMIT),
    ("RESOURCE_LIMIT:adapter_scan_work", PredictionReason.RESOURCE_LIMIT),
    ("bridge_RESOURCE_LIMIT:candidate_count", PredictionReason.RESOURCE_LIMIT),
    ("bridge_RESOURCE_LIMIT:match_scan_work", PredictionReason.RESOURCE_LIMIT),
    (
        "bridge_RESOURCE_LIMIT:role_binding_slice_product",
        PredictionReason.RESOURCE_LIMIT,
    ),
    (
        "bridge_RESOURCE_LIMIT:exact_operation_budget",
        PredictionReason.RESOURCE_LIMIT,
    ),
    (
        "bridge_RESOURCE_LIMIT:exact_fraction_bit_length",
        PredictionReason.RESOURCE_LIMIT,
    ),
)
_ABSTAIN_REASON_MAP: Final = dict(_ABSTAIN_REASON_PAIRS)
if len(_ABSTAIN_REASON_MAP) != len(_ABSTAIN_REASON_PAIRS):
    raise RuntimeError("prediction abstention reason table repeats a source reason")
_EXPECTED_RESOURCE_REASON_SOURCES: Final = tuple(
    sorted(
        (
            "RESOURCE_LIMIT:adapter_scan_work",
            "RESOURCE_LIMIT:projected_candidate_count",
            "RESOURCE_LIMIT:raw_role_binding_product",
            "RESOURCE_LIMIT:raw_role_binding_scale_product",
            "RESOURCE_LIMIT:selection_margin_bit_length",
            "bridge_RESOURCE_LIMIT:candidate_count",
            "bridge_RESOURCE_LIMIT:exact_fraction_bit_length",
            "bridge_RESOURCE_LIMIT:exact_operation_budget",
            "bridge_RESOURCE_LIMIT:match_scan_work",
            "bridge_RESOURCE_LIMIT:role_binding_slice_product",
        )
    )
)
if tuple(
    sorted(
        source
        for source, target in _ABSTAIN_REASON_PAIRS
        if target is PredictionReason.RESOURCE_LIMIT
    )
) != _EXPECTED_RESOURCE_REASON_SOURCES:
    raise RuntimeError("prediction resource abstention source closure drift")
_ROW_SEMANTIC_REASON_SOURCES: Final = tuple(
    sorted(
        source
        for source, target in _ABSTAIN_REASON_PAIRS
        if target is not PredictionReason.RESOURCE_LIMIT
    )
)
_INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED: Final = tuple(
    sorted(
        (
            "bridge_empty_derived_candidate_grid",
            "bridge_RESOURCE_LIMIT:aggregate_replay_work",
            "bridge_RESOURCE_LIMIT:raw_role_binding_product",
            "bridge_RESOURCE_LIMIT:slot_match_count",
            "bridge_RESOURCE_LIMIT:support_slice_count",
            "bridge_RESOURCE_LIMIT:total_slot_count",
            "RESOURCE_LIMIT:inventory_component_count",
            "RESOURCE_LIMIT:inventory_observation_count",
            "RESOURCE_LIMIT:observation_reference_width",
            "candidate_evaluation_payload_drift",
            "candidate_grid_missing",
            "duplicate_scale_aggregate",
            "explicit_support_required",
            "incomplete_candidate_grid",
            "incomplete_family_coverage",
            "scale_aggregate_key_coverage_mismatch",
            "scale_aggregate_provenance_drift",
            "scale_aggregate_slice_coverage_mismatch",
            "scale_aggregate_value_drift",
        )
    )
)
if set(_ABSTAIN_REASON_MAP) & set(_INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED):
    raise RuntimeError("prediction accepted/rejected abstention reason overlap")
_ALLOWED_ABSTAIN_REASONS: Final = frozenset(_ABSTAIN_REASON_MAP.values())
_ALLOWED_ABSTAIN_REASON_VALUES: Final = tuple(
    sorted(item.value for item in _ALLOWED_ABSTAIN_REASONS)
)
if not _ALLOWED_ABSTAIN_REASONS:
    raise RuntimeError("prediction abstention reason image is empty")

_DERIVED_BRIDGE_POLICY_BINDINGS: Final = (
    ("bridge_policy_id", EXACT_DERIVED_BRIDGE_POLICY_ID),
    ("matcher_semantics_id", EXACT_DERIVED_MATCHER_SEMANTICS_ID),
    ("selection_policy_id", EXACT_DERIVED_SELECTION_POLICY_ID),
)
_FROZEN_FAMILY_MAPPING_VALUES: Final = tuple(
    (law_kind.value, family_id.value)
    for law_kind, family_id in _FROZEN_EXACT_FREEZE.family_mapping
)
_PREDICTION_DEPENDENCY_BINDINGS: Final = (
    ("frozen_theory_version_id", _FROZEN_THEORY.version_id),
    ("exact_freeze_id", _FROZEN_EXACT_FREEZE.freeze_id),
    ("recognizer_input_archive_policy_id", RECOGNIZER_INPUT_ARCHIVE_POLICY_ID),
    ("public_recognizer_registry_schema_id", PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID),
    (
        "public_recognizer_family_alias_policy_id",
        PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID,
    ),
    ("typed_trusted_wire_replay_policy_id", TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID),
)
for _binding_name, _binding_value in (
    *_DERIVED_BRIDGE_POLICY_BINDINGS,
    *_PREDICTION_DEPENDENCY_BINDINGS,
):
    if (
        type(_binding_value) is not str
        or len(_binding_value.rsplit("_", 1)[-1]) != 64
        or any(
            item not in "0123456789abcdef"
            for item in _binding_value.rsplit("_", 1)[-1]
        )
    ):
        raise RuntimeError(f"derived bridge {_binding_name} public binding drift")


PUBLIC_RUN_CONTEXT_SCHEMA_ID: Final = stable_hash(
    {
        "context_fields": _CONTEXT_FIELDS,
        "expected_prediction_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "forbidden_public_field_tokens": _FORBIDDEN_RECOGNIZER_FIELD_TOKENS,
        "input_rows_root": "ordered_exact_input_row_ids_length_framed_sha256",
        "input_rows_shallow_preflight": (
            "before_full_archive_replay_or_hash_exact_rows_and_stored_row_ids_"
            "are_960_exact_digest_ids_equal_and_unique"
        ),
        "validation_order": (
            "shallow_input_row_identity",
            "exact_execution_freeze_manifest",
            "full_input_archive_replay",
            "context_root_hash",
        ),
        "protocol_id": _FROZEN_PROTOCOL.protocol_id,
        "root": {
            "closed_preimage_fields": _CONTEXT_PREIMAGE_FIELDS,
            "formula": "stable_hash(closed_context_fields_except_context_id)",
            "prefix": "phase2b_public_run_context_",
            "validate_before_hash": True,
        },
        "schema_version": PUBLIC_RUN_CONTEXT_SCHEMA_VERSION,
    },
    prefix="phase2b_public_run_context_schema_",
)

PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID: Final = stable_hash(
    {
        "abstain_reason_mapping": tuple(
            (source, target.value) for source, target in _ABSTAIN_REASON_PAIRS
        ),
        "abstain_reason_classification": {
            "integrity_dead_or_all_other_unlisted_rejected": (
                _INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED
            ),
            "resource": _EXPECTED_RESOURCE_REASON_SOURCES,
            "row_semantic": _ROW_SEMANTIC_REASON_SOURCES,
        },
        "allowed_abstain_reason_values": _ALLOWED_ABSTAIN_REASON_VALUES,
        "derived_bridge_policy_bindings": _DERIVED_BRIDGE_POLICY_BINDINGS,
        "decision_mapping": (
            "one_admissible_scale_is_unique_identification",
            "multiple_admissible_scales_is_admissible_scale_set",
            "both_positive_classes_use_PredictionBundle_UNIQUE_MATCH",
            "abstention_has_no_family_binding_or_scale",
        ),
        "prediction_bundle_fields": _PREDICTION_BUNDLE_FIELDS,
        "prediction_bundle_schema": PREDICTION_SCHEMA_VERSION,
        "record_fields": _RECORD_FIELDS,
        "record_preimage_fields": _RECORD_PREIMAGE_FIELDS,
        "role_binding_fields": _ROLE_BINDING_FIELDS,
        "bridge_outcome_roots": (
            "ExactDerivedBridgeRun.run_id",
            (
                "sha256(preflight_domain||accepted_jcs_exact_8_field_tuple)"
            ),
        ),
        "preflight_outcome": {
            "domain_hex": _PREFLIGHT_DOMAIN.hex(),
            "fields_in_order": _PREFLIGHT_OUTCOME_FIELDS,
            "grammar": (
                "disposition_serialized_as_exact_enum_value_then_7_exact_ascii_fields",
                "accepted_jcs_exact_tuple_no_null_placeholders",
                "all_fields_validate_before_hash",
            ),
        },
        "record_domain_hex": _RECORD_DOMAIN.hex(),
        "resource_reason_sources": _EXPECTED_RESOURCE_REASON_SOURCES,
        "schema_version": PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION,
    },
    prefix="phase2b_public_recognizer_prediction_record_schema_",
)

RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID: Final = stable_hash(
    {
        "archive_contract": (
            "manifest_then_exactly_960_independently_length_framed_accepted_jcs_rows",
            "rows_canonically_sorted_by_input_row_id",
            "input_row_and_prediction_record_roots_are_ordered_length_framed_sha256",
            "builder_replays_initial_theory_public_registry_and_exact_derived_bridge_as_gate",
            "selected_binding_is_unique_distinct_candidate_public_binding_with_double_mapping_check",
            "unknown_bridge_abstention_reason_rejects_the_whole_batch",
            "input_rows_and_stored_row_ids_are_exact_equal_unique_before_full_replay_or_hash",
            "raw_success_claims_only_byte_replayable_structure_not_bridge_execution_or_trust",
            "no_public_recognizer_field_contains_main_challenge_or_partition",
        ),
        "archive_schema_version": RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION,
        "binary": {
            "header": _ARCHIVE_HEADER.format,
            "magic_hex": PREDICTION_ARCHIVE_MAGIC.hex(),
            "wire_version": PREDICTION_ARCHIVE_WIRE_VERSION,
        },
        "caps": {
            "archive_bytes": MAXIMUM_PREDICTION_ARCHIVE_BYTES,
            "manifest_bytes": MAXIMUM_PREDICTION_MANIFEST_BYTES,
            "prediction_count": TOTAL_RECOGNIZER_CASE_COUNT,
            "record_bytes": MAXIMUM_PREDICTION_RECORD_BYTES,
        },
        "context_schema_id": PUBLIC_RUN_CONTEXT_SCHEMA_ID,
        "dependency_bindings": _PREDICTION_DEPENDENCY_BINDINGS,
        "derived_bridge_policy_bindings": _DERIVED_BRIDGE_POLICY_BINDINGS,
        "exact_freeze_family_mapping": _FROZEN_FAMILY_MAPPING_VALUES,
        "manifest_fields": _MANIFEST_FIELDS,
        "record_schema_id": PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID,
        "public_receipt_fields": {
            "decoded": _DECODED_FIELD_MANIFEST,
            "outcome": _OUTCOME_FIELD_MANIFEST,
            "rejection": _REJECTION_FIELD_MANIFEST,
        },
        "public_claims": {
            "raw_true": (
                "structural_archive_verified",
                "canonical_record_framing_verified",
                "record_schema_verified",
                "row_root_coverage_verified",
            ),
            "raw_false": (
                "input_archive_membership_verified",
                "execution_manifest_authority_verified",
                "derived_bridge_mapping_verified",
                "runtime_executed",
                "recognizer_capacity_evidence",
                "origin_authenticated",
                "sealed_holdout_eligible",
                "formal_covert_audit",
                "prediction_scored",
                "effect_evidence",
                "c1_exit_evidence",
            ),
            "rejection_false": ("recognizer_capacity_evidence",),
            "bridge_ids_are_unverified_content_commitments_not_receipts": True,
            "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
            "exact_bool": True,
        },
        "validated_parsed_context_reuse_is_private_token_gated": True,
        "root_domains_hex": {
            "archive": _ARCHIVE_DOMAIN.hex(),
            "input_rows": _INPUT_ROWS_DOMAIN.hex(),
            "preflight": _PREFLIGHT_DOMAIN.hex(),
            "record": _RECORD_DOMAIN.hex(),
            "record_ids": _RECORD_IDS_DOMAIN.hex(),
        },
        "root_preimages": {
            "archive": "domain||all_validated_archive_bytes",
            "input_rows": (
                "domain||u32_count||repeated(u16_ascii_length||exact_input_row_id)"
            ),
            "preflight": (
                "domain||accepted_jcs_tuple_in_preflight_fields_order_without_null_placeholders"
            ),
            "record": (
                "domain||accepted_jcs_closed_record_fields_except_record_id"
            ),
            "record_ids": (
                "domain||u32_count||repeated(u16_ascii_length||exact_record_id)"
            ),
            "validate_before_hash": True,
        },
        "preflight_outcome_fields_in_order": _PREFLIGHT_OUTCOME_FIELDS,
    },
    prefix="phase2b_recognizer_prediction_archive_policy_",
)


for _name, _manifest in (
    ("context", _CONTEXT_FIELDS),
    ("manifest", _MANIFEST_FIELDS),
    ("prediction", _PREDICTION_BUNDLE_FIELDS),
    ("record", _RECORD_FIELDS),
    ("role_binding", _ROLE_BINDING_FIELDS),
):
    if _manifest != tuple(sorted(_manifest)):
        raise RuntimeError(f"prediction {_name} field manifest is not canonical")
    if any(
        token in field.casefold()
        for field in _manifest
        for token in _FORBIDDEN_RECOGNIZER_FIELD_TOKENS
    ):
        raise RuntimeError(f"prediction {_name} field leaks a split label")
if tuple(
    item.name
    for item in fields(ExactDerivedBridgePreflightRejection)[:8]
) != _PREFLIGHT_OUTCOME_FIELDS:
    raise RuntimeError("derived preflight outcome field manifest drift")


_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _ascii(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > MAXIMUM_ASCII_STRING_BYTES
        or not value.isascii()
    ):
        raise ValueError(f"{name} must be exact bounded nonempty ASCII")
    return value


def _sha256_text(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError(f"{name} must use prefix {prefix}")
    _sha256_text(value[len(prefix) :], name)
    return value


def _safe_digest_or_none(value: object, prefix: str, name: str) -> str | None:
    try:
        return _digest(value, prefix, name)
    except (TypeError, ValueError):
        return None


class _PredictionInputShallowError(ValueError):
    def __init__(self, reason: str, *, input_count: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.input_count = input_count


def _shallow_prediction_input_rows(
    input_archive: DecodedRecognizerInputArchiveV1,
) -> tuple[tuple[TrustedRecognizerInputRowV1, ...], tuple[str, ...]]:
    """Validate the 960 public row identities before replay or hashing."""

    if type(input_archive) is not DecodedRecognizerInputArchiveV1:
        raise TypeError("prediction input shallow check needs exact archive")
    try:
        rows = input_archive.rows
        stored_row_ids = input_archive.row_ids
    except AttributeError as exc:
        raise _PredictionInputShallowError(
            "input_archive_shallow_invalid"
        ) from exc
    if type(rows) is not tuple:
        raise _PredictionInputShallowError("input_rows_type_invalid")
    input_count = len(rows)
    if input_count != TOTAL_RECOGNIZER_CASE_COUNT:
        raise _PredictionInputShallowError(
            "prediction_count_drift",
            input_count=input_count,
        )
    if type(stored_row_ids) is not tuple or len(stored_row_ids) != input_count:
        raise _PredictionInputShallowError(
            "input_row_ids_shallow_drift",
            input_count=input_count,
        )
    actual_row_ids: list[str] = []
    for row in rows:
        if type(row) is not TrustedRecognizerInputRowV1:
            raise _PredictionInputShallowError(
                "input_row_type_invalid",
                input_count=input_count,
            )
        try:
            row_id = row.row_id
        except AttributeError as exc:
            raise _PredictionInputShallowError(
                "input_row_id_shallow_invalid",
                input_count=input_count,
            ) from exc
        try:
            _digest(
                row_id,
                "phase2b_recognizer_input_row_",
                "prediction shallow input row ID",
            )
        except (TypeError, ValueError) as exc:
            raise _PredictionInputShallowError(
                "input_row_id_shallow_invalid",
                input_count=input_count,
            ) from exc
        actual_row_ids.append(row_id)
    for row_id in stored_row_ids:
        try:
            _digest(
                row_id,
                "phase2b_recognizer_input_row_",
                "prediction shallow stored row ID",
            )
        except (TypeError, ValueError) as exc:
            raise _PredictionInputShallowError(
                "stored_row_id_shallow_invalid",
                input_count=input_count,
            ) from exc
    frozen_row_ids = tuple(actual_row_ids)
    if stored_row_ids != frozen_row_ids:
        raise _PredictionInputShallowError(
            "input_row_ids_shallow_drift",
            input_count=input_count,
        )
    if len(set(frozen_row_ids)) != TOTAL_RECOGNIZER_CASE_COUNT:
        raise _PredictionInputShallowError(
            "duplicate_input_row_id",
            input_count=input_count,
        )
    return rows, frozen_row_ids


def _uuid4(value: object, name: str) -> str:
    if type(value) is not str or _UUID4.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact canonical lowercase UUIDv4")
    if UUID(value).version != 4:
        raise ValueError(f"{name} must be a UUIDv4")
    return value


def _closed_mapping(
    value: object,
    manifest: tuple[str, ...],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must use an exact mapping")
    if len(value) != len(manifest):
        raise ValueError(f"{name} field count drift")
    if tuple(sorted(value)) != manifest:
        raise ValueError(f"{name} closed schema drift")
    return value


def _sequence_root(
    values: tuple[str, ...],
    *,
    expected_count: int,
    item_prefix: str,
    domain: bytes,
    output_prefix: str,
    name: str,
) -> str:
    if type(values) is not tuple or len(values) != expected_count:
        raise ValueError(f"{name} count drift")
    encoded_values: list[bytes] = []
    for value in values:
        _digest(value, item_prefix, name)
        encoded = value.encode("ascii")
        if len(encoded) > 65_535:
            raise ValueError(f"{name} exceeds u16 framing")
        encoded_values.append(encoded)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(expected_count.to_bytes(4, "big"))
    for encoded in encoded_values:
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return output_prefix + digest.hexdigest()


def _input_row_ids_root(values: tuple[str, ...]) -> str:
    return _sequence_root(
        values,
        expected_count=TOTAL_RECOGNIZER_CASE_COUNT,
        item_prefix="phase2b_recognizer_input_row_",
        domain=_INPUT_ROWS_DOMAIN,
        output_prefix="phase2b_prediction_input_rows_",
        name="prediction input row ID",
    )


def _prediction_record_ids_root(values: tuple[str, ...]) -> str:
    return _sequence_root(
        values,
        expected_count=TOTAL_RECOGNIZER_CASE_COUNT,
        item_prefix="phase2b_recognizer_prediction_record_",
        domain=_RECORD_IDS_DOMAIN,
        output_prefix="phase2b_prediction_records_",
        name="prediction record ID",
    )


def _context_mapping(value: "PublicRunContextV1") -> dict[str, object]:
    mapping = _context_mapping_without_id(value)
    mapping["context_id"] = value.context_id
    return mapping


def _context_mapping_without_id(value: "PublicRunContextV1") -> dict[str, object]:
    return {
        "claim_level": value.claim_level,
        "execution_freeze_manifest_id": value.execution_freeze_manifest_id,
        "expected_prediction_count": value.expected_prediction_count,
        "input_archive_id": value.input_archive_id,
        "input_archive_sha256": value.input_archive_sha256,
        "input_row_ids_root": value.input_row_ids_root,
        "protocol_id": value.protocol_id,
        "schema_version": value.schema_version,
    }


def _validate_context_preimage(mapping_without_id: object) -> dict[str, object]:
    value = _closed_mapping(
        mapping_without_id,
        _CONTEXT_PREIMAGE_FIELDS,
        "public run context preimage",
    )
    if (
        type(value["schema_version"]) is not str
        or value["schema_version"] != PUBLIC_RUN_CONTEXT_SCHEMA_VERSION
    ):
        raise ValueError("public run context preimage schema drift")
    _ascii(value["schema_version"], "public run context preimage schema")
    _digest(
        value["input_archive_id"],
        "phase2b_recognizer_input_archive_",
        "public run context preimage input archive ID",
    )
    _sha256_text(
        value["input_archive_sha256"],
        "public run context preimage input SHA-256",
    )
    _digest(
        value["input_row_ids_root"],
        "phase2b_prediction_input_rows_",
        "public run context preimage input-row root",
    )
    _digest(
        value["protocol_id"],
        "phase2b_protocol_",
        "public run context preimage protocol",
    )
    if value["protocol_id"] != _FROZEN_PROTOCOL.protocol_id:
        raise ValueError("public run context preimage protocol drift")
    _digest(
        value["execution_freeze_manifest_id"],
        "phase2b_execution_freeze_",
        "public run context preimage execution freeze",
    )
    if (
        type(value["expected_prediction_count"]) is not int
        or value["expected_prediction_count"] != TOTAL_RECOGNIZER_CASE_COUNT
    ):
        raise ValueError("public run context preimage count drift")
    if (
        type(value["claim_level"]) is not str
        or value["claim_level"] != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("public run context preimage claim drift")
    return value


def _context_id(mapping_without_id: dict[str, object]) -> str:
    _validate_context_preimage(mapping_without_id)
    return stable_hash(mapping_without_id, prefix="phase2b_public_run_context_")


@dataclass(frozen=True, slots=True, init=False)
class PublicRunContextV1:
    schema_version: str
    input_archive_id: str
    input_archive_sha256: str
    input_row_ids_root: str
    protocol_id: str
    execution_freeze_manifest_id: str
    expected_prediction_count: int
    claim_level: str
    context_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("public run contexts are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        input_archive_id: str,
        input_archive_sha256: str,
        input_row_ids_root: str,
        protocol_id: str,
        execution_freeze_manifest_id: str,
    ) -> "PublicRunContextV1":
        if token is not _CONTEXT_ISSUE_TOKEN:
            raise TypeError("public run context issuer token mismatch")
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", PUBLIC_RUN_CONTEXT_SCHEMA_VERSION),
            ("input_archive_id", input_archive_id),
            ("input_archive_sha256", input_archive_sha256),
            ("input_row_ids_root", input_row_ids_root),
            ("protocol_id", protocol_id),
            ("execution_freeze_manifest_id", execution_freeze_manifest_id),
            ("expected_prediction_count", TOTAL_RECOGNIZER_CASE_COUNT),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
        ):
            object.__setattr__(value, name, item)
        body = _context_mapping_without_id(value)
        object.__setattr__(value, "context_id", _context_id(body))
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRunContextV1:
            raise TypeError("public run context must use the exact type")
        if self.schema_version != PUBLIC_RUN_CONTEXT_SCHEMA_VERSION:
            raise ValueError("public run context schema drift")
        _ascii(self.schema_version, "public run context schema")
        _digest(
            self.input_archive_id,
            "phase2b_recognizer_input_archive_",
            "public run context input archive ID",
        )
        _sha256_text(self.input_archive_sha256, "public run context input SHA-256")
        _digest(
            self.input_row_ids_root,
            "phase2b_prediction_input_rows_",
            "public run context input-row root",
        )
        _digest(self.protocol_id, "phase2b_protocol_", "public run context protocol")
        if self.protocol_id != _FROZEN_PROTOCOL.protocol_id:
            raise ValueError("public run context protocol drift")
        _digest(
            self.execution_freeze_manifest_id,
            "phase2b_execution_freeze_",
            "public run context execution freeze",
        )
        if (
            type(self.expected_prediction_count) is not int
            or self.expected_prediction_count != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            raise ValueError("public run context expected count drift")
        if (
            type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("public run context claim level drift")
        _digest(self.context_id, "phase2b_public_run_context_", "public run context ID")
        if self.context_id != _context_id(_context_mapping_without_id(self)):
            raise ValueError("public run context root drift")


def _decode_context(mapping: object) -> PublicRunContextV1:
    value = _closed_mapping(mapping, _CONTEXT_FIELDS, "public run context")
    context = PublicRunContextV1._issue(
        _CONTEXT_ISSUE_TOKEN,
        input_archive_id=value["input_archive_id"],  # type: ignore[arg-type]
        input_archive_sha256=value["input_archive_sha256"],  # type: ignore[arg-type]
        input_row_ids_root=value["input_row_ids_root"],  # type: ignore[arg-type]
        protocol_id=value["protocol_id"],  # type: ignore[arg-type]
        execution_freeze_manifest_id=value["execution_freeze_manifest_id"],  # type: ignore[arg-type]
    )
    if _context_mapping(context) != value:
        raise ValueError("public run context canonical roundtrip drift")
    return context


def _validate_execution_freeze_manifest(
    manifest: ExecutionFreezeManifest,
) -> None:
    """Validate exact builtins before evaluating the manifest content root."""

    if type(manifest) is not ExecutionFreezeManifest:
        raise TypeError("prediction mechanics require exact ExecutionFreezeManifest")
    for name in (
        "protocol_id",
        "exact_freeze_id",
        "git_commit",
        "recognizer_image_digest",
        "configuration_sha256",
        "theory_version_id",
        "adapter_implementation_sha256",
        "selector_implementation_sha256",
        "verifier_registry_sha256",
        "isolation_profile_id",
    ):
        _ascii(getattr(manifest, name), f"execution manifest {name}")
    if (
        type(manifest.baseline_registrations) is not tuple
        or len(manifest.baseline_registrations) != len(BaselineKind)
    ):
        raise TypeError("execution manifest baseline registrations are invalid")
    for registration in manifest.baseline_registrations:
        if type(registration) is not BaselineRegistration:
            raise TypeError("execution manifest baseline must use exact type")
        if type(registration.kind) is not BaselineKind:
            raise TypeError("execution manifest baseline kind must use exact enum")
        _ascii(registration.baseline_spec_id, "baseline specification ID")
        _ascii(registration.implementation_id, "baseline implementation ID")
        _sha256_text(registration.artifact_sha256, "baseline artifact SHA-256")
        if type(registration.frozen_before_holdout_generation) is not bool:
            raise TypeError("baseline frozen flag must use exact bool")
        registration.__post_init__()
    manifest.__post_init__()
    if (
        manifest.protocol_id != _FROZEN_PROTOCOL.protocol_id
        or manifest.exact_freeze_id != _FROZEN_EXACT_FREEZE.freeze_id
        or manifest.theory_version_id != _FROZEN_THEORY.version_id
    ):
        raise ValueError("execution manifest does not bind current frozen authorities")
    _digest(
        manifest.manifest_id,
        "phase2b_execution_freeze_",
        "execution freeze manifest ID",
    )


def build_public_run_context_v1(
    *,
    input_archive: DecodedRecognizerInputArchiveV1,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> PublicRunContextV1:
    """Bind exact public input bytes and current frozen execution identities."""

    if type(input_archive) is not DecodedRecognizerInputArchiveV1:
        raise TypeError("public run context requires exact decoded input archive")
    if type(execution_freeze_manifest) is not ExecutionFreezeManifest:
        raise TypeError("public run context requires exact execution freeze manifest")
    _, row_ids = _shallow_prediction_input_rows(input_archive)
    _validate_execution_freeze_manifest(execution_freeze_manifest)
    input_archive._validate()
    return PublicRunContextV1._issue(
        _CONTEXT_ISSUE_TOKEN,
        input_archive_id=input_archive.archive_id,
        input_archive_sha256=hashlib.sha256(input_archive.archive).hexdigest(),
        input_row_ids_root=_input_row_ids_root(tuple(sorted(row_ids))),
        protocol_id=_FROZEN_PROTOCOL.protocol_id,
        execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
    )


def _prediction_mapping_unchecked(value: PredictionBundle) -> dict[str, object]:
    mapping = value.to_mapping()
    if type(mapping) is not dict:
        raise TypeError("PredictionBundle mapping must use exact dict")
    return mapping


def _validate_prediction_bundle_object(value: PredictionBundle) -> None:
    if type(value) is not PredictionBundle:
        raise TypeError("prediction record requires exact PredictionBundle")
    if type(value.schema_version) is not str:
        raise TypeError("prediction schema version must use exact text")
    _uuid4(value.bundle_id, "prediction bundle ID")
    for name in (
        "input_root_sha256",
        "protocol_sha256",
        "freeze_manifest_sha256",
    ):
        _sha256_text(getattr(value, name), f"prediction {name}")
    if type(value.disposition) is not PredictionDisposition:
        raise TypeError("prediction disposition must use exact enum")
    if type(value.reason) is not PredictionReason:
        raise TypeError("prediction reason must use exact enum")
    if value.family_id is not None:
        _uuid4(value.family_id, "prediction family ID")
    if type(value.binding) is not tuple or len(value.binding) > 64:
        raise TypeError("prediction binding must use exact bounded tuple")
    for item in value.binding:
        if type(item) is not RoleBinding:
            raise TypeError("prediction binding item must use exact RoleBinding")
        _uuid4(item.role_id, "prediction binding role ID")
        _uuid4(item.entity_id, "prediction binding entity ID")
    if (
        type(value.admissible_scale_ids) is not tuple
        or len(value.admissible_scale_ids) > 4_096
    ):
        raise TypeError("prediction scales must use exact bounded tuple")
    for item in value.admissible_scale_ids:
        _uuid4(item, "prediction scale ID")


def _decode_prediction_bundle(mapping: object) -> PredictionBundle:
    value = _closed_mapping(
        mapping,
        _PREDICTION_BUNDLE_FIELDS,
        "public prediction bundle",
    )
    if type(value["schema_version"]) is not str:
        raise TypeError("prediction schema version must use exact text")
    if value["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError("prediction schema version drift")
    _uuid4(value["bundle_id"], "prediction bundle ID")
    for field_name in (
        "input_root_sha256",
        "protocol_sha256",
        "freeze_manifest_sha256",
    ):
        _sha256_text(value[field_name], f"prediction {field_name}")
    if type(value["disposition"]) is not str or value["disposition"] not in {
        item.value for item in PredictionDisposition
    }:
        raise ValueError("prediction disposition is unknown")
    if type(value["reason"]) is not str or value["reason"] not in {
        item.value for item in PredictionReason
    }:
        raise ValueError("prediction reason is unknown")
    if value["family_id"] is not None:
        _uuid4(value["family_id"], "prediction family ID")
    if type(value["binding"]) is not list or len(value["binding"]) > 64:
        raise TypeError("prediction binding must use one bounded exact array")
    for index, binding in enumerate(value["binding"]):
        item = _closed_mapping(
            binding,
            _ROLE_BINDING_FIELDS,
            f"prediction binding {index}",
        )
        _uuid4(item["role_id"], "prediction binding role ID")
        _uuid4(item["entity_id"], "prediction binding entity ID")
    if (
        type(value["admissible_scale_ids"]) is not list
        or len(value["admissible_scale_ids"]) > 4_096
    ):
        raise TypeError("prediction scales must use one bounded exact array")
    for scale_id in value["admissible_scale_ids"]:
        _uuid4(scale_id, "prediction scale ID")
    prediction = PredictionBundle.from_mapping(value)
    if type(prediction) is not PredictionBundle:
        raise TypeError("prediction decoder returned a nonexact bundle")
    if _prediction_mapping_unchecked(prediction) != value:
        raise ValueError("prediction bundle canonical roundtrip drift")
    return prediction


def _prediction_mapping(value: PredictionBundle) -> dict[str, object]:
    _validate_prediction_bundle_object(value)
    mapping = _prediction_mapping_unchecked(value)
    if _decode_prediction_bundle(mapping) != value:
        raise ValueError("prediction bundle object is structurally polluted")
    return mapping


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerPredictionOutcomeV1:
    """Ephemeral in-process row outcome; it has no durable trust receipt ID."""

    input_row_id: str
    input_payload_sha256: str
    decision: PredictionDecisionV1
    canonical_family_id: CanonicalFamilyId | None
    prediction: PredictionBundle
    bridge_outcome_id: str
    bridge_compilation_id: str | None
    bridge_decision_id: str | None
    claim_level: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("public prediction outcomes are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        input_row_id: str,
        input_payload_sha256: str,
        decision: PredictionDecisionV1,
        canonical_family_id: CanonicalFamilyId | None,
        prediction: PredictionBundle,
        bridge_outcome_id: str,
        bridge_compilation_id: str | None,
        bridge_decision_id: str | None,
    ) -> "PublicRecognizerPredictionOutcomeV1":
        if token is not _RECORD_ISSUE_TOKEN:
            raise TypeError("public prediction outcome issuer token mismatch")
        value = object.__new__(cls)
        for name, item in (
            ("input_row_id", input_row_id),
            ("input_payload_sha256", input_payload_sha256),
            ("decision", decision),
            ("canonical_family_id", canonical_family_id),
            ("prediction", prediction),
            ("bridge_outcome_id", bridge_outcome_id),
            ("bridge_compilation_id", bridge_compilation_id),
            ("bridge_decision_id", bridge_decision_id),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
        ):
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerPredictionOutcomeV1:
            raise TypeError("public prediction outcome must use exact type")
        _digest(
            self.input_row_id,
            "phase2b_recognizer_input_row_",
            "prediction outcome input row ID",
        )
        _sha256_text(
            self.input_payload_sha256,
            "prediction outcome input payload SHA-256",
        )
        if type(self.decision) is not PredictionDecisionV1:
            raise TypeError("prediction outcome decision must use exact enum")
        prediction = _decode_prediction_bundle(_prediction_mapping(self.prediction))
        if prediction.input_root_sha256 != self.input_payload_sha256:
            raise ValueError("prediction outcome input root drift")
        _validate_family_and_decision(
            decision=self.decision,
            canonical_family_id=self.canonical_family_id,
            prediction=prediction,
        )
        _bridge_outcome_digest(self.bridge_outcome_id)
        if (self.bridge_compilation_id is None) != (self.bridge_decision_id is None):
            raise ValueError("prediction outcome bridge roots are partial")
        if self.bridge_compilation_id is not None:
            _digest(
                self.bridge_compilation_id,
                "phase2b_exact_derived_bridge_result_",
                "prediction bridge compilation ID",
            )
            _digest(
                self.bridge_decision_id,
                "phase2b_exact_derived_decision_",
                "prediction bridge decision ID",
            )
        if (
            type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("prediction outcome claim level drift")


def _bridge_outcome_digest(value: object) -> str:
    if type(value) is not str:
        raise ValueError("prediction bridge outcome ID must use exact text")
    for prefix in (
        "phase2b_exact_derived_run_",
        "phase2b_exact_derived_preflight_",
    ):
        if value.startswith(prefix):
            return _digest(value, prefix, "prediction bridge outcome ID")
    raise ValueError("prediction bridge outcome ID prefix drift")


def _preflight_outcome_id(
    value: ExactDerivedBridgePreflightRejection,
) -> str:
    if type(value) is not ExactDerivedBridgePreflightRejection:
        raise TypeError("preflight outcome root requires exact rejection type")
    value.__post_init__()
    if value.disposition is not ExactBridgeDisposition.ABSTAIN:
        raise ValueError("preflight outcome root requires abstention")
    raw_items = tuple(getattr(value, name) for name in _PREFLIGHT_OUTCOME_FIELDS)
    items = (value.disposition.value, *raw_items[1:])
    for index, item in enumerate(items):
        _ascii(item, f"preflight outcome field {index}")
    payload = encode_phase2b_jcs_profile_v1(items)
    return "phase2b_exact_derived_preflight_" + hashlib.sha256(
        _PREFLIGHT_DOMAIN + payload
    ).hexdigest()


def _validate_family_and_decision(
    *,
    decision: PredictionDecisionV1,
    canonical_family_id: CanonicalFamilyId | None,
    prediction: PredictionBundle,
) -> None:
    if decision is PredictionDecisionV1.ABSTAIN:
        if (
            canonical_family_id is not None
            or prediction.disposition is not PredictionDisposition.ABSTAIN
            or type(prediction.reason) is not PredictionReason
            or prediction.reason not in _ALLOWED_ABSTAIN_REASONS
        ):
            raise ValueError(
                "abstaining prediction carries a family or an unissued reason"
            )
        return
    if type(canonical_family_id) is not CanonicalFamilyId:
        raise TypeError("answering prediction needs exact canonical family enum")
    law_kind = _KIND_BY_CANONICAL_FAMILY.get(canonical_family_id)
    if law_kind is None or prediction.family_id != _BRIDGE_FAMILY_BY_KIND[law_kind]:
        raise ValueError("prediction bridge/canonical family mapping drift")
    if (
        prediction.disposition is not PredictionDisposition.UNIQUE_MATCH
        or prediction.reason is not PredictionReason.UNIQUE_STRUCTURAL_MATCH
    ):
        raise ValueError("answering prediction bundle disposition drift")
    scale_count = len(prediction.admissible_scale_ids)
    if decision is PredictionDecisionV1.ANSWER and scale_count != 1:
        raise ValueError("ANSWER requires exactly one admissible scale")
    if decision is PredictionDecisionV1.ANSWER_SET and scale_count <= 1:
        raise ValueError("ANSWER_SET requires multiple admissible scales")


class _PredictionGateRejected(ValueError):
    pass


def _mapped_abstention_reason(source_reason: object) -> PredictionReason:
    if type(source_reason) is not str:
        raise _PredictionGateRejected("bridge_reason_type_invalid")
    mapped = _ABSTAIN_REASON_MAP.get(source_reason)
    if mapped is None:
        raise _PredictionGateRejected("unknown_bridge_abstention_reason")
    return mapped


def _prediction_bundle(
    *,
    input_row: TrustedRecognizerInputRowV1,
    typed: TypedTrustedEnvelopeReplayV1,
    execution_freeze_manifest_id: str,
    disposition: PredictionDisposition,
    reason: PredictionReason,
    family_id: str | None,
    binding: tuple[RoleBinding, ...],
    admissible_scale_ids: tuple[str, ...],
) -> PredictionBundle:
    prediction = PredictionBundle(
        schema_version=PREDICTION_SCHEMA_VERSION,
        bundle_id=typed.authority.base_bundle.bundle_id,
        input_root_sha256=input_row.payload_sha256,
        protocol_sha256=_FROZEN_PROTOCOL.protocol_id.rsplit("_", 1)[1],
        freeze_manifest_sha256=execution_freeze_manifest_id.rsplit("_", 1)[1],
        disposition=disposition,
        reason=reason,
        family_id=family_id,
        binding=binding,
        admissible_scale_ids=admissible_scale_ids,
    )
    return _decode_prediction_bundle(_prediction_mapping(prediction))


def _selected_public_binding(
    *,
    bridge_run: ExactDerivedBridgeRun,
    public_registry: PublicRecognizerRegistryV1,
) -> tuple[LawKind, CanonicalFamilyId, str, tuple[RoleBinding, ...], tuple[str, ...]]:
    decision = bridge_run.decision
    if decision.disposition not in {
        ExactSelectionDisposition.UNIQUE_IDENTIFICATION,
        ExactSelectionDisposition.ADMISSIBLE_SCALE_SET,
    }:
        raise _PredictionGateRejected("derived_decision_is_not_positive")
    law_kind = decision.selected_law_kind
    if type(law_kind) is not LawKind:
        raise _PredictionGateRejected("selected_law_kind_type_invalid")
    selected_laws = tuple(
        item for item in public_registry.law_bindings if item.law_kind is law_kind
    )
    if len(selected_laws) != 1:
        raise _PredictionGateRejected("selected_public_law_not_unique")
    selected_law = selected_laws[0]
    if (
        selected_law.bridge_family_id != _BRIDGE_FAMILY_BY_KIND[law_kind]
        or selected_law.canonical_family_id
        is not _CANONICAL_FAMILY_BY_KIND[law_kind]
    ):
        raise _PredictionGateRejected("selected_public_family_mapping_drift")
    if (
        type(decision.selected_role_binding) is not tuple
        or not decision.selected_role_binding
        or decision.selected_role_binding
        != tuple(sorted(decision.selected_role_binding))
    ):
        raise _PredictionGateRejected("selected_semantic_binding_invalid")
    semantic_to_entity: dict[str, str] = {}
    for item in decision.selected_role_binding:
        if type(item) is not tuple or len(item) != 2:
            raise _PredictionGateRejected("selected_semantic_binding_invalid")
        semantic_role, entity_id = item
        if type(semantic_role) is not str or type(entity_id) is not str:
            raise _PredictionGateRejected("selected_semantic_binding_invalid")
        if semantic_role in semantic_to_entity:
            raise _PredictionGateRejected("selected_semantic_binding_repeats_role")
        semantic_to_entity[semantic_role] = entity_id
    semantic_to_wire = dict(selected_law.role_ids)
    if (
        len(semantic_to_wire) != len(selected_law.role_ids)
        or set(semantic_to_entity) != set(semantic_to_wire)
    ):
        raise _PredictionGateRejected("selected_semantic_role_scope_drift")
    expected_public_binding = tuple(
        sorted(
            (
                RoleBinding(semantic_to_wire[semantic_role], entity_id)
                for semantic_role, entity_id in semantic_to_entity.items()
            ),
            key=lambda item: item.role_id,
        )
    )
    grid = bridge_run.compilation.candidate_grid
    if grid is None:
        raise _PredictionGateRejected("selected_candidate_grid_missing")
    matching_candidates = tuple(
        item
        for item in grid.candidates
        if item.law_kind is law_kind
        and item.role_binding == decision.selected_role_binding
    )
    if not matching_candidates:
        raise _PredictionGateRejected("selected_candidate_binding_missing")
    distinct_public_bindings = {item.public_binding for item in matching_candidates}
    if len(distinct_public_bindings) != 1:
        raise _PredictionGateRejected("selected_public_binding_not_unique")
    public_binding = next(iter(distinct_public_bindings))
    if public_binding != expected_public_binding:
        raise _PredictionGateRejected("selected_public_binding_mapping_drift")
    if any(
        item.family_id != selected_law.bridge_family_id
        or item.law_id != selected_law.law_id
        or item.public_binding != expected_public_binding
        or dict(item.role_binding) != semantic_to_entity
        for item in matching_candidates
    ):
        raise _PredictionGateRejected("selected_candidate_family_or_binding_drift")
    scales = decision.admissible_scale_ids
    if (
        type(scales) is not tuple
        or not scales
        or scales != tuple(sorted(set(scales)))
    ):
        raise _PredictionGateRejected("selected_admissible_scales_invalid")
    expected_disposition = (
        ExactSelectionDisposition.UNIQUE_IDENTIFICATION
        if len(scales) == 1
        else ExactSelectionDisposition.ADMISSIBLE_SCALE_SET
    )
    if decision.disposition is not expected_disposition:
        raise _PredictionGateRejected("selected_scale_cardinality_disposition_drift")
    passing_scales = tuple(
        sorted(
            item.scale_id
            for item in bridge_run.compilation.scale_aggregates
            if item.law_kind is law_kind
            and item.role_binding == decision.selected_role_binding
            and item.status is ExactCandidateStatus.PASS
        )
    )
    if passing_scales != scales:
        raise _PredictionGateRejected("selected_scale_aggregate_pass_set_drift")
    return (
        law_kind,
        selected_law.canonical_family_id,
        selected_law.bridge_family_id,
        public_binding,
        scales,
    )


def _compile_prediction_outcome_from_bridge(
    *,
    input_row: TrustedRecognizerInputRowV1,
    typed: TypedTrustedEnvelopeReplayV1,
    execution_freeze_manifest_id: str,
    bridge_result: ExactDerivedBridgeRun | ExactDerivedBridgePreflightRejection,
) -> PublicRecognizerPredictionOutcomeV1:
    if type(input_row) is not TrustedRecognizerInputRowV1:
        raise TypeError("prediction mapping needs exact input row")
    if type(typed) is not TypedTrustedEnvelopeReplayV1:
        raise TypeError("prediction mapping needs exact typed replay")
    if typed.envelope != input_row.envelope:
        raise _PredictionGateRejected("typed_row_envelope_drift")
    _digest(
        execution_freeze_manifest_id,
        "phase2b_execution_freeze_",
        "prediction execution freeze manifest ID",
    )
    if type(bridge_result) is ExactDerivedBridgePreflightRejection:
        if (
            bridge_result.bridge_policy_id != EXACT_DERIVED_BRIDGE_POLICY_ID
            or bridge_result.matcher_semantics_id
            != EXACT_DERIVED_MATCHER_SEMANTICS_ID
        ):
            raise _PredictionGateRejected("derived_preflight_policy_binding_drift")
        reason = _mapped_abstention_reason(bridge_result.reason)
        prediction = _prediction_bundle(
            input_row=input_row,
            typed=typed,
            execution_freeze_manifest_id=execution_freeze_manifest_id,
            disposition=PredictionDisposition.ABSTAIN,
            reason=reason,
            family_id=None,
            binding=(),
            admissible_scale_ids=(),
        )
        return PublicRecognizerPredictionOutcomeV1._issue(
            _RECORD_ISSUE_TOKEN,
            input_row_id=input_row.row_id,
            input_payload_sha256=input_row.payload_sha256,
            decision=PredictionDecisionV1.ABSTAIN,
            canonical_family_id=None,
            prediction=prediction,
            bridge_outcome_id=_preflight_outcome_id(bridge_result),
            bridge_compilation_id=None,
            bridge_decision_id=None,
        )
    if type(bridge_result) is not ExactDerivedBridgeRun:
        raise _PredictionGateRejected("derived_bridge_result_type_invalid")
    bridge_result.__post_init__()
    if (
        bridge_result.bridge_policy_id != EXACT_DERIVED_BRIDGE_POLICY_ID
        or bridge_result.matcher_semantics_id
        != EXACT_DERIVED_MATCHER_SEMANTICS_ID
        or bridge_result.decision.selection_policy_id
        != EXACT_DERIVED_SELECTION_POLICY_ID
    ):
        raise _PredictionGateRejected("derived_bridge_policy_binding_drift")
    adapter = input_row.public_registry.to_adapter_registry()
    if (
        bridge_result.wrapper_content_id != typed.authority_content_id
        or bridge_result.transform_result_id != input_row.transform_result_id
        or bridge_result.theory_version_id != _FROZEN_THEORY.version_id
        or bridge_result.registry_id != adapter.registry_id
    ):
        raise _PredictionGateRejected("derived_bridge_input_root_drift")
    bridge_outcome_id = bridge_result.run_id
    compilation_id = bridge_result.compilation.result_id
    decision_id = bridge_result.decision.decision_id
    if bridge_result.decision.disposition is ExactSelectionDisposition.ABSTAIN:
        reason = _mapped_abstention_reason(bridge_result.decision.reason)
        prediction = _prediction_bundle(
            input_row=input_row,
            typed=typed,
            execution_freeze_manifest_id=execution_freeze_manifest_id,
            disposition=PredictionDisposition.ABSTAIN,
            reason=reason,
            family_id=None,
            binding=(),
            admissible_scale_ids=(),
        )
        return PublicRecognizerPredictionOutcomeV1._issue(
            _RECORD_ISSUE_TOKEN,
            input_row_id=input_row.row_id,
            input_payload_sha256=input_row.payload_sha256,
            decision=PredictionDecisionV1.ABSTAIN,
            canonical_family_id=None,
            prediction=prediction,
            bridge_outcome_id=bridge_outcome_id,
            bridge_compilation_id=compilation_id,
            bridge_decision_id=decision_id,
        )
    if bridge_result.disposition is not ExactBridgeDisposition.COMPLETE:
        raise _PredictionGateRejected("positive_decision_from_incomplete_bridge")
    _, canonical_family_id, bridge_family_id, binding, scales = (
        _selected_public_binding(
            bridge_run=bridge_result,
            public_registry=input_row.public_registry,
        )
    )
    decision = (
        PredictionDecisionV1.ANSWER
        if len(scales) == 1
        else PredictionDecisionV1.ANSWER_SET
    )
    prediction = _prediction_bundle(
        input_row=input_row,
        typed=typed,
        execution_freeze_manifest_id=execution_freeze_manifest_id,
        disposition=PredictionDisposition.UNIQUE_MATCH,
        reason=PredictionReason.UNIQUE_STRUCTURAL_MATCH,
        family_id=bridge_family_id,
        binding=binding,
        admissible_scale_ids=scales,
    )
    return PublicRecognizerPredictionOutcomeV1._issue(
        _RECORD_ISSUE_TOKEN,
        input_row_id=input_row.row_id,
        input_payload_sha256=input_row.payload_sha256,
        decision=decision,
        canonical_family_id=canonical_family_id,
        prediction=prediction,
        bridge_outcome_id=bridge_outcome_id,
        bridge_compilation_id=compilation_id,
        bridge_decision_id=decision_id,
    )


def recognize_public_input_row_v1(
    *,
    input_row: TrustedRecognizerInputRowV1,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> PublicRecognizerPredictionOutcomeV1:
    """Run one split-blind row in process without issuing a trusted receipt."""

    if type(input_row) is not TrustedRecognizerInputRowV1:
        raise TypeError("single-row prediction requires exact input row")
    if type(execution_freeze_manifest) is not ExecutionFreezeManifest:
        raise TypeError("single-row prediction requires exact freeze manifest")
    _validate_execution_freeze_manifest(execution_freeze_manifest)
    input_row._validate()
    typed = decode_and_replay_typed_trusted_envelope_v1(input_row.envelope)
    bridge_result = run_exact_derived_witness_bridge(
        authority=typed.authority,
        theory=_FROZEN_THEORY,
        registry=input_row.public_registry.to_adapter_registry(),
    )
    return _compile_prediction_outcome_from_bridge(
        input_row=input_row,
        typed=typed,
        execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
        bridge_result=bridge_result,
    )


def _reject_forbidden_public_wire_tokens(value: object) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            for key, item in current.items():
                if type(key) is not str:
                    raise TypeError("public prediction wire key must use exact text")
                folded = key.casefold()
                if any(token in folded for token in _FORBIDDEN_RECOGNIZER_FIELD_TOKENS):
                    raise ValueError("public prediction wire contains a forbidden field")
                stack.append(item)
        elif type(current) in (list, tuple):
            stack.extend(current)
        elif type(current) is str:
            folded = current.casefold()
            if any(token in folded for token in _FORBIDDEN_RECOGNIZER_FIELD_TOKENS):
                raise ValueError("public prediction wire contains a forbidden value")


def _record_mapping_without_id(
    value: "PublicRecognizerPredictionRecordV1",
) -> dict[str, object]:
    return {
        "bridge_compilation_id": value.bridge_compilation_id,
        "bridge_decision_id": value.bridge_decision_id,
        "bridge_outcome_id": value.bridge_outcome_id,
        "canonical_family_id": (
            None
            if value.canonical_family_id is None
            else value.canonical_family_id.value
        ),
        "decision": value.decision.value,
        "input_authority_content_id": value.input_authority_content_id,
        "input_payload_sha256": value.input_payload_sha256,
        "input_row_id": value.input_row_id,
        "input_transform_result_id": value.input_transform_result_id,
        "prediction": _prediction_mapping(value.prediction),
        "prediction_content_id": value.prediction_content_id,
        "public_registry_id": value.public_registry_id,
        "run_context_id": value.run_context_id,
        "schema_version": value.schema_version,
    }


def _record_mapping(value: "PublicRecognizerPredictionRecordV1") -> dict[str, object]:
    mapping = _record_mapping_without_id(value)
    mapping["record_id"] = value.record_id
    return mapping


def _validate_record_preimage(mapping_without_id: object) -> dict[str, object]:
    value = _closed_mapping(
        mapping_without_id,
        _RECORD_PREIMAGE_FIELDS,
        "public prediction record preimage",
    )
    if (
        type(value["schema_version"]) is not str
        or value["schema_version"]
        != PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("public prediction record preimage schema drift")
    _ascii(value["schema_version"], "public prediction record preimage schema")
    _digest(
        value["run_context_id"],
        "phase2b_public_run_context_",
        "public prediction record preimage context ID",
    )
    _digest(
        value["input_row_id"],
        "phase2b_recognizer_input_row_",
        "public prediction record preimage input row ID",
    )
    _sha256_text(
        value["input_payload_sha256"],
        "public prediction record preimage input payload SHA-256",
    )
    _digest(
        value["input_authority_content_id"],
        "phase2b_public_transform_evidence_",
        "public prediction record preimage authority ID",
    )
    _digest(
        value["input_transform_result_id"],
        "phase2b_exact_transform_result_",
        "public prediction record preimage transform result ID",
    )
    _digest(
        value["public_registry_id"],
        "phase2b_public_recognizer_registry_",
        "public prediction record preimage registry ID",
    )
    _bridge_outcome_digest(value["bridge_outcome_id"])
    if (value["bridge_compilation_id"] is None) != (
        value["bridge_decision_id"] is None
    ):
        raise ValueError("public prediction record preimage bridge roots are partial")
    if value["bridge_compilation_id"] is not None:
        _digest(
            value["bridge_compilation_id"],
            "phase2b_exact_derived_bridge_result_",
            "public prediction record preimage bridge compilation ID",
        )
        _digest(
            value["bridge_decision_id"],
            "phase2b_exact_derived_decision_",
            "public prediction record preimage bridge decision ID",
        )
    if type(value["decision"]) is not str:
        raise TypeError("public prediction record preimage decision must use text")
    try:
        decision = PredictionDecisionV1(value["decision"])
    except ValueError as exc:
        raise ValueError("public prediction record preimage decision drift") from exc
    raw_family = value["canonical_family_id"]
    if raw_family is None:
        canonical_family_id = None
    else:
        if type(raw_family) is not str:
            raise TypeError("public prediction record preimage family must use text")
        try:
            canonical_family_id = CanonicalFamilyId(raw_family)
        except ValueError as exc:
            raise ValueError("public prediction record preimage family drift") from exc
    prediction = _decode_prediction_bundle(value["prediction"])
    _digest(
        value["prediction_content_id"],
        "phase2b_prediction_",
        "public prediction record preimage content ID",
    )
    if (
        prediction.input_root_sha256 != value["input_payload_sha256"]
        or prediction.content_id != value["prediction_content_id"]
    ):
        raise ValueError("public prediction record preimage bundle root drift")
    _validate_family_and_decision(
        decision=decision,
        canonical_family_id=canonical_family_id,
        prediction=prediction,
    )
    return value


def _record_id(mapping_without_id: dict[str, object]) -> str:
    _validate_record_preimage(mapping_without_id)
    _reject_forbidden_public_wire_tokens(mapping_without_id)
    payload = encode_phase2b_jcs_profile_v1(mapping_without_id)
    if len(payload) > MAXIMUM_PREDICTION_RECORD_BYTES:
        raise ValueError("public prediction record exceeds its byte cap")
    return "phase2b_recognizer_prediction_record_" + hashlib.sha256(
        _RECORD_DOMAIN + payload
    ).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerPredictionRecordV1:
    schema_version: str
    run_context_id: str
    input_row_id: str
    input_payload_sha256: str
    input_authority_content_id: str
    input_transform_result_id: str
    public_registry_id: str
    bridge_outcome_id: str
    bridge_compilation_id: str | None
    bridge_decision_id: str | None
    decision: PredictionDecisionV1
    canonical_family_id: CanonicalFamilyId | None
    prediction: PredictionBundle
    prediction_content_id: str
    record_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("public prediction records are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        context: PublicRunContextV1,
        input_row_id: str,
        input_payload_sha256: str,
        input_authority_content_id: str,
        input_transform_result_id: str,
        public_registry_id: str,
        outcome: PublicRecognizerPredictionOutcomeV1,
    ) -> "PublicRecognizerPredictionRecordV1":
        if token is not _RECORD_ISSUE_TOKEN:
            raise TypeError("public prediction record issuer token mismatch")
        if type(context) is not PublicRunContextV1:
            raise TypeError("public prediction record requires exact context")
        if type(outcome) is not PublicRecognizerPredictionOutcomeV1:
            raise TypeError("public prediction record requires exact outcome")
        _digest(
            input_row_id,
            "phase2b_recognizer_input_row_",
            "public prediction input row ID",
        )
        _sha256_text(input_payload_sha256, "public prediction input payload SHA-256")
        _digest(
            input_authority_content_id,
            "phase2b_public_transform_evidence_",
            "public prediction input authority ID",
        )
        _digest(
            input_transform_result_id,
            "phase2b_exact_transform_result_",
            "public prediction transform result ID",
        )
        _digest(
            public_registry_id,
            "phase2b_public_recognizer_registry_",
            "public prediction registry ID",
        )
        context._validate()
        outcome._validate()
        if (
            outcome.input_row_id != input_row_id
            or outcome.input_payload_sha256 != input_payload_sha256
        ):
            raise ValueError("public prediction outcome/input row drift")
        if (
            outcome.prediction.protocol_sha256
            != context.protocol_id.rsplit("_", 1)[1]
            or outcome.prediction.freeze_manifest_sha256
            != context.execution_freeze_manifest_id.rsplit("_", 1)[1]
        ):
            raise ValueError("public prediction outcome/context binding drift")
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION),
            ("run_context_id", context.context_id),
            ("input_row_id", input_row_id),
            ("input_payload_sha256", input_payload_sha256),
            ("input_authority_content_id", input_authority_content_id),
            ("input_transform_result_id", input_transform_result_id),
            ("public_registry_id", public_registry_id),
            ("bridge_outcome_id", outcome.bridge_outcome_id),
            ("bridge_compilation_id", outcome.bridge_compilation_id),
            ("bridge_decision_id", outcome.bridge_decision_id),
            ("decision", outcome.decision),
            ("canonical_family_id", outcome.canonical_family_id),
            ("prediction", outcome.prediction),
            ("prediction_content_id", outcome.prediction.content_id),
        ):
            object.__setattr__(value, name, item)
        object.__setattr__(value, "record_id", _record_id(_record_mapping_without_id(value)))
        value._validate(context=context, context_token=_RECORD_CONTEXT_TOKEN)
        return value

    def _validate(
        self,
        *,
        context: PublicRunContextV1 | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not PublicRecognizerPredictionRecordV1:
            raise TypeError("public prediction record must use exact type")
        if (context is None) is not (context_token is None):
            raise TypeError("public prediction record context token mismatch")
        if context is not None and context_token is not _RECORD_CONTEXT_TOKEN:
            raise TypeError("public prediction record context is private")
        if self.schema_version != PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION:
            raise ValueError("public prediction record schema drift")
        _ascii(self.schema_version, "public prediction record schema")
        _digest(
            self.run_context_id,
            "phase2b_public_run_context_",
            "public prediction record context ID",
        )
        _digest(
            self.input_row_id,
            "phase2b_recognizer_input_row_",
            "public prediction input row ID",
        )
        _sha256_text(self.input_payload_sha256, "public prediction input payload SHA-256")
        _digest(
            self.input_authority_content_id,
            "phase2b_public_transform_evidence_",
            "public prediction input authority ID",
        )
        _digest(
            self.input_transform_result_id,
            "phase2b_exact_transform_result_",
            "public prediction transform result ID",
        )
        _digest(
            self.public_registry_id,
            "phase2b_public_recognizer_registry_",
            "public prediction registry ID",
        )
        _bridge_outcome_digest(self.bridge_outcome_id)
        if (self.bridge_compilation_id is None) != (self.bridge_decision_id is None):
            raise ValueError("public prediction bridge roots are partial")
        if self.bridge_compilation_id is not None:
            _digest(
                self.bridge_compilation_id,
                "phase2b_exact_derived_bridge_result_",
                "public prediction bridge compilation ID",
            )
            _digest(
                self.bridge_decision_id,
                "phase2b_exact_derived_decision_",
                "public prediction bridge decision ID",
            )
        if type(self.decision) is not PredictionDecisionV1:
            raise TypeError("public prediction decision must use exact enum")
        prediction = _decode_prediction_bundle(_prediction_mapping(self.prediction))
        if (
            prediction.input_root_sha256 != self.input_payload_sha256
            or self.prediction_content_id != prediction.content_id
        ):
            raise ValueError("public prediction bundle root drift")
        _digest(
            self.prediction_content_id,
            "phase2b_prediction_",
            "public prediction content ID",
        )
        _validate_family_and_decision(
            decision=self.decision,
            canonical_family_id=self.canonical_family_id,
            prediction=prediction,
        )
        if context is not None:
            context._validate()
            if (
                self.run_context_id != context.context_id
                or prediction.protocol_sha256
                != context.protocol_id.rsplit("_", 1)[1]
                or prediction.freeze_manifest_sha256
                != context.execution_freeze_manifest_id.rsplit("_", 1)[1]
            ):
                raise ValueError("public prediction context binding drift")
        _digest(
            self.record_id,
            "phase2b_recognizer_prediction_record_",
            "public prediction record ID",
        )
        if self.record_id != _record_id(_record_mapping_without_id(self)):
            raise ValueError("public prediction record root drift")


def _decode_record(
    mapping: object,
    *,
    context: PublicRunContextV1,
) -> PublicRecognizerPredictionRecordV1:
    value = _closed_mapping(mapping, _RECORD_FIELDS, "public prediction record")
    _reject_forbidden_public_wire_tokens(value)
    if type(value["decision"]) is not str:
        raise TypeError("public prediction decision wire value must use exact text")
    try:
        decision = PredictionDecisionV1(value["decision"])
    except ValueError as exc:
        raise ValueError("public prediction decision wire value is unknown") from exc
    raw_family = value["canonical_family_id"]
    if raw_family is None:
        canonical_family_id = None
    else:
        if type(raw_family) is not str:
            raise TypeError("public canonical family must use exact text or null")
        try:
            canonical_family_id = CanonicalFamilyId(raw_family)
        except ValueError as exc:
            raise ValueError("public canonical family is unknown") from exc
    prediction = _decode_prediction_bundle(value["prediction"])
    outcome = PublicRecognizerPredictionOutcomeV1._issue(
        _RECORD_ISSUE_TOKEN,
        input_row_id=value["input_row_id"],  # type: ignore[arg-type]
        input_payload_sha256=value["input_payload_sha256"],  # type: ignore[arg-type]
        decision=decision,
        canonical_family_id=canonical_family_id,
        prediction=prediction,
        bridge_outcome_id=value["bridge_outcome_id"],  # type: ignore[arg-type]
        bridge_compilation_id=value["bridge_compilation_id"],  # type: ignore[arg-type]
        bridge_decision_id=value["bridge_decision_id"],  # type: ignore[arg-type]
    )
    record = PublicRecognizerPredictionRecordV1._issue(
        _RECORD_ISSUE_TOKEN,
        context=context,
        input_row_id=value["input_row_id"],  # type: ignore[arg-type]
        input_payload_sha256=value["input_payload_sha256"],  # type: ignore[arg-type]
        input_authority_content_id=value["input_authority_content_id"],  # type: ignore[arg-type]
        input_transform_result_id=value["input_transform_result_id"],  # type: ignore[arg-type]
        public_registry_id=value["public_registry_id"],  # type: ignore[arg-type]
        outcome=outcome,
    )
    if _record_mapping(record) != value:
        raise ValueError("public prediction record canonical roundtrip drift")
    return record


def _manifest_mapping(
    *,
    context: PublicRunContextV1,
    input_row_ids_root: str,
    prediction_record_ids_root: str,
) -> dict[str, object]:
    context._validate()
    if input_row_ids_root != context.input_row_ids_root:
        raise ValueError("prediction manifest/context input-row root drift")
    return {
        "archive_policy_id": RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID,
        "archive_schema_version": RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION,
        "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
        "execution_freeze_manifest_id": context.execution_freeze_manifest_id,
        "expected_prediction_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "input_archive_id": context.input_archive_id,
        "input_archive_sha256": context.input_archive_sha256,
        "input_row_ids_root": input_row_ids_root,
        "prediction_record_ids_root": prediction_record_ids_root,
        "protocol_id": context.protocol_id,
        "record_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "run_context_id": context.context_id,
    }


def _validate_manifest(
    value: object,
) -> tuple[dict[str, object], PublicRunContextV1]:
    mapping = _closed_mapping(value, _MANIFEST_FIELDS, "prediction archive manifest")
    _reject_forbidden_public_wire_tokens(mapping)
    if (
        mapping["archive_schema_version"]
        != RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION
        or mapping["archive_policy_id"]
        != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID
        or mapping["protocol_id"] != _FROZEN_PROTOCOL.protocol_id
        or mapping["claim_level"] != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("prediction archive manifest policy drift")
    for field_name in (
        "expected_prediction_count",
        "record_count",
    ):
        if (
            type(mapping[field_name]) is not int
            or mapping[field_name] != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            raise ValueError("prediction archive manifest count drift")
    _ascii(mapping["archive_schema_version"], "prediction archive schema")
    _digest(
        mapping["archive_policy_id"],
        "phase2b_recognizer_prediction_archive_policy_",
        "prediction archive policy ID",
    )
    _digest(
        mapping["input_archive_id"],
        "phase2b_recognizer_input_archive_",
        "prediction manifest input archive ID",
    )
    _sha256_text(mapping["input_archive_sha256"], "prediction manifest input SHA-256")
    _digest(
        mapping["input_row_ids_root"],
        "phase2b_prediction_input_rows_",
        "prediction manifest input-row root",
    )
    _digest(
        mapping["prediction_record_ids_root"],
        "phase2b_prediction_records_",
        "prediction manifest record root",
    )
    _digest(mapping["protocol_id"], "phase2b_protocol_", "prediction protocol ID")
    _digest(
        mapping["execution_freeze_manifest_id"],
        "phase2b_execution_freeze_",
        "prediction execution freeze manifest ID",
    )
    _digest(
        mapping["run_context_id"],
        "phase2b_public_run_context_",
        "prediction run context ID",
    )
    context = PublicRunContextV1._issue(
        _CONTEXT_ISSUE_TOKEN,
        input_archive_id=mapping["input_archive_id"],  # type: ignore[arg-type]
        input_archive_sha256=mapping["input_archive_sha256"],  # type: ignore[arg-type]
        input_row_ids_root=mapping["input_row_ids_root"],  # type: ignore[arg-type]
        protocol_id=mapping["protocol_id"],  # type: ignore[arg-type]
        execution_freeze_manifest_id=mapping["execution_freeze_manifest_id"],  # type: ignore[arg-type]
    )
    if context.context_id != mapping["run_context_id"]:
        raise ValueError("prediction archive manifest context root drift")
    return mapping, context


def _frame(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("prediction frame requires exact bytes")
    return len(payload).to_bytes(4, "big") + payload


def _encode_prediction_archive(
    *,
    context: PublicRunContextV1,
    records: tuple[PublicRecognizerPredictionRecordV1, ...],
) -> bytes:
    if (
        type(records) is not tuple
        or len(records) != TOTAL_RECOGNIZER_CASE_COUNT
        or any(type(item) is not PublicRecognizerPredictionRecordV1 for item in records)
    ):
        raise TypeError("prediction archive requires exactly 960 exact records")
    if records != tuple(sorted(records, key=lambda item: item.input_row_id)):
        raise ValueError("prediction archive records are not in canonical input-row order")
    input_row_ids = tuple(item.input_row_id for item in records)
    record_ids = tuple(item.record_id for item in records)
    prediction_content_ids = tuple(item.prediction_content_id for item in records)
    if any(
        len(set(values)) != TOTAL_RECOGNIZER_CASE_COUNT
        for values in (input_row_ids, record_ids, prediction_content_ids)
    ):
        raise ValueError("prediction archive repeats a row, record, or prediction root")
    for record in records:
        record._validate(context=context, context_token=_RECORD_CONTEXT_TOKEN)
    input_root = _input_row_ids_root(input_row_ids)
    record_root = _prediction_record_ids_root(record_ids)
    manifest = _manifest_mapping(
        context=context,
        input_row_ids_root=input_root,
        prediction_record_ids_root=record_root,
    )
    _reject_forbidden_public_wire_tokens(manifest)
    manifest_payload = encode_phase2b_jcs_profile_v1(manifest)
    if not manifest_payload or len(manifest_payload) > MAXIMUM_PREDICTION_MANIFEST_BYTES:
        raise ValueError("prediction archive manifest exceeds its byte cap")
    frames = [_frame(manifest_payload)]
    total = PREDICTION_ARCHIVE_HEADER_BYTES + len(frames[0])
    for record in records:
        mapping = _record_mapping(record)
        _reject_forbidden_public_wire_tokens(mapping)
        payload = encode_phase2b_jcs_profile_v1(mapping)
        if not payload or len(payload) > MAXIMUM_PREDICTION_RECORD_BYTES:
            raise ValueError("prediction archive record exceeds its byte cap")
        framed = _frame(payload)
        total += len(framed)
        if total > MAXIMUM_PREDICTION_ARCHIVE_BYTES:
            raise ValueError("prediction archive exceeds its byte cap")
        frames.append(framed)
    body = b"".join(frames)
    header = _ARCHIVE_HEADER.pack(
        PREDICTION_ARCHIVE_MAGIC,
        PREDICTION_ARCHIVE_WIRE_VERSION,
        0,
        TOTAL_RECOGNIZER_CASE_COUNT,
        hashlib.sha256(body).digest(),
    )
    archive = header + body
    if len(archive) != total:
        raise RuntimeError("prediction archive byte accounting drift")
    return archive


@dataclass(frozen=True, slots=True)
class _ParsedPredictionArchiveV1:
    archive_id: str
    context: PublicRunContextV1
    records: tuple[PublicRecognizerPredictionRecordV1, ...]


def _archive_id(archive: bytes) -> str:
    if type(archive) is not bytes:
        raise TypeError("prediction archive ID requires exact bytes")
    if not archive or len(archive) > MAXIMUM_PREDICTION_ARCHIVE_BYTES:
        raise ValueError("prediction archive byte budget drift")
    return "phase2b_recognizer_prediction_archive_" + hashlib.sha256(
        _ARCHIVE_DOMAIN + archive
    ).hexdigest()


def _read_frame(
    archive: bytes,
    offset: int,
    *,
    maximum_bytes: int,
    name: str,
) -> tuple[bytes, int]:
    if offset + 4 > len(archive):
        raise ValueError(f"{name} frame length is truncated")
    length = int.from_bytes(archive[offset : offset + 4], "big")
    offset += 4
    if not 1 <= length <= maximum_bytes:
        raise ValueError(f"{name} frame length exceeds its cap")
    end = offset + length
    if end > len(archive):
        raise ValueError(f"{name} frame is truncated")
    return archive[offset:end], end


def _parse_prediction_archive(archive: bytes) -> _ParsedPredictionArchiveV1:
    if type(archive) is not bytes:
        raise TypeError("prediction archive input must use exact bytes")
    minimum = PREDICTION_ARCHIVE_HEADER_BYTES + 4 + 1
    if len(archive) < minimum or len(archive) > MAXIMUM_PREDICTION_ARCHIVE_BYTES:
        raise ValueError("prediction archive byte budget drift")
    magic, wire_version, flags, record_count, body_sha = _ARCHIVE_HEADER.unpack_from(
        archive,
        0,
    )
    if (
        magic != PREDICTION_ARCHIVE_MAGIC
        or wire_version != PREDICTION_ARCHIVE_WIRE_VERSION
        or flags != 0
        or record_count != TOTAL_RECOGNIZER_CASE_COUNT
    ):
        raise ValueError("prediction archive header drift")
    body = archive[PREDICTION_ARCHIVE_HEADER_BYTES :]
    if hashlib.sha256(body).digest() != body_sha:
        raise ValueError("prediction archive body digest drift")
    offset = PREDICTION_ARCHIVE_HEADER_BYTES
    manifest_payload, offset = _read_frame(
        archive,
        offset,
        maximum_bytes=MAXIMUM_PREDICTION_MANIFEST_BYTES,
        name="prediction manifest",
    )
    manifest_object = decode_phase2b_jcs_profile_v1(manifest_payload)
    manifest, context = _validate_manifest(manifest_object)
    if encode_phase2b_jcs_profile_v1(manifest) != manifest_payload:
        raise ValueError("prediction archive manifest is not canonical")
    records: list[PublicRecognizerPredictionRecordV1] = []
    for index in range(TOTAL_RECOGNIZER_CASE_COUNT):
        payload, offset = _read_frame(
            archive,
            offset,
            maximum_bytes=MAXIMUM_PREDICTION_RECORD_BYTES,
            name=f"prediction record {index}",
        )
        record_object = decode_phase2b_jcs_profile_v1(payload)
        record = _decode_record(record_object, context=context)
        if encode_phase2b_jcs_profile_v1(_record_mapping(record)) != payload:
            raise ValueError("prediction archive record is not canonical")
        records.append(record)
    if offset != len(archive):
        raise ValueError("prediction archive has trailing bytes")
    frozen_records = tuple(records)
    if frozen_records != tuple(
        sorted(frozen_records, key=lambda item: item.input_row_id)
    ):
        raise ValueError("prediction archive record order drift")
    input_row_ids = tuple(item.input_row_id for item in frozen_records)
    record_ids = tuple(item.record_id for item in frozen_records)
    prediction_ids = tuple(item.prediction_content_id for item in frozen_records)
    if any(
        len(set(items)) != TOTAL_RECOGNIZER_CASE_COUNT
        for items in (input_row_ids, record_ids, prediction_ids)
    ):
        raise ValueError("prediction archive repeats a committed root")
    if (
        _input_row_ids_root(input_row_ids) != manifest["input_row_ids_root"]
        or _prediction_record_ids_root(record_ids)
        != manifest["prediction_record_ids_root"]
        or context.input_row_ids_root != manifest["input_row_ids_root"]
    ):
        raise ValueError("prediction archive ordered root drift")
    return _ParsedPredictionArchiveV1(
        archive_id=_archive_id(archive),
        context=context,
        records=frozen_records,
    )


@dataclass(frozen=True, slots=True, init=False)
class DecodedRecognizerPredictionArchiveV1:
    disposition: RecognizerPredictionArchiveDisposition
    archive: bytes
    archive_id: str
    schema_version: str
    policy_id: str
    context: PublicRunContextV1
    records: tuple[PublicRecognizerPredictionRecordV1, ...]
    input_row_ids: tuple[str, ...]
    prediction_record_ids: tuple[str, ...]
    prediction_content_ids: tuple[str, ...]
    claim_level: str
    structural_archive_verified: bool
    canonical_record_framing_verified: bool
    record_schema_verified: bool
    row_root_coverage_verified: bool
    input_archive_membership_verified: bool
    execution_manifest_authority_verified: bool
    derived_bridge_mapping_verified: bool
    runtime_executed: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    sealed_holdout_eligible: bool
    formal_covert_audit: bool
    prediction_scored: bool
    effect_evidence: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("decoded prediction archives are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        archive: bytes,
        parsed: _ParsedPredictionArchiveV1,
    ) -> "DecodedRecognizerPredictionArchiveV1":
        if token is not _DECODE_ISSUE_TOKEN:
            raise TypeError("decoded prediction archive issuer token mismatch")
        value = object.__new__(cls)
        records = parsed.records
        for name, item in (
            ("disposition", RecognizerPredictionArchiveDisposition.COMPLETE),
            ("archive", archive),
            ("archive_id", parsed.archive_id),
            ("schema_version", RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION),
            ("policy_id", RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID),
            ("context", parsed.context),
            ("records", records),
            ("input_row_ids", tuple(item.input_row_id for item in records)),
            ("prediction_record_ids", tuple(item.record_id for item in records)),
            ("prediction_content_ids", tuple(item.prediction_content_id for item in records)),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("structural_archive_verified", True),
            ("canonical_record_framing_verified", True),
            ("record_schema_verified", True),
            ("row_root_coverage_verified", True),
            ("input_archive_membership_verified", False),
            ("execution_manifest_authority_verified", False),
            ("derived_bridge_mapping_verified", False),
            ("runtime_executed", False),
            ("recognizer_capacity_evidence", False),
            ("origin_authenticated", False),
            ("sealed_holdout_eligible", False),
            ("formal_covert_audit", False),
            ("prediction_scored", False),
            ("effect_evidence", False),
            ("c1_exit_evidence", False),
        ):
            object.__setattr__(value, name, item)
        value._validate(
            parsed=parsed,
            context_token=_PARSED_CONTEXT_TOKEN,
        )
        return value

    def _validate(
        self,
        *,
        parsed: _ParsedPredictionArchiveV1 | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not DecodedRecognizerPredictionArchiveV1:
            raise TypeError("decoded prediction archive must use exact type")
        if (parsed is None) is not (context_token is None):
            raise TypeError("decoded prediction parsed context token mismatch")
        if parsed is not None and context_token is not _PARSED_CONTEXT_TOKEN:
            raise TypeError("decoded prediction parsed context is private")
        if type(self.archive) is not bytes:
            raise TypeError("decoded prediction archive must store exact bytes")
        if type(self.disposition) is not RecognizerPredictionArchiveDisposition:
            raise TypeError("decoded prediction disposition must use exact enum")
        if self.disposition is not RecognizerPredictionArchiveDisposition.COMPLETE:
            raise ValueError("decoded prediction archive disposition drift")
        for item in (
            self.structural_archive_verified,
            self.canonical_record_framing_verified,
            self.record_schema_verified,
            self.row_root_coverage_verified,
            self.input_archive_membership_verified,
            self.execution_manifest_authority_verified,
            self.derived_bridge_mapping_verified,
            self.runtime_executed,
            self.recognizer_capacity_evidence,
            self.origin_authenticated,
            self.sealed_holdout_eligible,
            self.formal_covert_audit,
            self.prediction_scored,
            self.effect_evidence,
            self.c1_exit_evidence,
        ):
            if type(item) is not bool:
                raise TypeError("decoded prediction claims require exact bool")
        if (
            not self.structural_archive_verified
            or not self.canonical_record_framing_verified
            or not self.record_schema_verified
            or not self.row_root_coverage_verified
            or any(
                (
                    self.input_archive_membership_verified,
                    self.execution_manifest_authority_verified,
                    self.derived_bridge_mapping_verified,
                    self.runtime_executed,
                    self.recognizer_capacity_evidence,
                    self.origin_authenticated,
                    self.sealed_holdout_eligible,
                    self.formal_covert_audit,
                    self.prediction_scored,
                    self.effect_evidence,
                    self.c1_exit_evidence,
                )
            )
        ):
            raise ValueError("decoded prediction claim boundary drift")
        _digest(
            self.archive_id,
            "phase2b_recognizer_prediction_archive_",
            "decoded prediction archive ID",
        )
        if (
            type(self.schema_version) is not str
            or self.schema_version != RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION
        ):
            raise ValueError("decoded prediction archive schema drift")
        _ascii(self.schema_version, "decoded prediction archive schema")
        _digest(
            self.policy_id,
            "phase2b_recognizer_prediction_archive_policy_",
            "decoded prediction archive policy ID",
        )
        if self.policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID:
            raise ValueError("decoded prediction archive policy drift")
        if type(self.context) is not PublicRunContextV1:
            raise TypeError("decoded prediction archive context must use exact type")
        self.context._validate()
        if (
            type(self.records) is not tuple
            or len(self.records) != TOTAL_RECOGNIZER_CASE_COUNT
            or any(
                type(item) is not PublicRecognizerPredictionRecordV1
                for item in self.records
            )
        ):
            raise TypeError("decoded prediction archive records are invalid")
        for record in self.records:
            record._validate(
                context=self.context,
                context_token=_RECORD_CONTEXT_TOKEN,
            )
        for values, prefix, name in (
            (
                self.input_row_ids,
                "phase2b_recognizer_input_row_",
                "decoded prediction input row ID",
            ),
            (
                self.prediction_record_ids,
                "phase2b_recognizer_prediction_record_",
                "decoded prediction record ID",
            ),
            (
                self.prediction_content_ids,
                "phase2b_prediction_",
                "decoded prediction content ID",
            ),
        ):
            if type(values) is not tuple or len(values) != TOTAL_RECOGNIZER_CASE_COUNT:
                raise TypeError("decoded prediction root arrays must use exact tuples")
            for item in values:
                _digest(item, prefix, name)
        if (
            type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("decoded prediction claim level drift")
        actual = _parse_prediction_archive(self.archive) if parsed is None else parsed
        if type(actual) is not _ParsedPredictionArchiveV1:
            raise TypeError("decoded prediction parsed context has wrong type")
        if (
            self.archive_id != actual.archive_id
            or self.schema_version != RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION
            or self.policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID
            or self.context != actual.context
            or self.records != actual.records
            or self.input_row_ids != tuple(item.input_row_id for item in actual.records)
            or self.prediction_record_ids != tuple(item.record_id for item in actual.records)
            or self.prediction_content_ids
            != tuple(item.prediction_content_id for item in actual.records)
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("decoded prediction archive replay drift")


@dataclass(frozen=True, slots=True)
class RecognizerPredictionArchiveRejectionV1:
    disposition: RecognizerPredictionArchiveDisposition
    reason: str
    input_count: int
    input_archive_id: str | None
    archive: None = None
    records: tuple[()] = ()
    prediction_record_ids: tuple[()] = ()
    recognizer_capacity_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not RecognizerPredictionArchiveRejectionV1:
            raise TypeError("prediction archive rejection must use exact type")
        if self.disposition is not RecognizerPredictionArchiveDisposition.ABSTAIN:
            raise ValueError("prediction archive rejection must abstain")
        _ascii(self.reason, "prediction archive rejection reason")
        if type(self.input_count) is not int or self.input_count < 0:
            raise ValueError("prediction archive rejection count is invalid")
        if self.input_archive_id is not None:
            _digest(
                self.input_archive_id,
                "phase2b_recognizer_input_archive_",
                "prediction rejection input archive ID",
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
            raise ValueError("prediction archive rejection leaked partial output")


def _rejection(
    reason: str,
    *,
    input_count: int,
    input_archive_id: str | None,
) -> RecognizerPredictionArchiveRejectionV1:
    return RecognizerPredictionArchiveRejectionV1(
        disposition=RecognizerPredictionArchiveDisposition.ABSTAIN,
        reason=reason,
        input_count=input_count,
        input_archive_id=input_archive_id,
    )


def decode_public_recognizer_prediction_archive_v1(
    archive: bytes,
) -> DecodedRecognizerPredictionArchiveV1:
    parsed = _parse_prediction_archive(archive)
    return DecodedRecognizerPredictionArchiveV1._issue(
        _DECODE_ISSUE_TOKEN,
        archive=archive,
        parsed=parsed,
    )


def build_recognizer_prediction_archive_v1(
    *,
    input_archive: DecodedRecognizerInputArchiveV1,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> DecodedRecognizerPredictionArchiveV1 | RecognizerPredictionArchiveRejectionV1:
    """Replay all 960 rows as one atomic, non-authoritative mechanics gate."""

    if type(input_archive) is not DecodedRecognizerInputArchiveV1:
        raise TypeError("prediction archive builder needs exact decoded input archive")
    if type(execution_freeze_manifest) is not ExecutionFreezeManifest:
        raise TypeError("prediction archive builder needs exact execution freeze manifest")
    try:
        raw_input_archive_id = input_archive.archive_id
    except AttributeError:
        return _rejection(
            "input_archive_shallow_invalid",
            input_count=0,
            input_archive_id=None,
        )
    input_archive_id = _safe_digest_or_none(
        raw_input_archive_id,
        "phase2b_recognizer_input_archive_",
        "prediction builder input archive ID",
    )
    try:
        input_rows, _ = _shallow_prediction_input_rows(input_archive)
    except _PredictionInputShallowError as exc:
        public_reason = (
            exc.reason
            if exc.reason
            in {
                "input_archive_shallow_invalid",
                "input_rows_type_invalid",
                "prediction_count_drift",
            }
            else "input_or_execution_context_invalid"
        )
        return _rejection(
            public_reason,
            input_count=exc.input_count,
            input_archive_id=input_archive_id,
        )
    input_count = len(input_rows)
    try:
        context = build_public_run_context_v1(
            input_archive=input_archive,
            execution_freeze_manifest=execution_freeze_manifest,
        )
    except (AttributeError, TypeError, ValueError):
        return _rejection(
            "input_or_execution_context_invalid",
            input_count=input_count,
            input_archive_id=input_archive_id,
        )
    records: list[PublicRecognizerPredictionRecordV1] = []
    try:
        for input_row in sorted(input_rows, key=lambda item: item.row_id):
            if type(input_row) is not TrustedRecognizerInputRowV1:
                raise _PredictionGateRejected("input_row_type_invalid")
            input_row._validate()
            typed = decode_and_replay_typed_trusted_envelope_v1(input_row.envelope)
            bridge_result = run_exact_derived_witness_bridge(
                authority=typed.authority,
                theory=_FROZEN_THEORY,
                registry=input_row.public_registry.to_adapter_registry(),
            )
            outcome = _compile_prediction_outcome_from_bridge(
                input_row=input_row,
                typed=typed,
                execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
                bridge_result=bridge_result,
            )
            record = PublicRecognizerPredictionRecordV1._issue(
                _RECORD_ISSUE_TOKEN,
                context=context,
                input_row_id=input_row.row_id,
                input_payload_sha256=input_row.payload_sha256,
                input_authority_content_id=input_row.authority_content_id,
                input_transform_result_id=input_row.transform_result_id,
                public_registry_id=input_row.public_registry_id,
                outcome=outcome,
            )
            if (
                record.input_row_id != input_row.row_id
                or record.input_payload_sha256 != input_row.payload_sha256
                or record.input_authority_content_id != input_row.authority_content_id
                or record.input_transform_result_id != input_row.transform_result_id
                or record.public_registry_id != input_row.public_registry_id
            ):
                raise _PredictionGateRejected("prediction_record_input_root_drift")
            records.append(record)
    except _PredictionGateRejected as exc:
        return _rejection(
            str(exc),
            input_count=input_count,
            input_archive_id=input_archive_id,
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _rejection(
            "prediction_row_mapping_failed",
            input_count=input_count,
            input_archive_id=input_archive_id,
        )
    try:
        archive = _encode_prediction_archive(context=context, records=tuple(records))
        return decode_public_recognizer_prediction_archive_v1(archive)
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _rejection(
            "prediction_archive_encoding_failed",
            input_count=input_count,
            input_archive_id=input_archive_id,
        )


def _assert_public_field_manifests() -> None:
    checks = (
        (
            tuple(sorted(item.name for item in fields(PublicRunContextV1))),
            _CONTEXT_FIELDS,
            "public run context",
        ),
        (
            tuple(item.name for item in fields(PublicRecognizerPredictionOutcomeV1)),
            _OUTCOME_FIELD_MANIFEST,
            "public prediction outcome",
        ),
        (
            tuple(sorted(item.name for item in fields(PublicRecognizerPredictionRecordV1))),
            _RECORD_FIELDS,
            "public prediction record",
        ),
        (
            tuple(item.name for item in fields(DecodedRecognizerPredictionArchiveV1)),
            _DECODED_FIELD_MANIFEST,
            "decoded prediction archive",
        ),
        (
            tuple(item.name for item in fields(RecognizerPredictionArchiveRejectionV1)),
            _REJECTION_FIELD_MANIFEST,
            "prediction archive rejection",
        ),
    )
    for actual, expected, name in checks:
        if actual != expected:
            raise RuntimeError(f"{name} field manifest drift")


_assert_public_field_manifests()


__all__ = (
    "MAXIMUM_PREDICTION_ARCHIVE_BYTES",
    "MAXIMUM_PREDICTION_MANIFEST_BYTES",
    "MAXIMUM_PREDICTION_RECORD_BYTES",
    "PREDICTION_ARCHIVE_HEADER_BYTES",
    "PREDICTION_ARCHIVE_MAGIC",
    "PREDICTION_ARCHIVE_WIRE_VERSION",
    "PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_ID",
    "PUBLIC_RECOGNIZER_PREDICTION_RECORD_SCHEMA_VERSION",
    "PUBLIC_RUN_CONTEXT_SCHEMA_ID",
    "PUBLIC_RUN_CONTEXT_SCHEMA_VERSION",
    "RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID",
    "RECOGNIZER_PREDICTION_ARCHIVE_SCHEMA_VERSION",
    "DecodedRecognizerPredictionArchiveV1",
    "PredictionDecisionV1",
    "PublicRecognizerPredictionOutcomeV1",
    "PublicRecognizerPredictionRecordV1",
    "PublicRunContextV1",
    "RecognizerPredictionArchiveDisposition",
    "RecognizerPredictionArchiveRejectionV1",
    "build_public_run_context_v1",
    "build_recognizer_prediction_archive_v1",
    "decode_public_recognizer_prediction_archive_v1",
    "recognize_public_input_row_v1",
)

"""Ephemeral public prediction mapping for one compact-V2 recognizer row.

The sole public operation validates a frozen execution manifest, replays one
exact V2 recognizer-input row, runs the existing exact derived-witness bridge,
and maps the bridge decision to the generic public ``PredictionBundle``.  The
returned outcome is an in-process mechanics object, not a durable receipt.  It
does not establish input-archive membership, execution, capacity, scoring,
effect, origin, formal audit, sealed-holdout eligibility, or C1 exit evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
import re
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
from .phase2b_exact_transform_semantics_v1 import PublicTransformEvidenceBundleV2
from .phase2b_freeze_v1 import CanonicalFamilyId, frozen_phase2b_exact_freeze
from . import phase2b_recognizer_input_archive_v2 as _archive_v2
from .phase2b_recognizer_input_archive_v2 import (
    FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2,
    PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
    PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
    PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
    PublicRecognizerRegistryV2,
    TrustedRecognizerInputRowV2,
)
from .phase2b_protocol import (
    BaselineKind,
    BaselineRegistration,
    ExecutionFreezeManifest,
    frozen_phase2b_protocol,
)
from .phase2b_trusted_wire_batch_v2 import (
    TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
    DecodedTrustedEnvelopeV2,
)
from . import phase2b_trusted_wire_batch_v2 as _batch_v2
from .phase2b_trusted_wire_typed_authority_v2 import (
    COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
    COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
)
from .phase2b_trusted_wire_typed_replay_v2 import (
    TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
    TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
    decode_and_replay_typed_trusted_envelope_v2,
)
from .phase2b_trusted_wire_v1 import (
    MAXIMUM_ASCII_STRING_BYTES,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
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


PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-public-recognizer-prediction-outcome/2"
)

_FROZEN_THEORY: Final = initial_theory()
_FROZEN_PROTOCOL: Final = frozen_phase2b_protocol()
_FROZEN_EXACT_FREEZE: Final = frozen_phase2b_exact_freeze()
_BRIDGE_FAMILY_BY_KIND: Final = dict(FROZEN_BRIDGE_FAMILY_UUID_ALIASES_V2)
_CANONICAL_FAMILY_BY_KIND: Final = dict(_FROZEN_EXACT_FREEZE.family_mapping)
_KIND_BY_CANONICAL_FAMILY: Final = {
    family_id: law_kind
    for law_kind, family_id in _FROZEN_EXACT_FREEZE.family_mapping
}

_OUTCOME_ISSUE_TOKEN: Final = object()
_PREFLIGHT_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/DERIVED_PREFLIGHT_OUTCOME/V2\x00"
)
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
_OUTCOME_FIELDS: Final = (
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


class PredictionDecisionV2(str, Enum):
    ANSWER = "unique_identification"
    ANSWER_SET = "admissible_scale_set"
    ABSTAIN = "abstain"


# This conversion is deliberately exact and closed.  New bridge reasons must
# revise the V2 row policy before they can become public prediction reasons.
_ABSTAIN_REASON_PAIRS: Final = (
    ("no_passing_structure", PredictionReason.NO_PASSING_CANDIDATE),
    ("multiple_passing_structures", PredictionReason.MULTIPLE_STRUCTURAL_MATCHES),
    ("nonidentifiable_interval_overlap", PredictionReason.NONIDENTIFIABLE_SCALE),
    (
        "selected_structure_has_inconclusive_scale",
        PredictionReason.NONIDENTIFIABLE_SCALE,
    ),
    ("inconclusive_structural_competitor", PredictionReason.NONIDENTIFIABLE_SCALE),
    ("missing_scale_competitor", PredictionReason.NONIDENTIFIABLE_SCALE),
    ("missing_binding_competitor", PredictionReason.INSUFFICIENT_EVIDENCE),
    ("missing_structural_competitor", PredictionReason.INSUFFICIENT_EVIDENCE),
    ("bridge_explicit_support_required", PredictionReason.INSUFFICIENT_EVIDENCE),
    ("bridge_no_injective_role_binding", PredictionReason.INSUFFICIENT_EVIDENCE),
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
    raise RuntimeError("V2 prediction abstention map repeats a source reason")
_ALLOWED_ABSTAIN_REASONS: Final = frozenset(_ABSTAIN_REASON_MAP.values())
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
    raise RuntimeError("V2 prediction resource-reason source closure drift")
_ROW_SEMANTIC_REASON_SOURCES: Final = tuple(
    sorted(
        source
        for source, target in _ABSTAIN_REASON_PAIRS
        if target is not PredictionReason.RESOURCE_LIMIT
    )
)

_INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED: Final = (
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
if set(_ABSTAIN_REASON_MAP) & set(_INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED):
    raise RuntimeError("V2 prediction accepted/rejected reason overlap")
if (
    set(_EXPECTED_RESOURCE_REASON_SOURCES)
    | set(_ROW_SEMANTIC_REASON_SOURCES)
) != set(_ABSTAIN_REASON_MAP):
    raise RuntimeError("V2 prediction accepted reason classification drift")
_ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS: Final = tuple(
    sorted(source for source in _ABSTAIN_REASON_MAP if source.startswith("bridge_"))
)
_ACCEPTED_COMPILATION_ABSTAIN_SOURCES: Final = tuple(
    source.removeprefix("bridge_")
    for source in _ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS
)
_ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES: Final = tuple(
    sorted(
        (
            "RESOURCE_LIMIT:selection_margin_bit_length",
            "candidate_evaluation_error",
            "inconclusive_structural_competitor",
            "insufficient_structural_margin",
            "missing_binding_competitor",
            "missing_scale_competitor",
            "missing_structural_competitor",
            "multiple_passing_structures",
            "no_passing_structure",
            "nonidentifiable_interval_overlap",
            "selected_structure_has_inconclusive_scale",
        )
    )
)
_ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES: Final = tuple(
    sorted(
        set(_ABSTAIN_REASON_MAP)
        - set(_ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS)
        - set(_ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES)
    )
)
if (
    set(_ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS)
    | set(_ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES)
    | set(_ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES)
) != set(_ABSTAIN_REASON_MAP) or any(
    (
        set(_ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS)
        & set(_ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES),
        set(_ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS)
        & set(_ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES),
        set(_ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES)
        & set(_ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES),
    )
):
    raise RuntimeError("V2 prediction abstention stage classification drift")

_FROZEN_FAMILY_MAPPING_VALUES: Final = tuple(
    (kind.value, family.value)
    for kind, family in _FROZEN_EXACT_FREEZE.family_mapping
)
_EPHEMERAL_MECHANICS_TRUE: Final = (
    "compact_typed_replay",
    "public_registry_adapter",
    "exact_derived_bridge",
    "closed_prediction_mapping",
)
_NON_EVIDENCE_CLAIMS: Final = (
    "input_archive_membership",
    "batch_policy_membership",
    "execution_manifest_authority",
    "recognizer_executed",
    "runtime_executed",
    "recognizer_capacity",
    "prediction_scored",
    "effect",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "c1_exit",
)
PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID: Final = stable_hash(
    {
        "version": PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION,
        "fields": _OUTCOME_FIELDS,
        "decision_values": tuple(item.value for item in PredictionDecisionV2),
        "prediction_bundle_schema_version": PREDICTION_SCHEMA_VERSION,
        "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
        "contract": (
            "exact_closed_fields",
            "privately_issued",
            "ephemeral_in_process",
            "no_durable_receipt_id",
        ),
    },
    prefix="phase2b_public_recognizer_prediction_outcome_schema_v2_",
)
_PREDICTION_ROW_POLICY_VALUE_V2: Final = {
    "schema_version": PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION,
    "schema_id": PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID,
    "outcome_fields": _OUTCOME_FIELDS,
    "outcome_contract": (
        "ephemeral_in_process",
        "privately_issued",
        "no_durable_receipt_id",
        "non_authoritative_mechanics_only",
    ),
    "public_api": (
        "recognize_public_input_row_v2_exact_keyword_only_row_and_freeze_manifest"
    ),
    "validation_order": (
        "exact_input_and_freeze_types",
        "exact_current_freeze_binding_before_deep_row_replay",
        "cheap_exact_row_roots_and_registry_shape_before_deep_row_replay",
        "one_public_compact_v2_typed_replay_API_operation",
        "local_exact_row_root_registry_scope_and_alias_parity_against_returned_replay_without_calling_row_or_typed_zero_argument_validation",
        "public_registry_exact_adapter",
        "existing_exact_derived_bridge",
        "closed_v2_decision_and_generic_prediction_bundle_mapping",
    ),
    "ephemeral_call_mechanics": _EPHEMERAL_MECHANICS_TRUE,
    "validated_replay_reuse": {
        "public_typed_replay_api_call_count": 1,
        "typed_zero_argument_validation_after_decode": False,
        "input_row_zero_argument_validation_after_decode": False,
        "upstream_structural_replay_count": (
            "bounded_performance_property_of_committed_typed_replay_v2_not_a_claim"
        ),
        "local_parity_checks": (
            "exact_DecodedTrustedEnvelopeV2",
            "exact_envelope_bytes_identity",
            "returned_payload_and_padding_lengths_equal_exact_envelope_header_partition",
            "exact_V1_logical_typed_oracle_resource_validation_before_authority_content_hash_or_UUID_scan",
            "seven_stored_public_roots",
            "deep_public_registry_validation",
            "registry_authority_exact_scope",
            "fixed_alias_authority_role_quantity_noncollision",
            "row_root_recalculation",
        ),
        "decoded_envelope_claim_manifest": {
            "exact_true": _batch_v2._ENVELOPE_TRUE_CLAIMS_V2,
            "exact_false": _batch_v2._ENVELOPE_FALSE_CLAIMS_V2,
        },
        "trust_boundary": (
            "public_decoder_is_the_typed_authority_and_transform_replay_authority"
        ),
    },
    "non_evidence_claims": _NON_EVIDENCE_CLAIMS,
    "dependency_bindings": {
        "recognizer_input_archive_version": TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "recognizer_input_archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "public_registry_version": PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION,
        "public_registry_schema_id": PUBLIC_RECOGNIZER_REGISTRY_SCHEMA_ID_V2,
        "public_family_alias_policy_id": PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2,
        "typed_replay_version": TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
        "typed_replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
        "compact_typed_authority_schema_id": COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        "compact_typed_authority_codec_version": COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        "compact_typed_authority_codec_policy_id": COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
        "trusted_wire_batch_schema_version": TRUSTED_WIRE_BATCH_V2_SCHEMA_VERSION,
        "trusted_wire_batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "prediction_bundle_schema_version": PREDICTION_SCHEMA_VERSION,
        "frozen_protocol_id": _FROZEN_PROTOCOL.protocol_id,
        "frozen_theory_version_id": _FROZEN_THEORY.version_id,
        "exact_freeze_id": _FROZEN_EXACT_FREEZE.freeze_id,
        "exact_freeze_family_mapping": _FROZEN_FAMILY_MAPPING_VALUES,
        "derived_bridge_policy_id": EXACT_DERIVED_BRIDGE_POLICY_ID,
        "derived_matcher_semantics_id": EXACT_DERIVED_MATCHER_SEMANTICS_ID,
        "derived_selection_policy_id": EXACT_DERIVED_SELECTION_POLICY_ID,
    },
    "decision_values": tuple(item.value for item in PredictionDecisionV2),
    "positive_source_reasons": {
        "complete_bridge_compilation": (
            "complete_exact_derived_witness_candidate_grid"
        ),
        "selected_decision": (
            "unique_structure_with_exact_derived_admissible_scales"
        ),
    },
    "abstention_reason_mapping": tuple(
        (source, target.value) for source, target in _ABSTAIN_REASON_PAIRS
    ),
    "abstention_reason_source_classification": {
        "accepted_compilation_abstain_sources": (
            _ACCEPTED_COMPILATION_ABSTAIN_SOURCES
        ),
        "accepted_compilation_decision_reason_keys": (
            _ACCEPTED_COMPILATION_ABSTAIN_REASON_KEYS
        ),
        "accepted_complete_selection_abstain_sources": (
            _ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES
        ),
        "accepted_preflight_abstain_sources": _ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES,
        "accepted_resource": _EXPECTED_RESOURCE_REASON_SOURCES,
        "accepted_row_semantic": _ROW_SEMANTIC_REASON_SOURCES,
        "rejected_integrity_or_dead": tuple(
            sorted(_INTEGRITY_OR_DEAD_REASON_SOURCES_REJECTED)
        ),
        "unknown_or_nonexact_source": "fail_closed",
    },
    "preflight_outcome": {
        "domain_hex": _PREFLIGHT_DOMAIN_V2.hex(),
        "fields": _PREFLIGHT_OUTCOME_FIELDS,
        "prefix": "phase2b_exact_derived_preflight_v2_",
        "exact_input_bindings": (
            "bundle_id_equals_replayed_authority_bundle_id",
            "wrapper_schema_equals_replayed_authority_schema",
            "theory_schema_equals_frozen_theory_schema",
            "registry_theory_version_equals_public_registry_theory_version",
        ),
    },
    "cross_version_rejection": (
        "TrustedRecognizerInputRowV1",
        "PredictionDecisionV1",
        "PublicRecognizerPredictionOutcomeV1",
    ),
}
RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2: Final = stable_hash(
    _PREDICTION_ROW_POLICY_VALUE_V2,
    prefix="phase2b_recognizer_prediction_row_policy_v2_",
)

_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _ascii(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > MAXIMUM_ASCII_STRING_BYTES
        or not value.isascii()
    ):
        raise ValueError(f"{name} must use exact bounded nonempty ASCII")
    return value


def _uuid4(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 36
        or _UUID4.fullmatch(value) is None
        or UUID(value).version != 4
    ):
        raise ValueError(f"{name} must use canonical lowercase UUIDv4")
    return value


def _hex64(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise ValueError(f"{name} must use exact lowercase SHA-256")
    return value


def _digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or len(value) != len(prefix) + 64:
        raise ValueError(f"{name} must use an exact prefixed SHA-256")
    if not value.startswith(prefix) or any(
        item not in "0123456789abcdef" for item in value[len(prefix) :]
    ):
        raise ValueError(f"{name} content ID drift")
    return value


def _closed(value: object, manifest: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must use an exact mapping")
    if (
        len(value) != len(manifest)
        or any(type(key) is not str for key in value)
        or set(value) != set(manifest)
    ):
        raise ValueError(f"{name} closed schema drift")
    return value


def _validate_execution_freeze_manifest(manifest: ExecutionFreezeManifest) -> None:
    """Reject polluted or stale freeze objects before any V2 row replay."""

    if type(manifest) is not ExecutionFreezeManifest:
        raise TypeError("V2 prediction requires exact ExecutionFreezeManifest")
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
        _ascii(getattr(manifest, name), f"V2 execution manifest {name}")
    if (
        type(manifest.baseline_registrations) is not tuple
        or len(manifest.baseline_registrations) != len(BaselineKind)
    ):
        raise TypeError("V2 execution manifest baseline tuple drift")
    for registration in manifest.baseline_registrations:
        if type(registration) is not BaselineRegistration:
            raise TypeError("V2 execution baseline must use exact registration type")
        if type(registration.kind) is not BaselineKind:
            raise TypeError("V2 execution baseline kind exact enum drift")
        _ascii(registration.baseline_spec_id, "V2 baseline specification ID")
        _ascii(registration.implementation_id, "V2 baseline implementation ID")
        _hex64(registration.artifact_sha256, "V2 baseline artifact SHA-256")
        if type(registration.frozen_before_holdout_generation) is not bool:
            raise TypeError("V2 baseline frozen flag exact bool drift")
        registration.__post_init__()
    manifest.__post_init__()
    if (
        manifest.protocol_id != _FROZEN_PROTOCOL.protocol_id
        or manifest.exact_freeze_id != _FROZEN_EXACT_FREEZE.freeze_id
        or manifest.theory_version_id != _FROZEN_THEORY.version_id
    ):
        raise ValueError("V2 execution manifest does not bind current authorities")
    _digest(
        manifest.manifest_id,
        "phase2b_execution_freeze_",
        "V2 execution manifest ID",
    )


def _validate_input_row_identity_v2(input_row: TrustedRecognizerInputRowV2) -> None:
    """Close cheap exact row fields before touching fixed-envelope bytes."""

    if type(input_row) is not TrustedRecognizerInputRowV2:
        raise TypeError("V2 prediction row must use exact input type")
    if type(input_row.envelope) is not bytes or len(input_row.envelope) != _archive_v2.ENVELOPE_BYTES:
        raise TypeError("V2 prediction row envelope exact length drift")
    _digest(input_row.envelope_id, "phase2b_trusted_envelope_v2_", "V2 row envelope ID")
    _hex64(input_row.payload_sha256, "V2 row payload SHA-256")
    _hex64(input_row.padding_sha256, "V2 row padding SHA-256")
    _digest(input_row.namespace_audit_id, "phase2b_namespace_audit_v2_", "V2 row namespace ID")
    _digest(input_row.authority_content_id, "phase2b_public_transform_evidence_", "V2 row authority ID")
    _digest(input_row.transform_result_id, "phase2b_exact_transform_result_", "V2 row transform ID")
    _digest(input_row.public_registry_id, "phase2b_public_recognizer_registry_v2_", "V2 row registry ID")
    _digest(input_row.row_id, "phase2b_recognizer_input_row_v2_", "V2 row ID")
    magic, version, header_bytes, payload_bytes, payload_hash, padding_hash = (
        _batch_v2._HEADER_V2.unpack(
            input_row.envelope[: _batch_v2.ENVELOPE_HEADER_BYTES]
        )
    )
    if (
        magic != _batch_v2.TRUSTED_WIRE_ENVELOPE_V2_MAGIC
        or version != _batch_v2.TRUSTED_WIRE_ENVELOPE_V2_VERSION
        or header_bytes != _batch_v2.ENVELOPE_HEADER_BYTES
        or type(payload_bytes) is not int
        or not 0 <= payload_bytes <= _batch_v2.MAXIMUM_PAYLOAD_BYTES
    ):
        raise ValueError("V2 prediction row envelope shallow header drift")
    if (
        input_row.envelope_id
        != "phase2b_trusted_envelope_v2_"
        + hashlib.sha256(
            _batch_v2._ENVELOPE_ID_DOMAIN_V2 + input_row.envelope
        ).hexdigest()
        or input_row.payload_sha256 != payload_hash.hex()
        or input_row.padding_sha256 != padding_hash.hex()
    ):
        raise ValueError("V2 prediction row envelope shallow roots drift")
    registry = input_row.public_registry
    if type(registry) is not PublicRecognizerRegistryV2:
        raise TypeError("V2 prediction row registry exact type drift")
    if (
        type(registry.schema_version) is not str
        or registry.schema_version != PUBLIC_RECOGNIZER_REGISTRY_V2_SCHEMA_VERSION
        or type(registry.theory_version_id) is not str
        or registry.theory_version_id != _FROZEN_THEORY.version_id
        or type(registry.family_alias_policy_id) is not str
        or registry.family_alias_policy_id != PUBLIC_RECOGNIZER_FAMILY_ALIAS_POLICY_ID_V2
    ):
        raise ValueError("V2 prediction row registry identity drift")
    if (
        type(registry.law_bindings) is not tuple
        or len(registry.law_bindings) != len(LawKind)
        or type(registry.observable_channels) is not tuple
        or not 1 <= len(registry.observable_channels) <= _archive_v2.MAXIMUM_ARRAY_ENTRIES
        or type(registry.maximum_candidate_count) is not int
        or registry.maximum_candidate_count != 50_000
    ):
        raise TypeError("V2 prediction row registry shallow shape drift")
    # Registry validation is bounded independently of the 65 KiB envelope and
    # closes all nested shapes before its root is accepted.
    registry._validate()
    if registry.registry_id != input_row.public_registry_id:
        raise ValueError("V2 prediction row stored registry root drift")
    if _archive_v2._row_id_v2(input_row) != input_row.row_id:
        raise ValueError("V2 prediction row stored row root drift")


def _validate_input_row_against_typed_v2(
    input_row: TrustedRecognizerInputRowV2,
    typed: DecodedTrustedEnvelopeV2,
) -> None:
    """Check returned public replay parity without another zero-arg validate."""

    if type(input_row) is not TrustedRecognizerInputRowV2:
        raise TypeError("V2 replay parity requires exact input row")
    if type(typed) is not DecodedTrustedEnvelopeV2:
        raise TypeError("V2 replay parity requires exact decoded envelope")
    if type(typed.authority) is not PublicTransformEvidenceBundleV2:
        raise TypeError("V2 replay parity authority exact type drift")
    logical_profile = _archive_v2.encode_typed_transform_authority_profile_v1(
        typed.authority
    )
    if typed.authority.content_id != typed.authority_content_id:
        raise ValueError("V2 replay parity authority content root drift")
    if type(typed.envelope) is not bytes or typed.envelope != input_row.envelope:
        raise ValueError("V2 replay parity envelope drift")
    _, _, _, header_payload_bytes, _, _ = _batch_v2._HEADER_V2.unpack(
        typed.envelope[: _batch_v2.ENVELOPE_HEADER_BYTES]
    )
    if (
        type(typed.payload_bytes) is not int
        or type(typed.padding_bytes) is not int
        or typed.payload_bytes != header_payload_bytes
        or typed.padding_bytes
        != _batch_v2.ENVELOPE_BYTES
        - _batch_v2.ENVELOPE_HEADER_BYTES
        - header_payload_bytes
        or not 0 <= typed.payload_bytes <= _batch_v2.MAXIMUM_PAYLOAD_BYTES
        or typed.padding_bytes < _batch_v2.MINIMUM_PADDING_BYTES
        or _batch_v2.ENVELOPE_HEADER_BYTES
        + typed.payload_bytes
        + typed.padding_bytes
        != _batch_v2.ENVELOPE_BYTES
    ):
        raise ValueError("V2 replay parity envelope length drift")
    if type(typed.namespace_audit) is not _batch_v2.NamespaceFieldAuditV1:
        raise TypeError("V2 replay parity namespace audit exact type drift")
    if _batch_v2._namespace_audit_id_v2(typed.namespace_audit) != typed.namespace_audit_id:
        raise ValueError("V2 replay parity namespace audit root drift")
    expected_identities = (
        _batch_v2.TRUSTED_WIRE_BATCH_V2_PAYLOAD_SCHEMA_VERSION,
        _batch_v2._batch_v1.TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
        COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
        COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
        COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
    )
    actual_identities = (
        typed.payload_schema_version,
        typed.public_provenance_version,
        typed.typed_authority_schema_id,
        typed.typed_authority_codec_version,
        typed.typed_authority_codec_policy_id,
    )
    if any(type(item) is not str for item in actual_identities):
        raise TypeError("V2 replay parity discriminator exact text drift")
    if actual_identities != expected_identities:
        raise ValueError("V2 replay parity discriminator drift")
    for name in (
        *_batch_v2._ENVELOPE_TRUE_CLAIMS_V2,
        *_batch_v2._ENVELOPE_FALSE_CLAIMS_V2,
    ):
        if type(getattr(typed, name)) is not bool:
            raise TypeError("V2 replay parity claim exact bool drift")
    if not all(
        getattr(typed, name) for name in _batch_v2._ENVELOPE_TRUE_CLAIMS_V2
    ) or any(
        getattr(typed, name) for name in _batch_v2._ENVELOPE_FALSE_CLAIMS_V2
    ):
        raise ValueError("V2 replay parity claim boundary drift")
    registry = input_row.public_registry
    if type(registry) is not PublicRecognizerRegistryV2:
        raise TypeError("V2 replay parity registry exact type drift")
    registry._validate()
    if (
        typed.envelope_id,
        typed.payload_sha256,
        typed.padding_sha256,
        typed.namespace_audit_id,
        typed.authority_content_id,
        typed.transform_result_id,
        registry.registry_id,
    ) != (
        input_row.envelope_id,
        input_row.payload_sha256,
        input_row.padding_sha256,
        input_row.namespace_audit_id,
        input_row.authority_content_id,
        input_row.transform_result_id,
        input_row.public_registry_id,
    ):
        raise ValueError("V2 replay parity stored public roots drift")
    _archive_v2._validate_registry_authority_scope_v2(registry, typed.authority)
    roles, quantities = _archive_v2._registry_scope_v2(registry)
    authority_uuids = _archive_v2._profile_uuid4_values_v2(
        logical_profile
    )
    aliases = set(_BRIDGE_FAMILY_BY_KIND.values())
    if aliases & (authority_uuids | set(roles) | set(quantities)):
        raise ValueError("V2 replay parity fixed alias collision")
    if _archive_v2._row_id_v2(input_row) != input_row.row_id:
        raise ValueError("V2 replay parity row root drift")


def _prediction_mapping_unchecked(value: PredictionBundle) -> dict[str, object]:
    mapping = value.to_mapping()
    if type(mapping) is not dict:
        raise TypeError("V2 PredictionBundle mapping must use exact dict")
    return mapping


def _validate_prediction_bundle_object(value: PredictionBundle) -> None:
    if type(value) is not PredictionBundle:
        raise TypeError("V2 outcome requires exact PredictionBundle")
    if type(value.schema_version) is not str or value.schema_version != PREDICTION_SCHEMA_VERSION:
        raise ValueError("V2 outcome prediction schema drift")
    _uuid4(value.bundle_id, "V2 prediction bundle ID")
    for name in ("input_root_sha256", "protocol_sha256", "freeze_manifest_sha256"):
        _hex64(getattr(value, name), f"V2 prediction {name}")
    if type(value.disposition) is not PredictionDisposition:
        raise TypeError("V2 prediction disposition exact enum drift")
    if type(value.reason) is not PredictionReason:
        raise TypeError("V2 prediction reason exact enum drift")
    if value.family_id is not None:
        _uuid4(value.family_id, "V2 prediction family ID")
    if type(value.binding) is not tuple or len(value.binding) > 64:
        raise TypeError("V2 prediction binding exact bounded tuple drift")
    for item in value.binding:
        if type(item) is not RoleBinding:
            raise TypeError("V2 prediction binding needs exact RoleBinding")
        _uuid4(item.role_id, "V2 prediction role ID")
        _uuid4(item.entity_id, "V2 prediction entity ID")
    if type(value.admissible_scale_ids) is not tuple or len(value.admissible_scale_ids) > 4_096:
        raise TypeError("V2 prediction scales exact bounded tuple drift")
    for item in value.admissible_scale_ids:
        _uuid4(item, "V2 prediction scale ID")


def _decode_prediction_bundle(mapping: object) -> PredictionBundle:
    root = _closed(mapping, _PREDICTION_BUNDLE_FIELDS, "V2 prediction bundle")
    if (
        type(root["schema_version"]) is not str
        or root["schema_version"] != PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("V2 prediction bundle schema drift")
    _uuid4(root["bundle_id"], "V2 prediction bundle ID")
    for name in ("input_root_sha256", "protocol_sha256", "freeze_manifest_sha256"):
        _hex64(root[name], f"V2 prediction {name}")
    if type(root["disposition"]) is not str or root["disposition"] not in {
        item.value for item in PredictionDisposition
    }:
        raise ValueError("V2 prediction disposition drift")
    if type(root["reason"]) is not str or root["reason"] not in {
        item.value for item in PredictionReason
    }:
        raise ValueError("V2 prediction reason drift")
    if root["family_id"] is not None:
        _uuid4(root["family_id"], "V2 prediction family ID")
    raw_bindings = root["binding"]
    if type(raw_bindings) is not list or len(raw_bindings) > 64:
        raise TypeError("V2 prediction binding wire drift")
    for raw in raw_bindings:
        binding = _closed(raw, _ROLE_BINDING_FIELDS, "V2 prediction role binding")
        _uuid4(binding["role_id"], "V2 prediction role ID")
        _uuid4(binding["entity_id"], "V2 prediction entity ID")
    raw_scales = root["admissible_scale_ids"]
    if type(raw_scales) is not list or len(raw_scales) > 4_096:
        raise TypeError("V2 prediction scale wire drift")
    for scale in raw_scales:
        _uuid4(scale, "V2 prediction scale ID")
    prediction = PredictionBundle.from_mapping(root)
    if type(prediction) is not PredictionBundle:
        raise TypeError("V2 prediction decoder returned nonexact bundle")
    if _prediction_mapping_unchecked(prediction) != root:
        raise ValueError("V2 prediction canonical roundtrip drift")
    return prediction


def _prediction_mapping(value: PredictionBundle) -> dict[str, object]:
    _validate_prediction_bundle_object(value)
    mapping = _prediction_mapping_unchecked(value)
    if _decode_prediction_bundle(mapping) != value:
        raise ValueError("V2 PredictionBundle structural pollution")
    return mapping


def _bridge_outcome_digest(value: object) -> str:
    if type(value) is not str:
        raise ValueError("V2 bridge outcome ID exact text drift")
    for prefix in (
        "phase2b_exact_derived_run_",
        "phase2b_exact_derived_preflight_v2_",
    ):
        if value.startswith(prefix):
            return _digest(value, prefix, "V2 bridge outcome ID")
    raise ValueError("V2 bridge outcome ID prefix drift")


def _preflight_outcome_id(value: ExactDerivedBridgePreflightRejection) -> str:
    if type(value) is not ExactDerivedBridgePreflightRejection:
        raise TypeError("V2 preflight root needs exact rejection type")
    value.__post_init__()
    if value.disposition is not ExactBridgeDisposition.ABSTAIN:
        raise ValueError("V2 preflight outcome must abstain")
    raw = tuple(getattr(value, name) for name in _PREFLIGHT_OUTCOME_FIELDS)
    items = (value.disposition.value, *raw[1:])
    for index, item in enumerate(items):
        _ascii(item, f"V2 preflight outcome field {index}")
    payload = encode_phase2b_jcs_profile_v1(items)
    return "phase2b_exact_derived_preflight_v2_" + hashlib.sha256(
        _PREFLIGHT_DOMAIN_V2 + payload
    ).hexdigest()


def _validate_family_and_decision(
    *,
    decision: PredictionDecisionV2,
    canonical_family_id: CanonicalFamilyId | None,
    prediction: PredictionBundle,
) -> None:
    if type(decision) is not PredictionDecisionV2:
        raise TypeError("V2 outcome decision must use exact V2 enum")
    if decision is PredictionDecisionV2.ABSTAIN:
        if (
            canonical_family_id is not None
            or prediction.disposition is not PredictionDisposition.ABSTAIN
            or type(prediction.reason) is not PredictionReason
            or prediction.reason not in _ALLOWED_ABSTAIN_REASONS
        ):
            raise ValueError("V2 abstention carries an unissued positive field or reason")
        return
    if type(canonical_family_id) is not CanonicalFamilyId:
        raise TypeError("V2 answering outcome needs exact canonical family enum")
    law_kind = _KIND_BY_CANONICAL_FAMILY.get(canonical_family_id)
    if law_kind is None or prediction.family_id != _BRIDGE_FAMILY_BY_KIND[law_kind]:
        raise ValueError("V2 prediction bridge/canonical family mapping drift")
    if (
        prediction.disposition is not PredictionDisposition.UNIQUE_MATCH
        or prediction.reason is not PredictionReason.UNIQUE_STRUCTURAL_MATCH
    ):
        raise ValueError("V2 answering prediction bundle disposition drift")
    scale_count = len(prediction.admissible_scale_ids)
    if decision is PredictionDecisionV2.ANSWER and scale_count != 1:
        raise ValueError("V2 ANSWER requires exactly one admissible scale")
    if decision is PredictionDecisionV2.ANSWER_SET and scale_count <= 1:
        raise ValueError("V2 ANSWER_SET requires multiple admissible scales")


@dataclass(frozen=True, slots=True, init=False)
class PublicRecognizerPredictionOutcomeV2:
    """Ephemeral in-process mapping result; never a durable trust receipt."""

    input_row_id: str
    input_payload_sha256: str
    decision: PredictionDecisionV2
    canonical_family_id: CanonicalFamilyId | None
    prediction: PredictionBundle
    bridge_outcome_id: str
    bridge_compilation_id: str | None
    bridge_decision_id: str | None
    claim_level: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 public prediction outcomes are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        input_row_id: str,
        input_payload_sha256: str,
        decision: PredictionDecisionV2,
        canonical_family_id: CanonicalFamilyId | None,
        prediction: PredictionBundle,
        bridge_outcome_id: str,
        bridge_compilation_id: str | None,
        bridge_decision_id: str | None,
    ) -> "PublicRecognizerPredictionOutcomeV2":
        if token is not _OUTCOME_ISSUE_TOKEN:
            raise TypeError("V2 prediction outcome issuer token drift")
        value = object.__new__(cls)
        frozen = (
            ("input_row_id", input_row_id),
            ("input_payload_sha256", input_payload_sha256),
            ("decision", decision),
            ("canonical_family_id", canonical_family_id),
            ("prediction", prediction),
            ("bridge_outcome_id", bridge_outcome_id),
            ("bridge_compilation_id", bridge_compilation_id),
            ("bridge_decision_id", bridge_decision_id),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not PublicRecognizerPredictionOutcomeV2:
            raise TypeError("V2 prediction outcome exact type drift")
        _digest(
            self.input_row_id,
            "phase2b_recognizer_input_row_v2_",
            "V2 outcome input row ID",
        )
        _hex64(self.input_payload_sha256, "V2 outcome input payload SHA-256")
        if type(self.decision) is not PredictionDecisionV2:
            raise TypeError("V2 outcome decision exact enum drift")
        prediction = _decode_prediction_bundle(_prediction_mapping(self.prediction))
        if prediction.input_root_sha256 != self.input_payload_sha256:
            raise ValueError("V2 outcome prediction input root drift")
        _validate_family_and_decision(
            decision=self.decision,
            canonical_family_id=self.canonical_family_id,
            prediction=prediction,
        )
        _bridge_outcome_digest(self.bridge_outcome_id)
        if (self.bridge_compilation_id is None) != (self.bridge_decision_id is None):
            raise ValueError("V2 outcome bridge roots are partial")
        if self.bridge_compilation_id is not None:
            _digest(
                self.bridge_compilation_id,
                "phase2b_exact_derived_bridge_result_",
                "V2 bridge compilation ID",
            )
            _digest(
                self.bridge_decision_id,
                "phase2b_exact_derived_decision_",
                "V2 bridge decision ID",
            )
        if type(self.claim_level) is not str or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL:
            raise ValueError("V2 outcome claim level drift")


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
    input_row: TrustedRecognizerInputRowV2,
    typed: DecodedTrustedEnvelopeV2,
    execution_freeze_manifest_id: str,
    disposition: PredictionDisposition,
    reason: PredictionReason,
    family_id: str | None,
    binding: tuple[RoleBinding, ...],
    admissible_scale_ids: tuple[str, ...],
) -> PredictionBundle:
    if type(input_row) is not TrustedRecognizerInputRowV2:
        raise TypeError("V2 bundle mapping requires exact input row")
    if type(typed) is not DecodedTrustedEnvelopeV2:
        raise TypeError("V2 bundle mapping requires exact typed replay")
    _digest(
        execution_freeze_manifest_id,
        "phase2b_execution_freeze_",
        "V2 bundle execution manifest ID",
    )
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
    public_registry: PublicRecognizerRegistryV2,
) -> tuple[LawKind, CanonicalFamilyId, str, tuple[RoleBinding, ...], tuple[str, ...]]:
    if type(bridge_run) is not ExactDerivedBridgeRun:
        raise TypeError("V2 selected binding requires exact bridge run")
    if type(public_registry) is not PublicRecognizerRegistryV2:
        raise TypeError("V2 selected binding requires exact public registry")
    decision = bridge_run.decision
    if (
        bridge_run.disposition is not ExactBridgeDisposition.COMPLETE
        or bridge_run.reason != "complete_exact_derived_witness_candidate_grid"
        or bridge_run.compilation.reason
        != "complete_exact_derived_witness_candidate_grid"
        or decision.reason
        != "unique_structure_with_exact_derived_admissible_scales"
    ):
        raise _PredictionGateRejected("positive_derived_reason_closure_drift")
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
        or selected_law.canonical_family_id is not _CANONICAL_FAMILY_BY_KIND[law_kind]
    ):
        raise _PredictionGateRejected("selected_public_family_mapping_drift")
    if (
        type(decision.selected_role_binding) is not tuple
        or not decision.selected_role_binding
        or decision.selected_role_binding != tuple(sorted(decision.selected_role_binding))
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
    if type(scales) is not tuple or not scales or scales != tuple(sorted(set(scales))):
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
    input_row: TrustedRecognizerInputRowV2,
    typed: DecodedTrustedEnvelopeV2,
    execution_freeze_manifest_id: str,
    bridge_result: ExactDerivedBridgeRun | ExactDerivedBridgePreflightRejection,
) -> PublicRecognizerPredictionOutcomeV2:
    if type(input_row) is not TrustedRecognizerInputRowV2:
        raise TypeError("V2 prediction mapping needs exact input row")
    if type(typed) is not DecodedTrustedEnvelopeV2:
        raise TypeError("V2 prediction mapping needs exact typed replay")
    if typed.envelope != input_row.envelope:
        raise _PredictionGateRejected("typed_row_envelope_drift")
    _digest(
        execution_freeze_manifest_id,
        "phase2b_execution_freeze_",
        "V2 prediction execution manifest ID",
    )
    if type(bridge_result) is ExactDerivedBridgePreflightRejection:
        if (
            bridge_result.bridge_policy_id != EXACT_DERIVED_BRIDGE_POLICY_ID
            or bridge_result.matcher_semantics_id != EXACT_DERIVED_MATCHER_SEMANTICS_ID
            or bridge_result.bundle_id != typed.authority.base_bundle.bundle_id
            or bridge_result.wrapper_schema_version != typed.authority.schema_version
            or bridge_result.theory_schema_version != _FROZEN_THEORY.schema_version
            or bridge_result.registry_theory_version_id
            != input_row.public_registry.theory_version_id
        ):
            raise _PredictionGateRejected("derived_preflight_input_or_policy_binding_drift")
        if bridge_result.reason not in _ACCEPTED_PREFLIGHT_ABSTAIN_SOURCES:
            raise _PredictionGateRejected("derived_preflight_reason_stage_drift")
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
        return PublicRecognizerPredictionOutcomeV2._issue(
            _OUTCOME_ISSUE_TOKEN,
            input_row_id=input_row.row_id,
            input_payload_sha256=input_row.payload_sha256,
            decision=PredictionDecisionV2.ABSTAIN,
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
        or bridge_result.matcher_semantics_id != EXACT_DERIVED_MATCHER_SEMANTICS_ID
        or bridge_result.decision.selection_policy_id != EXACT_DERIVED_SELECTION_POLICY_ID
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
        if bridge_result.disposition is ExactBridgeDisposition.ABSTAIN:
            if (
                bridge_result.compilation.reason
                not in _ACCEPTED_COMPILATION_ABSTAIN_SOURCES
                or bridge_result.decision.reason
                != "bridge_" + bridge_result.compilation.reason
            ):
                raise _PredictionGateRejected(
                    "derived_compilation_abstention_reason_stage_drift"
                )
        elif bridge_result.disposition is ExactBridgeDisposition.COMPLETE:
            if (
                bridge_result.reason
                != "complete_exact_derived_witness_candidate_grid"
                or bridge_result.compilation.reason
                != "complete_exact_derived_witness_candidate_grid"
                or bridge_result.decision.reason
                not in _ACCEPTED_COMPLETE_SELECTION_ABSTAIN_SOURCES
            ):
                raise _PredictionGateRejected(
                    "derived_selection_abstention_reason_stage_drift"
                )
        else:
            raise _PredictionGateRejected("derived_abstention_disposition_drift")
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
        return PublicRecognizerPredictionOutcomeV2._issue(
            _OUTCOME_ISSUE_TOKEN,
            input_row_id=input_row.row_id,
            input_payload_sha256=input_row.payload_sha256,
            decision=PredictionDecisionV2.ABSTAIN,
            canonical_family_id=None,
            prediction=prediction,
            bridge_outcome_id=bridge_outcome_id,
            bridge_compilation_id=compilation_id,
            bridge_decision_id=decision_id,
        )
    if bridge_result.disposition is not ExactBridgeDisposition.COMPLETE:
        raise _PredictionGateRejected("positive_decision_from_incomplete_bridge")
    _, canonical_family_id, bridge_family_id, binding, scales = _selected_public_binding(
        bridge_run=bridge_result,
        public_registry=input_row.public_registry,
    )
    decision = (
        PredictionDecisionV2.ANSWER
        if len(scales) == 1
        else PredictionDecisionV2.ANSWER_SET
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
    return PublicRecognizerPredictionOutcomeV2._issue(
        _OUTCOME_ISSUE_TOKEN,
        input_row_id=input_row.row_id,
        input_payload_sha256=input_row.payload_sha256,
        decision=decision,
        canonical_family_id=canonical_family_id,
        prediction=prediction,
        bridge_outcome_id=bridge_outcome_id,
        bridge_compilation_id=compilation_id,
        bridge_decision_id=decision_id,
    )


def recognize_public_input_row_v2(
    *,
    input_row: TrustedRecognizerInputRowV2,
    execution_freeze_manifest: ExecutionFreezeManifest,
) -> PublicRecognizerPredictionOutcomeV2:
    """Map one exact public V2 row without issuing a durable receipt."""

    if type(input_row) is not TrustedRecognizerInputRowV2:
        raise TypeError("single-row V2 prediction requires exact input row")
    if type(execution_freeze_manifest) is not ExecutionFreezeManifest:
        raise TypeError("single-row V2 prediction requires exact freeze manifest")
    _validate_execution_freeze_manifest(execution_freeze_manifest)
    _validate_input_row_identity_v2(input_row)
    typed = decode_and_replay_typed_trusted_envelope_v2(input_row.envelope)
    if type(typed) is not DecodedTrustedEnvelopeV2:
        raise TypeError("single-row V2 typed replay result exact type drift")
    _validate_input_row_against_typed_v2(input_row, typed)
    adapter = input_row.public_registry.to_adapter_registry()
    bridge_result = run_exact_derived_witness_bridge(
        authority=typed.authority,
        theory=_FROZEN_THEORY,
        registry=adapter,
    )
    return _compile_prediction_outcome_from_bridge(
        input_row=input_row,
        typed=typed,
        execution_freeze_manifest_id=execution_freeze_manifest.manifest_id,
        bridge_result=bridge_result,
    )


def _assert_field_manifest() -> None:
    if tuple(item.name for item in fields(PublicRecognizerPredictionOutcomeV2)) != _OUTCOME_FIELDS:
        raise RuntimeError("V2 public prediction outcome field manifest drift")


_assert_field_manifest()


__all__ = (
    "PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_ID",
    "PUBLIC_RECOGNIZER_PREDICTION_OUTCOME_V2_SCHEMA_VERSION",
    "RECOGNIZER_PREDICTION_ROW_POLICY_ID_V2",
    "PredictionDecisionV2",
    "PublicRecognizerPredictionOutcomeV2",
    "recognize_public_input_row_v2",
)

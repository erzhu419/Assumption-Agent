"""Unsealed V2 structural partition replay for Phase-2B predictions.

The recognizer-facing V2 archive deliberately contains no main/challenge split
labels.  This evaluator accepts those labels only in a privately issued,
evaluator-side manifest and verifies that two sorted, unique 720/240 row-ID
partitions are disjoint and exhaust the exact row-ID set in one canonical V2
prediction archive.  Archive wire order is bound independently from the sorted
partition union because valid V2 archives need not be row-ID sorted.

Success is mechanics-only structural completeness.  This module has no answer
key, scorer, metric callback, runtime, membership authority, or effect claim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
from typing import Final

from .hashing import stable_hash
from .phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from .phase2b_recognizer_prediction_archive_v2 import (
    RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
    RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
    MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID,
    PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION,
    DecodedRecognizerPredictionArchiveV2,
    PredictionArchiveDispositionV2,
    PublicPredictionRunContextV2,
    PublicRecognizerPredictionRecordV2,
    decode_public_recognizer_prediction_archive_v2,
)
from .phase2b_recognizer_input_archive_v2 import (
    RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
    TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
)
from .phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT
from .phase2b_trusted_wire_batch_v2 import TRUSTED_WIRE_BATCH_V2_POLICY_ID
from .phase2b_trusted_wire_v1 import (
    MAXIMUM_ASCII_STRING_BYTES,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
)


UNSEALED_PREDICTION_EVALUATOR_V2_VERSION: Final = (
    "hegel-machine-phase2b-unsealed-prediction-evaluator/2"
)
_EXACT_FREEZE_V2: Final = frozen_phase2b_exact_freeze()
_EXACT_FREEZE_ID_V2: Final = _EXACT_FREEZE_V2.freeze_id
_MAIN_COUNT_V2: Final = _EXACT_FREEZE_V2.holdout.independent_latent_case_count
_SEMANTIC_CONFLICT_COUNT_V2: Final = _EXACT_FREEZE_V2.semantic_conflict.case_count
if (
    type(_EXACT_FREEZE_ID_V2) is not str
    or not _EXACT_FREEZE_ID_V2.startswith("phase2b_exact_freeze_")
    or type(_MAIN_COUNT_V2) is not int
    or _MAIN_COUNT_V2 != 720
    or type(_SEMANTIC_CONFLICT_COUNT_V2) is not int
    or _SEMANTIC_CONFLICT_COUNT_V2 != 240
    or _MAIN_COUNT_V2 + _SEMANTIC_CONFLICT_COUNT_V2
    != TOTAL_RECOGNIZER_CASE_COUNT
):
    raise RuntimeError("V2 unsealed evaluator exact count freeze drift")
if (
    type(
        _EXACT_FREEZE_V2.semantic_conflict.included_in_main_accuracy_denominator
    )
    is not bool
    or _EXACT_FREEZE_V2.semantic_conflict.included_in_main_accuracy_denominator
    or type(_EXACT_FREEZE_V2.semantic_conflict.threshold_tuning_allowed) is not bool
    or _EXACT_FREEZE_V2.semantic_conflict.threshold_tuning_allowed
    or type(_EXACT_FREEZE_V2.semantic_conflict.same_freeze_and_reveal_as_main)
    is not bool
    or not _EXACT_FREEZE_V2.semantic_conflict.same_freeze_and_reveal_as_main
):
    raise RuntimeError("V2 unsealed evaluator semantic-conflict freeze drift")

_MAIN_ROWS_DOMAIN_V2: Final = b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V2\x00"
_SEMANTIC_CONFLICT_ROWS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/UNSEALED/SEMANTIC_CONFLICT_ROWS/V2\x00"
)
_PARTITION_UNION_ROWS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/UNSEALED/PARTITION_UNION_ROWS/V2\x00"
)
_ORDERED_ARCHIVE_INPUT_ROWS_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/PREDICTION_INPUT_ROWS/V2\x00"
)
_MANIFEST_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/UNSEALED/PARTITION_MANIFEST/V2\x00"
)
_MANIFEST_ISSUE_TOKEN_V2: Final = object()
_EVALUATION_ISSUE_TOKEN_V2: Final = object()

_MANIFEST_FIELDS_V2: Final = (
    "prediction_archive_id",
    "prediction_archive_schema_version",
    "prediction_archive_policy_id",
    "exact_freeze_id",
    "evaluator_policy_id",
    "main_row_ids",
    "semantic_conflict_row_ids",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "ordered_archive_input_row_ids_root",
    "manifest_id",
)
_FALSE_EVALUATION_CLAIMS_V2: Final = (
    "challenge_in_main_denominator",
    "input_archive_membership_verified",
    "batch_policy_membership_verified",
    "source_registry_projection_verified",
    "source_public_disjoint_verified",
    "single_live_allocation_verified",
    "secret_custodian_replay_verified",
    "execution_manifest_authority_verified",
    "partition_manifest_authority_verified",
    "derived_mapping_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "scoring_performed",
    "prediction_scored",
    "effect_evidence",
    "c1_exit_evidence",
)
_EVALUATION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "prediction_archive_id",
    "prediction_archive_schema_version",
    "prediction_archive_policy_id",
    "partition_manifest_id",
    "exact_freeze_id",
    "evaluator_policy_id",
    "main_count",
    "semantic_conflict_count",
    "total_count",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "partition_union_row_ids_root",
    "ordered_archive_input_row_ids_root",
    "claim_level",
    "structural_completeness_verified",
    *_FALSE_EVALUATION_CLAIMS_V2,
    "metric_results",
    "scored_rows",
)
_REJECTION_FIELDS_V2: Final = (
    "disposition",
    "reason",
    "prediction_archive_id",
    "partition_manifest_id",
    "metric_results",
    "scored_rows",
    "structural_completeness_verified",
    "scoring_performed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "recognizer_capacity_evidence",
    "effect_evidence",
    "c1_exit_evidence",
)
_ARCHIVE_TRUE_CLAIMS_V2: Final = (
    "structural_archive_verified",
    "canonical_record_framing_verified",
    "record_schema_verified",
    "row_root_coverage_verified",
)
_ARCHIVE_FALSE_CLAIMS_V2: Final = (
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


class UnsealedPredictionEvaluationDispositionV2(str, Enum):
    STRUCTURALLY_COMPLETE_NOT_SCORED = "STRUCTURALLY_COMPLETE_NOT_SCORED"
    ABSTAIN = "ABSTAIN"


_SUCCESS_REASON_V2: Final = (
    "sorted_disjoint_exhaustive_720_240_same_v2_archive_row_set_and_ordered_root"
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


def _digest_v2(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError(f"{name} must end in exact lowercase SHA-256")
    return value


def _safe_digest_or_none_v2(
    value: object,
    prefix: str,
    name: str,
) -> str | None:
    try:
        return _digest_v2(value, prefix, name)
    except (TypeError, ValueError):
        return None


def _row_sequence_root_v2(
    values: tuple[str, ...],
    *,
    expected_count: int,
    domain: bytes,
    output_prefix: str,
    require_sorted_unique: bool,
    name: str,
) -> str:
    if (
        type(values) is not tuple
        or type(expected_count) is not int
        or len(values) != expected_count
    ):
        raise ValueError(f"{name} count drift")
    encoded_values: list[bytes] = []
    for value in values:
        _digest_v2(
            value,
            "phase2b_recognizer_input_row_v2_",
            f"{name} row ID",
        )
        encoded = value.encode("ascii")
        if len(encoded) > 65_535:
            raise ValueError(f"{name} row ID length drift")
        encoded_values.append(encoded)
    if require_sorted_unique and (
        values != tuple(sorted(values)) or len(set(values)) != expected_count
    ):
        raise ValueError(f"{name} rows are not sorted unique")
    if not require_sorted_unique and len(set(values)) != expected_count:
        raise ValueError(f"{name} rows are not unique")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(expected_count.to_bytes(4, "big"))
    for encoded in encoded_values:
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return output_prefix + digest.hexdigest()


def _main_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=_MAIN_COUNT_V2,
        domain=_MAIN_ROWS_DOMAIN_V2,
        output_prefix="phase2b_unsealed_main_rows_v2_",
        require_sorted_unique=True,
        name="V2 unsealed main partition",
    )


def _semantic_conflict_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=_SEMANTIC_CONFLICT_COUNT_V2,
        domain=_SEMANTIC_CONFLICT_ROWS_DOMAIN_V2,
        output_prefix="phase2b_unsealed_semantic_conflict_rows_v2_",
        require_sorted_unique=True,
        name="V2 unsealed semantic-conflict partition",
    )


def _partition_union_row_ids_root_v2(values: tuple[str, ...]) -> str:
    return _row_sequence_root_v2(
        values,
        expected_count=TOTAL_RECOGNIZER_CASE_COUNT,
        domain=_PARTITION_UNION_ROWS_DOMAIN_V2,
        output_prefix="phase2b_unsealed_partition_union_rows_v2_",
        require_sorted_unique=True,
        name="V2 unsealed sorted partition union",
    )


def _ordered_archive_input_row_ids_root_v2(values: tuple[str, ...]) -> str:
    """Independently replay the committed V2 ordered archive-row root."""

    return _row_sequence_root_v2(
        values,
        expected_count=TOTAL_RECOGNIZER_CASE_COUNT,
        domain=_ORDERED_ARCHIVE_INPUT_ROWS_DOMAIN_V2,
        output_prefix="phase2b_prediction_input_rows_v2_",
        require_sorted_unique=False,
        name="V2 ordered archive input rows",
    )


def _preflight_partitions_v2(
    *,
    main_row_ids: tuple[str, ...],
    semantic_conflict_row_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Close both evaluator-side partitions before hashing or archive replay."""

    if type(main_row_ids) is not tuple or type(semantic_conflict_row_ids) is not tuple:
        raise TypeError("V2 unsealed partition rows must use exact tuples")
    if (
        len(main_row_ids) != _MAIN_COUNT_V2
        or len(semantic_conflict_row_ids) != _SEMANTIC_CONFLICT_COUNT_V2
    ):
        raise ValueError("V2 unsealed partition requires exact 720/240 counts")
    for name, values, count in (
        ("main", main_row_ids, _MAIN_COUNT_V2),
        (
            "semantic-conflict",
            semantic_conflict_row_ids,
            _SEMANTIC_CONFLICT_COUNT_V2,
        ),
    ):
        for value in values:
            _digest_v2(
                value,
                "phase2b_recognizer_input_row_v2_",
                f"V2 unsealed {name} partition row ID",
            )
        if values != tuple(sorted(values)) or len(set(values)) != count:
            raise ValueError(f"V2 unsealed {name} partition is not sorted unique")
    if set(main_row_ids) & set(semantic_conflict_row_ids):
        raise ValueError("V2 unsealed partitions overlap")
    union_rows = tuple(sorted((*main_row_ids, *semantic_conflict_row_ids)))
    if len(union_rows) != TOTAL_RECOGNIZER_CASE_COUNT:
        raise ValueError("V2 unsealed partition union count drift")
    return union_rows


_digest_v2(
    _EXACT_FREEZE_ID_V2,
    "phase2b_exact_freeze_",
    "V2 unsealed evaluator exact freeze ID",
)
for _identity_value_v2, _identity_prefix_v2, _identity_name_v2 in (
    (
        RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
        "phase2b_recognizer_prediction_archive_policy_v2_",
        "V2 prediction archive policy ID",
    ),
    (
        TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "phase2b_trusted_wire_batch_v2_policy_",
        "V2 trusted-wire batch policy ID",
    ),
    (
        RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "phase2b_recognizer_input_archive_policy_v2_",
        "V2 recognizer input archive policy ID",
    ),
    (
        PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID,
        "phase2b_public_prediction_run_context_schema_v2_",
        "V2 public prediction run-context schema ID",
    ),
):
    _digest_v2(
        _identity_value_v2,
        _identity_prefix_v2,
        _identity_name_v2,
    )
for _version_value_v2, _version_name_v2 in (
    (
        UNSEALED_PREDICTION_EVALUATOR_V2_VERSION,
        "V2 unsealed evaluator version",
    ),
    (
        RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
        "V2 prediction archive version",
    ),
    (
        TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION,
        "V2 recognizer input archive version",
    ),
    (
        PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION,
        "V2 public prediction run-context schema version",
    ),
):
    _ascii_v2(_version_value_v2, _version_name_v2)


_EVALUATOR_POLICY_VALUE_V2: Final = {
    "version": UNSEALED_PREDICTION_EVALUATOR_V2_VERSION,
    "prediction_archive_schema_version": RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
    "prediction_archive_policy_id": RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
    "context_dependency_bindings": {
        "public_prediction_run_context_schema_version": (
            PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION
        ),
        "public_prediction_run_context_schema_id": (
            PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_ID
        ),
        "trusted_wire_batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
        "recognizer_input_archive_policy_id": RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2,
        "recognizer_input_archive_version": (
            TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
        ),
    },
    "prediction_archive_wire_caps": {
        "minimum_archive_bytes": PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        "maximum_archive_bytes": MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2,
    },
    "exact_freeze_id": _EXACT_FREEZE_ID_V2,
    "counts": {
        "main": _MAIN_COUNT_V2,
        "semantic_conflict": _SEMANTIC_CONFLICT_COUNT_V2,
        "total": TOTAL_RECOGNIZER_CASE_COUNT,
    },
    "semantic_conflict_contract": {
        "included_in_main_accuracy_denominator": False,
        "threshold_tuning_allowed": False,
        "same_freeze_and_reveal_as_main": True,
    },
    "domains": {
        "main_rows": _MAIN_ROWS_DOMAIN_V2.hex(),
        "semantic_conflict_rows": _SEMANTIC_CONFLICT_ROWS_DOMAIN_V2.hex(),
        "partition_union_rows": _PARTITION_UNION_ROWS_DOMAIN_V2.hex(),
        "ordered_archive_input_rows": _ORDERED_ARCHIVE_INPUT_ROWS_DOMAIN_V2.hex(),
        "manifest": _MANIFEST_DOMAIN_V2.hex(),
    },
    "field_manifests": {
        "partition_manifest": _MANIFEST_FIELDS_V2,
        "structural_evaluation": _EVALUATION_FIELDS_V2,
        "rejection": _REJECTION_FIELDS_V2,
    },
    "contract": {
        "local_partition_preflight_order": (
            "exact_tuple_types",
            "exact_720_240_counts",
            "exact_row_id_string_types_and_prefixed_sha256_closure",
            "each_partition_sorted_unique",
            "cross_partition_disjoint",
            "no_hash_or_public_archive_replay_before_closure",
        ),
        "supplied_archive_preflight_order": (
            "exact_decoded_V2_wrapper_type",
            "exact_archive_bytes_type_and_bounds",
            "exact_disposition_schema_policy_claim_level_and_bool_boundaries",
            "exact_context_type_and_identity_root_closure",
            "exact_record_and_three_column_tuple_counts_types_and_digest_prefixes",
            "stored_column_item_identity_parity",
            "one_public_V2_decode_of_original_bytes",
            "explicit_safe_scalar_and_tuple_parity",
            "canonical_decoder_result_exclusively_afterward",
        ),
        "public_prediction_archive_decode_call_count": 1,
        "supplied_prediction_archive_private_validate_or_parse_call_count": 0,
        "supplied_record_nonroot_fields": (
            "ignored_non_authoritative_canonical_decoder_records_exclusively_used"
        ),
        "set_exhaustiveness_checked_before_manifest_root_hash": True,
        "partition_rows": "exact_720_240_each_sorted_unique",
        "cross_partition": "disjoint_and_set_exhaustive_for_exact_archive_rows",
        "archive_order": "independently_replayed_and_bound_not_assumed_sorted",
        "labels": "evaluator_side_manifest_only_archive_bytes_unchanged",
        "success_true": ("structural_completeness_verified",),
        "success_false": _FALSE_EVALUATION_CLAIMS_V2,
        "metrics": "empty_no_scorer_no_answer_key",
    },
    "root_formulas": {
        "row_sequence": (
            "sha256(domain||u32be_count||repeated(u16be_ascii_length||row_id))"
        ),
        "partition_roots": (
            "main_and_semantic_conflict_use_sorted_exact_partition_order;"
            "partition_union_uses_sorted_union;ordered_archive_uses_wire_order"
        ),
        "manifest": (
            "sha256(manifest_domain||repeated(u16be_ascii_length||identity_or_root)"
            "||u32be_main_count||u32be_semantic_conflict_count||u32be_total_count)"
        ),
        "validate_before_hash": True,
    },
    "claim_level": NON_AUTHORITATIVE_CLAIM_LEVEL,
}
UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2: Final = stable_hash(
    _EVALUATOR_POLICY_VALUE_V2,
    prefix="phase2b_unsealed_prediction_evaluator_policy_v2_",
)


def _manifest_id_v2(
    *,
    prediction_archive_id: str,
    prediction_archive_schema_version: str,
    prediction_archive_policy_id: str,
    exact_freeze_id: str,
    evaluator_policy_id: str,
    main_row_ids_root: str,
    semantic_conflict_row_ids_root: str,
    partition_union_row_ids_root: str,
    ordered_archive_input_row_ids_root: str,
) -> str:
    _digest_v2(
        prediction_archive_id,
        "phase2b_recognizer_prediction_archive_v2_",
        "V2 unsealed manifest prediction archive ID",
    )
    if (
        type(prediction_archive_schema_version) is not str
        or prediction_archive_schema_version
        != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
    ):
        raise ValueError("V2 unsealed manifest archive schema drift")
    _digest_v2(
        prediction_archive_policy_id,
        "phase2b_recognizer_prediction_archive_policy_v2_",
        "V2 unsealed manifest archive policy ID",
    )
    _digest_v2(exact_freeze_id, "phase2b_exact_freeze_", "V2 exact freeze ID")
    _digest_v2(
        evaluator_policy_id,
        "phase2b_unsealed_prediction_evaluator_policy_v2_",
        "V2 unsealed evaluator policy ID",
    )
    _digest_v2(
        main_row_ids_root,
        "phase2b_unsealed_main_rows_v2_",
        "V2 main partition root",
    )
    _digest_v2(
        semantic_conflict_row_ids_root,
        "phase2b_unsealed_semantic_conflict_rows_v2_",
        "V2 semantic-conflict partition root",
    )
    _digest_v2(
        partition_union_row_ids_root,
        "phase2b_unsealed_partition_union_rows_v2_",
        "V2 partition-union root",
    )
    _digest_v2(
        ordered_archive_input_row_ids_root,
        "phase2b_prediction_input_rows_v2_",
        "V2 ordered archive input-row root",
    )
    digest = hashlib.sha256()
    digest.update(_MANIFEST_DOMAIN_V2)
    for value in (
        prediction_archive_id,
        prediction_archive_schema_version,
        prediction_archive_policy_id,
        exact_freeze_id,
        evaluator_policy_id,
        main_row_ids_root,
        semantic_conflict_row_ids_root,
        partition_union_row_ids_root,
        ordered_archive_input_row_ids_root,
    ):
        encoded = value.encode("ascii")
        if len(encoded) > 65_535:
            raise ValueError("V2 unsealed manifest identity length drift")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    digest.update(_MAIN_COUNT_V2.to_bytes(4, "big"))
    digest.update(_SEMANTIC_CONFLICT_COUNT_V2.to_bytes(4, "big"))
    digest.update(TOTAL_RECOGNIZER_CASE_COUNT.to_bytes(4, "big"))
    return "phase2b_unsealed_prediction_partition_v2_" + digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class UnsealedPredictionPartitionManifestV2:
    prediction_archive_id: str
    prediction_archive_schema_version: str
    prediction_archive_policy_id: str
    exact_freeze_id: str
    evaluator_policy_id: str
    main_row_ids: tuple[str, ...]
    semantic_conflict_row_ids: tuple[str, ...]
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    partition_union_row_ids_root: str
    ordered_archive_input_row_ids_root: str
    manifest_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 unsealed partition manifests are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        prediction_archive: DecodedRecognizerPredictionArchiveV2,
        main_row_ids: tuple[str, ...],
        semantic_conflict_row_ids: tuple[str, ...],
    ) -> "UnsealedPredictionPartitionManifestV2":
        if token is not _MANIFEST_ISSUE_TOKEN_V2:
            raise TypeError("V2 unsealed partition manifest issuer token mismatch")
        if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2:
            raise TypeError("V2 unsealed manifest requires exact prediction archive")
        union_rows = _preflight_partitions_v2(
            main_row_ids=main_row_ids,
            semantic_conflict_row_ids=semantic_conflict_row_ids,
        )
        main_root = _main_row_ids_root_v2(main_row_ids)
        conflict_root = _semantic_conflict_row_ids_root_v2(
            semantic_conflict_row_ids
        )
        union_root = _partition_union_row_ids_root_v2(union_rows)
        ordered_root = _ordered_archive_input_row_ids_root_v2(
            prediction_archive.input_row_ids
        )
        value = object.__new__(cls)
        frozen = (
            ("prediction_archive_id", prediction_archive.archive_id),
            (
                "prediction_archive_schema_version",
                RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
            ),
            (
                "prediction_archive_policy_id",
                RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
            ),
            ("exact_freeze_id", _EXACT_FREEZE_ID_V2),
            (
                "evaluator_policy_id",
                UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2,
            ),
            ("main_row_ids", main_row_ids),
            ("semantic_conflict_row_ids", semantic_conflict_row_ids),
            ("main_row_ids_root", main_root),
            ("semantic_conflict_row_ids_root", conflict_root),
            ("partition_union_row_ids_root", union_root),
            ("ordered_archive_input_row_ids_root", ordered_root),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        object.__setattr__(
            value,
            "manifest_id",
            _manifest_id_v2(
                prediction_archive_id=value.prediction_archive_id,
                prediction_archive_schema_version=(
                    value.prediction_archive_schema_version
                ),
                prediction_archive_policy_id=value.prediction_archive_policy_id,
                exact_freeze_id=value.exact_freeze_id,
                evaluator_policy_id=value.evaluator_policy_id,
                main_row_ids_root=value.main_row_ids_root,
                semantic_conflict_row_ids_root=(
                    value.semantic_conflict_row_ids_root
                ),
                partition_union_row_ids_root=value.partition_union_row_ids_root,
                ordered_archive_input_row_ids_root=(
                    value.ordered_archive_input_row_ids_root
                ),
            ),
        )
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not UnsealedPredictionPartitionManifestV2:
            raise TypeError("V2 unsealed partition manifest exact type drift")
        try:
            union_rows = _preflight_partitions_v2(
                main_row_ids=object.__getattribute__(self, "main_row_ids"),
                semantic_conflict_row_ids=object.__getattribute__(
                    self,
                    "semantic_conflict_row_ids",
                ),
            )
        except AttributeError as exc:
            raise ValueError("V2 unsealed manifest partition slot missing") from exc
        _digest_v2(
            self.prediction_archive_id,
            "phase2b_recognizer_prediction_archive_v2_",
            "V2 unsealed manifest archive ID",
        )
        if (
            type(self.prediction_archive_schema_version) is not str
            or self.prediction_archive_schema_version
            != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
            or type(self.prediction_archive_policy_id) is not str
            or self.prediction_archive_policy_id
            != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
            or type(self.exact_freeze_id) is not str
            or self.exact_freeze_id != _EXACT_FREEZE_ID_V2
            or type(self.evaluator_policy_id) is not str
            or self.evaluator_policy_id
            != UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2
        ):
            raise ValueError("V2 unsealed manifest frozen identity drift")
        for value, prefix, name in (
            (
                self.main_row_ids_root,
                "phase2b_unsealed_main_rows_v2_",
                "V2 main partition root",
            ),
            (
                self.semantic_conflict_row_ids_root,
                "phase2b_unsealed_semantic_conflict_rows_v2_",
                "V2 semantic-conflict partition root",
            ),
            (
                self.partition_union_row_ids_root,
                "phase2b_unsealed_partition_union_rows_v2_",
                "V2 partition-union root",
            ),
            (
                self.ordered_archive_input_row_ids_root,
                "phase2b_prediction_input_rows_v2_",
                "V2 ordered archive input-row root",
            ),
            (
                self.manifest_id,
                "phase2b_unsealed_prediction_partition_v2_",
                "V2 unsealed manifest ID",
            ),
        ):
            _digest_v2(value, prefix, name)
        main_root = _main_row_ids_root_v2(self.main_row_ids)
        conflict_root = _semantic_conflict_row_ids_root_v2(
            self.semantic_conflict_row_ids
        )
        union_root = _partition_union_row_ids_root_v2(union_rows)
        expected_manifest_id = _manifest_id_v2(
            prediction_archive_id=self.prediction_archive_id,
            prediction_archive_schema_version=self.prediction_archive_schema_version,
            prediction_archive_policy_id=self.prediction_archive_policy_id,
            exact_freeze_id=self.exact_freeze_id,
            evaluator_policy_id=self.evaluator_policy_id,
            main_row_ids_root=main_root,
            semantic_conflict_row_ids_root=conflict_root,
            partition_union_row_ids_root=union_root,
            ordered_archive_input_row_ids_root=(
                self.ordered_archive_input_row_ids_root
            ),
        )
        if (
            self.main_row_ids_root != main_root
            or self.semantic_conflict_row_ids_root != conflict_root
            or self.partition_union_row_ids_root != union_root
            or self.manifest_id != expected_manifest_id
        ):
            raise ValueError("V2 unsealed partition manifest root drift")


@dataclass(frozen=True, slots=True, init=False)
class UnsealedPredictionStructuralEvaluationV2:
    disposition: UnsealedPredictionEvaluationDispositionV2
    reason: str
    prediction_archive_id: str
    prediction_archive_schema_version: str
    prediction_archive_policy_id: str
    partition_manifest_id: str
    exact_freeze_id: str
    evaluator_policy_id: str
    main_count: int
    semantic_conflict_count: int
    total_count: int
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    partition_union_row_ids_root: str
    ordered_archive_input_row_ids_root: str
    claim_level: str
    structural_completeness_verified: bool
    challenge_in_main_denominator: bool
    input_archive_membership_verified: bool
    batch_policy_membership_verified: bool
    source_registry_projection_verified: bool
    source_public_disjoint_verified: bool
    single_live_allocation_verified: bool
    secret_custodian_replay_verified: bool
    execution_manifest_authority_verified: bool
    partition_manifest_authority_verified: bool
    derived_mapping_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    scoring_performed: bool
    prediction_scored: bool
    effect_evidence: bool
    c1_exit_evidence: bool
    metric_results: tuple[()]
    scored_rows: tuple[()]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 unsealed structural evaluations are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        prediction_archive: DecodedRecognizerPredictionArchiveV2,
        partition_manifest: UnsealedPredictionPartitionManifestV2,
    ) -> "UnsealedPredictionStructuralEvaluationV2":
        if token is not _EVALUATION_ISSUE_TOKEN_V2:
            raise TypeError("V2 unsealed evaluation issuer token mismatch")
        if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2:
            raise TypeError("V2 unsealed evaluation needs exact prediction archive")
        if type(partition_manifest) is not UnsealedPredictionPartitionManifestV2:
            raise TypeError("V2 unsealed evaluation needs exact partition manifest")
        value = object.__new__(cls)
        frozen = (
            (
                "disposition",
                UnsealedPredictionEvaluationDispositionV2.STRUCTURALLY_COMPLETE_NOT_SCORED,
            ),
            ("reason", _SUCCESS_REASON_V2),
            ("prediction_archive_id", prediction_archive.archive_id),
            (
                "prediction_archive_schema_version",
                RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION,
            ),
            (
                "prediction_archive_policy_id",
                RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2,
            ),
            ("partition_manifest_id", partition_manifest.manifest_id),
            ("exact_freeze_id", _EXACT_FREEZE_ID_V2),
            (
                "evaluator_policy_id",
                UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2,
            ),
            ("main_count", _MAIN_COUNT_V2),
            ("semantic_conflict_count", _SEMANTIC_CONFLICT_COUNT_V2),
            ("total_count", TOTAL_RECOGNIZER_CASE_COUNT),
            ("main_row_ids_root", partition_manifest.main_row_ids_root),
            (
                "semantic_conflict_row_ids_root",
                partition_manifest.semantic_conflict_row_ids_root,
            ),
            (
                "partition_union_row_ids_root",
                partition_manifest.partition_union_row_ids_root,
            ),
            (
                "ordered_archive_input_row_ids_root",
                partition_manifest.ordered_archive_input_row_ids_root,
            ),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("structural_completeness_verified", True),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        for name in _FALSE_EVALUATION_CLAIMS_V2:
            object.__setattr__(value, name, False)
        object.__setattr__(value, "metric_results", ())
        object.__setattr__(value, "scored_rows", ())
        value._validate(
            prediction_archive=prediction_archive,
            partition_manifest=partition_manifest,
        )
        return value

    def _validate(
        self,
        *,
        prediction_archive: DecodedRecognizerPredictionArchiveV2,
        partition_manifest: UnsealedPredictionPartitionManifestV2,
    ) -> None:
        if type(self) is not UnsealedPredictionStructuralEvaluationV2:
            raise TypeError("V2 unsealed evaluation exact type drift")
        if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2:
            raise TypeError("V2 unsealed evaluation archive exact type drift")
        if type(partition_manifest) is not UnsealedPredictionPartitionManifestV2:
            raise TypeError("V2 unsealed evaluation manifest exact type drift")
        if (
            type(self.disposition) is not UnsealedPredictionEvaluationDispositionV2
            or self.disposition
            is not UnsealedPredictionEvaluationDispositionV2.STRUCTURALLY_COMPLETE_NOT_SCORED
            or type(self.reason) is not str
            or self.reason != _SUCCESS_REASON_V2
        ):
            raise ValueError("V2 unsealed evaluation disposition or reason drift")
        _ascii_v2(self.reason, "V2 unsealed evaluation reason")
        for value, prefix, name in (
            (
                self.prediction_archive_id,
                "phase2b_recognizer_prediction_archive_v2_",
                "V2 unsealed evaluation archive ID",
            ),
            (
                self.prediction_archive_policy_id,
                "phase2b_recognizer_prediction_archive_policy_v2_",
                "V2 unsealed evaluation archive policy ID",
            ),
            (
                self.partition_manifest_id,
                "phase2b_unsealed_prediction_partition_v2_",
                "V2 unsealed evaluation manifest ID",
            ),
            (
                self.exact_freeze_id,
                "phase2b_exact_freeze_",
                "V2 unsealed evaluation exact freeze ID",
            ),
            (
                self.evaluator_policy_id,
                "phase2b_unsealed_prediction_evaluator_policy_v2_",
                "V2 unsealed evaluation policy ID",
            ),
            (
                self.main_row_ids_root,
                "phase2b_unsealed_main_rows_v2_",
                "V2 unsealed evaluation main root",
            ),
            (
                self.semantic_conflict_row_ids_root,
                "phase2b_unsealed_semantic_conflict_rows_v2_",
                "V2 unsealed evaluation semantic-conflict root",
            ),
            (
                self.partition_union_row_ids_root,
                "phase2b_unsealed_partition_union_rows_v2_",
                "V2 unsealed evaluation partition-union root",
            ),
            (
                self.ordered_archive_input_row_ids_root,
                "phase2b_prediction_input_rows_v2_",
                "V2 unsealed evaluation ordered archive root",
            ),
        ):
            _digest_v2(value, prefix, name)
        if (
            type(self.prediction_archive_schema_version) is not str
            or self.prediction_archive_schema_version
            != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
            or type(self.main_count) is not int
            or self.main_count != _MAIN_COUNT_V2
            or type(self.semantic_conflict_count) is not int
            or self.semantic_conflict_count != _SEMANTIC_CONFLICT_COUNT_V2
            or type(self.total_count) is not int
            or self.total_count != TOTAL_RECOGNIZER_CASE_COUNT
            or type(self.structural_completeness_verified) is not bool
            or not self.structural_completeness_verified
            or any(
                type(object.__getattribute__(self, name)) is not bool
                or object.__getattribute__(self, name)
                for name in _FALSE_EVALUATION_CLAIMS_V2
            )
            or type(self.metric_results) is not tuple
            or self.metric_results != ()
            or type(self.scored_rows) is not tuple
            or self.scored_rows != ()
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("V2 unsealed evaluation shallow claim or count drift")
        partition_manifest._validate()
        if (
            self.prediction_archive_id != partition_manifest.prediction_archive_id
            or self.prediction_archive_schema_version
            != partition_manifest.prediction_archive_schema_version
            or self.prediction_archive_policy_id
            != partition_manifest.prediction_archive_policy_id
            or self.partition_manifest_id != partition_manifest.manifest_id
            or self.exact_freeze_id != _EXACT_FREEZE_ID_V2
            or self.exact_freeze_id != partition_manifest.exact_freeze_id
            or self.evaluator_policy_id
            != UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2
            or self.evaluator_policy_id != partition_manifest.evaluator_policy_id
            or self.main_row_ids_root != partition_manifest.main_row_ids_root
            or self.semantic_conflict_row_ids_root
            != partition_manifest.semantic_conflict_row_ids_root
            or self.partition_union_row_ids_root
            != partition_manifest.partition_union_row_ids_root
            or self.ordered_archive_input_row_ids_root
            != partition_manifest.ordered_archive_input_row_ids_root
        ):
            raise ValueError("V2 unsealed evaluation identity drift")
        canonical_archive = _canonical_archive_replay_v2(prediction_archive)
        if self.prediction_archive_id != canonical_archive.archive_id:
            raise ValueError("V2 unsealed evaluation canonical archive ID drift")
        supplied_union = tuple(
            sorted(
                (
                    *partition_manifest.main_row_ids,
                    *partition_manifest.semantic_conflict_row_ids,
                )
            )
        )
        archive_row_ids = canonical_archive.input_row_ids
        ordered_root = _ordered_archive_input_row_ids_root_v2(archive_row_ids)
        if (
            frozenset(supplied_union) != frozenset(archive_row_ids)
            or len(supplied_union) != len(archive_row_ids)
            or self.partition_union_row_ids_root
            != _partition_union_row_ids_root_v2(supplied_union)
            or self.ordered_archive_input_row_ids_root != ordered_root
            or self.ordered_archive_input_row_ids_root
            != partition_manifest.ordered_archive_input_row_ids_root
            or self.ordered_archive_input_row_ids_root
            != canonical_archive.context.input_row_ids_root
        ):
            raise ValueError("V2 unsealed evaluation partition or ordered-root drift")


@dataclass(frozen=True, slots=True)
class UnsealedPredictionEvaluationRejectionV2:
    disposition: UnsealedPredictionEvaluationDispositionV2
    reason: str
    prediction_archive_id: str | None
    partition_manifest_id: str | None
    metric_results: tuple[()] = ()
    scored_rows: tuple[()] = ()
    structural_completeness_verified: bool = False
    scoring_performed: bool = False
    runtime_executed: bool = False
    actual_960_case_run_verified: bool = False
    recognizer_capacity_evidence: bool = False
    effect_evidence: bool = False
    c1_exit_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not UnsealedPredictionEvaluationRejectionV2:
            raise TypeError("V2 unsealed rejection exact type drift")
        if self.disposition is not UnsealedPredictionEvaluationDispositionV2.ABSTAIN:
            raise ValueError("V2 unsealed rejection must abstain")
        _ascii_v2(self.reason, "V2 unsealed rejection reason")
        if self.prediction_archive_id is not None:
            _digest_v2(
                self.prediction_archive_id,
                "phase2b_recognizer_prediction_archive_v2_",
                "V2 unsealed rejection archive ID",
            )
        if self.partition_manifest_id is not None:
            _digest_v2(
                self.partition_manifest_id,
                "phase2b_unsealed_prediction_partition_v2_",
                "V2 unsealed rejection partition ID",
            )
        if (
            type(self.metric_results) is not tuple
            or self.metric_results != ()
            or type(self.scored_rows) is not tuple
            or self.scored_rows != ()
            or any(
                type(item) is not bool or item
                for item in (
                    self.structural_completeness_verified,
                    self.scoring_performed,
                    self.runtime_executed,
                    self.actual_960_case_run_verified,
                    self.recognizer_capacity_evidence,
                    self.effect_evidence,
                    self.c1_exit_evidence,
                )
            )
        ):
            raise ValueError("V2 unsealed rejection leaked partial evaluation")


def _preflight_supplied_prediction_archive_v2(
    prediction_archive: DecodedRecognizerPredictionArchiveV2,
) -> tuple[
    bytes,
    str,
    PublicPredictionRunContextV2,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Close a supplied wrapper locally without using archive private replay."""

    if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2:
        raise TypeError("V2 unsealed evaluator requires exact prediction archive")
    try:
        archive = object.__getattribute__(prediction_archive, "archive")
        disposition = object.__getattribute__(prediction_archive, "disposition")
        archive_id = object.__getattribute__(prediction_archive, "archive_id")
        schema_version = object.__getattribute__(prediction_archive, "schema_version")
        policy_id = object.__getattribute__(prediction_archive, "policy_id")
        claim_level = object.__getattribute__(prediction_archive, "claim_level")
        context = object.__getattribute__(prediction_archive, "context")
        records = object.__getattribute__(prediction_archive, "records")
        input_row_ids = object.__getattribute__(prediction_archive, "input_row_ids")
        prediction_record_ids = object.__getattribute__(
            prediction_archive,
            "prediction_record_ids",
        )
        prediction_content_ids = object.__getattribute__(
            prediction_archive,
            "prediction_content_ids",
        )
    except AttributeError as exc:
        raise ValueError("V2 supplied prediction archive slot is missing") from exc
    if type(archive) is not bytes or not (
        PREDICTION_ARCHIVE_HEADER_BYTES_V2
        <= len(archive)
        <= MAXIMUM_PREDICTION_ARCHIVE_BYTES_V2
    ):
        raise TypeError("V2 supplied prediction archive bytes or cap drift")
    if (
        type(disposition) is not PredictionArchiveDispositionV2
        or disposition is not PredictionArchiveDispositionV2.COMPLETE
        or type(schema_version) is not str
        or schema_version != RECOGNIZER_PREDICTION_ARCHIVE_V2_VERSION
        or type(policy_id) is not str
        or policy_id != RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID_V2
        or type(claim_level) is not str
        or claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
    ):
        raise ValueError("V2 supplied prediction archive identity drift")
    _digest_v2(
        archive_id,
        "phase2b_recognizer_prediction_archive_v2_",
        "V2 supplied prediction archive ID",
    )
    _digest_v2(
        policy_id,
        "phase2b_recognizer_prediction_archive_policy_v2_",
        "V2 supplied prediction archive policy ID",
    )
    claims = tuple(
        object.__getattribute__(prediction_archive, name)
        for name in (*_ARCHIVE_TRUE_CLAIMS_V2, *_ARCHIVE_FALSE_CLAIMS_V2)
    )
    if (
        any(type(item) is not bool for item in claims)
        or not all(
            object.__getattribute__(prediction_archive, name)
            for name in _ARCHIVE_TRUE_CLAIMS_V2
        )
        or any(
            object.__getattribute__(prediction_archive, name)
            for name in _ARCHIVE_FALSE_CLAIMS_V2
        )
    ):
        raise ValueError("V2 supplied prediction archive claim boundary drift")
    if type(context) is not PublicPredictionRunContextV2:
        raise TypeError("V2 supplied prediction context exact type drift")
    try:
        context_id = object.__getattribute__(context, "context_id")
        context_row_root = object.__getattribute__(context, "input_row_ids_root")
        context_schema = object.__getattribute__(context, "schema_version")
        context_claim = object.__getattribute__(context, "claim_level")
        expected_count = object.__getattribute__(context, "expected_prediction_count")
        context_input_archive_id = object.__getattribute__(context, "input_archive_id")
        context_input_archive_sha = object.__getattribute__(
            context,
            "input_archive_sha256",
        )
        context_execution_freeze_id = object.__getattribute__(
            context,
            "execution_freeze_manifest_id",
        )
        context_batch_id = object.__getattribute__(context, "batch_id")
        context_batch_policy_id = object.__getattribute__(context, "batch_policy_id")
        context_input_archive_policy_id = object.__getattribute__(
            context,
            "input_archive_policy_id",
        )
        context_input_archive_version = object.__getattribute__(
            context,
            "input_archive_version",
        )
        context_protocol_id = object.__getattribute__(context, "protocol_id")
    except AttributeError as exc:
        raise ValueError("V2 supplied prediction context slot is missing") from exc
    if (
        type(context_schema) is not str
        or context_schema != PUBLIC_PREDICTION_RUN_CONTEXT_V2_SCHEMA_VERSION
        or type(context_claim) is not str
        or context_claim != NON_AUTHORITATIVE_CLAIM_LEVEL
        or type(expected_count) is not int
        or expected_count != TOTAL_RECOGNIZER_CASE_COUNT
        or type(context_batch_policy_id) is not str
        or context_batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
        or type(context_input_archive_policy_id) is not str
        or context_input_archive_policy_id != RECOGNIZER_INPUT_ARCHIVE_POLICY_ID_V2
        or type(context_input_archive_version) is not str
        or context_input_archive_version
        != TRUSTED_RECOGNIZER_INPUT_ARCHIVE_V2_VERSION
    ):
        raise ValueError("V2 supplied prediction context scalar drift")
    for value, prefix, name in (
        (
            context_id,
            "phase2b_public_prediction_run_context_v2_",
            "V2 supplied prediction context ID",
        ),
        (
            context_row_root,
            "phase2b_prediction_input_rows_v2_",
            "V2 supplied prediction context row root",
        ),
        (
            context_input_archive_id,
            "phase2b_recognizer_input_archive_v2_",
            "V2 supplied prediction input archive ID",
        ),
        (
            context_execution_freeze_id,
            "phase2b_execution_freeze_",
            "V2 supplied prediction execution freeze ID",
        ),
        (
            context_batch_id,
            "phase2b_trusted_wire_batch_v2_",
            "V2 supplied prediction batch ID",
        ),
    ):
        _digest_v2(value, prefix, name)
    if (
        type(context_input_archive_sha) is not str
        or len(context_input_archive_sha) != 64
        or any(item not in "0123456789abcdef" for item in context_input_archive_sha)
    ):
        raise ValueError("V2 supplied prediction input archive SHA drift")
    _digest_v2(
        context_protocol_id,
        "phase2b_protocol_",
        "V2 supplied prediction protocol ID",
    )
    columns = (
        (
            input_row_ids,
            "phase2b_recognizer_input_row_v2_",
            "V2 supplied prediction input row ID",
        ),
        (
            prediction_record_ids,
            "phase2b_recognizer_prediction_record_v2_",
            "V2 supplied prediction record ID",
        ),
        (
            prediction_content_ids,
            "phase2b_prediction_",
            "V2 supplied prediction content ID",
        ),
    )
    for column, prefix, name in columns:
        if type(column) is not tuple or len(column) != TOTAL_RECOGNIZER_CASE_COUNT:
            raise TypeError("V2 supplied prediction root column shape drift")
        for item in column:
            _digest_v2(item, prefix, name)
    if (
        len(set(input_row_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
        or len(set(prediction_record_ids)) != TOTAL_RECOGNIZER_CASE_COUNT
        or type(records) is not tuple
        or len(records) != TOTAL_RECOGNIZER_CASE_COUNT
        or any(type(record) is not PublicRecognizerPredictionRecordV2 for record in records)
    ):
        raise ValueError("V2 supplied prediction record or identity multiplicity drift")
    for index, record in enumerate(records):
        try:
            record_input_row_id = object.__getattribute__(record, "input_row_id")
            record_id = object.__getattribute__(record, "record_id")
            prediction_content_id = object.__getattribute__(
                record,
                "prediction_content_id",
            )
            run_context_id = object.__getattribute__(record, "run_context_id")
        except AttributeError as exc:
            raise ValueError("V2 supplied prediction record slot is missing") from exc
        _digest_v2(
            record_input_row_id,
            "phase2b_recognizer_input_row_v2_",
            "V2 supplied record input row ID",
        )
        _digest_v2(
            record_id,
            "phase2b_recognizer_prediction_record_v2_",
            "V2 supplied record ID",
        )
        _digest_v2(
            prediction_content_id,
            "phase2b_prediction_",
            "V2 supplied record prediction content ID",
        )
        _digest_v2(
            run_context_id,
            "phase2b_public_prediction_run_context_v2_",
            "V2 supplied record run context ID",
        )
        if (
            record_input_row_id != input_row_ids[index]
            or record_id != prediction_record_ids[index]
            or prediction_content_id != prediction_content_ids[index]
            or run_context_id != context_id
        ):
            raise ValueError("V2 supplied prediction stored column parity drift")
    if _ordered_archive_input_row_ids_root_v2(input_row_ids) != context_row_root:
        raise ValueError("V2 supplied prediction ordered row-root parity drift")
    return (
        archive,
        archive_id,
        context,
        input_row_ids,
        prediction_record_ids,
        prediction_content_ids,
    )


def _canonical_archive_replay_v2(
    prediction_archive: DecodedRecognizerPredictionArchiveV2,
) -> DecodedRecognizerPredictionArchiveV2:
    (
        archive,
        archive_id,
        supplied_context,
        input_row_ids,
        prediction_record_ids,
        prediction_content_ids,
    ) = _preflight_supplied_prediction_archive_v2(prediction_archive)
    canonical = decode_public_recognizer_prediction_archive_v2(archive)
    if type(canonical) is not DecodedRecognizerPredictionArchiveV2:
        raise TypeError("V2 public prediction decoder returned nonexact archive")
    canonical_context = canonical.context
    if type(canonical_context) is not PublicPredictionRunContextV2:
        raise TypeError("V2 canonical prediction context exact type drift")
    for item in fields(PublicPredictionRunContextV2):
        supplied = object.__getattribute__(supplied_context, item.name)
        replayed = object.__getattribute__(canonical_context, item.name)
        if type(supplied) is not type(replayed) or supplied != replayed:
            raise ValueError("V2 supplied prediction context differs from replay")
    if (
        canonical.archive is not archive
        and canonical.archive != archive
        or canonical.archive_id != archive_id
        or canonical.input_row_ids != input_row_ids
        or canonical.prediction_record_ids != prediction_record_ids
        or canonical.prediction_content_ids != prediction_content_ids
    ):
        raise ValueError("V2 supplied prediction columns differ from public replay")
    return canonical


def build_unsealed_prediction_partition_manifest_v2(
    *,
    prediction_archive: DecodedRecognizerPredictionArchiveV2,
    main_row_ids: tuple[str, ...],
    semantic_conflict_row_ids: tuple[str, ...],
) -> UnsealedPredictionPartitionManifestV2:
    if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2:
        raise TypeError("V2 partition manifest requires exact prediction archive")
    supplied_union = _preflight_partitions_v2(
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
    )
    canonical_archive = _canonical_archive_replay_v2(prediction_archive)
    archive_row_ids = canonical_archive.input_row_ids
    if (
        len(supplied_union) != len(archive_row_ids)
        or frozenset(supplied_union) != frozenset(archive_row_ids)
    ):
        raise ValueError(
            "V2 partition manifest is not exhaustive for prediction archive"
        )
    manifest = UnsealedPredictionPartitionManifestV2._issue(
        _MANIFEST_ISSUE_TOKEN_V2,
        prediction_archive=canonical_archive,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
    )
    if (
        manifest.partition_union_row_ids_root
        != _partition_union_row_ids_root_v2(supplied_union)
        or manifest.ordered_archive_input_row_ids_root
        != _ordered_archive_input_row_ids_root_v2(archive_row_ids)
        or manifest.ordered_archive_input_row_ids_root
        != canonical_archive.context.input_row_ids_root
    ):
        raise ValueError(
            "V2 partition manifest is not exhaustive for prediction archive"
        )
    return manifest


def _reject_v2(
    reason: str,
    *,
    prediction_archive_id: str | None,
    partition_manifest_id: str | None,
) -> UnsealedPredictionEvaluationRejectionV2:
    return UnsealedPredictionEvaluationRejectionV2(
        disposition=UnsealedPredictionEvaluationDispositionV2.ABSTAIN,
        reason=reason,
        prediction_archive_id=prediction_archive_id,
        partition_manifest_id=partition_manifest_id,
    )


def evaluate_unsealed_prediction_archive_structure_v2(
    *,
    prediction_archive: DecodedRecognizerPredictionArchiveV2,
    partition_manifest: UnsealedPredictionPartitionManifestV2,
) -> UnsealedPredictionStructuralEvaluationV2 | UnsealedPredictionEvaluationRejectionV2:
    if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV2:
        raise TypeError("V2 unsealed evaluator requires exact prediction archive")
    if type(partition_manifest) is not UnsealedPredictionPartitionManifestV2:
        raise TypeError("V2 unsealed evaluator requires exact partition manifest")
    try:
        raw_archive_id = object.__getattribute__(prediction_archive, "archive_id")
        raw_manifest_id = object.__getattribute__(partition_manifest, "manifest_id")
        main_row_ids = object.__getattribute__(partition_manifest, "main_row_ids")
        semantic_conflict_row_ids = object.__getattribute__(
            partition_manifest,
            "semantic_conflict_row_ids",
        )
    except AttributeError:
        return _reject_v2(
            "partition_or_archive_shallow_invalid",
            prediction_archive_id=None,
            partition_manifest_id=None,
        )
    archive_id = _safe_digest_or_none_v2(
        raw_archive_id,
        "phase2b_recognizer_prediction_archive_v2_",
        "V2 unsealed evaluator archive ID",
    )
    manifest_id = _safe_digest_or_none_v2(
        raw_manifest_id,
        "phase2b_unsealed_prediction_partition_v2_",
        "V2 unsealed evaluator partition ID",
    )
    if (
        type(main_row_ids) is not tuple
        or len(main_row_ids) != _MAIN_COUNT_V2
        or type(semantic_conflict_row_ids) is not tuple
        or len(semantic_conflict_row_ids) != _SEMANTIC_CONFLICT_COUNT_V2
    ):
        return _reject_v2(
            "partition_count_drift",
            prediction_archive_id=archive_id,
            partition_manifest_id=manifest_id,
        )
    try:
        return UnsealedPredictionStructuralEvaluationV2._issue(
            _EVALUATION_ISSUE_TOKEN_V2,
            prediction_archive=prediction_archive,
            partition_manifest=partition_manifest,
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _reject_v2(
            "partition_or_archive_structural_drift",
            prediction_archive_id=archive_id,
            partition_manifest_id=manifest_id,
        )


def _assert_field_manifests_v2() -> None:
    actual = (
        tuple(item.name for item in fields(UnsealedPredictionPartitionManifestV2)),
        tuple(item.name for item in fields(UnsealedPredictionStructuralEvaluationV2)),
        tuple(item.name for item in fields(UnsealedPredictionEvaluationRejectionV2)),
    )
    expected = (
        _MANIFEST_FIELDS_V2,
        _EVALUATION_FIELDS_V2,
        _REJECTION_FIELDS_V2,
    )
    if actual != expected:
        raise RuntimeError("V2 unsealed evaluator field manifest drift")


_assert_field_manifests_v2()


__all__ = (
    "UNSEALED_PREDICTION_EVALUATOR_POLICY_ID_V2",
    "UNSEALED_PREDICTION_EVALUATOR_V2_VERSION",
    "UnsealedPredictionEvaluationDispositionV2",
    "UnsealedPredictionEvaluationRejectionV2",
    "UnsealedPredictionPartitionManifestV2",
    "UnsealedPredictionStructuralEvaluationV2",
    "build_unsealed_prediction_partition_manifest_v2",
    "evaluate_unsealed_prediction_archive_structure_v2",
)

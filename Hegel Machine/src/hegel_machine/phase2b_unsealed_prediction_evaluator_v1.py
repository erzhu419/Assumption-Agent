"""Unsealed structural completeness evaluator for Phase-2B predictions.

This evaluator is deliberately outside the recognizer-facing wire contract.
It checks only that an evaluator-side 720/240 manifest is sorted, disjoint,
exhaustive, and rooted in the same 960 public input-row IDs as one decoded
prediction archive.  It has no scorer, answer key, metric computation, sealed
authority, or effect claim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import hashlib
from typing import Final

from .phase2b_freeze_v1 import frozen_phase2b_exact_freeze
from .phase2b_recognizer_prediction_archive_v1 import (
    RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID,
    DecodedRecognizerPredictionArchiveV1,
    _input_row_ids_root,
)
from .phase2b_runner import TOTAL_RECOGNIZER_CASE_COUNT
from .phase2b_trusted_wire_v1 import (
    MAXIMUM_ASCII_STRING_BYTES,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
)


UNSEALED_PREDICTION_EVALUATOR_VERSION: Final = (
    "hegel-machine-phase2b-unsealed-prediction-evaluator/1"
)
_EXACT_FREEZE: Final = frozen_phase2b_exact_freeze()
_EXACT_FREEZE_ID: Final = _EXACT_FREEZE.freeze_id
_MAIN_COUNT: Final = _EXACT_FREEZE.holdout.independent_latent_case_count
_SEMANTIC_CONFLICT_COUNT: Final = _EXACT_FREEZE.semantic_conflict.case_count
if (
    type(_EXACT_FREEZE_ID) is not str
    or not _EXACT_FREEZE_ID.startswith("phase2b_exact_freeze_")
    or type(_MAIN_COUNT) is not int
    or _MAIN_COUNT != 720
    or type(_SEMANTIC_CONFLICT_COUNT) is not int
    or _SEMANTIC_CONFLICT_COUNT != 240
    or _MAIN_COUNT + _SEMANTIC_CONFLICT_COUNT != TOTAL_RECOGNIZER_CASE_COUNT
):
    raise RuntimeError("unsealed evaluator exact count freeze drift")
if (
    type(_EXACT_FREEZE.semantic_conflict.included_in_main_accuracy_denominator)
    is not bool
    or _EXACT_FREEZE.semantic_conflict.included_in_main_accuracy_denominator
    or type(_EXACT_FREEZE.semantic_conflict.threshold_tuning_allowed) is not bool
    or _EXACT_FREEZE.semantic_conflict.threshold_tuning_allowed
    or type(_EXACT_FREEZE.semantic_conflict.same_freeze_and_reveal_as_main)
    is not bool
    or not _EXACT_FREEZE.semantic_conflict.same_freeze_and_reveal_as_main
):
    raise RuntimeError("unsealed evaluator semantic-conflict freeze drift")

_MAIN_ROWS_DOMAIN: Final = b"HEGEL/PHASE2B/UNSEALED/MAIN_ROWS/V1\x00"
_SEMANTIC_CONFLICT_ROWS_DOMAIN: Final = (
    b"HEGEL/PHASE2B/UNSEALED/SEMANTIC_CONFLICT_ROWS/V1\x00"
)
_MANIFEST_DOMAIN: Final = b"HEGEL/PHASE2B/UNSEALED/PARTITION_MANIFEST/V1\x00"
_MANIFEST_ISSUE_TOKEN: Final = object()
_EVALUATION_ISSUE_TOKEN: Final = object()

_MANIFEST_FIELDS: Final = (
    "prediction_archive_id",
    "exact_freeze_id",
    "evaluator_policy_id",
    "main_row_ids",
    "semantic_conflict_row_ids",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "all_input_row_ids_root",
    "manifest_id",
)
_EVALUATION_FIELDS: Final = (
    "disposition",
    "reason",
    "prediction_archive_id",
    "partition_manifest_id",
    "exact_freeze_id",
    "evaluator_policy_id",
    "main_count",
    "semantic_conflict_count",
    "total_count",
    "main_row_ids_root",
    "semantic_conflict_row_ids_root",
    "all_input_row_ids_root",
    "claim_level",
    "structural_completeness_verified",
    "challenge_in_main_denominator",
    "scoring_performed",
    "runtime_executed",
    "recognizer_capacity_evidence",
    "metric_results",
    "scored_rows",
    "origin_authenticated",
    "sealed_holdout_eligible",
    "formal_covert_audit",
    "effect_evidence",
    "c1_exit_evidence",
)
_REJECTION_FIELDS: Final = (
    "disposition",
    "reason",
    "prediction_archive_id",
    "partition_manifest_id",
    "metric_results",
    "scored_rows",
    "structural_completeness_verified",
    "scoring_performed",
    "runtime_executed",
    "recognizer_capacity_evidence",
    "effect_evidence",
)


class UnsealedPredictionEvaluationDisposition(str, Enum):
    STRUCTURALLY_COMPLETE_NOT_SCORED = "STRUCTURALLY_COMPLETE_NOT_SCORED"
    ABSTAIN = "ABSTAIN"


_SUCCESS_REASON: Final = "sorted_disjoint_exhaustive_same_960_input_roots"
_SUCCESS_CLAIMS: Final = (
    ("structural_completeness_verified", True),
    ("challenge_in_main_denominator", False),
    ("scoring_performed", False),
    ("runtime_executed", False),
    ("recognizer_capacity_evidence", False),
    ("metric_results", ()),
    ("scored_rows", ()),
    ("origin_authenticated", False),
    ("sealed_holdout_eligible", False),
    ("formal_covert_audit", False),
    ("effect_evidence", False),
    ("c1_exit_evidence", False),
)
_REJECTION_CLAIMS: Final = (
    ("metric_results", ()),
    ("scored_rows", ()),
    ("structural_completeness_verified", False),
    ("scoring_performed", False),
    ("runtime_executed", False),
    ("recognizer_capacity_evidence", False),
    ("effect_evidence", False),
)


UNSEALED_PREDICTION_EVALUATOR_POLICY_ID: Final = hashlib.sha256(
    (
        "phase2b_unsealed_prediction_evaluator_policy_v1|"
        + UNSEALED_PREDICTION_EVALUATOR_VERSION
        + "|prediction_policy="
        + RECOGNIZER_PREDICTION_ARCHIVE_POLICY_ID
        + "|exact_freeze_id="
        + _EXACT_FREEZE_ID
        + "|semantic_conflict_contract="
        + repr(
            (
                (
                    "included_in_main_accuracy_denominator",
                    _EXACT_FREEZE.semantic_conflict.included_in_main_accuracy_denominator,
                ),
                (
                    "threshold_tuning_allowed",
                    _EXACT_FREEZE.semantic_conflict.threshold_tuning_allowed,
                ),
                (
                    "same_freeze_and_reveal_as_main",
                    _EXACT_FREEZE.semantic_conflict.same_freeze_and_reveal_as_main,
                ),
            )
        )
        + "|counts="
        + str((_MAIN_COUNT, _SEMANTIC_CONFLICT_COUNT, TOTAL_RECOGNIZER_CASE_COUNT))
        + "|domains="
        + str(
            (
                _MAIN_ROWS_DOMAIN.hex(),
                _SEMANTIC_CONFLICT_ROWS_DOMAIN.hex(),
                _MANIFEST_DOMAIN.hex(),
            )
        )
        + "|manifest_fields="
        + repr(_MANIFEST_FIELDS)
        + "|evaluation_fields="
        + repr(_EVALUATION_FIELDS)
        + "|rejection_fields="
        + repr(_REJECTION_FIELDS)
        + "|manifest_root_formula=sha256(manifest_domain||length_framed_exact_"
        + "prediction_archive_id||exact_freeze_id||evaluator_policy_id||main_root||"
        + "semantic_conflict_root||all_input_root||u32_counts);validate_before_hash"
        + "|success_disposition="
        + UnsealedPredictionEvaluationDisposition.STRUCTURALLY_COMPLETE_NOT_SCORED.value
        + "|success_reason="
        + _SUCCESS_REASON
        + "|success_claims="
        + repr(_SUCCESS_CLAIMS)
        + "|rejection_claims="
        + repr(_REJECTION_CLAIMS)
        + "|contract=sorted_disjoint_exhaustive_same_960_root_no_score_callback_no_metrics"
        + "|claims=structural_only_runtime_capacity_scoring_origin_sealed_formal_effect_c1_false"
    ).encode("ascii")
).hexdigest()
UNSEALED_PREDICTION_EVALUATOR_POLICY_ID = (
    "phase2b_unsealed_prediction_evaluator_policy_"
    + UNSEALED_PREDICTION_EVALUATOR_POLICY_ID
)


def _ascii(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > MAXIMUM_ASCII_STRING_BYTES
        or not value.isascii()
    ):
        raise ValueError(f"{name} must use exact bounded nonempty ASCII")
    return value


def _digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError(f"{name} prefix drift")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError(f"{name} must end in exact lowercase SHA-256")
    return value


def _safe_digest_or_none(value: object, prefix: str, name: str) -> str | None:
    try:
        return _digest(value, prefix, name)
    except (TypeError, ValueError):
        return None


def _partition_root(
    values: tuple[str, ...],
    *,
    expected_count: int,
    domain: bytes,
    prefix: str,
) -> str:
    if type(values) is not tuple or len(values) != expected_count:
        raise ValueError("unsealed partition count drift")
    for value in values:
        _digest(value, "phase2b_recognizer_input_row_", "partition row ID")
    if values != tuple(sorted(values)) or len(set(values)) != expected_count:
        raise ValueError("unsealed partition rows are not sorted unique")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(expected_count.to_bytes(4, "big"))
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    return prefix + digest.hexdigest()


def _manifest_id(
    *,
    prediction_archive_id: str,
    exact_freeze_id: str,
    evaluator_policy_id: str,
    main_row_ids_root: str,
    semantic_conflict_row_ids_root: str,
    all_input_row_ids_root: str,
) -> str:
    _digest(
        prediction_archive_id,
        "phase2b_recognizer_prediction_archive_",
        "unsealed manifest prediction archive ID",
    )
    _digest(exact_freeze_id, "phase2b_exact_freeze_", "exact freeze ID")
    _digest(
        evaluator_policy_id,
        "phase2b_unsealed_prediction_evaluator_policy_",
        "unsealed evaluator policy ID",
    )
    _digest(main_row_ids_root, "phase2b_unsealed_main_rows_", "main row root")
    _digest(
        semantic_conflict_row_ids_root,
        "phase2b_unsealed_semantic_conflict_rows_",
        "semantic-conflict row root",
    )
    _digest(
        all_input_row_ids_root,
        "phase2b_prediction_input_rows_",
        "all input-row root",
    )
    digest = hashlib.sha256()
    digest.update(_MANIFEST_DOMAIN)
    for value in (
        prediction_archive_id,
        exact_freeze_id,
        evaluator_policy_id,
        main_row_ids_root,
        semantic_conflict_row_ids_root,
        all_input_row_ids_root,
    ):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
    digest.update(_MAIN_COUNT.to_bytes(4, "big"))
    digest.update(_SEMANTIC_CONFLICT_COUNT.to_bytes(4, "big"))
    digest.update(TOTAL_RECOGNIZER_CASE_COUNT.to_bytes(4, "big"))
    return "phase2b_unsealed_prediction_partition_" + digest.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class UnsealedPredictionPartitionManifestV1:
    prediction_archive_id: str
    exact_freeze_id: str
    evaluator_policy_id: str
    main_row_ids: tuple[str, ...]
    semantic_conflict_row_ids: tuple[str, ...]
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    all_input_row_ids_root: str
    manifest_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed partition manifests are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        prediction_archive_id: str,
        main_row_ids: tuple[str, ...],
        semantic_conflict_row_ids: tuple[str, ...],
    ) -> "UnsealedPredictionPartitionManifestV1":
        if token is not _MANIFEST_ISSUE_TOKEN:
            raise TypeError("unsealed partition manifest issuer token mismatch")
        if type(main_row_ids) is not tuple or type(semantic_conflict_row_ids) is not tuple:
            raise TypeError("unsealed partition rows must use exact tuples")
        main_root = _partition_root(
            main_row_ids,
            expected_count=_MAIN_COUNT,
            domain=_MAIN_ROWS_DOMAIN,
            prefix="phase2b_unsealed_main_rows_",
        )
        conflict_root = _partition_root(
            semantic_conflict_row_ids,
            expected_count=_SEMANTIC_CONFLICT_COUNT,
            domain=_SEMANTIC_CONFLICT_ROWS_DOMAIN,
            prefix="phase2b_unsealed_semantic_conflict_rows_",
        )
        if set(main_row_ids) & set(semantic_conflict_row_ids):
            raise ValueError("unsealed partitions overlap")
        all_rows = tuple(sorted((*main_row_ids, *semantic_conflict_row_ids)))
        all_root = _input_row_ids_root(all_rows)
        value = object.__new__(cls)
        for name, item in (
            ("prediction_archive_id", prediction_archive_id),
            ("exact_freeze_id", _EXACT_FREEZE_ID),
            (
                "evaluator_policy_id",
                UNSEALED_PREDICTION_EVALUATOR_POLICY_ID,
            ),
            ("main_row_ids", main_row_ids),
            ("semantic_conflict_row_ids", semantic_conflict_row_ids),
            ("main_row_ids_root", main_root),
            ("semantic_conflict_row_ids_root", conflict_root),
            ("all_input_row_ids_root", all_root),
            (
                "manifest_id",
                _manifest_id(
                    prediction_archive_id=prediction_archive_id,
                    exact_freeze_id=_EXACT_FREEZE_ID,
                    evaluator_policy_id=UNSEALED_PREDICTION_EVALUATOR_POLICY_ID,
                    main_row_ids_root=main_root,
                    semantic_conflict_row_ids_root=conflict_root,
                    all_input_row_ids_root=all_root,
                ),
            ),
        ):
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not UnsealedPredictionPartitionManifestV1:
            raise TypeError("unsealed partition manifest must use exact type")
        _digest(
            self.prediction_archive_id,
            "phase2b_recognizer_prediction_archive_",
            "unsealed manifest prediction archive ID",
        )
        _digest(self.exact_freeze_id, "phase2b_exact_freeze_", "exact freeze ID")
        _digest(
            self.evaluator_policy_id,
            "phase2b_unsealed_prediction_evaluator_policy_",
            "unsealed evaluator policy ID",
        )
        if (
            self.exact_freeze_id != _EXACT_FREEZE_ID
            or self.evaluator_policy_id
            != UNSEALED_PREDICTION_EVALUATOR_POLICY_ID
        ):
            raise ValueError("unsealed partition freeze or policy drift")
        if type(self.main_row_ids) is not tuple or type(
            self.semantic_conflict_row_ids
        ) is not tuple:
            raise TypeError("unsealed manifest row arrays must use exact tuples")
        expected_main_root = _partition_root(
            self.main_row_ids,
            expected_count=_MAIN_COUNT,
            domain=_MAIN_ROWS_DOMAIN,
            prefix="phase2b_unsealed_main_rows_",
        )
        expected_conflict_root = _partition_root(
            self.semantic_conflict_row_ids,
            expected_count=_SEMANTIC_CONFLICT_COUNT,
            domain=_SEMANTIC_CONFLICT_ROWS_DOMAIN,
            prefix="phase2b_unsealed_semantic_conflict_rows_",
        )
        if set(self.main_row_ids) & set(self.semantic_conflict_row_ids):
            raise ValueError("unsealed manifest partitions overlap")
        all_rows = tuple(sorted((*self.main_row_ids, *self.semantic_conflict_row_ids)))
        expected_all_root = _input_row_ids_root(all_rows)
        _digest(self.main_row_ids_root, "phase2b_unsealed_main_rows_", "main row root")
        _digest(
            self.semantic_conflict_row_ids_root,
            "phase2b_unsealed_semantic_conflict_rows_",
            "semantic-conflict row root",
        )
        _digest(
            self.all_input_row_ids_root,
            "phase2b_prediction_input_rows_",
            "all input-row root",
        )
        _digest(
            self.manifest_id,
            "phase2b_unsealed_prediction_partition_",
            "unsealed partition manifest ID",
        )
        if (
            self.main_row_ids_root != expected_main_root
            or self.semantic_conflict_row_ids_root != expected_conflict_root
            or self.all_input_row_ids_root != expected_all_root
            or self.manifest_id
            != _manifest_id(
                prediction_archive_id=self.prediction_archive_id,
                exact_freeze_id=self.exact_freeze_id,
                evaluator_policy_id=self.evaluator_policy_id,
                main_row_ids_root=expected_main_root,
                semantic_conflict_row_ids_root=expected_conflict_root,
                all_input_row_ids_root=expected_all_root,
            )
        ):
            raise ValueError("unsealed partition manifest root drift")
@dataclass(frozen=True, slots=True, init=False)
class UnsealedPredictionStructuralEvaluationV1:
    disposition: UnsealedPredictionEvaluationDisposition
    reason: str
    prediction_archive_id: str
    partition_manifest_id: str
    exact_freeze_id: str
    evaluator_policy_id: str
    main_count: int
    semantic_conflict_count: int
    total_count: int
    main_row_ids_root: str
    semantic_conflict_row_ids_root: str
    all_input_row_ids_root: str
    claim_level: str
    structural_completeness_verified: bool
    challenge_in_main_denominator: bool
    scoring_performed: bool
    runtime_executed: bool
    recognizer_capacity_evidence: bool
    metric_results: tuple[()]
    scored_rows: tuple[()]
    origin_authenticated: bool
    sealed_holdout_eligible: bool
    formal_covert_audit: bool
    effect_evidence: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("unsealed structural evaluations are privately issued")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        prediction_archive: DecodedRecognizerPredictionArchiveV1,
        partition_manifest: UnsealedPredictionPartitionManifestV1,
    ) -> "UnsealedPredictionStructuralEvaluationV1":
        if token is not _EVALUATION_ISSUE_TOKEN:
            raise TypeError("unsealed evaluation issuer token mismatch")
        value = object.__new__(cls)
        for name, item in (
            (
                "disposition",
                UnsealedPredictionEvaluationDisposition.STRUCTURALLY_COMPLETE_NOT_SCORED,
            ),
            ("reason", _SUCCESS_REASON),
            ("prediction_archive_id", prediction_archive.archive_id),
            ("partition_manifest_id", partition_manifest.manifest_id),
            ("exact_freeze_id", _EXACT_FREEZE_ID),
            (
                "evaluator_policy_id",
                UNSEALED_PREDICTION_EVALUATOR_POLICY_ID,
            ),
            ("main_count", _MAIN_COUNT),
            ("semantic_conflict_count", _SEMANTIC_CONFLICT_COUNT),
            ("total_count", TOTAL_RECOGNIZER_CASE_COUNT),
            ("main_row_ids_root", partition_manifest.main_row_ids_root),
            (
                "semantic_conflict_row_ids_root",
                partition_manifest.semantic_conflict_row_ids_root,
            ),
            ("all_input_row_ids_root", partition_manifest.all_input_row_ids_root),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("structural_completeness_verified", True),
            ("challenge_in_main_denominator", False),
            ("scoring_performed", False),
            ("runtime_executed", False),
            ("recognizer_capacity_evidence", False),
            ("metric_results", ()),
            ("scored_rows", ()),
            ("origin_authenticated", False),
            ("sealed_holdout_eligible", False),
            ("formal_covert_audit", False),
            ("effect_evidence", False),
            ("c1_exit_evidence", False),
        ):
            object.__setattr__(value, name, item)
        value._validate(
            prediction_archive=prediction_archive,
            partition_manifest=partition_manifest,
        )
        return value

    def _validate(
        self,
        *,
        prediction_archive: DecodedRecognizerPredictionArchiveV1,
        partition_manifest: UnsealedPredictionPartitionManifestV1,
    ) -> None:
        if type(self) is not UnsealedPredictionStructuralEvaluationV1:
            raise TypeError("unsealed evaluation must use exact type")
        if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV1:
            raise TypeError("unsealed evaluation needs exact prediction archive")
        if type(partition_manifest) is not UnsealedPredictionPartitionManifestV1:
            raise TypeError("unsealed evaluation needs exact partition manifest")
        prediction_archive._validate()
        partition_manifest._validate()
        if (
            type(self.disposition) is not UnsealedPredictionEvaluationDisposition
            or self.disposition
            is not UnsealedPredictionEvaluationDisposition.STRUCTURALLY_COMPLETE_NOT_SCORED
        ):
            raise ValueError("unsealed evaluation disposition drift")
        _ascii(self.reason, "unsealed evaluation reason")
        if self.reason != _SUCCESS_REASON:
            raise ValueError("unsealed evaluation success reason drift")
        _digest(
            self.prediction_archive_id,
            "phase2b_recognizer_prediction_archive_",
            "unsealed evaluation archive ID",
        )
        _digest(
            self.partition_manifest_id,
            "phase2b_unsealed_prediction_partition_",
            "unsealed evaluation partition ID",
        )
        _digest(self.exact_freeze_id, "phase2b_exact_freeze_", "exact freeze ID")
        _digest(
            self.evaluator_policy_id,
            "phase2b_unsealed_prediction_evaluator_policy_",
            "unsealed evaluator policy ID",
        )
        _digest(self.main_row_ids_root, "phase2b_unsealed_main_rows_", "main row root")
        _digest(
            self.semantic_conflict_row_ids_root,
            "phase2b_unsealed_semantic_conflict_rows_",
            "semantic-conflict row root",
        )
        _digest(
            self.all_input_row_ids_root,
            "phase2b_prediction_input_rows_",
            "all input-row root",
        )
        supplied_rows = tuple(
            sorted(
                (
                    *partition_manifest.main_row_ids,
                    *partition_manifest.semantic_conflict_row_ids,
                )
            )
        )
        if (
            self.partition_manifest_id != partition_manifest.manifest_id
            or self.prediction_archive_id != partition_manifest.prediction_archive_id
            or self.prediction_archive_id != prediction_archive.archive_id
            or self.exact_freeze_id != _EXACT_FREEZE_ID
            or self.exact_freeze_id != partition_manifest.exact_freeze_id
            or self.evaluator_policy_id
            != UNSEALED_PREDICTION_EVALUATOR_POLICY_ID
            or self.evaluator_policy_id != partition_manifest.evaluator_policy_id
            or self.main_row_ids_root != partition_manifest.main_row_ids_root
            or self.semantic_conflict_row_ids_root
            != partition_manifest.semantic_conflict_row_ids_root
            or self.all_input_row_ids_root != partition_manifest.all_input_row_ids_root
            or supplied_rows != prediction_archive.input_row_ids
            or self.all_input_row_ids_root
            != prediction_archive.context.input_row_ids_root
            or _input_row_ids_root(supplied_rows)
            != prediction_archive.context.input_row_ids_root
        ):
            raise ValueError("unsealed evaluation root drift")
        if (
            type(self.main_count) is not int
            or self.main_count != _MAIN_COUNT
            or type(self.semantic_conflict_count) is not int
            or self.semantic_conflict_count != _SEMANTIC_CONFLICT_COUNT
            or type(self.total_count) is not int
            or self.total_count != TOTAL_RECOGNIZER_CASE_COUNT
        ):
            raise ValueError("unsealed evaluation count drift")
        bools = (
            self.structural_completeness_verified,
            self.challenge_in_main_denominator,
            self.scoring_performed,
            self.runtime_executed,
            self.recognizer_capacity_evidence,
            self.origin_authenticated,
            self.sealed_holdout_eligible,
            self.formal_covert_audit,
            self.effect_evidence,
            self.c1_exit_evidence,
        )
        if any(type(item) is not bool for item in bools):
            raise TypeError("unsealed evaluation claims require exact bool")
        if (
            not self.structural_completeness_verified
            or any(bools[1:])
            or type(self.metric_results) is not tuple
            or self.metric_results != ()
            or type(self.scored_rows) is not tuple
            or self.scored_rows != ()
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("unsealed evaluation claim boundary drift")


@dataclass(frozen=True, slots=True)
class UnsealedPredictionEvaluationRejectionV1:
    disposition: UnsealedPredictionEvaluationDisposition
    reason: str
    prediction_archive_id: str | None
    partition_manifest_id: str | None
    metric_results: tuple[()] = ()
    scored_rows: tuple[()] = ()
    structural_completeness_verified: bool = False
    scoring_performed: bool = False
    runtime_executed: bool = False
    recognizer_capacity_evidence: bool = False
    effect_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not UnsealedPredictionEvaluationRejectionV1:
            raise TypeError("unsealed rejection must use exact type")
        if self.disposition is not UnsealedPredictionEvaluationDisposition.ABSTAIN:
            raise ValueError("unsealed rejection must abstain")
        _ascii(self.reason, "unsealed rejection reason")
        if self.prediction_archive_id is not None:
            _digest(
                self.prediction_archive_id,
                "phase2b_recognizer_prediction_archive_",
                "unsealed rejection archive ID",
            )
        if self.partition_manifest_id is not None:
            _digest(
                self.partition_manifest_id,
                "phase2b_unsealed_prediction_partition_",
                "unsealed rejection partition ID",
            )
        if (
            type(self.metric_results) is not tuple
            or self.metric_results != ()
            or type(self.scored_rows) is not tuple
            or self.scored_rows != ()
            or type(self.structural_completeness_verified) is not bool
            or self.structural_completeness_verified
            or type(self.scoring_performed) is not bool
            or self.scoring_performed
            or type(self.runtime_executed) is not bool
            or self.runtime_executed
            or type(self.recognizer_capacity_evidence) is not bool
            or self.recognizer_capacity_evidence
            or type(self.effect_evidence) is not bool
            or self.effect_evidence
        ):
            raise ValueError("unsealed rejection leaked partial evaluation")


def build_unsealed_prediction_partition_manifest_v1(
    *,
    prediction_archive: DecodedRecognizerPredictionArchiveV1,
    main_row_ids: tuple[str, ...],
    semantic_conflict_row_ids: tuple[str, ...],
) -> UnsealedPredictionPartitionManifestV1:
    if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV1:
        raise TypeError("partition manifest requires exact prediction archive")
    if type(main_row_ids) is not tuple or type(semantic_conflict_row_ids) is not tuple:
        raise TypeError("partition manifest row IDs must use exact tuples")
    if len(main_row_ids) != _MAIN_COUNT or len(
        semantic_conflict_row_ids
    ) != _SEMANTIC_CONFLICT_COUNT:
        raise ValueError("partition manifest requires exact 720/240 counts")
    prediction_archive._validate()
    manifest = UnsealedPredictionPartitionManifestV1._issue(
        _MANIFEST_ISSUE_TOKEN,
        prediction_archive_id=prediction_archive.archive_id,
        main_row_ids=main_row_ids,
        semantic_conflict_row_ids=semantic_conflict_row_ids,
    )
    expected_rows = tuple(prediction_archive.input_row_ids)
    supplied_rows = tuple(sorted((*main_row_ids, *semantic_conflict_row_ids)))
    if (
        supplied_rows != expected_rows
        or manifest.all_input_row_ids_root
        != prediction_archive.context.input_row_ids_root
    ):
        raise ValueError("partition manifest is not exhaustive for prediction archive")
    return manifest


def _reject(
    reason: str,
    *,
    prediction_archive_id: str | None,
    partition_manifest_id: str | None,
) -> UnsealedPredictionEvaluationRejectionV1:
    return UnsealedPredictionEvaluationRejectionV1(
        disposition=UnsealedPredictionEvaluationDisposition.ABSTAIN,
        reason=reason,
        prediction_archive_id=prediction_archive_id,
        partition_manifest_id=partition_manifest_id,
    )


def evaluate_unsealed_prediction_archive_structure_v1(
    *,
    prediction_archive: DecodedRecognizerPredictionArchiveV1,
    partition_manifest: UnsealedPredictionPartitionManifestV1,
) -> UnsealedPredictionStructuralEvaluationV1 | UnsealedPredictionEvaluationRejectionV1:
    if type(prediction_archive) is not DecodedRecognizerPredictionArchiveV1:
        raise TypeError("unsealed evaluator requires exact prediction archive")
    if type(partition_manifest) is not UnsealedPredictionPartitionManifestV1:
        raise TypeError("unsealed evaluator requires exact partition manifest")
    try:
        raw_archive_id = prediction_archive.archive_id
        raw_manifest_id = partition_manifest.manifest_id
        main_row_ids = partition_manifest.main_row_ids
        semantic_conflict_row_ids = partition_manifest.semantic_conflict_row_ids
    except AttributeError:
        return _reject(
            "partition_or_archive_shallow_invalid",
            prediction_archive_id=None,
            partition_manifest_id=None,
        )
    archive_id = _safe_digest_or_none(
        raw_archive_id,
        "phase2b_recognizer_prediction_archive_",
        "unsealed evaluator archive ID",
    )
    manifest_id = _safe_digest_or_none(
        raw_manifest_id,
        "phase2b_unsealed_prediction_partition_",
        "unsealed evaluator partition ID",
    )
    if (
        type(main_row_ids) is not tuple
        or len(main_row_ids) != _MAIN_COUNT
        or type(semantic_conflict_row_ids) is not tuple
        or len(semantic_conflict_row_ids) != _SEMANTIC_CONFLICT_COUNT
    ):
        return _reject(
            "partition_count_drift",
            prediction_archive_id=archive_id,
            partition_manifest_id=manifest_id,
        )
    try:
        return UnsealedPredictionStructuralEvaluationV1._issue(
            _EVALUATION_ISSUE_TOKEN,
            prediction_archive=prediction_archive,
            partition_manifest=partition_manifest,
        )
    except (AttributeError, KeyError, OverflowError, RuntimeError, TypeError, ValueError):
        return _reject(
            "partition_or_archive_structural_drift",
            prediction_archive_id=archive_id,
            partition_manifest_id=manifest_id,
        )


def _assert_field_manifests() -> None:
    for actual, expected, name in (
        (
            tuple(item.name for item in fields(UnsealedPredictionPartitionManifestV1)),
            _MANIFEST_FIELDS,
            "partition manifest",
        ),
        (
            tuple(item.name for item in fields(UnsealedPredictionStructuralEvaluationV1)),
            _EVALUATION_FIELDS,
            "structural evaluation",
        ),
        (
            tuple(item.name for item in fields(UnsealedPredictionEvaluationRejectionV1)),
            _REJECTION_FIELDS,
            "evaluation rejection",
        ),
    ):
        if actual != expected:
            raise RuntimeError(f"unsealed {name} field manifest drift")


_assert_field_manifests()


__all__ = (
    "UNSEALED_PREDICTION_EVALUATOR_POLICY_ID",
    "UNSEALED_PREDICTION_EVALUATOR_VERSION",
    "UnsealedPredictionEvaluationDisposition",
    "UnsealedPredictionEvaluationRejectionV1",
    "UnsealedPredictionPartitionManifestV1",
    "UnsealedPredictionStructuralEvaluationV1",
    "build_unsealed_prediction_partition_manifest_v1",
    "evaluate_unsealed_prediction_archive_structure_v1",
)
